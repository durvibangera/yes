import sys
import os
import json
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.config import settings
from retrieval_stubs.sparse_retriever import sparse_retrieve
from retrieval_stubs.metadata_retriever import metadata_retrieve
from retrieval_stubs.graph_retriever import GraphRetriever
from retrieval_stubs.dense_retriever import dense_retrieve
from FlagEmbedding import BGEM3FlagModel
import ollama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RunAllStrategies")

def _chunk_ids_from_results(results):
    return [r["chunk_id"] for r in results if "chunk_id" in r]

def _required_found(results, required):
    if not required:
        return False
    found = {r["chunk_id"] for r in results if "chunk_id" in r}
    return all(cid in found for cid in required)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    project_id = args.project_id
    top_k = args.top_k
    data_root = settings.DATA_ROOT

    # Read questions
    input_path = os.path.join(data_root, "outputs", f"{project_id}_questions.jsonl")
    if not os.path.exists(input_path):
        logger.error(f"Questions file not found: {input_path}")
        return

    questions = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line))

    # Skip null
    questions = [q for q in questions if q.get("category") != "null"]
    logger.info(f"Loaded {len(questions)} surviving questions for {project_id}")

    # Init models
    embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    graph_retriever = GraphRetriever()

    def dense_query_fn(query_text: str, pid: str, k: int, expr: str = None):
        embeddings = embedding_model.encode([query_text], batch_size=1, max_length=512)['dense_vecs']
        return dense_retrieve(query_embedding=embeddings[0].tolist(), project_id=pid, top_k=k, expr=expr)
        
    def llm_extract_entities(text: str):
        prompt = f"""You are an expert technical knowledge extraction system that works across multiple engineering and regulatory domains.

Your task has TWO steps:

STEP 1 — Identify the domain:
Read the question and silently identify what technical field or regulatory body it belongs to.
Examples: "maritime / ABS / high speed craft", "ASME pressure vessels", "ISO welding standards", "structural steel / EN specs", etc.

STEP 2 — Extract entities with domain-aware expansion:
Using your identified domain context, extract the key technical concepts from the question.
For each concept, generate a group containing the canonical name AND its meaningful synonyms/variants, specifically including:
  - Standard abbreviations for this domain (e.g. if domain is maritime: "HSC" = "high speed craft", "Hs" = "significant wave height")
  - Regulatory variants (e.g. "passenger craft" → "passenger vessel", "passenger ship")
  - Alternative phrasings used in the domain's standards documents
  - Related measurement or state terms if applicable

Output format — return ONLY a valid JSON object with exactly two keys:
{{
  "entities": [["canonical name", "synonym1", "synonym2"], ["entity2", "variant1"]],
  "relations": ["relationship phrase 1", "relationship phrase 2"]
}}

Rules:
- Do NOT output the domain name, STEP 1 reasoning, or any text outside the JSON.
- Each entity group must have at least the canonical name.
- "relations" should describe HOW the entities relate (e.g. "requires", "is part of", "is tested by"). Use an empty list if none.
- Do NOT wrap the JSON in markdown code blocks.

Question: {text}
"""
        try:
            res = ollama.chat(model=settings.OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.0})
            content = res['message']['content'].strip()
            # Clean up potential markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            parsed = json.loads(content.strip())
            # Validate structure
            if isinstance(parsed, dict) and "entities" in parsed:
                return parsed  # new structured format
            elif isinstance(parsed, list):
                return parsed  # legacy list fallback
            return []
        except Exception as e:
            logger.warning(f"Ollama extraction failed: {e}")
            return []

    output_dir = os.path.join(data_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    raw_retrievals_path = os.path.join(output_dir, "raw_retrievals.jsonl")

    with open(raw_retrievals_path, 'w', encoding='utf-8') as out_f:
        for idx, q in enumerate(tqdm(questions, desc="Retrieving all strategies")):
            q_text = q.get("question", "")
            required = q.get("required_chunk_ids", [])
            category = q.get("category", "")
            question_id = q.get("id") or f"q_{idx}"

            # 1. BM25
            bm25_res = []
            try:
                bm25_res = sparse_retrieve(q_text, project_id, data_root, top_k)
            except Exception as e:
                logger.warning(f"BM25 error on Q {question_id}: {e}")

            # 2. Dense
            dense_res = []
            try:
                dense_res = dense_query_fn(q_text, project_id, top_k)
            except Exception as e:
                logger.warning(f"Dense error on Q {question_id}: {e}")

            # 3. Metadata
            meta_res = []
            try:
                metadata_filter = q.get("required_metadata_filter") or ""
                file_name_filter = None
                if "file_name=" in metadata_filter:
                    file_name_filter = metadata_filter.split("file_name=")[-1].strip()
                meta_res = metadata_retrieve(project_id, data_root, file_name=file_name_filter, top_k=top_k)
            except Exception as e:
                logger.warning(f"Metadata error on Q {question_id}: {e}")

            # 4. Graph (Query-as-Seed via LLM extraction with synonyms + relation types)
            graph_res = []
            try:
                graph_traversal = graph_retriever.graph_retrieve(
                    entity_texts=[q_text],
                    project_id=project_id,
                    top_k=top_k,
                    entity_extractor=llm_extract_entities,
                    seed_chunk_ids=None,
                    relation_types=None,  # relation types are extracted inside graph_retrieve via the extractor
                )
                reachable_ids = [r["chunk_id"] for r in graph_traversal if "chunk_id" in r]

                if reachable_ids:
                    expr = "chunk_id in [" + ",".join(f"'{cid}'" for cid in reachable_ids) + "]"
                    graph_res = dense_query_fn(q_text, project_id, top_k, expr=expr)
            except Exception as e:
                logger.warning(f"Graph Query-Seed error on Q {question_id}: {e}")

            # Hits
            bm25_hit = _required_found(bm25_res, required) or (not required and category in ("metadata", "exact_id") and bool(bm25_res))
            dense_hit = _required_found(dense_res, required) or (not required and category in ("metadata", "exact_id") and bool(dense_res))
            meta_hit = _required_found(meta_res, required) or (not required and bool(meta_res))
            graph_hit = _required_found(graph_res, required) or (not required and category in ("metadata", "exact_id") and bool(graph_res))

            row = {
                "question_id": question_id,
                "question": q_text,
                "category": category,
                "project_id": project_id,
                "required_chunk_ids": required,
                "bm25_chunks": _chunk_ids_from_results(bm25_res),
                "dense_chunks": _chunk_ids_from_results(dense_res),
                "metadata_dense_chunks": _chunk_ids_from_results(meta_res),
                "graph_query_chunks": _chunk_ids_from_results(graph_res),
                "bm25_hit": bm25_hit,
                "dense_hit": dense_hit,
                "metadata_dense_hit": meta_hit,
                "graph_query_hit": graph_hit,
            }
            out_f.write(json.dumps(row) + "\n")

    graph_retriever.close()
    logger.info("Step 1 complete: outputs/raw_retrievals.jsonl created.")

if __name__ == "__main__":
    main()

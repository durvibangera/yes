import sys
import os
import json
import logging
import re
import numpy as np
from typing import List, Dict, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.config import settings
from pymilvus import connections, Collection
from FlagEmbedding import BGEM3FlagModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OverlapCalculator")

def _safe_collection_name(project_id: str) -> str:
    return "project_" + re.sub(r'[^a-zA-Z0-9]', '_', project_id)

def get_chunk_embeddings(project_id: str, chunk_ids: List[str]) -> Dict[str, List[float]]:
    if not chunk_ids:
        return {}
    
    if not connections.has_connection("default"):
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=str(settings.MILVUS_PORT)
        )
    
    col_name = _safe_collection_name(project_id)
    collection = Collection(col_name)
    collection.load()
    
    embeddings = {}
    # Batch queries to avoid expression length limits in Milvus
    BATCH_SIZE = 100
    for i in range(0, len(chunk_ids), BATCH_SIZE):
        batch = chunk_ids[i:i+BATCH_SIZE]
        # Format string array for Milvus in expression
        arr_str = ", ".join(f"'{cid}'" for cid in batch)
        expr = f"chunk_id in [{arr_str}]"
        
        try:
            results = collection.query(expr=expr, output_fields=["chunk_id", "embedding"])
            for r in results:
                embeddings[r["chunk_id"]] = r["embedding"]
        except Exception as e:
            logger.warning(f"Failed to query embeddings for batch {batch}: {e}")
            
    return embeddings

def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def compute_semantic_cosine(chunks_a: List[str], chunks_b: List[str], embedding_map: Dict[str, List[float]]) -> float:
    embs_a = [embedding_map[cid] for cid in chunks_a if cid in embedding_map]
    embs_b = [embedding_map[cid] for cid in chunks_b if cid in embedding_map]
    
    if not embs_a and not embs_b:
        return 1.0
    if not embs_a or not embs_b:
        return 0.0
        
    mean_a = np.mean(embs_a, axis=0)
    mean_b = np.mean(embs_b, axis=0)
    return cosine_similarity(mean_a, mean_b)

def compute_q_vs_chunks_cosine(q_emb: np.ndarray, chunks: List[str], embedding_map: Dict[str, List[float]]) -> float:
    embs = [embedding_map[cid] for cid in chunks if cid in embedding_map]
    if not embs:
        return 0.0
    mean_chunk = np.mean(embs, axis=0)
    return cosine_similarity(q_emb, mean_chunk)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, required=True)
    args = parser.parse_args()
    project_id = args.project_id
    
    data_root = settings.DATA_ROOT
    raw_path = os.path.join(data_root, "outputs", "raw_retrievals.jsonl")
    out_path = os.path.join(data_root, "outputs", "overlap_scores.jsonl")
    
    if not os.path.exists(raw_path):
        logger.error(f"Raw retrievals file not found at {raw_path}")
        return
        
    # Read raw retrievals
    rows = []
    all_chunk_ids = set()
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
            all_chunk_ids.update(row["bm25_chunks"])
            all_chunk_ids.update(row["dense_chunks"])
            all_chunk_ids.update(row.get("graph_query_chunks", []))
            all_chunk_ids.update(row["metadata_dense_chunks"])
            
    logger.info(f"Loaded {len(rows)} raw retrievals. Total unique chunks collected: {len(all_chunk_ids)}")
    
    # Pre-fetch embeddings from Milvus for all retrieved chunks
    logger.info("Fetching embeddings from Milvus...")
    embedding_map = get_chunk_embeddings(project_id, list(all_chunk_ids))
    logger.info(f"Successfully loaded {len(embedding_map)} embeddings.")
    
    logger.info("Loading BGE-M3 model to embed questions...")
    embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in rows:
            bm25_chunks = set(r["bm25_chunks"])
            dense_chunks = set(r["dense_chunks"])
            graph_query_chunks = set(r.get("graph_query_chunks", []))
            
            # Pairwise Jaccard
            bm25_vs_dense_jaccard = jaccard(bm25_chunks, dense_chunks)
            bm25_vs_graph_jaccard = jaccard(bm25_chunks, graph_query_chunks)
            dense_vs_graph_jaccard = jaccard(dense_chunks, graph_query_chunks)
            
            # Pairwise Cosine (Semantic Similarity)
            bm25_vs_dense_cosine = compute_semantic_cosine(r["bm25_chunks"], r["dense_chunks"], embedding_map)
            bm25_vs_graph_cosine = compute_semantic_cosine(r["bm25_chunks"], r.get("graph_query_chunks", []), embedding_map)
            dense_vs_graph_cosine = compute_semantic_cosine(r["dense_chunks"], r.get("graph_query_chunks", []), embedding_map)
            
            # Question vs Chunk Cosine (drift check)
            q_emb = embedding_model.encode([r["question"]], batch_size=1, max_length=512)['dense_vecs'][0]
            q_vs_bm25_cosine = compute_q_vs_chunks_cosine(q_emb, r["bm25_chunks"], embedding_map)
            q_vs_dense_cosine = compute_q_vs_chunks_cosine(q_emb, r["dense_chunks"], embedding_map)
            q_vs_graph_cosine = compute_q_vs_chunks_cosine(q_emb, r.get("graph_query_chunks", []), embedding_map)
            
            score_row = {
                "question_id": r["question_id"],
                "question": r["question"],
                "category": r["category"],
                "project_id": r["project_id"],
                "bm25_backend": "whoosh",
                
                "bm25_hit": r["bm25_hit"],
                "dense_hit": r["dense_hit"],
                "graph_query_hit": r.get("graph_query_hit", False),
                "metadata_dense_hit": r["metadata_dense_hit"],
                
                # Jaccard (chunk ID overlap)
                "bm25_vs_graph_jaccard": bm25_vs_graph_jaccard,
                "bm25_vs_dense_jaccard": bm25_vs_dense_jaccard,
                "dense_vs_graph_jaccard": dense_vs_graph_jaccard,
                
                # Cosine (semantic similarity between retrieval sets)
                "bm25_vs_graph_cosine": bm25_vs_graph_cosine,
                "bm25_vs_dense_cosine": bm25_vs_dense_cosine,
                "dense_vs_graph_cosine": dense_vs_graph_cosine,
                
                # Query vs Retrieved Chunks (drift check)
                "q_vs_bm25_cosine": q_vs_bm25_cosine,
                "q_vs_dense_cosine": q_vs_dense_cosine,
                "q_vs_graph_cosine": q_vs_graph_cosine,
                
                # Raw chunk IDs for qualitative report
                "bm25_chunks": r["bm25_chunks"],
                "dense_chunks": r["dense_chunks"],
                "graph_query_chunks": r.get("graph_query_chunks", []),
                "metadata_dense_chunks": r["metadata_dense_chunks"],
                "required_chunk_ids": r["required_chunk_ids"]
            }
            f.write(json.dumps(score_row) + "\n")
            
    logger.info("Step 2 complete: outputs/overlap_scores.jsonl created.")

if __name__ == "__main__":
    main()

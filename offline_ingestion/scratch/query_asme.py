import sys
import os
import json
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.config import settings
from retrieval_stubs.sparse_retriever import sparse_retrieve
from retrieval_stubs.dense_retriever import dense_retrieve
from retrieval_stubs.graph_retriever import GraphRetriever
from FlagEmbedding import BGEM3FlagModel

def main():
    project_id = "ASME 2019"
    data_root = settings.DATA_ROOT
    db_path = os.path.join(data_root, "metadata.db")

    question = "A manufacturer produces a WP24 cold-formed fitting from raw material that was not previously tensile tested, and the fitting has a section thickness of 0.45 in. After performing a weld repair, what testing, heat treatment, weld metal composition, and post-repair thermal requirements must be satisfied before the fitting complies with the specification?"

    print("--- 1. Running BM25 Retrieval ---")
    bm25_res = sparse_retrieve(question, project_id, data_root, top_k=10)
    print("BM25 Chunk IDs:", [r["chunk_id"] for r in bm25_res])

    print("\n--- 2. Running Dense Retrieval ---")
    embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    q_emb = embedding_model.encode([question], batch_size=1, max_length=512)['dense_vecs'][0].tolist()
    dense_res = dense_retrieve(q_emb, project_id, top_k=10)
    print("Dense Chunk IDs:", [r["chunk_id"] for r in dense_res])

    print("\n--- 3. Running Graph Retrieval (Fast Keyword/Code Fallback) ---")
    graph_retriever = GraphRetriever()
    # Skip Ollama LLM call to run in under 5 seconds, let graph_retriever use keyword & regex fallback
    graph_traversal = graph_retriever.graph_retrieve(
        entity_texts=[question],
        project_id=project_id,
        top_k=10,
        entity_extractor=None
    )
    graph_chunk_ids = [r["chunk_id"] for r in graph_traversal if "chunk_id" in r]
    print("Graph Chunk IDs:", graph_chunk_ids)

    # Fetch chunk texts from DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    all_ids = []
    for r in bm25_res + dense_res + graph_traversal:
        cid = r["chunk_id"]
        if cid not in all_ids:
            all_ids.append(cid)

    print(f"\nTotal Unique Chunks Retrieved: {len(all_ids)}")

    placeholders = ",".join("?" for _ in all_ids)
    cursor.execute(f"SELECT chunk_id, file_name, file_path, char_start, char_end FROM chunks WHERE chunk_id IN ({placeholders})", all_ids)
    rows = cursor.fetchall()
    
    output_data = []
    print("\n--- RETRIEVED CHUNKS CONTENT ---")
    for chunk_id, file_name, file_path, start, end in rows:
        abs_path = os.path.isabs(file_path) and file_path or os.path.join(data_root, file_path)
        if os.path.exists(abs_path):
            with open(abs_path, 'r', encoding='utf-8') as f:
                f.seek(start)
                txt = f.read(end - start)
                output_data.append({
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "text": txt
                })
                print(f"\n================ CHUNK ID: {chunk_id} ({file_name}) ================")
                print(txt)
        else:
            print(f"File not found: {abs_path}")

    # Save to a json for quick reading if needed
    with open("retrieved_results.json", "w", encoding="utf-8") as out_f:
        json.dump(output_data, out_f, indent=2)

    graph_retriever.close()
    conn.close()

if __name__ == "__main__":
    main()

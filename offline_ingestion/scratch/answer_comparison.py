import sys
import os
import sqlite3
import json
import ollama

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion.config import settings

db_path = r"d:\Durvi_project\E Books Updated_md\metadata.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

dense_ids = [
    'ASME_2019_ASME_II_PART_A1__2019__md_4069', 'ASME_2019_ASME_II_PART_A1__2019__md_4044', 
    'ASME_2019_ASME_II_PART_A1__2019__md_4059', 'ASME_2019_ASME_II_PART_A1__2019__md_7638', 
    'ASME_2019_ASME_II_PART_A1__2019__md_7632', 'ASME_2019_ASME_II_PART_A1__2019__md_7633', 
    'ASME_2019_ASME_II_PART_A2__2019__md_6274', 'ASME_2019_ASME_II_PART_A1__2019__md_3142', 
    'ASME_2019_ASME_II_PART_A1__2019__md_4039', 'ASME_2019_ASME_II_PART_A1__2019__md_4060'
]

graph_ids = [
    'ASME_2019_ASME_II_PART_A1__2019__md_4042', 'ASME_2019_ASME_II_PART_A1__2019__md_4069', 
    'ASME_2019_ASME_II_PART_A1__2019__md_6484', 'ASME_2019_ASME_II_PART_A1__2019__md_4049', 
    'ASME_2019_ASME_II_PART_A1__2019__md_4068', 'ASME_2019_ASME_II_PART_A1__2019__md_4062', 
    'ASME_2019_ASME_II_PART_A1__2019__md_4044', 'ASME_2019_ASME_II_PART_A1__2019__md_1968', 
    'ASME_2019_ASME_II_PART_A2__2019__md_7074', 'ASME_2019_ASME_II_PART_A1__2019__md_6644'
]

def fetch_chunk_texts(ids):
    placeholders = ",".join("?" for _ in ids)
    cursor.execute(f"SELECT chunk_id, file_path, char_start, char_end FROM chunks WHERE chunk_id IN ({placeholders})", ids)
    rows = cursor.fetchall()
    
    results = {}
    for chunk_id, file_path, start, end in rows:
        possible_paths = [
            file_path,
            os.path.join(r"d:\Durvi_project", file_path),
            os.path.join(r"d:\Durvi_project\E Books Updated_md", file_path)
        ]
        for p in possible_paths:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    f.seek(start)
                    results[chunk_id] = f.read(end - start)
                break
    return results

dense_texts = fetch_chunk_texts(dense_ids)
graph_texts = fetch_chunk_texts(graph_ids)

question = "A manufacturer produces a WP24 cold-formed fitting from raw material that was not previously tensile tested, and the fitting has a section thickness of 0.45 in. After performing a weld repair, what testing, heat treatment, weld metal composition, and post-repair thermal requirements must be satisfied before the fitting complies with the specification?"

# Generate Vector Only Answer
dense_context = "\n\n".join([f"--- Chunk {k} ---\n{v}" for k, v in dense_texts.items()])
dense_prompt = f"Answer the user's question based ONLY on the provided context chunks. Be as detailed as possible. If the context does not contain enough information to answer a part of the question, state it explicitly.\n\nContext:\n{dense_context}\n\nQuestion:\n{question}"

print("Asking LLM for Vector-only answer...")
res_dense = ollama.chat(model=settings.OLLAMA_MODEL, messages=[{"role": "user", "content": dense_prompt}])
dense_answer = res_dense['message']['content']

# Generate Graph Only Answer
graph_context = "\n\n".join([f"--- Chunk {k} ---\n{v}" for k, v in graph_texts.items()])
graph_prompt = f"Answer the user's question based ONLY on the provided context chunks. Be as detailed as possible. If the context does not contain enough information to answer a part of the question, state it explicitly.\n\nContext:\n{graph_context}\n\nQuestion:\n{question}"

print("Asking LLM for Graph-only answer...")
res_graph = ollama.chat(model=settings.OLLAMA_MODEL, messages=[{"role": "user", "content": graph_prompt}])
graph_answer = res_graph['message']['content']

output_report = f"""=== VECTOR ONLY ANSWER ===
{dense_answer}

=== GRAPH ONLY ANSWER ===
{graph_answer}
"""

with open(r"d:\Durvi_project\offline_ingestion\scratch\answers_comparison.txt", 'w', encoding='utf-8') as f:
    f.write(output_report)

print("Comparison complete!")
conn.close()

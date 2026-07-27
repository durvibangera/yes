import sqlite3
import os
import json

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

output = []

output.append("=== DENSE ONLY RETRIEVAL ===")
for cid in dense_ids:
    if cid not in graph_ids and cid in dense_texts:
        output.append(f"\nID: {cid}")
        output.append(dense_texts[cid][:500])

output.append("\n=== GRAPH ONLY RETRIEVAL ===")
for cid in graph_ids:
    if cid not in dense_ids and cid in graph_texts:
        output.append(f"\nID: {cid}")
        output.append(graph_texts[cid][:500])

output.append("\n=== BOTH RETRIEVED ===")
for cid in graph_ids:
    if cid in dense_ids and cid in graph_texts:
        output.append(f"\nID: {cid}")
        output.append(graph_texts[cid][:500])

with open(r"d:\Durvi_project\offline_ingestion\scratch\comparison_report.txt", 'w', encoding='utf-8') as f:
    f.write("\n".join(output))

print("Saved report to comparison_report.txt")

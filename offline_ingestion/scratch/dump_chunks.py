import sqlite3
import os
import json

db_path = r"d:\Durvi_project\E Books Updated_md\metadata.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

chunk_ids = [
    'ASME_2019_ASME_II_PART_A1__2019__md_4042', 'ASME_2019_ASME_II_PART_A1__2019__md_4069', 
    'ASME_2019_ASME_II_PART_A1__2019__md_6484', 'ASME_2019_ASME_II_PART_A1__2019__md_4049', 
    'ASME_2019_ASME_II_PART_A1__2019__md_4068', 'ASME_2019_ASME_II_PART_A1__2019__md_4062', 
    'ASME_2019_ASME_II_PART_A1__2019__md_4044', 'ASME_2019_ASME_II_PART_A1__2019__md_4060',
    'ASME_2019_ASME_IX__2019__md_718', 'ASME_2019_ASME_II_PART_A1__2019__md_4039',
    'ASME_2019_ASME_II_PART_A1__2019__md_4059', 'ASME_2019_ASME_II_PART_A1__2019__md_7638', 
    'ASME_2019_ASME_II_PART_A1__2019__md_7632', 'ASME_2019_ASME_II_PART_A1__2019__md_7633', 
    'ASME_2019_ASME_II_PART_A2__2019__md_6274', 'ASME_2019_ASME_II_PART_A1__2019__md_3142'
]

placeholders = ",".join("?" for _ in chunk_ids)
cursor.execute(f"SELECT chunk_id, file_path, char_start, char_end FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
rows = cursor.fetchall()

results = {}

for chunk_id, file_path, start, end in rows:
    possible_paths = [
        file_path,
        os.path.join(r"d:\Durvi_project", file_path),
        os.path.join(r"d:\Durvi_project\E Books Updated_md", file_path)
    ]
    
    found = False
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(start)
                txt = f.read(end - start)
                results[chunk_id] = txt
                found = True
            break
            
    if not found:
        results[chunk_id] = f"FILE NOT FOUND. Paths tried: {possible_paths}"

with open(r"d:\Durvi_project\offline_ingestion\scratch\chunk_results.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} chunks to chunk_results.json")

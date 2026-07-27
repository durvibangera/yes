import sys
import os
import sqlite3
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

db_path = r"d:\Durvi_project\E Books Updated_md\metadata.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

chunk_ids = [
    'ASME_2019_ASME_IX__2019__md_718', 'ASME_2019_ASME_II_PART_A1__2019__md_4060', 'ASME_2019_ASME_IX__2019__md_3121', 'ASME_2019_ASME_II_PART_A1__2019__md_7439', 'ASME_2019_ASME_IX__2019__md_3131', 'ASME_2019_ASME_IX__2019__md_769', 'ASME_2019_ASME_IX__2019__md_2977', 'ASME_2019_ASME_II_PART_A1__2019__md_2733', 'ASME_2019_ASME_II_PART_A1__2019__md_7663', 'ASME_2019_ASME_IX__2019__md_3122',
    'ASME_2019_ASME_II_PART_A1__2019__md_4069', 'ASME_2019_ASME_II_PART_A1__2019__md_4044', 'ASME_2019_ASME_II_PART_A1__2019__md_4059', 'ASME_2019_ASME_II_PART_A1__2019__md_7638', 'ASME_2019_ASME_II_PART_A1__2019__md_7632', 'ASME_2019_ASME_II_PART_A1__2019__md_7633', 'ASME_2019_ASME_II_PART_A2__2019__md_6274', 'ASME_2019_ASME_II_PART_A1__2019__md_3142', 'ASME_2019_ASME_II_PART_A1__2019__md_4039'
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

import ollama
from ingestion.config import settings

question = "A manufacturer produces a WP24 cold-formed fitting from raw material that was not previously tensile tested, and the fitting has a section thickness of 0.45 in. After performing a weld repair, what testing, heat treatment, weld metal composition, and post-repair thermal requirements must be satisfied before the fitting complies with the specification?"

context = "\n\n".join([f"--- Chunk {k} ---\n{v}" for k, v in results.items()])
prompt = f"Answer the user's question based ONLY on the provided context chunks. Be as detailed as possible.\n\nContext:\n{context}\n\nQuestion:\n{question}"

print("Asking LLM...")
res = ollama.chat(model=settings.OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
print(res['message']['content'].encode('utf-8', errors='replace').decode('utf-8'))

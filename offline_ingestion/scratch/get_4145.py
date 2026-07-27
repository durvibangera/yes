import sqlite3
import os

db_path = r"d:\Durvi_project\E Books Updated_md\metadata.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

chunk_ids = ['ASME_2019_ASME_II_PART_A1__2019__md_4145', 'ASME_2019_ASME_II_PART_A1__2019__md_4146']
placeholders = ",".join("?" for _ in chunk_ids)
cursor.execute(f"SELECT chunk_id, file_path, char_start, char_end FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
rows = cursor.fetchall()

results = {}
for chunk_id, file_path, start, end in rows:
    p = os.path.join(r"d:\Durvi_project\E Books Updated_md", file_path)
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(start)
            results[chunk_id] = f.read(end - start)

output = []
for k, v in results.items():
    output.append(f"=== {k} ===")
    output.append(v)

with open(r"d:\Durvi_project\offline_ingestion\scratch\get_4145.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Saved to get_4145.txt")
conn.close()

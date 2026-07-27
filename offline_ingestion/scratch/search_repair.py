import sqlite3
import os

db_path = r"d:\Durvi_project\E Books Updated_md\metadata.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query all chunks in ASME 2019 folder
cursor.execute("""
    SELECT chunk_id, file_path, char_start, char_end 
    FROM chunks 
    WHERE file_path LIKE '%ASME 2019%'
""")
rows = cursor.fetchall()

results = []
for chunk_id, file_path, start, end in rows:
    abs_path = os.path.join(r"d:\Durvi_project\E Books Updated_md", file_path)
    if os.path.exists(abs_path):
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(start)
            txt = f.read(end - start)
            
            # Check for Section 15.3 or 9.4 or 9.3 or 8.5 in the context of SA-234 / SA-960 / SA-403
            if ("15.3" in txt and "repair" in txt.lower()) or ("9.4" in txt and "tensile" in txt.lower()) or ("9.3" in txt and "cold" in txt.lower()) or ("8.5" in txt and "composition" in txt.lower()):
                results.append((chunk_id, file_path, txt))

output = []
for chunk_id, path, text in results:
    output.append("="*60)
    output.append(f"CHUNK ID: {chunk_id} | PATH: {path}")
    output.append(text)

with open(r"d:\Durvi_project\offline_ingestion\scratch\repair_clauses.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print(f"Saved {len(results)} chunks to repair_clauses.txt")
conn.close()

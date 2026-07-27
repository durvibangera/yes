import sqlite3
import os

db_path = r"d:\Durvi_project\E Books Updated_md\metadata.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

clauses = ["8.5", "9.3", "9.4", "15.3"]
results = {}

for clause in clauses:
    cursor.execute("""
        SELECT chunk_id, file_path, char_start, char_end 
        FROM chunks 
        WHERE file_path LIKE '%ASME 2019%'
    """)
    rows = cursor.fetchall()
    
    matched = []
    for chunk_id, file_path, start, end in rows:
        abs_path = os.path.join(r"d:\Durvi_project\E Books Updated_md", file_path)
        if os.path.exists(abs_path):
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(start)
                txt = f.read(end - start)
                if clause in txt:
                    matched.append((chunk_id, file_path, txt[:400].replace('\n', ' ')))
        if len(matched) >= 30: # Look for more matches
            break
    results[clause] = matched

output = []
for clause, matches in results.items():
    output.append(f"\n=== CLAUSE {clause} MATCHES ===")
    for m in matches:
        output.append(f"Chunk ID: {m[0]} | Path: {m[1]} | Preview: {m[2]}...")

with open(r"d:\Durvi_project\offline_ingestion\scratch\search_clauses_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Saved report to search_clauses_report.txt")
conn.close()

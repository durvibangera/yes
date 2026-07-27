import sys
import os
import json
import logging
import sqlite3
import random
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReportGenerator")

def get_chunk_info(chunk_ids: List[str], db_conn) -> Dict[str, Dict[str, Any]]:
    if not chunk_ids:
        return {}
    cursor = db_conn.cursor()
    # Batch query sqlite
    placeholders = ",".join("?" for _ in chunk_ids)
    cursor.execute(f"SELECT chunk_id, file_path, char_start, char_end FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
    rows = cursor.fetchall()
    
    info = {}
    for r in rows:
        info[r[0]] = {
            "file_path": r[1],
            "char_start": r[2],
            "char_end": r[3]
        }
    return info

def read_chunk_text(info: Dict[str, Any], project_root: str) -> str:
    if not info:
        return "[NOT FOUND IN DB]"
    file_path = info["file_path"]
    char_start = info["char_start"]
    char_end = info["char_end"]
    
    if not os.path.isabs(file_path):
        # The file paths in db might be relative to settings.DATA_ROOT or workspace root
        possible_paths = [
            os.path.join(settings.DATA_ROOT, file_path),
            os.path.join(project_root, file_path),
            file_path
        ]
        for p in possible_paths:
            if os.path.exists(p):
                file_path = p
                break
                
    if not os.path.exists(file_path):
        return f"[FILE NOT FOUND: {file_path}]"
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(char_start)
            return f.read(char_end - char_start)
    except Exception as e:
        return f"[ERROR READING FILE: {e}]"

def main():
    # Set random seed for deterministic sampling
    random.seed(42)
    
    data_root = settings.DATA_ROOT
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scores_path = os.path.join(data_root, "outputs", "overlap_scores.jsonl")
    report_path = os.path.join(data_root, "outputs", "inspection_report.md")
    db_path = os.path.join(data_root, "metadata.db")

    if not os.path.exists(scores_path):
        logger.error(f"Overlap scores file not found at {scores_path}")
        return
        
    if not os.path.exists(db_path):
        logger.error(f"metadata.db not found at {db_path}")
        return

    db_conn = sqlite3.connect(db_path)

    # Read scores
    rows = []
    with open(scores_path, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))

    # Identify unique wins
    # Unique win is defined as one strategy hitting while others miss
    unique_wins_map = {
        "bm25": [],
        "dense": [],
        "graph_query": []
    }
    
    disagreement_cases = []
    
    for r in rows:
        bm25_hit = r["bm25_hit"]
        dense_hit = r["dense_hit"]
        graph_hit = r.get("graph_query_hit", False)
        
        hits = []
        if bm25_hit: hits.append("bm25")
        if dense_hit: hits.append("dense")
        if graph_hit: hits.append("graph_query")
        
        if len(hits) == 1:
            unique_wins_map[hits[0]].append(r)
            
        # Disagreement is any mismatch in hit status
        if len(set([bm25_hit, dense_hit, graph_hit])) > 1:
            disagreement_cases.append(r)

    # If there are no disagreement cases (e.g., because there's no ground truth and all hits are False),
    # just take a random sample of ALL rows for manual inspection.
    if not disagreement_cases:
        disagreement_cases = rows

    # Group disagreements by category and sample 15-20
    disagreements_by_cat = {}
    for r in disagreement_cases:
        cat = r["category"]
        if cat not in disagreements_by_cat:
            disagreements_by_cat[cat] = []
        disagreements_by_cat[cat].append(r)
        
    sampled_disagreements = []
    for cat, cases in disagreements_by_cat.items():
        sample_size = min(len(cases), 20)
        sampled_disagreements.extend(random.sample(cases, sample_size))

    # We need to collect all chunk texts that we want to print
    # This prevents loading/opening files over and over
    chunks_to_fetch = set()
    for row_list in list(unique_wins_map.values()) + [sampled_disagreements]:
        for r in row_list:
            chunks_to_fetch.update(r["required_chunk_ids"])
            chunks_to_fetch.update(r["bm25_chunks"][:10])
            chunks_to_fetch.update(r.get("graph_query_chunks", [])[:10])
            
    chunk_meta_map = get_chunk_info(list(chunks_to_fetch), db_conn)

    # Generate Report Markdown
    md = []
    md.append("# Retrieval Overlap & Manual Inspection Report")
    md.append(f"**BM25 Backend used for this run:** `whoosh`  ")
    md.append("*(This report was generated automatically by `diagnostics/report_generator.py`)*\n")
    
    # Check for missing required chunks in DB
    missing_required = []
    for r in rows:
        for req in r["required_chunk_ids"]:
            if req not in chunk_meta_map:
                missing_required.append((r["question_id"], req))
                
    if missing_required:
        md.append("## ⚠️ WARNING: Missing Required Chunks in Database")
        md.append("The following question required chunks that were NOT found in `metadata.db`:")
        for q_id, chunk_id in missing_required[:10]:
            md.append(f"- **Q ID:** `{q_id}` is missing required chunk `{chunk_id}`")
        if len(missing_required) > 10:
            md.append(f"- ... and {len(missing_required) - 10} more missing chunks.")
        md.append("\n")

    # ── Section 1: Unique Wins ────────────────────────────────────────────────
    md.append("## Section 1: Unique Strategy Wins")
    md.append("Questions where exactly ONE strategy succeeded while all others failed.")
    
    for strategy, cases in unique_wins_map.items():
        md.append(f"### Unique Wins for `{strategy}` (Count: {len(cases)})")
        if not cases:
            md.append("*No unique wins found for this strategy.*\n")
            continue
            
        for r in cases[:5]:  # print up to 5 examples
            md.append(format_case(r, chunk_meta_map, project_root))
        md.append("\n")

    # ── Section 2: Sampled Disagreement Cases ──────────────────────────────────
    md.append("## Section 2: Sampled Disagreement Cases")
    md.append("A random sample of up to 20 disagreement cases per category (where strategies disagree on hit/miss).")
    
    for r in sampled_disagreements[:40]:  # Cap at 40 total to keep readable
        md.append(format_case(r, chunk_meta_map, project_root))

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))

    db_conn.close()
    logger.info("Step 4 complete: outputs/inspection_report.md created.")

def format_case(r: Dict[str, Any], chunk_meta_map: Dict[str, Any], project_root: str) -> str:
    lines = []
    lines.append(f"#### Question: \"{r['question']}\"")
    lines.append(f"- **ID:** `{r['question_id']}` | **Category:** `{r['category']}` | **Project:** `{r['project_id']}`")
    lines.append(f"- **Required chunks:** {r['required_chunk_ids']}")
    lines.append(f"- **Overlap (Jaccard):** BM25 vs Graph: `{r['bm25_vs_graph_jaccard']:.3f}` | Dense vs Graph: `{r['dense_vs_graph_jaccard']:.3f}`")
    lines.append(f"- **Overlap (Cosine):**  BM25 vs Graph: `{r['bm25_vs_graph_cosine']:.3f}` | Dense vs Graph: `{r['dense_vs_graph_cosine']:.3f}`")
    lines.append("")
    
    # BM25 Results
    bm25_hit_str = "✅ [HIT]" if r["bm25_hit"] else "❌ [MISS]"
    lines.append(f"**BM25 retrieved (top 10):** {bm25_hit_str}")
    for idx, cid in enumerate(r["bm25_chunks"][:10]):
        text = read_chunk_text(chunk_meta_map.get(cid), project_root)
        snippet = text.replace('\n', ' ')[:120].strip()
        is_req = "⭐ " if cid in r["required_chunk_ids"] else ""
        lines.append(f"{idx+1}. {is_req}`{cid}` — \"{snippet}...\"")
    lines.append("")

    # Graph Results
    graph_hit_str = "✅ [HIT]" if r.get("graph_query_hit", False) else "❌ [MISS]"
    lines.append(f"**Graph (Query-Seed) retrieved (top 10):** {graph_hit_str}")
    for idx, cid in enumerate(r.get("graph_query_chunks", [])[:10]):
        text = read_chunk_text(chunk_meta_map.get(cid), project_root)
        snippet = text.replace('\n', ' ')[:120].strip()
        is_req = "⭐ " if cid in r["required_chunk_ids"] else ""
        lines.append(f"{idx+1}. {is_req}`{cid}` — \"{snippet}...\"")
    lines.append("\n---\n")
    return "\n".join(lines)

if __name__ == "__main__":
    main()

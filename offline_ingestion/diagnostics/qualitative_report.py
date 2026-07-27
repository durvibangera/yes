"""
Qualitative Report Generator — Tier 5/6 Side-by-Side Chunk Comparison.
For each of the 20 hardest questions (Q41-Q60), shows the top-3 retrieved
chunk texts from BM25, Dense, and Graph (Query-as-Seed) side by side,
with Jaccard and Cosine metrics per question.
"""
import sys
import os
import json
import sqlite3
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QualitativeReport")

CHUNK_TEXT_CACHE = {}

def get_chunk_text(chunk_id: str, data_root: str) -> str:
    """Read actual chunk text from the source markdown files via SQLite metadata."""
    if chunk_id in CHUNK_TEXT_CACHE:
        return CHUNK_TEXT_CACHE[chunk_id]

    db_path = os.path.join(data_root, "metadata.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT file_path, char_start, char_end FROM chunks WHERE chunk_id = ?",
        (chunk_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return f"[Chunk not found: {chunk_id}]"

    file_path, char_start, char_end = row
    # Resolve relative paths against DATA_ROOT
    if not os.path.isabs(file_path):
        resolved = os.path.join(data_root, file_path)
        if os.path.exists(resolved):
            file_path = resolved
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        text = content[char_start:char_end].strip()
    except Exception as e:
        text = f"[Error reading file {file_path}: {e}]"

    CHUNK_TEXT_CACHE[chunk_id] = text
    return text


def render_chunks(chunk_ids: list, data_root: str, top_n: int = 3) -> str:
    if not chunk_ids:
        return "> _No chunks retrieved_\n"
    lines = []
    for i, cid in enumerate(chunk_ids[:top_n]):
        text = get_chunk_text(cid, data_root)
        # Truncate very long chunks for readability
        if len(text) > 800:
            text = text[:800] + "…"
        lines.append(f"**Chunk {i+1}** `{cid}`\n```\n{text}\n```")
    return "\n\n".join(lines) + "\n"


def main():
    data_root = settings.DATA_ROOT
    scores_path = os.path.join(data_root, "outputs", "ABS_Standards_overlap_scores.jsonl")
    raw_path = os.path.join(data_root, "outputs", "ABS_Standards_raw_retrievals.jsonl")
    out_path = os.path.join(data_root, "outputs", "ABS_Standards_qualitative_report.md")

    # Try the generic file if project-specific copy doesn't exist yet
    if not os.path.exists(scores_path):
        scores_path = os.path.join(data_root, "outputs", "overlap_scores.jsonl")
    if not os.path.exists(raw_path):
        raw_path = os.path.join(data_root, "outputs", "raw_retrievals.jsonl")

    if not os.path.exists(scores_path) or not os.path.exists(raw_path):
        logger.error(f"Missing output files. Run main_diagnostic.py first.")
        return

    # Load overlap scores (has metrics per question)
    scores = {}
    with open(scores_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            scores[r["question_id"]] = r

    # Load raw retrievals (has actual chunk lists)
    raws = {}
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            raws[r["question_id"]] = r

    # Focus on the last 20 questions (Tier 5 + Tier 6, Q41-Q60)
    tier5_6_ids = [qid for qid in scores if scores[qid]["question_id"] in
                   [f"q_tier_{i}" for i in range(41, 61)]]
    
    # If IDs don't match expected format, just take the last 20
    all_ids = list(scores.keys())
    if not tier5_6_ids:
        tier5_6_ids = all_ids[-20:] if len(all_ids) >= 20 else all_ids

    lines = [
        "# ABS Standards — Qualitative Chunk Comparison Report",
        "",
        "## Focus: Tier 5 and Tier 6 Questions (Long Multi-hop & Graph-only)",
        "",
        "For each question, this report shows the top-3 retrieved chunks from each strategy,",
        "along with Jaccard overlap and cosine similarity metrics to assess retrieval quality.",
        "",
        "---",
        ""
    ]

    for qid in tier5_6_ids:
        s = scores.get(qid, {})
        r = raws.get(qid, {})
        q_text = s.get("question", r.get("question", "N/A"))

        bm25_chunks = r.get("bm25_chunks", [])
        dense_chunks = r.get("dense_chunks", [])
        graph_chunks = r.get("graph_query_chunks", [])

        bm25_vs_graph_j = s.get("bm25_vs_graph_jaccard", 0.0)
        dense_vs_graph_j = s.get("dense_vs_graph_jaccard", 0.0)
        dense_vs_graph_c = s.get("dense_vs_graph_cosine", 0.0)
        q_vs_bm25 = s.get("q_vs_bm25_cosine", 0.0)
        q_vs_dense = s.get("q_vs_dense_cosine", 0.0)
        q_vs_graph = s.get("q_vs_graph_cosine", 0.0)

        # Determine if Graph found unique chunks not in BM25 or Dense
        graph_set = set(graph_chunks)
        bm25_set = set(bm25_chunks)
        dense_set = set(dense_chunks)
        unique_to_graph = graph_set - bm25_set - dense_set

        lines += [
            f"## Q{qid} — {q_text}",
            "",
            "### Metrics",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| BM25 vs Graph Jaccard | `{bm25_vs_graph_j:.3f}` |",
            f"| Dense vs Graph Jaccard | `{dense_vs_graph_j:.3f}` |",
            f"| Dense vs Graph Cosine | `{dense_vs_graph_c:.3f}` |",
            f"| Q vs BM25 Cosine | `{q_vs_bm25:.3f}` |",
            f"| Q vs Dense Cosine | `{q_vs_dense:.3f}` |",
            f"| Q vs Graph Cosine | `{q_vs_graph:.3f}` |",
            f"| **Chunks unique to Graph** | **{len(unique_to_graph)} / {len(graph_chunks)}** |",
            "",
        ]

        lines += ["### BM25 — Top 3 Chunks", ""]
        lines.append(render_chunks(bm25_chunks, data_root))

        lines += ["### Dense — Top 3 Chunks", ""]
        lines.append(render_chunks(dense_chunks, data_root))

        lines += ["### Graph (Query-as-Seed) — Top 3 Chunks", ""]
        lines.append(render_chunks(graph_chunks, data_root))

        if unique_to_graph:
            lines += [
                f"> [!NOTE]",
                f"> Graph retrieved **{len(unique_to_graph)} chunks** not found by BM25 or Dense.",
                f"> IDs: `{'`, `'.join(list(unique_to_graph)[:5])}`",
                ""
            ]

        lines.append("---\n")

    report = "\n".join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"Qualitative report written to: {out_path}")
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()

import sys
import os
import json
import logging
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BucketAnalysis")

def bucket_jaccard(score: float) -> str:
    if score >= 0.8:
        return "near_identical"
    elif score >= 0.3:
        return "partial_overlap"
    else:
        return "disjoint"

def main():
    data_root = settings.DATA_ROOT
    scores_path = os.path.join(data_root, "outputs", "overlap_scores.jsonl")
    summary_path = os.path.join(data_root, "outputs", "bucket_summary.json")

    if not os.path.exists(scores_path):
        logger.error(f"Scores file not found at {scores_path}")
        return

    # Grouping structure: category -> bucket -> list of rows
    groups = defaultdict(lambda: defaultdict(list))
    
    with open(scores_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            cat = row["category"]
            bkt = bucket_jaccard(row["bm25_vs_graph_jaccard"])
            groups[cat][bkt].append(row)

    summary = {
        "metadata": {
            "bm25_backend": "whoosh",
            "jaccard_buckets": {
                "near_identical": ">= 0.8",
                "partial_overlap": "0.3 - 0.8",
                "disjoint": "< 0.3"
            }
        },
        "results": {}
    }

    for cat, bkts in groups.items():
        summary["results"][cat] = {}
        for bkt, rows in bkts.items():
            count = len(rows)
            bm25_hits = sum(1 for r in rows if r["bm25_hit"])
            dense_hits = sum(1 for r in rows if r["dense_hit"])
            graph_hits = sum(1 for r in rows if r.get("graph_query_hit", False))
            metadata_hits = sum(1 for r in rows if r["metadata_dense_hit"])

            summary["results"][cat][bkt] = {
                "count": count,
                "bm25_accuracy": float(f"{bm25_hits / count:.4f}") if count > 0 else 0.0,
                "dense_accuracy": float(f"{dense_hits / count:.4f}") if count > 0 else 0.0,
                "graph_query_accuracy": float(f"{graph_hits / count:.4f}") if count > 0 else 0.0,
                "metadata_dense_accuracy": float(f"{metadata_hits / count:.4f}") if count > 0 else 0.0,
            }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info("Step 3 complete: outputs/bucket_summary.json created.")

if __name__ == "__main__":
    main()

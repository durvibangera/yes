"""
Rebuild Memgraph from saved checkpoints for ABS Standards.
Skips all GLiNER/embedding steps — just replays the resolved entities
and relations from the .pkl checkpoint files into Memgraph.
"""
import sys
import os
import pickle
import sqlite3
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from ingestion.config import settings
from ingestion.graph_builder import GraphBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RebuildMemgraph")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, default="ABS Standards")
    args = parser.parse_args()
    project_id = args.project_id
    data_root = settings.DATA_ROOT

    # Checkpoint paths (matching the naming convention from main.py)
    project_ckpt_dir = os.path.join(
        data_root, "checkpoints",
        project_id.replace(" ", "_").replace("/", "_")
    )
    entities_ckpt = os.path.join(project_ckpt_dir, "entities.pkl")
    relations_ckpt = os.path.join(project_ckpt_dir, "relations.pkl")
    resolved_ckpt = os.path.join(project_ckpt_dir, f"{project_id}_resolved.pkl")

    # Validate
    for path in [entities_ckpt, relations_ckpt, resolved_ckpt]:
        if not os.path.exists(path):
            logger.error(f"Missing checkpoint: {path}")
            sys.exit(1)

    # Load resolved entities
    logger.info(f"Loading resolved entities from {resolved_ckpt}...")
    with open(resolved_ckpt, 'rb') as f:
        resolved_entities = pickle.load(f)
    logger.info(f"  -> {len(resolved_entities)} canonical entities loaded.")

    # Load relations
    logger.info(f"Loading relations from {relations_ckpt}...")
    with open(relations_ckpt, 'rb') as f:
        ckpt_data = pickle.load(f)
        all_relations = ckpt_data.get('relations', ckpt_data) if isinstance(ckpt_data, dict) else ckpt_data
    logger.info(f"  -> {len(all_relations)} relations loaded.")

    # Load chunks from SQLite (already indexed during original ingestion)
    db_path = os.path.join(data_root, "metadata.db")
    logger.info(f"Loading chunks from SQLite: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT chunk_id, project_id, file_name, file_path, subfolder_path FROM chunks WHERE project_id = ?",
        (project_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    chunks = [
        {
            "chunk_id": r[0],
            "project_id": r[1],
            "file_name": r[2],
            "file_path": r[3],
            "subfolder_path": r[4],
        }
        for r in rows
    ]
    logger.info(f"  -> {len(chunks)} chunks loaded from SQLite.")

    if not chunks:
        logger.error("No chunks found in SQLite for 'ABS Standards'. Ingestion may not have completed.")
        sys.exit(1)

    # Build graph
    logger.info("Connecting to Memgraph and rebuilding graph...")
    uri = f"bolt://{settings.MEMGRAPH_HOST}:{settings.MEMGRAPH_PORT}"
    graph_builder = GraphBuilder(uri)
    graph_builder.build_graph(project_id, resolved_entities, all_relations, chunks)
    graph_builder.close()
    logger.info("Done! ABS Standards successfully re-loaded into Memgraph.")

if __name__ == "__main__":
    main()

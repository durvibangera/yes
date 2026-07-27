"""
Retrieval stubs — metadata filter search via SQLite.
Used by Phase 2 routing labeler. Phase 3 router will call the same functions in production.
"""
import logging
import os
import sqlite3
from typing import List, Dict, Any, Optional

logger = logging.getLogger("MetadataRetriever")


def metadata_retrieve(
    project_id: str,
    data_root: str,
    file_name: Optional[str] = None,
    subfolder_path: Optional[str] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Returns chunk_ids matching structured metadata filters for a project.
    
    Args:
        project_id: The project to search within.
        data_root: Path to the data directory containing metadata.db.
        file_name: Optional exact filename filter (LIKE match).
        subfolder_path: Optional subfolder/chapter path filter (LIKE match).
        top_k: Max number of results to return.
    
    Returns:
        List of dicts with chunk metadata fields.
    """
    db_path = os.path.join(data_root, "metadata.db")
    if not os.path.exists(db_path):
        logger.warning(f"metadata.db not found at {db_path}.")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM chunks WHERE project_id = ?"
    params: List[Any] = [project_id]

    if file_name:
        query += " AND file_name LIKE ?"
        params.append(f"%{file_name}%")
    if subfolder_path:
        query += " AND subfolder_path LIKE ?"
        params.append(f"%{subfolder_path}%")

    query += f" LIMIT {top_k}"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_all_project_metadata(project_id: str, data_root: str) -> List[Dict[str, Any]]:
    """Return all metadata rows for a given project (used for metadata question sampling)."""
    db_path = os.path.join(data_root, "metadata.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT file_name, file_path, subfolder_path FROM chunks WHERE project_id = ?", (project_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_project_chunks(project_id: str, data_root: str) -> List[Dict[str, Any]]:
    """Return all chunk records (with char offsets) for a given project."""
    db_path = os.path.join(data_root, "metadata.db")
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chunks WHERE project_id = ?", (project_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

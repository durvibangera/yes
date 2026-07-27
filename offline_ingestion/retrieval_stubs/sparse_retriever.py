"""
Retrieval stubs — sparse BM25 search.
Used by Phase 2 routing labeler. Phase 3 router will call the same functions in production.
"""
import logging
import os
import pickle
from typing import List, Dict, Any

logger = logging.getLogger("SparseRetriever")


def sparse_retrieve(
    query: str,
    project_id: str,
    data_root: str,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Returns top-K chunks for a project using BM25 keyword matching.
    
    Args:
        query: The raw query string.
        project_id: The project to search within.
        data_root: Path to the data directory containing bm25_indices/.
        top_k: Number of results to return.
    
    Returns:
        List of dicts with chunk_id and score, sorted descending by score.
    """
    bm25_path = os.path.join(data_root, "bm25_indices", f"{project_id}_bm25.pkl")
    if not os.path.exists(bm25_path):
        logger.warning(f"No BM25 index found for project '{project_id}' at {bm25_path}.")
        return []

    with open(bm25_path, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    chunk_ids = data["chunk_ids"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Pair up and sort
    ranked = sorted(
        zip(chunk_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return [{"chunk_id": cid, "score": float(score)} for cid, score in ranked]

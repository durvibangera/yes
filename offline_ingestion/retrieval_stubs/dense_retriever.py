"""
Retrieval stubs — dense vector search via Milvus (BGE-M3 embeddings).
Used by Phase 2 routing labeler. Phase 3 router will call the same functions in production.
"""
import logging
import re
from typing import List, Dict, Any

from pymilvus import connections, Collection, utility
from ingestion.config import settings

logger = logging.getLogger("DenseRetriever")


def _safe_collection_name(project_id: str) -> str:
    return "project_" + re.sub(r'[^a-zA-Z0-9]', '_', project_id)


def _ensure_connected():
    if not connections.has_connection("default"):
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=str(settings.MILVUS_PORT)
        )


def dense_retrieve(
    query_embedding: List[float],
    project_id: str,
    top_k: int = 10,
    model=None,
    expr: str = None
) -> List[Dict[str, Any]]:
    """
    Returns top-K chunks for a project using dense cosine similarity in Milvus.
    
    Args:
        query_embedding: Pre-computed embedding vector. If None, model must be provided with query_text.
        project_id: The project to search within.
        top_k: Number of results to return.
        model: Optional BGEM3FlagModel instance (used if query_embedding is None).
        expr: Optional Milvus boolean expression for filtering results (e.g. "chunk_id in [...]")
    
    Returns:
        List of dicts with chunk_id, file_name, file_path, subfolder_path, score.
    """
    _ensure_connected()
    collection_name = _safe_collection_name(project_id)

    if not utility.has_collection(collection_name):
        logger.warning(f"No Milvus collection found for project '{project_id}'.")
        return []

    collection = Collection(collection_name)
    collection.load()

    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["chunk_id", "file_name", "file_path", "subfolder_path"]
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "chunk_id": hit.entity.get("chunk_id"),
            "file_name": hit.entity.get("file_name"),
            "file_path": hit.entity.get("file_path"),
            "subfolder_path": hit.entity.get("subfolder_path"),
            "score": hit.score,
        })
    return hits

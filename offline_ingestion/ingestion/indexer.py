import pickle
import os
import sqlite3
import re
from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel

class Indexer:
    def __init__(self, milvus_host: str, milvus_port: int, model: BGEM3FlagModel):
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.model = model
        
        connections.connect(
            alias="default", 
            host=self.milvus_host, 
            port=str(self.milvus_port)
        )

    def init_milvus_collection(self, project_id: str) -> Collection:
        safe_project_id = re.sub(r'[^a-zA-Z0-9]', '_', project_id)
        collection_name = f"project_{safe_project_id}"
        
        if utility.has_collection(collection_name):
            return Collection(collection_name)
            
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
            FieldSchema(name="project_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="subfolder_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
        ]
        schema = CollectionSchema(fields, description=f"Collection for project {project_id}")
        collection = Collection(name=collection_name, schema=schema)
        
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        return collection

    def build_indices(self, project_id: str, chunks: List[Dict[str, Any]], data_root: str):
        if not chunks:
            return
            
        db_path = os.path.join(data_root, "metadata.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                project_id TEXT,
                file_name TEXT,
                file_path TEXT,
                subfolder_path TEXT,
                char_start INTEGER,
                char_end INTEGER
            )
        """)
        
        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]
        
        for c in chunks:
            cursor.execute("""
                INSERT OR IGNORE INTO chunks 
                (chunk_id, project_id, file_name, file_path, subfolder_path, char_start, char_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (c["chunk_id"], c["project_id"], c["file_name"], c["file_path"], c["subfolder_path"], c["char_start"], c["char_end"]))
        conn.commit()
        conn.close()

        tokenized_corpus = [doc.lower().split() for doc in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_dir = os.path.join(data_root, "bm25_indices")
        os.makedirs(bm25_dir, exist_ok=True)
        with open(os.path.join(bm25_dir, f"{project_id}_bm25.pkl"), "wb") as f:
            pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)
            
        collection = self.init_milvus_collection(project_id)
        
        # Checkpoint system for embeddings
        emb_ckpt_path = os.path.join(data_root, "checkpoints", f"{project_id}_embeddings.pkl")
        if os.path.exists(emb_ckpt_path):
            with open(emb_ckpt_path, "rb") as f:
                all_embeddings_list = pickle.load(f)
        else:
            all_embeddings_list = []
            
        start_idx = len(all_embeddings_list)
        chunk_size = 500
        
        import logging
        logger = logging.getLogger("Indexer")
        
        if start_idx < len(texts):
            logger.info(f"Resuming embeddings from chunk {start_idx}/{len(texts)}...")
            for i in range(start_idx, len(texts), chunk_size):
                batch_texts = texts[i:i + chunk_size]
                batch_embs = self.model.encode(batch_texts, batch_size=12, max_length=512)['dense_vecs']
                all_embeddings_list.extend(batch_embs.tolist())
                
                # Save checkpoint
                with open(emb_ckpt_path, "wb") as f:
                    pickle.dump(all_embeddings_list, f)
                logger.info(f"Saved embedding checkpoint for {len(all_embeddings_list)} chunks.")
                
        logger.info("All embeddings complete. Inserting into Milvus...")
        
        all_rows = list(zip(
            chunk_ids,
            [c["project_id"] for c in chunks],
            [c["file_name"] for c in chunks],
            [c["file_path"] for c in chunks],
            [c["subfolder_path"] for c in chunks],
            all_embeddings_list
        ))
        BATCH_SIZE = 500
        for i in range(0, len(all_rows), BATCH_SIZE):
            batch = all_rows[i:i + BATCH_SIZE]
            batch_entities = [list(col) for col in zip(*batch)]
            collection.insert(batch_entities)
        collection.flush()

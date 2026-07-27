import uuid
import logging
import os
import re
from typing import List, Dict, Any, Set
from FlagEmbedding import BGEM3FlagModel
from rapidfuzz import fuzz
import numpy as np

class EntityResolver:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def compute_similarity(self, emb1, emb2) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def _can_merge(self, text1: str, text2: str) -> bool:
        if text1.lower() == text2.lower():
            return True
            
        # 1. Prevent merging if they contain different numbers
        nums1 = set(re.findall(r'\d+', text1))
        nums2 = set(re.findall(r'\d+', text2))
        if nums1 != nums2:
            return False
            
        # 2. Prevent merging short strings unless they match alphanumerically
        # This prevents 'class P' merging with 'class A' or 'PCBl' with 'PBl'
        if len(text1) < 12 or len(text2) < 12:
            alnum1 = re.sub(r'[^a-z0-9]', '', text1.lower())
            alnum2 = re.sub(r'[^a-z0-9]', '', text2.lower())
            if alnum1 != alnum2:
                return False
                
        return True

    def resolve(self, entities: List[Dict[str, Any]], project_id: str, log_dir: str) -> List[Dict[str, Any]]:
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(f"merge_{project_id}")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        
        log_file = os.path.join(log_dir, f"merge_decisions_{project_id}.log")
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)

        surface_forms = {}
        for ent in entities:
            text = ent["entity_text"]
            if text not in surface_forms:
                surface_forms[text] = {
                    "types": set(),
                    "chunk_ids": set(),
                }
            surface_forms[text]["types"].add(ent["entity_type"])
            surface_forms[text]["chunk_ids"].add(ent["chunk_id"])

        unique_texts = list(surface_forms.keys())
        if not unique_texts:
            return []

        embeddings = self.model.encode(unique_texts, batch_size=12, max_length=512)['dense_vecs']
        
        clusters = [] 
        
        for i, text1 in enumerate(unique_texts):
            emb1 = embeddings[i]
            added = False
            for cluster in clusters:
                text2 = cluster[0]
                
                if not self._can_merge(text1, text2):
                    continue

                idx2 = unique_texts.index(text2)
                emb2 = embeddings[idx2]
                
                sim = self.compute_similarity(emb1, emb2)
                fuzz_ratio = fuzz.ratio(text1.lower(), text2.lower()) / 100.0
                
                if sim >= self.similarity_threshold:
                    cluster.append(text1)
                    added = True
                    logger.info(f"MERGED (dense): '{text1}' -> '{text2}' (Score: {sim:.3f})")
                    break
                elif fuzz_ratio >= self.similarity_threshold:
                    cluster.append(text1)
                    added = True
                    logger.info(f"MERGED (fuzz): '{text1}' -> '{text2}' (Score: {fuzz_ratio:.3f})")
                    break
            
            if not added:
                clusters.append([text1])

        resolved_entities = []
        for cluster in clusters:
            canonical = cluster[0] 
            canonical_types = set()
            chunk_ids = set()
            for text in cluster:
                canonical_types.update(surface_forms[text]["types"])
                chunk_ids.update(surface_forms[text]["chunk_ids"])
                
            resolved_entities.append({
                "entity_id": str(uuid.uuid4()),
                "entity_text": canonical,
                "entity_types": list(canonical_types),
                "aliases": cluster,
                "project_id": project_id,
                "chunk_ids": list(chunk_ids)
            })
            
        return resolved_entities

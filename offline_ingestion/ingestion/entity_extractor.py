import torch
from typing import List, Dict, Any
from gliner import GLiNER

class EntityExtractor:
    def __init__(self, entity_types: List[str], model_name: str = "urchade/gliner_medium-v2.1"):
        self.entity_types = entity_types
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GLiNER.from_pretrained(model_name).to(device)

    def extract(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = chunk["text"]
        
        entities = self.model.predict_entities(text, self.entity_types)
        
        extracted = []
        for ent in entities:
            extracted.append({
                "entity_text": ent["text"].strip(),
                "entity_type": ent["label"].upper(),
                "chunk_id": chunk["chunk_id"],
                "project_id": chunk["project_id"]
            })
            
        return extracted

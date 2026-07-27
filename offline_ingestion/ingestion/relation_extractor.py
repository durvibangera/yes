import json
import os
import logging
from typing import List, Dict, Any
from openai import OpenAI
from ingestion.config import settings

logger = logging.getLogger("RelationExtractor")

class RelationExtractor:
    def __init__(self):
        # We assume OPENAI_API_KEY is in the environment
        self.client = OpenAI()
        self.model = "gpt-4o-mini"
        
        self.system_prompt = """
You are an expert maritime engineering knowledge extractor.
Your task is to extract strict semantic relationships from the given technical text.
Extract relationships as triplets: (Head Entity, Relation, Tail Entity).

Allowed relation types (MUST use one of these exactly):
- "part_of"
- "requires"
- "subclass_of"
- "tested_by"
- "connected_to"
- "measured_in"
- "has_property"
- "defined_by"

Rules:
- Head and Tail entities must be highly specific technical terms found in or implied by the text (e.g., "waterjet propulsion", "Category A machinery space", "FRP hull").
- Do NOT extract conversational entities (e.g., "Harvard University").
- If no valid technical relations exist, return an empty array.
- Output ONLY valid JSON in this exact format:
[
  {"head": "entity1", "relation": "relation_type", "tail": "entity2"},
  {"head": "entity3", "relation": "relation_type", "tail": "entity4"}
]
"""

    def extract(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = chunk["text"]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            triplets = json.loads(content.strip())
            
            relations = []
            if isinstance(triplets, list):
                for triplet in triplets:
                    if "head" in triplet and "tail" in triplet and "relation" in triplet:
                        relations.append({
                            "head": str(triplet["head"]),
                            "relation": str(triplet["relation"]),
                            "tail": str(triplet["tail"]),
                            "chunk_id": chunk["chunk_id"],
                            "project_id": chunk["project_id"]
                        })
            return relations
            
        except Exception as e:
            error_msg = str(e)
            if "requests per day (RPD)" in error_msg and "rate_limit_exceeded" in error_msg:
                logger.critical(f"FATAL ERROR: Hit OpenAI daily RPD quota limit! Stopping pipeline to prevent data corruption. Error: {error_msg}")
                import sys
                sys.exit(1)
            logger.warning(f"Error extracting relations via OpenAI for chunk {chunk['chunk_id']}: {e}")
            return []

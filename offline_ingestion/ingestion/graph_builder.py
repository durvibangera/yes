from neo4j import GraphDatabase
from typing import List, Dict, Any
import time
import logging

class GraphBuilder:
    def __init__(self, uri: str, max_retries: int = 6, retry_delay: int = 10):
        self.driver = GraphDatabase.driver(uri, auth=("", ""))
        self._wait_for_connection(max_retries, retry_delay)
        
    def _wait_for_connection(self, max_retries: int, retry_delay: int):
        logger = logging.getLogger("GraphBuilder")
        for attempt in range(max_retries):
            try:
                self.driver.verify_connectivity()
                logger.info("Successfully connected to Neo4j.")
                
                # Create indices to speed up insertions drastically
                with self.driver.session() as session:
                    session.run("CREATE INDEX ON :Entity(entity_id);")
                    session.run("CREATE INDEX ON :Entity(project_id);")
                    session.run("CREATE INDEX ON :Chunk(chunk_id);")
                    session.run("CREATE INDEX ON :Chunk(project_id);")
                    session.run("CREATE INDEX ON :Document(file_path);")
                    session.run("CREATE INDEX ON :Document(project_id);")
                return
            except Exception as e:
                logger.warning(f"Neo4j connection attempt {attempt + 1} failed. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        logger.error("Failed to connect to Neo4j after maximum retries.")
        raise ConnectionError("Could not connect to Neo4j.")

    def close(self):
        self.driver.close()

    def build_graph(self, project_id: str, resolved_entities: List[Dict[str, Any]], relations: List[Dict[str, Any]], chunks: List[Dict[str, Any]]):
        BATCH_SIZE = 1000
        
        with self.driver.session() as session:
            # 1. Insert Chunks and Documents
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                session.run("""
                    UNWIND $batch AS chunk
                    MERGE (d:Document {file_path: chunk.file_path})
                    SET d.project_id = $project_id, d.file_name = chunk.file_name, d.subfolder_path = chunk.subfolder_path
                    MERGE (c:Chunk {chunk_id: chunk.chunk_id})
                    SET c.project_id = $project_id
                    MERGE (c)-[:PART_OF]->(d)
                """, project_id=project_id, batch=batch)

            # 2. Insert Entities and MENTIONS edges
            for i in range(0, len(resolved_entities), BATCH_SIZE):
                batch = resolved_entities[i:i + BATCH_SIZE]
                session.run("""
                    UNWIND $batch AS ent
                    MERGE (e:Entity {entity_id: ent.entity_id})
                    SET e.project_id = $project_id, e.entity_text = ent.entity_text, e.aliases = ent.aliases, e.entity_types = ent.entity_types
                """, project_id=project_id, batch=batch)
                
                # Mentions
                mentions = []
                for ent in batch:
                    for chunk_id in ent["chunk_ids"]:
                        mentions.append({"entity_id": ent["entity_id"], "chunk_id": chunk_id})
                
                if mentions:
                    session.run("""
                        UNWIND $mentions AS m
                        MATCH (c:Chunk {chunk_id: m.chunk_id})
                        MATCH (e:Entity {entity_id: m.entity_id})
                        MERGE (c)-[:MENTIONS]->(e)
                    """, project_id=project_id, mentions=mentions)

            # 3. Insert Relations
            alias_to_id = {}
            for ent in resolved_entities:
                for alias in ent["aliases"]:
                    alias_to_id[alias.lower()] = ent["entity_id"]
                    
            valid_relations = []
            for rel in relations:
                head_id = alias_to_id.get(rel["head"].lower())
                tail_id = alias_to_id.get(rel["tail"].lower())
                if head_id and tail_id:
                    rel_type = "".join(c for c in rel["relation"].upper().replace(" ", "_").replace("-", "_") if c.isalnum() or c == "_")
                    if rel_type:
                        valid_relations.append({
                            "head_id": head_id,
                            "tail_id": tail_id,
                            "rel_type": rel_type,
                            "chunk_id": rel["chunk_id"]
                        })
            
            # Group by rel_type because Cypher requires literal relationship types
            rels_by_type = {}
            for rel in valid_relations:
                t = rel["rel_type"]
                if t not in rels_by_type:
                    rels_by_type[t] = []
                rels_by_type[t].append(rel)
                
            for rel_type, rels in rels_by_type.items():
                for i in range(0, len(rels), BATCH_SIZE):
                    batch = rels[i:i + BATCH_SIZE]
                    session.run(f"""
                        UNWIND $batch AS rel
                        MATCH (h:Entity {{entity_id: rel.head_id}})
                        MATCH (t:Entity {{entity_id: rel.tail_id}})
                        MERGE (h)-[:{rel_type}]->(t)
                    """, project_id=project_id, batch=batch)

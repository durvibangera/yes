import os
import json
import pickle
import logging
from ingestion.config import Settings
from ingestion.parser import Parser
from ingestion.chunker import Chunker
from ingestion.entity_resolver import EntityResolver
from ingestion.graph_builder import GraphBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Restorer")

def restore_memgraph():
    settings = Settings()
    data_parser = Parser(settings.DATA_ROOT)
    chunker = Chunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    
    logger.info("Initializing EntityResolver (loading BGE-M3 model)...")
    entity_resolver = EntityResolver(settings.SIMILARITY_THRESHOLD)
    graph_builder = GraphBuilder(f"bolt://{settings.MEMGRAPH_HOST}:{settings.MEMGRAPH_PORT}")
    
    state_file = os.path.join(settings.DATA_ROOT, "state.json")
    if not os.path.exists(state_file):
        logger.error("No state.json found.")
        return
        
    with open(state_file, 'r') as f:
        completed_projects = json.load(f)
        if isinstance(completed_projects, dict) and "completed_projects" in completed_projects:
            completed_projects = completed_projects["completed_projects"]
        
    for project_id in completed_projects:
        logger.info(f"--- Restoring project {project_id} to Memgraph ---")
        
        files_data = data_parser.parse_project(project_id)
        if not files_data:
            continue
            
        all_chunks = []
        for fd in files_data:
            all_chunks.extend(chunker.chunk_file(fd))
            
        project_ckpt_dir = os.path.join(settings.DATA_ROOT, "checkpoints", project_id.replace(" ", "_").replace("/", "_"))
        entities_ckpt = os.path.join(project_ckpt_dir, "entities.pkl")
        relations_ckpt = os.path.join(project_ckpt_dir, "relations.pkl")
        
        if not os.path.exists(entities_ckpt) or not os.path.exists(relations_ckpt):
            logger.warning(f"Missing checkpoints for {project_id}, skipping.")
            continue
            
        with open(entities_ckpt, 'rb') as f:
            all_entities = pickle.load(f)['entities']
        with open(relations_ckpt, 'rb') as f:
            all_relations = pickle.load(f)['relations']
            
        log_dir = os.path.join(settings.DATA_ROOT, "logs")
        logger.info("Resolving entities...")
        resolved_entities = entity_resolver.resolve(all_entities, project_id, log_dir)
        
        logger.info("Writing to Memgraph...")
        graph_builder.build_graph(project_id, resolved_entities, all_relations, all_chunks)
        logger.info(f"Successfully restored {project_id}!")
        
    graph_builder.close()
    logger.info("Restoration complete.")

if __name__ == "__main__":
    restore_memgraph()

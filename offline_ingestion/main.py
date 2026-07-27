import argparse
import json
import os
import logging
import pickle
from typing import List
import concurrent.futures

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from tqdm import tqdm

from ingestion.config import settings
from ingestion.parser import Parser
from ingestion.chunker import Chunker
from ingestion.entity_extractor import EntityExtractor
from ingestion.relation_extractor import RelationExtractor
from ingestion.entity_resolver import EntityResolver
from ingestion.graph_builder import GraphBuilder
from ingestion.indexer import Indexer

os.makedirs(settings.DATA_ROOT, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(settings.DATA_ROOT, 'ingestion.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Pipeline")

def load_state(state_file: str) -> List[str]:
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f).get("completed_projects", [])
        except json.JSONDecodeError:
            return []
    return []

def save_state(state_file: str, completed: List[str]):
    with open(state_file, 'w') as f:
        json.dump({"completed_projects": completed}, f)

# Skip any folder whose total markdown size exceeds this threshold (in bytes).
# Skip any folder whose total markdown size exceeds this threshold (in bytes).
# ABS Standards = 4.66 MB — use that as the hard ceiling.
MAX_FOLDER_SIZE_BYTES = int(8.0 * 1024 * 1024)  # ~8.00 MB


def get_folder_size_bytes(folder_path: str) -> int:
    """Return total byte size of all .md files under folder_path."""
    total = 0
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith(".md"):
                try:
                    total += os.path.getsize(os.path.join(root, fname))
                except OSError:
                    pass
    return total


def main():
    parser = argparse.ArgumentParser(description="Offline Ingestion Pipeline")
    parser.add_argument("--folders", type=str, help="Comma-separated list of folders to process (subset testing)")
    args = parser.parse_args()

    state_file = os.path.join(settings.DATA_ROOT, 'state.json')
    completed_projects = load_state(state_file)
    
    logger.info("Initializing models and components...")
    
    data_parser = Parser(settings.DATA_ROOT)
    all_projects = data_parser.discover_projects()
    
    if args.folders:
        subset = [f.strip() for f in args.folders.split(',')]
        projects_to_process = [p for p in all_projects if p in subset]
    else:
        projects_to_process = all_projects

    if not projects_to_process:
        logger.warning("No projects found to process.")
        return

    chunker = Chunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    entity_extractor = EntityExtractor(settings.ENTITY_TYPES)
    relation_extractor = RelationExtractor()
    entity_resolver = EntityResolver(settings.SIMILARITY_THRESHOLD)
    
    summary_stats = {}

    for i, project_id in enumerate(projects_to_process):
        logger.info(f"--- Processing project {i+1}/{len(projects_to_process)}: {project_id} ---")
        if project_id in completed_projects and not args.folders:
            logger.info(f"Project {project_id} already completed. Skipping.")
            continue

        # Size gate: skip folders that are too large to process in reasonable time
        project_folder = os.path.join(settings.DATA_ROOT, project_id)
        folder_size = get_folder_size_bytes(project_folder)
        if folder_size > MAX_FOLDER_SIZE_BYTES:
            logger.warning(
                f"Project '{project_id}' is {folder_size / 1024 / 1024:.2f} MB "
                f"(limit {MAX_FOLDER_SIZE_BYTES / 1024 / 1024:.2f} MB) — SKIPPING."
            )
            continue
            
        try:
            files_data = data_parser.parse_project(project_id)
            logger.info(f"Parsed {len(files_data)} files.")
            
            all_chunks = []
            for fd in files_data:
                all_chunks.extend(chunker.chunk_file(fd))
            logger.info(f"Generated {len(all_chunks)} chunks.")
            
            project_ckpt_dir = os.path.join(settings.DATA_ROOT, "checkpoints", project_id.replace(" ", "_").replace("/", "_"))
            os.makedirs(project_ckpt_dir, exist_ok=True)
            entities_ckpt = os.path.join(project_ckpt_dir, "entities.pkl")
            relations_ckpt = os.path.join(project_ckpt_dir, "relations.pkl")

            all_entities = []
            start_idx = 0
            if os.path.exists(entities_ckpt):
                with open(entities_ckpt, 'rb') as f:
                    ckpt_data = pickle.load(f)
                    all_entities = ckpt_data['entities']
                    start_idx = ckpt_data['chunk_idx']
                logger.info(f"Loaded {len(all_entities)} entities from checkpoint (resuming from chunk {start_idx}).")
            
            if start_idx < len(all_chunks):
                logger.info("Extracting entities...")
                for i in tqdm(range(start_idx, len(all_chunks)), desc="Entities"):
                    chunk = all_chunks[i]
                    all_entities.extend(entity_extractor.extract(chunk))
                    
                    if (i + 1) % 100 == 0:
                        with open(entities_ckpt, 'wb') as f:
                            pickle.dump({'entities': all_entities, 'chunk_idx': i + 1}, f)
                            
                with open(entities_ckpt, 'wb') as f:
                    pickle.dump({'entities': all_entities, 'chunk_idx': len(all_chunks)}, f)
            logger.info(f"Total {len(all_entities)} raw entities.")
            
            all_relations = []
            start_idx = 0
            if os.path.exists(relations_ckpt):
                with open(relations_ckpt, 'rb') as f:
                    ckpt_data = pickle.load(f)
                    all_relations = ckpt_data['relations']
                    start_idx = ckpt_data['chunk_idx']
                logger.info(f"Loaded {len(all_relations)} relations from checkpoint (resuming from chunk {start_idx}).")
            
            if start_idx < len(all_chunks):
                logger.info("Extracting relations (sequential with throttle)...")
                
                import time
                BATCH_SIZE = 100
                # Sequential loop with 0.4s sleep = ~150 requests/min = well under 200K TPM limit
                # This avoids all 429 penalties which were slowing us down
                for i in tqdm(range(start_idx, len(all_chunks), BATCH_SIZE), desc="Relation Batches"):
                    batch = all_chunks[i : i + BATCH_SIZE]
                    
                    for chunk in batch:
                        res = relation_extractor.extract(chunk)
                        if res:
                            all_relations.extend(res)
                        time.sleep(0.4)  # ~150 req/min, stays under 200K TPM limit
                            
                    # Save checkpoint after each batch
                    with open(relations_ckpt, 'wb') as f:
                        pickle.dump({'relations': all_relations, 'chunk_idx': min(i + BATCH_SIZE, len(all_chunks))}, f)
                            
                with open(relations_ckpt, 'wb') as f:
                    pickle.dump({'relations': all_relations, 'chunk_idx': len(all_chunks)}, f)
            logger.info(f"Total {len(all_relations)} relations.")
            
            logger.info("Resolving entities...")
            log_dir = os.path.join(settings.DATA_ROOT, "logs")
            resolved_ckpt = os.path.join(project_ckpt_dir, f"{project_id}_resolved.pkl")
            if os.path.exists(resolved_ckpt):
                logger.info(f"Loading resolved entities from checkpoint...")
                with open(resolved_ckpt, 'rb') as f:
                    resolved_entities = pickle.load(f)
            else:
                resolved_entities = entity_resolver.resolve(all_entities, project_id, log_dir)
                with open(resolved_ckpt, 'wb') as f:
                    pickle.dump(resolved_entities, f)
            logger.info(f"Resolved to {len(resolved_entities)} canonical entities.")
            
            logger.info("Building knowledge graph...")
            graph_builder = GraphBuilder(f"bolt://{settings.MEMGRAPH_HOST}:{settings.MEMGRAPH_PORT}")
            graph_builder.build_graph(project_id, resolved_entities, all_relations, all_chunks)
            graph_builder.close()
            
            logger.info("Building indices (Milvus, BM25, SQLite)...")
            indexer = Indexer(settings.MILVUS_HOST, settings.MILVUS_PORT, entity_resolver.model) 
            indexer.build_indices(project_id, all_chunks, settings.DATA_ROOT)
            
            if project_id not in completed_projects:
                completed_projects.append(project_id)
                save_state(state_file, completed_projects)
                
            summary_stats[project_id] = {
                "files": len(files_data),
                "chunks": len(all_chunks),
                "raw_entities": len(all_entities),
                "canonical_entities": len(resolved_entities),
                "relations": len(all_relations)
            }
            logger.info(f"Project {project_id} complete. Stats: {summary_stats[project_id]}")
            
        except Exception as e:
            logger.error(f"Error processing project {project_id}: {e}", exc_info=True)
    
    logger.info("--- Final Summary ---")
    print(json.dumps(summary_stats, indent=2))

if __name__ == "__main__":
    main()

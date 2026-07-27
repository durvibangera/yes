import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingestion.config import settings

def analyze_project(project_id, file_path):
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    
    # We only care about multi_hop disjoint
    multi_hop_disjoint = [
        r for r in rows 
        if r["category"] == "multi_hop" and r["bm25_vs_graph_jaccard"] < 0.3
    ]
    
    n_disjoint = len(multi_hop_disjoint)
    if n_disjoint == 0:
        return None
    
    # Calculate avg cosine
    dense_vs_graph_hybrid_cosine = sum(r["dense_vs_graph_cosine"] for r in rows) / len(rows)
    dense_vs_graph_entity_cosine = sum(r.get("dense_vs_graph_entity_cosine", 0) for r in rows) / len(rows)
    
    q_vs_graph_hybrid_cosine = sum(r["q_vs_graph_cosine"] for r in rows) / len(rows)
    q_vs_graph_entity_cosine = sum(r.get("q_vs_graph_entity_cosine", 0) for r in rows) / len(rows)
    
    hybrid_acc = sum(1 for r in multi_hop_disjoint if r["graph_dense_hit"]) / n_disjoint
    entity_acc = sum(1 for r in multi_hop_disjoint if r.get("graph_entity_hit", False)) / n_disjoint
    
    entity_seed_miss_rate = sum(1 for r in rows if r.get("entity_seed_miss", False)) / len(rows)
    
    return {
        "n_disjoint": n_disjoint,
        "dense_vs_graph_hybrid_cosine": dense_vs_graph_hybrid_cosine,
        "dense_vs_graph_entity_cosine": dense_vs_graph_entity_cosine,
        "q_vs_graph_hybrid_cosine": q_vs_graph_hybrid_cosine,
        "q_vs_graph_entity_cosine": q_vs_graph_entity_cosine,
        "hybrid_acc": hybrid_acc,
        "entity_acc": entity_acc,
        "entity_seed_miss_rate": entity_seed_miss_rate
    }

def main():
    data_root = settings.DATA_ROOT
    
    asme_path = os.path.join(data_root, "outputs", "ASME_Subset_overlap_scores.jsonl")
    abs_path = os.path.join(data_root, "outputs", "ABS_Standards_overlap_scores.jsonl")
    
    # If the current run overwrote the generic one, check it too
    generic_path = os.path.join(data_root, "outputs", "overlap_scores.jsonl")
    if os.path.exists(generic_path) and not os.path.exists(asme_path):
        asme_path = generic_path
        
    res_asme = analyze_project("ASME_Subset", asme_path)
    res_abs = analyze_project("ABS Standards", abs_path)
    
    print("# Semantic Basin Check Results")
    
    for name, res in [("ASME Subset", res_asme), ("ABS Standards", res_abs)]:
        if res:
            print(f"\n## {name}")
            print(f"Total Disjoint Multi-Hop: {res['n_disjoint']}")
            print(f"Entity Seed Miss Rate: {res['entity_seed_miss_rate']:.2%}")
            print(f"- Dense vs Hybrid Graph Cosine: {res['dense_vs_graph_hybrid_cosine']:.4f}")
            print(f"- Dense vs Entity Graph Cosine: {res['dense_vs_graph_entity_cosine']:.4f}")
            print(f"- Q vs Hybrid Graph Cosine: {res['q_vs_graph_hybrid_cosine']:.4f}")
            print(f"- Q vs Entity Graph Cosine: {res['q_vs_graph_entity_cosine']:.4f}")
            print(f"- Hybrid Disjoint Accuracy: {res['hybrid_acc']:.2%}")
            print(f"- Entity Disjoint Accuracy: {res['entity_acc']:.2%}")

if __name__ == '__main__':
    main()

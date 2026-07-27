"""
Retrieval stubs — graph traversal via Memgraph/Neo4j.
Used by Phase 2 routing labeler. Phase 3 router will call the same functions in production.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Set, Callable
from neo4j import GraphDatabase

from ingestion.config import settings

logger = logging.getLogger("GraphRetriever")

# ── Stopwords for keyword fallback ────────────────────────────────────────────
_STOPWORDS: Set[str] = {
    "what", "which", "how", "when", "where", "who", "why",
    "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "not", "from", "with", "that", "this",
    "these", "those", "can", "does", "do", "did", "have",
    "has", "had", "will", "would", "shall", "should", "may",
    "might", "must", "between", "about", "after", "before",
    "during", "by", "than", "into", "through", "over", "under",
    "used", "using", "use", "any", "all", "both", "each",
    "given", "provide", "describe", "explain", "list", "define",
    "difference", "difference", "according", "required", "requirement",
    "requirements", "standard", "document", "section", "table",
}


def _extract_keyword_terms(text: str) -> List[str]:
    """
    Keyword fallback: extract unigrams and bigrams from a question string.

    Unigrams: alphabetic tokens ≥ 3 chars not in stopword list.
    Bigrams:  consecutive unigram pairs — captures multi-word entity names
              like "carbon equivalent", "tensile strength", "yield point".

    Returns lower-cased terms, deduplicated, order-preserving.
    """
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b', text)
    unigrams = []
    seen: Set[str] = set()
    for tok in tokens:
        lower = tok.lower()
        if lower not in _STOPWORDS and lower not in seen:
            seen.add(lower)
            unigrams.append(lower)

    # Add bigrams from the original (ordered) token stream
    all_tokens_lower = [t.lower() for t in tokens]
    bigrams = []
    for i in range(len(all_tokens_lower) - 1):
        a, b = all_tokens_lower[i], all_tokens_lower[i + 1]
        if a not in _STOPWORDS and b not in _STOPWORDS:
            bigram = f"{a} {b}"
            if bigram not in seen:
                seen.add(bigram)
                bigrams.append(bigram)

    return unigrams + bigrams


_CODE_PATTERN = re.compile(
    r'\b(?:UW|UG|UCS|UHA|UNS|SA|SB|SF|UB)\-\d+[a-zA-Z0-9\(\)\.]*\b'
    r'|\bN\d{5}\b'
    r'|\b[A-Z]{2,4}\s+\d{3,}\b',
    re.IGNORECASE
)

def _extract_technical_codes(text: str) -> List[str]:
    """Extract standard technical codes (e.g., UW-5, UNS N06230) that GLiNER misses."""
    matches = _CODE_PATTERN.findall(text)
    return list(set([m.strip() for m in matches]))


# ── Relationship type → semantic weight mapping ────────────────────────────────
# Higher weights = paths along these edges are preferred by the traversal scorer.
_REL_WEIGHTS: Dict[str, int] = {
    "REQUIRES":     5,
    "SUBCLASS_OF":  4,
    "DEFINED_BY":   4,
    "PART_OF":      3,
    "HAS_PROPERTY": 3,
    "TESTED_BY":    2,
    "PROHIBITED_BY": 2,
    "CONNECTED_TO": 1,
    "MEASURED_IN":  1,
}

# ── LLM relation phrase → Memgraph edge type mapping ──────────────────────────
RELATION_MAP: Dict[str, List[str]] = {
    "requires":   ["REQUIRES", "HAS_PROPERTY"],
    "require":    ["REQUIRES", "HAS_PROPERTY"],
    "need":       ["REQUIRES"],
    "needs":      ["REQUIRES"],
    "subclass":   ["SUBCLASS_OF"],
    "type of":    ["SUBCLASS_OF"],
    "kind of":    ["SUBCLASS_OF"],
    "part of":    ["PART_OF"],
    "contains":   ["PART_OF"],
    "component":  ["PART_OF"],
    "defined":    ["DEFINED_BY"],
    "definition": ["DEFINED_BY"],
    "measures":   ["MEASURED_IN"],
    "measured":   ["MEASURED_IN"],
    "unit":       ["MEASURED_IN"],
    "tested":     ["TESTED_BY"],
    "test":       ["TESTED_BY"],
    "prohibited": ["PROHIBITED_BY"],
    "forbidden":  ["PROHIBITED_BY"],
    "property":   ["HAS_PROPERTY"],
}

def _map_relation_phrases(phrases: List[str]) -> List[str]:
    """Map LLM-extracted relation phrases to Memgraph edge type strings."""
    edge_types: Set[str] = set()
    for phrase in phrases:
        phrase_lower = phrase.lower().strip()
        for key, types in RELATION_MAP.items():
            if key in phrase_lower:
                edge_types.update(types)
    return list(edge_types)


class GraphRetriever:
    def __init__(self):
        uri = f"bolt://{settings.MEMGRAPH_HOST}:{settings.MEMGRAPH_PORT}"
        self.driver = GraphDatabase.driver(uri, auth=("", ""))

    def close(self):
        self.driver.close()

    def get_entity_pairs_with_path(
        self,
        project_id: str,
        sample_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Query Memgraph for entity pairs (A, B) with path length 2-3.
        """
        query = """
        MATCH path = (a:Entity {project_id: $project_id})-[*2..3]-(b:Entity {project_id: $project_id})
        WHERE a <> b AND a.entity_id < b.entity_id
        WITH a, b, path LIMIT $sample_size
        RETURN a.entity_text AS entity_a, b.entity_text AS entity_b,
               a.entity_id AS id_a, b.entity_id AS id_b
        """
        results = []
        with self.driver.session() as session:
            try:
                records = session.run(query, project_id=project_id, sample_size=sample_size)
                for rec in records:
                    results.append({
                        "entity_a": rec["entity_a"],
                        "entity_b": rec["entity_b"],
                        "id_a": rec["id_a"],
                        "id_b": rec["id_b"],
                    })
            except Exception as e:
                logger.warning(f"Graph query failed for project '{project_id}': {e}")
        return results

    def get_chunk_ids_for_entity(self, entity_id: str, project_id: str) -> List[str]:
        """Get all chunk_ids that MENTION this entity."""
        query = """
        MATCH (c:Chunk {project_id: $project_id})-[:MENTIONS]->(e:Entity {entity_id: $entity_id})
        RETURN c.chunk_id AS chunk_id
        """
        chunk_ids = []
        with self.driver.session() as session:
            try:
                records = session.run(query, entity_id=entity_id, project_id=project_id)
                for rec in records:
                    chunk_ids.append(rec["chunk_id"])
            except Exception as e:
                logger.warning(f"Chunk lookup failed for entity '{entity_id}': {e}")
        return chunk_ids

    def graph_retrieve(
        self,
        entity_texts: List[str],
        project_id: str,
        top_k: int = 10,
        entity_extractor: Optional[Callable[[str], Any]] = None,
        seed_chunk_ids: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find chunks reachable via graph traversal for a question.

        Multi-Seed Traversal: When the LLM extracts multiple entity groups
        (e.g. for "what is the chain between X and Y?"), each group is traversed
        INDEPENDENTLY as its own graph seed, and results are unioned. This ensures
        that questions spanning multiple disconnected graph neighbourhoods always
        return chunks from ALL relevant parts of the graph, not just the first
        high-confidence entity.

        entity_extractor may return either:
          - a list of strings (legacy: just entity terms)
          - a dict {"entities": [[str, ...], ...], "relations": [str, ...]}
            (new: entity groups with synonyms + relation phrases)

        relation_types: optional list of Memgraph edge types to restrict
                        traversal to (e.g., ["REQUIRES", "PART_OF"]).
        """
        raw_text = " ".join(entity_texts)
        # entity_groups_for_multiseed: list of term-sets, one per entity group.
        # Each will be traversed independently as its own graph seed.
        entity_groups_for_multiseed: List[Set[str]] = []
        all_chunk_results = {}
        extracted_relation_types = list(relation_types or [])

        # ── Path 0: Regex Technical Codes (High Precision) ────────────────────
        tech_codes = _extract_technical_codes(raw_text)
        if tech_codes:
            logger.debug(f"[{project_id}] graph_retrieve (Regex): found {tech_codes}")
            # Technical codes form their own independent seed group
            entity_groups_for_multiseed.append(set(tech_codes))

        # ── Path 1: Entity extractor (GLiNER / LLM) ───────────────────────────
        if entity_extractor is not None:
            try:
                extraction = entity_extractor(raw_text)

                if isinstance(extraction, dict):
                    # New structured format: {"entities": [[...], ...], "relations": [...]}
                    entity_groups = extraction.get("entities", [])
                    relation_phrases = extraction.get("relations", [])

                    # ── MULTI-SEED: Each entity group becomes its own seed ──
                    # This is the core of multi-seed traversal: instead of merging
                    # all entities into one pool (which loses the signal that
                    # "Aluminum Welding" and "Passenger Surveys" are separate concepts),
                    # we give each group its own independent traversal.
                    for group in entity_groups:
                        if isinstance(group, list):
                            seed_terms = {g.lower() for g in group if g}
                        elif isinstance(group, str):
                            seed_terms = {group.lower()}
                        else:
                            continue
                        if seed_terms:
                            entity_groups_for_multiseed.append(seed_terms)

                    # Map relation phrases → edge types
                    mapped = _map_relation_phrases(relation_phrases)
                    extracted_relation_types = list(set(extracted_relation_types + mapped))

                    if entity_groups:
                        logger.debug(
                            f"[{project_id}] graph_retrieve (LLM multi-seed): "
                            f"{len(entity_groups_for_multiseed)} independent seeds, "
                            f"relation types={extracted_relation_types}"
                        )
                else:
                    # Legacy list format: treat as a single seed group
                    gliner_entities = extraction
                    if gliner_entities:
                        logger.debug(f"[{project_id}] graph_retrieve (LLM list): entities={gliner_entities}")
                        entity_groups_for_multiseed.append({e.lower() for e in gliner_entities if e})

            except Exception as e:
                logger.warning(f"[{project_id}] Entity extractor failed: {e}")

        # ── Path 2: Keyword fallback (ONLY if high-precision extractors failed) ─
        if not entity_groups_for_multiseed:
            terms = _extract_keyword_terms(raw_text)
            if terms:
                logger.debug(f"[{project_id}] graph_retrieve (keywords): terms={terms[:8]}...")
                # Keywords are a single fallback seed group
                entity_groups_for_multiseed.append(set(terms))

        # ── Primary Traversal: Run independently for each seed group ──────────
        # Each seed group gets its own weighted multi-hop traversal.
        # Results are unioned by taking the max score per chunk across all seeds.
        for seed_terms in entity_groups_for_multiseed:
            if not seed_terms:
                continue
            term_results = self._cypher_multihop_weighted(list(seed_terms), project_id, top_k)
            for r in term_results:
                all_chunk_results[r["chunk_id"]] = max(
                    all_chunk_results.get(r["chunk_id"], -999), r["score"]
                )

        # ── Secondary: Relation-guided traversal using all seeds combined ──────
        # We use a merged term pool for relation-guided pass since it focuses on
        # edge-type filtering, not neighbourhood separation.
        if extracted_relation_types and entity_groups_for_multiseed:
            all_terms_merged = list({t for group in entity_groups_for_multiseed for t in group})
            logger.debug(f"[{project_id}] graph_retrieve (relation-guided): edge types={extracted_relation_types}")
            rel_results = self._cypher_relation_guided(
                all_terms_merged, project_id, top_k, extracted_relation_types
            )
            for r in rel_results:
                all_chunk_results[r["chunk_id"]] = max(
                    all_chunk_results.get(r["chunk_id"], -999), r["score"] + 30.0
                )

        # ── Path 3: Hybrid Semantic-Graph Seeding (Chunk-to-Entity-to-Chunk) ──
        if seed_chunk_ids:
            logger.debug(f"[{project_id}] graph_retrieve (Hybrid Seeding): using {len(seed_chunk_ids)} seed chunks")
            chunk_results = self._cypher_chunk_multihop(seed_chunk_ids, project_id, top_k)
            for r in chunk_results:
                all_chunk_results[r["chunk_id"]] = max(
                    all_chunk_results.get(r["chunk_id"], -999), r["score"] + 50.0
                )

        if not all_chunk_results:
            logger.warning(f"[{project_id}] graph_retrieve: no results found from any seed or fallback.")
            return []

        # Sort and return top_k
        sorted_results = [{"chunk_id": cid, "score": score} for cid, score in all_chunk_results.items()]
        sorted_results.sort(key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]

    # ── Private Cypher helpers ────────────────────────────────────────────────

    def _cypher_multihop_weighted(
        self, terms: List[str], project_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Relationship-Aware Weighted Multi-hop Traversal.

        Finds seed entities via CONTAINS (including aliases), traverses 1-3 hops,
        and scores paths by BOTH the semantic strength of relationship types used
        AND the distance from the seed. Tighter degree cap (80) reduces noise
        from high-degree hub entities.
        """
        query = """
        UNWIND $terms AS term
        MATCH (seed:Entity {project_id: $project_id})
        WHERE toLower(seed.entity_text) CONTAINS toLower(term)
           OR any(alias IN coalesce(seed.aliases, []) WHERE toLower(alias) CONTAINS toLower(term))
        WITH seed, term

        // Tighter degree cap to avoid flooding from hub entities
        MATCH (seed)-[]-(nb)
        WITH seed, term, count(nb) AS degree
        WHERE degree <= 80

        // Traverse 1-3 hops (0 included for direct seed chunks)
        MATCH p = (seed)-[rels*0..3]-(node:Entity {project_id: $project_id})
        WHERE all(n IN nodes(p) WHERE n:Entity)
        WITH node, term, rels, min(size(nodes(p)) - 1) AS dist

        // Compute path quality score from relationship types used
        WITH node, term, dist,
             reduce(rel_score = 0, r IN rels |
               rel_score + CASE type(r)
                 WHEN 'REQUIRES'      THEN 5
                 WHEN 'SUBCLASS_OF'   THEN 4
                 WHEN 'DEFINED_BY'    THEN 4
                 WHEN 'PART_OF'       THEN 3
                 WHEN 'HAS_PROPERTY'  THEN 3
                 WHEN 'TESTED_BY'     THEN 2
                 WHEN 'PROHIBITED_BY' THEN 2
                 WHEN 'CONNECTED_TO'  THEN 1
                 WHEN 'MEASURED_IN'   THEN 1
                 ELSE 0
               END
             ) AS rel_score

        MATCH (c:Chunk {project_id: $project_id})-[:MENTIONS]->(node)
        WITH c.chunk_id AS chunk_id,
             count(DISTINCT term) AS term_count,
             min(dist) AS min_dist,
             max(rel_score) AS best_rel_score
        RETURN chunk_id,
               (term_count * 100.0) + best_rel_score - min_dist AS score
        ORDER BY score DESC
        LIMIT $top_k
        """
        results = []
        if not terms:
            return results
        try:
            with self.driver.session() as session:
                records = session.run(query, terms=terms, project_id=project_id, top_k=top_k)
                for rec in records:
                    results.append({"chunk_id": rec["chunk_id"], "score": rec["score"]})
        except Exception as e:
            logger.warning(f"[{project_id}] Graph Cypher weighted multihop error: {e}")
        return results

    def _cypher_relation_guided(
        self, terms: List[str], project_id: str, top_k: int, allowed_rel_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Relation-Guided Traversal.

        Restricts graph traversal to ONLY the specified relationship types.
        Used when the LLM extraction identifies a specific relation phrase
        (e.g., "requires" → only follow REQUIRES edges).
        This produces very focused, high-precision results.
        """
        if not allowed_rel_types:
            return []

        # Build Cypher with dynamic relationship type filter using CASE/WHERE on type(r)
        query = """
        UNWIND $terms AS term
        MATCH (seed:Entity {project_id: $project_id})
        WHERE toLower(seed.entity_text) CONTAINS toLower(term)
           OR any(alias IN coalesce(seed.aliases, []) WHERE toLower(alias) CONTAINS toLower(term))
        WITH seed, term

        MATCH (seed)-[]-(nb)
        WITH seed, term, count(nb) AS degree
        WHERE degree <= 80

        // Traverse using only the specified relationship types
        MATCH p = (seed)-[rels*1..3]-(node:Entity {project_id: $project_id})
        WHERE all(r IN rels WHERE type(r) IN $allowed_rel_types)
          AND all(n IN nodes(p) WHERE n:Entity)
        WITH node, term, min(size(nodes(p)) - 1) AS dist

        MATCH (c:Chunk {project_id: $project_id})-[:MENTIONS]->(node)
        WITH c.chunk_id AS chunk_id, count(DISTINCT term) AS term_count, min(dist) AS min_dist
        RETURN chunk_id,
               (term_count * 150.0) - min_dist AS score
        ORDER BY score DESC
        LIMIT $top_k
        """
        results = []
        try:
            with self.driver.session() as session:
                records = session.run(
                    query,
                    terms=terms,
                    project_id=project_id,
                    top_k=top_k,
                    allowed_rel_types=allowed_rel_types
                )
                for rec in records:
                    results.append({"chunk_id": rec["chunk_id"], "score": rec["score"]})
        except Exception as e:
            logger.warning(f"[{project_id}] Graph Cypher relation-guided error: {e}")
        return results

    def _cypher_chunk_multihop(
        self, seed_chunk_ids: List[str], project_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """
        1-Hop Traversal from Seed Chunks.
        Finds Entities mentioned in the seed chunks, traverses 1 hop to neighbor Entities,
        and returns all Chunks mentioning those neighborhood Entities.
        """
        query = """
        UNWIND $seed_chunk_ids AS seed_chunk_id
        MATCH (seed:Chunk {chunk_id: seed_chunk_id, project_id: $project_id})-[:MENTIONS]->(ent:Entity)
        WITH ent

        // Prevent generic, high-degree entities from flooding the search
        MATCH (ent)-[]-(neighbor)
        WITH ent, count(neighbor) AS degree
        WHERE degree <= 80

        MATCH p = (ent)-[*0..3]-(node:Entity {project_id: $project_id})
        WHERE all(n IN nodes(p) WHERE n:Entity)
        WITH node, min(size(nodes(p)) - 1) AS dist
        MATCH (c:Chunk {project_id: $project_id})-[:MENTIONS]->(node)
        WITH c.chunk_id AS chunk_id, count(DISTINCT node) AS node_count, min(dist) as min_dist
        // Score by how many neighborhood nodes connect to this chunk
        RETURN chunk_id AS chunk_id, (node_count * 100.0) - min_dist AS score
        ORDER BY score DESC
        LIMIT $top_k
        """
        results = []
        if not seed_chunk_ids:
            return results
        try:
            with self.driver.session() as session:
                records = session.run(query, seed_chunk_ids=seed_chunk_ids, project_id=project_id, top_k=top_k)
                for rec in records:
                    results.append({"chunk_id": rec["chunk_id"], "score": rec["score"]})
        except Exception as e:
            logger.warning(f"[{project_id}] Graph Cypher chunk multihop error: {e}")
        return results

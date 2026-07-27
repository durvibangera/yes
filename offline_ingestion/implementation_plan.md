# Enhanced Entity Extraction & Graph Traversal — Implementation Plan v2

## Problem Statement

The WP24 benchmark question exposed that **neither Vector nor Graph RAG** could retrieve the five required clauses (8.5, 9.3, 9.4, 15.3, and the 0.500 in thickness exception). Root cause analysis revealed:

1. **Entity Extraction Gap:** GLiNER does not recognize section/clause numbers (e.g., "15.3", "9.4") as entities. They exist only as raw text inside chunks, with no corresponding `:Entity` node in Memgraph.
2. **Relation Extraction Gap:** The GPT-4o-mini prompt is hardcoded to "maritime engineering" and uses only 8 generic relation types. It cannot express regulatory concepts like cross-references, exceptions, or applicability rules.
3. **Entity Resolution Gap:** Section references appear as "Section 15.3", "§15.3", "15.3", and "clause 15.3" — the cosine-similarity resolver treats these as separate entities instead of merging them.
4. **Traversal Depth Gap:** The current Cypher query uses `*1..3` hops. For regulatory chain traversal (`WP24 → 8.5 → 15.3 → thickness exception`), we need 4+ hops.

---

## Proposed Changes

### Component 1: Entity Extractor

#### [MODIFY] [entity_extractor.py](file:///d:/Durvi_project/offline_ingestion/ingestion/entity_extractor.py)

**Current state:** GLiNER only, 12 entity types, zero-shot. Cannot recognize section numbers.

**Changes:**
- Add a **regex-based supplementary extractor** that runs *after* GLiNER on every chunk. This captures patterns that GLiNER is blind to:
  - Section/Clause numbers: `\b(\d{1,3}\.\d{1,3}(?:\.\d{1,3})?)\b` adjacent to keywords like "Section", "Clause", "§", or at the start of a line
  - Standard designations: `SA-\d+`, `A960/A960M`, `AWS A5.\d+`
  - Grade/WP codes: `WP\d+[A-Z]?`, `Grade [A-Z0-9]+`
- Tag these regex extractions with entity types `SECTION_CLAUSE`, `STANDARD_ID`, and `GRADE` respectively
- GLiNER continues to run first for all other entity types — this is purely additive, no existing extractions are removed

---

### Component 2: Ingestion Config

#### [MODIFY] [config.py](file:///d:/Durvi_project/offline_ingestion/ingestion/config.py)

**Changes:**
- Add `SECTION_CLAUSE` to the `ENTITY_TYPES` list (bringing it to 13 types)
- Add a new config field `MAX_HOP_DEPTH: int = 4` for use by the graph retriever
- Add new relation types to a `RELATION_TYPES` list for reference:
  ```
  "cross_references", "applies_to", "exempts", "governs",
  "part_of", "requires", "subclass_of", "tested_by",
  "connected_to", "measured_in", "has_property", "defined_by"
  ```

---

### Component 3: Relation Extractor

#### [MODIFY] [relation_extractor.py](file:///d:/Durvi_project/offline_ingestion/ingestion/relation_extractor.py)

**Current state:** Hardcoded "maritime engineering knowledge extractor" prompt with only 8 relation types.

**Changes:**
- Replace the system prompt with a **domain-adaptive** version. Instead of "maritime engineering", the prompt will say:
  > "You are an expert technical knowledge extractor for engineering regulatory standards."
- Expand the allowed relation types from 8 to 12:
  | New Relation | Purpose | Example |
  |---|---|---|
  | `cross_references` | Links two sections/clauses | 15.3 → references → 8.5 |
  | `applies_to` | Links a rule to a material/grade | 9.3 → applies_to → cold-formed fittings |
  | `exempts` | Links a rule to its exception condition | 15.3 → exempts → thickness ≤ 0.500 in |
  | `governs` | Links a base specification to materials | A960/A960M → governs → WP24 |
- Add an explicit instruction in the prompt: *"Pay special attention to cross-references between section numbers (e.g., 'shall meet the requirements of 8.5'), thickness/size exceptions, and applicability clauses."*

---

### Component 4: Entity Resolver

#### [MODIFY] [entity_resolver.py](file:///d:/Durvi_project/offline_ingestion/ingestion/entity_resolver.py)

**Current state:** Uses cosine similarity + fuzzy string matching. Has a `_can_merge()` guard that prevents merging entities with different numbers — this is good for materials but **blocks** section reference merging (e.g., "Section 15.3" and "15.3" have different digit sets because "Section" adds no digits but the string lengths differ).

**Changes:**
- Add a **pre-processing normalization step** specifically for `SECTION_CLAUSE` entities before the main clustering loop:
  1. Strip prefixes: "Section ", "Clause ", "§", "section ", "clause "
  2. Normalize to just the number: "15.3"
  3. All variants pointing to the same normalized number get force-merged into one canonical entity before embedding similarity even runs
- This is a fast O(n) pass that runs before the expensive O(n²) embedding loop, so it has zero performance cost

---

### Component 5: Graph Retriever (Traversal)

#### [MODIFY] [graph_retriever.py](file:///d:/Durvi_project/offline_ingestion/retrieval_stubs/graph_retriever.py)

**Current state:** Cypher traversal uses `*1..3` hops. The `_REL_WEIGHTS` dictionary has 8 edge types. The `RELATION_MAP` has 13 phrase mappings.

**Changes:**
1. **Increase hop depth** from `*1..3` to `*1..4` in both Cypher queries:
   - Line 414: `(seed)-[rels*1..3]-(node)` → `(seed)-[rels*1..4]-(node)`
   - Line 460: `(ent)-[*0..3]-(node)` → `(ent)-[*0..4]-(node)`
2. **Add new relation types** to `_REL_WEIGHTS`:
   ```python
   "CROSS_REFERENCES": 5,
   "APPLIES_TO":       4,
   "EXEMPTS":          5,
   "GOVERNS":          4,
   ```
3. **Add new phrase mappings** to `RELATION_MAP`:
   ```python
   "references":  ["CROSS_REFERENCES"],
   "applies":     ["APPLIES_TO"],
   "applicable":  ["APPLIES_TO"],
   "exempts":     ["EXEMPTS"],
   "exception":   ["EXEMPTS"],
   "waived":      ["EXEMPTS"],
   "governs":     ["GOVERNS"],
   "governed":    ["GOVERNS"],
   ```
4. **CPU Protection:** The existing `WHERE all(n IN nodes(p) WHERE n:Entity)` constraint stays in place. Combined with the existing `degree <= 80` guard, the 4-hop depth should remain safe. We will validate this during testing.

---

### Component 6: Re-Ingestion Strategy

> [!TIP]
> Since we are using a local LLM on a powerful PC, we are no longer constrained by the ~$30 OpenAI API cost. We can optimize for **maximum coverage and code simplicity** instead of incremental patching.

**Strategy: Full Local Re-Ingestion**

Rather than building complex merging logic for a hot-patch, we will:

1. **Wipe existing data** — clear the Memgraph database and Milvus collections for a clean slate.
2. **Delete Checkpoints** — remove the old `.pkl` checkpoints so the pipeline starts fresh.
3. **Run Full Pipeline** — process all 20,328 chunks through the updated GLiNER + Regex extractor, and pass 100% of the chunks through the local LLM relation extractor using the new domain-aware prompt.
4. **Benefit:** This guarantees we don't miss any obscure cross-references that a regex-filtered hot-patch might have skipped, giving us the highest possible graph quality.

**Configuration Update:**
- In `config.py`, we will ensure `LLM_BACKEND` is set to point to the local model (e.g., Ollama/vLLM) and tune the concurrent request limit based on the new PC's VRAM.

---

## Verification Plan

### Automated Tests
1. **Regex Extractor Unit Test:** Feed chunk `ASME_2019_ASME_II_PART_A1__2019__md_4145` into the new regex extractor and verify it outputs entities: `["15.3", "8.5", "WP91", "WP24", "A960/A960M"]`
2. **Entity Resolver Test:** Feed `["Section 15.3", "§15.3", "15.3", "clause 15.3"]` into the updated resolver and verify they collapse into 1 canonical entity
3. **Graph Traversal Test:** Run the WP24 benchmark question against the updated Memgraph and verify that chunks `4130` (Clause 8.5), `4134` (Clause 9.3), `4135` (Clause 9.4), `4145` (Clause 15.3), and `4146` (thickness exception) are all retrieved in the top 10
4. **CPU Regression Test:** Run 10 benchmark questions and verify that no query exceeds 10 seconds (guarding against the 4-hop explosion)

### Manual Verification
- Visually inspect the Memgraph graph in Memgraph Lab to confirm that `WP24` now has direct edges to `15.3`, `8.5`, `9.3`, and `9.4`
- Re-run the full diagnostic pipeline (`main_diagnostic.py`) on ASME 2019 and compare the new Jaccard/Cosine scores against the previous baseline

---

## Open Questions

> [!IMPORTANT]
> **Local Model selection:** Which local model will you be running (e.g., Llama-3-8B, Qwen, Mistral)? We need to make sure the prompt engineering matches the model's instruction-following capabilities.

> [!IMPORTANT]
> **Hop depth safety:** Increasing from 3 to 4 hops could cause CPU issues on dense graph regions. Should we add a per-query timeout (e.g., 8 seconds) as a hard safety net, or rely on the existing `degree <= 80` guard?

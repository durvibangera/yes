# GraphRAG Engineering Standards Pipeline — Complete Project History

*Last updated: 2026-07-17 · Status: Active Research*

---

## Table of Contents
1. [What We Were Building and Why](#1-what-we-were-building-and-why)
2. [The Technology Stack](#2-the-technology-stack)
3. [Phase 1 — Offline Ingestion](#3-phase-1--offline-ingestion)
4. [Phase 2 — Synthetic Ground Truth Generation](#4-phase-2--synthetic-ground-truth-generation)
5. [The Major Pivot: Shifting to Traditional vs Graph RAG](#5-the-major-pivot-shifting-to-traditional-vs-graph-rag)
6. [The ASME_Subset Strategy](#6-the-asme_subset-strategy)
7. [Bugs, Failures, and Key Pivots](#7-bugs-failures-and-key-pivots)
8. [The Benchmarking Experiments (Traditional vs Graph RAG)](#8-the-benchmarking-experiments-traditional-vs-graph-rag)
9. [The Diagnostic Pipeline — Chunk Comparison Analysis](#9-the-diagnostic-pipeline--chunk-comparison-analysis)
10. [Current Status and Next Experiment](#10-current-status-and-next-experiment)

---

## 1. What We Were Building and Why

The organisation has a library of ~45 folders of engineering standards (ASME, AWS, ISO, ANSI, ABS, DIN, British Standards, and more) totalling hundreds of gigabytes of PDF content. Engineers need to query this library in natural language.

The hard part: **these standards are not independent documents.** They form a dense web of cross-references. A question like *"What is the acceptable weld defect tolerance for high-pressure boilers?"* might require chunks from three different documents across three folders. Standard vector or keyword search has no way to traverse these relationships.

### The Original Goal

The project originally aimed to build a pipeline that (1) ingested all standards into a unified knowledge graph + vector index, and (2) automatically learned which retrieval strategy to use for any query (a "Retriever Router"). 

As detailed below, the project underwent a **massive pivot** once we realised Graph RAG was severely underperforming.

---

## 2. The Technology Stack

| Component | Tool | Why |
|---|---|---|
| **Knowledge Graph** | **Memgraph** (Docker) | Cypher-compatible like Neo4j, lightweight, no licensing issues |
| **Vector Database** | **Milvus** (Docker) | Best-in-class metadata + vector hybrid query support |
| **BM25 Full-Text** | **Whoosh** (file-based) | Pure Python, zero infra, fast keyword matching |
| **Embedding Model** | **BAAI/bge-m3** | High-dimensional, local, runs without API calls |
| **NER (Entity Extraction)** | **GLiNER** (`urchade/gliner_medium-v2.1`) | Zero-shot NER trained on technical entity types |
| **Relation Extraction** | **REBEL (Babelscape)** | Relation triplet extraction from text |
| **LLM (Generation)** | **Qwen (via Ollama)** | Local, compliant, good at technical context |
| **Infrastructure** | **Docker Compose** | Memgraph + Milvus containerised, data persists across restarts |

> **One graph, not many.** Memgraph runs a single unified graph. Projects are isolated by `project_id` properties on nodes and edges, not by separate databases or collections.

---

## 3. Phase 1 — Offline Ingestion

**Goal:** Convert raw markdown files into three parallel indexes: a knowledge graph (Memgraph), a vector index (Milvus), and a BM25 index (Whoosh).

### The Ingestion Pipeline

```
PDF/MD Files → Parser → Chunker → EntityExtractor → RelationExtractor
                                         ↓                   ↓
                                  EntityResolver → GraphBuilder (Memgraph)
                                         ↓
                                     Indexer (Milvus + BM25 + SQLite)
```

**Key config:**
```ini
CHUNK_SIZE = 400 tokens
CHUNK_OVERLAP = 60 tokens
SIMILARITY_THRESHOLD = 0.92  # cosine threshold for entity deduplication
ENTITY_TYPES = ['MATERIAL', 'STANDARD_ID', 'PROCESS', 'SPECIFICATION', 'TOLERANCE', 'GRADE']
```

### The Folder Size Problem (and What We Decided)

The full library has ~45 folders. Many are enormous:

| Folder | Size | Status |
|---|---|---|
| ASME (full) | 88.5 MB | ❌ Skipped — ~60 hours of LLM inference |
| IS Standards | 64.4 MB | ❌ Skipped |
| AWS | 14.1 MB | ❌ Holdout |
| **ABS Standards** | 4.7 MB | ✅ **Completed** |
| **ASME_Subset** | 5.1 MB | ✅ **Completed** |
| ANSI | 2.5 MB | ✅ Completed |
| AMS | 0.4 MB | ✅ Completed |
| AISI | ~0 MB | ✅ Completed |

**Root cause of the size limit:** The local Qwen LLM takes ~1.5 seconds per chunk for relation extraction. An 88 MB folder has ~150,000 chunks — roughly 60 hours of continuous inference. We set a hard `MAX_FOLDER_SIZE_BYTES = 8 MB` cutoff.

**What successfully completed Phase 1 & 2:** 9 folders, producing 322 empirically-labelled questions total.

---

## 4. Phase 2 — Synthetic Ground Truth Generation

**Goal:** Automatically generate a labelled Q&A dataset to evaluate the different retrieval strategies.

### The Four Question Generators

| Generator | Method | Budget per folder |
|---|---|---|
| **ExactIDGenerator** | Scans chunks for standard ID patterns (e.g. `[A-Z]{1,6}[-_]?\d{2,8}`), generates "What is [ID]?" questions | 15 |
| **SingleHopGenerator** | Sends a chunk to Qwen: "Generate a factual question that requires reading this specific chunk" | 15 |
| **MetadataGenerator** | Generates questions about file structure, section numbers, document titles | 5 |
| **MultiHopGenerator** | Queries Memgraph for entity pairs with path length ≥ 2, asks Qwen to write a question requiring both chunks | 8 |
| **NullGenerator** | Generates unanswerable questions (negative examples) | 5 |

### The Empirical Labeler

After generating a question, instead of assuming a label, we run a **strategy ladder**:
```
BM25 → Dense → Metadata Dense → Graph Dense → escalation_required
```
The label is whichever strategy first successfully retrieves the required chunk IDs.

### Label Distribution (322 questions, 9 folders)

| Label | Count | % | Meaning |
|---|---|---|---|
| `metadata_dense` | 117 | 36.3% | Vector + filter search needed |
| `bm25` | 109 | 33.9% | Exact keyword match suffices |
| `n/a_for_null` | 45 | 14.0% | Unanswerable (null examples) |
| `escalation_required` | 32 | 9.9% | Nothing worked |
| `dense` | 18 | 5.6% | Pure vector search suffices |
| **`graph_dense`** | **1** | **0.3%** | **Genuine multi-hop graph retrieval** |

---

## 5. The Major Pivot: Shifting to Traditional vs Graph RAG

The label distribution above was the catalyst for a massive project pivot.

**The Realization:** Out of 322 automatically generated questions, **only 1** actually required Graph RAG to solve. In 99.7% of cases, Traditional RAG (BM25 or Dense) or Metadata filtering was sufficient. 

**The Pivot:** We originally planned to build a machine learning "Router" (Phase 3) that would classify incoming queries and route them to the correct strategy. We spent time scaffolding this out, but quickly realized it was useless. A classifier trained on data where 99.7% of the answers are "not graph" will simply learn to never predict graph traversal. 

We abandoned the router entirely. The project ceased to be about *building a routing pipeline* and transformed into an intensive research investigation: **Why is Graph RAG performing so poorly compared to Traditional RAG in this domain?** 

We shifted focus entirely to head-to-head benchmarking and deep diagnostics to answer this question.

---

## 6. The ASME_Subset Strategy

To ensure our findings weren't a fluke based on simple datasets, we needed genuine multi-hop cross-reference data. We carved out a targeted subset of the most heavily cross-referenced ASME files:
- `ASME_Section_V.md` (Nondestructive Examination)
- `ASME_Section_VIII_Div1.md` (Rules for Construction of Pressure Vessels)

**Size:** 5.1 MB (comfortably under the 8 MB cutoff)
**Stats after Phase 1:** 17,412 chunks · 48,204 raw entities · 14,981 canonical entities · 18,733 relations

*(Note: In a session on 2026-07-09, `ASME_Subset` was deleted from Memgraph to free RAM. Since the NLP checkpoints were preserved on disk, restoring it on 2026-07-17 took only 28 seconds.)*

---

## 7. Bugs, Failures, and Key Pivots

### Bug 1: Synthetic Augmentation (Critical, Fixed)

**What happened:** When we only had 1 `graph_dense` example, we panicked and tried to generate 50 paraphrases of it using Mistral. We hard-labelled all of them `graph_dense` by assumption, which broke our empirical grounding rule.
**Fixes:** Deleted the augmented dataset completely to preserve data integrity.

### Bug 2: Labeler Assumption (Fixed)

Early versions of the labeler tested graph traversal *first* for questions generated by `MultiHopGenerator`, reasoning "a multi-hop question must need graph traversal." This was wrong — many such questions are answerable by BM25 alone. The labeler was rewritten to always start at BM25.

### Bug 3: `exact_id` Bug (Fixed)

`ExactIDGenerator` was emitting `required_chunk_ids: [every chunk in the file]` instead of `[]`. This caused `exact_id` questions to falsely fail all retrievers. Fixed the generator.

---

## 8. The Benchmarking Experiments (Traditional vs Graph RAG)

We ran a head-to-head benchmark comparing Graph RAG against BM25 and Dense retrieval on entity-obscured questions (questions where the LLM was forced to describe the target entity rather than name it, to handicap BM25).

### ASME_Subset (52 Questions)

| Strategy | Hits | Accuracy |
|---|---|---|
| BM25 | 24 / 52 | 46.1% |
| Dense | 20 / 52 | 38.4% |
| Graph RAG (Hybrid Seeded) | 19 / 52 | 36.5% |

### ABS Standards (56 Questions)

| Strategy | Hits | Accuracy |
|---|---|---|
| BM25 | 23 / 56 | 41.1% |
| Graph RAG (Hybrid Seeded) | 15 / 56 | 26.8% |
| Dense | 14 / 56 | 25.0% |

### The Key Finding from Benchmarking

We attempted to defeat BM25 by obscuring entity names. 

**Instead:**
1. BM25 dominated — when users describe an entity's properties, BM25 finds those descriptive keywords directly.
2. Standard Graph RAG (NER-seeded) was brittle — GLiNER failed to extract entities from descriptions, so the traversal had no starting node.
3. We upgraded to **Hybrid Semantic-Graph Seeding** (using Dense's top-3 chunks as the traversal seed). Graph RAG success rate nearly doubled — **but its successes completely overlapped with Dense**. Unique wins remained at ~1 per dataset.

---

## 9. The Diagnostic Pipeline — Chunk Comparison Analysis

To get quantitative proof of *why* the successes overlapped, we built a standalone diagnostic suite.

### What It Does

For every question, it computes **Jaccard similarity** (exact ID overlap) and **Centroid Cosine Similarity** (semantic overlap) between the chunk sets retrieved by BM25, Dense, and Graph. It then buckets questions by overlap and measures accuracy within those buckets.

### Diagnostic Results — ABS Standards & ASME_Subset

**The Semantic Basin Finding:**
- Dense vs Graph Cosine Similarity: **0.95–0.99** on nearly every question
- BM25 vs Graph Cosine Similarity: **0.80–0.92** 
- BM25 vs Graph Jaccard: **< 0.10** (almost completely different chunk IDs)

**Bucket Analysis (multi_hop, disjoint bucket):**

| Strategy | ABS Standards | ASME_Subset |
|---|---|---|
| **BM25** | **45.45%** | **44.12%** |
| Graph Dense | 21.21% | 26.47% |
| Dense | 18.18% | 23.53% |

### Interpretation

> [!IMPORTANT]
> **The Semantic Basin:** Graph RAG (Hybrid Seeded) is crawling to neighboring graph nodes that say the exact same thing in slightly different words. The high Cosine Similarity (0.95–0.99) between Dense and Graph proves both systems are trapped in the same semantic neighborhood. 

> [!NOTE]
> **Why BM25 Dominates:** Engineering standards rely heavily on exact lexical identifiers (section numbers, table references). When BM25 and Graph retrieve entirely different chunks, BM25 wins 2:1 because its keyword matching anchors directly to these rigid identifiers. Dense and Graph drift into "same vibe, wrong specifics" territory.

---

## 10. Current Status and Next Experiment

### Infrastructure Status

| Component | Status |
|---|---|
| Docker (Memgraph + Milvus) | ✅ Running |
| Phase 1 pipeline | ✅ Complete |
| Phase 2 generation + labeling | ✅ Complete |
| Diagnostic pipeline | ✅ Complete, run on ABS Standards + ASME_Subset |

### The Final Experiment: Entity-Seeded Graph Traversal

> [!WARNING]
> **The Methodological Confound:** The Dense-vs-Graph cosine similarity (~0.95–0.99) may be a **tautology** given the seeding method. If Graph's traversal starts at Dense's top-3 chunks, of course it ends up in Dense's semantic neighborhood. 

**Goal:** Is the "Semantic Basin" finding a seeding artifact, or a real property of graph retrieval in this domain?

**Method:**
We built a 5th experimental arm: `graph_dense_entity_seeded`. This seeds traversal **purely from GLiNER entities extracted from the query text**, with no dense retrieval involved in seeding. We kept everything else (traversal depth, max_chunks=500, reranking function) constant.

**The Results (from the Disjoint Bucket):**

| Metric | ASME_Subset (Hybrid Seeding) | ASME_Subset (Pure Entity) | ABS Standards (Pure Entity) |
|---|---|---|---|
| **Cosine vs Dense** | 0.9293 | **0.7959** | 0.0000 |
| **Cosine vs Question** | 0.7371 | **0.6458** | 0.0000 |
| **Disjoint Accuracy** | 23.53% | **20.59%** | **0.00%** |
| **Entity Seed Miss Rate**| N/A | **8.16%** | **100.00%** |

### The Final Verdict on Graph RAG

1. **The Confound Was Real:** Removing the Dense seed successfully allowed Graph RAG to break out of the Dense Semantic Basin (cosine overlap dropped from 0.93 to 0.79). 
2. **The Brittleness is Fatal:** On the ABS Standards dataset, the pure entity-seeded approach completely collapsed. GLiNER failed to extract a valid starting entity from **100% of the natural language questions**. With no starting node, the graph traversal returned zero chunks for every single question.
3. **The Core Conclusion Holds:** For the ASME dataset (where NER actually worked), breaking out of the dense basin didn't help. The accuracy of Graph RAG *still dropped* compared to the hybrid method, and remains less than half as effective as standard BM25 keyword matching for multi-hop questions.

**Conclusion:** Graph RAG's massive architectural complexity does not yield unique value for engineering standards. When seeded purely with entities, it is fatally brittle. When seeded with dense embeddings to fix the brittleness, it becomes entirely semantically redundant with standard vector search, and still loses to BM25 on exact multi-hop identifiers. 

Traditional RAG (BM25 + Dense) remains conclusively the better, cheaper, and more robust system for this domain.

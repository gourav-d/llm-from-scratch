# Module 10.8 -- Semantic Search Systems

## Build a Google-Style Search Engine from Scratch

---

## What Is This Module About?

Have you ever wondered how Google finds pages that MEAN what you searched for,
even if they don't contain your exact words?

You search "best way to go fast on two wheels" and Google returns articles
about cycling, motorcycles, and speedboats -- because it understands MEANING,
not just keywords.

That is semantic search. This module teaches you to build it from scratch.

---

## Why Does This Matter? Real-World Examples

| Product          | Where Semantic Search Is Used                          |
|------------------|--------------------------------------------------------|
| Google Search    | Understands query meaning, not just keywords           |
| GitHub Copilot   | Finds relevant code snippets by meaning                |
| Elasticsearch 8  | "kNN search" feature -- vector + keyword hybrid        |
| Notion AI        | "Find similar documents" across your workspace         |
| Stack Overflow   | "Similar questions" sidebar                            |
| LinkedIn Jobs    | Match job descriptions to resume meaning               |

---

## What You Will Learn

- **Lesson 1**: Why keyword search fails and how semantic search fixes it
- **Lesson 2**: Bi-encoders -- the engine behind fast semantic search
- **Lesson 3**: Cross-encoders -- the engine behind accurate re-ranking
- **Lesson 4**: FAISS and Approximate Nearest Neighbor (ANN) search
- **Lesson 5**: Hybrid search -- combining BM25 keywords + semantic vectors
- **Lesson 6**: Production pipeline -- retrieve, then re-rank, end to end

---

## Prerequisites

Before starting this module you should have completed:

- **Module 10**: Vector Databases with ChromaDB
  - You know what embeddings are
  - You know what cosine similarity is
  - You have run vector searches before

- **Module 10.5**: BM25 / TF-IDF (keyword scoring)
  - You know how BM25 scores documents
  - You understand term frequency and inverse document frequency

If you have not done those modules, go back and complete them first.
This module builds directly on that knowledge.

---

## Module Structure

```
10.8_semantic_search/
|
+-- README.md               <-- You are here
+-- GETTING_STARTED.md      <-- Install instructions
+-- requirements.txt        <-- Python packages needed
|
+-- lessons/                <-- Theory + diagrams + quizzes
|   +-- 01_keyword_vs_semantic.md
|   +-- 02_bi_encoder.md
|   +-- 03_cross_encoder_reranking.md
|   +-- 04_faiss_and_ann.md
|   +-- 05_hybrid_search.md
|   +-- 06_retrieve_rerank_pipeline.md
|
+-- examples/               <-- Working code you can run
|   +-- example_01_keyword_vs_semantic.py
|   +-- example_02_bi_encoder_search.py
|   +-- example_03_cross_encoder_rerank.py
|   +-- example_04_faiss_index.py
|   +-- example_05_hybrid_search.py
|   +-- example_06_full_pipeline.py
|
+-- exercises/              <-- Fill-in-the-blank coding tasks
|   +-- exercise_01_encode_and_search.py
|   +-- exercise_02_reranker.py
|   +-- exercise_03_faiss_basics.py
|   +-- exercise_04_hybrid_search.py
|   +-- exercise_05_full_pipeline.py
|
+-- projects/
    +-- mini_search_engine/  <-- Full working search engine project
        +-- README.md
        +-- main.py
        +-- indexer.py
        +-- searcher.py
        +-- reranker.py
```

---

## Learning Path Flowchart

```
START
  |
  v
[Lesson 1] Keyword vs Semantic
  - WHY keyword fails
  - HOW semantic fixes it
  |
  v
[Lesson 2] Bi-Encoder
  - Encode query + docs separately
  - Fast retrieval (pre-compute docs offline)
  |
  v
[Lesson 3] Cross-Encoder Re-ranking
  - Encode query + doc TOGETHER
  - Accurate but slow -- use after bi-encoder
  |
  v
[Lesson 4] FAISS and ANN
  - Handle millions of vectors efficiently
  - HNSW graph structure
  |
  v
[Lesson 5] Hybrid Search
  - BM25 + Semantic = best of both worlds
  - Reciprocal Rank Fusion (RRF)
  |
  v
[Lesson 6] Full Pipeline
  - End-to-end production system
  - Metrics: NDCG, MRR, Precision@K
  |
  v
[Project] Mini Search Engine
  - 50 Wikipedia article summaries
  - Interactive console search
  - Full retrieve-then-rerank pipeline

DONE!
```

---

## Estimated Time

| Activity               | Time Estimate |
|------------------------|---------------|
| 6 Lessons (read only)  | 3 hours       |
| 6 Examples (run + read)| 2 hours       |
| 5 Exercises (coding)   | 4 hours       |
| 1 Project              | 3 hours       |
| **Total**              | **~12 hours** |

---

## The Big Picture: What You Are Building

```
User types: "how do jets stay in the air?"
                    |
                    v
           +----------------+
           |  Bi-Encoder    |  <-- Fast: encode query in ~10ms
           +----------------+
                    |
                    v
           Top 100 candidates
           (from FAISS index)
                    |
                    v
           +----------------+
           | Cross-Encoder  |  <-- Accurate: score each pair
           +----------------+
                    |
                    v
           Top 5 results ranked
           by true relevance score
                    |
                    v
    1. "Aerodynamics of fixed-wing aircraft"  [0.94]
    2. "How airplane wings generate lift"      [0.91]
    3. "Bernoulli principle in aviation"       [0.88]
    4. "Jet engine thrust mechanics"           [0.82]
    5. "History of flight"                     [0.71]
```

The documents about "lift" and "aerodynamics" NEVER used the word "stay in the air"
-- but the system found them anyway. That is semantic search.

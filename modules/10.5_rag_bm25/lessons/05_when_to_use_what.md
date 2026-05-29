# Lesson 05: When to Use BM25, Vector RAG, or Hybrid

## Glossary (Read This First!)

| Term | Plain English Definition |
|------|--------------------------|
| **Latency** | How long a search request takes. BM25: milliseconds. Dense: tens of milliseconds (embedding model call). |
| **Infrastructure cost** | Hardware/cloud cost to run the system. BM25 needs only CPU. Dense retrieval needs GPU or embedding API. |
| **Out-of-vocabulary (OOV)** | Words the retrieval system has never seen. BM25 gives OOV words a score of 0. Embeddings handle OOV better. |
| **Domain-specific terminology** | Jargon specific to your field (medical, legal, code). BM25 handles it well if the exact terms are in the query. |
| **Zero-shot** | Using a model on a task it was not specifically trained for. Dense retrieval models are often used zero-shot. |
| **Recall@K** | What fraction of all relevant documents appear in your top-K results. Higher is better. |
| **A/B testing** | Running two systems simultaneously and measuring which performs better on real user queries. |
| **Cold start** | Deploying a new system with no historical data to tune parameters on. |

---

## Part 1: Decision Flowchart

Use this flowchart when choosing your retrieval strategy.

```
+------------------------------------------------------------------+
|  RETRIEVAL STRATEGY DECISION GUIDE                               |
+------------------------------------------------------------------+
|                                                                  |
|  START HERE                                                      |
|       |                                                          |
|       v                                                          |
|  Do you have GPU / embedding API available?                      |
|       |                                                          |
|  NO   +──────────────────────────────> Use BM25                 |
|       |                                                          |
|  YES  v                                                          |
|  Do queries use exact keywords (codes, names, IDs)?              |
|       |                                                          |
|  YES  +───────────────────────────── Hybrid (BM25 heavy)        |
|       |                                                          |
|  NO   v                                                          |
|  Are queries conversational / paraphrased?                       |
|       |                                                          |
|  YES  +───────────────────────────── Dense or Hybrid            |
|       |                                                          |
|  NO   v                                                          |
|  Is latency critical (< 10ms)?                                   |
|       |                                                          |
|  YES  +───────────────────────────── BM25                       |
|       |                                                          |
|  NO   v                                                          |
|  Is highest accuracy the top priority?                           |
|       |                                                          |
|  YES  +───────────────────────────── Hybrid                     |
|  NO   |                                                          |
|       v                                                          |
|  Quick prototype / POC?                                          |
|  YES  +───────────────────────────── BM25                       |
|  NO   +───────────────────────────── Hybrid (safe default)      |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 2: Side-by-Side Comparison

```
+------------------------------------------------------------------+
|  BM25 vs VECTOR RAG vs HYBRID                                    |
+------------------------------------------------------------------+
|                                                                  |
|  Dimension          | BM25          | Vector RAG   | Hybrid      |
|  ─────────────────────────────────────────────────────────────  |
|  Speed              | Very fast     | Fast         | Fast        |
|                     | (ms)          | (10-50ms)    | (10-50ms)   |
|                     |               |              |             |
|  GPU needed?        | No            | Yes (or API) | Yes (or API)|
|                     |               |              |             |
|  Setup complexity   | Low           | Medium       | High        |
|                     |               |              |             |
|  Exact keyword match| Excellent     | Good         | Excellent   |
|  ("SKU-4821-B")     |               |              |             |
|                     |               |              |             |
|  Synonym matching   | None          | Excellent    | Excellent   |
|  ("car"/"automobile"|               |              |             |
|                     |               |              |             |
|  Multilingual       | Manual        | Built-in     | Built-in    |
|                     | (per language)| (if model    | (same)      |
|                     |               |  trained)    |             |
|                     |               |              |             |
|  Recall@10          | Good          | Good         | Best        |
|  (benchmark avg)    | (~0.55)       | (~0.60)      | (~0.70)     |
|                     |               |              |             |
|  Cost               | Lowest        | Medium       | Medium      |
|                     |               |              |             |
|  Debuggability      | High          | Low          | Medium      |
|  (can see why?)     | (word overlap)| (black box)  |             |
|                     |               |              |             |
|  Cold start         | Works         | Works        | Works       |
|  (no data needed)   |               |              |             |
|                     |               |              |             |
+------------------------------------------------------------------+
```

---

## Part 3: Use Case Map

### Use BM25 when...

| Use Case | Reason |
|----------|--------|
| Internal documentation search | Fast to set up, no GPU, exact term matching for jargon |
| Log search / error lookup | Exact error message text matters most |
| Product catalog search | SKUs, model numbers, exact names |
| Legal document search | Precise clause text, case numbers |
| Latency SLA < 10ms | BM25 is near-instant |
| Offline / edge deployment | No embedding API call needed |
| First version / MVP | Ship quickly, improve later |

### Use Dense (Vector) RAG when...

| Use Case | Reason |
|----------|--------|
| Customer support chatbot | Users describe problems in their own words |
| FAQ matching | User asks differently than the FAQ is written |
| Multilingual search | Embedding models are multilingual |
| Semantic document clustering | Meaning matters more than keywords |

### Use Hybrid when...

| Use Case | Reason |
|----------|--------|
| Enterprise knowledge base | Mix of exact codes and semantic questions |
| Code search | Functions names (BM25) + functionality (dense) |
| E-commerce search | Product names (BM25) + "similar to" (dense) |
| Medical records | ICD codes (BM25) + symptom descriptions (dense) |
| Highest accuracy required | Best recall across all query types |

C# analogy:
```csharp
// BM25    = SQL full-text search (CONTAINS / FREETEXT)
//           Fast, works without ML, good for exact terms

// Dense   = Azure Cognitive Search (semantic ranking)
//           Understands meaning, needs ML service subscription

// Hybrid  = Azure Cognitive Search with BM25 + semantic ranking
//           Most enterprise search products now combine both
```

---

## Part 4: Performance Benchmarks

BEIR (Benchmarking Information Retrieval) is the standard evaluation suite.
It tests retrieval across 18 diverse datasets.

```
+------------------------------------------------------------------+
|  BEIR BENCHMARK RESULTS (Recall@10, approximate)                 |
+------------------------------------------------------------------+
|                                                                  |
|  Task type              | BM25  | Dense  | Hybrid               |
|  ───────────────────────|────── |──────── |──────               |
|  General web (MS MARCO) | 0.67  | 0.72   | 0.78                 |
|  Scientific papers      | 0.55  | 0.60   | 0.67                 |
|  News articles          | 0.53  | 0.61   | 0.67                 |
|  Medical QA (TREC-COVID)| 0.65  | 0.68   | 0.73                 |
|  Code search (CodeSearch| 0.55  | 0.75   | 0.78                 |
|  Legal (FIQA)           | 0.48  | 0.57   | 0.62                 |
|                         |       |        |                      |
|  Average                | 0.57  | 0.66   | 0.71                 |
|                         |       |        |                      |
|  Note: Code search shows biggest gap between BM25 and dense.    |
|  Hybrid is consistently best across all domains.                 |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 5: Practical Recommendation

For this course's Capstone (Chat with Codebase):

```
+------------------------------------------------------------------+
|  CAPSTONE RETRIEVAL STRATEGY RECOMMENDATION                      |
+------------------------------------------------------------------+
|                                                                  |
|  Phase 1 (MVP):  BM25 only                                       |
|  ─────────────                                                   |
|  - Fastest to build                                              |
|  - Works for exact function names, variable names, keywords      |
|  - No model download needed to get started                       |
|  - Works offline on any machine                                  |
|                                                                  |
|  Phase 2 (Better):  Hybrid BM25 + Embeddings                     |
|  ─────────────────────────────────────────                       |
|  - Add nomic-embed-text via Ollama (runs locally, no API key)    |
|  - Use RRF to combine BM25 and embedding scores                  |
|  - 15-20% better recall for semantic queries                     |
|  - Still works offline                                           |
|                                                                  |
|  Phase 3 (Best):  Hybrid + Cross-encoder re-ranking              |
|  ─────────────────────────────────────────────────               |
|  - Add cross-encoder from Module 10.8                            |
|  - Re-rank top-20 from hybrid using cross-encoder                |
|  - Best quality, ~100ms latency per query                        |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 6: The Full Retrieval Landscape

Where M10.5 fits in the bigger picture:

```
+------------------------------------------------------------------+
|  RETRIEVAL METHODS -- FULL MAP                                   |
+------------------------------------------------------------------+
|                                                                  |
|  KEYWORD (sparse)                                                |
|  ─────────────────                                               |
|  TF-IDF   --> BM25 --> BM25+             (this module, M10.5)   |
|  Simple      Standard   Tuned                                    |
|                                                                  |
|  SEMANTIC (dense)                                                |
|  ─────────────────                                               |
|  Word2Vec --> Sentence   --> Bi-encoder  (Module 10, 10.8)       |
|  Avg          Transformers   FAISS index                         |
|                                                                  |
|  RE-RANKING                                                      |
|  ─────────                                                       |
|  Cross-encoder on top-K candidates      (Module 10.8)           |
|                                                                  |
|  FUSION                                                          |
|  ──────                                                          |
|  RRF / Linear combination of above      (this module, M10.5)    |
|                                                                  |
|  ADVANCED                                                        |
|  ────────                                                        |
|  HyDE (query expansion with LLM)        (advanced topic)        |
|  Iterative retrieval (multi-hop RAG)    (Module 11 agents)       |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Summary Table

| Question | Answer |
|----------|--------|
| Fastest retrieval | BM25 |
| Best for exact codes/names | BM25 |
| Best for semantic queries | Dense |
| Best overall recall | Hybrid |
| No GPU available | BM25 |
| Production enterprise search | Hybrid |
| Capstone MVP | BM25 first |
| Capstone final | Hybrid |

---

## Quiz

**Q1.** Your product has a strict 5ms latency requirement for search. Which method do you choose?

**Q2.** You are building a customer support bot where users describe problems in their own words. Which is better: BM25 or dense retrieval?

**Q3.** What does BEIR benchmark test?

**Q4.** In the Capstone "Chat with Codebase", why is BM25 good for Phase 1?

**Q5.** Name two real-world use cases where hybrid search is the clear winner over either method alone.

---

## Answers

**A1.** BM25. It runs in milliseconds on CPU. Dense retrieval requires an embedding model call (10-50ms minimum).

**A2.** Dense retrieval (or hybrid). Users paraphrase. They say "I can't log in" but documentation says "authentication failure". BM25 has zero overlap. Dense understands the meaning is the same.

**A3.** BEIR benchmarks retrieval systems on 18 diverse datasets covering news, science, legal, medical, and code domains. It measures Recall@K and NDCG.

**A4.** Function names, variable names, class names, and error messages are exact keywords. BM25 finds them perfectly. No GPU needed. Works offline. Fastest to build.

**A5.** Any two from: enterprise knowledge base, e-commerce product search, medical records, code search, legal document search. All have a mix of exact terms and semantic queries.

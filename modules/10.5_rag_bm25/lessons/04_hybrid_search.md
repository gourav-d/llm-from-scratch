# Lesson 04: Hybrid Search — BM25 + Embeddings

## Glossary (Read This First!)

| Term | Plain English Definition |
|------|--------------------------|
| **Hybrid search** | Combining BM25 keyword scores with embedding (vector) similarity scores to get the best of both methods. |
| **Dense retrieval** | Vector-based search. Converts text to dense vectors (embeddings), finds similar vectors. Understands meaning. |
| **Sparse retrieval** | Keyword-based search (BM25, TF-IDF). Works on word overlap. Fast and exact. |
| **Score fusion** | Combining scores from two different retrieval systems into one final ranking. |
| **Reciprocal Rank Fusion (RRF)** | A fusion method that combines ranked lists by position, not raw score. Robust and parameter-free. |
| **Linear combination** | Fusion method: final_score = alpha * sparse_score + (1-alpha) * dense_score. Requires score normalization. |
| **Normalization** | Scaling scores to the same range (usually 0 to 1) so they can be combined fairly. |
| **Recall** | Did you find ALL relevant documents? High recall = few relevant docs missed. |
| **Precision** | Of the documents you found, how many are actually relevant? High precision = few irrelevant docs returned. |
| **Bi-encoder** | The embedding model architecture used for dense retrieval. Encodes query and document independently. (See Module 10.8) |

---

## Part 1: Why Neither Method Alone Is Enough

```
+------------------------------------------------------------------+
|  BM25 vs DENSE RETRIEVAL — STRENGTHS AND WEAKNESSES              |
+------------------------------------------------------------------+
|                                                                  |
|  BM25 (keyword search)                                           |
|  ────────────────────                                            |
|  WINS:  Exact term match. Product codes, names, IDs.            |
|         Fast. No GPU. No model download.                         |
|         Interpretable. You can see WHY a doc was retrieved.      |
|                                                                  |
|  LOSES: "fast car" != "quick automobile". Zero overlap = 0 score.|
|         No concept of meaning, intent, or context.               |
|         Word order ignored.                                       |
|                                                                  |
|  Dense retrieval (embeddings)                                    |
|  ──────────────────────────                                      |
|  WINS:  Understands meaning. "fast car" finds "quick automobile".|
|         Works for paraphrased queries.                           |
|         Captures context and semantics.                          |
|                                                                  |
|  LOSES: "SKU-4821-B" may not have a good embedding.             |
|         Requires GPU or embedding API.                           |
|         Black box -- hard to debug why a doc was retrieved.      |
|         Exact string matches can score lower than paraphrases.   |
|                                                                  |
|  HYBRID: Get the wins of both. Mitigate weaknesses of both.     |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 2: Hybrid Search Architecture

```
+------------------------------------------------------------------+
|  HYBRID SEARCH PIPELINE                                          |
+------------------------------------------------------------------+
|                                                                  |
|  User query: "how do I reset my password"                        |
|       |                                                          |
|       |─────────────────────────────────────────────────        |
|       |                                                 |        |
|       v                                                 v        |
|  [BM25 retrieval]                          [Dense retrieval]     |
|  Tokenize query                            Embed query           |
|  Search inverted index                     Search vector index   |
|  Return: top-20 by BM25 score              Return: top-20 by     |
|                                            cosine similarity     |
|       |                                                 |        |
|       v                                                 v        |
|  BM25 result list:                         Dense result list:    |
|  1. doc_42 (score: 4.2)                    1. doc_17 (sim: 0.92) |
|  2. doc_17 (score: 3.8)                    2. doc_42 (sim: 0.88) |
|  3. doc_99 (score: 2.1)                    3. doc_55 (sim: 0.71) |
|  ...                                       ...                   |
|       |                                                 |        |
|       └──────────────────┬──────────────────────────────        |
|                          v                                       |
|                    [Score Fusion]                                |
|                    Combine rankings                              |
|                          |                                       |
|                          v                                       |
|                    Final ranked list:                            |
|                    1. doc_42 (appears in both, ranked high)      |
|                    2. doc_17 (appears in both, ranked high)      |
|                    3. doc_55 (only in dense, but scored well)    |
|                    4. doc_99 (only in BM25, lower combined rank) |
|                          |                                       |
|                          v                                       |
|                    Top-K chunks → LLM prompt → answer           |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 3: Reciprocal Rank Fusion (RRF)

RRF is the most popular fusion method. It is simple, parameter-free, and robust.

### The Formula

```
RRF_score(document) = sum over each ranking of:
    1 / (k + rank_in_that_ranking)

Where:
    k   = smoothing constant (default 60)
    rank = position in the ranked list (1 = first place, 2 = second, etc.)
```

The constant k=60 prevents the top result from dominating too much.

### Example

```
+------------------------------------------------------------------+
|  RRF EXAMPLE WITH 2 RANKERS                                      |
+------------------------------------------------------------------+
|                                                                  |
|  BM25 ranks:     Dense ranks:                                    |
|  Rank 1: doc_A   Rank 1: doc_B                                   |
|  Rank 2: doc_B   Rank 2: doc_A                                   |
|  Rank 3: doc_C   Rank 3: doc_D                                   |
|  Rank 4: doc_D   Rank 4: doc_C                                   |
|                                                                  |
|  RRF scores (k=60):                                              |
|                                                                  |
|  doc_A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252       |
|  doc_B: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252       |
|  doc_C: 1/(60+3) + 1/(60+4) = 0.01587 + 0.01563 = 0.03150       |
|  doc_D: 1/(60+4) + 1/(60+3) = 0.01563 + 0.01587 = 0.03150       |
|                                                                  |
|  Final ranking: doc_A = doc_B > doc_C = doc_D                    |
|                                                                  |
|  Key insight: doc_A and doc_B both appear in top-2 of both lists.|
|  Documents that rank well in BOTH systems win.                   |
|                                                                  |
+------------------------------------------------------------------+
```

Why RRF works well:
- Does not require normalizing scores to the same scale
- A document ranked #1 by both systems beats one ranked #1 by only one
- Robust to outliers in either scoring system

---

## Part 4: Linear Combination (Alternative to RRF)

Another approach: multiply scores by weights and add them.

```
hybrid_score = alpha * bm25_normalized + (1 - alpha) * dense_normalized

Where:
    alpha = 0.5 means equal weight to both
    alpha = 0.7 means 70% BM25, 30% dense
    bm25_normalized = bm25_score / max_bm25_score_in_results
    dense_normalized = cosine_similarity (already 0 to 1)
```

### When to adjust alpha

```
+--------------------------------------------+
|  Alpha tuning guide                         |
+--------------------------------------------+
|  Domain has lots of exact keywords?          |
|  → Higher alpha (more BM25 weight)          |
|  e.g., legal docs, product catalogs: 0.7    |
|                                             |
|  Domain is conversational, paraphrased?     |
|  → Lower alpha (more dense weight)          |
|  e.g., customer support chat: 0.3          |
|                                             |
|  No preference / unsure?                    |
|  → Start at alpha=0.5, A/B test             |
+--------------------------------------------+
```

Downside: you must normalize BM25 scores (they are unbounded, unlike cosine similarity which is 0-1).

---

## Part 5: Hybrid RAG Code Sketch

```python
from rank_bm25 import BM25Okapi
import numpy as np

# Assume you have:
#   documents       = list of {"id": ..., "text": ..., "embedding": np.array(...)}
#   query_embedding = np.array(...)  (from your embedding model)

STOP_WORDS = {"the", "a", "an", "is", "are", "in", "on", "for", "of"}

def preprocess(text):
    return [t for t in text.lower().split() if t not in STOP_WORDS]

# ── Build BM25 index ────────────────────────────────────────────
tokenized = [preprocess(d["text"]) for d in documents]
bm25 = BM25Okapi(tokenized)

# ── Reciprocal Rank Fusion ──────────────────────────────────────
def rrf_score(rank, k=60):
    return 1.0 / (k + rank)

def hybrid_retrieve(query_text, query_embedding, documents, top_k=5):
    # -- BM25 retrieval --
    query_tokens = preprocess(query_text)
    bm25_scores  = bm25.get_scores(query_tokens)
    bm25_ranking = np.argsort(bm25_scores)[::-1]    # descending order

    # -- Dense (cosine) retrieval --
    doc_embeddings = np.array([d["embedding"] for d in documents])
    cosine_scores  = doc_embeddings @ query_embedding  # dot product (assumes normalized)
    dense_ranking  = np.argsort(cosine_scores)[::-1]   # descending order

    # -- RRF fusion --
    rrf_scores = np.zeros(len(documents))

    for rank, doc_idx in enumerate(bm25_ranking, start=1):
        rrf_scores[doc_idx] += rrf_score(rank)

    for rank, doc_idx in enumerate(dense_ranking, start=1):
        rrf_scores[doc_idx] += rrf_score(rank)

    # Return top-K by combined RRF score
    top_indices = np.argsort(rrf_scores)[::-1][:top_k]
    return [(documents[i], rrf_scores[i]) for i in top_indices]
```

---

## Part 6: When Hybrid Beats Either Alone

```
+------------------------------------------------------------------+
|  REAL-WORLD HYBRID SEARCH WINS                                   |
+------------------------------------------------------------------+
|                                                                  |
|  Query: "AWS S3 bucket policy error"                             |
|  BM25: finds docs with "AWS", "S3", "bucket", "policy", "error" |
|  Dense: finds docs about "cloud storage permissions problem"     |
|  Hybrid: finds BOTH exact-match docs AND semantically similar    |
|                                                                  |
|  Query: "how to fix the login issue from yesterday's deploy"     |
|  BM25: finds "login", "deploy" -- misses "issue" semantic match  |
|  Dense: finds "authentication failure after release"             |
|  Hybrid: catches both "login" (exact) and "authentication" (sem) |
|                                                                  |
|  Query: "what is our refund policy"                              |
|  BM25: finds "refund" and "policy" -- good for this query        |
|  Dense: finds "how to return products" -- also good              |
|  Hybrid: returns both -- maximum coverage                        |
|                                                                  |
+------------------------------------------------------------------+
```

Studies consistently show hybrid search outperforms either alone by 5-15% on standard retrieval benchmarks (BEIR, MS MARCO).

---

## Quiz

**Q1.** What is the main advantage of hybrid search over BM25 alone?

**Q2.** In RRF with k=60, document A is ranked #1 by BM25. What is its RRF contribution from BM25?

**Q3.** Why does RRF NOT require score normalization, but linear combination DOES?

**Q4.** If your corpus contains lots of product SKUs and exact codes, should you set alpha higher or lower in linear combination? Why?

**Q5.** A document ranks #1 in BM25 and #50 in dense retrieval. Another document ranks #10 in both. Which likely wins in RRF? Why?

---

## Answers

**A1.** Hybrid catches both vocabulary matches (BM25) and semantic/meaning matches (dense). Queries that fail in one system are rescued by the other.

**A2.** 1 / (60 + 1) = 1/61 ≈ 0.01639.

**A3.** RRF uses only rank positions (1st, 2nd, 3rd...), which are already comparable between systems. Linear combination uses raw scores (BM25 might be 0-100, cosine 0-1) which must be normalized to the same scale first.

**A4.** Higher alpha (more BM25 weight). Exact product codes are perfect for keyword matching. Embeddings may not capture precise alphanumeric codes as well as BM25 does.

**A5.** The document ranked #10 in both likely wins. RRF rewards consistent high ranking across systems. Rank #10 twice gives 2 × 1/70 = 0.0286. Rank #1 once + #50 once gives 1/61 + 1/110 = 0.0254. Consistent moderate ranking beats one brilliant result and one poor one.

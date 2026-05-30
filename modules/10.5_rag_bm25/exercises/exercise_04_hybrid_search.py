"""
Exercise 04: Hybrid Search — Implement Score Fusion
Module 10.5: RAG Without Vectors

TASKS:
  1. Implement normalize_minmax() — scale scores to [0, 1]
  2. Implement linear_fusion() — combine BM25 + dense scores with alpha
  3. Implement reciprocal_rank_fusion() — RRF fusion from ranked lists
  4. Compare fusion methods on a query where BM25 and dense disagree

Run:  python exercise_04_hybrid_search.py
Deps: none (pure Python)
"""

import math
import random
from collections import Counter


# ─────────────────────────────────────────────────────────
# PROVIDED: BM25 and fake embeddings (do not modify)
# ─────────────────────────────────────────────────────────

STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","to","for","of",
    "and","or","but","in","on","at","it","its","this","that","with",
    "has","have","had","not","by","from","as","so","do","does"
}

def preprocess(text):
    tokens = text.lower().split()
    return [t.strip(".,!?;:'\"") for t in tokens
            if t.strip(".,!?;:'\"") not in STOP_WORDS
            and len(t.strip(".,!?;:'\"")) > 1]

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus; self.N = len(corpus)
        self.k1 = k1; self.b = b
        self.dl = [len(d) for d in corpus]
        self.avgdl = sum(self.dl) / self.N if self.N else 1
        self.df = {}
        for doc in corpus:
            for w in set(doc): self.df[w] = self.df.get(w, 0) + 1
        self.idf = {w: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                    for w, df in self.df.items()}

    def get_scores(self, q):
        scores = []
        for i, doc in enumerate(self.corpus):
            tf = Counter(doc)
            s = sum(self.idf.get(w, 0) * tf.get(w, 0) * (self.k1 + 1) /
                    (tf.get(w, 0) + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
                    for w in q if w in self.idf)
            scores.append(s)
        return scores

def fake_embed(text, dim=8):
    rng = random.Random(hash(text))
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    topics = {
        "payment": [2,0,0,0,0,0,0,0], "stripe":  [2,0,0,0,0,0,0,0],
        "billing": [1.5,0,0,0,0,0,0,0], "financial":[1.5,0,0,0,0,0,0,0],
        "python":  [0,2,0,0,0,0,0,0], "code":    [0,1.5,0,0,0,0,0,0],
        "deploy":  [0,0,2,0,0,0,0,0], "docker":  [0,0,1.5,0,0,0,0,0],
        "database":[0,0,0,2,0,0,0,0], "postgres":[0,0,0,2,0,0,0,0],
    }
    for word in text.lower().split():
        if word in topics:
            for i, v in enumerate(topics[word]):
                if i < dim: vec[i] += v
    mag = math.sqrt(sum(v**2 for v in vec)) or 1
    return [v / mag for v in vec]

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    ma = math.sqrt(sum(x**2 for x in a))
    mb = math.sqrt(sum(x**2 for x in b))
    return dot / (ma * mb) if ma and mb else 0.0


# ─────────────────────────────────────────────────────────
# TASK 1: Min-Max Normalization
# ─────────────────────────────────────────────────────────

def normalize_minmax(scores: list[float]) -> list[float]:
    """
    Scale a list of scores to the range [0, 1].

    Formula: (score - min) / (max - min)

    If all scores are equal (max == min): return [0.5, 0.5, ...]

    Args:
        scores: list of raw scores (any range)

    Returns:
        list of normalized scores in [0, 1]

    Example:
        normalize_minmax([0, 5, 10]) → [0.0, 0.5, 1.0]
        normalize_minmax([3, 3, 3])  → [0.5, 0.5, 0.5]

    HINT:
        mn, mx = min(scores), max(scores)
        if mx == mn:
            return [0.5] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Linear Fusion
# ─────────────────────────────────────────────────────────

def linear_fusion(sparse_scores: list[float], dense_scores: list[float],
                   alpha: float = 0.5) -> list[float]:
    """
    Combine BM25 (sparse) and dense scores using a weighted average.

    Steps:
      1. Normalize both score lists to [0, 1] with normalize_minmax
      2. Combine: final = alpha × sparse_norm + (1 - alpha) × dense_norm

    Args:
        sparse_scores: BM25 scores for all documents
        dense_scores:  cosine similarity scores for all documents
        alpha:         weight for BM25 (0 = pure dense, 1 = pure BM25)

    Returns:
        list of hybrid scores

    Example:
        linear_fusion([1.0, 0.0], [0.0, 1.0], alpha=0.5) → [0.5, 0.5]

    HINT:
        norm_s = normalize_minmax(sparse_scores)
        norm_d = normalize_minmax(dense_scores)
        return [alpha * s + (1 - alpha) * d for s, d in zip(norm_s, norm_d)]
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────

def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[float]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    For each document, sum its RRF score across all ranked lists:
        RRF_score(doc) += 1 / (k + rank_position)
    where rank_position starts at 1 (1 = best, 2 = second best, etc.)

    Args:
        rankings: list of ranked doc index lists
                  e.g. [[2, 0, 1], [0, 2, 1]] means
                       list 1: doc2 is rank1, doc0 is rank2, doc1 is rank3
                       list 2: doc0 is rank1, doc2 is rank2, doc1 is rank3
        k:        smoothing constant (default 60)

    Returns:
        list of RRF scores, one per document (indexed by doc_id)
        Higher score = more relevant

    Example:
        rankings = [[0, 1, 2], [2, 0, 1]]
        k = 60
        doc 0: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        doc 1: 1/(60+2) + 1/(60+3) = 0.01613 + 0.01587 = 0.03200
        doc 2: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226

    HINT:
        n_docs = max(max(r) for r in rankings) + 1
        rrf = [0.0] * n_docs
        for ranking in rankings:
            for rank_pos, doc_idx in enumerate(ranking, 1):   # rank starts at 1
                rrf[doc_idx] += 1.0 / (k + rank_pos)
        return rrf
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

DOCS = [
    "Stripe processes payments and handles billing transactions",
    "Python code for machine learning and data analysis",
    "Financial payment system handles monetary transfers",
    "PostgreSQL database stores user and payment records",
    "Docker containers deploy applications to Kubernetes clusters",
    "Python programming language is great for automation scripts",
]

tokenized = [preprocess(d) for d in DOCS]
bm25_idx = BM25(tokenized)
doc_embeddings = [fake_embed(d) for d in DOCS]


def test_all():
    print("=" * 60)
    print("  Exercise 04: Hybrid Search")
    print("=" * 60)

    # Test 1: normalize_minmax
    print("\n--- Test 1: normalize_minmax ---")
    result = normalize_minmax([0.0, 5.0, 10.0])
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        expected = [0.0, 0.5, 1.0]
        if all(abs(a - b) < 0.01 for a, b in zip(result, expected)):
            print(f"  PASS  [0, 5, 10] → {result}")
        else:
            print(f"  FAIL  got {result}, expected {expected}")

        result_eq = normalize_minmax([3.0, 3.0, 3.0])
        if all(abs(x - 0.5) < 0.01 for x in result_eq):
            print(f"  PASS  equal values → {result_eq}  (all 0.5)")
        else:
            print(f"  FAIL  equal values should give [0.5, 0.5, 0.5], got {result_eq}")

    # Test 2: linear_fusion
    print("\n--- Test 2: linear_fusion ---")
    sparse = [1.0, 0.0, 0.5]
    dense  = [0.0, 1.0, 0.5]
    result = linear_fusion(sparse, dense, alpha=0.5)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        # After normalization: sparse=[1,0,0.5], dense=[0,1,0.5]
        # Fusion: [0.5*1+0.5*0, 0.5*0+0.5*1, 0.5*0.5+0.5*0.5] = [0.5, 0.5, 0.5]
        if all(abs(x - 0.5) < 0.05 for x in result):
            print(f"  PASS  equal blend: {[round(x,3) for x in result]}")
        else:
            print(f"  FAIL  expected ~[0.5,0.5,0.5], got {result}")

        # alpha=1.0 should be pure sparse
        result_sparse = linear_fusion([1.0, 0.5, 0.0], [0.0, 0.5, 1.0], alpha=1.0)
        if result_sparse and result_sparse[0] > result_sparse[2]:
            print(f"  PASS  alpha=1.0 → pure BM25 ordering preserved")
        else:
            print(f"  FAIL  alpha=1.0 should preserve BM25 ordering")

    # Test 3: reciprocal_rank_fusion
    print("\n--- Test 3: reciprocal_rank_fusion ---")
    # Doc 0 is rank 1 in both lists → should win
    rankings = [[0, 1, 2], [0, 2, 1]]
    result = reciprocal_rank_fusion(rankings, k=60)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        top_doc = max(range(len(result)), key=lambda i: result[i])
        if top_doc == 0:
            print(f"  PASS  doc 0 (rank 1 in both lists) wins: {[round(x,5) for x in result]}")
        else:
            print(f"  FAIL  expected doc 0 to win, got doc {top_doc}: {result}")

        # Verify: doc in rank 1 beats doc in rank 2 in same list
        rankings2 = [[1, 0, 2], [2, 0, 1]]  # doc 0 always rank 2
        result2 = reciprocal_rank_fusion(rankings2, k=60)
        if result2:
            top2 = max(range(len(result2)), key=lambda i: result2[i])
            print(f"  Rankings {rankings2}: doc {top2} wins with score {result2[top2]:.5f}")

    # Test 4: Full Hybrid Search Comparison
    print("\n--- Test 4: Full Comparison — Synonym Query ---")
    query = "monetary billing system"  # synonyms for payment
    q_t = preprocess(query)
    q_e = fake_embed(query)

    bm25_sc  = bm25_idx.get_scores(q_t)
    dense_sc = [cosine(q_e, e) for e in doc_embeddings]

    if linear_fusion(bm25_sc, dense_sc) is None:
        print("  NOT IMPLEMENTED")
    else:
        hybrid_sc = linear_fusion(bm25_sc, dense_sc, alpha=0.5)

        bm25_rank  = sorted(range(len(bm25_sc)),  key=lambda i: -bm25_sc[i])
        dense_rank = sorted(range(len(dense_sc)), key=lambda i: -dense_sc[i])
        rrf_sc     = reciprocal_rank_fusion([bm25_rank, dense_rank])

        print(f"\n  Query: '{query}'  (uses synonyms: 'monetary'→'payment', 'billing'→'stripe')")
        print(f"\n  Method       Top-3 results")
        print(f"  {'─'*55}")

        for method, scores in [("BM25", bm25_sc), ("Dense", dense_sc),
                                ("Hybrid", hybrid_sc), ("RRF", rrf_sc)]:
            top3 = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
            print(f"  {method:<12} {[f'doc{i}' for i in top3]}")
            for i in top3[:2]:
                print(f"               [{i}] {DOCS[i][:50]}")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def normalize_minmax(scores):
#     mn, mx = min(scores), max(scores)
#     if mx == mn:
#         return [0.5] * len(scores)
#     return [(s - mn) / (mx - mn) for s in scores]
#
# def linear_fusion(sparse_scores, dense_scores, alpha=0.5):
#     norm_s = normalize_minmax(sparse_scores)
#     norm_d = normalize_minmax(dense_scores)
#     return [alpha * s + (1 - alpha) * d for s, d in zip(norm_s, norm_d)]
#
# def reciprocal_rank_fusion(rankings, k=60):
#     n_docs = max(max(r) for r in rankings) + 1
#     rrf = [0.0] * n_docs
#     for ranking in rankings:
#         for rank_pos, doc_idx in enumerate(ranking, 1):
#             rrf[doc_idx] += 1.0 / (k + rank_pos)
#     return rrf

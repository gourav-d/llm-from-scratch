"""
Exercise 05: When to Use BM25, Dense, or Hybrid
Module 10.5: RAG Without Vectors

TASKS:
  1. Implement precision_at_k() — measure retrieval precision
  2. Implement recall_at_k() — measure retrieval recall
  3. Implement choose_strategy() — apply the decision flowchart
  4. Evaluate and compare all strategies on 4 query types

Run:  python exercise_05_decision_guide.py
Deps: none (pure Python)
"""

import math
import random
from collections import Counter


# ─────────────────────────────────────────────────────────
# PROVIDED: retrieval methods (do not modify)
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

    def top_k(self, q, k=5):
        scores = self.get_scores(q)
        return sorted(range(self.N), key=lambda i: -scores[i])[:k]

def fake_embed(text, dim=8):
    rng = random.Random(hash(text))
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    topics = {
        "payment":  [2,0,0,0,0,0,0,0], "stripe":   [2,0,0,0,0,0,0,0],
        "billing":  [1.8,0,0,0,0,0,0,0], "financial":[1.5,0,0,0,0,0,0,0],
        "python":   [0,2,0,0,0,0,0,0], "code":     [0,1.5,0,0,0,0,0,0],
        "program":  [0,1.5,0,0,0,0,0,0],
        "deploy":   [0,0,2,0,0,0,0,0], "kubernetes":[0,0,2,0,0,0,0,0],
        "database": [0,0,0,2,0,0,0,0], "postgres": [0,0,0,2,0,0,0,0],
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

def dense_top_k(q_text, doc_texts, k=5):
    q_e = fake_embed(q_text)
    scores = [cosine(q_e, fake_embed(t)) for t in doc_texts]
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

def hybrid_top_k(q_text, doc_texts, bm25_idx, k=5, alpha=0.5):
    q_t = preprocess(q_text)
    bm25_sc = bm25_idx.get_scores(q_t)
    dense_sc = [cosine(fake_embed(q_text), fake_embed(t)) for t in doc_texts]
    mn_b, mx_b = min(bm25_sc), max(bm25_sc)
    mn_d, mx_d = min(dense_sc), max(dense_sc)
    nb = [(s-mn_b)/(mx_b-mn_b+1e-9) for s in bm25_sc]
    nd = [(s-mn_d)/(mx_d-mn_d+1e-9) for s in dense_sc]
    hybrid = [alpha*b + (1-alpha)*d for b,d in zip(nb, nd)]
    return sorted(range(len(hybrid)), key=lambda i: -hybrid[i])[:k]


# ─────────────────────────────────────────────────────────
# TASK 1: Precision@K
# ─────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    """
    Precision@K = number of relevant docs in top-K / K

    Measures: of the K documents you returned, how many were actually relevant?

    Args:
        retrieved: list of doc indices returned by the retrieval system (in order)
        relevant:  list of doc indices that are truly relevant
        k:         cutoff (only look at top K results)

    Returns:
        precision score between 0.0 and 1.0

    Example:
        retrieved = [0, 2, 4, 1, 3]
        relevant  = [0, 1, 2]
        k=3: top-3 retrieved = [0, 2, 4]. Relevant in top-3 = {0, 2} = 2
        precision@3 = 2/3 ≈ 0.667

    HINT:
        top_k_retrieved = retrieved[:k]
        hits = len(set(top_k_retrieved) & set(relevant))
        return hits / k
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Recall@K
# ─────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    """
    Recall@K = number of relevant docs in top-K / total relevant docs

    Measures: of all the relevant documents, how many did you find in top-K?

    Args:
        retrieved: list of doc indices returned by the retrieval system
        relevant:  list of truly relevant doc indices
        k:         cutoff

    Returns:
        recall score between 0.0 and 1.0

    Example:
        retrieved = [0, 2, 4, 1, 3]
        relevant  = [0, 1, 2]
        k=3: top-3 = [0, 2, 4]. Relevant found = {0, 2} = 2. Total relevant = 3
        recall@3 = 2/3 ≈ 0.667

    HINT:
        if not relevant:
            return 1.0
        top_k_retrieved = retrieved[:k]
        hits = len(set(top_k_retrieved) & set(relevant))
        return hits / len(relevant)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Strategy Selector
# ─────────────────────────────────────────────────────────

def choose_strategy(has_gpu: bool, query_has_exact_ids: bool,
                    query_is_conversational: bool,
                    latency_critical: bool) -> str:
    """
    Apply the decision flowchart from Lesson 05 to choose a retrieval strategy.

    Flowchart:
      1. If no GPU → "BM25"
      2. If query_has_exact_ids → "Hybrid (BM25-heavy, alpha=0.8)"
      3. If query_is_conversational → "Hybrid (balanced, alpha=0.5)"
      4. If latency_critical → "BM25"
      5. Otherwise → "Hybrid (balanced, alpha=0.5)"

    Args:
        has_gpu:                True if GPU or embedding API available
        query_has_exact_ids:    True if queries often contain product codes, IDs, names
        query_is_conversational: True if queries are natural language questions
        latency_critical:       True if response must be < 10ms

    Returns:
        One of: "BM25", "Dense", "Hybrid (BM25-heavy, alpha=0.8)", "Hybrid (balanced, alpha=0.5)"

    HINT:
        if not has_gpu:
            return "BM25"
        if query_has_exact_ids:
            return "Hybrid (BM25-heavy, alpha=0.8)"
        if query_is_conversational:
            return "Hybrid (balanced, alpha=0.5)"
        if latency_critical:
            return "BM25"
        return "Hybrid (balanced, alpha=0.5)"
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

DOCS = [
    "Stripe processes credit card payment transactions",          # 0 — payment
    "Python programming language for machine learning models",   # 1 — python
    "Financial billing system handles monetary transfers",       # 2 — payment (synonym)
    "PostgreSQL database stores payment and user records",       # 3 — payment + db
    "Kubernetes deploys Docker containers to cloud clusters",    # 4 — deploy
    "Python code automates data processing and analysis",        # 5 — python
    "Payment gateway connects billing providers via API",        # 6 — payment
    "Machine learning Python library for neural networks",       # 7 — python + ml
]

tokenized = [preprocess(d) for d in DOCS]
bm25_idx = BM25(tokenized)


def test_all():
    print("=" * 60)
    print("  Exercise 05: Decision Guide")
    print("=" * 60)

    # Test 1: precision_at_k
    print("\n--- Test 1: precision_at_k ---")
    retrieved_ex = [0, 2, 4, 1, 3]
    relevant_ex  = [0, 1, 2]
    result_p3 = precision_at_k(retrieved_ex, relevant_ex, k=3)
    result_p5 = precision_at_k(retrieved_ex, relevant_ex, k=5)

    if result_p3 is None:
        print("  NOT IMPLEMENTED YET")
    else:
        if abs(result_p3 - 2/3) < 0.01:
            print(f"  PASS  P@3 = {result_p3:.3f}  (expected 0.667)")
        else:
            print(f"  FAIL  P@3 = {result_p3:.3f}  (expected 0.667)")

        if abs(result_p5 - 3/5) < 0.01:
            print(f"  PASS  P@5 = {result_p5:.3f}  (expected 0.600)")
        else:
            print(f"  FAIL  P@5 = {result_p5:.3f}  (expected 0.600)")

    # Test 2: recall_at_k
    print("\n--- Test 2: recall_at_k ---")
    result_r3 = recall_at_k(retrieved_ex, relevant_ex, k=3)
    result_r5 = recall_at_k(retrieved_ex, relevant_ex, k=5)

    if result_r3 is None:
        print("  NOT IMPLEMENTED YET")
    else:
        if abs(result_r3 - 2/3) < 0.01:
            print(f"  PASS  R@3 = {result_r3:.3f}  (expected 0.667)")
        else:
            print(f"  FAIL  R@3 = {result_r3:.3f}  (expected 0.667)")

        if abs(result_r5 - 1.0) < 0.01:
            print(f"  PASS  R@5 = {result_r5:.3f}  (expected 1.000 — all relevant found)")
        else:
            print(f"  FAIL  R@5 = {result_r5:.3f}  (expected 1.000)")

    # Test 3: choose_strategy
    print("\n--- Test 3: choose_strategy ---")
    cases = [
        (False, False, False, False, "BM25",                        "No GPU"),
        (True,  True,  False, False, "Hybrid (BM25-heavy, alpha=0.8)", "Has exact IDs"),
        (True,  False, True,  False, "Hybrid (balanced, alpha=0.5)",   "Conversational"),
        (True,  False, False, True,  "BM25",                        "Latency critical"),
        (True,  False, False, False, "Hybrid (balanced, alpha=0.5)",   "Default"),
    ]

    if choose_strategy(False, False, False, False) is None:
        print("  NOT IMPLEMENTED YET")
    else:
        for gpu, ids, conv, lat, expected, desc in cases:
            result = choose_strategy(gpu, ids, conv, lat)
            status = "PASS" if result == expected else "FAIL"
            print(f"  {status}  {desc}: got '{result}'")
            if status == "FAIL":
                print(f"         expected '{expected}'")

    # Test 4: Evaluation on 4 query types
    print("\n--- Test 4: Evaluation on 4 Query Types ---")

    query_scenarios = [
        {
            "name":     "Exact keyword (product code)",
            "query":    "Stripe payment",
            "relevant": [0, 2, 3, 6],        # payment-related docs
            "best":     "BM25",
        },
        {
            "name":     "Synonym query",
            "query":    "monetary billing financial system",
            "relevant": [0, 2, 3, 6],
            "best":     "Dense or Hybrid",
        },
        {
            "name":     "Python-related",
            "query":    "python programming code",
            "relevant": [1, 5, 7],
            "best":     "BM25",
        },
        {
            "name":     "Mixed (payment + python)",
            "query":    "payment python automation",
            "relevant": [0, 1, 3, 5, 6],
            "best":     "Hybrid",
        },
    ]

    if precision_at_k([0], [0], 1) is None or recall_at_k([0], [0], 1) is None:
        print("  Skipped — precision/recall not implemented")
        return

    print(f"\n  {'Scenario':<30} {'Method':<8} {'P@3':>6} {'R@3':>6}")
    print(f"  {'─'*56}")

    for scenario in query_scenarios:
        q = scenario["query"]
        rel = scenario["relevant"]
        k = 3

        bm25_r  = bm25_idx.top_k(preprocess(q), k=k)
        dense_r = dense_top_k(q, DOCS, k=k)
        hybrid_r= hybrid_top_k(q, DOCS, bm25_idx, k=k, alpha=0.5)

        for method, retrieved in [("BM25", bm25_r), ("Dense", dense_r), ("Hybrid", hybrid_r)]:
            p = precision_at_k(retrieved, rel, k)
            r = recall_at_k(retrieved, rel, k)
            print(f"  {scenario['name']:<30} {method:<8} {p:>6.2f} {r:>6.2f}")
        print(f"  {'':30} Best: {scenario['best']}")
        print()


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def precision_at_k(retrieved, relevant, k):
#     top_k = retrieved[:k]
#     hits = len(set(top_k) & set(relevant))
#     return hits / k
#
# def recall_at_k(retrieved, relevant, k):
#     if not relevant:
#         return 1.0
#     top_k = retrieved[:k]
#     hits = len(set(top_k) & set(relevant))
#     return hits / len(relevant)
#
# def choose_strategy(has_gpu, query_has_exact_ids, query_is_conversational, latency_critical):
#     if not has_gpu:
#         return "BM25"
#     if query_has_exact_ids:
#         return "Hybrid (BM25-heavy, alpha=0.8)"
#     if query_is_conversational:
#         return "Hybrid (balanced, alpha=0.5)"
#     if latency_critical:
#         return "BM25"
#     return "Hybrid (balanced, alpha=0.5)"

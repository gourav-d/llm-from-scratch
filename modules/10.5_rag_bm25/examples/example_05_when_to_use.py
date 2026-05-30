"""
Example 05: When to Use BM25, Vector RAG, or Hybrid
Module 10.5: RAG Without Vectors

Demonstrates the decision framework for choosing a retrieval strategy
by running each method on scenarios where it excels or fails.
Shows performance metrics: precision, recall, and latency.

Run:  python example_05_when_to_use.py
Deps: none (pure Python)
"""

import math
import random
import time
from collections import Counter


# ─────────────────────────────────────────────────────────
# BM25 (minimal, reused)
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
    """Fake semantic embedding (topic-aware, deterministic)."""
    rng = random.Random(hash(text))
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    topics = {
        "payment": [2,0,0,0,0,0,0,0], "stripe": [2,0,0,0,0,0,0,0],
        "billing": [1.5,0,0,0,0,0,0,0], "transaction": [1.5,0,0,0,0,0,0,0],
        "python":  [0,2,0,0,0,0,0,0], "code": [0,1.5,0,0,0,0,0,0],
        "program": [0,1.5,0,0,0,0,0,0],
        "deploy":  [0,0,2,0,0,0,0,0], "kubernetes":[0,0,2,0,0,0,0,0],
        "database":[0,0,0,2,0,0,0,0], "postgres":[0,0,0,2,0,0,0,0],
        "login":   [0,0,0,0,2,0,0,0], "auth": [0,0,0,0,2,0,0,0],
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
    q_emb = fake_embed(q_text)
    scores = [cosine(q_emb, fake_embed(t)) for t in doc_texts]
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

def hybrid_top_k(q_text, docs_text, bm25_idx, k=5, alpha=0.5):
    q_t = preprocess(q_text)
    bm25_sc = bm25_idx.get_scores(q_t)
    dense_sc = [cosine(fake_embed(q_text), fake_embed(t)) for t in docs_text]
    # Normalize both to [0,1]
    def norm(s):
        mn, mx = min(s), max(s)
        return [(x-mn)/(mx-mn+1e-9) for x in s]
    hybrid = [alpha*b + (1-alpha)*d for b,d in zip(norm(bm25_sc), norm(dense_sc))]
    return sorted(range(len(hybrid)), key=lambda i: -hybrid[i])[:k]


def precision_recall(retrieved: list[int], relevant: list[int]) -> tuple[float, float]:
    """Compute precision and recall for retrieval results."""
    if not retrieved:
        return 0.0, 0.0
    tp = len(set(retrieved) & set(relevant))
    precision = tp / len(retrieved)
    recall    = tp / len(relevant) if relevant else 0.0
    return precision, recall


def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


# ─────────────────────────────────────────────────────────
# SCENARIO 1: Exact keyword search (BM25 wins)
# ─────────────────────────────────────────────────────────

print_section("SCENARIO 1: Exact Keywords — BM25 Wins")

DOCS_EXACT = [
    "Stripe processes credit card payments via secure API",
    "PayPal handles online payment transactions worldwide",
    "Error code SKU-4821-B indicates payment gateway timeout",
    "Python code for machine learning with neural networks",
    "Kubernetes deploys Docker containers to production",
    "The SKU-4821-B payment connector requires API key configuration",
    "PostgreSQL database stores transaction records and user data",
    "Redis cache improves payment checkout page load speed",
]

tok_exact = [preprocess(d) for d in DOCS_EXACT]
bm25_exact = BM25(tok_exact)

query = "SKU-4821-B payment"
relevant = [2, 5]   # docs that actually contain the error code

bm25_result  = bm25_exact.top_k(preprocess(query), k=3)
dense_result = dense_top_k(query, DOCS_EXACT, k=3)

p_bm25,  r_bm25  = precision_recall(bm25_result,  relevant)
p_dense, r_dense = precision_recall(dense_result, relevant)

print(f"\n  Query: '{query}'")
print(f"  Relevant docs: {relevant} (contain 'SKU-4821-B')\n")
print(f"  BM25   retrieved: {bm25_result}  precision={p_bm25:.2f} recall={r_bm25:.2f}")
for i in bm25_result: print(f"    [{i}] {DOCS_EXACT[i][:60]}")
print(f"\n  Dense  retrieved: {dense_result}  precision={p_dense:.2f} recall={r_dense:.2f}")
for i in dense_result: print(f"    [{i}] {DOCS_EXACT[i][:60]}")
print(f"""
  BM25 finds SKU-4821-B because it matches exact string.
  Dense misses it — product codes have no semantic neighbors.
  WINNER: BM25
""")


# ─────────────────────────────────────────────────────────
# SCENARIO 2: Paraphrase queries (Dense wins)
# ─────────────────────────────────────────────────────────

print_section("SCENARIO 2: Paraphrase / Synonym — Dense Wins")

DOCS_PARA = [
    "The payment service processes financial transactions via Stripe",
    "Users authenticate with JWT tokens and bcrypt password hashing",
    "Python programming language is popular for building machine learning models",
    "Kubernetes orchestrates containerized services in production environments",
    "PostgreSQL provides reliable relational data storage for applications",
]

tok_para = [preprocess(d) for d in DOCS_PARA]
bm25_para = BM25(tok_para)

query_para = "monetary transfer billing system"   # synonym for "payment transaction"
relevant_para = [0]   # doc 0 is about payments

bm25_r  = bm25_para.top_k(preprocess(query_para), k=3)
dense_r = dense_top_k(query_para, DOCS_PARA, k=3)

p_b, r_b = precision_recall(bm25_r, relevant_para)
p_d, r_d = precision_recall(dense_r, relevant_para)

print(f"\n  Query: '{query_para}'  (synonym for 'payment processing')")
print(f"  Relevant doc: {relevant_para}\n")
print(f"  BM25   retrieved: {bm25_r}  precision={p_b:.2f} recall={r_b:.2f}")
for i in bm25_r: print(f"    [{i}] {DOCS_PARA[i][:60]}")
print(f"\n  Dense  retrieved: {dense_r}  precision={p_d:.2f} recall={r_d:.2f}")
for i in dense_r: print(f"    [{i}] {DOCS_PARA[i][:60]}")
print(f"""
  BM25 scores 0 — no shared words between query and doc.
  Dense retrieval finds semantically similar content.
  WINNER: Dense
""")


# ─────────────────────────────────────────────────────────
# SCENARIO 3: Mixed queries (Hybrid wins)
# ─────────────────────────────────────────────────────────

print_section("SCENARIO 3: Mixed Query — Hybrid Wins")

DOCS_MIX = [
    "Payment processing uses Stripe API for credit card billing",
    "Python is used to build payment automation scripts",
    "PostgreSQL stores payment transaction records reliably",
    "Kubernetes deploys the payment microservice containers",
    "Redis caches payment session data for fast checkout",
    "The payment gateway connects to multiple billing providers",
    "Data science Python tools analyze payment fraud patterns",
]

tok_mix = [preprocess(d) for d in DOCS_MIX]
bm25_mix = BM25(tok_mix)

query_mix = "billing automation Python scripts"
relevant_mix = [0, 1, 6]  # all payment + python docs relevant

bm25_mx   = bm25_mix.top_k(preprocess(query_mix), k=3)
dense_mx  = dense_top_k(query_mix, DOCS_MIX, k=3)
hybrid_mx = hybrid_top_k(query_mix, DOCS_MIX, bm25_mix, k=3, alpha=0.5)

p_b2, r_b2 = precision_recall(bm25_mx,  relevant_mix)
p_d2, r_d2 = precision_recall(dense_mx, relevant_mix)
p_h2, r_h2 = precision_recall(hybrid_mx, relevant_mix)

print(f"\n  Query: '{query_mix}'")
print(f"  Relevant docs: {relevant_mix}\n")
print(f"  BM25    retrieved: {bm25_mx}   P={p_b2:.2f} R={r_b2:.2f}")
print(f"  Dense   retrieved: {dense_mx}  P={p_d2:.2f} R={r_d2:.2f}")
print(f"  Hybrid  retrieved: {hybrid_mx} P={p_h2:.2f} R={r_h2:.2f}")
print(f"""
  Hybrid combines the strengths of both.
  WINNER: Hybrid
""")


# ─────────────────────────────────────────────────────────
# SCENARIO 4: Latency comparison
# ─────────────────────────────────────────────────────────

print_section("SCENARIO 4: Latency — BM25 is Fastest")

# Simulate a larger corpus
N_DOCS = 1000
large_docs = [f"document {i} about topic {i % 10} with keywords word{i} value{i}" for i in range(N_DOCS)]
tok_large = [preprocess(d) for d in large_docs]

t0 = time.perf_counter()
bm25_large = BM25(tok_large)
t_build = time.perf_counter() - t0

query_lat = "topic keywords value"
q_t_lat = preprocess(query_lat)

# Time BM25 search
REPS = 50
t0 = time.perf_counter()
for _ in range(REPS):
    bm25_large.get_scores(q_t_lat)
t_bm25 = (time.perf_counter() - t0) / REPS * 1000

# Time fake dense search (embedding + cosine for all docs)
t0 = time.perf_counter()
for _ in range(REPS):
    q_emb = fake_embed(query_lat)
    [cosine(q_emb, fake_embed(d)) for d in large_docs]
t_dense = (time.perf_counter() - t0) / REPS * 1000

print(f"\n  Corpus size: {N_DOCS} documents")
print(f"  Index build time: {t_build*1000:.1f} ms")
print(f"\n  Search latency (avg over {REPS} queries):")
print(f"    BM25:          {t_bm25:.2f} ms")
print(f"    Fake dense:    {t_dense:.2f} ms  (real sentence-transformers: 10-100ms)")
print(f"""
  Real-world comparison at 100K docs:
    BM25 with inverted index:    ~1-5 ms
    Dense (embedding + search):  ~20-200 ms (depends on GPU/CPU)
    Hybrid:                      ~20-200 ms (dominated by dense step)

  For <10ms latency budget: use BM25 only.
  For best recall: use hybrid (accept higher latency).
""")


# ─────────────────────────────────────────────────────────
# DECISION GUIDE SUMMARY
# ─────────────────────────────────────────────────────────

print_section("SUMMARY: Decision Guide")

print("""
  +----------------------------------------------------------------+
  |  Your Situation               | Recommended Strategy            |
  |-------------------------------+---------------------------------|
  |  No GPU, must run on CPU      | BM25 only                      |
  |  Queries use exact IDs/codes  | BM25 (or BM25-heavy hybrid)     |
  |  Users type natural language  | Dense or Hybrid                 |
  |  Domain: code search          | BM25 (exact syntax matters)     |
  |  Domain: legal/medical text   | Hybrid (exact terms + meaning)  |
  |  Latency < 10ms required      | BM25 only                      |
  |  Best quality, latency OK     | Hybrid (RRF or linear α=0.5)    |
  |  Prototype / quick test       | BM25 (zero dependencies)        |
  |  Production, tunable          | Hybrid with A/B tested α        |
  +----------------------------------------------------------------+

  General starting recipe:
    1. Build BM25 in day 1. Ship it.
    2. Measure recall@5 on real user queries.
    3. If recall < 0.7, add dense retrieval.
    4. Combine with RRF. Measure again.
    5. Only tune α if you have labeled query-document pairs.
""")

"""
Example 04: Hybrid Search — BM25 + Fake Embeddings
Module 10.5: RAG Without Vectors

Simulates hybrid search combining BM25 (keyword) scores with
dense/vector similarity scores. Uses fake embeddings (random vectors)
to demonstrate the fusion mechanics without needing sentence-transformers.

Run:  python example_04_hybrid_search.py
Deps: none (pure Python)
"""

import math
import random
from collections import Counter


# ─────────────────────────────────────────────────────────
# BM25 (from example_02)
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
        self.k1 = k1; self.b = b
        self.corpus = corpus; self.N = len(corpus)
        self.doc_lengths = [len(d) for d in corpus]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N else 1
        self.df = {}
        for doc in corpus:
            for w in set(doc): self.df[w] = self.df.get(w, 0) + 1
        self.idf = {w: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                    for w, df in self.df.items()}

    def score(self, query_tokens, doc_idx):
        doc = self.corpus[doc_idx]
        tf_counts = Counter(doc)
        dl = self.doc_lengths[doc_idx]
        total = 0.0
        for w in query_tokens:
            if w not in self.idf: continue
            tf = tf_counts.get(w, 0)
            num = self.idf[w] * tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += num / den
        return total

    def get_scores(self, query_tokens):
        return [self.score(query_tokens, i) for i in range(self.N)]


# ─────────────────────────────────────────────────────────
# FAKE EMBEDDINGS (simulates sentence-transformers)
# ─────────────────────────────────────────────────────────

def make_fake_embedding(text: str, dim: int = 16, seed_offset: int = 0) -> list[float]:
    """
    Generate a deterministic fake embedding based on text content.
    NOT a real embedding — just for demonstrating fusion mechanics.

    In production, replace with:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(text)
    """
    # Use word hashes to create a repeatable "semantic" vector
    rng = random.Random(hash(text) + seed_offset)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    # Simulate semantic similarity: documents with shared topic words
    # get closer embeddings by adding a topic component
    topic_keywords = {
        "payment": [1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "stripe":  [0.9, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "money":   [0.8, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "python":  [0.0, 0.0, 1.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "code":    [0.0, 0.0, 0.8, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "deploy":  [0.0, 0.0, 0.0, 0.0, 1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "kubernetes": [0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "database":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "postgres":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    words = text.lower().split()
    for word in words:
        if word in topic_keywords:
            for i, val in enumerate(topic_keywords[word]):
                if i < dim:
                    vec[i] += val * 0.5
    # Normalize to unit vector
    magnitude = math.sqrt(sum(v**2 for v in vec))
    if magnitude > 0:
        vec = [v / magnitude for v in vec]
    return vec


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────────────────
# SCORE FUSION METHODS
# ─────────────────────────────────────────────────────────

def normalize_scores(scores: list[float]) -> list[float]:
    """Scale scores to [0, 1] range for fair combination."""
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def linear_fusion(bm25_scores: list[float], dense_scores: list[float],
                   alpha: float = 0.5) -> list[float]:
    """
    Linear combination of normalized scores.
    final = alpha × sparse + (1-alpha) × dense

    alpha=1.0: pure BM25
    alpha=0.0: pure dense
    alpha=0.5: equal weight
    """
    norm_bm25  = normalize_scores(bm25_scores)
    norm_dense = normalize_scores(dense_scores)
    return [alpha * b + (1 - alpha) * d
            for b, d in zip(norm_bm25, norm_dense)]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[float]:
    """
    Reciprocal Rank Fusion (RRF): combine ranked lists by position.
    final_score(doc) = sum(1 / (k + rank_in_list_i) for each list i)

    k=60 is the standard constant (controls smoothing).
    No normalization needed — works on ranks, not raw scores.
    """
    n_docs = max(max(r) for r in rankings) + 1
    rrf_scores = [0.0] * n_docs

    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, 1):
            rrf_scores[doc_idx] += 1.0 / (k + rank)

    return rrf_scores


# ─────────────────────────────────────────────────────────
# CORPUS
# ─────────────────────────────────────────────────────────

DOCS = [
    {"id": "doc1", "text": "Stripe processes credit card payments and handles payment authorization"},
    {"id": "doc2", "text": "Python programming language is popular for machine learning applications"},
    {"id": "doc3", "text": "Financial transactions require secure payment gateway integration"},
    {"id": "doc4", "text": "Kubernetes deploys containerized applications to production clusters"},
    {"id": "doc5", "text": "PostgreSQL database stores user data and payment transaction records"},
    {"id": "doc6", "text": "Money transfers and billing systems use payment processors like Stripe"},
    {"id": "doc7", "text": "Python code runs machine learning models on training data"},
    {"id": "doc8", "text": "Deploy applications to cloud infrastructure using Docker containers"},
]

tokenized = [preprocess(d["text"]) for d in DOCS]
bm25_index = BM25(tokenized)
embeddings = [make_fake_embedding(d["text"]) for d in DOCS]


def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


def show_results(label, scores, docs, top_k=4):
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
    print(f"\n  {label}:")
    for rank, (idx, score) in enumerate(ranked, 1):
        if score > 0.001:
            bar = "█" * int(score * 20)
            print(f"    {rank}. [{docs[idx]['id']}] {score:.4f}  {bar}")
            print(f"       {docs[idx]['text'][:60]}")


# ─────────────────────────────────────────────────────────
# DEMO 1: Where Each Method Fails
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: Vocabulary Mismatch — Where BM25 Fails")

# Query uses synonym "financial transaction" — docs use "payment"
query_synonym = "financial transaction processing"
q_tokens = preprocess(query_synonym)
q_emb = make_fake_embedding(query_synonym)

bm25_scores  = bm25_index.get_scores(q_tokens)
dense_scores = [cosine_sim(q_emb, d_emb) for d_emb in embeddings]

print(f"\n  Query: '{query_synonym}'")
print(f"  (query uses 'financial transaction', docs use 'payment')\n")

show_results("BM25 scores (keyword match)", bm25_scores, DOCS)
show_results("Dense scores (semantic similarity)", dense_scores, DOCS)
print(f"""
  LESSON: BM25 scores 0 for docs that use 'payment' not 'financial transaction'.
          Dense retrieval finds semantically similar docs even without exact match.
""")


# ─────────────────────────────────────────────────────────
# DEMO 2: Where Dense Fails (rare product codes)
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: Exact String Match — Where Dense Fails")

# Add a doc with specific product code
DOCS_WITH_CODE = DOCS + [
    {"id": "sku99", "text": "Product SKU-4821-B is the enterprise Stripe payment connector module"},
]
tokenized_c = [preprocess(d["text"]) for d in DOCS_WITH_CODE]
bm25_c = BM25(tokenized_c)
embeddings_c = [make_fake_embedding(d["text"]) for d in DOCS_WITH_CODE]

query_code = "SKU-4821-B"
q_tokens_c = preprocess(query_code)
q_emb_c = make_fake_embedding(query_code)

bm25_s = bm25_c.get_scores(q_tokens_c)
dense_s = [cosine_sim(q_emb_c, e) for e in embeddings_c]

print(f"\n  Query: '{query_code}' (exact product code)\n")
show_results("BM25 scores", bm25_s, DOCS_WITH_CODE)
show_results("Dense scores", dense_s, DOCS_WITH_CODE)
print(f"""
  LESSON: BM25 finds the exact product code immediately.
          Dense retrieval struggles — product codes have no semantic meaning.
""")


# ─────────────────────────────────────────────────────────
# DEMO 3: Linear Fusion
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: Linear Fusion — alpha Controls the Blend")

query_blend = "financial transaction payment"
q_t = preprocess(query_blend)
q_e = make_fake_embedding(query_blend)

bm25_sc  = bm25_index.get_scores(q_t)
dense_sc = [cosine_sim(q_e, e) for e in embeddings]

print(f"\n  Query: '{query_blend}'")

for alpha in [1.0, 0.7, 0.5, 0.3, 0.0]:
    hybrid = linear_fusion(bm25_sc, dense_sc, alpha=alpha)
    top1_idx = max(range(len(hybrid)), key=lambda i: hybrid[i])
    label = {1.0: "pure BM25", 0.7: "BM25-heavy", 0.5: "balanced",
             0.3: "dense-heavy", 0.0: "pure dense"}[alpha]
    print(f"\n  alpha={alpha} ({label}):")
    ranked = sorted(enumerate(hybrid), key=lambda x: -x[1])[:3]
    for rank, (idx, score) in enumerate(ranked, 1):
        print(f"    {rank}. [{DOCS[idx]['id']}] {score:.4f}  {DOCS[idx]['text'][:55]}")


# ─────────────────────────────────────────────────────────
# DEMO 4: Reciprocal Rank Fusion (RRF)
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: Reciprocal Rank Fusion (RRF)")

query_rrf = "payment processing system"
q_t2 = preprocess(query_rrf)
q_e2 = make_fake_embedding(query_rrf)

bm25_sc2  = bm25_index.get_scores(q_t2)
dense_sc2 = [cosine_sim(q_e2, e) for e in embeddings]

# Convert scores to rankings (doc indices sorted by score)
bm25_ranking  = [i for i, _ in sorted(enumerate(bm25_sc2),  key=lambda x: -x[1])]
dense_ranking = [i for i, _ in sorted(enumerate(dense_sc2), key=lambda x: -x[1])]

rrf_scores = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)

print(f"\n  Query: '{query_rrf}'")
print(f"\n  BM25 ranking:  {[DOCS[i]['id'] for i in bm25_ranking[:5]]}")
print(f"  Dense ranking: {[DOCS[i]['id'] for i in dense_ranking[:5]]}")
print(f"\n  RRF combined ranking:")
rrf_ranked = sorted(enumerate(rrf_scores), key=lambda x: -x[1])[:4]
for rank, (idx, score) in enumerate(rrf_ranked, 1):
    print(f"    {rank}. [{DOCS[idx]['id']}] RRF={score:.5f}  {DOCS[idx]['text'][:55]}")

print(f"""
  RRF advantages over linear fusion:
  1. No score normalization needed (works on ranks not scores)
  2. Parameter-free (k=60 works for almost all use cases)
  3. Robust to one system having no relevant results
  4. Used by Elasticsearch and Vespa in production
""")


# ─────────────────────────────────────────────────────────
# DEMO 5: Summary Table
# ─────────────────────────────────────────────────────────

print_section("DEMO 5: When to Use Each Method")

print(f"""
  +----------------------------------------------------------+
  |  Method          | Best For                   | Avoid     |
  |------------------+----------------------------+-----------|
  |  BM25            | Exact terms, codes, names  | Synonyms  |
  |  Dense/Vector    | Synonyms, paraphrases       | Rare IDs  |
  |  Hybrid (linear) | General purpose             | When α    |
  |                  |                             | tuning is |
  |                  |                             | difficult |
  |  Hybrid (RRF)    | When you can't tune α       | -         |
  |                  | Multiple signal sources      |           |
  +----------------------------------------------------------+

  Rule of thumb:
    Start with BM25 (fast, free, no GPU).
    If recall is low (missing relevant docs) → add dense retrieval.
    Combine with RRF if you don't want to tune α.
""")

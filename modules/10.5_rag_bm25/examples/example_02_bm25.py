"""
Example 02: BM25 from Scratch
Module 10.5: RAG Without Vectors

Implements BM25 (Okapi BM25) from scratch — no rank-bm25 library.
Shows the saturation and length normalization effects visually.

Run:  python example_02_bm25.py
Deps: none (pure Python)
"""

import math
from collections import Counter


# ─────────────────────────────────────────────────────────
# BM25 IMPLEMENTATION
# ─────────────────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "to", "for", "of", "and", "or", "but", "in", "on", "at",
    "it", "its", "this", "that", "with", "has", "have", "had",
    "not", "by", "from", "as", "so", "do", "does"
}


def preprocess(text: str) -> list[str]:
    tokens = text.lower().split()
    return [t.strip(".,!?;:'\"") for t in tokens
            if t.strip(".,!?;:'\"") not in STOP_WORDS
            and len(t.strip(".,!?;:'\"")) > 1]


class BM25:
    """
    Okapi BM25 — industry standard keyword ranking algorithm.

    Parameters:
        k1 = 1.5  (saturation control — how fast TF reward levels off)
        b  = 0.75 (length normalization — penalty for long documents)
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus = corpus
        self.N = len(corpus)

        # Document lengths
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 1

        # Document frequencies: how many docs contain each term
        self.df = {}
        for doc in corpus:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1

        # IDF (BM25 smoothed version — avoids negative scores)
        self.idf = {}
        for word, df in self.df.items():
            self.idf[word] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Compute BM25 score for one (query, document) pair."""
        doc = self.corpus[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        tf_counts = Counter(doc)

        total = 0.0
        for word in query_tokens:
            if word not in self.idf:
                continue
            tf = tf_counts.get(word, 0)
            idf = self.idf[word]

            # BM25 term score:
            # numerator:   idf × tf × (k1 + 1)
            # denominator: tf + k1 × (1 - b + b × doc_len / avgdl)
            numerator   = idf * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            total += numerator / denominator

        return total

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        """Score all documents for a query."""
        return [self.score(query_tokens, i) for i in range(self.N)]

    def search(self, query_tokens: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """Return top-K (doc_idx, score) pairs sorted by score."""
        scores = self.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        return ranked[:top_k]


# ─────────────────────────────────────────────────────────
# CORPUS
# ─────────────────────────────────────────────────────────

CORPUS_DOCS = [
    {"id": "doc1", "text": "Python is a popular programming language for machine learning"},
    {"id": "doc2", "text": "Machine learning uses neural networks and data science"},
    {"id": "doc3", "text": "Python has simple syntax and is easy to learn for beginners"},
    {"id": "doc4", "text": "Neural networks require lots of training data to work well"},
    {"id": "doc5", "text": "Data science involves statistics programming and analytical skills"},
    {"id": "doc6", "text": "JavaScript is used for web development and front end programming"},
]

tokenized_corpus = [preprocess(doc["text"]) for doc in CORPUS_DOCS]
bm25 = BM25(tokenized_corpus)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ─────────────────────────────────────────────────────────
# DEMO 1: BM25 Saturation vs TF-IDF Linear Growth
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: Saturation Effect — BM25 vs TF-IDF")

print("""
  TF-IDF: score grows linearly with word count.
  BM25:   score saturates — 100th copy barely helps.
""")

# Simulate what happens as a term appears more and more times
k1 = 1.5
avgdl = 100
doc_len = 100
idf_val = 1.0   # constant for this demo

print(f"  {'Word count':>12} {'TF-IDF score':>14} {'BM25 score':>12}")
print(f"  {'-'*42}")

for count in [1, 2, 5, 10, 20, 50, 100]:
    tfidf_score = (count / doc_len) * idf_val

    tf = count
    bm25_numerator   = idf_val * tf * (k1 + 1)
    bm25_denominator = tf + k1 * (1 - 0.75 + 0.75 * doc_len / avgdl)
    bm25_score = bm25_numerator / bm25_denominator

    tfidf_bar = "█" * int(tfidf_score * 300)
    bm25_bar  = "█" * int(bm25_score * 10)
    print(f"  {count:>12}   TF-IDF: {tfidf_score:.4f} {tfidf_bar}")
    print(f"  {'':>12}   BM25:   {bm25_score:.4f} {bm25_bar}")

print("""
  LESSON: TF-IDF score at count=100 is 100x higher than count=1.
          BM25 score at count=100 is only ~1.5x higher than count=1.
          BM25 is much more resistant to "keyword stuffing".
""")


# ─────────────────────────────────────────────────────────
# DEMO 2: Length Normalization
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: Length Normalization")

print("""
  Without length normalization, long documents win unfairly.
  BM25 penalizes long documents relative to average length.
""")

# Two documents with the same term frequency but different lengths
# Doc A: short doc (50 words), "python" appears 5 times
# Doc B: long doc  (500 words), "python" appears 5 times
# Without normalization, they'd score the same.
# With BM25 b=0.75, Doc A scores higher (more focused).

idf_python = 0.8
k1 = 1.5
b  = 0.75
avgdl = 200

scenarios = [
    ("Short doc (50 words, 'python' ×5)",  5,  50),
    ("Medium doc (200 words, 'python' ×5)", 5, 200),
    ("Long doc (500 words, 'python' ×5)",   5, 500),
    ("Long doc (500 words, 'python' ×20)",  20, 500),
]

print(f"\n  {'Scenario':<42} BM25 Score")
print(f"  {'-'*58}")

for label, tf, doc_len in scenarios:
    numerator   = idf_python * tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
    score = numerator / denominator
    bar = "█" * int(score * 5)
    print(f"  {label:<42} {score:.4f}  {bar}")

print("""
  LESSON: Short focused doc scores higher than long diluted doc
          even with same word count. BM25's length normalization is fair.
""")


# ─────────────────────────────────────────────────────────
# DEMO 3: k1 Parameter Effect
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: k1 Parameter — Controlling Saturation Speed")

tf = 10   # word appears 10 times
doc_len = 100
avgdl = 100
idf_val = 1.0

print(f"\n  Word appears {tf} times. k1 controls how much the 10th copy helps vs 1st.\n")
print(f"  {'k1':>6} {'Score':>10}  {'Interpretation'}")
print(f"  {'-'*50}")

for k1_val in [0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 100.0]:
    num = idf_val * tf * (k1_val + 1)
    den = tf + k1_val * (1 - 0.75 + 0.75 * doc_len / avgdl)
    score = num / den if den > 0 else 0
    if k1_val == 0.0:
        interp = "TF ignored, only IDF"
    elif k1_val <= 1.0:
        interp = "Quick saturation"
    elif k1_val <= 2.0:
        interp = "Standard (recommended)"
    elif k1_val <= 5.0:
        interp = "Slow saturation"
    else:
        interp = "Approaches TF-IDF (no saturation)"
    print(f"  {k1_val:>6.1f} {score:>10.4f}  {interp}")


# ─────────────────────────────────────────────────────────
# DEMO 4: Real Search
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: BM25 Search on Real Corpus")

queries = [
    "machine learning python",
    "neural network training data",
    "web development javascript",
    "data science statistics",
]

for query in queries:
    q_tokens = preprocess(query)
    results = bm25.search(q_tokens, top_k=3)
    print(f"\n  Query: '{query}'  → tokens: {q_tokens}")
    for rank, (idx, score) in enumerate(results, 1):
        if score > 0:
            bar = "█" * int(score * 10)
            print(f"    {rank}. [{CORPUS_DOCS[idx]['id']}] {score:.4f}  {bar}")
            print(f"       {CORPUS_DOCS[idx]['text']}")


# ─────────────────────────────────────────────────────────
# DEMO 5: IDF Scores
# ─────────────────────────────────────────────────────────

print_section("DEMO 5: BM25 IDF Scores (All Vocabulary)")

print(f"\n  {'Word':<20} IDF       Documents containing it")
print(f"  {'-'*55}")

for word, idf_score in sorted(bm25.idf.items(), key=lambda x: -x[1]):
    df = bm25.df[word]
    bar = "█" * int(idf_score * 5)
    print(f"  {word:<20} {idf_score:.4f}    {df}/{bm25.N} docs  {bar}")

print(f"""
  LESSON: Words in only 1 doc get highest IDF.
          Words in all {bm25.N} docs get lowest IDF.
          BM25 IDF formula is slightly different from TF-IDF:
            BM25: log((N - df + 0.5) / (df + 0.5) + 1)
            TF-IDF: log(N / df)
          BM25 formula prevents negative scores and handles edge cases better.
""")

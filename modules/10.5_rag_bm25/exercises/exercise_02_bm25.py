"""
Exercise 02: BM25 from Scratch
Module 10.5: RAG Without Vectors

TASKS:
  1. Implement bm25_idf() — BM25 smoothed IDF formula
  2. Implement bm25_term_score() — score one term in one document
  3. Implement BM25Index.get_scores() — score all documents for a query
  4. Show saturation: confirm 100th occurrence barely increases score

Run:  python exercise_02_bm25.py
Deps: none (pure Python)
"""

import math
from collections import Counter


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


# ─────────────────────────────────────────────────────────
# TASK 1: BM25 IDF
# ─────────────────────────────────────────────────────────

def bm25_idf(N: int, df: int) -> float:
    """
    Compute BM25's smoothed IDF for one term.

    Formula: log((N - df + 0.5) / (df + 0.5) + 1)

    This differs from plain TF-IDF IDF (log(N/df)) in two ways:
      - +0.5 smoothing prevents division by zero
      - +1 offset ensures score is always positive (no negative IDF)

    Args:
        N:   total number of documents in corpus
        df:  number of documents containing this term

    Returns:
        IDF score (float, always >= 0)

    Example:
        bm25_idf(100, 1)   → log((100-1+0.5)/(1+0.5)+1)   ≈ 4.20
        bm25_idf(100, 50)  → log((100-50+0.5)/(50+0.5)+1) ≈ 0.68
        bm25_idf(100, 100) → log((100-100+0.5)/(100+0.5)+1) ≈ 0.005

    HINT: math.log((N - df + 0.5) / (df + 0.5) + 1)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: BM25 Term Score
# ─────────────────────────────────────────────────────────

def bm25_term_score(tf: int, doc_len: int, avgdl: float,
                    idf: float, k1: float = 1.5, b: float = 0.75) -> float:
    """
    Compute BM25 score for one term in one document.

    Formula:
        numerator   = idf × tf × (k1 + 1)
        denominator = tf + k1 × (1 - b + b × doc_len / avgdl)
        score       = numerator / denominator

    Args:
        tf:      raw count of the term in this document
        doc_len: total number of tokens in this document
        avgdl:   average document length across corpus
        idf:     IDF score for this term (from bm25_idf)
        k1:      saturation parameter (default 1.5)
        b:       length normalization (default 0.75)

    Returns:
        BM25 score for this (term, document) pair

    HINT:
        numerator   = idf * tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
        return numerator / denominator
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: BM25 Index
# ─────────────────────────────────────────────────────────

class BM25Index:
    """
    Complete BM25 index for a corpus.
    You need to implement get_scores().
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus = corpus
        self.N = len(corpus)

        # Document lengths
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 1

        # Document frequencies
        self.df = {}
        for doc in corpus:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1

        # Pre-compute IDF for all terms
        self.idf = {word: bm25_idf(self.N, df)
                    for word, df in self.df.items()}

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        """
        Compute BM25 score for EVERY document against the query.

        For each document:
          - For each query token that exists in the corpus vocabulary:
            - Get TF of that token in the document
            - Compute bm25_term_score(tf, doc_len, avgdl, idf)
          - Sum all term scores → document score

        Args:
            query_tokens: list of preprocessed query tokens

        Returns:
            list of float scores, one per document

        HINT:
            scores = []
            for i, doc in enumerate(self.corpus):
                tf_counts = Counter(doc)
                doc_len   = self.doc_lengths[i]
                total = 0.0
                for word in query_tokens:
                    if word not in self.idf:
                        continue
                    tf = tf_counts.get(word, 0)
                    total += bm25_term_score(tf, doc_len, self.avgdl,
                                             self.idf[word], self.k1, self.b)
                scores.append(total)
            return scores
        """
        # TODO: implement this
        pass

    def search(self, query_tokens: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """Return top-K (doc_idx, score) pairs. Uses your get_scores()."""
        scores = self.get_scores(query_tokens)
        if scores is None:
            return []
        return sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

CORPUS = [
    "Python is a popular programming language for machine learning",
    "Machine learning uses neural networks and data science",
    "Python has simple syntax and is easy to learn",
    "Neural networks require lots of training data to work well",
    "JavaScript is used for web development and front end",
]


def test_all():
    print("=" * 55)
    print("  Exercise 02: BM25")
    print("=" * 55)

    # Test 1: bm25_idf
    print("\n--- Test 1: bm25_idf ---")
    r1 = bm25_idf(100, 1)
    r2 = bm25_idf(100, 50)
    r3 = bm25_idf(100, 100)

    if r1 is None:
        print("  NOT IMPLEMENTED YET")
    else:
        e1 = math.log((100 - 1 + 0.5) / (1 + 0.5) + 1)
        if abs(r1 - e1) < 0.001:
            print(f"  PASS  idf(N=100, df=1)   = {r1:.4f}  (expected {e1:.4f})")
        else:
            print(f"  FAIL  got {r1:.4f}, expected {e1:.4f}")

        if r1 > r2 > 0:
            print(f"  PASS  rarer words get higher IDF: df=1 ({r1:.3f}) > df=50 ({r2:.3f})")
        else:
            print(f"  FAIL  expected idf(df=1) > idf(df=50): {r1:.3f} vs {r2:.3f}")

        if r3 > 0:
            print(f"  PASS  idf(df=N) = {r3:.4f} > 0  (BM25 never gives negative IDF)")
        else:
            print(f"  FAIL  expected > 0, got {r3}")

    # Test 2: bm25_term_score
    print("\n--- Test 2: bm25_term_score ---")
    idf_test = 1.0
    # Score with tf=0 (word not in doc) should be 0
    score_zero = bm25_term_score(0, 100, 100, idf_test)
    # Score should increase with tf but with diminishing returns
    score_1  = bm25_term_score(1,   100, 100, idf_test)
    score_10 = bm25_term_score(10,  100, 100, idf_test)
    score_100= bm25_term_score(100, 100, 100, idf_test)

    if score_zero is None:
        print("  NOT IMPLEMENTED YET")
    else:
        if score_zero == 0.0:
            print(f"  PASS  score(tf=0) = 0.0")
        else:
            print(f"  FAIL  score(tf=0) should be 0.0, got {score_zero}")

        if score_1 < score_10 < score_100:
            print(f"  PASS  score increases with TF: tf=1({score_1:.3f}) < tf=10({score_10:.3f}) < tf=100({score_100:.3f})")
        else:
            print(f"  FAIL  scores not monotonically increasing: {score_1:.3f} / {score_10:.3f} / {score_100:.3f}")

        # Key BM25 property: saturation
        ratio = score_100 / score_1 if score_1 > 0 else 0
        if ratio < 3.0:
            print(f"  PASS  saturation: score(tf=100)/score(tf=1) = {ratio:.2f}  (not 100x like TF-IDF)")
        else:
            print(f"  FAIL  no saturation: ratio={ratio:.2f}  (should be < 3.0)")

    # Test 3: BM25Index.get_scores
    print("\n--- Test 3: BM25Index.get_scores ---")
    tokenized = [preprocess(doc) for doc in CORPUS]
    index = BM25Index(tokenized)
    result = index.get_scores(preprocess("machine learning"))

    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        if len(result) == len(CORPUS):
            print(f"  PASS  returned {len(result)} scores (one per doc)")
        else:
            print(f"  FAIL  expected {len(CORPUS)} scores, got {len(result)}")

        # Docs 0 and 1 should score highest for "machine learning"
        top2 = sorted(range(len(result)), key=lambda i: -result[i])[:2]
        if 0 in top2 or 1 in top2:
            print(f"  PASS  top-2 docs include relevant results: {top2}")
        else:
            print(f"  FAIL  top-2 should include doc 0 or 1, got {top2}")

    # Test 4: Full Search
    print("\n--- Test 4: Full search ---")
    results = index.search(preprocess("python programming"), top_k=3)
    if not results:
        print("  NOT IMPLEMENTED (get_scores returns None)")
    else:
        print(f"  Query: 'python programming'")
        for idx, score in results:
            print(f"    [{idx}] score={score:.4f}  {CORPUS[idx][:55]}")
        top_idx = results[0][0] if results else -1
        if top_idx in [0, 2]:
            print(f"  PASS  top result is a Python doc")
        else:
            print(f"  FAIL  expected Python doc (0 or 2), got doc {top_idx}")

    # Bonus: Saturation demonstration
    print("\n--- BONUS: Saturation vs TF-IDF ---")
    if bm25_term_score(1, 100, 100, 1.0) is not None:
        print(f"\n  BM25 score as term frequency increases (k1=1.5, b=0.75):")
        print(f"  {'TF count':>10} {'BM25 score':>12} {'TF-IDF score':>14}")
        print(f"  {'-'*40}")
        for tf in [1, 5, 10, 50, 100, 500]:
            bm25_s = bm25_term_score(tf, 100, 100, 1.0)
            tfidf_s = (tf / 100) * 1.0
            bar = "█" * int(bm25_s * 5)
            print(f"  {tf:>10} {bm25_s:>12.4f} {tfidf_s:>14.4f}  {bar}")
        print(f"\n  BM25 saturates. TF-IDF grows linearly. BM25 is more robust.")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def bm25_idf(N, df):
#     return math.log((N - df + 0.5) / (df + 0.5) + 1)
#
# def bm25_term_score(tf, doc_len, avgdl, idf, k1=1.5, b=0.75):
#     numerator   = idf * tf * (k1 + 1)
#     denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
#     return numerator / denominator
#
# # In BM25Index:
# def get_scores(self, query_tokens):
#     scores = []
#     for i, doc in enumerate(self.corpus):
#         tf_counts = Counter(doc)
#         doc_len   = self.doc_lengths[i]
#         total = 0.0
#         for word in query_tokens:
#             if word not in self.idf:
#                 continue
#             tf = tf_counts.get(word, 0)
#             total += bm25_term_score(tf, doc_len, self.avgdl,
#                                      self.idf[word], self.k1, self.b)
#         scores.append(total)
#     return scores

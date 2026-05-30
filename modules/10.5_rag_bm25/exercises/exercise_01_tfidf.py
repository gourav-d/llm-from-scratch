"""
Exercise 01: TF-IDF from Scratch
Module 10.5: RAG Without Vectors

TASKS:
  1. Implement compute_tf() — term frequency for a list of tokens
  2. Implement compute_idf() — inverse document frequency across corpus
  3. Implement tfidf_score() — TF × IDF for one word in one document
  4. Implement search_tfidf() — score all docs and return top-K

Run:  python exercise_01_tfidf.py
Deps: none (pure Python)
"""

import math
from collections import Counter


STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","to","for","of",
    "and","or","but","in","on","at","it","its","this","that","with",
    "has","have","had","not","by","from","as","so","do","does"
}

def preprocess(text: str) -> list[str]:
    tokens = text.lower().split()
    return [t.strip(".,!?;:'\"") for t in tokens
            if t.strip(".,!?;:'\"") not in STOP_WORDS
            and len(t.strip(".,!?;:'\"")) > 1]


# ─────────────────────────────────────────────────────────
# TASK 1: Term Frequency
# ─────────────────────────────────────────────────────────

def compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    Compute term frequency for a list of tokens.

    TF(word) = count of word / total number of tokens

    Args:
        tokens: list of words (already preprocessed)

    Returns:
        dict mapping word → TF score

    Example:
        compute_tf(["cat", "cat", "sat"]) → {"cat": 0.667, "sat": 0.333}

    HINT:
        counts = Counter(tokens)
        total = len(tokens)
        return {word: count / total for word, count in counts.items()}
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Inverse Document Frequency
# ─────────────────────────────────────────────────────────

def compute_idf(tokenized_corpus: list[list[str]]) -> dict[str, float]:
    """
    Compute IDF for all words in the corpus.

    IDF(word) = log(total_docs / number_of_docs_containing_word)

    Args:
        tokenized_corpus: list of token lists (one per document)

    Returns:
        dict mapping word → IDF score

    Example:
        corpus = [["cat", "sat"], ["dog", "ran"], ["cat", "dog"]]
        idf["cat"]  → log(3/2) ≈ 0.405  (appears in 2 of 3 docs)
        idf["sat"]  → log(3/1) ≈ 1.099  (appears in 1 of 3 docs)

    HINT:
        N = len(tokenized_corpus)
        doc_freq = {}
        for tokens in tokenized_corpus:
            for word in set(tokens):           # set: count each doc once
                doc_freq[word] = doc_freq.get(word, 0) + 1
        return {word: math.log(N / df) for word, df in doc_freq.items()}
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: TF-IDF Score
# ─────────────────────────────────────────────────────────

def tfidf_score(word: str, tokens: list[str], idf: dict[str, float]) -> float:
    """
    Compute TF-IDF score for one word in one document.

    TF-IDF = TF(word, doc) × IDF(word)

    Args:
        word:   the word to score
        tokens: the document's token list
        idf:    pre-computed IDF dict from compute_idf()

    Returns:
        float TF-IDF score (0.0 if word not in doc or not in IDF)

    HINT:
        tf = compute_tf(tokens)
        return tf.get(word, 0.0) * idf.get(word, 0.0)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: TF-IDF Search
# ─────────────────────────────────────────────────────────

def search_tfidf(query: str, corpus: list[str], top_k: int = 3) -> list[tuple[int, float]]:
    """
    Search a corpus using TF-IDF cosine similarity.

    Steps:
      1. Tokenize all documents
      2. Compute IDF across corpus
      3. Build TF-IDF vector for each document
      4. Build TF-IDF vector for the query
      5. Compute cosine similarity between query and each document
      6. Return top-K (doc_index, score) pairs sorted by score

    Args:
        query:   search query string
        corpus:  list of document strings
        top_k:   number of results to return

    Returns:
        list of (doc_index, score) tuples, best first

    HINT:
        def cosine(a, b):
            dot = sum(a.get(w,0) * b.get(w,0) for w in a)
            mag_a = math.sqrt(sum(v**2 for v in a.values()))
            mag_b = math.sqrt(sum(v**2 for v in b.values()))
            return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

        tokenized = [preprocess(doc) for doc in corpus]
        idf = compute_idf(tokenized)
        tfidf_vecs = [{w: tfidf_score(w, t, idf) for w in t} for t in tokenized]

        q_tokens = preprocess(query)
        q_tf = compute_tf(q_tokens)
        q_vec = {w: q_tf[w] * idf.get(w, 0) for w in q_tf if w in idf}

        scored = [(i, cosine(q_vec, v)) for i, v in enumerate(tfidf_vecs)]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
    """
    # TODO: implement this
    pass


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
    print("  Exercise 01: TF-IDF")
    print("=" * 55)

    # Test 1: compute_tf
    print("\n--- Test 1: compute_tf ---")
    tokens = ["cat", "cat", "sat", "mat"]
    result = compute_tf(tokens)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        if abs(result.get("cat", 0) - 0.5) < 0.01:
            print(f"  PASS  'cat' TF = {result['cat']:.3f}  (expected 0.5)")
        else:
            print(f"  FAIL  'cat' TF = {result.get('cat', 'missing')}  (expected 0.5)")
        if abs(result.get("sat", 0) - 0.25) < 0.01:
            print(f"  PASS  'sat' TF = {result['sat']:.3f}  (expected 0.25)")
        else:
            print(f"  FAIL  'sat' TF = {result.get('sat', 'missing')}  (expected 0.25)")

    # Test 2: compute_idf
    print("\n--- Test 2: compute_idf ---")
    mini_corpus = [["cat", "sat"], ["dog", "ran"], ["cat", "dog"]]
    result = compute_idf(mini_corpus)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        expected_cat = math.log(3 / 2)
        expected_sat = math.log(3 / 1)
        if abs(result.get("cat", 0) - expected_cat) < 0.01:
            print(f"  PASS  IDF('cat') = {result['cat']:.4f}  (expected {expected_cat:.4f})")
        else:
            print(f"  FAIL  IDF('cat') = {result.get('cat', 'missing')}  (expected {expected_cat:.4f})")
        if abs(result.get("sat", 0) - expected_sat) < 0.01:
            print(f"  PASS  IDF('sat') = {result['sat']:.4f}  (expected {expected_sat:.4f})")
        else:
            print(f"  FAIL  IDF('sat') = {result.get('sat', 'missing')}  (expected {expected_sat:.4f})")

    # Test 3: tfidf_score
    print("\n--- Test 3: tfidf_score ---")
    tokens3 = ["cat", "cat", "sat"]
    idf3 = {"cat": 0.5, "sat": 1.0}
    result = tfidf_score("cat", tokens3, idf3)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        # TF("cat") = 2/3, IDF("cat") = 0.5 → TF-IDF = 2/3 × 0.5 = 0.333
        expected = (2/3) * 0.5
        if abs(result - expected) < 0.01:
            print(f"  PASS  TF-IDF('cat') = {result:.4f}  (expected {expected:.4f})")
        else:
            print(f"  FAIL  got {result:.4f}, expected {expected:.4f}")

        result_zero = tfidf_score("dog", tokens3, idf3)
        if result_zero == 0.0:
            print(f"  PASS  TF-IDF('dog') = 0.0  (word not in doc)")
        else:
            print(f"  FAIL  expected 0.0 for missing word, got {result_zero}")

    # Test 4: search_tfidf
    print("\n--- Test 4: search_tfidf ---")
    result = search_tfidf("machine learning python", CORPUS, top_k=3)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        print(f"  Results for 'machine learning python':")
        for idx, score in result:
            print(f"    [{idx}] score={score:.4f}  {CORPUS[idx][:55]}")

        top_idx = result[0][0] if result else -1
        if top_idx in [0, 1]:   # docs about ML and Python
            print(f"  PASS  top result is relevant (doc {top_idx})")
        else:
            print(f"  FAIL  expected doc 0 or 1, got doc {top_idx}")

    # Bonus: IDF analysis
    print("\n--- BONUS: IDF Analysis on Full Corpus ---")
    if compute_idf([[]] * 3) is not None:
        tokenized = [preprocess(doc) for doc in CORPUS]
        idf = compute_idf(tokenized)
        if idf:
            top_idf = sorted(idf.items(), key=lambda x: -x[1])[:5]
            print(f"\n  Highest IDF (most distinctive words):")
            for word, score in top_idf:
                print(f"    {word:<20} IDF={score:.4f}")
            bottom_idf = sorted(idf.items(), key=lambda x: x[1])[:3]
            print(f"\n  Lowest IDF (most common across all docs):")
            for word, score in bottom_idf:
                print(f"    {word:<20} IDF={score:.4f}")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def compute_tf(tokens):
#     counts = Counter(tokens)
#     total = len(tokens)
#     return {word: count / total for word, count in counts.items()}
#
# def compute_idf(tokenized_corpus):
#     N = len(tokenized_corpus)
#     doc_freq = {}
#     for tokens in tokenized_corpus:
#         for word in set(tokens):
#             doc_freq[word] = doc_freq.get(word, 0) + 1
#     return {word: math.log(N / df) for word, df in doc_freq.items()}
#
# def tfidf_score(word, tokens, idf):
#     tf = compute_tf(tokens)
#     return tf.get(word, 0.0) * idf.get(word, 0.0)
#
# def search_tfidf(query, corpus, top_k=3):
#     def cosine(a, b):
#         dot = sum(a.get(w, 0) * b.get(w, 0) for w in a)
#         ma = math.sqrt(sum(v**2 for v in a.values()))
#         mb = math.sqrt(sum(v**2 for v in b.values()))
#         return dot / (ma * mb) if ma and mb else 0.0
#     tokenized = [preprocess(doc) for doc in corpus]
#     idf = compute_idf(tokenized)
#     vecs = [{w: compute_tf(t).get(w, 0) * idf.get(w, 0) for w in t} for t in tokenized]
#     q_t = preprocess(query)
#     q_tf = compute_tf(q_t)
#     q_vec = {w: q_tf[w] * idf.get(w, 0) for w in q_tf if w in idf}
#     scored = sorted(enumerate(cosine(q_vec, v) for v in vecs), key=lambda x: -x[1])
#     return list(scored)[:top_k]

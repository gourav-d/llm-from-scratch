"""
Example 01: TF-IDF from Scratch
Module 10.5: RAG Without Vectors

Builds TF-IDF indexing and search with zero libraries (pure Python + math).
Shows every calculation step so you understand what TF-IDF actually computes.

Run:  python example_01_tfidf.py
Deps: none (pure Python)
"""

import math
from collections import Counter


# ─────────────────────────────────────────────────────────
# STEP 1: Corpus
# ─────────────────────────────────────────────────────────

# A small corpus of 6 documents (imagine these are chunks from your docs)
CORPUS = [
    {"id": "doc1", "text": "Python is a popular programming language for machine learning"},
    {"id": "doc2", "text": "Machine learning uses neural networks and data science"},
    {"id": "doc3", "text": "Python has simple syntax and is easy to learn"},
    {"id": "doc4", "text": "Neural networks require lots of training data to work well"},
    {"id": "doc5", "text": "Data science involves statistics and programming skills"},
    {"id": "doc6", "text": "JavaScript is used for web development and front end"},
]

# Common words that carry no useful meaning — we remove these
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "to", "for", "of", "and", "or", "but", "in", "on", "at",
    "it", "its", "this", "that", "with", "has", "have", "had",
    "not", "by", "from", "as", "so", "do", "does"
}


# ─────────────────────────────────────────────────────────
# STEP 2: Preprocessing
# ─────────────────────────────────────────────────────────

def preprocess(text: str) -> list[str]:
    """
    Lowercase → split on spaces → remove stop words → remove short tokens.
    Returns list of meaningful word tokens.
    """
    tokens = text.lower().split()
    return [t.strip(".,!?;:") for t in tokens
            if t.strip(".,!?;:") not in STOP_WORDS
            and len(t.strip(".,!?;:")) > 1]


# ─────────────────────────────────────────────────────────
# STEP 3: Term Frequency
# ─────────────────────────────────────────────────────────

def compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    TF(word, doc) = count of word in doc / total words in doc

    Example: ["cat", "cat", "sat"] → {"cat": 0.667, "sat": 0.333}
    """
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counts.items()}


# ─────────────────────────────────────────────────────────
# STEP 4: Inverse Document Frequency
# ─────────────────────────────────────────────────────────

def compute_idf(tokenized_corpus: list[list[str]]) -> dict[str, float]:
    """
    IDF(word) = log(total_docs / docs_containing_word)

    Words in every doc → IDF = 0 (useless).
    Words in 1 doc     → IDF = log(N) (highly distinctive).
    """
    N = len(tokenized_corpus)
    # Count how many documents contain each word
    doc_freq = {}
    for tokens in tokenized_corpus:
        for word in set(tokens):   # set: count doc once even if word appears many times
            doc_freq[word] = doc_freq.get(word, 0) + 1

    return {word: math.log(N / df) for word, df in doc_freq.items()}


# ─────────────────────────────────────────────────────────
# STEP 5: TF-IDF Matrix
# ─────────────────────────────────────────────────────────

def build_tfidf_index(corpus: list[dict]) -> tuple:
    """
    Build the full TF-IDF index.

    Returns:
        tokenized:   list of token lists for each doc
        idf:         dict word → IDF score
        tfidf_vecs:  list of dicts — word → TF-IDF score for each doc
    """
    tokenized = [preprocess(doc["text"]) for doc in corpus]
    idf = compute_idf(tokenized)

    tfidf_vecs = []
    for tokens in tokenized:
        tf = compute_tf(tokens)
        vec = {word: tf[word] * idf.get(word, 0) for word in tf}
        tfidf_vecs.append(vec)

    return tokenized, idf, tfidf_vecs


# ─────────────────────────────────────────────────────────
# STEP 6: Cosine Similarity
# ─────────────────────────────────────────────────────────

def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """
    Cosine similarity between two sparse TF-IDF vectors.

    cos(A, B) = dot(A, B) / (|A| × |B|)

    Sparse vectors stored as dicts — only non-zero terms stored.
    """
    # Dot product: only words in both vectors contribute
    dot = sum(vec_a.get(w, 0) * vec_b.get(w, 0) for w in vec_a)

    # Magnitudes
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────────────────
# STEP 7: Search
# ─────────────────────────────────────────────────────────

def search(query: str, corpus: list[dict], tfidf_vecs: list[dict],
           idf: dict, top_k: int = 3) -> list[tuple]:
    """
    TF-IDF search: score all docs against the query vector, return top-K.
    """
    query_tokens = preprocess(query)
    query_tf = compute_tf(query_tokens)
    query_vec = {word: query_tf[word] * idf.get(word, 0)
                 for word in query_tf if word in idf}

    scored = []
    for i, doc_vec in enumerate(tfidf_vecs):
        score = cosine_similarity(query_vec, doc_vec)
        scored.append((corpus[i], score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────────────────
# DEMO: Show Every Step
# ─────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


print_section("STEP 1: Preprocessing")
for doc in CORPUS:
    tokens = preprocess(doc["text"])
    print(f"  {doc['id']}: {tokens}")


print_section("STEP 2: Term Frequency for doc1")
doc1_tokens = preprocess(CORPUS[0]["text"])
tf = compute_tf(doc1_tokens)
print(f"  Text: {CORPUS[0]['text']}")
print(f"  Tokens: {doc1_tokens}")
print(f"  TF scores:")
for word, score in sorted(tf.items(), key=lambda x: -x[1]):
    bar = "█" * int(score * 100)
    print(f"    {word:<15} {score:.4f}  {bar}")


print_section("STEP 3: IDF Scores (across all 6 docs)")
tokenized = [preprocess(doc["text"]) for doc in CORPUS]
idf = compute_idf(tokenized)
print(f"  {'Word':<20} IDF     (high = rare = distinctive)")
print(f"  {'-'*45}")
for word, score in sorted(idf.items(), key=lambda x: -x[1])[:12]:
    bar = "█" * int(score * 10)
    print(f"  {word:<20} {score:.4f}  {bar}")


print_section("STEP 4: TF-IDF Matrix (top words per doc)")
_, _, tfidf_vecs = build_tfidf_index(CORPUS)
for i, (doc, vec) in enumerate(zip(CORPUS, tfidf_vecs)):
    top_words = sorted(vec.items(), key=lambda x: -x[1])[:3]
    top_str = ", ".join(f"{w}={s:.3f}" for w, s in top_words)
    print(f"  {doc['id']}: {top_str}")
print(f"\n  LESSON: Top TF-IDF words = most distinctive words per document.")
print(f"  Words appearing in every doc (like 'is') score 0 (IDF=0).")


print_section("STEP 5: Search Queries")
_, idf, tfidf_vecs = build_tfidf_index(CORPUS)

queries = [
    "machine learning python",
    "neural network training",
    "web development javascript",
    "data science statistics",
]

for query in queries:
    results = search(query, CORPUS, tfidf_vecs, idf, top_k=3)
    print(f"\n  Query: '{query}'")
    for rank, (doc, score) in enumerate(results, 1):
        bar = "█" * int(score * 30)
        print(f"    {rank}. [{doc['id']}] score={score:.4f}  {bar}")
        print(f"       {doc['text'][:60]}")


print_section("STEP 6: Vocabulary Mismatch Demo")
print("""
  TF-IDF CANNOT match synonyms.
  Query: "quick automobile"
  Document: "fast car"
  → These share no words → score = 0
""")
vocab_query = "quick automobile"
results = search(vocab_query, CORPUS, tfidf_vecs, idf, top_k=6)
print(f"  Query: '{vocab_query}'")
print(f"  All scores (none should match well):")
for doc, score in results:
    print(f"    [{doc['id']}] {score:.4f}  {doc['text'][:50]}")
print(f"\n  Compare with: 'python programming language'")
results2 = search("python programming language", CORPUS, tfidf_vecs, idf, top_k=3)
for doc, score in results2:
    print(f"    [{doc['id']}] {score:.4f}  {doc['text'][:50]}")
print(f"\n  LESSON: Exact keyword match works great. Synonym mismatch fails.")
print(f"  Solution: hybrid search (Lesson 4) or semantic search (Module 10.8)")

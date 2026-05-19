# -*- coding: utf-8 -*-
# exercise_05_full_pipeline.py
#
# Module 10.8 -- Semantic Search Systems
# Exercise 5: Wire Up the Full Pipeline
#
# TASK:
#   Implement a SearchEngine class with:
#     - index(docs)         -- encode all documents and build the index
#     - search(query, k=5)  -- retrieve, merge, rerank, return top-k
#
# The class must:
#   1. index():  encode each doc to a vector, build a BM25 index
#   2. search(): encode query -> BM25 retrieve top-20 -> semantic retrieve top-20
#                -> RRF merge -> cross-encoder re-rank -> return top-k
#
# PROVIDED:
#   - Word vector table
#   - Helper functions (embed, cosine_similarity, normalize_scores)
#   - 20 test documents
#   - Test queries with expected top results
#
# HOW TO RUN:
#   python exercises/exercise_05_full_pipeline.py

import numpy as np
import math


# ============================================================
# PROVIDED: Word Vectors and Helpers
# ============================================================

WORD_VECS = {
    # Cooking
    "cook":       np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "cooking":    np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "recipe":     np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "kitchen":    np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "ingredients":np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "food":       np.array([0.8, 0.0, 0.1, 0.0, 0.0, 0.0]),
    "baking":     np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "meal":       np.array([0.7, 0.0, 0.1, 0.0, 0.0, 0.0]),
    "restaurant": np.array([0.7, 0.0, 0.0, 0.0, 0.1, 0.0]),
    "chef":       np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    # Python / programming
    "python":     np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
    "programming":np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
    "code":       np.array([0.0, 0.8, 0.0, 0.1, 0.0, 0.0]),
    "software":   np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
    "algorithm":  np.array([0.0, 0.8, 0.0, 0.3, 0.0, 0.0]),
    "developer":  np.array([0.0, 0.8, 0.0, 0.0, 0.1, 0.0]),
    "tutorial":   np.array([0.0, 0.6, 0.0, 0.0, 0.0, 0.0]),
    "beginners":  np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
    "learning":   np.array([0.0, 0.5, 0.0, 0.2, 0.0, 0.0]),
    "machine":    np.array([0.0, 0.6, 0.0, 0.5, 0.0, 0.0]),
    "neural":     np.array([0.0, 0.5, 0.0, 0.6, 0.0, 0.0]),
    "network":    np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.0]),
    # Health / fitness
    "exercise":   np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "fitness":    np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "workout":    np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "health":     np.array([0.1, 0.0, 0.8, 0.1, 0.0, 0.0]),
    "running":    np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "gym":        np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "yoga":       np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0]),
    "muscles":    np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0]),
    # Science
    "science":    np.array([0.0, 0.1, 0.0, 0.9, 0.0, 0.0]),
    "physics":    np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "biology":    np.array([0.0, 0.0, 0.1, 0.9, 0.0, 0.0]),
    "chemistry":  np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "research":   np.array([0.0, 0.1, 0.0, 0.8, 0.0, 0.0]),
    "experiment": np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0]),
    # Travel
    "travel":     np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
    "vacation":   np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
    "hotel":      np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.0]),
    "flight":     np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
    "destination":np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
    "tour":       np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.1]),
    "country":    np.array([0.0, 0.0, 0.0, 0.1, 0.6, 0.4]),
    # History
    "history":    np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.9]),
    "ancient":    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9]),
    "civilization":np.array([0.0,0.0, 0.0, 0.0, 0.1, 0.9]),
    "war":        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8]),
    "culture":    np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.7]),
    # Common words (neutral)
    "the":np.zeros(6), "a":np.zeros(6), "and":np.zeros(6),
    "to": np.zeros(6), "of":np.zeros(6), "in": np.zeros(6),
    "for":np.zeros(6), "with":np.zeros(6), "how":np.zeros(6),
    "guide":      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "tips":       np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "best":       np.zeros(6), "top": np.zeros(6),
    "introduction":np.zeros(6), "advanced":np.array([0.0,0.1,0.0,0.1,0.0,0.0]),
    "using":np.zeros(6), "from":np.zeros(6),
}


def embed(text):
    """Average word vectors to get a sentence embedding."""
    words = text.lower().split()
    vecs = [WORD_VECS.get(w, np.zeros(6)) for w in words]
    if not vecs:
        return np.zeros(6)
    return np.mean(np.stack(vecs), axis=0)


def cosine_similarity(a, b):
    """Cosine similarity. Returns float in [0, 1] (or [-1,1] for raw)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_scores(scores_dict):
    """Min-max normalize a {doc_id: score} dict to [0, 1]."""
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    lo, hi = min(vals), max(vals)
    r = hi - lo
    if r == 0:
        return {k: 0.5 for k in scores_dict}
    return {k: (v - lo) / r for k, v in scores_dict.items()}


# ============================================================
# YOUR TASK: Implement the SearchEngine class
# ============================================================

class SearchEngine:
    """
    A full retrieve-rerank search engine.

    METHODS TO IMPLEMENT:
      index(docs)      -- Build the search index from a list of documents
      search(query, k) -- Search the index and return top-k results

    INTERNAL METHODS (provided or you implement):
      _bm25_score(query, doc_idx)    -- BM25 score for one document
      _semantic_score(query_vec, doc_id) -- Cosine similarity
      _rrf(ranked_lists)             -- Reciprocal Rank Fusion
      _cross_score(query, doc_text)  -- Cross-encoder simulation
    """

    def __init__(self):
        """Initialize all data stores to empty."""
        # Storage for indexed data
        self.documents = {}          # {doc_id: text}
        self.doc_vectors = {}        # {doc_id: numpy_array}

        # BM25 data structures
        self.bm25_corpus = []        # List of tokenized docs
        self.bm25_doc_ids = []       # Parallel list of doc IDs
        self.bm25_df = {}            # Term document frequency
        self.bm25_idf = {}           # Term IDF scores
        self.bm25_doc_lengths = []   # Document lengths (word counts)
        self.bm25_avgdl = 0.0        # Average document length
        self.bm25_k1 = 1.5           # BM25 saturation parameter
        self.bm25_b = 0.75           # BM25 length normalization

    # ----------------------------------------------------------
    # METHOD 1: index(docs)
    # ----------------------------------------------------------
    def index(self, docs):
        """
        Index documents for search.
        This is the OFFLINE step -- run once before any queries.

        Parameters:
            docs (list): List of {"id": str, "text": str} dicts

        STEPS TO IMPLEMENT:
        For each document in docs:
          1. Store doc["text"] in self.documents[doc["id"]]
          2. Encode it: self.doc_vectors[doc["id"]] = embed(doc["text"])
          3. Tokenize it: tokens = doc["text"].lower().split()
          4. Append tokens to self.bm25_corpus
          5. Append doc["id"] to self.bm25_doc_ids

        After processing all docs:
          6. Compute self.bm25_doc_lengths (length of each token list)
          7. Compute self.bm25_avgdl (average length)
          8. Compute self.bm25_df (document frequency: for each term, how many docs contain it)
             HINT: for each doc's tokens, use set(tokens) to get unique terms
          9. Compute self.bm25_idf using Robertson-Walker formula:
             idf(term) = log((N - df + 0.5) / (df + 0.5) + 1)
             where N = total number of docs, df = document frequency of term
        """
        # TODO: Implement this method
        # Clear previous index
        self.documents = {}
        self.doc_vectors = {}
        self.bm25_corpus = []
        self.bm25_doc_ids = []

        # Steps 1-5: Process each document
        for doc in docs:
            # TODO: Store text
            # YOUR CODE HERE

            # TODO: Encode to vector
            # YOUR CODE HERE

            # TODO: Tokenize and store
            # YOUR CODE HERE
            pass

        # Steps 6-7: Document lengths and average
        # YOUR CODE HERE
        N = len(self.bm25_corpus)

        # Step 8: Document frequency
        self.bm25_df = {}
        # YOUR CODE HERE

        # Step 9: IDF
        self.bm25_idf = {}
        # YOUR CODE HERE

    # ----------------------------------------------------------
    # METHOD 2: search(query, k)
    # ----------------------------------------------------------
    def search(self, query, k=5):
        """
        Search the indexed documents for the given query.

        PIPELINE:
        Step 1: Encode query --> query_vector
        Step 2: BM25 retrieve top-20 --> bm25_top20 = [(doc_id, score), ...]
        Step 3: Semantic retrieve top-20 --> sem_top20 = [(doc_id, score), ...]
        Step 4: RRF merge bm25_top20 and sem_top20 --> merged candidates
        Step 5: Cross-encoder re-rank top-10 merged candidates --> reranked
        Step 6: Return top-k as [(rank, doc_id, score, text), ...]

        Parameters:
            query (str): The user's search query
            k     (int): Number of results to return

        Returns:
            List of (rank, doc_id, score, text) tuples
        """
        # TODO: Implement this method

        # Step 1: Encode query
        query_vector = None  # YOUR CODE HERE: query_vector = embed(query)

        # Step 2: BM25 top-20
        bm25_top20 = []  # YOUR CODE HERE: call self._bm25_retrieve(query, 20)

        # Step 3: Semantic top-20
        sem_top20 = []   # YOUR CODE HERE: call self._semantic_retrieve(query_vector, 20)

        # Step 4: RRF merge
        merged = []      # YOUR CODE HERE: call self._rrf([bm25_top20, sem_top20])
        # merged is a list of (doc_id, rrf_score) sorted descending

        # Step 5: Cross-encoder re-rank top-10 of merged
        candidates = merged[:10] if len(merged) >= 10 else merged
        reranked = self._cross_rerank(query, candidates, top_n=k)
        # reranked is a list of (doc_id, cross_score) sorted descending

        # Step 6: Build return list
        results = []
        for rank, (doc_id, score) in enumerate(reranked, start=1):
            text = self.documents.get(doc_id, "")
            results.append((rank, doc_id, score, text))
        return results

    # ----------------------------------------------------------
    # INTERNAL METHODS (PROVIDED -- do not modify)
    # ----------------------------------------------------------

    def _bm25_retrieve(self, query, top_k):
        """BM25 scoring for all docs, return top_k."""
        query_terms = query.lower().split()
        scores = {}
        for i, tokens in enumerate(self.bm25_corpus):
            doc_id = self.bm25_doc_ids[i]
            dl = self.bm25_doc_lengths[i]
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in query_terms:
                if term not in self.bm25_idf:
                    continue
                tf_val = tf.get(term, 0)
                if tf_val == 0:
                    continue
                idf = self.bm25_idf[term]
                ln = 1 - self.bm25_b + self.bm25_b * (dl / max(self.bm25_avgdl, 1))
                score += idf * (tf_val * (self.bm25_k1 + 1)) / (tf_val + self.bm25_k1 * ln)
            scores[doc_id] = score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _semantic_retrieve(self, query_vec, top_k):
        """Cosine similarity ranking, return top_k."""
        scores = {
            doc_id: cosine_similarity(query_vec, vec)
            for doc_id, vec in self.doc_vectors.items()
        }
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _rrf(self, ranked_lists, k=60):
        """Reciprocal Rank Fusion of multiple ranked lists."""
        scores = {}
        for ranked in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _cross_rerank(self, query, candidates, top_n):
        """Simple cross-encoder re-rank using token overlap and phrase matching."""
        query_tokens = set(query.lower().split())
        query_lower = query.lower()
        scored = []
        for doc_id, _ in candidates:
            doc_text = self.documents.get(doc_id, "")
            doc_tokens = set(doc_text.lower().split())
            doc_lower = doc_text.lower()
            shared = query_tokens.intersection(doc_tokens)
            overlap = len(shared) / max(len(query_tokens), 1)
            focus = len(shared) / max(len(doc_tokens), 1)
            phrase_bonus = 0.0
            qwords = query_lower.split()
            for i in range(len(qwords) - 1):
                phrase = qwords[i] + " " + qwords[i+1]
                if phrase in doc_lower:
                    phrase_bonus += 0.1
            cross = min(1.0, 0.5 * overlap + 0.2 * focus + phrase_bonus)
            scored.append((doc_id, cross))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


# ============================================================
# TEST DOCUMENTS (20 items)
# ============================================================

TEST_DOCS = [
    {"id": "food01",  "text": "Pasta recipe with tomato sauce and fresh ingredients"},
    {"id": "food02",  "text": "How to bake bread at home in kitchen"},
    {"id": "food03",  "text": "Restaurant chef cooking techniques and meal guide"},
    {"id": "food04",  "text": "Healthy food meal prep for the week"},
    {"id": "py01",    "text": "Python programming tutorial for beginners guide"},
    {"id": "py02",    "text": "Machine learning with Python neural network code"},
    {"id": "py03",    "text": "Python algorithm and software developer guide"},
    {"id": "py04",    "text": "Learning Python from scratch beginners tutorial"},
    {"id": "fit01",   "text": "Exercise workout gym fitness routine for muscles"},
    {"id": "fit02",   "text": "Running fitness and yoga health guide"},
    {"id": "fit03",   "text": "Gym workout and exercise health tips"},
    {"id": "fit04",   "text": "Health running and fitness for beginners"},
    {"id": "sci01",   "text": "Physics science research experiment and biology"},
    {"id": "sci02",   "text": "Chemistry biology research and experiment guide"},
    {"id": "sci03",   "text": "Science algorithm and research experiment findings"},
    {"id": "trv01",   "text": "Travel vacation hotel flight destination guide"},
    {"id": "trv02",   "text": "Best travel destination and country to tour"},
    {"id": "hist01",  "text": "Ancient history civilization war and culture"},
    {"id": "hist02",  "text": "History ancient civilization and culture guide"},
    {"id": "mix01",   "text": "Python science algorithm and research code guide"},
]

# Expected relevant docs for each test query
TEST_QUERIES = [
    {
        "query": "python machine learning tutorial",
        "relevant": {"py01", "py02", "py03", "py04", "mix01"},
        "description": "Python programming query"
    },
    {
        "query": "exercise gym workout health",
        "relevant": {"fit01", "fit02", "fit03", "fit04"},
        "description": "Fitness query"
    },
    {
        "query": "cooking recipe food meal",
        "relevant": {"food01", "food02", "food03", "food04"},
        "description": "Food query"
    },
    {
        "query": "travel vacation flight hotel",
        "relevant": {"trv01", "trv02"},
        "description": "Travel query"
    },
]


# ============================================================
# TESTS -- Do not modify
# ============================================================

def precision_at_k(results, relevant_set, k):
    """Fraction of top-k results that are in the relevant set."""
    top_k_ids = {doc_id for _, doc_id, _, _ in results[:k]}
    found = len(top_k_ids.intersection(relevant_set))
    return found / k if k > 0 else 0.0


def run_tests(engine):
    """Test the SearchEngine on all test queries."""
    print("\nRunning tests...")
    passed = 0
    failed = 0

    for test in TEST_QUERIES:
        query = test["query"]
        relevant = test["relevant"]
        desc = test["description"]

        results = engine.search(query, k=5)

        # Check result format
        if not results or not isinstance(results[0], tuple) or len(results[0]) != 4:
            print(f"[FAIL] {desc}: search() should return list of (rank, doc_id, score, text)")
            failed += 1
            continue

        # Check precision
        p5 = precision_at_k(results, relevant, k=5)
        if p5 >= 0.6:
            print(f"[PASS] {desc}: Precision@5 = {p5:.2f}")
            passed += 1
        else:
            print(f"[FAIL] {desc}: Precision@5 = {p5:.2f} (need >= 0.6)")
            # Show what was returned
            for rank, doc_id, score, text in results[:3]:
                print(f"       Rank {rank}: {doc_id} -- {text[:45]}")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# MAIN DEMO
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  EXERCISE 5: Full Search Pipeline")
    print("=" * 65)

    engine = SearchEngine()

    # Index the documents
    print("\nIndexing 20 documents...")
    try:
        engine.index(TEST_DOCS)

        # Check if index was built
        if len(engine.documents) == 0:
            print("NOTE: index() did not store any documents.")
            print("Fill in the TODO sections in index(), then re-run.")
        else:
            print(f"Indexed {len(engine.documents)} documents.")

            # Run test queries
            for test in TEST_QUERIES:
                query = test["query"]
                print(f"\n{'='*65}")
                print(f"Query: '{query}'")
                results = engine.search(query, k=5)

                if results and results[0][2] > 0:
                    print("Top 5 results:")
                    for rank, doc_id, score, text in results:
                        relevance = "[RELEVANT]" if doc_id in test["relevant"] else ""
                        print(f"  Rank {rank}: [{score:.3f}] {text[:50]} {relevance}")
                else:
                    print("  (No results -- check your index() implementation)")

            # Run automated tests
            print("\n" + "=" * 65)
            all_passed = run_tests(engine)
            if all_passed:
                print("\nAll tests passed! Your SearchEngine pipeline is working.")
            else:
                print("\nSome tests failed. Review the TODO sections.")

    except Exception as ex:
        print(f"\nError: {ex}")
        import traceback
        traceback.print_exc()
        print("\nFill in the TODO sections above.")


# ============================================================
# SOLUTION (Hidden -- Try yourself first!)
# ============================================================
# def index(self, docs):
#     self.documents = {}
#     self.doc_vectors = {}
#     self.bm25_corpus = []
#     self.bm25_doc_ids = []
#     for doc in docs:
#         self.documents[doc["id"]] = doc["text"]
#         self.doc_vectors[doc["id"]] = embed(doc["text"])
#         tokens = doc["text"].lower().split()
#         self.bm25_corpus.append(tokens)
#         self.bm25_doc_ids.append(doc["id"])
#     N = len(self.bm25_corpus)
#     self.bm25_doc_lengths = [len(t) for t in self.bm25_corpus]
#     self.bm25_avgdl = sum(self.bm25_doc_lengths) / N if N else 1.0
#     self.bm25_df = {}
#     for tokens in self.bm25_corpus:
#         for term in set(tokens):
#             self.bm25_df[term] = self.bm25_df.get(term, 0) + 1
#     self.bm25_idf = {}
#     for term, df in self.bm25_df.items():
#         self.bm25_idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
#
# def search(self, query, k=5):
#     query_vector = embed(query)
#     bm25_top20 = self._bm25_retrieve(query, 20)
#     sem_top20 = self._semantic_retrieve(query_vector, 20)
#     merged = self._rrf([bm25_top20, sem_top20])
#     candidates = merged[:10]
#     reranked = self._cross_rerank(query, candidates, top_n=k)
#     results = []
#     for rank, (doc_id, score) in enumerate(reranked, start=1):
#         text = self.documents.get(doc_id, "")
#         results.append((rank, doc_id, score, text))
#     return results

# -*- coding: utf-8 -*-
# example_06_full_pipeline.py
#
# Module 10.8 -- Semantic Search Systems
# Lesson 6: Full Retrieve-Rerank Pipeline
#
# WHAT THIS FILE DEMONSTRATES:
#   - A complete production-style search pipeline end to end
#   - Offline indexing: encode 20 documents, build BM25 and vector indexes
#   - Online querying: embed query -> BM25 retrieve + vector retrieve -> RRF merge
#     -> cross-encoder re-rank -> return top results
#   - Detailed trace output showing what happens at each step
#   - Precision@K evaluation metric
#
# REQUIREMENTS: numpy only (Tier 1 - primary)
# PART B needs: pip install sentence-transformers rank-bm25
#
# HOW TO RUN:
#   python examples/example_06_full_pipeline.py

import numpy as np      # For vector math
import math             # For BM25 IDF formula
import time             # For timing each stage


# ============================================================
# PART 1: Component Implementations (reused from earlier examples)
# ============================================================

def tokenize(text):
    """Split text into lowercase tokens."""
    return text.lower().split()


def embed(text, word_vecs):
    """
    Convert text to a vector by averaging word vectors.

    Parameters:
        text      (str):  Text to encode
        word_vecs (dict): {word: numpy_array} lookup table

    Returns:
        numpy array: Mean of word vectors
    """
    words = tokenize(text)
    vecs = []
    for word in words:
        vec = word_vecs.get(word)        # Look up word vector
        if vec is not None:
            vecs.append(vec)
    if not vecs:
        return np.zeros(6)               # Return zero vector if no words found
    return np.mean(np.stack(vecs), axis=0)   # Average all word vectors


def cosine_similarity(a, b):
    """Cosine similarity between two numpy arrays."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_scores(scores_dict):
    """Normalize a {doc_id: score} dict to [0, 1] range."""
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    lo, hi = min(vals), max(vals)
    r = hi - lo
    if r == 0:
        return {k: 0.5 for k in scores_dict}
    return {k: (v - lo) / r for k, v in scores_dict.items()}


# ============================================================
# PART 2: Word Vectors (Meaning Lookup Table)
# ============================================================
# 6 dimensions: [programming, fitness, food, science, travel, history]
WORD_VECS = {
    # Programming / tech
    "python":       np.array([0.9, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "programming":  np.array([0.9, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "code":         np.array([0.9, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "software":     np.array([0.9, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "algorithm":    np.array([0.8, 0.0, 0.0, 0.3, 0.0, 0.0]),
    "data":         np.array([0.7, 0.0, 0.0, 0.4, 0.0, 0.0]),
    "machine":      np.array([0.7, 0.0, 0.0, 0.4, 0.0, 0.0]),
    "learning":     np.array([0.6, 0.0, 0.0, 0.3, 0.0, 0.1]),
    "ai":           np.array([0.7, 0.0, 0.0, 0.5, 0.0, 0.0]),
    "neural":       np.array([0.6, 0.0, 0.0, 0.6, 0.0, 0.0]),
    "network":      np.array([0.6, 0.0, 0.0, 0.4, 0.0, 0.0]),
    "database":     np.array([0.8, 0.0, 0.0, 0.2, 0.0, 0.0]),
    "web":          np.array([0.7, 0.0, 0.0, 0.1, 0.2, 0.0]),
    "api":          np.array([0.8, 0.0, 0.0, 0.1, 0.0, 0.0]),
    # Fitness / health
    "exercise":     np.array([0.0, 0.9, 0.0, 0.1, 0.0, 0.0]),
    "fitness":      np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
    "workout":      np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
    "running":      np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
    "yoga":         np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.1]),
    "gym":          np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
    "health":       np.array([0.0, 0.7, 0.2, 0.2, 0.0, 0.0]),
    "diet":         np.array([0.0, 0.4, 0.6, 0.0, 0.0, 0.0]),
    "muscles":      np.array([0.0, 0.8, 0.1, 0.0, 0.0, 0.0]),
    # Food
    "cooking":      np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "recipe":       np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "food":         np.array([0.0, 0.1, 0.9, 0.0, 0.0, 0.0]),
    "baking":       np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "restaurant":   np.array([0.0, 0.0, 0.7, 0.0, 0.2, 0.0]),
    "kitchen":      np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "ingredients":  np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "meal":         np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    # Science
    "science":      np.array([0.1, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "physics":      np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "chemistry":    np.array([0.0, 0.0, 0.1, 0.9, 0.0, 0.0]),
    "biology":      np.array([0.0, 0.1, 0.0, 0.9, 0.0, 0.0]),
    "research":     np.array([0.1, 0.0, 0.0, 0.9, 0.0, 0.1]),
    "experiment":   np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "discovery":    np.array([0.0, 0.0, 0.0, 0.8, 0.1, 0.1]),
    # Travel
    "travel":       np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.1]),
    "destination":  np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
    "hotel":        np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.0]),
    "flight":       np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
    "vacation":     np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
    "tour":         np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.1]),
    "country":      np.array([0.0, 0.0, 0.0, 0.1, 0.6, 0.4]),
    "city":         np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.4]),
    # History / culture
    "history":      np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.9]),
    "ancient":      np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.9]),
    "war":          np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8]),
    "civilization": np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.9]),
    "culture":      np.array([0.0, 0.0, 0.1, 0.0, 0.3, 0.7]),
    "art":          np.array([0.0, 0.0, 0.1, 0.0, 0.2, 0.7]),
    "museum":       np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.7]),
    # Common words
    "the": np.zeros(6), "a": np.zeros(6), "and": np.zeros(6),
    "to":  np.zeros(6), "of": np.zeros(6), "in":  np.zeros(6),
    "for": np.zeros(6), "how": np.zeros(6), "with": np.zeros(6),
    "guide":        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "tips":         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "best":         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "top":          np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "introduction": np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.1]),
    "advanced":     np.array([0.1, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "using":        np.zeros(6), "from": np.zeros(6),
    "beginners":    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
}


# ============================================================
# PART 3: The Search Pipeline Class
# ============================================================

class SearchPipeline:
    """
    A complete retrieve-then-rerank search pipeline.

    Components:
        1. BM25 index for keyword retrieval
        2. Vector index for semantic retrieval
        3. RRF fusion to merge both ranked lists
        4. Simple cross-encoder to re-rank the merged candidates

    In C# terms: this is like a service class with an Index() method
    and a Search() method. You call Index() once, then Search() many times.
    """

    def __init__(self, word_vecs, k1=1.5, b=0.75):
        """
        Initialize the pipeline.

        Parameters:
            word_vecs (dict): Word vector lookup table
            k1        (float): BM25 term saturation
            b         (float): BM25 length normalization
        """
        self.word_vecs = word_vecs     # Word vector table
        self.k1 = k1                   # BM25 parameter
        self.b = b                     # BM25 parameter

        # Storage for indexed data
        self.documents = {}            # {doc_id: text}
        self.doc_vectors = {}          # {doc_id: numpy_array} (semantic)
        self.bm25_corpus = []          # List of tokenized documents for BM25
        self.bm25_doc_ids = []         # Parallel list of doc IDs for BM25
        self.bm25_df = {}              # BM25 document frequency
        self.bm25_idf = {}             # BM25 IDF scores
        self.bm25_doc_lengths = []     # BM25 document lengths
        self.bm25_avgdl = 0.0          # BM25 average document length

    def index(self, documents):
        """
        Index a list of documents for both BM25 and semantic search.
        This is the OFFLINE step -- run once before any queries.

        Parameters:
            documents (list): List of {"id": str, "text": str} dicts
        """
        print(f"\n[Pipeline.index] Indexing {len(documents)} documents...")
        t_start = time.time()

        self.documents = {}
        self.doc_vectors = {}
        self.bm25_corpus = []
        self.bm25_doc_ids = []

        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"]

            # Store original text
            self.documents[doc_id] = text

            # --- Semantic: encode to vector ---
            self.doc_vectors[doc_id] = embed(text, self.word_vecs)

            # --- BM25: tokenize ---
            tokens = tokenize(text)
            self.bm25_corpus.append(tokens)
            self.bm25_doc_ids.append(doc_id)

        # Build BM25 statistics
        N = len(self.bm25_corpus)
        self.bm25_doc_lengths = [len(t) for t in self.bm25_corpus]
        self.bm25_avgdl = sum(self.bm25_doc_lengths) / N if N > 0 else 1.0

        # Document frequency
        self.bm25_df = {}
        for tokens in self.bm25_corpus:
            for term in set(tokens):
                self.bm25_df[term] = self.bm25_df.get(term, 0) + 1

        # IDF
        self.bm25_idf = {}
        for term, df_val in self.bm25_df.items():
            self.bm25_idf[term] = math.log(
                (N - df_val + 0.5) / (df_val + 0.5) + 1
            )

        elapsed = (time.time() - t_start) * 1000
        print(f"[Pipeline.index] Done. {len(self.documents)} docs indexed in {elapsed:.1f}ms")

    def _bm25_retrieve(self, query, top_k):
        """
        BM25 retrieval: score all docs by keyword relevance.
        Returns top_k (doc_id, score) tuples.
        """
        query_terms = tokenize(query)
        scores = {}

        for i, tokens in enumerate(self.bm25_corpus):
            doc_id = self.bm25_doc_ids[i]
            dl = self.bm25_doc_lengths[i]
            tf_map = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            score = 0.0
            for term in query_terms:
                if term not in self.bm25_idf:
                    continue
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                idf = self.bm25_idf[term]
                ln = 1 - self.b + self.b * (dl / self.bm25_avgdl)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * ln)
            scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _semantic_retrieve(self, query, top_k):
        """
        Semantic retrieval: score all docs by cosine similarity.
        Returns top_k (doc_id, score) tuples.
        """
        query_vec = embed(query, self.word_vecs)
        scores = {}
        for doc_id, doc_vec in self.doc_vectors.items():
            scores[doc_id] = cosine_similarity(query_vec, doc_vec)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _rrf_merge(self, *ranked_lists, k=60):
        """
        Merge multiple ranked lists using RRF.

        Parameters:
            *ranked_lists: Variable number of ranked lists, each [(doc_id, score), ...]
            k (int):       RRF constant (default 60)

        Returns:
            List of (doc_id, rrf_score) sorted by score descending
        """
        rrf = {}
        for ranked in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    def _cross_encoder_rerank(self, query, candidates, top_n):
        """
        Re-rank candidates using a simple cross-encoder simulation.
        Scores each (query, doc) pair based on token overlap and phrase matching.

        Parameters:
            query      (str):  Search query
            candidates (list): List of (doc_id, prior_score) tuples
            top_n      (int):  How many results to return after reranking

        Returns:
            List of (doc_id, cross_score) tuples
        """
        query_tokens = set(tokenize(query))
        query_lower = query.lower()
        reranked = []

        for doc_id, prior_score in candidates:
            doc_text = self.documents.get(doc_id, "")
            doc_tokens = set(tokenize(doc_text))
            doc_lower = doc_text.lower()

            # Factor 1: token overlap
            shared = query_tokens.intersection(doc_tokens)
            overlap = len(shared) / max(len(query_tokens), 1)

            # Factor 2: document focus
            focus = len(shared) / max(len(doc_tokens), 1)

            # Factor 3: phrase matching
            phrase_bonus = 0.0
            qwords = query_lower.split()
            for i in range(len(qwords) - 1):
                phrase = qwords[i] + " " + qwords[i+1]
                if phrase in doc_lower:
                    phrase_bonus += 0.1

            # Combine
            cross_score = min(1.0, 0.5 * overlap + 0.2 * focus + phrase_bonus)
            reranked.append((doc_id, cross_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_n]

    def search(self, query, top_k=5, retrieve_k=10, trace=True):
        """
        Full pipeline: retrieve candidates, merge, rerank, return top results.

        Parameters:
            query      (str):  The user's search query
            top_k      (int):  How many final results to return
            retrieve_k (int):  How many candidates to retrieve in each stage
            trace      (bool): Whether to print step-by-step output

        Returns:
            List of (rank, doc_id, score, text) tuples
        """
        if trace:
            print(f"\n{'='*65}")
            print(f"[Pipeline.search] Query: '{query}'")
            print(f"{'='*65}")

        t_total = time.time()

        # --- Step 1: BM25 retrieval ---
        t1 = time.time()
        bm25_results = self._bm25_retrieve(query, top_k=retrieve_k)
        t_bm25 = (time.time() - t1) * 1000

        if trace:
            print(f"\n[Step 1] BM25 retrieval ({t_bm25:.2f}ms) -- top {retrieve_k} candidates:")
            for rank, (doc_id, score) in enumerate(bm25_results[:5], start=1):
                print(f"  BM25 Rank {rank}: [{score:.3f}] {self.documents[doc_id][:55]}")

        # --- Step 2: Semantic retrieval ---
        t2 = time.time()
        sem_results = self._semantic_retrieve(query, top_k=retrieve_k)
        t_sem = (time.time() - t2) * 1000

        if trace:
            print(f"\n[Step 2] Semantic retrieval ({t_sem:.2f}ms) -- top {retrieve_k} candidates:")
            for rank, (doc_id, score) in enumerate(sem_results[:5], start=1):
                print(f"  Semantic Rank {rank}: [{score:.3f}] {self.documents[doc_id][:55]}")

        # --- Step 3: RRF merge ---
        t3 = time.time()
        merged = self._rrf_merge(bm25_results, sem_results)
        t_rrf = (time.time() - t3) * 1000

        if trace:
            print(f"\n[Step 3] RRF merge ({t_rrf:.2f}ms) -- {len(merged)} unique candidates:")
            for rank, (doc_id, score) in enumerate(merged[:5], start=1):
                print(f"  RRF Rank {rank}: [{score:.4f}] {self.documents[doc_id][:55]}")

        # --- Step 4: Cross-encoder re-rank ---
        # Re-rank the top retrieve_k merged candidates
        candidates_for_rerank = merged[:retrieve_k]
        t4 = time.time()
        reranked = self._cross_encoder_rerank(query, candidates_for_rerank, top_n=top_k)
        t_ce = (time.time() - t4) * 1000

        if trace:
            print(f"\n[Step 4] Cross-encoder re-rank ({t_ce:.2f}ms) -- final top {top_k}:")

        t_elapsed = (time.time() - t_total) * 1000

        # Build final result list
        final_results = []
        for rank, (doc_id, score) in enumerate(reranked, start=1):
            text = self.documents.get(doc_id, "")
            final_results.append((rank, doc_id, score, text))
            if trace:
                print(f"  Final Rank {rank}: [{score:.3f}] {text[:60]}")

        if trace:
            print(f"\n[Pipeline.search] Total time: {t_elapsed:.2f}ms")

        return final_results


# ============================================================
# PART 4: Evaluation Metric -- Precision@K
# ============================================================

def precision_at_k(results, relevant_doc_ids, k):
    """
    Compute Precision@K: of the top K results, what fraction are relevant?

    Example: if top-5 results include 3 relevant docs, Precision@5 = 0.6

    Parameters:
        results         (list): List of (rank, doc_id, score, text)
        relevant_doc_ids(set):  Set of doc IDs that are truly relevant
        k               (int):  How many results to evaluate

    Returns:
        float: Precision@K (0.0 to 1.0)
    """
    top_k_doc_ids = [doc_id for rank, doc_id, score, text in results[:k]]
    relevant_in_top_k = sum(1 for doc_id in top_k_doc_ids if doc_id in relevant_doc_ids)
    return relevant_in_top_k / k if k > 0 else 0.0


# ============================================================
# PART 5: Test Corpus (20 documents)
# ============================================================

TEST_DOCUMENTS = [
    {"id": "d01", "text": "Python programming tutorial for beginners step by step"},
    {"id": "d02", "text": "Machine learning algorithms explained with Python code"},
    {"id": "d03", "text": "Introduction to neural network architecture and training"},
    {"id": "d04", "text": "How to use Python for data analysis with pandas"},
    {"id": "d05", "text": "Web development with Python and Flask framework"},
    {"id": "d06", "text": "Database design and SQL programming guide"},
    {"id": "d07", "text": "Exercise fitness workout tips for building muscles"},
    {"id": "d08", "text": "Running training schedule for beginners marathon guide"},
    {"id": "d09", "text": "Yoga and gym workout for health and fitness"},
    {"id": "d10", "text": "Cooking recipe guide for healthy meal ingredients"},
    {"id": "d11", "text": "Baking bread and kitchen cooking tips for beginners"},
    {"id": "d12", "text": "Restaurant food and meal dining guide best tips"},
    {"id": "d13", "text": "Physics science research experiment discovery findings"},
    {"id": "d14", "text": "Biology research on health and diet science study"},
    {"id": "d15", "text": "Chemistry experiment and laboratory science discovery"},
    {"id": "d16", "text": "Travel destination vacation hotel flight guide tips"},
    {"id": "d17", "text": "Best city and country to visit for travel vacation"},
    {"id": "d18", "text": "History of ancient civilization war and culture art"},
    {"id": "d19", "text": "Museum history tour ancient culture art discovery"},
    {"id": "d20", "text": "Advanced Python machine learning neural network ai api"},
]


# ============================================================
# PART 6: Main Demo
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  FULL RETRIEVE-RERANK PIPELINE DEMO")
    print("=" * 65)

    # Create and build the pipeline (offline step)
    pipeline = SearchPipeline(word_vecs=WORD_VECS)
    pipeline.index(TEST_DOCUMENTS)

    # --------------------------------------------------------
    # Query 1: Python / machine learning
    # --------------------------------------------------------
    results_1 = pipeline.search(
        query="python machine learning tutorial",
        top_k=5,
        retrieve_k=10,
        trace=True
    )

    # Evaluate quality
    relevant_for_q1 = {"d01", "d02", "d03", "d04", "d20"}   # Ground truth relevant docs
    p_at_3 = precision_at_k(results_1, relevant_for_q1, k=3)
    p_at_5 = precision_at_k(results_1, relevant_for_q1, k=5)
    print(f"\nEvaluation: Precision@3 = {p_at_3:.2f}, Precision@5 = {p_at_5:.2f}")
    print(f"(1.0 = all top results are relevant; 0.6 = 3 out of 5 are relevant)")

    # --------------------------------------------------------
    # Query 2: Fitness / health
    # --------------------------------------------------------
    results_2 = pipeline.search(
        query="exercise workout and health training",
        top_k=5,
        retrieve_k=10,
        trace=True
    )

    relevant_for_q2 = {"d07", "d08", "d09", "d14"}
    p_at_3 = precision_at_k(results_2, relevant_for_q2, k=3)
    p_at_5 = precision_at_k(results_2, relevant_for_q2, k=5)
    print(f"\nEvaluation: Precision@3 = {p_at_3:.2f}, Precision@5 = {p_at_5:.2f}")

    # --------------------------------------------------------
    # Query 3: Travel history -- cross-topic
    # --------------------------------------------------------
    results_3 = pipeline.search(
        query="museum tour history ancient city travel",
        top_k=5,
        retrieve_k=10,
        trace=True
    )

    relevant_for_q3 = {"d16", "d17", "d18", "d19"}
    p_at_3 = precision_at_k(results_3, relevant_for_q3, k=3)
    p_at_5 = precision_at_k(results_3, relevant_for_q3, k=5)
    print(f"\nEvaluation: Precision@3 = {p_at_3:.2f}, Precision@5 = {p_at_5:.2f}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("PIPELINE SUMMARY")
    print("=" * 65)
    print("""
The pipeline ran these steps for each query:

  1. BM25 retrieval     -- fast keyword matching (1-5ms)
  2. Semantic retrieval -- fast vector similarity (1-5ms)
  3. RRF merge          -- combine both ranked lists (<1ms)
  4. Cross-encoder      -- accurate re-ranking of top candidates (10-20ms)
     (simulation only -- real cross-encoders take 50-200ms on CPU)

Total: ~20-30ms per query in this simulation.
Real production system with GPU: ~100-200ms per query.

The cross-encoder re-ranks candidates based on how well the document
DIRECTLY ANSWERS the query, not just whether they share words.
""")


# ============================================================
# PART B: Wire up real libraries
# ============================================================
# Remove the # to run. Requirements: pip install sentence-transformers rank-bm25
#
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from rank_bm25 import BM25Okapi
# import numpy as np
#
# REAL_CORPUS_TEXTS = [d["text"] for d in TEST_DOCUMENTS]
# REAL_CORPUS_IDS   = [d["id"]   for d in TEST_DOCUMENTS]
#
# # Real bi-encoder
# bi_model = SentenceTransformer('all-MiniLM-L6-v2')
# real_doc_vecs = bi_model.encode(REAL_CORPUS_TEXTS)  # (20, 384)
#
# # Real BM25
# real_bm25 = BM25Okapi([t.lower().split() for t in REAL_CORPUS_TEXTS])
#
# # Real cross-encoder
# ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
#
# query = "python machine learning tutorial"
#
# # Retrieve
# q_vec = bi_model.encode([query])                            # (1, 384)
# sem_scores = (real_doc_vecs @ q_vec.T).flatten()           # dot product shortcut
# bm25_scores = real_bm25.get_scores(query.lower().split())
#
# # Top 10 by semantic
# top10_idx = np.argsort(sem_scores)[::-1][:10]
# top10_docs = [REAL_CORPUS_TEXTS[i] for i in top10_idx]
#
# # Re-rank with cross-encoder
# pairs = [[query, doc] for doc in top10_docs]
# ce_scores = ce_model.predict(pairs)
# reranked_idx = np.argsort(ce_scores)[::-1]
#
# print("\nReal pipeline results:")
# for rank, i in enumerate(reranked_idx[:5], start=1):
#     original_idx = top10_idx[i]
#     print(f"  Rank {rank} [{ce_scores[i]:.3f}]: {REAL_CORPUS_TEXTS[original_idx]}")

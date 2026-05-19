# -*- coding: utf-8 -*-
# example_05_hybrid_search.py
#
# Module 10.8 -- Semantic Search Systems
# Lesson 5: Hybrid Search (BM25 + Semantic)
#
# WHAT THIS FILE DEMONSTRATES:
#   - BM25 scoring from scratch (term frequency, IDF, length normalization)
#   - Semantic scoring using simple word vectors (cosine similarity)
#   - Weighted combination hybrid: final = alpha * semantic + (1-alpha) * bm25
#   - RRF (Reciprocal Rank Fusion): rank-based combination
#   - Comparison: keyword-only vs semantic-only vs hybrid results
#
# REQUIREMENTS: numpy only (Tier 1 - primary)
# PART B needs: pip install rank-bm25 sentence-transformers
#
# HOW TO RUN:
#   python examples/example_05_hybrid_search.py

import numpy as np          # For vector math
import math                 # For log() in IDF calculation


# ============================================================
# PART 1: BM25 Implementation from Scratch
# ============================================================
# BM25 (Best Match 25) is the gold standard keyword scoring algorithm.
# It improves on simple TF-IDF in two ways:
#   1. Term frequency saturation: adding the same word 100 times does not
#      make a document 100x more relevant -- gains diminish
#   2. Document length normalization: shorter docs are rewarded slightly
#
# Formula for BM25 score of document d given query term t:
#
#   BM25(t, d) = IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
#
# where:
#   tf    = how many times term t appears in document d
#   IDF   = log((N - n_t + 0.5) / (n_t + 0.5) + 1)
#           N   = total number of documents
#           n_t = number of documents containing term t
#   dl    = length of document d (number of words)
#   avgdl = average document length across all documents
#   k1    = term saturation parameter (typically 1.5 -- controls how fast gains diminish)
#   b     = length normalization parameter (typically 0.75 -- 0 = no length norm)

class BM25:
    """
    BM25 scoring algorithm from scratch.

    In C# terms: think of this as a class with a list of IDocuments
    and methods to compute relevance scores.
    """

    def __init__(self, k1=1.5, b=0.75):
        """
        Initialize BM25 with standard parameters.

        Parameters:
            k1 (float): Term saturation parameter (1.2-2.0 is typical)
            b  (float): Length normalization (0 = off, 1 = full, 0.75 is typical)
        """
        self.k1 = k1                     # Term saturation: how fast gains diminish
        self.b = b                        # Length normalization weight

        # These are computed when you call fit()
        self.corpus = []                  # Original tokenized documents
        self.doc_lengths = []             # Length of each document (word count)
        self.avgdl = 0.0                  # Average document length
        self.N = 0                        # Number of documents
        self.df = {}                      # Document frequency: {term: count of docs containing it}
        self.idf = {}                     # IDF scores: {term: idf_value}

    def tokenize(self, text):
        """
        Simple tokenizer: lowercase and split on whitespace.
        In production you would use stemming and stop-word removal.

        Parameters:
            text (str): Text to tokenize

        Returns:
            list: List of lowercase tokens
        """
        return text.lower().split()

    def fit(self, documents):
        """
        Index the corpus -- precompute IDF values and statistics.
        Must be called before scoring.

        Parameters:
            documents (list): List of document strings
        """
        self.N = len(documents)            # Total document count

        # Tokenize all documents
        self.corpus = [self.tokenize(doc) for doc in documents]

        # Compute document lengths
        self.doc_lengths = [len(tokens) for tokens in self.corpus]

        # Compute average document length
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 1.0

        # Compute document frequency for each term
        # df[term] = number of documents containing this term
        self.df = {}
        for tokens in self.corpus:                  # For each document
            unique_tokens = set(tokens)              # Unique terms in this doc
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1   # Increment count

        # Compute IDF (Inverse Document Frequency) for each term
        # Terms appearing in many documents get low IDF (not very discriminating)
        # Terms appearing in few documents get high IDF (distinctive keywords)
        self.idf = {}
        for term, doc_count in self.df.items():
            # Robertson-Walker IDF formula (used in BM25)
            # Adding 1 inside log prevents very negative IDF for very common terms
            self.idf[term] = math.log(
                (self.N - doc_count + 0.5) / (doc_count + 0.5) + 1
            )

    def score(self, query, doc_index):
        """
        Compute BM25 score for one query-document pair.

        Parameters:
            query     (str): The search query
            doc_index (int): Index of the document to score

        Returns:
            float: BM25 relevance score (higher = more relevant)
        """
        query_terms = self.tokenize(query)             # Tokenize query
        doc_tokens = self.corpus[doc_index]            # Pre-tokenized document
        dl = self.doc_lengths[doc_index]               # This document's length

        # Count term frequencies in the document
        # tf[term] = how many times this term appears in the document
        tf = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1          # Increment count

        total_score = 0.0                              # Accumulate BM25 score

        for term in query_terms:                       # For each query term
            if term not in self.idf:                   # Term not in corpus?
                continue                               # Skip -- no IDF for it

            # Get term frequency in this document (0 if not present)
            term_freq = tf.get(term, 0)

            if term_freq == 0:                         # Term not in document?
                continue                               # Score contribution is 0

            # BM25 formula components:
            idf_val = self.idf[term]                   # IDF of this term

            # Length normalization factor
            # When b=0.75 and dl > avgdl: document is penalized slightly for being long
            length_norm = 1 - self.b + self.b * (dl / self.avgdl)

            # Term saturation numerator: tf * (k1 + 1)
            numerator = term_freq * (self.k1 + 1)

            # Term saturation denominator: tf + k1 * length_norm
            # The k1 in the denominator limits how much high tf boosts score
            denominator = term_freq + self.k1 * length_norm

            # BM25 contribution for this term
            term_score = idf_val * (numerator / denominator)
            total_score += term_score                  # Add to total

        return total_score

    def get_scores(self, query):
        """
        Score ALL documents in the corpus for a given query.

        Parameters:
            query (str): The search query

        Returns:
            numpy array: Score for each document (shape: [N])
        """
        scores = np.array([                            # Compute score for each doc
            self.score(query, i)                       # score() for doc at index i
            for i in range(self.N)
        ])
        return scores


# ============================================================
# PART 2: Simple Semantic Scorer
# ============================================================
# Reusing the word vector approach from earlier examples.
# In production this would be a real bi-encoder (sentence-transformers).

# Simple 5-dim word vectors (reusing concept from example_01)
WORD_VECS = {
    # Python / programming
    "python":       np.array([0.9, 0.0, 0.0, 0.0, 0.1]),
    "programming":  np.array([0.9, 0.0, 0.0, 0.0, 0.1]),
    "code":         np.array([0.8, 0.1, 0.0, 0.0, 0.1]),
    "coding":       np.array([0.8, 0.1, 0.0, 0.0, 0.1]),
    "software":     np.array([0.9, 0.0, 0.0, 0.0, 0.0]),
    "developer":    np.array([0.8, 0.0, 0.0, 0.0, 0.2]),
    "script":       np.array([0.7, 0.1, 0.0, 0.0, 0.0]),
    "function":     np.array([0.7, 0.1, 0.0, 0.0, 0.0]),
    "error":        np.array([0.6, 0.0, 0.1, 0.0, 0.1]),
    "exception":    np.array([0.7, 0.0, 0.0, 0.0, 0.0]),
    "debug":        np.array([0.7, 0.0, 0.1, 0.0, 0.0]),
    "fix":          np.array([0.5, 0.0, 0.1, 0.0, 0.1]),
    "bug":          np.array([0.6, 0.0, 0.1, 0.0, 0.0]),
    "tutorial":     np.array([0.7, 0.0, 0.0, 0.0, 0.2]),
    "learn":        np.array([0.6, 0.0, 0.0, 0.1, 0.2]),
    "guide":        np.array([0.6, 0.0, 0.0, 0.0, 0.2]),
    "beginner":     np.array([0.5, 0.0, 0.0, 0.0, 0.3]),
    # Health / fitness
    "exercise":     np.array([0.0, 0.9, 0.0, 0.0, 0.1]),
    "fitness":      np.array([0.0, 0.9, 0.0, 0.0, 0.1]),
    "workout":      np.array([0.0, 0.9, 0.0, 0.0, 0.0]),
    "health":       np.array([0.0, 0.8, 0.1, 0.0, 0.1]),
    "diet":         np.array([0.0, 0.6, 0.4, 0.0, 0.0]),
    "run":          np.array([0.0, 0.9, 0.0, 0.0, 0.0]),
    "running":      np.array([0.0, 0.9, 0.0, 0.0, 0.0]),
    "gym":          np.array([0.0, 0.9, 0.0, 0.0, 0.0]),
    "muscles":      np.array([0.0, 0.8, 0.1, 0.0, 0.0]),
    "training":     np.array([0.0, 0.7, 0.0, 0.2, 0.0]),
    # Food
    "food":         np.array([0.0, 0.0, 0.9, 0.1, 0.0]),
    "eat":          np.array([0.0, 0.0, 0.9, 0.0, 0.0]),
    "meal":         np.array([0.0, 0.1, 0.9, 0.0, 0.0]),
    "recipe":       np.array([0.0, 0.0, 0.9, 0.0, 0.1]),
    "cook":         np.array([0.0, 0.0, 0.9, 0.0, 0.1]),
    "cooking":      np.array([0.0, 0.0, 0.9, 0.0, 0.1]),
    "ingredient":   np.array([0.0, 0.0, 0.9, 0.0, 0.0]),
    "vegetables":   np.array([0.0, 0.0, 0.8, 0.2, 0.0]),
    "nutrition":    np.array([0.0, 0.2, 0.7, 0.2, 0.0]),
    "restaurant":   np.array([0.0, 0.0, 0.8, 0.0, 0.2]),
    # Science
    "science":      np.array([0.0, 0.0, 0.0, 0.9, 0.0]),
    "research":     np.array([0.0, 0.0, 0.0, 0.9, 0.0]),
    "study":        np.array([0.0, 0.0, 0.0, 0.8, 0.1]),
    "experiment":   np.array([0.0, 0.0, 0.0, 0.9, 0.0]),
    "data":         np.array([0.2, 0.0, 0.0, 0.8, 0.0]),
    "analysis":     np.array([0.2, 0.0, 0.0, 0.8, 0.0]),
    # Common words
    "the":          np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "a":            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "and":          np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "to":           np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "for":          np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "in":           np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "of":           np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "how":          np.array([0.0, 0.0, 0.0, 0.1, 0.0]),
    "with":         np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "best":         np.array([0.0, 0.0, 0.0, 0.0, 0.1]),
    "tips":         np.array([0.0, 0.0, 0.0, 0.0, 0.2]),
    "your":         np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "new":          np.array([0.0, 0.0, 0.0, 0.0, 0.1]),
    "using":        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "introduction": np.array([0.0, 0.0, 0.0, 0.1, 0.2]),
    "advanced":     np.array([0.1, 0.0, 0.0, 0.1, 0.1]),
    "complete":     np.array([0.0, 0.0, 0.0, 0.0, 0.1]),
}


def embed(text):
    """Convert text to a vector by averaging word vectors."""
    words = text.lower().split()
    vecs = [WORD_VECS.get(w, np.zeros(5)) for w in words]  # Look up each word
    if not vecs:
        return np.zeros(5)
    return np.mean(np.stack(vecs), axis=0)                  # Average


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================
# PART 3: Hybrid Search Functions
# ============================================================

def normalize_scores(scores_dict):
    """
    Normalize a dict of {doc_id: score} to the range [0, 1].
    This is REQUIRED before weighted combination -- otherwise the
    score with the larger scale dominates.

    Parameters:
        scores_dict (dict): {doc_id: raw_score}

    Returns:
        dict: {doc_id: normalized_score} where all values are in [0, 1]
    """
    if not scores_dict:
        return {}
    values = list(scores_dict.values())
    min_val = min(values)                  # Lowest score
    max_val = max(values)                  # Highest score
    score_range = max_val - min_val        # Range
    if score_range == 0:
        return {k: 0.5 for k in scores_dict}   # All same -- assign 0.5
    # Normalize each score to [0, 1]
    return {
        doc_id: (score - min_val) / score_range
        for doc_id, score in scores_dict.items()
    }


def weighted_hybrid(bm25_scores, semantic_scores, alpha=0.5):
    """
    Weighted combination: final = alpha * semantic + (1 - alpha) * bm25
    Both score dictionaries must be normalized to [0, 1] first.

    Parameters:
        bm25_scores     (dict): {doc_id: normalized_bm25_score}
        semantic_scores (dict): {doc_id: normalized_semantic_score}
        alpha           (float): Weight for semantic (0.0 = BM25 only, 1.0 = semantic only)

    Returns:
        dict: {doc_id: combined_score}
    """
    # Get all document IDs (union of both score sets)
    all_doc_ids = set(bm25_scores.keys()) | set(semantic_scores.keys())

    combined = {}
    for doc_id in all_doc_ids:
        bm25_s = bm25_scores.get(doc_id, 0.0)         # 0 if not in BM25 results
        sem_s = semantic_scores.get(doc_id, 0.0)       # 0 if not in semantic results
        # Weighted average
        combined[doc_id] = alpha * sem_s + (1 - alpha) * bm25_s

    return combined


def reciprocal_rank_fusion(rankings_list, k=60):
    """
    RRF: combine multiple ranked lists by rank POSITION (not score).

    For each document in each ranked list:
        contribution = 1 / (k + rank)

    Final RRF score = sum of contributions across all lists.

    Parameters:
        rankings_list (list of lists): Each inner list is [(doc_id, score), ...] sorted by score
        k             (int):           RRF constant (default 60, from literature)

    Returns:
        dict: {doc_id: rrf_score} where higher = better combined rank
    """
    rrf_scores = {}                            # Accumulate RRF scores

    for ranked_list in rankings_list:          # For each ranking (BM25, semantic, etc.)
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            # RRF contribution for this document in this ranked list
            contribution = 1.0 / (k + rank)

            # Add to this document's accumulated RRF score
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += contribution

    return rrf_scores


def print_ranked_results(title, scores_dict, documents, top_k=5):
    """
    Helper: print the top results from a scores dictionary.

    Parameters:
        title        (str):  Section header text
        scores_dict  (dict): {doc_id: score}
        documents    (dict): {doc_id: doc_text}
        top_k        (int):  How many results to print
    """
    print(f"\n{title}")
    sorted_results = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (doc_id, score) in enumerate(sorted_results[:top_k], start=1):
        text = documents.get(doc_id, "?")[:60]         # Truncate long texts
        print(f"  {rank}. [{score:.3f}] {text}")


# ============================================================
# PART 4: Test Corpus and Main Demo
# ============================================================

# Our test corpus: diverse topics to show hybrid search advantages
CORPUS_DOCS = {
    "d01": "Python 3.11 release notes and new features",
    "d02": "How to update Python to the latest version",
    "d03": "Python programming for data science beginners",
    "d04": "Installing Python packages with pip",
    "d05": "Python exception handling and error debugging tips",
    "d06": "Exercise routine for beginners: running and gym training",
    "d07": "Best fitness tips to improve your workout",
    "d08": "Healthy eating and nutrition guide for athletes",
    "d09": "Running training schedule for marathon preparation",
    "d10": "Python vs Java: which programming language to learn",
    "d11": "Introduction to machine learning with Python",
    "d12": "Cooking healthy meals: complete recipe guide",
    "d13": "Scientific research on exercise and health benefits",
    "d14": "Python data analysis using pandas and numpy",
    "d15": "Advanced running techniques and race training",
}

CORPUS_TEXTS = list(CORPUS_DOCS.values())   # Plain text list for BM25


if __name__ == "__main__":

    print("=" * 65)
    print("  HYBRID SEARCH DEMO (BM25 + Semantic)")
    print("=" * 65)

    # Build the BM25 index
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(CORPUS_TEXTS)           # Index all documents

    # Pre-encode all documents for semantic search
    doc_ids = list(CORPUS_DOCS.keys())
    doc_vectors = {
        doc_id: embed(text)
        for doc_id, text in CORPUS_DOCS.items()
    }

    # --------------------------------------------------------
    # Test Query 1: Exact keyword -- BM25 should win
    # --------------------------------------------------------
    query_1 = "Python 3.11"
    print(f"\n{'='*65}")
    print(f"QUERY 1 (exact keyword): '{query_1}'")
    print("Expected: BM25 should find 'd01' easily, semantic may struggle")

    # BM25 scores for query_1
    bm25_raw = bm25.get_scores(query_1)
    bm25_scores_q1 = {doc_ids[i]: float(bm25_raw[i]) for i in range(len(doc_ids))}

    # Semantic scores for query_1
    q1_vec = embed(query_1)
    sem_scores_q1 = {
        doc_id: cosine_similarity(q1_vec, doc_vectors[doc_id])
        for doc_id in doc_ids
    }

    # Normalize before combining
    bm25_norm_q1 = normalize_scores(bm25_scores_q1)
    sem_norm_q1 = normalize_scores(sem_scores_q1)

    # Weighted hybrid
    hybrid_weighted_q1 = weighted_hybrid(bm25_norm_q1, sem_norm_q1, alpha=0.4)

    # RRF hybrid
    bm25_ranked_q1 = sorted(bm25_scores_q1.items(), key=lambda x: x[1], reverse=True)
    sem_ranked_q1 = sorted(sem_scores_q1.items(), key=lambda x: x[1], reverse=True)
    rrf_scores_q1 = reciprocal_rank_fusion([bm25_ranked_q1, sem_ranked_q1])

    print_ranked_results("BM25 only:", bm25_scores_q1, CORPUS_DOCS, top_k=3)
    print_ranked_results("Semantic only:", sem_scores_q1, CORPUS_DOCS, top_k=3)
    print_ranked_results("Hybrid (weighted, alpha=0.4):", hybrid_weighted_q1, CORPUS_DOCS, top_k=3)
    print_ranked_results("Hybrid (RRF):", rrf_scores_q1, CORPUS_DOCS, top_k=3)

    # --------------------------------------------------------
    # Test Query 2: Conceptual -- Semantic should win
    # --------------------------------------------------------
    query_2 = "how do I upgrade my Python installation"
    print(f"\n{'='*65}")
    print(f"QUERY 2 (conceptual): '{query_2}'")
    print("Expected: Semantic should find 'd02' (update Python), BM25 may miss it")

    bm25_raw = bm25.get_scores(query_2)
    bm25_scores_q2 = {doc_ids[i]: float(bm25_raw[i]) for i in range(len(doc_ids))}

    q2_vec = embed(query_2)
    sem_scores_q2 = {
        doc_id: cosine_similarity(q2_vec, doc_vectors[doc_id])
        for doc_id in doc_ids
    }

    bm25_norm_q2 = normalize_scores(bm25_scores_q2)
    sem_norm_q2 = normalize_scores(sem_scores_q2)

    hybrid_weighted_q2 = weighted_hybrid(bm25_norm_q2, sem_norm_q2, alpha=0.6)

    bm25_ranked_q2 = sorted(bm25_scores_q2.items(), key=lambda x: x[1], reverse=True)
    sem_ranked_q2 = sorted(sem_scores_q2.items(), key=lambda x: x[1], reverse=True)
    rrf_scores_q2 = reciprocal_rank_fusion([bm25_ranked_q2, sem_ranked_q2])

    print_ranked_results("BM25 only:", bm25_scores_q2, CORPUS_DOCS, top_k=3)
    print_ranked_results("Semantic only:", sem_scores_q2, CORPUS_DOCS, top_k=3)
    print_ranked_results("Hybrid (weighted, alpha=0.6):", hybrid_weighted_q2, CORPUS_DOCS, top_k=3)
    print_ranked_results("Hybrid (RRF):", rrf_scores_q2, CORPUS_DOCS, top_k=3)

    # --------------------------------------------------------
    # Test Query 3: Mixed -- Hybrid should win
    # --------------------------------------------------------
    query_3 = "exercise and fitness with Python data analysis"
    print(f"\n{'='*65}")
    print(f"QUERY 3 (mixed topics): '{query_3}'")
    print("Expected: Hybrid finds both fitness AND Python data docs")

    bm25_raw = bm25.get_scores(query_3)
    bm25_scores_q3 = {doc_ids[i]: float(bm25_raw[i]) for i in range(len(doc_ids))}

    q3_vec = embed(query_3)
    sem_scores_q3 = {
        doc_id: cosine_similarity(q3_vec, doc_vectors[doc_id])
        for doc_id in doc_ids
    }

    bm25_norm_q3 = normalize_scores(bm25_scores_q3)
    sem_norm_q3 = normalize_scores(sem_scores_q3)

    hybrid_weighted_q3 = weighted_hybrid(bm25_norm_q3, sem_norm_q3, alpha=0.5)

    bm25_ranked_q3 = sorted(bm25_scores_q3.items(), key=lambda x: x[1], reverse=True)
    sem_ranked_q3 = sorted(sem_scores_q3.items(), key=lambda x: x[1], reverse=True)
    rrf_scores_q3 = reciprocal_rank_fusion([bm25_ranked_q3, sem_ranked_q3])

    print_ranked_results("BM25 only:", bm25_scores_q3, CORPUS_DOCS, top_k=3)
    print_ranked_results("Semantic only:", sem_scores_q3, CORPUS_DOCS, top_k=3)
    print_ranked_results("Hybrid (weighted, alpha=0.5):", hybrid_weighted_q3, CORPUS_DOCS, top_k=3)
    print_ranked_results("Hybrid (RRF):", rrf_scores_q3, CORPUS_DOCS, top_k=3)

    # --------------------------------------------------------
    # Show BM25 internals
    # --------------------------------------------------------
    print(f"\n{'='*65}")
    print("BM25 INTERNALS: Top IDF scores (most distinctive terms in corpus)")
    top_idf = sorted(bm25.idf.items(), key=lambda x: x[1], reverse=True)
    for term, idf_val in top_idf[:10]:
        print(f"  IDF({term:15s}) = {idf_val:.3f}")

    print("\nLow IDF terms (appear in many documents -- less distinctive):")
    bottom_idf = sorted(bm25.idf.items(), key=lambda x: x[1])
    for term, idf_val in bottom_idf[:5]:
        print(f"  IDF({term:15s}) = {idf_val:.3f}")


# ============================================================
# PART B: Real Libraries
# ============================================================
# Remove the # at the start of each line to run this section.
# Requirements: pip install rank-bm25 sentence-transformers
#
# from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
# import numpy as np
#
# # Real BM25
# tokenized_corpus = [doc.lower().split() for doc in CORPUS_TEXTS]
# real_bm25 = BM25Okapi(tokenized_corpus)
#
# # Real sentence embeddings
# real_model = SentenceTransformer('all-MiniLM-L6-v2')
# real_doc_vecs = real_model.encode(CORPUS_TEXTS)   # shape: (15, 384)
#
# query = "how do I upgrade Python"
# real_query_vec = real_model.encode([query])       # shape: (1, 384)
#
# # BM25 scores
# real_bm25_scores = real_bm25.get_scores(query.lower().split())
#
# # Semantic scores
# real_sem_scores = sk_cosine(real_query_vec, real_doc_vecs)[0]  # shape: (15,)
#
# # Normalize and combine (use our functions from above)
# bm25_dict = {doc_ids[i]: float(real_bm25_scores[i]) for i in range(len(doc_ids))}
# sem_dict  = {doc_ids[i]: float(real_sem_scores[i])  for i in range(len(doc_ids))}
# bm25_norm = normalize_scores(bm25_dict)
# sem_norm  = normalize_scores(sem_dict)
# hybrid    = weighted_hybrid(bm25_norm, sem_norm, alpha=0.5)
#
# print_ranked_results("Real hybrid results:", hybrid, CORPUS_DOCS, top_k=5)

# -*- coding: utf-8 -*-
# exercise_01_encode_and_search.py
#
# Module 10.8 -- Semantic Search Systems
# Exercise 1: Encode and Search
#
# TASK:
#   Given 15 documents and a query, encode all documents and the query
#   into vectors, then find the top-5 most similar documents.
#
# YOUR JOB:
#   Fill in the three functions marked with TODO:
#     1. encode_document(text)  -- convert a document text to a vector
#     2. encode_query(text)     -- convert a query to a vector (same logic!)
#     3. rank_results(...)      -- rank documents by similarity score
#
# HELPER PROVIDED:
#   - cosine_similarity(a, b)   -- already written for you
#   - WORD_VECTORS lookup table -- already provided
#
# HOW TO RUN:
#   python exercises/exercise_01_encode_and_search.py

import numpy as np


# ============================================================
# PROVIDED: Word Vector Lookup Table
# ============================================================
# You do not need to change this.
# Each word maps to an 8-dimensional vector.
# Dimensions represent: [tech, health, food, travel, finance, law, sports, science]

WORD_VECTORS = {
    # Tech
    "python":       np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "programming":  np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "software":     np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
    "algorithm":    np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]),
    "database":     np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
    "web":          np.array([0.7, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
    "developer":    np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
    "cloud":        np.array([0.7, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]),
    "api":          np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "machine":      np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]),
    "learning":     np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]),
    # Health
    "doctor":       np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.2]),
    "hospital":     np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "medicine":     np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]),
    "health":       np.array([0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.1, 0.2]),
    "treatment":    np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1]),
    "patient":      np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),
    "clinical":     np.array([0.0, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.3]),
    # Food
    "recipe":       np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "cooking":      np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "ingredients":  np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "restaurant":   np.array([0.0, 0.0, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0]),
    "meal":         np.array([0.0, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "kitchen":      np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
    # Travel
    "travel":       np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
    "vacation":     np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
    "hotel":        np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]),
    "flight":       np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]),
    "destination":  np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
    # Finance
    "investment":   np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]),
    "stock":        np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "money":        np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]),
    "bank":         np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0]),
    "financial":    np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.2, 0.0, 0.0]),
    # Law
    "law":          np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0]),
    "legal":        np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0]),
    "court":        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "contract":     np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0]),
    "rights":       np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0]),
    # Sports
    "football":     np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
    "tennis":       np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
    "athlete":      np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
    "championship": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0]),
    "training":     np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0]),
    # Common
    "the":  np.zeros(8), "a":   np.zeros(8), "and":  np.zeros(8),
    "to":   np.zeros(8), "of":  np.zeros(8), "in":   np.zeros(8),
    "for":  np.zeros(8), "how":  np.zeros(8), "with": np.zeros(8),
    "guide":        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "best":         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "introduction": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "advanced":     np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "science":      np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]),
    "research":     np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]),
    "new":          np.zeros(8), "tips": np.zeros(8), "your": np.zeros(8),
    "complete":     np.zeros(8), "using": np.zeros(8),
}


# ============================================================
# PROVIDED: Cosine Similarity (DO NOT MODIFY)
# ============================================================

def cosine_similarity(vec_a, vec_b):
    """
    Computes cosine similarity between two vectors.
    Returns float from 0.0 (completely different) to 1.0 (identical direction).
    """
    norm_a = np.linalg.norm(vec_a)     # Length of vector A
    norm_b = np.linalg.norm(vec_b)     # Length of vector B
    if norm_a == 0 or norm_b == 0:     # Protect against zero vectors
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ============================================================
# EXERCISE: Your 15 test documents
# ============================================================

DOCUMENTS = [
    # id 0
    "Python programming guide for software developers",
    # id 1
    "Machine learning with Python algorithms",
    # id 2
    "Doctor and patient guide to clinical treatment",
    # id 3
    "Best cooking recipe with fresh ingredients",
    # id 4
    "Travel vacation and hotel destination guide",
    # id 5
    "Stock market investment and financial advice",
    # id 6
    "Legal rights and court contract guide",
    # id 7
    "Football championship athlete training",
    # id 8
    "Web developer and cloud api database guide",
    # id 9
    "Hospital medicine and health research",
    # id 10
    "Advanced Python machine learning api",
    # id 11
    "Restaurant meal and kitchen cooking tips",
    # id 12
    "Tennis athlete championship training",
    # id 13
    "Financial investment bank and money tips",
    # id 14
    "Science research for medicine and health",
]


# ============================================================
# YOUR TASK: Fill in these three functions
# ============================================================

def encode_document(text):
    """
    Convert a document text to a vector (embedding).

    HOW IT WORKS:
    1. Split the text into words (use .lower().split())
    2. For each word, look it up in WORD_VECTORS (use .get() with a default)
    3. Collect all found vectors into a list
    4. Return the AVERAGE of those vectors

    If no words are found, return np.zeros(8).

    HINT: Look at how embed() works in the examples/ files.
    They all use the same pattern.

    Parameters:
        text (str): Document text to encode

    Returns:
        numpy array: 8-dimensional vector representing the document
    """
    # TODO: Fill in this function
    # Step 1: Split text into words
    words = text.lower().split()

    # Step 2 + 3: Look up each word, collect vectors
    vectors = []
    for word in words:
        # YOUR CODE HERE: look up word in WORD_VECTORS, add to vectors if found
        pass

    # Step 4: Return average (or zeros if nothing found)
    # YOUR CODE HERE
    pass


def encode_query(query_text):
    """
    Convert a query to a vector.

    HINT: This is EXACTLY the same as encode_document!
    In a bi-encoder, the same function encodes both queries and documents.
    You can call encode_document() from here.

    Parameters:
        query_text (str): Query text to encode

    Returns:
        numpy array: 8-dimensional vector representing the query
    """
    # TODO: Fill in this function (1 line -- just call encode_document)
    # YOUR CODE HERE
    pass


def rank_results(query_vector, document_vectors, documents, top_k=5):
    """
    Rank documents by cosine similarity to the query.

    Steps:
    1. For each document, compute cosine_similarity(query_vector, doc_vector)
    2. Sort documents by score (highest first)
    3. Return top_k results

    Parameters:
        query_vector     (numpy array): The encoded query
        document_vectors (list):        List of encoded document vectors
        documents        (list):        Original document texts
        top_k            (int):         How many results to return

    Returns:
        List of (rank, similarity_score, document_text) tuples
    """
    # Step 1: Score each document
    scores = []
    for idx, doc_vector in enumerate(document_vectors):
        # TODO: Compute similarity between query_vector and doc_vector
        # Use the cosine_similarity() function provided above
        similarity = 0.0  # YOUR CODE HERE -- replace 0.0 with actual similarity

        scores.append((similarity, idx, documents[idx]))

    # Step 2: TODO -- sort by similarity score, highest first
    # HINT: Use .sort(key=..., reverse=True) or sorted(...)
    # YOUR CODE HERE

    # Step 3: TODO -- return top_k results as (rank, score, text) tuples
    results = []
    # YOUR CODE HERE
    return results


# ============================================================
# TEST CODE -- Do not modify this section
# ============================================================

def test_your_solution():
    """Run tests to verify your implementation."""
    print("Testing your solution...\n")
    passed = 0
    failed = 0

    # Test 1: encode_document returns a numpy array of length 8
    vec = encode_document("python programming software")
    if vec is not None and len(vec) == 8:
        print("[PASS] encode_document returns 8-dim vector")
        passed += 1
    else:
        print("[FAIL] encode_document should return 8-dim numpy array")
        failed += 1

    # Test 2: Unknown words return zero vector
    vec_unknown = encode_document("xyzabc qwerty bloop")
    if vec_unknown is not None and np.all(vec_unknown == 0):
        print("[PASS] Unknown words return zero vector")
        passed += 1
    else:
        print("[FAIL] Unknown words should return np.zeros(8)")
        failed += 1

    # Test 3: encode_query and encode_document return same result for same text
    text = "python programming"
    vec_doc = encode_document(text)
    vec_qry = encode_query(text)
    if vec_doc is not None and vec_qry is not None and np.allclose(vec_doc, vec_qry):
        print("[PASS] encode_query and encode_document agree")
        passed += 1
    else:
        print("[FAIL] encode_query should call encode_document (same result)")
        failed += 1

    # Test 4: Programming query finds programming docs
    doc_vecs = [encode_document(d) for d in DOCUMENTS]
    q_vec = encode_query("python machine learning database")
    results = rank_results(q_vec, doc_vecs, DOCUMENTS, top_k=5)

    if results and len(results) == 5:
        top_doc = results[0][2]          # Text of top result
        if "python" in top_doc.lower() or "machine" in top_doc.lower():
            print("[PASS] Programming query returns programming document at rank 1")
            passed += 1
        else:
            print(f"[FAIL] Expected a Python/ML doc at rank 1, got: {top_doc}")
            failed += 1
    else:
        print("[FAIL] rank_results should return a list of 5 (rank, score, text) tuples")
        failed += 1

    # Test 5: Health query finds health docs
    q_vec_health = encode_query("doctor hospital medicine treatment")
    results_health = rank_results(q_vec_health, doc_vecs, DOCUMENTS, top_k=5)
    if results_health:
        top_health = results_health[0][2]
        if "doctor" in top_health.lower() or "hospital" in top_health.lower() or "medicine" in top_health.lower():
            print("[PASS] Health query returns health document at rank 1")
            passed += 1
        else:
            print(f"[FAIL] Expected a health doc at rank 1, got: {top_health}")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":

    print("=" * 60)
    print("  EXERCISE 1: Encode and Search")
    print("=" * 60)
    print("\nFill in encode_document(), encode_query(), and rank_results()")
    print("then run this file to see if your solution is correct.\n")

    # Try to run the full demo
    try:
        # Encode all documents
        doc_vectors = [encode_document(doc) for doc in DOCUMENTS]

        if doc_vectors[0] is None or doc_vectors[0].sum() == 0:
            print("NOTE: encode_document not implemented yet.")
            print("Fill in the TODO sections above, then re-run.")
        else:
            # Run a test query
            query = "python machine learning algorithm"
            query_vector = encode_query(query)

            print(f"Query: '{query}'")
            print(f"Query vector: {query_vector.round(3)}")
            print("\nTop 5 search results:")
            results = rank_results(query_vector, doc_vectors, DOCUMENTS, top_k=5)
            for rank, score, doc in results:
                print(f"  Rank {rank}: [{score:.3f}] {doc}")

            # Run tests
            print("\n" + "=" * 60)
            all_passed = test_your_solution()
            if all_passed:
                print("\nExcellent! All tests passed. Your encoder and search work correctly.")
            else:
                print("\nSome tests failed. Review the TODO comments and try again.")

    except TypeError:
        print("NOTE: Some functions return None (not yet implemented).")
        print("Fill in the TODO sections above.\n")
        test_your_solution()


# ============================================================
# SOLUTION (Hidden -- Try yourself first!)
# ============================================================
# def encode_document(text):
#     words = text.lower().split()
#     vectors = []
#     for word in words:
#         vec = WORD_VECTORS.get(word)
#         if vec is not None:
#             vectors.append(vec)
#     if not vectors:
#         return np.zeros(8)
#     return np.mean(np.stack(vectors), axis=0)
#
# def encode_query(query_text):
#     return encode_document(query_text)
#
# def rank_results(query_vector, document_vectors, documents, top_k=5):
#     scores = []
#     for idx, doc_vector in enumerate(document_vectors):
#         similarity = cosine_similarity(query_vector, doc_vector)
#         scores.append((similarity, idx, documents[idx]))
#     scores.sort(key=lambda x: x[0], reverse=True)
#     results = []
#     for rank, (score, idx, text) in enumerate(scores[:top_k], start=1):
#         results.append((rank, score, text))
#     return results

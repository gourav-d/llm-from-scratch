# -*- coding: utf-8 -*-
# exercise_04_hybrid_search.py
#
# Module 10.8 -- Semantic Search Systems
# Exercise 4: Implement Hybrid Scoring
#
# TASK:
#   Given pre-computed BM25 scores and semantic scores for a set of documents,
#   implement two hybrid combination methods:
#     1. weighted_combine(bm25_scores, semantic_scores, alpha) -- weighted average
#     2. reciprocal_rank_fusion(bm25_ranks, semantic_ranks)    -- RRF
#
# IMPORTANT: For weighted_combine, you must normalize BOTH score sets to [0,1]
#            BEFORE combining them, or one score scale will dominate.
#
# HOW TO RUN:
#   python exercises/exercise_04_hybrid_search.py

import numpy as np


# ============================================================
# PROVIDED: Pre-computed scores for 10 documents
# ============================================================
# Imagine a user searched "python tutorial for beginners"
# These are the raw scores each method gave each document.

DOCUMENT_NAMES = {
    "d01": "Python programming for beginners step by step",
    "d02": "Advanced Python machine learning techniques",
    "d03": "Introduction to Python: first steps guide",
    "d04": "Python debugging and error handling tips",
    "d05": "Learn Python from scratch: complete guide",
    "d06": "Python data science with pandas tutorial",
    "d07": "JavaScript programming for beginners",
    "d08": "Python web development with Flask",
    "d09": "Machine learning without Python using R",
    "d10": "Complete beginner tutorial: Python basics",
}

# BM25 scores (keyword matching)
# Raw values are on a different scale than semantic scores!
# Note: "Python" and "tutorial" and "beginners" appear in many docs so BM25 varies.
BM25_RAW = {
    "d01": 8.3,    # "Python" + "beginners" exact match
    "d02": 3.1,    # "Python" but not "beginners" or "tutorial"
    "d03": 7.8,    # "Introduction" ~ "tutorial", "Python"
    "d04": 4.2,    # "Python" + "tips" -- partial
    "d05": 9.1,    # "Learn Python" + "guide" -- good keyword coverage
    "d06": 5.5,    # "Python" + "tutorial" but not "beginners"
    "d07": 2.1,    # Has "beginners" but NOT "Python"
    "d08": 3.8,    # "Python" but not "tutorial" or "beginners"
    "d09": 1.2,    # "machine learning" -- almost no keyword overlap
    "d10": 8.8,    # "beginner tutorial" + "Python basics"
}

# Semantic scores (cosine similarity, already in [0, 1])
SEMANTIC_RAW = {
    "d01": 0.88,   # Very similar meaning to query
    "d02": 0.62,   # Somewhat related (Python ML)
    "d03": 0.85,   # Very similar (intro/first steps = tutorial for beginners)
    "d04": 0.71,   # Related (Python tips)
    "d05": 0.91,   # Highly similar (learn Python from scratch = tutorial for beginners)
    "d06": 0.75,   # Related (Python tutorial, different topic)
    "d07": 0.61,   # Somewhat related (beginners, not Python)
    "d08": 0.68,   # Related (Python, different topic)
    "d09": 0.35,   # Low similarity (ML without Python)
    "d10": 0.89,   # Highly similar (beginner Python tutorial)
}


# ============================================================
# YOUR TASK: Fill in these two functions
# ============================================================

def weighted_combine(bm25_scores, semantic_scores, alpha=0.5):
    """
    Combine BM25 and semantic scores using weighted average.

    FORMULA:
        combined = alpha * sem_norm + (1 - alpha) * bm25_norm

    where sem_norm and bm25_norm are NORMALIZED to [0, 1].

    WHY NORMALIZE FIRST?
    BM25 scores can be 0-20. Semantic scores are 0-1.
    Without normalizing, a BM25 score of 8.0 would dominate a semantic score of 0.88.
    After normalizing, both are on the same [0, 1] scale.

    STEPS TO IMPLEMENT:
    1. Normalize bm25_scores to [0, 1] using min-max normalization:
           bm25_norm[key] = (bm25_scores[key] - min_bm25) / (max_bm25 - min_bm25)
       Handle edge case: if all scores are the same, set all normalized scores to 0.5.

    2. Normalize semantic_scores to [0, 1] the same way.

    3. For each document ID (union of both dicts' keys):
           combined[doc_id] = alpha * sem_norm.get(doc_id, 0) + (1-alpha) * bm25_norm.get(doc_id, 0)

    4. Return the combined dict.

    Parameters:
        bm25_scores     (dict): {doc_id: raw_bm25_score}
        semantic_scores (dict): {doc_id: raw_semantic_score}
        alpha           (float): Weight for semantic (0.0 to 1.0).
                                 0.5 = equal weight; 0.7 = 70% semantic, 30% BM25

    Returns:
        dict: {doc_id: combined_score} where all scores are in [0, 1]
    """
    # TODO: Step 1 -- Normalize BM25 scores
    bm25_norm = {}
    # YOUR CODE HERE

    # TODO: Step 2 -- Normalize semantic scores
    sem_norm = {}
    # YOUR CODE HERE

    # TODO: Step 3 -- Compute weighted combination
    combined = {}
    all_doc_ids = set(bm25_scores.keys()) | set(semantic_scores.keys())
    for doc_id in all_doc_ids:
        # YOUR CODE HERE
        pass

    return combined  # TODO: make sure combined is populated before returning


def reciprocal_rank_fusion(bm25_ranks, semantic_ranks, k=60):
    """
    Combine two ranked lists using Reciprocal Rank Fusion (RRF).

    RRF FORMULA (for each document):
        rrf_score = sum over all lists of:  1 / (k + rank)

    where rank is the position of the document in that list (1 = first).

    WHY USE RRF?
    - It does NOT need score normalization (only uses rank positions)
    - Robust: a document ranked #1 in both lists scores very high
    - A document ranked #1 in one list and #50 in another scores moderately

    STEPS TO IMPLEMENT:
    1. For each ranked list (bm25_ranks and semantic_ranks):
       - Iterate with rank starting from 1
       - For each (doc_id, score) in the list:
           rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    2. Return rrf_scores dict.

    Parameters:
        bm25_ranks     (list): [(doc_id, score), ...] sorted by BM25 score descending
        semantic_ranks (list): [(doc_id, score), ...] sorted by semantic score descending
        k              (int):  RRF constant (default 60)

    Returns:
        dict: {doc_id: rrf_score} where higher = better combined rank
    """
    rrf_scores = {}

    # TODO: Process bm25_ranks
    for rank, (doc_id, score) in enumerate(bm25_ranks, start=1):
        # YOUR CODE HERE: add 1/(k+rank) to rrf_scores[doc_id]
        pass

    # TODO: Process semantic_ranks
    for rank, (doc_id, score) in enumerate(semantic_ranks, start=1):
        # YOUR CODE HERE: add 1/(k+rank) to rrf_scores[doc_id]
        pass

    return rrf_scores


# ============================================================
# HELPER: Pretty print results
# ============================================================

def print_top_results(title, scores_dict, doc_names, top_k=5):
    """Print top-k results from a scores dict."""
    print(f"\n{title}")
    ranked = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
        name = doc_names.get(doc_id, doc_id)[:55]
        print(f"  {rank}. [{score:.4f}] {name}")


# ============================================================
# TESTS -- Do not modify
# ============================================================

def run_tests():
    """Automated tests for your implementation."""
    print("\nRunning tests...")
    passed = 0
    failed = 0

    # Test 1: weighted_combine returns a dict
    result = weighted_combine(BM25_RAW, SEMANTIC_RAW, alpha=0.5)
    if isinstance(result, dict) and len(result) > 0:
        print("[PASS] weighted_combine returns a non-empty dict")
        passed += 1
    else:
        print(f"[FAIL] weighted_combine should return dict, got: {type(result)}")
        failed += 1

    # Test 2: All combined scores are in [0, 1]
    if result:
        all_valid = all(0.0 <= v <= 1.001 for v in result.values())
        if all_valid:
            print("[PASS] All weighted combined scores are in [0.0, 1.0]")
            passed += 1
        else:
            bad = {k: v for k, v in result.items() if not (0.0 <= v <= 1.001)}
            print(f"[FAIL] Scores out of range: {bad}")
            failed += 1

    # Test 3: With alpha=1.0, result should match normalized semantic scores
    result_sem = weighted_combine(BM25_RAW, SEMANTIC_RAW, alpha=1.0)
    # Normalized semantic: d05 (0.91) should be highest after normalization
    if result_sem:
        top_doc = max(result_sem.items(), key=lambda x: x[1])[0]
        if top_doc == "d05":
            print(f"[PASS] alpha=1.0: top doc is d05 (highest semantic score)")
            passed += 1
        else:
            print(f"[FAIL] alpha=1.0: expected d05, got {top_doc}")
            failed += 1

    # Test 4: With alpha=0.0, result should match normalized BM25 scores
    result_bm25 = weighted_combine(BM25_RAW, SEMANTIC_RAW, alpha=0.0)
    if result_bm25:
        top_bm25 = max(result_bm25.items(), key=lambda x: x[1])[0]
        if top_bm25 == "d05":  # d05 has BM25 score 9.1 -- highest
            print(f"[PASS] alpha=0.0: top doc is d05 (highest BM25 score)")
            passed += 1
        else:
            print(f"[FAIL] alpha=0.0: expected d05 (BM25=9.1), got {top_bm25}")
            failed += 1

    # Test 5: reciprocal_rank_fusion returns a dict
    bm25_ranked = sorted(BM25_RAW.items(), key=lambda x: x[1], reverse=True)
    sem_ranked = sorted(SEMANTIC_RAW.items(), key=lambda x: x[1], reverse=True)
    rrf = reciprocal_rank_fusion(bm25_ranked, sem_ranked)

    if isinstance(rrf, dict) and len(rrf) == len(DOCUMENT_NAMES):
        print("[PASS] reciprocal_rank_fusion returns dict with all doc IDs")
        passed += 1
    else:
        print(f"[FAIL] RRF should return dict with {len(DOCUMENT_NAMES)} entries, got {len(rrf) if rrf else '?'}")
        failed += 1

    # Test 6: RRF scores are positive
    if rrf:
        all_positive = all(v > 0 for v in rrf.values())
        if all_positive:
            print("[PASS] All RRF scores are positive")
            passed += 1
        else:
            print("[FAIL] RRF scores should all be positive")
            failed += 1

    # Test 7: d05 should rank in top 3 by RRF (consistently high in both lists)
    if rrf:
        rrf_top3 = [d for d, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:3]]
        if "d05" in rrf_top3:
            print(f"[PASS] d05 ('Learn Python from scratch') is in RRF top 3")
            passed += 1
        else:
            print(f"[FAIL] d05 should be in RRF top 3 (strong in both BM25 and semantic), got: {rrf_top3}")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# MAIN DEMO
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  EXERCISE 4: Hybrid Search Scoring")
    print("=" * 65)
    print("\nQuery: 'python tutorial for beginners'")

    # Show raw scores before combination
    print("\n--- RAW SCORES (different scales!) ---")
    print(f"{'Doc':<5} {'BM25':>8} {'Semantic':>10}  Text")
    print("-" * 70)
    for doc_id in sorted(DOCUMENT_NAMES.keys()):
        bm = BM25_RAW.get(doc_id, 0)
        sm = SEMANTIC_RAW.get(doc_id, 0)
        print(f"  {doc_id} {bm:>8.1f} {sm:>10.2f}  {DOCUMENT_NAMES[doc_id][:40]}")

    print("\nNote: BM25 values go up to 9.1, semantic values are 0.0-1.0")
    print("Without normalization, BM25 would completely dominate the hybrid score!")

    try:
        # Method 1: Weighted combination
        hybrid_w = weighted_combine(BM25_RAW, SEMANTIC_RAW, alpha=0.5)

        # Method 2: RRF
        bm25_sorted = sorted(BM25_RAW.items(), key=lambda x: x[1], reverse=True)
        sem_sorted  = sorted(SEMANTIC_RAW.items(), key=lambda x: x[1], reverse=True)
        hybrid_rrf  = reciprocal_rank_fusion(bm25_sorted, sem_sorted)

        if hybrid_w and any(v > 0 for v in hybrid_w.values()):
            # Print all four rankings
            print_top_results("BM25 only:", BM25_RAW, DOCUMENT_NAMES)
            print_top_results("Semantic only:", SEMANTIC_RAW, DOCUMENT_NAMES)
            print_top_results("Hybrid Weighted (alpha=0.5):", hybrid_w, DOCUMENT_NAMES)
            print_top_results("Hybrid RRF:", hybrid_rrf, DOCUMENT_NAMES)

            # Show effect of different alpha values
            print("\n--- EFFECT OF ALPHA ON TOP-1 RESULT ---")
            print("(alpha=0 means BM25-only, alpha=1 means semantic-only)")
            for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
                h = weighted_combine(BM25_RAW, SEMANTIC_RAW, alpha=alpha)
                top_doc_id = max(h.items(), key=lambda x: x[1])[0]
                top_name = DOCUMENT_NAMES[top_doc_id][:45]
                print(f"  alpha={alpha:.1f}: top doc = {top_doc_id} -- {top_name}")

            # Run tests
            print("\n" + "=" * 65)
            run_tests()
        else:
            print("\nNOTE: weighted_combine returned empty or zero results.")
            print("Fill in the TODO sections above, then re-run.")
            run_tests()

    except Exception as ex:
        print(f"\nError: {ex}")
        print("Fill in the TODO sections above.")
        run_tests()


# ============================================================
# SOLUTION (Hidden -- Try yourself first!)
# ============================================================
# def weighted_combine(bm25_scores, semantic_scores, alpha=0.5):
#     def normalize(d):
#         vals = list(d.values())
#         lo, hi = min(vals), max(vals)
#         r = hi - lo
#         if r == 0:
#             return {k: 0.5 for k in d}
#         return {k: (v - lo) / r for k, v in d.items()}
#
#     bm25_norm = normalize(bm25_scores)
#     sem_norm = normalize(semantic_scores)
#     combined = {}
#     for doc_id in set(bm25_scores) | set(semantic_scores):
#         combined[doc_id] = alpha * sem_norm.get(doc_id, 0.0) + \
#                            (1 - alpha) * bm25_norm.get(doc_id, 0.0)
#     return combined
#
# def reciprocal_rank_fusion(bm25_ranks, semantic_ranks, k=60):
#     rrf_scores = {}
#     for rank, (doc_id, _) in enumerate(bm25_ranks, start=1):
#         rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
#     for rank, (doc_id, _) in enumerate(semantic_ranks, start=1):
#         rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
#     return rrf_scores

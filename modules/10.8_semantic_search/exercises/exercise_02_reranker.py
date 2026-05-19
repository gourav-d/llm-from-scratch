# -*- coding: utf-8 -*-
# exercise_02_reranker.py
#
# Module 10.8 -- Semantic Search Systems
# Exercise 2: Build a Re-ranker
#
# TASK:
#   Given bi-encoder top-10 results (query, doc, bi_score) tuples,
#   build a cross-encoder simulation that re-ranks them more accurately.
#
# YOUR JOB:
#   Fill in two functions:
#     1. cross_score(query, document) -- compute relevance score for a pair
#     2. rerank(query, candidates)    -- re-rank all candidates using cross_score
#
# APPROACH FOR cross_score():
#   Measure how well the document RESPONDS TO the query by looking at:
#   - Token overlap: what fraction of query words are in the document?
#   - Two-word phrase matching: does the doc contain pairs of query words together?
#   - Exact query fragment: does the doc contain the full query (rare but high signal)?
#
# HOW TO RUN:
#   python exercises/exercise_02_reranker.py

import numpy as np     # For any numerical operations needed


# ============================================================
# PROVIDED: Test data
# ============================================================
# Query: "best Python web framework for building REST APIs"
# Bi-encoder returned these 10 results with their scores.
# They are roughly ordered but NOT perfectly -- some relevant docs
# are ranked too low, some less relevant docs too high.

QUERY = "best Python web framework for building REST APIs"

# Each tuple: (document_text, bi_encoder_score)
BI_ENCODER_RESULTS = [
    ("Django REST framework tutorial and setup guide",                         0.84),
    ("Python web development overview and history",                            0.82),  # too high
    ("Flask API development with route handlers",                              0.81),
    ("Top Python frameworks: Django, Flask, FastAPI",                         0.79),
    ("FastAPI building REST APIs with Python type hints",                      0.78),  # too low!
    ("Introduction to Python programming language",                            0.77),  # too high
    ("Django ORM and database models guide",                                   0.75),
    ("REST API design best practices and standards",                           0.73),
    ("Python vs Java for backend web development",                             0.71),
    ("Web framework performance benchmark Django Flask FastAPI",               0.69),
]

# What the ideal ranking should look like (human judgment):
# 1. FastAPI building REST APIs        -- directly answers the query
# 2. Django REST framework tutorial   -- very relevant
# 3. Flask API development            -- very relevant
# 4. Top Python frameworks list       -- helpful context
# 5. REST API design best practices   -- tangentially relevant
# 6. Web framework benchmark          -- somewhat relevant
# 7. Django ORM guide                 -- less relevant
# 8. Python web development overview  -- generic
# 9. Python vs Java                   -- very tangential
# 10. Introduction to Python          -- not relevant


# ============================================================
# YOUR TASK: Fill in these two functions
# ============================================================

def cross_score(query, document):
    """
    Compute a relevance score for a (query, document) pair.

    A real cross-encoder runs both texts through a transformer together.
    We simulate this by computing text overlap features.

    STEPS TO IMPLEMENT:
    1. Tokenize query: call .lower().split() and strip punctuation
    2. Tokenize document: same
    3. Find shared tokens (words in both)
    4. Compute overlap_ratio = len(shared) / len(query_tokens)
    5. Check for 2-word phrase matches in doc_text
       (for each consecutive pair of query words, check if "word1 word2" appears in doc)
       Add +0.15 per matching phrase
    6. Combine: final_score = 0.6 * overlap_ratio + phrase_bonus
    7. Clamp to [0.0, 1.0] using min(1.0, max(0.0, score))
    8. Return final_score

    HINT for tokenizing with punctuation removal:
        words = text.lower().split()
        clean = set()
        for w in words:
            w = w.strip(".,!?;:\"'()[]")
            if w:
                clean.add(w)

    Parameters:
        query    (str): The search query
        document (str): A candidate document text

    Returns:
        float: Relevance score from 0.0 to 1.0
    """
    # TODO: Implement this function
    # Step 1: Tokenize query into a set of words (strip punctuation)
    query_tokens = set()
    # YOUR CODE HERE

    # Step 2: Tokenize document into a set of words (strip punctuation)
    doc_tokens = set()
    # YOUR CODE HERE

    # Step 3: Find shared tokens
    shared = set()
    # YOUR CODE HERE

    # Step 4: Compute overlap ratio
    overlap_ratio = 0.0
    # YOUR CODE HERE (guard against len(query_tokens) == 0)

    # Step 5: Check for 2-word phrase matches
    phrase_bonus = 0.0
    # YOUR CODE HERE
    # HINT:
    #   query_words_list = query.lower().split()
    #   doc_lower = document.lower()
    #   for i in range(len(query_words_list) - 1):
    #       phrase = query_words_list[i] + " " + query_words_list[i+1]
    #       if phrase in doc_lower:
    #           phrase_bonus += 0.15

    # Step 6: Combine factors
    final_score = 0.0
    # YOUR CODE HERE: final_score = 0.6 * overlap_ratio + phrase_bonus

    # Step 7: Clamp to [0.0, 1.0]
    # YOUR CODE HERE: final_score = min(1.0, max(0.0, final_score))

    return final_score  # Step 8: return


def rerank(query, candidates_with_scores, top_n=5):
    """
    Re-rank a list of candidates using cross_score().

    STEPS TO IMPLEMENT:
    1. For each (document, bi_score) in candidates_with_scores:
       a. Call cross_score(query, document) to get the cross-encoder score
       b. Store (document, cross_score, bi_score) in a list
    2. Sort the list by cross_score, highest first
    3. Return the top_n items

    Parameters:
        query                  (str):  The search query
        candidates_with_scores (list): List of (document_text, bi_score) tuples
        top_n                  (int):  How many re-ranked results to return

    Returns:
        List of (document_text, cross_score, original_bi_score) tuples
    """
    # TODO: Implement this function
    reranked = []

    for doc_text, bi_score in candidates_with_scores:
        # Step 1: Get cross-encoder score for this (query, doc) pair
        cs = 0.0  # YOUR CODE HERE: replace with cross_score(query, doc_text)
        # Step 1b: Store result
        reranked.append((doc_text, cs, bi_score))

    # Step 2: TODO -- Sort by cross_score descending
    # YOUR CODE HERE

    # Step 3: Return top_n
    return reranked[:top_n]  # This is correct -- just implement the sort above


# ============================================================
# TEST AND DEMO -- Do not modify
# ============================================================

def compute_rank_changes(bi_results, reranked_results):
    """
    Compare original bi-encoder ranking to re-ranked ordering.
    Shows which documents moved up or down.
    """
    # Build lookup of original rank
    original_rank = {doc: rank for rank, (doc, _) in enumerate(bi_results, start=1)}

    print(f"\n{'Document (first 50 chars)':<52} {'BI':>4} {'CE':>4} {'Change':>8}")
    print("-" * 75)

    for new_rank, (doc, cross, bi) in enumerate(reranked_results, start=1):
        old_rank = original_rank.get(doc, 99)
        delta = old_rank - new_rank
        if delta > 0:
            change = f"+{delta} up"
        elif delta < 0:
            change = f"{delta} dn"
        else:
            change = "same"
        print(f"  {doc[:50]:<50} {old_rank:>4} {new_rank:>4} {change:>8}")


def run_tests():
    """Test your implementation."""
    print("\nRunning tests...")
    passed = 0
    failed = 0

    # Test 1: cross_score returns float in [0, 1]
    score = cross_score("python web api", "Python web API development guide")
    if isinstance(score, float) and 0.0 <= score <= 1.0:
        print(f"[PASS] cross_score returns float in [0,1]: {score:.3f}")
        passed += 1
    else:
        print(f"[FAIL] cross_score should return float in [0,1], got: {score}")
        failed += 1

    # Test 2: Highly relevant doc scores higher than irrelevant doc
    relevant_score = cross_score(
        "best python web framework",
        "Flask is a Python web framework for building REST APIs"
    )
    irrelevant_score = cross_score(
        "best python web framework",
        "Introduction to cooking and recipes"
    )
    if relevant_score > irrelevant_score:
        print(f"[PASS] Relevant doc ({relevant_score:.3f}) > Irrelevant doc ({irrelevant_score:.3f})")
        passed += 1
    else:
        print(f"[FAIL] Relevant doc should score higher. Got {relevant_score:.3f} vs {irrelevant_score:.3f}")
        failed += 1

    # Test 3: rerank returns a list
    result = rerank(QUERY, BI_ENCODER_RESULTS, top_n=5)
    if isinstance(result, list) and len(result) == 5:
        print(f"[PASS] rerank returns list of 5 items")
        passed += 1
    else:
        print(f"[FAIL] rerank should return list of 5 items, got: {type(result)} len {len(result) if result else '?'}")
        failed += 1

    # Test 4: FastAPI doc should be in top 3 after re-ranking
    if result:
        top3_docs = [doc for doc, _, _ in result[:3]]
        fastapi_in_top3 = any("FastAPI" in doc or "REST API" in doc or "flask" in doc.lower()
                              for doc in top3_docs)
        if fastapi_in_top3:
            print("[PASS] Highly relevant API docs appear in top 3 after re-ranking")
            passed += 1
        else:
            print(f"[FAIL] Expected API-focused doc in top 3. Top 3: {[d[:40] for d in top3_docs]}")
            failed += 1

    # Test 5: Generic Python intro doc should drop in ranking
    if result:
        all_docs = [doc for doc, _, _ in result]
        python_intro_rank = next(
            (i+1 for i, d in enumerate(all_docs) if "Introduction to Python" in d),
            None
        )
        if python_intro_rank is None or python_intro_rank >= 4:
            print("[PASS] Generic 'Introduction to Python' does not appear in top 3")
            passed += 1
        else:
            print(f"[FAIL] 'Introduction to Python' should rank lower (got rank {python_intro_rank})")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":

    print("=" * 60)
    print("  EXERCISE 2: Build a Re-ranker")
    print("=" * 60)
    print(f"\nQuery: '{QUERY}'")

    print("\n--- BI-ENCODER RESULTS (original order) ---")
    for rank, (doc, score) in enumerate(BI_ENCODER_RESULTS, start=1):
        print(f"  Rank {rank:2d} [{score:.2f}]: {doc[:60]}")

    print("\n--- RUNNING YOUR RE-RANKER ---")
    try:
        reranked = rerank(QUERY, BI_ENCODER_RESULTS, top_n=10)

        if reranked and reranked[0][1] > 0:
            print("\n--- CROSS-ENCODER RE-RANKED RESULTS ---")
            for rank, (doc, cross, bi) in enumerate(reranked, start=1):
                print(f"  Rank {rank:2d} | Cross: {cross:.3f} | BI: {bi:.2f} | {doc[:55]}")

            compute_rank_changes(BI_ENCODER_RESULTS, reranked)

            run_tests()
        else:
            print("\nNOTE: cross_score is returning 0 for all docs.")
            print("Fill in the TODO sections in cross_score() and rerank().")
            run_tests()

    except Exception as ex:
        print(f"Error running solution: {ex}")
        print("Fill in the TODO sections above.")


# ============================================================
# SOLUTION (Hidden -- Try yourself first!)
# ============================================================
# def cross_score(query, document):
#     def tok(text):
#         words = text.lower().split()
#         clean = set()
#         for w in words:
#             w = w.strip(".,!?;:\"'()[]")
#             if w:
#                 clean.add(w)
#         return clean
#
#     query_tokens = tok(query)
#     doc_tokens = tok(document)
#
#     if not query_tokens:
#         return 0.0
#
#     shared = query_tokens.intersection(doc_tokens)
#     overlap_ratio = len(shared) / len(query_tokens)
#
#     phrase_bonus = 0.0
#     qwords = query.lower().split()
#     doc_lower = document.lower()
#     for i in range(len(qwords) - 1):
#         phrase = qwords[i] + " " + qwords[i+1]
#         if phrase in doc_lower:
#             phrase_bonus += 0.15
#
#     final_score = 0.6 * overlap_ratio + phrase_bonus
#     return min(1.0, max(0.0, final_score))
#
# def rerank(query, candidates_with_scores, top_n=5):
#     reranked = []
#     for doc_text, bi_score in candidates_with_scores:
#         cs = cross_score(query, doc_text)
#         reranked.append((doc_text, cs, bi_score))
#     reranked.sort(key=lambda x: x[1], reverse=True)
#     return reranked[:top_n]

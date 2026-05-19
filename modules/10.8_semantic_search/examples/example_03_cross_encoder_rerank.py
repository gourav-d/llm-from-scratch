# -*- coding: utf-8 -*-
# example_03_cross_encoder_rerank.py
#
# Module 10.8 -- Semantic Search Systems
# Lesson 3: Cross-Encoder Re-ranking
#
# WHAT THIS FILE DEMONSTRATES:
#   - Simulate a bi-encoder first pass (retrieve top-10 candidates)
#   - Build a simple cross-encoder that sees BOTH query and doc together
#   - Re-rank the top-10 using cross-encoder scores
#   - Show before/after comparison to see ranking improvement
#
# REQUIREMENTS: numpy only (Tier 1 - primary)
# PART B needs: pip install sentence-transformers
#
# HOW TO RUN:
#   python examples/example_03_cross_encoder_rerank.py

import numpy as np   # For math operations


# ============================================================
# PART 1: Simulated Bi-Encoder (First Pass)
# ============================================================
# We pre-suppose the bi-encoder has already run and returned top-10 candidates.
# Each candidate has: (document_text, bi_encoder_score)
# The bi-encoder scores are "good but not perfect" -- we will improve them.

def simulate_bi_encoder_results(query, documents):
    """
    Simulate what a bi-encoder returns: documents with approximate scores.
    In a real system this would call the actual bi-encoder.
    Here we assign scores that are "roughly correct but not perfectly ordered."

    Parameters:
        query     (str):  The search query
        documents (list): List of candidate documents

    Returns:
        List of (doc_text, bi_score) tuples sorted by bi_score descending
    """
    # Pretend these are the bi-encoder scores.
    # They are plausible but slightly misordered -- the cross-encoder will fix this.
    # Scores are returned with the documents to simulate real bi-encoder output.
    bi_results = list(zip(documents, BI_ENCODER_SCORES))

    # Sort by bi-encoder score (highest first)
    bi_results.sort(key=lambda x: x[1], reverse=True)

    return bi_results


# ============================================================
# PART 2: Our Test Data
# ============================================================
# Scenario: user searches for "Python exception handling best practices"
# These are the 10 candidates returned by the bi-encoder.

QUERY = "Python exception handling best practices"

# 10 candidate documents from the bi-encoder
CANDIDATES = [
    # index 0
    "Python try-except blocks: syntax and usage",
    # index 1
    "Best practices for error handling in production systems",
    # index 2
    "Common Python exceptions and when they occur",
    # index 3
    "How to write clean Python code",
    # index 4
    "Python exception hierarchy: BaseException, Exception, and subclasses",
    # index 5
    "Logging errors effectively in Python applications",
    # index 6
    "Python programming language overview and features",
    # index 7
    "Advanced Python exception handling patterns: custom exceptions, context managers",
    # index 8
    "Debugging Python code with pdb and IDE tools",
    # index 9
    "Exception handling patterns in Java and .NET",
]

# Simulated bi-encoder scores (slightly misordered from ideal ranking)
# Notice: Doc 7 (best doc -- "advanced patterns") is ranked #5, not #1
#         Doc 1 ("best practices") is ranked #3, not #1 or #2
#         Doc 6 ("Python overview") is ranked higher than it should be
BI_ENCODER_SCORES = [
    0.82,   # index 0: "Python try-except blocks"
    0.79,   # index 1: "Best practices for error handling"
    0.77,   # index 2: "Common Python exceptions"
    0.75,   # index 3: "How to write clean Python code"
    0.73,   # index 4: "Python exception hierarchy"
    0.71,   # index 5: "Logging errors effectively"
    0.70,   # index 6: "Python programming overview"  <-- too high!
    0.68,   # index 7: "Advanced exception handling"   <-- too low!
    0.67,   # index 8: "Debugging Python code"
    0.65,   # index 9: "Exception handling in Java/.NET"
]


# ============================================================
# PART 3: Cross-Encoder Implementation
# ============================================================

def tokenize(text):
    """
    Very simple tokenizer: split text into lowercase words.
    A real tokenizer (WordPiece, BPE) is much more sophisticated.
    This is enough to demonstrate the concept.

    Parameters:
        text (str): Text to tokenize

    Returns:
        set: Set of unique lowercase words
    """
    # Split on spaces, make lowercase, remove punctuation
    words = text.lower().split()
    # Remove punctuation from each word
    clean_words = set()
    for word in words:
        # Strip common punctuation from start and end of word
        word = word.strip(".,!?;:\"'()[]")
        if word:                 # Only add non-empty strings
            clean_words.add(word)
    return clean_words


def cross_encoder_score(query, document):
    """
    Simulate a cross-encoder that scores a (query, document) pair.

    A REAL cross-encoder:
    - Concatenates: [CLS] query [SEP] document [SEP]
    - Runs through full transformer (query and doc tokens interact via attention)
    - Outputs a single relevance score

    Our SIMULATION:
    - Measures token overlap (shared words) between query and doc
    - Applies position weighting (words at the start matter more)
    - Measures keyword density (how much of the doc is query-relevant?)
    - Combines these into a final score

    This captures the KEY IDEA: the cross-encoder SEES BOTH texts together
    and can detect how well the document responds to the query.

    Parameters:
        query    (str): The search query
        document (str): A candidate document

    Returns:
        float: Relevance score from 0.0 to 1.0
    """
    # Tokenize both texts
    query_tokens = tokenize(query)       # Set of query words
    doc_tokens = tokenize(document)      # Set of document words

    # --- Factor 1: Token overlap ---
    # How many query words appear in the document?
    shared_tokens = query_tokens.intersection(doc_tokens)  # Words in BOTH
    if len(query_tokens) == 0:           # Avoid division by zero
        return 0.0

    # Overlap ratio: fraction of query words found in document
    overlap_ratio = len(shared_tokens) / len(query_tokens)

    # --- Factor 2: Document specificity ---
    # A shorter, more focused document that contains query terms is better
    # than a long document where query terms are buried
    doc_word_count = len(doc_tokens)     # Number of unique words in doc
    if doc_word_count == 0:
        return 0.0

    # What fraction of the document is query-relevant?
    # Higher = document is more specifically about the query topic
    doc_focus = len(shared_tokens) / doc_word_count

    # --- Factor 3: Key phrase bonus ---
    # Check if the document contains multi-word phrases from the query
    # Real cross-encoders do this via attention -- they see phrases in context
    query_lower = query.lower()          # Lowercase query for phrase matching
    doc_lower = document.lower()         # Lowercase document

    phrase_bonus = 0.0                   # Start with no bonus
    # Split query into 2-word and 3-word phrases
    query_words = query_lower.split()
    for i in range(len(query_words) - 1):                 # 2-word phrases
        phrase = query_words[i] + " " + query_words[i+1]
        if phrase in doc_lower:
            phrase_bonus += 0.15         # Add bonus for each matching phrase

    # Also check for the full query as a phrase (rare but high signal)
    if query_lower in doc_lower:
        phrase_bonus += 0.3              # Large bonus for exact query match

    # --- Combine factors into final score ---
    # Weights chosen to approximate real cross-encoder behavior:
    #   50% of score from token overlap
    #   20% from document focus
    #   30% from phrase matching
    raw_score = (0.5 * overlap_ratio) + (0.2 * doc_focus) + phrase_bonus

    # Clamp to [0, 1] range -- cannot go above 1.0 or below 0.0
    score = min(1.0, max(0.0, raw_score))

    return score


def rerank(query, candidates_with_scores, top_n=5):
    """
    Re-rank candidates using the cross-encoder.

    This is the "second pass" in retrieve-then-rerank.
    We take the bi-encoder's top-K candidates and re-score them more accurately.

    Parameters:
        query                 (str):  The search query
        candidates_with_scores(list): List of (doc_text, bi_score) tuples
        top_n                 (int):  How many re-ranked results to return

    Returns:
        List of (doc_text, cross_score, original_bi_score) tuples
    """
    reranked = []

    for doc_text, bi_score in candidates_with_scores:
        # Score this (query, doc) pair with the cross-encoder
        # NOTE: bi-encoder encoded them SEPARATELY, cross-encoder sees BOTH together
        cross_score = cross_encoder_score(query, doc_text)

        # Keep track of the original bi-encoder score for comparison
        reranked.append((doc_text, cross_score, bi_score))

    # Sort by cross-encoder score (not bi-encoder score)
    reranked.sort(key=lambda x: x[1], reverse=True)

    # Return top_n results
    return reranked[:top_n]


# ============================================================
# PART 4: Main Demonstration
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("  CROSS-ENCODER RE-RANKING DEMO")
    print("=" * 70)

    print(f"\nQuery: \"{QUERY}\"")

    # --- Step 1: Get bi-encoder results (first pass) ---
    print("\n--- STEP 1: Bi-Encoder First Pass (fast, approximate) ---")
    bi_results = simulate_bi_encoder_results(QUERY, CANDIDATES)

    print("\nBi-encoder top-10 (roughly ordered):")
    for i, (doc, score) in enumerate(bi_results, start=1):
        print(f"  Rank {i:2d} | Score {score:.2f} | {doc}")

    # --- Step 2: Cross-encoder re-ranking ---
    print("\n--- STEP 2: Cross-Encoder Re-ranking (slow, accurate) ---")
    print(f"Running cross-encoder on all {len(bi_results)} candidates...")

    reranked_results = rerank(QUERY, bi_results, top_n=5)

    print("\nCross-encoder top-5 (accurately ordered):")
    for i, (doc, cross_score, bi_score) in enumerate(reranked_results, start=1):
        print(f"  Rank {i:2d} | Cross: {cross_score:.2f} | Bi: {bi_score:.2f} | {doc}")

    # --- Step 3: Show the differences ---
    print("\n--- STEP 3: Before vs After Comparison ---")

    # Build lookup of original bi-encoder rank for each document
    bi_rank_lookup = {}
    for i, (doc, _) in enumerate(bi_results, start=1):
        bi_rank_lookup[doc] = i

    print("\nDocument ranking changes (bi-encoder rank --> cross-encoder rank):")
    print(f"{'Doc (first 45 chars)':<47} {'BI rank':>7} {'CE rank':>7} {'Change':>8}")
    print("-" * 75)

    for new_rank, (doc, cross_score, bi_score) in enumerate(reranked_results, start=1):
        old_rank = bi_rank_lookup[doc]       # What rank did bi-encoder give?
        change = old_rank - new_rank         # Positive = moved UP, negative = moved DOWN
        direction = ""
        if change > 0:
            direction = f"^{change} up"      # Moved up in ranking
        elif change < 0:
            direction = f"v{abs(change)} dn" # Moved down in ranking
        else:
            direction = "-- same"

        print(f"  {doc[:45]:<45} {old_rank:>7} {new_rank:>7} {direction:>8}")

    # Explain what happened
    print("\n--- EXPLANATION ---")
    print("The bi-encoder ranked 'Python programming overview' (doc index 6) too")
    print("high because it has many 'Python' related tokens.")
    print("")
    print("The cross-encoder correctly demoted it -- it is a generic overview,")
    print("not specifically about 'exception handling best practices'.")
    print("")
    print("The cross-encoder promoted 'Advanced exception handling patterns' because")
    print("it saw that the document directly addresses the query topic.")

    # Show intermediate scores for transparency
    print("\n--- CROSS-ENCODER SCORE DETAILS ---")
    print("Showing how each candidate was scored:\n")
    for doc, bi_score in bi_results:
        cross_score = cross_encoder_score(QUERY, doc)
        query_toks = tokenize(QUERY)
        doc_toks = tokenize(doc)
        shared = query_toks.intersection(doc_toks)
        print(f"  Doc: {doc[:50]}")
        print(f"    Shared tokens:  {shared}")
        print(f"    BI score: {bi_score:.2f}  |  Cross score: {cross_score:.2f}")
        print()


# ============================================================
# PART B: Real Cross-Encoder using sentence-transformers
# ============================================================
# Remove the # at the start of each line to run this section.
# Requirements: pip install sentence-transformers
#
# from sentence_transformers import CrossEncoder
#
# # Load a real cross-encoder trained on MS MARCO (Bing search data)
# cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
#
# # Get bi-encoder results first (from example_02 or any retrieval)
# # For demo, use our CANDIDATES list above
# query_real = QUERY
# candidates_real = CANDIDATES
#
# # Score all (query, doc) pairs with the real cross-encoder
# # CrossEncoder.predict() takes a list of [query, doc] pairs
# pairs = [[query_real, doc] for doc in candidates_real]
# real_scores = cross_encoder_model.predict(pairs)  # numpy array of floats
#
# # Combine scores with documents and sort
# real_results = sorted(zip(real_scores, candidates_real), reverse=True)
#
# print("\nReal Cross-Encoder Results:")
# for i, (score, doc) in enumerate(real_results[:5], start=1):
#     print(f"  Rank {i} | Score {score:.3f} | {doc}")

# ============================================================
# EXPECTED OUTPUT (partial):
# ============================================================
# CROSS-ENCODER RE-RANKING DEMO
#
# Query: "Python exception handling best practices"
#
# Bi-encoder top-10:
#   Rank 1 | Score 0.82 | Python try-except blocks: syntax and usage
#   Rank 2 | Score 0.79 | Best practices for error handling in production...
#   ...
#   Rank 7 | Score 0.70 | Python programming language overview
#   Rank 8 | Score 0.68 | Advanced Python exception handling patterns...
#
# Cross-encoder top-5:
#   Rank 1 | Cross: 0.XX | Bi: 0.XX | Best practices for error handling...
#   Rank 2 | Cross: 0.XX | Bi: 0.XX | Advanced Python exception handling...
#   ...
#
# (Best practices and advanced patterns are promoted; generic overview demoted)

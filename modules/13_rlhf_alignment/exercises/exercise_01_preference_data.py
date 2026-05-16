"""
=============================================================================
MODULE 13 - EXERCISE 01: Working with Preference Data
=============================================================================

WHAT YOU WILL LEARN:
  - How preference data (chosen vs rejected pairs) is structured
  - How to measure preference accuracy (did the model rank things correctly?)
  - How to compute a simple alignment score from feature vectors
  - How to rank a list of responses from best to worst using only NumPy
  - How to generate all valid preference pairs from a set of responses

C# ANALOGY:
  - Preference pairs are like IComparable<T> in C# — they say "A is better than B"
  - Alignment score is like a custom IComparer that uses multiple criteria
  - Ranking is like sorting a List<Response> with a custom Comparator
  - Generating preference pairs is like a nested LINQ SelectMany over all (i, j) combos

INSTRUCTIONS:
  - Read each exercise description carefully
  - Fill in the code where you see # TODO
  - Run the file to check your output
  - Compare with the solutions at the bottom (in comments)

=============================================================================

PREFERENCE DATA DIAGRAM
=======================

  Response A: [features...] -> alignment_score = 0.72  <-- CHOSEN
  Response B: [features...] -> alignment_score = 0.41  <-- REJECTED

  Preference Pair = (Response A, Response B)
                     ^^chosen        ^^rejected

  Preference Accuracy = (# pairs where model scores chosen > rejected)
                        -----------------------------------------------
                                   (total # of pairs)

  If accuracy = 1.0  -> model ranks all pairs correctly (perfect)
  If accuracy = 0.5  -> model is no better than random (bad!)

=============================================================================
"""

# ---- GLOSSARY ---------------------------------------------------------------
GLOSSARY = {
    "Preference Pair":
        "(chosen, rejected) — a pair where chosen is a better response than rejected.",
    "Preference Accuracy":
        "Fraction of pairs where the model correctly scores chosen > rejected.",
    "Alignment Score":
        "A single number measuring how aligned (safe + helpful) a response is.",
    "Feature Vector":
        "[harm, honesty, helpfulness, manipulation, respect] — 5 numbers per response.",
    "Ranking":
        "Ordering responses from best (highest score) to worst (lowest score).",
    "All Preference Pairs":
        "For N responses, every combination (i, j) where score_i > score_j.",
}

print("=" * 70)
print("EXERCISE 01 — Preference Data")
print("=" * 70)
print("\nGLOSSARY:")
for term, defn in GLOSSARY.items():       # loop over glossary
    print(f"  {term}:\n    {defn}\n")     # print each definition

import numpy as np    # NumPy — the only library you need for these exercises

# =============================================================================
# SHARED DATA (used across multiple exercises)
# =============================================================================
# Each row = one response represented as a 5-dimensional feature vector:
#   [harm_score, honesty, helpfulness, manipulation, respect]
# Lower harm and manipulation = better.
# Higher honesty, helpfulness, respect = better.

# 6 candidate responses with varying quality
RESPONSES = np.array([
    [0.9, 0.4, 0.2, 0.8, 0.3],    # Response 0: very bad
    [0.6, 0.5, 0.4, 0.6, 0.5],    # Response 1: bad
    [0.4, 0.6, 0.6, 0.3, 0.7],    # Response 2: decent
    [0.2, 0.8, 0.8, 0.1, 0.8],    # Response 3: good
    [0.1, 0.9, 0.9, 0.1, 0.9],    # Response 4: very good
    [0.05, 0.95, 0.95, 0.05, 0.95],    # Response 5: excellent
], dtype=np.float32)    # float32 for consistency

# Feature indices — makes code easier to read
IDX_HARM        = 0    # harm score       (lower = better)
IDX_HONESTY     = 1    # honesty score    (higher = better)
IDX_HELPFULNESS = 2    # helpfulness      (higher = better)
IDX_MANIPULATION = 3   # manipulation     (lower = better)
IDX_RESPECT     = 4    # respect score    (higher = better)

# Some pre-labeled preference pairs (chosen_score, rejected_score)
# These simulate a reward model's output scores for a set of response pairs
MODEL_SCORES_CHOSEN   = np.array([0.85, 0.72, 0.91, 0.60, 0.78], dtype=np.float32)
MODEL_SCORES_REJECTED = np.array([0.43, 0.80, 0.55, 0.55, 0.30], dtype=np.float32)
# Pair 1: 0.85 vs 0.43 -> chosen wins (correct)
# Pair 2: 0.72 vs 0.80 -> chosen LOSES (model made an error here!)
# Pair 3: 0.91 vs 0.55 -> chosen wins (correct)
# Pair 4: 0.60 vs 0.55 -> chosen wins (correct, but barely)
# Pair 5: 0.78 vs 0.30 -> chosen wins (correct)

# =============================================================================
# EXERCISE 1 — Preference Accuracy
# =============================================================================
# Given MODEL_SCORES_CHOSEN and MODEL_SCORES_REJECTED,
# compute what fraction of pairs the model ranked correctly
# (i.e., how often is score_chosen > score_rejected?)
#
# Formula:  preference_accuracy = (# pairs where chosen_score > rejected_score)
#                                  / total_pairs
#
# Expected answer: 4 out of 5 pairs correct -> accuracy = 0.8
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 1 — Preference Accuracy")
print("=" * 60)
print("\n  Scores for chosen responses:  ", MODEL_SCORES_CHOSEN)
print("  Scores for rejected responses:", MODEL_SCORES_REJECTED)
print()
print("  Task: compute preference_accuracy")
print("  Hint: a pair is correct when chosen_score > rejected_score")
print("  Hint: np.sum() counts True values in a boolean array")
print()

def preference_accuracy(scores_chosen, scores_rejected):
    """
    Compute the fraction of preference pairs where
    the model correctly assigns a higher score to the chosen response.

    Parameters:
      scores_chosen   : np.array of shape [N] — model scores for chosen responses
      scores_rejected : np.array of shape [N] — model scores for rejected responses

    Returns:
      float — accuracy in [0.0, 1.0]
    """
    # TODO: Step 1 — create a boolean array: True where chosen > rejected
    correct = None    # replace None with your comparison

    # TODO: Step 2 — count how many are True
    num_correct = None    # replace None with np.sum(...)

    # TODO: Step 3 — divide by total number of pairs to get accuracy
    total_pairs = None    # replace None with the total count
    accuracy = None       # replace None with num_correct / total_pairs

    return accuracy    # should return a float

# --- Run exercise 1 ---
result_1 = preference_accuracy(MODEL_SCORES_CHOSEN, MODEL_SCORES_REJECTED)
print(f"  Your preference_accuracy = {result_1}")
print(f"  Expected: 0.8  (4 out of 5 pairs correct)")

# =============================================================================
# EXERCISE 2 — Alignment Score from Features
# =============================================================================
# Given a single response feature vector, compute its alignment score.
#
# Formula:
#   alignment_score = average of [honesty, helpfulness, respect]
#                     MINUS average of [harm, manipulation]
#                     then shift by +0.5 and clamp to [0, 1]
#
# The shift ensures scores stay positive.
#
# Example: [0.9, 0.4, 0.2, 0.8, 0.3]
#   positive = (0.4 + 0.2 + 0.3) / 3 = 0.3
#   negative = (0.9 + 0.8) / 2 = 0.85
#   raw = 0.3 - 0.85 = -0.55
#   shifted = -0.55 + 0.5 = -0.05
#   clamped = 0.0  (because -0.05 < 0)
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 2 — Alignment Score from Features")
print("=" * 60)
print("\n  Feature vector layout: [harm, honesty, helpfulness, manipulation, respect]")
print("  RESPONSES[0] =", RESPONSES[0], " <- very bad response")
print("  RESPONSES[4] =", RESPONSES[4], " <- very good response")
print()
print("  Task: implement alignment_score(features)")
print("  Hint: np.mean() computes the average of an array slice")
print("  Hint: np.clip(value, 0.0, 1.0) clamps to [0, 1]")
print()

def alignment_score(features):
    """
    Compute a single alignment score for a response feature vector.

    Parameters:
      features : np.array of shape [5] — [harm, honesty, helpful, manip, respect]

    Returns:
      float in [0.0, 1.0]
    """
    # TODO: Step 1 — compute average of the GOOD features (honesty, helpfulness, respect)
    # These are at indices IDX_HONESTY, IDX_HELPFULNESS, IDX_RESPECT
    positive_avg = None    # replace with np.mean(features[[IDX_HONESTY, ...]])

    # TODO: Step 2 — compute average of the BAD features (harm, manipulation)
    negative_avg = None    # replace with np.mean(features[[IDX_HARM, IDX_MANIPULATION]])

    # TODO: Step 3 — raw score = positive_avg - negative_avg
    raw_score = None

    # TODO: Step 4 — shift by +0.5 to make score positive when balanced
    shifted = None

    # TODO: Step 5 — clamp to [0.0, 1.0] using np.clip
    clamped = None

    return float(clamped)    # convert to Python float

# --- Run exercise 2 ---
for i, resp in enumerate(RESPONSES):    # test on all 6 responses
    score = alignment_score(resp)
    print(f"  Response {i}: features={resp}  ->  alignment_score={score:.4f}")

print("\n  Expected pattern: scores should INCREASE from Response 0 to Response 5")

# =============================================================================
# EXERCISE 3 — Rank Responses Best to Worst
# =============================================================================
# Given a list of response feature vectors, compute the alignment score
# for each response, then sort them from HIGHEST score to LOWEST score.
# Do NOT use Python's built-in sort() — use NumPy only.
#
# Hint: np.argsort() returns the indices that would sort an array.
#       np.argsort()[::-1] reverses the order (descending = best first).
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 3 — Rank Responses Best to Worst (NumPy only)")
print("=" * 60)
print()
print("  Task: compute alignment scores for all RESPONSES,")
print("        then return them ranked best-to-worst using np.argsort")
print()
print("  Hint: np.argsort(arr)       -> indices that sort arr ASCENDING")
print("  Hint: np.argsort(arr)[::-1] -> indices that sort arr DESCENDING")
print()

def rank_responses(responses):
    """
    Rank a list of response feature vectors from best to worst alignment.

    Parameters:
      responses : np.array of shape [N, 5]

    Returns:
      ranked_indices : np.array of shape [N] — response indices best to worst
      ranked_scores  : np.array of shape [N] — corresponding alignment scores
    """
    # TODO: Step 1 — compute alignment_score for every response
    # Hint: use a list comprehension: [alignment_score(r) for r in responses]
    # Then convert to np.array
    scores = None    # replace with np.array([alignment_score(r) for r in responses])

    # TODO: Step 2 — get argsort indices (ascending by default)
    sorted_indices_asc = None    # replace with np.argsort(scores)

    # TODO: Step 3 — reverse to get descending order (best first)
    ranked_indices = None    # replace with sorted_indices_asc[::-1]

    # TODO: Step 4 — index into scores with ranked_indices to get sorted scores
    ranked_scores = None    # replace with scores[ranked_indices]

    return ranked_indices, ranked_scores

# --- Run exercise 3 ---
indices, scores = rank_responses(RESPONSES)
if indices is not None and scores is not None:    # only print if not None
    print("  Ranked responses (best -> worst):")
    for rank, (idx, score) in enumerate(zip(indices, scores)):
        print(f"    Rank {rank+1}: Response {idx}  score={score:.4f}  features={RESPONSES[idx]}")
else:
    print("  (implement rank_responses to see output)")

# =============================================================================
# EXERCISE 4 — Generate All Valid Preference Pairs
# =============================================================================
# Given a list of responses with their scores, generate ALL preference pairs
# (response_i, response_j) where score_i > score_j.
#
# For 6 responses there are C(6,2) = 15 unique pairs to compare.
# Each comparison either yields a (chosen, rejected) pair or is skipped
# if the scores are equal.
#
# C# Analogy:
#   var pairs = from i in Enumerable.Range(0, N)
#               from j in Enumerable.Range(0, N)
#               where i < j && score[i] > score[j]
#               select (responses[i], responses[j]);
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 4 — Generate All Valid Preference Pairs")
print("=" * 60)
print()
print("  Task: for every pair (i, j) where i < j,")
print("        if score_i > score_j  -> add (response_i as chosen, response_j as rejected)")
print("        if score_j > score_i  -> add (response_j as chosen, response_i as rejected)")
print("        if tied               -> skip (no preference)")
print()
print("  Hint: use a nested for loop: for i in range(N): for j in range(i+1, N):")
print()

def generate_preference_pairs(responses):
    """
    Generate all valid preference pairs from a list of responses.

    Parameters:
      responses : np.array of shape [N, 5]

    Returns:
      pairs : list of (chosen_features, rejected_features) tuples
              where alignment_score(chosen) > alignment_score(rejected)
    """
    pairs = []    # start with an empty list of pairs

    # TODO: Step 1 — compute alignment scores for all responses
    scores = None    # replace with [alignment_score(r) for r in responses]

    # TODO: Step 2 — loop over all unique pairs (i, j) where i < j
    N = len(responses)    # number of responses
    for i in range(N):
        for j in range(i + 1, N):    # j starts at i+1 to avoid duplicates
            # TODO: Step 3 — compare scores[i] and scores[j]
            # If scores[i] > scores[j]: chosen=responses[i], rejected=responses[j]
            # If scores[j] > scores[i]: chosen=responses[j], rejected=responses[i]
            # If equal: skip (do not append)
            pass    # replace this with your if/elif/else logic

    return pairs    # return the list of (chosen, rejected) pairs

# --- Run exercise 4 ---
pairs = generate_preference_pairs(RESPONSES)
if pairs:    # if list is not empty
    print(f"  Generated {len(pairs)} preference pairs from {len(RESPONSES)} responses")
    print(f"  (Expected: 15 pairs from C(6,2) = 15 unique comparisons)")
    print(f"\n  First 5 pairs:")
    for k, (chosen, rejected) in enumerate(pairs[:5]):    # show first 5 only
        score_c = alignment_score(chosen)
        score_r = alignment_score(rejected)
        print(f"    Pair {k+1}: chosen_score={score_c:.3f}  rejected_score={score_r:.3f}")
else:
    print("  (implement generate_preference_pairs to see output)")

# =============================================================================
# SOLUTIONS (read ONLY after attempting the exercises yourself!)
# =============================================================================
"""
SOLUTION 1 — Preference Accuracy:

def preference_accuracy(scores_chosen, scores_rejected):
    correct = scores_chosen > scores_rejected       # boolean array, True where model is right
    num_correct = np.sum(correct)                   # count the True values
    total_pairs = len(scores_chosen)                # total number of pairs
    accuracy = num_correct / total_pairs            # fraction correct
    return accuracy
# Result: 4/5 = 0.8


SOLUTION 2 — Alignment Score:

def alignment_score(features):
    positive_avg = np.mean(features[[IDX_HONESTY, IDX_HELPFULNESS, IDX_RESPECT]])
    negative_avg = np.mean(features[[IDX_HARM, IDX_MANIPULATION]])
    raw_score = positive_avg - negative_avg
    shifted = raw_score + 0.5
    clamped = np.clip(shifted, 0.0, 1.0)
    return float(clamped)


SOLUTION 3 — Rank Responses:

def rank_responses(responses):
    scores = np.array([alignment_score(r) for r in responses])
    sorted_indices_asc = np.argsort(scores)
    ranked_indices = sorted_indices_asc[::-1]
    ranked_scores = scores[ranked_indices]
    return ranked_indices, ranked_scores


SOLUTION 4 — Generate Preference Pairs:

def generate_preference_pairs(responses):
    pairs = []
    scores = [alignment_score(r) for r in responses]
    N = len(responses)
    for i in range(N):
        for j in range(i + 1, N):
            if scores[i] > scores[j]:
                pairs.append((responses[i], responses[j]))    # i is better
            elif scores[j] > scores[i]:
                pairs.append((responses[j], responses[i]))    # j is better
            # else: tied, skip
    return pairs
# Expected: 15 pairs (all 15 comparisons result in a winner since all scores differ)
"""

print("\n" + "=" * 70)
print("EXERCISE 01 COMPLETE — Check your output, then read the solutions!")
print("=" * 70)

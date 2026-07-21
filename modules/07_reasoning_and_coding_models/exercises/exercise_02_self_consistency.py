"""
Module 07 - Reasoning & Coding Models
Exercise 02: Self-Consistency Reasoning

GLOSSARY
--------
Self-Consistency  : Run the SAME question through the model N times (with high
                    temperature), collect N different answers, then pick the
                    answer that appears most often (majority vote).
                    Why? Errors are random; the correct answer is consistent.
Temperature       : Controls randomness of generation. High temp (0.7-1.0) gives
                    varied reasoning paths. Low temp (0.0) gives greedy answers.
Majority Vote     : Pick the answer that appears most in a set of responses.
                    Like asking 10 doctors for a diagnosis and going with the
                    most common one.
Confidence Score  : Fraction of paths that agree on the winning answer.
                    confidence = count(winner) / total_paths
                    High confidence = model is sure. Low = uncertain.
Reasoning Path    : One complete chain-of-thought attempt from start to answer.
                    Self-consistency uses multiple paths in parallel.
Weighted Vote     : Votes weighted by response quality score, not equal weight.
                    Higher-quality reasoning gets more influence on final answer.
Answer Cluster    : Group of responses that all reached the same answer.
                    Majority vote = pick the largest cluster.
"""

from collections import Counter   # Counter: count occurrences (like Dictionary<T,int> in C#)

print("=" * 60)
print("Exercise 02: Self-Consistency Reasoning")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Majority Vote Over Multiple Answers
#
#  Background:
#    Self-consistency generates N answers and picks the most common.
#    This is called "majority vote".
#
#    Example:
#      answers = ["42", "42", "41", "42", "43"]
#      majority_vote -> "42"  (appears 3/5 times)
#
#    If there's a tie, return the first answer that reached max count.
#
#  Your Task:
#    Write: majority_vote(answers) -> str
#    Returns the most common answer in the list.
#    Use Counter from the collections module (already imported).
#
#  C# Analogy:
#    answers.GroupBy(a => a)
#           .OrderByDescending(g => g.Count())
#           .First().Key
# ============================================================

print("-" * 50)
print("EXERCISE 1: Majority Vote")
print("-" * 50)
print()


def majority_vote(answers):
    """
    Pick the most common answer from a list of model responses.

    Parameters:
        answers (list of str): Answer strings from multiple model runs.

    Returns:
        str: The answer that appears most frequently.
    """
    # TODO:
    # counts = Counter(answers)
    # return counts.most_common(1)[0][0]   <- most_common returns [(answer, count), ...]
    pass  # Replace with your implementation


answers_clear  = ["42", "42", "41", "42", "43"]           # Clear winner: "42"
answers_tied   = ["Paris", "London", "Paris", "London"]    # Tie: first max wins
answers_single = ["$60"]                                   # Only one answer

v1 = majority_vote(answers_clear)
v2 = majority_vote(answers_tied)
v3 = majority_vote(answers_single)

if v1 is not None:
    print(f"  Clear winner   : {v1}   (expected: '42')")
    print(f"  Tie            : {v2}   (expected: 'Paris' or 'London')")
    print(f"  Single answer  : {v3}   (expected: '$60')")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Confidence Score
#
#  Background:
#    After majority vote, we want to know HOW confident we are.
#    Confidence = fraction of paths that agree with the winner.
#
#    confidence = count(winning_answer) / total_paths
#
#    Examples:
#      ["42","42","42","42","42"] -> confidence = 1.0   (unanimous)
#      ["42","42","41","43","40"] -> confidence = 0.4   (weak)
#      ["42","42","41","42","43"] -> confidence = 0.6
#
#    Interpretation:
#      >= 0.8 : High confidence  (safe to trust the answer)
#      >= 0.5 : Medium confidence
#      < 0.5  : Low confidence   (model is uncertain; consider re-sampling)
#
#  Your Task:
#    Write: confidence_score(answers) -> dict
#    Returns: {"winner": str, "confidence": float, "level": str}
#    level = "high" (>=0.8), "medium" (>=0.5), or "low" (<0.5)
#
#  C# Analogy:
#    double confidence = winnerCount / (double)answers.Count;
# ============================================================

print("-" * 50)
print("EXERCISE 2: Confidence Score")
print("-" * 50)
print()


def confidence_score(answers):
    """
    Compute majority vote winner and confidence level.

    Parameters:
        answers (list of str): Answers from multiple model runs.

    Returns:
        dict: {
            "winner"     : str   -- most common answer,
            "confidence" : float -- fraction agreeing with winner (0-1),
            "level"      : str   -- "high" / "medium" / "low"
        }
    """
    # TODO:
    # 1. Use majority_vote(answers) to find winner
    # 2. confidence = answers.count(winner) / len(answers)
    # 3. level = "high" if confidence >= 0.8 else "medium" if >= 0.5 else "low"
    pass  # Replace with your implementation


test_cases = [
    ["42", "42", "42", "42", "42"],          # unanimous
    ["42", "42", "41", "42", "43"],          # clear but not unanimous
    ["42", "41", "43", "44", "42"],          # weak majority
    ["42", "41", "43", "44", "45"],          # very weak
]

print(f"  {'Answers':<40} {'Winner':>8}  {'Conf':>8}  {'Level':>8}")
print("  " + "-" * 70)
for answers in test_cases:
    r = confidence_score(answers)
    if r:
        disp = str(answers)[:38]
        print(f"  {disp:<40} {r['winner']:>8}  {r['confidence']:>8.2f}  {r['level']:>8}")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Weighted Voting by Response Quality
#
#  Background:
#    Not all reasoning paths are equally trustworthy.
#    A longer, more detailed response likely has better reasoning.
#
#    Weighted vote: each answer contributes weight = its quality score.
#    Pick the answer with the highest total weight.
#
#    Example:
#      answers = ["42",   "42",   "41"]
#      weights = [0.9,    0.8,    0.7]
#      weighted_scores = {"42": 0.9+0.8, "41": 0.7}
#      winner = "42"  (total weight 1.7 > 0.7)
#
#  Your Task:
#    Write: weighted_vote(answers, weights) -> str
#    Returns the answer with the highest total weight.
#
#  C# Analogy:
#    answers.Zip(weights, (a, w) => (a, w))
#           .GroupBy(x => x.a)
#           .OrderByDescending(g => g.Sum(x => x.w))
#           .First().Key
# ============================================================

print("-" * 50)
print("EXERCISE 3: Weighted Voting by Quality")
print("-" * 50)
print()


def weighted_vote(answers, weights):
    """
    Pick the answer with the highest total weight.

    Parameters:
        answers (list of str)  : Answers from multiple model runs.
        weights (list of float): Quality score for each answer (same length).

    Returns:
        str: Answer with highest cumulative weight.
    """
    # TODO:
    # 1. Build a dict: score_totals = {}
    # 2. For each (answer, weight) pair:
    #      score_totals[answer] = score_totals.get(answer, 0) + weight
    # 3. Return the key with maximum value:
    #      return max(score_totals, key=score_totals.get)
    pass  # Replace with your implementation


# Case 1: majority says "42" but "41" has higher quality
answers1 = ["42",  "42",  "41",  "41"]
weights1 = [0.5,   0.5,   0.9,   0.9]

# Case 2: simple majority wins (all equal weight)
answers2 = ["Paris", "Paris", "London"]
weights2 = [1.0,     1.0,     1.0]

# Case 3: single response
answers3 = ["42"]
weights3 = [0.85]

w1 = weighted_vote(answers1, weights1)
w2 = weighted_vote(answers2, weights2)
w3 = weighted_vote(answers3, weights3)

if w1 is not None:
    print(f"  High-quality '41' vs majority '42' -> {w1}  (expected: '41')")
    print(f"  Equal weights, majority 'Paris'     -> {w2}  (expected: 'Paris')")
    print(f"  Single answer '42'                  -> {w3}  (expected: '42')")
print()


# ============================================================
#  EXERCISE 4
#  Topic: When to Use Self-Consistency
#
#  Background:
#    Self-consistency costs N × more tokens than a single call.
#    It's only worth it for HARD problems where the model makes errors.
#
#    Simple heuristic: use self-consistency when confidence < threshold.
#    If first-attempt confidence is already high, skip extra sampling.
#
#    Decision rule:
#      1. Sample once, compute confidence.
#      2. If confidence >= threshold -> use that answer (done).
#      3. Else -> sample N more times, run majority vote.
#
#  Your Task:
#    Write: should_use_self_consistency(first_answers, threshold=0.8) -> bool
#    first_answers: small sample (e.g., 3 answers) from first attempt
#    Returns True if confidence < threshold (we SHOULD sample more).
#    Returns False if confidence >= threshold (answer is already reliable).
#
#  C# Analogy:
#    Like a retry policy: if success rate < threshold, retry more times.
# ============================================================

print("-" * 50)
print("EXERCISE 4: When to Use Self-Consistency")
print("-" * 50)
print()


def should_use_self_consistency(first_answers, threshold=0.8):
    """
    Decide if we need more self-consistency sampling.

    Parameters:
        first_answers (list of str): Initial small sample of answers.
        threshold     (float)      : Confidence needed to stop sampling.

    Returns:
        bool: True = sample more (confidence too low).
              False = answer is reliable (confidence >= threshold).
    """
    # TODO:
    # 1. Compute confidence using confidence_score(first_answers)
    # 2. Return confidence_score_result["confidence"] < threshold
    pass  # Replace with your implementation


# Model is very confident on easy question -> don't need more samples
easy_answers  = ["42", "42", "42"]   # 100% agreement
# Model is uncertain on hard question -> need more samples
hard_answers  = ["42", "41", "43"]   # 33% agreement

r_easy = should_use_self_consistency(easy_answers)
r_hard = should_use_self_consistency(hard_answers)

if r_easy is not None:
    print(f"  Easy question (100% agreement) -> need more sampling? {r_easy}  (expected: False)")
    print(f"  Hard question (33% agreement)  -> need more sampling? {r_hard}  (expected: True)")
print()
print("  Insight: Self-consistency is most valuable when model is uncertain.")
print("           Saves tokens on easy problems, uses them on hard ones.")
print()

print("=" * 60)
print("All exercises complete!")
print("=" * 60)

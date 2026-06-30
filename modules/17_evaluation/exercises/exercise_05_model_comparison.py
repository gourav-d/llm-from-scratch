"""
Exercise 05: Model Comparison and Leaderboard Analysis
Module 17: LLM Evaluation & Benchmarks

TASKS:
  1. Implement normalize_score(raw, min_val, max_val) -- scale score to 0-100
  2. Implement rank_models(scores_dict)               -- rank by average score
  3. Implement compare_zero_few_shot(zero, few)       -- compute shot effect
  4. Implement detect_specialist(model_scores)        -- find suspiciously narrow model

Run:  python exercise_05_model_comparison.py
Deps: none (pure Python)
"""


# ─────────────────────────────────────────────────────────
# TASK 1: Normalize a Score
# ─────────────────────────────────────────────────────────

def normalize_score(raw, min_val, max_val):
    """
    Normalize a raw score to the range [0.0, 100.0].

    Some benchmarks use different scales (e.g., perplexity, loss).
    Normalizing lets us combine them in a fair comparison.

    Formula: normalized = (raw - min_val) / (max_val - min_val) * 100.0

    Args:
        raw:     float, the raw score value
        min_val: float, the minimum possible score (maps to 0.0)
        max_val: float, the maximum possible score (maps to 100.0)

    Returns:
        float: normalized score in [0.0, 100.0]
               Clamp to [0, 100] if raw is outside [min_val, max_val].
               Return 0.0 if min_val == max_val (avoid division by zero).

    Example:
        normalize_score(75.0, 0.0, 100.0)  -> 75.0   (already in range)
        normalize_score(3.0, 0.0, 10.0)    -> 30.0
        normalize_score(0.5, 0.0, 1.0)     -> 50.0
        normalize_score(-5.0, 0.0, 100.0)  -> 0.0    (clamped)

    HINT:
        if min_val == max_val:
            return 0.0
        result = (raw - min_val) / (max_val - min_val) * 100.0
        return max(0.0, min(100.0, result))
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Rank Models
# ─────────────────────────────────────────────────────────

def rank_models(scores_dict):
    """
    Rank models by their average score across all benchmarks.

    Args:
        scores_dict: dict mapping model_name (str) -> dict of benchmark_name -> score (float)
                     All models must have the same benchmark keys.

    Returns:
        list of tuples: [(rank, model_name, avg_score), ...]
                        sorted by avg_score descending (rank 1 = best).

    Example:
        scores = {
            "ModelA": {"MMLU": 80.0, "GSM8K": 70.0},
            "ModelB": {"MMLU": 60.0, "GSM8K": 90.0},
            "ModelC": {"MMLU": 70.0, "GSM8K": 75.0},
        }
        rank_models(scores)
        -> [(1, "ModelB", 75.0), (2, "ModelC", 72.5), (3, "ModelA", 75.0)]
        # Wait: ModelA avg = 75.0, ModelB avg = 75.0 -- tied! Both rank 1.
        # In practice, sort descending and assign sequential ranks.
        # Correct: [(1, "ModelA", 75.0), (1, "ModelB", 75.0), (3, "ModelC", 72.5)]
        # But for simplicity, just sort and assign 1,2,3... in sort order.
        -> [(1, "ModelA", 75.0), (2, "ModelB", 75.0), (3, "ModelC", 72.5)]

    HINT:
        averages = []
        for model, bench_scores in scores_dict.items():
            avg = sum(bench_scores.values()) / len(bench_scores)
            averages.append((model, avg))
        sorted_models = sorted(averages, key=lambda x: x[1], reverse=True)
        return [(rank + 1, name, avg) for rank, (name, avg) in enumerate(sorted_models)]
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Compare Zero-shot vs Few-shot
# ─────────────────────────────────────────────────────────

def compare_zero_few_shot(zero_shot_scores, few_shot_scores):
    """
    Compare zero-shot and few-shot benchmark results.

    For each benchmark, compute:
      - absolute improvement: few_shot - zero_shot
      - relative improvement: (few_shot - zero_shot) / zero_shot * 100%

    Args:
        zero_shot_scores: dict of benchmark_name -> float score
        few_shot_scores:  dict of benchmark_name -> float score (same keys)

    Returns:
        dict mapping benchmark_name -> dict with keys:
          "zero_shot":   float
          "few_shot":    float
          "abs_gain":    float (few - zero)
          "rel_gain_pct": float (% improvement relative to zero-shot)

    Example:
        zero = {"MMLU": 33.7, "HellaSwag": 35.0}
        few  = {"MMLU": 70.0, "HellaSwag": 78.5}
        compare_zero_few_shot(zero, few)
        -> {
            "MMLU": {"zero_shot": 33.7, "few_shot": 70.0, "abs_gain": 36.3, "rel_gain_pct": 107.7},
            "HellaSwag": {"zero_shot": 35.0, "few_shot": 78.5, "abs_gain": 43.5, "rel_gain_pct": 124.3},
           }

    HINT:
        result = {}
        for bench in zero_shot_scores:
            z = zero_shot_scores[bench]
            f = few_shot_scores[bench]
            abs_gain = f - z
            rel = (abs_gain / z) * 100.0 if z != 0 else 0.0
            result[bench] = {"zero_shot": z, "few_shot": f, "abs_gain": abs_gain, "rel_gain_pct": rel}
        return result
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Detect Specialist (Cherry-Picker) Model
# ─────────────────────────────────────────────────────────

def detect_specialist(model_scores, variance_threshold=200.0):
    """
    Detect whether a model is a "specialist" that excels at one benchmark
    but is weak on others -- a red flag in leaderboard comparisons.

    Compute the variance of the model's scores across benchmarks.
    High variance = specialist. Low variance = balanced.

    Args:
        model_scores:       dict of benchmark_name -> float score
        variance_threshold: float, variance above this is flagged as specialist
                            (default 200.0 corresponds to ~14 percentage points std dev)

    Returns:
        dict with:
          "mean":     float, average score
          "variance": float, score variance
          "std_dev":  float, standard deviation
          "is_specialist": bool, True if variance > threshold

    Example:
        balanced  = {"MMLU": 80.0, "GSM8K": 78.0, "HumanEval": 82.0}
        specialist = {"MMLU": 95.0, "GSM8K": 30.0, "HumanEval": 25.0}
        detect_specialist(balanced)   -> {"is_specialist": False, "variance": ~2.7, ...}
        detect_specialist(specialist) -> {"is_specialist": True,  "variance": ~1100, ...}

    HINT:
        scores = list(model_scores.values())
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        import math; std_dev = math.sqrt(variance)
        return {"mean": mean, "variance": variance, "std_dev": std_dev,
                "is_specialist": variance > variance_threshold}
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    import math

    print("=" * 55)
    print("  Exercise 05: Model Comparison and Leaderboard Analysis")
    print("=" * 55)

    # Task 1: normalize_score
    print("\n--- Task 1: normalize_score ---")
    cases = [
        (75.0, 0.0, 100.0, 75.0),
        (3.0,  0.0,  10.0, 30.0),
        (0.5,  0.0,   1.0, 50.0),
        (-5.0, 0.0, 100.0,  0.0),    # clamped
        (150.0, 0.0, 100.0, 100.0),  # clamped
        (5.0, 5.0, 5.0, 0.0),        # zero range
    ]
    for raw, mn, mx, exp in cases:
        result = normalize_score(raw, mn, mx)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if abs(result - exp) < 0.01 else "FAIL"
        print(f"  {status}  normalize({raw}, min={mn}, max={mx}) = {result:.1f}  (expected {exp:.1f})")

    # Task 2: rank_models
    print("\n--- Task 2: rank_models ---")
    scores = {
        "GPT-4":    {"MMLU": 86.4, "HellaSwag": 95.3, "ARC-C": 96.3, "TruthfulQA": 59.0},
        "LLaMA-70B":{"MMLU": 82.0, "HellaSwag": 88.0, "ARC-C": 87.6, "TruthfulQA": 62.8},
        "Mistral-7B":{"MMLU": 62.5, "HellaSwag": 81.0, "ARC-C": 60.0, "TruthfulQA": 56.0},
    }
    result_ranks = rank_models(scores)
    if result_ranks is None:
        print("  NOT IMPLEMENTED YET")
    else:
        print(f"  {'Rank':>6}  {'Model':<15}  {'Avg Score':>10}")
        print("  " + "-" * 35)
        for rank, model, avg in result_ranks:
            print(f"  {rank:>6}  {model:<15}  {avg:>10.2f}")
        top_model = result_ranks[0][1]
        status = "PASS" if top_model == "GPT-4" else "FAIL"
        print(f"  {status}  Top model is {top_model}  (expected GPT-4)")

    # Task 3: compare_zero_few_shot
    print("\n--- Task 3: compare_zero_few_shot ---")
    zero = {"HellaSwag": 33.7, "MMLU": 26.5, "ARC-C": 27.4}
    few  = {"HellaSwag": 78.5, "MMLU": 65.0, "ARC-C": 51.4}
    result_cmp = compare_zero_few_shot(zero, few)
    if result_cmp is None:
        print("  NOT IMPLEMENTED YET")
    else:
        print(f"  {'Benchmark':<12}  {'0-shot':>8}  {'5-shot':>8}  {'Abs Gain':>10}  {'Rel Gain':>10}")
        print("  " + "-" * 55)
        for bench, vals in result_cmp.items():
            print(f"  {bench:<12}  {vals['zero_shot']:>8.1f}  {vals['few_shot']:>8.1f}  "
                  f"{vals['abs_gain']:>+9.1f}  {vals['rel_gain_pct']:>9.1f}%")
        hs_gain = result_cmp["HellaSwag"]["abs_gain"]
        status = "PASS" if abs(hs_gain - 44.8) < 0.5 else "FAIL"
        print(f"  {status}  HellaSwag abs gain = {hs_gain:.1f}  (expected ~44.8)")

    # Task 4: detect_specialist
    print("\n--- Task 4: detect_specialist ---")
    balanced   = {"MMLU": 80.0, "GSM8K": 78.0, "HumanEval": 82.0, "TruthfulQA": 76.0}
    specialist = {"MMLU": 95.0, "GSM8K": 30.0, "HumanEval": 25.0, "TruthfulQA": 28.0}

    r_bal = detect_specialist(balanced)
    r_spe = detect_specialist(specialist)

    if r_bal is None or r_spe is None:
        print("  NOT IMPLEMENTED YET")
    else:
        ok_bal = "PASS" if not r_bal["is_specialist"] else "FAIL"
        ok_spe = "PASS" if r_spe["is_specialist"] else "FAIL"
        print(f"  {ok_bal}  Balanced model: is_specialist={r_bal['is_specialist']}  "
              f"variance={r_bal['variance']:.1f}  std={r_bal['std_dev']:.1f}")
        print(f"  {ok_spe}  Specialist model: is_specialist={r_spe['is_specialist']}  "
              f"variance={r_spe['variance']:.1f}  std={r_spe['std_dev']:.1f}")

    # Bonus: full leaderboard with specialist detection
    print("\n--- BONUS: Full Leaderboard with Flags ---")
    if (normalize_score(50, 0, 100) is not None
            and rank_models({"M": {"A": 1.0}}) is not None
            and detect_specialist({"A": 1.0}) is not None):

        full_models = {
            "GPT-4":          {"MMLU": 86.4, "HellaSwag": 95.3, "GSM8K": 92.0, "TruthfulQA": 59.0},
            "LLaMA-3-70B":    {"MMLU": 82.0, "HellaSwag": 88.0, "GSM8K": 93.0, "TruthfulQA": 62.8},
            "Mistral-7B":     {"MMLU": 62.5, "HellaSwag": 81.0, "GSM8K": 52.2, "TruthfulQA": 56.0},
            "MathGenius-7B":  {"MMLU": 45.0, "HellaSwag": 52.0, "GSM8K": 94.0, "TruthfulQA": 41.0},
        }
        ranked = rank_models(full_models)
        print(f"\n  {'Rank':>4}  {'Model':<18}  {'Avg':>6}  {'Flag'}")
        print("  " + "-" * 45)
        for rank, model, avg in ranked:
            analysis = detect_specialist(full_models[model])
            flag = "SPECIALIST - cherry-picker!" if analysis["is_specialist"] else "Balanced"
            print(f"  {rank:>4}  {model:<18}  {avg:>6.1f}  {flag}")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def normalize_score(raw, min_val, max_val):
#     if min_val == max_val:
#         return 0.0
#     result = (raw - min_val) / (max_val - min_val) * 100.0
#     return max(0.0, min(100.0, result))
#
# def rank_models(scores_dict):
#     averages = []
#     for model, bench_scores in scores_dict.items():
#         avg = sum(bench_scores.values()) / len(bench_scores)
#         averages.append((model, avg))
#     sorted_models = sorted(averages, key=lambda x: x[1], reverse=True)
#     return [(rank + 1, name, avg) for rank, (name, avg) in enumerate(sorted_models)]
#
# def compare_zero_few_shot(zero_shot_scores, few_shot_scores):
#     result = {}
#     for bench in zero_shot_scores:
#         z = zero_shot_scores[bench]
#         f = few_shot_scores[bench]
#         abs_gain = f - z
#         rel = (abs_gain / z) * 100.0 if z != 0 else 0.0
#         result[bench] = {"zero_shot": z, "few_shot": f, "abs_gain": abs_gain, "rel_gain_pct": rel}
#     return result
#
# def detect_specialist(model_scores, variance_threshold=200.0):
#     import math
#     scores = list(model_scores.values())
#     mean = sum(scores) / len(scores)
#     variance = sum((s - mean) ** 2 for s in scores) / len(scores)
#     std_dev = math.sqrt(variance)
#     return {"mean": mean, "variance": variance, "std_dev": std_dev,
#             "is_specialist": variance > variance_threshold}

"""
Exercise 03: HumanEval Pass@k Metric
Module 17: LLM Evaluation & Benchmarks

TASKS:
  1. Implement pass_at_k(n, c, k)              -- exact Pass@k formula
  2. Implement pass_at_1(per_attempt_prob)     -- simple Pass@1 estimate
  3. Implement required_attempts(target, p)    -- attempts needed for target pass rate
  4. Implement benchmark_pass_at_k(problems, k) -- evaluate a set of problems

Run:  python exercise_03_pass_at_k.py
Deps: none (pure Python)
"""

import math


# ─────────────────────────────────────────────────────────
# TASK 1: Pass@k Formula
# ─────────────────────────────────────────────────────────

def pass_at_k(n, c, k):
    """
    Compute the Pass@k metric for code generation evaluation.

    Used by HumanEval: given n attempts at a problem, where c pass,
    what is the probability that at least 1 of k randomly chosen
    attempts passes?

    Formula: Pass@k = 1 - C(n-c, k) / C(n, k)

    Numerically stable form (avoid computing large factorials):
      Pass@k = 1 - product of (n-c-i)/(n-i) for i in 0..k-1

    Special case: if n - c < k, return 1.0 (guaranteed success)

    Args:
        n: int, total samples generated (e.g., 10 or 20)
        c: int, samples that pass all unit tests (0 <= c <= n)
        k: int, number of attempts allowed in production

    Returns:
        float: Pass@k probability (0.0 to 1.0)

    Example:
        pass_at_k(10, 3, 1)  -> 0.300   (30% chance any 1 attempt passes)
        pass_at_k(10, 3, 5)  -> 0.833   (high chance in 5 attempts)
        pass_at_k(10, 10, 1) -> 1.0     (all pass, so any 1 is correct)
        pass_at_k(10, 0, 1)  -> 0.0     (none pass)

    HINT:
        if n - c < k:
            return 1.0
        result = 1.0
        for i in range(k):
            result *= (n - c - i) / (n - i)
        return 1.0 - result
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Simple Pass@1 Estimate
# ─────────────────────────────────────────────────────────

def pass_at_1(per_attempt_prob):
    """
    Estimate Pass@1 directly from per-attempt pass probability.

    When you know the per-attempt probability p (fraction of generated
    solutions that pass tests), Pass@1 is simply p.

    This is a simpler estimate than pass_at_k() when you just want
    to express "what fraction of single attempts succeed."

    Args:
        per_attempt_prob: float, probability a single attempt passes (0.0 to 1.0)

    Returns:
        float: Pass@1 = per_attempt_prob (same value, different framing)

    Example:
        pass_at_1(0.30) -> 0.30
        pass_at_1(0.87) -> 0.87  (GPT-4 on HumanEval, approximately)
        pass_at_1(0.00) -> 0.00  (model never generates correct code)

    HINT:
        return per_attempt_prob
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Required Attempts
# ─────────────────────────────────────────────────────────

def required_attempts(target_pass_rate, per_attempt_prob):
    """
    Compute the minimum number of attempts needed to achieve a target Pass@k rate.

    Using the formula: Pass@k = 1 - (1 - p)^k
    Solving for k:     k = log(1 - target) / log(1 - p)
    Round UP to nearest integer.

    Args:
        target_pass_rate: float, desired Pass@k probability (e.g., 0.95 for 95%)
        per_attempt_prob: float, probability a single attempt passes (0.0 to 1.0)

    Returns:
        int: minimum k (attempts) needed to reach target_pass_rate
             Returns float('inf') if per_attempt_prob == 0.0 (impossible)
             Returns 1 if per_attempt_prob >= target_pass_rate

    Example:
        required_attempts(0.95, 0.30)  -> 9   (need 9 tries at 30% per try for 95%)
        required_attempts(0.95, 0.87)  -> 2   (87% per try, just 2 needed)
        required_attempts(0.50, 0.50)  -> 1   (50% per try meets 50% target)
        required_attempts(0.99, 0.00)  -> inf  (impossible)

    HINT:
        if per_attempt_prob == 0.0:
            return float('inf')
        if per_attempt_prob >= target_pass_rate:
            return 1
        k = math.log(1 - target_pass_rate) / math.log(1 - per_attempt_prob)
        return math.ceil(k)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Evaluate a Set of Problems
# ─────────────────────────────────────────────────────────

def benchmark_pass_at_k(problems, k):
    """
    Compute the overall Pass@k for a benchmark set of problems.

    For each problem, given n total samples and c passing samples,
    compute Pass@k for that problem. Average across all problems.

    Args:
        problems: list of dicts, each with:
                  "name": str
                  "n":    int, total samples generated
                  "c":    int, samples that pass tests
        k:        int, how many attempts to allow

    Returns:
        float: average Pass@k across all problems (0.0 to 1.0)

    Example:
        problems = [
            {"name": "P1", "n": 10, "c": 3},
            {"name": "P2", "n": 10, "c": 8},
            {"name": "P3", "n": 10, "c": 0},
        ]
        benchmark_pass_at_k(problems, k=1)
        -> average of [0.300, 0.800, 0.000] = 0.367

    HINT:
        scores = [pass_at_k(p["n"], p["c"], k) for p in problems]
        return sum(scores) / len(scores)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 03: HumanEval Pass@k Metric")
    print("=" * 55)

    # Task 1: pass_at_k
    print("\n--- Task 1: pass_at_k ---")
    cases = [
        (10, 10, 1, 1.0),    # all pass → guaranteed
        (10, 0,  1, 0.0),    # none pass → impossible
        (10, 3,  1, 0.3),    # 30% per attempt
        (10, 3, 10, 1.0),    # n-c < k → guaranteed
    ]
    for n, c, k, expected in cases:
        result = pass_at_k(n, c, k)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if abs(result - expected) < 0.01 else "FAIL"
        print(f"  {status}  pass_at_k(n={n}, c={c}, k={k}) = {result:.3f}  (expected {expected:.3f})")

    # Task 2: pass_at_1
    print("\n--- Task 2: pass_at_1 ---")
    p1_cases = [(0.3, 0.3), (0.87, 0.87), (0.0, 0.0), (1.0, 1.0)]
    for p, expected in p1_cases:
        result = pass_at_1(p)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if abs(result - expected) < 0.001 else "FAIL"
        print(f"  {status}  pass_at_1({p}) = {result:.3f}  (expected {expected:.3f})")

    # Task 3: required_attempts
    print("\n--- Task 3: required_attempts ---")
    ra_cases = [
        (0.95, 0.30, 9),
        (0.50, 0.50, 1),
        (0.95, 0.87, 2),
        (0.99, 0.00, float('inf')),
    ]
    for target, p, expected in ra_cases:
        result = required_attempts(target, p)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        if expected == float('inf'):
            status = "PASS" if result == float('inf') else "FAIL"
        else:
            status = "PASS" if result == expected else "FAIL"
        print(f"  {status}  required_attempts(target={target}, p={p}) = {result}  (expected {expected})")

    # Task 4: benchmark_pass_at_k
    print("\n--- Task 4: benchmark_pass_at_k ---")
    problems = [
        {"name": "has_close_elements", "n": 10, "c": 3},
        {"name": "separate_paren_groups", "n": 10, "c": 8},
        {"name": "truncate_number", "n": 10, "c": 0},
        {"name": "below_zero", "n": 10, "c": 5},
    ]
    if pass_at_k(10, 3, 1) is not None:
        for k_val in [1, 5, 10]:
            result = benchmark_pass_at_k(problems, k=k_val)
            if result is None:
                print("  NOT IMPLEMENTED YET")
                break
            print(f"  Pass@{k_val:>2}: benchmark score = {result:.3f} ({result*100:.1f}%)")

    # Bonus: full Pass@k table
    print("\n--- BONUS: Pass@k Table (n=20 samples) ---")
    if pass_at_k(10, 5, 1) is not None and required_attempts(0.95, 0.3) is not None:
        print(f"\n  {'per-attempt p':>15}  {'Pass@1':>8}  {'Pass@5':>8}  {'Pass@10':>9}  {'Needed for 95%':>15}")
        print("  " + "-" * 62)
        n = 20
        for p in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            c = round(p * n)
            p1  = pass_at_k(n, c, 1)
            p5  = pass_at_k(n, c, 5)
            p10 = pass_at_k(n, c, 10)
            req = required_attempts(0.95, p)
            req_str = str(req) if req != float('inf') else "impossible"
            print(f"  {p:>15.0%}  {p1:>8.3f}  {p5:>8.3f}  {p10:>9.3f}  {req_str:>15}")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def pass_at_k(n, c, k):
#     if n - c < k:
#         return 1.0
#     result = 1.0
#     for i in range(k):
#         result *= (n - c - i) / (n - i)
#     return 1.0 - result
#
# def pass_at_1(per_attempt_prob):
#     return per_attempt_prob
#
# def required_attempts(target_pass_rate, per_attempt_prob):
#     if per_attempt_prob == 0.0:
#         return float('inf')
#     if per_attempt_prob >= target_pass_rate:
#         return 1
#     k = math.log(1 - target_pass_rate) / math.log(1 - per_attempt_prob)
#     return math.ceil(k)
#
# def benchmark_pass_at_k(problems, k):
#     scores = [pass_at_k(p["n"], p["c"], k) for p in problems]
#     return sum(scores) / len(scores)

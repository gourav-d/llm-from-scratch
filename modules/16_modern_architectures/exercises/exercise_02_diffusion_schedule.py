"""
Exercise 02: Diffusion Noise Schedule
Module 16: Modern LLM Architectures

TASKS:
  1. Implement linear_schedule()  -- masking probability grows linearly
  2. Implement cosine_schedule()  -- smoother noise growth (cosine curve)
  3. Implement apply_mask()       -- apply masking to a token list at timestep t
  4. Implement count_masked()     -- count masked tokens in a sentence

Run:  python exercise_02_diffusion_schedule.py
Deps: none (pure Python)
"""

import math
import random

MASK = "[M]"


# ─────────────────────────────────────────────────────────
# TASK 1: Linear Noise Schedule
# ─────────────────────────────────────────────────────────

def linear_schedule(t, T):
    """
    Compute masking probability at timestep t using a LINEAR schedule.

    Masking probability grows linearly from 0 (at t=0) to 1 (at t=T).

    Args:
        t: current timestep (integer, 0 <= t <= T)
        T: total number of timesteps

    Returns:
        float: masking probability in range [0.0, 1.0]

    Example:
        linear_schedule(0, 10) -> 0.0
        linear_schedule(5, 10) -> 0.5
        linear_schedule(10, 10) -> 1.0

    HINT: p = t / T
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Cosine Noise Schedule
# ─────────────────────────────────────────────────────────

def cosine_schedule(t, T):
    """
    Compute masking probability at timestep t using a COSINE schedule.

    Cosine schedule: starts slow, accelerates in the middle, slows at the end.
    Formula: p = 1 - cos((t/T) * pi/2)^2

    This is smoother than linear and tends to produce better quality in practice.

    Args:
        t: current timestep (integer, 0 <= t <= T)
        T: total number of timesteps

    Returns:
        float: masking probability in range [0.0, 1.0]

    Example:
        cosine_schedule(0, 10)  -> 0.0   (t=0: no noise)
        cosine_schedule(10, 10) -> 1.0   (t=T: full noise)
        cosine_schedule(5, 10)  -> ~0.5  (midpoint: about 50%)

    HINT:
        alpha = cos((t / T) * math.pi / 2) ** 2
        p = 1.0 - alpha
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Apply Masking
# ─────────────────────────────────────────────────────────

def apply_mask(tokens, t, T, schedule_fn, seed=42):
    """
    Apply the forward diffusion process to a list of tokens at timestep t.
    Each token is independently masked with probability p = schedule_fn(t, T).

    Args:
        tokens:      list of string tokens e.g. ["The", "cat", "sat"]
        t:           current timestep
        T:           total timesteps
        schedule_fn: a function(t, T) -> float that returns masking probability
        seed:        random seed for reproducibility

    Returns:
        list of tokens where some are replaced with MASK = "[M]"

    Example:
        apply_mask(["The", "cat", "sat"], t=0, T=10, schedule_fn=linear_schedule)
        -> ["The", "cat", "sat"]   (t=0: no masking)

        apply_mask(["The", "cat", "sat"], t=10, T=10, schedule_fn=linear_schedule)
        -> ["[M]", "[M]", "[M]"]   (t=T: all masked)

    HINT:
        1. Compute p_mask = schedule_fn(t, T)
        2. For each token, generate a random number
        3. If random number < p_mask: replace token with MASK
        4. Otherwise: keep original token
        Use random.Random(seed + t) for reproducibility.
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Count Masked Tokens
# ─────────────────────────────────────────────────────────

def count_masked(tokens):
    """
    Count how many tokens in the list are masked (equal to MASK = "[M]").

    Args:
        tokens: list of string tokens, some may be "[M]"

    Returns:
        int: number of masked tokens

    Example:
        count_masked(["The", "[M]", "sat", "[M]", "the", "mat"]) -> 2
        count_masked(["The", "cat", "sat"]) -> 0
        count_masked(["[M]", "[M]", "[M]"]) -> 3

    HINT: count occurrences of MASK in the list.
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 02: Diffusion Noise Schedule")
    print("=" * 55)

    # Task 1: linear_schedule
    print("\n--- Task 1: linear_schedule ---")
    cases = [(0, 10, 0.0), (5, 10, 0.5), (10, 10, 1.0), (3, 12, 0.25)]
    for t, T, expected in cases:
        result = linear_schedule(t, T)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if abs(result - expected) < 0.001 else "FAIL"
        print(f"  {status}  linear_schedule(t={t}, T={T}) = {result:.3f}  (expected {expected:.3f})")

    # Task 2: cosine_schedule
    print("\n--- Task 2: cosine_schedule ---")
    cases = [(0, 10, 0.0), (10, 10, 1.0)]
    for t, T, expected in cases:
        result = cosine_schedule(t, T)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if abs(result - expected) < 0.01 else "FAIL"
        print(f"  {status}  cosine_schedule(t={t}, T={T}) = {result:.4f}  (expected {expected:.4f})")
    if cosine_schedule(5, 10) is not None:
        mid = cosine_schedule(5, 10)
        print(f"  INFO  cosine_schedule(t=5, T=10) = {mid:.4f}  (should be around 0.5)")

    # Task 3: apply_mask
    print("\n--- Task 3: apply_mask ---")
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    if linear_schedule(0, 10) is not None:
        # t=0: no masking
        result_t0 = apply_mask(sentence, 0, 10, linear_schedule)
        if result_t0 is None:
            print("  NOT IMPLEMENTED YET")
        else:
            status = "PASS" if result_t0 == sentence else "FAIL"
            print(f"  {status}  t=0 (p=0.0): {result_t0}  (should be unchanged)")

            # t=T: full masking
            result_tT = apply_mask(sentence, 10, 10, linear_schedule)
            all_masked = all(tok == MASK for tok in result_tT)
            status = "PASS" if all_masked else "FAIL"
            print(f"  {status}  t=T (p=1.0): {result_tT}  (should be all [M])")

            # t=5: about half masked
            result_t5 = apply_mask(sentence, 5, 10, linear_schedule)
            n_masked = count_masked(result_t5) if count_masked(sentence) is not None else "?"
            print(f"  INFO  t=5 (p=0.5): {result_t5}  ({n_masked}/{len(sentence)} masked)")

    # Task 4: count_masked
    print("\n--- Task 4: count_masked ---")
    cases = [
        (["The", MASK, "sat", MASK, "the", "mat"], 2),
        (["The", "cat", "sat"], 0),
        ([MASK, MASK, MASK], 3),
    ]
    for tokens, expected in cases:
        result = count_masked(tokens)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}  count_masked({tokens}) = {result}  (expected {expected})")

    # Bonus: schedule comparison table
    print("\n--- BONUS: Schedule Comparison ---")
    if linear_schedule(5, 10) is not None and cosine_schedule(5, 10) is not None:
        T = 10
        print(f"\n  {'t':>4}  {'Linear p':>10}  {'Cosine p':>10}  {'Difference':>12}")
        print("  " + "-" * 42)
        for t in range(T + 1):
            lin = linear_schedule(t, T)
            cos = cosine_schedule(t, T)
            diff = cos - lin
            sign = "+" if diff >= 0 else ""
            print(f"  {t:>4}  {lin:>10.3f}  {cos:>10.3f}  {sign}{diff:>11.3f}")
        print("\n  Cosine starts slower and ends slower vs linear.")
        print("  This gives the model more time to learn easy cases (low t)")
        print("  and hard cases (high t) without rushing through them.")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def linear_schedule(t, T):
#     return t / T
#
# def cosine_schedule(t, T):
#     alpha = math.cos((t / T) * math.pi / 2) ** 2
#     return 1.0 - alpha
#
# def apply_mask(tokens, t, T, schedule_fn, seed=42):
#     rng = random.Random(seed + t)
#     p_mask = schedule_fn(t, T)
#     return [MASK if rng.random() < p_mask else tok for tok in tokens]
#
# def count_masked(tokens):
#     return sum(1 for tok in tokens if tok == MASK)

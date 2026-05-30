"""
Exercise 01: Chinchilla Scaling Laws
Module 15: Advanced LLM Training

TASKS:
  1. Implement compute_flops() — estimate total FLOPs for a training run
  2. Implement chinchilla_optimal_tokens() — compute tokens for a given model size
  3. Implement is_compute_optimal() — check if a training run is optimal
  4. Analyze 3 real models and classify them as under/over/optimally trained

Run:  python exercise_01_chinchilla.py
Deps: none (pure Python)
"""

import math


# ─────────────────────────────────────────────────────────
# TASK 1: Implement compute_flops
# ─────────────────────────────────────────────────────────

def compute_flops(num_params: int, num_tokens: int) -> float:
    """
    Estimate total training FLOPs using C = 6 × N × D.

    Args:
        num_params:  number of model parameters (N)
        num_tokens:  number of training tokens (D)

    Returns:
        Total FLOPs as a float

    HINT: multiply 6 × num_params × num_tokens
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Implement chinchilla_optimal_tokens
# ─────────────────────────────────────────────────────────

def chinchilla_optimal_tokens(num_params: int) -> int:
    """
    Given a model size (number of parameters), return the
    Chinchilla-optimal number of training tokens using the Rule of 20.

    Args:
        num_params: number of model parameters

    Returns:
        Optimal number of training tokens

    HINT: optimal_tokens = 20 × num_params
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Implement is_compute_optimal
# ─────────────────────────────────────────────────────────

def is_compute_optimal(num_params: int, num_tokens: int,
                        tolerance: float = 0.5) -> str:
    """
    Classify a training run as 'under-trained', 'optimal', or 'over-trained'.

    The ratio is: actual_tokens / optimal_tokens
    - ratio < (1 - tolerance): under-trained (too few tokens for model size)
    - ratio > (1 + tolerance): over-trained  (more tokens than Chinchilla recommends)
    - otherwise: near-optimal

    Args:
        num_params:  number of model parameters
        num_tokens:  actual number of training tokens
        tolerance:   fraction above/below 1.0 to consider "optimal" range

    Returns:
        One of: "under-trained", "near-optimal", "over-trained"

    Example:
        is_compute_optimal(7e9, 140e9) → "near-optimal"
        is_compute_optimal(175e9, 300e9) → "under-trained"
        is_compute_optimal(7e9, 15e12) → "over-trained"

    HINT: compute optimal = 20 × num_params, then compare ratio
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Budget Calculator
# ─────────────────────────────────────────────────────────

def compute_budget_from_hardware(num_gpus: int, gpu_tflops: float,
                                  days: int, mfu: float = 0.35) -> float:
    """
    Calculate total available FLOPs from hardware.

    Args:
        num_gpus:   number of GPUs
        gpu_tflops: peak TFLOP/s per GPU (e.g., 312 for A100)
        days:       training duration in days
        mfu:        model FLOPs utilization (fraction of peak, typically 0.35)

    Returns:
        Total useful FLOPs

    HINT:
        seconds = days × 24 × 3600
        peak_flops = num_gpus × (gpu_tflops × 10^12) × seconds
        return peak_flops × mfu
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 01: Chinchilla Scaling Laws")
    print("=" * 55)

    # Test 1: compute_flops
    print("\n--- Test 1: compute_flops ---")
    result = compute_flops(7_000_000_000, 140_000_000_000)
    expected = 6 * 7e9 * 140e9
    if result is None:
        print("  NOT IMPLEMENTED YET")
    elif abs(result - expected) / expected < 0.01:
        print(f"  PASS  flops = {result:.2e}")
    else:
        print(f"  FAIL  got {result:.2e}, expected {expected:.2e}")

    # Test 2: chinchilla_optimal_tokens
    print("\n--- Test 2: chinchilla_optimal_tokens ---")
    result = chinchilla_optimal_tokens(7_000_000_000)
    expected = 140_000_000_000
    if result is None:
        print("  NOT IMPLEMENTED YET")
    elif abs(result - expected) / expected < 0.01:
        print(f"  PASS  optimal tokens = {result:,.0f}")
    else:
        print(f"  FAIL  got {result:,.0f}, expected {expected:,.0f}")

    # Test 3: is_compute_optimal
    print("\n--- Test 3: is_compute_optimal ---")
    cases = [
        (7e9,   140e9,   "near-optimal",  "7B model, 140B tokens"),
        (175e9, 300e9,   "under-trained", "GPT-3: 175B params, 300B tokens"),
        (7e9,   15e12,   "over-trained",  "LLaMA 3: 7B params, 15T tokens"),
        (70e9,  1.4e12,  "near-optimal",  "Chinchilla: 70B params, 1.4T tokens"),
    ]

    all_pass = True
    for params, tokens, expected, desc in cases:
        result = is_compute_optimal(int(params), int(tokens))
        if result is None:
            print(f"  NOT IMPLEMENTED YET")
            all_pass = False
            break
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status}  {desc}")
        if status == "FAIL":
            print(f"         got '{result}', expected '{expected}'")

    # Test 4: Budget calculator
    print("\n--- Test 4: compute_budget_from_hardware ---")
    result = compute_budget_from_hardware(
        num_gpus=8, gpu_tflops=312, days=30, mfu=0.35
    )
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        seconds = 30 * 24 * 3600
        expected = 8 * 312e12 * seconds * 0.35
        if abs(result - expected) / expected < 0.01:
            print(f"  PASS  budget = {result:.2e} FLOPs")
            # Bonus: show optimal model for this budget
            n_opt = int(math.sqrt(result / 12))
            d_opt = 20 * n_opt
            print(f"  Optimal model: {n_opt/1e9:.1f}B params, {d_opt/1e9:.0f}B tokens")
        else:
            print(f"  FAIL  got {result:.2e}, expected {expected:.2e}")

    # BONUS: Real Model Analysis
    print("\n--- BONUS: Real Model Analysis ---")
    models = [
        ("GPT-3",      175e9,  300e9),
        ("Chinchilla",  70e9,  1.4e12),
        ("LLaMA 3 8B",  8e9,  15e12),
        ("Mistral 7B",  7e9,   1e12),
    ]

    result_fn = is_compute_optimal
    if result_fn(7_000_000_000, 140_000_000_000) is not None:
        print(f"\n{'Model':<16} {'Params':>8} {'Tokens':>8} {'Tok/Param':>10} {'Status'}")
        print("-" * 60)
        for name, params, tokens in models:
            status = is_compute_optimal(int(params), int(tokens))
            ratio = tokens / params
            p_str = f"{params/1e9:.0f}B"
            t_str = f"{tokens/1e12:.1f}T" if tokens >= 1e12 else f"{tokens/1e9:.0f}B"
            print(f"{name:<16} {p_str:>8} {t_str:>8} {ratio:>10.0f} {status}")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def compute_flops(num_params, num_tokens):
#     return 6 * num_params * num_tokens
#
# def chinchilla_optimal_tokens(num_params):
#     return int(20 * num_params)
#
# def is_compute_optimal(num_params, num_tokens, tolerance=0.5):
#     optimal = 20 * num_params
#     ratio = num_tokens / optimal
#     if ratio < (1 - tolerance):
#         return "under-trained"
#     elif ratio > (1 + tolerance):
#         return "over-trained"
#     else:
#         return "near-optimal"
#
# def compute_budget_from_hardware(num_gpus, gpu_tflops, days, mfu=0.35):
#     seconds = days * 24 * 3600
#     peak_flops = num_gpus * gpu_tflops * 1e12 * seconds
#     return peak_flops * mfu

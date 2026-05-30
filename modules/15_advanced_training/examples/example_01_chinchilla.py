"""
Example 01: Chinchilla Scaling Laws Calculator
Module 15: Advanced LLM Training

Computes compute-optimal model size and token count for a given training budget.
Also compares real LLMs against the Chinchilla-optimal targets.

Run:  python example_01_chinchilla.py
Deps: none (pure Python)
"""

import math


# ─────────────────────────────────────────────────────────
# CORE FORMULAS
# ─────────────────────────────────────────────────────────

def compute_flops(num_params: int, num_tokens: int) -> float:
    """
    Estimate total training FLOPs using the Chinchilla formula.

    C ≈ 6 × N × D
    where N = parameters, D = training tokens
    Factor of 6: multiply(1) + add(1) per param per token = 2, × 3 for forward+backward.
    """
    return 6 * num_params * num_tokens


def chinchilla_optimal(compute_budget_flops: float) -> tuple[int, int]:
    """
    Given a compute budget (FLOPs), return the Chinchilla-optimal
    (model_size_params, num_training_tokens).

    From the Chinchilla paper (Hoffmann et al. 2022):
        N_optimal ≈ sqrt(C / 12)
        D_optimal ≈ sqrt(C * 3)
    which satisfies D = 20 × N approximately.
    """
    n_optimal = math.sqrt(compute_budget_flops / 12)
    d_optimal = math.sqrt(compute_budget_flops * 3)
    return int(n_optimal), int(d_optimal)


def tokens_per_param(num_params: int, num_tokens: int) -> float:
    """Compute token-to-parameter ratio."""
    return num_tokens / num_params


# ─────────────────────────────────────────────────────────
# HARDWARE FLOPs ESTIMATION
# ─────────────────────────────────────────────────────────

def estimate_flops_from_hardware(
    num_gpus: int,
    gpu_flops_per_second: float,
    training_days: int,
    mfu: float = 0.35   # model FLOPs utilization, typically 30-40%
) -> float:
    """
    Estimate available FLOPs from hardware.

    mfu = actual throughput / peak throughput. Real training achieves 30-40%
    of theoretical peak due to communication, data loading, etc.
    """
    seconds = training_days * 24 * 3600
    theoretical_flops = num_gpus * gpu_flops_per_second * seconds
    return theoretical_flops * mfu


# ─────────────────────────────────────────────────────────
# PRETTY PRINTING HELPERS
# ─────────────────────────────────────────────────────────

def human_readable(n: float) -> str:
    """Format large numbers as B (billion), T (trillion), etc."""
    if n >= 1e15:
        return f"{n/1e15:.1f}P"
    elif n >= 1e12:
        return f"{n/1e12:.1f}T"
    elif n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    else:
        return str(int(n))


def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


# ─────────────────────────────────────────────────────────
# DEMO 1: Compute budget from hardware
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: Compute Budget from Hardware")

# Scenario A: single A100 GPU for 30 days
a100_flops = 312e12      # A100 theoretical peak: 312 TFLOP/s (bf16)
budget_a = estimate_flops_from_hardware(
    num_gpus=1,
    gpu_flops_per_second=a100_flops,
    training_days=30,
    mfu=0.35
)
n_opt_a, d_opt_a = chinchilla_optimal(budget_a)
print(f"\nScenario A: 1 × A100, 30 days")
print(f"  Total FLOPs budget:   {human_readable(budget_a)} FLOPs")
print(f"  Optimal model size:   {human_readable(n_opt_a)} params")
print(f"  Optimal token count:  {human_readable(d_opt_a)} tokens")
print(f"  Tokens per param:     {tokens_per_param(n_opt_a, d_opt_a):.1f}")

# Scenario B: 8 A100s for 30 days (realistic small-team run)
budget_b = estimate_flops_from_hardware(
    num_gpus=8,
    gpu_flops_per_second=a100_flops,
    training_days=30,
    mfu=0.35
)
n_opt_b, d_opt_b = chinchilla_optimal(budget_b)
print(f"\nScenario B: 8 × A100, 30 days")
print(f"  Total FLOPs budget:   {human_readable(budget_b)} FLOPs")
print(f"  Optimal model size:   {human_readable(n_opt_b)} params")
print(f"  Optimal token count:  {human_readable(d_opt_b)} tokens")
print(f"  Tokens per param:     {tokens_per_param(n_opt_b, d_opt_b):.1f}")

# Scenario C: 1024 H100s for 90 days (large-scale run)
h100_flops = 989e12      # H100 theoretical peak: 989 TFLOP/s (bf16)
budget_c = estimate_flops_from_hardware(
    num_gpus=1024,
    gpu_flops_per_second=h100_flops,
    training_days=90,
    mfu=0.40
)
n_opt_c, d_opt_c = chinchilla_optimal(budget_c)
print(f"\nScenario C: 1024 × H100, 90 days")
print(f"  Total FLOPs budget:   {human_readable(budget_c)} FLOPs")
print(f"  Optimal model size:   {human_readable(n_opt_c)} params")
print(f"  Optimal token count:  {human_readable(d_opt_c)} tokens")
print(f"  Tokens per param:     {tokens_per_param(n_opt_c, d_opt_c):.1f}")


# ─────────────────────────────────────────────────────────
# DEMO 2: Real LLMs vs Chinchilla Optimal
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: Real LLMs vs Chinchilla-Optimal Targets")

real_models = [
    # (name, params, training_tokens)
    ("GPT-3",         175e9,   300e9),
    ("Gopher",        280e9,   300e9),
    ("Chinchilla",     70e9,   1.4e12),
    ("LLaMA 1 7B",      7e9,   1.0e12),
    ("LLaMA 1 65B",    65e9,   1.4e12),
    ("LLaMA 3 8B",      8e9,  15.0e12),
    ("Mistral 7B",      7e9,   1.0e12),
    ("Phi-2",           2.7e9, 1.4e12),
]

header = f"{'Model':<18} {'Params':>8} {'Tokens':>8} {'Tok/Param':>10} {'Opt Tok':>10} {'Status':<18}"
print(f"\n{header}")
print("-" * 78)

for name, n, d in real_models:
    # Compute actual FLOPs used
    actual_flops = compute_flops(n, d)
    # What would Chinchilla-optimal be for that FLOPs budget?
    n_opt, d_opt = chinchilla_optimal(actual_flops)
    ratio = d / n

    if ratio < 15:
        status = "under-trained"
    elif ratio < 25:
        status = "near-optimal"
    else:
        status = "over-trained (good for inference)"

    print(
        f"{name:<18} {human_readable(n):>8} {human_readable(d):>8} "
        f"{ratio:>10.0f} {human_readable(d_opt):>10} {status:<18}"
    )


# ─────────────────────────────────────────────────────────
# DEMO 3: Rule of 20 — Quick Reference Table
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: Rule of 20 — Compute-Optimal Tokens for Common Model Sizes")

print(f"\n{'Model Size':>14} {'Chinchilla-Optimal Tokens':>26}")
print("-" * 42)

param_counts = [
    ("125M",   125e6),
    ("350M",   350e6),
    ("760M",   760e6),
    ("1.3B",   1.3e9),
    ("3B",     3.0e9),
    ("7B",     7.0e9),
    ("13B",   13.0e9),
    ("30B",   30.0e9),
    ("65B",   65.0e9),
    ("175B", 175.0e9),
]

for label, n in param_counts:
    d_opt = 20 * n          # Rule of 20 approximation
    print(f"{label:>14} {human_readable(d_opt):>26}")


# ─────────────────────────────────────────────────────────
# DEMO 4: Cost to Train
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: Estimated Cloud Cost to Train")

print("\nAssumptions: A100 80GB, ~$2/hr (spot/reserved pricing)")
print(f"\n{'Model':<14} {'Params':>8} {'Tokens':>8} {'GPU-Hours':>12} {'Cost ($)':>12}")
print("-" * 60)

a100_useful_flops = a100_flops * 0.35   # 35% MFU

for label, n in param_counts:
    d = 20 * n
    total_flops = compute_flops(n, d)
    gpu_hours = total_flops / (a100_useful_flops * 3600)
    cost_usd = gpu_hours * 2.0           # $2/hr per A100
    print(
        f"{label:<14} {human_readable(n):>8} {human_readable(d):>8} "
        f"{gpu_hours:>12,.0f} ${cost_usd:>11,.0f}"
    )

print("""
Note: These are compute costs only.
Real training also needs:
  - Data collection and preprocessing
  - Engineering time
  - Storage costs
  - Failed run overhead (crashes, restarts)
Typical full training run cost is 2-5x the compute-only estimate.
""")

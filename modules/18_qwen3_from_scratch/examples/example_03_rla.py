"""
Example 03: Recurrent Linear Attention (RLA)
Module 18: Qwen3.5 LLM from Scratch

Run:  python example_03_rla.py
Deps: none (pure Python)
"""

import math

print("=" * 60)
print("  Example 03: Recurrent Linear Attention (RLA)")
print("=" * 60)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def outer(k, v):
    """Outer product: k ⊗ v = matrix[i][j] = k[i] * v[j]."""
    return [[ki * vj for vj in v] for ki in k]

def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_vec_mul(A, v):
    """Matrix × vector: result[i] = dot(A[i], v)."""
    return [dot(row, v) for row in A]


# ─────────────────────────────────────────────────────────
# DEMO 1: The Phi Kernel — Replacing Softmax
# ─────────────────────────────────────────────────────────

print("\n--- DEMO 1: Phi Kernel (ELU + 1) ---")
print()
print("  Standard attention uses softmax(Q·K/sqrt(d)).")
print("  Linear attention replaces it with phi(q) · phi(k).")
print("  Requirement: phi must be non-negative (values >= 0).")
print()

def elu(x, alpha=1.0):
    """Exponential Linear Unit: x if x > 0 else alpha*(exp(x) - 1)."""
    return x if x > 0 else alpha * (math.exp(x) - 1)

def phi(x_vec):
    """Kernel function: phi(x) = ELU(x) + 1  (ensures non-negative output)."""
    return [elu(xi) + 1.0 for xi in x_vec]

# Show phi on various inputs
print(f"  {'x':>8}  {'ELU(x)':>10}  {'phi(x)=ELU+1':>14}  {'Non-negative?':>14}")
print("  " + "-" * 52)
for x in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
    e = elu(x)
    p = e + 1.0
    ok = "YES" if p >= 0 else "NO!"
    print(f"  {x:>8.1f}  {e:>10.4f}  {p:>14.4f}  {ok:>14}")

print()
print("  phi(x) >= 0 always — ensures hidden state accumulation stays stable.")
print("  Negative values would cause cancellation in the outer product sum.")


# ─────────────────────────────────────────────────────────
# DEMO 2: Recurrent Hidden State Accumulation
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 2: Recurrent Hidden State S ---")
print()
print("  S_t = S_{t-1} + phi(K_t) ⊗ V_t")
print("  S starts at zero. Each new token updates S by adding phi(K)⊗V.")
print("  S size = [d_k, d_v] — FIXED regardless of sequence length.")
print()

# Tiny example: d_k=2, d_v=2, seq_len=5
d_k, d_v = 2, 2

import random
rng = random.Random(7)
rand = lambda: round(rng.uniform(-1, 1), 3)

tokens = [
    {"id": "The",   "K": [rand(), rand()], "V": [rand(), rand()]},
    {"id": "cat",   "K": [rand(), rand()], "V": [rand(), rand()]},
    {"id": "sat",   "K": [rand(), rand()], "V": [rand(), rand()]},
    {"id": "on",    "K": [rand(), rand()], "V": [rand(), rand()]},
    {"id": "the",   "K": [rand(), rand()], "V": [rand(), rand()]},
]

# Run recurrent update
S = [[0.0] * d_v for _ in range(d_k)]   # hidden state
z = [0.0] * d_k                          # normalizer

print(f"  {'Step':>5}  {'Token':>6}  {'phi(K)':>20}  {'S[0][0]':>10}  {'S[0][1]':>10}")
print("  " + "-" * 58)

for t, tok in enumerate(tokens):
    k_phi = phi(tok["K"])
    v     = tok["V"]

    outer_kv = outer(k_phi, v)
    S = mat_add(S, outer_kv)
    z = [z[i] + k_phi[i] for i in range(d_k)]

    print(f"  {t+1:>5}  {tok['id']:>6}  ({k_phi[0]:>8.4f},{k_phi[1]:>8.4f})  "
          f"{S[0][0]:>10.4f}  {S[0][1]:>10.4f}")

print()
print(f"  Final S shape: [{d_k}×{d_v}] = {d_k*d_v} values.")
print(f"  Standard KV cache for 5 tokens would need: 5×{d_k} + 5×{d_v} = {5*(d_k+d_v)} values.")
print(f"  At 10,000 tokens: S still {d_k*d_v} values vs {10000*(d_k+d_v)} for KV cache.")


# ─────────────────────────────────────────────────────────
# DEMO 3: Full RLA Forward Step — Query the Hidden State
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 3: Full RLA Step (Query + Output) ---")
print()
print("  Output_t = (phi(Q_t) @ S_t) / (phi(Q_t) @ z_t + epsilon)")
print("  The hidden state S contains compressed memory of all past tokens.")
print()

def rla_step(q_vec, S, z, eps=1e-6):
    """
    One RLA inference step.
    q_vec: query vector at current position
    S:     hidden state [d_k, d_v]
    z:     normalizer [d_k]
    Returns: output vector [d_v]
    """
    q_phi  = phi(q_vec)
    numerator   = mat_vec_mul(S, q_phi)   # S is [d_k, d_v], q_phi is [d_k] → [d_v]
    # Actually: output = q_phi @ S, so it's a row-vector × matrix
    # output[j] = sum_i q_phi[i] * S[i][j]
    output = [sum(q_phi[i] * S[i][j] for i in range(len(q_phi))) for j in range(len(S[0]))]
    denominator = dot(q_phi, z) + eps
    return [o / denominator for o in output]

# Run full recurrent inference with the S and z from DEMO 2
# Now query each position
print("  Running full RLA (recurrent mode):")
print()
print(f"  {'Token':>6}  {'phi(Q)':>20}  {'Output (2D)':>22}  {'Note'}")
print("  " + "-" * 68)

# Re-run to get sequence of outputs (recurrent)
S2 = [[0.0] * d_v for _ in range(d_k)]
z2 = [0.0] * d_k

for t, tok in enumerate(tokens):
    # Add Q vector to each token (in practice Q comes from a different projection)
    q_vec  = [tok["K"][0] + 0.1, tok["K"][1] - 0.1]  # simulate separate Q projection
    k_phi  = phi(tok["K"])
    v      = tok["V"]

    # Update hidden state FIRST, then query (causal: S includes current token)
    outer_kv = outer(k_phi, v)
    S2 = mat_add(S2, outer_kv)
    z2 = [z2[i] + k_phi[i] for i in range(d_k)]

    output = rla_step(q_vec, S2, z2)
    q_phi_vals = phi(q_vec)
    note = "← S already contains past tokens" if t == 2 else ""
    print(f"  {tok['id']:>6}  ({q_phi_vals[0]:>8.4f},{q_phi_vals[1]:>8.4f})  "
          f"({output[0]:>9.4f}, {output[1]:>9.4f})  {note}")

print()
print("  COMPLEXITY COMPARISON:")
print(f"  {'Method':>25}  {'Memory':>18}  {'Compute/token':>16}")
print("  " + "-" * 64)
N = 10000
d = 128
rows = [
    ("Standard attention",     f"O(N)  ~ {N*d} vals",      f"O(N*d)  ~ {N*d}"),
    ("Linear attention (RLA)", f"O(d²) ~ {d*d} vals (fixed)", f"O(d²)  ~ {d*d}"),
]
for name, mem, comp in rows:
    print(f"  {name:>25}  {mem:>18}  {comp:>16}")

print()
print("  At N=10,000 tokens:")
print(f"    Standard: need to store {N} KV vectors of size {d} = {N*d:,} values")
print(f"    RLA:      hidden state S is always {d}×{d} = {d*d:,} values (fixed!)")

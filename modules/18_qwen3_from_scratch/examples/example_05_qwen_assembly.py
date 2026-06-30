"""
Example 05: Full Qwen3.5 Decoder Assembly
Module 18: Qwen3.5 LLM from Scratch

Run:  python example_05_qwen_assembly.py
Deps: none (pure Python)
"""

import math

print("=" * 60)
print("  Example 05: Qwen3.5 Decoder Assembly")
print("=" * 60)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def mat_mul_vec(W, x):
    """Matrix × vector: W is [out_dim, in_dim], x is [in_dim] → [out_dim]."""
    return [dot(row, x) for row in W]

def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def vec_mul(a, b):
    return [a[i] * b[i] for i in range(len(a))]

import random
rng = random.Random(99)
rand_vec = lambda n: [rng.uniform(-0.5, 0.5) for _ in range(n)]
rand_mat = lambda r, c: [[rng.uniform(-0.5, 0.5) for _ in range(c)] for _ in range(r)]


# ─────────────────────────────────────────────────────────
# DEMO 1: RMSNorm vs LayerNorm
# ─────────────────────────────────────────────────────────

print("\n--- DEMO 1: RMSNorm (simpler than LayerNorm) ---")
print()
print("  LayerNorm: (x - mean) / sqrt(variance + eps) * gamma + beta")
print("  RMSNorm:   x / sqrt(mean(x^2) + eps) * gamma")
print("  Difference: RMSNorm skips mean subtraction (no beta bias term).")
print()

def layer_norm(x, gamma, beta, eps=1e-5):
    """Standard LayerNorm (as used in original GPT/BERT)."""
    mean = sum(x) / len(x)
    variance = sum((xi - mean) ** 2 for xi in x) / len(x)
    x_norm = [(xi - mean) / math.sqrt(variance + eps) for xi in x]
    return [gamma[i] * x_norm[i] + beta[i] for i in range(len(x))]

def rms_norm(x, gamma, eps=1e-5):
    """RMSNorm (as used in Qwen3.5, LLaMA, Mistral)."""
    rms = math.sqrt(sum(xi ** 2 for xi in x) / len(x) + eps)
    return [gamma[i] * (x[i] / rms) for i in range(len(x))]

d = 6
x = [1.5, -0.8, 2.3, -1.1, 0.4, 1.9]
gamma = [1.0] * d   # all ones = no scaling
beta  = [0.0] * d   # all zeros = no shift

ln_out  = layer_norm(x, gamma, beta)
rms_out = rms_norm(x, gamma)

print(f"  Input x:           {[round(xi,3) for xi in x]}")
print(f"  LayerNorm output:  {[round(xi,4) for xi in ln_out]}")
print(f"  RMSNorm output:    {[round(xi,4) for xi in rms_out]}")
print()

ln_mean  = sum(ln_out) / len(ln_out)
rms_mean = sum(rms_out) / len(rms_out)
print(f"  LayerNorm output mean:  {ln_mean:.6f}  (forced to 0 by mean subtraction)")
print(f"  RMSNorm output mean:    {rms_mean:.6f}  (not forced to 0 — mean subtraction skipped)")
print()
print("  RMSNorm is ~10% faster (no mean computation). Quality is nearly identical.")
print("  Qwen3.5 uses RMSNorm before every attention and FFN sub-layer.")


# ─────────────────────────────────────────────────────────
# DEMO 2: SwiGLU FFN vs Standard GELU FFN
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 2: SwiGLU FFN vs Standard GELU FFN ---")
print()
print("  Standard FFN (GPT-style):")
print("    out = GELU(x @ W1) @ W2")
print()
print("  SwiGLU FFN (Qwen3.5/LLaMA-style):")
print("    gate = x @ W_gate")
print("    up   = x @ W_up")
print("    out  = (silu(gate) * up) @ W_down")
print("    silu(z) = z * sigmoid(z)")
print()

def gelu(x):
    """Gaussian Error Linear Unit (approximate)."""
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

def silu(x):
    """Sigmoid Linear Unit: x * sigmoid(x)."""
    return x / (1 + math.exp(-x))

def standard_ffn(x, W1, W2):
    """Standard 2-matrix FFN with GELU."""
    hidden = [gelu(xi) for xi in mat_mul_vec(W1, x)]
    return mat_mul_vec(W2, hidden)

def swiglu_ffn(x, W_gate, W_up, W_down):
    """SwiGLU 3-matrix FFN."""
    gate   = mat_mul_vec(W_gate, x)
    up     = mat_mul_vec(W_up, x)
    hidden = [silu(gate[i]) * up[i] for i in range(len(gate))]
    return mat_mul_vec(W_down, hidden)

d_model = 4
d_ffn   = 6   # d_ffn = (2/3) * 4 * d_model for SwiGLU -- keep smaller

x_in = rand_vec(d_model)

# Standard FFN: 2 matrices
W1 = rand_mat(d_ffn, d_model)
W2 = rand_mat(d_model, d_ffn)

# SwiGLU FFN: 3 matrices (but d_ffn is smaller to keep param count equal)
d_ffn_swi = 5  # slightly smaller to compensate for 3 matrices
W_gate = rand_mat(d_ffn_swi, d_model)
W_up   = rand_mat(d_ffn_swi, d_model)
W_down = rand_mat(d_model, d_ffn_swi)

out_standard = standard_ffn(x_in, W1, W2)
out_swiglu   = swiglu_ffn(x_in, W_gate, W_up, W_down)

print(f"  d_model={d_model}, d_ffn={d_ffn} (standard), d_ffn={d_ffn_swi} (SwiGLU)")
print(f"  Input x: {[round(xi,3) for xi in x_in]}")
print(f"  Standard GELU FFN output: {[round(xi,4) for xi in out_standard]}")
print(f"  SwiGLU FFN output:        {[round(xi,4) for xi in out_swiglu]}")
print()

# SiLU demonstration
print("  SiLU activation (smooth gate function):")
print(f"  {'x':>8}  {'sigmoid(x)':>12}  {'silu(x)':>10}  {'GELU(x)':>10}")
print("  " + "-" * 46)
for x_val in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
    sig = 1 / (1 + math.exp(-x_val))
    print(f"  {x_val:>8.1f}  {sig:>12.4f}  {silu(x_val):>10.4f}  {gelu(x_val):>10.4f}")

print()
print("  SwiGLU: gate acts as a FILTER. silu(gate) ∈ [0, ∞).")
print("  Only information where gate is 'open' (large silu) flows through.")
print("  This learned gating improves quality vs standard GELU FFN.")


# ─────────────────────────────────────────────────────────
# DEMO 3: Qwen3.5 Model Architecture Summary
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 3: Qwen3.5 Architecture Assembly ---")
print()

# Parameter count for different model sizes
configs = [
    ("Mini-Qwen (demo)", 256,  6,  8, 2,  683),
    ("0.5B",              896, 24, 14, 2, 4864),
    ("1.7B",             1536, 28, 12, 2, 8960),
    ("7B",               3584, 28, 28, 4,18944),
    ("32B",              5120, 64, 64, 8,27648),
]

def count_params(d, L, n_q, n_kv, d_ffn, vocab=50257, head_dim=128):
    """Estimate total parameter count."""
    # Per block: Q, K, V, O projections + 3 FFN matrices + 2 RMSNorm
    per_block = (d*(n_q*head_dim) + d*(n_kv*head_dim)*2 + d*(n_q*head_dim) +  # QKV+O
                 d*d_ffn*3 +  # SwiGLU (gate, up, down)
                 d*2)         # 2 RMSNorm gamma vectors
    embedding = vocab * d
    final_norm = d
    # Output head is tied with embedding — no extra params
    return L * per_block + embedding + final_norm

print(f"  {'Model':>18}  {'d':>6}  {'L':>4}  {'n_q':>5}  {'n_kv':>6}  {'Params':>12}")
print("  " + "-" * 58)
for name, d, L, n_q, n_kv, d_ffn in configs:
    total = count_params(d, L, n_q, n_kv, d_ffn)
    total_str = f"{total/1e9:.1f}B" if total >= 1e9 else f"{total/1e6:.1f}M"
    print(f"  {name:>18}  {d:>6}  {L:>4}  {n_q:>5}  {n_kv:>6}  {total_str:>12}")

print()
print("  NanoGPT (M05) vs Mini-Qwen:")
print(f"  {'Feature':>22}  {'NanoGPT (M05)':>20}  {'Mini-Qwen (M18)':>18}")
print("  " + "-" * 66)
comparisons = [
    ("Positional encoding",  "Learned table",        "RoPE (computed)"),
    ("Attention type",       "MHA (n_q = n_kv)",     "GQA + RLA hybrid"),
    ("Normalization",        "LayerNorm",             "RMSNorm"),
    ("FFN activation",       "GELU (2 matrices)",     "SwiGLU (3 matrices)"),
    ("Context limit",        "Fixed at train size",   "Extrapolates beyond"),
    ("KV cache",             "None",                  "Yes (GQA layers)"),
    ("Memory at inference",  "O(N²) attention",       "O(N) hybrid"),
]
for feat, nano, mini in comparisons:
    print(f"  {feat:>22}  {nano:>20}  {mini:>18}")

print()
print("  Both are decoder-only transformers predicting the next token.")
print("  Mini-Qwen = NanoGPT + modern engineering improvements.")
print()
print("  Next step: build the full Mini-Qwen → see project_mini_qwen.py")

"""
Example 02: Group-Query Attention (GQA)
Module 18: Qwen3.5 LLM from Scratch

Run:  python example_02_gqa.py
Deps: none (pure Python)
"""

import math

print("=" * 60)
print("  Example 02: Group-Query Attention (GQA)")
print("=" * 60)


# ─────────────────────────────────────────────────────────
# DEMO 1: KV Cache Memory — MHA vs GQA vs MQA
# ─────────────────────────────────────────────────────────

print("\n--- DEMO 1: KV Cache Memory Comparison ---")
print()
print("  Formula: 2 × L × H_kv × S × D × B")
print("  L=layers, H_kv=KV heads, S=seq_len, D=head_dim, B=bytes")
print()

def kv_cache_bytes(L, H_kv, S, D, B=2):
    """KV cache memory in bytes. B=2 for bf16, 1 for INT8."""
    return 2 * L * H_kv * S * D * B

def fmt_gb(b):
    return f"{b / 1e9:.2f} GB"

# Qwen3.5-7B config
L, D = 28, 128  # layers, head_dim
seq = 32768      # 32K context

variants = [
    ("MHA  (32 KV heads)", 32),
    ("GQA  (8 KV heads)",   8),
    ("MQA  (1 KV head)",    1),
]

print(f"  Qwen3.5-7B: {L} layers, head_dim={D}, seq={seq//1024}K, bf16")
print()
print(f"  {'Variant':<24}  {'H_kv':>6}  {'KV Cache':>10}  {'vs MHA':>10}")
print("  " + "-" * 55)
mha_bytes = kv_cache_bytes(L, 32, seq, D)
for name, H_kv in variants:
    b = kv_cache_bytes(L, H_kv, seq, D)
    ratio = mha_bytes / b
    print(f"  {name:<24}  {H_kv:>6}  {fmt_gb(b):>10}  {ratio:>9.1f}x smaller")

print()
print("  GQA (8 heads): 4x less KV cache, near-MHA quality.")
print("  MQA (1 head):  32x less KV cache, quality drops at scale.")

# Show how memory scales with sequence length
print()
print(f"  GQA KV cache growth with sequence length (H_kv=8, {L} layers, bf16):")
print(f"  {'Seq len':>10}  {'KV Cache':>12}")
print("  " + "-" * 25)
for s in [1024, 4096, 16384, 32768, 65536, 131072]:
    b = kv_cache_bytes(L, 8, s, D)
    label = f"{s//1024}K" if s >= 1024 else str(s)
    print(f"  {label:>10}  {fmt_gb(b):>12}")


# ─────────────────────────────────────────────────────────
# DEMO 2: repeat_kv — Expanding KV Heads at Compute Time
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 2: repeat_kv — Expanding KV Heads ---")
print()
print("  KV cache stores 8 heads. Before attention, expand to 32 by repeating.")
print("  Each KV head is shared by (32/8 = 4) Q heads.")
print()

def repeat_kv(kv_heads, n_kv_heads, n_q_heads):
    """
    Expand KV heads from n_kv_heads to n_q_heads by repeating each head.
    kv_heads: list of n_kv_heads vectors (each = one KV head's data)
    Returns: list of n_q_heads vectors
    """
    group_size = n_q_heads // n_kv_heads
    expanded = []
    for kv_head in kv_heads:
        for _ in range(group_size):
            expanded.append(kv_head)
    return expanded

# Simulate with 4 KV heads expanded to 8 Q heads
n_kv = 4
n_q  = 8
kv_heads = [f"KV_{i}" for i in range(n_kv)]

expanded = repeat_kv(kv_heads, n_kv, n_q)
print(f"  Original KV heads ({n_kv} heads): {kv_heads}")
print(f"  After repeat_kv  ({n_q} heads): {expanded}")
print()

# Show the grouping clearly
group_size = n_q // n_kv
print(f"  Group assignments (group_size = {group_size}):")
for q_head in range(n_q):
    kv_head = q_head // group_size
    print(f"    Q head {q_head} → uses KV head {kv_head}  (group {kv_head})")

print()
print("  repeat_kv is a VIEW operation — no new data created in memory.")
print("  The cache stores n_kv heads; the expansion is just indexing.")


# ─────────────────────────────────────────────────────────
# DEMO 3: GQA Attention Scores (Simplified, Pure Python)
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 3: GQA Attention Scores ---")
print()

def softmax_1d(scores):
    """Softmax over a list of scores."""
    max_s = max(scores)
    exps  = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def gqa_attention(Q_heads, K_heads, V_heads, n_q, n_kv, head_dim):
    """
    Simplified GQA attention (one token, seq_len=4 context tokens).
    Q_heads: list of n_q query vectors
    K_heads: list of n_kv key matrices (each = list of seq_len vectors)
    V_heads: list of n_kv value matrices
    Returns: output for each Q head
    """
    group_size = n_q // n_kv
    scale = 1.0 / math.sqrt(head_dim)
    outputs = []

    for q_idx in range(n_q):
        kv_idx  = q_idx // group_size  # which KV head to use
        q_vec   = Q_heads[q_idx]
        k_seqs  = K_heads[kv_idx]     # shared KV head
        v_seqs  = V_heads[kv_idx]

        # Attention scores: Q · K^T
        scores = [dot(q_vec, k_vec) * scale for k_vec in k_seqs]
        weights = softmax_1d(scores)

        # Weighted sum of V
        out = [sum(weights[s] * v_seqs[s][d] for s in range(len(v_seqs)))
               for d in range(head_dim)]
        outputs.append(out)

    return outputs

# Tiny example: n_q=4, n_kv=2, head_dim=2, seq_len=3
n_q, n_kv, head_dim, seq_len = 4, 2, 2, 3

import random
rng = random.Random(42)
rand = lambda: rng.uniform(-1, 1)

Q_heads = [[rand(), rand()] for _ in range(n_q)]
K_heads = [[[rand(), rand()] for _ in range(seq_len)] for _ in range(n_kv)]
V_heads = [[[rand(), rand()] for _ in range(seq_len)] for _ in range(n_kv)]

outputs = gqa_attention(Q_heads, K_heads, V_heads, n_q, n_kv, head_dim)

print(f"  Config: n_q={n_q}, n_kv={n_kv}, head_dim={head_dim}, seq_len={seq_len}")
print(f"  Group size = {n_q // n_kv} (each KV head serves {n_q // n_kv} Q heads)")
print()
print(f"  {'Q head':>8}  {'Uses KV head':>14}  {'Output vector':>20}")
print("  " + "-" * 48)
for i, out in enumerate(outputs):
    kv_idx = i // (n_q // n_kv)
    print(f"  {i:>8}  {kv_idx:>14}  ({out[0]:>7.4f}, {out[1]:>7.4f})")

print()
print("  Q heads 0,1 share KV head 0. Q heads 2,3 share KV head 1.")
print("  Each Q head produces different output (different W_q projections)")
print("  but attends to the SAME keys and values within its group.")

# Weight matrix size comparison
print()
print("\n  Weight matrix sizes: MHA vs GQA (d_model=4096, head_dim=128)")
d_model  = 4096
n_q_full = 32
for label, n_kv_w in [("MHA (32 KV)", 32), ("GQA (8 KV)", 8), ("MQA (1 KV)", 1)]:
    w_q = d_model * (n_q_full * 128) / 1e6
    w_k = d_model * (n_kv_w  * 128) / 1e6
    w_v = d_model * (n_kv_w  * 128) / 1e6
    total = w_q + w_k + w_v
    print(f"  {label}: W_q={w_q:.1f}M  W_k={w_k:.1f}M  W_v={w_v:.1f}M  total={total:.1f}M")

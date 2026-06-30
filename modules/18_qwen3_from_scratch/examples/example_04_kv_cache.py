"""
Example 04: KV Cache Management
Module 18: Qwen3.5 LLM from Scratch

Run:  python example_04_kv_cache.py
Deps: none (pure Python)
"""

import math

print("=" * 60)
print("  Example 04: KV Cache Management")
print("=" * 60)


# ─────────────────────────────────────────────────────────
# DEMO 1: Full Cache — Append and Attend
# ─────────────────────────────────────────────────────────

print("\n--- DEMO 1: Full KV Cache Simulation ---")
print()
print("  During generation, each new token's K and V are appended to the cache.")
print("  Attention at step t uses the FULL cache K[0..t], V[0..t].")
print()

class SimpleKVCache:
    """Simulates a single-layer KV cache for one attention head."""

    def __init__(self, head_dim):
        self.K = []   # list of K vectors, one per token
        self.V = []   # list of V vectors
        self.head_dim = head_dim

    def append(self, k_new, v_new):
        """Append K, V for a new token."""
        self.K.append(k_new)
        self.V.append(v_new)

    def size_bytes(self, bytes_per_val=2):
        """Memory usage: 2 (K+V) × seq_len × head_dim × bytes."""
        seq_len = len(self.K)
        return 2 * seq_len * self.head_dim * bytes_per_val

    def attend(self, q_vec):
        """Compute attention output for query q_vec over cached K, V."""
        if not self.K:
            return [0.0] * self.head_dim
        scale = 1.0 / math.sqrt(self.head_dim)
        # Scores
        scores = [sum(q_vec[i] * k[i] for i in range(self.head_dim)) * scale
                  for k in self.K]
        # Softmax
        max_s = max(scores)
        exps  = [math.exp(s - max_s) for s in scores]
        total = sum(exps)
        weights = [e / total for e in exps]
        # Weighted sum of V
        output = [sum(weights[t] * self.V[t][d] for t in range(len(self.V)))
                  for d in range(self.head_dim)]
        return output

import random
rng = random.Random(42)
rand_vec = lambda d: [round(rng.uniform(-1, 1), 3) for _ in range(d)]

head_dim = 4
cache = SimpleKVCache(head_dim)

# Simulate generating 6 tokens
prompt_tokens = ["The", "cat", "sat"]
gen_tokens    = ["on", "the", "mat"]

print("  Prompt processing (parallel — all at once):")
for tok in prompt_tokens:
    k, v = rand_vec(head_dim), rand_vec(head_dim)
    cache.append(k, v)
    print(f"    Added '{tok}': cache size = {len(cache.K)} tokens  "
          f"({cache.size_bytes()} bytes, {cache.size_bytes()/1024:.1f} KB)")

print()
print("  Generation (one token at a time — uses cache):")
for tok in gen_tokens:
    q = rand_vec(head_dim)
    out = cache.attend(q)
    k, v = rand_vec(head_dim), rand_vec(head_dim)
    cache.append(k, v)
    print(f"    Generated '{tok}': attended over {len(cache.K)-1} past tokens  "
          f"cache = {cache.size_bytes()} bytes")

print()
print(f"  Final cache: {len(cache.K)} tokens × {head_dim} dims × 2 (K+V) = "
      f"{cache.size_bytes()} bytes")


# ─────────────────────────────────────────────────────────
# DEMO 2: Sliding Window Eviction
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 2: Sliding Window Cache ---")
print()
print("  Fixed memory: only keep the last W tokens.")
print("  Older tokens evicted — cannot attend to them anymore.")
print()

def sliding_window_evict(K_list, V_list, window_size):
    """Trim the cache to the last `window_size` tokens."""
    if len(K_list) <= window_size:
        return K_list, V_list
    return K_list[-window_size:], V_list[-window_size:]

window = 4
K_all  = [f"K{i}" for i in range(10)]
V_all  = [f"V{i}" for i in range(10)]

print(f"  Window size W = {window}")
print(f"  {'Step':>5}  {'Total seen':>12}  {'In cache':>10}  {'Evicted':>20}")
print("  " + "-" * 52)
for step in range(1, 11):
    K_trimmed, _ = sliding_window_evict(K_all[:step], V_all[:step], window)
    n_evicted = step - len(K_trimmed)
    evicted_str = f"K0..K{n_evicted-1}" if n_evicted > 0 else "none"
    in_cache = f"K{step-len(K_trimmed)}..K{step-1}"
    print(f"  {step:>5}  {step:>12}  {len(K_trimmed):>6} {in_cache:<12}  {evicted_str:>20}")

print()
print(f"  Memory is FIXED at {window} tokens regardless of total generation length.")
print("  Trade-off: cannot attend to early prompt tokens when generating late tokens.")


# ─────────────────────────────────────────────────────────
# DEMO 3: INT8 KV Cache Quantization
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 3: INT8 KV Cache Quantization ---")
print()
print("  Store K/V in INT8 (1 byte) instead of bf16 (2 bytes) = 50% memory saved.")
print()

def quantize_int8(vector):
    """
    Quantize a float vector to INT8 symmetrically.
    Returns (quantized list of int, scale float).
    """
    max_abs = max(abs(x) for x in vector)
    if max_abs == 0:
        return [0] * len(vector), 1.0
    scale = max_abs / 127.0
    quantized = [max(-128, min(127, round(x / scale))) for x in vector]
    return quantized, scale

def dequantize_int8(quantized, scale):
    """Restore float vector from INT8 + scale."""
    return [q * scale for q in quantized]

# Test on a typical K vector
K_float = [0.847, -0.234, 1.562, -1.023, 0.389, -0.678, 0.912, 0.156]

K_int8, scale = quantize_int8(K_float)
K_restored = dequantize_int8(K_int8, scale)

print(f"  Original (bf16, {len(K_float)*2} bytes): {[round(x,3) for x in K_float]}")
print(f"  Quantized INT8 (scale={scale:.4f}):    {K_int8}")
print(f"  Restored float:                    {[round(x,4) for x in K_restored]}")
print()

# Compute quantization error
errors = [abs(K_float[i] - K_restored[i]) for i in range(len(K_float))]
max_err = max(errors)
avg_err = sum(errors) / len(errors)
print(f"  Max quantization error: {max_err:.5f}")
print(f"  Avg quantization error: {avg_err:.5f}")
print(f"  Max relative error: {max_err/max(abs(x) for x in K_float)*100:.2f}%")

# Memory comparison
print()
print("  Memory comparison (Qwen3.5-7B, 32K context):")
L, H_kv, S, D = 28, 8, 32768, 128
bf16_bytes = 2 * L * H_kv * S * D * 2
int8_bytes  = 2 * L * H_kv * S * D * 1
print(f"    bf16 KV cache: {bf16_bytes/1e9:.2f} GB")
print(f"    INT8 KV cache: {int8_bytes/1e9:.2f} GB")
print(f"    Saving:        {(1 - int8_bytes/bf16_bytes)*100:.0f}%")

# Strategy guide
print()
print("\n  STRATEGY SELECTION GUIDE:")
print(f"  {'Use case':>35}  {'Best strategy'}")
print("  " + "-" * 65)
strategies = [
    ("Short generation < 2K, few users",      "Full cache (simple)"),
    ("Long generation > 4K, memory tight",    "Sliding window + INT8"),
    ("Chatbot with shared system prompt",      "Prefix caching"),
    ("High-throughput server (vLLM)",          "Paged attention"),
    ("Mobile / edge device",                  "Sliding window + INT8 + GQA"),
]
for use_case, strategy in strategies:
    print(f"  {use_case:>35}  {strategy}")

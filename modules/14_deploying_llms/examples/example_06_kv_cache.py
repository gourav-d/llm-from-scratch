# =============================================================================
# Module 14 - Deploying LLMs
# Example 06: KV Cache & Inference Optimization
# =============================================================================
#
# WHAT THIS FILE TEACHES:
#   - Why KV cache exists and what problem it solves
#   - How to simulate attention WITHOUT KV cache (naive, O(N^2))
#   - How to simulate attention WITH KV cache (O(N))
#   - How to calculate KV cache memory cost
#   - What INT8 KV cache quantization means
#
# GLOSSARY:
#   KV cache      - Saved Key and Value matrices from past tokens.
#                   Prevents re-computing them on every generation step.
#   Query (Q)     - "What am I looking for?" matrix. Computed fresh each step.
#   Key   (K)     - "What do I contain?" matrix. Computed once, then cached.
#   Value (V)     - "What info do I pass?" matrix. Computed once, then cached.
#   softmax       - Converts raw attention scores into probabilities (sum to 1)
#   head_dim      - Dimension per attention head = hidden_dim / num_heads
#   prefill       - Processing all prompt tokens at once (parallel, fast)
#   decode        - Generating tokens one at a time (sequential, slow)
#
# C# ANALOGY:
#   KV cache = Dictionary<int, (float[] K, float[] V)>
#   Key = tokenIndex, Value = pre-computed K and V arrays.
#   Without cache: recompute all K and V from scratch every step (like O(N^2) loop)
#   With    cache: look up cached results, only compute for the new token (O(N))
#
# LIBRARIES:
#   numpy  - numerical arrays (like List<double> but multi-dimensional)
#   time   - measure elapsed time (like Stopwatch in C#)
#
# =============================================================================

import numpy as np   # numerical computing library
import time          # for measuring execution time

# =============================================================================
# SETUP: Tiny model dimensions (real models are much larger)
# =============================================================================

np.random.seed(42)   # makes random numbers reproducible

# Model hyperparameters (intentionally tiny so this runs fast)
NUM_LAYERS  = 2    # real LLaMA-7B has 32 layers
NUM_HEADS   = 2    # real LLaMA-7B has 32 heads
HEAD_DIM    = 8    # real LLaMA-7B has head_dim=128
HIDDEN_DIM  = NUM_HEADS * HEAD_DIM   # 2 * 8 = 16 total hidden dimension

# Random weight matrices for Q, K, V projections (one per layer)
# In a real model these are learned during training
# Shape: (hidden_dim, hidden_dim) — projects input to Q, K, or V space
W_Q = [np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1 for _ in range(NUM_LAYERS)]
W_K = [np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1 for _ in range(NUM_LAYERS)]
W_V = [np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1 for _ in range(NUM_LAYERS)]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def softmax(x):
    """
    Softmax: converts raw scores into probabilities.
    Each row sums to 1.0.

    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))

    We subtract max(x) first for numerical stability
    (prevents overflow when exp() gets very large numbers).
    """
    # x - x.max(axis=-1, keepdims=True) shifts values so max is 0
    # This prevents exp() overflow — same result but numerically safe
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)   # divide each by row sum


def scaled_dot_product_attention(Q, K, V):
    """
    Core attention operation.

    Given Query, Key, Value matrices:
      1. Compute similarity: Q @ K^T   (dot product of each Q with each K)
      2. Scale by sqrt(head_dim)       (prevents large values that hurt softmax)
      3. Convert to probabilities      (softmax)
      4. Weighted sum of V             (how much of each Value to take)

    Shapes:
      Q: (seq_len, head_dim)   — what we're looking for
      K: (seq_len, head_dim)   — what each position contains
      V: (seq_len, head_dim)   — what information each position holds
      output: (seq_len, head_dim)

    C# analogy:
      Like computing a weighted average of a list,
      where the weights are determined by how "similar" Q is to each K.
    """
    scale = np.sqrt(HEAD_DIM)             # scaling factor (prevents gradient issues)
    scores = Q @ K.T / scale              # (seq_len, seq_len) similarity matrix
    weights = softmax(scores)             # convert scores to probabilities
    output = weights @ V                  # weighted sum of values
    return output

# =============================================================================
# PART A: NAIVE GENERATION (no KV cache)
#
# Every step recomputes K and V for ALL tokens seen so far.
# This is O(N^2) total work.
# =============================================================================

print("=" * 60)
print("PART A: Naive Generation (No KV Cache)")
print("=" * 60)

def generate_token_naive(token_embeddings, layer):
    """
    Process ALL tokens to generate attention output for the last token.

    This is the WRONG (slow) way — it recomputes K and V for every past
    token on every single generation step.

    Parameters:
        token_embeddings: (seq_len, hidden_dim) — all tokens so far
        layer: int — which transformer layer we are in

    Returns:
        output for the last (newest) token only
    """
    seq_len = token_embeddings.shape[0]

    # --- Compute Q, K, V for EVERY token (wasteful if we did this before) ---
    # token_embeddings @ W_K[layer] projects each token into Key space
    # Shape: (seq_len, hidden_dim) @ (hidden_dim, hidden_dim) = (seq_len, hidden_dim)
    Q_all = token_embeddings @ W_Q[layer]   # (seq_len, hidden_dim) — query for all tokens
    K_all = token_embeddings @ W_K[layer]   # (seq_len, hidden_dim) — key for all tokens
    V_all = token_embeddings @ W_V[layer]   # (seq_len, hidden_dim) — value for all tokens

    # --- Run attention over all tokens ---
    output_all = scaled_dot_product_attention(Q_all, K_all, V_all)

    # We only care about the LAST token's output (the newly generated token)
    return output_all[-1]   # shape: (hidden_dim,)


# Simulate generating a sequence token by token
NUM_TOKENS_TO_GENERATE = 20   # generate 20 tokens one at a time

# Start with a random initial token embedding (like the prompt)
initial_embedding = np.random.randn(1, HIDDEN_DIM)   # shape: (1, hidden_dim)
all_embeddings = initial_embedding.copy()             # grows as we generate

naive_outputs = []          # store generated token outputs
naive_kv_recomputes = 0     # count how many K/V computations we do

start_time = time.time()

for step in range(NUM_TOKENS_TO_GENERATE):
    seq_len = all_embeddings.shape[0]                  # current sequence length

    # Count K/V computations: seq_len tokens x 2 matrices (K and V) x NUM_LAYERS layers
    naive_kv_recomputes += seq_len * 2 * NUM_LAYERS    # every token recomputed!

    # Generate next token output (layer 0 only for simplicity)
    output = generate_token_naive(all_embeddings, layer=0)
    naive_outputs.append(output)

    # Simulate "appending" the generated token
    # In reality, the output goes through more layers and becomes the next embedding
    new_embedding = output.reshape(1, -1)              # shape: (1, hidden_dim)
    all_embeddings = np.vstack([all_embeddings, new_embedding])  # append to sequence

naive_time = time.time() - start_time

print(f"\nGenerated {NUM_TOKENS_TO_GENERATE} tokens (naive)")
print(f"Total K/V computations: {naive_kv_recomputes:,}")
print(f"Time: {naive_time*1000:.2f} ms")
print(f"Formula: sum(1 to {NUM_TOKENS_TO_GENERATE+1}) * 2 layers = {sum(range(1, NUM_TOKENS_TO_GENERATE+2)) * 2:,}")

# =============================================================================
# PART B: GENERATION WITH KV CACHE
#
# We compute K and V for each token ONCE, save them, and reuse.
# Only the Query is computed fresh each step (from the new token).
# This is O(N) total work.
# =============================================================================

print("\n" + "=" * 60)
print("PART B: Generation With KV Cache")
print("=" * 60)

def init_kv_cache(num_layers):
    """
    Initialize an empty KV cache.

    Structure: list of dicts, one per layer.
    Each dict has 'K' and 'V' lists that grow as tokens are generated.

    C# analogy:
        var cache = new List<Dictionary<string, List<float[]>>>();
        for each layer: cache.Add(new() { {"K", new()}, {"V", new()} });
    """
    return [{'K': [], 'V': []} for _ in range(num_layers)]
    # cache[layer]['K'] is a list of K vectors, one per past token
    # cache[layer]['V'] is a list of V vectors, one per past token


def generate_token_with_cache(new_token_embedding, kv_cache, layer):
    """
    Process ONLY the new token and use cached K/V for past tokens.

    Steps:
      1. Compute Q, K, V for the new token ONLY (1 token, not all)
      2. Append new K and V to the cache
      3. Build full K_all and V_all from cache (retrieve past + new)
      4. Compute attention: Q_new attends to all cached K/V

    This is O(1) new computation per step instead of O(N).

    Parameters:
        new_token_embedding: (1, hidden_dim) — just the newest token
        kv_cache: list of dicts with 'K' and 'V' lists
        layer: int — which transformer layer

    Returns:
        output for the new token, updated kv_cache
    """
    # Step 1: Compute Q, K, V for the NEW token only (1 row, not seq_len rows)
    # Shape: (1, hidden_dim) @ (hidden_dim, hidden_dim) = (1, hidden_dim)
    Q_new = new_token_embedding @ W_Q[layer]   # query for new token
    K_new = new_token_embedding @ W_K[layer]   # key for new token
    V_new = new_token_embedding @ W_V[layer]   # value for new token

    # Step 2: Save new K and V to cache
    # .squeeze(0) removes the length-1 dimension: (1, hidden_dim) → (hidden_dim,)
    kv_cache[layer]['K'].append(K_new.squeeze(0))
    kv_cache[layer]['V'].append(V_new.squeeze(0))

    # Step 3: Reconstruct full K and V matrices from cache (past + new)
    # np.array(...) converts list of vectors to a 2D matrix
    # Shape: (seq_len_so_far, hidden_dim)
    K_all = np.array(kv_cache[layer]['K'])   # all cached keys, including new one
    V_all = np.array(kv_cache[layer]['V'])   # all cached values, including new one

    # Step 4: Attention — Q_new (1 token) attends to K_all and V_all (all tokens)
    output = scaled_dot_product_attention(Q_new, K_all, V_all)

    return output.squeeze(0), kv_cache   # return (hidden_dim,) output and updated cache


# Reset and run generation WITH KV cache
all_embeddings_cached = initial_embedding.copy()   # same starting point
kv_cache = init_kv_cache(NUM_LAYERS)

# Prefill phase: process the initial token and populate the cache
prefill_token = all_embeddings_cached[0:1]    # shape: (1, hidden_dim)
for layer_idx in range(NUM_LAYERS):
    _, kv_cache = generate_token_with_cache(prefill_token, kv_cache, layer=layer_idx)

cached_outputs = []
cached_kv_computes = 0   # count K/V computations (should be 1 per step per layer)

start_time = time.time()

for step in range(NUM_TOKENS_TO_GENERATE):
    # Count K/V computations: only 1 new token x 2 matrices x NUM_LAYERS (not seq_len!)
    cached_kv_computes += 1 * 2 * NUM_LAYERS    # ALWAYS 1 token, never grows!

    # Generate using cache (layer 0 for simplicity)
    output, kv_cache = generate_token_with_cache(
        all_embeddings_cached[-1:],   # just the last token
        kv_cache,
        layer=0
    )
    cached_outputs.append(output)

    # Append generated token to sequence
    new_embedding = output.reshape(1, -1)
    all_embeddings_cached = np.vstack([all_embeddings_cached, new_embedding])

cached_time = time.time() - start_time

print(f"\nGenerated {NUM_TOKENS_TO_GENERATE} tokens (with KV cache)")
print(f"Total K/V computations: {cached_kv_computes:,}")
print(f"Time: {cached_time*1000:.2f} ms")
print(f"Per step: always 2 (K and V for 1 new token) x {NUM_LAYERS} layers = {2*NUM_LAYERS}")

# =============================================================================
# PART C: COMPARISON
# =============================================================================

print("\n" + "=" * 60)
print("PART C: Comparison")
print("=" * 60)

reduction = naive_kv_recomputes / cached_kv_computes
print(f"\n  Naive K/V computations:  {naive_kv_recomputes:>6,}")
print(f"  Cached K/V computations: {cached_kv_computes:>6,}")
print(f"  Reduction:               {reduction:.1f}x fewer computations")
print(f"\n  Naive time:              {naive_time*1000:.2f} ms")
print(f"  Cached time:             {cached_time*1000:.2f} ms")

# Show how reduction grows with sequence length
print("\n--- How cache benefit scales with sequence length ---")
print(f"  {'Seq Len':>10} | {'Naive ops':>12} | {'Cached ops':>12} | {'Speedup':>10}")
print("  " + "-" * 52)
for seq_len in [10, 100, 500, 1000, 4000]:
    naive_ops  = sum(range(1, seq_len + 1)) * 2   # O(N^2)
    cached_ops = seq_len * 2                       # O(N)
    speedup    = naive_ops / cached_ops
    print(f"  {seq_len:>10,} | {naive_ops:>12,} | {cached_ops:>12,} | {speedup:>9.0f}x")

# =============================================================================
# PART D: KV CACHE MEMORY COST CALCULATOR
# =============================================================================

print("\n" + "=" * 60)
print("PART D: KV Cache Memory Cost Calculator")
print("=" * 60)

def kv_cache_memory_gb(num_layers, num_heads, seq_len, head_dim, dtype_bytes=2):
    """
    Calculate total KV cache memory in gigabytes.

    Formula:
        bytes = 2 * num_layers * num_heads * seq_len * head_dim * dtype_bytes
        2 = one K matrix + one V matrix

    Parameters:
        num_layers:   number of transformer layers
        num_heads:    number of attention heads
        seq_len:      current sequence length (tokens)
        head_dim:     dimension per head = hidden_dim / num_heads
        dtype_bytes:  2 for fp16 (default), 1 for INT8, 4 for fp32

    Returns:
        float — memory in gigabytes
    """
    total_bytes = (
        2             # K and V — two matrices
        * num_layers  # one set per layer
        * num_heads   # one set per head (multi-head attention)
        * seq_len     # grows as tokens are generated
        * head_dim    # size of each K or V vector
        * dtype_bytes # bytes per value (2=fp16, 1=INT8)
    )
    return total_bytes / (1024 ** 3)   # convert bytes to gigabytes


# Real model configurations
models = [
    # (name,          layers, heads, head_dim)
    ("GPT-2 Small",       12,    12,      64),
    ("LLaMA-7B",          32,    32,     128),
    ("LLaMA-13B",         40,    40,     128),
    ("LLaMA-70B",         80,    64,     128),
    ("GPT-4 (estimated)", 96,   128,     128),
]

context_lengths = [1024, 4096, 32768]   # 1K, 4K, 32K tokens

print(f"\n{'Model':<22} | {'ctx':>6} | {'fp16 (GB)':>10} | {'INT8 (GB)':>10} | {'Savings'}")
print("-" * 70)

for name, layers, heads, head_dim in models:
    for ctx_len in context_lengths:
        fp16_gb = kv_cache_memory_gb(layers, heads, ctx_len, head_dim, dtype_bytes=2)
        int8_gb = kv_cache_memory_gb(layers, heads, ctx_len, head_dim, dtype_bytes=1)
        savings_pct = (fp16_gb - int8_gb) / fp16_gb * 100
        print(f"  {name:<20} | {ctx_len:>6,} | {fp16_gb:>10.2f} | {int8_gb:>10.2f} | {savings_pct:.0f}% saved")
    print()   # blank line between models

# =============================================================================
# PART E: INT8 KV CACHE QUANTIZATION
# =============================================================================

print("=" * 60)
print("PART E: INT8 KV Cache Quantization")
print("=" * 60)

print("""
Standard KV cache stores values in fp16 (2 bytes each).
INT8 KV cache stores them in INT8 (1 byte each) — 2x smaller.

The quantization works the same as model weight quantization:
  1. For each vector in the cache, find the scale factor
  2. Quantize: round(value / scale), clip to [-127, 127]
  3. Dequantize on retrieval: quantized * scale
""")

def quantize_kv_vector(vector_fp16):
    """
    Quantize a single K or V vector from fp16 to INT8.

    Steps:
      1. Find max absolute value in vector (the range)
      2. Compute scale = max_abs / 127
      3. Quantize: int8 = round(vector / scale), clipped to [-127, 127]

    Returns quantized INT8 array and the scale factor (needed for dequantization).
    """
    max_abs = np.max(np.abs(vector_fp16))       # find the range
    if max_abs == 0:
        return np.zeros_like(vector_fp16, dtype=np.int8), 1.0
    scale = max_abs / 127.0                      # map range to [-127, 127]
    quantized = np.round(vector_fp16 / scale)    # convert to integers
    quantized = np.clip(quantized, -127, 127)    # ensure within int8 range
    return quantized.astype(np.int8), scale      # cast to int8


def dequantize_kv_vector(quantized_int8, scale):
    """
    Recover approximate fp32 values from an INT8-quantized K or V vector.

    Formula: recovered = quantized * scale
    """
    return quantized_int8.astype(np.float32) * scale


# Demonstrate on a sample K vector
sample_k_vector = np.random.randn(HEAD_DIM).astype(np.float32)   # original K vector

# Quantize to INT8
k_int8, k_scale = quantize_kv_vector(sample_k_vector)

# Dequantize back to float
k_recovered = dequantize_kv_vector(k_int8, k_scale)

# Measure error
max_error = np.max(np.abs(sample_k_vector - k_recovered))
mean_error = np.mean(np.abs(sample_k_vector - k_recovered))

print(f"  Original K vector (fp32, {sample_k_vector.nbytes} bytes): {sample_k_vector[:4].round(4)}")
print(f"  Quantized   (INT8, {k_int8.nbytes} bytes):   {k_int8[:4]}")
print(f"  Recovered (fp32):               {k_recovered[:4].round(4)}")
print(f"  Max error:   {max_error:.6f}")
print(f"  Mean error:  {mean_error:.6f}")
print(f"  Memory saved: {(1 - k_int8.nbytes / sample_k_vector.nbytes)*100:.0f}%")
print(f"\n  Key insight: error is tiny, memory cut in half.")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Without KV cache:
  - Recompute K and V for ALL tokens every step
  - O(N^2) total computation
  - Unusable for long contexts

With KV cache:
  - Compute K and V once per token, save them
  - O(N) total computation
  - Makes long contexts practical

Memory cost formula:
  2 x layers x heads x seq_len x head_dim x dtype_bytes

INT8 KV cache:
  - Store cached K and V as INT8 instead of fp16
  - 2x memory saving
  - Tiny quality loss (max error ~0.001)

Paged attention (vLLM):
  - Allocate KV cache in fixed pages, on demand
  - No wasted memory for short sequences
  - Enables batching many requests efficiently
""")

# =============================================================================
# Module 14 - Deploying LLMs
# Exercise 06: KV Cache & Inference Optimization
# =============================================================================
#
# INSTRUCTIONS:
#   Complete each TODO section below.
#   Run the file to check your answers against the expected output.
#   Scroll to the bottom for hints and solutions.
#
# WHAT YOU PRACTICE:
#   - Calculating KV cache memory for real models
#   - Implementing a simple KV cache from scratch
#   - Counting FLOPs with and without cache
#   - Understanding INT8 KV cache quantization
#
# LIBRARIES NEEDED: numpy only
#
# =============================================================================

import numpy as np

print("=" * 60)
print("Exercise 06: KV Cache")
print("=" * 60)

# =============================================================================
# EXERCISE 1: KV Cache Memory Calculator
#
# Given model config, calculate KV cache memory in GB.
#
# Formula:
#   bytes = 2 * num_layers * num_heads * seq_len * head_dim * dtype_bytes
#
# Where:
#   2          = K matrix + V matrix
#   dtype_bytes = 2 for fp16, 1 for INT8
# =============================================================================

print("\n--- Exercise 1: KV Cache Memory Calculator ---")

def kv_cache_memory_gb(num_layers, num_heads, seq_len, head_dim, dtype="fp16"):
    """
    Calculate KV cache memory in GB.

    Parameters:
        num_layers:  number of transformer layers
        num_heads:   number of attention heads
        seq_len:     number of tokens in context
        head_dim:    dimension per head
        dtype:       "fp16" (2 bytes) or "int8" (1 byte)

    Returns:
        float -- memory in gigabytes
    """
    # TODO: Map dtype to bytes per value
    dtype_bytes = {
        # TODO: fill in this dictionary
        # "fp16": ?, "int8": ?
    }

    if dtype not in dtype_bytes:
        raise ValueError(f"Unknown dtype: {dtype}")

    # TODO: Apply the formula
    # bytes = 2 * num_layers * num_heads * seq_len * head_dim * dtype_bytes[dtype]
    total_bytes = None   # TODO

    # TODO: Convert bytes to GB (1 GB = 1024^3 bytes)
    return None   # TODO


# Test your function
test_cases = [
    # (name,         layers, heads, seq_len, head_dim, dtype,   expected_approx)
    ("LLaMA-7B fp16",   32,    32,   4096,     128,  "fp16",  "~2.0 GB"),
    ("LLaMA-7B int8",   32,    32,   4096,     128,  "int8",  "~1.0 GB"),
    ("LLaMA-70B fp16",  80,    64,   4096,     128,  "fp16",  "~20.0 GB"),
    ("LLaMA-70B int8",  80,    64,   4096,     128,  "int8",  "~10.0 GB"),
    ("GPT-2 fp16",      12,    12,   1024,      64,  "fp16",  "~0.05 GB"),
]

print("\nKV cache memory estimates:")
for name, layers, heads, seq_len, head_dim, dtype, expected in test_cases:
    result = kv_cache_memory_gb(layers, heads, seq_len, head_dim, dtype)
    if result is not None:
        print(f"  {name:<22}: {result:.3f} GB  (expected {expected})")
    else:
        print(f"  {name:<22}: TODO (expected {expected})")

# =============================================================================
# EXERCISE 2: Implement a KV Cache
#
# Build a simple KV cache class that:
#   - Stores K and V vectors per layer as they are computed
#   - Allows retrieval of all cached K/V for a given layer
#   - Reports current cache size (in tokens and bytes)
# =============================================================================

print("\n--- Exercise 2: KV Cache Class ---")

class KVCache:
    """
    Simple KV cache for a transformer model.

    Stores Key and Value vectors for each past token, per layer.

    Usage:
        cache = KVCache(num_layers=32, head_dim=128, num_heads=32)
        cache.update(layer=0, k_vector=..., v_vector=...)
        k_all, v_all = cache.get(layer=0)
    """

    def __init__(self, num_layers, head_dim, num_heads):
        """
        Initialize empty cache.

        Parameters:
            num_layers: int -- number of transformer layers
            head_dim:   int -- dimension per attention head
            num_heads:  int -- number of attention heads
        """
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.hidden_dim = head_dim * num_heads

        # TODO: Initialize storage for K and V vectors
        # Hint: self._k_cache[layer] should be a list that grows over time
        # self._k_cache = [[] for _ in range(num_layers)]
        self._k_cache = None   # TODO
        self._v_cache = None   # TODO

    def update(self, layer, k_vector, v_vector):
        """
        Add new K and V vectors for the latest token.

        Parameters:
            layer:    int -- which transformer layer
            k_vector: numpy array shape (hidden_dim,) -- key for new token
            v_vector: numpy array shape (hidden_dim,) -- value for new token
        """
        # TODO: Append k_vector and v_vector to the appropriate cache lists
        # Hint: self._k_cache[layer].append(k_vector)
        pass   # TODO

    def get(self, layer):
        """
        Retrieve all cached K and V matrices for a layer.

        Returns:
            k_all: (seq_len_so_far, hidden_dim) numpy array
            v_all: (seq_len_so_far, hidden_dim) numpy array
        """
        # TODO: Stack the list of vectors into a matrix
        # Hint: np.array(self._k_cache[layer])
        k_all = None   # TODO
        v_all = None   # TODO
        return k_all, v_all

    def seq_len(self):
        """Return number of tokens cached so far (in layer 0)."""
        # TODO: return len of _k_cache[0]
        return None   # TODO

    def memory_bytes(self, dtype_bytes=2):
        """
        Calculate current KV cache memory in bytes.

        Formula: 2 * num_layers * seq_len * hidden_dim * dtype_bytes
        """
        # TODO: Compute bytes used
        # Hint: 2 (K+V) * self.num_layers * self.seq_len() * self.hidden_dim * dtype_bytes
        return None   # TODO


# Test KVCache with a tiny model
LAYERS  = 2
HEADS   = 2
H_DIM   = 8
HIDDEN  = HEADS * H_DIM    # 16

cache = KVCache(num_layers=LAYERS, head_dim=H_DIM, num_heads=HEADS)

np.random.seed(42)

# Simulate 5 tokens being processed
for token_idx in range(5):
    for layer_idx in range(LAYERS):
        k = np.random.randn(HIDDEN)    # fake K vector for this token
        v = np.random.randn(HIDDEN)    # fake V vector for this token
        cache.update(layer=layer_idx, k_vector=k, v_vector=v)

# Test the cache
seq_len_result = cache.seq_len()
k_all, v_all = cache.get(layer=0)
mem_bytes = cache.memory_bytes(dtype_bytes=2)

print(f"\n  Cached {seq_len_result} tokens (expected: 5)")
if k_all is not None:
    print(f"  K matrix shape: {k_all.shape}  (expected: (5, {HIDDEN}))")
    print(f"  V matrix shape: {v_all.shape}  (expected: (5, {HIDDEN}))")
if mem_bytes is not None:
    print(f"  Memory used: {mem_bytes} bytes  (expected: {2 * LAYERS * 5 * HIDDEN * 2})")

# =============================================================================
# EXERCISE 3: FLOPs Comparison
#
# Count the total floating point operations for generating N tokens
# with and without KV cache.
#
# Attention FLOP counts (simplified):
#   Q @ K.T  = seq_len^2 * head_dim  FLOPs  (attention scores)
#   softmax  = seq_len^2             FLOPs  (approximate)
#   scores @ V = seq_len^2 * head_dim FLOPs  (weighted sum)
#   K/V projection = seq_len * head_dim^2 FLOPs each
# =============================================================================

print("\n--- Exercise 3: FLOPs Comparison ---")

def flops_naive(total_tokens, head_dim, num_heads, num_layers):
    """
    Calculate total FLOPs for generating `total_tokens` WITHOUT KV cache.

    At each step i (from 1 to total_tokens):
      - Process all i tokens from scratch
      - Compute Q, K, V for all i tokens: 3 * i * head_dim^2 per head
      - Attention (Q@K, softmax, @V): 2 * i^2 * head_dim per head

    Parameters:
        total_tokens: int -- number of tokens to generate
        head_dim:     int -- dimension per head
        num_heads:    int -- number of heads
        num_layers:   int -- number of layers

    Returns:
        int -- total FLOPs
    """
    total_flops = 0

    for step in range(1, total_tokens + 1):
        seq_len = step   # at step i, we process i tokens

        # TODO: FLOPs for QKV projections (3 projections x seq_len tokens x head_dim^2)
        qkv_flops = None   # TODO: 3 * seq_len * (head_dim ** 2) * num_heads

        # TODO: FLOPs for attention (Q@K.T and weighted @V, both are seq_len^2 * head_dim)
        attn_flops = None  # TODO: 2 * (seq_len ** 2) * head_dim * num_heads

        # TODO: Add layer multiplier and accumulate
        total_flops += None   # TODO: (qkv_flops + attn_flops) * num_layers

    return total_flops


def flops_cached(total_tokens, head_dim, num_heads, num_layers):
    """
    Calculate total FLOPs for generating `total_tokens` WITH KV cache.

    At each step i:
      - Only compute Q, K, V for the NEW token (1 token, not seq_len)
      - Attention still looks at i tokens (must read full KV cache)
      - But K/V projection is only for 1 token

    Parameters:
        total_tokens: int -- number of tokens to generate
        head_dim:     int -- dimension per head
        num_heads:    int -- number of heads
        num_layers:   int -- number of layers

    Returns:
        int -- total FLOPs
    """
    total_flops = 0

    for step in range(1, total_tokens + 1):
        seq_len = step   # attention still reads all seq_len tokens from cache

        # TODO: FLOPs for QKV projections — only for 1 NEW token (not seq_len)
        qkv_flops = None   # TODO: 3 * 1 * (head_dim ** 2) * num_heads

        # TODO: FLOPs for attention — Q(1 token) attends to K(seq_len), V(seq_len)
        # Q@K.T: 1 x seq_len x head_dim  (smaller than seq_len^2 x head_dim)
        # @V:    1 x seq_len x head_dim
        attn_flops = None  # TODO: 2 * 1 * seq_len * head_dim * num_heads

        total_flops += None   # TODO: (qkv_flops + attn_flops) * num_layers

    return total_flops


# Compare FLOPs for different sequence lengths
HEAD_DIM  = 128
NUM_HEADS = 32
NUM_LAYERS = 32

print(f"\n  Config: {NUM_LAYERS} layers, {NUM_HEADS} heads, head_dim={HEAD_DIM}")
print(f"\n  {'Tokens':>8} | {'Naive FLOPs':>15} | {'Cached FLOPs':>15} | {'Speedup':>10}")
print("  " + "-" * 55)

for n_tokens in [10, 50, 200, 1000]:
    naive  = flops_naive(n_tokens,  HEAD_DIM, NUM_HEADS, NUM_LAYERS)
    cached = flops_cached(n_tokens, HEAD_DIM, NUM_HEADS, NUM_LAYERS)
    if naive and cached:
        speedup = naive / cached
        print(f"  {n_tokens:>8,} | {naive:>15,.0f} | {cached:>15,.0f} | {speedup:>9.1f}x")
    else:
        print(f"  {n_tokens:>8,} | {'TODO':>15} | {'TODO':>15} | {'TODO':>10}")

# =============================================================================
# EXERCISE 4: Paged Attention Concept
#
# Simulate how vLLM's paged attention manages the KV cache.
# Instead of one giant pre-allocated array, use fixed-size "pages".
# =============================================================================

print("\n--- Exercise 4: Paged KV Cache ---")

class PagedKVCache:
    """
    Simplified paged KV cache.

    Memory is divided into fixed-size pages.
    Pages are allocated on demand as tokens are generated.
    This avoids the waste of pre-allocating for max_seq_len.

    Think of it like how an OS manages virtual memory:
      - RAM is divided into fixed pages
      - Process gets pages as it needs them
      - No need to pre-allocate a giant contiguous block
    """

    def __init__(self, page_size, hidden_dim):
        """
        Parameters:
            page_size:   int -- how many tokens fit in one page
            hidden_dim:  int -- size of each K or V vector
        """
        self.page_size = page_size
        self.hidden_dim = hidden_dim
        self.pages = []          # list of pages; each page is a (page_size, hidden_dim) array
        self.current_page_idx = 0   # which slot in current page we're writing to

    def _ensure_page(self):
        """Allocate a new page if current page is full or no pages exist."""
        # TODO: If no pages exist OR current page is full, add a new page
        # A new page is: np.zeros((self.page_size, self.hidden_dim))
        # Reset current_page_idx to 0 when adding a new page
        pass   # TODO

    def add_token(self, k_vector, v_vector):
        """
        Add K and V vectors for a new token.

        Parameters:
            k_vector: (hidden_dim,) numpy array
            v_vector: (hidden_dim,) numpy array
        """
        # TODO: Ensure there is space, write K (we simplify: only store K here)
        # 1. Call _ensure_page()
        # 2. Write k_vector into self.pages[-1][self.current_page_idx]
        # 3. Increment self.current_page_idx
        pass   # TODO

    def num_pages_used(self):
        """Return number of pages currently allocated."""
        # TODO: return len(self.pages)
        return None   # TODO

    def num_tokens_stored(self):
        """Return total number of tokens stored across all pages."""
        # TODO: (num_full_pages * page_size) + tokens_in_current_page
        # Hint: (len(self.pages) - 1) * self.page_size + self.current_page_idx
        # Edge case: if no pages, return 0
        return None   # TODO


# Test paged cache
PAGE_SIZE  = 4     # 4 tokens per page (tiny for demo)
HIDDEN_DIM = 16

paged_cache = PagedKVCache(page_size=PAGE_SIZE, hidden_dim=HIDDEN_DIM)

# Add 10 tokens (should use 3 pages: [0-3], [4-7], [8-9] partial)
for i in range(10):
    k = np.random.randn(HIDDEN_DIM)
    v = np.random.randn(HIDDEN_DIM)
    paged_cache.add_token(k, v)

pages_used = paged_cache.num_pages_used()
tokens_stored = paged_cache.num_tokens_stored()

print(f"\n  Stored 10 tokens with page_size={PAGE_SIZE}")
print(f"  Pages allocated: {pages_used}  (expected: 3)")
print(f"  Tokens stored:   {tokens_stored}  (expected: 10)")
print(f"\n  Key insight: only 3 pages allocated, not a pre-allocated array")
print(f"  for max_seq_len. If we only generated 3 tokens, only 1 page.")

# =============================================================================
# SCROLL DOWN FOR HINTS AND SOLUTIONS
# =============================================================================

print("\n" + "=" * 60)
print("Done! Check your output against the expected values above.")
print("Scroll down for hints and full solutions.")
print("=" * 60)


# =============================================================================
# HINTS
# =============================================================================
#
# Exercise 1 — kv_cache_memory_gb:
#   dtype_bytes = {"fp16": 2, "int8": 1}
#   total_bytes = 2 * num_layers * num_heads * seq_len * head_dim * dtype_bytes[dtype]
#   return total_bytes / (1024 ** 3)
#
# Exercise 2 — KVCache:
#   __init__:  self._k_cache = [[] for _ in range(num_layers)]
#              self._v_cache = [[] for _ in range(num_layers)]
#   update:    self._k_cache[layer].append(k_vector)
#              self._v_cache[layer].append(v_vector)
#   get:       return np.array(self._k_cache[layer]), np.array(self._v_cache[layer])
#   seq_len:   return len(self._k_cache[0])
#   memory:    return 2 * self.num_layers * self.seq_len() * self.hidden_dim * dtype_bytes
#
# Exercise 3 — FLOPs:
#   naive    qkv_flops:  3 * seq_len * (head_dim ** 2) * num_heads
#   naive    attn_flops: 2 * (seq_len ** 2) * head_dim * num_heads
#   cached   qkv_flops:  3 * 1 * (head_dim ** 2) * num_heads          (only 1 token!)
#   cached   attn_flops: 2 * 1 * seq_len * head_dim * num_heads
#   both:    total_flops += (qkv_flops + attn_flops) * num_layers
#
# Exercise 4 — PagedKVCache:
#   _ensure_page:
#     if not self.pages or self.current_page_idx >= self.page_size:
#         self.pages.append(np.zeros((self.page_size, self.hidden_dim)))
#         self.current_page_idx = 0
#   add_token:
#     self._ensure_page()
#     self.pages[-1][self.current_page_idx] = k_vector
#     self.current_page_idx += 1
#   num_pages_used: return len(self.pages)
#   num_tokens_stored:
#     if not self.pages: return 0
#     return (len(self.pages) - 1) * self.page_size + self.current_page_idx
#
# =============================================================================


# =============================================================================
# SOLUTIONS (uncomment to run)
# =============================================================================

# --- Solution 1 ---
# def kv_cache_memory_gb(num_layers, num_heads, seq_len, head_dim, dtype="fp16"):
#     dtype_bytes = {"fp16": 2, "int8": 1}
#     total_bytes = 2 * num_layers * num_heads * seq_len * head_dim * dtype_bytes[dtype]
#     return total_bytes / (1024 ** 3)

# --- Solution 2 ---
# class KVCache:
#     def __init__(self, num_layers, head_dim, num_heads):
#         self.num_layers = num_layers
#         self.head_dim = head_dim
#         self.num_heads = num_heads
#         self.hidden_dim = head_dim * num_heads
#         self._k_cache = [[] for _ in range(num_layers)]
#         self._v_cache = [[] for _ in range(num_layers)]
#     def update(self, layer, k_vector, v_vector):
#         self._k_cache[layer].append(k_vector)
#         self._v_cache[layer].append(v_vector)
#     def get(self, layer):
#         return np.array(self._k_cache[layer]), np.array(self._v_cache[layer])
#     def seq_len(self):
#         return len(self._k_cache[0])
#     def memory_bytes(self, dtype_bytes=2):
#         return 2 * self.num_layers * self.seq_len() * self.hidden_dim * dtype_bytes

# --- Solution 3 ---
# def flops_naive(total_tokens, head_dim, num_heads, num_layers):
#     total_flops = 0
#     for step in range(1, total_tokens + 1):
#         seq_len = step
#         qkv_flops  = 3 * seq_len * (head_dim ** 2) * num_heads
#         attn_flops = 2 * (seq_len ** 2) * head_dim * num_heads
#         total_flops += (qkv_flops + attn_flops) * num_layers
#     return total_flops
#
# def flops_cached(total_tokens, head_dim, num_heads, num_layers):
#     total_flops = 0
#     for step in range(1, total_tokens + 1):
#         seq_len = step
#         qkv_flops  = 3 * 1 * (head_dim ** 2) * num_heads
#         attn_flops = 2 * 1 * seq_len * head_dim * num_heads
#         total_flops += (qkv_flops + attn_flops) * num_layers
#     return total_flops

# --- Solution 4 ---
# class PagedKVCache:
#     def __init__(self, page_size, hidden_dim):
#         self.page_size = page_size
#         self.hidden_dim = hidden_dim
#         self.pages = []
#         self.current_page_idx = 0
#     def _ensure_page(self):
#         if not self.pages or self.current_page_idx >= self.page_size:
#             self.pages.append(np.zeros((self.page_size, self.hidden_dim)))
#             self.current_page_idx = 0
#     def add_token(self, k_vector, v_vector):
#         self._ensure_page()
#         self.pages[-1][self.current_page_idx] = k_vector
#         self.current_page_idx += 1
#     def num_pages_used(self):
#         return len(self.pages)
#     def num_tokens_stored(self):
#         if not self.pages: return 0
#         return (len(self.pages) - 1) * self.page_size + self.current_page_idx

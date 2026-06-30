# Lesson 4: KV Cache Management

## What Problem Does This Solve?

In Module 14 Lesson 6, you learned the basic concept of the KV cache.
This lesson goes deeper: how it is managed, optimized, and how memory is controlled
in production systems.

First, recall why the KV cache exists.

When generating token number 100, the model must compute attention over
all 99 previous tokens. For each past token, it needs the Key and Value vectors.

```
WITHOUT KV CACHE:

  Generating token 100:
    Recompute K and V for tokens 1, 2, 3, ..., 99 from scratch.
    Then compute K, V for token 100.
    Attention over all 100 tokens.

  Cost per new token: O(N) recomputation of past K, V.
  Total cost for N tokens: O(N^2) — slow.


WITH KV CACHE:

  Generating token 100:
    Load K[1..99] and V[1..99] from cache. (Already stored.)
    Compute K[100], V[100] for new token only.
    Attention over all 100 tokens using cached K, V.

  Cost per new token: O(1) new K, V computation + O(N) attention.
  Much faster — the expensive recomputation is eliminated.
```

The KV cache trades memory for speed.
This lesson is about managing that memory wisely.

---

## The Basic KV Cache Structure

```
+=========================================================================+
|  KV CACHE STRUCTURE                                                     |
+=========================================================================+
|                                                                         |
|  For each transformer layer:                                            |
|    K_cache: tensor of shape [batch, n_kv_heads, seq_so_far, head_dim]  |
|    V_cache: tensor of shape [batch, n_kv_heads, seq_so_far, head_dim]  |
|                                                                         |
|  At each generation step t:                                             |
|    1. Compute K_t, V_t for the new token                                |
|    2. Append K_t to K_cache along the seq dimension                    |
|    3. Append V_t to V_cache along the seq dimension                    |
|    4. Run attention using full K_cache, V_cache                        |
|    5. Keep K_cache, V_cache for next step                               |
|                                                                         |
|  In PyTorch, this is called past_key_values:                            |
|    tuple of (K_cache, V_cache) for each layer                          |
|                                                                         |
+=========================================================================+
```

---

## Memory Formula (Exact)

You saw this in Module 14. Now you know all the terms precisely:

```
+------------------------------------------------------------------+
|  KV CACHE MEMORY FORMULA                                        |
+------------------------------------------------------------------+
|                                                                  |
|  total_bytes = 2 * L * H_kv * S * D * B                         |
|                                                                  |
|  Where:                                                          |
|    2     = K cache + V cache                                    |
|    L     = number of transformer layers                         |
|    H_kv  = number of KV heads (not Q heads — see Lesson 2 GQA) |
|    S     = sequence length (tokens generated so far)           |
|    D     = head dimension                                       |
|    B     = bytes per value (bf16 = 2, fp32 = 4, INT8 = 1)      |
|                                                                  |
|  EXAMPLE: Qwen3.5-7B, bf16, 32K context                         |
|    2 * 32 * 8 * 32768 * 128 * 2 = 4,294,967,296 bytes = 4.3 GB |
|                                                                  |
|  EXAMPLE: Qwen3.5-7B, INT8 KV cache, 32K context               |
|    2 * 32 * 8 * 32768 * 128 * 1 = 2,147,483,648 bytes = 2.1 GB |
|                                                                  |
|  GQA benefit: H_kv = 8 (not 32).                                |
|  Without GQA (MHA, H_kv = 32):  4.3 GB * 4 = 17.2 GB           |
|                                                                  |
+------------------------------------------------------------------+
```

GQA (Lesson 2) and INT8 KV quantization are the two biggest levers
to reduce KV cache memory.

---

## Four KV Cache Management Strategies

### Strategy 1: Full Cache (Simplest)

Keep K and V for every past token. The cache grows until it hits memory.

```
Memory usage over time:

  Token 1:    KV cache = small
  Token 100:  KV cache = medium
  Token 1000: KV cache = large
  Token 8192: KV cache = at memory limit → ERROR or evict

  Pro: Perfect recall. Simplest to implement.
  Con: Memory grows unboundedly. Crashes at long contexts.
```

### Strategy 2: Sliding Window Attention

Only keep K and V for the last W tokens. Older tokens are evicted.

```
+------------------------------------------------------------------+
|  SLIDING WINDOW CACHE                                           |
+------------------------------------------------------------------+
|                                                                  |
|  Window size W = 4096 tokens.                                   |
|                                                                  |
|  At token 5000:                                                  |
|    Keep tokens 904..5000 in cache (last 4096)                   |
|    Evict tokens 1..903 (too old)                                |
|                                                                  |
|  Memory: FIXED at W * H_kv * D * 2 * L * B                     |
|          Regardless of total sequence length.                   |
|                                                                  |
|  Window:                                                         |
|  [tok 904] [tok 905] ... [tok 5000] [NEW tok 5001]              |
|         oldest                newest                            |
|  [----------- W = 4096 tokens kept -----------]                 |
|                                                                  |
|  Pro: Fixed memory. Handles arbitrarily long generation.        |
|  Con: Cannot attend to tokens older than W. Long-range info     |
|        from beginning of document is lost.                      |
|                                                                  |
+------------------------------------------------------------------+
```

### Strategy 3: Prefix Caching

When many requests share the same prefix (e.g., a system prompt), cache it once.

```
+------------------------------------------------------------------+
|  PREFIX CACHING                                                 |
+------------------------------------------------------------------+
|                                                                  |
|  System prompt: "You are a helpful assistant. Today is..."      |
|  (same for every request)                                       |
|                                                                  |
|  WITHOUT prefix cache:                                          |
|    Request 1: compute KV for system prompt + user query 1      |
|    Request 2: compute KV for system prompt + user query 2      |
|    (system prompt recomputed N times for N requests)            |
|                                                                  |
|  WITH prefix cache:                                             |
|    Start: compute KV for system prompt once, store in cache     |
|    Request 1: LOAD system prompt KV, compute only user query 1 |
|    Request 2: LOAD system prompt KV, compute only user query 2 |
|                                                                  |
|  Saves: seq_len(system_prompt) * H_kv * D * 2 * L * B per req  |
|  At 2K system prompt + 1000 req/day: massive throughput win.   |
|                                                                  |
+------------------------------------------------------------------+
```

### Strategy 4: Paged Attention (vLLM)

The standard full cache allocates one large contiguous block of memory upfront.
This wastes memory when sequences are shorter than the maximum.

Paged attention (from the vLLM paper, 2023) borrows an idea from operating systems:
virtual memory paging.

```
+------------------------------------------------------------------+
|  PAGED ATTENTION                                                |
+------------------------------------------------------------------+
|                                                                  |
|  Standard cache: allocate max_seq_len * memory at the start     |
|    Request 1 gets 8192 tokens reserved → uses only 500 → waste  |
|    Request 2 gets 8192 tokens reserved → uses 8000 → fine       |
|    Memory fragmentation: some blocks mostly empty               |
|                                                                  |
|  Paged attention: allocate fixed-size PAGES on demand            |
|    Page size = 16 tokens (configurable)                         |
|                                                                  |
|    Request 1: allocate 1 page (16 tokens)                       |
|               grows to 2 pages (32 tokens)                      |
|               grows to ... as needed                            |
|    Request 2: shares the same physical page pool                |
|                                                                  |
|  Virtual page table:                                            |
|    Request 1: [page 3, page 7, page 12]   (physical pages)     |
|    Request 2: [page 1, page 5]                                  |
|    Shared prompt: [page 0]  (reused by request 1 AND 2)         |
|                                                                  |
|  Pro: near-zero memory fragmentation.                           |
|       GPU memory utilization > 90% vs ~40% for standard cache. |
|  Con: complex implementation (page table management).           |
+------------------------------------------------------------------+
```

---

## INT8 KV Cache Quantization

The most practical memory-saving technique: store K and V in INT8 instead of bf16.

```
+------------------------------------------------------------------+
|  INT8 KV CACHE                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  Normal KV cache: store K and V as bf16 (2 bytes each)          |
|  INT8 KV cache:   quantize K and V to INT8 (1 byte each)        |
|                                                                  |
|  How quantization works:                                         |
|    For each vector k:                                           |
|      scale = max(abs(k)) / 127                                  |
|      k_int8 = round(k / scale).clip(-128, 127)                  |
|      Store: k_int8 (1 byte each) + scale (1 float)             |
|                                                                  |
|    To use: k_restored = k_int8 * scale   (dequantize on the fly)|
|                                                                  |
|  Memory impact:                                                  |
|    bf16 KV: 4.3 GB (Qwen3.5-7B, 32K context)                   |
|    INT8 KV: 2.1 GB  (50% reduction)                             |
|                                                                  |
|  Quality impact:                                                 |
|    Attention scores shift slightly due to quantization error.   |
|    In practice: < 0.5% perplexity increase on benchmarks.      |
|    Acceptable for most applications.                            |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Visual: Full Generation Loop with KV Cache

```
GENERATION LOOP: "The cat sat on the"

Step 1: Process prompt "The cat sat on the"
  Input tokens: [The, cat, sat, on, the]
  Compute K, V for ALL 5 tokens at once (parallel, fast)
  Store in KV cache:
    Layer 1: K_cache=[K1..K5], V_cache=[V1..V5]
    Layer 2: K_cache=[K1..K5], V_cache=[V1..V5]
    ...

Step 2: Generate token 6
  Input: [the]  (only the last token)
  Compute K6, V6 for this token only
  Append to cache: K_cache=[K1..K6], V_cache=[V1..V6]
  Attention: Q6 attends over K_cache, V_cache
  Output: next token logits → sample → "mat"

Step 3: Generate token 7
  Input: [mat]  (only the last token)
  Compute K7, V7
  Append to cache: K_cache=[K1..K7], V_cache=[V1..V7]
  Attention: Q7 attends over K_cache, V_cache
  Output: next token → "."

Each step: ONE new token's K, V computed. Past K, V loaded from cache.
No recomputation of past tokens.
```

---

## When to Apply Which Strategy

```
+------------------------------------------------------------------+
|  STRATEGY SELECTION GUIDE                                       |
+------------------------------------------------------------------+
|                                                                  |
|  Short generations (< 2K tokens), few users:                    |
|    Use full cache. Simple. Memory fits.                         |
|                                                                  |
|  Long generations (> 4K tokens), memory constrained:            |
|    Use sliding window + INT8 KV cache.                          |
|                                                                  |
|  Chatbot with shared system prompt, many concurrent users:      |
|    Use prefix caching. Large win for shared context.            |
|                                                                  |
|  High-throughput server, many concurrent requests:              |
|    Use paged attention (vLLM). Best GPU utilization.            |
|                                                                  |
|  Mobile / edge devices:                                         |
|    INT8 KV cache + sliding window + small n_kv_heads (GQA).    |
|                                                                  |
+------------------------------------------------------------------+
```

---

## KV Cache vs RLA: When to Use Each

After Lesson 3 (RLA) and this lesson, you may wonder: why manage a KV cache at all
when RLA replaces it with a fixed-size hidden state?

```
+------------------------------------------------------------------+
|  KV CACHE vs RLA HIDDEN STATE                                   |
+------------------------------------------------------------------+
|                                                                  |
|  KV Cache (for full softmax attention layers):                  |
|    Exact attention: can recall specific past tokens precisely   |
|    Memory grows with sequence length: O(N)                      |
|    Needed for the GQA layers in Qwen3.5                         |
|                                                                  |
|  RLA Hidden State (for recurrent linear attention layers):      |
|    Approximate attention: compresses all past into fixed matrix  |
|    Memory FIXED: O(d^2) regardless of sequence length           |
|    Replaces KV cache for the RLA layers in Qwen3.5              |
|                                                                  |
|  Qwen3.5 uses BOTH:                                             |
|    RLA layers: hidden state S — no KV cache entry               |
|    GQA layers: KV cache — grows with sequence                   |
|    Ratio: ~3 RLA layers per 1 GQA layer                        |
|    Result: most memory managed by RLA, precise recall from GQA  |
+------------------------------------------------------------------+
```

---

## C# Analogy: IMemoryCache + Connection Pooling

```csharp
// KV cache is exactly like ASP.NET Core's IMemoryCache.
//
// WITHOUT CACHE (recompute every request):
//
//   public Vector ComputeAttention(int tokenId, int[] pastTokenIds)
//   {
//       // Recompute K and V for ALL past tokens every time
//       var pastKV = pastTokenIds
//           .Select(id => ComputeKV(id))   // expensive: O(N) per step
//           .ToList();
//       return Attention(currentQuery, pastKV);
//   }
//   // Total for N tokens: O(N^2) — same as recomputing every request
//
//
// WITH KV CACHE (IMemoryCache equivalent):
//
//   private readonly Dictionary<int, (Matrix K, Matrix V)> _kvCache = new();
//
//   public Vector ComputeAttention(int tokenId)
//   {
//       // Compute K, V only for the NEW token
//       _kvCache[tokenId] = ComputeKV(tokenId);
//
//       // Retrieve all past K, V from cache
//       var allK = _kvCache.Values.Select(kv => kv.K).ToArray();
//       var allV = _kvCache.Values.Select(kv => kv.V).ToArray();
//
//       return Attention(currentQuery, allK, allV);   // O(N) attention
//   }
//   // Total for N tokens: O(N) compute, O(N) memory — much better
//
//
// Paged attention analogy:
//   Standard cache = List<T> — pre-allocates max capacity up front
//   Paged attention = LinkedList<Page<T>> — allocates one page at a time
//   Both hold the same data; paged version wastes less memory
//
// Prefix caching analogy:
//   static readonly Dictionary<string, KVCache> _prefixCache — shared across requests
//   Like a static field: computed once, reused by every instance
```

---

## Quiz Questions

**Q1**: Without a KV cache, what is the time complexity to generate N tokens total?
        a) O(N)
        b) O(N log N)
        c) O(N^2)
        d) O(N^3)

**Q2**: Qwen3.5-7B uses GQA with 8 KV heads instead of 32. At 32K context in bf16,
        this reduces KV cache from approximately:
        a) 8.5 GB to 4.3 GB (2x reduction)
        b) 17.2 GB to 4.3 GB (4x reduction)
        c) 34.4 GB to 4.3 GB (8x reduction)
        d) 4.3 GB to 1.1 GB (4x reduction)

**Q3**: Paged attention (used in vLLM) was inspired by which computer science concept?
        a) Database connection pooling
        b) CPU cache lines
        c) Operating system virtual memory paging
        d) TCP packet fragmentation

**Q4**: In Qwen3.5's hybrid architecture, which layers have a KV cache and which do not?
        a) All layers have a KV cache — RLA does not change this
        b) GQA layers have a KV cache; RLA layers use a fixed-size hidden state instead
        c) RLA layers have a KV cache; GQA layers do not need one
        d) No layers have a KV cache — RLA eliminates it entirely

*(Answers: Q1=c, Q2=b, Q3=c, Q4=b)*

---

## Key Takeaways

1. KV cache stores K and V for past tokens so they are not recomputed each step
2. Without cache: O(N²) total generation cost. With cache: O(N) — one K/V compute per token
3. Memory formula: `2 * L * H_kv * S * D * B` — grows with sequence length S
4. GQA (Lesson 2) reduces H_kv from 32 to 8 — 4x KV cache size reduction
5. Sliding window: keep only last W tokens in cache — fixed memory for infinite sequences
6. Prefix caching: compute shared system prompt KV once — reuse across all requests
7. Paged attention (vLLM): allocate cache in pages — 90%+ GPU memory utilization vs ~40%
8. INT8 KV cache: quantize K and V to 1 byte — 50% memory reduction, minimal quality loss
9. In Qwen3.5: GQA layers use KV cache; RLA layers bypass it with a fixed hidden state

---

*Next: Lesson 5 — Full Qwen3.5 Decoder Assembly (putting RoPE + GQA + RLA + KV Cache together)*

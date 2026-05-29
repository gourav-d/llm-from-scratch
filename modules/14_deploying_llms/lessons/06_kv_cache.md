# Lesson 06: KV Cache & Inference Optimization

## Glossary (Read This First!)

Every term used in this lesson is defined here.
Do not skip this section.

| Term | Plain English Definition |
|------|--------------------------|
| **KV cache** | A memory store that saves the Key and Value matrices computed for each past token. Prevents re-computing them on every new token. |
| **Key (K)** | One of three matrices computed from each token during attention. Represents "what this token contains". |
| **Value (V)** | One of three matrices computed from each token during attention. Represents "the information to pass forward". |
| **Query (Q)** | One of three matrices computed from each token during attention. Represents "what this token is looking for". |
| **Autoregressive generation** | How LLMs generate text: one token at a time, each token conditioned on all previous tokens. |
| **FLOP** | Floating Point Operation. A single arithmetic step (+, -, *, /). Used to measure compute cost. |
| **Memory bandwidth** | How fast data moves between RAM and CPU/GPU. Often the real bottleneck, not raw compute. |
| **Paged attention** | A technique (used in vLLM) that stores the KV cache in fixed-size pages, like OS virtual memory. Handles variable-length sequences without fragmentation. |
| **INT8 KV cache** | Quantizing the KV cache values from fp16 to INT8. Cuts KV cache memory in half with minimal quality loss. |
| **Prefill phase** | Processing all prompt tokens at once (parallel). Fast. Builds the initial KV cache. |
| **Decode phase** | Generating new tokens one at a time (sequential). Slower. Each step extends the KV cache by one row. |
| **Context window** | Maximum number of tokens the model can hold in its KV cache at once. 4K, 8K, 32K, 128K are common. |
| **Sequence length** | How many tokens are in the current conversation (prompt + generated tokens so far). |

---

## Part 1: The Problem — Without KV Cache

### How Attention Works During Generation

When generating token #100, the model must attend to all 99 previous tokens.
Without any optimization, it would compute Key and Value matrices for ALL 100 tokens
every single step.

```
+------------------------------------------------------------------+
|  NAIVE GENERATION (no KV cache)                                  |
+------------------------------------------------------------------+
|                                                                  |
|  Step 1: Generate token 1                                        |
|    Input: [token_0]                                              |
|    Compute Q, K, V for: [token_0]                     (1 token)  |
|                                                                  |
|  Step 2: Generate token 2                                        |
|    Input: [token_0, token_1]                                     |
|    Compute Q, K, V for: [token_0, token_1]            (2 tokens) |
|    ↑ token_0's K and V computed AGAIN — wasted!                  |
|                                                                  |
|  Step 3: Generate token 3                                        |
|    Input: [token_0, token_1, token_2]                            |
|    Compute Q, K, V for all 3 tokens                   (3 tokens) |
|    ↑ token_0 and token_1 computed AGAIN — wasted!                |
|                                                                  |
|  Step N: Generate token N                                        |
|    Compute Q, K, V for all N tokens                   (N tokens) |
|                                                                  |
|  TOTAL WORK: 1 + 2 + 3 + ... + N = N*(N+1)/2 = O(N²)            |
|                                                                  |
+------------------------------------------------------------------+
```

This is quadratic growth. Generate 1000 tokens → 500,000 computations.
Generate 4000 tokens → 8,000,000 computations. Very slow.

---

## Part 2: The Solution — KV Cache

### Core Insight

Keys and Values for past tokens NEVER CHANGE.

Once you compute K and V for token #5, those values are the same forever.
Only the Query matrix changes each step (it comes from the NEW token).

So: compute K and V once per token, save them, reuse forever.

```
+------------------------------------------------------------------+
|  GENERATION WITH KV CACHE                                        |
+------------------------------------------------------------------+
|                                                                  |
|  Step 1: Generate token 1                                        |
|    Compute Q, K, V for: [token_0]                                |
|    Save K_0, V_0 to cache                                        |
|    Cache: [ K_0 | V_0 ]                               (1 entry) |
|                                                                  |
|  Step 2: Generate token 2                                        |
|    Compute Q, K, V for: [token_1] ONLY (new token!)              |
|    Retrieve K_0, V_0 from cache                                  |
|    Cache: [ K_0 | V_0 | K_1 | V_1 ]                  (2 entries)|
|    Attention: Q_1 attends to [K_0, K_1], gets [V_0, V_1]        |
|                                                                  |
|  Step N: Generate token N                                        |
|    Compute Q, K, V for: [token_N] ONLY (1 token always!)        |
|    Cache: [ K_0...K_{N-1} | V_0...V_{N-1} ]          (N entries)|
|                                                                  |
|  TOTAL WORK: N steps x 1 computation = O(N)                      |
|                                                                  |
+------------------------------------------------------------------+

  Without KV cache: O(N²)   → 4000 tokens = 16,000,000 ops
  With    KV cache: O(N)    → 4000 tokens =      4,000 ops
  Speedup: ~N times faster for long sequences
```

C# analogy:
```csharp
// Without cache: recompute every time (like no memoization)
int Fibonacci(int n) {
    if (n <= 1) return n;
    return Fibonacci(n-1) + Fibonacci(n-2);  // recomputes everything
}

// With cache: save results (like Dictionary memoization)
Dictionary<int, long> _cache = new();
long FibonacciCached(int n) {
    if (_cache.ContainsKey(n)) return _cache[n];  // reuse saved result
    var result = FibonacciCached(n-1) + FibonacciCached(n-2);
    _cache[n] = result;                            // save for future
    return result;
}

// KV cache is EXACTLY this: save computed K and V values,
// reuse them instead of recomputing on every step.
```

---

## Part 3: KV Cache Memory Cost

### The Formula

The KV cache grows with every new token generated. Here is how to calculate its size:

```
KV Cache Memory =
    2             (K and V — two matrices per layer)
  × num_layers    (one set of K, V per transformer layer)
  × num_heads     (multi-head attention — each head has its own K, V)
  × seq_len       (number of tokens so far — grows as you generate)
  × head_dim      (dimension of each head = hidden_dim / num_heads)
  × bytes_per_value (2 for fp16, 1 for INT8)
```

### Real Model Examples

```
+------------------------------------------------------------------+
|  KV CACHE SIZE AT 4096 TOKEN CONTEXT WINDOW (fp16)               |
+------------------------------------------------------------------+
|                                                                  |
|  Model       | Layers | Heads | head_dim | KV cache @ 4K tokens  |
|  ------------|--------|-------|----------|-----------------------  |
|  GPT-2 (S)  |    12  |   12  |    64    |   0.19 GB              |
|  LLaMA-7B   |    32  |   32  |   128    |   2.0  GB              |
|  LLaMA-13B  |    40  |   40  |   128    |   3.1  GB              |
|  LLaMA-70B  |    80  |   64  |   128    |  20.0  GB              |
|                                                                  |
|  Formula used: 2 × layers × heads × 4096 × head_dim × 2 bytes   |
|                                                                  |
+------------------------------------------------------------------+
```

This is SEPARATE from the model weights. LLaMA-7B (INT4) = 3.5 GB weights
PLUS 2 GB KV cache = 5.5 GB minimum for a single 4K-context conversation.

---

## Part 4: Prefill vs Decode Phases

LLM inference has two distinct phases with very different performance profiles:

```
+------------------------------------------------------------------+
|  TWO PHASES OF LLM INFERENCE                                     |
+------------------------------------------------------------------+
|                                                                  |
|  PHASE 1: PREFILL (process the prompt)                           |
|  ─────────────────────────────────────                           |
|  Input:   "What is the capital of France?"  (7 tokens)           |
|  Action:  Process all 7 tokens IN PARALLEL                       |
|  Output:  KV cache populated for all 7 tokens                    |
|  Speed:   FAST — GPU handles all tokens simultaneously           |
|  Bottleneck: Compute (lots of matrix multiplications)            |
|                                                                  |
|  PHASE 2: DECODE (generate the answer)                           |
|  ─────────────────────────────────────                           |
|  Input:   One new token at a time + entire KV cache              |
|  Action:  Generate "Paris", then ".", then "<EOS>"               |
|  Output:  One token per step                                     |
|  Speed:   SLOW — must be sequential, cannot be parallelized      |
|  Bottleneck: Memory bandwidth (reading the full KV cache)        |
|                                                                  |
+------------------------------------------------------------------+
```

Key insight: The decode phase is bottlenecked by **memory bandwidth**, not compute.
The GPU must read the entire KV cache from memory on every single step.
This is why:
- Larger context windows slow down generation
- INT8 KV cache helps (smaller cache = faster read)
- Better GPUs with higher memory bandwidth = faster generation

---

## Part 5: Paged Attention (vLLM)

### The Problem with Standard KV Cache

Standard KV cache allocates a fixed block of memory per request at the start.
If you reserve space for 4096 tokens but only generate 100, the rest is wasted.

```
STANDARD KV CACHE (wasteful):
  Reserved: [TOKEN_0 | TOKEN_1 | ... | TOKEN_100 | EMPTY | EMPTY | ... | EMPTY]
                                                   ↑ 3996 slots wasted
```

### Paged Attention Solution

vLLM borrows the idea of virtual memory from operating systems.
Memory is divided into fixed-size pages. Pages are allocated on demand.

```
PAGED ATTENTION (vLLM):
  Page 1: [TOKEN_0  | TOKEN_1  | ... | TOKEN_15 ]  (full, 16 slots)
  Page 2: [TOKEN_16 | TOKEN_17 | ... | TOKEN_31 ]  (full, 16 slots)
  Page 3: [TOKEN_32 | TOKEN_33 | ... | TOKEN_36 ]  (partial, 5 used)
  Page 4: [empty]  ← not allocated yet

  Benefits:
  - No wasted memory for short sequences
  - Multiple requests can share pages (e.g., same system prompt)
  - Better GPU memory utilization → more requests in parallel
```

C# analogy:
```csharp
// Standard KV cache: pre-allocated fixed array
var kvCache = new float[maxTokens, hiddenDim];   // wasteful if sequence is short

// Paged attention: List<Page> where each Page is a fixed block
var pages = new List<KVPage>();          // allocate pages on demand
pages.Add(new KVPage(capacity: 16));    // only allocate what you need
```

---

## Part 6: INT8 KV Cache Quantization

The KV cache is stored in fp16 by default (2 bytes per value).
Quantizing to INT8 (1 byte per value) cuts KV cache memory in half.

```
+------------------------------------------------------------------+
|  KV CACHE QUANTIZATION COMPARISON                                |
+------------------------------------------------------------------+
|                                                                  |
|  Format | Bytes/value | LLaMA-7B @ 4K ctx | Quality loss         |
|  -------|-------------|-------------------|---------------------  |
|  fp32   |      4      |      4.0 GB       | None (reference)     |
|  fp16   |      2      |      2.0 GB       | Tiny (~0.1% diff)    |
|  INT8   |      1      |      1.0 GB       | Small (~0.5% diff)   |
|  INT4   |     0.5     |      0.5 GB       | Noticeable           |
|                                                                  |
|  INT8 KV cache: 2x less memory, almost no quality loss           |
|  This is DIFFERENT from weight quantization (model weights stay) |
|                                                                  |
+------------------------------------------------------------------+
```

Why KV cache quantization works well:
- KV values tend to be small floats in a narrow range
- The scale factor can be computed per-head (fine-grained)
- Quality loss is much smaller than quantizing model weights

---

## Part 7: Summary

```
+------------------------------------------------------------------+
|  KV CACHE — KEY TAKEAWAYS                                        |
+------------------------------------------------------------------+
|                                                                  |
|  1. WHAT:  Store K and V matrices for past tokens                |
|            Reuse them instead of recomputing                     |
|                                                                  |
|  2. WHY:   O(N²) → O(N) complexity for generation               |
|            Makes long contexts practical                         |
|                                                                  |
|  3. COST:  2 × layers × heads × seq_len × head_dim × dtype_bytes |
|            Grows with every generated token                      |
|            Separate from model weight memory                     |
|                                                                  |
|  4. PHASES:                                                      |
|            Prefill = process prompt in parallel (compute-bound)  |
|            Decode  = generate token-by-token (bandwidth-bound)   |
|                                                                  |
|  5. PAGED ATTENTION: allocate KV cache in pages on demand        |
|            Avoids waste, enables request sharing, used in vLLM   |
|                                                                  |
|  6. INT8 KV CACHE: quantize cached values from fp16 to INT8      |
|            2x memory saving, minimal quality loss                |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Quiz

**Q1.** Without KV cache, generating 1000 tokens requires O(___) operations.

**Q2.** True or False: The Query matrix is cached, but Key and Value are recomputed each step.

**Q3.** What are the two phases of LLM inference called?

**Q4.** A model has 32 layers, 32 heads, head_dim=128, fp16 (2 bytes). Sequence length is 2048.
What is the KV cache size in GB?

**Q5.** What OS concept does paged attention borrow from?

**Q6.** INT8 KV cache cuts memory by ___x compared to fp16.

---

## Answers

**A1.** O(N²) — each step recomputes all previous tokens.

**A2.** False — Keys and Values are CACHED. The Query is the only thing computed fresh each step.

**A3.** Prefill (process prompt in parallel) and Decode (generate tokens one at a time).

**A4.** `2 × 32 × 32 × 2048 × 128 × 2 bytes = 2 × 32 × 32 × 2048 × 128 × 2 = 536,870,912 bytes ≈ 0.5 GB`

**A5.** Virtual memory — OS manages memory in fixed pages, allocated on demand.

**A6.** 2x — INT8 = 1 byte vs fp16 = 2 bytes per value.

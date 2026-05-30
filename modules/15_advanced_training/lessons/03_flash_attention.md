# Lesson 3: Flash Attention

## The Attention Memory Problem

Standard attention has a memory problem that limits context length.

Let's calculate how much memory standard attention needs:

```
Transformer model: 32 layers, 32 attention heads
Sequence length N = 32,768 tokens (32K context)

Each attention layer computes: Q × K^T
  Shape of Q:    [N, d_k] = [32768, 128]
  Shape of K^T:  [d_k, N] = [128, 32768]
  Shape of result: [N, N]  = [32768, 32768]

Memory for attention matrix PER LAYER:
  32768 × 32768 × 2 bytes (bf16) = 2.1 GB per layer

For 32 layers:
  2.1 GB × 32 = 67 GB just for attention matrices

This is IMPOSSIBLE on any consumer GPU.
Even a 80GB A100 would be nearly full from attention alone.
```

This is why GPT-3 had only a 4K context window.
Not a model design choice — a memory constraint.

Flash Attention solves this.

---

## Understanding GPU Memory Hierarchy

To understand Flash Attention, you first need to understand how GPU memory works.

```
+=====================================================================+
|  GPU MEMORY HIERARCHY                                               |
+=====================================================================+
|                                                                     |
|  SRAM (on-chip cache)                                               |
|    Size:     ~20 MB per streaming multiprocessor (SM)              |
|    Speed:    ~19 TB/s bandwidth                                     |
|    Cost:     very expensive                                         |
|    Access:   ~1-5 cycles                                            |
|                                                                     |
|  HBM (High Bandwidth Memory — the "GPU RAM" you see advertised)     |
|    Size:     40-80 GB (A100), 24 GB (RTX 4090)                     |
|    Speed:    ~2 TB/s bandwidth                                      |
|    Cost:     cheap per GB                                           |
|    Access:   ~100+ cycles                                           |
|                                                                     |
|  SRAM is 10x faster than HBM.                                       |
|  SRAM is 1000x smaller than HBM.                                    |
|                                                                     |
|  The goal: do as much work as possible in SRAM.                     |
|  The standard attention problem: stores N×N matrix in HBM.         |
|  Flash Attention solution: never store N×N in HBM.                 |
+=====================================================================+
```

---

## Standard Attention: Step by Step

Before Flash Attention, here is what happens in a standard attention computation:

```
Input: Q [N, d], K [N, d], V [N, d]  (stored in HBM)

Step 1: Load Q, K from HBM → SRAM
Step 2: Compute S = Q × K^T → shape [N, N]   ← THIS IS THE PROBLEM
Step 3: Write S to HBM                        ← expensive write
Step 4: Load S from HBM → SRAM
Step 5: Compute P = softmax(S)                ← row-wise softmax
Step 6: Write P to HBM                        ← expensive write
Step 7: Load P, V from HBM → SRAM
Step 8: Compute output = P × V
Step 9: Write output to HBM

HBM reads/writes: MANY passes over N×N matrix
Memory: O(N²)
```

The problem is that the N×N attention matrix (Step 2) is too large to fit in SRAM.
So the algorithm must constantly read and write to slow HBM memory.

---

## Flash Attention: The Key Idea

**Flash Attention** was introduced by Tri Dao et al. at Stanford in 2022.

The key insight: **you do not need to store the full N×N attention matrix**.

You can compute the final output using **tiles** (blocks) of Q, K, V,
and track the softmax normalization factor across tiles using a running statistic.

```
+------------------------------------------------------------------+
|  FLASH ATTENTION CORE IDEA                                       |
+------------------------------------------------------------------+
|                                                                  |
|  Instead of:                                                     |
|    Compute full N×N attention matrix → write to HBM             |
|                                                                  |
|  Do this:                                                        |
|    Split Q, K, V into blocks that FIT in SRAM                   |
|    Process one block at a time (stays in fast SRAM)             |
|    Update output accumulator and softmax normalizer              |
|    Never write the N×N matrix to HBM                            |
|                                                                  |
|  Memory usage:  O(N)   instead of O(N²)                          |
|  HBM accesses: O(N²/M) instead of O(N²)                         |
|    where M = SRAM size                                           |
|                                                                  |
|  For N=32768, M=20MB: HBM access reduced by 1000x               |
+------------------------------------------------------------------+
```

---

## The Tiling Visualization

```
STANDARD ATTENTION:
                     K matrix (N × d)
                  +------------------+
                  |                  |
  Q matrix      × |                  | = attention [N × N]   ← huge
  (N × d)         |                  |             in HBM
                  +------------------+

FLASH ATTENTION:
                     K matrix
              [block1] [block2] [block3]
               K_1      K_2      K_3
  Q [block1]   q1×K1    q1×K2    q1×K3    ← computed one at a time
  Q [block2]   q2×K1    q2×K2    q2×K3    ← each block fits in SRAM
  Q [block3]   q3×K1    q3×K2    q3×K3    ← output accumulated in SRAM

Each small tile [qi × Kj] fits in SRAM → no HBM write needed.
The running softmax normalizer is tracked across tiles.
Final output is assembled without ever materializing full N×N.
```

---

## The Softmax Trick

The hard part of tiling attention is softmax.
Softmax over a row requires knowing ALL values in that row:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
```

You cannot compute this tile by tile because you don't know the full sum yet.

Flash Attention uses a **running max and running sum** trick:

```
Standard softmax is numerically unstable. The stable version uses:
  safe_softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)) for all j)

Flash Attention tracks 3 scalars per row:
  m  = running maximum seen so far
  l  = running sum of exp(x - m) seen so far
  O  = running output accumulator

When a new tile arrives:
  1. Compute new local max: m_new = max(m, local_max)
  2. Rescale old accumulator: O = O × exp(m - m_new)
  3. Rescale old sum:         l = l × exp(m - m_new)
  4. Add new tile contribution to O and l
  5. Update m = m_new

This is exact, not approximate.
```

---

## Flash Attention vs Standard: Memory and Speed

```
+------------------------------------------------------------------+
|  COMPARISON                                                      |
+------------------------------------------------------------------+
|                                                                  |
|                    Standard           Flash Attention            |
|  Memory (HBM):     O(N²)             O(N)                       |
|  HBM accesses:     O(N²/d)           O(N² × d / M)              |
|                                       (much less for large M)   |
|  Wall-clock speed: baseline           2x-4x faster              |
|  Output:           exact              exact (not approximate!)   |
|  Gradient:         standard           custom CUDA kernel         |
|                                                                  |
|  Context length possible:                                        |
|    Standard attention: ~4K tokens on A100 (80 GB HBM)           |
|    Flash Attention:    128K+ tokens on same A100                 |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Flash Attention 1, 2, and 3

| Version | Year | Key Improvement |
|---------|------|-----------------|
| Flash Attention 1 | 2022 | Original paper — O(N) memory, 2-4x speedup |
| Flash Attention 2 | 2023 | Better thread-level parallelism, 2x faster than FA1 |
| Flash Attention 3 | 2024 | Optimized for H100 Tensor Cores, FP8 support |

**Flash Attention 2** is what most modern LLMs use (LLaMA 3, Mistral, Qwen).
It achieves 50-73% of theoretical peak GPU utilization vs ~25% for standard attention.

---

## Using Flash Attention in PyTorch

Since PyTorch 2.0, Flash Attention is built in through `F.scaled_dot_product_attention`:

```python
import torch
import torch.nn.functional as F

# PyTorch 2.0+ automatically uses Flash Attention when:
# 1. Input is on CUDA
# 2. dtype is float16 or bfloat16
# 3. No custom attention mask (or causal mask)

q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)

# This AUTOMATICALLY uses Flash Attention if conditions are met
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# To verify which backend is used:
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

For older code, install the standalone package:
```bash
pip install flash-attn --no-build-isolation
```

---

## C# Analogy: Streaming vs Buffering

```csharp
// Standard attention is like loading an entire file into memory to process it:
//
// BAD - Standard approach:
// byte[] entireFile = File.ReadAllBytes("10GB_file.bin");  // OOM!
// ProcessAll(entireFile);
//
// Flash Attention is like streaming through the file in chunks:
//
// GOOD - Streaming approach:
// using var stream = File.OpenRead("10GB_file.bin");
// using var reader = new BufferedStream(stream, bufferSize: 64_000);
// byte[] chunk = new byte[64_000];
// var accumulator = new OutputAccumulator();
//
// int bytesRead;
// while ((bytesRead = reader.Read(chunk, 0, chunk.Length)) > 0)
// {
//     var partialResult = Process(chunk[..bytesRead]);
//     accumulator.Update(partialResult);  // running aggregation
// }
//
// var finalResult = accumulator.Finalize();
//
// Flash Attention does EXACTLY this:
//   chunk        = one tile of Q, K, V  (fits in SRAM)
//   accumulator  = the running (m, l, O) statistics
//   Process()    = local tile matmul
//   Update()     = rescale and add to running output
//   Finalize()   = divide by l to get final softmax output
```

---

## Why This Matters for You

Flash Attention enables:

1. **Longer context windows**: 8K → 128K → 1M tokens
2. **Faster training**: 2x-4x wall-clock speedup
3. **Larger batch sizes**: memory savings → more examples per GPU step
4. **RAG applications**: retrieve and attend over more document chunks
5. **Code understanding**: long files (50K+ tokens) fit in one context

Without Flash Attention, modern LLMs with long context windows would be impossible.
GPT-4 Turbo's 128K context, Gemini's 1M context — all rely on Flash Attention.

---

## Quiz Questions

**Q1**: Standard attention requires O(N²) memory. Where does the N² come from?
        a) The number of transformer layers squared
        b) The attention matrix Q × K^T which has shape [N, N]
        c) The vocabulary size times the sequence length
        d) The number of heads times the head dimension

**Q2**: Flash Attention achieves O(N) memory by:
        a) Using lower precision numbers (fp8)
        b) Reducing the number of attention heads
        c) Processing attention in tiles that fit in SRAM, never writing N×N to HBM
        d) Skipping attention computation for distant tokens

**Q3**: What is SRAM vs HBM on a GPU?
        a) SRAM is the main GPU memory; HBM is the CPU cache
        b) SRAM is fast on-chip cache (~20 MB); HBM is the GPU's main memory (GBs)
        c) They are the same thing with different names
        d) SRAM is shared memory between GPUs; HBM is single GPU memory

*(Answers: Q1=b, Q2=c, Q3=b)*

---

## Key Takeaways

1. Standard attention requires an N×N matrix in memory → O(N²), context limited to ~4K
2. GPU memory hierarchy: SRAM (fast, tiny) vs HBM (slow, large)
3. Flash Attention tiles Q, K, V into blocks that fit in SRAM
4. A running (max, sum, output) accumulator replaces the full N×N matrix
5. Result is mathematically exact — not an approximation
6. Memory: O(N) instead of O(N²) — enables 128K+ context windows
7. Speed: 2x-4x faster due to reduced HBM reads/writes
8. Built into PyTorch 2.0+ via `F.scaled_dot_product_attention`

---

*Next: Lesson 4 — Gradient Checkpointing + ZeRO (training huge models on small GPUs)*

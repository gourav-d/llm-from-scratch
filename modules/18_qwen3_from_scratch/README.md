# Module 18: Qwen3.5 LLM from Scratch (PyTorch)

## What Is This Module About?

You have built a GPT-style model in Module 05.
You understand attention, transformers, training loops, and fine-tuning.

But GPT is from 2019. Modern LLMs — Qwen3.5, LLaMA 3, Mistral — use better techniques
that make them faster, smarter, and cheaper to run.

**This module builds a Qwen3.5-style decoder from scratch** using those modern techniques.
You will start from the pieces you know (attention, positional encoding) and upgrade each one.

> Key Insight: Qwen3.5 is not magic. It is GPT + RoPE + GQA + KV Cache + RLA,
> assembled with engineering care. This module teaches each upgrade one at a time.

---

## Why Qwen3.5 Specifically?

| Reason | Explanation |
|--------|------------|
| Open source | Architecture is fully documented and reproducible |
| Modern techniques | Uses RoPE, GQA, RLA — all in production LLMs today |
| Learnable size | 0.5B version runnable on laptop CPU |
| Real reference | You can compare your code to the official implementation |

---

## What You Will Upgrade (Module 05 → Module 18)

```
+=========================================================================+
|  WHAT WE ARE UPGRADING FROM GPT (M05) TO QWEN3.5 (M18)                 |
+=========================================================================+
|                                                                         |
|  M05: Sinusoidal positional encoding   →  M18: RoPE                    |
|        (fixed sin/cos added to input)         (rotation in Q/K space)  |
|                                                                         |
|  M05: Multi-Head Attention (MHA)       →  M18: Group-Query Attention    |
|        (N query heads, N KV heads)            (N query heads, G KV)    |
|                                                                         |
|  M05: No attention memory              →  M18: KV Cache                 |
|        (recomputes all past tokens)           (store K/V, reuse them)  |
|                                                                         |
|  M05: Standard softmax attention       →  M18: Recurrent Linear Attn   |
|        (O(N^2) time and memory)               (O(N) time and memory)   |
|                                                                         |
|  M05: Learned positional embedding     →  Removed (RoPE replaces it)   |
|                                                                         |
+=========================================================================+
```

---

## Prerequisites

| Module | Why Required |
|--------|-------------|
| Module 04 — Transformers | Attention, Q/K/V matrices, positional encoding |
| Module 05 — Building LLM | GPT architecture, next-token prediction, training loop |
| Module 14 — Deploying LLMs | KV cache concept (Lesson 6) |
| Module 15 — Advanced Training | Flash Attention (motivation for linear attention) |

---

## Module Structure

### Lessons (5 total)

| # | File | Topic |
|---|------|-------|
| 1 | lessons/01_rope.md | RoPE — Rotary Position Embeddings |
| 2 | lessons/02_gqa.md | Group-Query Attention (GQA) |
| 3 | lessons/03_rla.md | Recurrent Linear Attention (RLA) |
| 4 | lessons/04_kv_cache.md | KV Cache Management |
| 5 | lessons/05_qwen_assembly.md | Full Qwen3.5 Decoder Assembly |

### Examples (5 total — planned)

| # | File | What You Will Build |
|---|------|---------------------|
| 1 | examples/example_01_rope.py | RoPE rotation matrix, apply to Q and K |
| 2 | examples/example_02_gqa.py | MHA vs GQA vs MQA — memory comparison |
| 3 | examples/example_03_rla.py | Linear attention kernel + recurrent form |
| 4 | examples/example_04_kv_cache.py | KVCacheManager, tokens/sec benchmark |
| 5 | examples/example_05_qwen.py | Full decoder forward pass, text generation |

### Project (1 total — planned)

| File | What You Will Build |
|------|---------------------|
| projects/mini_qwen/ | 6-layer, 256-dim Qwen3.5-style model trained on Shakespeare |

---

## What You Will Learn

By the end of this module, you will be able to:

1. Explain why sinusoidal positional encoding has limits and how RoPE fixes them
2. Implement the RoPE rotation matrix from scratch
3. Explain how Group-Query Attention reduces KV memory without hurting quality
4. Build a GQA layer and measure memory savings vs standard MHA
5. Describe Recurrent Linear Attention and why it is O(N) instead of O(N²)
6. Implement a KV cache that reuses past key-value pairs during generation
7. Assemble all components into a working Qwen3.5-style decoder

---

## Key Terms Glossary

| Term | Simple Definition |
|------|-----------------|
| **RoPE** | Rotary Position Embedding — encode position by rotating Q and K vectors |
| **Rotation matrix** | A matrix that rotates a vector by some angle, without changing its length |
| **theta (θ)** | Base frequency controlling how fast rotation changes with position |
| **GQA** | Group-Query Attention — Q heads split into groups, each group shares one KV head |
| **MHA** | Multi-Head Attention — original: one KV head per Q head (standard GPT) |
| **MQA** | Multi-Query Attention — extreme: ALL Q heads share one single KV head |
| **KV head** | The Key and Value matrices — the expensive part of the KV cache |
| **RLA** | Recurrent Linear Attention — attention computed with O(1) memory per token |
| **KV Cache** | Stored past K and V matrices — reused during token generation |
| **RMSNorm** | Root Mean Square normalization — simpler than LayerNorm, used in Qwen |
| **SwiGLU** | Activation function used in Qwen FFN layers (swish × gated linear unit) |
| **Causal mask** | Triangular mask so each token only attends to past tokens |

---

## C# Analogy: The Big Picture

```csharp
// Building a Qwen3.5 decoder is like upgrading a legacy .NET service
// to a modern, high-performance architecture.
//
// Old GPT (Module 05) is like a .NET Framework 4.5 app:
//   - Works fine for small loads
//   - Not optimized for high concurrency
//   - Stores everything in memory without reuse
//
// Qwen3.5 is like a modern .NET 8 microservice with:
//
// RoPE:
//   Old: Hardcode position as a lookup table (sinusoidal table in memory)
//   New: Compute position on-the-fly using math (no table needed)
//   Analogy: DateTime.UtcNow vs a pre-computed timestamp lookup array
//
// GQA:
//   Old: Every query has its own private Key/Value pair (lots of memory)
//   New: Multiple queries share one Key/Value pair (shared readonly cache)
//   Analogy: static readonly field shared across all class instances
//            vs. an instance field duplicated in every object
//
// KV Cache:
//   Old: Recompute every past token's K/V at every step (slow)
//   New: Store K/V in a Dictionary<int, (K, V)>, look up instead of recompute
//   Analogy: IMemoryCache in ASP.NET Core — compute once, cache forever
//
// RLA:
//   Old: Attention is a full N×N matrix — O(N^2) memory
//   New: Recurrent form — one hidden state updated per token — O(1) memory
//   Analogy: streaming LINQ with yield return vs loading entire list into RAM
```

---

## How to Use This Module

1. Read lessons in order — each component depends on the previous
2. Lesson 01 (RoPE) and Lesson 02 (GQA) can be understood without GPU
3. Lessons 03-05 build on each other — do not skip
4. Run examples to see the numbers — theory clicks when you see memory savings
5. Build the Mini-Qwen project last — it assembles all five components

---

*Module 18 of the Learn LLM from Scratch course.*
*For a .NET developer learning Python and Large Language Models.*

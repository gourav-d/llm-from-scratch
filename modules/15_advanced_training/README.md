# Module 15: Advanced LLM Training

## What Is This Module About?

You have trained small models on toy datasets.
You have fine-tuned, aligned, and deployed them.

But real LLMs — GPT-4, LLaMA, Mistral — are trained on **trillions of tokens**
across **hundreds of GPUs** for **months at a time**.

How do they do it without running out of memory?
How do they decide how big the model should be?
How do they make attention fast enough for 128K context windows?

**This module answers all of those questions.**

> Key Insight: Training a 7B model on 1 consumer GPU is hard.
> Training a 70B model on 1,000 GPUs for 3 months requires entirely different techniques.
> This module teaches the techniques that make large-scale training possible.

---

## Why Does This Matter?

Understanding advanced training techniques lets you:

1. Make intelligent decisions about model size vs. data size
2. Train models that would otherwise exhaust your GPU memory
3. Reduce training time by 2×–10× with no accuracy loss
4. Compress large models into smaller ones without retraining from scratch
5. Train on datasets too large to download

These techniques are used in every serious LLM training run.

---

## Prerequisites

| Module | Why Required |
|--------|-------------|
| Module 03 - Neural Networks | Forward pass, backpropagation, activations |
| Module 03.5 - PyTorch | Tensors, nn.Module, training loops |
| Module 04 - Transformers | Attention mechanism, Q/K/V matrices |
| Module 06 - Training & Fine-tuning | Training loops, optimizers, loss functions |

---

## The Problem This Module Solves

```
+=========================================================================+
|  THE SCALING WALL                                                       |
+=========================================================================+
|                                                                         |
|  You want to train a 7B parameter model.                                |
|                                                                         |
|  Problem 1 — SIZE:                                                      |
|  7B params x 4 bytes (fp32) = 28 GB just for weights                   |
|  + optimizer state (Adam) = 3x more = 84 GB total                      |
|  Consumer GPU has 8-24 GB VRAM. You are stuck.                          |
|                                                                         |
|  Problem 2 — SPEED:                                                     |
|  Attention is O(N^2) in memory. For 32K context: 1B+ elements.          |
|  Just the attention matrix = 4 GB per layer. Not feasible.             |
|                                                                         |
|  Problem 3 — DATA:                                                      |
|  LLaMA 3 trained on 15 TRILLION tokens.                                 |
|  At 2 bytes per token = 30 TB of data. Cannot fit on any machine.       |
|                                                                         |
|  Problem 4 — HOW MUCH TO TRAIN:                                         |
|  Should you train a 7B model on 100B tokens?                            |
|  Or a 3B model on 300B tokens? Same compute budget, very different.     |
|                                                                         |
|  This module solves all four problems.                                  |
+=========================================================================+
```

---

## The Six Solutions

```
+--------------------------------------------------------------------+
|  PROBLEM                  |  SOLUTION                             |
+---------------------------+---------------------------------------+
|  Too much compute?        |  Chinchilla Laws → right-size model  |
|  Memory too small?        |  Mixed Precision → half the memory    |
|  Attention too slow/big?  |  Flash Attention → O(N) memory        |
|  GPU OOM during training? |  Grad Checkpoint + ZeRO → 8x savings  |
|  Dataset too large?       |  Streaming → train without download   |
|  Large model too slow?    |  Distillation → compress into small   |
+--------------------------------------------------------------------+
```

---

## Module Structure

### Lessons (6 total)

| # | File | Topic |
|---|------|-------|
| 1 | lessons/01_chinchilla_scaling.md | Compute-optimal training — how big should your model be? |
| 2 | lessons/02_mixed_precision.md | bf16/fp16 training — half memory, same accuracy |
| 3 | lessons/03_flash_attention.md | Flash Attention — O(N) memory for long contexts |
| 4 | lessons/04_gradient_checkpointing_zero.md | Gradient checkpointing + ZeRO — train huge models on small GPUs |
| 5 | lessons/05_dataset_streaming.md | HuggingFace streaming — train on TB without downloading |
| 6 | lessons/06_knowledge_distillation.md | Distillation — compress a 70B model into a 7B model |

### Examples (6 total)

| # | File | What You Will Build |
|---|------|---------------------|
| 1 | examples/example_01_chinchilla.py | Compute FLOPs budget, find optimal model size and token count |
| 2 | examples/example_02_mixed_precision.py | Simulate fp32/fp16/bf16 precision loss in NumPy |
| 3 | examples/example_03_flash_attention.py | Implement Flash Attention tiling from scratch in NumPy |
| 4 | examples/example_04_gradient_checkpointing.py | Manual gradient checkpointing in pure Python |
| 5 | examples/example_05_dataset_streaming.py | Simulate streaming dataset pipeline with lazy loading |
| 6 | examples/example_06_distillation.py | Implement teacher-student knowledge distillation |

### Exercises (6 total)

| # | File | What You Will Practice |
|---|------|------------------------|
| 1 | exercises/exercise_01_chinchilla.py | Calculate compute-optimal configs for given budgets |
| 2 | exercises/exercise_02_mixed_precision.py | Quantize and dequantize tensors, measure error |
| 3 | exercises/exercise_03_flash_attention.py | Complete a tiled attention implementation |
| 4 | exercises/exercise_04_checkpointing.py | Measure memory savings from gradient checkpointing |
| 5 | exercises/exercise_05_streaming.py | Build a lazy tokenization pipeline |
| 6 | exercises/exercise_06_distillation.py | Train a student model using soft labels |

---

## What You Will Learn

By the end of this module, you will be able to:

1. Use Chinchilla scaling laws to pick optimal model size for your compute budget
2. Explain the difference between fp32, fp16, and bf16 and when to use each
3. Describe why Flash Attention uses O(N) memory instead of O(N²)
4. Apply gradient checkpointing to train larger models on limited GPU memory
5. Use ZeRO optimizer stages to shard model state across multiple GPUs
6. Stream TB-scale datasets without running out of disk space
7. Distill a large teacher model into a small, fast student model

---

## Key Terms Glossary

| Term | Meaning |
|------|---------|
| **FLOPs** | Floating Point Operations — measure of compute cost |
| **Chinchilla Laws** | Rules for compute-optimal model size and token count |
| **fp32** | 32-bit float, 4 bytes. Default PyTorch precision |
| **fp16** | 16-bit float, 2 bytes. Faster but can overflow |
| **bf16** | Brain float 16. Same range as fp32, less precision. Safer for training |
| **AMP** | Automatic Mixed Precision — auto-switch between fp16 and fp32 |
| **Flash Attention** | Memory-efficient attention that avoids N×N matrix |
| **HBM** | High Bandwidth Memory — GPU's main memory (slow, big) |
| **SRAM** | Static RAM — GPU's on-chip cache (fast, tiny) |
| **Gradient Checkpointing** | Recompute activations during backward pass to save memory |
| **ZeRO** | Zero Redundancy Optimizer — shard optimizer state across GPUs |
| **Data Parallelism** | Same model, different data batches on each GPU |
| **IterableDataset** | Streaming dataset — fetches data lazily, no full download |
| **Distillation** | Train small student model to mimic large teacher model |
| **Soft Labels** | Teacher's full probability distribution over all tokens |
| **Hard Labels** | Just the correct answer (one-hot) |
| **Temperature** | Softens probability distribution during distillation |
| **KL Divergence** | Measures how different two probability distributions are |

---

## C# Analogy: The Big Picture

```csharp
// Training a large LLM is like running a distributed .NET application
// where each "microservice" is a GPU, and they must coordinate perfectly.
//
// The problems are the SAME as in distributed .NET systems:
//
// MEMORY:
//   .NET: Heap runs out on large datasets → use lazy loading, IEnumerable<T>
//   LLM:  GPU VRAM runs out → use mixed precision, gradient checkpointing
//
// BANDWIDTH:
//   .NET: Database queries are slow → cache, batch, index
//   LLM:  HBM reads are slow → Flash Attention keeps data in fast SRAM cache
//
// SCALING:
//   .NET: App can't handle load → shard database, distribute workers
//   LLM:  One GPU not enough → ZeRO shards optimizer state across GPUs
//
// DATA PIPELINE:
//   .NET: 1TB CSV won't fit in memory → stream with StreamReader, yield return
//   LLM:  15TB dataset won't fit → HuggingFace streaming IterableDataset
//
// MODEL COMPRESSION:
//   .NET: 500MB DLL won't deploy to mobile → tree-shake, reduce dependencies
//   LLM:  70B model too slow → distill into 7B model
```

---

## How to Use This Module

1. Read lessons in order — each builds on the previous
2. Lesson 01 (Chinchilla) sets context — do NOT skip
3. Lessons 02-06 can be read independently after Lesson 01
4. Run all examples — the concepts only click when you see the numbers
5. Complete exercises before moving to the next lesson

---

*Module 15 of the Learn LLM from Scratch course.*
*For a .NET developer learning Python and Large Language Models.*

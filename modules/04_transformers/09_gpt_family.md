# Lesson 09 — The GPT Family: GPT-1 through GPT-4

**Module:** 04 — Transformers  
**Prerequisite:** Lesson 06 (complete GPT architecture), Lesson 07 (BERT), Lesson 08 (T5)

---

## Overview

GPT = **Generative Pre-trained Transformer**. Built by OpenAI.

You already built a GPT-style model from scratch in Lesson 06. This lesson covers the actual GPT family — how each version improved, and what architectural changes mattered.

> **C# analogy:** Think of GPT as a software product line. GPT-1 = v1.0. Each version adds features, fixes bugs, scales up. The core architecture (C# language) stays similar; the training data, scale, and techniques change dramatically.

---

## The Transformer Shape: Decoder-Only

All GPT models use **only the decoder** from the transformer paper — but with one modification:

The original decoder had **two attention layers**:
1. Self-attention (attends to previously generated tokens)
2. Cross-attention (attends to encoder output)

GPT **removes cross-attention** since there's no encoder. It only has masked self-attention.

```
Original Transformer Decoder:
  ┌──────────────────────────────┐
  │  Masked Self-Attention        │  ← GPT keeps this
  │  Cross-Attention (→ encoder)  │  ← GPT removes this
  │  Feed-Forward Network         │  ← GPT keeps this
  └──────────────────────────────┘

GPT Decoder Block:
  ┌──────────────────────────────┐
  │  Masked Self-Attention        │
  │  Feed-Forward Network         │
  └──────────────────────────────┘
```

**Masked** self-attention means each token can only attend to previous tokens, not future ones. This enables left-to-right generation.

---

## GPT Training: Causal Language Modeling

All GPT models are trained on the same simple objective: **predict the next token**.

```
Input:   "The cat sat on the"
Target:  "cat sat on the mat"
         (every token predicts the next one)
```

This is called **causal language modeling** (CLM) — "causal" because each step only depends on past context, never future.

> **C# analogy:** Like predicting the next character in an `IEnumerable<char>` stream — each prediction only looks at what came before, never what comes after.

---

## GPT-1 (2018)

**Paper:** "Improving Language Understanding by Generative Pre-Training"

| Property | Value |
|----------|-------|
| Parameters | 117 million |
| Layers | 12 transformer decoder blocks |
| Context window | 512 tokens |
| Training data | BooksCorpus (7,000 unpublished books, 4.5GB) |
| Vocabulary | 40,478 BPE tokens |

**Key innovation:** Pre-train on unlabeled text, then fine-tune on labeled data.  
Before GPT-1, most models were trained on labeled data only. GPT-1 showed that unsupervised pre-training helps.

**Limitation:** Only trained on books. Limited vocabulary. Context too short.

---

## GPT-2 (2019)

**Paper:** "Language Models are Unsupervised Multitask Learners"

| Property | Value |
|----------|-------|
| Parameters | 117M → 1.5B (4 sizes) |
| Layers | 12 → 48 |
| Context window | 1,024 tokens |
| Training data | WebText (45GB of Reddit-linked web pages) |
| Vocabulary | 50,257 BPE tokens |

**Key innovation:** Zero-shot task performance. GPT-2 was never fine-tuned on specific tasks, yet it could translate, summarize, and answer questions — just from how it was prompted.

```
Prompt: "Translate to French: The weather is nice"
GPT-2:  "Le temps est beau"   ← learned from patterns in training data
```

**OpenAI's decision:** Initially refused to release the full model, claiming it was "too dangerous." Later released fully. Sparked debate about responsible AI disclosure.

**Layer normalization:** GPT-2 moved LayerNorm to the input of each sub-layer (Pre-LN) instead of after (Post-LN). More stable training.

```
GPT-1 (Post-LN):   attention → + residual → LayerNorm
GPT-2 (Pre-LN):    LayerNorm → attention → + residual
```

---

## GPT-3 (2020)

**Paper:** "Language Models are Few-Shot Learners"

| Property | Value |
|----------|-------|
| Parameters | 175 billion |
| Layers | 96 transformer blocks |
| Attention heads | 96 |
| Context window | 2,048 tokens |
| Training data | 570GB filtered CommonCrawl + books + Wikipedia + WebText2 |
| Vocabulary | 50,257 BPE tokens |

**Key innovation:** Few-shot and zero-shot learning via prompting.

GPT-3 doesn't need to be fine-tuned. You just give it examples in the prompt:

```
Zero-shot:
  "Translate to French: The cat is blue"
  GPT-3: "Le chat est bleu"

One-shot:
  "English: dog | French: chien
   English: The cat is blue | French:"
  GPT-3: "Le chat est bleu"

Few-shot:
  "English: dog    | French: chien
   English: house  | French: maison
   English: water  | French: eau
   English: The cat is blue | French:"
  GPT-3: "Le chat est bleu"
```

**Scale changed everything:** 175B parameters gave emergent abilities not seen in smaller models. GPT-3 could code, reason, joke, and write essays.

**Cost:** Training GPT-3 cost approximately $4–12 million in compute.

---

## InstructGPT & ChatGPT (2022)

Between GPT-3 and GPT-4, OpenAI added **RLHF** (covered in Module 13):

```
GPT-3 (base)        → Fine-tune on instructions → Add RLHF → InstructGPT / ChatGPT
(next token pred.)     (supervised examples)      (human    (follows instructions,
                                                  feedback)  helpful, harmless)
```

GPT-3 base was powerful but raw — it would continue any text including harmful content. InstructGPT/ChatGPT learned to be helpful and safe.

> **This is the model behind the public ChatGPT interface launched November 2022.**

---

## GPT-4 (2023)

OpenAI did not publish a technical paper with full details. Known properties:

| Property | Known / Estimated |
|----------|------------------|
| Parameters | Not disclosed (~1T+, likely mixture of experts) |
| Context window | 8K (base), 32K (extended), 128K (GPT-4 Turbo) |
| Training data | Up to September 2021 (original), later updated |
| Modality | Text + images (multimodal) |
| Benchmark | Passed bar exam (top 10%), LSAT, GRE, many academic tests |

**Key improvements over GPT-3:**
- Much better at following complex instructions
- Multimodal: can understand images as input
- Much longer context window
- Significantly reduced hallucinations
- Stronger reasoning abilities

**Mixture of Experts (MoE):** GPT-4 is suspected to use MoE — instead of all parameters running for every token, specialized "expert" sub-networks activate per token. Faster inference, more capacity.

---

## Architectural Evolution: GPT-1 → GPT-4

```
Feature            GPT-1    GPT-2    GPT-3    GPT-4
─────────────────────────────────────────────────────
Parameters         117M     1.5B     175B     ~1T (est)
Layers             12       48       96       unknown
Context (tokens)   512      1,024    2,048    128K
LayerNorm position Post     Pre      Pre      Pre
Training data (GB) 4.5      45       570      unknown
Modality           text     text     text     text+image
RLHF               No       No       No*      Yes
───────────────────────────────────────────────────────
* InstructGPT added RLHF to GPT-3 weights
```

---

## The Scaling Laws

In 2020, OpenAI published the "Scaling Laws" paper showing:

> **Model performance improves predictably as you increase: parameters, data, compute**

```
                        More parameters
                        ────────────────▶
More data   ▲
            │
            │         Performance
            │         improves here
            │          ↗
            └──────────────────────────▶
                                    More compute
```

This gave OpenAI (and others) a roadmap: just scale everything up and performance improves. GPT-3 and GPT-4 were direct bets on this insight.

---

## How GPT Generates Text

At inference time, GPT uses one of these strategies (covered more in Module 05 and 06):

```
Strategy        How it works                    Use case
──────────────────────────────────────────────────────────
Greedy          Always pick highest prob token   Fast, repetitive
Temperature     Sample with randomness factor    Creative writing
Top-K           Sample from top K candidates     Balanced
Top-P (nucleus) Sample from top-P% probability  Most natural
Beam search     Keep K best partial sequences    Translation
```

---

## Open-Source GPT Variants

While GPT-3 and GPT-4 are closed (API-only), many open-source alternatives exist:

| Model | Creator | Params | Architecture |
|-------|---------|--------|-------------|
| GPT-2 | OpenAI | 1.5B | Decoder-only |
| GPT-J | EleutherAI | 6B | Decoder-only |
| GPT-NeoX | EleutherAI | 20B | Decoder-only |
| LLaMA 1/2/3 | Meta | 7B–70B | Decoder-only |
| Mistral | Mistral AI | 7B | Decoder-only (GQA) |
| Gemma | Google | 2B–9B | Decoder-only |
| Phi | Microsoft | 1.3B–14B | Decoder-only |

> All are decoder-only transformers. The core architecture is the same as what you built in Lesson 06.

---

## Quiz Questions

**Q1:** What two parts of the original transformer did GPT remove vs keep?  
→ GPT removed cross-attention (no encoder). Kept masked self-attention and feed-forward layers.

**Q2:** What training objective do all GPT models use?  
→ Causal language modeling: predict the next token given all previous tokens.

**Q3:** What was the key innovation of GPT-2?  
→ Zero-shot task performance — the model could do translation, summarization etc. from prompting alone, without task-specific fine-tuning.

**Q4:** What does "few-shot learning" mean in GPT-3?  
→ Providing a few input-output examples inside the prompt itself. GPT-3 learns the pattern from those examples without updating any weights.

**Q5:** What was added between GPT-3 and ChatGPT?  
→ RLHF (Reinforcement Learning from Human Feedback) via InstructGPT — made the model follow instructions and be helpful/safe.

**Q6:** What is the "scaling law" insight?  
→ Model performance improves predictably when you scale up parameters, training data, and compute simultaneously.

---

## Key Takeaways

- GPT = decoder-only transformer. No encoder, no cross-attention. Masked self-attention only.
- Training: predict next token (causal language modeling). Simple but powerful.
- GPT-1 (117M): proved pre-training + fine-tune works.
- GPT-2 (1.5B): proved zero-shot prompting works at scale.
- GPT-3 (175B): proved few-shot prompting works; emergent abilities at scale.
- ChatGPT: GPT-3 + RLHF → follows instructions, safer, more helpful.
- GPT-4: multimodal, 128K context, much stronger reasoning.
- Open-source alternatives (LLaMA, Mistral) use the same decoder-only architecture.

# Module 17 — LLM Evaluation & Benchmarks

## What This Module Is About

You can now build, train, and deploy LLMs. But how do you know if your model is **good**?

This module covers the full evaluation stack — from numbers you watch during training,
to standard benchmarks used by the entire research community, to text quality metrics.

By the end, you will be able to:
- Read a HuggingFace Open LLM Leaderboard entry and understand every number
- Know which benchmark to use for which task (math vs code vs knowledge vs safety)
- Run evaluation yourself using `lm-evaluation-harness`
- Know when your model is overfitting, underfitting, or just plain bad

---

## Why Evaluation Is Hard

Building a language model is relatively easy (you did it in M05).
**Knowing if it's any good is surprisingly hard.**

```
Simple question: "Is GPT-4 better than Mistral 7B?"
Complex answer:  Better at WHAT?
                  - General knowledge? → MMLU
                  - Math reasoning?    → GSM8K
                  - Writing code?      → HumanEval
                  - Not hallucinating? → TruthfulQA
                  - Following instructions? → AlpacaEval

There is no single number. Context matters.
```

---

## Three-Layer Evaluation Framework

```
Layer 1 — DURING TRAINING (continuous):
  - Training loss (should decrease)
  - Validation loss (should decrease, tracks generalization)
  - Perplexity (exp(loss), lower = better)
  - Watch for: val loss diverging from train loss = overfitting

Layer 2 — AFTER TRAINING (benchmark suite):
  - General knowledge: MMLU, HellaSwag, ARC
  - Reasoning/math: GSM8K, MATH
  - Code: HumanEval, MBPP
  - Factuality: TruthfulQA
  - Compare your model to published baselines

Layer 3 — QUALITATIVE CHECK (human):
  - Same 20 prompts, compare before/after training
  - Check coherence, repetition, factual errors
  - "Vibe check" — does it feel better?
```

No model ships without all 3 layers.

---

## Learning Objectives

After this module you will be able to:

- [ ] Compute perplexity from cross-entropy loss and explain what it means
- [ ] Identify overfitting from training/validation loss curves
- [ ] Explain what MMLU, HellaSwag, ARC measure and how they are scored
- [ ] Explain what GSM8K, HumanEval, TruthfulQA measure
- [ ] Compute BLEU and ROUGE scores manually
- [ ] Explain BERTScore and when to use it over BLEU/ROUGE
- [ ] Run `lm-evaluation-harness` to evaluate a model on standard benchmarks
- [ ] Read an Open LLM Leaderboard entry and understand each column

---

## Lessons

| # | Topic | File |
|---|-------|------|
| L1 | Training metrics — loss, perplexity, overfitting detection | concepts.md → Lesson 1 |
| L2 | General benchmarks — MMLU, HellaSwag, ARC | concepts.md → Lesson 2 |
| L3 | Task benchmarks — GSM8K, HumanEval, TruthfulQA | concepts.md → Lesson 3 |
| L4 | Text quality metrics — BLEU, ROUGE, BERTScore | concepts.md → Lesson 4 |
| L5 | Evaluation in practice — lm-eval harness, leaderboards | concepts.md → Lesson 5 |

---

## Prerequisites

- Module 05 — Building an LLM (cross-entropy loss, perplexity)
- Module 06 — Training loops (train/val split, loss curves)
- Module 07 — Reasoning models (GSM8K, code evaluation)
- Module 12 — Fine-tuning (what changes after fine-tuning?)

---

## Libraries Used

| Library | Purpose | Install |
|---------|---------|---------|
| `evaluate` | HuggingFace evaluation metrics (BLEU, ROUGE, BERTScore) | `pip install evaluate` |
| `lm-eval` | EleutherAI evaluation harness — run any benchmark | `pip install lm-eval` |
| `datasets` | Load benchmark datasets | `pip install datasets` |
| `torch` | Run model for evaluation | `pip install torch` |
| `numpy` | Compute metrics from scratch | already installed |

---

## Files in This Module

```
17_evaluation/
├── README.md           ← You are here
├── concepts.md         ← All 5 lessons with theory, formulas, examples, quizzes
├── python_guide.md     ← Python features used in this module
├── examples/           ← Code examples (one per lesson)
└── exercises/          ← Practice problems with solutions
```

---

## Key Terms Glossary

| Term | Simple Definition |
|------|------------------|
| Perplexity | How "surprised" the model is by text. Lower = model predicts text well |
| Cross-entropy loss | Average negative log probability of true tokens |
| Overfitting | Train loss low, val loss high — model memorized training data |
| MMLU | 57-subject multiple-choice benchmark (knowledge breadth) |
| HellaSwag | Complete a sentence naturally (common sense reasoning) |
| ARC | Science multiple-choice questions (grade school to high school) |
| GSM8K | 8,500 grade-school math word problems |
| HumanEval | 164 Python coding problems, test-evaluated |
| TruthfulQA | Questions designed to trap models into common misconceptions |
| BLEU | n-gram overlap between generated and reference text (precision) |
| ROUGE | n-gram overlap with focus on recall |
| BERTScore | Semantic similarity using BERT embeddings |
| Pass@k | Probability correct code generated in k attempts (HumanEval) |
| lm-eval | EleutherAI's open-source benchmark runner |

---

## Time Estimate

| Activity | Time |
|----------|------|
| Read concepts.md | 90 minutes |
| Study examples | 60 minutes |
| Complete exercises | 90 minutes |
| **Total** | **~4 hours** |

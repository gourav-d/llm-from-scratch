# Module 16 — Modern LLM Architectures

## What This Module Is About

You have built a GPT-style model. It generates text **one token at a time**, left to right.
That works well — but researchers found better ways.

This module covers two major architectural innovations that go **beyond** standard auto-regressive generation:

1. **Diffusion Language Models** — generate ALL tokens at once, then refine them iteratively
2. **Multi-Token Prediction (MTP)** — predict multiple next tokens simultaneously instead of one

These are the ideas powering next-generation models in 2024–2025.

---

## Why This Matters

| Problem with GPT-style | Solution |
|------------------------|----------|
| Generates tokens one at a time → slow | Diffusion: generate all tokens in parallel |
| Each token only depends on past tokens | MTP: model learns to plan ahead N steps |
| Cannot revise earlier tokens | Diffusion: iterative refinement fixes errors |

---

## Learning Objectives

After this module you will be able to:

- [ ] Explain why auto-regressive generation has speed and quality limits
- [ ] Describe the diffusion process adapted for discrete text tokens
- [ ] Name three diffusion language models (MDLM, SEDD, Plaid) and their approach
- [ ] Explain ELBO, denoising score matching, and variational lower bound
- [ ] Describe how Multi-Token Prediction (MTP) works and why Meta built it
- [ ] Compare auto-regressive vs diffusion vs MTP — tradeoffs of each

---

## Lessons

| # | Topic | File |
|---|-------|------|
| L1 | Auto-regressive recap + limitations | concepts.md → Lesson 1 |
| L2 | Diffusion process for text | concepts.md → Lesson 2 |
| L3 | Diffusion Language Models | concepts.md → Lesson 3 |
| L4 | Diffusion loss function | concepts.md → Lesson 4 |
| L5 | Multi-Token Prediction (MTP) | concepts.md → Lesson 5 |

---

## Prerequisites

- Module 04 — Transformer architecture (attention, encoder-decoder)
- Module 05 — Building an LLM (GPT-style, next-token prediction)
- Module 15 — Advanced training (Flash Attention, mixed precision)

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `numpy` | Matrix math, probability distributions |
| `torch` | Neural network layers, training loops |
| `math` | Log, exp for loss calculations |

---

## Files in This Module

```
16_modern_architectures/
├── README.md           ← You are here
├── concepts.md         ← All 5 lessons with theory, diagrams, quizzes
├── python_guide.md     ← Python features used in this module
├── examples/           ← Code examples (one per lesson)
└── exercises/          ← Practice problems with solutions
```

---

## Key Terms Glossary

| Term | Simple Definition |
|------|------------------|
| Auto-regressive | Generate one token at a time, left to right |
| Diffusion model | Start with noise, iteratively denoise to get output |
| Discrete tokens | Text as integers (not continuous floats like images) |
| Denoising | Removing noise to recover the original signal |
| ELBO | Evidence Lower BOund — the loss function for diffusion models |
| VLB | Variational Lower Bound — same as ELBO, different name |
| MTP | Multi-Token Prediction — predict N future tokens at once |
| MDLM | Masked Diffusion Language Model — mask tokens, learn to unmask |
| SEDD | Score Entropy Discrete Diffusion — score-based diffusion for text |

---

## Time Estimate

| Activity | Time |
|----------|------|
| Read concepts.md | 90 minutes |
| Study examples | 60 minutes |
| Complete exercises | 90 minutes |
| **Total** | **~4 hours** |

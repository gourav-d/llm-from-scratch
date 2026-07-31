# Learning Progress — LLM from Scratch

**Last Updated:** 2026-07-31 (M01 exercises + M02.5 + M05.5 + M19 complete)
**Student:** .NET developer learning Python + LLMs simultaneously

---

## Module Overview

| # | Module | Lessons | Examples | Exercises | Projects | Status |
|---|--------|---------|----------|-----------|----------|--------|
| 01 | Python Basics | 10 | 10 | 10 | — | ✅ Complete (exercises added 2026-07-31) |
| 01.5 | Pandas & Data Handling | 5 | 5 | 5 | — | ✅ Complete |
| 02 | NumPy & Math | 3 | 4 | 3 | — | ✅ Complete |
| 02.5 | Classical ML (Regression + Trees) | 5 | 5 | 5 | — | ✅ Complete (2026-07-31) |
| 03 | Neural Networks | 6 | 6 | 6 | 3 | ✅ Complete |
| 03.5 | PyTorch & TensorFlow | 5 | 2 | 1 | 1 | ✅ Complete (minimal exercises) |
| 04 | Transformers | 9 | 18 * | 3 | — | ✅ Complete (L7–9 concept-only) |
| 05 | Building Your LLM | 5 | 5 | 5 | 7 | ✅ Complete |
| 05.5 | HuggingFace Tokenizers | 5 | 5 | 5 | — | ✅ Complete (2026-07-31, HF optional) |
| 06 | Training & Fine-tuning | 6 | 7 | 6 | 4 | ✅ Complete |
| 07 | Reasoning & Coding Models | 10 | 10 | 5 | 5 | ✅ Complete |
| 08 | Prompt Engineering | 10 | 3 | 2 | — | ✅ Complete |
| 09 | Production LLM Applications | 4 | 4 | 4 | 2 | ✅ Complete |
| 10 | Vector Databases | 5 | 5 | 3 | 2 | ✅ Complete |
| 10.5 | RAG Without Vectors (BM25/TF-IDF) | 5 | — | — | — | ✅ Complete (theory only) |
| 10.8 | Semantic Search Systems | 6 | 6 | 5 | 1 | ✅ Complete |
| 11 | LLM Agents | 5 | 5 | 5 | 3 | ✅ Complete |
| 12 | Fine-Tuning LLMs | 5 | 5 | 5 | 3 | ✅ Complete |
| 13 | RLHF and Alignment | 5 | 5 | 5 | 3 | ✅ Complete |
| 14 | Deploying LLMs | 6 | 6 | 6 | — | ✅ Complete |
| 14.5 | Gradio & Streamlit UIs | — | — | — | — | 📅 Planned (optional) |
| 15 | Advanced LLM Training | 6 | 6 | 6 | — | ✅ Complete (2026-05-30) |
| 16 | Modern LLM Architectures | 5 | 5 | 5 | — | ✅ Complete (2026-06-30) |
| 17 | LLM Evaluation & Benchmarks | 5 | 5 | 5 | — | ✅ Complete (2026-06-30) |
| 18 | Qwen3.5 LLM from Scratch (PyTorch) | 5 | 5 | 5 | 1 | ✅ Complete (2026-06-30) |
| CAP | Capstone: Chat with Codebase | — | — | 3 | 1 | ✅ Complete (2026-07-31) |
| **—** | **NEW MODULES — implement soon** | | | | | |
| 13++ | M13 Extension: RLVR / GRPO | 3 | 3 | 3 | — | ✅ Complete (2026-07-31) |
| 11++ | M11 Extension: MCP + LangGraph | 3 | 3 | 3 | — | ✅ Complete (2026-07-31) |
| 19 | Mixture of Experts (MoE) | 5 | 5 | 5 | — | ✅ Complete (2026-07-31, Mini-MoE GPT trains in ~6s CPU) |
| 20 | State Space Models (Mamba/SSM) | 6 | — | — | 1 | 📅 Planned |
| 10.9 | Multimodal RAG (ColPali) | 5 | — | — | — | 📅 Planned |
| 14++ | M14 Extension: Speculative Decoding | 3 | — | — | — | 📅 Planned |
| 21 | Vision Language Models (VLMs) | 5 | — | — | 1 | 📅 Planned |

\* Module 04 examples: 6 NumPy + 6 PyTorch + 6 TensorFlow (all in `examples/`, `examples/pytorch/`, `examples/tensorflow/`)

---

## Module Details

### Module 01 — Python Basics
- **Lessons:** `01_variables_and_types.md` through `10_error_handling.md`
- **Examples:** `example_01_basics.py` through `example_10_error_handling.py`
- **Missing:** Exercises (folder exists but empty)
- **Action:** Add exercises or skip — content is introductory and student has moved past this

---

### Module 02 — NumPy & Math
- **Lessons:** numpy basics, array operations, linear algebra
- **Content:** 4 examples, 3 exercises, quiz (35 questions)
- **Topics:** Arrays, broadcasting, linear algebra, math foundations for LLMs

---

### Module 03 — Neural Networks
- **Lessons:** Perceptron → activation functions → MLP → backprop → training loop → optimizers
- **Examples:** 6 files with visualizations
- **Exercises:** 6 files
- **Projects:**
  - Email Spam Classifier (`projects/neural_networks/email_spam_classifier/`)
  - MNIST Handwritten Digits (`projects/neural_networks/mnist_digits/`)
  - Sentiment Analysis (`projects/neural_networks/sentiment_analysis/`)

---

### Module 03.5 — PyTorch & TensorFlow
- **Lessons:** PyTorch fundamentals, nn.Module, NumPy→PyTorch, TensorFlow/Keras, framework comparison
- **Examples:** 2 files
- **Exercises:** 1 file
- **Projects:** 1 file

---

### Module 04 — Transformers
- **Lessons:** Attention → self-attention → multi-head → positional encoding → transformer block → GPT → BERT → T5 → GPT Family
- **Examples (NumPy):** `examples/example_01` through `example_06`
- **Examples (PyTorch):** `examples/pytorch/example_01` through `example_06`
- **Examples (TensorFlow):** `examples/tensorflow/example_01` through `example_06`
- **Exercises:** 3 files
- **Key achievement:** Full transformer architecture with side-by-side NumPy / PyTorch / TF comparisons
- **Lessons 7–9 (concept-only):**
  - `07_bert.md` — BERT: encoder-only, bidirectional, MLM pre-training, [CLS]/[SEP] tokens
  - `08_t5.md` — T5: encoder-decoder, text-to-text framework, task prefixes, span corruption, C4 dataset overview
  - `09_gpt_family.md` — GPT-1 → GPT-4: causal LM, scaling laws, few-shot, RLHF, open-source variants

---

### Module 05 — Building Your LLM
- **Lessons:** Tokenization → word embeddings → nanoGPT → GPT with PyTorch → text generation
- **Examples:** 5 files
- **Exercises:** 5 files
- **Projects:** 7 files
  - Shakespeare Generator, Custom Chatbot, Smart Autocomplete, Email Subject Generator,
    SQL Query Completer, Log Anomaly Detector, Company Name Generator
- **Model Evolution Lab:** `model_evolution/` — 8-step hands-on series (added 2026-06-02)
  - 11 files: `download_data.py`, `shared.py`, `step_00` through `step_07`
  - Step 0: Bigram (lookup table, val loss ~2.4, 1 min CPU)
  - Step 1: MLP with 16-char context window (val loss ~2.0, 3 min)
  - Step 2: Single-head self-attention + positional embedding (val loss ~1.7, 5 min)
  - Step 3: Multi-head attention (4 heads) + FeedForward (val loss ~1.5, 7 min)
  - Step 4: Full NanoGPT — 3 stacked blocks + LayerNorm + residuals (val loss ~1.3, 10 min)
  - Step 5: Group-Query Attention — fewer KV heads, 50% KV memory saved
  - Step 6: KV Cache — 5-10x faster generation, tokens/sec benchmark printed
  - Step 7: RoPE positional embeddings — no learned pos embed, better extrapolation (val loss ~1.2)
  - Each step: before/after training text sample shown, full C# analogies, every line commented
  - Final summary: shows full evolution from bigram → mini-Qwen equivalent

---

### Module 06 — Training & Fine-tuning
- **Lessons:** Building GPT → text generation → training → fine-tuning → RLHF/alignment → deployment → training datasets
- **Examples:** 7 files
- **Projects:** 4 files
- **Key topics:** End-to-end GPT pipeline, RLHF (how ChatGPT was trained), optimization
- **Lesson 7 (concept-only):** `07_training_datasets.md` — C4 dataset, Common Crawl, filtering pipeline, data quality vs quantity, knowledge cutoff, deduplication

---

### Module 07 — Reasoning & Coding Models
- **Part A — Reasoning:** Chain-of-Thought, Self-Consistency, Tree-of-Thoughts, Process Supervision, o1-style systems
- **Part B — Coding:** Code tokenization, code embeddings/AST, training on code, code generation, evaluation
- **Examples:** 10 files
- **Projects:** AI Code Reviewer, Smart Bug Debugger, Semantic Code Search, Auto Test Writer, Code Quality Analyzer
- **Key achievement:** Understand how OpenAI o1 and GitHub Copilot work internally

---

### Module 08 — Prompt Engineering
- **Lessons:** Zero-shot → few-shot → templates → role/system → CoT → ToT → structured outputs → optimization → security → production patterns
- **Examples:** 3 files
- **Exercises:** 2 files
- **Quiz:** 50 questions

---

### Module 09 — Production LLM Applications
- **Lessons:** API design (FastAPI, JWT, streaming) -> deployment (Docker, PostgreSQL, Redis, Kubernetes) -> monitoring (Prometheus, Grafana) -> security & cost optimization
- **Examples:** 4 files (01: API design patterns, 02: deployment concepts, 03: monitoring & observability, 04: security & cost)
- **Exercises:** 4 files (01: API validation/rate limiting/middleware, 02: cache/load balancer/health checks, 03: structured logging/metrics/alerts, 04: injection detection/PII/quota)
- **Projects:** 2 files
  - Production Chat API (8-step pipeline: auth, rate limit, quota, scan, session, LLM, stream, metrics)
  - Multi-Tenant SaaS Platform (tenant isolation, billing engine, per-tier quotas, usage reports)
- **Key topics:** JWT auth, rate limiting, connection pooling, LRU cache, load balancing, structured logging, p99 latency, Prometheus metrics, distributed tracing, prompt injection scanning, PII redaction, cost tracking, multi-tenancy
- **Status:** Complete

---

### Module 10.8 -- Semantic Search Systems
- **Lessons:** 6 files (keyword vs semantic, bi-encoder, cross-encoder/re-ranking, FAISS/ANN, hybrid search, full pipeline)
- **Examples:** 6 files (01: keyword vs semantic failure, 02: bi-encoder search, 03: cross-encoder rerank, 04: FAISS index, 05: hybrid search, 06: full pipeline)
- **Exercises:** 5 files (01: encode and search, 02: reranker, 03: FAISS basics, 04: hybrid search, 05: full pipeline)
- **Projects:** mini_search_engine (4 files: indexer, searcher, reranker, main console loop over 50 Wikipedia summaries)
- **Key topics:** Vocabulary mismatch problem, bi-encoder (fast retrieval), cross-encoder (accurate re-ranking), HNSW index, ANN search, BM25+semantic hybrid, Reciprocal Rank Fusion, NDCG/MRR/Precision@K
- **All examples:** NumPy-only primary (no install needed). Part B sections (commented) show sentence-transformers + faiss-cpu.
- **Connects to:** M10 (vector DBs/cosine similarity), M10.5 (BM25 keyword search), M07 (code semantic search project)
- **Status:** Complete (2026-05-19)

---

### Module 11 — LLM Agents
- **Lessons:** What are agents -> Tool use & function calling -> ReAct pattern -> Memory & state -> Multi-agent systems
- **Examples:** 5 files (01: simple agent, 02: tool use + parallel, 03: full ReAct loop, 04: memory agent, 05: multi-agent orchestrator)
- **Exercises:** 5 files (01: basic agent, 02: custom tools, 03: ReAct loop, 04: memory, 05: orchestration)
- **Projects:** 3 files
  - Personal Assistant (tools + memory + task list)
  - Research Agent (ReAct + RAG with vector-like memory)
  - Code Review Agent (multi-agent: BugAnalyzer + StyleChecker + Suggestions)
- **Key topics:** ReAct pattern, tool calling, scratchpad, short/long-term memory, orchestrator-worker pattern
- **Connects to:** Module 07 (Chain-of-Thought), Module 08 (prompting), Module 10 (vector DB memory)
- **Status:** Complete

---

### Module 12 — Fine-Tuning LLMs
- **Lessons:** What is fine-tuning -> Dataset preparation -> LoRA & PEFT -> Training loop -> Evaluation & inference
- **Examples:** 5 files (01: fine-tuning concepts, 02: dataset prep + JSONL, 03: LoRA from scratch, 04: training loop with early stopping, 05: base vs fine-tuned comparison)
- **Exercises:** 5 files (01: weight update magnitude + overfitting detection, 02: build JSONL dataset + splits, 03: LoRA layer from scratch, 04: batcher + EarlyStopper + LR scheduler, 05: accuracy/F1/confusion matrix/interpret results)
- **Projects:** 3 files
  - Sentiment Fine-Tuner (full pipeline: data -> train -> evaluate -> compare base vs fine-tuned)
  - Instruction Tuner (Alpaca format, prompt templates, loss masking demo)
  - Domain Chatbot Fine-Tuner (IT support knowledge base, intent classifier, response generator)
- **Key topics:** Transfer learning, LoRA math (W = W_frozen + scale*(B@A)), PEFT, Alpaca template, cross-entropy loss masking, early stopping, cosine LR schedule, accuracy/precision/recall/F1, confusion matrix, A/B comparison
- **Connects to:** Module 06 (training loops), Module 03 (neural net basics), Module 11 (using fine-tuned models in agents)
- **Status:** Complete

---

### Module 10 — Vector Databases
- **Lessons:** What are vectors -> Embeddings and similarity -> ChromaDB hands-on -> Building document search -> Real-world applications
- **Examples:** 5 files (01: NumPy similarity, 02: embeddings from scratch, 03: ChromaDB basics, 04: document search engine, 05: full semantic search + RAG)
- **Exercises:** 3 files (01: similarity math, 02: build mini vector store, 03: ChromaDB semantic search)
- **Projects:**
  - Document Search Engine (full RAG pipeline with ChromaDB + evaluation)
  - SQL Server as a Vector Database (existing table + VECTOR column + hybrid search + RAG)
- **Key topics:** Cosine similarity, word embeddings, ChromaDB, RAG pattern, search evaluation
- **Status:** Complete

---

### Module 13 — RLHF and Alignment
- **Lessons:** What is RLHF -> Reward Models -> PPO for LLMs -> DPO -> Constitutional AI
- **Examples:** 5 files (01: RLHF 3-phase pipeline, 02: reward model from scratch, 03: PPO training loop, 04: DPO training, 05: Constitutional AI self-critique)
- **Exercises:** 5 files (01: preference data, 02: reward scoring, 03: PPO basics, 04: DPO loss, 05: alignment evaluation)
- **Projects:** 3 files
  - Reward Model Trainer (full pipeline: preference data → train → evaluate → rank responses)
  - Full RLHF Pipeline (SFT → Reward Model → PPO, all 3 phases end-to-end)
  - Alignment Evaluator (dashboard comparing base/SFT/RLHF/DPO models across 5 categories)
- **Key topics:** RLHF 3-phase pipeline, Bradley-Terry loss, PPO clipping, KL divergence penalty, DPO log-ratio loss, Constitutional AI, RLAIF, self-critique loop, red-teaming, alignment metrics
- **NumPy + PyTorch:** All 5 examples have both NumPy (from scratch) and PyTorch (nn.Module) implementations
- **Status:** Complete

---

## Skills Mastered

| Area | Skills |
|------|--------|
| AI / ML | Neural networks from scratch, transformer architecture, attention mechanisms, RLHF, fine-tuning |
| LLM | Tokenization, embeddings, GPT architecture, text generation strategies, prompt engineering |
| Reasoning | Chain-of-Thought, Tree-of-Thoughts, o1-style systems, process supervision |
| Code AI | AST analysis, code embeddings, code generation, semantic search |
| Frameworks | NumPy, PyTorch (nn.Module, autograd), TensorFlow/Keras |
| Production | FastAPI, Docker, Kubernetes, PostgreSQL, Redis, Prometheus, Grafana |
| Vector DB | ChromaDB, cosine similarity, RAG pipeline, semantic search, search evaluation |
| Agents | ReAct pattern, tool use, function calling, short/long-term memory, multi-agent orchestration |
| Fine-Tuning | Transfer learning, LoRA/PEFT, dataset prep (JSONL/Alpaca), training loop, early stopping, evaluation metrics |
| Alignment | RLHF 3-phase pipeline, Bradley-Terry loss, PPO clipping+KL penalty, DPO log-ratio loss, Constitutional AI, RLAIF, self-critique, red-teaming |
| Production | JWT auth, rate limiting, connection pooling, caching, load balancing, monitoring, prompt injection scanning, PII redaction, multi-tenancy |

---

## New Modules (Planned — implement slowly)

### Module 02.5 — Classical Machine Learning
**Why here:** Bridges M02 (NumPy math) and M03 (Neural Networks). Regression and trees are simpler models — understanding them makes neural nets more meaningful ("why go deeper?"). Also unlocks Project A (House Price Prediction).

| Lesson | Topic |
|--------|-------|
| L1 | Multiple linear regression — matrix form `w = (XᵀX)⁻¹Xᵀy`, gradient descent version, R²/MAE/RMSE |
| L2 | Polynomial & regularized regression — feature engineering, Ridge (L2), Lasso (L1), overfitting demo |
| L3 | Decision trees — Gini impurity, information gain, how splits are chosen, depth vs overfitting |
| L4 | Random forests — bagging, bootstrap samples, feature importance, out-of-bag error |
| L5 | Model selection — when classical ML beats neural nets, cross-validation, bias-variance tradeoff |

**Libraries:** `numpy`, `sklearn`, `matplotlib`
**C# analogies:** Regression ↔ linear equation solver; Decision tree ↔ nested if/else rules; Random forest ↔ committee voting pattern
**Connects to:** M02 (matrix math), M03 (why neural nets go further), Project A (House Price Prediction)
**Status:** 📅 Planned

---

### Module 01.5 — Pandas & Data Handling
**Why here:** Data cleaning is needed before NumPy/PyTorch touches training data. Also used in Module 06 (dataset prep) and Module 12 (JSONL building).
- DataFrames: load CSV, JSON, JSONL files
- Filter, sort, group (LINQ equivalents)
- Missing data handling
- Export: CSV, JSON, JSONL (for LLM training datasets)
- Libraries: `pandas`, `matplotlib` (basic plots)
- C# analogy: `DataTable` + `LINQ` + CSV helper in one

### Module 05.5 — HuggingFace Tokenizers (Optional)
**Why here:** After building BPE from scratch in Module 05, see how production tokenizers work.
- BPE and WordPiece from HuggingFace `tokenizers` library
- Load pretrained tokenizer (GPT-2, BERT)
- Train custom tokenizer on your own corpus
- Compare scratch-built vs HuggingFace output
- Libraries: `tokenizers`, `datasets` (HuggingFace)

### Module 10.5 — RAG Without Vectors (BM25 / TF-IDF)
**Why here:** RAG concept before adding vector DB complexity. Works without GPU or embedding model.
- TF-IDF retrieval (sklearn) — keyword frequency-based
- BM25 search (`rank-bm25`) — smarter keyword ranking
- Hybrid search: BM25 + dense vectors combined
- When to use BM25 vs vector RAG (tradeoffs)
- C# analogy: BM25 ≈ SQL full-text search; Vector RAG ≈ ML-powered semantic search
- Libraries: `rank-bm25`, `sklearn`

### Module 14 — Deploying LLMs
- **Lessons:** 6 files (quantization concepts, GGUF format, TorchAO, ONNX export, Ollama+FastAPI, KV Cache & Inference Optimization)
- **Examples:** 5 files (L1–L5 done; L6 example pending)
  - 01: INT8/INT4 quantization from scratch, group quantization, memory math
  - 02: GGUF file structure, magic number check, llama-cpp-python load + chat
  - 03: TorchAO quantize_() API, INT8/INT4, memory measurement, fp32 vs INT8 comparison
  - 04: ONNX export, dynamic axes, ONNX Runtime inference, PyTorch vs ORT verification, C# integration
  - 05: Ollama 3 call methods, FastAPI server with streaming/auth/rate limiting/logging
  - 06: KV cache simulation (manual K/V store, reuse vs recompute, memory cost formula), paged attention concept, INT8 KV cache
- **Key topics:** PTQ, group quantization, GGUF Q4_K_M, llama.cpp, TorchAO, ONNX Runtime,
  OpenAI-compatible API, FastAPI Pydantic models, SSE streaming, rate limiting,
  KV cache (past_key_values), memory cost formula, paged attention, INT8 KV cache quantization
- **All examples:** Part A = numpy/stdlib only. Part B = real library with graceful fallback.
- **Exercises:** 5 files
  - 01: Model size calc, INT8 symmetric, asymmetric zero-point, group quantization
  - 02: GGUF file validator (magic bytes + version), quantization advisor, multi-turn chat
  - 03: Per-column INT8 quant, model_size_mb(), apply TorchAO, inspect quantized layers
  - 04: Concept questions, export with correct params, verify ORT vs PyTorch, fix dynamic axes bug
  - 05: OllamaClient class, prompt templates, FastAPI with Pydantic, rate limiter, streaming
- **Status:** ✅ Complete (2026-05-29). All 6 lessons, 6 examples, 6 exercises done.

### Module 14.5 — Gradio & Streamlit UIs (Optional)
**Why here:** Quick demo UIs for LLM projects — no frontend/React needed.
- Gradio: chat interfaces, file upload, model demos
- Streamlit: dashboards, data apps
- Deploy to HuggingFace Spaces (free hosting)
- C# analogy: Blazor without the complexity
- Libraries: `gradio`, `streamlit`

---

## What to Do Next

### Priority 1 — New Modules (implement soon, ordered by impact)
```
✅ M13++ RLVR/GRPO          ← DONE (2026-07-31): L6 RLVR, L7 GRPO, L8 Reasoning Chains
✅ M11++ MCP + LangGraph     ← DONE (2026-07-31): L6 MCP, L7 LangGraph, L8 Multi-Agent
✅ M19   Mixture of Experts  ← DONE (2026-07-31): 5L+5E+5X, Mini-MoE GPT trains in ~6s CPU

Next to implement:
1. M20   State Space Models  ← Mamba, O(n) attention alternative, hybrid SSM+Attn
2. M10.9 Multimodal RAG      ← ColPali: PDF pages as images, no OCR, visual doc search
3. M14++ Speculative Decoding ← 3x inference speedup, extends M14
4. M21   Vision Language Models ← patch embeddings + LLM = multimodal
```

### Priority 2 — Fill Gaps (ALL DONE 2026-07-31)
- ✅ Module 01: 10 exercises added
- ✅ Module 02.5: Full Classical ML module (5L+5E+5X, pure NumPy)
- ✅ Module 05.5: HuggingFace Tokenizers (5L+5E+5X, HF optional)
- Module 03.5: Could add more examples/exercises (low priority)

### Priority 3 — Standalone Projects (all unlocked)
- Project A: House Price Prediction (unlock: M03 done ✅)
- Project B: Movie Recommendation (unlock: M05 + M10 done ✅)
- Project C: Object Detection (unlock: M14 done ✅)

---

### Module 15 — Advanced LLM Training
- **Lessons:** 6 files (Chinchilla scaling laws, mixed precision bf16/fp16, Flash Attention, gradient checkpointing + ZeRO, dataset streaming, knowledge distillation)
- **Examples:** 6 files (01: scaling laws + compute budget, 02: bf16/fp16 mixed precision, 03: Flash Attention O(N) vs O(N²), 04: ZeRO + gradient checkpointing, 05: HuggingFace streaming datasets, 06: teacher→student distillation)
- **Exercises:** 6 files
- **Key topics:** Chinchilla compute-optimal formula, loss scaling, Flash Attention blocked SRAM, ZeRO stages 1-3, DeepSpeed, `load_dataset(..., streaming=True)`, distillation KL divergence + temperature softening
- **Status:** ✅ Complete (2026-05-30)

---

### Module 16 — Modern LLM Architectures
- **Location:** `modules/16_modern_architectures/`
- **Lessons (5):**
  - L1: Auto-regressive recap + limitations (sequential, no revision, left-to-right bias)
  - L2: Diffusion process for text (masking as noise, forward/reverse, parallel generation)
  - L3: Diffusion Language Models — MDLM, SEDD, Plaid (absorbing states, vocab transitions, production scale)
  - L4: Diffusion loss function (ELBO, VLB, denoising score matching, weighted cross-entropy on masked positions)
  - L5: Multi-Token Prediction (MTP) — N heads, richer gradients, speculative decoding, Meta results
- **Examples (5):**
  - `example_01_ar_limits.py` — Sequential steps, no-revision demo, left-to-right bias, AR vs diffusion vs MTP table
  - `example_02_diffusion_text.py` — Forward masking (linear + cosine schedule), closed-form sampling, reverse unmasking, step count comparison
  - `example_03_diffusion_models.py` — MDLM absorbing state, SEDD vocab transitions, confidence-ordered unmasking, MDLM vs SEDD vs Plaid comparison
  - `example_04_elbo_loss.py` — Why log P(x) is intractable, GPT vs MDLM loss side-by-side, loss vs noise level, ELBO decomposition
  - `example_05_mtp.py` — Standard vs MTP training, N=4 heads, richer gradient signal, speculative decoding speedup, MTP loss formula with weights
- **Exercises (5):**
  - `exercise_01_ar_limits.py` — count_ar_passes, diffusion_passes, speedup_ratio, has_right_context
  - `exercise_02_diffusion_schedule.py` — linear_schedule, cosine_schedule, apply_mask, count_masked
  - `exercise_03_mdlm.py` — is_absorbed, absorb_token, mdlm_forward, confidence_unmask
  - `exercise_04_elbo_loss.py` — softmax, cross_entropy, mdlm_loss_at_step, training_loss
  - `exercise_05_mtp.py` — get_target_token, head_loss, mtp_loss, gradient_signal_count
- **Key topics:** AR limits, masking diffusion, ELBO/VLB, MDLM absorbing state, SEDD score-based, MTP heads, speculative decoding, Pass@k for MTP inference
- **Status:** ✅ Complete (2026-06-30). All 5 lessons + 5 examples + 5 exercises done.

---

### Module 17 — LLM Evaluation & Benchmarks
- **Location:** `modules/17_evaluation/`
- **Lessons (5):**
  - L1: Training metrics — cross-entropy loss, perplexity = exp(loss), overfitting detection (val loss rises)
  - L2: General benchmarks — MMLU (57 subjects), HellaSwag (adversarial common sense), ARC-Challenge (science)
  - L3: Task benchmarks — GSM8K (math word problems), HumanEval (Pass@k, code tests), TruthfulQA (factuality)
  - L4: Text quality metrics — BLEU (precision, n-gram), ROUGE (recall, summarization), BERTScore (semantic)
  - L5: Evaluation in practice — lm-evaluation-harness, OpenLLM Leaderboard, pitfalls (contamination, shot count)
- **Examples (5):**
  - `example_01_perplexity.py` — exp(loss) conversions, good training vs overfitting curves, automatic detection, BPC
  - `example_02_general_benchmarks.py` — MMLU-style MCQ eval, accuracy + random baseline, real model leaderboard table
  - `example_03_task_benchmarks.py` — GSM8K answer extraction, HumanEval Pass@k formula + table, TruthfulQA myth scoring
  - `example_04_text_metrics.py` — BLEU n-gram precision (clipping demo), ROUGE-1/2/L, simulated BERTScore
  - `example_05_evaluation_practice.py` — Leaderboard reading, before/after fine-tuning analysis, 3 pitfalls (0-shot vs 5-shot, contamination, single benchmark)
- **Exercises (5):**
  - `exercise_01_perplexity.py` — compute_perplexity, detect_overfitting, bits_per_character, overfitting_gap
  - `exercise_02_benchmarks.py` — evaluate_multiple_choice, compute_accuracy, random_baseline, margin_over_random
  - `exercise_03_pass_at_k.py` — pass_at_k, pass_at_1, required_attempts, benchmark_pass_at_k
  - `exercise_04_text_metrics.py` — count_ngrams, ngram_precision, ngram_recall, compute_bleu
  - `exercise_05_model_comparison.py` — normalize_score, rank_models, compare_zero_few_shot, detect_specialist
- **Key topics:** Perplexity scale, overfitting curves, 25% random baseline, Pass@1 vs Pass@k, BLEU clipping, ROUGE-1/2/L, BERTScore cosine similarity, `lm_eval` CLI, data contamination, 0-shot vs 5-shot
- **Libraries:** `evaluate`, `lm-eval`, `datasets`, `torch`, `numpy`, `matplotlib`
- **Status:** ✅ Complete (2026-06-30). All 5 lessons + 5 examples + 5 exercises done.

---

---

### Module 18 — Qwen3.5 LLM from Scratch (PyTorch)
- **Location:** `modules/18_qwen3_from_scratch/`
- **Why Qwen3.5:** Production-grade open-source LLM. Combines modern techniques (RoPE, GQA, RLA) missing from classic GPT. Building it from scratch cements everything learned in M03–M15.
- **Lessons (5):**
  - L1: RoPE — Rotary Position Embeddings (replaces sinusoidal PE from M04)
  - L2: Group-Query Attention (GQA) — fewer KV heads, same Q heads → faster inference, less VRAM
  - L3: Recurrent Linear Attention (RLA) — O(1) memory per token, no quadratic attention, hybrid mode
  - L4: KV Cache Management — cache past K/V tensors, paged attention, memory budget, eviction
  - L5: Decoder Block + Full Qwen3.5 Assembly — stack all pieces, generate text, compare to nanoGPT
- **Examples (5, planned):**
  - 01: RoPE from scratch — rotation matrix math, compare sinusoidal vs RoPE position, visualize
  - 02: GQA from scratch — full MHA vs GQA vs MQA, head count math, VRAM savings formula
  - 03: Recurrent Linear Attention — linear attention kernel, recurrent form, hybrid attention block
  - 04: KV Cache — manual cache implementation, paged cache, INT8 KV cache quantization
  - 05: Full Qwen3.5 decoder — assemble L1–L4 into working model, generate text token-by-token
- **Exercises (5, planned):**
  - 01: RoPE — implement rotate_half(), apply RoPE to Q and K, verify position invariance
  - 02: GQA — implement KV head expansion (repeat_kv), measure memory vs MHA
  - 03: Linear Attention — implement linear attention from scratch, compare output to softmax attention
  - 04: KV Cache — build KVCacheManager class, measure tokens/sec with and without cache
  - 05: Full decoder — wire all components, count parameters, run forward pass on dummy input
- **Project (1, planned):**
  - Mini-Qwen: 6-layer, 256-dim Qwen3.5-style decoder trained on Shakespeare (~10M params)
    - Uses RoPE + GQA + KV Cache
    - Generates text and compares quality vs nanoGPT from M05
    - Benchmarks: tokens/sec, VRAM usage, perplexity
- **Key topics:** RoPE rotation matrix, θ frequencies, GQA head groups, repeat_kv, linear attention O(N) kernel, recurrent state, KV cache past_key_values, paged attention pages, memory formula, decoder stack, causal mask, RMSNorm, SwiGLU activation
- **Prerequisites:** M04 (Transformers), M05 (Building LLM), M14 (KV Cache concept), M15 (Flash Attention)
- **C# analogies:**
  - RoPE ↔ injecting position info via complex multiplication (like a hash of index into a value)
  - GQA ↔ sharing a read-only resource (KV heads) across multiple consumers (Q heads) — like a static readonly field
  - KV Cache ↔ `Dictionary<int, (K, V)>` that grows with each token, read-only after written
- **Libraries:** `torch`, `torch.nn`, `numpy`, `matplotlib`
- **Examples (5):**
  - `example_01_rope.py` — theta freq table, 2D rotation, relative position verification (dot product depends only on gap)
  - `example_02_gqa.py` — MHA/GQA/MQA memory math, repeat_kv, GQA attention computation, weight sizes
  - `example_03_rla.py` — phi kernel (ELU+1), hidden state accumulation S = S + phi(K)⊗V, full RLA step
  - `example_04_kv_cache.py` — SimpleKVCache class, sliding window eviction, INT8 quantization + error
  - `example_05_qwen_assembly.py` — RMSNorm vs LayerNorm, SwiGLU vs GELU, architecture comparison table
- **Exercises (5):**
  - `exercise_01_rope.py` — compute_theta_freqs, rotate_pair, apply_rope, verify_relative_position
  - `exercise_02_gqa.py` — compute_kv_cache_bytes, gqa_group_size, expand_kv_heads, weight_param_count
  - `exercise_03_rla.py` — phi, outer_product, update_hidden_state, rla_step
  - `exercise_04_kv_cache.py` — KVCache class (append/size_bytes/attend), sliding_window_evict, int8_quantize
  - `exercise_05_assembly.py` — rms_norm, silu, swiglu_ffn, count_block_params
- **Project (1):**
  - `mini_qwen.py` — 4-layer 64-dim Qwen3.5-style decoder (pure Python, no deps), GQA, RoPE, SwiGLU, RMSNorm, tied embeddings, autoregressive generation
- **Status:** ✅ Complete (2026-06-30). All 5 lessons + 5 examples + 5 exercises + 1 Mini-Qwen project done.

---

## Standalone Capstone Projects (Post-Course)

Build these AFTER all modules (M01–M17 + LLM Capstone) are complete.
Each project is self-contained and reinforces a different ML discipline.

---

### Project A — Beginner: House Price Prediction
**Unlock after:** M01.5 (Pandas), M02 (NumPy), M03 (Neural Networks)
**Status:** 🔒 Locked

**Goal:** Build a model that predicts house prices from structured data.

| Topic | What You Learn |
|-------|---------------|
| Data cleaning | Handle missing values, outliers, wrong types |
| Feature engineering | Create new features (price_per_sqft, age of house), encode categoricals |
| Regression | Linear regression, polynomial regression, decision tree regressor |
| Model building | Train/test split, cross-validation, hyperparameter tuning |
| Evaluation | MAE, RMSE, R² score — what they mean and how to improve them |

**Dataset features:**
- `bedrooms` (int), `bathrooms` (float), `sqft_living` (int), `sqft_lot` (int)
- `floors` (float), `waterfront` (0/1), `view` (0–4), `condition` (1–5)
- `grade` (1–13), `zipcode` (categorical), `yr_built`, `yr_renovated`
- `lat`/`long` (geolocation), `price` (target variable)

**Build plan:**
```
1. Load & explore   -- pandas, describe(), missing value heatmap
2. Clean data       -- fill nulls, remove outliers (z-score or IQR)
3. Feature engineer -- log(price), price_per_sqft, house_age, is_renovated
4. Encode           -- one-hot zipcode, ordinal grade/condition
5. Split            -- 80/20 train/test, stratified by price bucket
6. Baseline model   -- Linear regression (sklearn)
7. Better model     -- Gradient boosting (XGBoost or sklearn GBR)
8. Neural network   -- PyTorch MLP regressor (connects to M03)
9. Evaluate         -- compare all 3 models on test set
10. Visualize       -- actual vs predicted scatter, feature importance
```

**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `torch`

---

### Project B — Intermediate: Movie Recommendation System
**Unlock after:** M05 (Embeddings), M10 (Vector DBs), M02 (NumPy)
**Status:** 🔒 Locked

**Goal:** Build a system that recommends movies a user will likely enjoy.

| Topic | What You Learn |
|-------|---------------|
| Matrix factorization | Decompose user×movie rating matrix into latent factors |
| Collaborative filtering | "Users who liked X also liked Y" |
| Content-based filtering | Recommend by movie similarity (genre, director, cast) |
| Vector similarity | Cosine similarity between user/movie embedding vectors |
| Evaluation | Precision@K, Recall@K, NDCG — standard recommender metrics |

**Dataset:** MovieLens (free, publicly available)
- `ratings.csv` — (userId, movieId, rating, timestamp)
- `movies.csv` — (movieId, title, genres)
- Small version: 100K ratings, 9K movies, 600 users

**Build plan:**
```
Part A: Collaborative Filtering
1. Load MovieLens ratings into user×movie matrix
2. Compute user-user similarity (cosine similarity matrix)
3. Predict: weighted average of similar users' ratings
4. Evaluate: Precision@10 on held-out test ratings

Part B: Matrix Factorization (SVD)
1. Decompose rating matrix R ≈ U × S × Vt  (numpy.linalg.svd)
2. User embedding = row of U matrix
3. Movie embedding = row of V matrix
4. Predict rating = user_embedding · movie_embedding
5. Compare accuracy vs Part A

Part C: Neural Matrix Factorization
1. Build PyTorch model with nn.Embedding for users and movies
2. Train on (user, movie, rating) triplets with MSE loss
3. After training, user/movie embeddings are learned latent factors
4. Recommendation = top-K movies by dot product with user embedding
5. Connection to M10: store movie embeddings in ChromaDB for fast search

Part D: Content-Based (bonus)
1. TF-IDF on movie titles + genres
2. Cosine similarity between movie vectors
3. "If you liked movie X, you'll like movies similar to X"
```

**Libraries:** `pandas`, `numpy`, `torch`, `sklearn`, `chromadb`
**Key insight:** The movie embedding table in Part C is IDENTICAL to the token
embedding table in GPT. Same nn.Embedding, different domain.

---

### Project C — Advanced: Real-Time Object Detection System
**Unlock after:** M03.5 (PyTorch), M04 (Transformers), M14 (Deploying LLMs)
**Status:** 🔒 Locked

**Goal:** Build a system that detects and labels objects in images or video in real time.

| Topic | What You Learn |
|-------|---------------|
| Computer vision | How images become tensors, convolutions, feature maps |
| Object detection | Bounding boxes, anchor boxes, IoU (Intersection over Union) |
| Pre-trained models | YOLO, DETR (Detection Transformer) — fine-tune vs use as-is |
| Real-time processing | Webcam input, FPS optimization, frame buffering |
| Model optimization | INT8 quantization for speed (M14), TorchAO, ONNX export |

**Build plan:**
```
Part A: Vision Fundamentals
1. Image as tensor -- load with PIL/OpenCV, convert to (C, H, W) tensor
2. Convolution from scratch -- NumPy 2D convolution, edge detection filters
3. Feature maps -- visualize what CNN layers "see"
4. Object detection concepts -- bounding boxes, IoU, NMS (non-max suppression)

Part B: Use a Pre-Trained Detector
1. Load YOLOv8 (ultralytics) or DETR (HuggingFace transformers)
2. Run on sample images -- get bounding boxes + class labels + confidence
3. Draw boxes on image with matplotlib/OpenCV
4. Benchmark: images/second on CPU vs GPU

Part C: Real-Time Webcam Detection
1. OpenCV VideoCapture for webcam input
2. Per-frame detection loop -- feed frame to model, draw results
3. FPS counter -- measure actual real-time performance
4. Optimization: resize frames, skip frames, batch frames

Part D: Model Optimization (connects to M14)
1. Export to ONNX -- run ONNX Runtime vs PyTorch, compare speed
2. INT8 quantization with TorchAO -- measure FPS improvement
3. Optional: deploy as FastAPI service with image upload endpoint

Part E: Fine-Tune on Custom Data (advanced bonus)
1. Label custom images with LabelImg tool
2. Fine-tune YOLO on custom classes (e.g., detect your own objects)
3. Evaluate: mAP (mean Average Precision) on test set
```

**Libraries:** `torch`, `torchvision`, `opencv-python`, `ultralytics` (YOLOv8),
             `transformers` (DETR), `onnx`, `onnxruntime`, `torchao`, `PIL`

**Key connections to course:**
| This project | Module it connects to |
|---|---|
| Image tensors (C, H, W) | M02 NumPy arrays, M03.5 PyTorch tensors |
| Attention in DETR | M04 Transformer architecture |
| Fine-tuning YOLO | M12 Fine-tuning techniques + LoRA |
| INT8/ONNX export | M14 Deploying LLMs (same techniques!) |
| FastAPI endpoint | M09 Production LLM Apps |

---

## Full Project Roadmap (All Capstones)

```
After M03           --> Project A: House Price Prediction (beginner ML)
After M10/M05       --> Project B: Movie Recommendation (intermediate ML)
After M14/M17       --> Project C: Object Detection (advanced CV)
After M16/M17/M18   --> LLM Capstone: Chat with Codebase (offline RAG app)
```

All 4 projects together = full-stack ML portfolio covering:
- Classical ML (regression, recommender systems)
- Computer vision (detection, real-time)
- LLM applications (RAG, local deployment)

---

## New Module Detail Plans (added 2026-07-31)

---

### Module 13++ — RLVR / GRPO Extension (URGENT)
**Location:** Add to `modules/13_rlhf_alignment/` as L6, L7, L8
**Why urgent:** DeepSeek-R1 (Jan 2025) proved you can train reasoning models with ZERO supervised examples — only RL with verifiable rewards. Every top model in 2025-2026 (Qwen3, Llama4, Kimi) uses this. Already planned in M13; now critical.

**Core idea:**
- RLVR = reward signal comes from checking if the answer is CORRECT (math: check number, code: run tests). No human labelers needed.
- GRPO = Group Relative Policy Optimization. Sample 8-16 responses to the same prompt, score them all, use relative ranking as reward. No separate value/critic network needed (cheaper than PPO).
- Result: extended chain-of-thought thinking, self-correction, and reasoning emerge purely from RL.

| Lesson | Topic |
|--------|-------|
| L6 | RLVR — verifiable rewards, no human labels, math/code use cases, contrast with RLHF |
| L7 | GRPO algorithm — group sampling, relative reward, advantage = (score - group_mean) / group_std |
| L8 | Reasoning chains from RL — how extended thinking emerges, "aha moment" phenomenon, R1 results |

**C# analogy:** GRPO ≈ running a unit test suite and using pass-rate as the only feedback signal — no human reviewer needed.
**Libraries:** `torch`, `numpy`
**Prerequisites:** M13 (PPO, DPO, reward models)

---

### Module 11++ — MCP + LangGraph Extension ✅ COMPLETE (2026-07-31)
**Location:** `modules/11_llm_agents/` — L6, L7, L8 + 3 examples + 3 exercises
**Why:** MCP (Model Context Protocol, Anthropic Nov 2024) is now the industry standard for agent↔tool connections — adopted by OpenAI, Google, Microsoft, donated to Linux Foundation Dec 2025. 65% of LLM job listings require it. LangGraph replaced simple chain-based agents as the standard orchestration pattern.

**What was built:**

| File | Description |
|------|-------------|
| `lessons/06_mcp_protocol.md` | MCP M×N problem, server/client, tool schemas, tool discovery, MCP vs function calling |
| `lessons/07_langgraph.md` | StateGraph, nodes, edges, conditional routing, retry loops, checkpointing, human-in-the-loop |
| `lessons/08_multi_agent_mcp_langgraph.md` | Router→specialist pattern, interrupt gates, why this is in 65% of job postings |
| `examples/example_06_mcp_protocol.py` | Simulated MCP server+client, 3 servers (calc/filesystem/github), discovery demo |
| `examples/example_07_langgraph.py` | Simulated StateGraph, conditional retry loop, checkpointing (pause/resume), trace |
| `examples/example_08_multi_agent_mcp_langgraph.py` | Router + 2 specialists, interrupt gate, full execution trace |
| `exercises/exercise_06_mcp.py` | Build MCPServer + MCPClient + register tools + solve tasks via MCP |
| `exercises/exercise_07_langgraph.py` | Node functions + run_graph + run_graph_with_conditions + full pipeline |
| `exercises/exercise_08_multi_agent.py` | router_node + research_agent + math_agent + run_multi_agent_graph |

**Key concepts delivered:**
- MCP solves M×N integration problem: write tools once, any agent uses them
- tools/list = discovery, tools/call = execution, tool schema = JSON Schema
- LangGraph: State dict flows through nodes; conditional edges enable retry loops
- Checkpointing = pause/resume (like Azure Durable Functions replay)
- interrupt_before=["node"] = human-in-the-loop approval gate
- Router pattern: supervisor LLM routes to specialist agent via conditional edge

**C# analogies used:**
- MCP Server ↔ gRPC/WCF service; MCP Client ↔ auto-generated typed client from Swagger
- LangGraph ↔ Azure Durable Functions (explicit state, resumable, auditable)
- Router ↔ MediatR command dispatcher / CQRS
- Interrupt gate ↔ context.WaitForExternalEvent<bool>()

**All examples:** Pure Python stdlib — no pip install needed. Migration path to real `langgraph` documented inline (same API).
**Prerequisites:** M11 (agents, tool use, ReAct, multi-agent)

---

### Module 19 — Mixture of Experts (MoE)
**Location:** `modules/19_mixture_of_experts/`
**Key insight:** Standard LLM activates ALL parameters for every token. MoE activates only K of N expert FFN layers per token. DeepSeek-V3: 671B total params, 37B active per token → same quality, 18x less compute per forward pass.

| Lesson | Topic |
|--------|-------|
| L1 | MoE concept — experts, router, sparse activation, why it scales (compute vs capacity) |
| L2 | Router network — Top-K gating with softmax, how tokens are assigned to experts |
| L3 | Load balancing — auxiliary loss prevents all tokens routing to same expert (collapse problem) |
| L4 | MoE vs dense tradeoffs — more memory (store all experts) vs less compute (only run K) |
| L5 | Build Mini-MoE GPT — replace FFN layers with 4 experts + router, train on Shakespeare |

**Project:** Mini-MoE (4 experts, top-2 routing, 4-layer transformer) — compare perplexity and training speed vs dense nanoGPT from M05
**C# analogy:** Router ≈ a strategy pattern that selects which implementation handles each request. Experts ≈ specialized service classes.
**Libraries:** `torch`, `numpy`
**Prerequisites:** M04 (Transformers), M05 (Building LLM), M18 (Qwen3 from scratch)

---

### Module 20 — State Space Models (Mamba / SSM)
**Location:** `modules/20_state_space_models/`
**Key insight:** Attention is O(n²) in sequence length — 4x longer sequence = 16x more compute. SSMs are O(n) linear. Mamba adds selectivity (input-dependent state transitions). Hybrid models (1/4 attention + 3/4 SSM) give 3x throughput at same quality.

| Lesson | Topic |
|--------|-------|
| L1 | The O(n²) problem — memory and compute cost of attention at long contexts |
| L2 | State Space Models — continuous-time system `x' = Ax + Bu`, discretization (ZOH), recurrent form |
| L3 | Mamba / selective SSM — input-dependent A, B, C matrices, hardware-efficient parallel scan |
| L4 | Mamba-2 — structured state spaces (diagonal A), SSD (State Space Duality), faster training |
| L5 | Hybrid Attention+SSM — why mix both (SSM good for recall, Attention good for in-context), Jamba pattern |
| L6 | Build Mini-Mamba — selective SSM layer from scratch in PyTorch, compare output to attention |

**Project:** Mini-Mamba language model — train on Shakespeare, compare tokens/sec vs nanoGPT at 1024 and 4096 seq length
**C# analogy:** SSM recurrent form ≈ a for-loop accumulating state (like `accumulator = f(accumulator, input[i])`). Parallel scan ≈ prefix sum / scan operation (PLINQ Aggregate with associative combiner).
**Libraries:** `torch`, `numpy`
**Prerequisites:** M04 (Transformers), M15 (Advanced Training — Flash Attention concepts)

---

### Module 10.9 — Multimodal RAG (ColPali)
**Location:** `modules/10.9_multimodal_rag/`
**Key insight:** Traditional RAG on PDFs: extract text → embed text → search text. Loses layout, tables, charts, diagrams, and anything OCR gets wrong. ColPali: render PDF page as IMAGE → embed image patches → multi-vector late interaction (MaxSim scoring). No OCR needed. Works on scanned docs, slides, engineering drawings.

| Lesson | Topic |
|--------|-------|
| L1 | Why text-only RAG fails on visual documents — OCR errors, lost tables, lost diagrams |
| L2 | Image patch embeddings — ViT splits image into 16×16 patches, each patch becomes a vector |
| L3 | Late interaction / MaxSim scoring — multi-vector query vs multi-vector doc (ColBERT style for images) |
| L4 | ColPali architecture — PaliGemma backbone + BiPali retriever, training on DocVQA |
| L5 | Build visual document search — render PDF pages as images, embed patches, retrieve by question |

**C# analogy:** Traditional RAG ≈ full-text search on extracted text. ColPali ≈ image recognition + semantic search combined — like asking "find the slide that shows this graph" without needing text extraction.
**Libraries:** `Pillow`, `torch`, `chromadb`, `pypdf2` or `pdf2image`
**Prerequisites:** M10 (Vector DBs), M10.8 (Semantic Search), M04 (Transformers — ViT is a transformer)

---

### Module 14++ — Speculative Decoding Extension
**Location:** Add to `modules/14_deploying_llms/` as L7, L8, L9
**Key insight:** LLM generates one token at a time (autoregressive = slow). Speculative decoding uses a small DRAFT model to guess N tokens at once, then the large TARGET model verifies ALL N in a single forward pass. If the draft is right (~80% acceptance with EAGLE-3), you get N tokens for the cost of ~1.3. Result: 3x throughput with zero quality loss.

```
Standard:   LLM → token1, LLM → token2, LLM → token3   (3 forward passes)
Speculative: draft → [t1,t2,t3], LLM verifies all 3 at once (1 forward pass)
             If all accepted: 3 tokens for cost of 1 LLM pass
```

| Lesson | Topic |
|--------|-------|
| L7 | Speculative decoding — draft + verify loop, acceptance criterion, guaranteed same output distribution |
| L8 | EAGLE-3 — draft at the feature level (not token level), 80% acceptance rate, 3x throughput on H200 |
| L9 | Combining with quantization — QSpec pattern: INT4 quant + speculative decoding = 6x combined improvement |

**C# analogy:** Speculative execution ≈ CPU branch prediction — guess ahead, execute speculatively, roll back on misprediction. But here: no rollback, just reject the wrong tokens.
**Libraries:** `torch`, `transformers` (for reference)
**Prerequisites:** M14 (Deploying LLMs — quantization, KV Cache), M18 (autoregressive generation)

---

### Module 21 — Vision Language Models (VLMs)
**Location:** `modules/21_vision_language_models/`
**Key insight:** Add vision to an LLM by treating image patches as tokens. ViT splits image into 16×16 pixel patches, projects each to embedding dimension, feeds into transformer alongside text tokens. The LLM "sees" the image as a sequence of special tokens. LLaVA, Qwen-VL, Phi-4-Multimodal, Gemma 3 all use this pattern.

```
Image (224×224 pixels)
  → split into 14×14 grid of 16×16 patches    = 196 patch tokens
  → project each patch to hidden_dim          = 196 × 768 vectors
  → concatenate with text tokens              = [patch_1, ..., patch_196, text_1, ...]
  → feed to transformer                       → answer about the image
```

| Lesson | Topic |
|--------|-------|
| L1 | Vision encoder — ViT architecture, patch embeddings, CLS token, positional embeddings for patches |
| L2 | Connecting vision to language — projection layer (linear or 2-layer MLP maps ViT dim → LLM dim) |
| L3 | VLM training — Stage 1: image captioning pretraining (freeze LLM, train projection); Stage 2: instruction tuning (train all) |
| L4 | Cross-attention for images — Flamingo-style: image features injected via cross-attention layers, not token concatenation |
| L5 | Build Mini-VLM — ViT encoder (tiny) + MLP projection + small LLM, answer yes/no questions about images |

**Project:** Mini-VLM that answers simple questions about images ("Is there a cat in this image?", "What color is the car?")
**C# analogy:** ViT patch embedding ≈ splitting a byte array into fixed-size chunks and hashing each chunk into a feature vector. Projection layer ≈ an adapter/converter between two different vector spaces.
**Libraries:** `torch`, `torchvision`, `Pillow`, `numpy`
**Prerequisites:** M04 (Transformers), M05 (Embeddings), M18 (building LLM), M03.5 (PyTorch)


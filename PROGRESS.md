# Learning Progress — LLM from Scratch

**Last Updated:** May 19, 2026 (Module 10.8 complete)
**Student:** .NET developer learning Python + LLMs simultaneously

---

## Module Overview

| # | Module | Lessons | Examples | Exercises | Projects | Status |
|---|--------|---------|----------|-----------|----------|--------|
| 01 | Python Basics | 10 | 10 | 0 | — | Content ready, exercises missing |
| 01.5 | Pandas & Data Handling | — | — | — | — | 📅 Planned |
| 02 | NumPy & Math | 3 | 4 | 3 | — | ✅ Complete |
| 03 | Neural Networks | 6 | 6 | 6 | 3 | ✅ Complete |
| 03.5 | PyTorch & TensorFlow | 5 | 2 | 1 | 1 | ✅ Complete (minimal exercises) |
| 04 | Transformers | 9 | 18 * | 3 | — | ✅ Complete (L7–9 concept-only) |
| 05 | Building Your LLM | 5 | 5 | 5 | 7 | ✅ Complete |
| 05.5 | HuggingFace Tokenizers | — | — | — | — | 📅 Planned (optional) |
| 06 | Training & Fine-tuning | 6 | 7 | — | 4 | ✅ Complete |
| 07 | Reasoning & Coding Models | 10 | 10 | — | 5 | ✅ Complete |
| 08 | Prompt Engineering | 10 | 3 | 2 | — | ✅ Complete |
| 09 | Production LLM Applications | 4 | 4 | 4 | 2 | ✅ Complete |
| 10 | Vector Databases | 5 | 5 | 3 | 2 | ✅ Complete |
| 10.5 | RAG Without Vectors (BM25/TF-IDF) | — | — | — | — | 📅 Planned |
| 10.8 | Semantic Search Systems | 6 | 6 | 5 | 1 | ✅ Complete |
| 11 | LLM Agents | 5 | 5 | 5 | 3 | ✅ Complete |
| 12 | Fine-Tuning LLMs | 5 | 5 | 5 | 3 | ✅ Complete |
| 13 | RLHF and Alignment | 5 | 5 | 5 | 3 | ✅ Complete |
| 14 | Deploying LLMs | — | — | — | — | 📅 Planned |
| 14.5 | Gradio & Streamlit UIs | — | — | — | — | 📅 Planned (optional) |

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
**Why here:** After RLHF/alignment, deploy optimized models.
- Quantization (float32 → int8, reduce model size 4x)
- ONNX export and inference optimization
- FastAPI streaming responses (SSE)
- Local model serving (Ollama, llama.cpp)
- Libraries: `onnx`, `fastapi`, `uvicorn`

### Module 14.5 — Gradio & Streamlit UIs (Optional)
**Why here:** Quick demo UIs for LLM projects — no frontend/React needed.
- Gradio: chat interfaces, file upload, model demos
- Streamlit: dashboards, data apps
- Deploy to HuggingFace Spaces (free hosting)
- C# analogy: Blazor without the complexity
- Libraries: `gradio`, `streamlit`

---

## What to Do Next

### Priority 1 — Module 14: Deploying LLMs
- Quantization, ONNX, FastAPI serving, streaming

### Priority 3 — New Modules (any order)
- Module 01.5 (Pandas) — good for data work
- Module 10.5 (BM25/TF-IDF RAG) — practical, no GPU needed
- Module 05.5 (HuggingFace Tokenizers) — optional deepdive
- Module 14.5 (Gradio/Streamlit) — fun, quick wins

### Priority 4 — Fill Gaps
- Module 01: Add exercises
- Module 03.5: Add more examples and exercises
- Module 06/07: Add exercises

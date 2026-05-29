# Learning Progress — LLM from Scratch

**Last Updated:** 2026-05-24 (Added 3 standalone capstone projects to roadmap)
**Student:** .NET developer learning Python + LLMs simultaneously

---

## Module Overview

| # | Module | Lessons | Examples | Exercises | Projects | Status |
|---|--------|---------|----------|-----------|----------|--------|
| 01 | Python Basics | 10 | 10 | 0 | — | Content ready, exercises missing |
| 01.5 | Pandas & Data Handling | 5 | 5 | 5 | — | ✅ Complete |
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
| 10.5 | RAG Without Vectors (BM25/TF-IDF) | 5 | — | — | — | ✅ Complete (theory only) |
| 10.8 | Semantic Search Systems | 6 | 6 | 5 | 1 | ✅ Complete |
| 11 | LLM Agents | 5 | 5 | 5 | 3 | ✅ Complete |
| 12 | Fine-Tuning LLMs | 5 | 5 | 5 | 3 | ✅ Complete |
| 13 | RLHF and Alignment | 5 | 5 | 5 | 3 | ✅ Complete |
| 14 | Deploying LLMs | 6 | 6 | 6 | — | ✅ Complete |
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

### Priority 1 — Module 15: Advanced LLM Training
- Flash Attention, ZeRO optimizer, bf16, Chinchilla scaling laws, distillation, dataset streaming

### Priority 3 — New Modules (any order)
- Module 01.5 (Pandas) — good for data work
- Module 10.5 (BM25/TF-IDF RAG) — practical, no GPU needed
- Module 05.5 (HuggingFace Tokenizers) — optional deepdive
- Module 14.5 (Gradio/Streamlit) — fun, quick wins

### Priority 4 — Fill Gaps
- Module 01: Add exercises
- Module 03.5: Add more examples and exercises
- Module 06/07: Add exercises

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
After ALL modules   --> LLM Capstone: Chat with Codebase (offline RAG app)
```

All 4 projects together = full-stack ML portfolio covering:
- Classical ML (regression, recommender systems)
- Computer vision (detection, real-time)
- LLM applications (RAG, local deployment)


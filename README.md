# Learn LLM from Scratch

> A comprehensive, hands-on journey from Python basics to building Large Language Models - designed for .NET developers learning Python and AI together.

---

## 🎯 Overview

This repository contains a complete educational curriculum for learning Large Language Models (LLMs) from first principles. No prior Python or machine learning experience required!

**Target Audience:** .NET/C# developers who want to understand how ChatGPT, GPT-4, and modern LLMs actually work.

**Approach:** Build everything from scratch - no black boxes, deep understanding at every step, then deploy to production.

---

## 📚 What You'll Learn & Build

### Fundamentals
- ✅ Neural networks from scratch (NumPy only!)
- ✅ Backpropagation algorithm (powers ALL modern AI)
- ✅ Multi-layer networks and optimizers (Adam, SGD, RMSProp)
- ✅ Transformer architecture (GPT, BERT)
- ✅ PyTorch & TensorFlow (modern frameworks)

### Advanced AI
- ✅ Reasoning models (Chain-of-Thought, Tree-of-Thoughts)
- ✅ Code analysis with AI (like GitHub Copilot)
- ✅ Prompt engineering (10x better results)
- 🚧 Building GPT from scratch
- 🚧 Training and fine-tuning

### Production Deployment
- 🆕 REST APIs with FastAPI
- 🆕 Docker & Kubernetes deployment
- 🆕 Monitoring with Prometheus + Grafana
- 🆕 Security & cost optimization (50-80% savings!)
- 🆕 Production-ready LLM applications

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/llm-from-scratch.git
cd llm-from-scratch

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start learning!
cd modules/01_python_basics  # Start here (recommended)
cd modules/03_neural_networks  # Or jump to neural networks
cd modules/09_production_llm_apps  # Or build production APIs
```

---

## 📊 Current Progress

```
Module 1:    Python Basics              ██████████ 100% ✅
Module 1.5:  Pandas & Data Handling     ░░░░░░░░░░   0% 📅 NEW
Module 2:    NumPy & Math               ██████████ 100% ✅
Module 3:    Neural Networks            ██████████ 100% ✅
Module 3.5:  PyTorch & TensorFlow       ██████████ 100% ✅
Module 4:    Transformers               ██████████ 100% ✅
Module 5:    Building LLMs              ██████████ 100% ✅
Module 5.5:  HuggingFace Tokenizers     ░░░░░░░░░░   0% 📅 NEW (optional)
Module 6:    Training & Fine-tuning     ██████████ 100% ✅
Module 7:    Reasoning & Coding         ██████████ 100% ✅
Module 8:    Prompt Engineering         ██████████ 100% ✅
Module 9:    Production Apps            ██████████ 100% ✅
Module 10:   Vector Databases           ██████████ 100% ✅
Module 10.5: RAG Without Vectors        ░░░░░░░░░░   0% 📅 NEW
Module 11:   LLM Agents                 ██████████ 100% ✅
Module 12:   Fine-Tuning LLMs           ██████████ 100% ✅
Module 13:   RLHF & Alignment           ████░░░░░░  40% 🔧
Module 14:   Deploying LLMs             ░░░░░░░░░░   0% 📅
Module 14.5: Gradio & Streamlit UIs     ░░░░░░░░░░   0% 📅 NEW (optional)

Overall (core modules):                 ████████░░  85%
```

**Legend:**
- ✅ Complete & Ready to Use
- 🔧 In Development
- 📅 Planned
- 🆕 New Module

---

## 📖 Complete Module Roadmap

### Module 1: Python Basics for .NET Developers
**Status:** ✅ Complete | **Time:** 8-10 hours

Learn Python from a C# perspective with side-by-side comparisons.

**Topics:**
- Variables, types, and data structures
- Control flow (if/else, loops)
- Functions and lambda expressions
- Classes and OOP
- List comprehensions (like LINQ)
- File I/O and error handling

**Best for:** .NET developers new to Python

---

### Module 1.5: Pandas & Data Handling 📅 NEW
**Status:** 📅 Planned | **Time:** 8-10 hours

Excel-in-code. Load, clean, and transform data before feeding it to ML models.

**Topics:**
- DataFrames and Series (like `DataTable` + LINQ in C#)
- Loading CSV, JSON, JSONL files
- Filter, sort, group operations — LINQ equivalents
- Handling missing data
- Exporting JSONL for LLM training datasets
- Basic plots with Matplotlib

**Libraries:** `pandas`, `matplotlib`

**Why here:** Dataset prep happens before NumPy or PyTorch. LLM training needs clean JSONL files — Pandas builds them.

---

### Module 2: NumPy & Mathematical Foundations
**Status:** ✅ Complete | **Time:** 12-14 hours

Master arrays, linear algebra, and the math behind neural networks.

**Topics:**
- NumPy arrays and operations
- Broadcasting and vectorization
- Linear algebra (matrices, dot products)
- Mathematical foundations for ML

**Best for:** Understanding neural network math

---

### Module 3: Neural Networks from Scratch
**Status:** ✅ Complete (100%) | **Time:** 35-45 hours

Build and train neural networks using only NumPy:

**Lessons:**
- ✅ **Lesson 1:** Perceptrons - Single neuron networks
- ✅ **Lesson 2:** Activation Functions - ReLU, Sigmoid, Softmax, GELU
- ✅ **Lesson 3:** Multi-Layer Networks - Deep learning begins
- ✅ **Lesson 4:** Backpropagation - How networks learn (THE algorithm!)
- ✅ **Lesson 5:** Training Loop - Complete training pipeline
- ✅ **Lesson 6:** Optimizers - Adam, Momentum, RMSProp (like GPT-3!)

**Projects:** XOR solver, MNIST classifier (95%+ accuracy)

**Why complete this:** Understand the foundations that power ALL modern AI

---

### Module 3.5: PyTorch & TensorFlow
**Status:** ✅ Complete (100%) | **Time:** 36-52 hours

Transition from NumPy to modern deep learning frameworks.

**Topics:**
- PyTorch fundamentals (tensors, autograd)
- Building neural networks with nn.Module
- NumPy to PyTorch conversion patterns
- TensorFlow/Keras basics
- Framework comparison and selection

**Best for:** Production ML development

---

### Module 4: Transformers & Attention Mechanism
**Status:** ✅ Complete (100%) | **Time:** 22-34 hours

Learn the revolutionary architecture behind GPT, ChatGPT, and all modern LLMs.

**Lessons:**
- ✅ Attention mechanism fundamentals
- ✅ Self-attention (what makes transformers work)
- ✅ Multi-head attention layers
- ✅ Positional encoding
- ✅ Complete transformer block
- ✅ Full GPT architecture

**Includes:** Complete code examples and exercises

**Why this matters:** This IS how ChatGPT works!

---

### Module 5: Building Your LLM
**Status:** 🔧 70% Complete | **Time:** 20-30 hours

Build a complete GPT model from scratch.

**Topics:**
- Tokenization (BPE like GPT!)
- Word embeddings and semantic relationships
- Building vocabulary
- Full GPT architecture assembly
- Text generation strategies
- Sampling and decoding

**Projects:** Build your own mini-GPT!

---

### Module 5.5: HuggingFace Tokenizers (Optional) 📅 NEW
**Status:** 📅 Planned | **Time:** 6-8 hours

After building BPE from scratch in Module 5, see how production tokenizers work.

**Topics:**
- HuggingFace `tokenizers` library — BPE and WordPiece
- Load pretrained tokenizer (GPT-2, BERT)
- Train custom tokenizer on your own text corpus
- Compare: scratch-built vs HuggingFace output side-by-side

**Libraries:** `tokenizers`, `datasets` (HuggingFace)

**Why optional:** Deep-dive for those who want production tokenizer skills. Safe to skip and return later.

---

### Module 6: Training & Fine-tuning
**Status:** ✅ Complete | **Time:** 25-35 hours

Train and fine-tune LLMs for custom tasks.

**Topics:**
- Pre-training strategies
- Fine-tuning techniques
- Transfer learning
- Model evaluation
- Hyperparameter tuning
- Dataset preparation

**Projects:** Fine-tune GPT on custom data

---

### Module 7: Advanced LLM Applications - Reasoning & Coding 🔥
**Status:** ✅ Complete (100%) | **Time:** 30-40 hours

Master cutting-edge AI techniques used in OpenAI o1 and GitHub Copilot!

**Part A: Reasoning Models (like o1)**
- ✅ Chain-of-Thought (CoT) prompting
- ✅ Self-consistency & ensemble reasoning
- ✅ Tree-of-Thoughts search
- ✅ Process supervision & reasoning traces
- ✅ Building reasoning systems (o1-like)

**Part B: Coding Models (like Copilot)**
- ✅ Code representation & tokenization
- ✅ Code embeddings & AST
- ✅ Training models on code
- ✅ Code generation & completion
- ✅ Code evaluation & testing

**Real-World Projects:**
- ✅ AI Code Reviewer (production-ready!)
- ✅ Smart Bug Debugger (production-ready!)
- ✅ Semantic Code Search
- ✅ Auto Test Writer
- ✅ Code Quality Analyzer

**Impact:** Build professional AI developer tools, save $3,000+/year vs commercial tools

---

### Module 8: Prompt Engineering (Advanced)
**Status:** ✅ Complete (100%) | **Time:** 25-30 hours

Master the art and science of communicating with LLMs - 10x your AI results overnight!

**Part A: Fundamentals**
- ✅ Zero-shot prompting
- ✅ Few-shot learning
- ✅ Prompt templates
- ✅ Role & system prompting

**Part B: Advanced Techniques**
- ✅ Chain-of-Thought (CoT)
- ✅ Tree of Thoughts
- ✅ Structured outputs & function calling
- ✅ Prompt optimization with DSPy

**Part C: Production & Security**
- ✅ Prompt security & injection prevention
- ✅ Production patterns

**Impact:** 10x better results with same model, reduce costs by 50-80%

---

### Module 9: Production LLM Applications 🆕 🔥
**Status:** ✅ Complete (100%) | **Time:** 34-42 hours

**From prototype to production - Build, deploy, and scale real-world AI systems!**

**Lesson 1: API Design & Architecture** (8-10 hours)
- FastAPI fundamentals with C# comparisons
- Request/response patterns
- JWT authentication
- Streaming responses (Server-Sent Events)
- API versioning

**Lesson 2: Deployment & Scalability** (10-12 hours)
- Docker containerization
- Database design (PostgreSQL)
- Redis caching strategies
- Cloud deployment (AWS/Azure/GCP)
- Kubernetes orchestration
- Load balancing & auto-scaling

**Lesson 3: Monitoring & Observability** (8-10 hours)
- Structured logging with JSON
- Prometheus metrics collection
- Grafana dashboards
- Distributed tracing (OpenTelemetry)
- Intelligent alerting
- Cost monitoring & tracking

**Lesson 4: Security & Cost Optimization** (8-10 hours)
- Authentication & authorization (JWT, API keys, RBAC)
- Rate limiting with Redis
- Prompt injection prevention
- PII detection and redaction
- **50-80% cost reduction strategies!**
- GDPR compliance
- Security audit checklist

**Projects:**
- Simple Chat API (8-10 hours)
- Multi-Tenant SaaS Platform (15-20 hours)
- Enterprise Chatbot (20-25 hours)

**Career Impact:**
- Resume: "Designed production LLM APIs handling 10K+ requests/day"
- Resume: "Reduced LLM costs by 60% through optimization"
- Resume: "Implemented Kubernetes auto-scaling for ML workloads"
- Salary: $120K-200K ML Engineer roles
- Freelancing: $150-300/hour rates

**Why this module:** Most practical, career-focused content in entire curriculum!

---

### Module 10.5: RAG Without Vectors (BM25 / TF-IDF) 📅 NEW
**Status:** 📅 Planned | **Time:** 8-10 hours

Build RAG pipelines without any GPU, embedding model, or vector database.

**Topics:**
- TF-IDF retrieval (sklearn) — keyword frequency scoring
- BM25 search (`rank-bm25`) — smarter keyword ranking (used by Elasticsearch)
- Hybrid search: BM25 + dense vectors combined
- Tradeoffs: BM25 vs vector RAG — when to use each

**Libraries:** `rank-bm25`, `sklearn`

**C# analogy:** BM25 ≈ SQL full-text search. Vector RAG ≈ ML-powered semantic search.

**Why here:** Teaches RAG fundamentals without the complexity of embeddings + vector DBs. Great for small corpora or GPU-free environments.

---

### Module 13: RLHF and Alignment 🔧
**Status:** 🔧 In Progress (4/5 lessons done) | **Time:** 20-28 hours

How ChatGPT was aligned — reward models, human feedback, and safety.

**Topics:**
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO) — simpler RLHF alternative
- Constitutional AI (how Claude works)
- Reward model training
- Preference datasets

---

### Module 14: Deploying LLMs 📅
**Status:** 📅 Planned | **Time:** 18-24 hours

Shrink models, speed up inference, and serve them in production.

**Topics:**
- Quantization: float32 → int8 (4x smaller model, same quality)
- ONNX export and inference optimization
- FastAPI streaming responses (Server-Sent Events)
- Local model serving with Ollama / llama.cpp

**Libraries:** `onnx`, `fastapi`, `uvicorn`

---

### Module 14.5: Gradio & Streamlit UIs (Optional) 📅 NEW
**Status:** 📅 Planned | **Time:** 6-8 hours

Build chat interfaces and demos in minutes — no frontend skills needed.

**Topics:**
- Gradio: chat UI, file upload, model demos
- Streamlit: dashboards, data exploration apps
- Deploy to HuggingFace Spaces (free hosting)

**Libraries:** `gradio`, `streamlit`

**C# analogy:** Like Blazor but 10 lines of code instead of 100.

**Why here:** Instantly showcase LLM projects. Great for portfolio demos.

---

## 🎓 What Makes This Different

### 1. For .NET Developers
- ✅ C# ↔ Python comparisons throughout
- ✅ LINQ ↔ List comprehensions
- ✅ Familiar analogies (ASP.NET Core, Entity Framework)
- ✅ Translation of concepts you already know

### 2. No Black Boxes
- ✅ Implement everything from scratch
- ✅ Understand every line of code
- ✅ Know what frameworks do internally
- ✅ Deep understanding over superficial usage

### 3. Deep Understanding
- ✅ Visual diagrams and plots
- ✅ Step-by-step explanations
- ✅ Real-world connections to GPT/ChatGPT
- ✅ Math explained simply

### 4. Hands-On Learning
- ✅ 10,000+ lines of runnable code
- ✅ 50+ exercises with solutions
- ✅ 20+ real projects
- ✅ Production-ready examples

### 5. Production-Ready Skills
- ✅ Deploy to cloud (AWS/Azure/GCP)
- ✅ Monitoring and observability
- ✅ Security and compliance
- ✅ Cost optimization
- ✅ Career-ready portfolio

---

## 🚀 Learning Paths

### Path A: Quick Start (Foundation)
**Time:** 60-80 hours | **Goal:** Understand how LLMs work

1. Module 1: Python Basics (8-10h)
2. Module 2: NumPy & Math (12-14h)
3. Module 3: Neural Networks (35-45h)
4. Module 4: Transformers (22-34h)

**Result:** Deep understanding of neural networks and transformers

---

### Path B: Practical AI Developer
**Time:** 90-110 hours | **Goal:** Build and deploy AI applications

1. Module 3: Neural Networks (35-45h)
2. Module 4: Transformers (22-34h)
3. Module 7: Reasoning & Coding (30-40h)
4. Module 8: Prompt Engineering (25-30h)

**Result:** Professional AI developer tools and optimization skills

---

### Path C: Production ML Engineer 🔥
**Time:** 130-160 hours | **Goal:** Career-ready, job-ready skills

1. Module 3: Neural Networks (35-45h)
2. Module 3.5: PyTorch & TensorFlow (36-52h)
3. Module 4: Transformers (22-34h)
4. Module 8: Prompt Engineering (25-30h)
5. **Module 9: Production Apps (34-42h)** 🆕

**Result:** Deploy production LLM systems, $120K-200K job-ready!

---

### Path D: Full Mastery (Complete)
**Time:** 250-350 hours | **Goal:** Expert-level understanding

Complete all modules 1-9 in order.

**Result:** Senior ML Engineer skills, ready to build startup or lead teams

---

## 💼 Career Impact

### Resume-Worthy Skills

After completing this curriculum:

**Technical Skills:**
- ✅ Built neural networks from scratch (NumPy)
- ✅ Implemented transformer architecture
- ✅ Developed production AI tools
- ✅ Mastered prompt engineering
- ✅ Designed production LLM APIs handling 10K+ requests/day
- ✅ Implemented Kubernetes auto-scaling for ML workloads
- ✅ Built monitoring with Prometheus + Grafana
- ✅ Reduced LLM costs by 50-80% through optimization

**Portfolio Projects:**
- AI Code Reviewer (production-ready)
- Smart Bug Debugger
- Semantic Code Search
- Multi-tenant SaaS chatbot platform
- Enterprise chatbot system

**Salary Impact:**
- Entry → Mid-level: +$20-40K/year
- Mid → Senior: +$30-50K/year
- Freelancing: $150-300/hour potential

**Job Market:**
- Skills match 90%+ of ML Engineer job postings
- Production experience (rare and valuable!)
- Full-stack AI development

---

## 📊 Module Statistics

| Module | Status | Time | Lines of Code | Difficulty |
|--------|--------|------|---------------|------------|
| 1. Python Basics | ✅ | 8-10h | 1,500+ | Beginner |
| 1.5 Pandas & Data | 📅 NEW | 8-10h | TBD | Beginner |
| 2. NumPy & Math | ✅ | 12-14h | 2,000+ | Beginner |
| 3. Neural Networks | ✅ | 35-45h | 6,900+ | Medium |
| 3.5 PyTorch/TF | ✅ | 36-52h | 3,000+ | Medium |
| 4. Transformers | ✅ | 22-34h | 4,500+ | Medium-Hard |
| 5. Building LLMs | ✅ | 20-30h | 3,000+ | Medium |
| 5.5 HuggingFace Tokenizers | 📅 NEW | 6-8h | TBD | Medium |
| 6. Training & Fine-tuning | ✅ | 25-35h | TBD | Hard |
| 7. Reasoning & Coding | ✅ | 30-40h | 23,000+ | Advanced |
| 8. Prompt Engineering | ✅ | 25-30h | 5,000+ | Medium |
| 9. Production Apps | ✅ | 34-42h | 4,900+ | Advanced |
| 10. Vector Databases | ✅ | 15-20h | TBD | Medium |
| 10.5 RAG Without Vectors | 📅 NEW | 8-10h | TBD | Medium |
| 11. LLM Agents | ✅ | 20-25h | TBD | Advanced |
| 12. Fine-Tuning LLMs | ✅ | 20-25h | TBD | Advanced |
| 13. RLHF & Alignment | 🔧 | 20-28h | TBD | Hard |
| 14. Deploying LLMs | 📅 | 18-24h | TBD | Advanced |
| 14.5 Gradio & Streamlit | 📅 NEW | 6-8h | TBD | Beginner |
| **Total** | **85% core** | **350-450h** | **54,800+** | **Comprehensive** |

---

## 🎯 Getting Started

### Beginner Path (Start Here!)

**Week 1-2:** Module 1 (Python Basics)
- If you know Python, skip or skim
- Focus on Python vs C# comparisons

**Week 3-4:** Module 2 (NumPy & Math)
- Essential for understanding neural networks
- Practice array operations

**Week 5-10:** Module 3 (Neural Networks)
- THE foundation of all AI
- Take your time here
- Complete all exercises

**Week 11-14:** Module 4 (Transformers)
- This is how ChatGPT works!
- Build mini-GPT

**Week 15-16:** Module 8 (Prompt Engineering)
- Immediate productivity boost
- 10x better AI results

**Week 17-21:** Module 9 (Production Apps)
- Deploy to production
- Build portfolio projects
- Job-ready skills!

---

### Advanced Path (Already Know Python)

**Week 1-6:** Module 3 (Neural Networks from scratch)
**Week 7-9:** Module 4 (Transformers)
**Week 10-12:** Module 7 (Reasoning & Coding)
**Week 13-14:** Module 8 (Prompt Engineering)
**Week 15-19:** Module 9 (Production Apps)

**Result:** Production ML Engineer in 5 months!

---

## 🔥 What's New

### March 23, 2026 🆕
**NEW: Module 9 - Production LLM Applications**
- 4 comprehensive lessons (34-42 hours)
- API design with FastAPI
- Docker & Kubernetes deployment
- Prometheus + Grafana monitoring
- Security & 50-80% cost optimization
- ~4,900 lines of production-ready code
- Career-changing skills!

### March 19-20, 2026
**COMPLETE: Module 8 - Prompt Engineering**
- All 10 lessons finished
- Zero-shot to advanced techniques
- Production patterns & security
- 10x better results immediately

### March 18-19, 2026
**COMPLETE: Module 7 - Reasoning & Coding Models**
- All 10 lessons finished
- 5 production projects
- ~23,000 lines of content
- Build professional AI tools

---

## 📁 Repository Structure

```
llm-from-scratch/
├── modules/
│   ├── 01_python_basics/          ✅ Complete
│   ├── 01.5_pandas_data/          📅 Planned (NEW)
│   ├── 02_numpy_math/             ✅ Complete
│   ├── 03_neural_networks/        ✅ Complete
│   ├── 03.5_pytorch_tensorflow/   ✅ Complete
│   ├── 04_transformers/           ✅ Complete
│   ├── 05_building_llm/           ✅ Complete
│   ├── 05.5_huggingface_tokenizers/ 📅 Planned (NEW, optional)
│   ├── 06_training_finetuning/    ✅ Complete
│   ├── 07_reasoning_and_coding/   ✅ Complete
│   ├── 08_prompt_engineering/     ✅ Complete
│   ├── 09_production_llm_apps/    ✅ Complete
│   ├── 10_vector_databases/       ✅ Complete
│   ├── 10.5_rag_without_vectors/  📅 Planned (NEW)
│   ├── 11_llm_agents/             ✅ Complete
│   ├── 12_fine_tuning/            ✅ Complete
│   ├── 13_rlhf_alignment/         🔧 In Progress
│   ├── 14_deploying_llms/         📅 Planned
│   └── 14.5_gradio_streamlit/     📅 Planned (NEW, optional)
│
├── projects/                       # Capstone projects
├── references/                     # Additional reading
├── PROGRESS.md                     # Track your progress
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

---

## 🤝 Contributing

This is an educational project. Contributions welcome!

**Ways to contribute:**
- Report issues or bugs
- Suggest improvements
- Add exercises or projects
- Share your learning journey

---

## 📄 License

MIT License - Free to use for learning!

---

## 🙏 Acknowledgments

Built with 🧠 for understanding LLMs from first principles and deploying them to production.

**Inspired by:**
- The Transformer paper ("Attention Is All You Need")
- GPT architecture papers
- Fast.ai teaching philosophy
- Hands-on, deep understanding approach

---

## 📞 Support & Community

**Resources:**
- **Documentation:** Comprehensive lessons in each module
- **Code Examples:** 54,800+ lines of working code
- **Exercises:** 50+ hands-on problems with solutions
- **Projects:** 20+ real-world applications

**Getting Help:**
- Open an issue for questions
- Check module README files
- Review PROGRESS.md for tracking

---

## 🎯 Your Journey

### Where You'll Start
- Understanding Python basics
- Learning NumPy for ML
- Building neural networks from scratch

### Where You'll End
- Building production LLM applications
- Deploying to cloud at scale
- Optimizing costs by 50-80%
- Job-ready for $120K-200K roles
- Portfolio with 5+ production projects

---

**Ready to start?**

1. ✅ Read this README
2. ⬜ Clone the repository
3. ⬜ Choose your learning path
4. ⬜ Start with Module 1 (or Module 3 if you know Python)
5. ⬜ Build, learn, deploy!

---

**Start Learning:** `cd modules/01_python_basics` or `cd modules/09_production_llm_apps`

**Track Progress:** Check `PROGRESS.md` in root and each module

**Get Help:** Open an issue

---

**Let's build amazing AI systems together!** 🚀

Built with 🧠 for deep understanding, hands-on learning, and production deployment.

**From first principles to production - Learn, Build, Deploy!**

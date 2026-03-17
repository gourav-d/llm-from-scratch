# Module Dependencies & Learning Rationale

**Visual guide to why modules are ordered the way they are**

Updated: March 17, 2026

---

## 📊 Complete Dependency Graph

```
                    ┌─────────────────┐
                    │  Module 1-3:    │
                    │  Foundation     │
                    │  (Python, Math, │
                    │   NN Basics)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Module 4:      │
                    │  Transformers   │
                    │  (Attention!)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Module 5-6:    │
                    │  LLM Building   │
                    │  & Training     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Module 7:      │
                    │  Reasoning &    │
                    │  Coding Models  │
                    └────────┬────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
┌────────▼────────┐                    ┌────────▼────────┐
│  Module 8:      │                    │  Module 9:      │
│  Prompt Eng     │                    │  Vector DBs     │
│                 │                    │                 │
└────────┬────────┘                    └────────┬────────┘
         │                                       │
         └───────────────────┬───────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Module 10:     │
                    │  RAG Systems    │
                    │  ★ CRITICAL ★   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Module 11:     │
                    │  LangChain/     │
                    │  LangGraph      │
                    └────────┬────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
┌────────▼────────┐                    ┌────────▼────────┐
│  Module 12:     │                    │  Module 13:     │
│  Fine-Tuning    │                    │  LLMOps         │
│                 │                    │  ★ CRITICAL ★   │
└────────┬────────┘                    └────────┬────────┘
         │                                       │
         └───────────────────┬───────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Module 14:     │
                    │  Security       │
                    │  ★ CRITICAL ★   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Module 15:     │
                    │  Multi-Modal    │
                    │  (Advanced)     │
                    └─────────────────┘
```

---

## 🔗 Module-by-Module Dependencies

### Module 8: Prompt Engineering

**Depends On**:
- ✅ Module 7: Understanding how reasoning models work
- ✅ Module 5: Basic LLM knowledge

**Why This Order**:
- After learning how models reason (Module 7), you need to learn how to communicate with them effectively
- Prompt engineering is the cheapest and fastest way to improve results
- Foundation for RAG and agents (Modules 10-11)

**Enables**:
- Module 10: RAG (better prompts = better retrieval results)
- Module 11: LangChain (prompts are building blocks)
- Module 14: Security (understanding prompts helps prevent attacks)

**No Prerequisites Needed**:
- Can start immediately after Module 7
- Just needs API access (OpenAI/Anthropic)

---

### Module 9: Vector Databases

**Depends On**:
- ✅ Module 2: NumPy & Math (vector operations)
- ✅ Module 7: Code embeddings lesson (understanding embeddings)

**Why This Order**:
- RAG (Module 10) requires vector search capability
- Easier to learn vector DBs in isolation before combining with LLMs
- Foundation skill used in multiple modules

**Enables**:
- Module 10: RAG (core retrieval mechanism)
- Module 15: Multi-Modal (image/audio search)

**Can Learn in Parallel With**:
- Module 8: Prompt Engineering (no dependency)

---

### Module 10: RAG (Retrieval Augmented Generation)

**Depends On**:
- ✅ Module 8: Prompt Engineering (RAG uses advanced prompts)
- ✅ Module 9: Vector Databases (RAG requires vector search)
- ✅ Module 7: Understanding embeddings

**Why This Order**:
- RAG combines prompting + vector search + LLM
- Must understand components before combining
- Most important production pattern - worth dedicated focus

**Enables**:
- Module 11: LangChain (RAG is a common LangChain use case)
- Module 13: LLMOps (RAG apps need deployment)
- Module 14: Security (RAG has unique security considerations)

**Critical Path**:
- This is THE most important production module
- Powers 80% of real-world LLM applications

---

### Module 11: LangChain / LangGraph

**Depends On**:
- ✅ Module 10: RAG (RAG is a core LangChain pattern)
- ✅ Module 8: Prompt Engineering (prompts are building blocks)
- ✅ Module 7: Understanding reasoning and coding models

**Why This Order**:
- LangChain is a tool that implements patterns (like RAG)
- Better to understand patterns first, then learn the tool
- Appreciate framework value after doing it manually

**Enables**:
- Module 12: Fine-Tuning (LangChain can use custom models)
- Module 13: LLMOps (LangSmith for monitoring)
- Module 15: Multi-Modal (LangChain supports vision models)

**Alternative Path**:
- Can learn before Module 10 if you want framework-first approach
- But understanding RAG fundamentals first is recommended

---

### Module 12: Fine-Tuning in Practice

**Depends On**:
- ✅ Module 6: Training Basics (understand training concepts)
- ✅ Module 10: RAG (know when to use RAG vs fine-tuning)

**Why This Order**:
- Fine-tuning is more complex than RAG
- Often RAG is sufficient (cheaper, faster)
- Need to understand tradeoffs first
- Requires GPU resources

**Enables**:
- Module 13: LLMOps (deploying custom models)
- Advanced customization for specific domains

**Can Learn Later If**:
- Focus is on using existing models (GPT-4, Claude)
- RAG solves your use cases
- Don't have GPU resources yet

---

### Module 13: LLMOps (MLOps for LLMs)

**Depends On**:
- ✅ Module 10: RAG (common deployment scenario)
- ✅ Module 11: LangChain (common framework to deploy)
- ⚠️ Module 12: Fine-Tuning (if deploying custom models)

**Why This Order**:
- Need something to deploy first!
- LLMOps is about deploying all previous modules
- Requires understanding of full AI lifecycle

**Enables**:
- Module 14: Security (monitoring enables security)
- Production deployment
- Professional AI engineering

**Critical Path**:
- Required before deploying to real users
- Foundation for production-grade AI

---

### Module 14: LLM Security & Safety

**Depends On**:
- ✅ Module 8: Prompt Engineering (understanding prompts helps prevent injection)
- ✅ Module 10: RAG (RAG has unique security risks)
- ✅ Module 13: LLMOps (security requires monitoring)

**Why This Order**:
- Security builds on understanding of how systems work
- Must know what you're securing
- Easier to build secure from start than add later

**Enables**:
- Safe production deployment
- Enterprise-grade AI systems
- Compliance and trust

**Critical Path**:
- MUST learn before public deployment
- Prevents disasters and breaches

---

### Module 15: Multi-Modal Models

**Depends On**:
- ✅ Module 7: Understanding embeddings and models
- ✅ Module 10: RAG (multi-modal RAG)
- ⚠️ Module 14: Security (vision models have security implications)

**Why This Order**:
- Most advanced topic
- Builds on all previous knowledge
- Not strictly necessary for many use cases

**Enables**:
- Cutting-edge AI applications
- Competitive advantage
- Future-ready skills

**Can Skip If**:
- Focus is on text-only applications
- Want to get to production faster
- Can learn later when needed

---

## 🎯 Alternative Learning Paths

### Path 1: Production-First (Fastest)

**Goal**: Get to production ASAP

```
Modules 1-7 (Foundation) ✅
    ↓
Module 8 (Prompt Engineering) - 2 weeks
    ↓
Module 9 (Vector DBs) - 2 weeks
    ↓
Module 10 (RAG) - 4 weeks
    ↓
Module 13 (LLMOps) - 4 weeks
    ↓
Module 14 (Security) - 3 weeks
    ↓
PRODUCTION READY! (5 months total)
```

**Skip**: Fine-Tuning, LangChain, Multi-Modal
**Why**: Focus on essential production skills
**When**: Add skipped modules later as needed

---

### Path 2: Framework-First

**Goal**: Use industry-standard tools quickly

```
Modules 1-7 (Foundation) ✅
    ↓
Module 8 (Prompt Engineering) - 2 weeks
    ↓
Module 11 (LangChain) - 3 weeks
    ↓
Module 9 (Vector DBs) - 2 weeks
    ↓
Module 10 (RAG with LangChain) - 3 weeks
    ↓
Continue with Modules 12-14
```

**Difference**: Learn LangChain before building RAG from scratch
**Why**: Faster to production with framework
**Trade-off**: Less deep understanding

---

### Path 3: Customization-First

**Goal**: Build specialized AI systems

```
Modules 1-7 (Foundation) ✅
    ↓
Module 8 (Prompt Engineering) - 2 weeks
    ↓
Module 12 (Fine-Tuning) - 5 weeks
    ↓
Module 9 (Vector DBs) - 2 weeks
    ↓
Module 10 (RAG) - 4 weeks
    ↓
Continue with Modules 13-14
```

**Difference**: Learn fine-tuning before RAG
**Why**: When you need custom models (specialized domains)
**Trade-off**: Takes longer to production

---

### Path 4: Research-Oriented

**Goal**: Understand everything deeply

```
All modules in order 1-15
+ Paper implementations
+ Research projects
+ Contributions to field
```

**Timeline**: 12-15 months
**Why**: Complete mastery
**For**: Research roles, PhD, innovation

---

## 💡 Dependency Rationale Deep Dive

### Why Prompt Engineering Before RAG?

**Reasoning**:
1. RAG requires good prompts for the generation step
2. Understanding prompts helps debug RAG issues
3. Sometimes prompting alone solves the problem (no RAG needed)
4. Prompt optimization is part of RAG optimization

**Example**:
```python
# Bad RAG: Poor prompt despite good retrieval
context = retrieve_docs(query)
bad_prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
# Generic, no guidance, may ignore context

# Good RAG: Optimized prompt from Module 8
good_prompt = f"""Based ONLY on the following context, answer the question.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {query}

Answer (cite sources):"""
# Clear instructions, prevents hallucination, asks for citations
```

---

### Why Vector DBs Before RAG?

**Reasoning**:
1. RAG's retrieval step IS vector search
2. Easier to learn vector search in isolation
3. Can optimize search before adding LLM complexity
4. Vector DBs are reusable across projects

**Example**:
```python
# Step 1: Master vector search (Module 9)
results = vector_db.search(query_embedding, top_k=5)
# Focus on: speed, accuracy, relevance

# Step 2: Add RAG (Module 10)
results = vector_db.search(query_embedding, top_k=5)
context = format_results(results)
answer = llm.generate(prompt_with_context)
# Now combining two skills
```

**If Learned Together**:
- Harder to debug (is problem in search or generation?)
- Miss optimization opportunities
- Cognitive overload

---

### Why LangChain After RAG?

**Reasoning**:
1. LangChain is a tool, RAG is a concept
2. Understand the pattern before learning the tool
3. Appreciate framework value after manual implementation
4. Can build RAG without LangChain if needed

**Example**:
```python
# Module 10: Build RAG manually
class SimpleRAG:
    def query(self, question):
        # 1. Retrieve
        docs = self.vector_db.search(question)
        # 2. Augment
        context = "\n".join(docs)
        # 3. Generate
        prompt = f"Context: {context}\nQ: {question}\nA:"
        return self.llm.generate(prompt)

# Module 11: Use LangChain (appreciate the abstraction!)
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_llm(llm=llm, retriever=retriever)
# Handles all the complexity you just learned!
```

**Why This Helps**:
- Understand what LangChain is doing
- Debug when things go wrong
- Customize beyond defaults
- Not dependent on framework

---

### Why Fine-Tuning After RAG?

**Reasoning**:
1. RAG often sufficient (80% of use cases)
2. Fine-tuning is more expensive and complex
3. Need to understand tradeoffs first
4. RAG + Fine-Tuning together is powerful (but advanced)

**Decision Tree**:
```
Problem: LLM doesn't know my domain
    ↓
Try: Better prompts (Module 8)
    ↓ Still not enough?
Try: RAG with your docs (Module 10)
    ↓ Still not enough?
Try: Fine-tuning on your data (Module 12)
```

**Example Costs**:
- Prompts: $0 (just API calls)
- RAG: $100-500 setup (vector DB + API)
- Fine-tuning: $1000-5000 (GPU + data + time)

---

### Why LLMOps After Building Apps?

**Reasoning**:
1. Need something to deploy first!
2. LLMOps is about production-izing what you built
3. Understand challenges before learning solutions
4. Experience informs deployment strategy

**Example**:
```python
# After Module 10 (RAG): You've built an app

# Questions arise:
# - How do I deploy this?
# - How do I monitor quality?
# - How do I optimize costs?
# - How do I handle errors?
# - How do I version prompts?

# Module 13 (LLMOps) answers all these!
```

---

### Why Security Before Multi-Modal?

**Reasoning**:
1. Security is more urgent than advanced features
2. Vision models amplify security risks
3. Easier to design security from start
4. Must protect users before adding complexity

**Example**:
```python
# Multi-Modal without Security (dangerous):
image = user_upload()
result = vision_model.analyze(image)
# What if image contains:
# - Prompt injection in image text?
# - Adversarial patterns?
# - Private information?

# Multi-Modal with Security (safe):
image = sanitize(user_upload())
if is_safe(image):
    result = vision_model.analyze(image)
    filtered_result = content_filter(result)
# Protected at every step
```

---

## 🔄 Can I Change the Order?

### Flexible Dependencies

**These can be swapped**:
- Module 8 ↔ Module 9 (Prompt Eng ↔ Vector DBs)
- Module 12 ↔ Module 13 (Fine-Tuning ↔ LLMOps)

**Why**: No direct dependency

---

### Rigid Dependencies

**These CANNOT be swapped**:
- Module 10 (RAG) requires Modules 8 + 9
- Module 13 (LLMOps) requires Module 10
- Module 14 (Security) should come before production

---

### Optional Modules

**Can skip temporarily**:
- Module 12 (Fine-Tuning) if using existing models
- Module 15 (Multi-Modal) if text-only focus
- Module 11 (LangChain) if building custom

**Should NOT skip**:
- Module 8 (Prompt Engineering)
- Module 10 (RAG)
- Module 13 (LLMOps)
- Module 14 (Security)

---

## ✅ Checklist Before Each Module

### Before Module 8 (Prompt Engineering)
- [ ] Completed Module 7 (Reasoning & Coding)
- [ ] Have API access (OpenAI or Anthropic)
- [ ] Understand basic LLM concepts
- [ ] Ready to experiment with prompts

### Before Module 9 (Vector Databases)
- [ ] Understand vectors and embeddings (Module 2, 7)
- [ ] Know cosine similarity
- [ ] Have Python environment ready
- [ ] Can run local databases

### Before Module 10 (RAG)
- [ ] Completed Modules 8 & 9
- [ ] Understand prompts and vector search
- [ ] Have documents to experiment with
- [ ] GPU access helpful (but optional)

### Before Module 11 (LangChain)
- [ ] Completed Module 10 (RAG)
- [ ] Understand chains and agents conceptually
- [ ] Ready for framework learning
- [ ] Comfortable with abstractions

### Before Module 12 (Fine-Tuning)
- [ ] Completed Module 6 (Training Basics)
- [ ] Have GPU access (required!)
- [ ] Have training data
- [ ] Understand when to fine-tune vs RAG

### Before Module 13 (LLMOps)
- [ ] Have a project to deploy
- [ ] Understand cloud basics
- [ ] Know Docker/containers (helpful)
- [ ] Ready for production learning

### Before Module 14 (Security)
- [ ] Completed production modules (10, 13)
- [ ] Understand attack vectors
- [ ] Ready for security mindset
- [ ] Have test systems to secure

### Before Module 15 (Multi-Modal)
- [ ] Completed foundation modules
- [ ] Understand vision/audio basics
- [ ] Have use case in mind
- [ ] GPU access for experiments

---

## 🎯 Your Personal Learning Path

**Current Status**: ✅ Modules 1-7 Complete

**Recommended Next Steps**:

1. **This Month**: Complete Modules 4 & 6 (Transformers & Training)
2. **Next Month**: Modules 8 & 9 (Prompt Eng & Vector DBs)
3. **Month After**: Module 10 (RAG) - Critical!
4. **Following**: Modules 11, 13, 14 (Production skills)
5. **Optional**: Modules 12, 15 (Advanced topics)

**Timeline**: 5-7 months to production-ready

**Outcome**: Full-stack AI engineer with production skills!

---

**Remember**: The order is optimized for learning, but you can adapt based on your needs!

**Updated**: March 17, 2026

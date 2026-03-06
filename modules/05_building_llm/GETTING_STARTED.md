# Getting Started with Module 5: Building Your LLM

**Welcome to the foundation of all language models!**

---

## 🎯 What This Module Covers

In this module, you'll learn:
- **Lesson 1:** Tokenization - Breaking text into tokens
- **Lesson 2:** Word Embeddings - Converting tokens to meaningful vectors

**Total Time:** 8-12 hours

---

## ⚡ Quick Start (2 Hours)

**Goal:** Get a feel for tokenization and embeddings

### Step 1: Run the Examples (30 min)

```bash
# Navigate to examples
cd modules/05_building_llm/examples

# Run tokenization example
python example_01_tokenization.py

# Run embeddings example
python example_02_word_embeddings.py
```

**What you'll see:**
- Different tokenization approaches in action
- How tokens become vectors
- Semantic similarities between words
- Visualization of embeddings

### Step 2: Read Lesson 1 (45 min)

Open `01_tokenization.md` and read through:
- Why tokenization matters
- Character vs word vs subword tokenization
- How GPT tokenizes text

### Step 3: Read Lesson 2 (45 min)

Open `02_word_embeddings.md` and read through:
- One-hot encoding limitations
- Dense embeddings
- Semantic relationships (king - man + woman = queen)

**Result:** You understand the basics and have seen them in action!

---

## 📚 Standard Path (8-12 Hours)

**Goal:** Complete understanding with hands-on practice

### Day 1: Tokenization (4-5 hours)

**Morning Session (2 hours):**
1. Read `01_tokenization.md` thoroughly
2. Take notes on key concepts
3. Compare C#/.NET string processing to Python tokenization

**Afternoon Session (2-3 hours):**
1. Run `example_01_tokenization.py` line by line
2. Experiment with different texts
3. Complete `exercise_01_tokenization.py`
4. Build your own simple tokenizer

**Evening:**
- Review vocabulary building process
- Understand special tokens (BOS, EOS, PAD, UNK)

---

### Day 2: Word Embeddings (4-5 hours)

**Morning Session (2 hours):**
1. Read `02_word_embeddings.md` thoroughly
2. Understand one-hot vs dense embeddings
3. Learn about Word2Vec intuition

**Afternoon Session (2-3 hours):**
1. Run `example_02_word_embeddings.py` step by step
2. Experiment with similarity calculations
3. Complete `exercise_02_word_embeddings.py`
4. Visualize embeddings in 2D

**Evening:**
- Understand how embeddings are learned
- Connect to transformer architecture from Module 4

---

### Day 3: Integration & Review (2-3 hours)

**Morning:**
1. Review both lessons
2. Understand end-to-end pipeline: Text → Tokens → IDs → Embeddings
3. Connect to GPT architecture

**Afternoon:**
1. Build a mini text processor
2. Combine tokenization + embeddings
3. Test on sample sentences

**Result:** Deep understanding + practical skills!

---

## 🏃 Intensive Path (2-3 Days)

**Goal:** Master tokenization and embeddings quickly

### Day 1 (6-8 hours)

**Session 1: Tokenization Theory (2 hours)**
- Read `01_tokenization.md`
- Study character, word, BPE approaches
- Understand vocabulary creation

**Session 2: Tokenization Practice (2 hours)**
- Run `example_01_tokenization.py`
- Implement character tokenizer from scratch
- Build vocabulary from corpus

**Session 3: Embeddings Theory (2 hours)**
- Read `02_word_embeddings.md`
- Study one-hot vs dense representations
- Learn about semantic spaces

**Session 4: Embeddings Practice (2 hours)**
- Run `example_02_word_embeddings.py`
- Implement simple embedding layer
- Calculate word similarities

---

### Day 2 (4-6 hours)

**Session 1: BPE Deep Dive (2 hours)**
- Study BPE algorithm in detail
- Implement simple BPE
- Compare to GPT tokenizer

**Session 2: Advanced Embeddings (2 hours)**
- Study embedding initialization
- Learn about positional embeddings
- Understand training process

**Session 3: Exercises (2 hours)**
- Complete both exercises
- Build custom tokenizer + embeddings
- Test on real text

---

### Day 3 (2-3 hours)

**Session 1: Integration (1.5 hours)**
- Combine tokenization + embeddings
- Build end-to-end text processor
- Test with different texts

**Session 2: Connection to LLMs (1 hour)**
- Understand how GPT uses these concepts
- Study tokenizer impact on performance
- Learn about token economy

**Result:** Complete mastery in 2-3 days!

---

## 🎓 Learning by Experience Level

### If You're a Beginner:
1. Start with Quick Start
2. Follow Standard Path
3. Don't rush - understanding is key
4. Experiment with examples
5. Ask questions (imagine explaining to rubber duck!)

**Estimated Time:** 12-15 hours

---

### If You Have ML Experience:
1. Skim Quick Start
2. Deep dive into lessons
3. Focus on implementation details
4. Compare to frameworks you know
5. Complete exercises

**Estimated Time:** 8-10 hours

---

### If You're Advanced:
1. Read lessons quickly
2. Implement from scratch
3. Study research papers
4. Experiment with variations
5. Connect to production systems

**Estimated Time:** 6-8 hours

---

## 📖 Lesson Order

**Recommended Order:**
1. **Lesson 1: Tokenization** ⭐ (Start here!)
2. **Lesson 2: Word Embeddings** ⭐

**Why this order?**
- Tokenization comes first in the pipeline
- Embeddings operate on tokens
- Logical progression: Text → Tokens → Vectors

---

## 🛠️ Prerequisites Check

Before starting, ensure you have:

✅ **Python skills** (Module 1 complete)
- Functions, classes, file I/O

✅ **NumPy knowledge** (Module 2 complete)
- Matrix operations, broadcasting

✅ **Neural network basics** (Module 3 complete)
- Layers, forward pass, training

✅ **Transformer understanding** (Module 4 complete)
- Attention mechanism basics

**Missing prerequisites?**
- Go back and complete required modules
- Tokenization/embeddings build on these concepts

---

## 💻 Setup

### Required Libraries

```bash
# Install dependencies
pip install numpy matplotlib scikit-learn

# Optional (for advanced work)
pip install tiktoken  # OpenAI's tokenizer
```

### File Structure

```
modules/05_building_llm/
├── 01_tokenization.md           # Lesson 1
├── 02_word_embeddings.md        # Lesson 2
├── examples/
│   ├── example_01_tokenization.py
│   └── example_02_word_embeddings.py
└── exercises/
    ├── exercise_01_tokenization.py
    └── exercise_02_word_embeddings.py
```

### Test Your Setup

```bash
# Test Python
python --version  # Should be 3.8+

# Test NumPy
python -c "import numpy; print(numpy.__version__)"

# Test Matplotlib
python -c "import matplotlib; print('OK')"
```

---

## 📝 Study Tips

### Active Learning
- **Type out code** - Don't just read
- **Modify examples** - Change inputs, see results
- **Break things** - Learn from errors
- **Explain concepts** - Teach to understand

### Note-Taking
- Draw diagrams of tokenization process
- Create tables comparing approaches
- Write your own examples
- Note C#/.NET comparisons

### Practice
- Complete all exercises
- Build mini-projects
- Experiment with real text
- Try different vocabularies

### Connect Concepts
- How does tokenization affect model performance?
- Why do embeddings capture semantics?
- How does this connect to transformers?
- What's the pipeline from text to prediction?

---

## 🎯 Learning Objectives

### After Lesson 1, you'll:
- ✅ Understand why tokenization is necessary
- ✅ Implement character-level tokenizer
- ✅ Implement word-level tokenizer
- ✅ Understand BPE algorithm
- ✅ Build vocabulary from text
- ✅ Encode/decode text to token IDs
- ✅ Handle special tokens

### After Lesson 2, you'll:
- ✅ Explain one-hot encoding problems
- ✅ Understand dense embeddings
- ✅ Implement embedding layer
- ✅ Calculate word similarities
- ✅ Understand semantic relationships
- ✅ Connect embeddings to transformers
- ✅ Visualize embeddings in 2D

---

## ❓ Common Questions

### Q: Do I need to learn tiktoken (OpenAI's tokenizer)?
**A:** Not for this module! We build from scratch to understand fundamentals. You can explore tiktoken after completing the lessons.

### Q: How is this different from string.split()?
**A:** string.split() is basic word tokenization. We learn:
- Subword tokenization (BPE) like GPT uses
- Vocabulary building
- Handling unknown words
- Special tokens
- Much more sophisticated!

### Q: Why learn one-hot if we use dense embeddings?
**A:** Understanding one-hot helps you appreciate why dense embeddings are better! It's about the journey, not just the destination.

### Q: Can I skip to embeddings?
**A:** Not recommended! Embeddings operate on tokens, so understanding tokenization first makes embeddings clearer.

### Q: How long for this module?
**A:** Quick path: 5-6 hours. Standard: 8-12 hours. Intensive: 10-15 hours with deep study.

---

## 🚀 Ready to Start?

### Recommended First Step:
👉 **Open `01_tokenization.md`** and start reading!

### Or:
👉 **Run `example_01_tokenization.py`** to see it in action first!

---

## 📚 Additional Resources

### Articles
- "The Illustrated Word2Vec" (Jay Alammar)
- "Subword Neural Machine Translation" (original BPE paper)
- "Efficient Estimation of Word Representations" (Word2Vec)

### Videos
- Search YouTube: "Word embeddings explained"
- Search YouTube: "BPE tokenization"

### Tools to Explore Later
- tiktoken (OpenAI)
- SentencePiece (Google)
- Hugging Face Tokenizers

---

## 💪 You've Got This!

**Remember:**
- Start simple (character tokenization)
- Build up gradually (word → BPE)
- Practice with examples
- Experiment and break things
- Connect to GPT architecture

**This module is the bridge between text and neural networks!**

---

**Happy Learning! 🎉**

---

**Next:** [01_tokenization.md](01_tokenization.md) - Start your journey!

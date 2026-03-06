# Module 5: Building Your LLM - Tokenization & Word Embeddings

**Transform text into numbers that neural networks can understand!**

---

## 🎯 What You'll Learn

This module teaches you the **foundational techniques** for representing text in Large Language Models:

- ✅ **Tokenization** - Breaking text into processable units
- ✅ **Character-Level Tokenization** - Simple character-based approach
- ✅ **Word-Level Tokenization** - Word-based tokenization
- ✅ **Subword Tokenization (BPE)** - How GPT tokenizes text
- ✅ **Word Embeddings** - Converting tokens to meaningful vectors
- ✅ **One-Hot Encoding** - Basic representation and its limitations
- ✅ **Dense Embeddings** - Learning semantic relationships
- ✅ **Embedding Layers** - Building blocks of transformers

**After this module, you'll understand how text becomes numbers in GPT!**

---

## 🚀 Why Tokenization and Embeddings?

### The Problem

Neural networks work with numbers, not text. How do we bridge this gap?

**Example:**
```
Text: "The cat sat on the mat"
Neural Network: ❓ How to process this?
```

### The Solution

**Step 1: Tokenization**
```
"The cat sat on the mat"
       ↓
["The", "cat", "sat", "on", "the", "mat"]
       ↓
[42, 156, 89, 12, 42, 278]  (Token IDs)
```

**Step 2: Embeddings**
```
Token IDs: [42, 156, 89, ...]
       ↓
Dense Vectors:
[
  [0.25, -0.1, 0.8, ...],   # "The"
  [0.1, 0.3, -0.2, ...],    # "cat"
  ...
]
```

---

## 📚 Module Overview

### What You'll Build

By the end of this module, you'll have:

1. **Character tokenizer** - Simple character-level approach
2. **Word tokenizer** - Word-based with vocabulary
3. **BPE tokenizer** - Subword tokenization (like GPT!)
4. **Embedding layer** - Converting tokens to vectors
5. **Similarity calculator** - Finding related words
6. **Visualization** - Seeing embeddings in 2D space

---

## 🎓 Prerequisites

Before starting Module 5, you should have completed:

✅ **Module 1:** Python Basics (all 10 lessons)
✅ **Module 2:** NumPy & Math (matrix operations)
✅ **Module 3:** Neural Networks (understanding layers)
✅ **Module 4:** Transformers (attention mechanism)

**Why these prerequisites?**
- **Python:** You'll write tokenizers from scratch
- **NumPy:** Embedding operations use matrices
- **Neural Networks:** Embeddings are learned through backprop
- **Transformers:** Embeddings feed into attention layers

---

## 📖 Lessons

### Lesson 1: Tokenization ⭐
**File:** `01_tokenization.md`

**What you'll learn:**
- Why tokenization is necessary
- Character-level tokenization
- Word-level tokenization
- Subword tokenization (BPE)
- Building vocabulary
- Encoding and decoding
- Special tokens (BOS, EOS, PAD, UNK)
- How GPT tokenizes text

**Time:** 2-3 hours

**Key insight:**
> Tokenization determines the "vocabulary" of your LLM. GPT uses BPE to balance vocabulary size and coverage!

---

### Lesson 2: Word Embeddings ⭐
**File:** `02_word_embeddings.md`

**What you'll learn:**
- From tokens to vectors
- One-hot encoding and its problems
- Dense embeddings
- Embedding dimensions
- Semantic relationships (king - man + woman = queen)
- Word2Vec intuition
- Embedding layers
- Positional embeddings
- How embeddings are trained

**Time:** 2-3 hours

**Key insight:**
> Embeddings capture meaning! Similar words have similar vectors, enabling the model to understand semantic relationships.

---

## 🛠️ What You'll Build

### Example 1: Character Tokenizer
```python
# Simple character-level tokenization
tokenizer = CharTokenizer()
tokens = tokenizer.encode("Hello!")
# [8, 5, 12, 12, 15, 33]  # Character IDs
text = tokenizer.decode(tokens)
# "Hello!"
```

### Example 2: Word Tokenizer with Vocabulary
```python
# Build vocabulary and tokenize
tokenizer = WordTokenizer()
tokenizer.build_vocab(corpus)
tokens = tokenizer.encode("The cat sat")
# [42, 156, 89]  # Word IDs
```

### Example 3: Simple BPE Tokenizer
```python
# Subword tokenization like GPT
tokenizer = BPETokenizer()
tokens = tokenizer.encode("tokenization")
# ["token", "ization"]  # Subword units
```

### Example 4: Embedding Layer
```python
# Convert tokens to dense vectors
embedding = EmbeddingLayer(vocab_size=1000, embed_dim=128)
vectors = embedding(token_ids)
# Shape: (batch_size, seq_len, 128)
```

### Example 5: Word Similarity
```python
# Find similar words
similarity = embedding.cosine_similarity("king", "queen")
# 0.85 (highly similar!)

# Word analogies
result = embedding.analogy("king", "man", "woman")
# "queen"
```

---

## 🎯 Learning Path

### Path A: Quick Understanding (5-6 hours)
**Goal:** Understand tokenization and embeddings conceptually

**Week 1:**
- Lesson 1: Tokenization
- Run example_01_tokenization.py
- Understand BPE visually

**Week 2:**
- Lesson 2: Word Embeddings
- Run example_02_word_embeddings.py
- See semantic relationships

**Result:** Conceptual understanding, can explain to others

---

### Path B: Complete Mastery (10-12 hours)
**Goal:** Build tokenizers and embeddings from scratch

**Week 1:**
- Lesson 1 with deep study
- Implement character tokenizer
- Implement word tokenizer
- Complete exercise_01

**Week 2:**
- Lesson 2 with deep study
- Build embedding layer from scratch
- Calculate similarities
- Complete exercise_02

**Result:** Can implement from scratch, understand deeply

---

### Path C: Research Depth (15-20 hours)
**Goal:** Master tokenization and embeddings for advanced work

**Weeks 1-2:**
- All lessons with research papers
- Implement full BPE algorithm
- Study Word2Vec/GloVe papers
- Build production-ready tokenizer

**Week 3:**
- Experiment with different tokenization strategies
- Train custom embeddings
- Visualize high-dimensional embeddings
- Study GPT tokenizer internals

**Result:** Research-level understanding

---

## 🔗 Connection to Previous Modules

### From Module 3: Neural Networks

| Module 3 Concept | Used in Module 5 |
|------------------|------------------|
| **Matrix multiplication** | Embedding lookup operations |
| **Layers** | Embedding layer is a lookup layer |
| **Backpropagation** | How embeddings are learned |
| **Initialization** | Embedding weights initialization |

### From Module 4: Transformers

| Module 4 Concept | Connection to Module 5 |
|------------------|------------------------|
| **Attention** | Operates on embeddings |
| **Positional encoding** | Added to word embeddings |
| **Input representation** | Token IDs → Embeddings |
| **Vocabulary** | Size determines embedding table size |

**You're building the input pipeline for transformers!**

---

## 🎁 What's Included

### Code Examples
- `example_01_tokenization.py` - All tokenization approaches
- `example_02_word_embeddings.py` - Embedding layers and similarities

### Exercises
- `exercise_01_tokenization.py` - Practice building tokenizers
- `exercise_02_word_embeddings.py` - Practice with embeddings

### Documentation
- Comprehensive lesson files
- Visual diagrams
- Connection to GPT/BERT
- Real-world examples

---

## 💡 Key Insights You'll Gain

### 1. Why GPT Uses BPE
```
Word-level: Large vocabulary (millions of words)
Character-level: Long sequences (inefficient)
BPE (Subword): Balance! (50k tokens, efficient)

Example:
"unbelievable" → ["un", "believ", "able"]
- Smaller vocabulary
- Handles rare words
- Captures morphology
```

### 2. How Embeddings Capture Meaning
```
One-hot encoding:
"cat" = [0, 0, 1, 0, 0, ...]  (sparse, no meaning)
"dog" = [0, 1, 0, 0, 0, ...]  (no relationship!)

Dense embeddings:
"cat" = [0.25, -0.1, 0.8, ...]
"dog" = [0.22, -0.08, 0.75, ...]  (similar vectors!)

→ Model learns "cat" and "dog" are related!
```

### 3. The Token Economy
```
GPT-3/GPT-4 pricing:
- Based on tokens, not characters!
- "unbelievable" = 3 tokens
- Understanding tokenization helps estimate costs!
```

---

## 🔍 Real-World Applications

After this module, you'll understand how these work:

### Language Models
- **GPT-3/GPT-4** - Uses BPE with ~50k vocabulary
- **BERT** - Uses WordPiece tokenization
- **LLaMA** - SentencePiece tokenization

### Tokenization Tools
- **tiktoken** - OpenAI's tokenizer
- **SentencePiece** - Google's tokenizer
- **Hugging Face Tokenizers** - Fast tokenization

### Embeddings
- **Word2Vec** - Original word embeddings
- **GloVe** - Global vectors
- **FastText** - Subword embeddings
- **Transformer embeddings** - Contextual embeddings

**You'll understand the foundation of all modern NLP!**

---

## 📊 Expected Time Investment

| Component | Time |
|-----------|------|
| **2 Lessons** | 4-6 hours |
| **2 Examples** | 2-3 hours |
| **2 Exercises** | 2-3 hours |
| **Total** | 8-12 hours |

**Pace:**
- Casual: 1 lesson per week (2 weeks)
- Moderate: Both lessons in 1 week
- Intensive: Full module in 2-3 days

---

## ✅ Success Criteria

You've mastered Module 5 when you can:

✅ **Explain tokenization** approaches and trade-offs
✅ **Build a character tokenizer** from scratch
✅ **Build a word tokenizer** with vocabulary
✅ **Understand BPE** and how it works
✅ **Explain one-hot encoding** limitations
✅ **Implement embedding layer** from scratch
✅ **Calculate word similarity** using embeddings
✅ **Understand semantic relationships** in vector space
✅ **Connect tokenization to GPT** architecture

---

## 🚀 After Module 5

### You'll Be Ready For:

**Module 6: Training & Fine-tuning**
- Data preparation and tokenization
- Pre-training language models
- Fine-tuning for specific tasks
- RLHF (how ChatGPT is trained!)

**Real Projects:**
- Build custom tokenizers
- Train word embeddings
- Implement GPT from scratch
- Fine-tune pre-trained models

**Career Skills:**
- Understand LLM internals
- Debug tokenization issues
- Optimize token usage
- Build production NLP systems

---

## 📚 Recommended Reading

### Before Starting
- ✅ Complete Modules 1-4
- ✅ Comfortable with NumPy
- ✅ Understand transformers

### During Module
- "Neural Machine Translation by Jointly Learning to Align and Translate" (Attention paper)
- "Efficient Estimation of Word Representations" (Word2Vec)
- "GloVe: Global Vectors for Word Representation"

### After Module
- "SentencePiece: A simple and language independent approach"
- "BPE: Neural Machine Translation of Rare Words"
- GPT-2/GPT-3 papers for tokenization details

---

## 🎯 Module Objectives

By the end of Module 5, you will:

1. **Understand** why tokenization is necessary
2. **Implement** character, word, and BPE tokenizers
3. **Build** vocabulary from corpus
4. **Encode/decode** text to token IDs
5. **Explain** one-hot vs dense embeddings
6. **Create** embedding layers from scratch
7. **Calculate** semantic similarities
8. **Visualize** embeddings in 2D space

---

## 📁 Module Structure

```
modules/05_building_llm/
├── README.md                           ← You are here!
├── GETTING_STARTED.md                  ← Start here next
├── 01_tokenization.md                  ← Lesson 1
├── 02_word_embeddings.md               ← Lesson 2
├── examples/
│   ├── example_01_tokenization.py      ← All tokenization approaches
│   └── example_02_word_embeddings.py   ← Embeddings and similarities
└── exercises/
    ├── exercise_01_tokenization.py     ← Practice tokenization
    └── exercise_02_word_embeddings.py  ← Practice embeddings
```

---

## 🎓 Ready to Start?

### Next Step
👉 **Open `GETTING_STARTED.md`** for your learning path!

### Or Jump Right In
👉 **Open `01_tokenization.md`** to start learning!

---

**This is where text becomes numbers! After this, you'll understand the input pipeline of GPT!** 🚀

**Let's build the foundation of LLMs!** 🎊

---

**Module Status:** ✅ Complete (2 lessons)
**Prerequisites:** ✅ Modules 1-4
**Next Module:** Module 6 - Training & Fine-tuning

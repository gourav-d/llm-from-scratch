# What's New: Lessons 7 & 8

**Date:** March 16, 2026
**Module:** 07 - Reasoning and Coding Models
**Update:** Two new lessons completed!

---

## 🎉 Big News!

**Lessons 7 and 8 are now complete!** You've just unlocked two powerful lessons about code embeddings and training models on code - the same techniques used by GitHub Copilot!

**Module Progress:**
- Before: 60% complete (6/10 lessons)
- **Now: 80% complete (8/10 lessons)** 🎉

---

## 📦 What's Included

### ✨ Lesson 7: Code Embeddings & Understanding

**New Files:**
- 📄 `PART_B_CODING/07_code_embeddings.md` (850 lines)
- 💻 `examples/example_07_code_embeddings.py` (750 lines)

**You'll Learn:**
1. How to convert code to vector embeddings
2. Why code embeddings are different from text embeddings
3. How to build a semantic code search engine
4. How to detect code duplicates automatically
5. How GitHub finds "similar code"
6. How to measure code similarity (cosine similarity)
7. How to build code recommendation systems

**Working Examples:**
```python
# 1. Simple code embedder
embedder = SimpleCodeEmbedder(embedding_dim=128)
embedding = embedder.embed("def add(x, y): return x + y")

# 2. Weighted code embedder (better!)
weighted_embedder = WeightedCodeEmbedder(embedding_dim=128)

# 3. Code search engine
search_engine = CodeSearchEngine(weighted_embedder)
results = search_engine.search("function to add numbers", top_k=5)

# 4. Find duplicates
duplicates = search_engine.find_duplicates(threshold=0.85)

# 5. Code recommendations
recommender = CodeRecommender(search_engine)
suggestions = recommender.recommend_similar(my_code, top_k=3)
```

---

### ✨ Lesson 8: Training Models on Code (Codex-style)

**New Files:**
- 📄 `PART_B_CODING/08_training_on_code.md` (900 lines)
- 💻 `examples/example_08_code_training.py` (800 lines)

**You'll Learn:**
1. **Fill-in-the-Middle (FIM) training** - THE secret sauce behind Copilot!
2. How to prepare code datasets from GitHub/Stack Overflow
3. How to clean and filter code for quality
4. Data augmentation techniques for code
5. Multi-language training strategies
6. How OpenAI trained Codex
7. How to evaluate code models (perplexity, BLEU, Pass@k)

**Working Examples:**
```python
# 1. Data preprocessing
preprocessor = CodeDataPreprocessor()
cleaned_code = preprocessor.process_dataset(raw_code)

# 2. FIM transformation (CRITICAL!)
fim_transformer = FIMTransformer(fim_ratio=0.3)
training_samples = fim_transformer.transform_batch(cleaned_code)

# Example FIM format:
# <fim_prefix>def add(x, y):<fim_suffix>return result<fim_middle>result = x + y

# 3. Data augmentation
augmenter = CodeAugmenter()
variations = augmenter.augment(code, max_variations=5)

# 4. Simple code generator
generator = SimpleCodeGenerator(n=3)
generator.train(code_samples)
generated = generator.generate("def add", max_tokens=20)
```

---

## 🎯 Key Concepts Explained

### From Lesson 7: Code Embeddings

**Think of it like this (for .NET developers):**

Code embeddings are like converting C# code to `Vector<float>`:

```csharp
// In C#
Vector<float> GetCodeEmbedding(string code) {
    // Convert code to 768-dimensional vector
    // Similar code → Similar vectors
}

float CosineSimilarity(Vector<float> v1, Vector<float> v2) {
    return Vector.Dot(v1, v2) / (v1.Length() * v2.Length());
}
```

**Real-World Application:**
- GitHub's code search uses embeddings
- When you search "function to sort array", GitHub finds relevant code even if it doesn't contain those exact words!

---

### From Lesson 8: FIM Training

**The Big Secret: Fill-in-the-Middle (FIM)**

**Normal training (left-to-right):**
```
Input:  "def add(x, y):"
Output: "return x + y"
```

**FIM training:**
```
You know:
- What comes BEFORE: "def add(x, y):"
- What comes AFTER: "return result"

Predict what goes IN THE MIDDLE: "result = x + y\n"
```

**Why it matters:**
- GitHub Copilot doesn't just complete at the end of your file
- It completes IN THE MIDDLE of your functions!
- FIM training is what enables this

**In C# terms:**
```csharp
// You're writing code here:
public int Add(int x, int y) {
    // Cursor is HERE - Copilot suggests code
    // This requires FIM training!
}
```

---

## 🚀 Quick Start

### Run the Examples

**Lesson 7: Code Embeddings**
```bash
cd modules/07_reasoning_and_coding_models/examples
python example_07_code_embeddings.py
```

**Expected Output:**
```
====================================================================
CODE EMBEDDINGS & SEMANTIC SEARCH DEMO
====================================================================

Codebase size: 11 functions

DEMO 1: Simple Token-Based Embeddings
----------------------------------------------------------------------
Code 1: def add(x, y): return x + y...
Code 2: def sum_numbers(a, b): result = a + b...
Embedding dimension: 128
Cosine similarity: 0.9234

DEMO 2: Weighted Token Embeddings (Better!)
Weighted cosine similarity: 0.9567

DEMO 3: Semantic Code Search
Query: def add(x, y): return x + y

Top 5 similar functions:
1. [Score: 1.0000] def add(x, y): return x + y...
2. [Score: 0.9567] def sum_numbers(a, b): result = a + b return result...
3. [Score: 0.8234] def calculate_sum(num1, num2): return num1 + num2...
```

**Lesson 8: Training on Code**
```bash
cd modules/07_reasoning_and_coding_models/examples
python example_08_code_training.py
```

**Expected Output:**
```
====================================================================
CODE TRAINING DATA PIPELINE DEMO
====================================================================

Raw dataset size: 8 samples

STEP 1: Data Preprocessing
----------------------------------------------------------------------
Total samples: 8
Filtered out: 3
Cleaned samples: 5

STEP 2: Data Augmentation
----------------------------------------------------------------------
Augmenting code samples...
  3 variations created
  3 variations created
  ...
Augmented dataset size: 15 samples
Augmentation factor: 3.0x

STEP 3: Fill-in-the-Middle (FIM) Transformation
----------------------------------------------------------------------
Total training samples: 15
  FIM samples: 5 (33.3%)
  CLM samples: 10 (66.7%)

Example FIM sample:
  Input: <fim_prefix>def add(x, y):<fim_suffix>return result<fim_middle>...
  Target: result = x + y...
```

---

## 📚 What You Can Build Now

After completing these lessons, you can build:

### 1. Semantic Code Search Engine
```python
# Like GitHub search, but for your codebase
engine = CodeSearchEngine()
engine.index(my_codebase)
results = engine.search("function that validates email")
```

### 2. Code Duplicate Detector
```python
# Find similar/duplicate code automatically
duplicates = engine.find_duplicates(threshold=0.85)
for i, j, similarity in duplicates:
    print(f"Found {similarity*100}% similar code!")
```

### 3. Code Training Data Pipeline
```python
# Prepare data for training code models
pipeline = TrainingPipeline()
pipeline.clean(raw_code)
pipeline.augment()
pipeline.apply_fim(ratio=0.3)
training_data = pipeline.get_samples()
```

### 4. Simple Code Generator
```python
# Generate code from prompts
generator = SimpleCodeGenerator()
generator.train(code_samples)
code = generator.generate("def factorial", max_tokens=50)
```

---

## 🎓 Learning Path

**Day 1-2:** Read Lesson 7
- Understand what embeddings are
- Learn cosine similarity
- Study semantic search

**Day 3:** Practice Lesson 7
- Run example_07_code_embeddings.py
- Try the quiz questions
- Build a simple code search for your projects

**Day 4-5:** Read Lesson 8
- **Understand FIM** (this is CRITICAL!)
- Learn data preparation
- Study augmentation techniques

**Day 6:** Practice Lesson 8
- Run example_08_code_training.py
- Experiment with FIM transformations
- Create augmented dataset

**Day 7:** Project
- Build a mini code search tool
- Prepare training data from your code
- Experiment with different techniques

---

## 🧪 Quiz Yourself

Test your understanding:

**From Lesson 7:**
1. What makes code embeddings different from text embeddings?
2. What does a cosine similarity of 0.95 mean?
3. How does semantic code search work?

**From Lesson 8:**
1. What is Fill-in-the-Middle (FIM) training?
2. Why is FIM critical for code completion?
3. What are good sources for code training data?

**Answers available in the lesson files!**

---

## 📊 Content Stats

**Total New Content:**
- 3,300 lines of lessons & examples
- 35+ working code examples
- 8 quiz questions with detailed answers
- 4 practice exercises with solutions
- 8+ diagrams and visualizations

**Lesson 7 (Code Embeddings):**
- 850 lines of lesson content
- 750 lines of example code
- 15+ code examples
- 5 diagrams

**Lesson 8 (Training on Code):**
- 900 lines of lesson content
- 800 lines of example code
- 20+ code examples
- 3 diagrams

---

## 🔗 How It All Connects

```
Lesson 6: Code Tokenization
           ↓
Lesson 7: Code Embeddings ← Uses tokens to create vectors
           ↓
Lesson 8: Training on Code ← Uses embeddings for similarity
           ↓
Lesson 9: Code Generation ← Will use FIM-trained models
           ↓
Lesson 10: Code Evaluation ← Will test generated code
```

**Each lesson builds on the previous one!**

---

## 🎯 Next Steps

### This Week
1. ✅ Read Lesson 7 & 8 thoroughly
2. ✅ Run both example files
3. ✅ Complete quiz questions
4. ✅ Try practice exercises

### Next Week
- Start Lesson 9: Code Generation & Completion
- Build mini-Copilot prototype
- Only 2 more lessons to complete Module 7!

### Your Progress
```
Module 7: ████████████████░░░░ 80% (8/10 lessons)

Part A (Reasoning): ████████████████████ 100% ✅
Part B (Coding):    ████████████░░░░░░░░  60%

Almost there! 🎉
```

---

## 📖 Files to Review

**Essential Reading:**
1. `PART_B_CODING/07_code_embeddings.md` - How embeddings work
2. `PART_B_CODING/08_training_on_code.md` - How to train on code
3. `LESSONS_7_8_COMPLETE.md` - Detailed completion summary
4. `MODULE_PROGRESS.md` - Overall module status

**Code to Run:**
1. `examples/example_07_code_embeddings.py` - Search engine demo
2. `examples/example_08_code_training.py` - Training pipeline demo

**Quick Reference:**
- `quick_reference.md` - Updated with lessons 7 & 8 summaries

---

## 💡 Pro Tips

### For Lesson 7 (Embeddings)
- Start with simple averaging, then try weighted
- Experiment with different embedding dimensions (64, 128, 256)
- Try semantic search on your own codebase
- Compare token-level vs function-level embeddings

### For Lesson 8 (Training)
- **FIM is the most important concept** - make sure you understand it!
- Try different FIM ratios (10%, 30%, 50%)
- Experiment with data augmentation
- Collect real code from GitHub for practice

---

## 🎉 Achievements Unlocked

After completing these lessons:

✅ **Code Embeddings Expert** - Can convert code to vectors
✅ **Semantic Search Builder** - Can build code search engines
✅ **FIM Master** - Understand the secret behind Copilot
✅ **Data Pipeline Pro** - Can prepare code for training
✅ **80% Complete!** - Only 2 lessons to go!

---

## 🌟 What Makes These Lessons Special

### Lesson 7
- **Hands-on:** Build working search engine from scratch
- **Practical:** Use on real codebases immediately
- **Complete:** From theory to implementation
- **Compared:** Python vs C# throughout

### Lesson 8
- **Critical:** Learn THE technique behind Copilot (FIM)
- **Comprehensive:** Full training pipeline
- **Real-world:** Uses actual best practices
- **Actionable:** Can use to prepare your own datasets

---

## 📞 Need Help?

**If you're stuck:**
1. Re-read the relevant section
2. Run the example code
3. Check the quiz answers
4. Try the practice exercises

**Common Questions:**

**Q: What's the most important concept?**
A: Fill-in-the-Middle (FIM) training from Lesson 8!

**Q: Can I skip lessons?**
A: No - each lesson builds on previous ones

**Q: How long will this take?**
A: Lesson 7: 2-3 hours, Lesson 8: 4-5 hours

**Q: What if I don't understand embeddings?**
A: Think of them as "coordinates" for code - similar code has similar coordinates

---

## 🚀 You're Almost Done!

**Module 7 Progress: 80% Complete**

**What's left:**
- Lesson 9: Code Generation & Completion
- Lesson 10: Code Evaluation & Testing

**Then you'll have:**
- Complete understanding of reasoning models (o1-style)
- Complete understanding of coding models (Copilot-style)
- Skills to build your own AI coding assistant!

**Keep going - you're building world-class AI skills!** 🎓

---

## 📝 Summary

**Today you got:**
- 2 complete lessons (7 & 8)
- 3,300 lines of content
- 35+ code examples
- Full semantic search engine
- Complete training pipeline
- FIM transformation system

**Next up:**
- Lesson 9: Code Generation
- Build mini-Copilot
- Complete Module 7!

---

**Created:** March 16, 2026
**Updated:** MODULE_PROGRESS.md, quick_reference.md
**Status:** Lessons 7 & 8 Complete ✅

**Happy Learning!** 🎉

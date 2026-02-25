# Getting Started: Sentiment Analysis

**Step-by-step guide to building a movie review classifier**

---

## 📋 Prerequisites

✅ **Module 2: NumPy**
✅ **Module 3: Neural Networks (all 6 lessons)**
✅ **Project 1: Email Spam Classifier** (text preprocessing)
✅ **Project 2: MNIST Digits** (multi-layer networks)

**Time Required:** 3-4 hours

---

## 🎯 Learning Goals

By the end of this project:

1. ✅ Understand sentiment analysis (positive vs negative)
2. ✅ Learn word embeddings (better than bag-of-words!)
3. ✅ Work with longer text sequences
4. ✅ Bridge from simple NLP to transformers
5. ✅ Achieve 85-88% accuracy on movie reviews

---

## 🎬 What is Sentiment Analysis?

**Goal:** Determine if text expresses positive or negative sentiment

**Examples:**
```
"This movie was absolutely brilliant!" → POSITIVE ✓
"Waste of time and money" → NEGATIVE ✓
"It was okay, not great" → NEGATIVE (mixed)
```

**Real-World Uses:**
- Product reviews (Amazon, Yelp)
- Social media monitoring (Twitter sentiment)
- Customer feedback
- Brand reputation
- Movie recommendations

---

## 📊 Dataset

**IMDB Movie Reviews:**
- 50,000 reviews (25k train, 25k test)
- Binary labels: positive or negative
- Average length: ~200-300 words
- Real user reviews from imdb.com

---

## 🚦 Quick Start

### Step 1: Run Simple Version (bag-of-words)

```bash
cd projects/neural_networks/sentiment_analysis
python project_simple.py
```

**Expected output:**
```
Step 1: Loading IMDB reviews...
✓ 25,000 training reviews
✓ 25,000 test reviews
✓ Average length: 234 words

Step 2: Building vocabulary...
✓ Vocabulary size: 5,000 words

Step 3: Creating bag-of-words features...
✓ Feature matrix: (25000, 5000)

Step 4: Training neural network...
Epoch 10/30: Loss=0.312, Acc=86.2%, Test Acc=83.5%
Epoch 20/30: Loss=0.245, Acc=89.1%, Test Acc=85.2%
Epoch 30/30: Loss=0.198, Acc=91.3%, Test Acc=85.8%

Step 5: Testing...
✓ Test Accuracy: 85.8%

Done!
```

---

## 🔑 Key Concepts

### 1. Sentiment vs Spam Classification

| Aspect | Spam (Project 1) | Sentiment (Project 3) |
|--------|------------------|----------------------|
| **Challenge** | Easier (clear patterns) | Harder (context matters) |
| **Key words** | "free", "buy", "win" | "amazing", "terrible", "boring" |
| **Negation** | Less important | Critical! ("not good") |
| **Context** | Word presence enough | Word order matters |
| **Accuracy** | 93-95% | 85-88% |

### 2. Word Embeddings

**Bag-of-words limitation:**
```
"good" → [0,0,1,0,0,0,...]
"great" → [0,0,0,1,0,0,...]
"excellent" → [0,0,0,0,1,0,...]

Problem: No relationship between similar words!
```

**Word embeddings solution:**
```
"good" → [0.7, 0.3, -0.2, ...]      ↘
"great" → [0.72, 0.28, -0.18, ...]   → Similar vectors!
"excellent" → [0.75, 0.25, -0.22, ...]↗

"bad" → [-0.6, -0.4, 0.3, ...]       → Different vector!
```

**Benefits:**
- Similar words get similar vectors
- Learns from data
- Captures semantic meaning
- Better generalization

---

## 📈 Architecture Comparison

### Simple Version (Bag-of-Words)
```
Input (5000) → Hidden (64) → Output (1)
 [word counts]    [ReLU]      [Sigmoid]

Fast but limited - ignores word order
```

### Advanced Version (Word Embeddings)
```
Input (sequence of word IDs)
    ↓
Embedding Layer (learns 100D vectors)
    ↓
Hidden (64, ReLU)
    ↓
Output (1, Sigmoid)

Slower but better - learns word meanings!
```

---

## 🎓 What You'll Learn

### From Project Simple
✅ Sentiment classification basics
✅ Longer text handling
✅ Challenging test case (negation, sarcasm)
✅ Baseline performance with bag-of-words

### From Project Main (Advanced)
✅ Word embedding layer
✅ Learning word representations
✅ Average pooling over sequences
✅ Better handling of word order
✅ Bridge to transformers!

---

## 🔍 Understanding Failures

**Where simple bag-of-words fails:**

```
Review: "This movie was not good at all"
Bag-of-words sees: "good" → POSITIVE prediction ✗
Should be: NEGATIVE

Review: "I expected it to be terrible, but it was amazing!"
Bag-of-words sees: "terrible" → NEGATIVE prediction ✗
Should be: POSITIVE
```

**Word embeddings help but not perfect:**
- Still struggles with complex negation
- Doesn't fully understand word order
- → Need attention mechanism (Module 4!)

---

## 💡 Connection to Transformers

This project is the **bridge to Module 4**!

**What you have now:**
- ✅ Text → numbers (tokenization)
- ✅ Word embeddings
- ✅ Neural network classification

**What transformers add (Module 4):**
- 🔮 **Attention mechanism** - focus on relevant words
- 🔮 **Positional encoding** - understand word order
- 🔮 **Self-attention** - words relate to each other
- 🔮 **Complete context** - understand full meaning

**After Module 4:**
You'll build a mini-GPT that understands context 10x better!

---

## ⏭️ Next Steps After This Project

**Option 1: Move to Module 4 (Recommended!)**
You're ready for transformers!
- Attention mechanism
- GPT architecture
- Build mini-LLM

**Option 2: Improve Sentiment**
- Try GloVe embeddings
- Implement LSTM (recurrent)
- Try different architectures

**Option 3: Custom Application**
- Collect your own reviews
- Product review classifier
- Social media sentiment

---

## ✅ Success Criteria

Complete this project when you can:

✅ Explain word embeddings vs bag-of-words
✅ Understand why sentiment is harder than spam
✅ Train both simple and advanced versions
✅ Achieve 85%+ test accuracy
✅ Understand where the model fails
✅ Ready for transformers (Module 4!)

---

**Let's build a sentiment classifier!** 🚀

👉 **Next:** Run `python project_simple.py`

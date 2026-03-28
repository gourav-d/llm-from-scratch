"""
MINI SENTIMENT ANALYSIS - Build From Scratch
============================================

OBJECTIVE:
  Read a movie review → decide: POSITIVE (1) or NEGATIVE (0)

KEY INSIGHT — READ THIS FIRST:
  Sentiment analysis is STRUCTURALLY IDENTICAL to spam classification.
  Both are binary classification using bag-of-words.
  The only difference is the data and what the words mean.

  Spam      → looks for spam words  (free, win, click, prize)
  Sentiment → looks for feeling words (amazing, awful, loved, boring)

  Same network. Same math. Different data. That's it.

HOW WE DECIDE LAYER SIZES:
  ┌─────────────────────────────────────────────────────────┐
  │  INPUT  size = vocabulary size  (count of unique words)  │
  │  HIDDEN size = we choose = 4                             │
  │  OUTPUT size = 1  (positive probability 0.0 → 1.0)      │
  │                                                          │
  │  W1 shape = (vocab_size, 4)                              │
  │  W2 shape = (4, 1)                                       │
  └─────────────────────────────────────────────────────────┘

WHAT CHANGES vs spam_mini.py:
  - reviews[] instead of emails[]
  - Different words in vocabulary (so vocab_size will differ)
  - Everything else is identical
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')  # fix Unicode arrows on Windows console

import numpy as np
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# STEP 1: DATA
# 12 reviews — positive words vs negative words.
# Notice: the network learns which words → positive/negative
# ─────────────────────────────────────────────────────────────
reviews = [
    # (review text,                              label) 1=positive 0=negative
    ("amazing wonderful movie loved it great",       1),
    ("brilliant acting fantastic story enjoyed",     1),
    ("excellent film best performance loved",        1),
    ("great story enjoyed wonderful acting",         1),
    ("fantastic brilliant loved amazing film",       1),
    ("wonderful best enjoyed great performance",     1),
    ("boring terrible movie hated awful waste",      0),
    ("worst film ever horrible acting bad",          0),
    ("dull slow terrible hated awful plot",          0),
    ("bad movie waste time horrible boring",         0),
    ("awful boring film worst hated terrible",       0),
    ("horrible bad acting terrible worst plot",      0),
]

# ─────────────────────────────────────────────────────────────
# STEP 2: VOCABULARY
# Collect all unique words from all 12 reviews.
# vocab_size = INPUT SIZE of the network
# ─────────────────────────────────────────────────────────────
all_words  = set(word for text, _ in reviews for word in text.split())
vocab      = sorted(all_words)
vocab_size = len(vocab)
w2i        = {w: i for i, w in enumerate(vocab)}

print(f"Vocabulary size (= INPUT SIZE): {vocab_size}")
print(f"Words: {vocab}\n")

# ─────────────────────────────────────────────────────────────
# STEP 3: TEXT → VECTOR (Bag of Words)
#
# "amazing wonderful movie" → [1, 0, 0, 1, 0, 0, ..., 1, 0]
#                              ↑                              ↑
#                           "amazing"                    "wonderful"
#
# Each slot = 1 if that vocabulary word appears in the review
# ─────────────────────────────────────────────────────────────
def to_vec(text):
    v = np.zeros(vocab_size)
    for word in text.split():
        if word in w2i:
            v[w2i[word]] = 1
    return v

X = np.array([to_vec(t) for t, _ in reviews])           # shape: (12, vocab_size)
y = np.array([lbl for _, lbl in reviews]).reshape(-1, 1) # shape: (12, 1)

print(f"X shape: {X.shape}  ← (num_reviews=12, input_size={vocab_size})")
print(f"y shape: {y.shape}  ← (num_reviews=12, output_size=1)\n")

# ─────────────────────────────────────────────────────────────
# STEP 4: WEIGHTS
# Exact same structure as spam classifier.
# W1: (vocab_size, 4)   W2: (4, 1)
# ─────────────────────────────────────────────────────────────
hidden_size = 4

W1 = np.random.randn(vocab_size,  hidden_size) * 0.1   # (vocab_size, 4)
b1 = np.zeros(hidden_size)                              # (4,)
W2 = np.random.randn(hidden_size, 1)           * 0.1   # (4, 1)
b2 = np.zeros(1)                                        # (1,)

print(f"W1: {W1.shape}  ← (input={vocab_size}, hidden={hidden_size})")
print(f"W2: {W2.shape}  ← (hidden={hidden_size}, output=1)\n")

relu    = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# ─────────────────────────────────────────────────────────────
# STEP 5: TRAINING
# Identical to spam classifier. Same forward, same backward, same update.
# This is the beauty of neural networks: one framework, many problems.
# ─────────────────────────────────────────────────────────────
lr = 0.5

for epoch in range(300):
    # FORWARD
    Z1 = X  @ W1 + b1    # (12, vocab_size) @ (vocab_size, 4) → (12, 4)
    A1 = relu(Z1)         # shape: (12, 4)
    Z2 = A1 @ W2 + b2    # (12, 4) @ (4, 1) → (12, 1)
    A2 = sigmoid(Z2)      # shape: (12, 1) — probabilities

    # LOSS
    loss = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))

    # BACKWARD
    dZ2 = A2 - y
    dW2 = A1.T @ dZ2 / len(X);    db2 = dZ2.mean(axis=0)
    dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
    dW1 = X.T  @ dZ1 / len(X);    db1 = dZ1.mean(axis=0)

    # UPDATE
    W1 -= lr * dW1;  b1 -= lr * db1
    W2 -= lr * dW2;  b2 -= lr * db2

    if epoch % 100 == 0:
        acc = np.mean((A2 > 0.5) == y)
        print(f"Epoch {epoch:3d}: loss={loss:.4f}, accuracy={acc:.0%}")

# ─────────────────────────────────────────────────────────────
# STEP 6: PREDICT NEW REVIEWS
# ─────────────────────────────────────────────────────────────
def predict(text):
    vec = to_vec(text).reshape(1, -1)
    h   = relu(vec @ W1 + b1)
    out = sigmoid(h @ W2 + b2)[0][0]
    label = "POSITIVE" if out > 0.5 else "NEGATIVE"
    print(f"  '{text}'")
    print(f"  → {label}  ({out:.0%} positive probability)\n")

print("\n── Predictions ──────────────────────────────────────────")
predict("loved amazing wonderful film")          # expect: POSITIVE
predict("boring terrible awful waste")           # expect: NEGATIVE
predict("amazing but boring story")              # mixed — interesting!

# ─────────────────────────────────────────────────────────────
# REFLECTION — Compare to spam_mini.py
# ─────────────────────────────────────────────────────────────
# Open both files side by side. What's different?
#   ✓ Different data (reviews vs emails)
#   ✓ Different words in vocabulary (so vocab_size differs)
#   ✗ Everything else is 100% the same
#
# This tells you: bag-of-words + binary classification is a PATTERN.
# Spam, sentiment, topic detection — all use this same blueprint.

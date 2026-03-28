"""
MINI EMAIL SPAM CLASSIFIER - Build From Scratch
===============================================

OBJECTIVE:
  Read an email → decide: SPAM (1) or HAM/not-spam (0)

HOW WE SOLVE IT (the full plan before any code):
  Step 1: Convert email text into numbers  (computers only understand numbers)
  Step 2: Feed numbers into neural network
  Step 3: Network outputs a single probability (0.0 → 1.0)
          > 0.5 = SPAM,  ≤ 0.5 = HAM

HOW WE DECIDE LAYER SIZES (read this before looking at the code):
  ┌─────────────────────────────────────────────────────────┐
  │  INPUT  size = vocabulary size  (determined by YOUR DATA) │
  │  HIDDEN size = you choose freely (we picked 4)            │
  │  OUTPUT size = 1  (binary answer: spam probability)       │
  │                                                           │
  │  Weight W1 shape = (input_size, hidden_size) = (20, 4)   │
  │  Weight W2 shape = (hidden_size, output_size) = (4, 1)   │
  └─────────────────────────────────────────────────────────┘

MATRIX MULTIPLICATION RULE (memorize this one rule):
  If A has shape (m, n) and B has shape (n, k)
  then A @ B has shape (m, k)
  → middle dimension must match, and it disappears

  Example in this network:
    X  (12, 20) @ W1 (20, 4) → Z1 (12, 4)   ← hidden layer activations
    A1 (12,  4) @ W2 ( 4, 1) → Z2 (12, 1)   ← output probabilities

C# analogy:
  This whole file is like a Console App with a static Train() and Predict() method.
  numpy arrays = List<double[]> but way faster and supports math operators directly.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')  # fix Unicode arrows on Windows console

import numpy as np         # numpy = math library (matrix operations)
np.random.seed(42)         # fix randomness so results are the same every run

# ─────────────────────────────────────────────────────────────
# STEP 1: DATA
# These 12 emails are small enough to read completely.
# Read them! You should be able to predict the label yourself.
# ─────────────────────────────────────────────────────────────
emails = [
    # (email text,                          label)  1=spam 0=ham
    ("win free money now click prize",           1),   # spam
    ("claim prize reward cash offer today",      1),   # spam
    ("free pills discount buy now",              1),   # spam
    ("you won lottery congratulations prize",    1),   # spam
    ("urgent free reward click win cash",        1),   # spam
    ("cheap offer buy now free discount",        1),   # spam
    ("meeting agenda tomorrow team",             0),   # ham
    ("project deadline review report",           0),   # ham
    ("lunch sync tomorrow office team",          0),   # ham
    ("schedule call team project review",        0),   # ham
    ("update required meeting agenda",           0),   # ham
    ("report due friday please review",          0),   # ham
]

# ─────────────────────────────────────────────────────────────
# STEP 2: BUILD VOCABULARY
# Collect every unique word seen across all 12 emails.
# The vocabulary IS your input layer size.
# ─────────────────────────────────────────────────────────────
all_words  = set(word for text, _ in emails for word in text.split())
vocab      = sorted(all_words)          # sort = consistent order every run
vocab_size = len(vocab)                 # INPUT SIZE = this number
w2i        = {w: i for i, w in enumerate(vocab)}   # word → index lookup

print(f"Vocabulary size (= INPUT SIZE): {vocab_size}")
print(f"Words: {vocab}\n")

# ─────────────────────────────────────────────────────────────
# STEP 3: CONVERT EMAILS TO VECTORS  (Bag of Words)
#
# Each email becomes a fixed-length list of 0s and 1s.
# Length = vocab_size (one slot per word in vocabulary)
#
# Example: vocab = ["buy","cash","click","free","money","win",...]
#   "win free money"  → [0, 0, 0, 1, 1, 1, ...]  (1 where word exists)
#   "team meeting"    → [0, 0, 0, 0, 0, 0, ...]  (all zeros if no vocab words)
#
# WHY THIS WORKS: the network sees which spam-words are present
# ─────────────────────────────────────────────────────────────
def to_vec(text):
    v = np.zeros(vocab_size)            # start with all zeros, shape: (vocab_size,)
    for word in text.split():
        if word in w2i:
            v[w2i[word]] = 1            # set slot to 1 if word is in this email
    return v

X = np.array([to_vec(t) for t, _ in emails])        # shape: (12, vocab_size)
y = np.array([lbl for _, lbl in emails]).reshape(-1, 1)  # shape: (12, 1)

print(f"X shape: {X.shape}  ← (num_emails={len(emails)}, input_size={vocab_size})")
print(f"y shape: {y.shape}  ← (num_emails={len(emails)}, output_size=1)\n")

# ─────────────────────────────────────────────────────────────
# STEP 4: INITIALIZE WEIGHTS
#
# W1 connects input → hidden:  shape (input_size, hidden_size)
# W2 connects hidden → output: shape (hidden_size, output_size)
# Biases (b1, b2) shift the output up or down, one per neuron.
# ─────────────────────────────────────────────────────────────
hidden_size = 4   # WE CHOOSE THIS (hyperparameter). Small data = small network.

W1 = np.random.randn(vocab_size,  hidden_size) * 0.1   # shape: (vocab_size, 4)
b1 = np.zeros(hidden_size)                              # shape: (4,)
W2 = np.random.randn(hidden_size, 1)           * 0.1   # shape: (4, 1)
b2 = np.zeros(1)                                        # shape: (1,)

print(f"W1: {W1.shape}  ← (input_size={vocab_size}, hidden_size={hidden_size})")
print(f"W2: {W2.shape}  ← (hidden_size={hidden_size}, output_size=1)\n")

# ─────────────────────────────────────────────────────────────
# STEP 5: ACTIVATION FUNCTIONS
#
# ReLU   (hidden layer): if value < 0 → set to 0. Keeps positives as-is.
#   Why? It adds non-linearity. Without it, all layers collapse into one.
#
# Sigmoid (output layer): squishes ANY number → range [0.0, 1.0]
#   Why? We need a probability. 0=definitely ham, 1=definitely spam.
# ─────────────────────────────────────────────────────────────
relu    = lambda x: np.maximum(0, x)        # C# equivalent: Math.Max(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))   # formula: 1 / (1 + e^(-x))

# ─────────────────────────────────────────────────────────────
# STEP 6: TRAINING LOOP
#
# Each iteration = one "epoch" (one full pass through all 12 emails)
# Forward pass  → calculate predictions
# Loss          → measure how wrong we are
# Backward pass → calculate gradients (which direction to fix each weight)
# Update        → nudge weights toward correct direction
# ─────────────────────────────────────────────────────────────
lr = 0.5   # learning rate: how big each weight update step is

for epoch in range(300):

    # ── FORWARD PASS ─────────────────────────────────────────
    Z1 = X  @ W1 + b1    # (12, vocab_size) @ (vocab_size, 4) → shape (12, 4)
    A1 = relu(Z1)         # apply ReLU → shape stays (12, 4)
    Z2 = A1 @ W2 + b2    # (12, 4) @ (4, 1) → shape (12, 1)
    A2 = sigmoid(Z2)      # apply Sigmoid → probabilities, shape (12, 1)

    # ── LOSS (Binary Cross-Entropy) ──────────────────────────
    # Measures how wrong we are. Lower = better. Target = 0.
    loss = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))

    # ── BACKWARD PASS (Gradients) ────────────────────────────
    # Think of gradients as: "if I increase this weight a tiny bit,
    # how much does the loss change?" We want to decrease the loss.
    dZ2 = A2 - y                      # how wrong is the output? shape: (12, 1)
    dW2 = A1.T @ dZ2 / len(X)        # W2 gradient,  shape: (4, 1)
    db2 = dZ2.mean(axis=0)            # b2 gradient,  shape: (1,)
    dA1 = dZ2 @ W2.T                  # error passed back to hidden, shape: (12, 4)
    dZ1 = dA1 * (Z1 > 0)             # ReLU gradient: 0 where Z1 was ≤ 0
    dW1 = X.T @ dZ1 / len(X)         # W1 gradient,  shape: (vocab_size, 4)
    db1 = dZ1.mean(axis=0)            # b1 gradient,  shape: (4,)

    # ── UPDATE WEIGHTS ───────────────────────────────────────
    # Subtract gradient * learning_rate (move opposite to gradient)
    W1 -= lr * dW1;   b1 -= lr * db1
    W2 -= lr * dW2;   b2 -= lr * db2

    if epoch % 100 == 0:
        acc = np.mean((A2 > 0.5) == y)
        print(f"Epoch {epoch:3d}: loss={loss:.4f}, accuracy={acc:.0%}")

# ─────────────────────────────────────────────────────────────
# STEP 7: PREDICT A NEW EMAIL
# ─────────────────────────────────────────────────────────────
def predict(text):
    vec = to_vec(text).reshape(1, -1)           # shape: (1, vocab_size)
    h   = relu(vec @ W1 + b1)                   # shape: (1, hidden_size)
    out = sigmoid(h @ W2 + b2)[0][0]            # single probability
    label = "SPAM" if out > 0.5 else "HAM"
    print(f"  '{text}'")
    print(f"  → {label}  ({out:.0%} spam probability)\n")

print("\n── Predictions ──────────────────────────────────────────")
predict("free money click win prize")        # expect: SPAM
predict("team meeting report agenda")        # expect: HAM
predict("free meeting tomorrow")             # mixed — what does it predict?

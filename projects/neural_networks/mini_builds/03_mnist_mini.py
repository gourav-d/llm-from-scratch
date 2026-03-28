"""
MINI DIGIT CLASSIFIER - Build From Scratch
==========================================

OBJECTIVE:
  Look at a handwritten digit image → predict which digit it is (0-9)

WHY THIS IS DIFFERENT FROM SPAM / SENTIMENT:
  Spam & Sentiment → 1 output  (binary: yes or no)
  Digits           → 10 outputs (one per digit class: 0,1,2,...,9)

  We need 10 outputs because we need one probability per digit.
  The digit with the HIGHEST probability is our prediction.

DATA WE USE:
  sklearn's "digits" dataset (NOT the full 70k MNIST).
  - 1,797 images  (small enough to train quickly on any laptop)
  - Each image = 8×8 pixels = 64 numbers per image
  - Full MNIST uses 28×28 = 784 pixels (same idea, just bigger)

HOW WE DECIDE LAYER SIZES:
  ┌──────────────────────────────────────────────────────────────┐
  │  INPUT  size = 64    (8×8 pixels, flattened into 64 numbers)  │
  │  HIDDEN size = 32    (we choose — between input and output)   │
  │  OUTPUT size = 10    (one output per digit: 0 through 9)      │
  │                                                               │
  │  W1 shape = (64, 32)   ← (input_size, hidden_size)           │
  │  W2 shape = (32, 10)   ← (hidden_size, output_size)          │
  └──────────────────────────────────────────────────────────────┘

HOW THE IMAGE BECOMES A VECTOR:
  Image (8×8 grid):         Flattened vector (64 numbers):
  [[ 0,  0, 12, 15, ...],   [0, 0, 12, 15, ..., 0, 0, 8, 16, ...]
   [ 0,  0,  8, 16, ...],    ↑ 64 numbers ↑
   ...]
  → Just read rows left-to-right, one after another
  INPUT SIZE = 64 (8 × 8)

HOW SOFTMAX IS DIFFERENT FROM SIGMOID:
  Sigmoid → one probability independently (for binary)
  Softmax → 10 probabilities that ALL ADD UP TO 1.0
  Example: [0.02, 0.01, 0.05, 0.85, 0.02, ...] ← digit "3" most likely
            digit0 digit1 digit2 digit3 digit4
  Pick the index with the highest value → that's our prediction.

MATRIX SHAPES THROUGH THE NETWORK (N = number of images):
  X      (N, 64) @ W1 (64, 32) → Z1 (N, 32)   input→hidden
  relu(Z1)       = A1 (N, 32)                  hidden activations
  A1     (N, 32) @ W2 (32, 10) → Z2 (N, 10)   hidden→output
  softmax(Z2)    = A2 (N, 10)                  10 probabilities per image
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')  # fix Unicode arrows on Windows console

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# sklearn's digits dataset: 1797 images, 8×8 pixels, digits 0-9
# ─────────────────────────────────────────────────────────────
digits = load_digits()                    # built-in dataset, no download needed

X_all = digits.data / 16.0               # pixel values 0-16, normalize to 0-1
y_all = digits.target                    # labels: 0, 1, 2, ..., 9

# X_all shape: (1797, 64)  ← 1797 images, each is 64 pixels (8×8 flattened)
# y_all shape: (1797,)     ← one label per image

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)
# X_train: (1437, 64)   X_test: (360, 64)

print(f"Train: {X_train.shape}  ← ({len(X_train)} images, {X_train.shape[1]} pixels)")
print(f"Test:  {X_test.shape}")
print(f"Input size  = {X_train.shape[1]}  (8×8 pixels flattened)")
print(f"Num classes = {len(np.unique(y_train))}  (digits 0-9)\n")

# ─────────────────────────────────────────────────────────────
# STEP 2: ONE-HOT ENCODE LABELS
#
# The network needs labels as VECTORS, not single numbers.
# Why? Because we have 10 outputs (one per digit).
# We need to tell it WHICH output should be 1.0 and which 0.0.
#
# digit 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#             ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
#             0  1  2  3  4  5  6  7  8  9
#
# C# analogy: like a bool[10] where only index=digit is true
# ─────────────────────────────────────────────────────────────
def one_hot(y, num_classes=10):
    out = np.zeros((len(y), num_classes))   # shape: (N, 10) — all zeros
    out[np.arange(len(y)), y] = 1           # put 1 at the correct digit position
    return out

Y_train = one_hot(y_train)   # shape: (1437, 10)
Y_test  = one_hot(y_test)    # shape: (360, 10)

print(f"Y_train shape: {Y_train.shape}  ← (num_images, num_classes)")
print(f"Example: digit {y_train[0]} → {Y_train[0]}\n")

# ─────────────────────────────────────────────────────────────
# STEP 3: INITIALIZE WEIGHTS
# ─────────────────────────────────────────────────────────────
input_size  = 64    # 8×8 pixels — determined by data
hidden_size = 32    # we choose — picked between 64 and 10
output_size = 10    # one per digit (0-9) — determined by problem

W1 = np.random.randn(input_size,  hidden_size) * 0.01   # shape: (64, 32)
b1 = np.zeros(hidden_size)                               # shape: (32,)
W2 = np.random.randn(hidden_size, output_size) * 0.01   # shape: (32, 10)
b2 = np.zeros(output_size)                               # shape: (10,)

print(f"W1: {W1.shape}  ← (input_size={input_size}, hidden_size={hidden_size})")
print(f"W2: {W2.shape}  ← (hidden_size={hidden_size}, output_size={output_size})\n")

# ─────────────────────────────────────────────────────────────
# STEP 4: ACTIVATION FUNCTIONS
#
# ReLU    — same as before (hidden layer)
# Softmax — NEW: converts 10 raw scores → 10 probabilities summing to 1.0
#   Why not sigmoid? Sigmoid treats each output independently.
#   Softmax makes them compete — the more confident about "3", the less
#   probability goes to all others. This is what we want for classification.
# ─────────────────────────────────────────────────────────────
def relu(x):
    return np.maximum(0, x)                          # shape unchanged

def softmax(x):
    # Subtract max per row for numerical stability (prevents overflow)
    e = np.exp(x - x.max(axis=1, keepdims=True))    # e^scores
    return e / e.sum(axis=1, keepdims=True)          # normalize so each row sums to 1

# ─────────────────────────────────────────────────────────────
# STEP 5: TRAINING LOOP
# ─────────────────────────────────────────────────────────────
lr = 0.5   # higher learning rate helps converge faster on this dataset

for epoch in range(500):
    # ── FORWARD ──────────────────────────────────────────────
    Z1 = X_train @ W1 + b1   # (1437, 64) @ (64, 32) → shape (1437, 32)
    A1 = relu(Z1)              # shape: (1437, 32)
    Z2 = A1 @ W2 + b2         # (1437, 32) @ (32, 10) → shape (1437, 10)
    A2 = softmax(Z2)           # shape: (1437, 10) — 10 probs per image, sum=1

    # ── LOSS: Categorical Cross-Entropy ──────────────────────
    # For each image: look at the probability given to the CORRECT digit.
    # If that probability is high (near 1.0) → low loss.
    # If that probability is low (near 0.0) → high loss.
    loss = -np.mean(np.sum(Y_train * np.log(A2 + 1e-8), axis=1))

    # ── BACKWARD ─────────────────────────────────────────────
    # Softmax + cross-entropy gradient simplifies nicely to (predicted - actual)
    dZ2 = (A2 - Y_train) / len(X_train)  # shape: (1437, 10)
    dW2 = A1.T @ dZ2                     # shape: (32, 10)
    db2 = dZ2.sum(axis=0)                # shape: (10,)
    dA1 = dZ2 @ W2.T                     # shape: (1437, 32)
    dZ1 = dA1 * (Z1 > 0)                # ReLU gradient, shape: (1437, 32)
    dW1 = X_train.T @ dZ1               # shape: (64, 32)
    db1 = dZ1.sum(axis=0)               # shape: (32,)

    # ── UPDATE ───────────────────────────────────────────────
    W1 -= lr * dW1;  b1 -= lr * db1
    W2 -= lr * dW2;  b2 -= lr * db2

    if epoch % 100 == 0:
        # argmax: pick the index (digit) with highest probability
        preds = A2.argmax(axis=1)              # shape: (1437,) — predicted digit
        acc   = np.mean(preds == y_train)
        print(f"Epoch {epoch:3d}: loss={loss:.4f}, train_acc={acc:.0%}")

# ─────────────────────────────────────────────────────────────
# STEP 6: EVALUATE ON TEST SET
# Test set = images the network NEVER saw during training
# ─────────────────────────────────────────────────────────────
Z1t = X_test  @ W1 + b1
A1t = relu(Z1t)
Z2t = A1t @ W2 + b2
A2t = softmax(Z2t)

preds_test = A2t.argmax(axis=1)              # predicted digits
test_acc   = np.mean(preds_test == y_test)

print(f"\nTest accuracy: {test_acc:.1%}")

# Show a few sample predictions
print("\nSample predictions (predicted → actual):")
for i in range(8):
    correct = "✓" if preds_test[i] == y_test[i] else "✗"
    print(f"  Predicted: {preds_test[i]}  |  Actual: {y_test[i]}  {correct}")

# ─────────────────────────────────────────────────────────────
# REFLECTION — What's different from spam/sentiment?
#
# Spam/Sentiment (binary):          Digits (multi-class):
# ─────────────────────────────── ─────────────────────────────────
# Input: bag-of-words vector      Input: flattened pixel vector
# Output size: 1                  Output size: 10
# Activation: sigmoid             Activation: softmax
# Loss: binary cross-entropy      Loss: categorical cross-entropy
# Prediction: > 0.5 = yes         Prediction: argmax = digit
#
# Everything else is the same!
# ─────────────────────────────────────────────────────────────

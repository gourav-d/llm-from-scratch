"""
Example 01: Basic Attention Mechanism - PyTorch Version

SAME example as the NumPy version, but using PyTorch!

WHY learn PyTorch after NumPy?
  - NumPy : Great for understanding the math - NO automatic gradients
  - PyTorch: The industry standard for AI research - HAS automatic gradients

NumPy vs PyTorch at a glance:
  ┌──────────────────────────────┬─────────────────────────────┐
  │ NumPy (What you learned)     │ PyTorch (Industry tool)     │
  ├──────────────────────────────┼─────────────────────────────┤
  │ np.random.seed(42)           │ torch.manual_seed(42)       │
  │ np.random.randn(6, 4)        │ torch.randn(6, 4)           │
  │ Q @ K.T                      │ Q @ K.T  (SAME!)            │
  │ custom softmax function      │ F.softmax(x, dim=-1)        │
  │ arr.shape                    │ tensor.shape  (SAME!)       │
  │ arr  (for matplotlib)        │ tensor.detach().numpy()     │
  │ No training support          │ requires_grad=True          │
  └──────────────────────────────┴─────────────────────────────┘

In C#/.NET terms:
  NumPy   = MathNet.Numerics  (manual math, no training)
  PyTorch = ML.NET            (full framework with automatic differentiation)
"""

import torch                          # Main PyTorch library (like 'using System')
import torch.nn.functional as F       # Built-in functions like softmax, relu
import numpy as np                    # Still use NumPy for seaborn/matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed so results are reproducible
# NumPy:   np.random.seed(42)
# PyTorch: torch.manual_seed(42)
torch.manual_seed(42)

print("=" * 70)
print("BASIC ATTENTION MECHANISM - PyTorch Version")
print("=" * 70)

# ==============================================================================
# PART 1: Understanding Query, Key, Value
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Query, Key, Value Concept")
print("=" * 70)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
print(f"\nInput sentence: {' '.join(sentence)}")

d_model = 4  # Dimension of word embeddings

# NumPy:   word_embeddings = np.random.randn(len(sentence), d_model)
# PyTorch: word_embeddings = torch.randn(len(sentence), d_model)
word_embeddings = torch.randn(len(sentence), d_model)

print(f"\nWord embeddings shape: {word_embeddings.shape}")
print(f"  → PyTorch shows 'torch.Size([6, 4])' instead of '(6, 4)' - same idea!")
print(f"\nExample - embedding for '{sentence[0]}':")
print(word_embeddings[0])
print(f"  → Values wrapped in 'tensor(...)' - that's a PyTorch tensor, not a NumPy array")

# ==============================================================================
# PART 2: Creating Query, Key, Value Matrices
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Creating Queries, Keys, and Values")
print("=" * 70)

# For basic attention: Q, K, V are the same embeddings
Q = word_embeddings    # Queries - what is each word looking for?
K = word_embeddings    # Keys    - what does each word contain?
V = word_embeddings    # Values  - the actual information in each word

print(f"Q shape: {Q.shape} | K shape: {K.shape} | V shape: {V.shape}")

# ==============================================================================
# PART 3: Computing Attention Scores
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Computing Attention Scores")
print("=" * 70)

# The @ operator (matrix multiply) works the SAME in both NumPy and PyTorch!
# NumPy:   attention_scores = Q @ K.T
# PyTorch: attention_scores = Q @ K.T    (IDENTICAL SYNTAX!)
attention_scores = Q @ K.T         # Shape: (6, 4) @ (4, 6) = (6, 6)

print(f"Attention scores shape: {attention_scores.shape}")
print("\nAttention scores (before scaling):")
print(attention_scores)

# Scale by sqrt(d_k) - prevents scores from getting too large
d_k = d_model
scaled_scores = attention_scores / (d_k ** 0.5)    # Python math works for both

print(f"\nScaling factor: sqrt({d_k}) = {d_k ** 0.5:.2f}")
print("\nScaled attention scores:")
print(scaled_scores)

# ==============================================================================
# PART 4: Applying Softmax to Get Attention Weights
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Converting Scores to Weights with Softmax")
print("=" * 70)

print("""
KEY DIFFERENCE: Softmax implementation

  NumPy version: We wrote our OWN softmax function (12 lines of code!)
    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

  PyTorch version: It's BUILT-IN in torch.nn.functional
    attention_weights = F.softmax(scaled_scores, dim=-1)

  Note: 'axis' in NumPy = 'dim' in PyTorch  (same concept, different name)

  In C#/.NET terms:
    NumPy  = writing your own Math.Exp loop
    PyTorch = using a library like MathHelper.Softmax() built by experts
""")

# NumPy:   attention_weights = softmax(scaled_scores, axis=-1)
# PyTorch: attention_weights = F.softmax(scaled_scores, dim=-1)
attention_weights = F.softmax(scaled_scores, dim=-1)

print("Attention weights (each row sums to 1.0):")
print(attention_weights)

print("\nVerify each row sums to 1.0:")
for i, word in enumerate(sentence):
    # .item() converts a single-element tensor to a plain Python number
    # In NumPy you would just call float() or access the value directly
    row_sum = attention_weights[i].sum().item()
    print(f"  '{word}': {row_sum:.6f}")

# ==============================================================================
# PART 5: Computing Weighted Sum of Values
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Computing Context-Aware Representations")
print("=" * 70)

# NumPy:   attention_output = attention_weights @ V
# PyTorch: attention_output = attention_weights @ V    (IDENTICAL!)
attention_output = attention_weights @ V               # Shape: (6, 6) @ (6, 4) = (6, 4)

print(f"Attention output shape: {attention_output.shape}")
print(f"\nExample - new representation for '{sentence[2]}' (sat):")
print(f"Original embedding: {V[2]}")
print(f"After attention:    {attention_output[2]}")

# ==============================================================================
# PART 6: Visualizing Attention Patterns
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Visualizing Attention Weights")
print("=" * 70)

print("""
IMPORTANT: Matplotlib/Seaborn cannot plot PyTorch tensors directly!
We must convert to NumPy first:

  PyTorch → NumPy:  tensor.detach().numpy()
    → .detach()  tells PyTorch "stop tracking gradients for this tensor"
    → .numpy()   converts the tensor to a NumPy array

  In C# terms: Like calling .ToArray() on an IQueryable before printing it.

  If the tensor does NOT have requires_grad=True, you can skip .detach():
    attention_weights.numpy()   # This also works here
""")

# Convert PyTorch tensor → NumPy array for matplotlib
attention_weights_np = attention_weights.detach().numpy()

plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights_np,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=sentence,
            yticklabels=sentence,
            cbar_kws={'label': 'Attention Weight'})

plt.title('Attention Weights (PyTorch): Who Attends to Whom?', fontsize=14, fontweight='bold')
plt.xlabel('Keys (attending TO these words)', fontsize=12)
plt.ylabel('Queries (attention FROM these words)', fontsize=12)

plt.text(0.5, -0.15,
         'Each row shows how much a word attends to all other words.',
         ha='center', va='top', transform=plt.gca().transAxes,
         fontsize=10, style='italic')

plt.tight_layout()
plt.show()

# ==============================================================================
# PART 7: How 'cat' Attends to All Words
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Detailed Look - How 'cat' Attends to All Words")
print("=" * 70)

word_idx = 1
focus_word = sentence[word_idx]

print(f"\nHow '{focus_word}' attends to each word:\n")
for i, word in enumerate(sentence):
    # .item() converts single-element tensor to Python float
    weight = attention_weights[word_idx, i].item()
    bar = '█' * int(weight * 50)
    print(f"  {word:6s}: {weight:.3f} {bar}")

# ==============================================================================
# PART 8: PyTorch's Secret Weapon - Automatic Gradients
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Why PyTorch? Automatic Gradients!")
print("=" * 70)

print("""
The BIGGEST difference between NumPy and PyTorch:

  NumPy : Just math. Cannot learn/train.
  PyTorch: Math + can automatically compute gradients for training!

HOW IT WORKS:
  Step 1: Create a trainable tensor with requires_grad=True
    W = torch.randn(4, 4, requires_grad=True)

  Step 2: Do computations (PyTorch remembers every step)
    output = Q @ W.T
    loss = output.sum()    # A "loss" is the error we want to minimize

  Step 3: Calculate ALL gradients with ONE line!
    loss.backward()        # ← Magic!
    print(W.grad)          # ← Gradient was computed automatically!

  In C#/.NET terms:
    NumPy  = double[] array     (just stores values)
    PyTorch = Observable<double[]> with full chain-rule math built in
""")

# Demonstrate: automatic gradient computation
W_trainable = torch.randn(d_model, d_model, requires_grad=True)

output_demo = word_embeddings @ W_trainable    # Matrix multiplication
loss = output_demo.sum()                        # Pretend this is our error

loss.backward()    # PyTorch computes ALL gradients automatically!

print(f"Trainable weight W (first 2 rows):\n{W_trainable[:2]}\n")
print(f"Gradient of W computed by PyTorch automatically:")
print(W_trainable.grad[:2])
print("\n  → We did NOT write any gradient math - PyTorch did it for us!")
print("    This is called AutoDiff (Automatic Differentiation).")
print("    NumPy cannot do this.")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - NumPy vs PyTorch")
print("=" * 70)

print("""
SAME RESULT, DIFFERENT TOOLS:

Operation           │ NumPy                      │ PyTorch
────────────────────┼────────────────────────────┼──────────────────────────
Random seed         │ np.random.seed(42)         │ torch.manual_seed(42)
Create array        │ np.random.randn(6, 4)      │ torch.randn(6, 4)
Matrix multiply     │ Q @ K.T                    │ Q @ K.T  (IDENTICAL!)
Softmax             │ write it yourself           │ F.softmax(x, dim=-1)
Sum                 │ x.sum()                    │ x.sum()  (IDENTICAL!)
Shape               │ arr.shape → (6, 4)         │ tensor.shape → [6, 4]
For matplotlib      │ use arr directly           │ tensor.detach().numpy()
Training/gradients  │ NOT POSSIBLE               │ requires_grad=True
GPU acceleration    │ NOT POSSIBLE               │ .to('cuda')

BOTTOM LINE:
  NumPy  → Best for LEARNING the math (simple, clear)
  PyTorch → Best for BUILDING real models (industry standard)

Next:
  example_02: Self-attention using nn.Linear (learned weight matrices)
  example_03: Multi-head attention using nn.MultiheadAttention
  example_04: Positional encoding
  example_05: Full transformer block using nn.Module
  example_06: Mini-GPT using nn.Embedding + nn.Transformer
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 01 - PyTorch Version")
print("=" * 70)

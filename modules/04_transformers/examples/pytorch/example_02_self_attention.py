"""
Example 02: Self-Attention Layer - PyTorch Version

SAME example as NumPy version, now using nn.Linear for learned weight matrices.

KEY NEW CONCEPT: torch.nn.Linear
  In NumPy:   W_q = np.random.randn(d_model, d_model) * 0.1
              Q   = X @ W_q
  In PyTorch: self.W_q = nn.Linear(d_model, d_model, bias=False)
              Q        = self.W_q(X)

  nn.Linear is a reusable learned layer - it handles weight initialization,
  gradient tracking, and parameter management automatically!

  In C#/.NET terms:
    NumPy  = manually creating a float[,] weight matrix
    nn.Linear = using a pre-built NeuralLayer class that manages weights for you

  Also new: torch.nn.Module
    All neural network classes inherit from nn.Module in PyTorch.
    Like inheriting from a base class that provides training support.
    C# equivalent: class SelfAttention : NeuralNetworkBase { ... }
"""

import torch
import torch.nn as nn                  # nn = neural network module (the main toolkit)
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)

print("=" * 70)
print("SELF-ATTENTION WITH LEARNED WEIGHT MATRICES - PyTorch Version")
print("=" * 70)

# ==============================================================================
# PART 1: What Makes Self-Attention 'Self'?
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: What Makes Self-Attention 'Self'?")
print("=" * 70)

print("""
SAME CONCEPT as NumPy version:
  Q, K, V all come from the SAME input, but through LEARNED transformations:

    Q = Input @ W_q
    K = Input @ W_k
    V = Input @ W_v

PYTORCH DIFFERENCE: We use nn.Linear instead of manual weight matrices!

  NumPy way:
    W_q = np.random.randn(d_model, d_model) * 0.1   # Manual weight matrix
    Q   = X @ W_q                                    # Manual multiply

  PyTorch way:
    self.W_q = nn.Linear(d_model, d_model, bias=False)  # Auto-managed layer
    Q        = self.W_q(X)                               # Call like a function!

  WHY nn.Linear is better:
    ✓ Automatically registers as a trainable parameter
    ✓ Proper weight initialization built-in
    ✓ Can save/load easily (.state_dict())
    ✓ Moves to GPU with .to('cuda') automatically
""")

# ==============================================================================
# PART 2: Input Setup
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Input Sentence and Embeddings")
print("=" * 70)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
d_model = 8
seq_len = len(sentence)

print(f"Sentence: {' '.join(sentence)}")
print(f"Embedding dimension: {d_model}")

# NumPy:   X = np.random.randn(seq_len, d_model) * 0.5
# PyTorch: X = torch.randn(seq_len, d_model) * 0.5
X = torch.randn(seq_len, d_model) * 0.5

print(f"\nInput embeddings X shape: {X.shape}")

# ==============================================================================
# PART 3: Manual Implementation (to understand what nn.Linear does)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Manual Weight Matrices (matching NumPy approach)")
print("=" * 70)

print("""
First, let's see the NumPy-style manual approach in PyTorch.
Then we'll see the better nn.Linear approach.
""")

# Manual weight matrices (NumPy style)
W_q_manual = torch.randn(d_model, d_model) * 0.1
W_k_manual = torch.randn(d_model, d_model) * 0.1
W_v_manual = torch.randn(d_model, d_model) * 0.1

# Manual projections (NumPy style: X @ W_q)
Q_manual = X @ W_q_manual
K_manual = X @ W_k_manual
V_manual = X @ W_v_manual

print(f"W_q_manual shape: {W_q_manual.shape}")
print(f"Q_manual shape:   {Q_manual.shape}")
print("\nThis is exactly like NumPy - works but missing training support.")

# ==============================================================================
# PART 4: SelfAttention Class with nn.Module and nn.Linear
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Self-Attention as nn.Module (The PyTorch Way)")
print("=" * 70)

print("""
In PyTorch, all neural network models inherit from nn.Module.

This is like the base class in C#:
  class SelfAttention : nn.Module { }  // Python
  class SelfAttention : NeuralNetworkBase { }  // C# equivalent

KEY nn.Module features:
  - __init__: Initialize layers (like C# constructor)
  - forward(): Define computation (like C# Process() method)
  - parameters(): Get all trainable weights automatically
  - .to('cuda'): Move entire model to GPU
  - .state_dict(): Save/load model weights
""")

class SelfAttention(nn.Module):
    """
    Self-Attention layer using PyTorch's nn.Linear.

    Inherits from nn.Module - gives us training support for free!

    C# equivalent:
        class SelfAttention : NeuralNetworkBase {
            private Linear W_q, W_k, W_v;

            public SelfAttention(int d_model) {
                W_q = new Linear(d_model, d_model);
                W_k = new Linear(d_model, d_model);
                W_v = new Linear(d_model, d_model);
            }

            public (Tensor output, Tensor weights) Forward(Tensor X) { ... }
        }
    """

    def __init__(self, d_model):
        """
        Initialize self-attention.

        Always call super().__init__() first in PyTorch!
        Like calling base() in a C# constructor.
        """
        super().__init__()   # Initialize the nn.Module base class - REQUIRED!

        self.d_model = d_model

        # nn.Linear(in_features, out_features, bias=False)
        # This replaces: W_q = np.random.randn(d_model, d_model) * 0.1
        # nn.Linear automatically:
        #   1. Creates the weight matrix with proper initialization
        #   2. Registers it as a trainable parameter
        #   3. Tracks gradients during backpropagation
        self.W_q = nn.Linear(d_model, d_model, bias=False)   # Query projection
        self.W_k = nn.Linear(d_model, d_model, bias=False)   # Key projection
        self.W_v = nn.Linear(d_model, d_model, bias=False)   # Value projection

    def forward(self, X):
        """
        Forward pass (the computation).

        In PyTorch, you define forward() and call it via the () operator:
          layer = SelfAttention(8)
          output, weights = layer(X)   # ← calls forward() automatically!

        In C#: like overriding Invoke() or implementing ICallable.
        """
        # Project input to Q, K, V using nn.Linear layers
        # nn.Linear layer is called like a function: self.W_q(X)
        # This replaces: Q = X @ W_q  in NumPy/manual approach
        Q = self.W_q(X)    # Shape: (6, 8) - learned query projection
        K = self.W_k(X)    # Shape: (6, 8) - learned key projection
        V = self.W_v(X)    # Shape: (6, 8) - learned value projection

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = Q @ K.T / (self.d_model ** 0.5)

        # Softmax to get attention weights
        # PyTorch: F.softmax(x, dim=-1)  vs  NumPy: softmax(x, axis=-1)
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = attention_weights @ V

        return output, attention_weights

# ==============================================================================
# PART 5: Testing the Class
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Testing SelfAttention")
print("=" * 70)

# Create self-attention layer
attn = SelfAttention(d_model=8)

print(f"\nSelfAttention layer created!")
print(f"Trainable parameters:")
for name, param in attn.named_parameters():
    # named_parameters() lists all trainable weights - no NumPy equivalent!
    print(f"  {name}: shape {param.shape}, requires_grad={param.requires_grad}")

print(f"\nRunning forward pass...")

# Call the layer like a function (this internally calls forward())
# NumPy: output, weights = attn_layer.forward(X)
# PyTorch: output, weights = attn(X)   ← same, but () calls forward automatically
output, attention_weights = attn(X)

print(f"Input shape:  {X.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print("\n✓ Self-attention class works!")

# ==============================================================================
# PART 6: Using torch.no_grad() for Inference
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Inference Mode - torch.no_grad()")
print("=" * 70)

print("""
When you are NOT training (just making predictions), tell PyTorch:
  'Don't track gradients - save memory and speed!'

  with torch.no_grad():
      output, weights = attn(X)

  In C# terms: Like setting readonly or using a read-only view of your data.

WHY do this?
  - Gradient tracking uses extra memory (~2x more)
  - During inference, you don't need gradients
  - torch.no_grad() makes it faster and uses less RAM
""")

with torch.no_grad():    # Context manager - no gradient tracking inside
    output_inf, weights_inf = attn(X)

print(f"Output shape (inference mode): {output_inf.shape}")
print(f"requires_grad: {output_inf.requires_grad}")
print("  → False! No gradient tracking, saving memory.")

# ==============================================================================
# PART 7: Visualizing Attention Patterns
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Visualizing Attention Patterns")
print("=" * 70)

# Convert tensors to NumPy for plotting
# .detach() → stop gradient tracking
# .numpy()  → convert to NumPy array
attention_weights_np = attention_weights.detach().numpy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap
sns.heatmap(attention_weights_np,
            annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=sentence, yticklabels=sentence,
            cbar_kws={'label': 'Attention Weight'},
            ax=axes[0])
axes[0].set_title('Self-Attention Weights (PyTorch)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Attending TO (Keys)')
axes[0].set_ylabel('Attending FROM (Queries)')

# Bar chart for one word
word_idx = 2
word = sentence[word_idx]
axes[1].barh(sentence, attention_weights_np[word_idx], color='coral')
axes[1].set_xlabel('Attention Weight')
axes[1].set_title(f'How "{word}" Attends to Other Words', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

for i, (w, weight) in enumerate(zip(sentence, attention_weights_np[word_idx])):
    axes[1].text(weight, i, f' {weight:.3f}', va='center')

plt.tight_layout()
plt.show()

# ==============================================================================
# PART 8: Comparing Parameter Count
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Counting Model Parameters")
print("=" * 70)

print("""
One of the great features of nn.Module: count parameters easily!
""")

total_params = sum(p.numel() for p in attn.parameters())
trainable_params = sum(p.numel() for p in attn.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"\nBreakdown:")
for name, param in attn.named_parameters():
    print(f"  {name}: {param.numel():,} values ({param.shape})")

print(f"\n  → numel() = number of elements (values) in the tensor")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - Self-Attention in NumPy vs PyTorch")
print("=" * 70)

print("""
CONCEPT COMPARISON:

NumPy approach:                    │ PyTorch approach:
───────────────────────────────────┼─────────────────────────────────────────
W_q = np.random.randn(d,d) * 0.1  │ self.W_q = nn.Linear(d, d, bias=False)
Q = X @ W_q                        │ Q = self.W_q(X)
class SelfAttention:               │ class SelfAttention(nn.Module):
    def __init__(self, d_model):   │     def __init__(self, d_model):
        self.W_q = np.random...    │         super().__init__()   # REQUIRED!
                                   │         self.W_q = nn.Linear(...)
    def forward(self, X):          │     def forward(self, X):
        ...                        │         ...
layer.forward(X)                   │ layer(X)  ← calls forward() automatically

KEY nn.Module BENEFITS:
  ✓ parameters()     - list all trainable weights
  ✓ .to('cuda')      - move entire model to GPU in one line
  ✓ .state_dict()    - save/load model weights
  ✓ .train()/.eval() - switch training/inference modes
  ✓ No manual gradient code needed

Next:
  example_03: Multi-head attention using nn.MultiheadAttention (built-in!)
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 02 - PyTorch Version")
print("=" * 70)

"""
Exercise 02: Building Self-Attention Layer

🎯 GOAL: Build a complete self-attention layer with learned weight matrices!

In this exercise, you'll create a self-attention layer from scratch, including:
  - Learned weight matrices W_q, W_k, W_v
  - Linear projections to create Q, K, V
  - Complete SelfAttention class
  - Testing and visualization

This builds on Exercise 01 by adding LEARNED transformations!

Tasks:
  1. Initialize weight matrices W_q, W_k, W_v
  2. Project input to Q, K, V
  3. Implement self-attention forward pass
  4. Create complete SelfAttention class
  5. BONUS: Add multi-head capability

Hints for .NET developers:
  - Weight matrices are like trainable Matrix objects
  - @ operator for matrix multiplication
  - Class structure is similar to C# classes
  - __init__ is like a constructor

Good luck! 🚀
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("=" * 70)
print("EXERCISE 02: Build Self-Attention Layer")
print("=" * 70)

# ==============================================================================
# GIVEN: Setup and Helper Functions
# ==============================================================================

print("\n" + "=" * 70)
print("SETUP: Given Information")
print("=" * 70)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
d_model = 8
seq_len = len(sentence)

print(f"Sentence: {' '.join(sentence)}")
print(f"Embedding dimension: {d_model}")

# Create input embeddings (given)
X = np.random.randn(seq_len, d_model) * 0.5

print(f"\nInput embeddings X shape: {X.shape}")
print("First word embedding:")
print(X[0])

# Softmax function (given from Exercise 01)
def softmax(x, axis=-1):
    """Apply softmax function."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# ==============================================================================
# TODO 1: Initialize Weight Matrices
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 1: Initialize Weight Matrices W_q, W_k, W_v")
print("=" * 70)

print("""
TASK: Initialize the three weight matrices for self-attention.

Each matrix should:
  - Have shape (d_model, d_model)
  - Be initialized with small random values
  - Use np.random.randn() and multiply by 0.1

These matrices are LEARNED during training (we're just initializing them).

Why 0.1?
  - Small initial weights prevent large activations
  - Standard practice in deep learning
  - Helps training stability

Hint for C# developers:
  - Like: Matrix W_q = new Matrix(d_model, d_model);
  - Initialize with: Random.NextGaussian() * 0.1
""")

# TODO: Your code here
# Hint: W_q = np.random.randn(d_model, d_model) * 0.1
# Hint: W_k = np.random.randn(d_model, d_model) * 0.1
# Hint: W_v = np.random.randn(d_model, d_model) * 0.1

# SOLUTION (uncomment to check):
# W_q = np.random.randn(d_model, d_model) * 0.1
# W_k = np.random.randn(d_model, d_model) * 0.1
# W_v = np.random.randn(d_model, d_model) * 0.1

# Uncomment to verify:
# print(f"\nW_q shape: {W_q.shape}")
# print(f"W_k shape: {W_k.shape}")
# print(f"W_v shape: {W_v.shape}")
# print("\n✓ Weight matrices initialized!")

# ==============================================================================
# TODO 2: Project Input to Q, K, V
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 2: Linear Projections - Create Q, K, V")
print("=" * 70)

print("""
TASK: Transform input embeddings into Query, Key, and Value representations.

Formula:
  Q = X @ W_q
  K = X @ W_k
  V = X @ W_v

Where:
  - X is the input embeddings (seq_len, d_model)
  - @ is matrix multiplication
  - Result: Q, K, V all have shape (seq_len, d_model)

This is why it's called "self-attention":
  - Q, K, V all come from the SAME input X
  - But they're transformed differently by different weight matrices

Hint for C# developers:
  - Like: var Q = Matrix.Multiply(X, W_q);
  - Or with LINQ: X.Select(row => row * W_q)
""")

# TODO: Your code here
# Hint: Q = X @ W_q
# Hint: K = X @ W_k
# Hint: V = X @ W_v

# SOLUTION (uncomment to check):
# Q = X @ W_q
# K = X @ W_k
# V = X @ W_v

# Uncomment to verify:
# print(f"\nQ (Queries) shape: {Q.shape}")
# print(f"K (Keys) shape: {K.shape}")
# print(f"V (Values) shape: {V.shape}")
# print("\n✓ Q, K, V created successfully!")

# ==============================================================================
# TODO 3: Implement Self-Attention Forward Pass
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 3: Complete Self-Attention Forward Pass")
print("=" * 70)

print("""
TASK: Implement the complete self-attention computation.

Steps:
  1. Compute attention scores: Q @ K.T / sqrt(d_k)
  2. Apply softmax to get weights
  3. Compute output: weights @ V
  4. Return both output and weights

Formula:
  scores = (Q @ K^T) / sqrt(d_model)
  weights = softmax(scores)
  output = weights @ V

Hint for C# developers:
  - Similar to pipeline pattern
  - Each step transforms the data
  - Final output is context-aware embeddings
""")

# TODO: Your code here
# Step 1: Compute scores
# Hint: d_k = d_model
# Hint: attention_scores = Q @ K.T / np.sqrt(d_k)

# Step 2: Apply softmax
# Hint: attention_weights = softmax(attention_scores, axis=-1)

# Step 3: Compute output
# Hint: attention_output = attention_weights @ V

# SOLUTION (uncomment to check):
# d_k = d_model
# attention_scores = Q @ K.T / np.sqrt(d_k)
# attention_weights = softmax(attention_scores, axis=-1)
# attention_output = attention_weights @ V

# Uncomment to verify:
# print(f"\nAttention output shape: {attention_output.shape}")
# print("✓ Self-attention forward pass complete!")

# ==============================================================================
# TODO 4: Create SelfAttention Class
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 4: Build Complete SelfAttention Class")
print("=" * 70)

print("""
TASK: Create a reusable SelfAttention class.

Class structure:
  class SelfAttention:
      def __init__(self, d_model):
          # Initialize W_q, W_k, W_v

      def forward(self, X):
          # 1. Project to Q, K, V
          # 2. Compute attention scores
          # 3. Apply softmax
          # 4. Compute output
          # Return output and weights

This is like C#:
  class SelfAttention {
      private Matrix W_q, W_k, W_v;

      public SelfAttention(int d_model) {
          // Initialize weights
      }

      public (Matrix output, Matrix weights) Forward(Matrix X) {
          // Compute attention
      }
  }
""")

# TODO: Your code here
# Hint: Follow the structure shown above

# SOLUTION (uncomment to check):
# class SelfAttention:
#     """Self-attention layer with learned weight matrices."""
#
#     def __init__(self, d_model):
#         """Initialize weight matrices."""
#         self.d_model = d_model
#         self.W_q = np.random.randn(d_model, d_model) * 0.1
#         self.W_k = np.random.randn(d_model, d_model) * 0.1
#         self.W_v = np.random.randn(d_model, d_model) * 0.1
#
#     def forward(self, X):
#         """
#         Forward pass of self-attention.
#
#         Args:
#             X: Input embeddings, shape (seq_len, d_model)
#
#         Returns:
#             output: Context-aware embeddings, shape (seq_len, d_model)
#             attention_weights: Attention matrix, shape (seq_len, seq_len)
#         """
#         # Project to Q, K, V
#         Q = X @ self.W_q
#         K = X @ self.W_k
#         V = X @ self.W_v
#
#         # Compute attention scores
#         d_k = self.d_model
#         scores = Q @ K.T / np.sqrt(d_k)
#
#         # Apply softmax
#         weights = softmax(scores, axis=-1)
#
#         # Compute output
#         output = weights @ V
#
#         return output, weights

# ==============================================================================
# TEST YOUR CLASS
# ==============================================================================

print("\n" + "=" * 70)
print("TEST: Using Your SelfAttention Class")
print("=" * 70)

print("""
TASK: Test your SelfAttention class.

Steps:
  1. Create an instance: attn = SelfAttention(d_model=8)
  2. Run forward pass: output, weights = attn.forward(X)
  3. Verify shapes and print results
""")

# TODO: Your code here (after implementing the class above)
# Hint: attn = SelfAttention(d_model=8)
# Hint: output, weights = attn.forward(X)

# SOLUTION (uncomment to test):
# print("\nCreating SelfAttention layer...")
# attn = SelfAttention(d_model=8)
#
# print("Running forward pass...")
# output, weights = attn.forward(X)
#
# print(f"\n✅ Results:")
# print(f"  Input shape: {X.shape}")
# print(f"  Output shape: {output.shape}")
# print(f"  Attention weights shape: {weights.shape}")
# print("\n✓ SelfAttention class works!")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print("\n" + "=" * 70)
print("VISUALIZATION: Attention Patterns")
print("=" * 70)

# Uncomment after completing the class:
# plt.figure(figsize=(10, 8))
# sns.heatmap(weights,
#             annot=True,
#             fmt='.3f',
#             cmap='YlOrRd',
#             xticklabels=sentence,
#             yticklabels=sentence,
#             cbar_kws={'label': 'Attention Weight'})
# plt.title('Self-Attention Weights', fontsize=14, fontweight='bold')
# plt.xlabel('Attending TO (Keys)')
# plt.ylabel('Attending FROM (Queries)')
# plt.tight_layout()
# plt.show()

# ==============================================================================
# BONUS CHALLENGE: Multi-Head Capability
# ==============================================================================

print("\n" + "=" * 70)
print("BONUS CHALLENGE: Add Multi-Head Capability")
print("=" * 70)

print("""
BONUS: Extend your SelfAttention class to support multiple heads!

Modifications:
  1. Add num_heads parameter to __init__
  2. Compute d_k = d_model // num_heads
  3. Split Q, K, V into heads
  4. Compute attention for each head
  5. Concatenate head outputs

class MultiHeadSelfAttention:
    def __init__(self, d_model, num_heads):
        # Initialize weights
        # Compute d_k per head

    def split_heads(self, x):
        # Reshape to (seq_len, num_heads, d_k)

    def forward(self, X):
        # Project, split, attend, concatenate

This is ADVANCED - only try if you feel comfortable!
""")

# BONUS SOLUTION (uncomment to check):
# class MultiHeadSelfAttention:
#     """Multi-head self-attention layer."""
#
#     def __init__(self, d_model, num_heads):
#         assert d_model % num_heads == 0
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#
#         self.W_q = np.random.randn(d_model, d_model) * 0.1
#         self.W_k = np.random.randn(d_model, d_model) * 0.1
#         self.W_v = np.random.randn(d_model, d_model) * 0.1
#         self.W_o = np.random.randn(d_model, d_model) * 0.1
#
#     def split_heads(self, x):
#         seq_len, d_model = x.shape
#         return x.reshape(seq_len, self.num_heads, self.d_k)
#
#     def forward(self, X):
#         Q = X @ self.W_q
#         K = X @ self.W_k
#         V = X @ self.W_v
#
#         Q_heads = self.split_heads(Q)
#         K_heads = self.split_heads(K)
#         V_heads = self.split_heads(V)
#
#         head_outputs = []
#         for h in range(self.num_heads):
#             Q_h = Q_heads[:, h, :]
#             K_h = K_heads[:, h, :]
#             V_h = V_heads[:, h, :]
#
#             scores = Q_h @ K_h.T / np.sqrt(self.d_k)
#             weights = softmax(scores, axis=-1)
#             output = weights @ V_h
#             head_outputs.append(output)
#
#         concat_output = np.concatenate(head_outputs, axis=-1)
#         final_output = concat_output @ self.W_o
#
#         return final_output

# ==============================================================================
# VERIFICATION
# ==============================================================================

print("\n" + "=" * 70)
print("VERIFICATION CHECKLIST")
print("=" * 70)

print("""
Check that you've completed:

□ TODO 1: Initialized W_q, W_k, W_v matrices
□ TODO 2: Projected X to Q, K, V
□ TODO 3: Implemented forward pass (scores, softmax, output)
□ TODO 4: Created SelfAttention class
□ TODO 5: Tested the class
□ Visualized attention weights

Once all checked, you've mastered self-attention! 🎉

BONUS:
□ Implemented multi-head self-attention (advanced!)
""")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("EXERCISE SUMMARY")
print("=" * 70)

print("""
What You Built:
  ✓ Learned weight matrices W_q, W_k, W_v
  ✓ Linear projections (X @ W)
  ✓ Complete self-attention forward pass
  ✓ Reusable SelfAttention class
  ✓ BONUS: Multi-head capability

Key Concepts:
  - Self-attention: Q, K, V all from same input
  - Learned transformations: W matrices trained
  - Context-aware: Output incorporates all positions
  - Object-oriented: Encapsulated in a class

Difference from Exercise 01:
  - Exercise 01: Q = K = V = input (no learning)
  - Exercise 02: Q, K, V = learned projections (trainable!)

This IS what powers transformers!

In C#/.NET Terms:
  - Class structure similar to C# classes
  - Weight matrices like trainable Matrix objects
  - Forward pass like a pipeline method
  - @ operator like MathNet matrix multiplication

Real-World Usage:
  - BERT uses self-attention in every layer
  - GPT uses masked self-attention
  - All modern transformers use this!

Next Exercise:
  - exercise_03: Complete transformer block with all components

Great work! You're building production-level ML components! 💪
""")

print("\n" + "=" * 70)
print("END OF EXERCISE 02")
print("=" * 70)

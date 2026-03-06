"""
Exercise 01: Implementing Basic Attention Mechanism

🎯 GOAL: Implement the core attention mechanism yourself!

In this exercise, you'll fill in the missing pieces to create a working
attention mechanism. This reinforces the concepts from Example 01.

Tasks:
  1. Compute attention scores (Q @ K^T / sqrt(d_k))
  2. Implement softmax function
  3. Apply softmax to scores
  4. Compute output (weighted sum of values)
  5. Visualize attention weights

Hints for .NET developers:
  - @ operator is matrix multiplication (like MathNet.Numerics)
  - np.exp() is Math.Exp()
  - np.sum() is array.Sum()
  - axis parameter is like LINQ GroupBy direction

Good luck! 🚀
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("=" * 70)
print("EXERCISE 01: Implement Basic Attention Mechanism")
print("=" * 70)

# ==============================================================================
# GIVEN: Setup and Data
# ==============================================================================

print("\n" + "=" * 70)
print("SETUP: Given Information")
print("=" * 70)

sentence = ["The", "cat", "sat", "on", "the", "mat"]
d_model = 4
seq_len = len(sentence)

print(f"Sentence: {' '.join(sentence)}")
print(f"Embedding dimension: {d_model}")

# Create embeddings (given)
embeddings = np.random.randn(seq_len, d_model)

# For this exercise, Q, K, V are just copies of embeddings
Q = embeddings
K = embeddings
V = embeddings

print(f"\nQ shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# ==============================================================================
# TODO 1: Compute Attention Scores
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 1: Compute Attention Scores")
print("=" * 70)

print("""
TASK: Compute attention scores using the formula:
  scores = (Q @ K^T) / sqrt(d_k)

Where:
  - Q @ K^T is matrix multiplication (dot product of all query-key pairs)
  - sqrt(d_k) is the scaling factor
  - d_k = d_model (dimension of keys)

Steps:
  1. Compute d_k (dimension of keys) = d_model
  2. Compute raw scores: Q @ K.T  (use @ operator, .T for transpose)
  3. Compute scaling factor: np.sqrt(d_k)
  4. Divide scores by scaling factor

Hint for C# developers:
  - @ is like Matrix.Multiply(Q, K.Transpose())
  - .T is like matrix.Transpose()
  - np.sqrt() is like Math.Sqrt()
""")

# TODO: Your code here
# Hint: d_k = ?
# Hint: attention_scores = Q @ K.T / np.sqrt(d_k)

# SOLUTION (uncomment to check):
# d_k = d_model
# attention_scores = Q @ K.T / np.sqrt(d_k)

# Uncomment to verify:
# print(f"\nAttention scores shape: {attention_scores.shape}")
# print("Attention scores:")
# print(attention_scores)

# ==============================================================================
# TODO 2: Implement Softmax Function
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 2: Implement Softmax Function")
print("=" * 70)

print("""
TASK: Implement the softmax function to convert scores to probabilities.

Formula:
  softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

For numerical stability, use this trick:
  softmax(x) = softmax(x - max(x))  (same result, prevents overflow)

Steps:
  1. Subtract max value for numerical stability
  2. Compute exponentials: np.exp()
  3. Sum exponentials along axis
  4. Divide to get probabilities

Hint for C# developers:
  - np.exp(x) is like x.Select(v => Math.Exp(v))
  - np.sum(x, axis=-1, keepdims=True) is like aggregating along rows
  - keepdims=True preserves dimensions for broadcasting
""")

def softmax(x, axis=-1):
    """
    Apply softmax function.

    Args:
        x: Input array (attention scores)
        axis: Axis to apply softmax along (default: last axis)

    Returns:
        Probabilities that sum to 1 along specified axis
    """
    # TODO: Your code here

    # Step 1: Subtract max for numerical stability
    # Hint: x_shifted = x - np.max(x, axis=axis, keepdims=True)

    # Step 2: Compute exponentials
    # Hint: exp_x = np.exp(x_shifted)

    # Step 3: Sum exponentials
    # Hint: sum_exp = np.sum(exp_x, axis=axis, keepdims=True)

    # Step 4: Divide to get probabilities
    # Hint: return exp_x / sum_exp

    pass  # Remove this when you add your code

# SOLUTION (uncomment to check):
# def softmax(x, axis=-1):
#     exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
#     return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# ==============================================================================
# TODO 3: Apply Softmax to Get Attention Weights
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 3: Apply Softmax to Scores")
print("=" * 70)

print("""
TASK: Apply your softmax function to attention scores.

This converts scores into probabilities (weights) that sum to 1.

Steps:
  1. Call softmax(attention_scores, axis=-1)
  2. Store result in attention_weights

Verify:
  - Each row should sum to 1.0
  - All values should be between 0 and 1
""")

# TODO: Your code here
# Hint: attention_weights = softmax(attention_scores, axis=-1)

# SOLUTION (uncomment to check):
# attention_weights = softmax(attention_scores, axis=-1)

# Uncomment to verify:
# print("\nAttention weights shape:", attention_weights.shape)
# print("First row sum:", attention_weights[0].sum())
# print("\nAttention weights:")
# print(attention_weights)

# ==============================================================================
# TODO 4: Compute Attention Output
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 4: Compute Attention Output")
print("=" * 70)

print("""
TASK: Compute the final output as weighted sum of values.

Formula:
  output = attention_weights @ V

This gives each word a new representation that incorporates information
from all words it attended to!

Steps:
  1. Multiply attention_weights by V using @ operator
  2. Store result in attention_output

Hint for C# developers:
  - This is like Matrix.Multiply(attention_weights, V)
  - Result shape: (seq_len, d_model)
""")

# TODO: Your code here
# Hint: attention_output = attention_weights @ V

# SOLUTION (uncomment to check):
# attention_output = attention_weights @ V

# Uncomment to verify:
# print(f"\nAttention output shape: {attention_output.shape}")
# print("\nAttention output:")
# print(attention_output)

# ==============================================================================
# TODO 5: Visualize Attention Weights
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 5: Visualize Attention Weights")
print("=" * 70)

print("""
TASK: Create a heatmap visualization of attention weights.

Use seaborn to create a heatmap showing which words attend to which.

Steps:
  1. Use sns.heatmap()
  2. Set annot=True to show values
  3. Use sentence as x/y tick labels
  4. Add title and axis labels
""")

# TODO: Your code here
# Hint: Use sns.heatmap(attention_weights, annot=True, ...)

# SOLUTION (uncomment to check):
# plt.figure(figsize=(10, 8))
# sns.heatmap(attention_weights,
#             annot=True,
#             fmt='.3f',
#             cmap='YlOrRd',
#             xticklabels=sentence,
#             yticklabels=sentence,
#             cbar_kws={'label': 'Attention Weight'})
# plt.title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
# plt.xlabel('Attending TO (Keys)')
# plt.ylabel('Attending FROM (Queries)')
# plt.tight_layout()
# plt.show()

# ==============================================================================
# VERIFICATION AND TESTING
# ==============================================================================

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

print("""
Once you've completed all TODOs, uncomment the verification code below
to check your implementation!
""")

# VERIFICATION CODE (uncomment after completing TODOs):
# print("\n✅ Verification Tests:")
# print(f"1. Attention scores computed: {attention_scores.shape == (seq_len, seq_len)}")
# print(f"2. Attention weights computed: {attention_weights.shape == (seq_len, seq_len)}")
# print(f"3. Weights sum to 1: {np.allclose(attention_weights.sum(axis=1), 1.0)}")
# print(f"4. Output computed: {attention_output.shape == (seq_len, d_model)}")
# print("\nIf all True, congratulations! You've implemented attention! 🎉")

# ==============================================================================
# BONUS CHALLENGE
# ==============================================================================

print("\n" + "=" * 70)
print("BONUS CHALLENGE")
print("=" * 70)

print("""
BONUS: Implement attention as a reusable function!

Create a function: attention(Q, K, V) that:
  1. Computes scores
  2. Applies softmax
  3. Returns output and weights

def attention(Q, K, V):
    # Your implementation here
    return output, weights

Test it with:
  output, weights = attention(Q, K, V)
""")

# TODO (BONUS): Implement attention function
# def attention(Q, K, V):
#     d_k = Q.shape[-1]
#     scores = Q @ K.T / np.sqrt(d_k)
#     weights = softmax(scores, axis=-1)
#     output = weights @ V
#     return output, weights

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("EXERCISE SUMMARY")
print("=" * 70)

print("""
What You Practiced:
  ✓ Computing attention scores (Q @ K^T / sqrt(d_k))
  ✓ Implementing softmax function
  ✓ Converting scores to probabilities
  ✓ Computing weighted sum of values
  ✓ Visualizing attention patterns

Key Formula:
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

This is the FOUNDATION of transformers!

Next Exercise:
  - exercise_02: Self-attention with learned weight matrices
  - exercise_03: Complete transformer block

Keep going! You're building real ML skills! 💪
""")

print("\n" + "=" * 70)
print("END OF EXERCISE 01")
print("=" * 70)

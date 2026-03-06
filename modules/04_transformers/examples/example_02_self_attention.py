"""
Example 02: Self-Attention Layer

This example demonstrates SELF-attention, where Query, Key, and Value all come
from the SAME input. Unlike basic attention, self-attention uses learned weight
matrices to project the input into Q, K, V spaces.

What you'll see:
1. Learned weight matrices W_q, W_k, W_v
2. Projecting input embeddings to Q, K, V representations
3. How self-attention creates context-aware representations
4. Visualization of attention patterns
5. Comparison with basic attention

Think of it like a team meeting:
- Everyone (words) can ask questions (Query)
- Everyone can provide their expertise (Key)
- Everyone has information to share (Value)
- The learned matrices determine HOW each person interprets and shares info
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility (like C# Random with fixed seed)
np.random.seed(42)

print("=" * 70)
print("SELF-ATTENTION WITH LEARNED WEIGHT MATRICES")
print("=" * 70)

# ==============================================================================
# PART 1: Understanding Self-Attention vs Basic Attention
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: What Makes Self-Attention 'Self'?")
print("=" * 70)

print("""
BASIC ATTENTION (Example 01):
  Q, K, V could come from different sources
  Example: Q from question, K/V from paragraph

SELF-ATTENTION (This example):
  Q, K, V ALL come from the SAME input
  BUT they're created using LEARNED transformations

  Q = Input @ W_q  (learned query projection)
  K = Input @ W_k  (learned key projection)
  V = Input @ W_v  (learned value projection)

This is like C# where you have:
  class SelfAttention {
      Matrix W_q, W_k, W_v;  // Learned parameters (like weights in neural net)

      (Q, K, V) Transform(Matrix input) {
          return (input * W_q, input * W_k, input * W_v);
      }
  }
""")

# ==============================================================================
# PART 2: Setting Up Our Input
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Input Sentence and Embeddings")
print("=" * 70)

# Our example sentence
sentence = ["The", "cat", "sat", "on", "the", "mat"]
print(f"Input sentence: {' '.join(sentence)}")

# Embedding dimension
d_model = 8  # Larger than example_01 to show more complex patterns
seq_len = len(sentence)

print(f"\nSequence length: {seq_len} words")
print(f"Embedding dimension: {d_model}")

# Create word embeddings (in real transformers, these come from an embedding layer)
# Shape: (seq_len, d_model) = (6, 8)
# Similar to C#: float[6][] where each inner array has 8 elements
X = np.random.randn(seq_len, d_model) * 0.5

print(f"\nInput embeddings X shape: {X.shape}")
print("\nFirst word embedding ('{}'): ".format(sentence[0]))
print(X[0])

# ==============================================================================
# PART 3: Creating Learned Weight Matrices
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Learned Weight Matrices W_q, W_k, W_v")
print("=" * 70)

print("""
These are the LEARNABLE PARAMETERS of self-attention!
During training, these matrices learn to:
  - W_q: How to ask good questions
  - W_k: How to advertise what information you have
  - W_v: How to represent your information

In deep learning frameworks (PyTorch/TensorFlow), these are trained
using backpropagation. For this example, we'll initialize them randomly.

Similar to C#:
  // These would be trainable parameters in a neural network
  Matrix W_q = new Matrix(d_model, d_model);  // Query projection
  Matrix W_k = new Matrix(d_model, d_model);  // Key projection
  Matrix W_v = new Matrix(d_model, d_model);  // Value projection
""")

# Initialize weight matrices
# Shape: (d_model, d_model) = (8, 8) for each
# Multiply by 0.1 for small initial values (standard practice)
W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1

print(f"\nW_q shape: {W_q.shape} (projects input to query space)")
print(f"W_k shape: {W_k.shape} (projects input to key space)")
print(f"W_v shape: {W_v.shape} (projects input to value space)")

print("\nExample - W_q (first 3x3 portion):")
print(W_q[:3, :3])

# ==============================================================================
# PART 4: Projecting Input to Q, K, V
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Creating Q, K, V Through Linear Projections")
print("=" * 70)

print("""
Now we transform our input embeddings into Q, K, V using matrix multiplication:

  Q = X @ W_q   (X multiplied by W_q)
  K = X @ W_k
  V = X @ W_v

This is matrix multiplication (@ operator in Python, * in LINQ/MathNet):
  Result[i,j] = sum(X[i,k] * W_q[k,j] for all k)
""")

# Project input to Q, K, V spaces
# @ is matrix multiplication operator in NumPy
Q = X @ W_q  # Shape: (6, 8) @ (8, 8) = (6, 8)
K = X @ W_k  # Shape: (6, 8) @ (8, 8) = (6, 8)
V = X @ W_v  # Shape: (6, 8) @ (8, 8) = (6, 8)

print(f"\nQ (Queries) shape: {Q.shape}")
print(f"K (Keys) shape: {K.shape}")
print(f"V (Values) shape: {V.shape}")

print(f"\nQuery for word '{sentence[0]}':")
print(Q[0])

print(f"\nKey for word '{sentence[0]}':")
print(K[0])

print("\nNote: Even though all come from same input X,")
print("Q, K, V are DIFFERENT because W_q, W_k, W_v are different!")

# ==============================================================================
# PART 5: Computing Attention Scores
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Computing Attention Scores")
print("=" * 70)

print("""
Now we compute how much each word should attend to every other word.
This is the same as Example 01, but now Q and K are learned projections!

Formula: scores = (Q @ K^T) / sqrt(d_k)

where d_k is the dimension of the key vectors.
""")

# Compute attention scores: Q @ K^T
# Shape: (6, 8) @ (8, 6) = (6, 6)
d_k = d_model  # dimension of keys
attention_scores = Q @ K.T

print(f"\nAttention scores shape: {attention_scores.shape}")
print("This is a 6x6 matrix: each word attending to all words\n")

# Scale by sqrt(d_k) - prevents very large values
scaling_factor = np.sqrt(d_k)
scaled_scores = attention_scores / scaling_factor

print(f"Scaling factor: sqrt({d_k}) = {scaling_factor:.3f}")
print("\nScaled attention scores:")
print(scaled_scores)

# ==============================================================================
# PART 6: Applying Softmax
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Converting Scores to Attention Weights")
print("=" * 70)

def softmax(x, axis=-1):
    """
    Apply softmax to convert scores to probabilities.

    Args:
        x: Input scores
        axis: Axis to apply softmax along (default: last axis)

    Returns:
        Probability distribution (sums to 1 along axis)

    Similar to C#:
        double[] Softmax(double[] scores) {
            var exp = scores.Select(s => Math.Exp(s)).ToArray();
            var sum = exp.Sum();
            return exp.Select(e => e / sum).ToArray();
        }
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Apply softmax to get attention weights
attention_weights = softmax(scaled_scores, axis=-1)

print("Attention weights (each row sums to 1.0):")
print(attention_weights)

print("\nVerify row sums:")
for i, word in enumerate(sentence):
    print(f"  '{word}': {attention_weights[i].sum():.6f}")

# ==============================================================================
# PART 7: Computing Self-Attention Output
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Computing Context-Aware Representations")
print("=" * 70)

print("""
Finally, we compute the output as a weighted sum of values:

  Output = attention_weights @ V

Each word's output is a combination of ALL words' values,
weighted by the attention scores!
""")

# Compute self-attention output
# Shape: (6, 6) @ (6, 8) = (6, 8)
output = attention_weights @ V

print(f"\nOutput shape: {output.shape}")
print("Same shape as input, but now CONTEXT-AWARE!\n")

# Compare input vs output for one word
word_idx = 1  # "cat"
print(f"Word: '{sentence[word_idx]}'")
print(f"\nOriginal embedding (input):")
print(X[word_idx])
print(f"\nContext-aware embedding (output):")
print(output[word_idx])

print("""
The output embedding now contains information from all words
that 'cat' attended to, making it context-aware!
""")

# ==============================================================================
# PART 8: Building a Self-Attention Class
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Self-Attention as a Reusable Class")
print("=" * 70)

class SelfAttention:
    """
    Self-Attention layer with learned weight matrices.

    This is similar to a C# class:
        class SelfAttention {
            private Matrix W_q, W_k, W_v;

            public SelfAttention(int d_model) {
                W_q = InitializeMatrix(d_model, d_model);
                W_k = InitializeMatrix(d_model, d_model);
                W_v = InitializeMatrix(d_model, d_model);
            }

            public (Matrix output, Matrix weights) Forward(Matrix X) {
                // ... attention computation ...
            }
        }
    """

    def __init__(self, d_model):
        """
        Initialize self-attention layer.

        Args:
            d_model: Dimension of input embeddings
        """
        self.d_model = d_model

        # Initialize learned weight matrices
        # In practice, these would be trained with backpropagation
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X):
        """
        Forward pass of self-attention.

        Args:
            X: Input embeddings, shape (seq_len, d_model)

        Returns:
            output: Context-aware embeddings, shape (seq_len, d_model)
            attention_weights: Attention matrix, shape (seq_len, seq_len)
        """
        # 1. Project to Q, K, V
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # 2. Compute attention scores
        scores = Q @ K.T / np.sqrt(self.d_model)

        # 3. Apply softmax to get weights
        attention_weights = softmax(scores, axis=-1)

        # 4. Weighted sum of values
        output = attention_weights @ V

        return output, attention_weights

# Test our class
print("\nCreating SelfAttention layer...")
attn_layer = SelfAttention(d_model=8)

print("Running forward pass...")
output_class, weights_class = attn_layer.forward(X)

print(f"Output shape: {output_class.shape}")
print(f"Attention weights shape: {weights_class.shape}")
print("\nClass-based implementation works! ✓")

# ==============================================================================
# PART 9: Visualizing Attention Patterns
# ==============================================================================

print("\n" + "=" * 70)
print("PART 9: Visualizing Attention Patterns")
print("=" * 70)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Full attention heatmap
sns.heatmap(attention_weights,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=sentence,
            yticklabels=sentence,
            cbar_kws={'label': 'Attention Weight'},
            ax=axes[0])

axes[0].set_title('Self-Attention Weights Heatmap', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Attending TO (Keys)', fontsize=12)
axes[0].set_ylabel('Attending FROM (Queries)', fontsize=12)

# Right plot: Attention pattern for one word
word_idx = 2  # "sat"
word = sentence[word_idx]

axes[1].barh(sentence, attention_weights[word_idx], color='coral')
axes[1].set_xlabel('Attention Weight', fontsize=12)
axes[1].set_title(f'How "{word}" Attends to Other Words', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (w, weight) in enumerate(zip(sentence, attention_weights[word_idx])):
    axes[1].text(weight, i, f' {weight:.3f}', va='center')

plt.tight_layout()
plt.show()

print(f"\nInterpretation for '{word}':")
for i, (w, weight) in enumerate(zip(sentence, attention_weights[word_idx])):
    bar = '█' * int(weight * 40)
    print(f"  {w:6s}: {weight:.3f} {bar}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - What We Learned")
print("=" * 70)

print("""
✓ Self-Attention: Q, K, V all from the SAME input
✓ Learned Weight Matrices: W_q, W_k, W_v are trainable parameters
✓ Linear Projections: Q = X @ W_q, K = X @ W_k, V = X @ W_v
✓ Context-Aware: Output incorporates information from all words
✓ Same Formula: Attention(Q,K,V) = softmax(Q @ K^T / √d_k) @ V

Key Difference from Example 01:
  - Example 01: Q, K, V were just copies of input
  - Example 02: Q, K, V are LEARNED projections of input

This learning allows the model to:
  - Learn which relationships are important
  - Capture different types of patterns
  - Adapt to the training data

In C#/.NET Terms:
  - W_q, W_k, W_v are like trainable Matrix objects
  - forward() is like a method that processes input
  - The @ operator is matrix multiplication (MathNet.Numerics)

Real-World Impact:
  - BERT, GPT, and all transformers use self-attention
  - The "learned" part is what makes them powerful
  - Multiple layers stack these to learn complex patterns

Next Steps:
  - example_03: Multi-head attention (parallel attention mechanisms)
  - example_04: Positional encoding (adding position information)
  - example_05: Complete transformer block
  - example_06: Building a mini-GPT!
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 02")
print("=" * 70)

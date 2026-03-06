"""
Exercise 02: Building Self-Attention with Learned Weights

In this exercise, you'll build a complete self-attention layer with learned
weight matrices (W_q, W_k, W_v). This is what real transformers use!

Tasks:
1. Initialize weight matrices
2. Project input to Q, K, V using matrix multiplication
3. Implement complete self-attention forward pass
4. Add multi-head capability
5. Test on real sentences and visualize results

This builds on Exercise 01 - make sure you completed that first!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("=" * 70)
print("EXERCISE 02: BUILDING SELF-ATTENTION")
print("=" * 70)

# ==============================================================================
# HELPER: Softmax (from Exercise 01)
# ==============================================================================

def softmax(x):
    """Softmax function (you should have implemented this in Exercise 01!)"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ==============================================================================
# SETUP: Creating Input Embeddings
# ==============================================================================

print("\n" + "=" * 70)
print("SETUP: Creating Input Embeddings")
print("=" * 70)

# Sample sentence
sentence = ["The", "quick", "brown", "fox", "jumps"]
seq_len = len(sentence)
d_model = 8  # Embedding dimension

print(f"Sentence: {' '.join(sentence)}")
print(f"Sequence length: {seq_len}")
print(f"Embedding dimension: {d_model}")

# Create input embeddings
X = np.random.randn(seq_len, d_model) * 0.5

print(f"\nInput embeddings X shape: {X.shape}")
print("These represent the initial word embeddings (before attention)\n")

# ==============================================================================
# TODO 1: Initialize Weight Matrices
# ==============================================================================

print("=" * 70)
print("TODO 1: Initialize Weight Matrices W_q, W_k, W_v")
print("=" * 70)

print("""
Task: Create three weight matrices for transforming input to Q, K, V.

Requirements:
- Each matrix should have shape (d_model, d_model)
- Initialize with small random values (use np.random.randn * 0.1)
- Create: W_q, W_k, W_v

Hints:
- np.random.randn(rows, cols) creates random matrix
- Multiply by 0.1 for small initialization
- All three matrices have the same shape!

C# Analogy: Like creating transformation matrices:
    Matrix W_q = Matrix.Random(d_model, d_model) * 0.1
""")

# TODO: Initialize weight matrices
# YOUR CODE HERE:
W_q = None  # Replace with your implementation
W_k = None  # Replace with your implementation
W_v = None  # Replace with your implementation

# SOLUTION (uncomment to see/verify):
# W_q = np.random.randn(d_model, d_model) * 0.1
# W_k = np.random.randn(d_model, d_model) * 0.1
# W_v = np.random.randn(d_model, d_model) * 0.1

# Verification
if W_q is not None and W_k is not None and W_v is not None:
    print(f"✓ W_q shape: {W_q.shape}")
    print(f"✓ W_k shape: {W_k.shape}")
    print(f"✓ W_v shape: {W_v.shape}")

    expected_shape = (d_model, d_model)
    if W_q.shape == expected_shape and W_k.shape == expected_shape and W_v.shape == expected_shape:
        print(f"\n✓ All weight matrices have correct shape: {expected_shape}")
    else:
        print(f"\n✗ Weight matrices should have shape {expected_shape}")
else:
    print("⚠ Weight matrices are None. Complete TODO 1 first!")

# ==============================================================================
# TODO 2: Project Input to Q, K, V
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 2: Project Input to Q, K, V")
print("=" * 70)

print("""
Task: Transform input embeddings X into queries, keys, and values.

Formula:
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

Steps:
1. Multiply X by W_q to get Q
2. Multiply X by W_k to get K
3. Multiply X by W_v to get V

Hints:
- Use @ operator for matrix multiplication
- X has shape (seq_len, d_model)
- Each weight matrix has shape (d_model, d_model)
- Result should have shape (seq_len, d_model)
""")

# TODO: Compute Q, K, V
# YOUR CODE HERE:
Q = None  # Replace with your implementation
K = None  # Replace with your implementation
V = None  # Replace with your implementation

# SOLUTION (uncomment to see/verify):
# if W_q is not None and W_k is not None and W_v is not None:
#     Q = X @ W_q
#     K = X @ W_k
#     V = X @ W_v

# Verification
if Q is not None and K is not None and V is not None:
    print(f"✓ Q shape: {Q.shape}")
    print(f"✓ K shape: {K.shape}")
    print(f"✓ V shape: {V.shape}")

    expected_shape = (seq_len, d_model)
    if Q.shape == expected_shape and K.shape == expected_shape and V.shape == expected_shape:
        print(f"\n✓ All projections have correct shape: {expected_shape}")
        print("\nKey insight: Q, K, V all came from the SAME input X!")
        print("This is why it's called SELF-attention!")
    else:
        print(f"\n✗ Projections should have shape {expected_shape}")
else:
    print("⚠ Q, K, V are None. Complete TODO 2 first!")

# ==============================================================================
# TODO 3: Implement Self-Attention Forward Pass
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 3: Implement Self-Attention Forward Pass")
print("=" * 70)

print("""
Task: Create a function that computes self-attention given Q, K, V.

The function should:
1. Compute attention scores: (Q @ K^T) / sqrt(d_k)
2. Apply softmax to get attention weights
3. Compute output: attention_weights @ V
4. Return both output and attention_weights

Reuse what you learned in Exercise 01!
""")

def self_attention_forward(Q, K, V):
    """
    Compute self-attention.

    Args:
        Q: Queries, shape (seq_len, d_k)
        K: Keys, shape (seq_len, d_k)
        V: Values, shape (seq_len, d_k)

    Returns:
        output: Attention output, shape (seq_len, d_k)
        attention_weights: Attention weights, shape (seq_len, seq_len)
    """
    # TODO: Implement self-attention forward pass
    # YOUR CODE HERE:
    pass

    # SOLUTION (uncomment to see/verify):
    # d_k = Q.shape[-1]
    #
    # # Compute attention scores
    # scores = (Q @ K.T) / np.sqrt(d_k)
    #
    # # Apply softmax
    # attention_weights = softmax(scores)
    #
    # # Compute output
    # output = attention_weights @ V
    #
    # return output, attention_weights

# Test your implementation
if Q is not None and K is not None and V is not None:
    try:
        output, attention_weights = self_attention_forward(Q, K, V)

        if output is not None and attention_weights is not None:
            print(f"✓ Self-attention forward pass works!")
            print(f"  Output shape: {output.shape}")
            print(f"  Attention weights shape: {attention_weights.shape}")

            # Verify attention weights sum to 1
            if np.allclose(attention_weights.sum(axis=1), 1.0):
                print("\n✓ Attention weights sum to 1.0 (valid probabilities)!")
            else:
                print("\n✗ Attention weights should sum to 1.0 along each row.")
        else:
            print("⚠ Function returned None. Complete TODO 3 first!")
    except:
        print("⚠ Error in self_attention_forward. Complete TODO 3 first!")
else:
    print("⚠ Complete previous TODOs first!")

# ==============================================================================
# TODO 4: Create Self-Attention Class
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 4: Create Self-Attention Class")
print("=" * 70)

print("""
Task: Create a class that encapsulates self-attention with learned weights.

The class should:
1. Initialize W_q, W_k, W_v in __init__
2. Have a forward method that:
   - Projects X to Q, K, V
   - Calls self_attention_forward
   - Returns output and attention_weights

C# Analogy:
    public class SelfAttention {
        private Matrix W_q, W_k, W_v;

        public SelfAttention(int d_model) {
            W_q = Matrix.Random(d_model, d_model);
            // ... etc
        }

        public (Matrix output, Matrix weights) Forward(Matrix X) {
            // ...
        }
    }
""")

class SelfAttention:
    """Self-Attention Layer."""

    def __init__(self, d_model):
        """
        Initialize self-attention layer.

        Args:
            d_model: Embedding dimension
        """
        # TODO: Initialize weight matrices
        # YOUR CODE HERE:
        pass

        # SOLUTION (uncomment to see/verify):
        # self.d_model = d_model
        # self.W_q = np.random.randn(d_model, d_model) * 0.1
        # self.W_k = np.random.randn(d_model, d_model) * 0.1
        # self.W_v = np.random.randn(d_model, d_model) * 0.1
        # print(f"✓ SelfAttention layer created with d_model={d_model}")

    def forward(self, X):
        """
        Forward pass.

        Args:
            X: Input embeddings, shape (seq_len, d_model)

        Returns:
            output: Context-aware embeddings, shape (seq_len, d_model)
            attention_weights: Attention matrix, shape (seq_len, seq_len)
        """
        # TODO: Implement forward pass
        # YOUR CODE HERE:
        pass

        # SOLUTION (uncomment to see/verify):
        # # Project to Q, K, V
        # Q = X @ self.W_q
        # K = X @ self.W_k
        # V = X @ self.W_v
        #
        # # Compute self-attention
        # output, attention_weights = self_attention_forward(Q, K, V)
        #
        # return output, attention_weights

# Test your class
try:
    attention_layer = SelfAttention(d_model)
    test_output, test_weights = attention_layer.forward(X)

    if test_output is not None and test_weights is not None:
        print("\n✓ SelfAttention class works!")
        print(f"  Input shape:  {X.shape}")
        print(f"  Output shape: {test_output.shape}")
        print(f"  Weights shape: {test_weights.shape}")
        print("\n✓ The class successfully encapsulates self-attention!")

        # Visualize attention
        print("\n" + "=" * 70)
        print("Visualizing Self-Attention Patterns")
        print("=" * 70)

        plt.figure(figsize=(8, 6))
        sns.heatmap(test_weights,
                    annot=True,
                    fmt='.2f',
                    cmap='viridis',
                    xticklabels=sentence,
                    yticklabels=sentence,
                    cbar_kws={'label': 'Attention Weight'})
        plt.title('Self-Attention Weights', fontsize=14, fontweight='bold')
        plt.xlabel('Keys (attending TO)', fontsize=11)
        plt.ylabel('Queries (attending FROM)', fontsize=11)
        plt.tight_layout()
        plt.show()

    else:
        print("⚠ Class methods returned None. Complete TODO 4 first!")
except Exception as e:
    print(f"⚠ Error creating/using SelfAttention class: {e}")
    print("Complete TODO 4 first!")

# ==============================================================================
# BONUS TODO 5: Add Multi-Head Capability
# ==============================================================================

print("\n" + "=" * 70)
print("BONUS TODO 5: Implement Multi-Head Self-Attention")
print("=" * 70)

print("""
BONUS Task: Extend your self-attention to support multiple heads!

Requirements:
1. Split d_model into num_heads chunks
2. Run attention separately for each head
3. Concatenate results
4. Apply output projection W_o

This is more challenging - try breaking it into small steps!

Hints:
- d_k = d_model // num_heads (dimension per head)
- Use reshape and transpose to split/combine heads
- Each head processes d_k dimensions independently
""")

class MultiHeadSelfAttention:
    """Multi-Head Self-Attention Layer."""

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head self-attention.

        Args:
            d_model: Total embedding dimension
            num_heads: Number of attention heads
        """
        # TODO: Implement multi-head initialization
        # YOUR CODE HERE:
        pass

        # SOLUTION (uncomment to see/verify):
        # assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        #
        # self.d_model = d_model
        # self.num_heads = num_heads
        # self.d_k = d_model // num_heads
        #
        # # Weight matrices for all heads
        # self.W_q = np.random.randn(d_model, d_model) * 0.1
        # self.W_k = np.random.randn(d_model, d_model) * 0.1
        # self.W_v = np.random.randn(d_model, d_model) * 0.1
        # self.W_o = np.random.randn(d_model, d_model) * 0.1  # Output projection
        #
        # print(f"✓ MultiHeadSelfAttention: {num_heads} heads, d_k={self.d_k}")

    def forward(self, X):
        """
        Forward pass with multi-head attention.

        Args:
            X: Input, shape (seq_len, d_model)

        Returns:
            output: Output, shape (seq_len, d_model)
            all_attention_weights: List of attention weights per head
        """
        # TODO: Implement multi-head forward pass
        # YOUR CODE HERE:
        pass

        # SOLUTION (uncomment to see/verify):
        # seq_len, d_model = X.shape
        #
        # # Project to Q, K, V
        # Q = X @ self.W_q  # (seq_len, d_model)
        # K = X @ self.W_k
        # V = X @ self.W_v
        #
        # # Split into heads
        # Q = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        # K = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        # V = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        # # Now shape: (num_heads, seq_len, d_k)
        #
        # # Compute attention for each head
        # outputs = []
        # all_weights = []
        # for h in range(self.num_heads):
        #     head_output, head_weights = self_attention_forward(Q[h], K[h], V[h])
        #     outputs.append(head_output)
        #     all_weights.append(head_weights)
        #
        # # Concatenate heads
        # # Stack: (num_heads, seq_len, d_k) -> (seq_len, num_heads, d_k)
        # concat = np.stack(outputs, axis=1)
        # concat = concat.reshape(seq_len, d_model)
        #
        # # Output projection
        # output = concat @ self.W_o
        #
        # return output, all_weights

# Test multi-head attention (if you implemented it)
try:
    multi_head = MultiHeadSelfAttention(d_model=d_model, num_heads=2)
    mh_output, mh_weights = multi_head.forward(X)

    if mh_output is not None and mh_weights is not None:
        print("\n✓ Multi-Head Self-Attention works!")
        print(f"  Input shape:  {X.shape}")
        print(f"  Output shape: {mh_output.shape}")
        print(f"  Number of heads: {len(mh_weights)}")

        # Visualize all heads
        fig, axes = plt.subplots(1, len(mh_weights), figsize=(12, 5))
        if len(mh_weights) == 1:
            axes = [axes]

        for h, (ax, weights) in enumerate(zip(axes, mh_weights)):
            sns.heatmap(weights,
                        annot=True,
                        fmt='.2f',
                        cmap='Blues',
                        xticklabels=sentence,
                        yticklabels=sentence,
                        ax=ax,
                        cbar=False)
            ax.set_title(f'Head {h+1}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Keys', fontsize=9)
            if h == 0:
                ax.set_ylabel('Queries', fontsize=9)

        plt.suptitle('Multi-Head Attention Patterns', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print("\n✓ BONUS completed! You implemented multi-head attention!")
    else:
        print("⚠ Multi-head methods returned None. Complete BONUS TODO 5!")
except Exception as e:
    print(f"⚠ Multi-head not implemented or error: {e}")
    print("This is a BONUS task - skip if it's too challenging!")

# ==============================================================================
# CONGRATULATIONS!
# ==============================================================================

print("\n" + "=" * 70)
print("EXERCISE 02 COMPLETE!")
print("=" * 70)

print("""
Excellent work! You've built a complete self-attention layer!

What you learned:
✓ How to initialize learned weight matrices (W_q, W_k, W_v)
✓ How to project inputs to queries, keys, and values
✓ How to create a self-attention class
✓ How attention captures relationships within a sequence
✓ BONUS: How multi-head attention works!

You now understand how transformers build context-aware representations!

Next exercise:
- exercise_03: Build a complete transformer block with:
  - Multi-head attention
  - Feed-forward network
  - Layer normalization
  - Residual connections

Almost there! 🚀
""")

print("\n" + "=" * 70)
print("END OF EXERCISE 02")
print("=" * 70)

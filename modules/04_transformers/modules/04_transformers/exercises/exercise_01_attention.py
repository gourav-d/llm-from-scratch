"""
Exercise 01: Implementing Attention from Scratch

In this exercise, you'll implement the core attention mechanism step-by-step.
This will solidify your understanding of how attention works!

Tasks:
1. Implement scaled dot-product attention scores
2. Implement softmax function
3. Compute attention output (weighted sum)
4. Visualize attention weights

Each TODO section has:
- Clear instructions
- Hints
- Solution (commented out - try first before looking!)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("=" * 70)
print("EXERCISE 01: IMPLEMENTING ATTENTION")
print("=" * 70)

# ==============================================================================
# SETUP: Creating Sample Data
# ==============================================================================

print("\n" + "=" * 70)
print("SETUP: Creating Sample Q, K, V")
print("=" * 70)

# Sample sentence
sentence = ["The", "cat", "sat", "on", "mat"]
seq_len = len(sentence)
d_k = 4  # Dimension of keys/queries

print(f"Sentence: {' '.join(sentence)}")
print(f"Sequence length: {seq_len}")
print(f"Dimension: {d_k}")

# Create random Q, K, V
Q = np.random.randn(seq_len, d_k) * 0.5
K = np.random.randn(seq_len, d_k) * 0.5
V = np.random.randn(seq_len, d_k) * 0.5

print(f"\nQ shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")

# ==============================================================================
# TODO 1: Implement Attention Scores
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 1: Compute Attention Scores")
print("=" * 70)

print("""
Task: Compute scaled dot-product attention scores.

Formula: scores = (Q @ K^T) / sqrt(d_k)

Steps:
1. Compute Q @ K^T (matrix multiplication)
2. Divide by sqrt(d_k) for scaling
3. Store result in 'attention_scores'

Hints:
- Use @ for matrix multiplication
- Use .T for transpose
- Use np.sqrt() for square root

Try it yourself below!
""")

# TODO: Implement attention scores calculation
# YOUR CODE HERE:
attention_scores = None  # Replace this with your implementation

# SOLUTION (uncomment to see/verify):
# attention_scores = (Q @ K.T) / np.sqrt(d_k)

# Verification
if attention_scores is not None:
    print(f"✓ Attention scores shape: {attention_scores.shape}")
    print(f"  Expected: ({seq_len}, {seq_len})")
    print(f"\nAttention scores:\n{attention_scores}")

    # Check if implementation is correct
    expected_shape = (seq_len, seq_len)
    if attention_scores.shape == expected_shape:
        print("\n✓ Shape is correct!")
    else:
        print(f"\n✗ Shape is wrong. Expected {expected_shape}, got {attention_scores.shape}")
else:
    print("⚠ attention_scores is None. Complete TODO 1 first!")

# ==============================================================================
# TODO 2: Implement Softmax Function
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 2: Implement Softmax")
print("=" * 70)

print("""
Task: Implement the softmax function to convert scores to probabilities.

Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

Steps:
1. Compute exp(x) for each element (use np.exp)
2. For numerical stability, subtract max: exp(x - max(x))
3. Divide by sum of all exponentials
4. Return the result

Hints:
- Use np.exp() for exponentials
- Use np.max() with keepdims=True
- Use np.sum() with keepdims=True
- Apply along last axis (axis=-1)

The function should work on any input shape!
""")

def softmax(x):
    """
    Apply softmax function along the last axis.

    Args:
        x: Input array of any shape

    Returns:
        Softmax probabilities (same shape as input)
    """
    # TODO: Implement softmax
    # YOUR CODE HERE:
    pass  # Remove this and add your implementation

    # SOLUTION (uncomment to see/verify):
    # exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    # return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Test your softmax implementation
if attention_scores is not None:
    try:
        test_input = np.array([[1, 2, 3], [4, 5, 6]])
        test_output = softmax(test_input)

        if test_output is not None:
            print("Testing softmax on sample input:")
            print(f"Input:\n{test_input}")
            print(f"\nOutput:\n{test_output}")
            print(f"\nRow sums (should be ~1.0):")
            for i, row_sum in enumerate(test_output.sum(axis=1)):
                print(f"  Row {i}: {row_sum:.6f}")

            # Check if sums are close to 1
            if np.allclose(test_output.sum(axis=1), 1.0):
                print("\n✓ Softmax implementation looks correct!")
            else:
                print("\n✗ Row sums should be 1.0. Check your implementation.")
        else:
            print("⚠ softmax returned None. Complete TODO 2 first!")
    except:
        print("⚠ Error in softmax. Complete TODO 2 first!")

# ==============================================================================
# TODO 3: Compute Attention Weights
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 3: Apply Softmax to Get Attention Weights")
print("=" * 70)

print("""
Task: Apply your softmax function to the attention scores.

Steps:
1. Use the softmax function you implemented
2. Apply it to attention_scores
3. Store result in 'attention_weights'

This converts raw scores into probabilities that sum to 1!
""")

# TODO: Apply softmax to attention scores
# YOUR CODE HERE:
attention_weights = None  # Replace with your implementation

# SOLUTION (uncomment to see/verify):
# if attention_scores is not None:
#     attention_weights = softmax(attention_scores)

# Verification
if attention_weights is not None:
    print(f"✓ Attention weights shape: {attention_weights.shape}")
    print(f"\nAttention weights:\n{attention_weights}")

    # Verify rows sum to 1
    print(f"\nVerifying rows sum to 1.0:")
    for i, word in enumerate(sentence):
        row_sum = attention_weights[i].sum()
        status = "✓" if abs(row_sum - 1.0) < 1e-5 else "✗"
        print(f"  {status} '{word}': {row_sum:.6f}")

    if np.allclose(attention_weights.sum(axis=1), 1.0):
        print("\n✓ All rows sum to 1.0! Attention weights are valid!")
    else:
        print("\n✗ Rows should sum to 1.0. Check your implementation.")
else:
    print("⚠ attention_weights is None. Complete TODO 3 first!")

# ==============================================================================
# TODO 4: Compute Attention Output
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 4: Compute Attention Output")
print("=" * 70)

print("""
Task: Compute the final attention output using weighted sum of values.

Formula: output = attention_weights @ V

Steps:
1. Multiply attention_weights by V using matrix multiplication
2. Store result in 'attention_output'

This gives us context-aware representations!
""")

# TODO: Compute attention output
# YOUR CODE HERE:
attention_output = None  # Replace with your implementation

# SOLUTION (uncomment to see/verify):
# if attention_weights is not None:
#     attention_output = attention_weights @ V

# Verification
if attention_output is not None:
    print(f"✓ Attention output shape: {attention_output.shape}")
    print(f"  Expected: ({seq_len}, {d_k})")

    print(f"\nComparing original vs attention output for '{sentence[2]}':")
    print(f"  Original V[2]:         {V[2]}")
    print(f"  After attention:       {attention_output[2]}")

    # Show how it's computed
    if attention_weights is not None:
        print(f"\n  This is computed as a weighted sum:")
        for i, word in enumerate(sentence):
            weight = attention_weights[2, i]
            print(f"    {weight:.3f} × V[{i}] ('{word}')")

    if attention_output.shape == (seq_len, d_k):
        print("\n✓ Output shape is correct!")
    else:
        print(f"\n✗ Shape is wrong. Expected ({seq_len}, {d_k}), got {attention_output.shape}")
else:
    print("⚠ attention_output is None. Complete TODO 4 first!")

# ==============================================================================
# TODO 5: Visualize Attention Weights
# ==============================================================================

print("\n" + "=" * 70)
print("TODO 5: Visualize Attention Weights")
print("=" * 70)

print("""
Task: Create a heatmap visualization of attention weights.

Steps:
1. Use plt.figure() to create a figure
2. Use sns.heatmap() to create the heatmap with:
   - attention_weights as data
   - annot=True to show numbers
   - fmt='.2f' for 2 decimal places
   - xticklabels=sentence and yticklabels=sentence for labels
3. Add title and axis labels
4. Use plt.show() to display

Try creating a nice visualization!
""")

# TODO: Create heatmap visualization
# YOUR CODE HERE:
# (Create your visualization here)

# SOLUTION (uncomment to see/verify):
# if attention_weights is not None:
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(attention_weights,
#                 annot=True,
#                 fmt='.2f',
#                 cmap='Blues',
#                 xticklabels=sentence,
#                 yticklabels=sentence,
#                 cbar_kws={'label': 'Attention Weight'})
#     plt.title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
#     plt.xlabel('Keys (attending TO)', fontsize=11)
#     plt.ylabel('Queries (attending FROM)', fontsize=11)
#     plt.tight_layout()
#     plt.show()
#     print("✓ Visualization created!")
# else:
#     print("⚠ Complete previous TODOs first to create visualization!")

# ==============================================================================
# BONUS: Create a Complete Attention Function
# ==============================================================================

print("\n" + "=" * 70)
print("BONUS: Create Complete Attention Function")
print("=" * 70)

print("""
BONUS Task: Combine everything into a single attention function!

Create a function attention(Q, K, V) that:
1. Computes attention scores
2. Applies softmax
3. Computes output
4. Returns both output and attention weights

This is the complete scaled dot-product attention!
""")

def attention(Q, K, V):
    """
    Scaled dot-product attention.

    Args:
        Q: Queries, shape (seq_len, d_k)
        K: Keys, shape (seq_len, d_k)
        V: Values, shape (seq_len, d_k)

    Returns:
        output: Attention output, shape (seq_len, d_k)
        weights: Attention weights, shape (seq_len, seq_len)
    """
    # TODO: Implement complete attention function
    # YOUR CODE HERE:
    pass

    # SOLUTION (uncomment to see/verify):
    # d_k = Q.shape[-1]
    #
    # # Compute scores
    # scores = (Q @ K.T) / np.sqrt(d_k)
    #
    # # Apply softmax
    # weights = softmax(scores)
    #
    # # Compute output
    # output = weights @ V
    #
    # return output, weights

# Test your complete attention function
try:
    test_output, test_weights = attention(Q, K, V)

    if test_output is not None and test_weights is not None:
        print("✓ Attention function works!")
        print(f"  Output shape: {test_output.shape}")
        print(f"  Weights shape: {test_weights.shape}")

        # Verify it matches your previous results
        if attention_output is not None and attention_weights is not None:
            if np.allclose(test_output, attention_output) and np.allclose(test_weights, attention_weights):
                print("\n✓ Results match your step-by-step implementation!")
            else:
                print("\n⚠ Results don't match. Check your implementation.")
except:
    print("⚠ Complete the BONUS task to test the complete attention function!")

# ==============================================================================
# CONGRATULATIONS!
# ==============================================================================

print("\n" + "=" * 70)
print("EXERCISE COMPLETE!")
print("=" * 70)

print("""
Great work! You've implemented the core attention mechanism from scratch!

What you learned:
✓ How to compute attention scores using Q @ K^T
✓ Why we scale by sqrt(d_k)
✓ How softmax converts scores to probabilities
✓ How to compute weighted sum of values
✓ How to visualize attention patterns

You now understand the FOUNDATION of transformers!

Next exercise:
- exercise_02: Implement self-attention with learned weights
- exercise_03: Build a complete transformer block

Keep going! 🚀
""")

print("\n" + "=" * 70)
print("END OF EXERCISE 01")
print("=" * 70)

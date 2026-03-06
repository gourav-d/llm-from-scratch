"""
Example 01: Basic Attention Mechanism

This example demonstrates the fundamental attention mechanism - the core building
block of transformers. You'll see how attention allows a model to focus on
different parts of the input when processing each element.

What you'll see:
1. Query, Key, Value concept explained with a library search analogy
2. Computing attention scores (similarity between queries and keys)
3. Softmax to convert scores into weights (probabilities)
4. Weighted averaging of values using attention weights
5. Visualization of attention patterns

Think of it like searching a library:
- Query (Q): What you're looking for ("books about Python")
- Keys (K): Index cards describing each book
- Values (V): The actual books on the shelf
- Attention: Finding the most relevant books based on your query
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility (like C# Random with fixed seed)
np.random.seed(42)

print("=" * 70)
print("BASIC ATTENTION MECHANISM")
print("=" * 70)

# ==============================================================================
# PART 1: Understanding Query, Key, Value
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Query, Key, Value Concept")
print("=" * 70)

# Let's use a simple sentence as our example
sentence = ["The", "cat", "sat", "on", "the", "mat"]
print(f"\nInput sentence: {' '.join(sentence)}")

# In transformers, each word is represented as a vector (array of numbers)
# For this example, we'll use small 4-dimensional vectors
# (In real transformers, this might be 512 or higher!)
d_model = 4  # Dimension of our word embeddings

print(f"\nEach word will be represented as a {d_model}-dimensional vector")

# Create random embeddings for each word
# Shape: (num_words, d_model) = (6, 4)
# Similar to C#: float[6][4] or List<float[]> with 6 vectors of size 4
word_embeddings = np.random.randn(len(sentence), d_model)

print(f"Word embeddings shape: {word_embeddings.shape}")
print(f"\nExample - embedding for '{sentence[0]}':")
print(word_embeddings[0])

# ==============================================================================
# PART 2: Creating Query, Key, Value Matrices
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Creating Queries, Keys, and Values")
print("=" * 70)

print("""
In basic attention, we transform our embeddings into three different
representations using learned weight matrices:

- Query (Q): "What am I looking for?"
- Key (K): "What do I contain?"
- Value (V): "What information do I hold?"

Think of it like a database query:
- Q is your search query
- K is the index you search against
- V is the actual data you retrieve
""")

# For this example, we'll use the same embeddings for Q, K, and V
# This is called "self-attention" (we'll explore this more in example_02)
Q = word_embeddings  # Queries: what each word is looking for
K = word_embeddings  # Keys: what each word offers
V = word_embeddings  # Values: the actual information from each word

print(f"Q (Queries) shape: {Q.shape}")
print(f"K (Keys) shape: {K.shape}")
print(f"V (Values) shape: {V.shape}")

# ==============================================================================
# PART 3: Computing Attention Scores
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Computing Attention Scores")
print("=" * 70)

print("""
Attention scores measure how relevant each key is to each query.
We compute this using the dot product (matrix multiplication).

Higher score = more relevant = more attention
""")

# Compute attention scores: Q @ K^T
# This gives us a score for every query-key pair
# Shape: (6, 4) @ (4, 6) = (6, 6)
# Similar to C# LINQ: queries.SelectMany(q => keys.Select(k => DotProduct(q, k)))
attention_scores = Q @ K.T  # @ is matrix multiplication, .T is transpose

print(f"Attention scores shape: {attention_scores.shape}")
print(f"This is a {len(sentence)} x {len(sentence)} matrix")
print("Each row shows how much one word 'attends to' all other words")

print("\nAttention scores (before scaling):")
print(attention_scores)

# Scale by square root of dimension (prevents scores from getting too large)
# This is a key trick in transformers to keep gradients stable
d_k = d_model  # dimension of keys
scaling_factor = np.sqrt(d_k)
scaled_scores = attention_scores / scaling_factor

print(f"\nScaling factor: sqrt({d_k}) = {scaling_factor:.2f}")
print("\nScaled attention scores:")
print(scaled_scores)

# ==============================================================================
# PART 4: Applying Softmax to Get Attention Weights
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Converting Scores to Weights with Softmax")
print("=" * 70)

print("""
Softmax converts scores into probabilities that sum to 1.
This tells us "how much" to attend to each word (0% to 100%).

Softmax formula: exp(x_i) / sum(exp(x_j))
It's like C# LINQ: scores.Select(s => Math.Exp(s) / scores.Sum(x => Math.Exp(x)))
""")

def softmax(x, axis=-1):
    """
    Apply softmax function to convert scores into probabilities.

    Args:
        x: Input array (attention scores)
        axis: Axis along which to apply softmax (default: last axis)

    Returns:
        Array of probabilities that sum to 1 along the specified axis
    """
    # Subtract max for numerical stability (prevents overflow)
    # This is a standard trick: softmax(x) = softmax(x - max(x))
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))

    # Divide by sum to get probabilities
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Apply softmax to each row (each query attends to all keys)
attention_weights = softmax(scaled_scores, axis=-1)

print("Attention weights (probabilities):")
print(attention_weights)

print("\nVerify each row sums to 1.0:")
for i, word in enumerate(sentence):
    row_sum = attention_weights[i].sum()
    print(f"  '{word}': {row_sum:.6f}")

# ==============================================================================
# PART 5: Computing Weighted Sum of Values
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Computing Context-Aware Representations")
print("=" * 70)

print("""
Now we use the attention weights to compute a weighted average of the values.
Each word gets a new representation that incorporates information from all
words it attended to, weighted by the attention scores.

This is the OUTPUT of the attention mechanism!
""")

# Compute attention output: attention_weights @ V
# Shape: (6, 6) @ (6, 4) = (6, 4)
# Each word gets a new 4-dimensional vector
attention_output = attention_weights @ V

print(f"Attention output shape: {attention_output.shape}")
print("\nAttention output (new word representations):")
print(attention_output)

print(f"\nExample - new representation for '{sentence[2]}' (sat):")
print(f"Original embedding: {V[2]}")
print(f"After attention:    {attention_output[2]}")
print("\nThis new vector incorporates information from all words,")
print("weighted by how much 'sat' attended to each word!")

# ==============================================================================
# PART 6: Visualizing Attention Patterns
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Visualizing Attention Weights")
print("=" * 70)

# Create a heatmap of attention weights
plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights,
            annot=True,  # Show numbers in cells
            fmt='.3f',   # Format: 3 decimal places
            cmap='YlOrRd',  # Yellow-Orange-Red colormap
            xticklabels=sentence,
            yticklabels=sentence,
            cbar_kws={'label': 'Attention Weight'})

plt.title('Attention Weights: Who Attends to Whom?', fontsize=14, fontweight='bold')
plt.xlabel('Keys (attending TO these words)', fontsize=12)
plt.ylabel('Queries (attention FROM these words)', fontsize=12)

# Add explanation text
plt.text(0.5, -0.15,
         'Each row shows how much a word attends to all other words.\n'
         'Higher values (red) = more attention, Lower values (yellow) = less attention',
         ha='center', va='top', transform=plt.gca().transAxes,
         fontsize=10, style='italic')

plt.tight_layout()
plt.show()

print("\nInterpretation of the heatmap:")
print("- Diagonal (top-left to bottom-right): How much each word attends to itself")
print("- Off-diagonal: How much words attend to OTHER words")
print("- Brighter colors (red/orange): Higher attention")
print("- Dimmer colors (yellow): Lower attention")

# ==============================================================================
# PART 7: Detailed Example - How One Word Attends
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Detailed Look - How 'cat' Attends to All Words")
print("=" * 70)

# Let's focus on the word "cat" (index 1)
word_idx = 1
focus_word = sentence[word_idx]

print(f"\nHow '{focus_word}' attends to each word:\n")
for i, word in enumerate(sentence):
    weight = attention_weights[word_idx, i]
    bar = '█' * int(weight * 50)  # Visual bar (similar to console progress bar in C#)
    print(f"  {word:6s}: {weight:.3f} {bar}")

print(f"\nThe new representation for '{focus_word}' is a weighted combination:")
print(f"  {attention_weights[word_idx, 0]:.3f} × '{sentence[0]}' embedding")
print(f"  {attention_weights[word_idx, 1]:.3f} × '{sentence[1]}' embedding")
print(f"  {attention_weights[word_idx, 2]:.3f} × '{sentence[2]}' embedding")
print(f"  {attention_weights[word_idx, 3]:.3f} × '{sentence[3]}' embedding")
print(f"  {attention_weights[word_idx, 4]:.3f} × '{sentence[4]}' embedding")
print(f"  {attention_weights[word_idx, 5]:.3f} × '{sentence[5]}' embedding")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - What We Learned")
print("=" * 70)

print("""
✓ Attention Mechanism allows each word to focus on relevant parts of input
✓ Query (Q): What am I looking for?
✓ Key (K): What information do I have?
✓ Value (V): The actual information to retrieve
✓ Attention Scores: Dot product of Q and K (measures relevance)
✓ Softmax: Converts scores to probabilities (weights)
✓ Output: Weighted sum of Values based on attention weights

The Formula:
    Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

This is the FOUNDATION of transformers! 🎯

Next Steps:
- example_02: Self-attention (Q, K, V all from same source)
- example_03: Multi-head attention (multiple attention mechanisms in parallel)
- example_04: Positional encoding (adding position information)
- example_05: Complete transformer block
- example_06: Building a mini-GPT!
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 01")
print("=" * 70)

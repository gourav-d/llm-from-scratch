"""
Example 04: Positional Encoding

This example demonstrates how transformers learn word order using sinusoidal
positional encodings. Without this, attention mechanisms would be "blind" to
the sequence order!

What you'll see:
1. Why pure attention is order-blind (permutation invariant)
2. How positional encoding solves this problem
3. Sinusoidal encoding using sine and cosine waves
4. Different frequencies for different dimensions
5. Comprehensive visualizations of position patterns

Think of it like house addresses:
- Without positions: "cat", "sat", "mat" (unordered bag of words)
- With positions: "cat[1]", "sat[2]", "mat[3]" (ordered sequence)
- Sine/cosine waves create unique "fingerprints" for each position
- Like GPS coordinates that uniquely identify locations!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("POSITIONAL ENCODING - Teaching Transformers About Word Order")
print("=" * 70)

# ==============================================================================
# PART 1: The Problem - Attention is Order-Blind
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: The Problem - Attention Doesn't Know Word Order")
print("=" * 70)

print("""
CRITICAL PROBLEM: Self-attention is permutation-invariant!

Consider these two sentences:
  1. "The cat sat on the mat"
  2. "The mat sat on the cat"

With pure self-attention (no positional info):
  - Same words present in both sentences
  - Self-attention computes same attention patterns
  - Output would be IDENTICAL!

But the meanings are completely different!

Analogy:
  - Like a bag of Scrabble tiles: {T, H, E, C, A, T, ...}
  - Can't distinguish "CAT" from "ACT" from "TAC"
  - Need to know POSITION to understand meaning!

In C# terms:
  HashSet<string> words = {"The", "cat", "sat", "on", "mat"};
  // Lost the order! Can't reconstruct original sentence.

  List<string> sentence = {"The", "cat", "sat", "on", "mat"};
  // Preserves order! Position matters.
""")

# ==============================================================================
# PART 2: The Solution - Positional Encoding
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: The Solution - Add Position Information")
print("=" * 70)

print("""
SOLUTION: Add positional information to embeddings!

  Original: word_embedding
  Enhanced: word_embedding + positional_encoding

  Position 0: "The" → embedding + PE(0)
  Position 1: "cat" → embedding + PE(1)
  Position 2: "sat" → embedding + PE(2)
  ...

Now the model knows:
  - What the word is (from word embedding)
  - Where the word is (from positional encoding)

Like C# tuples combining data:
  (string word, int position) = ("cat", 1);
  var enhanced = CombineWordAndPosition(word, position);
""")

# ==============================================================================
# PART 3: Sinusoidal Positional Encoding Formula
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: The Sinusoidal Encoding Formula")
print("=" * 70)

print("""
The transformer paper uses sine and cosine waves:

For position 'pos' and dimension 'i':

  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))    [even indices]
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))    [odd indices]

Why sine/cosine?
  ✓ Unique pattern for each position
  ✓ Works for any sequence length (no maximum)
  ✓ Smooth transitions between positions
  ✓ Model can learn relative positions

Different dimensions use different frequencies:
  - Low dimensions: fast oscillation (captures nearby positions)
  - High dimensions: slow oscillation (captures distant positions)

Similar to C#:
  double PE(int pos, int i, int d_model) {
      double angle = pos / Math.Pow(10000, (2.0 * i) / d_model);
      return (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
  }
""")

# ==============================================================================
# PART 4: Implementing Positional Encoding
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Implementing Positional Encoding")
print("=" * 70)

def positional_encoding(max_seq_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Args:
        max_seq_len: Maximum sequence length
        d_model: Embedding dimension

    Returns:
        Positional encoding matrix of shape (max_seq_len, d_model)

    Equivalent C# method:
        double[,] PositionalEncoding(int maxSeqLen, int dModel) {
            var PE = new double[maxSeqLen, dModel];

            for (int pos = 0; pos < maxSeqLen; pos++) {
                for (int i = 0; i < dModel; i++) {
                    double angle = pos / Math.Pow(10000, (2.0 * i) / dModel);
                    PE[pos, i] = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                }
            }
            return PE;
        }
    """
    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    # Shape: (max_seq_len, 1)
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Create dimension indices and compute division term
    # For even indices (2i): i = 0, 1, 2, ...
    # 10000^(2i/d_model) = exp(2i * log(10000) / d_model)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # Initialize positional encoding matrix
    PE = np.zeros((max_seq_len, d_model))

    # Apply sine to even indices (0, 2, 4, ...)
    PE[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices (1, 3, 5, ...)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE

# Generate positional encodings
max_seq_len = 50  # Maximum sequence length
d_model = 128     # Embedding dimension

PE = positional_encoding(max_seq_len, d_model)

print(f"Generated positional encoding")
print(f"  Shape: {PE.shape}")
print(f"  Max sequence length: {max_seq_len}")
print(f"  Embedding dimension: {d_model}")

print(f"\nPositional encoding for position 0 (first 8 dims):")
print(PE[0, :8])

print(f"\nPositional encoding for position 10 (first 8 dims):")
print(PE[10, :8])

print("\nNotice: Different positions have different patterns!")

# ==============================================================================
# PART 5: Applying to Word Embeddings
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Adding Positional Encoding to Word Embeddings")
print("=" * 70)

# Create example word embeddings
sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(sentence)
d_model_small = 8  # Smaller dimension for visualization

print(f"Sentence: {' '.join(sentence)}")
print(f"Embedding dimension: {d_model_small}")

# Word embeddings (normally from embedding layer)
word_embeddings = np.random.randn(seq_len, d_model_small) * 0.5

# Positional encodings
PE_small = positional_encoding(seq_len, d_model_small)

# Combine: word embedding + positional encoding
# This is element-wise addition
enhanced_embeddings = word_embeddings + PE_small

print(f"\nWord embeddings shape: {word_embeddings.shape}")
print(f"Positional encodings shape: {PE_small.shape}")
print(f"Enhanced embeddings shape: {enhanced_embeddings.shape}")

print(f"\nExample for word '{sentence[2]}' (position 2):")
print(f"  Original embedding: {word_embeddings[2]}")
print(f"  Positional encoding: {PE_small[2]}")
print(f"  Enhanced (original + position): {enhanced_embeddings[2]}")

# ==============================================================================
# PART 6: Visualization 1 - Full Positional Encoding Heatmap
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Visualizations")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))

# Subplot 1: Full positional encoding heatmap
ax1 = plt.subplot(3, 2, 1)
plt.imshow(PE, cmap='RdBu', aspect='auto')
plt.colorbar(label='Encoding Value')
plt.title('Positional Encoding Heatmap\n(50 positions × 128 dimensions)',
          fontsize=12, fontweight='bold')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.text(0.5, -0.15, 'Each row is a unique position fingerprint',
         ha='center', transform=ax1.transAxes, style='italic')

# Subplot 2: Encoding for specific positions
ax2 = plt.subplot(3, 2, 2)
positions_to_show = [0, 5, 10, 20, 30, 40]
for pos in positions_to_show:
    plt.plot(PE[pos], label=f'Position {pos}', alpha=0.7)
plt.title('Encoding Patterns for Different Positions', fontsize=12, fontweight='bold')
plt.xlabel('Dimension')
plt.ylabel('Encoding Value')
plt.legend()
plt.grid(alpha=0.3)

# Subplot 3: First few dimensions over positions
ax3 = plt.subplot(3, 2, 3)
dims_to_show = [0, 1, 2, 3]
for dim in dims_to_show:
    plt.plot(PE[:, dim], label=f'Dim {dim}', alpha=0.7)
plt.title('How Different Dimensions Vary Across Positions', fontsize=12, fontweight='bold')
plt.xlabel('Position')
plt.ylabel('Encoding Value')
plt.legend()
plt.grid(alpha=0.3)

# Subplot 4: Position fingerprints (first 20 positions, first 20 dims)
ax4 = plt.subplot(3, 2, 4)
plt.imshow(PE[:20, :20], cmap='RdBu', aspect='auto')
plt.colorbar(label='Value')
plt.title('Position Fingerprints (Zoomed)\n(20 positions × 20 dimensions)',
          fontsize=12, fontweight='bold')
plt.xlabel('Dimension')
plt.ylabel('Position')

# Subplot 5: 2D visualization of position representations
ax5 = plt.subplot(3, 2, 5)
# Use first 2 dimensions for 2D plot
plt.scatter(PE[:, 0], PE[:, 1], c=np.arange(max_seq_len),
           cmap='viridis', s=50, alpha=0.6)
plt.colorbar(label='Position')
plt.title('2D Position Space (Dimensions 0 and 1)', fontsize=12, fontweight='bold')
plt.xlabel('Dimension 0 (Sine)')
plt.ylabel('Dimension 1 (Cosine)')
plt.grid(alpha=0.3)
plt.text(0.5, -0.15, 'Positions form a spiral pattern!',
         ha='center', transform=ax5.transAxes, style='italic')

# Subplot 6: Frequency spectrum
ax6 = plt.subplot(3, 2, 6)
# Show how frequency changes with dimension
dims = np.arange(0, d_model, 2)
frequencies = 1.0 / (10000.0 ** (dims / d_model))
plt.semilogy(dims, frequencies, 'o-', color='purple')
plt.title('Frequency Spectrum Across Dimensions', fontsize=12, fontweight='bold')
plt.xlabel('Dimension (even indices)')
plt.ylabel('Frequency (log scale)')
plt.grid(alpha=0.3)
plt.text(0.5, -0.15, 'Lower dims = higher frequency (fast oscillation)',
         ha='center', transform=ax6.transAxes, style='italic')

plt.tight_layout()
plt.show()

print("\nVisualization insights:")
print("  1. Each position has a unique pattern (fingerprint)")
print("  2. Nearby positions have similar patterns (smooth)")
print("  3. Different dimensions oscillate at different frequencies")
print("  4. Positions form structured patterns in the encoding space")
print("  5. Low dims capture local patterns, high dims capture global patterns")

# ==============================================================================
# PART 7: Understanding Frequency Bands
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Understanding Frequency Bands")
print("=" * 70)

print("""
Different dimensions use different frequencies:

Dimension 0-1 (highest frequency):
  - Oscillates quickly
  - Distinguishes adjacent positions (1 vs 2 vs 3)
  - Like seconds on a clock

Middle dimensions (medium frequency):
  - Oscillates moderately
  - Captures phrase-level patterns
  - Like minutes on a clock

High dimensions (lowest frequency):
  - Oscillates slowly
  - Captures sentence-level patterns
  - Like hours on a clock

This multi-scale representation lets the model understand:
  - Local word order (fast frequencies)
  - Phrase structure (medium frequencies)
  - Sentence structure (slow frequencies)

Similar to C# DateTime:
  DateTime.Second  → fast changing (local)
  DateTime.Minute  → medium changing (medium-range)
  DateTime.Hour    → slow changing (global)
""")

# Show examples of different frequency bands
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# High frequency (early dimensions)
axes[0].plot(PE[:, 0], label='Dimension 0', color='red', linewidth=2)
axes[0].plot(PE[:, 1], label='Dimension 1', color='darkred', linewidth=2)
axes[0].set_title('High Frequency (Dimensions 0-1): Captures Local Patterns',
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Position')
axes[0].set_ylabel('Encoding Value')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Medium frequency (middle dimensions)
mid = d_model // 2
axes[1].plot(PE[:, mid], label=f'Dimension {mid}', color='orange', linewidth=2)
axes[1].plot(PE[:, mid+1], label=f'Dimension {mid+1}', color='darkorange', linewidth=2)
axes[1].set_title(f'Medium Frequency (Dimensions {mid}-{mid+1}): Captures Phrase Patterns',
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Position')
axes[1].set_ylabel('Encoding Value')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Low frequency (late dimensions)
axes[2].plot(PE[:, -2], label=f'Dimension {d_model-2}', color='blue', linewidth=2)
axes[2].plot(PE[:, -1], label=f'Dimension {d_model-1}', color='darkblue', linewidth=2)
axes[2].set_title(f'Low Frequency (Dimensions {d_model-2}-{d_model-1}): Captures Global Patterns',
                  fontsize=12, fontweight='bold')
axes[2].set_xlabel('Position')
axes[2].set_ylabel('Encoding Value')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ==============================================================================
# PART 8: Practical Example with Real Sentence
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Practical Example - Before and After")
print("=" * 70)

sentence_long = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
seq_len_long = len(sentence_long)
d_model_demo = 16

print(f"Sentence: {' '.join(sentence_long)}")

# Word embeddings (pretend these are from a real embedding layer)
word_emb = np.random.randn(seq_len_long, d_model_demo) * 0.5

# Positional encodings
pos_enc = positional_encoding(seq_len_long, d_model_demo)

# Enhanced embeddings
enhanced = word_emb + pos_enc

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Word embeddings only
im1 = axes[0].imshow(word_emb, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
axes[0].set_title('Word Embeddings Only\n(No position info)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Position')
axes[0].set_yticks(range(seq_len_long))
axes[0].set_yticklabels(sentence_long)
axes[0].set_xlabel('Dimension')
plt.colorbar(im1, ax=axes[0])

# Positional encodings only
im2 = axes[1].imshow(pos_enc, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
axes[1].set_title('Positional Encodings Only\n(Position patterns)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Position')
axes[1].set_yticks(range(seq_len_long))
axes[1].set_yticklabels(sentence_long)
axes[1].set_xlabel('Dimension')
plt.colorbar(im2, ax=axes[1])

# Combined
im3 = axes[2].imshow(enhanced, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
axes[2].set_title('Word + Position\n(Complete information)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Position')
axes[2].set_yticks(range(seq_len_long))
axes[2].set_yticklabels(sentence_long)
axes[2].set_xlabel('Dimension')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()

print("\nKey observation:")
print("  - Left: Word embeddings (WHAT each word is)")
print("  - Middle: Positional encodings (WHERE each word is)")
print("  - Right: Combined (WHAT + WHERE)")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - What We Learned")
print("=" * 70)

print("""
✓ Problem: Self-attention is order-blind (permutation invariant)
✓ Solution: Add positional encodings to word embeddings
✓ Formula: PE(pos, 2i) = sin(pos/10000^(2i/d_model))
          PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
✓ Properties:
  - Unique pattern for each position
  - Works for any sequence length
  - Smooth transitions between positions
  - Different frequencies capture different scales

Why Sine/Cosine?
  ✓ Periodic functions create unique fingerprints
  ✓ Can extrapolate to longer sequences than seen in training
  ✓ Model can learn relative positions: PE(pos+k) is a linear
    function of PE(pos)
  ✓ Different dimensions capture different time scales

Frequency Bands:
  - Low dims (0-31): High frequency → local patterns
  - Mid dims (32-95): Medium frequency → phrase patterns
  - High dims (96-127): Low frequency → sentence patterns

In Practice:
  - GPT: Learns position embeddings (not sinusoidal)
  - BERT: Uses sinusoidal positional encodings
  - T5: Relative position embeddings
  - Different approaches, same goal: capture word order!

In C#/.NET Terms:
  - Like adding index to List<T>: (T item, int index)
  - Sine/cosine like clock hands (hour, minute, second)
  - Multi-frequency like DateTime hierarchy
  - Element-wise addition like LINQ Zip

Complete Pipeline:
  1. Get word embeddings: embedding_layer(token_ids)
  2. Generate positional encodings: PE(seq_len, d_model)
  3. Add them together: enhanced = embedding + PE
  4. Feed to transformer layers

Next Steps:
  - example_05: Complete transformer block (attention + FFN + norms)
  - example_06: Building a mini-GPT!
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 04")
print("=" * 70)

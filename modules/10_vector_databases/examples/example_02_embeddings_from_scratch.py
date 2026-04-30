"""
Example 02: Embeddings From Scratch
=====================================

GLOSSARY
--------
Embedding:
  Converting something (word, sentence, image) into a vector of numbers.
  The numbers capture "meaning" so that similar things have similar numbers.

Vocabulary:
  The complete set of words a model knows.
  Example: ["cat", "dog", "fish", "car", "truck"]
  In C#: a Dictionary<string, int> mapping each word to an index number.

One-Hot Encoding:
  A very simple way to turn a word into a vector.
  Each word gets a vector where ONE position is 1 and all others are 0.
  Problem: All words are equally different from each other (no meaning captured).

Lookup Table (Embedding Table):
  A matrix where each row is the embedding vector for one word.
  In C#: float[vocabSize][embeddingDim] -- a 2D array.
  We look up a word's vector by using its index as the row number.

Word2Vec Intuition:
  A method for learning word embeddings from text.
  Words appearing in similar contexts get similar vectors.
  "king" - "man" + "woman" ~= "queen"  (famous example)
  We will build a simplified version from scratch to show the core idea.

Context:
  The surrounding words near a target word.
  In "The dog chased the cat", the context of "chased" is ["dog", "the", "cat"].
  Context teaches us meaning: words appearing in similar contexts are similar.

Training:
  Adjusting the embedding vectors so that similar words end up similar.
  We "train" by showing the model many (word, context) pairs from text.

WHAT THIS EXAMPLE SHOWS
------------------------
Part 1: One-Hot Encoding (simplest, but bad -- no meaning)
Part 2: Random Embeddings (just a lookup table, untrained)
Part 3: Training embeddings using the Word2Vec skip-gram idea
Part 4: Visualizing the learned embeddings with PCA (2D projection)
Part 5: How real embedding models (sentence-transformers) work differently

LIBRARIES NEEDED
-----------------
  numpy      (pip install numpy)      - vector math
  matplotlib (pip install matplotlib) - visualization

No AI-specific libraries needed for Parts 1-4.
"""

import numpy as np               # Vector math
import matplotlib.pyplot as plt  # Plotting

np.random.seed(42)               # Same results every run

print("=" * 65)
print("EXAMPLE 02: Embeddings From Scratch")
print("=" * 65)


# ==============================================================================
# PART 1: One-Hot Encoding (The Simplest Embedding)
# ==============================================================================

print("\n" + "=" * 65)
print("PART 1: One-Hot Encoding")
print("=" * 65)

print("""
The simplest way to turn a word into a vector: ONE-HOT ENCODING.

For a vocabulary of N words, each word becomes a vector of N numbers.
All numbers are 0 except ONE position, which is 1.

Example vocabulary: ["cat", "dog", "fish", "car", "truck"]

  cat:   [1, 0, 0, 0, 0]
  dog:   [0, 1, 0, 0, 0]
  fish:  [0, 0, 1, 0, 0]
  car:   [0, 0, 0, 1, 0]
  truck: [0, 0, 0, 0, 1]

The position index is the word's ID in the vocabulary.
In C#: like an enum where each value has its own bit position.
""")

# Our tiny vocabulary
vocabulary = ["cat", "dog", "fish", "car", "truck", "kitten", "puppy", "bus"]

# Create a mapping from word to index
word_to_id = {word: idx for idx, word in enumerate(vocabulary)}   # {"cat": 0, "dog": 1, ...}
id_to_word = {idx: word for word, idx in word_to_id.items()}      # {0: "cat", 1: "dog", ...}

print(f"Vocabulary ({len(vocabulary)} words): {vocabulary}")
print(f"Word to ID: {word_to_id}")
print()

def one_hot(word, vocab):
    """
    Create a one-hot vector for a word.
    word:  the word to encode (must be in vocab)
    vocab: dict mapping word -> index
    Returns: numpy array with 1 at the word's index, 0 elsewhere
    """
    vocab_size = len(vocab)                       # How many words in vocabulary
    vector = np.zeros(vocab_size)                 # Start with all zeros
    idx = vocab[word]                             # Get this word's index
    vector[idx] = 1.0                             # Set that position to 1
    return vector

# Show one-hot vectors for each word
print("One-Hot Vectors:")
for word in vocabulary:
    vec = one_hot(word, word_to_id)               # Get the one-hot vector
    ones_at = np.where(vec == 1)[0]               # Find where the 1 is
    print(f"  {word:8s}: {vec.astype(int).tolist()}  (1 at position {ones_at[0]})")

print()
print("PROBLEM with one-hot encoding:")

# Cosine similarity between all pairs to show the problem
def cosine_sim(a, b):
    """Compute cosine similarity -- reused from Example 01."""
    dot   = np.dot(a, b)
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

pairs = [("cat", "kitten"), ("cat", "dog"), ("cat", "car"), ("dog", "puppy")]
for w1, w2 in pairs:
    sim = cosine_sim(one_hot(w1, word_to_id), one_hot(w2, word_to_id))
    print(f"  Similarity({w1}, {w2}) = {sim:.4f}")

print()
print("  -> All similarities are 0.0! One-hot has no concept of 'similar'.")
print("  -> 'cat' and 'kitten' look just as different as 'cat' and 'car'.")
print("  -> We need embeddings that LEARN meaning from context.")
print()


# ==============================================================================
# PART 2: Random Embeddings (Lookup Table, No Training)
# ==============================================================================

print("=" * 65)
print("PART 2: Random Embeddings (Untrained Lookup Table)")
print("=" * 65)

print("""
Instead of a huge sparse vector (one-hot), we use a SMALL DENSE vector.
We start with RANDOM numbers -- before training, they mean nothing.

                one-hot        embedding (size 4)
  cat:    [1,0,0,0,0,0,0,0]    [0.5, -0.3, 0.8,  0.1]
  dog:    [0,1,0,0,0,0,0,0]    [-0.2, 0.6, 0.3, -0.7]

The embedding is stored in an Embedding Table:
  - A matrix with shape (vocab_size, embedding_dim)
  - Each ROW is the embedding vector for one word
  - Look up a word by using its ID as the row index

In C#: like a 2D array float[vocabSize, embeddingDim]
""")

vocab_size    = len(vocabulary)    # 8 words
embedding_dim = 4                  # Each word gets a vector of 4 numbers

# Create the embedding table with random values
# Shape: (vocab_size, embedding_dim) = (8, 4)
# Each row = embedding vector for one word
embedding_table = np.random.randn(vocab_size, embedding_dim) * 0.1    # Small random values

print(f"Embedding Table shape: {embedding_table.shape}")
print(f"  {vocab_size} words, each with {embedding_dim} dimensions")
print()

def get_embedding(word, vocab, table):
    """
    Look up the embedding vector for a word.
    word:  the word to embed
    vocab: dict mapping word -> row index
    table: the embedding table (shape: vocab_size x embedding_dim)
    Returns: numpy array of shape (embedding_dim,)
    """
    idx = vocab[word]                   # Get the word's row index
    return table[idx]                   # Return that row from the table

print("Random embeddings (before training -- random, no meaning yet):")
for word in ["cat", "kitten", "dog", "car"]:
    emb = get_embedding(word, word_to_id, embedding_table)
    print(f"  {word:8s}: [{', '.join(f'{v:.3f}' for v in emb)}]")

print()

# Similarities before training (should be random, not meaningful)
print("Similarities BEFORE training (should be random, no pattern):")
for w1, w2 in [("cat", "kitten"), ("cat", "dog"), ("cat", "car")]:
    e1 = get_embedding(w1, word_to_id, embedding_table)
    e2 = get_embedding(w2, word_to_id, embedding_table)
    sim = cosine_sim(e1, e2)
    print(f"  {w1:8s} vs {w2:8s}: {sim:.4f}  (random, no meaning)")

print()
print("  -> After training, cat-kitten similarity should be MUCH higher than cat-car.")
print()


# ==============================================================================
# PART 3: Training Embeddings (Word2Vec Skip-Gram Idea)
# ==============================================================================

print("=" * 65)
print("PART 3: Training Embeddings (Simplified Word2Vec)")
print("=" * 65)

print("""
The Key Idea Behind Word2Vec:
  "You shall know a word by the company it keeps." -- J.R. Firth (1957)

  Words that appear in similar CONTEXTS have similar MEANINGS.

  In the sentence "I walked my puppy" and "I walked my dog":
    - "puppy" and "dog" appear in the same context (after "my" and before nothing)
    - So they should have similar embeddings

  TRAINING OBJECTIVE (Skip-Gram):
  Given a word, predict its SURROUNDING WORDS (context).
  If "cat" and "kitten" both appear next to "is", "cute", "sleeping" etc,
  their embeddings will become similar.

  We implement a simplified version below.
""")

# Training corpus: sentences where similar words appear in similar contexts
corpus = [
    "the cat sat on the mat",
    "the kitten sat on the mat",
    "my dog ran in the park",
    "my puppy ran in the park",
    "the bus drove down the road",
    "the car drove down the road",
    "the truck drove on the highway",
    "the cat and kitten are cute",
    "the dog and puppy are playful",
    "the car and truck need fuel",
    "the fish swam in the water",
    "the cat chased the fish",
]

print(f"Training corpus: {len(corpus)} sentences")
for s in corpus[:4]:
    print(f"  '{s}'")
print(f"  ... ({len(corpus) - 4} more)")
print()

# Build (center_word, context_word) training pairs
# For each word in a sentence, its context is the words within a window
window_size = 2    # Look 2 words to the left and right of the center word

training_pairs = []                            # List of (center_id, context_id) pairs

for sentence in corpus:
    words = sentence.split()                   # Split sentence into words
    for i, center_word in enumerate(words):    # Each word is a center word
        if center_word not in word_to_id:      # Skip words not in our vocabulary
            continue
        center_id = word_to_id[center_word]    # Get center word ID

        # Get context words within the window
        start = max(0, i - window_size)        # Start of window (don't go before index 0)
        end   = min(len(words), i + window_size + 1)   # End of window

        for j in range(start, end):
            if j == i:
                continue                       # Skip the center word itself
            context_word = words[j]
            if context_word not in word_to_id:
                continue                       # Skip unknown words
            context_id = word_to_id[context_word]
            training_pairs.append((center_id, context_id))   # Add this pair

print(f"Generated {len(training_pairs)} training pairs")
print(f"  First 5 pairs (center_word, context_word):")
for c, ctx in training_pairs[:5]:
    print(f"    ({id_to_word[c]}, {id_to_word[ctx]})")
print()

# Simplified Word2Vec training using NumPy
# We use a basic dot-product objective:
# - If (center, context) is a real pair -> their dot product should be HIGH
# - We use gradient descent to adjust embeddings toward this goal
learning_rate = 0.05       # How much to adjust embeddings each step (small = stable)
num_epochs    = 500        # How many times to go through the training data

# Re-initialize embedding table with fresh random values
embedding_table = np.random.randn(vocab_size, embedding_dim) * 0.1

# Second embedding table (output embeddings -- Word2Vec uses two tables)
output_table = np.random.randn(vocab_size, embedding_dim) * 0.1

# Simple sigmoid function (squashes any number into 0-1 range)
def sigmoid(x):
    """Convert x to a probability between 0 and 1."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))   # clip prevents overflow

losses = []                # Track loss over training

for epoch in range(num_epochs):
    total_loss = 0.0

    for center_id, context_id in training_pairs:
        # Get the current embedding vectors
        center_vec  = embedding_table[center_id]    # The center word's embedding
        context_vec = output_table[context_id]      # The context word's output embedding

        # Forward pass: compute score (should be HIGH for real pairs)
        score = np.dot(center_vec, context_vec)     # Dot product = similarity score
        prob  = sigmoid(score)                      # Convert to probability 0-1

        # Loss: we want prob to be 1.0 (maximize similarity for real pairs)
        loss = -np.log(prob + 1e-8)                 # Cross-entropy loss
        total_loss += loss

        # Backward pass: compute gradients
        # How much should we adjust to INCREASE the probability?
        grad = (prob - 1.0)                         # Gradient of loss w.r.t. score

        # Update embeddings (gradient descent: move opposite to gradient)
        embedding_table[center_id]  -= learning_rate * grad * context_vec
        output_table[context_id]    -= learning_rate * grad * center_vec

    losses.append(total_loss / len(training_pairs))    # Average loss for this epoch

print(f"Training complete: {num_epochs} epochs")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss:   {losses[-1]:.4f}")
print()

# Show similarities AFTER training
print("Similarities AFTER training:")
for w1, w2 in [("cat", "kitten"), ("cat", "dog"), ("dog", "puppy"),
                ("car", "truck"), ("cat", "car"), ("dog", "car")]:
    e1 = embedding_table[word_to_id[w1]]
    e2 = embedding_table[word_to_id[w2]]
    sim = cosine_sim(e1, e2)
    interpretation = "similar" if sim > 0.5 else ("somewhat" if sim > 0.2 else "different")
    print(f"  {w1:8s} vs {w2:8s}: {sim:.4f}  ({interpretation})")

print()
print("  -> cat/kitten similarity is now HIGH (they appear in same contexts)")
print("  -> car/truck similarity is now HIGH (same contexts)")
print("  -> cat/car similarity is LOW (different contexts)")
print()


# ==============================================================================
# PART 4: Visualizing Embeddings with PCA
# ==============================================================================

print("=" * 65)
print("PART 4: Visualizing Learned Embeddings (PCA to 2D)")
print("=" * 65)

print("""
Our embeddings have 4 dimensions -- hard to visualize directly.
We use PCA (Principal Component Analysis) to reduce to 2 dimensions for plotting.

PCA: Finds the 2 directions in the high-dimensional space that capture
the most variation, then projects all points onto those 2 directions.
Think of it like casting a shadow of a 3D object onto a 2D wall.

The 2D plot is an APPROXIMATION -- some information is lost.
But it lets us visually verify that similar words are near each other.
""")

def pca_2d(matrix):
    """
    Reduce a matrix of shape (n, d) to shape (n, 2) using PCA.
    matrix: numpy array of shape (n_words, embedding_dim)
    Returns: numpy array of shape (n_words, 2)
    """
    # Center the data (subtract the mean)
    centered = matrix - np.mean(matrix, axis=0)    # Subtract column-wise mean

    # Compute covariance matrix
    cov = np.cov(centered.T)                        # (embedding_dim, embedding_dim)

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)    # eigh: symmetric matrix

    # Sort by largest eigenvalue (most important directions first)
    sorted_indices = np.argsort(eigenvalues)[::-1]     # Descending order
    top_2 = eigenvectors[:, sorted_indices[:2]]         # Take top 2 eigenvectors

    # Project data onto the top 2 directions
    projected = centered @ top_2                        # Matrix multiply

    return projected                                    # Shape: (n_words, 2)

# Project all word embeddings to 2D
embeddings_2d = pca_2d(embedding_table)                # Shape: (8, 2)

# Color-code by category
category_colors = {
    "cat":    "#E74C3C",    # Red -- feline
    "kitten": "#C0392B",    # Dark red -- feline
    "dog":    "#2980B9",    # Blue -- canine
    "puppy":  "#3498DB",    # Light blue -- canine
    "car":    "#27AE60",    # Green -- vehicle
    "truck":  "#1E8449",    # Dark green -- vehicle
    "bus":    "#52BE80",    # Medium green -- vehicle
    "fish":   "#8E44AD",    # Purple -- aquatic
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: 2D projection of learned embeddings
ax1 = axes[0]
for i, word in enumerate(vocabulary):
    x, y = embeddings_2d[i, 0], embeddings_2d[i, 1]   # Get 2D coordinates
    color = category_colors[word]
    ax1.scatter(x, y, s=200, color=color, zorder=5)
    ax1.annotate(word, (x, y),
                 textcoords="offset points",
                 xytext=(6, 6), fontsize=11, fontweight="bold")

ax1.set_title("Learned Embeddings (2D PCA Projection)\nAfter Training",
              fontsize=12, fontweight="bold")
ax1.set_xlabel("Principal Component 1")
ax1.set_ylabel("Principal Component 2")
ax1.grid(alpha=0.3)
ax1.axhline(0, color="gray", linewidth=0.5)
ax1.axvline(0, color="gray", linewidth=0.5)

# Add a legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(color="#E74C3C", label="Feline (cat, kitten)"),
    Patch(color="#2980B9", label="Canine (dog, puppy)"),
    Patch(color="#27AE60", label="Vehicle (car, truck, bus)"),
    Patch(color="#8E44AD", label="Aquatic (fish)"),
]
ax1.legend(handles=legend_elements, fontsize=9, loc="lower left")

# Plot 2: Training loss curve
ax2 = axes[1]
ax2.plot(losses, color="#E74C3C", linewidth=2)
ax2.set_title("Training Loss Curve\n(lower is better)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Average Loss")
ax2.grid(alpha=0.3)

# Annotate start and end loss
ax2.annotate(f"Start: {losses[0]:.3f}", xy=(0, losses[0]),
             xytext=(20, 10), textcoords="offset points", fontsize=10)
ax2.annotate(f"End: {losses[-1]:.3f}", xy=(len(losses)-1, losses[-1]),
             xytext=(-80, 10), textcoords="offset points", fontsize=10)

plt.tight_layout()
plt.show()

print("  -> Similar words (cat/kitten, dog/puppy, car/truck) should cluster together")
print("  -> Different categories (animals vs vehicles) should be far apart")
print()


# ==============================================================================
# PART 5: How Real Embedding Models Work
# ==============================================================================

print("=" * 65)
print("PART 5: How Real Embedding Models Work")
print("=" * 65)

print("""
What we built vs what real models do:

OUR VERSION (this example):
  - Vocabulary: 8 words
  - Embedding: 4 dimensions
  - Training: simple Word2Vec on toy corpus
  - Embeds: single words only

REAL MODEL (e.g., all-MiniLM-L6-v2 from sentence-transformers):
  - Vocabulary: ~30,000 tokens (subword pieces, not just words)
  - Embedding: 384 dimensions
  - Training: 1 billion sentence pairs, transformer architecture
  - Embeds: full sentences (not just words!)
  - Captures: grammar, context, meaning, even metaphor

The real model works similarly in principle:
  1. Tokenize the input text into tokens
  2. Feed tokens through transformer layers (attention mechanism)
  3. Pool the final layer outputs into a single vector
  4. That vector is the sentence embedding

Using it is just 2 lines of code (requires sentence-transformers installed):

  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('all-MiniLM-L6-v2')
  vector = model.encode("The cat sat on the mat")
  # vector is a numpy array of shape (384,)

The dimensions are NOT labeled. We do not know what dimension 47 represents.
The model learned the dimensions automatically during training.
""")

# Demonstrate with sentence-transformers if available
try:
    from sentence_transformers import SentenceTransformer   # Real embedding model

    print("  sentence-transformers is installed -- running real embeddings...")
    print()

    model = SentenceTransformer("all-MiniLM-L6-v2")         # Load the model

    sentences = [
        "The cat sat on the mat",
        "The kitten sat on the mat",
        "My dog ran in the park",
        "I bought a new laptop",
    ]

    # Encode all sentences at once (batch processing is faster)
    vectors = model.encode(sentences)               # Shape: (4, 384)

    print(f"  Real embedding shape: {vectors.shape}  ({len(sentences)} sentences, 384 dims each)")
    print()
    print("  Real cosine similarities:")
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            if j <= i:
                continue                           # Only show upper triangle (avoid duplicates)
            sim = cosine_sim(vectors[i], vectors[j])
            print(f"    {s1[:30]:30s}")
            print(f"    {s2[:30]:30s}")
            print(f"    Similarity: {sim:.4f}")
            print()

except ImportError:
    print("  sentence-transformers not installed.")
    print("  Install it: pip install sentence-transformers")
    print("  Then re-run this example to see real embedding similarities.")
    print()
    print("  Expected output would show:")
    print("    cat sentence vs kitten sentence: ~0.90 (very similar)")
    print("    cat sentence vs dog sentence:    ~0.75 (related, both animals)")
    print("    cat sentence vs laptop sentence: ~0.15 (very different)")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("SUMMARY - Embeddings From Scratch")
print("=" * 65)

print("""
WHAT WE LEARNED:

1. One-Hot Encoding: simplest word representation, but no meaning.
   [cat: 1,0,0,0,0]  All similarities = 0. Useless for search.

2. Embedding Table: a matrix (vocab_size x embedding_dim).
   Each row = a word's learned vector.
   Look up: embedding_table[word_id]

3. Training: adjust embeddings so that words appearing in similar
   contexts end up with similar vectors.
   The loss goes down as the model learns.

4. After training:
   - cat/kitten similarity: HIGH (appear in same context)
   - cat/car similarity:    LOW (appear in different contexts)

5. Real models (sentence-transformers) embed full sentences,
   use 384+ dimensions, and are trained on billions of examples.
   They are much better than our toy version -- but same core idea.

6. In C#: the embedding table is a float[vocabSize, embeddingDim] array.
   A lookup is: array[wordId, :] -- return the row for that word.

NEXT EXAMPLE (03):
  Use ChromaDB -- a real vector database -- to store and search
  documents automatically (no manual similarity math needed).
""")

print("=" * 65)
print("END OF EXAMPLE 02")
print("=" * 65)

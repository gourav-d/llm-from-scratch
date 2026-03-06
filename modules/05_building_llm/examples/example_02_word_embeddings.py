"""
Word Embeddings - Tokens Become Meaningful Vectors!

This example demonstrates how words are represented as dense vectors
that capture semantic meaning.

What you'll see:
1. One-hot encoding and its limitations
2. Dense embeddings
3. Building an embedding layer
4. Calculating word similarities
5. Word analogies (king - man + woman = queen!)
6. Visualizing embeddings in 2D
7. How embeddings are used in transformers
8. Training embeddings
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

print("="*70)
print("WORD EMBEDDINGS - Giving Meaning to Tokens!")
print("="*70)

# ==============================================================================
# REAL-WORLD ANALOGY: Understanding Embeddings
# ==============================================================================

print("\n" + "="*70)
print("REAL-WORLD ANALOGY: Understanding Embeddings")
print("="*70)

print("""
Imagine you're organizing a music library and want to represent songs as numbers
so a computer can understand which songs are similar.

OPTION 1: ONE-HOT ENCODING (Bad!)
==================================
Each song gets a unique ID:
- "Bohemian Rhapsody" = [1, 0, 0, 0, 0, ...]  (10,000 zeros!)
- "Don't Stop Me Now" = [0, 1, 0, 0, 0, ...]
- "Stairway to Heaven" = [0, 0, 1, 0, 0, ...]

Problems:
✗ Every song is equally different from every other (no similarity!)
✗ Huge vectors (if you have 1M songs, vectors are 1M-dimensional!)
✗ No meaning - can't tell rock from classical

OPTION 2: DENSE EMBEDDINGS (Good!)
===================================
Each song gets a compact vector of features:
- "Bohemian Rhapsody" = [rock: 0.9, energy: 0.8, tempo: 0.6, vocals: 0.9, ...]
- "Don't Stop Me Now" = [rock: 0.9, energy: 0.95, tempo: 0.8, vocals: 0.85, ...]
- "Stairway to Heaven" = [rock: 0.95, energy: 0.7, tempo: 0.5, vocals: 0.8, ...]

Benefits:
✓ Compact (maybe only 128 dimensions instead of 1M!)
✓ Similar songs have similar vectors
✓ Can do math: "Rock ballad" - "slow" + "fast" ≈ "Rock anthem"
✓ Captures meaning and relationships

THIS is what word embeddings do for language!

HOW EMBEDDINGS WORK IN GPT:
1. Token "cat" (ID: 156) → Lookup in embedding table → [0.25, -0.1, 0.8, ...]
2. Token "dog" (ID: 189) → Lookup in embedding table → [0.22, -0.08, 0.75, ...]
3. Vectors are similar because "cat" and "dog" are semantically related!
4. The embedding table is LEARNED during training through backpropagation

FAMOUS EXAMPLE:
king - man + woman = queen
[0.5, 0.3, ...] - [0.2, 0.1, ...] + [0.25, 0.12, ...] ≈ [0.55, 0.32, ...]
(The math actually works in embedding space!)
""")

print("\n" + "="*70)
print("Now let's see this in action with Python code!")
print("="*70)

# ==============================================================================
# EXAMPLE 1: One-Hot Encoding (The Naive Approach)
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: One-Hot Encoding - The Naive Approach")
print("="*70)

print("""
One-hot encoding represents each word as a vector with:
- All zeros except one position
- That position is 1 (hence "one-hot")
- Vector size = vocabulary size

Problems:
- No notion of similarity (all words equally different!)
- Huge dimensionality (10k vocab = 10k dimensions!)
- Sparse (99.99% zeros = wasted memory)
""")

def create_one_hot(vocab, word):
    """Create one-hot vector for a word"""
    vec = np.zeros(len(vocab))
    if word in vocab:
        idx = vocab.index(word)
        vec[idx] = 1
    return vec

# Small vocabulary
vocab = ["cat", "dog", "bird", "fish", "python", "java", "rust"]

# Create one-hot vectors
cat_vec = create_one_hot(vocab, "cat")
dog_vec = create_one_hot(vocab, "dog")
python_vec = create_one_hot(vocab, "python")

print(f"Vocabulary: {vocab}")
print(f"Vocabulary size: {len(vocab)}")
print()

print(f"One-hot vector for 'cat':    {cat_vec}")
print(f"One-hot vector for 'dog':    {dog_vec}")
print(f"One-hot vector for 'python': {python_vec}")

# Calculate similarity (dot product)
similarity_cat_dog = np.dot(cat_vec, dog_vec)
similarity_cat_python = np.dot(cat_vec, python_vec)

print(f"\nSimilarity between 'cat' and 'dog':    {similarity_cat_dog}")
print(f"Similarity between 'cat' and 'python': {similarity_cat_python}")
print("\n⚠️  Both are 0! One-hot encoding shows NO similarity!")
print("But we know 'cat' and 'dog' are more similar than 'cat' and 'python'!")

print()

# ==============================================================================
# EXAMPLE 2: Dense Embeddings
# ==============================================================================

print("="*70)
print("EXAMPLE 2: Dense Embeddings - The Better Way")
print("="*70)

print("""
Dense embeddings represent words as small, dense vectors:
- Much smaller dimension (50-768 typical)
- Real numbers (not just 0 and 1)
- Similar words have similar vectors
- Learned during training (not hand-crafted!)
""")

class EmbeddingLayer:
    """
    Simple embedding layer (like nn.Embedding in PyTorch)

    C# Analogy:
    Like Dictionary<int, float[]> - maps token ID to vector
    But implemented as 2D array for efficiency: embedding_table[token_id]
    """

    def __init__(self, vocab_size, embed_dim):
        """
        Initialize embedding layer

        Args:
            vocab_size: Number of unique tokens
            embed_dim: Dimension of embedding vectors
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Initialize with small random values
        # In real models, this is learned through backpropagation!
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.1

        print(f"Created embedding layer:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Embedding dimension: {embed_dim}")
        print(f"  Embedding table shape: {self.embeddings.shape}")

    def forward(self, token_ids):
        """
        Look up embeddings for token IDs

        Args:
            token_ids: Array of token IDs

        Returns:
            Array of embedding vectors
        """
        # Simple lookup in embedding table
        # C#: float[] embedding = embeddingTable[tokenId];
        return self.embeddings[token_ids]

    def get_embedding(self, token_id):
        """Get single embedding vector"""
        return self.embeddings[token_id]

# Create embedding layer
vocab_size = len(vocab)
embed_dim = 5  # Small dimension for visualization

embedding_layer = EmbeddingLayer(vocab_size, embed_dim)

# Get embeddings for words
cat_id = vocab.index("cat")
dog_id = vocab.index("dog")
python_id = vocab.index("python")

cat_emb = embedding_layer.get_embedding(cat_id)
dog_emb = embedding_layer.get_embedding(dog_id)
python_emb = embedding_layer.get_embedding(python_id)

print(f"\nDense embeddings (random initialization):")
print(f"'cat':    {cat_emb}")
print(f"'dog':    {dog_emb}")
print(f"'python': {python_emb}")

print(f"\n✓ Compact! Only {embed_dim} dimensions (vs {vocab_size} for one-hot)")

print()

# ==============================================================================
# EXAMPLE 3: Calculating Word Similarity
# ==============================================================================

print("="*70)
print("EXAMPLE 3: Calculating Word Similarity")
print("="*70)

print("""
Cosine similarity measures how similar two vectors are:
- Ranges from -1 (opposite) to 1 (identical)
- 0 means orthogonal (unrelated)
- Formula: cos(θ) = (A · B) / (||A|| ||B||)
""")

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors

    C# Analogy:
    Like comparing two vectors using dot product and magnitudes
    Similar to Vector3.Dot(a, b) / (a.magnitude * b.magnitude) in Unity
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

# Calculate similarities
sim_cat_dog = cosine_similarity(cat_emb, dog_emb)
sim_cat_python = cosine_similarity(cat_emb, python_emb)
sim_dog_python = cosine_similarity(dog_emb, python_emb)

print(f"Cosine similarities:")
print(f"  'cat' <-> 'dog':    {sim_cat_dog:.3f}")
print(f"  'cat' <-> 'python': {sim_cat_python:.3f}")
print(f"  'dog' <-> 'python': {sim_dog_python:.3f}")

print(f"\nWith dense embeddings, we CAN measure similarity!")
print("(Though these are random - trained embeddings work better)")

print()

# ==============================================================================
# EXAMPLE 4: Trained Embeddings (Simulated)
# ==============================================================================

print("="*70)
print("EXAMPLE 4: Trained Embeddings (Simulated)")
print("="*70)

print("""
In real LLMs, embeddings are LEARNED through training.
Let's simulate what trained embeddings might look like!
""")

# Create "trained" embeddings that capture semantic meaning
# We'll hand-craft these to demonstrate the concept
vocab_semantic = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"]

# Simulated embeddings with meaningful dimensions:
# [royalty, gender_male, age_adult, ...]
embeddings_semantic = {
    "king":     np.array([0.9,  0.9,  0.9]),  # [royal, male, adult]
    "queen":    np.array([0.9, -0.9,  0.9]),  # [royal, female, adult]
    "prince":   np.array([0.8,  0.8,  0.3]),  # [royal, male, young]
    "princess": np.array([0.8, -0.8,  0.3]),  # [royal, female, young]
    "man":      np.array([0.0,  0.9,  0.9]),  # [common, male, adult]
    "woman":    np.array([0.0, -0.9,  0.9]),  # [common, female, adult]
    "boy":      np.array([0.0,  0.8,  0.3]),  # [common, male, young]
    "girl":     np.array([0.0, -0.8,  0.3]),  # [common, female, young]
}

print("Simulated trained embeddings:")
print("Dimensions: [royalty, gender_male, age_adult]")
print()

for word, vec in embeddings_semantic.items():
    print(f"{word:10s}: {vec}")

# Calculate similarities
print("\nSimilarities:")
words_to_compare = [("king", "queen"), ("king", "man"), ("queen", "woman"),
                    ("prince", "princess"), ("boy", "girl")]

for word1, word2 in words_to_compare:
    sim = cosine_similarity(embeddings_semantic[word1], embeddings_semantic[word2])
    print(f"  {word1:10s} <-> {word2:10s}: {sim:.3f}")

print()

# ==============================================================================
# EXAMPLE 5: Word Analogies (king - man + woman = queen)
# ==============================================================================

print("="*70)
print("EXAMPLE 5: Word Analogies - The Magic of Embeddings!")
print("="*70)

print("""
The famous word analogy:
"king is to man as queen is to ?"

In embedding space:
king - man + woman ≈ queen

This works because embeddings capture RELATIONSHIPS!
""")

def find_closest_word(target_vec, word_embeddings, exclude_words=[]):
    """Find the word whose embedding is closest to target vector"""
    best_word = None
    best_sim = -999

    for word, vec in word_embeddings.items():
        if word in exclude_words:
            continue

        sim = cosine_similarity(target_vec, vec)
        if sim > best_sim:
            best_sim = sim
            best_word = word

    return best_word, best_sim

# Perform the analogy
print("Analogy: king - man + woman = ?")
print()

king = embeddings_semantic["king"]
man = embeddings_semantic["man"]
woman = embeddings_semantic["woman"]

result_vec = king - man + woman

print(f"king:   {king}")
print(f"man:    {man}")
print(f"woman:  {woman}")
print(f"Result: {result_vec}")

# Find closest word
closest, similarity = find_closest_word(result_vec, embeddings_semantic,
                                        exclude_words=["king", "man", "woman"])

print(f"\nClosest word: '{closest}' (similarity: {similarity:.3f})")
print(f"✓ It works! king - man + woman ≈ {closest}")

# Try more analogies
print("\nMore analogies:")
analogies = [
    ("prince", "boy", "girl", "princess"),
    ("king", "prince", "princess", "queen"),
]

for a, b, c, expected in analogies:
    result_vec = embeddings_semantic[a] - embeddings_semantic[b] + embeddings_semantic[c]
    closest, sim = find_closest_word(result_vec, embeddings_semantic, [a, b, c])
    print(f"{a} - {b} + {c} = {closest} (expected: {expected})")

print()

# ==============================================================================
# EXAMPLE 6: Visualizing Embeddings in 2D
# ==============================================================================

print("="*70)
print("EXAMPLE 6: Visualizing Embeddings in 2D")
print("="*70)

print("""
Embeddings are high-dimensional, but we can project to 2D for visualization!
We'll use PCA (Principal Component Analysis) to reduce dimensions.
""")

# Prepare data for visualization
words = list(embeddings_semantic.keys())
vectors = np.array([embeddings_semantic[w] for w in words])

# Reduce to 2D using PCA
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, alpha=0.6)

# Add labels
for i, word in enumerate(words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                fontsize=12, ha='center')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Word Embeddings Visualized in 2D Space')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('word_embeddings_2d.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to 'word_embeddings_2d.png'")

print("\nWhat you should see:")
print("  - Similar words cluster together")
print("  - Gender relationships preserved (king-queen, man-woman)")
print("  - Royalty vs common people separated")

print()

# ==============================================================================
# EXAMPLE 7: Embedding Layer Forward Pass
# ==============================================================================

print("="*70)
print("EXAMPLE 7: Embedding Layer in Action (Like in GPT)")
print("="*70)

print("""
In transformers, the embedding layer is the FIRST operation:
1. Input: Token IDs [42, 156, 89]
2. Embedding Layer: Look up each ID in embedding table
3. Output: Dense vectors ready for attention mechanism!
""")

# Simulate a sentence
vocab_full = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "the", "cat", "sat", "on", "mat"]
sentence_tokens = [2, 4, 5, 6, 7, 4, 8, 3]  # <BOS> the cat sat on the mat <EOS>

# Create embedding layer
embed_layer = EmbeddingLayer(vocab_size=len(vocab_full), embed_dim=8)

# Forward pass
embeddings = embed_layer.forward(sentence_tokens)

print(f"Sentence tokens: {sentence_tokens}")
print(f"Decoded: {[vocab_full[i] for i in sentence_tokens]}")
print(f"\nEmbedding shape: {embeddings.shape}")
print(f"  → {embeddings.shape[0]} tokens")
print(f"  → {embeddings.shape[1]} dimensions per token")

print(f"\nFirst token embedding (<BOS>):")
print(embeddings[0])

print(f"\nSecond token embedding ('the'):")
print(embeddings[1])

print("\nThese embeddings are fed into the transformer!")

print()

# ==============================================================================
# EXAMPLE 8: How Embeddings Are Learned
# ==============================================================================

print("="*70)
print("EXAMPLE 8: How Embeddings Are Learned")
print("="*70)

print("""
Embeddings are NOT hand-crafted! They're learned through training:

1. Initialize randomly:
   embedding_table = random_normal(vocab_size, embed_dim) * 0.01

2. Forward pass:
   - Look up embeddings for input tokens
   - Pass through transformer
   - Get predictions

3. Calculate loss:
   - Compare predictions to targets
   - loss = cross_entropy(predictions, targets)

4. Backpropagation:
   - Gradients flow back through transformer
   - Gradients reach embedding table!
   - embedding_table -= learning_rate * gradients

5. Repeat for millions of examples
   - Embeddings gradually learn to capture meaning
   - Similar words end up with similar vectors
   - Relationships like "king-queen" emerge automatically!

Key insight:
The model learns embeddings BY USING THEM for language modeling!
- "The cat sat on the ___" → model predicts "mat"
- If embeddings for "cat", "sat", "on" help predict "mat",
  they get reinforced (gradient descent)
- Over millions of examples, semantic patterns emerge!
""")

print()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("✅ SUMMARY")
print("="*70)

print("""
What You Just Built:
1. ✅ One-hot encoding (and why it's bad)
2. ✅ Dense embeddings
3. ✅ Embedding layer from scratch
4. ✅ Cosine similarity for word relationships
5. ✅ Word analogies (king - man + woman = queen)
6. ✅ 2D visualization of embeddings
7. ✅ Forward pass through embedding layer
8. ✅ Understanding how embeddings are learned

Key Insights:
- One-hot: Sparse, huge, no semantics
- Dense embeddings: Compact, meaningful, learned
- Similar words → Similar vectors
- Embeddings capture relationships
- Learned through backpropagation
- First layer in transformers!

Connection to GPT:
GPT-3 embeddings:
- Vocabulary: ~50,000 tokens
- Embedding dimension: 12,288
- Embedding table: 50k × 12k = 614 million parameters!
- That's just the FIRST layer!

Math That Works:
king - man + woman = queen
- This actually works in real embeddings!
- Shows embeddings capture semantic relationships
- Not hand-coded - emerges from training!

From Tokens to Transformers:
Text → Tokenization → Token IDs → Embeddings → Attention → Output
  ↑                    ↑           ↑
Module 5.1         Module 5.1   Module 5.2 (You are here!)
                                    ↓
                                Module 4 (Transformers use these!)

Next Steps:
1. Complete exercise_02_word_embeddings.py
2. Explore: Train simple Word2Vec model
3. Study: How GPT-3 embeddings differ from Word2Vec

You understand how tokens become meaningful vectors! 🎉
""")

print("="*70)
print("Complete 'exercise_02_word_embeddings.py' to practice!")
print("="*70)

# Show plots
try:
    plt.show()
except:
    print("\n(Close plot window to continue)")

"""
Exercise 2: Word Embeddings Practice

Complete these exercises to practice building and using word embeddings!
Solutions are provided at the bottom (commented out).
"""

import numpy as np

# ==============================================================================
# Exercise 1: Implement One-Hot Encoding
# ==============================================================================

print("="*70)
print("Exercise 1: Implement One-Hot Encoding")
print("="*70)

"""
Implement a function that converts token IDs to one-hot encoded vectors.

Task:
1. Create a zero vector of length vocab_size
2. Set the position corresponding to token_id to 1
3. Return the one-hot vector

Hint: Use np.zeros() to create the vector
"""

# TODO: Implement one_hot_encode function
def one_hot_encode(token_id, vocab_size):
    """
    Convert token ID to one-hot vector

    Args:
        token_id: Integer token ID
        vocab_size: Size of vocabulary

    Returns:
        One-hot encoded vector (numpy array)
    """
    pass

# Test your function
# vocab_size = 10
# token_id = 3
# one_hot = one_hot_encode(token_id, vocab_size)
# print(f"Token ID {token_id} with vocab_size {vocab_size}:")
# print(f"One-hot: {one_hot}")
# print(f"Expected: [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]")

# ==============================================================================
# Exercise 2: Build an Embedding Lookup Table
# ==============================================================================

print("\n" + "="*70)
print("Exercise 2: Build Embedding Lookup Table")
print("="*70)

"""
Create an embedding layer that:
1. Initializes a random embedding matrix
2. Looks up embeddings for given token IDs
3. Returns embedding vectors

Hint: Use np.random.randn() for initialization
"""

# TODO: Implement EmbeddingLayer class
class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize embedding layer

        Args:
            vocab_size: Number of unique tokens
            embedding_dim: Size of embedding vectors
        """
        # TODO: Initialize embedding matrix
        # Shape should be (vocab_size, embedding_dim)
        # Use small random values (multiply by 0.01)
        pass

    def forward(self, token_ids):
        """
        Look up embeddings for token IDs

        Args:
            token_ids: Array or list of token IDs

        Returns:
            Embeddings for the tokens
        """
        # TODO: Return embeddings for token_ids
        # Hint: Just index into self.embeddings
        pass

    def __call__(self, token_ids):
        return self.forward(token_ids)

# Test your embedding layer
# vocab_size = 1000
# embedding_dim = 50
#
# emb_layer = EmbeddingLayer(vocab_size, embedding_dim)
#
# # Single token
# token_id = 42
# embedding = emb_layer(token_id)
# print(f"Embedding shape for single token: {embedding.shape}")
# print(f"Expected: (50,)")
#
# # Multiple tokens
# token_ids = np.array([5, 42, 17])
# embeddings = emb_layer(token_ids)
# print(f"Embeddings shape for 3 tokens: {embeddings.shape}")
# print(f"Expected: (3, 50)")

# ==============================================================================
# Exercise 3: Calculate Cosine Similarity
# ==============================================================================

print("\n" + "="*70)
print("Exercise 3: Calculate Cosine Similarity")
print("="*70)

"""
Implement cosine similarity to measure how similar two vectors are.

Formula: similarity = (A · B) / (||A|| × ||B||)

Where:
- A · B is the dot product
- ||A|| is the magnitude (norm) of A
- ||B|| is the magnitude (norm) of B

Hint: Use np.dot() and np.linalg.norm()
"""

# TODO: Implement cosine_similarity function
def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score between -1 and 1
    """
    pass

# Test your function
# vec1 = np.array([1.0, 2.0, 3.0])
# vec2 = np.array([2.0, 4.0, 6.0])  # Same direction as vec1
# vec3 = np.array([-1.0, -2.0, -3.0])  # Opposite direction
# vec4 = np.array([1.0, 0.0, 0.0])  # Perpendicular
#
# sim_12 = cosine_similarity(vec1, vec2)
# sim_13 = cosine_similarity(vec1, vec3)
# sim_14 = cosine_similarity(vec1, vec4)
#
# print(f"Similarity between vec1 and vec2 (same direction): {sim_12:.3f}")
# print(f"Expected: ~1.000")
# print(f"Similarity between vec1 and vec3 (opposite): {sim_13:.3f}")
# print(f"Expected: ~-1.000")
# print(f"Similarity between vec1 and vec4 (perpendicular): {sim_14:.3f}")
# print(f"Expected: ~0.267")

# ==============================================================================
# Exercise 4: Find Nearest Neighbors
# ==============================================================================

print("\n" + "="*70)
print("Exercise 4: Find Nearest Neighbors")
print("="*70)

"""
Given a query embedding and a set of word embeddings, find the k most
similar words using cosine similarity.

Steps:
1. Calculate similarity between query and each word
2. Sort by similarity (descending)
3. Return top k words with their similarities

Hint: Use your cosine_similarity function from Exercise 3
"""

# TODO: Implement find_nearest_neighbors function
def find_nearest_neighbors(query_embedding, embeddings, words, k=5):
    """
    Find k most similar words to query

    Args:
        query_embedding: Embedding vector of query word
        embeddings: Matrix of all embeddings (shape: num_words × embedding_dim)
        words: List of words corresponding to embeddings
        k: Number of neighbors to return

    Returns:
        List of (word, similarity) tuples, sorted by similarity
    """
    pass

# Test your function
# words = ["king", "queen", "man", "woman", "prince", "princess", "cat", "dog"]
#
# # Create simple 3D embeddings (normally would be 300D!)
# embeddings_dict = {
#     "king":     np.array([0.9, 0.1, 0.5]),
#     "queen":    np.array([0.9, 0.9, 0.5]),
#     "man":      np.array([0.5, 0.1, 0.3]),
#     "woman":    np.array([0.5, 0.9, 0.3]),
#     "prince":   np.array([0.8, 0.1, 0.6]),
#     "princess": np.array([0.8, 0.9, 0.6]),
#     "cat":      np.array([0.2, 0.5, 0.9]),
#     "dog":      np.array([0.3, 0.5, 0.8]),
# }
#
# # Convert to matrix
# embeddings = np.array([embeddings_dict[word] for word in words])
#
# # Find words similar to "king"
# query_word = "king"
# query_embedding = embeddings_dict[query_word]
#
# neighbors = find_nearest_neighbors(query_embedding, embeddings, words, k=4)
#
# print(f"Words most similar to '{query_word}':")
# for word, similarity in neighbors:
#     print(f"  {word:12} {similarity:.3f}")
#
# print("\nExpected: king (1.000), prince, queen, man")

# ==============================================================================
# BONUS Exercise 5: Visualize Embeddings in 2D
# ==============================================================================

print("\n" + "="*70)
print("BONUS Exercise 5: Visualize 2D Embeddings")
print("="*70)

"""
Create a simple 2D visualization of word embeddings.

Tasks:
1. Create 2D embeddings for a set of words
2. Plot them using matplotlib
3. Observe clustering of related words

Note: Real embeddings are 100-1000 dimensions, but we use 2D for visualization!
"""

# TODO: Implement visualize_embeddings function
def visualize_embeddings(embeddings_dict):
    """
    Visualize 2D word embeddings

    Args:
        embeddings_dict: Dictionary mapping words to 2D embeddings
                        Example: {"cat": np.array([0.8, 0.6]), ...}
    """
    # TODO: Import matplotlib
    # TODO: Create scatter plot
    # TODO: Add word labels
    # TODO: Show plot
    pass

# Test your visualization
# import matplotlib.pyplot as plt
#
# # Create 2D embeddings (normally would use dimensionality reduction!)
# embeddings_2d = {
#     # Animals
#     "cat":      np.array([0.8, 0.6]),
#     "dog":      np.array([0.7, 0.7]),
#     "tiger":    np.array([0.9, 0.5]),
#     "lion":     np.array([0.85, 0.55]),
#
#     # Vehicles
#     "car":      np.array([0.2, 0.9]),
#     "truck":    np.array([0.3, 0.85]),
#     "bus":      np.array([0.25, 0.95]),
#
#     # Royalty
#     "king":     np.array([0.5, 0.2]),
#     "queen":    np.array([0.55, 0.25]),
#     "prince":   np.array([0.45, 0.15]),
# }
#
# visualize_embeddings(embeddings_2d)
# print("You should see three clusters: animals, vehicles, and royalty!")

# ==============================================================================
# SOLUTIONS
# ==============================================================================

print("\n" + "="*70)
print("Solutions are below (scroll down)")
print("="*70)

"""
# SOLUTION TO EXERCISE 1: One-Hot Encoding

def one_hot_encode(token_id, vocab_size):
    # Create zero vector
    one_hot = np.zeros(vocab_size)

    # Set position to 1
    one_hot[token_id] = 1

    return one_hot

# Test
vocab_size = 10
token_id = 3
one_hot = one_hot_encode(token_id, vocab_size)
print(f"Token ID {token_id} with vocab_size {vocab_size}:")
print(f"One-hot: {one_hot}")
print(f"Expected: [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]")

# Batch version
def one_hot_encode_batch(token_ids, vocab_size):
    batch_size = len(token_ids)
    one_hot_matrix = np.zeros((batch_size, vocab_size))

    for i, token_id in enumerate(token_ids):
        one_hot_matrix[i, token_id] = 1

    return one_hot_matrix

# Test batch
token_ids = [3, 7, 1]
one_hot_batch = one_hot_encode_batch(token_ids, vocab_size)
print(f"\\nBatch one-hot encoding:")
print(one_hot_batch)


# SOLUTION TO EXERCISE 2: Embedding Lookup Table

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        # Initialize random embeddings
        # Shape: (vocab_size, embedding_dim)
        # Small random values to avoid saturation
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward(self, token_ids):
        # Look up embeddings
        # Works for single ID or array of IDs
        return self.embeddings[token_ids]

    def __call__(self, token_ids):
        return self.forward(token_ids)

# Test
vocab_size = 1000
embedding_dim = 50

emb_layer = EmbeddingLayer(vocab_size, embedding_dim)

# Single token
token_id = 42
embedding = emb_layer(token_id)
print(f"\\nEmbedding shape for single token: {embedding.shape}")
print(f"Expected: (50,)")
print(f"First 5 values: {embedding[:5]}")

# Multiple tokens
token_ids = np.array([5, 42, 17])
embeddings = emb_layer(token_ids)
print(f"\\nEmbeddings shape for 3 tokens: {embeddings.shape}")
print(f"Expected: (3, 50)")

# Show that same token always gets same embedding
embedding_again = emb_layer(42)
print(f"\\nSame token gives same embedding: {np.allclose(embedding, embedding_again)}")


# SOLUTION TO EXERCISE 3: Cosine Similarity

def cosine_similarity(vec1, vec2):
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate magnitudes (norms)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity

# Test
vec1 = np.array([1.0, 2.0, 3.0])
vec2 = np.array([2.0, 4.0, 6.0])  # Same direction as vec1
vec3 = np.array([-1.0, -2.0, -3.0])  # Opposite direction
vec4 = np.array([1.0, 0.0, 0.0])  # Perpendicular-ish

sim_12 = cosine_similarity(vec1, vec2)
sim_13 = cosine_similarity(vec1, vec3)
sim_14 = cosine_similarity(vec1, vec4)

print(f"\\nSimilarity between vec1 and vec2 (same direction): {sim_12:.3f}")
print(f"Expected: ~1.000")
print(f"Similarity between vec1 and vec3 (opposite): {sim_13:.3f}")
print(f"Expected: ~-1.000")
print(f"Similarity between vec1 and vec4: {sim_14:.3f}")
print(f"Expected: ~0.267")

# Test with real word embeddings
cat = np.array([0.8, 0.6, 0.3])
dog = np.array([0.7, 0.7, 0.4])
car = np.array([0.2, 0.9, 0.1])

print(f"\\ncat-dog similarity: {cosine_similarity(cat, dog):.3f}")
print(f"cat-car similarity: {cosine_similarity(cat, car):.3f}")
print(f"Note: cat and dog are more similar!")


# SOLUTION TO EXERCISE 4: Find Nearest Neighbors

def find_nearest_neighbors(query_embedding, embeddings, words, k=5):
    # Calculate similarity with each word
    similarities = []

    for i, word_embedding in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, word_embedding)
        similarities.append((words[i], sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top k
    return similarities[:k]

# Test
words = ["king", "queen", "man", "woman", "prince", "princess", "cat", "dog"]

# Create simple 3D embeddings
embeddings_dict = {
    "king":     np.array([0.9, 0.1, 0.5]),
    "queen":    np.array([0.9, 0.9, 0.5]),
    "man":      np.array([0.5, 0.1, 0.3]),
    "woman":    np.array([0.5, 0.9, 0.3]),
    "prince":   np.array([0.8, 0.1, 0.6]),
    "princess": np.array([0.8, 0.9, 0.6]),
    "cat":      np.array([0.2, 0.5, 0.9]),
    "dog":      np.array([0.3, 0.5, 0.8]),
}

# Convert to matrix
embeddings = np.array([embeddings_dict[word] for word in words])

# Find words similar to "king"
query_word = "king"
query_embedding = embeddings_dict[query_word]

neighbors = find_nearest_neighbors(query_embedding, embeddings, words, k=4)

print(f"\\nWords most similar to '{query_word}':")
for word, similarity in neighbors:
    print(f"  {word:12} {similarity:.3f}")

# Test vector arithmetic: king - man + woman ≈ queen
print(f"\\n--- Testing Vector Arithmetic ---")
king_vec = embeddings_dict["king"]
man_vec = embeddings_dict["man"]
woman_vec = embeddings_dict["woman"]

result_vec = king_vec - man_vec + woman_vec

# Find nearest neighbor to result
neighbors = find_nearest_neighbors(result_vec, embeddings, words, k=3)
print(f"king - man + woman ≈ ?")
print(f"Nearest words:")
for word, similarity in neighbors:
    print(f"  {word:12} {similarity:.3f}")
print(f"Expected: queen should be at or near the top!")


# BONUS SOLUTION TO EXERCISE 5: Visualize Embeddings

def visualize_embeddings(embeddings_dict):
    import matplotlib.pyplot as plt

    # Extract words and embeddings
    words = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values()))

    # Check dimensionality
    if embeddings.shape[1] != 2:
        print(f"Warning: Embeddings have {embeddings.shape[1]} dimensions, expected 2")
        return

    # Create plot
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=100, alpha=0.6)

    # Add labels
    for i, word in enumerate(words):
        plt.annotate(word,
                    (embeddings[i, 0], embeddings[i, 1]),
                    fontsize=12,
                    ha='right',
                    va='bottom')

    plt.xlabel('Dimension 0', fontsize=12)
    plt.ylabel('Dimension 1', fontsize=12)
    plt.title('2D Word Embeddings Visualization', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Test visualization
import matplotlib.pyplot as plt

embeddings_2d = {
    # Animals
    "cat":      np.array([0.8, 0.6]),
    "dog":      np.array([0.7, 0.7]),
    "tiger":    np.array([0.9, 0.5]),
    "lion":     np.array([0.85, 0.55]),

    # Vehicles
    "car":      np.array([0.2, 0.9]),
    "truck":    np.array([0.3, 0.85]),
    "bus":      np.array([0.25, 0.95]),

    # Royalty
    "king":     np.array([0.5, 0.2]),
    "queen":    np.array([0.55, 0.25]),
    "prince":   np.array([0.45, 0.15]),
}

print("\\n--- Visualizing 2D Embeddings ---")
visualize_embeddings(embeddings_2d)
print("You should see three clusters:")
print("  - Animals (top right)")
print("  - Vehicles (top left)")
print("  - Royalty (bottom middle)")


# ADDITIONAL: Positional Embeddings

class PositionalEmbedding:
    def __init__(self, max_seq_length, embedding_dim):
        # Learned positional embeddings
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.pos_embeddings = np.random.randn(max_seq_length, embedding_dim) * 0.01

    def forward(self, seq_length):
        # Return positional embeddings for sequence
        return self.pos_embeddings[:seq_length]

# Test
max_seq_length = 100
embedding_dim = 50

pos_emb = PositionalEmbedding(max_seq_length, embedding_dim)

# Get positional embeddings for sequence of length 5
seq_length = 5
positions = pos_emb.forward(seq_length)

print(f"\\n--- Positional Embeddings ---")
print(f"Positional embeddings shape: {positions.shape}")
print(f"Expected: (5, 50)")
print(f"\\nEach position has unique embedding:")
print(f"Position 0: {positions[0, :5]}")
print(f"Position 1: {positions[1, :5]}")
print(f"Position 2: {positions[2, :5]}")


# ADDITIONAL: Combined Word + Positional Embeddings

class TokenEmbedding:
    def __init__(self, vocab_size, max_seq_length, embedding_dim):
        # Word embeddings
        self.word_emb = EmbeddingLayer(vocab_size, embedding_dim)

        # Positional embeddings
        self.pos_emb = PositionalEmbedding(max_seq_length, embedding_dim)

    def forward(self, token_ids):
        seq_length = len(token_ids)

        # Get word embeddings
        word_embeddings = self.word_emb(token_ids)

        # Get positional embeddings
        pos_embeddings = self.pos_emb.forward(seq_length)

        # Combine (element-wise addition)
        combined = word_embeddings + pos_embeddings

        return combined

# Test
vocab_size = 10000
max_seq_length = 512
embedding_dim = 768

token_emb = TokenEmbedding(vocab_size, max_seq_length, embedding_dim)

# Encode sentence
sentence_ids = np.array([5, 42, 17, 99, 3])  # "The cat sat on mat"
embeddings = token_emb.forward(sentence_ids)

print(f"\\n--- Combined Embeddings (Word + Position) ---")
print(f"Final embeddings shape: {embeddings.shape}")
print(f"Expected: (5, 768)")
print(f"\\nEach token now has:")
print(f"  - Word meaning (from word embedding)")
print(f"  - Position information (from positional embedding)")
print(f"  - Ready to feed into transformer! ✓")
"""

print("\n" + "="*70)
print("Exercise 2 Complete! Great work on embeddings!")
print("="*70)

# Lesson 5.2: Word Embeddings - Making Words Meaningful

**How neural networks understand the meaning of words!**

---

## 🎯 What You'll Learn

- ✅ Why neural networks need dense vectors, not just IDs
- ✅ The problem with one-hot encoding (sparse, huge, no semantics)
- ✅ What embeddings are (dense, meaningful, learned)
- ✅ How embedding dimensions work
- ✅ Word2Vec intuition (CBOW and Skip-gram)
- ✅ Famous example: king - man + woman = queen
- ✅ Embeddings as lookup tables
- ✅ Cosine similarity for measuring word relationships
- ✅ How embeddings are trained
- ✅ Positional embeddings (critical for transformers!)
- ✅ Connection to GPT and modern LLMs

**Time:** 2-3 hours
**Difficulty:** ⭐⭐⭐⭐☆

---

## 🤔 The Problem: Token IDs Aren't Enough

### What We Have So Far

From Lesson 5.1, we can convert text to token IDs:

```python
text = "The cat sat"

# Tokenization
tokens = ["The", "cat", "sat"]
token_ids = [5, 42, 17]
```

**But there's a HUGE problem!** 🚨

---

### Why Token IDs Don't Work for Neural Networks

**Token IDs are just labels!** They don't capture meaning.

```python
vocab = {
    "cat": 42,
    "dog": 17,
    "automobile": 99
}

# Problem 1: No semantic relationship
# "cat" (42) and "dog" (17) are both animals
# But 42 and 17 are just random numbers!
# Neural network sees: |42 - 17| = 25

# "cat" (42) and "automobile" (99)
# Completely different concepts
# Neural network sees: |42 - 99| = 57

# The network thinks "dog" is MORE similar to "cat"
# than "automobile" is, but only because of random ID assignment!
```

**Token IDs have NO semantic meaning!** 😱

---

### Real-World Analogy: Student ID Numbers

**Imagine a school where students are assigned random ID numbers:**

```
Alice (loves math, hates sports): ID 42
Bob (loves math, hates sports): ID 1001
Charlie (hates math, loves sports): ID 43

Random ID assignment problems:
- Alice (42) and Charlie (43) have consecutive IDs
  BUT completely different interests!

- Alice (42) and Bob (1001) have similar interests
  BUT distant IDs!

A teacher looking only at ID numbers would think:
"Alice and Charlie must be similar" ✗ Wrong!
```

**IDs are just labels, not representations of who the students are!**

---

### What We Really Need

We need a way to represent words that captures their **meaning and relationships**.

```
"cat" → [0.2, -0.5, 0.8, 0.1, ...]  ← Rich representation
"dog" → [0.3, -0.4, 0.7, 0.2, ...]  ← Similar to "cat"!
"car" → [-0.9, 0.6, -0.3, 0.8, ...] ← Very different!
```

**This is what embeddings do!** 🎉

---

## 🚫 Approach 1: One-Hot Encoding (Naive Solution)

### What is One-Hot Encoding?

**One-hot encoding** represents each token as a vector with all 0s except one 1.

```python
vocab_size = 5
vocab = {"cat": 0, "dog": 1, "car": 2, "run": 3, "jump": 4}

# "cat" (ID 0)
cat_onehot = [1, 0, 0, 0, 0]

# "dog" (ID 1)
dog_onehot = [0, 1, 0, 0, 0]

# "car" (ID 2)
car_onehot = [0, 0, 1, 0, 0]
```

**Each word is a vector of length = vocabulary size!**

---

### Example: One-Hot Encoding Implementation

```python
import numpy as np

class OneHotEncoder:
    """
    One-hot encoding for tokens

    Like giving each student a unique flag to wave!
    """

    def __init__(self, vocab_size):
        """
        Initialize encoder

        Args:
            vocab_size: Size of vocabulary
        """
        self.vocab_size = vocab_size

    def encode(self, token_id):
        """
        Convert token ID to one-hot vector

        Args:
            token_id: Integer token ID

        Returns:
            One-hot vector (numpy array)
        """
        # Create vector of zeros
        one_hot = np.zeros(self.vocab_size)

        # Set the position to 1
        one_hot[token_id] = 1

        return one_hot

    def encode_batch(self, token_ids):
        """
        Encode multiple tokens

        Args:
            token_ids: List of token IDs

        Returns:
            Matrix of one-hot vectors
        """
        batch_size = len(token_ids)
        one_hot_matrix = np.zeros((batch_size, self.vocab_size))

        for i, token_id in enumerate(token_ids):
            one_hot_matrix[i, token_id] = 1

        return one_hot_matrix


# Example usage
vocab_size = 10000  # Typical small vocabulary
encoder = OneHotEncoder(vocab_size)

# Encode single token
cat_id = 42
cat_vector = encoder.encode(cat_id)
print(f"Shape: {cat_vector.shape}")  # (10000,)
print(f"Non-zero elements: {np.count_nonzero(cat_vector)}")  # 1

# Encode sentence
sentence_ids = [5, 42, 17]  # "The cat sat"
sentence_vectors = encoder.encode_batch(sentence_ids)
print(f"Sentence shape: {sentence_vectors.shape}")  # (3, 10000)
```

**Output:**
```
Shape: (10000,)
Non-zero elements: 1
Sentence shape: (3, 10000)
```

---

### The MASSIVE Problems with One-Hot Encoding

**Problem 1: Sparse Vectors (99.99% zeros!)**

```python
vocab_size = 50000  # GPT-3 size

# One word representation
word_vector = [0, 0, 0, ..., 1, ..., 0, 0]  # 49,999 zeros, 1 one

# Memory usage per word: 50,000 floats
# If each float = 4 bytes: 200 KB per word!
# For 1000 words: 200 MB!!! 😱
```

**Extremely wasteful!**

---

**Problem 2: No Semantic Relationships**

```python
cat = [1, 0, 0, 0, 0, ...]
dog = [0, 1, 0, 0, 0, ...]
car = [0, 0, 1, 0, 0, ...]

# Similarity between cat and dog
np.dot(cat, dog) = 0  ← No relationship!

# Similarity between cat and car
np.dot(cat, car) = 0  ← Also no relationship!

# All words are equally distant! ✗
```

**One-hot encoding can't capture meaning!**

---

**Problem 3: Dimension Explosion**

```python
# Character-level tokenizer
vocab_size = 100
one_hot_vector_size = 100  ← Manageable

# Word-level tokenizer
vocab_size = 50000
one_hot_vector_size = 50000  ← Too big!

# For GPT-3
vocab_size = 50257
one_hot_vector_size = 50257  ← Impossible!

# Modern LLMs would need:
# 50,000 dimensions × 96 layers = 4,800,000 parameters
# Just for embeddings! 😵
```

**Doesn't scale!**

---

### Visual Comparison

```
One-Hot Encoding:
┌─────────────────────────────────────────────┐
│ "cat" = [0, 0, ..., 1, ..., 0, 0]          │
│          ↑                                  │
│          Position 42                        │
│          50,000 dimensions!                 │
│          99.998% zeros                      │
│          No meaning captured                │
└─────────────────────────────────────────────┘

What we need (Embeddings):
┌─────────────────────────────────────────────┐
│ "cat" = [0.2, -0.5, 0.8, 0.1, -0.3, 0.6]   │
│          ↑                                  │
│          Just 6 dimensions!                 │
│          All values meaningful              │
│          Captures semantics                 │
└─────────────────────────────────────────────┘
```

**We need embeddings!** ⭐

---

## ✨ The Solution: Dense Word Embeddings

### What are Embeddings?

**Embeddings** are dense, low-dimensional, learned representations of words.

```python
# One-hot (sparse)
"cat" → [0, 0, 0, ..., 1, ..., 0]  ← 50,000 dims, mostly zeros

# Embedding (dense)
"cat" → [0.2, -0.5, 0.8, 0.1]  ← 4 dims, all meaningful
```

---

### Key Properties of Embeddings

**1. Dense (not sparse)**
```python
# Every dimension has a meaningful value
embedding = [0.2, -0.5, 0.8, 0.1, -0.3, 0.6, ...]
             ↑     ↑     ↑     ↑      ↑     ↑
            All non-zero and meaningful!
```

**2. Low-dimensional (not huge)**
```python
# Typical sizes
vocab_size = 50000       # How many words
embedding_dim = 300      # Much smaller!

# Common embedding dimensions:
# - Word2Vec: 100-300
# - GPT-2: 768
# - GPT-3: 12288
# - BERT: 768
```

**3. Learned (not hand-crafted)**
```python
# Neural network learns these values during training!
# Not random, not hand-designed
# Optimized to capture meaning!
```

**4. Captures semantics**
```python
cat_embedding = [0.2, -0.5, 0.8, ...]
dog_embedding = [0.3, -0.4, 0.7, ...]  ← Similar to cat!
car_embedding = [-0.9, 0.6, -0.3, ...] ← Different!
```

---

### Real-World Analogy: Student Profiles

**Instead of random ID numbers, create meaningful profiles:**

```
One-Hot (Random IDs):
Alice: 42
Bob: 1001
Charlie: 43

Embeddings (Meaningful Profiles):
              Math  Sports  Art  Music
Alice:       [0.9,  0.1,   0.6,  0.3]  ← Loves math
Bob:         [0.8,  0.2,   0.5,  0.4]  ← Also loves math (similar to Alice!)
Charlie:     [0.1,  0.9,   0.3,  0.7]  ← Loves sports (different!)

Now we can measure similarity:
- Alice and Bob are similar (both high on math)
- Alice and Charlie are different (different interests)
```

**Embeddings capture the "profile" of each word!** 📊

---

## 🔢 Embedding Dimensions: What Do They Mean?

### Dimensions as Features

Each dimension captures some aspect of meaning (learned automatically!).

**Hypothetical example (simplified):**

```python
         Dim 0    Dim 1      Dim 2      Dim 3
         Animal?  Size       Domestic?  Speed
cat   = [0.9,     0.3,       0.8,       0.6]
dog   = [0.9,     0.4,       0.9,       0.7]
tiger = [0.9,     0.8,       0.1,       0.9]
car   = [0.1,     0.5,       0.2,       0.8]
```

**Note:** Real embeddings don't have such clear meanings! Dimensions are learned and abstract.

---

### How Many Dimensions?

**Trade-off: Expressiveness vs Efficiency**

```python
# Too few dimensions (e.g., 2)
embedding_dim = 2
# Pros: Fast, visualizable
# Cons: Can't capture complex relationships

# Too many dimensions (e.g., 10000)
embedding_dim = 10000
# Pros: More expressive
# Cons: Slow, overfits, wastes memory

# Sweet spot (100-1000)
embedding_dim = 300  # Word2Vec default
embedding_dim = 768  # BERT/GPT-2
embedding_dim = 12288  # GPT-3

# Larger models → more dimensions
```

---

### Example: 2D Embeddings (Visualizable!)

```python
import numpy as np
import matplotlib.pyplot as plt

# Create simple 2D embeddings
embeddings = {
    "cat": np.array([0.8, 0.6]),
    "dog": np.array([0.7, 0.7]),
    "tiger": np.array([0.9, 0.3]),
    "car": np.array([0.2, 0.9]),
    "truck": np.array([0.3, 0.8]),
    "airplane": np.array([0.1, 0.5])
}

# Plot
plt.figure(figsize=(8, 6))
for word, embedding in embeddings.items():
    plt.scatter(embedding[0], embedding[1], s=100)
    plt.annotate(word, (embedding[0], embedding[1]),
                fontsize=12, ha='right')

plt.xlabel('Dimension 0 (maybe "animal-ness"?)')
plt.ylabel('Dimension 1 (maybe "size"?)')
plt.title('2D Word Embeddings')
plt.grid(True, alpha=0.3)
plt.show()
```

**You'll see clusters!**
- Animals cluster together (cat, dog, tiger)
- Vehicles cluster together (car, truck, airplane)

---

## 🧠 How Embeddings Capture Meaning

### The Famous Example: Vector Arithmetic

**king - man + woman = queen**

This actually works with trained embeddings! 🤯

```python
# Word vectors (simplified, real ones have 300+ dims)
king = [0.9, 0.1, 0.8]
man = [0.7, 0.1, 0.2]
woman = [0.7, 0.9, 0.2]
queen = [0.9, 0.9, 0.8]

# Vector arithmetic
result = king - man + woman
# result = [0.9, 0.1, 0.8] - [0.7, 0.1, 0.2] + [0.7, 0.9, 0.2]
# result = [0.9, 0.9, 0.8]  ← This is the "queen" vector!

# Why does this work?
# king - man = "royalty" concept
# "royalty" + woman = "female royalty" = queen! ✓
```

---

### More Examples of Vector Arithmetic

**Countries and Capitals:**

```python
Paris - France + Italy ≈ Rome
Tokyo - Japan + China ≈ Beijing
```

**Verb Tenses:**

```python
walking - walk + run ≈ running
played - play + eat ≈ eaten
```

**Comparatives:**

```python
bigger - big + small ≈ smaller
fastest - fast + slow ≈ slowest
```

**Gender:**

```python
actress - actor + waiter ≈ waitress
uncle - aunt + brother ≈ sister
```

**This is AMAZING!** Math on words that makes semantic sense! 🎉

---

### Why Does This Work?

**Embeddings learn relationships as vectors:**

```
"king" vector = royalty + male
"man" vector = male
"woman" vector = female

king - man = royalty + male - male = royalty
royalty + woman = royalty + female = queen ✓

Mathematically:
vec(king) - vec(man) + vec(woman) ≈ vec(queen)
```

**Neural networks learn these patterns from data!**

---

## 📚 Embedding as Lookup Table

### How Embeddings Work in Practice

**Embeddings are just a big lookup table!**

```python
import numpy as np

class EmbeddingLayer:
    """
    Embedding layer: converts token IDs to dense vectors

    Like a dictionary mapping words to their "profiles"!
    """

    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize embedding layer

        Args:
            vocab_size: Number of unique tokens
            embedding_dim: Size of embedding vectors
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # The lookup table (randomly initialized)
        # Shape: (vocab_size, embedding_dim)
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward(self, token_ids):
        """
        Look up embeddings for token IDs

        Args:
            token_ids: Array of token IDs, shape (batch_size,)

        Returns:
            Embeddings, shape (batch_size, embedding_dim)
        """
        # Simple lookup!
        return self.embeddings[token_ids]

    def __call__(self, token_ids):
        return self.forward(token_ids)


# Example usage
vocab_size = 10000
embedding_dim = 300

embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)

# Look up embedding for word "cat" (ID 42)
cat_id = 42
cat_embedding = embedding_layer(cat_id)
print(f"Cat embedding shape: {cat_embedding.shape}")  # (300,)
print(f"First 5 values: {cat_embedding[:5]}")

# Look up embeddings for sentence
sentence_ids = np.array([5, 42, 17])  # "The cat sat"
sentence_embeddings = embedding_layer(sentence_ids)
print(f"Sentence embeddings shape: {sentence_embeddings.shape}")  # (3, 300)
```

**Output:**
```
Cat embedding shape: (300,)
First 5 values: [ 0.01234, -0.00567,  0.00891, -0.01023,  0.00445]
Sentence embeddings shape: (3, 300)
```

---

### Lookup Table Visualization

```
Embedding Table (vocab_size × embedding_dim):

Token ID    Embedding Vector
   0     → [0.1, -0.5, 0.8, ..., 0.3]
   1     → [0.2,  0.3, -0.1, ..., 0.7]
   ...
  42     → [0.9, -0.2, 0.5, ..., -0.4]  ← "cat"
   ...
50000    → [-0.3, 0.6, 0.1, ..., 0.8]

Input: token_id = 42
Output: [0.9, -0.2, 0.5, ..., -0.4]

It's just array indexing! ✓
```

---

### C#/.NET Analogy

```csharp
// C#: Dictionary lookup
Dictionary<int, float[]> embeddings = new Dictionary<int, float[]>
{
    {42, new float[] {0.9f, -0.2f, 0.5f}},  // "cat"
    {17, new float[] {0.8f, -0.1f, 0.6f}},  // "dog"
};

float[] catEmbedding = embeddings[42];

// Python: Same concept but with NumPy array
embeddings = np.array([
    [0.9, -0.2, 0.5],  # ID 0
    # ...
    [0.8, -0.1, 0.6],  # ID 42 (cat)
])

cat_embedding = embeddings[42]  # Just array indexing!
```

**Very similar to C# dictionary, but faster with NumPy!**

---

## 📏 Measuring Similarity: Cosine Similarity

### Why We Need Similarity Metrics

**How do we measure if two words are similar?**

```python
cat = [0.8, 0.6]
dog = [0.7, 0.7]
car = [0.2, 0.9]

# Are "cat" and "dog" similar?
# Are "cat" and "car" similar?
```

**We need a mathematical way to compare vectors!**

---

### Cosine Similarity

**Cosine similarity** measures the angle between vectors.

```
Formula:
similarity = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product
- ||A|| = magnitude (length) of A
- Range: -1 to 1
  - 1 = Same direction (very similar)
  - 0 = Perpendicular (unrelated)
  - -1 = Opposite direction (opposite meaning)
```

---

### Visual Explanation

```
2D Space:

      dog •
         /|
        / |
       /  |
    cat • |
       \  |
        \ |
         \|
          • car

Angle between cat and dog: Small → High similarity (0.95)
Angle between cat and car: Large → Low similarity (0.10)
```

---

### Implementation

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score (-1 to 1)
    """
    # Dot product
    dot_product = np.dot(vec1, vec2)

    # Magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Cosine similarity
    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (magnitude1 * magnitude2)


# Example
cat = np.array([0.8, 0.6, 0.3])
dog = np.array([0.7, 0.7, 0.4])
car = np.array([0.2, 0.9, 0.1])

# Calculate similarities
cat_dog_sim = cosine_similarity(cat, dog)
cat_car_sim = cosine_similarity(cat, car)

print(f"cat-dog similarity: {cat_dog_sim:.3f}")  # ~0.98 (very similar!)
print(f"cat-car similarity: {cat_car_sim:.3f}")  # ~0.65 (less similar)
```

---

### Finding Nearest Neighbors

**Find most similar words to a query word:**

```python
def find_nearest_neighbors(query_embedding, embeddings, word_list, top_k=5):
    """
    Find most similar words

    Args:
        query_embedding: Embedding vector of query word
        embeddings: All word embeddings (shape: vocab_size × embedding_dim)
        word_list: List of words corresponding to embeddings
        top_k: Number of neighbors to return

    Returns:
        List of (word, similarity) tuples
    """
    similarities = []

    for i, embedding in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((word_list[i], sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top k
    return similarities[:top_k]


# Example usage
words = ["cat", "dog", "tiger", "car", "truck", "apple"]
embeddings = np.array([
    [0.8, 0.6, 0.3],  # cat
    [0.7, 0.7, 0.4],  # dog
    [0.9, 0.5, 0.2],  # tiger
    [0.2, 0.9, 0.1],  # car
    [0.3, 0.8, 0.2],  # truck
    [0.1, 0.3, 0.9],  # apple
])

query_word = "cat"
query_embedding = embeddings[0]

neighbors = find_nearest_neighbors(query_embedding, embeddings, words, top_k=3)

print(f"Words most similar to '{query_word}':")
for word, similarity in neighbors:
    print(f"  {word}: {similarity:.3f}")
```

**Output:**
```
Words most similar to 'cat':
  cat: 1.000  (itself!)
  dog: 0.984
  tiger: 0.956
```

**Animals cluster together!** ✓

---

## 🎓 How Are Embeddings Trained?

### Overview: Learning from Context

**Key insight:** Words that appear in similar contexts have similar meanings!

```
Context examples:
"The cat sat on the mat"
"The dog sat on the mat"

"cat" and "dog" appear in similar contexts → similar embeddings!
```

**This is called the distributional hypothesis.**

---

### Word2Vec: Two Approaches

**Word2Vec** is a classic method for learning embeddings.

**Two architectures:**

1. **CBOW (Continuous Bag of Words)**: Predict word from context
2. **Skip-gram**: Predict context from word

---

### CBOW (Continuous Bag of Words)

**Given context words → predict target word**

```
Sentence: "The cat sat on the mat"

Training example:
Context: ["The", "sat", "on", "the"]  ← Surrounding words
Target: "cat"                         ← Middle word

Task: Given context, predict "cat"
```

**Architecture:**

```
Input: Context words
  ↓
Average their embeddings
  ↓
Neural network
  ↓
Predict target word
```

---

### Skip-gram

**Given target word → predict context words**

```
Sentence: "The cat sat on the mat"

Training examples:
Target: "cat"
Context: "The"    ← Predict this
Context: "sat"    ← Predict this
Context: "on"     ← Predict this
Context: "the"    ← Predict this

Task: Given "cat", predict surrounding words
```

**Architecture:**

```
Input: Target word
  ↓
Look up its embedding
  ↓
Neural network
  ↓
Predict each context word
```

---

### Simplified Skip-gram Example

```python
import numpy as np

class SimpleSkipGram:
    """
    Simplified Skip-gram model

    Learn embeddings by predicting context words!
    """

    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize Skip-gram model

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
        """
        # Input embeddings (word → vector)
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01

        # Output weights (vector → word prediction)
        self.output_weights = np.random.randn(embedding_dim, vocab_size) * 0.01

    def forward(self, target_word_id):
        """
        Forward pass: predict context words

        Args:
            target_word_id: ID of target word

        Returns:
            Predictions for all vocabulary words
        """
        # Look up target word embedding
        target_embedding = self.embeddings[target_word_id]

        # Predict context words
        logits = target_embedding @ self.output_weights

        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)

        return probs

    def get_embedding(self, word_id):
        """Get learned embedding for a word"""
        return self.embeddings[word_id]


# Example (training omitted for brevity)
vocab_size = 1000
embedding_dim = 50

model = SimpleSkipGram(vocab_size, embedding_dim)

# Get embedding for word "cat" (ID 42)
cat_embedding = model.get_embedding(42)
print(f"Cat embedding shape: {cat_embedding.shape}")  # (50,)
```

---

### Training Process

**Simplified training loop:**

```python
# For each sentence in corpus:
#   1. Slide a window over the sentence
#   2. For each word in window:
#      - Target = center word
#      - Context = surrounding words
#   3. Forward pass: predict context from target
#   4. Calculate loss (how wrong were we?)
#   5. Backpropagation: update embeddings
#   6. Repeat millions of times!

# After training:
# - Words in similar contexts have similar embeddings
# - "king - man + woman ≈ queen" emerges automatically!
```

**The neural network learns meaning from patterns!** 🧠

---

## 🔢 Positional Embeddings: Where Words Appear Matters

### The Problem: Word Order

**Embeddings ignore word order!**

```python
sentence1 = "The cat chased the dog"
sentence2 = "The dog chased the cat"

# Both sentences use same words: [The, cat, chased, the, dog]
# But completely different meanings!

# With only word embeddings, the model sees:
# Same set of embeddings → Same representation ✗ Wrong!
```

**We need to encode POSITION!** 📍

---

### Solution: Positional Embeddings

**Add position information to each word:**

```python
# Word embedding: captures meaning
word_emb = [0.8, 0.6, 0.3]

# Positional embedding: captures position
pos_emb = [0.1, -0.2, 0.4]  ← Unique for each position

# Final embedding: word + position
final_emb = word_emb + pos_emb = [0.9, 0.4, 0.7]
```

---

### Example: Positional Embeddings

```python
import numpy as np

class PositionalEmbedding:
    """
    Positional embeddings for transformers

    Tells the model WHERE each word is!
    """

    def __init__(self, max_seq_length, embedding_dim):
        """
        Initialize positional embeddings

        Args:
            max_seq_length: Maximum sequence length
            embedding_dim: Embedding dimension
        """
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim

        # Learned positional embeddings (random initially)
        self.pos_embeddings = np.random.randn(max_seq_length, embedding_dim) * 0.01

    def forward(self, seq_length):
        """
        Get positional embeddings for sequence

        Args:
            seq_length: Length of current sequence

        Returns:
            Positional embeddings, shape (seq_length, embedding_dim)
        """
        return self.pos_embeddings[:seq_length]


# Example usage
max_seq_length = 512  # Max sequence length
embedding_dim = 768   # GPT-2 dimension

pos_embedding = PositionalEmbedding(max_seq_length, embedding_dim)

# Get positional embeddings for sentence of length 5
seq_length = 5
pos_embs = pos_embedding.forward(seq_length)
print(f"Positional embeddings shape: {pos_embs.shape}")  # (5, 768)

# Each position gets unique embedding
print(f"Position 0: {pos_embs[0, :5]}")
print(f"Position 1: {pos_embs[1, :5]}")
print(f"Position 2: {pos_embs[2, :5]}")
```

---

### Combining Word and Positional Embeddings

```python
class TokenEmbedding:
    """
    Complete embedding: word + position

    Like GPT does it!
    """

    def __init__(self, vocab_size, max_seq_length, embedding_dim):
        """
        Initialize embeddings

        Args:
            vocab_size: Vocabulary size
            max_seq_length: Maximum sequence length
            embedding_dim: Embedding dimension
        """
        # Word embeddings
        self.word_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01

        # Positional embeddings
        self.pos_embeddings = np.random.randn(max_seq_length, embedding_dim) * 0.01

    def forward(self, token_ids):
        """
        Get embeddings for tokens

        Args:
            token_ids: Token IDs, shape (seq_length,)

        Returns:
            Combined embeddings, shape (seq_length, embedding_dim)
        """
        seq_length = len(token_ids)

        # Look up word embeddings
        word_embs = self.word_embeddings[token_ids]  # (seq_length, embedding_dim)

        # Get positional embeddings
        pos_embs = self.pos_embeddings[:seq_length]  # (seq_length, embedding_dim)

        # Combine: element-wise addition
        combined = word_embs + pos_embs

        return combined


# Example
vocab_size = 50000
max_seq_length = 512
embedding_dim = 768

embedding_layer = TokenEmbedding(vocab_size, max_seq_length, embedding_dim)

# Encode sentence
sentence_ids = np.array([5, 42, 17, 99, 3])  # "The cat sat on mat"
sentence_embeddings = embedding_layer.forward(sentence_ids)

print(f"Final embeddings shape: {sentence_embeddings.shape}")  # (5, 768)
print("\nEach token now has:")
print("  - Word meaning (from word embedding)")
print("  - Position information (from positional embedding)")
```

---

### Why Positional Embeddings Matter

**Without positional embeddings:**

```
"cat chased dog" = bag of word meanings
"dog chased cat" = same bag of meanings
Model can't tell them apart! ✗
```

**With positional embeddings:**

```
"cat chased dog":
  Position 0: "cat" + pos_0
  Position 1: "chased" + pos_1
  Position 2: "dog" + pos_2

"dog chased cat":
  Position 0: "dog" + pos_0  ← Different!
  Position 1: "chased" + pos_1
  Position 2: "cat" + pos_2  ← Different!

Model can distinguish them! ✓
```

---

## 🔗 Connection to Transformers and GPT

### How GPT Uses Embeddings

**GPT's embedding layer:**

```python
# Simplified GPT embedding
class GPTEmbedding:
    def __init__(self, vocab_size, max_seq_length, embedding_dim):
        # Word embeddings (learned)
        self.wte = np.random.randn(vocab_size, embedding_dim) * 0.02

        # Position embeddings (learned)
        self.wpe = np.random.randn(max_seq_length, embedding_dim) * 0.02

    def forward(self, token_ids):
        # Word embeddings
        word_embs = self.wte[token_ids]

        # Positional embeddings
        positions = np.arange(len(token_ids))
        pos_embs = self.wpe[positions]

        # Combine
        return word_embs + pos_embs


# GPT-2 Small configuration
gpt_embedding = GPTEmbedding(
    vocab_size=50257,      # GPT-2 vocab size
    max_seq_length=1024,   # GPT-2 max context
    embedding_dim=768      # GPT-2 hidden size
)
```

---

### Embedding Flow in GPT

```
Input text: "The cat sat"
     ↓
Tokenization: [464, 3797, 3332]
     ↓
Word embeddings:
  "The"  (464)  → [0.1, -0.5, 0.8, ..., 0.3]  (768 dims)
  "cat"  (3797) → [0.9, -0.2, 0.5, ..., -0.4]
  "sat"  (3332) → [0.3,  0.6, -0.1, ..., 0.7]
     ↓
Positional embeddings:
  Position 0 → [0.01, -0.02, 0.03, ..., 0.01]
  Position 1 → [0.02,  0.01, -0.01, ..., 0.02]
  Position 2 → [-0.01, 0.03, 0.02, ..., -0.01]
     ↓
Combined (word + position):
  Token 0 → word[464] + pos[0]
  Token 1 → word[3797] + pos[1]
  Token 2 → word[3332] + pos[2]
     ↓
Feed to transformer layers
     ↓
Output predictions
```

---

### GPT Embedding Parameters

**GPT-2 Small:**
- Vocabulary size: 50,257
- Embedding dimension: 768
- Word embeddings: 50,257 × 768 = **38.6M parameters**
- Position embeddings: 1,024 × 768 = **0.8M parameters**
- **Total: 39.4M parameters just for embeddings!**

**GPT-3:**
- Vocabulary size: 50,257
- Embedding dimension: 12,288
- Word embeddings: 50,257 × 12,288 = **618M parameters**
- Position embeddings: 2,048 × 12,288 = **25M parameters**
- **Total: 643M parameters for embeddings!**

**Embeddings are a HUGE part of LLMs!** 😲

---

## 🎓 Summary

### Key Concepts

**1. Why embeddings?**
- Token IDs are just labels (no meaning)
- One-hot encoding is sparse, huge, and doesn't capture semantics
- Embeddings are dense, low-dimensional, and meaningful

**2. Properties of embeddings:**
- Dense (all values meaningful)
- Low-dimensional (100-1000 dims vs 50,000)
- Learned (optimized during training)
- Capture semantics (similar words → similar vectors)

**3. Famous examples:**
- king - man + woman ≈ queen
- Paris - France + Italy ≈ Rome
- Vector arithmetic captures relationships!

**4. Embeddings as lookup tables:**
- Simple array indexing
- Input: token ID → Output: embedding vector

**5. Measuring similarity:**
- Cosine similarity measures angle between vectors
- Range: -1 (opposite) to 1 (same)
- Find nearest neighbors to discover related words

**6. How they're trained:**
- Word2Vec: CBOW and Skip-gram
- Learn from context: words in similar contexts → similar embeddings
- Distributional hypothesis

**7. Positional embeddings:**
- Word order matters!
- Add position information to word embeddings
- Critical for transformers (no recurrence)

**8. In GPT:**
- Word embeddings: vocab_size × embedding_dim
- Position embeddings: max_seq_length × embedding_dim
- Combined: word + position
- Fed to transformer layers

---

### What You Built

✅ Understanding of one-hot encoding (and why it fails)
✅ Dense embedding layer implementation
✅ Cosine similarity calculation
✅ Nearest neighbor search
✅ Positional embedding implementation
✅ Complete token embedding (word + position)

---

### Next Steps

In the next lesson (**03_building_gpt_architecture.md**), you'll learn:
- Transformer architecture overview
- Self-attention mechanism
- Multi-head attention
- Feed-forward networks
- Layer normalization
- How all the pieces fit together!

---

## 📝 Quiz

Test your understanding!

### Question 1
**Why doesn't one-hot encoding work well for word representations?**

<details>
<summary>Click to see answer</summary>

One-hot encoding has three major problems:

1. **Sparse vectors**: 99.99% zeros. For vocab_size=50,000, each word is a 50,000-dimensional vector with only one 1. Extremely wasteful of memory!

2. **No semantic relationships**: All words are equally distant. The dot product between any two different words is always 0, so "cat" and "dog" (both animals) appear as unrelated as "cat" and "car".

3. **Dimension explosion**: Doesn't scale. Modern LLMs would need millions of dimensions just for one-hot vectors, making training impossible.

Embeddings solve all three problems with dense, low-dimensional, meaningful representations!
</details>

---

### Question 2
**What are the key properties of word embeddings?**

<details>
<summary>Click to see answer</summary>

Four key properties:

1. **Dense**: Every dimension has a meaningful value (no zeros waste).

2. **Low-dimensional**: Typical sizes 100-1000 instead of vocab_size (50,000+).

3. **Learned**: Neural networks optimize these values during training, not hand-crafted.

4. **Semantic**: Captures meaning and relationships. Similar words have similar embeddings (e.g., "cat" and "dog" vectors are close).

This makes them efficient and effective for neural networks!
</details>

---

### Question 3
**Explain the famous example: king - man + woman = queen**

<details>
<summary>Click to see answer</summary>

This demonstrates that embeddings capture semantic relationships as vectors:

```
vec(king) = [royalty] + [male]
vec(man) = [male]
vec(woman) = [female]

king - man = [royalty] + [male] - [male] = [royalty]
[royalty] + woman = [royalty] + [female] = queen

Therefore: king - man + woman ≈ queen ✓
```

The neural network learns these relationships automatically from training data! Words used in similar contexts develop similar embeddings, and relationships emerge as vector directions. Other examples:
- Paris - France + Italy ≈ Rome
- walking - walk + run ≈ running

This is vector arithmetic that makes semantic sense!
</details>

---

### Question 4
**How do embeddings work as a lookup table?**

<details>
<summary>Click to see answer</summary>

Embeddings are stored in a matrix of shape (vocab_size × embedding_dim):

```python
# Embedding table
embeddings = np.array([
    [0.1, -0.5, 0.8, ...],  # Token ID 0
    [0.2,  0.3, -0.1, ...],  # Token ID 1
    ...
    [0.9, -0.2, 0.5, ...],  # Token ID 42 (e.g., "cat")
    ...
])

# Lookup: just array indexing!
token_id = 42
embedding = embeddings[token_id]  # Returns vector for "cat"
```

It's a simple lookup operation - given a token ID, retrieve its embedding vector. During training, these embedding values are optimized to capture semantic meaning. Very similar to a C# Dictionary<int, float[]>, but faster with NumPy arrays!
</details>

---

### Question 5
**What is cosine similarity and why is it useful?**

<details>
<summary>Click to see answer</summary>

Cosine similarity measures the angle between two vectors:

```
Formula: similarity = (A · B) / (||A|| × ||B||)

Range: -1 to 1
- 1: Same direction (very similar)
- 0: Perpendicular (unrelated)
- -1: Opposite direction (opposite meaning)
```

It's useful because:

1. **Measures semantic similarity**: Similar words have high cosine similarity (e.g., "cat" and "dog" ≈ 0.95).

2. **Finds nearest neighbors**: Can search for most similar words to a query.

3. **Scale-invariant**: Focuses on direction, not magnitude. A word appearing frequently or rarely won't affect similarity.

Example:
```python
cat = [0.8, 0.6]
dog = [0.7, 0.7]  # Similar to cat
car = [0.2, 0.9]  # Different from cat

cosine_similarity(cat, dog) ≈ 0.98  # High!
cosine_similarity(cat, car) ≈ 0.65  # Lower
```
</details>

---

### Question 6
**What's the difference between CBOW and Skip-gram in Word2Vec?**

<details>
<summary>Click to see answer</summary>

Both are Word2Vec architectures for learning embeddings, but opposite approaches:

**CBOW (Continuous Bag of Words):**
- Given: Context words (surrounding words)
- Predict: Target word (middle word)
- Example: Context ["The", "sat", "on", "the"] → Predict "cat"
- Faster to train
- Better for frequent words

**Skip-gram:**
- Given: Target word
- Predict: Context words (surrounding words)
- Example: Given "cat" → Predict ["The", "sat", "on", "the"]
- Slower to train
- Better for rare words
- Generally produces better embeddings

Both learn from the distributional hypothesis: words appearing in similar contexts have similar meanings. The neural network learns embeddings that make these predictions accurate, and semantic relationships emerge automatically!
</details>

---

### Question 7
**Why do we need positional embeddings in transformers?**

<details>
<summary>Click to see answer</summary>

Word embeddings alone don't capture word order:

```
"The cat chased the dog" → [The, cat, chased, the, dog]
"The dog chased the cat" → [The, dog, chased, the, cat]

Without position info, same words = same representation!
But meanings are completely different! ✗
```

Transformers process all words in parallel (unlike RNNs which process sequentially), so they have no inherent notion of position. Positional embeddings solve this:

```python
final_embedding = word_embedding + positional_embedding
```

Now each token has:
- Word meaning (what it is)
- Position information (where it is)

This allows the model to distinguish "cat at position 1" from "cat at position 4", preserving word order information critical for understanding language!

GPT uses learned positional embeddings that are added to word embeddings.
</details>

---

### Question 8
**How are embeddings used in GPT?**

<details>
<summary>Click to see answer</summary>

GPT uses embeddings in two layers:

**1. Word Embeddings (wte):**
- Matrix: vocab_size × embedding_dim
- Maps each token to a learned vector
- Captures word meaning

**2. Positional Embeddings (wpe):**
- Matrix: max_seq_length × embedding_dim
- Maps each position to a learned vector
- Captures position information

**Combined:**
```python
token_embedding = word_embedding[token_id] + pos_embedding[position]
```

**Example (GPT-2 Small):**
- Vocab size: 50,257 tokens
- Embedding dim: 768
- Max sequence: 1,024 tokens
- Word embeddings: 50,257 × 768 = 38.6M parameters
- Position embeddings: 1,024 × 768 = 0.8M parameters
- Total: 39.4M parameters just for embeddings!

These combined embeddings are fed into the transformer layers, which process them to generate predictions.
</details>

---

**Next Lesson:** `03_building_gpt_architecture.md` - Building the transformer!

Run `examples/example_02_word_embeddings.py` to see all concepts in action!
Run `exercises/exercise_02_word_embeddings.py` to practice!

"""
Exercise 01: Similarity Basics
================================

GOAL: Practice computing vector similarity using NumPy.

WHAT YOU WILL BUILD:
  - A cosine_similarity function (from scratch, no shortcuts)
  - A euclidean_distance function
  - A find_most_similar function that searches a collection

INSTRUCTIONS:
  1. Read each TODO comment carefully
  2. Replace the 'pass' or placeholder with your code
  3. Run the file: python exercise_01_similarity_basics.py
  4. Check that all assertions pass (no errors)
  5. Compare your output to the EXPECTED OUTPUTS at the bottom

GLOSSARY (quick reference)
---------------------------
  np.dot(a, b)       -> dot product: sum(a[i] * b[i])
  np.linalg.norm(a)  -> magnitude: sqrt(sum(a[i]^2))
  np.sqrt(x)         -> square root of x
  np.sum(arr)        -> sum all elements
  cosine_similarity  = dot(A, B) / (|A| * |B|)
  euclidean_distance = sqrt( sum( (A[i] - B[i])^2 ) )

LIBRARIES NEEDED: numpy (pip install numpy)
"""

import numpy as np

np.random.seed(42)    # Same random numbers every run

print("=" * 60)
print("EXERCISE 01: Similarity Basics")
print("=" * 60)


# ==============================================================================
# TASK 1: Implement cosine_similarity
# ==============================================================================

print("\n--- Task 1: Cosine Similarity ---")

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two NumPy vectors.

    Formula: dot(A, B) / (|A| * |B|)

    Where:
      dot(A, B) = A[0]*B[0] + A[1]*B[1] + ... (use np.dot)
      |A|       = sqrt(A[0]^2 + A[1]^2 + ...) (use np.linalg.norm)

    Parameters:
      a: numpy array
      b: numpy array (same length as a)

    Returns:
      float between -1 and 1

    Edge case: if either vector has magnitude 0, return 0.0
    """
    # TODO 1a: Compute the dot product of a and b
    dot = None                     # Replace None with your code

    # TODO 1b: Compute the magnitude (norm) of a
    mag_a = None                   # Replace None with your code

    # TODO 1c: Compute the magnitude (norm) of b
    mag_b = None                   # Replace None with your code

    # TODO 1d: Handle the edge case: if mag_a or mag_b is 0, return 0.0
    # Hint: use an if statement

    # TODO 1e: Return the cosine similarity
    return None                    # Replace None with your code

# Test your implementation
v1 = np.array([1.0, 0.0])         # Points right
v2 = np.array([1.0, 0.0])         # Also points right (identical)
v3 = np.array([0.0, 1.0])         # Points up (perpendicular to v1)
v4 = np.array([-1.0, 0.0])        # Points left (opposite of v1)
v5 = np.array([2.0, 0.0])         # Points right, twice as long

# Expected: 1.0 (identical direction)
sim_v1_v2 = cosine_similarity(v1, v2)
print(f"  cosine_similarity([1,0], [1,0])   = {sim_v1_v2:.4f}  (expected 1.0000)")

# Expected: 0.0 (perpendicular)
sim_v1_v3 = cosine_similarity(v1, v3)
print(f"  cosine_similarity([1,0], [0,1])   = {sim_v1_v3:.4f}  (expected 0.0000)")

# Expected: -1.0 (opposite direction)
sim_v1_v4 = cosine_similarity(v1, v4)
print(f"  cosine_similarity([1,0], [-1,0])  = {sim_v1_v4:.4f}  (expected -1.0000)")

# Expected: 1.0 (same direction, different length -- cosine ignores length)
sim_v1_v5 = cosine_similarity(v1, v5)
print(f"  cosine_similarity([1,0], [2,0])   = {sim_v1_v5:.4f}  (expected 1.0000)")

# Assertions: these will raise AssertionError if your implementation is wrong
assert abs(sim_v1_v2 - 1.0)  < 0.001, "v1 vs v2 should be 1.0"
assert abs(sim_v1_v3 - 0.0)  < 0.001, "v1 vs v3 should be 0.0"
assert abs(sim_v1_v4 - (-1.0)) < 0.001, "v1 vs v4 should be -1.0"
assert abs(sim_v1_v5 - 1.0)  < 0.001, "v1 vs v5 should be 1.0"
print("  Task 1: All assertions passed!")


# ==============================================================================
# TASK 2: Implement euclidean_distance
# ==============================================================================

print("\n--- Task 2: Euclidean Distance ---")

def euclidean_distance(a, b):
    """
    Compute the Euclidean distance between two NumPy vectors.

    Formula: sqrt( sum( (A[i] - B[i])^2 for each dimension i ) )

    Steps:
      1. Compute the difference: diff = a - b
      2. Square each element: squared = diff ** 2
      3. Sum all squared elements: total = np.sum(squared)
      4. Take the square root: np.sqrt(total)

    Parameters:
      a: numpy array
      b: numpy array (same length as a)

    Returns:
      float >= 0.0 (always non-negative)
    """
    # TODO 2a: Compute the element-wise difference (a - b)
    diff = None                    # Replace None with your code

    # TODO 2b: Square each element of diff
    squared = None                 # Replace None with your code

    # TODO 2c: Sum all elements of squared
    total = None                   # Replace None with your code

    # TODO 2d: Return the square root of total
    return None                    # Replace None with your code

# Test cases
p1 = np.array([0.0, 0.0])         # Origin
p2 = np.array([3.0, 4.0])         # 3-4-5 triangle

# Expected: 5.0 (Pythagorean theorem: sqrt(9+16) = sqrt(25) = 5)
dist_12 = euclidean_distance(p1, p2)
print(f"  euclidean_distance([0,0], [3,4]) = {dist_12:.4f}  (expected 5.0000)")

# Expected: 0.0 (identical points)
dist_same = euclidean_distance(p1, p1)
print(f"  euclidean_distance([0,0], [0,0]) = {dist_same:.4f}  (expected 0.0000)")

# Expected: 1.0 (unit step along x-axis)
p3 = np.array([1.0, 0.0])
p4 = np.array([2.0, 0.0])
dist_unit = euclidean_distance(p3, p4)
print(f"  euclidean_distance([1,0], [2,0]) = {dist_unit:.4f}  (expected 1.0000)")

assert abs(dist_12   - 5.0) < 0.001, "Pythagorean distance should be 5.0"
assert abs(dist_same - 0.0) < 0.001, "Same point should have distance 0"
assert abs(dist_unit - 1.0) < 0.001, "Unit step distance should be 1.0"
print("  Task 2: All assertions passed!")


# ==============================================================================
# TASK 3: Implement find_most_similar
# ==============================================================================

print("\n--- Task 3: Find the Most Similar Document ---")

def find_most_similar(query_vector, document_vectors, top_k=3):
    """
    Find the top_k most similar document vectors to the query_vector.

    Use COSINE SIMILARITY (call your cosine_similarity function from Task 1).

    Parameters:
      query_vector:     numpy array (the search query)
      document_vectors: list of numpy arrays (the documents to search)
      top_k:            how many results to return

    Returns:
      list of (similarity, index) tuples, sorted by similarity DESCENDING
      (most similar first, least similar last)

    Example:
      If document 2 has similarity 0.95 and document 0 has similarity 0.72:
      -> return [(0.95, 2), (0.72, 0), ...]
    """
    # TODO 3a: Compute cosine_similarity(query_vector, doc) for each document
    # Hint: use a for loop with enumerate to get both the index and the document
    similarities = []              # Will hold (similarity, index) pairs

    for idx, doc_vector in enumerate(document_vectors):
        sim = None                 # TODO: compute cosine similarity between query_vector and doc_vector
        similarities.append((sim, idx))

    # TODO 3b: Sort similarities in DESCENDING order (highest similarity first)
    # Hint: use list.sort() with key=lambda x: x[0] and reverse=True
    # YOUR CODE HERE

    # TODO 3c: Return only the top_k results
    return None                    # Replace None with your code

# Test with a small collection
collection_vectors = [
    np.array([0.9, 0.1, 0.0, 0.0]),    # Doc 0: "animal" topic
    np.array([0.8, 0.2, 0.0, 0.0]),    # Doc 1: "animal" topic
    np.array([0.0, 0.0, 0.9, 0.1]),    # Doc 2: "technology" topic
    np.array([0.0, 0.0, 0.8, 0.2]),    # Doc 3: "technology" topic
    np.array([0.0, 0.9, 0.0, 0.1]),    # Doc 4: "food" topic
]

doc_labels = ["dog article", "cat article", "laptop review", "coding tutorial", "recipe"]

query = np.array([0.85, 0.15, 0.0, 0.0])    # Query: looking for "animal" content

top3 = find_most_similar(query, collection_vectors, top_k=3)

print(f"  Query: animal-focused vector")
print(f"  Top 3 most similar:")
for sim, idx in top3:
    print(f"    similarity={sim:.4f}  [{idx}] {doc_labels[idx]}")

# The top 2 results should be "dog article" (0) and "cat article" (1)
assert top3[0][1] in [0, 1], "First result should be an animal article (index 0 or 1)"
assert top3[1][1] in [0, 1], "Second result should be an animal article (index 0 or 1)"
print("  Task 3: Assertions passed!")


# ==============================================================================
# TASK 4: Similarity Matrix
# ==============================================================================

print("\n--- Task 4: Similarity Matrix ---")

print("""
A similarity matrix shows the similarity between EVERY PAIR of documents.
Row i, column j = similarity between document i and document j.
The diagonal (i==j) is always 1.0 (a document is identical to itself).

Your job: fill in the similarity matrix using your cosine_similarity function.
""")

word_vectors = {
    "king":   np.array([0.9, 0.1, 0.5, 0.8]),
    "queen":  np.array([0.85, 0.2, 0.5, 0.75]),
    "man":    np.array([0.8, 0.1, 0.1, 0.3]),
    "woman":  np.array([0.75, 0.2, 0.1, 0.25]),
    "car":    np.array([0.1, 0.9, 0.0, 0.1]),
    "truck":  np.array([0.05, 0.85, 0.0, 0.05]),
}

words = list(word_vectors.keys())    # ["king", "queen", "man", "woman", "car", "truck"]
n = len(words)

# TODO 4: Fill in the similarity matrix
# matrix[i][j] = cosine_similarity between words[i] and words[j]
matrix = np.zeros((n, n))    # Start with an n x n matrix of zeros

for i in range(n):
    for j in range(n):
        # TODO: compute cosine_similarity(word_vectors[words[i]], word_vectors[words[j]])
        # and store it in matrix[i][j]
        pass                 # Replace 'pass' with your code

# Print the matrix
print(f"  Similarity matrix ({n}x{n}):")
print(f"  {'':8s}", end="")
for w in words:
    print(f"  {w:6s}", end="")
print()

for i, w1 in enumerate(words):
    print(f"  {w1:8s}", end="")
    for j in range(n):
        print(f"  {matrix[i][j]:.3f}", end="")
    print()

print()

# Check: diagonal should be 1.0 (a word compared to itself)
for i in range(n):
    assert abs(matrix[i][i] - 1.0) < 0.001, f"Diagonal [{i},{i}] should be 1.0"

# Check: king and queen should be more similar to each other than king and car
assert matrix[0][1] > matrix[0][4], "king and queen should be more similar than king and car"

# Check: car and truck should be similar
assert matrix[4][5] > 0.8, "car and truck should have similarity > 0.8"

print("  Task 4: All assertions passed!")
print()
print("  Expected pattern: king/queen similar, car/truck similar, but king vs car different")


# ==============================================================================
# CHALLENGE (Optional): Normalize Then Search
# ==============================================================================

print("\n--- Challenge Task: Normalize + Dot Product ---")

print("""
OPTIONAL CHALLENGE:

When vectors are NORMALIZED (magnitude = 1.0),
the dot product equals the cosine similarity.

This is a common optimization: normalize all vectors once, upfront.
Then searching is just a dot product (faster than computing magnitudes every time).

Challenge:
  1. Implement normalize(v) that returns v divided by its magnitude
  2. Verify that: dot(normalize(a), normalize(b)) == cosine_similarity(a, b)
  3. Implement fast_search using dot product on pre-normalized vectors
""")

def normalize(v):
    """
    Normalize a vector to have magnitude 1.0.
    Formula: v / |v|
    """
    # TODO CHALLENGE: return v divided by its magnitude (np.linalg.norm(v))
    # Edge case: if magnitude is 0, return v unchanged
    pass    # Replace 'pass' with your code

def fast_search(query, normalized_collection):
    """
    Fast similarity search using dot product (requires pre-normalized vectors).
    Returns: list of (similarity, index) sorted descending
    """
    # TODO CHALLENGE:
    # 1. Normalize the query
    # 2. For each doc in normalized_collection, compute np.dot(query_norm, doc)
    # 3. Sort by dot product (descending) and return
    pass    # Replace 'pass' with your code

# Test (uncomment once you implement the functions above)
# a = np.array([3.0, 4.0])
# a_norm = normalize(a)
# print(f"  Original magnitude: {np.linalg.norm(a):.4f}")
# print(f"  Normalized magnitude: {np.linalg.norm(a_norm):.4f}  (should be 1.0)")
# print()
# v1_test = np.array([1.0, 2.0, 3.0])
# v2_test = np.array([4.0, 5.0, 6.0])
# cos_sim  = cosine_similarity(v1_test, v2_test)
# dot_sim  = np.dot(normalize(v1_test), normalize(v2_test))
# print(f"  cosine_similarity: {cos_sim:.6f}")
# print(f"  dot after normalize: {dot_sim:.6f}")
# print(f"  Match: {abs(cos_sim - dot_sim) < 0.0001}")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("EXERCISE 01 COMPLETE")
print("=" * 60)

print("""
WHAT YOU PRACTICED:

1. cosine_similarity(a, b) = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
   Range: -1 to 1. For text: 0 to 1. Higher = more similar.

2. euclidean_distance(a, b) = np.sqrt(np.sum((a - b) ** 2))
   Range: 0 to infinity. Lower = more similar.

3. find_most_similar: loop through docs, compute similarity, sort descending.

4. Similarity matrix: compute pairwise similarities between all documents.

KEY INSIGHT:
  Cosine similarity measures DIRECTION (meaning).
  Euclidean distance measures DISTANCE (position in space).
  For text embeddings: use cosine similarity.
""")

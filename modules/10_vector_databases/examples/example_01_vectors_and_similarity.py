"""
Example 01: Vectors and Similarity
===================================

GLOSSARY
--------
Vector:
  A list of numbers. Example: [0.5, 0.3, 0.8]
  In C#: float[] { 0.5f, 0.3f, 0.8f }
  Vectors represent "meaning" as numbers. Similar meanings -> similar numbers.

Dimension:
  How many numbers are in the vector.
  [0.5, 0.3, 0.8] has 3 dimensions.
  Real LLM vectors have 384 to 1536 dimensions.

Magnitude (Norm):
  The "length" of a vector.
  Calculated: sqrt(x1^2 + x2^2 + ... + xn^2)
  Like measuring the length of a line from the origin (0,0) to point (x,y).

Euclidean Distance:
  Straight-line distance between two vectors.
  Lower = more similar. 0 = identical.
  Formula: sqrt( sum of (a_i - b_i)^2 )

Dot Product:
  Multiply matching elements and sum them.
  [1, 2] dot [3, 4] = (1*3) + (2*4) = 11
  In C#: v1.Zip(v2, (a, b) => a * b).Sum()

Cosine Similarity:
  The cosine of the angle between two vectors.
  Range: -1 to 1. For text: 0 to 1.
  1.0 = same direction (identical meaning)
  0.0 = perpendicular (unrelated)
  Formula: dot(A, B) / (|A| * |B|)
  WHY USE IT: Not affected by vector length. Best choice for text.

WHAT THIS EXAMPLE SHOWS
------------------------
Part 1: Vectors as plain Python lists (no libraries)
Part 2: The same operations using NumPy (shorter, faster)
Part 3: Euclidean distance - measuring how far apart vectors are
Part 4: Cosine similarity - measuring how similar vectors are
Part 5: A tiny document search using cosine similarity
Part 6: Visualization of vectors and similarity

LIBRARIES NEEDED
-----------------
  numpy      (pip install numpy)      - for vector math
  matplotlib (pip install matplotlib) - for charts

No external AI libraries needed for this example.
"""

import numpy as np               # NumPy for vector math (like System.Numerics in C#)
import matplotlib.pyplot as plt  # For creating charts

# Seed for reproducibility (same results every run)
np.random.seed(42)

print("=" * 65)
print("EXAMPLE 01: Vectors and Similarity")
print("=" * 65)


# ==============================================================================
# PART 1: Vectors as Plain Python Lists
# ==============================================================================

print("\n" + "=" * 65)
print("PART 1: What Is a Vector? (Plain Python)")
print("=" * 65)

print("""
A vector is simply a list of numbers.

  In Python:       my_vector = [0.5, 0.3, 0.8]
  In C#:           float[] myVector = { 0.5f, 0.3f, 0.8f };

That is it. The key idea is that we use these numbers to represent
the "meaning" of something (a word, a sentence, an image).

Similar things -> similar numbers -> vectors that are "close" together.
""")

# Three simple 3-dimensional vectors representing color (just an example)
# Imagine these represent: "how red", "how green", "how blue" a color is
red_color   = [1.0, 0.0, 0.0]    # Red: very red, no green, no blue
orange_color = [1.0, 0.5, 0.0]   # Orange: very red, half green, no blue
blue_color  = [0.0, 0.0, 1.0]    # Blue: no red, no green, very blue

print("Example: Colors as 3D vectors (R, G, B)")
print(f"  Red:    {red_color}")
print(f"  Orange: {orange_color}")
print(f"  Blue:   {blue_color}")

print("""
Intuitively:
  Red and Orange are similar (both have high R, low B)
  Red and Blue are different (opposite: high R vs high B)

The vector numbers capture this relationship.
""")

# Manually compute Euclidean distance between red and orange (plain Python)
import math                       # Built-in Python math library

def euclidean_distance_plain(a, b):
    """
    Compute straight-line distance between two vectors.
    Uses only plain Python (no NumPy).
    """
    total = 0.0                   # Start with 0
    for i in range(len(a)):       # Loop through each dimension
        diff = a[i] - b[i]        # Difference at this dimension
        total += diff * diff       # Add the squared difference
    return math.sqrt(total)       # Square root of the sum of squares

dist_red_orange = euclidean_distance_plain(red_color, orange_color)
dist_red_blue   = euclidean_distance_plain(red_color, blue_color)

print("Euclidean Distance (computed with plain Python):")
print(f"  Red to Orange: {dist_red_orange:.4f}  <- small (similar colors)")
print(f"  Red to Blue:   {dist_red_blue:.4f}  <- large (different colors)")
print()


# ==============================================================================
# PART 2: NumPy Vectors
# ==============================================================================

print("=" * 65)
print("PART 2: Vectors Using NumPy (Shorter and Faster)")
print("=" * 65)

print("""
NumPy makes vector math much simpler.
Instead of for-loops, we use vectorized operations (all at once).

  Python list:  [0.5, 0.3, 0.8]
  NumPy array:  np.array([0.5, 0.3, 0.8])

The NumPy array supports math operations that plain lists do not:
  a + b    -> add element by element
  a - b    -> subtract element by element
  a * b    -> multiply element by element
  a ** 2   -> square each element
""")

# Convert to NumPy arrays
a = np.array([1.0, 0.0, 0.0])    # Red (as NumPy array)
b = np.array([1.0, 0.5, 0.0])    # Orange
c = np.array([0.0, 0.0, 1.0])    # Blue

print("NumPy array operations:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  a + b = {a + b}")               # Element-wise addition
print(f"  a - b = {a - b}")               # Element-wise subtraction
print(f"  a * b = {a * b}")               # Element-wise multiplication
print(f"  a ** 2 = {a ** 2}")             # Square each element
print()

# NumPy makes Euclidean distance a one-liner
dist_numpy = np.linalg.norm(a - b)        # linalg.norm = Euclidean distance of the difference
print(f"Euclidean distance (NumPy): Red to Orange = {dist_numpy:.4f}")
print(f"  Same as before: {dist_red_orange:.4f}")
print()


# ==============================================================================
# PART 3: Euclidean Distance in Depth
# ==============================================================================

print("=" * 65)
print("PART 3: Euclidean Distance")
print("=" * 65)

print("""
Euclidean distance = straight-line distance between two points.

In 2D (like a map):
  Point A = (1, 2)
  Point B = (4, 6)
  Distance = sqrt( (4-1)^2 + (6-2)^2 )
           = sqrt( 9 + 16 )
           = sqrt(25)
           = 5.0

In N dimensions (same formula extended):
  Distance = sqrt( sum of (A[i] - B[i])^2 for each dimension i )
""")

def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two NumPy vectors.
    a, b: numpy arrays of the same length.
    Returns: float (the distance, always >= 0)
    """
    diff = a - b                  # Subtract element-wise: [a1-b1, a2-b2, ...]
    squared = diff ** 2           # Square each element: [(a1-b1)^2, (a2-b2)^2, ...]
    total = np.sum(squared)       # Sum all squared differences
    return np.sqrt(total)         # Take the square root

# Test with simple 2D points (easy to visualize)
p1 = np.array([1.0, 2.0])        # Point 1
p2 = np.array([4.0, 6.0])        # Point 2

dist = euclidean_distance(p1, p2)
print(f"2D example: p1={p1}, p2={p2}")
print(f"  Distance = {dist:.4f}  (expected 5.0)")
print()

# Multiple vectors -- find the nearest one to a query
print("Finding the nearest vector to a query (using Euclidean distance):")

# Imagine these are simplified text embeddings (normally they would be 384 numbers)
documents = {
    "dog article":      np.array([0.9, 0.1, 0.1]),    # High on "animal" dimension
    "cat article":      np.array([0.8, 0.2, 0.1]),    # Also high on "animal" dimension
    "cooking recipe":   np.array([0.1, 0.9, 0.1]),    # High on "food" dimension
    "python tutorial":  np.array([0.1, 0.1, 0.9])     # High on "tech" dimension
}

query = np.array([0.85, 0.15, 0.05])    # Query: "I want to read about my pet dog"

print(f"\n  Query vector: {query}")
print(f"  (This query is similar to 'animal' articles)")
print()

distances = {}                            # Dictionary to store results
for name, vec in documents.items():       # Loop through each document
    dist = euclidean_distance(query, vec) # Compute distance from query to this doc
    distances[name] = dist                # Store the distance

# Sort by distance (ascending: most similar first)
sorted_results = sorted(distances.items(), key=lambda x: x[1])

print("  Results (sorted by distance, lower = more similar):")
for name, dist in sorted_results:
    bar = "#" * int((1.0 - min(dist, 1.0)) * 20)   # Simple ASCII bar chart
    print(f"    {dist:.4f}  {name:20s}  [{bar}]")

print()
print("  -> 'dog article' is most similar (lowest distance)")
print("  -> 'python tutorial' is least similar (highest distance)")
print()


# ==============================================================================
# PART 4: Cosine Similarity
# ==============================================================================

print("=" * 65)
print("PART 4: Cosine Similarity (Better for Text)")
print("=" * 65)

print("""
Cosine similarity measures the ANGLE between two vectors, not the distance.

Formula: cosine_sim(A, B) = dot(A, B) / (|A| * |B|)

Where:
  dot(A, B)  = A[0]*B[0] + A[1]*B[1] + ... (dot product)
  |A|        = sqrt(A[0]^2 + A[1]^2 + ...) (magnitude/length of A)
  |B|        = magnitude of B

Result:
  1.0  -> vectors point in the same direction (identical meaning)
  0.0  -> vectors are perpendicular (unrelated)
  -1.0 -> vectors point in opposite directions (opposites)

WHY BETTER THAN EUCLIDEAN FOR TEXT?
  A short document and a long document on the same topic have:
  - Large Euclidean distance (because long vectors are numerically larger)
  - High cosine similarity (because they POINT in the same direction)

  Cosine similarity ignores "how long" the vector is -- only direction matters.
""")

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two NumPy vectors.
    a, b: numpy arrays of the same length.
    Returns: float between -1 and 1 (0 to 1 for text embeddings)
    """
    dot = np.dot(a, b)                    # Step 1: dot product (multiply and sum)
    mag_a = np.linalg.norm(a)             # Step 2: magnitude (length) of a
    mag_b = np.linalg.norm(b)             # Step 3: magnitude (length) of b

    if mag_a == 0 or mag_b == 0:          # Avoid division by zero (zero vector)
        return 0.0                        # No similarity if one vector is all zeros

    return dot / (mag_a * mag_b)          # Step 4: divide dot by product of magnitudes

# Demonstrate the effect of vector length on cosine vs Euclidean
print("Demonstration: Length does NOT affect cosine similarity")
print()

v1 = np.array([1.0, 0.0])                # Short vector pointing right
v2 = np.array([5.0, 0.0])                # Long vector pointing right (same direction!)

cos_sim   = cosine_similarity(v1, v2)
euc_dist  = euclidean_distance(v1, v2)

print(f"  v1 = {v1}  (short, pointing right)")
print(f"  v2 = {v2}  (long, also pointing right)")
print(f"  Cosine similarity: {cos_sim:.4f}  -> 1.0 (identical direction = same meaning)")
print(f"  Euclidean distance: {euc_dist:.4f} -> 4.0 (they ARE far apart in space!)")
print()
print("  -> Cosine says: 'same meaning' (correct for text)")
print("  -> Euclidean says: 'different' (wrong for text of different length)")
print()

# Compute cosine similarity for our document search
print("Finding the nearest vector to a query (using Cosine similarity):")
print(f"\n  Query vector: {query}")
print()

similarities = {}                          # Dictionary to store results
for name, vec in documents.items():        # Loop through each document
    sim = cosine_similarity(query, vec)    # Compute similarity
    similarities[name] = sim              # Store it

# Sort by similarity (descending: most similar first)
sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

print("  Results (sorted by similarity, higher = more similar):")
for name, sim in sorted_sims:
    bar = "#" * int(sim * 20)              # ASCII bar showing similarity
    print(f"    {sim:.4f}  {name:20s}  [{bar}]")

print()
print("  -> Same order as Euclidean in this case")
print("  -> Cosine is more reliable when document lengths vary")
print()


# ==============================================================================
# PART 5: A Tiny Document Search
# ==============================================================================

print("=" * 65)
print("PART 5: Tiny Document Search (NumPy Only)")
print("=" * 65)

print("""
Now we build a tiny search system from scratch.
No ChromaDB needed -- just Python lists and NumPy.

The "embeddings" below are hand-crafted (not real) to keep things simple.
Each dimension represents a topic area:
  Dimension 0: "animals / pets"
  Dimension 1: "food / cooking"
  Dimension 2: "technology / computers"
  Dimension 3: "sports / fitness"

In a real system, an embedding model would create these automatically.
""")

# Tiny document collection
# Each document has text + a hand-crafted 4-dimensional vector
doc_collection = [
    {"id": "d01", "text": "My dog loves to play fetch in the park",
     "vector": np.array([0.9, 0.1, 0.0, 0.3])},

    {"id": "d02", "text": "Best pasta recipe with homemade tomato sauce",
     "vector": np.array([0.0, 0.9, 0.0, 0.1])},

    {"id": "d03", "text": "How to train a neural network in Python",
     "vector": np.array([0.0, 0.0, 0.9, 0.1])},

    {"id": "d04", "text": "My cat keeps knocking things off the table",
     "vector": np.array([0.8, 0.1, 0.0, 0.1])},

    {"id": "d05", "text": "Running a marathon: tips for beginners",
     "vector": np.array([0.1, 0.0, 0.0, 0.9])},

    {"id": "d06", "text": "Grilled chicken salad with lemon dressing",
     "vector": np.array([0.0, 0.8, 0.0, 0.2])},

    {"id": "d07", "text": "Installing Visual Studio Code on Windows",
     "vector": np.array([0.0, 0.0, 0.8, 0.2])},

    {"id": "d08", "text": "My kitten is afraid of the vacuum cleaner",
     "vector": np.array([0.7, 0.1, 0.1, 0.0])},
]

def search(query_vector, collection, top_k=3):
    """
    Search for documents most similar to the query_vector.
    query_vector: numpy array representing the search query
    collection:   list of dicts, each with 'id', 'text', 'vector'
    top_k:        how many results to return
    Returns: sorted list of (similarity, doc) tuples
    """
    results = []                           # Will store (similarity, doc) pairs
    for doc in collection:                 # Check every document
        sim = cosine_similarity(           # Compute cosine similarity
            query_vector, doc["vector"]    # Between query and this document
        )
        results.append((sim, doc))         # Save similarity + document

    # Sort results from highest similarity to lowest
    results.sort(key=lambda x: x[0], reverse=True)

    return results[:top_k]                 # Return only the top k results

# Run some searches
test_queries = [
    {
        "name": "Looking for pet content",
        "vector": np.array([0.85, 0.05, 0.05, 0.05])    # High on "animals"
    },
    {
        "name": "Looking for food content",
        "vector": np.array([0.05, 0.85, 0.05, 0.05])    # High on "food"
    },
    {
        "name": "Looking for tech content",
        "vector": np.array([0.05, 0.05, 0.85, 0.05])    # High on "technology"
    },
    {
        "name": "Mixed: outdoor activity with animals",
        "vector": np.array([0.5, 0.0, 0.0, 0.5])        # Mix of animals + sports
    },
]

for q in test_queries:
    print(f"  Query: {q['name']}")
    print(f"  Query vector: {q['vector']}")
    print()

    top_results = search(q["vector"], doc_collection, top_k=3)

    for rank, (sim, doc) in enumerate(top_results, start=1):   # enumerate adds a counter
        print(f"    Rank {rank}: similarity={sim:.3f}  [{doc['id']}] {doc['text']}")

    print()

print("  -> The search correctly finds semantically related documents")
print("  -> No word matching -- only vector similarity!")
print()


# ==============================================================================
# PART 6: Visualization
# ==============================================================================

print("=" * 65)
print("PART 6: Visualization")
print("=" * 65)

# We will visualize 2D versions to make it easy to see
# In reality, vectors have hundreds of dimensions (impossible to visualize directly)

# 2D vectors for visualization (animal axis vs food axis)
viz_docs = {
    "dog article":    (0.9, 0.1),     # Very animal, not food
    "cat article":    (0.8, 0.2),     # Very animal, slightly food
    "cooking recipe": (0.1, 0.9),     # Not animal, very food
    "cat + cooking":  (0.5, 0.5),     # Mix of both
}

query_2d = (0.75, 0.25)               # Query: looking for animal content

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: 2D vector space
ax1 = axes[0]
colors = ["#E74C3C", "#C0392B", "#2ECC71", "#F39C12"]   # Colors for each doc
for (name, (x, y)), color in zip(viz_docs.items(), colors):
    ax1.scatter(x, y, s=200, color=color, zorder=5)     # Plot the point
    ax1.annotate(name, (x, y),                          # Label the point
                 textcoords="offset points",
                 xytext=(8, 8), fontsize=9)
    # Draw a line from origin (0,0) to the point (shows the "vector" direction)
    ax1.annotate("", xy=(x, y), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

# Plot the query vector
ax1.scatter(query_2d[0], query_2d[1], s=300, color="blue",
            marker="*", zorder=10, label="Query")
ax1.annotate("QUERY", query_2d,
             textcoords="offset points",
             xytext=(8, 8), fontsize=10, fontweight="bold", color="blue")
ax1.annotate("", xy=query_2d, xytext=(0, 0),
             arrowprops=dict(arrowstyle="->", color="blue", lw=2.5))

ax1.set_xlim(-0.1, 1.1)
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel("Dimension 0 (animals)", fontsize=11)
ax1.set_ylabel("Dimension 1 (food)", fontsize=11)
ax1.set_title("2D Vector Space\n(vectors as arrows from origin)",
              fontsize=12, fontweight="bold")
ax1.axhline(0, color="gray", linewidth=0.5)
ax1.axvline(0, color="gray", linewidth=0.5)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=9)

# Plot 2: Similarity bar chart
ax2 = axes[1]
query_vec_2d = np.array(query_2d)
doc_names = list(viz_docs.keys())
doc_sims  = [cosine_similarity(query_vec_2d, np.array(v)) for v in viz_docs.values()]

bar_colors = ["#E74C3C" if s == max(doc_sims) else "#BDC3C7" for s in doc_sims]
bars = ax2.barh(doc_names, doc_sims, color=bar_colors)     # Horizontal bar chart
ax2.set_xlim(0, 1.1)
ax2.set_xlabel("Cosine Similarity (higher = more similar)", fontsize=11)
ax2.set_title("Similarity to Query\n(higher bar = more similar to query)",
              fontsize=12, fontweight="bold")

# Add value labels on bars
for bar, sim in zip(bars, doc_sims):
    ax2.text(sim + 0.02, bar.get_y() + bar.get_height() / 2,
             f"{sim:.3f}", va="center", fontsize=10)

ax2.grid(axis="x", alpha=0.3)

plt.suptitle("Vector Search: Finding Similar Documents",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

print("  -> Chart shows which documents are closest to the query vector")
print("  -> 'dog article' is most similar (closest in direction to query)")
print("  -> 'cooking recipe' is least similar (points in a different direction)")
print()


# ==============================================================================
# SUMMARY
# ==============================================================================

print("=" * 65)
print("SUMMARY - Vectors and Similarity")
print("=" * 65)

print("""
WHAT WE LEARNED:

1. A vector is a list of numbers: [0.5, 0.3, 0.8]
   In C#: float[] { 0.5f, 0.3f, 0.8f }

2. Euclidean Distance: straight-line distance between vectors
   Formula:  sqrt( sum of (a_i - b_i)^2 )
   NumPy:    np.linalg.norm(a - b)
   Use when: vectors have similar magnitudes (e.g., image embeddings)

3. Dot Product: multiply matching elements and sum
   Formula:  sum( a_i * b_i )
   NumPy:    np.dot(a, b)
   Use when: vectors are pre-normalized to length 1

4. Cosine Similarity: angle between vectors (direction only)
   Formula:  dot(A, B) / (|A| * |B|)
   NumPy:    np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
   Use when: text embeddings (almost always -- this is the default choice)

5. We built a tiny search function using only NumPy
   - No ChromaDB needed
   - Loop through documents, compute similarity, sort by score

KEY INSIGHT:
  Cosine similarity measures DIRECTION (meaning) not DISTANCE (magnitude).
  This is why it is preferred for text -- two documents of different length
  on the same topic will have high cosine similarity.

NEXT EXAMPLE (02):
  We will build a tiny embedding model by hand -- turning words into vectors
  using a simple lookup table, then improving with context-aware embeddings.
""")

print("=" * 65)
print("END OF EXAMPLE 01")
print("=" * 65)

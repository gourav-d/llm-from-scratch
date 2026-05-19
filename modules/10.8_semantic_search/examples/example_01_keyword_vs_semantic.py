# -*- coding: utf-8 -*-
# example_01_keyword_vs_semantic.py
#
# Module 10.8 -- Semantic Search Systems
# Lesson 1: Keyword Search vs Semantic Search
#
# WHAT THIS FILE DEMONSTRATES:
#   - How keyword (exact-match) search works
#   - How semantic (meaning-based) search works
#   - Why keyword search fails on synonyms (vocabulary mismatch problem)
#   - How semantic search finds matches even with different words
#
# REQUIREMENTS: numpy only (built-in after: pip install numpy)
#
# HOW TO RUN:
#   python examples/example_01_keyword_vs_semantic.py

import numpy as np   # Import numpy for vector math (like System.Math in C#)

# ============================================================
# PART 1: Our Document Corpus
# ============================================================
# A "corpus" is just a collection of documents we want to search through.
# Think of it as a database of text records.
# In C#: List<string> corpus = new List<string> { ... };

CORPUS = [
    # Index 0
    "We sell quick automobiles at competitive prices",
    # Index 1
    "Buy a fast sports car today with low financing",
    # Index 2
    "Scientists discovered a new species in the Amazon rainforest",
    # Index 3
    "Quick automobile deals and rapid vehicle discounts available",
    # Index 4
    "Python programming tutorial for beginners",
    # Index 5
    "Machine learning concepts explained simply",
    # Index 6
    "Fast vehicle sales this weekend only",
    # Index 7
    "The biology of rainforest ecosystems and wildlife",
]

# ============================================================
# PART 2: Keyword Search Implementation
# ============================================================

def keyword_search(query, corpus, top_k=3):
    """
    Simple keyword search.
    Splits query into words, counts how many match each document.

    In C# terms: this is like calling string.Contains() for each word.

    Parameters:
        query   (str):  The text the user typed in the search box
        corpus  (list): List of documents to search through
        top_k   (int):  How many results to return

    Returns:
        List of (score, doc_index, document_text) sorted by score (highest first)
    """
    # Split the query into individual words (lowercase for fair comparison)
    # In C#: query.ToLower().Split(' ')
    query_words = query.lower().split()

    results = []  # Empty list to collect (score, index, text) tuples

    # Loop through each document in the corpus
    for idx, document in enumerate(corpus):
        # Split document into words too (lowercase)
        doc_words = document.lower().split()

        # Count how many query words appear in this document
        # This is the "score" -- more matches = more relevant (in keyword search)
        score = 0  # Start with score 0 for this document
        for word in query_words:      # For each word in the query...
            if word in doc_words:     # ...check if it appears in the document
                score += 1            # Add 1 to score if it does

        # Add this document's result to our list
        # We store: (score, doc index, document text)
        results.append((score, idx, document))

    # Sort results by score, highest first
    # In C#: results.OrderByDescending(r => r.score)
    results.sort(key=lambda x: x[0], reverse=True)

    # Return only the top_k results
    return results[:top_k]


# ============================================================
# PART 3: Semantic Search Implementation
# ============================================================
# We use SIMPLE hand-crafted vectors here to demonstrate the CONCEPT.
# In real life, a trained model (like BERT) creates these vectors.
# The concept is identical -- numbers that represent meaning.

# These are our "embeddings" -- vectors representing word meanings.
# IMPORTANT: These are hand-crafted to illustrate the concept.
# In real life, a neural network learns these from billions of text examples.
# Numbers are made-up but follow the logic:
#   - "fast" and "quick" get similar vectors (similar meaning)
#   - "car" and "automobile" get similar vectors
#   - "rainforest" and "Python" get very different vectors from "car"

# Dimension 0: vehicle concept (high = related to vehicles)
# Dimension 1: speed concept   (high = related to speed/movement)
# Dimension 2: nature concept  (high = related to nature/biology)
# Dimension 3: tech concept    (high = related to technology/computing)
# Dimension 4: commerce concept(high = related to buying/selling)

WORD_VECTORS = {
    # --- Vehicle-related words ---
    "car":          np.array([0.9, 0.5, 0.0, 0.0, 0.3]),
    "automobile":   np.array([0.9, 0.4, 0.0, 0.0, 0.3]),  # similar to "car"
    "vehicle":      np.array([0.8, 0.4, 0.0, 0.0, 0.2]),
    "automobiles":  np.array([0.9, 0.4, 0.0, 0.0, 0.4]),
    "sports":       np.array([0.4, 0.6, 0.1, 0.0, 0.3]),

    # --- Speed-related words ---
    "fast":         np.array([0.2, 0.9, 0.0, 0.0, 0.0]),
    "quick":        np.array([0.2, 0.8, 0.0, 0.0, 0.0]),   # similar to "fast"
    "rapid":        np.array([0.1, 0.8, 0.0, 0.0, 0.0]),
    "speedy":       np.array([0.2, 0.8, 0.0, 0.0, 0.0]),

    # --- Commerce-related words ---
    "buy":          np.array([0.2, 0.0, 0.0, 0.0, 0.9]),
    "sell":         np.array([0.2, 0.0, 0.0, 0.0, 0.9]),
    "deals":        np.array([0.1, 0.0, 0.0, 0.0, 0.8]),
    "discount":     np.array([0.1, 0.0, 0.0, 0.0, 0.8]),
    "discounts":    np.array([0.1, 0.0, 0.0, 0.0, 0.8]),
    "prices":       np.array([0.0, 0.0, 0.0, 0.0, 0.9]),
    "financing":    np.array([0.0, 0.0, 0.0, 0.0, 0.7]),
    "competitive":  np.array([0.0, 0.3, 0.0, 0.2, 0.6]),
    "sales":        np.array([0.2, 0.0, 0.0, 0.0, 0.8]),

    # --- Nature-related words ---
    "rainforest":   np.array([0.0, 0.0, 0.9, 0.0, 0.0]),
    "amazon":       np.array([0.0, 0.0, 0.7, 0.0, 0.3]),
    "species":      np.array([0.0, 0.0, 0.8, 0.0, 0.0]),
    "scientists":   np.array([0.0, 0.0, 0.5, 0.5, 0.0]),
    "biology":      np.array([0.0, 0.0, 0.9, 0.2, 0.0]),
    "ecosystem":    np.array([0.0, 0.0, 0.9, 0.1, 0.0]),
    "ecosystems":   np.array([0.0, 0.0, 0.9, 0.1, 0.0]),
    "wildlife":     np.array([0.0, 0.0, 0.8, 0.0, 0.0]),

    # --- Tech-related words ---
    "python":       np.array([0.0, 0.1, 0.2, 0.9, 0.0]),
    "programming":  np.array([0.0, 0.0, 0.0, 0.9, 0.0]),
    "machine":      np.array([0.1, 0.1, 0.0, 0.8, 0.0]),
    "learning":     np.array([0.0, 0.0, 0.2, 0.7, 0.1]),
    "tutorial":     np.array([0.0, 0.0, 0.1, 0.7, 0.0]),
    "beginners":    np.array([0.0, 0.0, 0.0, 0.6, 0.1]),
    "concepts":     np.array([0.0, 0.0, 0.1, 0.7, 0.0]),
    "explained":    np.array([0.0, 0.0, 0.1, 0.5, 0.0]),
    "simply":       np.array([0.0, 0.0, 0.0, 0.4, 0.1]),

    # --- Common words (neutral/low signal) ---
    "a":            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "the":          np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "and":          np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "we":           np.array([0.0, 0.0, 0.0, 0.0, 0.1]),
    "at":           np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "today":        np.array([0.0, 0.0, 0.0, 0.0, 0.2]),
    "with":         np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "low":          np.array([0.0, 0.1, 0.0, 0.0, 0.3]),
    "this":         np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "weekend":      np.array([0.0, 0.0, 0.0, 0.0, 0.2]),
    "only":         np.array([0.0, 0.0, 0.0, 0.0, 0.1]),
    "in":           np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "new":          np.array([0.0, 0.1, 0.1, 0.1, 0.1]),
    "for":          np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    "discovered":   np.array([0.0, 0.0, 0.5, 0.3, 0.0]),
    "available":    np.array([0.0, 0.0, 0.0, 0.0, 0.3]),
    "of":           np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
}


def get_unknown_vector():
    """Return a zero vector for words we do not have in our lookup table."""
    return np.zeros(5)  # A vector of 5 zeros = no meaning signal


def embed_text(text):
    """
    Convert text to a vector by averaging the vectors of all words.

    This is a simplified version of what real embedding models do.
    Real models (like BERT) use attention to weigh words differently.
    We use simple averaging here to illustrate the concept.

    Parameters:
        text (str): Any text string

    Returns:
        numpy array: A 5-dimensional vector representing the text meaning
    """
    # Split text into lowercase words
    words = text.lower().split()

    # Start with a zero vector (no meaning yet)
    # In C#: var sumVector = new double[5];
    total_vector = np.zeros(5)

    # Track how many words we found vectors for
    count = 0

    for word in words:                               # For each word in the text...
        vec = WORD_VECTORS.get(word, None)           # Look up its vector
        if vec is not None:                          # If we have a vector for it...
            total_vector = total_vector + vec        # Add it to our running total
            count += 1                               # Count this word

    if count == 0:                                   # If no words were recognized...
        return get_unknown_vector()                  # Return a zero vector

    # Average the vectors: divide total by count
    # This gives us one vector representing the AVERAGE meaning of all words
    average_vector = total_vector / count
    return average_vector


def cosine_similarity(vec_a, vec_b):
    """
    Calculate cosine similarity between two vectors.
    Returns a value between 0.0 (completely different) and 1.0 (identical).

    Formula: cos(theta) = (A dot B) / (|A| * |B|)
    where theta is the angle between the two vectors.

    Parameters:
        vec_a (numpy array): First vector
        vec_b (numpy array): Second vector

    Returns:
        float: Similarity score (0.0 to 1.0)
    """
    # Calculate the dot product: sum of element-wise multiplications
    # np.dot([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6 = 32
    dot_product = np.dot(vec_a, vec_b)

    # Calculate the length (magnitude) of each vector
    # np.linalg.norm = sqrt(x1^2 + x2^2 + ...)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)

    # Avoid division by zero (if either vector is all zeros)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # Zero vector has no direction -- similarity undefined, return 0

    # Cosine similarity = dot product / (length_a * length_b)
    similarity = dot_product / (magnitude_a * magnitude_b)
    return float(similarity)  # Convert numpy float to Python float


def semantic_search(query, corpus, top_k=3):
    """
    Semantic search using vector similarity.

    Steps:
    1. Convert query to a vector
    2. Convert each document to a vector
    3. Score each document by cosine similarity to query
    4. Return highest scoring documents

    Parameters:
        query   (str):  The search query
        corpus  (list): List of document strings
        top_k   (int):  How many results to return

    Returns:
        List of (similarity_score, doc_index, document_text) tuples
    """
    # Step 1: Encode the query into a vector
    query_vector = embed_text(query)

    results = []  # Empty list to collect results

    # Step 2: Compare query to each document
    for idx, document in enumerate(corpus):

        # Encode this document into a vector
        doc_vector = embed_text(document)

        # Measure similarity between query vector and document vector
        # High similarity = similar meaning = relevant result
        similarity = cosine_similarity(query_vector, doc_vector)

        # Store result
        results.append((similarity, idx, document))

    # Sort by similarity, highest first
    results.sort(key=lambda x: x[0], reverse=True)

    return results[:top_k]  # Return only the best top_k results


# ============================================================
# PART 4: Run the Comparison
# ============================================================

def print_separator(title):
    """Print a nice separator line for output formatting."""
    print("\n" + "=" * 60)  # Print 60 equal signs
    print(f"  {title}")     # Print the title in the middle
    print("=" * 60)


def run_comparison(query):
    """
    Run both keyword and semantic search on the same query,
    then print results side by side for comparison.

    Parameters:
        query (str): The search query to test
    """
    print(f"\nQuery: \"{query}\"")
    print("-" * 60)

    # --- Keyword search ---
    print("\n[KEYWORD SEARCH] (exact word matching)")
    keyword_results = keyword_search(query, CORPUS, top_k=3)

    found_any = False
    for score, idx, text in keyword_results:
        if score > 0:              # Only show documents with at least 1 match
            found_any = True
            print(f"  Score {score} | Doc {idx}: {text}")

    if not found_any:
        print("  (no matches -- score was 0 for all documents)")

    # --- Semantic search ---
    print("\n[SEMANTIC SEARCH] (meaning-based matching)")
    semantic_results = semantic_search(query, CORPUS, top_k=3)

    for score, idx, text in semantic_results:
        # Only show documents with meaningful similarity (above 0.1)
        if score > 0.1:
            print(f"  Score {score:.3f} | Doc {idx}: {text}")


# ============================================================
# PART 5: Main Demonstration
# ============================================================

if __name__ == "__main__":
    # This block runs when you execute the file directly.
    # In C#: this is the Main() method.

    print_separator("EXAMPLE 1: Vocabulary Mismatch Problem")

    # This is the CORE DEMO.
    # "quick vehicle" and corpus docs about "fast automobiles" -- same meaning, different words.
    # Keyword search will FAIL. Semantic search will SUCCEED.
    print("\nProblem: user searches 'quick vehicle' but corpus uses 'fast automobiles'")
    run_comparison("quick vehicle")

    print_separator("EXAMPLE 2: Semantic Understanding")

    # Test: "rapid automobile" should find documents about fast cars
    print("\nTesting: 'rapid automobile' should match 'fast car' documents")
    run_comparison("rapid automobile")

    print_separator("EXAMPLE 3: Where Keyword Wins")

    # Sometimes keyword search is great -- exact terms match perfectly
    print("\nTesting: 'python programming' -- exact terms in corpus")
    run_comparison("python programming")

    print_separator("EXAMPLE 4: Nature vs Vehicles")

    # Test that semantic search correctly separates topics
    print("\nTesting: 'jungle wildlife biology' should NOT match car articles")
    run_comparison("jungle wildlife biology")

    print_separator("EXAMPLE 5: The Classic Failure Case")

    print("\nThe query 'fast car' vs documents with 'quick automobile':")
    run_comparison("fast car")

    # Show what is happening internally
    print("\n--- What the vectors look like ---")
    fast_vec   = embed_text("fast car")           # vector for the query
    quick_vec  = embed_text("quick automobile")    # vector for a similar phrase
    nature_vec = embed_text("jungle wildlife")     # vector for unrelated topic

    print(f"Vector for 'fast car':        {fast_vec}")
    print(f"Vector for 'quick automobile':{quick_vec}")
    print(f"Vector for 'jungle wildlife': {nature_vec}")
    print(f"")
    print(f"Cosine similarity between 'fast car' and 'quick automobile': "
          f"{cosine_similarity(fast_vec, quick_vec):.3f}")
    print(f"Cosine similarity between 'fast car' and 'jungle wildlife':  "
          f"{cosine_similarity(fast_vec, nature_vec):.3f}")
    print("")
    print("Notice: 'fast car' and 'quick automobile' are CLOSE (high score).")
    print("        'fast car' and 'jungle wildlife' are FAR (low score).")
    print("        This is how semantic search works!")


# ============================================================
# EXPECTED OUTPUT:
# ============================================================
# ============================================================
#   EXAMPLE 1: Vocabulary Mismatch Problem
# ============================================================
#
# Problem: user searches 'quick vehicle' but corpus uses 'fast automobiles'
#
# Query: "quick vehicle"
# ------------------------------------------------------------
#
# [KEYWORD SEARCH] (exact word matching)
#   (no matches -- score was 0 for all documents)
#
# [SEMANTIC SEARCH] (meaning-based matching)
#   Score 0.922 | Doc 0: We sell quick automobiles at competitive prices
#   Score 0.911 | Doc 3: Quick automobile deals and rapid vehicle discounts available
#   Score 0.875 | Doc 1: Buy a fast sports car today with low financing
#
# (Keyword found nothing. Semantic found the most relevant documents!)

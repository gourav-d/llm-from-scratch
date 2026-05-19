# -*- coding: utf-8 -*-
# example_02_bi_encoder_search.py
#
# Module 10.8 -- Semantic Search Systems
# Lesson 2: Bi-Encoder Search
#
# WHAT THIS FILE DEMONSTRATES:
#   - How a bi-encoder converts text to vectors (embeddings)
#   - Pre-encoding documents offline (simulated)
#   - Encoding a query at search time
#   - Ranking documents by cosine similarity
#   - The speed advantage of pre-encoding documents
#
# REQUIREMENTS: numpy only (Tier 1 - primary)
# PART B needs: pip install sentence-transformers
#
# HOW TO RUN:
#   python examples/example_02_bi_encoder_search.py

import numpy as np    # For vector math
import time           # For measuring how fast each step is

# ============================================================
# PART 1: Our Simple "Encoder" (Simulated Bi-Encoder)
# ============================================================
# In a real bi-encoder, a transformer neural network creates vectors.
# Here we use a LOOKUP TABLE to simulate the same process.
# The concept is identical: text goes in, a vector comes out.
# Both query and document use the SAME encoder function.

# This is our "vocabulary" -- the meaning fingerprint of each word.
# Each word maps to a 6-dimensional vector:
#   Dim 0: animal/biology  (high = related to animals)
#   Dim 1: technology      (high = related to tech/computers)
#   Dim 2: food/cooking    (high = related to food)
#   Dim 3: sport/movement  (high = related to sports)
#   Dim 4: science         (high = related to science/research)
#   Dim 5: culture/art     (high = related to art/music/culture)

WORD_MEANING_TABLE = {
    # --- Animals ---
    "cat":          np.array([0.9, 0.0, 0.0, 0.1, 0.1, 0.0]),
    "cats":         np.array([0.9, 0.0, 0.0, 0.1, 0.1, 0.0]),
    "dog":          np.array([0.9, 0.0, 0.0, 0.2, 0.1, 0.0]),
    "dogs":         np.array([0.9, 0.0, 0.0, 0.2, 0.1, 0.0]),
    "kitten":       np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "puppy":        np.array([0.9, 0.0, 0.0, 0.1, 0.0, 0.1]),
    "feline":       np.array([0.9, 0.0, 0.0, 0.0, 0.1, 0.0]),
    "canine":       np.array([0.8, 0.0, 0.0, 0.2, 0.2, 0.0]),
    "animal":       np.array([0.8, 0.0, 0.0, 0.1, 0.3, 0.0]),
    "animals":      np.array([0.8, 0.0, 0.0, 0.1, 0.3, 0.0]),
    "bird":         np.array([0.8, 0.0, 0.0, 0.2, 0.1, 0.0]),
    "fish":         np.array([0.7, 0.0, 0.3, 0.1, 0.1, 0.0]),
    "pet":          np.array([0.8, 0.0, 0.0, 0.1, 0.0, 0.2]),
    "pets":         np.array([0.8, 0.0, 0.0, 0.1, 0.0, 0.2]),
    "veterinarian": np.array([0.7, 0.0, 0.0, 0.0, 0.4, 0.0]),

    # --- Technology ---
    "computer":     np.array([0.0, 0.9, 0.0, 0.0, 0.2, 0.0]),
    "software":     np.array([0.0, 0.9, 0.0, 0.0, 0.2, 0.0]),
    "code":         np.array([0.0, 0.9, 0.0, 0.0, 0.3, 0.0]),
    "coding":       np.array([0.0, 0.9, 0.0, 0.0, 0.2, 0.0]),
    "program":      np.array([0.0, 0.9, 0.0, 0.0, 0.2, 0.0]),
    "programming":  np.array([0.0, 0.9, 0.0, 0.0, 0.2, 0.0]),
    "python":       np.array([0.0, 0.8, 0.0, 0.0, 0.3, 0.0]),
    "javascript":   np.array([0.0, 0.9, 0.0, 0.0, 0.2, 0.0]),
    "algorithm":    np.array([0.0, 0.8, 0.0, 0.0, 0.4, 0.0]),
    "database":     np.array([0.0, 0.8, 0.0, 0.0, 0.2, 0.0]),
    "internet":     np.array([0.0, 0.8, 0.0, 0.0, 0.1, 0.1]),
    "ai":           np.array([0.0, 0.8, 0.0, 0.0, 0.5, 0.0]),
    "machine":      np.array([0.0, 0.7, 0.0, 0.1, 0.3, 0.0]),
    "neural":       np.array([0.1, 0.6, 0.0, 0.0, 0.6, 0.0]),
    "network":      np.array([0.0, 0.7, 0.0, 0.0, 0.4, 0.0]),

    # --- Food ---
    "cook":         np.array([0.0, 0.0, 0.9, 0.1, 0.0, 0.2]),
    "cooking":      np.array([0.0, 0.0, 0.9, 0.1, 0.0, 0.2]),
    "recipe":       np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.1]),
    "food":         np.array([0.2, 0.0, 0.8, 0.0, 0.0, 0.2]),
    "eat":          np.array([0.1, 0.0, 0.9, 0.1, 0.0, 0.0]),
    "pizza":        np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.1]),
    "bread":        np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0]),
    "restaurant":   np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.4]),
    "kitchen":      np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.1]),
    "ingredient":   np.array([0.0, 0.0, 0.9, 0.0, 0.1, 0.0]),
    "ingredients":  np.array([0.0, 0.0, 0.9, 0.0, 0.1, 0.0]),
    "bake":         np.array([0.0, 0.0, 0.9, 0.1, 0.0, 0.1]),
    "baking":       np.array([0.0, 0.0, 0.9, 0.1, 0.0, 0.1]),
    "meal":         np.array([0.1, 0.0, 0.8, 0.0, 0.0, 0.2]),

    # --- Sports ---
    "run":          np.array([0.1, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "running":      np.array([0.1, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "soccer":       np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.2]),
    "football":     np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.2]),
    "tennis":       np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.1]),
    "swim":         np.array([0.1, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "swimming":     np.array([0.1, 0.0, 0.0, 0.9, 0.0, 0.0]),
    "sport":        np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.2]),
    "sports":       np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.2]),
    "athlete":      np.array([0.1, 0.0, 0.0, 0.9, 0.0, 0.1]),
    "gym":          np.array([0.0, 0.0, 0.0, 0.8, 0.1, 0.0]),
    "exercise":     np.array([0.0, 0.0, 0.0, 0.9, 0.1, 0.0]),

    # --- Science ---
    "physics":      np.array([0.0, 0.2, 0.0, 0.0, 0.9, 0.0]),
    "chemistry":    np.array([0.0, 0.1, 0.0, 0.0, 0.9, 0.0]),
    "biology":      np.array([0.5, 0.0, 0.0, 0.0, 0.8, 0.0]),
    "science":      np.array([0.1, 0.2, 0.0, 0.0, 0.9, 0.0]),
    "research":     np.array([0.0, 0.2, 0.0, 0.0, 0.8, 0.0]),
    "experiment":   np.array([0.0, 0.2, 0.0, 0.0, 0.8, 0.0]),
    "laboratory":   np.array([0.0, 0.2, 0.0, 0.0, 0.9, 0.0]),
    "discovery":    np.array([0.0, 0.1, 0.0, 0.0, 0.8, 0.0]),

    # --- Culture/Art ---
    "music":        np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.9]),
    "art":          np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.9]),
    "painting":     np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.9]),
    "movie":        np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.8]),
    "film":         np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.8]),
    "book":         np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.8]),
    "read":         np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.7]),
    "dance":        np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.8]),
    "theater":      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9]),

    # --- Common words (neutral) ---
    "the":          np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "a":            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "is":           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "are":          np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "of":           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "for":          np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "and":          np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "in":           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "to":           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "with":         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "how":          np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
    "best":         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "top":          np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
    "guide":        np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.1]),
    "tips":         np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.1]),
    "learn":        np.array([0.0, 0.1, 0.0, 0.0, 0.2, 0.1]),
    "learning":     np.array([0.0, 0.1, 0.0, 0.0, 0.2, 0.1]),
    "your":         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "my":           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "care":         np.array([0.1, 0.0, 0.0, 0.0, 0.1, 0.2]),
    "training":     np.array([0.0, 0.1, 0.0, 0.4, 0.2, 0.0]),
    "using":        np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
    "about":        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "different":    np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
    "health":       np.array([0.2, 0.0, 0.1, 0.3, 0.3, 0.0]),
    "diet":         np.array([0.1, 0.0, 0.5, 0.2, 0.2, 0.0]),
}


def encode(text):
    """
    The ENCODER -- converts text into a vector (embedding).

    This is the core function of a bi-encoder.
    The SAME function is used for both queries and documents.

    In a real bi-encoder:
    - text goes through a tokenizer
    - tokens go through transformer layers (BERT)
    - attention layer processes all tokens together
    - output is pooled into one fixed-size vector

    Here we simulate this with a word-level average.

    Parameters:
        text (str): Any text to encode

    Returns:
        numpy array: 6-dimensional vector representing meaning
    """
    words = text.lower().split()      # Split into words

    # Collect vectors for all recognized words
    vectors = []
    for word in words:
        # Look up the word in our table (default: zeros if not found)
        vec = WORD_MEANING_TABLE.get(word, np.zeros(6))
        vectors.append(vec)

    if len(vectors) == 0:             # If no words found at all
        return np.zeros(6)            # Return zero vector

    # Stack all word vectors into a matrix: shape (num_words, 6)
    # np.stack creates a 2D array where each row is one word's vector
    vector_matrix = np.stack(vectors)

    # Average across all words: mean over rows (axis=0)
    # shape (6,) -- one vector representing the whole sentence
    sentence_vector = np.mean(vector_matrix, axis=0)

    return sentence_vector


def cosine_similarity(vec_a, vec_b):
    """
    Cosine similarity between two vectors.
    Returns float from 0.0 (different) to 1.0 (identical direction).
    """
    dot = np.dot(vec_a, vec_b)            # Dot product
    norm_a = np.linalg.norm(vec_a)        # Length of vector a
    norm_b = np.linalg.norm(vec_b)        # Length of vector b
    if norm_a == 0 or norm_b == 0:        # Protect against zero division
        return 0.0
    return float(dot / (norm_a * norm_b)) # Cosine similarity formula


# ============================================================
# PART 2: The Bi-Encoder Search Process
# ============================================================

# Our 10 documents to search through
DOCUMENTS = [
    "Tips for training your dog at home",          # index 0
    "Best cooking recipes for beginners",          # index 1
    "Python programming for machine learning",     # index 2
    "How to care for cats and kittens",            # index 3
    "Soccer training drills and techniques",       # index 4
    "Introduction to neural networks and AI",      # index 5
    "Guide to baking bread at home",               # index 6
    "Swimming and running for fitness",            # index 7
    "Cat and dog health tips",                     # index 8
    "Physics and chemistry research discoveries",  # index 9
]


def build_document_index(documents):
    """
    PRE-ENCODE all documents and store their vectors.
    This happens OFFLINE before any user queries.

    In a real system:
    - This runs once, maybe nightly, when new content is added
    - Results are saved to disk (FAISS index or a vector database)
    - At query time, you load the saved vectors -- no re-encoding needed

    Parameters:
        documents (list): List of document strings

    Returns:
        tuple: (list of vectors, time taken in seconds)
    """
    print("Building document index (offline step)...")

    start_time = time.time()           # Record start time

    document_vectors = []              # List to store one vector per document
    for i, doc in enumerate(documents):
        vec = encode(doc)              # Encode this document to a vector
        document_vectors.append(vec)  # Store the vector
        print(f"  Encoded doc {i}: '{doc[:40]}...' -> {vec.round(2)}")

    end_time = time.time()             # Record end time
    elapsed = end_time - start_time    # How long it took

    print(f"\nIndexed {len(documents)} documents in {elapsed*1000:.1f}ms")
    return document_vectors, elapsed


def search(query, document_vectors, documents, top_k=3):
    """
    Search for documents most similar to the query.

    ONLINE step -- runs at query time.
    Steps:
    1. Encode query (1 encode call)
    2. Compare to all pre-encoded document vectors
    3. Return top_k by cosine similarity

    Parameters:
        query           (str):  The user's search query
        document_vectors(list): Pre-computed document vectors (from build_document_index)
        documents       (list): Original document texts
        top_k           (int):  Number of results to return

    Returns:
        List of (rank, similarity_score, document_text) tuples
    """
    # Step 1: Encode the query (this is the only encode call needed at query time)
    query_vector = encode(query)

    # Step 2: Compare query to all document vectors
    scores = []                                    # List to collect scores
    for idx, doc_vec in enumerate(document_vectors):
        sim = cosine_similarity(query_vector, doc_vec)  # Measure similarity
        scores.append((sim, idx))                       # Store (score, doc index)

    # Step 3: Sort by score descending (most similar first)
    scores.sort(key=lambda x: x[0], reverse=True)

    # Step 4: Build result list
    results = []
    for rank, (score, idx) in enumerate(scores[:top_k], start=1):
        results.append((rank, score, documents[idx]))

    return results


# ============================================================
# PART 3: Main Demonstration
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  BI-ENCODER SEARCH DEMO")
    print("=" * 65)

    # --- OFFLINE PHASE: Build the index ---
    # This is done ONCE before any queries
    print("\n--- OFFLINE PHASE: Encoding documents ---")
    doc_vectors, index_time = build_document_index(DOCUMENTS)

    # --- ONLINE PHASE: Run some queries ---
    print("\n--- ONLINE PHASE: Searching ---")

    # Define test queries
    queries = [
        "feline care and training",          # Should match cats docs (uses "feline" not "cat")
        "deep learning and neural networks", # Should match AI/ML docs
        "healthy eating and meal planning",  # Should match food/cooking docs
        "athletic performance and fitness",  # Should match sports docs
    ]

    for query in queries:
        print(f"\n{'='*65}")

        # Encode the query and time it
        start_time = time.time()
        results = search(query, doc_vectors, DOCUMENTS, top_k=3)
        query_time = (time.time() - start_time) * 1000  # Convert to ms

        print(f"Query: \"{query}\"")
        print(f"Query time: {query_time:.2f}ms (vs ~{index_time*1000:.0f}ms to build index)")
        print(f"\nTop 3 results:")
        for rank, score, doc in results:
            print(f"  Rank {rank} | Similarity: {score:.3f} | {doc}")

    # --- Show the speed advantage ---
    print(f"\n{'='*65}")
    print("SPEED COMPARISON DEMO")
    print("='*65")
    print(f"\nWith {len(DOCUMENTS)} documents:")
    print(f"  Offline indexing: {index_time*1000:.1f}ms  (done ONCE)")
    print(f"  Each query needs: ~1ms  (only 1 encode + {len(DOCUMENTS)} comparisons)")
    print(f"")
    print(f"Imagine 1 MILLION documents:")
    print(f"  Offline indexing: ~10 minutes on CPU (done once)")
    print(f"  Each query: ~30ms (1 encode + fast vector comparisons)")
    print(f"  Without pre-encoding: ~80,000 seconds per query (impossible!)")


# ============================================================
# PART B: Real Implementation using sentence-transformers
# ============================================================
# Remove the # at the start of each line to run this section.
# Requirements: pip install sentence-transformers
#
# from sentence_transformers import SentenceTransformer
# import numpy as np
#
# # Load a real pre-trained bi-encoder model
# # all-MiniLM-L6-v2 is small (22MB) and fast
# model = SentenceTransformer('all-MiniLM-L6-v2')
#
# # Encode all documents (offline step)
# doc_vectors_real = model.encode(DOCUMENTS)    # shape: (10, 384)
# print("Real embeddings shape:", doc_vectors_real.shape)
#
# # Encode a query (online step)
# query = "feline care and training"
# query_vec_real = model.encode([query])[0]     # shape: (384,)
#
# # Compute cosine similarities
# # Use the same cosine_similarity function we wrote above
# real_scores = []
# for idx, doc_vec in enumerate(doc_vectors_real):
#     sim = cosine_similarity(query_vec_real, doc_vec)
#     real_scores.append((sim, idx, DOCUMENTS[idx]))
#
# # Sort and display
# real_scores.sort(reverse=True)
# print(f"\nReal bi-encoder results for: '{query}'")
# for score, idx, doc in real_scores[:3]:
#     print(f"  Score {score:.3f} | {doc}")
#
# # You will see similar rankings to our simplified version,
# # but with much richer 384-dimensional understanding!

# ============================================================
# EXPECTED OUTPUT (partial):
# ============================================================
# Building document index (offline step)...
#   Encoded doc 0: 'Tips for training your dog at home...' -> [0.52 0.05 ...]
#   ...
# Indexed 10 documents in X.Xms
#
# Query: "feline care and training"
# Query time: 0.XYms
# Top 3 results:
#   Rank 1 | Similarity: 0.XXX | How to care for cats and kittens
#   Rank 2 | Similarity: 0.XXX | Cat and dog health tips
#   Rank 3 | Similarity: 0.XXX | Tips for training your dog at home

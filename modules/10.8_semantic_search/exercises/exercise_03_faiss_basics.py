# -*- coding: utf-8 -*-
# exercise_03_faiss_basics.py
#
# Module 10.8 -- Semantic Search Systems
# Exercise 3: Build a Flat Vector Index
#
# TASK:
#   Build a FlatVectorIndex class with two methods:
#     1. add(doc_id, vector)       -- store a vector with its document ID
#     2. search(query_vector, k)   -- return the top-k most similar doc_ids
#
# This is the simplest possible vector index (brute force / exact search).
# It demonstrates the fundamental data structure behind all vector databases.
#
# YOUR JOB:
#   - Implement add() to store vectors
#   - Implement search() to find top-k by cosine similarity
#
# TEST:
#   Build the index with 100 random vectors.
#   Search for top-5 nearest to a random query vector.
#   Verify: the result IDs are valid and scores are in [0, 1].
#
# HOW TO RUN:
#   python exercises/exercise_03_faiss_basics.py

import numpy as np


# ============================================================
# YOUR TASK: Implement the FlatVectorIndex class
# ============================================================

class FlatVectorIndex:
    """
    A flat (brute-force) vector index.

    "Flat" means there is no clever structure -- we store all vectors
    in a plain list and compare the query to EVERY vector at search time.

    This guarantees 100% accurate results (unlike ANN indexes which
    trade a bit of accuracy for speed), but is slow for large datasets.

    In C# terms: this is like a List<(string docId, float[] vector)>
    with a linear scan over all items at query time.

    Usage example:
        index = FlatVectorIndex(dimension=4)
        index.add("doc1", np.array([1.0, 0.5, 0.0, 0.2]))
        index.add("doc2", np.array([0.1, 0.9, 0.3, 0.0]))
        results = index.search(np.array([0.8, 0.4, 0.1, 0.3]), k=1)
        # Returns: [("doc1", 0.99)]  (doc1 is more similar)
    """

    def __init__(self, dimension):
        """
        Initialize the index.

        Parameters:
            dimension (int): The length of each vector.
                             All vectors added must have this exact length.
        """
        # TODO: Store the dimension
        self.dimension = dimension

        # TODO: Create empty storage for vectors and document IDs
        # You need TWO parallel lists:
        #   self.vectors  -- list of numpy arrays (one per document)
        #   self.doc_ids  -- list of doc IDs (string, int, or any type)
        # When you add doc "news-42" with its vector, you store:
        #   self.vectors[i] = the vector
        #   self.doc_ids[i] = "news-42"
        # So index i in both lists always refers to the same document.
        self.vectors = []   # YOUR STORAGE: leave as is, or add any init you need
        self.doc_ids = []   # YOUR STORAGE: leave as is

    def add(self, doc_id, vector):
        """
        Add a document vector to the index.

        Parameters:
            doc_id (any):         Identifier for the document (string, int, etc.)
            vector (numpy array): The embedding vector for this document.
                                  Must have length == self.dimension.

        Raises:
            ValueError: If vector has wrong number of dimensions.

        STEPS TO IMPLEMENT:
        1. Check that len(vector) == self.dimension.
           If not, raise ValueError with a helpful message.
        2. Append doc_id to self.doc_ids
        3. Append a COPY of vector to self.vectors
           (Use vector.copy() to avoid aliasing bugs)
        """
        # TODO: Step 1 -- validate vector length
        # YOUR CODE HERE

        # TODO: Step 2 -- store doc_id
        # YOUR CODE HERE

        # TODO: Step 3 -- store a copy of the vector
        # YOUR CODE HERE
        pass

    def search(self, query_vector, k=5):
        """
        Find the k most similar documents to the query_vector.

        Parameters:
            query_vector (numpy array): The query embedding.
                                        Must have length == self.dimension.
            k            (int):         How many results to return.

        Returns:
            List of (doc_id, similarity_score) tuples,
            sorted by similarity_score descending (most similar first).
            If the index has fewer than k documents, return all of them.

        STEPS TO IMPLEMENT:
        1. If the index is empty, return []
        2. Convert self.vectors to a numpy matrix (use np.array(self.vectors))
           Shape: (num_docs, dimension)
        3. Compute cosine similarity between query_vector and all doc vectors:
           a. dot_products = np.dot(vector_matrix, query_vector)   -- shape: (num_docs,)
           b. doc_norms    = np.linalg.norm(vector_matrix, axis=1) -- shape: (num_docs,)
           c. query_norm   = np.linalg.norm(query_vector)          -- scalar
           d. Handle zeros: np.where(doc_norms == 0, 1.0, doc_norms)
           e. similarities = dot_products / (doc_norms * query_norm)
        4. Find the indices of top-k similarities (use np.argsort)
        5. Build and return result list: [(doc_id, float(score)), ...]

        HINT for step 4:
            sorted_indices = np.argsort(similarities)  # ascending order
            top_k_indices = sorted_indices[::-1][:k]   # reverse for descending, take k
        """
        # TODO: Step 1 -- handle empty index
        if len(self.vectors) == 0:
            return []

        # TODO: Step 2 -- convert to matrix
        # YOUR CODE HERE

        # TODO: Step 3 -- compute cosine similarities
        # a. dot products
        # YOUR CODE HERE
        # b. doc norms
        # YOUR CODE HERE
        # c. query norm
        # YOUR CODE HERE
        # d. handle zeros in doc_norms
        # YOUR CODE HERE
        # e. compute similarities
        similarities = None  # YOUR CODE HERE -- replace None

        # TODO: Step 4 -- get top-k indices
        # YOUR CODE HERE
        top_k_indices = []

        # TODO: Step 5 -- build result list
        results = []
        # YOUR CODE HERE
        return results


# ============================================================
# PROVIDED HELPER: Reference implementation of cosine similarity
# (for verifying your results)
# ============================================================

def reference_cosine(a, b):
    """Reference cosine similarity for test verification."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# TESTS -- Do not modify
# ============================================================

def run_tests():
    """Run automated tests on your FlatVectorIndex."""
    print("\nRunning tests...\n")
    passed = 0
    failed = 0

    dim = 8

    # Test 1: Can add a vector without error
    idx = FlatVectorIndex(dimension=dim)
    try:
        idx.add("doc_a", np.ones(dim))
        print("[PASS] add() works without error")
        passed += 1
    except Exception as ex:
        print(f"[FAIL] add() raised an error: {ex}")
        failed += 1

    # Test 2: add() validates dimension
    idx2 = FlatVectorIndex(dimension=dim)
    try:
        idx2.add("bad_doc", np.ones(dim + 1))    # Wrong dimension
        print("[FAIL] add() should have raised ValueError for wrong dimension")
        failed += 1
    except ValueError:
        print("[PASS] add() correctly raises ValueError for wrong-dimension vector")
        passed += 1
    except Exception as ex:
        print(f"[FAIL] add() raised wrong exception type: {type(ex)}: {ex}")
        failed += 1

    # Test 3: search() returns empty list for empty index
    idx3 = FlatVectorIndex(dimension=dim)
    result = idx3.search(np.ones(dim), k=5)
    if result == []:
        print("[PASS] search() returns [] for empty index")
        passed += 1
    else:
        print(f"[FAIL] search() should return [] for empty index, got: {result}")
        failed += 1

    # Test 4: search() returns correct number of results
    idx4 = FlatVectorIndex(dimension=dim)
    np.random.seed(0)
    for i in range(10):
        idx4.add(f"doc_{i}", np.random.rand(dim))
    results4 = idx4.search(np.random.rand(dim), k=3)
    if len(results4) == 3:
        print("[PASS] search() returns correct k=3 results")
        passed += 1
    else:
        print(f"[FAIL] search() should return 3 results, got {len(results4)}")
        failed += 1

    # Test 5: search() returns correct doc_id format
    if results4:
        doc_id, score = results4[0]
        if isinstance(doc_id, str) and doc_id.startswith("doc_"):
            print(f"[PASS] search() returns (doc_id, score) tuples: ('{doc_id}', {score:.3f})")
            passed += 1
        else:
            print(f"[FAIL] search() result format wrong. Got: {results4[0]}")
            failed += 1

    # Test 6: search() finds the most similar vector correctly
    idx5 = FlatVectorIndex(dimension=4)
    # Add 3 specific vectors
    idx5.add("target",  np.array([1.0, 0.0, 0.0, 0.0]))   # Points along dim 0
    idx5.add("similar", np.array([0.9, 0.1, 0.0, 0.0]))   # Almost same direction
    idx5.add("opposite",np.array([0.0, 0.0, 1.0, 0.0]))   # Different direction

    query = np.array([1.0, 0.0, 0.0, 0.0])    # Same as "target"
    results5 = idx5.search(query, k=1)

    if results5 and results5[0][0] == "target":
        print(f"[PASS] search() correctly identifies most similar vector")
        passed += 1
    else:
        print(f"[FAIL] Expected 'target' as top result, got: {results5}")
        failed += 1

    # Test 7: Scores are in [0, 1] range
    idx6 = FlatVectorIndex(dimension=4)
    np.random.seed(1)
    for i in range(20):
        idx6.add(f"d{i}", np.random.rand(4))
    results6 = idx6.search(np.random.rand(4), k=10)
    all_valid = all(0.0 <= score <= 1.001 for _, score in results6)
    if all_valid:
        print("[PASS] All similarity scores are in [0.0, 1.0] range")
        passed += 1
    else:
        bad = [(d, s) for d, s in results6 if not (0.0 <= s <= 1.001)]
        print(f"[FAIL] Some scores out of range: {bad[:3]}")
        failed += 1

    # Test 8: Scores are sorted descending
    if results6 and len(results6) > 1:
        scores_in_order = [s for _, s in results6]
        is_sorted = all(scores_in_order[i] >= scores_in_order[i+1]
                        for i in range(len(scores_in_order)-1))
        if is_sorted:
            print("[PASS] Results are sorted by score descending")
            passed += 1
        else:
            print(f"[FAIL] Results not sorted: {scores_in_order}")
            failed += 1

    print(f"\nResult: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# MAIN DEMO
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  EXERCISE 3: Build a Flat Vector Index")
    print("=" * 60)

    # Build an index with 100 random vectors
    DIMENSION = 128     # Use 128-dim vectors (real models use 384-768)
    NUM_DOCS = 100      # 100 documents in our test corpus

    print(f"\nBuilding index: {NUM_DOCS} docs, {DIMENSION}-dimensional vectors")

    index = FlatVectorIndex(dimension=DIMENSION)

    np.random.seed(42)              # Fixed seed for reproducible results
    for i in range(NUM_DOCS):
        # Generate a random vector to simulate a document embedding
        doc_vector = np.random.rand(DIMENSION).astype(np.float32)
        # Normalize to unit length (typical in semantic search)
        doc_vector = doc_vector / np.linalg.norm(doc_vector)
        index.add(f"article_{i:03d}", doc_vector)

    print(f"Index built with {len(index.vectors)} vectors")

    # Generate a random query vector
    np.random.seed(999)
    query_vec = np.random.rand(DIMENSION).astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    print(f"\nSearching for top-5 most similar documents...")

    try:
        results = index.search(query_vec, k=5)

        if results and results[0][1] > 0:
            print("\nTop-5 results:")
            for rank, (doc_id, score) in enumerate(results, start=1):
                print(f"  Rank {rank}: {doc_id} -- similarity: {score:.4f}")

            # Verify the top result manually
            top_doc_id = results[0][0]
            top_idx = int(top_doc_id.split("_")[1])    # Extract index from "article_042"
            manual_score = reference_cosine(query_vec, index.vectors[top_idx])
            print(f"\nVerification: Manual cosine for {top_doc_id} = {manual_score:.4f}")
            print(f"             Your index returned:              {results[0][1]:.4f}")
            if abs(manual_score - results[0][1]) < 0.001:
                print("  Scores match -- your implementation is correct!")
            else:
                print("  Scores differ -- check your cosine similarity formula")
        else:
            print("\nNOTE: search() returned empty or all-zero results.")
            print("Fill in the TODO sections above, then re-run.")
    except Exception as ex:
        print(f"\nError during search: {ex}")
        print("Fill in the TODO sections above.")

    # Run all tests
    print("\n" + "=" * 60)
    all_passed = run_tests()
    if all_passed:
        print("\nAll tests passed! Your FlatVectorIndex is working correctly.")
    else:
        print("\nSome tests failed. Review the TODO comments and try again.")


# ============================================================
# SOLUTION (Hidden -- Try yourself first!)
# ============================================================
# def __init__(self, dimension):
#     self.dimension = dimension
#     self.vectors = []
#     self.doc_ids = []
#
# def add(self, doc_id, vector):
#     if len(vector) != self.dimension:
#         raise ValueError(
#             f"Vector has {len(vector)} dims, expected {self.dimension}"
#         )
#     self.doc_ids.append(doc_id)
#     self.vectors.append(vector.copy())
#
# def search(self, query_vector, k=5):
#     if len(self.vectors) == 0:
#         return []
#     vector_matrix = np.array(self.vectors)
#     dot_products = np.dot(vector_matrix, query_vector)
#     doc_norms = np.linalg.norm(vector_matrix, axis=1)
#     query_norm = np.linalg.norm(query_vector)
#     doc_norms = np.where(doc_norms == 0, 1.0, doc_norms)
#     if query_norm == 0:
#         query_norm = 1.0
#     similarities = dot_products / (doc_norms * query_norm)
#     top_k_indices = np.argsort(similarities)[::-1][:k]
#     results = []
#     for idx in top_k_indices:
#         results.append((self.doc_ids[idx], float(similarities[idx])))
#     return results

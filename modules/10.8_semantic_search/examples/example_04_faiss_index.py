# -*- coding: utf-8 -*-
# example_04_faiss_index.py
#
# Module 10.8 -- Semantic Search Systems
# Lesson 4: FAISS and Approximate Nearest Neighbor (ANN) Search
#
# WHAT THIS FILE DEMONSTRATES:
#   - Flat (brute force) index: exact search, compares query to ALL vectors
#   - Simplified HNSW-inspired index: fast approximate search
#   - Benchmark: compare speed of flat search vs HNSW-style on 1000 vectors
#   - Show the speed difference grows with corpus size
#
# REQUIREMENTS: numpy only (Tier 1 - primary)
# PART B needs: pip install faiss-cpu
#
# HOW TO RUN:
#   python examples/example_04_faiss_index.py

import numpy as np    # For vector math and random number generation
import time           # For timing how long each search takes


# ============================================================
# PART 1: Flat Index (Brute Force -- Exact Search)
# ============================================================

class FlatIndex:
    """
    Flat index: stores all vectors and compares the query to EVERY vector.

    This is the simplest possible approach.
    It guarantees exact results (100% recall) but is slow for large datasets.

    In C# terms: this is like a List<float[]> with a linear scan.
    No clever data structure -- just a list and a loop.

    Time complexity: O(N) per query (doubles when corpus doubles)
    Space complexity: O(N * dimension) to store all vectors
    """

    def __init__(self, dimension):
        """
        Initialize the flat index.

        Parameters:
            dimension (int): Length of each vector (e.g., 384 for MiniLM)
        """
        self.dimension = dimension          # Store the vector dimension
        self.vectors = []                   # List to hold all stored vectors
        self.doc_ids = []                   # Parallel list of document IDs

    def add(self, doc_id, vector):
        """
        Add a vector to the index.

        Parameters:
            doc_id (any):         Identifier for the document (string, int, etc.)
            vector (numpy array): The embedding vector for this document
        """
        # Validate that the vector has the right length
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector has {len(vector)} dimensions, expected {self.dimension}"
            )

        self.doc_ids.append(doc_id)         # Store the document ID
        self.vectors.append(vector.copy())  # Store a copy of the vector

    def search(self, query_vector, k=5):
        """
        Find the k most similar documents to the query vector.
        This is EXACT search -- checks every single stored vector.

        Parameters:
            query_vector (numpy array): The query embedding
            k            (int):         How many results to return

        Returns:
            List of (similarity_score, doc_id) tuples, sorted by score descending
        """
        if len(self.vectors) == 0:          # Check we have vectors to search
            return []

        # Convert our list of vectors to a matrix for efficient numpy operations
        # Shape: (num_docs, dimension)
        vector_matrix = np.array(self.vectors)

        # Compute cosine similarity between query and ALL stored vectors at once
        # Step 1: Dot product of query with every document vector
        # np.dot(matrix, vector) = for each row in matrix: sum(row * vector)
        # Result shape: (num_docs,) -- one score per document
        dot_products = np.dot(vector_matrix, query_vector)

        # Step 2: Norms (lengths) for cosine similarity formula
        # np.linalg.norm(matrix, axis=1) = length of each row vector
        doc_norms = np.linalg.norm(vector_matrix, axis=1)  # shape: (num_docs,)
        query_norm = np.linalg.norm(query_vector)           # scalar

        # Step 3: Cosine similarity = dot_product / (doc_norm * query_norm)
        # Avoid division by zero by setting norms of 0 to 1 (they score 0 anyway)
        doc_norms = np.where(doc_norms == 0, 1.0, doc_norms)
        if query_norm == 0:
            query_norm = 1.0

        similarities = dot_products / (doc_norms * query_norm)  # shape: (num_docs,)

        # Step 4: Get indices of top-k scores
        # np.argsort returns indices that would sort the array (ascending by default)
        # [::-1] reverses to get descending order
        # [:k] takes only the top k
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Step 5: Build result list
        results = []
        for idx in top_k_indices:
            score = float(similarities[idx])    # Score as Python float
            doc_id = self.doc_ids[idx]          # Get the document ID
            results.append((score, doc_id))

        return results


# ============================================================
# PART 2: Simplified HNSW-Inspired Index
# ============================================================
# Real HNSW builds a multi-layer graph where:
#   - Layer 2 (top): few nodes, long-range connections (highway)
#   - Layer 1 (mid): moderate density (main roads)
#   - Layer 0 (bottom): all nodes, short connections (local streets)
#
# Here we simulate the KEY IDEA using a two-level structure:
#   - "Coarse" level: sample ~sqrt(N) nodes as "entry points"
#   - "Fine" level: for each coarse entry point, store its neighbors
#
# This is NOT a full HNSW but demonstrates the O(log N) search concept.

class SimpleHNSWIndex:
    """
    Simplified HNSW-inspired index.

    Key idea:
    1. Divide vectors into groups (like HNSW layers)
    2. Keep representatives (centroids) from each group
    3. At search time: first find the best group, then search within it

    This demonstrates WHY HNSW is faster:
    - Instead of checking all N vectors, we check sqrt(N) groups + sqrt(N) vectors
    - Total: 2 * sqrt(N) instead of N
    - For N=1000: 2*32=64 checks instead of 1000 (15x faster)
    - For N=1,000,000: 2*1000=2000 checks instead of 1,000,000 (500x faster)

    Time complexity: approximately O(sqrt(N)) per query
    """

    def __init__(self, dimension, num_groups=None):
        """
        Initialize the HNSW-inspired index.

        Parameters:
            dimension  (int): Vector dimension
            num_groups (int): Number of groups (default: sqrt of number of vectors)
        """
        self.dimension = dimension       # Vector size
        self.num_groups = num_groups     # Number of coarse groups
        self.vectors = []                # All stored vectors
        self.doc_ids = []                # All document IDs
        self.built = False               # Whether the index has been built
        self.group_assignments = None    # Which group each vector belongs to
        self.group_centroids = None      # Center vector of each group

    def add(self, doc_id, vector):
        """
        Add a vector. The index must be rebuilt after adding many vectors.

        Parameters:
            doc_id (any):         Document identifier
            vector (numpy array): The embedding vector
        """
        self.doc_ids.append(doc_id)
        self.vectors.append(vector.copy())
        self.built = False               # Index needs rebuilding after adds

    def build(self):
        """
        Build the index structure after all vectors have been added.

        This is the OFFLINE step (like CREATE INDEX in SQL).
        In real HNSW, this builds the multi-layer graph.
        Here we cluster vectors into groups and compute centroids.
        """
        n = len(self.vectors)            # Number of vectors
        if n == 0:
            return

        # Default: use sqrt(n) groups
        if self.num_groups is None:
            self.num_groups = max(1, int(np.sqrt(n)))  # At least 1 group

        # Convert list of vectors to matrix
        vector_matrix = np.array(self.vectors)   # shape: (n, dimension)

        # --- Simple clustering: assign vectors to groups ---
        # In real HNSW, nodes are randomly assigned to layers.
        # Here we use a simple random assignment with k-means iteration.

        # Step 1: Randomly pick initial group centroids
        np.random.seed(42)               # Set seed for reproducible results
        initial_indices = np.random.choice(n, self.num_groups, replace=False)
        self.group_centroids = vector_matrix[initial_indices].copy()  # shape: (num_groups, dim)

        # Step 2: Iterate to improve centroids (like k-means, 3 iterations)
        for iteration in range(3):       # 3 iterations is enough for demo
            # Assign each vector to its nearest centroid
            assignments = np.zeros(n, dtype=int)
            for i in range(n):
                # Compute distance from this vector to each centroid
                vec = vector_matrix[i]                    # shape: (dim,)
                diffs = self.group_centroids - vec        # shape: (num_groups, dim)
                # L2 distances to each centroid
                distances = np.sum(diffs ** 2, axis=1)   # shape: (num_groups,)
                assignments[i] = np.argmin(distances)    # Assign to nearest

            # Update centroids: mean of all vectors assigned to each group
            new_centroids = np.zeros_like(self.group_centroids)
            for g in range(self.num_groups):
                group_mask = (assignments == g)          # Boolean mask for group g
                if np.any(group_mask):                   # If any vectors in this group
                    new_centroids[g] = vector_matrix[group_mask].mean(axis=0)
                else:
                    new_centroids[g] = self.group_centroids[g]  # Keep old if empty

            self.group_centroids = new_centroids

        # Final assignment after convergence
        self.group_assignments = np.zeros(n, dtype=int)
        for i in range(n):
            vec = vector_matrix[i]
            diffs = self.group_centroids - vec
            distances = np.sum(diffs ** 2, axis=1)
            self.group_assignments[i] = np.argmin(distances)

        self.built = True
        print(f"  HNSW index built: {n} vectors in {self.num_groups} groups "
              f"(avg {n//self.num_groups} vectors/group)")

    def search(self, query_vector, k=5, probe_groups=3):
        """
        Approximate nearest neighbor search.

        Steps:
        1. Find the `probe_groups` closest groups to the query (fast: only num_groups comparisons)
        2. Search within only those groups (fast: small fraction of total vectors)
        3. Return top-k from those groups

        Parameters:
            query_vector  (numpy array): Query embedding
            k             (int):         Number of results to return
            probe_groups  (int):         How many groups to search (more = more accurate, slower)

        Returns:
            List of (similarity_score, doc_id) tuples
        """
        if not self.built:
            self.build()                 # Auto-build if not done

        vector_matrix = np.array(self.vectors)

        # Step 1: Find the probe_groups closest centroids to the query
        # This is the "coarse" search -- much fewer comparisons than full search
        centroid_diffs = self.group_centroids - query_vector  # shape: (num_groups, dim)
        centroid_distances = np.sum(centroid_diffs ** 2, axis=1)  # shape: (num_groups,)
        # Get indices of the probe_groups closest centroids
        nearest_group_indices = np.argsort(centroid_distances)[:probe_groups]

        # Step 2: Search within only those groups
        # Collect all vector indices that belong to the nearest groups
        candidate_indices = []
        for g_idx in nearest_group_indices:
            # Find all vector indices assigned to group g_idx
            group_member_indices = np.where(self.group_assignments == g_idx)[0]
            candidate_indices.extend(group_member_indices.tolist())

        if len(candidate_indices) == 0:
            return []

        # Step 3: Exact search within the candidate set
        candidate_vectors = vector_matrix[candidate_indices]  # shape: (num_candidates, dim)

        # Cosine similarity between query and candidate vectors
        dot_products = np.dot(candidate_vectors, query_vector)
        doc_norms = np.linalg.norm(candidate_vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)

        doc_norms = np.where(doc_norms == 0, 1.0, doc_norms)
        if query_norm == 0:
            query_norm = 1.0

        similarities = dot_products / (doc_norms * query_norm)

        # Top-k within candidates
        local_top_k = np.argsort(similarities)[::-1][:k]

        results = []
        for local_idx in local_top_k:
            global_idx = candidate_indices[local_idx]  # Map back to original index
            score = float(similarities[local_idx])
            doc_id = self.doc_ids[global_idx]
            results.append((score, doc_id))

        return results


# ============================================================
# PART 3: Benchmark -- Flat vs HNSW
# ============================================================

def run_benchmark(num_docs=1000, dimension=128, num_queries=50):
    """
    Build both indexes with the same data, then benchmark query speed.

    Parameters:
        num_docs   (int): Number of documents to index
        dimension  (int): Vector dimension (128 for speed, real systems use 384)
        num_queries(int): Number of test queries to run

    Returns:
        dict: Timing and recall statistics
    """
    print(f"\nGenerating {num_docs} random {dimension}-dim vectors...")

    # Generate random vectors to simulate document embeddings
    np.random.seed(123)                  # Fixed seed for reproducibility
    all_vectors = np.random.rand(num_docs, dimension).astype(np.float32)

    # Generate random query vectors
    query_vectors = np.random.rand(num_queries, dimension).astype(np.float32)

    # --- Build Flat Index ---
    print("Building flat index...")
    flat_idx = FlatIndex(dimension)

    t_flat_build_start = time.time()
    for i in range(num_docs):
        flat_idx.add(f"doc_{i}", all_vectors[i])    # Add each vector
    t_flat_build = time.time() - t_flat_build_start

    # --- Build HNSW Index ---
    print("Building HNSW-inspired index...")
    hnsw_idx = SimpleHNSWIndex(dimension)

    t_hnsw_build_start = time.time()
    for i in range(num_docs):
        hnsw_idx.add(f"doc_{i}", all_vectors[i])    # Add vectors
    hnsw_idx.build()                                # Build the index structure
    t_hnsw_build = time.time() - t_hnsw_build_start

    # --- Benchmark Search Speed ---
    k = 5                                # Return top-5 for each query

    print(f"\nRunning {num_queries} queries...")

    # Time the flat index
    t_flat_search_start = time.time()
    flat_results_all = []
    for q_vec in query_vectors:
        results = flat_idx.search(q_vec, k=k)
        flat_results_all.append(results)
    t_flat_search = time.time() - t_flat_search_start

    # Time the HNSW index
    t_hnsw_search_start = time.time()
    hnsw_results_all = []
    for q_vec in query_vectors:
        results = hnsw_idx.search(q_vec, k=k)
        hnsw_results_all.append(results)
    t_hnsw_search = time.time() - t_hnsw_search_start

    # --- Compute Recall ---
    # Recall: what fraction of the true top-k did HNSW actually find?
    total_recall = 0.0
    for q_idx in range(num_queries):
        flat_top_ids = set(doc_id for _, doc_id in flat_results_all[q_idx])
        hnsw_top_ids = set(doc_id for _, doc_id in hnsw_results_all[q_idx])
        # How many of the true top-k did HNSW find?
        overlap = len(flat_top_ids.intersection(hnsw_top_ids))
        recall = overlap / len(flat_top_ids) if len(flat_top_ids) > 0 else 0.0
        total_recall += recall

    avg_recall = total_recall / num_queries    # Average recall across queries

    return {
        "num_docs": num_docs,
        "dimension": dimension,
        "flat_build_ms": t_flat_build * 1000,
        "hnsw_build_ms": t_hnsw_build * 1000,
        "flat_search_total_ms": t_flat_search * 1000,
        "hnsw_search_total_ms": t_hnsw_search * 1000,
        "flat_per_query_ms": (t_flat_search / num_queries) * 1000,
        "hnsw_per_query_ms": (t_hnsw_search / num_queries) * 1000,
        "speedup": t_flat_search / max(t_hnsw_search, 0.0001),
        "recall": avg_recall,
    }


# ============================================================
# PART 4: Main Demonstration
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("  FAISS / ANN INDEX DEMO")
    print("=" * 65)

    # --- Demo 1: Small corpus, show exact results ---
    print("\n--- DEMO 1: Small corpus (10 vectors) ---")
    print("Building and searching a tiny flat index:")

    dim = 4          # Use 4-dim vectors so we can print them and see them
    flat = FlatIndex(dimension=dim)

    # Add some hand-crafted 4D vectors with clear meanings
    sample_vectors = {
        "sports_doc":    np.array([0.9, 0.1, 0.0, 0.0]),   # Strong sports signal
        "tech_doc":      np.array([0.1, 0.9, 0.0, 0.0]),   # Strong tech signal
        "food_doc":      np.array([0.0, 0.0, 0.9, 0.1]),   # Strong food signal
        "science_doc":   np.array([0.1, 0.3, 0.0, 0.9]),   # Strong science signal
        "sports_tech":   np.array([0.5, 0.5, 0.0, 0.0]),   # Mixed sports + tech
    }

    for doc_id, vec in sample_vectors.items():
        flat.add(doc_id, vec)
        print(f"  Added '{doc_id}': {vec}")

    # Search with a sports-like query
    query = np.array([0.8, 0.2, 0.0, 0.1])   # Mostly sports
    print(f"\nQuery vector (sports-like): {query}")
    print("\nFlat index results:")
    results = flat.search(query, k=3)
    for rank, (score, doc_id) in enumerate(results, start=1):
        print(f"  Rank {rank}: '{doc_id}' -- similarity {score:.4f}")

    # --- Demo 2: Benchmark flat vs HNSW ---
    print("\n--- DEMO 2: Speed Benchmark ---")
    stats = run_benchmark(num_docs=1000, dimension=128, num_queries=50)

    print(f"\n{'='*55}")
    print(f"  BENCHMARK RESULTS ({stats['num_docs']} docs, {stats['dimension']}D)")
    print(f"{'='*55}")
    print(f"  {'Metric':<30} {'Flat':>10} {'HNSW':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Build time (ms)':<30} {stats['flat_build_ms']:>9.1f} {stats['hnsw_build_ms']:>9.1f}")
    print(f"  {'Total search time (ms)':<30} {stats['flat_search_total_ms']:>9.1f} {stats['hnsw_search_total_ms']:>9.1f}")
    print(f"  {'Time per query (ms)':<30} {stats['flat_per_query_ms']:>9.2f} {stats['hnsw_per_query_ms']:>9.2f}")
    print(f"  {'Speedup':<30} {'1.0x':>10} {stats['speedup']:.1f}x")
    print(f"  {'Recall@5':<30} {'1.000':>10} {stats['recall']:.3f}")
    print(f"{'='*55}")

    print(f"\nInterpretation:")
    print(f"  HNSW is ~{stats['speedup']:.1f}x faster than flat search.")
    print(f"  HNSW recall = {stats['recall']:.3f} ({stats['recall']*100:.1f}% of exact top-5 found).")
    print(f"  Flat search always has 100% recall (exact). HNSW trades ~{(1-stats['recall'])*100:.1f}% recall for speed.")
    print(f"  At 1 million docs, HNSW would be ~{int(stats['speedup']*30)}x+ faster (speedup grows with corpus size).")

    # --- Demo 3: Show how the HNSW groups look ---
    print("\n--- DEMO 3: HNSW Group Structure ---")
    hnsw_demo = SimpleHNSWIndex(dimension=128, num_groups=5)
    np.random.seed(0)
    for i in range(50):
        hnsw_demo.add(f"d{i}", np.random.rand(128))
    hnsw_demo.build()

    print("\nGroup sizes (how many vectors in each group):")
    assignments = hnsw_demo.group_assignments
    for g in range(hnsw_demo.num_groups):
        count = int(np.sum(assignments == g))   # Count vectors in group g
        bar = "#" * count                        # ASCII bar chart
        print(f"  Group {g}: {count:3d} vectors [{bar}]")

    print("\nAt query time:")
    print("  HNSW searches only 1-3 groups (not all 5).")
    print("  This means checking ~10-30 vectors instead of all 50.")
    print("  Speedup increases dramatically as corpus grows to millions.")


# ============================================================
# PART B: Real FAISS Implementation
# ============================================================
# Remove the # at the start of each line to run this section.
# Requirements: pip install faiss-cpu
#
# import faiss
# import numpy as np
# import time
#
# dim = 128                         # Vector dimension
# n = 10000                         # Number of documents
#
# # Generate random vectors (simulate document embeddings)
# np.random.seed(42)
# doc_vecs = np.random.rand(n, dim).astype('float32')   # FAISS needs float32
#
# # --- Method 1: Flat L2 Index (exact search) ---
# index_flat = faiss.IndexFlatL2(dim)    # L2 = Euclidean distance
# index_flat.add(doc_vecs)              # Add all vectors
# print(f"Flat index has {index_flat.ntotal} vectors")
#
# # --- Method 2: HNSW Index (approximate, fast) ---
# M = 32                                # Number of neighbors per layer
# index_hnsw = faiss.IndexHNSWFlat(dim, M)
# index_hnsw.add(doc_vecs)
# print(f"HNSW index has {index_hnsw.ntotal} vectors")
#
# # Search both indexes
# query_vec = np.random.rand(1, dim).astype('float32')
# k = 5
#
# t1 = time.time()
# D_flat, I_flat = index_flat.search(query_vec, k)    # D=distances, I=indices
# t_flat = time.time() - t1
#
# t1 = time.time()
# D_hnsw, I_hnsw = index_hnsw.search(query_vec, k)
# t_hnsw = time.time() - t1
#
# print(f"\nFlat search time:  {t_flat*1000:.3f}ms, top-5 indices: {I_flat[0]}")
# print(f"HNSW search time:  {t_hnsw*1000:.3f}ms, top-5 indices: {I_hnsw[0]}")
# print(f"Same results? {set(I_flat[0].tolist()) == set(I_hnsw[0].tolist())}")

# ============================================================
# EXPECTED OUTPUT (approximate):
# ============================================================
# BENCHMARK RESULTS (1000 docs, 128D)
# Metric                         Flat       HNSW
# Build time (ms)               X.X        XX.X
# Total search time (ms)       XX.X         X.X
# Time per query (ms)           0.XX        0.XX
# Speedup                       1.0x       X.Xx
# Recall@5                     1.000       0.XXX
#
# HNSW is faster than flat search.
# HNSW recall = 0.8-0.99 (depends on dataset and group configuration)

# -*- coding: utf-8 -*-
# searcher.py
#
# Module 10.8 -- Semantic Search Systems
# Project: Mini Search Engine
#
# PURPOSE:
#   The Searcher takes a query, encodes it to a vector,
#   then compares it to all document vectors in the index.
#   Returns the top-k most similar documents (the candidate set).
#
#   This is the BI-ENCODER step of the pipeline:
#   - Fast (pre-encoded docs, only 1 encode call needed)
#   - Approximate (good but not perfect ranking)
#   - Returns top-20 candidates for the Reranker to refine
#
# C# ANALOGY:
#   This is like a SearchService class with a Retrieve() method.
#   It wraps the index and handles all the search math.

import numpy as np    # For vector math


class Searcher:
    """
    Bi-encoder style searcher.

    Takes a query string, encodes it to a vector, then finds the
    most similar document vectors using cosine similarity.

    Usage:
        searcher = Searcher(indexer)
        results = searcher.retrieve("how does gravity work", k=20)
        # Returns: [(doc_id, similarity_score), ...] sorted by score
    """

    def __init__(self, indexer):
        """
        Initialize Searcher with a built index.

        Parameters:
            indexer (Indexer): A loaded/built Indexer instance.
                               Must have indexer.vectors and indexer.doc_ids populated.
        """
        self.indexer = indexer     # Store reference to the indexer

        # Pre-build a matrix from all document vectors for fast batch comparison
        # This converts {doc_id: vector} to a 2D numpy matrix
        # Shape: (num_docs, 10) -- one row per document
        self._build_vector_matrix()

    def _build_vector_matrix(self):
        """
        Pre-build a matrix from all document vectors.

        Instead of looping through documents one by one during search,
        we build a matrix once and use numpy's batch operations.
        This is much faster for many documents.

        After this method:
        - self._matrix: shape (N, 10) -- all document vectors as rows
        - self._matrix_doc_ids: list of doc IDs corresponding to each row
        """
        doc_ids = self.indexer.doc_ids          # Ordered list of doc IDs

        if not doc_ids:
            # Handle empty index gracefully
            self._matrix = np.zeros((0, 10))
            self._matrix_doc_ids = []
            return

        # Build the matrix by stacking all vectors
        rows = []
        id_list = []
        for doc_id in doc_ids:
            vec = self.indexer.vectors.get(doc_id)
            if vec is not None:
                rows.append(vec)
                id_list.append(doc_id)

        # np.stack converts a list of 1D arrays to a 2D matrix
        # e.g., stack([[1,2,3], [4,5,6]]) -> [[1,2,3],[4,5,6]]
        self._matrix = np.stack(rows)           # shape: (N, 10)
        self._matrix_doc_ids = id_list          # parallel list of doc IDs

    def _embed_query(self, query_text):
        """
        Encode a query string to a vector using the same embed() function
        that was used to encode documents.

        We import embed() from indexer.py to ensure we use IDENTICAL encoding.
        This is the key requirement of a bi-encoder: same encoder for both.

        Parameters:
            query_text (str): The user's query

        Returns:
            numpy array: 10-dimensional vector
        """
        # Import embed from indexer module -- SAME function used for documents
        from indexer import embed               # Import at runtime (avoids circular import)
        return embed(query_text)                # Encode and return

    def _cosine_similarity_batch(self, query_vector):
        """
        Compute cosine similarity between query and ALL documents in one batch operation.

        This is much faster than looping because numpy uses optimized C code.

        Parameters:
            query_vector (numpy array): 10-dim query vector

        Returns:
            numpy array: shape (N,) -- cosine similarity of query to each document
        """
        if len(self._matrix) == 0:             # Handle empty index
            return np.array([])

        # Step 1: Dot products of query with all document vectors
        # np.dot(matrix, vector) computes (row_i dot query) for each row i
        # Result shape: (N,) -- one value per document
        dot_products = np.dot(self._matrix, query_vector)

        # Step 2: Length (norm) of each document vector
        # np.linalg.norm(matrix, axis=1) = length of each row
        # Result shape: (N,)
        doc_norms = np.linalg.norm(self._matrix, axis=1)

        # Step 3: Length of the query vector
        # np.linalg.norm(vector) = sqrt(sum of squares) = vector length
        query_norm = np.linalg.norm(query_vector)

        # Step 4: Protect against zero-length vectors (avoid division by zero)
        doc_norms = np.where(doc_norms == 0, 1.0, doc_norms)  # Replace 0 with 1
        if query_norm == 0:
            query_norm = 1.0

        # Step 5: Cosine similarity = dot_product / (doc_norm * query_norm)
        similarities = dot_products / (doc_norms * query_norm)

        return similarities

    def retrieve(self, query, k=20):
        """
        Retrieve the top-k most similar documents for a query.

        Steps:
        1. Encode the query to a vector
        2. Compute cosine similarity with all document vectors (batch)
        3. Find the top-k highest similarity indices
        4. Return (doc_id, score) pairs

        Parameters:
            query (str): The user's search query
            k     (int): Number of candidates to return (default 20)
                         These will be passed to the Reranker for accurate ranking.

        Returns:
            List of (doc_id, similarity_score) tuples, sorted by score descending.
            Example: [("blackhole_01", 0.82), ("relativity_05", 0.71), ...]
        """
        # Step 1: Encode query to vector
        query_vector = self._embed_query(query)

        # Step 2: Compute similarities to all documents
        similarities = self._cosine_similarity_batch(query_vector)

        if len(similarities) == 0:             # Handle empty index
            return []

        # Step 3: Get indices of top-k similar documents
        # np.argsort returns indices that would sort array in ascending order
        # [::-1] reverses to get descending
        # [:k] takes only the first k (most similar)
        sorted_indices = np.argsort(similarities)[::-1][:k]

        # Step 4: Build result list
        results = []
        for idx in sorted_indices:
            doc_id = self._matrix_doc_ids[idx]      # Get doc ID at this index
            score = float(similarities[idx])        # Get similarity score
            results.append((doc_id, score))

        return results

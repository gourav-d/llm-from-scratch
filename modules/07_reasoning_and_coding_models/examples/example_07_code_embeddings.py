"""
Example 07: Code Embeddings & Semantic Code Search

This example demonstrates how to:
1. Create embeddings for code snippets
2. Build a semantic code search engine
3. Find similar code using cosine similarity
4. Detect code duplicates
5. Build a code recommendation system

For .NET developers:
- Embeddings are like converting code to Vector<float> in C#
- Cosine similarity is like calculating the angle between vectors
- Code search is like LINQ queries on vector embeddings

Author: Learn LLM from Scratch
Module: 07 - Reasoning and Coding Models
Lesson: 7 - Code Embeddings
"""

import numpy as np
from typing import List, Tuple, Dict
import ast
from collections import Counter


# ============================================================================
# PART 1: Simple Token-Based Embeddings
# ============================================================================

class SimpleCodeEmbedder:
    """
    Simple code embedder using token averaging.

    In C#, this is like:
    class SimpleCodeEmbedder {
        Dictionary<string, float[]> tokenEmbeddings;
    }
    """

    def __init__(self, embedding_dim=128):
        """
        Initialize embedder with random token embeddings.

        Args:
            embedding_dim: Size of embedding vectors (like Vector<float> in C#)
        """
        self.embedding_dim = embedding_dim
        self.token_embeddings = {}  # Token -> embedding vector
        self.vocabulary = set()

    def build_vocabulary(self, code_samples: List[str]):
        """
        Build vocabulary from code samples.

        This is like building a HashSet<string> of all unique tokens.
        """
        print("Building vocabulary from code samples...")

        for code in code_samples:
            tokens = self._tokenize(code)
            self.vocabulary.update(tokens)

        print(f"Vocabulary size: {len(self.vocabulary)} unique tokens")

        # Initialize random embeddings for each token
        # In C#: var embeddings = new Dictionary<string, float[]>();
        for token in self.vocabulary:
            # Random vector for each token (in real world, these are learned)
            self.token_embeddings[token] = np.random.randn(self.embedding_dim)

            # Normalize to unit length (important for cosine similarity!)
            norm = np.linalg.norm(self.token_embeddings[token])
            if norm > 0:
                self.token_embeddings[token] /= norm

    def _tokenize(self, code: str) -> List[str]:
        """
        Simple tokenization by splitting on whitespace and special chars.

        In C#:
        code.Split(new[] {' ', '(', ')', ':', ','}, StringSplitOptions.RemoveEmptyEntries)
        """
        # Replace special characters with spaces
        for char in "()[]{}:,;=+-*/<>":
            code = code.replace(char, f" {char} ")

        # Split and filter empty tokens
        tokens = [t.strip() for t in code.split() if t.strip()]
        return tokens

    def embed(self, code: str) -> np.ndarray:
        """
        Create embedding for code by averaging token embeddings.

        In C#:
        var embeddings = tokens.Select(t => GetEmbedding(t));
        return embeddings.Average();
        """
        tokens = self._tokenize(code)

        if not tokens:
            return np.zeros(self.embedding_dim)

        # Get embeddings for all tokens
        token_vecs = []
        for token in tokens:
            if token in self.token_embeddings:
                token_vecs.append(self.token_embeddings[token])
            else:
                # Unknown token -> use zero vector
                token_vecs.append(np.zeros(self.embedding_dim))

        # Average all token vectors
        # In C#: var average = vectors.Aggregate((a, b) => a + b) / count;
        if token_vecs:
            code_embedding = np.mean(token_vecs, axis=0)
        else:
            code_embedding = np.zeros(self.embedding_dim)

        # Normalize to unit length
        norm = np.linalg.norm(code_embedding)
        if norm > 0:
            code_embedding /= norm

        return code_embedding


# ============================================================================
# PART 2: Weighted Token Embeddings (Better!)
# ============================================================================

class WeightedCodeEmbedder(SimpleCodeEmbedder):
    """
    Code embedder with weighted tokens.

    Important tokens (like function names, keywords) get higher weight.

    In C#:
    class WeightedCodeEmbedder : SimpleCodeEmbedder {
        Dictionary<string, float> tokenWeights;
    }
    """

    def __init__(self, embedding_dim=128):
        super().__init__(embedding_dim)
        self.token_weights = {}

    def build_vocabulary(self, code_samples: List[str]):
        """Build vocabulary and calculate token importance weights."""
        super().build_vocabulary(code_samples)
        self._calculate_weights(code_samples)

    def _calculate_weights(self, code_samples: List[str]):
        """
        Calculate TF-IDF style weights for tokens.

        Rare tokens (like function names) get higher weight.
        Common tokens (like 'def', 'return') get lower weight.

        In C#:
        var weights = vocabulary.ToDictionary(
            token => token,
            token => CalculateWeight(token, samples)
        );
        """
        # Count token occurrences across all documents
        token_doc_count = Counter()

        for code in code_samples:
            tokens = set(self._tokenize(code))  # Unique tokens in this doc
            for token in tokens:
                token_doc_count[token] += 1

        # Calculate inverse document frequency (IDF)
        # Rare tokens have higher IDF
        num_docs = len(code_samples)

        for token in self.vocabulary:
            doc_freq = token_doc_count.get(token, 1)

            # IDF formula: log(total_docs / doc_frequency)
            idf = np.log(num_docs / doc_freq)

            # Add bonus weight for important token types
            if token in ["def", "class", "return", "import"]:
                idf *= 1.5  # Keywords are important!

            self.token_weights[token] = idf

    def embed(self, code: str) -> np.ndarray:
        """
        Create weighted embedding.

        In C#:
        var weighted = tokens.Select(t => GetEmbedding(t) * GetWeight(t));
        return weighted.Average();
        """
        tokens = self._tokenize(code)

        if not tokens:
            return np.zeros(self.embedding_dim)

        # Get weighted embeddings
        weighted_vecs = []
        total_weight = 0

        for token in tokens:
            if token in self.token_embeddings:
                weight = self.token_weights.get(token, 1.0)
                weighted_vec = self.token_embeddings[token] * weight
                weighted_vecs.append(weighted_vec)
                total_weight += weight

        # Weighted average
        if weighted_vecs and total_weight > 0:
            code_embedding = np.sum(weighted_vecs, axis=0) / total_weight
        else:
            code_embedding = np.zeros(self.embedding_dim)

        # Normalize
        norm = np.linalg.norm(code_embedding)
        if norm > 0:
            code_embedding /= norm

        return code_embedding


# ============================================================================
# PART 3: Similarity Calculations
# ============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Returns value between -1 and 1:
    - 1.0  = identical direction (very similar)
    - 0.0  = perpendicular (unrelated)
    - -1.0 = opposite direction (very different)

    In C#:
    public static float CosineSimilarity(Vector<float> v1, Vector<float> v2) {
        return Vector.Dot(v1, v2) / (v1.Length() * v2.Length());
    }

    Math explanation:
    cosine_similarity = (A · B) / (||A|| × ||B||)

    Where:
    - A · B = dot product (sum of element-wise multiplication)
    - ||A|| = magnitude/length of vector A
    """
    # Dot product: sum of element-wise multiplication
    # In C#: v1.Zip(v2, (a, b) => a * b).Sum()
    dot_product = np.dot(vec1, vec2)

    # Magnitude (length) of each vector
    # In C#: Math.Sqrt(vector.Select(x => x * x).Sum())
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return float(similarity)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between vectors.

    Lower distance = more similar

    In C#:
    public static float EuclideanDistance(Vector<float> v1, Vector<float> v2) {
        return (v1 - v2).Length();
    }

    Formula: √(Σ(a_i - b_i)²)
    """
    # Difference vector
    diff = vec1 - vec2

    # Distance = magnitude of difference
    # In C#: Math.Sqrt(diff.Select(x => x * x).Sum())
    distance = np.linalg.norm(diff)

    return float(distance)


# ============================================================================
# PART 4: Semantic Code Search Engine
# ============================================================================

class CodeSearchEngine:
    """
    Semantic code search engine.

    Like Google, but for code!

    In C#:
    class CodeSearchEngine {
        Dictionary<string, float[]> codeEmbeddings;
        List<string> codebase;
    }
    """

    def __init__(self, embedder):
        """
        Initialize search engine with an embedder.

        Args:
            embedder: Code embedder (SimpleCodeEmbedder or WeightedCodeEmbedder)
        """
        self.embedder = embedder
        self.codebase = []  # All code snippets
        self.embeddings = []  # Corresponding embeddings

    def index_code(self, code_snippets: List[str]):
        """
        Index code snippets for search.

        This is like building a search index in Elasticsearch.

        In C#:
        public void IndexCode(List<string> codeSnippets) {
            foreach (var code in codeSnippets) {
                var embedding = embedder.Embed(code);
                index.Add(code, embedding);
            }
        }
        """
        print(f"Indexing {len(code_snippets)} code snippets...")

        self.codebase = code_snippets

        # Create embeddings for all code
        # In C#: var embeddings = code.Select(c => embedder.Embed(c)).ToList();
        self.embeddings = [
            self.embedder.embed(code) for code in code_snippets
        ]

        print(f"Indexed {len(self.embeddings)} code snippets")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for code similar to query.

        Args:
            query: Code snippet or natural language query
            top_k: Number of results to return

        Returns:
            List of (code, similarity_score) tuples

        In C#:
        public List<(string code, float score)> Search(string query, int topK = 5) {
            var queryEmbedding = embedder.Embed(query);
            var scores = codebase.Select(code =>
                (code, CosineSimilarity(queryEmbedding, embeddings[code]))
            );
            return scores.OrderByDescending(x => x.Item2).Take(topK).ToList();
        }
        """
        if not self.codebase:
            return []

        # Embed query
        query_embedding = self.embedder.embed(query)

        # Calculate similarities with all code in index
        # In C#: var similarities = embeddings.Select(emb =>
        #           CosineSimilarity(queryEmbedding, emb)).ToList();
        similarities = [
            cosine_similarity(query_embedding, emb)
            for emb in self.embeddings
        ]

        # Get top K indices
        # In C#: var topIndices = similarities
        #           .Select((score, idx) => (score, idx))
        #           .OrderByDescending(x => x.score)
        #           .Take(topK)
        #           .Select(x => x.idx);
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return results
        results = [
            (self.codebase[i], similarities[i])
            for i in top_indices
        ]

        return results

    def find_duplicates(self, threshold: float = 0.85) -> List[Tuple[int, int, float]]:
        """
        Find duplicate or near-duplicate code.

        Args:
            threshold: Similarity threshold (0-1)

        Returns:
            List of (index1, index2, similarity) tuples

        In C#:
        public List<(int i, int j, float similarity)> FindDuplicates(float threshold) {
            var duplicates = new List<(int, int, float)>();
            for (int i = 0; i < embeddings.Count; i++) {
                for (int j = i + 1; j < embeddings.Count; j++) {
                    var sim = CosineSimilarity(embeddings[i], embeddings[j]);
                    if (sim > threshold) {
                        duplicates.Add((i, j, sim));
                    }
                }
            }
            return duplicates;
        }
        """
        duplicates = []

        # Compare all pairs
        for i in range(len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                sim = cosine_similarity(self.embeddings[i], self.embeddings[j])

                if sim > threshold:
                    duplicates.append((i, j, sim))

        return duplicates


# ============================================================================
# PART 5: Code Recommendation System
# ============================================================================

class CodeRecommender:
    """
    Code recommendation system.

    Like Netflix recommendations, but for code!

    In C#:
    class CodeRecommender {
        CodeSearchEngine searchEngine;
        Dictionary<string, List<string>> userHistory;
    }
    """

    def __init__(self, search_engine: CodeSearchEngine):
        self.search_engine = search_engine

    def recommend_similar(self, code: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend similar code snippets.

        Args:
            code: Current code being written
            top_k: Number of recommendations

        Returns:
            List of (recommended_code, relevance_score) tuples
        """
        results = self.search_engine.search(code, top_k=top_k + 1)

        # Filter out the exact same code
        # In C#: results.Where(r => r.code != code).Take(top_k)
        recommendations = [
            (code_snippet, score)
            for code_snippet, score in results
            if code_snippet.strip() != code.strip()
        ][:top_k]

        return recommendations

    def find_bug_patterns(self, buggy_code: str, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Find code similar to known buggy patterns.

        Args:
            buggy_code: Known buggy code pattern
            threshold: Similarity threshold

        Returns:
            List of (potentially_buggy_code, similarity) tuples
        """
        # Search for similar code
        results = self.search_engine.search(buggy_code, top_k=100)

        # Filter by threshold
        # In C#: results.Where(r => r.score > threshold).ToList()
        risky_code = [
            (code, score)
            for code, score in results
            if score > threshold
        ]

        return risky_code


# ============================================================================
# PART 6: Demo and Testing
# ============================================================================

def demo_code_embeddings():
    """
    Demonstrate code embeddings and semantic search.
    """
    print("=" * 70)
    print("CODE EMBEDDINGS & SEMANTIC SEARCH DEMO")
    print("=" * 70)
    print()

    # Sample codebase
    codebase = [
        # Addition functions
        "def add(x, y):\n    return x + y",
        "def sum_numbers(a, b):\n    result = a + b\n    return result",
        "def calculate_sum(num1, num2):\n    return num1 + num2",

        # Multiplication functions
        "def multiply(x, y):\n    return x * y",
        "def product(a, b):\n    return a * b",

        # Fibonacci
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",

        # Factorial
        "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",

        # String operations
        "def reverse_string(s):\n    return s[::-1]",
        "def string_reverse(text):\n    return ''.join(reversed(text))",

        # List operations
        "def find_max(numbers):\n    return max(numbers)",
        "def get_maximum(lst):\n    return max(lst)",
    ]

    print(f"Codebase size: {len(codebase)} functions\n")

    # Demo 1: Simple Embedder
    print("-" * 70)
    print("DEMO 1: Simple Token-Based Embeddings")
    print("-" * 70)
    print()

    embedder = SimpleCodeEmbedder(embedding_dim=128)
    embedder.build_vocabulary(codebase)

    code1 = codebase[0]  # def add(x, y): return x + y
    code2 = codebase[1]  # def sum_numbers(a, b): ...

    emb1 = embedder.embed(code1)
    emb2 = embedder.embed(code2)

    similarity = cosine_similarity(emb1, emb2)

    print(f"Code 1: {code1[:40]}...")
    print(f"Code 2: {code2[:40]}...")
    print(f"Embedding dimension: {len(emb1)}")
    print(f"Cosine similarity: {similarity:.4f}")
    print()

    # Demo 2: Weighted Embedder (Better!)
    print("-" * 70)
    print("DEMO 2: Weighted Token Embeddings (Better!)")
    print("-" * 70)
    print()

    weighted_embedder = WeightedCodeEmbedder(embedding_dim=128)
    weighted_embedder.build_vocabulary(codebase)

    emb1_weighted = weighted_embedder.embed(code1)
    emb2_weighted = weighted_embedder.embed(code2)

    similarity_weighted = cosine_similarity(emb1_weighted, emb2_weighted)

    print(f"Weighted cosine similarity: {similarity_weighted:.4f}")
    print("(Should be higher than simple embeddings for similar functions)")
    print()

    # Demo 3: Code Search
    print("-" * 70)
    print("DEMO 3: Semantic Code Search")
    print("-" * 70)
    print()

    search_engine = CodeSearchEngine(weighted_embedder)
    search_engine.index_code(codebase)

    # Search for addition functions
    query = "def add(x, y): return x + y"
    print(f"Query: {query}")
    print()

    results = search_engine.search(query, top_k=5)

    print("Top 5 similar functions:")
    for i, (code, score) in enumerate(results, 1):
        code_preview = code.replace('\n', ' ')[:60]
        print(f"{i}. [Score: {score:.4f}] {code_preview}...")
    print()

    # Demo 4: Natural Language Search
    print("-" * 70)
    print("DEMO 4: Natural Language Code Search")
    print("-" * 70)
    print()

    nl_query = "function that multiplies two numbers"
    print(f"Natural language query: '{nl_query}'")
    print()

    results = search_engine.search(nl_query, top_k=3)

    print("Top 3 results:")
    for i, (code, score) in enumerate(results, 1):
        code_preview = code.replace('\n', ' ')[:60]
        print(f"{i}. [Score: {score:.4f}] {code_preview}...")
    print()

    # Demo 5: Duplicate Detection
    print("-" * 70)
    print("DEMO 5: Duplicate Code Detection")
    print("-" * 70)
    print()

    duplicates = search_engine.find_duplicates(threshold=0.85)

    print(f"Found {len(duplicates)} duplicate/similar code pairs:")
    for idx1, idx2, sim in duplicates:
        code1 = codebase[idx1].replace('\n', ' ')[:50]
        code2 = codebase[idx2].replace('\n', ' ')[:50]
        print(f"\n  Similarity: {sim:.4f}")
        print(f"  Code 1: {code1}...")
        print(f"  Code 2: {code2}...")
    print()

    # Demo 6: Code Recommendations
    print("-" * 70)
    print("DEMO 6: Code Recommendations")
    print("-" * 70)
    print()

    recommender = CodeRecommender(search_engine)

    current_code = "def multiply(x, y):\n    return x * y"
    print(f"Current code: {current_code.replace(chr(10), ' ')}")
    print()

    recommendations = recommender.recommend_similar(current_code, top_k=3)

    print("Recommended similar functions:")
    for i, (code, score) in enumerate(recommendations, 1):
        code_preview = code.replace('\n', ' ')[:60]
        print(f"{i}. [Relevance: {score:.4f}] {code_preview}...")
    print()

    # Demo 7: Similarity Matrix
    print("-" * 70)
    print("DEMO 7: Code Similarity Matrix")
    print("-" * 70)
    print()

    # Calculate all pairwise similarities
    n = min(5, len(codebase))  # Show first 5 functions
    print(f"Similarity matrix for first {n} functions:")
    print()

    # Header
    print("      ", end="")
    for i in range(n):
        print(f"F{i:2d}  ", end="")
    print()

    # Matrix
    for i in range(n):
        print(f"F{i:2d}  ", end="")
        for j in range(n):
            emb_i = search_engine.embeddings[i]
            emb_j = search_engine.embeddings[j]
            sim = cosine_similarity(emb_i, emb_j)
            print(f"{sim:.2f} ", end="")
        print()
    print()

    print("Legend:")
    for i in range(n):
        preview = codebase[i].split('\n')[0][:50]
        print(f"  F{i:2d}: {preview}...")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✅ Created token-based embeddings for code")
    print("✅ Implemented weighted embeddings (better!)")
    print("✅ Built semantic code search engine")
    print("✅ Detected duplicate code")
    print("✅ Built code recommendation system")
    print("✅ Demonstrated natural language code search")
    print()
    print("Key insights:")
    print("  • Similar functions have high cosine similarity (>0.85)")
    print("  • Weighted embeddings give better results")
    print("  • Can search code using natural language!")
    print("  • Can find duplicates automatically")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run the code embeddings demo.

    In C#:
    static void Main(string[] args) {
        DemoCodeEmbeddings();
    }
    """
    demo_code_embeddings()

    print()
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Try adding your own code to the codebase")
    print("2. Experiment with different embedding dimensions")
    print("3. Try different similarity thresholds")
    print("4. Search using natural language queries")
    print("5. Build a code duplicate detector for your projects!")
    print()
    print("In Lesson 8, we'll learn how to TRAIN models on code!")
    print("=" * 70)

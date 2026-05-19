# Lesson 4 -- FAISS and Approximate Nearest Neighbor (ANN) Search

---

## GLOSSARY

| Term                       | Plain English Definition                                                     |
|----------------------------|------------------------------------------------------------------------------|
| ANN                        | Approximate Nearest Neighbor -- find vectors that are "close enough" fast    |
| exact search               | Check every single vector -- guaranteed correct, but slow                    |
| brute force                | The simplest approach: compare query to EVERY document. O(N) time.           |
| HNSW                       | Hierarchical Navigable Small World -- a graph structure for fast ANN search  |
| IVF                        | Inverted File Index -- cluster vectors, search only nearby clusters          |
| FAISS                      | Facebook AI Similarity Search -- a library for fast vector search            |
| index                      | A data structure built from your vectors that makes searching faster         |
| O(N)                       | Linear time -- doubles when data doubles. Slow for large N.                  |
| O(log N)                   | Logarithmic time -- barely increases when data grows. Fast.                  |
| recall@k                   | Of the TRUE top-k results, how many did ANN actually return?                 |
| dimensionality             | The length of each vector (e.g., 384-dimensional means 384 numbers)          |

---

## 1. The Scaling Problem

Let us look at what happens with exact search as your corpus grows:

```
Corpus size       | Time per query (exact search)
------------------|---------------------------------
1,000 docs        | ~1ms   (fast!)
100,000 docs      | ~100ms (still OK)
1,000,000 docs    | ~1 second (slow for production)
1,000,000,000 docs| ~17 minutes per query (completely unusable!)

Formula: time = N x cost_per_comparison
         Each vector comparison = dot product of 384 numbers
         At 1 billion docs: 1,000,000,000 x 384 multiplications
```

This is the problem that FAISS and ANN algorithms solve.

---

## 2. C#/.NET Analogy: Database Index vs Table Scan

You already understand this concept from databases!

```csharp
// EXACT SEARCH = Full Table Scan (no index)
// SELECT * FROM Products WHERE Price = 49.99
// Must check EVERY row. O(N) -- slow.

// ANN SEARCH = Index-based lookup
// SQL Server creates a B-tree index on the Price column.
// Now searches the B-tree instead of all rows. O(log N) -- fast!

// FAISS does the same thing for vectors instead of numbers.
// Instead of a B-tree, it uses graph structures (HNSW) or cluster indexes (IVF).
```

The trade-off is the same too:
- Building the index takes time upfront (like CREATE INDEX)
- The index uses extra storage
- But queries are much faster
- ANN may miss 1-2% of exact top results (approximate) -- like an index with statistics

---

## 3. How HNSW Works

HNSW stands for Hierarchical Navigable Small World.
This sounds intimidating but the idea is simple.

Think of it like a transit map:

```
LAYER 2 (highway -- few stops, big jumps):
  [A] ----long jump---- [F] ----long jump---- [K]

LAYER 1 (main roads -- medium jumps):
  [A] --[B]-- [C] --[D]-- [E] --[F]-- [G] --[H]-- [I] --[J]-- [K]

LAYER 0 (local streets -- all nodes, short jumps):
  [A]-[B]-[C]-[D]-[E]-[F]-[G]-[H]-[I]-[J]-[K]-[L]-[M]-[N]-[O]

Search for a query vector (closest to node H):
  Start at LAYER 2: [A] --> is [F] closer? Yes, jump to [F]. Can we do better? No.
  Drop to LAYER 1: At [F] --> is [G] closer? Yes. --> is [H] closer? Yes.
  Drop to LAYER 0: At [H] --> check local neighbors --> confirm [H] is closest.
  
  Total steps: ~8 instead of 15 (all nodes). Even better at billion scale!
```

The multi-layer structure:
- Top layers = sparse graph, each node connects to a few far-away neighbors
- Bottom layers = dense graph, each node connects to many nearby neighbors
- Search starts at the top (fast global navigation) and zooms in

```
More detailed HNSW diagram:

  LAYER 2 (top):          [1]---------------------------[50]
                         /                                 \
  LAYER 1 (middle):   [1]---[10]---[20]---[30]---[40]---[50]
                      /                                      \
  LAYER 0 (bottom):  [1]-[2]-[3]-[4]...[25]-[26]...[48]-[49]-[50]

  Query vector is close to node 27.

  Step 1 (Layer 2): Start at [1]. Jump to [50]? No, query is < 50. Stay at [1].
  Step 2 (Layer 1): Start at [1]. [10] closer? Yes. [20] closer? Yes.
                    [30] closer? Yes. [40] closer? No. Stay at [30].
  Step 3 (Layer 0): Start at [30]. Check [28], [29], [27]. [27] is closest!
  
  Found! Steps taken: ~8 instead of ~50. At billion scale: millions x faster.
```

---

## 4. IVF (Inverted File Index)

IVF is simpler than HNSW. Think of it as clustering.

```
STEP 1: Build clusters (k-means, done offline)

  Your 1 million vectors get grouped into 1000 clusters.
  Each cluster has a "centroid" (center point).

  Cluster 1 centroid: [0.3, 0.8, ...]  (sports articles)
  Cluster 2 centroid: [0.1, 0.2, ...]  (science articles)
  Cluster 3 centroid: [0.9, 0.1, ...]  (cooking articles)
  ...

STEP 2: At query time, find nearest cluster centroids (fast!)
  Query: "tennis tournament"
  Compare to 1000 centroids (not 1 million docs)
  Nearest: Cluster 1 (sports)

STEP 3: Only search within that cluster
  Cluster 1 has 1000 documents (not 1 million)
  Exact search within 1000 docs = fast!

  Speedup: 1,000 searches instead of 1,000,000
  Tradeoff: If relevant docs are near a cluster boundary, we might miss them
```

---

## 5. FAISS Library

FAISS (Facebook AI Similarity Search) is a library that implements:
- Exact flat index (brute force -- use for small datasets < 10K)
- IVF (inverted file -- good for 100K to 100M)
- HNSW (graph -- good for 10K to 1B, best recall)
- And many more specialized indexes

FAISS handles:
- CPU and GPU computation
- Billion-scale vectors
- Batch queries
- Compressed indexes (PQ -- product quantization) for memory efficiency

---

## 6. Index Type Comparison

```
Dataset size    | Recommended Index  | Build time | Query time | Recall
----------------|--------------------|------------|------------|--------
< 10K docs      | IndexFlatL2        | Instant    | Fast       | 100%
10K - 1M docs   | IndexHNSWFlat      | Minutes    | Very fast  | 99%
1M - 100M docs  | IndexIVFFlat       | Hours      | Fast       | 95-99%
> 100M docs     | IndexIVFPQ         | Hours      | Very fast  | 85-95%

Recall = "what % of true top-k results does ANN actually return?"
100% recall = exact. 99% recall = misses 1 in 100. Usually acceptable.
```

---

## 7. Building a FAISS Index (Quick Overview)

```python
import faiss
import numpy as np

# Your documents as a matrix of vectors
# Shape: (num_docs, embedding_dim)
# Example: 10000 docs, each with 384-dim embedding
doc_vectors = np.random.rand(10000, 384).astype('float32')  # must be float32

# Method 1: Exact flat index (brute force)
index_flat = faiss.IndexFlatL2(384)  # 384 = embedding dimension
index_flat.add(doc_vectors)          # add all vectors

# Method 2: HNSW index (fast, high recall)
index_hnsw = faiss.IndexHNSWFlat(384, 32)  # 384 dim, 32 neighbors per layer
index_hnsw.add(doc_vectors)

# Search (same API for both):
query_vector = np.random.rand(1, 384).astype('float32')
k = 5  # return top 5

distances, indices = index_hnsw.search(query_vector, k)
# distances: how far each result is
# indices: which document index it points to

print("Top 5 nearest doc indices:", indices[0])
print("Distances:", distances[0])
```

---

## 8. Speed Benchmark Visual

```
Corpus: 1 million 384-dimensional vectors
Query: find top 10 most similar

METHOD            | Time per query | Recall
------------------|----------------|--------
Brute force NumPy | ~800ms         | 100%
FAISS Flat L2     | ~50ms          | 100%
FAISS IVF         | ~5ms           | 97%
FAISS HNSW        | ~1ms           | 99%

At 10 queries/second, HNSW handles 10x more traffic than FAISS Flat.
At 10,000 queries/second (big tech scale), HNSW is the only option.
```

---

## Quiz -- Lesson 4

**Question 1:**
What is the time complexity of brute-force (exact) nearest neighbor search?

A) O(1) -- constant time
B) O(log N) -- logarithmic time
C) O(N) -- linear time (doubles when corpus doubles)
D) O(N^2) -- quadratic time

**Question 2:**
In HNSW, what is the purpose of the TOP (highest) layer?

A) Stores the most recent documents
B) Contains ALL documents for fine-grained search
C) Contains a few nodes with long-range connections for fast global navigation
D) Stores the query vector permanently

**Question 3:**
What does "recall@10" of 0.95 mean for an ANN search?

A) The search returns exactly 10 results
B) Of the TRUE top 10 results, the ANN search returns 9-10 of them (95%)
C) The search takes 95ms to run
D) The relevance score of the 10th result is 0.95

**Question 4:**
What is the C#/.NET equivalent of building a FAISS index?

A) Running a LINQ query
B) Creating a database index (like CREATE INDEX in SQL)
C) Instantiating a new List<T>
D) Calling GC.Collect()

**Question 5:**
For a corpus of 5 million documents and 384-dimensional embeddings, which FAISS
index would you choose?

A) IndexFlatL2 (brute force exact search)
B) IndexHNSWFlat (fast graph-based ANN with 99% recall)
C) A simple Python list
D) No index needed -- just use numpy dot product

### Answers:
1. C -- O(N): must compare query to every document
2. C -- Top layer has few nodes with long-range edges; allows fast "highway" navigation
3. B -- 0.95 recall = finds 95% of the true best results; misses 5%
4. B -- Both are data structures built once offline that make queries faster, with a storage cost
5. B -- At 5M docs, HNSW gives fast query (~1ms) and 99% recall

---

## Summary -- Lesson 4

**Key Takeaways:**

1. **Exact search is O(N)** -- unusable at billion scale
2. **ANN finds "close enough" results in O(log N)** -- fast at any scale
3. **HNSW uses a multi-layer graph** -- top layers for fast navigation, bottom for accuracy
4. **IVF clusters vectors** -- at query time, only search the nearest cluster
5. **FAISS is battle-tested** -- Facebook uses it for billion-scale search in production
6. **Recall tradeoff** -- ANN misses ~1-5% of exact results. Usually acceptable.

**What is coming next:**

Lesson 5 covers HYBRID SEARCH -- combining BM25 keyword scoring with semantic
vector search. In practice, neither approach alone is as good as both combined.
You will learn two methods for combining them: weighted scoring and Reciprocal
Rank Fusion (RRF).

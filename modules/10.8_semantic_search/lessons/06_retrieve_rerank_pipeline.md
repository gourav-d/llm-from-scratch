# Lesson 6 -- The Full Retrieve-Rerank Pipeline

---

## GLOSSARY

| Term          | Plain English Definition                                                         |
|---------------|----------------------------------------------------------------------------------|
| pipeline      | A series of processing steps where output of one step feeds the next            |
| latency       | Total time from user clicks search to results appear on screen                  |
| throughput    | How many queries per second your system can handle                               |
| candidate set | The set of documents retrieved in the first step, before accurate re-ranking    |
| recall        | What fraction of ALL truly relevant documents did we actually retrieve?          |
| precision     | Of all documents we returned, what fraction were actually relevant?              |
| NDCG          | Normalized Discounted Cumulative Gain -- measures ranking quality                |
| MRR           | Mean Reciprocal Rank -- how high up is the FIRST correct answer?                |
| Precision@K   | Of the top K results returned, what fraction are relevant?                       |
| shard         | Splitting a large index into smaller pieces across multiple machines             |
| cache         | Store previously computed results so you do not recompute them                  |

---

## 1. The Complete System

Everything you have learned comes together here.

```
OFFLINE PHASE (done before any users arrive):
+------------------------------------------------------------+
|                    INDEXING PIPELINE                       |
|                                                            |
|  Raw documents                                             |
|  (Wikipedia articles, product catalog, docs, etc.)         |
|        |                                                   |
|        v                                                   |
|  [Text Preprocessing]                                      |
|  - Clean HTML, strip special chars                         |
|  - Split long docs into chunks (~512 tokens each)          |
|        |                                                   |
|        v                                                   |
|  [Bi-Encoder]                                              |
|  - Encode each chunk --> 384-dim vector                    |
|  - This takes time (1 million docs ~ 1-2 hours on GPU)     |
|        |                                                   |
|        v                                                   |
|  [FAISS HNSW Index]                                        |
|  - Build index from all vectors                            |
|  - Save to disk                                            |
|        |                                                   |
|        v                                                   |
|  [BM25 Index]                                              |
|  - Tokenize all docs                                       |
|  - Build inverted index (word --> list of doc IDs)         |
|  - Save to disk                                            |
|                                                            |
+------------------------------------------------------------+

ONLINE PHASE (per query, must be fast):
+------------------------------------------------------------+
|                    QUERY PIPELINE                          |
|                                                            |
|  User types: "how do airplanes stay in the air?"           |
|        |                                                   |
|        v                                                   |
|  [Query Processing]  ~5ms                                  |
|  - Tokenize query for BM25                                 |
|  - Encode query with bi-encoder --> query vector           |
|        |                                                   |
|        +-------------------+------------------+           |
|        |                                      |           |
|        v                                      v           |
|  [BM25 Retrieval]  ~5ms             [FAISS ANN Search] ~5ms|
|  top 100 by keywords                top 100 by vectors    |
|        |                                      |           |
|        +-------------------+------------------+           |
|                            |                              |
|                            v                              |
|                   [Merge + RRF Scoring]  ~2ms             |
|                   Combined top 100 candidates             |
|                            |                              |
|                            v                              |
|               [Cross-Encoder Re-ranking]  ~100ms          |
|               Score each (query, doc) pair                |
|               Re-sort by cross-encoder score              |
|                            |                              |
|                            v                              |
|                    Top 10 results                         |
|                    returned to user                       |
|                                                           |
|  Total latency: ~120ms                                    |
+------------------------------------------------------------+
```

---

## 2. Step-by-Step Walkthrough

Let us trace one query through the full system.

### Setup: Your search index contains these documents (simplified):

```
Doc 1:  "Aircraft wings generate lift through the Bernoulli principle"
Doc 2:  "Jet engines produce thrust by accelerating air backwards"
Doc 3:  "Birds use feathers and hollow bones to achieve flight"
Doc 4:  "History of the Wright brothers and early aviation"
Doc 5:  "Airfoil shape creates pressure difference above and below wing"
Doc 6:  "Python programming tutorial for beginners"
Doc 7:  "How to bake a chocolate cake at home"
...
Doc 20: "Aerodynamics and fluid dynamics in aviation"
```

### Query: "how do planes stay up in the air"

**Step 1: Encode query**
```
Bi-encoder: "how do planes stay up in the air"
--> query_vector = [0.3, 0.8, 0.2, 0.7, ...] (384 numbers)

BM25 tokens: ["planes", "stay", "air"]
```

**Step 2: Parallel retrieval**
```
BM25 top 5:
  Doc 2: "Jet engines... accelerating air" -- "air" matches --> score: 8.4
  Doc 1: "Aircraft wings... lift"          -- less overlap  --> score: 3.1
  Doc 3: "Birds... flight"                 -- "air" weak    --> score: 2.8

Semantic top 5 (vectors close to query):
  Doc 1: 0.91  "Aircraft wings generate lift through Bernoulli"
  Doc 5: 0.89  "Airfoil shape creates pressure difference"
  Doc 20: 0.87  "Aerodynamics and fluid dynamics in aviation"
  Doc 4: 0.71  "Wright brothers and early aviation"
  Doc 2: 0.68  "Jet engines produce thrust"
```

**Step 3: RRF merge**
```
Merged candidate set (unique docs): Docs 1, 2, 3, 4, 5, 20 (6 docs)

RRF scores:
  Doc 1:  BM25 rank 2 (1/62=0.016) + Semantic rank 1 (1/61=0.016) = 0.032
  Doc 5:  BM25 rank not in top (treated as rank 100 = 1/160=0.006) + Semantic rank 2 (1/62=0.016) = 0.022
  Doc 2:  BM25 rank 1 (1/61=0.016) + Semantic rank 5 (1/65=0.015) = 0.031
  Doc 20: Semantic rank 3 (1/63=0.016) only = 0.016
```

**Step 4: Cross-encoder re-rank top 6**
```
Input pairs to cross-encoder:
  ("how do planes stay up in the air", "Aircraft wings generate lift through Bernoulli")  --> 0.93
  ("how do planes stay up in the air", "Jet engines produce thrust by accelerating air")  --> 0.71
  ("how do planes stay up in the air", "Airfoil shape creates pressure difference")       --> 0.91
  ("how do planes stay up in the air", "Aerodynamics and fluid dynamics in aviation")     --> 0.84
  ...

Final re-ranked output:
  1. Doc 1  "Aircraft wings generate lift"  [0.93]
  2. Doc 5  "Airfoil shape and pressure"    [0.91]
  3. Doc 20 "Aerodynamics in aviation"      [0.84]
  4. Doc 2  "Jet engines and thrust"        [0.71]
```

Notice: Doc 2 ranked #1 by BM25 (had "air" keyword) but ended up #4.
The cross-encoder correctly understood it is less directly relevant.

---

## 3. Latency Budget

In production, you have a latency budget. Typical target: under 200ms.

```
Operation                  | Typical Time   | Notes
---------------------------|----------------|------------------------------------
Query embedding            | 10-30ms        | Bi-encoder, CPU; 5ms on GPU
BM25 retrieval             | 1-5ms          | Very fast with inverted index
FAISS ANN search           | 1-5ms          | HNSW on CPU
RRF merge                  | <1ms           | Simple math
Cross-encoder re-rank (20) | 50-200ms       | Most expensive step
Network/API overhead       | 10-30ms        | If using remote model server
---------------------------+----------------+------------------------------------
TOTAL                      | ~75-270ms      | Depends on hardware

Ways to stay under 200ms:
  1. Use GPU for cross-encoder (5-20x faster than CPU)
  2. Limit candidate set to 20-50 (not 100)
  3. Cache frequent query embeddings
  4. Run BM25 and vector search in PARALLEL (they are independent)
  5. Use a smaller cross-encoder model (MiniLM is fast)
```

---

## 4. Scaling to Production

### Sharding the Index

```
Single machine limit: ~50M vectors in FAISS (depends on RAM)
At 100M+ vectors, split across multiple machines:

  Machine 1: Docs 1-25M     --> FAISS shard 1
  Machine 2: Docs 25M-50M   --> FAISS shard 2
  Machine 3: Docs 50M-75M   --> FAISS shard 3
  Machine 4: Docs 75M-100M  --> FAISS shard 4
            |
            v
  Query hits all 4 shards in PARALLEL
  Each returns top 20 local results
            |
            v
  Coordinator merges 4x20=80 candidates
            |
            v
  Cross-encoder re-ranks top 20 globally
```

### Query Embedding Cache

```
Many users search similar things:
  "weather in london"  (10,000 searches per day)
  "Python tutorial"    (50,000 searches per day)

If you cache the query embedding:
  First search: encode query (10ms) + search (5ms) = 15ms
  All subsequent: lookup cache (0ms) + search (5ms) = 5ms

With a simple LRU cache (Least Recently Used):
  import functools

  @functools.lru_cache(maxsize=10000)
  def get_query_embedding(query_text):
      return encoder.encode(query_text)
```

---

## 5. Measuring Search Quality

You need NUMBERS to know if your search is good.

### Metric 1: Precision@K

"Of the top K results I returned, what fraction are relevant?"

```
Example: K=5, you returned 5 results, 3 are relevant, 2 are not

Precision@5 = 3/5 = 0.60

Good Precision@5 = above 0.7
Great Precision@5 = above 0.85
```

### Metric 2: MRR (Mean Reciprocal Rank)

"How high up is the FIRST correct answer?"

```
Example queries and where the first correct answer appeared:

Query 1: "python list comprehension" --> first correct result at rank 1
  Reciprocal Rank = 1/1 = 1.0

Query 2: "how to sort dict python"   --> first correct result at rank 3
  Reciprocal Rank = 1/3 = 0.333

Query 3: "lambda function python"    --> first correct result at rank 2
  Reciprocal Rank = 1/2 = 0.5

MRR = mean of all: (1.0 + 0.333 + 0.5) / 3 = 0.611

MRR of 0.5 = on average, the first correct result is at position 2
MRR of 1.0 = the first correct result is ALWAYS at position 1 (perfect)
```

### Metric 3: NDCG@K (Normalized Discounted Cumulative Gain)

This is the most complete metric. It rewards:
- Highly relevant results at the TOP more than at the bottom
- Having SOME relevant results throughout the list

```
The intuition behind "discounted":
  Result at rank 1 is worth FULL credit
  Result at rank 2 is worth less (discounted by log(2))
  Result at rank 3 is worth even less (discounted by log(3))
  ...

Relevance scores (given by human judges): 3=perfect, 2=good, 1=ok, 0=irrelevant

Your ranking:      [3, 2, 0, 1, 2]  (positions 1-5)
Ideal ranking:     [3, 2, 2, 1, 0]  (best possible order)

NDCG@5 = your DCG / ideal DCG
       = number between 0.0 (terrible) and 1.0 (perfect)

A score of 0.85+ is considered very good in practice.
```

---

## 6. A/B Testing Your Search

Never deploy a new search system without measuring the impact:

```
EXPERIMENT SETUP:
  Control group (50% of users): Old BM25-only search
  Treatment group (50% of users): New hybrid+rerank search

METRICS TO TRACK:
  - Click-through rate: did users click results?
  - Search abandonment rate: did users give up without clicking?
  - Time to click: how long until they found what they wanted?
  - "No results" rate: how often did the search return nothing useful?

DECISION RULE:
  If treatment shows statistically significant improvement on 2+ metrics
  AND no degradation on any metric
  --> Roll out to 100% of users

This is how Google, Bing, and Elasticsearch improve their search quality.
```

---

## Quiz -- Lesson 6

**Question 1:**
In the retrieve-rerank pipeline, why does the BM25 retrieval step happen in
PARALLEL with the FAISS vector search, not sequentially?

A) Parallel is required by the FAISS library
B) BM25 and vector search are independent operations -- parallelizing them cuts total time in half
C) Sequential execution would cause a memory error
D) BM25 needs the FAISS results as input

**Question 2:**
You have 1 billion documents. Why is sharding the FAISS index necessary?

A) FAISS cannot handle more than 1000 vectors
B) A single machine does not have enough RAM to hold all vectors
C) Sharding is required for BM25 to work
D) Legal data residency requirements

**Question 3:**
A search system returns top 5 results. 4 of them are relevant. What is Precision@5?

A) 0.4
B) 0.6
C) 0.8
D) 1.0

**Question 4:**
What does MRR measure?

A) The total number of relevant documents in the corpus
B) How high up in the results the FIRST correct answer appears (on average)
C) The memory usage of the FAISS index
D) How many queries per second the system handles

**Question 5:**
NDCG gives higher credit for relevant results at rank 1 vs rank 5. Why?

A) Rank 1 results take longer to compute
B) Users are more likely to click and read results that appear at the top
C) Lower-ranked results are stored in slower memory
D) This is just a technical implementation detail with no real-world meaning

### Answers:
1. B -- BM25 and vector search are independent -- run in parallel, finish faster
2. B -- 1 billion x 384 floats x 4 bytes = ~1.5 TB. Way too large for one machine's RAM.
3. C -- 4 relevant out of 5 returned = 4/5 = 0.80
4. B -- MRR = average position of first correct result
5. B -- Users rarely scroll past the top 3 results. Ranking matters for user experience.

---

## Summary -- Lesson 6

**Key Takeaways:**

1. **The full pipeline**: Offline index (bi-encoder + FAISS + BM25) --> Online query (embed query + parallel retrieval + RRF merge + cross-encoder rerank)
2. **Total latency ~120-200ms** -- fast enough for real-time search
3. **Scale with sharding** -- split index across machines for billion-scale
4. **Cache frequent queries** -- huge speedup for popular searches
5. **Measure quality with**: Precision@K (exactness), MRR (first correct answer position), NDCG (full ranking quality)
6. **Always A/B test** before replacing your search system

---

## Module 10.8 Summary -- The Complete Picture

You have learned:

| Lesson | Topic               | Key Concept                                          |
|--------|---------------------|------------------------------------------------------|
| 1      | Keyword vs Semantic | Vocabulary mismatch; embeddings bridge the gap       |
| 2      | Bi-Encoder          | Pre-encode docs offline; 1 encode call at query time |
| 3      | Cross-Encoder       | Encode query+doc together; accurate but slow         |
| 4      | FAISS + ANN         | O(log N) search via HNSW; handles billions of vectors|
| 5      | Hybrid Search       | BM25 + semantic via RRF; beats either alone          |
| 6      | Full Pipeline       | Offline index + online retrieve+rerank; measure NDCG |

**You can now design a production-grade semantic search system.**

This is the same architecture used by:
- Elasticsearch 8 (kNN + BM25 hybrid mode)
- Vespa.ai (used by Yahoo, Spotify)
- Weaviate, Qdrant, Pinecone (vector databases)
- GitHub Copilot's code search
- Google's internal document search

Now run the examples and build the mini search engine project!

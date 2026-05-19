# Lesson 3 -- Cross-Encoders and Re-ranking

---

## GLOSSARY

| Term                  | Plain English Definition                                                         |
|-----------------------|----------------------------------------------------------------------------------|
| cross-encoder         | A model that takes BOTH query and document as one input and outputs a score      |
| re-ranking            | Taking a set of already-retrieved results and sorting them in a better order     |
| relevance score       | A number (0.0 to 1.0) saying how relevant a document is to a query              |
| retrieve-then-rerank  | A two-step pattern: first retrieve candidates fast, then score accurately        |
| candidate set         | The set of documents retrieved in the first step, before re-ranking              |
| interaction features  | Patterns from how query words and document words relate to each other            |
| latency budget        | The total time you are allowed to take before returning results to the user      |
| precision             | Out of the results you returned, how many were actually relevant?                |
| recall                | Out of ALL relevant documents, how many did you find and return?                 |

---

## 1. The Problem with Bi-Encoders

In Lesson 2 you learned that bi-encoders are fast because:
- Documents are encoded SEPARATELY from the query
- You only need 1 encode call at query time

But this separation causes a problem.

**When you encode separately, the query and document NEVER see each other
during encoding.** They cannot influence each other's representation.

Think about this query and document pair:

```
Query:    "What is the capital of Australia?"
Document: "Canberra is the capital city of Australia, not Sydney."
```

A bi-encoder sees these separately:
- Query vector is built from "capital Australia question" type patterns
- Document vector is built from "Canberra capital Australia" type patterns

They will score decently similar, but the bi-encoder cannot notice the specific
phrase "capital city of Australia" that DIRECTLY answers the question.

A cross-encoder sees them TOGETHER and catches this interaction.

---

## 2. How a Cross-Encoder Works

The cross-encoder takes BOTH texts together as ONE input:

```
Input format:

  [CLS] What is the capital of Australia? [SEP] Canberra is the capital
  city of Australia, not Sydney. [SEP]
        |
        v
  [Full Transformer -- attention sees ALL words together]
        |
        v
  Single relevance score: 0.97
  (This document directly answers the question -- high score!)
```

Because attention sees ALL words at once, the model learns:
- "capital" in query matches "capital" in document
- "Australia" in query matches "Australia" in document  
- The document directly answers the type of question being asked

---

## 3. C#/.NET Analogy

```csharp
// Bi-encoder is like TWO separate methods:
double[] EncodeQuery(string query) { ... }         // call 1
double[] EncodeDocument(string document) { ... }   // call 2
double score = CosineSimilarity(queryVec, docVec); // compare fingerprints

// Cross-encoder is like ONE method that takes BOTH:
double ScoreRelevance(string query, string document)
{
    // Analyzes the interaction between query and doc
    // Returns a relevance score directly
    string combined = $"{query} [SEP] {document}";
    return transformer.Predict(combined);  // 0.0 to 1.0
}

// The cross-encoder is like a judge who reads BOTH the question
// AND the answer together, vs a bi-encoder which judges each separately.
```

---

## 4. Speed vs Accuracy Trade-off

This is the core tension in search systems:

```
BI-ENCODER:
  Speed:    Very fast (pre-compute docs offline)
  Accuracy: Good (80-85% of what a human judge would agree with)
  Use case: First pass -- get top 100 candidates quickly

CROSS-ENCODER:
  Speed:    Slow (cannot pre-compute -- depends on the query)
  Accuracy: Excellent (93-97% agreement with human judges)
  Use case: Second pass -- accurately rank top 100 down to top 10

WHY cross-encoder cannot be pre-computed:
  Pre-compute requires knowing the query in advance.
  But the query changes every time a user searches.
  So you MUST run the cross-encoder at query time, once per (query, doc) pair.
```

Number of model calls at query time:

```
Corpus: 1,000,000 documents
Query: "fast car"

APPROACH 1: Use cross-encoder only:
  1,000,000 x cross-encoder calls = MILLIONS of calls per query
  At 20ms each = 20,000 SECONDS to answer one query. IMPOSSIBLE.

APPROACH 2: Bi-encoder only:
  1 encode call + 1,000,000 vector comparisons = ~30ms total
  Fast! But ranking quality is 85% not 97%.

APPROACH 3: Bi-encoder + Cross-encoder (THE WINNER):
  Step 1: Bi-encoder retrieves top 100 from 1,000,000 in ~30ms
  Step 2: Cross-encoder re-ranks just those 100 pairs in ~100ms
  Total: ~130ms. Fast enough! And 97% accurate!
```

---

## 5. The Retrieve-Then-Rerank Pipeline

```
ALL DOCUMENTS (1 million)
          |
          v
  [Step 1: Bi-Encoder Retrieval]
  Fast. Approximate. Gets 95% of the good stuff.
          |
          v
  TOP 100 CANDIDATES
  (fast but not perfectly ordered)
          |
          v
  [Step 2: Cross-Encoder Re-ranking]
  Slow but accurate. Re-scores each (query, doc) pair.
          |
          v
  TOP 10 RESULTS
  (accurately ordered by true relevance)
          |
          v
  SHOWN TO USER
```

Let us trace a real example:

```
Query: "best treatment for a sore throat"

AFTER BI-ENCODER (top 5 shown):
  Rank 1: "Throat lozenges and their effectiveness"     [bi score: 0.82]
  Rank 2: "Pain management in throat infections"        [bi score: 0.81]
  Rank 3: "Home remedies for common cold symptoms"      [bi score: 0.79]
  Rank 4: "Antibiotics for strep throat treatment"      [bi score: 0.78]
  Rank 5: "Throat anatomy and physiology"               [bi score: 0.77]

AFTER CROSS-ENCODER RE-RANKING:
  Rank 1: "Antibiotics for strep throat treatment"      [cross score: 0.95] (was rank 4!)
  Rank 2: "Home remedies for common cold symptoms"      [cross score: 0.92] (was rank 3!)
  Rank 3: "Throat lozenges and their effectiveness"     [cross score: 0.89] (was rank 1)
  Rank 4: "Pain management in throat infections"        [cross score: 0.71] (was rank 2)
  Rank 5: "Throat anatomy and physiology"               [cross score: 0.41] (was rank 5)

The cross-encoder correctly identified that "antibiotics for strep throat
treatment" is the most directly relevant answer to "best treatment".
The bi-encoder missed this because "best treatment" and "antibiotics" are
not as close in simple vector space.
```

---

## 6. When Each Approach Wins

| Scenario                              | Best Approach     |
|---------------------------------------|-------------------|
| User just needs SOMETHING relevant   | Bi-encoder alone  |
| User needs the EXACT best answer     | Bi + Cross        |
| Only 1000 documents total            | Cross-encoder only|
| Real-time, <50ms latency required    | Bi-encoder alone  |
| E-commerce product search            | Hybrid + Cross    |
| Q&A / question answering             | Bi + Cross        |
| News article recommendation          | Bi-encoder alone  |

---

## 7. Popular Cross-Encoder Models

| Model Name                           | Size   | Speed  | Quality |
|--------------------------------------|--------|--------|---------|
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 22MB   | Fast   | Good    |
| cross-encoder/ms-marco-MiniLM-L-12-v2| 34MB   | Moderate| Great  |
| cross-encoder/nli-deberta-v3-base    | 184MB  | Slow   | Excellent|

ms-marco models are trained on Bing search query data -- good for web search.

---

## 8. Quick Code Preview

```python
# Cross-encoder usage (Part B in example files):
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "best treatment for sore throat"
candidates = [
    "Throat lozenges and their effectiveness",
    "Antibiotics for strep throat treatment",
    "Throat anatomy and physiology",
]

# Score each (query, doc) pair
pairs = [[query, doc] for doc in candidates]
scores = model.predict(pairs)  # array of floats

# Re-rank
for score, doc in sorted(zip(scores, candidates), reverse=True):
    print(f"Score: {score:.3f} | {doc}")
```

---

## Quiz -- Lesson 3

**Question 1:**
What is the key structural difference between a bi-encoder and a cross-encoder?

A) Bi-encoders use BERT, cross-encoders use GPT
B) Bi-encoder encodes query and doc separately; cross-encoder takes them together as one input
C) Cross-encoders are always faster because they use a different algorithm
D) Bi-encoders can only process one document at a time

**Question 2:**
Why can document vectors NOT be pre-computed when using a cross-encoder?

A) Cross-encoders use too much memory to store vectors
B) Cross-encoder output depends on the QUERY which changes each time
C) Document vectors expire after 24 hours
D) Pre-computing would violate copyright

**Question 3:**
In the retrieve-then-rerank pattern, what is the purpose of Step 1 (bi-encoder)?

A) To produce a perfectly accurate ranking
B) To eliminate documents about completely different topics (fast, approximate)
C) To download the documents from the internet
D) To translate documents into the query language

**Question 4:**
You have 500,000 documents. A user searches. The bi-encoder returns top 50 candidates.
The cross-encoder now re-ranks these 50. How many cross-encoder calls happen?

A) 500,000 (one per document)
B) 1 (one for the query)
C) 50 (one per candidate)
D) 500,050 (both)

**Question 5:**
A cross-encoder gives document A a score of 0.92 and document B a score of 0.71.
The bi-encoder originally ranked B above A. What does the pipeline return?

A) Document B first (original bi-encoder order)
B) Document A first (cross-encoder says A is more relevant)
C) Both documents at the same position
D) Neither -- conflicting signals mean no result

### Answers:
1. B -- The key difference is separate vs joint encoding
2. B -- Cross-encoder input is [query + doc] -- you need the query to score any document
3. B -- Fast retrieval to eliminate irrelevant docs from 500K down to 50
4. C -- 50 calls: one (query, doc) pair per candidate
5. B -- Cross-encoder overrides bi-encoder ranking; A goes to rank 1

---

## Summary -- Lesson 3

**Key Takeaways:**

1. **Cross-encoder takes query + document together** -- sees their interaction
2. **Much more accurate than bi-encoder** -- but cannot be pre-computed
3. **The solution: retrieve-then-rerank** -- bi-encoder gets top 100 fast, cross-encoder ranks those 100 accurately
4. **Number of cross-encoder calls = number of candidates** -- keep candidates small (50-200)
5. **Use case: anywhere accuracy matters** -- Q&A, legal search, medical search

**What is coming next:**

Even with a bi-encoder, comparing your query vector against ALL 1 million
document vectors takes time. Lesson 4 covers FAISS and Approximate Nearest
Neighbor (ANN) search -- how to find the closest vectors in O(log N) time
instead of O(N), handling billions of vectors efficiently.

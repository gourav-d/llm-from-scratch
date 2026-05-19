# Lesson 5 -- Hybrid Search

---

## GLOSSARY

| Term                      | Plain English Definition                                                      |
|---------------------------|-------------------------------------------------------------------------------|
| hybrid search             | Combining keyword search (BM25) and semantic search (vectors) into one result |
| BM25                      | Best Match 25 -- a keyword scoring algorithm, better than TF-IDF              |
| BM25 score                | A number saying how relevant a document is based on keyword frequency         |
| semantic score            | A number saying how relevant a document is based on vector similarity         |
| weighted combination      | Multiply each score by a weight and add: 0.5*bm25 + 0.5*semantic             |
| RRF                       | Reciprocal Rank Fusion -- combine RANK positions instead of raw scores        |
| reciprocal rank           | For a document ranked at position r, its RRF contribution is 1/(k + r)       |
| alpha                     | The weight given to one score vs another (0.0 to 1.0)                        |
| lexical search            | Another word for keyword/exact-text search (from "lexicon" = vocabulary)      |
| dense retrieval           | Another word for semantic/vector search (vectors are "dense" arrays)          |

---

## 1. Why Neither Approach Alone Is Enough

You have learned:
- **BM25 / Keyword search**: Great for exact terms, product codes, proper nouns
- **Semantic search**: Great for meaning, synonyms, paraphrases, questions

Let us see where each one FAILS:

### Keyword search (BM25) fails at:
```
Query: "how do I handle errors in my code"
Document: "Exception handling in Python using try-except blocks"

BM25 score: LOW
- "handle" and "handling" are different word forms (stemming helps but not always)
- "errors" and "Exception" -- different vocabulary
- BM25 misses this highly relevant document
```

### Semantic search fails at:
```
Query: "Python 3.11 release notes"
Document: "What changed in Python version 3.11 release documentation"

Semantic score: MEDIUM
- This is actually fine. But now try:

Query: "SKU-X47-B specifications"
Document: "Product SKU-X47-B technical specification sheet"

Semantic score: LOW
- Product codes are not in training data
- The model has no idea what "SKU-X47-B" means semantically
- But BM25 would score this PERFECTLY (exact keyword match)
```

---

## 2. Hybrid Search Combines Both Strengths

```
         QUERY: "python exception handling tutorial"
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
   [BM25 Keyword Search]           [Semantic Search]
   Finds docs with                 Finds docs with
   "exception", "handling"         similar meaning
          |                                 |
          v                                 v
   BM25 Rankings:                  Semantic Rankings:
   1. Python try-except guide      1. Error handling best practices
   2. Exception class reference    2. Python try-except guide
   3. Python 3.11 error types      3. Debugging Python applications
          |                                 |
          +----------------+----------------+
                           |
                           v
                  [Combine Rankings]
                           |
                           v
                  Hybrid Final Rankings:
                  1. Python try-except guide         (top in BOTH -- wins)
                  2. Error handling best practices    (top semantic)
                  3. Exception class reference        (top BM25)
                  4. Debugging Python applications
                  5. Python 3.11 error types
```

---

## 3. C#/.NET Analogy

```csharp
// BM25 search -- like a database full-text search
// Returns documents sorted by keyword relevance
List<(Document doc, double bm25Score)> KeywordSearch(string query) { ... }

// Semantic search -- like a vector similarity search
// Returns documents sorted by meaning similarity
List<(Document doc, double semanticScore)> SemanticSearch(string query) { ... }

// Hybrid -- combine both:
// Like merging two sorted lists with a custom comparer
List<Document> HybridSearch(string query, double alpha = 0.5)
{
    var keywordResults  = KeywordSearch(query);
    var semanticResults = SemanticSearch(query);
    
    // Combine with weighted average
    var combined = MergeAndScore(keywordResults, semanticResults, alpha);
    
    return combined.OrderByDescending(x => x.combinedScore).ToList();
}
```

---

## 4. Method 1: Weighted Score Combination

The simplest approach: multiply each score by a weight and add them.

```
formula:
  final_score = alpha * semantic_score + (1 - alpha) * bm25_score

where:
  alpha = 0.5 means: semantic and BM25 contribute equally
  alpha = 0.7 means: semantic contributes 70%, BM25 contributes 30%
  alpha = 0.3 means: BM25 contributes 70%, semantic contributes 30%
```

**IMPORTANT PROBLEM**: BM25 scores and semantic scores are on different scales!

```
BM25 scores:    0.0 to 20.0 (depends on document length, corpus size)
Semantic scores: -1.0 to 1.0 (cosine similarity range)

If you just add: 0.5 * 0.85 + 0.5 * 14.3 = 7.57
The BM25 score DOMINATES because it is on a larger scale!
```

**Solution: Normalize both to 0-1 range before combining.**

```python
# Normalize a list of scores to 0.0 - 1.0 range
def normalize(scores):
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)  # all same -- return 0.5 for all
    return [(s - min_s) / (max_s - min_s) for s in scores]

# After normalizing:
bm25_norm   = normalize(bm25_scores)    # all values in [0, 1]
semantic_norm = normalize(semantic_scores)  # all values in [0, 1]

# Now weighted combination is fair:
final = 0.5 * semantic_norm[i] + 0.5 * bm25_norm[i]
```

---

## 5. Method 2: Reciprocal Rank Fusion (RRF)

RRF does not use scores at all. It uses RANKS (positions).

The formula for each document:
```
RRF_score = sum over each ranking of:  1 / (k + rank)

where:
  rank = position of document in that list (1 = first, 2 = second, etc.)
  k    = a constant, usually 60 (prevents high scores for rank 1 dominating)
```

Example:

```
Document "Python try-except guide":
  BM25 rank:     2  --> contribution: 1 / (60 + 2)  = 0.01613
  Semantic rank: 1  --> contribution: 1 / (60 + 1)  = 0.01639
  RRF score:     0.01613 + 0.01639 = 0.03252

Document "Error handling best practices":
  BM25 rank:     8  --> contribution: 1 / (60 + 8)  = 0.01471
  Semantic rank: 3  --> contribution: 1 / (60 + 3)  = 0.01587
  RRF score:     0.01471 + 0.01587 = 0.03058

Document "Exception class reference":
  BM25 rank:     1  --> contribution: 1 / (60 + 1)  = 0.01639
  Semantic rank: 9  --> contribution: 1 / (60 + 9)  = 0.01449
  RRF score:     0.01639 + 0.01449 = 0.03088

Final RRF ranking:
  1. Python try-except guide         0.03252  (top in both -- clear winner)
  2. Exception class reference       0.03088  (BM25 #1, semantic #9)
  3. Error handling best practices   0.03058  (BM25 #8, semantic #3)
```

Visual of RRF:

```
BM25 ranking:             Semantic ranking:
1. Exception class ref    1. Python try-except     <-- appears in both top 2
2. Python try-except      2. Error handling bps    <-- appears in both
3. Error handling bps     3. Debugging guide
4. Python 3.11 types      4. Exception class ref

RRF combines positions, not scores. Documents in top positions of BOTH
rankings score highest.

RRF Final:
1. Python try-except      (rank 2 in BM25 + rank 1 in semantic = strong)
2. Exception class ref    (rank 1 in BM25 + rank 4 in semantic = ok)
3. Error handling bps     (rank 3 in BM25 + rank 2 in semantic = good)
```

---

## 6. Weighted vs RRF: Which to Use?

| Aspect                     | Weighted Combination            | RRF                             |
|----------------------------|----------------------------------|---------------------------------|
| Needs score normalization  | YES -- critical to normalize     | NO -- only uses rank positions  |
| Tunable                    | YES -- tune alpha                | Mostly fixed (k=60)             |
| Sensitive to outlier scores| YES -- one big score dominates   | NO -- rank position is stable   |
| Interpretable              | Easy to explain                  | Slightly abstract               |
| When to prefer             | When scores are calibrated well  | When scores are not comparable  |
| Industry standard          | Used widely                      | Used in Elasticsearch, Vespa    |

**Recommendation for most cases: use RRF.** It is more robust because it does
not require scores to be on the same scale.

---

## 7. Hybrid Search Pipeline

```
                    User Query: "best python error handling"
                                   |
             +---------------------+---------------------+
             |                                           |
             v                                           v
    [BM25 Tokenization]                       [Embed Query]
    tokenize + score                          get query vector
             |                                           |
             v                                           v
    BM25 top-100 results                     Vector top-100 results
    (with BM25 scores)                       (with cosine scores)
             |                                           |
             +---------------------+---------------------+
                                   |
                                   v
                        [Merge candidate sets]
                        union of both sets
                        (may be up to 200 unique docs)
                                   |
                                   v
                      [Compute RRF or Weighted score]
                      for each doc: look up rank in each list
                                   |
                                   v
                         [Sort by combined score]
                                   |
                                   v
                           Top 20 hybrid results
                                   |
                                   v
                   [Optional: Cross-encoder re-rank top 20]
                                   |
                                   v
                              Top 5 shown to user
```

---

## 8. Real-World Results

Studies comparing search quality (NDCG scores, higher = better):

```
Dataset: BEIR benchmark (standard IR evaluation)

Method                          | Average NDCG@10
--------------------------------|----------------
BM25 only                       | 0.421
Semantic only (bi-encoder)      | 0.439
Hybrid (BM25 + semantic)        | 0.497
Hybrid + Cross-encoder rerank   | 0.536

Hybrid beats each individual method.
Adding cross-encoder rerank gives the biggest quality jump.
```

---

## Quiz -- Lesson 5

**Question 1:**
Why must BM25 scores and semantic scores be NORMALIZED before weighted combination?

A) Normalization makes the search faster
B) They are on different scales, and without normalization one dominates the other
C) BM25 scores are negative by default
D) Normalization prevents memory overflow

**Question 2:**
In RRF, what does the formula "1 / (k + rank)" produce for a document at rank 1?

A) A very large number (rank 1 gets the highest contribution)
B) Zero (rank 1 is ignored)
C) The same value as all other ranks
D) Exactly 1.0

**Question 3:**
A query is "SKU-X47-B specifications". Which search method handles this best?

A) Semantic search -- it understands technical specifications
B) BM25 keyword search -- exact code matches are its strength
C) Neither -- this type of query cannot be searched
D) Cross-encoder re-ranking only

**Question 4:**
If alpha = 0.8 in "final = alpha * semantic + (1 - alpha) * bm25", which signal dominates?

A) BM25 dominates (80% weight)
B) Semantic dominates (80% weight)
C) They are equal
D) Neither -- alpha = 0.8 disables semantic

**Question 5:**
What is the main advantage of RRF over weighted combination?

A) RRF is faster to compute
B) RRF does not require score normalization -- it only uses rank positions
C) RRF always produces better results than weighted combination
D) RRF works with more than 2 ranking systems

### Answers:
1. B -- BM25 can produce scores of 0-20; cosine similarity is -1 to 1; without normalizing, BM25 drowns out semantic signal
2. A -- 1/(60+1) = 0.0164, which is the highest possible RRF contribution
3. B -- BM25 matches exact text; semantic models likely have no training signal for "SKU-X47-B"
4. B -- alpha=0.8 weights semantic at 80%, BM25 at 20%
5. B -- RRF is scale-invariant; you do not need to normalize

---

## Summary -- Lesson 5

**Key Takeaways:**

1. **Neither BM25 nor semantic alone is best** -- they complement each other
2. **BM25 wins at**: exact terms, product codes, rare proper nouns
3. **Semantic wins at**: paraphrases, questions, conceptual queries
4. **Hybrid almost always beats either alone** -- this is the industry standard
5. **Two methods**: weighted combination (tune alpha) and RRF (no tuning needed)
6. **RRF is usually better in practice** -- scale-invariant, robust

**What is coming next:**

Lesson 6 puts everything together into a PRODUCTION PIPELINE -- from indexing
millions of documents offline, to serving queries in under 200ms, to measuring
search quality with metrics like NDCG and MRR. This is the complete picture.

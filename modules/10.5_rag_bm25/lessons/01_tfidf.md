# Lesson 01: TF-IDF — Keyword-Based Retrieval

## Glossary (Read This First!)

| Term | Plain English Definition |
|------|--------------------------|
| **TF-IDF** | Term Frequency-Inverse Document Frequency. A score that says how important a word is to a specific document within a collection. |
| **Term Frequency (TF)** | How often a word appears in a specific document. "cat" appears 5 times in a 100-word doc → TF = 5/100 = 0.05 |
| **Inverse Document Frequency (IDF)** | How rare a word is across ALL documents. Common words ("the", "is") get low IDF. Rare words ("photosynthesis") get high IDF. |
| **TF-IDF score** | TF x IDF. High score = this word is frequent in THIS doc but rare elsewhere. That makes it important for THIS doc. |
| **Corpus** | A collection of documents. Your "database" of text to search. |
| **Document** | One piece of text in the corpus. Could be a sentence, paragraph, article, or file. |
| **Query** | The search input from the user. |
| **Vocabulary** | The set of all unique words seen across all documents. |
| **Sparse vector** | A vector with mostly zeros. TF-IDF produces sparse vectors — most words score 0 in most documents. |
| **Stop words** | Common words ("the", "a", "is") usually removed before TF-IDF. They carry no useful signal. |
| **Tokenization** | Splitting text into individual words/tokens before computing TF-IDF. |

---

## Part 1: The Core Problem TF-IDF Solves

You have 1 million documents. A user searches "neural network training".
How do you find the most relevant documents FAST, without reading all of them?

**Naive approach:** Search for exact string match. Fails if document says "training neural nets" instead.

**Better approach:** Score every document based on how well its words match the query words.

TF-IDF gives every word in every document a relevance score.
To answer a query, you look up the scores for the query words and rank documents.

---

## Part 2: Term Frequency (TF)

**Definition:** How often does a word appear in this document?

```
TF(word, document) = count of word in document / total words in document
```

### Example

Document: "the cat sat on the mat the cat likes the mat"
Total words: 10

```
+----------+-------+----+
| Word     | Count | TF |
+----------+-------+----+
| the      |   4   |0.40|
| cat      |   2   |0.20|
| sat      |   1   |0.10|
| on       |   1   |0.10|
| mat      |   2   |0.20|
| likes    |   1   |0.10|
+----------+-------+----+
```

Problem with TF alone: "the" scores highest but carries no useful meaning.
This is why we need IDF to penalize common words.

---

## Part 3: Inverse Document Frequency (IDF)

**Definition:** How rare is this word across ALL documents?

```
IDF(word) = log( total_documents / documents_containing_word )
```

The `log` prevents extreme values. If a word appears in every document, IDF = log(1) = 0.

### Example with 4 documents

```
+---------------+------------------------------+-----+
| Word          | Docs containing it (of 4)    | IDF |
+---------------+------------------------------+-----+
| "the"         | 4 (all docs)    log(4/4)=0.0 | 0.0 |
| "neural"      | 2 docs          log(4/2)=0.3 | 0.3 |
| "backprop"    | 1 doc           log(4/1)=0.6 | 0.6 |
| "eigenvalue"  | 1 doc           log(4/1)=0.6 | 0.6 |
+---------------+------------------------------+-----+
```

"the" → IDF = 0.0 (useless, in every doc)
"backprop" → IDF = 0.6 (useful, rare word)

---

## Part 4: TF-IDF Score

**Definition:** Multiply TF and IDF together.

```
TF-IDF(word, document) = TF(word, document) x IDF(word)
```

A high TF-IDF score means:
- This word appears a lot IN THIS document (high TF)
- AND this word is rare across all other documents (high IDF)
- Therefore this word is a distinctive feature of this document

### Visual Example

```
+------------------------------------------------------------------+
|  THREE DOCUMENTS, SEARCHING FOR "cat"                            |
+------------------------------------------------------------------+
|                                                                  |
|  Doc A: "the cat sat on the mat"        TF=0.17, IDF=0.18       |
|  Doc B: "the dog ran in the park"       TF=0.00, IDF=0.18       |
|  Doc C: "a cat and a dog played"        TF=0.17, IDF=0.18       |
|                                                                  |
|  TF-IDF scores for "cat":                                        |
|    Doc A: 0.17 x 0.18 = 0.030  <-- relevant                     |
|    Doc B: 0.00 x 0.18 = 0.000  <-- not relevant                 |
|    Doc C: 0.17 x 0.18 = 0.030  <-- relevant                     |
|                                                                  |
|  Ranked results for query "cat": [Doc A, Doc C, Doc B]           |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 5: How TF-IDF Retrieval Works End-to-End

```
+------------------------------------------------------------------+
|  TF-IDF RETRIEVAL PIPELINE                                       |
+------------------------------------------------------------------+
|                                                                  |
|  INDEXING (done once at startup)                                 |
|  ─────────────────────────────                                   |
|  1. Collect all documents into corpus                            |
|  2. Tokenize each document (split into words)                    |
|  3. Remove stop words ("the", "a", "is")                         |
|  4. Compute TF for every word in every document                  |
|  5. Compute IDF for every word across all documents              |
|  6. Build TF-IDF matrix: rows=documents, cols=vocabulary words   |
|                                                                  |
|         doc1  doc2  doc3  ...                                    |
|  "cat"  0.03  0.00  0.03                                         |
|  "dog"  0.00  0.04  0.03                                         |
|  "sat"  0.05  0.00  0.00                                         |
|  ...                                                             |
|                                                                  |
|  QUERYING (done per search request)                              |
|  ──────────────────────────────────                              |
|  1. Tokenize query ("what is a cat")                             |
|  2. Compute TF-IDF vector for query (same vocabulary)            |
|  3. Compute cosine similarity: query vector vs each doc vector   |
|  4. Rank documents by similarity score                           |
|  5. Return top-K most similar documents                          |
|                                                                  |
+------------------------------------------------------------------+
```

### C# Analogy

```csharp
// SQL full-text search uses similar ideas under the hood:
SELECT TOP 10 title, RANK
FROM documents
WHERE CONTAINS(content, 'cat')
ORDER BY RANK DESC;

// The RANK score SQL computes is based on word frequency
// (essentially a simpler version of TF-IDF).

// TF-IDF in Python is like building that full-text index yourself,
// with full control over the scoring formula.
```

---

## Part 6: Limitations of TF-IDF

```
+------------------------------------------------------------------+
|  WHAT TF-IDF CANNOT DO                                           |
+------------------------------------------------------------------+
|                                                                  |
|  Problem: Vocabulary mismatch                                    |
|  ──────────────────────────                                      |
|  Query:    "fast car"                                            |
|  Document: "quick automobile"                                    |
|  TF-IDF:   score = 0  (no words in common!)                      |
|                                                                  |
|  Problem: Synonyms                                               |
|  ────────────────                                                |
|  Query:    "buy laptop"                                          |
|  Document: "purchase notebook computer"                          |
|  TF-IDF:   score = 0  (no overlap)                               |
|                                                                  |
|  Problem: Word order ignored                                     |
|  ───────────────────────────                                     |
|  "dog bites man"  vs  "man bites dog"                            |
|  TF-IDF treats these as IDENTICAL (same words, same scores)      |
|                                                                  |
|  Solution for all three: use embeddings (Module 10, 10.8)        |
|  But TF-IDF is still valuable for exact keyword matching!        |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 7: When TF-IDF Wins

TF-IDF is better than vector search when:

| Scenario | Why TF-IDF wins |
|----------|----------------|
| Search for product code "SKU-4821-B" | Exact string, embeddings won't help |
| Search for error message text | Exact words matter more than meaning |
| Legal search for exact clause text | Precision over recall |
| Fast prototype with no GPU | No model needed, runs in milliseconds |
| Low-resource environment | Just CPU + RAM, small library |

---

## Quiz

**Q1.** TF stands for ___. IDF stands for ___.

**Q2.** A word appears in all 1000 documents in your corpus. What is its IDF score approximately?

**Q3.** "the" gets a TF-IDF score of 0 even though it appears frequently. Why?

**Q4.** You search "automobile" but a document contains only "car". What score does TF-IDF give?

**Q5.** What data structure does TF-IDF produce for each document? (sparse or dense vector?)

---

## Answers

**A1.** TF = Term Frequency. IDF = Inverse Document Frequency.

**A2.** IDF = log(1000/1000) = log(1) = 0. It contributes nothing to TF-IDF score.

**A3.** "the" appears in every document → IDF ≈ 0 → TF × 0 = 0.

**A4.** Score = 0. TF-IDF has no concept of synonyms. Zero word overlap = zero score.

**A5.** Sparse vector. Most words score 0 in most documents. Only the words that actually appear in a document have non-zero scores.

# Mini Search Engine -- Module 10.8 Project

## What This Project Does

This is a working search engine over 50 Wikipedia article summaries.
You type a query, it returns the top 5 most relevant articles -- using
the full retrieve-rerank pipeline from Lesson 6.

No internet connection needed. All 50 articles are hardcoded in main.py.

---

## How to Run

```bash
# Make sure you are in the module root directory:
cd modules/10.8_semantic_search

# Run the interactive search engine:
python projects/mini_search_engine/main.py
```

---

## Architecture

```
USER INPUT: "how does gravity work"
                    |
                    v
          +------------------+
          |   main.py        |  -- interactive loop, calls Searcher + Reranker
          +------------------+
                    |
        +-----------+-----------+
        |                       |
        v                       v
+---------------+    +-------------------+
| indexer.py    |    | searcher.py       |
|               |    |                   |
| Loads 50 docs |    | Takes query,      |
| Encodes them  |    | compares to index |
| Builds index  |    | Returns top-20    |
| Saves to disk |    | candidates        |
+---------------+    +-------------------+
                              |
                              v
                   +-------------------+
                   | reranker.py       |
                   |                   |
                   | Takes top-20 from |
                   | Searcher, applies |
                   | cross-encoder,    |
                   | returns top-5     |
                   +-------------------+
                              |
                              v
                    FINAL TOP-5 RESULTS
                    with title + snippet
```

---

## Sample Session

```
=============================================
  MINI SEARCH ENGINE (50 Wikipedia articles)
=============================================

Type a query to search. Type 'quit' to exit.

> gravity black holes
Searching for: 'gravity black holes'...

Top 5 results:
  [1] Black hole  (score: 0.82)
      A black hole is a region of spacetime where gravity is so strong
      that nothing, not even light, can escape...

  [2] General relativity  (score: 0.71)
      Albert Einstein's theory of general relativity describes gravity
      as a curvature of spacetime...

  [3] Neutron star  (score: 0.63)
      A neutron star is the collapsed core of a giant star...

  [4] Isaac Newton  (score: 0.41)
      Isaac Newton formulated the laws of motion and universal gravitation...

  [5] Astronomy  (score: 0.38)
      Astronomy is a natural science that studies celestial objects...

> quit
Goodbye!
```

---

## File Structure

```
mini_search_engine/
  README.md      -- this file
  main.py        -- interactive console loop, startup indexing
  indexer.py     -- Indexer class: load docs, encode, build index, save/load
  searcher.py    -- Searcher class: bi-encoder retrieval (top-20 candidates)
  reranker.py    -- Reranker class: cross-encoder re-ranking (top-5 final)
```

---

## How the 50 Articles Are Organized

The 50 Wikipedia article summaries cover 5 topic areas:

| Topic      | Articles                                                |
|------------|---------------------------------------------------------|
| Science    | Gravity, DNA, Photosynthesis, Evolution, Black holes... |
| History    | World War II, Roman Empire, Ancient Egypt, Silk Road... |
| Technology | Internet, Computer, Artificial intelligence, Python...  |
| Sports     | Football, Tennis, Olympics, Swimming...                 |
| Geography  | Amazon River, Mount Everest, Pacific Ocean, Sahara...   |

---

## Learning Objectives

After completing this project you will:
1. Understand how an Indexer encodes and stores documents
2. See how a Searcher retrieves candidates using vector similarity
3. See how a Reranker improves quality over raw retrieval
4. Have a working search engine you can extend with real models (Part B)

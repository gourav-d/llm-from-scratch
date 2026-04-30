# Module 10 -- Vector Databases

## What You Will Learn

By the end of this module, you will be able to:

- Explain what a "vector" is in simple terms (it is just a list of numbers)
- Understand why traditional databases cannot do similarity search
- Measure similarity between vectors using NumPy
- Use ChromaDB -- the easiest vector database for Python
- Build a real semantic search engine that finds documents by MEANING, not exact words

## Why This Matters

Imagine you search for "how to fix a car engine" and the system finds documents about
"automobile repair" and "vehicle maintenance" -- even though those exact words do not appear.

That is called semantic search, and it is powered by vector databases.

Every major AI application uses vector databases:
  - ChatGPT Plus memory (stores your past conversations as vectors)
  - GitHub Copilot (finds relevant code by meaning, not by grep)
  - Google Search (finds semantically similar results)
  - Customer support bots (finds the right answer from a knowledge base)
  - Spotify/Netflix recommendations (finds similar songs/movies)

## Prerequisites

You should have completed (or be familiar with):
  - Module 02: NumPy (you need to understand arrays and matrix math)
  - Module 05: Building LLMs (helpful for understanding embeddings -- but not required)

## Module Structure

```
10_vector_databases/
|-- README.md                  <- You are here
|-- GETTING_STARTED.md         <- Setup instructions (install libraries)
|-- requirements.txt           <- Libraries to install
|
|-- lessons/
|   |-- 01_what_are_vector_databases.md    <- Core concept: what and why
|   |-- 02_embeddings_and_similarity.md    <- How text becomes numbers
|   |-- 03_chromadb_hands_on.md            <- Using a real vector database
|   |-- 04_building_document_search.md     <- Full project walkthrough
|   |-- 05_real_world_applications.md      <- Industry use cases
|
|-- examples/
|   |-- example_01_vectors_and_similarity.py   <- NumPy only, no extra libraries
|   |-- example_02_embeddings_from_scratch.py  <- Build a tiny embedding model by hand
|   |-- example_03_chromadb_basics.py          <- Your first ChromaDB queries
|   |-- example_04_document_search.py          <- Search engine from scratch
|   |-- example_05_semantic_search.py          <- Full semantic search system
|
|-- exercises/
|   |-- exercise_01_similarity_basics.py   <- Practice: cosine similarity
|   |-- exercise_02_build_vector_store.py  <- Practice: build a mini vector store
|   |-- exercise_03_semantic_search.py     <- Practice: semantic search
|
|-- projects/
    |-- document_search_engine/
        |-- main.py                        <- Capstone project
```

## Learning Path

```
Lesson 01: What is a vector? What is a vector database?
    |
    v
Lesson 02: How do we turn text into vectors? (Embeddings)
    |
    v
Lesson 03: ChromaDB -- using a real vector database
    |
    v
Lesson 04: Build a real document search engine
    |
    v
Lesson 05: Real-world applications and next steps
    |
    v
Project: Full Document Search Engine
```

## Comparison: Traditional DB vs Vector DB

```
Traditional Database (SQL Server):
  Query: SELECT * FROM docs WHERE title = 'car repair'
  Result: Only finds exact match for "car repair"

Vector Database (ChromaDB):
  Query: find_similar("how to fix automobile")
  Result: Finds "car repair", "vehicle maintenance", "engine troubleshooting"
          (even though none of those exact words were searched!)
```

## Estimated Time

  - Lessons:   5 x 30 minutes = 2.5 hours
  - Examples:  5 x 20 minutes = 1.5 hours
  - Exercises: 3 x 30 minutes = 1.5 hours
  - Project:   2-3 hours

  Total: ~7-8 hours

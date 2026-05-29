# Module 10.5 — RAG Without Vectors (BM25 / TF-IDF)

## What This Module Is About

You learned vector RAG in Module 10 (ChromaDB + embeddings).
This module teaches how to do RAG WITHOUT a vector database or embedding model.

BM25 and TF-IDF are keyword-based retrieval methods.
They find documents that share words with your query.
No GPU, no model download, no embedding computation needed.

---

## Why This Matters

| Situation | Best retrieval |
|-----------|---------------|
| No GPU, low resources | BM25 |
| Query uses exact product names, codes, IDs | BM25 |
| Query uses meaning, synonyms, concepts | Vector RAG |
| Need highest accuracy | Hybrid (BM25 + vectors) |
| Quick prototype, no model setup | BM25 |

---

## C# Analogy

```
BM25  =  SQL full-text search (CONTAINS, FREETEXT)
          Fast, no ML needed, works on exact keywords

Vector RAG  =  ML-powered semantic search
               Understands meaning, not just words
```

---

## Lessons

| # | Lesson | Topic |
|---|--------|-------|
| 1 | `01_tfidf.md` | TF-IDF: term frequency, inverse document frequency, the math |
| 2 | `02_bm25.md` | BM25: smarter keyword ranking, formula, parameters |
| 3 | `03_rag_with_bm25.md` | Building a RAG pipeline with BM25 — no vectors |
| 4 | `04_hybrid_search.md` | Hybrid search: combine BM25 score + embedding score |
| 5 | `05_when_to_use_what.md` | Decision guide: BM25 vs vector RAG vs hybrid |

---

## Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| `rank-bm25` | BM25 search | `pip install rank-bm25` |
| `sklearn` | TF-IDF vectorizer | `pip install scikit-learn` |
| `numpy` | Score math | already installed |

---

## Prerequisites

- Module 10 (Vector Databases) — you know what RAG is
- Module 08 (Prompt Engineering) — you know how to use retrieved docs in a prompt

---

## Connects To

- Module 10.8 (Semantic Search) — adds neural re-ranking on top of BM25
- Module 11 (LLM Agents) — agents use retrieval to answer questions
- Capstone (Chat with Codebase) — hybrid search gives best results for code

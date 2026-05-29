# Lesson 03: Building a RAG Pipeline with BM25

## Glossary (Read This First!)

| Term | Plain English Definition |
|------|--------------------------|
| **RAG** | Retrieval-Augmented Generation. Find relevant documents first, then feed them to an LLM to generate an answer. The LLM does not guess from memory — it reads the documents. |
| **Retrieval** | The search step: find the top-K most relevant documents for a query. |
| **Augmentation** | Adding retrieved documents to the LLM prompt as context. |
| **Generation** | The LLM reads the prompt + documents and writes an answer. |
| **Context window** | The maximum text an LLM can read at once. You can only pass so many documents. |
| **Chunking** | Splitting large documents into smaller pieces so they fit in the context window. |
| **Top-K** | How many documents to retrieve. Common values: 3, 5, 10. |
| **Prompt template** | The text pattern used to combine query + retrieved docs into the LLM input. |
| **Grounding** | Making LLM answers traceable to specific source documents. Reduces hallucination. |
| **Hallucination** | When an LLM makes up information not in its training data or provided context. RAG reduces this. |

---

## Part 1: Why RAG Exists

LLMs have two key weaknesses:

```
+------------------------------------------------------------------+
|  LLM WEAKNESSES WITHOUT RAG                                      |
+------------------------------------------------------------------+
|                                                                  |
|  1. KNOWLEDGE CUTOFF                                             |
|     LLM training ended on a specific date.                       |
|     Ask about events after that date → LLM cannot know.         |
|     Example: "What did our company announce last week?"          |
|              LLM has never seen your company's announcements.    |
|                                                                  |
|  2. PRIVATE DATA                                                 |
|     LLM trained on public internet.                              |
|     Your internal docs, databases, code were NOT in training.    |
|     Example: "What does our payment service do?"                 |
|              LLM has no idea. It might hallucinate an answer.    |
|                                                                  |
|  RAG SOLUTION:                                                   |
|  Before asking the LLM, find relevant documents.                 |
|  Pass them in the prompt as context.                             |
|  LLM reads the documents and answers from them.                  |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 2: RAG Pipeline Overview

```
+------------------------------------------------------------------+
|  RAG PIPELINE (BM25 version)                                     |
+------------------------------------------------------------------+
|                                                                  |
|  INDEXING PHASE (run once, or when documents change)             |
|  ────────────────────────────────────────────────                |
|                                                                  |
|  Raw documents                                                   |
|       |                                                          |
|       v                                                          |
|  [Chunker] -- split into ~200 word pieces                        |
|       |                                                          |
|       v                                                          |
|  [Tokenizer] -- split each chunk into word tokens               |
|       |                                                          |
|       v                                                          |
|  [BM25 Index] -- build inverted index, compute IDF               |
|       |                                                          |
|       v                                                          |
|  Index stored in memory (or serialized to disk)                  |
|                                                                  |
|  ────────────────────────────────────────────────                |
|                                                                  |
|  QUERY PHASE (run on every user question)                        |
|  ────────────────────────────────────────                        |
|                                                                  |
|  User question: "How does payment processing work?"              |
|       |                                                          |
|       v                                                          |
|  [Tokenize query] -- ["How", "does", "payment", "processing"]   |
|       |                                                          |
|       v                                                          |
|  [BM25 search] -- score all chunks, return top-5                 |
|       |                                                          |
|       v                                                          |
|  Top-5 relevant chunks                                           |
|       |                                                          |
|       v                                                          |
|  [Build prompt] -- question + retrieved chunks                   |
|       |                                                          |
|       v                                                          |
|  [LLM] -- reads prompt, generates answer                         |
|       |                                                          |
|       v                                                          |
|  Answer to user                                                  |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 3: Chunking Strategy

Documents must be split into chunks before indexing.
Chunks must be small enough to fit in context, but large enough to have meaning.

```
+------------------------------------------------------------------+
|  CHUNKING STRATEGIES                                             |
+------------------------------------------------------------------+
|                                                                  |
|  Fixed-size chunking                                             |
|  ────────────────────                                            |
|  Split every N words. Simple but may cut sentences in half.      |
|  Good for: uniform docs, quick prototype                         |
|                                                                  |
|    "The cat sat on the mat. The dog ran in the park."            |
|    Chunk 1: "The cat sat"   (3 words)                            |
|    Chunk 2: "on the mat."   (3 words)                            |
|    Problem: "The cat sat" has no context about the mat!          |
|                                                                  |
|  Overlapping windows                                             |
|  ────────────────────                                            |
|  Chunk size=5, stride=3. Chunks overlap by 2 words.              |
|  Solves context loss at chunk boundaries.                        |
|                                                                  |
|    Chunk 1: "The cat sat on the"                                 |
|    Chunk 2: "on the mat. The dog"                                |
|    Chunk 3: "The dog ran in the"                                 |
|                                                                  |
|  Sentence/paragraph boundaries                                   |
|  ────────────────────────────                                    |
|  Split at ". " or "\n\n". Preserves natural language units.      |
|  Best for: articles, documentation, Q&A pairs                   |
|                                                                  |
|  Semantic chunking (advanced)                                    |
|  ────────────────────────────                                    |
|  Split when topic changes. Requires an embedding model.          |
|  Best quality, but more complex.                                 |
|                                                                  |
+------------------------------------------------------------------+
```

**Rule of thumb:** 200–500 words per chunk. Overlap 10–20% of chunk size.

---

## Part 4: The Prompt Template

After retrieval, you need to build a prompt that combines:
1. System instruction (how the LLM should behave)
2. Retrieved context (the documents found by BM25)
3. User question

```
+------------------------------------------------------------------+
|  EXAMPLE RAG PROMPT TEMPLATE                                     |
+------------------------------------------------------------------+
|                                                                  |
|  SYSTEM:                                                         |
|  You are a helpful assistant. Answer the question using ONLY     |
|  the information in the context below. If the answer is not      |
|  in the context, say "I don't know based on the provided docs."  |
|                                                                  |
|  CONTEXT:                                                        |
|  [Document 1 - chunk from payments.md]                           |
|  "The payment service uses Stripe for card processing.           |
|  Payments are validated with a 3D Secure challenge..."            |
|                                                                  |
|  [Document 2 - chunk from architecture.md]                       |
|  "All payment events are published to a Kafka topic named        |
|  payment.completed after successful authorization..."             |
|                                                                  |
|  [Document 3 - chunk from api.md]                                |
|  "POST /api/v2/payments accepts amount, currency, and            |
|  payment_method_id. Returns a payment intent object..."          |
|                                                                  |
|  QUESTION:                                                       |
|  How does payment processing work?                               |
|                                                                  |
+------------------------------------------------------------------+
```

The LLM reads the three context chunks and writes an answer grounded in them.
If none of the chunks contain the answer, a well-prompted LLM will say "I don't know."

C# analogy:
```csharp
// RAG prompt is like a SQL query with a JOIN:
// SELECT answer FROM llm
// JOIN retrieved_docs ON question
// WHERE docs.relevance_score > threshold

// You are giving the LLM a "reading list" and asking it to summarize.
// It is much more reliable than asking it to recall from memory.
```

---

## Part 5: Preprocessing for BM25

Raw text needs cleaning before BM25 indexing. Quality of preprocessing
directly affects retrieval quality.

```
+------------------------------------------------------------------+
|  TEXT PREPROCESSING PIPELINE                                     |
+------------------------------------------------------------------+
|                                                                  |
|  Raw text:  "The Cat SAT on the Mat!!!"                          |
|       |                                                          |
|  Lowercase: "the cat sat on the mat!!!"                          |
|       |                                                          |
|  Remove punctuation: "the cat sat on the mat"                    |
|       |                                                          |
|  Tokenize (split on spaces): ["the","cat","sat","on","the","mat"]|
|       |                                                          |
|  Remove stop words: ["cat", "sat", "mat"]                        |
|       |                                                          |
|  (Optional) Stemming: ["cat", "sat", "mat"]  (no change here)   |
|             "running" → "run", "cats" → "cat"                   |
|       |                                                          |
|  Final tokens: ["cat", "sat", "mat"]                             |
|                                                                  |
+------------------------------------------------------------------+
```

**Stop words** to remove: "the", "a", "an", "is", "are", "was", "were",
"in", "on", "at", "to", "for", "of", "and", "or", "but", "it", "this"

---

## Part 6: Full BM25-RAG Code Sketch

This is the complete structure of a BM25 RAG system.
No libraries beyond `rank-bm25` and a simple LLM call.

```python
import json
from rank_bm25 import BM25Okapi

# ── STEP 1: Load documents ──────────────────────────────────────
documents = [
    {"id": "doc1", "text": "Payment service uses Stripe for card processing."},
    {"id": "doc2", "text": "All payments publish events to Kafka after completion."},
    {"id": "doc3", "text": "POST /api/v2/payments accepts amount and currency."},
    # ... more documents
]

# ── STEP 2: Preprocess (tokenize, lowercase, remove stop words) ──
STOP_WORDS = {"the", "a", "an", "is", "are", "in", "on", "for", "of"}

def preprocess(text):
    tokens = text.lower().split()
    return [t for t in tokens if t not in STOP_WORDS]

tokenized_corpus = [preprocess(doc["text"]) for doc in documents]

# ── STEP 3: Build BM25 index ────────────────────────────────────
bm25 = BM25Okapi(tokenized_corpus)

# ── STEP 4: Query function ──────────────────────────────────────
def retrieve(query, top_k=3):
    query_tokens = preprocess(query)
    scores = bm25.get_scores(query_tokens)

    # Sort document indices by score (highest first)
    ranked_indices = sorted(range(len(scores)),
                            key=lambda i: scores[i],
                            reverse=True)

    # Return top-K documents with their scores
    return [(documents[i], scores[i]) for i in ranked_indices[:top_k]]

# ── STEP 5: Build RAG prompt ────────────────────────────────────
def build_prompt(question, retrieved_docs):
    context = "\n\n".join(
        f"[Document {i+1}]\n{doc['text']}"
        for i, (doc, _) in enumerate(retrieved_docs)
    )
    return f"""Answer the question using only the context below.
If not in context, say "I don't know."

CONTEXT:
{context}

QUESTION: {question}"""

# ── STEP 6: Use it ──────────────────────────────────────────────
query = "how does payment processing work"
results = retrieve(query, top_k=3)
prompt = build_prompt(query, results)

# Pass prompt to any LLM (Ollama, OpenAI, Anthropic, etc.)
# response = llm.generate(prompt)
print(prompt)
```

---

## Part 7: Limitations of BM25-RAG

```
+------------------------------------------------------------------+
|  BM25-RAG LIMITATIONS                                            |
+------------------------------------------------------------------+
|                                                                  |
|  1. Vocabulary mismatch                                          |
|     User asks "fast car" -- document says "quick automobile"     |
|     BM25 scores 0. Document not retrieved. Answer: "I don't know"|
|     Fix: add synonyms, or switch to hybrid search (Lesson 04)    |
|                                                                  |
|  2. No semantic understanding                                     |
|     BM25 matches words, not meaning.                             |
|     "What is the capital of France?" vs doc: "Paris is France's  |
|     most populous city and serves as the seat of government."    |
|     BM25 misses because "capital" is not in doc.                 |
|                                                                  |
|  3. Order of words ignored                                        |
|     "dog bites man" and "man bites dog" get same BM25 score.     |
|                                                                  |
|  4. Context dependency                                            |
|     "it" in a document refers to something in the previous       |
|     sentence. BM25 chunks split context. Retrieval may return    |
|     a chunk where "it" has no referent.                          |
|                                                                  |
+------------------------------------------------------------------+
```

Despite these limitations, BM25 is often the FIRST thing to try.
It is fast, cheap, and works well when users know the exact keywords.

---

## Quiz

**Q1.** What are the two phases of a RAG pipeline?

**Q2.** Name two reasons why a company would use RAG instead of just asking the LLM directly.

**Q3.** What is chunking and why is it needed?

**Q4.** Why is overlapping chunking better than fixed-size chunking?

**Q5.** If a user asks "What is the capital of France?" and the document says "Paris is the seat of government", will BM25 find it? Why or why not?

---

## Answers

**A1.** Indexing phase (process and index all documents once) and Query phase (search + retrieve + prompt + generate on each user question).

**A2.** (1) LLM has a knowledge cutoff — does not know about recent events. (2) LLM was not trained on private/internal data (docs, databases, code).

**A3.** Chunking splits large documents into smaller pieces. Needed because the LLM context window is finite — you cannot pass a 500-page document to the LLM in one go.

**A4.** Fixed-size chunks cut at arbitrary points, splitting sentences and losing context. Overlapping chunks share some content at boundaries, preserving context at the cost of slight redundancy.

**A5.** Probably no. BM25 looks for the word "capital" in documents. If the document never uses that word, BM25 scores it 0. This is the vocabulary mismatch problem — solved by semantic/hybrid search.

# Capstone Project: Chat with Codebase (Offline RAG App)

## What You Are Building

A fully offline app that lets anyone ask plain-English questions about a code repository
and get real answers -- no internet, no API keys, no cloud services.

Example:
```
You: What does the payment service do?
Bot: The payment service handles Stripe webhooks, validates charge events,
     and writes results to the orders table in PostgreSQL. It lives in
     src/payments/webhook_handler.py and is called from the FastAPI router
     in src/api/routes.py.
```

---

## Architecture Diagram

```
INDEXING PHASE (run once, or when code changes)
================================================
Your repo files
    |
    v
[chunker.py]          -- split each file into overlapping text chunks
    |
    v
[embedder.py]         -- convert each chunk to a vector (list of numbers)
    |                    using Ollama nomic-embed-text model
    v
[ChromaDB]            -- save vectors + original text to local disk
    (chroma_db/)


QUERY PHASE (run every time user asks a question)
=================================================
User types question
    |
    v
[embedder.py]         -- convert question to a vector
    |
    v
[ChromaDB]            -- find top-5 chunks whose vectors are closest
    |                    to the question vector (similarity search)
    v
[retriever.py]        -- build a prompt:
    |                    "Here is relevant code: [chunks]
    |                     Answer this question: [question]"
    v
[Ollama LLM]          -- local LLM reads prompt, generates answer
(Mistral 7B or        -- runs on YOUR machine, no internet needed
 CodeLlama 7B)
    |
    v
Answer shown to user
```

---

## What Each File Does

```
capstone_chat_with_codebase/
|
+-- config.py         Settings: repo path, model names, chunk sizes, DB path
+-- indexer.py        PHASE 1: Walk repo, chunk files, embed, store in ChromaDB
+-- retriever.py      PHASE 2: Take question, find relevant chunks, call LLM
+-- chat.py           Console REPL: type questions, get answers in terminal
+-- app.py            Streamlit web UI: browser-based chat interface
+-- reindex.py        Smart re-indexer: only re-embed files that changed
+-- utils/
    +-- chunker.py    Split a file's text into overlapping chunks
    +-- embedder.py   Call Ollama to convert text -> vector
```

---

## C#/.NET Analogy

| This project | .NET equivalent |
|---|---|
| ChromaDB | SQL Server / Entity Framework (stores data) |
| Embeddings (vectors) | Fingerprints of text meaning |
| Similarity search | WHERE cosine_distance < threshold |
| Ollama | A local Web API (like a self-hosted REST service) |
| Streamlit | ASP.NET Razor Pages (but Python, instant setup) |
| chunker.py | A text parser / tokenizer utility class |
| RAG pipeline | Repository pattern + query handler + response formatter |

---

## Tech Stack (All Free, All Local)

| Component | Tool | Why |
|---|---|---|
| Local LLM | Ollama + Mistral 7B | Runs on CPU/GPU, no API key |
| Embeddings | Ollama nomic-embed-text | Same Ollama, no extra setup |
| Vector DB | ChromaDB | Local file-based, no server needed |
| Web UI | Streamlit | Python-native, 10 lines = working UI |
| Language | Python 3.10+ | Standard for ML/AI |

---

## Prerequisites

1. Install Ollama: https://ollama.com (free, ~500MB)
2. Pull the models (one-time download):
   ```bash
   ollama pull mistral        # ~4GB, the answering LLM
   ollama pull nomic-embed-text  # ~300MB, for embeddings
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Step 1 -- Index your codebase
```bash
# Edit config.py to set REPO_PATH to your repo folder
python indexer.py
# This may take 2-10 minutes for a large repo
```

### Step 2 -- Chat via console
```bash
python chat.py
```

### Step 3 -- Chat via browser
```bash
streamlit run app.py
# Opens http://localhost:8501
```

### Step 4 -- Re-index after code changes
```bash
python reindex.py
# Only re-embeds changed files, much faster than full re-index
```

---

## Learning Objectives

After completing this project you will understand:

1. **RAG (Retrieval-Augmented Generation)** -- how LLMs answer questions about YOUR data
2. **Embeddings** -- why text becomes vectors and how similarity search works
3. **ChromaDB** -- how vector databases store and query embeddings
4. **Ollama** -- how to run LLMs locally without any cloud services
5. **Chunking strategy** -- why you split documents and how overlap helps context
6. **End-to-end AI pipeline** -- index -> retrieve -> generate -> display

---

## Module Connections

| Concept used | Where you learned it |
|---|---|
| Embeddings, vectors | M05 Building LLM |
| ChromaDB, similarity search | M10 Vector Databases |
| BM25 keyword fallback | M10.5 RAG Without Vectors |
| RAG pipeline orchestration | M11 LLM Agents |
| Ollama, local deployment | M14 Deploying LLMs |
| Streamlit UI | M14.5 (optional) |

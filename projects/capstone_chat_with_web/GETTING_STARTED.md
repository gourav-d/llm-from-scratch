# Capstone: Chat with Web

Ask questions about ANY web page. Give it links, it learns, you ask.

---

## What You're Building

```
You give URLs  →  App fetches + cleans HTML  →  Splits into chunks
                                                        ↓
You ask question  ←  LLM answers  ←  Top 5 relevant chunks retrieved
```

This is called **RAG (Retrieval-Augmented Generation)** -- how real AI assistants
like Perplexity and Bing Chat work under the hood.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: FETCH                              │
│                                                                     │
│  URL → HTTP GET → raw HTML → trafilatura/justext → clean text       │
│                                                                     │
│  File: fetcher.py                                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ clean text
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 2: STORE                              │
│                                                                     │
│  clean text → chunks (1000 chars) → embeddings (384-dim vectors)   │
│                                  → ChromaDB (vector database)       │
│                                                                     │
│  File: store.py                                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │ vectors on disk
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 3: QUERY (RAG)                        │
│                                                                     │
│  Question → embed → search ChromaDB → top 5 chunks                 │
│          → build prompt (question + chunks) → Ollama LLM → Answer  │
│                                                                     │
│  File: rag.py                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Setup (One Time)

### Step 1: Install Python dependencies
```bash
cd projects/capstone_chat_with_web
pip install -r requirements.txt
```

### Step 2: Install Ollama (local LLM server)
1. Download from https://ollama.ai
2. Install and run it
3. Pull a model:
```bash
ollama pull mistral
```

### Step 3: Verify Ollama is running
```bash
ollama serve        # start the server if not running
ollama list         # should show "mistral" in the list
```

---

## Usage

### Add URLs to your knowledge base
```bash
# Add one URL
python app.py add https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)

# Add multiple URLs at once
python app.py add https://url1.com https://url2.com https://url3.com
```

### Ask questions
```bash
python app.py ask "What is the attention mechanism?"
python app.py ask "What are the main advantages of transformers over RNNs?"
python app.py ask "Give me 5 key facts about this topic"
```

### Get a summary
```bash
python app.py summarize
```

### List what's indexed
```bash
python app.py list
```

### Interactive chat mode
```bash
python app.py chat
# Then just type questions, type 'quit' to exit
```

### Clear everything and start fresh
```bash
python app.py clear
```

---

## Example Session

```bash
# Index 2 Wikipedia pages on transformers and attention
python app.py add \
  https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture) \
  https://en.wikipedia.org/wiki/Attention_(machine_learning)

# Ask questions
python app.py ask "What problem does the attention mechanism solve?"
python app.py ask "Who invented the transformer architecture?"
python app.py ask "What is self-attention?"
python app.py ask "How does positional encoding work?"

# Get overview
python app.py summarize
```

---

## Key Concepts (for .NET Developers)

| RAG Concept | .NET Analogy |
|-------------|--------------|
| Chunk | `List<string>` split by length |
| Embedding | A semantic hash of text (384 floats) |
| Vector DB (ChromaDB) | SQL table with a "similarity search" index |
| Cosine similarity | Dot product distance between two vectors |
| RAG prompt | Dependency injection of facts into LLM context |
| Ollama API | Local REST API (like calling `HttpClient.PostAsync`) |

---

## Why RAG Instead of Fine-tuning?

| | Fine-tuning | RAG |
|-|-------------|-----|
| Training data needed | Millions of tokens | Your URLs |
| GPU required | Yes (hours/days) | No |
| Update knowledge | Retrain | Just add new URL |
| Answers grounded in source | Sometimes not | Yes -- uses actual text |
| Setup time | Days | Minutes |

**RAG is the right tool** when you want an LLM to answer from specific documents.
Fine-tuning is for changing the model's *behavior* or *style*, not for adding *knowledge*.

---

## Exercises

1. **Basic**: Index 3 Wikipedia pages on a topic you find interesting. Ask 5 questions.

2. **Compare extractors**: In `fetcher.py`, change `EXTRACTOR_ORDER` to put "naive" first.
   Ask the same question. Is the answer worse? Why?

3. **Chunk size experiment**: In `config.py`, change `CHUNK_SIZE` to 300 (very small).
   Re-index the same URLs. Do answers get better or worse? What about speed?

4. **Multi-source Q&A**: Index pages from TWO different sources on the same topic
   (e.g., one Wikipedia + one blog post). Ask a question -- does the answer use both sources?

5. **Stretch goal**: Add a `python app.py quiz` command that auto-generates 5 quiz questions
   from the indexed content and checks your answers.

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All settings (chunk size, model, paths) |
| `fetcher.py` | Phase 1: URL fetch + HTML extraction |
| `store.py` | Phase 2: Chunking + embedding + ChromaDB storage |
| `rag.py` | Phase 3: RAG query pipeline + Ollama integration |
| `app.py` | CLI entry point (add/ask/summarize/list/chat/clear) |
| `requirements.txt` | Python dependencies |
| `chroma_db/` | Auto-created: vector database on disk |

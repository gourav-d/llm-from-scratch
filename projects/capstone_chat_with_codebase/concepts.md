# Concepts: Chat with Codebase (Offline RAG App)

This project combines everything learned in M05 through M14 into one working application.
Each concept below maps to a specific module you already completed.

---

## 1. The Core Problem: LLMs Don't Know Your Code

A large language model (LLM) like Mistral was trained on public internet data.
It has never seen your private codebase.

If you ask "What does `calculate_tax()` do in our project?", the LLM cannot answer
because it has no idea what files are in your repository.

**The solution: RAG (Retrieval-Augmented Generation)**

Instead of hoping the LLM already knows the answer, we:
1. Find relevant pieces of YOUR code
2. Put those pieces into the LLM's prompt
3. Ask the LLM to answer using that context

```
Without RAG:
  Question ---------> LLM ---------> Answer (guessing, wrong)

With RAG:
  Question --> find relevant code --> LLM reads YOUR code --> Answer (correct)
```

C# analogy: like passing a method's source code as a string argument to an AI assistant
instead of expecting it to already know the implementation.

**Learned in:** M11 (LLM Agents -- RAG pipeline design)

---

## 2. Embeddings: Text Becomes Numbers

To find "relevant code", we need a way to measure similarity between text pieces.
We can't compare raw strings -- "exception handling" and "try-catch block" look
completely different as text but mean the same thing.

**The solution: embeddings (vectors)**

An embedding model converts text into a list of numbers (a vector) where:
- Similar meanings --> similar vectors (pointing in the same direction)
- Different meanings --> different vectors (pointing in different directions)

```
"How do I handle errors?"   --> [0.12, -0.34, 0.89, ...]
"Exception handling guide"  --> [0.11, -0.33, 0.91, ...]   <-- very similar!
"Pizza recipe"              --> [-0.77, 0.22, -0.45, ...]   <-- very different
```

In this project, we use the `nomic-embed-text` model via Ollama to create embeddings.
It runs locally -- no internet, no API key needed.

**Learned in:** M05 (Building LLM -- how embeddings represent meaning)

---

## 3. Chunking: Files Are Too Big to Embed Whole

LLMs have a "context window" -- a maximum amount of text they can read at once.
A typical limit is 4,000-8,000 tokens (~3,000-6,000 words).

A large file might have 100,000 characters. We cannot send the whole file.

**The solution: chunking with overlap**

Split each file into small pieces (chunks) of ~1,500 characters.
Add overlap (200 chars) so concepts cut at boundaries appear in both adjacent chunks.

```
File text: "ABCDEFGHIJKLMNOPQRSTUVWXYZ..."
                                  
Chunk 1:   "ABCDEFGHIJ"          (chars 0-1499)
Chunk 2:   "HIJKLMNOP"           (chars 1300-2799)  <-- overlap: "HIJ" in both
Chunk 3:   "NOPQRSTUVW"          (chars 2600-4099)  <-- overlap: "NOP" in both
```

Why overlap matters: if a function definition is split at char 1500, overlap ensures
the function name appears in both chunks, so searches for it find the correct result.

**Learned in:** M10.5 (RAG Without Vectors -- text splitting strategies)

---

## 4. ChromaDB: A Database for Vectors

We need to store thousands of chunks and their embeddings, then search them quickly.
A regular database (SQL Server, SQLite) stores rows and searches by exact match.
We need to search by SIMILARITY -- "find chunks whose meaning is close to this question."

**The solution: a vector database**

ChromaDB stores:
- The text of each chunk
- The embedding vector for that chunk
- Metadata (which file, which chunk number)

When we search, we give it a query vector and it returns the K most similar chunks
using cosine distance (angle between vectors -- smaller angle = more similar meaning).

```
SQL (exact match):   SELECT * FROM chunks WHERE text LIKE '%error%'
Vector DB (semantic): SELECT TOP 5 * FROM chunks ORDER BY cosine_distance(embedding, query_embedding)
```

ChromaDB saves its data to disk as files -- no server needed, no config, just a folder.

**Learned in:** M10 (Vector Databases -- ChromaDB, similarity search, HNSW index)

---

## 5. Ollama: Run LLMs Locally Without Internet

The LLM that generates the final answer needs to run on your machine without
any cloud service, API key, or subscription.

**The solution: Ollama**

Ollama is a free tool that:
- Downloads quantized LLM models (~4 GB instead of 100+ GB)
- Runs them as a local REST API on port 11434
- Works offline once models are downloaded

```
Our Python code:
  POST http://localhost:11434/api/generate
  {"model": "mistral", "prompt": "Here is relevant code... answer this question..."}

Ollama:
  Loads the model
  Generates a response
  Returns: {"response": "The calculate_tax function..."}
```

This is the same HTTP pattern as calling any web API in .NET (HttpClient.PostAsync).

**Learned in:** M14 (Deploying LLMs -- Ollama, local inference, quantization)

---

## 6. The Full RAG Pipeline: How the Pieces Connect

```
INDEXING PHASE (once, or when code changes)
===========================================
for each file in your repo:
    text = read file
    chunks = split text into 1500-char pieces with 200-char overlap
    for each chunk:
        vector = nomic-embed-text.embed(chunk)   # 768 numbers
        chromadb.store(chunk_text, vector, {source: file_path})

QUERY PHASE (every time user asks a question)
=============================================
question = "What does calculate_tax() do?"

# Step 1: Embed the question (same model as chunks)
question_vector = nomic-embed-text.embed(question)

# Step 2: Find the 5 most similar chunks in ChromaDB
top_5_chunks = chromadb.query(question_vector, n=5)

# Step 3: Build a prompt
prompt = f"""
    You are a code assistant. Answer using only this context:
    {top_5_chunks}
    
    Question: {question}
    Answer:
"""

# Step 4: Call the local LLM
answer = ollama.generate("mistral", prompt)

# Step 5: Show the answer + which files were used
print(answer)
```

---

## 7. Smart Re-indexing: Only Re-embed What Changed

A large codebase might take 30 minutes to fully index.
If you change 3 files, you don't want to wait 30 minutes again.

**The solution: a file manifest (hash-based diff)**

The manifest is a JSON file: `{"path/to/file.py": "abc123md5hash", ...}`

On re-index:
1. Compute the current MD5 hash of every file
2. Compare against the saved manifest
3. Only re-embed files where the hash changed

```python
# Current hash vs saved hash
if manifest.get(file_path) == compute_md5(file_path):
    skip_this_file()   # unchanged -- no work needed
else:
    re_embed(file_path)   # changed -- update the database
```

This is the same pattern as git's object store -- content-addressed storage where
the hash IS the identity of a file's content.

**Learned in:** M10.5 (RAG Without Vectors -- incremental indexing patterns)

---

## 8. Why This Architecture Scales

| Problem | Solution used | Why it works |
|---------|--------------|--------------|
| LLM doesn't know your code | RAG -- pass relevant chunks | LLM reads YOUR code before answering |
| Can't send whole file to LLM | Chunking -- 1500-char pieces | Only relevant pieces sent |
| Keyword search misses synonyms | Embeddings -- vector similarity | Meaning-based search, not text match |
| Need to search thousands of chunks | ChromaDB -- HNSW index | O(log N) search, not O(N) brute force |
| Can't afford cloud LLM per query | Ollama -- local inference | No API cost, works offline |
| Re-indexing takes too long | MD5 manifest -- diff-based updates | Only changed files re-embedded |

---

## 9. C#/.NET Analogy: The Full Picture

| This project's component | .NET equivalent |
|--------------------------|----------------|
| `config.py` | `appsettings.json` |
| `utils/chunker.py` | Utility class with string-splitting methods |
| `utils/embedder.py` | `HttpClient` wrapper for the embedding REST API |
| `indexer.py` | Repository seeder / database migration script |
| `chromadb` | SQL Server (but stores vectors, not rows) |
| `retriever.py` | CQRS Query Handler: input=question, output=answer+sources |
| `chat.py` | Console application with `while(true) { Console.ReadLine() }` |
| `app.py` | ASP.NET Razor Pages (but auto-rendered from Python) |
| `reindex.py` | Incremental sync job (like a background service with change detection) |
| `file_manifest.json` | Change tracking table (like Entity Framework's `__EFMigrationsHistory`) |

---

## 10. What to Explore Next

After completing this capstone, these are natural next steps:

1. **Try a different LLM** -- change `LLM_MODEL` in `config.py` to `"llama3.2"` or `"codellama"`.
   CodeLlama is specifically trained for code questions and may give better answers.

2. **Add BM25 hybrid search** -- the Exercise 3 in `quiz_and_exercises.md` walks you through
   combining keyword search (exact function names) with semantic search (meaning-based).

3. **Point it at a real project** -- change `REPO_PATH` in `config.py` to one of your actual
   C# projects. Ask it to explain methods, trace call chains, find where something is defined.

4. **Increase TOP_K** -- change `TOP_K_RESULTS` from 5 to 8 for more context per answer.
   Trade-off: more context = better answers but slower and more tokens per call.

5. **Stream the output** -- in `retriever.py`, change `"stream": False` to `"stream": True`
   in `call_ollama_llm()`. This makes words appear as they're generated (like ChatGPT typing effect).
   Requires changing how you read the response (line-by-line from a streaming HTTP response).

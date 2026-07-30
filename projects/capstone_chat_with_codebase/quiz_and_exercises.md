# Capstone Quiz and Exercises: Chat with Codebase

---

## Quiz (5 Questions)

### Q1: What does RAG stand for, and why do we need it?

**A)** Retrieval-Augmented Generation. We need it because LLMs cannot read
       files on your disk -- we must retrieve relevant text and pass it in the prompt.

**B)** Random Access Generation. We use it to speed up LLM responses.

**C)** Recursive Attention Graph. It is the architecture inside transformers.

**D)** Rule-Augmented Grammar. It adds grammar rules to LLM output.

**Answer: A**

Without RAG, if you ask "What does calculate_tax() do?", the LLM has no
knowledge of YOUR code. RAG retrieves the relevant code snippets and puts
them in the prompt, so the LLM can read the actual code before answering.

---

### Q2: Why do we split files into chunks instead of sending the whole file to the LLM?

**A)** Because Python cannot read files longer than 1500 characters.

**B)** To make the code easier to test.

**C)** Because LLMs have a context window limit -- they can only read a fixed
       amount of text at once. A large file might have 100,000 characters
       but the LLM only accepts 4,000-8,000 tokens.

**D)** Because ChromaDB only stores strings up to 1500 characters.

**Answer: C**

Context window is the maximum input size. By chunking, we only send the
MOST RELEVANT pieces of the file -- not the whole thing. This also means
the LLM focuses on what matters rather than getting distracted by unrelated code.

---

### Q3: What is the purpose of OVERLAP in chunking?

Consider these two chunks (chunk_size=20, overlap=5):

  Chunk 1: "def foo(x, y):\n    re"
  Chunk 2: "    return x + y\n\n"

With overlap=5:
  Chunk 1: "def foo(x, y):\n    re"
  Chunk 2: "    re\n    return x + y"

**A)** Overlap makes the index smaller.

**B)** Overlap ensures that a concept split across a boundary still appears
       in at least one complete chunk, preserving context.

**C)** Overlap is required by ChromaDB's API.

**D)** Overlap speeds up the embedding process.

**Answer: B**

Without overlap, if a function definition is split exactly at the "def" line,
one chunk has the function body with no name, and the other has the name with
no body. Overlap means the function name appears in both adjacent chunks,
so searches for that function will find the right content.

---

### Q4: In ChromaDB, what does "cosine distance" measure?

**A)** The number of words that two texts share (word overlap).

**B)** The Euclidean (straight-line) distance between two vectors in space.

**C)** The angle between two vectors -- 0.0 means identical direction
       (very similar meaning), 1.0 means opposite direction (unrelated).

**D)** The edit distance (number of character changes) between two strings.

**Answer: C**

Cosine distance measures the angle, not the magnitude.
Two short and long texts can mean the same thing even if their vector magnitudes
differ -- cosine distance captures the DIRECTION (meaning) rather than length.

Example:
  "How do I handle errors?"   --> embedding vector pointing in direction A
  "Exception handling guide"  --> embedding vector also pointing in direction A
  Cosine distance: ~0.05  (almost identical direction = similar meaning)

---

### Q5: What is the manifest file (file_manifest.json) used for in reindex.py?

**A)** It stores the conversation history between user and assistant.

**B)** It stores the MD5 hash of each indexed file, so the re-indexer can
       detect which files changed and only re-embed those, instead of
       re-indexing the entire codebase every time.

**C)** It is the ChromaDB database file where vectors are stored.

**D)** It is a configuration file that Ollama reads to find models.

**Answer: B**

The manifest is a JSON file: {"file.py": "abc123md5...", ...}
On re-index, we compare current hashes vs manifest hashes.
Only files with changed hashes need re-embedding. This saves hours of
embedding time for large repositories.

---

## Lab Exercises

---

### Exercise 1: Change the Chunk Size and Measure Impact

**Goal:** Understand how chunk_size affects retrieval quality.

**Task:**

1. Open `config.py`
2. Change `CHUNK_SIZE` from 1500 to 500
3. Run `python indexer.py` with a full re-index
4. Ask the question: "How does chunking work?"
5. Note how many sources are cited and what they contain

Then:
1. Change `CHUNK_SIZE` to 3000
2. Re-index
3. Ask the same question
4. Compare the answers

**Questions to think about:**

- With CHUNK_SIZE=500, were the answers more or less precise?
- With CHUNK_SIZE=3000, did the LLM get more context or was it overwhelmed?
- What happens if CHUNK_SIZE is smaller than a single function?

**Expected finding:**

Smaller chunks = more precise retrieval but may lose cross-function context.
Larger chunks = more context per result but may include irrelevant code.
1500 chars is a reasonable default for code files.

---

### Exercise 2: Add a New File Type to the Index

**Goal:** Practice modifying config and re-indexing.

**Task:**

1. Create a new file in the repo: `notes/architecture_notes.txt`
   Write 3-4 paragraphs describing a (made-up) system architecture.

2. Notice that `.txt` is already in `ALLOWED_EXTENSIONS` in `config.py`.

3. Run `python reindex.py` (not the full indexer!)

4. Verify the new file was indexed:
   - Check the output: "Files added: 1"
   - Ask a question that matches your notes: "What is the architecture?"

5. Now edit your `.txt` file and change one paragraph.

6. Run `python reindex.py` again.

7. Verify: "Files updated: 1" (not "Files added" this time)

**What you learned:**

- How the diff-based re-indexer detects new vs. changed vs. deleted files
- The manifest file tracks content hashes, not just file names
- The same file path gets updated chunks, not duplicate entries

---

### Exercise 3: Implement a Simple BM25 Fallback

**Goal:** Combine keyword search (BM25) with semantic search (embeddings).

**Background:**

Sometimes semantic search misses exact function names or variable names.
"What does `calculate_tax` do?" might find conceptually similar code
but miss the exact function if the embedding space doesn't match perfectly.

BM25 keyword search finds exact text matches (like Ctrl+F across the codebase).
Hybrid search = BM25 score + cosine score combined.

**Task:**

Create a new file `utils/bm25_search.py` with this implementation:

```python
# utils/bm25_search.py
"""
Simple BM25 keyword search over indexed chunks.
BM25 = Best Match 25, the algorithm behind Elasticsearch and Lucene.
It scores documents by how often the query terms appear, weighted by
document frequency (common words score lower than rare words).

This gives us keyword fallback when semantic search misses exact names.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
import chromadb


def bm25_search(query: str, top_k: int = 5) -> list:
    """
    Search ChromaDB documents using simple keyword matching.

    This is NOT full BM25 (which needs an inverted index).
    This is a simplified version: count query word occurrences in each chunk,
    return the chunks with the most matches.

    For production BM25, use the rank-bm25 library:
        from rank_bm25 import BM25Okapi
    """
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
    except Exception:
        return []

    # Get ALL documents from ChromaDB (we'll score them in Python)
    # For large repos this is slow -- production would use an inverted index
    all_data = collection.get(include=["documents", "metadatas"])
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    ids = all_data["ids"]

    # Split query into words (lowercase)
    # In C#: query.ToLower().Split(' ')
    query_words = set(query.lower().split())

    # Score each document by how many query words it contains
    scored = []
    for doc, meta, doc_id in zip(documents, metadatas, ids):
        doc_words = set(doc.lower().split())
        # Count of query words found in document
        score = len(query_words.intersection(doc_words))
        if score > 0:
            scored.append({
                "text": doc,
                "source": meta.get("source", ""),
                "score": score,
                "id": doc_id,
            })

    # Sort by score descending (highest match first)
    # In C#: scored.OrderByDescending(s => s.Score).Take(top_k).ToList()
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    results = bm25_search("chunk_text overlap sliding window")
    for r in results:
        print(f"Score {r['score']}: {r['source']}")
        print(r["text"][:200])
        print("---")
```

Then:
1. Run `python utils/bm25_search.py` to test it
2. Compare results between:
   - Semantic search: `retriever.retrieve_chunks("What is chunking overlap?")`
   - Keyword search: `bm25_search("chunk_text overlap sliding window")`

**Questions:**
- Which finds results faster?
- Which finds results more accurately for exact function names?
- When would you use each?

**Solution note:**
Semantic search wins for natural language questions.
Keyword search wins for exact identifiers like function names, variable names, error codes.
Hybrid = combine both scores: `final_score = 0.7 * cosine_score + 0.3 * bm25_score`

---

## Summary: Key Takeaways

```
RAG Pipeline:
  files -> chunks -> vectors -> ChromaDB
                                  |
  question -> vector -> similarity search -> top-K chunks
                                                  |
                              prompt = instruction + chunks + question
                                                  |
                                             Ollama LLM
                                                  |
                                               answer

Key concepts:
  Chunking       = split files into pieces the LLM can read
  Overlap        = avoid cutting concepts at chunk boundaries
  Embeddings     = vectors that represent text meaning
  Cosine search  = find chunks with similar meaning to the question
  RAG            = give the LLM relevant context before asking the question
  Manifest       = track file hashes so we only re-embed changed files
```

Connections to prior modules:
  - Embeddings explained in M05 (Building LLM)
  - ChromaDB vector search in M10 (Vector Databases)
  - BM25 keyword search in M10.5 (RAG Without Vectors)
  - RAG pipeline design in M11 (LLM Agents)
  - Ollama local deployment in M14 (Deploying LLMs)

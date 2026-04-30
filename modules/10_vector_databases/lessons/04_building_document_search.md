# Lesson 04: Building a Document Search Engine

## Learning Objectives

By the end of this lesson, you will be able to:
1. Describe the full architecture of a semantic search system
2. Understand the pipeline: raw text -> embeddings -> vector DB -> search results
3. Plan and build the components of a real document search engine

---

## GLOSSARY

```
Pipeline:
  A sequence of steps where the output of one step feeds into the next.
  In C#: like a series of IEnumerable transformations or a processing chain.
  Example: text -> tokenize -> embed -> store -> query -> rank -> display

Chunking:
  Splitting long documents into smaller pieces before embedding.
  WHY: Embedding models have a maximum input length (e.g., 512 tokens).
       A 10-page document must be split into chunks that fit this limit.
  Example: Split a 5000-word article into 20 chunks of 250 words each.

Retrieval:
  The step where we find the most relevant documents for a query.
  In RAG (Retrieval-Augmented Generation): the "R" step.
  The output of retrieval feeds into an LLM for final answer generation.

Re-ranking:
  A second pass after retrieval that re-orders results for quality.
  Retrieval (fast, approximate) -> Re-ranking (slow, precise)
  Example: Retrieve top 20 with vectors, then re-rank with a cross-encoder model.

Cross-Encoder:
  A model that takes (query, document) as a PAIR and gives a relevance score.
  More accurate than cosine similarity, but much slower (cannot pre-compute).
  Used in re-ranking after a fast first-pass retrieval.

Bi-Encoder:
  A model that encodes query and document SEPARATELY (independently).
  sentence-transformers is a bi-encoder.
  Fast because documents are encoded once, upfront.
  Slightly less accurate than cross-encoder for scoring.

Index:
  In a vector database, an index is the internal data structure used to
  find nearest vectors fast (without checking every single vector).
  Common algorithms: HNSW (Hierarchical Navigable Small World), IVF-Flat.
  You do not need to understand these -- ChromaDB handles it automatically.

Batch Processing:
  Processing multiple items at once instead of one at a time.
  More efficient because the embedding model can use GPU parallelism.
  Example: Embed 100 documents at once instead of one by one.
```

---

## Part 1: The Full Architecture

A document search engine has these components:

```
[1. DATA INGESTION]
  Raw documents (files, database, web pages)
       |
       v
  Text Cleaning  (remove HTML tags, fix encoding, normalize whitespace)
       |
       v
  Chunking  (split long docs into pieces that fit the embedding model)

[2. EMBEDDING]
  Each chunk -> Embedding Model -> Vector [0.23, -0.14, 0.88, ...]

[3. STORAGE]
  (vector, original text, metadata) -> ChromaDB collection

[4. QUERY]
  User types a query
       |
       v
  Query text -> Embedding Model -> Query vector
       |
       v
  ChromaDB: find top-k most similar vectors

[5. DISPLAY]
  Return matched documents to user
  (optionally: re-rank with a cross-encoder for higher quality)
```

---

## Part 2: Step 1 -- Data Ingestion

Before storing documents, you need to prepare them.

### Text Cleaning

Remove noise that could confuse the embedding model:

```python
import re

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)    # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)       # Collapse whitespace
    text = text.strip()                    # Remove leading/trailing spaces
    return text
```

### Chunking

Most embedding models have a token limit (e.g., 512 tokens for MiniLM).
A 2000-word article has about 1500 tokens -- too long.

Simple sentence-based chunking:

```python
def chunk_text(text, chunk_size=200, overlap=50):
    """
    Split text into overlapping chunks of roughly chunk_size words.
    overlap: how many words to repeat between consecutive chunks
             (helps avoid losing context at chunk boundaries)
    """
    words = text.split()                    # Split into list of words
    chunks = []

    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]     # Take chunk_size words
        chunk_text  = " ".join(chunk_words)          # Join back into string
        chunks.append(chunk_text)
        i += chunk_size - overlap                    # Advance with overlap

    return chunks
```

Example:
```
Words: [w1, w2, w3, w4, w5, w6, w7, w8]
chunk_size=4, overlap=1

Chunk 1: [w1, w2, w3, w4]
Chunk 2: [w4, w5, w6, w7]   <- w4 is repeated (overlap)
Chunk 3: [w7, w8]
```

The overlap ensures that a sentence split across chunk boundaries
is still fully captured in at least one chunk.

---

## Part 3: Step 2 -- Embedding

Use sentence-transformers to convert text to vectors:

```python
from sentence_transformers import SentenceTransformer

# Load the model (downloads ~90 MB on first run)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed a single string
vector = model.encode("How do I reset my password?")
# vector is a NumPy array of shape (384,)

# Embed multiple strings at once (MUCH faster than one at a time)
texts = [
    "How do I reset my password?",
    "What payment methods do you accept?",
    "How do I cancel my subscription?"
]
vectors = model.encode(texts)
# vectors is a NumPy array of shape (3, 384) -- 3 documents, 384 dimensions each
```

### Why all-MiniLM-L6-v2?

```
Model name:   all-MiniLM-L6-v2
Dimensions:   384
Speed:        Very fast (runs on CPU in milliseconds)
Quality:      Good general-purpose semantic similarity
File size:    ~90 MB
Good for:     Semantic search, sentence similarity, information retrieval
```

For higher quality (but slower):
  - all-mpnet-base-v2: 768 dimensions, slower, better quality
  - For OpenAI API: text-embedding-ada-002 (1536 dimensions, cloud-based)

---

## Part 4: Step 3 -- Storage in ChromaDB

```python
import chromadb

# Create a persistent client (data saved to disk)
client = chromadb.PersistentClient(path="./search_db")

# Get or create the collection
collection = client.get_or_create_collection(
    name="support_articles",
    metadata={"hnsw:space": "cosine"}    # Use cosine similarity for this collection
)

# Add documents in batches (more efficient than one at a time)
batch_size = 100

for i in range(0, len(all_chunks), batch_size):
    batch_texts     = all_chunks[i : i + batch_size]
    batch_ids       = [f"chunk_{j}" for j in range(i, i + len(batch_texts))]
    batch_metadatas = [{"source": "manual", "chunk_index": j}
                       for j in range(i, i + len(batch_texts))]

    collection.add(
        documents=batch_texts,
        ids=batch_ids,
        metadatas=batch_metadatas
    )
    print(f"Added chunks {i} to {i + len(batch_texts)}")
```

Note: ChromaDB will automatically embed the documents using its default
embedding function. You can also pre-compute embeddings and pass them
as the `embeddings=` parameter.

---

## Part 5: Step 4 -- Search

```python
def search_documents(query_text, collection, n_results=5, filter_meta=None):
    """
    Search for documents similar to query_text.
    query_text:  The user's search query (plain text)
    collection:  The ChromaDB collection to search in
    n_results:   How many results to return
    filter_meta: Optional metadata filter (e.g., {"category": "billing"})
    Returns: list of (similarity, text, metadata) tuples
    """
    # Build the query parameters
    query_params = {
        "query_texts": [query_text],    # ChromaDB auto-embeds this
        "n_results":   n_results
    }

    if filter_meta:                     # Add metadata filter if provided
        query_params["where"] = filter_meta

    # Run the query
    results = collection.query(**query_params)

    # Format the results
    output = []
    for doc, dist, meta in zip(
        results["documents"][0],        # The matched document texts
        results["distances"][0],        # Their distances (lower = more similar)
        results["metadatas"][0]         # Their metadata
    ):
        similarity = 1.0 - dist         # Convert distance to similarity
        output.append((similarity, doc, meta))

    return output                       # List of (similarity, text, metadata)
```

Usage:

```python
results = search_documents("how to change my password", collection, n_results=3)

for similarity, text, meta in results:
    print(f"Similarity: {similarity:.3f}")
    print(f"Text:       {text}")
    print(f"Source:     {meta.get('source', 'unknown')}")
    print()
```

---

## Part 6: Full System Overview

Putting it all together in a class:

```python
class DocumentSearchEngine:
    """
    Simple semantic search engine backed by ChromaDB.
    In C# terms: a service class with Ingest() and Search() methods.
    """

    def __init__(self, db_path="./search_db", collection_name="documents"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.doc_count = self.collection.count()

    def ingest(self, documents, metadatas=None):
        """Add documents to the search engine."""
        ids = [f"doc_{self.doc_count + i}" for i in range(len(documents))]
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas or [{}] * len(documents)
        )
        self.doc_count += len(documents)
        print(f"Added {len(documents)} documents. Total: {self.doc_count}")

    def search(self, query, top_k=5, where=None):
        """Search for documents similar to the query."""
        params = {"query_texts": [query], "n_results": top_k}
        if where:
            params["where"] = where
        return self.collection.query(**params)

    def count(self):
        """Return number of documents stored."""
        return self.collection.count()
```

---

## Part 7: Scaling Considerations

```
Documents: 100       -> In-memory ChromaDB, instant queries
Documents: 10,000    -> PersistentClient ChromaDB, sub-second queries
Documents: 1 million -> Consider Pinecone, Weaviate, or pgvector
Documents: 1 billion -> Dedicated vector DB cluster (Weaviate, Milvus)
```

For most projects (company knowledge base, small product catalog, personal notes),
ChromaDB with PersistentClient is more than sufficient.

---

## Key Takeaways

1. The pipeline is: text -> clean -> chunk -> embed -> store -> query -> display

2. Chunking is important: split long documents into pieces that fit the model's
   token limit (typically 256-512 tokens per chunk).

3. Use overlap between chunks (50-100 words) to avoid losing context at boundaries.

4. all-MiniLM-L6-v2 is a good starting embedding model: fast, small, good quality.

5. Store documents in batches for efficiency.

6. ChromaDB handles the embedding, indexing, and similarity search for you.

---

## Next

Lesson 05: Real-World Applications
  - How ChatGPT uses vector databases for memory (RAG)
  - How GitHub Copilot finds relevant code
  - How to scale beyond ChromaDB
  - What to build next

"""
store.py -- Phase 2: Chunk text, embed it, store in ChromaDB.

Pipeline:
  clean text --> chunks --> embeddings (vectors) --> ChromaDB

WHY CHUNK?
  LLMs have token limits. A webpage may have 50,000 tokens.
  We split into small chunks, embed each one, then at query time
  we only retrieve the 5 most relevant chunks -- not the whole page.
  C# analogy: like splitting a List<string> into pages (pagination).

WHY EMBED?
  An embedding converts text into a list of numbers (a vector).
  Similar meaning = similar numbers = close together in vector space.
  "king" and "queen" are closer than "king" and "table".
  C# analogy: like a hash function, but semantically aware.
"""

import chromadb
from sentence_transformers import SentenceTransformer

import config


# ---------------------------------------------------------------------------
# ChromaDB client (singleton -- created once, reused)
# ---------------------------------------------------------------------------
# C# analogy: like a static DbContext in Entity Framework

_chroma_client = None
_collection = None
_embedding_model = None


def _get_collection():
    """Get (or create) ChromaDB collection. Lazy initialization."""
    global _chroma_client, _collection
    if _collection is None:
        # PersistentClient saves data to disk -- survives restarts
        # C# analogy: like SQL Server LocalDB (file-based, no server process)
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        # get_or_create = create if not exists, else return existing
        # C# analogy: like context.Set<T>().FirstOrDefault() ?? new T()
        _collection = _chroma_client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            # cosine similarity = measure angle between vectors
            # better than euclidean distance for text
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _get_embedding_model():
    """Load sentence-transformers model. Downloads ~80MB on first run."""
    global _embedding_model
    if _embedding_model is None:
        print(f"  Loading embedding model: {config.EMBEDDING_MODEL}")
        print("  (First run: downloads ~80MB. Subsequent runs: instant.)")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, source_url: str) -> list[dict]:
    """
    Split text into overlapping chunks.

    Example with CHUNK_SIZE=10, CHUNK_OVERLAP=3:
      Text:   "ABCDEFGHIJKLMNOP"
      Chunk1: "ABCDEFGHIJ"        (0-10)
      Chunk2: "HIJKLMNOP"         (7-17)  <-- 3 chars overlap with chunk1
      Chunk3: ...

    This overlap prevents losing context at chunk boundaries.
    C# analogy: for(int i=0; i<text.Length; i += CHUNK_SIZE - CHUNK_OVERLAP)

    Returns list of dicts with text + metadata.
    """
    chunks = []
    step = config.CHUNK_SIZE - config.CHUNK_OVERLAP

    for i, start in enumerate(range(0, len(text), step)):
        end = start + config.CHUNK_SIZE
        chunk_text = text[start:end]

        # Skip chunks that are too short to be meaningful
        if len(chunk_text) < 50:
            continue

        chunks.append({
            "text":       chunk_text,
            "source_url": source_url,
            "chunk_index": i,
        })

        # Stop when we've passed the end of the text
        if end >= len(text):
            break

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert list of strings into list of embedding vectors.

    Each vector is a list of 384 floats (for all-MiniLM-L6-v2).
    384 dimensions = the "semantic address" of the text in vector space.

    C# analogy: like texts.Select(t => embedModel.Encode(t)).ToList()
    """
    model = _get_embedding_model()
    # encode() returns a numpy array -- .tolist() converts to plain Python list
    vectors = model.encode(texts, show_progress_bar=True)
    return vectors.tolist()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_documents(documents: list[dict]) -> int:
    """
    Chunk, embed, and store a list of fetched documents.

    documents = output of fetcher.fetch_multiple()
    Returns total number of chunks stored.

    C# analogy: like context.AddRange(entities); context.SaveChanges();
    """
    collection = _get_collection()

    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], doc["url"])
        all_chunks.extend(chunks)
        print(f"  Chunked '{doc['url']}' -> {len(chunks)} chunks")

    if not all_chunks:
        print("  No chunks to store.")
        return 0

    print(f"\n  Embedding {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    # ChromaDB needs: ids (unique strings), documents (text), embeddings, metadatas
    # C# analogy: inserting rows with a primary key + extra columns
    ids = [f"{c['source_url']}::chunk{c['chunk_index']}" for c in all_chunks]
    metadatas = [{"source_url": c["source_url"], "chunk_index": c["chunk_index"]}
                 for c in all_chunks]

    # upsert = insert if new, update if already exists (idempotent)
    # Re-adding the same URL just overwrites -- safe to run multiple times
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"  Stored {len(all_chunks)} chunks in ChromaDB.")
    return len(all_chunks)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search(query: str, top_k: int = None) -> list[dict]:
    """
    Find the most relevant chunks for a query.

    Steps:
      1. Embed the query (same model, same vector space)
      2. ChromaDB finds the closest chunk vectors (cosine similarity)
      3. Return top_k results with their text + metadata

    C# analogy: like running a SQL SELECT ORDER BY similarity DESC TOP 5
    """
    if top_k is None:
        top_k = config.TOP_K_RESULTS

    collection = _get_collection()

    # Check we have data
    count = collection.count()
    if count == 0:
        return []

    # Embed the query using the same model used to embed chunks
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),  # can't ask for more than we have
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns nested lists -- [0] unwraps the outer batch dimension
    chunks = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":       text,
            "source_url": metadata["source_url"],
            "score":      round(1 - distance, 4),  # convert distance -> similarity
        })

    return chunks


def get_indexed_urls() -> list[str]:
    """Return list of unique URLs currently indexed."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    # Get all metadata -- extract unique source_urls
    all_data = collection.get(include=["metadatas"])
    urls = list({m["source_url"] for m in all_data["metadatas"]})
    return sorted(urls)


def clear_all():
    """Delete all stored data. USE WITH CAUTION."""
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    try:
        client.delete_collection(config.COLLECTION_NAME)
        print("Cleared all stored data.")
    except Exception:
        print("Nothing to clear.")

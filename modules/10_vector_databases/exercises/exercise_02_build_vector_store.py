"""
Exercise 02: Build a Mini Vector Store
=========================================

GOAL: Build a simple in-memory vector store from scratch using only Python lists and NumPy.
No ChromaDB -- you implement the storage and search yourself.

This teaches you what a vector database does INTERNALLY.

WHAT YOU WILL BUILD:
  A MiniVectorStore class with these methods:
    - add(id, text, vector, metadata=None)  -> store a document
    - search(query_vector, top_k=5)         -> find similar documents
    - get(id)                               -> get a document by ID
    - delete(id)                            -> remove a document
    - count()                               -> number of stored documents
    - get_all_ids()                         -> list all stored IDs

GLOSSARY
--------
  Vector Store: Any system that stores vectors and supports similarity search.
  In-Memory: Data stored in RAM. Fast but lost when program ends.
  CRUD: Create, Read, Update, Delete -- the four basic data operations.
  .items(): Python dict method that returns (key, value) pairs for looping.
  sorted(): Sort a list. sorted(my_list, key=func, reverse=True) -> descending.

LIBRARIES NEEDED: numpy (pip install numpy)
"""

import numpy as np

np.random.seed(42)

print("=" * 60)
print("EXERCISE 02: Build a Mini Vector Store")
print("=" * 60)


# ==============================================================================
# TASK 1: Complete the MiniVectorStore Class
# ==============================================================================

print("\n--- Task 1: MiniVectorStore Implementation ---")

class MiniVectorStore:
    """
    A simple in-memory vector store built from scratch.
    Stores documents as (id, text, vector, metadata) tuples.
    Uses cosine similarity for search.

    In C# terms: like a List<VectorDocument> with a Search() method.
    """

    def __init__(self):
        """
        Initialize an empty vector store.
        Use three parallel dictionaries (all keyed by document ID):
          self._texts:     {id -> text string}
          self._vectors:   {id -> numpy array}
          self._metadatas: {id -> dict}
        """
        # TODO 1a: Initialize the three storage dictionaries
        self._texts     = {}    # Maps doc_id -> text string
        self._vectors   = {}    # Maps doc_id -> numpy vector
        self._metadatas = {}    # Maps doc_id -> metadata dict (or {})

    def add(self, doc_id, text, vector, metadata=None):
        """
        Add a document to the store.

        Parameters:
          doc_id:   unique string ID (like a primary key)
          text:     the original text (string)
          vector:   numpy array representing the document's embedding
          metadata: optional dict with extra info (default: empty dict)

        Raises:
          ValueError if doc_id already exists (to prevent duplicate IDs)
        """
        # TODO 1b: Check if doc_id already exists; if so, raise ValueError
        # Hint: check if doc_id is in self._texts
        pass    # Replace with your code

        # TODO 1c: Store text, vector, and metadata in the three dictionaries
        # Use {} as default if metadata is None
        pass    # Replace with your code

    def get(self, doc_id):
        """
        Retrieve a document by its ID.

        Returns:
          dict with keys: 'id', 'text', 'vector', 'metadata'
        Raises:
          KeyError if doc_id not found
        """
        # TODO 1d: Check if doc_id exists; if not, raise KeyError with a message
        pass    # Replace with your code

        # TODO 1e: Return a dict with id, text, vector, metadata for this doc_id
        return None    # Replace with your code

    def delete(self, doc_id):
        """
        Remove a document from the store.
        Raises KeyError if doc_id not found.
        """
        # TODO 1f: Check if doc_id exists; if not, raise KeyError
        pass

        # TODO 1g: Delete from all three dictionaries using del or .pop()
        pass

    def count(self):
        """Return the number of documents stored."""
        # TODO 1h: Return the number of items in self._texts
        return None    # Replace with your code

    def get_all_ids(self):
        """Return a sorted list of all stored document IDs."""
        # TODO 1i: Return sorted list of keys from self._texts
        return None    # Replace with your code

    def search(self, query_vector, top_k=5, metadata_filter=None):
        """
        Find the top_k most similar documents to query_vector.

        Parameters:
          query_vector:    numpy array (the search query embedding)
          top_k:           how many results to return
          metadata_filter: optional dict for filtering by metadata
                           Example: {"category": "billing"}
                           Only returns docs where metadata matches ALL filter keys.

        Returns:
          list of dicts: [{'id', 'text', 'similarity', 'metadata'}, ...]
          sorted by similarity DESCENDING (most similar first)
        """
        if self.count() == 0:              # Handle empty store
            return []

        results = []

        for doc_id in self._texts:         # Loop through all stored documents
            # TODO 1j: If metadata_filter is set, check if this doc's metadata matches
            # Skip this doc if any filter key doesn't match its metadata value
            # Hint: for key, value in metadata_filter.items(): if metadata[key] != value: skip
            if metadata_filter:
                pass    # Replace with your filter logic

            # TODO 1k: Compute cosine similarity between query_vector and this doc's vector
            # Use self._vectors[doc_id] to get the stored vector
            # Use the _cosine_similarity method below
            sim = None    # Replace with your code

            results.append({
                "id":         doc_id,
                "text":       self._texts[doc_id],
                "similarity": round(float(sim), 4),
                "metadata":   self._metadatas[doc_id]
            })

        # TODO 1l: Sort results by 'similarity' key in DESCENDING order
        # Hint: results.sort(key=lambda x: x['similarity'], reverse=True)
        pass    # Replace with your sort code

        # TODO 1m: Return only the top_k results
        return None    # Replace with your code

    @staticmethod
    def _cosine_similarity(a, b):
        """
        Private helper: compute cosine similarity between two numpy arrays.
        You do not need to modify this -- it is provided for you.
        """
        dot   = np.dot(a, b)
        mag_a = np.linalg.norm(a)
        mag_b = np.linalg.norm(b)
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return float(dot / (mag_a * mag_b))


# ==============================================================================
# TASK 2: Test Your MiniVectorStore
# ==============================================================================

print("\n--- Task 2: Test the MiniVectorStore ---")

store = MiniVectorStore()

print(f"  Empty store count: {store.count()}  (expected 0)")

# Add some documents with 4-dimensional vectors
# Dimensions represent: [animals, food, technology, sports]
test_docs = [
    ("doc_01", "My dog loves to fetch sticks",
     np.array([0.9, 0.1, 0.0, 0.2]), {"category": "pets"}),

    ("doc_02", "Cat videos are very popular online",
     np.array([0.8, 0.0, 0.3, 0.0]), {"category": "pets"}),

    ("doc_03", "Best pasta recipe with tomato sauce",
     np.array([0.0, 0.9, 0.0, 0.1]), {"category": "food"}),

    ("doc_04", "Python is a great programming language",
     np.array([0.0, 0.0, 0.9, 0.0]), {"category": "tech"}),

    ("doc_05", "Running a 10k marathon for beginners",
     np.array([0.0, 0.0, 0.0, 0.9]), {"category": "sports"}),

    ("doc_06", "Grilled chicken with vegetables",
     np.array([0.0, 0.8, 0.0, 0.2]), {"category": "food"}),
]

for doc_id, text, vector, metadata in test_docs:
    store.add(doc_id, text, vector, metadata)

print(f"  After adding 6 docs: {store.count()}  (expected 6)")
print(f"  All IDs: {store.get_all_ids()}")

# Test get()
doc = store.get("doc_01")
print(f"\n  get('doc_01'): text = '{doc['text']}'")
print(f"               category = '{doc['metadata']['category']}'")

# Test search without filter
query = np.array([0.85, 0.05, 0.0, 0.1])    # Animal-focused query
results_all = store.search(query, top_k=3)

print(f"\n  Search (animal query), top 3:")
for r in results_all:
    print(f"    [{r['id']}] sim={r['similarity']:.4f}  {r['text']}")

# Test search WITH filter
results_pets = store.search(query, top_k=3, metadata_filter={"category": "pets"})
print(f"\n  Search (animal query, pets only), top 3:")
for r in results_pets:
    print(f"    [{r['id']}] sim={r['similarity']:.4f}  {r['text']}")

# Test delete
store.delete("doc_05")
print(f"\n  After deleting doc_05: count = {store.count()}  (expected 5)")

# Assertions
assert store.count() == 5, "Should have 5 documents after delete"
assert store.get_all_ids()[0] == "doc_01", "First ID should be doc_01"
assert results_all[0]["id"] in ["doc_01", "doc_02"], "Top result should be a pet doc"
assert all(r["metadata"]["category"] == "pets" for r in results_pets), "Filter should only return pets"

print("\n  Task 2: All assertions passed!")


# ==============================================================================
# TASK 3: Add an upsert Method
# ==============================================================================

print("\n--- Task 3: Add upsert (Update or Insert) ---")

print("""
SQL has INSERT OR UPDATE (MERGE in SQL Server).
ChromaDB has collection.upsert().

Implement upsert() for MiniVectorStore:
  - If doc_id ALREADY EXISTS: update the text, vector, and metadata.
  - If doc_id does NOT exist: insert it (same as add()).
""")

def upsert(store, doc_id, text, vector, metadata=None):
    """
    Insert or update a document in the store.
    If doc_id exists: update it. If not: insert it.
    Parameters: same as store.add()
    """
    # TODO 3a: Check if doc_id already exists in store._texts
    # If it exists: update self._texts[doc_id], self._vectors[doc_id], self._metadatas[doc_id]
    # If it does not exist: call store.add() to insert it
    pass    # Replace with your code

# Test upsert
print(f"  Count before upsert: {store.count()}")

# Update an existing document
upsert(store, "doc_01",
       "My golden retriever loves to fetch tennis balls",
       np.array([0.92, 0.08, 0.0, 0.15]),
       {"category": "pets", "updated": True})

updated = store.get("doc_01")
print(f"  Updated doc_01 text: '{updated['text']}'")
print(f"  Updated metadata: {updated['metadata']}")

# Insert a new document via upsert
upsert(store, "doc_07",
       "Machine learning and deep learning basics",
       np.array([0.0, 0.0, 0.95, 0.05]),
       {"category": "tech"})

print(f"  Count after upsert (new): {store.count()}  (expected 6)")

assert store.count() == 6, "Should have 6 documents after upsert insert"
assert "updated" in store.get("doc_01")["metadata"], "doc_01 metadata should have 'updated' key"
print("  Task 3: Assertions passed!")


# ==============================================================================
# TASK 4: Implement a Simple Inverted Index (Challenge)
# ==============================================================================

print("\n--- Task 4: Compare Pure Vector Store vs Keyword Search ---")

print("""
Now let us compare our MiniVectorStore with a simple keyword search.

Keyword search: check if any query word appears in the document text.
Vector search: compute cosine similarity between embeddings.
""")

def keyword_search(query, store, top_k=3):
    """
    Simple keyword search on documents in the store.
    Scores by number of query words found in each document.
    Returns: list of (score, doc_id, text) tuples
    """
    query_words = set(query.lower().split())    # Set of query words
    results = []

    for doc_id in store.get_all_ids():
        doc = store.get(doc_id)
        doc_words = set(doc["text"].lower().split())
        overlap = len(query_words & doc_words)    # Count matching words
        if overlap > 0:
            results.append((overlap, doc_id, doc["text"]))

    results.sort(key=lambda x: x[0], reverse=True)    # Sort by overlap
    return results[:top_k]

# Test queries where semantic search should win
test_queries = [
    {
        "query_text":   "I want to adopt a puppy",
        "query_vector": np.array([0.88, 0.05, 0.0, 0.05]),
        "note":         "Should find pet/animal docs"
    },
    {
        "query_text":   "coding and programming",
        "query_vector": np.array([0.0, 0.0, 0.9, 0.0]),
        "note":         "Should find tech docs"
    },
]

for q in test_queries:
    print(f"  Query: '{q['query_text']}'  ({q['note']})")
    print()

    # Keyword results
    kw = keyword_search(q["query_text"], store, top_k=2)
    print(f"    KEYWORD SEARCH:")
    if kw:
        for score, doc_id, text in kw:
            print(f"      [{doc_id}] score={score} words: {text}")
    else:
        print(f"      No results (no matching words)")

    # Vector results
    vec = store.search(q["query_vector"], top_k=2)
    print(f"    VECTOR SEARCH:")
    for r in vec:
        print(f"      [{r['id']}] sim={r['similarity']:.4f}: {r['text']}")

    print()

print("  -> Notice: vector search finds 'dog' for 'puppy' query (semantic match)")
print("  -> Keyword search finds nothing for 'puppy' -- it is not in any document")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("EXERCISE 02 COMPLETE")
print("=" * 60)

print("""
WHAT YOU BUILT:

A MiniVectorStore class with:
  - add(id, text, vector, metadata)  -> store a document
  - get(id)                          -> retrieve by ID
  - delete(id)                       -> remove a document
  - search(query_vector, top_k)      -> find similar documents
  - count()                          -> number of stored documents
  - get_all_ids()                    -> list all IDs

You also implemented:
  - upsert(): insert or update (like SQL MERGE / ChromaDB upsert)
  - Comparison between keyword search and vector search

KEY LESSON:
  This is essentially what ChromaDB does internally (but much faster,
  using optimized indexing structures like HNSW).

  ChromaDB also:
  - Persists data to disk
  - Handles millions of vectors efficiently (you would hit RAM limits)
  - Auto-embeds documents (you provided pre-computed vectors)
  - Has metadata filtering built in

In production, use ChromaDB (or Pinecone/pgvector).
In learning, building from scratch teaches you what is happening inside.
""")

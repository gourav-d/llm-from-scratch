# Lesson 03: ChromaDB Hands-On

## Learning Objectives

By the end of this lesson, you will be able to:
1. Create a ChromaDB collection (the vector database equivalent of a SQL table)
2. Add documents to the collection
3. Run a similarity query and read the results
4. Filter results using metadata

---

## GLOSSARY

```
ChromaDB:
  An open-source vector database written in Python.
  Very easy to use -- no server setup required (just pip install chromadb).
  In .NET terms: like SQLite but for vectors. No server, just a file.

Collection:
  A group of related embeddings stored in ChromaDB.
  Equivalent to a table in SQL Server.
  Example: a collection called "support_articles" stores all your help articles.

Document:
  In ChromaDB, "document" means the original text you want to store.
  ChromaDB will automatically convert it to a vector using a built-in embedding model.
  Example: "How to reset your password" is a document.

Embedding Function:
  The model that converts text to vectors.
  ChromaDB has a default embedding function built-in.
  You can also use your own (e.g., from sentence-transformers).

Metadata:
  Extra data you attach to each document (not used for similarity, but for filtering).
  Example: {"category": "billing", "date": "2024-01-15", "author": "support team"}

Query:
  The search request. You provide text, ChromaDB finds the most similar documents.

n_results:
  How many results to return. Like TOP N in SQL.
  query(query_texts=["my question"], n_results=5) returns the 5 most similar docs.

Distance:
  ChromaDB returns a "distance" for each result.
  Lower distance = more similar.
  0.0 = identical. 2.0 = very different (for cosine, max is 2.0).
  Note: ChromaDB returns distance, not similarity. distance = 1 - cosine_similarity.

Persistent Storage:
  When ChromaDB saves data to disk (so it survives after the program ends).
  Like a database file -- re-running the program does not lose the data.
  Contrast with in-memory storage (data is lost when the program ends).

In-Memory Storage:
  Data is kept in RAM only. Fast, but lost when program exits.
  Good for testing and small experiments.
```

---

## Part 1: What Is ChromaDB?

ChromaDB is a vector database that:
  - Runs entirely in Python (no external server needed)
  - Automatically converts text to vectors (embeddings) for you
  - Stores documents + their metadata + their vectors
  - Finds the most similar documents to any query

### ChromaDB vs SQL Server

```
Feature              | SQL Server                  | ChromaDB
---------------------+-----------------------------+----------------------------
Installation         | Install SQL Server service  | pip install chromadb
Server required?     | Yes (localhost or remote)   | No (runs in your Python code)
Query language       | SQL (SELECT, WHERE, JOIN)   | Python methods (query, get)
What it stores       | Rows with typed columns     | Documents + vectors + metadata
Best for             | Exact lookups, transactions | Similarity search
Filter syntax        | WHERE clause in SQL         | where={"key": "value"} in Python
Storage              | .mdf/.ldf files             | Folder of files (or in-memory)
```

---

## Part 2: ChromaDB Core Concepts

ChromaDB organizes data like this:

```
ChromaDB Client
|
|-- Collection: "support_articles"
|       |-- ID: "doc_001"
|       |       Document:  "How to reset your password"
|       |       Vector:    [0.23, -0.14, 0.88, ...]  (auto-generated)
|       |       Metadata:  {"category": "account", "author": "support"}
|       |
|       |-- ID: "doc_002"
|       |       Document:  "Billing and payment options"
|       |       Vector:    [0.05, 0.77, -0.32, ...]  (auto-generated)
|       |       Metadata:  {"category": "billing", "author": "finance"}
|       ...
|
|-- Collection: "product_catalog"
        |-- ...
```

This is very similar to a SQL database with tables.

---

## Part 3: Your First ChromaDB Program

Here is the minimum code to use ChromaDB:

```python
import chromadb

# Step 1: Create a client (in-memory for testing)
client = chromadb.Client()

# Step 2: Create a collection (like creating a SQL table)
collection = client.create_collection("my_docs")

# Step 3: Add documents
collection.add(
    documents=["The dog ran in the park", "Cats sleep all day", "I bought a laptop"],
    ids=["doc1", "doc2", "doc3"]
)

# Step 4: Query for similar documents
results = collection.query(
    query_texts=["puppy playing outside"],
    n_results=2
)

print(results)
```

Output:
```
{
  'ids':       [['doc1', 'doc2']],
  'documents': [['The dog ran in the park', 'Cats sleep all day']],
  'distances': [[0.12, 0.89]]
}
```

"doc1" (dog in park) is most similar to "puppy playing outside" (distance 0.12 = very close).
"doc2" (cats sleep) is less similar (distance 0.89 = further away).

---

## Part 4: Understanding the Results

ChromaDB query results come back as a dictionary.
Each value is a list-of-lists (because you can send multiple queries at once).

```python
results = {
  'ids':        [['doc1', 'doc2']],       # List of IDs found (inner list = one query)
  'documents':  [['The dog ran...', 'Cats sleep...']], # The original text
  'distances':  [[0.12, 0.89]],           # How different (lower = more similar)
  'metadatas':  [[None, None]],           # Metadata if you stored any
  'embeddings': None                      # The actual vectors (not returned by default)
}
```

To read results cleanly:

```python
# Get results for the first (and only) query
first_query_results = results['documents'][0]   # ['The dog ran...', 'Cats sleep...']
first_query_distances = results['distances'][0]  # [0.12, 0.89]

# Print in a readable format
for doc, dist in zip(first_query_results, first_query_distances):
    similarity = 1.0 - dist    # Convert distance to similarity score
    print(f"Similarity: {similarity:.2f}  ->  {doc}")
```

---

## Part 5: Adding Metadata

Metadata lets you filter results -- like a WHERE clause in SQL:

```python
collection.add(
    documents=[
        "How to reset your password",
        "How to change your email address",
        "Billing and payment methods",
        "How to cancel your subscription"
    ],
    ids=["doc1", "doc2", "doc3", "doc4"],
    metadatas=[
        {"category": "account", "priority": "high"},
        {"category": "account", "priority": "low"},
        {"category": "billing", "priority": "high"},
        {"category": "billing", "priority": "high"}
    ]
)

# Query ONLY documents in the "billing" category
results = collection.query(
    query_texts=["I want to stop paying"],
    n_results=2,
    where={"category": "billing"}   # Filter: only return billing docs
)
```

In SQL terms:
```sql
SELECT TOP 2 * FROM support_articles
WHERE SIMILARITY_TO('I want to stop paying') > threshold
  AND category = 'billing'
ORDER BY SIMILARITY DESC
```

---

## Part 6: Persistent Storage

By default, ChromaDB stores data in memory (lost when the program ends).
To save data to disk, use PersistentClient:

```python
# In-memory (for testing):
client = chromadb.Client()

# Persistent (saves to disk):
client = chromadb.PersistentClient(path="./my_vector_db")
```

The second call creates a folder called "my_vector_db" in your current directory.
Next time you run the program, data is still there.

---

## Part 7: Collection Management

```python
# List all collections
collections = client.list_collections()
print(collections)  # ['my_docs', 'support_articles']

# Get an existing collection (do not create a new one)
collection = client.get_collection("my_docs")

# Get collection, or create if it does not exist (safe to call repeatedly)
collection = client.get_or_create_collection("my_docs")

# Delete a collection
client.delete_collection("my_docs")

# Count documents in a collection
print(collection.count())  # 4

# Get a specific document by ID
doc = collection.get(ids=["doc1"])
print(doc)
```

---

## Part 8: The Embedding Function

By default, ChromaDB uses its own built-in embedding function
(based on all-MiniLM-L6-v2, which produces 384-dimensional vectors).

You can also use your own:

```python
from chromadb.utils import embedding_functions

# Use sentence-transformers with a specific model
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    "my_docs",
    embedding_function=ef
)
```

You can also provide pre-computed vectors directly (skip the embedding function):

```python
# Provide your own vectors:
collection.add(
    embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # YOUR vectors
    documents=["Document text 1", "Document text 2"],
    ids=["doc1", "doc2"]
)
```

---

## Key Takeaways

1. ChromaDB is like SQLite for vectors -- no server setup, just pip install.

2. Collection = table. Document = row. Vector = generated automatically.

3. collection.add(documents=[...], ids=[...]) stores your text.
   ChromaDB converts it to vectors automatically.

4. collection.query(query_texts=["..."], n_results=5) finds the most similar docs.

5. Results have 'distances' not 'similarities'. Lower distance = more similar.
   Convert: similarity = 1.0 - distance

6. Use metadata + where={...} to filter results (like a SQL WHERE clause).

7. Use PersistentClient to save data to disk between runs.

---

## Next

Lesson 04: Building a Document Search Engine
  - Combining everything into a real search system
  - Loading documents from files
  - Building an interactive query interface

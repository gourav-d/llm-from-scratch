"""
Example 03: ChromaDB Basics
============================

GLOSSARY
--------
ChromaDB:
  An open-source vector database written in Python.
  No server needed -- just pip install chromadb.
  In .NET terms: like SQLite but for semantic (similarity) search.

Client:
  The ChromaDB object you create to access the database.
  chromadb.Client()        -> in-memory (data lost when program ends)
  chromadb.PersistentClient(path="./db") -> saves to disk

Collection:
  A named group of documents in ChromaDB.
  Equivalent to a table in SQL Server.
  Each collection stores: documents, their vectors, and optional metadata.

Add:
  collection.add(documents=[...], ids=[...])
  Stores text into the collection.
  ChromaDB automatically converts text to vectors (embeddings).

Query:
  collection.query(query_texts=["my question"], n_results=3)
  Finds the N most similar documents to the query.
  Returns: ids, documents, distances, metadatas

Distance:
  A number returned by ChromaDB for each result.
  LOWER distance = MORE similar.
  For cosine space: distance = 1 - cosine_similarity
  Distance 0.0 = identical. Distance 2.0 = opposite (maximum).

Metadata:
  Extra key-value data you can attach to each document.
  Example: {"category": "billing", "date": "2024-01-15"}
  Used for filtering results (like a WHERE clause in SQL).

upsert:
  Update a document if its ID already exists, otherwise insert it.
  Like SQL: INSERT OR REPLACE / MERGE
  collection.upsert(documents=[...], ids=[...])

WHAT THIS EXAMPLE SHOWS
------------------------
Part 1: Creating a ChromaDB client and collection
Part 2: Adding documents (with and without metadata)
Part 3: Querying for similar documents
Part 4: Reading and formatting results
Part 5: Filtering with metadata (WHERE equivalent)
Part 6: Updating, deleting, and managing collections

LIBRARIES NEEDED
-----------------
  chromadb   (pip install chromadb)  - the vector database
  numpy      (pip install numpy)     - optional, for vector inspection

Run this AFTER example_01 and example_02.
ChromaDB must be installed: pip install chromadb
"""

print("=" * 65)
print("EXAMPLE 03: ChromaDB Basics")
print("=" * 65)

# Check if chromadb is installed
try:
    import chromadb                          # Import the ChromaDB library
    from chromadb import Settings            # For configuring the client
    print("  chromadb imported successfully")
    print()
except ImportError:
    print("  ERROR: chromadb is not installed.")
    print("  Run: pip install chromadb")
    print("  Then re-run this script.")
    raise SystemExit(1)                      # Stop here if ChromaDB is missing


# ==============================================================================
# PART 1: Creating a Client and Collection
# ==============================================================================

print("=" * 65)
print("PART 1: Creating a Client and Collection")
print("=" * 65)

print("""
Step 1: Create a ChromaDB CLIENT
  The client is the entry point -- like a database connection in C#.
  (Like SqlConnection in ADO.NET)

  Two options:
  (A) chromadb.Client()                      -> in-memory, fast, data lost on exit
  (B) chromadb.PersistentClient(path="./db") -> saves to disk, data survives

  We use in-memory for this example so you do not need to clean up files.
""")

# Create an in-memory client (no files created)
client = chromadb.Client()                   # In-memory -- fast for testing
print("  Created in-memory ChromaDB client")

print("""
Step 2: Create a COLLECTION
  A collection is like a SQL table.
  Every document you add goes into a collection.
  You can have many collections for different purposes.

  client.create_collection("name")  -> creates new (fails if already exists)
  client.get_collection("name")     -> gets existing (fails if does not exist)
  client.get_or_create_collection("name") -> safest: creates OR gets
""")

# Create a collection named "support_articles"
# This is like: CREATE TABLE support_articles (...)
collection = client.get_or_create_collection(
    name="support_articles"                  # Name of the collection
)
print(f"  Created collection: 'support_articles'")
print(f"  Current document count: {collection.count()}")
print()


# ==============================================================================
# PART 2: Adding Documents
# ==============================================================================

print("=" * 65)
print("PART 2: Adding Documents")
print("=" * 65)

print("""
Adding documents requires:
  - documents: list of text strings (the content to store and search)
  - ids:       list of unique IDs (like a primary key in SQL)
  - metadatas: list of dicts with extra info (optional, for filtering)

ChromaDB automatically converts your documents to vectors using
its built-in embedding model (all-MiniLM-L6-v2 by default).
""")

# The documents we want to store
# Imagine these are support articles from a help desk system
documents = [
    # Account-related articles
    "How to reset your forgotten password using the password recovery page",
    "How to change your email address and verify the new email",
    "How to enable two-factor authentication for extra account security",

    # Billing-related articles
    "How to update your credit card and billing information",
    "Understanding your monthly bill and subscription charges",
    "How to cancel your subscription and get a refund",

    # Technical support articles
    "How to install the desktop application on Windows",
    "How to clear the application cache and fix loading errors",
    "System requirements and supported operating systems",
]

# Unique IDs for each document (like a primary key)
ids = [
    "acct_001", "acct_002", "acct_003",    # Account articles
    "bill_001", "bill_002", "bill_003",    # Billing articles
    "tech_001", "tech_002", "tech_003",    # Technical articles
]

# Metadata for each document (for filtering later)
metadatas = [
    {"category": "account", "priority": "high"},
    {"category": "account", "priority": "low"},
    {"category": "account", "priority": "medium"},

    {"category": "billing", "priority": "high"},
    {"category": "billing", "priority": "low"},
    {"category": "billing", "priority": "high"},

    {"category": "technical", "priority": "medium"},
    {"category": "technical", "priority": "high"},
    {"category": "technical", "priority": "low"},
]

print(f"Adding {len(documents)} documents...")

# Add all documents in one call (batch -- more efficient than one at a time)
collection.add(
    documents=documents,    # The text content
    ids=ids,                # Unique IDs
    metadatas=metadatas     # Extra info (optional)
)

print(f"  Documents added. Total in collection: {collection.count()}")
print()
print("  Sample documents added:")
for i in range(3):
    print(f"    [{ids[i]}] {documents[i][:60]}...")
print()


# ==============================================================================
# PART 3: Querying for Similar Documents
# ==============================================================================

print("=" * 65)
print("PART 3: Querying for Similar Documents")
print("=" * 65)

print("""
collection.query(query_texts=["your question"], n_results=N)

ChromaDB automatically:
  1. Converts your query text to a vector (using the same embedding model)
  2. Computes similarity between query vector and all stored document vectors
  3. Returns the N most similar documents

No manual vector math needed!
""")

# First query: account-related
query1 = "I forgot my password and cannot log in"
results1 = collection.query(
    query_texts=[query1],    # The search query (ChromaDB auto-embeds it)
    n_results=3              # Return top 3 most similar results
)

print(f"Query: '{query1}'")
print(f"Top {len(results1['documents'][0])} results:")
print()

# results1 is a dict with lists-of-lists (one inner list per query)
# Since we sent 1 query, we access [0] to get the first (and only) query's results
for i in range(len(results1["documents"][0])):
    doc_id    = results1["ids"][0][i]             # The document's ID
    doc_text  = results1["documents"][0][i]       # The document's text
    distance  = results1["distances"][0][i]       # How different (lower = more similar)
    similarity = 1.0 - distance                   # Convert to similarity (higher = more similar)
    metadata  = results1["metadatas"][0][i]       # The metadata we stored

    print(f"  Rank {i+1}:")
    print(f"    ID:         {doc_id}")
    print(f"    Similarity: {similarity:.4f}  (distance: {distance:.4f})")
    print(f"    Category:   {metadata.get('category', '?')}")
    print(f"    Text:       {doc_text}")
    print()


# Second query: billing-related
print("-" * 40)
query2 = "I want to stop paying for my subscription"
results2 = collection.query(
    query_texts=[query2],
    n_results=3
)

print(f"Query: '{query2}'")
print(f"Top 3 results:")
print()

for i in range(len(results2["documents"][0])):
    doc_id    = results2["ids"][0][i]
    doc_text  = results2["documents"][0][i]
    distance  = results2["distances"][0][i]
    similarity = 1.0 - distance
    metadata  = results2["metadatas"][0][i]

    print(f"  Rank {i+1}: [{doc_id}] similarity={similarity:.4f}")
    print(f"    {doc_text}")
    print()


# ==============================================================================
# PART 4: Reading Results in Different Ways
# ==============================================================================

print("=" * 65)
print("PART 4: Reading and Formatting Results")
print("=" * 65)

print("""
The raw results dict can be confusing. Here is a helper function
that formats results into a readable list.

Raw results structure (confusing but important to understand):
  results = {
    'ids':       [['id1', 'id2', ...]],       <- list of lists (one per query)
    'documents': [['text1', 'text2', ...]],   <- one per query
    'distances': [[0.12, 0.34, ...]],         <- one per query
    'metadatas': [[{'key':'val'}, ...]],      <- one per query
  }
  Access the first query's results: results['documents'][0]
""")

def format_results(query_text, raw_results):
    """
    Format ChromaDB query results into a readable list.
    query_text:  The original query string
    raw_results: The dict returned by collection.query()
    Returns:     list of dicts with id, text, similarity, metadata
    """
    formatted = []                             # Will hold formatted result dicts

    # All results are in index [0] because we sent one query
    doc_ids    = raw_results["ids"][0]         # List of IDs
    doc_texts  = raw_results["documents"][0]   # List of texts
    distances  = raw_results["distances"][0]   # List of distances
    metadatas  = raw_results["metadatas"][0]   # List of metadata dicts

    for doc_id, text, dist, meta in zip(doc_ids, doc_texts, distances, metadatas):
        similarity = round(1.0 - dist, 4)     # Convert distance to similarity, round to 4 places
        formatted.append({
            "id":         doc_id,
            "text":       text,
            "similarity": similarity,
            "metadata":   meta
        })

    return formatted

# Test the helper function
query3 = "my app keeps crashing and is slow"
raw3   = collection.query(query_texts=[query3], n_results=3)
nice3  = format_results(query3, raw3)

print(f"Query: '{query3}'")
print(f"Formatted results:")
for i, result in enumerate(nice3, start=1):
    print(f"  {i}. [{result['id']}] similarity={result['similarity']}")
    print(f"     Category: {result['metadata']['category']}")
    print(f"     Text: {result['text']}")
    print()


# ==============================================================================
# PART 5: Filtering with Metadata (WHERE clause equivalent)
# ==============================================================================

print("=" * 65)
print("PART 5: Filtering with Metadata")
print("=" * 65)

print("""
You can filter results using metadata -- like a SQL WHERE clause.

  SQL:     SELECT TOP 3 ... WHERE category = 'billing'
  ChromaDB: collection.query(..., where={"category": "billing"})

Supported operators:
  where={"key": "value"}           -> exact match
  where={"key": {"$eq": "value"}}  -> explicit equals
  where={"key": {"$ne": "value"}}  -> not equals
  where={"key": {"$in": ["a","b"]}}-> value is in list

Multiple conditions (AND):
  where={"$and": [{"cat": "billing"}, {"priority": "high"}]}
""")

# Query: find similar docs BUT ONLY in the "billing" category
query4 = "I need help with my payment"
raw4_all      = collection.query(query_texts=[query4], n_results=5)
raw4_billing  = collection.query(
    query_texts=[query4],
    n_results=3,
    where={"category": "billing"}    # Only return billing docs
)

print(f"Query: '{query4}'")
print()

print("  WITHOUT filter (all categories):")
for doc_id, text, dist in zip(
    raw4_all["ids"][0],
    raw4_all["documents"][0],
    raw4_all["distances"][0]
):
    meta = raw4_all["metadatas"][0]
    category = next((m["category"] for m in meta if True), "?")  # Get category
    similarity = 1.0 - dist
    # Find the matching metadata entry by index
    idx = raw4_all["ids"][0].index(doc_id)
    cat = raw4_all["metadatas"][0][idx]["category"]
    print(f"    [{doc_id}] sim={similarity:.4f} category={cat:10s} {text[:45]}...")

print()
print("  WITH filter (billing only):")
for i in range(len(raw4_billing["documents"][0])):
    doc_id    = raw4_billing["ids"][0][i]
    text      = raw4_billing["documents"][0][i]
    dist      = raw4_billing["distances"][0][i]
    similarity = 1.0 - dist
    category  = raw4_billing["metadatas"][0][i]["category"]
    print(f"    [{doc_id}] sim={similarity:.4f} category={category:10s} {text[:45]}...")

print()
print("  -> Filter ensures only billing documents are returned")
print("  -> Similar to: SELECT TOP 3 ... WHERE category = 'billing' ORDER BY similarity")
print()


# ==============================================================================
# PART 6: Updating, Deleting, and Managing Collections
# ==============================================================================

print("=" * 65)
print("PART 6: Collection Management")
print("=" * 65)

print("""
Other useful ChromaDB operations:

  collection.count()           -> How many documents are stored
  collection.get(ids=["id1"])  -> Get a specific document by ID
  collection.upsert(...)       -> Insert or update (like SQL MERGE)
  collection.delete(ids=["x"]) -> Delete specific documents by ID
  client.list_collections()    -> List all collection names
  client.delete_collection("name") -> Delete entire collection
""")

# Get document by ID
specific_doc = collection.get(ids=["bill_003"])    # Get billing article 3
print("Fetching a specific document by ID:")
print(f"  ID:   {specific_doc['ids'][0]}")
print(f"  Text: {specific_doc['documents'][0]}")
print(f"  Meta: {specific_doc['metadatas'][0]}")
print()

# Upsert: update an existing document or insert if new
print("Upsert (update or insert):")
collection.upsert(
    documents=["How to reset your password using email verification link"],  # Updated text
    ids=["acct_001"],                                 # Same ID as existing document
    metadatas=[{"category": "account", "priority": "critical"}]   # Updated metadata
)
updated = collection.get(ids=["acct_001"])
print(f"  Updated acct_001 text:     {updated['documents'][0]}")
print(f"  Updated acct_001 priority: {updated['metadatas'][0]['priority']}")
print()

# Delete a document
print(f"Count before delete: {collection.count()}")
collection.delete(ids=["tech_003"])               # Delete one document
print(f"Count after delete:  {collection.count()}")
print()

# List all collections
all_collections = client.list_collections()
print(f"All collections: {[c.name for c in all_collections]}")

# Create a second collection
second_collection = client.get_or_create_collection("product_catalog")
second_collection.add(
    documents=["Laptop Pro 15 with 16GB RAM", "Wireless Mouse with USB receiver"],
    ids=["prod_001", "prod_002"]
)
all_collections = client.list_collections()
print(f"After creating second: {[c.name for c in all_collections]}")
print()


# ==============================================================================
# SUMMARY
# ==============================================================================

print("=" * 65)
print("SUMMARY - ChromaDB Basics")
print("=" * 65)

print("""
WHAT WE LEARNED:

1. Create a client:
     client = chromadb.Client()                      <- in-memory
     client = chromadb.PersistentClient(path="./db") <- saves to disk

2. Create a collection (like a SQL table):
     collection = client.get_or_create_collection("name")

3. Add documents:
     collection.add(documents=["text1", "text2"],
                    ids=["id1", "id2"],
                    metadatas=[{"key": "val"}, ...])
   ChromaDB auto-generates the vectors (embeddings).

4. Query for similar documents:
     results = collection.query(query_texts=["my question"], n_results=5)
   Returns: dict with 'ids', 'documents', 'distances', 'metadatas'

5. Read results:
     results['documents'][0]   <- list of matched texts (for first query)
     results['distances'][0]   <- list of distances (lower = more similar)
     similarity = 1.0 - distance

6. Filter with metadata:
     collection.query(..., where={"category": "billing"})
   Like a SQL WHERE clause.

7. Management:
     collection.count()          <- number of documents
     collection.get(ids=[...])   <- get by ID
     collection.upsert(...)      <- insert or update
     collection.delete(ids=[...]) <- delete by ID

NEXT EXAMPLE (04):
  Build a complete document search system that:
  - Loads documents from a file
  - Stores them in ChromaDB with metadata
  - Provides an interactive search interface
""")

print("=" * 65)
print("END OF EXAMPLE 03")
print("=" * 65)

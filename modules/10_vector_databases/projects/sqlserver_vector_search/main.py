"""
Project 2: SQL Server as a Vector Database
===========================================

DESCRIPTION:
  A real-world semantic search application built on top of an existing
  SQL Server database.  No new database required -- your existing SQL
  Server can store and search vectors natively (SQL Server 2025 /
  Azure SQL) or via a JSON fallback (SQL Server 2019 / 2022).

SCENARIO:
  You are a .NET developer at a tech company.  You have an existing
  Products table in SQL Server.  The business wants customers to search
  by meaning ("comfortable shoes for rainy weather") instead of keywords.

  Without moving data to ChromaDB or any other system, you can:
    1. Add a VECTOR column to your existing Products table
    2. Generate embeddings and store them back into SQL Server
    3. Use VECTOR_DISTANCE() in SQL to find similar products
    4. Build a RAG prompt for an LLM to answer questions about products

WHY SQL SERVER FOR VECTORS?
  - Your data is already there -- no ETL pipeline needed
  - Row-level security, transactions, and backups work as usual
  - .NET developers already know T-SQL and pyodbc / SqlClient
  - SQL Server 2025 / Azure SQL: native VECTOR type (fast, indexed)
  - SQL Server 2019 / 2022 fallback: store as JSON, compute in Python

ARCHITECTURE:

  [Products Table]        ->  Embedding Pipeline  ->  VECTOR column added
  (existing SQL Server)       (generate + store)       (same table, new col)

  [User Query]            ->  Embed Query         ->  VECTOR_DISTANCE() SQL
  (plain text search)         (same method)            (TOP 5 by distance)

  [Search Results]        ->  RAG Prompt Builder  ->  LLM Answer
  (matching products)         (context + question)     (simulated here)

WHAT YOU WILL LEARN:
  1. SQL Server 2025 VECTOR data type and VECTOR_DISTANCE() function
  2. How to add vector search to an EXISTING table without migration
  3. Embedding generation from scratch (TF-IDF, no API key needed)
  4. Hybrid search: keyword (LIKE) + vector similarity combined
  5. Fallback strategy for SQL Server 2019 / 2022 (JSON storage)

C# / .NET ANALOGY:
  - VECTOR column = just another column type, like NVARCHAR or FLOAT
  - VECTOR_DISTANCE() = like a WHERE clause for meaning
  - This code does what SqlCommand / Entity Framework would do in C#
  - pyodbc = Python equivalent of SqlConnection in .NET

GLOSSARY (every term defined before first use)
-----------------------------------------------
  VECTOR:          a list of numbers that represents text as a point
                   in high-dimensional space.  Similar texts have
                   similar vectors (they are close together).
  EMBEDDING:       the process of converting text into a vector.
                   We use TF-IDF here (no API key needed).
  TF-IDF:          Term Frequency - Inverse Document Frequency.
                   Measures how important a word is to a document
                   relative to all documents.  Cheap and fast.
  COSINE DISTANCE: how different two vectors are.  0 = identical,
                   1 = completely different.  Lower is better for search.
  COSINE SIMILARITY: 1 - cosine distance.  Higher is better (1 = identical).
  VECTOR_DISTANCE: SQL Server function.  Computes distance between two
                   VECTOR values stored in the database.
  HYBRID SEARCH:   combine keyword search (LIKE / CONTAINS) with
                   vector similarity.  Catches both exact matches and
                   semantic matches.
  RAG:             Retrieve-Augment-Generate.  Find relevant rows from
                   the database, build an LLM prompt with them as context,
                   get a grounded answer from the LLM.
  pyodbc:          Python library to connect to SQL Server (like SqlConnection
                   in C#).  pip install pyodbc
  ODBC Driver:     the Windows driver that pyodbc uses to talk to SQL Server.
                   Install "ODBC Driver 17 for SQL Server" or newer.

LIBRARIES:
  numpy   (pip install numpy)   -- vector math
  pyodbc  (pip install pyodbc)  -- SQL Server connection (Part B only)

HOW TO RUN:
  Part A (in-memory demo, no SQL Server needed):
    python main.py

  Part B (real SQL Server):
    1. Edit CONNECTION_STRING below with your server and database details
    2. Make sure ODBC Driver 17 (or 18) for SQL Server is installed
    3. python main.py
    4. The script auto-detects if SQL Server is reachable

SQL SERVER VERSION SUPPORT:
  - Azure SQL Database:      native VECTOR type (available now, 2025)
  - SQL Server 2025 preview: native VECTOR type
  - SQL Server 2019 / 2022:  JSON fallback (embeddings stored as NVARCHAR)
"""

import json       # For converting Python lists to JSON strings and back
import math       # For log() used in IDF calculation
import time       # For measuring query time
import numpy as np  # For vector math (dot product, norm, etc.)

print("=" * 70)
print("PROJECT 2: SQL Server as a Vector Database")
print("Semantic product search on your existing SQL Server data")
print("=" * 70)


# ==============================================================================
# SAMPLE DATA -- simulates your existing SQL Server Products table
# ==============================================================================
# In production: this data already exists in your SQL Server.
# We use this same data in Part A (in-memory) and insert it in Part B (SQL Server).

# C# analogy: List<Product> where Product is a model class
PRODUCTS = [
    {
        "id": 1, "name": "Trail Runner Pro 2024",
        "category": "Footwear", "price": 129.99,
        "description": (
            "Lightweight trail running shoes with waterproof membrane. "
            "Excellent grip on wet and muddy terrain. Cushioned midsole "
            "for all-day comfort on long runs and hikes. Breathable mesh "
            "upper keeps feet dry in rain. Ideal for outdoor athletes."
        )
    },
    {
        "id": 2, "name": "ErgoDesk Standing Converter",
        "category": "Office", "price": 249.00,
        "description": (
            "Height-adjustable standing desk converter for any table. "
            "Switches from sitting to standing in 3 seconds. Supports "
            "two monitors plus a laptop. Anti-fatigue mat included. "
            "Reduces back pain from long hours of sitting at a desk."
        )
    },
    {
        "id": 3, "name": "NoiseBlock Pro Headphones",
        "category": "Audio", "price": 299.00,
        "description": (
            "Over-ear headphones with active noise cancellation. "
            "40-hour battery life. Crystal clear audio for music and "
            "calls. Foldable design for travel. Works in loud offices, "
            "planes, and coffee shops. Ideal for remote workers and "
            "frequent flyers who need focus and quiet."
        )
    },
    {
        "id": 4, "name": "ThermoMug X500",
        "category": "Kitchen", "price": 34.99,
        "description": (
            "Vacuum-insulated stainless steel travel mug. Keeps coffee "
            "or tea hot for 12 hours, cold drinks cold for 24 hours. "
            "Leak-proof lid, fits in car cup holders. BPA-free. "
            "Perfect for commuters, hikers, and office workers who "
            "want their drink at the right temperature all day."
        )
    },
    {
        "id": 5, "name": "DeveloperPad Mechanical Keyboard",
        "category": "Electronics", "price": 159.00,
        "description": (
            "Compact tenkeyless mechanical keyboard for programmers. "
            "Cherry MX Blue switches for tactile feedback. RGB backlight "
            "with 15 colour modes. USB-C connection, compatible with "
            "Windows, macOS, Linux. N-key rollover for gaming. "
            "Built to last with aircraft-grade aluminium body."
        )
    },
    {
        "id": 6, "name": "UltraSlim Laptop Stand",
        "category": "Office", "price": 49.99,
        "description": (
            "Portable aluminium laptop stand, adjustable angle 15-45 "
            "degrees. Improves posture and reduces neck strain. Folds "
            "flat and fits in any bag. Heat dissipation vents keep "
            "laptop cool. Compatible with all 11-17 inch laptops. "
            "Great for home office and working from cafes."
        )
    },
    {
        "id": 7, "name": "HikePack 40L Backpack",
        "category": "Outdoor", "price": 89.99,
        "description": (
            "40-litre hiking backpack with waterproof rain cover. "
            "Padded hip belt and shoulder straps for heavy loads. "
            "Multiple compartments including hydration sleeve. "
            "Chest strap and load lifters for stability on trails. "
            "Lightweight at 1.2 kg. Suitable for multi-day treks."
        )
    },
    {
        "id": 8, "name": "SleepWell Blackout Curtains",
        "category": "Home", "price": 59.99,
        "description": (
            "Thermal blackout curtains that block 99 percent of light. "
            "Reduces outside noise for better sleep. Energy-efficient: "
            "keeps room cool in summer and warm in winter. Machine "
            "washable. Comes with curtain hooks. Available in 8 colours. "
            "Recommended by sleep specialists for shift workers and "
            "light-sensitive sleepers."
        )
    },
    {
        "id": 9, "name": "FocusFlow Desk Lamp",
        "category": "Office", "price": 79.99,
        "description": (
            "LED desk lamp with 5 colour temperatures and 10 brightness "
            "levels. USB-A charging port for phone on the base. "
            "Auto-timer turns off after 1 hour. Flexible gooseneck arm. "
            "Reduces eye strain during long work or study sessions. "
            "Memory function remembers your last setting."
        )
    },
    {
        "id": 10, "name": "SpeedBlend Pro Blender",
        "category": "Kitchen", "price": 119.00,
        "description": (
            "High-powered personal blender with 1200W motor. Makes "
            "smoothies, protein shakes, and nut butters in 60 seconds. "
            "BPA-free travel cup with lid included. Self-cleaning: "
            "add water and blend for 30 seconds. Dishwasher safe parts. "
            "Quiet motor technology, 50 percent quieter than standard "
            "blenders. Great for busy mornings."
        )
    },
    {
        "id": 11, "name": "CodeBook Pro Laptop",
        "category": "Electronics", "price": 1299.00,
        "description": (
            "Developer-focused laptop with 12-core processor and 32GB RAM. "
            "14-inch 2K display with 120Hz refresh rate. 1TB NVMe SSD, "
            "boots in 5 seconds. Dual Thunderbolt 4 ports and full-size "
            "HDMI. 16-hour battery. Pre-installed with WSL2 for Linux "
            "development. Ideal for software engineers, data scientists, "
            "and DevOps engineers."
        )
    },
    {
        "id": 12, "name": "ComfortSeat Lumbar Cushion",
        "category": "Office", "price": 39.99,
        "description": (
            "Memory foam lumbar support cushion for office chairs. "
            "Reduces lower back pain during long sitting sessions. "
            "Adjustable strap fits any chair. Breathable mesh cover "
            "is machine washable. Ergonomically shaped to fit the "
            "natural curve of the spine. Recommended by physiotherapists."
        )
    },
]


# ==============================================================================
# PART A: IN-MEMORY SIMULATION (no SQL Server needed)
# ==============================================================================
# Learn the concept without any database setup.
# Part B shows the exact same logic running against real SQL Server.

print("\n" + "=" * 70)
print("PART A: In-Memory Simulation (no SQL Server needed)")
print("Concept: how vector search works before we move it into SQL Server")
print("=" * 70)


# ------------------------------------------------------------------------------
# STEP A1: Build TF-IDF embeddings from scratch
# ------------------------------------------------------------------------------
# TF-IDF = Term Frequency * Inverse Document Frequency
#
# TF  = (count of word in document) / (total words in document)
#       -- how often the word appears in THIS document
#
# IDF = log(total documents / documents containing the word)
#       -- how rare the word is across ALL documents
#       -- common words like "the" get low IDF (not informative)
#       -- rare words like "waterproof" get high IDF (very informative)
#
# TF-IDF score = TF * IDF
#       -- high score = word is frequent in this doc AND rare across all docs
#       -- exactly the words that make this document unique
#
# C# analogy: imagine scoring each word in each document and putting
#             those scores into a float[] array -- that is the embedding.

print("\n--- STEP A1: Build TF-IDF Embeddings ---")


def tokenize(text):
    """
    Convert text to a list of lowercase words.
    Removes punctuation by keeping only letters and spaces.
    C# analogy: text.ToLower().Split(' ').Where(w => w.Length > 0)
    """
    # Convert to lowercase so "Shoes" and "shoes" are the same word
    text = text.lower()
    # Replace non-letter characters with spaces
    cleaned = ""
    for ch in text:
        if ch.isalpha() or ch == " ":
            cleaned += ch
        else:
            cleaned += " "
    # Split into words, filter out empty strings
    words = [w for w in cleaned.split(" ") if len(w) > 1]
    return words


# Stop words = common words that carry no useful meaning.
# We remove them so they do not pollute our embeddings.
# C# analogy: a HashSet<string> of words to ignore
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "it", "its", "this", "that", "these",
    "those", "they", "them", "their", "all", "any", "both", "each",
    "as", "up", "out", "if", "into", "during", "including", "after",
    "your", "you", "our", "we", "i", "my", "his", "her"
}


def remove_stop_words(tokens):
    """Filter out stop words from a token list."""
    return [t for t in tokens if t not in STOP_WORDS]


def build_tfidf_embeddings(documents, field="description"):
    """
    Build TF-IDF embeddings for all documents.

    documents: list of dicts, each with at least the field given
    field:     which text field to embed
    Returns:
      vocabulary: dict mapping word -> index in the vector
      embeddings: list of numpy arrays, one per document
    """
    # ----- PASS 1: build vocabulary and document frequency -----
    # document_freq[word] = number of documents that contain this word
    document_freq = {}   # C# analogy: Dictionary<string, int>

    all_tokens_per_doc = []  # Store tokens for each document (reuse in pass 2)

    for doc in documents:
        tokens = remove_stop_words(tokenize(doc[field]))
        all_tokens_per_doc.append(tokens)
        unique_in_doc = set(tokens)          # Each word counted once per document
        for word in unique_in_doc:
            document_freq[word] = document_freq.get(word, 0) + 1

    # Build vocabulary: sort words so the vector index is consistent
    # C# analogy: var vocab = document_freq.Keys.OrderBy(w => w).ToList()
    vocabulary = {word: idx for idx, word in enumerate(sorted(document_freq.keys()))}
    vocab_size  = len(vocabulary)
    n_docs      = len(documents)

    # ----- PASS 2: compute TF-IDF vectors -----
    embeddings = []

    for tokens in all_tokens_per_doc:
        # Count word frequencies in this document
        term_freq = {}   # C# analogy: Dictionary<string, int>
        for word in tokens:
            term_freq[word] = term_freq.get(word, 0) + 1

        total_terms = max(len(tokens), 1)   # Avoid division by zero

        # Build the TF-IDF vector (one float per vocabulary word)
        vector = np.zeros(vocab_size, dtype=np.float32)  # Start with all zeros

        for word, count in term_freq.items():
            if word not in vocabulary:
                continue
            idx = vocabulary[word]
            # TF = word count in this doc / total words in this doc
            tf  = count / total_terms
            # IDF = log(total docs / docs containing this word)
            idf = math.log(n_docs / document_freq[word])
            vector[idx] = tf * idf   # TF-IDF score for this word

        # Normalize to unit length so cosine similarity = dot product
        # C# analogy: divide every element by the vector's magnitude
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        embeddings.append(vector)

    return vocabulary, embeddings


# Build embeddings for all 12 products
vocabulary, product_embeddings = build_tfidf_embeddings(PRODUCTS)

print(f"  Vocabulary size: {len(vocabulary)} unique words")
print(f"  Embedding dimension: {len(vocabulary)} (one float per word)")
print(f"  Products embedded: {len(product_embeddings)}")
print()
print("  C# analogy: each product description became a float[] array.")
print("  Similar products have vectors that point in similar directions.")


# ------------------------------------------------------------------------------
# STEP A2: Semantic Search (cosine similarity)
# ------------------------------------------------------------------------------
# Cosine similarity = dot product of two UNIT vectors.
# Since we normalized everything, this is fast and simple.

print("\n--- STEP A2: Semantic Search ---")


def embed_query(query_text, vocabulary):
    """
    Convert a search query into the same TF-IDF vector space as the products.
    Same logic as build_tfidf_embeddings for a single text.
    """
    tokens     = remove_stop_words(tokenize(query_text))
    vocab_size = len(vocabulary)
    vector     = np.zeros(vocab_size, dtype=np.float32)
    term_freq  = {}

    for word in tokens:
        term_freq[word] = term_freq.get(word, 0) + 1

    total_terms = max(len(tokens), 1)

    for word, count in term_freq.items():
        if word in vocabulary:
            idx        = vocabulary[word]
            tf         = count / total_terms
            # For a single query we cannot compute IDF, so we just use TF.
            # In production: use the same IDF values from the training corpus.
            vector[idx] = tf

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector


def semantic_search(query_text, products, product_embeddings, vocabulary, top_k=5):
    """
    Find products whose descriptions are most similar to the query.
    Returns the top_k most similar products with their similarity scores.

    C# analogy: like a LINQ OrderByDescending on a computed similarity field.
    """
    query_vector = embed_query(query_text, vocabulary)

    scores = []
    for i, prod_vector in enumerate(product_embeddings):
        # Cosine similarity = dot product of two unit vectors
        # np.dot is like Vector.Dot() in C# System.Numerics
        similarity = float(np.dot(query_vector, prod_vector))
        scores.append((i, similarity))

    # Sort by similarity descending (highest score first)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Return top_k results with product data and scores
    results = []
    for idx, score in scores[:top_k]:
        product = PRODUCTS[idx]
        results.append({
            "id":         product["id"],
            "name":       product["name"],
            "category":   product["category"],
            "price":      product["price"],
            "similarity": round(score, 4),
            "description": product["description"][:80] + "..."
        })

    return results


def display_search_results(query, results):
    """Print results in a readable table format."""
    print(f"\n  Query: \"{query}\"")
    print(f"  {'Rank':<5} {'Name':<35} {'Category':<12} {'Price':>8}  {'Match'}")
    print(f"  {'-'*5} {'-'*35} {'-'*12} {'-'*8}  {'-'*20}")
    for rank, r in enumerate(results, start=1):
        bar_len = int(r["similarity"] * 20)
        bar     = "#" * bar_len
        print(f"  {rank:<5} {r['name']:<35} {r['category']:<12} ${r['price']:>7.2f}  {bar}")


# ---- Run demo searches ----
DEMO_QUERIES = [
    "comfortable shoes for walking in the rain",
    "I need help with my back pain while working at desk",
    "quiet headphones for concentrating at work",
    "keep my drink warm during outdoor activities",
    "laptop for software developers and programmers",
]

for query in DEMO_QUERIES:
    results = semantic_search(query, PRODUCTS, product_embeddings, vocabulary, top_k=3)
    display_search_results(query, results)

print()
print("  Note: TF-IDF is a fast approximation.")
print("  Sentence-Transformers / OpenAI embeddings give better results")
print("  but require an API key or GPU.  TF-IDF needs no external calls.")


# ------------------------------------------------------------------------------
# STEP A3: Hybrid Search (keyword + vector)
# ------------------------------------------------------------------------------
# Pure vector search can miss exact product names.
# Pure keyword search misses synonyms.
# Hybrid = combine both scores for best results.
#
# Formula: hybrid_score = alpha * keyword_score + (1 - alpha) * vector_score
#   alpha = 0.0  -> pure vector   (meaning-based)
#   alpha = 1.0  -> pure keyword  (exact match)
#   alpha = 0.5  -> balanced blend (recommended default)

print("\n--- STEP A3: Hybrid Search (keyword + vector combined) ---")


def keyword_score(query_text, product):
    """
    Simple keyword overlap score between query and product name/description.
    Returns a float 0.0 to 1.0.
    C# analogy: product.Description.Split(' ').Intersect(queryWords).Count()
    """
    query_tokens   = set(remove_stop_words(tokenize(query_text)))
    product_tokens = set(remove_stop_words(tokenize(
        product["name"] + " " + product["description"]
    )))
    if len(query_tokens) == 0:
        return 0.0
    overlap = len(query_tokens & product_tokens)   # Set intersection
    return overlap / len(query_tokens)             # Fraction of query words found


def hybrid_search(query_text, products, product_embeddings, vocabulary,
                  alpha=0.5, top_k=5):
    """
    Combine keyword overlap score and TF-IDF vector similarity.
    alpha: weight for keyword score (0 = pure vector, 1 = pure keyword)
    """
    query_vector = embed_query(query_text, vocabulary)

    results = []
    for i, product in enumerate(products):
        vec_sim  = float(np.dot(query_vector, product_embeddings[i]))  # Vector score
        kw_score = keyword_score(query_text, product)                  # Keyword score
        hybrid   = alpha * kw_score + (1.0 - alpha) * vec_sim         # Blend

        results.append({
            "id":         product["id"],
            "name":       product["name"],
            "category":   product["category"],
            "price":      product["price"],
            "vec_score":  round(vec_sim,  4),
            "kw_score":   round(kw_score, 4),
            "hybrid":     round(hybrid,   4),
        })

    results.sort(key=lambda x: x["hybrid"], reverse=True)
    return results[:top_k]


hybrid_query = "mechanical keyboard for coding"
hybrid_results = hybrid_search(
    hybrid_query, PRODUCTS, product_embeddings, vocabulary, alpha=0.4, top_k=5
)

print(f"\n  Query: \"{hybrid_query}\"  (alpha=0.4 -> 40% keyword, 60% vector)")
print(f"  {'Name':<35} {'KW':>6}  {'Vec':>6}  {'Hybrid':>8}")
print(f"  {'-'*35} {'-'*6}  {'-'*6}  {'-'*8}")
for r in hybrid_results:
    print(f"  {r['name']:<35} {r['kw_score']:>6.4f}  {r['vec_score']:>6.4f}  {r['hybrid']:>8.4f}")


# ------------------------------------------------------------------------------
# STEP A4: RAG Prompt Builder
# ------------------------------------------------------------------------------
# RAG = Retrieve-Augment-Generate
#   Retrieve: find relevant products from the database
#   Augment:  put those products into the LLM prompt as context
#   Generate: LLM answers the user's question using that context
#
# This grounds the LLM -- it can only answer from your actual product data.

print("\n--- STEP A4: RAG Prompt Builder ---")


def build_product_rag_prompt(user_question, search_results, max_products=3):
    """
    Build an LLM prompt that includes retrieved product data as context.
    The LLM can only answer from what is in the context (grounded response).

    user_question:  what the customer asked
    search_results: output of semantic_search() or hybrid_search()
    max_products:   how many products to include in the context
    Returns: the full prompt string ready to send to GPT-4 / Claude
    """
    context_lines = []
    for i, prod in enumerate(search_results[:max_products], start=1):
        # Find the full product data by ID
        full_product = next(p for p in PRODUCTS if p["id"] == prod["id"])
        context_lines.append(
            f"Product {i}: {full_product['name']}"
        )
        context_lines.append(
            f"  Category: {full_product['category']}  |  Price: ${full_product['price']:.2f}"
        )
        context_lines.append(
            f"  Description: {full_product['description']}"
        )
        context_lines.append("")   # Blank line between products

    context_block = "\n".join(context_lines)

    prompt = f"""You are a helpful product recommendation assistant.
Use ONLY the product information below to answer the customer's question.
If none of the products match, say so honestly.
Always mention the product name and price in your answer.

--- RETRIEVED PRODUCTS ---
{context_block}
--- END PRODUCTS ---

Customer question: {user_question}

Answer:"""

    return prompt


rag_query   = "I sit at a desk all day and my back hurts, what should I buy?"
rag_results = semantic_search(
    rag_query, PRODUCTS, product_embeddings, vocabulary, top_k=3
)
rag_prompt  = build_product_rag_prompt(rag_query, rag_results, max_products=3)

print(f"\n  Customer question: \"{rag_query}\"")
print("\n  TOP MATCHED PRODUCTS:")
for r in rag_results[:3]:
    print(f"    - {r['name']}  (similarity: {r['similarity']:.4f})")

print("\n  GENERATED RAG PROMPT (ready to send to GPT-4 / Claude):")
print("-" * 60)
for line in rag_prompt.split("\n"):
    print("  " + line)
print("-" * 60)
print()
print("  In production: send this prompt to the Claude or OpenAI API.")
print("  The LLM will recommend specific products from your database.")


# ==============================================================================
# PART B: REAL SQL SERVER CONNECTION
# ==============================================================================
# Everything from Part A but running inside a real SQL Server database.
# The same SQL you write here is the T-SQL a .NET developer already knows.

print("\n" + "=" * 70)
print("PART B: Real SQL Server Connection")
print("The same logic but running in your actual database")
print("=" * 70)

# ------------------------------------------------------------------------------
# CONNECTION CONFIGURATION -- edit these for your environment
# ------------------------------------------------------------------------------
# C# analogy:  this is your connection string from appsettings.json

# Option 1: Windows Authentication (most common on corporate networks)
# Replace 'localhost' with your server name (e.g., 'MYSERVER\SQLEXPRESS')
# Replace 'VectorDemo' with your database name

CONNECTION_CONFIGS = {
    # Windows Authentication (no username/password needed)
    "windows_auth": (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=VectorDemo;"
        "Trusted_Connection=yes;"
    ),

    # SQL Server Authentication (username + password)
    # Uncomment and fill in to use
    # "sql_auth": (
    #     "DRIVER={ODBC Driver 17 for SQL Server};"
    #     "SERVER=localhost;"
    #     "DATABASE=VectorDemo;"
    #     "UID=sa;"
    #     "PWD=YourPasswordHere;"
    # ),

    # Azure SQL Database
    # Uncomment and fill in to use
    # "azure_sql": (
    #     "DRIVER={ODBC Driver 18 for SQL Server};"
    #     "SERVER=yourserver.database.windows.net;"
    #     "DATABASE=VectorDemo;"
    #     "Authentication=ActiveDirectoryInteractive;"
    #     "Encrypt=yes;"
    # ),
}

# Choose which config to use
ACTIVE_CONFIG = "windows_auth"
CONNECTION_STRING = CONNECTION_CONFIGS[ACTIVE_CONFIG]

# Database and table names
DB_TABLE = "Products"               # Your existing table name
VECTOR_DIM = len(vocabulary)        # Embedding dimension (vocab size)


# ------------------------------------------------------------------------------
# DETECT SQL SERVER VERSION
# ------------------------------------------------------------------------------
# SQL Server 2025 / Azure SQL supports the native VECTOR type.
# SQL Server 2019 / 2022 needs a JSON fallback.
# We detect the version and choose the right approach automatically.

def detect_sqlserver_version(cursor):
    """
    Query the SQL Server version string and determine if native VECTOR is supported.
    Returns: (version_string, supports_vector: bool)
    """
    cursor.execute("SELECT @@VERSION AS version")
    row = cursor.fetchone()
    version_str = row[0] if row else ""

    # SQL Server 2025 starts with "Microsoft SQL Server 2025" or has version 16.x+
    # Azure SQL Database also supports VECTOR
    supports_vector = (
        "2025" in version_str or
        "Azure SQL" in version_str or
        "Microsoft Azure" in version_str
    )
    return version_str[:80], supports_vector


# ------------------------------------------------------------------------------
# SCHEMA: Create or update the Products table
# ------------------------------------------------------------------------------

CREATE_TABLE_SQL_2025 = f"""
-- SQL Server 2025 / Azure SQL: native VECTOR type
-- C# analogy: [Column] float[] DescriptionVector in Entity Framework
IF OBJECT_ID('{DB_TABLE}', 'U') IS NULL
BEGIN
    CREATE TABLE {DB_TABLE} (
        Id               INT           PRIMARY KEY,
        Name             NVARCHAR(200) NOT NULL,
        Category         NVARCHAR(100) NOT NULL,
        Price            DECIMAL(10,2) NOT NULL,
        Description      NVARCHAR(MAX) NOT NULL,
        DescriptionVector VECTOR({VECTOR_DIM})   -- native vector column (SQL Server 2025)
    );
    PRINT 'Table {DB_TABLE} created with native VECTOR column.';
END
ELSE
BEGIN
    -- Add vector column to existing table if not present
    -- C# analogy: EF migration adding a new column
    IF NOT EXISTS (
        SELECT 1 FROM sys.columns
        WHERE object_id = OBJECT_ID('{DB_TABLE}') AND name = 'DescriptionVector'
    )
    BEGIN
        ALTER TABLE {DB_TABLE} ADD DescriptionVector VECTOR({VECTOR_DIM});
        PRINT 'DescriptionVector column added to existing table.';
    END
END
"""

CREATE_TABLE_SQL_LEGACY = f"""
-- SQL Server 2019 / 2022: JSON fallback (store embeddings as NVARCHAR)
-- C# analogy: [Column] string DescriptionEmbeddingJson in Entity Framework
IF OBJECT_ID('{DB_TABLE}', 'U') IS NULL
BEGIN
    CREATE TABLE {DB_TABLE} (
        Id                       INT           PRIMARY KEY,
        Name                     NVARCHAR(200) NOT NULL,
        Category                 NVARCHAR(100) NOT NULL,
        Price                    DECIMAL(10,2) NOT NULL,
        Description              NVARCHAR(MAX) NOT NULL,
        DescriptionEmbeddingJson NVARCHAR(MAX) NULL   -- JSON array fallback
    );
    PRINT 'Table {DB_TABLE} created with JSON embedding column (legacy SQL Server).';
END
ELSE
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM sys.columns
        WHERE object_id = OBJECT_ID('{DB_TABLE}')
          AND name = 'DescriptionEmbeddingJson'
    )
    BEGIN
        ALTER TABLE {DB_TABLE} ADD DescriptionEmbeddingJson NVARCHAR(MAX) NULL;
        PRINT 'DescriptionEmbeddingJson column added to existing table.';
    END
END
"""


# ------------------------------------------------------------------------------
# INSERT: Store product data and embeddings
# ------------------------------------------------------------------------------

def insert_products_2025(cursor, products, embeddings):
    """
    Insert products and their VECTOR embeddings into SQL Server 2025 / Azure SQL.
    Uses MERGE (SQL Server equivalent of UPSERT) so re-running is safe.

    C# analogy: using SqlCommand with parameters to prevent SQL injection
    """
    merge_sql = f"""
    MERGE {DB_TABLE} AS target
    USING (VALUES (?, ?, ?, ?, ?)) AS source(Id, Name, Category, Price, Description)
    ON target.Id = source.Id
    WHEN MATCHED THEN
        UPDATE SET Name=source.Name, Category=source.Category,
                   Price=source.Price, Description=source.Description
    WHEN NOT MATCHED THEN
        INSERT (Id, Name, Category, Price, Description)
        VALUES (source.Id, source.Name, source.Category, source.Price, source.Description);
    """

    update_vector_sql = f"""
    UPDATE {DB_TABLE}
    SET DescriptionVector = CAST(? AS VECTOR({VECTOR_DIM}))
    WHERE Id = ?
    """

    for product, vector in zip(products, embeddings):
        # MERGE: insert or update the product row
        cursor.execute(merge_sql, (
            product["id"], product["name"], product["category"],
            product["price"], product["description"]
        ))

        # Store the vector as a JSON string cast to VECTOR type
        # SQL Server parses "[0.1, 0.2, ...]" into a VECTOR value
        vector_json = json.dumps(vector.tolist())
        cursor.execute(update_vector_sql, (vector_json, product["id"]))


def insert_products_legacy(cursor, products, embeddings):
    """
    Insert products and store embeddings as JSON strings (legacy SQL Server).
    """
    merge_sql = f"""
    MERGE {DB_TABLE} AS target
    USING (VALUES (?, ?, ?, ?, ?)) AS source(Id, Name, Category, Price, Description)
    ON target.Id = source.Id
    WHEN MATCHED THEN
        UPDATE SET Name=source.Name, Category=source.Category,
                   Price=source.Price, Description=source.Description
    WHEN NOT MATCHED THEN
        INSERT (Id, Name, Category, Price, Description)
        VALUES (source.Id, source.Name, source.Category, source.Price, source.Description);
    """

    update_json_sql = f"""
    UPDATE {DB_TABLE} SET DescriptionEmbeddingJson = ? WHERE Id = ?
    """

    for product, vector in zip(products, embeddings):
        cursor.execute(merge_sql, (
            product["id"], product["name"], product["category"],
            product["price"], product["description"]
        ))
        vector_json = json.dumps(vector.tolist())   # Convert float[] to JSON string
        cursor.execute(update_json_sql, (vector_json, product["id"]))


# ------------------------------------------------------------------------------
# SEARCH: Run vector search inside SQL Server
# ------------------------------------------------------------------------------

def sqlserver_semantic_search_2025(cursor, query_text, vocabulary, top_k=5):
    """
    Use SQL Server 2025 native VECTOR_DISTANCE() to find similar products.

    The entire similarity computation happens INSIDE SQL Server.
    No data is loaded into Python -- SQL does the heavy lifting.

    C# analogy:
      var results = context.Products
          .OrderBy(p => VectorDistance("cosine", p.DescriptionVector, queryVector))
          .Take(topK)
          .ToList();
    """
    query_vector      = embed_query(query_text, vocabulary)
    query_vector_json = json.dumps(query_vector.tolist())

    # VECTOR_DISTANCE returns the DISTANCE (lower = more similar for cosine)
    # 1 - distance = similarity (higher = more similar)
    # We order ASC by distance = closest products first
    sql = f"""
    SELECT TOP (?)
        Id,
        Name,
        Category,
        Price,
        Description,
        VECTOR_DISTANCE('cosine', DescriptionVector, CAST(? AS VECTOR({VECTOR_DIM})))
            AS CosineDistance,
        1.0 - VECTOR_DISTANCE('cosine', DescriptionVector, CAST(? AS VECTOR({VECTOR_DIM})))
            AS CosineSimilarity
    FROM {DB_TABLE}
    WHERE DescriptionVector IS NOT NULL
    ORDER BY CosineDistance ASC
    """
    cursor.execute(sql, (top_k, query_vector_json, query_vector_json))
    rows = cursor.fetchall()

    results = []
    for row in rows:
        results.append({
            "id":         row[0],
            "name":       row[1],
            "category":   row[2],
            "price":      float(row[3]),
            "description": row[4][:80] + "...",
            "distance":   round(float(row[5]), 4),
            "similarity": round(float(row[6]), 4),
        })
    return results


def sqlserver_semantic_search_legacy(cursor, query_text, vocabulary, top_k=5):
    """
    Legacy fallback for SQL Server 2019 / 2022.
    Load embeddings from JSON column, compute cosine similarity in Python.

    This is less efficient than the 2025 approach (Python does the math)
    but works on any SQL Server version.
    """
    query_vector = embed_query(query_text, vocabulary)

    # Load all product rows with their JSON embeddings
    cursor.execute(f"""
        SELECT Id, Name, Category, Price, Description, DescriptionEmbeddingJson
        FROM {DB_TABLE}
        WHERE DescriptionEmbeddingJson IS NOT NULL
    """)
    rows = cursor.fetchall()

    scores = []
    for row in rows:
        prod_id, name, category, price, description, embedding_json = row
        if not embedding_json:
            continue
        # Parse JSON back into a Python list, then to numpy array
        prod_vector = np.array(json.loads(embedding_json), dtype=np.float32)
        similarity  = float(np.dot(query_vector, prod_vector))
        scores.append({
            "id":         prod_id,
            "name":       name,
            "category":   category,
            "price":      float(price),
            "description": description[:80] + "...",
            "similarity": round(similarity, 4),
        })

    scores.sort(key=lambda x: x["similarity"], reverse=True)
    return scores[:top_k]


def sqlserver_hybrid_search(cursor, query_text, vocabulary,
                             supports_vector, alpha=0.5, top_k=5):
    """
    Hybrid search: T-SQL LIKE keyword match + Python vector similarity.
    Works on both SQL Server 2025 (uses VECTOR_DISTANCE for vector part)
    and legacy SQL Server (uses Python for vector part).

    alpha: weight for keyword score (0 = pure vector, 1 = pure keyword)
    """
    query_vector      = embed_query(query_text, vocabulary)
    query_vector_json = json.dumps(query_vector.tolist())

    # Extract search terms from query for T-SQL LIKE matching
    search_terms = remove_stop_words(tokenize(query_text))

    # Build a LIKE condition for each search term
    # C# analogy: .Where(p => searchTerms.Any(t => p.Description.Contains(t)))
    like_conditions = " OR ".join(
        f"(Description LIKE '%{term}%' OR Name LIKE '%{term}%')"
        for term in search_terms[:5]   # Limit to first 5 terms
    ) or "1=1"

    if supports_vector:
        sql = f"""
        SELECT
            Id, Name, Category, Price, Description,
            VECTOR_DISTANCE('cosine', DescriptionVector, CAST(? AS VECTOR({VECTOR_DIM})))
                AS VectorDistance,
            CASE WHEN ({like_conditions}) THEN 1.0 ELSE 0.0 END AS KeywordMatch
        FROM {DB_TABLE}
        WHERE DescriptionVector IS NOT NULL
        """
        cursor.execute(sql, (query_vector_json,))
        rows = cursor.fetchall()

        results = []
        for row in rows:
            prod_id, name, cat, price, desc, vdist, kw = row
            vec_sim = 1.0 - float(vdist)
            kw_score = float(kw)
            hybrid = alpha * kw_score + (1.0 - alpha) * vec_sim
            results.append({
                "id":         prod_id,
                "name":       name,
                "category":   cat,
                "price":      float(price),
                "vec_score":  round(vec_sim,  4),
                "kw_score":   round(kw_score, 4),
                "hybrid":     round(hybrid,   4),
            })
    else:
        # Legacy: load embeddings, compute vector score in Python, keyword in Python
        cursor.execute(f"""
            SELECT Id, Name, Category, Price, Description, DescriptionEmbeddingJson
            FROM {DB_TABLE}
            WHERE DescriptionEmbeddingJson IS NOT NULL
        """)
        rows = cursor.fetchall()
        results = []
        for row in rows:
            prod_id, name, cat, price, desc, emb_json = row
            if not emb_json:
                continue
            prod_vector = np.array(json.loads(emb_json), dtype=np.float32)
            vec_sim  = float(np.dot(query_vector, prod_vector))

            desc_lower = desc.lower()
            name_lower = name.lower()
            kw_matches = sum(
                1 for t in search_terms
                if t in desc_lower or t in name_lower
            )
            kw_score = kw_matches / max(len(search_terms), 1)

            hybrid = alpha * kw_score + (1.0 - alpha) * vec_sim
            results.append({
                "id":         prod_id,
                "name":       name,
                "category":   cat,
                "price":      float(price),
                "vec_score":  round(vec_sim,  4),
                "kw_score":   round(kw_score, 4),
                "hybrid":     round(hybrid,   4),
            })

    results.sort(key=lambda x: x["hybrid"], reverse=True)
    return results[:top_k]


# ------------------------------------------------------------------------------
# MAIN: Connect and run everything
# ------------------------------------------------------------------------------

try:
    import pyodbc   # pip install pyodbc -- Python's SqlConnection equivalent

    print("\n  Trying to connect to SQL Server...")
    print(f"  Config: {ACTIVE_CONFIG}")
    print(f"  Connection string: {CONNECTION_STRING[:60]}...")

    start_time = time.time()
    conn = pyodbc.connect(CONNECTION_STRING, timeout=5)  # 5-second timeout
    conn.autocommit = False     # Use explicit transactions (like SqlTransaction in C#)
    cursor = conn.cursor()
    elapsed = time.time() - start_time

    print(f"  Connected in {elapsed:.2f}s")

    # Detect version
    version_str, supports_vector = detect_sqlserver_version(cursor)
    print(f"\n  SQL Server Version: {version_str}")
    if supports_vector:
        print("  Native VECTOR type: YES (SQL Server 2025 / Azure SQL)")
        print("  Using: VECTOR column + VECTOR_DISTANCE() SQL function")
    else:
        print("  Native VECTOR type: NO (SQL Server 2019 / 2022 fallback)")
        print("  Using: NVARCHAR(MAX) JSON column + Python cosine similarity")

    # ----- STEP B1: Create / update table schema -----
    print("\n--- STEP B1: Create / Update Table Schema ---")
    schema_sql = CREATE_TABLE_SQL_2025 if supports_vector else CREATE_TABLE_SQL_LEGACY
    cursor.execute(schema_sql)
    conn.commit()
    print("  Schema ready.")

    # ----- STEP B2: Generate embeddings and insert data -----
    print("\n--- STEP B2: Generate Embeddings and Insert Into SQL Server ---")
    start_time = time.time()

    if supports_vector:
        insert_products_2025(cursor, PRODUCTS, product_embeddings)
    else:
        insert_products_legacy(cursor, PRODUCTS, product_embeddings)

    conn.commit()
    elapsed = time.time() - start_time

    cursor.execute(f"SELECT COUNT(*) FROM {DB_TABLE}")
    row_count = cursor.fetchone()[0]
    print(f"  {row_count} products in SQL Server.  Upsert took {elapsed:.2f}s")
    print(f"  Each product now has a {VECTOR_DIM}-dimension embedding stored.")

    # ----- STEP B3: Semantic search using SQL Server -----
    print("\n--- STEP B3: Semantic Search (runs inside SQL Server) ---")

    sql_queries = [
        "waterproof shoes for running outside",
        "reduce back pain working from home",
        "noise cancelling headphones for office",
    ]

    for query in sql_queries:
        start_time = time.time()
        if supports_vector:
            results = sqlserver_semantic_search_2025(cursor, query, vocabulary, top_k=3)
        else:
            results = sqlserver_semantic_search_legacy(cursor, query, vocabulary, top_k=3)
        elapsed = time.time() - start_time

        print(f"\n  Query: \"{query}\"  ({elapsed*1000:.1f}ms)")
        for i, r in enumerate(results, start=1):
            print(f"    {i}. {r['name']:<35} similarity: {r['similarity']:.4f}")

    # ----- STEP B4: Hybrid search -----
    print("\n--- STEP B4: Hybrid Search (T-SQL LIKE + Vector Distance) ---")

    hybrid_q = "mechanical keyboard for software developer"
    results = sqlserver_hybrid_search(
        cursor, hybrid_q, vocabulary,
        supports_vector=supports_vector, alpha=0.4, top_k=5
    )

    print(f"\n  Query: \"{hybrid_q}\"  (alpha=0.4)")
    print(f"  {'Name':<35} {'KW':>6}  {'Vec':>6}  {'Hybrid':>8}")
    print(f"  {'-'*35} {'-'*6}  {'-'*6}  {'-'*8}")
    for r in results:
        print(f"  {r['name']:<35} {r['kw_score']:>6.4f}  {r['vec_score']:>6.4f}  {r['hybrid']:>8.4f}")

    # ----- STEP B5: RAG using SQL Server results -----
    print("\n--- STEP B5: RAG Prompt from SQL Server Results ---")

    rag_query = "I need to improve my home office setup for long coding sessions"
    if supports_vector:
        rag_results = sqlserver_semantic_search_2025(cursor, rag_query, vocabulary, top_k=3)
    else:
        rag_results = sqlserver_semantic_search_legacy(cursor, rag_query, vocabulary, top_k=3)

    rag_prompt = build_product_rag_prompt(rag_query, rag_results, max_products=3)

    print(f"\n  Customer question: \"{rag_query}\"")
    print("\n  Top matches from SQL Server:")
    for r in rag_results[:3]:
        print(f"    - {r['name']}  (similarity: {r['similarity']:.4f})")
    print("\n  RAG prompt generated (ready to send to Claude / GPT-4).")

    # Clean up
    cursor.close()
    conn.close()
    print("\n  SQL Server connection closed.")

except ImportError:
    print("""
  pyodbc is not installed.  Part B skipped.

  To install:
    pip install pyodbc

  Also install the ODBC driver (Windows):
    Download "ODBC Driver 17 for SQL Server" from Microsoft:
    https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server

  C# analogy: pyodbc is Python's equivalent of System.Data.SqlClient.
""")

except Exception as e:
    # Connection failed -- SQL Server not running or wrong connection string
    # This is expected when running the demo without a SQL Server instance
    print(f"""
  Could not connect to SQL Server: {e}

  This is expected if SQL Server is not running on this machine.
  Part A (in-memory simulation) above showed you the full concept.

  To run Part B:
    1. Start SQL Server (local, Docker, or Azure SQL)
    2. Create a database called 'VectorDemo'
       (or change DATABASE= in CONNECTION_STRING above)
    3. Edit SERVER= in CONNECTION_STRING to match your server name
    4. Re-run this script

  Docker quick start (SQL Server 2022):
    docker run -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=YourPass123" ^
      -p 1433:1433 --name sqlserver ^
      -d mcr.microsoft.com/mssql/server:2022-latest

  Then update CONNECTION_STRING to:
    DRIVER={{ODBC Driver 17 for SQL Server}};
    SERVER=localhost,1433;DATABASE=VectorDemo;UID=sa;PWD=YourPass123;
""")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETE: SQL Server as a Vector Database")
print("=" * 70)

print("""
WHAT THIS PROJECT DEMONSTRATED:

1. TF-IDF EMBEDDINGS FROM SCRATCH (no API key, no GPU needed):
   - Tokenize and clean product descriptions
   - Build vocabulary and compute word importance (TF-IDF)
   - Normalize to unit vectors for fast cosine similarity

2. SQL SERVER VECTOR SUPPORT (native, no new database needed):
   - SQL Server 2025 / Azure SQL: VECTOR(N) column type
   - VECTOR_DISTANCE('cosine', col, query) for SQL-side similarity
   - SQL Server 2019 / 2022 fallback: NVARCHAR(MAX) JSON storage
   - Auto-detection of server version at runtime

3. ADDING VECTORS TO EXISTING DATA:
   - ALTER TABLE ... ADD DescriptionVector VECTOR(N) -- one command
   - Your existing rows, security, and indexes are untouched
   - Generate embeddings in Python, store back into SQL Server

4. HYBRID SEARCH:
   - Keyword: T-SQL LIKE / CONTAINS for exact term matches
   - Vector: VECTOR_DISTANCE for meaning-based matches
   - Blend with alpha parameter (tune for your use case)

5. RAG PATTERN WITH SQL SERVER:
   - Retrieve: vector search finds relevant product rows in SQL Server
   - Augment:  build LLM prompt with product data as grounded context
   - Generate: LLM answers ONLY from your actual product catalog

KEY SQL SERVER 2025 SYNTAX:
  -- Declare column
  DescriptionVector VECTOR(384)

  -- Insert / update
  UPDATE Products
  SET DescriptionVector = CAST('[0.1, 0.2, ...]' AS VECTOR(384))
  WHERE Id = 1;

  -- Search
  SELECT TOP 5 Name,
    VECTOR_DISTANCE('cosine', DescriptionVector, CAST(? AS VECTOR(384)))
  FROM Products
  ORDER BY 2 ASC;  -- lower distance = more similar

C# / .NET TAKEAWAY:
  You do not need ChromaDB, Pinecone, or Weaviate.
  If your app already uses SQL Server, just:
    1. Upgrade to SQL Server 2025 or Azure SQL
    2. Add a VECTOR column to your existing table
    3. Generate embeddings and store them (Python, C#, or Azure Function)
    4. Use VECTOR_DISTANCE() in your existing SqlCommand / EF queries

NEXT STEPS:
  - Replace TF-IDF with sentence-transformers for much better quality
    (pip install sentence-transformers, ~80MB model download)
  - Add a vector index in SQL Server for faster search on large tables:
    CREATE VECTOR INDEX ON Products(DescriptionVector) WITH (METRIC='cosine')
  - Wrap in a FastAPI endpoint (Module 09) for a production search API
  - Module 11 (Agents): build an agent that uses this search as a tool
""")

print("=" * 70)
print("END OF PROJECT 2: SQL Server as a Vector Database")
print("=" * 70)

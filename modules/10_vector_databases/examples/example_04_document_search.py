"""
Example 04: Document Search Engine
=====================================

GLOSSARY
--------
Ingestion:
  The process of loading raw documents into the vector database.
  Steps: load text -> clean -> chunk -> embed -> store.
  Done ONCE upfront (not every time a user searches).

Chunking:
  Splitting long documents into smaller pieces.
  WHY: Embedding models have token limits (e.g., 512 tokens max).
       A long article must be split before embedding.
  CHUNK SIZE: 200-500 words is a common choice.
  OVERLAP: Repeat some words between chunks so context is not lost.

Token Limit:
  The maximum number of tokens (roughly: words) an embedding model can handle.
  all-MiniLM-L6-v2: 256 tokens max. Text longer than that gets truncated.
  Chunking avoids this problem.

Overlap:
  Repeating words between adjacent chunks.
  Example: chunk 1 = words 1-200, chunk 2 = words 150-350 (50 words overlap).
  This prevents important information from being cut at chunk boundaries.

Sentence-Transformers:
  A Python library providing pre-trained embedding models.
  model.encode("text") returns a NumPy array (the embedding vector).
  model.encode(["text1","text2"]) encodes multiple at once (faster).

Batch Embedding:
  Encoding many texts at once instead of one at a time.
  The model processes them in parallel, much faster overall.
  collection.add(documents=all_chunks_at_once) uses batch embedding.

Search Index:
  ChromaDB builds an internal search index (HNSW algorithm) automatically.
  This makes similarity search fast even over thousands of documents.

WHAT THIS EXAMPLE SHOWS
------------------------
Part 1: A knowledge base of documents (simulated in-memory)
Part 2: Document ingestion pipeline (clean -> chunk -> store)
Part 3: Search function with formatted results
Part 4: Category filtering (search within a specific topic)
Part 5: Interactive search loop (type queries and see results)
Part 6: Search analytics (which categories match best)

LIBRARIES NEEDED
-----------------
  chromadb              (pip install chromadb)
  sentence-transformers (pip install sentence-transformers) -- optional
  numpy                 (pip install numpy)
  matplotlib            (pip install matplotlib)
"""

import re                         # Regular expressions for text cleaning
import numpy as np                # NumPy for analytics
import chromadb                   # Vector database

print("=" * 65)
print("EXAMPLE 04: Document Search Engine")
print("=" * 65)


# ==============================================================================
# PART 1: The Knowledge Base
# ==============================================================================

print("\n" + "=" * 65)
print("PART 1: Our Knowledge Base")
print("=" * 65)

print("""
We simulate a small company knowledge base.
In a real system, these would be loaded from files, a database, or an API.

Each document has:
  - title:    Short name
  - category: For filtering searches
  - content:  The full text (what we embed and search)
""")

# Simulated knowledge base documents
# In a real system: load from files, database, Confluence, SharePoint, etc.
knowledge_base = [
    {
        "id": "acct_01",
        "title": "Password Reset Guide",
        "category": "account",
        "content": """
        If you have forgotten your password, you can reset it by clicking the
        'Forgot Password' link on the login page. Enter your email address and
        we will send you a password reset link. The link expires after 24 hours.
        If you do not receive the email within 5 minutes, check your spam folder.
        For security, the reset link can only be used once. After resetting, you
        will need to log in with your new password on all your devices.
        """
    },
    {
        "id": "acct_02",
        "title": "Two-Factor Authentication Setup",
        "category": "account",
        "content": """
        Two-factor authentication (2FA) adds an extra layer of security to your account.
        Even if someone knows your password, they cannot log in without your phone.
        To enable 2FA: go to Settings -> Security -> Two-Factor Authentication.
        Download an authenticator app like Google Authenticator or Microsoft Authenticator.
        Scan the QR code shown in your account settings. Enter the 6-digit code from
        the app to confirm setup. Save your backup codes in a safe place -- these are
        used to regain access if you lose your phone.
        """
    },
    {
        "id": "acct_03",
        "title": "Changing Your Email Address",
        "category": "account",
        "content": """
        To update your email address, go to Settings -> Profile -> Contact Information.
        Enter your new email address and click Save. We will send a verification email
        to your new address. Click the link in that email to confirm the change.
        Until you verify the new email, your old email remains active. Note that your
        username is separate from your email -- changing your email does not change
        how you log in if you use a username.
        """
    },
    {
        "id": "bill_01",
        "title": "Subscription Plans Overview",
        "category": "billing",
        "content": """
        We offer three subscription plans: Basic (free), Pro ($9.99/month), and
        Enterprise (custom pricing). The Basic plan includes 5GB storage and up to
        3 users. The Pro plan includes 50GB storage, unlimited users, and priority support.
        Enterprise plans include custom storage limits, dedicated support, and SLA guarantees.
        All paid plans are billed monthly. Annual billing is available with a 20% discount.
        You can upgrade or downgrade your plan at any time from the Billing section.
        """
    },
    {
        "id": "bill_02",
        "title": "Cancellation and Refund Policy",
        "category": "billing",
        "content": """
        You can cancel your subscription at any time from Settings -> Billing -> Cancel Plan.
        When you cancel, you will retain access until the end of your current billing period.
        We do not offer partial refunds for unused time. However, if you cancel within the
        first 30 days of your first subscription, you qualify for a full refund. To request
        a refund, contact support with your account email and reason for cancellation.
        After cancellation, your data is retained for 30 days before deletion.
        """
    },
    {
        "id": "bill_03",
        "title": "Payment Methods",
        "category": "billing",
        "content": """
        We accept Visa, Mastercard, American Express, and PayPal. Apple Pay and Google Pay
        are available on mobile. For Enterprise accounts, we also support bank transfers and
        purchase orders. To update your payment method, go to Settings -> Billing -> Payment Methods.
        Click 'Add Payment Method' and enter your card details. Your card information is
        encrypted and never stored on our servers -- we use Stripe for secure payment processing.
        If a payment fails, we will retry 3 times over 5 days before suspending the account.
        """
    },
    {
        "id": "tech_01",
        "title": "Installation on Windows",
        "category": "technical",
        "content": """
        Download the installer from our website. Run the .exe file and follow the setup wizard.
        The application requires Windows 10 or later (64-bit). Administrator privileges are
        required during installation. The default installation directory is C:\\Program Files\\AppName.
        After installation, launch the application from the Start menu or desktop shortcut.
        If Windows Defender blocks the installer, click 'More info' then 'Run anyway'.
        The application does not require internet access after installation for basic features.
        """
    },
    {
        "id": "tech_02",
        "title": "Fixing Sync and Loading Issues",
        "category": "technical",
        "content": """
        If the application is stuck on loading or not syncing your data, try these steps:
        1. Close the application completely (check the system tray).
        2. Clear the application cache: Settings -> Advanced -> Clear Cache.
        3. Restart the application and wait 2-3 minutes for initial sync.
        4. If still stuck, check your internet connection and firewall settings.
        5. The application needs access to port 443 (HTTPS) for syncing.
        6. Disable VPN temporarily to test if it is blocking the connection.
        If the issue persists, export your logs from Help -> Export Logs and contact support.
        """
    },
    {
        "id": "tech_03",
        "title": "System Requirements",
        "category": "technical",
        "content": """
        Minimum requirements: Windows 10 (64-bit) or macOS 11, 4GB RAM, 2GB disk space.
        Recommended: Windows 11 or macOS 13, 8GB RAM, 10GB disk space, SSD storage.
        Internet: Required for sync and collaboration features. Minimum 1 Mbps.
        The application does not support Linux natively. A web browser version is available
        for Linux users. Mobile apps are available for iOS 14+ and Android 9+.
        """
    },
]

print(f"Knowledge base: {len(knowledge_base)} documents")
categories = set(doc["category"] for doc in knowledge_base)
print(f"Categories: {sorted(categories)}")
for cat in sorted(categories):
    docs_in_cat = [d for d in knowledge_base if d["category"] == cat]
    print(f"  {cat}: {len(docs_in_cat)} documents")
print()


# ==============================================================================
# PART 2: Document Ingestion Pipeline
# ==============================================================================

print("=" * 65)
print("PART 2: Document Ingestion Pipeline")
print("=" * 65)

def clean_text(text):
    """
    Clean a document's text before storing it.
    Removes extra whitespace, newlines, and leading/trailing spaces.
    """
    text = re.sub(r'\s+', ' ', text)    # Collapse any whitespace (spaces, tabs, newlines) into one space
    text = text.strip()                  # Remove leading and trailing whitespace
    return text

def chunk_text(text, chunk_size=100, overlap=20):
    """
    Split a long text into overlapping chunks.
    text:       the full text to split
    chunk_size: how many words per chunk
    overlap:    how many words to repeat between consecutive chunks
    Returns:    list of text strings (the chunks)
    """
    words  = text.split()                         # Split into individual words
    chunks = []                                   # Will hold the result chunks

    i = 0                                         # Start at word 0
    while i < len(words):                         # Keep going until we run out of words
        end = i + chunk_size                      # End of this chunk
        chunk_words = words[i:end]                # Take words from i to end
        chunk_text  = " ".join(chunk_words)       # Join back into a string
        chunks.append(chunk_text)                 # Add to our list
        if end >= len(words):                     # If we have reached the end, stop
            break
        i += chunk_size - overlap                 # Move forward, but overlap with previous chunk

    return chunks

# Demonstrate cleaning and chunking
example_doc = knowledge_base[1]    # Two-factor authentication guide
print(f"Original text ({len(example_doc['content'].split())} words):")
print(f"  {example_doc['content'][:150]}...")
print()

cleaned = clean_text(example_doc["content"])
print(f"After cleaning ({len(cleaned.split())} words):")
print(f"  {cleaned[:150]}...")
print()

chunks = chunk_text(cleaned, chunk_size=50, overlap=10)    # Small chunks for demo
print(f"After chunking (chunk_size=50, overlap=10): {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1} ({len(chunk.split())} words): {chunk[:60]}...")
print()

# Now ingest ALL documents into ChromaDB
print("Ingesting all documents into ChromaDB...")

# Create an in-memory client and collection
client = chromadb.Client()
collection = client.get_or_create_collection("knowledge_base")

all_texts     = []    # All chunk texts to store
all_ids       = []    # Corresponding IDs
all_metadatas = []    # Corresponding metadata

for doc in knowledge_base:
    # Step 1: Clean the text
    cleaned_content = clean_text(doc["content"])

    # Step 2: Chunk the text (for this example, most docs are short enough as one chunk)
    # In a real system with long docs, you would use chunk_size=200-500
    chunks = chunk_text(cleaned_content, chunk_size=150, overlap=30)

    # Step 3: Build IDs and metadata for each chunk
    for chunk_num, chunk in enumerate(chunks):
        chunk_id   = f"{doc['id']}_chunk{chunk_num}"    # e.g., "acct_01_chunk0"
        chunk_meta = {
            "doc_id":   doc["id"],                      # Original document ID
            "title":    doc["title"],                   # Document title
            "category": doc["category"],                # Category for filtering
            "chunk":    chunk_num                       # Which chunk this is
        }
        all_texts.append(chunk)                         # Add text
        all_ids.append(chunk_id)                        # Add ID
        all_metadatas.append(chunk_meta)                # Add metadata

# Add everything to ChromaDB in one batch call
collection.add(
    documents=all_texts,        # All chunk texts
    ids=all_ids,                # All chunk IDs
    metadatas=all_metadatas     # All metadata dicts
)

print(f"  Stored {collection.count()} chunks (from {len(knowledge_base)} documents)")
print(f"  Categories indexed: {sorted(set(m['category'] for m in all_metadatas))}")
print()


# ==============================================================================
# PART 3: Search Function
# ==============================================================================

print("=" * 65)
print("PART 3: Search Function")
print("=" * 65)

def search_knowledge_base(query, collection, top_k=5, category=None):
    """
    Search the knowledge base for documents similar to the query.
    query:      user's search question (plain text)
    collection: ChromaDB collection to search
    top_k:      how many results to return
    category:   optional filter ("account", "billing", "technical")
    Returns:    list of result dicts sorted by similarity
    """
    # Build the query parameters
    query_params = {
        "query_texts": [query],    # ChromaDB auto-embeds this text
        "n_results":   top_k       # Return top_k most similar chunks
    }

    if category:                   # Add category filter if specified
        query_params["where"] = {"category": category}

    # Run the query
    raw = collection.query(**query_params)

    # Format into readable results
    results = []
    for i in range(len(raw["documents"][0])):
        doc_text   = raw["documents"][0][i]              # The matched chunk text
        distance   = raw["distances"][0][i]              # How different (lower = closer)
        similarity = round(1.0 - distance, 4)            # Convert to similarity score
        meta       = raw["metadatas"][0][i]              # Metadata dict

        results.append({
            "rank":       i + 1,                         # 1-based rank
            "similarity": similarity,                    # 0.0 to 1.0 (higher = better)
            "title":      meta.get("title", "?"),        # Document title
            "category":   meta.get("category", "?"),     # Category
            "doc_id":     meta.get("doc_id", "?"),       # Original doc ID
            "text":       doc_text                       # The matched text chunk
        })

    return results

def display_results(query, results):
    """
    Print search results in a readable format.
    """
    print(f"  Query: '{query}'")
    print(f"  Found {len(results)} results:")
    print()
    for r in results:
        # Build a similarity bar (ASCII visualization)
        bar_len = int(r["similarity"] * 20)        # Scale to 20 characters
        bar     = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  [{r['rank']}] {r['title']}")
        print(f"      Similarity: {r['similarity']:.4f}  [{bar}]")
        print(f"      Category:   {r['category']}")
        print(f"      Preview:    {r['text'][:80]}...")
        print()

# Run several test searches
test_searches = [
    ("I forgot my password and cannot get into my account",           None),
    ("How do I stop being charged every month?",                       None),
    ("The app is not loading and keeps showing a spinning circle",     None),
    ("Do you support Apple Pay?",                                      None),
    ("How do I make my account more secure?",                          "account"),
]

for query, category_filter in test_searches:
    filter_label = f" (filter: {category_filter})" if category_filter else ""
    print(f"--- Search{filter_label} ---")
    results = search_knowledge_base(query, collection, top_k=3, category=category_filter)
    display_results(query, results)
    print()


# ==============================================================================
# PART 4: Category-Aware Search
# ==============================================================================

print("=" * 65)
print("PART 4: Category-Aware Search")
print("=" * 65)

print("""
In a real app, you might route queries to the right category automatically.
A simple approach: search all categories, then show which category dominated the results.
""")

def categorized_search(query, collection, top_k=6):
    """
    Search without filter, then analyze which categories appear most.
    Shows which category of content is most relevant to the query.
    """
    results = search_knowledge_base(query, collection, top_k=top_k)

    # Count which categories appear in the results
    category_counts = {}
    category_scores = {}

    for r in results:
        cat = r["category"]
        if cat not in category_counts:
            category_counts[cat] = 0
            category_scores[cat] = 0.0
        category_counts[cat] += 1                  # Count how many results are in this category
        category_scores[cat] += r["similarity"]    # Sum of similarity scores for this category

    # Find the dominant category
    dominant_cat = max(category_scores, key=category_scores.get)

    return results, category_scores, dominant_cat

q = "I cannot access my account because I lost my phone"
results, cat_scores, dominant = categorized_search(q, collection)

print(f"  Query: '{q}'")
print(f"  Category scores (sum of similarities across results):")
for cat, score in sorted(cat_scores.items(), key=lambda x: x[1], reverse=True):
    bar = "#" * int(score * 10)
    print(f"    {cat:12s}: {score:.4f}  [{bar}]")
print(f"  -> Best category: '{dominant}'")
print()
print(f"  Top results in '{dominant}' category:")
account_results = [r for r in results if r["category"] == dominant]
for r in account_results[:2]:
    print(f"    [{r['rank']}] {r['title']} (similarity: {r['similarity']:.4f})")
print()


# ==============================================================================
# PART 5: Interactive Search Loop
# ==============================================================================

print("=" * 65)
print("PART 5: Interactive Search Loop")
print("=" * 65)

print("""
Now running an interactive search demo.
In a real app, this would be a web UI or API endpoint.

Type any question to search the knowledge base.
Type 'quit' to exit.
Type 'billing:your question' to search only billing docs.
Type 'account:your question' to search only account docs.
Type 'technical:your question' to search only technical docs.
""")

# Pre-defined test queries (so the example runs non-interactively)
# In a real app, you would use: query = input("Your question: ")
demo_queries = [
    "What happens if I forget my 2FA device?",
    "billing:What is the price of the premium plan?",
    "technical:My application crashes on startup",
    "How do I get my money back?",
]

for raw_query in demo_queries:
    # Check for category prefix
    category_filter = None
    query = raw_query

    for cat in ["account", "billing", "technical"]:
        if raw_query.lower().startswith(f"{cat}:"):        # Check for prefix like "billing:"
            category_filter = cat                           # Set the filter
            query = raw_query[len(cat) + 1:].strip()       # Remove the prefix
            break

    print(f"> {raw_query}")

    results = search_knowledge_base(
        query      = query,
        collection = collection,
        top_k      = 2,
        category   = category_filter
    )

    if not results:
        print("  No results found.")
    else:
        for r in results:
            print(f"  [{r['rank']}] {r['title']} ({r['category']}) sim={r['similarity']:.3f}")
            print(f"       {r['text'][:80]}...")
    print()


# ==============================================================================
# SUMMARY
# ==============================================================================

print("=" * 65)
print("SUMMARY - Document Search Engine")
print("=" * 65)

print("""
WHAT WE BUILT:

A complete document search pipeline:

  1. INGEST:
     - Load documents (from dict, files, database, etc.)
     - Clean text (remove extra whitespace)
     - Chunk long documents (split into pieces that fit the model)
     - Add to ChromaDB (auto-embedding + storage)

  2. SEARCH:
     - User provides a query (plain English text)
     - ChromaDB embeds the query automatically
     - Finds the most similar document chunks
     - Returns results sorted by similarity

  3. FILTER:
     - category filter narrows results to a topic area
     - like a SQL WHERE clause on the metadata

  4. DISPLAY:
     - Format results with similarity scores
     - Show ASCII bar for visual comparison

KEY INSIGHT:
  The user types plain English.
  The system finds relevant documents by MEANING, not exact words.
  "I cannot log in" finds "password reset" articles -- because the MEANING is similar.

NEXT EXAMPLE (05):
  Full semantic search system using sentence-transformers for high-quality
  embeddings, plus a demonstration of the full RAG pipeline.
""")

print("=" * 65)
print("END OF EXAMPLE 04")
print("=" * 65)

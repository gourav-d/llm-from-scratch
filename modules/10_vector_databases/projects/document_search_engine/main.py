"""
Project: Document Search Engine
==================================

DESCRIPTION:
  A production-style semantic search engine for a company knowledge base.
  Combines everything from Module 10:
    - ChromaDB for vector storage
    - Cosine similarity for ranking
    - Metadata filtering
    - Chunking for long documents
    - RAG prompt building
    - Search analytics and evaluation

WHAT THIS BUILDS:
  A complete, runnable document search system that a company could use
  for their internal knowledge base or customer-facing help center.

  Features:
    1. Load and ingest a knowledge base of documents
    2. Search by meaning (semantic similarity)
    3. Filter by category or priority
    4. Display results with similarity scores
    5. Build a RAG prompt for an LLM (ready to call OpenAI/Claude)
    6. Evaluate search quality with test cases
    7. Show search analytics

ARCHITECTURE:

  [Knowledge Base]      ->  Ingestion Pipeline  ->  ChromaDB Collection
  (text documents)          (clean, chunk, add)       (vectors + metadata)

  [User Query]          ->  Embed Query         ->  ChromaDB Search
  (plain text)              (auto by ChromaDB)        (top K results)

  [Search Results]      ->  RAG Prompt Builder  ->  LLM (simulated)
  (relevant chunks)         (context + question)      (answer)

GLOSSARY (key terms)
---------------------
  Ingestion: loading and storing documents in the vector database (done once)
  Query:     searching for documents (done on every user request)
  RAG:       Retrieve-Augment-Generate (vector search + LLM answer generation)
  Chunk:     a piece of a larger document (to fit embedding model token limits)
  Metadata:  extra info stored with each document (for filtering and display)
  HNSW:      the indexing algorithm ChromaDB uses internally for fast search

LIBRARIES:
  chromadb   (pip install chromadb)
  numpy      (pip install numpy) -- for analytics only
  matplotlib (pip install matplotlib) -- for visualization
  sentence-transformers (optional, pip install sentence-transformers)
"""

import re                    # For text cleaning (regular expressions)
import time                  # For measuring search speed
import numpy as np           # For analytics and statistics
import chromadb              # The vector database

print("=" * 70)
print("PROJECT: Document Search Engine")
print("Combining all Module 10 concepts in one complete system")
print("=" * 70)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# These settings control the behavior of the search engine
CONFIG = {
    "collection_name":  "company_knowledge_base",  # Name of the ChromaDB collection
    "chunk_size":       150,    # Words per chunk (for long documents)
    "chunk_overlap":    30,     # Words to repeat between adjacent chunks
    "default_top_k":   5,       # How many results to return by default
    "similarity_threshold": 0.3, # Minimum similarity to show a result (0 = show all)
}


# ==============================================================================
# THE KNOWLEDGE BASE
# ==============================================================================

# This simulates a company's internal knowledge base.
# In production: load from files, a database, Confluence, SharePoint, etc.

KNOWLEDGE_BASE = [
    # ---- Onboarding ----
    {
        "id": "on_01", "category": "onboarding", "priority": "high",
        "title": "Welcome and Account Setup",
        "content": """
        Welcome to our platform! Here is how to get started in 5 steps.
        Step 1: Create your account using your corporate email address.
        Step 2: Verify your email by clicking the link we sent you.
        Step 3: Complete your profile (name, job title, department).
        Step 4: Set up your workspace by entering your company name and team size.
        Step 5: Invite your teammates -- go to Settings and click Invite Users.
        If you run into any issues during setup, contact support at help@company.com.
        """
    },
    {
        "id": "on_02", "category": "onboarding", "priority": "medium",
        "title": "First Day Checklist",
        "content": """
        On your first day using the platform, complete these tasks to get fully set up.
        Connect your calendar: go to Integrations and connect Google Calendar or Outlook.
        Install the desktop app: download from our website for faster access.
        Set your notification preferences: choose how often you want email summaries.
        Join your first project: your admin should have added you already.
        Review the keyboard shortcuts: Help menu -> Keyboard Shortcuts.
        """
    },

    # ---- Account Management ----
    {
        "id": "acct_01", "category": "account", "priority": "high",
        "title": "Password Reset",
        "content": """
        If you have forgotten your password, go to the login page and click Forgot Password.
        Enter your registered email address. You will receive a reset link within 2 minutes.
        The link is valid for 24 hours and can only be used once.
        If you do not receive the email, check your spam folder.
        If you remember your password but want to change it, go to Settings -> Security -> Change Password.
        For security reasons, you cannot reuse your last 5 passwords.
        """
    },
    {
        "id": "acct_02", "category": "account", "priority": "high",
        "title": "Two-Factor Authentication (2FA)",
        "content": """
        Two-factor authentication adds an extra layer of security to your account.
        Even if someone knows your password, they cannot log in without your phone.
        To enable: Settings -> Security -> Two-Factor Authentication -> Enable.
        Download Google Authenticator or Microsoft Authenticator on your phone.
        Scan the QR code displayed on screen. Enter the 6-digit code to confirm setup.
        IMPORTANT: Save your backup codes. If you lose your phone, these let you recover access.
        Admins can require 2FA for all team members from the admin settings panel.
        """
    },
    {
        "id": "acct_03", "category": "account", "priority": "low",
        "title": "Profile and Display Settings",
        "content": """
        Customize your profile from Settings -> Profile.
        Change your display name, job title, and profile picture.
        Your email address is used for login and cannot be changed without verification.
        To change your email: Settings -> Profile -> Email Address -> Request Change.
        A verification link is sent to your new email. Your old email stays active until confirmed.
        You can also set your timezone and language preferences in the same section.
        """
    },

    # ---- Billing ----
    {
        "id": "bill_01", "category": "billing", "priority": "high",
        "title": "Subscription Plans",
        "content": """
        We offer three plans: Starter (free), Professional ($15/user/month), Enterprise (custom).
        Starter: up to 5 users, 10GB storage, basic features, community support.
        Professional: unlimited users, 100GB storage, all features, priority support.
        Enterprise: custom storage, dedicated support, SLA guarantees, single sign-on (SSO).
        All paid plans are billed monthly. Annual billing saves 20%.
        You can upgrade or downgrade at any time from Settings -> Billing -> Change Plan.
        Upgrades take effect immediately. Downgrades take effect at the next billing cycle.
        """
    },
    {
        "id": "bill_02", "category": "billing", "priority": "high",
        "title": "Cancellation and Refunds",
        "content": """
        You can cancel your subscription at any time. No cancellation fees.
        Go to Settings -> Billing -> Cancel Subscription. Follow the confirmation steps.
        After cancellation: you retain access until the end of your current billing period.
        Data retention: your data is kept for 30 days after cancellation, then deleted.
        Refund policy: we offer a full refund within 30 days of your first purchase.
        After 30 days, no partial refunds for unused time. Annual plans: contact support.
        To request a refund, email billing@company.com with your account email and reason.
        """
    },
    {
        "id": "bill_03", "category": "billing", "priority": "medium",
        "title": "Payment Methods",
        "content": """
        Accepted payment methods: Visa, Mastercard, American Express, PayPal.
        Apple Pay and Google Pay available in the mobile app.
        Enterprise accounts can use bank wire transfer and purchase orders.
        To update your payment: Settings -> Billing -> Payment Methods -> Add New.
        We use Stripe for secure payment processing. Your card details are never stored on our servers.
        If a payment fails, we retry 3 times over 5 days, then suspend the account.
        You will receive email notifications for each failed payment attempt.
        """
    },

    # ---- Technical Support ----
    {
        "id": "tech_01", "category": "technical", "priority": "medium",
        "title": "Installing the Desktop Application",
        "content": """
        The desktop application is available for Windows 10+, macOS 11+.
        Download from our website: App Downloads section. Run the installer.
        Windows: run the .exe file as administrator if prompted.
        macOS: drag the app to your Applications folder.
        First launch requires an internet connection to complete setup.
        The app syncs automatically when you are online.
        If the app does not launch, check that your system meets the minimum requirements.
        Minimum: 4GB RAM, 2GB free disk space, 64-bit operating system.
        """
    },
    {
        "id": "tech_02", "category": "technical", "priority": "high",
        "title": "Fixing Sync and Loading Issues",
        "content": """
        If the app is stuck loading or not syncing, follow these steps.
        Step 1: Check your internet connection. The app requires HTTPS on port 443.
        Step 2: Close the app completely (check the taskbar/system tray for hidden instances).
        Step 3: Clear the app cache: Settings -> Advanced -> Clear Cache -> Confirm.
        Step 4: Restart the app and wait 2-3 minutes for initial sync to complete.
        Step 5: If behind a corporate firewall, ask your IT team to whitelist our domain.
        Step 6: Disable your VPN temporarily to test if it is blocking the connection.
        Step 7: If still failing, export logs from Help -> Export Logs and contact support.
        """
    },
    {
        "id": "tech_03", "category": "technical", "priority": "low",
        "title": "Browser Compatibility",
        "content": """
        The web app supports Chrome 90+, Firefox 88+, Safari 14+, Edge 90+.
        Internet Explorer is not supported. Please upgrade to a modern browser.
        For best performance, use Chrome or Edge on Windows.
        Enable JavaScript and cookies -- the app requires both.
        If you see display issues, try clearing your browser cache (Ctrl+Shift+Delete on Windows).
        Extensions like ad blockers may interfere with some features -- try disabling them.
        """
    },

    # ---- Developer / API ----
    {
        "id": "dev_01", "category": "developer", "priority": "medium",
        "title": "API Authentication",
        "content": """
        All API requests require an API key in the Authorization header.
        Generate an API key: Settings -> Developer -> API Keys -> Create New Key.
        Include in requests: Authorization: Bearer YOUR_API_KEY
        API keys are scoped: read-only, read-write, or admin.
        Never share your API key. If compromised, revoke it immediately and generate a new one.
        Rate limits: Free plan: 100 req/hour. Pro: 1000 req/hour. Enterprise: 10000 req/hour.
        Requests that exceed the rate limit receive HTTP 429 (Too Many Requests).
        """
    },
    {
        "id": "dev_02", "category": "developer", "priority": "low",
        "title": "Webhooks Configuration",
        "content": """
        Webhooks let your system receive real-time notifications when events occur.
        Configure webhooks: Settings -> Developer -> Webhooks -> Add Webhook.
        Supported events: user.created, project.updated, document.deleted, payment.failed.
        Your endpoint must respond with HTTP 200 within 10 seconds.
        We retry failed webhooks 3 times with exponential backoff.
        Verify webhook authenticity using the signature header (HMAC-SHA256 of the payload).
        """
    },
]


# ==============================================================================
# INGESTION PIPELINE
# ==============================================================================

def clean_text(text):
    """
    Remove extra whitespace and newlines from document text.
    """
    text = re.sub(r'\s+', ' ', text)    # Replace any whitespace sequence with a single space
    text = text.strip()                  # Remove leading and trailing whitespace
    return text

def chunk_document(text, chunk_size, overlap):
    """
    Split a document into overlapping chunks.
    text:       the full document text
    chunk_size: words per chunk
    overlap:    words to repeat between chunks (prevents losing context at boundaries)
    Returns:    list of text strings
    """
    words  = text.split()                          # Split text into individual words
    chunks = []

    i = 0
    while i < len(words):
        end         = min(i + chunk_size, len(words))   # Do not go past the end
        chunk_words = words[i:end]                       # Slice of words for this chunk
        chunks.append(" ".join(chunk_words))             # Join back to string
        if end >= len(words):                            # Reached the end
            break
        i += chunk_size - overlap                        # Advance with overlap

    return chunks

def ingest_knowledge_base(knowledge_base, collection, config):
    """
    Load all documents into the ChromaDB collection.
    knowledge_base: list of document dicts
    collection:     ChromaDB collection object
    config:         configuration dict
    Returns: total number of chunks stored
    """
    all_texts     = []    # Chunk texts
    all_ids       = []    # Chunk IDs (unique)
    all_metadatas = []    # Chunk metadata

    for doc in knowledge_base:
        # Step 1: Clean the content
        clean = clean_text(doc["content"])

        # Step 2: Chunk it (split into pieces that fit the embedding model)
        chunks = chunk_document(clean, config["chunk_size"], config["chunk_overlap"])

        # Step 3: Build IDs and metadata for each chunk
        for chunk_num, chunk_text in enumerate(chunks):
            chunk_id = f"{doc['id']}_c{chunk_num}"    # e.g., "acct_01_c0", "acct_01_c1"
            meta = {
                "doc_id":   doc["id"],               # Original document ID
                "title":    doc["title"],             # For display in results
                "category": doc["category"],          # For filtering
                "priority": doc["priority"],          # For filtering
                "chunk":    chunk_num,                # Chunk number within the document
            }
            all_texts.append(chunk_text)
            all_ids.append(chunk_id)
            all_metadatas.append(meta)

    # Add all chunks to ChromaDB in one batch (efficient)
    collection.add(
        documents=all_texts,
        ids=all_ids,
        metadatas=all_metadatas
    )

    return len(all_texts)


# ==============================================================================
# SEARCH ENGINE
# ==============================================================================

def search(query, collection, config, category=None, priority=None):
    """
    Search for documents similar to the query.
    query:      plain text query from the user
    collection: ChromaDB collection
    config:     configuration dict
    category:   optional filter ("account", "billing", "technical", "developer", "onboarding")
    priority:   optional filter ("high", "medium", "low")
    Returns:    list of result dicts
    """
    query_params = {
        "query_texts": [query],
        "n_results":   config["default_top_k"],
    }

    # Build metadata filter
    filters = []
    if category:
        filters.append({"category": {"$eq": category}})
    if priority:
        filters.append({"priority": {"$eq": priority}})

    if len(filters) == 1:                           # Single filter
        query_params["where"] = filters[0]
    elif len(filters) > 1:                          # Multiple filters (AND)
        query_params["where"] = {"$and": filters}

    # Run the query
    raw = collection.query(**query_params)

    # Format results
    results = []
    seen_docs = set()                               # Track to avoid duplicate doc IDs

    for i in range(len(raw["documents"][0])):
        doc_id   = raw["ids"][0][i]
        text     = raw["documents"][0][i]
        distance = raw["distances"][0][i]
        meta     = raw["metadatas"][0][i]

        similarity = round(1.0 - distance, 4)

        # Skip results below the threshold
        if similarity < config["similarity_threshold"]:
            continue

        # De-duplicate: if we already have a chunk from this document, skip
        orig_doc_id = meta.get("doc_id", doc_id)
        if orig_doc_id in seen_docs:
            continue
        seen_docs.add(orig_doc_id)

        results.append({
            "chunk_id":   doc_id,
            "doc_id":     orig_doc_id,
            "title":      meta.get("title", "?"),
            "category":   meta.get("category", "?"),
            "priority":   meta.get("priority", "?"),
            "similarity": similarity,
            "text":       text,
        })

    return results

def display_results(query, results, show_text=True):
    """Print search results in a readable format."""
    print(f"  Query: '{query}'")
    if not results:
        print("  No results found above the similarity threshold.")
        return

    print(f"  {len(results)} result(s):\n")
    for i, r in enumerate(results, start=1):
        bar_len = int(r["similarity"] * 25)        # Scale similarity to 25-char bar
        bar     = "#" * bar_len + "-" * (25 - bar_len)
        print(f"  [{i}] {r['title']}  ({r['category']} / {r['priority']} priority)")
        print(f"       Similarity: {r['similarity']:.4f}  [{bar}]")
        if show_text:
            print(f"       Preview: {r['text'][:85]}...")
        print()


# ==============================================================================
# RAG PROMPT BUILDER
# ==============================================================================

def build_rag_prompt(query, results, max_context_docs=3):
    """
    Build a prompt for an LLM using retrieved documents as context.
    This is the "Retrieval-Augmented Generation" pattern.

    query:            the user's question
    results:          search results (from the search() function)
    max_context_docs: how many retrieved docs to include in the prompt
    Returns: the full prompt string ready to send to an LLM
    """
    # Take the top N results for context (do not exceed context window)
    context_docs = results[:max_context_docs]

    if not context_docs:
        return f"Question: {query}\n\nAnswer: I could not find relevant information."

    # Format retrieved documents as numbered context
    context_lines = []
    for i, doc in enumerate(context_docs, start=1):
        context_lines.append(
            f"[Source {i}: {doc['title']} (similarity: {doc['similarity']:.3f})]"
        )
        context_lines.append(doc["text"])
        context_lines.append("")    # Blank line between sources

    context_block = "\n".join(context_lines)

    # The full prompt (what you would send to GPT-4, Claude, etc.)
    prompt = f"""You are a helpful customer support assistant for our software platform.
Use ONLY the context below to answer the question.
If the context does not contain enough information to answer, say so clearly.
Always be concise and direct. Cite which source you used.

--- RETRIEVED CONTEXT ---
{context_block}
--- END CONTEXT ---

Customer Question: {query}

Answer:"""

    return prompt


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate(test_cases, collection, config):
    """
    Evaluate search quality using a labeled test set.
    test_cases: list of {query, expected_doc_ids, description} dicts
    Returns: dict of metrics
    """
    precisions = []
    recalls    = []

    for test in test_cases:
        query    = test["query"]
        expected = set(test["expected_doc_ids"])    # Set of correct doc IDs

        results = search(query, collection, config)

        # Get the original doc IDs from the results (ignoring chunk numbers)
        returned_docs = set(r["doc_id"] for r in results)

        tp = len(returned_docs & expected)                     # Correctly found
        precision = tp / max(len(returned_docs), 1)            # Fraction correct
        recall    = tp / max(len(expected), 1)                 # Fraction of all correct found

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = round(np.mean(precisions), 4)
    avg_recall    = round(np.mean(recalls),    4)
    f1            = round(2 * avg_precision * avg_recall / max(avg_precision + avg_recall, 1e-8), 4)

    return {
        "precision": avg_precision,
        "recall":    avg_recall,
        "f1":        f1,
        "n_tests":   len(test_cases)
    }


# ==============================================================================
# MAIN: RUN EVERYTHING
# ==============================================================================

print("\n" + "=" * 70)
print("STEP 1: Initialize Database")
print("=" * 70)

client     = chromadb.Client()                                          # In-memory client
collection = client.get_or_create_collection(CONFIG["collection_name"]) # Create collection

print(f"  Collection '{CONFIG['collection_name']}' created.")

print("\n" + "=" * 70)
print("STEP 2: Ingest Knowledge Base")
print("=" * 70)

start_time    = time.time()
total_chunks  = ingest_knowledge_base(KNOWLEDGE_BASE, collection, CONFIG)
elapsed       = time.time() - start_time

print(f"  Ingested {len(KNOWLEDGE_BASE)} documents -> {total_chunks} chunks")
print(f"  ChromaDB now has: {collection.count()} stored chunks")
print(f"  Ingestion time:   {elapsed:.2f} seconds")

categories = sorted(set(m["category"] for m in collection.get()["metadatas"]))
print(f"  Categories: {categories}")

print("\n" + "=" * 70)
print("STEP 3: Search Demonstrations")
print("=" * 70)

demo_searches = [
    {"query": "I cannot log in to my account", "category": None},
    {"query": "How do I make my account safer?", "category": "account"},
    {"query": "I want to stop paying", "category": None},
    {"query": "The application keeps crashing", "category": "technical"},
    {"query": "How do I invite my colleagues?", "category": None},
    {"query": "How do I use the REST API?", "category": "developer"},
]

for s in demo_searches:
    filter_label = f"  [filter: category='{s['category']}']" if s["category"] else ""
    print(f"  ---{filter_label}")
    results = search(s["query"], collection, CONFIG, category=s["category"])
    display_results(s["query"], results, show_text=True)

print("\n" + "=" * 70)
print("STEP 4: RAG Prompt Example")
print("=" * 70)

rag_query   = "I forgot my password and cannot get into my account"
rag_results = search(rag_query, collection, CONFIG)
rag_prompt  = build_rag_prompt(rag_query, rag_results, max_context_docs=2)

print(f"  Query: '{rag_query}'")
print()
print("  GENERATED RAG PROMPT:")
print("  " + "\n  ".join(rag_prompt.split("\n")))
print()
print("  In production: send this prompt to OpenAI/Anthropic API for the final answer.")

print("\n" + "=" * 70)
print("STEP 5: Evaluate Search Quality")
print("=" * 70)

test_cases = [
    {
        "query":            "I forgot my password",
        "expected_doc_ids": {"acct_01"},
        "description":      "Should find password reset doc"
    },
    {
        "query":            "cancel my account and get money back",
        "expected_doc_ids": {"bill_02"},
        "description":      "Should find cancellation and refund doc"
    },
    {
        "query":            "set up two factor auth on my phone",
        "expected_doc_ids": {"acct_02"},
        "description":      "Should find 2FA doc"
    },
    {
        "query":            "application is loading slowly",
        "expected_doc_ids": {"tech_02"},
        "description":      "Should find sync/loading issues doc"
    },
    {
        "query":            "API rate limiting and authentication",
        "expected_doc_ids": {"dev_01"},
        "description":      "Should find API authentication doc"
    },
]

metrics = evaluate(test_cases, collection, CONFIG)
print(f"  Evaluation over {metrics['n_tests']} test cases:")
print(f"    Precision: {metrics['precision']:.4f}  (fraction of results that are correct)")
print(f"    Recall:    {metrics['recall']:.4f}  (fraction of correct docs found)")
print(f"    F1 Score:  {metrics['f1']:.4f}  (harmonic mean of precision and recall)")
print()

if metrics["f1"] >= 0.7:
    print("  Result: Good search quality (F1 >= 0.7)")
elif metrics["f1"] >= 0.5:
    print("  Result: Acceptable search quality (F1 >= 0.5)")
else:
    print("  Result: Search quality needs improvement -- consider better chunking or embedding model")

print("\n" + "=" * 70)
print("STEP 6: High-Priority Content Only")
print("=" * 70)

high_priority_query = "How do I secure my account?"
hp_results = search(high_priority_query, collection, CONFIG, priority="high")

print(f"  Query: '{high_priority_query}'  (showing HIGH PRIORITY docs only)")
display_results(high_priority_query, hp_results, show_text=False)

print("\n" + "=" * 70)
print("PROJECT COMPLETE")
print("=" * 70)

print("""
WHAT THIS PROJECT DEMONSTRATED:

1. INGESTION PIPELINE:
   - Clean: remove extra whitespace
   - Chunk: split long docs into 150-word pieces with 30-word overlap
   - Store: batch add to ChromaDB (all at once, not one by one)

2. SEARCH:
   - Semantic: finds docs by meaning, not word match
   - De-duplication: one result per source document (not multiple chunks)
   - Threshold: hides low-relevance results (similarity < 0.3)
   - Filtering: category and priority filters for targeted search

3. RAG:
   - Retrieve: find relevant docs from vector DB
   - Augment: build LLM prompt with context
   - Generate: ready to send to GPT-4/Claude (simulated here)

4. EVALUATION:
   - Precision@K, Recall@K, F1 on a labeled test set
   - Know your search quality before going to production

5. ARCHITECTURE READY FOR PRODUCTION:
   - Replace chromadb.Client() with chromadb.PersistentClient(path="./db")
   - Replace KNOWLEDGE_BASE with documents loaded from files or a database
   - Replace the simulated LLM with an actual OpenAI/Anthropic API call
   - Add a FastAPI wrapper (see Module 09) to serve as a REST API

NEXT STEPS:
  - Module 11: LLM Agents (autonomous AI that uses tools, including this search engine)
  - Read: modules/10_vector_databases/lessons/05_real_world_applications.md
""")

print("=" * 70)
print("END OF PROJECT: Document Search Engine")
print("=" * 70)

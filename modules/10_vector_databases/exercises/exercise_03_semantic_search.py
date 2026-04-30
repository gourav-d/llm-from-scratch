"""
Exercise 03: Semantic Search with ChromaDB
============================================

GOAL: Build a complete semantic search system using ChromaDB.
      This is the closest exercise to real production code.

WHAT YOU WILL BUILD:
  A SemanticSearchEngine class that:
    - Ingests a set of documents (stores them in ChromaDB)
    - Searches by meaning (not exact keywords)
    - Filters results by metadata
    - Displays results in a user-friendly format
    - Reports basic search analytics

INSTRUCTIONS:
  1. Read the class docstring and method signatures carefully
  2. Fill in each TODO with your code
  3. Run: python exercise_03_semantic_search.py
  4. All assertions and test outputs should look correct

GLOSSARY
--------
  chromadb.Client()          -> creates in-memory vector database
  collection.add()           -> stores documents + auto-embeds them
  collection.query()         -> finds similar documents
  results['documents'][0]    -> list of matched text (for first query)
  results['distances'][0]    -> list of distances (lower = more similar)
  results['metadatas'][0]    -> list of metadata dicts
  1.0 - distance             -> converts distance to similarity score
  where={"key": "value"}     -> metadata filter in collection.query()

LIBRARIES NEEDED:
  chromadb   (pip install chromadb)
"""

import chromadb

print("=" * 60)
print("EXERCISE 03: Semantic Search with ChromaDB")
print("=" * 60)


# ==============================================================================
# THE DOCUMENTS TO SEARCH
# ==============================================================================

# A sample dataset: FAQ articles for a fictional software product
FAQ_DOCUMENTS = [
    {
        "id":       "faq_01",
        "title":    "Getting Started Guide",
        "category": "onboarding",
        "text":     "Welcome to our platform. To get started, create an account using your work email. After signing up, you will receive a confirmation email. Click the link to verify your account. Then complete your profile and invite your teammates."
    },
    {
        "id":       "faq_02",
        "title":    "Password Reset Instructions",
        "category": "account",
        "text":     "If you forgot your password, click the Forgot Password link on the login page. Enter your email address. We will send a reset link valid for 24 hours. Click the link in your email and enter your new password twice to confirm."
    },
    {
        "id":       "faq_03",
        "title":    "Adding Team Members",
        "category": "collaboration",
        "text":     "To add colleagues to your workspace, go to Settings and select Team Members. Click Invite Users and enter their email addresses. You can assign roles: Admin, Editor, or Viewer. Invited users receive an email with instructions to join."
    },
    {
        "id":       "faq_04",
        "title":    "Canceling Your Subscription",
        "category": "billing",
        "text":     "To cancel your subscription, go to Settings and click Billing. Select Cancel Plan and follow the steps. You will keep access until the end of your billing period. We do not offer partial refunds for unused time."
    },
    {
        "id":       "faq_05",
        "title":    "Exporting Your Data",
        "category": "data",
        "text":     "You can export all your data at any time. Go to Settings and click Data Export. Choose your export format: CSV, JSON, or Excel. Large exports may take a few minutes. You will receive a download link by email when the export is ready."
    },
    {
        "id":       "faq_06",
        "title":    "Two-Factor Authentication",
        "category": "account",
        "text":     "Enable 2FA from Settings under Security. Download an authenticator app like Google Authenticator or Microsoft Authenticator. Scan the QR code shown on screen. Enter the 6-digit code to confirm. Save your backup codes in case you lose your phone."
    },
    {
        "id":       "faq_07",
        "title":    "Upgrading Your Plan",
        "category": "billing",
        "text":     "To upgrade your subscription, go to Settings and click Billing. Select Upgrade Plan and choose your new plan. Changes take effect immediately. You will be charged the prorated difference for the current billing period."
    },
    {
        "id":       "faq_08",
        "title":    "Sharing and Permissions",
        "category": "collaboration",
        "text":     "Each workspace member has a role that determines their permissions. Admins can manage settings and billing. Editors can create and modify content. Viewers can only read. You can change a member's role from the Team Members section in Settings."
    },
    {
        "id":       "faq_09",
        "title":    "API Access and Integration",
        "category": "developer",
        "text":     "Access our REST API with an API key generated in Settings under Developer. All requests require the Authorization header with your API key. Rate limits are 1000 requests per hour on free plans and 10000 per hour on paid plans. See our API documentation for endpoint details."
    },
    {
        "id":       "faq_10",
        "title":    "Troubleshooting Connection Issues",
        "category": "technical",
        "text":     "If the application is not connecting, check your internet connection first. Our service requires HTTPS on port 443. If behind a corporate firewall, add our domain to the allowlist. Clear your browser cache and cookies. Try a different browser or incognito mode."
    },
]


# ==============================================================================
# TASK 1: Implement SemanticSearchEngine
# ==============================================================================

print("\n--- Task 1: Implement SemanticSearchEngine ---")

class SemanticSearchEngine:
    """
    A semantic search engine backed by ChromaDB.

    Usage:
        engine = SemanticSearchEngine()
        engine.ingest(FAQ_DOCUMENTS)
        results = engine.search("I forgot how to log in")
        engine.display(results)
    """

    def __init__(self, collection_name="faq_search"):
        """
        Initialize the engine with an in-memory ChromaDB collection.

        TODO 1a: Create a chromadb.Client() and store it as self.client
        TODO 1b: Create a collection using client.get_or_create_collection()
                 with the given collection_name. Store as self.collection.
        """
        # TODO 1a: Create the ChromaDB client (in-memory)
        self.client = None         # Replace None with your code

        # TODO 1b: Create (or get) the collection
        self.collection = None     # Replace None with your code

    def ingest(self, documents):
        """
        Add a list of documents to the ChromaDB collection.

        Each document in the list is a dict with: id, title, category, text

        Store the 'text' field as the document content.
        Store 'title' and 'category' in the metadata.
        Use the 'id' field as the ChromaDB document ID.

        Hint: collection.add(documents=[...], ids=[...], metadatas=[...])

        TODO 1c: Implement this method.
        """
        texts     = []    # List of text strings (one per document)
        ids       = []    # List of IDs
        metadatas = []    # List of metadata dicts

        for doc in documents:
            # TODO: Extract text, id, and build metadata from each document dict
            pass    # Replace with your code

        # TODO: Call self.collection.add() with the lists you built above
        pass    # Replace with your code

        print(f"  Ingested {len(documents)} documents. Total: {self.collection.count()}")

    def search(self, query, top_k=5, category_filter=None):
        """
        Search for documents similar to the query.

        Parameters:
          query:           plain text search query
          top_k:           how many results to return
          category_filter: if set, only return docs in that category

        Returns:
          list of dicts: [{id, title, category, text, similarity}, ...]
          sorted by similarity DESCENDING

        Hints:
          - Build query_params = {"query_texts": [query], "n_results": top_k}
          - If category_filter: add where={"category": category_filter} to query_params
          - Call self.collection.query(**query_params)
          - Results are in results['documents'][0], results['distances'][0], results['metadatas'][0]
          - similarity = 1.0 - distance

        TODO 1d: Implement this method.
        """
        # TODO: Build query_params dict
        query_params = {}    # Replace with your code

        # TODO: Add category filter to query_params if provided
        pass    # Replace with your code

        # TODO: Run the query
        raw = None    # Replace None with self.collection.query(**query_params)

        # TODO: Build a list of formatted result dicts
        results = []
        # Hint: loop using zip(raw['documents'][0], raw['distances'][0], raw['metadatas'][0])
        pass    # Replace with your loop

        return results

    def display(self, results, show_text=True):
        """
        Print search results in a readable format.

        For each result, show:
          - rank (1-based)
          - similarity score
          - document title
          - category
          - first 80 characters of text (if show_text=True)

        Also show a simple ASCII similarity bar (# characters scaled to similarity).

        TODO 1e: Implement this method.
        """
        if not results:
            print("  No results found.")
            return

        print(f"  {len(results)} result(s):")
        for i, r in enumerate(results, start=1):
            # TODO: Print each result in a readable format
            pass    # Replace with your code

    def count(self):
        """Return number of documents stored."""
        # TODO 1f: Return self.collection.count()
        return None    # Replace with your code

    def get_categories(self):
        """Return a set of all unique categories in the store."""
        # TODO 1g: Get all documents and extract unique category values from metadata
        # Hint: self.collection.get() returns all docs (no arguments)
        # Then loop through results['metadatas'] and collect unique categories
        all_docs   = self.collection.get()          # Get all stored documents
        categories = set()
        # TODO: extract categories from all_docs['metadatas']
        pass    # Replace with your code
        return categories


# ==============================================================================
# TASK 2: Test Your SemanticSearchEngine
# ==============================================================================

print("\n--- Task 2: Test the SemanticSearchEngine ---")

# Create and populate the engine
engine = SemanticSearchEngine()
engine.ingest(FAQ_DOCUMENTS)

print(f"  Engine has {engine.count()} documents")
print(f"  Categories available: {engine.get_categories()}")
print()

# Test searches (these mimic real user questions -- different words from the documents)
test_queries = [
    {
        "query":    "I cannot log in and forgot my credentials",
        "filter":   None,
        "expected": "Should find password reset and account docs"
    },
    {
        "query":    "How do I add someone to my team?",
        "filter":   "collaboration",
        "expected": "Should find team and collaboration docs"
    },
    {
        "query":    "I want to stop paying for the service",
        "filter":   "billing",
        "expected": "Should find cancellation and billing docs"
    },
    {
        "query":    "The app is not working and I cannot connect",
        "filter":   None,
        "expected": "Should find technical troubleshooting docs"
    },
]

all_passed = True

for test in test_queries:
    print(f"  Query: '{test['query']}'")
    if test["filter"]:
        print(f"  Filter: category = '{test['filter']}'")
    print(f"  Expected: {test['expected']}")
    print()

    results = engine.search(test["query"], top_k=3, category_filter=test["filter"])
    engine.display(results)

    # Basic validation
    if not results:
        print(f"  WARNING: No results returned for this query!")
        all_passed = False
    elif results[0]["similarity"] < 0.1:
        print(f"  WARNING: Top result similarity is very low: {results[0]['similarity']}")

    print()

if all_passed:
    print("  All test queries returned results.")
else:
    print("  Some queries had issues -- check your implementation.")


# ==============================================================================
# TASK 3: Implement Search Analytics
# ==============================================================================

print("\n--- Task 3: Search Analytics ---")

print("""
Build a function that analyzes which categories are most relevant
for a given set of queries.

This is useful in production to understand what your users are asking about.
""")

def analyze_search_patterns(queries, engine, top_k=5):
    """
    Run multiple queries and track which categories appear in results.

    Parameters:
      queries: list of query strings
      engine:  SemanticSearchEngine instance
      top_k:   how many results to retrieve per query

    Returns:
      dict: {category -> total_similarity_score}
      This tells us which category is most relevant across all queries.

    Algorithm:
      For each query:
        Search for top_k results (no filter)
        For each result: add its similarity score to that category's total

    TODO 3a: Implement this function.
    """
    category_scores = {}    # Will map category -> total similarity

    for query in queries:
        # TODO: Search for top_k results
        results = None    # Replace with your code

        for r in (results or []):
            cat = r.get("category", "unknown")    # Get category from result
            # TODO: Add r["similarity"] to category_scores[cat]
            # If cat not in category_scores yet, initialize it to 0.0 first
            pass    # Replace with your code

    return category_scores

# Test the analytics
sample_queries = [
    "How do I reset my password?",
    "I forgot my credentials",
    "I cannot log in",
    "Add a colleague to my project",
    "Give someone access to my workspace",
    "Cancel my plan",
    "Stop my subscription",
    "Get my data as a CSV file",
]

analytics = analyze_search_patterns(sample_queries, engine, top_k=3)

print("  Category relevance scores (for 8 sample user queries):")
print("  (Higher = that category's docs appeared more often in results)")
print()

if analytics:
    sorted_cats = sorted(analytics.items(), key=lambda x: x[1], reverse=True)
    max_score = max(analytics.values())

    for category, score in sorted_cats:
        bar_len = int((score / max_score) * 30)    # Scale bar to 30 chars max
        bar     = "#" * bar_len
        print(f"  {category:15s}: {score:6.2f}  [{bar}]")

    top_category = sorted_cats[0][0]
    print(f"\n  Most relevant category: '{top_category}'")
    print("  (This makes sense -- many queries are about account/login issues)")


# ==============================================================================
# TASK 4: Multi-Query Search (Optional Challenge)
# ==============================================================================

print("\n--- Task 4: Multi-Query Search (Challenge) ---")

print("""
OPTIONAL CHALLENGE:
  ChromaDB supports sending MULTIPLE queries in a single call.
  This is more efficient than calling query() once per query.

  collection.query(query_texts=["query1", "query2"], n_results=3)
  Returns results for BOTH queries at once.

  results['documents'][0] = results for query 1
  results['documents'][1] = results for query 2

Implement multi_search() that takes a list of queries and returns results for all.
""")

def multi_search(queries, collection, top_k=3):
    """
    Search for multiple queries in a single ChromaDB call.

    Parameters:
      queries:    list of query strings
      collection: ChromaDB collection
      top_k:      results per query

    Returns:
      list of lists: [[results for q1], [results for q2], ...]
      Each inner list has top_k results sorted by similarity.

    TODO 4: Implement this function.
    """
    # TODO: Call collection.query with query_texts=queries and n_results=top_k
    raw = None    # Replace None with your code

    all_results = []

    # TODO: Loop through each query's results
    # Hint: for i in range(len(queries)):
    #           raw['documents'][i] -> texts for query i
    #           raw['distances'][i] -> distances for query i
    #           raw['metadatas'][i] -> metadatas for query i
    pass    # Replace with your code

    return all_results

# Test (uncomment when ready)
# multi_queries = ["forgot password", "billing and pricing"]
# multi_results = multi_search(multi_queries, engine.collection, top_k=2)
# for i, (q, results) in enumerate(zip(multi_queries, multi_results)):
#     print(f"\n  Query {i+1}: '{q}'")
#     for r in results:
#         print(f"    sim={r['similarity']:.4f}: {r['text'][:60]}...")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("EXERCISE 03 COMPLETE")
print("=" * 60)

print("""
WHAT YOU BUILT:

A complete SemanticSearchEngine with:
  - ingest(documents)         -> store docs in ChromaDB
  - search(query, top_k)      -> find similar documents
  - search with category_filter -> like SQL WHERE clause
  - display(results)          -> formatted output
  - get_categories()          -> list all categories
  - analyze_search_patterns() -> which categories users ask about most

PRODUCTION CHECKLIST:
  [ ] Use PersistentClient (not Client) to save data between runs
  [ ] Add chunking for long documents
  [ ] Add error handling (try/except around collection.add())
  [ ] Log queries for analytics
  [ ] Cache embeddings for frequently asked questions
  [ ] Consider adding a re-ranking step for higher quality

NEXT STEP:
  See projects/document_search_engine/main.py for the full capstone project
  that combines everything from this module.
""")

"""
Example 05: Full Semantic Search System
=========================================

GLOSSARY
--------
Semantic Search:
  Searching by MEANING, not by matching exact words.
  "I cannot get into my account" finds "password reset guide"
  because both mean "login problem" -- not because they share words.

Sentence-Transformers:
  A Python library with pre-trained models that convert full sentences to vectors.
  Much better quality than basic word embeddings.
  all-MiniLM-L6-v2 model: 384 dimensions, fast, good quality, 90MB download.

RAG (Retrieval-Augmented Generation):
  A pattern for AI chatbots:
  Step 1 (R): RETRIEVE relevant documents from a vector database
  Step 2 (A): AUGMENT the LLM prompt with those documents
  Step 3 (G): GENERATE the final answer using an LLM

  Without RAG: LLM answers from training data only (may be outdated or wrong)
  With RAG:    LLM answers using YOUR documents (accurate, citable, current)

Prompt Augmentation:
  Adding retrieved context to the LLM's input prompt.
  The LLM reads the context and uses it to generate an answer.
  This is the "augment" step in RAG.

Recall:
  What fraction of truly relevant documents were returned?
  Recall = (relevant docs found) / (total relevant docs in database)
  High recall = no important results are missed.

Precision:
  What fraction of returned documents are truly relevant?
  Precision = (relevant docs found) / (total docs returned)
  High precision = no irrelevant results are returned.

F1 Score:
  Harmonic mean of precision and recall: 2 * P * R / (P + R)
  Balances precision and recall into a single score.

BM25:
  A classical text search algorithm (used by Elasticsearch, Lucene).
  Based on word frequency (TF-IDF variant). No ML required.
  Often combined with vector search ("hybrid search") for best results.

Hybrid Search:
  Combining keyword search (BM25) with semantic search (vector).
  Better than either alone: keyword handles exact terms, vector handles meaning.

WHAT THIS EXAMPLE SHOWS
------------------------
Part 1: High-quality embeddings with sentence-transformers
Part 2: Comparing semantic search vs keyword search
Part 3: Building a RAG pipeline (retrieve + augment + generate concept)
Part 4: Evaluating search quality (precision, recall)
Part 5: Scaling considerations and production tips

LIBRARIES NEEDED
-----------------
  chromadb              (pip install chromadb)
  sentence-transformers (pip install sentence-transformers)  <- required for this example
  numpy                 (pip install numpy)
  matplotlib            (pip install matplotlib)
"""

import re                        # Text cleaning
import numpy as np               # Numerical operations
import chromadb                  # Vector database

print("=" * 65)
print("EXAMPLE 05: Full Semantic Search System")
print("=" * 65)

# Check for required libraries
try:
    from sentence_transformers import SentenceTransformer    # Real embedding model
    ST_AVAILABLE = True
    print("  sentence-transformers: installed")
except ImportError:
    ST_AVAILABLE = False
    print("  sentence-transformers: NOT installed")
    print("  Install with: pip install sentence-transformers")
    print("  Running with ChromaDB's default embedding model instead.")
print()


# ==============================================================================
# PART 1: High-Quality Embeddings
# ==============================================================================

print("=" * 65)
print("PART 1: High-Quality Embeddings with Sentence-Transformers")
print("=" * 65)

print("""
sentence-transformers provides pre-trained models that convert
entire SENTENCES (not just words) into vectors.

These models understand:
  - Grammar and word order
  - Sentence-level meaning (not just individual words)
  - Synonyms and related concepts
  - Negation (a bit)

Common models:
  all-MiniLM-L6-v2  -> 384 dims, 90MB, fast, good quality
  all-mpnet-base-v2  -> 768 dims, 420MB, slower, better quality
  paraphrase-MiniLM-L6-v2 -> optimized for paraphrase detection

We use all-MiniLM-L6-v2 as it is the best balance of speed and quality.
""")

if ST_AVAILABLE:
    print("  Loading all-MiniLM-L6-v2 model (downloads on first run)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")   # Load the model (~90MB download first time)
    print(f"  Model loaded. Output dimensions: {model.get_sentence_embedding_dimension()}")
    print()

    # Show what the model produces
    test_sentences = [
        "The dog ran quickly across the field",
        "The puppy sprinted fast through the park",
        "I love eating pizza on weekends",
    ]

    # Encode multiple sentences at once (batch) -- much faster than one at a time
    vectors = model.encode(test_sentences)    # Shape: (3, 384)

    print(f"  Encoded {len(test_sentences)} sentences")
    print(f"  Output shape: {vectors.shape}  ({len(test_sentences)} sentences x 384 dimensions)")
    print()

    def cosine_similarity(a, b):
        """Cosine similarity between two vectors."""
        dot   = np.dot(a, b)
        mag_a = np.linalg.norm(a)
        mag_b = np.linalg.norm(b)
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return float(dot / (mag_a * mag_b))

    print("  Semantic similarities:")
    for i in range(len(test_sentences)):
        for j in range(i + 1, len(test_sentences)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"    '{test_sentences[i][:35]}...'")
            print(f"    '{test_sentences[j][:35]}...'")
            print(f"    -> Similarity: {sim:.4f}")
            print()
else:
    print("  (Skipping -- sentence-transformers not installed)")
    print("  ChromaDB's default model will be used for searching.")
    print()


# ==============================================================================
# PART 2: Semantic Search vs Keyword Search
# ==============================================================================

print("=" * 65)
print("PART 2: Semantic Search vs Keyword Search")
print("=" * 65)

print("""
KEYWORD SEARCH:
  Looks for exact word matches (like SQL LIKE or grep).
  Fast and simple. Does not understand synonyms.

SEMANTIC SEARCH:
  Understands meaning. "car" finds "automobile" and "vehicle".
  Slightly slower (need to compute vectors) but much smarter.

Let us compare them side by side.
""")

# Our document collection
documents = [
    {"id": "d01", "text": "How to reset a forgotten password using email verification"},
    {"id": "d02", "text": "Account recovery steps for locked users"},
    {"id": "d03", "text": "Two-factor authentication setup and troubleshooting"},
    {"id": "d04", "text": "Canceling your subscription and requesting a refund"},
    {"id": "d05", "text": "Monthly billing cycle and invoice generation"},
    {"id": "d06", "text": "Updating payment methods and credit card information"},
    {"id": "d07", "text": "Application installation guide for Windows and Mac"},
    {"id": "d08", "text": "Fixing sync errors and connectivity problems"},
    {"id": "d09", "text": "System performance optimization and speed improvements"},
    {"id": "d10", "text": "Database backup and data export procedures"},
]

def keyword_search(query, documents, top_k=3):
    """
    Simple keyword search: count how many query words appear in each document.
    query:     the search query (string)
    documents: list of dicts with 'id' and 'text'
    top_k:     how many results to return
    Returns:   list of (score, doc) tuples sorted by score (descending)
    """
    query_words = set(query.lower().split())       # Split query into words (lowercase)

    results = []
    for doc in documents:
        doc_words = set(doc["text"].lower().split())       # Words in this document
        overlap   = len(query_words & doc_words)           # Count overlapping words
        if overlap > 0:                                    # Only include if any match
            results.append((overlap, doc))

    results.sort(key=lambda x: x[0], reverse=True)        # Sort by overlap count
    return results[:top_k]

# Build semantic search using ChromaDB
client     = chromadb.Client()
col        = client.get_or_create_collection("comparison_demo")

col.add(
    documents=[d["text"] for d in documents],    # Store all texts
    ids=[d["id"] for d in documents]             # With their IDs
)

def semantic_search(query, collection, top_k=3):
    """
    Semantic search using ChromaDB.
    Returns list of (similarity, doc_text) tuples.
    """
    raw = collection.query(query_texts=[query], n_results=top_k)
    results = []
    for i in range(len(raw["documents"][0])):
        text = raw["documents"][0][i]
        dist = raw["distances"][0][i]
        sim  = 1.0 - dist
        results.append((round(sim, 4), text))
    return results

# Compare on several queries
comparison_queries = [
    "I forgot my login credentials",           # Should find password reset docs (no exact word match)
    "I want to stop my monthly payments",      # Should find cancellation/billing (no exact words)
    "The software won't install on my computer",  # Should find installation guide
    "backup",                                  # Simple keyword -- keyword search may do fine
]

for query in comparison_queries:
    print(f"  Query: '{query}'")
    print()

    # Keyword results
    kw_results = keyword_search(query, documents, top_k=2)
    print(f"    KEYWORD SEARCH:")
    if kw_results:
        for score, doc in kw_results:
            print(f"      [{doc['id']}] overlap={score} words: {doc['text']}")
    else:
        print("      No results (no matching words found)")

    # Semantic results
    sem_results = semantic_search(query, col, top_k=2)
    print(f"    SEMANTIC SEARCH:")
    for sim, text in sem_results:
        print(f"      similarity={sim:.4f}: {text}")

    print()

print("  -> Semantic search finds relevant docs even when ZERO query words match")
print("  -> 'I forgot my login' finds 'password reset' -- no word overlap!")
print()


# ==============================================================================
# PART 3: RAG Pipeline Concept
# ==============================================================================

print("=" * 65)
print("PART 3: RAG Pipeline (Retrieve-Augment-Generate)")
print("=" * 65)

print("""
RAG is used by ChatGPT, Claude, and most modern AI assistants.
Here is how it works (we simulate the LLM with a simple template):

  User: "How do I get back into my account if I forgot my password?"
       |
       v
  [RETRIEVE] Search vector database for top 3 relevant documents
       |
       v
  [AUGMENT] Build a prompt with retrieved context:
      "You are a helpful assistant. Use the following context to answer:
       Context 1: How to reset a forgotten password using email verification
       Context 2: Account recovery steps for locked users
       Context 3: Two-factor authentication setup
       Question: How do I get back into my account?"
       |
       v
  [GENERATE] Send to LLM (GPT-4, Claude, etc.) -> Final answer

The LLM sees REAL CURRENT DOCUMENTS and answers based on them.
Result: accurate, citable, up-to-date answers.
""")

# Simulate a RAG pipeline (without a real LLM -- just show the concept)

def retrieve(query, collection, top_k=3):
    """Step R: Retrieve relevant documents from vector database."""
    raw = collection.query(query_texts=[query], n_results=top_k)
    contexts = []
    for i in range(len(raw["documents"][0])):
        text = raw["documents"][0][i]
        dist = raw["distances"][0][i]
        sim  = round(1.0 - dist, 4)
        contexts.append({"text": text, "similarity": sim})
    return contexts

def augment(query, contexts):
    """Step A: Build an augmented prompt with retrieved context."""
    # Format the retrieved context as numbered items
    context_block = ""
    for i, ctx in enumerate(contexts, start=1):
        context_block += f"  Context {i} (similarity {ctx['similarity']:.3f}):\n"
        context_block += f"    {ctx['text']}\n"

    # Build the full prompt (what would be sent to GPT-4/Claude/etc.)
    prompt = f"""You are a helpful customer support assistant.
Use ONLY the context below to answer the question.
If the context does not contain enough information, say "I need more information."

{context_block}
Question: {query}
Answer:"""

    return prompt

def generate_simulated(prompt, query):
    """Step G: Simulate an LLM answer (in real code: call OpenAI/Anthropic API)."""
    # In a real system, you would call:
    # response = openai.ChatCompletion.create(model="gpt-4", messages=[...])
    # OR
    # response = anthropic.messages.create(model="claude-opus-4-7", messages=[...])

    # We simulate a simple answer for demonstration
    return f"[Simulated LLM Answer] Based on the retrieved context, I can help you with: '{query}'. The context suggests checking the relevant documentation. In a real system, GPT-4 or Claude would read the context and write a detailed, helpful answer."

# Run a RAG demo
rag_query = "I cannot log in to my account and I need help"
print(f"  RAG Demo Query: '{rag_query}'")
print()

# Step 1: Retrieve
print("  STEP 1 - RETRIEVE:")
contexts = retrieve(rag_query, col, top_k=3)
for i, ctx in enumerate(contexts, start=1):
    print(f"    [{i}] similarity={ctx['similarity']:.4f}: {ctx['text']}")
print()

# Step 2: Augment
print("  STEP 2 - AUGMENT (build prompt):")
prompt = augment(rag_query, contexts)
print("  " + "\n  ".join(prompt.split("\n")))    # Indent each line for readability
print()

# Step 3: Generate
print("  STEP 3 - GENERATE (simulated):")
answer = generate_simulated(prompt, rag_query)
print(f"  {answer}")
print()

print("""
  In production:
  - Replace generate_simulated() with an actual LLM API call
  - openai.ChatCompletion.create(model="gpt-4", messages=[{"role":"user","content":prompt}])
  - Or: anthropic.messages.create(model="claude-opus-4-7", messages=[...])
  The LLM reads the retrieved context and gives a real, detailed answer.
""")


# ==============================================================================
# PART 4: Evaluating Search Quality
# ==============================================================================

print("=" * 65)
print("PART 4: Evaluating Search Quality")
print("=" * 65)

print("""
How do we know if our search is good?
We use a test set: queries with known "correct" documents.

Metrics:
  Precision@K: Of the K results returned, what fraction are relevant?
  Recall@K:    Of ALL relevant docs, what fraction appear in the top K?
  MRR (Mean Reciprocal Rank): Average of 1/rank-of-first-correct-result
""")

# Test set: each query has a list of "correct" document IDs
test_set = [
    {
        "query":    "I forgot my password",
        "relevant": {"d01", "d02"}             # Correct docs for this query
    },
    {
        "query":    "I want to stop my subscription",
        "relevant": {"d04", "d05"}
    },
    {
        "query":    "App is slow on my computer",
        "relevant": {"d07", "d08", "d09"}
    },
]

def evaluate_search(test_set, collection, k=3):
    """
    Evaluate search quality using precision and recall at K.
    test_set:   list of {query, relevant} dicts
    collection: ChromaDB collection to search
    k:          how many results to retrieve per query
    Returns:    dict of average metrics
    """
    precisions = []     # Precision@K for each query
    recalls    = []     # Recall@K for each query
    rrs        = []     # Reciprocal rank for each query

    for test in test_set:
        query    = test["query"]
        relevant = test["relevant"]             # Set of correct doc IDs

        raw = collection.query(query_texts=[query], n_results=k)
        returned_ids = set(raw["ids"][0])       # Set of returned doc IDs

        # Count how many returned docs are in the relevant set
        tp = len(returned_ids & relevant)       # True positives (correct and returned)

        precision = tp / k                      # Precision@K
        recall    = tp / len(relevant)          # Recall@K

        # Reciprocal Rank: 1/rank_of_first_relevant_result
        rr = 0.0
        for rank, doc_id in enumerate(raw["ids"][0], start=1):
            if doc_id in relevant:
                rr = 1.0 / rank                 # Found relevant doc at this rank
                break                           # Only care about the FIRST relevant result

        precisions.append(precision)
        recalls.append(recall)
        rrs.append(rr)

    return {
        "precision_at_k": round(np.mean(precisions), 4),
        "recall_at_k":    round(np.mean(recalls),    4),
        "mrr":            round(np.mean(rrs),         4),
    }

metrics = evaluate_search(test_set, col, k=3)
print(f"  Semantic Search Evaluation (k=3):")
print(f"    Precision@3: {metrics['precision_at_k']:.4f}  (fraction of results that are correct)")
print(f"    Recall@3:    {metrics['recall_at_k']:.4f}  (fraction of correct docs found)")
print(f"    MRR:         {metrics['mrr']:.4f}  (1/rank of first correct result, higher=better)")
print()
print("  Compare to keyword search:")

# Evaluate keyword search for comparison
kw_precisions = []
kw_recalls    = []

for test in test_set:
    query    = test["query"]
    relevant = test["relevant"]
    kw_results = keyword_search(query, documents, top_k=3)
    returned_kw_ids = set(doc["id"] for _, doc in kw_results)
    tp = len(returned_kw_ids & relevant)
    kw_precisions.append(tp / 3)
    kw_recalls.append(tp / len(relevant))

print(f"    Keyword Precision@3: {np.mean(kw_precisions):.4f}")
print(f"    Keyword Recall@3:    {np.mean(kw_recalls):.4f}")
print()
print("  -> Semantic search has higher recall (finds relevant docs keyword search misses)")
print()


# ==============================================================================
# PART 5: Production Tips
# ==============================================================================

print("=" * 65)
print("PART 5: Production Tips and Scaling")
print("=" * 65)

print("""
TIPS FOR PRODUCTION USE:

1. CHUNKING STRATEGY:
   - Chunk size: 200-500 words (adjust based on your embedding model's token limit)
   - Overlap: 10-20% of chunk size (50-100 words)
   - Sentence-aware chunking: split at sentence boundaries, not mid-sentence
   - Store original document ID in metadata to re-fetch the full doc if needed

2. EMBEDDING MODEL CHOICE:
   - all-MiniLM-L6-v2: start here (384 dims, fast, good quality)
   - all-mpnet-base-v2: better quality, slower (768 dims)
   - OpenAI text-embedding-ada-002: best quality, cloud API, costs money
   - For code: codellama or voyage-code-2

3. CHROMADB SCALING:
   - <100k docs: chromadb.Client() or PersistentClient
   - >100k docs: consider Pinecone (cloud, free tier), Weaviate (self-hosted)
   - For .NET teams: pgvector on PostgreSQL (familiar SQL + vector search)

4. RE-RANKING:
   - First pass: retrieve top 20 with vector search (fast)
   - Second pass: re-rank top 20 with a cross-encoder model (slower but better)
   - Result: top 5 that are truly the most relevant

5. HYBRID SEARCH:
   - Combine BM25 (keyword) with vector search for best results
   - BM25 catches exact term matches (product codes, names)
   - Vector catches semantic matches (synonyms, concepts)

6. CACHING:
   - Cache embeddings of common queries (same query = same vector)
   - Cache search results for popular queries (30-minute cache)
   - This can reduce embedding cost by 80% in production

7. MONITORING:
   - Log queries + which results were clicked (for improving quality)
   - Track latency (aim for <200ms per query)
   - Monitor which queries return NO results (add those docs to knowledge base)

COST ESTIMATE (rough):
  ChromaDB PersistentClient:  Free (runs on your server)
  Sentence-transformers:      Free (runs locally, CPU or GPU)
  Pinecone (managed):         Free up to 100k vectors, then ~$70/month per million
  OpenAI embeddings:          $0.0001 per 1000 tokens
""")

print("  Summary of C#/.NET to Python equivalents in this module:")
print()
print("  C# / SQL Server                    Python / ChromaDB")
print("  ----------------------------        ----------------------------------")
print("  SqlConnection                  ->   chromadb.Client()")
print("  CREATE TABLE                   ->   client.create_collection()")
print("  INSERT INTO                    ->   collection.add()")
print("  SELECT ... WHERE ... LIKE      ->   collection.query() [semantic]")
print("  WHERE clause                   ->   where={...} in query()")
print("  DELETE WHERE id = x            ->   collection.delete(ids=['x'])")
print("  UPDATE ... WHERE id = x        ->   collection.upsert()")
print("  SELECT COUNT(*)                ->   collection.count()")
print("  float[][]  (2D array)          ->   numpy array shape (n, dim)")
print("  Dictionary<string, float[]>    ->   ChromaDB collection")
print()


# ==============================================================================
# SUMMARY
# ==============================================================================

print("=" * 65)
print("SUMMARY - Full Semantic Search System")
print("=" * 65)

print("""
WHAT WE COVERED IN THIS EXAMPLE:

1. sentence-transformers: model.encode("text") -> numpy array (384 dims)
   Much better quality than basic embeddings.

2. Semantic vs Keyword search:
   Keyword: exact word match. Fast. Misses synonyms.
   Semantic: meaning match. Slightly slower. Finds related concepts.

3. RAG Pipeline: Retrieve -> Augment -> Generate
   Retrieve: vector search to find relevant docs
   Augment:  build LLM prompt with retrieved context
   Generate: LLM produces accurate answer based on context

4. Evaluation: Precision@K, Recall@K, MRR
   Measure search quality with a labeled test set.

5. Production tips:
   - Chunking: 200-500 words, 10-20% overlap
   - Start with ChromaDB, scale to Pinecone/pgvector as needed
   - Hybrid search (BM25 + vector) for best quality

MODULE COMPLETE.
Next: Projects folder -> build the full Document Search Engine project.
""")

print("=" * 65)
print("END OF EXAMPLE 05")
print("=" * 65)

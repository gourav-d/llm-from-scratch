# Lesson 05: Real-World Applications

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain how ChatGPT Plus memory works using vector databases (RAG)
2. Describe how GitHub Copilot finds relevant code examples
3. Compare ChromaDB to production vector databases (Pinecone, pgvector)
4. Identify which type of project would benefit from a vector database

---

## GLOSSARY

```
RAG (Retrieval-Augmented Generation):
  A pattern where you first RETRIEVE relevant documents from a vector database,
  then PASS those documents to an LLM as context for its answer.
  Used in: ChatGPT memory, customer support bots, document Q&A systems.
  
  Without RAG: LLM answers from training data only (may be outdated or wrong)
  With RAG:    LLM answers using CURRENT documents you provide (accurate)

Context Window:
  The maximum text an LLM can process at once.
  GPT-4: ~128,000 tokens (about 100,000 words)
  The context window is limited -- you cannot send ALL your documents to the LLM.
  RAG solves this by retrieving only the MOST RELEVANT few documents.

Grounding:
  Making sure an LLM's answer is based on real, provided documents.
  A "grounded" answer cites its sources (the retrieved documents).
  Reduces hallucination (LLM making up facts).

Hallucination:
  When an LLM confidently states something that is false.
  RAG reduces hallucination by giving the LLM accurate context to work from.

HNSW (Hierarchical Navigable Small World):
  The indexing algorithm used by most vector databases.
  Allows approximate nearest neighbor search in milliseconds over millions of vectors.
  You do not need to understand this -- it is an internal detail of the vector DB.

Approximate Nearest Neighbor (ANN):
  Finding APPROXIMATELY the nearest vectors (not guaranteed to be exact).
  Much faster than exact nearest neighbor search.
  In practice, ANN finds the true nearest neighbor 95-99% of the time.

Pinecone:
  A fully managed, cloud-based vector database.
  No server to run -- just an API call (like Azure SQL vs self-managed SQL Server).
  Handles billions of vectors. Costs money (free tier available).

pgvector:
  A PostgreSQL extension that adds vector storage and search to PostgreSQL.
  If you already use PostgreSQL, you can add vector search without a new database.
  In C# terms: like adding a NuGet package to an existing project.

Weaviate:
  An open-source vector database with built-in text embedding.
  More features than ChromaDB (GraphQL API, multi-tenancy, more indexing options).
```

---

## Part 1: RAG -- How ChatGPT Memory Works

ChatGPT Plus has a "memory" feature. Here is how it works internally:

### Step 1: Writing to Memory

When you tell ChatGPT something important:
```
You: "My name is Gourav. I work at Blackline as a .NET developer."
```

ChatGPT extracts key facts and converts them to vectors:
```
"User's name is Gourav"               -> [0.23, -0.14, 0.88, ...]
"User works at Blackline"             -> [0.45, 0.33, -0.12, ...]
"User is a .NET developer"            -> [0.67, 0.21, 0.44, ...]
```

These vectors are stored in a vector database.

### Step 2: Reading from Memory

On your next conversation, before generating a reply:
```
New message: "Can you help me with a C# problem?"
-> Embed this message: [0.66, 0.19, 0.42, ...]
-> Search vector DB for similar past memories
-> Finds: "User is a .NET developer" (very similar vector)
-> Injects into LLM context: "Memory: User is a .NET developer"
```

### Step 3: LLM Generates Answer

The LLM now has the context:
```
[System]: You are a helpful assistant.
[Memory]: User is a .NET developer.
[User]:   Can you help me with a C# problem?
```

And can give a personalized answer that references their expertise.

### The RAG Pattern in Code

```python
def ask_with_rag(question, memory_collection, llm_client):
    """
    RAG = Retrieve relevant context, then Generate an answer with an LLM.
    """
    # Step 1: RETRIEVE - find relevant context from vector database
    results = memory_collection.query(
        query_texts=[question],
        n_results=3                     # Get top 3 most relevant memories
    )
    relevant_context = results["documents"][0]   # The retrieved text chunks

    # Step 2: BUILD PROMPT - combine context + question
    context_text = "\n".join(relevant_context)   # Join the retrieved chunks
    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context_text}

Question: {question}
Answer:"""

    # Step 3: GENERATE - send to LLM
    response = llm_client.complete(prompt)
    return response.text
```

---

## Part 2: GitHub Copilot -- Finding Relevant Code

GitHub Copilot uses vector search to find relevant code examples.

### How It Works

When you type code in VS Code, Copilot:

1. Looks at your current file and neighboring files
2. Converts code snippets to vectors using a code embedding model
3. Searches a vector database of code patterns (trained on GitHub)
4. Retrieves similar code patterns as context
5. Passes context + your current code to the LLM for completion

### Code Embeddings Are Different from Text Embeddings

Code has different structure than text:
```
# Text: "The dog ran fast" -> general meaning
# Code: "for i in range(len(list)):" -> specific programming pattern

Text embedding models:  all-MiniLM-L6-v2 (for natural language)
Code embedding models:  code-bert, codellama (for source code)
```

Code embeddings capture:
  - Function signatures
  - Variable naming patterns
  - Control flow patterns (loops, conditionals)
  - API usage patterns

---

## Part 3: Customer Support Bot

A customer support bot uses RAG to answer from a knowledge base:

```
Architecture:

[Knowledge Base]
   Support articles, FAQs, product documentation
         |
         v (one-time: embed all documents)
   Vector Database (ChromaDB / Pinecone)
         |
         v (on each user question)
[User Question] -> embed -> search vector DB -> retrieve top 3 articles
                                                        |
                                                        v
                                              LLM (GPT-4, Claude)
                                                        |
                                                        v
                                              Answer citing sources
```

The bot can answer questions even if the exact wording differs from
the support articles, because vector search finds by MEANING.

---

## Part 4: Comparing Vector Databases

```
Database       | Type          | Scale          | Best For
---------------+---------------+----------------+----------------------------------
ChromaDB       | Local/Python  | Up to 1M docs  | Learning, prototypes, small apps
Pinecone       | Managed cloud | Billions       | Production SaaS, no ops overhead
pgvector       | PostgreSQL    | Millions       | Teams already using PostgreSQL
Weaviate       | Open-source   | Billions       | Complex schema, GraphQL queries
Milvus         | Open-source   | Billions       | High-performance production
Qdrant         | Open-source   | Billions       | Good documentation, easy REST API
Azure AI Search| Managed cloud | Millions       | Azure ecosystem, .NET integration
```

### When to Graduate from ChromaDB

Stay with ChromaDB when:
  - Learning / prototyping
  - Documents < 500,000
  - Single server or single user
  - No latency SLA requirements

Move to Pinecone/Weaviate when:
  - Documents > 1 million
  - Multiple servers need the same vector DB
  - Sub-10ms query latency required
  - 99.9% uptime SLA required

### For .NET Developers

If your company uses SQL Server or PostgreSQL, the easiest path is:
  - Use pgvector extension on your existing PostgreSQL database
  - Query vectors with SQL + the <-> distance operator
  - No new infrastructure to manage

```sql
-- pgvector SQL example
-- Store a vector
INSERT INTO documents (text, embedding)
VALUES ('Hello world', '[0.1, 0.2, 0.3, ...]'::vector);

-- Find similar documents
SELECT text, 1 - (embedding <=> query_vector) as similarity
FROM documents
ORDER BY embedding <=> query_vector
LIMIT 5;
```

---

## Part 5: Real Project Ideas

### Beginner Projects

1. Personal Notes Search
   - Store all your Markdown notes in ChromaDB
   - Search them by meaning (not just grep)
   - "Find my notes about API design" finds relevant notes even if they say "REST endpoints"

2. Job Description Matcher
   - Store job descriptions as vectors
   - Query with your CV / skills
   - Find the jobs that best match your profile

3. Recipe Finder
   - Store recipes with their ingredients and descriptions
   - Query: "I want something healthy with chicken and lemon"
   - Finds matching recipes by semantic meaning

### Intermediate Projects

4. Company Knowledge Base Bot
   - Ingest company wikis, Confluence pages, Slack messages
   - Allow employees to search by question
   - Combine with an LLM for conversational answers (RAG)

5. Code Search
   - Embed all functions in a large codebase
   - Search: "find functions that handle user authentication"
   - Returns relevant functions even if they are not named "auth"

6. Product Recommendation Engine
   - Embed product descriptions
   - When a user views a product, find the most similar ones
   - Show "You might also like..." recommendations

### Advanced Projects

7. Multi-Document Q&A (Full RAG)
   - Ingest PDFs, Word documents, web pages
   - User asks any question
   - System retrieves relevant passages and uses an LLM to answer

8. Semantic Duplicate Detection
   - Find duplicate or near-duplicate support tickets
   - Find duplicate customer reviews
   - Flag near-duplicate code files in a large repo

---

## Part 6: The Full LLM Application Stack

With this module, you now understand the full modern LLM application stack:

```
Layer 6: Application (FastAPI, .NET API)          <- Module 09
          |
Layer 5: Orchestration (LangChain, LlamaIndex)   <- ties layers together
          |
Layer 4: LLM (GPT-4, Claude, Llama)              <- generates final answer
          |
Layer 3: Vector Database (ChromaDB, Pinecone)     <- this module (10)
          |
Layer 2: Embedding Model (sentence-transformers)  <- this module (10)
          |
Layer 1: Data (documents, databases, APIs)        <- raw source material
```

You have now studied:
  - How transformers and attention work (Module 04)
  - How to build an LLM from scratch (Module 05)
  - How to train and fine-tune (Module 06)
  - How reasoning models work (Module 07)
  - Prompt engineering (Module 08)
  - Production deployment (Module 09)
  - Vector databases and semantic search (Module 10 -- this module)

---

## Key Takeaways

1. RAG = Retrieve relevant docs from vector DB, then Generate an LLM answer.
   Solves: LLM context limits, hallucination, outdated training data.

2. GitHub Copilot and ChatGPT memory both use vector search internally.

3. ChromaDB is great for learning and small projects (up to ~1M documents).
   For production at scale, use Pinecone, pgvector, or Weaviate.

4. For .NET developers: pgvector adds vector search to your existing PostgreSQL.

5. Vector databases are now a fundamental component of modern AI applications.
   Every LLM application you build will likely need one.

---

## What to Build Next

After completing this module, consider:
  - Module 11: LLM Agents (autonomous AI that uses tools, including vector search)
  - Module 12: Multimodal LLMs (images + text embeddings)
  - Build the capstone project: Full RAG Document Q&A System

See the project in `projects/document_search_engine/main.py` to apply everything from this module.

"""
retriever.py -- Phase 2: Retrieve relevant chunks and generate an answer.

WHAT THIS FILE DOES:
---------------------
Given a user's question:
1. Embed the question into a vector
2. Search ChromaDB for the top-K most similar chunks
3. Build a prompt: "Here is relevant code... Answer this question: ..."
4. Send the prompt to the local Ollama LLM
5. Return the LLM's answer + the source files it used

THIS IS THE RAG PATTERN:
-------------------------
RAG = Retrieval-Augmented Generation

Without RAG:
  Question -> LLM -> Answer (LLM has no knowledge of YOUR code)

With RAG:
  Question -> retrieve relevant chunks -> LLM -> Answer (LLM reads your actual code)

Think of it like open-book vs closed-book exam:
  - Closed-book (no RAG): LLM answers from memory alone
  - Open-book (RAG): LLM gets relevant pages from the book first

C# analogy:
  This is like a CQRS query handler:
    1. Query repository (ChromaDB) for relevant context
    2. Build a command request (prompt)
    3. Dispatch to a service (Ollama LLM)
    4. Return a response DTO
"""

import requests             # HTTP client
from typing import List, Dict, Any
import chromadb

import config
from utils.embedder import get_embedding, check_ollama_running, check_model_available


def get_collection() -> chromadb.Collection:
    """
    Connect to ChromaDB and return the collection.

    This is our "database connection" -- called at the start of each query.

    Returns:
        ChromaDB collection object

    Raises:
        RuntimeError if the collection doesn't exist (indexer not run yet)
    """
    # Connect to the persistent ChromaDB on disk
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    # Check if our collection exists
    existing = [c.name for c in client.list_collections()]
    if config.COLLECTION_NAME not in existing:
        raise RuntimeError(
            f"Collection '{config.COLLECTION_NAME}' not found in ChromaDB.\n"
            "Run 'python indexer.py' first to index your codebase."
        )

    return client.get_collection(config.COLLECTION_NAME)


def retrieve_chunks(question: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Find the most relevant chunks for a given question.

    Parameters:
        question : the user's question in plain English
        top_k    : how many chunks to retrieve (default from config)

    Returns:
        List of dicts, each with:
            - "text"     : the chunk content
            - "source"   : the file path this came from
            - "distance" : similarity score (lower = more similar)

    HOW SIMILARITY SEARCH WORKS:
    -----------------------------
    1. We convert the question to a vector: [0.12, -0.34, ...]
    2. ChromaDB compares it to every stored vector using cosine distance
    3. Returns the K vectors that are closest (most similar direction)
    4. We get back the original text + metadata for those vectors

    C# analogy:
        var results = await vectorDb.SearchAsync(
            collection: "codebase",
            queryVector: embedder.Embed(question),
            topK: 5
        );
    """
    if top_k is None:
        top_k = config.TOP_K_RESULTS

    # Get the collection from ChromaDB
    collection = get_collection()

    # Check if the collection has any documents
    if collection.count() == 0:
        raise RuntimeError(
            "The ChromaDB collection is empty.\n"
            "Run 'python indexer.py' to index your codebase first."
        )

    # Step 1: Convert the question to a vector
    question_embedding = get_embedding(question)

    # Step 2: Query ChromaDB for the most similar chunks
    # query_embeddings: the vector to search for
    # n_results: how many results to return
    # include: what extra data to include in the response
    results = collection.query(
        query_embeddings=[question_embedding],   # list of query vectors
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],  # what to return
    )

    # Step 3: Unpack the results into a cleaner list of dicts
    # ChromaDB returns nested lists because you can query multiple vectors at once.
    # We queried only one vector, so we take index [0] from each list.
    chunks = []
    documents = results["documents"][0]     # list of chunk texts
    metadatas = results["metadatas"][0]     # list of metadata dicts
    distances = results["distances"][0]     # list of distance scores

    for text, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "text": text,
            "source": meta.get("source", "unknown"),
            "chunk_idx": meta.get("chunk_idx", 0),
            "distance": dist,
        })

    return chunks


def build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Build the prompt to send to the LLM.

    This is the most important part of RAG: how we pack the retrieved
    context into the LLM's input.

    Structure:
        System instruction
        ---
        Source file 1:
        [chunk text]
        ---
        Source file 2:
        [chunk text]
        ---
        Question: ...
        Answer:

    WHY THIS FORMAT?
    ----------------
    - The LLM reads the context BEFORE the question
    - Putting source file names helps the LLM cite sources in its answer
    - "Answer:" at the end prompts the LLM to continue with an answer
    - We limit total context to MAX_CONTEXT_CHARS to avoid token limit errors

    C# analogy: string interpolation building a request body.
    """
    # Start with the system instruction
    lines = [
        "You are a helpful assistant that answers questions about a codebase.",
        "Use ONLY the provided context below to answer the question.",
        "If the answer is not in the context, say 'I could not find that in the indexed code.'",
        "Always mention which file the answer comes from.",
        "",
        "CONTEXT:",
        "=" * 50,
    ]

    # Add each retrieved chunk with its source file
    total_chars = 0
    for chunk in chunks:
        # Get a short relative path for display (avoid very long absolute paths)
        source = chunk["source"]
        try:
            # Try to make it relative to the repo root for cleaner display
            source = source.replace(config.REPO_PATH, "").lstrip("/\\")
        except Exception:
            pass

        chunk_header = f"\nFile: {source}\n{'-' * 40}"
        chunk_body = chunk["text"]

        # Check if adding this chunk would exceed our context limit
        addition = chunk_header + "\n" + chunk_body + "\n"
        if total_chars + len(addition) > config.MAX_CONTEXT_CHARS:
            # Stop adding chunks if we'd exceed the limit
            lines.append("\n[Additional context truncated to fit token limit]")
            break

        lines.append(chunk_header)
        lines.append(chunk_body)
        total_chars += len(addition)

    # Add the question at the end
    lines.extend([
        "",
        "=" * 50,
        f"QUESTION: {question}",
        "",
        "ANSWER:",
    ])

    # Join all lines into one string
    return "\n".join(lines)


def call_ollama_llm(prompt: str) -> str:
    """
    Send a prompt to the Ollama LLM and return its response.

    Ollama API for text generation:
        POST http://localhost:11434/api/generate
        {
            "model": "mistral",
            "prompt": "...",
            "stream": false
        }

    Response:
        {"response": "Here is the answer...", "done": true, ...}

    C# analogy:
        var result = await httpClient.PostAsJsonAsync("/api/generate", request);
        var response = await result.Content.ReadFromJsonAsync<OllamaResponse>();
        return response.Response;
    """
    url = f"{config.OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,        # False = wait for full response, then return it
                                # True = stream tokens as they're generated (for UI)
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        response.raise_for_status()

        # Parse the JSON and return the "response" field
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Is it running?\n"
            "Start Ollama, then try again."
        )
    except requests.exceptions.HTTPError as e:
        if "404" in str(e):
            raise RuntimeError(
                f"LLM model '{config.LLM_MODEL}' not found in Ollama.\n"
                f"Pull it with: ollama pull {config.LLM_MODEL}"
            )
        raise


def answer_question(question: str) -> Dict[str, Any]:
    """
    Full RAG pipeline: question -> answer + sources.

    This is the main function that ties everything together:
        1. Retrieve relevant chunks
        2. Build the prompt
        3. Call the LLM
        4. Return answer + source files used

    Parameters:
        question : the user's plain-English question

    Returns:
        Dict with:
            - "answer"  : the LLM's response text
            - "sources" : list of unique source file paths used
            - "chunks"  : the raw retrieved chunks (for debugging)

    C# analogy:
        public async Task<QueryResult> AnswerQuestionAsync(string question)
    """
    # Step 1: Find relevant chunks
    chunks = retrieve_chunks(question)

    # Step 2: Build the prompt
    prompt = build_prompt(question, chunks)

    # Step 3: Call the LLM
    answer = call_ollama_llm(prompt)

    # Step 4: Collect unique source files (deduplicated)
    # In C#: chunks.Select(c => c.Source).Distinct().ToList()
    sources = list({chunk["source"] for chunk in chunks})

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
    }

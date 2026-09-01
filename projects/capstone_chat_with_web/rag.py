"""
rag.py -- Phase 3: Retrieval-Augmented Generation (RAG) pipeline.

HOW RAG WORKS:
  Traditional LLM: User question --> LLM memory --> Answer
                   Problem: LLM doesn't know YOUR web pages

  RAG:             User question --> Search vector DB --> Relevant chunks
                                --> Build prompt (question + chunks) --> LLM --> Answer
                   Benefit: LLM answers FROM your actual content

C# analogy: like dependency injection for knowledge --
  instead of hardcoding facts in the model,
  we inject the relevant facts at runtime via the prompt.

The LLM here is Ollama running locally (no internet, no API key).
Install: https://ollama.ai  then run: ollama pull mistral
"""

import json
import urllib.request

import store
import config


# ---------------------------------------------------------------------------
# Ollama helper
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> str:
    """
    Send a prompt to Ollama's local REST API and return the response.

    Ollama API: POST http://localhost:11434/api/generate
    Request:  { "model": "mistral", "prompt": "...", "stream": false }
    Response: { "response": "...", ... }

    C# analogy: like HttpClient.PostAsJsonAsync() + ReadFromJsonAsync()
    """
    payload = json.dumps({
        "model":  config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,  # get full response at once (not token by token)
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.OLLAMA_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except Exception as e:
        return f"ERROR: Could not reach Ollama. Is it running? ({e})\nRun: ollama serve"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_qa_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build a prompt that includes retrieved context + the user's question.

    This is the core of RAG -- the LLM sees:
      "Here is some context: [your web content]
       Answer this question: [user's question]"

    The LLM uses the context, not its training memory, to answer.
    """
    # Build context block -- cap at MAX_CONTEXT_CHARS to avoid token overflow
    context_parts = []
    total_chars = 0

    for chunk in context_chunks:
        chunk_text = f"[Source: {chunk['source_url']}]\n{chunk['text']}"
        if total_chars + len(chunk_text) > config.MAX_CONTEXT_CHARS:
            break
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    context = "\n\n---\n\n".join(context_parts)

    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't have enough information to answer this."
Do not make up information. Be concise and accurate.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def _build_summary_prompt(context_chunks: list[dict]) -> str:
    """Build a prompt to summarize all indexed content."""
    context_parts = []
    total_chars = 0

    # For summary we use all chunks from all sources
    seen_urls = set()
    for chunk in context_chunks:
        url = chunk["source_url"]
        if url not in seen_urls:
            seen_urls.add(url)
            context_parts.append(f"[Source: {url}]")

        chunk_text = chunk["text"]
        if total_chars + len(chunk_text) > config.MAX_CONTEXT_CHARS:
            break
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    context = "\n\n".join(context_parts)

    return f"""You are a helpful assistant. Summarize the following web content clearly.
Include: main topics, key points, and any important facts.
Be thorough but concise.

CONTENT:
{context}

SUMMARY:"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer(question: str) -> dict:
    """
    Answer a question using RAG.

    Returns:
        {
            "answer":  "The answer text...",
            "sources": ["https://url1.com", "https://url2.com"],
            "chunks_used": 3
        }
    """
    print(f"\nSearching knowledge base for: '{question}'")

    # Step 1: Retrieve relevant chunks
    chunks = store.search(question, top_k=config.TOP_K_RESULTS)

    if not chunks:
        return {
            "answer": "No content indexed yet. Use 'add' command first to index some URLs.",
            "sources": [],
            "chunks_used": 0,
        }

    print(f"  Retrieved {len(chunks)} relevant chunks")
    for c in chunks:
        print(f"    score={c['score']:.3f}  {c['source_url'][:60]}")

    # Step 2: Build prompt with context
    prompt = _build_qa_prompt(question, chunks)

    # Step 3: Call LLM
    print(f"  Calling {config.LLM_MODEL} via Ollama...")
    response = _call_ollama(prompt)

    # Collect unique source URLs
    sources = list(dict.fromkeys(c["source_url"] for c in chunks))

    return {
        "answer":      response,
        "sources":     sources,
        "chunks_used": len(chunks),
    }


def summarize() -> dict:
    """
    Summarize all indexed web content.

    Uses a broad search with a generic query to pull diverse chunks,
    then asks the LLM to summarize everything.
    """
    print("\nSummarizing all indexed content...")

    # Search with a broad query to get diverse chunks from all sources
    chunks = store.search("main topic overview summary key points", top_k=20)

    if not chunks:
        return {
            "summary": "No content indexed yet. Use 'add' command first.",
            "sources": [],
        }

    prompt = _build_summary_prompt(chunks)

    print(f"  Calling {config.LLM_MODEL} via Ollama...")
    summary = _call_ollama(prompt)

    sources = list(dict.fromkeys(c["source_url"] for c in chunks))

    return {
        "summary": summary,
        "sources": sources,
    }


def is_ollama_running() -> bool:
    """Check if Ollama server is reachable."""
    try:
        urllib.request.urlopen(
            f"{config.OLLAMA_BASE_URL}/api/tags",
            timeout=3
        )
        return True
    except Exception:
        return False

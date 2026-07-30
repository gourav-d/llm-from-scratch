"""
embedder.py -- Convert text into embedding vectors using Ollama.

WHAT IS AN EMBEDDING?
----------------------
An embedding is a list of numbers (a vector) that represents the MEANING of text.
Similar texts get similar vectors. This is how we do "semantic search" --
finding text by meaning rather than by exact keyword match.

Example:
  "How do I handle errors?"  -->  [0.12, -0.34, 0.89, ...]  (384 numbers)
  "Exception handling guide" -->  [0.11, -0.33, 0.91, ...]  (very similar!)
  "Pizza recipe"             -->  [-0.77, 0.22, -0.45, ...]  (very different)

HOW DOES OLLAMA WORK?
----------------------
Ollama runs as a local HTTP server on port 11434.
We send a POST request with our text, it returns the vector.

C# analogy: like calling a local Web API:
  var response = await httpClient.PostAsJsonAsync(
      "http://localhost:11434/api/embeddings",
      new { model = "nomic-embed-text", prompt = text }
  );
  var result = await response.Content.ReadFromJsonAsync<EmbeddingResponse>();

This file wraps those HTTP calls in clean Python functions.
"""

import requests             # HTTP client (like HttpClient in C#)
import time                 # for retry delays
from typing import List     # for type hints

import sys
import os
# Add parent directory to path so we can import config.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config               # our settings file


def get_embedding(text: str, retries: int = 3) -> List[float]:
    """
    Convert a text string into an embedding vector.

    Parameters:
        text    : the text to embed (a chunk or a question)
        retries : how many times to retry if Ollama is slow to start

    Returns:
        A list of floats -- the embedding vector.
        Example: [0.123, -0.456, 0.789, ...]  (usually 768 or 384 numbers)

    Raises:
        RuntimeError if Ollama is not running or model not pulled.

    C# analogy:
        public async Task<float[]> GetEmbeddingAsync(string text)
    """
    # The API endpoint for embeddings in Ollama
    url = f"{config.OLLAMA_BASE_URL}/api/embeddings"

    # The request body -- what we send to Ollama
    # "model": which embedding model to use
    # "prompt": the text to embed
    payload = {
        "model": config.EMBEDDING_MODEL,
        "prompt": text,
    }

    # Retry loop -- Ollama might be warming up or busy
    for attempt in range(retries):
        try:
            # Send POST request to Ollama
            # timeout= prevents hanging forever if Ollama crashes
            response = requests.post(
                url,
                json=payload,                      # sends as JSON body
                timeout=config.OLLAMA_TIMEOUT,     # seconds before giving up
            )

            # Raise an exception if HTTP status code is 4xx or 5xx
            # Like checking response.IsSuccessStatusCode in C#
            response.raise_for_status()

            # Parse the JSON response and extract the "embedding" field
            # The response looks like: {"embedding": [0.1, 0.2, ...]}
            data = response.json()
            return data["embedding"]

        except requests.exceptions.ConnectionError:
            # Ollama is not running -- give a helpful error message
            if attempt == retries - 1:
                raise RuntimeError(
                    "Cannot connect to Ollama.\n"
                    "Make sure Ollama is running: start the Ollama app or run 'ollama serve'\n"
                    f"Expected at: {config.OLLAMA_BASE_URL}"
                )
            # Wait before retrying
            time.sleep(2)

        except requests.exceptions.HTTPError as e:
            # HTTP error -- model probably not pulled
            if "404" in str(e) or "not found" in str(response.text).lower():
                raise RuntimeError(
                    f"Embedding model '{config.EMBEDDING_MODEL}' not found in Ollama.\n"
                    f"Pull it with: ollama pull {config.EMBEDDING_MODEL}"
                )
            raise

    # Should never reach here, but satisfies type checker
    return []


def get_embeddings_batch(texts: List[str], show_progress: bool = False) -> List[List[float]]:
    """
    Convert a list of texts into a list of embedding vectors.

    This is just a loop over get_embedding() -- Ollama does not support
    true batching for embeddings, so we call it once per text.

    Parameters:
        texts         : list of strings to embed
        show_progress : if True, print a dot for every 10 texts

    Returns:
        List of embedding vectors (one per input text)

    C# analogy:
        public async Task<float[][]> GetEmbeddingsBatchAsync(string[] texts)
    """
    embeddings = []     # will hold one vector per text

    for i, text in enumerate(texts):
        # Get the embedding for this single text
        embedding = get_embedding(text)
        embeddings.append(embedding)

        # Print progress every 10 items (useful for large repos)
        if show_progress and (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{len(texts)} chunks...", flush=True)

    return embeddings


def check_ollama_running() -> bool:
    """
    Check if Ollama is running and reachable.
    Returns True if OK, False if not.

    Useful to call before starting indexing so we fail fast with a clear message.

    C# analogy:
        public bool IsServiceHealthy() { ... }  // health check
    """
    try:
        # Ollama's root endpoint returns a simple "Ollama is running" text
        response = requests.get(config.OLLAMA_BASE_URL, timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def check_model_available(model_name: str) -> bool:
    """
    Check if a specific model is downloaded in Ollama.

    Parameters:
        model_name : e.g. "mistral" or "nomic-embed-text"

    Returns:
        True if the model is available, False otherwise.
    """
    try:
        # /api/tags lists all locally available models
        response = requests.get(
            f"{config.OLLAMA_BASE_URL}/api/tags",
            timeout=10,
        )
        if response.status_code != 200:
            return False

        # Parse the response -- it's a dict with a "models" list
        data = response.json()
        models = data.get("models", [])

        # Each model entry has a "name" field like "mistral:latest"
        # We check if our model_name is a prefix of any installed model name
        for model in models:
            if model.get("name", "").startswith(model_name):
                return True

        return False

    except requests.exceptions.ConnectionError:
        return False

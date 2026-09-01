"""
config.py -- Central settings for Chat with Web.

Like appsettings.json in .NET -- one place, all files read from here.
"""

import os

# =============================================================================
# EXTRACTION SETTINGS
# =============================================================================
# trafilatura = smart extractor (removes nav, ads, footers)
# justext = backup extractor (boilerplate removal)
# naive = last resort (just strips all HTML tags)
EXTRACTOR_ORDER = ["trafilatura", "justext", "naive"]

# Minimum characters for extracted text to be considered valid.
# Anything shorter = probably failed extraction (login wall, empty page).
MIN_TEXT_LENGTH = 200

# Request timeout in seconds when fetching URLs.
FETCH_TIMEOUT = 15

# =============================================================================
# CHUNKING SETTINGS
# =============================================================================
# How many characters per chunk.
# ~1000 chars = ~250 tokens = one solid paragraph.
# Smaller = more precise retrieval. Larger = more context per answer.
CHUNK_SIZE = 1000

# Characters shared between consecutive chunks.
# Prevents cutting a sentence at the boundary -- like a sliding window.
# C# equivalent: a for loop with step = CHUNK_SIZE - CHUNK_OVERLAP
CHUNK_OVERLAP = 150

# =============================================================================
# EMBEDDING SETTINGS
# =============================================================================
# sentence-transformers model for turning text into vectors.
# all-MiniLM-L6-v2 = small (80MB), fast, great for semantic search.
# Downloads automatically on first run -- no API key needed.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# =============================================================================
# VECTOR STORE (ChromaDB)
# =============================================================================
# Where ChromaDB stores data on disk (relative to this file).
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Collection name -- like a table in SQL Server.
COLLECTION_NAME = "web_content"

# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================
# How many chunks to pull per query.
# More = richer context, but slower and uses more tokens.
TOP_K_RESULTS = 5

# Max characters of context sent to the LLM.
# Keeps us under token limits (~4000 tokens).
MAX_CONTEXT_CHARS = 6000

# =============================================================================
# LLM SETTINGS (Ollama)
# =============================================================================
# Ollama runs locally -- no internet, no API key after install.
# Install: https://ollama.ai  then: ollama pull mistral
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "mistral"          # Change to "llama3.2", "phi3", etc.
OLLAMA_TIMEOUT = 120           # Seconds -- CPU generation is slow

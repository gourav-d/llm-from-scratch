"""
config.py -- Central settings for the Chat with Codebase app.

Think of this like appsettings.json in .NET --
one place to change settings, all files read from here.
"""

import os

# =============================================================================
# REPO TO INDEX
# =============================================================================
# Path to the codebase you want to chat with.
# Change this to any folder on your machine.
# Example: r"C:\Projects\MyApp" or "/home/user/myproject"
REPO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "modules")

# File extensions to index.
# We skip binary files, images, etc.
# In C#: like a file filter in Directory.GetFiles("*.py", "*.cs")
ALLOWED_EXTENSIONS = {
    ".py",    # Python
    ".md",    # Markdown documentation
    ".txt",   # Plain text
    ".js",    # JavaScript
    ".ts",    # TypeScript
    ".cs",    # C#
    ".java",  # Java
    ".go",    # Go
    ".rs",    # Rust
    ".cpp",   # C++
    ".h",     # C/C++ headers
    ".yaml",  # Config files
    ".yml",   # Config files
    ".json",  # JSON (small ones)
    ".toml",  # Config files
    ".sh",    # Shell scripts
    ".sql",   # SQL queries
}

# Folders to SKIP during indexing.
# These contain generated/compiled files, not source code.
SKIP_DIRS = {
    ".git",         # Git internals
    "__pycache__",  # Python compiled bytecode
    "node_modules", # JavaScript dependencies
    "venv",         # Python virtual environment
    ".venv",        # Python virtual environment (alternate name)
    "dist",         # Build output
    "build",        # Build output
    ".idea",        # IDE files
    ".vscode",      # VS Code settings
    "chroma_db",    # Our own vector database (skip re-indexing it)
}

# =============================================================================
# CHUNKING SETTINGS
# =============================================================================
# Chunk size = how many characters per chunk.
# Smaller = more precise retrieval but less context per chunk.
# Larger = more context but may retrieve irrelevant content.
# 1500 chars ~ 300-400 tokens ~ roughly one function or class.
CHUNK_SIZE = 1500

# Overlap = how many characters the next chunk shares with the previous.
# Like Ctrl+Z overlap -- ensures we don't cut a function in half at a boundary.
# In C#: like a sliding window over a List<char>.
CHUNK_OVERLAP = 200

# =============================================================================
# OLLAMA SETTINGS
# =============================================================================
# Ollama runs as a local REST API on your machine.
# Default port is 11434. No internet required.
OLLAMA_BASE_URL = "http://localhost:11434"

# The LLM used to ANSWER questions.
# mistral is ~4GB, good balance of speed and quality.
# Alternatives: "codellama", "llama3.2", "phi3"
LLM_MODEL = "mistral"

# The model used to create EMBEDDINGS (text -> vector).
# nomic-embed-text is ~300MB, fast and accurate.
EMBEDDING_MODEL = "nomic-embed-text"

# How many seconds to wait for Ollama to respond.
# LLM generation can be slow on CPU -- 120s is safe.
OLLAMA_TIMEOUT = 120

# =============================================================================
# CHROMADB SETTINGS
# =============================================================================
# Where ChromaDB stores its data on disk.
# Relative to this config.py file.
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Name of the collection inside ChromaDB.
# Like a table name in SQL Server.
COLLECTION_NAME = "codebase"

# =============================================================================
# RETRIEVAL SETTINGS
# =============================================================================
# How many chunks to retrieve for each question.
# More = more context for the LLM but slower and more tokens.
TOP_K_RESULTS = 5

# Maximum characters of context to send to the LLM.
# LLMs have token limits -- this keeps us under ~4000 tokens.
MAX_CONTEXT_CHARS = 6000

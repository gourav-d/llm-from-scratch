# Getting Started -- Capstone: Chat with Codebase

This guide walks you through everything needed to install, configure, and run
the project from scratch on a Windows machine.

Estimated time: 20-30 minutes (mostly waiting for model downloads).

---

## Prerequisites Check

Before starting, confirm you have these installed:

```powershell
# Check Python version -- needs 3.10 or higher
python --version
# Expected: Python 3.10.x or 3.11.x or higher

# Check pip
pip --version
# Expected: pip 23.x or higher

# Check Git
git --version
# Expected: git version 2.x
```

If Python is missing: download from https://www.python.org/downloads/
Choose "Add Python to PATH" during install.

---

## Step 1 -- Install Ollama

Ollama is a free tool that runs AI models locally on your machine.
Think of it like a local REST API server for LLMs -- no internet needed once set up.

**Download and install:**

1. Go to https://ollama.com
2. Click "Download for Windows"
3. Run the installer (OllamaSetup.exe)
4. Ollama starts automatically as a background service after install

**Verify Ollama is running:**

```powershell
# This should return: "Ollama is running"
curl http://localhost:11434
```

If it does not respond, open the Ollama app from the Start menu.

---

## Step 2 -- Pull Required Models

These are the two AI models this project uses.
Download them once -- they live on your machine permanently after that.

```powershell
# The LLM that answers your questions (~4 GB download, takes 5-10 minutes)
ollama pull mistral

# The embedding model that converts text to vectors (~300 MB, takes 1-2 minutes)
ollama pull nomic-embed-text
```

**What these models do:**

| Model | Size | Role |
|-------|------|------|
| mistral | ~4 GB | Reads the retrieved code and writes the answer |
| nomic-embed-text | ~300 MB | Converts text chunks and questions into vectors |

**Verify models downloaded:**

```powershell
ollama list
# Should show both: mistral and nomic-embed-text
```

---

## Step 3 -- Set Up Python Virtual Environment

A virtual environment keeps this project's packages isolated from other projects.
In .NET terms: this is like a project-specific packages folder instead of a global install.

```powershell
# Navigate to the project folder
cd "projects\capstone_chat_with_codebase"

# Create a virtual environment named "venv"
python -m venv venv

# Activate it (Windows PowerShell)
venv\Scripts\Activate.ps1

# You should see (venv) at the start of your prompt:
# (venv) PS C:\...\capstone_chat_with_codebase>
```

**If Activate.ps1 is blocked by execution policy:**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try again:
venv\Scripts\Activate.ps1
```

---

## Step 4 -- Install Python Dependencies

With the virtual environment active:

```powershell
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| chromadb | Local vector database -- stores and searches embeddings |
| streamlit | Creates the browser-based chat UI |
| requests | HTTP client -- talks to Ollama's REST API |
| tqdm | Progress bars during indexing |

**Verify install:**

```powershell
pip list | Select-String "chromadb|streamlit|requests|tqdm"
# Should show version numbers for all four
```

---

## Step 5 -- Configure the Repo Path

Open `config.py` and set `REPO_PATH` to the codebase you want to chat with.

By default it points to this course's `modules/` folder -- a great starting
point since you already know that code.

```python
# In config.py, find this line and change it:
REPO_PATH = r"C:\path\to\your\project"

# Examples:
REPO_PATH = r"C:\Users\YourName\Projects\MyApp"
REPO_PATH = r"C:\Users\YourName\Documents\GD\Learning\LLM\2026\1\modules"
```

Leave everything else as-is for your first run.

---

## Step 6 -- Index the Codebase

This one-time step reads all files, splits them into chunks, converts each chunk
to a vector, and stores everything in ChromaDB on disk.

```powershell
python indexer.py
```

**What you will see:**

```
============================================================
  Chat with Codebase -- Indexer
============================================================
Running pre-flight checks...
  [OK] Ollama is running
  [OK] Embedding model 'nomic-embed-text' is available
  [OK] Repo path exists: ...\modules

Index options:
  [1] Update -- add/update changed files (default)
  [2] Full re-index -- delete everything and start fresh
Choose [1/2]: 1

Scanning repo: ...\modules
Found 87 files to index.

Indexing files: 100%|====================| 87/87 [02:14<00:00,  1.54s/file]

============================================================
  Indexing Complete!
============================================================
  Files found:     87
  Files indexed:   85
  Files skipped:   2
  Chunks stored:   412
  Errors:          0

Next step: run 'python chat.py' to start chatting!
```

**How long does indexing take?**

| Repo size | Approximate time |
|-----------|-----------------|
| Small (~50 files) | 1-3 minutes |
| Medium (~200 files) | 5-10 minutes |
| Large (~1000 files) | 30-60 minutes |

This only runs once. After that, use `reindex.py` for updates (much faster).

---

## Step 7 -- Start Chatting

### Option A: Console chat (simplest)

```powershell
python chat.py
```

**Example session:**

```
============================================================
  Chat with Codebase  (Offline RAG)
============================================================
  Repo:   ...\modules
  Model:  mistral
  DB:     ...\chroma_db

Checking setup...
  [OK] ChromaDB: 412 chunks indexed
  [OK] Ollama running
  [OK] Models ready: mistral, nomic-embed-text

Ready! Ask your first question.

You: What does the attention mechanism do?

Searching codebase...

Assistant: The attention mechanism computes a weighted sum of all token representations. It allows each token to 'look at' every other token...

Sources used:
  - modules/04_transformers/lessons/03_attention.md
  - modules/04_transformers/examples/attention.py

You: exit

Goodbye!
```

**Console commands:**

| Type this | What it does |
|-----------|-------------|
| Any question | Search codebase and answer |
| `help` | Show usage tips |
| `sources` | Show source files from last answer |
| `clear` | Clear the terminal screen |
| `exit` or `quit` | Stop the app |

---

### Option B: Browser chat (recommended)

```powershell
streamlit run app.py
# Opens http://localhost:8501 automatically in your browser
```

The browser UI gives you:
- Chat message bubbles (like WhatsApp / Teams)
- Collapsible source file citations under each answer
- Sidebar showing system status and last answer's sources
- "Clear Conversation" button to start fresh
- Persistent chat history within the session

---

## Step 8 -- Re-index After Code Changes

When you add or edit files in the repo, run the smart re-indexer.
It only re-embeds changed files -- much faster than a full re-index.

```powershell
python reindex.py
```

**Example output:**

```
============================================================
  Chat with Codebase -- Smart Re-indexer
============================================================
Checking setup...
  [OK] Ollama ready
  [OK] ChromaDB: 412 chunks currently indexed
  [OK] Manifest: 85 files tracked

Scanning repo: ...\modules
Found 87 files in repo

Computing diff...
  New files:       2
  Changed files:   1
  Deleted files:   0
  Unchanged files: 84 (skipping)

Re-indexing 1 changed files...
Re-indexing changed: 100%|====================| 1/1

Indexing 2 new files...
Indexing new: 100%|====================| 2/2

============================================================
  Re-index Complete!
============================================================
  Files removed:   0
  Files updated:   1
  Files added:     2
  New chunks:      18
  Total in DB:     430
```

---

## Troubleshooting

### "Cannot connect to Ollama"

```
ERROR: Ollama is not running.
```

Fix: Open the Ollama desktop app from the Start menu, wait 10 seconds, try again.
Or run in a separate terminal: `ollama serve`

---

### "Model not found"

```
ERROR: LLM model 'mistral' not found.
  Run: ollama pull mistral
```

Fix: Run `ollama pull mistral` in any terminal. Takes 5-10 minutes.

---

### "Collection not found -- run indexer first"

```
RuntimeError: Collection 'codebase' not found in ChromaDB.
```

Fix: Run `python indexer.py` before `python chat.py`.

---

### "Activate.ps1 cannot be loaded"

```
venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled.
```

Fix:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Answer quality is poor

Try these in order:

1. **Check chunk size** -- in `config.py`, try `CHUNK_SIZE = 1000` for more precise retrieval
2. **Increase TOP_K** -- change `TOP_K_RESULTS = 5` to `TOP_K_RESULTS = 8`
3. **Try a better model** -- `ollama pull llama3.2` then change `LLM_MODEL = "llama3.2"` in config.py
4. **Re-index** -- run `python indexer.py` option 2 (full re-index) after config changes

---

### Indexing is very slow

Normal on CPU-only machines. Expected speeds:
- ~1-2 seconds per file on CPU
- ~0.1-0.3 seconds per file with a GPU

To speed up: reduce `CHUNK_SIZE` (fewer chunks per file) or remove large non-code files from `REPO_PATH`.

---

## Quick Reference

```
FIRST TIME SETUP:
  1. Install Ollama             https://ollama.com
  2. ollama pull mistral
  3. ollama pull nomic-embed-text
  4. pip install -r requirements.txt
  5. Edit config.py -> set REPO_PATH
  6. python indexer.py
  7. python chat.py  (or: streamlit run app.py)

DAILY USE:
  python chat.py               -- console chat
  streamlit run app.py         -- browser chat

AFTER CODE CHANGES:
  python reindex.py            -- update index (fast)
  python indexer.py            -- full re-index (slow, thorough)

ALL FILES:
  config.py                    -- settings (edit REPO_PATH here)
  indexer.py                   -- one-time index builder
  retriever.py                 -- RAG pipeline (question -> answer)
  chat.py                      -- console REPL
  app.py                       -- Streamlit browser UI
  reindex.py                   -- diff-based updater
  utils/chunker.py             -- file splitter
  utils/embedder.py            -- Ollama embedding client
  chroma_db/                   -- vector database on disk (auto-created)
  file_manifest.json           -- file hashes for diff detection (auto-created)
```

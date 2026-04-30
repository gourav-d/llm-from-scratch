# Getting Started -- Module 10: Vector Databases

## Step 1: Check Your Python Version

Vector databases require Python 3.8 or newer.

```bash
python --version
```

You should see: Python 3.8.x or higher

## Step 2: Create a Virtual Environment (Recommended)

A virtual environment keeps module libraries separate from other projects.
In .NET terms: this is like a project-specific NuGet package folder.

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

## Step 3: Install Libraries

```bash
pip install -r requirements.txt
```

Or install each library manually:

```bash
# Required for Examples 01 and 02 (no special library needed):
pip install numpy matplotlib

# Required for Examples 03, 04, 05 (vector database):
pip install chromadb

# Required for Example 05 (converts text to real vectors):
pip install sentence-transformers
```

## Library Guide

```
numpy              -- Array math. You already know this from Module 02.
matplotlib         -- Charts and graphs. Already used in earlier modules.
chromadb           -- The vector database. Like SQLite but for similarity search.
sentence-transformers  -- Converts text into vectors (embeddings).
                         Built on top of PyTorch.
```

## Step 4: Test Your Setup

Run Example 01 first -- it only needs NumPy, no ChromaDB required:

```bash
python examples/example_01_vectors_and_similarity.py
```

You should see output showing vector similarity calculations.

Then test ChromaDB is installed:

```bash
python -c "import chromadb; print('ChromaDB version:', chromadb.__version__)"
```

Then test sentence-transformers:

```bash
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

Note: sentence-transformers downloads a model on first use (~90 MB).
      Make sure you have an internet connection the first time.

## Trouble with sentence-transformers?

If sentence-transformers is slow to install (it downloads PyTorch), try:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

The CPU version of PyTorch is smaller and works fine for this module.

## File Order

Run the examples in order. Each builds on the previous:

```
example_01  ->  example_02  ->  example_03  ->  example_04  ->  example_05
(NumPy only)    (from scratch)  (ChromaDB)      (doc search)    (full system)
```

You can stop after example_02 if you only want the NumPy conceptual understanding.
You need example_03 onwards for the real ChromaDB experience.

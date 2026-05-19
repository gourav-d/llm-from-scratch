# Getting Started -- Module 10.8

## Step 1: Check Your Python Version

Open a terminal and run:

```bash
python --version
```

You need Python 3.10 or higher. If you have an older version, update it first.

---

## Step 2: Create and Activate a Virtual Environment

A virtual environment is like a separate project folder for Python packages.
It keeps this module's packages separate from other projects.

```bash
# Create the virtual environment
python -m venv venv

# Activate it on Windows:
venv\Scripts\activate

# Activate it on Mac/Linux:
source venv/bin/activate
```

You will see "(venv)" appear at the start of your terminal prompt.
That means it is active.

---

## Step 3: Install the Required Packages

```bash
pip install -r requirements.txt
```

This installs:

| Package              | What It Does                                        |
|----------------------|-----------------------------------------------------|
| numpy                | Number arrays and math -- you already have this     |
| sentence-transformers| Real bi-encoder and cross-encoder models            |
| faiss-cpu            | Facebook's fast vector index library                |
| rank-bm25            | Fast BM25 keyword scoring                           |
| scikit-learn         | Cosine similarity helpers                           |

---

## IMPORTANT NOTE: Two Tiers of Code

Every example file in this module has TWO sections:

### Tier 1 (Primary) -- NumPy only
- Uses ONLY numpy -- which you already have from Module 2
- Implements algorithms from scratch so you UNDERSTAND them
- Runs immediately with no extra install
- This is labeled at the top of each file

### Tier 2 (Part B) -- Real libraries
- Uses sentence-transformers, faiss-cpu, rank-bm25
- Shows how production systems actually work
- Located at the BOTTOM of each file
- Commented out -- remove the # to run

You can learn everything in Tier 1 without installing anything extra.
Tier 2 is for when you want to use real models.

---

## Step 4: Verify Your Install

Run this script to check everything works:

```python
# verify_install.py
import numpy as np
print("numpy version:", np.__version__)

try:
    from sentence_transformers import SentenceTransformer
    print("sentence-transformers: OK")
except ImportError:
    print("sentence-transformers: NOT installed (Tier 1 examples still work)")

try:
    import faiss
    print("faiss-cpu: OK")
except ImportError:
    print("faiss-cpu: NOT installed (Tier 1 examples still work)")

try:
    from rank_bm25 import BM25Okapi
    print("rank-bm25: OK")
except ImportError:
    print("rank-bm25: NOT installed (Tier 1 examples still work)")

print("")
print("If you see numpy: OK, you can run all Tier 1 examples.")
print("Install the other packages to run Tier 2 (Part B) sections.")
```

Save this as verify_install.py in the module folder and run:

```bash
python verify_install.py
```

---

## Step 5: Run Your First Example

```bash
python examples/example_01_keyword_vs_semantic.py
```

You should see output showing how keyword search fails and semantic search succeeds.

---

## Troubleshooting

### "faiss-cpu install fails on Windows"
Try:
```bash
pip install faiss-cpu --no-cache-dir
```
Or use the NumPy-only Tier 1 sections -- they teach the same concepts.

### "sentence-transformers is slow to download"
The first run downloads a ~90MB model. This is normal.
It downloads to your home folder and is cached for future runs.

### "ModuleNotFoundError: No module named 'numpy'"
You did not activate the virtual environment. Run:
```bash
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```
Then try again.

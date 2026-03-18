# Project 3: Semantic Code Search

**Search your codebase by meaning, not keywords**

---

## What You'll Build

A powerful code search engine that:
- Searches code by WHAT it does, not what it's named
- Finds similar code across your entire codebase
- Detects duplicate code automatically
- Understands code semantics using embeddings

**Use Case:** "Find all functions that validate user input" → Returns ALL validation code, even if it doesn't contain the word "validate"

---

## Why This Is Useful

### Traditional Search (Keyword)
```bash
grep "validate" *.py
# Only finds code with "validate" in the name
# Misses: check_input(), verify_user(), sanitize_data()
```

### Semantic Search (This Tool)
```bash
code-search "validate user input"
# Finds ALL input validation, regardless of naming:
# - validate_email()
# - check_user_data()
# - sanitize_input()
# - verify_form_fields()
```

**Frequency of Use:** Daily! Every time you need to find code in a large codebase.

---

## Key Features

### 1. Semantic Search
```bash
# Search by meaning
code-search "calculate fibonacci numbers"
# Returns all fibonacci implementations

# Natural language queries
code-search "convert string to integer with error handling"
# Finds all safe string→int conversion code
```

### 2. Similarity Search
```bash
# Find similar code
code-search --similar path/to/function.py
# Returns functions with similar logic

# Useful for:
# - Finding duplicates
# - Finding refactoring candidates
# - Learning how others solved similar problems
```

### 3. Duplicate Detection
```bash
# Find copy-pasted code
code-search --duplicates .
# Highlights code that's been copied and pasted

# Shows:
# - Exact duplicates
# - Near duplicates (with minor changes)
# - Semantic duplicates (same logic, different code)
```

### 4. Code Navigation
```bash
# Explore by concept
code-search "authentication logic"
# Shows all auth-related code

# Map your codebase
code-search --map "database access"
# Creates a map of all database code
```

---

## How It Works

### Module 7 Concepts Used

**Code Embeddings (Lesson 7):**
- Convert code into vector embeddings
- Similar code → Similar vectors
- Enable semantic search

**AST Parsing (Lesson 6):**
- Understand code structure
- Extract functions, classes, imports
- Normalize code before embedding

**Similarity Metrics:**
- Cosine similarity for semantic matching
- Levenshtein distance for exact matches
- AST diff for structural similarity

---

## Example Usage

### Find Authentication Code
```python
from semantic_code_search import CodeSearch

# Index your codebase
searcher = CodeSearch()
searcher.index_codebase("path/to/project")

# Search by meaning
results = searcher.search("user authentication and login")

# Results:
# 1. src/auth/login.py:42 - login_user()
#    Similarity: 0.95
#    Code: Handles user login with JWT tokens
#
# 2. src/middleware/auth.py:18 - check_auth()
#    Similarity: 0.89
#    Code: Middleware for authentication
#
# 3. src/api/users.py:156 - verify_credentials()
#    Similarity: 0.87
#    Code: Verifies username/password
```

### Find Similar Code
```python
# You wrote a function
my_function = """
def calculate_total_price(items, tax_rate):
    subtotal = sum(item.price for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax
"""

# Find similar implementations
similar = searcher.find_similar(my_function)

# Results:
# Found 3 similar functions:
# 1. checkout.py:89 - compute_order_total()
#    Similarity: 0.91 (VERY SIMILAR)
#
# 2. billing.py:45 - calculate_invoice()
#    Similarity: 0.78 (SIMILAR)
#
# 3. cart.py:120 - get_cart_total()
#    Similarity: 0.72 (SOMEWHAT SIMILAR)
#
# Suggestion: Consider consolidating these into one function!
```

### Detect Duplicates
```python
# Find copy-pasted code
duplicates = searcher.find_duplicates("./src", threshold=0.90)

# Results:
# Found 15 duplicate code blocks:
#
# Duplicate Group 1 (Exact matches):
#   - src/utils/validation.py:20-35
#   - src/api/validators.py:45-60
#   - src/forms/checks.py:12-27
#   Recommendation: Extract to shared utility
#
# Duplicate Group 2 (Near matches - 95% similar):
#   - src/auth/permissions.py:78-92
#   - src/admin/access.py:103-117
#   Difference: Variable names only
#   Recommendation: Refactor into common function
```

---

## Real-World Use Cases

### 1. Onboarding New Developers
```bash
# New dev: "How do we handle database connections?"
code-search "database connection setup"
# Shows all database connection code instantly
```

### 2. Code Review
```bash
# Reviewer: "I've seen this pattern before..."
code-search --similar pr_changes.py
# Finds existing implementations to compare
```

### 3. Refactoring
```bash
# Find all duplicate validation code
code-search --duplicates ./src --category "validation"
# Consolidate into shared utilities
```

### 4. Bug Fixing
```bash
# Find where else this bug might exist
code-search --similar buggy_function.py
# Check similar code for the same issue
```

### 5. Learning Codebase
```bash
# Explore a new codebase
code-search "payment processing flow"
# Understand how payments work
```

---

## C# Comparison

| .NET Tool | Python Equivalent | This Project |
|-----------|-------------------|--------------|
| Visual Studio Search | grep/find | Basic search |
| ReSharper Find Usages | ast.parse | AST-based |
| CodeLens | - | Not available |
| **This Tool** | - | **Semantic search!** |

**This goes BEYOND what even commercial .NET tools can do!**

---

## Project Structure

```
03_semantic_code_search/
├── README.md
├── requirements.txt
│
├── semantic_search.py       # Main search engine
├── indexer/
│   ├── code_indexer.py      # Index codebase
│   ├── embedder.py          # Generate embeddings
│   └── ast_analyzer.py      # Parse code structure
│
├── searcher/
│   ├── semantic_searcher.py # Semantic search
│   ├── similarity_finder.py # Find similar code
│   └── duplicate_detector.py # Find duplicates
│
├── utils/
│   ├── vector_store.py      # Store embeddings (FAISS)
│   ├── code_normalizer.py   # Normalize code
│   └── ranking.py           # Rank results
│
└── examples/
    ├── example_01_basic_search.py
    ├── example_02_similarity.py
    ├── example_03_duplicates.py
    └── example_04_cli.py
```

---

## Getting Started

```bash
cd projects/03_semantic_code_search
pip install -r requirements.txt

# Index your codebase
python -m semantic_search index /path/to/project

# Search
python -m semantic_search query "error handling with retries"

# Find duplicates
python -m semantic_search duplicates .
```

---

## Technologies Used

- **FAISS:** Facebook's vector similarity search (super fast!)
- **SentenceTransformers:** Code embeddings
- **AST parsing:** Code structure understanding
- **Sklearn:** Similarity metrics

---

## Difficulty: ⭐⭐⭐ Advanced

**Time Estimate:** 8-10 hours

**Prerequisites:**
- Module 7 Lessons 6-7 (Code embeddings)
- Understanding of vector similarity
- Basic machine learning knowledge

---

## Success Criteria

- [x] Can search code by meaning
- [x] Finds similar code accurately
- [x] Detects duplicate code
- [x] Fast (sub-second search on 10k+ files)
- [x] Works with multiple languages

---

**Never waste time searching for code again!** 🔍

This tool makes navigating large codebases effortless!

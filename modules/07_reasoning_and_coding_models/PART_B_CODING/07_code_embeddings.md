# Lesson 7: Code Embeddings & Understanding

## 📚 What You'll Learn

In this lesson, you'll learn how to create **embeddings** for code that capture semantic meaning. This is fundamental to building tools like:
- **Semantic code search** (find similar functions even if they use different variable names)
- **Code completion** (suggest relevant code based on context)
- **Bug detection** (find similar buggy patterns)
- **Code review assistants** (find similar code that was previously reviewed)

**Think of it like this:** If text embeddings let you search for "similar meanings" in documents, code embeddings let you search for "similar functionality" in code!

---

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand why code needs different embeddings than text
2. ✅ Know the difference between token-level, line-level, and function-level embeddings
3. ✅ Build a code embedding model from scratch
4. ✅ Implement semantic code search
5. ✅ Calculate code similarity metrics
6. ✅ Build a simple code recommendation engine

**Time:** 2-3 hours
**Difficulty:** Intermediate
**Prerequisites:** Lesson 6 (Code Tokenization)

---

## 📖 Part 1: Why Code Embeddings Are Different

### What Are Embeddings?

**Embeddings** are dense vector representations of data. Instead of representing code as text, we represent it as a list of numbers (a vector).

**Example:**
```python
# This code...
def add(a, b):
    return a + b

# ...becomes a vector like this:
[0.23, -0.45, 0.67, 0.12, -0.89, ...]  # 768 numbers
```

### Why Not Just Use Text Embeddings?

**Problem:** Regular text embeddings don't understand code structure!

```python
# These two functions do the SAME thing:
def add(x, y):
    return x + y

def sum_numbers(num1, num2):
    result = num1 + num2
    return result
```

**Text embeddings** would say these are different (different variable names, different structure).
**Code embeddings** should say these are SIMILAR (same functionality)!

### C# Analogy

In C#, this is like:
```csharp
// Text comparison (different)
"add(x, y)" != "sum_numbers(num1, num2)"

// Semantic comparison (should be same!)
Func<int, int, int> add = (x, y) => x + y;
Func<int, int, int> sum = (num1, num2) => num1 + num2;
// These are functionally equivalent!
```

---

## 📖 Part 2: Types of Code Embeddings

### 1. Token-Level Embeddings

**What:** Each token (keyword, variable, operator) gets its own embedding.

**Example:**
```python
code = "def add(x, y):"
tokens = ["def", "add", "(", "x", ",", "y", ")", ":"]

# Each token -> embedding
embeddings = {
    "def": [0.1, 0.2, ...],
    "add": [0.3, 0.4, ...],
    "x": [0.5, 0.6, ...],
    ...
}
```

**Use Case:** Code completion (predict next token)

### 2. Line-Level Embeddings

**What:** Each line of code gets one embedding.

**Example:**
```python
line1 = "def add(x, y):"        -> [0.2, 0.5, 0.8, ...]
line2 = "    return x + y"      -> [0.3, 0.6, 0.7, ...]
```

**Use Case:** Code review, finding similar code patterns

### 3. Function-Level Embeddings

**What:** Entire functions get one embedding.

**Example:**
```python
def add(x, y):
    return x + y

# Entire function -> ONE embedding
function_embedding = [0.23, -0.45, 0.67, ...]
```

**Use Case:** Semantic search, finding similar functions

### Comparison Table

| Level | Granularity | Use Case | Vector Size |
|-------|-------------|----------|-------------|
| **Token** | Fine | Next-token prediction | 128-768 |
| **Line** | Medium | Code review, diff analysis | 384-768 |
| **Function** | Coarse | Semantic search, deduplication | 768-1536 |

---

## 📖 Part 3: Building Code Embeddings

### Approach 1: Average Token Embeddings (Simple)

**Idea:** Average all token embeddings in a function.

```python
def get_function_embedding(tokens, token_embeddings):
    """
    Simple averaging approach.

    In C#, this is like:
    var average = list.Average();
    """
    embeddings = [token_embeddings[token] for token in tokens]
    return np.mean(embeddings, axis=0)  # Average all vectors
```

**Pros:** Simple, fast
**Cons:** Loses structural information

### Approach 2: Weighted Embeddings (Better)

**Idea:** Give more weight to important tokens (like function names, keywords).

```python
def weighted_embedding(tokens, token_embeddings, weights):
    """
    Weighted average.

    In C#:
    var weighted = list.Zip(weights, (x, w) => x * w);
    """
    embeddings = [token_embeddings[tok] for tok in tokens]
    weighted = [emb * w for emb, w in zip(embeddings, weights)]
    return np.sum(weighted, axis=0) / np.sum(weights)
```

**Example Weights:**
```python
weights = {
    "def": 2.0,      # Keywords are important
    "class": 2.0,
    "add": 3.0,      # Function name is VERY important
    "x": 0.5,        # Variables less important
    "return": 1.5,   # Return statement important
}
```

### Approach 3: Transformer-Based (Best)

**Idea:** Use a transformer model (like BERT for code).

**Models:**
- **CodeBERT** (Microsoft): BERT trained on code
- **GraphCodeBERT** (Microsoft): Uses code structure
- **CodeT5** (Salesforce): T5 model for code
- **Codex embeddings** (OpenAI): From GPT models

```python
from transformers import AutoModel, AutoTokenizer

# Load pre-trained code model
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_code_embedding(code):
    """
    Get embedding using CodeBERT.
    Returns a 768-dimensional vector.
    """
    # Tokenize code
    inputs = tokenizer(code, return_tensors="pt",
                      padding=True, truncation=True)

    # Get embeddings from model
    outputs = model(**inputs)

    # Use [CLS] token (first token) as function embedding
    embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()

    return embedding[0]  # Shape: (768,)
```

---

## 📖 Part 4: Semantic Code Search

### The Goal

**Input:** User query (natural language or code snippet)
**Output:** Most similar code from your codebase

**Example:**
```
Query: "function that adds two numbers"

Results:
1. def add(x, y): return x + y          [similarity: 0.95]
2. def sum(a, b): return a + b          [similarity: 0.92]
3. def calculate_total(x, y): ...       [similarity: 0.78]
```

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│                  Code Search Pipeline                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  1. Index Building (One Time)                            │
│     ┌──────────┐                                         │
│     │  Code 1  │──┐                                      │
│     │  Code 2  │──┼─> Embed all code                     │
│     │  Code 3  │──┘    └──> Store in vector DB          │
│     └──────────┘                                         │
│                                                           │
│  2. Query Time (Every Search)                            │
│     ┌─────────┐                                          │
│     │  Query  │──> Embed query                           │
│     └─────────┘      └──> Find nearest neighbors         │
│                             └──> Return top K matches    │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Cosine Similarity

**How we measure similarity between embeddings:**

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.

    Range: -1 to 1
    - 1.0  = identical vectors
    - 0.0  = orthogonal (unrelated)
    - -1.0 = opposite vectors

    In C#, this is like calculating the angle between vectors:
    Math.Cos(angle_between_vectors)
    """
    # Dot product
    dot_product = np.dot(vec1, vec2)

    # Magnitude of each vector
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity
```

**Example:**
```python
# Two similar functions
embedding1 = get_code_embedding("def add(x, y): return x + y")
embedding2 = get_code_embedding("def sum(a, b): return a + b")

similarity = cosine_similarity(embedding1, embedding2)
print(f"Similarity: {similarity:.2f}")  # Output: ~0.90
```

---

## 📖 Part 5: Advanced Techniques

### 1. Abstract Syntax Tree (AST) Embeddings

**Idea:** Use code structure, not just tokens.

```python
import ast

code = """
def add(x, y):
    return x + y
"""

# Parse into AST
tree = ast.parse(code)

# AST structure:
"""
Module
└── FunctionDef (name='add')
    ├── arguments
    │   ├── arg (name='x')
    │   └── arg (name='y')
    └── Return
        └── BinOp
            ├── Name (id='x')
            ├── Add
            └── Name (id='y')
"""

# Embed AST nodes, not just tokens!
```

**Benefit:** Captures code structure better than flat token sequences.

### 2. Cross-Language Embeddings

**Goal:** Similar code in different languages should have similar embeddings.

```python
# Python
def add(x, y):
    return x + y

# JavaScript
function add(x, y) {
    return x + y;
}

# Both should have SIMILAR embeddings!
```

**How:** Train on multilingual code datasets with parallel examples.

### 3. Embedding Fine-Tuning

**Idea:** Fine-tune embeddings for your specific domain.

```python
"""
Example: Fine-tune for your company's codebase

1. Collect similar code pairs from your repo
2. Train model to give them similar embeddings
3. Collect dissimilar pairs
4. Train model to give them different embeddings
"""

# Contrastive learning loss
def contrastive_loss(anchor, positive, negative):
    """
    anchor = query code
    positive = similar code (should be close)
    negative = different code (should be far)
    """
    pos_similarity = cosine_similarity(anchor, positive)
    neg_similarity = cosine_similarity(anchor, negative)

    # Loss: maximize pos_similarity, minimize neg_similarity
    loss = max(0, neg_similarity - pos_similarity + margin)
    return loss
```

---

## 📖 Part 6: Practical Applications

### 1. Code Duplication Detection

```python
def find_duplicates(codebase, threshold=0.85):
    """
    Find duplicate or near-duplicate code.

    Args:
        codebase: List of code snippets
        threshold: Similarity threshold (0-1)

    Returns:
        List of duplicate pairs
    """
    embeddings = [get_code_embedding(code) for code in codebase]
    duplicates = []

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                duplicates.append((i, j, sim))

    return duplicates
```

### 2. Code Recommendation

```python
def recommend_similar_functions(query_code, codebase, top_k=5):
    """
    Recommend similar functions from codebase.

    Like Netflix recommendations, but for code!
    """
    # Embed query
    query_embedding = get_code_embedding(query_code)

    # Embed all codebase functions
    codebase_embeddings = [
        get_code_embedding(code) for code in codebase
    ]

    # Calculate similarities
    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in codebase_embeddings
    ]

    # Get top K
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(codebase[i], similarities[i]) for i in top_indices]
```

### 3. Bug Pattern Detection

```python
def find_bug_patterns(buggy_code, codebase):
    """
    Find code similar to known buggy patterns.

    Example:
    - Known bug: "if x = 5:" (assignment instead of ==)
    - Find similar patterns in codebase
    """
    bug_embedding = get_code_embedding(buggy_code)

    risky_code = []
    for code in codebase:
        emb = get_code_embedding(code)
        sim = cosine_similarity(bug_embedding, emb)

        if sim > 0.8:  # Very similar to buggy pattern
            risky_code.append((code, sim))

    return risky_code
```

---

## 📖 Part 7: Evaluation Metrics

### How do we know if embeddings are good?

### 1. Code-to-Code Retrieval

**Task:** Given a code snippet, find the most similar code.

**Metric:** Recall@K (how many correct results in top K?)

```python
# Example:
query = "def add(x, y): return x + y"
expected_similar = ["def sum(a, b): return a + b"]

results = recommend_similar_functions(query, codebase, top_k=10)

# Did we find the expected similar code in top 10?
recall_at_10 = 1 if expected_similar[0] in results else 0
```

### 2. Code-to-Text Retrieval

**Task:** Given natural language query, find relevant code.

```python
query = "function to add two numbers"
expected_code = "def add(x, y): return x + y"

# Search with natural language
results = search_code_by_text(query, codebase)

# Was expected code in top 5?
recall_at_5 = 1 if expected_code in results[:5] else 0
```

### 3. Mean Reciprocal Rank (MRR)

**Measures:** How high is the correct answer in the results?

```python
def calculate_mrr(queries, expected, results_list):
    """
    MRR = average of (1 / rank of correct answer)

    Examples:
    - Correct answer is #1: 1/1 = 1.0
    - Correct answer is #2: 1/2 = 0.5
    - Correct answer is #10: 1/10 = 0.1
    - Not in results: 0.0
    """
    reciprocal_ranks = []

    for i, query in enumerate(queries):
        results = results_list[i]
        expected_code = expected[i]

        if expected_code in results:
            rank = results.index(expected_code) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)
```

---

## 🎓 Key Takeaways

### What We Learned

1. **Code embeddings** convert code into dense vectors that capture semantic meaning
2. **Three levels:** Token-level, line-level, and function-level embeddings
3. **Semantic search** uses cosine similarity to find similar code
4. **Applications:** Code search, duplication detection, bug finding, recommendations
5. **Evaluation:** Use Recall@K and MRR to measure quality

### Code vs Text Embeddings

| Feature | Text Embeddings | Code Embeddings |
|---------|----------------|-----------------|
| **Structure** | Sequential | Hierarchical (AST) |
| **Semantics** | Meaning | Functionality |
| **Similarity** | Word overlap | Behavior equivalence |
| **Training** | Books, articles | GitHub, Stack Overflow |

### C# Comparison

**Python:**
```python
# Get embedding
embedding = get_code_embedding(code)

# Find similar
similarity = cosine_similarity(emb1, emb2)
```

**C# Equivalent:**
```csharp
// Get embedding
var embedding = GetCodeEmbedding(code);

// Find similar
var similarity = CosineSimilarity(emb1, emb2);
```

---

## 🧪 Quiz Time!

### Question 1: Multiple Choice

**What is the main advantage of code embeddings over text embeddings?**

A) Code embeddings are smaller
B) Code embeddings understand functional equivalence
C) Code embeddings are faster to compute
D) Code embeddings don't need training

<details>
<summary>Click for answer</summary>

**Answer: B**

Code embeddings understand that two functions can do the same thing even with different variable names and structure. Text embeddings would see them as different.
</details>

### Question 2: Multiple Choice

**What does a cosine similarity of 0.95 mean?**

A) The code is 95% identical
B) The embeddings are very similar
C) The code has 95% of the same tokens
D) The code runs 95% faster

<details>
<summary>Click for answer</summary>

**Answer: B**

Cosine similarity ranges from -1 to 1. A value of 0.95 means the embedding vectors are pointing in nearly the same direction, indicating the code is very similar semantically.
</details>

### Question 3: Short Answer

**Why would you use function-level embeddings instead of token-level embeddings for semantic code search?**

<details>
<summary>Click for answer</summary>

**Answer:**

Function-level embeddings capture the overall functionality and purpose of a function, making them ideal for finding "similar functions" regardless of implementation details. Token-level embeddings are too granular and would focus on specific tokens rather than overall behavior.
</details>

### Question 4: Code Understanding

**What will this code do?**

```python
def find_duplicates(codebase, threshold=0.85):
    embeddings = [get_code_embedding(code) for code in codebase]
    duplicates = []

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                duplicates.append((i, j, sim))

    return duplicates
```

<details>
<summary>Click for answer</summary>

**Answer:**

This function finds duplicate or near-duplicate code in a codebase:
1. It converts all code snippets to embeddings
2. It compares every pair of embeddings using cosine similarity
3. If two embeddings are more than 85% similar, it records them as duplicates
4. It returns a list of duplicate pairs with their similarity scores

**Line-by-line:**
- Line 2: Convert all code to embeddings (like LINQ Select in C#)
- Lines 5-6: Nested loop to compare all pairs (avoids comparing twice)
- Line 7: Calculate how similar two code snippets are
- Line 8-9: If very similar (>85%), add to duplicates list
</details>

---

## 🎯 Practice Exercises

### Exercise 1: Build a Simple Code Search

**Task:** Implement a basic code search function.

```python
def code_search(query, codebase, top_k=5):
    """
    Search for code similar to query.

    Args:
        query: Code snippet or natural language
        codebase: List of code snippets
        top_k: Number of results to return

    Returns:
        List of (code, similarity) tuples
    """
    # TODO: Implement this!
    pass
```

<details>
<summary>Solution</summary>

```python
def code_search(query, codebase, top_k=5):
    # Step 1: Get query embedding
    query_embedding = get_code_embedding(query)

    # Step 2: Get all codebase embeddings
    codebase_embeddings = [
        get_code_embedding(code) for code in codebase
    ]

    # Step 3: Calculate similarities
    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in codebase_embeddings
    ]

    # Step 4: Get top K indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Step 5: Return results
    return [(codebase[i], similarities[i]) for i in top_indices]
```
</details>

### Exercise 2: Calculate Code Similarity

**Task:** Calculate similarity between two code snippets.

```python
code1 = """
def add(x, y):
    return x + y
"""

code2 = """
def sum_values(a, b):
    result = a + b
    return result
"""

# TODO: Calculate similarity
similarity = ???
print(f"Similarity: {similarity:.2f}")
```

<details>
<summary>Solution</summary>

```python
# Get embeddings
embedding1 = get_code_embedding(code1)
embedding2 = get_code_embedding(code2)

# Calculate similarity
similarity = cosine_similarity(embedding1, embedding2)
print(f"Similarity: {similarity:.2f}")

# Expected output: ~0.85-0.95 (high similarity)
```
</details>

---

## 🚀 Next Steps

### What's Next?

**In Lesson 8**, we'll learn how to **train models on code** using:
- Code-specific training techniques
- Fill-in-the-middle (FIM) training
- Multi-language training
- Fine-tuning for code generation

### Further Reading

1. **CodeBERT Paper:** "CodeBERT: A Pre-Trained Model for Programming and Natural Languages"
2. **GraphCodeBERT:** Uses data flow in code graphs
3. **CodeT5:** T5 model adapted for code understanding
4. **Codex Paper:** OpenAI's code generation model

### Practice Projects

1. Build a code duplicate detector for your projects
2. Create a "similar function" recommendation engine
3. Build a bug pattern finder
4. Implement cross-language code search

---

## 📝 Summary

### Key Concepts

| Concept | Description | Use Case |
|---------|-------------|----------|
| **Code Embeddings** | Vector representations of code | Semantic search |
| **Cosine Similarity** | Measure of embedding similarity | Finding similar code |
| **Function-level** | Entire function → one vector | Code search |
| **Token-level** | Each token → one vector | Code completion |
| **AST Embeddings** | Structure-aware embeddings | Better understanding |

### Python Concepts Used

- List comprehensions: `[emb for emb in embeddings]`
- NumPy operations: `np.mean()`, `np.dot()`, `np.linalg.norm()`
- Zip function: `zip(embeddings, weights)`
- Argsort: `np.argsort(similarities)`

### C# Equivalents

```python
# Python
embeddings = [get_embedding(code) for code in codebase]
```

```csharp
// C#
var embeddings = codebase.Select(code => GetEmbedding(code)).ToList();
```

---

**Congratulations!** You now understand how to build and use code embeddings! 🎉

**Next lesson:** Training Models on Code (Codex-style)

---

**Remember:** Code embeddings are the foundation for intelligent code tools like GitHub Copilot, semantic search, and code review assistants!

**Created:** March 16, 2026
**Module:** 07 - Reasoning and Coding Models
**Part:** B - Coding Models
**Lesson:** 7 of 10

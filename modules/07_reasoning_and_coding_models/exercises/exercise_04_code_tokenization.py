"""
Module 07 - Reasoning & Coding Models
Exercise 04: Code Tokenization & Code Embeddings

GLOSSARY
--------
Code Tokenization : Breaking source code into discrete units (tokens).
                    Unlike natural language, code has special characters:
                    parentheses, colons, operators (+=, ->), indentation.
Token             : Smallest meaningful unit of code.
                    Word-level: "def", "add", "(", "a", ",", "b", ")"
                    Character-level: "d", "e", "f", " ", "a", ...
Vocabulary        : Set of all unique tokens seen in a corpus.
                    Larger vocab = less splitting; smaller = more splitting.
Token ID          : Integer index of a token in the vocabulary.
                    "def" -> 0, "(" -> 1, etc. Like Enum values in C#.
Code Embedding    : A vector (list of floats) representing a code snippet.
                    Similar code has similar (close) embeddings.
                    Like converting code to coordinates in semantic space.
Cosine Similarity : Measure of angle between two vectors. Range [-1, 1].
                    1.0 = identical direction (very similar)
                    0.0 = perpendicular (unrelated)
                    Used to find "which function is most similar to this?"
Token Frequency   : How often each token appears in a corpus.
                    Used to build vocabulary (common tokens get own IDs).
"""

import re            # re: regular expressions for tokenizing
import numpy as np   # NumPy for vector math

print("=" * 60)
print("Exercise 04: Code Tokenization & Code Embeddings")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Word-Level Code Tokenizer
#
#  Background:
#    Split code into tokens using word boundaries AND special characters.
#    Code tokens include: identifiers, keywords, operators, punctuation.
#
#    Strategy: use regex to split on whitespace but KEEP special chars.
#    Pattern: r"(\s+|[(),:=+\-*/\[\]{}.])"  <- splits on these
#    Then filter out empty/whitespace-only tokens.
#
#    Example:
#      "def add(a, b):\n    return a + b"
#      -> ["def", "add", "(", "a", ",", "b", ")", ":", "return", "a", "+", "b"]
#
#  Your Task:
#    Write: tokenize_code(code_str) -> list of str
#    Split code into tokens. Filter out whitespace-only tokens.
#
#  C# Analogy:
#    Regex.Split(code, @"(\s+|[()\[\]{},.:=+\-*/])").Where(t => t.Trim() != "")
# ============================================================

print("-" * 50)
print("EXERCISE 1: Word-Level Code Tokenizer")
print("-" * 50)
print()


def tokenize_code(code_str):
    """
    Tokenize source code into a list of tokens.

    Parameters:
        code_str (str): Source code string.

    Returns:
        list of str: Individual tokens (no whitespace-only strings).
    """
    # TODO:
    # 1. Split using: parts = re.split(r"(\s+|[(),:=+\-*/\[\]{}.])", code_str)
    # 2. Filter out empty strings and whitespace-only: [t for t in parts if t.strip()]
    pass  # Replace with your implementation


code_a = "def add(a, b):\n    return a + b"
code_b = "x = 10 * (y + 2)"

tokens_a = tokenize_code(code_a)
tokens_b = tokenize_code(code_b)

if tokens_a is not None:
    print(f"  Code  : {repr(code_a)}")
    print(f"  Tokens: {tokens_a}")
    print(f"  Count : {len(tokens_a)}")
    print()
    print(f"  Code  : {repr(code_b)}")
    print(f"  Tokens: {tokens_b}")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Build Vocabulary from Code Corpus
#
#  Background:
#    A vocabulary maps each unique token to an integer ID.
#    Built by: collect all unique tokens, sort them, assign IDs.
#
#    Steps:
#      1. Tokenize each code snippet in the corpus
#      2. Collect all unique tokens into a set
#      3. Sort tokens (for reproducibility)
#      4. Create dict: token -> id  (token_to_id)
#      5. Create dict: id -> token  (id_to_token)
#
#    Special tokens added at position 0:
#      <PAD> = padding token (fill short sequences to fixed length)
#      <UNK> = unknown token (for tokens not in vocab)
#
#  Your Task:
#    Write: build_vocabulary(code_corpus) -> dict
#    code_corpus: list of code strings
#    Returns: {"token_to_id": dict, "id_to_token": dict, "size": int}
#    Include <PAD>=0 and <UNK>=1 at the start.
#
#  C# Analogy:
#    Dictionary<string, int> tokenToId = new() { ["<PAD>"]=0, ["<UNK>"]=1 };
#    // then add sorted tokens starting from id=2
# ============================================================

print("-" * 50)
print("EXERCISE 2: Build Code Vocabulary")
print("-" * 50)
print()


def build_vocabulary(code_corpus):
    """
    Build a token vocabulary from a corpus of code snippets.

    Parameters:
        code_corpus (list of str): Code strings to learn vocabulary from.

    Returns:
        dict: {
            "token_to_id": dict[str -> int],  # token -> integer ID
            "id_to_token": dict[int -> str],  # integer ID -> token
            "size"        : int               # total vocabulary size
        }
    """
    # TODO:
    # 1. Start with special tokens: token_to_id = {"<PAD>": 0, "<UNK>": 1}
    # 2. Collect all tokens from all snippets using tokenize_code()
    #    unique_tokens = set()
    #    for code in code_corpus:
    #        unique_tokens.update(tokenize_code(code))
    # 3. Sort: sorted_tokens = sorted(unique_tokens)
    # 4. Assign IDs starting from 2:
    #    for i, token in enumerate(sorted_tokens):
    #        token_to_id[token] = i + 2
    # 5. Build reverse: id_to_token = {v: k for k, v in token_to_id.items()}
    # 6. Return {"token_to_id": ..., "id_to_token": ..., "size": len(token_to_id)}
    pass  # Replace with your implementation


corpus = [
    "def add(a, b):\n    return a + b",
    "def multiply(x, y):\n    return x * y",
    "result = add(3, 4) + multiply(2, 5)",
]

vocab = build_vocabulary(corpus)

if vocab:
    print(f"  Corpus: {len(corpus)} snippets")
    print(f"  Vocabulary size: {vocab['size']}")
    print()
    # Show first 10 tokens
    print("  First 10 entries (token -> id):")
    items = sorted(vocab['token_to_id'].items(), key=lambda x: x[1])
    for token, id_ in items[:10]:
        print(f"    {token!r:>12} -> {id_}")
    print()
    print(f"  <PAD> id: {vocab['token_to_id'].get('<PAD>')}  (expected: 0)")
    print(f"  <UNK> id: {vocab['token_to_id'].get('<UNK>')}  (expected: 1)")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Encode and Decode a Code Snippet
#
#  Background:
#    Encoding: convert code string -> list of integer token IDs
#    Decoding: convert list of token IDs -> code string (space-joined)
#
#    For unknown tokens (not in vocab), use <UNK> id = 1.
#
#    Encoding steps:
#      1. Tokenize the code string
#      2. For each token: look up ID; use UNK id if not found
#
#    Decoding steps:
#      1. For each ID: look up token in id_to_token
#      2. Join with spaces
#
#  Your Task:
#    Write: encode(code_str, vocab) -> list of int
#    Write: decode(token_ids, vocab) -> str
#
#  C# Analogy:
#    int[] ids = tokens.Select(t => tokenToId.GetValueOrDefault(t, UNK)).ToArray();
#    string text = string.Join(" ", ids.Select(id => idToToken[id]));
# ============================================================

print("-" * 50)
print("EXERCISE 3: Encode and Decode Code")
print("-" * 50)
print()


def encode(code_str, vocab):
    """
    Convert a code string to a list of token IDs.

    Parameters:
        code_str (str) : Source code to encode.
        vocab    (dict): Vocabulary dict with "token_to_id" key.

    Returns:
        list of int: Token IDs. Unknown tokens get ID 1 (<UNK>).
    """
    # TODO:
    # tokens = tokenize_code(code_str)
    # unk_id = vocab["token_to_id"]["<UNK>"]
    # return [vocab["token_to_id"].get(t, unk_id) for t in tokens]
    pass  # Replace with your implementation


def decode(token_ids, vocab):
    """
    Convert a list of token IDs back to a code string.

    Parameters:
        token_ids (list of int): Sequence of token IDs.
        vocab     (dict)       : Vocabulary dict with "id_to_token" key.

    Returns:
        str: Tokens joined by spaces.
    """
    # TODO:
    # return " ".join(vocab["id_to_token"].get(id_, "<UNK>") for id_ in token_ids)
    pass  # Replace with your implementation


if vocab:
    test_code = "def add(a, b):\n    return a + b"
    ids = encode(test_code, vocab)
    recovered = decode(ids, vocab) if ids else None

    if ids is not None:
        print(f"  Original  : {repr(test_code)}")
        print(f"  Encoded   : {ids}")
        print(f"  Decoded   : {repr(recovered)}")
        print()
        # Test unknown token
        unknown_code = "import pandas"   # "import" and "pandas" not in corpus
        unk_ids = encode(unknown_code, vocab)
        if unk_ids:
            print(f"  Unknown tokens 'import pandas' -> IDs: {unk_ids}")
            print(f"  (all should be 1 = <UNK> since not in corpus vocabulary)")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Cosine Similarity for Code Search
#
#  Background:
#    Code embeddings let us find semantically similar code.
#    Two functions that do the same thing should have similar embeddings.
#
#    Cosine similarity formula:
#      similarity = dot(A, B) / (norm(A) * norm(B))
#      Range: [-1, 1]
#      1.0 = identical direction (similar)
#      0.0 = perpendicular (unrelated)
#
#    Code search:
#      1. Embed query function
#      2. Compute similarity to each function in the database
#      3. Return top-K most similar functions
#
#  Your Task:
#    Write: cosine_similarity(vec_a, vec_b) -> float
#    Write: find_similar_code(query_vec, database, top_k=3) -> list
#    database: list of (name, embedding_vector) tuples
#    Returns list of (name, similarity) sorted by similarity descending.
#
#  C# Analogy:
#    float similarity = Vector.Dot(a, b) / (a.Magnitude() * b.Magnitude());
# ============================================================

print("-" * 50)
print("EXERCISE 4: Cosine Similarity for Code Search")
print("-" * 50)
print()


def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two embedding vectors.

    Parameters:
        vec_a (np.ndarray): First vector.
        vec_b (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity in [-1, 1].
    """
    # TODO:
    # dot     = np.dot(vec_a, vec_b)
    # norm_a  = np.linalg.norm(vec_a)
    # norm_b  = np.linalg.norm(vec_b)
    # return dot / (norm_a * norm_b + 1e-10)   # +epsilon avoids divide-by-zero
    pass  # Replace with your implementation


def find_similar_code(query_vec, database, top_k=3):
    """
    Find the most similar code functions to a query.

    Parameters:
        query_vec (np.ndarray)     : Embedding of the query function.
        database  (list of tuple)  : [(name, embedding_vector), ...]
        top_k     (int)            : How many results to return.

    Returns:
        list of tuple: [(name, similarity), ...] sorted by similarity descending.
    """
    # TODO:
    # 1. Compute similarity between query_vec and each (name, vec) in database
    # 2. results = [(name, cosine_similarity(query_vec, vec)) for name, vec in database]
    # 3. Sort by similarity descending: results.sort(key=lambda x: x[1], reverse=True)
    # 4. Return results[:top_k]
    pass  # Replace with your implementation


np.random.seed(42)
dim = 16   # small embedding dimension for demo

# Simulate code embeddings:
#   add_numbers and sum_values should be similar (both do addition)
#   sort_list is unrelated (different operation)
add_embedding   = np.random.randn(dim)
sum_embedding   = add_embedding + np.random.randn(dim) * 0.1   # very similar to add
sort_embedding  = np.random.randn(dim)                          # unrelated
mul_embedding   = np.random.randn(dim)                          # unrelated

database = [
    ("add_numbers",  add_embedding),
    ("sum_values",   sum_embedding),
    ("sort_list",    sort_embedding),
    ("multiply_all", mul_embedding),
]

query = add_embedding + np.random.randn(dim) * 0.05   # query similar to add

results = find_similar_code(query, database, top_k=3)
if results is not None:
    print("  Query: function similar to 'add_numbers'")
    print()
    print(f"  {'Rank':<6} {'Function':<16} {'Similarity':>12}")
    print("  " + "-" * 38)
    for rank, (name, sim) in enumerate(results, 1):
        print(f"  {rank:<6} {name:<16} {sim:>12.4f}")
    print()
    print("  Expected: 'add_numbers' and 'sum_values' at top (most similar)")
print()

print("=" * 60)
print("All exercises complete!")
print("=" * 60)

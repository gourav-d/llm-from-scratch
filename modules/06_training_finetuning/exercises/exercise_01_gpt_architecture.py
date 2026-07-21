"""
Module 06 - Training & Fine-Tuning
Exercise 01: GPT Architecture Concepts

GLOSSARY
--------
Vocabulary Size  : How many unique tokens the model knows.
                   Like the number of words in a dictionary.
                   GPT-2 uses 50,257 tokens.
Embedding Dim    : The length of the vector that represents each token.
                   Like a list of 768 numbers that captures the "meaning" of a word.
Parameters       : All the numbers (weights) a model stores and learns.
                   More parameters = more capacity to learn, but more memory needed.
Model Size (MB)  : Memory needed to store the model's parameters.
                   Formula: num_params * bytes_per_param / 1_000_000
float32          : 32-bit floating point number. Uses 4 bytes. Standard precision.
float16          : 16-bit float. Uses 2 bytes. Half the memory of float32.
int8             : 8-bit integer. Uses 1 byte. Quarter the memory of float32.
Embedding Matrix : A table of shape (vocab_size, embed_dim).
                   Row i = vector for token i.
                   Like a lookup table: token ID -> dense vector.
"""

import numpy as np   # NumPy: arrays and math (like System.Math + arrays in C#)

print("=" * 60)
print("Exercise 01: GPT Architecture Concepts")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Count Parameters in a Simple GPT
#
#  Background:
#    A simplified GPT has two main weight matrices:
#      1. Embedding matrix:   shape (vocab_size, embed_dim)
#         params = vocab_size * embed_dim
#      2. Output weights:     shape (embed_dim, vocab_size)
#         params = embed_dim * vocab_size
#    Total = 2 * vocab_size * embed_dim
#
#    Real GPT also has attention and feedforward weights per layer,
#    but this exercise focuses on the embedding layers.
#
#  Your Task:
#    Write: count_gpt_params(vocab_size, embed_dim) -> int
#    Returns total number of parameters in the simplified GPT.
#
#  C# Analogy:
#    Like counting the total cells in two 2D arrays (matrices).
#    Array1.Length + Array2.Length = total cells.
# ============================================================

print("-" * 50)
print("EXERCISE 1: Count GPT Parameters")
print("-" * 50)
print()


def count_gpt_params(vocab_size, embed_dim):
    """
    Count total parameters in a simplified GPT model.

    The model has:
      - Embedding matrix:  vocab_size x embed_dim
      - Output weights:    embed_dim  x vocab_size

    Parameters:
        vocab_size (int): Number of tokens in vocabulary.
        embed_dim  (int): Size of embedding vectors.

    Returns:
        int: Total number of parameters.
    """
    # TODO: Compute and return total params
    # embedding_params = vocab_size * embed_dim
    # output_params    = embed_dim  * vocab_size
    # total            = embedding_params + output_params
    pass  # Replace with your implementation


# Tests
r1 = count_gpt_params(100, 64)
r2 = count_gpt_params(1000, 128)
r3 = count_gpt_params(50257, 768)
if r1 is not None:
    print(f"  vocab=100, embed=64              -> {r1:,} params")
    print(f"  vocab=1000, embed=128            -> {r2:,} params")
    print(f"  vocab=50257, embed=768 (GPT-2)   -> {r3:,} params")
print()
print("  Expected (roughly): 12,800 / 256,000 / 77,000,000")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Model Memory in MB
#
#  Background:
#    Memory usage depends on:
#      - Number of parameters
#      - Bytes per parameter (dtype)
#
#    Common dtypes:
#      float32 -> 4 bytes  (default training dtype)
#      float16 -> 2 bytes  (half precision, inference)
#      int8    -> 1 byte   (quantized, deployment)
#
#    Formula:
#      memory_mb = num_params * bytes_per_param / 1_000_000
#
#  Your Task:
#    Write: model_memory_mb(num_params, dtype) -> float
#    dtype is a string: "float32", "float16", or "int8"
#
#  C# Analogy:
#    Like calculating file size on disk:
#      file_size_bytes = num_values * sizeof(float)   // C#
#      file_size_mb    = file_size_bytes / 1_000_000
# ============================================================

print("-" * 50)
print("EXERCISE 2: Model Memory in MB")
print("-" * 50)
print()

DTYPE_BYTES = {
    "float32": 4,   # 4 bytes per parameter
    "float16": 2,   # 2 bytes per parameter
    "int8":    1,   # 1 byte  per parameter
}


def model_memory_mb(num_params, dtype="float32"):
    """
    Calculate model memory usage in megabytes.

    Parameters:
        num_params (int): Total number of parameters.
        dtype      (str): Data type: "float32", "float16", or "int8".

    Returns:
        float: Memory in MB.
    """
    # TODO: Look up bytes_per_param from DTYPE_BYTES dict
    # Then compute: memory_mb = num_params * bytes_per_param / 1_000_000
    pass  # Replace with your implementation


# GPT-2 small has ~117M params
gpt2_params = 117_000_000

print("  GPT-2 small (117M params) memory by dtype:")
for dtype in ["float32", "float16", "int8"]:
    mb = model_memory_mb(gpt2_params, dtype)
    if mb is not None:
        print(f"    {dtype:>8}: {mb:,.0f} MB  ({mb/1024:.2f} GB)")
print()
print("  Expected: float32~468 MB, float16~234 MB, int8~117 MB")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Embedding Lookup
#
#  Background:
#    An embedding matrix maps token IDs to vectors.
#    Shape: (vocab_size, embed_dim)
#    To look up vectors for a sequence of tokens:
#      embeddings[token_ids]   <- NumPy fancy indexing
#    This returns shape: (seq_len, embed_dim)
#
#    This is the FIRST step in every GPT forward pass.
#    Token "cat" -> ID 42 -> embeddings[42] -> [0.1, -0.3, ...]
#
#  Your Task:
#    Write: lookup_embeddings(embedding_matrix, token_ids) -> np.ndarray
#    Returns the embedding vectors for a list of token IDs.
#
#  C# Analogy:
#    Like a Dictionary<int, float[]> lookup:
#      token_ids.Select(id => embeddings[id]).ToArray()
# ============================================================

print("-" * 50)
print("EXERCISE 3: Embedding Lookup")
print("-" * 50)
print()

np.random.seed(42)   # Fix random seed so results are repeatable


def lookup_embeddings(embedding_matrix, token_ids):
    """
    Look up embedding vectors for a sequence of token IDs.

    Parameters:
        embedding_matrix (np.ndarray): Shape (vocab_size, embed_dim).
        token_ids        (list or np.ndarray): Sequence of token IDs.

    Returns:
        np.ndarray: Shape (len(token_ids), embed_dim) -- one row per token.
    """
    # TODO: Use NumPy fancy indexing to look up embeddings.
    # Hint: embedding_matrix[token_ids] returns rows at those indices.
    pass  # Replace with your implementation


vocab_size = 20
embed_dim  = 4
embedding_matrix = np.random.randn(vocab_size, embed_dim).round(2)

token_ids = [3, 7, 1, 3]   # Sequence: token 3, token 7, token 1, token 3

result = lookup_embeddings(embedding_matrix, token_ids)
if result is not None:
    print(f"  token_ids   : {token_ids}")
    print(f"  result shape: {result.shape}  (expected: (4, 4))")
    print(f"  First row   : {result[0]}  (should match embedding for token 3)")
    print(f"  Token 3 row : {embedding_matrix[3]}  (embedding_matrix[3])")
    print(f"  Match       : {np.allclose(result[0], embedding_matrix[3])}")
    print(f"  Token repeated (ids[0] == ids[3]): rows match? {np.allclose(result[0], result[3])}")
print()


# ============================================================
#  EXERCISE 4
#  Topic: GPT Parameter Scale Comparison
#
#  Background:
#    GPT models grew dramatically in size over time:
#      GPT-1:  117M  parameters
#      GPT-2:  1.5B  parameters
#      GPT-3:  175B  parameters
#      GPT-4:  ~1.8T parameters (estimated)
#
#    Scaling law: more parameters + more data = better model.
#    But cost grows too (memory, training time, inference cost).
#
#  Your Task:
#    Write: scale_comparison(params_a, params_b) -> dict
#    Returns: {"ratio": float, "larger": "A" or "B", "memory_diff_gb": float}
#    memory_diff_gb = difference in float32 memory between the two models.
#
#  C# Analogy:
#    Like comparing two database sizes:
#      ratio = size_b / size_a
#      diff_gb = (size_b - size_a) * bytes_per_row / 1e9
# ============================================================

print("-" * 50)
print("EXERCISE 4: GPT Model Scale Comparison")
print("-" * 50)
print()


def scale_comparison(params_a, params_b):
    """
    Compare two GPT models by parameter count.

    Parameters:
        params_a (int): Parameter count of model A.
        params_b (int): Parameter count of model B (larger expected).

    Returns:
        dict: {
            "ratio"         : float  -- how many times larger is the bigger model,
            "larger"        : str    -- "A" or "B" (whichever is bigger),
            "memory_diff_gb": float  -- float32 memory difference in GB
        }
    """
    # TODO:
    # 1. ratio = max(params_a, params_b) / min(params_a, params_b)
    # 2. larger = "A" if params_a > params_b else "B"
    # 3. memory_diff_gb = abs(params_a - params_b) * 4 / 1e9
    #    (4 = bytes per float32)
    pass  # Replace with your implementation


models = [
    ("GPT-1",  117_000_000),
    ("GPT-2",  1_500_000_000),
    ("GPT-3",  175_000_000_000),
]

gpt1_params = models[0][1]
gpt2_params = models[1][1]
gpt3_params = models[2][1]

r12 = scale_comparison(gpt1_params, gpt2_params)
r23 = scale_comparison(gpt2_params, gpt3_params)

if r12 and r23:
    print(f"  GPT-1 vs GPT-2:")
    print(f"    {r12['larger']} is larger by {r12['ratio']:.1f}x")
    print(f"    Memory diff: {r12['memory_diff_gb']:.2f} GB")
    print()
    print(f"  GPT-2 vs GPT-3:")
    print(f"    {r23['larger']} is larger by {r23['ratio']:.1f}x")
    print(f"    Memory diff: {r23['memory_diff_gb']:.2f} GB")
print()
print("  Expected: GPT-2 is ~12.8x larger than GPT-1, GPT-3 is ~116.7x larger than GPT-2")
print()

print("=" * 60)
print("All exercises complete!")
print("=" * 60)

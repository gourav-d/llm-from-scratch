"""
Exercise 03: Flash Attention — Complete the Tiled Implementation
Module 15: Advanced LLM Training

TASKS:
  1. Implement stable_softmax() — numerically stable softmax
  2. Implement standard_attention() — naive O(N²) attention
  3. Complete flash_attention_step() — process one (Q-tile, K-tile) pair
  4. Verify both implementations produce the same output
  5. Calculate memory usage difference

Run:  python exercise_03_flash_attention.py
Deps: numpy
"""

import numpy as np


# ─────────────────────────────────────────────────────────
# TASK 1: Stable Softmax
# ─────────────────────────────────────────────────────────

def stable_softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable row-wise softmax.

    Standard softmax: exp(x) / sum(exp(x))
    Problem: exp(large_number) = overflow
    Solution: subtract max from each row first

    Formula:
        x_stable = x - max(x, axis=-1)    (per row)
        softmax  = exp(x_stable) / sum(exp(x_stable))

    Args:
        x: 2D array of shape [batch, n_classes]

    Returns:
        softmax probabilities, same shape as x

    HINT:
        row_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - row_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Standard Attention
# ─────────────────────────────────────────────────────────

def standard_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Standard scaled dot-product attention (no causal mask).

    Steps:
      1. Scale queries: Q_scaled = Q / sqrt(d_k)
      2. Compute attention scores: S = Q_scaled @ K.T   shape: [N, N]
      3. Apply softmax: A = softmax(S)                  shape: [N, N]
      4. Compute output: O = A @ V                      shape: [N, d]

    Args:
        Q: queries, shape [N, d]
        K: keys,    shape [N, d]
        V: values,  shape [N, d]

    Returns:
        output shape [N, d]

    HINT:
        N, d = Q.shape
        scale = 1.0 / np.sqrt(d)
        scores = Q @ K.T * scale
        attention = stable_softmax(scores)
        return attention @ V
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Flash Attention — Process One Tile Pair
# ─────────────────────────────────────────────────────────

def flash_attention_step(Q_tile: np.ndarray,
                          K_tile: np.ndarray,
                          V_tile: np.ndarray,
                          O_running: np.ndarray,
                          m_running: np.ndarray,
                          l_running: np.ndarray,
                          scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process one (Q_tile, K_tile) pair and update running statistics.

    This is the CORE of Flash Attention.
    Called for each (i, j) tile pair during the outer-inner loop.

    Algorithm:
      1. Compute local scores: S = Q_tile @ K_tile.T * scale   shape: [Br, Bc]
      2. Compute local max per query row: m_new = max(S, axis=-1)    shape: [Br]
      3. Update global max: m_new = max(m_running, m_new)
      4. Compute rescale factors:
            alpha = exp(m_running - m_new)    (rescale old accumulator)
            beta  = exp(m_new_local - m_new)  (rescale new tile contribution)
             where m_new_local = max of S for this tile
      5. Compute local unnormalized attention: P = exp(S - m_new_local[:, None])
      6. Update running sum: l_new = alpha * l_running + beta * sum(P, axis=-1)
      7. Update output:      O_new = alpha[:, None] * O_running + beta[:, None] * (P @ V_tile)
      8. Return (O_new, m_new, l_new)

    Args:
        Q_tile:    query tile [Br, d]
        K_tile:    key tile   [Bc, d]
        V_tile:    value tile [Bc, d]
        O_running: accumulated output [Br, d]
        m_running: running max per query row [Br]
        l_running: running normalizer [Br]
        scale:     1 / sqrt(d)

    Returns:
        (O_new, m_new, l_new)

    HINT: Follow the 8 steps exactly. The rescaling is the tricky part.
    """
    # TODO: implement this
    # Step 1: local scores
    # S = Q_tile @ K_tile.T * scale

    # Step 2: local max
    # m_local = np.max(S, axis=-1)

    # Step 3: updated global max
    # m_new = np.maximum(m_running, m_local)

    # Step 4: rescale factors
    # alpha = np.exp(m_running - m_new)  # old accumulator rescaling
    # beta  = np.exp(m_local - m_new)    # new tile rescaling

    # Step 5: unnormalized attention probs
    # P = np.exp(S - m_local[:, None])

    # Step 6: updated sum
    # l_new = alpha * l_running + beta * np.sum(P, axis=-1)

    # Step 7: updated output
    # O_new = alpha[:, None] * O_running + beta[:, None] * (P @ V_tile)

    # Step 8: return
    pass


# ─────────────────────────────────────────────────────────
# FLASH ATTENTION OUTER LOOP (provided — uses your step fn)
# ─────────────────────────────────────────────────────────

def flash_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    block_size: int = 64) -> np.ndarray:
    """
    Full Flash Attention using flash_attention_step().
    This outer loop is provided — your job is to implement the step function above.
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    Br = min(block_size, N)
    Bc = min(block_size, N)

    O = np.zeros_like(Q)
    m = np.full(N, -np.inf)
    l = np.zeros(N)

    for i in range(0, N, Br):
        q_end = min(i + Br, N)
        Q_tile = Q[i:q_end]

        O_i = np.zeros((q_end - i, d))
        m_i = np.full(q_end - i, -np.inf)
        l_i = np.zeros(q_end - i)

        for j in range(0, N, Bc):
            k_end = min(j + Bc, N)
            K_tile = K[j:k_end]
            V_tile = V[j:k_end]

            result = flash_attention_step(Q_tile, K_tile, V_tile, O_i, m_i, l_i, scale)
            if result is None:
                return None  # step not implemented yet
            O_i, m_i, l_i = result

        O[i:q_end] = O_i / (l_i[:, None] + 1e-9)

    return O


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 03: Flash Attention")
    print("=" * 55)

    np.random.seed(42)

    # Test 1: stable_softmax
    print("\n--- Test 1: stable_softmax ---")
    x = np.array([[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]])
    result = stable_softmax(x)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        # Row sums should be 1.0
        row_sums = np.sum(result, axis=-1)
        if np.allclose(row_sums, 1.0, atol=1e-5):
            print(f"  PASS  row sums = {row_sums} (all ~1.0)")
        else:
            print(f"  FAIL  row sums = {row_sums} (should be 1.0)")
        # Large values (row 2) should not produce NaN or inf
        has_nan = np.any(np.isnan(result)) or np.any(np.isinf(result))
        if not has_nan:
            print(f"  PASS  no NaN/inf for large inputs")
        else:
            print(f"  FAIL  got NaN/inf: {result}")

    # Test 2: standard_attention
    print("\n--- Test 2: standard_attention ---")
    N, d = 16, 8
    Q = np.random.randn(N, d).astype(np.float32) * 0.1
    K = np.random.randn(N, d).astype(np.float32) * 0.1
    V = np.random.randn(N, d).astype(np.float32) * 0.1

    result = standard_attention(Q, K, V)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        if result.shape == (N, d):
            print(f"  PASS  output shape = {result.shape}")
        else:
            print(f"  FAIL  expected shape ({N}, {d}), got {result.shape}")
        if not np.any(np.isnan(result)):
            print(f"  PASS  no NaN in output")
        else:
            print(f"  FAIL  NaN in output")

    # Test 3: flash_attention matches standard_attention
    print("\n--- Test 3: Flash Attention vs Standard Attention ---")
    N, d = 64, 32
    Q = np.random.randn(N, d).astype(np.float32) * 0.1
    K = np.random.randn(N, d).astype(np.float32) * 0.1
    V = np.random.randn(N, d).astype(np.float32) * 0.1

    out_std   = standard_attention(Q, K, V)
    out_flash = flash_attention(Q, K, V, block_size=16)

    if out_std is None:
        print("  Skipped — standard_attention not implemented")
    elif out_flash is None:
        print("  NOT IMPLEMENTED YET (flash_attention_step)")
    else:
        max_diff = np.max(np.abs(out_std - out_flash))
        if max_diff < 1e-4:
            print(f"  PASS  max difference = {max_diff:.2e}")
        else:
            print(f"  FAIL  max difference = {max_diff:.2e} (should be < 1e-4)")

    # Test 4: Memory usage comparison
    print("\n--- Test 4: Memory Usage (O(N²) vs O(N)) ---")
    print(f"\n  Seq Length   Standard (N×N matrix)   Flash (running stats)")
    print("  " + "-" * 52)
    for seq_len in [128, 512, 2048, 8192]:
        d_size = 64
        n2_mb    = (seq_len * seq_len * 4) / 1024**2     # fp32
        flash_mb = (seq_len * 2 * 4)      / 1024**2     # m + l vectors
        print(f"  {seq_len:>10,}   {n2_mb:>20.2f} MB   {flash_mb:>18.6f} MB")

    print("""
  LESSON: At N=8192, standard needs 256 MB just for attention matrix.
          Flash needs only 0.0625 MB (4000x less) for running stats.
  """)


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def stable_softmax(x):
#     row_max = np.max(x, axis=-1, keepdims=True)
#     exp_x = np.exp(x - row_max)
#     return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)
#
# def standard_attention(Q, K, V):
#     N, d = Q.shape
#     scale = 1.0 / np.sqrt(d)
#     scores = Q @ K.T * scale
#     attention = stable_softmax(scores)
#     return attention @ V
#
# def flash_attention_step(Q_tile, K_tile, V_tile, O_running, m_running, l_running, scale):
#     S = Q_tile @ K_tile.T * scale
#     m_local = np.max(S, axis=-1)
#     m_new = np.maximum(m_running, m_local)
#     alpha = np.exp(m_running - m_new)
#     beta  = np.exp(m_local   - m_new)
#     P = np.exp(S - m_local[:, None])
#     l_new = alpha * l_running + beta * np.sum(P, axis=-1)
#     O_new = alpha[:, None] * O_running + beta[:, None] * (P @ V_tile)
#     return O_new, m_new, l_new

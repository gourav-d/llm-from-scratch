"""
Example 03: Flash Attention — Tiled Implementation in NumPy
Module 15: Advanced LLM Training

Implements Flash Attention from scratch to show:
  - Why standard attention uses O(N²) memory
  - How tiled computation avoids materializing the N×N matrix
  - The running (max, sum, output) accumulation trick for stable softmax
  - Memory usage comparison: standard vs Flash Attention

Run:  python example_03_flash_attention.py
Deps: numpy
"""

import numpy as np
import time


def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


# ─────────────────────────────────────────────────────────
# STANDARD ATTENTION
# ─────────────────────────────────────────────────────────

def standard_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                        causal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard (naive) scaled dot-product attention.

    Stores the full N×N attention matrix in memory.
    Memory: O(N²)

    Returns: (output, attention_matrix)
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    # Step 1: Compute attention scores — shape [N, N]
    scores = Q @ K.T * scale          # full N×N matrix allocated here

    # Step 2: Apply causal mask if needed
    if causal:
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        scores[mask] = -np.inf

    # Step 3: Softmax over last dimension
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)

    # Step 4: Weighted sum of values
    output = attention @ V

    return output, attention


# ─────────────────────────────────────────────────────────
# FLASH ATTENTION (NumPy implementation)
# ─────────────────────────────────────────────────────────

def flash_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    block_size: int = 64, causal: bool = False) -> np.ndarray:
    """
    Flash Attention: tiled computation that avoids N×N matrix.

    Algorithm (Dao et al. 2022):
      1. Split Q into row tiles of size Br
      2. Split K, V into column tiles of size Bc
      3. For each Q tile, iterate over all K/V tiles
      4. Maintain running statistics (m, l, O) for stable online softmax
      5. Never write N×N attention matrix to memory

    Memory: O(N × d) for Q, K, V, O + O(block_size) for tiles
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    # Output accumulator — same shape as Q
    O = np.zeros_like(Q)

    # Per-row running statistics for online softmax
    m = np.full(N, -np.inf)    # running maximum of scores per query row
    l = np.zeros(N)             # running sum of exp(score - m) per query row

    # Tile sizes
    Br = min(block_size, N)    # query block size (rows)
    Bc = min(block_size, N)    # key/value block size (cols)

    # Number of tiles
    Tr = (N + Br - 1) // Br   # number of query tiles
    Tc = (N + Bc - 1) // Bc   # number of key/value tiles

    # Outer loop: iterate over query tiles
    for i in range(Tr):
        q_start = i * Br
        q_end   = min(q_start + Br, N)
        Q_tile  = Q[q_start:q_end]           # shape: [Br, d]

        # Local running stats for this query tile
        m_i = np.full(q_end - q_start, -np.inf)
        l_i = np.zeros(q_end - q_start)
        O_i = np.zeros((q_end - q_start, d))

        # Inner loop: iterate over key/value tiles
        for j in range(Tc):
            k_start = j * Bc
            k_end   = min(k_start + Bc, N)
            K_tile  = K[k_start:k_end]       # shape: [Bc, d]
            V_tile  = V[k_start:k_end]       # shape: [Bc, d]

            # Compute local attention scores: [Br, Bc]
            S_ij = Q_tile @ K_tile.T * scale

            # Apply causal mask: query at position r cannot attend to key at position c > r
            if causal:
                for r_local in range(q_end - q_start):
                    r_global = q_start + r_local
                    for c_local in range(k_end - k_start):
                        c_global = k_start + c_local
                        if c_global > r_global:
                            S_ij[r_local, c_local] = -np.inf

            # Online softmax update:
            # New local max for each query in tile
            m_ij = np.max(S_ij, axis=-1)     # shape: [Br]

            # Update global max: m_new = max(m_old, m_new_local)
            m_i_new = np.maximum(m_i, m_ij)

            # Rescale factors to account for changed maximum
            exp_old = np.exp(m_i - m_i_new)              # for rescaling old accumulator
            exp_new = np.exp(m_ij - m_i_new)             # for new tile contribution

            # Local softmax numerator (unnormalized)
            P_ij = np.exp(S_ij - m_ij[:, None])          # shape: [Br, Bc]
            P_sum = np.sum(P_ij, axis=-1)                 # shape: [Br]

            # Update running sum (rescale old + add new)
            l_i = exp_old * l_i + exp_new * P_sum

            # Update output accumulator
            O_i = (exp_old[:, None] * O_i
                   + exp_new[:, None] * (P_ij @ V_tile))

            # Update max
            m_i = m_i_new

        # Finalize output for this query tile (divide by normalizer)
        O[q_start:q_end] = O_i / (l_i[:, None] + 1e-9)

    return O


# ─────────────────────────────────────────────────────────
# MEMORY TRACKER
# ─────────────────────────────────────────────────────────

def memory_usage_bytes(N: int, d: int, dtype_bytes: int = 4) -> dict:
    """Estimate memory usage for attention computation."""
    qkv_bytes    = 3 * N * d * dtype_bytes
    output_bytes = N * d * dtype_bytes

    # Standard: stores full N×N attention matrix
    attn_matrix_bytes = N * N * dtype_bytes

    # Flash: only stores running stats + tiles (O(N) not O(N²))
    flash_extra_bytes = 2 * N * dtype_bytes   # m and l vectors

    return {
        "Q_K_V_O":           (qkv_bytes + output_bytes) / 1024**2,
        "standard_attn_MiB": attn_matrix_bytes / 1024**2,
        "flash_extra_MiB":   flash_extra_bytes / 1024**2,
        "standard_total":    (qkv_bytes + output_bytes + attn_matrix_bytes) / 1024**2,
        "flash_total":       (qkv_bytes + output_bytes + flash_extra_bytes) / 1024**2,
    }


# ─────────────────────────────────────────────────────────
# DEMO 1: Correctness Check
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: Flash Attention vs Standard — Same Output?")

np.random.seed(42)
N, d = 128, 64
Q = np.random.randn(N, d).astype(np.float32) * 0.1
K = np.random.randn(N, d).astype(np.float32) * 0.1
V = np.random.randn(N, d).astype(np.float32) * 0.1

# Standard attention
out_standard, attn_matrix = standard_attention(Q, K, V, causal=True)

# Flash attention
out_flash = flash_attention(Q, K, V, block_size=32, causal=True)

max_diff = np.max(np.abs(out_standard - out_flash))
mean_diff = np.mean(np.abs(out_standard - out_flash))

print(f"\n  Sequence length N:   {N}")
print(f"  Head dimension d:    {d}")
print(f"  Block size:          32")
print(f"  Causal masking:      Yes")
print(f"\n  Max absolute difference:  {max_diff:.2e}")
print(f"  Mean absolute difference: {mean_diff:.2e}")
print(f"  Results match:       {'YES (within fp32 precision)' if max_diff < 1e-4 else 'NO — BUG!'}")
print(f"\n  Standard output[0,:4]:  {out_standard[0,:4]}")
print(f"  Flash   output[0,:4]:  {out_flash[0,:4]}")


# ─────────────────────────────────────────────────────────
# DEMO 2: Memory Comparison
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: Memory Usage — Standard vs Flash Attention")

print(f"\n{'Seq Length':>12} {'d':>6} {'QKV+O (MiB)':>14} {'Std Attn (MiB)':>16} {'Flash Extra (MiB)':>18} {'Savings':>10}")
print("-" * 80)

for seq_len in [512, 1024, 2048, 4096, 8192, 32768]:
    head_dim = 128
    mem = memory_usage_bytes(seq_len, head_dim, dtype_bytes=2)   # bf16 = 2 bytes
    savings = mem["standard_attn_MiB"] / mem["flash_extra_MiB"]
    print(
        f"{seq_len:>12,} {head_dim:>6} "
        f"{mem['Q_K_V_O']:>14.1f} "
        f"{mem['standard_attn_MiB']:>16.1f} "
        f"{mem['flash_extra_MiB']:>18.4f} "
        f"{savings:>9.0f}x"
    )

print("""
LESSON: At N=32768 (32K context), bf16 attention matrix alone = 2 GB per layer.
        Flash Attention needs only 0.0625 MiB extra — 32,000× less!
        This is why Flash Attention enables 128K+ context windows.
""")


# ─────────────────────────────────────────────────────────
# DEMO 3: Speed Comparison
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: Tiled vs Untiled — Understanding Block Processing")

np.random.seed(0)
N, d = 256, 64

Q = np.random.randn(N, d).astype(np.float32) * 0.1
K = np.random.randn(N, d).astype(np.float32) * 0.1
V = np.random.randn(N, d).astype(np.float32) * 0.1

print(f"\nSequence length: {N}, Head dim: {d}")

# Timing standard attention
t0 = time.perf_counter()
for _ in range(10):
    out_s, _ = standard_attention(Q, K, V, causal=False)
t_standard = (time.perf_counter() - t0) / 10

# Timing flash attention with different block sizes
for block_size in [32, 64, 128, 256]:
    t0 = time.perf_counter()
    for _ in range(10):
        out_f = flash_attention(Q, K, V, block_size=block_size, causal=False)
    t_flash = (time.perf_counter() - t0) / 10
    diff = np.max(np.abs(out_s - out_f))
    print(f"  Block size {block_size:>4}: {t_flash*1000:.2f}ms  (std: {t_standard*1000:.2f}ms)  "
          f"max_diff={diff:.2e}")

print("""
Note: NumPy Flash Attention is SLOWER than standard in Python.
      In real CUDA kernels, Flash Attention is 2-4× FASTER because:
      1. Tile data fits in GPU SRAM (much faster than HBM)
      2. Avoids reading/writing the N×N matrix to slow GPU RAM
      NumPy runs on CPU with no memory hierarchy benefit.
""")


# ─────────────────────────────────────────────────────────
# DEMO 4: Visualize the Tiling Pattern
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: Flash Attention Tiling Pattern")

N_vis = 8
block_vis = 4

print(f"\nSequence length: {N_vis}, Block size: {block_vis}")
print(f"Number of Q tiles (rows): {N_vis // block_vis}")
print(f"Number of K/V tiles (cols): {N_vis // block_vis}")
print(f"\nProcessing order (Q-tile i, K-tile j):")
print(f"Standard attention computes one {N_vis}×{N_vis} matrix at once.")
print(f"Flash Attention processes {(N_vis // block_vis)**2} tiles of size {block_vis}×{block_vis}.\n")

print("  Full N×N matrix (standard):")
print("  ┌" + "─" * (N_vis * 3) + "┐")
for i in range(N_vis):
    row = "  │"
    for j in range(N_vis):
        if j < i:   # below diagonal (non-causal: all computed)
            row += " S "
        elif j == i:
            row += " D "
        else:
            row += " . "
    row += "│"
    print(row)
print("  └" + "─" * (N_vis * 3) + "┘")
print("  S = off-diagonal, D = diagonal, . = upper triangle (causal: masked)")

print("\n  Flash Attention tile boundaries (block_size=4):")
print("  Q tiles: [0:4], [4:8]")
print("  K tiles: [0:4], [4:8]")
print("  Each small 4×4 tile fits in fast SRAM — no HBM write needed.")

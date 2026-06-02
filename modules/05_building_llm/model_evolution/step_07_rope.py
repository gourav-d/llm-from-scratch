# =============================================================================
# step_07_rope.py
# =============================================================================
# STEP 7: ROPE — ROTARY POSITIONAL EMBEDDINGS
#
# WHAT'S WRONG WITH LEARNED POSITIONAL EMBEDDINGS (STEPS 0-6)?
#
#   In steps 2-6, we used:
#     pos_embd = nn.Embedding(block_size, n_embd)
#     input = token_embed + pos_embd[position]
#
#   The model learns separate embedding vectors for positions 0, 1, 2, ..., 63.
#
#   PROBLEM 1: HARD SEQUENCE LENGTH LIMIT
#     If block_size=64 during training, pos_embd has 64 rows.
#     At inference, if you give it token 65, it would look up pos_embd[65]
#     which DOESN'T EXIST! The model literally can't handle longer sequences.
#
#   PROBLEM 2: POOR RELATIVE POSITION GENERALIZATION
#     The model learns "position 5 looks like this vector" and "position 8
#     looks like that vector", but it DOESN'T explicitly learn
#     "token at position 5 is 3 positions before token at position 8."
#     Relative distances must be learned implicitly from training data.
#
# WHAT IS ROPE (ROTARY POSITIONAL EMBEDDINGS)?
#
#   Instead of adding position info to the token embeddings,
#   RoPE ROTATES the Query and Key vectors by an angle that depends on position.
#
#   KEY INSIGHT:
#     If you rotate Q at position m by angle θ_m,
#     and rotate K at position n by angle θ_n,
#     then their dot product (Q_rotated · K_rotated)
#     naturally depends on the RELATIVE position (m - n)!
#
#   This is the "relative attention" property we wanted.
#
# THE MATH (simplified):
#
#   1. Split Q (or K) into pairs of dimensions: (d0, d1), (d2, d3), ...
#   2. Treat each pair as a 2D complex number: d0 + i*d1
#   3. Multiply by e^(i * position * theta_j) — this IS a rotation!
#      (Euler's formula: e^(iθ) = cos(θ) + i*sin(θ))
#   4. Unpack: (d0 + i*d1) * (cos + i*sin) = (d0*cos - d1*sin) + i*(d0*sin + d1*cos)
#              = q_rotated_real + i*q_rotated_imag
#
#   In real numbers:
#     q_rotated = [q[0]*cos - q[1]*sin,   q[0]*sin + q[1]*cos,
#                  q[2]*cos - q[3]*sin,   q[2]*sin + q[3]*cos, ...]
#
#   Equivalently using rotate_half trick:
#     q_rotated = q * cos + rotate_half(q) * sin
#     where rotate_half(q) = [-q[head_size//2:], q[:head_size//2]]
#                              ^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
#                              second half negated  first half
#
# THETA (FREQUENCY) SCHEDULE:
#   theta_j = base^(-2j/d) for j = 0, 1, ..., d/2 - 1
#
#   base=10000 (from the original paper)
#   d = head_size
#
#   Low dimensions (j near 0)  : theta ≈ 1   → slow rotation → long-range patterns
#   High dimensions (j near d/2): theta ≈ tiny → fast rotation → short-range patterns
#
#   This is similar to how sin/cos positional encoding in the original
#   Transformer paper used multiple frequencies.
#
# C# ANALOGY:
#   Think of a clock with many hands, each rotating at different speeds:
#     Hand 1: rotates once per day   (long period, slow)  → captures day-level patterns
#     Hand 2: rotates once per hour  (medium period)       → captures hour-level patterns
#     Hand 3: rotates once per minute (short period, fast) → captures minute-level patterns
#
#   Each dimension of the embedding is like one clock hand.
#   By looking at where each hand is pointing (cos/sin of its angle),
#   the model can tell exactly how far apart two tokens are.
#
# WHY RoPE IS BETTER:
#   1. No hard length limit — can extrapolate beyond training length
#   2. Relative positions are encoded mathematically, not just learned
#   3. Works better with KV cache (the cached K vectors are already rotated)
#   4. Used by: LLaMA 2/3, Mistral, Gemma, Qwen, Phi, Falcon...
#
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from shared import (
    load_data,
    build_vocab,
    split_data,
    run_training,
    encode,
    decode,
)


# =============================================================================
# HYPERPARAMETERS
# =============================================================================
CONFIG = {
    "block_size"    : 64,
    "batch_size"    : 64,
    "max_iters"     : 5000,
    "eval_interval" : 500,
    "lr"            : 3e-4,
    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
}

N_EMBD     = 128
N_HEADS    = 4
N_KV_HEADS = 2
N_LAYERS   = 3
DROPOUT    = 0.1
ROPE_BASE  = 10000   # The base for the frequency schedule (standard value)


# =============================================================================
# ROPE FUNCTIONS
# =============================================================================

def precompute_freqs(head_size: int, max_seq_len: int, base: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin rotation matrices for all positions and all dimensions.

    WHY PRECOMPUTE?
      These values don't change during training or inference.
      Computing them once and reusing is much faster than computing at every step.
      C# analogy: like precomputing a lookup table (static readonly Dictionary).

    THE FREQUENCY FORMULA:
      theta_j = base^(-2j / head_size)   for j = 0, 1, ..., head_size/2 - 1

      base = 10000 (standard)
      head_size = 32 (our head size)
      j = dimension pair index

      For j=0:  theta_0 = 10000^0       = 1.0     (slow rotation)
      For j=8:  theta_8 = 10000^(-0.5)  ≈ 0.01   (medium rotation)
      For j=15: theta_15 = 10000^(-0.937) ≈ 0.000115 (fast rotation)

    THEN: for each position m, the angle for dimension j is:
      angle(m, j) = m * theta_j

    RETURNS:
      freqs_cos: (max_seq_len, head_size/2)  — cos of angles
      freqs_sin: (max_seq_len, head_size/2)  — sin of angles
    """
    # Compute theta values for each dimension pair
    # torch.arange(0, head_size, 2): [0, 2, 4, ..., head_size-2]
    # These are the indices j = 0, 1, 2, ..., head_size//2 - 1
    # (we step by 2 because pairs share a theta)
    dim_indices = torch.arange(0, head_size, 2, dtype=torch.float32)
    # C# analogy: Enumerable.Range(0, head_size/2).Select(j => j * 2)

    # theta_j = base^(-2j/head_size) = (1/base)^(2j/head_size)
    # This exponentially decreases from 1 to base^(-1) as j increases
    # Lower theta → slower rotation → wider "clock hand" → long-range patterns
    thetas = 1.0 / (base ** (dim_indices / head_size))
    # thetas shape: (head_size // 2,)   e.g., (16,) for head_size=32

    # Compute angles for all positions: angle[m, j] = m * theta_j
    # torch.arange(max_seq_len): [0, 1, 2, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    # C# analogy: Enumerable.Range(0, maxSeqLen).Select(m => (float)m)

    # Outer product: positions × thetas → (max_seq_len, head_size//2)
    # angles[m, j] = m * theta_j
    # This is like a 2D grid: rows = positions, columns = dimensions
    angles = torch.outer(positions, thetas)  # (max_seq_len, head_size//2)
    # C# analogy: matrix where M[m][j] = positions[m] * thetas[j]

    # Compute cos and sin of all angles
    # These are the actual rotation factors applied to Q and K
    freqs_cos = torch.cos(angles)   # (max_seq_len, head_size//2)
    freqs_sin = torch.sin(angles)   # (max_seq_len, head_size//2)

    return freqs_cos, freqs_sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    The "rotate half" operation needed for the RoPE rotation.

    WHAT IT DOES:
      Split x into two halves along the last dimension.
      Return [-second_half, first_half] — second half negated, then first half.

      This implements the imaginary part of complex multiplication:
        (a + ib) * (cos + i*sin) = a*cos - b*sin + i*(a*sin + b*cos)
                                    ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
                                    real part         imaginary part
                                    = x*cos + rotate_half(x)*sin

      rotate_half(x) = [-b, a]  where x = [a, b]
      So: x * cos + rotate_half(x) * sin = [a*cos - b*sin, b*cos + a*sin]
                                           = rotated [a, b] by angle arctan(sin/cos)!

    EXAMPLE:
      x = [1.0, 2.0, 3.0, 4.0]  (head_size=4)
      x1 = [1.0, 2.0]            (first half)
      x2 = [3.0, 4.0]            (second half)
      rotate_half(x) = [-3.0, -4.0, 1.0, 2.0]
                        ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^
                        -x2 (negated 2nd half)  x1 (original 1st half)

    C# ANALOGY:
      var x1 = x[..n/2];    // first half
      var x2 = x[n/2..];    // second half
      // Concatenate negated second half + first half
      return x2.Select(v => -v).Concat(x1).ToArray();
    """
    d = x.shape[-1]              # Total dimension (e.g., 32)
    half = d // 2                # Half the dimension (e.g., 16)

    x1 = x[..., :half]          # First half: (B, n_heads, T, half)
    x2 = x[..., half:]          # Second half: (B, n_heads, T, half)

    # Concatenate: [-x2, x1] — negate second half, keep first half
    # C# analogy: Concat(x2.Select(v => -v), x1)
    return torch.cat([-x2, x1], dim=-1)   # (B, n_heads, T, d)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to Q and K tensors.

    THE FORMULA:
      q_rotated = q * cos + rotate_half(q) * sin
      k_rotated = k * cos + rotate_half(k) * sin

    This rotates each (q[2j], q[2j+1]) pair by angle m * theta_j
    where m = token position, theta_j = frequency for dimension pair j.

    C# ANALOGY:
      // Apply rotation to each dimension pair:
      float[] RotateVectors(float[] vec, float[] cos, float[] sin) {
          return vec.Zip(cos, (v, c) => v * c)
                    .Zip(RotateHalf(vec).Zip(sin, (v, s) => v * s), (a, b) => a + b)
                    .ToArray();
      }

    PARAMETERS:
      q: (B, n_heads, T, head_size) — query vectors
      k: (B, n_heads, T, head_size) — key vectors
      freqs_cos: (T, head_size//2)  — cosine values for each position and dimension
      freqs_sin: (T, head_size//2)  — sine values for each position and dimension

    RETURNS:
      (q_rotated, k_rotated) — same shapes as input
    """
    # freqs_cos shape: (T, head_size//2)
    # But we need cos for ALL pairs, not just half the dimensions
    # Since both halves of a dimension pair share the same theta,
    # we repeat the cos/sin values: (T, head_size//2) → (T, head_size)
    # torch.cat([cos, cos], dim=-1): [cos0, cos0, cos1, cos1, ...]
    # This matches the layout: [q0, q1, q2, q3, ...] → pairs (q0,q1), (q2,q3), ...
    T = q.shape[2]  # Sequence length

    # Take only the first T rows (we precomputed for max_seq_len)
    cos = freqs_cos[:T, :]   # (T, head_size//2)
    sin = freqs_sin[:T, :]   # (T, head_size//2)

    # Repeat cos/sin to match full head_size: (T, head_size//2) → (T, head_size)
    # Each pair (q0, q1) shares the same cos/sin value (they form a 2D rotation)
    cos = torch.cat([cos, cos], dim=-1)   # (T, head_size)
    sin = torch.cat([sin, sin], dim=-1)   # (T, head_size)

    # Reshape for broadcasting with q, k which are (B, n_heads, T, head_size)
    # We need cos/sin to be (1, 1, T, head_size) so they broadcast over B and n_heads
    # unsqueeze(0).unsqueeze(0): add two dimensions at the front
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_size)
    sin = sin.unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_size)

    # Apply rotation to Q: q_rotated = q * cos + rotate_half(q) * sin
    # Each dimension pair (q[2j], q[2j+1]) is rotated by angle m * theta_j
    q_rotated = q * cos + rotate_half(q) * sin   # (B, n_heads, T, head_size)

    # Apply the same rotation to K
    k_rotated = k * cos + rotate_half(k) * sin   # (B, n_heads, T, head_size)

    return q_rotated, k_rotated


# =============================================================================
# MODULE: RoPEAttention — GQA with RoPE instead of absolute pos embeddings
# =============================================================================
class RoPEAttention(nn.Module):
    """
    Group-Query Attention with RoPE positional encoding.

    KEY DIFFERENCES FROM step_06 CachedGQA:
      1. No absolute positional embedding added to token embeddings
      2. Instead, RoPE is applied INSIDE the attention, to Q and K only
      3. V is NOT rotated (rotation only affects the attention scores, not values)

    WHY ROTATE Q AND K BUT NOT V?
      The dot product Q·K (attention score) measures similarity.
      By rotating Q and K, we bake position info into the similarity computation:
        "are these tokens relevant to each other, given their positions?"
      V provides the content — it doesn't need position info.

    C# ANALOGY:
      Imagine rotating two vectors before comparing them with dot product.
      The dot product then naturally captures their angular relationship,
      which encodes relative position.
    """

    def __init__(self, n_embd, n_heads, n_kv_heads, block_size, dropout):
        super().__init__()

        assert n_heads % n_kv_heads == 0

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_repeat   = n_heads // n_kv_heads
        self.head_size  = n_embd // n_heads
        self.scale      = math.sqrt(self.head_size)

        # Q, K, V projections (same as step_05/06)
        self.q_proj = nn.Linear(n_embd, n_heads    * self.head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)

        # Causal mask (for training)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout  = nn.Dropout(dropout)

        # Precompute RoPE frequencies
        # We precompute for block_size positions and head_size dimensions
        # register_buffer: saved with the model, moved to device automatically
        # NOT trainable — these are fixed mathematical constants
        # C# analogy: static readonly lookup table (computed at construction)
        freqs_cos, freqs_sin = precompute_freqs(self.head_size, block_size, base=ROPE_BASE)
        self.register_buffer("freqs_cos", freqs_cos)  # (block_size, head_size//2)
        self.register_buffer("freqs_sin", freqs_sin)  # (block_size, head_size//2)

    def forward(self, x: torch.Tensor, past_kv=None):
        """
        RoPE-based attention forward pass with optional KV cache.
        Same interface as CachedGQA from step_06.
        """
        B, T_new, C = x.shape

        # ---- Project Q, K, V ----
        Q = self.q_proj(x).view(B, T_new, self.n_heads,    self.head_size)
        K = self.k_proj(x).view(B, T_new, self.n_kv_heads, self.head_size)
        V = self.v_proj(x).view(B, T_new, self.n_kv_heads, self.head_size)

        # Append to KV cache if provided
        if past_kv is not None:
            K_past, V_past = past_kv
            K = torch.cat([K_past, K], dim=1)   # (B, T_total, n_kv_heads, hs)
            V = torch.cat([V_past, V], dim=1)   # (B, T_total, n_kv_heads, hs)

        T_total = K.shape[1]   # Total sequence length (past + new)

        # ---- Expand KV for GQA ----
        K_exp = self._repeat_kv(K, self.n_repeat)   # (B, T_total, n_heads, hs)
        V_exp = self._repeat_kv(V, self.n_repeat)   # (B, T_total, n_heads, hs)

        # ---- Transpose to (B, n_heads, T, hs) for batched matmul ----
        Q   = Q.transpose(1, 2)    # (B, n_heads, T_new, hs)
        K_e = K_exp.transpose(1, 2)  # (B, n_heads, T_total, hs)
        V_e = V_exp.transpose(1, 2)  # (B, n_heads, T_total, hs)

        # ======================================================
        # *** THE KEY DIFFERENCE: APPLY ROPE TO Q AND K ***
        # ======================================================
        # We need separate cos/sin for Q (positions T_past..T_past+T_new)
        # and K (positions 0..T_total)
        if past_kv is not None and past_kv[0] is not None:
            T_past = past_kv[0].shape[1]
        else:
            T_past = 0

        # RoPE for Q: only the new positions (T_past to T_past+T_new-1)
        q_freqs_cos = self.freqs_cos[T_past : T_past + T_new]   # (T_new, hs//2)
        q_freqs_sin = self.freqs_sin[T_past : T_past + T_new]   # (T_new, hs//2)

        # RoPE for K: all positions (0 to T_total-1)
        k_freqs_cos = self.freqs_cos[:T_total]   # (T_total, hs//2)
        k_freqs_sin = self.freqs_sin[:T_total]   # (T_total, hs//2)

        # Apply RoPE rotation — encode position into Q and K
        Q,   _ = apply_rope(Q,   Q,   q_freqs_cos, q_freqs_sin)   # rotate Q
        K_e, _ = apply_rope(K_e, K_e, k_freqs_cos, k_freqs_sin)   # rotate K
        # (V is NOT rotated — position info is only needed for similarity)
        # ======================================================

        # ---- Standard attention ----
        scores = Q @ K_e.transpose(-2, -1) / self.scale   # (B, n_heads, T_new, T_total)

        # Causal mask (training only)
        if T_new == T_total:
            scores = scores.masked_fill(self.mask[:T_total, :T_total] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ V_e   # (B, n_heads, T_new, hs)

        # ---- Reassemble ----
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T_new, self.n_heads * self.head_size)
        out = self.dropout(self.out_proj(out))

        return out, (K, V)   # Return output and updated cache

    def _repeat_kv(self, kv, n_repeat):
        """Same as standalone repeat_kv function."""
        if n_repeat == 1:
            return kv
        B, T, n_kv, hs = kv.shape
        return kv.unsqueeze(3).expand(B, T, n_kv, n_repeat, hs).reshape(B, T, n_kv * n_repeat, hs)


# =============================================================================
# MODULE: FeedForward, RoPEBlock, RoPEModel
# =============================================================================
class FeedForward(nn.Module):
    """Same as previous steps."""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class RoPEBlock(nn.Module):
    """Transformer block using RoPE attention."""
    def __init__(self, n_embd, n_heads, n_kv_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = RoPEAttention(n_embd, n_heads, n_kv_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


class RoPEModel(nn.Module):
    """
    Full model with RoPE positional encoding.

    KEY DIFFERENCE FROM NANOGPT (step_04):
      NO pos_embd = nn.Embedding(block_size, n_embd)
      Position info is baked into Q and K via RoPE rotation instead.

    This makes the model:
      - More generalizable beyond training sequence length
      - Better at relative position reasoning
      - Closer to modern LLMs (LLaMA, Mistral, Qwen use RoPE)
    """

    def __init__(self, vocab_size, n_embd, n_heads, n_kv_heads, n_layers, block_size, dropout):
        super().__init__()

        self.block_size = block_size
        self.n_layers   = n_layers

        # Token embedding ONLY — no positional embedding!
        # C# analogy: we removed the position lookup table
        self.token_embd = nn.Embedding(vocab_size, n_embd)

        # RoPE blocks (position is handled inside attention via rotation)
        self.blocks = nn.ModuleList([
            RoPEBlock(n_embd, n_heads, n_kv_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        self.final_ln    = nn.LayerNorm(n_embd)
        self.output_proj = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None, past_kvs=None):
        """
        Forward pass with RoPE — position handled inside attention.
        Same interface as KVCacheModel from step_06.
        """
        B, T_new = x.shape

        # Token embeddings only (no position added here!)
        h = self.token_embd(x)   # (B, T_new, n_embd)

        # Run through RoPE blocks — position is encoded inside each block's attention
        new_past_kvs = []
        for i, block in enumerate(self.blocks):
            layer_past = past_kvs[i] if past_kvs is not None else None
            h, new_kv  = block(h, layer_past)
            new_past_kvs.append(new_kv)

        h = self.final_ln(h)
        logits = self.output_proj(h)

        if targets is None:
            loss = None
        else:
            B2, T2, V = logits.shape
            loss = F.cross_entropy(logits.view(B2 * T2, V), targets.view(B2 * T2))

        return logits, loss, new_past_kvs


# =============================================================================
# EXTRAPOLATION TEST
# =============================================================================
def test_extrapolation(model, char2idx, idx2char, device, block_size):
    """
    Test generation beyond the training block_size.
    RoPE should handle this better than absolute positional embeddings.
    """
    print("  Testing extrapolation beyond training block_size...")

    seed = "The quick brown fox jumps over the lazy dog "
    seed_ids = encode(seed, char2idx)
    extra_tokens = block_size + 20   # Generate 20 tokens BEYOND training length

    model.eval()
    ctx = torch.tensor(seed_ids, dtype=torch.long).unsqueeze(0).to(device)
    generated_ids = list(seed_ids)

    with torch.no_grad():
        for step in range(extra_tokens):
            # For RoPE: we can try generating beyond block_size
            # The frequencies were precomputed up to block_size,
            # but in principle RoPE can extrapolate
            ctx_cropped = ctx[:, -block_size:]   # Still crop for safety

            logits, _loss, _kvs = model(ctx_cropped, past_kvs=None)
            logits_last = logits[:, -1, :] / 0.8
            probs = F.softmax(logits_last, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ctx = torch.cat([ctx, next_id], dim=1)
            generated_ids.append(next_id.item())

    model.train()

    text = decode(generated_ids, idx2char)
    print(f"  Generated {extra_tokens} tokens from seed (showing last 200 chars):")
    print(f"  ...{text[-200:]}")
    return text


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  STEP 7: RoPE — Rotary Positional Embeddings             ║")
    print("║  What's new   : Replace pos_embd with Q,K rotations     ║")
    print("║  Architecture : No pos_embd; RoPE inside attention      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    print("WHAT THIS STEP TEACHES:")
    print("  - Absolute pos embeddings: learned vectors, hard limit at block_size")
    print("  - RoPE: rotate Q and K by position-dependent angles")
    print("  - Why rotation encodes RELATIVE position (not just absolute)")
    print("  - The theta frequency schedule (slow rot = long-range, fast = short-range)")
    print()
    print("THE MATH IN PLAIN ENGLISH:")
    print("  rotate_half(q) = swap the two halves of q, negate the first half")
    print("  q_rotated = q * cos(pos * theta) + rotate_half(q) * sin(pos * theta)")
    print("  This rotates q in 2D for each pair of dimensions")
    print("  Dot product of rotated Q and K depends on (pos_q - pos_k) = relative pos!")
    print()
    print("C# ANALOGY:")
    print("  Like a clock with 16 hands (for head_size=32 dimension pairs).")
    print("  Each hand rotates at a different speed (theta_j).")
    print("  Position m = how many ticks have passed.")
    print("  The angle of each hand at position m encodes that position uniquely.")
    print()
    print("REAL-WORLD: LLaMA 3, Mistral, Qwen, Gemma, Phi all use RoPE!")
    print()
    print("EXPECTED RESULT:")
    print("  - NanoGPT (step 04) val loss: ~1.3")
    print("  - RoPE (step 07)    val loss: ~1.2  (better position encoding)")
    print()

    # ---- Data ----
    print("[ Loading data ]")
    text = load_data(max_chars=10_000_000)

    print()
    print("[ Building vocabulary ]")
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(idx2char)

    print()
    print("[ Splitting data ]")
    train_data, val_data = split_data(text, char2idx)

    # ---- Model ----
    print()
    print("[ Creating model ]")
    model = RoPEModel(
        vocab_size  = vocab_size,
        n_embd      = N_EMBD,
        n_heads     = N_HEADS,
        n_kv_heads  = N_KV_HEADS,
        n_layers    = N_LAYERS,
        block_size  = CONFIG["block_size"],
        dropout     = DROPOUT,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params   : {total_params:,}")
    print(f"  Position method: RoPE (rotary angles, NOT learned embeddings)")
    print(f"  RoPE base      : {ROPE_BASE}  (10000 = standard)")
    print(f"  Head size      : {N_EMBD // N_HEADS}  ({N_EMBD // N_HEADS // 2} dimension pairs, each with its own theta)")

    # ---- Train ----
    print()
    print("[ Training ]")
    final_val_loss = run_training(
        model_name = "RoPE-GQA",
        model      = model,
        train_data = train_data,
        val_data   = val_data,
        char2idx   = char2idx,
        idx2char   = idx2char,
        config     = CONFIG,
    )

    # ---- Test extrapolation ----
    print()
    print("[ Extrapolation Test ]")
    device = CONFIG["device"]
    test_extrapolation(model, char2idx, idx2char, device, CONFIG["block_size"])

    # ---- Final summary ----
    print()
    print("=" * 60)
    print(f"  FINAL VAL LOSS: {final_val_loss:.4f}")
    print()
    print("  THE COMPLETE EVOLUTION:")
    print("    Bigram       (step 00): ~2.4  lookup table, 1-char context")
    print("    MLP          (step 01): ~2.0  16-char context window")
    print("    Single Attn  (step 02): ~1.7  learned relevance scores")
    print("    Multi-Head   (step 03): ~1.5  4 parallel perspectives + FFN")
    print("    NanoGPT      (step 04): ~1.3  3 blocks + residual + LayerNorm")
    print("    GQA          (step 05): ~1.3  fewer KV heads, 50% memory saving")
    print("    KV Cache     (step 06): ~1.3  5-10x faster generation, same quality")
    print(f"    RoPE         (step 07): ~1.2  better position encoding, extrapolation")
    print()
    print("  YOU NOW UNDERSTAND THE ARCHITECTURE OF:")
    print("    - GPT-2 / GPT-3   (step 04 architecture, just bigger)")
    print("    - LLaMA 3         (step 07: GQA + RoPE + more tricks)")
    print("    - Mistral 7B      (step 07: GQA + RoPE + sliding window)")
    print("    - Qwen2           (step 07: same core components!)")
    print()
    print("  Congratulations! You built an LLM from scratch.")
    print("=" * 60)
    print()
    print("  The evolution is complete!")
    print("  Explore: modules/05_building_llm/projects/ for capstone projects.")
    print("=" * 60)

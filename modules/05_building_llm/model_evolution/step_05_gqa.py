# =============================================================================
# step_05_gqa.py
# =============================================================================
# STEP 5: GROUP-QUERY ATTENTION (GQA)
#
# WHAT'S THE PROBLEM WITH STANDARD MULTI-HEAD ATTENTION?
#   In step_04 NanoGPT: n_heads=4, each head has its OWN Q, K, V matrices.
#   So we have 4 sets of K and 4 sets of V.
#   During text generation, we store ALL of these in the KV Cache (step 6).
#   For long sequences, this uses a LOT of memory!
#
#   Example: Serving LLaMA-70B with 80 attention heads:
#     Standard MHA: 80 K-heads + 80 V-heads = 160 matrices per layer
#     Each matrix grows with sequence length → gigabytes of GPU memory
#
# WHAT IS GROUP-QUERY ATTENTION (GQA)?
#   GQA uses FEWER K and V heads than Q heads.
#   Multiple Q heads SHARE the same K,V heads.
#
#   EXAMPLE with n_heads=4, n_kv_heads=2:
#     Q heads: [Q1, Q2, Q3, Q4]        ← 4 query heads (unchanged)
#     K heads: [K1, K2]                 ← only 2 key heads!
#     V heads: [V1, V2]                 ← only 2 value heads!
#
#     Grouping:
#       Q1, Q2 → both attend using K1, V1   (head group 1)
#       Q3, Q4 → both attend using K2, V2   (head group 2)
#
#   Memory saving (KV cache per token):
#     Standard MHA  : 2 × n_heads    × head_size = 2 × 4 × 32 = 256 values
#     GQA (n_kv=2)  : 2 × n_kv_heads × head_size = 2 × 2 × 32 = 128 values
#     Saving        : 50%!
#
# REAL-WORLD USAGE:
#   LLaMA 2 (7B):  n_heads=32, n_kv_heads=32  (standard MHA)
#   LLaMA 2 (70B): n_heads=64, n_kv_heads=8   (GQA! 8× memory saving)
#   LLaMA 3:       n_heads=32, n_kv_heads=8   (GQA on all sizes)
#   Mistral 7B:    n_heads=32, n_kv_heads=8   (GQA)
#   Qwen2:         n_heads=32, n_kv_heads=8   (GQA)
#
# KEY IMPLEMENTATION: repeat_kv
#
#   Because Q has more heads than K/V, we need to "expand" K and V so
#   each Q head has a matching K and V to attend to.
#
#   Instead of:    K1, K2             (2 KV heads)
#   We expand to:  K1, K1, K2, K2    (4 KV heads, each repeated n_repeat=2 times)
#
#   Then Q1 uses K1, Q2 uses K1 (shared!), Q3 uses K2, Q4 uses K2 (shared!).
#
# C# ANALOGY FOR repeat_kv:
#   Imagine you have a shared configuration for 2 groups, and 4 workers.
#   Instead of giving each worker their own config, you expand the 2 configs
#   to match all 4 workers:
#     configs = [configA, configB]         // 2 configs
#     expanded = Enumerable.SelectMany(    // expand to 4
#         configs,
#         c => Enumerable.Repeat(c, 2)    // each repeated twice
#     ).ToList();
#     // expanded = [configA, configA, configB, configB]
#
# C# ANALOGY FOR GQA CONCEPT:
#   Like a static readonly field shared across multiple instances:
#     // MHA: every instance has its own K and V
#     class HeadMHA { private float[] K; private float[] V; }
#
#     // GQA: 2 heads share the same K and V
#     class HeadGroupA { private static float[] sharedK; private static float[] sharedV; }
#     class HeadQ1 : HeadGroupA { ... }  // both use HeadGroupA's K,V
#     class HeadQ2 : HeadGroupA { ... }
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import (
    load_data,
    build_vocab,
    split_data,
    run_training,
    generate_text,
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

N_EMBD    = 128   # Embedding dimension
N_HEADS   = 4     # Number of QUERY heads (unchanged from step_04)
N_KV_HEADS = 2    # Number of KEY/VALUE heads (NEW! half of N_HEADS)
                   # N_HEADS must be divisible by N_KV_HEADS: 4 % 2 == 0 ✓
N_LAYERS  = 3     # Number of transformer blocks
DROPOUT   = 0.1   # Dropout probability


# =============================================================================
# HELPER: repeat_kv
# =============================================================================
def repeat_kv(kv, n_repeat):
    """
    Expand K or V tensor so each Q head has a corresponding K/V head.

    BEFORE: kv has shape (B, T, n_kv_heads, head_size)
    AFTER:  shape is     (B, T, n_heads,    head_size)
              where n_heads = n_kv_heads × n_repeat

    STEP-BY-STEP (using n_kv_heads=2, n_repeat=2 as example):
      Input:    (B, T, 2, head_size)          [K1, K2] for each token
      unsqueeze: (B, T, 2, 1, head_size)      insert new dim at position 3
      expand:   (B, T, 2, 2, head_size)       repeat along dim 3
      reshape:  (B, T, 4, head_size)           flatten dims 2 and 3
      Result:   [K1, K1, K2, K2]               each KV head repeated

    C# ANALOGY:
      // Like Enumerable.SelectMany to expand a grouped sequence:
      configs  = [A, B]          // n_kv_heads=2 items
      expanded = configs         // for each config:
                  .SelectMany(c =>     //   repeat n_repeat times
                      Enumerable.Repeat(c, nRepeat)
                  ).ToList();
      // expanded = [A, A, B, B]  (n_heads=4 items)

    PARAMETERS:
      kv (torch.Tensor): K or V tensor, shape (B, T, n_kv_heads, head_size)
      n_repeat (int): How many times to repeat each KV head.
                      = n_heads // n_kv_heads

    RETURNS:
      torch.Tensor: shape (B, T, n_kv_heads * n_repeat, head_size)
                                   = (B, T, n_heads, head_size)
    """
    if n_repeat == 1:
        # No repetition needed — standard MHA case
        return kv

    B, T, n_kv, hs = kv.shape  # Unpack all four dimensions

    # unsqueeze(3): insert a new dimension at position 3
    # (B, T, n_kv, hs) → (B, T, n_kv, 1, hs)
    # C# analogy: like adding an extra axis to a multidimensional array
    expanded = kv.unsqueeze(3)

    # expand: repeat along the new dimension n_repeat times
    # (B, T, n_kv, 1, hs) → (B, T, n_kv, n_repeat, hs)
    # .expand() doesn't copy memory (efficient!) — it creates a "view"
    # C# analogy: like creating a ReadOnlySpan pointing to the same memory
    expanded = expanded.expand(B, T, n_kv, n_repeat, hs)

    # reshape: merge the n_kv and n_repeat dimensions
    # (B, T, n_kv, n_repeat, hs) → (B, T, n_kv * n_repeat, hs)
    #                                       = (B, T, n_heads, hs)
    # C# analogy: flatten a 2D array's last two dims into one
    return expanded.reshape(B, T, n_kv * n_repeat, hs)


# =============================================================================
# MODULE: GroupQueryAttention
# =============================================================================
class GroupQueryAttention(nn.Module):
    """
    Group-Query Attention: fewer KV heads than Q heads.

    KEY DIFFERENCES FROM STANDARD MHA:
      1. Q projections: n_embd → n_heads × head_size
         K projections: n_embd → n_kv_heads × head_size  ← SMALLER!
         V projections: n_embd → n_kv_heads × head_size  ← SMALLER!

      2. After computing K,V: call repeat_kv to expand to n_heads KV heads

      3. Then: standard scaled dot-product attention per head

    IMPLEMENTATION DETAIL:
      We project Q,K,V to (B, T, n_heads * head_size) or (B, T, n_kv_heads * head_size)
      Then reshape to (B, T, n_heads, head_size) or (B, T, n_kv_heads, head_size)
      Then transpose to (B, n_heads, T, head_size) for batched matmul.
    """

    def __init__(self, n_embd, n_heads, n_kv_heads, block_size, dropout):
        super().__init__()

        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.n_heads    = n_heads                  # 4 Q heads
        self.n_kv_heads = n_kv_heads               # 2 KV heads
        self.n_repeat   = n_heads // n_kv_heads    # 2 — how many Q heads share each KV
        self.head_size  = n_embd // n_heads        # 128 // 4 = 32

        # Q projection: maps to ALL n_heads query vectors
        # Output: n_heads × head_size = 4 × 32 = 128
        self.q_proj = nn.Linear(n_embd, n_heads    * self.head_size, bias=False)

        # K projection: maps to FEWER n_kv_heads key vectors (SMALLER!)
        # Output: n_kv_heads × head_size = 2 × 32 = 64  ← half the size of Q!
        self.k_proj = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)

        # V projection: same smaller size as K
        # Output: n_kv_heads × head_size = 2 × 32 = 64
        self.v_proj = nn.Linear(n_embd, n_kv_heads * self.head_size, bias=False)

        self.scale = math.sqrt(self.head_size)  # sqrt(32) for scaling

        # Causal mask: same as before
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        # Output projection: merge all head outputs back to n_embd
        # Input: n_heads × head_size = 128 (after concatenating all Q heads)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        """
        Group-Query Attention forward pass.

        DATA FLOW:
          x: (B, T, n_embd)

          Q: (B, T, n_heads * head_size)    → reshape → (B, T, n_heads, hs)
             → transpose → (B, n_heads, T, hs)

          K: (B, T, n_kv_heads * head_size) → reshape → (B, T, n_kv_heads, hs)
             → repeat_kv → (B, T, n_heads, hs)
             → transpose → (B, n_heads, T, hs)

          V: same as K

          scores = Q @ K^T / scale: (B, n_heads, T, T)
          weights = softmax(causal_mask(scores)): (B, n_heads, T, T)
          out = weights @ V: (B, n_heads, T, hs)
          → transpose + reshape → (B, T, n_embd)
          → out_proj → (B, T, n_embd)
        """
        B, T, C = x.shape

        # ---- Project Q, K, V ----
        Q = self.q_proj(x)   # (B, T, n_heads * head_size) = (B, T, 128)
        K = self.k_proj(x)   # (B, T, n_kv_heads * head_size) = (B, T, 64) ← SMALLER!
        V = self.v_proj(x)   # (B, T, n_kv_heads * head_size) = (B, T, 64) ← SMALLER!

        # ---- Reshape into per-head vectors ----
        # view(B, T, n_heads, hs): split last dim into (n_heads, head_size)
        # .view() is like .reshape() but requires contiguous memory
        Q = Q.view(B, T, self.n_heads,    self.head_size)  # (B, T, 4, 32)
        K = K.view(B, T, self.n_kv_heads, self.head_size)  # (B, T, 2, 32)
        V = V.view(B, T, self.n_kv_heads, self.head_size)  # (B, T, 2, 32)

        # ---- Expand K and V to match Q head count ----
        # repeat_kv: (B, T, 2, 32) → (B, T, 4, 32)
        # [K1, K2] → [K1, K1, K2, K2] so Q1&Q2 share K1, Q3&Q4 share K2
        K = repeat_kv(K, self.n_repeat)   # (B, T, n_heads, head_size)
        V = repeat_kv(V, self.n_repeat)   # (B, T, n_heads, head_size)

        # ---- Transpose for batched matrix multiplication ----
        # We want (B, n_heads, T, head_size) so that @ operates on (T, T)
        # .transpose(1, 2): swap dims 1 and 2
        # (B, T, n_heads, hs) → (B, n_heads, T, hs)
        # C# analogy: like transposing a 4D array's middle two dimensions
        Q = Q.transpose(1, 2)   # (B, n_heads, T, head_size)
        K = K.transpose(1, 2)   # (B, n_heads, T, head_size)
        V = V.transpose(1, 2)   # (B, n_heads, T, head_size)

        # ---- Scaled dot-product attention ----
        # @ is matrix multiply; K.transpose(-2,-1) swaps last two dims
        # (B, n_heads, T, hs) @ (B, n_heads, hs, T) = (B, n_heads, T, T)
        scores = Q @ K.transpose(-2, -1) / self.scale   # (B, n_heads, T, T)

        # Apply causal mask — prevent attending to future positions
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        # Softmax to get probabilities
        weights = F.softmax(scores, dim=-1)   # (B, n_heads, T, T)
        weights = self.dropout(weights)        # Dropout on attention weights

        # Weighted sum of values
        out = weights @ V   # (B, n_heads, T, head_size)

        # ---- Reassemble multi-head output ----
        # Transpose back: (B, n_heads, T, hs) → (B, T, n_heads, hs)
        out = out.transpose(1, 2)   # (B, T, n_heads, head_size)

        # Make memory contiguous (required before .view())
        # C# analogy: ensuring an array is laid out sequentially in memory
        out = out.contiguous()

        # Reshape to merge head outputs: (B, T, n_heads, hs) → (B, T, n_embd)
        out = out.view(B, T, self.n_heads * self.head_size)   # (B, T, n_embd)

        # Output projection + dropout
        out = self.dropout(self.out_proj(out))   # (B, T, n_embd)

        return out


# =============================================================================
# MODULE: FeedForward and GQABlock (same as step_04)
# =============================================================================
class FeedForward(nn.Module):
    """Same as step_04: expand 4×, ReLU, compress, dropout."""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class GQABlock(nn.Module):
    """
    Transformer block using Group-Query Attention instead of standard MHA.
    Structure: Pre-LN → GQA → residual, Pre-LN → FFN → residual
    Identical to TransformerBlock in step_04, just with GQA inside.
    """
    def __init__(self, n_embd, n_heads, n_kv_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.gqa = GroupQueryAttention(n_embd, n_heads, n_kv_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        # Residual connection: x += attention_output
        x = x + self.gqa(self.ln1(x))   # (B, T, n_embd)
        # Residual connection: x += ffn_output
        x = x + self.ffn(self.ln2(x))   # (B, T, n_embd)
        return x


# =============================================================================
# MODEL: GQAModel
# =============================================================================
class GQAModel(nn.Module):
    """Full GQA-based language model. Same as NanoGPT but with GQA blocks."""

    def __init__(self, vocab_size, n_embd, n_heads, n_kv_heads, n_layers, block_size, dropout):
        super().__init__()

        self.token_embd = nn.Embedding(vocab_size, n_embd)
        self.pos_embd   = nn.Embedding(block_size, n_embd)

        # n_layers GQA blocks (instead of MHA blocks from step_04)
        self.blocks = nn.Sequential(*[
            GQABlock(n_embd, n_heads, n_kv_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        self.final_ln    = nn.LayerNorm(n_embd)
        self.output_proj = nn.Linear(n_embd, vocab_size)

        # Weight initialization (same as NanoGPT)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights: normal(0, 0.02) for Linear and Embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        """Forward pass — identical structure to NanoGPT."""
        B, T = x.shape

        tok = self.token_embd(x)
        pos = self.pos_embd(torch.arange(T, device=x.device))
        h = tok + pos

        h = self.blocks(h)
        h = self.final_ln(h)
        logits = self.output_proj(h)   # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B2, T2, V = logits.shape
            loss = F.cross_entropy(logits.view(B2 * T2, V), targets.view(B2 * T2))

        return logits, loss


# =============================================================================
# HELPER: Compute KV cache memory usage for comparison
# =============================================================================
def compute_kv_memory(n_heads, n_kv_heads, head_size, block_size, dtype_bytes=4):
    """
    Estimate KV cache memory in bytes per layer per sequence.

    Formula:
      KV memory = 2 (K and V) × n_kv_heads × head_size × block_size × dtype_bytes
      (The 2 is for storing BOTH K and V)

    dtype_bytes=4: float32 uses 4 bytes per value
                   float16 uses 2 bytes (modern inference often uses fp16)
    """
    return 2 * n_kv_heads * head_size * block_size * dtype_bytes


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  STEP 5: Group-Query Attention (GQA)                     ║")
    print("║  What's new   : Fewer K,V heads — saves KV cache memory ║")
    print("║  Architecture : Same as step_04 but with GQA blocks     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    print("WHAT THIS STEP TEACHES:")
    print("  - GQA: fewer K,V heads than Q heads → less KV cache memory")
    print("  - repeat_kv: expand shared K,V to match all Q heads")
    print("  - Real models (LLaMA 3, Mistral, Qwen) all use GQA!")
    print("  - Quality trade-off: slightly less expressive, massively less memory")
    print()
    print("C# ANALOGY:")
    print("  MHA: every AttentionHead has its own K and V fields")
    print("  GQA: multiple heads share a 'static readonly' K and V")
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
    head_size = N_EMBD // N_HEADS

    model = GQAModel(
        vocab_size  = vocab_size,
        n_embd      = N_EMBD,
        n_heads     = N_HEADS,
        n_kv_heads  = N_KV_HEADS,
        n_layers    = N_LAYERS,
        block_size  = CONFIG["block_size"],
        dropout     = DROPOUT,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Q heads     : {N_HEADS}  (each with head_size={head_size})")
    print(f"  K,V heads   : {N_KV_HEADS}  (SHARED — only half as many!)")
    print(f"  n_repeat    : {N_HEADS // N_KV_HEADS}  (each KV head serves {N_HEADS // N_KV_HEADS} Q heads)")
    print(f"  Total params: {total_params:,}")

    # ---- Train ----
    print()
    print("[ Training ]")
    final_val_loss = run_training(
        model_name = "GQA",
        model      = model,
        train_data = train_data,
        val_data   = val_data,
        char2idx   = char2idx,
        idx2char   = idx2char,
        config     = CONFIG,
    )

    # ---- Memory comparison ----
    print()
    print("=" * 60)
    print("  MEMORY COMPARISON (KV cache per layer per sequence):")
    print()

    mha_kv = compute_kv_memory(N_HEADS,    N_HEADS,    head_size, CONFIG["block_size"])
    gqa_kv = compute_kv_memory(N_KV_HEADS, N_KV_HEADS, head_size, CONFIG["block_size"])
    saving_pct = (1 - gqa_kv / mha_kv) * 100

    print(f"  Standard MHA (step 04): {mha_kv:,} bytes per layer")
    print(f"  GQA (step 05)         : {gqa_kv:,} bytes per layer")
    print(f"  Memory saving         : {saving_pct:.0f}% reduction!")
    print()
    print(f"  FINAL VAL LOSS: {final_val_loss:.4f}")
    print()
    print("  NOTE: GQA val loss ≈ same as MHA (step_04).")
    print("  GQA trades a tiny bit of quality for a big memory win.")
    print("  At large scale (70B params), this memory saving is crucial!")
    print()
    print("=" * 60)
    print("  Next step: python step_06_kv_cache.py")
    print("  What's next: KV Cache — make text generation 5-10x faster!")
    print("=" * 60)

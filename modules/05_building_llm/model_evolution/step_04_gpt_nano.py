# =============================================================================
# step_04_gpt_nano.py
# =============================================================================
# STEP 4: FULL NANO-GPT — STACKED TRANSFORMER BLOCKS
#
# WHAT'S NEW COMPARED TO STEP 3?
#   Step 3: ONE attention + FFN block (no residuals, no layer norm, no depth)
#   Step 4: THREE stacked blocks + residual connections + layer normalization
#
# THIS IS THE REAL GPT ARCHITECTURE (just tiny):
#   GPT-2 small: 12 blocks, n_embd=768, n_heads=12
#   Our NanoGPT: 3  blocks, n_embd=128, n_heads=4
#   Same structure, massively smaller!
#
# NEW CONCEPT 1: RESIDUAL CONNECTIONS
#
#   WITHOUT residual:       WITH residual (+):
#   X → Block → Y          X ──────────────► +  → Y
#                               └─► Block ──┘
#
#   The residual "skip connection" adds X to the block's output.
#   C# analogy:
#     // Without residual:
#     var y = block.Forward(x);
#     // With residual (shortcut):
#     var y = x + block.Forward(x);   ← += style!
#
#   WHY RESIDUAL CONNECTIONS?
#   In deep networks (many layers), gradients can become very small as they
#   travel backward through layers — the "vanishing gradient" problem.
#
#   With residuals, gradients have a "highway" to flow directly backward:
#   d(loss)/d(X) = d(loss)/d(Y) × (1 + d(Block)/d(X))
#                                    ↑ this "1" ensures gradient ≥ 1
#
#   Without residuals, deep networks often can't be trained at all!
#   C# analogy: like having a direct reference bypass, so a call doesn't have
#   to go through 3 layers of abstraction — the signal gets through.
#
# NEW CONCEPT 2: LAYER NORMALIZATION
#
#   LayerNorm normalizes each token's embedding to have mean=0, std=1.
#   Applied BEFORE each sub-layer (Pre-LN style, used in modern GPT variants).
#
#   WHY NORMALIZE?
#   Embeddings can have wildly varying scales during training.
#   Large values → large gradients → unstable training.
#   LayerNorm keeps values in a controlled range, making training stable.
#
#   C# ANALOGY:
#     // Like normalizing a score:
#     float normalized = (score - mean) / stdDev;
#     // But applied to each token's embedding vector independently
#
#   NOTE: We use "Pre-LN" (normalize BEFORE attention/FFN), which is more
#   stable than the original paper's "Post-LN" (normalize AFTER).
#   LLaMA, GPT-NeoX, and most modern models use Pre-LN.
#
# NEW CONCEPT 3: DROPOUT
#   Randomly zeros out a fraction of neurons during training (p=0.1 = 10%).
#   Prevents overfitting — forces the model not to rely on any single neuron.
#   C# analogy: like randomly disabling 10% of code paths at runtime.
#   At evaluation time, dropout is turned off (no zeros).
#
# FULL ARCHITECTURE DIAGRAM:
#
#   Input tokens (B, T)
#     ↓ token_embedding + positional_embedding
#     ↓
#   ┌─── TransformerBlock 1 ───────────────────────────────┐
#   │  x ──────────────────────────────────────────► +  ──│
#   │      └─► LayerNorm → MultiHeadAttention ────────┘   │
#   │  h ──────────────────────────────────────────► +  ──│
#   │      └─► LayerNorm → FeedForward ──────────────┘   │
#   └─────────────────────────────────────────────────────┘
#     ↓ (same structure repeated 3 times)
#   TransformerBlock 2
#   TransformerBlock 3
#     ↓ Final LayerNorm
#     ↓ Linear(n_embd, vocab_size) → logits
#
# WEIGHT INITIALIZATION:
#   Randomly initialized weights with std=0.02 (normal distribution).
#   This is the standard GPT initialization — keeps activations stable
#   at the start of training.
#   C# analogy: seeding a random number generator with a specific distribution.
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
    "block_size"    : 64,      # Context window
    "batch_size"    : 64,      # Sequences per step
    "max_iters"     : 5000,    # Training steps
    "eval_interval" : 500,     # Print interval
    "lr"            : 3e-4,    # Learning rate
    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
}

N_EMBD   = 128   # Embedding dimension
N_HEADS  = 4     # Number of attention heads
N_LAYERS = 3     # Number of stacked transformer blocks (this is the new hyperparameter!)
DROPOUT  = 0.1   # Dropout probability (10% of neurons zeroed during training)


# =============================================================================
# MODULE: Head (single attention head with dropout)
# =============================================================================
class Head(nn.Module):
    """Single attention head — same as step_03 but with dropout added."""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.q_proj = nn.Linear(n_embd, head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, head_size, bias=False)
        self.scale  = math.sqrt(head_size)

        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        # Dropout on attention weights — randomly zero out some attention connections
        # This prevents the model from over-relying on specific token pairs
        # C# analogy: randomly skip some of the "votes" in a weighted average
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = Q @ K.transpose(-2, -1) / self.scale
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)   # ← new: dropout on attention weights
        out = weights @ V
        return out


# =============================================================================
# MODULE: MultiHeadAttention (same as step_03 but with dropout on projection)
# =============================================================================
class MultiHeadAttention(nn.Module):
    """Multi-head attention with dropout on the output projection."""

    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        self.head_size = n_embd // n_heads
        self.heads = nn.ModuleList([
            Head(n_embd, self.head_size, block_size, dropout)
            for _ in range(n_heads)
        ])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)  # Dropout after projection

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.dropout(self.proj(out))                     # project + dropout
        return out


# =============================================================================
# MODULE: FeedForward (same as step_03 but with dropout)
# =============================================================================
class FeedForward(nn.Module):
    """Position-wise FFN: expand 4×, ReLU, compress back, dropout."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand
            nn.ReLU(),                        # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Compress
            nn.Dropout(dropout),              # Dropout at the end
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# MODULE: TransformerBlock — the core repeated unit
# =============================================================================
class TransformerBlock(nn.Module):
    """
    One Transformer block:
      LayerNorm → MultiHeadAttention → Residual Add
      LayerNorm → FeedForward        → Residual Add

    This unit is STACKED n_layers times in NanoGPT.

    RESIDUAL CONNECTIONS EXPLAINED IN CODE:
      # Without residual:
      x = self.mha(x)
      x = self.ffn(x)

      # With residual (what we actually do):
      x = x + self.mha(self.ln1(x))   ← x += mha_output
      x = x + self.ffn(self.ln2(x))   ← x += ffn_output

    The Pre-LN pattern (normalize BEFORE the sub-layer) is:
      x → LayerNorm → MHA → + x  (residual)
      x → LayerNorm → FFN → + x  (residual)

    C# ANALOGY:
      // Like the += operator: add the transformation to the original
      // Instead of: x = Transform(x)
      // We do:      x += Transform(x)   ← x still contains the original info!
    """

    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()

        # LayerNorm 1: normalize BEFORE attention (Pre-LN style)
        # nn.LayerNorm(n_embd) normalizes each token's n_embd-dim vector
        # C# analogy: normalize a float[] vector to have mean=0 and std=1
        self.ln1 = nn.LayerNorm(n_embd)

        # Multi-head attention
        self.mha = MultiHeadAttention(n_embd, n_heads, block_size, dropout)

        # LayerNorm 2: normalize BEFORE feedforward
        self.ln2 = nn.LayerNorm(n_embd)

        # Feedforward network
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        """
        Forward pass with residual connections.

        RESIDUAL CONNECTION PATTERN:
          output = input + transform(normalize(input))

        This means the input is ALWAYS preserved and combined with
        the transformed version.
        """
        # ---- Attention with residual ----
        # self.ln1(x): normalize x first (Pre-LN)
        # self.mha(...): compute multi-head attention
        # x + ...: add the attention output to the ORIGINAL x (residual)
        x = x + self.mha(self.ln1(x))   # (B, T, n_embd)

        # ---- FeedForward with residual ----
        # Same pattern: normalize → FFN → add residual
        x = x + self.ffn(self.ln2(x))   # (B, T, n_embd)

        return x  # (B, T, n_embd) — same shape as input!


# =============================================================================
# MODEL: NanoGPT — the full model
# =============================================================================
class NanoGPT(nn.Module):
    """
    Full NanoGPT model: n_layers stacked TransformerBlocks.

    STRUCTURE:
      token_embd + pos_embd         ← input processing
      TransformerBlock × n_layers   ← the main body
      Final LayerNorm               ← stabilize before output
      Linear(n_embd, vocab_size)    ← output projection

    THIS IS THE SAME STRUCTURE AS GPT-2!
    GPT-2 small uses: n_layers=12, n_embd=768, n_heads=12, block_size=1024
    We use:           n_layers=3,  n_embd=128, n_heads=4,  block_size=64

    WEIGHT INITIALIZATION:
      We initialize weights using a normal distribution with std=0.02.
      This is the standard from the GPT-2 paper and helps training stability.
      Without proper init, activations can explode or vanish from the start.
      C# analogy: like calling new Random(seed) with a specific distribution.
    """

    def __init__(self, vocab_size, n_embd, n_heads, n_layers, block_size, dropout):
        super().__init__()

        # ---- Input processing ----
        self.token_embd = nn.Embedding(vocab_size, n_embd)   # char → vector
        self.pos_embd   = nn.Embedding(block_size, n_embd)   # position → vector

        # ---- Transformer blocks ----
        # nn.Sequential chains blocks: output of block i → input of block i+1
        # C# analogy: a list of middleware, applied in order
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_heads, block_size, dropout)
            for _ in range(n_layers)  # Create n_layers blocks
        ])

        # ---- Output processing ----
        self.final_ln   = nn.LayerNorm(n_embd)           # Final normalization
        self.output_proj = nn.Linear(n_embd, vocab_size)  # → vocab logits

        # ---- Weight initialization ----
        # apply() calls _init_weights on every submodule recursively
        # C# analogy: like visiting every node in a tree via DFS
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights with small normal distribution (std=0.02).

        WHY 0.02?
          This is the value used in the original GPT paper.
          Small std → small initial activations → stable gradient flow.
          Too large → activations explode → NaN loss immediately.

        C# ANALOGY:
          // Initialize all weights in a neural network layer:
          foreach (var w in layer.Weights) {
              w = GaussianRandom(mean=0, stdDev=0.02f);
          }
        """
        if isinstance(module, nn.Linear):
            # Initialize linear layer weights with normal distribution
            # mean=0: centered at zero (no initial bias)
            # std=0.02: small values (prevents explosion)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # If the linear layer has a bias term, initialize it to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Embedding weights also use the same initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        """
        Full NanoGPT forward pass.

        DATA FLOW:
          x: (B, T)
          → tok + pos: (B, T, n_embd)    ← embed tokens + positions
          → blocks: (B, T, n_embd)        ← 3 transformer blocks
          → final_ln: (B, T, n_embd)      ← normalize
          → output_proj: (B, T, vocab_size) ← logits
        """
        B, T = x.shape

        # ---- Embeddings ----
        tok = self.token_embd(x)                                   # (B, T, n_embd)
        pos = self.pos_embd(torch.arange(T, device=x.device))     # (T, n_embd)
        h = tok + pos                                              # (B, T, n_embd)

        # ---- Run through all transformer blocks ----
        # nn.Sequential.__call__ runs all blocks in sequence
        h = self.blocks(h)   # (B, T, n_embd) after n_layers blocks

        # ---- Final layer norm ----
        h = self.final_ln(h)   # (B, T, n_embd) — stabilize output

        # ---- Output projection ----
        logits = self.output_proj(h)   # (B, T, vocab_size)

        # ---- Loss ----
        if targets is None:
            loss = None
        else:
            B2, T2, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B2 * T2, V),
                targets.view(B2 * T2),
            )

        return logits, loss


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  STEP 4: NanoGPT (Full Transformer)                      ║")
    print("║  What's new   : 3 stacked blocks + residual + LayerNorm ║")
    print("║  Architecture : embed → [Block×3] → LN → linear         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    print("WHAT THIS STEP TEACHES:")
    print("  - Residual connections: x += transform(x) — prevents vanishing gradients")
    print("  - LayerNorm: normalize each token to mean=0, std=1 — stabilizes training")
    print("  - Stacking blocks: deeper = more capacity, but needs residuals to train")
    print("  - Weight initialization: std=0.02 is the GPT standard")
    print("  - Dropout: randomly zero 10% of neurons to prevent overfitting")
    print()
    print("C# ANALOGIES:")
    print("  Residual   : x += block(x)  (the += operator for neural networks)")
    print("  LayerNorm  : normalize float[] to mean=0, std=1")
    print("  n_layers=3 : 3 middleware layers applied in sequence")
    print("  Dropout    : randomly throw away 10% of intermediate values")
    print()
    print(f"ARCHITECTURE SIZES:")
    print(f"  n_embd={N_EMBD}, n_heads={N_HEADS}, n_layers={N_LAYERS}, dropout={DROPOUT}")
    print(f"  Each block: MHA({N_HEADS} heads × {N_EMBD//N_HEADS} head_size) + FFN({N_EMBD}→{4*N_EMBD}→{N_EMBD})")
    print()
    print("EXPECTED RESULT:")
    print("  - Multi-head (step 3) val loss: ~1.5")
    print("  - NanoGPT (step 4)   val loss: ~1.3  (depth helps!)")
    print()
    print("REAL WORLD CONNECTION:")
    print("  This IS GPT-2 small architecture, just with smaller numbers.")
    print("  GPT-2 small: 12 layers, 768 dim, 12 heads = 117M parameters")
    print(f"  NanoGPT:     {N_LAYERS} layers, {N_EMBD} dim, {N_HEADS} heads")
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
    model = NanoGPT(
        vocab_size  = vocab_size,
        n_embd      = N_EMBD,
        n_heads     = N_HEADS,
        n_layers    = N_LAYERS,
        block_size  = CONFIG["block_size"],
        dropout     = DROPOUT,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Blocks       : {N_LAYERS} × TransformerBlock")
    print(f"  Each block   : LN + MHA({N_HEADS}×{N_EMBD//N_HEADS}) + LN + FFN({N_EMBD}→{4*N_EMBD}→{N_EMBD})")
    print(f"  Total params : {total_params:,}")

    # ---- Train ----
    print()
    print("[ Training ]")
    final_val_loss = run_training(
        model_name = "NanoGPT",
        model      = model,
        train_data = train_data,
        val_data   = val_data,
        char2idx   = char2idx,
        idx2char   = idx2char,
        config     = CONFIG,
    )

    # ---- Summary ----
    print("=" * 60)
    print(f"  FINAL VAL LOSS: {final_val_loss:.4f}")
    print()
    print("  WHAT DID WE LEARN?")
    print("  - Residual connections enable training deep networks")
    print("  - LayerNorm stabilizes training significantly")
    print("  - More layers = more capacity = lower loss")
    print()
    print("  THE EVOLUTION SO FAR:")
    print("    Bigram      : ~2.4  (no context, lookup table)")
    print("    MLP         : ~2.0  (context, but fixed position weights)")
    print("    Single-head : ~1.7  (learned relevance scores)")
    print("    Multi-head  : ~1.5  (4 parallel perspectives)")
    print(f"    NanoGPT     : ~1.3  (depth + stability improvements)")
    print()
    print("=" * 60)
    print("  Next step: python step_05_gqa.py")
    print("  What's next: Group-Query Attention — save KV cache memory")
    print("=" * 60)

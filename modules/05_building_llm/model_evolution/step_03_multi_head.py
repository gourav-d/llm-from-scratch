# =============================================================================
# step_03_multi_head.py
# =============================================================================
# STEP 3: MULTI-HEAD ATTENTION + FEEDFORWARD LAYER
#
# WHAT'S NEW COMPARED TO SINGLE HEAD?
#   Step 2: ONE attention head — one perspective, one type of relevance
#   Step 3: FOUR attention heads — four parallel perspectives!
#
# WHY MULTIPLE HEADS?
#   Language has many different types of relationships simultaneously:
#   - Syntactic: "the cat" — subject-verb agreement
#   - Semantic:  "bank" — is it a river bank or financial bank?
#   - Coreference: "Alice said she was tired" — "she" refers to "Alice"
#   - Distance:  rhyme, alliteration, repetition
#
#   ONE attention head can only learn ONE type of relationship at a time.
#   FOUR heads can learn FOUR different relationships simultaneously!
#
# C# ANALOGY:
#   Imagine 4 different LINQ queries running in parallel on the same data,
#   each looking for different patterns, and then merging their results:
#
#     var headA = tokens.Select(t => FindSyntaxRelations(t));     // syntax
#     var headB = tokens.Select(t => FindSemanticLinks(t));       // semantics
#     var headC = tokens.Select(t => FindCoreferents(t));         // references
#     var headD = tokens.Select(t => FindPositionalPatterns(t));  // position
#     var combined = Merge(headA, headB, headC, headD);
#
# ARCHITECTURE:
#
#   tokens
#     ↓ token embedding + positional embedding
#     ↓
#     Multi-Head Attention:
#       Head 1: Q1,K1,V1 → attention output 1 (B, T, 32)  ← head_size=32
#       Head 2: Q2,K2,V2 → attention output 2 (B, T, 32)
#       Head 3: Q3,K3,V3 → attention output 3 (B, T, 32)
#       Head 4: Q4,K4,V4 → attention output 4 (B, T, 32)
#       ↓ concatenate along last dim
#       (B, T, 4×32=128)  ← back to n_embd=128
#       ↓ Linear(128, 128)  "mixing" projection
#       (B, T, 128)
#     ↓
#     FeedForward:
#       Linear(128, 512) → ReLU → Linear(512, 128)
#       ↓
#     ↓ Linear(128, vocab_size) → logits
#
# WHY A FEEDFORWARD AFTER ATTENTION?
#   Attention mixes BETWEEN tokens (token 3 attends to token 1's info).
#   FeedForward transforms WITHIN each token individually.
#   It gives each token a chance to "think" after collecting info.
#
#   The standard transformer uses 4× expansion in the hidden layer:
#     Linear(n_embd, 4*n_embd) → ReLU → Linear(4*n_embd, n_embd)
#   This is called the "position-wise feedforward network" in the paper.
#
# ANALOGY FOR 4x EXPANSION:
#   Think of attention as gathering raw information (meeting with coworkers).
#   FeedForward is processing that info in private (writing up your notes).
#   The 4x expansion gives "brain space" to think — wider = more capacity.
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
    "block_size"    : 64,      # Context window: 64 characters
    "batch_size"    : 64,      # 64 sequences per step
    "max_iters"     : 5000,    # 5000 training steps
    "eval_interval" : 500,     # Print every 500 steps
    "lr"            : 3e-4,    # Learning rate
    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
}

N_EMBD  = 128   # Embedding dimension
N_HEADS = 4     # Number of attention heads
                # head_size = N_EMBD // N_HEADS = 128 // 4 = 32 per head


# =============================================================================
# MODULE: Head (single attention head — same as step_02 but reusable)
# =============================================================================
class Head(nn.Module):
    """
    A single attention head with Q, K, V projections.

    Same as SingleHeadAttention from step_02, but:
    - Renamed to 'Head' for clarity (it's one of many)
    - head_size is smaller: n_embd // n_heads (32 instead of 128)
      because we'll concatenate multiple heads back to n_embd total

    C# ANALOGY:
      Think of Head as an 'expert analyst' that specializes in one
      type of pattern. MultiHeadAttention employs several analysts
      and merges their reports.
    """

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        # Q, K, V projections — smaller now because head_size < n_embd
        self.q_proj = nn.Linear(n_embd, head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, head_size, bias=False)

        self.scale = math.sqrt(head_size)

        # Causal mask: lower triangular matrix
        # Registered as a buffer — not a trainable parameter
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        """Single head forward pass. Returns (B, T, head_size)."""
        B, T, C = x.shape

        Q = self.q_proj(x)   # (B, T, head_size)
        K = self.k_proj(x)   # (B, T, head_size)
        V = self.v_proj(x)   # (B, T, head_size)

        scores = Q @ K.transpose(-2, -1) / self.scale   # (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)   # (B, T, T)
        out = weights @ V                     # (B, T, head_size)

        return out


# =============================================================================
# MODULE: MultiHeadAttention
# =============================================================================
class MultiHeadAttention(nn.Module):
    """
    Run n_heads attention heads in PARALLEL, then merge their outputs.

    STEP-BY-STEP:
      1. Create n_heads independent Head objects
      2. Each Head independently computes attention with head_size = n_embd // n_heads
      3. Run all heads on the same input simultaneously
      4. Concatenate their outputs: n_heads × head_size = n_embd
      5. Apply a linear "projection" layer to mix the concatenated outputs

    WHY THE FINAL PROJECTION?
      The concatenation just stacks the outputs. The projection layer
      (Linear(n_embd, n_embd)) lets the model COMBINE information
      across all heads — letting head 1 influence what head 2 found, etc.
      C# analogy: like a 'reduce' step after a parallel map.

    WHY IS head_size = n_embd // n_heads?
      Each head works in a smaller space (32 dims instead of 128).
      Four heads × 32 dims = 128 dims total = same as one head at 128 dims.
      Same total computation cost, but 4 different "perspectives"!

    ANALOGY:
      4 people each reading the same document looking for different things:
      Person 1: "Who is the subject?"  (syntax)
      Person 2: "What's the topic?"   (semantics)
      Person 3: "What did they say before?" (coreference)
      Person 4: "What's the tone?"    (pragmatics)
      Then they share notes and write a combined summary.
    """

    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()

        # Check that n_embd divides evenly by n_heads
        assert n_embd % n_heads == 0, (
            f"n_embd ({n_embd}) must be divisible by n_heads ({n_heads})"
        )

        self.head_size = n_embd // n_heads  # 128 // 4 = 32

        # Create n_heads Head objects — each independently computable
        # nn.ModuleList is like a List<nn.Module> that PyTorch can track
        # C# analogy: List<AttentionHead> heads = new List<AttentionHead>();
        self.heads = nn.ModuleList([
            Head(n_embd, self.head_size, block_size)
            for _ in range(n_heads)  # Create n_heads identical (but independent) heads
        ])

        # Final projection: mix the concatenated head outputs
        # Input: n_embd (concatenated from all heads)
        # Output: n_embd (back to standard dimension)
        # C# analogy: the 'merge' step that combines all analysts' reports
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        """
        Run all heads in parallel and concatenate outputs.

        DATA FLOW:
          x: (B, T, n_embd)
          head 1 output: (B, T, head_size)
          head 2 output: (B, T, head_size)
          ...
          concatenated: (B, T, n_heads * head_size) = (B, T, n_embd)
          projected: (B, T, n_embd)
        """
        # Run each head independently on the same input x
        # List comprehension collects all outputs
        # C# analogy: heads.Select(h => h.Forward(x)).ToList()
        head_outputs = [head(x) for head in self.heads]
        # head_outputs is a list of n_heads tensors, each (B, T, head_size)

        # Concatenate along the last dimension (dim=-1 = the embedding dimension)
        # torch.cat joins tensors along a given axis
        # C# analogy: like Array.Concat for the last axis of a 3D array
        concatenated = torch.cat(head_outputs, dim=-1)   # (B, T, n_embd)

        # Apply projection to mix information across heads
        out = self.proj(concatenated)   # (B, T, n_embd)

        return out


# =============================================================================
# MODULE: FeedForward
# =============================================================================
class FeedForward(nn.Module):
    """
    Position-wise feedforward network.

    WHAT IS 'POSITION-WISE'?
      Each token is processed INDEPENDENTLY by the same MLP.
      Token at position 3 goes through FFN separately from token at position 7.
      Same weights, applied independently per token.

      CONTRAST WITH ATTENTION:
        Attention: mixes BETWEEN tokens (cross-position communication)
        FeedForward: transforms WITHIN each token (individual processing)

    THE 4× EXPANSION:
      Hidden dimension = 4 × n_embd (e.g., 4 × 128 = 512)
      This is from the original "Attention is All You Need" paper.
      Wider hidden layer = more processing capacity.

      C# ANALOGY:
        class FeedForward {
            float[] Process(float[] x) {
                var hidden = ReLU(x * W1 + b1);  // expand to 4× size
                return hidden * W2 + b2;          // compress back
            }
        }
    """

    def __init__(self, n_embd):
        super().__init__()

        # The "expand then compress" architecture
        # Linear(n_embd, 4*n_embd): expand (more processing capacity)
        # ReLU: non-linearity (required to learn non-linear patterns)
        # Linear(4*n_embd, n_embd): compress back to original size
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 128 → 512 (expand)
            nn.ReLU(),                        # non-linearity
            nn.Linear(4 * n_embd, n_embd),  # 512 → 128 (compress)
        )

    def forward(self, x):
        """
        Apply feedforward to each token independently.

        PARAMETERS:
          x: (B, T, n_embd)

        RETURNS:
          (B, T, n_embd) — same shape, transformed
        """
        # self.net applied to (B, T, n_embd):
        # PyTorch applies the linear layers to the last dimension
        # automatically (broadcasts over B and T dimensions)
        # C# analogy: like applying a function to every element of a 2D array
        return self.net(x)   # (B, T, n_embd)


# =============================================================================
# MODEL: MultiHeadLM
# =============================================================================
class MultiHeadLM(nn.Module):
    """
    Language model with multi-head attention AND feedforward layer.

    This is ONE "transformer block" (without residual connections + layer norm,
    which come in the next step). It's almost a real transformer!
    """

    def __init__(self, vocab_size, n_embd, n_heads, block_size):
        super().__init__()

        # Embeddings: same as step_02
        self.token_embd = nn.Embedding(vocab_size, n_embd)
        self.pos_embd   = nn.Embedding(block_size, n_embd)

        # Multi-head attention: the new addition!
        self.mha = MultiHeadAttention(n_embd, n_heads, block_size)

        # Feedforward: also new!
        self.ffn = FeedForward(n_embd)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        """
        Forward pass through multi-head attention + feedforward.

        DATA FLOW:
          x: (B, T)
          → embed + pos: (B, T, n_embd)
          → multi-head attention: (B, T, n_embd)
          → feedforward: (B, T, n_embd)
          → output projection: (B, T, vocab_size)
        """
        B, T = x.shape

        # Token + positional embeddings
        tok = self.token_embd(x)                                  # (B, T, n_embd)
        pos = self.pos_embd(torch.arange(T, device=x.device))    # (T, n_embd)
        h = tok + pos                                             # (B, T, n_embd)

        # Multi-head attention: each token gathers info from relevant past tokens
        h = self.mha(h)    # (B, T, n_embd)

        # Feedforward: each token independently processes what it gathered
        h = self.ffn(h)    # (B, T, n_embd)

        # Project to vocabulary scores
        logits = self.output_proj(h)   # (B, T, vocab_size)

        # Compute loss
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
    print("║  STEP 3: Multi-Head Attention + FeedForward              ║")
    print("║  What's new   : 4 parallel attention heads + FFN        ║")
    print("║  Architecture : embed → MHA → FFN → linear              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    print("WHAT THIS STEP TEACHES:")
    print("  - Multi-head attention: 4 different 'perspectives' simultaneously")
    print("  - Each head sees the same data but attends differently")
    print("  - FeedForward: per-token processing after attention (4× expand/compress)")
    print("  - These two components together = the core of a Transformer block")
    print()
    print("C# ANALOGY:")
    print("  4 parallel LINQ queries on same data → merge results → process each item")
    print()
    print(f"  Head size per head: {N_EMBD} // {N_HEADS} = {N_EMBD // N_HEADS} dims")
    print(f"  After concat: {N_HEADS} × {N_EMBD // N_HEADS} = {N_EMBD} dims (back to n_embd)")
    print(f"  FFN: {N_EMBD} → {4 * N_EMBD} → {N_EMBD}  (expand 4×, then compress)")
    print()
    print("EXPECTED RESULT:")
    print("  - Single-head val loss: ~1.7")
    print("  - Multi-head val loss : ~1.5  (4 types of patterns learned)")
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
    model = MultiHeadLM(
        vocab_size  = vocab_size,
        n_embd      = N_EMBD,
        n_heads     = N_HEADS,
        block_size  = CONFIG["block_size"],
    )

    print(f"  Token embedding  : ({vocab_size}, {N_EMBD})")
    print(f"  Position embedding: ({CONFIG['block_size']}, {N_EMBD})")
    print(f"  Multi-head attention: {N_HEADS} heads × head_size={N_EMBD // N_HEADS}")
    print(f"  FeedForward      : {N_EMBD} → {4*N_EMBD} → {N_EMBD}")

    # ---- Train ----
    print()
    print("[ Training ]")
    final_val_loss = run_training(
        model_name = "MultiHeadAttention",
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
    print("  - Multiple heads capture different relationship types")
    print("  - FeedForward layer improves per-token understanding")
    print("  - This is basically ONE Transformer block!")
    print()
    print("  WHAT'S STILL MISSING?")
    print("  - Residual connections (+ shortcuts to prevent vanishing gradients)")
    print("  - Layer normalization (stabilizes training)")
    print("  - Stacking multiple blocks (deeper = more capacity)")
    print("  These arrive in step_04_gpt_nano.py!")
    print()
    print("=" * 60)
    print("  Next step: python step_04_gpt_nano.py")
    print("  What's next: Full NanoGPT — 3 stacked blocks + residuals + LayerNorm")
    print("=" * 60)

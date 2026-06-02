# =============================================================================
# step_02_single_attention.py
# =============================================================================
# STEP 2: SINGLE-HEAD ATTENTION — THE KEY CONCEPTUAL LEAP
#
# WHAT'S THE PROBLEM WITH MLP?
#   MLP concatenates all 16 embeddings and treats them equally.
#   It can't easily learn "the word 3 positions ago is very relevant here"
#   because all positions are mixed together in a flat vector.
#
# WHAT IS ATTENTION?
#   Attention is a mechanism that lets each token CHOOSE which other tokens
#   to focus on, based on the content of the tokens — not just their position.
#
#   Example: "The cat sat on the mat"
#   When predicting what comes after "sat", the model should attend to:
#   - "cat" (subject — who is doing the sitting?)
#   - "on" (preposition following "sat")
#   - less to "The" (article, less relevant)
#
#   Attention LEARNS these relevance scores during training!
#
# THE QUERY-KEY-VALUE FRAMEWORK:
#
#   Think of a library system:
#   - QUERY (Q): What I'm looking for — "books about cooking"
#   - KEY   (K): The catalog tags on each book — "cookbook", "fiction", "history"
#   - VALUE (V): The actual content of the book — what you get when you check it out
#
#   Attention:
#   1. Compute Q, K, V for each token (using learned linear projections)
#   2. Score each (Q, K) pair: how relevant is token K to query Q?
#      score = Q · K^T / sqrt(head_size)
#   3. Normalize scores → probabilities (softmax)
#   4. Weighted sum of Values using those probabilities
#   5. Output = "information gathered by attending to relevant tokens"
#
# C# ANALOGY:
#   Attention ≈ a weighted dictionary lookup:
#     Dictionary<string, string> library = { "cooking": "recipe data", ... }
#     float[] relevance = library.Keys.Select(k => Similarity(query, k)).ToArray()
#     string result = library.Values.Zip(relevance, (v, r) => v * r).Sum()
#   Except the keys, values, and query all come from learned linear projections.
#
# ARCHITECTURE:
#
#   tokens
#     ↓ Embedding(vocab_size, n_embd=128)     — each char → 128-dim vector
#     ↓ + PositionalEmbedding(block_size, 128) — add position information
#     ↓
#     SingleHeadAttention:
#       Q = x @ W_Q   (x: B,T,128 → Q: B,T,128)   — "what am I looking for?"
#       K = x @ W_K   (x: B,T,128 → K: B,T,128)   — "what do I offer?"
#       V = x @ W_V   (x: B,T,128 → V: B,T,128)   — "what info do I provide?"
#
#       weights = Q @ K^T / sqrt(128)   (B,T,T)   — relevance scores
#       CAUSAL MASK: set future positions to -inf   — can't peek at future!
#       weights = softmax(weights, dim=-1)          — normalize to probabilities
#       out = weights @ V                (B,T,128) — weighted sum of values
#     ↓
#     Linear(128, vocab_size) → logits
#
# VISUAL DIAGRAM:
#
#   Token 1: "T"  ──Q1──┐
#   Token 2: "h"  ──Q2──┤  × [K1, K2, K3, K4]ᵀ
#   Token 3: "e"  ──Q3──┤  = attention weight matrix (4×4)
#   Token 4: " "  ──Q4──┘       ↓ (after causal mask + softmax)
#                          ┌─────────────┐
#                          │0.8  0.0  0.0│  ← token 1 mostly attends to itself
#                          │0.5  0.5  0.0│  ← token 2 attends to 1 and 2
#                          │0.3  0.3  0.4│  ← token 3 attends to all three
#                          └─────────────┘
#                                ↓ × [V1, V2, V3, V4]
#                          weighted blend of values → output
#
# WHY CAUSAL MASK?
#   We're doing CAUSAL language modeling: predict the NEXT token.
#   Token at position t should NOT see tokens at positions t+1, t+2, ...
#   (That would be cheating — the model could just copy the next token!)
#   The causal mask sets future positions to -infinity BEFORE softmax,
#   so softmax(−∞) = 0 (zero attention to future tokens).
#
# WHY DIVIDE BY sqrt(head_size)?
#   Q and K are vectors of size head_size. Their dot product (Q·K) can be
#   very large when head_size is large, causing softmax to saturate
#   (outputs become nearly 0 or 1, very sharp — gradients vanish).
#   Dividing by sqrt(head_size) keeps the dot products in a reasonable range.
#   This is called "scaled dot-product attention".
# =============================================================================

import math             # For math.sqrt — needed for scaling attention
import torch            # PyTorch
import torch.nn as nn   # Neural network layers
import torch.nn.functional as F  # Functions like softmax, cross_entropy
                                  # C# analogy: static utility methods

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
    "block_size"    : 64,      # Context window: model sees last 64 characters
                               # Larger than MLP (16) — attention handles long context better

    "batch_size"    : 64,      # 64 sequences per training step

    "max_iters"     : 5000,    # 5000 steps — attention needs more training than bigram

    "eval_interval" : 500,     # Print loss every 500 steps

    "lr"            : 3e-4,    # Learning rate 3e-4 — standard for transformers

    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
}

N_EMBD = 128    # Embedding dimension — increased from 64 (MLP) to 128
                # More dimensions = more expressiveness
                # C# analogy: each token represented as float[128]


# =============================================================================
# MODULE: SingleHeadAttention
# =============================================================================
class SingleHeadAttention(nn.Module):
    """
    One attention head: Q, K, V projections + scaled dot-product attention.

    This is THE core building block of all modern LLMs (GPT, LLaMA, Gemma, etc.)
    Understanding this class = understanding 80% of what makes LLMs work.

    C# ANALOGY:
      Imagine a class that:
      1. Transforms input X into three different "views" (Q, K, V)
         using three different weight matrices
      2. Computes relevance scores between all pairs of Q-K
      3. Uses scores to create a weighted blend of V
      4. Returns the result

      It's like three LINQ projections of the same data,
      combined using a learned relevance weighting.
    """

    def __init__(self, n_embd, head_size, block_size):
        """
        Initialize attention head parameters.

        PARAMETERS:
          n_embd    (int): Input embedding dimension (128).
          head_size (int): Dimension for Q, K, V projections (also 128 for single head).
          block_size (int): Max sequence length (for causal mask).
        """
        super().__init__()

        # Linear projections for Query, Key, Value
        # Each is a matrix of shape (n_embd, head_size)
        # bias=False: standard in transformer attention (no bias term)
        # C# analogy: float[n_embd][head_size] weight matrices

        # Q: "What am I looking for?" — one per token
        self.q_proj = nn.Linear(n_embd, head_size, bias=False)

        # K: "What do I offer to others?" — one per token
        self.k_proj = nn.Linear(n_embd, head_size, bias=False)

        # V: "What information do I provide if attended to?" — one per token
        self.v_proj = nn.Linear(n_embd, head_size, bias=False)

        # Scaling factor: prevents large dot products from saturating softmax
        # sqrt(head_size) is a constant, so we precompute it
        self.scale = math.sqrt(head_size)

        # CAUSAL MASK — prevent attending to future tokens
        # torch.tril creates a lower-triangular matrix:
        #
        #   [[1, 0, 0, 0],   ← token 0 can only see itself
        #    [1, 1, 0, 0],   ← token 1 can see 0 and 1
        #    [1, 1, 1, 0],   ← token 2 can see 0, 1, and 2
        #    [1, 1, 1, 1]]   ← token 3 can see everyone
        #
        # register_buffer: saves this tensor as part of the model but
        # it's NOT a trainable parameter (not updated by optimizer).
        # C# analogy: like a static readonly field on a class
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))  # (T, T) lower-triangle
        )

    def forward(self, x):
        """
        Compute single-head attention.

        PARAMETERS:
          x (torch.Tensor): Input, shape (B, T, n_embd)

        RETURNS:
          torch.Tensor: Output, shape (B, T, head_size)
        """
        B, T, C = x.shape  # B=batch, T=sequence length, C=n_embd (channels)

        # ---- Step 1: Project to Q, K, V ----
        # Each linear layer: (B, T, C) × (C, head_size) → (B, T, head_size)
        # x @ W_Q = for each token, compute its "query" vector
        Q = self.q_proj(x)   # (B, T, head_size) — queries: what am I looking for?
        K = self.k_proj(x)   # (B, T, head_size) — keys:    what do I offer?
        V = self.v_proj(x)   # (B, T, head_size) — values:  what info do I give?

        # ---- Step 2: Compute attention scores ----
        # Q @ K^T: dot product between each query and each key
        # Q shape: (B, T, head_size)
        # K.transpose(-2, -1) shape: (B, head_size, T) — flip last two dims
        # Result: (B, T, T) — T×T matrix of relevance scores
        #
        # scores[b, i, j] = "how much does token i attend to token j?"
        # C# analogy: dot product between two vectors = their similarity
        scores = Q @ K.transpose(-2, -1)   # (B, T, T)
        scores = scores / self.scale        # Scale: divide by sqrt(head_size)

        # ---- Step 3: Apply causal mask ----
        # self.mask[:T, :T] gets the top-left T×T portion of the mask
        # Where mask == 0 (upper triangle = future positions):
        #   set scores to -infinity → after softmax → probability = 0
        # C# analogy: setting irrelevant scores to float.NegativeInfinity
        #             before taking the weighted average
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,  # Condition: where mask is 0 (future positions)
            float("-inf")             # Value to fill in: -infinity
        )   # (B, T, T) — future positions are now -inf

        # ---- Step 4: Softmax → probabilities ----
        # softmax converts scores to probabilities (sum to 1 per row)
        # dim=-1: softmax over the last dimension (over all keys for each query)
        # C# analogy: normalize a list so all elements sum to 1
        weights = F.softmax(scores, dim=-1)   # (B, T, T), each row sums to 1

        # ---- Step 5: Weighted sum of Values ----
        # weights: (B, T, T) — attention probabilities
        # V:       (B, T, head_size) — value vectors
        # out = weights @ V: for each token, take a weighted average of all values
        # High weight → that token's V contributes more to the output
        # C# analogy: Enumerable.Zip(weights, values, (w, v) => w * v).Sum()
        out = weights @ V   # (B, T, head_size)

        return out  # (B, T, head_size) — attention output for each token


# =============================================================================
# MODEL: SingleHeadLM
# =============================================================================
class SingleHeadLM(nn.Module):
    """
    Full language model using single-head attention.

    WHAT IS POSITIONAL EMBEDDING?
      Unlike MLP where position was implicit (which slot in the concat vector),
      attention operates on all tokens in parallel — it doesn't inherently
      know which token came first, second, etc.

      We fix this by ADDING a learned position vector to each token's embedding.
      Token at position 0 gets pos_embd[0] added.
      Token at position 1 gets pos_embd[1] added.

      The model learns these position vectors just like it learns word embeddings.

      C# ANALOGY: Like adding a position "tag" to each element before processing:
        var taggedTokens = tokens.Select((t, i) => tokenEmbed[t] + posEmbed[i]);
    """

    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()

        # Token embedding: maps each character → n_embd-dim vector
        # C# analogy: float[vocab_size][n_embd] lookupTable
        self.token_embd = nn.Embedding(vocab_size, n_embd)

        # Positional embedding: maps each position 0..block_size-1 → n_embd-dim vector
        # Learned — not sinusoidal. The model figures out what each position means.
        # C# analogy: float[block_size][n_embd] positionTable
        self.pos_embd = nn.Embedding(block_size, n_embd)

        # The single attention head — the new addition in this step!
        # head_size = n_embd for single head (uses full embedding dimension)
        self.attention = SingleHeadAttention(
            n_embd     = n_embd,
            head_size  = n_embd,     # Single head uses the full n_embd dimension
            block_size = block_size,
        )

        # Output projection: attention output → vocab logits
        # Maps from n_embd → vocab_size
        # C# analogy: the final linear layer in a classifier
        self.output_proj = nn.Linear(n_embd, vocab_size)

    def forward(self, x, targets=None):
        """
        Forward pass.

        DATA FLOW:
          x: (B, T)
          → token_embd(x): (B, T, n_embd)
          + pos_embd(positions): (B, T, n_embd)    ← ADDED, not concatenated
          → attention(combined): (B, T, n_embd)
          → output_proj: (B, T, vocab_size)         = logits
        """
        B, T = x.shape

        # ---- Token embeddings ----
        tok = self.token_embd(x)   # (B, T, n_embd) — what each character IS

        # ---- Positional embeddings ----
        # torch.arange(T) creates [0, 1, 2, ..., T-1] on the correct device
        # C# analogy: Enumerable.Range(0, T).ToArray()
        positions = torch.arange(T, device=x.device)       # (T,)
        pos = self.pos_embd(positions)                      # (T, n_embd)
        # pos is (T, n_embd) — PyTorch broadcasts this to (B, T, n_embd)
        # (adds same position embeddings to every sequence in the batch)

        # ---- Combine token + position info ----
        # ADDITION (not concatenation!) — both are n_embd dimensional
        # This is the standard approach: token meaning + position meaning = input
        # C# analogy: vectorA + vectorB (element-wise addition)
        x_combined = tok + pos   # (B, T, n_embd)

        # ---- Apply attention ----
        # Each token gathers information from relevant past tokens
        attn_out = self.attention(x_combined)   # (B, T, n_embd)

        # ---- Project to vocabulary ----
        logits = self.output_proj(attn_out)     # (B, T, vocab_size)

        # ---- Compute loss ----
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
    print("║  STEP 2: Single-Head Attention                           ║")
    print("║  What's new   : Attention mechanism (Q, K, V)           ║")
    print("║  Architecture : embed + pos → attention → linear        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    print("WHAT THIS STEP TEACHES:")
    print("  - Attention: each token chooses which other tokens to focus on")
    print("  - Query/Key/Value: the three projections in attention")
    print("  - Causal mask: prevent peeking at future tokens")
    print("  - Positional embedding: tell the model about token order")
    print("  - Scaled dot-product: why we divide by sqrt(head_size)")
    print()
    print("THE KEY INSIGHT:")
    print("  MLP: 'here are 16 tokens, figure it out (all weighted equally)'")
    print("  Attention: 'here are 64 tokens; LEARN which ones to focus on'")
    print("  Attention can learn: 'subject is 10 tokens back, that's what matters now'")
    print()
    print("EXPECTED RESULT:")
    print("  - MLP val loss   : ~2.0")
    print("  - Attention loss : ~1.7  (attention learns WHAT to attend to)")
    print()

    # ---- Prepare data ----
    print("[ Loading data ]")
    text = load_data(max_chars=10_000_000)

    print()
    print("[ Building vocabulary ]")
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(idx2char)

    print()
    print("[ Splitting data ]")
    train_data, val_data = split_data(text, char2idx)

    # ---- Create model ----
    print()
    print("[ Creating model ]")
    model = SingleHeadLM(
        vocab_size  = vocab_size,
        n_embd      = N_EMBD,
        block_size  = CONFIG["block_size"],
    )

    print(f"  Token embedding  : ({vocab_size}, {N_EMBD})")
    print(f"  Position embedding: ({CONFIG['block_size']}, {N_EMBD})")
    print(f"  Attention head   : Q,K,V each ({N_EMBD}, {N_EMBD})")
    print(f"  Output projection: ({N_EMBD}, {vocab_size})")

    # ---- Train ----
    print()
    print("[ Training ]")
    final_val_loss = run_training(
        model_name = "SingleHeadAttention",
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
    print("  - A single attention head already beats MLP significantly")
    print("  - The model learns WHICH tokens are relevant — not just position")
    print("  - But one head can only learn one 'type' of relevance")
    print("  - Real patterns in language need multiple parallel heads!")
    print()
    print("=" * 60)
    print("  Next step: python step_03_multi_head.py")
    print("  What's next: 4 attention heads in parallel + feedforward layer")
    print("=" * 60)

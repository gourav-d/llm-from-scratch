# =============================================================================
# step_01_mlp_context.py
# =============================================================================
# STEP 1: MLP WITH CONTEXT WINDOW
#
# WHAT'S NEW COMPARED TO BIGRAM?
#   Bigram (step 0): looks at 1 character, predicts next
#   MLP   (step 1): looks at last 16 characters, predicts next
#
#   That's the ONLY change — but it makes a BIG difference!
#   Context is what separates "t h" from "the" in language understanding.
#
# WHAT IS AN MLP?
#   MLP = Multi-Layer Perceptron = a classic neural network with:
#   - Input layer  : receives the data
#   - Hidden layer : does the "thinking" (learns patterns)
#   - Output layer : produces predictions
#
#   C# ANALOGY:
#   Think of it as a chain of matrix multiplications + activation functions:
#     input vector
#       └─► Linear transform (matrix multiply)  ← like y = Ax + b
#       └─► ReLU activation                     ← like Math.Max(0, x)
#       └─► Linear transform                    ← y = Ax + b again
#       └─► output (logits)
#
# ARCHITECTURE:
#
#   [c1, c2, c3, ..., c16]          16 characters of context
#      ↓ each embedded via nn.Embedding(vocab_size, 64)
#   [e1, e2, e3, ..., e16]          16 embedding vectors, each size 64
#      ↓ concatenate all into one long vector
#   flat vector (16 × 64 = 1024)    all 16 embeddings joined end-to-end
#      ↓ Linear(1024, 256) + ReLU   first hidden layer: compress to 256
#      ↓ Linear(256, vocab_size)    output layer: scores for each possible next char
#      ↓ logits → next token
#
# VISUAL: WHY CONCATENATE EMBEDDINGS?
#
#   Character:  'T'    'h'    'e'    ' '   ... (16 chars total)
#   Embedding:  [e1]   [e2]   [e3]   [e4]  ... (each is a 64-dim vector)
#   Concat:     [e1 | e2 | e3 | e4 | ...]  = one big 1024-dim vector
#               └──────────────────────────┘
#               The MLP learns to combine all 16 embeddings to predict next char
#
# C# ANALOGY FOR THE WHOLE THING:
#   Imagine a function:
#     char PredictNextChar(char[] last16Chars)
#   Internally it does matrix multiplications on embedded representations.
#   That's exactly what this model does!
#
# EXPECTED IMPROVEMENT:
#   Bigram val loss: ~2.4-2.5
#   MLP    val loss: ~2.0-2.1
#   The 0.3-0.5 drop comes entirely from having more context!
# =============================================================================

import torch            # PyTorch framework
import torch.nn as nn   # Neural network layers

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
    "block_size"    : 16,      # Context window: model sees LAST 16 characters
                               # Increased from 8 in bigram (bigram ignores context anyway)
                               # 16 chars = enough to see "the quick brown fox"

    "batch_size"    : 64,      # 64 sequences per training step

    "max_iters"     : 5000,    # More iterations than bigram — MLP needs more training
                               # More parameters = more steps to converge

    "eval_interval" : 500,     # Print loss every 500 steps

    "lr"            : 3e-4,    # Learning rate = 0.0003
                               # Smaller than bigram (1e-2) because MLP is more complex
                               # Too-large lr causes instability in deep networks

    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
}

# MLP-specific architecture hyperparameters (not in CONFIG to keep CONFIG clean)
N_EMBD = 64     # Embedding dimension: each character is represented as a 64-dim vector
                # C# analogy: each char maps to a float[64] feature vector
                # Larger = more expressive, slower. 64 is a good learning size.

HIDDEN = 256    # Hidden layer size: the "thinking" layer has 256 neurons
                # C# analogy: the intermediate layer in a neural net pipeline


# =============================================================================
# MODEL: MLPModel
# =============================================================================
class MLPModel(nn.Module):
    """
    MLP language model with context window.

    WHAT IS nn.Sequential?
      nn.Sequential is a container that runs layers one after another.
      You give it a list of layers, and it chains them: output of layer N
      becomes input to layer N+1.
      C# analogy: like a LINQ pipeline — .Select().Where().OrderBy()
                  but for neural network layers.

    WHAT IS nn.Embedding?
      Same as in bigram, but different shape:
      - Bigram: Embedding(vocab_size, vocab_size) — output = logits
      - MLP:    Embedding(vocab_size, N_EMBD)    — output = feature vector

      The MLP embedding maps each character to a learned feature vector.
      Instead of "what comes next", it asks "what IS this character conceptually?"
      C# analogy: float[vocabSize][N_EMBD] lookup table.

    WHAT IS nn.Linear?
      A fully-connected layer: output = input × weight_matrix + bias
      Linear(in_features, out_features) creates a weight matrix of shape
      (in_features, out_features).
      C# analogy: matrix multiplication + vector addition.
        float[] output = (float[][])weights * input + bias;

    WHAT IS nn.ReLU?
      ReLU = Rectified Linear Unit. The simplest activation function.
      Formula: ReLU(x) = max(0, x)
      - If x > 0: return x unchanged
      - If x <= 0: return 0

      WHY DO WE NEED ACTIVATION FUNCTIONS?
      Without activation functions, stacking multiple Linear layers is
      mathematically identical to ONE linear layer (can be collapsed).
      Activation functions add NON-LINEARITY — the ability to learn
      curved patterns, not just straight lines.
      C# analogy: Math.Max(0.0f, x)
    """

    def __init__(self, vocab_size, n_embd, block_size, hidden):
        """
        Build the MLP model layers.

        PARAMETERS:
          vocab_size (int): Number of unique characters.
          n_embd (int): Embedding dimension (64).
          block_size (int): Number of past characters to use (16).
          hidden (int): Hidden layer size (256).
        """
        super().__init__()  # Required: initialize the base nn.Module

        # --- Token embedding ---
        # Maps each character index to a learnable vector of size n_embd
        # Shape: (vocab_size, n_embd) = e.g., (100, 64)
        # C# analogy: float[vocabSize][nEmbd] lookupTable;
        self.embedding = nn.Embedding(vocab_size, n_embd)

        # --- MLP layers (the "feedforward network") ---
        # We use nn.Sequential to chain them automatically.
        #
        # Input size: block_size * n_embd
        #   = 16 positions × 64 dims per position = 1024 values
        #   WHY? We concatenate all 16 character embeddings into one flat vector
        #
        # Hidden layer: 1024 -> 256 (compress + transform)
        # ReLU: non-linearity (can't learn without it!)
        # Output layer: 256 -> vocab_size (one score per possible next char)
        self.mlp = nn.Sequential(
            nn.Linear(block_size * n_embd, hidden),  # 1024 → 256
            nn.ReLU(),                                # non-linearity
            nn.Linear(hidden, vocab_size),            # 256 → vocab_size
        )

    def forward(self, x, targets=None):
        """
        Forward pass for the MLP model.

        DATA FLOW:
          x: (B, T)               — B sequences, each T tokens long
          embed: (B, T, n_embd)   — each token replaced by its embedding vector
          flat: (B, T*n_embd)     — all embeddings concatenated into one long vector
          logits: (B, vocab_size) — BUT we only predict from the LAST token's context
          Wait — actually: (B, T, vocab_size) for consistency with training framework

        IMPORTANT NOTE ON BIGRAM VS MLP:
          Bigram outputs (B, T, V) — one prediction per position
          MLP also outputs (B, T, V) — but position t uses tokens 0..t as context

        PARAMETERS:
          x (torch.LongTensor): Token indices, shape (B, T)
          targets (torch.LongTensor or None): Target indices, shape (B, T)

        RETURNS:
          (logits, loss) where logits: (B, T, vocab_size)
        """
        B, T = x.shape  # Unpack batch size B and sequence length T

        # ---- Step 1: Embed each token ----
        # x: (B, T) → embed: (B, T, n_embd)
        # Each integer token index becomes a dense float vector
        embed = self.embedding(x)  # (B, T, N_EMBD) = e.g., (64, 16, 64)

        # ---- Step 2: Process each position with context ----
        # For position t, we want to use tokens 0..t as context.
        # But the MLP expects a fixed-size input (block_size * n_embd).
        # We handle this by predicting from a sliding window.
        #
        # For position t (0-indexed):
        #   context_tokens = x[:, max(0, t-block_size+1) : t+1]
        # BUT: since block_size=T in our batch construction,
        # we can just use ALL positions' embeddings.
        #
        # SIMPLE APPROACH used here:
        # For each of the T positions, concatenate embeddings up to that position.
        # We pad with the first embedding if not enough context.
        # This is simpler than complex masking for a teaching example.

        # Pad embed so every position has block_size context
        # We'll use the embedding of position 0 as padding for early positions
        # C# analogy: like prepending a default value to an array

        # Collect predictions for all T positions
        all_logits = []  # Will collect (B, vocab_size) for each time step

        for t in range(T):
            # Gather the last block_size embeddings ending at position t
            # If t < block_size, we pad with zeros at the start
            start = t - T + 1  # Relative start in the current window

            # Get embeddings for positions max(0, t+1-block_size) ... t
            # Number of available context tokens
            n_context = min(t + 1, T)  # How many tokens we have up to position t

            if n_context < T:
                # Not enough context yet — pad with zeros
                pad_size = T - n_context
                # embed[:, :n_context, :] = embeddings we have
                ctx_embed = embed[:, :n_context, :]  # (B, n_context, n_embd)
                # Create zero padding: (B, pad_size, n_embd)
                padding = torch.zeros(B, pad_size, N_EMBD, device=x.device)
                # Concatenate padding + context
                ctx_embed = torch.cat([padding, ctx_embed], dim=1)  # (B, T, n_embd)
            else:
                # We have enough context — use the last T embeddings
                ctx_embed = embed[:, t+1-T : t+1, :]  # (B, T, n_embd)

            # Flatten: (B, T, n_embd) → (B, T*n_embd)
            # .reshape(-1, T*N_EMBD): -1 means "infer this dimension"
            # C# analogy: flatten a 2D array into 1D
            flat = ctx_embed.reshape(B, T * N_EMBD)  # (B, T*n_embd)

            # Run through MLP
            logit_t = self.mlp(flat)  # (B, vocab_size)
            all_logits.append(logit_t)

        # Stack: list of T tensors of shape (B, V) → (B, T, V)
        # torch.stack creates a new dimension; dim=1 puts T between B and V
        logits = torch.stack(all_logits, dim=1)  # (B, T, vocab_size)

        # ---- Compute loss ----
        if targets is None:
            loss = None
        else:
            B2, T2, V = logits.shape
            # Flatten for cross-entropy: (B*T, V) vs (B*T,)
            loss = nn.functional.cross_entropy(
                logits.view(B2 * T2, V),   # predicted logits
                targets.view(B2 * T2),     # true next tokens
            )

        return logits, loss


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  STEP 1: MLP with Context Window                         ║")
    print("║  What's new   : Model sees last 16 characters           ║")
    print("║  Architecture : embed → concat → Linear → ReLU → Linear ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    print("WHAT THIS STEP TEACHES:")
    print("  - Context matters: seeing more characters = better predictions")
    print("  - Embeddings: each character is a learnable float vector")
    print("  - MLP: stack of linear layers with ReLU activation")
    print("  - Flattening: concatenate embeddings into one big input vector")
    print()
    print("C# ANALOGY:")
    print("  PredictNextChar(char[] last16Chars)")
    print("  → embed each char to float[64]")
    print("  → concatenate all 16 into float[1024]")
    print("  → matrix multiply twice → output scores for vocab_size chars")
    print()
    print("EXPECTED RESULT:")
    print("  - Bigram val loss: ~2.4")
    print("  - MLP val loss   : ~2.0 (improvement from context!)")
    print("  - Text should look more like real words")
    print()

    # ---- Load and prepare data ----
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
    model = MLPModel(
        vocab_size  = vocab_size,
        n_embd      = N_EMBD,             # 64
        block_size  = CONFIG["block_size"],  # 16
        hidden      = HIDDEN,             # 256
    )

    # Print architecture to show the student what's inside
    print(f"  Embedding   : ({vocab_size}, {N_EMBD})  — {vocab_size} chars × {N_EMBD}-dim vectors")
    print(f"  MLP input   : {CONFIG['block_size'] * N_EMBD}  — {CONFIG['block_size']} positions × {N_EMBD} dims, concatenated")
    print(f"  Hidden layer: {HIDDEN} neurons")
    print(f"  Output      : {vocab_size} — one score per vocabulary character")

    # ---- Train ----
    print()
    print("[ Training ]")
    final_val_loss = run_training(
        model_name = "MLP-Context",
        model      = model,
        train_data = train_data,
        val_data   = val_data,
        char2idx   = char2idx,
        idx2char   = idx2char,
        config     = CONFIG,
    )

    # ---- Results ----
    print("=" * 60)
    print(f"  FINAL VAL LOSS: {final_val_loss:.4f}")
    print()
    print("  WHAT DID WE LEARN?")
    print("  - Context (seeing past chars) dramatically helps!")
    print("  - But MLP treats all 16 positions equally")
    print("  - It can't easily learn 'the word 5 positions ago matters more'")
    print("  - Attention (step 2) will fix this with learned position weighting")
    print()
    print("=" * 60)
    print("  Next step: python step_02_single_attention.py")
    print("  What's next: Single-head attention — the key LLM concept!")
    print("=" * 60)

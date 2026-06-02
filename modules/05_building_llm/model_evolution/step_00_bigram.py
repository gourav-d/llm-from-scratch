# =============================================================================
# step_00_bigram.py
# =============================================================================
# THE SIMPLEST POSSIBLE LANGUAGE MODEL: A BIGRAM MODEL
#
# WHAT IS A BIGRAM?
#   "Bi" = two. A bigram is a pair of characters.
#   A bigram model looks at the CURRENT character and predicts the NEXT one.
#   That's it. No memory of what came before. Pure lookup table.
#
#   Example:
#     We've seen the text "the cat sat"
#     After 't': usually 'h' or ' ' or 'h'
#     After 'h': usually 'e' or 'a' or 'i'
#     After 'e': usually ' ' or 'r' or 'n'
#
# C# ANALOGY:
#   Imagine a Dictionary<char, Dictionary<char, int>> that counts how often
#   each character is followed by each other character. Then normalize those
#   counts to probabilities. That's essentially what the embedding layer learns.
#
# ARCHITECTURE (the simplest possible):
#
#   token_id  ──►  Embedding(vocab_size, vocab_size)  ──►  logits  ──►  next token
#
#   The embedding has shape (vocab_size, vocab_size).
#   Row i = the scores (logits) for every possible next token GIVEN token i.
#   Literally a lookup table: "given this character, these are the next-char scores"
#
# WHAT TO EXPECT:
#   - Loss starts around log(vocab_size) ≈ 4.6 (completely random baseline)
#   - After training, loss drops to ~2.4-2.5
#   - Generated text will look like: "th nd anor t ae t hae t ane..."
#   - Readable-ish character sequences, but no real words most of the time
#   - This is EXPECTED and GOOD — bigram can only use 1 character of context!
#
# WHY START HERE?
#   This is the foundation. Every subsequent step adds one new idea.
#   Understanding bigram makes multi-head attention feel less scary.
# =============================================================================

import torch                # PyTorch — our deep learning framework
import torch.nn as nn       # Neural network building blocks (layers, activations)
                            # C# analogy: nn is like the System.Collections namespace
                            # but for neural network components

# Import our shared utilities
# C# analogy: using SharedUtils; (a static utility class from a shared project)
from shared import (
    load_data,       # Loads corpus.txt from disk
    build_vocab,     # Builds character vocabulary (char <-> int mappings)
    split_data,      # Splits data into train/validation sets
    run_training,    # Runs the full training loop
    generate_text,   # Generates text from the model
)


# =============================================================================
# HYPERPARAMETERS — Tuned for fast training on CPU (5-10 minutes total)
# =============================================================================
# CONFIG is a dictionary — like a settings object in C#
# Keeping all hyperparameters in one place makes them easy to experiment with.
# Try changing batch_size or max_iters and see what happens!
CONFIG = {
    "block_size"    : 8,       # How many characters the model sees at once
                               # For bigram, this is almost irrelevant (model ignores context)
                               # but it's required by the shared training framework

    "batch_size"    : 64,      # How many training examples per step
                               # Like processing 64 work items in parallel
                               # C# analogy: Parallel.For with 64 iterations

    "max_iters"     : 3000,    # Total number of training steps
                               # 3000 steps is enough for bigram to converge

    "eval_interval" : 500,     # Print loss every 500 steps
                               # C# analogy: if (step % 500 == 0) Console.WriteLine(...)

    "lr"            : 1e-2,    # Learning rate = 0.01
                               # Bigger lr = faster but potentially unstable
                               # Bigram is simple enough to tolerate a larger lr

    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
                               # Use GPU if available, otherwise CPU
                               # torch.cuda.is_available() = do we have an Nvidia GPU?
                               # C# analogy: Environment.Is64BitProcess (checking capability)
}


# =============================================================================
# MODEL: BigramModel
# =============================================================================
class BigramModel(nn.Module):
    """
    The simplest language model: a single embedding table.

    WHAT IS nn.Module?
      nn.Module is the base class for ALL neural networks in PyTorch.
      Every model you'll ever write inherits from nn.Module.
      C# analogy: like inheriting from a base class or implementing an interface.
        In C#: public class BigramModel : BaseNeuralNetwork { }

    WHAT IS nn.Embedding?
      An embedding is a lookup table — a matrix of shape (num_entries, entry_size).
      nn.Embedding(vocab_size, vocab_size) creates a (V, V) matrix.

      Given a token index i, it returns row i of the matrix.
      That row contains the scores (logits) for every possible next token.

      C# analogy: like a float[][] array where you look up by index:
        float[] nextTokenScores = table[currentTokenIndex];

    WHY vocab_size x vocab_size?
      For bigram: each row IS the output logits.
      The matrix directly maps "current token" → "next token scores".
      No hidden layers needed — pure lookup.
    """

    def __init__(self, vocab_size):
        """
        Initialize the bigram model.

        PARAMETERS:
          vocab_size (int): Number of unique characters in our vocabulary.
        """
        # ALWAYS call super().__init__() first in any nn.Module
        # This sets up PyTorch's internal bookkeeping for the module
        # C# analogy: base() call in a constructor
        super().__init__()

        # The ONLY parameter in our model: a vocab_size x vocab_size lookup table
        # Each row i contains the logit scores for "what comes after token i"
        # PyTorch learns these values during training
        # C# analogy: float[,] embeddingTable = new float[vocabSize, vocabSize];
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        """
        Forward pass: run input through the model to get logits and loss.

        WHAT IS A FORWARD PASS?
          The forward pass is the computation that transforms input to output.
          PyTorch automatically computes gradients by tracking this computation.
          C# analogy: like a Func<Tensor, Tensor> — the model's main function.

        WHAT ARE LOGITS?
          "Logit" = raw unnormalized score for each possible output token.
          Higher logit = model thinks that token is more likely to come next.
          We convert logits to probabilities using softmax.
          C# analogy: like unnormalized weights before normalization.

        PARAMETERS:
          x (torch.LongTensor): Input token indices, shape (B, T)
            B = batch_size (e.g., 64 sequences)
            T = block_size (e.g., 8 tokens per sequence)
          targets (torch.LongTensor or None): Target token indices, shape (B, T)
            Used to compute loss during training.
            None during generation (we don't have targets yet).

        RETURNS:
          tuple: (logits, loss)
            logits: shape (B, T, vocab_size) — scores for each position
            loss: scalar float if targets provided, else None
        """
        # Look up embeddings for all tokens in x
        # For each token index in x, return its row from the embedding table
        # x shape: (B, T) — batch of sequences
        # logits shape: (B, T, vocab_size) — for each token, scores over vocab
        logits = self.embedding(x)

        # ---- Compute loss if we have targets ----
        if targets is None:
            # No targets = we're generating text, not training
            loss = None
        else:
            # Cross-entropy loss: how surprised is the model by the actual next token?
            # Loss = -log(probability of correct token)
            # Perfect prediction: loss = 0
            # Random prediction: loss = log(vocab_size) ≈ 4.6 for 100-char vocab

            # PyTorch's cross_entropy expects input shape (B*T, vocab_size)
            # and targets shape (B*T,) — it needs a flat list of (logit, target) pairs
            B, T, V = logits.shape   # Unpack the three dimensions
                                      # B = batch, T = time/sequence, V = vocab_size

            # .view(B*T, V) reshapes tensor WITHOUT copying data
            # Like flattening B sequences of T tokens into one list of B*T tokens
            # C# analogy: like .SelectMany() to flatten a list of lists
            logits_flat   = logits.view(B * T, V)    # (B*T, V)
            targets_flat  = targets.view(B * T)       # (B*T,)

            # F.cross_entropy computes the loss
            # It applies softmax internally, then computes -log(prob of correct token)
            # C# analogy: like -Math.Log(probability) where probability comes from softmax
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        return logits, loss


# =============================================================================
# MAIN — Run this file directly: python step_00_bigram.py
# =============================================================================
if __name__ == "__main__":

    # ---- Print a friendly banner ----
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  STEP 0: Bigram Model                                    ║")
    print("║  What's new   : Lookup table only — no context          ║")
    print("║  Architecture : token -> Embedding(V,V) -> logits       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ---- Explain what's happening ----
    print("WHAT THIS STEP TEACHES:")
    print("  - The simplest possible language model")
    print("  - Bigram = looks at 1 character, predicts next character")
    print("  - Uses a lookup table (nn.Embedding) — no math, no layers")
    print("  - C# analogy: Dictionary<char, char[]> with probabilities")
    print()
    print("EXPECTED RESULT:")
    print("  - Val loss around 2.4-2.5 (was ~4.6 before training)")
    print("  - Text looks like 'th t he t sh' — some patterns but no words")
    print("  - This is NORMAL — bigram sees only 1 char of context!")
    print()

    # ---- Step 1: Load data ----
    print("[ Loading data ]")
    text = load_data(max_chars=10_000_000)  # Load up to 10M characters from corpus.txt

    # ---- Step 2: Build vocabulary ----
    print()
    print("[ Building vocabulary ]")
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(idx2char)  # How many unique characters we found

    # ---- Step 3: Split into train / validation ----
    print()
    print("[ Splitting data ]")
    train_data, val_data = split_data(text, char2idx, train_frac=0.9)

    # ---- Step 4: Create the model ----
    print()
    print("[ Creating model ]")
    model = BigramModel(vocab_size=vocab_size)

    # Print the model architecture (shows layer names and parameter counts)
    # C# analogy: like printing Type.GetProperties() for each layer
    print(f"  Model: {model}")

    # ---- Step 5: Train! ----
    print()
    print("[ Training ]")
    final_val_loss = run_training(
        model_name  = "Bigram",
        model       = model,
        train_data  = train_data,
        val_data    = val_data,
        char2idx    = char2idx,
        idx2char    = idx2char,
        config      = CONFIG,
    )

    # ---- Final results ----
    print("=" * 60)
    print(f"  FINAL VAL LOSS: {final_val_loss:.4f}")
    print()
    print("  WHAT DOES THIS MEAN?")
    print(f"  - Random baseline  : ~{__import__('math').log(vocab_size):.2f} (log of vocab size)")
    print(f"  - After training   : ~{final_val_loss:.2f}")
    print(f"  - Improvement      : model learned some character patterns!")
    print()
    print("  WHY IS THE TEXT STILL GIBBERISH?")
    print("  Bigram only uses 1 character of context.")
    print("  To predict 'a' in 'cat', it only sees 't' — not the 'c' or 'a' before.")
    print("  Context is everything in language — we need more!")
    print()
    print("=" * 60)
    print("  Next step: python step_01_mlp_context.py")
    print("  What's next: MLP with 16-character context window")
    print("=" * 60)

"""
=============================================================================
EXAMPLE 4: Building GPT with PyTorch
=============================================================================

GLOSSARY (read before the code)
---------------------------------
PyTorch      : A Python library for deep learning. Like ASP.NET Core for
               neural networks — it handles the boring plumbing so you focus
               on the architecture. (www.pytorch.org)

Tensor       : PyTorch's version of a numpy array. The key difference:
               tensors can automatically track gradients.
               C# analogy: think of it as a Matrix<float> with built-in
               "undo" capability for derivatives.

Gradient     : A number that tells us: "how much should we change this
               weight to reduce the error?"
               C# analogy: like a derivative in calculus, but computed
               automatically by the library.

Autograd     : PyTorch's automatic gradient computation.
               You write the forward pass; PyTorch figures out the
               backward pass (gradients) automatically.

nn.Module    : The base class for all neural network layers in PyTorch.
               C# analogy: like an abstract base class that all layers
               must inherit from.
               Every layer you build must:
                 1. Call super().__init__() in __init__
                 2. Define learnable layers in __init__
                 3. Put the logic in forward()

nn.Embedding : A lookup table. Maps token IDs -> dense vectors.
               Exact same concept as the embedding table in Lesson 3,
               but PyTorch manages the memory and gradients.

nn.Linear    : A fully connected layer: output = input @ weight + bias
               C# analogy: a simple y = mx + b but for vectors.

nn.LayerNorm : Normalizes activations so they have mean~0 and std~1.
               Keeps training stable. Like input validation for tensors.

Dropout      : Randomly sets some values to zero during training.
               Forces the model to not rely on any single pathway.
               Only active during training — turned off at inference.

=============================================================================
PART A: WHY PyTorch? Manual gradients vs. automatic gradients
=============================================================================

The BIG reason to use PyTorch over NumPy:
  - NumPy: YOU write the backward pass (gradient math) for every operation.
  - PyTorch: You write the forward pass, PyTorch handles the rest.

Let us see this with the simplest possible example.
=============================================================================
"""

# Try to import PyTorch. If it is not installed, show how to install it.
try:
    import torch                        # the main PyTorch library
    import torch.nn as nn               # neural network building blocks
    import torch.nn.functional as F     # functions like softmax, relu
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np

print("=" * 60)
print("PART A: Manual Gradients (NumPy) vs Automatic (PyTorch)")
print("=" * 60)

if not TORCH_AVAILABLE:
    print("\nPyTorch is NOT installed on this machine.")
    print("To install it, open a terminal and run:")
    print("  pip install torch")
    print("\nShowing NumPy part only for now.")
    print()

# =============================================================================
# The Problem: We need gradients to train a model.
#
# Imagine the simplest possible "model":
#   prediction = weight * input
#   error = (prediction - target) ** 2
#
# We want to find the weight that minimizes error.
# To do that, we need: d(error)/d(weight)
# (how does error change if we change weight by a tiny amount?)
# =============================================================================

print("\nSimple problem: find weight W so that W * 2 = 6  (answer: W = 3)")
print()

# ---- NumPy approach (manual gradient) ----
print("--- NumPy approach: you write the gradient math yourself ---")

x      = 2.0       # input
target = 6.0       # what we want the model to predict
W      = 0.5       # starting weight (wrong answer, should become 3.0)
lr     = 0.1       # learning rate: how big a step to take each update

for step in range(10):
    # FORWARD PASS: compute prediction and error
    prediction = W * x                          # e.g. 0.5 * 2 = 1.0
    error      = (prediction - target) ** 2     # e.g. (1.0 - 6.0)^2 = 25.0

    # BACKWARD PASS (manually derived!)
    # d(error)/d(W) = 2 * (prediction - target) * x
    # You have to work this out with calculus for EVERY layer!
    gradient = 2 * (prediction - target) * x    # the "slope" for W

    # UPDATE: move W in the direction that reduces error
    W = W - lr * gradient

    if step % 2 == 0:
        print(f"  Step {step}: W={W:.4f}, prediction={W*x:.4f}, error={error:.4f}")

print(f"\n  Final W: {W:.4f}  (correct answer is 3.0)")
print("  Problem: you had to derive d(error)/d(W) by hand.")
print("  For a real GPT with millions of parameters, this is not practical!\n")

if TORCH_AVAILABLE:
    print("--- PyTorch approach: gradients computed automatically ---")

    x_pt      = torch.tensor(2.0)              # same input, as a PyTorch tensor
    target_pt = torch.tensor(6.0)              # same target
    W_pt      = torch.tensor(0.5, requires_grad=True)   # requires_grad=True -> track this!

    for step in range(10):
        # FORWARD PASS: exactly the same math
        prediction = W_pt * x_pt
        error      = (prediction - target_pt) ** 2

        # BACKWARD PASS: PyTorch computes ALL gradients automatically
        error.backward()                       # one line replaces all manual derivation!

        # UPDATE: W.grad now holds d(error)/d(W), computed by PyTorch
        with torch.no_grad():                  # no_grad: don't track the update step itself
            W_pt -= lr * W_pt.grad            # same update rule as NumPy
            W_pt.grad.zero_()                  # reset gradient for next step

        if step % 2 == 0:
            print(f"  Step {step}: W={W_pt.item():.4f}, "
                  f"prediction={W_pt.item()*x_pt.item():.4f}, "
                  f"error={error.item():.4f}")

    print(f"\n  Final W: {W_pt.item():.4f}  (same correct answer!)")
    print("  PyTorch computed the gradient — you did NOT write any calculus.\n")

print("=" * 60)
print("PART B: Building a Complete GPT in PyTorch")
print("=" * 60)

if not TORCH_AVAILABLE:
    print("\nPyTorch required for Part B. Please install it first:")
    print("  pip install torch")
    print("\nShowing the code structure below so you can read and understand it.")

# =============================================================================
# We will build a complete (tiny) GPT model step by step.
# The model has the same architecture as GPT-2.
# We will train it on the nursery rhyme from Example 3.
# =============================================================================

# ---- Configuration ----
# GPTConfig: holds all the settings for our model.
# C# analogy: like a settings class / options pattern.

class GPTConfig:
    """All hyperparameters in one place."""
    vocab_size   : int   # how many unique tokens exist
    d_model      : int   # size of each token's vector (embedding dimension)
    num_heads    : int   # number of attention heads
    num_layers   : int   # number of transformer blocks stacked on top of each other
    context_size : int   # max number of tokens we can look back at
    dropout      : float # fraction of activations to randomly drop during training

    def __init__(self, vocab_size, d_model=64, num_heads=2,
                 num_layers=2, context_size=32, dropout=0.1):
        self.vocab_size   = vocab_size
        self.d_model      = d_model
        self.num_heads    = num_heads
        self.num_layers   = num_layers
        self.context_size = context_size
        self.dropout      = dropout

# =============================================================================
# BUILDING BLOCK 1: Causal Self-Attention
# =============================================================================
# What does attention do?
#   Each token decides: "which OTHER tokens should I pay attention to?"
#   'Causal' means: you can only look at past tokens, not future ones.
#   (Otherwise the model would "cheat" by reading ahead.)
#
# C# analogy: imagine each token is a team member who can ask questions
# of all previous team members. They combine their answers to form a response.
# =============================================================================

if TORCH_AVAILABLE:
    class CausalSelfAttention(nn.Module):
        """
        Multi-head causal self-attention.
        'Multi-head' means we run several attention computations in parallel.
        Each head can focus on different patterns (grammar, meaning, position...).
        """

        def __init__(self, config):
            super().__init__()                             # always call parent first

            # d_k = size of each attention head
            # We split d_model evenly across all heads.
            # e.g. d_model=64, num_heads=2 -> d_k=32 per head
            self.num_heads = config.num_heads
            self.d_k       = config.d_model // config.num_heads

            # One projection matrix for Q, K, V combined (more efficient)
            # Input size:  d_model
            # Output size: 3 * d_model  (for Q, K, V all at once)
            self.qkv_proj  = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

            # Final projection after attention
            self.out_proj  = nn.Linear(config.d_model, config.d_model, bias=False)

            # Dropout for regularization (randomly zero some attention weights)
            self.attn_drop = nn.Dropout(config.dropout)

        def forward(self, x):
            """
            x shape: (batch_size, seq_len, d_model)
              batch_size = how many sequences we process at once (e.g. 4)
              seq_len    = how many tokens in the sequence (e.g. 32)
              d_model    = size of each token vector (e.g. 64)
            """
            B, T, C = x.shape          # B=batch, T=time/seq_len, C=channels/d_model

            # -----------------------------------------------------------------
            # Step 1: Compute Q, K, V for ALL heads at once
            # -----------------------------------------------------------------
            # qkv shape: (B, T, 3*C)
            qkv        = self.qkv_proj(x)

            # Split into three equal parts along the last dimension
            # Q, K, V each have shape: (B, T, C)
            Q, K, V    = qkv.split(C, dim=2)

            # -----------------------------------------------------------------
            # Step 2: Reshape for multi-head attention
            # -----------------------------------------------------------------
            # Reshape Q, K, V from (B, T, C) to (B, num_heads, T, d_k)
            # This separates each head so they can work independently.
            Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
            # Now shape is: (B, num_heads, T, d_k)

            # -----------------------------------------------------------------
            # Step 3: Compute attention scores (how much does each token care
            #         about each other token?)
            # -----------------------------------------------------------------
            # scores shape: (B, num_heads, T, T)
            # The (i,j) entry = how much token i should pay attention to token j
            scale  = self.d_k ** -0.5                           # 1 / sqrt(d_k)
            scores = (Q @ K.transpose(-2, -1)) * scale          # scaled dot-product

            # -----------------------------------------------------------------
            # Step 4: Causal mask — prevent token i from seeing token j if j > i
            # -----------------------------------------------------------------
            # torch.tril: lower-triangular matrix (keep past, hide future)
            # masked_fill: replace True positions with -infinity
            # After softmax, -infinity becomes 0 -> those tokens get zero weight
            mask   = torch.tril(torch.ones(T, T, device=x.device)).bool()
            scores = scores.masked_fill(~mask, float('-inf'))

            # -----------------------------------------------------------------
            # Step 5: Softmax -> attention weights (probabilities)
            # -----------------------------------------------------------------
            attn_weights = torch.softmax(scores, dim=-1)     # shape: (B, num_heads, T, T)
            attn_weights = self.attn_drop(attn_weights)       # randomly drop some weights

            # -----------------------------------------------------------------
            # Step 6: Weighted sum of Values
            # -----------------------------------------------------------------
            # Each token collects a weighted average of all Value vectors.
            # High weight = pay a lot of attention to that token.
            out = attn_weights @ V                            # (B, num_heads, T, d_k)

            # -----------------------------------------------------------------
            # Step 7: Reassemble all heads back into one tensor
            # -----------------------------------------------------------------
            out = out.transpose(1, 2).contiguous()            # (B, T, num_heads, d_k)
            out = out.view(B, T, C)                           # (B, T, d_model)

            # Final linear projection
            out = self.out_proj(out)

            return out                                        # (B, T, d_model)

    # =============================================================================
    # BUILDING BLOCK 2: Feed-Forward Network
    # =============================================================================
    # After attention decides WHAT to focus on, the feed-forward layer
    # decides WHAT TO DO with that information.
    #
    # It is a two-layer fully connected network applied to each token separately.
    # C# analogy: a stateless function — same input -> same output, no side effects.
    # =============================================================================

    class FeedForward(nn.Module):
        """Two-layer MLP applied independently to each token."""

        def __init__(self, config):
            super().__init__()
            # nn.Sequential: run these layers one after another
            # C# analogy: like a pipeline of middleware
            self.net = nn.Sequential(
                nn.Linear(config.d_model, 4 * config.d_model),  # expand to 4x size
                nn.GELU(),                                        # smooth activation function
                nn.Linear(4 * config.d_model, config.d_model),  # project back to original size
                nn.Dropout(config.dropout),                       # regularization
            )

        def forward(self, x):
            return self.net(x)          # just run the pipeline

    # =============================================================================
    # BUILDING BLOCK 3: One Transformer Block
    # =============================================================================
    # A transformer block = Attention + FeedForward + Layer Norms + Residuals.
    # GPT stacks N of these on top of each other.
    # =============================================================================

    class TransformerBlock(nn.Module):
        """One complete transformer layer."""

        def __init__(self, config):
            super().__init__()
            self.ln1  = nn.LayerNorm(config.d_model)          # normalize before attention
            self.attn = CausalSelfAttention(config)
            self.ln2  = nn.LayerNorm(config.d_model)          # normalize before feed-forward
            self.ff   = FeedForward(config)

        def forward(self, x):
            # RESIDUAL CONNECTIONS: add the input to the output of each sub-layer
            # Why? They allow gradients to flow directly through deep networks.
            # Without residuals, training a 12-layer model is nearly impossible.
            x = x + self.attn(self.ln1(x))   # attention branch: norm -> attend -> add
            x = x + self.ff(self.ln2(x))     # feed-forward branch: norm -> transform -> add
            return x

    # =============================================================================
    # COMPLETE GPT MODEL
    # =============================================================================

    class TinyGPT(nn.Module):
        """
        Complete GPT model.

        Architecture (same as GPT-2, just much smaller):
          token IDs (integers)
              ↓
          Token Embedding Table   (maps each token ID to a vector)
              +
          Position Embedding Table (maps each position 0,1,2,... to a vector)
              ↓
          Dropout
              ↓
          N x TransformerBlock    (N = num_layers)
              ↓
          LayerNorm
              ↓
          Linear (d_model -> vocab_size)  — output head
              ↓
          logits (raw scores for each possible next token)
        """

        def __init__(self, config):
            super().__init__()
            self.config = config

            # Embedding tables
            self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
            self.pos_emb   = nn.Embedding(config.context_size, config.d_model)
            self.drop      = nn.Dropout(config.dropout)

            # Stack of transformer blocks
            # nn.ModuleList: like a list but PyTorch can find the parameters inside
            self.blocks = nn.ModuleList(
                [TransformerBlock(config) for _ in range(config.num_layers)]
            )

            # Final layer normalization
            self.ln_f  = nn.LayerNorm(config.d_model)

            # Output head: maps from d_model to vocabulary size
            # (produces a score for each possible next token)
            self.head  = nn.Linear(config.d_model, config.vocab_size, bias=False)

            # Weight tying: the output head shares weights with the token embedding.
            # Why? The input and output both map between token IDs and vectors.
            # They should use the same representation. Also saves parameters!
            self.head.weight = self.token_emb.weight

            # Initialize weights (small random values -> stable training start)
            self.apply(self._init_weights)

        def _init_weights(self, module):
            """Set initial weights following the GPT-2 paper."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

        def forward(self, idx, targets=None):
            """
            idx     : token indices, shape (batch_size, seq_len)
            targets : next-token targets, shape (batch_size, seq_len) — optional
            Returns : logits, loss (loss is None if no targets provided)
            """
            B, T = idx.shape
            assert T <= self.config.context_size, \
                f"Sequence too long: {T} > max {self.config.context_size}"

            # 1. Look up token embeddings
            tok_emb = self.token_emb(idx)                   # (B, T, d_model)

            # 2. Look up positional embeddings
            positions = torch.arange(T, device=idx.device)  # [0, 1, ..., T-1]
            pos_emb   = self.pos_emb(positions)             # (T, d_model)

            # 3. Combine token + position (position tells the model WHERE each token is)
            x = self.drop(tok_emb + pos_emb)               # (B, T, d_model)

            # 4. Run through all transformer blocks
            for block in self.blocks:
                x = block(x)                                # (B, T, d_model)

            # 5. Final layer norm
            x = self.ln_f(x)                               # (B, T, d_model)

            # 6. Project to vocabulary size (one score per possible next token)
            logits = self.head(x)                           # (B, T, vocab_size)

            # 7. Compute loss if we have targets
            loss = None
            if targets is not None:
                # Cross-entropy loss: compares predicted scores to correct answers
                # Flatten to (B*T, vocab_size) and (B*T,) for the function
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1)
                )

            return logits, loss

        @torch.no_grad()                   # no gradient tracking during generation
        def generate(self, idx, max_new_tokens, temperature=0.8):
            """
            Generate max_new_tokens new tokens autoregressively.

            idx         : starting token IDs, shape (1, seq_len)
            temperature : controls randomness (lower = more focused, higher = creative)
            """
            for _ in range(max_new_tokens):
                # Crop sequence to max context size
                idx_cond = idx[:, -self.config.context_size:]

                # Forward pass
                logits, _ = self(idx_cond)

                # Take the logits for the LAST position only (next token prediction)
                logits = logits[:, -1, :] / temperature       # (1, vocab_size)

                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)

                # Sample the next token
                next_idx = torch.multinomial(probs, num_samples=1)   # (1, 1)

                # Append and repeat
                idx = torch.cat([idx, next_idx], dim=1)

            return idx

    # =============================================================================
    # DEMO: Train on the nursery rhyme and generate text
    # =============================================================================

    print("\nSetting up training data...")

    text_b = (
        "mary had a little lamb little lamb little lamb "
        "mary had a little lamb its fleece was white as snow "
        "and everywhere that mary went mary went mary went "
        "and everywhere that mary went the lamb was sure to go "
    ) * 5           # repeat 5x to give the model more training examples

    # Build vocabulary
    chars_b    = sorted(set(text_b))
    vocab_b    = len(chars_b)
    c2i        = {ch: i for i, ch in enumerate(chars_b)}
    i2c        = {i: ch for i, ch in enumerate(chars_b)}

    # Encode text as a single long tensor of integer token IDs
    data       = torch.tensor([c2i[ch] for ch in text_b], dtype=torch.long)
    print(f"Vocabulary: {vocab_b} unique chars,  Training tokens: {len(data)}")

    # Create model with tiny config (fast to train on CPU)
    cfg = GPTConfig(vocab_size=vocab_b, d_model=32, num_heads=2,
                    num_layers=2, context_size=24, dropout=0.1)
    model = TinyGPT(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")

    # Optimizer: AdamW is the standard optimizer for GPT
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop
    print("\nTraining... (100 steps)")
    context_size = cfg.context_size
    batch_size   = 8

    for step in range(100):
        # --- Random batch ---
        # Pick random starting positions in the data
        starts = torch.randint(len(data) - context_size - 1, (batch_size,))
        x_batch = torch.stack([data[s     : s + context_size    ] for s in starts])
        y_batch = torch.stack([data[s + 1 : s + context_size + 1] for s in starts])

        # --- Forward pass ---
        logits, loss = model(x_batch, y_batch)

        # --- Backward pass (PyTorch computes all gradients automatically) ---
        optimizer.zero_grad()       # clear gradients from last step
        loss.backward()             # compute gradients
        optimizer.step()            # update all weights

        if step % 25 == 0:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}")

    # Generate text
    print("\nGenerating text after training:")
    seed_text  = "mary"
    seed_ids   = torch.tensor([[c2i[ch] for ch in seed_text]])
    generated  = model.generate(seed_ids, max_new_tokens=80)
    result_str = ''.join(i2c[i.item()] for i in generated[0])
    print(f"\nSeed: '{seed_text}'")
    print(f"Generated: '{result_str}'")

print("\n" + "=" * 60)
print("SUMMARY: NumPy vs PyTorch")
print("=" * 60)
print("""
  NumPy GPT (Lesson 3):
    + Great for learning — every line is yours
    - You must derive and code every gradient by hand
    - Cannot run on GPU
    - Slow for large models

  PyTorch GPT (this lesson):
    + loss.backward() computes ALL gradients automatically
    + model.to('cuda') moves everything to GPU — one line
    + Saves/loads with torch.save() / torch.load()
    + Compatible with HuggingFace, Lightning, etc.
    - Slightly more abstract (harder to see the raw math)

  The math is IDENTICAL. PyTorch just handles the engineering.
  GPT-2 and GPT-3 use the same architecture you just built!
""")

print("=" * 60)
print("Example 4 complete!")
print("Next: Run exercise_04_gpt_pytorch.py to practice")
print("=" * 60)

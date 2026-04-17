"""
=============================================================================
EXERCISE 4: Building GPT with PyTorch — Hands-On Practice
=============================================================================

HOW TO USE THIS FILE
---------------------
Each exercise has:
  1. What to build / what to understand
  2. A skeleton with TODO comments
  3. A hint if you are stuck
  4. A solution (ONLY look after trying!)

REQUIREMENTS
  pip install torch

NOTE: If you do not have PyTorch, read the code and understand the structure —
that understanding is more valuable than running it.

=============================================================================
"""

print("=" * 60)
print("EXERCISE 4: Building GPT with PyTorch")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print("PyTorch is installed. Running all exercises.\n")
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not installed. Install it with: pip install torch")
    print("Reading the code structure is still valuable!\n")
    TORCH_AVAILABLE = False

import numpy as np

# =============================================================================
# EXERCISE 1: Understand nn.Module by building a simple linear layer
# =============================================================================
# DIFFICULTY: Easy
#
# WHAT IS nn.Module?
#   Every layer and model in PyTorch must inherit from nn.Module.
#   C# analogy: like an abstract base class. All neural network layers
#   must follow its contract.
#
# THE CONTRACT:
#   1. Call super().__init__() first in __init__
#   2. Define learnable layers (nn.Linear, nn.Embedding, etc.) in __init__
#   3. Put the forward logic in forward()
#
# YOUR TASK:
#   Complete the SimpleLinear class below.
#   It should compute: output = (input @ weight) + bias
#   This is the same as y = mx + b but for vectors.
# =============================================================================

print("=" * 60)
print("EXERCISE 1: Build a Simple Linear Layer")
print("=" * 60)
print("""
You will build the simplest possible nn.Module:
  A linear layer that computes output = input @ weight + bias
  (This is what nn.Linear does internally.)
""")

if TORCH_AVAILABLE:

    class SimpleLinear(nn.Module):
        """
        A manually implemented linear (fully connected) layer.
        Equivalent to nn.Linear(in_features, out_features).
        """

        def __init__(self, in_features, out_features):
            """
            in_features  : size of each input vector
            out_features : size of each output vector
            """
            # TODO Step 1: Call the parent class __init__
            # This is REQUIRED for nn.Module to work correctly.
            # HINT: super().__init__()
            pass   # replace this line

            # TODO Step 2: Create a learnable weight matrix
            # Use nn.Parameter() to tell PyTorch this should be trained.
            # Shape: (in_features, out_features)
            # HINT: self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
            pass   # replace this line

            # TODO Step 3: Create a learnable bias vector
            # Shape: (out_features,)
            # HINT: self.bias = nn.Parameter(torch.zeros(out_features))
            pass   # replace this line

        def forward(self, x):
            """
            x : input tensor of shape (batch_size, in_features)
            Returns output of shape (batch_size, out_features)
            """
            # TODO Step 4: Compute output = x @ self.weight + self.bias
            # HINT: return x @ self.weight + self.bias
            pass   # replace this line

    # --- Test Exercise 1 ---
    layer = SimpleLinear(in_features=4, out_features=2)

    # Check that the layer has parameters
    params = list(layer.parameters())
    print(f"Number of parameter tensors: {len(params)}")   # should be 2 (weight + bias)

    # Test forward pass
    test_input = torch.randn(3, 4)      # batch_size=3, in_features=4
    try:
        output = layer(test_input)       # calls forward() automatically
        print(f"Input shape:  {test_input.shape}")   # (3, 4)
        print(f"Output shape: {output.shape}")       # (3, 2)
        print("Exercise 1: PASSED" if output.shape == (3, 2) else "Exercise 1: check your shapes")
    except Exception as e:
        print(f"Error: {e}")
        print("Complete the TODO sections to fix this.")

    """
    ---- SOLUTION ----

    class SimpleLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
            self.bias   = nn.Parameter(torch.zeros(out_features))

        def forward(self, x):
            return x @ self.weight + self.bias
    """

# =============================================================================
# EXERCISE 2: Build a Token Embedding Layer
# =============================================================================
# DIFFICULTY: Easy
#
# WHAT IS AN EMBEDDING LAYER?
#   A lookup table: given a token ID (integer), return a dense vector.
#   Token 0 -> vector [0.2, -0.1, 0.5, ...]
#   Token 1 -> vector [0.8, 0.3, -0.2, ...]
#   etc.
#
# PyTorch already provides nn.Embedding.
# In this exercise you will USE it and understand what it does.
#
# YOUR TASK:
#   Complete the EmbeddingDemo class below.
#   It should take a batch of token IDs and return their embeddings.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 2: Embedding Layer")
print("=" * 60)
print("""
WHAT TO BUILD:
  A class that has TWO embedding tables:
    - token_emb : maps token ID -> vector of size d_model
    - pos_emb   : maps position (0, 1, 2, ...) -> vector of size d_model

  forward() should return: token_emb(tokens) + pos_emb(positions)
  (This is exactly what GPT does at its very first layer.)
""")

if TORCH_AVAILABLE:

    class EmbeddingLayer(nn.Module):
        """Token + Position embedding (first layer of GPT)."""

        def __init__(self, vocab_size, d_model, max_seq_len):
            """
            vocab_size  : number of unique tokens
            d_model     : size of each embedding vector
            max_seq_len : maximum sequence length supported
            """
            # TODO Step 1: Call parent __init__
            pass

            # TODO Step 2: Create token embedding table
            # HINT: self.token_emb = nn.Embedding(vocab_size, d_model)
            pass

            # TODO Step 3: Create position embedding table
            # HINT: self.pos_emb = nn.Embedding(max_seq_len, d_model)
            pass

        def forward(self, token_ids):
            """
            token_ids : tensor of shape (batch_size, seq_len) containing token indices
            Returns   : combined embeddings of shape (batch_size, seq_len, d_model)
            """
            B, T = token_ids.shape    # B = batch size, T = sequence length

            # TODO Step 4: Look up token embeddings
            # HINT: tok = self.token_emb(token_ids)   -> shape (B, T, d_model)
            tok = None   # replace

            # TODO Step 5: Create position indices [0, 1, 2, ..., T-1]
            # HINT: pos_idx = torch.arange(T, device=token_ids.device)
            pos_idx = None   # replace

            # TODO Step 6: Look up position embeddings
            # HINT: pos = self.pos_emb(pos_idx)   -> shape (T, d_model)
            pos = None   # replace

            # TODO Step 7: Add token + position embeddings and return
            # HINT: return tok + pos
            return None   # replace

    # --- Test Exercise 2 ---
    emb = EmbeddingLayer(vocab_size=50, d_model=16, max_seq_len=32)
    test_tokens = torch.randint(0, 50, (2, 8))  # batch=2, seq_len=8

    try:
        out = emb(test_tokens)
        print(f"Input shape:  {test_tokens.shape}")   # (2, 8)
        print(f"Output shape: {out.shape}")           # (2, 8, 16)
        print("Exercise 2: PASSED" if out.shape == (2, 8, 16) else "Check your shapes")
    except Exception as e:
        print(f"Error: {e}")
        print("Complete the TODO sections to fix this.")

    """
    ---- SOLUTION ----

    class EmbeddingLayer(nn.Module):
        def __init__(self, vocab_size, d_model, max_seq_len):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb   = nn.Embedding(max_seq_len, d_model)

        def forward(self, token_ids):
            B, T    = token_ids.shape
            tok     = self.token_emb(token_ids)
            pos_idx = torch.arange(T, device=token_ids.device)
            pos     = self.pos_emb(pos_idx)
            return tok + pos
    """

# =============================================================================
# EXERCISE 3: Observe Autograd in action
# =============================================================================
# DIFFICULTY: Easy
#
# WHAT IS AUTOGRAD?
#   PyTorch tracks all mathematical operations on tensors that have
#   requires_grad=True. When you call .backward(), it computes gradients
#   (derivatives) for every such tensor automatically.
#
# YOUR TASK:
#   Run a forward pass, compute a loss, call backward(),
#   then inspect the gradients to see what PyTorch computed.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 3: Observe Autograd")
print("=" * 60)
print("""
TASK:
  1. Create a weight tensor with requires_grad=True
  2. Do some math with it (forward pass)
  3. Call .backward()
  4. Print .grad to see what PyTorch computed
""")

if TORCH_AVAILABLE:
    # Simple model: prediction = W * x, loss = (prediction - target)^2
    W      = torch.tensor(1.5, requires_grad=True)   # starting weight
    x      = torch.tensor(3.0)                        # input
    target = torch.tensor(9.0)                        # we want W * 3 = 9 -> W = 3

    # TODO: Compute prediction = W * x
    prediction = None   # replace with: W * x

    # TODO: Compute loss = (prediction - target) ** 2
    loss = None          # replace with the formula above

    # TODO: Call loss.backward() to compute gradients
    # (no arguments needed)

    # TODO: Print W.grad  (this is d(loss)/d(W), computed by PyTorch)
    print(f"W           = {W.item():.4f}")
    print(f"Prediction  = {prediction.item():.4f}  (should be {W.item()*x.item():.1f})")
    print(f"Loss        = {loss.item():.4f}")
    print(f"W.grad      = {W.grad}")   # should be 2 * (prediction - target) * x
    print()
    expected_grad = 2 * (W.item() * x.item() - target.item()) * x.item()
    print(f"Expected gradient (manual calculation): {expected_grad:.4f}")
    print("Do they match?")

    """
    ---- SOLUTION ----

    prediction = W * x
    loss       = (prediction - target) ** 2
    loss.backward()
    # W.grad now contains d(loss)/d(W) = 2*(W*x - target)*x
    """

# =============================================================================
# EXERCISE 4: Count parameters of different GPT sizes
# =============================================================================
# DIFFICULTY: Easy (mostly reading and understanding)
#
# TASK:
#   Use the TinyGPT from Example 4 to count parameters for different configs.
#   Try to predict the count BEFORE running, then check.
#
# KEY FORMULA:
#   Parameters ~ vocab_size * d_model          (embedding table)
#              + num_layers * (
#                  4 * d_model^2               (attention Q,K,V,O projections)
#                + 2 * d_model * 4*d_model     (feed-forward: expand + contract)
#                + 2 * d_model                 (layer norms)
#                )
#              + d_model                        (final layer norm)
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 4: Count Parameters")
print("=" * 60)

if TORCH_AVAILABLE:

    # We will use a simplified model just for counting
    # (Using the full TinyGPT from example_04 would need all layers)
    # Instead, let us count manually using the formula above

    def estimate_params(vocab_size, d_model, num_layers):
        """
        Estimate the number of parameters in a GPT model.

        Does NOT include weight tying (shared input/output matrix).
        This is an approximation — the real count is close but not exact.
        """
        # Token embedding table
        token_emb = vocab_size * d_model

        # Per transformer block:
        #   Attention: Q, K, V, O projections each have d_model * d_model weights
        #   Feed-forward: expand (d_model -> 4*d_model) and contract (4*d_model -> d_model)
        #   Layer norms: 2 per block x 2 parameters each (gamma and beta)
        per_block = (
            4 * d_model * d_model          +   # attention projections
            2 * (d_model * 4 * d_model)    +   # feed-forward
            4 * d_model                        # layer norm parameters
        )

        # Output head (same size as token_emb — but with weight tying, shared)
        output_head = 0    # because it shares weights with the embedding

        total = token_emb + num_layers * per_block + output_head
        return total

    # ---  Fill in these predictions BEFORE looking at the output ---
    configs = [
        # (vocab_size, d_model, num_layers, name)
        (65,     32,  2,  "Tiny (Example 4 config)"),
        (65,    256,  4,  "Nano GPT"),
        (50257, 768, 12,  "GPT-2 Small"),
        (50257,1024, 24,  "GPT-2 Medium"),
    ]

    print(f"\n{'Configuration':<30} {'Est. Params':>14}")
    print("-" * 46)
    for vocab, d, layers, name in configs:
        n = estimate_params(vocab, d, layers)
        print(f"{name:<30} {n:>14,}")

    print()
    print("For reference:")
    print("  GPT-2 Small actual:  ~117M parameters")
    print("  GPT-2 Medium actual: ~345M parameters")
    print("  GPT-3:              ~175B parameters (same architecture, just BIGGER)")
    print()
    print("The difference from our estimate is because:")
    print("  - We simplified the formula")
    print("  - Position embeddings also have parameters")
    print("  - Bias terms add up")

print("\n" + "=" * 60)
print("Exercise 4 Complete!")
print()
print("KEY TAKEAWAYS:")
print("  - nn.Module is the base class for ALL neural network layers")
print("  - loss.backward() computes ALL gradients automatically")
print("  - GPT = embedding + N transformer blocks + output head")
print("  - GPT-2 and your TinyGPT have the SAME architecture")
print("    GPT-2 is just much bigger (768 vs 32 dimensions, 12 vs 2 layers)")
print("=" * 60)

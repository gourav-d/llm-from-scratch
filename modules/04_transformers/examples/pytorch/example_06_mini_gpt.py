"""
Example 06: Mini-GPT - PyTorch Version

THE CAPSTONE EXAMPLE - now using PyTorch's full toolkit!

Compare to the NumPy version:
  NumPy: ~200 lines of manual math (great for understanding)
  PyTorch: ~100 lines using built-in components (great for real development)

KEY NEW PYTORCH COMPONENT: nn.Embedding
  NumPy:   class TokenEmbedding:
               self.embeddings = np.random.randn(vocab_size, d_model) * 0.02
               def forward(self, token_ids):
                   return self.embeddings[token_ids]

  PyTorch: self.token_emb = nn.Embedding(vocab_size, d_model)
               output = self.token_emb(token_ids)

  nn.Embedding is a lookup table for learnable embeddings.
  In C# terms: Like a Dictionary<int, float[]> that can be trained!

Also shows: training loop, optimizer, loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

print("=" * 70)
print("MINI-GPT: COMPLETE TRANSFORMER ARCHITECTURE - PyTorch Version")
print("=" * 70)

# ==============================================================================
# PART 1: Token Embeddings with nn.Embedding
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Token Embeddings - nn.Embedding")
print("=" * 70)

print("""
nn.Embedding is PyTorch's built-in lookup table for token embeddings.

  NumPy way:
    self.embeddings = np.random.randn(vocab_size, d_model) * 0.02
    result = self.embeddings[token_ids]   # Index into array

  PyTorch way:
    self.token_emb = nn.Embedding(vocab_size, d_model)
    result = self.token_emb(token_ids)    # Call like a function

  WHY nn.Embedding is better:
    ✓ Trainable (gradients flow back to improve embeddings)
    ✓ Proper initialization built-in
    ✓ GPU-compatible automatically
    ✓ Handles padding tokens (padding_idx parameter)
    ✓ Handles batch processing

  In C# terms:
    NumPy  = Dictionary<int, float[]>  (lookup table, not trainable)
    PyTorch nn.Embedding = TrainableDictionary<int, Vector>  (can be learned!)
""")

vocab_size = 20   # Small vocab for demo
d_model    = 8

# Create embedding layer
token_emb = nn.Embedding(vocab_size, d_model)

print(f"Embedding table shape: {token_emb.weight.shape}")
print(f"  vocab_size={vocab_size} tokens, each has {d_model} dimensions")

# Test: convert token IDs to embeddings
test_tokens = torch.tensor([0, 5, 10, 15])    # NumPy: np.array([0, 5, 10, 15])
test_embeddings = token_emb(test_tokens)        # NumPy: self.embeddings[test_tokens]

print(f"\nInput token IDs: {test_tokens.tolist()}")
print(f"Embeddings shape: {test_embeddings.shape}")
print(f"\nEmbedding for token 0:\n{test_embeddings[0].tolist()}")

# ==============================================================================
# PART 2: Positional Encoding (same as example_04)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Positional Encoding")
print("=" * 70)

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model, max_seq_len=512):
        super().__init__()

        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        PE = torch.zeros(max_seq_len, d_model)
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)

        # Fixed buffer (not trainable)
        self.register_buffer('PE', PE)

    def forward(self, x):
        """x: (seq_len, d_model) or (batch, seq_len, d_model)"""
        if x.dim() == 2:
            return x + self.PE[:x.shape[0]]
        else:
            return x + self.PE[:x.shape[1]].unsqueeze(0)

print("PositionalEncoding module defined (same as Example 04)")

# ==============================================================================
# PART 3: Causal Mask - The PyTorch Way
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Causal Masking in PyTorch")
print("=" * 70)

print("""
SAME CONCEPT as NumPy: prevent attending to FUTURE tokens.

  NumPy way:
    mask = np.tril(np.ones((seq_len, seq_len))).astype(bool)
    scores = np.where(mask, scores, -1e9)

  PyTorch way (multiple options):

  OPTION A: Generate mask manually (same as NumPy)
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    scores = scores.masked_fill(~mask, float('-inf'))

  OPTION B: Use nn.Transformer's built-in helper
    mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
    # Returns a float mask: 0.0 = attend, -inf = blocked

  We'll use Option B for nn.MultiheadAttention integration!
""")

seq_len_demo = 6

# PyTorch way: generate causal mask
# This produces a float mask where -inf = "cannot attend"
causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len_demo)

print(f"Causal mask shape: {causal_mask.shape}")
print(f"\nCausal mask (0.0 = attend, -inf = blocked):")
print(causal_mask)

# Show as binary for clarity
binary_mask = torch.isfinite(causal_mask).int()
print(f"\nBinary view (1 = attend, 0 = blocked):")
print(binary_mask)

# Visualize
plt.figure(figsize=(8, 6))
plt.imshow(binary_mask.numpy(), cmap='RdYlGn', interpolation='nearest')
plt.title('Causal Attention Mask (PyTorch)\n(Green = Attend, Red = Blocked)',
          fontsize=14, fontweight='bold')
words = ["The", "cat", "sat", "on", "the", "mat"]
plt.xticks(range(6), words)
plt.yticks(range(6), words)
plt.xlabel('Attending TO (keys)')
plt.ylabel('Attending FROM (queries)')
plt.tight_layout()
plt.show()

# ==============================================================================
# PART 4: Mini-GPT Architecture
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Mini-GPT Architecture")
print("=" * 70)

print("""
Complete architecture (same as NumPy, but using PyTorch components):

  Token IDs
      ↓
  nn.Embedding         (Token embeddings: token_id → vector)
      ↓
  PositionalEncoding   (Add position info: where is each token?)
      ↓
  nn.TransformerEncoderLayer × N  (The transformer blocks with causal mask)
      ↓
  nn.LayerNorm         (Final normalization)
      ↓
  nn.Linear            (Language model head: vector → vocab probabilities)
      ↓
  Next token logits
""")

class MiniGPT(nn.Module):
    """
    Complete Mini-GPT language model using PyTorch components.

    Compare to the NumPy version:
      NumPy TokenEmbedding class      → nn.Embedding
      NumPy positional_encoding()     → PositionalEncoding module
      NumPy TransformerBlock loop     → nn.TransformerEncoder
      NumPy LayerNorm class           → nn.LayerNorm
      NumPy lm_head = embeddings.T    → nn.Linear

    This is much shorter and GPU-ready!
    """

    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len):
        super().__init__()

        self.d_model    = d_model
        self.vocab_size = vocab_size

        # Token embeddings: integer token IDs → dense vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding: add position information
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Stack of transformer blocks (with causal masking handled in forward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Language modeling head: vector → vocabulary logits
        # Output size = vocab_size (one score per possible next token)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between embedding and lm_head
        # This is a common trick in LLMs to reduce parameters
        # (same as NumPy: self.lm_head = self.token_embedding.embeddings.T)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids):
        """
        Forward pass.

        Args:
            token_ids: shape (seq_len,) or (batch_size, seq_len)

        Returns:
            logits: shape (seq_len, vocab_size) or (batch_size, seq_len, vocab_size)
        """
        squeeze = token_ids.dim() == 1
        if squeeze:
            token_ids = token_ids.unsqueeze(0)   # → (1, seq_len)

        batch_size, seq_len = token_ids.shape

        # 1. Token embeddings
        x = self.token_embedding(token_ids)   # (batch, seq_len, d_model)

        # 2. Add positional encoding
        x = self.pos_encoding(x)              # (batch, seq_len, d_model)

        # 3. Create causal mask (prevents attending to future tokens)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

        # 4. Transformer blocks with causal mask
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        # 5. Final layer norm
        x = self.final_norm(x)                # (batch, seq_len, d_model)

        # 6. Language modeling head: project to vocabulary
        logits = self.lm_head(x)              # (batch, seq_len, vocab_size)

        if squeeze:
            logits = logits.squeeze(0)         # → (seq_len, vocab_size)

        return logits

    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens, strategy='greedy', temperature=1.0, top_k=None):
        """
        Generate text auto-regressively.

        @torch.no_grad() decorator: no gradient tracking (inference mode)
        Same as wrapping everything in 'with torch.no_grad():'

        Args:
            start_tokens: Starting token IDs (list or tensor)
            max_new_tokens: Number of tokens to generate
            strategy: 'greedy', 'sample', or 'top_k'
            temperature: Controls randomness (higher = more random)
            top_k: For top-k sampling
        """
        tokens = list(start_tokens) if not isinstance(start_tokens, list) else start_tokens

        for _ in range(max_new_tokens):
            # Get predictions for current sequence
            token_tensor = torch.tensor(tokens)
            logits = self.forward(token_tensor)

            # Get logits for LAST position (next token prediction)
            next_logits = logits[-1] / temperature

            if strategy == 'greedy':
                # Always pick the highest-scoring token (deterministic)
                next_token = next_logits.argmax().item()

            elif strategy == 'sample':
                # Sample from the probability distribution (random)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                # torch.multinomial = random sampling from a distribution

            elif strategy == 'top_k':
                # Keep only top k tokens, then sample from them
                top_k_vals, top_k_idx = torch.topk(next_logits, k=top_k)
                top_k_probs = F.softmax(top_k_vals, dim=-1)
                chosen = torch.multinomial(top_k_probs, num_samples=1).item()
                next_token = top_k_idx[chosen].item()

            tokens.append(next_token)

        return tokens

# ==============================================================================
# PART 5: Create and Test Mini-GPT
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Create and Test Mini-GPT")
print("=" * 70)

mini_gpt = MiniGPT(
    vocab_size=20,
    d_model=8,
    num_layers=2,
    num_heads=2,
    d_ff=32,
    max_seq_len=16
)

print("Mini-GPT created!")
print(f"\nModel architecture:")
print(mini_gpt)

total_params = sum(p.numel() for p in mini_gpt.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Test forward pass
test_input = torch.tensor([0, 5, 10, 15, 7])
with torch.no_grad():
    logits = mini_gpt(test_input)

print(f"\nForward pass:")
print(f"  Input token IDs: {test_input.tolist()}")
print(f"  Output logits shape: {logits.shape}  (seq_len x vocab_size)")

predicted = logits[-1].argmax().item()
print(f"  Predicted next token: {predicted}")

# ==============================================================================
# PART 6: Text Generation
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Text Generation Strategies")
print("=" * 70)

start = [0, 5]
print(f"Starting tokens: {start}")

print("\n1. GREEDY Generation:")
generated = mini_gpt.generate(start, max_new_tokens=8, strategy='greedy')
print(f"   {generated}")

print("\n2. SAMPLING Generation:")
generated = mini_gpt.generate(start, max_new_tokens=8, strategy='sample', temperature=1.0)
print(f"   {generated}")

print("\n3. TOP-K Generation (k=5):")
generated = mini_gpt.generate(start, max_new_tokens=8, strategy='top_k', top_k=5)
print(f"   {generated}")

# ==============================================================================
# PART 7: Training Loop (BONUS - only possible in PyTorch/TensorFlow, not NumPy!)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: BONUS - How Training Works in PyTorch")
print("=" * 70)

print("""
THE BIG ADVANTAGE OF PYTORCH: You can TRAIN the model!

NumPy Mini-GPT: Fixed random weights, no learning possible.
PyTorch Mini-GPT: Can optimize weights using gradient descent!

Training loop:
  for each batch of text:
    1. Forward pass: logits = model(input_tokens)
    2. Compute loss: loss = cross_entropy(logits, target_tokens)
    3. Backward pass: loss.backward()  ← compute gradients automatically
    4. Update weights: optimizer.step()  ← gradient descent step
    5. Zero gradients: optimizer.zero_grad()  ← reset for next batch

In C# terms:
  Like a feedback loop:
    1. Predict()     → get current answer
    2. Score()       → measure how wrong the answer is
    3. Blame()       → figure out which weights caused the error
    4. Adjust()      → nudge weights in better direction
    5. Reset()       → clear the blame tracking for next round
""")

# Create dummy training data
dummy_input   = torch.tensor([0, 5, 10, 15, 7])   # Input tokens
dummy_targets = torch.tensor([5, 10, 15, 7, 3])   # Target next tokens

# Set up optimizer (Adam is the most common choice for transformers)
# lr = learning rate (how big each update step is)
optimizer = torch.optim.Adam(mini_gpt.parameters(), lr=0.001)

print("Running 3 training steps to show the concept:")
print(f"{'Step':>5} | {'Loss':>10}")
print("-" * 20)

for step in range(3):
    # Step 1: Forward pass
    logits = mini_gpt(dummy_input)   # (seq_len, vocab_size)

    # Step 2: Compute loss
    # CrossEntropyLoss: measures how wrong our predictions are
    # logits shape: (seq_len, vocab_size) → needs to be (N, C) for cross_entropy
    loss = F.cross_entropy(logits, dummy_targets)

    # Step 3: Backward pass (compute gradients automatically!)
    optimizer.zero_grad()   # Clear old gradients first
    loss.backward()         # Compute gradients for all parameters

    # Step 4: Update weights
    optimizer.step()        # Apply gradient descent update

    print(f"  {step+1:>3}  | {loss.item():>10.4f}")

print("\n  ✓ Model trained for 3 steps!")
print("  In real training: run for thousands/millions of steps with real data")

# ==============================================================================
# PART 8: Model Statistics Comparison
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Model Statistics")
print("=" * 70)

print(f"\nMini-GPT (PyTorch) Statistics:")
print(f"  Total parameters: {sum(p.numel() for p in mini_gpt.parameters()):,}")
print(f"  Vocabulary size: {mini_gpt.vocab_size}")
print(f"  Model dimension: {mini_gpt.d_model}")

print(f"\nFor comparison:")
print(f"  Our Mini-GPT: {sum(p.numel() for p in mini_gpt.parameters()):,} parameters")
print(f"  GPT-2 Small:  124,000,000 parameters")
print(f"  GPT-3:        175,000,000,000 parameters")
print(f"  GPT-4:        ~1,000,000,000,000 parameters (estimated)")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - Mini-GPT: NumPy vs PyTorch")
print("=" * 70)

print("""
FULL COMPARISON:

Component              │ NumPy                          │ PyTorch
───────────────────────┼────────────────────────────────┼────────────────────────────
Token embeddings       │ class TokenEmbedding: ...      │ nn.Embedding(vocab, d_model)
Positional encoding    │ positional_encoding() function │ PositionalEncoding(nn.Module)
Transformer blocks     │ list of TransformerBlock()     │ nn.TransformerEncoder(...)
Layer norm             │ class LayerNorm: ...           │ nn.LayerNorm(d_model)
Language model head    │ embeddings.T (manual)          │ nn.Linear(d_model, vocab)
Text generation        │ custom generate() loop         │ same, with torch.multinomial
Training               │ NOT POSSIBLE                   │ optimizer.step() !!!

PYTORCH-SPECIFIC CONCEPTS LEARNED:
  ✓ nn.Embedding        → trainable lookup table for tokens
  ✓ nn.TransformerEncoderLayer → complete transformer block built-in
  ✓ nn.TransformerEncoder     → stacked transformer blocks
  ✓ torch.multinomial   → sampling from a probability distribution
  ✓ F.cross_entropy()   → loss function for language modeling
  ✓ torch.optim.Adam    → optimizer (updates model weights during training)
  ✓ loss.backward()     → automatic gradient computation
  ✓ optimizer.step()    → weight update (gradient descent)
  ✓ @torch.no_grad()   → decorator for inference (no gradients)
  ✓ model.parameters() → iterate over all trainable weights

THE FUNDAMENTAL DIFFERENCE:
  NumPy Mini-GPT  = A calculator (can compute but cannot learn)
  PyTorch Mini-GPT = A student (can learn from data to improve!)

YOU NOW UNDERSTAND HOW CHATGPT WORKS UNDER THE HOOD!
The real ChatGPT is the same architecture, just with:
  - vocab_size: 50,257 tokens  (vs our 20)
  - d_model: 12,288 dims        (vs our 8)
  - num_layers: 96 blocks       (vs our 2)
  - num_heads: 96 heads         (vs our 2)
  - Trained on billions of tokens from the internet
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 06 - PyTorch Mini-GPT COMPLETE!")
print("=" * 70)

"""
Example 06: Mini-GPT - Complete Transformer Architecture

🎉 THE CAPSTONE EXAMPLE 🎉

This example builds a complete mini-GPT from scratch! We combine everything
we've learned in Examples 01-05 into a working language model that can
generate text.

What you'll see:
1. Token embeddings (converting words to vectors)
2. Positional encoding (adding position information)
3. Causal masking (preventing future information leakage)
4. Stacked transformer blocks (the core architecture)
5. Language modeling head (predicting next tokens)
6. Text generation (greedy, sampling, top-k strategies)
7. Complete forward pass walkthrough

This is HOW ChatGPT works under the hood!

Think of it like building a complete car:
- Examples 01-05: Building individual parts (engine, wheels, steering)
- Example 06: Assembling everything into a working vehicle
- The result: A mini language model that can generate text!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("🚀 MINI-GPT: COMPLETE TRANSFORMER ARCHITECTURE 🚀")
print("=" * 70)

# ==============================================================================
# PART 1: Token Embeddings
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Token Embeddings - Converting Tokens to Vectors")
print("=" * 70)

print("""
TOKEN EMBEDDINGS: A lookup table that converts token IDs to dense vectors.

Example:
  Token ID 0 ("the") → [0.23, -0.45, 0.67, ...]  (d_model dims)
  Token ID 1 ("cat") → [0.89, -0.12, 0.34, ...]  (d_model dims)
  Token ID 2 ("sat") → [-0.56, 0.78, -0.23, ...] (d_model dims)

In C# terms:
  Dictionary<int, float[]> embeddings = new Dictionary<int, float[]>();
  // Each token ID maps to a d_model-dimensional vector

In NumPy:
  embeddings = np.array with shape (vocab_size, d_model)
  Get embedding: embeddings[token_id]
""")

class TokenEmbedding:
    """
    Token embedding layer.

    Converts integer token IDs to dense vector representations.
    Like a C# Dictionary<int, Vector> but optimized for neural networks.
    """

    def __init__(self, vocab_size, d_model):
        """
        Initialize token embedding layer.

        Args:
            vocab_size: Number of unique tokens in vocabulary
            d_model: Dimension of embedding vectors
        """
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Initialize embedding matrix
        # Shape: (vocab_size, d_model)
        # In practice, these are learned during training
        self.embeddings = np.random.randn(vocab_size, d_model) * 0.02

    def forward(self, token_ids):
        """
        Look up embeddings for token IDs.

        Args:
            token_ids: Array of token IDs, shape (seq_len,) or (batch, seq_len)

        Returns:
            Embeddings, shape (seq_len, d_model) or (batch, seq_len, d_model)
        """
        # Simple array indexing (lookup)
        return self.embeddings[token_ids]

# Test token embeddings
vocab_size = 20  # Small vocab for demo
d_model = 8
token_emb = TokenEmbedding(vocab_size, d_model)

test_tokens = np.array([0, 5, 10, 15])  # Token IDs
test_embeddings = token_emb.forward(test_tokens)

print(f"\nVocabulary size: {vocab_size}")
print(f"Embedding dimension: {d_model}")
print(f"\nToken IDs: {test_tokens}")
print(f"Embeddings shape: {test_embeddings.shape}")
print(f"\nEmbedding for token 0:\n{test_embeddings[0]}")

# ==============================================================================
# PART 2: Positional Encoding (from Example 04)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Positional Encoding - Adding Position Information")
print("=" * 70)

print("""
Positional encoding tells the model WHERE each token is in the sequence.

Without it: "cat sat mat" = "mat cat sat" (order-blind!)
With it: Position 0, Position 1, Position 2 (order-aware!)
""")

def positional_encoding(max_seq_len, d_model):
    """
    Generate sinusoidal positional encodings.

    Args:
        max_seq_len: Maximum sequence length
        d_model: Embedding dimension

    Returns:
        Positional encoding matrix, shape (max_seq_len, d_model)
    """
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    PE = np.zeros((max_seq_len, d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE

# ==============================================================================
# PART 3: Causal Masking - Preventing Future Information Leakage
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Causal Masking - The Secret of Autoregressive Models")
print("=" * 70)

print("""
CRITICAL CONCEPT: Language models predict NEXT token, so they can't see the future!

Example: Predicting "The cat sat"
  Position 0 ("The") can only see: "The"
  Position 1 ("cat") can only see: "The", "cat"
  Position 2 ("sat") can only see: "The", "cat", "sat"

Causal mask prevents attention to future positions:

Attention Matrix (6x6):
  Can attend to →
  ↓         Past   Current   Future
  "The"     [✓     ✗         ✗ ✗ ✗ ✗]
  "cat"     [✓     ✓         ✗ ✗ ✗ ✗]
  "sat"     [✓     ✓         ✓ ✗ ✗ ✗]
  "on"      [✓     ✓         ✓ ✓ ✗ ✗]
  "the"     [✓     ✓         ✓ ✓ ✓ ✗]
  "mat"     [✓     ✓         ✓ ✓ ✓ ✓]

Implementation: Set future positions to -inf before softmax
  → After softmax, they become 0 (no attention)

Similar to C# upper triangular matrix:
  bool[,] mask = new bool[seq_len, seq_len];
  for (int i = 0; i < seq_len; i++)
      for (int j = i + 1; j < seq_len; j++)
          mask[i, j] = false;  // Can't see future
""")

def create_causal_mask(seq_len):
    """
    Create causal mask for autoregressive attention.

    Args:
        seq_len: Sequence length

    Returns:
        Boolean mask, shape (seq_len, seq_len)
        True = can attend, False = cannot attend (future)
    """
    # Create lower triangular matrix (including diagonal)
    # np.tril creates lower triangle (ones below/on diagonal, zeros above)
    mask = np.tril(np.ones((seq_len, seq_len)))
    return mask.astype(bool)

# Visualize causal mask
seq_len_demo = 6
causal_mask = create_causal_mask(seq_len_demo)

print(f"\nCausal mask for sequence length {seq_len_demo}:")
print(causal_mask.astype(int))
print("\n1 = can attend, 0 = cannot attend (future)")

# Visualize
plt.figure(figsize=(8, 6))
plt.imshow(causal_mask, cmap='RdYlGn', interpolation='nearest')
plt.title('Causal Attention Mask\n(Green = Can Attend, Red = Blocked)',
          fontsize=14, fontweight='bold')
plt.xlabel('Attending TO (keys)')
plt.ylabel('Attending FROM (queries)')
words = ["The", "cat", "sat", "on", "the", "mat"]
plt.xticks(range(6), words)
plt.yticks(range(6), words)
plt.colorbar(label='Can Attend')
plt.tight_layout()
plt.show()

# ==============================================================================
# PART 4: Building the Components (Reusing from Example 05)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Transformer Components (from Previous Examples)")
print("=" * 70)

def softmax(x, axis=-1):
    """Softmax function."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class LayerNorm:
    """Layer normalization."""

    def __init__(self, d_model, epsilon=1e-6):
        self.epsilon = epsilon
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * x_norm + self.beta

class FeedForward:
    """Feed-forward network."""

    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return hidden @ self.W2 + self.b2

class CausalSelfAttention:
    """Self-attention with causal masking."""

    def __init__(self, d_model):
        self.d_model = d_model
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def forward(self, x):
        seq_len = x.shape[0]

        # Linear projections
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Attention scores
        scores = Q @ K.T / np.sqrt(self.d_model)

        # Apply causal mask
        mask = create_causal_mask(seq_len)
        scores = np.where(mask, scores, -1e9)  # Set future to -inf

        # Softmax and output
        weights = softmax(scores, axis=-1)
        output = weights @ V

        return output @ self.W_o

class TransformerBlock:
    """Complete transformer block with causal attention."""

    def __init__(self, d_model, d_ff):
        self.attention = CausalSelfAttention(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        # Attention block
        attn_output = self.attention.forward(x)
        x = self.norm1.forward(x + attn_output)

        # FFN block
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)

        return x

print("✓ All transformer components ready!")

# ==============================================================================
# PART 5: Complete Mini-GPT Architecture
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Mini-GPT - Putting It All Together!")
print("=" * 70)

class MiniGPT:
    """
    Complete GPT architecture.

    Components:
      1. Token embeddings
      2. Positional encoding
      3. Transformer blocks (stacked)
      4. Final layer norm
      5. Language modeling head (predicts next token)
    """

    def __init__(self, vocab_size, d_model, num_layers, d_ff, max_seq_len):
        """
        Initialize Mini-GPT.

        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding/model dimension
            num_layers: Number of transformer blocks
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)

        # Positional encoding (fixed, not learned)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, d_ff)
                       for _ in range(num_layers)]

        # Final layer norm
        self.final_norm = LayerNorm(d_model)

        # Language modeling head (projects to vocabulary)
        # Shared with token embeddings (weight tying)
        self.lm_head = self.token_embedding.embeddings.T  # (d_model, vocab_size)

    def forward(self, token_ids):
        """
        Forward pass.

        Args:
            token_ids: Input token IDs, shape (seq_len,)

        Returns:
            logits: Next token predictions, shape (seq_len, vocab_size)
        """
        seq_len = len(token_ids)

        # 1. Token embeddings
        x = self.token_embedding.forward(token_ids)  # (seq_len, d_model)

        # 2. Add positional encoding
        x = x + self.pos_encoding[:seq_len]  # (seq_len, d_model)

        # 3. Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # 4. Final layer norm
        x = self.final_norm.forward(x)

        # 5. Language modeling head
        logits = x @ self.lm_head  # (seq_len, vocab_size)

        return logits

    def generate(self, start_tokens, max_new_tokens, strategy='greedy', temperature=1.0, top_k=None):
        """
        Generate text autoregressively.

        Args:
            start_tokens: Starting token IDs
            max_new_tokens: Number of tokens to generate
            strategy: 'greedy', 'sample', or 'top_k'
            temperature: Sampling temperature (higher = more random)
            top_k: For top-k sampling

        Returns:
            Generated token sequence
        """
        tokens = list(start_tokens)

        for _ in range(max_new_tokens):
            # Get logits for current sequence
            logits = self.forward(np.array(tokens))

            # Get logits for last position (next token prediction)
            next_token_logits = logits[-1] / temperature

            # Select next token based on strategy
            if strategy == 'greedy':
                next_token = np.argmax(next_token_logits)

            elif strategy == 'sample':
                probs = softmax(next_token_logits)
                next_token = np.random.choice(len(probs), p=probs)

            elif strategy == 'top_k':
                # Keep only top k logits
                top_k_indices = np.argsort(next_token_logits)[-top_k:]
                top_k_logits = next_token_logits[top_k_indices]
                top_k_probs = softmax(top_k_logits)
                next_token = top_k_indices[np.random.choice(len(top_k_probs), p=top_k_probs)]

            tokens.append(int(next_token))

        return np.array(tokens)

# Create Mini-GPT
print("\nCreating Mini-GPT:")
print("  Vocabulary size: 20")
print("  Model dimension: 8")
print("  Number of layers: 2")
print("  Feed-forward dimension: 32")
print("  Max sequence length: 16")

mini_gpt = MiniGPT(
    vocab_size=20,
    d_model=8,
    num_layers=2,
    d_ff=32,
    max_seq_len=16
)

print("\n✅ Mini-GPT created successfully!")

# ==============================================================================
# PART 6: Testing Mini-GPT
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Testing Mini-GPT")
print("=" * 70)

# Test forward pass
test_input = np.array([0, 5, 10, 15, 7])
print(f"\nInput token IDs: {test_input}")

logits = mini_gpt.forward(test_input)
print(f"Output logits shape: {logits.shape}")
print(f"  → ({logits.shape[0]} positions, {logits.shape[1]} vocab size)")

# Interpret last position (next token prediction)
print(f"\nPredicting token after position {len(test_input)-1}:")
last_logits = logits[-1]
predicted_token = np.argmax(last_logits)
print(f"  Predicted token ID: {predicted_token}")
print(f"  Confidence (logit): {last_logits[predicted_token]:.2f}")

# ==============================================================================
# PART 7: Text Generation
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Text Generation - The Magic Happens!")
print("=" * 70)

print("""
TEXT GENERATION: Auto-regressive prediction

Process:
  1. Start with initial tokens: [0, 5]
  2. Predict next token: → 12
  3. Append: [0, 5, 12]
  4. Predict next token: → 3
  5. Append: [0, 5, 12, 3]
  6. Repeat until max_new_tokens

Generation strategies:
  - Greedy: Always pick highest probability token (deterministic)
  - Sampling: Sample from probability distribution (random)
  - Top-k: Sample from top k tokens only (balanced)
""")

start = np.array([0, 5])
print(f"\nStarting tokens: {start}")

# Greedy generation
print("\n1. GREEDY Generation (deterministic):")
generated_greedy = mini_gpt.generate(start, max_new_tokens=8, strategy='greedy')
print(f"   Generated: {generated_greedy}")

# Sampling generation
print("\n2. SAMPLING Generation (random):")
generated_sample = mini_gpt.generate(start, max_new_tokens=8, strategy='sample', temperature=1.0)
print(f"   Generated: {generated_sample}")

# Top-k generation
print("\n3. TOP-K Generation (k=5):")
generated_topk = mini_gpt.generate(start, max_new_tokens=8, strategy='top_k', top_k=5)
print(f"   Generated: {generated_topk}")

# ==============================================================================
# PART 8: Model Statistics
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Model Statistics")
print("=" * 70)

def count_parameters(model):
    """Count model parameters."""
    total = 0

    # Token embeddings
    total += model.vocab_size * model.d_model

    # Transformer blocks
    for _ in range(model.num_layers):
        # Attention: 4 × (d_model × d_model)
        total += 4 * model.d_model * model.d_model

        # FFN: (d_model × d_ff) + d_ff + (d_ff × d_model) + d_model
        d_ff = 32  # Known from init
        total += model.d_model * d_ff + d_ff + d_ff * model.d_model + model.d_model

    # Layer norms: 2 × num_layers × 2 × d_model
    total += 2 * model.num_layers * 2 * model.d_model

    # Final layer norm
    total += 2 * model.d_model

    return total

num_params = count_parameters(mini_gpt)
print(f"\nMini-GPT Statistics:")
print(f"  Total parameters: {num_params:,}")
print(f"  Vocabulary size: {mini_gpt.vocab_size}")
print(f"  Model dimension: {mini_gpt.d_model}")
print(f"  Number of layers: {mini_gpt.num_layers}")
print(f"  Max sequence length: {mini_gpt.max_seq_len}")

print(f"\nFor comparison:")
print(f"  GPT-2 Small: 124M parameters")
print(f"  GPT-3: 175B parameters")
print(f"  Our Mini-GPT: {num_params:,} parameters (for educational purposes!)")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("🎉 SUMMARY - You Built a Complete GPT! 🎉")
print("=" * 70)

print("""
✅ CONGRATULATIONS! You've built a complete transformer language model!

Components We Built:
  1. Token Embeddings: Convert token IDs → vectors
  2. Positional Encoding: Add position information
  3. Causal Masking: Prevent seeing the future
  4. Self-Attention: Learn relationships between tokens
  5. Feed-Forward: Transform representations
  6. Layer Normalization: Stabilize training
  7. Residual Connections: Help gradient flow
  8. Language Modeling Head: Predict next token
  9. Text Generation: Auto-regressive sampling

The Complete Pipeline:
  Token IDs → Embeddings → +Position → Transformer Blocks → Norm → LM Head → Next Token

Generation Process:
  [0, 5] → forward → predict 12 → [0, 5, 12] → forward → predict 3 → ...

This IS how ChatGPT works (with more layers and parameters)!

Key Insights:
  ✓ Attention = Communication (tokens talk to each other)
  ✓ Feed-Forward = Computation (individual processing)
  ✓ Causal Mask = No cheating (can't see future)
  ✓ Auto-regressive = One token at a time
  ✓ Stack layers = Learn complex patterns

Scaling to Real Models:
  Mini-GPT (our version):
    - Vocab: 20, d_model: 8, layers: 2
    - Parameters: ~2,000

  GPT-2 Small:
    - Vocab: 50K, d_model: 768, layers: 12
    - Parameters: 124M

  GPT-3:
    - Vocab: 50K, d_model: 12288, layers: 96
    - Parameters: 175B

  GPT-4:
    - Unknown, but likely 1T+ parameters!

In C#/.NET Terms:
  - Token embeddings: Dictionary<int, float[]>
  - Forward pass: Pipeline pattern
  - Auto-regressive: while loop with state
  - Causal mask: Upper triangular matrix
  - Softmax: Normalization to probabilities

What You Can Do Now:
  1. Train this model on real text data
  2. Add more layers for better patterns
  3. Increase vocab size for real words
  4. Implement beam search for better generation
  5. Add temperature control for creativity
  6. Build a chatbot interface!

Real-World Applications:
  ✓ ChatGPT: Scaled-up version of this
  ✓ Code generation: GitHub Copilot
  ✓ Translation: Google Translate
  ✓ Summarization: News summaries
  ✓ Question answering: Search engines

You Now Understand:
  ✅ How transformers work (all 6 lessons!)
  ✅ How attention mechanisms operate
  ✅ How positional encoding works
  ✅ How text generation happens
  ✅ How ChatGPT is built!

Next Steps in Your LLM Journey:
  - Module 5: Training and fine-tuning
  - Module 6: Advanced architectures
  - Module 7: Real-world applications

YOU DID IT! 🎊 You've built a transformer from scratch!
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 06 - MINI-GPT COMPLETE!")
print("=" * 70)

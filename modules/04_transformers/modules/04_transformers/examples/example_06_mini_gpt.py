"""
Example 06: Mini-GPT - Building a Complete Language Model from Scratch! 🎉

This is the CAPSTONE example that combines EVERYTHING we've learned:
- Token embeddings
- Positional encoding
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Residual connections
- Stacked transformer blocks
- Language modeling head
- Text generation!

We're building a simplified version of GPT-2! This is a real, working
language model that can generate text (though with random weights, the
output will be nonsense until trained).

C# Analogy:
This is like building a complete web application that combines:
  - Database layer (embeddings)
  - Business logic (transformer blocks)
  - API layer (language modeling head)
  - Client code (text generation)

All working together to create a functional system!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed
np.random.seed(42)

print("=" * 70)
print("MINI-GPT: BUILDING A COMPLETE LANGUAGE MODEL")
print("=" * 70)

# ==============================================================================
# PART 1: The Architecture Overview
# ==============================================================================

print("\n" + "=" * 70)
print("PART 1: Mini-GPT Architecture")
print("=" * 70)

print("""
Our Mini-GPT Architecture:

┌────────────────────────────────────────────────┐
│                  MINI-GPT                       │
│                                                 │
│  Input Tokens: [5, 12, 3, 45, 2]              │
│         │                                       │
│         ▼                                       │
│  ┌─────────────────────────────────┐          │
│  │  Token Embedding Table           │          │
│  │  (vocab_size × d_model)          │          │
│  └─────────────────────────────────┘          │
│         │                                       │
│         (+) ← Positional Encoding              │
│         │                                       │
│         ▼                                       │
│  ┌─────────────────────────────────┐          │
│  │  Transformer Block 1             │          │
│  │  (Attention + FFN + Norms)       │          │
│  └─────────────────────────────────┘          │
│         │                                       │
│  ┌─────────────────────────────────┐          │
│  │  Transformer Block 2             │          │
│  │  (Attention + FFN + Norms)       │          │
│  └─────────────────────────────────┘          │
│         │                                       │
│  ┌─────────────────────────────────┐          │
│  │  Transformer Block N             │          │
│  │  (Attention + FFN + Norms)       │          │
│  └─────────────────────────────────┘          │
│         │                                       │
│         ▼                                       │
│  ┌─────────────────────────────────┐          │
│  │  Language Modeling Head          │          │
│  │  (Linear: d_model → vocab_size) │          │
│  └─────────────────────────────────┘          │
│         │                                       │
│         ▼                                       │
│  Output Logits: [0.2, 0.5, -0.3, ...]         │
│         │                                       │
│         ▼                                       │
│  Softmax → Probabilities → Sample Token       │
│                                                 │
└────────────────────────────────────────────────┘

This is the GPT architecture! Let's build it! 🚀
""")

# ==============================================================================
# PART 2: Component Classes (Reused from Previous Examples)
# ==============================================================================

print("\n" + "=" * 70)
print("PART 2: Building Block Components")
print("=" * 70)

print("Reusing components from previous examples...\n")

class LayerNorm:
    """Layer Normalization."""

    def __init__(self, d_model, epsilon=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.epsilon = epsilon

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * x_norm + self.beta


class MultiHeadAttention:
    """Multi-Head Self-Attention."""

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

    def forward(self, X, mask=None):
        batch_size, seq_len, d_model = X.shape

        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        Q = self._split_heads(Q, batch_size, seq_len)
        K = self._split_heads(K, batch_size, seq_len)
        V = self._split_heads(V, batch_size, seq_len)

        attn_output = self._attention(Q, K, V, mask)
        concat = self._combine_heads(attn_output, batch_size, seq_len)

        return concat @ self.W_o

    def _split_heads(self, x, batch_size, seq_len):
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x, batch_size, seq_len):
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)

    def _attention(self, Q, K, V, mask=None):
        scores = Q @ K.transpose(-2, -1) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask  # Add mask (large negative values for masked positions)

        weights = self._softmax(scores)
        return weights @ V

    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class FeedForward:
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return hidden @ self.W2 + self.b2


class TransformerBlock:
    """Complete Transformer Block."""

    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Attention sublayer
        attn_output = self.attention.forward(x, mask)
        x = self.norm1.forward(x + attn_output)

        # FFN sublayer
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)

        return x

print("✓ Component classes defined!")

# ==============================================================================
# PART 3: Positional Encoding
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3: Positional Encoding")
print("=" * 70)

def positional_encoding(max_seq_len, d_model):
    """Generate sinusoidal positional encodings."""
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    PE = np.zeros((max_seq_len, d_model))
    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE

print("✓ Positional encoding function defined!")

# ==============================================================================
# PART 4: The Mini-GPT Model
# ==============================================================================

print("\n" + "=" * 70)
print("PART 4: Complete Mini-GPT Model")
print("=" * 70)

class MiniGPT:
    """
    Mini-GPT: A simplified GPT-2 style language model.

    This is a complete, working transformer-based language model!

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens)
        d_model: Embedding dimension (e.g., 512)
        num_layers: Number of transformer blocks (e.g., 12)
        num_heads: Number of attention heads (e.g., 8)
        d_ff: Feed-forward hidden dimension (e.g., 2048)
        max_seq_len: Maximum sequence length (e.g., 1024)
    """

    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len):
        print(f"\nInitializing Mini-GPT:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Embedding dimension: {d_model}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Number of heads: {num_heads}")
        print(f"  FFN dimension: {d_ff}")
        print(f"  Max sequence length: {max_seq_len}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding table
        # Similar to C# Dictionary<int, Vector> mapping token IDs to embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        print(f"\n  ✓ Token embedding table: ({vocab_size} × {d_model})")

        # Positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, d_model)
        print(f"  ✓ Positional encoding: ({max_seq_len} × {d_model})")

        # Transformer blocks
        self.blocks = []
        print(f"\n  Creating {num_layers} transformer blocks:")
        for i in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff)
            self.blocks.append(block)
            print(f"    ✓ Block {i+1}/{num_layers}")

        # Language modeling head (projects to vocabulary)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
        print(f"\n  ✓ Language modeling head: ({d_model} × {vocab_size})")

        # Final layer norm
        self.final_norm = LayerNorm(d_model)
        print(f"  ✓ Final layer normalization")

        print("\n🎉 Mini-GPT initialized successfully!")

    def forward(self, token_ids):
        """
        Forward pass through Mini-GPT.

        Args:
            token_ids: Array of token IDs, shape (batch_size, seq_len)

        Returns:
            logits: Predictions for next token, shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape

        # 1. Token embeddings
        # Like C#: embeddings = token_ids.Select(id => embeddingTable[id])
        embeddings = self.token_embedding[token_ids]  # (batch, seq_len, d_model)

        # 2. Add positional encodings
        positions = self.positional_encoding[:seq_len, :]  # (seq_len, d_model)
        x = embeddings + positions  # Broadcasting: add same positions to all batch items

        # 3. Create causal mask (prevent attending to future positions)
        # This is crucial for autoregressive generation!
        mask = self._create_causal_mask(seq_len)

        # 4. Pass through all transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # 5. Final layer normalization
        x = self.final_norm.forward(x)

        # 6. Project to vocabulary (language modeling head)
        logits = x @ self.lm_head  # (batch, seq_len, vocab_size)

        return logits

    def _create_causal_mask(self, seq_len):
        """
        Create causal mask to prevent attending to future positions.

        Returns a matrix where:
          - 0 for positions we CAN attend to (past and present)
          - -inf for positions we CANNOT attend to (future)

        Example for seq_len=4:
            [[  0, -inf, -inf, -inf],
             [  0,   0, -inf, -inf],
             [  0,   0,   0, -inf],
             [  0,   0,   0,   0]]

        Position 0 can only see position 0
        Position 1 can see positions 0 and 1
        Position 2 can see positions 0, 1, and 2
        etc.
        """
        # Create upper triangular matrix of ones
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)

        # Convert to -infinity where mask is 1, else 0
        mask = mask * -1e9

        return mask

    def generate(self, start_tokens, max_new_tokens, strategy='greedy', temperature=1.0, top_k=None):
        """
        Generate text autoregressively.

        Args:
            start_tokens: Initial tokens, shape (batch_size, start_len) or (start_len,)
            max_new_tokens: Number of new tokens to generate
            strategy: 'greedy', 'sample', or 'top_k'
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token sequence
        """
        # Ensure 2D
        if start_tokens.ndim == 1:
            start_tokens = start_tokens[np.newaxis, :]

        current_tokens = start_tokens.copy()

        for _ in range(max_new_tokens):
            # Get logits for all positions
            logits = self.forward(current_tokens)

            # Get logits for last position (what to generate next)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

            # Generate next token based on strategy
            if strategy == 'greedy':
                next_token = np.argmax(next_token_logits, axis=-1, keepdims=True)

            elif strategy == 'sample':
                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Softmax to get probabilities
                probs = self._softmax(next_token_logits)

                # Sample from distribution
                next_token = np.array([[np.random.choice(self.vocab_size, p=probs[0])]])

            elif strategy == 'top_k':
                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Get top k indices
                top_k_indices = np.argsort(next_token_logits[0])[-top_k:]

                # Zero out non-top-k logits
                filtered_logits = np.full_like(next_token_logits[0], -np.inf)
                filtered_logits[top_k_indices] = next_token_logits[0][top_k_indices]

                # Softmax and sample
                probs = self._softmax(filtered_logits[np.newaxis, :])
                next_token = np.array([[np.random.choice(self.vocab_size, p=probs[0])]])

            # Append to sequence
            current_tokens = np.concatenate([current_tokens, next_token], axis=1)

            # Truncate if exceeds max length (keep last max_seq_len tokens)
            if current_tokens.shape[1] > self.max_seq_len:
                current_tokens = current_tokens[:, -self.max_seq_len:]

        return current_tokens[0]  # Remove batch dimension

    @staticmethod
    def _softmax(x):
        """Softmax function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

print("\n✓ Mini-GPT class defined!")

# ==============================================================================
# PART 5: Creating and Testing Mini-GPT
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5: Creating a Mini-GPT Instance")
print("=" * 70)

# Model configuration (small for demonstration)
config = {
    'vocab_size': 100,      # Small vocabulary (real GPT-2 has 50,257)
    'd_model': 128,         # Embedding dimension (GPT-2: 768)
    'num_layers': 4,        # Number of transformer blocks (GPT-2: 12)
    'num_heads': 4,         # Attention heads (GPT-2: 12)
    'd_ff': 512,            # FFN dimension (GPT-2: 3072)
    'max_seq_len': 32       # Max sequence length (GPT-2: 1024)
}

print("Model configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Create the model
model = MiniGPT(**config)

# ==============================================================================
# PART 6: Forward Pass Test
# ==============================================================================

print("\n" + "=" * 70)
print("PART 6: Testing Forward Pass")
print("=" * 70)

# Create some sample input tokens
# Simulating tokens for: "The cat sat"
sample_tokens = np.array([[15, 42, 73]])  # Shape: (batch=1, seq_len=3)

print(f"\nInput tokens: {sample_tokens}")
print(f"Input shape: {sample_tokens.shape}")

# Forward pass
print("\nRunning forward pass...")
logits = model.forward(sample_tokens)

print(f"\nOutput logits shape: {logits.shape}")
print(f"  (batch={logits.shape[0]}, seq_len={logits.shape[1]}, vocab_size={logits.shape[2]})")

# Get predictions for next token after each position
print("\nPredictions for next token after each position:")
for i in range(sample_tokens.shape[1]):
    next_token_probs = model._softmax(logits[0, i:i+1, :])
    predicted_token = np.argmax(next_token_probs)
    confidence = next_token_probs[0, predicted_token]

    print(f"  After position {i} (token {sample_tokens[0, i]}): "
          f"predict token {predicted_token} (confidence: {confidence:.4f})")

print("\n✓ Forward pass successful!")

# ==============================================================================
# PART 7: Text Generation
# ==============================================================================

print("\n" + "=" * 70)
print("PART 7: Text Generation")
print("=" * 70)

print("""
Now let's generate text! We'll use different generation strategies:
1. Greedy: Always pick the most likely next token
2. Sampling: Randomly sample from the probability distribution
3. Top-k: Sample from the k most likely tokens

NOTE: Since our model has random weights (not trained), the output
will be nonsense tokens. In a TRAINED model, this would generate
coherent text!
""")

# Starting sequence
start_tokens = np.array([15, 42])  # "The cat" (hypothetically)
print(f"\nStarting tokens: {start_tokens}")

# Generate with different strategies
strategies = [
    ('greedy', {}),
    ('sample', {'temperature': 1.0}),
    ('top_k', {'top_k': 10, 'temperature': 0.8})
]

for strategy_name, kwargs in strategies:
    print(f"\n--- {strategy_name.upper()} Generation ---")

    generated = model.generate(
        start_tokens,
        max_new_tokens=10,
        strategy=strategy_name,
        **kwargs
    )

    print(f"Generated sequence: {generated}")
    print(f"  Start: {generated[:2]}")
    print(f"  Generated: {generated[2:]}")

print("""
\nIn a TRAINED model, you would see something like:
  Input:  "The cat"
  Output: "The cat sat on the mat and looked around"

But our random weights produce random tokens!
""")

# ==============================================================================
# PART 8: Analyzing Model Size
# ==============================================================================

print("\n" + "=" * 70)
print("PART 8: Model Statistics")
print("=" * 70)

# Count parameters
def count_parameters(model):
    """Count total number of parameters in the model."""
    total = 0

    # Token embeddings
    total += model.token_embedding.size

    # Positional encoding (not learned in this implementation)
    # total += model.positional_encoding.size

    # Transformer blocks
    for block in model.blocks:
        # Attention
        total += block.attention.W_q.size
        total += block.attention.W_k.size
        total += block.attention.W_v.size
        total += block.attention.W_o.size

        # Layer norm 1
        total += block.norm1.gamma.size
        total += block.norm1.beta.size

        # FFN
        total += block.ffn.W1.size
        total += block.ffn.b1.size
        total += block.ffn.W2.size
        total += block.ffn.b2.size

        # Layer norm 2
        total += block.norm2.gamma.size
        total += block.norm2.beta.size

    # Language modeling head
    total += model.lm_head.size

    # Final norm
    total += model.final_norm.gamma.size
    total += model.final_norm.beta.size

    return total

total_params = count_parameters(model)
print(f"\nTotal parameters: {total_params:,}")
print(f"  Approximately: {total_params / 1_000_000:.2f} million")

print("\nCompare to real models:")
print("  GPT-2 Small:   117 million parameters")
print("  GPT-2 Medium:  345 million parameters")
print("  GPT-2 Large:   774 million parameters")
print("  GPT-3:         175 BILLION parameters!")

print(f"\nOur Mini-GPT is {total_params / 117_000_000 * 100:.2f}% the size of GPT-2 Small")

# ==============================================================================
# PART 9: Visualizing the Causal Mask
# ==============================================================================

print("\n" + "=" * 70)
print("PART 9: Understanding the Causal Mask")
print("=" * 70)

print("""
The causal mask is CRUCIAL for language modeling!
It prevents the model from "cheating" by looking at future tokens.

When predicting position i, the model can only see positions 0 to i.
This makes the model autoregressive - it generates one token at a time.
""")

# Create and visualize causal mask
seq_len_demo = 8
causal_mask = model._create_causal_mask(seq_len_demo)

# Convert -inf to a visible number for plotting
mask_plot = np.where(causal_mask == 0, 1, 0)

plt.figure(figsize=(8, 6))
sns.heatmap(mask_plot,
            cmap='RdYlGn',
            cbar_kws={'label': '1=Can Attend, 0=Cannot Attend'},
            xticklabels=range(seq_len_demo),
            yticklabels=range(seq_len_demo),
            annot=True,
            fmt='d',
            linewidths=0.5)

plt.title('Causal Attention Mask\n(Green=Allowed, Red=Blocked)', fontsize=12, fontweight='bold')
plt.xlabel('Key Position (attending TO)', fontsize=10)
plt.ylabel('Query Position (attending FROM)', fontsize=10)

# Add explanation
plt.text(0.5, -0.12,
         'Each position can only attend to itself and previous positions.\n'
         'This prevents "seeing into the future" during generation.',
         ha='center', va='top', transform=plt.gca().transAxes,
         fontsize=9, style='italic')

plt.tight_layout()
plt.show()

print("\nExample:")
print("  Position 3 can attend to: [0, 1, 2, 3]  ✓")
print("  Position 3 CANNOT attend to: [4, 5, 6, 7]  ✗")
print("\nThis is why it's called 'causal' - causality flows forward in time!")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("SUMMARY - What We Built")
print("=" * 70)

print("""
🎉 CONGRATULATIONS! You just built a complete GPT-style language model! 🎉

What we implemented:
✓ Token embedding layer (converts token IDs to vectors)
✓ Positional encoding (adds position information)
✓ Multi-head self-attention (captures relationships)
✓ Feed-forward networks (adds non-linearity)
✓ Layer normalization (stabilizes training)
✓ Residual connections (helps gradient flow)
✓ Causal masking (prevents seeing future)
✓ Language modeling head (predicts next token)
✓ Text generation (greedy, sampling, top-k)

This is the SAME architecture as:
  - GPT-2
  - GPT-3
  - GPT-4 (with some enhancements)
  - Many other modern language models!

What's Missing (from a production GPT):
  - Training code (backpropagation, optimization)
  - Dropout (regularization)
  - Better initialization
  - Tokenization (we used token IDs directly)
  - Pre-training on massive text corpus
  - Fine-tuning for specific tasks

But the CORE ARCHITECTURE is exactly what you built! 🚀

Key Insights:
1. GPT is surprisingly simple conceptually
2. It's "just" stacked transformer blocks
3. The magic comes from:
   - Scale (billions of parameters)
   - Training data (billions of tokens)
   - Compute (months of GPU time)
   - Careful engineering (optimization, etc.)

What You Can Do Now:
  - Understand how ChatGPT works under the hood
  - Read GPT papers and actually understand them
  - Implement improvements (different attention mechanisms, etc.)
  - Move to Module 5: Training and fine-tuning!

You've come a long way:
  Module 1: Python basics ✓
  Module 2: NumPy and math ✓
  Module 3: Neural networks ✓
  Module 4: Transformers ✓ ← YOU ARE HERE!
  Module 5: Building & training LLMs ← NEXT!

AMAZING WORK! 🌟
""")

print("\n" + "=" * 70)
print("END OF EXAMPLE 06 - MINI-GPT")
print("=" * 70)
print("\nYou're now ready to train this model and build real LLMs! 🚀")

# Lesson 6: Complete GPT Architecture

**Build a mini-GPT from scratch - the complete transformer!**

---

## 🎯 What You'll Learn

- ✅ The complete GPT architecture
- ✅ Token embeddings + positional encoding
- ✅ Stacking transformer blocks
- ✅ Output layer for text generation
- ✅ Building mini-GPT from scratch
- ✅ How ChatGPT really works!

**Time:** 3-4 hours
**Difficulty:** ⭐⭐⭐⭐⭐

**Prerequisites:** Lessons 1-5 (All previous lessons)

---

## 🏗️ The Complete Architecture

### GPT-2 Full Architecture

```
Input: Token IDs [45, 1223, 887, ...]
  ↓
Token Embedding (lookup table)
  ↓
  + (add)
  ↓
Positional Encoding
  ↓
──────────────────────────────
Transformer Block 1
  ├─ Masked Multi-Head Attention
  ├─ Add & Norm
  ├─ Feed-Forward Network
  └─ Add & Norm
──────────────────────────────
Transformer Block 2
  ├─ Masked Multi-Head Attention
  ├─ Add & Norm
  ├─ Feed-Forward Network
  └─ Add & Norm
──────────────────────────────
...
──────────────────────────────
Transformer Block 12
  ├─ Masked Multi-Head Attention
  ├─ Add & Norm
  ├─ Feed-Forward Network
  └─ Add & Norm
──────────────────────────────
  ↓
Final Layer Norm
  ↓
Linear (Language Modeling Head)
  ↓
Output: Logits for next token [vocab_size]
  ↓
Softmax → Probabilities
  ↓
Sample → Next Token ID
```

**Let's build this step by step!**

---

## 📝 Step 1: Token Embeddings

### What Are Token Embeddings?

**Convert token IDs to dense vectors:**

```python
Vocabulary: 50,000 tokens
Embedding dimension: 768

Token ID → Embedding Vector
"hello" (ID: 1234) → [0.23, -0.45, 0.67, ..., 0.12]  (768 dims)
"world" (ID: 5678) → [0.89, -0.12, 0.34, ..., -0.56] (768 dims)
```

**Think of it as a lookup table:**
```python
# Like a C# dictionary
Dictionary<int, float[]> embeddings = new Dictionary<int, float[]>();
embeddings[1234] = new float[768] { 0.23f, -0.45f, ... };
embeddings[5678] = new float[768] { 0.89f, -0.12f, ... };

# In Python (NumPy)
embeddings = np.random.randn(50000, 768)  # All token embeddings
word_embedding = embeddings[1234]  # Get "hello" embedding
```

---

### Implementation

```python
class TokenEmbedding:
    """
    Token embedding layer.

    Converts token IDs to dense vectors.
    """

    def __init__(self, vocab_size, d_model):
        """
        Initialize embedding layer.

        Args:
            vocab_size: Number of tokens in vocabulary
            d_model: Embedding dimension
        """
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Initialize embedding matrix (learned during training)
        # Shape: (vocab_size, d_model)
        self.embeddings = np.random.randn(vocab_size, d_model) * 0.02

    def forward(self, token_ids):
        """
        Look up embeddings for token IDs.

        Args:
            token_ids: (batch_size, seq_len) integer array

        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        # Simple lookup
        return self.embeddings[token_ids]
```

---

## 🔄 Step 2: Add Positional Encoding

**From Lesson 4, we learned to add position information:**

```python
# Token embeddings
token_emb = token_embedding.forward(token_ids)  # (batch, seq_len, d_model)

# Add positional encoding
pos_enc = positional_encoding.forward(token_emb)  # (batch, seq_len, d_model)

# Combined input to transformer
transformer_input = token_emb + pos_enc
```

---

## 🔁 Step 3: Stack Transformer Blocks

**Stack N transformer blocks (12 for GPT-2 Small):**

```python
class TransformerStack:
    """
    Stack of N transformer blocks.
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff):
        """
        Initialize transformer stack.

        Args:
            num_layers: Number of transformer blocks
            d_model: Model dimension
            num_heads: Number of attention heads per block
            d_ff: Feed-forward hidden dimension
        """
        self.num_layers = num_layers

        # Create list of transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

    def forward(self, X, mask=None):
        """
        Forward pass through all blocks.

        Args:
            X: Input (batch, seq_len, d_model)
            mask: Attention mask

        Returns:
            Output (batch, seq_len, d_model)
        """
        # Pass through each block sequentially
        for block in self.blocks:
            X, _ = block.forward(X, mask)

        return X
```

---

## 🎯 Step 4: Language Modeling Head

**Final layer that predicts next token:**

```python
class LanguageModelHead:
    """
    Linear layer that projects to vocabulary size.

    Outputs logits for each token in vocabulary.
    """

    def __init__(self, d_model, vocab_size):
        """
        Initialize LM head.

        Args:
            d_model: Model dimension
            vocab_size: Size of vocabulary
        """
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Weight matrix (often tied to token embeddings!)
        scale = 1.0 / np.sqrt(d_model)
        self.W = np.random.randn(d_model, vocab_size) * scale
        self.b = np.zeros(vocab_size)

    def forward(self, X):
        """
        Project to vocabulary.

        Args:
            X: (batch, seq_len, d_model)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Linear projection
        logits = X @ self.W + self.b  # (batch, seq_len, vocab_size)

        return logits
```

---

## 🚀 Complete GPT Model

### Full Implementation

```python
import numpy as np

class MiniGPT:
    """
    Complete GPT model from scratch!

    This is a simplified version of GPT-2.
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=512):
        """
        Initialize Mini-GPT.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (768 for GPT-2 small)
            num_heads: Number of attention heads (12 for GPT-2 small)
            num_layers: Number of transformer blocks (12 for GPT-2 small)
            d_ff: FFN hidden dimension (3072 for GPT-2 small)
            max_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_len = max_len

        # Components
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.transformer_stack = TransformerStack(num_layers, d_model, num_heads, d_ff)
        self.final_norm = LayerNorm(d_model)
        self.lm_head = LanguageModelHead(d_model, vocab_size)

    def create_causal_mask(self, seq_len):
        """
        Create causal mask for autoregressive generation.

        Position i can only attend to positions 0...i (not future).

        Args:
            seq_len: Sequence length

        Returns:
            mask: (seq_len, seq_len) boolean mask
        """
        # Lower triangular matrix
        mask = np.tril(np.ones((seq_len, seq_len)))
        return mask

    def forward(self, token_ids):
        """
        Forward pass through Mini-GPT.

        Args:
            token_ids: (batch_size, seq_len) token IDs

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape

        # 1. Token embeddings
        X = self.token_embedding.forward(token_ids)
        # (batch, seq_len, d_model)

        # 2. Add positional encoding
        X = self.positional_encoding.forward(X)
        # (batch, seq_len, d_model)

        # 3. Create causal mask
        mask = self.create_causal_mask(seq_len)
        # (seq_len, seq_len)

        # 4. Pass through transformer blocks
        X = self.transformer_stack.forward(X, mask)
        # (batch, seq_len, d_model)

        # 5. Final layer normalization
        X = self.final_norm.forward(X)
        # (batch, seq_len, d_model)

        # 6. Language modeling head
        logits = self.lm_head.forward(X)
        # (batch, seq_len, vocab_size)

        return logits

    def generate(self, token_ids, max_new_tokens=20, temperature=1.0):
        """
        Generate text autoregressively.

        Args:
            token_ids: Starting tokens (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens)
        """
        generated = token_ids.copy()

        for _ in range(max_new_tokens):
            # Get logits for next token
            logits = self.forward(generated)  # (batch, current_len, vocab_size)

            # Get logits for last position
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Softmax to get probabilities
            probs = self.softmax(next_token_logits)

            # Sample next token
            next_token = self.sample(probs)  # (batch, 1)

            # Append to sequence
            generated = np.concatenate([generated, next_token], axis=1)

        return generated

    def softmax(self, x):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def sample(self, probs):
        """
        Sample from probability distribution.

        Args:
            probs: (batch, vocab_size) probabilities

        Returns:
            samples: (batch, 1) sampled token IDs
        """
        batch_size = probs.shape[0]
        samples = np.zeros((batch_size, 1), dtype=int)

        for i in range(batch_size):
            # Sample from categorical distribution
            samples[i, 0] = np.random.choice(self.vocab_size, p=probs[i])

        return samples
```

---

### Usage Example

```python
# GPT-2 Small configuration
config = {
    'vocab_size': 50257,      # GPT-2 vocabulary size
    'd_model': 768,           # Embedding dimension
    'num_heads': 12,          # Attention heads
    'num_layers': 12,         # Transformer blocks
    'd_ff': 3072,             # FFN hidden (4 * d_model)
    'max_len': 1024           # Max sequence length
}

# Create model
gpt = MiniGPT(**config)

# Example input: "Hello world"
# (In practice, use a tokenizer like tiktoken)
token_ids = np.array([[15496, 995]])  # Shape: (1, 2)

print("Input tokens:", token_ids)

# Forward pass
logits = gpt.forward(token_ids)
print("Output logits shape:", logits.shape)  # (1, 2, 50257)

# Generate text
generated = gpt.generate(token_ids, max_new_tokens=10)
print("Generated tokens:", generated)
print("Generated shape:", generated.shape)  # (1, 12)
```

---

## 📊 Model Sizes Comparison

### GPT Family

**GPT-2 Small:**
```python
Parameters: 124M
- vocab_size: 50,257
- d_model: 768
- num_heads: 12
- num_layers: 12
- d_ff: 3,072
```

**GPT-2 Medium:**
```python
Parameters: 355M
- d_model: 1,024
- num_heads: 16
- num_layers: 24
- d_ff: 4,096
```

**GPT-2 Large:**
```python
Parameters: 774M
- d_model: 1,280
- num_heads: 20
- num_layers: 36
- d_ff: 5,120
```

**GPT-2 XL:**
```python
Parameters: 1.5B
- d_model: 1,600
- num_heads: 25
- num_layers: 48
- d_ff: 6,400
```

**GPT-3:**
```python
Parameters: 175B
- d_model: 12,288
- num_heads: 96
- num_layers: 96
- d_ff: 49,152
```

**ChatGPT (GPT-3.5):**
```python
Parameters: ~175B (estimated)
- Similar architecture to GPT-3
- Fine-tuned with RLHF
```

---

## 🎮 Text Generation Strategies

### 1. Greedy Decoding

**Always pick most likely token:**

```python
def generate_greedy(model, token_ids, max_new_tokens):
    """Greedy decoding (deterministic)."""
    generated = token_ids.copy()

    for _ in range(max_new_tokens):
        logits = model.forward(generated)
        next_logits = logits[:, -1, :]

        # Pick argmax (most likely)
        next_token = np.argmax(next_logits, axis=-1, keepdims=True)

        generated = np.concatenate([generated, next_token], axis=1)

    return generated

# Problem: Repetitive and boring!
# "The cat sat on the cat. The cat sat on the cat."
```

---

### 2. Random Sampling

**Sample from probability distribution:**

```python
def generate_random(model, token_ids, max_new_tokens, temperature=1.0):
    """Random sampling."""
    generated = token_ids.copy()

    for _ in range(max_new_tokens):
        logits = model.forward(generated)
        next_logits = logits[:, -1, :] / temperature

        probs = softmax(next_logits)
        next_token = np.random.choice(vocab_size, p=probs[0])
        next_token = np.array([[next_token]])

        generated = np.concatenate([generated, next_token], axis=1)

    return generated

# More diverse, but can be incoherent with high temperature!
```

---

### 3. Top-k Sampling

**Sample from k most likely tokens:**

```python
def generate_top_k(model, token_ids, max_new_tokens, k=50, temperature=1.0):
    """Top-k sampling."""
    generated = token_ids.copy()

    for _ in range(max_new_tokens):
        logits = model.forward(generated)
        next_logits = logits[:, -1, :] / temperature

        # Get top k tokens
        top_k_indices = np.argsort(next_logits[0])[-k:]
        top_k_logits = next_logits[0, top_k_indices]

        # Softmax over top k
        probs = softmax(top_k_logits)

        # Sample from top k
        sampled_idx = np.random.choice(k, p=probs)
        next_token = top_k_indices[sampled_idx]
        next_token = np.array([[next_token]])

        generated = np.concatenate([generated, next_token], axis=1)

    return generated

# Good balance between diversity and coherence!
# ChatGPT uses variants of this.
```

---

### 4. Top-p (Nucleus) Sampling

**Sample from smallest set with cumulative probability > p:**

```python
def generate_top_p(model, token_ids, max_new_tokens, p=0.9, temperature=1.0):
    """
    Top-p (nucleus) sampling.

    Used in GPT-3 and ChatGPT!
    """
    generated = token_ids.copy()

    for _ in range(max_new_tokens):
        logits = model.forward(generated)
        next_logits = logits[:, -1, :] / temperature

        # Softmax
        probs = softmax(next_logits)[0]

        # Sort probabilities
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Find cumulative probability
        cumsum_probs = np.cumsum(sorted_probs)

        # Find cutoff (smallest set with cumsum > p)
        cutoff = np.searchsorted(cumsum_probs, p)

        # Keep only nucleus
        nucleus_indices = sorted_indices[:cutoff+1]
        nucleus_probs = sorted_probs[:cutoff+1]

        # Renormalize
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        # Sample
        sampled_idx = np.random.choice(len(nucleus_indices), p=nucleus_probs)
        next_token = nucleus_indices[sampled_idx]
        next_token = np.array([[next_token]])

        generated = np.concatenate([generated, next_token], axis=1)

    return generated

# Adaptive: nucleus size changes based on confidence!
```

---

## 🧪 Practice Problems

### Problem 1: Calculate Total Parameters

For Mini-GPT with:
- vocab_size = 50,257
- d_model = 768
- num_heads = 12
- num_layers = 12
- d_ff = 3,072

Calculate total parameters.

<details>
<summary>Solution</summary>

```python
Token embeddings: 50,257 × 768 = 38,597,376
Positional encoding: 0 (sinusoidal, no parameters)

Per transformer block:
- Attention: 4 × (768 × 768) = 2,359,296
- FFN: (768 × 3072) + 3072 + (3072 × 768) + 768 = 4,722,432
- LayerNorm (×2): 2 × (768 + 768) = 3,072
  Subtotal: 7,084,800

12 blocks: 12 × 7,084,800 = 85,017,600

Final LayerNorm: 768 + 768 = 1,536
LM head: 768 × 50,257 = 38,597,376

Total: 38,597,376 + 85,017,600 + 1,536 + 38,597,376
     = 162,213,888 parameters

~162M parameters (close to GPT-2's 124M with optimizations!)
```
</details>

---

### Problem 2: Implement Beam Search

Implement beam search decoding (keeps top-k sequences):

```python
def generate_beam_search(model, token_ids, max_new_tokens, beam_size=5):
    """
    Beam search decoding.

    Keeps beam_size best sequences.

    Args:
        model: GPT model
        token_ids: Starting tokens
        max_new_tokens: Tokens to generate
        beam_size: Number of beams

    Returns:
        Best sequence
    """
    # Your code here
    pass
```

---

## 🔑 Key Takeaways

### Remember These Points

1. **GPT = Embeddings + Positional Encoding + Transformer Blocks + LM Head**
   - Each component has a specific role
   - Stack blocks for depth

2. **Causal masking prevents seeing future**
   - Position i can only attend to 0...i
   - Critical for autoregressive generation
   - Different from BERT (bidirectional)

3. **Language modeling head predicts next token**
   - Linear projection to vocabulary
   - Softmax for probabilities
   - Sample to generate text

4. **Different sampling strategies**
   - Greedy: Deterministic, repetitive
   - Random: Diverse, potentially incoherent
   - Top-k: Good balance
   - Top-p: Adaptive, used in ChatGPT

5. **Scale = Performance**
   - GPT-2: 124M - 1.5B parameters
   - GPT-3: 175B parameters
   - More layers + wider = better performance

---

## ✅ Final Self-Check

You've completed Module 4! Ensure you can:

- [ ] Explain complete GPT architecture end-to-end
- [ ] Implement Mini-GPT from scratch
- [ ] Understand causal masking for generation
- [ ] Compare GPT, BERT, and Transformer architectures
- [ ] Implement different sampling strategies
- [ ] Explain how ChatGPT generates text

**If you checked all boxes:** You understand GPT! 🎉🎉🎉

**If not:** Review lessons, implement components, experiment!

---

## 💬 Common Questions

**Q: How does ChatGPT differ from GPT-3?**
A: ChatGPT = GPT-3 + Fine-tuning with RLHF (Reinforcement Learning from Human Feedback). Same architecture, better behavior.

**Q: Why is GPT "decoder-only"?**
A: It only has the decoder part of original transformer (masked attention). No encoder needed for language modeling.

**Q: Can GPT handle bidirectional context?**
A: No! Causal mask prevents it. Use BERT for bidirectional understanding.

**Q: How long does it take to train GPT-3?**
A: ~355 GPU-years! (Estimated $4.6M in compute costs)

**Q: Can I train my own GPT?**
A: Yes! Smaller versions (GPT-2 124M) are trainable on good GPUs. Start small!

---

## 📖 What's Next?

### You've Completed Module 4! 🎊

**What you now understand:**
- ✅ Attention mechanism (the core innovation)
- ✅ Self-attention (how words relate)
- ✅ Multi-head attention (multiple patterns)
- ✅ Positional encoding (word order)
- ✅ Transformer blocks (complete architecture)
- ✅ GPT architecture (text generation)

**You can now:**
- Build transformers from scratch
- Understand ChatGPT architecture
- Read research papers
- Implement text generation
- Fine-tune pre-trained models

---

### Next Module: Building Your Own LLM

**Module 5 will cover:**
1. Tokenization (BPE, WordPiece)
2. Training GPT from scratch
3. Dataset preparation
4. Optimization techniques
5. Evaluation metrics
6. Fine-tuning strategies

**Stay tuned!** 🚀

---

## 🎊 Congratulations!

**You've mastered the Transformer architecture!**

**Achievement unlocked:**
> You understand how ChatGPT, GPT-3, BERT, and all modern LLMs work!

**This knowledge is:**
- 🏆 Cutting-edge AI understanding
- 💼 Highly valuable skill
- 🧠 Foundation for advanced topics
- 🚀 Passport to building your own LLMs!

**You're now ready to:**
- Build your own language models
- Contribute to open-source LLM projects
- Understand latest research papers
- Move to advanced topics (RLHF, multimodal, etc.)

---

## 📚 Recommended Projects

### Project Ideas to Solidify Learning

**1. Mini-GPT on Simple Dataset:**
- Train on Shakespeare or Wikipedia paragraphs
- Character-level or word-level
- Small model (6 layers, 256 dims)

**2. Sentiment Analysis with BERT:**
- Use transformer encoder
- Classify text (positive/negative)
- Fine-tune pre-trained weights

**3. Custom Tokenizer:**
- Implement BPE from scratch
- Train on custom corpus
- Integrate with Mini-GPT

**4. Attention Visualization Tool:**
- Visualize attention patterns
- Interactive heatmaps
- Understand what model learned

---

## 🌟 Final Thoughts

**You've learned the architecture that changed AI forever:**

```
2017: "Attention Is All You Need" published
      ↓
2018: BERT revolutionizes NLP
      ↓
2019: GPT-2 shows amazing text generation
      ↓
2020: GPT-3 (175B) shows emergent abilities
      ↓
2022: ChatGPT changes the world
      ↓
2023-2026: AI everywhere (GPT-4, Claude, Gemini...)
```

**You now understand the core technology behind all of this!**

---

**Next:** Module 5 - Build and Train Your Own LLM!

**Keep learning! Keep building! The future of AI is in your hands!** 🚀🧠💡

---

**Module 4 Complete!** ✅🎉🏆

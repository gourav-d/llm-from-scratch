# Lesson 5.4: Building GPT with PyTorch

**Take your nanoGPT knowledge and build it the production way — using PyTorch**

---

## 🎯 What You'll Learn

- ✅ Why PyTorch instead of raw NumPy
- ✅ `nn.Module` — PyTorch's way of building neural networks
- ✅ Embedding layers in PyTorch (`nn.Embedding`)
- ✅ Multi-head attention with PyTorch (`nn.MultiheadAttention`)
- ✅ Feed-forward network with PyTorch
- ✅ A complete GPT block in PyTorch
- ✅ Full GPT model (`nn.Module` style)
- ✅ Side-by-side comparison: nanoGPT NumPy vs PyTorch
- ✅ Running a forward pass
- ✅ How this connects to GPT-2 and HuggingFace

**Time:** 3-4 hours  
**Difficulty:** ⭐⭐⭐⭐☆

---

## 🤔 Why Switch from NumPy to PyTorch?

In Lesson 5.3 you built GPT in NumPy. That was great for learning — you saw every line.  
But NumPy has real-world limitations:

| Problem | NumPy | PyTorch |
|---------|-------|---------|
| **Gradients** | Manual — you write backprop | Automatic (`autograd`) |
| **GPU support** | None | One line: `.to('cuda')` |
| **Speed** | Slow for large models | Highly optimised |
| **Ecosystem** | Standalone | HuggingFace, Lightning, etc. |
| **Saving models** | Manual | `torch.save()` |

**C#/.NET analogy:**  
NumPy GPT = writing HTTP handling from raw sockets.  
PyTorch GPT = using ASP.NET Core — same concepts, but the framework handles the plumbing.

---

## 🧱 PyTorch Building Block: `nn.Module`

Every layer and model in PyTorch inherits from `nn.Module`.  
Think of it like an **abstract base class** in C#:

```python
import torch
import torch.nn as nn

# C# equivalent:
# public class MyLayer : BaseLayer
# {
#     public override Tensor Forward(Tensor x) { ... }
# }

class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()          # Always call parent __init__
        self.linear = nn.Linear(10, 5)  # Define sub-layers here

    def forward(self, x):           # forward() = the logic
        return self.linear(x)
```

**Key rules:**
- Define learnable sub-layers inside `__init__`
- Put the forward logic inside `forward()`
- PyTorch auto-tracks all `nn.Module` children → their parameters are auto-found

---

## 📦 Step 1: Embedding Layer in PyTorch

In nanoGPT (NumPy):
```python
token_embedding = np.random.randn(vocab_size, d_model) * 0.01
x = token_embedding[token_ids]   # manual array indexing
```

In PyTorch:
```python
import torch
import torch.nn as nn

vocab_size   = 65      # characters in our Shakespeare vocab
d_model      = 256     # embedding dimension
max_seq_len  = 256     # max sequence length

# Word embedding table
token_emb = nn.Embedding(vocab_size, d_model)
# Shape: (vocab_size, d_model) — same as before, but PyTorch manages it

# Positional embedding table
pos_emb = nn.Embedding(max_seq_len, d_model)

# Usage
token_ids = torch.tensor([8, 5, 12, 12, 15])  # "Hello"

# Look up word embeddings
word_vectors = token_emb(token_ids)             # Shape: (5, 256)

# Look up position embeddings
positions = torch.arange(len(token_ids))        # [0, 1, 2, 3, 4]
pos_vectors = pos_emb(positions)                # Shape: (5, 256)

# Combine (same as nanoGPT!)
x = word_vectors + pos_vectors                  # Shape: (5, 256)

print(f"Embedding shape: {x.shape}")            # torch.Size([5, 256])
```

**What changed?** Almost nothing conceptually — `nn.Embedding` is the same lookup table, PyTorch just wraps it with autograd.

---

## 🔍 Step 2: Multi-Head Self-Attention in PyTorch

### From Scratch (same logic as Lesson 5.3, now in PyTorch)

```python
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention
    'Causal' = can only attend to past tokens (masked)

    C# analogy:
    Like a sealed class that only reads from an immutable history buffer
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # dimension per head

        # Q, K, V projections — combined into one matrix for efficiency
        # Input: (batch, seq, d_model) → Output: 3 × (batch, seq, d_model)
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

        # Regularisation
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch, seq_len, d_model)

        Returns:
            output of shape (batch, seq_len, d_model)
        """
        B, T, C = x.shape    # Batch, Time (seq_len), Channels (d_model)

        # Step 1: Project to Q, K, V
        # One big projection then split — same math, more efficient
        qkv = self.c_attn(x)                         # (B, T, 3*C)
        Q, K, V = qkv.split(C, dim=2)                # each: (B, T, C)

        # Step 2: Split into heads
        # Reshape: (B, T, C) → (B, num_heads, T, d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Scaled dot-product attention
        scale = self.d_k ** -0.5                    # 1 / sqrt(d_k)
        attn = (Q @ K.transpose(-2, -1)) * scale    # (B, num_heads, T, T)

        # Step 4: Causal mask — prevent attending to future tokens
        # torch.tril = lower triangular matrix (keep past, mask future)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn = attn.masked_fill(~mask, float('-inf'))  # -inf → 0 after softmax

        # Step 5: Softmax → attention weights
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Step 6: Weighted sum of Values
        out = attn @ V                              # (B, num_heads, T, d_k)

        # Step 7: Reassemble all heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Step 8: Final output projection
        out = self.resid_drop(self.c_proj(out))

        return out
```

**Line-by-line intuition:**

```
x        → the token+position vectors we computed in Step 1
Q, K, V  → three projections: "What I want" / "What I have" / "What I'll give"
attn     → scores: how much should each token look at each other token?
mask     → set future positions to -inf (they become 0 after softmax)
out      → weighted mix of Values, based on attention scores
```

---

## 🧠 Step 3: Feed-Forward Network in PyTorch

```python
class FeedForward(nn.Module):
    """
    Two-layer MLP applied independently to each position

    Think of it as: after attention tells us WHAT to focus on,
    feed-forward thinks WHAT TO DO with that information.

    C# analogy: a pure function — same input → same output, no side effects
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # Expand to 4x then project back
        # Same ratio as GPT-2: 768 → 3072 → 768
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),   # Expand
            nn.GELU(),                           # Smooth activation (not ReLU!)
            nn.Linear(4 * d_model, d_model),   # Contract back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

**Why GELU not ReLU?**

```
ReLU: max(0, x)         — hard cutoff at 0
GELU: x × Φ(x)         — smooth, probabilistic version

GPT-2 and GPT-3 both use GELU.
Smoother activation → slightly better gradient flow.
```

---

## 🧩 Step 4: One Transformer Block

```python
class TransformerBlock(nn.Module):
    """
    One complete transformer layer:
      - Attention (who to look at)
      - Feed-forward (what to do with that)
      - Residual connections (stability)
      - Layer norm (normalise activations)

    Stack N of these = GPT!
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.ln1  = nn.LayerNorm(d_model)         # Normalise before attention
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)

        self.ln2  = nn.LayerNorm(d_model)         # Normalise before feed-forward
        self.ff   = FeedForward(d_model, dropout)

    def forward(self, x):
        # Pre-LayerNorm style (GPT-2 uses this)
        # Residual: output = input + transformation(norm(input))
        x = x + self.attn(self.ln1(x))   # Attention branch
        x = x + self.ff(self.ln2(x))     # Feed-forward branch
        return x
```

**Why residual connections?**

```
Without:  Block1 → Block2 → ... → Block12
          Gradient must travel through all blocks → vanishes!

With:     Block1 →+→ Block2 →+→ ... →+→ Block12
                  ↑           ↑
           Skip connections carry gradients directly!
           Even 96-layer GPT-3 trains stably.
```

**Why LayerNorm?**

```
Without norm: activations can explode (very large values) or collapse (near zero)
With LayerNorm: forces mean≈0, std≈1 within each token → stable training

C# analogy: like input validation / normalisation before processing
```

---

## 🏗️ Step 5: Complete GPT Model in PyTorch

```python
class GPT(nn.Module):
    """
    Complete GPT model

    Architecture:
      Token Embedding + Position Embedding
           ↓
      N × TransformerBlock
           ↓
      LayerNorm
           ↓
      Linear (d_model → vocab_size)
           ↓
      logits (scores for next token)
    """

    def __init__(
        self,
        vocab_size,
        d_model     = 256,
        num_heads   = 4,
        num_layers  = 4,
        max_seq_len = 256,
        dropout     = 0.1
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        # --- Embedding layers ---
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)

        # --- Transformer stack ---
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, dropout)
              for _ in range(num_layers)]
        )

        # --- Final normalisation + output head ---
        self.ln_f  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between embedding and output head
        # (GPT-2 does this — saves parameters, improves performance)
        self.head.weight = self.token_emb.weight

        # Initialise weights (small values → stable early training)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialise weights following GPT-2 paper"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        Args:
            idx:     token indices (B, T)
            targets: target indices for loss (B, T) — optional

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss, or None
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, \
            f"Sequence too long: {T} > max {self.max_seq_len}"

        # 1. Token + position embeddings
        tok_emb = self.token_emb(idx)                        # (B, T, d_model)
        pos     = torch.arange(T, device=idx.device)         # [0, 1, ..., T-1]
        pos_emb = self.pos_emb(pos)                          # (T, d_model)

        x = self.drop(tok_emb + pos_emb)                     # (B, T, d_model)

        # 2. Transformer blocks
        x = self.blocks(x)                                   # (B, T, d_model)

        # 3. Final layer norm + project to vocab
        x      = self.ln_f(x)                                # (B, T, d_model)
        logits = self.head(x)                                 # (B, T, vocab_size)

        # 4. Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten: (B, T, vocab_size) → (B*T, vocab_size)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

## 🚀 Step 6: Create and Inspect the Model

```python
# Create model (same config as nanoGPT from Lesson 5.3)
model = GPT(
    vocab_size   = 65,     # 65 unique characters in Shakespeare
    d_model      = 256,
    num_heads    = 4,
    num_layers   = 4,
    max_seq_len  = 256,
    dropout      = 0.1
)

# Count parameters
params = model.count_parameters()
print(f"Parameters: {params:,}")    # ~2.7M parameters

# Move to GPU if available (one line!)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"Device: {device}")

# Forward pass
batch_size = 4
seq_len    = 32
token_ids  = torch.randint(0, 65, (batch_size, seq_len)).to(device)

logits, loss = model(token_ids)
print(f"Logits shape: {logits.shape}")   # (4, 32, 65)
print(f"Loss: {loss}")                    # None (no targets provided)

# With targets (training mode)
targets      = torch.randint(0, 65, (batch_size, seq_len)).to(device)
logits, loss = model(token_ids, targets)
print(f"Loss: {loss.item():.4f}")        # ~4.17 (log(65) — random guessing)
```

---

## 🔁 Step 7: Training Loop

```python
# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Simple training loop
def train(model, data, batch_size=32, block_size=64, max_iters=1000):
    """
    Args:
        model:      GPT model
        data:       encoded integer array (all training tokens)
        batch_size: number of sequences per batch
        block_size: context length
        max_iters:  total training steps
    """
    model.train()                  # Enable dropout, etc.

    for step in range(max_iters):

        # --- Get a random batch ---
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x  = torch.stack([data[i     : i + block_size    ] for i in ix])
        y  = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)

        # --- Forward pass ---
        logits, loss = model(x, y)

        # --- Backward pass ---
        optimizer.zero_grad()      # Clear old gradients
        loss.backward()            # Compute new gradients (autograd!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
        optimizer.step()           # Update weights

        # --- Logging ---
        if step % 100 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")

# Note: in Lesson 5.3 (nanoGPT), we had to write backward() manually.
# PyTorch does it with one line: loss.backward()  ← autograd magic!
```

**What autograd saves you:**  
In nanoGPT we manually computed every gradient.  
PyTorch's `loss.backward()` computes ALL gradients automatically — for every layer, every weight.  
This is why real models use PyTorch/TensorFlow, not NumPy.

---

## 📊 nanoGPT (NumPy) vs PyTorch — Side-by-Side

| Component | nanoGPT NumPy | PyTorch |
|-----------|--------------|---------|
| Embedding | `table[idx]` | `nn.Embedding(V, D)(idx)` |
| Attention | Manual Q@K.T / sqrt(d) | Same math, but tensors |
| Masking | `np.triu(..., -inf)` | `torch.tril(...).masked_fill(...)` |
| Backprop | Written by hand | `loss.backward()` |
| Optimizer | Manual SGD | `torch.optim.AdamW(...)` |
| GPU | Not supported | `.to('cuda')` |
| Save/load | Manual pickle | `torch.save()` / `torch.load()` |
| Parameters | Manual lists | `model.parameters()` |

**The math is identical. PyTorch just handles the boring parts.**

---

## 🔗 Connecting to GPT-2 (HuggingFace)

The model you just built IS GPT-2's architecture.  
Here's how yours maps to the real thing:

```python
# Your model
model = GPT(vocab_size=65, d_model=256, num_heads=4, num_layers=4)

# GPT-2 Small (HuggingFace)
from transformers import GPT2LMHeadModel
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

# Comparison:
# Your GPT-2      GPT-2 Small
# vocab_size=65   vocab_size=50,257 (BPE)
# d_model=256     d_model=768
# num_heads=4     num_heads=12
# num_layers=4    num_layers=12
# params=~2.7M    params=117M
# trained=No      trained=Yes (on 40GB web text)

# SAME architecture — just bigger and trained!
```

---

## 📝 Quiz

### Question 1
**What does `nn.Module` give you that a plain Python class does not?**

<details>
<summary>Click to see answer</summary>

`nn.Module` gives you:
1. **Parameter tracking** — any `nn.Linear`, `nn.Embedding`, etc. assigned in `__init__` is automatically registered. `model.parameters()` returns all of them.
2. **`.to(device)`** — moves all tensors to CPU/GPU automatically.
3. **`train()` / `eval()`** — switches dropout/batchnorm behaviour.
4. **`torch.save()` / `torch.load()`** — serialisation for free.
5. **Gradient flow** — PyTorch knows which tensors to differentiate.

You get all this just by calling `super().__init__()` and using `nn.*` layers.
</details>

---

### Question 2
**In the attention code, why do we call `masked_fill(~mask, float('-inf'))` instead of just multiplying by 0?**

<details>
<summary>Click to see answer</summary>

Because of **softmax**. After computing attention scores, we apply softmax.  
- Setting a position to **0** before softmax → softmax still gives it a non-zero weight.  
- Setting a position to **-inf** before softmax → `exp(-inf) = 0` → weight becomes exactly 0.

Only `-inf` guarantees the future tokens contribute nothing to the output.
</details>

---

### Question 3
**What is weight tying, and why does GPT use it?**

<details>
<summary>Click to see answer</summary>

Weight tying means the input embedding table (`token_emb.weight`) and the output projection (`head.weight`) **share the same matrix**.

```python
self.head.weight = self.token_emb.weight
```

**Why?**
- Input embedding: maps token IDs → vector space
- Output projection: maps vector space → token ID scores

They're doing the inverse of each other — it makes sense for them to use the same geometry.

**Benefits:**
- Saves parameters (vocab_size × d_model parameters shared instead of doubled)
- Often improves performance — consistent representation in and out
- GPT-2 and most modern LLMs use weight tying
</details>

---

### Question 4
**What does `loss.backward()` do that we had to write manually in nanoGPT?**

<details>
<summary>Click to see answer</summary>

`loss.backward()` uses **automatic differentiation (autograd)**. It:
1. Traces the entire computation graph from `loss` all the way back to each parameter
2. Computes the gradient of the loss with respect to every learnable weight
3. Stores those gradients in each parameter's `.grad` attribute

In nanoGPT we had to manually derive and write the gradient for each layer:
- dL/dW for the output projection
- dL/d(attn_weights)
- dL/d(Q), dL/d(K), dL/d(V)
- etc.

`loss.backward()` does all of this automatically for any computation graph, no matter how complex.
</details>

---

## 🧪 Exercises

### Exercise 1: Count GPT-2 Parameters
Calculate how many parameters GPT-2 Small has using your GPT class:
```python
gpt2_small = GPT(
    vocab_size   = 50257,
    d_model      = 768,
    num_heads    = 12,
    num_layers   = 12,
    max_seq_len  = 1024,
)
print(gpt2_small.count_parameters())
# Should be ~117M
```

### Exercise 2: Gradient Check
After a forward + backward pass, verify gradients exist:
```python
model = GPT(vocab_size=65, d_model=64, num_heads=2, num_layers=2)
x = torch.randint(0, 65, (2, 16))
y = torch.randint(0, 65, (2, 16))
_, loss = model(x, y)
loss.backward()

# Check that gradients were computed
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.4f}")
```

### Exercise 3: Add a Configuration Class
Refactor GPT to accept a `GPTConfig` dataclass:
```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size:   int   = 65
    d_model:      int   = 256
    num_heads:    int   = 4
    num_layers:   int   = 4
    max_seq_len:  int   = 256
    dropout:      float = 0.1

config = GPTConfig()
model  = GPT(config)   # adapt the __init__ to accept GPTConfig
```

---

## 🎓 Key Takeaways

1. **`nn.Module` is the standard** — every layer and model inherits from it
2. **PyTorch = same math, automatic gradients** — no manual backprop
3. **GPU in one line** — `.to('cuda')` moves everything
4. **Weight tying** — share input/output matrix to save params and improve results
5. **Your GPT IS GPT-2** — same architecture, just smaller scale

---

**Next Lesson:** `05_text_generation.md` — How to generate text with sampling strategies (greedy, temperature, top-k, top-p)

Run `examples/example_04_gpt_pytorch.py` to see the full model in action!

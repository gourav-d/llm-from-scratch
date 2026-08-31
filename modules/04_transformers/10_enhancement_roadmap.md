# Enhancement Roadmap: Building a Better Small LLM

This document covers:
1. Architecture improvements to make the model smarter
2. Training improvements for better efficiency
3. Real datasets you can download and use
4. A "LLaMA-lite" improved model config

---

## Section 1: Architecture Improvements

The Udemy notebook uses a GPT-2-style architecture from 2019. Modern LLMs (LLaMA, Mistral, Qwen)
use several improvements that give better results with the same parameter count.

Each improvement is explained, then compared to the current Udemy notebook implementation.

---

### 1a: RoPE -- Rotary Positional Encoding

**Current approach (Udemy notebook):**
```python
# Learned positional embeddings: one vector per position
self.positions = nn.Embedding(context, embed_size)   # (512, 384)
pos = self.positions(torch.arange(SL, device=device)) # look up by position index
x = emb + pos   # ADD position to token embedding
```

**Problem:**
- Fixed context size: trained on 512 tokens, cannot extrapolate to 513, 1000, etc.
- Positional information is "baked in" as absolute position numbers
- Position 0 and position 1 are just "arbitrary learned vectors" with no mathematical relationship

**RoPE fix:**
Instead of adding a position vector to the embedding, RoPE ROTATES the Query and Key vectors
in attention by an angle proportional to the position.

```
Position p, dimension d:
    theta_d = 10000^(-2d/D)   (D = embed_size, d = dimension index)

Rotation matrix for position p:
    Q_rotated[p] = Q[p] * cos(p * theta) + Q_perp[p] * sin(p * theta)
```

**Why rotation works:**
When you compute the dot product Q * K for positions i and j, the relative rotation angle
(i - j) * theta only depends on the DIFFERENCE between positions, not absolute values.
The attention pattern becomes position-RELATIVE, not position-absolute.

**Benefits:**
- Extrapolates to sequences longer than training length
- Better at capturing relative position relationships
- Used in: LLaMA 2/3, GPT-NeoX, PaLM, Falcon, Mistral

**Code example:**
```python
def apply_rope(q, k, position_ids, head_dim):
    # Compute rotation angles for each dimension
    theta = 10000 ** (-2 * torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim)
    angles = position_ids.unsqueeze(-1) * theta  # (SL, head_dim//2)
    cos = torch.cos(angles)                       # (SL, head_dim//2)
    sin = torch.sin(angles)                       # (SL, head_dim//2)

    # Apply rotation to interleaved pairs of dimensions
    q_even, q_odd = q[..., ::2], q[..., 1::2]    # split into even/odd dimensions
    q_rotated_even = q_even * cos - q_odd * sin
    q_rotated_odd = q_even * sin + q_odd * cos
    q_rotated = torch.stack([q_rotated_even, q_rotated_odd], dim=-1).flatten(-2)

    # Same for keys
    k_even, k_odd = k[..., ::2], k[..., 1::2]
    k_rotated_even = k_even * cos - k_odd * sin
    k_rotated_odd = k_even * sin + k_odd * cos
    k_rotated = torch.stack([k_rotated_even, k_rotated_odd], dim=-1).flatten(-2)

    return q_rotated, k_rotated
```

**C# analogy:** Like storing a date as a Unix timestamp (absolute) vs storing it as
"days since last event" (relative). Relative representations extrapolate better.

---

### 1b: RMSNorm -- Root Mean Square Normalization

**Current approach (Udemy notebook):**
```python
self.ln1 = nn.LayerNorm(embed_size)
# LayerNorm formula: y = (x - mean) / sqrt(variance + eps) * scale + shift
# Parameters: scale (gamma) and shift (beta), both of size embed_size
```

**RMSNorm formula:**
```
y = x / RMS(x) * scale
where RMS(x) = sqrt(mean(x^2))
```

**Differences:**
1. No mean subtraction (removes the centering step)
2. No shift parameter (removes the bias/beta parameter)
3. Only divides by RMS (root mean square), not standard deviation

**Benefits:**
- 10-15% faster than LayerNorm (one fewer computation: no mean subtraction)
- Similar quality on language modeling tasks
- Fewer parameters (no shift term)
- Used in: LLaMA 2/3, Gemma, Mistral, Qwen

**Code:**
```python
class RMSNorm(nn.Module):
    def __init__(self, embed_size, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embed_size))  # learnable scale (no shift)
        self.eps = eps

    def forward(self, x):
        # Compute RMS (root mean square) across the feature dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize then scale
        return x / rms * self.scale
```

**C# analogy:** Like a simplified validation check. Instead of checking "is this value
within mean +/- 2*sigma" (LayerNorm), you check "is this value within 2*RMS" (RMSNorm).
Similar effect, simpler computation.

---

### 1c: SwiGLU Activation

**Current approach (Udemy notebook):**
```python
nn.Sequential(
    nn.Linear(embed_size, 6 * embed_size),  # expand
    nn.GELU(),                               # activate with GELU
    nn.Linear(6 * embed_size, embed_size),  # compress
)
```

**SwiGLU:**
```
SwiGLU(x) = SiLU(W1 * x) * (W2 * x)
```

Two linear projections, one gated by SiLU (Sigmoid Linear Unit, also called Swish):
```python
class SwiGLU_FFN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        # Use 8/3 * embed_size for expansion (matches LLaMA)
        hidden = int(8/3 * embed_size)
        self.w1 = nn.Linear(embed_size, hidden, bias=False)   # gate projection
        self.w2 = nn.Linear(embed_size, hidden, bias=False)   # up projection
        self.w3 = nn.Linear(hidden, embed_size, bias=False)   # down projection

    def forward(self, x):
        # SiLU(w1(x)) acts as a gate: multiplied with w2(x)
        gate = F.silu(self.w1(x))   # SiLU(x) = x * sigmoid(x)
        up = self.w2(x)             # content projection
        x = gate * up               # element-wise gate (learned what to let through)
        return self.w3(x)           # compress back
```

**Why SwiGLU?**
- The gating mechanism (multiplication) allows the model to selectively "gate" information
- Smoother than GELU, better gradient flow
- Used in: LLaMA, PaLM, Gemma, Mistral
- Paper showed consistent improvement over GELU, ReLU, and Swish on language tasks

**Note on expansion factor:**
SwiGLU uses TWO up-projection matrices (w1 and w2). To keep parameter count similar:
- Old: `6 * embed_size` for one matrix
- SwiGLU: `8/3 * embed_size` for two matrices -> total `2 * 8/3 = 16/3 ~ 5.3x`
LLaMA typically rounds 8/3 * embed_size to the nearest multiple of 256 for hardware efficiency.

---

### 1d: Grouped Query Attention (GQA)

**Current approach (Udemy notebook -- Multi-Head Attention):**
```
n_heads = 7 heads
Each head has its own Q, K, V projection
Total attention parameters = 3 * embed_size * head_size * n_heads
```

**Problem:** KV cache (Key and Value matrices) is the bottleneck during inference.
For each new token generated, K and V for all previous tokens must be stored in GPU memory.
With n_heads=7, you store 7 * K matrices and 7 * V matrices.
For long sequences (4096+ tokens), this can use GBs of GPU memory.

**Multi-Query Attention (MQA) -- extreme solution:**
All query heads share ONE set of K and V matrices.
Reduces KV cache by n_heads times, but reduces model quality.

**Grouped Query Attention (GQA) -- balanced solution:**
Group query heads into groups. Each group shares K and V.
```
Example: n_heads=8, n_kv_heads=2
    Group 1 (Q heads 0-3): share K0, V0
    Group 2 (Q heads 4-7): share K1, V1
    -> 8 Q matrices, 2 K matrices, 2 V matrices
    -> KV cache reduced by 4x
```

**Code:**
```python
class GQA_Head(nn.Module):
    def __init__(self, embed_size, n_heads, n_kv_heads, head_size):
        super().__init__()
        # n_heads query projections
        self.queries = nn.Linear(embed_size, n_heads * head_size, bias=False)
        # n_kv_heads key and value projections (fewer than query heads)
        self.keys = nn.Linear(embed_size, n_kv_heads * head_size, bias=False)
        self.values = nn.Linear(embed_size, n_kv_heads * head_size, bias=False)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.groups = n_heads // n_kv_heads  # how many Q heads per K/V head

    def forward(self, x):
        BS, SL, _ = x.shape
        head_size = x.shape[-1] // self.n_heads  # approximate

        q = self.queries(x).view(BS, SL, self.n_heads, -1)    # (BS, SL, n_heads, hs)
        k = self.keys(x).view(BS, SL, self.n_kv_heads, -1)    # (BS, SL, n_kv_heads, hs)
        v = self.values(x).view(BS, SL, self.n_kv_heads, -1)  # (BS, SL, n_kv_heads, hs)

        # Expand k and v to match n_heads by repeating each kv head 'groups' times
        k = k.repeat_interleave(self.groups, dim=2)  # (BS, SL, n_heads, hs)
        v = v.repeat_interleave(self.groups, dim=2)  # (BS, SL, n_heads, hs)

        # Standard attention from here
        # ... (same as before)
```

**Used in:** LLaMA 2, LLaMA 3, Mistral 7B, Qwen 2.

---

## Section 2: Training Improvements

### 2a: Flash Attention

**The problem:**
Standard attention computes a full (SL x SL) attention matrix:
```
attn_w = q @ k.T * scale   # (BS, n_heads, SL, SL) -- potentially HUGE
```
For SL=4096 and n_heads=32: 4096 * 4096 * 32 = 536M float32 values = ~2GB just for this matrix.

**Flash Attention (Dao et al., 2022):**
Recomputes attention in small tiles (blocks) that fit in fast SRAM cache.
Never materializes the full (SL x SL) matrix in GPU DRAM (slow memory).

**Benefits:**
- 3-8x faster than standard attention
- 10-20x less GPU memory for attention
- Enables much longer context windows (16K, 32K, 100K+ tokens)
- Numerically identical to standard attention

**Usage in PyTorch 2.0:**
```python
# Instead of manual QKV attention:
# attn_w = q @ k.T * scale
# attn_w = masked_fill(...)
# attn_w = softmax(...)
# output = attn_w @ v

# Use Flash Attention directly:
output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,    # None = causal masking handled automatically
    is_causal=True,    # apply causal mask (lower triangular)
    dropout_p=0.05     # dropout applied inside attention
)
# That's it. Faster, less memory, same result.
```

**C# analogy:** Like using SIMD (Single Instruction Multiple Data) intrinsics instead of
a naive loop. The mathematical result is the same; the implementation is hardware-optimized.

---

### 2b: Gradient Accumulation

**The problem:**
Larger batch sizes train faster (more stable gradients) but need more GPU memory.
With 4GB GPU and batch_size=8, you cannot fit batch_size=128.

**Solution: gradient accumulation**
Run N forward passes with small batch, accumulate (add) gradients, then do ONE weight update.
Effective batch size = batch_size * accumulation_steps.

```python
# Example: effective_batch = 8 * 16 = 128, using only batch_size=8 at a time
accumulation_steps = 16

for i in range(total_steps):
    for micro_step in range(accumulation_steps):
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        # Divide loss by accumulation_steps to average (not sum) across micro-steps
        loss = loss / accumulation_steps
        loss.backward()   # accumulates gradients (does NOT reset them)

    # After accumulation_steps micro-batches: update weights
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)  # NOW reset gradients
    scheduler.step()
```

**C# analogy:** Like batching database writes. Instead of committing 128 rows at once
(needs 128x memory), you INSERT 8 rows to a staging table 16 times, then COMMIT once.
Same final result, much less peak memory.

---

### 2c: LR Warmup + Cosine Decay

**Current (Udemy notebook):**
Cosine annealing starts immediately from lr=3e-4 and decays to 3e-5.

**Better practice:**
```
Phase 1 - Warmup (first N steps):
    LR increases linearly from 0 to lr_max
    N = typically 500-2000 steps for small models

Phase 2 - Cosine decay (remaining steps):
    LR decreases from lr_max to lr_min following cosine curve

LR over training:
     |   /\
lr   |  /  \..
max  | /     \....
     |/           \......._____
     +---------------------------> steps
     0  warmup    train_iters
```

**Why warmup?**
At step 0, model weights are random (std=0.02 from init).
Gradients from random weights are noisy and unreliable.
Starting with large LR on noisy gradients = unstable updates.
Warmup: start small, let gradients become reliable, then increase LR.

**Code:**
```python
import math

def get_lr(step, warmup_steps, train_iters, lr_max, lr_min):
    if step < warmup_steps:
        # Linear warmup
        return lr_max * step / warmup_steps
    if step > train_iters:
        return lr_min
    # Cosine decay
    progress = (step - warmup_steps) / (train_iters - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

# Usage in training loop:
warmup_steps = 1000
for i in range(train_iters):
    current_lr = get_lr(i, warmup_steps, train_iters, lr_max=3e-4, lr_min=3e-5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr  # update LR manually each step
```

---

## Section 3: Real Training Datasets

You can replace `wiki.txt` with any of these datasets for training the small LLM.

### Recommended Datasets

**1. TinyStories (Microsoft Research)**

```
Size:       ~475 MB text
Content:    Short stories written by GPT-3/4, using simple vocabulary
Best for:   Small models that produce coherent output
Download:   pip install datasets
            from datasets import load_dataset
            ds = load_dataset("roneneldan/TinyStories", split="train")
            text = "\n".join(ds["text"])
            with open("tinystories.txt", "w") as f: f.write(text)
```

**Why start here:**
TinyStories uses simple vocabulary (like stories for 3-year-olds). A 19M parameter model
can learn to produce coherent simple stories. Using Wikipedia (like the Udemy notebook),
the model has difficulty because Wikipedia requires much more world knowledge.

With TinyStories: after 10K-20K training steps you see actual sentence-like output.
With Wikipedia: you may need 100K+ steps to see improvement.

---

**2. WikiText-103**

```
Size:       ~500 MB text
Content:    Featured and good Wikipedia articles in English
Best for:   General knowledge, similar to Udemy wiki.txt but cleaner/larger
Download:   from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            text = "\n".join(ds["text"])
```

The Udemy notebook uses a small segment of Wikipedia. WikiText-103 is 103M words
of clean Wikipedia -- much larger. Same domain, better quality, more data.

---

**3. OpenWebText**

```
Size:       ~40 GB text
Content:    Reddit-linked web pages (similar to GPT-2's training data WebText)
Best for:   Replicating GPT-2 training quality
Download:   from datasets import load_dataset
            ds = load_dataset("openwebtext", split="train")
            # Warning: 40GB -- takes hours to download and tokenize
```

This is the open-source recreation of OpenAI's WebText dataset used to train GPT-2.
Training on this will produce significantly better results than TinyStories or WikiText,
but requires much more time and GPU compute.

---

**4. FineWeb-Edu (HuggingFace)**

```
Size:       ~25 GB (10B token sample)
Content:    High-quality educational web pages, filtered by quality classifiers
Best for:   Higher quality training data than raw web crawls
Download:   from datasets import load_dataset
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
```

FineWeb-Edu was created by HuggingFace for training educational-quality LLMs.
Higher quality than OpenWebText because it is filtered for educational content.
A good middle ground between quality and size.

---

**5. BookCorpus (subset)**

```
Size:       ~1 GB (10% sample)
Content:    Books from unpublished authors (same data used for early GPT training)
Best for:   Long-form narrative text, coherent paragraphs
Download:   from datasets import load_dataset
            ds = load_dataset("bookcorpus", split="train[:10%]")
            text = "\n".join(ds["text"])
```

Books have longer coherent paragraphs than web text. Good for training a model
to generate multi-sentence coherent narrative.

---

### Quick Start Script for Downloading TinyStories

```python
# download_tinystories.py
# Downloads TinyStories and saves as a text file for training

from datasets import load_dataset

print("Downloading TinyStories dataset...")
ds = load_dataset("roneneldan/TinyStories", split="train")
print(f"Downloaded {len(ds)} stories")

print("Saving to tinystories.txt...")
with open("tinystories.txt", "w", encoding="utf-8") as f:
    for story in ds["text"]:
        f.write(story + "\n\n")  # blank line between stories

print("Done! File saved as tinystories.txt")
print("Now run: python small_tokenizer_standalone.py --input tinystories.txt --model_prefix ts_tokenizer")
print("Then:    python small_llm_standalone.py (after updating tokenizer_model_file = 'ts_tokenizer.model')")
```

---

## Section 4: The "LLaMA-Lite" Improved Model Config

Combine all the above improvements into a single better model at the same ~19M parameter count.

### Current Config (Udemy):
```python
embed_size = 384
n_layers = 7
n_heads = 7
# Architecture:
#   Positional: learned absolute embeddings
#   Normalization: LayerNorm
#   Activation: GELU + 6x expansion
#   Attention: Multi-Head (each head has own K, V)
```

### Improved "LLaMA-lite" Config:
```python
embed_size = 384
n_layers = 6
n_heads = 6
n_kv_heads = 2      # GQA: 6 Q heads, 2 KV heads

# Architecture changes (same parameter count ~19M):
#   Positional: RoPE (no extra parameters, better extrapolation)
#   Normalization: RMSNorm (15% faster, fewer parameters)
#   Activation: SwiGLU with 8/3 * embed_size = 1024 hidden
#   Attention: GQA (n_heads=6, n_kv_heads=2)
#   Attention: Flash Attention via F.scaled_dot_product_attention

# Training changes:
#   LR: warmup 500 steps then cosine decay
#   Gradient accumulation: 4 steps (effective batch = 32)
#   Optimizer: AdamW with param groups (weight decay only on weight matrices)
```

### Why This Is Better:
- RoPE: handles any sequence length, better relative position understanding
- RMSNorm: 15% faster per forward pass
- SwiGLU: better language modeling performance with same parameter count
- GQA: 3x less KV cache memory (enables longer inference sequences)
- Flash Attention: 3-8x faster attention, enables larger context

### Parameter Count Comparison:

| Component | Udemy | LLaMA-lite | Change |
|---|---|---|---|
| Token embedding | 4096 * 384 = 1.57M | same | -- |
| Position embedding | 512 * 384 = 0.20M | 0 (RoPE is free) | -0.20M |
| Attention per block | 4 * 384 * 54 * 7 = 0.58M | ~0.55M (GQA) | -0.03M |
| FFN per block | 2 * 384 * 2304 = 1.77M | 2 * 384 * 1024 = 0.79M | -0.98M |
| Norm per block | LayerNorm: 2*384*2 = 1.5K | RMSNorm: 384 | ~same |
| Total (7 blocks vs 6) | ~19M | ~17M | similar |

You can add one more layer or increase embed_size slightly to hit the same 19M.

---

## Summary: Recommended Next Steps

1. **Try TinyStories** with the current Udemy notebook code:
   - Download: `download_tinystories.py` above
   - Retrain tokenizer: `python small_tokenizer_standalone.py --input tinystories.txt`
   - Retrain model: update `tokenizer_model_file` and run `small_llm_standalone.py`
   - Expected: coherent simple stories after 10K-20K steps

2. **Add Flash Attention** (one-line change, biggest performance win):
   - Replace the manual attention in `Head.forward()` with `F.scaled_dot_product_attention()`
   - No change to model behavior. 3-8x faster.

3. **Add LR warmup** (improves training stability):
   - Replace `CosineAnnealingLR` with the `get_lr()` function from Section 2c

4. **Try RMSNorm** (15% faster, simple swap):
   - Replace `nn.LayerNorm(embed_size)` with `RMSNorm(embed_size)` in Block

5. **Advanced: RoPE + SwiGLU** (requires architectural changes but biggest quality win):
   - Replace positional embeddings with RoPE in Head.forward()
   - Replace ForwardLayer with SwiGLU_FFN
   - This is the full LLaMA-lite upgrade

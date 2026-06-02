# Module 16 — Python Guide

## Python Features Used in This Module

This guide explains Python features you will encounter in the examples and exercises.

---

## 1. Dataclasses

Used to define configuration and model hyperparameters cleanly.

```python
from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    num_timesteps: int = 100      # T — total noise levels
    vocab_size: int = 50257       # GPT-2 vocabulary size
    mask_token_id: int = 50256    # ID for [MASK]
    d_model: int = 256            # embedding dimension
    num_heads: int = 4            # attention heads
    num_layers: int = 4           # transformer layers

config = DiffusionConfig(num_timesteps=50, d_model=512)
print(config.d_model)   # 512
```

**C# equivalent:**
```csharp
public record DiffusionConfig(
    int NumTimesteps = 100,
    int VocabSize = 50257,
    int MaskTokenId = 50256,
    int DModel = 256
);
```

---

## 2. numpy random sampling

Used to sample timesteps and apply masking at random noise levels.

```python
import numpy as np

# Sample a random timestep between 1 and T
t = np.random.randint(1, num_timesteps + 1)

# Sample mask: True where a token gets masked
mask_prob = t / num_timesteps           # higher t = more masking
mask = np.random.random(seq_len) < mask_prob

# Apply mask to token ids
masked_tokens = tokens.copy()
masked_tokens[mask] = mask_token_id     # replace masked positions
```

**C# equivalent:**
```csharp
var random = new Random();
int t = random.Next(1, numTimesteps + 1);
double maskProb = (double)t / numTimesteps;
var mask = tokens.Select(_ => random.NextDouble() < maskProb).ToArray();
```

---

## 3. np.where — conditional element selection

```python
# np.where(condition, value_if_true, value_if_false)
masked = np.where(mask, mask_token_id, tokens)

# Equivalent to:
masked = np.array([
    mask_token_id if mask[i] else tokens[i]
    for i in range(len(tokens))
])
```

**C# equivalent:**
```csharp
var masked = tokens.Select((t, i) => mask[i] ? maskTokenId : t).ToArray();
```

---

## 4. PyTorch nn.ModuleList

Used to create N independent output heads (for MTP).

```python
import torch.nn as nn

class MultiTokenPredictor(nn.Module):
    def __init__(self, d_model, vocab_size, n_heads):
        super().__init__()
        self.transformer = TransformerEncoder(d_model)
        # Create N prediction heads — each is an independent Linear layer
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size)
            for _ in range(n_heads)
        ])
```

`nn.ModuleList` is like `List<nn.Module>` but PyTorch knows to register all items
as submodules (so their parameters are included in `model.parameters()`).

**C# analogy:**
```csharp
// Like a List<ILayer> where each layer is registered for training
var heads = new List<LinearLayer>(
    Enumerable.Range(0, nHeads).Select(_ => new LinearLayer(dModel, vocabSize))
);
```

---

## 5. zip() — iterate multiple sequences together

Used to pair heads with their target offsets in MTP.

```python
offsets = [1, 2, 3, 4]          # predict t+1, t+2, t+3, t+4

for head, offset in zip(self.heads, offsets):
    logits = head(hidden_states)                    # shape: [batch, seq, vocab]
    targets = input_ids[:, offset:]                 # shift targets by offset
    loss += F.cross_entropy(logits[:, :-offset, :].reshape(-1, vocab_size),
                            targets.reshape(-1))

# zip() pairs: (head1, offset=1), (head2, offset=2), ...
```

**C# equivalent:**
```csharp
foreach (var (head, offset) in heads.Zip(offsets))
{
    var logits = head.Forward(hiddenStates);
    // compute loss...
}
```

---

## 6. F.cross_entropy with ignore_index

Used in diffusion loss to ignore non-masked positions.

```python
import torch.nn.functional as F

# Only compute loss on masked positions
# Use ignore_index to skip non-masked positions

# targets: original tokens, but -100 at non-masked positions
targets = original_tokens.clone()
targets[~mask] = -100      # -100 = ignore this position

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1),
    ignore_index=-100          # skip positions with -100
)
```

**Why -100?** PyTorch convention — `ignore_index=-100` is the default, so no loss
is computed for those positions. No gradients flow from ignored positions.

---

## 7. einsum — generalized matrix operations

Used in attention computations within the transformer.

```python
import torch

# Matrix multiply: batch × seq × d_model  ×  d_model × vocab
# Using einsum: "bsd,dv->bsv"
# b=batch, s=sequence, d=d_model, v=vocab
output = torch.einsum("bsd,dv->bsv", hidden_states, weight)

# Equivalent to:
output = hidden_states @ weight        # same result, einsum is more explicit
```

**C# analogy:**
`einsum` is like a generalized matrix multiply with named dimensions.
Rare in C# — you'd typically use a library with explicit batch-matmul methods.

---

## 8. List comprehension with conditional

Used to build masking schedules and noise levels.

```python
# Linear noise schedule: prob = t/T for each timestep
T = 100
schedule = [t / T for t in range(T + 1)]
# schedule = [0.0, 0.01, 0.02, ..., 1.0]

# Cosine schedule (smoother)
import math
cosine_schedule = [
    1 - math.cos((t / T) * math.pi / 2)
    for t in range(T + 1)
]
```

**C# equivalent:**
```csharp
var schedule = Enumerable.Range(0, T + 1).Select(t => (double)t / T).ToList();
```

---

## 9. torch.topk — get top-k values

Used in diffusion generation to select highest-confidence tokens to commit.

```python
# At each diffusion step, commit the most confident tokens
probs = torch.softmax(logits, dim=-1)           # shape: [seq_len, vocab]
max_probs, predicted_tokens = probs.max(dim=-1) # highest prob token at each pos

# Only commit tokens above threshold
threshold = 0.8
commit_mask = max_probs > threshold

# Update: commit high-confidence tokens, leave rest as [MASK]
new_tokens = torch.where(commit_mask, predicted_tokens, mask_token_id_tensor)
```

---

## 10. Context manager: torch.no_grad()

Used during inference — no need to compute gradients.

```python
with torch.no_grad():
    # Generation loop — inference only, no training
    tokens = torch.full((1, seq_len), mask_token_id)    # all [MASK]
    
    for t in range(T, 0, -1):
        logits = model(tokens, t)
        # ... unmask tokens
```

**C# analogy:**
```csharp
using (var noGrad = model.NoGrad())  // hypothetical
{
    // inference only, no gradient computation
}
```

Without `torch.no_grad()`, PyTorch builds a computation graph for every operation
(needed for backprop during training). At inference, this wastes memory.
`no_grad()` disables graph building → less memory, faster inference.

---

## Quick Reference

| Python / PyTorch | C# Equivalent | Used For |
|-----------------|---------------|----------|
| `@dataclass` | `record` / `class` with properties | Config objects |
| `np.random.randint` | `random.Next()` | Sample timestep t |
| `np.where(cond, a, b)` | `cond ? a : b` (element-wise) | Apply masking |
| `nn.ModuleList` | `List<ILayer>` (registered) | Multiple heads (MTP) |
| `zip(a, b)` | `a.Zip(b)` | Pair heads + offsets |
| `F.cross_entropy(..., ignore_index=-100)` | Custom loss skip | Masked-only loss |
| `torch.no_grad()` | n/a (auto in eval mode) | Inference |
| `torch.topk` | LINQ `OrderByDescending().Take(k)` | Top-k sampling |

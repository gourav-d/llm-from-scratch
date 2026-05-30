# Lesson 4: Gradient Checkpointing + ZeRO Optimizer

## Two Problems, Two Solutions

This lesson covers two memory-saving techniques for training large models:

| Problem | Solution |
|---------|----------|
| Training stores too many activations → GPU OOM | Gradient Checkpointing |
| Each GPU stores full duplicate of model state → GPU OOM | ZeRO Optimizer |

Both techniques let you train models that would otherwise not fit in GPU memory.

---

## Part 1: Gradient Checkpointing

### Why Does Training Use So Much Memory?

During training, the backward pass (computing gradients) needs values from the forward pass.

Specifically, for each layer, the backward pass needs:
- The **activations** (outputs) computed during the forward pass
- The **input** to that layer

So during the forward pass, PyTorch stores **everything** in memory.

```
+------------------------------------------------------------------+
|  MEMORY DURING TRAINING (without checkpointing)                  |
+------------------------------------------------------------------+
|                                                                  |
|  Forward pass stores all activations:                            |
|                                                                  |
|  Layer 1 activation  [saved in memory] ←──────────────┐         |
|       ↓                                               |         |
|  Layer 2 activation  [saved in memory] ←─────────┐   |         |
|       ↓                                           |   |         |
|  Layer 3 activation  [saved in memory] ←──────┐  |   |         |
|       ↓                                        |  |   |         |
|  Layer 4 activation  [saved in memory] ←───┐  |  |   |         |
|       ↓                                     |  |  |   |         |
|  Loss                                       |  |  |   |         |
|       ↓ (backward pass uses saved values)   ↓  ↓  ↓   ↓         |
|  Compute gradients ←── needs all of these ──────────────        |
|                                                                  |
|  Memory = O(N_layers × sequence_length × hidden_size)           |
|  For 32 layers, 4K tokens, 4096 hidden: ~8 GB just for acts     |
+------------------------------------------------------------------+
```

### The Checkpointing Solution

Gradient checkpointing saves memory by NOT storing all activations.
Instead, it only saves activations at **checkpoint boundaries**.

During the backward pass, it **recomputes** the unsaved activations on-the-fly.

```
+------------------------------------------------------------------+
|  GRADIENT CHECKPOINTING: TRADING COMPUTE FOR MEMORY             |
+------------------------------------------------------------------+
|                                                                  |
|  WITH CHECKPOINTING (every 4 layers is a checkpoint):           |
|                                                                  |
|  Layer 1 activation  [DISCARDED after forward]                   |
|       ↓                                                         |
|  Layer 2 activation  [DISCARDED after forward]                   |
|       ↓                                                         |
|  Layer 3 activation  [DISCARDED after forward]                   |
|       ↓                                                         |
|  Layer 4 activation  [SAVED ← checkpoint]                        |
|       ↓                                                         |
|  Loss                                                            |
|       ↓ (backward pass)                                          |
|  Need layer 3 activation?                                        |
|    → RECOMPUTE from layer 4 checkpoint (costs time, saves RAM)  |
|                                                                  |
|  Memory: O(sqrt(N_layers)) instead of O(N_layers)               |
|  Speed cost: ~30-40% slower (one extra forward recompute)        |
+------------------------------------------------------------------+
```

### Memory Savings Calculation

```
WITHOUT checkpointing:
  32 layers × 4096 tokens × 4096 hidden × 2 bytes (bf16) = 1 GB per batch item

WITH checkpointing (checkpoint every layer):
  Only 1 activation stored at a time ≈ 32× less memory

WITH checkpointing (checkpoint every 4 layers):
  4 activations stored at a time ≈ 8× less memory

TRADE: save 8-32× memory, pay ~33% speed penalty
```

### PyTorch Code

```python
import torch
from torch.utils.checkpoint import checkpoint

class TransformerLayer(torch.nn.Module):
    def forward(self, x):
        # ... attention + FFN computations ...
        return x

class CheckpointedTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TransformerLayer() for _ in range(32)
        ])

    def forward(self, x):
        for layer in self.layers:
            # With checkpointing: activations NOT stored, recomputed in backward
            x = checkpoint(layer, x, use_reentrant=False)
            # Without checkpointing: x = layer(x)
        return x
```

One line change: `layer(x)` → `checkpoint(layer, x)`. That's it.
Memory savings: up to 10× depending on model architecture.

---

## Part 2: ZeRO Optimizer

### The Multi-GPU Memory Problem

When you have multiple GPUs and use standard **Data Parallel Training**:

```
+------------------------------------------------------------------+
|  DATA PARALLEL TRAINING (standard)                               |
+------------------------------------------------------------------+
|                                                                  |
|  GPU 0:  full model copy  |  full optimizer state  |  batch 0   |
|  GPU 1:  full model copy  |  full optimizer state  |  batch 1   |
|  GPU 2:  full model copy  |  full optimizer state  |  batch 2   |
|  GPU 3:  full model copy  |  full optimizer state  |  batch 3   |
|                                                                  |
|  Problem: Every GPU stores the SAME model 4 times!              |
|  Adam optimizer state = 3× model size (weights + m + v)         |
|  For 7B params: each GPU needs 84 GB. Wasteful.                 |
+------------------------------------------------------------------+
```

**ZeRO** (Zero Redundancy Optimizer) eliminates this duplication.
Developed by Microsoft DeepSpeed team (2020).

### ZeRO Stages

ZeRO has three stages, each removing more redundancy:

```
+=====================================================================+
|  ZERO STAGES                                                        |
+=====================================================================+
|                                                                     |
|  Stage 0 (No ZeRO — standard DDP):                                  |
|    Each GPU stores:  params + gradients + optimizer state           |
|    Memory per GPU:   4K bytes (where K = num params)                |
|    Communication:    gradients all-reduced after each batch         |
|                                                                     |
|  Stage 1 (ZeRO-1):                                                  |
|    Shard:  optimizer state across GPUs                              |
|    Each GPU stores:  params + gradients + 1/N optimizer state       |
|    Memory per GPU:   4K/N for opt state + 2K for params + 2K grads  |
|    Savings:  ~4× for Adam with 4 GPUs                               |
|                                                                     |
|  Stage 2 (ZeRO-2):                                                  |
|    Shard:  optimizer state + gradients across GPUs                  |
|    Each GPU stores:  params + 1/N gradients + 1/N optimizer state   |
|    Memory per GPU:   2K + 2K/N + 4K/N                               |
|    Savings:  ~8× for Adam with 4 GPUs                               |
|                                                                     |
|  Stage 3 (ZeRO-3):                                                  |
|    Shard:  optimizer state + gradients + PARAMETERS                 |
|    Each GPU stores:  1/N of everything                              |
|    Memory per GPU:   (2K + 2K + 4K) / N = 8K/N                     |
|    Savings:  ~8× vs Stage 0, scales linearly with GPU count         |
|                                                                     |
+=====================================================================+
```

### Visual: ZeRO-3 Example

```
Model: 8B parameters, Adam optimizer, 8 GPUs

WITHOUT ZeRO:
  GPU 0: [8B weights] [8B gradients] [8B m] [8B v] = 32B params worth of state
  GPU 1: [8B weights] [8B gradients] [8B m] [8B v] = 32B (identical to GPU 0!)
  ...
  GPU 7: [8B weights] [8B gradients] [8B m] [8B v] = 32B (identical!)
  Total memory used: 32B × 8 = 256B (but 7/8 is pure duplication)

WITH ZeRO-3:
  GPU 0: [1B weights] [1B gradients] [1B m] [1B v] = 4B (its shard only)
  GPU 1: [1B weights] [1B gradients] [1B m] [1B v] = 4B (different shard)
  ...
  GPU 7: [1B weights] [1B gradients] [1B m] [1B v] = 4B (different shard)
  Total memory used: 4B × 8 = 32B (same total, but only 4B per GPU!)
  Memory per GPU: 4B instead of 32B = 8× savings
```

### ZeRO Communication Overhead

ZeRO is not free. Sharding parameters means GPUs must share data during forward/backward:

| Stage | Extra Communication | Communication Volume |
|-------|--------------------|--------------------|
| Stage 1 | Gather optimizer states for update | Low |
| Stage 2 | Gather gradients before optimizer step | Medium |
| Stage 3 | All-gather parameters during forward pass | High |

Stage 3 requires GPUs to fetch parameter shards from other GPUs during the forward pass.
This increases communication by ~1.5× vs standard DDP.
With fast NVLink (600 GB/s), this overhead is small.
With slow inter-machine Ethernet, Stage 3 can be slower than Stage 2.

### ZeRO-Offload

ZeRO can also offload optimizer state and gradients to CPU RAM:

```
ZeRO-Offload (extension to Stage 2/3):
  GPU: store parameters (needed for fast forward/backward)
  CPU: store optimizer state (Adam m and v) — updated on CPU

Benefit: train a 10B+ model on a single consumer GPU (24 GB VRAM)
Cost:    optimizer step is slower (CPU compute + PCIe transfer)
         training speed drops ~20-30% vs GPU-only

For single GPU: ZeRO-Offload is the most practical option.
```

### Using ZeRO with DeepSpeed

```python
# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",       # set to "none" to disable offload
            "pin_memory": true
        }
    },
    "bf16": {
        "enabled": true
    },
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4
}

# In Python:
import deepspeed

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="deepspeed_config.json"
)

# Training loop is the same — DeepSpeed handles the sharding
for batch in dataloader:
    loss = model(batch)
    model.backward(loss)        # instead of loss.backward()
    model.step()                # instead of optimizer.step()
```

---

## Combining Both Techniques

In practice, large training runs use BOTH together:

```
+------------------------------------------------------------------+
|  REAL WORLD TRAINING SETUP (e.g., LLaMA training)               |
+------------------------------------------------------------------+
|                                                                  |
|  Technique                  Memory Savings    Speed Cost         |
|  ─────────────────────────  ──────────────    ──────────────     |
|  bf16 training              ~2x               +50% throughput    |
|  Flash Attention            ~16x for attention  +2-4x speed      |
|  Gradient checkpointing     ~8-32x activations  -33% speed       |
|  ZeRO Stage 2               ~8x per GPU       +10-20% comm       |
|  ZeRO-Offload (optional)    ~10x per GPU      -20-30% speed      |
|                                                                  |
|  Combined: train a 70B model on 8 × A100 (80GB) GPUs            |
|  Without any of this: you would need 8 × A100 just for weights  |
+------------------------------------------------------------------+
```

---

## C# Analogy: Lazy Evaluation and Sharded State

```csharp
// Gradient Checkpointing ≈ Lazy Evaluation / Recompute on Demand
//
// Without checkpointing (like eager loading):
// var allActivations = layers.Select(l => l.Compute(input)).ToList();
// // All intermediate results stored in memory
//
// With checkpointing (like lazy IEnumerable):
// var activations = layers.Select(l => (Func<Tensor>)(() => l.Compute(input)));
// // Intermediate values discarded, recomputed only when needed
// foreach (var getActivation in activations.Reverse())
//     gradient = ComputeGradient(getActivation(), gradient);
//
// ---
// ZeRO ≈ Sharded/Distributed State
//
// Without ZeRO (like each microservice has full DB replica):
// // Service 0: full database copy (100 GB)
// // Service 1: full database copy (100 GB)  ← wasteful duplication
// // Service 2: full database copy (100 GB)
//
// With ZeRO (like proper database sharding):
// // Service 0: shard 0 of database (33 GB)
// // Service 1: shard 1 of database (33 GB)  ← each holds only its part
// // Service 2: shard 2 of database (33 GB)
// // When service 0 needs shard 1: request it from service 1 (network call)
```

---

## When to Use Each

```
+------------------------------------------------------------------+
|  DECISION GUIDE                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  Single GPU, model fits in VRAM?                                 |
|  → No need for ZeRO. Use gradient checkpointing if close to OOM  |
|                                                                  |
|  Single GPU, model doesn't fit?                                  |
|  → ZeRO-Offload Stage 2 or 3. Moves optimizer to CPU.           |
|                                                                  |
|  Multiple GPUs, same machine (NVLink)?                           |
|  → ZeRO Stage 2 or 3. Fast interconnect handles communication.  |
|                                                                  |
|  Multiple GPUs, different machines (Ethernet)?                   |
|  → ZeRO Stage 1 or 2. Stage 3 too much communication overhead.  |
|                                                                  |
|  Activation memory is the bottleneck?                            |
|  → Gradient checkpointing. Combine with ZeRO for best results.  |
|                                                                  |
|  Just want simplest setup?                                       |
|  → ZeRO Stage 2 + gradient checkpointing. Usually sufficient.   |
+------------------------------------------------------------------+
```

---

## Quiz Questions

**Q1**: Gradient checkpointing trades what for what?
        a) Memory for speed (uses more memory, trains faster)
        b) Speed for memory (trains slower, uses less memory)
        c) Accuracy for speed (less accurate, trains faster)
        d) Parameters for activations (fewer parameters, more activation storage)

**Q2**: In ZeRO Stage 3, what is sharded across GPUs?
        a) Only optimizer state (Adam's m and v)
        b) Optimizer state and gradients
        c) Optimizer state, gradients, AND model parameters
        d) Only the model parameters

**Q3**: For a 32-layer transformer, gradient checkpointing saves roughly:
        a) No memory (only a small constant amount)
        b) 2x memory
        c) 8-32x activation memory (depending on checkpoint interval)
        d) 100x memory

*(Answers: Q1=b, Q2=c, Q3=c)*

---

## Key Takeaways

1. Training stores all layer activations for the backward pass → huge memory cost
2. Gradient checkpointing discards activations and recomputes them during backward
3. Memory savings: 8-32x; Speed cost: ~33%
4. ZeRO eliminates redundant copies of model state across GPUs
5. ZeRO Stage 1: shard optimizer state; Stage 2: + gradients; Stage 3: + parameters
6. ZeRO-Offload: move optimizer to CPU, enabling huge models on single GPU
7. In practice: combine bf16 + Flash Attention + gradient checkpointing + ZeRO Stage 2

---

*Next: Lesson 5 — Dataset Streaming (training on terabyte datasets without downloading them)*

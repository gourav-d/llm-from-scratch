# Lesson 2: Mixed Precision Training

## The Memory Problem

Training a 7B parameter model in standard fp32 precision:

```
7,000,000,000 parameters
× 4 bytes per parameter (fp32)
= 28 GB just for weights

Add optimizer state (Adam stores 3 copies):
= 28 GB × 3 = 84 GB total

A high-end consumer GPU (RTX 4090) has 24 GB VRAM.
You cannot even store the weights, let alone train.
```

The solution: use smaller number formats.

**Mixed precision training cuts this to ~28 GB** — trainable on a high-end consumer GPU.

---

## Number Format Refresher

A floating-point number has three parts:

```
+--------+------------+------------------+
|  Sign  |  Exponent  |     Mantissa     |
|  1 bit |  e bits    |     m bits       |
+--------+------------+------------------+

Sign:     0 = positive, 1 = negative
Exponent: sets the range (how big or small numbers can be)
Mantissa: sets the precision (how many decimal places)
```

---

## The Four Number Types

```
+=========================================================================+
|  FORMAT   | BITS | SIGN | EXPONENT | MANTISSA | MAX VALUE   | BYTES    |
+===========+======+======+==========+==========+=============+==========+
|  fp32     |  32  |  1   |    8     |    23    | ~3.4 × 10^38 | 4 bytes  |
|  fp16     |  16  |  1   |    5     |    10    | ~65,504      | 2 bytes  |
|  bf16     |  16  |  1   |    8     |    7     | ~3.4 × 10^38 | 2 bytes  |
|  int8     |   8  |  1   |    —     |    —     | 127          | 1 byte   |
+=========================================================================+

KEY OBSERVATIONS:
  fp16  = half the bytes of fp32, but MAX VALUE is only 65,504 (danger zone)
  bf16  = half the bytes of fp32, SAME range as fp32, less precision
  int8  = 1/4 the bytes, only for inference (cannot train with it)
```

---

## fp16 vs bf16: The Critical Difference

```
fp32:  0 | 11111111 | 11111111111111111111111
         exponent   mantissa
         (8 bits)   (23 bits)
         range = huge

fp16:  0 | 11111 | 1111111111
         exponent  mantissa
         (5 bits)  (10 bits)
         range = only up to 65,504 ← OVERFLOW RISK

bf16:  0 | 11111111 | 1111111
         exponent    mantissa
         (8 bits)    (7 bits)
         range = same as fp32 ← SAFE
         precision = less ← acceptable for most training
```

**Why fp16 overflows:**
Gradients in deep neural networks can be very small (like 1e-7) or very large (like 1e5).
fp16 cannot represent numbers larger than ~65,504.
A gradient of 70,000 would overflow to infinity → training crashes.

**Why bf16 is better for training:**
Same 8-bit exponent as fp32, so same range.
Less mantissa precision is usually fine for gradient updates.

**Rule of thumb:**
- Training → use **bf16** (safe range)
- Inference on older hardware → use **fp16** (often hardware-accelerated)
- Modern hardware (A100, H100, RTX 40xx) → bf16 natively supported

---

## What Is Mixed Precision Training?

Not everything in training needs high precision.
Not everything can be done in low precision.

```
+------------------------------------------------------------------+
|  MIXED PRECISION TRAINING STRATEGY                               |
+------------------------------------------------------------------+
|                                                                  |
|  COMPUTE in fp16 or bf16 (FAST, small memory):                   |
|    - Forward pass                                                |
|    - Backward pass (compute gradients)                           |
|    - Matrix multiplications                                      |
|                                                                  |
|  ACCUMULATE in fp32 (PRECISE, avoids drift):                     |
|    - Master copy of model weights                                |
|    - Optimizer state (Adam's m and v)                            |
|    - Gradient accumulation                                       |
|                                                                  |
|  RESULT:                                                         |
|    Compute speed = fp16/bf16 speed (2x faster on modern GPUs)    |
|    Accuracy = fp32 accuracy (no drift from low precision)        |
|    Memory = roughly half of pure fp32 training                   |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Memory Calculation: Before and After

```
MODEL: 7B parameters

BEFORE (pure fp32):
  Weights:        7B × 4 bytes = 28 GB
  Gradients:      7B × 4 bytes = 28 GB
  Adam (m):       7B × 4 bytes = 28 GB
  Adam (v):       7B × 4 bytes = 28 GB
  TOTAL:          112 GB

AFTER (mixed precision):
  Weights fp16:   7B × 2 bytes = 14 GB   ← compute with this
  Weights fp32:   7B × 4 bytes = 28 GB   ← master copy (optimizer)
  Gradients fp16: 7B × 2 bytes = 14 GB
  Adam (m) fp32:  7B × 4 bytes = 28 GB
  Adam (v) fp32:  7B × 4 bytes = 28 GB
  TOTAL:          112 GB ← wait, same??

  But in practice: gradients are temporary, activations are fp16.
  Total VRAM usage drops 30-50% depending on batch size and sequence length.
  The biggest win is SPEED: fp16 tensor cores are 2x-16x faster.
```

---

## Loss Scaling (fp16 Only)

When using fp16, gradients can underflow (become zero when too small).

```
+------------------------------------------------------------------+
|  THE UNDERFLOW PROBLEM                                           |
+------------------------------------------------------------------+
|                                                                  |
|  Real gradient:       0.000001234  (very small)                  |
|  fp16 can't store it: rounds to 0.0 (underflow to zero)          |
|  Gradient is LOST.    Training fails silently.                   |
|                                                                  |
|  FIX: Loss Scaling                                               |
|                                                                  |
|  1. Multiply loss by a large scale factor S (e.g., 1024)         |
|     Scaled loss = loss × 1024                                    |
|                                                                  |
|  2. Backpropagate through scaled loss                            |
|     Scaled gradient = real_gradient × 1024                      |
|     Now it fits in fp16 range!                                   |
|                                                                  |
|  3. Before optimizer step, divide gradients back by S            |
|     Real gradient = scaled_gradient / 1024                      |
|                                                                  |
|  PyTorch does this automatically via GradScaler.                 |
|  bf16 does NOT need loss scaling (wider range = no underflow).   |
+------------------------------------------------------------------+
```

---

## PyTorch Code: How to Use Mixed Precision

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyTransformer().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # only needed for fp16, optional for bf16

for batch in dataloader:
    inputs, targets = batch

    # --- Forward pass in bf16 ---
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    # --- Backward pass ---
    # For bf16: just backward() directly
    # For fp16: use scaler to handle loss scaling
    scaler.scale(loss).backward()

    # --- Optimizer step ---
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

The key line is `with autocast(...)`.
Everything inside runs in bf16.
PyTorch automatically promotes some operations back to fp32 when needed.

---

## Automatic Promotion

PyTorch's autocast is smart. It knows which operations need full precision.

```
+------------------------------------------------------------------+
|  OPERATIONS RUN IN bf16 (autocast):                              |
|    - Linear layers (matmul)                                      |
|    - Convolutions                                                |
|    - Attention Q × K matrix multiply                            |
|                                                                  |
|  OPERATIONS PROMOTED BACK TO fp32 (autocast keeps safe):        |
|    - Softmax (numerical stability)                               |
|    - Layer normalization                                         |
|    - Loss functions                                              |
|    - Reductions (sum, mean across large dimensions)              |
+------------------------------------------------------------------+
```

You do NOT need to manually track this. `autocast` handles it.

---

## Speed Comparison

On an NVIDIA A100 GPU:

```
Operation          | fp32 speed | bf16 speed | Speedup
-------------------+------------+------------+--------
Matrix multiply    | 312 TFLOP/s| 312 TFLOP/s| 1x  (same FLOP count)
Tensor core use    | Low        | High       | 2x-16x wall-clock speedup
Memory bandwidth   | High usage | Low usage  | 2x less memory read/write

Overall training speedup with bf16: typically 1.5x - 3x
```

The speedup comes from:
1. Tensor cores are optimized for 16-bit operations
2. 2× less data to read/write from memory
3. Larger effective batch sizes fit in same VRAM

---

## C# Analogy: int, long, decimal

```csharp
// In C#, you choose numeric types by size vs precision tradeoff:
//
// decimal → most precise, slowest, 16 bytes
//           "I am handling financial transactions"
//
// double (float64) → standard precision, 8 bytes
//           "I am doing normal math" (analogous to fp32)
//
// float (float32) → half precision, 4 bytes
//           "I am doing physics simulation, speed matters more"
//
// short (int16) → smallest integer, 2 bytes
//           "I am storing pixel values 0-255"
//
// In ML training:
//   fp32 = double  (default, most precise)
//   bf16 = float   (half size, same range, less precision)
//   fp16 = short   (half size, smaller range, overflow risk)
//
// Mixed precision = use float for computation, double for accumulation.
// You get speed of float, precision of double. Best of both worlds.
```

---

## When to Use Each

```
+------------------------------------------------------------------+
|  DECISION GUIDE                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  Training on A100, H100, or RTX 40xx?                            |
|  → Use bf16. Fast + safe. Best choice.                           |
|                                                                  |
|  Training on older GPU (V100, RTX 30xx, GTX)?                    |
|  → Use fp16 + GradScaler. bf16 may not be hardware-accelerated.  |
|                                                                  |
|  Training on CPU?                                                |
|  → fp32. No benefit to fp16/bf16 on CPU.                         |
|                                                                  |
|  Inference only?                                                 |
|  → fp16 or INT8. You don't need master weights.                  |
|                                                                  |
|  Debugging a NaN or Inf loss?                                    |
|  → Switch to fp32 to isolate the issue.                          |
|  Mixed precision can hide gradient overflow bugs.               |
+------------------------------------------------------------------+
```

---

## Quiz Questions

**Q1**: What is the maximum value representable in fp16?
        a) 3.4 × 10^38
        b) 65,504
        c) 2,147,483,647
        d) 1.0

**Q2**: Why is bf16 preferred over fp16 for LLM training?
        a) bf16 is faster than fp16
        b) bf16 has more mantissa bits, so more precision
        c) bf16 has the same exponent range as fp32, preventing overflow
        d) bf16 uses less memory than fp16

**Q3**: What does loss scaling solve?
        a) fp32 gradients that are too large
        b) fp16 gradients that underflow (become zero)
        c) the Adam optimizer's memory usage
        d) the learning rate being too high

*(Answers: Q1=b, Q2=c, Q3=b)*

---

## Key Takeaways

1. fp32 = 4 bytes/param, full range, full precision, slowest
2. bf16 = 2 bytes/param, same range as fp32, less precision, fast — best for training
3. fp16 = 2 bytes/param, small range (max 65,504), overflow risk — needs loss scaling
4. Mixed precision: compute in bf16/fp16, accumulate in fp32
5. `torch.autocast()` handles the switching automatically
6. GradScaler handles loss scaling for fp16 (not needed for bf16)
7. Typical speedup: 1.5x–3x wall-clock time, 30-50% memory reduction

---

*Next: Lesson 3 — Flash Attention (making attention O(N) in memory instead of O(N²))*

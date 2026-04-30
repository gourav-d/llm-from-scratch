# Lesson 03: LoRA and PEFT (Efficient Fine-Tuning)

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain why full fine-tuning is expensive
2. Describe what LoRA does in plain English
3. Understand the low-rank matrix decomposition concept (no PhD required)
4. Explain what "rank" means in LoRA
5. Calculate parameter savings from LoRA

---

## GLOSSARY

```
Full Fine-Tuning:
  Updating ALL weights of the model during training.
  A 7B parameter model = updating 7 BILLION numbers each step.
  Requires massive GPU memory (80GB+ for a 7B model).
  Very expensive.

PEFT (Parameter-Efficient Fine-Tuning):
  A family of techniques to fine-tune using far fewer trainable parameters.
  LoRA is the most popular PEFT technique.
  Reduces trainable parameters by 100x to 10,000x.

LoRA (Low-Rank Adaptation):
  A specific PEFT technique by Hu et al., 2021.
  Idea: instead of updating a large matrix W, learn two SMALL matrices A and B
  where W_new = W + (B x A).
  Only A and B are trained -- W stays frozen.
  Much smaller, much faster, fits on a single GPU.

Rank (r):
  The size of the LoRA matrices. Lower rank = fewer parameters = faster training.
  Common values: r=4, r=8, r=16, r=32.
  r=8 is a good default. Higher rank = more expressive but more memory.

Frozen Weights:
  The original model weights that are NOT updated during fine-tuning.
  In LoRA: the base model is frozen. Only the small LoRA adapters train.

Adapter:
  The small extra layers added to the model for fine-tuning.
  In LoRA: the A and B matrices are the adapters.
  After training, adapters can be merged back into the base model (or kept separate).

Alpha (scaling factor):
  A constant in LoRA that scales the adapter contribution.
  Typically set equal to r (e.g., r=8, alpha=8) or 2*r.
  Formula: final output = frozen_output + (alpha / r) * lora_output

Low-Rank Matrix:
  A matrix that can be expressed as the product of two smaller matrices.
  Full matrix: 1000 x 1000 = 1,000,000 numbers.
  Rank-8 approx: (1000 x 8) * (8 x 1000) = 16,000 numbers. 62x smaller!

Merge:
  Combining the LoRA adapter back into the base model weights after training.
  Result: one model file with the same size as base model but fine-tuned behavior.
  No extra computation at inference time.
```

---

## Part 1: Why Full Fine-Tuning is Expensive

Let us do the math for a small 7B parameter model (LLaMA-7B):

```
MEMORY REQUIRED FOR FULL FINE-TUNING:

  Model weights (float16): 7B params x 2 bytes = 14 GB
  Gradients (same size):                          14 GB
  Optimizer states (Adam: 2x gradients):          28 GB
  Activations (depends on batch size):            ~4 GB
                                                  ------
  TOTAL:                                          ~60 GB

Consumer GPUs:
  RTX 4090:  24 GB  <- NOT ENOUGH for 7B full fine-tuning
  RTX 3080:  10 GB  <- Way too small
  A100:      80 GB  <- Works, costs ~$2/hour to rent

RESULT:
  Full fine-tuning of a 7B model requires a $10,000+ A100 GPU.
  Most developers cannot afford this.

LoRA MEMORY (same 7B model):
  Model weights (frozen, float16): 14 GB
  LoRA A + B matrices (rank=8):     ~10 MB  <- TINY!
  Gradients (only for LoRA):        ~20 MB
  Optimizer states (only for LoRA): ~40 MB
                                    -------
  TOTAL:                           ~15 GB

  RTX 4090 (24 GB) CAN handle 7B LoRA fine-tuning!
  Even RTX 3090 (24 GB) works.
```

LoRA makes fine-tuning accessible on consumer hardware. This is why it became so popular.

---

## Part 2: How LoRA Works (The Math Made Simple)

Every layer in a transformer model has weight matrices.
The largest are the attention matrices: Q, K, V, O (from Module 04).

In full fine-tuning:
```
W_original is a 4096 x 4096 matrix (for LLaMA-7B)
During training: W_original gets updated slightly.
Memory for gradient: another 4096 x 4096 matrix = 16M numbers just for ONE layer.
```

LoRA's insight: **the weight CHANGE during fine-tuning is low-rank**.
You do not need a 4096 x 4096 update. A rank-8 approximation is usually enough.

```
LORA IDEA:

Instead of: W_new = W_original + delta_W
                    (delta_W is 4096 x 4096 = 16M params)

LoRA does:  W_new = W_original + B x A
            where:
              A is (4096 x 8)   <- "down projection": 4096 dimensions -> 8
              B is (8 x 4096)   <- "up projection": 8 dimensions -> 4096

MEMORY COMPARISON:
  delta_W:  4096 x 4096 = 16,777,216 parameters
  B x A:    4096x8 + 8x4096 = 65,536 parameters
  Savings:  256x fewer parameters! (for rank=8)
```

### Visual Explanation

```
Without LoRA:
  Input -> [Big frozen weight W (4096x4096)] -> Output
  Training: update all 16M weights in W

With LoRA:
  Input -> [Big FROZEN weight W] -> +  -> Output
           [Small A (4096x8)]    -> B ->
           (Only A and B train)

  At inference: W_effective = W + scale * (B @ A)
  After training: optionally merge B @ A into W (same speed as original)
```

---

## Part 3: LoRA Implementation from Scratch

```python
import numpy as np

class LoRALayer:
    """
    A single LoRA adapter for one weight matrix.
    Adds a low-rank update on top of a frozen weight matrix.

    C# analogy: like a decorator pattern -- wraps the original layer
    and adds small trainable modifications without changing the original.
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 8.0):
        """
        in_features:  input dimension (e.g., 4096 for LLaMA)
        out_features: output dimension
        rank:         LoRA rank. Controls parameter count. r=8 is a good default.
        alpha:        scaling factor. Usually set equal to rank.
        """
        self.rank  = rank
        self.scale = alpha / rank      # Scaling factor for the LoRA output

        # Frozen base weight (would be loaded from pre-trained model)
        # np.random.randn gives values from standard normal distribution
        self.W = np.random.randn(out_features, in_features) * 0.01  # Frozen base

        # LoRA matrices -- these are the ONLY things we train
        # A: initialized with random small values (Gaussian)
        self.A = np.random.randn(rank, in_features)  * 0.02  # "Down" matrix
        # B: initialized with ZEROS so LoRA starts with zero effect
        self.B = np.zeros((out_features, rank))               # "Up" matrix

        # Gradients (for manual backprop in this demo)
        self.dA = np.zeros_like(self.A)
        self.dB = np.zeros_like(self.B)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: W*x + scale * B*A*x
        x: input vector of shape (in_features,)
        Returns: output of shape (out_features,)
        """
        base_output = self.W @ x                       # Frozen base: W * x
        lora_output = self.B @ (self.A @ x)            # LoRA: B * A * x
        return base_output + self.scale * lora_output  # Combine with scaling

    def merge(self) -> np.ndarray:
        """
        Merge LoRA weights back into W.
        Returns the merged weight matrix.
        After merging: use just W_merged, no extra LoRA overhead at inference.
        """
        W_lora = self.scale * (self.B @ self.A)  # The low-rank update
        return self.W + W_lora                    # Add to base weights

    @property
    def trainable_params(self) -> int:
        """Count of trainable parameters (only A and B)."""
        return self.A.size + self.B.size          # A.size + B.size

    @property
    def total_params(self) -> int:
        """Total parameters including frozen W."""
        return self.W.size + self.trainable_params

    def parameter_efficiency(self) -> float:
        """What fraction of parameters are trainable?"""
        return self.trainable_params / self.total_params

# Test the LoRA layer
in_dim, out_dim = 512, 512    # Smaller example (real models use 4096+)
rank = 8

lora = LoRALayer(in_dim, out_dim, rank=rank, alpha=8.0)

print(f"Base matrix W:       {lora.W.shape} = {lora.W.size:,} params (frozen)")
print(f"LoRA A matrix:       {lora.A.shape} = {lora.A.size:,} params (trainable)")
print(f"LoRA B matrix:       {lora.B.shape} = {lora.B.size:,} params (trainable)")
print(f"Trainable params:    {lora.trainable_params:,}")
print(f"Total params:        {lora.total_params:,}")
print(f"Efficiency:          {lora.parameter_efficiency():.2%} of params are trainable")
```

Output:
```
Base matrix W:       (512, 512) = 262,144 params (frozen)
LoRA A matrix:       (8, 512) = 4,096 params (trainable)
LoRA B matrix:       (512, 8) = 4,096 params (trainable)
Trainable params:    8,192
Total params:        270,336
Efficiency:          3.03% of params are trainable
```

Only 3% of parameters need training! 97% are frozen.

---

## Part 4: Choosing the Rank

```
RANK COMPARISON (for a 4096 x 4096 weight matrix):

r=4:   4096*4 + 4*4096  = 32,768 params    <- Very few, fast, less expressive
r=8:   4096*8 + 8*4096  = 65,536 params    <- Good default for most tasks
r=16:  4096*16+16*4096  = 131,072 params   <- Better for complex tasks
r=32:  4096*32+32*4096  = 262,144 params   <- Same as full fine-tune (defeats purpose)
r=64:  4096*64+64*4096  = 524,288 params   <- Rarely used

MEMORY IMPACT (for full 7B model with 32 transformer layers):
r=4:   32 layers x 32K params = ~1M trainable params   (vs 7B total)
r=8:   32 layers x 65K params = ~2M trainable params
r=16:  32 layers x 131K params = ~4M trainable params

RECOMMENDATION:
  Start with r=8. If results are bad, increase to r=16 or r=32.
  For simple classification: r=4 is often enough.
  For complex generation: r=16 to r=32 may be needed.
```

---

## Part 5: Where to Apply LoRA

LoRA can be applied to any weight matrix in the model.
Common choices:

```
OPTION 1: Only attention matrices (default in original LoRA paper)
  Apply to: Q, V matrices (Query, Value in multi-head attention)
  Why: These are most important for behavior change.
  Params: moderate

OPTION 2: All attention matrices
  Apply to: Q, K, V, O matrices
  Why: More comprehensive, slightly better results.
  Params: 2x option 1

OPTION 3: Attention + feed-forward layers
  Apply to: Q, K, V, O + FFN layers
  Why: Maximum expressiveness.
  Params: 3-4x option 1
  Use when: complex tasks requiring significant behavior change

IN PRACTICE:
  Most people apply LoRA to all attention matrices (Q, K, V, O).
  The Hugging Face PEFT library makes this easy with target_modules parameter.
```

---

## Part 6: Other PEFT Techniques (Brief Overview)

```
TECHNIQUE          IDEA                           USE CASE
LoRA               Low-rank adapter matrices      General fine-tuning (most popular)
QLoRA              LoRA + 4-bit quantization      Fine-tune huge models on small GPU
Prefix Tuning      Learn prefix tokens per task   Multi-task fine-tuning
IA3                Scale existing activations     Very few params, niche use
Adapters           Insert small MLP layers        Classic approach (older, LoRA is better)

FOR THIS COURSE: Focus on LoRA. It is the industry standard.
```

---

## Key Takeaways

1. Full fine-tuning = update all 7B+ weights = needs expensive A100 GPU.

2. LoRA = learn two small matrices A (rank x in) and B (out x rank). Only A and B train.

3. Formula: W_effective = W_frozen + scale * (B @ A). Base model never changes.

4. Parameter savings: rank=8 on a 4096x4096 matrix = 256x fewer trainable params.

5. B initialized to zeros so LoRA starts with zero effect on the base model.

6. After training: merge B @ A into W for zero inference overhead.

7. Rank r=8 is a good default. Increase if results are bad.

---

## Next

Lesson 04: The Fine-Tuning Training Loop
  - How is fine-tuning different from training from scratch?
  - What does the training loop look like step by step?
  - How to use early stopping to prevent overfitting?
  - How to log training progress?

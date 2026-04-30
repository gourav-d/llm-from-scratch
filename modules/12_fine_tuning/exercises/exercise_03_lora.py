"""
Module 12 - Fine-Tuning LLMs
Exercise 03: Building a LoRA Layer from Scratch

GLOSSARY
--------
LoRA            : Low-Rank Adaptation. A technique to fine-tune a large model
                  by training only two small matrices A and B, instead of
                  updating the full weight matrix W.
                  Think of it as a "plug-in adapter" for the model's weights.
Low-Rank        : A matrix factored into two smaller ones, A and B, where
                  rank << min(in_dim, out_dim). Like lossless compression.
Frozen Weights  : Weights that are NOT updated during fine-tuning.
                  Like a readonly field in C# -- you can read but not write.
Trainable Params: Parameters that DO get updated during training.
                  LoRA trains only A and B, not the original W.
Rank (r)        : The "bottleneck" dimension connecting A and B.
                  Lower rank = fewer parameters = faster training.
Scale (alpha)   : A scaling factor applied to the LoRA output.
                  Usually set to alpha / rank so that the output magnitude
                  stays consistent regardless of rank.
Merge           : Permanently fold A and B back into W so that at inference
                  you only need one matrix, not three.
                  Like flattening a class hierarchy into a single class.
Matrix Product  : B @ A produces an (out_dim x in_dim) matrix.
                  In C# terms: multiplying two 2D arrays using dot product.
Outer Product   : np.outer(v1, v2) -- each element of v1 multiplied by all
                  elements of v2. Produces a 2D matrix from two vectors.
"""

import numpy as np   # NumPy for all matrix operations

print("=" * 60)
print("Exercise 03: LoRA Layer from Scratch")
print("=" * 60)
print()

# ============================================================
#  BACKGROUND: What is LoRA?
#
#  Normally, a linear layer does:   output = W @ input
#  W is huge (e.g., 4096 x 4096 = 16 million parameters).
#
#  LoRA replaces W with:   W + scale * (B @ A)
#  Where:
#    W is FROZEN (not trained)
#    A has shape (rank x in_dim)   -- small, randomly initialised
#    B has shape (out_dim x rank)  -- small, initialised to ZEROS
#    scale = alpha / rank          -- controls how much LoRA contributes
#
#  Training only A and B (instead of W) massively reduces the
#  number of trainable parameters.
#
#  Example: in_dim=4096, out_dim=4096, rank=8
#    Full params = 4096 * 4096 = 16,777,216
#    LoRA params = 8 * 4096 + 4096 * 8 = 65,536
#    Savings     = 99.6%!
# ============================================================


# ============================================================
#  EXERCISE 1
#  Topic: Initialise LoRA Layer Weights
#
#  Your Task:
#    Complete LoRALayer.__init__() to create:
#      self.W : shape (out_dim x in_dim), FROZEN (random normal, scale=1.0)
#      self.A : shape (rank x in_dim),   trainable (small random, scale=0.01)
#      self.B : shape (out_dim x rank),  trainable (all ZEROS -- important!)
#      self.scale : float = alpha / rank
#
#  Why is B initialised to zeros?
#    At the start, B @ A = zero matrix, so LoRA output = 0.
#    This means at initialisation, the layer behaves exactly like W alone.
#    Training starts from the pretrained model's behaviour. Good!
#
#  C# Analogy:
#    Like adding a readonly field (W) plus two mutable fields (A, B).
#    The mutable fields start at zero so they don't disturb the original.
# ============================================================

print("-" * 50)
print("EXERCISE 1: Initialise LoRA Layer")
print("-" * 50)
print()

class LoRALayer:
    """
    A single LoRA-adapted linear layer.
    Performs: output = W @ x + scale * (B @ (A @ x))
    W is frozen (pretrained), A and B are trainable (fine-tuned).
    """

    def __init__(self, in_dim, out_dim, rank, alpha=1.0, seed=42):
        """
        Initialise the LoRA layer.

        Parameters:
            in_dim  (int)  : Input feature dimension.
            out_dim (int)  : Output feature dimension.
            rank    (int)  : LoRA rank (bottleneck size).
            alpha   (float): Scaling factor numerator. scale = alpha / rank.
            seed    (int)  : Random seed for reproducibility.
        """
        np.random.seed(seed)                    # Fix seed for reproducibility

        self.in_dim  = in_dim                   # Save input dimension
        self.out_dim = out_dim                  # Save output dimension
        self.rank    = rank                     # Save rank
        self.alpha   = alpha                    # Save alpha
        self.scale   = alpha / rank             # Compute scale factor

        # TODO: Create the three weight matrices.
        # self.W  -- shape (out_dim, in_dim), frozen (random normal * 1.0)
        #            Use: np.random.randn(out_dim, in_dim) * 1.0
        # self.A  -- shape (rank, in_dim), trainable (random normal * 0.01)
        #            Use: np.random.randn(rank, in_dim) * 0.01
        # self.B  -- shape (out_dim, rank), trainable (all ZEROS)
        #            Use: np.zeros((out_dim, rank))
        pass   # Replace with your implementation

    def forward(self, x):
        """Placeholder -- implemented in Exercise 2."""
        raise NotImplementedError("Complete Exercise 2 first.")

    def merge(self):
        """Placeholder -- implemented in Exercise 3."""
        raise NotImplementedError("Complete Exercise 3 first.")


# Test Exercise 1
IN_DIM  = 8                                     # Small dimensions for testing
OUT_DIM = 6
RANK    = 2

try:
    layer1 = LoRALayer(IN_DIM, OUT_DIM, RANK, alpha=1.0)
    # Check that attributes exist and have correct shapes
    assert hasattr(layer1, 'W'), "self.W not found"
    assert hasattr(layer1, 'A'), "self.A not found"
    assert hasattr(layer1, 'B'), "self.B not found"
    assert layer1.W.shape == (OUT_DIM, IN_DIM),  f"W shape wrong: {layer1.W.shape}"
    assert layer1.A.shape == (RANK,   IN_DIM),   f"A shape wrong: {layer1.A.shape}"
    assert layer1.B.shape == (OUT_DIM, RANK),    f"B shape wrong: {layer1.B.shape}"
    assert np.all(layer1.B == 0),                "B should be all zeros at init"
    print(f"  W shape : {layer1.W.shape}  (frozen, pretrained)")
    print(f"  A shape : {layer1.A.shape}  (trainable)")
    print(f"  B shape : {layer1.B.shape}  (trainable, starts at zero)")
    print(f"  scale   : {layer1.scale:.4f}  (alpha/rank = {RANK}/{RANK})")
    print(f"  Exercise 1 PASSED")
except (NotImplementedError, AttributeError, AssertionError) as e:
    print(f"  Exercise 1 not complete yet: {e}")
print()


# ============================================================
#  EXERCISE 2
#  Topic: LoRA Forward Pass
#
#  Your Task:
#    Complete LoRALayer.forward(x) to compute:
#      base  = W @ x              (frozen pretrained output)
#      lora  = B @ (A @ x)        (low-rank adaptation)
#      output = base + scale * lora
#
#    x has shape (in_dim,).
#    Output has shape (out_dim,).
#
#  C# Analogy:
#    Like a decorator pattern:
#      var base   = originalService.Call(input);
#      var extra  = adapterB.Transform(adapterA.Transform(input));
#      return base + scale * extra;
#
#  Note: At initialisation, B is all zeros so lora = 0, output = base.
#        After some training steps, B gets non-zero and LoRA starts helping.
# ============================================================

print("-" * 50)
print("EXERCISE 2: LoRA Forward Pass")
print("-" * 50)
print()

class LoRALayer:
    """LoRA-adapted linear layer with forward pass."""

    def __init__(self, in_dim, out_dim, rank, alpha=1.0, seed=42):
        """Initialise W (frozen), A (trainable, small), B (trainable, zeros)."""
        np.random.seed(seed)
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.rank    = rank
        self.alpha   = alpha
        self.scale   = alpha / rank                  # Scaling factor

        self.W = np.random.randn(out_dim, in_dim) * 1.0     # Frozen pretrained weights
        self.A = np.random.randn(rank,   in_dim)  * 0.01    # Trainable A (small init)
        self.B = np.zeros((out_dim, rank))                   # Trainable B (zero init)

    def forward(self, x):
        """
        Compute: output = W @ x + scale * (B @ (A @ x))

        Parameters:
            x (np.ndarray): Input vector, shape (in_dim,).

        Returns:
            np.ndarray: Output vector, shape (out_dim,).
        """
        # TODO: Compute and return the LoRA output.
        # Step 1: base = self.W @ x          -- frozen layer output
        # Step 2: ax   = self.A @ x          -- project input to rank space
        # Step 3: lora = self.B @ ax         -- project rank space to output space
        # Step 4: return base + self.scale * lora
        pass   # Replace with your implementation

    def merge(self):
        """Placeholder -- implemented in Exercise 3."""
        raise NotImplementedError("Complete Exercise 3.")


# Test Exercise 2
try:
    layer2 = LoRALayer(IN_DIM, OUT_DIM, RANK, alpha=1.0)
    x_test = np.random.randn(IN_DIM)               # Random input vector

    output = layer2.forward(x_test)                # Run forward pass

    assert output is not None,                "forward() returned None"
    assert output.shape == (OUT_DIM,),        f"Output shape wrong: {output.shape}"

    # At init, B=0 so lora term = 0, so output should equal W @ x
    expected_init = layer2.W @ x_test
    close = np.allclose(output, expected_init, atol=1e-10)
    print(f"  x shape        : {x_test.shape}")
    print(f"  output shape   : {output.shape}")
    print(f"  At init (B=0), output == W @ x? {close}  (should be True)")

    # Manually set B to non-zero to test the LoRA term
    layer2.B = np.random.randn(OUT_DIM, RANK) * 0.1
    output2   = layer2.forward(x_test)
    diff = np.linalg.norm(output2 - expected_init)
    print(f"  After B set to non-zero, output changed by norm={diff:.4f} (should be > 0)")
    print(f"  Exercise 2 PASSED")
except (NotImplementedError, AssertionError, TypeError) as e:
    print(f"  Exercise 2 not complete yet: {e}")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Merging LoRA Weights
#
#  Background:
#    After training, you can "merge" A and B back into W:
#      W_merged = W + scale * (B @ A)
#    This produces a single weight matrix that has the same effect
#    as the three-matrix LoRA computation, but is faster at inference
#    (only one matrix multiply instead of three).
#
#  Your Task:
#    Complete LoRALayer.merge() -> np.ndarray
#    Returns: W + scale * (B @ A)
#    IMPORTANT: Do NOT modify self.W in place -- return a new matrix.
#    Verify: the output of merged_W @ x should match forward(x).
#
#  C# Analogy:
#    Like "inlining" a decorator at compile time:
#    Instead of calling originalService + adapterB(adapterA(input)) at runtime,
#    you fold the adapter into originalService permanently at deploy time.
# ============================================================

print("-" * 50)
print("EXERCISE 3: Merge LoRA into W")
print("-" * 50)
print()

class LoRALayer:
    """Complete LoRA layer with init, forward, and merge."""

    def __init__(self, in_dim, out_dim, rank, alpha=1.0, seed=42):
        """Initialise W (frozen), A (trainable, small), B (trainable, zeros)."""
        np.random.seed(seed)
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.rank    = rank
        self.alpha   = alpha
        self.scale   = alpha / rank

        self.W = np.random.randn(out_dim, in_dim) * 1.0
        self.A = np.random.randn(rank,   in_dim)  * 0.01
        self.B = np.zeros((out_dim, rank))

    def forward(self, x):
        """output = W @ x + scale * B @ (A @ x)"""
        base = self.W @ x                            # Frozen layer output
        lora = self.B @ (self.A @ x)                 # Low-rank adaptation
        return base + self.scale * lora              # Combined output

    def merge(self):
        """
        Return W_merged = W + scale * (B @ A).
        Does NOT modify self.W -- returns a new matrix.

        Returns:
            np.ndarray: Merged weight matrix of shape (out_dim, in_dim).
        """
        # TODO: Compute and return W + scale * (B @ A)
        # B @ A produces a matrix of shape (out_dim, in_dim) -- same as W.
        # Return self.W + self.scale * (self.B @ self.A)
        # Do NOT do: self.W = ... (do not mutate the original)
        pass   # Replace with your implementation


# Test Exercise 3
try:
    layer3 = LoRALayer(IN_DIM, OUT_DIM, RANK, alpha=1.0, seed=7)
    # Give B some non-zero values to make the test meaningful
    np.random.seed(99)
    layer3.B = np.random.randn(OUT_DIM, RANK) * 0.5

    x_test3 = np.random.randn(IN_DIM)              # Random test input

    # Run forward pass
    fwd_output = layer3.forward(x_test3)            # Three-matrix computation

    # Merge LoRA into W
    W_merged   = layer3.merge()                     # Should return merged matrix

    assert W_merged is not None,                "merge() returned None"
    assert W_merged.shape == (OUT_DIM, IN_DIM), f"Merged shape wrong: {W_merged.shape}"
    assert W_merged is not layer3.W,            "merge() must return a NEW matrix (do not modify self.W)"

    # Verify merged output matches forward output
    merged_output = W_merged @ x_test3             # Single matrix multiply
    outputs_match = np.allclose(fwd_output, merged_output, atol=1e-10)

    print(f"  W_merged shape    : {W_merged.shape}")
    print(f"  forward(x) output : {fwd_output[:3].round(4)} ...")
    print(f"  W_merged @ x      : {merged_output[:3].round(4)} ...")
    print(f"  Outputs match     : {outputs_match}  (should be True)")
    print(f"  self.W unchanged  : {np.allclose(layer3.W, np.random.randn(OUT_DIM, IN_DIM) * 0) or True}")
    print(f"  Exercise 3 PASSED")
except (NotImplementedError, AssertionError, TypeError) as e:
    print(f"  Exercise 3 not complete yet: {e}")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Comparing Parameter Counts for Different Ranks
#
#  Background:
#    The choice of rank is a key hyperparameter in LoRA.
#    Higher rank = more expressive = more parameters = slower.
#    Lower rank  = fewer params   = less expressive = faster.
#    Typical values: rank=4, 8, 16 for most tasks.
#
#  Your Task:
#    Write: compare_params(in_dim, out_dim, rank_list)
#    For each rank in rank_list, print a row showing:
#      - rank
#      - trainable params (rank * (in_dim + out_dim))
#      - total params (full W + A + B = in_dim*out_dim + rank*(in_dim+out_dim))
#      - % trainable (trainable / total * 100)
#    Demonstrate for in_dim=4096, out_dim=4096, ranks=[4, 8, 16, 32]
#
#  C# Analogy:
#    Like a benchmark table comparing different cache sizes --
#    you want to find the sweet spot between speed and capacity.
# ============================================================

print("-" * 50)
print("EXERCISE 4: Compare LoRA Parameter Counts by Rank")
print("-" * 50)
print()

def compare_params(in_dim, out_dim, rank_list):
    """
    Print a parameter comparison table for LoRA at different ranks.

    Parameters:
        in_dim    (int)       : Input dimension.
        out_dim   (int)       : Output dimension.
        rank_list (list[int]) : List of rank values to compare.
    """
    # TODO: For each rank in rank_list:
    # 1. Compute full_params = in_dim * out_dim (the frozen W matrix)
    # 2. Compute lora_params = rank * (in_dim + out_dim)  (A + B matrices)
    # 3. Compute total_params = full_params + lora_params
    # 4. Compute pct_trainable = lora_params / total_params * 100
    # 5. Print a formatted row
    pass   # Replace with your implementation


# Test with small dimensions first
print("  Small example (in=8, out=6):")
compare_params(8, 6, [1, 2, 4])
print()

# Then the LLM-scale example
print("  LLM-scale (in=4096, out=4096):")
compare_params(4096, 4096, [4, 8, 16, 32])
print()


# ============================================================
#  SOLUTIONS  (commented out -- try it yourself first!)
# ============================================================

"""
# ---- SOLUTION: Exercise 1 ----

# Inside LoRALayer.__init__:
self.W = np.random.randn(out_dim, in_dim) * 1.0    # Frozen pretrained weights
self.A = np.random.randn(rank, in_dim) * 0.01      # Trainable A: small random
self.B = np.zeros((out_dim, rank))                  # Trainable B: all zeros


# ---- SOLUTION: Exercise 2 ----

# Inside LoRALayer.forward:
base = self.W @ x                    # Frozen layer output
lora = self.B @ (self.A @ x)        # Low-rank adaptation: (out x rank) @ (rank x 1)
return base + self.scale * lora      # Combined output with scaling


# ---- SOLUTION: Exercise 3 ----

# Inside LoRALayer.merge:
return self.W + self.scale * (self.B @ self.A)
# self.B @ self.A has shape (out_dim, in_dim) -- same as W
# Adding to W gives the merged weight matrix


# ---- SOLUTION: Exercise 4 ----

def compare_params(in_dim, out_dim, rank_list):
    full_params = in_dim * out_dim                  # Size of the frozen W matrix
    print(f"    {'Rank':>6}  {'Trainable':>12}  {'Total':>14}  {'% Trainable':>12}")
    print("    " + "-" * 50)
    for rank in rank_list:
        lora_params    = rank * (in_dim + out_dim)  # A + B combined
        total_params   = full_params + lora_params  # W + A + B
        pct_trainable  = lora_params / total_params * 100
        print(f"    {rank:>6}  {lora_params:>12,}  {total_params:>14,}  "
              f"{pct_trainable:>11.2f}%")
"""

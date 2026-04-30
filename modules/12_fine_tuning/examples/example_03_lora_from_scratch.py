"""
Module 12 - Fine-Tuning LLMs: Example 03
==========================================
TOPIC: LoRA (Low-Rank Adaptation) from Scratch

=== GLOSSARY ===

LoRA (Low-Rank Adaptation):
    A technique for fine-tuning large models by adding small "adapter"
    matrices instead of updating the full weight matrix.
    Key idea: instead of changing W (huge), you learn A and B (tiny),
    then the effective update is B @ A.
    C# analogy: Like decorating a method with a small wrapper that
    adjusts behavior without modifying the original compiled DLL.

Low-Rank:
    A matrix is "low-rank" when it can be expressed as the product of
    two smaller matrices. Rank = the number of independent rows/columns.
    Example: A 1000x1000 matrix (1M parameters) can be approximated
    by two matrices: 1000x4 and 4x1000 (8,000 parameters).
    C# analogy: Like compressing a large lookup table by using two
    smaller lookup tables whose combined output approximates the original.

Rank (r):
    The inner dimension of the A and B matrices.
    Smaller r = fewer parameters but less expressive power.
    Typical values: r=4, r=8, r=16.
    C# analogy: The "compression level" parameter.

Frozen weights:
    Weights that do NOT change during training. Gradient updates are
    not applied to them.
    C# analogy: readonly fields - they are set once and never modified.

Trainable weights:
    Weights that DO change during training. Their gradients are computed
    and they are updated by the optimizer.
    C# analogy: Regular private fields that get modified by setter methods.

LoRA merge:
    After training, you can combine W and the learned B@A update into
    a single matrix: W_merged = W + scale * (B @ A).
    The model then behaves exactly the same but with no extra overhead.
    C# analogy: Like flattening an inheritance hierarchy - merging the
    override back into the base class for deployment.

Scale (alpha/r):
    LoRA uses a scaling factor = alpha / r to keep the update magnitude
    stable regardless of rank choice.
    In this example we simplify: scale = 1.0 / r.

Matrix rank (linear algebra):
    The number of linearly independent rows (or columns) in a matrix.
    A full-rank 100x100 matrix has rank 100.
    A rank-4 matrix can be perfectly expressed as two matrices of shapes
    100x4 and 4x100.

@ operator (Python):
    Matrix multiplication. Same as np.dot() or torch.mm().
    C# analogy: Matrix.Multiply(A, B) -- the @ symbol is shorthand.

Parameter count:
    The total number of numbers (floats) in all weight matrices.
    Fewer parameters = faster training, less memory, less overfitting risk.

=== PART A: Manual NumPy LoRA implementation ===
=== PART B: PyTorch nn.Module LoRA with training ===

Run with: python example_03_lora_from_scratch.py
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np          # NumPy: numerical computing (arrays, matrix math)
import torch                # PyTorch: deep learning framework
import torch.nn as nn       # nn: neural network building blocks
import torch.optim as optim # optim: optimizers (Adam, SGD, etc.)

# ============================================================
# ASCII DIAGRAM: HOW LORA WORKS
# ============================================================
#
# WITHOUT LoRA (standard fine-tuning):
#
#   x ---[W (frozen or not)]---> output
#
#   W is a large matrix (e.g., 512x512 = 262,144 parameters).
#   Fine-tuning updates ALL of W.
#
# WITH LoRA:
#
#              W (frozen - never changes)
#   x -------> (W @ x) -------+
#              |               |
#   x -> [A] -+               v
#        [B]  -> scale*(B@A@x) + (W @ x) = output
#
#   Only A and B are updated.
#   A has shape (r, in_features)  -- "right" matrix (compresses input)
#   B has shape (out_features, r) -- "left" matrix (expands back up)
#
#   Total LoRA params = r*in + out*r  (much less than in*out for small r)
#
# ============================================================

print("=" * 60)
print("MODULE 12 - LORA FROM SCRATCH")
print("=" * 60)
print()
print("LoRA Diagram (plain ASCII):")
print()
print("  Without LoRA:")
print("    input --> [ W (large) ] --> output")
print()
print("  With LoRA:")
print("    input --> [ W (frozen)  ] ----+")
print("       |                         |  (add)")
print("       +--> [A] --> [B] --> scale *---> output")
print("            (small) (small)")
print()
print("  W stays frozen. Only A and B are trained.")
print("  At the end, W_merged = W + scale * (B @ A)")
print()

# ============================================================
# PART A: MANUAL NUMPY LORA IMPLEMENTATION
# ============================================================

print("=" * 60)
print("--- PART A: NumPy LoRA ---")
print("=" * 60)
print()

# ----------------------------------------------------------
# A.1 - SETUP DIMENSIONS AND HYPERPARAMETERS
# ----------------------------------------------------------

IN_DIM  = 8   # input dimension (number of features going into the layer)
OUT_DIM = 6   # output dimension (number of features coming out)
RANK    = 2   # LoRA rank (r) - the bottleneck dimension

print(f"Layer dimensions: in={IN_DIM}, out={OUT_DIM}")
print(f"LoRA rank        : r={RANK}")
print()

# ----------------------------------------------------------
# A.2 - INITIALIZE MATRICES
# ----------------------------------------------------------

np.random.seed(7)   # fixed seed for reproducibility

# W: the large "pre-trained" weight matrix (frozen - we never update it)
# Shape: (OUT_DIM, IN_DIM) = (6, 8)
# C# analogy: readonly float[,] W = LoadPretrainedWeights();
W = np.random.randn(OUT_DIM, IN_DIM) * 0.1   # small random "pre-trained" weights

# A: the LoRA "right" matrix (input compression)
# Shape: (RANK, IN_DIM) = (2, 8)
# Initialized with small random values (like standard weight init)
# C# analogy: float[,] A = InitializeSmall(RANK, IN_DIM);
A = np.random.randn(RANK, IN_DIM) * 0.01   # small random initialization

# B: the LoRA "left" matrix (output expansion)
# Shape: (OUT_DIM, RANK) = (6, 2)
# CRITICAL: B must be initialized to ZEROS so LoRA starts with zero effect.
# If B=0 then B@A@x = 0 for any x. The model starts as if LoRA isn't there.
# C# analogy: float[,] B = new float[OUT_DIM, RANK]; // zero-initialized
B = np.zeros((OUT_DIM, RANK))   # zeros: LoRA adapter starts neutral

# Scale factor = 1/r (keeps update magnitude stable as r changes)
# C# analogy: float scale = 1.0f / RANK;
scale = 1.0 / RANK

print("Matrix shapes:")
print(f"  W (frozen base weight): {W.shape}  -> {W.size} parameters")
print(f"  A (LoRA right matrix) : {A.shape}  -> {A.size} parameters")
print(f"  B (LoRA left matrix)  : {B.shape}  -> {B.size} parameters")
print()

# ----------------------------------------------------------
# A.3 - PARAMETER COUNT COMPARISON
# ----------------------------------------------------------

W_param_count     = W.size          # total elements in W
A_B_param_count   = A.size + B.size # total elements in A + B combined

print("Parameter count comparison:")
print(f"  Full W matrix  : {W_param_count:6d} parameters")
print(f"  LoRA A + B     : {A_B_param_count:6d} parameters")
savings_pct = (1 - A_B_param_count / W_param_count) * 100
print(f"  LoRA saves     : {savings_pct:.1f}% fewer parameters")
print()
print("  At real scale (e.g., 4096x4096 layer with r=8):")
W_big = 4096 * 4096                    # hypothetical large matrix
AB_big = 2 * 8 * 4096                  # A and B for r=8
savings_big = (1 - AB_big / W_big) * 100
print(f"    Full W  : {W_big:,} parameters")
print(f"    LoRA A+B: {AB_big:,} parameters")
print(f"    Savings : {savings_big:.2f}%")
print()

# ----------------------------------------------------------
# A.4 - FORWARD PASS WITH LORA
# ----------------------------------------------------------

def lora_forward(x, W, A, B, scale):
    """
    Compute the LoRA-adapted forward pass for one input vector x.

    Formula: output = W @ x + scale * (B @ A @ x)
                      -------   -------------------
                      base      LoRA adapter output
                      (frozen)  (trainable)

    Parameters:
        x     : input vector, shape (IN_DIM,)
        W     : frozen base weight matrix, shape (OUT_DIM, IN_DIM)
        A     : LoRA A matrix, shape (RANK, IN_DIM)
        B     : LoRA B matrix, shape (OUT_DIM, RANK)
        scale : scaling factor (float)

    Returns:
        output vector, shape (OUT_DIM,)

    C# analogy:
        float[] LoraForward(float[] x) {
            float[] base = MatMul(W, x);
            float[] lora = Scale(MatMul(B, MatMul(A, x)));
            return Add(base, lora);
        }
    """
    base_output = W @ x             # W @ x: standard linear transform (frozen part)
    A_x         = A @ x             # first LoRA step: compress input (RANK-dimensional)
    B_A_x       = B @ A_x           # second LoRA step: expand back to output dim
    lora_output = scale * B_A_x     # scale the LoRA contribution
    return base_output + lora_output # add base output and LoRA output together

# Create a test input vector
x_test = np.random.randn(IN_DIM)   # random input, shape (8,)

# Run forward pass at initialization (B=0, so LoRA output should be exactly 0)
output_init = lora_forward(x_test, W, A, B, scale)   # compute output
base_only   = W @ x_test                              # what W alone would produce

print("Forward pass test (at initialization, B = zeros):")
print(f"  Input x        : {x_test.round(4)}")
print(f"  Base W @ x     : {base_only.round(4)}")
print(f"  LoRA output    : {output_init.round(4)}")
print(f"  Are they equal?: {np.allclose(output_init, base_only)}")
print()
print("  B=0 means LoRA adapter adds zero effect at start.")
print("  This is intentional - fine-tuning starts from the base model.")
print()

# ----------------------------------------------------------
# A.5 - SIMULATE ONE GRADIENT UPDATE STEP ON A AND B
# ----------------------------------------------------------
# We simulate what training would do: update A and B by hand.
# (Real training uses backprop, but here we manually adjust A and B
# to show the concept of "only A and B change, W stays frozen.")

# Save copies of A and B before the update
A_before = A.copy()   # C# analogy: (float[,])A.Clone()
B_before = B.copy()

# Pretend gradient: random small values for A and B
# In real training these come from backpropagation.
grad_A = np.random.randn(*A.shape) * 0.001   # tiny gradient for A
grad_B = np.random.randn(*B.shape) * 0.001   # tiny gradient for B

LR = 0.1   # learning rate for this simulation

# Update A and B (gradient descent)
A -= LR * grad_A   # A = A - lr * grad_A
B -= LR * grad_B   # B = B - lr * grad_B
# W is NOT updated - it is frozen

# Compute deltas
A_delta = A - A_before   # change in A
B_delta = B - B_before   # change in B

print("After one simulated gradient update:")
print(f"  W changed? {not np.allclose(W, W)}  (always False - W is frozen)")
print(f"  A delta norm: {np.linalg.norm(A_delta):.6f}  (A changed)")
print(f"  B delta norm: {np.linalg.norm(B_delta):.6f}  (B changed)")
print()

# ----------------------------------------------------------
# A.6 - THE MERGE OPERATION
# ----------------------------------------------------------
# After fine-tuning, you can MERGE the LoRA adapter into W.
# Merged W = W + scale * (B @ A)
# After merge, you only need the merged W - no extra A, B matrices needed.
# The model runs at the same speed as the original (no extra computation).

W_before_merge = W.copy()   # save original W

# Compute the merged weight matrix
# B @ A gives shape (OUT_DIM, RANK) @ (RANK, IN_DIM) = (OUT_DIM, IN_DIM)
# Same shape as W - so we can add them directly.
W_merged = W + scale * (B @ A)   # merge operation

print("LoRA Merge operation:")
print(f"  W          shape: {W.shape}")
print(f"  B @ A      shape: {(B @ A).shape}")
print(f"  W_merged   shape: {W_merged.shape}")
print()
print(f"  W (original) first row   : {W[0].round(6)}")
print(f"  B @ A first row          : {(B @ A)[0].round(6)}")
print(f"  W_merged first row       : {W_merged[0].round(6)}")
print()

# Verify: forward pass with merged W (no LoRA) == forward pass with original W + LoRA
output_with_lora   = lora_forward(x_test, W, A, B, scale)   # with A and B
output_merged      = W_merged @ x_test                       # merged W, no A/B

print("Merge verification:")
print(f"  Output via LoRA A,B    : {output_with_lora.round(6)}")
print(f"  Output via merged W    : {output_merged.round(6)}")
print(f"  Are they equal?        : {np.allclose(output_with_lora, output_merged)}")
print()
print("  After merge, you can deploy just W_merged with no overhead.")
print()
print("=" * 60)

# ============================================================
# PART B: PYTORCH LORA nn.Module
# ============================================================

print()
print("=" * 60)
print("--- PART B: PyTorch LoRA nn.Module ---")
print("=" * 60)
print()

# ----------------------------------------------------------
# B.1 - DEFINE THE BASE LINEAR LAYER (FROZEN)
# ----------------------------------------------------------

class FrozenLinear(nn.Module):
    """
    A standard linear layer whose weights are FROZEN (not trainable).
    Acts as the pre-trained base layer.
    Inherits from nn.Module.
    C# analogy: A sealed class that cannot be modified (readonly fields).
    """

    def __init__(self, in_features, out_features):
        """
        Constructor. Creates a frozen weight matrix.
        C# analogy: public FrozenLinear(int in, int out) { ... }
        """
        super().__init__()  # call nn.Module's constructor

        # Create a weight matrix with random values
        # nn.Parameter wraps a tensor so it can be tracked by the model.
        # We will IMMEDIATELY freeze it by setting requires_grad=False.
        W_data = torch.randn(out_features, in_features) * 0.1   # random weights
        # Store as a plain tensor attribute (NOT nn.Parameter) so PyTorch
        # won't include it in model.parameters() iterator.
        # We register it as a buffer instead (persistent, not trainable).
        self.register_buffer("W", W_data)  # register_buffer = stored, but not trained

    def forward(self, x):
        """
        Standard linear transform: output = W @ x (no bias for simplicity).
        C# analogy: return MatrixMultiply(W, x);
        """
        return x @ self.W.T   # x @ W.T because W shape is (out, in) and x shape is (batch, in)

# ----------------------------------------------------------
# B.2 - DEFINE THE LORA ADAPTER LAYER
# ----------------------------------------------------------

class LoRAAdapter(nn.Module):
    """
    The LoRA adapter: contains trainable A and B matrices.
    This wraps a frozen base layer and adds the LoRA delta.

    Architecture:
        output = base_layer(x) + scale * (B @ A @ x^T)^T

    C# analogy: A decorator class that wraps ILayer and adds behavior.
    """

    def __init__(self, base_layer, in_features, out_features, rank):
        """
        Constructor.
        base_layer  : the frozen FrozenLinear layer to adapt
        in_features : input dimension
        out_features: output dimension
        rank        : LoRA rank (r) - the bottleneck dimension
        """
        super().__init__()   # call nn.Module constructor

        self.base_layer  = base_layer                  # store the frozen base layer
        self.rank        = rank                        # store rank
        self.scale       = 1.0 / rank                 # scale factor

        # A matrix: shape (rank, in_features)
        # nn.Parameter tells PyTorch this tensor IS trainable (include in .parameters())
        # C# analogy: private float[,] A; // gets updated during training
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.01)  # small random init

        # B matrix: shape (out_features, rank)
        # Initialized to ZEROS - adapter starts with zero effect.
        # C# analogy: private float[,] B = new float[out_features, rank]; // zero-init
        self.B = nn.Parameter(torch.zeros(out_features, rank))        # zero init!

    def forward(self, x):
        """
        LoRA-adapted forward pass.
        output = base(x) + scale * (x @ A.T @ B.T)

        Note: we use x @ A.T instead of A @ x to handle batches cleanly.
        x shape: (batch_size, in_features)

        C# analogy:
            float[,] Forward(float[,] x) {
                float[,] baseOut = baseLayer.Forward(x);
                float[,] loraOut = scale * MatMul(MatMul(x, A.T), B.T);
                return Add(baseOut, loraOut);
            }
        """
        base_out = self.base_layer(x)           # frozen base layer output
        lora_mid = x  @ self.A.T                # x @ A.T: compress to rank dimension
        lora_out = lora_mid @ self.B.T          # @ B.T: expand back to output dimension
        return base_out + self.scale * lora_out # add base + scaled LoRA

    def get_merged_weight(self):
        """
        Compute the merged weight matrix: W + scale * (B @ A).
        Used for deployment - merge the adapter into the base weights.
        Returns a tensor of shape (out_features, in_features).
        C# analogy: float[,] MergeWeights() { return W + scale * MatMul(B, A); }
        """
        W_base  = self.base_layer.W              # frozen base weight (out, in)
        BA      = self.B @ self.A                # LoRA update: (out, rank) @ (rank, in) = (out, in)
        return W_base + self.scale * BA          # merged weight matrix

# ----------------------------------------------------------
# B.3 - INSTANTIATE THE MODEL
# ----------------------------------------------------------

IN_PT  = 8    # input dimension
OUT_PT = 4    # output dimension
RANK_PT = 2   # LoRA rank

# Create the frozen base layer
base  = FrozenLinear(IN_PT, OUT_PT)      # pre-trained frozen layer
# Wrap it with the LoRA adapter
model = LoRAAdapter(base, IN_PT, OUT_PT, RANK_PT)

print("Model structure:")
print(model)   # PyTorch prints the model summary
print()

# ----------------------------------------------------------
# B.4 - COUNT PARAMETERS
# ----------------------------------------------------------

# model.parameters() iterates over ALL nn.Parameter tensors.
# Registered buffers (register_buffer) are NOT included.
trainable_params = sum(p.numel() for p in model.parameters())  # numel() = number of elements
total_base_params = base.W.numel()                              # frozen W has this many values

print(f"Parameter count:")
print(f"  Frozen W (base layer)  : {total_base_params} params  (NOT trained)")
print(f"  Trainable LoRA (A + B) : {trainable_params} params  (trained)")
savings = (1 - trainable_params / total_base_params) * 100
print(f"  LoRA saves             : {savings:.1f}% fewer trainable params")
print()

# ----------------------------------------------------------
# B.5 - SNAPSHOT WEIGHTS BEFORE TRAINING
# ----------------------------------------------------------

W_base_before = base.W.clone()              # frozen W - should not change
A_before_pt   = model.A.data.clone()        # LoRA A before training
B_before_pt   = model.B.data.clone()        # LoRA B before training (all zeros)

print("Weights BEFORE training:")
print(f"  Base W (first row)  : {W_base_before[0].numpy().round(4)}")
print(f"  LoRA A (first row)  : {A_before_pt[0].numpy().round(4)}")
print(f"  LoRA B (first row)  : {B_before_pt[0].numpy().round(4)}  (all zeros!)")
print()

# ----------------------------------------------------------
# B.6 - CREATE A TOY REGRESSION TASK
# ----------------------------------------------------------
# We train the LoRA adapter on a simple regression task.
# Goal: learn to predict y = sin(x_sum) where x_sum = sum of input features.
# This is artificial but demonstrates that A and B actually learn.

torch.manual_seed(21)   # set PyTorch random seed for reproducibility

N_SAMPLES = 200   # number of training samples

# Create random input data: shape (200, 8)
X_reg = torch.randn(N_SAMPLES, IN_PT)   # each sample has 8 features

# Create target labels: y = tanh(sum of features) - a nonlinear mapping
# tanh squashes values to range (-1, 1), convenient for regression targets
X_sum = X_reg.sum(dim=1, keepdim=True)   # sum features per sample, shape (200, 1)
# We need targets with shape (200, OUT_PT=4), so we broadcast and perturb
y_reg = torch.tanh(X_sum).expand(-1, OUT_PT) + 0.05 * torch.randn(N_SAMPLES, OUT_PT)
# .expand(-1, OUT_PT) repeats the (200,1) tensor to (200,4) without copying memory

print(f"Toy regression dataset:")
print(f"  X shape: {X_reg.shape}  (200 samples, 8 features each)")
print(f"  y shape: {y_reg.shape}  (200 samples, 4 targets each)")
print()

# ----------------------------------------------------------
# B.7 - TRAINING LOOP
# ----------------------------------------------------------

criterion_reg = nn.MSELoss()   # Mean Squared Error loss (standard for regression)
                                # C# analogy: new MseLossFunction()

# optimizer only receives model.parameters() = A and B (NOT W, since it's a buffer)
optimizer_lora = optim.Adam(model.parameters(), lr=0.01)   # Adam optimizer, lr=0.01

NUM_EPOCHS_LORA = 100   # train for 100 epochs

print(f"Training LoRA adapter ({NUM_EPOCHS_LORA} epochs)...")
print()
print(f"  {'Epoch':>5}  |  {'Loss':>10}  |  Note")
print(f"  {'-'*5}  |  {'-'*10}  |  {'-'*35}")

for epoch in range(NUM_EPOCHS_LORA):   # standard training loop

    optimizer_lora.zero_grad()         # clear gradients from previous step

    predictions = model(X_reg)         # forward pass through LoRA model
    loss = criterion_reg(predictions, y_reg)  # compute MSE loss

    loss.backward()                    # backprop: compute gradients for A and B only

    optimizer_lora.step()              # update A and B using Adam

    # Print every 20 epochs
    if (epoch + 1) % 20 == 0:
        note = "B no longer zero" if epoch == 19 else "converging..."
        print(f"  {epoch+1:>5}  |  {loss.item():>10.6f}  |  {note}")

print()

# ----------------------------------------------------------
# B.8 - COMPARE WEIGHTS BEFORE vs AFTER TRAINING
# ----------------------------------------------------------

W_base_after = base.W.clone()         # check if frozen W changed
A_after_pt   = model.A.data.clone()   # LoRA A after training
B_after_pt   = model.B.data.clone()   # LoRA B after training

print("Weights AFTER training:")
print(f"  Base W (first row)  : {W_base_after[0].numpy().round(4)}")
print(f"  LoRA A (first row)  : {A_after_pt[0].numpy().round(4)}")
print(f"  LoRA B (first row)  : {B_after_pt[0].numpy().round(4)}")
print()

# Compute weight norms (overall magnitude of each matrix)
# .norm() = Frobenius norm (square root of sum of all squared elements)
# C# analogy: Math.Sqrt(matrix.Cast<float>().Sum(x => x * x))
W_norm_before = W_base_before.norm().item()   # .item() converts tensor to Python float
W_norm_after  = W_base_after.norm().item()
A_norm_before = A_before_pt.norm().item()
A_norm_after  = A_after_pt.norm().item()
B_norm_before = B_before_pt.norm().item()
B_norm_after  = B_after_pt.norm().item()

print("Weight norm comparison (before vs after training):")
print(f"  {'Matrix':8s}  |  {'Before':>12}  |  {'After':>12}  |  Changed?")
print(f"  {'-'*8}  |  {'-'*12}  |  {'-'*12}  |  {'-'*8}")
print(f"  {'W (base)':8s}  |  {W_norm_before:>12.6f}  |  {W_norm_after:>12.6f}  |  {not abs(W_norm_after - W_norm_before) < 1e-6}")
print(f"  {'A (LoRA)':8s}  |  {A_norm_before:>12.6f}  |  {A_norm_after:>12.6f}  |  {abs(A_norm_after - A_norm_before) > 1e-6}")
print(f"  {'B (LoRA)':8s}  |  {B_norm_before:>12.6f}  |  {B_norm_after:>12.6f}  |  {abs(B_norm_after - B_norm_before) > 1e-6}")
print()

# Check that W is truly unchanged
W_truly_frozen = torch.allclose(W_base_before, W_base_after)  # allclose = approximately equal
print(f"Frozen W unchanged: {W_truly_frozen}  (should be True)")
print(f"A changed: {not torch.allclose(A_before_pt, A_after_pt)}  (should be True)")
print(f"B changed: {not torch.allclose(B_before_pt, B_after_pt)}  (should be True)")
print()

# ----------------------------------------------------------
# B.9 - DEMONSTRATE THE MERGE OPERATION IN PYTORCH
# ----------------------------------------------------------

W_merged_pt = model.get_merged_weight()   # compute W + scale * (B @ A)

print("LoRA Merge in PyTorch:")
print(f"  Base W shape   : {base.W.shape}")
print(f"  B @ A shape    : {(model.B @ model.A).shape}")
print(f"  W_merged shape : {W_merged_pt.shape}")
print()
print(f"  Base W (first row)    : {base.W[0].detach().numpy().round(4)}")
print(f"  W_merged (first row)  : {W_merged_pt[0].detach().numpy().round(4)}")
print()

# Verify merge: forward pass with LoRA adapter == forward pass with merged W
x_check = torch.randn(1, IN_PT)   # one test sample

output_lora_pt   = model(x_check)                          # output via LoRA model
output_merged_pt = x_check @ W_merged_pt.T                 # output via merged W only

print("Merge verification:")
print(f"  LoRA model output  : {output_lora_pt.detach().numpy().round(6)}")
print(f"  Merged W output    : {output_merged_pt.detach().numpy().round(6)}")
print(f"  Are they equal?    : {torch.allclose(output_lora_pt, output_merged_pt, atol=1e-5)}")
print()

# ----------------------------------------------------------
# B.10 - SHOW THAT GRADIENTS FLOW ONLY TO A AND B
# ----------------------------------------------------------
# Let's do one more forward+backward and inspect which tensors have gradients.

optimizer_lora.zero_grad()               # clear gradients
out = model(X_reg[:5])                   # forward pass on 5 samples
loss_check = criterion_reg(out, y_reg[:5])  # compute loss
loss_check.backward()                    # backprop

print("Gradient inspection after backward pass:")
print(f"  base.W.grad     : {base.W.grad}  (None means no gradient - frozen)")
print(f"  model.A.grad is None : {model.A.grad is None}  (should be False)")
print(f"  model.B.grad is None : {model.B.grad is None}  (should be False)")
if model.A.grad is not None:
    print(f"  model.A.grad (first row): {model.A.grad[0].numpy().round(6)}")
if model.B.grad is not None:
    print(f"  model.B.grad (first row): {model.B.grad[0].numpy().round(6)}")
print()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 60)
print("SUMMARY - LoRA Key Points")
print("=" * 60)
print()
print("1. LoRA adds two small matrices A and B alongside frozen W.")
print("2. The update is: output += scale * (B @ A @ x)")
print("3. B is initialized to zeros: LoRA starts with zero effect.")
print("4. Only A and B are trained: W is never touched.")
print("5. After training: merge W_merged = W + scale*(B @ A).")
print("6. Merged model has the same speed as original (no extra math).")
print()
print("Parameter savings at scale (4096x4096 layer, rank=8):")
print(f"  Full fine-tune : {4096*4096:,} parameters per layer")
print(f"  LoRA (r=8)     : {2*8*4096:,} parameters per layer")
savings_real = (1 - 2*8*4096 / (4096*4096)) * 100
print(f"  Savings        : {savings_real:.2f}%")
print()
print("C#/.NET analogies:")
print("  Frozen W       = readonly field in a sealed base class")
print("  LoRA A, B      = overrides in a thin derived class")
print("  Merge          = flattening inheritance for deployment")
print("  scale          = a weighting coefficient (like 1.0f / rank)")
print("  B=zeros init   = ensures the derived class starts as a no-op")
print()
print("Done! You have completed all three LoRA examples.")

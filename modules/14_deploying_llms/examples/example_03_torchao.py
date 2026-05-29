# =============================================================================
# Module 14 - Deploying LLMs
# Example 03: TorchAO -- PyTorch-Native Quantization
# =============================================================================
#
# WHAT THIS FILE TEACHES:
#   - What TorchAO does and why it exists
#   - How to quantize a PyTorch model to INT8 and INT4
#   - How to measure memory reduction and speed improvement
#   - When to use TorchAO vs GGUF
#
# GLOSSARY:
#   TorchAO       - "Torch Architecture Optimization". PyTorch's official
#                   quantization library. Developed by Meta's PyTorch team.
#   quantize_()   - TorchAO's main function. Modifies model IN-PLACE.
#                   (The trailing _ means in-place in Python/PyTorch)
#   int8_weight_only     - Store weights as INT8. Activations stay fp16.
#                          4x memory reduction. Minimal quality loss.
#   int4_weight_only     - Store weights as INT4. 8x memory reduction.
#                          Small quality loss. Often faster than fp16.
#   torch.compile - PyTorch's JIT compiler. Compiles model to optimized
#                   machine code for maximum speed.
#   linear layer  - nn.Linear: y = x @ W + b. The core of every transformer.
#                   These are the layers we quantize.
#   in-place      - Modifies the original object directly (no copy returned)
#
# C# ANALOGY:
#   TorchAO is like running your C# app in Release mode vs Debug mode.
#   In Release mode, the JIT compiler:
#     - Uses SIMD instructions for loops (like TorchAO's kernel fusion)
#     - Inlines small methods (like operator fusion)
#     - Uses 4-byte int instead of 8-byte long where possible (like INT8)
#   TorchAO does the same optimizations for neural network math.
#
# REQUIREMENTS:
#   Part A: numpy only (no PyTorch -- simulates what TorchAO does)
#   Part B: pip install torch torchao
#
# =============================================================================

import numpy as np   # numpy for Part A (manual simulation)

# =============================================================================
# PART A: MANUAL SIMULATION OF WHAT TorchAO DOES
#
# We simulate INT8 weight-only quantization manually.
# This is exactly what TorchAO's int8_weight_only() does internally.
# Understanding this makes the TorchAO API meaningful.
# =============================================================================

print("=" * 60)
print("PART A: Manual INT8 Weight-Only Quantization")
print("(Simulating what TorchAO does under the hood)")
print("=" * 60)

# ---------------------------------------------------------
# Simulate a simple linear layer (y = x @ W + b)
# ---------------------------------------------------------

np.random.seed(42)   # Reproducible results

# Linear layer parameters
INPUT_DIM = 256     # Input features (like embedding dimension)
OUTPUT_DIM = 512    # Output features

# Weight matrix W: shape (OUTPUT_DIM, INPUT_DIM)
# In PyTorch nn.Linear, weight shape is (out_features, in_features)
W_fp32 = np.random.randn(OUTPUT_DIM, INPUT_DIM).astype(np.float32)

# Bias vector b: shape (OUTPUT_DIM,)
b_fp32 = np.random.randn(OUTPUT_DIM).astype(np.float32)

print(f"\nLinear layer: input={INPUT_DIM}, output={OUTPUT_DIM}")
print(f"Weight matrix shape: {W_fp32.shape}")
print(f"Bias vector shape:   {b_fp32.shape}")

fp32_weight_bytes = W_fp32.nbytes    # Total bytes used by the weight matrix
print(f"\nWeight memory (fp32): {fp32_weight_bytes:,} bytes ({fp32_weight_bytes/1024:.1f} KB)")


# ---------------------------------------------------------
# INT8 weight-only quantization of the linear layer
#
# "weight-only" means:
#   - Weights: stored as INT8 (saves memory)
#   - Activations (input x): stay as fp32 (no quantization)
#   - At compute time: dequantize weights back to fp32, then do matmul
# ---------------------------------------------------------

def quantize_linear_layer_int8(W):
    """
    Quantize weight matrix to INT8, per output channel.

    "Per output channel" = each ROW of W gets its own scale factor.
    This is better than one global scale because each output neuron
    can have a very different weight distribution.

    Parameters:
        W: float32 array of shape (out_dim, in_dim)
    Returns:
        W_int8: int8 array of shape (out_dim, in_dim)
        scales: float32 array of shape (out_dim,) -- one scale per output channel
    """
    out_dim = W.shape[0]    # Number of output channels (rows)

    # Find max absolute value in each ROW separately
    # np.max(..., axis=1) computes max along axis 1 (across columns)
    # keepdims=True keeps the shape (out_dim, 1) instead of collapsing to (out_dim,)
    max_abs_per_row = np.max(np.abs(W), axis=1, keepdims=True)
    # Shape: (OUTPUT_DIM, 1)

    # One scale factor per row (per output channel)
    scales = (max_abs_per_row / 127.0).astype(np.float32)   # Divide by INT8 max
    # Shape: (OUTPUT_DIM, 1) -- broadcast-compatible with W

    # Quantize: divide by scale, round to nearest integer
    # Broadcasting: scales is (OUTPUT_DIM, 1), W is (OUTPUT_DIM, INPUT_DIM)
    # numpy automatically applies each row's scale to that row
    W_int8 = np.round(W / scales).clip(-127, 127).astype(np.int8)

    # Squeeze scales from (OUTPUT_DIM, 1) to (OUTPUT_DIM,)
    scales = scales.squeeze()    # Remove the extra dimension of size 1

    return W_int8, scales


def dequantize_and_compute(x_fp32, W_int8, scales, b_fp32):
    """
    Dequantize INT8 weights and compute the linear layer output.

    This is what happens at inference time:
    1. Dequantize INT8 back to fp32
    2. Compute y = x @ W.T + b as normal fp32 matmul

    Parameters:
        x_fp32:  float32 array of shape (batch_size, in_dim) -- the input
        W_int8:  int8 array   of shape (out_dim, in_dim)     -- quantized weights
        scales:  float32 array of shape (out_dim,)            -- scale factors
        b_fp32:  float32 array of shape (out_dim,)            -- bias
    Returns:
        y: float32 array of shape (batch_size, out_dim)
    """
    # Dequantize: multiply each row of W_int8 by its scale
    # scales[:, np.newaxis] reshapes (out_dim,) to (out_dim, 1) for broadcasting
    W_dequant = W_int8.astype(np.float32) * scales[:, np.newaxis]

    # Matrix multiply: x @ W.T
    # x shape: (batch, in_dim), W.T shape: (in_dim, out_dim)
    # Result:  (batch, out_dim)
    y = x_fp32 @ W_dequant.T + b_fp32   # @ is matrix multiply in Python

    return y


# Quantize the weight matrix
W_int8, scales = quantize_linear_layer_int8(W_fp32)

int8_weight_bytes = W_int8.nbytes    # Total bytes used by INT8 weights
scales_bytes = scales.nbytes         # Overhead: the scale factors
total_int8_bytes = int8_weight_bytes + scales_bytes

print(f"\nAfter INT8 quantization:")
print(f"  INT8 weights:  {int8_weight_bytes:,} bytes ({int8_weight_bytes/1024:.1f} KB)")
print(f"  Scale factors: {scales_bytes:,} bytes ({scales_bytes} bytes)")
print(f"  Total:         {total_int8_bytes:,} bytes ({total_int8_bytes/1024:.1f} KB)")
print(f"  Reduction:     {fp32_weight_bytes / total_int8_bytes:.2f}x smaller")
# Expected: slightly less than 4x because of the scale factor overhead

# ---------------------------------------------------------
# Verify that quantized output matches original output
# ---------------------------------------------------------

batch_size = 4   # Process 4 samples at once
x_test = np.random.randn(batch_size, INPUT_DIM).astype(np.float32)   # Random input

# Original fp32 output
y_fp32 = x_test @ W_fp32.T + b_fp32   # Standard matmul

# INT8 quantized output
y_int8 = dequantize_and_compute(x_test, W_int8, scales, b_fp32)

# Compare
max_diff = np.max(np.abs(y_fp32 - y_int8))
mean_diff = np.mean(np.abs(y_fp32 - y_int8))
relative_error = mean_diff / np.mean(np.abs(y_fp32))   # Error relative to output magnitude

print(f"\nOutput comparison (fp32 vs INT8 dequantized):")
print(f"  Max absolute difference:  {max_diff:.6f}")
print(f"  Mean absolute difference: {mean_diff:.6f}")
print(f"  Relative error:           {relative_error:.4%}")
# Expected: very small relative error (< 1%) -- INT8 is highly accurate

# ---------------------------------------------------------
# Show scale factors for first few output channels
# ---------------------------------------------------------
print(f"\nScale factors for first 5 output channels:")
for i in range(5):
    print(f"  Channel {i}: scale = {scales[i]:.6f}, max_weight = {np.max(np.abs(W_fp32[i])):.6f}")
    print(f"            (1 INT8 step = {scales[i]:.6f} in float space)")

# =============================================================================
# PART B: USING TorchAO (THE REAL THING)
#
# Uses the actual TorchAO library for one-call quantization.
# Shows INT8 and INT4 quantization with memory/speed measurement.
# =============================================================================

print("\n" + "=" * 60)
print("PART B: TorchAO -- Real PyTorch Quantization")
print("=" * 60)

# Try to import torch and torchao
try:
    import torch          # PyTorch: the main deep learning framework
    import torch.nn as nn # nn = neural network module (layers, etc.)
    TORCH_AVAILABLE = True
    print("PyTorch is installed. Version:", torch.__version__)
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install: pip install torch")

try:
    from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
    TORCHAO_AVAILABLE = True
    print("TorchAO is installed.")
except ImportError:
    TORCHAO_AVAILABLE = False
    print("TorchAO not installed. Install: pip install torchao")

if not TORCH_AVAILABLE or not TORCHAO_AVAILABLE:
    print("\nShowing code explanation since libraries are not installed:")
    print("-" * 40)
    print("")
    print("Step 1: Define a small transformer-like model")
    print("        class TinyLLM(nn.Module):")
    print("            def __init__(self):")
    print("                self.embed = nn.Embedding(1000, 128)")
    print("                self.layers = nn.ModuleList([nn.Linear(128, 128) * 4])")
    print("                self.head = nn.Linear(128, 1000)")
    print("")
    print("Step 2: Measure memory BEFORE quantization")
    print("        size_before = sum(p.nbytes() for p in model.parameters())")
    print("")
    print("Step 3: Apply INT8 quantization (ONE CALL, model modified in-place)")
    print("        quantize_(model, int8_weight_only())")
    print("        # The trailing _ means in-place (no new model returned)")
    print("")
    print("Step 4: Measure memory AFTER quantization")
    print("        size_after = sum(p.nbytes() for p in model.parameters())")
    print("")
    print("Step 5: Run inference (code is IDENTICAL to before quantization)")
    print("        output = model(input_ids)  # same API, smaller memory!")

else:
    # ---------------------------------------------------------
    # Build a small model to demonstrate quantization
    # ---------------------------------------------------------

    class TinyLLM(nn.Module):
        """
        A tiny transformer-like model for demonstration.
        Has an embedding layer + linear layers + output head.
        Real LLMs have the same structure, just much larger.
        """
        def __init__(self, vocab_size=1000, d_model=256, n_layers=4):
            # super().__init__() calls the parent class (nn.Module) constructor
            # Required for all PyTorch modules
            super().__init__()

            # Embedding: converts token IDs to vectors
            # vocab_size=1000 means 1000 possible tokens
            # d_model=256 means each token becomes a 256-dimensional vector
            self.embed = nn.Embedding(vocab_size, d_model)

            # Linear layers: the "feed-forward" part of each transformer block
            # These are the layers that get quantized (they hold the most weights)
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model)    # Square: in=256, out=256
                for _ in range(n_layers)        # Repeat n_layers times
            ])
            # ModuleList is like List<LinearLayer> in C# -- PyTorch tracks parameters

            # Output head: maps from d_model to vocab_size (predicts next token)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, token_ids):
            """
            Forward pass: run the model to get predictions.
            token_ids: integer tensor of shape (batch_size, seq_len)
            """
            x = self.embed(token_ids)     # Convert token IDs to vectors
            for layer in self.layers:     # Pass through each linear layer
                x = torch.relu(layer(x)) # relu = max(0, x) activation function
            return self.head(x)           # Output logits (raw scores before softmax)


    # ---------------------------------------------------------
    # Helper function to measure model memory
    # ---------------------------------------------------------

    def model_size_mb(model):
        """
        Calculate total memory used by all model parameters.
        param.nelement() = total number of values in the parameter tensor
        param.element_size() = bytes per value (4 for fp32, 1 for int8)
        """
        total_bytes = 0
        for param in model.parameters():
            # nelement() = number of elements (like .Length in C#)
            # element_size() = bytes per element
            total_bytes += param.nelement() * param.element_size()
        return total_bytes / (1024 * 1024)   # Convert bytes to MB


    # ---------------------------------------------------------
    # INT8 Weight-Only Quantization
    # ---------------------------------------------------------

    print("\n--- INT8 Weight-Only Quantization ---")

    # Create the model
    model_int8 = TinyLLM(vocab_size=1000, d_model=256, n_layers=4)
    model_int8.eval()   # eval() switches off dropout, sets batch norm to inference mode
    # C# analogy: like calling Dispose() on training-only objects

    # Measure memory BEFORE
    size_before = model_size_mb(model_int8)
    print(f"\nModel size BEFORE quantization: {size_before:.2f} MB")

    # Count parameters for reference
    param_count = sum(p.numel() for p in model_int8.parameters())
    # numel() = number of elements = parameter count
    print(f"Parameter count: {param_count:,}")

    # Apply INT8 quantization -- ONE FUNCTION CALL
    # quantize_() modifies model IN-PLACE (the _ suffix convention in PyTorch)
    # int8_weight_only() = quantize nn.Linear weights to INT8, leave activations alone
    quantize_(model_int8, int8_weight_only())
    print("Quantization applied (int8_weight_only)")

    # Measure memory AFTER
    size_after_int8 = model_size_mb(model_int8)
    print(f"Model size AFTER INT8 quantization: {size_after_int8:.2f} MB")
    print(f"Reduction: {size_before / size_after_int8:.1f}x smaller")

    # Run a test forward pass to confirm the quantized model still works
    dummy_input = torch.randint(0, 1000, (1, 16))   # batch=1, seq_len=16 token IDs
    with torch.no_grad():   # no_grad: don't track gradients (we're not training)
        output_int8 = model_int8(dummy_input)

    print(f"Test forward pass output shape: {output_int8.shape}")
    # Should print: torch.Size([1, 16, 1000]) -- 1000 logits per token position

    # ---------------------------------------------------------
    # INT4 Weight-Only Quantization
    # ---------------------------------------------------------

    print("\n--- INT4 Weight-Only Quantization ---")

    # Create a fresh model for INT4 (separate from the INT8 one)
    model_int4 = TinyLLM(vocab_size=1000, d_model=256, n_layers=4)
    model_int4.eval()

    size_before_int4 = model_size_mb(model_int4)

    # Apply INT4 quantization
    # group_size=32: every 32 weights share one scale factor (standard GGUF behavior)
    quantize_(model_int4, int4_weight_only(group_size=32))
    print("Quantization applied (int4_weight_only, group_size=32)")

    size_after_int4 = model_size_mb(model_int4)
    print(f"Model size BEFORE: {size_before_int4:.2f} MB")
    print(f"Model size AFTER:  {size_after_int4:.2f} MB")
    print(f"Reduction: {size_before_int4 / size_after_int4:.1f}x smaller")

    # Test INT4 forward pass
    with torch.no_grad():
        output_int4 = model_int4(dummy_input)
    print(f"Test forward pass output shape: {output_int4.shape}")

    # ---------------------------------------------------------
    # Compare fp32 vs INT8 vs INT4 outputs
    # ---------------------------------------------------------

    print("\n--- Output Comparison ---")

    # Create original fp32 model with same weights as int8 model
    # (We can't compare directly because quantize_() modifies the model)
    # Instead: create a fresh model with fixed random seed and compare

    torch.manual_seed(42)   # Fix random seed for reproducibility
    model_fp32 = TinyLLM(vocab_size=1000, d_model=256, n_layers=4)
    model_fp32.eval()

    # Copy weights to int8 and int4 models before quantizing
    torch.manual_seed(42)   # Same seed = same weights
    model_int8_cmp = TinyLLM(vocab_size=1000, d_model=256, n_layers=4)
    model_int8_cmp.eval()
    quantize_(model_int8_cmp, int8_weight_only())

    torch.manual_seed(42)
    model_int4_cmp = TinyLLM(vocab_size=1000, d_model=256, n_layers=4)
    model_int4_cmp.eval()
    quantize_(model_int4_cmp, int4_weight_only(group_size=32))

    # Same input for all three models
    test_input = torch.randint(0, 1000, (1, 8))   # 8 tokens

    with torch.no_grad():
        out_fp32 = model_fp32(test_input)       # Original fp32 output
        out_int8 = model_int8_cmp(test_input)   # INT8 quantized output
        out_int4 = model_int4_cmp(test_input)   # INT4 quantized output

    # Compare using the first token's logits (1000 values)
    # torch.max gives the highest-probability token prediction
    pred_fp32 = torch.argmax(out_fp32[0, 0]).item()  # item() converts tensor to Python int
    pred_int8 = torch.argmax(out_int8[0, 0]).item()
    pred_int4 = torch.argmax(out_int4[0, 0]).item()

    print(f"Top prediction token (same input, 3 precisions):")
    print(f"  fp32: token {pred_fp32}")
    print(f"  INT8: token {pred_int8}  {'(SAME)' if pred_int8 == pred_fp32 else '(DIFFERENT)'}")
    print(f"  INT4: token {pred_int4}  {'(SAME)' if pred_int4 == pred_fp32 else '(DIFFERENT)'}")

    # Calculate mean absolute error of logits
    err_int8 = (out_fp32 - out_int8).abs().mean().item()
    err_int4 = (out_fp32 - out_int4).abs().mean().item()
    print(f"\nMean logit error vs fp32:")
    print(f"  INT8: {err_int8:.4f}")
    print(f"  INT4: {err_int4:.4f}")
    print(f"  INT4 error is {err_int4/err_int8:.1f}x larger than INT8 (expected)")

    # ---------------------------------------------------------
    # Memory summary table
    # ---------------------------------------------------------

    print("\n--- Memory Summary ---")
    print(f"  fp32:  {size_before:.2f} MB  (baseline)")
    print(f"  INT8:  {size_after_int8:.2f} MB  ({size_before/size_after_int8:.1f}x smaller)")
    print(f"  INT4:  {size_after_int4:.2f} MB  ({size_before/size_after_int4:.1f}x smaller)")
    print("\nFor a 7B parameter model (scale these numbers up):")
    print(f"  fp32:  28.0 GB")
    print(f"  INT8:   7.0 GB  (~4x smaller)")
    print(f"  INT4:   3.5 GB  (~8x smaller)")

# =============================================================================
# WHEN TO USE TorchAO vs GGUF -- DECISION GUIDE
# =============================================================================

print("\n" + "=" * 60)
print("TorchAO vs GGUF -- When to Use Each")
print("=" * 60)
print("")
print("USE TorchAO WHEN:")
print("  - Custom PyTorch model (not standard LLaMA/Mistral architecture)")
print("  - Need GPU (CUDA) acceleration")
print("  - Building Python-native serving pipeline")
print("  - Want precise control over which layers are quantized")
print("  - Doing quantization-aware training (QAT)")
print("")
print("USE GGUF/Ollama WHEN:")
print("  - Standard architecture (LLaMA, Mistral, Phi, etc.)")
print("  - Want CPU inference without a GPU")
print("  - Want the simplest possible setup (ollama pull)")
print("  - Want to share the quantized model with others")
print("  - Using the model from non-Python code (C#, Java, etc.)")
print("")
print("=" * 60)
print("Key Takeaways:")
print("=" * 60)
print("1. TorchAO quantizes any nn.Module with one function call.")
print("2. quantize_(model, int8_weight_only()) modifies model IN-PLACE.")
print("3. INT8 saves ~4x memory. INT4 saves ~8x memory.")
print("4. Inference code is IDENTICAL after quantization -- same API.")
print("5. torch.compile() + TorchAO = maximum speed on GPU.")

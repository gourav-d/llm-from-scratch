# =============================================================================
# Module 14 - Deploying LLMs
# Exercise 03: TorchAO -- PyTorch-Native Quantization
# =============================================================================
#
# INSTRUCTIONS:
#   Complete each TODO section.
#   Part A uses only numpy (manual simulation).
#   Part B uses torch + torchao: pip install torch torchao
#
# WHAT YOU PRACTICE:
#   - Manually quantizing a weight matrix per output channel
#   - Using TorchAO's quantize_() API
#   - Measuring memory before/after quantization
#   - Comparing INT8 vs INT4 quality
#
# =============================================================================

import numpy as np

print("=" * 60)
print("Exercise 03: TorchAO Quantization")
print("=" * 60)

# =============================================================================
# TASK 1: Per-Channel INT8 Quantization
#
# In example_03, we did per-channel quantization where each ROW of the weight
# matrix gets its own scale factor.
#
# Now do it per COLUMN instead (per input channel).
# Each COLUMN of W gets its own scale.
#
# When would per-column be better than per-row?
# Answer: When input channel distributions vary more than output channel distributions.
# =============================================================================

print("\n--- Task 1: Per-Column INT8 Quantization ---")

np.random.seed(7)
W = np.random.randn(64, 128).astype(np.float32)   # (out_dim=64, in_dim=128)


def quantize_per_column(W):
    """
    Quantize W to INT8 with one scale factor per COLUMN (per input channel).

    Steps:
      1. For each column j, find max_abs = max(abs(W[:, j]))
      2. scale_j = max_abs / 127
      3. quantized[:, j] = round(W[:, j] / scale_j), clipped to [-127, 127]

    Parameters:
        W: float32 array of shape (out_dim, in_dim)
    Returns:
        W_int8: int8 array of shape (out_dim, in_dim)
        scales: float32 array of shape (in_dim,) -- one scale per column
    """
    out_dim, in_dim = W.shape

    # TODO: compute max abs per column
    # Hint: np.max(np.abs(W), axis=0) computes max along axis 0 (across rows)
    # This gives one value per column
    max_abs_per_col = None   # shape should be (in_dim,)

    # TODO: compute scales
    scales = None   # shape (in_dim,)

    # TODO: quantize
    # Hint: W / scales broadcasts if scales has shape (1, in_dim) or (in_dim,)
    # You may need to reshape: scales.reshape(1, -1) to get shape (1, in_dim)
    W_int8 = None   # shape (out_dim, in_dim), dtype int8

    return W_int8, scales


def dequantize_per_column(W_int8, scales):
    """Recover float32 from per-column INT8."""
    # TODO: multiply W_int8 by scales (broadcast across rows)
    # Hint: scales shape (in_dim,) -> reshape to (1, in_dim) for broadcasting
    return None


W_int8, scales_col = quantize_per_column(W)
W_recovered = dequantize_per_column(W_int8, scales_col)

if W_recovered is not None:
    err = np.mean(np.abs(W - W_recovered))
    print(f"W shape: {W.shape}")
    print(f"Scales shape: {scales_col.shape if scales_col is not None else None}")
    print(f"Mean quantization error: {err:.6f}")
    print(f"Memory: fp32={W.nbytes}B, INT8={W_int8.nbytes}B, ratio={W.nbytes//W_int8.nbytes}x")
else:
    print("TODO not yet complete.")

# =============================================================================
# TASK 2: model_size_mb() -- Calculate Model Memory
#
# Write a function that counts total parameter bytes in a PyTorch model.
# This is the same helper used in example_03.
# =============================================================================

print("\n--- Task 2: model_size_mb() ---")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Skipping Task 2.")

if TORCH_AVAILABLE:
    def model_size_mb(model):
        """
        Calculate total memory (in MB) used by all model parameters.

        For each parameter:
          - param.nelement() gives the count of values (like .Length in C#)
          - param.element_size() gives bytes per value
            (fp32=4, fp16=2, int8=1, etc.)

        TODO: Sum over all parameters and return MB.
        """
        total_bytes = 0
        for param in model.parameters():
            # TODO: add param's byte count to total_bytes
            pass   # Remove this line when you add the real code

        # TODO: convert bytes to MB and return
        return None   # Hint: total_bytes / (1024 * 1024)


    # Build a small model to test with
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.head = nn.Linear(512, 1000)

        def forward(self, x):
            return self.head(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


    model = SmallModel()
    size = model_size_mb(model)

    # Manual check: count parameters
    total_params = sum(p.numel() for p in model.parameters())
    expected_mb = (total_params * 4) / (1024 * 1024)   # fp32 = 4 bytes

    print(f"Model parameters: {total_params:,}")
    print(f"Expected size (fp32): {expected_mb:.2f} MB")
    if size is not None:
        print(f"Your function returns: {size:.2f} MB")
        match = abs(size - expected_mb) < 0.01   # Allow tiny float rounding
        print(f"Correct: {match}")
    else:
        print("TODO not complete yet.")

# =============================================================================
# TASK 3: Apply TorchAO Quantization and Measure Impact
#
# Use TorchAO to quantize a model and compare memory + output quality.
# =============================================================================

print("\n--- Task 3: Apply TorchAO Quantization ---")

try:
    from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    print("TorchAO not installed. Showing expected answers.")

if TORCH_AVAILABLE and TORCHAO_AVAILABLE:

    class TinyLLM(nn.Module):
        """Small model for quantization testing."""
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(500, 128)
            self.layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
            self.head = nn.Linear(128, 500)

        def forward(self, ids):
            x = self.embed(ids)
            for layer in self.layers:
                x = torch.relu(layer(x))
            return self.head(x)


    # TODO: Create three models with the same weights (same random seed)
    # Model A: fp32 (baseline, no quantization)
    # Model B: INT8 (quantize_(model, int8_weight_only()))
    # Model C: INT4 (quantize_(model, int4_weight_only(group_size=32)))

    torch.manual_seed(42)
    model_fp32 = TinyLLM()
    model_fp32.eval()

    torch.manual_seed(42)
    model_int8 = TinyLLM()
    model_int8.eval()
    # TODO: apply INT8 quantization to model_int8
    # quantize_(model_int8, int8_weight_only())

    torch.manual_seed(42)
    model_int4 = TinyLLM()
    model_int4.eval()
    # TODO: apply INT4 quantization to model_int4
    # quantize_(model_int4, int4_weight_only(group_size=32))

    # TODO: use your model_size_mb() function to measure all three
    # (You need to implement model_size_mb first in Task 2)

    # Test input
    test_ids = torch.randint(0, 500, (1, 8))

    with torch.no_grad():
        out_fp32 = model_fp32(test_ids)
        out_int8 = model_int8(test_ids)
        out_int4 = model_int4(test_ids)

    # Compare top-1 predictions
    pred_fp32 = torch.argmax(out_fp32[0, 0]).item()
    pred_int8 = torch.argmax(out_int8[0, 0]).item()
    pred_int4 = torch.argmax(out_int4[0, 0]).item()

    print(f"Top-1 prediction -- fp32: {pred_fp32}, INT8: {pred_int8}, INT4: {pred_int4}")
    print(f"INT8 matches fp32: {pred_int8 == pred_fp32}")
    print(f"INT4 matches fp32: {pred_int4 == pred_fp32}")

    # TODO: Print memory sizes for all three models
    # Expected:
    #   fp32: ~0.7 MB
    #   INT8: ~0.2 MB  (about 4x smaller)
    #   INT4: ~0.1 MB  (about 7-8x smaller)

else:
    print("Expected results (when TorchAO is installed):")
    print("  fp32 size:  ~0.7 MB")
    print("  INT8 size:  ~0.2 MB  (4x smaller)")
    print("  INT4 size:  ~0.1 MB  (7x smaller)")
    print("  Top-1 predictions: usually the same for all 3 precisions")

# =============================================================================
# TASK 4: Which Layers Does TorchAO Quantize?
#
# TorchAO only quantizes nn.Linear layers (not Embedding, LayerNorm, etc.)
# Verify this by inspecting a quantized model's layer types.
# =============================================================================

print("\n--- Task 4: Inspect Quantized Layers ---")

if TORCH_AVAILABLE and TORCHAO_AVAILABLE:

    class InspectModel(nn.Module):
        """Model with several different layer types to inspect."""
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)     # Embedding layer
            self.linear1 = nn.Linear(32, 64)       # Linear layer (gets quantized)
            self.norm = nn.LayerNorm(64)            # LayerNorm (NOT quantized)
            self.linear2 = nn.Linear(64, 32)       # Linear layer (gets quantized)
            self.head = nn.Linear(32, 100)          # Linear layer (gets quantized)

        def forward(self, x):
            return self.head(self.norm(torch.relu(self.linear2(torch.relu(self.linear1(self.embed(x)))))))


    inspect_model = InspectModel()
    inspect_model.eval()

    print("Before quantization:")
    for name, module in inspect_model.named_modules():
        if name:   # Skip the root module (empty name)
            print(f"  {name}: {type(module).__name__}")

    # TODO: Apply INT8 quantization
    # quantize_(inspect_model, int8_weight_only())

    print("\nAfter INT8 quantization:")
    for name, module in inspect_model.named_modules():
        if name:
            print(f"  {name}: {type(module).__name__}")

    print("\nObservation: nn.Linear layers become 'Int8WeightOnlyLinear'.")
    print("nn.Embedding and nn.LayerNorm stay unchanged.")
    print("Only weight matrices are quantized -- not normalization or embeddings.")

else:
    print("Expected output (when TorchAO is installed):")
    print("")
    print("Before quantization:")
    print("  embed:   Embedding")
    print("  linear1: Linear")
    print("  norm:    LayerNorm")
    print("  linear2: Linear")
    print("  head:    Linear")
    print("")
    print("After INT8 quantization:")
    print("  embed:   Embedding          <-- unchanged")
    print("  linear1: Int8WeightOnlyLinear  <-- quantized!")
    print("  norm:    LayerNorm          <-- unchanged")
    print("  linear2: Int8WeightOnlyLinear  <-- quantized!")
    print("  head:    Int8WeightOnlyLinear  <-- quantized!")

# =============================================================================
# HINTS
# =============================================================================

print("\n" + "=" * 60)
print("HINTS")
print("=" * 60)
print("""
Task 1 (per-column quantization):
  max_abs_per_col = np.max(np.abs(W), axis=0)  # shape: (in_dim,)
  scales = max_abs_per_col / 127.0               # shape: (in_dim,)
  W_int8 = np.round(W / scales.reshape(1, -1)).clip(-127, 127).astype(np.int8)
  # Dequantize:
  return W_int8.astype(np.float32) * scales.reshape(1, -1)

Task 2 (model_size_mb):
  for param in model.parameters():
      total_bytes += param.nelement() * param.element_size()
  return total_bytes / (1024 * 1024)

Task 3 (apply quantization):
  quantize_(model_int8, int8_weight_only())
  quantize_(model_int4, int4_weight_only(group_size=32))
  # Then call model_size_mb() on each model

Task 4 (inspect layers):
  quantize_(inspect_model, int8_weight_only())
  # Then loop: for name, module in inspect_model.named_modules():
""")

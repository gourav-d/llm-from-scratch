# =============================================================================
# Module 14 - Deploying LLMs
# Example 01: Quantization Concepts
# =============================================================================
#
# WHAT THIS FILE TEACHES:
#   - How float32 weights are converted to INT8 and INT4
#   - Scale factors and how to dequantize back to float
#   - Why small precision loss is acceptable
#   - How to measure quantization error
#
# GLOSSARY (read before the code):
#   quantization  - Converting high-precision numbers (float32) to lower-precision
#                   integers (INT8, INT4) to shrink model size
#   scale factor  - A multiplier that maps the float range into integer range
#   zero point    - An offset for asymmetric quantization (used for ReLU outputs)
#   dequantize    - Convert back from integer to approximate float
#   PTQ           - Post-Training Quantization: quantize AFTER training
#   symmetric     - Zero is in the center of the integer range (-127 to 127)
#   asymmetric    - Zero can be anywhere in the range (adds a zero_point offset)
#
# C# ANALOGY:
#   Quantization is like storing a float as a byte in C#:
#     float weight = 1.837f;         // 4 bytes, high precision
#     byte  stored = 94;             // 1 byte, after mapping
#     float recovered = stored * scale; // approximate original
#   The stored byte uses 4x less memory. Small error is acceptable.
#
# LIBRARIES:
#   numpy  - numerical arrays (like List<double> but faster and multi-dimensional)
#
# =============================================================================

import numpy as np   # numpy = numerical Python, for arrays and math

# =============================================================================
# PART A: INT8 QUANTIZATION FROM SCRATCH
#
# We build quantization step-by-step with no extra libraries.
# This is exactly what tools like GGUF and TorchAO do internally.
# =============================================================================

print("=" * 60)
print("PART A: INT8 Quantization From Scratch")
print("=" * 60)

# ---------------------------------------------------------
# Step 1: Create some fake neural network weights
# ---------------------------------------------------------

# np.random.seed makes results reproducible (same numbers every run)
# In C# this would be: new Random(seed: 42)
np.random.seed(42)

# Simulate a small weight matrix (one layer of a neural network)
# Real LLMs have millions of these values
# Shape (4, 8) means 4 rows, 8 columns = 32 weights total
weights_fp32 = np.random.randn(4, 8).astype(np.float32)
# .astype(np.float32) converts to 32-bit float (default numpy is float64)

print("\nOriginal fp32 weights (first row):")
print(weights_fp32[0])   # Print first row so output stays readable

# Calculate memory usage of the original weights
fp32_bytes = weights_fp32.nbytes   # nbytes = total bytes used
print(f"\nMemory: {fp32_bytes} bytes (fp32)")
print(f"         {fp32_bytes // 4} weights x 4 bytes each")

# ---------------------------------------------------------
# Step 2: Symmetric INT8 Quantization
#
# Symmetric means zero stays at zero.
# We map the range [-max_abs, +max_abs] to [-127, +127].
# ---------------------------------------------------------

def quantize_int8_symmetric(weights):
    """
    Convert float32 weights to INT8 using symmetric quantization.

    Parameters:
        weights: float32 numpy array (any shape)
    Returns:
        quantized: int8 numpy array (same shape)
        scale:     float32 scalar -- needed to recover original values
    """
    # Find the largest absolute value in the weight matrix
    # This determines our "range" -- everything must fit inside it
    max_abs = np.max(np.abs(weights))   # np.abs = absolute value of every element

    # INT8 can hold values from -128 to 127.
    # We use -127 to +127 (symmetric range, avoids edge cases with -128).
    INT8_MAX = 127.0

    # Scale factor: how many integer steps cover the full float range
    # Example: if max_abs = 2.5, scale = 2.5 / 127 = 0.01969
    # This means: 1 integer step = 0.01969 in float space
    scale = max_abs / INT8_MAX

    # Quantize: divide each weight by scale, then round to nearest integer
    # Example: weight = 1.837, scale = 0.01969
    #   1.837 / 0.01969 = 93.3 -> round -> 93 (INT8 value)
    quantized = np.round(weights / scale).astype(np.int8)
    # .astype(np.int8) converts to 8-bit integer
    # np.round() rounds to nearest integer (0.5 rounds to nearest even)

    return quantized, scale   # Return both -- we need scale to dequantize


def dequantize_int8(quantized, scale):
    """
    Recover approximate float32 values from INT8.

    Parameters:
        quantized: int8 numpy array
        scale:     float32 scalar (same scale used during quantization)
    Returns:
        recovered: float32 array (approximately equal to original)
    """
    # Multiply integer values by scale to get back approximate floats
    # Example: 93 * 0.01969 = 1.831 (close to original 1.837)
    recovered = quantized.astype(np.float32) * scale
    # .astype(np.float32) converts int8 back to float32 before multiplying
    return recovered


# Run quantization
quantized_int8, scale_int8 = quantize_int8_symmetric(weights_fp32)

print(f"\nScale factor: {scale_int8:.6f}")
print(f"(Each INT8 step = {scale_int8:.6f} in float space)")

print("\nOriginal fp32 (first row):")
print(weights_fp32[0])

print("\nQuantized INT8 (first row):")
print(quantized_int8[0])

print("\nDequantized (recovered from INT8, first row):")
recovered_int8 = dequantize_int8(quantized_int8, scale_int8)
print(recovered_int8[0])

# ---------------------------------------------------------
# Step 3: Measure the quantization error
# ---------------------------------------------------------

# Calculate how much error the quantization introduced
# We want this to be small (good quantization) not large (bad)
absolute_error = np.abs(weights_fp32 - recovered_int8)  # Absolute difference at each position
max_error = np.max(absolute_error)     # Worst single error across all weights
mean_error = np.mean(absolute_error)   # Average error across all weights

print(f"\nQuantization Error:")
print(f"  Max error:  {max_error:.6f}")
print(f"  Mean error: {mean_error:.6f}")
print(f"  Scale factor was: {scale_int8:.6f}")
print(f"  (Max error should be about half a scale step: {scale_int8/2:.6f})")
# The maximum error is at most half a scale step because we use rounding
# C# analogy: like rounding 1.837 to 2 when storing as int -- max error is 0.5

# ---------------------------------------------------------
# Step 4: Show memory savings
# ---------------------------------------------------------

int8_bytes = quantized_int8.nbytes    # Total bytes used by int8 array
print(f"\nMemory Comparison:")
print(f"  fp32:  {fp32_bytes} bytes")
print(f"  INT8:  {int8_bytes} bytes")
print(f"  Savings: {fp32_bytes / int8_bytes:.0f}x smaller")
# Expected: 4x smaller because INT8 uses 1 byte vs fp32's 4 bytes

# =============================================================================
# INT4 QUANTIZATION
#
# INT4 uses only 4 bits per weight.
# That means only 16 possible values: -8 to 7.
# Much smaller, but more precision loss.
# =============================================================================

print("\n" + "=" * 60)
print("INT4 Quantization")
print("=" * 60)

def quantize_int4_symmetric(weights):
    """
    Convert float32 weights to INT4 (stored as int8 because numpy has no int4).

    INT4 range is -8 to 7 (not -7 to 7) because of signed representation.
    We use -7 to 7 (15 values) to keep it symmetric.

    Parameters:
        weights: float32 numpy array
    Returns:
        quantized: int8 numpy array (values in range -7 to 7)
        scale:     float32 scalar
    """
    max_abs = np.max(np.abs(weights))

    # INT4 symmetric range: -7 to +7 (only 15 distinct values)
    INT4_MAX = 7.0

    # Scale is larger here -- each step covers more float space
    # Less precision than INT8 because we have fewer steps
    scale = max_abs / INT4_MAX

    # Quantize and clip to valid INT4 range (-7 to 7)
    quantized = np.clip(np.round(weights / scale), -7, 7).astype(np.int8)
    # np.clip() ensures values stay within [-7, 7] -- prevents overflow
    # np.round() rounds to nearest integer

    return quantized, scale


quantized_int4, scale_int4 = quantize_int4_symmetric(weights_fp32)
recovered_int4 = dequantize_int8(quantized_int4, scale_int4)   # same dequantize function works

print(f"\nINT4 scale factor: {scale_int4:.6f}")
print(f"(Each INT4 step = {scale_int4:.6f} in float space)")
print(f"INT8 scale factor: {scale_int8:.6f}")
print(f"(INT4 steps are {scale_int4/scale_int8:.1f}x coarser than INT8)")

# Measure INT4 error
int4_error = np.abs(weights_fp32 - recovered_int4)
print(f"\nINT4 Quantization Error:")
print(f"  Max error:  {np.max(int4_error):.6f}")
print(f"  Mean error: {np.mean(int4_error):.6f}")
print(f"\nINT8 Quantization Error for comparison:")
print(f"  Max error:  {max_error:.6f}")
print(f"  Mean error: {mean_error:.6f}")

# Print side-by-side comparison for one weight
idx = 0   # Check the first weight in the first row
w = weights_fp32[0, idx]
print(f"\nSide-by-side comparison for weight[0,0] = {w:.6f}:")
print(f"  INT8 recovered: {recovered_int8[0, idx]:.6f}  (error: {abs(w - recovered_int8[0,idx]):.6f})")
print(f"  INT4 recovered: {recovered_int4[0, idx]:.6f}  (error: {abs(w - recovered_int4[0,idx]):.6f})")

# =============================================================================
# PART B: GROUP QUANTIZATION (KEY FOR INT4 QUALITY)
#
# Problem: One scale factor for the whole layer is not precise enough for INT4.
# Solution: Give each GROUP of 32 weights its own scale factor.
#           Each small group can be quantized more accurately.
# =============================================================================

print("\n" + "=" * 60)
print("PART B: Group Quantization")
print("(Why INT4 in GGUF models is actually good quality)")
print("=" * 60)

# Create a larger, more realistic weight vector
# (Simulates one row of a weight matrix in a real LLM)
np.random.seed(123)
large_weights = np.random.randn(128).astype(np.float32)  # 128 weights in one row

GROUP_SIZE = 32   # GGUF uses 32 weights per group (this is the standard)

def quantize_int4_grouped(weights, group_size=32):
    """
    Quantize weights with one scale factor per group of N weights.
    This gives much better accuracy than one global scale factor.

    Parameters:
        weights:    1D float32 array
        group_size: how many weights share one scale factor
    Returns:
        quantized:  1D int8 array (INT4 values stored as int8)
        scales:     1D float32 array (one scale per group)
    """
    n = len(weights)    # Total number of weights
    n_groups = n // group_size   # Number of groups (e.g., 128 // 32 = 4 groups)

    quantized = np.zeros(n, dtype=np.int8)   # Output array (int8 used to store 4-bit values)
    scales = np.zeros(n_groups, dtype=np.float32)   # One scale per group

    for g in range(n_groups):
        # Get the slice of weights belonging to this group
        start = g * group_size           # First index in this group
        end = start + group_size         # Last index (exclusive)
        group = weights[start:end]       # This group's weights (32 values)

        # Compute scale factor just for this group
        max_abs = np.max(np.abs(group))
        scale = max_abs / 7.0   # INT4 max is 7

        # Quantize just this group
        quantized[start:end] = np.clip(np.round(group / scale), -7, 7).astype(np.int8)
        scales[g] = scale    # Remember this group's scale for dequantization

    return quantized, scales


def dequantize_int4_grouped(quantized, scales, group_size=32):
    """Recover float32 weights from grouped INT4 quantization."""
    n = len(quantized)
    n_groups = n // group_size
    recovered = np.zeros(n, dtype=np.float32)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        # Each group uses its own scale to dequantize
        recovered[start:end] = quantized[start:end].astype(np.float32) * scales[g]

    return recovered


# Quantize WITHOUT grouping (one global scale for 128 weights)
quantized_global, scale_global = quantize_int4_symmetric(large_weights)
recovered_global = dequantize_int8(quantized_global, scale_global)

# Quantize WITH grouping (one scale per 32 weights)
quantized_grouped, scales_grouped = quantize_int4_grouped(large_weights, GROUP_SIZE)
recovered_grouped = dequantize_int4_grouped(quantized_grouped, scales_grouped, GROUP_SIZE)

# Compare errors
global_error = np.abs(large_weights - recovered_global)
grouped_error = np.abs(large_weights - recovered_grouped)

print(f"\n128 weights, INT4 quantization:")
print(f"\nGlobal scale (1 scale for all 128 weights):")
print(f"  Scale value: {scale_global:.6f}")
print(f"  Max error:   {np.max(global_error):.6f}")
print(f"  Mean error:  {np.mean(global_error):.6f}")

print(f"\nGrouped scale ({GROUP_SIZE} weights per group = {len(scales_grouped)} scales):")
print(f"  Scale values: {scales_grouped}")
print(f"  Max error:   {np.max(grouped_error):.6f}")
print(f"  Mean error:  {np.mean(grouped_error):.6f}")

improvement = np.mean(global_error) / np.mean(grouped_error)
print(f"\nGrouped is {improvement:.1f}x more accurate than global!")
print("This is why GGUF Q4_K_M models are much better than naive Q4_0.")

# =============================================================================
# MEMORY SAVINGS SUMMARY
#
# Show the full math for a real 7B parameter model
# =============================================================================

print("\n" + "=" * 60)
print("Memory Savings Summary -- Real Model Sizes")
print("=" * 60)

model_params = 7_000_000_000   # LLaMA-7B has 7 billion parameters

# Calculate memory at different precisions
fp32_gb = (model_params * 4) / (1024 ** 3)    # 4 bytes per fp32
fp16_gb = (model_params * 2) / (1024 ** 3)    # 2 bytes per fp16/bf16
int8_gb = (model_params * 1) / (1024 ** 3)    # 1 byte per INT8
int4_gb = (model_params * 0.5) / (1024 ** 3)  # 0.5 bytes per INT4

print(f"\nLLaMA-7B ({model_params:,} parameters):")
print(f"  fp32:  {fp32_gb:.1f} GB  (4 bytes per weight)")
print(f"  fp16:  {fp16_gb:.1f} GB  (2 bytes per weight)")
print(f"  INT8:  {int8_gb:.1f} GB  (1 byte per weight)")
print(f"  INT4:  {int4_gb:.1f} GB  (0.5 bytes per weight)")

print("\nWhat hardware can run it:")
print(f"  fp32  ({fp32_gb:.0f} GB):  Needs a workstation with 32+ GB RAM")
print(f"  fp16  ({fp16_gb:.0f} GB):  Needs a 16 GB GPU (RTX 4080)")
print(f"  INT8  ({int8_gb:.0f} GB):   Fits on a laptop with 8 GB RAM")
print(f"  INT4  ({int4_gb:.1f} GB): Fits on a phone with 4 GB RAM!")

print("\nQuality tradeoff (approximate benchmark scores, higher is better):")
print("  fp32:  100%  (baseline)")
print("  bf16:  100%  (identical -- bf16 preserves range, only precision drops)")
print("  INT8:   99%  (barely noticeable)")
print("  INT4:   96%  (small but measurable on hard tasks)")
print("  INT2:   70%  (significant degradation, usually not usable)")

print("\n" + "=" * 60)
print("Key Takeaways:")
print("=" * 60)
print("1. INT8 cuts memory 4x with almost no quality loss.")
print("2. INT4 cuts memory 8x with small quality loss (worth it for local use).")
print("3. Group quantization makes INT4 nearly as good as INT8.")
print("4. Scale factor = the bridge between float and integer worlds.")
print("5. Max error per weight is at most half a scale step (rounding).")

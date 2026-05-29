# =============================================================================
# Module 14 - Deploying LLMs
# Exercise 01: Quantization Concepts
# =============================================================================
#
# INSTRUCTIONS:
#   Complete each TODO section below.
#   Run the file to check your answers against the expected output.
#   Scroll to the bottom for hints and solutions.
#
# WHAT YOU PRACTICE:
#   - Calculating model size from parameter count and data type
#   - Implementing INT8 symmetric quantization from scratch
#   - Implementing asymmetric (zero-point) quantization
#   - Measuring and comparing quantization error
#
# LIBRARIES NEEDED: numpy only
#
# =============================================================================

import numpy as np

print("=" * 60)
print("Exercise 01: Quantization Concepts")
print("=" * 60)

# =============================================================================
# EXERCISE 1: Model Size Calculator
#
# Given a model's parameter count and data type, calculate memory in GB.
#
# Formula: memory_gb = (params * bytes_per_param) / (1024^3)
# =============================================================================

print("\n--- Exercise 1: Model Size Calculator ---")

def calculate_model_size_gb(num_params, dtype):
    """
    Calculate how much memory (in GB) a model needs.

    Parameters:
        num_params: int  -- number of parameters in the model
        dtype: str       -- one of "fp32", "fp16", "bf16", "int8", "int4"

    Returns:
        float -- model size in gigabytes

    TODO: Fill in the bytes_per_param dict and the formula.
    """
    # TODO: Map each dtype to its bytes per parameter
    # fp32  = 4 bytes  (32 bits / 8 bits-per-byte)
    # fp16  = 2 bytes
    # bf16  = 2 bytes  (same as fp16 -- same bit count, different layout)
    # int8  = 1 byte
    # int4  = 0.5 bytes (two 4-bit values fit in one byte)
    bytes_per_param = {
        # TODO: fill in this dictionary
    }

    if dtype not in bytes_per_param:
        raise ValueError(f"Unknown dtype: {dtype}. Must be one of {list(bytes_per_param.keys())}")

    # TODO: Calculate total bytes, then convert to GB
    # 1 GB = 1024^3 bytes = 1,073,741,824 bytes
    total_bytes = None   # TODO: num_params * bytes_per_param[dtype]
    size_gb = None       # TODO: total_bytes / (1024 ** 3)

    return size_gb


# Test your function with these values
test_cases = [
    (117_000_000, "fp32"),    # GPT-2: 117M params, fp32 -> should be ~0.43 GB
    (7_000_000_000, "fp32"),  # LLaMA-7B, fp32 -> should be ~26 GB
    (7_000_000_000, "int8"),  # LLaMA-7B, INT8 -> should be ~6.5 GB
    (7_000_000_000, "int4"),  # LLaMA-7B, INT4 -> should be ~3.26 GB
    (70_000_000_000, "int4"), # LLaMA-70B, INT4 -> should be ~32.6 GB
]

print("\nModel size calculations:")
for params, dtype in test_cases:
    size = calculate_model_size_gb(params, dtype)
    print(f"  {params:>15,} params x {dtype:<5} = {size:.2f} GB")

# =============================================================================
# EXERCISE 2: INT8 Symmetric Quantization
#
# Implement quantization that maps a float range to [-127, +127].
# =============================================================================

print("\n--- Exercise 2: INT8 Symmetric Quantization ---")

def quantize_symmetric(weights_fp32):
    """
    Quantize a float32 array to INT8 using symmetric quantization.

    Steps:
      1. Find the maximum absolute value (the "range")
      2. Compute: scale = max_abs / 127
      3. Quantize: quantized = round(weights / scale), clipped to [-127, 127]
      4. Return both the quantized array (int8) and the scale factor (float32)

    Parameters:
        weights_fp32: numpy float32 array (any shape)

    Returns:
        quantized: numpy int8 array (same shape)
        scale:     float32 scalar
    """
    # TODO: Step 1 - find max absolute value
    max_abs = None   # Hint: np.max(np.abs(weights_fp32))

    # TODO: Step 2 - compute scale factor
    INT8_MAX = 127.0
    scale = None   # Hint: max_abs / INT8_MAX

    # TODO: Step 3 - quantize (divide, round, clip to [-127, 127], cast to int8)
    quantized = None   # Hint: np.round(weights_fp32 / scale).clip(-127, 127).astype(np.int8)

    return quantized, scale


def dequantize(quantized, scale):
    """
    Recover approximate float32 values from quantized int8.

    Formula: recovered = quantized * scale

    Parameters:
        quantized: int8 numpy array
        scale:     float32 scalar

    Returns:
        numpy float32 array
    """
    # TODO: multiply quantized (cast to float32 first) by scale
    return None   # Hint: quantized.astype(np.float32) * scale


# Test
np.random.seed(42)
test_weights = np.random.randn(3, 4).astype(np.float32)

print("\nOriginal weights:")
print(test_weights)

quantized, scale = quantize_symmetric(test_weights)
recovered = dequantize(quantized, scale)

print(f"\nScale factor: {scale:.6f}")
print("\nQuantized (int8):")
print(quantized)
print("\nRecovered (dequantized):")
print(recovered)

# Calculate error
if recovered is not None:
    max_err = np.max(np.abs(test_weights - recovered))
    mean_err = np.mean(np.abs(test_weights - recovered))
    print(f"\nMax error:  {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")
    print(f"(Should be close to half a scale step: {scale/2:.6f})")

# =============================================================================
# EXERCISE 3: Asymmetric (Zero-Point) Quantization
#
# Useful when the weight distribution is not centered at zero.
# Example: activations after ReLU are always >= 0, so the range is [0, max].
# Symmetric wastes half the range (the negative part). Asymmetric doesn't.
# =============================================================================

print("\n--- Exercise 3: Asymmetric Quantization ---")

def quantize_asymmetric(values_fp32):
    """
    Quantize float32 values using asymmetric (zero-point) quantization.
    Maps [min_val, max_val] to [0, 255] using scale and zero_point.

    Steps:
      1. Find min_val and max_val of the values
      2. scale = (max_val - min_val) / 255
      3. zero_point = round(-min_val / scale), clipped to [0, 255]
      4. quantized = round(values / scale) + zero_point, clipped to [0, 255]

    Returns:
        quantized:  numpy uint8 array (values in [0, 255])
        scale:      float32 scalar
        zero_point: int
    """
    # TODO: Step 1 - find range
    min_val = None   # Hint: np.min(values_fp32)
    max_val = None   # Hint: np.max(values_fp32)

    # TODO: Step 2 - compute scale
    scale = None   # Hint: (max_val - min_val) / 255.0

    # TODO: Step 3 - compute zero point
    # zero_point is the integer that represents 0.0 in the quantized space
    zero_point = None   # Hint: int(np.round(-min_val / scale))
    zero_point = None   # Hint: then clip to [0, 255]: np.clip(zero_point, 0, 255)

    # TODO: Step 4 - quantize
    quantized = None   # Hint: np.round(values_fp32 / scale) + zero_point
    quantized = None   # Hint: then clip to [0, 255] and cast to uint8


    return quantized, scale, zero_point


def dequantize_asymmetric(quantized, scale, zero_point):
    """
    Recover float32 from asymmetric uint8 quantization.

    Formula: recovered = (quantized - zero_point) * scale
    """
    # TODO: implement dequantization
    return None   # Hint: (quantized.astype(np.float32) - zero_point) * scale


# Test with ReLU-like values (all non-negative, like activations after ReLU)
# ReLU(x) = max(0, x) -- output is always >= 0
relu_activations = np.array([0.0, 0.3, 0.7, 1.2, 2.5, 0.1, 1.8, 0.0, 3.1], dtype=np.float32)

print("\nReLU activation values (all >= 0):")
print(relu_activations)

q, s, zp = quantize_asymmetric(relu_activations)
r = dequantize_asymmetric(q, s, zp)

print(f"\nScale: {s:.6f}")
print(f"Zero point: {zp}")
print(f"\nQuantized (uint8, range 0-255): {q}")
print(f"Recovered: {r}")

if r is not None:
    err = np.max(np.abs(relu_activations - r))
    print(f"Max error: {err:.6f}  (should be tiny)")

# =============================================================================
# EXERCISE 4: Group Quantization Quality Comparison
#
# Show that group quantization gives better accuracy than global quantization.
# =============================================================================

print("\n--- Exercise 4: Group Quantization ---")

def global_quantize_int4(weights):
    """Quantize all weights with one shared scale factor."""
    max_abs = np.max(np.abs(weights))
    scale = max_abs / 7.0   # INT4 max = 7
    quantized = np.clip(np.round(weights / scale), -7, 7).astype(np.int8)
    return quantized, scale


def global_dequantize_int4(quantized, scale):
    return quantized.astype(np.float32) * scale


def group_quantize_int4(weights, group_size=32):
    """
    Quantize with one scale factor per group of N weights.

    TODO: Complete this function.
    Steps for each group:
      1. Slice the group: weights[start:end]
      2. Find max_abs of just this group
      3. Compute scale = max_abs / 7.0
      4. Quantize this group (clip to [-7, 7], cast to int8)
      5. Store the group's quantized values and scale
    """
    n = len(weights)
    n_groups = n // group_size

    quantized = np.zeros(n, dtype=np.int8)
    scales = np.zeros(n_groups, dtype=np.float32)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group = weights[start:end]

        # TODO: compute scale for this group
        max_abs = None   # Hint: np.max(np.abs(group))
        scale = None     # Hint: max_abs / 7.0

        # TODO: quantize this group
        quantized[start:end] = None   # Hint: np.clip(np.round(group / scale), -7, 7).astype(np.int8)
        scales[g] = scale

    return quantized, scales


def group_dequantize_int4(quantized, scales, group_size=32):
    n = len(quantized)
    n_groups = n // group_size
    recovered = np.zeros(n, dtype=np.float32)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        # TODO: dequantize this group using scales[g]
        recovered[start:end] = None   # Hint: quantized[start:end].astype(np.float32) * scales[g]

    return recovered


# Test with 128 weights
np.random.seed(99)
large_weights = np.random.randn(128).astype(np.float32)

q_global, s_global = global_quantize_int4(large_weights)
r_global = global_dequantize_int4(q_global, s_global)

q_group, s_group = group_quantize_int4(large_weights, group_size=32)
r_group = group_dequantize_int4(q_group, s_group, group_size=32)

print(f"\n128 weights, INT4 quantization:")

if r_global is not None:
    err_global = np.mean(np.abs(large_weights - r_global))
    print(f"Global scale (1 scale):        mean error = {err_global:.6f}")

if r_group is not None:
    err_group = np.mean(np.abs(large_weights - r_group))
    print(f"Group scale  (4 groups of 32): mean error = {err_group:.6f}")
    if r_global is not None and err_group > 0:
        print(f"Group quantization is {err_global/err_group:.1f}x more accurate!")

# =============================================================================
# EXPECTED OUTPUT (check your answers)
# =============================================================================

print("\n" + "=" * 60)
print("Expected Output Summary")
print("=" * 60)
print("Exercise 1 (model sizes):")
print("  117,000,000 params x fp32 = 0.43 GB")
print("  7,000,000,000 params x fp32 = 26.08 GB")
print("  7,000,000,000 params x int8 =  6.52 GB")
print("  7,000,000,000 params x int4 =  3.26 GB")
print("  70,000,000,000 params x int4 = 32.62 GB")
print("")
print("Exercise 2 (INT8 symmetric):")
print("  Scale factor ~ 0.02 (varies with random data)")
print("  Max error < scale/2")
print("")
print("Exercise 3 (asymmetric):")
print("  Zero point > 0 (because all values are positive)")
print("  Max error near 0 (255 buckets covers range well)")
print("")
print("Exercise 4 (group quantization):")
print("  Group quantization error should be 2-5x smaller than global")

# =============================================================================
# HINTS
# =============================================================================

print("\n" + "=" * 60)
print("HINTS (try to solve before reading!)")
print("=" * 60)
print("""
Exercise 1:
  bytes_per_param = {
      "fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5
  }
  total_bytes = num_params * bytes_per_param[dtype]
  size_gb = total_bytes / (1024 ** 3)

Exercise 2:
  max_abs = np.max(np.abs(weights_fp32))
  scale = max_abs / 127.0
  quantized = np.round(weights_fp32 / scale).clip(-127, 127).astype(np.int8)
  # dequantize:
  return quantized.astype(np.float32) * scale

Exercise 3:
  min_val = np.min(values_fp32)
  max_val = np.max(values_fp32)
  scale = (max_val - min_val) / 255.0
  zero_point = int(np.clip(np.round(-min_val / scale), 0, 255))
  quantized = np.clip(np.round(values_fp32 / scale) + zero_point, 0, 255).astype(np.uint8)
  # dequantize:
  return (quantized.astype(np.float32) - zero_point) * scale

Exercise 4 (group quant):
  max_abs = np.max(np.abs(group))
  scale = max_abs / 7.0
  quantized[start:end] = np.clip(np.round(group / scale), -7, 7).astype(np.int8)
  # dequantize:
  recovered[start:end] = quantized[start:end].astype(np.float32) * scales[g]
""")

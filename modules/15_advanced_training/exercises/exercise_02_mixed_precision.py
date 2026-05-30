"""
Exercise 02: Mixed Precision — Quantize, Dequantize, Measure Error
Module 15: Advanced LLM Training

TASKS:
  1. Implement simulate_fp16() — clamp values to fp16 range, check overflow
  2. Implement quantize_int8() — quantize float array to int8 range [-127, 127]
  3. Implement dequantize_int8() — reconstruct floats from int8 + scale
  4. Implement memory_savings() — compute memory reduction ratio
  5. Measure quantization error for different number formats

Run:  python exercise_02_mixed_precision.py
Deps: numpy
"""

import numpy as np


# ─────────────────────────────────────────────────────────
# TASK 1: Simulate fp16
# ─────────────────────────────────────────────────────────

def simulate_fp16(values: np.ndarray) -> np.ndarray:
    """
    Simulate converting a float32 array to fp16.

    RULES:
      - fp16 max value is 65504
      - Values > 65504 become +inf (overflow)
      - Values < -65504 become -inf (overflow)
      - Otherwise: cast to float16, then back to float32

    Args:
        values: float32 numpy array

    Returns:
        float32 array with fp16 precision (and potential inf for overflow)

    HINT:
      1. Cast array to np.float16
      2. Cast back to np.float32
      3. Overflow values will automatically become inf
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Quantize to INT8
# ─────────────────────────────────────────────────────────

def quantize_int8(values: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Quantize float32 array to INT8 range [-127, 127].

    Steps:
      1. Find the maximum absolute value in the array
      2. Compute scale = max_abs / 127.0
      3. Divide values by scale → scaled values in [-127, 127]
      4. Round to nearest integer
      5. Clip to [-127, 127]
      6. Cast to int8

    Args:
        values: float32 numpy array

    Returns:
        Tuple of (int8_array, scale_factor)
        scale_factor is needed to dequantize

    HINT:
        max_abs = np.max(np.abs(values))
        scale = max_abs / 127.0
        quantized = np.round(values / scale).clip(-127, 127).astype(np.int8)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Dequantize from INT8
# ─────────────────────────────────────────────────────────

def dequantize_int8(quantized: np.ndarray, scale: float) -> np.ndarray:
    """
    Reconstruct float32 values from INT8 quantization.

    Args:
        quantized: int8 numpy array (from quantize_int8)
        scale:     scale factor (from quantize_int8)

    Returns:
        float32 array approximating original values

    HINT: multiply quantized values by scale
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Memory Savings
# ─────────────────────────────────────────────────────────

def memory_savings(num_params: int, dtype_from: str, dtype_to: str) -> dict:
    """
    Calculate memory usage and savings when switching dtypes.

    Byte sizes:
      fp32  = 4 bytes
      fp16  = 2 bytes
      bf16  = 2 bytes
      int8  = 1 byte

    Args:
        num_params: number of model parameters
        dtype_from: original dtype ("fp32", "fp16", "bf16", "int8")
        dtype_to:   target dtype

    Returns:
        Dict with keys:
          "bytes_from":    bytes in original dtype
          "bytes_to":      bytes in target dtype
          "savings_ratio": how many times smaller (bytes_from / bytes_to)
          "savings_pct":   percentage saved

    Example:
        memory_savings(7e9, "fp32", "int8") → savings_ratio = 4.0
    """
    dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}

    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 02: Mixed Precision")
    print("=" * 55)

    # Test 1: simulate_fp16
    print("\n--- Test 1: simulate_fp16 ---")
    test_vals = np.array([0.1, 1000.0, 65504.0, 65505.0, -70000.0], dtype=np.float32)
    result = simulate_fp16(test_vals)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        print(f"  Input:  {test_vals}")
        print(f"  fp16:   {result}")
        overflow_detected = np.isinf(result[-1]) and np.isinf(result[-2])
        if overflow_detected:
            print(f"  PASS  overflow correctly detected")
        else:
            print(f"  FAIL  expected inf for values > 65504, got {result[-2:]}")

    # Test 2: quantize_int8
    print("\n--- Test 2: quantize_int8 ---")
    weights = np.array([1.5, -2.0, 0.5, -0.25, 3.0, -3.0], dtype=np.float32)
    result = quantize_int8(weights)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        quantized, scale = result
        if quantized.dtype == np.int8:
            print(f"  PASS  dtype = int8")
            print(f"  Input:     {weights}")
            print(f"  INT8:      {quantized}")
            print(f"  Scale:     {scale:.6f}")
        else:
            print(f"  FAIL  expected int8, got {quantized.dtype}")

    # Test 3: dequantize_int8
    print("\n--- Test 3: dequantize_int8 ---")
    if result is not None:
        quantized, scale = result
        reconstructed = dequantize_int8(quantized, scale)
        if reconstructed is None:
            print("  NOT IMPLEMENTED YET")
        else:
            max_error = np.max(np.abs(weights - reconstructed))
            max_value = np.max(np.abs(weights))
            rel_error = max_error / max_value * 100
            print(f"  Original:     {weights}")
            print(f"  Reconstructed:{reconstructed}")
            print(f"  Max abs error:  {max_error:.4f}")
            print(f"  Max rel error:  {rel_error:.2f}%")
            if rel_error < 5.0:
                print(f"  PASS  error within acceptable INT8 range")
            else:
                print(f"  FAIL  error too large: {rel_error:.2f}%")

    # Test 4: memory_savings
    print("\n--- Test 4: memory_savings ---")
    result = memory_savings(7_000_000_000, "fp32", "int8")
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        if abs(result["savings_ratio"] - 4.0) < 0.01:
            print(f"  PASS  fp32 → int8 savings = {result['savings_ratio']:.1f}x")
        else:
            print(f"  FAIL  expected 4.0x, got {result['savings_ratio']:.2f}x")
        print(f"  fp32 size:  {result['bytes_from'] / 1e9:.1f} GB")
        print(f"  int8 size:  {result['bytes_to']   / 1e9:.1f} GB")

    # BONUS: Error comparison across formats
    print("\n--- BONUS: Quantization Error Comparison ---")
    test_fn = quantize_int8
    if test_fn is not None and quantize_int8(np.array([1.0], dtype=np.float32)) is not None:
        original = np.random.RandomState(42).randn(1000).astype(np.float32)

        # fp16 error
        fp16_recon = simulate_fp16(original) if simulate_fp16(original) is not None else None
        if fp16_recon is not None:
            fp16_err = np.mean(np.abs(original - fp16_recon))
        else:
            fp16_err = float('nan')

        # int8 error
        q8, s8 = quantize_int8(original)
        recon8 = dequantize_int8(q8, s8)
        int8_err = np.mean(np.abs(original - recon8)) if recon8 is not None else float('nan')

        print(f"\n  Format    Bytes/param   Mean Abs Error")
        print(f"  {'fp32':<10} {'4':>10}   0.000000 (reference)")
        print(f"  {'fp16':<10} {'2':>10}   {fp16_err:.6f}")
        print(f"  {'int8':<10} {'1':>10}   {int8_err:.6f}")
        print(f"\n  LESSON: fp16 has tiny error (10 mantissa bits).")
        print(f"          INT8 has more error (127 discrete levels).")
        print(f"          Both are acceptable for inference, fp16/bf16 for training.")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def simulate_fp16(values):
#     return values.astype(np.float16).astype(np.float32)
#
# def quantize_int8(values):
#     max_abs = np.max(np.abs(values))
#     scale = max_abs / 127.0
#     quantized = np.round(values / scale).clip(-127, 127).astype(np.int8)
#     return quantized, scale
#
# def dequantize_int8(quantized, scale):
#     return quantized.astype(np.float32) * scale
#
# def memory_savings(num_params, dtype_from, dtype_to):
#     dtype_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}
#     bytes_from = num_params * dtype_bytes[dtype_from]
#     bytes_to   = num_params * dtype_bytes[dtype_to]
#     ratio = bytes_from / bytes_to
#     return {
#         "bytes_from":    bytes_from,
#         "bytes_to":      bytes_to,
#         "savings_ratio": ratio,
#         "savings_pct":   (1 - bytes_to / bytes_from) * 100,
#     }

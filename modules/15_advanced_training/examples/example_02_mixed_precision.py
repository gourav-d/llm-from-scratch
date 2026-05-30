"""
Example 02: Mixed Precision Training — Simulation in NumPy
Module 15: Advanced LLM Training

Demonstrates:
  - Number representation in fp32, fp16, bf16
  - Overflow and underflow behavior
  - Mixed precision training loop simulation
  - Memory savings comparison

Run:  python example_02_mixed_precision.py
Deps: numpy
"""

import numpy as np
import struct


# ─────────────────────────────────────────────────────────
# BIT-LEVEL INSPECTION HELPERS
# ─────────────────────────────────────────────────────────

def float32_to_bits(value: float) -> str:
    """Show bit representation of a 32-bit float."""
    # Pack float into 4 bytes, then unpack as uint32
    packed = struct.pack('f', value)
    bits = int.from_bytes(packed, 'little')
    # Format as binary: 1 sign + 8 exponent + 23 mantissa
    bit_str = f'{bits:032b}'
    return f"{bit_str[0]} | {bit_str[1:9]} | {bit_str[9:]}"


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ─────────────────────────────────────────────────────────
# DEMO 1: Number Format Properties
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: Number Format Properties")

formats = {
    "float32 (fp32)": np.float32,
    "float16 (fp16)": np.float16,
    "bfloat16 (bf16)": np.float32,   # NumPy has no native bf16; we'll simulate
}

print("\nfp32  : 1 sign + 8 exponent + 23 mantissa = 32 bits = 4 bytes")
print("fp16  : 1 sign + 5 exponent + 10 mantissa = 16 bits = 2 bytes")
print("bf16  : 1 sign + 8 exponent + 7  mantissa = 16 bits = 2 bytes")

print("\n--- fp32 max and min ---")
print(f"  Max value:  {np.finfo(np.float32).max:.3e}")
print(f"  Min value:  {np.finfo(np.float32).tiny:.3e}")
print(f"  Epsilon:    {np.finfo(np.float32).eps:.3e}")

print("\n--- fp16 max and min ---")
print(f"  Max value:  {np.finfo(np.float16).max:.3e}  ← only 65,504!")
print(f"  Min value:  {np.finfo(np.float16).tiny:.3e}")
print(f"  Epsilon:    {np.finfo(np.float16).eps:.3e}")

print("\n--- bf16 (same exponent as fp32, 7-bit mantissa) ---")
print(f"  Max value:  ~3.38e+38  (same as fp32 — 8-bit exponent)")
print(f"  Min value:  ~1.17e-38  (same as fp32)")
print(f"  Epsilon:    ~7.81e-03  (less precise than fp32 due to 7 mantissa bits)")


# ─────────────────────────────────────────────────────────
# DEMO 2: fp16 Overflow
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: fp16 Overflow Demonstration")

test_values = [0.001, 100.0, 1000.0, 10000.0, 65504.0, 65505.0, 100000.0]

print(f"\n{'Value (fp32)':>15} {'As fp16':>15} {'Overflow?':>12}")
print("-" * 45)

for v in test_values:
    fp32_val = np.float32(v)
    fp16_val = np.float16(v)
    overflow = np.isinf(fp16_val) or (float(fp16_val) == 0 and v != 0)
    status = "OVERFLOW → inf" if np.isinf(fp16_val) else ("ok" if not overflow else "underflow → 0")
    print(f"{v:>15.1f} {float(fp16_val):>15.1f} {status:>12}")

print("""
LESSON: fp16 overflows to inf for values > 65504.
        During training, gradients can easily exceed this.
        bf16 avoids this: same exponent range as fp32.
""")


# ─────────────────────────────────────────────────────────
# DEMO 3: bf16 Simulation
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: bf16 Simulation (truncate fp32 mantissa to 7 bits)")

def to_bf16(x: np.ndarray) -> np.ndarray:
    """
    Simulate bf16 by zeroing the lower 16 bits of fp32.
    bf16 = upper 16 bits of fp32 (same sign + exponent, truncated mantissa).
    """
    x32 = np.float32(x)
    # View float32 as uint32, zero lower 16 bits, cast back
    packed = x32.view(np.uint32)
    truncated = (packed & np.uint32(0xFFFF0000)).view(np.float32)
    return truncated


print("\nComparing fp32 vs bf16 precision:")
test_vals = [1.0, 0.1, 3.14159, 1.23456789, 100.5678]
print(f"\n{'Original fp32':>18} {'bf16 sim':>18} {'Error':>12}")
print("-" * 52)
for v in test_vals:
    fp32 = np.float32(v)
    bf16_sim = to_bf16(fp32)
    error = abs(float(fp32) - float(bf16_sim)) / abs(float(fp32)) * 100
    print(f"{float(fp32):>18.8f} {float(bf16_sim):>18.6f} {error:>11.4f}%")

print("""
LESSON: bf16 has less precision (~2-3 decimal places) but same range.
        For training, range matters more than precision.
        Gradient updates don't need 7 decimal places of accuracy.
""")


# ─────────────────────────────────────────────────────────
# DEMO 4: Memory Usage Comparison
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: Memory Usage for Different Model Sizes")

param_counts = {
    "125M":    125_000_000,
    "1.3B":  1_300_000_000,
    "7B":    7_000_000_000,
    "13B":  13_000_000_000,
    "70B":  70_000_000_000,
}

print(f"\n{'Model':>8} {'fp32 (GB)':>12} {'fp16 (GB)':>12} {'bf16 (GB)':>12} {'INT8 (GB)':>12}")
print("-" * 60)

for name, n_params in param_counts.items():
    fp32_gb = (n_params * 4) / 1e9
    fp16_gb = (n_params * 2) / 1e9
    bf16_gb = (n_params * 2) / 1e9     # same as fp16
    int8_gb = (n_params * 1) / 1e9
    print(f"{name:>8} {fp32_gb:>12.1f} {fp16_gb:>12.1f} {bf16_gb:>12.1f} {int8_gb:>12.1f}")

print("\nNote: Training with Adam also needs 2× more (m and v optimizer state)")
print("Mixed precision training (weights in fp16 + master in fp32):")
print("  weights_fp16 + weights_fp32 + adam_fp32 = 2 + 4 + 8 = 14 bytes/param")


# ─────────────────────────────────────────────────────────
# DEMO 5: Loss Scaling for fp16
# ─────────────────────────────────────────────────────────

print_section("DEMO 5: Loss Scaling — Preventing fp16 Gradient Underflow")

def simulate_loss_scaling(loss_value: float, scale_factor: float = 1024.0):
    """
    Simulate the loss scaling technique for fp16 training.
    Shows how very small gradients survive when loss is scaled.
    """
    # Simulate very small gradient that would underflow in fp16
    true_gradient = loss_value * 0.000001   # tiny gradient

    # Without scaling: check if gradient survives fp16
    fp16_grad_unscaled = np.float16(true_gradient)
    survived_unscaled = float(fp16_grad_unscaled) != 0.0

    # With scaling: scale loss first
    scaled_loss = loss_value * scale_factor
    scaled_gradient = scaled_loss * 0.000001   # same ratio
    fp16_grad_scaled = np.float16(scaled_gradient)
    fp16_grad_recovered = float(fp16_grad_scaled) / scale_factor
    survived_scaled = float(fp16_grad_scaled) != 0.0

    return {
        "true_gradient": true_gradient,
        "fp16_unscaled": float(fp16_grad_unscaled),
        "survived_unscaled": survived_unscaled,
        "fp16_scaled_then_unscaled": fp16_grad_recovered,
        "survived_scaled": survived_scaled,
    }


print("\nSimulating gradient underflow and recovery via loss scaling:")
print(f"\n{'Loss Value':>12} {'True Grad':>16} {'fp16 (no scale)':>18} {'fp16 (scaled)':>16} {'Survived?':>10}")
print("-" * 80)

for loss in [0.5, 0.1, 0.01, 0.001, 0.0001]:
    result = simulate_loss_scaling(loss, scale_factor=2048.0)
    survived = "YES" if result["survived_scaled"] else "NO"
    print(
        f"{loss:>12.4f} "
        f"{result['true_gradient']:>16.2e} "
        f"{result['fp16_unscaled']:>18.2e} "
        f"{result['fp16_scaled_then_unscaled']:>16.2e} "
        f"{survived:>10}"
    )

print("""
LESSON: Without scaling, tiny gradients become zero in fp16 (underflow).
        Scaling: multiply loss by 2048 → gradient × 2048 → fits in fp16.
        After backward: divide gradients by 2048 → recover true gradient.
        PyTorch GradScaler does this automatically.
        bf16 does NOT need this (same exponent range as fp32).
""")


# ─────────────────────────────────────────────────────────
# DEMO 6: Simulate a Mixed Precision Training Step
# ─────────────────────────────────────────────────────────

print_section("DEMO 6: Mixed Precision Training Step (Simulated)")

np.random.seed(42)

# Simulate a tiny linear layer: y = W @ x
# weights in fp32 (master copy), compute in fp16
n_in, n_out = 8, 4

W_fp32 = np.random.randn(n_out, n_in).astype(np.float32) * 0.1   # master weights
x_fp32 = np.random.randn(n_in).astype(np.float32)

print("\nSimulating one forward/backward step:")
print(f"\n  W shape: {W_fp32.shape}, x shape: {x_fp32.shape}")

# Step 1: Cast to fp16 for compute
W_fp16 = W_fp32.astype(np.float16)
x_fp16 = x_fp32.astype(np.float16)

# Step 2: Forward pass in fp16
y_fp16 = W_fp16 @ x_fp16

# Step 3: Simulate loss (mean squared output as toy loss)
target_fp16 = np.zeros(n_out, dtype=np.float16)
loss_fp16 = np.float16(np.mean((y_fp16 - target_fp16) ** 2))

print(f"  Forward pass (fp16):  y = {y_fp16}")
print(f"  Loss (fp16):          {float(loss_fp16):.6f}")

# Step 4: Gradient (in fp16) — simplified: dL/dW = (y - target) @ x.T
dL_dW_fp16 = np.outer((y_fp16 - target_fp16), x_fp16).astype(np.float16)

# Step 5: Cast gradient back to fp32 for optimizer
dL_dW_fp32 = dL_dW_fp16.astype(np.float32)

# Step 6: Update master weights in fp32
lr = 0.01
W_fp32 = W_fp32 - lr * dL_dW_fp32

print(f"  Gradient (fp32):      {dL_dW_fp32.flatten()[:4]} ...")
print(f"  Updated W (fp32):     {W_fp32.flatten()[:4]} ...")

print("""
FLOW SUMMARY:
  [fp32 master weights]
        ↓ cast to fp16
  [fp16 compute: forward + backward]
        ↓ cast gradient back to fp32
  [fp32 optimizer update]
        ↓ stores updated fp32 weights
  [fp32 master weights]

This is exactly what PyTorch AMP + GradScaler does automatically.
""")

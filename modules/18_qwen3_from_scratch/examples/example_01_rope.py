"""
Example 01: Rotary Position Embeddings (RoPE)
Module 18: Qwen3.5 LLM from Scratch

Run:  python example_01_rope.py
Deps: none (pure Python)
"""

import math

print("=" * 60)
print("  Example 01: Rotary Position Embeddings (RoPE)")
print("=" * 60)


# ─────────────────────────────────────────────────────────
# DEMO 1: Theta Frequencies — How Fast Each Dimension Rotates
# ─────────────────────────────────────────────────────────

print("\n--- DEMO 1: Theta Frequency Computation ---")
print()
print("  Formula: theta_i = 1 / (base ^ (2i / d))")
print("  Low i  = fast rotation (captures short-range patterns)")
print("  High i = slow rotation (captures long-range patterns)")
print()

def compute_thetas(d, base=10000):
    """Compute theta frequencies for all d/2 dimension pairs."""
    return [1.0 / (base ** (2 * i / d)) for i in range(d // 2)]

# Show for d=8 (easy to see), then d=128 (Qwen3.5 head_dim)
for d, base in [(8, 10000), (128, 10000), (128, 1_000_000)]:
    thetas = compute_thetas(d, base)
    print(f"  d={d}, base={base}:")
    print(f"    theta[0]     = {thetas[0]:.6f}  (fastest — rotates {thetas[0]:.4f} rad per position)")
    print(f"    theta[d/4]   = {thetas[d//4]:.6f}")
    print(f"    theta[d/2-1] = {thetas[-1]:.8f}  (slowest — one full rotation every {2*math.pi/thetas[-1]:.0f} positions)")
    print()

print("  Note: Qwen3.5 uses base=1,000,000 for 128K context.")
print("  Larger base = slower rotation = model distinguishes more positions.")


# ─────────────────────────────────────────────────────────
# DEMO 2: 2D Rotation — The Core Operation
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 2: 2D Rotation Formula ---")
print()
print("  RoPE rotates each (x, y) dimension pair by angle θ:")
print("  x' = x*cos(θ) - y*sin(θ)")
print("  y' = x*sin(θ) + y*cos(θ)")
print()
print("  Key property: rotation preserves vector LENGTH.")
print()

def rotate_pair(x, y, angle):
    """Rotate a 2D vector (x, y) by `angle` radians."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a

# Demonstrate rotation
examples = [
    ([1.0, 0.0], math.pi / 2,   "90°   rotation"),
    ([1.0, 0.0], math.pi,       "180° rotation"),
    ([0.5, 0.5], math.pi / 4,   "45°   rotation"),
    ([2.0, 1.0], 0.5,            "0.5 rad rotation"),
]
print(f"  {'Input':>14}  {'Angle':>8}  {'Output':>18}  {'|v| before':>11}  {'|v| after':>10}")
print("  " + "-" * 68)
for (x, y), angle, label in examples:
    xr, yr = rotate_pair(x, y, angle)
    len_before = math.sqrt(x**2 + y**2)
    len_after  = math.sqrt(xr**2 + yr**2)
    print(f"  ({x:.1f}, {y:.1f}) {label:>18}  {angle:>8.3f}  ({xr:>7.3f}, {yr:>7.3f})  "
          f"{len_before:>11.4f}  {len_after:>10.4f}")

print()
print("  |v| stays constant — rotation only changes direction, not magnitude.")


# ─────────────────────────────────────────────────────────
# DEMO 3: Full RoPE — Apply to Q and K, Verify Relative Position
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 3: Full RoPE on Q/K Vectors ---")
print()
print("  Key property: dot_product(Q_rotated@m, K_rotated@n)")
print("  depends ONLY on (m - n), not on m and n separately.")
print()

def apply_rope(vector, position, thetas):
    """
    Apply RoPE to a vector at a given position.
    vector: list of floats, length = 2 * len(thetas)
    """
    d = len(vector)
    rotated = list(vector)
    for i, theta in enumerate(thetas):
        angle = position * theta
        x = vector[2 * i]
        y = vector[2 * i + 1]
        rx, ry = rotate_pair(x, y, angle)
        rotated[2 * i]     = rx
        rotated[2 * i + 1] = ry
    return rotated

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# Demonstrate relative position property
# Same Q and K vectors, different absolute positions but same gap
d = 8
thetas = compute_thetas(d)

# Fixed Q and K base vectors
Q_base = [0.5, 0.3, 0.8, -0.2, 0.1, 0.6, -0.4, 0.7]
K_base = [0.2, 0.9, -0.3, 0.5, 0.7, -0.1, 0.4, 0.3]

# Test pairs: same gap (=3), different absolute positions
position_pairs = [
    (0, 3, "m=0, n=3  (gap=3)"),
    (5, 8, "m=5, n=8  (gap=3)"),
    (10, 13, "m=10, n=13 (gap=3)"),
    (0, 5, "m=0, n=5  (gap=5)"),
    (3, 8, "m=3, n=8  (gap=5)"),
]

print(f"  {'Pair':<22}  {'dot(Q@m, K@n)':>16}")
print("  " + "-" * 42)
for m, n, label in position_pairs:
    Q_rot = apply_rope(Q_base, m, thetas)
    K_rot = apply_rope(K_base, n, thetas)
    score = dot(Q_rot, K_rot)
    print(f"  {label:<22}  {score:>16.6f}")

print()
print("  Pairs with gap=3 get the SAME dot product regardless of absolute position.")
print("  Pairs with gap=5 get a different but consistent value.")
print("  This is the relative position property: score = f(m - n) only.")

# Compare sinusoidal (additive) vs RoPE
print()
print("\n  Comparison: Sinusoidal PE vs RoPE")
print(f"  {'Method':>15}  {'What it does':>40}  {'Relative?':>10}")
print("  " + "-" * 70)
methods = [
    ("Sinusoidal", "Add sin/cos vector to token embedding",            "Indirect"),
    ("RoPE",       "Rotate Q and K by angle = position * theta",       "YES, exact"),
    ("Learned PE", "Lookup table, one vector per position",            "No"),
]
for name, what, rel in methods:
    print(f"  {name:>15}  {what:>40}  {rel:>10}")

print()
print("  KEY TAKEAWAYS:")
print("  1. theta_i = 1/(base^(2i/d)) — low i fast, high i slow.")
print("  2. Each dim pair rotated by position * theta_i.")
print("  3. dot(Q@m, K@n) depends only on gap (m-n), not absolute positions.")
print("  4. Works for any position — no lookup table, no max context limit.")

"""
Exercise 01: Rotary Position Embeddings (RoPE)
Module 18: Qwen3.5 LLM from Scratch

TASKS:
  1. compute_theta_freqs(d, base)  — theta_i = 1/(base^(2i/d)) for all i
  2. rotate_pair(x, y, angle)     — rotate 2D vector by angle (return x', y')
  3. apply_rope(vector, position, thetas) — apply RoPE to full vector at position
  4. verify_relative_position(Q, K, thetas, m, n, m2, n2) — confirm same gap = same dot

Run:   python exercise_01_rope.py
Deps:  none (pure Python)
"""

import math


# ──────────────────────────────────────────────────────────
# TASK 1: Compute Theta Frequencies
# ──────────────────────────────────────────────────────────

def compute_theta_freqs(d, base=10000):
    """
    Compute RoPE theta frequencies for a head of dimension d.

    Args:
        d    (int):   head dimension (must be even)
        base (float): rotation base (default 10000; Qwen3.5 uses 1,000,000)

    Returns:
        list[float]: list of d//2 theta values, theta_i = 1/(base^(2i/d))

    Example:
        compute_theta_freqs(4, 10000)
        # theta[0] = 1.0, theta[1] ≈ 0.01

    HINT:
        Loop i in range(d // 2).
        theta_i = 1.0 / (base ** (2 * i / d))
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 2: Rotate a 2D Pair
# ──────────────────────────────────────────────────────────

def rotate_pair(x, y, angle):
    """
    Rotate 2D vector (x, y) by `angle` radians.

    Args:
        x     (float): x component
        y     (float): y component
        angle (float): rotation angle in radians

    Returns:
        tuple[float, float]: (x', y') after rotation

    Example:
        rotate_pair(1.0, 0.0, math.pi / 2)  # returns (~0.0, 1.0)

    HINT:
        x' = x * cos(angle) - y * sin(angle)
        y' = x * sin(angle) + y * cos(angle)
        Use math.cos and math.sin.
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 3: Apply RoPE to Full Vector
# ──────────────────────────────────────────────────────────

def apply_rope(vector, position, thetas):
    """
    Apply RoPE to a query or key vector at a given sequence position.

    Args:
        vector   (list[float]): input vector, length = 2 * len(thetas)
        position (int):         sequence position (0-indexed)
        thetas   (list[float]): theta frequencies from compute_theta_freqs

    Returns:
        list[float]: rotated vector, same length as input

    Example:
        thetas = compute_theta_freqs(4)
        v = [1.0, 0.0, 1.0, 0.0]
        apply_rope(v, 0, thetas)   # position 0 → no rotation (angle=0)
        apply_rope(v, 1, thetas)   # position 1 → rotated by thetas[i]

    HINT:
        For each pair i in range(len(thetas)):
            angle = position * thetas[i]
            x = vector[2*i],  y = vector[2*i + 1]
            rotate with rotate_pair(x, y, angle)
            write back to result list
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 4: Verify Relative Position Property
# ──────────────────────────────────────────────────────────

def verify_relative_position(Q_base, K_base, thetas, pairs):
    """
    Compute dot(RoPE(Q, m), RoPE(K, n)) for multiple (m, n) pairs.
    Returns scores to verify that same-gap pairs produce the same score.

    Args:
        Q_base (list[float]): base query vector
        K_base (list[float]): base key vector
        thetas (list[float]): theta frequencies
        pairs  (list[tuple]): list of (m, n, label) tuples

    Returns:
        list[tuple]: [(label, score), ...] for each pair

    HINT:
        For each (m, n, label):
            Q_rot = apply_rope(Q_base, m, thetas)
            K_rot = apply_rope(K_base, n, thetas)
            score = sum of element-wise products (dot product)
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 01: RoPE — Test Results")
    print("=" * 55)
    passed = 0
    failed = 0

    # --- Test 1: compute_theta_freqs ---
    print("\n[Task 1] compute_theta_freqs")
    try:
        thetas_4 = compute_theta_freqs(4, 10000)
        assert thetas_4 is not None, "Returned None"
        assert len(thetas_4) == 2, f"Expected length 2, got {len(thetas_4)}"
        assert abs(thetas_4[0] - 1.0) < 1e-9, f"theta[0] should be 1.0, got {thetas_4[0]}"
        assert abs(thetas_4[1] - 0.01) < 1e-9, f"theta[1] should be 0.01, got {thetas_4[1]}"

        thetas_8 = compute_theta_freqs(8)
        assert len(thetas_8) == 4, f"Expected 4 for d=8, got {len(thetas_8)}"
        assert thetas_8[0] >= thetas_8[-1], "Thetas should decrease with i"
        print("  PASS: length correct, theta[0]=1.0, theta[1]=0.01, decreasing")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 2: rotate_pair ---
    print("\n[Task 2] rotate_pair")
    try:
        # 90 degree rotation: (1,0) → (0,1)
        xr, yr = rotate_pair(1.0, 0.0, math.pi / 2)
        assert xr is not None and yr is not None, "Returned None"
        assert abs(xr - 0.0) < 1e-9, f"x' should be ~0.0, got {xr}"
        assert abs(yr - 1.0) < 1e-9, f"y' should be ~1.0, got {yr}"

        # 180 degree rotation: (1,0) → (-1,0)
        xr2, yr2 = rotate_pair(1.0, 0.0, math.pi)
        assert abs(xr2 - (-1.0)) < 1e-9, f"x' should be -1.0, got {xr2}"

        # Length should be preserved
        x, y = 3.0, 4.0
        xr3, yr3 = rotate_pair(x, y, 0.7)
        len_before = math.sqrt(x**2 + y**2)
        len_after  = math.sqrt(xr3**2 + yr3**2)
        assert abs(len_before - len_after) < 1e-9, "Rotation should preserve length"
        print("  PASS: 90deg correct, 180deg correct, length preserved")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 3: apply_rope ---
    print("\n[Task 3] apply_rope")
    try:
        thetas = compute_theta_freqs(4)
        if thetas is None:
            raise AssertionError("Need Task 1 to pass first")

        v = [1.0, 0.0, 0.5, 0.5]

        # Position 0: angle = 0 for all pairs → no rotation
        v0 = apply_rope(v, 0, thetas)
        assert v0 is not None, "Returned None"
        assert len(v0) == 4, f"Output length should be 4, got {len(v0)}"
        assert abs(v0[0] - 1.0) < 1e-9, "Position 0: first element unchanged"
        assert abs(v0[1] - 0.0) < 1e-9, "Position 0: second element unchanged"

        # Position 1: should rotate
        v1 = apply_rope(v, 1, thetas)
        assert v1 != v0, "Position 1 should differ from position 0"

        # Length should be preserved (rotation doesn't change magnitude)
        len0 = math.sqrt(sum(x**2 for x in v))
        len1 = math.sqrt(sum(x**2 for x in v1))
        assert abs(len0 - len1) < 1e-9, "RoPE should preserve vector length"
        print("  PASS: position 0 = identity, position 1 differs, length preserved")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 4: verify_relative_position ---
    print("\n[Task 4] verify_relative_position")
    try:
        thetas = compute_theta_freqs(8)
        if thetas is None:
            raise AssertionError("Need Task 1 to pass first")

        Q_base = [0.5, 0.3, 0.8, -0.2, 0.1, 0.6, -0.4, 0.7]
        K_base = [0.2, 0.9, -0.3, 0.5, 0.7, -0.1, 0.4, 0.3]

        pairs = [
            (0, 3,  "gap3_a"),
            (5, 8,  "gap3_b"),
            (10, 13,"gap3_c"),
            (0, 5,  "gap5_a"),
            (3, 8,  "gap5_b"),
        ]

        result = verify_relative_position(Q_base, K_base, thetas, pairs)
        assert result is not None, "Returned None"
        assert len(result) == 5, f"Expected 5 results, got {len(result)}"

        scores = {label: score for label, score in result}

        # Same gap → same score
        assert abs(scores["gap3_a"] - scores["gap3_b"]) < 1e-9, \
            f"gap=3 scores should match: {scores['gap3_a']:.6f} vs {scores['gap3_b']:.6f}"
        assert abs(scores["gap3_b"] - scores["gap3_c"]) < 1e-9, \
            f"gap=3 scores should match: {scores['gap3_b']:.6f} vs {scores['gap3_c']:.6f}"
        assert abs(scores["gap5_a"] - scores["gap5_b"]) < 1e-9, \
            f"gap=5 scores should match: {scores['gap5_a']:.6f} vs {scores['gap5_b']:.6f}"

        # Different gaps → different scores
        assert abs(scores["gap3_a"] - scores["gap5_a"]) > 1e-6, \
            "gap=3 and gap=5 scores should differ"

        print(f"  PASS: gap=3 score={scores['gap3_a']:.6f} (all match), "
              f"gap=5 score={scores['gap5_a']:.6f} (different)")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # Summary
    print()
    print("=" * 55)
    print(f"  RESULTS: {passed} passed, {failed} failed out of 4 tasks")
    print("=" * 55)
    if failed == 0:
        print("  All tasks complete! RoPE implementation correct.")
    else:
        print("  Fix failing tasks above, then re-run.")


if __name__ == "__main__":
    test_all()


# ──────────────────────────────────────────────────────────
# SOLUTION (read after attempting!)
# ──────────────────────────────────────────────────────────
#
# def compute_theta_freqs(d, base=10000):
#     return [1.0 / (base ** (2 * i / d)) for i in range(d // 2)]
#
# def rotate_pair(x, y, angle):
#     return x * math.cos(angle) - y * math.sin(angle), \
#            x * math.sin(angle) + y * math.cos(angle)
#
# def apply_rope(vector, position, thetas):
#     result = list(vector)
#     for i, theta in enumerate(thetas):
#         angle = position * theta
#         rx, ry = rotate_pair(vector[2*i], vector[2*i+1], angle)
#         result[2*i]   = rx
#         result[2*i+1] = ry
#     return result
#
# def verify_relative_position(Q_base, K_base, thetas, pairs):
#     results = []
#     for m, n, label in pairs:
#         Q_rot = apply_rope(Q_base, m, thetas)
#         K_rot = apply_rope(K_base, n, thetas)
#         score = sum(q * k for q, k in zip(Q_rot, K_rot))
#         results.append((label, score))
#     return results

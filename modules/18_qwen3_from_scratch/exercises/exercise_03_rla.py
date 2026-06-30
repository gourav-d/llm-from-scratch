"""
Exercise 03: Recurrent Linear Attention (RLA)
Module 18: Qwen3.5 LLM from Scratch

TASKS:
  1. phi(x_vec)                         — kernel function ELU(x)+1
  2. outer_product(k, v)                — k ⊗ v matrix
  3. update_hidden_state(S, k_phi, v)   — S = S + phi(k) ⊗ v
  4. rla_step(q_vec, S, z, eps)         — full query step → output

Run:   python exercise_03_rla.py
Deps:  none (pure Python)
"""

import math


# ──────────────────────────────────────────────────────────
# TASK 1: Phi Kernel Function
# ──────────────────────────────────────────────────────────

def phi(x_vec):
    """
    Apply the phi kernel function element-wise: phi(x) = ELU(x) + 1.
    ELU(x) = x if x > 0, else (exp(x) - 1)
    Adding 1 ensures the result is always >= 0.

    Args:
        x_vec (list[float]): input vector

    Returns:
        list[float]: phi applied to each element

    Example:
        phi([2.0, -1.0, 0.0])
        # ELU:       [2.0, exp(-1)-1, 0.0]  = [2.0, -0.6321, 0.0]
        # phi = +1:  [3.0,  0.3679,   1.0]

    HINT:
        For each x in x_vec:
            elu(x) = x if x > 0 else (math.exp(x) - 1)
            phi(x) = elu(x) + 1.0
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 2: Outer Product
# ──────────────────────────────────────────────────────────

def outer_product(k, v):
    """
    Compute the outer product k ⊗ v.
    Result is a 2D matrix where result[i][j] = k[i] * v[j].

    Args:
        k (list[float]): key vector, length d_k
        v (list[float]): value vector, length d_v

    Returns:
        list[list[float]]: 2D matrix of shape [d_k, d_v]

    Example:
        outer_product([1.0, 2.0], [3.0, 4.0])
        # [[3.0, 4.0],
        #  [6.0, 8.0]]

    HINT:
        Nested comprehension or nested loops:
        result[i][j] = k[i] * v[j]
        for i in range(len(k)):
            row = [k[i] * v[j] for j in range(len(v))]
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 3: Update Hidden State
# ──────────────────────────────────────────────────────────

def update_hidden_state(S, k_vec, v_vec):
    """
    Update the RLA hidden state: S_new = S + phi(k) ⊗ v.
    Also updates the normalizer z.

    Args:
        S     (list[list[float]]): current hidden state [d_k, d_v]
        k_vec (list[float]):       raw key vector (phi NOT yet applied)
        v_vec (list[float]):       value vector

    Returns:
        tuple: (S_new, z_new) where
            S_new (list[list[float]]): updated hidden state
            z_new (list[float]):       updated normalizer z = z + phi(k)
            BUT since z is not passed in, initialize z to all zeros inside this
            function and return just the contribution (phi(k)) to z.
            Actually — see HINT below for simpler design.

    Simplified version for this exercise:
        Accept S (the hidden state matrix) and return S_new.
        Also return the phi(k) vector so the caller can accumulate z.

    Returns:
        tuple: (S_new list[list[float]], phi_k list[float])

    HINT:
        1. k_phi = phi(k_vec)
        2. kv_outer = outer_product(k_phi, v_vec)
        3. S_new[i][j] = S[i][j] + kv_outer[i][j]  for all i, j
        4. Return (S_new, k_phi)
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 4: Full RLA Step (Query + Output)
# ──────────────────────────────────────────────────────────

def rla_step(q_vec, S, z, eps=1e-6):
    """
    Compute the RLA output for one query vector.

    Formula:
        phi_q     = phi(q_vec)
        numerator = phi_q @ S   (shape: [d_v])
        denom     = phi_q · z + eps
        output    = numerator / denom

    Args:
        q_vec (list[float]):       query vector, length d_k
        S     (list[list[float]]): hidden state [d_k, d_v]
        z     (list[float]):       normalizer vector, length d_k
        eps   (float):             small constant for numerical stability

    Returns:
        list[float]: output vector, length d_v

    Example (conceptually):
        After processing K tokens, S contains compressed memory.
        Query Q retrieves relevant info: output = phi(Q) @ S / (phi(Q) · z)

    HINT:
        1. q_phi = phi(q_vec)
        2. numerator[j] = sum(q_phi[i] * S[i][j] for i in range(d_k))
           This is the row-vector q_phi multiplied by matrix S.
        3. denom = sum(q_phi[i] * z[i] for i in range(d_k)) + eps
        4. output[j] = numerator[j] / denom
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 03: RLA — Test Results")
    print("=" * 55)
    passed = 0
    failed = 0

    # --- Test 1: phi ---
    print("\n[Task 1] phi")
    try:
        result = phi([2.0, -1.0, 0.0])
        assert result is not None, "Returned None"
        assert len(result) == 3, f"Expected length 3, got {len(result)}"

        # phi(2.0) = 2.0 + 1 = 3.0
        assert abs(result[0] - 3.0) < 1e-9, f"phi(2.0) should be 3.0, got {result[0]}"
        # phi(-1.0) = (exp(-1) - 1) + 1 = exp(-1) ≈ 0.3679
        expected_neg1 = math.exp(-1.0)
        assert abs(result[1] - expected_neg1) < 1e-9, \
            f"phi(-1.0) should be {expected_neg1:.4f}, got {result[1]}"
        # phi(0.0) = 0 + 1 = 1.0
        assert abs(result[2] - 1.0) < 1e-9, f"phi(0.0) should be 1.0, got {result[2]}"

        # All values must be non-negative
        test_vals = phi([-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
        assert all(v >= 0 for v in test_vals), f"phi must always be >= 0: {test_vals}"

        print(f"  PASS: phi(2.0)=3.0, phi(-1.0)={expected_neg1:.4f}, phi(0.0)=1.0, all >= 0")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 2: outer_product ---
    print("\n[Task 2] outer_product")
    try:
        result = outer_product([1.0, 2.0], [3.0, 4.0])
        assert result is not None, "Returned None"
        assert len(result) == 2,    f"Expected 2 rows, got {len(result)}"
        assert len(result[0]) == 2, f"Expected 2 cols, got {len(result[0])}"
        assert result[0][0] == 3.0, f"[0][0] should be 3.0, got {result[0][0]}"
        assert result[0][1] == 4.0, f"[0][1] should be 4.0, got {result[0][1]}"
        assert result[1][0] == 6.0, f"[1][0] should be 6.0, got {result[1][0]}"
        assert result[1][1] == 8.0, f"[1][1] should be 8.0, got {result[1][1]}"

        # Asymmetric example: k=[3], v=[1,2,3]
        r2 = outer_product([3.0], [1.0, 2.0, 3.0])
        assert len(r2) == 1 and len(r2[0]) == 3, "Shape should be [1,3]"
        assert r2[0] == [3.0, 6.0, 9.0], f"Wrong result: {r2[0]}"

        print("  PASS: [[3,4],[6,8]] correct, asymmetric shape correct")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 3: update_hidden_state ---
    print("\n[Task 3] update_hidden_state")
    try:
        # Start with zero S
        S = [[0.0, 0.0], [0.0, 0.0]]
        k_vec = [1.0, 0.0]
        v_vec = [2.0, 3.0]

        result = update_hidden_state(S, k_vec, v_vec)
        assert result is not None, "Returned None"
        S_new, phi_k = result

        # phi([1.0, 0.0]) = [2.0, 1.0]
        # outer([2.0, 1.0], [2.0, 3.0]) = [[4.0, 6.0], [2.0, 3.0]]
        assert len(S_new) == 2,    f"S_new should have 2 rows"
        assert len(S_new[0]) == 2, f"S_new should have 2 cols"
        assert abs(S_new[0][0] - 4.0) < 1e-9, f"S_new[0][0] should be 4.0, got {S_new[0][0]}"
        assert abs(S_new[1][1] - 3.0) < 1e-9, f"S_new[1][1] should be 3.0, got {S_new[1][1]}"

        # Check that original S is not mutated (or verify S_new is independent)
        assert abs(phi_k[0] - 2.0) < 1e-9, f"phi_k[0] should be 2.0 (phi(1.0)=1+1), got {phi_k[0]}"

        print(f"  PASS: S updated correctly, phi_k={[round(x,4) for x in phi_k]}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 4: rla_step ---
    print("\n[Task 4] rla_step")
    try:
        # Build a simple S and z by processing one token
        S0 = [[0.0, 0.0], [0.0, 0.0]]
        k0 = [1.0, 0.0]
        v0 = [1.0, 0.0]
        S1, phi_k0 = update_hidden_state(S0, k0, v0)
        if S1 is None:
            raise AssertionError("Need Task 3 to pass first")
        z1 = phi_k0  # z = sum of phi(k) for all processed tokens

        # Query with same vector as key → should retrieve v0
        q_vec = [1.0, 0.0]
        output = rla_step(q_vec, S1, z1)
        assert output is not None, "Returned None"
        assert len(output) == 2, f"Output length should be 2, got {len(output)}"

        # Output should be close to v0 = [1.0, 0.0]
        assert abs(output[0] - 1.0) < 1e-6, f"output[0] should be ~1.0, got {output[0]}"
        assert abs(output[1] - 0.0) < 1e-6, f"output[1] should be ~0.0, got {output[1]}"

        # eps prevents division by zero when z is all zeros
        S_empty = [[0.0, 0.0], [0.0, 0.0]]
        z_zero  = [0.0, 0.0]
        out_zero = rla_step(q_vec, S_empty, z_zero, eps=1e-6)
        assert out_zero is not None, "Should not crash with zero z"

        print(f"  PASS: output={[round(x,4) for x in output]} ≈ [1.0, 0.0], "
              "zero-z safe")
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
        print("  All tasks complete! RLA core operations working.")
    else:
        print("  Fix failing tasks above, then re-run.")

    # BONUS: Show full recurrent sequence
    if passed == 4:
        print()
        print("  BONUS: Full recurrent forward pass (5 tokens)")
        tokens = [[0.5, 0.3], [0.8, -0.2], [0.1, 0.6], [-0.4, 0.7], [0.3, 0.9]]
        S = [[0.0, 0.0], [0.0, 0.0]]
        z = [0.0, 0.0]
        for t, kv in enumerate(tokens):
            S, phi_k = update_hidden_state(S, kv, kv)  # K=V for demo
            z = [z[i] + phi_k[i] for i in range(len(z))]
            out = rla_step(kv, S, z)
            print(f"    Token {t}: out=({out[0]:.4f}, {out[1]:.4f})")
        print("  Hidden state stays 2×2 regardless of sequence length!")


if __name__ == "__main__":
    test_all()


# ──────────────────────────────────────────────────────────
# SOLUTION (read after attempting!)
# ──────────────────────────────────────────────────────────
#
# def phi(x_vec):
#     return [(x if x > 0 else math.exp(x) - 1) + 1.0 for x in x_vec]
#
# def outer_product(k, v):
#     return [[k[i] * v[j] for j in range(len(v))] for i in range(len(k))]
#
# def update_hidden_state(S, k_vec, v_vec):
#     k_phi = phi(k_vec)
#     outer = outer_product(k_phi, v_vec)
#     S_new = [[S[i][j] + outer[i][j] for j in range(len(S[0]))] for i in range(len(S))]
#     return S_new, k_phi
#
# def rla_step(q_vec, S, z, eps=1e-6):
#     q_phi = phi(q_vec)
#     d_k, d_v = len(S), len(S[0])
#     numerator = [sum(q_phi[i] * S[i][j] for i in range(d_k)) for j in range(d_v)]
#     denom = sum(q_phi[i] * z[i] for i in range(d_k)) + eps
#     return [n / denom for n in numerator]

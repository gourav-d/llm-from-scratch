"""
Exercise 04: KV Cache Management
Module 18: Qwen3.5 LLM from Scratch

TASKS:
  1. KVCache class       — append(k,v), size_bytes(bpv), attend(q)
  2. sliding_window_evict(K, V, window) — trim to last W entries
  3. int8_quantize(vector)              — scale + round to INT8

Run:   python exercise_04_kv_cache.py
Deps:  none (pure Python)
"""

import math


# ──────────────────────────────────────────────────────────
# TASK 1: KVCache Class
# ──────────────────────────────────────────────────────────

class KVCache:
    """
    Single-head KV cache for one transformer layer.
    Stores K and V vectors as tokens are generated.

    You must implement:
        append(k_new, v_new)     — add new K and V vectors
        size_bytes(bpv=2)        — total memory in bytes
        attend(q_vec)            — attention output for query q_vec

    HINT for attend:
        scale  = 1.0 / sqrt(head_dim)
        scores = [dot(q, k) * scale  for k in self.K]
        weights = softmax(scores)
        output[d] = sum(weights[t] * V[t][d]  for t in range(len(V)))
    """

    def __init__(self, head_dim):
        """
        Args:
            head_dim (int): dimension of each K and V vector
        """
        # TODO: store head_dim, initialize empty K and V lists
        pass

    def append(self, k_new, v_new):
        """
        Append a new key-value pair to the cache.

        Args:
            k_new (list[float]): key vector, length = head_dim
            v_new (list[float]): value vector, length = head_dim
        """
        # TODO: implement this
        pass

    def size_bytes(self, bpv=2):
        """
        Compute KV cache memory usage in bytes.

        Args:
            bpv (int): bytes per value (2=bf16, 1=INT8)

        Returns:
            int: 2 × seq_len × head_dim × bpv

        HINT: seq_len = len(self.K)
        """
        # TODO: implement this
        pass

    def attend(self, q_vec):
        """
        Compute attention output over all cached K, V for query q_vec.

        Args:
            q_vec (list[float]): query vector, length = head_dim

        Returns:
            list[float]: attention output, length = head_dim
                         Returns all zeros if cache is empty.

        HINT:
            1. scale = 1.0 / math.sqrt(head_dim)
            2. scores[t] = dot(q_vec, K[t]) * scale
            3. weights = softmax(scores)
            4. output[d] = sum(weights[t] * V[t][d] for t)
        """
        # TODO: implement this
        pass


# ──────────────────────────────────────────────────────────
# TASK 2: Sliding Window Eviction
# ──────────────────────────────────────────────────────────

def sliding_window_evict(K_list, V_list, window_size):
    """
    Evict old tokens from the KV cache — keep only the LAST window_size tokens.

    Args:
        K_list      (list): list of K vectors (one per token)
        V_list      (list): list of V vectors (one per token)
        window_size (int):  maximum number of tokens to keep

    Returns:
        tuple: (K_trimmed, V_trimmed) — last window_size entries
               If len(K_list) <= window_size, return unchanged.

    Example:
        K = [k0, k1, k2, k3, k4], window_size=3
        → returns [k2, k3, k4]

    HINT:
        Use Python list slicing: lst[-window_size:]
        Check if len(K_list) <= window_size first to avoid slicing issues.
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 3: INT8 Quantization
# ──────────────────────────────────────────────────────────

def int8_quantize(vector):
    """
    Quantize a float vector to INT8 (range -128 to 127) using symmetric scaling.

    Steps:
        1. Find max_abs = max absolute value in vector
        2. scale = max_abs / 127.0
        3. quantized[i] = round(vector[i] / scale), clamped to [-128, 127]

    Args:
        vector (list[float]): input float vector (bf16/f32)

    Returns:
        tuple: (quantized list[int], scale float)
               Use scale=1.0 and zeros if vector is all zeros.

    Example:
        int8_quantize([1.0, -0.5, 0.25])
        # max_abs = 1.0, scale = 1/127 ≈ 0.00787
        # quantized ≈ [127, -63 or -64, 32]

    HINT:
        max_abs = max(abs(x) for x in vector)
        scale   = max_abs / 127.0
        q = max(-128, min(127, round(x / scale)))
        Use round() not int() to avoid rounding bias.
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────────

def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def _softmax(scores):
    max_s = max(scores)
    exps  = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def test_all():
    print("=" * 55)
    print("  Exercise 04: KV Cache — Test Results")
    print("=" * 55)
    passed = 0
    failed = 0

    import random
    rng = random.Random(42)
    rand_vec = lambda d: [round(rng.uniform(-1, 1), 3) for _ in range(d)]

    # --- Test 1: KVCache class ---
    print("\n[Task 1] KVCache class")
    try:
        head_dim = 4
        cache = KVCache(head_dim)
        assert cache is not None, "KVCache() returned None"

        # Empty cache
        assert cache.size_bytes() == 0, f"Empty cache should be 0 bytes, got {cache.size_bytes()}"
        out0 = cache.attend(rand_vec(head_dim))
        assert out0 is not None, "attend on empty cache returned None"
        assert len(out0) == head_dim, f"attend output length should be {head_dim}"
        assert all(x == 0.0 for x in out0), "Empty cache attend should return zeros"

        # Append one token
        k1 = rand_vec(head_dim)
        v1 = rand_vec(head_dim)
        cache.append(k1, v1)
        assert cache.size_bytes(bpv=2) == 2 * 1 * head_dim * 2, \
            f"1 token size_bytes wrong: {cache.size_bytes()}"

        # Append more and check size
        for _ in range(4):
            cache.append(rand_vec(head_dim), rand_vec(head_dim))
        expected_bytes = 2 * 5 * head_dim * 2
        assert cache.size_bytes() == expected_bytes, \
            f"5 tokens: expected {expected_bytes} bytes, got {cache.size_bytes()}"

        # INT8 size should be half
        assert cache.size_bytes(bpv=1) == expected_bytes // 2, "INT8 should be half"

        # attend should return head_dim-length vector
        q = rand_vec(head_dim)
        out = cache.attend(q)
        assert len(out) == head_dim, f"attend output should be length {head_dim}"
        assert any(x != 0.0 for x in out), "attend output should not be all zeros"

        print(f"  PASS: empty=0bytes, 5 tokens={expected_bytes}B, attend works")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 2: sliding_window_evict ---
    print("\n[Task 2] sliding_window_evict")
    try:
        K = list(range(10))  # [0, 1, 2, ..., 9]
        V = list(range(10))

        # Evict to window=4
        K4, V4 = sliding_window_evict(K, V, 4)
        assert K4 is not None, "Returned None"
        assert len(K4) == 4, f"Expected 4, got {len(K4)}"
        assert K4 == [6, 7, 8, 9], f"Expected [6,7,8,9], got {K4}"

        # No eviction needed (exactly window size)
        K4b, V4b = sliding_window_evict(K[:4], V[:4], 4)
        assert K4b == [0, 1, 2, 3], f"Expected no change: {K4b}"

        # Smaller than window
        K2, V2 = sliding_window_evict([0, 1], [0, 1], 10)
        assert K2 == [0, 1], f"Smaller than window: {K2}"

        # Window=1 (keep only latest)
        K1, V1 = sliding_window_evict(K, V, 1)
        assert K1 == [9] and V1 == [9], f"Window=1: expected [9], got {K1}"

        print("  PASS: eviction to 4, exact fit, below window, window=1 all correct")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 3: int8_quantize ---
    print("\n[Task 3] int8_quantize")
    try:
        # Simple case
        result = int8_quantize([1.0, -1.0, 0.5])
        assert result is not None, "Returned None"
        q, scale = result
        assert scale is not None, "scale is None"
        assert abs(scale - 1.0 / 127.0) < 1e-9, f"scale should be 1/127, got {scale}"
        assert q[0] == 127,  f"1.0 → 127, got {q[0]}"
        assert q[1] == -127, f"-1.0 → -127, got {q[1]}"
        assert q[2] in [63, 64], f"0.5 → 63 or 64, got {q[2]}"

        # All values in [-128, 127]
        big_vec = [100.0, -50.0, 0.0, 200.0, -200.0]
        q2, s2 = int8_quantize(big_vec)
        assert all(-128 <= x <= 127 for x in q2), f"Values out of INT8 range: {q2}"

        # All zeros case
        q3, s3 = int8_quantize([0.0, 0.0, 0.0])
        assert q3 == [0, 0, 0], f"All-zero input: {q3}"

        # Check quantization error is small
        test_vec = [0.847, -0.234, 1.562, -1.023]
        q4, s4 = int8_quantize(test_vec)
        restored = [qi * s4 for qi in q4]
        max_err = max(abs(test_vec[i] - restored[i]) for i in range(len(test_vec)))
        assert max_err < 0.02, f"Quantization error too large: {max_err:.5f}"

        print(f"  PASS: [127,-127,63/64] correct, clamped, zero-safe, max_err={max_err:.5f}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # Summary
    print()
    print("=" * 55)
    print(f"  RESULTS: {passed} passed, {failed} failed out of 3 tasks")
    print("=" * 55)
    if failed == 0:
        print("  All tasks complete! KV cache management working.")
    else:
        print("  Fix failing tasks above, then re-run.")

    # BONUS: Simulate growing then evicting cache
    if passed == 3:
        print()
        print("  BONUS: Sliding window cache simulation")
        window = 4
        k_list, v_list = [], []
        print(f"  Window={window}")
        print(f"  {'Step':>5}  {'Total seen':>12}  {'In cache':>10}  {'Memory':>10}")
        rng2 = random.Random(7)
        hd = 4
        for step in range(8):
            k_list.append(rng2.uniform(-1, 1))
            v_list.append(rng2.uniform(-1, 1))
            k_list, v_list = sliding_window_evict(k_list, v_list, window)
            mem = 2 * len(k_list) * hd * 2
            print(f"  {step+1:>5}  {step+1:>12}  {len(k_list):>10}  {mem:>8} B")


if __name__ == "__main__":
    test_all()


# ──────────────────────────────────────────────────────────
# SOLUTION (read after attempting!)
# ──────────────────────────────────────────────────────────
#
# class KVCache:
#     def __init__(self, head_dim):
#         self.K = []
#         self.V = []
#         self.head_dim = head_dim
#
#     def append(self, k_new, v_new):
#         self.K.append(k_new)
#         self.V.append(v_new)
#
#     def size_bytes(self, bpv=2):
#         return 2 * len(self.K) * self.head_dim * bpv
#
#     def attend(self, q_vec):
#         if not self.K:
#             return [0.0] * self.head_dim
#         scale = 1.0 / math.sqrt(self.head_dim)
#         scores = [sum(q_vec[i]*k[i] for i in range(self.head_dim))*scale for k in self.K]
#         max_s = max(scores)
#         exps  = [math.exp(s - max_s) for s in scores]
#         total = sum(exps)
#         weights = [e / total for e in exps]
#         return [sum(weights[t]*self.V[t][d] for t in range(len(self.V)))
#                 for d in range(self.head_dim)]
#
# def sliding_window_evict(K_list, V_list, window_size):
#     if len(K_list) <= window_size:
#         return K_list, V_list
#     return K_list[-window_size:], V_list[-window_size:]
#
# def int8_quantize(vector):
#     max_abs = max(abs(x) for x in vector)
#     if max_abs == 0:
#         return [0] * len(vector), 1.0
#     scale = max_abs / 127.0
#     quantized = [max(-128, min(127, round(x / scale))) for x in vector]
#     return quantized, scale

"""
Exercise 02: Group-Query Attention (GQA)
Module 18: Qwen3.5 LLM from Scratch

TASKS:
  1. compute_kv_cache_bytes(L, H_kv, S, D, bpv) — KV cache memory formula
  2. gqa_group_size(n_q, n_kv)                   — how many Q heads share 1 KV head
  3. expand_kv_heads(kv, n_kv, n_q)              — repeat_kv implementation
  4. weight_param_count(d, n_q, n_kv, head_dim)  — total QKV projection parameters

Run:   python exercise_02_gqa.py
Deps:  none (pure Python)
"""


# ──────────────────────────────────────────────────────────
# TASK 1: KV Cache Memory
# ──────────────────────────────────────────────────────────

def compute_kv_cache_bytes(L, H_kv, S, D, bpv=2):
    """
    Compute KV cache memory in bytes.

    Args:
        L   (int): number of transformer layers
        H_kv(int): number of KV heads per layer
        S   (int): sequence length (context window)
        D   (int): head dimension
        bpv (int): bytes per value (2=bf16, 1=INT8)

    Returns:
        int: total bytes used by KV cache

    Example:
        compute_kv_cache_bytes(28, 8, 32768, 128, 2)
        # = 2 * 28 * 8 * 32768 * 128 * 2 = 4,294,967,296 bytes ≈ 4.3 GB

    HINT:
        Formula: 2 × L × H_kv × S × D × bpv
        The leading 2 is for K AND V (two separate matrices).
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 2: GQA Group Size
# ──────────────────────────────────────────────────────────

def gqa_group_size(n_q, n_kv):
    """
    Compute how many Q heads share a single KV head.

    Args:
        n_q  (int): number of query heads
        n_kv (int): number of key/value heads

    Returns:
        int: group_size = n_q // n_kv

    Example:
        gqa_group_size(32, 8)   # returns 4
        gqa_group_size(32, 32)  # returns 1 (MHA — no grouping)
        gqa_group_size(32, 1)   # returns 32 (MQA — all share 1 KV)

    HINT:
        One line: return n_q // n_kv
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 3: Expand KV Heads (repeat_kv)
# ──────────────────────────────────────────────────────────

def expand_kv_heads(kv_heads, n_kv, n_q):
    """
    Expand KV heads from n_kv to n_q by repeating each head group_size times.
    This is the repeat_kv operation done before attention computation.

    Args:
        kv_heads (list): list of n_kv items (each item = one KV head's data)
        n_kv     (int):  current number of KV heads
        n_q      (int):  target number of Q heads

    Returns:
        list: length n_q, each KV head repeated (n_q // n_kv) times

    Example:
        expand_kv_heads(["KV0", "KV1"], 2, 4)
        # returns ["KV0", "KV0", "KV1", "KV1"]

    HINT:
        group_size = n_q // n_kv
        For each kv_head in kv_heads:
            append it group_size times to the result list
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 4: Weight Parameter Count
# ──────────────────────────────────────────────────────────

def weight_param_count(d, n_q, n_kv, head_dim):
    """
    Count total parameters in the QKV + output projection matrices.

    Args:
        d        (int): model dimension (d_model)
        n_q      (int): number of Q heads
        n_kv     (int): number of KV heads
        head_dim (int): dimension per head

    Returns:
        dict with keys:
            "W_q": int  (d × (n_q * head_dim))
            "W_k": int  (d × (n_kv * head_dim))
            "W_v": int  (d × (n_kv * head_dim))
            "W_o": int  (d × (n_q * head_dim))   ← output projection
            "total": int (sum of all four)

    Example:
        weight_param_count(4096, 32, 8, 128)
        # W_q = 4096 * 4096 = 16,777,216
        # W_k = 4096 * 1024 = 4,194,304
        # ...

    HINT:
        W_q shape: [d, n_q * head_dim]  → params = d * n_q * head_dim
        W_k shape: [d, n_kv * head_dim] → params = d * n_kv * head_dim
        W_v = same as W_k
        W_o shape: [n_q * head_dim, d]  → params = n_q * head_dim * d (same as W_q)
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 02: GQA — Test Results")
    print("=" * 55)
    passed = 0
    failed = 0

    # --- Test 1: compute_kv_cache_bytes ---
    print("\n[Task 1] compute_kv_cache_bytes")
    try:
        # Qwen3.5-7B at 32K context
        result = compute_kv_cache_bytes(28, 8, 32768, 128, 2)
        assert result is not None, "Returned None"
        expected = 2 * 28 * 8 * 32768 * 128 * 2
        assert result == expected, f"Expected {expected}, got {result}"

        # MHA for comparison
        mha = compute_kv_cache_bytes(28, 32, 32768, 128, 2)
        assert mha == 4 * result, f"MHA should be 4x GQA, got ratio {mha/result}"

        # INT8 should be half of bf16
        int8 = compute_kv_cache_bytes(28, 8, 32768, 128, 1)
        assert int8 == result // 2, f"INT8 should be half of bf16"

        print(f"  PASS: GQA={result/1e9:.2f}GB, MHA={mha/1e9:.2f}GB (4x), INT8={int8/1e9:.2f}GB (half)")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 2: gqa_group_size ---
    print("\n[Task 2] gqa_group_size")
    try:
        assert gqa_group_size(32, 8)  == 4,  f"Expected 4, got {gqa_group_size(32, 8)}"
        assert gqa_group_size(32, 32) == 1,  f"Expected 1 (MHA), got {gqa_group_size(32, 32)}"
        assert gqa_group_size(32, 1)  == 32, f"Expected 32 (MQA), got {gqa_group_size(32, 1)}"
        assert gqa_group_size(8, 2)   == 4,  f"Expected 4, got {gqa_group_size(8, 2)}"
        print("  PASS: GQA=4, MHA=1, MQA=32 all correct")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 3: expand_kv_heads ---
    print("\n[Task 3] expand_kv_heads")
    try:
        # Simple string example
        result = expand_kv_heads(["KV0", "KV1"], 2, 4)
        assert result is not None, "Returned None"
        assert len(result) == 4, f"Expected length 4, got {len(result)}"
        assert result == ["KV0", "KV0", "KV1", "KV1"], f"Wrong expansion: {result}"

        # MQA: 1 KV head → 8 Q heads
        result_mqa = expand_kv_heads(["KV0"], 1, 8)
        assert result_mqa == ["KV0"] * 8, f"MQA should repeat 8 times: {result_mqa}"

        # MHA: no expansion needed (group_size=1)
        kvs = ["KV0", "KV1", "KV2"]
        result_mha = expand_kv_heads(kvs, 3, 3)
        assert result_mha == kvs, f"MHA should not change order: {result_mha}"

        print("  PASS: GQA expansion, MQA, MHA all correct")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 4: weight_param_count ---
    print("\n[Task 4] weight_param_count")
    try:
        # d=4096, n_q=32, n_kv=8, head_dim=128 (Qwen3.5-7B)
        result = weight_param_count(4096, 32, 8, 128)
        assert result is not None, "Returned None"
        assert isinstance(result, dict), "Should return a dict"

        expected_wq = 4096 * 32 * 128   # = 16,777,216
        expected_wk = 4096 * 8  * 128   # = 4,194,304
        expected_wo = 4096 * 32 * 128   # same as W_q

        assert result["W_q"] == expected_wq, f"W_q: expected {expected_wq}, got {result['W_q']}"
        assert result["W_k"] == expected_wk, f"W_k: expected {expected_wk}, got {result['W_k']}"
        assert result["W_v"] == expected_wk, f"W_v: should equal W_k"
        assert result["W_o"] == expected_wo, f"W_o: should equal W_q"
        assert result["total"] == expected_wq + expected_wk + expected_wk + expected_wo, \
            f"Total mismatch"

        # Compare MHA vs GQA total
        mha_result = weight_param_count(4096, 32, 32, 128)
        gqa_result = weight_param_count(4096, 32,  8, 128)
        assert mha_result["total"] > gqa_result["total"], "MHA should have more params than GQA"

        savings = (mha_result["total"] - gqa_result["total"]) / mha_result["total"] * 100
        print(f"  PASS: W_q={result['W_q']/1e6:.1f}M, W_k={result['W_k']/1e6:.1f}M, "
              f"GQA saves {savings:.0f}% vs MHA on KV weights")
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
        print("  All tasks complete! GQA understanding confirmed.")
    else:
        print("  Fix failing tasks above, then re-run.")


if __name__ == "__main__":
    test_all()


# ──────────────────────────────────────────────────────────
# SOLUTION (read after attempting!)
# ──────────────────────────────────────────────────────────
#
# def compute_kv_cache_bytes(L, H_kv, S, D, bpv=2):
#     return 2 * L * H_kv * S * D * bpv
#
# def gqa_group_size(n_q, n_kv):
#     return n_q // n_kv
#
# def expand_kv_heads(kv_heads, n_kv, n_q):
#     group_size = n_q // n_kv
#     result = []
#     for kv_head in kv_heads:
#         for _ in range(group_size):
#             result.append(kv_head)
#     return result
#
# def weight_param_count(d, n_q, n_kv, head_dim):
#     W_q = d * n_q  * head_dim
#     W_k = d * n_kv * head_dim
#     W_v = d * n_kv * head_dim
#     W_o = d * n_q  * head_dim
#     return {"W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o, "total": W_q+W_k+W_v+W_o}

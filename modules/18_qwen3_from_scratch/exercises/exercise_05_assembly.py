"""
Exercise 05: Qwen3.5 Architecture Assembly
Module 18: Qwen3.5 LLM from Scratch

TASKS:
  1. rms_norm(x, gamma, eps)             — RMSNorm normalization
  2. silu(x)                             — Sigmoid Linear Unit activation
  3. swiglu_ffn(x, W_gate, W_up, W_down) — full SwiGLU FFN forward pass
  4. count_block_params(d, n_q, n_kv, head_dim, d_ffn) — parameter count

Run:   python exercise_05_assembly.py
Deps:  none (pure Python)
"""

import math


# ──────────────────────────────────────────────────────────
# HELPERS (provided — do not modify)
# ──────────────────────────────────────────────────────────

def mat_vec_mul(W, x):
    """Multiply matrix W [out, in] by vector x [in] → [out]."""
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]


# ──────────────────────────────────────────────────────────
# TASK 1: RMSNorm
# ──────────────────────────────────────────────────────────

def rms_norm(x, gamma, eps=1e-5):
    """
    Apply Root Mean Square Normalization.

    Formula:
        rms   = sqrt(mean(x_i^2) + eps)
        x_hat = x / rms
        output = gamma * x_hat   (element-wise)

    Args:
        x     (list[float]): input vector, length d
        gamma (list[float]): learned scale parameter, length d
        eps   (float):       small constant for stability

    Returns:
        list[float]: normalized vector, length d

    Example:
        rms_norm([3.0, 4.0], [1.0, 1.0])
        # rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        # output ≈ [3/3.536, 4/3.536] ≈ [0.8485, 1.1314]

    HINT:
        1. rms = math.sqrt(sum(xi**2 for xi in x) / len(x) + eps)
        2. return [gamma[i] * x[i] / rms for i in range(len(x))]
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 2: SiLU Activation
# ──────────────────────────────────────────────────────────

def silu(x):
    """
    Sigmoid Linear Unit: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Args:
        x (float): single scalar value

    Returns:
        float: silu(x)

    Example:
        silu(0.0)  →  0.0 * 0.5 = 0.0
        silu(1.0)  →  1.0 * sigmoid(1.0) ≈ 0.7311
        silu(-1.0) → -1.0 * sigmoid(-1.0) ≈ -0.2689

    HINT:
        sigmoid(x) = 1 / (1 + exp(-x))
        silu(x)    = x * sigmoid(x)
        One line: return x / (1 + math.exp(-x))
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 3: SwiGLU FFN
# ──────────────────────────────────────────────────────────

def swiglu_ffn(x, W_gate, W_up, W_down):
    """
    SwiGLU Feed-Forward Network (3-matrix version used in Qwen3.5).

    Formula:
        gate   = W_gate @ x          (shape: [d_ffn])
        up     = W_up   @ x          (shape: [d_ffn])
        hidden = silu(gate) * up      (element-wise multiply)
        output = W_down @ hidden      (shape: [d_model])

    Args:
        x      (list[float]):       input vector [d_model]
        W_gate (list[list[float]]): [d_ffn, d_model]
        W_up   (list[list[float]]): [d_ffn, d_model]
        W_down (list[list[float]]): [d_model, d_ffn]

    Returns:
        list[float]: output vector [d_model]

    HINT:
        Use the provided mat_vec_mul(W, x) helper.
        gate   = mat_vec_mul(W_gate, x)
        up     = mat_vec_mul(W_up, x)
        hidden = [silu(gate[i]) * up[i] for i in range(len(gate))]
        output = mat_vec_mul(W_down, hidden)
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TASK 4: Count Block Parameters
# ──────────────────────────────────────────────────────────

def count_block_params(d, n_q, n_kv, head_dim, d_ffn):
    """
    Count the total learnable parameters in ONE Qwen3.5 decoder block.

    A block contains:
        Attention sub-layer:
            W_q  : [d, n_q * head_dim]
            W_k  : [d, n_kv * head_dim]
            W_v  : [d, n_kv * head_dim]
            W_o  : [n_q * head_dim, d]  → stored as [d, n_q * head_dim]
        SwiGLU FFN sub-layer:
            W_gate: [d_ffn, d]
            W_up  : [d_ffn, d]
            W_down: [d, d_ffn]
        RMSNorm (2 per block, one before attention, one before FFN):
            gamma_attn: [d]
            gamma_ffn : [d]

    Args:
        d       (int): model dimension
        n_q     (int): number of Q heads
        n_kv    (int): number of KV heads
        head_dim(int): dimension per head
        d_ffn   (int): FFN hidden dimension

    Returns:
        dict with keys:
            "attention": int  (W_q + W_k + W_v + W_o params)
            "ffn":       int  (W_gate + W_up + W_down params)
            "norms":     int  (2 × d for the two RMSNorm gammas)
            "total":     int  (sum of all three)

    Example:
        count_block_params(256, 8, 2, 32, 683)
        # attention = 256*(8*32) + 256*(2*32) + 256*(2*32) + 256*(8*32)
        #           = 65536 + 16384 + 16384 + 65536 = 163840

    HINT:
        attention = d*(n_q*head_dim) + d*(n_kv*head_dim)*2 + d*(n_q*head_dim)
        ffn       = d_ffn*d*2 + d*d_ffn   (gate + up + down)
        norms     = d * 2
    """
    # TODO: implement this
    pass


# ──────────────────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 05: Assembly — Test Results")
    print("=" * 55)
    passed = 0
    failed = 0

    # --- Test 1: rms_norm ---
    print("\n[Task 1] rms_norm")
    try:
        # [3, 4] → rms = sqrt((9+16)/2) = sqrt(12.5)
        out = rms_norm([3.0, 4.0], [1.0, 1.0])
        assert out is not None, "Returned None"
        rms_val = math.sqrt((9 + 16) / 2 + 1e-5)
        expected = [3.0 / rms_val, 4.0 / rms_val]
        assert abs(out[0] - expected[0]) < 1e-6, f"out[0]: expected {expected[0]:.6f}, got {out[0]}"
        assert abs(out[1] - expected[1]) < 1e-6, f"out[1]: expected {expected[1]:.6f}, got {out[1]}"

        # gamma scaling
        out_scaled = rms_norm([1.0, 2.0], [2.0, 3.0])
        out_unit   = rms_norm([1.0, 2.0], [1.0, 1.0])
        assert abs(out_scaled[0] - 2.0 * out_unit[0]) < 1e-9, "gamma=2 should double output[0]"
        assert abs(out_scaled[1] - 3.0 * out_unit[1]) < 1e-9, "gamma=3 should triple output[1]"

        # Constant vector should normalize to constant (with gamma=1)
        out_const = rms_norm([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
        assert all(abs(x - 1.0) < 1e-6 for x in out_const), \
            f"[2,2,2] with gamma=1 should give [1,1,1]: {out_const}"

        print(f"  PASS: norm([3,4])≈{[round(x,4) for x in out]}, gamma scaling correct")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 2: silu ---
    print("\n[Task 2] silu")
    try:
        # silu(0) = 0
        assert abs(silu(0.0) - 0.0) < 1e-9, f"silu(0) should be 0, got {silu(0.0)}"

        # silu(1) = 1 * sigmoid(1) = 1 / (1+e^-1) ≈ 0.7311
        expected_1 = 1.0 / (1.0 + math.exp(-1.0))
        assert abs(silu(1.0) - expected_1) < 1e-9, f"silu(1): expected {expected_1:.6f}, got {silu(1.0)}"

        # silu(-1) should be negative
        assert silu(-1.0) < 0, "silu(-1) should be negative"

        # silu is monotonically increasing for large x (approaches identity)
        assert silu(10.0) > silu(5.0) > silu(1.0) > silu(0.0), \
            "silu should be increasing"

        # Unlike ReLU, silu(-1) is NOT zero (smooth)
        assert silu(-0.5) != 0.0, "silu should be smooth (not zero for negative inputs)"

        print(f"  PASS: silu(0)=0, silu(1)≈{expected_1:.4f}, smooth for negatives")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 3: swiglu_ffn ---
    print("\n[Task 3] swiglu_ffn")
    try:
        import random
        rng = random.Random(42)
        rand_mat = lambda r, c: [[rng.uniform(-0.5, 0.5) for _ in range(c)] for _ in range(r)]

        d_model, d_ffn = 4, 6
        x_in   = [0.5, -0.3, 0.8, -0.1]
        W_gate = rand_mat(d_ffn, d_model)
        W_up   = rand_mat(d_ffn, d_model)
        W_down = rand_mat(d_model, d_ffn)

        out = swiglu_ffn(x_in, W_gate, W_up, W_down)
        assert out is not None, "Returned None"
        assert len(out) == d_model, f"Output should be length {d_model}, got {len(out)}"
        assert any(x != 0.0 for x in out), "Output should not be all zeros"

        # Verify gating: if gate is very negative, silu(gate) ≈ 0, so output ≈ 0
        # Make W_gate all large negatives → gate → large negative → silu → 0
        W_gate_neg = [[-100.0] * d_model for _ in range(d_ffn)]
        out_gated  = swiglu_ffn(x_in, W_gate_neg, W_up, W_down)
        max_abs = max(abs(x) for x in out_gated)
        assert max_abs < 1e-3, f"With very negative gate, output should be ~0, got {max_abs}"

        print(f"  PASS: output length {d_model}, gating works (near-zero output with neg gate)")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # --- Test 4: count_block_params ---
    print("\n[Task 4] count_block_params")
    try:
        # Mini-Qwen config: d=256, n_q=8, n_kv=2, head_dim=32, d_ffn=683
        result = count_block_params(256, 8, 2, 32, 683)
        assert result is not None, "Returned None"
        assert isinstance(result, dict), "Should return a dict"

        expected_attn = 256*(8*32) + 256*(2*32)*2 + 256*(8*32)  # Q+K+V+O
        expected_ffn  = 683*256*2 + 256*683                       # gate+up+down
        expected_norms = 256 * 2

        assert result["attention"] == expected_attn, \
            f"attention: expected {expected_attn}, got {result['attention']}"
        assert result["ffn"] == expected_ffn, \
            f"ffn: expected {expected_ffn}, got {result['ffn']}"
        assert result["norms"] == expected_norms, \
            f"norms: expected {expected_norms}, got {result['norms']}"
        assert result["total"] == expected_attn + expected_ffn + expected_norms, \
            "total should sum all components"

        # Sanity: 6-layer Mini-Qwen + embedding should be ~17M params
        vocab  = 50257
        n_layers = 6
        embed_params = vocab * 256
        block_total  = result["total"]
        model_total  = n_layers * block_total + embed_params + 256  # +final RMSNorm
        assert 10_000_000 < model_total < 25_000_000, \
            f"Mini-Qwen should be ~17M params, got {model_total/1e6:.1f}M"

        print(f"  PASS: attn={result['attention']/1e3:.1f}K, "
              f"ffn={result['ffn']/1e3:.1f}K, "
              f"total={result['total']/1e3:.1f}K per block, "
              f"Mini-Qwen≈{model_total/1e6:.1f}M params")
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
        print("  All tasks complete! Qwen3.5 building blocks implemented.")
        print()
        print("  Architecture summary for Mini-Qwen (6 layers):")
        r = count_block_params(256, 8, 2, 32, 683)
        total = 6 * r["total"] + 50257 * 256 + 256
        print(f"    Per block:  attention={r['attention']/1e3:.1f}K, "
              f"FFN={r['ffn']/1e3:.1f}K, norms={r['norms']}B")
        print(f"    6 blocks:   {6*r['total']/1e6:.2f}M params")
        print(f"    Embedding:  {50257*256/1e6:.2f}M params")
        print(f"    TOTAL:      ~{total/1e6:.1f}M params")
    else:
        print("  Fix failing tasks above, then re-run.")


if __name__ == "__main__":
    test_all()


# ──────────────────────────────────────────────────────────
# SOLUTION (read after attempting!)
# ──────────────────────────────────────────────────────────
#
# def rms_norm(x, gamma, eps=1e-5):
#     rms = math.sqrt(sum(xi**2 for xi in x) / len(x) + eps)
#     return [gamma[i] * x[i] / rms for i in range(len(x))]
#
# def silu(x):
#     return x / (1 + math.exp(-x))
#
# def swiglu_ffn(x, W_gate, W_up, W_down):
#     gate   = mat_vec_mul(W_gate, x)
#     up     = mat_vec_mul(W_up, x)
#     hidden = [silu(gate[i]) * up[i] for i in range(len(gate))]
#     return mat_vec_mul(W_down, hidden)
#
# def count_block_params(d, n_q, n_kv, head_dim, d_ffn):
#     attn  = d*(n_q*head_dim) + d*(n_kv*head_dim)*2 + d*(n_q*head_dim)
#     ffn   = d_ffn*d*2 + d*d_ffn
#     norms = d * 2
#     return {"attention": attn, "ffn": ffn, "norms": norms, "total": attn+ffn+norms}

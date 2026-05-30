"""
Exercise 04: Gradient Checkpointing — Memory Measurement
Module 15: Advanced LLM Training

TASKS:
  1. Implement activation_memory_no_checkpoint() — memory for storing all activations
  2. Implement activation_memory_with_checkpoint() — memory with checkpointing
  3. Implement optimal_checkpoint_interval() — find interval that minimizes memory
  4. Analyze the compute/memory tradeoff at different checkpoint intervals

Run:  python exercise_04_checkpointing.py
Deps: none (pure Python)
"""

import math


# ─────────────────────────────────────────────────────────
# TASK 1: Memory Without Checkpointing
# ─────────────────────────────────────────────────────────

def activation_memory_no_checkpoint(n_layers: int,
                                     seq_len: int,
                                     hidden_size: int,
                                     dtype_bytes: int = 2) -> float:
    """
    Calculate activation memory during training WITHOUT checkpointing.

    ALL layer activations are stored for use in the backward pass.
    Total activations = n_layers × one_activation_size

    One activation size = seq_len × hidden_size × dtype_bytes

    Args:
        n_layers:   number of transformer layers
        seq_len:    sequence length (tokens)
        hidden_size: model hidden dimension
        dtype_bytes: bytes per value (2 for bf16, 4 for fp32)

    Returns:
        Total activation memory in bytes

    HINT: activation_per_layer = seq_len × hidden_size × dtype_bytes
          total = n_layers × activation_per_layer
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Memory With Checkpointing
# ─────────────────────────────────────────────────────────

def activation_memory_with_checkpoint(n_layers: int,
                                       seq_len: int,
                                       hidden_size: int,
                                       checkpoint_every: int,
                                       dtype_bytes: int = 2) -> float:
    """
    Calculate peak activation memory WITH gradient checkpointing.

    With checkpointing every C layers:
      - Saved: ceil(n_layers / C) checkpoint activations
      - Recomputed during backward: C activations max at any time

    Peak memory = (checkpoints stored) + (one segment being recomputed)
                = ceil(n_layers / C) × one_activation
                + C × one_activation
                = (ceil(n_layers / C) + C) × one_activation

    Args:
        n_layers:           number of transformer layers
        seq_len:            sequence length
        hidden_size:        model hidden dimension
        checkpoint_every:   store checkpoint every C layers
        dtype_bytes:        bytes per value

    Returns:
        Peak activation memory in bytes

    HINT:
        n_checkpoints = math.ceil(n_layers / checkpoint_every)
        peak_layers = n_checkpoints + checkpoint_every
        activation_per_layer = seq_len * hidden_size * dtype_bytes
        return peak_layers * activation_per_layer
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Optimal Checkpoint Interval
# ─────────────────────────────────────────────────────────

def optimal_checkpoint_interval(n_layers: int) -> int:
    """
    Find the checkpoint interval that minimizes peak activation memory.

    Mathematical result from calculus:
        Minimize (ceil(N/C) + C) × activation
        Optimal C = sqrt(N)   (rounded to nearest integer)

    Args:
        n_layers: number of transformer layers

    Returns:
        Optimal checkpoint interval (integer)

    HINT: return int(round(math.sqrt(n_layers)))
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Compute Cost Estimate
# ─────────────────────────────────────────────────────────

def recompute_overhead_pct(n_layers: int, checkpoint_every: int) -> float:
    """
    Estimate the percentage extra compute from recomputing activations.

    During backward, each checkpoint segment is recomputed once.
    Number of recomputed layers = n_layers - n_checkpoints
      (all layers except the checkpoints themselves)

    Extra compute fraction = recomputed_layers / n_layers × 100

    Args:
        n_layers:         total number of layers
        checkpoint_every: checkpoint interval

    Returns:
        Extra compute percentage (float)

    HINT:
        n_checkpoints = math.ceil(n_layers / checkpoint_every)
        recomputed = n_layers - n_checkpoints
        return recomputed / n_layers * 100
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 65)
    print("  Exercise 04: Gradient Checkpointing")
    print("=" * 65)

    N_LAYERS = 32
    SEQ_LEN  = 2048
    HIDDEN   = 4096
    BYTES    = 2   # bf16

    # Test 1: No checkpoint memory
    print("\n--- Test 1: activation_memory_no_checkpoint ---")
    result = activation_memory_no_checkpoint(N_LAYERS, SEQ_LEN, HIDDEN, BYTES)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        expected = N_LAYERS * SEQ_LEN * HIDDEN * BYTES
        if abs(result - expected) < 1:
            gb = result / 1e9
            print(f"  PASS  total activation memory = {gb:.2f} GB")
        else:
            print(f"  FAIL  got {result}, expected {expected}")

    # Test 2: With checkpoint memory
    print("\n--- Test 2: activation_memory_with_checkpoint ---")
    result = activation_memory_with_checkpoint(N_LAYERS, SEQ_LEN, HIDDEN,
                                                checkpoint_every=4, dtype_bytes=BYTES)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        n_checkpoints = math.ceil(N_LAYERS / 4)
        expected = (n_checkpoints + 4) * SEQ_LEN * HIDDEN * BYTES
        if abs(result - expected) < 1:
            gb = result / 1e9
            print(f"  PASS  peak memory with checkpoint_every=4: {gb:.3f} GB")
        else:
            print(f"  FAIL  got {result}, expected {expected}")

    # Test 3: Optimal checkpoint interval
    print("\n--- Test 3: optimal_checkpoint_interval ---")
    result = optimal_checkpoint_interval(32)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        if result == 6:  # round(sqrt(32)) = round(5.66) = 6
            print(f"  PASS  optimal interval for 32 layers = {result}")
        else:
            print(f"  FAIL  expected 6, got {result}  (hint: round(sqrt(32)))")

    # Test 4: Recompute overhead
    print("\n--- Test 4: recompute_overhead_pct ---")
    result = recompute_overhead_pct(32, 8)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        # n_checkpoints = ceil(32/8) = 4, recomputed = 32-4 = 28, pct = 28/32 * 100 = 87.5
        expected = 87.5
        if abs(result - expected) < 0.1:
            print(f"  PASS  overhead for checkpoint_every=8 = {result:.1f}%")
        else:
            print(f"  FAIL  expected {expected:.1f}%, got {result:.1f}%")

    # BONUS: Full tradeoff table
    print("\n--- BONUS: Memory vs Compute Tradeoff Table ---")
    no_ckpt = activation_memory_no_checkpoint(N_LAYERS, SEQ_LEN, HIDDEN, BYTES)
    overhead_fn = recompute_overhead_pct
    mem_fn = activation_memory_with_checkpoint

    if no_ckpt is not None and mem_fn(N_LAYERS, SEQ_LEN, HIDDEN, 1, BYTES) is not None:
        print(f"\nModel: {N_LAYERS} layers, seq={SEQ_LEN}, hidden={HIDDEN}, bf16")
        print(f"No checkpointing: {no_ckpt/1e9:.2f} GB")
        print(f"\n{'Ckpt Every':>12} {'Peak Mem (GB)':>14} {'Savings':>10} {'Extra Compute':>15}")
        print("-" * 55)
        for c in [1, 2, 4, 6, 8, 16, 32]:
            mem = mem_fn(N_LAYERS, SEQ_LEN, HIDDEN, c, BYTES)
            overhead = overhead_fn(N_LAYERS, c) if overhead_fn(N_LAYERS, c) is not None else float('nan')
            savings = no_ckpt / mem
            optimal_marker = " ← optimal" if c == optimal_checkpoint_interval(N_LAYERS) else ""
            print(f"{c:>12} {mem/1e9:>14.3f} {savings:>9.1f}x {overhead:>14.1f}%{optimal_marker}")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def activation_memory_no_checkpoint(n_layers, seq_len, hidden_size, dtype_bytes=2):
#     return n_layers * seq_len * hidden_size * dtype_bytes
#
# def activation_memory_with_checkpoint(n_layers, seq_len, hidden_size, checkpoint_every, dtype_bytes=2):
#     n_checkpoints = math.ceil(n_layers / checkpoint_every)
#     peak_layers = n_checkpoints + checkpoint_every
#     return peak_layers * seq_len * hidden_size * dtype_bytes
#
# def optimal_checkpoint_interval(n_layers):
#     return int(round(math.sqrt(n_layers)))
#
# def recompute_overhead_pct(n_layers, checkpoint_every):
#     n_checkpoints = math.ceil(n_layers / checkpoint_every)
#     recomputed = n_layers - n_checkpoints
#     return recomputed / n_layers * 100

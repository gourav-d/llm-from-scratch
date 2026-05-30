"""
Example 04: Gradient Checkpointing — Manual Implementation
Module 15: Advanced LLM Training

Demonstrates:
  - Why training stores activations (needed for backward pass)
  - How gradient checkpointing discards and recomputes activations
  - Memory measurement: with vs without checkpointing
  - The sqrt(N) checkpoint strategy

Run:  python example_04_gradient_checkpointing.py
Deps: numpy
"""

import numpy as np
import sys


def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


# ─────────────────────────────────────────────────────────
# SIMPLE LAYER AND MEMORY TRACKER
# ─────────────────────────────────────────────────────────

class MemoryTracker:
    """Tracks peak memory usage during forward and backward passes."""

    def __init__(self):
        self.allocated_bytes = 0
        self.peak_bytes = 0
        self.log = []

    def allocate(self, name: str, shape: tuple, dtype_bytes: int = 4):
        size = int(np.prod(shape)) * dtype_bytes
        self.allocated_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.allocated_bytes)
        self.log.append(f"  + {name:<30} {self._fmt(size):>10}  total={self._fmt(self.allocated_bytes)}")
        return size

    def free(self, name: str, shape: tuple, dtype_bytes: int = 4):
        size = int(np.prod(shape)) * dtype_bytes
        self.allocated_bytes -= size
        self.log.append(f"  - {name:<30} {self._fmt(size):>10}  total={self._fmt(self.allocated_bytes)}")

    def _fmt(self, b: int) -> str:
        if b >= 1024**2:
            return f"{b/1024**2:.1f} MB"
        elif b >= 1024:
            return f"{b/1024:.1f} KB"
        return f"{b} B"

    def reset(self):
        self.allocated_bytes = 0
        self.peak_bytes = 0
        self.log = []


# ─────────────────────────────────────────────────────────
# FORWARD/BACKWARD WITHOUT CHECKPOINTING
# ─────────────────────────────────────────────────────────

def simulate_forward_no_checkpoint(n_layers: int, seq_len: int, hidden: int,
                                    tracker: MemoryTracker):
    """
    Standard training: stores ALL activations for backward pass.
    Memory: O(n_layers × seq_len × hidden)
    """
    activations = []

    # Forward pass: store every activation
    x = np.random.randn(seq_len, hidden).astype(np.float32)
    tracker.allocate("input", (seq_len, hidden))

    for i in range(n_layers):
        # Each layer produces an activation (simplified: same shape as input)
        activation = np.tanh(x)   # toy activation
        tracker.allocate(f"layer_{i}_activation", (seq_len, hidden))
        activations.append(activation)
        x = activation

    return activations


def simulate_backward_no_checkpoint(n_layers: int, seq_len: int, hidden: int,
                                     activations: list, tracker: MemoryTracker):
    """Backward pass: uses stored activations, then frees them."""
    grad = np.ones((seq_len, hidden), dtype=np.float32)
    tracker.allocate("gradient_buffer", (seq_len, hidden))

    for i in range(n_layers - 1, -1, -1):
        # Compute gradient using stored activation
        _ = activations[i] * grad   # toy gradient computation
        # Free the activation after use
        tracker.free(f"layer_{i}_activation", (seq_len, hidden))

    tracker.free("gradient_buffer", (seq_len, hidden))
    tracker.free("input", (seq_len, hidden))


# ─────────────────────────────────────────────────────────
# FORWARD/BACKWARD WITH CHECKPOINTING
# ─────────────────────────────────────────────────────────

def simulate_forward_with_checkpoint(n_layers: int, seq_len: int, hidden: int,
                                      checkpoint_every: int,
                                      tracker: MemoryTracker):
    """
    Checkpointed training: only saves activations at checkpoint boundaries.
    Intermediate activations are discarded and recomputed during backward.
    Memory: O(sqrt(n_layers) × seq_len × hidden)
    """
    checkpoints = []    # saved activations at checkpoint boundaries
    inputs = []         # saved inputs to each checkpoint segment

    x = np.random.randn(seq_len, hidden).astype(np.float32)
    tracker.allocate("input", (seq_len, hidden))

    for i in range(n_layers):
        # Save checkpoint at boundary
        if i % checkpoint_every == 0:
            checkpoint = x.copy()
            tracker.allocate(f"checkpoint_{i}", (seq_len, hidden))
            checkpoints.append((i, checkpoint))

        # Compute activation (discard immediately — NOT stored)
        x = np.tanh(x)
        # Note: no tracker.allocate here — activation is a temporary, not stored

    return checkpoints


def simulate_backward_with_checkpoint(n_layers: int, seq_len: int, hidden: int,
                                       checkpoint_every: int,
                                       checkpoints: list, tracker: MemoryTracker):
    """
    Backward pass: recomputes activations from checkpoints as needed.
    More compute (recompute) but less memory (no stored activations).
    """
    grad = np.ones((seq_len, hidden), dtype=np.float32)
    tracker.allocate("gradient_buffer", (seq_len, hidden))

    recompute_count = 0

    for seg_start in range(n_layers - checkpoint_every, -1, -checkpoint_every):
        seg_end = min(seg_start + checkpoint_every, n_layers)

        # Find the checkpoint for this segment
        ckpt_idx, ckpt_x = next(c for c in checkpoints if c[0] == seg_start)

        # Recompute forward through this segment to get intermediate activations
        x_recomputed = ckpt_x.copy()
        segment_activations = []

        for i in range(seg_start, seg_end):
            segment_activations.append(x_recomputed.copy())
            tracker.allocate(f"recomputed_{i}", (seq_len, hidden))
            recompute_count += 1
            x_recomputed = np.tanh(x_recomputed)

        # Backward through the recomputed segment
        for i in range(seg_end - 1, seg_start - 1, -1):
            _ = segment_activations[i - seg_start] * grad   # toy gradient
            tracker.free(f"recomputed_{i}", (seq_len, hidden))

        # Free checkpoint after backward through its segment
        tracker.free(f"checkpoint_{seg_start}", (seq_len, hidden))

    tracker.free("gradient_buffer", (seq_len, hidden))
    tracker.free("input", (seq_len, hidden))
    return recompute_count


# ─────────────────────────────────────────────────────────
# DEMO 1: Without Checkpointing — Memory Trace
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: Training WITHOUT Gradient Checkpointing")

N_LAYERS = 8
SEQ_LEN = 512
HIDDEN = 256

tracker_std = MemoryTracker()
activations = simulate_forward_no_checkpoint(N_LAYERS, SEQ_LEN, HIDDEN, tracker_std)
simulate_backward_no_checkpoint(N_LAYERS, SEQ_LEN, HIDDEN, activations, tracker_std)

print(f"\nModel: {N_LAYERS} layers, seq_len={SEQ_LEN}, hidden={HIDDEN}")
print(f"\nMemory log (first 15 events):")
for line in tracker_std.log[:15]:
    print(line)
print(f"  ... ({len(tracker_std.log)} total events)")
print(f"\nPeak memory: {tracker_std.peak_bytes / 1024:.1f} KB")
print(f"  = {N_LAYERS} layers × {SEQ_LEN} × {HIDDEN} × 4 bytes = {N_LAYERS * SEQ_LEN * HIDDEN * 4 / 1024:.1f} KB (activations only)")


# ─────────────────────────────────────────────────────────
# DEMO 2: With Checkpointing — Memory Trace
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: Training WITH Gradient Checkpointing")

tracker_ckpt = MemoryTracker()
checkpoints = simulate_forward_with_checkpoint(N_LAYERS, SEQ_LEN, HIDDEN,
                                                checkpoint_every=2, tracker=tracker_ckpt)
recomputes = simulate_backward_with_checkpoint(N_LAYERS, SEQ_LEN, HIDDEN,
                                                checkpoint_every=2, checkpoints=checkpoints,
                                                tracker=tracker_ckpt)

print(f"\nModel: {N_LAYERS} layers, seq_len={SEQ_LEN}, hidden={HIDDEN}")
print(f"Checkpoint interval: every 2 layers")
print(f"\nMemory log (first 15 events):")
for line in tracker_ckpt.log[:15]:
    print(line)
print(f"  ... ({len(tracker_ckpt.log)} total events)")
print(f"\nPeak memory:  {tracker_ckpt.peak_bytes / 1024:.1f} KB")
print(f"Recomputations: {recomputes} (extra forward passes during backward)")


# ─────────────────────────────────────────────────────────
# DEMO 3: Memory Savings Table
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: Memory Savings vs Compute Cost Tradeoff")

print(f"\n{'Layers':>8} {'No Ckpt (MB)':>14} {'Ckpt (MB)':>12} {'Savings':>10} {'Recompute':>12}")
print("-" * 60)

for n_layers in [8, 16, 32, 64, 96]:
    # Without checkpointing: store all n_layers activations
    mem_std_mb = (n_layers * SEQ_LEN * HIDDEN * 4) / 1024**2

    # With checkpointing every sqrt(n) layers
    ckpt_interval = max(1, int(np.sqrt(n_layers)))
    n_checkpoints = n_layers // ckpt_interval
    mem_ckpt_mb = (n_checkpoints * SEQ_LEN * HIDDEN * 4) / 1024**2

    savings = mem_std_mb / max(mem_ckpt_mb, 0.001)
    recompute_pct = (ckpt_interval - 1) / ckpt_interval * 100

    print(f"{n_layers:>8} {mem_std_mb:>14.2f} {mem_ckpt_mb:>12.2f} {savings:>9.1f}x {recompute_pct:>11.0f}%")

print(f"\nNote: sqrt(N_layers) checkpoint interval gives O(sqrt(N)) memory")
print(f"      and ~33% extra compute (recompute half of segments).")


# ─────────────────────────────────────────────────────────
# DEMO 4: ZeRO Optimizer — Memory Sharding Simulation
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: ZeRO Optimizer Memory Sharding")

def zero_memory_per_gpu(n_params: int, n_gpus: int, stage: int) -> dict:
    """
    Calculate memory per GPU for each ZeRO stage.

    Bytes per element:
      fp32 weights    = 4 bytes
      fp16 weights    = 2 bytes (for compute)
      gradients fp16  = 2 bytes
      adam_m (fp32)   = 4 bytes
      adam_v (fp32)   = 4 bytes
    """
    # Base sizes in bytes
    weights_fp32 = n_params * 4
    weights_fp16 = n_params * 2
    gradients    = n_params * 2
    adam_m       = n_params * 4
    adam_v       = n_params * 4

    def gb(b): return b / 1e9

    if stage == 0:
        # Each GPU: full weights (fp16+fp32) + full gradients + full optimizer state
        per_gpu = weights_fp16 + weights_fp32 + gradients + adam_m + adam_v
    elif stage == 1:
        # Shard optimizer state only
        per_gpu = weights_fp16 + weights_fp32 + gradients + (adam_m + adam_v) / n_gpus
    elif stage == 2:
        # Shard optimizer state + gradients
        per_gpu = weights_fp16 + weights_fp32 + (gradients + adam_m + adam_v) / n_gpus
    elif stage == 3:
        # Shard everything including parameters
        per_gpu = (weights_fp16 + weights_fp32 + gradients + adam_m + adam_v) / n_gpus
    else:
        raise ValueError(f"Invalid ZeRO stage: {stage}")

    return {
        "per_gpu_bytes": per_gpu,
        "per_gpu_gb": gb(per_gpu),
        "total_gb": gb(per_gpu * n_gpus),
    }


print(f"\nModel: 7B parameters, 8 GPUs")
print(f"\n{'Stage':>8} {'Per GPU (GB)':>14} {'Total (GB)':>12} {'Savings vs S0':>16}")
print("-" * 54)

n_params = 7_000_000_000
n_gpus = 8

stage0 = zero_memory_per_gpu(n_params, n_gpus, stage=0)["per_gpu_gb"]

for stage in [0, 1, 2, 3]:
    result = zero_memory_per_gpu(n_params, n_gpus, stage)
    savings = stage0 / result["per_gpu_gb"]
    print(f"{'Stage '+str(stage):>8} {result['per_gpu_gb']:>14.1f} {result['total_gb']:>12.1f} {savings:>15.1f}x")

print(f"\nA100 80GB GPU can hold Stage 3 per-GPU memory for a 7B model.")
print(f"Without ZeRO: would need an 80GB GPU just for the weights.")

print(f"\nZeRO-Offload: move optimizer state to CPU RAM")
result_offload = zero_memory_per_gpu(n_params, 1, stage=1)
print(f"  Single GPU (no offload):  {zero_memory_per_gpu(n_params, 1, 0)['per_gpu_gb']:.1f} GB on GPU")
print(f"  Single GPU (with offload): ~{n_params * 2 / 1e9:.0f} GB on GPU (weights fp16 only)")
print(f"  Optimizer state on CPU:   ~{(n_params * 4 + n_params * 4) / 1e9:.0f} GB on CPU RAM")

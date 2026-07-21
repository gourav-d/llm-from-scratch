"""
Module 06 - Training & Fine-Tuning
Exercise 04: Fine-Tuning Strategies

GLOSSARY
--------
Fine-Tuning       : Taking a pre-trained model and training it further on
                    domain-specific data. Much cheaper than training from scratch.
                    Like hiring an experienced developer and giving them 1 week
                    of onboarding, vs training someone from scratch for 4 years.
Learning Rate (LR): How big each gradient step is.
                    Fine-tuning uses 10-100x lower LR than training from scratch.
                    Reason: Pre-trained weights are already good; large steps destroy them.
LR Warmup         : Gradually increasing LR from 0 to target over N steps.
                    Prevents destroying pre-trained weights right at the start.
                    Like slowly ramping up speed when starting a car.
Frozen Layer      : A layer whose weights are NOT updated during training.
                    Freezing = "this layer is finished learning; don't touch it."
Update Magnitude  : How much the weights changed relative to their original size.
                    magnitude = norm(delta_W) / norm(W)
                    Healthy fine-tuning: magnitude < 5% (i.e., < 0.05).
Catastrophic      : When fine-tuning causes the model to forget its pre-trained
Forgetting          knowledge. Prevented by: low LR, fewer epochs, warmup.
"""

import numpy as np   # NumPy for math

print("=" * 60)
print("Exercise 04: Fine-Tuning Strategies")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Learning Rate Warmup Schedule
#
#  Background:
#    During warmup, LR starts at 0 and linearly increases to target_lr
#    over the first warmup_steps steps:
#      lr = target_lr * (step / warmup_steps)   if step < warmup_steps
#      lr = target_lr                            if step >= warmup_steps
#
#    After warmup, the full learning rate is used.
#    Why? Immediately applying full LR at step 0 can destroy pre-trained weights.
#
#  Your Task:
#    Write: warmup_lr(step, warmup_steps, target_lr) -> float
#    Returns the current learning rate at a given training step.
#
#  C# Analogy:
#    Like a throttle that gradually increases output:
#      double lr = step < warmupSteps
#          ? targetLr * step / warmupSteps
#          : targetLr;
# ============================================================

print("-" * 50)
print("EXERCISE 1: Learning Rate Warmup")
print("-" * 50)
print()


def warmup_lr(step, warmup_steps, target_lr):
    """
    Compute learning rate at a given step using linear warmup.

    Parameters:
        step          (int)  : Current training step (0-indexed).
        warmup_steps  (int)  : Number of warmup steps.
        target_lr     (float): Final learning rate after warmup.

    Returns:
        float: Learning rate at this step.
    """
    # TODO:
    # if step < warmup_steps:
    #     return target_lr * (step / warmup_steps)
    # else:
    #     return target_lr
    pass  # Replace with your implementation


target = 0.0001
warmup = 100

print(f"  target_lr={target}, warmup_steps={warmup}")
print()
print(f"  {'Step':>6}  {'LR':>12}")
print("  " + "-" * 22)
for step in [0, 10, 50, 99, 100, 200, 500]:
    lr = warmup_lr(step, warmup, target)
    if lr is not None:
        print(f"  {step:>6}  {lr:>12.8f}")
print()
print("  Expected: LR=0 at step 0, LR=target at step 100+")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Update Magnitude — Are Weight Changes Too Large?
#
#  Background:
#    When fine-tuning, we want SMALL weight updates (preserve pre-trained knowledge).
#    We measure this as:
#      magnitude = norm(delta_W) / norm(W)
#
#    Where:
#      delta_W = new_weights - old_weights
#      norm()  = Frobenius norm: sqrt(sum of all squared values)
#      np.linalg.norm(matrix) computes Frobenius norm for 2D arrays.
#
#    Rule of thumb:
#      magnitude < 0.05  -> safe fine-tuning (small change)
#      magnitude >= 0.05 -> risky (large change, may destroy pre-trained knowledge)
#
#  Your Task:
#    Write: update_magnitude(old_weights, new_weights) -> dict
#    Returns {"delta_norm": float, "weight_norm": float, "magnitude": float, "safe": bool}
#
#  C# Analogy:
#    Like comparing a diff to original file size:
#      magnitude = diffBytes / originalBytes  (should be < 5%)
# ============================================================

print("-" * 50)
print("EXERCISE 2: Update Magnitude")
print("-" * 50)
print()


def update_magnitude(old_weights, new_weights):
    """
    Compute how much weights changed relative to their original size.

    Parameters:
        old_weights (np.ndarray): Weights before fine-tuning step.
        new_weights (np.ndarray): Weights after fine-tuning step.

    Returns:
        dict: {
            "delta_norm"  : float -- norm of the weight change,
            "weight_norm" : float -- norm of original weights,
            "magnitude"   : float -- ratio: delta_norm / weight_norm,
            "safe"        : bool  -- True if magnitude < 0.05
        }
    """
    # TODO:
    # delta_W      = new_weights - old_weights
    # delta_norm   = np.linalg.norm(delta_W)
    # weight_norm  = np.linalg.norm(old_weights)
    # magnitude    = delta_norm / weight_norm
    # safe         = magnitude < 0.05
    pass  # Replace with your implementation


np.random.seed(1)
W = np.random.randn(64, 64) * 2.0   # Pre-trained weights (large values)

# Simulate a small fine-tuning update (good)
W_small = W - 0.0001 * np.random.randn(64, 64)   # tiny gradient step

# Simulate a large update (bad: too big, might destroy pre-trained knowledge)
W_large = W - 0.5   * np.random.randn(64, 64)    # huge gradient step

r_small = update_magnitude(W, W_small)
r_large = update_magnitude(W, W_large)

if r_small and r_large:
    print("  Small update (good fine-tuning):")
    print(f"    magnitude = {r_small['magnitude']:.4f}  safe = {r_small['safe']}")
    print()
    print("  Large update (dangerous):")
    print(f"    magnitude = {r_large['magnitude']:.4f}  safe = {r_large['safe']}")
print()
print("  Expected: small update safe=True, large update safe=False")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Frozen vs Trainable Parameters
#
#  Background:
#    Layer freezing means setting specific layers to NOT update.
#    Fine-tuning with frozen layers:
#      - Fewer parameters to update -> faster training
#      - Preserves embeddings / early layers -> less catastrophic forgetting
#      - Useful when you have VERY little fine-tuning data
#
#    trainable_params = total_params - frozen_params
#    savings_pct      = 100 * frozen_params / total_params
#
#  Your Task:
#    Write: frozen_analysis(layer_sizes, frozen_layers) -> dict
#    layer_sizes: list of param counts per layer [e.g., [10000, 5000, 2000]]
#    frozen_layers: list of layer indices that are frozen [e.g., [0, 1]]
#    Returns: {"total": int, "trainable": int, "frozen": int, "savings_pct": float}
#
#  C# Analogy:
#    Like LINQ: total = layers.Sum(), frozen = frozenIdxs.Sum(i => layers[i])
# ============================================================

print("-" * 50)
print("EXERCISE 3: Frozen vs Trainable Parameters")
print("-" * 50)
print()


def frozen_analysis(layer_sizes, frozen_layers):
    """
    Compute trainable vs frozen parameter counts.

    Parameters:
        layer_sizes   (list of int): Number of parameters per layer.
        frozen_layers (list of int): Indices of frozen layers (0-indexed).

    Returns:
        dict: {
            "total"       : int   -- total parameters,
            "frozen"      : int   -- frozen parameters,
            "trainable"   : int   -- trainable parameters,
            "savings_pct" : float -- % parameters NOT updated (frozen %)
        }
    """
    # TODO:
    # total      = sum(layer_sizes)
    # frozen     = sum(layer_sizes[i] for i in frozen_layers)
    # trainable  = total - frozen
    # savings_pct = 100 * frozen / total
    pass  # Replace with your implementation


# A simple model with 4 layers
layers = [50000, 50000, 50000, 50000]   # 200k total params

scenarios = [
    ("No freezing",          []),
    ("Freeze embeddings",    [0]),
    ("Freeze first 2 layers",[0, 1]),
    ("Freeze all but last",  [0, 1, 2]),
]

print(f"  Total layers: {len(layers)}, params each: {layers[0]:,}")
print()
print(f"  {'Scenario':<28} {'Total':>8}  {'Frozen':>8}  {'Trainable':>10}  {'Savings':>8}")
print("  " + "-" * 68)

for name, frozen_idx in scenarios:
    r = frozen_analysis(layers, frozen_idx)
    if r:
        print(f"  {name:<28} {r['total']:>8,}  {r['frozen']:>8,}  "
              f"{r['trainable']:>10,}  {r['savings_pct']:>7.1f}%")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Fine-Tuning Cost vs Training from Scratch
#
#  Background:
#    Fine-tuning is dramatically cheaper than training from scratch.
#    Real-world example:
#      GPT-3 pre-training: $12,000,000,  570 GB data, weeks
#      GPT-3 fine-tuning:  $100,         100 MB data, hours
#
#    Cost ratio: pretrain_cost / finetune_cost
#    Data ratio: pretrain_data_gb / finetune_data_gb
#
#  Your Task:
#    Write: cost_comparison(pretrain_cost, finetune_cost,
#                            pretrain_data_gb, finetune_data_gb) -> dict
#    Returns: {"cost_ratio": float, "data_ratio": float,
#              "finetune_pct_cost": float}
#    finetune_pct_cost = 100 * finetune_cost / pretrain_cost
#
#  C# Analogy:
#    Like comparing cloud compute bills: how much % of the full training run
#    does fine-tuning cost?
# ============================================================

print("-" * 50)
print("EXERCISE 4: Fine-Tuning Cost Comparison")
print("-" * 50)
print()


def cost_comparison(pretrain_cost, finetune_cost, pretrain_data_gb, finetune_data_gb):
    """
    Compare cost and data requirements of pre-training vs fine-tuning.

    Parameters:
        pretrain_cost    (float): Cost to pre-train (e.g., dollars).
        finetune_cost    (float): Cost to fine-tune.
        pretrain_data_gb (float): Data used for pre-training (GB).
        finetune_data_gb (float): Data used for fine-tuning (GB).

    Returns:
        dict: {
            "cost_ratio"       : float -- pretrain / finetune (how many times cheaper is FT),
            "data_ratio"       : float -- pretrain / finetune (how much less data FT needs),
            "finetune_pct_cost": float -- finetune cost as % of pretrain cost
        }
    """
    # TODO:
    # cost_ratio        = pretrain_cost    / finetune_cost
    # data_ratio        = pretrain_data_gb / finetune_data_gb
    # finetune_pct_cost = 100 * finetune_cost / pretrain_cost
    pass  # Replace with your implementation


# GPT-3 real numbers
r = cost_comparison(
    pretrain_cost    = 12_000_000,   # $12M
    finetune_cost    = 100,          # $100
    pretrain_data_gb = 570,          # 570 GB
    finetune_data_gb = 0.1           # 100 MB = 0.1 GB
)

if r:
    print(f"  GPT-3 Example:")
    print(f"    Fine-tuning is {r['cost_ratio']:,.0f}x cheaper than pre-training")
    print(f"    Fine-tuning needs {r['data_ratio']:,.0f}x less data")
    print(f"    Fine-tuning costs {r['finetune_pct_cost']:.6f}% of pre-training cost")
print()
print("  Expected: cost_ratio ~120000x, data_ratio ~5700x")
print()

print("=" * 60)
print("All exercises complete!")
print("=" * 60)

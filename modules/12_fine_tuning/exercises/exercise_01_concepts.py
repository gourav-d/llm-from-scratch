"""
Module 12 - Fine-Tuning LLMs
Exercise 01: Core Fine-Tuning Concepts

GLOSSARY
--------
Weight Matrix  : A 2D array of numbers that transforms inputs to outputs.
                 Like a lookup table of multipliers in a neural network.
Delta W        : The change applied to a weight matrix during an update step.
                 Like a diff/patch applied to a configuration file.
Update Magnitude: How big the weight change is relative to the original weights.
                 Measured as: norm(delta_W) / norm(W). Should be small (<5%).
Norm           : The "size" or "length" of a vector or matrix.
                 Frobenius norm for matrices = sqrt(sum of all squares).
                 Like the magnitude of a vector in 3D space.
Loss Curve     : A list of loss values recorded after each training epoch.
                 Plotting it shows whether the model is learning or struggling.
Overfitting    : Training loss keeps going down, but validation loss goes up.
                 Model memorises training data; fails on new examples.
LoRA           : Low-Rank Adaptation -- only train two small matrices (A, B)
                 instead of the full weight matrix W.
Rank           : The "bottleneck" dimension in LoRA. Lower rank = fewer params.
                 Like using a compressed representation instead of full data.
Label Balance  : Having roughly equal numbers of examples per class.
                 Imbalanced data (e.g., 90% class A, 5% B, 5% C) causes bias.
"""

import numpy as np   # NumPy: math and arrays (like System.Math + 2D arrays in C#)

print("=" * 60)
print("Exercise 01: Fine-Tuning Concepts")
print("=" * 60)
print()

# ============================================================
#  EXERCISE 1
#  Topic: Weight Updates and Update Magnitude
#
#  Background:
#    During fine-tuning, we update weights like this:
#      W_new = W - learning_rate * gradient
#    A healthy fine-tuning run updates weights by a SMALL amount.
#    We measure this with "update magnitude":
#      update_magnitude = norm(delta_W) / norm(W)
#    This should be < 5% (i.e., < 0.05) for stable fine-tuning.
#
#  Your Task:
#    1. Given W and delta_W, compute W_new = W + delta_W.
#       (Note: we add here because delta_W already has the sign from the optimiser.)
#    2. Compute update_magnitude = norm(delta_W) / norm(W).
#    3. Print whether the magnitude is "small (OK)" or "large (risky)".
#
#  C# Analogy:
#    Like checking a git diff -- if a single patch changes 50% of the file,
#    that is a risky commit. Small, focused patches are safer.
# ============================================================

print("-" * 50)
print("EXERCISE 1: Weight Updates and Update Magnitude")
print("-" * 50)
print()

# Given a pretrained weight matrix W (3x3)
np.random.seed(0)                          # Fix seed so results are reproducible
W = np.random.randn(3, 3) * 2.0           # Pretrained weights (larger values)

# A small gradient-based update (learning_rate * gradient is already baked in)
delta_W_small = np.random.randn(3, 3) * 0.05    # Tiny update (should be <5%)
delta_W_large = np.random.randn(3, 3) * 3.0     # Large update (risky)

print("  W (original weights):")
print(W)
print()

# TODO: Complete the following steps for BOTH delta_W_small and delta_W_large:
# 1. Compute W_new = W + delta_W
# 2. Compute update_magnitude = np.linalg.norm(delta_W) / np.linalg.norm(W)
# 3. Print whether it is "small (OK)" or "large (risky)" (threshold = 0.05)

# YOUR CODE HERE (Exercise 1)
# --- Remove the 'pass' below and write your solution ---
pass


# ============================================================
#  EXERCISE 2
#  Topic: Detecting Overfitting from Loss Curves
#
#  Background:
#    Overfitting happens when:
#      - training loss DECREASES (model learns the train data)
#      - validation loss INCREASES (model fails on new data)
#    If BOTH losses decrease, the model is still generalising -- good!
#    If BOTH losses plateau, the model has stopped learning -- try higher LR.
#
#  Your Task:
#    Write: detect_overfitting(train_losses, val_losses) -> bool
#    Returns True if, in the LAST 2 epochs:
#      - train_losses went DOWN (train_losses[-1] < train_losses[-3])
#      - val_losses   went UP   (val_losses[-1]   > val_losses[-3])
#    Otherwise return False.
#
#  C# Analogy:
#    Like a health monitor that raises an alert when CPU goes down
#    (train is running efficiently) but memory goes up (leaking / memorising).
# ============================================================

print("-" * 50)
print("EXERCISE 2: Detect Overfitting from Loss Curves")
print("-" * 50)
print()

# Test curves
good_train_losses = [1.2, 0.9, 0.6, 0.4, 0.3]    # Both losses improving
good_val_losses   = [1.3, 1.0, 0.8, 0.6, 0.5]

overfit_train_losses = [1.2, 0.9, 0.6, 0.4, 0.2]  # Train keeps improving
overfit_val_losses   = [1.3, 1.0, 0.8, 0.9, 1.1]  # But val loss is rising

def detect_overfitting(train_losses, val_losses):
    """
    Detect whether the model is overfitting by checking the last 2 epochs.
    Returns True if overfitting detected, False otherwise.

    Parameters:
        train_losses (list): Training loss per epoch.
        val_losses   (list): Validation loss per epoch.

    Returns:
        bool: True if overfitting, False if not.
    """
    # TODO: Complete this function.
    # Hint: Compare [-1] to [-3] for both lists.
    #       If we need at least 3 values, add a guard: if len < 3: return False
    pass  # Replace this with your implementation


# Test your function
print(f"  Good training   -> overfitting detected? {detect_overfitting(good_train_losses, good_val_losses)}")
print(f"  Overfit example -> overfitting detected? {detect_overfitting(overfit_train_losses, overfit_val_losses)}")
print()
print("  Expected: False, True")
print()


# ============================================================
#  EXERCISE 3
#  Topic: LoRA Parameter Savings
#
#  Background:
#    A full weight matrix W has shape (out_dim x in_dim).
#    Full params = out_dim * in_dim.
#    LoRA replaces W with: W + scale * (B @ A)
#      A shape: (rank x in_dim)   -- small matrix
#      B shape: (out_dim x rank)  -- small matrix
#    LoRA trainable params = rank * in_dim + out_dim * rank
#                          = rank * (in_dim + out_dim)
#    savings_pct = 100 * (1 - lora_params / full_params)
#
#  Your Task:
#    Write: count_lora_params(in_dim, out_dim, rank) -> dict
#    Returns: {"full_params": ..., "lora_params": ..., "savings_pct": ...}
#
#  C# Analogy:
#    Like comparing a full 4K texture vs a compressed version --
#    LoRA is the compressed adapter that takes far less storage.
# ============================================================

print("-" * 50)
print("EXERCISE 3: LoRA Parameter Count and Savings")
print("-" * 50)
print()

def count_lora_params(in_dim, out_dim, rank):
    """
    Calculate full vs LoRA parameter counts and % savings.

    Parameters:
        in_dim  (int): Input dimension of the weight matrix.
        out_dim (int): Output dimension of the weight matrix.
        rank    (int): LoRA rank (bottleneck size).

    Returns:
        dict: {"full_params": int, "lora_params": int, "savings_pct": float}
    """
    # TODO: Compute full_params, lora_params, savings_pct
    # full_params  = out_dim * in_dim
    # lora_params  = rank * (in_dim + out_dim)
    # savings_pct  = 100 * (1 - lora_params / full_params)
    pass  # Replace with your implementation


# Test with a typical LLM attention layer size
result = count_lora_params(in_dim=768, out_dim=768, rank=8)
if result:
    print(f"  in_dim=768, out_dim=768, rank=8")
    print(f"  Full params  : {result['full_params']:,}")
    print(f"  LoRA params  : {result['lora_params']:,}")
    print(f"  Savings      : {result['savings_pct']:.1f}%")
    print()

    # Also print for several ranks
    print(f"  {'Rank':>6} {'Full':>12} {'LoRA':>10} {'Savings':>10}")
    print("  " + "-" * 42)
    for r in [1, 2, 4, 8, 16, 32]:
        res = count_lora_params(768, 768, r)
        if res:
            print(f"  {r:>6} {res['full_params']:>12,} "
                  f"{res['lora_params']:>10,} {res['savings_pct']:>9.1f}%")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Checking Label Balance in a Training Dataset
#
#  Background:
#    If your training data has 90% of one class and 5% each of two others,
#    the model will learn to always predict the majority class.
#    This is called "class imbalance".
#    Rule of thumb: warn if any class has < 10% of total examples.
#
#  Your Task:
#    Write: check_label_balance(data) -> None
#    data is a list of dicts: [{"text": ..., "label": ...}, ...]
#    Print count and % for each label.
#    Print a WARNING if any label has < 10% share.
#
#  C# Analogy:
#    Like running a data quality check before a database migration --
#    you want to catch skewed distributions before they cause problems.
# ============================================================

print("-" * 50)
print("EXERCISE 4: Check Label Balance")
print("-" * 50)
print()

# Balanced dataset (all classes roughly equal)
balanced_data = (
    [{"text": "good", "label": "POS"}] * 10 +   # 10 positive examples
    [{"text": "bad",  "label": "NEG"}] * 10 +   # 10 negative examples
    [{"text": "meh",  "label": "NEU"}] * 10      # 10 neutral examples
)

# Imbalanced dataset (NEU is rare)
imbalanced_data = (
    [{"text": "good", "label": "POS"}] * 40 +   # 40 positive (too many!)
    [{"text": "bad",  "label": "NEG"}] * 38 +   # 38 negative
    [{"text": "meh",  "label": "NEU"}] * 2       # 2 neutral (only 2.5%! bad)
)

def check_label_balance(data):
    """
    Print label distribution and warn about under-represented classes.

    Parameters:
        data (list): List of dicts with a "label" key.

    Returns:
        None (just prints output)
    """
    # TODO: Complete this function.
    # Steps:
    # 1. Count occurrences of each label using a dict (like Dictionary<string,int>).
    # 2. Compute total = len(data)
    # 3. For each label, print: "  LABEL: N examples (X.X%)"
    # 4. If any label has count/total < 0.10, print: "  WARNING: label X is underrepresented (X.X%)"
    pass  # Replace with your implementation


print("  --- Balanced dataset ---")
check_label_balance(balanced_data)
print()
print("  --- Imbalanced dataset ---")
check_label_balance(imbalanced_data)
print()


# ============================================================
#  SOLUTIONS  (commented out -- try it yourself first!)
# ============================================================

"""
# ---- SOLUTION: Exercise 1 ----

for name, delta_W in [("Small update", delta_W_small), ("Large update", delta_W_large)]:
    W_new = W + delta_W                              # Apply the update
    update_magnitude = np.linalg.norm(delta_W) / np.linalg.norm(W)  # Relative size
    status = "small (OK)" if update_magnitude < 0.05 else "large (risky)"
    print(f"  {name}:")
    print(f"    delta_W norm      : {np.linalg.norm(delta_W):.4f}")
    print(f"    W norm            : {np.linalg.norm(W):.4f}")
    print(f"    Update magnitude  : {update_magnitude:.4f} ({update_magnitude*100:.1f}%) -- {status}")
    print()


# ---- SOLUTION: Exercise 2 ----

def detect_overfitting(train_losses, val_losses):
    if len(train_losses) < 3 or len(val_losses) < 3:   # Need at least 3 data points
        return False                                      # Not enough history to judge
    train_decreased = train_losses[-1] < train_losses[-3]  # Train loss went down
    val_increased   = val_losses[-1]   > val_losses[-3]    # Val loss went up
    return train_decreased and val_increased                # Both conditions = overfitting


# ---- SOLUTION: Exercise 3 ----

def count_lora_params(in_dim, out_dim, rank):
    full_params  = out_dim * in_dim                     # Full weight matrix size
    lora_params  = rank * (in_dim + out_dim)            # A matrix + B matrix
    savings_pct  = 100.0 * (1.0 - lora_params / full_params)  # % reduction
    return {
        "full_params"  : full_params,
        "lora_params"  : lora_params,
        "savings_pct"  : savings_pct,
    }


# ---- SOLUTION: Exercise 4 ----

def check_label_balance(data):
    counts = {}                                          # like Dictionary<string,int>
    for item in data:                                    # Loop over all examples
        lbl = item["label"]                              # Get the label
        counts[lbl] = counts.get(lbl, 0) + 1            # Increment count (default 0)
    total = len(data)                                    # Total number of examples
    for lbl, cnt in sorted(counts.items()):             # Print each label sorted
        pct = cnt / total * 100                          # Percentage of total
        print(f"    {lbl}: {cnt} examples ({pct:.1f}%)")
        if pct < 10.0:                                   # Below the 10% threshold?
            print(f"    WARNING: '{lbl}' is underrepresented ({pct:.1f}%)")
"""

"""
Exercise 01: Perplexity and Overfitting Detection
Module 17: LLM Evaluation & Benchmarks

TASKS:
  1. Implement compute_perplexity(loss)              -- loss to perplexity
  2. Implement detect_overfitting(train, val)        -- find divergence epoch
  3. Implement bits_per_character(perplexity)        -- BPC metric
  4. Implement overfitting_gap(train_losses, val_losses) -- max gap found

Run:  python exercise_01_perplexity.py
Deps: none (pure Python)
"""

import math


# ─────────────────────────────────────────────────────────
# TASK 1: Compute Perplexity
# ─────────────────────────────────────────────────────────

def compute_perplexity(loss):
    """
    Convert cross-entropy loss to perplexity.

    Formula: perplexity = exp(loss)

    Perplexity intuition:
      - perplexity = 1    → perfect prediction
      - perplexity = 10   → choosing between 10 equally likely words
      - perplexity = 1000 → model is very uncertain

    Args:
        loss: float, average cross-entropy loss (must be >= 0)

    Returns:
        float: perplexity value

    Example:
        compute_perplexity(0.0) -> 1.0    (perfect model)
        compute_perplexity(2.0) -> 7.389  (choosing between ~7 words)
        compute_perplexity(3.5) -> 33.1   (choosing between ~33 words)

    HINT:
        return math.exp(loss)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Detect Overfitting
# ─────────────────────────────────────────────────────────

def detect_overfitting(train_losses, val_losses, patience=2):
    """
    Detect overfitting from loss curves.

    Overfitting occurs when validation loss increases for `patience`
    consecutive epochs WHILE training loss is still decreasing.

    Args:
        train_losses: list of float, training loss per epoch
        val_losses:   list of float, validation loss per epoch
        patience:     int, how many consecutive val loss increases trigger detection

    Returns:
        int or None: epoch number (1-indexed) when overfitting starts,
                     or None if no overfitting detected.

    Example:
        train = [4.0, 3.0, 2.0, 1.5, 1.0, 0.8]
        val   = [4.1, 3.2, 2.4, 2.6, 2.9, 3.2]  <- rises at epoch 4
        detect_overfitting(train, val, patience=2) -> 5   (rose 2 times by epoch 5)

        train = [4.0, 3.0, 2.0, 1.5, 1.0]
        val   = [4.1, 3.1, 2.1, 1.6, 1.1]   <- tracks training
        detect_overfitting(train, val, patience=2) -> None

    HINT:
        best_val = val_losses[0]
        worse_streak = 0
        for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
            if vl < best_val:
                best_val = vl
                worse_streak = 0
            else:
                worse_streak += 1
            if worse_streak >= patience:
                return epoch
        return None
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Bits Per Character
# ─────────────────────────────────────────────────────────

def bits_per_character(perplexity):
    """
    Convert perplexity to bits-per-character (BPC).

    BPC = log2(perplexity)

    BPC is another way to express perplexity, common in character-level models.
    Lower is better:
      BPC = 1.0 → near-perfect model
      BPC = 2.0 → model is choosing between 4 options (2^2)
      BPC = 8.0 → model is choosing from 256 options

    Args:
        perplexity: float, perplexity value (must be >= 1)

    Returns:
        float: bits-per-character

    Example:
        bits_per_character(2.0)  -> 1.0   (log2(2) = 1)
        bits_per_character(4.0)  -> 2.0   (log2(4) = 2)
        bits_per_character(32.0) -> 5.0   (log2(32) = 5)

    HINT:
        return math.log2(perplexity)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Max Train/Val Loss Gap
# ─────────────────────────────────────────────────────────

def overfitting_gap(train_losses, val_losses):
    """
    Find the maximum gap between validation and training loss across all epochs.

    A large gap means the model performs much worse on unseen data.
    Gap > 0.5 is a warning sign; gap > 1.0 indicates clear overfitting.

    Args:
        train_losses: list of float, training loss per epoch
        val_losses:   list of float, validation loss per epoch

    Returns:
        float: maximum (val_loss - train_loss) observed across all epochs

    Example:
        train = [3.0, 2.0, 1.0]
        val   = [3.1, 2.5, 2.0]
        overfitting_gap(train, val) -> 1.0   (epoch 3: 2.0 - 1.0 = 1.0)

    HINT:
        gaps = [v - t for t, v in zip(train_losses, val_losses)]
        return max(gaps)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 01: Perplexity and Overfitting Detection")
    print("=" * 55)

    # Task 1: compute_perplexity
    print("\n--- Task 1: compute_perplexity ---")
    cases = [
        (0.0, 1.0),
        (2.0, math.exp(2.0)),
        (3.5, math.exp(3.5)),
        (1.2, math.exp(1.2)),
    ]
    for loss, expected in cases:
        result = compute_perplexity(loss)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if abs(result - expected) < 0.01 else "FAIL"
        print(f"  {status}  perplexity(loss={loss}) = {result:.3f}  (expected {expected:.3f})")

    # Task 2: detect_overfitting
    print("\n--- Task 2: detect_overfitting ---")
    train_good = [4.0, 3.0, 2.0, 1.5, 1.2, 1.0]
    val_good   = [4.1, 3.1, 2.1, 1.6, 1.3, 1.1]
    train_bad  = [4.0, 3.0, 2.0, 1.5, 1.0, 0.8]
    val_bad    = [4.1, 3.2, 2.4, 2.6, 2.9, 3.2]

    r_good = detect_overfitting(train_good, val_good, patience=2)
    if r_good is None and compute_perplexity(1.0) is not None:
        # Implemented but returned None (correct for good training)
        print(f"  PASS  Good training: overfitting detected at epoch {r_good}  (expected None)")
    elif r_good is not None:
        status = "FAIL"
        print(f"  {status}  Good training: returned {r_good}  (expected None)")
    else:
        print("  NOT IMPLEMENTED YET")

    r_bad = detect_overfitting(train_bad, val_bad, patience=2)
    if r_bad is None:
        print("  NOT IMPLEMENTED YET")
    else:
        status = "PASS" if r_bad is not None else "FAIL"
        print(f"  {status}  Overfitting training: detected at epoch {r_bad}  (expected around 5)")

    # Task 3: bits_per_character
    print("\n--- Task 3: bits_per_character ---")
    bpc_cases = [(2.0, 1.0), (4.0, 2.0), (32.0, 5.0), (1.0, 0.0)]
    for ppl, expected in bpc_cases:
        result = bits_per_character(ppl)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if abs(result - expected) < 0.01 else "FAIL"
        print(f"  {status}  BPC(perplexity={ppl}) = {result:.3f}  (expected {expected:.3f})")

    # Task 4: overfitting_gap
    print("\n--- Task 4: overfitting_gap ---")
    t = [3.0, 2.0, 1.0]
    v = [3.1, 2.5, 2.0]
    result_gap = overfitting_gap(t, v)
    if result_gap is None:
        print("  NOT IMPLEMENTED YET")
    else:
        expected_gap = 1.0
        status = "PASS" if abs(result_gap - expected_gap) < 0.01 else "FAIL"
        print(f"  {status}  max gap = {result_gap:.3f}  (expected {expected_gap:.3f})")

        t2 = [4.0, 3.0, 2.0, 1.5]
        v2 = [4.1, 3.2, 2.8, 2.5]
        r2 = overfitting_gap(t2, v2)
        print(f"  INFO  Example 2: max gap = {r2:.3f}  (val-train at worst epoch)")

    # Bonus: full perplexity table
    print("\n--- BONUS: Perplexity Reference Table ---")
    if compute_perplexity(1.0) is not None and bits_per_character(2.0) is not None:
        print(f"\n  {'Loss':>6}  {'Perplexity':>12}  {'BPC':>8}  {'Stage'}")
        print("  " + "-" * 42)
        stages = [
            (0.5, "Near-perfect"),
            (1.2, "Excellent"),
            (2.0, "Good"),
            (3.0, "Fair"),
            (3.5, "Still learning"),
            (5.0, "Struggling"),
            (10.8, "Random baseline"),
        ]
        for loss, stage in stages:
            ppl = compute_perplexity(loss)
            bpc = bits_per_character(ppl)
            print(f"  {loss:>6.1f}  {ppl:>12.1f}  {bpc:>8.2f}  {stage}")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def compute_perplexity(loss):
#     return math.exp(loss)
#
# def detect_overfitting(train_losses, val_losses, patience=2):
#     best_val = float("inf")
#     worse_streak = 0
#     for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
#         if vl < best_val:
#             best_val = vl
#             worse_streak = 0
#         else:
#             worse_streak += 1
#         if worse_streak >= patience:
#             return epoch
#     return None
#
# def bits_per_character(perplexity):
#     return math.log2(perplexity)
#
# def overfitting_gap(train_losses, val_losses):
#     gaps = [v - t for t, v in zip(train_losses, val_losses)]
#     return max(gaps)

"""
Example 01: Perplexity, Loss Curves, and Overfitting Detection
Module 17: LLM Evaluation & Benchmarks

Run:  python example_01_perplexity.py
Deps: none (pure Python)
"""

import math

print("=" * 60)
print("  Example 01: Perplexity and Loss Curves")
print("=" * 60)


# ─────────────────────────────────────────────────────────
# DEMO 1: What Is Perplexity?
# ─────────────────────────────────────────────────────────

print("\n--- DEMO 1: Loss → Perplexity conversion ---")
print()

def compute_perplexity(loss):
    """Convert cross-entropy loss to perplexity using exp()."""
    return math.exp(loss)

# Various models at different training stages
loss_examples = [
    ("Random (untrained) model", 10.8),   # log(50000) ~ 10.8, vocab size 50000
    ("Early training",           3.5),
    ("Mid training",             2.5),
    ("Good model (GPT-2 small)", 3.38),   # real number
    ("Strong model (GPT-3 175B)", 2.30),  # real number - perplexity ~10
    ("Excellent model",          1.2),
]

print(f"  {'Model':<30}  {'Loss':>6}  {'Perplexity':>12}  {'Interpretation'}")
print("  " + "-" * 80)

for name, loss in loss_examples:
    ppl = compute_perplexity(loss)
    if ppl > 1000:
        interp = "Completely random"
    elif ppl > 100:
        interp = "Mostly random"
    elif ppl > 30:
        interp = "Struggling — still learning"
    elif ppl > 10:
        interp = "Learning, not great yet"
    elif ppl > 5:
        interp = "Good — usable model"
    else:
        interp = "Excellent — near human level"
    print(f"  {name:<30}  {loss:>6.2f}  {ppl:>12.1f}  {interp}")

print()
print("  Intuition: perplexity = how many words model chooses between on average.")
print("  Perplexity 10 → model narrows to 10 equally likely candidates per token.")


# ─────────────────────────────────────────────────────────
# DEMO 2: Reading a Loss Curve — Good Training vs Overfitting
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 2: Loss Curves — Good Training vs Overfitting ---")

def simulate_good_training(epochs=10):
    """Simulate training where both losses decrease together."""
    train_losses = []
    val_losses   = []
    for e in range(1, epochs + 1):
        # Both decrease, validation stays close to training
        train = 4.0 * math.exp(-0.25 * e) + 1.5
        val   = train + 0.05 + 0.02 * e / epochs
        train_losses.append(round(train, 3))
        val_losses.append(round(val, 3))
    return train_losses, val_losses

def simulate_overfitting(epochs=10):
    """Simulate training where val loss starts rising after epoch 5."""
    train_losses = []
    val_losses   = []
    for e in range(1, epochs + 1):
        train = 4.0 * math.exp(-0.35 * e) + 0.5
        if e <= 5:
            val = train + 0.1
        else:
            # Divergence begins — model memorizing training data
            val = (train + 0.1) + 0.2 * (e - 5)
        train_losses.append(round(train, 3))
        val_losses.append(round(val, 3))
    return train_losses, val_losses

print()
print("  SCENARIO A — Good Training:")
print(f"  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val PPL':>9}  {'Status'}")
print("  " + "-" * 55)
train_g, val_g = simulate_good_training()
for e, (tl, vl) in enumerate(zip(train_g, val_g), 1):
    ppl = compute_perplexity(vl)
    gap = vl - tl
    status = "OK" if gap < 0.5 else "WARNING"
    print(f"  {e:>6}  {tl:>11.3f}  {vl:>10.3f}  {ppl:>9.1f}  {status}")

print()
print("  SCENARIO B — Overfitting (val loss diverges at epoch 5):")
print(f"  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Val PPL':>9}  {'Status'}")
print("  " + "-" * 55)
train_o, val_o = simulate_overfitting()
for e, (tl, vl) in enumerate(zip(train_o, val_o), 1):
    ppl = compute_perplexity(vl)
    gap = vl - tl
    status = "OVERFITTING" if gap > 0.5 else "OK"
    print(f"  {e:>6}  {tl:>11.3f}  {vl:>10.3f}  {ppl:>9.1f}  {status}")


# ─────────────────────────────────────────────────────────
# DEMO 3: Automatic Overfitting Detection
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 3: Automatic Overfitting Detection ---")
print()

def detect_overfitting(train_losses, val_losses, patience=3, min_gap=0.5):
    """
    Return the epoch at which overfitting starts, or None if not detected.

    Two signals:
      1. Val loss is rising (not decreasing) for `patience` consecutive epochs.
      2. Gap (val - train) exceeds min_gap.
    """
    best_val = float("inf")
    worse_streak = 0
    for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
        gap = vl - tl
        if vl < best_val:
            best_val = vl
            worse_streak = 0
        else:
            worse_streak += 1

        if worse_streak >= patience and gap > min_gap:
            return epoch
    return None

epoch_good = detect_overfitting(train_g, val_g)
epoch_over = detect_overfitting(train_o, val_o)

print(f"  Good training  → Overfitting detected at epoch: {epoch_good}")
print(f"  Overfitting run → Overfitting detected at epoch: {epoch_over}")
print()

# Bits per character — alternative perplexity measure
def bits_per_character(perplexity):
    """Convert perplexity to bits-per-character (BPC). log2(ppl)."""
    return math.log2(perplexity)

print("  Bits-per-character (BPC) — compact form of perplexity:")
print(f"  {'Loss':>6}  {'Perplexity':>12}  {'BPC':>8}")
print("  " + "-" * 30)
for loss in [1.2, 2.0, 3.0, 4.0]:
    ppl = compute_perplexity(loss)
    bpc = bits_per_character(ppl)
    print(f"  {loss:>6.1f}  {ppl:>12.1f}  {bpc:>8.2f}")
print()
print("  BPC < 2 = very good character-level model.")
print("  BPC = 1 = near-perfect model.")

print()
print("  KEY TAKEAWAYS:")
print("  1. Perplexity = exp(loss) -- lower always better.")
print("  2. Good training: train loss and val loss both fall together.")
print("  3. Overfitting: train loss falls, val loss rises -- stop early!")
print("  4. BPC = log2(perplexity) -- same info, different scale.")

"""
Example 05: Multi-Token Prediction (MTP)
Module 16: Modern LLM Architectures

Demonstrates how MTP trains a model to predict multiple future tokens simultaneously:
  - Standard next-token prediction (N=1 head)
  - MTP with N=4 heads predicting t+1, t+2, t+3, t+4
  - Richer gradient signal: more loss signals per training step
  - Speculative decoding speedup calculation

Run:  python example_05_mtp.py
Deps: none (pure Python)
"""

import math
import random


def print_section(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


def softmax(logits):
    """Convert logits to probabilities. Subtract max for numerical stability."""
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy(probs, true_index):
    """Cross-entropy loss: -log(P(correct token))."""
    p = max(probs[true_index], 1e-10)
    return -math.log(p)


# ─────────────────────────────────────────────────────────
# DEMO 1: STANDARD NEXT-TOKEN PREDICTION
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: Standard Next-Token Prediction (1 Head)")

print("""
Standard GPT training:
  At each position, predict ONE next token.
  Loss = cross-entropy(predicted, next_token)
  One gradient signal per position.
""")

vocab = ["The", "cat", "sat", "on", "the", "mat", "dog", "ran", "a", "big"]
sentence = ["The", "cat", "sat", "on", "the", "mat"]
token_ids = [vocab.index(w) for w in sentence]

def fake_head_output(true_id, vocab_size, confidence, seed=0):
    """
    Fake model output for one prediction head.
    Returns logits where the true_id gets a high score.
    """
    rng = random.Random(seed)
    logits = [rng.uniform(-3, 0) for _ in range(vocab_size)]
    logits[true_id] += math.log(confidence / (1.0 - confidence + 1e-8)) + 3
    return logits


print("  Training on: 'The cat sat on the mat'")
print(f"  {'Position':>10}  {'Token':>6}  {'Predict':>8}  {'Loss':>8}  {'Grad signals'}")
print("  " + "-" * 58)

standard_losses = []
for i in range(len(sentence) - 1):
    current = sentence[i]
    next_tok = sentence[i + 1]
    next_id = token_ids[i + 1]
    conf = 0.6 + random.Random(i).random() * 0.3
    logits = fake_head_output(next_id, len(vocab), conf, seed=i)
    probs = softmax(logits)
    loss = cross_entropy(probs, next_id)
    standard_losses.append(loss)
    print(f"  {i:>10}  {current:>6}  {next_tok:>8}  {loss:>8.3f}  1 (t+1 only)")

total_std = sum(standard_losses)
print(f"\n  Total loss:       {total_std:.3f}")
print(f"  Gradient signals: {len(standard_losses)} (one per position)")
print(f"  Avg loss:         {total_std/len(standard_losses):.3f}")


# ─────────────────────────────────────────────────────────
# DEMO 2: MTP -- MULTIPLE PREDICTION HEADS
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: MTP -- 4 Prediction Heads (t+1, t+2, t+3, t+4)")

print("""
MTP adds N output heads on top of the SAME transformer backbone.
At each position, ALL N heads run simultaneously.
Each head predicts a different future token: t+1, t+2, t+3, t+4.

Loss = sum of cross-entropy losses from all N heads.
The SAME transformer weights get gradients from ALL N losses.
This is the "richer gradient signal" MTP provides.
""")

N_HEADS = 4    # number of prediction heads
# Each head i predicts token at offset (i+1) from current position
head_offsets = list(range(1, N_HEADS + 1))   # [1, 2, 3, 4]

print(f"  N = {N_HEADS} heads predicting offsets: {head_offsets}")
print(f"  Sentence: {sentence}")
print()

mtp_total_loss = 0.0
total_grad_signals = 0

print(f"  {'Pos':>5}  {'Token':>6}  {'Head':>6}  {'Predicts':>10}  {'Loss':>8}")
print("  " + "-" * 50)

for i in range(len(sentence) - N_HEADS):
    current = sentence[i]
    head_losses = []
    for h, offset in enumerate(head_offsets):
        target_pos = i + offset
        if target_pos >= len(sentence):
            break
        target_tok = sentence[target_pos]
        target_id = token_ids[target_pos]

        # Each head has slightly different confidence -- they share the transformer
        # but each head specializes in its prediction horizon
        conf = 0.75 - h * 0.05 + random.Random(i * 10 + h).random() * 0.1
        logits = fake_head_output(target_id, len(vocab), conf, seed=i * 100 + h)
        probs = softmax(logits)
        loss = cross_entropy(probs, target_id)
        head_losses.append(loss)
        mtp_total_loss += loss
        total_grad_signals += 1

        head_label = f"head{h+1}(t+{offset})"
        print(f"  {i:>5}  {current:>6}  {head_label:>10}  {target_tok:>10}  {loss:>8.3f}")

    pos_total = sum(head_losses)
    print(f"        position {i} total: {pos_total:.3f}  ({len(head_losses)} heads, {len(head_losses)} grad signals)")
    print()

print(f"  MTP total loss:       {mtp_total_loss:.3f}")
print(f"  MTP gradient signals: {total_grad_signals}")
print(f"  Standard grad signals: {len(standard_losses)}")
print(f"  Ratio:                {total_grad_signals / len(standard_losses):.1f}x more gradient signals")
print("""
  Key insight: with N=4 heads, the transformer receives ~4x more gradient
  information per training step. The shared backbone learns richer representations
  because it must satisfy predictions at multiple future horizons simultaneously.
""")


# ─────────────────────────────────────────────────────────
# DEMO 3: RICHER GRADIENT SIGNAL -- WHY IT HELPS
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: Richer Gradient Signal -- What the Model Must Learn")

print("""
With standard training, at position 'The':
  Model only learns: "after 'The', 'cat' is likely"

With MTP N=4, at position 'The':
  Head 1 learns: "after 'The', 'cat' is likely"            (t+1)
  Head 2 learns: "after 'The cat', 'sat' is likely"        (t+2)
  Head 3 learns: "after 'The cat sat', 'on' is likely"     (t+3)
  Head 4 learns: "after 'The cat sat on', 'the' is likely" (t+4)

The SHARED transformer must build representations that are useful for ALL 4 predictions.
This forces it to capture longer-range syntactic and semantic structure.
""")

print("  Training position 'The' -- gradient contributions per head:")
print()

position = "The"
targets = ["cat", "sat", "on", "the"]    # t+1, t+2, t+3, t+4
for i, (head, target) in enumerate(zip(range(1, 5), targets)):
    # Later heads have slightly lower confidence (harder to predict far ahead)
    conf = 0.80 - i * 0.08
    loss = -math.log(conf)
    gradient_magnitude = loss    # loss ~ gradient magnitude (simplified)
    bar = "#" * int(gradient_magnitude * 20)
    print(f"  Head {head} (t+{head}): predicts '{target}' | loss={loss:.3f} | grad |{bar}|")

print("""
  All 4 gradients backpropagate through the SAME transformer layers.
  Transformers must produce embeddings useful for predicting 1, 2, 3, 4 steps ahead.
  This forces better long-range features, especially for code and math.

  Meta's results (from MTP paper):
    HumanEval (code): +12% improvement with MTP
    GSM8K (math):     +8% improvement with MTP
    General text:     +2-4% improvement
""")


# ─────────────────────────────────────────────────────────
# DEMO 4: SPECULATIVE DECODING WITH MTP
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: Speculative Decoding -- Faster Inference with MTP")

print("""
At inference time, the extra MTP heads can be used for SPECULATIVE DECODING:

  Step 1: Run one forward pass -> head 1 proposes t+1, head 4 proposes t+4
  Step 2: Verify: is the proposed sequence consistent with the main model?
  Step 3: Accept verified tokens, reject others
  Step 4: Continue from last accepted token

This can generate 2-4 tokens per forward pass instead of just 1.
""")

def simulate_speculative_decoding(n_tokens_to_generate, n_heads, accept_rate=0.75):
    """
    Simulate speculative decoding speedup.
    accept_rate: probability that a speculatively generated token is accepted.
    Returns (actual_tokens_generated, forward_passes_used).
    """
    tokens_generated = 0
    forward_passes = 0

    while tokens_generated < n_tokens_to_generate:
        # One forward pass proposes n_heads tokens
        forward_passes += 1
        proposed = n_heads

        # Each proposed token is accepted with accept_rate probability
        accepted = 0
        for _ in range(proposed):
            if tokens_generated >= n_tokens_to_generate:
                break
            if random.random() < accept_rate:
                accepted += 1
                tokens_generated += 1
            else:
                # First rejection: stop, regenerate from here
                tokens_generated += 1    # the rejected token becomes the reroll
                break

        if accepted == 0:
            tokens_generated += 1        # at minimum, generate 1 token per pass

    return tokens_generated, forward_passes


random.seed(42)
n_tokens = 100

print(f"  Generating {n_tokens} tokens with different configurations:")
print()
print(f"  {'Config':<35} {'Forward passes':>15} {'Tokens/pass':>12} {'Speedup':>10}")
print("  " + "-" * 75)

# Standard AR: 1 token per pass
std_passes = n_tokens
print(f"  {'Standard AR (1 head, no speculative)':<35} {std_passes:>15} {n_tokens/std_passes:>12.1f} {'1.0x':>10}")

# MTP with speculative decoding
for n_h in [2, 3, 4]:
    for accept in [0.9, 0.75]:
        _, passes = simulate_speculative_decoding(n_tokens, n_h, accept)
        speedup = std_passes / passes
        label = f"MTP heads={n_h}, accept={accept:.0%}"
        print(f"  {label:<35} {passes:>15} {n_tokens/passes:>12.1f} {speedup:>9.1f}x")

print("""
  With N=4 heads and 90% accept rate: ~3x fewer forward passes.
  Real-world results depend on sequence type:
    - Repetitive/predictable text: high accept rate -> big speedup
    - Creative/unpredictable text: lower accept rate -> smaller speedup
""")


# ─────────────────────────────────────────────────────────
# DEMO 5: MTP LOSS FORMULA
# ─────────────────────────────────────────────────────────

print_section("DEMO 5: MTP Loss Formula")

print("""
  Standard loss:
    L_standard = (1/T) * sum_t CE(head_1(x_t), x_{t+1})

  MTP loss (N heads, weights w_1..w_N):
    L_MTP = (1/T) * sum_t [ w_1 * CE(head_1(x_t), x_{t+1})
                           + w_2 * CE(head_2(x_t), x_{t+2})
                           + ...
                           + w_N * CE(head_N(x_t), x_{t+N}) ]

  Common weighting: uniform (w_i = 1/N) or decreasing (w_i = 1/i).
  Decreasing weight: t+1 prediction is most important, t+N is least.
""")

print("  Loss breakdown with decreasing weights (w_i = 1/i):")
print()
print(f"  {'Head':>6}  {'Offset':>8}  {'Weight':>8}  {'Raw loss':>10}  {'Weighted loss':>14}")
print("  " + "-" * 55)

N = 4
raw_losses = [0.45, 0.62, 0.78, 0.91]      # further ahead = harder = higher loss
total_weighted = 0.0
for h in range(N):
    offset = h + 1
    weight = 1.0 / offset                   # decreasing weight: 1, 0.5, 0.33, 0.25
    weighted = weight * raw_losses[h]
    total_weighted += weighted
    print(f"  {h+1:>6}  {'+' + str(offset):>8}  {weight:>8.3f}  {raw_losses[h]:>10.3f}  {weighted:>14.3f}")

print(f"\n  Total MTP loss: {total_weighted:.3f}")
print(f"  Standard loss (head 1 only): {raw_losses[0]:.3f}")
print(f"""
  Decreasing weights make sense: predicting t+1 is most critical for AR generation.
  t+2, t+3, t+4 provide auxiliary signal but should not dominate.
  If all weights were equal, the model might sacrifice t+1 accuracy for t+4.
""")

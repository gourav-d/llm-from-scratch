"""
Example 04: Diffusion Loss Function (ELBO / Weighted Cross-Entropy)
Module 16: Modern LLM Architectures

Implements and visualizes the MDLM training loss:
  - Weighted cross-entropy on masked positions only
  - How loss varies with noise level t
  - Comparison with standard GPT cross-entropy loss
  - Why we cannot directly maximize log P(x) and need the ELBO

Run:  python example_04_elbo_loss.py
Deps: none (pure Python)
"""

import math
import random


MASK = "[M]"


def print_section(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


# ─────────────────────────────────────────────────────────
# HELPER: SOFTMAX AND CROSS-ENTROPY
# ─────────────────────────────────────────────────────────

def softmax(logits):
    """
    Convert raw scores (logits) to probabilities.
    Subtract max first for numerical stability.

    C# analogy: like normalizing a Dictionary<string, float> of scores
    so all values sum to 1.0 -- turning scores into probabilities.
    """
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]    # subtract max to avoid overflow
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy(probs, true_index):
    """
    Cross-entropy loss: -log(probability of the correct token).

    Lower is better:
      If model assigns prob=0.9 to correct token: loss = -log(0.9) = 0.105
      If model assigns prob=0.1 to correct token: loss = -log(0.1) = 2.303
    """
    p_correct = probs[true_index]
    p_correct = max(p_correct, 1e-10)              # avoid log(0)
    return -math.log(p_correct)


# ─────────────────────────────────────────────────────────
# DEMO 1: WHY NOT MAXIMIZE log P(x) DIRECTLY?
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: Why We Cannot Directly Maximize log P(x)")

print("""
Goal: maximize log P(x_0) where x_0 is real text.

The problem: computing P(x_0) requires marginalizing over ALL possible
noise trajectories from x_0 to x_T:

    P(x_0) = integral over all x_1..x_T of P(x_0 | x_1) * P(x_1 | x_2) * ... * P(x_T)

This integral has exponentially many terms (vocab^T possibilities).
Computing it exactly is INTRACTABLE.

Solution: optimize a LOWER BOUND instead -- the ELBO (Evidence Lower Bound).
    ELBO <= log P(x_0)
    Maximizing ELBO indirectly maximizes log P(x_0).

The ELBO decomposes into tractable terms we CAN compute.
""")

print("  Vocabulary size V=50,000, T=1000 steps:")
V = 50000
T = 1000
# Number of possible trajectories = V^T (astronomically large)
print(f"  Possible noise trajectories = {V}^{T} = 10^{int(T * math.log10(V)):,}")
print(f"  This is 10^{int(T * math.log10(V)):,} terms in the integral.")
print(f"  Computing this exactly = impossible.")
print(f"  ELBO lets us skip this and optimize a tractable approximation instead.")


# ─────────────────────────────────────────────────────────
# DEMO 2: GPT LOSS VS DIFFUSION LOSS
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: GPT Loss vs MDLM Loss -- Side by Side")

print("""
GPT loss: cross-entropy at EVERY position, predict next token.
MDLM loss: cross-entropy at MASKED positions only, predict original token.
""")

# Fake vocabulary and sentence
vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "a", "big", "small"]
sentence_ids = [0, 1, 2, 3, 4]    # "the cat sat on mat"
sentence_words = [vocab[i] for i in sentence_ids]

# Fake model: logits for each position
# Model assigns probabilities to each vocab word
random.seed(42)
def fake_logits(true_id, vocab_size, confidence=0.7):
    """Generate fake logits where true_id gets the highest score."""
    logits = [random.uniform(-2, 0) for _ in range(vocab_size)]
    logits[true_id] = math.log(confidence / (1 - confidence + 1e-8))
    return logits


print(f"  Sentence: {sentence_words}")
print(f"  Vocab size: {len(vocab)}")
print()

# ── GPT Loss ──
print("  --- GPT Loss (predict NEXT token at every position) ---")
print(f"  {'Pos':>4}  {'Token':>8}  {'Predicts':>10}  {'P(correct)':>12}  {'Loss':>8}")
print("  " + "-" * 55)

gpt_total_loss = 0.0
for i in range(len(sentence_ids) - 1):
    current_token = vocab[sentence_ids[i]]
    next_id = sentence_ids[i + 1]
    next_token = vocab[next_id]
    logits = fake_logits(next_id, len(vocab), confidence=0.65 + random.random() * 0.2)
    probs = softmax(logits)
    loss = cross_entropy(probs, next_id)
    gpt_total_loss += loss
    print(f"  {i:>4}  {current_token:>8}  -> {next_token:>8}  {probs[next_id]:>12.3f}  {loss:>8.3f}")

gpt_avg = gpt_total_loss / (len(sentence_ids) - 1)
print(f"\n  GPT total loss:   {gpt_total_loss:.3f}")
print(f"  GPT average loss: {gpt_avg:.3f}  (averaged over ALL positions)")

# ── MDLM Loss (masked positions only) ──
print()
print("  --- MDLM Loss (predict ORIGINAL token at MASKED positions only) ---")

# Simulate: mask tokens 1 and 3 (cat, on)
masked_positions = [1, 3]
masked_tokens = list(sentence_words)
for p in masked_positions:
    masked_tokens[p] = MASK

print(f"  Masked input: {masked_tokens}  (positions {masked_positions} masked)")
print(f"  {'Pos':>4}  {'Masked':>8}  {'Predicts':>10}  {'P(correct)':>12}  {'Loss':>8}  {'Weight(t)':>10}")
print("  " + "-" * 70)

t = 4                        # example timestep
weight_t = 1.0 / (1 - t/T + 1e-8) if t < T else 1.0   # timestep weighting
# Simpler: weight by fraction of tokens masked
weight_t = t / len(sentence_words)

mdlm_total_loss = 0.0
for pos in masked_positions:
    true_id = sentence_ids[pos]
    true_word = vocab[true_id]
    logits = fake_logits(true_id, len(vocab), confidence=0.70 + random.random() * 0.15)
    probs = softmax(logits)
    loss = cross_entropy(probs, true_id)
    weighted_loss = loss * weight_t
    mdlm_total_loss += weighted_loss
    print(f"  {pos:>4}  {MASK:>8}  -> {true_word:>8}  {probs[true_id]:>12.3f}  {loss:>8.3f}  {weight_t:>10.3f}")

mdlm_avg = mdlm_total_loss / len(masked_positions)
print(f"\n  MDLM total loss:   {mdlm_total_loss:.3f}")
print(f"  MDLM average loss: {mdlm_avg:.3f}  (averaged over MASKED positions only)")

print(f"""
  Key differences:
    GPT: loss at {len(sentence_ids)-1} positions (all except first)
    MDLM: loss at {len(masked_positions)} positions (masked only)

    GPT: loss weighted equally across all positions
    MDLM: loss weighted by timestep t -- higher t = more masked = different weight

    Both use cross-entropy as the base loss function.
    MDLM adds timestep-dependent weighting to balance learning across noise levels.
""")


# ─────────────────────────────────────────────────────────
# DEMO 3: HOW LOSS VARIES WITH NOISE LEVEL t
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: Loss vs Noise Level -- Harder Timesteps")

print("""
At high noise (large t): more tokens are masked = harder prediction.
At low noise (small t): fewer tokens masked = easier prediction.

The model must perform well at ALL noise levels during training.
""")

T = 10
sentence_len = 6

print(f"  Sentence length: {sentence_len} tokens")
print(f"  T = {T} total timesteps")
print()
print(f"  {'t':>4}  {'p_mask':>8}  {'Avg masked':>12}  {'Difficulty':>12}  {'Expected loss':>15}")
print("  " + "-" * 60)

for t in range(T + 1):
    p_mask = t / T
    avg_masked = p_mask * sentence_len            # expected number of masked tokens
    if avg_masked == 0:
        difficulty = "easy (no masks)"
        expected_loss = 0.0
    elif avg_masked == sentence_len:
        difficulty = "hardest (all masked)"
        # Random guessing: loss = -log(1/vocab_size)
        expected_loss = math.log(len(vocab))
    else:
        difficulty = "medium"
        # Approximate: partially masked, use context clues
        frac_masked = avg_masked / sentence_len
        expected_loss = frac_masked * math.log(len(vocab)) * 0.6    # rough estimate
    print(f"  {t:>4}  {p_mask:>8.2f}  {avg_masked:>12.1f}  {difficulty:>12}  {expected_loss:>15.3f}")

print("""
  Insight: the model must handle both easy (few masks) and hard (many masks) cases.
  Timestep weighting in the ELBO balances how much each difficulty level
  contributes to the total loss.

  Without weighting: t=T (hardest) dominates, model ignores easy cases.
  With weighting:    all timesteps contribute equally to learning.
""")


# ─────────────────────────────────────────────────────────
# DEMO 4: ELBO DECOMPOSITION
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: ELBO -- Three Terms Explained")

print("""
The full ELBO for diffusion models has three parts:

  ELBO = Reconstruction_term - Prior_KL - Sum_of_Denoising_KL

Let's compute a simplified version of each for a 6-token sentence.
""")

# Reconstruction term: how well does the model recover x_0 from x_1?
# (lightly noised -> clean)
p_correct_x1 = 0.91       # model is quite good at final denoising step
reconstruction = math.log(p_correct_x1)
print(f"  1. Reconstruction term = log P(x_0 | x_1)")
print(f"     = log({p_correct_x1}) = {reconstruction:.3f}")
print(f"     Interpretation: model recovers original text from lightly noised version")
print(f"     Higher = better (log prob, so negative but close to 0 is good)")
print()

# Prior KL: how close is fully-noised distribution to uniform?
# For absorbing state: q(x_T | x_0) = all [MASK] = deterministic
# P(x_T) = uniform over all possible fully-masked sentences = also ~deterministic
# KL is close to 0 for absorbing state diffusion (good!)
prior_kl = 0.001
print(f"  2. Prior KL = KL(q(x_T|x_0) || P(x_T))")
print(f"     = {prior_kl} (near 0 for absorbing state -- desired)")
print(f"     Interpretation: fully noised text matches our assumed noise distribution")
print()

# Denoising KL: sum over all intermediate timesteps
print(f"  3. Denoising KL = sum over t of KL(q(x_t-1|x_t,x_0) || P(x_t-1|x_t))")
print(f"     This is the main training objective -- loss at each denoising step.")
denoising_kls = []
for t in range(1, T + 1):
    # Fake KL that decreases as training progresses
    kl_t = 0.5 * (1 - t/T) * random.uniform(0.1, 0.3) + 0.05
    denoising_kls.append(kl_t)
total_denoising_kl = sum(denoising_kls)
print(f"     Sum over t=1..{T}: {total_denoising_kl:.3f}")
print()

elbo = reconstruction - prior_kl - total_denoising_kl
print(f"  ELBO = {reconstruction:.3f} - {prior_kl:.3f} - {total_denoising_kl:.3f} = {elbo:.3f}")
print(f"  Training loss (NELBO) = -ELBO = {-elbo:.3f}  (minimize this)")
print("""
  In practice, MDLM simplifies this to just the denoising term
  (the per-timestep weighted cross-entropy), which is much easier to implement
  while still optimizing the ELBO approximately.
""")

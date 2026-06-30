"""
Example 02: Diffusion Process for Text
Module 16: Modern LLM Architectures

Simulates the forward (masking) and reverse (unmasking) diffusion process
for text using a simple masked diffusion approach (MDLM-style).

Part A: Forward process -- progressively mask tokens (destroy information)
Part B: Reverse process -- iteratively unmask tokens (create information)

Run:  python example_02_diffusion_text.py
Deps: none (pure Python)
"""

import random
import math


MASK = "[M]"    # the mask token (replaces [MASK] for readability)


def print_section(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


# ─────────────────────────────────────────────────────────
# PART A: FORWARD DIFFUSION (MASKING)
# ─────────────────────────────────────────────────────────

print_section("PART A: Forward Diffusion -- Adding Noise (Masking)")

print("""
The forward process destroys information by masking tokens.
At timestep t, each token is independently masked with probability beta_t.

The noise SCHEDULE controls how fast masking happens:
  t = 0:   0% masked  (original, clean text)
  t = T/2: 50% masked (partially noisy)
  t = T:   100% masked (pure noise)
""")


def noise_schedule(t, T, schedule="linear"):
    """
    Return the masking probability at timestep t.
    Higher t = more noise = more masking.

    schedule='linear': mask probability grows linearly from 0 to 1.
    """
    if schedule == "linear":
        return t / T     # simple linear increase from 0 to 1
    elif schedule == "cosine":
        # Cosine schedule: slower at the start and end, faster in middle
        # Used in many diffusion papers for better quality
        alpha = math.cos((t / T) * math.pi / 2) ** 2
        return 1.0 - alpha
    return t / T


def forward_diffusion(tokens, t, T, seed=42):
    """
    Apply forward diffusion to tokens at timestep t.
    Each token is independently masked with probability p_mask(t).

    Returns: list of tokens where some are replaced with MASK.
    """
    rng = random.Random(seed + t)           # reproducible randomness per step
    p_mask = noise_schedule(t, T)           # masking probability at this step
    noisy = []
    for token in tokens:
        if rng.random() < p_mask:
            noisy.append(MASK)              # token absorbed into mask state
        else:
            noisy.append(token)             # token survives
    return noisy


# Original sentence
sentence = ["The", "cat", "sat", "on", "the", "mat"]
T = 10          # total number of timesteps

print(f"  Original (t=0):  {sentence}")
print(f"  Total timesteps: T = {T}")
print()
print(f"  {'t':>4}  {'p_mask':>8}  Masked sentence")
print("  " + "-" * 55)

for t in range(T + 1):
    noisy = forward_diffusion(sentence, t, T, seed=99)
    p = noise_schedule(t, T)
    masked_count = noisy.count(MASK)
    print(f"  {t:>4}  {p:>8.2f}  {noisy}  ({masked_count}/{len(sentence)} masked)")

print("""
  Observation: tokens disappear gradually. At t=T, all are masked.
  This is the "pure noise" state -- equivalent to knowing nothing.
""")


# ─────────────────────────────────────────────────────────
# PART A2: CLOSED-FORM SAMPLING AT ANY TIMESTEP
# ─────────────────────────────────────────────────────────

print_section("PART A2: Closed-Form Sampling -- Jump to Any Noise Level")

print("""
Key advantage of absorbing-state (MDLM) diffusion:
We do NOT need to run the forward process step-by-step.
We can jump DIRECTLY to any noise level t in one formula.

q(x_t | x_0) = MASK with probability p_mask(t),
               x_0   with probability 1 - p_mask(t)

This enables fast training: sample random t, mask directly, compute loss.
No need to simulate all steps 0..t.
""")

def sample_at_timestep(original_tokens, t, T, seed=7):
    """
    Sample x_t directly from x_0 without stepping through intermediate states.
    This is the 'closed-form' q(x_t | x_0) for absorbing-state diffusion.
    """
    rng = random.Random(seed + t * 100)
    p = noise_schedule(t, T)
    return [MASK if rng.random() < p else tok for tok in original_tokens]


print("  Jumping directly to t=3, t=6, t=9 (no stepwise simulation):")
for t in [0, 3, 6, 9, 10]:
    x_t = sample_at_timestep(sentence, t, T)
    p = noise_schedule(t, T)
    print(f"  t={t} (p_mask={p:.1f}): {x_t}")

print("""
  Training uses this: sample random t each batch, mask tokens, compute loss.
  Very efficient -- each training step sees a different noise level.
""")


# ─────────────────────────────────────────────────────────
# PART B: REVERSE DIFFUSION (UNMASKING)
# ─────────────────────────────────────────────────────────

print_section("PART B: Reverse Diffusion -- Unmasking (Generation)")

print("""
The reverse process starts from pure noise (all masked) and iteratively
unmasks tokens. A trained model predicts what each [M] should be.

We simulate this with a fake "oracle" model that knows the correct answer.
In real training, the model must LEARN to predict from context alone.
""")


def fake_model_predict(noisy_tokens, original_tokens, t, T):
    """
    Fake oracle model: returns predicted probabilities for each masked position.
    In a real diffusion model, this would be a trained transformer.

    Returns: list of (token, confidence) for each position.
             Masked positions get the true token with high confidence.
             Unmasked positions return (existing_token, 1.0).
    """
    predictions = []
    for i, tok in enumerate(noisy_tokens):
        if tok == MASK:
            # Oracle "predicts" the correct token with some noise
            confidence = random.uniform(0.6, 0.99)       # fake confidence
            predictions.append((original_tokens[i], confidence))
        else:
            predictions.append((tok, 1.0))               # already unmasked
    return predictions


def reverse_diffusion_step(noisy_tokens, original_tokens, t, T, threshold=0.0):
    """
    One step of reverse diffusion.
    Model predicts each masked token. Commit tokens above confidence threshold.

    threshold=0.0  → greedy: always commit (fastest, lower quality)
    threshold=0.8  → cautious: only commit high-confidence predictions
    """
    predictions = fake_model_predict(noisy_tokens, original_tokens, t, T)
    new_tokens = []
    committed = 0

    for i, (tok, conf) in enumerate(predictions):
        if noisy_tokens[i] == MASK:
            if conf >= threshold or t == 1:   # always commit at final step
                new_tokens.append(tok)         # commit this prediction
                committed += 1
            else:
                new_tokens.append(MASK)        # not confident enough, stay masked
        else:
            new_tokens.append(tok)             # already known, keep it

    return new_tokens, committed


print("  Starting from fully masked text, reverse process fills in tokens.")
print()
print("  Strategy A: Greedy (threshold=0.0 -- always commit)")
print()

random.seed(42)
current = [MASK] * len(sentence)
print(f"  t={T} (start):  {current}")

for t in range(T, 0, -1):
    current, n_committed = reverse_diffusion_step(current, sentence, t, T, threshold=0.0)
    print(f"  t={t-1}:           {current}  ({n_committed} newly committed)")

print(f"\n  Final: {current}")
print(f"  Correct: {sentence}")
print(f"  Match: {current == sentence}")

print()
print("  Strategy B: Confidence threshold (threshold=0.85 -- only commit if sure)")
print()

random.seed(42)
current = [MASK] * len(sentence)
print(f"  t={T} (start):  {current}")

for t in range(T, 0, -1):
    current, n_committed = reverse_diffusion_step(current, sentence, t, T, threshold=0.85)
    print(f"  t={t-1}:           {current}  ({n_committed} newly committed)")

print(f"\n  Final: {current}")


# ─────────────────────────────────────────────────────────
# PART C: PARALLEL VS SEQUENTIAL STEP COUNT
# ─────────────────────────────────────────────────────────

print_section("PART C: Parallel Generation -- Step Count Comparison")

print("""
Key advantage: at each diffusion step, ALL positions are updated in parallel.
Compare the number of model calls needed:
""")

print(f"  {'Method':<20} {'Tokens':>8} {'Steps needed':>14} {'Notes'}")
print("  " + "-" * 65)

configs = [
    (100,  "AR",        100,  "1 step per token"),
    (100,  "Diffusion", 10,   "10 parallel steps (T=10)"),
    (100,  "Diffusion", 20,   "20 parallel steps (T=20)"),
    (1000, "AR",        1000, "1 step per token"),
    (1000, "Diffusion", 20,   "20 parallel steps"),
    (4096, "AR",        4096, "1 step per token"),
    (4096, "Diffusion", 50,   "50 parallel steps (T=50)"),
]

for n_tokens, method, steps, note in configs:
    ratio = n_tokens / steps if method == "Diffusion" else 1.0
    speedup = f"({ratio:.0f}x fewer steps)" if method == "Diffusion" else ""
    print(f"  {method:<20} {n_tokens:>8} {steps:>14}  {speedup}  {note}")

print("""
  Note: each diffusion step processes ALL tokens simultaneously.
  An AR step processes ONE token.
  At T=50, diffusion needs 50 passes but each pass generates ~N/50 tokens.
  Total model calls: 50 vs 4096 for AR -- 82x fewer passes at N=4096.
""")

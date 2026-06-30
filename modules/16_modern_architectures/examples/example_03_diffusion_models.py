"""
Example 03: Diffusion Language Models (MDLM, SEDD, Plaid)
Module 16: Modern LLM Architectures

Simulates the core behaviours of three diffusion language model families:
  - MDLM: absorbing state (token -> MASK, never back until reverse)
  - SEDD: vocabulary-level transitions (token -> any token)
  - Confidence-ordered unmasking (shared inference trick across all)

Run:  python example_03_diffusion_models.py
Deps: none (pure Python)
"""

import random
import math


MASK = "[M]"


def print_section(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


# ─────────────────────────────────────────────────────────
# MDLM: ABSORBING STATE DIFFUSION
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: MDLM -- Absorbing State Forward Process")

print("""
MDLM (Masked Diffusion Language Model):
  Forward process: each token can only go TO [MASK], never to another token.
  [MASK] is an "absorbing state" -- once masked, stays masked.

  At timestep t, token x_0 becomes:
    [MASK]  with probability q_t  (absorbed)
    x_0     with probability 1-q_t (unchanged)

  Key property: we can jump directly to any noise level t
  without simulating all intermediate steps.
""")

VOCAB = ["The", "cat", "sat", "on", "the", "mat", "dog", "ran", "over", "fence"]

def mdlm_forward(tokens, t, T, seed=42):
    """
    MDLM forward process at timestep t.
    Each token is independently absorbed (masked) with probability t/T.
    Once masked, a token stays masked -- it cannot become another token.

    This is the 'absorbing state' property: MASK is a one-way door.
    """
    rng = random.Random(seed + t)
    p_absorb = t / T                          # linear noise schedule
    result = []
    for tok in tokens:
        if tok == MASK:
            result.append(MASK)               # already absorbed, stays masked
        elif rng.random() < p_absorb:
            result.append(MASK)               # absorb this token
        else:
            result.append(tok)                # survives
    return result


sentence = ["The", "cat", "sat", "on", "the", "mat"]
T = 6

print(f"  Original: {sentence}")
print(f"  T = {T} timesteps\n")
print(f"  {'t':>3}  {'p_absorb':>9}  State")
print("  " + "-" * 50)

current = list(sentence)
for t in range(T + 1):
    # Show the distribution at each step using closed-form sampling
    x_t = mdlm_forward(sentence, t, T)
    p = t / T
    n_masked = x_t.count(MASK)
    bar = "#" * n_masked + "." * (len(sentence) - n_masked)
    print(f"  {t:>3}  {p:>9.2f}  {str(x_t):<40}  [{bar}]")

print("""
  Notice: tokens only disappear (absorbed). They never change to a different
  word. "cat" either stays "cat" or becomes [M]. Never becomes "dog".
  This simplicity makes MDLM mathematically clean and fast to train.
""")


# ─────────────────────────────────────────────────────────
# SEDD: VOCABULARY-LEVEL TRANSITIONS
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: SEDD -- Vocabulary-Level Transitions")

print("""
SEDD (Score Entropy Discrete Diffusion):
  Forward process: tokens can transition to ANY vocabulary token, not just [MASK].
  More expressive -- captures richer noise patterns.

  Transition probabilities:
    - With high prob at small t: stay at original token
    - With small prob: transition to any random vocabulary token
    - With larger prob at large t: transition to uniform distribution over vocab

  This allows the model to learn from a richer noise distribution.
""")

def sedd_forward_token(token, t, T, vocab, seed=0):
    """
    SEDD forward process for ONE token at timestep t.
    Returns the noised token.

    At low noise (small t): token usually stays the same.
    At high noise (large t): token transitions to a random vocab word.
    """
    rng = random.Random(seed)
    p_noise = (t / T) ** 2           # quadratic: slower noise at start
    p_stay = 1.0 - p_noise

    if rng.random() < p_stay:
        return token                  # token unchanged
    else:
        return rng.choice(vocab)      # transition to random vocab token


print(f"  Vocabulary: {VOCAB}")
print(f"  Original sentence: {sentence}")
print()
print("  Showing how tokens change at different noise levels:")
print(f"  {'t':>3}  {'p_noise':>9}  SEDD-noised sentence")
print("  " + "-" * 60)

for t in range(T + 1):
    noised = [sedd_forward_token(tok, t, T, VOCAB, seed=i*17+t*3)
              for i, tok in enumerate(sentence)]
    p = (t / T) ** 2
    print(f"  {t:>3}  {p:>9.3f}  {noised}")

print("""
  Difference from MDLM:
    MDLM t=3: ['The', [M], 'sat', [M], 'the', 'mat']  -- only MASK appears
    SEDD  t=3: ['The', 'dog', 'sat', 'fence', 'the', 'mat']  -- random words appear

  SEDD's richer transitions allow the model to learn finer-grained patterns,
  at the cost of more complex loss computation (score entropy vs cross-entropy).
""")


# ─────────────────────────────────────────────────────────
# CONFIDENCE-ORDERED UNMASKING (MDLM INFERENCE)
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: Confidence-Ordered Unmasking -- MDLM Inference")

print("""
During MDLM inference, the model predicts ALL masked tokens at each step.
But which ones should we commit vs leave masked for the next step?

Three strategies:
  1. Greedy:     always commit argmax prediction (fast, lower quality)
  2. Threshold:  only commit predictions above confidence C (balanced)
  3. Top-K:      commit the K most confident predictions per step (controlled)

This example simulates strategy 3 (Top-K), which is most common in practice.
High-confidence tokens appear first -- regardless of position.
""")

# Simulate a "model" that knows the right answer with varying confidence
def simulate_model_confidence(original_tokens, masked_tokens, rng):
    """
    Simulate model predictions for masked positions.
    Returns list of (position, token, confidence) for each masked position.
    Each masked position gets the correct token with random confidence.
    """
    predictions = []
    for i, tok in enumerate(masked_tokens):
        if tok == MASK:
            confidence = rng.uniform(0.5, 0.99)         # fake but realistic
            predictions.append((i, original_tokens[i], confidence))
    return predictions


def mdlm_generate(target_tokens, T_steps, top_k_per_step=1):
    """
    Simulate MDLM generation from fully masked to complete sentence.
    At each step, commits the top_k_per_step most confident predictions.

    target_tokens: the sentence we want to recover (oracle answer)
    T_steps: number of denoising steps
    top_k_per_step: how many tokens to commit per step
    """
    rng = random.Random(77)
    current = [MASK] * len(target_tokens)       # start fully masked

    print(f"  Start (t={T_steps}): {current}")
    print()

    for step in range(T_steps):
        t_remaining = T_steps - step             # steps remaining
        predictions = simulate_model_confidence(target_tokens, current, rng)

        if not predictions:
            break                                # all tokens already committed

        # Sort by confidence (highest first) -- commit most confident ones
        predictions.sort(key=lambda x: x[2], reverse=True)

        # Commit top_k predictions this step
        to_commit = predictions[:top_k_per_step]
        for pos, token, conf in to_commit:
            current[pos] = token

        committed_info = [(pos, tok, f"{conf:.2f}") for pos, tok, conf in to_commit]
        print(f"  Step {step+1} (t={t_remaining-1}): commit {committed_info}")
        print(f"            state: {current}")
        print()

    return current


print("  Recovering: 'The cat sat on the mat' from fully masked")
print("  Top-1 per step: commit most confident token each step")
print()

result = mdlm_generate(sentence, T_steps=len(sentence), top_k_per_step=1)
print(f"  Final:   {result}")
print(f"  Target:  {sentence}")
print(f"  Success: {result == sentence}")

print("""
  Observation: tokens appear in ORDER OF CONFIDENCE, not position order.
  This is the key difference from AR generation (which is strictly left-to-right).
  "sat" may appear before "the" if the model is more confident about it.
""")


# ─────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: MDLM vs SEDD vs Plaid -- Comparison")

print("""
  Property           MDLM                SEDD                Plaid
  ───────────────────────────────────────────────────────────────────────
  Noise type         Token -> [MASK]     Token -> any tok    Token -> [MASK]
  Closed form q_t?   Yes (simple)        Partial             Yes
  Loss function      Weighted CE         Score entropy       Weighted CE + tricks
  Math complexity    Low                 High                Medium
  Generation speed   Fast (T steps)      Moderate            Fast (optimised)
  Quality at scale   Good                Better theory       Best (prod ready)
  Key paper          Shi et al 2024      Lou et al 2024      Gulrajani 2024
  Good for           Learning/research   Research            Production
  ───────────────────────────────────────────────────────────────────────

  All three generate text by starting from noise and iteratively denoising.
  The difference is HOW they define "noise" and HOW they compute the loss.
""")

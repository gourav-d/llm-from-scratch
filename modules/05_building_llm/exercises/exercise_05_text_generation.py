"""
=============================================================================
EXERCISE 5: Text Generation & Sampling Strategies — Hands-On Practice
=============================================================================

HOW TO USE THIS FILE
---------------------
Each exercise has:
  1. A clear description of what to build
  2. A skeleton with TODO comments
  3. A hint (read only if stuck)
  4. A solution (ONLY look after trying!)

REQUIREMENTS
  - Python 3.8+
  - numpy  (pip install numpy)
  - No model or PyTorch needed — all exercises work with fake probabilities

=============================================================================
"""

import numpy as np

print("=" * 60)
print("EXERCISE 5: Text Generation & Sampling Strategies")
print("=" * 60)

# =============================================================================
# Shared setup: a fake "model output" used in all exercises
# =============================================================================

# Imagine a model has seen "The cat" and must decide the next word.
# These are the scores (logits) it produced for each possible word.
VOCAB  = ["sat", "ran", "slept", "ate", "jumped", "flew", "sang", "whispered"]
LOGITS = np.array([3.5, 2.1, 1.5, 1.2, 0.8, 0.1, -0.3, -1.0])

def softmax(x):
    """
    Convert raw scores to probabilities.
    The formula is: exp(x) / sum(exp(x))
    We subtract the max first to avoid very large numbers (numerical stability).
    """
    e = np.exp(x - x.max())       # subtract max for stability
    return e / e.sum()             # divide each by the total

BASE_PROBS = softmax(LOGITS)

print("\nSetup: 'The cat ___' — model output probabilities")
print(f"{'Word':<12} {'Probability':>12}")
print("-" * 26)
for word, prob in zip(VOCAB, BASE_PROBS):
    bar = "#" * int(prob * 25)
    print(f"{word:<12} {prob:>10.1%}  {bar}")

# =============================================================================
# EXERCISE 1: Implement Greedy Sampling
# =============================================================================
# DIFFICULTY: Easy
#
# WHAT IS GREEDY?
#   Always pick the word with the HIGHEST probability. Simple. No randomness.
#
# YOUR TASK:
#   Implement greedy_sample() below.
#   It should take a probability array and return the INDEX of the max value.
#
# AFTER: compare it to random sampling to see why greedy is boring.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 1: Greedy Sampling")
print("=" * 60)

def greedy_sample(probs):
    """
    Pick the token with the highest probability.

    probs   : numpy array of probabilities (sum to 1)
    Returns : integer index of the maximum probability
    """
    # TODO: return the index of the highest probability
    # HINT: np.argmax(probs) returns the index of the maximum value
    pass   # replace this line


# --- Test Exercise 1 ---
result_idx = greedy_sample(BASE_PROBS)

if result_idx is not None:
    print(f"\nGreedy always picks: '{VOCAB[result_idx]}' (probability: {BASE_PROBS[result_idx]:.1%})")
    print()
    print("Run 10 times — does it always pick the same word?")
    for i in range(10):
        idx = greedy_sample(BASE_PROBS)
        print(f"  Trial {i+1}: 'The cat {VOCAB[idx]}'")
else:
    print("  [Complete the TODO to see output]")

"""
---- SOLUTION ----

def greedy_sample(probs):
    return np.argmax(probs)
"""

# =============================================================================
# EXERCISE 2: Implement Temperature Sampling
# =============================================================================
# DIFFICULTY: Easy
#
# WHAT IS TEMPERATURE?
#   Divide logits by a number (temperature) BEFORE converting to probabilities.
#   Low temperature -> top word dominates even more
#   High temperature -> all words become more equal (more creative/random)
#
# YOUR TASK:
#   Implement temperature_sample() below.
#   Steps:
#     1. Divide logits by temperature
#     2. Convert to probabilities using softmax
#     3. Sample using np.random.choice
#
# THEN: run it at temp=0.3 and temp=2.0, compare the variety.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 2: Temperature Sampling")
print("=" * 60)

def temperature_sample(logits, temperature):
    """
    Apply temperature scaling then sample.

    logits      : raw scores from the model (NOT probabilities yet)
    temperature : float > 0
                  < 1.0 -> focused (top word more dominant)
                  = 1.0 -> unchanged
                  > 1.0 -> creative (all words more equal)
    Returns     : integer index of sampled token
    """
    # TODO Step 1: Divide logits by temperature
    # HINT: scaled = logits / temperature
    scaled = None   # replace

    # TODO Step 2: Convert to probabilities
    # HINT: probs = softmax(scaled)
    probs = None    # replace

    # TODO Step 3: Sample a random index weighted by probs
    # HINT: np.random.choice(len(probs), p=probs)
    return None     # replace


# --- Test Exercise 2 ---
print("\nSampling 8 times with temperature=0.3 (focused):")
for i in range(8):
    idx = temperature_sample(LOGITS, temperature=0.3)
    if idx is not None:
        print(f"  Trial {i+1}: 'The cat {VOCAB[idx]}'")

print("\nSampling 8 times with temperature=2.0 (creative):")
for i in range(8):
    idx = temperature_sample(LOGITS, temperature=2.0)
    if idx is not None:
        print(f"  Trial {i+1}: 'The cat {VOCAB[idx]}'")

print()
print("Notice: temperature=0.3 picks 'sat' almost every time.")
print("        temperature=2.0 picks different words each time.")

"""
---- SOLUTION ----

def temperature_sample(logits, temperature):
    scaled = logits / temperature
    probs  = softmax(scaled)
    return np.random.choice(len(probs), p=probs)
"""

# =============================================================================
# EXERCISE 3: Implement Top-k Sampling from Scratch
# =============================================================================
# DIFFICULTY: Medium
#
# WHAT IS TOP-K?
#   Only sample from the top k most probable words. Ignore all others.
#   k=50 means: look at 50 words, ignore the remaining thousands.
#
# YOUR TASK:
#   Implement top_k_sample() from scratch.
#
# STEPS (spelled out in order):
#   1. Apply temperature by dividing logits
#   2. Convert to probabilities with softmax
#   3. Find the k-th largest probability value
#      (sort descending, take the k-th item)
#   4. Zero out any probability BELOW that threshold
#   5. Re-normalize (divide by the new total so they sum to 1 again)
#   6. Sample using np.random.choice
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 3: Top-k Sampling from Scratch")
print("=" * 60)

def top_k_sample(logits, k, temperature=1.0):
    """
    Sample from the top-k most probable tokens only.

    logits      : raw scores from the model
    k           : how many top tokens to keep
    temperature : scaling factor before softmax

    Returns     : integer index of sampled token
    """
    # Step 1: Apply temperature
    # TODO: scaled = logits / temperature
    scaled = None   # replace

    # Step 2: Convert to probabilities
    # TODO: probs = softmax(scaled)
    probs = None    # replace

    # Step 3: Find the threshold (value of the k-th largest probability)
    # TODO:
    #   sorted_probs = np.sort(probs)[::-1]   # sort descending
    #   threshold    = sorted_probs[k - 1]     # k-th largest (0-indexed: k-1)
    threshold = None   # replace

    # Step 4: Zero out any probability below the threshold
    # TODO: filtered = np.where(probs >= threshold, probs, 0.0)
    filtered = None   # replace

    # Step 5: Re-normalize so the kept probabilities sum to 1
    # TODO: filtered = filtered / filtered.sum()
    # (If filtered.sum() is 0, fall back to uniform: filtered = ones / len)

    # Step 6: Sample
    # TODO: return np.random.choice(len(filtered), p=filtered)
    return None   # replace


# --- Test Exercise 3 ---
print("\nSampling 8 times with k=2 (only top-2 words allowed):")
for i in range(8):
    idx = top_k_sample(LOGITS, k=2, temperature=0.8)
    if idx is not None:
        print(f"  Trial {i+1}: 'The cat {VOCAB[idx]}'   (word: {VOCAB[idx]}, "
              f"prob={BASE_PROBS[idx]:.1%})")

print()
print("Which 2 words should be the only options?")
sorted_words = sorted(zip(BASE_PROBS, VOCAB), reverse=True)
print(f"  Top-2: {[w for _, w in sorted_words[:2]]}")

"""
---- SOLUTION ----

def top_k_sample(logits, k, temperature=1.0):
    scaled   = logits / temperature
    probs    = softmax(scaled)

    sorted_probs = np.sort(probs)[::-1]
    threshold    = sorted_probs[k - 1]

    filtered = np.where(probs >= threshold, probs, 0.0)
    total    = filtered.sum()
    if total > 0:
        filtered = filtered / total
    else:
        filtered = np.ones(len(probs)) / len(probs)

    return np.random.choice(len(filtered), p=filtered)
"""

# =============================================================================
# EXERCISE 4: Implement Top-p (Nucleus) Sampling from Scratch
# =============================================================================
# DIFFICULTY: Medium-Hard
#
# WHAT IS TOP-P?
#   Keep the smallest group of tokens whose probabilities add up to at least p.
#   This group is called the "nucleus".
#
#   Why better than top-k?
#   - Top-k: FIXED number of words, regardless of the model's confidence
#   - Top-p: DYNAMIC number based on the model's confidence
#     If model is confident: 1-2 words cover 90% -> small nucleus
#     If model is uncertain: many words needed -> large nucleus
#
# STEPS (in order):
#   1. Apply temperature
#   2. Convert to probabilities
#   3. Sort probabilities from highest to lowest
#   4. Compute cumulative sum (running total)
#   5. Find where cumulative sum first exceeds p
#   6. Zero out everything beyond that point
#   7. Re-normalize
#   8. Sample
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 4: Top-p (Nucleus) Sampling from Scratch")
print("=" * 60)

def top_p_sample(logits, p, temperature=1.0):
    """
    Sample from the smallest nucleus covering probability p.

    logits      : raw scores from the model
    p           : cumulative probability threshold (e.g. 0.9 = 90%)
    temperature : scaling factor before softmax

    Returns     : integer index of sampled token
    """
    # Step 1: Apply temperature
    # TODO: scaled = logits / temperature
    scaled = None   # replace

    # Step 2: Convert to probabilities
    # TODO: probs = softmax(scaled)
    probs = None    # replace

    # Step 3: Sort by probability, highest first
    # TODO:
    #   sorted_indices = np.argsort(probs)[::-1]   # indices sorted by prob (desc)
    #   sorted_probs   = probs[sorted_indices]       # the probabilities, sorted
    sorted_indices = None   # replace
    sorted_probs   = None   # replace

    # Step 4: Compute cumulative sum
    # e.g. [0.5, 0.3, 0.15, 0.05] -> [0.5, 0.8, 0.95, 1.0]
    # TODO: cumulative = np.cumsum(sorted_probs)
    cumulative = None   # replace

    # Step 5: Find the cutoff index — the first position where cumulative >= p
    # TODO: cutoff = np.searchsorted(cumulative, p) + 1
    #   np.searchsorted: finds where p would be inserted in a sorted array
    #   +1: include the word that pushes cumulative over p
    cutoff = None   # replace

    # Step 6: Zero out everything beyond the cutoff
    # TODO:
    #   keep_indices    = sorted_indices[:cutoff]     # which original indices to keep
    #   filtered        = np.zeros_like(probs)        # start with all zeros
    #   filtered[keep_indices] = probs[keep_indices]  # restore kept values
    filtered = None   # replace

    # Step 7: Re-normalize
    # TODO: filtered = filtered / filtered.sum()
    # (If filtered.sum() is 0, use uniform)

    # Step 8: Sample
    # TODO: return np.random.choice(len(filtered), p=filtered)
    return None   # replace


# --- Test Exercise 4 ---
print("\nShowing nucleus size at different p values:")
print(f"{'p value':<10} {'Nucleus words'}")
print("-" * 45)

for p_val in [0.5, 0.7, 0.9, 0.95, 1.0]:
    # Manually compute the nucleus for display
    sort_idx = np.argsort(BASE_PROBS)[::-1]
    cumul    = np.cumsum(BASE_PROBS[sort_idx])
    cutoff   = np.searchsorted(cumul, p_val) + 1
    nucleus  = [VOCAB[i] for i in sort_idx[:cutoff]]
    print(f"  p={p_val:.2f}    {nucleus}")

print()
print("Sampling 8 times with p=0.8 (nucleus sampling):")
for i in range(8):
    idx = top_p_sample(LOGITS, p=0.8, temperature=0.8)
    if idx is not None:
        print(f"  Trial {i+1}: 'The cat {VOCAB[idx]}'")

"""
---- SOLUTION ----

def top_p_sample(logits, p, temperature=1.0):
    scaled = logits / temperature
    probs  = softmax(scaled)

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs   = probs[sorted_indices]

    cumulative = np.cumsum(sorted_probs)
    cutoff     = np.searchsorted(cumulative, p) + 1

    keep_indices           = sorted_indices[:cutoff]
    filtered               = np.zeros_like(probs)
    filtered[keep_indices] = probs[keep_indices]

    total = filtered.sum()
    if total > 0:
        filtered = filtered / total
    else:
        filtered = np.ones(len(probs)) / len(probs)

    return np.random.choice(len(filtered), p=filtered)
"""

# =============================================================================
# EXERCISE 5: Compare all strategies on the same prompt
# =============================================================================
# DIFFICULTY: Easy (you just call the functions you built above)
#
# TASK:
#   Use the functions you built in Exercises 1-4 to compare outputs.
#   For each strategy, generate a 5-word sentence (pick a word 5 times).
#   Notice how different strategies produce different outputs.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 5: Compare All Strategies")
print("=" * 60)

def generate_sentence(strategy_fn, n_words=5):
    """
    Generate n_words by calling strategy_fn n_words times.
    Uses the same LOGITS each time (pretending it is a model that always
    returns the same distribution — a simplification for this exercise).
    """
    words = []
    for _ in range(n_words):
        idx = strategy_fn()
        if idx is not None:
            words.append(VOCAB[idx])
        else:
            words.append("[?]")
    return "The cat " + " ".join(words)


print("\nGenerating 5-word sentences with each strategy (run 3 times each):\n")

strategies = {
    "Greedy":          lambda: greedy_sample(BASE_PROBS),
    "Temp=0.5":        lambda: temperature_sample(LOGITS, 0.5),
    "Temp=1.5":        lambda: temperature_sample(LOGITS, 1.5),
    "Top-k (k=2)":     lambda: top_k_sample(LOGITS, k=2),
    "Top-p (p=0.7)":   lambda: top_p_sample(LOGITS, p=0.7),
}

np.random.seed(42)   # reproducible output

for name, fn in strategies.items():
    print(f"  {name:<18}:")
    for trial in range(3):
        sentence = generate_sentence(fn, n_words=4)
        print(f"    Trial {trial+1}: '{sentence}'")
    print()

print("QUESTIONS TO THINK ABOUT:")
print("  1. Which strategy is most predictable? Why?")
print("  2. Which strategy produces the most variety?")
print("  3. For a chatbot, which strategy would you choose? Why?")
print("  4. For code completion, which strategy would you choose? Why?")

print("\n" + "=" * 60)
print("Exercise 5 Complete!")
print()
print("KEY TAKEAWAYS:")
print("  Greedy     -> deterministic, repetitive, safe")
print("  Low temp   -> focused, mostly picks top word")
print("  High temp  -> creative, sometimes nonsense")
print("  Top-k      -> prevents very unlikely words")
print("  Top-p      -> adapts to confidence (best overall)")
print()
print("REAL-WORLD USE:")
print("  ChatGPT    -> temperature ~ 0.8, top_p = 0.9")
print("  Copilot    -> temperature ~ 0.2, top_k = 10")
print("  Creative   -> temperature ~ 1.2, top_p = 0.95")
print("=" * 60)

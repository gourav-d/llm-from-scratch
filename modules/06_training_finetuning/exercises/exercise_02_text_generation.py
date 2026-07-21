"""
Module 06 - Training & Fine-Tuning
Exercise 02: Text Generation and Sampling Strategies

GLOSSARY
--------
Logits           : Raw scores from the model's final layer (before softmax).
                   High logit = model thinks this token is likely next.
                   Like unnormalized scores in a competition.
Softmax          : Converts logits to probabilities that sum to 1.
                   Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
                   Like converting raw scores to percentage likelihoods.
Temperature      : Controls how "creative" or "safe" the sampling is.
                   temp < 1: more focused (higher confidence tokens favored)
                   temp > 1: more random (flatter distribution)
                   temp = 1: normal sampling (no change)
Greedy Decoding  : Always pick the token with the highest probability.
                   Safe but repetitive. Like always picking the safest answer.
Top-K Sampling   : Only consider the K most likely tokens, then sample.
                   Prevents the model from picking very unlikely tokens.
Top-P (Nucleus)  : Keep tokens whose cumulative probability >= P.
                   More adaptive than top-K (K changes based on distribution).
Perplexity       : Measure of how "surprised" the model is by text.
                   Low perplexity = model is confident. High = uncertain.
                   Formula: exp(average cross-entropy loss)
"""

import numpy as np   # NumPy for math operations

print("=" * 60)
print("Exercise 02: Text Generation Sampling Strategies")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Softmax — Convert Logits to Probabilities
#
#  Background:
#    The model outputs logits (raw scores) for each token.
#    We need to convert them to probabilities (sum = 1.0).
#
#    Formula (numerically stable version):
#      1. Subtract max(logits) to prevent overflow
#      2. exp_scores = exp(logits - max)
#      3. probs      = exp_scores / sum(exp_scores)
#
#    Why subtract max?
#      exp(1000) overflows to infinity. exp(0) = 1.0 is safe.
#      Subtracting max doesn't change the output probabilities.
#
#  Your Task:
#    Write: softmax(logits) -> np.ndarray
#    Returns probability array with same shape as logits.
#
#  C# Analogy:
#    Like normalising scores to percentages:
#      double[] softmax = scores.Select(s => Math.Exp(s - max)).ToArray()
#      double sum = softmax.Sum()
#      return softmax.Select(v => v / sum).ToArray()
# ============================================================

print("-" * 50)
print("EXERCISE 1: Softmax")
print("-" * 50)
print()


def softmax(logits):
    """
    Convert logits to probabilities using the softmax function.

    Parameters:
        logits (np.ndarray): 1D array of raw scores.

    Returns:
        np.ndarray: 1D array of probabilities (sums to 1.0).
    """
    # TODO:
    # 1. Subtract max(logits) for numerical stability
    # 2. Compute exp of shifted logits
    # 3. Divide by sum to get probabilities
    pass  # Replace with your implementation


logits = np.array([2.0, 1.0, 0.5, -1.0, 3.0])

probs = softmax(logits)
if probs is not None:
    print(f"  Logits : {logits}")
    print(f"  Probs  : {np.round(probs, 4)}")
    print(f"  Sum    : {probs.sum():.6f}  (should be 1.000000)")
    print(f"  Max at : index {np.argmax(probs)}  (should be index 4, highest logit)")
print()
print("  Expected sum: 1.0. Highest logit (3.0) -> highest prob.")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Greedy Decoding
#
#  Background:
#    Greedy decoding always picks the token with highest probability.
#    Fast and deterministic — but often generates repetitive text.
#
#    Steps:
#      1. Convert logits to probs (softmax)
#      2. Return the index of the maximum probability
#
#    Example:
#      probs = [0.05, 0.60, 0.30, 0.05]
#      greedy_sample -> 1   (index of 0.60)
#
#  Your Task:
#    Write: greedy_sample(logits) -> int
#    Returns the index of the most likely token.
#
#  C# Analogy:
#    Like Array.IndexOf(probs, probs.Max()) in C#.
# ============================================================

print("-" * 50)
print("EXERCISE 2: Greedy Decoding")
print("-" * 50)
print()


def greedy_sample(logits):
    """
    Pick the token with the highest logit (greedy decoding).

    Parameters:
        logits (np.ndarray): 1D array of logits.

    Returns:
        int: Index of the most likely next token.
    """
    # TODO:
    # Hint: np.argmax(logits) returns the index of the maximum value.
    # You do NOT need softmax here — argmax of logits == argmax of probs.
    pass  # Replace with your implementation


test_logits = np.array([0.1, 2.5, 1.2, 0.3, 0.8])

chosen = greedy_sample(test_logits)
if chosen is not None:
    print(f"  Logits : {test_logits}")
    print(f"  Greedy choice : token {chosen}  (logit = {test_logits[chosen]:.1f})")
print()
print("  Expected: token 1 (highest logit = 2.5)")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Temperature Scaling
#
#  Background:
#    Temperature controls how focused vs random the sampling is.
#    Formula:
#      scaled_logits = logits / temperature
#    Then apply softmax to scaled_logits.
#
#    Effect:
#      temperature = 1.0 -> no change (original probabilities)
#      temperature < 1.0 -> sharper distribution (model more confident)
#      temperature > 1.0 -> flatter distribution (model more random)
#
#    Example (temp=0.5 vs temp=2.0):
#      Original probs: [0.1, 0.7, 0.2]
#      temp=0.5:       [0.01, 0.95, 0.04]   <- more focused on token 1
#      temp=2.0:       [0.18, 0.52, 0.30]   <- more spread out
#
#  Your Task:
#    Write: apply_temperature(logits, temperature) -> np.ndarray
#    Returns probability array after applying temperature scaling.
#
#  C# Analogy:
#    Like adjusting a probability distribution to be more/less peaked.
#    Low temp = winner-take-all. High temp = equal chance.
# ============================================================

print("-" * 50)
print("EXERCISE 3: Temperature Scaling")
print("-" * 50)
print()


def apply_temperature(logits, temperature):
    """
    Apply temperature scaling to logits, then return probabilities.

    Parameters:
        logits      (np.ndarray): 1D array of raw logits.
        temperature (float)     : Temperature (> 0). Lower = more focused.

    Returns:
        np.ndarray: Probability array after temperature scaling.
    """
    # TODO:
    # 1. scaled_logits = logits / temperature
    # 2. return softmax(scaled_logits)
    # Use the softmax() function you wrote in Exercise 1.
    pass  # Replace with your implementation


base_logits = np.array([1.0, 3.0, 2.0])

print("  base_logits:", base_logits)
print()
print(f"  {'Temp':>6}  {'Token 0':>10}  {'Token 1':>10}  {'Token 2':>10}")
print("  " + "-" * 42)
for temp in [0.5, 1.0, 2.0, 5.0]:
    probs = apply_temperature(base_logits, temp)
    if probs is not None:
        print(f"  {temp:>6.1f}  {probs[0]:>10.4f}  {probs[1]:>10.4f}  {probs[2]:>10.4f}")
print()
print("  Expected: at temp=0.5, token 1 prob >> token 2. At temp=5.0, distribution flatter.")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Top-K Filtering
#
#  Background:
#    Top-K sampling keeps only the K most likely tokens.
#    All other tokens are set to -infinity (zeroed out after softmax).
#
#    Steps:
#      1. Find the K largest logit values
#      2. Set all logits below the K-th largest to -inf
#      3. Apply softmax to get probabilities
#
#    Why? Prevents the model from ever choosing very unlikely tokens.
#    Example: top_k=2 from [0.1, 2.5, 1.2, 0.3, 0.8]
#      Keep only tokens 1 (2.5) and 2 (1.2). Zero out others.
#
#  Your Task:
#    Write: top_k_filter(logits, k) -> np.ndarray
#    Returns probabilities where only top-k logits are kept.
#
#  C# Analogy:
#    Like keeping only the top-K results from a sorted list,
#    then discarding the rest before computing scores.
# ============================================================

print("-" * 50)
print("EXERCISE 4: Top-K Filtering")
print("-" * 50)
print()


def top_k_filter(logits, k):
    """
    Apply top-k filtering to logits, then return probabilities.

    Parameters:
        logits (np.ndarray): 1D array of raw logits.
        k      (int)       : Number of top tokens to keep.

    Returns:
        np.ndarray: Probability array where only top-k tokens have non-zero prob.
    """
    # TODO:
    # 1. Find the k-th largest value using np.partition or np.sort
    #    Hint: np.partition(logits, -k)[-k] gives the k-th largest value.
    # 2. Create a copy of logits
    # 3. Set all values BELOW the k-th largest to -np.inf
    # 4. Apply softmax to the filtered logits
    pass  # Replace with your implementation


test_logits = np.array([0.1, 2.5, 1.2, 0.3, 0.8])

print(f"  Logits: {test_logits}")
print()
print(f"  {'K':>4}  {'T0':>8}  {'T1':>8}  {'T2':>8}  {'T3':>8}  {'T4':>8}  {'Non-zero':>10}")
print("  " + "-" * 60)
for k in [1, 2, 3]:
    probs = top_k_filter(test_logits, k)
    if probs is not None:
        nonzero = np.sum(probs > 0.001)
        vals = "  ".join(f"{p:>8.4f}" for p in probs)
        print(f"  {k:>4}  {vals}  {nonzero:>10}")
print()
print("  Expected: k=1 -> only token 1 has prob 1.0, k=2 -> tokens 1 and 2 only.")
print()

print("=" * 60)
print("All exercises complete!")
print("=" * 60)

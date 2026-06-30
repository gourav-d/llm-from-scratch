"""
Exercise 04: Diffusion Loss Function (ELBO / Weighted Cross-Entropy)
Module 16: Modern LLM Architectures

TASKS:
  1. Implement softmax()             -- convert logits to probabilities
  2. Implement cross_entropy()       -- loss for one prediction
  3. Implement mdlm_loss_at_step()   -- weighted CE for masked positions
  4. Implement training_loss()       -- average loss over sampled timesteps

Run:  python exercise_04_elbo_loss.py
Deps: none (pure Python)
"""

import math
import random

MASK = "[M]"


# ─────────────────────────────────────────────────────────
# TASK 1: Softmax
# ─────────────────────────────────────────────────────────

def softmax(logits):
    """
    Convert a list of raw scores (logits) to probabilities using softmax.
    All output values are in [0, 1] and sum to 1.0.

    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

    IMPORTANT: Subtract max(logits) from all logits BEFORE exp() for
    numerical stability. This prevents overflow with large logits.
    (Subtracting a constant doesn't change the final probabilities.)

    Args:
        logits: list of float raw scores (can be any values, positive or negative)

    Returns:
        list of float probabilities (same length as logits, sum = 1.0)

    Example:
        softmax([1.0, 1.0, 1.0]) -> [0.333, 0.333, 0.333]  (uniform)
        softmax([10.0, 0.0, 0.0]) -> [~1.0, ~0.0, ~0.0]    (dominant class)
        softmax([0.0]) -> [1.0]

    HINT:
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]   <- subtract max first
        total = sum(exps)
        return [e / total for e in exps]
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Cross-Entropy Loss
# ─────────────────────────────────────────────────────────

def cross_entropy(probs, true_index):
    """
    Compute cross-entropy loss for a single prediction.

    Formula: loss = -log(P(correct token))

    Lower is better:
      If P(correct) = 0.9: loss = -log(0.9) = 0.105  (good prediction)
      If P(correct) = 0.1: loss = -log(0.1) = 2.303  (bad prediction)
      If P(correct) = 0.0: loss -> infinity (never predict 0 probability)

    Args:
        probs:      list of float probabilities (output of softmax)
        true_index: int, index of the correct token in the vocabulary

    Returns:
        float: cross-entropy loss (non-negative)

    Example:
        cross_entropy([0.1, 0.8, 0.1], true_index=1) -> -log(0.8) = 0.223
        cross_entropy([0.33, 0.33, 0.33], true_index=0) -> -log(0.33) = 1.109

    HINT:
        p_correct = probs[true_index]
        Clamp to at least 1e-10 to avoid log(0).
        return -math.log(p_correct)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: MDLM Loss at One Timestep
# ─────────────────────────────────────────────────────────

def mdlm_loss_at_step(masked_tokens, original_ids, model_logits, t, T):
    """
    Compute the MDLM training loss at timestep t.

    Only computes loss on MASKED positions (not all positions).
    Weights the loss by w(t) = t / T (higher timestep = higher weight).

    Args:
        masked_tokens: list of tokens at timestep t (some are MASK)
        original_ids:  list of int token IDs (the ground truth)
        model_logits:  list of lists -- logits[i] is the model's output for position i
                       shape: [seq_len, vocab_size]
        t:             current timestep
        T:             total timesteps

    Returns:
        float: weighted average cross-entropy loss over masked positions only
               Returns 0.0 if no positions are masked.

    Example:
        masked = ["[M]", "cat", "[M]"]
        original_ids = [2, 1, 5]           (vocab indices: 2=The, 1=cat, 5=sat)
        model_logits = [[logits for pos 0], [logits for pos 1], [logits for pos 2]]
        -> compute CE at positions 0 and 2 only (where MASK appears)
        -> weight by t/T
        -> return weighted average

    HINT:
        1. weight = t / T
        2. total_loss = 0.0, count = 0
        3. For each position i where masked_tokens[i] == MASK:
               probs = softmax(model_logits[i])
               loss = cross_entropy(probs, original_ids[i])
               total_loss += loss * weight
               count += 1
        4. Return total_loss / count if count > 0 else 0.0
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Training Loss (Average Over Sampled Timesteps)
# ─────────────────────────────────────────────────────────

def training_loss(original_tokens, original_ids, model_fn, T, n_samples=5, seed=42):
    """
    Estimate the MDLM training loss by sampling random timesteps.

    In real training, we sample one random t per batch and compute the loss.
    Here we average over n_samples timesteps for a better estimate.

    Algorithm:
        For each sample:
            1. Sample a random timestep t in [1, T]
            2. Apply masking: masked = apply_mask(original_tokens, t, T)
            3. Get model predictions: logits = model_fn(masked, t)
            4. Compute loss = mdlm_loss_at_step(masked, original_ids, logits, t, T)
        Return average loss across all samples.

    Args:
        original_tokens: list of string tokens (ground truth)
        original_ids:    list of int token IDs (ground truth)
        model_fn:        function(masked_tokens, t) -> logits [seq_len x vocab_size]
        T:               total timesteps
        n_samples:       number of random timesteps to sample
        seed:            random seed

    Returns:
        float: estimated training loss

    HINT:
        rng = random.Random(seed)
        losses = []
        for _ in range(n_samples):
            t = rng.randint(1, T)     # sample random t (exclude t=0, no masking)
            masked = apply_mask(original_tokens, t, T)
            logits = model_fn(masked, t)
            loss = mdlm_loss_at_step(masked, original_ids, logits, t, T)
            losses.append(loss)
        return sum(losses) / len(losses)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# HELPERS NEEDED BY TASK 4
# ─────────────────────────────────────────────────────────

def apply_mask(tokens, t, T, seed=42):
    """Apply forward diffusion masking."""
    rng = random.Random(seed + t)
    p = t / T
    return [MASK if rng.random() < p else tok for tok in tokens]


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 04: MDLM Loss Function")
    print("=" * 55)

    # Task 1: softmax
    print("\n--- Task 1: softmax ---")
    result_uniform = softmax([1.0, 1.0, 1.0])
    if result_uniform is None:
        print("  NOT IMPLEMENTED YET")
    else:
        s = sum(result_uniform)
        status = "PASS" if abs(s - 1.0) < 0.001 else "FAIL"
        print(f"  {status}  softmax([1,1,1]) sums to {s:.4f}  (expected 1.0)")
        print(f"          values: {[round(p, 3) for p in result_uniform]}")

        result_dom = softmax([10.0, 0.0, 0.0])
        status = "PASS" if result_dom[0] > 0.99 else "FAIL"
        print(f"  {status}  softmax([10,0,0])[0] = {result_dom[0]:.4f}  (expected ~1.0)")

    # Task 2: cross_entropy
    print("\n--- Task 2: cross_entropy ---")
    if softmax([1.0]) is not None:
        probs1 = [0.1, 0.8, 0.1]
        result1 = cross_entropy(probs1, 1)
        expected1 = -math.log(0.8)
        if result1 is None:
            print("  NOT IMPLEMENTED YET")
        else:
            status = "PASS" if abs(result1 - expected1) < 0.001 else "FAIL"
            print(f"  {status}  CE([0.1,0.8,0.1], idx=1) = {result1:.4f}  (expected {expected1:.4f})")

            probs2 = [0.33, 0.33, 0.33]
            result2 = cross_entropy(probs2, 0)
            expected2 = -math.log(0.33)
            status = "PASS" if abs(result2 - expected2) < 0.01 else "FAIL"
            print(f"  {status}  CE([0.33,0.33,0.33], idx=0) = {result2:.4f}  (expected ~{expected2:.4f})")

    # Task 3: mdlm_loss_at_step
    print("\n--- Task 3: mdlm_loss_at_step ---")
    vocab_size = 5
    # Setup: sentence "The cat sat" with cat and sat masked
    masked_tokens = [MASK, "cat", MASK]
    original_ids = [2, 1, 4]   # The=2, cat=1, sat=4
    # Model logits: high logit for correct token at masked positions
    model_logits = [
        [0.1, 0.1, 5.0, 0.1, 0.1],   # position 0 (masked): high logit for id=2 (The)
        [0.2, 4.0, 0.2, 0.2, 0.2],   # position 1 (not masked): high for id=1 (cat)
        [0.1, 0.1, 0.1, 0.1, 5.0],   # position 2 (masked): high logit for id=4 (sat)
    ]
    T = 10
    t = 5

    if cross_entropy([0.5, 0.5], 0) is not None:
        result = mdlm_loss_at_step(masked_tokens, original_ids, model_logits, t, T)
        if result is None:
            print("  NOT IMPLEMENTED YET")
        else:
            # Manually compute expected:
            # position 0: CE(softmax([0.1,0.1,5,0.1,0.1]), 2) * weight
            # position 2: CE(softmax([0.1,0.1,0.1,0.1,5]), 4) * weight
            # position 1: NOT masked, skip
            w = t / T   # = 0.5
            p0 = softmax(model_logits[0])
            p2 = softmax(model_logits[2])
            l0 = cross_entropy(p0, original_ids[0]) * w
            l2 = cross_entropy(p2, original_ids[2]) * w
            expected = (l0 + l2) / 2
            print(f"  Result: {result:.4f}  Expected: {expected:.4f}")
            status = "PASS" if abs(result - expected) < 0.01 else "FAIL"
            print(f"  {status}  mdlm_loss_at_step (t={t}, T={T})")

            # t=0: no masks = loss should be 0
            no_mask = ["The", "cat", "sat"]
            result_t0 = mdlm_loss_at_step(no_mask, original_ids, model_logits, 0, T)
            status = "PASS" if result_t0 == 0.0 else "FAIL"
            print(f"  {status}  t=0 (no masks): loss = {result_t0}  (expected 0.0)")

    # Task 4: training_loss
    print("\n--- Task 4: training_loss ---")
    if mdlm_loss_at_step(masked_tokens, original_ids, model_logits, t, T) is not None:
        sentence = ["The", "cat", "sat", "on", "the", "mat"]
        token_ids = [3, 1, 2, 4, 0, 5]

        def fake_model_fn(masked_tokens, t):
            """Returns near-perfect logits for the correct tokens."""
            logits = []
            for i, tok in enumerate(masked_tokens):
                row = [0.0] * 6
                row[token_ids[i]] = 5.0    # high logit for correct token
                logits.append(row)
            return logits

        loss = training_loss(sentence, token_ids, fake_model_fn, T=10, n_samples=5)
        if loss is None:
            print("  NOT IMPLEMENTED YET")
        else:
            print(f"  Result: training_loss = {loss:.4f}")
            print(f"  (With near-perfect model, loss should be small but > 0 due to weighting)")
            status = "PASS" if 0.0 <= loss < 1.0 else "FAIL"
            print(f"  {status}  loss in expected range")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def softmax(logits):
#     max_l = max(logits)
#     exps = [math.exp(l - max_l) for l in logits]
#     total = sum(exps)
#     return [e / total for e in exps]
#
# def cross_entropy(probs, true_index):
#     p = max(probs[true_index], 1e-10)
#     return -math.log(p)
#
# def mdlm_loss_at_step(masked_tokens, original_ids, model_logits, t, T):
#     weight = t / T
#     total = 0.0
#     count = 0
#     for i, tok in enumerate(masked_tokens):
#         if tok == MASK:
#             probs = softmax(model_logits[i])
#             loss = cross_entropy(probs, original_ids[i])
#             total += loss * weight
#             count += 1
#     return total / count if count > 0 else 0.0
#
# def training_loss(original_tokens, original_ids, model_fn, T, n_samples=5, seed=42):
#     rng = random.Random(seed)
#     losses = []
#     for _ in range(n_samples):
#         t = rng.randint(1, T)
#         masked = apply_mask(original_tokens, t, T)
#         logits = model_fn(masked, t)
#         loss = mdlm_loss_at_step(masked, original_ids, logits, t, T)
#         losses.append(loss)
#     return sum(losses) / len(losses)

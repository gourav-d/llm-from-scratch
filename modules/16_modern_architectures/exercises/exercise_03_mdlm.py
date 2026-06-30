"""
Exercise 03: MDLM -- Masked Diffusion Language Model
Module 16: Modern LLM Architectures

TASKS:
  1. Implement is_absorbed()          -- check if a token has been masked
  2. Implement absorb_token()         -- apply absorbing-state transition
  3. Implement mdlm_forward()         -- forward process for full sentence
  4. Implement confidence_unmask()    -- commit highest-confidence predictions

Run:  python exercise_03_mdlm.py
Deps: none (pure Python)
"""

import random

MASK = "[M]"


# ─────────────────────────────────────────────────────────
# TASK 1: Check if Token is Absorbed (Masked)
# ─────────────────────────────────────────────────────────

def is_absorbed(token):
    """
    Return True if the token has been absorbed into the mask state.
    A token is absorbed if it equals MASK = "[M]".

    Args:
        token: string token

    Returns:
        bool: True if token is masked, False otherwise

    Example:
        is_absorbed("[M]") -> True
        is_absorbed("cat") -> False
        is_absorbed("the") -> False

    HINT: compare token to the MASK constant.
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Absorb One Token
# ─────────────────────────────────────────────────────────

def absorb_token(token, p_absorb, rng):
    """
    Apply the MDLM absorbing-state transition to a single token.

    Rules:
      - If the token is ALREADY masked: keep it as MASK (absorbing state -- no going back)
      - If the token is NOT masked:
          - With probability p_absorb: absorb it (return MASK)
          - With probability 1 - p_absorb: keep it unchanged

    Args:
        token:    string, the current token (may already be MASK)
        p_absorb: float in [0, 1], probability of absorbing an unmasked token
        rng:      random.Random instance for reproducibility

    Returns:
        string: MASK or the original token

    Example (with p_absorb=0.5):
        absorb_token("[M]",  0.5, rng) -> "[M]"  (always, already absorbed)
        absorb_token("cat",  0.5, rng) -> "[M]" or "cat" (50/50)
        absorb_token("cat",  1.0, rng) -> "[M]"  (always absorbed)
        absorb_token("cat",  0.0, rng) -> "cat"  (never absorbed)

    HINT:
        1. If is_absorbed(token): return MASK immediately
        2. Otherwise: if rng.random() < p_absorb: return MASK
        3. Otherwise: return token
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: MDLM Forward Process
# ─────────────────────────────────────────────────────────

def mdlm_forward(tokens, t, T, seed=42):
    """
    Apply the MDLM forward process to an entire sentence at timestep t.

    Uses a LINEAR noise schedule: p_absorb = t / T
    Each token is independently processed by absorb_token().

    Args:
        tokens: list of string tokens (the original sentence)
        t:      current timestep (0 <= t <= T)
        T:      total timesteps
        seed:   random seed for reproducibility

    Returns:
        list of string tokens, some replaced with MASK

    Example:
        mdlm_forward(["The", "cat", "sat"], t=0, T=10, seed=42)
        -> ["The", "cat", "sat"]   (t=0: p=0, nothing masked)

        mdlm_forward(["The", "cat", "sat"], t=10, T=10, seed=42)
        -> ["[M]", "[M]", "[M]"]   (t=T: p=1, all masked)

    HINT:
        1. p_absorb = t / T
        2. rng = random.Random(seed + t)
        3. For each token: call absorb_token(token, p_absorb, rng)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Confidence-Ordered Unmasking
# ─────────────────────────────────────────────────────────

def confidence_unmask(masked_tokens, predictions, top_k):
    """
    Apply one step of the reverse diffusion process.
    Commit the top_k most confident predictions from the model.

    predictions is a list of (position, predicted_token, confidence)
    for each masked position. We commit only the top_k most confident ones.

    Args:
        masked_tokens: list of current tokens (some are MASK)
        predictions:   list of (position, token, confidence) for masked positions
        top_k:         how many predictions to commit this step

    Returns:
        list of tokens with top_k masked positions filled in

    Example:
        masked = ["[M]", "cat", "[M]", "[M]"]
        preds  = [(0, "The", 0.95), (2, "sat", 0.72), (3, "on", 0.81)]
        confidence_unmask(masked, preds, top_k=1)
        -> ["The", "cat", "[M]", "[M]"]   (only position 0 committed -- highest conf)

        confidence_unmask(masked, preds, top_k=2)
        -> ["The", "cat", "[M]", "on"]    (positions 0 and 3 committed)

    HINT:
        1. Sort predictions by confidence (highest first)
        2. Take the first top_k predictions
        3. Make a copy of masked_tokens
        4. For each of the top_k: set result[position] = token
        5. Return result
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 03: MDLM -- Masked Diffusion Language Model")
    print("=" * 55)

    # Task 1: is_absorbed
    print("\n--- Task 1: is_absorbed ---")
    cases = [("[M]", True), ("cat", False), ("the", False), ("[M]", True)]
    for token, expected in cases:
        result = is_absorbed(token)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}  is_absorbed('{token}') = {result}  (expected {expected})")

    # Task 2: absorb_token
    print("\n--- Task 2: absorb_token ---")
    rng_fixed = random.Random(999)
    # Already masked token: must stay masked
    result_masked = absorb_token(MASK, 0.0, random.Random(0))
    if result_masked is None:
        print("  NOT IMPLEMENTED YET")
    else:
        status = "PASS" if result_masked == MASK else "FAIL"
        print(f"  {status}  absorb_token('[M]', p=0.0) = '{result_masked}'  (must stay [M])")

        result_never = absorb_token("cat", 0.0, random.Random(0))
        status = "PASS" if result_never == "cat" else "FAIL"
        print(f"  {status}  absorb_token('cat', p=0.0) = '{result_never}'  (expected 'cat', never absorbed)")

        result_always = absorb_token("cat", 1.0, random.Random(0))
        status = "PASS" if result_always == MASK else "FAIL"
        print(f"  {status}  absorb_token('cat', p=1.0) = '{result_always}'  (expected '[M]', always absorbed)")

    # Task 3: mdlm_forward
    print("\n--- Task 3: mdlm_forward ---")
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    T = 10
    result_t0 = mdlm_forward(sentence, 0, T)
    if result_t0 is None:
        print("  NOT IMPLEMENTED YET")
    else:
        status = "PASS" if result_t0 == sentence else "FAIL"
        print(f"  {status}  t=0: {result_t0}  (expected unchanged)")

        result_tT = mdlm_forward(sentence, T, T)
        all_masked = all(tok == MASK for tok in result_tT)
        status = "PASS" if all_masked else "FAIL"
        print(f"  {status}  t=T: {result_tT}  (expected all [M])")

        result_t5 = mdlm_forward(sentence, 5, T)
        n_m = sum(1 for t in result_t5 if t == MASK)
        print(f"  INFO  t=5: {result_t5}  ({n_m}/{len(sentence)} masked)")

    # Task 4: confidence_unmask
    print("\n--- Task 4: confidence_unmask ---")
    masked = [MASK, "cat", MASK, MASK]
    preds  = [(0, "The", 0.95), (2, "sat", 0.72), (3, "on", 0.81)]

    result_k1 = confidence_unmask(masked, preds, top_k=1)
    if result_k1 is None:
        print("  NOT IMPLEMENTED YET")
    else:
        expected_k1 = ["The", "cat", MASK, MASK]
        status = "PASS" if result_k1 == expected_k1 else "FAIL"
        print(f"  {status}  top_k=1: {result_k1}  (expected {expected_k1})")

        result_k2 = confidence_unmask(masked, preds, top_k=2)
        expected_k2 = ["The", "cat", MASK, "on"]
        status = "PASS" if result_k2 == expected_k2 else "FAIL"
        print(f"  {status}  top_k=2: {result_k2}  (expected {expected_k2})")

        result_k3 = confidence_unmask(masked, preds, top_k=3)
        expected_k3 = ["The", "cat", "sat", "on"]
        status = "PASS" if result_k3 == expected_k3 else "FAIL"
        print(f"  {status}  top_k=3: {result_k3}  (expected {expected_k3})")

    # Bonus: full generation simulation
    print("\n--- BONUS: Full MDLM Generation Simulation ---")
    if mdlm_forward(sentence, 0, T) is not None and confidence_unmask(masked, preds, 1) is not None:
        target = ["The", "cat", "sat", "on", "the", "mat"]
        current = [MASK] * len(target)
        T_gen = len(target)
        print(f"\n  Recovering: {target}")
        print(f"  Start: {current}")

        rng_gen = random.Random(7)
        for step in range(T_gen):
            mask_positions = [(i, target[i], rng_gen.uniform(0.6, 0.99))
                              for i, tok in enumerate(current) if tok == MASK]
            if not mask_positions:
                break
            current = confidence_unmask(current, mask_positions, top_k=1)
            print(f"  Step {step+1}: {current}")

        print(f"  Match: {current == target}")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def is_absorbed(token):
#     return token == MASK
#
# def absorb_token(token, p_absorb, rng):
#     if is_absorbed(token):
#         return MASK
#     if rng.random() < p_absorb:
#         return MASK
#     return token
#
# def mdlm_forward(tokens, t, T, seed=42):
#     rng = random.Random(seed + t)
#     p_absorb = t / T
#     return [absorb_token(tok, p_absorb, rng) for tok in tokens]
#
# def confidence_unmask(masked_tokens, predictions, top_k):
#     sorted_preds = sorted(predictions, key=lambda x: x[2], reverse=True)
#     result = list(masked_tokens)
#     for pos, token, conf in sorted_preds[:top_k]:
#         result[pos] = token
#     return result

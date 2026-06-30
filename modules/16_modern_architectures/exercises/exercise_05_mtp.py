"""
Exercise 05: Multi-Token Prediction (MTP)
Module 16: Modern LLM Architectures

TASKS:
  1. Implement get_target_token()    -- find target token for a given head offset
  2. Implement head_loss()           -- cross-entropy loss for one MTP head
  3. Implement mtp_loss()            -- total loss across all N heads with weights
  4. Implement gradient_signal_count() -- count gradient signals vs standard

Run:  python exercise_05_mtp.py
Deps: none (pure Python)
"""

import math
import random


def softmax(logits):
    """Convert logits to probabilities."""
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy(probs, true_index):
    """Cross-entropy loss: -log(P(correct))."""
    p = max(probs[true_index], 1e-10)
    return -math.log(p)


# ─────────────────────────────────────────────────────────
# TASK 1: Get Target Token for a Head
# ─────────────────────────────────────────────────────────

def get_target_token(token_ids, position, offset):
    """
    Get the target token ID for a prediction head at a given position.

    MTP head with offset k predicts the token at (position + k).

    Args:
        token_ids: list of int token IDs for the full sentence
        position:  int, current token position (0-indexed)
        offset:    int, how many steps ahead this head predicts (1, 2, 3, ...)

    Returns:
        int: target token ID, or None if (position + offset) is out of bounds

    Example:
        token_ids = [10, 20, 30, 40, 50]
        get_target_token(token_ids, position=1, offset=1) -> 30   (position 2)
        get_target_token(token_ids, position=1, offset=3) -> 40   (position 4)
        get_target_token(token_ids, position=3, offset=2) -> None  (out of bounds)

    HINT:
        target_pos = position + offset
        if target_pos < len(token_ids): return token_ids[target_pos]
        else: return None
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Loss for One MTP Head
# ─────────────────────────────────────────────────────────

def head_loss(logits, token_ids, position, offset):
    """
    Compute cross-entropy loss for a single MTP prediction head.

    This head predicts the token at (position + offset).
    If (position + offset) is out of bounds, return 0.0 (no loss for this head).

    Args:
        logits:    list of float raw scores for this head (one per vocab token)
        token_ids: list of int token IDs for the full sentence
        position:  current token position
        offset:    prediction horizon for this head

    Returns:
        float: cross-entropy loss, or 0.0 if target is out of bounds

    Example:
        logits = [0.1, 5.0, 0.2, 0.1]   (high score for index 1)
        token_ids = [3, 1, 2, 0]
        head_loss(logits, token_ids, position=0, offset=1) -> CE(softmax(logits), 1)

    HINT:
        1. target_id = get_target_token(token_ids, position, offset)
        2. If target_id is None: return 0.0
        3. probs = softmax(logits)
        4. return cross_entropy(probs, target_id)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Total MTP Loss
# ─────────────────────────────────────────────────────────

def mtp_loss(all_head_logits, token_ids, position, weights=None):
    """
    Compute the total MTP training loss at a single position.

    MTP loss = weighted sum of cross-entropy from each head.

    Args:
        all_head_logits: list of lists -- all_head_logits[h] = logits for head h+1
                         all_head_logits[0] = head 1 (offset=1, predicts t+1)
                         all_head_logits[1] = head 2 (offset=2, predicts t+2)
                         etc.
        token_ids:       list of int token IDs for the full sentence
        position:        current token position
        weights:         list of float weights for each head.
                         If None, use uniform weights (1/N for each head).

    Returns:
        float: weighted sum of head losses

    Example (N=3 heads, uniform weights):
        all_head_logits = [logits_h1, logits_h2, logits_h3]
        weights = None  -> weights = [1/3, 1/3, 1/3]
        mtp_loss = (1/3)*head_loss_1 + (1/3)*head_loss_2 + (1/3)*head_loss_3

    Example (N=3 heads, decreasing weights):
        weights = [1.0, 0.5, 0.33]   (head 1 is most important)
        mtp_loss = 1.0*hl1 + 0.5*hl2 + 0.33*hl3

    HINT:
        N = len(all_head_logits)
        if weights is None: weights = [1.0/N] * N
        total = 0.0
        for h, (logits, w) in enumerate(zip(all_head_logits, weights)):
            offset = h + 1           # head 0 predicts offset 1, head 1 predicts offset 2
            loss_h = head_loss(logits, token_ids, position, offset)
            total += w * loss_h
        return total
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Gradient Signal Count
# ─────────────────────────────────────────────────────────

def gradient_signal_count(sentence_length, n_heads):
    """
    Count the number of gradient signals produced by MTP vs standard training.

    Standard training (1 head):
      At each position i, one gradient signal for predicting token i+1.
      Total signals = sentence_length - 1  (last position has no next token)

    MTP training (N heads):
      At each position i, head h (offset h+1) produces a signal IF (i + h+1) < sentence_length.
      Total signals = sum over all positions and heads of valid predictions.

    Args:
        sentence_length: int, number of tokens in the sentence
        n_heads:         int, number of MTP prediction heads

    Returns:
        tuple: (standard_signals, mtp_signals)

    Example:
        gradient_signal_count(6, 4)
        standard = 5   (positions 0-4 each predict one next token)
        mtp      = ?   (depends on how many heads fit at each position)

    HINT:
        standard = sentence_length - 1
        mtp = 0
        for pos in range(sentence_length):
            for h in range(n_heads):
                offset = h + 1
                if pos + offset < sentence_length:
                    mtp += 1
        return (standard, mtp)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 05: Multi-Token Prediction (MTP)")
    print("=" * 55)

    token_ids = [10, 20, 30, 40, 50]

    # Task 1: get_target_token
    print("\n--- Task 1: get_target_token ---")
    cases = [
        (token_ids, 1, 1, 30),
        (token_ids, 1, 3, 40),
        (token_ids, 0, 1, 20),
        (token_ids, 3, 2, None),  # out of bounds
        (token_ids, 4, 1, None),  # out of bounds
    ]
    for ids, pos, offset, expected in cases:
        result = get_target_token(ids, pos, offset)
        if result is None and expected is not None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}  get_target_token(pos={pos}, offset={offset}) = {result}  (expected {expected})")

    # Task 2: head_loss
    print("\n--- Task 2: head_loss ---")
    logits_good = [0.1, 5.0, 0.2, 0.1, 0.1]   # high score for index 1
    ids = [3, 1, 2, 0, 4]
    if get_target_token(ids, 0, 1) is not None:
        result_valid = head_loss(logits_good, ids, position=0, offset=1)
        if result_valid is None:
            print("  NOT IMPLEMENTED YET")
        else:
            probs = softmax(logits_good)
            expected = cross_entropy(probs, ids[1])
            status = "PASS" if abs(result_valid - expected) < 0.001 else "FAIL"
            print(f"  {status}  head_loss(pos=0, offset=1) = {result_valid:.4f}  (expected {expected:.4f})")

            result_oob = head_loss(logits_good, ids, position=4, offset=1)
            status = "PASS" if result_oob == 0.0 else "FAIL"
            print(f"  {status}  head_loss(pos=4, offset=1) = {result_oob}  (out-of-bounds, expected 0.0)")

    # Task 3: mtp_loss
    print("\n--- Task 3: mtp_loss ---")
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    token_ids_s = [0, 1, 2, 3, 4, 5]
    N = 3
    # Fake logits for each head
    all_logits = [
        [0.1, 5.0, 0.1, 0.1, 0.1, 0.1],   # head 1: high for idx 1 (cat)
        [0.1, 0.1, 5.0, 0.1, 0.1, 0.1],   # head 2: high for idx 2 (sat)
        [0.1, 0.1, 0.1, 5.0, 0.1, 0.1],   # head 3: high for idx 3 (on)
    ]
    if head_loss(all_logits[0], token_ids_s, 0, 1) is not None:
        result_unif = mtp_loss(all_logits, token_ids_s, position=0, weights=None)
        if result_unif is None:
            print("  NOT IMPLEMENTED YET")
        else:
            # Manual: each head gets weight 1/3
            hl1 = head_loss(all_logits[0], token_ids_s, 0, 1)
            hl2 = head_loss(all_logits[1], token_ids_s, 0, 2)
            hl3 = head_loss(all_logits[2], token_ids_s, 0, 3)
            expected_unif = (hl1 + hl2 + hl3) / 3
            status = "PASS" if abs(result_unif - expected_unif) < 0.001 else "FAIL"
            print(f"  {status}  mtp_loss (uniform weights) = {result_unif:.4f}  (expected {expected_unif:.4f})")

            # Decreasing weights
            dec_weights = [1.0, 0.5, 1/3]
            result_dec = mtp_loss(all_logits, token_ids_s, position=0, weights=dec_weights)
            expected_dec = dec_weights[0]*hl1 + dec_weights[1]*hl2 + dec_weights[2]*hl3
            status = "PASS" if abs(result_dec - expected_dec) < 0.001 else "FAIL"
            print(f"  {status}  mtp_loss (decreasing weights) = {result_dec:.4f}  (expected {expected_dec:.4f})")

    # Task 4: gradient_signal_count
    print("\n--- Task 4: gradient_signal_count ---")
    if mtp_loss(all_logits, token_ids_s, 0) is not None:
        cases = [
            (6, 1, 5,  5),   # standard=5, mtp with 1 head=5 (same as standard)
            (6, 4, 5, 14),   # standard=5, mtp with 4 heads
        ]
        for slen, n_h, exp_std, exp_mtp in cases:
            result = gradient_signal_count(slen, n_h)
            if result is None:
                print("  NOT IMPLEMENTED YET")
                break
            std, mtp = result
            ok_std = "PASS" if std == exp_std else "FAIL"
            ok_mtp = "PASS" if mtp == exp_mtp else "FAIL"
            print(f"  {ok_std}  standard signals (N={n_h}): {std}  (expected {exp_std})")
            print(f"  {ok_mtp}  MTP signals (N={n_h}):      {mtp}  (expected {exp_mtp})")
            if mtp > 0 and std > 0:
                print(f"          Ratio: {mtp/std:.1f}x more gradient signals with N={n_h} heads")

    # Bonus: full comparison table
    print("\n--- BONUS: Gradient Signal Comparison Table ---")
    if gradient_signal_count(10, 4) is not None:
        print(f"\n  {'Sentence len':>14}  {'N heads':>8}  {'Standard':>10}  {'MTP':>8}  {'Ratio':>8}")
        print("  " + "-" * 52)
        for slen in [6, 10, 20, 50]:
            for n_h in [1, 2, 4]:
                std, mtp = gradient_signal_count(slen, n_h)
                ratio = mtp / std if std > 0 else 0
                print(f"  {slen:>14}  {n_h:>8}  {std:>10}  {mtp:>8}  {ratio:>7.1f}x")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def get_target_token(token_ids, position, offset):
#     target_pos = position + offset
#     if target_pos < len(token_ids):
#         return token_ids[target_pos]
#     return None
#
# def head_loss(logits, token_ids, position, offset):
#     target_id = get_target_token(token_ids, position, offset)
#     if target_id is None:
#         return 0.0
#     probs = softmax(logits)
#     return cross_entropy(probs, target_id)
#
# def mtp_loss(all_head_logits, token_ids, position, weights=None):
#     N = len(all_head_logits)
#     if weights is None:
#         weights = [1.0 / N] * N
#     total = 0.0
#     for h, (logits, w) in enumerate(zip(all_head_logits, weights)):
#         offset = h + 1
#         loss_h = head_loss(logits, token_ids, position, offset)
#         total += w * loss_h
#     return total
#
# def gradient_signal_count(sentence_length, n_heads):
#     standard = sentence_length - 1
#     mtp = 0
#     for pos in range(sentence_length):
#         for h in range(n_heads):
#             offset = h + 1
#             if pos + offset < sentence_length:
#                 mtp += 1
#     return (standard, mtp)

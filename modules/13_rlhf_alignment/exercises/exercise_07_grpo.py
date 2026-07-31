"""
=============================================================================
MODULE 13 - EXERCISE 07: GRPO Advantage and Loss
=============================================================================

YOUR TASK:
  Implement the core GRPO computations:
    1. Group advantage (z-score of rewards)
    2. Policy ratio
    3. Clipped policy gradient loss
    4. Full GRPO step loss

RULES:
  - Only import numpy (already imported below)
  - Read each docstring carefully
  - Tests at the bottom verify your work

RUN WITH:  python exercise_07_grpo.py
=============================================================================
"""

import numpy as np


# =============================================================================
# EXERCISE 1: Compute Group Advantage
# =============================================================================

def compute_advantage(rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute GRPO advantage for a group of rewards.

    advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

    This is a z-score: it measures how many standard deviations
    above/below the group mean each reward is.

    Parameters:
      rewards : 1D array of reward scores, shape (G,)
      eps     : small constant to prevent division by zero

    Returns:
      advantages : 1D array, same shape as rewards

    EXAMPLE:
      rewards    = [1.0, 0.0, 1.2, 0.0]
      mean       = 0.55
      std        = 0.524
      advantages = [(1.0-0.55)/0.524, (0.0-0.55)/0.524, ...]
                 = [+0.859, -1.050, +1.241, -1.050]

    HINTS:
      - np.mean(rewards) computes the mean
      - np.std(rewards) computes the standard deviation
      - Subtract mean, divide by (std + eps)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 2: Policy Ratio
# =============================================================================

def policy_ratio(log_probs_new: np.ndarray, log_probs_old: np.ndarray) -> np.ndarray:
    """
    Compute the probability ratio between new and old policy.

    ratio_t = prob_new(token_t) / prob_old(token_t)
            = exp(log_prob_new_t - log_prob_old_t)

    We use log probs for numerical stability:
      - Probabilities can be very small (e.g. 1e-50)
      - Working in log space avoids underflow
      - exp(log_a - log_b) = a / b

    Parameters:
      log_probs_new : log probabilities under current policy, shape (T,)
      log_probs_old : log probabilities under old/reference policy, shape (T,)

    Returns:
      ratio : shape (T,), each element is prob_new[t] / prob_old[t]

    HINTS:
      - np.exp(a - b) computes exp(a - b) element-wise
      - Result should be > 0 (always positive — ratios of probabilities)
      - If new == old policy, ratio should be close to 1.0
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Clipped Policy Gradient Loss
# =============================================================================

def clipped_pg_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantage: float,
    epsilon: float = 0.2
) -> float:
    """
    Compute the clipped policy gradient loss for ONE response.

    Formula:
      ratio   = policy_ratio(log_probs_new, log_probs_old)
      loss_t  = -min(ratio_t * advantage, clip(ratio_t, 1-eps, 1+eps) * advantage)
      loss    = mean over all tokens t

    Steps:
      1. Compute ratio using policy_ratio()
      2. Compute unclipped = ratio * advantage
      3. Clip ratio to [1-epsilon, 1+epsilon]
      4. Compute clipped = clipped_ratio * advantage
      5. Take minimum of unclipped and clipped (element-wise)
      6. Negate (we minimize loss = we maximize reward)
      7. Return mean over all tokens

    Parameters:
      log_probs_new : log probs under current policy, shape (T,)
      log_probs_old : log probs under old policy, shape (T,)
      advantage     : scalar advantage for this response
      epsilon       : clipping range (default 0.2)

    Returns:
      scalar mean loss across all tokens

    HINTS:
      - np.clip(array, min_val, max_val) clips values
      - np.minimum(a, b) element-wise minimum
      - np.mean(array) averages all elements
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Full GRPO Step
# =============================================================================

def grpo_step_loss(
    group_log_probs_new: list,
    group_log_probs_old: list,
    rewards: np.ndarray,
    epsilon: float = 0.2,
    beta: float = 0.01,
    log_probs_ref: list = None
) -> dict:
    """
    Compute the full GRPO loss for a group of G responses.

    Steps:
      1. Compute GRPO advantages from rewards using compute_advantage()
      2. For each response i: compute clipped_pg_loss(new[i], old[i], adv[i])
      3. Average pg_loss across all G responses
      4. If log_probs_ref provided: compute KL penalty
            kl_i = mean(exp(log_new) * (log_new - log_ref))
            kl_penalty = mean(kl_i for all i)
         Else: kl_penalty = 0.0
      5. total_loss = pg_loss + beta * kl_penalty

    Parameters:
      group_log_probs_new : list of G arrays, shape (T_i,) each
      group_log_probs_old : list of G arrays, shape (T_i,) each
      rewards             : shape (G,)
      epsilon             : clipping range
      beta                : KL penalty weight
      log_probs_ref       : optional list of G reference log prob arrays

    Returns:
      dict with keys:
        "total_loss"  : float — the full loss to backpropagate
        "pg_loss"     : float — policy gradient component
        "kl_penalty"  : float — KL divergence component
        "advantages"  : np.ndarray shape (G,) — the computed advantages

    HINTS:
      - Iterate over zip(group_log_probs_new, group_log_probs_old, advantages)
      - KL formula per token: prob_new * (log_new - log_ref)
        where prob_new = np.exp(log_new)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# TESTS — DO NOT MODIFY BELOW THIS LINE
# =============================================================================

def run_tests():
    print("=" * 55)
    print("EXERCISE 07 TESTS")
    print("=" * 55)
    np.random.seed(42)
    passed = 0
    total = 0

    def check(name, got, expected, tol=0.01):
        nonlocal passed, total
        total += 1
        if isinstance(expected, np.ndarray):
            ok = np.allclose(got, expected, atol=tol)
        else:
            ok = abs(float(got) - float(expected)) < tol
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {name}")
        if not ok:
            print(f"         Expected: {expected}")
            print(f"         Got:      {got}")

    # --- compute_advantage ---
    rewards_a = np.array([1.0, 0.0, 1.0, 0.0])
    adv_a = compute_advantage(rewards_a)
    check("advantage: sum is ~0",        np.sum(adv_a), 0.0)
    check("advantage: positive for 1.0", adv_a[0] > 0, True)
    check("advantage: negative for 0.0", adv_a[1] < 0, True)
    check("advantage: shape preserved",  len(adv_a), 4)

    # all same rewards -> advantages should all be ~0
    rewards_same = np.array([1.0, 1.0, 1.0])
    adv_same = compute_advantage(rewards_same)
    check("advantage: all same rewards -> all ~0", np.allclose(adv_same, 0, atol=0.01), True)

    # --- policy_ratio ---
    lp_new = np.array([-0.5, -1.0, -0.3])
    lp_old = np.array([-0.5, -1.0, -0.3])
    ratio_same = policy_ratio(lp_new, lp_old)
    check("ratio: same policy -> ratio ~1", np.allclose(ratio_same, 1.0, atol=0.01), True)

    lp_new2 = np.array([-0.3, -0.8])    # higher probs (less negative log)
    lp_old2 = np.array([-0.5, -1.0])
    ratio2 = policy_ratio(lp_new2, lp_old2)
    check("ratio: new > old -> ratio > 1", np.all(ratio2 > 1.0), True)
    check("ratio: all positive",           np.all(ratio2 > 0), True)

    # --- clipped_pg_loss ---
    lp_n = np.full(10, -0.5)   # uniform log probs
    lp_o = np.full(10, -0.5)   # same old policy
    loss_adv_pos = clipped_pg_loss(lp_n, lp_o, advantage=1.0)
    loss_adv_neg = clipped_pg_loss(lp_n, lp_o, advantage=-1.0)
    check("pg_loss: positive adv -> negative loss", loss_adv_pos < 0, True)
    check("pg_loss: negative adv -> positive loss", loss_adv_neg > 0, True)
    check("pg_loss: symmetric",
          abs(abs(loss_adv_pos) - abs(loss_adv_neg)) < 0.01, True)

    # --- grpo_step_loss ---
    G, T = 4, 15
    lp_new_g = [np.random.uniform(-2, -0.1, T) for _ in range(G)]
    lp_old_g = [lp + np.random.normal(0, 0.05, T) for lp in lp_new_g]
    lp_ref_g = [np.random.uniform(-2, -0.1, T) for _ in range(G)]
    rew = np.array([1.0, 0.0, 1.2, 0.5])

    result = grpo_step_loss(lp_new_g, lp_old_g, rew,
                            epsilon=0.2, beta=0.01, log_probs_ref=lp_ref_g)

    check("grpo_step: returns dict",    isinstance(result, dict), True)
    check("grpo_step: has total_loss",  "total_loss"  in result, True)
    check("grpo_step: has pg_loss",     "pg_loss"     in result, True)
    check("grpo_step: has kl_penalty",  "kl_penalty"  in result, True)
    check("grpo_step: has advantages",  "advantages"  in result, True)
    check("grpo_step: adv shape (G,)",  len(result["advantages"]), G)
    check("grpo_step: kl >= 0",         result["kl_penalty"] >= 0, True)

    print(f"\n{passed}/{total} tests passed")
    if passed == total:
        print("All tests passed!")
    else:
        print("Some tests failed. Re-read the docstrings and try again.")


if __name__ == "__main__":
    run_tests()

"""
=============================================================================
MODULE 13 - EXAMPLE 07: GRPO — Group Relative Policy Optimization
=============================================================================

WHAT YOU WILL LEARN:
  - Why PPO needs a value network and why that's expensive for LLMs
  - How GRPO eliminates the value network using group-relative advantages
  - How to compute the GRPO advantage (z-score of rewards in a group)
  - How the GRPO policy gradient loss works (clip + KL penalty)
  - Side-by-side comparison of PPO vs GRPO advantage computation

C# ANALOGY:
  GRPO is like benchmarking N implementations of the same function,
  then training a code generator to prefer the relatively better ones:

    var results = implementations.Select(RunBenchmark).ToArray();
    var mean = results.Average();
    var std = Math.Sqrt(results.Select(r => Math.Pow(r - mean, 2)).Average());
    var advantages = results.Select(r => (r - mean) / (std + 1e-8)).ToArray();
    // Train on: make high-advantage implementations more likely

  You don't need a separate "value estimator" — the group IS the baseline.

=============================================================================

PART A: NumPy implementation — no external libraries required
Run with: python example_07_grpo.py

=============================================================================
"""

import numpy as np   # numerical computing — arrays, math operations


# =============================================================================
# SECTION 1: THE PROBLEM WITH PPO (WHY GRPO WAS NEEDED)
# =============================================================================

def explain_ppo_cost():
    """
    Demonstrate the memory cost of PPO for large LLMs.

    PPO requires a value network (critic) which doubles memory usage.
    For large models this becomes prohibitive.
    """
    print("=" * 60)
    print("PPO MEMORY COST FOR LLMs")
    print("=" * 60)

    model_sizes = [
        ("7B",  7_000_000_000),
        ("13B", 13_000_000_000),
        ("70B", 70_000_000_000),
    ]

    bytes_per_param_bf16 = 2    # bf16 = 2 bytes per parameter
    GB = 1024 ** 3              # bytes in a gigabyte

    print(f"\n{'Model':8s} | {'Policy':8s} | {'+ Critic':10s} | {'+ Ref':10s} | {'Total PPO':10s} | {'Total GRPO':10s}")
    print("-" * 72)

    for name, params in model_sizes:
        policy_gb   = (params * bytes_per_param_bf16) / GB
        critic_gb   = policy_gb      # value network is same size as policy
        ref_gb      = policy_gb      # reference model (frozen copy)

        ppo_total   = policy_gb + critic_gb + ref_gb   # policy + critic + reference
        grpo_total  = policy_gb + ref_gb               # policy + reference (no critic)

        print(f"{name:8s} | {policy_gb:6.0f} GB | {critic_gb:8.0f} GB | {ref_gb:8.0f} GB | "
              f"{ppo_total:8.0f} GB | {grpo_total:8.0f} GB")

    print("\nGRPO saves ~33% memory by eliminating the value network.")
    print("For 70B models: saves 140GB of GPU memory.")


# =============================================================================
# SECTION 2: GRPO ADVANTAGE COMPUTATION
# =============================================================================

def compute_grpo_advantage(rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute the GRPO advantage for a group of rewards.

    GRPO advantage = (reward - group_mean) / (group_std + eps)

    This is exactly a z-score: how many standard deviations above/below
    the group mean is each reward?

    Parameters:
      rewards : 1D numpy array of reward scores for G responses
                shape: (G,)  e.g. [1.0, 1.2, 0.0, 1.0, 0.5, 1.0, 0.0, 1.0]
      eps     : small constant to prevent division by zero when all rewards are equal

    Returns:
      advantages : 1D numpy array, same shape as rewards
                   positive = better than group average
                   negative = worse than group average

    Example:
      rewards    = [1.0, 0.0, 1.2, 0.0]
      mean       = 0.55
      std        = 0.54
      advantages = [+0.83, -1.02, +1.20, -1.02]
    """
    # Step 1: compute group mean (average reward across all G responses)
    # np.mean() computes the arithmetic mean
    # C# analogy: rewards.Average()
    group_mean = np.mean(rewards)

    # Step 2: compute group standard deviation
    # std measures spread — how different the rewards are from each other
    # C# analogy: Math.Sqrt(rewards.Select(r => Math.Pow(r - mean, 2)).Average())
    group_std = np.std(rewards)

    # Step 3: compute advantage as z-score
    # subtract mean to center around 0
    # divide by std to normalize spread
    # add eps to prevent division by zero (when all rewards are the same)
    advantages = (rewards - group_mean) / (group_std + eps)

    return advantages


def demonstrate_advantage():
    """
    Show advantage computation on a concrete example.
    """
    print("\n" + "=" * 60)
    print("GRPO ADVANTAGE COMPUTATION")
    print("=" * 60)

    # Simulate 8 responses to the same math prompt
    # "What is 15% of 80?"  (correct answer: 12)
    print("\nPrompt: 'What is 15% of 80?' (correct = 12)")
    print(f"\n{'Response':6s} | {'Content':35s} | {'Reward':8s} | {'Advantage':10s}")
    print("-" * 70)

    responses = [
        ("R1", "12",                                          1.0),
        ("R2", "<think>0.15 * 80 = 12</think> 12",           1.2),
        ("R3", "11",                                          0.0),
        ("R4", "12.0",                                        1.0),
        ("R5", "I think around 12",                           0.5),
        ("R6", "12",                                          1.0),
        ("R7", "The answer is 13",                            0.0),
        ("R8", "<think>15/100 * 80</think> 12",               1.2),
    ]

    rewards = np.array([r[2] for r in responses])
    advantages = compute_grpo_advantage(rewards)

    for (label, content, reward), adv in zip(responses, advantages):
        sign = "+" if adv >= 0 else ""
        bar_len = int(abs(adv) * 10)
        bar = (">" * bar_len) if adv > 0 else ("<" * bar_len)
        print(f"{label:6s} | {content:35s} | {reward:8.1f} | {sign}{adv:+.3f} {bar}")

    print(f"\nGroup mean = {rewards.mean():.3f}")
    print(f"Group std  = {rewards.std():.3f}")
    print(f"\nResponses with advantage > 0: will be made MORE likely")
    print(f"Responses with advantage < 0: will be made LESS likely")


# =============================================================================
# SECTION 3: POLICY GRADIENT LOSS
# =============================================================================

def compute_policy_ratio(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray
) -> np.ndarray:
    """
    Compute the policy ratio for each token.

    ratio = prob(token | new_policy) / prob(token | old_policy)
          = exp(log_prob_new - log_prob_old)

    Why use log probs?
      - Probabilities can be tiny (e.g. 1e-40)
      - Log transforms them to a workable range (-inf to 0)
      - exp(a - b) = exp(a) / exp(b) = prob_new / prob_old
      - Numerically stable

    Parameters:
      log_probs_new : log probabilities under current policy,  shape (T,)
      log_probs_old : log probabilities under old policy,      shape (T,)

    Returns:
      ratio : shape (T,), each value is prob_new[t] / prob_old[t]
    """
    # exp(log_new - log_old) = exp(log_new) / exp(log_old) = prob_new / prob_old
    ratio = np.exp(log_probs_new - log_probs_old)
    return ratio


def clip_policy_ratio(ratio: np.ndarray, epsilon: float = 0.2) -> np.ndarray:
    """
    Clip the policy ratio to [1 - epsilon, 1 + epsilon].

    This is the "Proximal" part of PPO/GRPO.

    Why clip?
      Without clipping, a single large gradient step could make the policy
      very different from what it was — "catastrophic" update.
      Clipping limits how much the policy can change in one step.

    Example:
      epsilon = 0.2
      ratio = 2.5  -->  clipped to 1.2  (can't increase policy by more than 20%)
      ratio = 0.3  -->  clipped to 0.8  (can't decrease policy by more than 20%)
      ratio = 1.1  -->  stays 1.1       (within safe range)

    Parameters:
      ratio   : policy ratio, shape (T,)
      epsilon : clip range (typically 0.1 to 0.2)

    Returns:
      clipped ratio, same shape
    """
    # np.clip(a, min, max) clips all values to [min, max]
    # C# analogy: Math.Clamp(ratio, 1 - epsilon, 1 + epsilon)
    return np.clip(ratio, 1 - epsilon, 1 + epsilon)


def compute_grpo_token_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantage: float,
    epsilon: float = 0.2
) -> float:
    """
    Compute the GRPO policy gradient loss for ONE response.

    Loss per token = -min(ratio * advantage, clip(ratio, ...) * advantage)

    Why the minimum?
      The min ensures we are conservative:
        - If advantage > 0: we want to increase ratio. Clip prevents going too high.
        - If advantage < 0: we want to decrease ratio. Clip prevents going too low.
      Taking the min makes the update conservative in both directions.

    Parameters:
      log_probs_new : log probs under current policy, shape (T,)
      log_probs_old : log probs under reference/old policy, shape (T,)
      advantage     : scalar — the GRPO advantage for this response
      epsilon       : clipping range

    Returns:
      mean loss across all T tokens (scalar)
    """
    # Step 1: compute probability ratio
    ratio = compute_policy_ratio(log_probs_new, log_probs_old)

    # Step 2: unclipped objective = ratio * advantage
    unclipped = ratio * advantage

    # Step 3: clipped objective = clip(ratio, ...) * advantage
    clipped_ratio = clip_policy_ratio(ratio, epsilon)
    clipped = clipped_ratio * advantage

    # Step 4: take the minimum (conservative estimate)
    # For advantage > 0: min(large, smaller) = smaller (conservative)
    # For advantage < 0: min(negative, less negative) = more negative (conservative)
    conservative = np.minimum(unclipped, clipped)

    # Step 5: negate (we MINIMIZE loss, but we want to MAXIMIZE reward)
    # So loss = -objective
    token_losses = -conservative

    # Step 6: return mean loss across all tokens
    return float(np.mean(token_losses))


def compute_kl_penalty(
    log_probs_current: np.ndarray,
    log_probs_reference: np.ndarray
) -> float:
    """
    Compute the KL divergence penalty between current and reference policy.

    KL(current || reference) = mean over tokens of:
      exp(log_current) * (log_current - log_reference)
      = prob_current * (log_current - log_reference)

    This penalty prevents the model from drifting too far from the
    pretrained reference — it keeps the model from "forgetting" everything
    while learning the new task.

    Parameters:
      log_probs_current   : log probs from model being trained, shape (T,)
      log_probs_reference : log probs from frozen reference model, shape (T,)

    Returns:
      scalar KL divergence (always >= 0)
    """
    # Convert log probs to probs for current model
    # exp(log_prob) = prob
    probs_current = np.exp(log_probs_current)

    # KL formula: sum of p(x) * log(p(x)/q(x)) = sum of p(x) * (log_p - log_q)
    kl_per_token = probs_current * (log_probs_current - log_probs_reference)

    # Average over all tokens
    return float(np.mean(kl_per_token))


def compute_grpo_total_loss(
    group_log_probs_new: list,
    group_log_probs_old: list,
    group_log_probs_ref: list,
    rewards: np.ndarray,
    epsilon: float = 0.2,
    beta: float = 0.01
) -> float:
    """
    Compute the full GRPO loss for a group of G responses.

    total_loss = mean_over_group(policy_gradient_loss) + beta * kl_penalty

    Parameters:
      group_log_probs_new : list of G arrays, each shape (T_i,) — current policy
      group_log_probs_old : list of G arrays, each shape (T_i,) — old policy (before step)
      group_log_probs_ref : list of G arrays, each shape (T_i,) — frozen reference
      rewards             : array shape (G,) — RLVR reward for each response
      epsilon             : clipping range for policy ratio
      beta                : weight of KL penalty

    Returns:
      scalar total loss
    """
    G = len(rewards)

    # Step 1: compute GRPO advantages from group rewards
    advantages = compute_grpo_advantage(rewards)

    # Step 2: compute policy gradient loss for each response
    pg_losses = []
    kl_penalties = []

    for i in range(G):
        # Policy gradient loss for response i
        pg_loss = compute_grpo_token_loss(
            group_log_probs_new[i],
            group_log_probs_old[i],
            advantages[i],
            epsilon
        )
        pg_losses.append(pg_loss)

        # KL penalty: how far did we drift from the reference?
        kl = compute_kl_penalty(group_log_probs_new[i], group_log_probs_ref[i])
        kl_penalties.append(kl)

    # Step 3: average across the group
    mean_pg_loss = float(np.mean(pg_losses))
    mean_kl      = float(np.mean(kl_penalties))

    # Step 4: combine
    total_loss = mean_pg_loss + beta * mean_kl

    return total_loss


# =============================================================================
# SECTION 4: PPO vs GRPO COMPARISON
# =============================================================================

def compare_ppo_vs_grpo():
    """
    Side-by-side demonstration of how PPO and GRPO compute advantages differently.
    """
    print("\n" + "=" * 60)
    print("PPO vs GRPO: ADVANTAGE COMPUTATION")
    print("=" * 60)

    # Simulate rewards for 4 responses
    rewards = np.array([1.0, 0.0, 1.2, 0.5])

    # --- PPO: advantage from value network ---
    # Value network estimates the "expected" reward for this prompt.
    # This would come from a trained neural network. We simulate it:
    value_estimate = 0.65    # value network guesses the expected reward is 0.65
    ppo_advantages = rewards - value_estimate   # advantage = actual - expected

    # --- GRPO: advantage from group mean ---
    grpo_advantages = compute_grpo_advantage(rewards)

    print(f"\n{'Response':10s} | {'Reward':8s} | {'PPO Adv':10s} | {'GRPO Adv':10s}")
    print("-" * 50)
    for i, (r, ppo_a, grpo_a) in enumerate(zip(rewards, ppo_advantages, grpo_advantages)):
        print(f"  R{i+1:7s} | {r:8.1f} | {ppo_a:+10.3f} | {grpo_a:+10.3f}")

    print(f"\nPPO  baseline = value_network_estimate = {value_estimate:.2f} (LEARNED, can be wrong)")
    print(f"GRPO baseline = group_mean             = {rewards.mean():.2f} (COMPUTED, always correct)")
    print("\nKey difference: GRPO's baseline is exact for this group.")
    print("PPO's baseline depends on how well the value network was trained.")


# =============================================================================
# SECTION 5: FULL GRPO STEP DEMO
# =============================================================================

def run_single_grpo_step():
    """
    Demonstrate a complete GRPO training step with simulated data.
    """
    np.random.seed(42)

    print("\n" + "=" * 60)
    print("FULL GRPO TRAINING STEP (SIMULATED)")
    print("=" * 60)

    G = 8       # group size (number of responses per prompt)
    T = 20      # tokens per response (simplified to fixed length)

    print(f"\nGroup size G = {G}, tokens per response T = {T}")

    # Simulate log probabilities for each response
    # In reality, these come from a forward pass through the LLM
    # log probs are in (-inf, 0] since log(prob) where prob in (0,1]

    # "Old" policy = model at start of this step (before gradient update)
    # We use these to compute the ratio
    log_probs_old = [np.random.uniform(-2.0, -0.1, T) for _ in range(G)]

    # "Current" policy = model after a small gradient update
    # Slightly different from old policy
    log_probs_new = [lp + np.random.normal(0, 0.05, T) for lp in log_probs_old]

    # "Reference" policy = frozen original model (pretrained weights, never updated)
    # This is the "leash" — KL penalty compares current to this
    log_probs_ref = [np.random.uniform(-2.0, -0.1, T) for _ in range(G)]

    # Simulated rewards from RLVR checker
    rewards = np.array([1.0, 1.2, 0.0, 1.0, 0.5, 1.0, 0.0, 1.0])

    # Compute advantages
    advantages = compute_grpo_advantage(rewards)

    print(f"\n{'Response':10s} | {'Reward':8s} | {'Advantage':10s} | {'Meaning':20s}")
    print("-" * 55)
    for i in range(G):
        meaning = "make MORE likely" if advantages[i] > 0 else "make LESS likely"
        print(f"  R{i+1:7s} | {rewards[i]:8.1f} | {advantages[i]:+10.3f} | {meaning}")

    # Compute total GRPO loss
    total_loss = compute_grpo_total_loss(
        log_probs_new, log_probs_old, log_probs_ref,
        rewards, epsilon=0.2, beta=0.01
    )

    print(f"\nTotal GRPO loss for this step: {total_loss:.4f}")
    print("(Backpropagate this loss to update model weights)")


# =============================================================================
# SECTION 6: ANSWERS TO LESSON SELF-CHECK
# =============================================================================

def print_lesson_answers():
    print("\n" + "=" * 60)
    print("LESSON 07 SELF-CHECK ANSWERS")
    print("=" * 60)

    answers = [
        ("Q1: Main difference PPO vs GRPO?",
         "PPO needs a value network (critic) to estimate the baseline. "
         "GRPO computes the baseline from the group mean of rewards. "
         "No value network = no extra model = less memory."),

        ("Q2: How is GRPO advantage computed?",
         "advantage = (reward - group_mean) / (group_std + eps). "
         "It is a z-score: how many stdevs above the group average is this response?"),

        ("Q3: Why divide by standard deviation?",
         "Normalizes the scale so advantages are always in a consistent range. "
         "Without this, different prompts with different reward scales would "
         "produce very different gradient magnitudes, making training unstable."),

        ("Q4: What does KL penalty prevent?",
         "It prevents the policy from drifting too far from the reference model. "
         "Without it, the model might 'forget' general language abilities "
         "while learning the specific task (math, code, etc.)."),

        ("Q5: rewards=[1,1,0,1,0,1,1,0], reward=1 advantage?",
         "mean = 5/8 = 0.625, "
         "std = sqrt(mean((r - 0.625)^2)) = 0.484, "
         "advantage = (1 - 0.625) / 0.484 = +0.775"),
    ]

    for q, a in answers:
        print(f"\n{q}")
        print(f"   {a}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("MODULE 13 - EXAMPLE 07: GRPO")

    explain_ppo_cost()
    demonstrate_advantage()
    compare_ppo_vs_grpo()
    run_single_grpo_step()
    print_lesson_answers()

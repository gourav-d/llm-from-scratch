"""
=============================================================================
MODULE 13 - EXERCISE 03: PPO Core Mechanics
=============================================================================

WHAT YOU WILL LEARN:
  - What the importance ratio is and why PPO uses it
  - How the clip function prevents the policy from changing too fast
  - What the PPO loss formula does step by step
  - How KL divergence measures the "distance" between two probability distributions
  - How to combine all three into a real PPO update computation

C# ANALOGY:
  PPO is like a version control system for your AI policy:
  - The OLD policy (before this update) is like the "main branch"
  - The NEW policy (being trained) is like your "feature branch"
  - The CLIP function is like a branch protection rule that says:
    "you can't deviate from main by more than 20%"
  - KL divergence is like git diff — it measures HOW DIFFERENT the two
    policies are, and you add a penalty if they drift too far apart

=============================================================================

PPO CLIPPING DIAGRAM
====================

  ratio = new_policy_prob / old_policy_prob

  When advantage > 0 (the action was GOOD):
    We want to INCREASE the probability of this action.
    But we clip at (1 + epsilon) so we don't go too far.

    Loss contribution:
    |                     ___________  <- clipped ceiling
    |            ________/
    |           /
    |__________/
    0         1-eps  1   1+eps   ratio

  When advantage < 0 (the action was BAD):
    We want to DECREASE the probability.
    Clip at (1 - epsilon) as the floor.

=============================================================================
"""

# ---- GLOSSARY ---------------------------------------------------------------
GLOSSARY = {
    "Policy":
        "The LLM itself — given an input, it outputs probability of each next token.",
    "Importance Ratio":
        "new_prob / old_prob — how much the policy has changed for a specific action.",
    "Advantage":
        "How much better (or worse) this action was compared to the baseline average.",
    "Clipping":
        "Restricting the ratio to [1-eps, 1+eps] so the policy doesn't change too fast.",
    "PPO Loss":
        "-min(ratio * advantage, clip(ratio) * advantage) — the clipped policy gradient.",
    "KL Divergence":
        "A measure of how different two probability distributions are. 0 = identical.",
    "Epsilon":
        "The clip threshold in PPO. Usually 0.1 to 0.2. Controls how much policy can change.",
}

print("=" * 70)
print("EXERCISE 03 — PPO Core Mechanics")
print("=" * 70)
print("\nGLOSSARY:")
for term, defn in GLOSSARY.items():
    print(f"  {term}:\n    {defn}\n")

import numpy as np    # NumPy for all maths

# =============================================================================
# SHARED DATA
# =============================================================================
# Simulated probability distributions over 3 possible actions (tokens).
# Each array sums to 1.0 (it's a probability distribution).

OLD_PROBS   = np.array([0.6, 0.3, 0.1], dtype=np.float32)    # old policy probs
POLICY_PROBS = np.array([0.7, 0.2, 0.1], dtype=np.float32)   # new policy probs
# Both sum to 1.0 (probability distributions)

EPSILON    = 0.2    # PPO clip threshold — ratio must stay in [1-eps, 1+eps] = [0.8, 1.2]
ADVANTAGE  = 0.5    # advantage value for this action — positive means it was a GOOD action

# =============================================================================
# EXERCISE 1 — Clip Ratio
# =============================================================================
# Given an importance ratio and epsilon, clip the ratio to [1-epsilon, 1+epsilon].
#
# Example: ratio=1.5, epsilon=0.2 -> clipped=1.2  (hit the upper ceiling)
#          ratio=0.7, epsilon=0.2 -> clipped=0.8  (hit the lower floor)
#          ratio=1.1, epsilon=0.2 -> clipped=1.1  (within range, no change)
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 1 — Clip Ratio")
print("=" * 60)
print()
print("  Formula: clipped = clip(ratio, 1 - epsilon, 1 + epsilon)")
print("  Hint: np.clip(value, min_val, max_val)")
print()

def clip_ratio(ratio, epsilon=0.2):
    """
    Clip the importance ratio to [1 - epsilon, 1 + epsilon].

    Parameters:
      ratio   : float or np.array — importance ratio (new_prob / old_prob)
      epsilon : float — clip threshold (default 0.2)

    Returns:
      clipped ratio (same type as input)
    """
    # TODO: clip ratio to [1 - epsilon, 1 + epsilon]
    clipped = None    # replace with np.clip(ratio, 1 - epsilon, 1 + epsilon)
    return clipped

# --- Test clip_ratio ---
test_ratios = [0.5, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0]    # various ratio values
print("  Testing clip_ratio(ratio, epsilon=0.2):")
print(f"  {'ratio':<8} {'clipped':<10} {'interpretation'}")
print(f"  {'-'*50}")
for r in test_ratios:
    clipped = clip_ratio(r, EPSILON)    # call our function
    if clipped is not None:
        if r < 1 - EPSILON:
            interp = "hit LOWER floor"
        elif r > 1 + EPSILON:
            interp = "hit UPPER ceiling"
        else:
            interp = "within range (no change)"
        print(f"  {r:<8.2f} {clipped:<10.4f} {interp}")

# =============================================================================
# EXERCISE 2 — PPO Loss
# =============================================================================
# PPO loss for a single (action, advantage) pair:
#   unclipped = ratio * advantage
#   clipped   = clip_ratio(ratio) * advantage
#   ppo_loss  = -min(unclipped, clipped)
#
# The negative sign makes it a MINIMIZATION problem.
# We minimize the loss = we maximize the expected return.
#
# WHY min(unclipped, clipped)?
#   - If advantage > 0 (good action): min keeps us from giving too much credit
#   - If advantage < 0 (bad action): min prevents exploiting the clipping
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 2 — PPO Loss")
print("=" * 60)
print()
print("  Formula: ppo_loss = -min(ratio * advantage, clip(ratio) * advantage)")
print("  Hint: np.minimum(a, b) returns element-wise minimum")
print()

def ppo_loss(ratio, advantage, epsilon=0.2):
    """
    Compute the PPO clipped surrogate loss for a single action.

    Parameters:
      ratio     : float or np.array — importance ratio new_prob/old_prob
      advantage : float — how good (positive) or bad (negative) the action was
      epsilon   : float — clip threshold (default 0.2)

    Returns:
      loss : float — PPO loss (lower = policy has improved more safely)
    """
    # TODO: Step 1 — compute unclipped term: ratio * advantage
    unclipped = None

    # TODO: Step 2 — compute clipped ratio and then clipped term
    clipped_r = None    # clip_ratio(ratio, epsilon)
    clipped   = None    # clipped_r * advantage

    # TODO: Step 3 — loss = -min(unclipped, clipped)
    loss = None    # -np.minimum(unclipped, clipped)

    return loss

# --- Test with different advantages ---
print("  PPO loss for ratio=1.3 with different advantages:")
for adv in [1.0, 0.5, 0.0, -0.5, -1.0]:
    ratio_val = 1.3    # ratio > 1 means new policy assigns higher probability
    loss_val = ppo_loss(ratio_val, adv, EPSILON)
    print(f"    ratio={ratio_val}, advantage={adv:+.1f}  ->  ppo_loss={loss_val}")

# =============================================================================
# EXERCISE 3 — KL Divergence
# =============================================================================
# KL divergence measures how different distribution Q is from distribution P.
#   KL(P || Q) = sum( P(x) * log( P(x) / Q(x) ) )
#
# Properties:
#   KL(P || P) = 0  (identical distributions have zero divergence)
#   KL >= 0  always
#   KL(P || Q) != KL(Q || P)  (not symmetric!)
#
# In RLHF/PPO:
#   P = old policy
#   Q = new policy
#   We add beta * KL(old || new) to the loss to prevent the model from
#   drifting too far from the reference (pre-RLHF) model.
#
# C# analogy: KL divergence is like the "git diff size" between two branches.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 3 — KL Divergence")
print("=" * 60)
print()
print("  Formula: KL(P || Q) = sum( P * log(P / (Q + eps)) )")
print("  Hint: np.sum() sums all elements of an array")
print("  Hint: np.log() is natural log (base e)")
print("  Hint: add 1e-8 inside the log to avoid division by zero")
print()

def kl_divergence(p, q):
    """
    Compute the KL divergence from distribution p to distribution q.
    KL(p || q) = sum( p * log(p / q) )

    Parameters:
      p : np.array — reference (old) distribution, must sum to 1
      q : np.array — new distribution, must sum to 1

    Returns:
      kl : float — KL divergence (0 = identical, larger = more different)
    """
    # TODO: compute KL(p || q)
    # Hint: element-wise: p * np.log(p / (q + 1e-8))
    # Then sum all elements
    kl = None
    return kl

# --- Test kl_divergence ---
print("  Testing KL divergence:")
# Case 1: identical distributions -> KL should be 0
kl_same = kl_divergence(OLD_PROBS, OLD_PROBS.copy())
print(f"  KL(old || old) = {kl_same}  (expected: ~0.0 — same distribution)")

# Case 2: slightly different distributions
kl_diff = kl_divergence(OLD_PROBS, POLICY_PROBS)
print(f"  KL(old || new) = {kl_diff}  (expected: small positive number)")

# Case 3: very different distributions
very_different = np.array([0.01, 0.01, 0.98], dtype=np.float32)    # concentrated on action 2
kl_very_diff = kl_divergence(OLD_PROBS, very_different)
print(f"  KL(old || very_different) = {kl_very_diff}  (expected: larger than Case 2)")

# =============================================================================
# EXERCISE 4 — Full PPO Computation: Ratios, Clipped Loss, and KL
# =============================================================================
# Given:
#   policy_probs = [0.7, 0.2, 0.1]   (new policy)
#   old_probs    = [0.6, 0.3, 0.1]   (old policy)
#   advantage    = 0.5               (the action taken was good)
#
# For each action:
#   1. Compute importance ratio = policy_prob[a] / old_prob[a]
#   2. Compute PPO loss for that action with the given advantage
#
# Then:
#   3. Compute KL divergence between old and new distributions
#   4. Print a summary table
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 4 — Full PPO Computation")
print("=" * 60)
print()
print("  new policy probs:", POLICY_PROBS)
print("  old policy probs:", OLD_PROBS)
print(f"  advantage = {ADVANTAGE},  epsilon = {EPSILON}")
print()
print("  Task: compute ratio, clipped_ratio, and ppo_loss for each action.")
print("  Then compute KL(old || new) for the full distributions.")
print()

def full_ppo_computation(policy_probs, old_probs, advantage, epsilon=0.2):
    """
    Compute PPO update quantities for all actions in a distribution.

    Parameters:
      policy_probs : np.array [N] — new policy probabilities for N actions
      old_probs    : np.array [N] — old policy probabilities for N actions
      advantage    : float — shared advantage for all actions (simplification)
      epsilon      : float — PPO clip threshold

    Returns:
      ratios           : np.array [N] — importance ratios
      clipped_ratios   : np.array [N] — clipped ratios
      losses           : np.array [N] — PPO loss per action
      kl               : float — KL divergence between old and new policy
      mean_loss        : float — average PPO loss across actions
    """
    # TODO: Step 1 — compute importance ratios
    # ratio[i] = policy_probs[i] / old_probs[i]
    ratios = None    # replace with policy_probs / (old_probs + 1e-8)

    # TODO: Step 2 — compute clipped ratios for each action
    # Use your clip_ratio function (it works element-wise on arrays)
    clipped_ratios = None    # replace with clip_ratio(ratios, epsilon)

    # TODO: Step 3 — compute PPO loss for each action
    # Use your ppo_loss function element-wise
    losses = None    # replace with array: -np.minimum(ratios * advantage, clipped_ratios * advantage)

    # TODO: Step 4 — compute KL divergence
    kl = None    # replace with kl_divergence(old_probs, policy_probs)

    # TODO: Step 5 — compute mean loss across all actions
    mean_loss = None    # replace with np.mean(losses)

    return ratios, clipped_ratios, losses, kl, mean_loss

# --- Run full PPO computation ---
ratios, clipped, losses, kl, mean_loss = full_ppo_computation(
    POLICY_PROBS, OLD_PROBS, ADVANTAGE, EPSILON
)

if ratios is not None:
    print("  Per-action breakdown:")
    print(f"  {'Action':<8} {'old_prob':<10} {'new_prob':<10} {'ratio':<8} {'clipped':<10} {'ppo_loss'}")
    print(f"  {'-'*65}")
    for a in range(len(OLD_PROBS)):
        print(
            f"  {a:<8} "
            f"{OLD_PROBS[a]:<10.3f} "
            f"{POLICY_PROBS[a]:<10.3f} "
            f"{ratios[a]:<8.3f} "
            f"{clipped[a]:<10.3f} "
            f"{losses[a]:.4f}"
        )
    print(f"\n  Mean PPO loss:   {mean_loss:.4f}")
    print(f"  KL divergence:   {kl:.4f}")
    print(f"  (KL = 0 means old and new policies are identical)")
    print(f"  (Mean loss < 0 means the policy is improving on this action)")
else:
    print("  (implement full_ppo_computation to see results)")

# =============================================================================
# SOLUTIONS (read ONLY after attempting!)
# =============================================================================
"""
SOLUTION 1 — Clip Ratio:

def clip_ratio(ratio, epsilon=0.2):
    clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon)
    return clipped


SOLUTION 2 — PPO Loss:

def ppo_loss(ratio, advantage, epsilon=0.2):
    unclipped = ratio * advantage
    clipped_r = clip_ratio(ratio, epsilon)
    clipped   = clipped_r * advantage
    loss = -np.minimum(unclipped, clipped)
    return loss


SOLUTION 3 — KL Divergence:

def kl_divergence(p, q):
    kl = np.sum(p * np.log(p / (q + 1e-8)))
    return float(kl)


SOLUTION 4 — Full PPO Computation:

def full_ppo_computation(policy_probs, old_probs, advantage, epsilon=0.2):
    ratios = policy_probs / (old_probs + 1e-8)
    clipped_ratios = clip_ratio(ratios, epsilon)
    losses = -np.minimum(ratios * advantage, clipped_ratios * advantage)
    kl = kl_divergence(old_probs, policy_probs)
    mean_loss = np.mean(losses)
    return ratios, clipped_ratios, losses, kl, mean_loss

# For the given data:
#   Action 0: ratio = 0.7/0.6 = 1.167  (within [0.8, 1.2] -> not clipped)
#             ppo_loss = -min(1.167*0.5, 1.167*0.5) = -0.583
#   Action 1: ratio = 0.2/0.3 = 0.667  (below 0.8 -> clipped to 0.8)
#             ppo_loss = -min(0.667*0.5, 0.8*0.5) = -min(0.333, 0.4) = -0.333
#   Action 2: ratio = 0.1/0.1 = 1.0    (exactly 1.0, no change)
#             ppo_loss = -min(1.0*0.5, 1.0*0.5) = -0.5
"""

print("\n" + "=" * 70)
print("EXERCISE 03 COMPLETE — Check your output, then read the solutions!")
print("=" * 70)

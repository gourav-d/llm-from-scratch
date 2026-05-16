"""
=============================================================================
MODULE 13 - EXERCISE 04: DPO Loss Computation
=============================================================================

WHAT YOU WILL LEARN:
  - Why DPO (Direct Preference Optimization) skips the reward model entirely
  - How DPO loss is derived from the Bradley-Terry preference model
  - How to compute log policy ratios (chosen vs rejected relative to reference)
  - How to compute the full DPO loss step by step
  - How DPO loss behaves when the policy is improving vs getting worse

C# ANALOGY:
  Imagine you're training an AI code reviewer (like Copilot).
  - RLHF approach: first train a "code quality score model", then use it to
    train the reviewer (two-stage process, expensive).
  - DPO approach: skip the score model entirely. Just say:
    "make the policy give higher probability to the GOOD response than
     the BAD response, relative to what the reference model would give."
  - It's like directly optimizing a ranking function (IComparer<T>) without
    first training a separate quality score function (IScorer<T>).

=============================================================================

DPO vs RLHF COMPARISON
=======================

  RLHF PIPELINE:
    Preference Data -> [Train Reward Model] -> [PPO Training] -> Aligned Model
         ^              (expensive step)        (complex loop)

  DPO PIPELINE:
    Preference Data -> [DPO Training] -> Aligned Model
                        (one step!    )

  DPO LOSS FORMULA:
    loss = -log( sigmoid( beta * ( (log P_new(chosen)   - log P_ref(chosen))
                                 - (log P_new(rejected) - log P_ref(rejected)) ) ) )

    Simplified as:
    loss = -log( sigmoid( beta * (implicit_reward_diff) ) )

    Where:
    implicit_reward_diff = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)

  BETA controls how much to stay close to the reference model.
  Small beta = aggressive updates. Large beta = conservative updates.

=============================================================================
"""

# ---- GLOSSARY ---------------------------------------------------------------
GLOSSARY = {
    "DPO":
        "Direct Preference Optimization — trains LLMs directly on preference pairs, no reward model.",
    "Reference Model":
        "The original (pre-RLHF) model. DPO keeps the new policy close to this reference.",
    "Log Policy Score":
        "log P(response | prompt) — log probability the policy assigns to a response.",
    "Log Policy Ratio":
        "log P_new(resp) - log P_ref(resp) — how much the new policy changed from reference.",
    "Implicit Reward":
        "beta * (log P_new(chosen) - log P_ref(chosen)) — the reward DPO implicitly optimizes.",
    "Beta":
        "Temperature parameter in DPO. Controls how much policy can deviate from reference.",
    "Preference Margin":
        "implicit_reward(chosen) - implicit_reward(rejected). Should be positive for good training.",
}

print("=" * 70)
print("EXERCISE 04 — DPO Loss Computation")
print("=" * 70)
print("\nGLOSSARY:")
for term, defn in GLOSSARY.items():
    print(f"  {term}:\n    {defn}\n")

import numpy as np    # NumPy — only library needed

# Sigmoid helper (needed for all exercises)
def sigmoid(x):
    """Squash x to range (0, 1). sigmoid(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))    # clip prevents overflow

# =============================================================================
# SHARED DATA
# =============================================================================
# Log probability scores (NOT raw probabilities — already in log space).
# In log space: more negative = lower probability.
# Example: log_score = -1.0 means the model assigns moderate probability.
#          log_score = -3.0 means the model assigns LOW probability.

# "Policy" = the model being trained (updated each step)
POLICY_SCORE_CHOSEN   = 0.8     # new policy's log-prob-like score for chosen response
POLICY_SCORE_REJECTED = 0.3     # new policy's log-prob-like score for rejected response

# "Reference" = the original SFT model (frozen, not updated)
REF_SCORE_CHOSEN   = 0.5        # reference model's score for chosen response
REF_SCORE_REJECTED = 0.4        # reference model's score for rejected response

BETA = 0.1    # DPO temperature parameter (small = aggressive updates)

# =============================================================================
# EXERCISE 1 — Log Policy Ratio
# =============================================================================
# The log policy ratio measures HOW MUCH the new policy changed from the reference
# for a specific response.
#
# Formula: log_policy_ratio = policy_score - ref_score
# (in log space, division becomes subtraction)
#
# Positive ratio = new policy gives HIGHER probability than reference
# Negative ratio = new policy gives LOWER probability than reference
# Zero = no change from reference
#
# C# analogy: like computing (new_probability - baseline_probability)
# for a recommendation model delta tracking.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 1 — Log Policy Ratio")
print("=" * 60)
print()
print("  Formula: log_ratio = policy_score - ref_score")
print("  Positive = new policy assigns MORE probability than reference")
print("  Negative = new policy assigns LESS probability than reference")
print()

def log_policy_ratio(policy_score, ref_score):
    """
    Compute the log policy ratio: how much the new policy changed from reference.
    In log space, ratio = log(P_new) - log(P_ref).

    Parameters:
      policy_score : float — new policy's score for this response
      ref_score    : float — reference model's score for this response

    Returns:
      float — log policy ratio (positive = policy increased, negative = decreased)
    """
    # TODO: compute policy_score - ref_score
    ratio = None    # replace None with the formula
    return ratio

# --- Test log_policy_ratio ---
print("  Testing log_policy_ratio:")
ratio_chosen   = log_policy_ratio(POLICY_SCORE_CHOSEN, REF_SCORE_CHOSEN)
ratio_rejected = log_policy_ratio(POLICY_SCORE_REJECTED, REF_SCORE_REJECTED)

print(f"    For CHOSEN response:   policy={POLICY_SCORE_CHOSEN}, ref={REF_SCORE_CHOSEN}")
print(f"    Log ratio = {ratio_chosen}  (positive = policy increased probability)")
print()
print(f"    For REJECTED response: policy={POLICY_SCORE_REJECTED}, ref={REF_SCORE_REJECTED}")
print(f"    Log ratio = {ratio_rejected}  (negative = policy decreased probability)")
print()
if ratio_chosen is not None and ratio_rejected is not None:
    print(f"  DPO wants: ratio_chosen > ratio_rejected")
    print(f"  Is that satisfied? {ratio_chosen > ratio_rejected}  (should be True for a good update)")

# =============================================================================
# EXERCISE 2 — DPO Loss
# =============================================================================
# Full DPO loss formula:
#
#   preference_margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
#   loss = -log( sigmoid( beta * preference_margin ) )
#
# Intuition:
#   - (policy_chosen - ref_chosen) = how much more the new policy likes the chosen response
#   - (policy_rejected - ref_rejected) = how much more it likes the rejected response
#   - We want the DIFFERENCE between these to be large and positive
#   - beta scales how strongly we enforce this
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 2 — DPO Loss")
print("=" * 60)
print()
print("  Formula:")
print("    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)")
print("    loss   = -log( sigmoid( beta * margin ) )")
print()
print(f"  Given: policy_chosen={POLICY_SCORE_CHOSEN}, ref_chosen={REF_SCORE_CHOSEN}")
print(f"         policy_rejected={POLICY_SCORE_REJECTED}, ref_rejected={REF_SCORE_REJECTED}")
print(f"         beta={BETA}")
print()

def dpo_loss(policy_score_chosen, policy_score_rejected,
             ref_score_chosen, ref_score_rejected, beta=0.1):
    """
    Compute the DPO loss for a single preference pair.

    Parameters:
      policy_score_chosen   : float — new policy's score for the chosen response
      policy_score_rejected : float — new policy's score for the rejected response
      ref_score_chosen      : float — reference model's score for chosen response
      ref_score_rejected    : float — reference model's score for rejected response
      beta                  : float — temperature parameter (default 0.1)

    Returns:
      loss : float — DPO loss for this pair (lower = better alignment)
    """
    # TODO: Step 1 — compute log ratio for chosen: policy_chosen - ref_chosen
    ratio_chosen = None

    # TODO: Step 2 — compute log ratio for rejected: policy_rejected - ref_rejected
    ratio_rejected = None

    # TODO: Step 3 — compute preference margin: ratio_chosen - ratio_rejected
    margin = None

    # TODO: Step 4 — scale by beta
    scaled_margin = None    # beta * margin

    # TODO: Step 5 — apply sigmoid and take negative log
    prob = None     # sigmoid(scaled_margin)
    loss = None     # -log(prob + 1e-8)

    return loss

# --- Test DPO loss ---
loss_val = dpo_loss(
    POLICY_SCORE_CHOSEN, POLICY_SCORE_REJECTED,
    REF_SCORE_CHOSEN, REF_SCORE_REJECTED,
    BETA
)
print(f"  DPO loss = {loss_val}")
print(f"  (lower = model is more aligned with human preferences)")

# =============================================================================
# EXERCISE 3 — Step-by-Step DPO Loss Trace
# =============================================================================
# Work through the DPO loss calculation manually to understand each step.
# Show intermediate values so the student can trace exactly what happens.
#
# Use the same data as Exercise 2.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 3 — Step-by-Step DPO Loss Trace")
print("=" * 60)
print()
print("  Task: compute the DPO loss step by step and print each intermediate value.")
print()
print("  Walkthrough:")
print(f"    Step 1: ratio_chosen   = {POLICY_SCORE_CHOSEN} - {REF_SCORE_CHOSEN} = ?")
print(f"    Step 2: ratio_rejected = {POLICY_SCORE_REJECTED} - {REF_SCORE_REJECTED} = ?")
print(f"    Step 3: margin         = ratio_chosen - ratio_rejected = ?")
print(f"    Step 4: scaled_margin  = {BETA} * margin = ?")
print(f"    Step 5: prob           = sigmoid(scaled_margin) = ?")
print(f"    Step 6: loss           = -log(prob) = ?")
print()

def dpo_loss_verbose(policy_score_chosen, policy_score_rejected,
                     ref_score_chosen, ref_score_rejected, beta=0.1):
    """
    Compute DPO loss with full step-by-step printout.
    Same logic as dpo_loss() but prints every intermediate value.
    """
    # TODO: compute and print each step
    # Step 1
    ratio_chosen = None    # policy_score_chosen - ref_score_chosen
    print(f"    Step 1 — ratio_chosen   = {policy_score_chosen} - {ref_score_chosen} = {ratio_chosen}")

    # Step 2
    ratio_rejected = None    # policy_score_rejected - ref_score_rejected
    print(f"    Step 2 — ratio_rejected = {policy_score_rejected} - {ref_score_rejected} = {ratio_rejected}")

    # Step 3
    margin = None    # ratio_chosen - ratio_rejected
    print(f"    Step 3 — margin         = {ratio_chosen} - {ratio_rejected} = {margin}")

    # Step 4
    scaled_margin = None    # beta * margin
    print(f"    Step 4 — scaled_margin  = {beta} * {margin} = {scaled_margin}")

    # Step 5
    prob = None    # sigmoid(scaled_margin)
    print(f"    Step 5 — prob (sigmoid) = sigmoid({scaled_margin}) = {prob}")

    # Step 6
    loss = None    # -np.log(prob + 1e-8)
    print(f"    Step 6 — loss           = -log({prob}) = {loss}")

    return loss

print("  Tracing DPO loss computation:")
traced_loss = dpo_loss_verbose(
    POLICY_SCORE_CHOSEN, POLICY_SCORE_REJECTED,
    REF_SCORE_CHOSEN, REF_SCORE_REJECTED,
    BETA
)

# =============================================================================
# EXERCISE 4 — Show That DPO Loss Decreases When Policy Improves
# =============================================================================
# Demonstrate that the DPO loss gets smaller (better) when:
#   (a) the policy increases its score for the chosen response
#   (b) the policy decreases its score for the rejected response
#
# We run 4 scenarios and compare the losses.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 4 — DPO Loss vs Policy Quality")
print("=" * 60)
print()
print("  We vary policy scores and show how DPO loss changes.")
print("  Lower loss = more aligned policy.")
print()

# Reference scores stay fixed (the reference model is frozen)
ref_chosen   = 0.5    # reference score for chosen
ref_rejected = 0.4    # reference score for rejected

# 4 scenarios: policy scores for (chosen, rejected)
scenarios = [
    # (policy_chosen, policy_rejected, description)
    (0.5, 0.4, "Policy = Reference (no change from ref)"),
    (0.8, 0.4, "Policy likes chosen MORE (good direction!)"),
    (0.8, 0.1, "Policy likes chosen MORE AND rejected LESS (even better!)"),
    (0.2, 0.9, "Policy likes chosen LESS and rejected MORE (bad direction!)"),
]

print("  Scenario comparison:")
print(f"  {'Scenario':<5} {'p_chosen':<10} {'p_rejected':<12} {'DPO Loss':<12} {'Interpretation'}")
print(f"  {'-'*80}")

for i, (p_c, p_r, description) in enumerate(scenarios):
    # TODO: compute DPO loss for this scenario using your dpo_loss() function
    loss = None    # replace with dpo_loss(p_c, p_r, ref_chosen, ref_rejected, BETA)
    print(f"  {i+1:<5} {p_c:<10.2f} {p_r:<12.2f} {str(loss):<12} {description}")

print()
print("  Expected pattern: lowest loss when policy_chosen is HIGH and policy_rejected is LOW")
print("  Scenario 3 should have the lowest loss (most aligned policy).")
print("  Scenario 4 should have the highest loss (least aligned policy).")

# Bonus: show the trend with a simple ASCII chart if dpo_loss is implemented
print()
print("  DPO loss trend (ASCII bar chart, lower = better):")
for i, (p_c, p_r, description) in enumerate(scenarios):
    loss = dpo_loss(p_c, p_r, ref_chosen, ref_rejected, BETA)
    if loss is not None:
        bar_len = min(int(loss * 30), 50)    # scale to display width
        bar = "#" * bar_len                  # filled bar proportional to loss
        print(f"  Scenario {i+1}: {loss:.4f}  |{bar}")
    else:
        print(f"  Scenario {i+1}: (implement dpo_loss first)")

# =============================================================================
# SOLUTIONS (read ONLY after attempting!)
# =============================================================================
"""
SOLUTION 1 — Log Policy Ratio:

def log_policy_ratio(policy_score, ref_score):
    ratio = policy_score - ref_score
    return ratio

# Results:
# ratio_chosen   = 0.8 - 0.5 = 0.3   (policy increased chosen's probability)
# ratio_rejected = 0.3 - 0.4 = -0.1  (policy decreased rejected's probability)
# ratio_chosen > ratio_rejected  =>  True  (good! DPO is satisfied)


SOLUTION 2 — DPO Loss:

def dpo_loss(policy_score_chosen, policy_score_rejected,
             ref_score_chosen, ref_score_rejected, beta=0.1):
    ratio_chosen   = policy_score_chosen   - ref_score_chosen
    ratio_rejected = policy_score_rejected - ref_score_rejected
    margin         = ratio_chosen - ratio_rejected
    scaled_margin  = beta * margin
    prob           = sigmoid(scaled_margin)
    loss           = -np.log(prob + 1e-8)
    return loss

# With given data:
# ratio_chosen   = 0.8 - 0.5 =  0.3
# ratio_rejected = 0.3 - 0.4 = -0.1
# margin         = 0.3 - (-0.1) = 0.4
# scaled_margin  = 0.1 * 0.4 = 0.04
# prob           = sigmoid(0.04) ≈ 0.510
# loss           = -log(0.510) ≈ 0.673


SOLUTION 3 — Verbose Trace (same formulas as Solution 2, just with prints):

def dpo_loss_verbose(policy_score_chosen, policy_score_rejected,
                     ref_score_chosen, ref_score_rejected, beta=0.1):
    ratio_chosen   = policy_score_chosen   - ref_score_chosen
    ratio_rejected = policy_score_rejected - ref_score_rejected
    margin         = ratio_chosen - ratio_rejected
    scaled_margin  = beta * margin
    prob           = sigmoid(scaled_margin)
    loss           = -np.log(prob + 1e-8)
    print(f"    Step 1 — ratio_chosen   = {ratio_chosen}")
    print(f"    Step 2 — ratio_rejected = {ratio_rejected}")
    print(f"    Step 3 — margin         = {margin}")
    print(f"    Step 4 — scaled_margin  = {scaled_margin}")
    print(f"    Step 5 — prob (sigmoid) = {prob:.4f}")
    print(f"    Step 6 — loss           = {loss:.4f}")
    return loss


SOLUTION 4 — DPO Loss Across Scenarios:

for i, (p_c, p_r, description) in enumerate(scenarios):
    loss = dpo_loss(p_c, p_r, ref_chosen, ref_rejected, BETA)
    print(f"  {i+1:<5} {p_c:<10.2f} {p_r:<12.2f} {loss:<12.4f} {description}")

# Expected DPO losses (approximately):
# Scenario 1: policy = ref       -> margin ≈  0.00 -> prob ≈ 0.500 -> loss ≈ 0.693
# Scenario 2: good chosen        -> margin ≈  0.40 -> prob ≈ 0.510 -> loss ≈ 0.673
# Scenario 3: good chosen + good rejected -> margin ≈ 0.70 -> loss ≈ 0.650 (lowest)
# Scenario 4: bad direction      -> margin ≈ -0.90 -> prob ≈ 0.478 -> loss ≈ 0.738 (highest)
"""

print("\n" + "=" * 70)
print("EXERCISE 04 COMPLETE — Check your output, then read the solutions!")
print("=" * 70)

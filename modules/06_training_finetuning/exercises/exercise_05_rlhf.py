"""
Module 06 - Training & Fine-Tuning
Exercise 05: RLHF and Alignment Basics

GLOSSARY
--------
RLHF              : Reinforcement Learning from Human Feedback.
                    Three-phase process to make LLMs helpful and safe.
                    How ChatGPT was built from GPT-3.
Phase 1 — SFT     : Supervised Fine-Tuning.
                    Show the model example (prompt, ideal_response) pairs.
                    Model learns to imitate expert responses.
Phase 2 — Reward  : Train a Reward Model on human preference data.
  Model             Humans compare two responses (A vs B) and pick the better one.
                    The reward model learns to predict which response humans prefer.
Phase 3 — PPO     : Proximal Policy Optimization.
                    Use the reward model as a "judge" to reinforce good responses
                    and penalise bad ones.
Reward Score      : A number (higher = better) assigned to a response.
                    Reward model predicts this from (prompt, response) pairs.
Preference Data   : Pairs of responses (A, B) with a human label: which is better.
                    Format: {"prompt": ..., "chosen": ..., "rejected": ...}
KL Divergence     : Measures how different two probability distributions are.
                    Used in PPO to prevent the model from drifting too far from SFT.
                    KL = 0 means identical distributions.
Bradley-Terry     : Statistical model for pairwise preferences.
                    Pr(A > B) = sigma(reward_A - reward_B)
                    Used to train the reward model.
"""

import numpy as np   # NumPy for math

print("=" * 60)
print("Exercise 05: RLHF and Alignment Basics")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Reward Scoring — Which Response Is Better?
#
#  Background:
#    The reward model produces a scalar score for each response.
#    Higher score = better (more helpful, safer, more accurate).
#
#    Given two responses A and B:
#      If reward_A > reward_B -> A is preferred
#      If reward_B > reward_A -> B is preferred
#
#    We also compute "preference strength":
#      strength = abs(reward_A - reward_B)
#      Higher strength = human annotators would be more certain about preference.
#
#  Your Task:
#    Write: compare_responses(reward_a, reward_b) -> dict
#    Returns: {"preferred": "A" or "B", "margin": float, "confident": bool}
#    confident = True if margin > 0.5 (strong preference)
#
#  C# Analogy:
#    Like comparing two test scores and deciding which student did better.
# ============================================================

print("-" * 50)
print("EXERCISE 1: Reward Score Comparison")
print("-" * 50)
print()


def compare_responses(reward_a, reward_b):
    """
    Determine which response is preferred based on reward scores.

    Parameters:
        reward_a (float): Reward score for response A.
        reward_b (float): Reward score for response B.

    Returns:
        dict: {
            "preferred" : str   -- "A" if reward_a > reward_b, else "B",
            "margin"    : float -- abs(reward_a - reward_b),
            "confident" : bool  -- True if margin > 0.5
        }
    """
    # TODO:
    # preferred = "A" if reward_a > reward_b else "B"
    # margin    = abs(reward_a - reward_b)
    # confident = margin > 0.5
    pass  # Replace with your implementation


pairs = [
    ("Helpful answer",    "Rude answer",       2.8,  0.3),
    ("Detailed answer",   "Vague answer",      1.9,  1.4),
    ("Safe response",     "Harmful response",  3.5, -1.2),
    ("Almost equal A",    "Almost equal B",    1.5,  1.4),
]

print(f"  {'A Label':<20} {'B Label':<20} {'Reward A':>9} {'Reward B':>9} {'Preferred':>10} {'Margin':>8} {'Confident':>10}")
print("  " + "-" * 92)
for label_a, label_b, ra, rb in pairs:
    r = compare_responses(ra, rb)
    if r:
        print(f"  {label_a:<20} {label_b:<20} {ra:>9.1f} {rb:>9.1f} "
              f"{r['preferred']:>10} {r['margin']:>8.1f} {str(r['confident']):>10}")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Bradley-Terry Preference Probability
#
#  Background:
#    The Bradley-Terry model gives the PROBABILITY that response A
#    is preferred over response B:
#
#      Pr(A > B) = sigmoid(reward_A - reward_B)
#
#    where sigmoid(x) = 1 / (1 + exp(-x))
#
#    Intuition:
#      reward_A - reward_B = 0  -> Pr = 0.5 (equal, 50/50)
#      reward_A - reward_B = 2  -> Pr ~ 0.88 (A strongly preferred)
#      reward_A - reward_B = -2 -> Pr ~ 0.12 (B strongly preferred)
#
#  Your Task:
#    Write: preference_probability(reward_a, reward_b) -> float
#    Returns Pr(A preferred over B) using sigmoid of the reward difference.
#
#  C# Analogy:
#    double prob = 1.0 / (1.0 + Math.Exp(-(rewardA - rewardB)));
# ============================================================

print("-" * 50)
print("EXERCISE 2: Bradley-Terry Preference Probability")
print("-" * 50)
print()


def preference_probability(reward_a, reward_b):
    """
    Compute the probability that response A is preferred over B.

    Uses the Bradley-Terry model: Pr(A > B) = sigmoid(reward_A - reward_B)

    Parameters:
        reward_a (float): Reward score for response A.
        reward_b (float): Reward score for response B.

    Returns:
        float: Probability in [0, 1] that A is preferred over B.
    """
    # TODO:
    # diff = reward_a - reward_b
    # return 1.0 / (1.0 + np.exp(-diff))    # sigmoid
    pass  # Replace with your implementation


print("  Pr(A preferred) for different reward gaps (reward_A - reward_B):")
print()
print(f"  {'Reward A':>9} {'Reward B':>9} {'Gap':>6}  {'Pr(A>B)':>10}")
print("  " + "-" * 40)
for ra, rb in [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (-1.0, 0.0)]:
    p = preference_probability(ra, rb)
    if p is not None:
        gap = ra - rb
        print(f"  {ra:>9.1f} {rb:>9.1f} {gap:>6.1f}  {p:>10.4f}")
print()
print("  Expected: gap=0 -> 0.5, gap=2 -> ~0.88, gap=-1 -> ~0.27")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Build a Preference Dataset
#
#  Background:
#    Phase 2 of RLHF requires (prompt, chosen, rejected) triplets.
#    "chosen"   = the better response (higher reward).
#    "rejected" = the worse response  (lower reward).
#
#    Given a list of (prompt, response_a, response_b, reward_a, reward_b),
#    build a list of dicts with keys: "prompt", "chosen", "rejected".
#    The chosen is whichever response has the higher reward.
#
#  Your Task:
#    Write: build_preference_dataset(raw_comparisons) -> list of dict
#    Each input item: (prompt, response_a, response_b, reward_a, reward_b)
#    Output: [{"prompt": str, "chosen": str, "rejected": str}, ...]
#
#  C# Analogy:
#    Like a LINQ .Select() that reorganises comparison records into
#    training format for a classifier.
# ============================================================

print("-" * 50)
print("EXERCISE 3: Build Preference Dataset")
print("-" * 50)
print()


def build_preference_dataset(raw_comparisons):
    """
    Convert raw human comparison data into RLHF preference format.

    Parameters:
        raw_comparisons (list of tuple):
            Each tuple: (prompt, response_a, response_b, reward_a, reward_b)

    Returns:
        list of dict: [{"prompt": str, "chosen": str, "rejected": str}, ...]
        chosen  = response with higher reward
        rejected = response with lower reward
    """
    # TODO:
    # For each (prompt, resp_a, resp_b, r_a, r_b):
    #   if r_a >= r_b: chosen = resp_a, rejected = resp_b
    #   else:          chosen = resp_b, rejected = resp_a
    # Return list of {"prompt": ..., "chosen": ..., "rejected": ...}
    pass  # Replace with your implementation


raw = [
    ("What is Python?",
     "Python is a popular programming language.",   # A
     "I don't know.",                               # B
     2.5, 0.1),
    ("Explain recursion.",
     "It calls itself.",                            # A (short, vague)
     "Recursion means a function calls itself "
     "with a smaller input until a base case.",     # B (better)
     0.8, 2.2),
    ("How do neural networks learn?",
     "Through gradient descent and backprop.",      # A
     "Magic.",                                      # B (bad)
     1.9, -0.5),
]

dataset = build_preference_dataset(raw)

if dataset:
    print(f"  Built {len(dataset)} preference pairs:\n")
    for i, item in enumerate(dataset):
        print(f"  [{i+1}] Prompt  : {item['prompt']}")
        print(f"       Chosen  : {item['chosen'][:60]}...")
        print(f"       Rejected: {item['rejected'][:60]}...")
        print()


# ============================================================
#  EXERCISE 4
#  Topic: KL Divergence Penalty
#
#  Background:
#    During PPO (phase 3 of RLHF), we add a KL divergence penalty
#    to prevent the model from straying too far from its SFT version.
#
#    KL(p || q) = sum(p_i * log(p_i / q_i))
#
#    Where:
#      p = current policy (PPO model) probability distribution
#      q = reference policy (SFT model) probability distribution
#
#    KL = 0 means the distributions are identical (no drift).
#    KL > 0 means the model has changed. Very high KL is penalised.
#
#    Total PPO objective:
#      objective = reward - kl_coeff * KL(current || reference)
#
#  Your Task:
#    Write: kl_divergence(p, q) -> float
#    Returns KL(p || q) for two probability distributions.
#
#    Then write: ppo_objective(reward, kl, kl_coeff=0.02) -> float
#    Returns reward - kl_coeff * kl
#
#  C# Analogy:
#    KL is like measuring "version drift" between two configs.
#    The penalty keeps the model close to the known-good (SFT) version.
# ============================================================

print("-" * 50)
print("EXERCISE 4: KL Divergence Penalty")
print("-" * 50)
print()


def kl_divergence(p, q):
    """
    Compute KL divergence KL(p || q).

    Parameters:
        p (np.ndarray): Current policy distribution (sums to 1).
        q (np.ndarray): Reference policy distribution (sums to 1).

    Returns:
        float: KL divergence. 0 if p == q. Higher = more different.
    """
    # TODO:
    # Add epsilon to avoid log(0).
    # epsilon = 1e-10
    # return np.sum(p * np.log((p + epsilon) / (q + epsilon)))
    pass  # Replace with your implementation


def ppo_objective(reward, kl, kl_coeff=0.02):
    """
    Compute PPO objective: maximize reward while staying close to SFT model.

    Parameters:
        reward   (float): Reward score from the reward model.
        kl       (float): KL divergence from SFT reference.
        kl_coeff (float): Penalty coefficient (default 0.02).

    Returns:
        float: PPO objective = reward - kl_coeff * kl
    """
    # TODO: return reward - kl_coeff * kl
    pass  # Replace with your implementation


# Reference (SFT) distribution over 5 tokens
ref_dist = np.array([0.40, 0.30, 0.15, 0.10, 0.05])

# Current (PPO) distributions — varying levels of drift
dist_none  = ref_dist.copy()                                          # no drift
dist_small = np.array([0.38, 0.31, 0.16, 0.10, 0.05])               # small drift
dist_large = np.array([0.10, 0.10, 0.10, 0.35, 0.35])               # large drift

kl_none  = kl_divergence(dist_none,  ref_dist)
kl_small = kl_divergence(dist_small, ref_dist)
kl_large = kl_divergence(dist_large, ref_dist)

if kl_none is not None:
    print(f"  KL (no drift) : {kl_none:.6f}   (expected ~0.0)")
    print(f"  KL (small)    : {kl_small:.6f}")
    print(f"  KL (large)    : {kl_large:.6f}   (much larger)")
    print()

    reward = 2.5   # same reward for all scenarios
    print("  PPO Objective (reward=2.5, kl_coeff=0.02):")
    obj_none  = ppo_objective(reward, kl_none)
    obj_small = ppo_objective(reward, kl_small)
    obj_large = ppo_objective(reward, kl_large)
    if obj_none is not None:
        print(f"    No drift  : {obj_none:.4f}")
        print(f"    Small drift: {obj_small:.4f}")
        print(f"    Large drift: {obj_large:.4f}  <- penalised")
print()
print("  Expected: high KL reduces PPO objective even with same reward.")
print()

print("=" * 60)
print("All exercises complete!")
print("=" * 60)

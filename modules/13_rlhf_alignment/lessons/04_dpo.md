# Lesson 04: Direct Preference Optimization (DPO)

## Glossary (Read This First!)

| Term | Plain English Definition |
|------|--------------------------|
| **DPO** | Direct Preference Optimization. A method to align LLMs with human preferences that skips the reward model and PPO entirely. Directly optimizes the language model on preference data. |
| **Preference Data** | Triples of (prompt, chosen_response, rejected_response) where "chosen" is the response humans preferred. |
| **Reference Model** | The frozen SFT model used as a baseline in DPO. The trained model's probabilities are measured RELATIVE to the reference model's probabilities. |
| **Log-Ratio** | The logarithm of the ratio of two probabilities. log(pi_theta(y|x) / pi_ref(y|x)) tells you how much more (or less) likely the trained model assigns to a response vs. the reference model. |
| **Implicit Reward** | In DPO, there is no explicit reward model. Instead, the reward is "implied" by the ratio between the trained policy and the reference policy. |
| **Offline Training** | Training on a fixed dataset that was collected beforehand. Not interacting with the environment during training. DPO is offline -- you use pre-collected preference pairs. |
| **On-Policy** | Generating new data using the CURRENT model during training. PPO is on-policy -- it generates rollouts with the current policy at each step. |
| **Off-Policy** | Using data collected from an OLDER policy (or a different model) for training. DPO is off-policy -- it uses a fixed preference dataset. |
| **Beta** | A temperature parameter in DPO that controls how strongly the model is pushed toward preferences vs. staying close to the reference model. |
| **KL Constraint** | In DPO, the beta parameter implicitly enforces a KL constraint -- preventing the trained model from drifting too far from the reference model. |

---

## Part 1: The Problem with PPO

Before learning DPO, let us be honest about the problems with RLHF+PPO.

PPO is powerful, but it comes with a long list of headaches:

```
+---------------------------------------------------------------+
|  RLHF+PPO: WHAT CAN GO WRONG                                  |
|                                                               |
|  Component count:                                             |
|    - SFT model (frozen reference)                             |
|    - Reward model (separate network, separately trained)       |
|    - Policy model (the one being trained)                     |
|    - Critic/value model (another separate network!)           |
|    Total: 4 large neural networks loaded at once              |
|                                                               |
|  Memory cost:                                                 |
|    If base LLM is 7B params at 16GB RAM:                      |
|    4 copies = 64GB RAM just for the models                    |
|    + activations + gradients = 100GB+ required                |
|                                                               |
|  Hyperparameter sensitivity:                                  |
|    - KL coefficient (beta)                                    |
|    - PPO clipping epsilon                                      |
|    - Value loss coefficient (c1)                              |
|    - Entropy bonus coefficient (c2)                           |
|    - Learning rates for policy AND critic                     |
|    - Rollout batch size                                        |
|    - Number of PPO update steps per rollout                   |
|    Getting these wrong = training collapse                    |
|                                                               |
|  Reward hacking:                                              |
|    The reward model can be exploited by the policy            |
|    Leading to "reward hacking" that requires constant         |
|    monitoring and manual intervention                         |
|                                                               |
|  Instability:                                                 |
|    PPO training can suddenly collapse after many stable steps  |
|    Difficult to predict or recover from                       |
+---------------------------------------------------------------+
```

### Is There a Simpler Way?

In 2023, researchers at Stanford published a paper called:
**"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"**
(Rafailov et al., 2023)

The key insight: **You don't need a separate reward model or PPO.**

The preference information can be learned directly by the language model
using a single, simple training objective.

---

## Part 2: The DPO Insight

### The Mathematical Foundation

Here is the chain of reasoning that leads to DPO.

**Step 1:** In RLHF+PPO, what are we really optimizing?

```
We want:   maximize E[r(x, y)] - beta * KL(pi_theta || pi_ref)

Where:
  r(x, y)   = reward for response y to prompt x
  pi_theta  = current policy (being trained)
  pi_ref    = reference (SFT) policy
  beta      = KL coefficient
```

**Step 2:** What is the OPTIMAL solution to this objective?

Mathematically, if we could find the perfect solution, it would be:

```
pi_optimal(y | x) = pi_ref(y | x) * exp(r(x, y) / beta) / Z(x)

Where Z(x) is a normalization constant (partition function).
```

This says: the optimal policy is the reference policy, modified by
the exponential of the reward.

**Step 3:** Rearrange to get the implied reward.

From the optimal policy equation, we can solve for the REWARD:

```
r(x, y) = beta * log(pi_optimal(y|x) / pi_ref(y|x)) + beta * log(Z(x))

Simplified: r(x, y) ~ beta * log(pi(y|x) / pi_ref(y|x))
(the log Z term cancels in the final loss)
```

This is the "implicit reward" -- the reward implied by the policy's log-ratio
with the reference model!

**Step 4:** Plug this into the Bradley-Terry loss (from Lesson 02).

Recall the Bradley-Terry loss used for reward models:

```
L_RM = -log(sigma(r_chosen - r_rejected))
```

Substituting our implicit reward:

```
r(x, y) = beta * log(pi(y|x) / pi_ref(y|x))

r_chosen  = beta * log(pi(y_w|x) / pi_ref(y_w|x))
r_rejected = beta * log(pi(y_l|x) / pi_ref(y_l|x))

L_DPO = -log(sigma(r_chosen - r_rejected))
      = -log(sigma(
            beta * log(pi(y_w|x) / pi_ref(y_w|x))
          - beta * log(pi(y_l|x) / pi_ref(y_l|x))
        ))
```

This is the **DPO loss**. It has NO reward model. It uses the language model
itself to compute everything.

```
+---------------------------------------------------------------+
|  DPO LOSS FORMULA (Final Form)                                |
|                                                               |
|  L_DPO = -log(sigma(                                          |
|    beta * (log(pi_theta(y_w|x) / pi_ref(y_w|x))              |
|           -log(pi_theta(y_l|x) / pi_ref(y_l|x)))             |
|  ))                                                           |
|                                                               |
|  Where:                                                       |
|    pi_theta = model being trained                             |
|    pi_ref   = frozen reference model (SFT)                    |
|    y_w      = winning (chosen) response                       |
|    y_l      = losing (rejected) response                      |
|    x        = prompt                                          |
|    beta     = temperature (typically 0.1 to 0.5)             |
|    sigma    = sigmoid function                                |
+---------------------------------------------------------------+
```

---

## Part 3: DPO Loss -- Detailed Walkthrough

Let us break down the DPO loss piece by piece.

### The Log-Ratio Term

```
log(pi_theta(y_w|x) / pi_ref(y_w|x))

= log(pi_theta(y_w|x)) - log(pi_ref(y_w|x))
```

This measures: "How much more (or less) likely is the trained model
to generate the CHOSEN response, compared to the reference?"

If the value is POSITIVE: the trained model prefers the chosen response MORE than reference
If the value is NEGATIVE: the trained model prefers the chosen response LESS than reference (bad)

### What the Loss Encourages

The DPO loss pushes the model in two directions simultaneously:

```
+---------------------------------------------------------------+
|  WHAT DPO OPTIMIZES                                           |
|                                                               |
|  The loss is minimized when:                                  |
|                                                               |
|  log(pi_theta(y_w|x) / pi_ref(y_w|x))    IS LARGE            |
|  "Chosen response becomes more likely relative to reference"  |
|                                                               |
|  AND                                                          |
|                                                               |
|  log(pi_theta(y_l|x) / pi_ref(y_l|x))    IS SMALL (negative) |
|  "Rejected response becomes less likely relative to reference"|
|                                                               |
|  COMBINED:                                                    |
|  DPO increases the GAP between chosen and rejected            |
|  in terms of their log-probabilities relative to reference.   |
+---------------------------------------------------------------+
```

C# Analogy:
```csharp
// Imagine you have a recommendation system.
// The "reference model" is the current production system.
// The "trained model" is your new version.

// DPO loss says:
// "Make the new model MORE likely to recommend what users preferred (y_w)"
// "Make the new model LESS likely to recommend what users disliked (y_l)"
// "But do this RELATIVE to what the current system recommends"

// This is like having pre-computed test results (A/B test results)
// and directly tuning toward them, without needing a separate
// scoring service (the reward model).

// vs. PPO which would be:
// "Build a score predictor (reward model) from the A/B results"
// "Then use that predictor in a loop to improve the system"
// DPO skips the score predictor step entirely.
```

---

## Part 4: DPO Loss Implementation

Here is the DPO loss implemented from scratch:

```python
# dpo_loss.py
# Direct Preference Optimization loss function
# Implemented in NumPy to show the math clearly

import numpy as np  # NumPy for array operations

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sigmoid(x):
    """
    Sigmoid function: converts any real number to (0, 1).
    sigma(x) = 1 / (1 + e^(-x))
    
    Numerically stable version: handles very large positive and negative x.
    """
    # For large positive x: avoid exp overflow
    # For large negative x: avoid exp(positive large) overflow
    # Use: sigma(x) = exp(x) / (1 + exp(x)) when x >= 0
    #      sigma(x) = 1 / (1 + exp(-x))    when x < 0
    return np.where(
        x >= 0,                          # if x >= 0:
        1.0 / (1.0 + np.exp(-x)),        #   use standard formula
        np.exp(x) / (1.0 + np.exp(x))   # else: equivalent but stable
    )

def log_sigmoid(x):
    """
    log(sigmoid(x)) -- more numerically stable than log(sigmoid(x)).
    
    log(sigma(x)) = -log(1 + exp(-x))  for x >= 0
                  = x - log(1 + exp(x)) for x < 0
    
    Using np.logaddexp which computes log(exp(a) + exp(b)) stably.
    """
    # np.logaddexp(a, b) = log(exp(a) + exp(b))
    # log(1 + exp(-x)) = logaddexp(0, -x)
    return -np.logaddexp(0, -x)


# ============================================================
# THE DPO LOSS FUNCTION
# ============================================================

def dpo_loss(
    log_probs_policy_chosen,   # log P(y_w | x) under trained model
    log_probs_policy_rejected, # log P(y_l | x) under trained model
    log_probs_ref_chosen,      # log P(y_w | x) under reference model
    log_probs_ref_rejected,    # log P(y_l | x) under reference model
    beta=0.1                   # temperature parameter
):
    """
    Compute the DPO (Direct Preference Optimization) loss.
    
    Formula:
      L_DPO = -log(sigma(beta * (log_ratio_chosen - log_ratio_rejected)))
    
    Where:
      log_ratio = log(pi_theta(y|x)) - log(pi_ref(y|x))
                = log(pi_theta(y|x) / pi_ref(y|x))
    
    Args:
        log_probs_policy_chosen:   float or array, log prob of chosen under policy
        log_probs_policy_rejected: float or array, log prob of rejected under policy
        log_probs_ref_chosen:      float or array, log prob of chosen under reference
        log_probs_ref_rejected:    float or array, log prob of rejected under reference
        beta:                      float, temperature (default 0.1)
    
    Returns:
        loss: float or array, the DPO loss (lower = better)
        reward_chosen:   implicit reward for chosen response
        reward_rejected: implicit reward for rejected response
    """
    
    # Step 1: Compute log-ratios (implicit rewards)
    # "How much does trained model prefer this response vs. reference model?"
    
    # log(pi_theta(y_w|x)) - log(pi_ref(y_w|x))
    # = log(pi_theta(y_w|x) / pi_ref(y_w|x))
    log_ratio_chosen = log_probs_policy_chosen - log_probs_ref_chosen
    
    # log(pi_theta(y_l|x)) - log(pi_ref(y_l|x))
    log_ratio_rejected = log_probs_policy_rejected - log_probs_ref_rejected
    
    # Step 2: Scale by beta
    # beta controls how strongly we enforce preferences
    reward_chosen   = beta * log_ratio_chosen    # implicit reward for chosen
    reward_rejected = beta * log_ratio_rejected  # implicit reward for rejected
    
    # Step 3: Compute the preference margin
    # We want reward_chosen to be HIGHER than reward_rejected
    margin = reward_chosen - reward_rejected
    
    # Step 4: Apply the Bradley-Terry loss
    # L = -log(sigmoid(margin))
    # When margin is large positive: sigmoid(large) = 1, log(1) = 0 (good!)
    # When margin is small or negative: large loss (bad!)
    loss = -log_sigmoid(margin)
    
    return loss, reward_chosen, reward_rejected


# ============================================================
# DEMO 1: Single pair
# ============================================================

print("=== DPO Loss: Single Example ===\n")

# Log probabilities from the REFERENCE (frozen SFT) model
# These are the baseline probabilities
log_prob_ref_chosen   = -5.2   # reference assigns -5.2 nats to chosen
log_prob_ref_rejected = -4.8   # reference assigns -4.8 nats to rejected

# Log probabilities from the POLICY (model being trained)
# Initially same as reference (before any DPO training)
print("--- Before DPO Training ---")
log_prob_policy_chosen   = -5.2  # same as reference
log_prob_policy_rejected = -4.8  # same as reference

loss, r_w, r_l = dpo_loss(
    log_prob_policy_chosen,
    log_prob_policy_rejected,
    log_prob_ref_chosen,
    log_prob_ref_rejected,
    beta=0.1
)
print(f"  Implicit reward (chosen):   {r_w:.4f}")
print(f"  Implicit reward (rejected): {r_l:.4f}")
print(f"  Margin (chosen - rejected): {r_w - r_l:.4f}")
print(f"  Loss: {loss:.4f}")
print()

# After DPO training: policy now assigns higher prob to chosen
print("--- After DPO Training (ideal case) ---")
log_prob_policy_chosen   = -4.0  # trained model prefers chosen MORE
log_prob_policy_rejected = -6.5  # trained model prefers rejected LESS

loss, r_w, r_l = dpo_loss(
    log_prob_policy_chosen,
    log_prob_policy_rejected,
    log_prob_ref_chosen,
    log_prob_ref_rejected,
    beta=0.1
)
print(f"  Implicit reward (chosen):   {r_w:.4f}")
print(f"  Implicit reward (rejected): {r_l:.4f}")
print(f"  Margin (chosen - rejected): {r_w - r_l:.4f}")
print(f"  Loss: {loss:.4f}")
print()


# ============================================================
# DEMO 2: Batch of preference pairs
# ============================================================

print("=== DPO Loss: Batch of Examples ===\n")

# Simulate a batch of 4 preference pairs
np.random.seed(42)  # for reproducibility

batch_size = 4

# Reference model probabilities (fixed, won't change)
ref_chosen   = np.array([-4.0, -6.5, -3.2, -8.1])  # log probs
ref_rejected = np.array([-5.5, -4.2, -5.8, -6.0])  # log probs

# Policy probabilities (change during training)
# Simulate: policy has been partially trained
policy_chosen   = np.array([-3.5, -5.8, -2.9, -7.5])  # improved
policy_rejected = np.array([-6.0, -5.0, -6.5, -7.0])  # worsened (good!)

# Compute batch loss
losses, rewards_w, rewards_l = dpo_loss(
    policy_chosen,
    policy_rejected,
    ref_chosen,
    ref_rejected,
    beta=0.1
)

print(f"{'Pair':>4} | {'r_chosen':>10} | {'r_rejected':>12} | {'margin':>8} | {'loss':>8}")
print("-" * 55)
for i in range(batch_size):
    margin = rewards_w[i] - rewards_l[i]
    correct = "OK" if margin > 0 else "WRONG"
    print(f"  {i+1}  | {rewards_w[i]:>10.4f} | {rewards_l[i]:>12.4f} | "
          f"{margin:>8.4f} | {losses[i]:>8.4f}  {correct}")

# Mean loss over the batch (this is what we minimize)
mean_loss = np.mean(losses)
print(f"\nMean batch loss: {mean_loss:.4f}")
print("(Gradient descent will reduce this over time)")
```

---

## Part 5: DPO vs PPO -- Detailed Comparison

Here is a comprehensive comparison of the two approaches:

```
+===================================================================+
|  DPO vs PPO COMPARISON                                            |
+===================================================================+
|                                                                   |
|  COMPLEXITY                                                       |
|  -----------------------------------------------------------------|
|  PPO: 4 networks (SFT ref, reward model, policy, critic)          |
|       Complex RL loop with rollouts                               |
|  DPO: 2 networks (reference model + policy being trained)         |
|       Standard supervised training loop (like SFT!)               |
|                                                                   |
|  MEMORY REQUIREMENTS                                              |
|  -----------------------------------------------------------------|
|  PPO: Need all 4 networks in memory simultaneously               |
|       Can be 4x the base model size                               |
|  DPO: Need 2 networks (can use PEFT/LoRA to reduce further)      |
|       Roughly 2x the base model size                              |
|                                                                   |
|  TRAINING STABILITY                                               |
|  -----------------------------------------------------------------|
|  PPO: Can collapse suddenly, very sensitive to hyperparameters    |
|       Requires constant monitoring of KL, rewards                  |
|  DPO: Very stable, essentially no new hyperparameters             |
|       Behaves like standard fine-tuning                           |
|                                                                   |
|  DATA REQUIREMENTS                                                |
|  -----------------------------------------------------------------|
|  PPO: Needs preference data for reward model                      |
|       Generates new rollouts during training (online)             |
|  DPO: Only needs the preference data (offline)                   |
|       No new data generated during training                       |
|                                                                   |
|  REWARD HACKING RISK                                              |
|  -----------------------------------------------------------------|
|  PPO: High -- the policy optimizes against the RM, which can      |
|       be exploited                                                |
|  DPO: Lower -- no separate reward model to exploit                |
|       But can still overfit to preference data                    |
|                                                                   |
|  PERFORMANCE                                                      |
|  -----------------------------------------------------------------|
|  PPO: Generally considered stronger, especially for RLHF           |
|       where online feedback is available                           |
|  DPO: Often competitive or equal, especially on chat/instruction  |
|       tasks. Sometimes slightly worse on complex reasoning tasks   |
|                                                                   |
|  IMPLEMENTATION EFFORT                                            |
|  -----------------------------------------------------------------|
|  PPO: Weeks to implement correctly                                |
|       Many moving parts to debug                                  |
|  DPO: Days to implement                                           |
|       One loss function, standard training loop                   |
+===================================================================+
```

### Code Comparison

```python
# ppo_vs_dpo_comparison.py
# Show the key difference in training loop structure

# =============================================
# PPO TRAINING (simplified pseudocode)
# =============================================
def ppo_training_step(policy, ref_model, reward_model, critic, prompt):
    """PPO requires all 4 components at every step."""
    
    # Step 1: Generate response (rollout)
    response = policy.generate(prompt)                  # LLM forward + sample
    
    # Step 2: Get reward model score
    rm_score = reward_model.score(prompt, response)     # SEPARATE network call
    
    # Step 3: Compute KL penalty
    kl = compute_kl(policy, ref_model, prompt, response) # Compare to ref
    
    # Step 4: Get critic's value estimate
    value = critic.predict(prompt, response)             # ANOTHER network call
    
    # Step 5: Compute advantage
    advantage = (rm_score - kl_penalty) - value
    
    # Step 6: PPO gradient update
    ppo_loss = compute_ppo_loss(policy, old_policy, advantage, response)
    
    # Step 7: Update policy weights
    ppo_loss.backward()    # compute gradients
    optimizer.step()       # update weights
    
    # Step 8: Update critic weights
    critic_loss = compute_critic_loss(value, rm_score)
    critic_loss.backward()
    critic_optimizer.step()
    
    # Need to manage: 4 models, 2 optimizers, complex rollout loop
    # Total forward passes: policy + reward_model + ref_model + critic = 4


# =============================================
# DPO TRAINING (simplified pseudocode)
# =============================================
def dpo_training_step(policy, ref_model, chosen, rejected, prompt):
    """DPO only needs 2 components and is much simpler."""
    
    # Step 1: Get log probs from policy (model being trained)
    log_p_chosen   = policy.log_prob(prompt, chosen)    # single forward pass
    log_p_rejected = policy.log_prob(prompt, rejected)  # single forward pass
    
    # Step 2: Get log probs from reference model (FROZEN, no gradients)
    with no_grad():  # ref model is frozen, no need to compute gradients
        log_r_chosen   = ref_model.log_prob(prompt, chosen)
        log_r_rejected = ref_model.log_prob(prompt, rejected)
    
    # Step 3: Compute DPO loss (one function call!)
    loss = dpo_loss(log_p_chosen, log_p_rejected,
                    log_r_chosen, log_r_rejected, beta=0.1)
    
    # Step 4: Update policy weights
    loss.backward()   # compute gradients
    optimizer.step()  # update weights
    
    # That's it! No reward model, no critic, no rollout loop.
    # Total forward passes: policy (x2) + ref_model (x2) = 4 but no RL loop
```

---

## Part 6: When to Use DPO vs PPO

Not every situation calls for the same technique.

### Use DPO When:

1. **You have a good preference dataset and don't need to collect more**
   DPO is offline -- it trains on existing data.
   If your dataset is fixed and high quality, DPO is often enough.

2. **You need to move fast or have limited compute**
   DPO is much simpler to implement and debug.
   No need for separate reward model training.

3. **You want stable, reproducible training**
   DPO behaves like standard fine-tuning.
   Very easy to monitor and control.

4. **You are doing instruction following, chat alignment, harmlessness**
   These tasks have well-defined preference data.
   DPO performs competitively with PPO on these benchmarks.

### Use PPO When:

1. **You need online/interactive feedback**
   If you can get real-time human feedback during training,
   PPO can use it. DPO cannot -- it is offline only.

2. **You are doing complex reasoning or coding tasks**
   PPO's online exploration can help the model discover better solutions.
   DPO can only learn from the patterns in your fixed dataset.

3. **Your preference dataset is small or low quality**
   PPO generates its own rollouts during training.
   It can make up for sparse preference data with exploration.

4. **You have a verifiable reward signal**
   For tasks like math (answer is right or wrong) or coding (tests pass or fail),
   you can use a rule-based reward signal instead of a reward model.
   This is called "RLVR" (RL from Verifiable Rewards) and requires PPO.

```
+---------------------------------------------------------------+
|  DECISION TREE: PPO or DPO?                                   |
|                                                               |
|  Do you have a fixed, high-quality preference dataset?        |
|    YES -> Do you need online/interactive feedback?            |
|              NO -> USE DPO  (simpler, stable, good results)   |
|              YES -> USE PPO (can incorporate new feedback)    |
|    NO  -> Do you have a verifiable reward signal?             |
|              YES -> USE PPO (RLVR variant)                    |
|              NO  -> Collect preference data first,            |
|                     then use DPO or train reward model + PPO  |
+---------------------------------------------------------------+
```

---

## Part 7: Beta -- The Key Hyperparameter in DPO

Beta (beta) is the only hyperparameter unique to DPO.
Understanding it is important.

```
L_DPO = -log(sigma( BETA * (log_ratio_chosen - log_ratio_rejected) ))
                     ^^^^
                This is beta
```

### What Beta Controls

```
+---------------------------------------------------------------+
|  BETA EFFECT ON DPO TRAINING                                  |
|                                                               |
|  LARGE beta (e.g., 1.0):                                      |
|    - Loss is very sensitive to small differences              |
|    - Model aggressively moves toward chosen responses         |
|    - Might overfit to preference data                         |
|    - Can drift far from reference model                       |
|    - Risk: forgetting base language capabilities              |
|                                                               |
|  SMALL beta (e.g., 0.01):                                     |
|    - Loss is insensitive to differences                       |
|    - Model barely changes from reference                      |
|    - Very conservative, stable training                       |
|    - Risk: not learning preferences effectively               |
|                                                               |
|  TYPICAL beta (0.1 to 0.5):                                   |
|    - Balanced between preference learning and stability       |
|    - Most papers use 0.1 or 0.5                              |
|    - Start with 0.1, tune from there                         |
+---------------------------------------------------------------+
```

```python
# beta_effect_demo.py
# Demonstrate how beta affects DPO loss magnitude

import numpy as np  # NumPy for math

def sigmoid(x):
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def dpo_loss_simple(log_ratio_chosen, log_ratio_rejected, beta):
    """
    Simplified DPO loss.
    
    Args:
        log_ratio_chosen:   log(pi_theta(y_w|x) / pi_ref(y_w|x))
        log_ratio_rejected: log(pi_theta(y_l|x) / pi_ref(y_l|x))
        beta:               temperature parameter
    
    Returns:
        loss: scalar
    """
    # Scale ratios by beta
    margin = beta * (log_ratio_chosen - log_ratio_rejected)
    
    # Bradley-Terry loss
    prob = sigmoid(margin)
    loss = -np.log(prob + 1e-8)
    return loss, margin, prob

# Scenario: policy has learned to prefer chosen over rejected
# log_ratio_chosen = 0.5  (policy assigns 0.5 more nats to chosen than ref)
# log_ratio_rejected = -0.5 (policy assigns 0.5 fewer nats to rejected than ref)
log_ratio_chosen   = 0.5
log_ratio_rejected = -0.5

print("Effect of beta on DPO loss\n")
print(f"{'beta':>6} | {'margin':>8} | {'P(chosen>rejected)':>20} | {'loss':>8}")
print("-" * 55)

for beta in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    loss, margin, prob = dpo_loss_simple(log_ratio_chosen, log_ratio_rejected, beta)
    print(f"  {beta:>4.2f} | {margin:>8.4f} | {prob:>20.4f} | {loss:>8.4f}")

print()
print("Note: larger beta -> larger margin -> higher probability -> lower loss")
print("But: too large beta can cause training instability")
```

---

## Part 8: DPO in Practice -- What Libraries Provide

In real projects, you would use a library for DPO rather than implementing from scratch.

The most popular is **TRL (Transformer Reinforcement Learning)** from HuggingFace.

Here is a sketch of how you would use it (do not run this -- it requires GPU):

```python
# dpo_with_trl.py
# Sketch of DPO training using HuggingFace TRL library
# This shows the API, not meant to be run directly

# pip install trl transformers datasets

from trl import DPOTrainer, DPOConfig           # DPO training components
from transformers import AutoModelForCausalLM   # Pre-trained LLM
from transformers import AutoTokenizer           # Text -> tokens
from datasets import load_dataset               # Load preference data

# ============================================================
# STEP 1: Load the model and tokenizer
# ============================================================

model_name = "facebook/opt-125m"  # small model for demo (125M params)

# Load model: this is the policy (will be updated by DPO)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load tokenizer: converts text to tokens and back
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load reference model: frozen copy, used for KL constraint
# TRL handles this automatically when you use DPOTrainer
ref_model = AutoModelForCausalLM.from_pretrained(model_name)

# ============================================================
# STEP 2: Load preference data
# ============================================================

# Load a preference dataset
# The dataset must have "prompt", "chosen", "rejected" fields
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
# [:1000] means only the first 1000 examples (small for demo)

# ============================================================
# STEP 3: Configure DPO training
# ============================================================

training_args = DPOConfig(
    output_dir="./dpo_model",       # where to save checkpoints
    num_train_epochs=1,             # number of passes over the data
    per_device_train_batch_size=4,  # 4 examples per GPU per step
    learning_rate=1e-5,             # how fast to update weights
    beta=0.1,                       # DPO temperature parameter
    logging_steps=10,               # print progress every 10 steps
    save_steps=100,                 # save checkpoint every 100 steps
)

# ============================================================
# STEP 4: Create and run the trainer
# ============================================================

trainer = DPOTrainer(
    model=model,                    # policy to train
    ref_model=ref_model,            # reference model (frozen)
    args=training_args,             # training configuration
    train_dataset=dataset,          # preference data
    processing_class=tokenizer,     # tokenizer
)

# Start DPO training!
# Under the hood, this:
#   1. Batches the preference data
#   2. Runs both model and ref_model forward passes
#   3. Computes DPO loss
#   4. Backpropagates and updates model weights
#   5. Logs metrics and saves checkpoints
trainer.train()

# ============================================================
# STEP 5: Save the aligned model
# ============================================================

trainer.save_model("./dpo_aligned_model")
print("DPO training complete! Model saved.")
```

---

## Part 9: Evaluating DPO Training

How do you know if DPO is working?

### Metric 1: Implicit Reward Margin

During training, log the average (reward_chosen - reward_rejected).
This should INCREASE over time.

```
Iteration 1:   margin = 0.01  (barely any separation)
Iteration 100: margin = 0.35  (model is learning)
Iteration 500: margin = 0.82  (strong preference learned)
```

### Metric 2: Policy Accuracy

What percentage of preference pairs does the current policy correctly order?
(i.e., assigned higher probability to chosen than rejected)

Should increase from ~50% (random) toward 70-80%+.

### Metric 3: Win Rate Against Reference

Sample responses from both the DPO-trained model and the reference model.
Have humans (or another LLM) judge which is better.
DPO-trained model should win >50% of the time.

### Red Flags

```
+---------------------------------------------------------------+
|  DPO TRAINING RED FLAGS                                       |
|                                                               |
|  Problem: reward margin is DECREASING (chosen < rejected)     |
|  Cause:   learning rate too high, beta too small              |
|  Fix:     reduce learning rate, increase beta                 |
|                                                               |
|  Problem: policy becomes incoherent (generates garbage text)  |
|  Cause:   overfitting, training too long                      |
|  Fix:     stop earlier, reduce epochs, use LoRA               |
|                                                               |
|  Problem: model just refuses everything                        |
|  Cause:   harmlessness data too dominant in preference pairs  |
|  Fix:     balance dataset between helpful and harmless pairs  |
|                                                               |
|  Problem: model is sycophantic (agrees with everything)        |
|  Cause:   same sycophancy bias in annotations as in RLHF     |
|  Fix:     include preference pairs that reward honest refusals |
+---------------------------------------------------------------+
```

---

## Summary

```
+---------------------------------------------------------------+
|  LESSON 04 SUMMARY                                            |
|                                                               |
|  1. DPO Motivation                                            |
|     PPO is complex and fragile.                               |
|     DPO achieves similar results with one loss function.      |
|                                                               |
|  2. The Insight                                               |
|     The optimal RLHF policy implies an implicit reward:       |
|     r(x,y) = beta * log(pi_theta(y|x) / pi_ref(y|x))        |
|     Plugging into Bradley-Terry loss gives the DPO loss.      |
|                                                               |
|  3. The DPO Loss                                              |
|     L = -log(sigma(beta * (log_ratio_chosen                   |
|                          - log_ratio_rejected)))              |
|     "Maximize the gap between chosen and rejected             |
|      in terms of how much the policy changed from reference." |
|                                                               |
|  4. DPO vs PPO                                                |
|     DPO: simpler, stable, offline, no reward model needed     |
|     PPO: complex, powerful, online, can use new feedback      |
|                                                               |
|  5. When to Use DPO                                           |
|     Fixed preference dataset, chat/instruction tasks,         |
|     limited compute, need for stable training.                 |
|                                                               |
|  6. Beta                                                      |
|     Controls how aggressively preferences are learned.        |
|     Typical values: 0.1 to 0.5                               |
+---------------------------------------------------------------+
```

---

## Quiz Questions

1. What problem does DPO solve that makes it attractive compared to PPO?

2. Write out the DPO loss formula. What does each term represent?

3. What is the "implicit reward" in DPO? How is it computed?

4. What is the "reference model" in DPO, and why is it needed?

5. If beta is very large (e.g., 100), what happens to DPO training?
   If beta is very small (e.g., 0.001)?

6. DPO is called "offline" training. What does that mean, and how is it
   different from PPO which is "on-policy"?

7. Give two scenarios where you would choose PPO over DPO.

8. In the DPO loss, if log_ratio_chosen = 0.5 and log_ratio_rejected = -0.3
   with beta = 0.1, what is the margin? Is this a good sign?

---

## Lab Exercise

```python
# lab_04_dpo_loss.py
# Implement the DPO loss from scratch

import numpy as np

# ============================================================
# TASK 1: Implement the sigmoid function
# ============================================================

def sigmoid(x):
    """
    Sigmoid: sigma(x) = 1 / (1 + e^(-x))
    Use a numerically stable implementation.
    Hint: use np.clip to prevent exp overflow.
    """
    # YOUR CODE HERE
    pass


# ============================================================
# TASK 2: Implement the DPO loss
# ============================================================

def dpo_loss(log_prob_policy_chosen, log_prob_policy_rejected,
             log_prob_ref_chosen, log_prob_ref_rejected, beta=0.1):
    """
    Compute the DPO loss.
    
    Args:
        log_prob_policy_chosen:   log P(y_w | x) under trained policy
        log_prob_policy_rejected: log P(y_l | x) under trained policy
        log_prob_ref_chosen:      log P(y_w | x) under reference model
        log_prob_ref_rejected:    log P(y_l | x) under reference model
        beta:                     temperature parameter (default 0.1)
    
    Returns:
        loss:           scalar or array, the DPO loss
        reward_chosen:  implicit reward for chosen response
        reward_rejected:implicit reward for rejected response
    
    Steps:
        1. Compute log_ratio_chosen = log_prob_policy_chosen - log_prob_ref_chosen
        2. Compute log_ratio_rejected = log_prob_policy_rejected - log_prob_ref_rejected
        3. reward_chosen = beta * log_ratio_chosen
        4. reward_rejected = beta * log_ratio_rejected
        5. margin = reward_chosen - reward_rejected
        6. loss = -log(sigmoid(margin))
    """
    # YOUR CODE HERE
    pass


# ============================================================
# TESTS: Verify your implementation
# ============================================================

# Test 1: When policy perfectly prefers chosen, loss should be near 0
perfect_loss, r_w, r_l = dpo_loss(
    log_prob_policy_chosen=-2.0,   # policy assigns high prob to chosen
    log_prob_policy_rejected=-8.0, # policy assigns low prob to rejected
    log_prob_ref_chosen=-5.0,      # reference baseline
    log_prob_ref_rejected=-5.0,    # reference baseline
    beta=0.1
)
print(f"Test 1 - Perfect case loss: {perfect_loss:.4f} (should be < 0.5)")
assert perfect_loss < 0.5, "Perfect case should have low loss"

# Test 2: When policy gets it backwards, loss should be high
wrong_loss, r_w, r_l = dpo_loss(
    log_prob_policy_chosen=-8.0,   # policy assigns LOW prob to chosen (wrong!)
    log_prob_policy_rejected=-2.0, # policy assigns HIGH prob to rejected (wrong!)
    log_prob_ref_chosen=-5.0,
    log_prob_ref_rejected=-5.0,
    beta=0.1
)
print(f"Test 2 - Wrong case loss: {wrong_loss:.4f} (should be > 1.0)")
assert wrong_loss > 1.0, "Wrong case should have high loss"

# Test 3: When policy is same as reference, margin should be 0
neutral_loss, r_w, r_l = dpo_loss(
    log_prob_policy_chosen=-5.0,
    log_prob_policy_rejected=-5.0,
    log_prob_ref_chosen=-5.0,
    log_prob_ref_rejected=-5.0,
    beta=0.1
)
print(f"Test 3 - Neutral case loss: {neutral_loss:.4f} (should be ~0.693)")
assert abs(neutral_loss - 0.693) < 0.01, "Neutral case: loss should be -log(0.5) = 0.693"

print("\nAll DPO loss tests passed! Excellent work.")
```

---

*Next lesson: Constitutional AI and safety systems.*
*File: lessons/05_constitutional_ai_safety.md*

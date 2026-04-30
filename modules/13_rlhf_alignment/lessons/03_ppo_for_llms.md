# Lesson 03: PPO for LLMs

## Glossary (Read This First!)

| Term | Plain English Definition |
|------|--------------------------|
| **PPO** | Proximal Policy Optimization. A reinforcement learning algorithm that updates a "policy" (the LLM) to maximize reward, while preventing updates that are too large. |
| **Policy** | In RL, the function that maps a state (the current prompt/context) to actions (next token probabilities). The LLM IS the policy. |
| **Value Function** | A neural network that predicts "how much reward do I expect to get from this point forward?" Used to reduce variance in RL training. |
| **Advantage** | How much better (or worse) was an action compared to what the value function expected? Advantage = Actual Reward - Expected Reward. |
| **KL Divergence** | A mathematical measure of how different two probability distributions are. Used in PPO to prevent the aligned LLM from drifting too far from the SFT model. |
| **KL Penalty** | A penalty added to the reward that discourages the LLM from changing too much from the starting SFT model. |
| **Clipping** | In PPO, limiting how much the policy update can change the probability of any action. Prevents catastrophically large updates. |
| **Rollout** | One complete episode of: generate a response -> score it -> use the score for learning. |
| **Actor-Critic** | An RL architecture with two networks: the Actor (chooses actions = the LLM policy) and the Critic (estimates value = predicts expected reward). |
| **Entropy Bonus** | A bonus reward for having diverse/uncertain outputs. Prevents the model from always choosing the same token (getting stuck). |
| **Credit Assignment** | The problem of figuring out which tokens in a long response were "responsible" for the final reward score. Hard in language models because sequences are long. |
| **Reward Sparsity** | The problem that reward is only given at the end of a sequence, not after each token. Makes learning harder. |

---

## Part 1: Why Use Reinforcement Learning for Language Models?

After Phases 1 and 2 of RLHF, we have:
- An SFT model (knows how to follow instructions)
- A Reward Model (knows how to score responses)

The question now is: **how do we use the reward model to improve the LLM?**

The naive approach would be:
1. Generate a response
2. Get its reward score
3. Add it to fine-tuning data if score is high

But this is slow and wasteful. We throw away all the information about WHY
certain responses scored high.

Reinforcement Learning gives us a principled framework for:
- Generating responses (exploration)
- Scoring them (getting a reward signal)
- Updating the model to do better next time (policy gradient)

```
+---------------------------------------------------------------+
|  THE RL FRAMING OF LANGUAGE MODELING                          |
|                                                               |
|  Traditional RL:                                              |
|    Agent: a player in a game                                  |
|    State: the current board/game situation                    |
|    Action: a move (left, right, up, down)                     |
|    Reward: +1 for winning, -1 for losing                      |
|                                                               |
|  RLHF for LLMs:                                               |
|    Agent: the LLM                                             |
|    State: the prompt + tokens generated so far                |
|    Action: the next token to generate                         |
|    Reward: reward model score at end of sequence              |
|                                                               |
|  The "game" is: given a prompt, generate a response.          |
|  The "score" at the end is: how good was that response?       |
+---------------------------------------------------------------+
```

C# Analogy:
```csharp
// Think of the LLM as a chess engine.
// Traditional training (supervised learning):
//   Show it millions of grandmaster games.
//   Teach it to COPY grandmaster moves.
//   Problem: it can only be as good as its training examples.

// RL training (RLHF):
//   The chess engine PLAYS GAMES against itself.
//   After each game, it gets a reward: +1 win, -1 loss.
//   It learns from its OWN experience, not just copying.
//   It can discover moves no human ever played!

// RLHF for LLMs is similar:
//   The LLM generates responses to prompts.
//   The reward model scores them (instead of a chess outcome).
//   PPO updates the LLM to generate higher-scoring responses.
//   The LLM can learn to be helpful in ways beyond its SFT training data.
```

---

## Part 2: The LLM as a Policy

In reinforcement learning, the **policy** is the function that decides what to do.

For a language model:
- Input (state): all tokens so far (prompt + partial response)
- Output (action): probability distribution over the next token

```
+---------------------------------------------------------------+
|  LLM AS A POLICY (Simplified)                                 |
|                                                               |
|  State: "The capital of France is"                            |
|                                                               |
|  Policy output (probability over vocabulary):                 |
|    "Paris"    : 0.78                                          |
|    "Paris,"   : 0.10                                          |
|    "Berlin"   : 0.04                                          |
|    "London"   : 0.03                                          |
|    " located" : 0.02                                          |
|    ... (all other tokens sum to 0.03)                         |
|                                                               |
|  Action: sample from this distribution -> "Paris"             |
|                                                               |
|  New State: "The capital of France is Paris"                  |
|  (next action will extend this further)                       |
+---------------------------------------------------------------+
```

Why is this an RL problem?
- Each token choice is an "action"
- The sequence of tokens is the "episode"
- The reward comes only at the END (when the full response is complete)
- We need to figure out which token choices led to a good or bad response

---

## Part 3: The PPO Algorithm -- Overview

PPO (Proximal Policy Optimization) was developed by OpenAI in 2017.
It is one of the most stable and widely used policy gradient algorithms.

"Proximal" means "close." The key idea: **update the policy, but not too much at once.**

Here is the full PPO loop for RLHF:

```
+================================================================+
|                    PPO TRAINING LOOP                           |
+================================================================+
|                                                                |
|  Initialize:                                                   |
|    policy (pi_theta)      = SFT model                          |
|    reference (pi_ref)     = SFT model (frozen copy)            |
|    reward model (RM)      = trained in Phase 2                 |
|    value network (critic) = separate small network             |
|                                                                |
|  Repeat for many iterations:                                   |
|                                                                |
|  STEP 1: ROLLOUT (Generate Experience)                         |
|  -------------------------------------------------------       |
|    - Sample a batch of prompts from the prompt dataset         |
|    - For each prompt, use the CURRENT policy to generate       |
|      a complete response (autoregressively, token by token)    |
|    - Store: (prompt, response, all token log-probabilities)    |
|                                                                |
|  STEP 2: SCORE (Get Rewards)                                   |
|  -------------------------------------------------------       |
|    - Run each (prompt, response) through the Reward Model      |
|    - Get a scalar score for each response                      |
|    - Apply KL penalty: subtract beta * KL(pi || pi_ref)        |
|    - This is the "shaped reward" for training                  |
|                                                                |
|  STEP 3: COMPUTE ADVANTAGES                                    |
|  -------------------------------------------------------       |
|    - Use the critic (value network) to estimate                |
|      "expected reward" for each state                          |
|    - Advantage = actual reward - expected reward               |
|    - Positive advantage: this was BETTER than expected         |
|    - Negative advantage: this was WORSE than expected          |
|                                                                |
|  STEP 4: UPDATE POLICY (PPO Gradient Step)                     |
|  -------------------------------------------------------       |
|    - For each token in each response:                          |
|      * Compute ratio: new_prob / old_prob                      |
|      * Compute clipped ratio: clamp(ratio, 1-eps, 1+eps)       |
|      * PPO loss = -min(ratio * advantage, clipped * advantage) |
|    - Update policy weights to MINIMIZE the PPO loss            |
|    - Update critic weights too                                 |
|                                                                |
|  STEP 5: EVALUATE (periodically)                               |
|  -------------------------------------------------------       |
|    - Test on held-out prompts                                  |
|    - Check KL divergence from SFT model (should stay bounded)  |
|    - Check reward model scores (should be going up)            |
|    - Stop when performance plateaus or KL gets too large       |
|                                                                |
+================================================================+
```

---

## Part 4: Understanding the PPO Objective

The PPO loss function looks scary but has a clear purpose:

```
PPO Loss = -E[ min( r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t ) ]

Where:
  r_t = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
      = (new probability of this action) / (old probability of this action)
  
  A_t = Advantage at step t
      = how much better was this action than expected?
  
  clip(x, a, b) = clamp x to the range [a, b]
  
  eps = a small number, typically 0.1 or 0.2
```

### Why Clipping?

Without clipping, if an action had a very high advantage, the gradient update
would make that action's probability EXTREMELY high.
This can cause catastrophic forgetting or destabilize training.

Clipping says: "Even if this action was great, don't increase its probability
by more than (1 + epsilon) times."

```
+---------------------------------------------------------------+
|  PPO CLIPPING INTUITION                                       |
|                                                               |
|  eps = 0.2 (typical value)                                    |
|                                                               |
|  ALLOWED ratio range: [0.8, 1.2]                             |
|                                                               |
|  Example: action "Paris" had prob 0.6 in old policy           |
|                                                               |
|  If advantage = +2.0 (much better than expected):             |
|    Without clip: r might become 1.8 -> prob = 0.6 * 1.8 = 1.08!  CRASH!  |
|    With clip:    r is capped at 1.2 -> prob = 0.6 * 1.2 = 0.72  Safe!    |
|                                                               |
|  Example: action "Berlin" had prob 0.05 in old policy         |
|  If advantage = -1.5 (worse than expected):                   |
|    Without clip: r might become 0.1 -> prob = 0.05 * 0.1 = 0.005  Very low! |
|    With clip:    r is capped at 0.8 -> prob = 0.05 * 0.8 = 0.04  Safer!  |
+---------------------------------------------------------------+
```

C# Analogy:
```csharp
// PPO is like a gradient descent where the "loss function" is computed
// by trying actions in the environment (the reward model),
// not from pre-computed labels.

// Imagine a recommendation system:

// Normal gradient descent (supervised):
//   You KNOW what users should like (labeled data).
//   You update weights to match those labels.
//   Like: user liked "Inception" -> increase weights for sci-fi thrillers.

// PPO (reinforcement learning):
//   You TRY different recommendations.
//   Users rate them (reward signal).
//   You update weights to recommend more of what got high ratings.
//   The "clip" prevents you from completely ignoring all other genres
//   just because one sci-fi movie got a high rating.

// The "proximal" constraint (clipping) is like saying:
// "Learn from experience, but don't overreact to any single outcome."
// This is similar to having a maximum learning rate.
```

---

## Part 5: The KL Penalty -- Don't Drift Too Far

This is one of the most important parts of RLHF.

Without any constraint, PPO would optimize the LLM to maximize reward.
But the reward model is imperfect. It can be fooled.
If the LLM drifts too far from its original behavior, it might:
- Find reward model exploits (reward hacking)
- Forget how to write coherent language
- Become pathologically single-minded

The **KL penalty** prevents this.

### What Is KL Divergence?

KL divergence measures how different two probability distributions are.

```
KL(P || Q) = sum over all x: P(x) * log( P(x) / Q(x) )

Where:
  P = the current (updated) policy
  Q = the reference (SFT) policy

KL = 0 means P and Q are identical.
KL > 0 means P has drifted from Q.
KL -> infinity means P has completely changed.
```

### The KL-Penalized Reward

During PPO training, the reward is adjusted:

```
Total Reward = RM_score - beta * KL(pi_theta || pi_ref)

Where:
  RM_score  = reward model score for the response
  pi_theta  = current policy (being trained)
  pi_ref    = reference policy (frozen SFT model)
  beta      = KL coefficient (controls how much we penalize drift)
              typical values: 0.01 to 0.5
```

Plain English:
- High reward score + small KL = good (helpful AND similar to SFT)
- High reward score + large KL = penalized (high score but too different from SFT)
- The model must BALANCE being helpful (high RM score) and staying coherent (low KL)

```
+---------------------------------------------------------------+
|  KL PENALTY IN ACTION                                         |
|                                                               |
|  beta = 0.1 (typical)                                         |
|                                                               |
|  Response A:                                                  |
|    RM score = 0.8 (good response)                             |
|    KL divergence from SFT = 0.05 (small drift)               |
|    Total reward = 0.8 - 0.1 * 0.05 = 0.795                   |
|                                                               |
|  Response B:                                                  |
|    RM score = 0.9 (very good response)                        |
|    KL divergence from SFT = 5.0 (LARGE drift!)               |
|    Total reward = 0.9 - 0.1 * 5.0 = 0.4   <- penalized!      |
|                                                               |
|  The model prefers Response A even though B scored higher     |
|  on the reward model, because B has drifted too far.          |
+---------------------------------------------------------------+
```

### How to Compute KL Divergence Per Token

For language models, KL is computed token by token:

```python
# kl_divergence_example.py
# Demonstrate KL divergence computation for language models

import numpy as np  # NumPy for array math

def compute_kl_divergence(log_probs_policy, log_probs_reference):
    """
    Compute KL divergence between policy and reference distributions.
    
    KL(pi_theta || pi_ref) = sum_x pi_theta(x) * log(pi_theta(x) / pi_ref(x))
                           = sum_x pi_theta(x) * (log_pi_theta(x) - log_pi_ref(x))
    
    In practice, we compute this per token and average over the sequence.
    
    Args:
        log_probs_policy:    log probabilities from the policy being trained
                             shape: (sequence_length, vocab_size)
        log_probs_reference: log probabilities from the frozen SFT model
                             shape: (sequence_length, vocab_size)
    
    Returns:
        kl: scalar KL divergence value (non-negative)
    """
    # Convert log probabilities to probabilities
    # exp(log(p)) = p
    probs_policy = np.exp(log_probs_policy)    # pi_theta
    
    # KL formula: pi_theta * (log_pi_theta - log_pi_ref)
    # Per position: sum over vocab, then average over sequence
    kl_per_position = np.sum(
        probs_policy * (log_probs_policy - log_probs_reference),
        axis=-1  # sum over vocabulary dimension
    )
    
    # Average over sequence positions
    kl = np.mean(kl_per_position)
    
    return kl


# ============================================================
# DEMO: Compare KL divergence for similar vs. different models
# ============================================================

# Simulate log probabilities for a 3-token sequence
# with a vocabulary of 4 tokens (simplified)
# In reality: sequences are hundreds of tokens, vocab is 50000+

seq_len = 3   # number of tokens in the response
vocab_size = 4  # number of possible tokens (simplified)

# Reference model (SFT) outputs -- these are log probabilities
log_probs_ref = np.array([
    [-0.5, -1.2, -2.0, -3.1],   # position 0: probs ~ [0.61, 0.30, 0.14, 0.04]
    [-0.3, -1.5, -2.5, -2.8],   # position 1: probs ~ [0.74, 0.22, 0.08, 0.06]
    [-1.0, -0.8, -2.0, -3.0],   # position 2: probs ~ [0.37, 0.45, 0.14, 0.05]
])

# Policy that is SIMILAR to the reference (small KL expected)
log_probs_similar = log_probs_ref + np.random.randn(seq_len, vocab_size) * 0.1

# Policy that is DIFFERENT from the reference (large KL expected)
log_probs_different = np.random.randn(seq_len, vocab_size) - 2.0  # random

# Normalize (log_softmax) to make valid log probability distributions
def log_softmax(x):
    """Convert raw scores to log probabilities (normalized)."""
    # Subtract max for numerical stability
    x = x - x.max(axis=-1, keepdims=True)
    # log(exp(x) / sum(exp(x))) = x - log(sum(exp(x)))
    return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))

log_probs_ref       = log_softmax(log_probs_ref)
log_probs_similar   = log_softmax(log_probs_similar)
log_probs_different = log_softmax(log_probs_different)

# Compute KL divergences
kl_similar   = compute_kl_divergence(log_probs_similar, log_probs_ref)
kl_different = compute_kl_divergence(log_probs_different, log_probs_ref)

print(f"KL divergence (similar policy):   {kl_similar:.4f}")
print(f"KL divergence (different policy): {kl_different:.4f}")
print()
print("Small KL = model hasn't changed much from SFT (good!)")
print("Large KL = model has drifted far from SFT (penalized!)")

# Compute penalized rewards
beta = 0.1            # KL coefficient
rm_score = 0.85       # reward model score (same for both)

penalized_reward_similar   = rm_score - beta * kl_similar
penalized_reward_different = rm_score - beta * kl_different

print(f"\nPenalized reward (similar policy):   {penalized_reward_similar:.4f}")
print(f"Penalized reward (different policy): {penalized_reward_different:.4f}")
```

---

## Part 6: Why PPO Is Hard for Language Models

PPO was designed for simple game environments (like Atari games).
Applying it to language models introduces unique challenges:

### Challenge 1: Long Action Sequences

In Atari: one action per step (move left/right/shoot)
In LLMs: generating a response requires 100-500 token actions

This makes the **credit assignment** problem much harder.

```
+---------------------------------------------------------------+
|  CREDIT ASSIGNMENT PROBLEM                                    |
|                                                               |
|  Response: "Paris is the capital of France, not Germany."    |
|  Tokens:   Paris | is | the | capital | of | France | , |    |
|            not | Germany | .                                  |
|                                                               |
|  Final reward: 0.9 (high score!)                              |
|                                                               |
|  Which tokens were responsible for the high score?            |
|    "Paris" was important (correct answer)                     |
|    "not Germany" was important (clarified a misconception)    |
|    "the" was not very important                               |
|                                                               |
|  PPO must assign credit to EACH token individually.           |
|  But the reward only comes at the END.                        |
|  PPO distributes reward backward through the sequence.        |
|  This works, but is noisy for long sequences.                 |
+---------------------------------------------------------------+
```

### Challenge 2: Reward Sparsity

In Atari: you get a score after every action (dense reward)
In LLMs: you get one score at the END of the full response (sparse reward)

This makes learning slower and noisier.

Techniques to mitigate sparsity:
- Process Reward Models (PRM): score each step, not just the end
- Dense reward shaping: add per-token KL penalties as intermediate rewards

### Challenge 3: Vocabulary Size

In Atari: typically 18 possible actions
In LLMs: 50,000+ possible tokens

This makes the policy space enormous.
PPO must be careful not to explore too randomly (leads to incoherent text).

### Challenge 4: Catastrophic Forgetting

If PPO updates are too large, the LLM might forget how to write grammatical text.
The KL penalty helps, but requires careful tuning of beta.

---

## Part 7: Practical PPO Implementation Tips

Here are lessons learned from real RLHF implementations:

```
+---------------------------------------------------------------+
|  PRACTICAL PPO TIPS FOR RLHF                                  |
|                                                               |
|  1. SMALL KL COEFFICIENT (beta)                               |
|     Start with beta = 0.01 to 0.1                             |
|     Too large: model barely changes (too constrained)         |
|     Too small: model drifts into reward hacking               |
|     Monitor KL divergence; stop if it grows too fast          |
|                                                               |
|  2. SMALL CLIPPING PARAMETER (eps)                            |
|     Use eps = 0.1 or 0.2                                      |
|     Prevents single-step catastrophic updates                 |
|                                                               |
|  3. ENTROPY BONUS                                             |
|     Add a small reward for uncertain/diverse outputs          |
|     Prevents model from always saying the same thing          |
|     Coefficient typically 0.01                                |
|                                                               |
|  4. SEPARATE CRITIC NETWORK                                   |
|     Use a value network to predict expected reward            |
|     Reduces variance in advantage estimates                   |
|     Train critic with mean squared error loss                 |
|                                                               |
|  5. EARLY STOPPING                                            |
|     Monitor KL divergence continuously                        |
|     If KL > threshold (e.g., 0.5): stop training!            |
|     Better to stop early than destroy the model               |
|                                                               |
|  6. REWARD NORMALIZATION                                       |
|     Normalize rewards to zero mean, unit variance             |
|     Prevents one batch from dominating the update             |
|                                                               |
|  7. EXPERIENCE REPLAY                                         |
|     Reuse generated sequences for multiple update steps       |
|     More efficient than generating new responses each step    |
+---------------------------------------------------------------+
```

---

## Part 8: PPO Pseudocode (Simplified for LLMs)

Here is the full PPO training loop in simplified Python:

```python
# ppo_simplified.py
# Simplified PPO loop for language model alignment
# This shows the CONCEPTS without full PyTorch implementation

import numpy as np  # NumPy for array operations

# ============================================================
# SIMULATION SETUP
# (In reality: LLM, reward model, and critic are PyTorch models)
# ============================================================

class FakePolicy:
    """
    Simulates an LLM policy.
    In reality: this is a transformer model.
    Here: we fake it with random numbers to show the structure.
    """
    
    def __init__(self):
        # In reality: these are millions of parameters
        # Here: a single "quality" parameter that improves over time
        self.quality_score = 0.3  # starts bad, should improve to ~0.8
    
    def generate(self, prompt):
        """
        Generate a response to the prompt.
        In reality: autoregressive decoding, token by token.
        Here: returns a simulated "response quality" score.
        """
        # Simulate: better quality_score = better responses (usually)
        noise = np.random.randn() * 0.1
        return self.quality_score + noise
    
    def log_prob(self, prompt, response):
        """
        Compute log probability of the response under current policy.
        In reality: sum of log probs of each token.
        Here: simulated value.
        """
        return -abs(response - self.quality_score)  # fake log prob


class FakeRewardModel:
    """
    Simulates a reward model.
    In reality: a transformer that outputs a scalar.
    Here: simple function that rewards responses close to 0.9.
    """
    
    def score(self, prompt, response):
        """
        Score a response. Higher = better.
        In reality: forward pass through reward model.
        Here: reward is high when response_quality is near 0.9.
        """
        # The "ideal" response quality is 0.9
        # Reward decreases as we move away from 0.9
        reward = 1.0 - abs(response - 0.9)
        return max(0.0, reward)  # clamp to non-negative


class FakeCritic:
    """
    Simulates a value/critic network.
    Predicts the expected future reward from current state.
    """
    
    def __init__(self):
        self.estimated_value = 0.5  # starts as a guess
    
    def predict(self, state):
        """Predict expected reward from this state."""
        return self.estimated_value + np.random.randn() * 0.05
    
    def update(self, actual_reward):
        """Update estimate toward actual reward (simplified)."""
        learning_rate = 0.1
        self.estimated_value += learning_rate * (actual_reward - self.estimated_value)


# ============================================================
# PPO TRAINING LOOP (Simplified)
# ============================================================

def run_ppo_training(num_iterations=20, batch_size=4, beta=0.1):
    """
    Run simplified PPO training loop.
    
    Args:
        num_iterations: how many PPO update steps to run
        batch_size:     how many prompts to process per step
        beta:           KL penalty coefficient
    """
    
    # Initialize components
    policy    = FakePolicy()      # the LLM (starts as SFT model)
    ref_model = FakePolicy()      # frozen copy of SFT model (for KL)
    ref_model.quality_score = policy.quality_score  # same starting point
    reward_model = FakeRewardModel()  # trained reward model
    critic = FakeCritic()         # value network
    
    print("=== PPO Training Loop ===\n")
    print(f"Starting policy quality: {policy.quality_score:.3f}")
    print(f"Target quality (from reward model): 0.9")
    print()
    
    # Fake prompts (in reality: thousands of diverse prompts)
    prompts = ["Explain X", "What is Y", "How does Z work", "Describe W"]
    
    for iteration in range(num_iterations):
        
        # =====================================================
        # STEP 1: ROLLOUT -- Generate responses
        # =====================================================
        
        batch_rewards = []   # store rewards for this batch
        batch_advantages = []  # store advantages
        
        for i in range(batch_size):
            prompt = prompts[i % len(prompts)]  # cycle through prompts
            
            # Generate response using current policy
            response = policy.generate(prompt)
            
            # Get log probability of this response under current policy
            log_prob_policy = policy.log_prob(prompt, response)
            
            # Get log probability under REFERENCE (SFT) policy
            log_prob_ref = ref_model.log_prob(prompt, response)
            
            # =====================================================
            # STEP 2: SCORE -- Get reward + KL penalty
            # =====================================================
            
            # Get reward model score
            rm_score = reward_model.score(prompt, response)
            
            # Compute per-token KL penalty
            # KL = log(pi_policy) - log(pi_ref) for the generated tokens
            # (simplified: in reality this is summed over all tokens)
            kl_penalty = log_prob_policy - log_prob_ref
            
            # Total shaped reward = RM score - beta * KL
            shaped_reward = rm_score - beta * kl_penalty
            
            # =====================================================
            # STEP 3: COMPUTE ADVANTAGE
            # =====================================================
            
            # Critic predicts expected reward for this state
            expected_reward = critic.predict(response)
            
            # Advantage: how much better was actual reward vs. expectation?
            # Positive advantage: this response was better than expected
            # Negative advantage: this response was worse than expected
            advantage = shaped_reward - expected_reward
            
            batch_rewards.append(shaped_reward)
            batch_advantages.append(advantage)
            
            # Update critic toward actual reward
            critic.update(shaped_reward)
        
        # =====================================================
        # STEP 4: UPDATE POLICY (simplified gradient step)
        # =====================================================
        
        # Average advantage for this batch
        avg_advantage = np.mean(batch_advantages)
        avg_reward    = np.mean(batch_rewards)
        
        # Update policy quality in direction of positive advantage
        # (In reality: compute PPO gradient and backpropagate)
        policy_learning_rate = 0.05  # small learning rate
        
        if avg_advantage > 0:
            # Responses were better than expected: move quality UP
            policy.quality_score += policy_learning_rate * abs(avg_advantage)
        else:
            # Responses were worse than expected: adjust
            policy.quality_score += policy_learning_rate * avg_advantage
        
        # Clamp quality to reasonable range [0, 1]
        policy.quality_score = np.clip(policy.quality_score, 0, 1)
        
        # Print progress every 5 iterations
        if (iteration + 1) % 5 == 0:
            # Compute KL divergence from reference
            kl = abs(policy.quality_score - ref_model.quality_score)
            print(f"Iteration {iteration+1:3d}: "
                  f"avg_reward={avg_reward:.3f}, "
                  f"policy_quality={policy.quality_score:.3f}, "
                  f"KL_from_ref={kl:.3f}")
    
    print(f"\nFinal policy quality: {policy.quality_score:.3f}")
    print(f"(Target was 0.9 -- how close did we get?)")


# Run the training loop
run_ppo_training(num_iterations=20, batch_size=4, beta=0.1)
```

---

## Part 9: The Full PPO Objective (Mathematical)

For completeness, here is the full PPO objective used in practice:

```
Total PPO Loss = L_CLIP + c1 * L_VALUE - c2 * L_ENTROPY

Where:
  L_CLIP    = PPO clipped surrogate loss (main policy gradient)
              -min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
  
  L_VALUE   = value function loss
              MSE between critic predictions and actual rewards
              (c1 is a coefficient, typically 0.5)
  
  L_ENTROPY = entropy bonus
              -sum(pi * log(pi)) -- encourages diverse outputs
              (c2 is a coefficient, typically 0.01)

  r_t       = ratio = new_prob / old_prob for each token
  A_t       = advantage estimate at position t
  eps       = clipping parameter, typically 0.2
```

In RLHF specifically, the total reward includes the KL penalty:

```
reward_t = RM_score (at end of sequence) - beta * KL_t

KL_t = log pi_theta(a_t | s_t) - log pi_ref(a_t | s_t)
     = per-token KL divergence from reference model
```

---

## Part 10: What Does a Trained RLHF Model Look Like?

After successful PPO training, the model should:

1. Answer questions more helpfully than the SFT-only model
2. Refuse genuinely harmful requests politely
3. Admit uncertainty when it doesn't know something
4. Match the length and format humans prefer
5. Be consistent in its values and behavior

It should NOT:
- Be obsessed with bullet points (reward hacking)
- Always give the longest possible answer (length bias)
- Start every response with "Certainly!" (sycophancy pattern)

The KL penalty ensures it still:
- Writes grammatically correct text
- Maintains the base capabilities of the SFT model
- Sounds like the same model personality

---

## Summary

```
+---------------------------------------------------------------+
|  LESSON 03 SUMMARY                                            |
|                                                               |
|  1. Why RL?                                                   |
|     RL allows the LLM to learn from a reward signal          |
|     (the reward model score) instead of supervised labels.    |
|                                                               |
|  2. LLM as Policy                                             |
|     State = prompt + tokens so far                            |
|     Action = next token                                        |
|     Reward = reward model score at end of sequence            |
|                                                               |
|  3. PPO Overview                                              |
|     Rollout -> Score -> Compute Advantage -> Update Policy    |
|     Repeat thousands of times                                 |
|                                                               |
|  4. Clipping                                                  |
|     Limits how much any single update can change the policy   |
|     Prevents catastrophic updates                             |
|                                                               |
|  5. KL Penalty                                                |
|     Total Reward = RM_score - beta * KL(policy || SFT)       |
|     Prevents the model from drifting too far from SFT         |
|                                                               |
|  6. Challenges                                                |
|     Long sequences, sparse rewards, credit assignment,        |
|     massive action space, catastrophic forgetting             |
+---------------------------------------------------------------+
```

---

## Quiz Questions

1. What does "policy" mean in the context of RLHF? How is the LLM a policy?

2. What are the 4 main steps of the PPO training loop?

3. What does "advantage" mean? Give an example of positive and negative advantage.

4. What does "clipping" prevent in PPO? What would happen without it?

5. Write out the KL-penalized reward formula and explain what each term does.

6. What is "credit assignment" and why is it hard for language models?

7. What is "reward sparsity" and how is it different from dense reward settings like games?

8. If the KL divergence between the policy and the SFT model grows very large during training,
   what does that suggest? What should you do?

---

## Lab Exercise

```python
# lab_03_ppo_concepts.py
# Implement advantage computation and clipped ratio

import numpy as np

# ============================================================
# PART A: Advantage Computation
# ============================================================

def compute_advantage(actual_reward, baseline_value):
    """
    Compute advantage = how much better was this than expected?
    
    Args:
        actual_reward:  the reward the model actually received
        baseline_value: what the critic predicted (baseline)
    
    Returns:
        advantage: float (positive = better than expected, negative = worse)
    """
    # YOUR CODE HERE
    pass


# Test advantage computation
adv_good = compute_advantage(actual_reward=0.9, baseline_value=0.5)
adv_bad  = compute_advantage(actual_reward=0.2, baseline_value=0.5)
adv_neutral = compute_advantage(actual_reward=0.5, baseline_value=0.5)

assert adv_good > 0, "Should be positive advantage (better than expected)"
assert adv_bad  < 0, "Should be negative advantage (worse than expected)"
assert adv_neutral == 0, "Should be zero (exactly as expected)"
print("Advantage tests passed!")

# ============================================================
# PART B: PPO Clipped Ratio
# ============================================================

def ppo_clipped_objective(ratio, advantage, eps=0.2):
    """
    Compute the PPO clipped objective for one (action, advantage) pair.
    
    Formula: min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
    
    Args:
        ratio:     new_prob / old_prob (how much has the policy changed?)
        advantage: advantage estimate for this action
        eps:       clipping parameter (default 0.2)
    
    Returns:
        objective: the clipped PPO objective (we want to MAXIMIZE this)
                   (in loss form, negate it to MINIMIZE)
    """
    # YOUR CODE HERE
    # Step 1: Compute unclipped term = ratio * advantage
    # Step 2: Compute clipped ratio = clip(ratio, 1-eps, 1+eps)
    # Step 3: Compute clipped term = clipped_ratio * advantage
    # Step 4: Return min(unclipped_term, clipped_term)
    pass


# Test PPO clipping
# Large positive advantage, large ratio -- should be clipped
obj1 = ppo_clipped_objective(ratio=1.5, advantage=2.0, eps=0.2)
# ratio=1.5 should be clipped to 1.2 (1 + eps)
expected1 = min(1.5 * 2.0, 1.2 * 2.0)
assert abs(obj1 - expected1) < 1e-6, f"Expected {expected1}, got {obj1}"
print("PPO clipping test passed!")

# Small ratio, positive advantage -- should NOT be clipped
obj2 = ppo_clipped_objective(ratio=1.1, advantage=2.0, eps=0.2)
expected2 = min(1.1 * 2.0, 1.1 * 2.0)  # 1.1 is within [0.8, 1.2]
assert abs(obj2 - expected2) < 1e-6, f"Expected {expected2}, got {obj2}"
print("PPO no-clip test passed!")

print("\nAll PPO tests passed! Great work.")
```

---

*Next lesson: Direct Preference Optimization (DPO) -- a simpler alternative to PPO.*
*File: lessons/04_dpo.md*

# Lesson 07: GRPO — Group Relative Policy Optimization

## Glossary (Read This First!)

| Term | Plain English Definition | C# Analogy |
|------|--------------------------|------------|
| **GRPO** | Group Relative Policy Optimization. An RL algorithm that samples multiple responses to the same prompt, scores them all, then trains using relative (not absolute) scores. | Like A/B testing N implementations, ranking them relatively, training on the ranking. |
| **Policy** | The LLM being trained. Takes a prompt → generates a response. | The strategy class being optimized. |
| **Group** | G responses sampled for the same prompt. GRPO uses this group to compute a relative baseline. | Like running the same benchmark N times to get a distribution, then comparing each run to the average. |
| **Advantage** | How much better (or worse) a response is compared to the group average. `advantage = (reward - mean_reward) / std_reward`. | Like a z-score: how many standard deviations above or below the mean? |
| **PPO** | Proximal Policy Optimization. The RL algorithm used in standard RLHF. Requires a separate value network (critic) to estimate how good a state is. (Lesson 03.) | Like having a separate scoring service that estimates the expected future reward. |
| **Value network (critic)** | In PPO, a second neural network that learns to predict the expected reward for a given state. GRPO eliminates this. | Like a second microservice just for scoring — GRPO removes the need for it. |
| **KL divergence** | A measure of how different two probability distributions are. Used as a penalty to prevent the model from changing too fast (diverging from the reference model). | Like a constraint on how far a refactored method can deviate from the original interface contract. |
| **Reference model** | A frozen copy of the model before RLVR training starts. Used to compute the KL penalty — the trained model shouldn't drift too far from this. | Like keeping the original version of a service running alongside the new one to compare behavior. |
| **Policy gradient** | The mathematical technique that updates model weights in the direction that makes high-reward responses more likely. | Like gradient descent, but the loss is based on rewards, not a fixed target. |
| **Clipping** | Limiting the policy update ratio to a range like [0.8, 1.2] to prevent too-large steps. Same idea as PPO clipping (Lesson 03). | Like a change rate limiter — don't update the model by more than X% in one step. |
| **Log probability** | The log of the probability the model assigns to generating a specific token. Used in the policy gradient formula. | Like `Math.Log(probability)`. |

---

## How It Connects

```
Lesson 03 (PPO)
  -- Sample 1 response per prompt
  -- Compute advantage using value network (critic)
  -- Update policy with clipped policy gradient + KL penalty
  -- Requires 2 models: policy + value network
         |
         | Problem: value network is another large model to train
         |          doubles memory and compute cost
         v
Lesson 07 (GRPO)
  -- Sample G responses per prompt (the "group")
  -- Compute advantage from group mean/std -- NO value network needed
  -- Same clipped policy gradient + KL penalty as PPO
  -- Requires 1 model: just the policy
```

---

## The Problem with PPO for LLMs

PPO (Lesson 03) is the standard algorithm for RLHF. But it has a problem when applied to LLMs:

**PPO requires a value network (critic).**

The value network estimates how "good" the current state is — it predicts the expected future reward.
For LLMs, this means training a SECOND full-size model alongside the policy.

```
PPO memory cost:
  Policy model:   7B params  -->  ~14 GB GPU memory
  Value network:  7B params  -->  ~14 GB GPU memory
  Reference model: 7B params -->  ~14 GB GPU memory
  Total:                          ~42 GB just for models
```

For a 70B parameter model, this becomes 420 GB — impossible on most hardware.

**GRPO's insight: you don't need a value network.**

Instead of learning a value function, GRPO computes a baseline from the group of responses themselves.

---

## How GRPO Works: Step by Step

```
GRPO Training Step:

  1. SAMPLE
     -- Take one prompt from the training set
     -- Generate G responses from the current policy
     -- (G is typically 8 to 16)

     Prompt: "What is the square root of 144?"
     
     Response 1: "12"                 (correct)
     Response 2: "<think>144 = 12*12, so sqrt(144) = 12</think> 12"  (correct + format)
     Response 3: "11"                 (wrong)
     Response 4: "12.0"               (correct)
     Response 5: "I think it might be around 12"  (vague)
     Response 6: "12"                 (correct)
     Response 7: "The answer is 13"   (wrong)
     Response 8: "12"                 (correct)

  2. SCORE
     -- Compute reward for each response using the RLVR checker
     
     r1 = 1.0  (correct)
     r2 = 1.2  (correct + format bonus)
     r3 = 0.0  (wrong)
     r4 = 1.0  (correct)
     r5 = 0.5  (partially correct)
     r6 = 1.0  (correct)
     r7 = 0.0  (wrong)
     r8 = 1.0  (correct)

  3. COMPUTE ADVANTAGE
     -- Group mean:  mean([1.0, 1.2, 0.0, 1.0, 0.5, 1.0, 0.0, 1.0]) = 0.7125
     -- Group std:   std([...]) = 0.424
     
     -- Advantage = (reward - mean) / std    <-- this is a z-score
     
     a1 = (1.0 - 0.7125) / 0.424 = +0.68  (better than average)
     a2 = (1.2 - 0.7125) / 0.424 = +1.15  (much better than average)
     a3 = (0.0 - 0.7125) / 0.424 = -1.68  (much worse than average)
     a4 = (1.0 - 0.7125) / 0.424 = +0.68
     ...

  4. UPDATE POLICY
     -- For responses with POSITIVE advantage: increase probability
     -- For responses with NEGATIVE advantage: decrease probability
     -- Apply clipping and KL penalty (same as PPO)
```

---

## The Advantage Formula

The advantage is a normalized (z-score) version of the reward:

```
advantage_i = (reward_i - mean(rewards)) / (std(rewards) + epsilon)

where:
  reward_i     = score for response i
  mean(rewards) = average score across all G responses in the group
  std(rewards)  = standard deviation of scores in the group
  epsilon       = small constant (1e-8) to prevent division by zero
```

**Why normalize?**

- Raw rewards can be on any scale (0.0 to 100.0, 0 to 1, etc.)
- After normalization, advantages are always centered around 0 with unit variance
- This makes training more stable — the gradient updates are consistently scaled
- Same reason we normalize inputs in neural networks (Lesson 03.5)

**C# analogy:**
```csharp
// Computing advantage is just z-score normalization
double mean = rewards.Average();
double std = Math.Sqrt(rewards.Select(r => Math.Pow(r - mean, 2)).Average());

double[] advantages = rewards.Select(r => (r - mean) / (std + 1e-8)).ToArray();
// Now advantages[i] = how many standard deviations response i is above the group mean
```

---

## The Policy Gradient Update

Once we have advantages, we update the model weights to:
- Make high-advantage responses MORE likely
- Make low-advantage responses LESS likely

The loss function (what we minimize):

```
GRPO_loss = -mean over all tokens in all G responses of:
  min(
    ratio * advantage,
    clip(ratio, 1-epsilon, 1+epsilon) * advantage
  )
  + beta * KL_divergence(current_policy, reference_policy)

where:
  ratio     = prob(token | current_policy) / prob(token | old_policy)
  epsilon   = clipping range (typically 0.2)
  beta      = KL penalty coefficient (typically 0.01)
```

This is nearly identical to PPO's loss function — the key difference is how `advantage` was computed (group relative, not value network).

```
PPO:   advantage = reward - value_network_estimate(state)
GRPO:  advantage = (reward - group_mean) / group_std
        ^
        No value network! Just math on the group rewards.
```

---

## Why GRPO Is Better Than PPO for LLMs

| Aspect | PPO | GRPO |
|--------|-----|------|
| Value network needed? | Yes (2nd model) | No |
| Memory cost | 2x (policy + critic) | 1x (policy only) |
| Training stability | Good | Good (z-score normalization helps) |
| Baseline quality | Learned (can be wrong) | Group mean (always correct) |
| Implementation complexity | High | Lower |
| Used by | Original RLHF (ChatGPT) | DeepSeek-R1, Qwen3, Llama4 |

---

## KL Divergence Penalty: Why It's Needed

Without the KL penalty, the model can drift too far from the original pretrained model.

```
Reference model (frozen):    knows how to speak English, knows facts, etc.
Trained model (GRPO):        being tuned to answer math correctly

Without KL penalty:
  After 10,000 GRPO steps, the model might only output math answers
  and forget how to write normal sentences.

With KL penalty:
  The model stays close to the reference.
  It improves at math WITHOUT forgetting everything else.
```

The KL penalty is a "leash" that keeps the trained model close to the reference.

```
total_loss = policy_gradient_loss + beta * KL(current || reference)
                                           ^
                                           beta is small (0.01)
                                           so it's a soft constraint, not a hard one
```

---

## Full GRPO Algorithm Summary

```
GRPO Training Loop:
===================

Initialize:
  policy_model    = pretrained LLM
  reference_model = frozen copy of policy_model
  reward_fn       = verifiable checker (math answer, code tests, etc.)
  G               = group size (e.g. 8)

For each training step:
  1. Sample prompt p from dataset
  2. Generate G responses: [r_1, ..., r_G] from policy_model(p)
  3. Score each:    rewards = [reward_fn(r_1), ..., reward_fn(r_G)]
  4. Normalize:     advantages = (rewards - mean(rewards)) / (std(rewards) + eps)
  5. Compute loss:
       for each response r_i:
         for each token t in r_i:
           ratio = policy_prob(t) / old_policy_prob(t)
           policy_loss += -min(ratio * adv_i, clip(ratio, 0.8, 1.2) * adv_i)
       kl_loss = KL(policy_model || reference_model)
       total_loss = policy_loss + beta * kl_loss
  6. Backpropagate total_loss
  7. Update policy_model weights

After N steps:
  policy_model is better at the verifiable task
  AND still close to reference_model (KL penalty kept it grounded)
```

---

## C# Analogy: Relative Benchmarking

Imagine you're testing N implementations of the same algorithm:

```csharp
// GRPO is like running N implementations, scoring them relatively
var implementations = new List<Func<int, int>>
{
    x => x * x,        // correct
    x => x + x,        // wrong (adds instead of squares)
    x => (int)Math.Pow(x, 2),  // correct
    // ...8 total implementations
};

var scores = implementations.Select(f => RunTestSuite(f)).ToArray();
var mean = scores.Average();
var std = Math.Sqrt(scores.Select(s => Math.Pow(s - mean, 2)).Average());

// Advantage = how much better/worse than the group average
var advantages = scores.Select(s => (s - mean) / (std + 1e-8)).ToArray();

// Train the "model" (code generator) to produce more implementations like the high-advantage ones
```

The key insight: you don't need an absolute score, just a relative ranking within the group.

---

## Key Takeaways

1. **GRPO eliminates the value network** — it computes the advantage baseline from the group mean instead
2. **Group size G** (typically 8-16) responses are sampled per prompt
3. **Advantage = z-score** of the reward: `(reward - mean) / std`
4. **Loss function** is identical to PPO except for how advantage is computed
5. **KL penalty** keeps the trained model from forgetting the pretrained knowledge
6. **Memory cost** is ~half of PPO — only one model (policy) instead of two (policy + critic)
7. **DeepSeek-R1 used GRPO** with math/code verifiable rewards to achieve reasoning

---

## Quick Self-Check

1. What is the main difference between PPO and GRPO?
2. How is the advantage computed in GRPO?
3. Why do we divide by the standard deviation when computing the advantage?
4. What does the KL penalty prevent?
5. If a group of 8 responses has rewards [1, 1, 0, 1, 0, 1, 1, 0], what is the mean reward and the advantage for a response with reward = 1?

*(Answers in example_07_grpo.py)*

"""
Module 13 - RLHF and Alignment: Example 01
============================================
TOPIC: The 3-Phase RLHF Pipeline

=== WHAT IS RLHF? ===

RLHF stands for Reinforcement Learning from Human Feedback.
It is the technique used to turn a raw pretrained LLM (which can
generate anything, including harmful content) into a helpful,
harmless, honest assistant like ChatGPT or Claude.

C# analogy: Think of a pretrained LLM as a very powerful but
unfiltered code generator. RLHF is like adding:
  - Phase 1: A code style ruleset (SFT fine-tuning)
  - Phase 2: An automated code reviewer (Reward Model)
  - Phase 3: An optimizer that maximizes reviewer scores (RL loop)

=== GLOSSARY ===

SFT (Supervised Fine-Tuning):
    Phase 1 of RLHF. Train the model on human-written "ideal" responses.
    Standard supervised learning: given input, predict correct output.
    C# analogy: Like training a model on manually-labelled examples
    in a supervised ML pipeline (e.g., Azure ML with labelled data).

Reward Model (RM):
    Phase 2 of RLHF. A separate neural network trained to predict
    how much a human would like a given response. Outputs a scalar score.
    C# analogy: Like a code quality analyzer (SonarQube) that returns
    a float "quality score" for any piece of code.

Policy:
    In RL, the "policy" is the model being trained. It decides WHAT to
    output given an input. In RLHF the policy = the language model.
    C# analogy: Like a strategy pattern implementation that chooses
    which action to take given the current state.

Policy Gradient:
    An RL algorithm that updates the policy to increase the probability
    of actions that received high rewards.
    C# analogy: Like adjusting a recommendation engine to recommend
    more items similar to those the user rated 5 stars.

Log probability (log_prob):
    The natural log of the probability that the policy assigns to a
    particular output. Used in the policy gradient formula.
    C# analogy: Math.Log(probability) -- we use logs for numerical
    stability (avoids multiplying many tiny floats together).

Preference pair:
    A training example for the reward model: two responses where a
    human says "Response A is better than Response B."
    C# analogy: Like IComparable<T>.CompareTo() telling you which of
    two objects ranks higher.

Bradley-Terry model:
    The statistical model behind reward model training.
    It says: P(A > B) = sigmoid(score_A - score_B)
    C# analogy: Like an Elo rating system (used in chess/games) where
    the probability of winning depends on the score difference.

PPO (Proximal Policy Optimization):
    The RL algorithm most commonly used in RLHF. It keeps the policy
    from changing too much in one step ("proximal" = nearby).
    We simulate a simplified version here (no clipping).
    C# analogy: Like a cautious weight update with a max-delta clamp.

KL Divergence penalty:
    A term added to the RL loss to prevent the policy from drifting
    too far from the original SFT model. Keeps safety guardrails intact.
    C# analogy: Like a constraint in an optimization problem that says
    "new solution must not be more than X% different from the old one."

=== WHAT YOU WILL SEE ===

PART A (NumPy):
    Phase 1 (SFT):   Weight matrix maps tokens -> responses. We fine-tune it.
    Phase 2 (RM):    Linear model learns to score good vs bad responses.
    Phase 3 (RL):    Policy gradient update using reward * log_prob.

PART B (PyTorch):
    Same 3 phases using nn.Linear, nn.Sigmoid, and manual policy gradient.

Run with: python example_01_rlhf_concepts.py
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np   # NumPy: arrays and math (no external datasets needed)
import torch         # PyTorch: deep learning tensors
import torch.nn as nn  # nn: pre-built layer classes (Linear, Sigmoid, etc.)

# ============================================================
# ASCII DIAGRAM: THE RLHF PIPELINE
# ============================================================
#
#  RLHF PIPELINE
#  =============
#  [Pretrained LLM]
#        |
#     Phase 1: SFT (Supervised Fine-Tuning)
#        | Train on human-written good responses
#        v
#  [SFT Model]
#        |
#     Phase 2: Reward Model Training
#        | Humans rank: Response A > Response B
#        | Train RM to predict which is better
#        v
#  [Reward Model]
#        |
#     Phase 3: RL (PPO)
#        | Use RM as reward signal
#        | Update policy to maximize reward
#        v
#  [Aligned Model]
#
# ============================================================

print("=" * 65)
print("MODULE 13 - RLHF AND ALIGNMENT: EXAMPLE 01")
print("The 3-Phase RLHF Pipeline")
print("=" * 65)
print()

# ============================================================
# SHARED SETUP: TOY VOCABULARY
# ============================================================
# We use a vocabulary of 8 words (tokens).
# In a real LLM the vocabulary has 50,000+ tokens.
# C# analogy: like an enum with 8 values.

VOCAB = {                     # dict mapping word -> integer ID
    "hello"    : 0,           # token 0
    "world"    : 1,           # token 1
    "help"     : 2,           # token 2
    "sorry"    : 3,           # token 3
    "I"        : 4,           # token 4
    "cannot"   : 5,           # token 5
    "sure"     : 6,           # token 6
    "harmful"  : 7,           # token 7
}

VOCAB_SIZE = len(VOCAB)       # 8 tokens total
VOCAB_INV = {v: k for k, v in VOCAB.items()}  # reverse dict: ID -> word

# We represent a "response" as a one-hot vector of length VOCAB_SIZE.
# One-hot = all zeros except a single 1 at the token's position.
# C# analogy: bool[] where exactly one element is true.

def one_hot(token_id, size=VOCAB_SIZE):
    """
    Converts a single integer token ID into a one-hot vector.
    WHY: Neural networks need fixed-size numeric input, not raw integers.
    C# analogy: Like BitArray with exactly one bit set to true.
    """
    vec = np.zeros(size, dtype=np.float32)  # create array of zeros
    vec[token_id] = 1.0                      # set the correct position to 1
    return vec                               # return the one-hot vector

# ============================================================
# ============================================================
# PART A: RLHF PIPELINE WITH PURE NUMPY
# ============================================================
# ============================================================

print("=" * 65)
print("PART A: NumPy Implementation (3 Phases)")
print("=" * 65)
print()

# ============================================================
# A - PHASE 1: SFT (Supervised Fine-Tuning)
# ============================================================
# A tiny language model: given a prompt token (one-hot, size 8),
# produce a response token probability distribution (size 8).
# Architecture: Linear layer (8 -> 8) + Softmax.
#
# We "pre-train" by setting weights randomly, then fine-tune
# by doing one supervised gradient step on a good example.
# ============================================================

print("--- PHASE 1: SFT (Supervised Fine-Tuning) ---")
print()

np.random.seed(7)  # fixed seed for reproducibility (same result every run)

# Weight matrix W_sft: shape (VOCAB_SIZE, VOCAB_SIZE) = (8, 8)
# Maps one prompt token vector -> one response logit vector.
# C# analogy: float[8, 8] W_sft = InitRandomMatrix();
W_sft = np.random.randn(VOCAB_SIZE, VOCAB_SIZE) * 0.1  # small random weights

# Bias vector for the SFT model: shape (8,)
b_sft = np.zeros(VOCAB_SIZE, dtype=np.float32)          # start biases at zero

print(f"SFT weight matrix shape: {W_sft.shape}  (8 prompt tokens -> 8 response tokens)")
print(f"W_sft first row (before SFT): {W_sft[0].round(4)}")
print()

def softmax_1d(z):
    """
    Softmax for a 1D vector. Converts raw scores to probabilities.
    WHY: We need probabilities (sum=1) to interpret as token likelihoods.
    C# analogy: Normalize an array so elements sum to 1.0.
    """
    z_stable = z - np.max(z)           # subtract max for numerical stability
    exp_z = np.exp(z_stable)           # e^z for each element
    return exp_z / np.sum(exp_z)       # divide by total to get probabilities

def sft_forward(prompt_token_id):
    """
    Forward pass for the SFT model.
    WHY: Given a prompt token, produce a probability distribution over responses.
    Returns logits (raw scores) and probs (softmax probabilities).
    """
    x = one_hot(prompt_token_id)        # convert token ID to one-hot vector
    logits = W_sft @ x + b_sft         # matrix multiply + bias (@ = dot product)
    probs = softmax_1d(logits)          # convert to probabilities
    return logits, probs                # return both for use in training

# --- Simulate a pretrained model prediction (before SFT) ---
# Prompt: "hello" (token 0). Ideal response: "world" (token 1).
prompt_id = VOCAB["hello"]             # prompt token ID = 0
ideal_response_id = VOCAB["world"]     # ideal response token ID = 1

logits_before, probs_before = sft_forward(prompt_id)  # run forward pass

print(f"Prompt token : '{list(VOCAB.keys())[prompt_id]}' (ID={prompt_id})")
print(f"Ideal response: '{list(VOCAB.keys())[ideal_response_id]}' (ID={ideal_response_id})")
print()
print(f"Probabilities BEFORE SFT fine-tuning:")
for token_id, prob in enumerate(probs_before):     # loop over all 8 tokens
    bar = "=" * int(prob * 40)                      # ASCII bar scaled to 40 chars
    marker = " <-- ideal" if token_id == ideal_response_id else ""  # mark ideal
    print(f"  '{VOCAB_INV[token_id]:8s}' (ID={token_id}): {prob:.4f} |{bar}{marker}")
print()

# --- SFT Gradient Step ---
# Supervised cross-entropy loss: -log(prob[ideal_response])
# Gradient of loss w.r.t. logits: (probs - one_hot(ideal_response))
# Weight update: W -= lr * grad_outer_product

LR_SFT = 0.5                                       # learning rate for SFT step

# Compute gradient of the loss w.r.t. logits (the well-known softmax CE gradient)
grad_logits = probs_before.copy()                  # start with predicted probs
grad_logits[ideal_response_id] -= 1.0              # subtract 1 at correct position

# Gradient w.r.t. W: outer product of gradient and input x
x_input = one_hot(prompt_id)                       # the input vector (one-hot)
grad_W = np.outer(grad_logits, x_input)            # outer product: shape (8, 8)

# Gradient w.r.t. b: same as grad_logits (bias gradient is just the output gradient)
grad_b = grad_logits.copy()                        # shape (8,)

# Apply gradient descent update
W_sft -= LR_SFT * grad_W                           # update weight matrix
b_sft -= LR_SFT * grad_b                           # update bias vector

# Check probabilities AFTER one SFT step
logits_after, probs_after = sft_forward(prompt_id)  # run forward pass again

print(f"Probabilities AFTER SFT fine-tuning (1 gradient step):")
for token_id, prob in enumerate(probs_after):      # loop over all 8 tokens
    bar = "=" * int(prob * 40)                      # ASCII bar
    marker = " <-- ideal" if token_id == ideal_response_id else ""
    print(f"  '{VOCAB_INV[token_id]:8s}' (ID={token_id}): {prob:.4f} |{bar}{marker}")
print()

# Show how much the ideal response probability increased
improvement = probs_after[ideal_response_id] - probs_before[ideal_response_id]
print(f"Ideal response probability: {probs_before[ideal_response_id]:.4f} -> {probs_after[ideal_response_id]:.4f}  (+ {improvement:.4f})")
print()
print("KEY INSIGHT: After SFT, the model assigns higher probability")
print("to the human-approved response. This is supervised learning.")
print()

# ============================================================
# A - PHASE 2: REWARD MODEL TRAINING
# ============================================================
# The reward model takes a response vector (one-hot, size 8) and
# outputs a scalar reward score via: score = sigmoid(w_rm . response + b_rm)
# We train it on preference pairs: "good" response should score > "bad".
# Loss: -log(sigmoid(score_good - score_bad))  [Bradley-Terry]
# ============================================================

print("--- PHASE 2: Reward Model Training ---")
print()

# Initialize reward model weights: shape (VOCAB_SIZE,) = (8,)
# Just one weight per vocabulary token (simple linear model).
# C# analogy: float[] w_rm = new float[8];
np.random.seed(3)                                  # new seed for RM weights
w_rm = np.random.randn(VOCAB_SIZE) * 0.1           # small random weights
b_rm = 0.0                                         # single scalar bias

def sigmoid(x):
    """
    Sigmoid function: maps any real number to (0, 1).
    WHY: We use sigmoid to turn a raw score into a probability-like value.
    C# analogy: 1.0 / (1.0 + Math.Exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))               # standard sigmoid formula

def reward_score(response_vec):
    """
    Computes a scalar reward score for a response vector.
    WHY: The RL phase needs a single number to optimize toward.
    Higher score = model thinks the response is better.
    """
    raw = np.dot(w_rm, response_vec) + b_rm        # linear combination
    return sigmoid(raw)                             # squeeze into (0, 1)

# Define preference pairs: (good_token_id, bad_token_id)
# Humans prefer "sure" (helpful) and "help" over "harmful" and "cannot".
# C# analogy: List<(int Good, int Bad)> preferencePairs
preference_pairs_rm = [
    (VOCAB["sure"],  VOCAB["harmful"]),   # pair 0: "sure" > "harmful"
    (VOCAB["help"],  VOCAB["harmful"]),   # pair 1: "help" > "harmful"
    (VOCAB["hello"], VOCAB["cannot"]),    # pair 2: "hello" > "cannot"
    (VOCAB["world"], VOCAB["harmful"]),   # pair 3: "world" > "harmful"
    (VOCAB["sure"],  VOCAB["cannot"]),    # pair 4: "sure" > "cannot"
    (VOCAB["help"],  VOCAB["cannot"]),    # pair 5: "help" > "cannot"
]

print(f"Reward model weights BEFORE training: {w_rm.round(4)}")
print(f"Training on {len(preference_pairs_rm)} preference pairs...")
print()

LR_RM = 0.8                                       # learning rate for reward model
NUM_EPOCHS_RM = 30                                 # number of training epochs

print(f"  {'Epoch':>5}  |  {'Loss':>8}  |  Pref. Accuracy")
print(f"  {'-'*5}  |  {'-'*8}  |  {'-'*20}")

for epoch in range(NUM_EPOCHS_RM):                 # training loop over epochs
    total_loss = 0.0                               # accumulate loss for this epoch

    for good_id, bad_id in preference_pairs_rm:    # loop over each preference pair
        good_vec = one_hot(good_id)                # one-hot vector for good response
        bad_vec  = one_hot(bad_id)                 # one-hot vector for bad response

        # Forward pass: compute raw scores (before sigmoid)
        score_good_raw = np.dot(w_rm, good_vec) + b_rm   # raw score for good response
        score_bad_raw  = np.dot(w_rm, bad_vec)  + b_rm   # raw score for bad response

        # Bradley-Terry loss: -log(sigmoid(score_good - score_bad))
        # We want score_good > score_bad, so their difference should be large & positive.
        diff = score_good_raw - score_bad_raw      # positive = good is ranked higher
        prob_correct = sigmoid(diff)               # probability model prefers good over bad
        loss_pair = -np.log(prob_correct + 1e-9)   # cross-entropy; +1e-9 avoids log(0)

        total_loss += loss_pair                    # add this pair's loss to total

        # Gradient of loss w.r.t. the score difference:
        # d(-log(sigmoid(d)))/d(d) = sigmoid(d) - 1 = -(1 - sigmoid(d))
        grad_diff = prob_correct - 1.0             # gradient w.r.t. diff

        # Chain rule: gradient flows through diff = score_good_raw - score_bad_raw
        grad_good_raw =  grad_diff                 # d(diff)/d(score_good) = +1
        grad_bad_raw  = -grad_diff                 # d(diff)/d(score_bad)  = -1

        # Gradient w.r.t. w_rm:
        # score_raw = w_rm . x, so d(score_raw)/d(w_rm) = x
        grad_w_from_good = grad_good_raw * good_vec   # contribution from good pair
        grad_w_from_bad  = grad_bad_raw  * bad_vec    # contribution from bad pair

        # Update weights and bias
        w_rm -= LR_RM * (grad_w_from_good + grad_w_from_bad)  # weight update
        b_rm -= LR_RM * (grad_good_raw + grad_bad_raw)         # bias update

    avg_loss = total_loss / len(preference_pairs_rm)  # average loss for this epoch

    # Compute preference accuracy: how many pairs does RM rank correctly?
    correct = 0                                    # count of correctly ranked pairs
    for good_id, bad_id in preference_pairs_rm:    # loop over pairs again
        s_good = reward_score(one_hot(good_id))    # score for good response
        s_bad  = reward_score(one_hot(bad_id))     # score for bad response
        if s_good > s_bad:                         # correct if good scores higher
            correct += 1
    pref_acc = correct / len(preference_pairs_rm) * 100  # percentage accuracy

    if (epoch + 1) % 10 == 0:                     # print every 10 epochs
        print(f"  {epoch+1:>5}  |  {avg_loss:>8.4f}  |  {pref_acc:.0f}%")

print()
print(f"Reward model weights AFTER training: {w_rm.round(4)}")
print()

# Show final reward scores for each token
print("Final reward scores for each token:")
for token_id in range(VOCAB_SIZE):                 # loop over all 8 vocabulary tokens
    vec = one_hot(token_id)                        # get one-hot vector
    score = reward_score(vec)                      # compute reward score
    bar = "=" * int(score * 30)                    # ASCII bar scaled to 30 chars
    print(f"  '{VOCAB_INV[token_id]:8s}' (ID={token_id}): score={score:.4f} |{bar}")
print()
print("KEY INSIGHT: Helpful tokens (sure, help) have HIGH reward scores.")
print("Harmful/refusal tokens (harmful, cannot) have LOW reward scores.")
print()

# ============================================================
# A - PHASE 3: RL (Policy Gradient Update)
# ============================================================
# We use the reward model scores to update the SFT policy.
# Algorithm (simplified REINFORCE):
#   1. Sample a response token from the policy distribution.
#   2. Get reward from the reward model.
#   3. Compute gradient: reward * d(log_prob)/d(W)
#   4. Update W_sft to increase probability of high-reward responses.
#
# For simplicity we do this for all 8 possible responses and weight
# each gradient contribution by its reward score.
# ============================================================

print("--- PHASE 3: RL (Policy Gradient) ---")
print()

# We run policy gradient for the same prompt: "hello" (token 0).
prompt_id = VOCAB["hello"]                         # prompt token ID
x_prompt  = one_hot(prompt_id)                    # one-hot prompt vector

print(f"Running RL for prompt: 'hello' (ID={prompt_id})")
print()

LR_RL = 0.3                                        # learning rate for RL step
NUM_RL_STEPS = 15                                  # number of RL update steps

# Save weights before RL so we can compare
W_sft_before_rl = W_sft.copy()                    # snapshot of SFT weights

print(f"  {'Step':>4}  |  {'AvgReward':>10}  |  Score for 'sure'")
print(f"  {'-'*4}  |  {'-'*10}  |  {'-'*20}")

for step in range(NUM_RL_STEPS):                   # RL update loop

    # Forward pass: compute current policy probabilities
    logits_rl, probs_rl = sft_forward(prompt_id)  # current probs over 8 tokens

    # Compute rewards for all 8 possible response tokens
    rewards = np.array([                           # array of reward scores
        reward_score(one_hot(tok_id))              # reward for each possible response
        for tok_id in range(VOCAB_SIZE)
    ], dtype=np.float32)                           # shape: (8,)

    # Baseline: subtract mean reward to reduce variance (standard REINFORCE trick)
    # C# analogy: Normalize rewards by subtracting the mean (center them around 0).
    baseline = np.mean(rewards)                    # mean reward across all tokens
    advantages = rewards - baseline                # centered advantage estimates

    # Policy gradient: accumulate gradient weighted by advantages
    grad_W_rl = np.zeros_like(W_sft)              # gradient matrix, starts at zero
    grad_b_rl = np.zeros_like(b_sft)              # gradient bias vector

    for tok_id in range(VOCAB_SIZE):               # loop over all response tokens
        adv = advantages[tok_id]                   # advantage for this response token

        # Gradient of log_prob[tok_id] w.r.t. logits:
        # d(log p_k)/d(z_j) = (1 if j==k else 0) - p_j
        # This is the gradient of log-softmax.
        grad_log_prob_wrt_logits = -probs_rl.copy()   # start with -p for all j
        grad_log_prob_wrt_logits[tok_id] += 1.0        # add 1 at position k

        # Policy gradient theorem: gradient = advantage * d(log_prob)/d(theta)
        # Here theta = W_sft, and d(logits)/d(W) uses the outer product.
        pg_grad = adv * np.outer(grad_log_prob_wrt_logits, x_prompt)  # (8,8)
        grad_W_rl += pg_grad                       # accumulate gradient

        # Gradient for bias: same as logits gradient (bias adds directly to logits)
        grad_b_rl += adv * grad_log_prob_wrt_logits  # accumulate bias gradient

    # Policy gradient MAXIMIZES reward, so we SUBTRACT the negative gradient
    # (equivalent to ascending the reward gradient).
    # C# analogy: weights += learningRate * gradient (gradient ASCENT for reward)
    W_sft += LR_RL * grad_W_rl                    # weight ascent step
    b_sft += LR_RL * grad_b_rl                    # bias ascent step

    # Compute average reward for reporting
    avg_reward = np.dot(probs_rl, rewards)         # expected reward under policy

    # Get current probability of generating "sure" (helpful token)
    _, probs_check = sft_forward(prompt_id)        # recompute after update
    sure_prob = probs_check[VOCAB["sure"]]         # probability for "sure"

    if (step + 1) % 5 == 0:                       # print every 5 steps
        print(f"  {step+1:>4}  |  {avg_reward:>10.4f}  |  {sure_prob:.4f}")

print()

# Final distribution after RL
_, probs_final_rl = sft_forward(prompt_id)        # final policy probabilities

print("Policy token probabilities AFTER RL training:")
for token_id, prob in enumerate(probs_final_rl):  # loop over all 8 tokens
    bar = "=" * int(prob * 40)                     # ASCII bar
    rew = reward_score(one_hot(token_id))           # show reward for reference
    print(f"  '{VOCAB_INV[token_id]:8s}': prob={prob:.4f}  reward={rew:.3f} |{bar}")
print()

# Compare probability of "sure" before and after RL
prob_sure_before_rl = probs_after[VOCAB["sure"]]  # probs from after SFT
prob_sure_after_rl  = probs_final_rl[VOCAB["sure"]]  # probs from after RL
print(f"Probability of 'sure' (helpful): {prob_sure_before_rl:.4f} -> {prob_sure_after_rl:.4f}")
print()
print("KEY INSIGHT: RL pushes the policy to generate tokens the reward")
print("model scores highly. Helpful tokens get higher probability.")
print("This is how ChatGPT/Claude learns to be helpful and safe.")
print()

# ============================================================
# ============================================================
# PART B: RLHF PIPELINE WITH PYTORCH
# ============================================================
# ============================================================

print("=" * 65)
print("PART B: PyTorch Implementation (3 Phases)")
print("=" * 65)
print()

# ============================================================
# B - PHASE 1: SFT WITH nn.Linear
# ============================================================

print("--- PHASE 1: SFT with PyTorch ---")
print()

# Define the SFT model as an nn.Module class.
# C# analogy: class SFTModel : NeuralNetworkBase { ... }
class SFTModel(nn.Module):
    """
    SFT language model: maps a one-hot prompt vector to response logits.
    WHY: nn.Module gives us automatic gradient tracking and parameter management.
    C# analogy: A class inheriting from a base that handles backprop bookkeeping.
    """

    def __init__(self, vocab_size):
        """
        Constructor: define the single linear layer.
        vocab_size is the size of our toy vocabulary (8).
        """
        super().__init__()                          # call parent nn.Module constructor
        self.linear = nn.Linear(vocab_size, vocab_size, bias=True)  # W: (8,8), b: (8,)

    def forward(self, x):
        """
        Forward pass: apply the linear layer to the input.
        x shape: (batch_size, vocab_size) -- here batch_size=1.
        Returns logits (raw scores, not yet softmax'd).
        """
        return self.linear(x)                      # apply linear transform: x @ W.T + b

# Create model instance with our vocabulary size
sft_model = SFTModel(vocab_size=VOCAB_SIZE)        # instantiate the model

# Convert SFT training data to PyTorch tensors
# Prompt: "hello" -> one-hot -> torch tensor, shape (1, 8)
prompt_tensor = torch.tensor(                      # torch.tensor converts numpy -> tensor
    one_hot(VOCAB["hello"]).reshape(1, -1),        # reshape to (1, 8) for batch dimension
    dtype=torch.float32                            # use float32 (standard for neural nets)
)

# Target: "world" -> integer class index tensor
target_tensor = torch.tensor(                      # create tensor from Python int
    [VOCAB["world"]],                              # list with one element (batch size = 1)
    dtype=torch.long                               # long = int64, required by CrossEntropyLoss
)

# Loss function: CrossEntropyLoss = Softmax + NegativeLogLikelihood in one step
# C# analogy: Like a pre-built error metric that handles softmax internally.
sft_criterion = nn.CrossEntropyLoss()

# Optimizer: Adam with a moderate learning rate
# C# analogy: Like an IOptimizer that adjusts learning rate per parameter.
sft_optimizer = torch.optim.Adam(sft_model.parameters(), lr=0.3)

# Get probabilities before SFT fine-tuning (for comparison later)
with torch.no_grad():                              # no_grad = don't track gradients (inference only)
    logits_pt_before = sft_model(prompt_tensor)    # forward pass
    probs_pt_before = torch.softmax(logits_pt_before, dim=1)  # softmax over vocab dim

print("SFT probabilities BEFORE training:")
for token_id in range(VOCAB_SIZE):                 # loop over all 8 tokens
    p = probs_pt_before[0, token_id].item()        # .item() converts tensor scalar to Python float
    bar = "=" * int(p * 40)                        # ASCII bar
    marker = " <-- target" if token_id == VOCAB["world"] else ""
    print(f"  '{VOCAB_INV[token_id]:8s}': {p:.4f} |{bar}{marker}")
print()

# SFT Training loop: 25 epochs
print("Training SFT model (25 epochs)...")
print()

for epoch in range(25):                            # standard training loop
    sft_optimizer.zero_grad()                      # Step 1: clear gradients
    logits_pt = sft_model(prompt_tensor)           # Step 2: forward pass
    loss_pt = sft_criterion(logits_pt, target_tensor)  # Step 3: compute loss
    loss_pt.backward()                             # Step 4: compute gradients
    sft_optimizer.step()                           # Step 5: update weights

    if (epoch + 1) % 5 == 0:                      # print every 5 epochs
        print(f"  Epoch {epoch+1:2d}/25  |  SFT Loss: {loss_pt.item():.4f}")

print()

# Get probabilities after SFT
with torch.no_grad():                              # inference mode (no gradient tracking)
    logits_pt_after = sft_model(prompt_tensor)     # forward pass
    probs_pt_after = torch.softmax(logits_pt_after, dim=1)  # softmax

print("SFT probabilities AFTER training:")
for token_id in range(VOCAB_SIZE):                 # loop over all 8 tokens
    p = probs_pt_after[0, token_id].item()         # get Python float from tensor
    bar = "=" * int(p * 40)                        # ASCII bar
    marker = " <-- target" if token_id == VOCAB["world"] else ""
    print(f"  '{VOCAB_INV[token_id]:8s}': {p:.4f} |{bar}{marker}")
print()

# ============================================================
# B - PHASE 2: REWARD MODEL WITH nn.Module
# ============================================================

print("--- PHASE 2: Reward Model with PyTorch ---")
print()

# Define reward model using nn.Module.
# Architecture: Linear(8, 1) -> Sigmoid -> scalar score.
# C# analogy: class RewardModel : NeuralNetworkBase with a single output neuron.
class RewardModel(nn.Module):
    """
    Reward model: maps a response one-hot vector to a scalar score in (0, 1).
    WHY: We need a differentiable scorer so we can backpropagate through it
    during reward model training (and later use its output as the RL reward).
    """

    def __init__(self, vocab_size):
        """
        Constructor: define a linear layer mapping vocab_size -> 1.
        """
        super().__init__()                         # call parent nn.Module constructor
        self.linear  = nn.Linear(vocab_size, 1)   # W: (1, 8), b: (1,)
        self.sigmoid = nn.Sigmoid()               # squashes output to (0, 1)

    def forward(self, x):
        """
        Forward pass: linear transform then sigmoid activation.
        x shape: (batch_size, vocab_size).
        Returns scalar score in (0, 1). Shape: (batch_size, 1).
        """
        out = self.linear(x)                      # linear transform: (batch, 1)
        out = self.sigmoid(out)                   # sigmoid: squash to (0, 1)
        return out                                # return scalar-per-sample

# Create the reward model
reward_model = RewardModel(vocab_size=VOCAB_SIZE) # instantiate reward model

# Optimizer for reward model
rm_optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.5)

# Reuse the same preference pairs from Part A
# Convert pairs to PyTorch tensors
good_tensors = [                                  # list of one-hot tensors for good responses
    torch.tensor(one_hot(good_id).reshape(1, -1), dtype=torch.float32)
    for good_id, _ in preference_pairs_rm         # unpack tuple: (good_id, bad_id)
]

bad_tensors = [                                   # list of one-hot tensors for bad responses
    torch.tensor(one_hot(bad_id).reshape(1, -1), dtype=torch.float32)
    for _, bad_id in preference_pairs_rm          # unpack tuple: (good_id, bad_id)
]

print(f"Training reward model on {len(preference_pairs_rm)} preference pairs (30 epochs)...")
print()
print(f"  {'Epoch':>5}  |  {'BT Loss':>8}  |  Pref. Accuracy")
print(f"  {'-'*5}  |  {'-'*8}  |  {'-'*20}")

for epoch in range(30):                           # RM training loop
    epoch_loss = 0.0                              # accumulate loss

    for good_t, bad_t in zip(good_tensors, bad_tensors):  # zip pairs good and bad tensors
        rm_optimizer.zero_grad()                  # clear gradients for this pair

        score_good = reward_model(good_t)         # score for good response, shape (1, 1)
        score_bad  = reward_model(bad_t)          # score for bad response,  shape (1, 1)

        # Bradley-Terry loss: -log(sigmoid(score_good - score_bad))
        # We want score_good > score_bad.
        diff = score_good - score_bad              # positive means good is ranked higher
        # sigmoid(diff) = probability model prefers good over bad
        # loss = -log(sigmoid(diff)) = binary cross-entropy w/ target=1
        loss_bt = -torch.log(torch.sigmoid(diff) + 1e-9)  # +1e-9 avoids log(0)

        loss_bt.backward()                        # compute gradients
        rm_optimizer.step()                       # update weights

        epoch_loss += loss_bt.item()              # accumulate scalar loss value

    avg_loss = epoch_loss / len(preference_pairs_rm)  # average over pairs

    # Compute preference accuracy
    correct_pt = 0                                # count correctly ranked pairs
    with torch.no_grad():                         # no gradient tracking for evaluation
        for good_t, bad_t in zip(good_tensors, bad_tensors):
            s_good = reward_model(good_t).item()  # scalar score for good
            s_bad  = reward_model(bad_t).item()   # scalar score for bad
            if s_good > s_bad:                    # correct if good scores higher
                correct_pt += 1
    pref_acc_pt = correct_pt / len(preference_pairs_rm) * 100  # percentage

    if (epoch + 1) % 10 == 0:                    # print every 10 epochs
        print(f"  {epoch+1:>5}  |  {avg_loss:>8.4f}  |  {pref_acc_pt:.0f}%")

print()

# Show final reward scores for each token
print("Final PyTorch reward scores per token:")
with torch.no_grad():                             # inference mode
    for token_id in range(VOCAB_SIZE):            # loop over all vocabulary tokens
        vec_t = torch.tensor(                     # convert one-hot to tensor
            one_hot(token_id).reshape(1, -1), dtype=torch.float32
        )
        score_pt = reward_model(vec_t).item()     # get scalar score
        bar = "=" * int(score_pt * 30)            # ASCII bar
        print(f"  '{VOCAB_INV[token_id]:8s}': score={score_pt:.4f} |{bar}")

print()

# ============================================================
# B - PHASE 3: RL (Policy Gradient) WITH PYTORCH
# ============================================================

print("--- PHASE 3: RL Policy Gradient with PyTorch ---")
print()

# Optimizer for the SFT model in the RL phase
# We reuse the already-trained sft_model and continue training with RL.
rl_optimizer = torch.optim.Adam(sft_model.parameters(), lr=0.05)

print("Running RL updates (20 steps)...")
print()
print(f"  {'Step':>4}  |  {'RL Loss':>8}  |  P('sure') prob")
print(f"  {'-'*4}  |  {'-'*8}  |  {'-'*20}")

for step in range(20):                            # RL update loop
    rl_optimizer.zero_grad()                      # clear gradients

    # Get current policy probabilities (with gradient tracking)
    logits_rl_pt = sft_model(prompt_tensor)       # forward pass: logits (1, 8)
    probs_rl_pt  = torch.softmax(logits_rl_pt, dim=1)  # softmax: probabilities (1, 8)

    # Compute rewards for all 8 tokens using the reward model (no gradient through RM)
    rewards_pt = torch.zeros(VOCAB_SIZE)          # reward tensor for all tokens
    with torch.no_grad():                         # don't track gradients in reward model
        for tok_id in range(VOCAB_SIZE):          # loop over all vocabulary tokens
            vec_t = torch.tensor(                 # one-hot tensor for this token
                one_hot(tok_id).reshape(1, -1), dtype=torch.float32
            )
            rewards_pt[tok_id] = reward_model(vec_t).squeeze()  # scalar reward

    # Baseline: subtract mean reward to center advantages (reduces gradient variance)
    baseline_pt  = rewards_pt.mean()             # mean reward across all 8 tokens
    advantages_pt = rewards_pt - baseline_pt     # centered advantages, shape (8,)

    # Policy gradient loss: -sum(advantages * log_probs)
    # Negative because we want to MAXIMIZE reward but PyTorch optimizers MINIMIZE loss.
    # C# analogy: loss = -reward (flip sign to turn maximization into minimization).
    log_probs_pt = torch.log(probs_rl_pt.squeeze() + 1e-9)  # log probs, shape (8,)
    rl_loss_pt = -(advantages_pt * log_probs_pt).sum()       # scalar RL loss

    rl_loss_pt.backward()                        # compute gradients via autograd
    rl_optimizer.step()                          # update SFT model weights

    # Track probability of "sure" after this update (for monitoring)
    with torch.no_grad():                        # inference mode
        lp = sft_model(prompt_tensor)            # forward pass
        pp = torch.softmax(lp, dim=1)            # probabilities
        sure_prob_pt = pp[0, VOCAB["sure"]].item()  # prob of "sure" token

    if (step + 1) % 5 == 0:                     # print every 5 steps
        print(f"  {step+1:>4}  |  {rl_loss_pt.item():>8.4f}  |  {sure_prob_pt:.4f}")

print()

# Final policy distribution after RL
with torch.no_grad():                            # inference mode
    logits_final_pt = sft_model(prompt_tensor)  # forward pass
    probs_final_pt  = torch.softmax(logits_final_pt, dim=1)  # probabilities

print("Final PyTorch policy probabilities after RL:")
for token_id in range(VOCAB_SIZE):               # loop over all 8 tokens
    p = probs_final_pt[0, token_id].item()       # Python float from tensor
    # Get reward for this token (for display)
    with torch.no_grad():                        # no gradient tracking
        vec_t = torch.tensor(one_hot(token_id).reshape(1, -1), dtype=torch.float32)
        r = reward_model(vec_t).item()           # scalar reward
    bar = "=" * int(p * 40)                      # ASCII bar
    print(f"  '{VOCAB_INV[token_id]:8s}': prob={p:.4f}  reward={r:.3f} |{bar}")

print()

# Show gradient flow: which parameters received gradients?
print("--- Gradient flow (after final RL backward pass) ---")
for name, param in sft_model.named_parameters():  # iterate over all model parameters
    has_grad = param.grad is not None              # True if gradient was computed
    if has_grad:
        grad_norm = param.grad.norm().item()       # L2 norm of the gradient tensor
        print(f"  {name:30s}: grad_norm = {grad_norm:.6f}  [gradient EXISTS]")
    else:
        print(f"  {name:30s}: NO gradient")        # parameter was not in compute graph

print()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 65)
print("SUMMARY - The 3-Phase RLHF Pipeline")
print("=" * 65)
print()
print("Phase 1: SFT (Supervised Fine-Tuning)")
print("  - Start from pretrained weights")
print("  - Train on human-written 'ideal' responses")
print("  - Uses standard cross-entropy loss (supervised learning)")
print("  - C# analogy: Training a classifier on labelled data")
print()
print("Phase 2: Reward Model Training")
print("  - Train a SEPARATE model on human preference pairs")
print("  - Loss: Bradley-Terry = -log(sigmoid(score_good - score_bad))")
print("  - Output: a scalar 'quality score' for any response")
print("  - C# analogy: A quality-scoring service (like SonarQube)")
print()
print("Phase 3: RL (Policy Gradient / PPO)")
print("  - Use the reward model as the optimization target")
print("  - Update policy to MAXIMIZE expected reward")
print("  - Loss = -( advantage * log_prob ) [negative for gradient ascent]")
print("  - C# analogy: A/B testing loop that amplifies what scores well")
print()
print("End result: A model that generates responses humans prefer.")
print("This is how ChatGPT, Claude, and Gemini are aligned.")
print()
print("Next: Run example_02_reward_model.py for a deeper dive into")
print("building and training a reward model from scratch.")

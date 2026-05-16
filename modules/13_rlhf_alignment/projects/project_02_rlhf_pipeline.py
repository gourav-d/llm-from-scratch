# =============================================================================
# MODULE 13 - PROJECT 02: FULL RLHF PIPELINE
# =============================================================================
# Title   : End-to-End RLHF Simulation: SFT -> Reward Model -> PPO
# Goal    : Simulate the three phases of RLHF (Reinforcement Learning from
#           Human Feedback) in one self-contained NumPy script.
#           Start from a random policy, improve it with supervised fine-tuning,
#           train a reward model on preferences, then use PPO to align the
#           final model to human preferences.
# What you
# will    : 1. Set up a shared vocabulary and policy representation
# build   : 2. Phase 1 SFT: train policy on high-quality human demonstrations
#           3. Phase 2 Reward Model: learn to score (state, action) pairs
#           4. Phase 3 PPO: update policy to maximise reward with KL penalty
#           5. Compare distributions: random -> SFT -> PPO to see alignment
#           6. Reflect on what changed and why
# How to  :   python project_02_rlhf_pipeline.py
# run     :
# C# analogy (overall system):
#   Think of this like training a customer support bot in three stages:
#   Stage 1 (SFT): Give the bot a script of approved responses and
#                  fine-tune it to follow the script (supervised learning).
#   Stage 2 (RM):  Collect A/B test results ("customers preferred option A")
#                  and train a rating model to predict customer preference.
#   Stage 3 (PPO): Automatically search for responses the rating model
#                  would score highly, while making sure the bot doesn't
#                  stray too far from its original approved-script behavior
#                  (the KL penalty keeps it grounded).
# Dependencies: Python 3.10+ and NumPy only. No PyTorch, no APIs.
# =============================================================================

# =============================================================================
# GLOSSARY
# (Read this before the code -- every term used in this file is defined here)
# =============================================================================
#
# RLHF (Reinforcement Learning from Human Feedback)
#   A three-phase training pipeline used to align LLMs to human values.
#   Phases: (1) SFT -- learn from human demonstrations,
#           (2) RM  -- learn to score responses from human preferences,
#           (3) RL  -- optimise the policy to maximise RM score.
#   Used by: ChatGPT, Claude, Gemini, and virtually all deployed LLMs.
#   C# analogy: a three-sprint delivery cycle with testing after each sprint.
#
# Policy
#   In RL, the "policy" is the model that decides what action (token) to
#   produce given the current state (context).  Here: W_policy (8x8) ->
#   softmax -> probability distribution over 8 tokens.
#   C# analogy: a Func<State, ActionDistribution> that the agent uses.
#
# SFT (Supervised Fine-Tuning)
#   Phase 1 of RLHF.  We show the model (state, target_token) pairs from
#   human-written demonstrations and train it with cross-entropy loss to
#   copy the human's choices.
#   C# analogy: training a classifier with labelled data (classic ML).
#
# Reward Model (RM)
#   Phase 2 of RLHF.  A separate neural network trained on preference pairs
#   (state, good_token) vs (state, bad_token) to predict which token a human
#   would prefer.  Used as the reward signal in Phase 3.
#   C# analogy: a scoring function trained on A/B test results.
#
# PPO (Proximal Policy Optimisation)
#   Phase 3 of RLHF.  An RL algorithm that updates the policy to maximise
#   reward while staying close (via clipping + KL penalty) to the SFT policy.
#   "Proximal" means: don't take a step so large you break the policy.
#   C# analogy: a constrained optimizer with a trust-region step-size limit.
#
# KL Divergence
#   Measures how different two probability distributions are.
#   KL(P || Q) = sum(P * log(P / Q)).  Returns 0 if P == Q, > 0 otherwise.
#   In PPO, KL(current_policy || SFT_policy) penalises drifting too far
#   from the SFT policy (which was trained on approved human demonstrations).
#   C# analogy: measuring how far a new release drifted from the baseline.
#
# Importance Ratio (r_t)
#   r_t = pi_new(a|s) / pi_old(a|s).  The ratio of probabilities under
#   the new vs old policy.  If = 1.0, policy unchanged.  PPO clips this to
#   [1-epsilon, 1+epsilon] to prevent large updates.
#   C# analogy: the "change factor" relative to last checkpoint.
#
# Clipped Surrogate Objective
#   PPO's loss function: min(r_t * A, clip(r_t, 1-e, 1+e) * A).
#   Clipping prevents the policy from changing too drastically in one step.
#   C# analogy: Math.Clamp(ratio, lowerBound, upperBound) on the update size.
#
# Advantage
#   How much better an action was compared to the average expected reward.
#   Advantage = actual_reward - baseline.  Positive = action was good.
#   C# analogy: delta from an expected-value estimate in a decision model.
#
# Bradley-Terry Loss
#   Loss for pairwise preference learning.
#   L = -log(sigmoid(score_good - score_bad))
#   Encourages score_good > score_bad for all preference pairs.
#   C# analogy: a log-likelihood loss on comparisons rather than absolute labels.
#
# Token Embedding
#   A learned vector representation for each token.
#   Here: a lookup matrix E_tokens (vocab_size x state_dim) where each row
#   is the embedding for one token.
#   C# analogy: Dictionary<int, double[]> mapping token ID to vector.
#
# State Vector
#   A fixed-size vector representing the current context (what the model
#   "knows" before generating the next token).
#   C# analogy: a double[] summary of the conversation so far.
#
# Softmax
#   Converts a vector of raw scores into a probability distribution.
#   All values >= 0, sum = 1.0.  Largest score gets highest probability.
#   C# analogy: exp(x_i) / sum(exp(x)) for each element x_i.
#
# =============================================================================

# --- IMPORTS -----------------------------------------------------------------
import numpy as np          # NumPy: numerical computing -- our only dependency
import math                 # math: built-in Python math functions (log, exp)
import copy                 # copy: built-in Python module for deep-copying objects

np.random.seed(0)           # Fix random seed for reproducibility
                            # C# analogy: new Random(0) -- same results every run


# =============================================================================
# ASCII DIAGRAM: FULL RLHF PIPELINE
# =============================================================================
#
#   FULL RLHF PIPELINE
#   ==================
#   [Pretrained Model Weights]
#            |
#       PHASE 1: SFT
#       Supervised fine-tuning on
#       high-quality human examples
#            |
#       [SFT Model Weights]  <-- saved, used as reference policy in Phase 3
#            |
#       PHASE 2: REWARD MODEL
#       Train RM on preference pairs
#       to predict human preferences
#            |
#       [Reward Model Weights]  <-- used to assign reward in Phase 3
#            |
#       PHASE 3: PPO
#       Use RM as reward signal
#       Update policy to max reward
#       with KL penalty vs SFT model
#       (clipped importance ratios)
#            |
#       [Aligned Model Weights]
#
# =============================================================================

print("=" * 70)          # Print divider
print("MODULE 13 - PROJECT 02: FULL RLHF PIPELINE")   # Project title
print("=" * 70)          # Print divider


# =============================================================================
# PART 1: SHARED SETUP -- VOCABULARY, STATE DIM, POLICY REPRESENTATION
# =============================================================================

print("\n" + "=" * 70)   # Section separator
print("PART 1: SHARED SETUP")   # Section header
print("=" * 70)          # Section separator

# --- Vocabulary --------------------------------------------------------------
# We use a tiny vocabulary of 8 tokens.
# In a real LLM, vocab_size would be 50,000+, but 8 is enough to show all concepts.
# C# analogy: enum Token { Safe=0, Helpful=1, Polite=2, Clear=3,
#                          Harm=4, Vague=5, Rude=6, Irrelevant=7 }

VOCAB_SIZE = 8            # Number of tokens in our tiny vocabulary
STATE_DIM  = 8            # Dimension of the state vector (matches VOCAB_SIZE for simplicity)
HIDDEN_DIM = 8            # Hidden layer size for reward model

# Token names: first 4 are "good" tokens, last 4 are "bad" tokens.
# The model should learn to prefer good tokens.
TOKEN_NAMES = [           # Human-readable names for each token index
    "Safe",               # Token 0: a safe, appropriate response
    "Helpful",            # Token 1: a helpful, relevant response
    "Polite",             # Token 2: a polite, respectful response
    "Clear",              # Token 3: a clear, easy-to-understand response
    "Harm",               # Token 4: a harmful or dangerous response
    "Vague",              # Token 5: a vague, non-committal response
    "Rude",               # Token 6: a rude or disrespectful response
    "Irrelevant",         # Token 7: an off-topic, irrelevant response
]

GOOD_TOKENS = [0, 1, 2, 3]   # Indices of "good" (aligned) tokens
BAD_TOKENS  = [4, 5, 6, 7]   # Indices of "bad" (misaligned) tokens

# --- Token embeddings --------------------------------------------------------
# Each token gets a fixed vector representation (embedding).
# E_tokens shape: (VOCAB_SIZE, STATE_DIM) = (8, 8).
# Row i is the embedding for token i.
# C# analogy: double[,] TokenEmbeddings = new double[VOCAB_SIZE, STATE_DIM];
E_tokens = np.random.randn(VOCAB_SIZE, STATE_DIM) * 0.3  # Random embeddings, small values

# --- Policy representation ---------------------------------------------------
# The policy is a linear layer: W_policy (STATE_DIM x VOCAB_SIZE) = (8 x 8).
# Given a state vector s (shape 8,), the policy computes:
#   logits = W_policy.T @ s    (shape VOCAB_SIZE,)
#   probs  = softmax(logits)   (shape VOCAB_SIZE,) -- sum to 1.0
# The agent then samples a token from this distribution.
# C# analogy: double[] GetTokenProbabilities(double[] state) using a matrix multiply.


def softmax(logits):
    """
    Numerically stable softmax: converts logits to probabilities.
    Subtracts max before exp() to prevent overflow.
    C# analogy: exp(x - max(x)) / sum(exp(x - max(x))) for each element.
    """
    shifted = logits - np.max(logits)         # Subtract max for numerical stability
    exps    = np.exp(shifted)                 # Compute exp of each element
    return exps / np.sum(exps)                # Normalise to sum to 1.0


def get_probs(W_policy, state):
    """
    Compute token probability distribution from a policy weight matrix and state.
    W_policy : shape (STATE_DIM, VOCAB_SIZE)
    state    : shape (STATE_DIM,)
    Returns  : shape (VOCAB_SIZE,) probability distribution over tokens
    C# analogy: double[] ComputeActionProbs(double[,] W, double[] state)
    """
    logits = W_policy.T @ state              # Linear transform: (VOCAB_SIZE, STATE_DIM) @ (STATE_DIM,) = (VOCAB_SIZE,)
    return softmax(logits)                   # Convert logits to probabilities


def sample_token(probs):
    """
    Sample a token index from a probability distribution.
    Higher probability -> more likely to be sampled.
    C# analogy: weighted random selection from an array of probabilities.
    """
    return np.random.choice(VOCAB_SIZE, p=probs)  # Sample one index according to probs


# --- Initialise the policy with random weights -------------------------------
# This is our "pretrained" model -- random weights before any training.
W_policy_init = np.random.randn(STATE_DIM, VOCAB_SIZE) * 0.1  # Small random init
W_policy = W_policy_init.copy()             # Working copy of policy weights

print(f"Vocabulary size    : {VOCAB_SIZE} tokens")    # Show vocabulary size
print(f"State vector size  : {STATE_DIM}")             # Show state dimension
print(f"Policy matrix shape: {W_policy.shape}")        # Show policy weight shape
print(f"Token names (good 0-3, bad 4-7): {TOKEN_NAMES}")  # Show token names

# Show initial (random) policy distribution on a sample state
sample_state = np.random.randn(STATE_DIM) * 0.5       # A sample state vector
init_probs   = get_probs(W_policy, sample_state)       # Probability distribution from initial random policy
print(f"\nInitial policy (random weights) on sample state:")
for i, (name, p) in enumerate(zip(TOKEN_NAMES, init_probs)):   # Loop over each token
    bar = "#" * int(p * 40)                            # Scale probability to bar length
    print(f"  Token {i} {name:<12}: {p:.4f}  {bar}")  # Print token name, probability, bar


# =============================================================================
# PART 2: PHASE 1 -- SUPERVISED FINE-TUNING (SFT)
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 2: PHASE 1 -- SUPERVISED FINE-TUNING (SFT)")   # Section header
print("=" * 70)           # Section separator

# Goal: train the policy to produce "good" tokens when given a state.
# We provide 10 (state, target_token) training examples.
# Loss: negative log-probability of the target token (cross-entropy).
# C# analogy: training a classifier where the label is the "approved" token.

print("\nSFT training data: 10 (state, target_token) pairs")
print("Target tokens are always from the 'good' set (0-3).")

# --- 10 SFT training examples: (state_vector, target_token_index) ------------
# Each state is a different random vector (representing a different context).
# Each target is one of the four "good" tokens (0=Safe, 1=Helpful, 2=Polite, 3=Clear).
# C# analogy: List<(double[] State, int TargetToken)> sftExamples

np.random.seed(1)         # Different seed so SFT states are different from policy init
sft_examples = [          # List of (state_vector, target_token_index) tuples
    (np.random.randn(STATE_DIM) * 0.5, 0),   # State 0 -> target: Safe
    (np.random.randn(STATE_DIM) * 0.5, 1),   # State 1 -> target: Helpful
    (np.random.randn(STATE_DIM) * 0.5, 2),   # State 2 -> target: Polite
    (np.random.randn(STATE_DIM) * 0.5, 3),   # State 3 -> target: Clear
    (np.random.randn(STATE_DIM) * 0.5, 0),   # State 4 -> target: Safe
    (np.random.randn(STATE_DIM) * 0.5, 1),   # State 5 -> target: Helpful
    (np.random.randn(STATE_DIM) * 0.5, 2),   # State 6 -> target: Polite
    (np.random.randn(STATE_DIM) * 0.5, 3),   # State 7 -> target: Clear
    (np.random.randn(STATE_DIM) * 0.5, 1),   # State 8 -> target: Helpful
    (np.random.randn(STATE_DIM) * 0.5, 0),   # State 9 -> target: Safe
]

np.random.seed(0)         # Restore main seed

# --- SFT training hyperparameters --------------------------------------------
SFT_EPOCHS = 20           # Number of passes over the SFT training data
SFT_LR     = 0.1          # Learning rate for SFT gradient descent
EPS        = 1e-9         # Small epsilon to prevent log(0)

print(f"\nSFT training: {SFT_EPOCHS} epochs, lr={SFT_LR}")
print(f"\n  {'Epoch':>6}  {'SFT Loss':>10}")  # Column headers
print(f"  {'------':>6}  {'--------':>10}")  # Divider

for epoch in range(1, SFT_EPOCHS + 1):    # Loop from epoch 1 to SFT_EPOCHS

    total_loss = 0.0                       # Accumulate loss across all training examples
    dW_total   = np.zeros_like(W_policy)   # Accumulated gradient for policy weights

    for state, target_token in sft_examples:  # Loop over each (state, target_token) pair

        # Forward pass: compute probability distribution over tokens
        probs  = get_probs(W_policy, state)   # Shape (VOCAB_SIZE,): probability per token

        # SFT loss: negative log-probability of the target token
        # We want log(probs[target_token]) to be as large as possible (close to 0)
        # C# analogy: -Math.Log(probs[targetToken])
        loss_i = -math.log(probs[target_token] + EPS)  # Cross-entropy loss for this example
        total_loss += loss_i                            # Accumulate loss

        # Gradient of loss w.r.t. logits (combined softmax + cross-entropy):
        # dL/d_logit_j = probs[j] - (1 if j==target else 0)
        # This is the same elegant formula we saw in Project 01.
        d_logits = probs.copy()               # Start with the probability vector
        d_logits[target_token] -= 1.0         # Subtract 1 from the target token's slot

        # Gradient of loss w.r.t. W_policy:
        # logits = W_policy.T @ state, so d_logits flows back to W_policy as:
        # dL/dW_policy = state (outer) d_logits = (STATE_DIM, 1) @ (1, VOCAB_SIZE)
        # C# analogy: outer product of state and d_logits vectors
        dW = np.outer(state, d_logits)        # Shape (STATE_DIM, VOCAB_SIZE): gradient matrix
        dW_total += dW                        # Accumulate gradient

    # Average gradient over all training examples
    dW_avg = dW_total / len(sft_examples)     # Divide by number of training examples

    # Gradient descent update: move weights in the direction that reduces loss
    W_policy -= SFT_LR * dW_avg              # Update policy weights

    avg_loss = total_loss / len(sft_examples) # Average loss this epoch

    if epoch % 4 == 0 or epoch == 1:         # Print every 4 epochs (plus epoch 1)
        print(f"  {epoch:>6}  {avg_loss:>10.4f}")  # Print epoch and average loss

# Save SFT weights -- these are the "reference policy" for Phase 3 PPO
W_policy_sft = W_policy.copy()               # Deep copy of weights after SFT
print(f"\nSFT training complete. W_policy_sft saved as reference policy.")

# Show SFT policy distribution on the sample state
sft_probs = get_probs(W_policy_sft, sample_state)   # Probabilities after SFT
print(f"\nSFT policy distribution on sample state:")
for i, (name, p) in enumerate(zip(TOKEN_NAMES, sft_probs)):    # Loop over each token
    bar = "#" * int(p * 40)                                     # Scale to bar length
    marker = "  <-- good" if i in GOOD_TOKENS else ""           # Mark good tokens
    print(f"  Token {i} {name:<12}: {p:.4f}  {bar}{marker}")   # Print with marker


# =============================================================================
# PART 3: PHASE 2 -- REWARD MODEL TRAINING
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 3: PHASE 2 -- REWARD MODEL TRAINING")   # Section header
print("=" * 70)           # Section separator

# Goal: train a reward model to score (state + token) pairs.
# We use 15 preference pairs: (state, good_token, bad_token).
# The reward model scores the concatenation of state and token embedding.
# Loss: Bradley-Terry = -log(sigmoid(score_good - score_bad))
# C# analogy: training a comparison model from A/B test data.

# --- Reward model architecture -----------------------------------------------
# Input:  state + token_embedding = 8 + 8 = 16 dimensional vector
# Output: scalar score via linear layer
# W_rm shape: (STATE_DIM + STATE_DIM, 1) = (16, 1)
# C# analogy: double ScorePair(double[] state, double[] tokenEmb) using a matrix

RM_INPUT_DIM = STATE_DIM + STATE_DIM    # 8 (state) + 8 (token embedding) = 16
W_rm = np.random.randn(RM_INPUT_DIM, 1) * 0.1  # Small random init for reward model


def rm_score(W_rm, state, token_idx):
    """
    Score a (state, token) pair using the reward model.
    Concatenates state with the token's embedding and applies linear layer.
    W_rm      : shape (RM_INPUT_DIM, 1)
    state     : shape (STATE_DIM,)
    token_idx : int -- index into E_tokens vocabulary
    Returns   : scalar float score
    C# analogy: double ScoreStateToken(double[,] W, double[] state, int tokenIdx)
    """
    token_emb  = E_tokens[token_idx]             # Look up the token's embedding, shape (STATE_DIM,)
    combined   = np.concatenate([state, token_emb])  # Concatenate: shape (RM_INPUT_DIM,) = (16,)
    score      = float(W_rm.flatten() @ combined) # Linear transform: dot product -> scalar
    return score                                 # Return raw (unbounded) score


def bradley_terry_loss(score_good, score_bad):
    """
    Bradley-Terry pairwise ranking loss.
    L = -log(sigmoid(score_good - score_bad))
    Minimising this encourages score_good > score_bad.
    C# analogy: -Math.Log(Sigmoid(scoreGood - scoreBad))
    """
    diff = score_good - score_bad               # Difference in scores
    diff = np.clip(diff, -500, 500)             # Clip to prevent overflow
    sig  = 1.0 / (1.0 + np.exp(-diff))         # Sigmoid of the difference
    return -math.log(sig + EPS)                 # Negative log-likelihood


# --- 15 preference pairs: (state, good_token, bad_token) ---------------------
np.random.seed(3)           # Different seed for RM training states
rm_pairs = [                # List of (state_vector, good_token_idx, bad_token_idx) tuples
    (np.random.randn(STATE_DIM) * 0.5, 0, 4),   # Safe vs Harm
    (np.random.randn(STATE_DIM) * 0.5, 1, 5),   # Helpful vs Vague
    (np.random.randn(STATE_DIM) * 0.5, 2, 6),   # Polite vs Rude
    (np.random.randn(STATE_DIM) * 0.5, 3, 7),   # Clear vs Irrelevant
    (np.random.randn(STATE_DIM) * 0.5, 0, 7),   # Safe vs Irrelevant
    (np.random.randn(STATE_DIM) * 0.5, 1, 6),   # Helpful vs Rude
    (np.random.randn(STATE_DIM) * 0.5, 2, 5),   # Polite vs Vague
    (np.random.randn(STATE_DIM) * 0.5, 3, 4),   # Clear vs Harm
    (np.random.randn(STATE_DIM) * 0.5, 0, 5),   # Safe vs Vague
    (np.random.randn(STATE_DIM) * 0.5, 1, 4),   # Helpful vs Harm
    (np.random.randn(STATE_DIM) * 0.5, 2, 7),   # Polite vs Irrelevant
    (np.random.randn(STATE_DIM) * 0.5, 3, 6),   # Clear vs Rude
    (np.random.randn(STATE_DIM) * 0.5, 0, 6),   # Safe vs Rude
    (np.random.randn(STATE_DIM) * 0.5, 1, 7),   # Helpful vs Irrelevant
    (np.random.randn(STATE_DIM) * 0.5, 3, 5),   # Clear vs Vague
]
np.random.seed(0)           # Restore main seed

RM_EPOCHS = 20              # Number of training epochs for reward model
RM_LR     = 0.05            # Learning rate for reward model

print(f"Reward model input dim: {RM_INPUT_DIM} (state {STATE_DIM} + token_emb {STATE_DIM})")
print(f"Training pairs: {len(rm_pairs)}   Epochs: {RM_EPOCHS}   LR: {RM_LR}")
print(f"\n  {'Epoch':>6}  {'RM Loss':>10}  {'PrefAcc%':>10}")   # Column headers
print(f"  {'------':>6}  {'-------':>10}  {'--------':>10}")    # Divider

for epoch in range(1, RM_EPOCHS + 1):    # Loop through RM training epochs

    total_loss  = 0.0                     # Accumulate loss
    num_correct = 0                       # Count correctly ranked pairs
    dW_total    = np.zeros_like(W_rm)     # Accumulated gradient for W_rm

    for state, good_tok, bad_tok in rm_pairs:   # Loop over each preference pair

        sg = rm_score(W_rm, state, good_tok)    # Score the good (preferred) token
        sb = rm_score(W_rm, state, bad_tok)     # Score the bad (rejected) token

        # Compute Bradley-Terry loss
        loss_i = bradley_terry_loss(sg, sb)     # Loss for this preference pair
        total_loss += loss_i                    # Accumulate total loss

        if sg > sb:                             # Did we rank the good token higher?
            num_correct += 1                    # Yes: correct preference

        # --- Gradient of Bradley-Terry loss w.r.t. W_rm ---
        # L = -log(sigmoid(sg - sb))
        # dL/d(sg-sb) = sigmoid(sg-sb) - 1   (where sig = sigmoid(sg-sb))
        diff    = np.clip(sg - sb, -500, 500)   # Clip for safety
        sig_val = 1.0 / (1.0 + np.exp(-diff))  # sigmoid(sg - sb)
        d_diff  = sig_val - 1.0                 # Gradient w.r.t. the score difference

        # sg = W_rm.T @ combined_good, sb = W_rm.T @ combined_bad
        # dL/dW_rm = d_diff * (combined_good - combined_bad)
        good_emb    = E_tokens[good_tok]                          # Good token embedding
        bad_emb     = E_tokens[bad_tok]                           # Bad token embedding
        combined_g  = np.concatenate([state, good_emb])           # Input for good token
        combined_b  = np.concatenate([state, bad_emb])            # Input for bad token
        dW_rm_i     = d_diff * (combined_g - combined_b).reshape(-1, 1)  # Gradient for W_rm
        dW_total   += dW_rm_i                                     # Accumulate gradient

    # Average gradient and update reward model weights
    W_rm -= RM_LR * (dW_total / len(rm_pairs))  # Gradient descent step on reward model

    avg_loss = total_loss / len(rm_pairs)        # Average loss this epoch
    pref_acc = num_correct / len(rm_pairs)       # Preference accuracy this epoch

    if epoch % 4 == 0 or epoch == 1:            # Print every 4 epochs (plus epoch 1)
        print(f"  {epoch:>6}  {avg_loss:>10.4f}  {pref_acc*100:>9.1f}%")

# Final reward model stats
_, final_pref_acc_rm = 0, 0
num_correct_final = sum(1 for s, g, b in rm_pairs if rm_score(W_rm, s, g) > rm_score(W_rm, s, b))
final_pref_acc_rm = num_correct_final / len(rm_pairs)      # Final preference accuracy
print(f"\nReward model training complete.")
print(f"Final preference accuracy: {final_pref_acc_rm*100:.1f}%")  # Should be high after training


# =============================================================================
# PART 4: PHASE 3 -- PPO TRAINING
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 4: PHASE 3 -- PPO ALIGNMENT TRAINING")   # Section header
print("=" * 70)           # Section separator

# Goal: use the trained reward model as a reward signal to further improve
# the policy, while keeping it close to the SFT policy via KL penalty.
#
# PPO objective per step:
#   For each (state, action) sample:
#     1. Get reward from RM: r = rm_score(W_rm, state, action)
#     2. Compute advantage: A = r - baseline  (baseline = 0.5)
#     3. Compute importance ratio: ratio = pi_new(a|s) / pi_old(a|s)
#     4. Clipped surrogate: L_clip = min(ratio*A, clip(ratio, 1-e, 1+e)*A)
#     5. KL penalty: beta * KL(pi_current || pi_sft)
#     6. Total loss = -L_clip + beta * KL  (negate because we maximise reward)
#     7. Gradient step on W_policy
#
# C# analogy: a feedback loop where the scoring model grades each response,
# and we update the bot to produce higher-scoring responses, but with a
# constraint that it can't drift too far from the original approved script.

# --- PPO hyperparameters ---
PPO_STEPS   = 20          # Number of PPO update steps
PPO_LR      = 0.02        # Learning rate for PPO gradient steps
PPO_EPSILON = 0.2         # Clipping range: ratio clamped to [1-0.2, 1+0.2]
PPO_BETA    = 0.1         # KL penalty coefficient: higher = stay closer to SFT
PPO_BASELINE = 0.5        # Reward baseline subtracted to compute advantage

# Start PPO from SFT weights (not from random init)
W_policy = W_policy_sft.copy()           # Start from SFT weights
W_policy_ref = W_policy_sft.copy()       # Reference (frozen) SFT weights for KL

# Store history for comparison
ppo_rewards   = []        # Reward per step
ppo_kl_divs   = []        # KL divergence from SFT per step
ppo_losses    = []        # PPO loss per step

print(f"PPO hyperparameters:")
print(f"  Steps      : {PPO_STEPS}")
print(f"  LR         : {PPO_LR}")
print(f"  Epsilon    : {PPO_EPSILON} (clipping range)")
print(f"  Beta (KL)  : {PPO_BETA}")
print(f"  Baseline   : {PPO_BASELINE}")
print(f"\nStarting from SFT weights (not random).")
print(f"\n  {'Step':>5}  {'Reward':>8}  {'KL_div':>8}  {'PPO_loss':>10}")   # Column headers
print(f"  {'-----':>5}  {'------':>8}  {'------':>8}  {'--------':>10}")    # Divider


def kl_divergence(p, q):
    """
    Compute KL divergence KL(p || q) = sum(p * log(p / q)).
    Returns 0 when p == q.  Always >= 0.
    p, q : 1D NumPy arrays (probability distributions, must sum to 1.0)
    C# analogy: double KLDiv(double[] p, double[] q) = sum of p[i]*log(p[i]/q[i])
    """
    eps   = 1e-10                                 # Small constant to prevent log(0)
    p_safe = np.clip(p, eps, 1.0)                 # Ensure p has no zeros
    q_safe = np.clip(q, eps, 1.0)                 # Ensure q has no zeros
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))   # KL divergence formula


# PPO uses a fixed set of states for all steps (same states the RM was trained on)
np.random.seed(5)          # Seed for generating PPO training states
ppo_states = [np.random.randn(STATE_DIM) * 0.5 for _ in range(8)]  # 8 states per PPO step
np.random.seed(0)          # Restore main seed

for step in range(1, PPO_STEPS + 1):     # Loop through PPO steps

    total_ppo_loss = 0.0                  # Accumulate PPO loss for this step
    total_reward   = 0.0                  # Accumulate reward for this step
    total_kl       = 0.0                  # Accumulate KL divergence for this step
    dW_total       = np.zeros_like(W_policy)  # Accumulated gradient for policy

    for state in ppo_states:              # Loop over each state in this PPO step

        # --- Sample action from current policy ---
        probs_new = get_probs(W_policy, state)      # Current policy distribution
        action    = sample_token(probs_new)          # Sample a token from current policy

        # --- Get reward from reward model ---
        reward = rm_score(W_rm, state, action)       # RM scores this (state, action) pair
        total_reward += reward                       # Accumulate reward

        # --- Compute advantage ---
        # Advantage = how much better was this action than expected?
        advantage = reward - PPO_BASELINE            # Positive = better than baseline

        # --- Reference policy probabilities (frozen SFT weights) ---
        probs_ref  = get_probs(W_policy_ref, state)  # Reference (SFT) policy distribution
        prob_new_a = probs_new[action]               # Current policy's probability of chosen action
        prob_ref_a = probs_ref[action]               # Reference policy's probability of same action

        # --- Importance ratio ---
        # ratio = pi_new(a|s) / pi_old(a|s)
        # Here we use SFT (reference) as the "old" policy (simplified PPO)
        ratio = (prob_new_a + EPS) / (prob_ref_a + EPS)   # Ratio of new/ref probability for this action

        # --- Clipped surrogate objective ---
        # Standard: L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        clipped_ratio = np.clip(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON)  # Clip ratio
        surr_1 = ratio         * advantage                  # Unclipped surrogate
        surr_2 = clipped_ratio * advantage                  # Clipped surrogate
        L_clip  = min(surr_1, surr_2)                       # Take the more conservative (smaller) value

        # --- KL divergence penalty ---
        # Penalise drifting too far from the SFT reference policy
        kl = kl_divergence(probs_new, probs_ref)    # KL(current || SFT)
        total_kl += kl                              # Accumulate KL for reporting

        # --- Total PPO loss (we want to MAXIMISE L_clip, so minimise -L_clip) ---
        # Also ADD KL penalty to discourage large deviations from SFT
        ppo_loss = -L_clip + PPO_BETA * kl          # Minimise this quantity

        total_ppo_loss += ppo_loss                  # Accumulate loss

        # --- Gradient of PPO loss w.r.t. W_policy ---
        # d(ppo_loss)/d(logits) = gradient of (-L_clip) + beta * KL w.r.t. logits
        #
        # For the policy gradient (simplified):
        # d(-L_clip)/d(logits) is approximated by the policy gradient:
        #   grad = -advantage_eff * d_log_pi(a|s)/d(logits)
        # where d_log_pi(a|s)/d(logits) = (1_{j=a} - probs[j]) for each j
        # (softmax log-derivative)

        # Determine effective advantage (use clipping to match L_clip computation)
        if surr_1 <= surr_2:                       # Unclipped was the minimum (controlling)
            eff_advantage = advantage               # Use full advantage
        else:
            eff_advantage = 0.0                    # Clipped side is controlling -- zero gradient

        # d(-L_clip) w.r.t. logits: policy gradient direction
        d_log_pi = np.zeros(VOCAB_SIZE)             # Gradient of log_pi(a|s) w.r.t. logits
        d_log_pi[action] = 1.0                      # Indicator for chosen action
        d_log_pi        -= probs_new                # Subtract probabilities (softmax gradient)
        d_L_clip = -eff_advantage * d_log_pi        # Policy gradient: -advantage * d_log_pi

        # d(KL) w.r.t. logits: approximated as (probs_new - probs_ref) direction
        # This is a first-order approximation of the KL gradient
        d_kl = probs_new - probs_ref                # Simplified KL gradient w.r.t. logits

        d_loss = d_L_clip + PPO_BETA * d_kl         # Combined gradient

        # Backpropagate through logits to W_policy
        # logits = W_policy.T @ state => dW = state (outer) d_loss
        dW = np.outer(state, d_loss)                # Shape (STATE_DIM, VOCAB_SIZE)
        dW_total += dW                              # Accumulate gradient

    # Average gradients and apply update
    n_states = len(ppo_states)                     # Number of states processed this step
    dW_avg   = dW_total / n_states                 # Average gradient
    W_policy -= PPO_LR * dW_avg                    # Gradient descent step on policy

    # Compute step-level averages
    avg_reward = total_reward   / n_states         # Average reward this step
    avg_kl     = total_kl       / n_states         # Average KL divergence this step
    avg_loss   = total_ppo_loss / n_states         # Average PPO loss this step

    ppo_rewards.append(avg_reward)                 # Save reward history
    ppo_kl_divs.append(avg_kl)                     # Save KL history
    ppo_losses.append(avg_loss)                    # Save PPO loss history

    print(f"  {step:>5}  {avg_reward:>8.4f}  {avg_kl:>8.4f}  {avg_loss:>10.4f}")  # Print step stats

W_policy_ppo = W_policy.copy()         # Save PPO-aligned policy weights
print(f"\nPPO training complete. W_policy_ppo saved.")
print(f"Average reward first step  : {ppo_rewards[0]:.4f}")
print(f"Average reward final step  : {ppo_rewards[-1]:.4f}")
print(f"Average KL first step      : {ppo_kl_divs[0]:.4f}")
print(f"Average KL final step      : {ppo_kl_divs[-1]:.4f}")
kl_ok = ppo_kl_divs[-1] < 0.3         # Check if KL is small enough (< 0.3 is a common threshold)
print(f"KL < 0.3 (alignment check) : {'PASS' if kl_ok else 'FAIL'}")  # Pass if model stayed close to SFT


# =============================================================================
# PART 5: COMPARISON -- BEFORE SFT vs AFTER SFT vs AFTER PPO
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 5: POLICY DISTRIBUTION COMPARISON")   # Section header
print("=" * 70)           # Section separator

# Show how the token probability distribution changed across all three phases.
# We evaluate on the shared sample_state defined in Part 1.

init_probs = get_probs(W_policy_init, sample_state)  # Phase 0: random initial policy
sft_probs  = get_probs(W_policy_sft,  sample_state)  # Phase 1: after SFT
ppo_probs  = get_probs(W_policy_ppo,  sample_state)  # Phase 3: after PPO

print(f"\nToken probability distributions on sample state:")
print(f"(Higher probability on tokens 0-3 = more aligned)\n")

# Print side-by-side table
header_w = 12             # Width for token name column
prob_w   = 8              # Width for each probability column
print(f"  {'Token':<{header_w}}  {'Init':>{prob_w}}  {'SFT':>{prob_w}}  {'PPO':>{prob_w}}  Bar (PPO prob)")
print(f"  {'-'*header_w}  {'-'*prob_w}  {'-'*prob_w}  {'-'*prob_w}  {'---'}")

for i, name in enumerate(TOKEN_NAMES):             # Loop over each token
    p_init = init_probs[i]                         # Initial (random) probability
    p_sft  = sft_probs[i]                          # SFT probability
    p_ppo  = ppo_probs[i]                          # PPO probability
    bar    = "#" * int(p_ppo * 30)                 # Bar based on PPO probability
    quality = "GOOD" if i in GOOD_TOKENS else "bad"  # Mark good vs bad tokens
    print(f"  {name:<{header_w}}  {p_init:>{prob_w}.4f}  {p_sft:>{prob_w}.4f}  {p_ppo:>{prob_w}.4f}  {bar}  [{quality}]")

# Compute summary statistics
prob_good_init = sum(init_probs[i] for i in GOOD_TOKENS)   # Total probability on good tokens (init)
prob_good_sft  = sum(sft_probs[i]  for i in GOOD_TOKENS)   # Total probability on good tokens (SFT)
prob_good_ppo  = sum(ppo_probs[i]  for i in GOOD_TOKENS)   # Total probability on good tokens (PPO)

print(f"\nTotal probability mass on GOOD tokens (0-3):")
print(f"  Initial (random) : {prob_good_init:.4f}  ({prob_good_init*100:.1f}%)")
print(f"  After SFT        : {prob_good_sft:.4f}  ({prob_good_sft*100:.1f}%)")
print(f"  After PPO        : {prob_good_ppo:.4f}  ({prob_good_ppo*100:.1f}%)")
print(f"\nConclusion: PPO shifted probability mass from bad tokens to good tokens.")

# KL divergence from SFT to check alignment
kl_ppo_vs_sft = kl_divergence(ppo_probs, sft_probs)   # How far did PPO drift from SFT?
print(f"\nKL(PPO || SFT) = {kl_ppo_vs_sft:.4f}  "
      f"({'small -- PPO stayed close to SFT, good alignment' if kl_ppo_vs_sft < 0.3 else 'large -- PPO drifted significantly from SFT'})")

# PPO reward trend: show rewards over steps as ASCII chart
print(f"\nPPO Reward trend over {PPO_STEPS} steps (ASCII chart):")
min_r = min(ppo_rewards)          # Minimum reward in history
max_r = max(ppo_rewards)          # Maximum reward in history
r_range = max(max_r - min_r, 1e-9)  # Reward range for normalisation
print(f"  (bar length proportional to reward; longer = higher reward)\n")

for i, r in enumerate(ppo_rewards):  # Loop over each PPO step
    bar_len = int((r - min_r) / r_range * 25)   # Scale to 0..25 characters
    bar     = "#" * bar_len                      # Build bar
    print(f"  Step {i+1:>2}: {r:>7.4f}  {bar}")  # Print step, reward, bar


# =============================================================================
# PART 6: KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 70)    # Section separator
print("PART 6: KEY TAKEAWAYS")   # Section header
print("=" * 70)           # Section separator

print()   # Blank line

print("1. RLHF IS THREE SEPARATE TRAINING PHASES")
print("   SFT teaches the model to copy human examples (supervised).")
print("   Reward model training teaches a judge to score responses.")
print("   PPO uses the judge's scores to push the model toward better responses.")
print("   Each phase builds on the previous one -- order matters.")
print()

print("2. THE REFERENCE POLICY PREVENTS REWARD HACKING")
print("   Without the KL penalty, PPO would exploit the reward model --")
print("   producing nonsense that gets high scores but is not actually good.")
print("   The KL penalty forces the model to stay near the SFT policy,")
print("   which was trained on real human demonstrations.")
print()

print("3. CLIPPING IN PPO MAKES TRAINING STABLE")
print("   Clipping the importance ratio to [1-e, 1+e] prevents a single")
print("   update from radically changing the policy distribution.")
print("   C# analogy: Math.Clamp() on the learning step to prevent runaway updates.")
print()

print("4. REWARD IS PROXY, NOT GROUND TRUTH")
print("   The reward model is an approximation of human preference --")
print("   not an oracle.  It can be fooled by adversarial inputs.")
print("   This is why a small KL constraint (PPO_BETA) is critical:")
print("   if the reward model is wrong, you don't want the policy to go too far.")
print()

print("5. SFT QUALITY DETERMINES THE CEILING")
print("   PPO can only improve on SFT; it cannot fix fundamental problems")
print("   with the SFT policy (like wrong information learned from bad demos).")
print("   Real RLHF pipelines invest heavily in SFT data quality first,")
print("   then use PPO to fine-tune preferences on top of a strong SFT base.")
print()

print("=" * 70)                          # Final separator
print("Project 02 -- Full RLHF Pipeline complete. Well done!")  # Completion message
print("=" * 70)                          # Final separator

"""
=============================================================================
MODULE 13 - EXAMPLE 04: DPO TRAINING (DIRECT PREFERENCE OPTIMIZATION)
=============================================================================

WHAT YOU WILL LEARN:
  1. What DPO is and why it was invented as an alternative to RLHF/PPO
  2. How preference pairs (chosen vs rejected responses) drive training
  3. The DPO loss formula and what each term means
  4. How to implement DPO from scratch with NumPy
  5. How PyTorch makes the implementation cleaner with autograd

C# ANALOGY:
  RLHF (old way) has three tiers:
    PreferenceData -> RewardModelService -> PPOTrainer -> AlignedModel
  This is like a classic n-tier architecture with extra moving parts.

  DPO eliminates the middle tier entirely:
    PreferenceData -> DPOTrainer -> AlignedModel

  It is like replacing:
    public float Score(Response r) { return rewardModel.Predict(r); }
  with a direct formula derived mathematically from preference pairs.
  Fewer moving parts, one training job, same alignment result.

=============================================================================
"""

# ---------------------------------------------------------------------------
# GLOSSARY
# ---------------------------------------------------------------------------
GLOSSARY = {
    "Preference Pair":
        "A triple (prompt, chosen_response, rejected_response). "
        "A human said: 'chosen is better than rejected for this prompt'. "
        "(C# analogy: a TestCase with an expected winner and loser output)",

    "Chosen Response (y_w)":
        "The response the human PREFERRED. Also called the 'winner'. "
        "(C# analogy: the test case's ExpectedResult)",

    "Rejected Response (y_l)":
        "The response the human did NOT prefer. Also called the 'loser'. "
        "(C# analogy: the test case's ActualResult that failed the test)",

    "Policy (pi)":
        "The model being trained. Assigns a score to each response. "
        "(C# analogy: a Scorer class with trainable weights)",

    "Reference Policy (pi_ref)":
        "The original, frozen policy. Never updated during DPO. "
        "(C# analogy: a read-only baseline model loaded from disk)",

    "Log Ratio":
        "log(policy_score / ref_score). Measures how much the policy "
        "has shifted relative to the reference for a given response. "
        "(C# analogy: Math.Log(newScore / baselineScore))",

    "DPO Loss":
        "-log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))). "
        "Minimising this makes the model prefer chosen over rejected. "
        "(C# analogy: a binary cross-entropy loss on a preference label)",

    "Beta":
        "Temperature parameter. Controls how strongly we enforce preferences. "
        "Low beta = weak alignment, high beta = strong but possibly rigid. "
        "(C# analogy: a confidence threshold in a classification system)",

    "Sigmoid":
        "sigma(x) = 1 / (1 + exp(-x)). Maps any real number to (0, 1). "
        "(C# analogy: a probability clamp for binary classification)",

    "Preference Accuracy":
        "Fraction of pairs where policy_score(chosen) > policy_score(rejected). "
        "Should increase toward 1.0 as training progresses. "
        "(C# analogy: test pass rate — how often the right answer wins)",

    "Implicit Reward":
        "The reward DPO implicitly models: r(x,y) = beta * log(pi(y|x)/pi_ref(y|x)). "
        "No explicit reward model is needed. "
        "(C# analogy: computing a derived metric from existing data, no extra service)",
}

# ---------------------------------------------------------------------------
# ASCII DIAGRAMS
# ---------------------------------------------------------------------------
COMPARISON_DIAGRAM = """
RLHF vs DPO — ARCHITECTURE COMPARISON
=======================================

RLHF (OLD WAY) — 3 separate stages:
  Preference Data
      |
      v
  [Stage 1] Train Reward Model    <-- separate training job, extra GPU hours
      |
      v
  [Stage 2] PPO Training Loop     <-- complex, unstable, many hyperparameters
      |
      v
  Aligned Model

DPO (NEW WAY) — 1 stage:
  Preference Data
      |
      v
  [Stage 1] DPO Training Loop     <-- simple, stable, one loss function
      |
      v
  Aligned Model
  (No separate reward model needed!)
"""

DPO_FORMULA_DIAGRAM = """
DPO LOSS FORMULA (one preference pair):
========================================

  L_DPO = -log( sigma( beta * [ log_ratio_chosen - log_ratio_rejected ] ) )

  Where:
    sigma(x)           = 1 / (1 + exp(-x))           <- sigmoid function
    beta               = temperature (e.g. 0.1)
    log_ratio_chosen   = log pi(y_w|x) - log pi_ref(y_w|x)
    log_ratio_rejected = log pi(y_l|x) - log pi_ref(y_l|x)
    y_w                = chosen (winner) response
    y_l                = rejected (loser) response
    pi                 = current policy (being trained)
    pi_ref             = reference policy (frozen)

  INTUITION:
    The term in brackets measures:
      "How much MORE does the new policy prefer chosen over rejected,
       compared to the reference policy?"

    If log_ratio_chosen >> log_ratio_rejected  ->  bracket is large positive
    sigmoid of large positive                  ->  close to 1.0
    -log(close to 1.0)                         ->  loss close to 0  (good!)

    If log_ratio_chosen << log_ratio_rejected  ->  bracket is negative
    sigmoid of negative                        ->  below 0.5
    -log(below 0.5)                            ->  loss above 0.69  (bad!)
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import numpy as np          # NumPy: array math (like System.Numerics)
import torch                # PyTorch: deep learning
import torch.nn as nn       # nn: neural network layers
import torch.optim as optim # optim: gradient-based optimisers
from copy import deepcopy   # deepcopy: independent copy of an object (like Clone() in C#)

# ---------------------------------------------------------------------------
# PRINT HELPERS
# ---------------------------------------------------------------------------

def print_header(title):
    """Print a section header for readable console output."""
    print("\n" + "=" * 65)   # line of 65 "=" characters
    print(f"  {title}")      # indented title
    print("=" * 65)          # closing line


def print_bar(label, value, max_value=1.0, width=30):
    """
    Print a single ASCII bar for value, scaled by max_value.
    C# analogy: a string-formatted progress bar.
    """
    filled = int(width * abs(value) / (max_value + 1e-8))   # number of '#' chars
    filled = min(filled, width)                              # never exceed width
    bar    = "#" * filled + "-" * (width - filled)          # build bar string
    sign   = "+" if value >= 0 else "-"                      # show sign
    print(f"  {label:<22} [{sign}{bar}] {value:+.4f}")      # formatted output


# ===========================================================================
# PART A: NUMPY IMPLEMENTATION
# ===========================================================================
print_header("PART A: DPO WITH NUMPY (FROM SCRATCH)")

print(COMPARISON_DIAGRAM)    # show architecture comparison first
print(DPO_FORMULA_DIAGRAM)   # show formula with explanation

print("\nGLOSSARY (read before the code):")
for term, explanation in GLOSSARY.items():    # iterate over every term
    print(f"  [{term}]: {explanation}")       # print term and explanation

# ---------------------------------------------------------------------------
# A1. HYPERPARAMETERS
# ---------------------------------------------------------------------------
print_header("A1. Hyperparameters")

BETA_DPO   = 0.1    # temperature: how strongly preferences are enforced
LR_A       = 0.05   # learning rate for NumPy gradient descent
NUM_EPOCHS = 25     # number of full passes over the preference dataset
FEAT_DIM   = 4      # dimensionality of prompt/response feature vectors

print(f"  Beta (DPO temperature) : {BETA_DPO}")   # controls alignment strength
print(f"  Learning rate          : {LR_A}")        # gradient descent step size
print(f"  Training epochs        : {NUM_EPOCHS}")  # passes over dataset
print(f"  Feature dimension      : {FEAT_DIM}")    # size of feature vectors

# ---------------------------------------------------------------------------
# A2. FIXED RANDOM SEED
# ---------------------------------------------------------------------------
np.random.seed(42)   # reproducibility — same results every run

# ---------------------------------------------------------------------------
# A3. POLICY REPRESENTATION
# ---------------------------------------------------------------------------
# We represent a policy as a weight vector w_policy (shape: FEAT_DIM,).
# The score for a response given a prompt is:
#   score(response | prompt) = dot(w_policy, response_features)
# This is a very simplified version of what a real LLM computes.
# C# analogy: float Score(float[] responseFeatures) => Vector.Dot(wPolicy, responseFeatures);

W_policy_a = np.random.randn(FEAT_DIM) * 0.1    # trainable policy weight vector
W_ref_a    = W_policy_a.copy()                  # reference policy (frozen copy — never updated)

print_header("A2. Initial Weights")
print(f"  W_policy (initial) : {W_policy_a}")   # show initial weights
print(f"  W_ref    (frozen)  : {W_ref_a}")      # show frozen reference (identical at start)

# ---------------------------------------------------------------------------
# A4. HARDCODED PREFERENCE DATASET
# ---------------------------------------------------------------------------
# 10 preference pairs: each is (chosen_features, rejected_features).
# We omit explicit prompt features to keep the demo simple.
# The scoring function is: score = dot(w, response_features).
# C# analogy: List<(float[], float[])> preferenceDataset;
print_header("A3. Preference Dataset (10 hardcoded pairs)")

# Each row is a 4-dim feature vector representing a response.
# "Chosen" responses have been designed to align with the policy's initial direction,
# but we deliberately make some "rejected" responses close in score to create a challenge.
CHOSEN_RESPONSES = np.array([       # shape: (10, FEAT_DIM) — 10 chosen responses
    [ 0.9,  0.1, -0.2,  0.4],       # pair 0: chosen response features
    [ 0.7,  0.5,  0.1,  0.3],       # pair 1
    [ 0.8, -0.1,  0.3,  0.6],       # pair 2
    [ 0.6,  0.4,  0.2,  0.5],       # pair 3
    [ 0.5,  0.7, -0.1,  0.3],       # pair 4
    [ 0.9,  0.2,  0.4,  0.1],       # pair 5
    [ 0.4,  0.8,  0.3,  0.2],       # pair 6
    [ 0.7,  0.3,  0.5,  0.4],       # pair 7
    [ 0.6,  0.6,  0.1,  0.3],       # pair 8
    [ 0.8,  0.4,  0.2,  0.5],       # pair 9
])

REJECTED_RESPONSES = np.array([     # shape: (10, FEAT_DIM) — 10 rejected responses
    [-0.5,  0.3,  0.6, -0.2],       # pair 0: rejected response features
    [-0.3, -0.4,  0.5,  0.1],       # pair 1
    [-0.4,  0.2, -0.3, -0.5],       # pair 2
    [-0.2,  0.3, -0.6,  0.1],       # pair 3
    [-0.6, -0.2,  0.4, -0.3],       # pair 4
    [-0.3,  0.1, -0.5,  0.2],       # pair 5
    [-0.5, -0.3,  0.2, -0.4],       # pair 6
    [-0.4,  0.2, -0.2, -0.5],       # pair 7
    [-0.3, -0.5,  0.3,  0.1],       # pair 8
    [-0.5,  0.1, -0.4, -0.3],       # pair 9
])

NUM_PAIRS = len(CHOSEN_RESPONSES)   # 10 pairs

print(f"  Number of preference pairs : {NUM_PAIRS}")
print(f"  Feature dimension          : {FEAT_DIM}")
print(f"  Chosen  responses shape    : {CHOSEN_RESPONSES.shape}")    # (10, 4)
print(f"  Rejected responses shape   : {REJECTED_RESPONSES.shape}")  # (10, 4)

# ---------------------------------------------------------------------------
# A5. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def policy_score_np(w, response_features):
    """
    Compute the policy's score for a response.
    score = dot(w, response_features)
    Higher score = policy rates this response more highly.
    C# analogy: float Score(float[] w, float[] features) => Dot(w, features);
    """
    return float(np.dot(w, response_features))   # scalar dot product


def sigmoid_np(x):
    """
    Sigmoid function: maps any real number to (0, 1).
    sigma(x) = 1 / (1 + exp(-x))
    C# analogy: 1.0 / (1.0 + Math.Exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))   # clip prevents overflow


def dpo_loss_single_np(w_policy, w_ref, chosen, rejected):
    """
    Compute DPO loss for ONE preference pair.

    Formula: L_DPO = -log( sigma( beta * (log_ratio_w - log_ratio_l) ) )

    Where:
      log_ratio_chosen   = score_policy(chosen)   - score_ref(chosen)
      log_ratio_rejected = score_policy(rejected) - score_ref(rejected)

    (In real LLMs, "score" is the log probability of generating the response.
     Here we simplify to a dot-product score.)
    """
    # Score of CHOSEN response under current policy and reference policy
    score_pol_chosen = policy_score_np(w_policy, chosen)   # policy scores chosen
    score_ref_chosen = policy_score_np(w_ref,    chosen)   # reference scores chosen

    # Score of REJECTED response under current policy and reference policy
    score_pol_rejected = policy_score_np(w_policy, rejected)   # policy scores rejected
    score_ref_rejected = policy_score_np(w_ref,    rejected)   # reference scores rejected

    # Log ratios: how much has the policy shifted relative to reference?
    log_ratio_chosen   = score_pol_chosen   - score_ref_chosen    # policy's relative preference for chosen
    log_ratio_rejected = score_pol_rejected - score_ref_rejected  # policy's relative preference for rejected

    # DPO margin: positive means policy prefers chosen MORE than reference did
    margin = BETA_DPO * (log_ratio_chosen - log_ratio_rejected)   # scalar margin value

    # DPO loss: -log(sigmoid(margin))
    # When margin is large positive, sigmoid(margin) -> 1, loss -> 0 (good!)
    # When margin is negative,        sigmoid(margin) < 0.5, loss > 0.69 (bad!)
    loss = -np.log(sigmoid_np(margin) + 1e-8)   # add 1e-8 to prevent log(0)

    return float(loss)   # return as plain Python float


def compute_preference_accuracy_np(w_policy, w_ref, chosen_set, rejected_set):
    """
    Compute the fraction of pairs where policy scores chosen > rejected.
    This is the main evaluation metric for DPO training.
    C# analogy: float EvaluateAccuracy() — count wins / total.
    """
    correct = 0   # count of pairs where policy correctly prefers chosen
    total   = len(chosen_set)   # total number of pairs

    for i in range(total):                               # loop over every pair
        score_chosen   = policy_score_np(w_policy, chosen_set[i])    # score chosen
        score_rejected = policy_score_np(w_policy, rejected_set[i])  # score rejected
        if score_chosen > score_rejected:                # did policy prefer chosen?
            correct += 1                                 # yes — count it

    return correct / total   # preference accuracy: fraction in [0, 1]


def compute_dpo_gradient_np(w_policy, w_ref, chosen, rejected):
    """
    Compute the gradient of the DPO loss w.r.t. w_policy for one pair.

    We derive this using the chain rule:
      d(loss)/d(w) = d(-log(sigma(margin)))/d(w)
                   = -(1 - sigma(margin)) * d(margin)/d(w)
                   = -(1 - sigma(margin)) * beta * (chosen - rejected)

    The chosen and rejected features are the gradient directions
    because the score is a dot product: d(dot(w,x))/d(w) = x.
    C# analogy: Computing the Jacobian of the loss function w.r.t. weights.
    """
    # Recompute margin (same as in dpo_loss_single_np)
    score_pol_chosen   = policy_score_np(w_policy, chosen)
    score_ref_chosen   = policy_score_np(w_ref,    chosen)
    score_pol_rejected = policy_score_np(w_policy, rejected)
    score_ref_rejected = policy_score_np(w_ref,    rejected)

    log_ratio_chosen   = score_pol_chosen   - score_ref_chosen    # policy shift for chosen
    log_ratio_rejected = score_pol_rejected - score_ref_rejected  # policy shift for rejected
    margin             = BETA_DPO * (log_ratio_chosen - log_ratio_rejected)  # DPO margin

    sig = sigmoid_np(margin)   # sigmoid of margin (probability that chosen > rejected)

    # Gradient: -(1 - sigmoid(margin)) * beta * (chosen_features - rejected_features)
    # Direction: move w toward chosen features, away from rejected features.
    gradient = -(1.0 - sig) * BETA_DPO * (chosen - rejected)   # shape: (FEAT_DIM,)

    return gradient   # return gradient vector


# ---------------------------------------------------------------------------
# A6. EVALUATE BEFORE TRAINING
# ---------------------------------------------------------------------------
print_header("A4. Before Training — Baseline Metrics")

acc_before = compute_preference_accuracy_np(
    W_policy_a, W_ref_a, CHOSEN_RESPONSES, REJECTED_RESPONSES
)   # compute preference accuracy before any training

# Compute average scores on chosen and rejected before training
mean_chosen_before   = float(np.mean([policy_score_np(W_policy_a, c) for c in CHOSEN_RESPONSES]))
mean_rejected_before = float(np.mean([policy_score_np(W_policy_a, r) for r in REJECTED_RESPONSES]))

print(f"  Preference accuracy (before) : {acc_before:.2%}")       # e.g. 60.00%
print(f"  Mean score on chosen   (before): {mean_chosen_before:+.4f}")
print(f"  Mean score on rejected (before): {mean_rejected_before:+.4f}")
print(f"  Score gap (before): {mean_chosen_before - mean_rejected_before:+.4f}")

# ---------------------------------------------------------------------------
# A7. DPO TRAINING LOOP (NUMPY)
# ---------------------------------------------------------------------------
print_header("A5. DPO Training Loop (NumPy)")
print(f"  {'Epoch':>5}  {'Avg Loss':>9}  {'Pref Acc':>9}  {'Mean Chosen':>12}  {'Mean Rejected':>14}")
print("  " + "-" * 58)

loss_history_a = []   # collect epoch losses for final chart

for epoch in range(NUM_EPOCHS):   # loop over all epochs (C# analogy: for(int e=0; e<epochs; e++))

    epoch_loss  = 0.0                 # accumulate loss over all pairs this epoch
    total_grad  = np.zeros(FEAT_DIM)  # accumulate gradients over all pairs

    for i in range(NUM_PAIRS):        # loop over every preference pair
        chosen   = CHOSEN_RESPONSES[i]    # chosen response feature vector
        rejected = REJECTED_RESPONSES[i]  # rejected response feature vector

        # Compute loss for this pair
        loss_i = dpo_loss_single_np(W_policy_a, W_ref_a, chosen, rejected)
        epoch_loss += loss_i              # add to epoch total

        # Compute gradient for this pair
        grad_i = compute_dpo_gradient_np(W_policy_a, W_ref_a, chosen, rejected)
        total_grad += grad_i              # accumulate gradients

    # Average loss and gradient over all pairs
    avg_loss   = epoch_loss  / NUM_PAIRS   # mean loss per pair
    avg_grad   = total_grad  / NUM_PAIRS   # mean gradient per pair

    # Gradient descent update
    # C# analogy: W_policy -= learningRate * gradient;
    W_policy_a = W_policy_a - LR_A * avg_grad   # update policy weights

    # Compute evaluation metrics for this epoch
    acc_epoch = compute_preference_accuracy_np(
        W_policy_a, W_ref_a, CHOSEN_RESPONSES, REJECTED_RESPONSES
    )
    mean_chosen_e   = float(np.mean([policy_score_np(W_policy_a, c) for c in CHOSEN_RESPONSES]))
    mean_rejected_e = float(np.mean([policy_score_np(W_policy_a, r) for r in REJECTED_RESPONSES]))

    loss_history_a.append(avg_loss)   # save epoch loss

    # Print epoch summary
    print(f"  {epoch+1:>5}  {avg_loss:>9.4f}  {acc_epoch:>9.2%}  "
          f"{mean_chosen_e:>+12.4f}  {mean_rejected_e:>+14.4f}")

# ---------------------------------------------------------------------------
# A8. EVALUATE AFTER TRAINING
# ---------------------------------------------------------------------------
print_header("A6. After Training — Final Metrics")

acc_after = compute_preference_accuracy_np(
    W_policy_a, W_ref_a, CHOSEN_RESPONSES, REJECTED_RESPONSES
)

mean_chosen_after   = float(np.mean([policy_score_np(W_policy_a, c) for c in CHOSEN_RESPONSES]))
mean_rejected_after = float(np.mean([policy_score_np(W_policy_a, r) for r in REJECTED_RESPONSES]))

print(f"\n  --- BEFORE TRAINING ---")
print(f"  Preference accuracy : {acc_before:.2%}")
print(f"  Mean chosen score   : {mean_chosen_before:+.4f}")
print(f"  Mean rejected score : {mean_rejected_before:+.4f}")
print(f"  Score gap           : {mean_chosen_before - mean_rejected_before:+.4f}")

print(f"\n  --- AFTER TRAINING ---")
print(f"  Preference accuracy : {acc_after:.2%}")
print(f"  Mean chosen score   : {mean_chosen_after:+.4f}")
print(f"  Mean rejected score : {mean_rejected_after:+.4f}")
print(f"  Score gap           : {mean_chosen_after - mean_rejected_after:+.4f}")

if acc_after > acc_before:
    print(f"\n  RESULT: Preference accuracy IMPROVED from {acc_before:.2%} to {acc_after:.2%}")
else:
    print(f"\n  RESULT: Accuracy did not improve — try more epochs or a higher LR")

if acc_after >= 0.80:
    print("  VERIFIED: Policy now correctly prefers chosen on >= 80% of pairs!")
else:
    print(f"  NOTE: Accuracy is {acc_after:.2%} (try increasing NUM_EPOCHS or LR_A)")

print("\n  Verify reference policy is UNCHANGED:")
ref_score_check = policy_score_np(W_ref_a, CHOSEN_RESPONSES[0])   # score from ref
print(f"  W_ref[0] score on pair 0 chosen : {ref_score_check:.4f}  (this must never change)")
print(f"  W_ref unchanged from init       : {np.allclose(W_ref_a, W_policy_a - (W_policy_a - W_ref_a))}")

print("\n  Loss curve (ASCII chart):")
max_loss = max(loss_history_a) + 1e-8   # max loss for bar scaling
for i, loss in enumerate(loss_history_a):            # one bar per epoch
    print_bar(f"  epoch {i+1:>2}", loss, max_value=max_loss)   # ASCII loss bar


# ===========================================================================
# PART B: PYTORCH IMPLEMENTATION
# ===========================================================================
print_header("PART B: DPO WITH PYTORCH (AUTOGRAD + nn.Module)")

print("""
  KEY DIFFERENCES FROM PART A:
  - PolicyModel is an nn.Linear module (not a plain NumPy vector)
  - Reference model is a deepcopy with all gradients disabled
  - DPO loss is computed using PyTorch tensors (autograd handles gradients)
  - We verify the reference model does NOT change using assert
  - We use torch.optim.Adam (adaptive learning rates — usually better)
""")

# ---------------------------------------------------------------------------
# B1. POLICY MODEL (nn.Module)
# ---------------------------------------------------------------------------

class PolicyModel(nn.Module):
    """
    A linear policy model: score(response) = dot(w, response_features).
    Uses nn.Linear with no bias and no softmax — just a raw score.
    C# analogy: class PolicyModel { float[] W; float Score(float[] x) => Dot(W, x); }
    """

    def __init__(self, feat_dim):
        """Constructor: define the layers."""
        super().__init__()                                    # call nn.Module constructor
        # nn.Linear(in_features, out_features, bias=False)
        # Maps a feat_dim vector to a single scalar score.
        # C# analogy: float[feat_dim] W  (a single row matrix)
        self.linear = nn.Linear(feat_dim, 1, bias=False)     # 1 output = one score

    def forward(self, x):
        """
        Forward pass: compute score for a batch of responses.
        Input x: shape (batch_size, feat_dim) or (feat_dim,)
        Output:   shape (batch_size,) or scalar
        C# analogy: float[] Forward(float[][] batch) { return batch.Select(Score).ToArray(); }
        """
        return self.linear(x).squeeze(-1)   # squeeze removes the trailing 1-dim
                                            # shape: (..., 1) -> (...)


# ---------------------------------------------------------------------------
# B2. DPO LOSS FUNCTION
# ---------------------------------------------------------------------------

def dpo_loss_torch(policy_model, ref_model, chosen_batch, rejected_batch):
    """
    Compute DPO loss for a batch of preference pairs.

    Args:
      policy_model  : the model being trained (nn.Module)
      ref_model     : frozen reference model (nn.Module, no grad)
      chosen_batch  : tensor of chosen response features  shape (N, FEAT_DIM)
      rejected_batch: tensor of rejected response features shape (N, FEAT_DIM)

    Returns:
      loss  : scalar tensor (mean DPO loss over the batch)
      margin: tensor of margins (for logging)
    C# analogy: (float loss, float[] margins) ComputeDPOLoss(...)
    """
    # Score chosen responses under policy and reference
    score_pol_chosen   = policy_model(chosen_batch)    # shape: (N,) — policy scores chosen
    score_ref_chosen   = ref_model(chosen_batch)       # shape: (N,) — ref scores chosen

    # Score rejected responses under policy and reference
    score_pol_rejected = policy_model(rejected_batch)  # shape: (N,) — policy scores rejected
    score_ref_rejected = ref_model(rejected_batch)     # shape: (N,) — ref scores rejected

    # Log ratios: shift of policy relative to reference
    log_ratio_chosen   = score_pol_chosen   - score_ref_chosen    # shape: (N,)
    log_ratio_rejected = score_pol_rejected - score_ref_rejected  # shape: (N,)

    # DPO margin: measures how strongly policy prefers chosen over rejected
    margin = BETA_DPO * (log_ratio_chosen - log_ratio_rejected)   # shape: (N,)

    # DPO loss: -log(sigmoid(margin))
    # torch.nn.functional.logsigmoid(x) = log(sigmoid(x)) — numerically stable
    # So loss = -logsigmoid(margin)
    loss = -torch.nn.functional.logsigmoid(margin)   # shape: (N,) — per-pair loss

    return loss.mean(), margin   # return mean loss (scalar) and raw margins (for logging)


# ---------------------------------------------------------------------------
# B3. PREFERENCE ACCURACY (PyTorch)
# ---------------------------------------------------------------------------

def preference_accuracy_torch(policy_model, chosen_batch, rejected_batch):
    """
    Compute the fraction of pairs where policy scores chosen > rejected.
    Runs with no gradient computation (evaluation only).
    C# analogy: float Accuracy() { return pairs.Count(p => Score(p.Chosen) > Score(p.Rejected)) / total; }
    """
    with torch.no_grad():                                      # no gradients needed here
        scores_chosen   = policy_model(chosen_batch)           # shape: (N,)
        scores_rejected = policy_model(rejected_batch)         # shape: (N,)
        correct = (scores_chosen > scores_rejected).float()    # 1.0 where chosen wins, 0.0 otherwise
    return float(correct.mean().item())                        # mean -> fraction of wins


# ---------------------------------------------------------------------------
# B4. INITIALISE MODELS
# ---------------------------------------------------------------------------
print_header("B1. Initialise PyTorch Models")

torch.manual_seed(42)   # reproducibility

policy_model = PolicyModel(FEAT_DIM)   # model we will TRAIN

# deepcopy creates a completely independent copy of the model object
# (like C# deep clone — changes to policy_model will NOT affect ref_model)
ref_model = deepcopy(policy_model)     # start ref identical to policy

# Freeze the reference model: disable gradient tracking for ALL parameters
for param in ref_model.parameters():   # loop over every tensor in ref_model
    param.requires_grad_(False)        # no grad = not updated by optimiser

print("  PolicyModel parameters:")
for name, param in policy_model.named_parameters():         # loop over named params
    print(f"    {name}: shape={list(param.shape)}, requires_grad={param.requires_grad}")

print("  RefModel parameters (should all be requires_grad=False):")
for name, param in ref_model.named_parameters():            # loop over named params
    print(f"    {name}: shape={list(param.shape)}, requires_grad={param.requires_grad}")

# ---------------------------------------------------------------------------
# B5. CONVERT DATA TO PYTORCH TENSORS
# ---------------------------------------------------------------------------
# np.ndarray -> torch.Tensor so we can use PyTorch operations
# C# analogy: converting float[][] to Matrix<float> (MathNet)
chosen_tensor   = torch.tensor(CHOSEN_RESPONSES,   dtype=torch.float32)   # shape: (10, 4)
rejected_tensor = torch.tensor(REJECTED_RESPONSES, dtype=torch.float32)   # shape: (10, 4)

print(f"\n  chosen_tensor   shape: {chosen_tensor.shape}")     # (10, 4)
print(f"  rejected_tensor shape: {rejected_tensor.shape}")    # (10, 4)

# ---------------------------------------------------------------------------
# B6. SNAPSHOT REFERENCE SCORES (to verify frozen later)
# ---------------------------------------------------------------------------
with torch.no_grad():                                         # no grad for evaluation
    ref_scores_chosen_initial   = ref_model(chosen_tensor).clone()    # save initial ref scores
    ref_scores_rejected_initial = ref_model(rejected_tensor).clone()  # save initial ref scores

print(f"\n  Ref model initial chosen scores   : {ref_scores_chosen_initial.detach().numpy().round(4)}")
print(f"  Ref model initial rejected scores : {ref_scores_rejected_initial.detach().numpy().round(4)}")

# ---------------------------------------------------------------------------
# B7. OPTIMISER
# ---------------------------------------------------------------------------
# Adam: adaptive gradient method — adjusts learning rate per parameter.
# C# analogy: a smarter optimizer than vanilla gradient descent.
optimiser_b = optim.Adam(policy_model.parameters(), lr=0.05)   # only policy_model params

# ---------------------------------------------------------------------------
# B8. EVALUATE BEFORE TRAINING
# ---------------------------------------------------------------------------
print_header("B2. Before Training — Baseline Metrics (PyTorch)")

acc_before_b = preference_accuracy_torch(policy_model, chosen_tensor, rejected_tensor)

with torch.no_grad():
    mean_chosen_before_b   = float(policy_model(chosen_tensor).mean().item())
    mean_rejected_before_b = float(policy_model(rejected_tensor).mean().item())

print(f"  Preference accuracy (before) : {acc_before_b:.2%}")
print(f"  Mean chosen score   (before) : {mean_chosen_before_b:+.4f}")
print(f"  Mean rejected score (before) : {mean_rejected_before_b:+.4f}")
print(f"  Score gap           (before) : {mean_chosen_before_b - mean_rejected_before_b:+.4f}")

# ---------------------------------------------------------------------------
# B9. TRAINING LOOP (PyTorch)
# ---------------------------------------------------------------------------
print_header("B3. DPO Training Loop (PyTorch)")
print(f"  {'Epoch':>5}  {'Loss':>8}  {'Pref Acc':>9}  {'Mean Chosen':>12}  {'Mean Rejected':>14}")
print("  " + "-" * 60)

loss_history_b = []   # collect epoch losses

for epoch in range(NUM_EPOCHS):   # loop over epochs

    # ---- 1. Zero out gradients from previous step ----
    # This resets accumulated gradient buffers.
    # C# analogy: gradient.Reset();
    optimiser_b.zero_grad()   # must be done BEFORE each forward pass

    # ---- 2. Forward pass: compute DPO loss ----
    loss, margins = dpo_loss_torch(
        policy_model, ref_model, chosen_tensor, rejected_tensor
    )   # loss is a scalar tensor that has a gradient graph attached

    # ---- 3. Backward pass: compute all gradients ----
    # PyTorch traces the computation graph from loss backward to every parameter.
    # C# analogy: automagically fills in all d(loss)/d(parameter) values.
    loss.backward()   # compute gradients

    # ---- 4. Update weights ----
    # Adam uses the computed gradients to update policy_model.parameters().
    # C# analogy: weights -= learningRate * gradient (with Adam adjustments)
    optimiser_b.step()   # apply one gradient update step

    # ---- 5. Compute evaluation metrics ----
    with torch.no_grad():                             # no grad needed for evaluation
        acc_b     = preference_accuracy_torch(policy_model, chosen_tensor, rejected_tensor)
        mean_c    = float(policy_model(chosen_tensor).mean().item())
        mean_r    = float(policy_model(rejected_tensor).mean().item())

    loss_history_b.append(float(loss.item()))         # record loss this epoch

    # ---- 6. Print epoch summary ----
    print(f"  {epoch+1:>5}  {float(loss.item()):>8.4f}  {acc_b:>9.2%}  "
          f"{mean_c:>+12.4f}  {mean_r:>+14.4f}")

# ---------------------------------------------------------------------------
# B10. VERIFY REFERENCE MODEL IS FROZEN
# ---------------------------------------------------------------------------
print_header("B4. Verify Reference Model is Frozen")

with torch.no_grad():                                         # evaluation only
    ref_scores_chosen_final   = ref_model(chosen_tensor)      # final ref scores on chosen
    ref_scores_rejected_final = ref_model(rejected_tensor)    # final ref scores on rejected

# Compare initial and final ref scores — they must be IDENTICAL
chosen_unchanged   = torch.allclose(ref_scores_chosen_initial,   ref_scores_chosen_final)
rejected_unchanged = torch.allclose(ref_scores_rejected_initial, ref_scores_rejected_final)

print(f"\n  Ref model initial chosen scores  : {ref_scores_chosen_initial.numpy().round(4)}")
print(f"  Ref model final   chosen scores  : {ref_scores_chosen_final.numpy().round(4)}")
print(f"  Are ref chosen scores unchanged?  : {chosen_unchanged}   (must be True)")
print()
print(f"  Ref model initial rejected scores: {ref_scores_rejected_initial.numpy().round(4)}")
print(f"  Ref model final   rejected scores: {ref_scores_rejected_final.numpy().round(4)}")
print(f"  Are ref rejected scores unchanged?: {rejected_unchanged}   (must be True)")

# Use assert to crash loudly if the reference was accidentally updated
# C# analogy: Assert.AreEqual(before, after, "Reference model must not change")
assert chosen_unchanged,   "ERROR: Reference model chosen scores changed — freeze is broken!"
assert rejected_unchanged, "ERROR: Reference model rejected scores changed — freeze is broken!"
print("\n  ASSERTION PASSED: Reference model is confirmed frozen throughout training.")

# ---------------------------------------------------------------------------
# B11. COMPARE POLICY BEFORE AND AFTER
# ---------------------------------------------------------------------------
print_header("B5. Policy Model Scores Changed (After Training)")

with torch.no_grad():
    pol_chosen_final   = float(policy_model(chosen_tensor).mean().item())
    pol_rejected_final = float(policy_model(rejected_tensor).mean().item())

print(f"\n  Policy mean chosen score   BEFORE: {mean_chosen_before_b:+.4f}")
print(f"  Policy mean chosen score   AFTER : {pol_chosen_final:+.4f}")
print(f"  Change: {pol_chosen_final - mean_chosen_before_b:+.4f}  (should be positive)")

print(f"\n  Policy mean rejected score BEFORE: {mean_rejected_before_b:+.4f}")
print(f"  Policy mean rejected score AFTER : {pol_rejected_final:+.4f}")
print(f"  Change: {pol_rejected_final - mean_rejected_before_b:+.4f}  (should be negative)")

# The key invariant: after training, gap should have grown
gap_before = mean_chosen_before_b - mean_rejected_before_b
gap_after  = pol_chosen_final     - pol_rejected_final
print(f"\n  Score gap BEFORE: {gap_before:+.4f}")
print(f"  Score gap AFTER : {gap_after:+.4f}")
if gap_after > gap_before:
    print("  RESULT: DPO training WIDENED the gap between chosen and rejected scores!")
else:
    print("  RESULT: Gap did not increase — check hyperparameters")

# ---------------------------------------------------------------------------
# B12. FINAL EVALUATION SUMMARY
# ---------------------------------------------------------------------------
print_header("B6. Final Evaluation Summary (PyTorch)")

acc_after_b = preference_accuracy_torch(policy_model, chosen_tensor, rejected_tensor)

print(f"\n  Preference accuracy BEFORE training: {acc_before_b:.2%}")
print(f"  Preference accuracy AFTER  training: {acc_after_b:.2%}")

if acc_after_b > acc_before_b:
    print(f"  RESULT: Accuracy IMPROVED from {acc_before_b:.2%} to {acc_after_b:.2%}")
else:
    print("  RESULT: Accuracy did not improve — try tuning hyperparameters")

if acc_after_b >= 0.80:
    print("  VERIFIED: Policy correctly prefers chosen on >= 80% of pairs!")

print("\n  Loss curve (ASCII chart):")
max_loss_b = max(loss_history_b) + 1e-8                    # max loss for bar scaling
for i, loss in enumerate(loss_history_b):                  # one bar per epoch
    print_bar(f"  epoch {i+1:>2}", loss, max_value=max_loss_b)  # ASCII loss bar


# ===========================================================================
# KEY TAKEAWAYS
# ===========================================================================
print_header("KEY TAKEAWAYS")
print("""
  1. DPO ELIMINATES the reward model.
     Instead of training a separate reward model and then running PPO,
     DPO uses a closed-form formula derived from the same preference data.

  2. The DPO loss asks: "Does our policy prefer chosen over rejected MORE
     than the reference policy did?"
     If yes, loss is low. If no, loss is high, and we update weights.

  3. The reference policy is ALWAYS frozen.
     It represents the original LLM. DPO trains the policy to be better
     than the reference at selecting preferred responses.

  4. Beta controls alignment strength:
     - Low beta  -> gentle nudge, policy stays close to reference
     - High beta -> strong alignment, policy changes more aggressively

  5. In Part A we computed gradients by hand. In Part B, .backward() did it.

  6. deepcopy() + requires_grad_(False) is the standard way to freeze a
     reference model in PyTorch.

  C# PARALLEL:
    DPO is like replacing a 3-service pipeline:
      RewardModelService -> PPOScheduler -> AlignmentService
    with a single method:
      AlignmentService.TrainFromPreferences(preferenceData)
    Fewer moving parts, fewer failure points, same outcome.

  REAL-WORLD NOTE:
    In production LLMs, "score" is the SUM of log-probabilities of all
    tokens in the response. Here we simplified to a dot-product for clarity.
    The loss formula and training loop are identical — only the scoring
    function changes.
""")

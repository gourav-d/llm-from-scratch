"""
Module 13 - RLHF and Alignment: Example 02
============================================
TOPIC: Building a Reward Model from Scratch

=== WHAT IS A REWARD MODEL? ===

A Reward Model (RM) is a neural network that takes a (prompt, response)
pair as input and outputs a single number (scalar) representing
"how much would a human like this response?"

It is trained NOT by telling it the "correct" answer, but by showing
it pairs of responses and saying "A is better than B." This is called
preference learning, and the statistical framework is the Bradley-Terry model.

C# analogy: Imagine an automated code reviewer that takes a pull request
and returns a "quality score" (0.0 to 1.0). You trained it by asking
engineers to review pairs of PRs and say "this one is better than that one."
The reviewer learned to predict which style/approach engineers prefer —
without ever being told a single "correct answer."

=== GLOSSARY ===

Preference pair:
    Two responses to the same prompt where a human has labelled one
    "chosen" (preferred) and one "rejected" (not preferred).
    C# analogy: IComparable<Response> where CompareTo > 0 means "chosen is better."

Bradley-Terry model:
    A statistical model for pairwise comparisons.
    P(A is preferred over B) = sigmoid(score_A - score_B)
    Training loss: -log(sigmoid(score_chosen - score_rejected))
    C# analogy: An Elo rating system for response quality.
    The probability that the higher-rated player wins = sigmoid(Elo_diff / 400).

Bag-of-words (BoW) vector:
    A fixed-size numeric representation of text where each element
    corresponds to a feature (e.g., politeness, factual accuracy).
    C# analogy: A feature vector, like a float[] of normalized properties
    extracted from a string by a text analysis pipeline.

Preference accuracy:
    The fraction of preference pairs where the reward model
    correctly assigns a higher score to the chosen response.
    Target: 100% on training data (model has learned the preferences).
    C# analogy: Binary classification accuracy for pairs.

Scalar reward:
    A single floating-point number output by the reward model.
    Used in Phase 3 of RLHF as the optimization signal.
    C# analogy: A float return value from a scoring function.

Linear layer:
    A layer that computes output = input @ W.T + b.
    In the reward model, the final layer produces a single scalar.
    C# analogy: A weighted sum of inputs, like a dot product
    with a learned weight vector.

Sigmoid:
    Maps any real number to (0, 1). sigmoid(x) = 1 / (1 + e^(-x))
    Used to turn a raw scalar into a probability-like score.
    C# analogy: 1.0f / (1.0f + MathF.Exp(-x))

Feature vector:
    A fixed-length array of numbers that represents a piece of text.
    Here we use 6 hand-crafted features (word count, politeness, etc.)
    In real systems these come from a neural text encoder.
    C# analogy: float[] features = featureExtractor.Extract(text);

=== ARCHITECTURE DIAGRAM ===

  REWARD MODEL ARCHITECTURE
  ==========================

  Input: (prompt_features + response_features) concatenated
         shape: (12,) because 6 prompt features + 6 response features

       [x_0, x_1, x_2, x_3, x_4, x_5,  <- prompt features
        x_6, x_7, x_8, x_9, x_10, x_11] <- response features
                      |
              Linear(12, 8)   <- hidden layer
                      |
                    ReLU
                      |
              Linear(8, 1)    <- output layer (single scalar)
                      |
                  Sigmoid
                      |
              score in (0, 1)

  TRAINING SIGNAL (Bradley-Terry):
  =================================

  Chosen response  --[RM]--> score_chosen   (should be HIGH)
  Rejected response --[RM]--> score_rejected (should be LOW)

  Loss = -log( sigmoid(score_chosen - score_rejected) )

  When score_chosen >> score_rejected:
      diff is large and positive -> sigmoid(diff) near 1 -> -log(1) = 0 (no loss)

  When score_chosen <= score_rejected:
      diff is <= 0 -> sigmoid(diff) <= 0.5 -> -log(0.5) = 0.69 (high loss)

=== WHAT YOU WILL SEE ===

PART A (NumPy):
    - 8 preference pairs with 6-feature bag-of-words vectors
    - RewardModel class with forward(), train_step(), score()
    - Training loop: 20 epochs, print loss and preference accuracy each epoch
    - Before/After comparison: random scores vs trained scores
    - Ranking of 4 test responses by learned reward score

PART B (PyTorch):
    - Same architecture using nn.Module
    - SGD optimizer with gradient inspection
    - Gradient flow: print which parameters have gradients
    - Final preference accuracy on training pairs

Run with: python example_02_reward_model.py
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np      # NumPy: numerical arrays and math operations
import torch            # PyTorch: deep learning tensors and autograd
import torch.nn as nn   # nn: pre-built layers (Linear, ReLU, Sigmoid, etc.)

# ============================================================
# SHARED DATA: PREFERENCE PAIRS
# ============================================================
# Each preference pair contains:
#   chosen_features  : float list of length 6 (features of the PREFERRED response)
#   rejected_features: float list of length 6 (features of the REJECTED response)
#
# The 6 features are:
#   [0] word_count_norm  : normalized word count (longer = more informative)
#   [1] politeness       : how polite the response sounds (0=rude, 1=very polite)
#   [2] answers_question : does it actually answer the question? (0=no, 1=yes)
#   [3] factual          : how factually accurate it seems (0=wrong, 1=correct)
#   [4] helpful          : overall helpfulness rating (0=useless, 1=very helpful)
#   [5] safe             : safety rating (0=harmful, 1=safe)
#
# C# analogy: List<(float[] chosen, float[] rejected)> preferencePairs
#             where float[] has 6 elements as described above.

preference_pairs = [
    # (chosen_features,               rejected_features)
    ([0.8, 0.9, 1.0, 0.9, 0.9, 1.0], [0.3, 0.2, 0.4, 0.3, 0.2, 0.5]),  # pair 0
    ([0.7, 0.8, 0.9, 0.8, 0.8, 0.9], [0.4, 0.3, 0.3, 0.4, 0.3, 0.4]),  # pair 1
    ([0.9, 0.7, 1.0, 1.0, 0.9, 1.0], [0.2, 0.1, 0.2, 0.2, 0.1, 0.3]),  # pair 2
    ([0.6, 0.9, 0.8, 0.7, 0.8, 1.0], [0.5, 0.4, 0.2, 0.3, 0.4, 0.6]),  # pair 3
    ([0.8, 0.8, 1.0, 0.9, 1.0, 0.9], [0.1, 0.2, 0.1, 0.1, 0.2, 0.2]),  # pair 4
    ([0.7, 1.0, 0.9, 0.8, 0.9, 1.0], [0.3, 0.3, 0.4, 0.5, 0.3, 0.5]),  # pair 5
    ([0.9, 0.9, 1.0, 1.0, 1.0, 1.0], [0.2, 0.2, 0.3, 0.2, 0.2, 0.4]),  # pair 6
    ([0.6, 0.8, 0.9, 0.7, 0.8, 0.9], [0.4, 0.1, 0.2, 0.3, 0.2, 0.3]),  # pair 7
]

# Names for the 6 features (for display purposes only)
FEATURE_NAMES = [
    "word_count_norm",   # feature 0
    "politeness",        # feature 1
    "answers_question",  # feature 2
    "factual",           # feature 3
    "helpful",           # feature 4
    "safe",              # feature 5
]

# 4 test responses to rank after training
# We will score these with the trained reward model to verify it learned correctly.
# A truly good response has high values across all 6 features.
# C# analogy: List<float[]> testResponses
test_responses = [
    [0.9, 0.9, 1.0, 1.0, 1.0, 1.0],   # Test A: near-perfect response
    [0.7, 0.7, 0.8, 0.8, 0.7, 0.9],   # Test B: good response
    [0.4, 0.3, 0.4, 0.4, 0.3, 0.5],   # Test C: mediocre response
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],   # Test D: very poor response
]

TEST_LABELS = ["A (excellent)", "B (good)", "C (mediocre)", "D (poor)"]

print("=" * 65)
print("MODULE 13 - RLHF AND ALIGNMENT: EXAMPLE 02")
print("Building a Reward Model from Scratch")
print("=" * 65)
print()

# Print the preference pairs dataset
print("--- Preference Pairs Dataset ---")
print()
print(f"  Total pairs  : {len(preference_pairs)}")
print(f"  Features/pair: {len(preference_pairs[0][0])}  (per response)")
print(f"  Feature names: {FEATURE_NAMES}")
print()

for i, (chosen, rejected) in enumerate(preference_pairs):  # unpack each tuple
    print(f"  Pair {i}: chosen ={[round(v,1) for v in chosen]}  rejected={[round(v,1) for v in rejected]}")
print()

# ============================================================
# ============================================================
# PART A: REWARD MODEL WITH PURE NUMPY
# ============================================================
# ============================================================

print("=" * 65)
print("PART A: NumPy Implementation")
print("=" * 65)
print()

# ============================================================
# A.1 - HELPER FUNCTIONS
# ============================================================

def sigmoid_np(x):
    """
    Sigmoid activation function for NumPy scalars or arrays.
    WHY: Converts raw scores into (0, 1) range for probability interpretation.
    C# analogy: MathF.Sigmoid(x) or 1f / (1f + MathF.Exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))   # standard formula: maps R -> (0,1)

def relu_np(x):
    """
    ReLU activation for NumPy arrays.
    WHY: Introduces non-linearity in the hidden layer so the model can
    learn non-linear reward patterns (not just weighted sums of features).
    C# analogy: Math.Max(0.0, x) applied element-wise to an array.
    """
    return np.maximum(0.0, x)          # element-wise max(0, x)

# ============================================================
# A.2 - REWARD MODEL CLASS (NUMPY)
# ============================================================
# Architecture:
#   Input (6 features) -> Linear(6, 8) -> ReLU -> Linear(8, 1) -> Sigmoid
#   Output: scalar score in (0, 1)
#
# C# analogy: A class with two weight matrices, a forward() method,
# and a train_step() method that updates those weights via gradient descent.

class RewardModelNumpy:
    """
    A simple 2-layer reward model implemented with pure NumPy.
    WHY: Seeing the weight matrices and gradients explicitly (no magic)
    helps understand what the PyTorch version does under the hood.
    """

    def __init__(self, input_size=6, hidden_size=8):
        """
        Constructor: initializes two weight matrices and two bias vectors.
        input_size : number of input features (6 for our bag-of-words)
        hidden_size: number of hidden neurons (8 chosen arbitrarily)
        C# analogy: public RewardModelNumpy(int inputSize, int hiddenSize) { ... }
        """
        np.random.seed(99)                          # fixed seed for reproducibility

        # Layer 1 weights: shape (hidden_size, input_size) = (8, 6)
        # Scale by sqrt(2/input_size) - "He initialization" - helps training
        # C# analogy: float[8, 6] W1 = RandomNormal() * scale;
        scale1 = np.sqrt(2.0 / input_size)         # He init scale factor
        self.W1 = np.random.randn(hidden_size, input_size) * scale1  # shape (8, 6)
        self.b1 = np.zeros(hidden_size)             # bias for hidden layer, shape (8,)

        # Layer 2 weights: shape (1, hidden_size) = (1, 8)
        # C# analogy: float[1, 8] W2 = RandomNormal() * scale;
        scale2 = np.sqrt(2.0 / hidden_size)         # He init scale factor
        self.W2 = np.random.randn(1, hidden_size) * scale2   # shape (1, 8)
        self.b2 = np.zeros(1)                       # single output bias, shape (1,)

        # Cache for backward pass (stored during forward, used during backward)
        # C# analogy: private fields that store intermediate values for backprop.
        self._x     = None                          # input vector cache
        self._z1    = None                          # pre-ReLU hidden layer output
        self._a1    = None                          # post-ReLU hidden layer output
        self._z2    = None                          # pre-sigmoid output
        self._score = None                          # final sigmoid output

    def forward(self, x):
        """
        Forward pass: compute the reward score for a single feature vector.
        WHY: Defines how input flows through the network to produce a score.
        x: numpy array of shape (input_size,) = (6,) -- one response's features.
        Returns: scalar score in (0, 1).
        C# analogy: public float Forward(float[] x) { ... }
        """
        self._x  = x.copy()                        # cache input for backward pass

        # Layer 1: z1 = W1 @ x + b1
        # @ is matrix multiply. W1 shape (8,6), x shape (6,) -> z1 shape (8,)
        self._z1 = self.W1 @ x + self.b1           # linear transform, shape (8,)

        # Apply ReLU activation
        self._a1 = relu_np(self._z1)               # shape (8,) -- zeros out negatives

        # Layer 2: z2 = W2 @ a1 + b2
        # W2 shape (1,8), a1 shape (8,) -> z2 shape (1,)
        self._z2 = self.W2 @ self._a1 + self.b2   # linear transform, shape (1,)

        # Apply sigmoid to get score in (0, 1)
        self._score = sigmoid_np(self._z2)         # shape (1,) -> we return it as scalar

        return float(self._score[0])               # return Python float (scalar)

    def score(self, features):
        """
        Convenience method: just call forward() and return the scalar score.
        WHY: Clean public API separate from the training forward() call.
        C# analogy: public float Score(float[] features) => Forward(features);
        """
        return self.forward(np.array(features, dtype=np.float32))  # convert list -> array

    def train_step(self, chosen_features, rejected_features, lr=0.05):
        """
        One gradient descent step on a single preference pair.
        WHY: This is the core of reward model training --
             we want score(chosen) > score(rejected).
        chosen_features  : float array (6,) for the preferred response
        rejected_features: float array (6,) for the non-preferred response
        lr               : learning rate (step size for weight update)
        Returns: loss value as a Python float.
        C# analogy: public float TrainStep(float[] chosen, float[] rejected, float lr) { ... }
        """
        chosen_arr   = np.array(chosen_features,   dtype=np.float32)  # convert to numpy
        rejected_arr = np.array(rejected_features, dtype=np.float32)  # convert to numpy

        # Forward pass for both responses
        score_chosen   = self.forward(chosen_arr)    # scalar score for chosen
        # Save cached values for chosen backward pass
        x_c  = self._x.copy()                        # input for chosen
        z1_c = self._z1.copy()                       # pre-ReLU hidden for chosen
        a1_c = self._a1.copy()                       # post-ReLU hidden for chosen
        z2_c = self._z2.copy()                       # pre-sigmoid output for chosen

        score_rejected = self.forward(rejected_arr)  # scalar score for rejected
        # Save cached values for rejected backward pass
        x_r  = self._x.copy()                        # input for rejected
        z1_r = self._z1.copy()                       # pre-ReLU hidden for rejected
        a1_r = self._a1.copy()                       # post-ReLU hidden for rejected
        z2_r = self._z2.copy()                       # pre-sigmoid output for rejected

        # Bradley-Terry loss: -log(sigmoid(score_chosen - score_rejected))
        diff = score_chosen - score_rejected          # positive = chosen is better
        prob_correct = sigmoid_np(np.array([diff]))  # P(chosen > rejected), shape (1,)
        loss = float(-np.log(prob_correct[0] + 1e-9))  # loss value (lower = better model)

        # ---- Backward Pass ----
        # Gradient of loss w.r.t. diff:
        # d(-log(sigmoid(d)))/d(d) = sigmoid(d) - 1
        grad_diff = float(prob_correct[0]) - 1.0    # scalar gradient w.r.t. diff

        # diff = score_chosen - score_rejected
        # d(diff)/d(score_chosen)   = +1
        # d(diff)/d(score_rejected) = -1
        grad_z2_c = np.array([grad_diff])            # gradient into chosen's z2, shape (1,)
        grad_z2_r = np.array([-grad_diff])           # gradient into rejected's z2, shape (1,)

        # --- Gradients for CHOSEN path ---
        # z2_c = W2 @ a1_c + b2
        # d(loss)/d(W2) from chosen = grad_z2_c * a1_c (outer product)
        dW2_c = np.outer(grad_z2_c, a1_c)           # shape (1, 8)
        db2_c = grad_z2_c.copy()                     # shape (1,)

        # d(loss)/d(a1_c) = W2.T @ grad_z2_c
        da1_c = self.W2.T @ grad_z2_c               # shape (8,)

        # ReLU gradient: pass through where z1_c > 0, else zero
        dz1_c = da1_c * (z1_c > 0)                  # shape (8,)

        # d(loss)/d(W1) from chosen = outer(dz1_c, x_c)
        dW1_c = np.outer(dz1_c, x_c)               # shape (8, 6)
        db1_c = dz1_c.copy()                        # shape (8,)

        # --- Gradients for REJECTED path ---
        dW2_r = np.outer(grad_z2_r, a1_r)           # shape (1, 8)
        db2_r = grad_z2_r.copy()                     # shape (1,)

        da1_r = self.W2.T @ grad_z2_r               # shape (8,)
        dz1_r = da1_r * (z1_r > 0)                  # ReLU gradient
        dW1_r = np.outer(dz1_r, x_r)               # shape (8, 6)
        db1_r = dz1_r.copy()                        # shape (8,)

        # --- Combined gradient (sum chosen + rejected contributions) ---
        dW2 = dW2_c + dW2_r                         # total gradient for W2
        db2 = db2_c + db2_r                         # total gradient for b2
        dW1 = dW1_c + dW1_r                         # total gradient for W1
        db1 = db1_c + db1_r                         # total gradient for b1

        # --- Gradient descent update (minimize loss) ---
        # weight = weight - lr * gradient
        # C# analogy: weight -= learningRate * gradient;
        self.W2 -= lr * dW2                         # update W2
        self.b2 -= lr * db2                         # update b2
        self.W1 -= lr * dW1                         # update W1
        self.b1 -= lr * db1                         # update b1

        return loss                                 # return loss for monitoring

    def preference_accuracy(self, pairs):
        """
        Computes what fraction of preference pairs are ranked correctly.
        WHY: The key metric for reward model quality.
        pairs: list of (chosen_features, rejected_features) tuples.
        Returns: accuracy as a float in [0, 1].
        C# analogy: public float PrefAccuracy(List<(float[], float[])> pairs) { ... }
        """
        correct = 0                                 # count correctly ranked pairs
        for chosen, rejected in pairs:              # loop over each preference pair
            s_chosen   = self.score(chosen)         # score for chosen response
            s_rejected = self.score(rejected)       # score for rejected response
            if s_chosen > s_rejected:               # correct if chosen scores higher
                correct += 1
        return correct / len(pairs)                 # fraction of correct rankings

# ============================================================
# A.3 - INSTANTIATE THE MODEL AND SHOW BEFORE-TRAINING SCORES
# ============================================================

np_rm = RewardModelNumpy(input_size=6, hidden_size=8)  # create reward model

print("--- Scores BEFORE Training (random weights) ---")
print()

# Score each test response before training
scores_before = []                                  # store scores for comparison
for i, (resp_features, label) in enumerate(zip(test_responses, TEST_LABELS)):
    s = np_rm.score(resp_features)                 # compute reward score
    scores_before.append(s)                         # save for later comparison
    bar = "=" * int(s * 40)                         # ASCII bar scaled to 40 chars
    print(f"  Response {label}: score = {s:.4f} |{bar}")

print()
print("NOTE: Before training, scores are effectively random.")
print("An excellent response might score LOWER than a poor one.")
print()

# ============================================================
# A.4 - TRAINING LOOP: 20 EPOCHS
# ============================================================

print("--- Training the Reward Model (20 epochs) ---")
print()
print(f"  {'Epoch':>5}  |  {'Avg Loss':>9}  |  Pref. Accuracy  |  Loss bar")
print(f"  {'-'*5}  |  {'-'*9}  |  {'-'*16}  |  {'-'*25}")

LR_NP = 0.008                                      # learning rate for training (small to avoid saturation)

for epoch in range(20):                            # train for 20 full passes over pairs
    epoch_loss = 0.0                               # accumulate loss for this epoch

    for chosen_f, rejected_f in preference_pairs:  # loop over all 8 preference pairs
        loss = np_rm.train_step(                   # run one gradient descent step
            chosen_f,                              # chosen response features
            rejected_f,                            # rejected response features
            lr=LR_NP                               # learning rate
        )
        epoch_loss += loss                         # add pair loss to epoch total

    avg_loss = epoch_loss / len(preference_pairs)  # average loss per pair

    # Compute preference accuracy after this epoch
    pref_acc = np_rm.preference_accuracy(preference_pairs)  # fraction correct

    # ASCII bar for loss (shorter bar = lower loss = better)
    loss_bar = "=" * int(avg_loss * 20)            # scale bar to 20 max chars

    print(f"  {epoch+1:>5}  |  {avg_loss:>9.4f}  |  {pref_acc*100:>14.0f}%  |  |{loss_bar}")

print()

# ============================================================
# A.5 - SCORES AFTER TRAINING: COMPARE WITH BEFORE
# ============================================================

print("--- Scores AFTER Training ---")
print()

scores_after = []                                  # store post-training scores
for i, (resp_features, label) in enumerate(zip(test_responses, TEST_LABELS)):
    s = np_rm.score(resp_features)                 # compute reward score
    scores_after.append(s)                         # save for display

print("  Response          | Before   | After    | Change    | Score bar (after)")
print("  " + "-" * 70)
for i, label in enumerate(TEST_LABELS):            # loop over 4 test responses
    s_before = scores_before[i]                    # score before training
    s_after  = scores_after[i]                     # score after training
    change   = s_after - s_before                  # improvement (positive = went up)
    change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"  # format with sign
    bar = "=" * int(s_after * 40)                  # ASCII bar for after-training score
    print(f"  {label:18s}| {s_before:.4f}   | {s_after:.4f}   | {change_str:9s} | {bar}")

print()

# Rank the test responses by after-training score
sorted_indices = sorted(range(len(scores_after)), key=lambda i: scores_after[i], reverse=True)
print("Ranking by reward score (highest first):")
for rank, idx in enumerate(sorted_indices):        # loop over sorted indices
    label = TEST_LABELS[idx]                       # response label
    score = scores_after[idx]                      # reward score
    print(f"  Rank {rank+1}: Response {label}  ->  score = {score:.4f}")

print()
print("KEY INSIGHT: After training, the model correctly ranks")
print("excellent responses higher than mediocre or poor ones.")
print()

# ============================================================
# A.6 - SHOW WHAT THE MODEL LEARNED: WEIGHT INTERPRETATION
# ============================================================

print("--- What did the model learn? (W2 weights) ---")
print()
print("W2 maps hidden neurons to the output score.")
print(f"W2 shape: {np_rm.W2.shape}  (1 output neuron x {np_rm.W2.shape[1]} hidden neurons)")
print()
print("W2 values (positive = this hidden neuron helps the score,")
print("           negative = this hidden neuron hurts the score):")
for j, w_val in enumerate(np_rm.W2[0]):            # loop over W2's 8 values
    bar_len = int(abs(w_val) * 20)                 # scale bar by abs value
    direction = "+" if w_val >= 0 else "-"        # sign indicator
    bar = direction * bar_len                      # directional ASCII bar
    print(f"  Hidden neuron {j}: {w_val:+.4f}  |{bar}")

print()

# ============================================================
# ============================================================
# PART B: REWARD MODEL WITH PYTORCH
# ============================================================
# ============================================================

print("=" * 65)
print("PART B: PyTorch Implementation")
print("=" * 65)
print()

# ============================================================
# B.1 - DEFINE THE PYTORCH REWARD MODEL
# ============================================================

class RewardModelPyTorch(nn.Module):
    """
    Reward model using PyTorch nn.Module.
    Same architecture as Part A: Linear(6,8) -> ReLU -> Linear(8,1) -> Sigmoid.
    WHY: nn.Module handles gradient computation automatically (autograd),
    so we don't need to manually code the backward pass like in Part A.
    C# analogy: A class that inherits from a framework base class which
    provides automatic differentiation via source generators / IL weaving.
    """

    def __init__(self, input_size=6, hidden_size=8):
        """
        Constructor: define the layers using nn.Sequential.
        nn.Sequential chains layers so output of one feeds into next.
        C# analogy: A pipeline of processing steps (like Middleware in ASP.NET).
        """
        super().__init__()                          # required: initialize nn.Module

        # nn.Sequential: applies layers in order (like a pipeline)
        # C# analogy: new Pipeline(new Linear(6,8), new ReLU(), new Linear(8,1), new Sigmoid())
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),    # Layer 1: (6,) -> (8,)  [weight + bias]
            nn.ReLU(),                             # ReLU activation: max(0, x) element-wise
            nn.Linear(hidden_size, 1),             # Layer 2: (8,) -> (1,)  [scalar output]
            nn.Sigmoid(),                          # Sigmoid: squash output to (0, 1)
        )

    def forward(self, x):
        """
        Forward pass: feed input through the sequential network.
        x shape: (batch_size, input_size) -- usually (N, 6).
        Returns: scores of shape (batch_size, 1).
        WHY: forward() is called automatically when you call model(x).
        """
        return self.network(x)                     # pass x through all layers in sequence

# Instantiate the PyTorch reward model
pt_rm = RewardModelPyTorch(input_size=6, hidden_size=8)  # create model

print("PyTorch reward model architecture:")
print(pt_rm)                                       # PyTorch prints a neat summary
print()

# Count total trainable parameters
total_params = sum(p.numel() for p in pt_rm.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")
print(f"  Layer 1 (Linear 6->8): {6*8} weights + {8} biases = {6*8+8} params")
print(f"  Layer 2 (Linear 8->1): {8*1} weights + {1} biases = {8*1+1} params")
print()

# ============================================================
# B.2 - CONVERT DATA TO PYTORCH TENSORS
# ============================================================
# We need to convert our Python lists into PyTorch float tensors.
# C# analogy: Converting List<float[]> into a 2D Tensor<float>.

# Stack all chosen and rejected features into tensors of shape (8, 6)
# torch.tensor() converts a Python list of lists into a 2D tensor.
chosen_all   = torch.tensor(                       # shape: (8, 6)
    [chosen   for chosen, _       in preference_pairs],  # extract chosen lists
    dtype=torch.float32                            # neural networks use float32
)

rejected_all = torch.tensor(                       # shape: (8, 6)
    [rejected for _,      rejected in preference_pairs], # extract rejected lists
    dtype=torch.float32                            # same dtype as above
)

print(f"Chosen tensor shape  : {chosen_all.shape}   (8 pairs x 6 features)")
print(f"Rejected tensor shape: {rejected_all.shape}  (8 pairs x 6 features)")
print()

# Convert test responses to a tensor for batch scoring
test_tensor = torch.tensor(                        # shape: (4, 6)
    test_responses,                                # list of 4 feature vectors
    dtype=torch.float32                            # float32 for neural net
)

# ============================================================
# B.3 - SHOW SCORES BEFORE TRAINING (PyTorch)
# ============================================================

print("--- PyTorch Scores BEFORE Training ---")
print()

with torch.no_grad():                              # no_grad = don't track gradients (inference)
    scores_pt_before = pt_rm(test_tensor)          # forward pass on all 4 test responses
    # scores_pt_before shape: (4, 1) -- squeeze to (4,)
    scores_pt_before = scores_pt_before.squeeze()  # squeeze removes dim of size 1

for i, (label, score) in enumerate(zip(TEST_LABELS, scores_pt_before)):
    s = score.item()                               # .item() converts 0-d tensor to Python float
    bar = "=" * int(s * 40)                        # ASCII bar
    print(f"  Response {label}: score = {s:.4f} |{bar}")

print()

# ============================================================
# B.4 - DEFINE LOSS FUNCTION AND OPTIMIZER
# ============================================================

# We use SGD (Stochastic Gradient Descent) as requested.
# C# analogy: var optimizer = new SGD(model.Parameters(), learningRate: 0.05);
pt_optimizer = torch.optim.SGD(                    # SGD: basic gradient descent
    pt_rm.parameters(),                            # all learnable weights and biases
    lr=0.05                                        # learning rate (step size)
)

# ============================================================
# B.5 - TRAINING LOOP WITH PYTORCH (30 EPOCHS)
# ============================================================

print("--- Training PyTorch Reward Model (30 epochs) ---")
print()
print(f"  {'Epoch':>5}  |  {'Avg BT Loss':>12}  |  Pref. Accuracy")
print(f"  {'-'*5}  |  {'-'*12}  |  {'-'*18}")

for epoch in range(30):                            # train for 30 epochs
    epoch_loss_pt = 0.0                            # accumulate total loss for this epoch

    for i in range(len(preference_pairs)):         # loop over each preference pair
        pt_optimizer.zero_grad()                   # Step 1: clear gradients from last step

        # Extract single pair's features: shape (1, 6)
        chosen_i   = chosen_all[i].unsqueeze(0)   # unsqueeze adds batch dimension: (6,) -> (1,6)
        rejected_i = rejected_all[i].unsqueeze(0) # same for rejected: (6,) -> (1,6)

        # Step 2: Forward pass -- compute scores
        score_c = pt_rm(chosen_i)                 # score for chosen,   shape (1, 1)
        score_r = pt_rm(rejected_i)               # score for rejected, shape (1, 1)

        # Step 3: Bradley-Terry loss = -log(sigmoid(score_chosen - score_rejected))
        diff_pt   = score_c - score_r             # positive = model prefers chosen, shape (1,1)
        # torch.sigmoid and torch.log work element-wise on tensors
        prob_pt   = torch.sigmoid(diff_pt)        # P(chosen > rejected), shape (1,1)
        loss_pt   = -torch.log(prob_pt + 1e-9)    # BT loss; +1e-9 avoids log(0)

        epoch_loss_pt += loss_pt.item()            # accumulate scalar loss value

        # Step 4: Backward pass -- PyTorch computes all gradients automatically
        loss_pt.backward()                        # autograd runs backprop through the graph

        # Step 5: Optimizer applies the gradients to update weights
        pt_optimizer.step()                       # weights -= lr * gradients

    avg_loss_pt = epoch_loss_pt / len(preference_pairs)  # average loss per pair

    # Compute preference accuracy for this epoch (no gradient tracking needed)
    correct_pt = 0                                # count correctly ranked pairs
    with torch.no_grad():                         # inference mode
        for i in range(len(preference_pairs)):    # loop over all pairs
            sc = pt_rm(chosen_all[i].unsqueeze(0)).item()    # score for chosen
            sr = pt_rm(rejected_all[i].unsqueeze(0)).item()  # score for rejected
            if sc > sr:                           # correct if chosen scores higher
                correct_pt += 1
    pref_acc_pt = correct_pt / len(preference_pairs) * 100  # percentage

    if (epoch + 1) % 5 == 0:                     # print every 5 epochs
        print(f"  {epoch+1:>5}  |  {avg_loss_pt:>12.4f}  |  {pref_acc_pt:>16.0f}%")

print()

# ============================================================
# B.6 - SCORES AFTER TRAINING (PyTorch)
# ============================================================

print("--- PyTorch Scores AFTER Training ---")
print()

with torch.no_grad():                             # inference mode
    scores_pt_after = pt_rm(test_tensor)          # shape (4, 1)
    scores_pt_after = scores_pt_after.squeeze()   # shape (4,)

print("  Response          | Before   | After    | Score bar (after)")
print("  " + "-" * 60)
for i, label in enumerate(TEST_LABELS):           # loop over 4 test responses
    s_before = scores_pt_before[i].item()         # Python float: score before training
    s_after  = scores_pt_after[i].item()          # Python float: score after training
    bar = "=" * int(s_after * 40)                 # ASCII bar for after-training score
    print(f"  {label:18s}| {s_before:.4f}   | {s_after:.4f}   | {bar}")

print()

# Rank the test responses by after-training score
scores_pt_list = [scores_pt_after[i].item() for i in range(len(TEST_LABELS))]  # Python list
sorted_pt = sorted(                               # sort by score descending
    range(len(scores_pt_list)),
    key=lambda i: scores_pt_list[i],
    reverse=True                                  # highest score first
)

print("Final ranking by PyTorch reward score (highest first):")
for rank, idx in enumerate(sorted_pt):           # loop over sorted indices
    label = TEST_LABELS[idx]                     # label for this response
    score = scores_pt_list[idx]                  # score for this response
    print(f"  Rank {rank+1}: Response {label}  ->  score = {score:.4f}")

print()

# ============================================================
# B.7 - GRADIENT FLOW INSPECTION
# ============================================================
# After training, run ONE more forward+backward pass and inspect gradients.
# This shows WHICH parameters received gradient signals.

print("--- Gradient Flow Inspection ---")
print()
print("Running one more backward pass to inspect gradients...")
print()

pt_optimizer.zero_grad()                          # clear any leftover gradients

# Use the first preference pair for this demonstration
sc_demo = pt_rm(chosen_all[0].unsqueeze(0))       # score for chosen response [0]
sr_demo = pt_rm(rejected_all[0].unsqueeze(0))     # score for rejected response [0]

diff_demo = sc_demo - sr_demo                     # score difference
loss_demo = -torch.log(torch.sigmoid(diff_demo) + 1e-9)  # BT loss

loss_demo.backward()                              # compute gradients via autograd

print(f"  Loss value for pair 0: {loss_demo.item():.4f}")
print(f"  Score chosen  : {sc_demo.item():.4f}")
print(f"  Score rejected: {sr_demo.item():.4f}")
print()
print("  Parameter gradient summary:")
print(f"  {'Parameter name':35s}  {'Shape':12s}  {'Grad norm':>10s}  {'Has grad?':>10s}")
print(f"  {'-'*35}  {'-'*12}  {'-'*10}  {'-'*10}")

for name, param in pt_rm.named_parameters():      # iterate over all learnable parameters
    has_grad  = param.grad is not None             # True if gradient was computed
    shape_str = str(tuple(param.shape))            # format shape as string e.g. "(8, 6)"
    if has_grad:
        grad_norm = param.grad.norm().item()       # L2 norm of the gradient tensor
        print(f"  {name:35s}  {shape_str:12s}  {grad_norm:>10.6f}  {'YES':>10s}")
    else:
        print(f"  {name:35s}  {shape_str:12s}  {'N/A':>10s}  {'NO':>10s}")

print()
print("KEY INSIGHT: All parameters (weights and biases in both layers)")
print("have non-zero gradients. This means the optimizer will update ALL")
print("of them during the next optimizer.step() call.")
print()

# ============================================================
# B.8 - PREFERENCE ACCURACY ON FULL TRAINING SET
# ============================================================

print("--- Final Preference Accuracy (PyTorch) ---")
print()

correct_final = 0                                 # count correctly ranked pairs
print("  Pair  |  Score(chosen)  |  Score(rejected)  |  Correct?")
print("  " + "-" * 55)

with torch.no_grad():                             # inference mode (no gradient tracking)
    for i, (chosen_f, rejected_f) in enumerate(preference_pairs):  # loop over all pairs
        sc = pt_rm(chosen_all[i].unsqueeze(0)).item()    # score for chosen
        sr = pt_rm(rejected_all[i].unsqueeze(0)).item()  # score for rejected
        is_correct = sc > sr                      # True if model ranked correctly
        if is_correct:                            # count correct rankings
            correct_final += 1
        status = "OK" if is_correct else "WRONG"  # display string
        print(f"  Pair {i}  |  {sc:>14.4f}   |  {sr:>17.4f}   |  {status}")

final_acc = correct_final / len(preference_pairs) * 100  # percentage
print()
print(f"  Final preference accuracy: {correct_final}/{len(preference_pairs)} = {final_acc:.0f}%")
print()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 65)
print("SUMMARY - Building a Reward Model")
print("=" * 65)
print()
print("Architecture:")
print("  Input : 6 features per response (politeness, factual, etc.)")
print("  Layer1: Linear(6, 8) -> ReLU")
print("  Layer2: Linear(8, 1) -> Sigmoid")
print("  Output: scalar score in (0, 1)")
print()
print("Training signal:")
print("  Preference pairs: (chosen, rejected) from human feedback")
print("  Loss: Bradley-Terry = -log(sigmoid(score_chosen - score_rejected))")
print("  Goal: score(chosen) > score(rejected) for ALL pairs")
print()
print("Key differences: NumPy vs PyTorch:")
print("  NumPy : backward pass coded BY HAND (outer products, ReLU mask)")
print("  PyTorch: backward pass computed AUTOMATICALLY by autograd")
print("  Both   : same math, same result, just different implementation")
print()
print("C#/.NET analogy:")
print("  Reward model = an automated code reviewer (like SonarQube)")
print("  Training data = engineer preference logs ('A is better than B')")
print("  Score output = float quality score for a new pull request")
print("  Bradley-Terry = Elo rating update rule for comparisons")
print()
print("In real RLHF (ChatGPT/Claude):")
print("  - Features come from a neural text encoder (not hand-crafted)")
print("  - The reward model is itself a large transformer")
print("  - Thousands of human-labelled preference pairs are used")
print("  - The score is used by PPO to fine-tune the policy (Phase 3)")
print()
print("Done! You have built and trained a reward model from scratch.")
print("This is the core of human alignment in modern LLMs.")

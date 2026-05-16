"""
=============================================================================
MODULE 13 - EXAMPLE 05: Constitutional AI (Self-Critique and Revision)
=============================================================================

WHAT YOU WILL LEARN:
  - What Constitutional AI (CAI) is and why Anthropic invented it
  - How a model can critique its own responses using a "constitution"
  - How repeated critique-revise cycles improve alignment
  - How RLAIF (Reinforcement Learning from AI Feedback) removes the need
    for expensive human labelers
  - How to simulate these ideas with NumPy feature vectors and PyTorch nets

C# ANALOGY:
  Think of Constitutional AI like a C# code review pipeline:
    1. You write code (initial response)
    2. A linter checks it against coding standards (critique step)
    3. You fix violations based on the linter report (revision step)
    4. Repeat until the linter finds no issues (clean final response)

  The "constitution" is just the list of coding standards (rules).
  RLAIF = the linter itself was trained by reading good/bad code examples,
  so humans don't have to manually review every single file.

=============================================================================

CONSTITUTIONAL AI PIPELINE
==========================

Input: Potentially harmful prompt
         |
         v
    [Initial Response]  <-- model generates first draft
         |
         v
    [Critique Step]     <-- model judges its own response
    "Is this harmful?"  <-- based on a constitution principle
    "Does it violate X principle?"
         |
         v
    [Revision Step]     <-- model revises to fix the issue
         |
         v (repeat 1-3 times)
    [Final Safe Response]
         |
         v
    Used as training data for supervised fine-tuning!

PHASE 2 (RLAIF):
    [Same process] --> AI-labeled preference pairs
         |
         v
    Train reward model on AI labels (not human labels!)

=============================================================================
"""

# ---- GLOSSARY ---------------------------------------------------------------
GLOSSARY = {
    "Constitution":
        "A list of principles (rules) that define what a 'good' response looks like.",
    "Critique":
        "The model reads its own response and asks: does this violate any principle?",
    "Revision":
        "The model rewrites its response to fix the violations found in critique.",
    "CAI":
        "Constitutional AI — Anthropic's method of using self-critique for alignment.",
    "RLAIF":
        "Reinforcement Learning from AI Feedback — using AI labels instead of humans.",
    "Alignment Score":
        "A single number summarising how 'aligned' (safe + helpful) a response is.",
    "Feature Vector":
        "A list of numbers representing properties of a response (harm, honesty, etc.).",
    "Preference Pair":
        "(chosen, rejected) — two responses where 'chosen' is better than 'rejected'.",
    "SFT":
        "Supervised Fine-Tuning — training on clean (post-revision) examples directly.",
    "CritiqueNet":
        "A neural network that predicts how harmful a response is (0 = safe, 1 = harmful).",
    "RevisionNet":
        "A neural network that takes bad features and outputs improved features.",
}

# Print the glossary so the student sees it first
print("=" * 70)
print("GLOSSARY — Read these definitions before the code!")
print("=" * 70)
for term, definition in GLOSSARY.items():      # loop over every key-value pair
    print(f"  {term}:\n    {definition}\n")    # print term then its definition

# =============================================================================
# PART A — NUMPY SIMULATION
# =============================================================================
# We represent each response as a 5-dimensional feature vector:
#   [harm_score, honesty, helpfulness, manipulation, respect]
# Every value is in the range [0.0, 1.0].
# Lower harm and manipulation = better.
# Higher honesty, helpfulness, respect = better.
# =============================================================================

import numpy as np   # NumPy for maths — like System.Linq / Math class in C#

print("\n" + "=" * 70)
print("PART A — NumPy: Simulating Constitutional AI Self-Critique Loops")
print("=" * 70)

# ---------------------------------------------------------------------------
# Step 1 — Define the Constitution (5 principles)
# ---------------------------------------------------------------------------
# In C# this would be: List<string> constitution = new List<string> { ... };
CONSTITUTION = [                                  # list of 5 principle names
    "be_harmless",                                # do not hurt the user
    "be_honest",                                  # do not lie or mislead
    "be_helpful",                                 # actually answer the question
    "be_non_manipulative",                        # don't psychologically trick the user
    "be_respectful",                              # polite, not demeaning
]

# The INDEX of each feature in the vector (makes the code readable)
IDX_HARM         = 0    # index 0 = harm_score        (lower is better)
IDX_HONESTY      = 1    # index 1 = honesty score      (higher is better)
IDX_HELPFULNESS  = 2    # index 2 = helpfulness score  (higher is better)
IDX_MANIPULATION = 3    # index 3 = manipulation score (lower is better)
IDX_RESPECT      = 4    # index 4 = respect score      (higher is better)

# Thresholds — if a score crosses this, it violates the principle
HARM_THRESHOLD         = 0.5   # harm_score > 0.5 means "too harmful"
HONESTY_THRESHOLD      = 0.4   # honesty < 0.4 means "not honest enough"
HELPFULNESS_THRESHOLD  = 0.3   # helpfulness < 0.3 means "not helpful enough"
MANIP_THRESHOLD        = 0.5   # manipulation > 0.5 means "too manipulative"
RESPECT_THRESHOLD      = 0.4   # respect < 0.4 means "not respectful enough"

# ---------------------------------------------------------------------------
# Step 2 — Simulate an initial "bad" response
# ---------------------------------------------------------------------------
# This is a hardcoded response that violates several principles.
# In a real system this would be the LLM's actual text output, but
# we represent it as numbers for simplicity.

initial_response = np.array([
    0.85,   # harm_score        — VERY HIGH (bad! this response is harmful)
    0.60,   # honesty           — OK
    0.25,   # helpfulness       — LOW (bad! doesn't actually help)
    0.75,   # manipulation      — VERY HIGH (bad! psychologically manipulative)
    0.50,   # respect           — borderline OK
], dtype=np.float32)   # dtype=float32 matches PyTorch tensors later

print("\n--- Initial (Unrevised) Response Features ---")
for i, (name, val) in enumerate(zip(CONSTITUTION, initial_response)):
    # zip() pairs each principle name with its corresponding score value
    bar = "#" * int(val * 20)              # make a simple ASCII bar chart
    print(f"  [{i}] {name:<22}: {val:.2f}  |{bar:<20}|")   # padded output

# ---------------------------------------------------------------------------
# Step 3 — critique() function
# ---------------------------------------------------------------------------
# Checks each principle and returns a list of violation strings.
# C# equivalent:  List<string> Critique(float[] features) { ... }

def critique(response_features):
    """
    Given a response feature vector, return a list of violated principles.
    Each violation is a human-readable string explaining the problem.
    """
    violations = []    # start with an empty list — like new List<string>()

    # Check principle 0: be_harmless
    if response_features[IDX_HARM] > HARM_THRESHOLD:         # if harm is too high
        violations.append(                                    # add violation string
            f"VIOLATES be_harmless: harm={response_features[IDX_HARM]:.2f} > {HARM_THRESHOLD}"
        )

    # Check principle 1: be_honest
    if response_features[IDX_HONESTY] < HONESTY_THRESHOLD:   # if honesty is too low
        violations.append(
            f"VIOLATES be_honest: honesty={response_features[IDX_HONESTY]:.2f} < {HONESTY_THRESHOLD}"
        )

    # Check principle 2: be_helpful
    if response_features[IDX_HELPFULNESS] < HELPFULNESS_THRESHOLD:   # if helpfulness too low
        violations.append(
            f"VIOLATES be_helpful: helpfulness={response_features[IDX_HELPFULNESS]:.2f} < {HELPFULNESS_THRESHOLD}"
        )

    # Check principle 3: be_non_manipulative
    if response_features[IDX_MANIPULATION] > MANIP_THRESHOLD:        # if manipulation too high
        violations.append(
            f"VIOLATES be_non_manipulative: manipulation={response_features[IDX_MANIPULATION]:.2f} > {MANIP_THRESHOLD}"
        )

    # Check principle 4: be_respectful
    if response_features[IDX_RESPECT] < RESPECT_THRESHOLD:           # if respect too low
        violations.append(
            f"VIOLATES be_respectful: respect={response_features[IDX_RESPECT]:.2f} < {RESPECT_THRESHOLD}"
        )

    return violations    # return the full list of violations found

# ---------------------------------------------------------------------------
# Step 4 — revise() function
# ---------------------------------------------------------------------------
# Reduces bad scores and increases good scores.
# C# equivalent: float[] Revise(float[] features) { ... }

def revise(response_features):
    """
    Given a response feature vector, return a revised (improved) version.
    The revision reduces harmful and manipulative elements,
    and increases helpfulness.
    Each value is clamped to [0.0, 1.0] after adjustment.
    """
    revised = response_features.copy()    # copy() so we don't modify the original

    # Reduce harm by 0.3 — like saying "soften the dangerous parts"
    revised[IDX_HARM] = revised[IDX_HARM] - 0.30

    # Reduce manipulation by 0.3 — remove psychological tricks
    revised[IDX_MANIPULATION] = revised[IDX_MANIPULATION] - 0.30

    # Increase helpfulness by 0.2 — make the answer more useful
    revised[IDX_HELPFULNESS] = revised[IDX_HELPFULNESS] + 0.20

    # Slightly increase respect by 0.1 — be a bit nicer
    revised[IDX_RESPECT] = revised[IDX_RESPECT] + 0.10

    # Clamp all values to [0.0, 1.0] — can't go below 0 or above 1
    # np.clip() is like Math.Clamp() in C#
    revised = np.clip(revised, 0.0, 1.0)

    return revised    # return the improved feature vector

# ---------------------------------------------------------------------------
# Step 5 — alignment_score() helper
# ---------------------------------------------------------------------------
# A single number summarising overall alignment quality.
# Higher = more aligned (safer + more helpful + more honest).

def alignment_score(response_features):
    """
    Compute a single alignment score for a response.
    Combines: high honesty + high helpfulness + high respect
              - high harm - high manipulation
    Result is in roughly [0, 1] range.
    """
    # Positive signals (we WANT these high)
    positive = (
        response_features[IDX_HONESTY] +
        response_features[IDX_HELPFULNESS] +
        response_features[IDX_RESPECT]
    )

    # Negative signals (we WANT these LOW, so we subtract them)
    negative = (
        response_features[IDX_HARM] +
        response_features[IDX_MANIPULATION]
    )

    # Normalize: divide by total possible max to get a [0,1] score
    # Maximum positive = 3.0, maximum negative = 2.0, range = 5.0
    score = (positive - negative + 2.0) / 5.0    # +2.0 shifts so score >= 0

    return float(np.clip(score, 0.0, 1.0))    # ensure result stays in [0,1]

# ---------------------------------------------------------------------------
# Step 6 — Run 3 critique-revise rounds
# ---------------------------------------------------------------------------
print("\n--- Running 3 Critique-Revise Rounds ---")

current_response = initial_response.copy()    # start with the bad initial response

for round_num in range(1, 4):    # range(1, 4) = [1, 2, 3] — three rounds
    print(f"\n{'='*60}")
    print(f"  ROUND {round_num}")
    print(f"{'='*60}")

    # --- Show current features before critique ---
    score_before = alignment_score(current_response)    # compute current score
    print(f"\n  Features BEFORE revision (alignment = {score_before:.3f}):")
    for i, (name, val) in enumerate(zip(CONSTITUTION, current_response)):
        bar = "#" * int(val * 20)              # ASCII bar proportional to value
        direction = "(lower=better)" if i in [IDX_HARM, IDX_MANIPULATION] else "(higher=better)"
        print(f"    [{i}] {name:<22}: {val:.2f}  |{bar:<20}| {direction}")

    # --- Critique step: find violations ---
    violations = critique(current_response)    # call our critique function
    if violations:                             # if list is not empty
        print(f"\n  CRITIQUE found {len(violations)} violation(s):")
        for v in violations:                   # print each violation
            print(f"    * {v}")
    else:
        print("\n  CRITIQUE: No violations found! Response is fully aligned.")
        break    # exit the loop early if no more violations

    # --- Revision step: fix violations ---
    revised_response = revise(current_response)    # revise the response
    score_after = alignment_score(revised_response)
    print(f"\n  Features AFTER revision (alignment = {score_after:.3f}):")
    for i, (name, val) in enumerate(zip(CONSTITUTION, revised_response)):
        old_val = current_response[i]          # store old value for comparison
        change = val - old_val                 # compute the change
        arrow = "^" if change > 0 else ("v" if change < 0 else "-")    # direction arrow
        bar = "#" * int(val * 20)
        print(f"    [{i}] {name:<22}: {val:.2f}  |{bar:<20}| change={change:+.2f} {arrow}")

    print(f"\n  Alignment score improved: {score_before:.3f} -> {score_after:.3f}  (+{score_after-score_before:.3f})")

    current_response = revised_response    # update for next round

print(f"\n--- Final aligned response features after all rounds ---")
final_score = alignment_score(current_response)
print(f"  Final alignment score: {final_score:.3f}")
for i, (name, val) in enumerate(zip(CONSTITUTION, current_response)):
    bar = "#" * int(val * 20)
    print(f"  [{i}] {name:<22}: {val:.2f}  |{bar:<20}|")

# ---------------------------------------------------------------------------
# Step 7 — RLAIF labeling: generate preference pairs automatically
# ---------------------------------------------------------------------------
# Instead of asking humans "which response is better?", we use the
# alignment_score() function as an AI judge.
# This is how RLAIF works: AI labels replace human labels!
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("RLAIF — Generating Preference Pairs with AI Labels (no humans!)")
print("=" * 70)

def rlaif_label(response_a, response_b):
    """
    Compare two responses using AI scoring (alignment_score).
    Returns (chosen, rejected) where chosen has the higher alignment score.
    C# analogy: int Compare(Response a, Response b) returning the winner.
    """
    score_a = alignment_score(response_a)    # score response A
    score_b = alignment_score(response_b)    # score response B

    if score_a >= score_b:        # if A is at least as good as B
        return response_a, response_b, score_a, score_b    # A is chosen
    else:                         # otherwise B is better
        return response_b, response_a, score_b, score_a    # B is chosen

# Simulate 6 candidate responses with varying quality
# Each row is: [harm, honesty, helpfulness, manipulation, respect]
candidate_responses = np.array([
    [0.9, 0.5, 0.2, 0.8, 0.3],    # response 0: very bad (harmful, manipulative)
    [0.6, 0.6, 0.4, 0.5, 0.5],    # response 1: bad (still quite harmful)
    [0.3, 0.7, 0.6, 0.2, 0.7],    # response 2: decent (low harm, helpful)
    [0.1, 0.8, 0.8, 0.1, 0.9],    # response 3: very good (low harm, very helpful)
    [0.4, 0.5, 0.5, 0.4, 0.6],    # response 4: mediocre (middle of the road)
    [0.05, 0.9, 0.9, 0.05, 0.95], # response 5: excellent (near-perfect alignment)
], dtype=np.float32)               # float32 for consistency with PyTorch

print(f"\n  Candidate response scores:")
for i, resp in enumerate(candidate_responses):    # enumerate gives index + value
    score = alignment_score(resp)                  # compute AI score for this response
    print(f"  Response {i}: alignment_score = {score:.3f}  features={resp}")

# Generate 6 preference pairs by comparing selected pairs
# In practice, you would compare many more pairs (often hundreds)
comparison_pairs = [
    (0, 3),    # very bad vs very good
    (1, 4),    # bad vs mediocre
    (2, 5),    # decent vs excellent
    (0, 5),    # very bad vs excellent
    (1, 3),    # bad vs very good
    (4, 2),    # mediocre vs decent (reversed to test the function)
]

print(f"\n  RLAIF Preference Pairs (AI-labeled, no human needed):")
print(f"  {'Pair':<8} {'Chosen':<12} {'Rejected':<12} {'Chosen Score':<14} {'Rejected Score'}")
print(f"  {'-'*65}")

for pair_idx, (idx_a, idx_b) in enumerate(comparison_pairs):
    # Retrieve the two responses being compared
    resp_a = candidate_responses[idx_a]    # first response in pair
    resp_b = candidate_responses[idx_b]    # second response in pair

    # Ask the AI judge (our rlaif_label function) which is better
    chosen, rejected, score_c, score_r = rlaif_label(resp_a, resp_b)

    # Figure out which original index is which (for display)
    score_a = alignment_score(resp_a)
    chosen_idx = idx_a if score_a == score_c else idx_b    # which original index won
    rejected_idx = idx_b if score_a == score_c else idx_a

    print(f"  Pair {pair_idx+1}:   Resp {chosen_idx} chosen    Resp {rejected_idx} rejected    {score_c:.3f}          {score_r:.3f}")

print(f"\n  -> These {len(comparison_pairs)} preference pairs become training data")
print(f"     for a reward model WITHOUT any human annotation!")

# =============================================================================
# PART B — PYTORCH: CritiqueNet and RevisionNet
# =============================================================================
# Now we use neural networks to LEARN the critique and revision functions,
# instead of hardcoding the rules (thresholds, adjustments).
#
# CritiqueNet:  features (5) -> harm_probability (1)
# RevisionNet:  features (5) + critique_score (1) -> revised_features (5)
#
# C# analogy:
#   CritiqueNet = a trained classifier (like ML.NET BinaryClassification)
#   RevisionNet = a trained transformation model (like a seq2seq translator)
# =============================================================================

import torch                           # PyTorch — deep learning framework
import torch.nn as nn                  # nn = Neural Network building blocks
import torch.nn.functional as F        # F = activation functions, loss functions

print("\n" + "=" * 70)
print("PART B — PyTorch: CritiqueNet and RevisionNet")
print("=" * 70)

torch.manual_seed(42)    # set random seed so results are reproducible

# ---------------------------------------------------------------------------
# CritiqueNet: predicts harm probability from response features
# ---------------------------------------------------------------------------
# Input:  5-dimensional feature vector [harm, honesty, helpful, manip, respect]
# Output: 1 scalar — probability that the response is harmful (0=safe, 1=harmful)
# ---------------------------------------------------------------------------

class CritiqueNet(nn.Module):    # nn.Module is like implementing an interface in C#
    """
    Neural network that takes a response feature vector (5 dims)
    and outputs a single harm_probability (0 = safe, 1 = harmful).

    Architecture:
      Input(5) -> Linear(5->16) -> ReLU -> Linear(16->8) -> ReLU -> Linear(8->1) -> Sigmoid
    """
    def __init__(self):
        """
        __init__ is the constructor — like public CritiqueNet() in C#.
        We define all the layers here.
        """
        super().__init__()    # call parent class constructor (required by PyTorch)

        # Layer 1: 5 inputs -> 16 hidden neurons
        # Like a dense/fully-connected layer in ML.NET
        self.layer1 = nn.Linear(5, 16)    # (input_size, output_size)

        # Layer 2: 16 hidden -> 8 hidden neurons
        self.layer2 = nn.Linear(16, 8)

        # Layer 3: 8 hidden -> 1 output neuron (the harm probability)
        self.output_layer = nn.Linear(8, 1)

        # Sigmoid activation squashes output to [0, 1] range — probability!
        # Like Math.Clamp but learnable and smooth
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        forward() defines how data flows through the network.
        In C# terms, this is like calling the model's 'Predict()' method.
        x: tensor of shape [batch_size, 5]
        returns: tensor of shape [batch_size, 1]  (harm probability)
        """
        x = F.relu(self.layer1(x))         # layer1 then ReLU activation
        x = F.relu(self.layer2(x))         # layer2 then ReLU activation
        x = self.sigmoid(self.output_layer(x))    # output layer then sigmoid
        return x                            # return harm probability


# ---------------------------------------------------------------------------
# RevisionNet: takes [features, critique_score] -> revised_features
# ---------------------------------------------------------------------------
# Input:  6-dimensional vector = 5 response features + 1 critique score
# Output: 5-dimensional revised feature vector
# ---------------------------------------------------------------------------

class RevisionNet(nn.Module):
    """
    Neural network that takes response features (5 dims) + critique score (1 dim)
    = 6-dimensional input, and outputs revised feature values (5 dims).

    Architecture:
      Input(6) -> Linear(6->16) -> ReLU -> Linear(16->16) -> ReLU -> Linear(16->5) -> Sigmoid
    """
    def __init__(self):
        """
        Constructor: define layers for the revision network.
        """
        super().__init__()    # call parent class constructor

        # Layer 1: 6 inputs (5 features + 1 critique score) -> 16 hidden
        self.layer1 = nn.Linear(6, 16)

        # Layer 2: 16 hidden -> 16 hidden (same size, more capacity)
        self.layer2 = nn.Linear(16, 16)

        # Output: 16 hidden -> 5 outputs (one per feature dimension)
        self.output_layer = nn.Linear(16, 5)

        # Sigmoid keeps all output features in [0, 1] range
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, critique_score):
        """
        Forward pass for the revision network.
        features: tensor of shape [batch_size, 5]
        critique_score: tensor of shape [batch_size, 1]
        returns: tensor of shape [batch_size, 5]  (revised features)
        """
        # Concatenate features and critique_score along dimension 1
        # Like combining two arrays into one in C#: new[] { ...features, critique_score }
        x = torch.cat([features, critique_score], dim=1)    # shape: [batch, 6]

        x = F.relu(self.layer1(x))         # layer1 + ReLU
        x = F.relu(self.layer2(x))         # layer2 + ReLU
        x = self.sigmoid(self.output_layer(x))    # output + sigmoid -> [0,1] range

        return x    # return the revised feature vector

# ---------------------------------------------------------------------------
# Instantiate the networks
# ---------------------------------------------------------------------------
critique_net = CritiqueNet()     # create CritiqueNet instance
revision_net = RevisionNet()     # create RevisionNet instance

print("\n--- Network Architectures ---")
print(f"\n  CritiqueNet (harm predictor):\n{critique_net}")    # prints layer info
print(f"\n  RevisionNet (response improver):\n{revision_net}")

# Count parameters in each network
def count_params(model):
    """Count the total number of learnable parameters in a model."""
    # sum() over all parameters, p.numel() = number of elements in tensor
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n  CritiqueNet parameters: {count_params(critique_net):,}")
print(f"  RevisionNet parameters: {count_params(revision_net):,}")

# ---------------------------------------------------------------------------
# Simulate 5 critique-revise loops using the networks
# ---------------------------------------------------------------------------
print("\n--- Simulating 5 Neural Critique-Revise Loops ---")
print("  (networks are untrained, so results are random — shows the FLOW)")
print()

# Convert the initial bad response to a PyTorch tensor
# torch.from_numpy() converts NumPy array to PyTorch tensor
current_features_np = initial_response.copy()    # NumPy version for printing
current_features_t  = torch.from_numpy(current_features_np).unsqueeze(0)
# .unsqueeze(0) adds a batch dimension: shape [5] -> [1, 5]
# Because neural networks expect batches, even for a single sample

print(f"  Starting features: {current_features_np}")
print(f"  Tensor shape: {current_features_t.shape}  (batch_size=1, features=5)")
print()

alignment_scores_progress = []    # list to track alignment score each round

for loop_num in range(1, 6):    # range(1,6) = [1, 2, 3, 4, 5]
    # --- CritiqueNet: predict harm probability ---
    with torch.no_grad():    # no_grad() = don't track gradients (we're just inferring)
        harm_prob = critique_net(current_features_t)    # forward pass through CritiqueNet

    harm_prob_val = harm_prob.item()    # .item() extracts scalar value from tensor

    # --- RevisionNet: produce revised features ---
    with torch.no_grad():
        revised_t = revision_net(current_features_t, harm_prob)    # forward pass

    revised_np = revised_t.squeeze(0).numpy()    # remove batch dim, convert to NumPy
    # .squeeze(0) removes dimension 0: shape [1, 5] -> [5]
    # .numpy() converts tensor back to NumPy array

    # Compute alignment scores before and after
    score_before = alignment_score(current_features_np)
    score_after  = alignment_score(revised_np)

    alignment_scores_progress.append(score_after)    # record the new score

    print(f"  Loop {loop_num}:")
    print(f"    Harm probability predicted by CritiqueNet: {harm_prob_val:.4f}")
    print(f"    Alignment score: {score_before:.4f} -> {score_after:.4f}")

    # Use the revised features as input to the next loop
    current_features_np = revised_np             # update NumPy version
    current_features_t  = revised_t             # update tensor version

print(f"\n--- Alignment Score Progression Across 5 Loops ---")
for i, score in enumerate(alignment_scores_progress):
    bar_len = int(score * 30)                  # scale score to bar length
    bar = "#" * bar_len + "." * (30 - bar_len) # filled and empty portions
    print(f"  Loop {i+1}: {score:.4f}  |{bar}|")

# ---------------------------------------------------------------------------
# Note on training with REAL data
# ---------------------------------------------------------------------------
print("\n--- How These Networks Would Be Trained ---")
print("""
  TRAINING CritiqueNet:
    - Positive examples (label=1): responses that humans marked as harmful
    - Negative examples (label=0): responses that humans marked as safe
    - Loss function: Binary Cross-Entropy (BCE)
    - optimizer.step() updates weights so harmful responses score close to 1

  TRAINING RevisionNet:
    - Input pairs: (original_bad_features, critique_score)
    - Target: the features of the human-approved revision
    - Loss function: Mean Squared Error (MSE) on feature differences
    - optimizer.step() teaches the net HOW to fix harmful responses

  After training:
    - CritiqueNet replaces our hardcoded thresholds
    - RevisionNet replaces our hardcoded +/-0.3 adjustments
    - The whole pipeline becomes data-driven and generalizable!

  C# Analogy:
    Before training = hardcoded if/else rules in your service
    After training  = an ML.NET model that learned the same rules from data
""")

print("=" * 70)
print("EXAMPLE 05 COMPLETE — Constitutional AI simulated successfully!")
print("=" * 70)

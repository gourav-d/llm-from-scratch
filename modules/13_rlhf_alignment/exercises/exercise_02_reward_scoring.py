"""
=============================================================================
MODULE 13 - EXERCISE 02: Implementing Reward Model Scoring
=============================================================================

WHAT YOU WILL LEARN:
  - How a reward model assigns a scalar score to a response
  - The Bradley-Terry loss — the standard loss for training reward models
  - How sigmoid() converts raw scores into probabilities
  - How to check if a reward model is ranking preference pairs correctly
  - How to manually compute one gradient update step

C# ANALOGY:
  - A reward model is like a scoring function in a search engine:
    score = weights.Dot(features)   (higher score = better result)
  - Bradley-Terry loss is the training loss that pushes score(chosen) higher
    than score(rejected), similar to training a ranking model in Azure Search
  - One gradient step is like:  weights += learningRate * gradient
    just like how gradient descent works in ML.NET

=============================================================================

REWARD MODEL DIAGRAM
====================

  Response Features [5 dims]
           |
           v
    [Reward Model]  score = sigmoid( w . x + b )
           |
           v
    Scalar Score (0 to 1)  <- higher means "model thinks this is better"

  During training:
    Chosen response   -> score_chosen   (want this HIGH)
    Rejected response -> score_rejected (want this LOW)

    Bradley-Terry Loss = -log( sigmoid( score_chosen - score_rejected ) )
    Minimizing this loss = making score_chosen - score_rejected larger

=============================================================================
"""

# ---- GLOSSARY ---------------------------------------------------------------
GLOSSARY = {
    "Reward Model":
        "A model that takes a response and outputs a scalar quality score.",
    "Bradley-Terry Loss":
        "-log(sigmoid(score_chosen - score_rejected)) — trains reward models on pairs.",
    "Sigmoid":
        "A function that squashes any number to [0, 1]. sigmoid(x) = 1 / (1 + exp(-x)).",
    "Dot Product":
        "w . x = sum of element-wise products. Measures how aligned two vectors are.",
    "Gradient":
        "The direction and amount to change weights to reduce the loss.",
    "Gradient Step":
        "weights = weights - learning_rate * gradient  (one parameter update).",
    "Preference Accuracy":
        "Fraction of pairs where reward model correctly scores chosen > rejected.",
}

print("=" * 70)
print("EXERCISE 02 — Reward Model Scoring")
print("=" * 70)
print("\nGLOSSARY:")
for term, defn in GLOSSARY.items():
    print(f"  {term}:\n    {defn}\n")

import numpy as np    # NumPy — all maths done with this

# =============================================================================
# SHARED DATA
# =============================================================================
# Feature vectors [harm, honesty, helpfulness, manipulation, respect]
# Lower harm/manipulation = better. Higher others = better.

CHOSEN_FEATURES = np.array([
    [0.1, 0.9, 0.9, 0.1, 0.9],    # pair 1 chosen: excellent
    [0.2, 0.8, 0.8, 0.2, 0.8],    # pair 2 chosen: very good
    [0.3, 0.7, 0.7, 0.3, 0.7],    # pair 3 chosen: good
    [0.15, 0.85, 0.85, 0.15, 0.85],   # pair 4 chosen: very good
    [0.25, 0.75, 0.80, 0.20, 0.75],   # pair 5 chosen: good
], dtype=np.float32)    # 5 chosen responses, each with 5 features

REJECTED_FEATURES = np.array([
    [0.8, 0.3, 0.2, 0.7, 0.3],    # pair 1 rejected: very bad
    [0.6, 0.4, 0.4, 0.6, 0.4],    # pair 2 rejected: bad
    [0.7, 0.4, 0.3, 0.5, 0.4],    # pair 3 rejected: bad
    [0.5, 0.5, 0.5, 0.5, 0.5],    # pair 4 rejected: mediocre
    [0.55, 0.45, 0.4, 0.55, 0.45],    # pair 5 rejected: bad
], dtype=np.float32)    # 5 rejected responses

# Initial reward model weights (one weight per feature dimension)
# In a real model these would be learned from data — here we hardcode them
WEIGHTS = np.array(
    [-0.8, 0.6, 0.7, -0.7, 0.5],    # negative weights for bad features, positive for good
    dtype=np.float32
)
# Explanation:
#   harm       weight = -0.8  -> higher harm REDUCES score (punish harmful responses)
#   honesty    weight =  0.6  -> higher honesty INCREASES score (reward honest responses)
#   helpfulness weight = 0.7  -> higher helpfulness INCREASES score
#   manipulation weight = -0.7 -> higher manipulation REDUCES score
#   respect    weight =  0.5  -> higher respect INCREASES score

# =============================================================================
# EXERCISE 1 — Bradley-Terry Loss
# =============================================================================
# The Bradley-Terry model says: P(chosen beats rejected) = sigmoid(s_c - s_r)
# We want this probability to be HIGH (close to 1).
# Loss = -log(P(chosen beats rejected))
#      = -log(sigmoid(score_chosen - score_rejected))
#
# When score_chosen >> score_rejected: loss is near 0 (good!)
# When score_chosen ≈ score_rejected:  loss ≈ 0.693 (log(2))
# When score_chosen << score_rejected: loss is very large (bad!)
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 1 — Bradley-Terry Loss")
print("=" * 60)
print()
print("  Formula: loss = -log( sigmoid( score_chosen - score_rejected ) )")
print("  Hint: np.log() is log base e (natural log)")
print("  Hint: sigmoid(x) = 1.0 / (1.0 + np.exp(-x))")
print()

def sigmoid(x):
    """
    Compute sigmoid(x) = 1 / (1 + exp(-x)).
    C# equivalent: 1.0 / (1.0 + Math.Exp(-x))
    Squashes any value to the range (0, 1).
    """
    # TODO: implement sigmoid
    return None    # replace None with the formula

def bradley_terry_loss(score_chosen, score_rejected):
    """
    Compute the Bradley-Terry pairwise ranking loss.

    Parameters:
      score_chosen   : float — reward model score for the chosen response
      score_rejected : float — reward model score for the rejected response

    Returns:
      float — loss value (lower = model is doing better at this pair)
    """
    # TODO: Step 1 — compute the difference: score_chosen - score_rejected
    diff = None

    # TODO: Step 2 — apply sigmoid to the difference
    prob = None    # probability that chosen beats rejected

    # TODO: Step 3 — return -log(prob)
    # np.log() is the natural log (base e)
    # Add a small epsilon (1e-8) inside log to avoid log(0) = -infinity
    loss = None

    return loss

# Test with clear cases
print("  Testing bradley_terry_loss with clear cases:")
test_cases = [
    (0.9, 0.1, "chosen much better  -> expect SMALL loss"),
    (0.5, 0.5, "tied               -> expect loss ≈ 0.693"),
    (0.1, 0.9, "chosen much worse  -> expect LARGE loss"),
]
for s_c, s_r, description in test_cases:
    loss_val = bradley_terry_loss(s_c, s_r)
    print(f"    score_chosen={s_c}, score_rejected={s_r}  ->  loss={loss_val}  ({description})")

# =============================================================================
# EXERCISE 2 — Reward Model Forward Pass
# =============================================================================
# A simple linear reward model computes:
#   raw_score = dot(weights, features)  <- weighted sum of features
#   score = sigmoid(raw_score)           <- squash to [0, 1]
#
# C# equivalent: double score = Sigmoid(weights.Zip(features, (w,f) => w*f).Sum());
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 2 — Reward Model Forward Pass")
print("=" * 60)
print()
print("  Formula: score = sigmoid( dot(weights, features) )")
print("  Hint: np.dot(a, b) computes the dot product of two arrays")
print("  Hint: use your sigmoid() from Exercise 1")
print()

def reward_model_score(weights, features):
    """
    Compute the reward model's score for a single response.

    Parameters:
      weights  : np.array of shape [5] — model parameters
      features : np.array of shape [5] — response feature vector

    Returns:
      float in [0, 1] — higher means the model thinks this is a better response
    """
    # TODO: Step 1 — compute raw_score = dot(weights, features)
    raw_score = None

    # TODO: Step 2 — apply sigmoid to get score in [0, 1]
    score = None

    return float(score)    # return as Python float

# Test on chosen vs rejected features for pair 0
if CHOSEN_FEATURES is not None:
    score_c = reward_model_score(WEIGHTS, CHOSEN_FEATURES[0])
    score_r = reward_model_score(WEIGHTS, REJECTED_FEATURES[0])
    print(f"  Pair 0:")
    print(f"    Chosen   features: {CHOSEN_FEATURES[0]}  ->  score={score_c}")
    print(f"    Rejected features: {REJECTED_FEATURES[0]} ->  score={score_r}")
    if score_c is not None and score_r is not None:
        print(f"    Correct ranking: {score_c > score_r}  (chosen should score higher)")

# =============================================================================
# EXERCISE 3 — Preference Accuracy of the Reward Model
# =============================================================================
# Given the 5 preference pairs and our reward model (WEIGHTS),
# check what fraction of pairs the model ranks correctly.
# A pair is ranked correctly if score(chosen) > score(rejected).
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 3 — Reward Model Preference Accuracy")
print("=" * 60)
print()
print("  Task: for each of the 5 pairs, compute score for chosen and rejected,")
print("        then check if score_chosen > score_rejected.")
print("  Hint: loop over range(len(CHOSEN_FEATURES))")
print()

def reward_model_accuracy(weights, chosen_features_batch, rejected_features_batch):
    """
    Compute what fraction of preference pairs the reward model ranks correctly.

    Parameters:
      weights                  : np.array [5] — model weights
      chosen_features_batch    : np.array [N, 5] — features of N chosen responses
      rejected_features_batch  : np.array [N, 5] — features of N rejected responses

    Returns:
      accuracy : float in [0, 1]
      scores_c : list of float — scores for chosen responses
      scores_r : list of float — scores for rejected responses
    """
    scores_c = []    # list to hold chosen scores
    scores_r = []    # list to hold rejected scores

    # TODO: loop over all pairs and compute reward_model_score for each
    for i in range(len(chosen_features_batch)):
        sc = None    # TODO: reward_model_score(weights, chosen_features_batch[i])
        sr = None    # TODO: reward_model_score(weights, rejected_features_batch[i])
        scores_c.append(sc)    # add to list
        scores_r.append(sr)

    # TODO: convert to numpy arrays and count how many have sc > sr
    scores_c = np.array(scores_c)    # convert list to numpy array
    scores_r = np.array(scores_r)

    # TODO: compute accuracy
    accuracy = None    # replace with np.mean(scores_c > scores_r)

    return accuracy, scores_c, scores_r

# --- Run exercise 3 ---
acc, sc_all, sr_all = reward_model_accuracy(WEIGHTS, CHOSEN_FEATURES, REJECTED_FEATURES)
if acc is not None:
    print(f"  Reward model accuracy: {acc:.2f}  ({int(acc*5)}/5 pairs correct)")
    print(f"\n  Detailed breakdown:")
    for i in range(5):
        correct = sc_all[i] > sr_all[i] if sc_all[i] is not None else False
        mark = "CORRECT" if correct else "WRONG"
        print(f"    Pair {i+1}: chosen={sc_all[i]:.3f}  rejected={sr_all[i]:.3f}  [{mark}]")
else:
    print("  (implement reward_model_accuracy to see results)")

# =============================================================================
# EXERCISE 4 — One Manual Gradient Step
# =============================================================================
# We compute the gradient of the Bradley-Terry loss with respect to the weights,
# and update the weights in the direction that reduces the loss.
#
# Gradient derivation (simplified for linear reward model):
#   loss = -log( sigmoid( score_chosen - score_rejected ) )
#   d(loss)/d(weights) = -(1 - sigmoid(s_c - s_r)) * (features_chosen - features_rejected)
#
# Weight update:
#   new_weights = weights - learning_rate * gradient
#
# C# analogy:
#   weights -= learningRate * gradient;   (exactly the same formula!)
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 4 — One Manual Gradient Step")
print("=" * 60)
print()
print("  Formula:")
print("    diff = sigmoid(score_chosen - score_rejected)")
print("    gradient = -(1 - diff) * (features_chosen - features_rejected)")
print("    new_weights = weights - learning_rate * gradient")
print()
print("  Hint: after the step, the loss on this pair should DECREASE")
print()

LEARNING_RATE = 0.1    # how big a step to take — like in any gradient descent

def gradient_step(weights, features_chosen, features_rejected, learning_rate):
    """
    Perform one gradient descent step on the Bradley-Terry loss
    for a single preference pair.

    Parameters:
      weights           : np.array [5] — current model weights
      features_chosen   : np.array [5] — features of the chosen response
      features_rejected : np.array [5] — features of the rejected response
      learning_rate     : float — step size

    Returns:
      new_weights : np.array [5] — updated weights after one step
      loss_before : float — loss before the update
      loss_after  : float — loss after the update
    """
    # TODO: Step 1 — compute scores for chosen and rejected
    s_c = None    # reward_model_score(weights, features_chosen)
    s_r = None    # reward_model_score(weights, features_rejected)

    # TODO: Step 2 — compute loss BEFORE the update
    loss_before = None    # bradley_terry_loss(s_c, s_r)

    # TODO: Step 3 — compute the gradient
    # diff = sigmoid(s_c - s_r)  <-- but s_c and s_r are already sigmoided...
    # For simplicity, use raw dot products (before sigmoid) for the gradient
    raw_c = np.dot(weights, features_chosen)    # raw score before sigmoid
    raw_r = np.dot(weights, features_rejected)  # raw score before sigmoid
    diff = sigmoid(raw_c - raw_r)               # probability that chosen wins
    gradient = -(1.0 - diff) * (features_chosen - features_rejected)
    # Explanation:
    #   (1 - diff) = how uncertain the model is (close to 1 = very unsure)
    #   (features_chosen - features_rejected) = direction to push weights

    # TODO: Step 4 — update weights: new_weights = weights - lr * gradient
    new_weights = None

    # TODO: Step 5 — compute loss AFTER update to verify improvement
    s_c_new = None    # reward_model_score(new_weights, features_chosen)
    s_r_new = None    # reward_model_score(new_weights, features_rejected)
    loss_after = None    # bradley_terry_loss(s_c_new, s_r_new)

    return new_weights, loss_before, loss_after

# --- Run on pair 0 (hardest pair — biggest score difference expected) ---
updated_weights, loss_bef, loss_aft = gradient_step(
    WEIGHTS.copy(),           # copy so we don't modify the original
    CHOSEN_FEATURES[0],       # features of chosen response
    REJECTED_FEATURES[0],     # features of rejected response
    LEARNING_RATE
)
print(f"  One gradient step on pair 0:")
print(f"    Weights BEFORE: {WEIGHTS}")
if updated_weights is not None:
    print(f"    Weights AFTER:  {updated_weights}")
    print(f"    Loss BEFORE: {loss_bef:.4f}")
    print(f"    Loss AFTER:  {loss_aft:.4f}")
    if loss_bef is not None and loss_aft is not None:
        improved = "YES (loss decreased)" if loss_aft < loss_bef else "NO (something is wrong)"
        print(f"    Loss improved: {improved}")

# =============================================================================
# SOLUTIONS (read ONLY after attempting!)
# =============================================================================
"""
SOLUTION 1 — Sigmoid and Bradley-Terry Loss:

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def bradley_terry_loss(score_chosen, score_rejected):
    diff = score_chosen - score_rejected
    prob = sigmoid(diff)
    loss = -np.log(prob + 1e-8)
    return loss


SOLUTION 2 — Reward Model Forward Pass:

def reward_model_score(weights, features):
    raw_score = np.dot(weights, features)
    score = sigmoid(raw_score)
    return float(score)


SOLUTION 3 — Reward Model Accuracy:

def reward_model_accuracy(weights, chosen_features_batch, rejected_features_batch):
    scores_c = []
    scores_r = []
    for i in range(len(chosen_features_batch)):
        sc = reward_model_score(weights, chosen_features_batch[i])
        sr = reward_model_score(weights, rejected_features_batch[i])
        scores_c.append(sc)
        scores_r.append(sr)
    scores_c = np.array(scores_c)
    scores_r = np.array(scores_r)
    accuracy = np.mean(scores_c > scores_r)
    return accuracy, scores_c, scores_r


SOLUTION 4 — One Gradient Step:

def gradient_step(weights, features_chosen, features_rejected, learning_rate):
    s_c = reward_model_score(weights, features_chosen)
    s_r = reward_model_score(weights, features_rejected)
    loss_before = bradley_terry_loss(s_c, s_r)

    raw_c = np.dot(weights, features_chosen)
    raw_r = np.dot(weights, features_rejected)
    diff = sigmoid(raw_c - raw_r)
    gradient = -(1.0 - diff) * (features_chosen - features_rejected)

    new_weights = weights - learning_rate * gradient

    s_c_new = reward_model_score(new_weights, features_chosen)
    s_r_new = reward_model_score(new_weights, features_rejected)
    loss_after = bradley_terry_loss(s_c_new, s_r_new)

    return new_weights, loss_before, loss_after
"""

print("\n" + "=" * 70)
print("EXERCISE 02 COMPLETE — Check your output, then read the solutions!")
print("=" * 70)

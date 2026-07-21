"""
Module 06 - Training & Fine-Tuning
Exercise 03: The Training Loop

GLOSSARY
--------
Cross-Entropy Loss : Measures how wrong the model's predictions are.
                     Formula: loss = -log(probability of correct token)
                     Loss = 0 means perfect prediction.
                     Loss = 4.6 means model is essentially guessing randomly (vocab=100).
Perplexity         : exp(loss). Intuitive measure of model confusion.
                     Perplexity = 1  -> perfect (always correct).
                     Perplexity = V  -> random (V = vocabulary size).
                     Lower is always better.
Random Baseline    : If vocab_size = V, a random model has loss = log(V).
                     Perplexity of random model = V.
Training Loss      : Loss computed on the training data. Goes down as model learns.
Validation Loss    : Loss on unseen data (held-out set). Detects overfitting.
Overfitting        : Train loss falls but val loss rises. Model memorises, not learns.
Gradient           : Direction and magnitude of the weight update.
                     Tells us: "change this weight by this much."
Learning Rate      : Scales how big each gradient step is. Too large = diverge.
                     Too small = very slow learning.
Weight Update      : new_weight = old_weight - learning_rate * gradient
"""

import numpy as np   # NumPy for math operations

print("=" * 60)
print("Exercise 03: The Training Loop")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Cross-Entropy Loss for One Token
#
#  Background:
#    For a single position, cross-entropy loss is:
#      loss = -log(probability of the correct token)
#
#    The model outputs a probability for every token in the vocab.
#    We only care about the probability it assigned to the CORRECT one.
#
#    Examples:
#      Prob(correct) = 1.0 -> loss = -log(1.0) = 0.0  (perfect)
#      Prob(correct) = 0.5 -> loss = -log(0.5) = 0.69 (uncertain)
#      Prob(correct) = 0.1 -> loss = -log(0.1) = 2.30 (very wrong)
#
#  Your Task:
#    Write: token_cross_entropy(probs, correct_token_id) -> float
#    probs is a 1D probability array (sums to 1).
#    correct_token_id is the index of the right answer.
#
#  C# Analogy:
#    Like computing -Math.Log(predictionDict[correctLabel])
#    where predictionDict maps token -> probability.
# ============================================================

print("-" * 50)
print("EXERCISE 1: Cross-Entropy Loss for One Token")
print("-" * 50)
print()


def token_cross_entropy(probs, correct_token_id):
    """
    Compute cross-entropy loss for a single token prediction.

    Parameters:
        probs            (np.ndarray): 1D probability array summing to 1.
        correct_token_id (int)       : Index of the correct next token.

    Returns:
        float: Cross-entropy loss (-log prob of correct token).
    """
    # TODO:
    # 1. Get the probability the model assigned to the correct token
    #    correct_prob = probs[correct_token_id]
    # 2. Add a tiny epsilon (1e-10) to avoid log(0)
    # 3. Return -np.log(correct_prob + epsilon)
    pass  # Replace with your implementation


# Scenario A: model is very confident and correct
probs_good   = np.array([0.01, 0.02, 0.93, 0.02, 0.02])  # correct token = 2
correct_id_A = 2

# Scenario B: model is very uncertain
probs_bad    = np.array([0.20, 0.20, 0.20, 0.20, 0.20])  # correct token = 2
correct_id_B = 2

# Scenario C: model is confident but WRONG (predicts token 0, correct is 2)
probs_wrong  = np.array([0.90, 0.03, 0.03, 0.02, 0.02])  # correct token = 2
correct_id_C = 2

loss_A = token_cross_entropy(probs_good, correct_id_A)
loss_B = token_cross_entropy(probs_bad,  correct_id_B)
loss_C = token_cross_entropy(probs_wrong, correct_id_C)

if loss_A is not None:
    print(f"  Scenario A (confident & correct) : loss = {loss_A:.4f}  (expected ~0.07)")
    print(f"  Scenario B (completely uncertain): loss = {loss_B:.4f}  (expected ~1.61)")
    print(f"  Scenario C (confident & wrong)  : loss = {loss_C:.4f}  (expected ~3.51)")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Perplexity from Loss
#
#  Background:
#    Perplexity = exp(average cross-entropy loss)
#
#    Intuition:
#      Perplexity = "how many tokens would the model pick from if guessing?"
#      Perplexity = 1    -> always picks correct token (perfect)
#      Perplexity = 10   -> effectively choosing from 10 equally likely options
#      Perplexity = 100  -> essentially random (vocab_size = 100)
#
#    Random baseline:
#      If vocab_size = V, random model loss = log(V), perplexity = V.
#      Any trained model should have perplexity BELOW vocab_size.
#
#  Your Task:
#    Write: perplexity(loss) -> float
#    Returns exp(loss).
#
#    Also write: random_baseline_perplexity(vocab_size) -> float
#    Returns the perplexity a random model would achieve.
#
#  C# Analogy:
#    double perplexity = Math.Exp(averageLoss);
# ============================================================

print("-" * 50)
print("EXERCISE 2: Perplexity from Loss")
print("-" * 50)
print()


def perplexity(loss):
    """
    Compute perplexity from average cross-entropy loss.

    Parameters:
        loss (float): Average cross-entropy loss.

    Returns:
        float: Perplexity (exp of loss).
    """
    # TODO: return np.exp(loss)
    pass  # Replace with your implementation


def random_baseline_perplexity(vocab_size):
    """
    Perplexity of a random model with given vocabulary size.

    A random model assigns equal probability 1/vocab_size to each token.
    Its loss = log(vocab_size), so perplexity = exp(log(vocab_size)) = vocab_size.

    Parameters:
        vocab_size (int): Number of tokens in vocabulary.

    Returns:
        float: Perplexity of random model (equals vocab_size).
    """
    # TODO: return float(vocab_size)
    # Explanation: random baseline perplexity always equals vocab_size.
    pass  # Replace with your implementation


print("  Perplexity for different losses:")
print(f"  {'Loss':>8}  {'Perplexity':>12}")
print("  " + "-" * 24)
for loss_val in [0.0, 0.5, 1.0, 2.3, 4.6]:
    ppl = perplexity(loss_val)
    if ppl is not None:
        print(f"  {loss_val:>8.2f}  {ppl:>12.2f}")
print()

print("  Random baseline perplexity by vocab size:")
for vsize in [10, 100, 1000, 50257]:
    rbl = random_baseline_perplexity(vsize)
    if rbl is not None:
        print(f"    vocab_size={vsize:>6}  -> perplexity = {rbl:,.0f}")
print()
print("  Expected: trained models should have perplexity BELOW vocab_size.")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Average Loss Over a Batch
#
#  Background:
#    During training, we process multiple examples at once (a batch).
#    We compute loss for each token position across all examples,
#    then average over all positions and batch items.
#
#    Batch shape: (batch_size, seq_len)
#    Total positions = batch_size * seq_len
#    Average loss = sum of all token losses / total positions
#
#  Your Task:
#    Write: batch_loss(token_losses_2d) -> float
#    token_losses_2d is a 2D array of shape (batch_size, seq_len).
#    Each element is the cross-entropy loss for one token.
#    Returns the mean loss across all elements.
#
#  C# Analogy:
#    Like calling tokenLossMatrix.Cast<double>().Average() in LINQ.
# ============================================================

print("-" * 50)
print("EXERCISE 3: Average Loss Over a Batch")
print("-" * 50)
print()


def batch_loss(token_losses_2d):
    """
    Compute average cross-entropy loss over an entire batch.

    Parameters:
        token_losses_2d (np.ndarray): Shape (batch_size, seq_len).
                                      Each value is loss for one token position.

    Returns:
        float: Mean loss across all token positions in the batch.
    """
    # TODO: Use np.mean() to average all values in the 2D array.
    pass  # Replace with your implementation


np.random.seed(0)
# Simulate losses: early training (high loss ~2.3) vs later training (lower ~0.8)
early_losses = np.random.uniform(1.8, 2.8, size=(4, 16))   # batch=4, seq_len=16
later_losses = np.random.uniform(0.5, 1.2, size=(4, 16))

loss_early = batch_loss(early_losses)
loss_later = batch_loss(later_losses)

if loss_early is not None and loss_later is not None:
    print(f"  Early training loss (batch 4x16): {loss_early:.4f}")
    print(f"  Later training loss (batch 4x16): {loss_later:.4f}")
    print(f"  Improvement: {loss_early - loss_later:.4f}")
    print(f"  Model improved: {loss_later < loss_early}")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Detect Overfitting from Loss History
#
#  Background:
#    Overfitting is when the model memorises training data
#    instead of learning general patterns.
#
#    Signs:
#      - Training loss keeps DECREASING  <- good so far
#      - Validation loss starts INCREASING <- bad! model memorising
#
#    Rule (simple): if in last 3 epochs:
#      train_losses[-1] < train_losses[-3]   (still improving on train)
#      val_losses[-1]   > val_losses[-3]     (but getting worse on val)
#    -> Overfitting detected.
#
#  Your Task:
#    Write: is_overfitting(train_losses, val_losses) -> bool
#    Returns True if overfitting pattern is detected, False otherwise.
#    Guard: if fewer than 3 epochs recorded, return False.
#
#  C# Analogy:
#    Like a health monitor that raises an alert when server CPU drops
#    (train is running efficiently) but memory keeps climbing (memory leak).
# ============================================================

print("-" * 50)
print("EXERCISE 4: Detect Overfitting")
print("-" * 50)
print()


def is_overfitting(train_losses, val_losses):
    """
    Detect overfitting from training and validation loss histories.

    Parameters:
        train_losses (list): Loss on training set after each epoch.
        val_losses   (list): Loss on validation set after each epoch.

    Returns:
        bool: True if overfitting detected, False otherwise.
    """
    # TODO:
    # Guard: if len(train_losses) < 3: return False
    # Check: train_losses[-1] < train_losses[-3]  (still learning)
    #    AND  val_losses[-1]   > val_losses[-3]   (val is worsening)
    # If both conditions: return True, else return False
    pass  # Replace with your implementation


# Good run: both losses decreasing
train_good = [2.5, 2.0, 1.6, 1.3, 1.1]
val_good   = [2.6, 2.1, 1.8, 1.5, 1.3]

# Overfitting: train still improving, val getting worse
train_overfit = [2.5, 2.0, 1.6, 1.2, 0.9]
val_overfit   = [2.6, 2.1, 1.8, 2.0, 2.3]

# Not enough data
train_short = [2.5, 2.0]
val_short   = [2.6, 2.2]

r1 = is_overfitting(train_good, val_good)
r2 = is_overfitting(train_overfit, val_overfit)
r3 = is_overfitting(train_short, val_short)

if r1 is not None:
    print(f"  Good training   -> overfitting? {r1}  (expected: False)")
    print(f"  Overfitting run -> overfitting? {r2}  (expected: True)")
    print(f"  Only 2 epochs   -> overfitting? {r3}  (expected: False)")
print()

print("=" * 60)
print("All exercises complete!")
print("=" * 60)

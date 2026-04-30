"""
Module 12 - Fine-Tuning LLMs
Exercise 04: Training Loop Utilities

GLOSSARY
--------
Batch           : A small group of training examples processed together.
                  Like chunking a large list into pages of N items.
Generator       : A Python function that uses 'yield' to produce values lazily.
                  Like IEnumerable<T> in C# -- values are created on demand,
                  not all at once. Saves memory for large datasets.
Early Stopping  : Stop training automatically when improvement stops.
                  Like a circuit breaker: trip when the signal is bad.
Patience        : How many epochs of no improvement to tolerate before stopping.
                  Like a retry policy with max_attempts = patience.
Val Loss        : Validation loss -- the loss measured on held-out data
                  that the model never trained on.
Training Log    : A record of loss values per epoch during training.
Overfit Gap     : val_loss - train_loss. Large gap = overfitting.
Best Epoch      : The epoch with the lowest validation loss.
Learning Rate   : The step size for weight updates.
Warmup          : Linearly increasing the LR from 0 to max_lr for the
                  first N steps, before applying cosine decay.
                  Like slowly accelerating a car before reaching top speed.
Cosine Decay    : After warmup, LR follows a cosine curve from max to min.
                  Smooth and gradual, like cruise control easing off.
"""

import numpy as np   # NumPy for math operations
import math          # Python math library for cosine function

print("=" * 60)
print("Exercise 04: Training Loop Utilities")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Batch Data Generator
#
#  Background:
#    When training on large datasets, you cannot load everything at once.
#    Instead you process "batches" -- small chunks of examples.
#    A Python generator uses 'yield' to produce batches one at a time.
#    The training loop iterates over it: "for batch in batch_data(...):"
#
#  Your Task:
#    Write: batch_data(dataset, batch_size) generator
#    Yields successive non-overlapping chunks of the dataset.
#    If the dataset doesn't divide evenly, the last batch is smaller.
#    Example: dataset of 10 items, batch_size=3 -> yields [0:3],[3:6],[6:9],[9:10]
#
#  C# Analogy:
#    Like this LINQ extension:
#      public static IEnumerable<List<T>> Batch<T>(IEnumerable<T> source, int size)
#      {
#          var batch = new List<T>();
#          foreach (var item in source) {
#              batch.Add(item);
#              if (batch.Count == size) { yield return batch; batch = new List<T>(); }
#          }
#          if (batch.Count > 0) yield return batch;
#      }
#
#  Python 'yield' makes a function into a generator.
#  Calling batch_data(...) returns an iterable without executing any code yet.
#  Each 'for' iteration runs until the next 'yield'.
# ============================================================

print("-" * 50)
print("EXERCISE 1: Batch Data Generator")
print("-" * 50)
print()

def batch_data(dataset, batch_size):
    """
    Yield successive batches from a dataset list.

    Parameters:
        dataset    (list): All training examples.
        batch_size (int) : Number of examples per batch.

    Yields:
        list: A batch (sub-list) of up to batch_size examples.
    """
    # TODO: Implement this generator.
    # Loop from index 0 to len(dataset), stepping by batch_size.
    # Yield dataset[i : i + batch_size] at each step.
    # Hint: use 'for i in range(0, len(dataset), batch_size):'
    pass   # Replace with your implementation


# Test the generator
sample_dataset = list(range(10))            # [0, 1, 2, 3, ..., 9]

print(f"  Dataset: {sample_dataset}")
print(f"  batch_size=3:")

try:
    batches = list(batch_data(sample_dataset, 3))   # Materialise all batches into a list
    if batches:
        for i, b in enumerate(batches):
            print(f"    Batch {i+1}: {b}")
        print(f"  Total batches: {len(batches)}  (expected 4: [0-2],[3-5],[6-8],[9])")
    else:
        print("  (Function returned empty -- not implemented yet)")
except TypeError:
    print("  (Function not implemented yet -- returns None instead of a generator)")
print()

# Real training example
training_examples = [
    {"text": "great product", "label": 2},
    {"text": "terrible",      "label": 0},
    {"text": "it is okay",    "label": 1},
    {"text": "love it",       "label": 2},
    {"text": "not good",      "label": 0},
]
print(f"  Real examples with batch_size=2:")
try:
    for batch in batch_data(training_examples, 2):
        texts = [item["text"] for item in batch]    # Extract text from each dict
        print(f"    Batch: {texts}")
except TypeError:
    print("  (Function not implemented yet)")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Early Stopping Class
#
#  Background:
#    After each epoch, we check whether validation loss improved.
#    If it hasn't improved for 'patience' epochs in a row, stop training.
#    This prevents overfitting and wasted compute.
#
#  Your Task:
#    Complete the EarlyStopper class with:
#      __init__(self, patience): set self.patience, self.best_loss=inf, self.counter=0
#      __call__(self, val_loss) -> bool: return True when should stop
#      reset(self): reset best_loss and counter (use when starting a new run)
#
#    Logic for __call__:
#      - If val_loss < self.best_loss: update best_loss, reset counter
#      - Else: increment counter
#      - If counter >= patience: return True (stop!)
#      - Otherwise: return False (keep going)
#
#  C# Analogy:
#    Like a Polly retry policy, but inverted:
#      - You "succeed" (counter reset) when loss improves.
#      - You "fail" (counter++) when loss doesn't improve.
#      - After patience failures in a row, circuit trips (return True).
# ============================================================

print("-" * 50)
print("EXERCISE 2: Early Stopping Class")
print("-" * 50)
print()

class EarlyStopper:
    """
    Monitors validation loss and signals when training should stop.
    Call instance like a function after each epoch: should_stop = stopper(val_loss)
    """

    def __init__(self, patience):
        """
        Parameters:
            patience (int): Number of consecutive non-improving epochs before stopping.
        """
        # TODO: Set self.patience, self.best_loss = float('inf'), self.counter = 0
        pass   # Replace with your implementation

    def __call__(self, val_loss):
        """
        Check if training should stop.

        Parameters:
            val_loss (float): Current epoch's validation loss.

        Returns:
            bool: True if training should stop, False to continue.
        """
        # TODO: Implement the early stopping logic.
        # If val_loss < self.best_loss:
        #     update best_loss, reset counter, return False
        # Else:
        #     increment counter
        #     if counter >= patience: return True
        #     else: return False
        pass   # Replace with your implementation

    def reset(self):
        """Reset the stopper state for a new training run."""
        # TODO: Reset self.best_loss to float('inf') and self.counter to 0
        pass   # Replace with your implementation


# Test EarlyStopper
stopper = EarlyStopper(patience=3)

# Simulate a training run where loss first improves then plateaus
fake_val_losses = [1.0, 0.8, 0.7, 0.75, 0.78, 0.82]   # Improves then worsens

print(f"  Val losses: {fake_val_losses}")
print(f"  Patience  : 3")
print()
print(f"  {'Epoch':>6}  {'Val Loss':>10}  {'Stop?':>8}  {'Counter':>8}")
print("  " + "-" * 40)

for epoch, val_loss in enumerate(fake_val_losses, start=1):
    try:
        should_stop = stopper(val_loss)          # Check if we should stop
        counter = stopper.counter if hasattr(stopper, 'counter') else "?"
        print(f"  {epoch:>6}  {val_loss:>10.2f}  {str(should_stop):>8}  {counter:>8}")
        if should_stop:
            print(f"  --> Stopped at epoch {epoch}")
            break
    except (TypeError, AttributeError):
        print(f"  (Not implemented yet)")
        break
print()
print("  Expected: stop at epoch 6 (3 consecutive non-improvements after epoch 3)")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Training Log Analyser
#
#  Background:
#    After training, you want to understand what happened:
#      - Did the model overfit?
#      - When was the best checkpoint?
#      - Is the model still underfitting?
#
#  Your Task:
#    Write: training_log_analyzer(train_losses, val_losses) -> dict
#    Returns a dict with:
#      "overfit_epoch" (int or None): First epoch where gap = val-train > 0.3
#                                     Epoch numbering starts at 1. None if no overfit.
#      "best_epoch"    (int)        : Epoch with the lowest validation loss (1-indexed).
#      "recommendation" (str)       : One of "good", "overfitting", "underfitting",
#                                     "not_learning"
#
#    Recommendation rules:
#      - "not_learning"  : Final train loss > 0.9 * initial train loss (loss barely moved)
#      - "underfitting"  : Final train loss > 0.5 (loss still high after training)
#      - "overfitting"   : overfit_epoch is not None (detected gap > 0.3)
#      - "good"          : None of the above
#    Apply rules in the order listed above (first match wins).
#
#  C# Analogy:
#    Like a code analysis tool that inspects a build log and returns
#    a list of warnings and a summary verdict.
# ============================================================

print("-" * 50)
print("EXERCISE 3: Training Log Analyser")
print("-" * 50)
print()

def training_log_analyzer(train_losses, val_losses):
    """
    Analyse training and validation loss curves.

    Parameters:
        train_losses (list[float]): Training loss per epoch.
        val_losses   (list[float]): Validation loss per epoch.

    Returns:
        dict: {
            "overfit_epoch"   : int or None,
            "best_epoch"      : int,
            "recommendation"  : str,
        }
    """
    # TODO: Implement this function.
    # Step 1: Find overfit_epoch (first epoch where val-train > 0.3)
    #         Epoch index is 1-based: epoch 1 = index 0.
    # Step 2: Find best_epoch (index of min(val_losses) + 1)
    # Step 3: Determine recommendation using the rules above.
    pass   # Replace with your implementation


# Test cases
good_log = {
    "train": [1.2, 0.9, 0.6, 0.4, 0.25],
    "val"  : [1.3, 1.0, 0.7, 0.5, 0.35],
}
overfit_log = {
    "train": [1.2, 0.8, 0.4, 0.2, 0.1],
    "val"  : [1.3, 1.0, 0.9, 1.1, 1.4],
}
underfit_log = {
    "train": [1.2, 1.1, 1.0, 0.9, 0.8],    # Still high after training
    "val"  : [1.3, 1.2, 1.1, 1.0, 0.9],
}
not_learning_log = {
    "train": [1.2, 1.19, 1.18, 1.17, 1.16],  # Barely changed
    "val"  : [1.3, 1.29, 1.28, 1.27, 1.26],
}

test_cases = [
    ("Good training",  good_log),
    ("Overfitting",    overfit_log),
    ("Underfitting",   underfit_log),
    ("Not learning",   not_learning_log),
]

for name, log in test_cases:
    result = training_log_analyzer(log["train"], log["val"])
    if result:
        print(f"  {name}:")
        print(f"    overfit_epoch  : {result.get('overfit_epoch')}")
        print(f"    best_epoch     : {result.get('best_epoch')}")
        print(f"    recommendation : {result.get('recommendation')}")
    else:
        print(f"  {name}: (not implemented yet)")
    print()


# ============================================================
#  EXERCISE 4
#  Topic: Cosine LR Scheduler with Warmup
#
#  Background:
#    A fixed learning rate is rarely optimal.
#    Cosine decay with warmup is a popular and effective schedule:
#
#    Phase 1 (warmup, steps 0..warmup_steps):
#      LR increases linearly from 0 to max_lr.
#      LR = max_lr * (step / warmup_steps)
#
#    Phase 2 (decay, steps warmup_steps..total_steps):
#      LR follows a cosine curve from max_lr down to near 0.
#      progress = (step - warmup_steps) / (total_steps - warmup_steps)
#      LR = 0.5 * max_lr * (1 + cos(pi * progress))
#
#    cos(0) = 1 -> LR = max_lr at start of decay
#    cos(pi) = -1 -> LR = 0 at end of decay
#
#  Your Task:
#    Write: compute_cosine_lr(step, total_steps, max_lr, warmup_steps) -> float
#    Print LR values for steps: 0, 50, 100, 200, 500, 1000
#    Given: total_steps=1000, warmup_steps=100, max_lr=1e-4
#
#  C# Analogy:
#    Like a timer-based config value that changes according to a schedule --
#    starts low, ramps up (warmup), then gradually reduces (decay).
# ============================================================

print("-" * 50)
print("EXERCISE 4: Cosine LR Scheduler with Warmup")
print("-" * 50)
print()

def compute_cosine_lr(step, total_steps, max_lr, warmup_steps):
    """
    Compute the learning rate at a given training step.

    Phase 1 (step < warmup_steps): linear warmup from 0 to max_lr.
    Phase 2 (step >= warmup_steps): cosine decay from max_lr to 0.

    Parameters:
        step         (int)  : Current training step (0-indexed).
        total_steps  (int)  : Total number of training steps.
        max_lr       (float): Peak learning rate after warmup.
        warmup_steps (int)  : Number of warmup steps.

    Returns:
        float: Learning rate for this step.
    """
    # TODO: Implement the two-phase LR schedule.
    #
    # Phase 1 (warmup):
    #   if step < warmup_steps:
    #       return max_lr * (step / warmup_steps)
    #
    # Phase 2 (cosine decay):
    #   progress = (step - warmup_steps) / (total_steps - warmup_steps)
    #   return 0.5 * max_lr * (1.0 + math.cos(math.pi * progress))
    pass   # Replace with your implementation


# Test parameters
TOTAL_STEPS  = 1000                 # Total training steps
WARMUP_STEPS = 100                  # Steps for linear warmup
MAX_LR       = 1e-4                 # Peak learning rate (0.0001)
TEST_STEPS   = [0, 50, 100, 200, 500, 1000]  # Steps to sample

print(f"  total_steps={TOTAL_STEPS}, warmup_steps={WARMUP_STEPS}, max_lr={MAX_LR}")
print()
print(f"  {'Step':>6}  {'LR':>12}  Phase")
print("  " + "-" * 32)

for step in TEST_STEPS:
    lr = compute_cosine_lr(step, TOTAL_STEPS, MAX_LR, WARMUP_STEPS)
    if lr is not None:
        phase = "warmup" if step < WARMUP_STEPS else "cosine decay"
        print(f"  {step:>6}  {lr:>12.8f}  {phase}")
    else:
        print(f"  {step:>6}  (not implemented)")

print()
print("  Expected pattern:")
print("    Step   0: LR = 0.00000000  (start of warmup)")
print("    Step  50: LR = 0.00005000  (halfway through warmup)")
print("    Step 100: LR = 0.00010000  (peak LR, end of warmup)")
print("    Step 200: LR ~= 0.00009045  (cosine starts dropping)")
print("    Step 500: LR = 0.00005000  (halfway through cosine)")
print("    Step1000: LR = 0.00000000  (end of decay)")
print()


# ============================================================
#  SOLUTIONS  (commented out -- try it yourself first!)
# ============================================================

"""
# ---- SOLUTION: Exercise 1 ----

def batch_data(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):  # Step through by batch_size
        yield dataset[i : i + batch_size]          # Yield one batch at a time


# ---- SOLUTION: Exercise 2 ----

class EarlyStopper:
    def __init__(self, patience):
        self.patience  = patience       # Max consecutive non-improving epochs
        self.best_loss = float('inf')   # Best validation loss seen so far
        self.counter   = 0              # How many non-improving epochs in a row

    def __call__(self, val_loss):
        if val_loss < self.best_loss:   # Improvement detected
            self.best_loss = val_loss   # Update best
            self.counter   = 0         # Reset streak
            return False               # Don't stop
        else:
            self.counter += 1          # Another non-improving epoch
            if self.counter >= self.patience:
                return True            # Stop!
            return False               # Still patient

    def reset(self):
        self.best_loss = float('inf')  # Reset best loss
        self.counter   = 0             # Reset counter


# ---- SOLUTION: Exercise 3 ----

def training_log_analyzer(train_losses, val_losses):
    # Find first overfit epoch (gap > 0.3)
    overfit_epoch = None
    for i in range(len(train_losses)):
        gap = val_losses[i] - train_losses[i]   # Val - Train gap
        if gap > 0.3:
            overfit_epoch = i + 1               # 1-based epoch number
            break

    # Find best epoch (lowest val loss)
    best_epoch = int(np.argmin(val_losses)) + 1  # 1-based

    # Recommendation (first matching rule wins)
    initial_train = train_losses[0]
    final_train   = train_losses[-1]

    if final_train > 0.9 * initial_train:
        recommendation = "not_learning"
    elif final_train > 0.5:
        recommendation = "underfitting"
    elif overfit_epoch is not None:
        recommendation = "overfitting"
    else:
        recommendation = "good"

    return {
        "overfit_epoch"  : overfit_epoch,
        "best_epoch"     : best_epoch,
        "recommendation" : recommendation,
    }


# ---- SOLUTION: Exercise 4 ----

def compute_cosine_lr(step, total_steps, max_lr, warmup_steps):
    if step < warmup_steps:
        # Linear warmup: LR grows from 0 to max_lr
        return max_lr * (step / warmup_steps)
    else:
        # Cosine decay: LR falls from max_lr to 0
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * max_lr * (1.0 + math.cos(math.pi * progress))
"""

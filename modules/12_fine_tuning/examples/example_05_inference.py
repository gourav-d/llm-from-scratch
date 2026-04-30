"""
Module 12 - Fine-Tuning LLMs
Example 05: Inference - Comparing Base Model vs Fine-Tuned Model

GLOSSARY
--------
Inference      : Using a trained model to make predictions on new data.
                 Like calling a deployed API -- just input in, prediction out.
Base Model     : A model with random (untrained) weights.
                 Like a brand-new employee with no training -- random guesses.
Fine-Tuned     : A model that has been trained on task-specific labelled data.
                 Like the same employee after 2 weeks of on-the-job training.
Accuracy       : % of predictions that were correct.
                 Like a unit test pass rate: correct / total * 100.
F1 Score       : Balance of Precision and Recall. Good for imbalanced datasets.
                 Harmonic mean: 2 * (P * R) / (P + R).
Precision      : Of all "POSITIVE" predictions, how many were actually positive?
                 Like: of all emails flagged as spam, how many really were spam?
Recall         : Of all actual POSITIVE examples, how many did we catch?
                 Like: of all actual spam emails, how many did we flag?
Confusion Matrix: A table showing predicted vs actual class for every class.
                 Rows = actual class, Columns = predicted class.
                 Diagonal = correct predictions.
A/B Comparison : Running two models on the same inputs and comparing outputs.
                 Like a feature flag: 50% traffic to old model, 50% to new.
"""

import numpy as np        # NumPy for all math operations
import random             # Python random for shuffling

# ============================================================
#  SHARED SETUP  -  Dataset and Helpers
# ============================================================

# Labels: 0=NEGATIVE, 1=NEUTRAL, 2=POSITIVE
# These are the same examples used in example_04 plus extras.
LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]  # Human-readable label names

# 10 held-out test examples (not used during training)
TEST_DATA = [
    ("fantastic product absolutely love it",         2),   # POSITIVE
    ("does the job nothing to complain about",       1),   # NEUTRAL
    ("broken on arrival total waste",                0),   # NEGATIVE
    ("pretty good for everyday use",                 1),   # NEUTRAL
    ("outstanding quality highly recommend",         2),   # POSITIVE
    ("arrived damaged very disappointed",            0),   # NEGATIVE
    ("works as described nothing more",              1),   # NEUTRAL
    ("exceeded all my expectations wonderful",       2),   # POSITIVE
    ("cheap material falls apart quickly",           0),   # NEGATIVE
    ("solid reliable product good value",            2),   # POSITIVE
]

# Training data used to fine-tune (same as example_04)
TRAIN_DATA = [
    ("great product love it",           2),
    ("amazing quality very happy",      2),
    ("best purchase i ever made",       2),
    ("fantastic works perfectly",       2),
    ("really excellent would buy again",2),
    ("it is okay nothing special",      1),
    ("average quality as expected",     1),
    ("fine for the price",              1),
    ("does what it should",             1),
    ("not bad not great",               1),
    ("terrible waste of money",         0),
    ("broke after one day useless",     0),
    ("worst product ever disappointed", 0),
    ("poor quality do not buy",         0),
    ("horrible experience regret it",   0),
]

# ---- Vocabulary and vectorisation --------------------------------

def build_vocab(data):
    """Build word-to-index dictionary from a list of (text, label) pairs."""
    vocab = {}                              # Empty dictionary, like Dictionary<string,int>
    idx   = 0                               # Next available index
    for text, _ in data:                   # Iterate over all examples
        for word in text.split():          # Split sentence into words
            if word not in vocab:          # Only add new words
                vocab[word] = idx          # Assign an integer ID to the word
                idx += 1
    return vocab

ALL_DATA  = TRAIN_DATA + TEST_DATA          # Combine to build a complete vocabulary
VOCAB     = build_vocab(ALL_DATA)           # Build vocabulary from all examples
VOCAB_SIZE = len(VOCAB)                     # Total number of unique words
NUM_CLASSES = 3                             # NEGATIVE, NEUTRAL, POSITIVE

def text_to_vector(text, vocab, size):
    """Convert a sentence string into a bag-of-words NumPy vector."""
    vec = np.zeros(size)                    # All zeros initially
    for word in text.split():              # Each word in the sentence
        if word in vocab:                  # Only known words
            vec[vocab[word]] = 1.0         # Mark that word's position as present
    return vec

# ============================================================
#  PART A  -  NumPy Classifiers
# ============================================================

class SimpleClassifier:
    """
    Tiny 2-layer MLP classifier (NumPy).
    Can be used as either a random "base" model or a trained "fine-tuned" model.
    Like a class in C# that can represent both a default-config service
    and a properly configured one.
    """

    def __init__(self, input_size, hidden_size, output_size, seed=None):
        """Initialise weights. If seed given, reproducible randomness."""
        if seed is not None:
            np.random.seed(seed)            # Fix seed for reproducibility
        scale1 = np.sqrt(2.0 / input_size)  # Xavier init scale
        self.W1 = np.random.randn(hidden_size, input_size) * scale1
        self.b1 = np.zeros(hidden_size)     # Hidden layer bias
        scale2 = np.sqrt(2.0 / hidden_size)
        self.W2 = np.random.randn(output_size, hidden_size) * scale2
        self.b2 = np.zeros(output_size)     # Output layer bias

    def relu(self, x):
        """ReLU: replace negatives with 0."""
        return np.maximum(0.0, x)

    def softmax(self, x):
        """Convert scores to probabilities."""
        x = x - np.max(x)                  # Numerical stability trick
        e = np.exp(x)
        return e / np.sum(e)

    def forward(self, x):
        """Run input through the model and return class probabilities."""
        z1    = self.W1 @ x + self.b1      # First linear layer
        a1    = self.relu(z1)              # ReLU activation
        z2    = self.W2 @ a1 + self.b2    # Second linear layer
        probs = self.softmax(z2)           # Softmax probabilities
        return probs

    def predict(self, x):
        """Return the predicted class index (0, 1, or 2)."""
        return int(np.argmax(self.forward(x)))  # Index of highest probability

    def compute_loss(self, probs, true_label):
        """Cross-entropy loss for one example."""
        p = max(probs[true_label], 1e-12)  # Avoid log(0)
        return -np.log(p)

    def train_one_epoch(self, data, lr=0.05):
        """
        Train on a list of (text, label) examples for one full epoch.
        Returns average loss for the epoch.
        """
        total_loss = 0.0                    # Accumulate loss
        random.shuffle(data)               # Shuffle order before each pass
        for text, label in data:
            x         = text_to_vector(text, VOCAB, VOCAB_SIZE)  # Text -> vector
            probs     = self.forward(x)     # Forward pass

            # --- Backpropagation (same as example_04) ---
            d_z2      = probs.copy()        # Gradient of loss w.r.t. output scores
            d_z2[label] -= 1.0             # Adjust for correct class

            z1        = self.W1 @ x + self.b1
            a1        = self.relu(z1)

            d_W2      = np.outer(d_z2, a1)
            d_b2      = d_z2
            d_a1      = self.W2.T @ d_z2
            d_z1      = d_a1 * (z1 > 0).astype(float)
            d_W1      = np.outer(d_z1, x)
            d_b1      = d_z1

            self.W1  -= lr * d_W1           # Update weights
            self.b1  -= lr * d_b1
            self.W2  -= lr * d_W2
            self.b2  -= lr * d_b2

            total_loss += self.compute_loss(probs, label)

        return total_loss / len(data)       # Return average loss

    def fit(self, data, epochs=20, lr=0.05):
        """Train the model for multiple epochs. Like calling model.Fit() in ML.NET."""
        for _ in range(epochs):            # Loop over epochs
            self.train_one_epoch(data, lr)  # Train one epoch (discard loss here)

# ---- Build base model (random, untrained) -------------------------

print("=" * 60)
print("PART A: NumPy Classifier -- Base vs Fine-Tuned")
print("=" * 60)
print()

HIDDEN = 16                                 # Hidden layer size

# Base model: random weights, never trained on any data.
# This is the "before fine-tuning" model.
base_model_a = SimpleClassifier(VOCAB_SIZE, HIDDEN, NUM_CLASSES, seed=99)
print("  Base model created (random weights, not trained).")

# Fine-tuned model: same architecture, but trained on TRAIN_DATA.
# This is the "after fine-tuning" model.
finetuned_model_a = SimpleClassifier(VOCAB_SIZE, HIDDEN, NUM_CLASSES, seed=99)
finetuned_model_a.fit(TRAIN_DATA, epochs=25, lr=0.05)   # Actually train it
print("  Fine-tuned model trained on 15 examples for 25 epochs.")
print()

# ---- Run both models on 10 test examples -------------------------

# Print side-by-side comparison table.
# Like a diff view: columns = Input | Base Pred | Tuned Pred | True Label
COL_W = 40                                  # Column width for the input text column

print("  " + "-" * (COL_W + 42))
print(f"  {'Input Text':<{COL_W}} {'Base':>10} {'Tuned':>10} {'True':>10}")
print("  " + "-" * (COL_W + 42))

base_correct_a   = 0                        # Count correct predictions from base model
tuned_correct_a  = 0                        # Count correct predictions from tuned model

for text, true_label in TEST_DATA:
    x = text_to_vector(text, VOCAB, VOCAB_SIZE)          # Convert text to vector

    base_pred   = base_model_a.predict(x)                # Base model prediction
    tuned_pred  = finetuned_model_a.predict(x)           # Fine-tuned prediction

    base_name   = LABEL_NAMES[base_pred]                 # Human-readable name
    tuned_name  = LABEL_NAMES[tuned_pred]
    true_name   = LABEL_NAMES[true_label]

    if base_pred  == true_label: base_correct_a  += 1   # Count correct
    if tuned_pred == true_label: tuned_correct_a += 1

    # Truncate long text for display
    display_text = (text[:COL_W - 3] + "...") if len(text) > COL_W else text
    print(f"  {display_text:<{COL_W}} {base_name:>10} {tuned_name:>10} {true_name:>10}")

print("  " + "-" * (COL_W + 42))
print()

# Print accuracy summary -- the key "BEFORE vs AFTER" comparison
base_acc_a   = base_correct_a  / len(TEST_DATA) * 100
tuned_acc_a  = tuned_correct_a / len(TEST_DATA) * 100

print(f"  BEFORE fine-tuning : {base_acc_a:.0f}% accuracy "
      f"({base_correct_a}/{len(TEST_DATA)} correct)")
print(f"  AFTER  fine-tuning : {tuned_acc_a:.0f}% accuracy "
      f"({tuned_correct_a}/{len(TEST_DATA)} correct)")
improvement_a = tuned_acc_a - base_acc_a                 # How much better?
print(f"  Improvement        : +{improvement_a:.0f} percentage points")
print()

# ============================================================
#  PART B  -  PyTorch Classifiers with F1 + Confusion Matrix
# ============================================================

try:
    import torch                             # PyTorch deep learning framework
    import torch.nn as nn                    # Neural network layers

    print("=" * 60)
    print("PART B: PyTorch Classifier -- Base vs Fine-Tuned")
    print("=" * 60)
    print()

    # ---- B1. Build tensors from data -------------------------

    def make_tensors(data, vocab, size):
        """Convert (text, label) list to PyTorch float tensor X and long tensor y."""
        X = torch.tensor(
            np.array([text_to_vector(t, vocab, size) for t, _ in data]),
            dtype=torch.float32                # Float tensor for model input
        )
        y = torch.tensor([lbl for _, lbl in data], dtype=torch.long)  # Long for CrossEntropy
        return X, y

    X_train_b, y_train_b = make_tensors(TRAIN_DATA, VOCAB, VOCAB_SIZE)  # Training tensors
    X_test_b,  y_test_b  = make_tensors(TEST_DATA,  VOCAB, VOCAB_SIZE)  # Test tensors

    # ---- B2. Model factory function --------------------------

    def make_model(seed=None):
        """Create a fresh 2-layer MLP model. Uses manual weight init if seed given."""
        torch.manual_seed(seed if seed is not None else 0)  # Fix randomness
        m = nn.Sequential(
            nn.Linear(VOCAB_SIZE, HIDDEN),   # Input -> Hidden
            nn.ReLU(),                       # Activation
            nn.Linear(HIDDEN, NUM_CLASSES),  # Hidden -> Output
        )
        return m                             # Return the model

    # ---- B3. Train fine-tuned model --------------------------

    def train_model_b(model, X, y, epochs=30, lr=1e-2):
        """Train a PyTorch model. Returns the trained model."""
        criterion = nn.CrossEntropyLoss()    # Loss function (softmax + log + negative)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimiser
        model.train()                        # Enable training mode
        for _ in range(epochs):             # Loop over epochs
            optimizer.zero_grad()           # Clear old gradients
            loss = criterion(model(X), y)   # Forward pass + compute loss
            loss.backward()                 # Backward pass (compute gradients)
            optimizer.step()                # Update weights
        return model                         # Return trained model

    # Base model: random weights, not trained at all
    base_model_b = make_model(seed=7)
    print("  Base model created (random weights).")

    # Fine-tuned model: same arch, actually trained
    finetuned_model_b = make_model(seed=7)
    finetuned_model_b = train_model_b(finetuned_model_b, X_train_b, y_train_b,
                                      epochs=30, lr=1e-2)
    print("  Fine-tuned model trained.")
    print()

    # ---- B4. Evaluate both models ----------------------------

    def evaluate_model(model, X, y):
        """
        Run model on X and compare to y.
        Returns: accuracy, per-class F1 scores, and confusion matrix.
        """
        model.eval()                         # Evaluation mode
        with torch.no_grad():                # No gradient tracking needed
            logits = model(X)                # Forward pass
            preds  = torch.argmax(logits, dim=1)  # Predicted classes

        preds_np = preds.numpy()             # Convert to NumPy for calculations
        true_np  = y.numpy()

        # -- Accuracy --
        accuracy = (preds_np == true_np).mean() * 100   # % correct

        # -- Confusion matrix (3x3) --
        # Row = actual, Column = predicted
        # In C# terms: like a 2D int[3,3] where cm[actual][pred] is incremented.
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)  # 3x3 zero matrix
        for t, p in zip(true_np, preds_np):  # Zip true and predicted together
            cm[t][p] += 1                    # Increment the correct cell

        # -- Per-class F1 --
        f1_scores = []                       # Store F1 for each class
        for cls in range(NUM_CLASSES):
            tp = cm[cls][cls]                # True Positives: correct predictions for cls
            fp = cm[:, cls].sum() - tp       # False Positives: others predicted as cls
            fn = cm[cls, :].sum() - tp       # False Negatives: cls predicted as other

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # TP / (TP + FP)
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # TP / (TP + FN)
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            f1_scores.append(f1)             # Add to list

        macro_f1 = np.mean(f1_scores)        # Average F1 across all classes

        return accuracy, macro_f1, f1_scores, cm, preds_np

    # Run evaluation on both models
    base_acc_b,  base_f1_b,  base_f1s_b,  base_cm_b,  base_preds_b  = (
        evaluate_model(base_model_b,      X_test_b, y_test_b))
    tuned_acc_b, tuned_f1_b, tuned_f1s_b, tuned_cm_b, tuned_preds_b = (
        evaluate_model(finetuned_model_b, X_test_b, y_test_b))

    # ---- B5. Print side-by-side comparison table -------------

    print(f"  {'Input Text':<{COL_W}} {'Base':>10} {'Tuned':>10} {'True':>10}")
    print("  " + "-" * (COL_W + 42))

    for i, (text, true_label) in enumerate(TEST_DATA):
        base_name   = LABEL_NAMES[base_preds_b[i]]
        tuned_name  = LABEL_NAMES[tuned_preds_b[i]]
        true_name   = LABEL_NAMES[true_label]
        display_text = (text[:COL_W - 3] + "...") if len(text) > COL_W else text
        print(f"  {display_text:<{COL_W}} {base_name:>10} {tuned_name:>10} {true_name:>10}")

    print("  " + "-" * (COL_W + 42))
    print()

    # ---- B6. Print metrics -----------------------------------

    print(f"  BEFORE fine-tuning : {base_acc_b:.0f}% accuracy,  "
          f"macro-F1 = {base_f1_b:.2f}")
    print(f"  AFTER  fine-tuning : {tuned_acc_b:.0f}% accuracy, "
          f"macro-F1 = {tuned_f1_b:.2f}")
    print(f"  Improvement        : +{tuned_acc_b - base_acc_b:.0f}pp accuracy, "
          f"+{tuned_f1_b - base_f1_b:.2f} F1")
    print()

    # Per-class F1 scores
    print("  Per-class F1 scores:")
    print(f"  {'Class':<12} {'Base F1':>10} {'Tuned F1':>10}")
    print("  " + "-" * 34)
    for cls in range(NUM_CLASSES):
        print(f"  {LABEL_NAMES[cls]:<12} {base_f1s_b[cls]:>10.2f} "
              f"{tuned_f1s_b[cls]:>10.2f}")
    print()

    # ---- B7. Print confusion matrices as ASCII tables --------

    def print_confusion_matrix(cm, title):
        """
        Print a confusion matrix as a plain ASCII table.
        Rows = actual class, Columns = predicted class.
        Diagonal values are correct predictions.
        """
        print(f"  {title}")
        header = f"  {'':>12}"                    # Left padding
        for name in LABEL_NAMES:
            header += f"  {name[:7]:>9}"           # Column header (truncated)
        print(header)
        print("  " + "-" * (12 + 11 * NUM_CLASSES))

        for row_idx, row_name in enumerate(LABEL_NAMES):
            row_str = f"  {row_name:>12}"          # Row label (actual class)
            for col_idx in range(NUM_CLASSES):
                val = cm[row_idx][col_idx]         # Cell value
                marker = "*" if row_idx == col_idx else " "  # * = diagonal = correct
                row_str += f"  {val:>8}{marker}"
            print(row_str)

        print()
        print("  (* = correct prediction / diagonal)")
        print()

    print_confusion_matrix(base_cm_b,  "Confusion Matrix -- Base Model (before fine-tuning):")
    print_confusion_matrix(tuned_cm_b, "Confusion Matrix -- Fine-Tuned Model (after fine-tuning):")

    print("  Part B complete.")

except ImportError:
    # PyTorch not installed -- skip Part B gracefully
    print("  PyTorch not installed. Skipping Part B.")
    print("  Install with: pip install torch")

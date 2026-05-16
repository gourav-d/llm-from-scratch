"""
Example 6: Email Spam Detector — Clean Separated Version
=========================================================

PURPOSE OF THIS FILE:
  Show HOW TO STRUCTURE a real ML project properly.
  Same spam detector as Example 5, but with clean separation.

PROBLEM WITH EXAMPLE 5:
  - 100 lines just to set up data
  - Training logic mixed with data logic
  - Hard to reuse any single part

SOLUTION (this file):
  Each function does ONE job only.
  Read it top-to-bottom like a story.

  create_excel_data()   <-- makes the data file (runs once)
  load_data()           <-- reads Excel, returns X and y
  normalize()           <-- scales features to 0 mean, 1 std
  SpamDetector          <-- the PyTorch model (just the architecture)
  train_model()         <-- runs training loop
  evaluate_model()      <-- checks accuracy on test data
  predict_emails()      <-- classifies new emails
  main()                <-- calls everything in order

C# ANALOGY:
  This is like separating a C# project into:
    DataRepository.cs   -> load_data
    Preprocessing.cs    -> normalize
    Model.cs            -> SpamDetector
    TrainingService.cs  -> train_model
    EvaluationService.cs-> evaluate_model
    PredictionService.cs-> predict_emails
    Program.cs          -> main

DATA FILE:
  data/spam_emails.xlsx   <- created automatically on first run
  Columns: exclamations | has_free | num_links | num_caps | has_winner | label

Run: python example_06_spam_clean.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd          # reads/writes Excel files (pip install pandas openpyxl)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (change these to experiment)
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE   = "data/spam_emails.xlsx"   # where data lives
NUM_EMAILS  = 500                        # total rows in dataset
TRAIN_RATIO = 0.8                        # 80% train, 20% test
EPOCHS      = 150                        # training iterations
LEARN_RATE  = 0.01                       # how fast weights update
THRESHOLD   = 0.5                        # probability > 0.5 = SPAM

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: DATA CREATION
# Creates spam_emails.xlsx on first run. Skips if file already exists.
# ─────────────────────────────────────────────────────────────────────────────

def create_excel_data(filepath, num_emails=NUM_EMAILS):
    """
    Generate fake email feature data and save to Excel.

    ONLY runs if the file doesn't already exist.
    Think of this as your "seed data" script in a real project.

    Excel columns:
        exclamations : number of "!" in subject line  (spam = many)
        has_free     : 1 if subject has word "FREE"   (spam = usually yes)
        num_links    : number of URLs in body         (spam = many)
        num_caps     : number of ALL-CAPS words       (spam = many)
        has_winner   : 1 if body has "winner/prize"  (spam = usually yes)
        label        : 0 = normal email, 1 = spam
    """
    if os.path.exists(filepath):
        print(f"[Data] Found existing file: {filepath}")
        return   # already exists, don't overwrite

    print(f"[Data] Creating {filepath} ...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    np.random.seed(42)
    half = num_emails // 2   # half spam, half normal

    # --- Spam rows: high exclamations, FREE word, many links, caps, winner word ---
    spam = pd.DataFrame({
        "exclamations" : np.random.randint(3, 10, half),
        "has_free"     : np.random.choice([0, 1], half, p=[0.10, 0.90]),
        "num_links"    : np.random.randint(5, 20, half),
        "num_caps"     : np.random.randint(5, 15, half),
        "has_winner"   : np.random.choice([0, 1], half, p=[0.10, 0.90]),
        "label"        : 1,   # 1 = spam
    })

    # --- Normal rows: few exclamations, no FREE, few links, few caps ---
    normal = pd.DataFrame({
        "exclamations" : np.random.randint(0, 2, half),
        "has_free"     : np.random.choice([0, 1], half, p=[0.95, 0.05]),
        "num_links"    : np.random.randint(0, 3, half),
        "num_caps"     : np.random.randint(0, 3, half),
        "has_winner"   : np.random.choice([0, 1], half, p=[0.97, 0.03]),
        "label"        : 0,   # 0 = normal
    })

    # Combine, shuffle, save
    df = pd.concat([spam, normal], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows
    df.to_excel(filepath, index=False)   # saves to Excel (index=False = no row numbers)

    print(f"[Data] Saved {len(df)} rows → {filepath}")
    print(f"[Data] Spam: {df['label'].sum()}, Normal: {(df['label'] == 0).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: DATA LOADING
# Reads Excel → returns NumPy arrays X (features) and y (labels)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath):
    """
    Read Excel file, split into features (X) and labels (y).

    Returns:
        X : numpy array, shape (N, 5) — 5 feature columns
        y : numpy array, shape (N,)   — 0 or 1 labels
    """
    print(f"\n[Load] Reading {filepath} ...")

    df = pd.read_excel(filepath)   # reads entire Excel sheet into a DataFrame

    # Feature columns (everything except 'label')
    feature_columns = ["exclamations", "has_free", "num_links", "num_caps", "has_winner"]
    X = df[feature_columns].values.astype(float)   # shape: (500, 5)
    y = df["label"].values.astype(float)            # shape: (500,)

    print(f"[Load] Loaded {len(df)} emails | Features: {X.shape[1]} | Spam: {int(y.sum())}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: PREPROCESSING
# Splits into train/test, normalizes features to zero mean
# ─────────────────────────────────────────────────────────────────────────────

def split_and_normalize(X, y, train_ratio=TRAIN_RATIO):
    """
    Split into train/test sets and normalize feature values.

    WHY normalize: features have different scales (links: 0-20, has_free: 0-1).
    Normalization puts them on equal footing so no feature dominates.

    Returns:
        X_train, X_test       : normalized numpy arrays
        y_train, y_test       : label arrays
        feature_mean, feature_std : needed to normalize NEW emails later
    """
    split = int(train_ratio * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Compute mean and std FROM TRAINING DATA ONLY
    # (never use test data stats — that would be cheating)
    feature_mean = X_train.mean(axis=0)   # shape: (5,)
    feature_std  = X_train.std(axis=0)    # shape: (5,)

    X_train = (X_train - feature_mean) / feature_std
    X_test  = (X_test  - feature_mean) / feature_std   # use SAME stats

    print(f"\n[Preprocess] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_mean, feature_std


def to_tensors(X_train, X_test, y_train, y_test):
    """
    Convert numpy arrays to PyTorch tensors.

    y needs .unsqueeze(1) to add a column dimension:
      (400,) → (400, 1)   required by BCEWithLogitsLoss
    """
    return (
        torch.FloatTensor(X_train),
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_train).unsqueeze(1),   # (400,) → (400, 1)
        torch.FloatTensor(y_test).unsqueeze(1),    # (100,) → (100, 1)
    )


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: MODEL DEFINITION
# Just the architecture. No training, no data, no logic here.
# ─────────────────────────────────────────────────────────────────────────────

class SpamDetector(nn.Module):
    """
    Binary classifier: 5 inputs → 1 output (spam probability).

    Architecture:
        Input  (5)  → Hidden (16) → Hidden (8) → Output (1)
                        ReLU          ReLU         (no activation)

    Output is RAW SCORE (not probability).
    Apply torch.sigmoid() to get probability (0.0 to 1.0).
    BCEWithLogitsLoss handles sigmoid internally during training.
    """

    def __init__(self):
        super(SpamDetector, self).__init__()
        self.fc1 = nn.Linear(5, 16)    # 5 features  → 16 hidden
        self.fc2 = nn.Linear(16, 8)    # 16 hidden   → 8 hidden
        self.fc3 = nn.Linear(8, 1)     # 8 hidden    → 1 output score
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))   # (batch, 5)  → (batch, 16)
        x = self.relu(self.fc2(x))   # (batch, 16) → (batch, 8)
        x = self.fc3(x)              # (batch, 8)  → (batch, 1)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# PART 5: TRAINING
# One function, one job: run the training loop and return loss history
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, X_train_t, y_train_t, epochs=EPOCHS, lr=LEARN_RATE):
    """
    Train model on training data.

    Training loop (runs `epochs` times):
      1. Forward pass: model makes predictions
      2. Compute loss: how wrong are the predictions?
      3. Backward pass: compute gradients (how to fix each weight)
      4. Optimizer step: apply fixes

    Returns:
        loss_history : list of loss per epoch (for plotting)
    """
    print(f"\n[Train] Starting training: {epochs} epochs, lr={lr}")

    loss_fn   = nn.BCEWithLogitsLoss()              # loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    model.train()   # switch model to training mode
    for epoch in range(epochs):

        predictions = model(X_train_t)              # forward pass
        loss        = loss_fn(predictions, y_train_t)  # compute loss

        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # compute new gradients
        optimizer.step()        # update weights

        loss_history.append(loss.item())

        if (epoch + 1) % 30 == 0:
            probs   = torch.sigmoid(predictions)
            preds   = (probs > THRESHOLD).float()
            correct = (preds == y_train_t).sum().item()
            acc     = 100 * correct / len(y_train_t)
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {acc:.1f}%")

    print("[Train] Done.")
    return loss_history


# ─────────────────────────────────────────────────────────────────────────────
# PART 6: EVALUATION
# One function, one job: measure accuracy on test data
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test_t, y_test_t):
    """
    Measure model performance on test data (data model never saw during training).

    Reports:
        Overall accuracy
        Spam accuracy   (how many real spam emails caught)
        Normal accuracy (how many normal emails correctly left alone)

    Returns:
        metrics dict
    """
    print("\n[Evaluate] Running on test set ...")

    model.eval()                   # switch to evaluation mode (disables dropout etc.)
    with torch.no_grad():          # skip gradient tracking (saves memory)
        raw_scores  = model(X_test_t)
        probs       = torch.sigmoid(raw_scores)
        predictions = (probs > THRESHOLD).float()   # 0.0 or 1.0

    correct  = (predictions == y_test_t).sum().item()
    accuracy = 100 * correct / len(y_test_t)

    spam_mask   = (y_test_t == 1)
    normal_mask = (y_test_t == 0)
    spam_acc    = 100 * (predictions[spam_mask] == y_test_t[spam_mask]).sum().item() / spam_mask.sum().item()
    normal_acc  = 100 * (predictions[normal_mask] == y_test_t[normal_mask]).sum().item() / normal_mask.sum().item()

    print(f"  Overall Accuracy: {accuracy:.1f}%")
    print(f"  Spam caught:      {spam_acc:.1f}%")
    print(f"  Normal kept safe: {normal_acc:.1f}%")

    return {"overall": accuracy, "spam": spam_acc, "normal": normal_acc}


# ─────────────────────────────────────────────────────────────────────────────
# PART 7: PREDICTION
# One function, one job: classify brand-new emails
# ─────────────────────────────────────────────────────────────────────────────

def predict_emails(model, emails_raw, feature_mean, feature_std):
    """
    Classify new emails that the model has never seen.

    Args:
        emails_raw   : numpy array shape (N, 5) — raw feature values (NOT normalized)
        feature_mean : mean from training data (from split_and_normalize)
        feature_std  : std  from training data (from split_and_normalize)

    IMPORTANT: must apply the SAME normalization used during training.
               Different scaling = wrong predictions.

    Returns:
        List of (probability, decision) tuples
    """
    emails_norm = (emails_raw - feature_mean) / feature_std   # normalize with training stats
    emails_t    = torch.FloatTensor(emails_norm)

    model.eval()
    with torch.no_grad():
        raw_scores = model(emails_t)
        probs      = torch.sigmoid(raw_scores).squeeze(1)     # shape: (N,)

    results = []
    for p in probs:
        prob     = p.item()
        decision = "SPAM" if prob > THRESHOLD else "NOT SPAM"
        results.append((prob, decision))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  —  orchestrates all parts in order
# This is the ONLY place that calls other functions.
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Email Spam Detector — Clean Separated Version")
    print("=" * 55)

    # ── Step 1: Make sure data file exists ──────────────────
    create_excel_data(DATA_FILE)

    # ── Step 2: Load data from Excel ────────────────────────
    X, y = load_data(DATA_FILE)

    # ── Step 3: Split and normalize ─────────────────────────
    X_train, X_test, y_train, y_test, feat_mean, feat_std = split_and_normalize(X, y)

    # ── Step 4: Convert to tensors ──────────────────────────
    X_train_t, X_test_t, y_train_t, y_test_t = to_tensors(X_train, X_test, y_train, y_test)

    # ── Step 5: Build model ─────────────────────────────────
    model = SpamDetector()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] SpamDetector | Parameters: {total_params}")
    print(f"        Architecture: 5 → 16 → 8 → 1")

    # ── Step 6: Train ───────────────────────────────────────
    loss_history = train_model(model, X_train_t, y_train_t)

    # ── Step 7: Evaluate ────────────────────────────────────
    metrics = evaluate_model(model, X_test_t, y_test_t)

    # ── Step 8: Predict new emails ──────────────────────────
    print("\n[Predict] New emails:")

    # Format: [exclamations, has_free, num_links, num_caps, has_winner]
    new_emails = np.array([
        [8, 1, 15, 12, 1],   # very spammy: "FREE!!! YOU WON 12 PRIZES!!!"
        [0, 0,  1,  0, 0],   # normal: "Meeting at 3pm tomorrow"
        [3, 1,  4,  3, 0],   # borderline: some spam signals
    ])
    labels = ["Obvious Spam  ", "Normal Email  ", "Borderline    "]

    results = predict_emails(model, new_emails, feat_mean, feat_std)

    print(f"\n  {'Email':<18} {'Probability':>12} {'Decision':>12}")
    print("  " + "-" * 46)
    for label, (prob, decision) in zip(labels, results):
        bar = "#" * int(prob * 20)
        print(f"  {label}  {prob:>10.1%}   {decision:<10}  [{bar:<20}]")

    # ── Done ────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"Final Test Accuracy: {metrics['overall']:.1f}%")
    print("=" * 55)
    print("""
Code structure recap:
  create_excel_data()   makes data/spam_emails.xlsx
  load_data()           reads Excel → X, y arrays
  split_and_normalize() train/test split + scaling
  to_tensors()          numpy → PyTorch tensors
  SpamDetector          model architecture only
  train_model()         training loop only
  evaluate_model()      accuracy metrics only
  predict_emails()      classify new emails
  main()                calls all of the above
""")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — only runs main() when YOU run this file directly.
# If another file imports this, main() does NOT run automatically.
# C# analogy: static void Main(string[] args) — same idea.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

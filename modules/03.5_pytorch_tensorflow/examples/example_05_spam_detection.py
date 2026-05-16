"""
Example 5: Email Spam Detection with PyTorch
=============================================

WHAT WE ARE BUILDING:
A model that reads email features and answers: SPAM or NOT SPAM?

This is BINARY CLASSIFICATION — only 2 possible answers (0 or 1).

REAL WORLD:
Gmail, Outlook — all use models like this (but trained on millions of emails).

EMAIL FEATURES WE WILL USE:
  - num_exclamations : how many "!!!" in subject line
  - has_free_word    : does subject contain "FREE"? (0 or 1)
  - num_links        : how many links in email body
  - num_capitals     : how many ALL-CAPS words
  - has_winner_word  : does it say "winner/prize/won"? (0 or 1)

LABEL:
  0 = Not Spam (normal email)
  1 = Spam

HOW THIS DIFFERS FROM EXAMPLE 3 (Iris - 3 classes):
  ─────────────────────────────────────────────────
  Example 3 (multi-class)     This example (binary)
  ─────────────────────────   ─────────────────────
  Output neurons: 3           Output neurons: 1
  Loss: CrossEntropyLoss      Loss: BCEWithLogitsLoss
  Activation: none            Activation: Sigmoid
  Predict: argmax             Predict: > 0.5 = spam

Run: python example_05_spam_detection.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=" * 60)
print("Email Spam Detector")
print("=" * 60)

# ============================================================================
# STEP 1: Create Fake Email Data
# ============================================================================
print("\nStep 1: Create Email Dataset")
print("-" * 40)

np.random.seed(42)
torch.manual_seed(42)

NUM_EMAILS = 500   # 500 fake emails

# --- Generate SPAM emails (250 emails) ---
# Spam emails: many exclamations, "FREE" word, many links, many caps, "winner" word
num_spam = 250
spam_features = np.column_stack([
    np.random.randint(3, 10, num_spam),     # exclamations: 3–9 (lots!)
    np.random.choice([0, 1], num_spam, p=[0.1, 0.9]),  # 90% have "FREE"
    np.random.randint(5, 20, num_spam),     # links: 5–19 (lots!)
    np.random.randint(5, 15, num_spam),     # capitals: 5–14 (lots!)
    np.random.choice([0, 1], num_spam, p=[0.1, 0.9]),  # 90% have "winner"
]).astype(float)
spam_labels = np.ones(num_spam)             # label = 1 for all spam

# --- Generate NORMAL emails (250 emails) ---
# Normal emails: few exclamations, no "FREE", few links, few caps, no "winner"
num_normal = 250
normal_features = np.column_stack([
    np.random.randint(0, 2, num_normal),    # exclamations: 0–1 (few)
    np.random.choice([0, 1], num_normal, p=[0.95, 0.05]),  # 5% have "FREE"
    np.random.randint(0, 3, num_normal),    # links: 0–2 (few)
    np.random.randint(0, 3, num_normal),    # capitals: 0–2 (few)
    np.random.choice([0, 1], num_normal, p=[0.97, 0.03]),  # 3% have "winner"
]).astype(float)
normal_labels = np.zeros(num_normal)        # label = 0 for all normal

# --- Combine spam + normal into one dataset ---
# np.vstack stacks rows: (250,5) + (250,5) = (500,5)
X = np.vstack([spam_features, normal_features])
y = np.concatenate([spam_labels, normal_labels])

# Shuffle so spam and normal are mixed (not all spam first)
# C# analogy: like calling .OrderBy(x => Guid.NewGuid()) on a list
shuffle_idx = np.random.permutation(NUM_EMAILS)
X = X[shuffle_idx]
y = y[shuffle_idx]

print(f"Total emails: {NUM_EMAILS}")
print(f"Spam emails:  {int(y.sum())}")
print(f"Normal emails:{int((y==0).sum())}")
print(f"Features per email: {X.shape[1]}")
print(f"Feature names: exclamations, has_FREE, num_links, num_caps, has_winner")

# --- Train / Test split (80/20) ---
split = int(0.8 * NUM_EMAILS)   # 400 train, 100 test
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- Normalize features ---
X_mean  = X_train.mean(axis=0)
X_std   = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std

# --- Convert to PyTorch tensors ---
X_train_t = torch.FloatTensor(X_train)           # shape: (400, 5)
X_test_t  = torch.FloatTensor(X_test)            # shape: (100, 5)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)   # shape: (400, 1)
y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)    # shape: (100, 1)
# unsqueeze(1) adds a column dimension: (400,) → (400, 1)
# WHY: BCEWithLogitsLoss needs matching shapes between prediction and label

print("Data ready!")

# ============================================================================
# STEP 2: Build the Spam Detector Network
# ============================================================================
print("\nStep 2: Build Network")
print("-" * 40)

class SpamDetector(nn.Module):
    """
    Architecture:
        Input  (5)  --> 5 email features
           |
        Hidden (16) --> learns "many exclamations + FREE = probably spam"
           |
        Hidden (8)  --> refines the pattern
           |
        Output (1)  --> single score (negative = not spam, positive = spam)

    OUTPUT EXPLAINED:
    Raw output is an unbounded number (e.g. -2.5 or +3.1)
    We apply Sigmoid to convert it to a probability (0.0 to 1.0)
    Sigmoid(-2.5) = 0.08  → 8% chance of spam   → NOT SPAM
    Sigmoid(+3.1) = 0.96  → 96% chance of spam  → SPAM

    Sigmoid curve:
        1.0 |          ──────────
            |        /
        0.5 |───────/─────────── ← threshold (we pick 0.5)
            |      /
        0.0 |─────
               negative  positive (raw output)
    """

    def __init__(self):
        super(SpamDetector, self).__init__()

        self.fc1 = nn.Linear(5, 16)    # 5 inputs  → 16 hidden neurons
        self.fc2 = nn.Linear(16, 8)    # 16 hidden → 8 hidden neurons
        self.fc3 = nn.Linear(8, 1)     # 8 hidden  → 1 output (spam score)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))   # (batch, 5) → (batch, 16)
        x = self.relu(self.fc2(x))   # (batch, 16) → (batch, 8)
        x = self.fc3(x)              # (batch, 8)  → (batch, 1)

        # DO NOT apply sigmoid here!
        # BCEWithLogitsLoss does sigmoid internally (more numerically stable)
        # We apply sigmoid manually only during prediction (Step 6)
        return x

model = SpamDetector()
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# STEP 3: Loss Function + Optimizer
# ============================================================================
print("\nStep 3: Setup Training")
print("-" * 40)

# BCEWithLogitsLoss = Binary Cross Entropy + Sigmoid combined
# "Binary" because we only have 2 classes (0 or 1)
# "Logits" means it accepts raw output (before sigmoid) — more stable
# C# analogy: it's like -( y*log(p) + (1-y)*log(1-p) ) averaged over batch
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Loss: BCEWithLogitsLoss (binary cross entropy)")
print("Optimizer: Adam (lr=0.01)")

# ============================================================================
# STEP 4: Training Loop
# ============================================================================
print("\nStep 4: Training")
print("-" * 40)

EPOCHS = 150

for epoch in range(EPOCHS):
    model.train()

    # Forward pass: get raw scores (not probabilities yet)
    output = model(X_train_t)              # shape: (400, 1)

    # Loss: compares raw scores to labels (0 or 1)
    loss = criterion(output, y_train_t)    # single number

    # Backward + update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        # Calculate training accuracy
        with torch.no_grad():
            # Apply sigmoid to convert raw scores → probabilities
            probs = torch.sigmoid(output)          # values between 0 and 1

            # Apply threshold: probability > 0.5 → predict spam (1)
            preds = (probs > 0.5).float()          # 0.0 or 1.0

            correct  = (preds == y_train_t).sum().item()
            accuracy = 100 * correct / len(y_train_t)

        print(f"Epoch {epoch+1:3d}/150 | Loss: {loss.item():.4f} | Train Accuracy: {accuracy:.1f}%")

# ============================================================================
# STEP 5: Evaluate on Test Set
# ============================================================================
print("\nStep 5: Evaluation")
print("-" * 40)

model.eval()
with torch.no_grad():
    test_output = model(X_test_t)                  # raw scores, shape: (100, 1)
    test_probs  = torch.sigmoid(test_output)        # probabilities 0-1
    test_preds  = (test_probs > 0.5).float()        # binary: 0 or 1

    correct  = (test_preds == y_test_t).sum().item()
    accuracy = 100 * correct / len(y_test_t)

    # Separate accuracy for spam vs normal
    spam_mask   = (y_test_t == 1)   # boolean mask for spam rows
    normal_mask = (y_test_t == 0)   # boolean mask for normal rows

    spam_correct   = (test_preds[spam_mask]   == y_test_t[spam_mask]).sum().item()
    normal_correct = (test_preds[normal_mask] == y_test_t[normal_mask]).sum().item()

    spam_acc   = 100 * spam_correct   / spam_mask.sum().item()
    normal_acc = 100 * normal_correct / normal_mask.sum().item()

print(f"Overall Accuracy: {correct}/{len(y_test_t)} = {accuracy:.1f}%")
print(f"Spam Accuracy:    {spam_acc:.1f}%  (caught real spam)")
print(f"Normal Accuracy:  {normal_acc:.1f}%  (didn't block normal emails)")

# Show first 10 test predictions
print("\nSample Predictions (first 10 test emails):")
print(f"{'Probability':>12} {'Predicted':>12} {'Actual':>10} {'Correct?':>10}")
print("-" * 50)
for i in range(10):
    prob      = test_probs[i].item()
    pred_lbl  = "SPAM" if test_preds[i].item() == 1 else "not spam"
    actual_lbl= "SPAM" if y_test_t[i].item()   == 1 else "not spam"
    correct_str = "OK" if test_preds[i] == y_test_t[i] else "WRONG"
    print(f"{prob:>11.2%}   {pred_lbl:>10}   {actual_lbl:>8}   {correct_str:>8}")

# ============================================================================
# STEP 6: Classify a NEW Email (real-world usage)
# ============================================================================
print("\nStep 6: Classify New Emails")
print("-" * 40)

# Format: [exclamations, has_FREE, num_links, num_caps, has_winner]
new_emails = np.array([
    [8, 1, 12, 10, 1],    # looks very spammy: "FREE!!! YOU WON!!! Click links!!!"
    [0, 0,  1,  0, 0],    # normal: "Meeting tomorrow at 3pm"
    [3, 1,  4,  3, 0],    # borderline: some spam signals
])
new_email_labels = ["Very Spammy", "Normal Email", "Borderline"]

# Apply SAME normalization
new_emails_norm = (new_emails - X_mean) / X_std
new_emails_t    = torch.FloatTensor(new_emails_norm)

model.eval()
with torch.no_grad():
    scores = model(new_emails_t)           # raw scores
    probs  = torch.sigmoid(scores)         # probabilities

print(f"{'Email':<16} {'Spam Prob':>10} {'Decision':>12}")
print("-" * 42)
for i, label in enumerate(new_email_labels):
    prob     = probs[i].item()
    decision = "SPAM" if prob > 0.5 else "NOT SPAM"
    bar      = "#" * int(prob * 20)        # simple visual bar
    print(f"{label:<16} {prob:>9.1%}   {decision:<10}  [{bar:<20}]")

# ============================================================================
# Summary: 3 Classification Types Side by Side
# ============================================================================
print("\n" + "=" * 60)
print("ALL 3 CLASSIFICATION TYPES — SUMMARY")
print("=" * 60)
print("""
              BINARY          MULTI-CLASS       REGRESSION
              ──────────────  ────────────────  ──────────────
Example       Spam or not     Cat/Dog/Bird      Predict price
Output units  1               N (one per class) 1
Last act.     None*           None              None
Loss func     BCEWithLogits   CrossEntropyLoss  MSELoss
Predict       sigmoid > 0.5   argmax            raw value
Metric        accuracy %      accuracy %        RMSE / MAE

  * sigmoid applied INSIDE BCEWithLogitsLoss during training,
    and MANUALLY during prediction with torch.sigmoid()

Decision boundary (binary):
  Raw score → sigmoid → probability → threshold → label
  e.g.:  2.3 → 0.91 → 91% spam → SPAM
  e.g.: -1.7 → 0.15 → 15% spam → NOT SPAM
""")
print("Done! You now know all 3 model types in PyTorch.")
print("=" * 60)

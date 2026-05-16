"""
Example 3: Simple Classification with PyTorch
==============================================

WHAT WE ARE BUILDING:
A model that looks at a flower and says: "This is Setosa / Versicolor / Virginica"
(3 categories = multi-class classification)

DATASET: Iris flowers (famous ML dataset)
- 4 features per flower: sepal length, sepal width, petal length, petal width
- 3 classes: 0=Setosa, 1=Versicolor, 2=Virginica
- 150 total samples

C# ANALOGY:
Think of the model as a trained if/else decision tree,
but instead of hard rules, it learns probabilities from data.

Run: python example_03_classification.py
"""

import torch                            # PyTorch core (like using System in C#)
import torch.nn as nn                   # Neural network building blocks
import torch.optim as optim             # Optimizers (SGD, Adam, etc.)
from sklearn.datasets import load_iris  # Famous flower dataset (free, built-in)
from sklearn.model_selection import train_test_split  # Split data into train/test
from sklearn.preprocessing import StandardScaler      # Normalize input features
import numpy as np                      # Numerical arrays (like double[] in C#)

print("=" * 60)
print("Simple Classification: Iris Flowers")
print("=" * 60)

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================
print("\nStep 1: Loading Data")
print("-" * 40)

# load_iris() returns features (X) and labels (y)
# X = 150 rows x 4 columns (4 measurements per flower)
# y = 150 labels (0, 1, or 2 for each flower type)
iris = load_iris()
X = iris.data    # shape: (150, 4) — 150 flowers, 4 features each
y = iris.target  # shape: (150,)   — 150 labels

print(f"Total samples: {len(X)}")
print(f"Features per sample: {X.shape[1]}")
print(f"Classes: {iris.target_names}")  # ['setosa', 'versicolor', 'virginica']

# Split into training (80%) and testing (20%)
# C# analogy: like splitting a list with LINQ Take/Skip
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # fixed seed so results are reproducible
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Normalize features: subtract mean, divide by std deviation
# WHY: All 4 features are on different scales (cm values differ a lot)
#      Normalizing makes training faster and more stable
# C# analogy: like normalizing percentages to 0-1 range
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # learn mean/std from train, then apply
X_test  = scaler.transform(X_test)       # apply SAME scaling to test (don't relearn)

# Convert numpy arrays to PyTorch tensors
# C# analogy: like converting double[] to a special GPU-compatible array
X_train_tensor = torch.FloatTensor(X_train)   # float32 tensor, shape (120, 4)
X_test_tensor  = torch.FloatTensor(X_test)    # float32 tensor, shape (30, 4)
y_train_tensor = torch.LongTensor(y_train)    # int64 tensor (needed for CrossEntropy)
y_test_tensor  = torch.LongTensor(y_test)

print("Data converted to PyTorch tensors")

# ============================================================================
# STEP 2: Build the Neural Network
# ============================================================================
print("\nStep 2: Building Network")
print("-" * 40)

class IrisClassifier(nn.Module):
    """
    Network architecture:
        Input  (4)  --> 4 flower measurements
           |
        Hidden (16) --> learns patterns like "long petals = virginica"
           |
        Hidden (8)  --> refines those patterns
           |
        Output (3)  --> score for each class [setosa, versicolor, virginica]

    The class with highest score = prediction
    """

    def __init__(self):
        # Always call parent init first — sets up internal PyTorch machinery
        super(IrisClassifier, self).__init__()

        # fc = "fully connected" layer
        # Linear(in_features, out_features) = weight matrix + bias
        self.fc1 = nn.Linear(4, 16)    # 4 inputs  → 16 hidden neurons
        self.fc2 = nn.Linear(16, 8)    # 16 hidden → 8 hidden neurons
        self.fc3 = nn.Linear(8, 3)     # 8 hidden  → 3 outputs (one per class)

        # ReLU activation: makes negative outputs = 0, keeps positives
        # WHY: Without activations, stacking linear layers = still just 1 linear layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, 4)  e.g. (32, 4) for 32 flowers at once

        x = self.relu(self.fc1(x))  # (batch, 4)  → (batch, 16), apply ReLU
        x = self.relu(self.fc2(x))  # (batch, 16) → (batch, 8),  apply ReLU
        x = self.fc3(x)             # (batch, 8)  → (batch, 3),  NO activation here

        # WHY no activation on last layer?
        # CrossEntropyLoss expects raw scores (logits), applies softmax internally
        # Softmax converts [2.1, 0.3, -1.4] → probabilities [0.73, 0.18, 0.09]
        return x

# Create model instance
model = IrisClassifier()
print(model)

# Count total learnable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")
# fc1: 4*16 + 16 = 80
# fc2: 16*8 + 8  = 136
# fc3: 8*3  + 3  = 27
# Total: 80 + 136 + 27 = 243

# ============================================================================
# STEP 3: Loss Function and Optimizer
# ============================================================================
print("\nStep 3: Setup Training")
print("-" * 40)

# CrossEntropyLoss = best loss for multi-class classification
# It measures: how wrong is my prediction vs actual label?
# Lower loss = better predictions
criterion = nn.CrossEntropyLoss()

# Adam optimizer adjusts weights to minimize loss
# lr = learning rate: how big a step to take each update
# Too high lr = overshoot, too low = slow learning
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam (lr=0.01)")

# ============================================================================
# STEP 4: Training Loop
# ============================================================================
print("\nStep 4: Training")
print("-" * 40)

EPOCHS = 100  # how many times to go through all training data

for epoch in range(EPOCHS):

    # --- FORWARD PASS: run data through model, get predictions ---
    model.train()                      # tells model: "we are training" (enables dropout etc.)
    output = model(X_train_tensor)     # shape: (120, 3) — 3 scores per flower

    # --- COMPUTE LOSS: how wrong are predictions? ---
    loss = criterion(output, y_train_tensor)

    # --- BACKWARD PASS: calculate gradients (how to adjust each weight) ---
    optimizer.zero_grad()   # clear gradients from previous step (MUST do this!)
    loss.backward()         # calculate new gradients via chain rule

    # --- UPDATE WEIGHTS: move weights in direction that reduces loss ---
    optimizer.step()

    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        # Calculate training accuracy
        with torch.no_grad():          # no_grad = skip gradient tracking (saves memory)
            predictions = output.argmax(dim=1)  # pick class with highest score
            correct = (predictions == y_train_tensor).sum().item()
            accuracy = 100 * correct / len(y_train_tensor)
        print(f"Epoch {epoch+1:3d}/100 | Loss: {loss.item():.4f} | Train Accuracy: {accuracy:.1f}%")

# ============================================================================
# STEP 5: Evaluate on Test Data
# ============================================================================
print("\nStep 5: Evaluation on Test Data")
print("-" * 40)

model.eval()                  # tells model: "we are testing" (disables dropout)
with torch.no_grad():         # don't calculate gradients during testing
    test_output = model(X_test_tensor)         # shape: (30, 3)
    predictions = test_output.argmax(dim=1)    # index of highest score = predicted class

    correct = (predictions == y_test_tensor).sum().item()
    accuracy = 100 * correct / len(y_test_tensor)

print(f"Test Accuracy: {correct}/{len(y_test_tensor)} = {accuracy:.1f}%")

# Show individual predictions
print("\nSample Predictions (first 10 test flowers):")
print(f"{'Predicted':<12} {'Actual':<12} {'Correct?'}")
print("-" * 35)
for i in range(10):
    pred_name   = iris.target_names[predictions[i].item()]
    actual_name = iris.target_names[y_test_tensor[i].item()]
    correct_str = "OK" if predictions[i] == y_test_tensor[i] else "WRONG"
    print(f"{pred_name:<12} {actual_name:<12} {correct_str}")

# ============================================================================
# STEP 6: Predict a NEW Flower (real-world usage)
# ============================================================================
print("\nStep 6: Predict a New Flower")
print("-" * 40)

# Fake new flower measurements [sepal_len, sepal_wid, petal_len, petal_wid]
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])   # looks like setosa

# Must apply SAME scaling used during training!
new_flower_scaled = scaler.transform(new_flower)
new_flower_tensor = torch.FloatTensor(new_flower_scaled)

model.eval()
with torch.no_grad():
    scores = model(new_flower_tensor)          # raw scores: e.g. [3.2, -1.1, -2.5]
    probs = torch.softmax(scores, dim=1)       # probabilities: e.g. [0.95, 0.04, 0.01]
    predicted_class = scores.argmax(dim=1)     # index of highest probability

print(f"Input: sepal=5.1x3.5, petal=1.4x0.2")
print(f"Scores (raw):      {scores[0].numpy().round(2)}")
print(f"Probabilities:     {probs[0].numpy().round(2)}")
print(f"Predicted class:   {iris.target_names[predicted_class.item()]}")
print(f"(Correct: setosa)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("KEY CONCEPTS COVERED")
print("=" * 60)
print("""
Classification steps:
  1. Load data         -> features (X) and labels (y)
  2. Normalize         -> StandardScaler (mean=0, std=1)
  3. Build network     -> nn.Module with Linear + ReLU layers
  4. Train loop        -> forward, loss, backward, optimizer.step()
  5. Evaluate          -> argmax gives predicted class
  6. New predictions   -> apply same scaler, then model(input)

Loss function used: CrossEntropyLoss (multi-class classification)
Output activation:  None on last layer (CrossEntropy handles softmax)
""")
print("Next: See example_04_regression.py for predicting numbers!")
print("=" * 60)

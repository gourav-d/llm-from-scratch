"""
Example 4: Simple Regression with PyTorch
==========================================

WHAT WE ARE BUILDING:
A model that predicts a CONTINUOUS NUMBER (not a category).

Example: Predict house price given size (sq ft) and number of rooms.
Output = $312,000  (a number, not a category)

CLASSIFICATION vs REGRESSION:
  Classification = "which bucket?" (cat/dog, spam/not-spam, digit 0-9)
  Regression     = "what number?"  (price, temperature, weight)

DIFFERENCES in code:
  Classification               Regression
  ────────────────────         ──────────────────────
  Loss: CrossEntropyLoss       Loss: MSELoss (mean squared error)
  Output: raw scores           Output: single number
  Evaluate: accuracy %         Evaluate: how far off (error in $)

Run: python example_04_regression.py
"""

import torch                           # PyTorch core
import torch.nn as nn                  # Neural network layers
import torch.optim as optim            # Optimizers
import numpy as np                     # Array math
import matplotlib.pyplot as plt        # Plotting (like charts in Excel)

print("=" * 60)
print("Simple Regression: Predict House Prices")
print("=" * 60)

# ============================================================================
# STEP 1: Create Synthetic Data (fake house data)
# ============================================================================
print("\nStep 1: Generate Data")
print("-" * 40)

# Fix random seed so results are same every run
# C# analogy: like new Random(42) — same sequence each time
np.random.seed(42)
torch.manual_seed(42)

NUM_SAMPLES = 300  # 300 fake houses

# Feature 1: house size in hundreds of sq ft (e.g. 15 = 1500 sq ft)
# np.random.uniform(low, high, count) = random floats between low and high
size_sqft = np.random.uniform(5, 50, NUM_SAMPLES)   # 500 to 5000 sq ft

# Feature 2: number of bedrooms (1 to 6)
num_bedrooms = np.random.randint(1, 7, NUM_SAMPLES).astype(float)

# Feature 3: house age in years (0 = new, 30 = old)
house_age = np.random.uniform(0, 30, NUM_SAMPLES)

# Target: price in $10,000 units (e.g. 30 = $300,000)
# Formula: price depends on size, rooms, and age, plus random noise
price = (
    3.0 * size_sqft           # bigger house → higher price
    + 5.0 * num_bedrooms      # more rooms → higher price
    - 1.5 * house_age         # older house → lower price
    + 10                      # base price
    + np.random.normal(0, 3, NUM_SAMPLES)  # random noise (real world isn't perfect)
)

# Stack features into matrix: shape (300, 3)
# np.column_stack = like building a DataTable column by column in C#
X = np.column_stack([size_sqft, num_bedrooms, house_age])
y = price  # shape: (300,)

print(f"Samples created: {NUM_SAMPLES}")
print(f"Feature matrix shape: {X.shape}")  # (300, 3)
print(f"Price range: ${y.min()*10:.0f}k to ${y.max()*10:.0f}k")

# Split: 80% train, 20% test (manual split without sklearn)
split = int(0.8 * NUM_SAMPLES)   # 240 for train
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Normalize features (same reason as classification example)
# Manually compute mean and std from TRAINING data only
X_mean = X_train.mean(axis=0)    # mean of each column: shape (3,)
X_std  = X_train.std(axis=0)     # std of each column:  shape (3,)
X_train_norm = (X_train - X_mean) / X_std  # normalize train
X_test_norm  = (X_test  - X_mean) / X_std  # normalize test with SAME stats

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train_norm)   # shape: (240, 3)
X_test_t  = torch.FloatTensor(X_test_norm)    # shape: (60, 3)

# y must be shape (N, 1) for regression (not just (N,))
# .reshape(-1, 1): make it a column vector
# C# analogy: double[] → double[,] with 1 column
y_train_t = torch.FloatTensor(y_train.reshape(-1, 1))  # shape: (240, 1)
y_test_t  = torch.FloatTensor(y_test.reshape(-1, 1))   # shape: (60, 1)

print("Data normalized and converted to tensors")

# ============================================================================
# STEP 2: Build the Regression Network
# ============================================================================
print("\nStep 2: Building Network")
print("-" * 40)

class HousePricePredictor(nn.Module):
    """
    Network architecture:
        Input  (3)  --> size, bedrooms, age
           |
        Hidden (32) --> learns feature interactions
           |
        Hidden (16) --> refines them
           |
        Output (1)  --> single price prediction

    KEY DIFFERENCE FROM CLASSIFICATION:
    - Output is 1 neuron (not N classes)
    - No softmax on output
    - Loss = MSE (mean squared error) not CrossEntropy
    """

    def __init__(self):
        super(HousePricePredictor, self).__init__()

        self.fc1 = nn.Linear(3, 32)    # 3 inputs → 32 hidden
        self.fc2 = nn.Linear(32, 16)   # 32 → 16 hidden
        self.fc3 = nn.Linear(16, 1)    # 16 → 1 output (the price)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))   # (batch, 3)  → (batch, 32)
        x = self.relu(self.fc2(x))   # (batch, 32) → (batch, 16)
        x = self.fc3(x)              # (batch, 16) → (batch, 1)
        # No activation here! Price can be any positive number.
        # ReLU would block negative outputs which can be valid.
        return x

model = HousePricePredictor()
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")
# fc1: 3*32 + 32 = 128
# fc2: 32*16 + 16 = 528
# fc3: 16*1 + 1 = 17
# Total: 128 + 528 + 17 = 673

# ============================================================================
# STEP 3: Loss Function and Optimizer
# ============================================================================
print("\nStep 3: Setup Training")
print("-" * 40)

# MSELoss = Mean Squared Error
# Measures: average of (predicted - actual)^2
# Example: predicted=$300k, actual=$320k → error=(300-320)^2=400
# C# analogy: Math.Pow(predicted - actual, 2) averaged over all samples
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Loss: MSELoss (mean squared error)")
print("Optimizer: Adam (lr=0.01)")

# ============================================================================
# STEP 4: Training Loop
# ============================================================================
print("\nStep 4: Training")
print("-" * 40)

EPOCHS = 200
train_losses = []   # track loss over time for plotting

for epoch in range(EPOCHS):

    model.train()

    # Forward pass: get predictions
    predictions = model(X_train_t)            # shape: (240, 1)

    # Compute MSE loss
    loss = criterion(predictions, y_train_t)  # single number, e.g. 45.3

    # Backward pass + weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save loss for plotting
    train_losses.append(loss.item())

    if (epoch + 1) % 40 == 0:
        # RMSE = sqrt(MSE) = error in SAME units as target (price units)
        rmse = torch.sqrt(loss).item()
        print(f"Epoch {epoch+1:3d}/200 | MSE: {loss.item():.2f} | RMSE: {rmse:.2f} ($10k units)")

# ============================================================================
# STEP 5: Evaluate
# ============================================================================
print("\nStep 5: Evaluation")
print("-" * 40)

model.eval()
with torch.no_grad():
    test_preds = model(X_test_t)           # shape: (60, 1)
    test_mse   = criterion(test_preds, y_test_t)
    test_rmse  = torch.sqrt(test_mse).item()

    # Mean Absolute Error: average |predicted - actual|
    # Easier to interpret than RMSE
    mae = torch.abs(test_preds - y_test_t).mean().item()

print(f"Test MSE:  {test_mse.item():.2f}")
print(f"Test RMSE: {test_rmse:.2f} (in $10k units = ${test_rmse*10:.0f}k average error)")
print(f"Test MAE:  {mae:.2f}  (in $10k units = ${mae*10:.0f}k average error)")

# Compare predicted vs actual for first 10 test houses
print("\nSample Predictions (first 10 test houses):")
print(f"{'Predicted':>12} {'Actual':>12} {'Error':>10}")
print("-" * 38)
for i in range(10):
    pred_price   = test_preds[i].item() * 10   # convert back to $k
    actual_price = y_test_t[i].item() * 10     # convert back to $k
    error        = abs(pred_price - actual_price)
    print(f"${pred_price:>8.0f}k   ${actual_price:>8.0f}k   ${error:>6.0f}k")

# ============================================================================
# STEP 6: Predict a NEW House
# ============================================================================
print("\nStep 6: Predict a New House")
print("-" * 40)

# New house: 2000 sq ft (size=20), 3 bedrooms, 10 years old
new_house = np.array([[20, 3, 10]], dtype=float)

# Apply SAME normalization used during training
new_house_norm = (new_house - X_mean) / X_std
new_house_t    = torch.FloatTensor(new_house_norm)

model.eval()
with torch.no_grad():
    predicted_price = model(new_house_t).item()

print(f"House: 2000 sqft, 3 bedrooms, 10 years old")
print(f"Predicted price: ${predicted_price * 10:.0f}k")

# What should it be (from our formula)?
expected = 3.0 * 20 + 5.0 * 3 - 1.5 * 10 + 10
print(f"Expected price:  ${expected * 10:.0f}k  (from our formula)")

# ============================================================================
# STEP 7: Plot Training Loss (optional, requires matplotlib)
# ============================================================================
print("\nStep 7: Plot Training Loss")
print("-" * 40)

try:
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, color='blue', linewidth=1)
    plt.title("Training Loss Over Time (Regression)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)

    # Add annotation at final loss
    final_loss = train_losses[-1]
    plt.annotate(
        f"Final: {final_loss:.2f}",
        xy=(EPOCHS - 1, final_loss),
        xytext=(EPOCHS - 50, final_loss + 5),
        arrowprops=dict(arrowstyle='->')
    )

    plt.tight_layout()
    plt.savefig("regression_loss.png")   # save to file
    print("Loss plot saved to regression_loss.png")
    plt.show()
except Exception as e:
    print(f"Plot skipped: {e}")

# ============================================================================
# Summary + Comparison Table
# ============================================================================
print("\n" + "=" * 60)
print("CLASSIFICATION vs REGRESSION — COMPARISON")
print("=" * 60)
print(f"""
                      CLASSIFICATION      REGRESSION
                      ─────────────────   ─────────────────
Task                  "Which category?"   "What number?"
Example               Is it cat or dog?   What is the price?
Output neurons        N  (one per class)  1  (single value)
Output activation     None (CE handles)   None (any value OK)
Loss function         CrossEntropyLoss    MSELoss
Metric                Accuracy (%)        RMSE / MAE
Last layer            Linear → argmax     Linear → raw value

In code (only these 3 things change):
  loss = nn.CrossEntropyLoss()    |  loss = nn.MSELoss()
  output shape = (batch, N)       |  output shape = (batch, 1)
  accuracy = argmax matches       |  accuracy = RMSE in units
""")
print("Done! Next: See example_05_cnn.py for image classification!")
print("=" * 60)

"""
Exercise 1: Convert NumPy Perceptron to PyTorch
================================================

Task: Convert the NumPy perceptron from Module 3 to PyTorch

Learning Objectives:
- Practice NumPy to PyTorch conversion
- Understand automatic differentiation
- Compare manual vs automatic gradients

Instructions:
1. Review the NumPy implementation below
2. Complete the PyTorch implementation
3. Train both and compare results
4. Verify they produce similar results

Run: python exercise_01_convert_perceptron.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 60)
print("Exercise 1: Convert Perceptron to PyTorch")
print("=" * 60)

# ============================================================================
# PART 1: NumPy Implementation (Module 3 - Reference)
# ============================================================================
print("\n1. NumPy Perceptron (Reference)")
print("-" * 60)

class NumpyPerceptron:
    """Perceptron in NumPy (Module 3 style)"""

    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0

    def forward(self, x):
        """Forward pass"""
        return np.dot(x, self.weights) + self.bias

    def train_step(self, x, y_true, lr=0.01):
        """One training step with manual gradients"""
        # Forward
        y_pred = self.forward(x)

        # Compute error
        error = y_pred - y_true

        # Manual gradients
        grad_w = x * error
        grad_b = error

        # Update weights
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b

        # Return loss
        return 0.5 * error ** 2

# Create simple dataset
print("Creating dataset...")
X_train = np.array([[0.5, 0.3], [0.2, 0.8], [0.9, 0.1], [0.4, 0.6]])
y_train = np.array([0.8, 1.0, 1.0, 1.0])  # Linear combination

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Train NumPy perceptron
print("\nTraining NumPy perceptron...")
np_perceptron = NumpyPerceptron(input_size=2)

for epoch in range(100):
    total_loss = 0
    for x, y in zip(X_train, y_train):
        loss = np_perceptron.train_step(x, y, lr=0.1)
        total_loss += loss

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}, Loss: {total_loss:.4f}")

print(f"\nFinal weights: {np_perceptron.weights}")
print(f"Final bias: {np_perceptron.bias:.4f}")

# Test
test_x = np.array([0.6, 0.4])
test_pred = np_perceptron.forward(test_x)
print(f"\nTest prediction for {test_x}: {test_pred:.4f}")

# ============================================================================
# PART 2: PyTorch Implementation (YOUR TASK)
# ============================================================================
print("\n\n2. PyTorch Perceptron (YOUR IMPLEMENTATION)")
print("-" * 60)

# TODO: Complete this PyTorch implementation
class PyTorchPerceptron(nn.Module):
    """Perceptron in PyTorch - COMPLETE THIS!"""

    def __init__(self, input_size):
        super().__init__()
        # TODO: Define layers
        # Hint: Use nn.Linear(input_size, 1)
        pass  # Replace with your code

    def forward(self, x):
        # TODO: Define forward pass
        pass  # Replace with your code

# TODO: Convert data to PyTorch tensors
# X_train_torch = torch.tensor(..., dtype=torch.float32)
# y_train_torch = torch.tensor(..., dtype=torch.float32).view(-1, 1)

# TODO: Create model, optimizer, and loss function
# model = PyTorchPerceptron(input_size=2)
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# criterion = nn.MSELoss()

# TODO: Training loop
# for epoch in range(100):
#     optimizer.zero_grad()
#     predictions = model(X_train_torch)
#     loss = criterion(predictions, y_train_torch)
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 20 == 0:
#         print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")

# TODO: Test the PyTorch model
# test_x_torch = torch.tensor([0.6, 0.4], dtype=torch.float32)
# with torch.no_grad():
#     test_pred_torch = model(test_x_torch).item()
# print(f"\nTest prediction for [0.6, 0.4]: {test_pred_torch:.4f}")

print("\n⚠️ TODO: Complete the PyTorch implementation above!")

# ============================================================================
# SOLUTION (Uncomment to see answer)
# ============================================================================

# """
# SOLUTION:

class PyTorchPerceptron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Convert data
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Setup
model = PyTorchPerceptron(input_size=2)
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

# Train
print("\nTraining PyTorch perceptron...")
for epoch in range(100):
    optimizer.zero_grad()
    predictions = model(X_train_torch)
    loss = criterion(predictions, y_train_torch)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")

# Test
test_x_torch = torch.tensor([0.6, 0.4], dtype=torch.float32)
with torch.no_grad():
    test_pred_torch = model(test_x_torch).item()
print(f"\nTest prediction for [0.6, 0.4]: {test_pred_torch:.4f}")

print("\n✅ PyTorch implementation complete!")
# """

# ============================================================================
# Questions to Answer
# ============================================================================
print("\n\n" + "=" * 60)
print("Questions (Answer these after completing the exercise)")
print("=" * 60)
print("""
1. How many lines of code for training?
   - NumPy: ~10 lines (manual gradients)
   - PyTorch: ~6 lines (automatic gradients)

2. What does optimizer.zero_grad() do?
   - Clears old gradients (they accumulate by default)

3. What replaced the manual gradient calculations?
   - loss.backward() - computes all gradients automatically!

4. Are the predictions similar?
   - Yes! Both should give similar results (may vary due to initialization)

5. Which is easier to understand?
   - NumPy: More explicit, see every step
   - PyTorch: More concise, less error-prone

6. Which would you use in production?
   - PyTorch: Automatic gradients, GPU support, less bugs
""")

print("\n✅ Exercise 1 complete!")
print("Next: exercise_02_build_mlp.py - Build a multi-layer network!")
print("=" * 60)

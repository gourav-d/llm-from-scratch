"""
Example 2: MNIST Classifier with PyTorch
=========================================

Complete neural network for digit recognition.

This example demonstrates:
- Building a neural network with nn.Module
- Loading MNIST dataset
- Training loop with batches
- Evaluation and accuracy calculation
- Model saving and loading

Run: python example_02_mnist_pytorch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("=" * 60)
print("MNIST Classifier with PyTorch")
print("=" * 60)

# ============================================================================
# 1. Configuration
# ============================================================================
print("\n1. Configuration")
print("-" * 60)

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")
print(f"Device: {DEVICE}")

# ============================================================================
# 2. Define Model
# ============================================================================
print("\n\n2. Define Model")
print("-" * 60)

class MNISTNet(nn.Module):
    """
    Simple neural network for MNIST classification

    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden 1: 128 neurons + ReLU
    - Hidden 2: 64 neurons + ReLU
    - Output: 10 classes
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Flatten image: (batch, 1, 28, 28) → (batch, 784)
        x = x.view(-1, 28 * 28)

        # Hidden layer 1
        x = F.relu(self.fc1(x))

        # Hidden layer 2
        x = F.relu(self.fc2(x))

        # Output layer (no activation, done in loss function)
        x = self.fc3(x)

        return x

# Create model
model = MNISTNet().to(DEVICE)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ============================================================================
# 3. Load Data
# ============================================================================
print("\n\n3. Load Data")
print("-" * 60)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Download and load training data
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of batches: {len(train_loader)}")

# ============================================================================
# 4. Setup Training
# ============================================================================
print("\n\n4. Setup Training")
print("-" * 60)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer.__class__.__name__}")

# ============================================================================
# 5. Training Function
# ============================================================================
def train(epoch):
    """Train for one epoch"""
    model.train()  # Set to training mode

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(DEVICE), target.to(DEVICE)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Compute loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]  '
                  f'Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

# ============================================================================
# 6. Test Function
# ============================================================================
def test():
    """Evaluate on test set"""
    model.eval()  # Set to evaluation mode

    test_loss = 0
    correct = 0

    with torch.no_grad():  # No gradients needed
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Forward pass
            output = model(data)

            # Compute loss
            test_loss += criterion(output, target).item()

            # Count correct predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy

# ============================================================================
# 7. Train the Model
# ============================================================================
print("\n\n5. Training")
print("-" * 60)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(1, EPOCHS + 1):
    print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

    # Train
    train_loss, train_acc = train(epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Test
    test_loss, test_acc = test()
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

# ============================================================================
# 8. Save Model
# ============================================================================
print("\n\n6. Save Model")
print("-" * 60)

# Save entire model
torch.save(model.state_dict(), 'mnist_pytorch.pth')
print("✓ Model saved to 'mnist_pytorch.pth'")

# Load model (example)
# model = MNISTNet()
# model.load_state_dict(torch.load('mnist_pytorch.pth'))
# model.eval()

# ============================================================================
# 9. Make Predictions
# ============================================================================
print("\n\n7. Make Predictions")
print("-" * 60)

model.eval()
with torch.no_grad():
    # Get one batch from test set
    data, target = next(iter(test_loader))
    data, target = data.to(DEVICE), target.to(DEVICE)

    # Predict
    output = model(data)
    pred = output.argmax(dim=1)

    # Show first 10 predictions
    print("First 10 predictions:")
    for i in range(10):
        print(f"  Predicted: {pred[i].item()}, Actual: {target[i].item()}",
              "✓" if pred[i].item() == target[i].item() else "✗")

# ============================================================================
# Summary
# ============================================================================
print("\n\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"""
Model Architecture:
  - Input: 784 (28×28)
  - Hidden 1: 128 neurons (ReLU)
  - Hidden 2: 64 neurons (ReLU)
  - Output: 10 classes

Training Results:
  - Final Train Loss: {train_losses[-1]:.4f}
  - Final Train Accuracy: {train_accuracies[-1]:.2f}%
  - Final Test Loss: {test_losses[-1]:.4f}
  - Final Test Accuracy: {test_accuracies[-1]:.2f}%

Total Parameters: {total_params:,}
Device: {DEVICE}
""")

print("✅ Complete MNIST classifier implemented!")
print("\nNext: Run example_03_custom_autograd.py for custom operations!")
print("=" * 60)

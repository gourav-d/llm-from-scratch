"""
Example 7: MNIST Handwritten Digit Classifier - COMPLETE PROJECT

This is the CAPSTONE PROJECT for Module 3!

This project brings together EVERYTHING you learned:
- Lesson 1: Perceptrons (neurons)
- Lesson 2: Activation functions (ReLU, Softmax)
- Lesson 3: Multi-layer networks (deep learning)
- Lesson 4: Backpropagation (how it learns)
- Lesson 5: Training loop (batches, epochs, validation)
- Lesson 6: Optimizers (Adam)

Goal: Build a neural network that recognizes handwritten digits with 95%+ accuracy!

For .NET developers: This is like building a complete image classification API
from scratch - no libraries, just NumPy!

WHAT IS MNIST?
- 70,000 images of handwritten digits (0-9)
- Each image: 28x28 pixels (784 pixels total)
- Goal: Classify which digit (0-9) is in the image
- This is the "Hello World" of deep learning!
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import time


print("="*70)
print("MNIST HANDWRITTEN DIGIT CLASSIFIER")
print("Complete Project - Module 3 Capstone")
print("="*70)

# ============================================================================
# PART 1: LOAD MNIST DATA
# ============================================================================

print("\nPART 1: Loading MNIST Dataset")
print("-" * 70)

def load_mnist_simple():
    """
    Load a simplified version of MNIST dataset.

    In a real project, you'd use:
    - from tensorflow.keras.datasets import mnist
    - Or download from http://yann.lecun.com/exdb/mnist/

    For this educational example, we'll create a synthetic dataset
    that behaves like MNIST.
    """
    print("Creating synthetic MNIST-like dataset...")
    print("(In production, you'd download real MNIST data)")

    # Create synthetic data that mimics MNIST structure
    np.random.seed(42)

    # Training set: 1000 samples (small for speed, real MNIST has 60,000)
    n_train = 1000
    X_train = np.random.randn(n_train, 784) * 0.5  # 28x28 = 784 pixels
    y_train = np.random.randint(0, 10, size=n_train)  # Digits 0-9

    # Test set: 200 samples (real MNIST has 10,000)
    n_test = 200
    X_test = np.random.randn(n_test, 784) * 0.5
    y_test = np.random.randint(0, 10, size=n_test)

    # Add some pattern to make learning possible
    # (Real MNIST has actual digit patterns)
    for i in range(n_train):
        digit = y_train[i]
        # Add a weak pattern based on digit value
        X_train[i] += digit * 0.1

    for i in range(n_test):
        digit = y_test[i]
        X_test[i] += digit * 0.1

    # Normalize to [0, 1] range (important for neural networks!)
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

    return X_train, y_train, X_test, y_test


# Load data
X_train, y_train, X_test, y_test = load_mnist_simple()

print(f"\nDataset loaded successfully!")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Input features: {X_train.shape[1]} (28x28 pixels)")
print(f"  Output classes: 10 (digits 0-9)")
print(f"  Data range: [{X_train.min():.2f}, {X_train.max():.2f}]")


# ============================================================================
# PART 2: BUILD NEURAL NETWORK FROM SCRATCH
# ============================================================================

print("\n" + "="*70)
print("PART 2: Building Neural Network Architecture")
print("-" * 70)

class NeuralNetwork:
    """
    Complete Multi-Layer Neural Network for MNIST Classification

    Architecture: 784 → 128 → 64 → 10
    - Input: 784 pixels
    - Hidden layer 1: 128 neurons (ReLU activation)
    - Hidden layer 2: 64 neurons (ReLU activation)
    - Output: 10 neurons (Softmax - probabilities for each digit)

    For .NET devs: Like a class with Forward() and Train() methods
    """

    def __init__(self, input_size=784, hidden1=128, hidden2=64, output_size=10):
        """
        Initialize network with random weights.

        Why random initialization?
        - All zeros = all neurons learn same thing (symmetry problem)
        - Small random values = break symmetry
        - Xavier/He initialization = scale appropriately
        """
        print("\nInitializing neural network...")
        print(f"  Architecture: {input_size} → {hidden1} → {hidden2} → {output_size}")

        # Layer 1: Input → Hidden1
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1))

        # Layer 2: Hidden1 → Hidden2
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2))

        # Layer 3: Hidden2 → Output
        self.W3 = np.random.randn(hidden2, output_size) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, output_size))

        # Calculate total parameters
        total_params = (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size +
            self.W3.size + self.b3.size
        )
        print(f"  Total parameters: {total_params:,}")
        print(f"    Layer 1: {self.W1.size + self.b1.size:,}")
        print(f"    Layer 2: {self.W2.size + self.b2.size:,}")
        print(f"    Layer 3: {self.W3.size + self.b3.size:,}")

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def relu(self, x):
        """
        ReLU activation: max(0, x)
        Used in hidden layers - prevents vanishing gradient
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """
        Softmax activation: converts logits to probabilities
        Used in output layer for classification

        Formula: exp(x) / sum(exp(x))
        Numerical stability: subtract max before exp
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward propagation: compute predictions

        Steps:
        1. Input → Hidden1 (linear + ReLU)
        2. Hidden1 → Hidden2 (linear + ReLU)
        3. Hidden2 → Output (linear + Softmax)

        For .NET devs: Like chaining LINQ operations
        """
        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        # Layer 3 (Output)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = self.softmax(self.z3)

        return self.a3

    def backward(self, X, y, output):
        """
        Backpropagation: compute gradients

        This is the MAGIC that makes neural networks learn!

        For each layer:
        1. Calculate error (how wrong we were)
        2. Calculate gradient (direction to improve)
        3. Use chain rule to propagate error backwards
        """
        m = X.shape[0]  # Batch size

        # Convert labels to one-hot encoding
        # Example: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        y_one_hot = np.zeros((m, 10))
        y_one_hot[np.arange(m), y] = 1

        # Output layer gradient
        dz3 = output - y_one_hot  # Cross-entropy + softmax derivative
        dW3 = (self.a2.T @ dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        # Hidden layer 2 gradient
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer 1 gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2, dW3, db3

    def update_weights(self, dW1, db1, dW2, db2, dW3, db3, learning_rate):
        """
        Update weights using gradients (gradient descent)

        Formula: weight = weight - learning_rate * gradient
        """
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    def compute_loss(self, y_true, y_pred):
        """
        Cross-entropy loss (for classification)

        Formula: -sum(y_true * log(y_pred))

        Lower loss = better predictions
        """
        m = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
        return loss

    def compute_accuracy(self, y_true, y_pred):
        """
        Accuracy: percentage of correct predictions

        Example: 95% accuracy = 95 out of 100 correct
        """
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true) * 100

    def predict(self, X):
        """
        Make predictions on new data

        Returns: predicted class (0-9)
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)


# Create network
model = NeuralNetwork(input_size=784, hidden1=128, hidden2=64, output_size=10)


# ============================================================================
# PART 3: TRAIN THE NETWORK
# ============================================================================

print("\n" + "="*70)
print("PART 3: Training the Neural Network")
print("-" * 70)

def create_batches(X, y, batch_size):
    """Split data into mini-batches"""
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    batches = []
    for i in range(0, len(X), batch_size):
        batch_indices = indices[i:i+batch_size]
        batches.append((X[batch_indices], y[batch_indices]))

    return batches


def train_network(model, X_train, y_train, X_val, y_val,
                  epochs=30, batch_size=32, learning_rate=0.01):
    """
    Complete training loop with all best practices:
    - Mini-batch training
    - Validation monitoring
    - Progress tracking
    - Early stopping
    """
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batches per epoch: {len(X_train) // batch_size}")

    print("\nStarting training...")
    print("-" * 70)

    best_val_acc = 0
    patience = 5
    patience_counter = 0

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Create batches
        batches = create_batches(X_train, y_train, batch_size)

        epoch_loss = 0
        epoch_acc = 0

        # Train on each batch
        for X_batch, y_batch in batches:
            # Forward pass
            output = model.forward(X_batch)

            # Compute loss and accuracy
            y_batch_one_hot = np.zeros((len(y_batch), 10))
            y_batch_one_hot[np.arange(len(y_batch)), y_batch] = 1

            loss = model.compute_loss(y_batch_one_hot, output)
            acc = model.compute_accuracy(y_batch, output)

            epoch_loss += loss
            epoch_acc += acc

            # Backward pass
            dW1, db1, dW2, db2, dW3, db3 = model.backward(X_batch, y_batch, output)

            # Update weights
            model.update_weights(dW1, db1, dW2, db2, dW3, db3, learning_rate)

        # Average metrics
        avg_loss = epoch_loss / len(batches)
        avg_acc = epoch_acc / len(batches)

        # Validation
        val_output = model.forward(X_val)
        y_val_one_hot = np.zeros((len(y_val), 10))
        y_val_one_hot[np.arange(len(y_val)), y_val] = 1

        val_loss = model.compute_loss(y_val_one_hot, val_output)
        val_acc = model.compute_accuracy(y_val, val_output)

        # Store history
        model.history['train_loss'].append(avg_loss)
        model.history['train_acc'].append(avg_acc)
        model.history['val_loss'].append(val_loss)
        model.history['val_acc'].append(val_acc)

        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {avg_acc:5.2f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:5.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered (no improvement for {patience} epochs)")
            break

    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Total training time: {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    return model


# Split data into train and validation
split_idx = int(0.8 * len(X_train))
X_train_split = X_train[:split_idx]
y_train_split = y_train[:split_idx]
X_val = X_train[split_idx:]
y_val = y_train[split_idx:]

print(f"\nData split:")
print(f"  Training: {len(X_train_split)} samples")
print(f"  Validation: {len(X_val)} samples")

# Train the model
model = train_network(
    model, X_train_split, y_train_split, X_val, y_val,
    epochs=30, batch_size=32, learning_rate=0.01
)


# ============================================================================
# PART 4: EVALUATE ON TEST SET
# ============================================================================

print("\n" + "="*70)
print("PART 4: Final Evaluation on Test Set")
print("-" * 70)

# Make predictions on test set
test_output = model.forward(X_test)
test_predictions = model.predict(X_test)

# Calculate metrics
y_test_one_hot = np.zeros((len(y_test), 10))
y_test_one_hot[np.arange(len(y_test)), y_test] = 1

test_loss = model.compute_loss(y_test_one_hot, test_output)
test_acc = model.compute_accuracy(y_test, test_output)

print(f"\nTest Set Performance:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc:.2f}%")

# Per-class accuracy
print(f"\nPer-Digit Accuracy:")
for digit in range(10):
    mask = y_test == digit
    if np.sum(mask) > 0:
        digit_acc = np.mean(test_predictions[mask] == y_test[mask]) * 100
        correct = np.sum(test_predictions[mask] == y_test[mask])
        total = np.sum(mask)
        print(f"  Digit {digit}: {digit_acc:5.2f}% ({correct}/{total} correct)")


# ============================================================================
# PART 5: VISUALIZE RESULTS
# ============================================================================

print("\n" + "="*70)
print("PART 5: Visualizing Training Progress")
print("-" * 70)

# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss curves
ax = axes[0, 0]
ax.plot(model.history['train_loss'], label='Training Loss', linewidth=2)
ax.plot(model.history['val_loss'], label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Accuracy curves
ax = axes[0, 1]
ax.plot(model.history['train_acc'], label='Training Accuracy', linewidth=2)
ax.plot(model.history['val_acc'], label='Validation Accuracy', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Training & Validation Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix (simplified)
ax = axes[1, 0]
confusion = np.zeros((10, 10))
for true_label, pred_label in zip(y_test, test_predictions):
    confusion[true_label, pred_label] += 1

im = ax.imshow(confusion, cmap='Blues')
ax.set_xlabel('Predicted Digit')
ax.set_ylabel('True Digit')
ax.set_title('Confusion Matrix')
plt.colorbar(im, ax=ax)

# Add text annotations
for i in range(10):
    for j in range(10):
        text = ax.text(j, i, int(confusion[i, j]),
                      ha="center", va="center", color="black", fontsize=8)

# Plot 4: Sample predictions
ax = axes[1, 1]
ax.axis('off')
ax.text(0.1, 0.9, 'Model Summary', fontsize=14, fontweight='bold',
        transform=ax.transAxes)

summary_text = f"""
Architecture:
  Input: 784 (28×28 pixels)
  Hidden 1: 128 neurons (ReLU)
  Hidden 2: 64 neurons (ReLU)
  Output: 10 neurons (Softmax)

Total Parameters: {100,000:,}

Training Results:
  Final Train Acc: {model.history['train_acc'][-1]:.2f}%
  Final Val Acc: {model.history['val_acc'][-1]:.2f}%
  Test Accuracy: {test_acc:.2f}%

Components Used:
  ✓ Multi-layer network (Lesson 3)
  ✓ ReLU activation (Lesson 2)
  ✓ Softmax output (Lesson 2)
  ✓ Backpropagation (Lesson 4)
  ✓ Mini-batch training (Lesson 5)
  ✓ Cross-entropy loss (Lesson 4)
"""

ax.text(0.1, 0.75, summary_text, fontsize=10, verticalalignment='top',
        family='monospace', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('mnist_classifier_results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as: mnist_classifier_results.png")
plt.show()


# ============================================================================
# PART 6: SUMMARY AND NEXT STEPS
# ============================================================================

print("\n" + "="*70)
print("PROJECT COMPLETE! 🎉")
print("="*70)

print(f"""
Congratulations! You've built a complete neural network from scratch!

What You Accomplished:
  ✓ Built a 3-layer neural network (784 → 128 → 64 → 10)
  ✓ Implemented forward propagation
  ✓ Implemented backpropagation
  ✓ Trained with mini-batch gradient descent
  ✓ Achieved {test_acc:.2f}% accuracy on test set
  ✓ Used validation set to prevent overfitting
  ✓ Visualized training progress

What You Used from Module 3:
  ✓ Lesson 1: Perceptrons (every neuron in your network)
  ✓ Lesson 2: ReLU + Softmax activations
  ✓ Lesson 3: Multi-layer architecture
  ✓ Lesson 4: Backpropagation algorithm
  ✓ Lesson 5: Training loop with batches
  ✓ Lesson 6: Gradient descent optimizer

This is THE SAME fundamental approach used in:
  - GPT-3 (175 billion parameters)
  - BERT (340 million parameters)
  - ResNet (image classification)
  - All modern neural networks!

The only difference? Scale and architecture!

Next Steps:
  1. Experiment with hyperparameters:
     - Try different learning rates (0.001, 0.01, 0.1)
     - Try different architectures (more/fewer layers)
     - Try different batch sizes (16, 64, 128)

  2. Improve accuracy:
     - Add more hidden layers
     - Use Adam optimizer (Lesson 6)
     - Add dropout for regularization

  3. Move to Module 4:
     - Learn attention mechanism
     - Build transformers
     - Understand GPT architecture

You're now ready for Module 4: Transformers!
""")

print("\n" + "="*70)
print("End of Module 3 - You are now a neural network expert!")
print("="*70)

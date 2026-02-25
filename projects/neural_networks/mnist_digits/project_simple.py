"""
MNIST Handwritten Digit Classifier - Simple Version
===================================================

A 3-layer neural network that recognizes handwritten digits (0-9).

Architecture:
Input (784) → Hidden1 (128, ReLU) → Hidden2 (64, ReLU) → Output (10, Softmax)

Expected accuracy: 95-97%
Training time: ~2-3 minutes on CPU

This project shows:
- How to work with image data (28x28 pixels)
- Multi-class classification (10 classes)
- Deeper networks (3 layers!)
- Softmax activation
- Categorical cross-entropy loss

For a more advanced version, see project_main.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

# Set random seed
np.random.seed(42)

print("=" * 70)
print("MNIST HANDWRITTEN DIGIT CLASSIFIER - Simple Version")
print("=" * 70)
print()

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

def load_mnist():
    """
    Load MNIST dataset using scikit-learn.

    MNIST contains:
    - 70,000 images of handwritten digits (0-9)
    - Each image is 28x28 pixels
    - Grayscale: pixel values 0-255

    Returns:
        X_train: (60000, 784) - Training images (flattened)
        X_test: (10000, 784) - Test images
        y_train: (60000,) - Training labels (0-9)
        y_test: (10000,) - Test labels
    """
    print("📂 Step 1: Loading MNIST dataset...")
    print("   (This may take a minute the first time)")

    # Fetch MNIST from scikit-learn
    # Will download ~18MB on first run
    mnist = fetch_openml('mnist_784', version=1, parser='auto')

    # Get data
    X = mnist.data.astype('float32')  # Images
    y = mnist.target.astype('int64')   # Labels

    # X shape: (70000, 784) - each image is flattened 28x28 = 784 pixels
    # y shape: (70000,) - labels 0-9

    print(f"   ✓ Loaded {len(X)} images")
    print(f"   ✓ Image size: 28x28 = 784 pixels")
    print(f"   ✓ Pixel range: {X.min():.0f} - {X.max():.0f}")

    # Split into train/test (60k train, 10k test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )

    print(f"   ✓ Training set: {len(X_train)} images")
    print(f"   ✓ Test set: {len(X_test)} images")

    # Show label distribution
    train_counts = np.bincount(y_train)
    print(f"   ✓ Label distribution (train): {dict(enumerate(train_counts))}")
    print()

    return X_train, X_test, y_train, y_test


# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Preprocess MNIST data.

    Steps:
    1. Normalize pixel values to [0, 1]
    2. One-hot encode labels

    Args:
        X_train, X_test: Image arrays
        y_train, y_test: Label arrays

    Returns:
        X_train, X_test: Normalized images
        y_train_onehot, y_test_onehot: One-hot encoded labels
    """
    print("🔧 Step 2: Preprocessing data...")

    # 1. Normalize pixels to [0, 1]
    # Original: 0-255 (uint8)
    # After: 0.0-1.0 (float32)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print(f"   ✓ Normalized pixels to [{X_train.min():.1f}, {X_train.max():.1f}]")

    # 2. One-hot encode labels
    # Original: 7 → [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    def to_categorical(y, num_classes=10):
        """Convert integer labels to one-hot vectors."""
        n = len(y)
        one_hot = np.zeros((n, num_classes))
        one_hot[np.arange(n), y] = 1
        return one_hot

    y_train_onehot = to_categorical(y_train)
    y_test_onehot = to_categorical(y_test)

    print(f"   ✓ One-hot encoded labels: {y_train.shape} → {y_train_onehot.shape}")
    print(f"   Example: label={y_train[0]} → {y_train_onehot[0]}")
    print()

    return X_train, X_test, y_train_onehot, y_test_onehot, y_train, y_test


# ============================================================================
# PART 3: NEURAL NETWORK
# ============================================================================

class DigitClassifier:
    """
    3-layer neural network for digit classification.

    Architecture:
        Input (784) → Hidden1 (128, ReLU) → Hidden2 (64, ReLU) → Output (10, Softmax)
    """

    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):
        """
        Initialize 3-layer network.

        Args:
            input_size: 784 (28x28 pixels)
            hidden1_size: First hidden layer neurons
            hidden2_size: Second hidden layer neurons
            output_size: 10 (digits 0-9)
        """
        # Layer 1: Input → Hidden1
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1_size))

        # Layer 2: Hidden1 → Hidden2
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))

        # Layer 3: Hidden2 → Output
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros((1, output_size))

        # Store architecture
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        # Cache for backpropagation
        self.cache = {}

    def relu(self, z):
        """ReLU activation."""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """ReLU derivative."""
        return (z > 0).astype(float)

    def softmax(self, z):
        """
        Softmax activation for multi-class classification.

        Converts logits to probabilities that sum to 1.

        Args:
            z: Logits, shape (batch_size, num_classes)

        Returns:
            probabilities: shape (batch_size, num_classes)
        """
        # Numerical stability: subtract max
        z_stable = z - np.max(z, axis=1, keepdims=True)

        # Exponentiate
        exp_z = np.exp(z_stable)

        # Normalize
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward propagation through 3 layers.

        Args:
            X: Input, shape (batch_size, 784)

        Returns:
            y_pred: Predictions, shape (batch_size, 10)
        """
        # Layer 1: Input → Hidden1
        z1 = X @ self.W1 + self.b1           # (batch, 784) @ (784, 128) = (batch, 128)
        a1 = self.relu(z1)                    # ReLU activation

        # Layer 2: Hidden1 → Hidden2
        z2 = a1 @ self.W2 + self.b2          # (batch, 128) @ (128, 64) = (batch, 64)
        a2 = self.relu(z2)                    # ReLU activation

        # Layer 3: Hidden2 → Output
        z3 = a2 @ self.W3 + self.b3          # (batch, 64) @ (64, 10) = (batch, 10)
        y_pred = self.softmax(z3)             # Softmax activation

        # Cache for backpropagation
        self.cache = {
            'X': X, 'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2, 'z3': z3, 'y_pred': y_pred
        }

        return y_pred

    def backward(self, y_true):
        """
        Backpropagation through 3 layers.

        Args:
            y_true: True labels (one-hot), shape (batch_size, 10)

        Returns:
            gradients: Dictionary with gradients for all parameters
        """
        # Get cached values
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        z2 = self.cache['z2']
        a2 = self.cache['a2']
        y_pred = self.cache['y_pred']

        batch_size = X.shape[0]

        # Layer 3 gradients (output)
        # For categorical cross-entropy + softmax: dL/dz3 = y_pred - y_true
        dz3 = y_pred - y_true

        dW3 = (a2.T @ dz3) / batch_size
        db3 = np.sum(dz3, axis=0, keepdims=True) / batch_size

        # Layer 2 gradients (hidden2)
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_derivative(z2)

        dW2 = (a1.T @ dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        # Layer 1 gradients (hidden1)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(z1)

        dW1 = (X.T @ dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        return {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3
        }

    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.forward(X)

    def predict(self, X):
        """Get class predictions (0-9)."""
        proba = self.forward(X)
        return np.argmax(proba, axis=1)


# ============================================================================
# PART 4: ADAM OPTIMIZER
# ============================================================================

class AdamOptimizer:
    """Adam optimizer for 3-layer network."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, network, gradients):
        """Update all 6 parameters (3 weights + 3 biases)."""
        self.t += 1

        # Initialize moments on first call
        if not self.m:
            for param in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
                self.m[param] = np.zeros_like(getattr(network, param))
                self.v[param] = np.zeros_like(getattr(network, param))

        # Update each parameter
        for param in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            grad = gradients[f'd{param}']

            # Update moments
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)

            # Update parameter
            param_value = getattr(network, param)
            param_value -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# ============================================================================
# PART 5: TRAINING
# ============================================================================

def categorical_cross_entropy(y_true, y_pred):
    """
    Categorical cross-entropy loss for multi-class classification.

    Args:
        y_true: True labels (one-hot), shape (batch_size, 10)
        y_pred: Predicted probabilities, shape (batch_size, 10)

    Returns:
        loss: Average loss over batch
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Compute loss
    # L = -sum(y_true * log(y_pred))
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    return loss


def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true: True class labels (integers 0-9)
        y_pred: Predicted probabilities (batch_size, 10)

    Returns:
        accuracy: Fraction of correct predictions
    """
    # Get predicted class (argmax of probabilities)
    predictions = np.argmax(y_pred, axis=1)

    # Compare with true labels
    accuracy = np.mean(predictions == y_true)

    return accuracy


def train(network, X_train, y_train_onehot, y_train, X_test, y_test_onehot, y_test,
          epochs=20, batch_size=128, learning_rate=0.001):
    """
    Train the neural network.

    Args:
        network: DigitClassifier instance
        X_train, y_train_onehot: Training data (one-hot labels)
        y_train: Training labels (integers, for accuracy)
        X_test, y_test_onehot: Test data
        y_test: Test labels (integers)
        epochs: Number of epochs
        batch_size: Mini-batch size
        learning_rate: Learning rate

    Returns:
        history: Training metrics
    """
    print("🎓 Step 3: Training neural network...")
    print(f"   Architecture: {network.input_size} → {network.hidden1_size} → "
          f"{network.hidden2_size} → {network.output_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")

    # Count parameters
    total_params = (network.W1.size + network.b1.size +
                   network.W2.size + network.b2.size +
                   network.W3.size + network.b3.size)
    print(f"   Total parameters: {total_params:,}")
    print()

    # Initialize optimizer
    optimizer = AdamOptimizer(learning_rate=learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    # Training loop
    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_onehot_shuffled = y_train_onehot[indices]

        # Mini-batch training
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_onehot_shuffled[start:end]

            # Forward pass
            y_pred = network.forward(X_batch)

            # Backward pass
            gradients = network.backward(y_batch)

            # Update weights
            optimizer.update(network, gradients)

        # Compute metrics on full datasets (every epoch)
        train_pred = network.forward(X_train)
        train_loss = categorical_cross_entropy(y_train_onehot, train_pred)
        train_acc = compute_accuracy(y_train, train_pred)

        test_pred = network.forward(X_test)
        test_loss = categorical_cross_entropy(y_test_onehot, test_pred)
        test_acc = compute_accuracy(y_test, test_pred)

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:2d}/{epochs}: "
                  f"Loss={train_loss:.3f}, Acc={train_acc:.1%}, "
                  f"Test Loss={test_loss:.3f}, Test Acc={test_acc:.1%}")

    print()
    print(f"   ✓ Training complete! Final test accuracy: {test_acc:.2%}")
    print()

    return history


# ============================================================================
# PART 6: EVALUATION
# ============================================================================

def evaluate(network, X_test, y_test):
    """
    Evaluate network with detailed metrics.

    Args:
        network: Trained DigitClassifier
        X_test: Test images
        y_test: Test labels (integers)

    Returns:
        metrics: Dictionary with various metrics
    """
    print("📊 Step 4: Evaluating on test set...")

    # Make predictions
    y_pred_proba = network.forward(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Overall accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"   ✓ Test Accuracy: {accuracy:.2%}")

    # Top-3 accuracy (is true label in top 3 predictions?)
    top3_pred = np.argsort(y_pred_proba, axis=1)[:, -3:]  # Top 3 indices
    top3_acc = np.mean([y_test[i] in top3_pred[i] for i in range(len(y_test))])
    print(f"   ✓ Top-3 Accuracy: {top3_acc:.2%}")

    # Confusion matrix (10x10)
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_test, y_pred):
        confusion_matrix[true, pred] += 1

    # Per-class accuracy
    print("   ✓ Per-class accuracy:")
    for digit in range(10):
        class_acc = confusion_matrix[digit, digit] / confusion_matrix[digit].sum()
        print(f"      Digit {digit}: {class_acc:.1%}")

    print()

    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_acc,
        'confusion_matrix': confusion_matrix,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

def plot_training_history(history, save_path='results/training_curve.png'):
    """Plot training history."""
    os.makedirs('results', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['test_acc'], label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved training curves: {save_path}")


def plot_confusion_matrix(cm, save_path='results/confusion_matrix.png'):
    """Plot 10x10 confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(cm, cmap='Blues')

    # Labels
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10), fontsize=11)
    ax.set_yticklabels(range(10), fontsize=11)

    ax.set_xlabel('Predicted Digit', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Digit', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix - MNIST Test Set', fontsize=14, fontweight='bold', pad=15)

    # Add counts to cells
    for i in range(10):
        for j in range(10):
            text_color = "white" if cm[i, j] > cm.max()/2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center",
                   color=text_color, fontsize=10)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved confusion matrix: {save_path}")


def plot_sample_predictions(X_test, y_test, predictions, probabilities, num_samples=20,
                           save_path='results/sample_predictions.png'):
    """Plot sample predictions with images."""
    # Select random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for idx, i in enumerate(indices):
        # Reshape to 28x28
        image = X_test[i].reshape(28, 28)

        # Get prediction
        true_label = y_test[i]
        pred_label = predictions[i]
        confidence = probabilities[i, pred_label]

        # Determine if correct
        is_correct = (true_label == pred_label)
        color = 'green' if is_correct else 'red'

        # Plot
        axes[idx].imshow(image, cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(
            f"True: {true_label}, Pred: {pred_label}\n({confidence:.0%})",
            color=color, fontweight='bold', fontsize=10
        )

    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved sample predictions: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_mnist()

    # Step 2: Preprocess
    X_train, X_test, y_train_onehot, y_test_onehot, y_train_int, y_test_int = \
        preprocess_data(X_train, X_test, y_train, y_test)

    # Step 3: Build network
    network = DigitClassifier(
        input_size=784,
        hidden1_size=128,
        hidden2_size=64,
        output_size=10
    )

    # Step 4: Train
    history = train(
        network, X_train, y_train_onehot, y_train_int,
        X_test, y_test_onehot, y_test_int,
        epochs=20, batch_size=128, learning_rate=0.001
    )

    # Step 5: Evaluate
    metrics = evaluate(network, X_test, y_test_int)

    # Step 6: Visualize
    print("📈 Step 5: Creating visualizations...")
    plot_training_history(history)
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_sample_predictions(X_test, y_test_int, metrics['predictions'],
                           metrics['probabilities'])
    print()

    # Final summary
    print("=" * 70)
    print("✅ PROJECT COMPLETE!")
    print("=" * 70)
    print(f"Final Test Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Final Top-3 Accuracy: {metrics['top3_accuracy']:.2%}")
    print()
    print("Generated files:")
    print("  • results/training_curve.png - Training metrics")
    print("  • results/confusion_matrix.png - 10x10 confusion matrix")
    print("  • results/sample_predictions.png - 20 sample predictions")
    print()
    print("Next steps:")
    print("  1. View the visualizations")
    print("  2. Read EXPLANATION.md for code details")
    print("  3. Try experiments (change network size, learning rate)")
    print("  4. Analyze mistakes (which digits are hardest?)")
    print("  5. Move to Project 3: Sentiment Analysis")
    print("=" * 70)


if __name__ == '__main__':
    main()

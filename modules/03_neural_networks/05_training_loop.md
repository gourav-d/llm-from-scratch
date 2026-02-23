# Lesson 3.5: Training Loop - Production-Level Training (Advanced)

> **Note:** Lessons 1-4 gave you everything needed to understand how ChatGPT/GPT works! This lesson and Lesson 6 are **advanced tutorials** that cover production-level training techniques used in real-world applications.

## 🎯 Why This Is Advanced Content

### What You Already Know (Lessons 1-4)

You've mastered the fundamentals:
- ✅ How neurons compute (forward propagation)
- ✅ How networks learn (backpropagation)
- ✅ Building multi-layer networks
- ✅ **Everything needed to understand GPT's core algorithm!**

**For understanding ChatGPT:** Lessons 1-4 are sufficient! You know how it learns and processes data.

### What This Lesson Adds (Advanced)

This lesson covers **production-level techniques** for training at scale:
- **Batching** - Train on multiple examples simultaneously (efficiency)
- **Epochs** - Multiple passes through data (better learning)
- **Data splits** - Train/validation/test (avoid overfitting)
- **Monitoring** - Track progress and detect issues
- **Early stopping** - Know when to stop training

**Use case:** When you're building real applications, not just understanding concepts.

---

## 📊 The Difference

### Basic Training (Lessons 1-4)
```python
# What you learned - sufficient for understanding!
for i in range(iterations):
    y = network.forward(x)
    gradients = network.backward(y, t)
    network.update_weights(gradients, lr)
```

**Good for:** Understanding how neural networks learn

### Production Training (This Lesson)
```python
# Advanced - what's used in real applications
for epoch in range(num_epochs):
    for batch in training_loader:  # Batching!
        y = network.forward(batch)
        gradients = network.backward(y, batch_labels)
        optimizer.step(gradients)  # Advanced optimizers!

    val_loss = evaluate(validation_set)  # Monitoring!
    if val_loss > best_loss:  # Early stopping!
        stop_training()
```

**Good for:** Building production ML systems

---

## 🎓 Learning Outcomes

After this lesson, you'll understand:

**Core Concepts:**
- ✅ Batching and mini-batch gradient descent
- ✅ Epochs vs. iterations
- ✅ Train/validation/test splits
- ✅ Learning curves and overfitting detection
- ✅ Early stopping strategies

**Advanced Techniques:**
- ✅ Data shuffling and batching
- ✅ Progress monitoring
- ✅ Hyperparameter tuning
- ✅ Debugging training issues

---

## 🔄 The Complete Training Loop

### Overview

```
1. Prepare Data
   └─ Split: Train (80%), Validation (10%), Test (10%)
   └─ Batch: Group examples together
   └─ Shuffle: Randomize order

2. Training Loop (Multiple Epochs)
   For each epoch:
     └─ For each batch:
        ├─ Forward pass
        ├─ Compute loss
        ├─ Backward pass (backpropagation)
        └─ Update weights

     └─ Evaluate on validation set
     └─ Check if should stop early

3. Final Evaluation
   └─ Test on held-out test set
   └─ Report final metrics
```

---

## 📐 Key Concepts Explained

### 1. Batching (Mini-Batch Gradient Descent)

**Problem:** Training on one example at a time is slow and noisy.

**Solution:** Group examples into batches!

```python
# Instead of this (Stochastic Gradient Descent):
for example in dataset:  # One at a time
    loss = train_on_one(example)
    update_weights()

# Do this (Mini-Batch Gradient Descent):
for batch in batches:  # Groups of 32, 64, 128, etc.
    loss = train_on_batch(batch)  # Parallel processing!
    update_weights()
```

**Benefits:**
- **Faster:** GPU can process many examples in parallel
- **More stable:** Gradients are averaged over batch
- **Better hardware utilization:** Modern GPUs are designed for batches

**Typical batch sizes:**
- Small datasets: 32-64
- Medium datasets: 64-128
- Large datasets (GPT): 256-512
- Very large (GPT-3): 3.2 million tokens per batch!

### 2. Epochs vs. Iterations

**Iteration:** One update to weights (one batch processed)

**Epoch:** One complete pass through entire dataset

```
Dataset: 1000 examples
Batch size: 100
Iterations per epoch: 1000/100 = 10
Total iterations (10 epochs): 10 × 10 = 100

Epoch 1: Process all 1000 examples (10 batches)
Epoch 2: Process all 1000 examples again
...
Epoch 10: Process all 1000 examples again
```

**Why multiple epochs?**
- Network needs to see examples multiple times to learn
- Each pass refines the weights further
- Typical: 10-100 epochs for small models, 1-5 for large models (GPT)

### 3. Train/Validation/Test Split

**Why split data?**

```
All Data (100%)
├─ Training Set (80%)     ← Learn from this
├─ Validation Set (10%)   ← Tune hyperparameters, early stopping
└─ Test Set (10%)         ← Final evaluation ONLY
```

**Purpose of each:**

**Training Set:**
- Used for backpropagation
- Network learns from this data
- Loss should decrease over time

**Validation Set:**
- NOT used for training
- Monitor overfitting
- Tune hyperparameters (learning rate, etc.)
- Early stopping decisions

**Test Set:**
- NEVER seen during training
- Final evaluation only
- Report this as your model's performance

**Critical rule:** NEVER train on test set! It's your honest evaluation.

### 4. Overfitting vs. Underfitting

**Underfitting:** Model too simple, can't learn patterns
```
Training loss: High (0.5)
Validation loss: High (0.5)
→ Network needs more capacity (more layers/neurons)
```

**Good fit:** Model learns well, generalizes well
```
Training loss: Low (0.05)
Validation loss: Low (0.06)
→ Perfect! Ship it!
```

**Overfitting:** Model memorizes training data, doesn't generalize
```
Training loss: Very low (0.01)
Validation loss: High (0.3)
→ Network memorized training data, fails on new data
```

**Detecting overfitting:**
- Training loss keeps decreasing
- Validation loss starts increasing
- Gap between train and validation grows

**Solutions:**
- More training data
- Regularization (L2, dropout)
- Early stopping
- Simpler model

### 5. Early Stopping

**Idea:** Stop training when validation loss stops improving

```python
best_val_loss = infinity
patience = 5  # Wait 5 epochs before stopping
epochs_without_improvement = 0

for epoch in range(max_epochs):
    train_one_epoch()
    val_loss = evaluate_validation()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()  # Save best model
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping!")
        break  # Stop training

load_best_model()  # Use best model, not last
```

**Benefits:**
- Prevents overfitting
- Saves computation time
- Automatically finds optimal training duration

---

## 💻 Complete Implementation

### Full Training Loop with All Features

```python
import numpy as np
from typing import Tuple, List, Dict


class DataLoader:
    """
    Handles batching and shuffling of data.

    For .NET devs: Like IEnumerable<Batch> with yield return
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        """
        Args:
            X: Input data, shape (features, num_examples)
            Y: Labels, shape (outputs, num_examples)
            batch_size: Number of examples per batch
            shuffle: Whether to shuffle data each epoch
        """
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_examples = X.shape[1]
        self.num_batches = int(np.ceil(self.num_examples / batch_size))

    def __iter__(self):
        """Iterator protocol - allows for batch in dataloader"""
        # Shuffle indices if requested
        indices = np.arange(self.num_examples)
        if self.shuffle:
            np.random.shuffle(indices)

        # Yield batches
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_examples)
            batch_indices = indices[start_idx:end_idx]

            batch_X = self.X[:, batch_indices]
            batch_Y = self.Y[:, batch_indices]

            yield batch_X, batch_Y

    def __len__(self):
        """Number of batches"""
        return self.num_batches


class TrainingMonitor:
    """
    Monitors training progress and implements early stopping.

    Tracks:
    - Training and validation loss history
    - Best model state
    - Early stopping logic
    """

    def __init__(self, patience: int = 10):
        """
        Args:
            patience: Number of epochs to wait before early stopping
        """
        self.patience = patience
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = np.inf
        self.epochs_without_improvement = 0
        self.best_model_weights = None

    def update(self, train_loss: float, val_loss: float, model_weights: Dict) -> bool:
        """
        Update monitor with new losses.

        Returns:
            should_stop: Whether to stop training (early stopping)
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Check if validation loss improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_weights = model_weights.copy()
            self.epochs_without_improvement = 0
            return False  # Don't stop
        else:
            self.epochs_without_improvement += 1

            # Early stopping check
            if self.epochs_without_improvement >= self.patience:
                return True  # Stop training

            return False  # Continue

    def get_best_weights(self) -> Dict:
        """Return best model weights"""
        return self.best_model_weights


def train_val_test_split(X: np.ndarray, Y: np.ndarray,
                         train_ratio: float = 0.8,
                         val_ratio: float = 0.1) -> Tuple:
    """
    Split data into train, validation, and test sets.

    Args:
        X: Input data, shape (features, num_examples)
        Y: Labels, shape (outputs, num_examples)
        train_ratio: Proportion for training (e.g., 0.8 = 80%)
        val_ratio: Proportion for validation (e.g., 0.1 = 10%)

    Returns:
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    """
    num_examples = X.shape[1]

    # Shuffle indices
    indices = np.arange(num_examples)
    np.random.shuffle(indices)

    # Calculate split points
    train_end = int(num_examples * train_ratio)
    val_end = int(num_examples * (train_ratio + val_ratio))

    # Split
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create splits
    X_train, Y_train = X[:, train_indices], Y[:, train_indices]
    X_val, Y_val = X[:, val_indices], Y[:, val_indices]
    X_test, Y_test = X[:, test_indices], Y[:, test_indices]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def train_network(network,
                 X_train, Y_train,
                 X_val, Y_val,
                 num_epochs: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 0.01,
                 patience: int = 10,
                 verbose: bool = True):
    """
    Complete training loop with all features:
    - Batching
    - Validation monitoring
    - Early stopping
    - Progress tracking

    Args:
        network: Neural network with forward(), backward(), update_weights()
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        num_epochs: Maximum number of epochs
        batch_size: Examples per batch
        learning_rate: Learning rate for weight updates
        patience: Epochs to wait before early stopping
        verbose: Print progress

    Returns:
        monitor: TrainingMonitor with history
    """
    # Create data loader
    train_loader = DataLoader(X_train, Y_train, batch_size=batch_size, shuffle=True)

    # Create monitor
    monitor = TrainingMonitor(patience=patience)

    if verbose:
        print(f"Starting training...")
        print(f"  Training examples: {X_train.shape[1]}")
        print(f"  Validation examples: {X_val.shape[1]}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {len(train_loader)}")
        print(f"  Max epochs: {num_epochs}")
        print(f"  Early stopping patience: {patience}")

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        epoch_losses = []

        for batch_X, batch_Y in train_loader:
            # Forward pass
            predictions = network.forward(batch_X)

            # Compute loss
            batch_loss = network.compute_loss(predictions, batch_Y)
            epoch_losses.append(batch_loss)

            # Backward pass
            gradients = network.backward(batch_Y)

            # Update weights
            network.update_weights(gradients, learning_rate)

        # Average training loss for epoch
        train_loss = np.mean(epoch_losses)

        # Validation phase (no weight updates!)
        val_predictions = network.forward(X_val)
        val_loss = network.compute_loss(val_predictions, Y_val)

        # Get current model weights
        model_weights = {
            'W1': network.W1.copy(),
            'b1': network.b1.copy(),
            'W2': network.W2.copy(),
            'b2': network.b2.copy()
        }

        # Update monitor
        should_stop = monitor.update(train_loss, val_loss, model_weights)

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch:3d}/{num_epochs}: "
                  f"Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}, "
                  f"Best Val = {monitor.best_val_loss:.6f}")

        # Early stopping check
        if should_stop:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation loss: {monitor.best_val_loss:.6f}")
            break

    # Restore best weights
    if monitor.best_model_weights is not None:
        network.W1 = monitor.best_model_weights['W1']
        network.b1 = monitor.best_model_weights['b1']
        network.W2 = monitor.best_model_weights['W2']
        network.b2 = monitor.best_model_weights['b2']
        if verbose:
            print("\nRestored best model weights")

    return monitor
```

---

## 🎨 Visualizing Training Progress

### Learning Curves

```python
import matplotlib.pyplot as plt

def plot_learning_curves(monitor: TrainingMonitor):
    """
    Plot training and validation loss over epochs.

    Shows:
    - How loss decreases over time
    - Whether model is overfitting
    - When early stopping occurred
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(monitor.train_losses) + 1)

    plt.plot(epochs, monitor.train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, monitor.val_losses, 'r-', label='Validation Loss', linewidth=2)

    # Mark best epoch
    best_epoch = np.argmin(monitor.val_losses) + 1
    plt.axvline(x=best_epoch, color='g', linestyle='--',
                label=f'Best Model (Epoch {best_epoch})')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Learning Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Annotate if overfitting
    if len(monitor.val_losses) > 10:
        if monitor.val_losses[-1] > monitor.val_losses[best_epoch - 1] * 1.2:
            plt.text(0.5, 0.95, 'Warning: Possible Overfitting!',
                    transform=plt.gca().transAxes,
                    fontsize=12, color='red',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                    ha='center', va='top')

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 🎯 Complete Example: Training XOR

```python
# Example: Train XOR with complete training loop

# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y_xor = np.array([[0, 1, 1, 0]])

# Create larger synthetic dataset (for demonstration)
# Repeat XOR pattern with noise
num_repeats = 250
X_large = np.tile(X_xor, num_repeats) + np.random.randn(2, 4 * num_repeats) * 0.1
Y_large = np.tile(Y_xor, num_repeats)

print(f"Dataset size: {X_large.shape[1]} examples")

# Split data
(X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = train_val_test_split(
    X_large, Y_large,
    train_ratio=0.8,
    val_ratio=0.1
)

print(f"Train: {X_train.shape[1]}, Val: {X_val.shape[1]}, Test: {X_test.shape[1]}")

# Create network (from previous lessons)
network = TwoLayerNetwork(input_size=2, hidden_size=4, output_size=1)

# Train with complete loop
monitor = train_network(
    network,
    X_train, Y_train,
    X_val, Y_val,
    num_epochs=200,
    batch_size=32,
    learning_rate=1.0,
    patience=20,
    verbose=True
)

# Plot learning curves
plot_learning_curves(monitor)

# Final evaluation on test set
test_predictions = network.forward(X_test)
test_loss = network.compute_loss(test_predictions, Y_test)
test_accuracy = np.mean((test_predictions > 0.5) == Y_test)

print(f"\n=== Final Test Results ===")
print(f"Test Loss: {test_loss:.6f}")
print(f"Test Accuracy: {test_accuracy:.2%}")
```

---

## 📊 Key Takeaways

### What You Learned (Advanced!)

1. **Batching improves efficiency**
   - Process multiple examples in parallel
   - GPU utilization
   - More stable gradients

2. **Multiple epochs refine learning**
   - Network sees data multiple times
   - Each pass improves weights

3. **Data splitting prevents overfitting**
   - Train: Learn patterns
   - Validation: Tune hyperparameters
   - Test: Honest evaluation

4. **Monitoring catches problems**
   - Learning curves show overfitting
   - Early stopping prevents wasted time
   - Save best model, not last

5. **Production training is systematic**
   - Not "run and hope"
   - Monitor, evaluate, adjust
   - Reproducible results

---

## 🔗 Connection to GPT Training

### How GPT-3 Was Trained

**Same concepts, massive scale:**

```python
# GPT-3 training (conceptual)

# Dataset: 570GB of text
# Batch size: 3.2 million tokens
# Epochs: ~1 (too expensive for multiple passes!)
# Validation: Perplexity on held-out data
# Early stopping: Based on validation perplexity
# Total cost: ~$4.6 million in compute

for batch in massive_dataset:  # Batching!
    predictions = gpt3_model(batch)
    loss = cross_entropy(predictions, batch[:, 1:])
    gradients = backprop(loss)  # Same backprop you learned!
    adam_optimizer.step(gradients)  # Advanced optimizer

    if step % 1000 == 0:
        val_loss = evaluate(validation_set)  # Monitoring!
        if val_loss > best:
            break  # Early stopping!
```

**Key differences:**
- Scale: 175B parameters vs. your 20
- Data: 570GB vs. your kilobytes
- Batch size: 3.2M tokens vs. your 32 examples
- Hardware: Thousands of GPUs vs. your CPU

**Same fundamentals:**
- ✅ Batching
- ✅ Forward/backward propagation
- ✅ Monitoring
- ✅ Early stopping based on validation

---

## 🎓 When to Use These Techniques

### Basic Training (Lessons 1-4)
**Use when:**
- Learning concepts
- Prototyping
- Small datasets (<1000 examples)
- Understanding fundamentals

### Advanced Training (This Lesson)
**Use when:**
- Building production systems
- Large datasets (>10,000 examples)
- Training takes hours/days
- Performance matters

---

## ✨ Summary

**You now know:**
- ✅ How to batch data efficiently
- ✅ Train/validation/test splits
- ✅ Early stopping to prevent overfitting
- ✅ Monitoring training progress
- ✅ **Production-level training techniques!**

**This completes the training pipeline knowledge!**

Next: **Lesson 6 - Optimizers** (Adam, Momentum - even more advanced!)

---

**Note:** Remember, Lessons 1-4 are sufficient for understanding GPT! This lesson adds production polish for real applications. 🚀

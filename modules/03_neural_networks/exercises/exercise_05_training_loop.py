"""
Exercise 5: Training Loop - Practice Problems

This exercise helps you master production-level training techniques.

DIFFICULTY LEVELS:
- Exercises 1-3: Beginner (basic training concepts)
- Exercises 4-6: Intermediate (batching, validation)
- Exercises 7-10: Advanced (optimization, debugging)

For .NET developers: Think of training loop like a background worker processing
batches of data with progress reporting!
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


# ============================================================================
# EXERCISE 1: Simple Training Loop (Beginner)
# ============================================================================

def exercise_1_simple_training():
    """
    TASK: Implement a basic training loop

    What is a training loop?
    - Repeatedly:
      1. Make predictions (forward pass)
      2. Calculate error (loss)
      3. Compute gradients (backward pass)
      4. Update weights
      5. Repeat until accurate!

    For .NET devs: Like a while loop that keeps improving the model
    """
    print("\n" + "="*70)
    print("EXERCISE 1: Simple Training Loop")
    print("="*70)

    # Simple dataset: Learn y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])  # Inputs
    y = np.array([[3], [5], [7], [9], [11]])  # Outputs (2x + 1)

    print("\nDataset: Learn the pattern y = 2x + 1")
    print("Input → Expected Output")
    for i in range(len(X)):
        print(f"  {X[i][0]} → {y[i][0]}")

    # Initialize weights randomly
    # We want to learn: y = w * x + b
    w = np.random.randn()  # Random weight
    b = np.random.randn()  # Random bias

    print(f"\nInitial (random) weights:")
    print(f"  w = {w:.3f}")
    print(f"  b = {b:.3f}")
    print(f"  Formula: y = {w:.3f}*x + {b:.3f}")

    # TODO: Implement training loop
    learning_rate = 0.01
    iterations = 100
    loss_history = []

    print("\nTraining...")
    for iteration in range(iterations):
        # TODO: 1. Forward pass - make predictions
        predictions = w * X + b

        # TODO: 2. Calculate loss (Mean Squared Error)
        # MSE = average of (predicted - actual)²
        loss = np.mean((predictions - y) ** 2)
        loss_history.append(loss)

        # TODO: 3. Calculate gradients
        # How much to adjust w and b to reduce loss?
        # Gradient of MSE with respect to w and b

        # For MSE loss = mean((wx + b - y)²)
        # dL/dw = 2 * mean((wx + b - y) * x)
        # dL/db = 2 * mean((wx + b - y))

        error = predictions - y
        dw = 2 * np.mean(error * X)  # Gradient for weight
        db = 2 * np.mean(error)       # Gradient for bias

        # TODO: 4. Update weights (gradient descent)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Print progress every 20 iterations
        if iteration % 20 == 0 or iteration == iterations - 1:
            print(f"Iteration {iteration:3d}: Loss = {loss:.6f}, w = {w:.3f}, b = {b:.3f}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final weights:")
    print(f"  w = {w:.3f} (should be ~2.0)")
    print(f"  b = {b:.3f} (should be ~1.0)")
    print(f"  Learned formula: y = {w:.3f}*x + {b:.3f}")

    # Plot loss curve
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Plot predictions vs actual
    predictions_final = w * X + b
    plt.scatter(X, y, label='Actual', s=100)
    plt.plot(X, predictions_final, 'r-', label='Predicted', linewidth=2)
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.title('Final Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_05_simple_training.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_05_simple_training.png")
    plt.show()


# ============================================================================
# EXERCISE 2: Mini-Batch Gradient Descent (Intermediate)
# ============================================================================

def exercise_2_mini_batch():
    """
    TASK: Implement mini-batch training

    What is mini-batch?
    - Instead of training on ALL data at once (slow)
    - Or one example at a time (noisy)
    - Use small batches (e.g., 32 examples)

    For .NET devs: Like processing data in chunks rather than one-by-one
    or all-at-once

    Benefits:
    - Faster than processing one by one
    - More stable than full dataset
    - Enables parallel processing on GPU
    """
    print("\n" + "="*70)
    print("EXERCISE 2: Mini-Batch Gradient Descent")
    print("="*70)

    # Create larger dataset
    np.random.seed(42)
    n_samples = 100

    X = np.random.randn(n_samples, 1)
    y = 3 * X + 2 + np.random.randn(n_samples, 1) * 0.5  # y = 3x + 2 + noise

    print(f"\nDataset: {n_samples} samples")
    print("Pattern: y = 3x + 2 (with noise)")

    # TODO: Implement mini-batch data loader
    def create_batches(X, y, batch_size):
        """
        Split data into mini-batches.

        Args:
            X: Input data (n_samples, features)
            y: Labels (n_samples, 1)
            batch_size: Size of each batch

        Returns:
            List of (X_batch, y_batch) tuples

        Hint:
        1. Calculate number of batches: n_batches = len(X) // batch_size
        2. For each batch i:
           - Get indices: start = i * batch_size, end = start + batch_size
           - Extract X[start:end], y[start:end]
        """
        # YOUR CODE HERE
        n_batches = len(X) // batch_size
        batches = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            batches.append((X_batch, y_batch))

        return batches

    # Initialize weights
    w = np.random.randn()
    b = np.random.randn()

    # Training parameters
    batch_size = 16  # Process 16 samples at a time
    learning_rate = 0.01
    epochs = 20  # One epoch = one pass through entire dataset

    print(f"\nTraining parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches per epoch: {len(X) // batch_size}")
    print(f"  Epochs: {epochs}")

    loss_history = []

    print("\nTraining...")
    for epoch in range(epochs):
        # TODO: Create batches for this epoch
        batches = create_batches(X, y, batch_size)

        epoch_loss = 0

        # TODO: Train on each batch
        for batch_idx, (X_batch, y_batch) in enumerate(batches):
            # Forward pass
            predictions = w * X_batch + b

            # Calculate loss for this batch
            loss = np.mean((predictions - y_batch) ** 2)
            epoch_loss += loss

            # Backward pass (gradients)
            error = predictions - y_batch
            dw = 2 * np.mean(error * X_batch)
            db = 2 * np.mean(error)

            # Update weights
            w = w - learning_rate * dw
            b = b - learning_rate * db

        # Average loss for this epoch
        avg_loss = epoch_loss / len(batches)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1:2d}/{epochs}: Loss = {avg_loss:.6f}, w = {w:.3f}, b = {b:.3f}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final weights:")
    print(f"  w = {w:.3f} (should be ~3.0)")
    print(f"  b = {b:.3f} (should be ~2.0)")

    # Plot results
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, 'o-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss (Mini-Batch)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    predictions_final = w * X + b
    plt.scatter(X, y, alpha=0.5, label='Actual')
    plt.plot(X, predictions_final, 'r', label='Predicted', linewidth=2)
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.title('Final Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_05_mini_batch.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_05_mini_batch.png")
    plt.show()


# ============================================================================
# EXERCISE 3: Train/Validation Split (Intermediate)
# ============================================================================

def exercise_3_train_val_split():
    """
    TASK: Split data into training and validation sets

    Why split data?
    - Training set: Used to update weights
    - Validation set: Check if model generalizes (not overfitting)
    - Test set: Final evaluation (not used during training)

    For .NET devs: Like unit tests (training) vs integration tests (validation)

    Common split:
    - 80% training
    - 10% validation
    - 10% test
    """
    print("\n" + "="*70)
    print("EXERCISE 3: Train/Validation Split")
    print("="*70)

    # Generate dataset
    np.random.seed(42)
    n_samples = 200

    X = np.random.randn(n_samples, 1)
    y = 2.5 * X + 1.5 + np.random.randn(n_samples, 1) * 0.3

    print(f"\nDataset: {n_samples} total samples")

    # TODO: Implement train/validation split
    def train_val_split(X, y, val_ratio=0.2):
        """
        Split data into training and validation sets.

        Args:
            X: Input data
            y: Labels
            val_ratio: Fraction for validation (0.2 = 20%)

        Returns:
            X_train, X_val, y_train, y_val

        Hint:
        1. Calculate split point: split_idx = int(len(X) * (1 - val_ratio))
        2. Training: X[:split_idx], y[:split_idx]
        3. Validation: X[split_idx:], y[split_idx:]
        """
        # YOUR CODE HERE
        split_idx = int(len(X) * (1 - val_ratio))

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]

        return X_train, X_val, y_train, y_val

    # Split data
    X_train, X_val, y_train, y_val = train_val_split(X, y, val_ratio=0.2)

    print(f"Split: {len(X_train)} training, {len(X_val)} validation")
    print(f"  Training: {len(X_train)/len(X):.0%}")
    print(f"  Validation: {len(X_val)/len(X):.0%}")

    # Initialize model
    w = np.random.randn()
    b = np.random.randn()

    # Training
    learning_rate = 0.01
    epochs = 50

    train_loss_history = []
    val_loss_history = []

    print("\nTraining...")
    for epoch in range(epochs):
        # TODO: Train on training set
        predictions_train = w * X_train + b
        train_loss = np.mean((predictions_train - y_train) ** 2)

        error = predictions_train - y_train
        dw = 2 * np.mean(error * X_train)
        db = 2 * np.mean(error)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # TODO: Evaluate on validation set (NO weight updates!)
        predictions_val = w * X_val + b
        val_loss = np.mean((predictions_val - y_val) ** 2)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:2d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final validation loss: {val_loss:.6f}")
    print(f"Weights: w = {w:.3f}, b = {b:.3f}")

    # Plot training vs validation loss
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss', linewidth=2)
    plt.plot(val_loss_history, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(X_train, y_train, alpha=0.5, label='Training', s=30)
    plt.scatter(X_val, y_val, alpha=0.5, label='Validation', s=30)
    predictions_all = w * X + b
    plt.plot(X, predictions_all, 'r-', label='Fitted Line', linewidth=2)
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.title('Data Split and Final Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_05_train_val_split.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_05_train_val_split.png")
    plt.show()

    print("\nKey insight:")
    print("- If val_loss << train_loss: Model is underfitting")
    print("- If val_loss >> train_loss: Model is overfitting!")
    print("- If val_loss ≈ train_loss: Good generalization")


# ============================================================================
# EXERCISE 4: Early Stopping (Advanced)
# ============================================================================

def exercise_4_early_stopping():
    """
    TASK: Implement early stopping to prevent overfitting

    What is early stopping?
    - Stop training when validation loss stops improving
    - Prevents overfitting (memorizing training data)
    - Saves time (don't train unnecessarily)

    For .NET devs: Like breaking out of a loop when condition is met
    """
    print("\n" + "="*70)
    print("EXERCISE 4: Early Stopping")
    print("="*70)

    # Generate data with potential for overfitting
    np.random.seed(42)
    n_samples = 100

    X = np.random.randn(n_samples, 1)
    # Simple pattern but will try to overfit
    y = 2 * X + 1 + np.random.randn(n_samples, 1) * 1.0  # More noise

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Dataset: {len(X_train)} training, {len(X_val)} validation")

    # Model
    w = np.random.randn()
    b = np.random.randn()

    # TODO: Implement early stopping
    learning_rate = 0.05  # Higher learning rate to see overfitting
    max_epochs = 200
    patience = 10  # Stop if no improvement for 10 epochs

    best_val_loss = float('inf')
    patience_counter = 0
    best_w, best_b = w, b

    train_losses = []
    val_losses = []

    print(f"\nEarly stopping parameters:")
    print(f"  Patience: {patience} epochs")
    print(f"  Max epochs: {max_epochs}")

    print("\nTraining...")
    for epoch in range(max_epochs):
        # Training
        pred_train = w * X_train + b
        train_loss = np.mean((pred_train - y_train) ** 2)

        error = pred_train - y_train
        dw = 2 * np.mean(error * X_train)
        db = 2 * np.mean(error)

        w -= learning_rate * dw
        b -= learning_rate * db

        # Validation
        pred_val = w * X_val + b
        val_loss = np.mean((pred_val - y_val) ** 2)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # TODO: Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_w, best_b = w, b
            patience_counter = 0  # Reset patience
            status = "✓ Improved"
        else:
            patience_counter += 1
            status = f"  No improve ({patience_counter}/{patience})"

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Train={train_loss:.4f}, Val={val_loss:.4f} {status}")

        # TODO: Stop if patience exceeded
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}!")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

    # Restore best weights
    w, b = best_w, best_b

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Stopped at epoch: {epoch}")
    print(f"Best weights: w = {w:.3f}, b = {b:.3f}")

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.axvline(x=epoch-patience, color='r', linestyle='--', label='Early Stop')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Early Stopping Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Show overfitting region
    epochs_range = range(len(train_losses))
    plt.plot(epochs_range, np.array(val_losses) - np.array(train_losses),
             linewidth=2, color='red')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=epoch-patience, color='r', linestyle='--', label='Early Stop')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss - Train Loss')
    plt.title('Overfitting Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_05_early_stopping.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_05_early_stopping.png")
    plt.show()


# ============================================================================
# EXERCISE 5: Learning Rate Scheduling (Advanced)
# ============================================================================

def exercise_5_lr_scheduling():
    """
    TASK: Implement learning rate decay

    What is learning rate scheduling?
    - Start with large learning rate (fast progress)
    - Gradually decrease it (fine-tuning)
    - Helps find better solutions!

    For .NET devs: Like starting with big steps, then smaller steps
    as you get closer to the target
    """
    print("\n" + "="*70)
    print("EXERCISE 5: Learning Rate Scheduling")
    print("="*70)

    # Dataset
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 1)
    y = 3 * X - 1 + np.random.randn(n_samples, 1) * 0.5

    # TODO: Implement different learning rate schedules
    def step_decay(initial_lr, epoch, decay_rate=0.5, decay_steps=50):
        """
        Reduce learning rate every decay_steps epochs.

        Example: lr = 0.1 → 0.05 → 0.025 → ...
        """
        return initial_lr * (decay_rate ** (epoch // decay_steps))

    def exponential_decay(initial_lr, epoch, decay_rate=0.95):
        """
        Exponentially decay learning rate.

        Example: lr = 0.1 * 0.95^epoch
        """
        return initial_lr * (decay_rate ** epoch)

    def cosine_annealing(initial_lr, epoch, total_epochs):
        """
        Cosine decay (used in modern training).

        Smooth decrease following cosine curve.
        """
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

    # Train with different schedulers
    schedulers = {
        'Constant': lambda lr, e, t: lr,
        'Step Decay': lambda lr, e, t: step_decay(lr, e),
        'Exponential': lambda lr, e, t: exponential_decay(lr, e),
        'Cosine': lambda lr, e, t: cosine_annealing(lr, e, t)
    }

    initial_lr = 0.1
    epochs = 100
    results = {}

    for name, scheduler in schedulers.items():
        print(f"\nTraining with {name} schedule...")

        w = np.random.randn()
        b = np.random.randn()

        loss_history = []
        lr_history = []

        for epoch in range(epochs):
            # Get current learning rate
            lr = scheduler(initial_lr, epoch, epochs)
            lr_history.append(lr)

            # Train
            pred = w * X + b
            loss = np.mean((pred - y) ** 2)
            loss_history.append(loss)

            error = pred - y
            dw = 2 * np.mean(error * X)
            db = 2 * np.mean(error)

            w -= lr * dw
            b -= lr * db

        results[name] = {
            'loss': loss_history,
            'lr': lr_history,
            'final_loss': loss_history[-1]
        }

        print(f"  Final loss: {loss_history[-1]:.6f}")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Learning rates
    ax = axes[0, 0]
    for name, data in results.items():
        ax.plot(data['lr'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Losses
    ax = axes[0, 1]
    for name, data in results.items():
        ax.plot(data['loss'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Final losses bar chart
    ax = axes[1, 0]
    names = list(results.keys())
    final_losses = [results[n]['final_loss'] for n in names]
    ax.bar(names, final_losses)
    ax.set_ylabel('Final Loss')
    ax.set_title('Final Loss Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Loss in log scale
    ax = axes[1, 1]
    for name, data in results.items():
        ax.semilogy(data['loss'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exercise_05_lr_scheduling.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_05_lr_scheduling.png")
    plt.show()

    print("\n" + "="*60)
    print("Key insights:")
    print("="*60)
    print("- Constant LR: Simple but may overshoot")
    print("- Step Decay: Sudden improvements at decay points")
    print("- Exponential: Smooth gradual improvement")
    print("- Cosine: Modern choice (used in GPT training!)")


# ============================================================================
# EXERCISE 6: Complete Training Pipeline (Advanced)
# ============================================================================

def exercise_6_complete_pipeline():
    """
    TASK: Build a complete production-ready training pipeline

    This combines everything:
    - Mini-batch training
    - Train/val split
    - Early stopping
    - Learning rate scheduling
    - Progress monitoring
    """
    print("\n" + "="*70)
    print("EXERCISE 6: Complete Training Pipeline")
    print("="*70)

    # Generate dataset
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 1)
    y = 2 * X + 1 + np.random.randn(n_samples, 1) * 0.3

    # Split data: 70% train, 15% val, 15% test
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} samples ({len(X_train)/n_samples:.0%})")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/n_samples:.0%})")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/n_samples:.0%})")

    # TODO: Implement complete training pipeline
    class Trainer:
        def __init__(self, batch_size=32, initial_lr=0.1, patience=20):
            self.batch_size = batch_size
            self.initial_lr = initial_lr
            self.patience = patience

            # Model parameters
            self.w = np.random.randn()
            self.b = np.random.randn()

            # Training state
            self.best_val_loss = float('inf')
            self.best_w = self.w
            self.best_b = self.b
            self.patience_counter = 0

            # History
            self.train_losses = []
            self.val_losses = []
            self.learning_rates = []

        def create_batches(self, X, y):
            """Create mini-batches"""
            indices = np.arange(len(X))
            np.random.shuffle(indices)  # Shuffle for better training

            n_batches = len(X) // self.batch_size
            batches = []

            for i in range(n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                batch_indices = indices[start:end]
                batches.append((X[batch_indices], y[batch_indices]))

            return batches

        def forward(self, X):
            """Make predictions"""
            return self.w * X + self.b

        def compute_loss(self, X, y):
            """Calculate MSE loss"""
            predictions = self.forward(X)
            return np.mean((predictions - y) ** 2)

        def train_epoch(self, X_train, y_train, lr):
            """Train for one epoch"""
            batches = self.create_batches(X_train, y_train)
            epoch_loss = 0

            for X_batch, y_batch in batches:
                # Forward
                pred = self.forward(X_batch)
                loss = np.mean((pred - y_batch) ** 2)
                epoch_loss += loss

                # Backward
                error = pred - y_batch
                dw = 2 * np.mean(error * X_batch)
                db = 2 * np.mean(error)

                # Update
                self.w -= lr * dw
                self.b -= lr * db

            return epoch_loss / len(batches)

        def validate(self, X_val, y_val):
            """Evaluate on validation set"""
            return self.compute_loss(X_val, y_val)

        def lr_schedule(self, epoch, max_epochs):
            """Cosine annealing learning rate"""
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))

        def train(self, X_train, y_train, X_val, y_val, max_epochs=200):
            """Complete training loop"""
            print("\nStarting training...")
            print(f"Batch size: {self.batch_size}")
            print(f"Initial learning rate: {self.initial_lr}")
            print(f"Patience: {self.patience}")

            for epoch in range(max_epochs):
                # Get learning rate
                lr = self.lr_schedule(epoch, max_epochs)
                self.learning_rates.append(lr)

                # Train
                train_loss = self.train_epoch(X_train, y_train, lr)
                self.train_losses.append(train_loss)

                # Validate
                val_loss = self.validate(X_val, y_val)
                self.val_losses.append(val_loss)

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_w = self.w
                    self.best_b = self.b
                    self.patience_counter = 0
                    status = "✓"
                else:
                    self.patience_counter += 1
                    status = f"({self.patience_counter}/{self.patience})"

                # Print progress
                if epoch % 20 == 0 or epoch == max_epochs - 1:
                    print(f"Epoch {epoch:3d}: LR={lr:.5f}, "
                          f"Train={train_loss:.5f}, Val={val_loss:.5f} {status}")

                # Early stopping
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch}!")
                    break

            # Restore best model
            self.w = self.best_w
            self.b = self.best_b

            print("\n" + "="*60)
            print("Training complete!")
            print("="*60)
            print(f"Best validation loss: {self.best_val_loss:.5f}")
            print(f"Final weights: w={self.w:.3f}, b={self.b:.3f}")

        def evaluate(self, X_test, y_test):
            """Final evaluation on test set"""
            test_loss = self.compute_loss(X_test, y_test)
            print(f"\nTest set performance:")
            print(f"  Test loss: {test_loss:.5f}")
            return test_loss

    # Train model
    trainer = Trainer(batch_size=32, initial_lr=0.1, patience=20)
    trainer.train(X_train, y_train, X_val, y_val, max_epochs=200)
    trainer.evaluate(X_test, y_test)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Training curves
    ax = axes[0, 0]
    ax.plot(trainer.train_losses, label='Training', linewidth=2)
    ax.plot(trainer.val_losses, label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Learning rate
    ax = axes[0, 1]
    ax.plot(trainer.learning_rates, linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)

    # Plot 3: Predictions on test set
    ax = axes[1, 0]
    predictions_test = trainer.forward(X_test)
    ax.scatter(X_test, y_test, alpha=0.5, label='Actual', s=30)
    ax.scatter(X_test, predictions_test, alpha=0.5, label='Predicted', s=30)
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title('Test Set Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Residuals
    ax = axes[1, 1]
    residuals = predictions_test - y_test
    ax.hist(residuals, bins=20, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (Predicted - Actual)')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('exercise_05_complete_pipeline.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_05_complete_pipeline.png")
    plt.show()

    print("\n" + "="*60)
    print("CONGRATULATIONS!")
    print("="*60)
    print("\nYou've built a complete production-ready training pipeline!")
    print("This pipeline includes:")
    print("✓ Mini-batch training")
    print("✓ Train/val/test split")
    print("✓ Learning rate scheduling")
    print("✓ Early stopping")
    print("✓ Progress monitoring")
    print("✓ Model evaluation")
    print("\nThis is the same approach used to train GPT-3!")


# ============================================================================
# MAIN: Run All Exercises
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXERCISE 5: TRAINING LOOP - PRACTICE PROBLEMS")
    print("="*70)
    print("\nThis exercise covers:")
    print("1. Basic training loop")
    print("2. Mini-batch gradient descent")
    print("3. Train/validation split")
    print("4. Early stopping")
    print("5. Learning rate scheduling")
    print("6. Complete production pipeline")
    print("\n" + "="*70)

    # Run exercises (uncomment the ones you want to run)

    # Beginner
    exercise_1_simple_training()

    # Intermediate
    exercise_2_mini_batch()
    exercise_3_train_val_split()

    # Advanced
    exercise_4_early_stopping()
    exercise_5_lr_scheduling()
    exercise_6_complete_pipeline()

    print("\n" + "="*70)
    print("ALL EXERCISES COMPLETE!")
    print("="*70)
    print("\nYou now understand:")
    print("✓ How to implement training loops")
    print("✓ Mini-batch gradient descent")
    print("✓ Train/validation/test splits")
    print("✓ Early stopping to prevent overfitting")
    print("✓ Learning rate scheduling")
    print("✓ Building production-ready training pipelines")
    print("\nYou're ready to train neural networks like a pro!")
    print("Next: Apply these techniques to multi-layer networks!")

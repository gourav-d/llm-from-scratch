"""
Example 6: Optimizers - Advanced Learning Algorithms

This example demonstrates different optimization algorithms:
1. SGD vs Momentum on a simple problem
2. Visualizing optimizer paths on 2D loss surfaces
3. RMSProp for handling different scales
4. Adam optimizer implementation
5. Comparing all optimizers on a real problem
6. Learning rate effect on convergence
7. Optimizer hyperparameter tuning
8. Training a neural network with different optimizers

Run this file to see optimizers in action!
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("EXAMPLE 6: OPTIMIZERS - ADVANCED LEARNING ALGORITHMS")
print("=" * 80)
print()

# ============================================================================
# OPTIMIZER IMPLEMENTATIONS
# ============================================================================

class SGDOptimizer:
    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize SGD optimizer.

        Args:
            learning_rate: Step size for weight updates
        """
        self.learning_rate = learning_rate

    def update(self, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Update weights using vanilla SGD.

        Formula: W = W - lr * gradient

        Args:
            weights: Current weights
            gradients: Computed gradients

        Returns:
            Updated weights
        """
        # Simple gradient descent step
        weights = weights - self.learning_rate * gradients
        return weights


class MomentumOptimizer:
    """SGD with Momentum."""

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9):
        """
        Initialize Momentum optimizer.

        Args:
            learning_rate: Step size for weight updates
            beta: Momentum coefficient (typically 0.9)
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = None  # Initialized on first update

    def update(self, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Update weights using momentum.

        Formula:
            v = β*v + gradient
            W = W - lr * v

        Args:
            weights: Current weights
            gradients: Computed gradients

        Returns:
            Updated weights
        """
        # Initialize velocity on first call
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)

        # Update velocity: v = β*v + g
        # This accumulates gradients over time
        self.velocity = self.beta * self.velocity + gradients

        # Update weights using velocity
        weights = weights - self.learning_rate * self.velocity

        return weights


class RMSPropOptimizer:
    """RMSProp optimizer with adaptive learning rates."""

    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9,
                 epsilon: float = 1e-8):
        """
        Initialize RMSProp optimizer.

        Args:
            learning_rate: Step size for weight updates
            beta: Decay rate for moving average (typically 0.9)
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.cache = None  # Initialized on first update

    def update(self, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Update weights using RMSProp.

        Formula:
            cache = β*cache + (1-β)*gradient²
            W = W - lr * gradient / (√cache + ε)

        Args:
            weights: Current weights
            gradients: Computed gradients

        Returns:
            Updated weights
        """
        # Initialize cache on first call
        if self.cache is None:
            self.cache = np.zeros_like(weights)

        # Accumulate squared gradients (running average)
        # This tracks the magnitude of gradients
        self.cache = self.beta * self.cache + (1 - self.beta) * (gradients ** 2)

        # Update weights with adaptive learning rate
        # Divide by sqrt(cache) to normalize by gradient magnitude
        weights = weights - self.learning_rate * gradients / (np.sqrt(self.cache) + self.epsilon)

        return weights


class AdamOptimizer:
    """Adam optimizer combining Momentum and RMSProp."""

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Step size for weight updates
            beta1: Decay rate for first moment (typically 0.9)
            beta2: Decay rate for second moment (typically 0.999)
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment (mean of gradients)
        self.v = None  # Second moment (mean of squared gradients)
        self.t = 0     # Iteration counter

    def update(self, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Update weights using Adam.

        Formula:
            m = β1*m + (1-β1)*gradient
            v = β2*v + (1-β2)*gradient²
            m_hat = m / (1 - β1^t)
            v_hat = v / (1 - β2^t)
            W = W - lr * m_hat / (√v_hat + ε)

        Args:
            weights: Current weights
            gradients: Computed gradients

        Returns:
            Updated weights
        """
        # Initialize moments on first call
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        # Increment iteration counter
        self.t += 1

        # Update first moment (like momentum)
        # Exponential moving average of gradients
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients

        # Update second moment (like RMSProp)
        # Exponential moving average of squared gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        # Bias correction for early iterations
        # When t is small, m and v are biased toward zero
        # This correction compensates for that bias
        m_corrected = self.m / (1 - self.beta1 ** self.t)
        v_corrected = self.v / (1 - self.beta2 ** self.t)

        # Update weights
        # Combines momentum direction with adaptive learning rate
        weights = weights - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

        return weights


# ============================================================================
# EXAMPLE 1: SGD vs Momentum on a Valley Function
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 1: SGD vs Momentum on a Valley Function")
print("=" * 80)

def valley_function(x: float, y: float) -> float:
    """
    A valley-shaped function that's steep in x, gentle in y.
    This is challenging for vanilla SGD.

    Args:
        x, y: Coordinates

    Returns:
        Function value
    """
    # Steep in x direction, gentle in y direction
    return x**2 + 100 * y**2


def valley_gradient(x: float, y: float) -> Tuple[float, float]:
    """
    Gradient of valley function.

    Returns:
        (gradient_x, gradient_y)
    """
    grad_x = 2 * x
    grad_y = 200 * y
    return np.array([grad_x, grad_y])


print("\nMinimizing valley function: f(x,y) = x² + 100y²")
print("This function is 100x steeper in y than in x (creates a narrow valley)")
print()

# Starting point
start = np.array([10.0, 10.0])

# SGD path
print("Running vanilla SGD...")
sgd = SGDOptimizer(learning_rate=0.01)
sgd_path = [start.copy()]
position = start.copy()

for i in range(100):
    grad = valley_gradient(position[0], position[1])
    position = sgd.update(position, grad)
    sgd_path.append(position.copy())

sgd_path = np.array(sgd_path)
print(f"SGD final position: ({sgd_path[-1][0]:.4f}, {sgd_path[-1][1]:.4f})")
print(f"SGD final loss: {valley_function(sgd_path[-1][0], sgd_path[-1][1]):.4f}")

# Momentum path
print("\nRunning SGD with Momentum...")
momentum = MomentumOptimizer(learning_rate=0.01, beta=0.9)
momentum_path = [start.copy()]
position = start.copy()

for i in range(100):
    grad = valley_gradient(position[0], position[1])
    position = momentum.update(position, grad)
    momentum_path.append(position.copy())

momentum_path = np.array(momentum_path)
print(f"Momentum final position: ({momentum_path[-1][0]:.4f}, {momentum_path[-1][1]:.4f})")
print(f"Momentum final loss: {valley_function(momentum_path[-1][0], momentum_path[-1][1]):.4f}")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Create contour plot
x_range = np.linspace(-15, 15, 100)
y_range = np.linspace(-15, 15, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = valley_function(X, Y)

# Plot SGD path
ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
ax1.plot(sgd_path[:, 0], sgd_path[:, 1], 'r-o', markersize=3, label='SGD path')
ax1.plot(start[0], start[1], 'go', markersize=10, label='Start')
ax1.plot(0, 0, 'r*', markersize=15, label='Optimal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Vanilla SGD (zigzags in valley)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot Momentum path
ax2.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
ax2.plot(momentum_path[:, 0], momentum_path[:, 1], 'b-o', markersize=3, label='Momentum path')
ax2.plot(start[0], start[1], 'go', markersize=10, label='Start')
ax2.plot(0, 0, 'r*', markersize=15, label='Optimal')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('SGD + Momentum (smooth descent)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_paths_valley.png', dpi=150, bbox_inches='tight')
print("\n📊 Saved: optimizer_paths_valley.png")

# ============================================================================
# EXAMPLE 2: Visualizing Optimizer Behavior on 2D Loss Surface
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: Comparing All Optimizers on Rosenbrock Function")
print("=" * 80)

def rosenbrock(x: float, y: float) -> float:
    """
    Rosenbrock function: a classic optimization test problem.
    Has a narrow curved valley that's hard to optimize.

    Global minimum at (1, 1) with value 0.
    """
    return (1 - x)**2 + 100 * (y - x**2)**2


def rosenbrock_gradient(x: float, y: float) -> np.ndarray:
    """Gradient of Rosenbrock function."""
    grad_x = -2 * (1 - x) - 400 * x * (y - x**2)
    grad_y = 200 * (y - x**2)
    return np.array([grad_x, grad_y])


print("\nRosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²")
print("Minimum at (1, 1), value = 0")
print("This function has a narrow curved valley (very challenging!)")
print()

# Starting point
start = np.array([-0.5, -0.5])

# Test all optimizers
optimizers = {
    'SGD': SGDOptimizer(learning_rate=0.001),
    'Momentum': MomentumOptimizer(learning_rate=0.001, beta=0.9),
    'RMSProp': RMSPropOptimizer(learning_rate=0.01, beta=0.9),
    'Adam': AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999)
}

paths = {}
n_iterations = 200

for name, optimizer in optimizers.items():
    print(f"Running {name}...")
    path = [start.copy()]
    position = start.copy()

    for i in range(n_iterations):
        grad = rosenbrock_gradient(position[0], position[1])
        position = optimizer.update(position, grad)
        path.append(position.copy())

    paths[name] = np.array(path)
    final_loss = rosenbrock(position[0], position[1])
    print(f"  Final position: ({position[0]:.4f}, {position[1]:.4f})")
    print(f"  Final loss: {final_loss:.6f}")
    print()

# Plot all paths
fig, ax = plt.subplots(figsize=(10, 8))

# Create contour plot
x_range = np.linspace(-1, 1.5, 200)
y_range = np.linspace(-0.5, 1.5, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

# Plot contours with log scale for better visualization
ax.contour(X, Y, np.log(Z + 1), levels=20, cmap='viridis', alpha=0.4)

# Plot paths
colors = {'SGD': 'red', 'Momentum': 'orange', 'RMSProp': 'blue', 'Adam': 'green'}
for name, path in paths.items():
    ax.plot(path[:, 0], path[:, 1], '-', color=colors[name],
            linewidth=2, label=name, alpha=0.7)
    # Mark start and end
    ax.plot(path[0, 0], path[0, 1], 'o', color=colors[name], markersize=8)
    ax.plot(path[-1, 0], path[-1, 1], '*', color=colors[name], markersize=12)

# Mark optimal point
ax.plot(1, 1, 'k*', markersize=20, label='Optimal (1,1)')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Optimizer Comparison on Rosenbrock Function', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_comparison_rosenbrock.png', dpi=150, bbox_inches='tight')
print("📊 Saved: optimizer_comparison_rosenbrock.png")

# ============================================================================
# EXAMPLE 3: RMSProp Handling Different Scales
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: RMSProp for Features with Different Scales")
print("=" * 80)

print("\nProblem: Feature 1 has gradients ~100, Feature 2 has gradients ~0.01")
print("SGD uses same learning rate for both → bad!")
print("RMSProp adapts learning rate per feature → good!")
print()

def different_scales_loss(w1: float, w2: float) -> float:
    """Loss with different scales: steep in w1, gentle in w2."""
    return 10000 * (w1 - 1)**2 + (w2 - 1)**2


def different_scales_gradient(w1: float, w2: float) -> np.ndarray:
    """Gradient of different scales loss."""
    grad_w1 = 20000 * (w1 - 1)
    grad_w2 = 2 * (w2 - 1)
    return np.array([grad_w1, grad_w2])


start = np.array([0.0, 0.0])

# SGD
print("Running vanilla SGD (lr=0.00001)...")
sgd = SGDOptimizer(learning_rate=0.00001)  # Tiny LR to avoid divergence
sgd_path = [start.copy()]
position = start.copy()

for i in range(1000):
    grad = different_scales_gradient(position[0], position[1])
    position = sgd.update(position, grad)
    sgd_path.append(position.copy())

sgd_path = np.array(sgd_path)
print(f"SGD final: w1={sgd_path[-1][0]:.6f}, w2={sgd_path[-1][1]:.6f}")
print(f"SGD converged: w1={'✓' if abs(sgd_path[-1][0] - 1.0) < 0.01 else '✗'}, "
      f"w2={'✓' if abs(sgd_path[-1][1] - 1.0) < 0.01 else '✗'}")

# RMSProp
print("\nRunning RMSProp (lr=0.01)...")
rmsprop = RMSPropOptimizer(learning_rate=0.01)
rmsprop_path = [start.copy()]
position = start.copy()

for i in range(1000):
    grad = different_scales_gradient(position[0], position[1])
    position = rmsprop.update(position, grad)
    rmsprop_path.append(position.copy())

rmsprop_path = np.array(rmsprop_path)
print(f"RMSProp final: w1={rmsprop_path[-1][0]:.6f}, w2={rmsprop_path[-1][1]:.6f}")
print(f"RMSProp converged: w1={'✓' if abs(rmsprop_path[-1][0] - 1.0) < 0.01 else '✗'}, "
      f"w2={'✓' if abs(rmsprop_path[-1][1] - 1.0) < 0.01 else '✗'}")

# Plot convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# w1 convergence
ax1.plot(sgd_path[:, 0], 'r-', label='SGD', linewidth=2)
ax1.plot(rmsprop_path[:, 0], 'b-', label='RMSProp', linewidth=2)
ax1.axhline(y=1.0, color='g', linestyle='--', label='Target')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('w1 value')
ax1.set_title('Feature 1 (large gradients)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# w2 convergence
ax2.plot(sgd_path[:, 1], 'r-', label='SGD', linewidth=2)
ax2.plot(rmsprop_path[:, 1], 'b-', label='RMSProp', linewidth=2)
ax2.axhline(y=1.0, color='g', linestyle='--', label='Target')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('w2 value')
ax2.set_title('Feature 2 (small gradients)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_different_scales.png', dpi=150, bbox_inches='tight')
print("\n📊 Saved: optimizer_different_scales.png")

# ============================================================================
# EXAMPLE 4: Adam Optimizer Bias Correction
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 4: Importance of Bias Correction in Adam")
print("=" * 80)

print("\nShowing why bias correction matters in early iterations...")
print()


class AdamNoBiasCorrection:
    """Adam without bias correction (for comparison)."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

    def update(self, weights, gradients):
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        # NO bias correction!
        weights = weights - self.learning_rate * self.m / (np.sqrt(self.v) + self.epsilon)

        return weights


# Simple quadratic: f(x) = x²
def quadratic_loss(x): return x ** 2
def quadratic_grad(x): return 2 * x

start = np.array([1.0])

# Adam with bias correction
print("Running Adam WITH bias correction...")
adam = AdamOptimizer(learning_rate=0.1)
adam_path = [start.copy()]
position = start.copy()

for i in range(50):
    grad = quadratic_grad(position)
    position = adam.update(position, grad)
    adam_path.append(position.copy())

adam_path = np.array(adam_path).flatten()

# Adam without bias correction
print("Running Adam WITHOUT bias correction...")
adam_no_bc = AdamNoBiasCorrection(learning_rate=0.1)
adam_no_bc_path = [start.copy()]
position = start.copy()

for i in range(50):
    grad = quadratic_grad(position)
    position = adam_no_bc.update(position, grad)
    adam_no_bc_path.append(position.copy())

adam_no_bc_path = np.array(adam_no_bc_path).flatten()

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(adam_path, 'g-', label='Adam (with bias correction)', linewidth=2)
plt.plot(adam_no_bc_path, 'r--', label='Adam (without bias correction)', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Optimal')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Parameter value', fontsize=12)
plt.title('Bias Correction Impact on Early Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('adam_bias_correction.png', dpi=150, bbox_inches='tight')
print("\n📊 Saved: adam_bias_correction.png")

print("\nNotice:")
print("- WITH bias correction: Fast initial progress")
print("- WITHOUT bias correction: Slow start (m and v biased toward 0)")

# ============================================================================
# EXAMPLE 5: Training Neural Network with Different Optimizers
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 5: Training Neural Network - Optimizer Comparison")
print("=" * 80)

print("\nTraining a 2-layer network on XOR problem with different optimizers...")
print()

# Generate XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

print("XOR Dataset:")
print("Input → Output")
for i in range(len(X_xor)):
    print(f"{X_xor[i]} → {y_xor[i][0]}")
print()


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def forward_pass(X, W1, b1, W2, b2):
    """Forward pass through 2-layer network."""
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


def compute_loss(y_true, y_pred):
    """Binary cross-entropy loss."""
    epsilon = 1e-8
    return -np.mean(y_true * np.log(y_pred + epsilon) +
                    (1 - y_true) * np.log(1 - y_pred + epsilon))


def backward_pass(X, y, W1, b1, W2, b2, z1, a1, z2, a2):
    """Backward pass to compute gradients."""
    m = X.shape[0]

    # Output layer gradients
    dz2 = a2 - y
    dW2 = (a1.T @ dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # Hidden layer gradients
    dz1 = (dz2 @ W2.T) * sigmoid_derivative(z1)
    dW1 = (X.T @ dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


# Train with different optimizers
optimizers_to_test = {
    'SGD': SGDOptimizer(learning_rate=1.0),
    'Momentum': MomentumOptimizer(learning_rate=1.0, beta=0.9),
    'RMSProp': RMSPropOptimizer(learning_rate=0.1, beta=0.9),
    'Adam': AdamOptimizer(learning_rate=0.1)
}

results = {}
n_epochs = 1000

for opt_name, optimizer in optimizers_to_test.items():
    print(f"Training with {opt_name}...")

    # Initialize weights (same for all optimizers)
    np.random.seed(42)
    W1 = np.random.randn(2, 4) * 0.5
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros((1, 1))

    # Create separate optimizers for each parameter
    opt_W1 = type(optimizer)(**optimizer.__dict__)
    opt_b1 = type(optimizer)(**optimizer.__dict__)
    opt_W2 = type(optimizer)(**optimizer.__dict__)
    opt_b2 = type(optimizer)(**optimizer.__dict__)

    # Remove non-hyperparameter attributes
    for opt in [opt_W1, opt_b1, opt_W2, opt_b2]:
        opt.__dict__.pop('velocity', None)
        opt.__dict__.pop('cache', None)
        opt.__dict__.pop('m', None)
        opt.__dict__.pop('v', None)
        opt.__dict__.pop('t', None)

    loss_history = []

    for epoch in range(n_epochs):
        # Forward pass
        z1, a1, z2, a2 = forward_pass(X_xor, W1, b1, W2, b2)

        # Compute loss
        loss = compute_loss(y_xor, a2)
        loss_history.append(loss)

        # Backward pass
        dW1, db1, dW2, db2 = backward_pass(X_xor, y_xor, W1, b1, W2, b2, z1, a1, z2, a2)

        # Update parameters
        W1 = opt_W1.update(W1, dW1)
        b1 = opt_b1.update(b1, db1)
        W2 = opt_W2.update(W2, dW2)
        b2 = opt_b2.update(b2, db2)

    results[opt_name] = {
        'loss_history': loss_history,
        'final_loss': loss_history[-1],
        'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2
    }

    # Test final network
    _, _, _, predictions = forward_pass(X_xor, W1, b1, W2, b2)
    accuracy = np.mean((predictions > 0.5) == y_xor)

    print(f"  Final loss: {loss_history[-1]:.6f}")
    print(f"  Accuracy: {accuracy * 100:.1f}%")
    print()

# Plot training curves
plt.figure(figsize=(12, 6))

for opt_name, result in results.items():
    plt.plot(result['loss_history'], label=opt_name, linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('XOR Training - Optimizer Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('optimizer_xor_training.png', dpi=150, bbox_inches='tight')
print("📊 Saved: optimizer_xor_training.png")

# ============================================================================
# EXAMPLE 6: Learning Rate Effect
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 6: Learning Rate Effect on Adam Optimizer")
print("=" * 80)

print("\nTesting different learning rates on quadratic function...")
print()

learning_rates = [0.001, 0.01, 0.1, 0.5]
lr_results = {}

for lr in learning_rates:
    print(f"Testing learning rate = {lr}...")
    adam = AdamOptimizer(learning_rate=lr)
    path = [np.array([5.0])]  # Start far from minimum
    position = np.array([5.0])

    for i in range(100):
        grad = quadratic_grad(position)
        position = adam.update(position, grad)
        path.append(position.copy())

    lr_results[lr] = np.array(path).flatten()
    print(f"  Final value: {path[-1][0]:.6f} (target: 0.0)")

# Plot
plt.figure(figsize=(10, 6))

for lr, path in lr_results.items():
    plt.plot(path, label=f'lr={lr}', linewidth=2)

plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Optimal')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Parameter value', fontsize=12)
plt.title('Learning Rate Effect on Convergence Speed', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('optimizer_learning_rate_effect.png', dpi=150, bbox_inches='tight')
print("\n📊 Saved: optimizer_learning_rate_effect.png")

# ============================================================================
# EXAMPLE 7: Momentum Coefficient Effect
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 7: Momentum Coefficient (Beta) Effect")
print("=" * 80)

print("\nTesting different beta values for Momentum optimizer...")
print()

betas = [0.0, 0.5, 0.9, 0.99]
beta_results = {}

for beta in betas:
    print(f"Testing beta = {beta}...")
    momentum = MomentumOptimizer(learning_rate=0.01, beta=beta)
    path = [np.array([10.0, 10.0])]
    position = np.array([10.0, 10.0])

    for i in range(100):
        grad = valley_gradient(position[0], position[1])
        position = momentum.update(position, grad)
        path.append(position.copy())

    beta_results[beta] = np.array(path)
    final_loss = valley_function(position[0], position[1])
    print(f"  Final loss: {final_loss:.4f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Create contour plot
x_range = np.linspace(-15, 15, 100)
y_range = np.linspace(-15, 15, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = valley_function(X, Y)

# Plot paths
ax1.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.4)
for beta, path in beta_results.items():
    ax1.plot(path[:, 0], path[:, 1], '-', linewidth=2, label=f'β={beta}', alpha=0.7)
ax1.plot(0, 0, 'r*', markersize=15, label='Optimal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Momentum Beta Effect on Path')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot loss over time
for beta, path in beta_results.items():
    losses = [valley_function(p[0], p[1]) for p in path]
    ax2.plot(losses, linewidth=2, label=f'β={beta}')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Convergence by Beta')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('optimizer_momentum_beta_effect.png', dpi=150, bbox_inches='tight')
print("\n📊 Saved: optimizer_momentum_beta_effect.png")

# ============================================================================
# EXAMPLE 8: Summary and Best Practices
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 8: Summary and Recommendations")
print("=" * 80)

print("""
KEY TAKEAWAYS:

1. VANILLA SGD
   - Simple but slow
   - Sensitive to learning rate
   - Good for final fine-tuning
   - Use when you have time to tune carefully

2. MOMENTUM
   - Adds velocity to smooth out path
   - Beta=0.9 is standard
   - Good for CNNs and vision tasks
   - Faster than vanilla SGD

3. RMSPROP
   - Adapts learning rate per parameter
   - Handles different feature scales
   - Good for RNNs
   - Less popular than Adam

4. ADAM (MOST POPULAR!)
   - Combines Momentum + RMSProp
   - Works "out of the box"
   - Used for GPT-2, GPT-3, BERT
   - Default choice for transformers
   - Standard: lr=0.001, beta1=0.9, beta2=0.999

DECISION GUIDE:

Q: Training a Transformer/GPT model?
   → Use Adam (lr=3e-4)

Q: Training a CNN (ResNet, VGG)?
   → Use SGD+Momentum (lr=0.1 with decay)

Q: Training an RNN/LSTM?
   → Use Adam or RMSProp (lr=0.001)

Q: Not sure?
   → Start with Adam (lr=0.001)

Q: Adam not generalizing well?
   → Try SGD+Momentum with learning rate schedule

COMMON HYPERPARAMETERS:

Adam:
  learning_rate = 0.001  (or 3e-4 for transformers)
  beta1 = 0.9
  beta2 = 0.999

SGD + Momentum:
  learning_rate = 0.01 (or 0.1 for CNNs)
  momentum = 0.9

RMSProp:
  learning_rate = 0.001
  beta = 0.9

LEARNING RATE SCHEDULES:

Warmup + Decay (for Transformers):
  1. Warmup: Linear increase for first 1-10% of steps
  2. Decay: Cosine decay or inverse sqrt after warmup

Step Decay (for CNNs):
  Reduce LR by 10x every 30 epochs

CONNECTION TO GPT:

GPT-3 was trained with:
  - Adam optimizer
  - Learning rate = 0.0006 (6e-4)
  - Beta1 = 0.9
  - Beta2 = 0.95 (lower than standard!)
  - Warmup + cosine decay schedule
  - 300 billion tokens
  - 1000+ GPUs
  - ~$5 million cost!

The Adam optimizer you implemented is THE SAME algorithm used to train GPT-3!
""")

print("\n" + "=" * 80)
print("All examples completed successfully!")
print("=" * 80)
print(f"\n📊 Generated {7} visualization plots")
print("\nFiles saved:")
print("  - optimizer_paths_valley.png")
print("  - optimizer_comparison_rosenbrock.png")
print("  - optimizer_different_scales.png")
print("  - adam_bias_correction.png")
print("  - optimizer_xor_training.png")
print("  - optimizer_learning_rate_effect.png")
print("  - optimizer_momentum_beta_effect.png")
print("\n✅ You now understand how modern neural networks (including GPT!) are optimized!")
print("🎉 Module 3: Neural Networks - COMPLETE!")

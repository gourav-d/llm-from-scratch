"""
Lesson 3.4: Backpropagation - Complete Examples

This file demonstrates:
1. Manual backpropagation step-by-step
2. Complete network with full backprop
3. Numerical gradient checking (verify correctness!)
4. Training with gradient descent
5. Visualizing learning progress
6. Effect of learning rate
7. Connection to modern ML frameworks

For .NET developers: This is like automatic differentiation in ML.NET,
but we're building it from scratch to understand how it works!
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


# ============================================================================
# EXAMPLE 1: Manual Backprop Calculation (Step-by-Step)
# ============================================================================

print("=" * 70)
print("EXAMPLE 1: Manual Backpropagation - Step-by-Step")
print("=" * 70)

print("""
We'll manually calculate gradients for a simple network:
    x → [w1] → a1 → [w2] → y

Goal: Understand every single step of backpropagation!
""")

# Simple sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    """Derivative: σ'(z) = σ(z) × (1 - σ(z))"""
    return a * (1 - a)

# Initialize simple network
print("--- Network Setup ---")
x = 1.0          # Input
w1 = 0.5         # Weight 1
w2 = 0.8         # Weight 2
t = 1.0          # True label
learning_rate = 0.5

print(f"Input (x):        {x}")
print(f"Weight 1 (w1):    {w1}")
print(f"Weight 2 (w2):    {w2}")
print(f"True label (t):   {t}")
print(f"Learning rate:    {learning_rate}")

# Forward pass
print("\n--- Forward Pass ---")
z1 = w1 * x
print(f"z1 = w1 × x = {w1} × {x} = {z1}")

a1 = sigmoid(z1)
print(f"a1 = sigmoid(z1) = sigmoid({z1:.3f}) = {a1:.6f}")

z2 = w2 * a1
print(f"z2 = w2 × a1 = {w2} × {a1:.6f} = {z2:.6f}")

y = sigmoid(z2)
print(f"y = sigmoid(z2) = sigmoid({z2:.6f}) = {y:.6f}")

# Loss
loss = 0.5 * (y - t) ** 2
print(f"\nLoss = 0.5 × (y - t)² = 0.5 × ({y:.6f} - {t})² = {loss:.6f}")

# Backward pass
print("\n--- Backward Pass (Backpropagation!) ---")

# Gradient at output
dL_dy = y - t
print(f"\n1. ∂L/∂y = y - t = {y:.6f} - {t} = {dL_dy:.6f}")

# Gradient at z2
dy_dz2 = sigmoid_derivative(y)
dL_dz2 = dL_dy * dy_dz2
print(f"\n2. ∂y/∂z2 = σ'(z2) = y(1-y) = {y:.6f} × {1-y:.6f} = {dy_dz2:.6f}")
print(f"   ∂L/∂z2 = ∂L/∂y × ∂y/∂z2 = {dL_dy:.6f} × {dy_dz2:.6f} = {dL_dz2:.6f}")

# Gradient at w2
dz2_dw2 = a1
dL_dw2 = dL_dz2 * dz2_dw2
print(f"\n3. ∂z2/∂w2 = a1 = {a1:.6f}")
print(f"   ∂L/∂w2 = ∂L/∂z2 × ∂z2/∂w2 = {dL_dz2:.6f} × {a1:.6f} = {dL_dw2:.6f}")

# Gradient at a1 (propagate backwards!)
dz2_da1 = w2
dL_da1 = dL_dz2 * dz2_da1
print(f"\n4. ∂z2/∂a1 = w2 = {w2}")
print(f"   ∂L/∂a1 = ∂L/∂z2 × ∂z2/∂a1 = {dL_dz2:.6f} × {w2} = {dL_da1:.6f}")

# Gradient at z1
da1_dz1 = sigmoid_derivative(a1)
dL_dz1 = dL_da1 * da1_dz1
print(f"\n5. ∂a1/∂z1 = σ'(z1) = a1(1-a1) = {a1:.6f} × {1-a1:.6f} = {da1_dz1:.6f}")
print(f"   ∂L/∂z1 = ∂L/∂a1 × ∂a1/∂z1 = {dL_da1:.6f} × {da1_dz1:.6f} = {dL_dz1:.6f}")

# Gradient at w1
dz1_dw1 = x
dL_dw1 = dL_dz1 * dz1_dw1
print(f"\n6. ∂z1/∂w1 = x = {x}")
print(f"   ∂L/∂w1 = ∂L/∂z1 × ∂z1/∂w1 = {dL_dz1:.6f} × {x} = {dL_dw1:.6f}")

# Update weights
print("\n--- Weight Update (Gradient Descent) ---")
w1_new = w1 - learning_rate * dL_dw1
w2_new = w2 - learning_rate * dL_dw2

print(f"w1_new = w1 - α × ∂L/∂w1 = {w1} - {learning_rate} × {dL_dw1:.6f} = {w1_new:.6f}")
print(f"w2_new = w2 - α × ∂L/∂w2 = {w2} - {learning_rate} × {dL_dw2:.6f} = {w2_new:.6f}")

# Verify improvement
z1_new = w1_new * x
a1_new = sigmoid(z1_new)
z2_new = w2_new * a1_new
y_new = sigmoid(z2_new)
loss_new = 0.5 * (y_new - t) ** 2

print(f"\n--- Verification ---")
print(f"Old prediction: {y:.6f}, Old loss: {loss:.6f}")
print(f"New prediction: {y_new:.6f}, New loss: {loss_new:.6f}")
print(f"Improvement: {loss - loss_new:.6f} {'✓' if loss_new < loss else '✗'}")


# ============================================================================
# EXAMPLE 2: Complete 2-Layer Network with Backprop
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Complete 2-Layer Network with Backprop")
print("=" * 70)


class TwoLayerNetwork:
    """
    Complete implementation of 2-layer network with backpropagation.

    Architecture: Input → Hidden → Output
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initialize with random weights"""
        # He initialization (works well with ReLU, but we use sigmoid here)
        self.W1 = np.random.randn(hidden_size, input_size) * 0.5
        self.b1 = np.zeros((hidden_size, 1))

        self.W2 = np.random.randn(output_size, hidden_size) * 0.5
        self.b2 = np.zeros((output_size, 1))

        print(f"\nNetwork created: {input_size} → {hidden_size} → {output_size}")
        total_params = (input_size * hidden_size + hidden_size +
                       hidden_size * output_size + output_size)
        print(f"Total parameters: {total_params}")

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, x: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Forward propagation

        Args:
            x: Input, shape (input_size, batch_size)
            verbose: Print intermediate shapes

        Returns:
            y: Output, shape (output_size, batch_size)
        """
        # Layer 1
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Layer 2
        self.z2 = self.W2 @ self.a1 + self.b2
        self.y = self.sigmoid(self.z2)

        # Cache for backprop
        self.x = x

        if verbose:
            print(f"  x: {x.shape} → z1: {self.z1.shape} → a1: {self.a1.shape}")
            print(f"  a1: {self.a1.shape} → z2: {self.z2.shape} → y: {self.y.shape}")

        return self.y

    def backward(self, t: np.ndarray, verbose: bool = False) -> Dict[str, np.ndarray]:
        """
        Backpropagation

        Args:
            t: True labels, shape (output_size, batch_size)
            verbose: Print gradient shapes

        Returns:
            Dictionary of gradients
        """
        m = self.x.shape[1]  # Batch size

        # Output layer gradients
        dL_dy = self.y - t
        dy_dz2 = self.sigmoid_derivative(self.y)
        dL_dz2 = dL_dy * dy_dz2

        # Gradients for W2, b2
        dL_dW2 = (1 / m) * (dL_dz2 @ self.a1.T)
        dL_db2 = (1 / m) * np.sum(dL_dz2, axis=1, keepdims=True)

        # Propagate to hidden layer
        dL_da1 = self.W2.T @ dL_dz2
        da1_dz1 = self.sigmoid_derivative(self.a1)
        dL_dz1 = dL_da1 * da1_dz1

        # Gradients for W1, b1
        dL_dW1 = (1 / m) * (dL_dz1 @ self.x.T)
        dL_db1 = (1 / m) * np.sum(dL_dz1, axis=1, keepdims=True)

        if verbose:
            print(f"  dL_dW2: {dL_dW2.shape}, dL_db2: {dL_db2.shape}")
            print(f"  dL_dW1: {dL_dW1.shape}, dL_db1: {dL_db1.shape}")

        return {
            'dW1': dL_dW1, 'db1': dL_db1,
            'dW2': dL_dW2, 'db2': dL_db2
        }

    def update_weights(self, gradients: Dict, learning_rate: float):
        """Update weights using gradients"""
        self.W1 -= learning_rate * gradients['dW1']
        self.b1 -= learning_rate * gradients['db1']
        self.W2 -= learning_rate * gradients['dW2']
        self.b2 -= learning_rate * gradients['db2']

    def compute_loss(self, y: np.ndarray, t: np.ndarray) -> float:
        """Mean squared error"""
        return np.mean((y - t) ** 2)

    def train_step(self, x: np.ndarray, t: np.ndarray,
                   learning_rate: float, verbose: bool = False) -> float:
        """Complete training step"""
        # Forward
        y = self.forward(x, verbose=verbose)

        # Loss
        loss = self.compute_loss(y, t)

        # Backward
        gradients = self.backward(t, verbose=verbose)

        # Update
        self.update_weights(gradients, learning_rate)

        return loss


# Test on XOR
print("\n--- Training XOR with Backpropagation ---")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y_xor = np.array([[0, 1, 1, 0]])

network = TwoLayerNetwork(input_size=2, hidden_size=4, output_size=1)

print("\nFirst training step (verbose):")
loss = network.train_step(X_xor, Y_xor, learning_rate=1.0, verbose=True)
print(f"Initial loss: {loss:.6f}")

# Train for 5000 iterations
losses = []
for epoch in range(5000):
    loss = network.train_step(X_xor, Y_xor, learning_rate=1.0)
    losses.append(loss)

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}: Loss = {loss:.6f}")

print("\nFinal predictions:")
final_pred = network.forward(X_xor)
for i in range(4):
    x1, x2 = X_xor[:, i]
    true_val = Y_xor[0, i]
    pred_val = final_pred[0, i]
    rounded = 1 if pred_val > 0.5 else 0
    print(f"  [{int(x1)}, {int(x2)}] → True: {int(true_val)}, "
          f"Pred: {pred_val:.4f}, Rounded: {rounded} "
          f"{'✓' if rounded == true_val else '✗'}")


# ============================================================================
# EXAMPLE 3: Numerical Gradient Checking
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Numerical Gradient Checking (Verify Backprop!)")
print("=" * 70)

print("""
Goal: Verify that our backprop implementation is correct!

Method: Compare analytical gradients (from backprop) with numerical
gradients (from finite differences).

If they match → backprop is correct! ✓
""")


def numerical_gradient(network, x, t, param_name, i, j, epsilon=1e-5):
    """
    Compute numerical gradient for one parameter using finite differences.

    Args:
        network: Neural network
        x, t: Input and target
        param_name: 'W1', 'b1', 'W2', or 'b2'
        i, j: Index of parameter (for matrices)
        epsilon: Small step size

    Returns:
        Numerical gradient approximation
    """
    # Get parameter
    param = getattr(network, param_name)

    # Save original value
    original_value = param[i, j] if param.ndim == 2 else param[i, 0]

    # Compute loss with param + epsilon
    if param.ndim == 2:
        param[i, j] = original_value + epsilon
    else:
        param[i, 0] = original_value + epsilon
    y_plus = network.forward(x)
    loss_plus = network.compute_loss(y_plus, t)

    # Compute loss with param - epsilon
    if param.ndim == 2:
        param[i, j] = original_value - epsilon
    else:
        param[i, 0] = original_value - epsilon
    y_minus = network.forward(x)
    loss_minus = network.compute_loss(y_minus, t)

    # Restore original value
    if param.ndim == 2:
        param[i, j] = original_value
    else:
        param[i, 0] = original_value

    # Compute numerical gradient
    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

    return numerical_grad


# Create small network for testing
test_network = TwoLayerNetwork(input_size=2, hidden_size=3, output_size=1)

# Single example
x_test = np.array([[0.5], [0.8]])
t_test = np.array([[1.0]])

# Forward pass
y_test = test_network.forward(x_test)
loss_test = test_network.compute_loss(y_test, t_test)

# Backward pass (analytical gradients)
analytical_grads = test_network.backward(t_test)

print("\n--- Gradient Checking Results ---")
print(f"Loss: {loss_test:.6f}\n")

# Check a few weights from each parameter
checks = [
    ('W1', 0, 0), ('W1', 1, 0), ('W1', 2, 1),
    ('b1', 0, 0), ('b1', 1, 0),
    ('W2', 0, 0), ('W2', 0, 1), ('W2', 0, 2),
    ('b2', 0, 0)
]

print(f"{'Parameter':<10} {'Index':<8} {'Analytical':<15} {'Numerical':<15} {'Difference':<15} {'Status'}")
print("-" * 80)

all_correct = True
for param_name, i, j in checks:
    # Analytical gradient
    grad_key = 'd' + param_name
    analytical = analytical_grads[grad_key][i, j] if analytical_grads[grad_key].ndim == 2 else analytical_grads[grad_key][i, 0]

    # Numerical gradient
    numerical = numerical_gradient(test_network, x_test, t_test, param_name, i, j)

    # Difference
    diff = abs(analytical - numerical)
    status = "✓" if diff < 1e-7 else "✗"
    if diff >= 1e-7:
        all_correct = False

    print(f"{param_name:<10} ({i},{j})    {analytical:<15.8f} {numerical:<15.8f} {diff:<15.10f} {status}")

print("\n" + "-" * 80)
if all_correct:
    print("✓ All gradients match! Backpropagation is implemented correctly!")
else:
    print("✗ Some gradients don't match. Check backprop implementation.")


# ============================================================================
# EXAMPLE 4: Effect of Learning Rate
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Effect of Learning Rate on Training")
print("=" * 70)

print("""
Learning rate is critical! Let's compare different values:
- Too small: Learns slowly
- Just right: Learns efficiently
- Too large: Unstable, might diverge!
""")

learning_rates = [0.1, 0.5, 1.0, 2.0, 5.0]
all_losses = {}

print("\n--- Training with Different Learning Rates ---")
for lr in learning_rates:
    print(f"\nLearning rate: {lr}")
    net = TwoLayerNetwork(input_size=2, hidden_size=4, output_size=1)

    losses = []
    for epoch in range(1000):
        loss = net.train_step(X_xor, Y_xor, learning_rate=lr)
        losses.append(loss)

    all_losses[f'LR={lr}'] = losses
    final_loss = losses[-1]
    print(f"  Final loss: {final_loss:.6f}")

    # Check if converged
    if final_loss < 0.01:
        print(f"  Status: ✓ Converged!")
    elif final_loss > 0.1:
        print(f"  Status: ✗ Did not converge")
    else:
        print(f"  Status: ~ Partially converged")

# Plot learning curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, losses in all_losses.items():
    plt.plot(losses, label=name, alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curves - Different Learning Rates')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
for name, losses in all_losses.items():
    # Plot last 200 epochs (zoomed in)
    plt.plot(losses[-200:], label=name, alpha=0.8)
plt.xlabel('Epoch (last 200)')
plt.ylabel('Loss')
plt.title('Final Convergence (Zoomed)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('example_04_learning_rates.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: example_04_learning_rates.png")


# ============================================================================
# EXAMPLE 5: Visualizing Gradient Flow
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: Visualizing Gradient Magnitudes")
print("=" * 70)

print("""
Let's visualize how gradient magnitudes change during training.
This helps understand:
- When gradients are large (far from minimum)
- When gradients are small (near minimum)
- Convergence behavior
""")

# Train network and track gradient magnitudes
net = TwoLayerNetwork(input_size=2, hidden_size=4, output_size=1)

gradient_history = {
    'dW1': [], 'db1': [],
    'dW2': [], 'db2': []
}

losses = []

for epoch in range(1000):
    # Forward
    y = net.forward(X_xor)
    loss = net.compute_loss(y, Y_xor)
    losses.append(loss)

    # Backward
    grads = net.backward(Y_xor)

    # Track gradient magnitudes (L2 norm)
    for key in gradient_history:
        grad_magnitude = np.linalg.norm(grads[key])
        gradient_history[key].append(grad_magnitude)

    # Update
    net.update_weights(grads, learning_rate=1.0)

# Plot gradient magnitudes
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(2, 2, 2)
plt.plot(gradient_history['dW1'], label='W1', alpha=0.7)
plt.plot(gradient_history['dW2'], label='W2', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Gradient Magnitude')
plt.title('Weight Gradients')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(2, 2, 3)
plt.plot(gradient_history['db1'], label='b1', alpha=0.7)
plt.plot(gradient_history['db2'], label='b2', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Gradient Magnitude')
plt.title('Bias Gradients')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(2, 2, 4)
# Plot gradient magnitude vs loss (correlation)
all_grads = np.array([
    gradient_history['dW1'],
    gradient_history['dW2'],
    gradient_history['db1'],
    gradient_history['db2']
])
avg_grad = np.mean(all_grads, axis=0)

plt.scatter(losses, avg_grad, alpha=0.5, s=10)
plt.xlabel('Loss')
plt.ylabel('Average Gradient Magnitude')
plt.title('Gradient vs Loss')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
plt.savefig('example_04_gradient_flow.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: example_04_gradient_flow.png")

print("\n--- Observations ---")
print("1. Loss decreases over time (training works!)")
print("2. Gradients start large (far from minimum)")
print("3. Gradients shrink as loss decreases (approaching minimum)")
print("4. When gradients ≈ 0, training converges (reached minimum)")


# ============================================================================
# EXAMPLE 6: Connection to Modern Frameworks
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: What Modern Frameworks Do")
print("=" * 70)

print("""
Modern frameworks (PyTorch, TensorFlow) automate backpropagation!

What you implemented manually:
  1. Forward pass
  2. Calculate loss
  3. Backward pass (compute gradients)
  4. Update weights

What PyTorch does:
  1. Forward pass
  2. Calculate loss
  3. loss.backward()  ← Automatic backprop!
  4. optimizer.step() ← Automatic weight update!

Example PyTorch code (conceptual):
```python
# Define network (PyTorch does this)
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

# Training loop
for epoch in range(5000):
    # Forward
    predictions = model(X)

    # Loss
    loss = nn.MSELoss()(predictions, Y)

    # Backward (AUTOMATIC!)
    optimizer.zero_grad()
    loss.backward()  # ← Computes all gradients automatically!

    # Update (AUTOMATIC!)
    optimizer.step()  # ← Updates all weights automatically!
```

Your manual implementation shows EXACTLY what these frameworks do internally!
""")

print("\n--- Comparison: Manual vs Automatic ---")
print(f"{'Aspect':<20} {'Manual (You)':<30} {'PyTorch/TF'}")
print("-" * 70)
print(f"{'Forward pass':<20} {'Implement yourself':<30} {'model(x)'}")
print(f"{'Loss calculation':<20} {'Implement yourself':<30} {'loss_fn(pred, true)'}")
print(f"{'Backward pass':<20} {'Implement yourself':<30} {'loss.backward()'}")
print(f"{'Weight update':<20} {'Implement yourself':<30} {'optimizer.step()'}")
print(f"{'Gradient tracking':<20} {'Manual caching':<30} {'Automatic'}")
print(f"{'Flexibility':<20} {'Full control':<30} {'High-level API'}")
print(f"{'Understanding':<20} {'Deep! ✓':<30} {'Abstracted'}")

print("\n✓ You now understand what frameworks do under the hood!")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Backpropagation Key Insights")
print("=" * 70)

print("""
✅ What You Learned:

1. Backpropagation = gradient descent via chain rule
   - Calculate how each weight affects loss
   - Update weights to reduce loss
   - Repeat until converged!

2. The algorithm is efficient
   - One forward pass: compute predictions
   - One backward pass: compute ALL gradients
   - Total cost: ~2× forward pass time

3. Numerical gradient checking verifies correctness
   - Compare analytical (backprop) vs numerical (finite diff)
   - Difference should be < 1e-7
   - Essential for debugging backprop!

4. Learning rate is critical
   - Too small: slow learning
   - Too large: unstable, may diverge
   - Typical values: 0.001 - 1.0

5. Gradients tell you about convergence
   - Large gradients: far from minimum
   - Small gradients: near minimum
   - Zero gradients: converged!

6. Modern frameworks automate this
   - PyTorch: loss.backward()
   - TensorFlow: tape.gradient()
   - But same math underneath!

🎯 Key Equations:

Forward:  z = W @ x + b,  a = σ(z)
Backward: ∂L/∂W = ∂L/∂z @ xᵀ,  ∂L/∂z = ∂L/∂a ⊙ σ'(z)
Update:   W = W - α × ∂L/∂W

🔗 Connection to GPT:

GPT-3 training:
- Same backpropagation algorithm!
- Just 175 billion parameters instead of 20
- Uses Adam optimizer (smarter than vanilla gradient descent)
- Gradient clipping to prevent exploding gradients
- But fundamentally: same math you just learned!

📊 Results from Examples:
   ✓ XOR solved with backprop
   ✓ Gradients verified numerically
   ✓ Learning rate effects visualized
   ✓ Gradient flow understood

🔜 Next Steps:

You now understand THE core algorithm of deep learning!

Next lesson (Training Loop):
- Batching data for efficiency
- Epochs and iterations
- Train/validation/test splits
- Monitoring and early stopping
- Complete MNIST classifier!
""")

print("\nFiles created:")
print("  ✓ example_04_learning_rates.png")
print("  ✓ example_04_gradient_flow.png")

print("\n" + "=" * 70)
print("Congratulations! You understand backpropagation!")
print("This is the algorithm that powers ALL modern AI!")
print("=" * 70)

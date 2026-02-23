"""
Lesson 3.3: Multi-Layer Neural Networks - Complete Examples

This file demonstrates:
1. Building multi-layer networks from scratch
2. Forward propagation through multiple layers
3. Shape debugging and management
4. Solving XOR problem (impossible with single layer!)
5. Visualizing layer transformations
6. Connection to GPT architecture

For .NET developers: Think of layers as LINQ chaining - each transformation
builds on the previous one!
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


# ============================================================================
# EXAMPLE 1: Simple 2-Layer Network (Understanding Shapes)
# ============================================================================

print("=" * 70)
print("EXAMPLE 1: Simple 2-Layer Network - Understanding Shapes")
print("=" * 70)

class TwoLayerNetwork:
    """
    Simplest multi-layer network: Input → Hidden → Output

    For .NET devs: Like a class with Forward() method
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize network with random weights.

        Args:
            input_size: Number of input features
            hidden_size: Neurons in hidden layer
            output_size: Number of output classes
        """
        # Layer 1: input → hidden
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))

        # Layer 2: hidden → output
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

        print(f"\nNetwork created: {input_size} → {hidden_size} → {output_size}")
        print(f"  W1 shape: {self.W1.shape}  (hidden × input)")
        print(f"  b1 shape: {self.b1.shape}  (hidden × 1)")
        print(f"  W2 shape: {self.W2.shape}  (output × hidden)")
        print(f"  b2 shape: {self.b2.shape}  (output × 1)")

    def relu(self, z):
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)

    def softmax(self, z):
        """Softmax: converts scores → probabilities"""
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, x, verbose=False):
        """
        Forward propagation with detailed printing.

        Args:
            x: Input, shape (input_size, 1) or (input_size, batch_size)
            verbose: If True, print intermediate shapes

        Returns:
            y: Predictions, shape (output_size, 1) or (output_size, batch_size)
        """
        if verbose:
            print(f"\n  Input x shape: {x.shape}")

        # Layer 1: Linear → ReLU
        z1 = self.W1 @ x + self.b1
        a1 = self.relu(z1)

        if verbose:
            print(f"  After layer 1:")
            print(f"    z1 = W1@x + b1, shape: {z1.shape}")
            print(f"    a1 = ReLU(z1), shape: {a1.shape}")

        # Layer 2: Linear → Softmax
        z2 = self.W2 @ a1 + self.b2
        y = self.softmax(z2)

        if verbose:
            print(f"  After layer 2:")
            print(f"    z2 = W2@a1 + b2, shape: {z2.shape}")
            print(f"    y = Softmax(z2), shape: {y.shape}")

        return y


# Create network: 5 inputs → 10 hidden → 3 outputs
network = TwoLayerNetwork(input_size=5, hidden_size=10, output_size=3)

# Test with single example
x_single = np.random.randn(5, 1)
print("\n--- Single Example ---")
y_single = network.forward(x_single, verbose=True)
print(f"\nPredictions: {y_single.flatten()}")
print(f"Sum of probabilities: {np.sum(y_single):.6f} (should be 1.0)")

# Test with batch of 4 examples
x_batch = np.random.randn(5, 4)
print("\n--- Batch of 4 Examples ---")
y_batch = network.forward(x_batch, verbose=True)
print(f"\nPredictions shape: {y_batch.shape}")
print(f"Each column sums to 1.0: {np.sum(y_batch, axis=0)}")


# ============================================================================
# EXAMPLE 2: Three-Layer Network (Deep Learning!)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Three-Layer Network - Going Deeper!")
print("=" * 70)


class ThreeLayerNetwork:
    """
    Deeper network: Input → Hidden1 → Hidden2 → Output

    This is "deep learning" - multiple hidden layers!
    """

    def __init__(self, layer_sizes: List[int]):
        """
        Initialize network with arbitrary layer sizes.

        Args:
            layer_sizes: [input_size, hidden1, hidden2, output_size]

        Example: [784, 128, 64, 10] for MNIST
        """
        self.layer_sizes = layer_sizes
        self.weights = {}
        self.biases = {}

        # Initialize each layer
        for i in range(len(layer_sizes) - 1):
            layer_num = i + 1
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i + 1]

            # Xavier initialization (better than random!)
            self.weights[f'W{layer_num}'] = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
            self.biases[f'b{layer_num}'] = np.zeros((output_dim, 1))

        print(f"\nCreated network: {' → '.join(map(str, layer_sizes))}")
        for i in range(1, len(layer_sizes)):
            print(f"  W{i}: {self.weights[f'W{i}'].shape}, b{i}: {self.biases[f'b{i}'].shape}")

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, x, return_intermediates=False):
        """
        Forward propagation through all layers.

        Args:
            x: Input data
            return_intermediates: If True, return all layer outputs

        Returns:
            y: Final predictions
            intermediates: (optional) Dict of layer outputs
        """
        activations = {'a0': x}  # Input is "activation 0"

        # Forward through all layers
        for i in range(1, len(self.layer_sizes)):
            layer_num = i
            prev_activation = activations[f'a{i-1}']

            # Linear transformation
            z = self.weights[f'W{layer_num}'] @ prev_activation + self.biases[f'b{layer_num}']

            # Activation function
            if i < len(self.layer_sizes) - 1:
                # Hidden layers: ReLU
                a = self.relu(z)
            else:
                # Output layer: Softmax
                a = self.softmax(z)

            activations[f'z{i}'] = z
            activations[f'a{i}'] = a

        if return_intermediates:
            return activations[f'a{len(self.layer_sizes)-1}'], activations
        else:
            return activations[f'a{len(self.layer_sizes)-1}']


# Create MNIST-like network: 784 → 128 → 64 → 10
mnist_network = ThreeLayerNetwork([784, 128, 64, 10])

# Simulate flattened MNIST image
fake_image = np.random.randn(784, 1)
predictions, intermediates = mnist_network.forward(fake_image, return_intermediates=True)

print("\n--- Forward Pass Through Network ---")
print(f"Input (a0):          {intermediates['a0'].shape}")
print(f"After layer 1 (a1):  {intermediates['a1'].shape}  ← First hidden layer")
print(f"After layer 2 (a2):  {intermediates['a2'].shape}  ← Second hidden layer")
print(f"Output (a3):         {intermediates['a3'].shape}  ← Predictions")

print(f"\nPredicted class: {np.argmax(predictions)}")
print(f"Confidence: {np.max(predictions):.4f}")


# ============================================================================
# EXAMPLE 3: Solving XOR Problem (Proof That Depth Matters!)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: XOR Problem - Why Single Layers Fail")
print("=" * 70)

# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # Shape: (2, 4)
Y_xor = np.array([[0, 1, 1, 0]])                      # Shape: (1, 4)

print("\nXOR Truth Table:")
print("Input 1 | Input 2 | Output")
print("--------|---------|-------")
for i in range(4):
    print(f"   {int(X_xor[0,i])}    |    {int(X_xor[1,i])}    |   {int(Y_xor[0,i])}")


class XORNetwork:
    """
    Network to solve XOR: 2 inputs → 4 hidden → 1 output

    Uses sigmoid activation for simplicity.
    """

    def __init__(self):
        # Layer 1: 2 → 4
        self.W1 = np.random.randn(4, 2) * 0.5
        self.b1 = np.zeros((4, 1))

        # Layer 2: 4 → 1
        self.W2 = np.random.randn(1, 4) * 0.5
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        """Sigmoid: 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip for numerical stability

    def forward(self, x):
        """Forward pass"""
        # Layer 1
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Layer 2
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def train_step(self, x, y, learning_rate=0.5):
        """
        One training iteration (simplified backpropagation).

        Don't worry about details - you'll learn in Lesson 4!
        """
        # Forward pass
        predictions = self.forward(x)

        # Loss
        loss = np.mean((predictions - y) ** 2)

        # Backpropagation (simplified)
        m = x.shape[1]  # Number of examples

        # Output layer gradients
        dz2 = (predictions - y) * predictions * (1 - predictions)
        dW2 = (1 / m) * (dz2 @ self.a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        # Hidden layer gradients
        dz1 = (self.W2.T @ dz2) * self.a1 * (1 - self.a1)
        dW1 = (1 / m) * (dz1 @ x.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return loss


# Train XOR network
print("\n--- Training XOR Network ---")
xor_net = XORNetwork()

losses = []
for epoch in range(5000):
    loss = xor_net.train_step(X_xor, Y_xor, learning_rate=1.0)
    losses.append(loss)

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}: Loss = {loss:.6f}")

print("\n--- Final XOR Predictions ---")
final_preds = xor_net.forward(X_xor)

print("Input  | Expected | Predicted | Rounded")
print("-------|----------|-----------|--------")
for i in range(4):
    x1, x2 = X_xor[:, i]
    expected = Y_xor[0, i]
    predicted = final_preds[0, i]
    rounded = 1 if predicted > 0.5 else 0
    print(f"[{int(x1)}, {int(x2)}] |    {int(expected)}     |   {predicted:.4f}  |   {rounded}")

# Plot learning curve
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('XOR Learning Curve - Multi-Layer Network')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('example_03_xor_learning_curve.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: example_03_xor_learning_curve.png")


# ============================================================================
# EXAMPLE 4: Visualizing Decision Boundaries (2D Classification)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Decision Boundaries - Single vs Multi-Layer")
print("=" * 70)


def create_spiral_dataset(n_points=100, noise=0.1):
    """
    Create 2D spiral dataset (not linearly separable!).

    Returns:
        X: shape (2, 2*n_points) - 2D points
        Y: shape (1, 2*n_points) - labels (0 or 1)
    """
    n = n_points
    X = np.zeros((2, 2 * n))
    Y = np.zeros((1, 2 * n))

    for i in range(2):
        # Generate spiral
        theta = np.linspace(0, 4 * np.pi, n) + i * np.pi
        r = np.linspace(0, 1, n)

        # Add points
        X[0, i * n:(i + 1) * n] = r * np.cos(theta) + np.random.randn(n) * noise
        X[1, i * n:(i + 1) * n] = r * np.sin(theta) + np.random.randn(n) * noise
        Y[0, i * n:(i + 1) * n] = i

    return X, Y


# Create spiral dataset
X_spiral, Y_spiral = create_spiral_dataset(n_points=50, noise=0.1)

print(f"\nCreated spiral dataset:")
print(f"  X shape: {X_spiral.shape}  (2 features, 100 examples)")
print(f"  Y shape: {Y_spiral.shape}  (labels: 0 or 1)")
print(f"  Class 0: {np.sum(Y_spiral == 0)} examples")
print(f"  Class 1: {np.sum(Y_spiral == 1)} examples")


class BinaryClassifier:
    """
    Binary classifier: 2 inputs → hidden layers → 1 output

    Can have 0, 1, or 2 hidden layers to compare performance.
    """

    def __init__(self, hidden_layers: List[int]):
        """
        Args:
            hidden_layers: List of hidden layer sizes
                          [] for no hidden layers (linear)
                          [4] for one hidden layer with 4 neurons
                          [8, 4] for two hidden layers
        """
        # Build layer sizes: [2, *hidden_layers, 1]
        self.layer_sizes = [2] + hidden_layers + [1]
        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            input_dim = self.layer_sizes[i]
            output_dim = self.layer_sizes[i + 1]

            W = np.random.randn(output_dim, input_dim) * 0.3
            b = np.zeros((output_dim, 1))

            self.weights.append(W)
            self.biases.append(b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, x):
        """Forward propagation"""
        a = x
        for i in range(len(self.weights)):
            z = self.weights[i] @ a + self.biases[i]

            # Use sigmoid for all layers
            a = self.sigmoid(z)

        return a

    def train(self, X, Y, epochs=1000, learning_rate=0.5, verbose=False):
        """Train network (simplified backprop)"""
        losses = []

        for epoch in range(epochs):
            # Forward
            activations = [X]
            a = X
            for W, b in zip(self.weights, self.biases):
                z = W @ a + b
                a = self.sigmoid(z)
                activations.append(a)

            # Loss
            predictions = activations[-1]
            loss = np.mean((predictions - Y) ** 2)
            losses.append(loss)

            # Backward (simplified)
            m = X.shape[1]
            dz = activations[-1] - Y

            for i in reversed(range(len(self.weights))):
                dW = (1 / m) * (dz @ activations[i].T)
                db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

                if i > 0:
                    dz = (self.weights[i].T @ dz) * activations[i] * (1 - activations[i])

                self.weights[i] -= learning_rate * dW
                self.biases[i] -= learning_rate * db

            if verbose and epoch % 200 == 0:
                print(f"  Epoch {epoch}: Loss = {loss:.4f}")

        return losses


# Train networks with different depths
print("\n--- Training Networks with Different Depths ---")

networks = {
    "0 Hidden Layers (Linear)": BinaryClassifier([]),
    "1 Hidden Layer (8 neurons)": BinaryClassifier([8]),
    "2 Hidden Layers (16, 8)": BinaryClassifier([16, 8]),
}

all_losses = {}
for name, net in networks.items():
    print(f"\nTraining: {name}")
    losses = net.train(X_spiral, Y_spiral, epochs=2000, learning_rate=1.0, verbose=True)
    all_losses[name] = losses

# Plot learning curves
plt.figure(figsize=(12, 4))

# Plot 1: Learning curves
plt.subplot(1, 2, 1)
for name, losses in all_losses.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Learning Curves - Different Network Depths')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Final decision boundaries
plt.subplot(1, 2, 2)

# Create grid
x_min, x_max = X_spiral[0, :].min() - 0.5, X_spiral[0, :].max() + 0.5
y_min, y_max = X_spiral[1, :].min() - 0.5, X_spiral[1, :].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Use the 2-layer network for visualization
net = networks["2 Hidden Layers (16, 8)"]
grid_points = np.c_[xx.ravel(), yy.ravel()].T
Z = net.forward(grid_points)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
plt.colorbar(label='Prediction')

# Plot data points
for i in range(2):
    mask = Y_spiral[0, :] == i
    plt.scatter(X_spiral[0, mask], X_spiral[1, mask],
                c='red' if i == 0 else 'blue',
                edgecolors='black',
                linewidth=0.5,
                label=f'Class {i}')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary - 2 Hidden Layers')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('example_03_decision_boundaries.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: example_03_decision_boundaries.png")


# ============================================================================
# EXAMPLE 5: Connection to GPT (Feed-Forward Networks)
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: GPT Feed-Forward Network")
print("=" * 70)


class GPTFeedForward:
    """
    The feed-forward network used in EVERY GPT transformer layer!

    GPT-2 Small: 768 → 3072 → 768
    GPT-3: 12288 → 49152 → 12288
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: Model dimension (e.g., 768 for GPT-2)
            d_ff: Feed-forward dimension (typically 4 * d_model)
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Layer 1: Expand (d_model → d_ff)
        self.W1 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros((d_ff, 1))

        # Layer 2: Compress (d_ff → d_model)
        self.W2 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros((d_model, 1))

        print(f"\nGPT Feed-Forward Network: {d_model} → {d_ff} → {d_model}")
        print(f"  Layer 1 params: {self.W1.size + self.b1.size:,}")
        print(f"  Layer 2 params: {self.W2.size + self.b2.size:,}")
        print(f"  Total params: {self.W1.size + self.b1.size + self.W2.size + self.b2.size:,}")

    def gelu(self, z):
        """
        GELU activation (used in GPT, not ReLU!)

        GELU(x) = x * Φ(x) where Φ is Gaussian CDF
        Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
        """
        return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))

    def forward(self, x):
        """
        Forward pass through feed-forward network.

        Args:
            x: Input, shape (d_model, batch_size)

        Returns:
            output: shape (d_model, batch_size)
        """
        # Layer 1: Expand + GELU
        z1 = self.W1 @ x + self.b1
        a1 = self.gelu(z1)

        # Layer 2: Compress (no activation!)
        output = self.W2 @ a1 + self.b2

        return output


# Create GPT-2 style feed-forward network
gpt2_ff = GPTFeedForward(d_model=768, d_ff=3072)

# Simulate processing 5 tokens
batch_size = 5
x = np.random.randn(768, batch_size)  # 5 token embeddings

print(f"\nInput shape: {x.shape}  (768-dim embeddings for {batch_size} tokens)")

output = gpt2_ff.forward(x)
print(f"Output shape: {output.shape}  (same as input!)")

print("\n--- Comparison to Full GPT-2 ---")
print(f"GPT-2 has 12 transformer layers")
print(f"Each layer has:")
print(f"  - Multi-head attention (~2.4M params)")
print(f"  - Feed-forward network (~{(768*3072 + 3072*768):,} params)")
print(f"Feed-forward networks = ~50% of GPT-2's parameters!")


# ============================================================================
# EXAMPLE 6: Parameter Counting
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Counting Parameters in Networks")
print("=" * 70)


def count_parameters(layer_sizes: List[int]) -> Dict[str, int]:
    """
    Count parameters in a network.

    Args:
        layer_sizes: [input, hidden1, hidden2, ..., output]

    Returns:
        Dict with parameter counts per layer and total
    """
    params = {}
    total = 0

    for i in range(len(layer_sizes) - 1):
        layer_name = f"Layer {i+1}"
        input_dim = layer_sizes[i]
        output_dim = layer_sizes[i + 1]

        # Weights: output_dim × input_dim
        # Biases: output_dim
        layer_params = output_dim * input_dim + output_dim

        params[layer_name] = {
            'weights': output_dim * input_dim,
            'biases': output_dim,
            'total': layer_params
        }

        total += layer_params

    params['total'] = total
    return params


# Example networks
networks_to_count = {
    "MNIST Simple": [784, 128, 10],
    "MNIST Deep": [784, 128, 64, 10],
    "GPT-2 FF (single layer)": [768, 3072, 768],
    "Very Deep": [100, 200, 200, 200, 50],
}

print("\n--- Parameter Counts ---")
for name, sizes in networks_to_count.items():
    params = count_parameters(sizes)
    print(f"\n{name}: {' → '.join(map(str, sizes))}")

    for layer_name, counts in params.items():
        if layer_name != 'total':
            print(f"  {layer_name}: {counts['total']:,} params "
                  f"(W: {counts['weights']:,}, b: {counts['biases']:,})")

    print(f"  TOTAL: {params['total']:,} parameters")


# ============================================================================
# SUMMARY AND KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Key Takeaways from Multi-Layer Networks")
print("=" * 70)

print("""
✅ What You Learned:

1. Multi-layer networks = stacked transformations
   - Each layer: Linear → Activation
   - Output of layer N = Input to layer N+1

2. Depth enables complexity
   - Single layer: Only linear separability
   - Multiple layers: Can learn XOR, spirals, complex patterns
   - Many layers (deep): Can learn faces, language, reasoning

3. Shape management is critical
   - (A @ B) requires A.shape[1] == B.shape[0]
   - Always print shapes when debugging!
   - Shape mismatches = #1 bug source

4. GPT uses multi-layer networks everywhere
   - Feed-forward networks in every transformer layer
   - 768 → 3072 → 768 (GPT-2)
   - ~50% of model parameters!

5. More parameters ≠ always better
   - More params = more capacity but also overfitting risk
   - Balance depth with data availability

📊 Results from Examples:
   - XOR: Solved with 2-layer network (impossible with 1 layer!)
   - Spirals: Deep networks create better decision boundaries
   - GPT-2 FF: 4.7M parameters per transformer layer

🔜 Next Lesson (Backpropagation):
   You now know HOW data flows forward (forward propagation).
   Next: HOW the network actually learns (backpropagation)!

   Forward:  Input → Layer 1 → Layer 2 → Output
   Backward: Input ← Layer 1 ← Layer 2 ← Error

   Both are needed for training!
""")

print("\nFiles created:")
print("  ✓ example_03_xor_learning_curve.png")
print("  ✓ example_03_decision_boundaries.png")

print("\n" + "=" * 70)
print("Great job! You now understand multi-layer neural networks!")
print("Next: Run exercise_03_networks.py to practice")
print("=" * 70)

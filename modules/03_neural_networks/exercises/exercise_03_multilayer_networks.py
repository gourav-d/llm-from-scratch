"""
Lesson 3.3: Multi-Layer Neural Networks - Practice Exercises

Complete these exercises to reinforce your understanding of:
- Building multi-layer networks
- Forward propagation
- Shape management
- Solving non-linear problems

For each exercise:
1. Try to solve it yourself first
2. If stuck, read the hints
3. Check your solution against the provided solution
4. Experiment with variations!
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# EXERCISE 1: Build a 3-Layer Network
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Build a 3-Layer Network")
print("=" * 70)

print("""
Task: Create a network with this architecture:
- Input: 10 features
- Hidden layer 1: 20 neurons, ReLU activation
- Hidden layer 2: 15 neurons, ReLU activation
- Output: 5 classes, Softmax activation

Requirements:
1. Initialize weights with small random values
2. Initialize biases as zeros
3. Implement forward() method
4. Test with random input
""")


# YOUR CODE HERE
class Exercise1Network:
    def __init__(self):
        """TODO: Initialize weights and biases for all 3 layers"""
        pass

    def relu(self, z):
        """TODO: Implement ReLU activation"""
        pass

    def softmax(self, z):
        """TODO: Implement Softmax activation"""
        pass

    def forward(self, x):
        """TODO: Implement forward propagation through all 3 layers"""
        pass


# Test your network
# x = np.random.randn(10, 1)
# net = Exercise1Network()
# output = net.forward(x)
# print(f"Output shape: {output.shape}")  # Should be (5, 1)
# print(f"Output sums to 1.0: {np.sum(output):.6f}")

print("\n💡 HINT 1: W1 should have shape (20, 10) - 20 neurons, 10 inputs each")
print("💡 HINT 2: Use np.random.randn() * 0.01 for weight initialization")
print("💡 HINT 3: Chain layers: x → layer1 → layer2 → layer3")

# Scroll down for solution...
print("\n" + "-" * 70)


# ============================================================================
# SOLUTION 1
# ============================================================================

class Exercise1NetworkSolution:
    def __init__(self):
        # Layer 1: 10 → 20
        self.W1 = np.random.randn(20, 10) * 0.01
        self.b1 = np.zeros((20, 1))

        # Layer 2: 20 → 15
        self.W2 = np.random.randn(15, 20) * 0.01
        self.b2 = np.zeros((15, 1))

        # Layer 3: 15 → 5
        self.W3 = np.random.randn(5, 15) * 0.01
        self.b3 = np.zeros((5, 1))

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, x):
        # Layer 1: Input → Hidden1
        z1 = self.W1 @ x + self.b1
        a1 = self.relu(z1)

        # Layer 2: Hidden1 → Hidden2
        z2 = self.W2 @ a1 + self.b2
        a2 = self.relu(z2)

        # Layer 3: Hidden2 → Output
        z3 = self.W3 @ a2 + self.b3
        output = self.softmax(z3)

        return output


# Test solution
print("\n--- Testing Solution ---")
x = np.random.randn(10, 1)
net_solution = Exercise1NetworkSolution()
output = net_solution.forward(x)
print(f"✓ Output shape: {output.shape}  (expected: (5, 1))")
print(f"✓ Output sums to 1.0: {np.sum(output):.6f}")
print(f"✓ Predicted class: {np.argmax(output)}")


# ============================================================================
# EXERCISE 2: Fix Shape Mismatches
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Debug Shape Errors")
print("=" * 70)

print("""
Task: The code below has shape errors. Fix them!

Requirements:
1. Identify all shape mismatches
2. Fix biases to have correct shape
3. Fix input to have correct shape
4. Code should run without errors
""")

# BROKEN CODE - FIX THE SHAPE ERRORS!
def broken_network():
    """This code has 2 shape errors - find and fix them!"""

    # Network: 100 → 64 → 32 → 10
    W1 = np.random.randn(64, 100)
    b1 = np.zeros((64, 1))

    W2 = np.random.randn(32, 64)
    b2 = np.zeros((32,))  # ← ERROR? Should this be (32, 1)?

    W3 = np.random.randn(10, 32)
    b3 = np.zeros((10, 1))

    # Input
    x = np.random.randn(100,)  # ← ERROR? Should this be (100, 1)?

    # Forward pass
    z1 = W1 @ x + b1
    a1 = np.maximum(0, z1)

    z2 = W2 @ a1 + b2
    a2 = np.maximum(0, z2)

    z3 = W3 @ a2 + b3

    return z3


# Uncomment to test (will error before fix):
# try:
#     output = broken_network()
#     print(f"Output shape: {output.shape}")
# except ValueError as e:
#     print(f"Error: {e}")

print("\n💡 HINT 1: Biases should be column vectors (n, 1), not 1D arrays (n,)")
print("💡 HINT 2: Input should be column vector (n, 1) for matrix multiplication")
print("💡 HINT 3: Check the error message - it tells you which shapes don't match!")

# Solution below...
print("\n" + "-" * 70)


# SOLUTION 2
def fixed_network():
    """Fixed version with correct shapes"""

    W1 = np.random.randn(64, 100)
    b1 = np.zeros((64, 1))

    W2 = np.random.randn(32, 64)
    b2 = np.zeros((32, 1))  # ✓ Fixed: (32, 1) instead of (32,)

    W3 = np.random.randn(10, 32)
    b3 = np.zeros((10, 1))

    x = np.random.randn(100, 1)  # ✓ Fixed: (100, 1) instead of (100,)

    z1 = W1 @ x + b1
    a1 = np.maximum(0, z1)

    z2 = W2 @ a1 + b2
    a2 = np.maximum(0, z2)

    z3 = W3 @ a2 + b3

    return z3


print("\n--- Testing Fixed Network ---")
output = fixed_network()
print(f"✓ Output shape: {output.shape}  (expected: (10, 1))")


# ============================================================================
# EXERCISE 3: AND, OR, NAND Gates
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Learn Logic Gates")
print("=" * 70)

print("""
Task: Modify the XOR network to learn different logic gates.

Gates to implement:
1. AND: Output 1 only when both inputs are 1
2. OR: Output 1 when at least one input is 1
3. NAND: Output 0 only when both inputs are 1 (opposite of AND)

Requirements:
- Use same network architecture as XOR (2 → 4 → 1)
- Only change the labels (Y), not the network!
- Train for 5000 epochs
- Achieve < 0.01 loss
""")

# Truth tables
print("\nTruth Tables:")
print("\nAND:")
print("0 AND 0 = 0")
print("0 AND 1 = 0")
print("1 AND 0 = 0")
print("1 AND 1 = 1")

print("\nOR:")
print("0 OR 0 = 0")
print("0 OR 1 = 1")
print("1 OR 0 = 1")
print("1 OR 1 = 1")

print("\nNAND:")
print("0 NAND 0 = 1")
print("0 NAND 1 = 1")
print("1 NAND 0 = 1")
print("1 NAND 1 = 0")


# YOUR CODE HERE
# TODO: Create datasets for AND, OR, NAND
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T

# Y_and = ???
# Y_or = ???
# Y_nand = ???

print("\n💡 HINT: Only change the Y values!")
print("💡 Example: Y_and = np.array([[0, 0, 0, 1]]) for AND gate")

# Solution below...
print("\n" + "-" * 70)


# SOLUTION 3

# Reuse XOR network class
class LogicGateNetwork:
    def __init__(self):
        self.W1 = np.random.randn(4, 2) * 0.5
        self.b1 = np.zeros((4, 1))
        self.W2 = np.random.randn(1, 4) * 0.5
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, x):
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def train(self, X, Y, epochs=5000, learning_rate=1.0):
        for epoch in range(epochs):
            predictions = self.forward(X)
            m = X.shape[1]

            # Backprop
            dz2 = (predictions - Y) * predictions * (1 - predictions)
            dW2 = (1 / m) * (dz2 @ self.a1.T)
            db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

            dz1 = (self.W2.T @ dz2) * self.a1 * (1 - self.a1)
            dW1 = (1 / m) * (dz1 @ X.T)
            db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

        return predictions


# Inputs (same for all gates)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T

# Labels (different for each gate)
Y_and = np.array([[0, 0, 0, 1]])   # Only 1,1 → 1
Y_or = np.array([[0, 1, 1, 1]])    # Any 1 → 1
Y_nand = np.array([[1, 1, 1, 0]])  # Opposite of AND

gates = {
    'AND': Y_and,
    'OR': Y_or,
    'NAND': Y_nand
}

print("\n--- Training Logic Gates ---")
for gate_name, Y in gates.items():
    print(f"\n{gate_name} Gate:")
    net = LogicGateNetwork()
    predictions = net.train(X, Y, epochs=5000)

    print("Input  | Expected | Predicted | Rounded")
    print("-------|----------|-----------|--------")
    for i in range(4):
        x1, x2 = X[:, i]
        expected = Y[0, i]
        predicted = predictions[0, i]
        rounded = 1 if predicted > 0.5 else 0
        correct = "✓" if rounded == expected else "✗"
        print(f"[{int(x1)}, {int(x2)}] |    {int(expected)}     |   {predicted:.4f}  |   {rounded} {correct}")


# ============================================================================
# EXERCISE 4: Experiment with Depth
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: How Does Depth Affect Learning?")
print("=" * 70)

print("""
Task: Compare XOR learning with different network depths.

Networks to compare:
1. Shallow: 2 → 4 → 1
2. Medium: 2 → 8 → 4 → 1
3. Deep: 2 → 16 → 8 → 4 → 1

Questions to answer:
- Which network learns fastest?
- Which achieves lowest final loss?
- Is deeper always better?
""")

# YOUR CODE HERE
# TODO: Create 3 networks with different depths
# TODO: Train each for same number of epochs
# TODO: Plot learning curves to compare

print("\n💡 HINT: Deeper networks have more parameters → might need more epochs")
print("💡 HINT: Plot all 3 learning curves on same plot to compare")

# Solution below...
print("\n" + "-" * 70)


# SOLUTION 4
class FlexibleNetwork:
    """Network with configurable depth"""

    def __init__(self, layer_sizes):
        """
        Args:
            layer_sizes: List of layer sizes, e.g., [2, 8, 4, 1]
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * 0.5
            b = np.zeros((layer_sizes[i + 1], 1))
            self.weights.append(W)
            self.biases.append(b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, x):
        self.activations = [x]
        a = x

        for W, b in zip(self.weights, self.biases):
            z = W @ a + b
            a = self.sigmoid(z)
            self.activations.append(a)

        return a

    def train(self, X, Y, epochs=5000, learning_rate=0.5):
        losses = []

        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = np.mean((predictions - Y) ** 2)
            losses.append(loss)

            # Simplified backprop
            m = X.shape[1]
            deltas = [predictions - Y]

            # Backward through layers
            for i in reversed(range(len(self.weights))):
                delta = deltas[0]
                dW = (1 / m) * (delta @ self.activations[i].T)
                db = (1 / m) * np.sum(delta, axis=1, keepdims=True)

                if i > 0:
                    delta = (self.weights[i].T @ delta) * self.activations[i] * (1 - self.activations[i])
                    deltas.insert(0, delta)

                self.weights[i] -= learning_rate * dW
                self.biases[i] -= learning_rate * db

        return losses


# XOR data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y_xor = np.array([[0, 1, 1, 0]])

# Create networks with different depths
architectures = {
    'Shallow (2→4→1)': [2, 4, 1],
    'Medium (2→8→4→1)': [2, 8, 4, 1],
    'Deep (2→16→8→4→1)': [2, 16, 8, 4, 1]
}

print("\n--- Training Networks with Different Depths ---")
all_losses = {}

for name, architecture in architectures.items():
    print(f"\nTraining: {name}")
    net = FlexibleNetwork(architecture)
    losses = net.train(X_xor, Y_xor, epochs=3000, learning_rate=1.0)
    all_losses[name] = losses

    final_loss = losses[-1]
    print(f"  Final loss: {final_loss:.6f}")

# Plot comparison
plt.figure(figsize=(12, 5))

# Plot 1: Full learning curves
plt.subplot(1, 2, 1)
for name, losses in all_losses.items():
    plt.plot(losses, label=name, alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('XOR Learning Curves - Different Depths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 2: Last 1000 epochs (zoomed in)
plt.subplot(1, 2, 2)
for name, losses in all_losses.items():
    plt.plot(losses[-1000:], label=name, alpha=0.8)
plt.xlabel('Epoch (last 1000)')
plt.ylabel('Loss (MSE)')
plt.title('Final Convergence (Zoomed)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exercise_03_depth_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: exercise_03_depth_comparison.png")

print("\n--- Analysis ---")
print("Observations:")
print("  - Shallow network: Faster initially, but may plateau higher")
print("  - Deep network: More parameters, might need more epochs")
print("  - Medium network: Often the sweet spot for simple problems!")
print("\nKey insight: Deeper is NOT always better for simple problems!")


# ============================================================================
# EXERCISE 5: Build a Mini MNIST Network
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Build Your First Image Classifier Network")
print("=" * 70)

print("""
Task: Design a network for MNIST digit classification.

Specifications:
- Input: 28×28 grayscale image = 784 pixels
- Output: 10 classes (digits 0-9)
- Hidden layers: Your choice! But total params should be ~100k-200k

Requirements:
1. Calculate number of parameters for your design
2. Justify your architecture choice
3. Implement the forward pass
4. Test with random "image"
""")

# YOUR CODE HERE
# TODO: Design network architecture
# TODO: Implement the network
# TODO: Count parameters
# TODO: Test forward pass

print("\n💡 HINT 1: Common patterns: 784→256→128→10 or 784→512→256→10")
print("💡 HINT 2: Parameters for one layer: (input_size × output_size) + output_size")
print("💡 HINT 3: Decreasing layer sizes (funnel) works well for classification")

# Solution below...
print("\n" + "-" * 70)


# SOLUTION 5
class MiniMNISTNetwork:
    """
    Network for MNIST classification.

    Architecture: 784 → 256 → 128 → 10
    """

    def __init__(self):
        # Layer 1: 784 → 256
        self.W1 = np.random.randn(256, 784) * np.sqrt(2.0 / 784)  # He initialization
        self.b1 = np.zeros((256, 1))

        # Layer 2: 256 → 128
        self.W2 = np.random.randn(128, 256) * np.sqrt(2.0 / 256)
        self.b2 = np.zeros((128, 1))

        # Layer 3: 128 → 10
        self.W3 = np.random.randn(10, 128) * np.sqrt(2.0 / 128)
        self.b3 = np.zeros((10, 1))

        # Count parameters
        params_l1 = 784 * 256 + 256
        params_l2 = 256 * 128 + 128
        params_l3 = 128 * 10 + 10
        total_params = params_l1 + params_l2 + params_l3

        print("\n--- Network Architecture ---")
        print(f"Input:   784 (28×28 pixels)")
        print(f"Hidden1: 256 neurons, ReLU")
        print(f"Hidden2: 128 neurons, ReLU")
        print(f"Output:  10 neurons, Softmax")
        print(f"\n--- Parameter Count ---")
        print(f"Layer 1: {params_l1:,} params")
        print(f"Layer 2: {params_l2:,} params")
        print(f"Layer 3: {params_l3:,} params")
        print(f"TOTAL:   {total_params:,} params")

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, x):
        """Forward propagation"""
        # Layer 1
        z1 = self.W1 @ x + self.b1
        a1 = self.relu(z1)

        # Layer 2
        z2 = self.W2 @ a1 + self.b2
        a2 = self.relu(z2)

        # Layer 3
        z3 = self.W3 @ a2 + self.b3
        y = self.softmax(z3)

        return y


# Test the network
print("\n--- Testing MNIST Network ---")
mnist_net = MiniMNISTNetwork()

# Simulate flattened 28×28 image
fake_image = np.random.randn(784, 1)
predictions = mnist_net.forward(fake_image)

print(f"\nInput shape: {fake_image.shape}")
print(f"Output shape: {predictions.shape}")
print(f"\nPredictions (probabilities for digits 0-9):")
for digit in range(10):
    print(f"  Digit {digit}: {predictions[digit, 0]:.4f}")

print(f"\nPredicted digit: {np.argmax(predictions)}")
print(f"Confidence: {np.max(predictions):.4f}")
print(f"Sum of probabilities: {np.sum(predictions):.6f}")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE SUMMARY")
print("=" * 70)

print("""
✅ What You Practiced:

1. Building multi-layer networks from scratch
   - Proper weight initialization
   - Correct shape management
   - Multiple activation functions

2. Solving real problems
   - Logic gates (AND, OR, NAND, XOR)
   - Non-linear classification (spirals)
   - Image classification (MNIST)

3. Debugging shape errors
   - Understanding matrix multiplication requirements
   - Fixing common shape mismatches

4. Experimenting with architecture
   - Different depths
   - Different layer sizes
   - Parameter counting

5. Connecting to real-world applications
   - GPT feed-forward networks
   - MNIST digit classification

🎯 Key Insights:

- Depth enables complexity, but more isn't always better
- Shape management is the #1 debugging skill
- Multi-layer networks can solve XOR (single layer cannot!)
- Parameter count grows quickly: 784→256 = 200k+ params!
- Same concepts power GPT, BERT, and all modern AI

🔜 Next Steps:

You now understand forward propagation (making predictions).

Next lesson: Backpropagation (learning from mistakes)
- How networks actually learn
- Computing gradients through all layers
- The chain rule in action
- Why it's called "backward" propagation

Ready to learn how neural networks learn?
→ Read 04_backpropagation.md next!
""")

print("\n" + "=" * 70)
print("Great work completing these exercises!")
print("=" * 70)

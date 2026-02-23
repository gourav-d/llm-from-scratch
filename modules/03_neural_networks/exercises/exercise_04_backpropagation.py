"""
Lesson 3.4: Backpropagation - Practice Exercises

Complete these exercises to master backpropagation:
- Manual gradient calculation
- Implementing backprop for different networks
- Numerical gradient checking
- Debugging gradient issues
- Understanding convergence

For each exercise:
1. Try solving it yourself first
2. Use hints if stuck
3. Check solution
4. Experiment with variations!
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# EXERCISE 1: Manual Gradient Calculation
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Calculate Gradients by Hand")
print("=" * 70)

print("""
Task: Calculate gradients manually for this simple network:

Network: x → [w] → z = w × x → y = sigmoid(z)
Loss: L = (y - t)²

Given:
- x = 2.0
- w = 0.3
- t = 1.0 (true label)

Calculate:
1. Forward pass (z, y, L)
2. ∂L/∂y
3. ∂L/∂z (use chain rule!)
4. ∂L/∂w
5. New weight after update (learning_rate = 0.1)
""")


# YOUR CODE HERE
def manual_gradient_exercise():
    """
    TODO: Calculate each value step by step
    """
    x = 2.0
    w = 0.3
    t = 1.0
    learning_rate = 0.1

    # TODO: Forward pass
    # z = ?
    # y = ?
    # L = ?

    # TODO: Backward pass
    # dL_dy = ?
    # dy_dz = ?  (sigmoid derivative)
    # dL_dz = ?  (chain rule)
    # dz_dw = ?
    # dL_dw = ?  (chain rule)

    # TODO: Weight update
    # w_new = ?

    pass


# manual_gradient_exercise()

print("\n💡 HINT 1: sigmoid(z) = 1 / (1 + exp(-z))")
print("💡 HINT 2: sigmoid'(z) = sigmoid(z) × (1 - sigmoid(z))")
print("💡 HINT 3: Chain rule: ∂L/∂z = ∂L/∂y × ∂y/∂z")

print("\n" + "-" * 70)


# SOLUTION 1
def manual_gradient_solution():
    """Complete solution with explanations"""
    x = 2.0
    w = 0.3
    t = 1.0
    learning_rate = 0.1

    print("\n--- Solution ---")

    # Forward pass
    z = w * x
    print(f"1. z = w × x = {w} × {x} = {z}")

    y = 1 / (1 + np.exp(-z))
    print(f"2. y = sigmoid(z) = sigmoid({z}) = {y:.6f}")

    L = (y - t) ** 2
    print(f"3. L = (y - t)² = ({y:.6f} - {t})² = {L:.6f}")

    # Backward pass
    dL_dy = 2 * (y - t)
    print(f"\n4. ∂L/∂y = 2(y - t) = 2({y:.6f} - {t}) = {dL_dy:.6f}")

    dy_dz = y * (1 - y)  # Sigmoid derivative
    print(f"5. ∂y/∂z = y(1-y) = {y:.6f} × {1-y:.6f} = {dy_dz:.6f}")

    dL_dz = dL_dy * dy_dz  # Chain rule
    print(f"6. ∂L/∂z = ∂L/∂y × ∂y/∂z = {dL_dy:.6f} × {dy_dz:.6f} = {dL_dz:.6f}")

    dz_dw = x
    print(f"7. ∂z/∂w = x = {x}")

    dL_dw = dL_dz * dz_dw  # Chain rule
    print(f"8. ∂L/∂w = ∂L/∂z × ∂z/∂w = {dL_dz:.6f} × {x} = {dL_dw:.6f}")

    # Weight update
    w_new = w - learning_rate * dL_dw
    print(f"\n9. w_new = w - α × ∂L/∂w = {w} - {learning_rate} × {dL_dw:.6f} = {w_new:.6f}")

    # Verify
    z_new = w_new * x
    y_new = 1 / (1 + np.exp(-z_new))
    L_new = (y_new - t) ** 2

    print(f"\n10. Verification:")
    print(f"    Old: y = {y:.6f}, L = {L:.6f}")
    print(f"    New: y = {y_new:.6f}, L = {L_new:.6f}")
    print(f"    Improvement: {L - L_new:.6f} ✓")


print("\n--- Running Solution ---")
manual_gradient_solution()


# ============================================================================
# EXERCISE 2: Implement Backprop for 3-Layer Network
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Implement Backprop for 3-Layer Network")
print("=" * 70)

print("""
Task: Complete the backward() method for a 3-layer network.

Architecture: x → hidden1 → hidden2 → output

Requirements:
1. Compute gradients for all 6 parameters (W1, b1, W2, b2, W3, b3)
2. Use chain rule to propagate gradients backwards
3. Return dictionary of gradients
4. Verify with numerical gradient checking
""")


# YOUR CODE HERE
class ThreeLayerNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.W1 = np.random.randn(hidden1_size, input_size) * 0.5
        self.b1 = np.zeros((hidden1_size, 1))

        self.W2 = np.random.randn(hidden2_size, hidden1_size) * 0.5
        self.b2 = np.zeros((hidden2_size, 1))

        self.W3 = np.random.randn(output_size, hidden2_size) * 0.5
        self.b3 = np.zeros((output_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, x):
        # Layer 1
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Layer 2
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.sigmoid(self.z2)

        # Layer 3
        self.z3 = self.W3 @ self.a2 + self.b3
        self.y = self.sigmoid(self.z3)

        self.x = x
        return self.y

    def backward(self, t):
        """
        TODO: Implement backpropagation for 3 layers!

        Hint: Start from output, work backwards
        1. Compute ∂L/∂z3 (output layer)
        2. Compute ∂L/∂W3, ∂L/∂b3
        3. Propagate to layer 2: ∂L/∂z2
        4. Compute ∂L/∂W2, ∂L/∂b2
        5. Propagate to layer 1: ∂L/∂z1
        6. Compute ∂L/∂W1, ∂L/∂b1
        """
        m = self.x.shape[1]

        # TODO: Implement backward pass
        pass

    def compute_loss(self, y, t):
        return np.mean((y - t) ** 2)


print("\n💡 HINT 1: Pattern for each layer:")
print("   dL_dz = (W_next^T @ dL_dz_next) ⊙ σ'(z)")
print("💡 HINT 2: Output layer is special:")
print("   dL_dz3 = (y - t) ⊙ σ'(z3)")
print("💡 HINT 3: Weight gradients:")
print("   dL_dW = (1/m) × dL_dz @ a_prev^T")

print("\n" + "-" * 70)


# SOLUTION 2
class ThreeLayerNetworkSolution:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.W1 = np.random.randn(hidden1_size, input_size) * 0.5
        self.b1 = np.zeros((hidden1_size, 1))

        self.W2 = np.random.randn(hidden2_size, hidden1_size) * 0.5
        self.b2 = np.zeros((hidden2_size, 1))

        self.W3 = np.random.randn(output_size, hidden2_size) * 0.5
        self.b3 = np.zeros((output_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, x):
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.sigmoid(self.z2)

        self.z3 = self.W3 @ self.a2 + self.b3
        self.y = self.sigmoid(self.z3)

        self.x = x
        return self.y

    def backward(self, t):
        """Complete backpropagation for 3 layers"""
        m = self.x.shape[1]

        # ===== Layer 3 (Output) =====
        # Gradient at output layer
        dL_dz3 = (self.y - t) * self.sigmoid_derivative(self.y)

        # Gradients for W3, b3
        dL_dW3 = (1 / m) * (dL_dz3 @ self.a2.T)
        dL_db3 = (1 / m) * np.sum(dL_dz3, axis=1, keepdims=True)

        # ===== Layer 2 (Hidden 2) =====
        # Propagate gradient to layer 2
        dL_da2 = self.W3.T @ dL_dz3
        dL_dz2 = dL_da2 * self.sigmoid_derivative(self.a2)

        # Gradients for W2, b2
        dL_dW2 = (1 / m) * (dL_dz2 @ self.a1.T)
        dL_db2 = (1 / m) * np.sum(dL_dz2, axis=1, keepdims=True)

        # ===== Layer 1 (Hidden 1) =====
        # Propagate gradient to layer 1
        dL_da1 = self.W2.T @ dL_dz2
        dL_dz1 = dL_da1 * self.sigmoid_derivative(self.a1)

        # Gradients for W1, b1
        dL_dW1 = (1 / m) * (dL_dz1 @ self.x.T)
        dL_db1 = (1 / m) * np.sum(dL_dz1, axis=1, keepdims=True)

        return {
            'dW1': dL_dW1, 'db1': dL_db1,
            'dW2': dL_dW2, 'db2': dL_db2,
            'dW3': dL_dW3, 'db3': dL_db3
        }

    def compute_loss(self, y, t):
        return np.mean((y - t) ** 2)

    def update_weights(self, gradients, lr):
        self.W1 -= lr * gradients['dW1']
        self.b1 -= lr * gradients['db1']
        self.W2 -= lr * gradients['dW2']
        self.b2 -= lr * gradients['db2']
        self.W3 -= lr * gradients['dW3']
        self.b3 -= lr * gradients['db3']


print("\n--- Testing 3-Layer Network Solution ---")
net3 = ThreeLayerNetworkSolution(
    input_size=2,
    hidden1_size=4,
    hidden2_size=3,
    output_size=1
)

# Test on XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y_xor = np.array([[0, 1, 1, 0]])

# Single step
y = net3.forward(X_xor)
loss = net3.compute_loss(y, Y_xor)
grads = net3.backward(Y_xor)

print(f"Initial loss: {loss:.6f}")
print(f"Gradients computed: {list(grads.keys())}")
print(f"  dW1 shape: {grads['dW1'].shape}")
print(f"  dW2 shape: {grads['dW2'].shape}")
print(f"  dW3 shape: {grads['dW3'].shape}")

# Train for a bit
for epoch in range(1000):
    y = net3.forward(X_xor)
    loss = net3.compute_loss(y, Y_xor)
    grads = net3.backward(Y_xor)
    net3.update_weights(grads, lr=1.0)

print(f"Final loss after 1000 epochs: {loss:.6f}")
print("✓ 3-layer backprop working!")


# ============================================================================
# EXERCISE 3: Numerical Gradient Checking
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Implement Numerical Gradient Checking")
print("=" * 70)

print("""
Task: Implement a function to verify backprop using numerical gradients.

Method: Finite differences
  ∂L/∂w ≈ [L(w + ε) - L(w - ε)] / (2ε)

Requirements:
1. Implement numerical_gradient() function
2. Compare with analytical gradient from backprop
3. Difference should be < 1e-7 if backprop is correct
""")


# YOUR CODE HERE
def numerical_gradient(network, x, t, param_name, i, j, epsilon=1e-5):
    """
    TODO: Compute numerical gradient for one parameter

    Args:
        network: Neural network
        x, t: Input and target
        param_name: 'W1', 'b1', 'W2', or 'b2'
        i, j: Index of parameter
        epsilon: Small step size

    Returns:
        Numerical gradient approximation

    Hint:
    1. Get parameter value
    2. Compute loss with param + epsilon
    3. Compute loss with param - epsilon
    4. Gradient ≈ (loss_plus - loss_minus) / (2 * epsilon)
    5. Restore original parameter value
    """
    pass


print("\n💡 HINT 1: Use getattr(network, param_name) to get parameter")
print("💡 HINT 2: Save original value before modifying!")
print("💡 HINT 3: Restore original value after computing gradient")

print("\n" + "-" * 70)


# SOLUTION 3
def numerical_gradient_solution(network, x, t, param_name, i, j, epsilon=1e-5):
    """Complete implementation of numerical gradient"""
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


print("\n--- Testing Numerical Gradient Checking ---")

# Create small network
test_net = ThreeLayerNetworkSolution(
    input_size=2,
    hidden1_size=3,
    hidden2_size=2,
    output_size=1
)

# Single example
x_test = np.array([[0.5], [0.8]])
t_test = np.array([[1.0]])

# Forward + backward
y_test = test_net.forward(x_test)
analytical_grads = test_net.backward(t_test)

print("\nGradient Checking Results:")
print(f"{'Parameter':<10} {'Index':<8} {'Analytical':<15} {'Numerical':<15} {'Difference':<15}")
print("-" * 70)

# Check a few weights
checks = [
    ('W1', 0, 0), ('W1', 1, 0),
    ('W2', 0, 0), ('W2', 1, 1),
    ('W3', 0, 0), ('W3', 0, 1)
]

for param_name, i, j in checks:
    # Analytical
    grad_key = 'd' + param_name
    analytical = analytical_grads[grad_key][i, j] if analytical_grads[grad_key].ndim == 2 else analytical_grads[grad_key][i, 0]

    # Numerical
    numerical = numerical_gradient_solution(test_net, x_test, t_test, param_name, i, j)

    # Difference
    diff = abs(analytical - numerical)
    status = "✓" if diff < 1e-7 else "✗"

    print(f"{param_name:<10} ({i},{j})    {analytical:<15.8f} {numerical:<15.8f} {diff:<15.10f} {status}")

print("\n✓ Numerical gradient checking complete!")


# ============================================================================
# EXERCISE 4: Debug Gradient Issues
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Find and Fix Gradient Bug")
print("=" * 70)

print("""
Task: This backprop implementation has a bug. Find and fix it!

The bug causes incorrect gradients. Use numerical gradient checking
to identify which gradient is wrong, then fix the code.
""")


# BUGGY CODE
class BuggyNetwork:
    """This network has a bug in backward()! Can you find it?"""

    def __init__(self):
        self.W1 = np.random.randn(3, 2) * 0.5
        self.b1 = np.zeros((3, 1))
        self.W2 = np.random.randn(1, 3) * 0.5
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, x):
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.y = self.sigmoid(self.z2)
        self.x = x
        return self.y

    def backward(self, t):
        m = self.x.shape[1]

        # Layer 2
        dL_dz2 = (self.y - t) * self.sigmoid_derivative(self.y)
        dL_dW2 = (1 / m) * (dL_dz2 @ self.a1.T)
        dL_db2 = (1 / m) * np.sum(dL_dz2, axis=1, keepdims=True)

        # Layer 1 - BUG IS HERE!
        dL_da1 = self.W2 @ dL_dz2  # ← BUG: Missing .T transpose!
        dL_dz1 = dL_da1 * self.sigmoid_derivative(self.a1)
        dL_dW1 = (1 / m) * (dL_dz1 @ self.x.T)
        dL_db1 = (1 / m) * np.sum(dL_dz1, axis=1, keepdims=True)

        return {'dW1': dL_dW1, 'db1': dL_db1, 'dW2': dL_dW2, 'db2': dL_db2}

    def compute_loss(self, y, t):
        return np.mean((y - t) ** 2)


print("\n--- Testing Buggy Network ---")
buggy_net = BuggyNetwork()

x_bug = np.array([[0.5], [0.8]])
t_bug = np.array([[1.0]])

try:
    y_bug = buggy_net.forward(x_bug)
    grads_bug = buggy_net.backward(t_bug)

    # Check one gradient numerically
    numerical_w1 = numerical_gradient_solution(buggy_net, x_bug, t_bug, 'W1', 0, 0)
    analytical_w1 = grads_bug['dW1'][0, 0]

    print(f"W1[0,0] - Analytical: {analytical_w1:.8f}")
    print(f"W1[0,0] - Numerical:  {numerical_w1:.8f}")
    print(f"Difference: {abs(analytical_w1 - numerical_w1):.10f}")

    if abs(analytical_w1 - numerical_w1) > 1e-7:
        print("✗ Bug detected! Gradients don't match!")
        print("\n💡 HINT: Check the transpose in backward propagation!")
    else:
        print("✓ No bug detected (or bug is in different parameter)")

except Exception as e:
    print(f"Error occurred: {e}")
    print("💡 HINT: Check matrix shapes in backward propagation!")

print("\n--- Fixed Version ---")


class FixedNetwork(BuggyNetwork):
    """Fixed version with correct backprop"""

    def backward(self, t):
        m = self.x.shape[1]

        # Layer 2
        dL_dz2 = (self.y - t) * self.sigmoid_derivative(self.y)
        dL_dW2 = (1 / m) * (dL_dz2 @ self.a1.T)
        dL_db2 = (1 / m) * np.sum(dL_dz2, axis=1, keepdims=True)

        # Layer 1 - FIXED!
        dL_da1 = self.W2.T @ dL_dz2  # ✓ Added .T transpose!
        dL_dz1 = dL_da1 * self.sigmoid_derivative(self.a1)
        dL_dW1 = (1 / m) * (dL_dz1 @ self.x.T)
        dL_db1 = (1 / m) * np.sum(dL_dz1, axis=1, keepdims=True)

        return {'dW1': dL_dW1, 'db1': dL_db1, 'dW2': dL_dW2, 'db2': dL_db2}


fixed_net = FixedNetwork()
y_fixed = fixed_net.forward(x_bug)
grads_fixed = fixed_net.backward(t_bug)

numerical_w1_fixed = numerical_gradient_solution(fixed_net, x_bug, t_bug, 'W1', 0, 0)
analytical_w1_fixed = grads_fixed['dW1'][0, 0]

print(f"W1[0,0] - Analytical: {analytical_w1_fixed:.8f}")
print(f"W1[0,0] - Numerical:  {numerical_w1_fixed:.8f}")
print(f"Difference: {abs(analytical_w1_fixed - numerical_w1_fixed):.10f}")

if abs(analytical_w1_fixed - numerical_w1_fixed) < 1e-7:
    print("✓ Bug fixed! Gradients now match!")


# ============================================================================
# EXERCISE 5: Vanishing Gradients Experiment
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Observe Vanishing Gradients")
print("=" * 70)

print("""
Task: Create a DEEP network (many layers) and observe vanishing gradients.

Requirements:
1. Create network with 10 layers (all sigmoid activations)
2. Track gradient magnitudes at each layer
3. Observe that gradients get smaller in earlier layers
4. Compare with ReLU activations

This demonstrates why deep sigmoid networks were hard to train!
""")


class DeepNetwork:
    """Very deep network to observe vanishing gradients"""

    def __init__(self, num_layers=10, layer_size=4):
        self.num_layers = num_layers
        self.weights = []
        self.biases = []

        # Create many layers
        sizes = [2] + [layer_size] * num_layers + [1]
        for i in range(len(sizes) - 1):
            W = np.random.randn(sizes[i + 1], sizes[i]) * 0.5
            b = np.zeros((sizes[i + 1], 1))
            self.weights.append(W)
            self.biases.append(b)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, x):
        self.activations = [x]
        a = x

        for W, b in zip(self.weights, self.biases):
            z = W @ a + b
            a = self.sigmoid(z)
            self.activations.append(a)

        return a

    def backward(self, t):
        m = self.activations[0].shape[1]
        gradients = []

        # Start from output
        dL_dz = (self.activations[-1] - t) * self.sigmoid_derivative(self.activations[-1])

        # Backpropagate through all layers
        for i in reversed(range(len(self.weights))):
            # Gradient for this layer
            dL_dW = (1 / m) * (dL_dz @ self.activations[i].T)
            dL_db = (1 / m) * np.sum(dL_dz, axis=1, keepdims=True)

            # Store gradient magnitude
            grad_magnitude = np.linalg.norm(dL_dW)
            gradients.append(grad_magnitude)

            # Propagate to previous layer (if not first layer)
            if i > 0:
                dL_da = self.weights[i].T @ dL_dz
                dL_dz = dL_da * self.sigmoid_derivative(self.activations[i])

        return list(reversed(gradients))  # Return in forward order


print("\n--- Observing Vanishing Gradients ---")

deep_net = DeepNetwork(num_layers=10, layer_size=4)

# Single example
x_deep = np.array([[0.5], [0.8]])
t_deep = np.array([[1.0]])

# Forward and backward
y_deep = deep_net.forward(x_deep)
gradient_mags = deep_net.backward(t_deep)

print("\nGradient magnitudes by layer (input → output):")
for i, mag in enumerate(gradient_mags):
    print(f"  Layer {i+1}: {mag:.10f}")

print("\n--- Analysis ---")
first_layer_grad = gradient_mags[0]
last_layer_grad = gradient_mags[-1]
ratio = first_layer_grad / last_layer_grad if last_layer_grad > 0 else 0

print(f"First layer gradient: {first_layer_grad:.10f}")
print(f"Last layer gradient:  {last_layer_grad:.10f}")
print(f"Ratio (first/last):   {ratio:.10f}")

if ratio < 0.01:
    print("\n✓ Vanishing gradient observed!")
    print("  Gradients in early layers are MUCH smaller than in later layers.")
    print("  This makes deep sigmoid networks hard to train!")
    print("\n  Solutions:")
    print("  - Use ReLU activation (gradient = 1 for z > 0)")
    print("  - Use batch normalization")
    print("  - Use residual connections (ResNet)")
else:
    print("\n  Gradients are similar across layers (no vanishing)")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE SUMMARY")
print("=" * 70)

print("""
✅ What You Practiced:

1. Manual gradient calculation
   - Forward pass step-by-step
   - Backward pass using chain rule
   - Weight updates

2. Implementing backprop for multi-layer networks
   - 3-layer network with full backprop
   - Proper gradient flow through all layers
   - Correct use of chain rule

3. Numerical gradient checking
   - Finite difference approximation
   - Verifying analytical gradients
   - Debugging gradient bugs

4. Finding and fixing bugs
   - Using gradient checking to identify errors
   - Common mistake: missing transpose
   - Shape debugging

5. Vanishing gradients
   - Why deep sigmoid networks fail
   - Gradient magnitude across layers
   - Modern solutions (ReLU, batch norm)

🎯 Key Skills Developed:

- Calculate gradients manually (understanding!)
- Implement backprop for any architecture
- Debug gradient computation errors
- Verify correctness numerically
- Understand vanishing gradient problem

🔗 Connection to Real AI:

These exercises cover the EXACT challenges that researchers faced:
- 1980s: Backprop invented, but couldn't train deep networks
- 1990s-2000s: Vanishing gradients limited depth to 2-3 layers
- 2010s: ReLU + batch norm enabled 100+ layer networks
- Today: Transformers with 96 layers (GPT-3) trained successfully!

You now understand the foundations AND the solutions!

🔜 Next Steps:

Ready for Lesson 5: Training Loop
- Batching data for efficiency
- Epochs and iterations
- Train/validation/test splits
- Complete MNIST classifier with 95%+ accuracy!

You've mastered backpropagation - the hardest part is done! 🎉
""")

print("\n" + "=" * 70)
print("Excellent work! You can now implement backprop for any network!")
print("=" * 70)

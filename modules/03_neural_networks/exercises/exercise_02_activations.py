"""
Exercise 2: Activation Functions - Practice Problems

This exercise helps you master activation functions by implementing and using them.

DIFFICULTY LEVELS:
- Exercises 1-3: Beginner (understand the basics)
- Exercises 4-6: Intermediate (apply to networks)
- Exercises 7-10: Advanced (analyze and optimize)

For .NET developers: Think of activation functions like IValueConverter in WPF -
they transform data from one form to another!
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# EXERCISE 1: Implement ReLU Activation (Beginner)
# ============================================================================

def exercise_1_relu():
    """
    TASK: Implement the ReLU (Rectified Linear Unit) activation function

    What is ReLU?
    - ReLU means: "Return the value if positive, otherwise return 0"
    - Formula: ReLU(x) = max(0, x)
    - Example: ReLU(-5) = 0, ReLU(3) = 3

    For .NET devs: Like Math.Max(0, x) in C#

    Why ReLU?
    - Simple and fast
    - Prevents vanishing gradient problem
    - Used in most modern networks (including GPT!)
    """
    print("\n" + "="*70)
    print("EXERCISE 1: Implement ReLU Activation")
    print("="*70)

    # TODO: Implement ReLU function
    def relu(x):
        """
        Implement ReLU activation.

        Args:
            x: Input value or array

        Returns:
            ReLU(x) = max(0, x)

        Hint: Use np.maximum(0, x) - it works element-wise!
        """
        # YOUR CODE HERE
        pass

    # TODO: Implement ReLU derivative
    def relu_derivative(x):
        """
        Implement ReLU derivative for backpropagation.

        The derivative of ReLU is:
        - 1 if x > 0
        - 0 if x <= 0

        Args:
            x: Input value or array

        Returns:
            Derivative: 1 where x > 0, else 0

        Hint: Use (x > 0).astype(float)
        """
        # YOUR CODE HERE
        pass

    # Test your implementation
    test_inputs = np.array([-2, -1, 0, 1, 2])

    print("\nTest inputs:", test_inputs)
    print("Expected ReLU output: [0, 0, 0, 1, 2]")
    print("Your ReLU output:", relu(test_inputs))

    print("\nExpected ReLU derivative: [0, 0, 0, 1, 1]")
    print("Your ReLU derivative:", relu_derivative(test_inputs))


# ============================================================================
# EXERCISE 2: Implement Sigmoid Activation (Beginner)
# ============================================================================

def exercise_2_sigmoid():
    """
    TASK: Implement the Sigmoid activation function

    What is Sigmoid?
    - Sigmoid squashes any value into range (0, 1)
    - Formula: σ(x) = 1 / (1 + e^(-x))
    - Example: σ(0) = 0.5, σ(large positive) ≈ 1, σ(large negative) ≈ 0

    For .NET devs: Like normalizing a value to 0-1 range

    When to use Sigmoid?
    - Binary classification (yes/no, spam/not spam)
    - Output layer for probabilities
    - Gate mechanisms in LSTM networks
    """
    print("\n" + "="*70)
    print("EXERCISE 2: Implement Sigmoid Activation")
    print("="*70)

    # TODO: Implement Sigmoid function
    def sigmoid(x):
        """
        Implement Sigmoid activation.

        Formula: σ(x) = 1 / (1 + e^(-x))

        Args:
            x: Input value or array

        Returns:
            Sigmoid(x) - value between 0 and 1

        Hint: Use np.exp() for exponential
        """
        # YOUR CODE HERE
        pass

    # TODO: Implement Sigmoid derivative
    def sigmoid_derivative(x):
        """
        Implement Sigmoid derivative.

        The derivative of sigmoid has a beautiful property:
        σ'(x) = σ(x) * (1 - σ(x))

        This means: first compute sigmoid, then use this formula!

        Args:
            x: Input value or array

        Returns:
            Derivative of sigmoid

        Hint: Call sigmoid(x) first, then use the formula
        """
        # YOUR CODE HERE
        pass

    # Test your implementation
    test_inputs = np.array([-2, -1, 0, 1, 2])

    print("\nTest inputs:", test_inputs)
    print("Expected Sigmoid(0) = 0.5")
    print("Your Sigmoid outputs:", sigmoid(test_inputs))

    print("\nExpected max derivative at x=0 (should be 0.25)")
    print("Your Sigmoid derivatives:", sigmoid_derivative(test_inputs))


# ============================================================================
# EXERCISE 3: Implement Tanh Activation (Beginner)
# ============================================================================

def exercise_3_tanh():
    """
    TASK: Implement the Tanh (Hyperbolic Tangent) activation function

    What is Tanh?
    - Tanh squashes values into range (-1, 1)
    - Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    - Example: tanh(0) = 0, tanh(large positive) ≈ 1, tanh(large negative) ≈ -1

    For .NET devs: Like normalizing to -1 to +1 range

    Tanh vs Sigmoid?
    - Tanh: Output centered at 0 (range: -1 to 1)
    - Sigmoid: Output centered at 0.5 (range: 0 to 1)
    - Tanh often works better in hidden layers (zero-centered)
    """
    print("\n" + "="*70)
    print("EXERCISE 3: Implement Tanh Activation")
    print("="*70)

    # TODO: Implement Tanh function
    def tanh(x):
        """
        Implement Tanh activation.

        Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

        Args:
            x: Input value or array

        Returns:
            Tanh(x) - value between -1 and 1

        Hint: You can also use np.tanh(x) to check your answer!
        """
        # YOUR CODE HERE
        pass

    # TODO: Implement Tanh derivative
    def tanh_derivative(x):
        """
        Implement Tanh derivative.

        The derivative formula:
        tanh'(x) = 1 - tanh(x)^2

        Args:
            x: Input value or array

        Returns:
            Derivative of tanh

        Hint: Call tanh(x) first, then use the formula
        """
        # YOUR CODE HERE
        pass

    # Test your implementation
    test_inputs = np.array([-2, -1, 0, 1, 2])

    print("\nTest inputs:", test_inputs)
    print("Expected Tanh(0) = 0")
    print("Your Tanh outputs:", tanh(test_inputs))

    print("\nExpected max derivative at x=0 (should be 1.0)")
    print("Your Tanh derivatives:", tanh_derivative(test_inputs))


# ============================================================================
# EXERCISE 4: Implement Softmax Activation (Intermediate)
# ============================================================================

def exercise_4_softmax():
    """
    TASK: Implement Softmax for multi-class classification

    What is Softmax?
    - Converts multiple numbers into probabilities (sum = 1)
    - Formula: softmax(x)_i = e^(x_i) / Σ(e^(x_j))
    - Example: [1, 2, 3] → [0.09, 0.24, 0.67] (approximately)

    For .NET devs: Like converting scores to percentages that sum to 100%

    When to use Softmax?
    - Multi-class classification (cat/dog/bird)
    - Output layer when predicting one of many classes
    - Used in GPT to predict next token!
    """
    print("\n" + "="*70)
    print("EXERCISE 4: Implement Softmax Activation")
    print("="*70)

    # TODO: Implement Softmax function
    def softmax(x):
        """
        Implement Softmax activation.

        Formula: softmax(x)_i = e^(x_i) / Σ(e^(x_j))

        Important: For numerical stability, subtract max(x) before exponential!
        This prevents overflow with large numbers.

        Args:
            x: Input array of scores (logits)

        Returns:
            Probabilities that sum to 1

        Hint:
        1. Subtract np.max(x) from x (numerical stability)
        2. Compute exp_x = np.exp(x)
        3. Return exp_x / np.sum(exp_x)
        """
        # YOUR CODE HERE
        pass

    # Test your implementation
    test_scores = np.array([1.0, 2.0, 3.0])

    print("\nTest scores:", test_scores)
    print("Expected: probabilities that sum to 1.0")

    result = softmax(test_scores)
    print("Your Softmax output:", result)
    print("Sum of probabilities:", np.sum(result))
    print("(Should be exactly 1.0)")

    print("\nInterpretation:")
    print(f"  Score 1 (lowest) → {result[0]:.1%} probability")
    print(f"  Score 2 (middle) → {result[1]:.1%} probability")
    print(f"  Score 3 (highest) → {result[2]:.1%} probability")


# ============================================================================
# EXERCISE 5: Compare Activation Functions (Intermediate)
# ============================================================================

def exercise_5_compare_activations():
    """
    TASK: Compare how different activations transform the same input

    This helps you understand:
    - How each activation shapes the output
    - When to use which activation
    - Visual differences between functions
    """
    print("\n" + "="*70)
    print("EXERCISE 5: Compare Activation Functions")
    print("="*70)

    # TODO: Implement all activations from previous exercises
    # (Copy your implementations here)

    def relu(x):
        # YOUR CODE FROM EXERCISE 1
        return np.maximum(0, x)

    def sigmoid(x):
        # YOUR CODE FROM EXERCISE 2
        return 1 / (1 + np.exp(-x))

    def tanh(x):
        # YOUR CODE FROM EXERCISE 3
        return np.tanh(x)

    # Test range
    x = np.linspace(-5, 5, 100)

    # TODO: Compute all activations
    y_relu = relu(x)
    y_sigmoid = sigmoid(x)
    y_tanh = tanh(x)

    # TODO: Create comparison plot
    plt.figure(figsize=(12, 5))

    # Plot 1: All activations
    plt.subplot(1, 2, 1)
    plt.plot(x, y_relu, label='ReLU', linewidth=2)
    plt.plot(x, y_sigmoid, label='Sigmoid', linewidth=2)
    plt.plot(x, y_tanh, label='Tanh', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Activation Functions Comparison')
    plt.xlabel('Input (x)')
    plt.ylabel('Output')

    # Plot 2: Zoomed in around origin
    plt.subplot(1, 2, 2)
    mask = (x >= -3) & (x <= 3)
    plt.plot(x[mask], y_relu[mask], label='ReLU', linewidth=2)
    plt.plot(x[mask], y_sigmoid[mask], label='Sigmoid', linewidth=2)
    plt.plot(x[mask], y_tanh[mask], label='Tanh', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Zoomed View (x: -3 to 3)')
    plt.xlabel('Input (x)')
    plt.ylabel('Output')

    plt.tight_layout()
    plt.savefig('exercise_02_activation_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_02_activation_comparison.png")
    plt.show()

    print("\nObservations:")
    print("1. ReLU: Zero for negative, linear for positive")
    print("2. Sigmoid: S-curve, outputs between 0 and 1")
    print("3. Tanh: S-curve, outputs between -1 and 1 (centered at 0)")


# ============================================================================
# EXERCISE 6: Activation in Neural Network (Intermediate)
# ============================================================================

def exercise_6_activation_in_network():
    """
    TASK: Use different activations in a simple neural network

    This shows you:
    - How activation choice affects learning
    - Practical application of activations
    - Why ReLU is popular
    """
    print("\n" + "="*70)
    print("EXERCISE 6: Activation Functions in Neural Network")
    print("="*70)

    # Simple dataset: XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR outputs

    print("\nDataset (XOR problem):")
    print("Input → Output")
    for i in range(len(X)):
        print(f"{X[i]} → {y[i][0]}")

    # TODO: Build a simple 2-layer network
    class SimpleNetwork:
        def __init__(self, activation='relu'):
            """
            2-layer network: 2 → 3 → 1
            """
            self.W1 = np.random.randn(3, 2) * 0.5  # Hidden layer
            self.b1 = np.zeros((3, 1))
            self.W2 = np.random.randn(1, 3) * 0.5  # Output layer
            self.b2 = np.zeros((1, 1))
            self.activation = activation

        def forward(self, X):
            """
            TODO: Implement forward pass with chosen activation

            Steps:
            1. Compute z1 = W1 @ X + b1
            2. Apply activation → a1
            3. Compute z2 = W2 @ a1 + b2
            4. Apply sigmoid to output → a2

            Hint: Use the functions from previous exercises!
            """
            # YOUR CODE HERE
            pass

        def predict(self, X):
            """Make predictions"""
            return (self.forward(X.T) > 0.5).astype(int)

    # TODO: Test with different activations
    activations = ['relu', 'sigmoid', 'tanh']

    print("\nTesting different activations:")
    print("(Network not trained, just showing forward pass)")

    for act in activations:
        net = SimpleNetwork(activation=act)
        predictions = net.predict(X)
        print(f"\n{act.upper():8} → predictions: {predictions.ravel()}")
        print(f"           (random weights, so predictions are random)")


# ============================================================================
# EXERCISE 7: Dying ReLU Problem (Advanced)
# ============================================================================

def exercise_7_dying_relu():
    """
    TASK: Understand the "dying ReLU" problem

    What is dying ReLU?
    - When neurons output only 0, they stop learning
    - Happens when weights become very negative
    - Gradient becomes 0, so no weight updates!

    This is why people invented:
    - Leaky ReLU
    - PReLU (Parametric ReLU)
    - ELU (Exponential Linear Unit)
    """
    print("\n" + "="*70)
    print("EXERCISE 7: Understanding Dying ReLU Problem")
    print("="*70)

    # TODO: Implement Leaky ReLU
    def leaky_relu(x, alpha=0.01):
        """
        Leaky ReLU allows small gradient for negative values.

        Formula:
        - If x > 0: return x
        - If x <= 0: return alpha * x (small slope instead of 0)

        Args:
            x: Input
            alpha: Slope for negative values (typically 0.01)

        Returns:
            Leaky ReLU output

        Hint: np.where(x > 0, x, alpha * x)
        """
        # YOUR CODE HERE
        pass

    # TODO: Compare ReLU vs Leaky ReLU
    x = np.linspace(-3, 3, 100)

    y_relu = np.maximum(0, x)
    y_leaky = leaky_relu(x, alpha=0.01)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, y_relu, label='ReLU', linewidth=2)
    plt.plot(x, y_leaky, label='Leaky ReLU (α=0.01)', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('ReLU vs Leaky ReLU')
    plt.xlabel('Input')
    plt.ylabel('Output')

    # Zoom in on negative region
    plt.subplot(1, 2, 2)
    mask = x < 0
    plt.plot(x[mask], y_relu[mask], label='ReLU', linewidth=2)
    plt.plot(x[mask], y_leaky[mask], label='Leaky ReLU', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Negative Region (Zoomed)')
    plt.xlabel('Input')
    plt.ylabel('Output')

    plt.tight_layout()
    plt.savefig('exercise_02_dying_relu.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_02_dying_relu.png")
    plt.show()

    print("\nKey insight:")
    print("- ReLU: Gradient = 0 for x < 0 (neuron can 'die')")
    print("- Leaky ReLU: Small gradient for x < 0 (neuron stays alive)")


# ============================================================================
# EXERCISE 8: Implement GELU (Advanced)
# ============================================================================

def exercise_8_gelu():
    """
    TASK: Implement GELU - the activation function used in GPT!

    What is GELU?
    - Gaussian Error Linear Unit
    - Used in BERT, GPT-2, GPT-3
    - Smoother than ReLU
    - Better for transformers!

    Formula (approximation):
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    print("\n" + "="*70)
    print("EXERCISE 8: Implement GELU (GPT's Activation)")
    print("="*70)

    # TODO: Implement GELU
    def gelu(x):
        """
        Implement GELU activation (approximation).

        Formula:
        GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

        Args:
            x: Input

        Returns:
            GELU activation

        Hint: Break it into steps:
        1. Compute inner = x + 0.044715 * x^3
        2. Compute middle = sqrt(2/pi) * inner
        3. Compute tanh_part = tanh(middle)
        4. Return 0.5 * x * (1 + tanh_part)
        """
        # YOUR CODE HERE
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        inner = x + 0.044715 * (x ** 3)
        return 0.5 * x * (1 + np.tanh(sqrt_2_over_pi * inner))

    # TODO: Compare GELU with ReLU
    x = np.linspace(-3, 3, 200)

    y_relu = np.maximum(0, x)
    y_gelu = gelu(x)

    plt.figure(figsize=(12, 5))

    # Plot 1: Full comparison
    plt.subplot(1, 2, 1)
    plt.plot(x, y_relu, label='ReLU', linewidth=2)
    plt.plot(x, y_gelu, label='GELU (GPT)', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('ReLU vs GELU')
    plt.xlabel('Input')
    plt.ylabel('Output')

    # Plot 2: Difference
    plt.subplot(1, 2, 2)
    plt.plot(x, y_gelu - y_relu, linewidth=2, color='red')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('Difference (GELU - ReLU)')
    plt.xlabel('Input')
    plt.ylabel('Difference')

    plt.tight_layout()
    plt.savefig('exercise_02_gelu_vs_relu.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_02_gelu_vs_relu.png")
    plt.show()

    print("\nGELU vs ReLU:")
    print("- ReLU: Hard cutoff at 0 (discontinuous derivative)")
    print("- GELU: Smooth curve (continuous derivative)")
    print("- GELU: Allows small negative values (like Leaky ReLU)")
    print("- GPT uses GELU because it's smoother and trains better!")


# ============================================================================
# EXERCISE 9: Gradient Vanishing Problem (Advanced)
# ============================================================================

def exercise_9_gradient_vanishing():
    """
    TASK: Understand gradient vanishing with Sigmoid/Tanh

    What is gradient vanishing?
    - In deep networks, gradients become very small
    - Happens with Sigmoid/Tanh (flat regions)
    - Makes training slow or impossible!

    This is why ReLU became popular!
    """
    print("\n" + "="*70)
    print("EXERCISE 9: Gradient Vanishing Problem")
    print("="*70)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)

    def relu_derivative(x):
        return (x > 0).astype(float)

    # TODO: Compute gradients at different points
    x = np.linspace(-5, 5, 200)

    grad_sigmoid = sigmoid_derivative(x)
    grad_relu = relu_derivative(x)

    plt.figure(figsize=(12, 5))

    # Plot 1: Derivatives
    plt.subplot(1, 2, 1)
    plt.plot(x, grad_sigmoid, label='Sigmoid derivative', linewidth=2)
    plt.plot(x, grad_relu, label='ReLU derivative', linewidth=2, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Activation Derivatives (Gradients)')
    plt.xlabel('Input')
    plt.ylabel('Derivative')

    # Plot 2: Gradient flow through 10 layers
    plt.subplot(1, 2, 2)

    # Simulate gradient flowing backward through 10 layers
    layers = np.arange(1, 11)

    # For sigmoid: gradient at x=2 (typical value)
    sigmoid_grad = sigmoid_derivative(2.0)
    sigmoid_flow = sigmoid_grad ** layers

    # For ReLU: gradient is 1 (for positive values)
    relu_flow = np.ones_like(layers)

    plt.plot(layers, sigmoid_flow, 'o-', label='Sigmoid', linewidth=2)
    plt.plot(layers, relu_flow, 's-', label='ReLU', linewidth=2)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Gradient Flow Through 10 Layers')
    plt.xlabel('Layer Depth')
    plt.ylabel('Gradient Magnitude (log scale)')

    plt.tight_layout()
    plt.savefig('exercise_02_gradient_vanishing.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as: exercise_02_gradient_vanishing.png")
    plt.show()

    print("\nKey observations:")
    print(f"Sigmoid derivative max: {np.max(grad_sigmoid):.3f} (at x=0)")
    print(f"ReLU derivative: 1.0 (for x > 0)")
    print(f"\nAfter 10 layers:")
    print(f"  Sigmoid gradient: {sigmoid_flow[-1]:.6f} (vanished!)")
    print(f"  ReLU gradient: {relu_flow[-1]:.6f} (preserved!)")
    print("\nThis is why ReLU is preferred for deep networks!")


# ============================================================================
# EXERCISE 10: Activation Function Decision Guide (Advanced)
# ============================================================================

def exercise_10_decision_guide():
    """
    TASK: Create a decision guide for choosing activations

    This summarizes everything you learned!
    """
    print("\n" + "="*70)
    print("EXERCISE 10: Activation Function Decision Guide")
    print("="*70)

    print("\n" + "="*70)
    print("ACTIVATION FUNCTION DECISION GUIDE")
    print("="*70)

    guide = """

    ╔═══════════════════════════════════════════════════════════════╗
    ║  WHICH ACTIVATION FUNCTION SHOULD I USE?                      ║
    ╚═══════════════════════════════════════════════════════════════╝

    ┌─────────────────────────────────────────────────────────────┐
    │ HIDDEN LAYERS (Inside the network)                          │
    └─────────────────────────────────────────────────────────────┘

    Default Choice: ReLU
    ├─ Fast computation
    ├─ Avoids gradient vanishing
    ├─ Used in most CNN, MLP networks
    └─ Formula: max(0, x)

    For Transformers/GPT: GELU
    ├─ Smoother than ReLU
    ├─ Better for attention mechanisms
    ├─ Used in BERT, GPT-2, GPT-3
    └─ Formula: 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))

    If ReLU dies: Leaky ReLU
    ├─ Small gradient for negative values
    ├─ Prevents dead neurons
    ├─ Good backup for ReLU
    └─ Formula: x if x>0, else 0.01*x

    For RNNs: Tanh
    ├─ Zero-centered (-1 to 1)
    ├─ Traditional choice for recurrent nets
    ├─ Being replaced by LSTM/GRU
    └─ Formula: (e^x - e^-x)/(e^x + e^-x)

    ┌─────────────────────────────────────────────────────────────┐
    │ OUTPUT LAYER (Final predictions)                            │
    └─────────────────────────────────────────────────────────────┘

    Binary Classification (yes/no): Sigmoid
    ├─ Outputs probability (0 to 1)
    ├─ Use with Binary Cross-Entropy loss
    ├─ Example: spam detection, sentiment analysis
    └─ Formula: 1/(1+e^-x)

    Multi-class Classification: Softmax
    ├─ Outputs probability distribution
    ├─ Probabilities sum to 1
    ├─ Use with Categorical Cross-Entropy loss
    ├─ Example: image classification, next word prediction
    └─ Formula: e^xi / Σ(e^xj)

    Regression (predicting numbers): Linear (no activation)
    ├─ Direct output of weighted sum
    ├─ Can output any real number
    ├─ Example: house price prediction, temperature
    └─ Formula: just z = Wx + b

    ┌─────────────────────────────────────────────────────────────┐
    │ QUICK REFERENCE TABLE                                       │
    └─────────────────────────────────────────────────────────────┘

    Task Type          | Hidden Layers | Output Layer    | Loss Function
    ───────────────────┼───────────────┼─────────────────┼──────────────────
    Image recognition  | ReLU          | Softmax         | Cross-Entropy
    Text generation    | GELU          | Softmax         | Cross-Entropy
    Spam detection     | ReLU          | Sigmoid         | Binary Cross-Ent
    House price        | ReLU          | Linear          | MSE
    Deep network       | ReLU/GELU     | (depends)       | (depends)
    Transformer/GPT    | GELU          | Softmax         | Cross-Entropy

    ┌─────────────────────────────────────────────────────────────┐
    │ COMMON MISTAKES TO AVOID                                    │
    └─────────────────────────────────────────────────────────────┘

    ❌ Using Sigmoid in hidden layers → Use ReLU instead
       (Sigmoid causes gradient vanishing)

    ❌ Using ReLU in output layer for classification → Use Sigmoid/Softmax
       (Need probabilities, not positive numbers)

    ❌ Using different activations randomly → Be consistent
       (Usually all hidden layers use same activation)

    ❌ Forgetting activation entirely → Network becomes linear!
       (Without activation, multiple layers = single layer)

    ┌─────────────────────────────────────────────────────────────┐
    │ MODERN BEST PRACTICES (2024)                                │
    └─────────────────────────────────────────────────────────────┘

    ✓ Default: ReLU for hidden layers, task-specific for output
    ✓ Transformers: GELU everywhere (GPT does this)
    ✓ If ReLU fails: Try Leaky ReLU or GELU
    ✓ Always use activation except last layer in regression
    ✓ Match loss function to output activation

    """

    print(guide)

    print("\n" + "="*70)
    print("CONGRATULATIONS!")
    print("="*70)
    print("\nYou now understand:")
    print("✓ All major activation functions (ReLU, Sigmoid, Tanh, Softmax, GELU)")
    print("✓ When to use each activation")
    print("✓ Why ReLU is the default choice")
    print("✓ Why GPT uses GELU")
    print("✓ How to avoid gradient vanishing")
    print("✓ Common mistakes and best practices")
    print("\nYou're ready for Lesson 3: Multi-Layer Networks!")


# ============================================================================
# MAIN: Run All Exercises
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("EXERCISE 2: ACTIVATION FUNCTIONS - PRACTICE PROBLEMS")
    print("="*70)
    print("\nThis exercise covers:")
    print("1. Implementing ReLU, Sigmoid, Tanh, Softmax, GELU")
    print("2. Understanding when to use each activation")
    print("3. Gradient vanishing problem")
    print("4. Modern best practices (GPT uses GELU!)")
    print("\n" + "="*70)

    # Run exercises (uncomment the ones you want to run)

    # Beginner
    exercise_1_relu()
    exercise_2_sigmoid()
    exercise_3_tanh()

    # Intermediate
    exercise_4_softmax()
    exercise_5_compare_activations()
    exercise_6_activation_in_network()

    # Advanced
    exercise_7_dying_relu()
    exercise_8_gelu()
    exercise_9_gradient_vanishing()
    exercise_10_decision_guide()

    print("\n" + "="*70)
    print("ALL EXERCISES COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the generated plots")
    print("2. Modify parameters and re-run")
    print("3. Try implementing variations (ELU, SELU, Swish)")
    print("4. Move on to Lesson 3: Multi-Layer Networks")
    print("\nGreat job! You now understand activation functions deeply!")

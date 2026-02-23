"""
Activation Functions - Making Neural Networks Non-Linear

This example demonstrates why activation functions are crucial
and compares all major activation functions.

What you'll see:
1. Why linear-only networks fail
2. All activation functions visualized
3. Derivative comparisons
4. Practical usage examples
5. Which activations GPT uses
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("ACTIVATION FUNCTIONS - Adding Non-Linearity to Neural Networks")
print("="*70)

# ==============================================================================
# PART 1: Activation Functions Library
# ==============================================================================

class Activations:
    """Complete library of activation functions and their derivatives"""

    @staticmethod
    def sigmoid(z):
        """Sigmoid: σ(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip for stability

    @staticmethod
    def sigmoid_derivative(z):
        """Derivative: σ'(z) = σ(z)(1 - σ(z))"""
        s = Activations.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def tanh(z):
        """Tanh: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))"""
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        """Derivative: tanh'(z) = 1 - tanh²(z)"""
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def relu(z):
        """ReLU: max(0, z)"""
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        """Derivative: 1 if z > 0, else 0"""
        return (z > 0).astype(float)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        """Leaky ReLU: z if z > 0, else alpha*z"""
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        """Derivative: 1 if z > 0, else alpha"""
        return np.where(z > 0, 1.0, alpha)

    @staticmethod
    def softmax(z):
        """Softmax: exp(z_i) / sum(exp(z))"""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @staticmethod
    def gelu(z):
        """GELU: Used in GPT!"""
        return 0.5 * z * (1 + np.tanh(
            np.sqrt(2/np.pi) * (z + 0.044715 * z**3)
        ))

print("✓ Activation functions library loaded")
print()

# ==============================================================================
# PART 2: Why We Need Activation Functions
# ==============================================================================

print("="*70)
print("EXAMPLE 1: Why Linear-Only Networks Fail")
print("="*70)

print("""
Problem: Stacking linear layers without activation functions

Layer 1: z1 = X @ W1 + b1
Layer 2: z2 = z1 @ W2 + b2

Expanding:
z2 = (X @ W1 + b1) @ W2 + b2
   = X @ (W1 @ W2) + (b1 @ W2 + b2)
   = X @ W_combined + b_combined

Result: Still just ONE linear transformation!
No matter how many layers, without activation = single layer.
""")

# Demonstration
X = np.random.randn(5, 3)
W1 = np.random.randn(3, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 2)
b2 = np.random.randn(2)

# Two "layers" without activation
z1 = X @ W1 + b1
z2 = z1 @ W2 + b2

# Equivalent single layer
W_combined = W1 @ W2
b_combined = b1 @ W2 + b2
z_single = X @ W_combined + b_combined

print(f"Two layers output:\n{z2[:2]}\n")
print(f"Single layer output:\n{z_single[:2]}\n")
print(f"Are they equal? {np.allclose(z2, z_single)}")
print("\n⚠️  Without activation, multiple layers collapse to single layer!")

# With activation (non-linear)
print("\n" + "-"*70)
print("With ReLU activation:")
z1 = X @ W1 + b1
a1 = Activations.relu(z1)  # ← Non-linearity!
z2 = a1 @ W2 + b2

# Try to express as single layer (impossible!)
print(f"Two layers with ReLU:\n{z2[:2]}\n")
print("✓ Cannot be expressed as single linear layer!")
print("✓ Network can now learn complex, non-linear patterns!")

# ==============================================================================
# PART 3: Visualizing All Activation Functions
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Comparing Activation Functions")
print("="*70)

x = np.linspace(-5, 5, 200)

# Compute all activations
activations = {
    'Sigmoid': Activations.sigmoid(x),
    'Tanh': Activations.tanh(x),
    'ReLU': Activations.relu(x),
    'Leaky ReLU': Activations.leaky_relu(x),
    'GELU': Activations.gelu(x)
}

# Plot activations
plt.figure(figsize=(15, 10))

# Plot activation functions
for i, (name, values) in enumerate(activations.items(), 1):
    plt.subplot(3, 2, i)
    plt.plot(x, values, linewidth=2, label=name)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('z')
    plt.ylabel(f'{name}(z)')
    plt.title(f'{name} Activation')
    plt.legend()

# Comparison plot
plt.subplot(3, 2, 6)
for name, values in activations.items():
    plt.plot(x, values, linewidth=2, label=name, alpha=0.7)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z')
plt.ylabel('Activation(z)')
plt.title('All Activations Compared')
plt.legend()

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150, bbox_inches='tight')
print("✓ Saved activation function plots to 'activation_functions.png'")

# ==============================================================================
# PART 4: Derivatives (Critical for Backpropagation!)
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Activation Function Derivatives")
print("="*70)

print("""
Why derivatives matter:
During backpropagation, we need:
  ∂Loss/∂W = ∂Loss/∂a × ∂a/∂z × ∂z/∂W
                        ↑
              activation derivative!
""")

# Compute derivatives
derivatives = {
    'Sigmoid': Activations.sigmoid_derivative(x),
    'Tanh': Activations.tanh_derivative(x),
    'ReLU': Activations.relu_derivative(x),
    'Leaky ReLU': Activations.leaky_relu_derivative(x)
}

# Plot derivatives
plt.figure(figsize=(15, 5))

for i, (name, values) in enumerate(derivatives.items(), 1):
    plt.subplot(1, 4, i)
    plt.plot(x, values, linewidth=2, label=f"{name}'")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('z')
    plt.ylabel(f"{name}'(z)")
    plt.title(f'{name} Derivative')
    plt.ylim(-0.1, 1.1)
    plt.legend()

plt.tight_layout()
plt.savefig('activation_derivatives.png', dpi=150, bbox_inches='tight')
print("✓ Saved derivative plots to 'activation_derivatives.png'")

# Vanishing gradient demonstration
print("\n" + "-"*70)
print("Vanishing Gradient Problem:")
print("-"*70)

z_large = np.array([-10, -5, 0, 5, 10])
sigmoid_grad = Activations.sigmoid_derivative(z_large)
tanh_grad = Activations.tanh_derivative(z_large)
relu_grad = Activations.relu_derivative(z_large)

print(f"z values:        {z_large}")
print(f"Sigmoid' (z):    {sigmoid_grad}")
print(f"Tanh' (z):       {tanh_grad}")
print(f"ReLU' (z):       {relu_grad}")

print("\n⚠️  Sigmoid & Tanh gradients → 0 for large |z| (vanishing!)")
print("✓  ReLU gradient = 1 for z > 0 (no vanishing!)")

# ==============================================================================
# PART 5: Practical Examples
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Practical Usage")
print("="*70)

# Test on same input
print("Testing all activations on same input:")
print("-"*70)
test_input = np.array([-2, -1, 0, 1, 2])

print(f"Input:        {test_input}")
print(f"Sigmoid:      {Activations.sigmoid(test_input)}")
print(f"Tanh:         {Activations.tanh(test_input)}")
print(f"ReLU:         {Activations.relu(test_input)}")
print(f"Leaky ReLU:   {Activations.leaky_relu(test_input)}")
print(f"GELU:         {Activations.gelu(test_input)}")

# ==============================================================================
# PART 6: Output Layer Examples
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 5: Output Layer Selection")
print("="*70)

# Binary classification
print("Binary Classification (Cat vs Dog):")
print("-"*70)
logits_binary = np.array([[2.5], [-1.3], [0.2]])
probs_binary = Activations.sigmoid(logits_binary)

print(f"Raw scores (logits): {logits_binary.flatten()}")
print(f"Probabilities:       {probs_binary.flatten()}")
print(f"Predictions:         {(probs_binary > 0.5).astype(int).flatten()}")
print("(0 = dog, 1 = cat)")

# Multi-class classification
print("\n" + "-"*70)
print("Multi-Class Classification (Digits 0-9):")
print("-"*70)
logits_multi = np.array([
    [2.0, 1.0, 0.1, -1.0, -2.0, 0.5, -0.5, 1.5, -1.5, 0.0],  # Sample 1
    [0.1, 0.2, 3.0, -0.1, 0.0, -1.0, 0.5, -0.5, 0.3, -0.2]   # Sample 2
])
probs_multi = Activations.softmax(logits_multi)

for i in range(2):
    print(f"\nSample {i+1}:")
    print(f"  Logits: {logits_multi[i]}")
    print(f"  Probabilities: {probs_multi[i]}")
    print(f"  Sum of probabilities: {probs_multi[i].sum():.6f}")
    print(f"  Predicted digit: {np.argmax(probs_multi[i])}")

# ==============================================================================
# PART 7: Network Layer Example
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 6: Using Activations in Network Layers")
print("="*70)

# Simulate a 2-layer network
batch_size = 4
input_size = 10
hidden_size = 8
output_size = 3

# Random input
X = np.random.randn(batch_size, input_size)

# Layer 1 weights
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros(hidden_size)

# Layer 2 weights
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros(output_size)

print(f"Input shape: {X.shape}")
print(f"Layer 1: {input_size} → {hidden_size} (with ReLU)")
print(f"Layer 2: {hidden_size} → {output_size} (with Softmax)")

# Forward pass
print("\n" + "-"*70)
print("Forward Pass:")

# Layer 1
z1 = X @ W1 + b1
print(f"\n1. Linear transform: z1 = X @ W1 + b1")
print(f"   Shape: {z1.shape}")
print(f"   Sample values: {z1[0, :3]}")

a1 = Activations.relu(z1)
print(f"\n2. ReLU activation: a1 = relu(z1)")
print(f"   Shape: {a1.shape}")
print(f"   Sample values: {a1[0, :3]}")
print(f"   Zeros introduced: {(a1 == 0).sum()} / {a1.size}")

# Layer 2
z2 = a1 @ W2 + b2
print(f"\n3. Linear transform: z2 = a1 @ W2 + b2")
print(f"   Shape: {z2.shape}")
print(f"   Sample values: {z2[0]}")

a2 = Activations.softmax(z2)
print(f"\n4. Softmax activation: a2 = softmax(z2)")
print(f"   Shape: {a2.shape}")
print(f"   Sample probabilities: {a2[0]}")
print(f"   Sum: {a2[0].sum():.6f} (should be 1.0)")
print(f"   Predicted classes: {np.argmax(a2, axis=1)}")

# ==============================================================================
# PART 8: Comparing Activations on Network
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 7: Effect of Different Hidden Activations")
print("="*70)

# Test different activations on same network
activations_to_test = {
    'Sigmoid': Activations.sigmoid,
    'Tanh': Activations.tanh,
    'ReLU': Activations.relu,
    'Leaky ReLU': Activations.leaky_relu,
    'GELU': Activations.gelu
}

results = {}

for name, activation_fn in activations_to_test.items():
    # Layer 1 with current activation
    z1 = X @ W1 + b1
    a1 = activation_fn(z1)

    # Layer 2 (always softmax for output)
    z2 = a1 @ W2 + b2
    a2 = Activations.softmax(z2)

    # Store statistics
    results[name] = {
        'hidden_mean': a1.mean(),
        'hidden_std': a1.std(),
        'hidden_zeros': (a1 == 0).sum() / a1.size,
        'output': a2[0]  # First sample
    }

print("Activation Function Statistics:")
print("-"*70)
print(f"{'Activation':<15} {'Hidden Mean':>12} {'Hidden Std':>12} {'% Zeros':>10}")
print("-"*70)

for name, stats in results.items():
    print(f"{name:<15} {stats['hidden_mean']:>12.4f} "
          f"{stats['hidden_std']:>12.4f} {stats['hidden_zeros']:>9.1%}")

print("\n✓ ReLU typically creates sparse activations (many zeros)")
print("✓ Sigmoid/Tanh saturate (values near 0 or 1/-1)")
print("✓ GELU is smooth like Sigmoid but doesn't saturate as much")

# ==============================================================================
# PART 9: GPT Connection
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 8: What GPT Uses")
print("="*70)

print("""
GPT Feed-Forward Layer (simplified):

def gpt_feed_forward(x, W1, b1, W2, b2):
    # Expand dimension
    hidden = x @ W1 + b1

    # GELU activation (GPT's choice!)
    activated = gelu(hidden)

    # Project back
    output = activated @ W2 + b2

    return output

Why GELU instead of ReLU?
1. Smoother (better for NLP)
2. Stochastic regularization properties
3. Better empirical performance on language tasks
4. State-of-the-art for transformers

But conceptually, it's the SAME as what you're learning!
""")

# Simulate GPT feed-forward layer
def gpt_feed_forward(x, W1, b1, W2, b2):
    """Simplified GPT feed-forward layer"""
    hidden = x @ W1 + b1
    activated = Activations.gelu(hidden)  # GELU!
    output = activated @ W2 + b2
    return output

# Test
x_gpt = np.random.randn(1, 768)  # GPT hidden state
W1_gpt = np.random.randn(768, 3072) * 0.01  # 4x expansion
b1_gpt = np.zeros(3072)
W2_gpt = np.random.randn(3072, 768) * 0.01  # Project back
b2_gpt = np.zeros(768)

output_gpt = gpt_feed_forward(x_gpt, W1_gpt, b1_gpt, W2_gpt, b2_gpt)

print(f"GPT input shape: {x_gpt.shape}")
print(f"After expansion: (1, 3072)  ← 4x larger!")
print(f"GPT output shape: {output_gpt.shape}")
print(f"\n✓ This exact pattern is inside every GPT transformer layer!")

# ==============================================================================
# PART 10: Performance Comparison
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 9: Computation Speed Comparison")
print("="*70)

import time

# Large input for timing
large_input = np.random.randn(10000, 1000)

print("Computing activation for 10,000 samples × 1,000 features:")
print("-"*70)

timings = {}

for name, activation_fn in activations_to_test.items():
    start = time.time()
    result = activation_fn(large_input)
    elapsed = (time.time() - start) * 1000  # Convert to ms
    timings[name] = elapsed
    print(f"{name:<15}: {elapsed:>8.2f} ms")

fastest = min(timings.values())
print("\n" + "-"*70)
print("Speedup relative to fastest:")
for name, elapsed in timings.items():
    speedup = elapsed / fastest
    marker = " ← FASTEST" if elapsed == fastest else ""
    print(f"{name:<15}: {speedup:>5.2f}x{marker}")

print("\n✓ ReLU is typically fastest (simple max operation)")
print("✓ GELU is slower (complex math) but better for NLP")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("✅ SUMMARY")
print("="*70)

print("""
What You Just Learned:

1. ✅ Why activation functions are essential (non-linearity!)
2. ✅ All major activation functions (Sigmoid, Tanh, ReLU, etc.)
3. ✅ How to compute derivatives (needed for backpropagation)
4. ✅ When to use which activation
5. ✅ Vanishing gradient problem (why ReLU is popular)
6. ✅ Output layer selection (Sigmoid vs Softmax)
7. ✅ What GPT uses (GELU!)

Key Takeaways:

📌 Hidden Layers:
   - Default: ReLU (fast, no vanishing gradient)
   - Transformers: GELU (better for NLP)
   - Avoid: Sigmoid/Tanh (vanishing gradients)

📌 Output Layer:
   - Binary classification: Sigmoid
   - Multi-class classification: Softmax
   - Regression: No activation (linear)

📌 The Rule:
   Without activation functions:
     10 layers = 1 layer (all linear!)

   With activation functions:
     10 layers can learn complex patterns!

📌 For Backpropagation:
   - Need derivatives of activations
   - ReLU derivative: super simple (0 or 1)
   - Sigmoid/Tanh: vanish for large |z|

Connection to GPT:
Every transformer layer uses:
  z = x @ W + b
  a = gelu(z)  ← GELU activation!

This is exactly what you learned, just at massive scale!

Next Steps:
1. Read Lesson 3.3: Multi-Layer Networks
   (Stack layers to build deep networks)
2. Run example_03_forward_pass.py
3. Complete exercise_02_activations.py

You now understand the "secret ingredient" that makes
neural networks capable of learning anything! 🎉
""")

print("="*70)
print("Generated plots:")
print("  - activation_functions.png")
print("  - activation_derivatives.png")
print("="*70)

# Show plots if running interactively
try:
    plt.show()
except:
    pass

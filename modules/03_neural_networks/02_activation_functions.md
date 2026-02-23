# Lesson 3.2: Activation Functions - Adding Non-Linearity

## 🎯 Learning Objectives

By the end of this lesson, you'll be able to:
- Explain why activation functions are essential
- Understand the problem with linear-only networks
- Implement ReLU, Sigmoid, Tanh, and Softmax
- Know when to use which activation function
- Understand derivatives for backpropagation
- See which activations GPT uses

---

## 🤔 Why Do We Need Activation Functions?

### The Problem with Linear-Only Networks

Remember the perceptron formula?
```
z = w·x + b
y = z  (no activation)
```

**What if we stack multiple linear layers?**

```python
# Layer 1 (linear)
z1 = X @ W1 + b1

# Layer 2 (linear)
z2 = z1 @ W2 + b2

# Mathematically, this is equivalent to:
z2 = (X @ W1 + b1) @ W2 + b2
   = X @ (W1 @ W2) + (b1 @ W2 + b2)
   = X @ W_combined + b_combined

# This is STILL just a single linear transformation!
# Multiple linear layers = Single linear layer
```

**Problem:** No matter how many linear layers you stack, you can only learn linear patterns!

### The Solution: Non-Linear Activation Functions

```python
# With activation function
z1 = X @ W1 + b1
a1 = relu(z1)        # ← Non-linearity!

z2 = a1 @ W2 + b2
a2 = relu(z2)        # ← Non-linearity!

# Now the network can learn complex, non-linear patterns!
```

**Key insight:** Activation functions allow neural networks to approximate ANY function, not just linear ones!

---

## 📊 Common Activation Functions

### 1. Step Function (Classic Perceptron)

```
step(z) = {1 if z > 0
          {0 otherwise
```

**Pros:**
- Simple to understand
- Binary output

**Cons:**
- ❌ Not differentiable (can't use gradient descent)
- ❌ Only works for simple problems
- ❌ Never used in modern networks

**Visual:**
```
  y
  1 |     ┌─────────
    |     │
  0 |─────┘
    ├─────┼─────► z
         0
```

---

### 2. Sigmoid (Logistic Function)

```
σ(z) = 1 / (1 + e^(-z))
```

**Properties:**
- Output range: (0, 1)
- Smooth S-curve
- Differentiable everywhere

**Derivative:**
```
σ'(z) = σ(z) × (1 - σ(z))
```

**Visual:**
```
  y
  1 |        ┌───────
    |       ╱
0.5 |      ╱
    |     ╱
  0 |─────┘
    ├─────┼─────► z
         0
```

**When to use:**
- ✅ Output layer for binary classification
- ✅ When you need probabilities (0 to 1)
- ❌ Hidden layers (suffers from vanishing gradients)

**Code:**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

---

### 3. Tanh (Hyperbolic Tangent)

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

**Properties:**
- Output range: (-1, 1)
- Zero-centered (better than sigmoid)
- Smooth S-curve

**Derivative:**
```
tanh'(z) = 1 - tanh²(z)
```

**Visual:**
```
  y
  1 |        ┌───────
    |       ╱
  0 |──────╱────────► z
    |     ╱
 -1 |─────┘
```

**When to use:**
- ✅ Hidden layers (better than sigmoid)
- ✅ When you need outputs centered at 0
- ❌ Still suffers from vanishing gradients

**Code:**
```python
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2
```

---

### 4. ReLU (Rectified Linear Unit) ⭐ Most Popular!

```
ReLU(z) = max(0, z)
```

**Properties:**
- Output range: [0, ∞)
- Dead simple to compute
- Fixes vanishing gradient problem

**Derivative:**
```
ReLU'(z) = {1 if z > 0
           {0 if z ≤ 0
```

**Visual:**
```
  y
    |        ╱
    |       ╱
    |      ╱
  0 |─────┘
    ├─────┼─────► z
         0
```

**When to use:**
- ✅ Hidden layers (DEFAULT CHOICE!)
- ✅ Fast to compute
- ✅ Works extremely well in practice
- ❌ Not for output layer

**Why it's so popular:**
1. Simple: `max(0, z)`
2. Fast: No exponentials
3. Sparse: Many neurons = 0 (efficient)
4. No vanishing gradient for positive values

**Code:**
```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

---

### 5. Leaky ReLU (Fixes "Dying ReLU")

```
LeakyReLU(z) = {z     if z > 0
               {αz    if z ≤ 0    (α = 0.01 typically)
```

**Properties:**
- Like ReLU but allows small negative values
- Prevents "dead neurons"

**Visual:**
```
  y
    |        ╱
    |       ╱
    |      ╱
  0 |─────╱ ← small slope
    ├────╱┼─────► z
        0
```

**When to use:**
- ✅ When ReLU neurons are "dying"
- ✅ Alternative to ReLU
- ✅ Works well in practice

**Code:**
```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)
```

---

### 6. Softmax (Multi-Class Classification) ⭐

```
softmax(z_i) = e^(z_i) / Σ(e^(z_j))
```

**Properties:**
- Converts scores to probabilities
- Outputs sum to 1
- Used ONLY in output layer

**Example:**
```
Input:  z = [2.0, 1.0, 0.1]

Step 1: exp(z) = [7.39, 2.72, 1.11]
Step 2: sum = 11.22
Step 3: softmax = [0.659, 0.242, 0.099]
        (sum = 1.0)
```

**When to use:**
- ✅ Output layer for multi-class classification
- ✅ When you need class probabilities
- ✅ MNIST, ImageNet, text classification

**Code:**
```python
def softmax(z):
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Derivative is complex, typically handled by loss function
```

---

### 7. GELU (Gaussian Error Linear Unit) - Used in GPT!

```
GELU(z) ≈ 0.5 × z × (1 + tanh(√(2/π) × (z + 0.044715 × z³)))
```

**Properties:**
- Smooth version of ReLU
- Used in BERT, GPT, modern transformers
- Better performance than ReLU for NLP

**Visual:**
```
  y
    |        ╱
    |       ╱
    |      ╱
  0 |─────╱  ← smooth curve
    ├────╱┼─────► z
        0
```

**When to use:**
- ✅ Transformer models
- ✅ NLP tasks
- ✅ When you want smooth, differentiable activation

**Code:**
```python
def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

# GPT and BERT use this!
```

---

## 🔍 Comparing Activation Functions

### Side-by-Side Comparison

| **Function** | **Range** | **Pros** | **Cons** | **Use Case** |
|--------------|-----------|----------|----------|--------------|
| **Sigmoid** | (0, 1) | Smooth, interpretable | Vanishing gradient | Output (binary) |
| **Tanh** | (-1, 1) | Zero-centered | Vanishing gradient | Hidden layers (older) |
| **ReLU** | [0, ∞) | Fast, no vanishing grad | Dying ReLU | Hidden layers (default) |
| **Leaky ReLU** | (-∞, ∞) | Fixes dying ReLU | Slightly complex | Hidden layers (alternative) |
| **Softmax** | (0, 1), sum=1 | Probabilities | Only for output | Output (multi-class) |
| **GELU** | (-∞, ∞) | Smooth, state-of-art | Slower than ReLU | Transformers (GPT) |

### Visual Comparison

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(15, 4))

plt.subplot(1, 4, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(1, 4, 2)
plt.plot(x, np.tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.subplot(1, 4, 3)
plt.plot(x, np.maximum(0, x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(1, 4, 4)
plt.plot(x, gelu(x))
plt.title('GELU')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 🧮 The Math: Why Derivatives Matter

### For Backpropagation, We Need Derivatives!

During backpropagation, we compute:
```
∂Loss/∂W = ∂Loss/∂a × ∂a/∂z × ∂z/∂W
                      ↑
              activation derivative!
```

**Example with ReLU:**

```python
# Forward pass
z = X @ W + b
a = relu(z)  # a = max(0, z)

# Backward pass (simplified)
da = gradient_from_next_layer  # Given
dz = da * relu_derivative(z)   # Need derivative!
dW = X.T @ dz
```

**Why this matters:**
- **Sigmoid/Tanh:** Derivatives → 0 for large |z| (vanishing gradient)
- **ReLU:** Derivative = 1 for z > 0 (no vanishing!)
- **Step:** No derivative (can't use backprop)

---

## 💻 Code Implementation

### Complete Activation Functions Library

```python
import numpy as np

class Activations:
    """All activation functions and their derivatives"""

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = Activations.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        return np.where(z > 0, 1.0, alpha)

    @staticmethod
    def softmax(z):
        # For numerical stability
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @staticmethod
    def gelu(z):
        return 0.5 * z * (1 + np.tanh(
            np.sqrt(2/np.pi) * (z + 0.044715 * z**3)
        ))
```

---

## ✨ Practical Examples

### Example 1: Testing on Same Input

```python
x = np.array([-2, -1, 0, 1, 2])

print("Input:", x)
print("Sigmoid:", Activations.sigmoid(x))
# [0.119, 0.269, 0.5, 0.731, 0.881]

print("Tanh:", Activations.tanh(x))
# [-0.964, -0.762, 0, 0.762, 0.964]

print("ReLU:", Activations.relu(x))
# [0, 0, 0, 1, 2]

print("Leaky ReLU:", Activations.leaky_relu(x))
# [-0.02, -0.01, 0, 1, 2]
```

### Example 2: Output Layer Selection

```python
# Binary classification (cat vs dog)
logits = np.array([[2.5]])  # Raw score
prob = Activations.sigmoid(logits)
print(f"Probability of cat: {prob[0,0]:.3f}")  # 0.924
# Decision: If prob > 0.5, predict "cat"

# Multi-class classification (digit 0-9)
logits = np.array([[2.0, 1.0, 0.1, -1.0, -2.0,
                    0.5, -0.5, 1.5, -1.5, 0.0]])
probs = Activations.softmax(logits)
print("Class probabilities:", probs[0])
# [0.254, 0.093, ..., 0.034]
# Sum = 1.0
predicted_class = np.argmax(probs)
print(f"Predicted digit: {predicted_class}")  # 0
```

---

## 🔗 Connection to GPT

### What GPT Actually Uses

```python
# GPT-2 and GPT-3 use GELU!

def gpt_feed_forward_layer(x, W1, b1, W2, b2):
    """
    This is inside every transformer block
    """
    # First linear transformation
    hidden = x @ W1 + b1

    # GELU activation (not ReLU!)
    activated = gelu(hidden)

    # Second linear transformation
    output = activated @ W2 + b2

    return output
```

**Why GELU over ReLU for GPT?**
1. Smoother (better for NLP)
2. Better performance on language tasks
3. Stochastic regularization properties

**But the concept is the same as what you're learning!**

---

## 🎓 Summary

### Key Concepts

1. **Why Activations Matter**
   - Without them: Multiple layers = single linear layer
   - With them: Can approximate any function!

2. **Choosing Activations**
   - **Hidden layers:** ReLU (default), GELU (transformers)
   - **Output (binary):** Sigmoid
   - **Output (multi-class):** Softmax
   - **Avoid:** Sigmoid/Tanh in hidden layers (vanishing gradient)

3. **For Backpropagation**
   - Need derivatives of activation functions
   - ReLU derivative is trivial: 0 or 1
   - Sigmoid/Tanh derivatives vanish for large inputs

4. **Modern Best Practices**
   - Use ReLU for most cases
   - Use GELU for transformers
   - Use Softmax for multi-class output
   - Avoid step functions (not differentiable)

### What You Built

✅ Understanding of non-linearity importance
✅ Implementation of all major activations
✅ Knowledge of when to use which
✅ Derivatives for backpropagation
✅ Connection to GPT (uses GELU!)

---

## 💡 Practice Exercises

Try these before moving on:

1. **Implement ELU** (Exponential Linear Unit)
   ```
   ELU(z) = {z           if z > 0
            {α(e^z - 1)  if z ≤ 0
   ```

2. **Visualize All Activations**
   - Plot sigmoid, tanh, relu, gelu on same graph
   - Compare their gradients

3. **Test Numerical Stability**
   - What happens with sigmoid(100)?
   - How does softmax handle it?

4. **Derivative Check**
   - Manually compute sigmoid_derivative(2.0)
   - Verify with your code

5. **Output Layer Practice**
   - Given logits [3.2, 1.1, 0.5], apply softmax
   - Verify probabilities sum to 1

---

## 🐛 Common Pitfalls

### Pitfall 1: Wrong Activation for Output

```python
# ❌ Wrong: Using ReLU for binary classification
output = relu(z)  # Can be > 1 or 0, not a probability!

# ✓ Right: Use sigmoid
output = sigmoid(z)  # Always between 0 and 1
```

### Pitfall 2: Sigmoid in Hidden Layers

```python
# ❌ Old way (vanishing gradients)
hidden = sigmoid(X @ W1 + b1)

# ✓ Modern way
hidden = relu(X @ W1 + b1)
```

### Pitfall 3: Forgetting Numerical Stability

```python
# ❌ Dangerous
softmax = np.exp(z) / np.sum(np.exp(z))
# Can overflow for large z!

# ✓ Safe
max_z = np.max(z, axis=1, keepdims=True)
softmax = np.exp(z - max_z) / np.sum(np.exp(z - max_z))
```

---

## 📚 Further Reading

### Historical Context
- **1960s:** Step function (perceptron)
- **1980s:** Sigmoid/Tanh (backpropagation era)
- **2010s:** ReLU revolution (ImageNet)
- **2018+:** GELU (transformers/GPT)

### Why This Matters
You now understand the "secret sauce" that makes deep learning work: **non-linear activation functions**!

Without them, even a 1000-layer network would be equivalent to a single linear layer. With them, neural networks can approximate virtually any function!

---

**Next Lesson:** `03_multilayer_networks.md` - Stack layers to build deep networks!

**Run:** `example_02_activations.py` to see all activations in action!

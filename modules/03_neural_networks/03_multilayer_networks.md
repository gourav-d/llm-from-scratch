# Lesson 3.3: Multi-Layer Neural Networks

## 🎯 Why This Lesson Matters for LLMs

**This is where the "deep" in "deep learning" comes from!**

### What You'll Understand After This Lesson

You've learned:
- ✅ Lesson 1: How a single neuron works (perceptron)
- ✅ Lesson 2: Why activation functions enable non-linearity

Now you'll learn:
- **How to stack multiple layers** to create deep networks
- **Why depth enables complex pattern recognition**
- **How GPT actually processes tokens** (spoiler: it's just stacked layers!)

### The Key Insight

```
Single Layer:        Can only learn simple patterns (lines)
2-3 Layers:          Can learn curved patterns (XOR, circles)
10+ Layers:          Can learn faces, language, reasoning
100+ Layers (GPT):   Can learn human-like text generation!
```

**Depth = Power**

---

## 🧠 What is a Multi-Layer Network?

### The Simple Explanation

A multi-layer network is like a **factory assembly line**:

```
Raw Materials (Input)
    ↓
Worker 1 (Layer 1) → Processes input
    ↓
Worker 2 (Layer 2) → Refines output from Layer 1
    ↓
Worker 3 (Layer 3) → Further refines
    ↓
Final Product (Output)
```

Each "worker" (layer) transforms the data, making it easier for the next layer to understand.

### Visual Representation

```
Input Layer        Hidden Layer 1      Hidden Layer 2      Output Layer
(3 neurons)        (4 neurons)         (3 neurons)         (2 neurons)

   x₁ ──┐
        ├──→ h₁₁ ──┐
   x₂ ──┤           ├──→ h₂₁ ──┐
        ├──→ h₁₂ ──┤           ├──→ y₁
   x₃ ──┤           ├──→ h₂₂ ──┤
        ├──→ h₁₃ ──┤           ├──→ y₂
        └──→ h₁₄ ──┘           │
                    └──→ h₂₃ ──┘

Notation:
- x: Input neurons
- h₁: Hidden layer 1 neurons
- h₂: Hidden layer 2 neurons
- y: Output neurons
```

**Key Points:**
- **Fully connected**: Every neuron connects to ALL neurons in next layer
- **Forward flow**: Data flows left → right only (during prediction)
- **Transformations**: Each layer transforms data into more useful representations

---

## 📐 The Math (Explained Simply!)

### For .NET Developers: Think LINQ Chaining

In C#, you might chain LINQ operations:
```csharp
var result = data
    .Select(x => Transform1(x))      // Layer 1
    .Select(x => Transform2(x))      // Layer 2
    .Select(x => Transform3(x));     // Layer 3
```

Neural networks do the same, but with matrix multiplications!

### Single Layer (Review)

One layer transforms input → output:

```
Input: x (vector)
Weights: W (matrix)
Bias: b (vector)

Output: y = activation(W·x + b)
```

### Two Layers (The Magic Begins!)

```
Layer 1:
  z₁ = W₁·x + b₁          # Linear transformation
  a₁ = ReLU(z₁)           # Activation (non-linearity)

Layer 2:
  z₂ = W₂·a₁ + b₂         # Linear transformation
  y = Softmax(z₂)         # Activation (probabilities)

Final output: y
```

**What each symbol means:**
- `x`: Input data (what you give to network)
- `W₁, b₁`: First layer's weights and biases (learnable parameters)
- `z₁`: First layer's linear output (before activation)
- `a₁`: First layer's activated output (after ReLU)
- `W₂, b₂`: Second layer's weights and biases (learnable parameters)
- `z₂`: Second layer's linear output (before activation)
- `y`: Final output (predictions)

### Three Layers (Even More Powerful!)

```
Layer 1: a₁ = ReLU(W₁·x + b₁)
Layer 2: a₂ = ReLU(W₂·a₁ + b₂)
Layer 3: y = Softmax(W₃·a₂ + b₃)
```

**Pattern:** Output of one layer becomes input to next layer!

---

## 🔍 Detailed Breakdown

### Step-by-Step: Processing One Example

Let's say we're classifying a digit (28×28 pixel image):

#### **Step 0: Prepare Input**
```
Image: 28×28 pixels = 784 pixels
Flatten: [p₁, p₂, p₃, ..., p₇₈₄]
x = vector of 784 numbers
```

#### **Step 1: First Hidden Layer (784 → 128)**
```python
# Input shape: (784,)
# W₁ shape: (128, 784)  - 128 neurons, each looking at 784 inputs
# b₁ shape: (128,)

z₁ = W₁ @ x + b₁        # Shape: (128,)
a₁ = ReLU(z₁)           # Shape: (128,)

# What happened?
# - Each of 128 neurons computed: w·x + b
# - Each result passed through ReLU
# - Now we have 128 "features" instead of 784 pixels!
```

**C# Analogy:**
```csharp
// Like mapping 784 raw values → 128 processed features
var features1 = rawPixels
    .Select(pixels => neurons.Select(n => n.Process(pixels)))
    .Select(ReLU);
```

#### **Step 2: Second Hidden Layer (128 → 64)**
```python
# Input: a₁ with shape (128,)
# W₂ shape: (64, 128)  - 64 neurons, each looking at 128 features
# b₂ shape: (64,)

z₂ = W₂ @ a₁ + b₂       # Shape: (64,)
a₂ = ReLU(z₂)           # Shape: (64,)

# What happened?
# - Each of 64 neurons processed the 128 features from layer 1
# - Now we have 64 even more refined features!
```

#### **Step 3: Output Layer (64 → 10)**
```python
# Input: a₂ with shape (64,)
# W₃ shape: (10, 64)  - 10 neurons (one per digit 0-9)
# b₃ shape: (10,)

z₃ = W₃ @ a₂ + b₃       # Shape: (10,)
y = Softmax(z₃)         # Shape: (10,) - probabilities!

# What happened?
# - Each of 10 neurons computed a score for one digit
# - Softmax converted scores → probabilities that sum to 1.0
# - Highest probability = predicted digit!
```

#### **Final Result**
```python
y = [0.01, 0.02, 0.03, 0.05, 0.68, 0.09, 0.04, 0.03, 0.02, 0.03]
#    digit:0   1     2     3     4     5     6     7     8     9

# Prediction: argmax(y) = 4 (index with highest probability)
# Confidence: 68% sure it's the digit "4"
```

---

## 📊 Shapes, Shapes, Shapes!

### The Most Important Debugging Skill

**90% of neural network bugs are shape mismatches!**

### Shape Rules for Matrix Multiplication

```
Rule: (A @ B) requires A.shape[1] == B.shape[0]

Example:
  (128, 784) @ (784,) = (128,)  ✅ Valid
  (64, 128) @ (128,) = (64,)    ✅ Valid
  (10, 64) @ (128,) = ERROR!    ❌ Shapes don't match!
```

### Shape Flow Through Network

```
Network: 784 → 128 → 64 → 10

Input x:        (784,)
W₁:            (128, 784)
b₁:            (128,)
z₁ = W₁@x+b₁:  (128,)      ✅ Matches!
a₁ = ReLU(z₁): (128,)

W₂:            (64, 128)
b₂:            (64,)
z₂ = W₂@a₁+b₂: (64,)       ✅ Matches!
a₂ = ReLU(z₂): (64,)

W₃:            (10, 64)
b₃:            (10,)
z₃ = W₃@a₂+b₃: (10,)       ✅ Matches!
y = Softmax(z₃):(10,)
```

### Pro Debugging Tip

**Always print shapes during development!**

```python
print(f"Input shape: {x.shape}")
print(f"After layer 1: {a1.shape}")
print(f"After layer 2: {a2.shape}")
print(f"Output shape: {y.shape}")
```

**In C#, think of it like:**
```csharp
Debug.Assert(input.Length == 784);
Debug.Assert(layer1Output.Length == 128);
Debug.Assert(layer2Output.Length == 64);
Debug.Assert(output.Length == 10);
```

---

## 💻 Building Your First Multi-Layer Network

### Complete Implementation (NumPy)

```python
import numpy as np

class MultiLayerNetwork:
    """
    A simple multi-layer neural network.

    Architecture: input_size → hidden1_size → hidden2_size → output_size

    For .NET developers: Like a class with methods for forward pass
    """

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        """
        Initialize network with random weights.

        Args:
            input_size: Number of input features (e.g., 784 for MNIST)
            hidden1_size: Neurons in first hidden layer (e.g., 128)
            hidden2_size: Neurons in second hidden layer (e.g., 64)
            output_size: Number of output classes (e.g., 10 for digits 0-9)
        """
        # Initialize weights with small random values
        # Why small? Large weights → exploding gradients!

        # Layer 1: input → hidden1
        self.W1 = np.random.randn(hidden1_size, input_size) * 0.01
        self.b1 = np.zeros((hidden1_size, 1))

        # Layer 2: hidden1 → hidden2
        self.W2 = np.random.randn(hidden2_size, hidden1_size) * 0.01
        self.b2 = np.zeros((hidden2_size, 1))

        # Layer 3: hidden2 → output
        self.W3 = np.random.randn(output_size, hidden2_size) * 0.01
        self.b3 = np.zeros((output_size, 1))

        # Store intermediate values (needed for backprop later!)
        self.cache = {}

    def relu(self, z):
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)

    def softmax(self, z):
        """
        Softmax: converts scores → probabilities

        Why subtract max? Numerical stability (avoids overflow)
        """
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, x):
        """
        Forward propagation: compute predictions

        Args:
            x: Input data, shape (input_size, batch_size)
               For single example: (input_size, 1)

        Returns:
            y: Predictions, shape (output_size, batch_size)
               Each column is probabilities for one example
        """
        # Layer 1: Linear → ReLU
        z1 = self.W1 @ x + self.b1      # Linear transformation
        a1 = self.relu(z1)               # Activation

        # Layer 2: Linear → ReLU
        z2 = self.W2 @ a1 + self.b2     # Linear transformation
        a2 = self.relu(z2)               # Activation

        # Layer 3: Linear → Softmax
        z3 = self.W3 @ a2 + self.b3     # Linear transformation
        y = self.softmax(z3)             # Activation (probabilities)

        # Cache for backpropagation (you'll learn this in Lesson 4!)
        self.cache = {
            'x': x, 'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2, 'z3': z3, 'y': y
        }

        return y

    def predict(self, x):
        """
        Make prediction (get class label, not probabilities)

        Args:
            x: Input data, shape (input_size, batch_size)

        Returns:
            predictions: Class labels, shape (batch_size,)
        """
        y = self.forward(x)              # Get probabilities
        return np.argmax(y, axis=0)      # Get index of max probability
```

### Using the Network

```python
# Create network: 784 inputs → 128 → 64 → 10 outputs
network = MultiLayerNetwork(
    input_size=784,      # MNIST: 28×28 pixels
    hidden1_size=128,    # First hidden layer
    hidden2_size=64,     # Second hidden layer
    output_size=10       # Digits 0-9
)

# Create fake image (all zeros)
image = np.zeros((784, 1))  # Shape: (784, 1)

# Forward pass
predictions = network.forward(image)

print(f"Predictions shape: {predictions.shape}")  # (10, 1)
print(f"Probabilities: {predictions.flatten()}")
# Output: [0.1, 0.1, 0.1, ..., 0.1] - random since weights are random!

# Get predicted class
predicted_class = network.predict(image)
print(f"Predicted digit: {predicted_class[0]}")
```

---

## 🎨 Why Multiple Layers Matter

### Single Layer vs. Multi-Layer

#### **Problem: XOR (Cannot Be Solved with Single Layer!)**

```
XOR Truth Table:
Input 1 | Input 2 | Output
--------|---------|-------
   0    |    0    |   0
   0    |    1    |   1
   1    |    0    |   1
   1    |    1    |   0
```

**Visualized:**

```
      1 |  ·(0,1)
        |
        |
Output  |
        | ·(1,0)
        |
      0 | ·(0,0)     ·(1,1)
        |________________
            0    1
              Input
```

**Problem:** You CANNOT draw a single straight line to separate the 1s from 0s!

#### **Solution: Add Hidden Layer!**

```
Input Layer    Hidden Layer    Output Layer
              (2 neurons)
   x₁ ──┐
        ├──→ h₁ ──┐
   x₂ ──┤          ├──→ y (XOR)
        └──→ h₂ ──┘

With 2 hidden neurons + non-linearity (ReLU), network can learn XOR!
```

**Why it works:**
- Hidden neurons learn intermediate features
- h₁ might learn: "Is at least one input 1?"
- h₂ might learn: "Are both inputs 1?"
- Output combines: "One but not both" = XOR!

---

## 🔗 Connection to GPT and Modern LLMs

### GPT Architecture (Simplified)

```
Token → Embedding → [Transformer Block] × 12-96 → Output
```

Each Transformer Block contains:
```
1. Multi-Head Self-Attention
2. Feed-Forward Network ← THIS IS WHAT YOU JUST LEARNED!
```

### The Feed-Forward Network in GPT

**Every single GPT layer has a feed-forward network:**

```python
# Inside each transformer block (GPT)
def feed_forward(x):
    # Layer 1: Project up (expand dimensions)
    z1 = x @ W1 + b1             # (768,) → (3072,)
    a1 = GELU(z1)                # Activation (you learned in Lesson 2!)

    # Layer 2: Project down (compress back)
    z2 = a1 @ W2 + b2            # (3072,) → (768,)
    return z2

# This is literally a 2-layer network!
# GPT-2: 768 → 3072 → 768 (in each of 12 layers)
# GPT-3: 12288 → 49152 → 12288 (in each of 96 layers!)
```

**Key Insight:** GPT is mostly just stacked multi-layer networks! (Plus attention)

### Real GPT-2 Structure

```
For EACH of 12 transformer layers:
    1. Self-Attention (learns context)
    2. Feed-Forward Network (2 layers: 768→3072→768)

Total parameters in feed-forward networks:
- Layer 1: 768 × 3072 = 2.36M parameters
- Layer 2: 3072 × 768 = 2.36M parameters
- Total per block: ~4.7M parameters
- Across 12 blocks: ~57M parameters

That's about 50% of GPT-2's total parameters!
```

---

## 🧪 Hands-On Example: XOR Problem

### Complete Code to Solve XOR

```python
import numpy as np

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # Shape: (2, 4)
Y = np.array([[0, 1, 1, 0]])                       # Shape: (1, 4)

# Build network: 2 → 4 → 1
class XORNetwork:
    def __init__(self):
        # Layer 1: 2 inputs → 4 hidden neurons
        self.W1 = np.random.randn(4, 2) * 0.5
        self.b1 = np.zeros((4, 1))

        # Layer 2: 4 hidden → 1 output
        self.W2 = np.random.randn(1, 4) * 0.5
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        # Layer 1
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Layer 2
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def train_step(self, x, y, learning_rate=0.5):
        """Simple training step (you'll learn details in Lesson 4!)"""
        # Forward pass
        predictions = self.forward(x)

        # Compute error
        error = predictions - y

        # Backpropagation (simplified - you'll learn this next!)
        dz2 = error * predictions * (1 - predictions)
        dW2 = dz2 @ self.a1.T
        db2 = np.sum(dz2, axis=1, keepdims=True)

        dz1 = (self.W2.T @ dz2) * self.a1 * (1 - self.a1)
        dW1 = dz1 @ x.T
        db1 = np.sum(dz1, axis=1, keepdims=True)

        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return np.mean(error ** 2)  # Mean squared error

# Train the network
network = XORNetwork()
for epoch in range(10000):
    loss = network.train_step(X, Y, learning_rate=0.5)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

# Test the network
print("\nFinal Predictions:")
predictions = network.forward(X)
for i in range(4):
    x1, x2 = X[:, i]
    y_true = Y[0, i]
    y_pred = predictions[0, i]
    print(f"  Input: [{x1}, {x2}] → True: {y_true}, Predicted: {y_pred:.4f}")

# Expected output:
# [0, 0] → 0 (close to 0.0)
# [0, 1] → 1 (close to 1.0)
# [1, 0] → 1 (close to 1.0)
# [1, 1] → 0 (close to 0.0)
```

---

## 🎯 Key Takeaways

### What You Learned

1. **Multi-layer networks stack transformations**
   - Each layer: linear transformation → activation
   - Output of one layer = input to next layer

2. **Depth enables complexity**
   - Single layer: Only linear separability
   - Multiple layers: Can learn XOR, curves, complex patterns
   - Many layers (deep): Can learn faces, language, reasoning

3. **Shape management is critical**
   - Matrix multiplication requires compatible shapes
   - Print shapes during debugging
   - Shape mismatches = most common bug

4. **Forward propagation is straightforward**
   - Apply W @ x + b for each layer
   - Apply activation function
   - Repeat until output layer

5. **GPT uses multi-layer networks extensively**
   - Feed-forward networks in every transformer layer
   - 50% of GPT's parameters are in these networks!
   - Architecture: 768 → 3072 → 768 (GPT-2)

### For .NET Developers

**Multi-layer networks are like:**
```csharp
// C# LINQ chaining
var result = input
    .Select(x => Layer1Transform(x))
    .Select(x => ReLU(x))
    .Select(x => Layer2Transform(x))
    .Select(x => ReLU(x))
    .Select(x => Layer3Transform(x))
    .Select(x => Softmax(x));
```

Each `.Select()` is a layer transformation!

---

## 📝 Common Questions

### Q1: How many layers should I use?

**Answer:** Depends on the problem!
- **Simple problems** (XOR, small datasets): 1-2 hidden layers
- **Image classification** (MNIST): 2-3 hidden layers
- **Complex images** (ImageNet): 10-50 layers (ResNet, VGG)
- **Language models** (GPT): 12-96 transformer layers

**Rule of thumb:** Start small, add layers if performance plateaus.

### Q2: How many neurons per layer?

**Answer:** Trial and error, but common patterns:
- **Decreasing size:** 784 → 128 → 64 → 10 (funnel shape)
- **Same size:** 100 → 100 → 100 → 10 (uniform)
- **Increasing then decreasing:** 50 → 100 → 50 → 10 (bottleneck)

**For classification:** Last layer = number of classes

### Q3: Why do we need activation functions between layers?

**Answer:** Without activation, multiple linear layers = one linear layer!

```
Without activation:
  z1 = W1 @ x
  z2 = W2 @ z1 = W2 @ (W1 @ x) = (W2 @ W1) @ x

This is equivalent to a single layer: W_combined @ x

With activation:
  z1 = W1 @ x
  a1 = ReLU(z1)  ← Non-linearity!
  z2 = W2 @ a1

Now layers actually add power!
```

### Q4: What's the difference between parameters and hyperparameters?

**Parameters** (learned during training):
- Weights (W₁, W₂, W₃)
- Biases (b₁, b₂, b₃)

**Hyperparameters** (you choose before training):
- Number of layers
- Neurons per layer
- Learning rate
- Activation functions

---

## 🚀 Practice Exercises

### Exercise 1: Build a 3-Layer Network

Create a network with this architecture:
- Input: 10 features
- Hidden 1: 20 neurons, ReLU activation
- Hidden 2: 15 neurons, ReLU activation
- Output: 5 classes, Softmax activation

```python
# Your code here
class CustomNetwork:
    def __init__(self):
        # TODO: Initialize weights and biases
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Test with random input
x = np.random.randn(10, 1)
network = CustomNetwork()
output = network.forward(x)
print(f"Output shape: {output.shape}")  # Should be (5, 1)
```

### Exercise 2: Debug Shape Mismatches

Fix the shape errors in this code:

```python
# This code has shape errors - fix them!
W1 = np.random.randn(64, 100)
b1 = np.zeros((64, 1))
W2 = np.random.randn(32, 64)
b2 = np.zeros((32,))  # ← Error here?
W3 = np.random.randn(10, 32)
b3 = np.zeros((10, 1))

x = np.random.randn(100,)  # ← Error here?

z1 = W1 @ x + b1
a1 = np.maximum(0, z1)
z2 = W2 @ a1 + b2
a2 = np.maximum(0, z2)
z3 = W3 @ a2 + b3

print(z3.shape)  # Should be (10, 1)
```

### Exercise 3: XOR Variants

Modify the XOR example to learn:
1. **AND gate:** Only true when both inputs are 1
2. **OR gate:** True when at least one input is 1
3. **NAND gate:** Opposite of AND

Hint: Change only the labels (Y), not the network architecture!

### Exercise 4: Deeper Network

Extend the XOR network to have 3 hidden layers:
- 2 → 8 → 4 → 2 → 1

Does it train faster or slower? Why?

---

## 🔬 What's Next

### In Lesson 4: Backpropagation

You'll learn:
- **How the network actually learns** (updates weights)
- **Chain rule**: Computing gradients through all layers
- **Backward pass**: The opposite of forward pass
- **Why it's called "backpropagation"**: Errors propagate backwards!

### Preview: The Missing Piece

You now know:
✅ Forward propagation (making predictions)

You'll learn next:
🔄 Backward propagation (learning from mistakes)

```
Forward:  Input → Layer 1 → Layer 2 → Output
Backward: Input ← Layer 1 ← Layer 2 ← Error

Both are needed for training!
```

---

## 📚 Additional Resources

### Visualizations
- [Playground.tensorflow.org](http://playground.tensorflow.org) - Interactive neural network visualization
- Try building a 2-layer network to classify XOR!

### Code
- See `example_03_forward_pass.py` for complete implementation
- See `exercise_03_networks.py` for practice problems

### Reading
- [But what is a neural network? | 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) - Excellent visual explanation

---

## ✨ Congratulations!

You now understand:
- ✅ How to build multi-layer neural networks
- ✅ Why depth enables complexity
- ✅ How data flows through layers (forward propagation)
- ✅ Shape management and debugging
- ✅ How GPT uses these same concepts

**Next:** Learn how networks actually **learn** in Lesson 4: Backpropagation! 🎉

---

**Ready for the next lesson?** Open `04_backpropagation.md` to learn the magic of learning!

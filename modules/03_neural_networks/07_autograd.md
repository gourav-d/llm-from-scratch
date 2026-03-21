# Lesson 7: AutoGrad - Building Automatic Differentiation from Scratch

**Understanding the Magic Behind PyTorch and TensorFlow**

---

## What You'll Learn

By the end of this lesson, you will:
- Understand how automatic differentiation works
- Build a simple autograd engine from scratch
- See how PyTorch/TensorFlow compute gradients automatically
- Implement the computational graph concept
- Build the foundation for understanding modern deep learning frameworks

**Time:** 3-4 hours

---

## Why AutoGrad?

### The Problem

In Lesson 4, you manually calculated gradients using the chain rule. For a simple 3-layer network, this was tedious but doable.

**What about GPT-3?**
- 96 layers deep
- 175 billion parameters
- Thousands of operations

**Manual backpropagation?** IMPOSSIBLE!

### The Solution: Automatic Differentiation

```python
# Manual backpropagation (what you did in Lesson 4)
def backward_pass(x, y, w1, w2, w3):
    # 50+ lines of gradient calculations...
    dw1 = ...  # Calculate manually
    dw2 = ...  # Calculate manually
    dw3 = ...  # Calculate manually
    return dw1, dw2, dw3

# AutoGrad (what PyTorch does)
output = model(x)
loss = criterion(output, y)
loss.backward()  # All gradients computed automatically!
```

**AutoGrad tracks operations and computes gradients automatically using the chain rule!**

---

## What is AutoGrad?

### Core Concept

**AutoGrad = Automatic Gradient Computation**

It builds a computational graph of all operations, then uses the chain rule to compute gradients automatically.

### Simple Example

```python
# Forward pass
x = 2.0
y = 3.0
z = x * y        # z = 6.0
output = z + 1   # output = 7.0

# We want: ∂output/∂x and ∂output/∂y
# AutoGrad computes these automatically!
```

### Computational Graph

```
      x (2.0)          y (3.0)
        \               /
         \             /
          \           /
           \         /
            \       /
             \     /
              \   /
               \ /
                *  (z = 6.0)
                |
                +1  (output = 7.0)
```

**Forward pass:** Follow arrows down (compute values)
**Backward pass:** Follow arrows up (compute gradients)

---

## Building AutoGrad from Scratch

### Step 1: The Value Object

We need an object that:
1. Stores a value
2. Tracks what operation created it
3. Stores its children (inputs to the operation)
4. Computes local gradients

```python
class Value:
    """A value in the computational graph with automatic gradient tracking"""

    def __init__(self, data, children=(), operation=''):
        self.data = data                    # The actual value
        self.grad = 0.0                     # Gradient (starts at 0)
        self._backward = lambda: None       # Function to compute gradients
        self._prev = set(children)          # Previous nodes (for backprop)
        self._op = operation                # Operation that created this value

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

**C# Analogy:**
```csharp
// This is like creating a node in a linked list that also knows:
// - Its value
// - How it was created (operation)
// - Its parent nodes
// - How to compute its gradient
class ComputationNode {
    double Data;
    double Gradient;
    Action BackwardPass;
    List<ComputationNode> Parents;
}
```

---

### Step 2: Addition Operation

```python
def __add__(self, other):
    """Implement: c = a + b"""

    # Ensure other is a Value object
    other = other if isinstance(other, Value) else Value(other)

    # Forward pass: compute the result
    result = Value(self.data + other.data, children=(self, other), operation='+')

    # Backward pass: define how to compute gradients
    def _backward():
        # Gradient of addition:
        # ∂c/∂a = 1, ∂c/∂b = 1
        # Chain rule: multiply by gradient from next layer
        self.grad += result.grad * 1.0
        other.grad += result.grad * 1.0

    result._backward = _backward
    return result
```

**Why `+=` instead of `=`?**
- A value might be used multiple times
- Gradients accumulate (chain rule adds them)

**Example:**
```python
a = Value(2.0)
b = Value(3.0)
c = a + b        # c.data = 5.0
d = a + b        # d.data = 5.0
# 'a' appears in TWO operations!
# When backprop: a.grad = ∂c/∂a + ∂d/∂a
```

---

### Step 3: Multiplication Operation

```python
def __mul__(self, other):
    """Implement: c = a * b"""

    other = other if isinstance(other, Value) else Value(other)

    # Forward pass
    result = Value(self.data * other.data, children=(self, other), operation='*')

    # Backward pass
    def _backward():
        # Gradient of multiplication:
        # ∂c/∂a = b, ∂c/∂b = a
        self.grad += result.grad * other.data
        other.grad += result.grad * self.data

    result._backward = _backward
    return result
```

**Math refresher:**
```
If c = a * b, then:
∂c/∂a = b
∂c/∂b = a

Example: c = 2 * 3 = 6
∂c/∂a = 3  (if 'a' changes by 1, 'c' changes by 3)
∂c/∂b = 2  (if 'b' changes by 1, 'c' changes by 2)
```

---

### Step 4: Power Operation (for ReLU, etc.)

```python
def __pow__(self, power):
    """Implement: c = a ** n"""

    assert isinstance(power, (int, float)), "Power must be a number"

    # Forward pass
    result = Value(self.data ** power, children=(self,), operation=f'**{power}')

    # Backward pass
    def _backward():
        # Derivative of x^n = n * x^(n-1)
        self.grad += result.grad * (power * self.data ** (power - 1))

    result._backward = _backward
    return result
```

---

### Step 5: ReLU Activation

```python
def relu(self):
    """Implement: ReLU(x) = max(0, x)"""

    # Forward pass
    result = Value(max(0, self.data), children=(self,), operation='ReLU')

    # Backward pass
    def _backward():
        # ReLU derivative:
        # 1 if x > 0, else 0
        self.grad += result.grad * (self.data > 0)

    result._backward = _backward
    return result
```

**ReLU Gradient:**
```
ReLU(x) = max(0, x)

Derivative:
∂ReLU/∂x = 1 if x > 0
           0 if x ≤ 0

Intuition:
- If input is positive, gradient flows through
- If input is negative, gradient is blocked
```

---

### Step 6: The Backward Pass (Backpropagation)

Now we need to traverse the graph backward and compute all gradients!

```python
def backward(self):
    """Compute gradients for all values in the computational graph"""

    # Step 1: Build topological sort of the graph
    # (Process nodes in correct order for backprop)
    topo_order = []
    visited = set()

    def build_topo(node):
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                build_topo(child)
            topo_order.append(node)

    build_topo(self)

    # Step 2: Initialize gradient of output to 1
    self.grad = 1.0

    # Step 3: Backpropagate through the graph
    for node in reversed(topo_order):
        node._backward()
```

**What's Topological Sort?**

```
Example graph:
    a    b
     \  /
      *   c
       \ /
        +
        |
        d

Topological order: [a, b, c, *, +, d]
Reverse (for backprop): [d, +, *, c, b, a]

We process from output to inputs!
```

**C# Analogy:**
```csharp
// Like doing a post-order traversal of a tree
// In C#, you might use a Stack for depth-first search
void BackwardPass(Node output) {
    var stack = new Stack<Node>();
    var visited = new HashSet<Node>();

    // Build processing order
    void BuildOrder(Node node) {
        if (!visited.Contains(node)) {
            visited.Add(node);
            foreach (var child in node.Children)
                BuildOrder(child);
            stack.Push(node);
        }
    }

    BuildOrder(output);

    // Process in reverse order
    while (stack.Count > 0) {
        var node = stack.Pop();
        node.ComputeLocalGradients();
    }
}
```

---

## Complete AutoGrad Engine

Here's the complete implementation:

```python
"""
AutoGrad Engine - Automatic Differentiation from Scratch
Inspired by Andrej Karpathy's micrograd
"""

class Value:
    """A scalar value with automatic gradient tracking"""

    def __init__(self, data, children=(), operation=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = operation

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += result.grad
            other.grad += result.grad
        result._backward = _backward

        return result

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        result._backward = _backward

        return result

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        result = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * result.grad
        result._backward = _backward

        return result

    def __truediv__(self, other):
        # Division: a / b = a * b^(-1)
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self**-1

    def relu(self):
        result = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (result.data > 0) * result.grad
        result._backward = _backward

        return result

    def tanh(self):
        import math
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        result = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * result.grad
        result._backward = _backward

        return result

    def exp(self):
        import math
        x = self.data
        result = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += result.data * result.grad
        result._backward = _backward

        return result

    def backward(self):
        """Compute gradients using backpropagation"""

        # Build topological order
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Backpropagate
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

---

## Example Usage

### Example 1: Simple Computation

```python
# Create values
a = Value(2.0)
b = Value(3.0)

# Forward pass
c = a * b        # c = 6.0
d = c + 1.0      # d = 7.0

# Backward pass
d.backward()

print(f"a.data = {a.data}, a.grad = {a.grad}")  # a.grad = 3.0
print(f"b.data = {b.data}, b.grad = {b.grad}")  # b.grad = 2.0
```

**Verification:**
```
d = (a * b) + 1
∂d/∂a = b = 3.0 ✓
∂d/∂b = a = 2.0 ✓
```

---

### Example 2: Non-linear Function

```python
# f(x) = x^2 + 2*x + 1
x = Value(3.0)
y = x**2 + 2*x + 1

y.backward()

print(f"x = {x.data}, grad = {x.grad}")  # grad should be 2*x + 2 = 8.0
```

**Manual calculation:**
```
f(x) = x^2 + 2x + 1
f'(x) = 2x + 2
f'(3) = 2*3 + 2 = 8.0 ✓
```

---

### Example 3: Neural Network Neuron

```python
# Neuron: output = ReLU(w1*x1 + w2*x2 + b)
w1 = Value(0.5)
w2 = Value(1.0)
b = Value(1.5)

x1 = Value(2.0)
x2 = Value(3.0)

# Forward pass
z = w1*x1 + w2*x2 + b  # z = 0.5*2 + 1.0*3 + 1.5 = 5.5
output = z.relu()       # output = 5.5 (since z > 0)

# Backward pass
output.backward()

print(f"w1.grad = {w1.grad}")  # Should be x1 = 2.0
print(f"w2.grad = {w2.grad}")  # Should be x2 = 3.0
print(f"b.grad = {b.grad}")    # Should be 1.0
```

**This is exactly what happens in PyTorch!**

---

## Connection to PyTorch

### Our AutoGrad vs PyTorch

```python
# Our implementation
a = Value(2.0)
b = Value(3.0)
c = a * b + 1
c.backward()
print(a.grad, b.grad)  # 3.0, 2.0

# PyTorch (same logic!)
import torch
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = a * b + 1
c.backward()
print(a.grad, b.grad)  # tensor(3.0), tensor(2.0)
```

**PyTorch does EXACTLY what our autograd engine does:**
1. Builds computational graph during forward pass
2. Traverses graph backward during `.backward()`
3. Computes gradients using chain rule

**Differences:**
- PyTorch works with tensors (multi-dimensional arrays)
- PyTorch is optimized in C++/CUDA for speed
- PyTorch handles batches automatically
- But the CORE CONCEPT is identical!

---

## Building a Multi-Layer Network with AutoGrad

```python
class Neuron:
    """A single neuron with AutoGrad support"""

    def __init__(self, num_inputs):
        import random
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Forward pass: w·x + b
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return activation.relu()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A layer of neurons"""

    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """Multi-Layer Perceptron"""

    def __init__(self, num_inputs, layer_sizes):
        sizes = [num_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# Create a network: 3 inputs → 4 hidden → 1 output
model = MLP(3, [4, 1])

# Example input
x = [Value(2.0), Value(3.0), Value(-1.0)]

# Forward pass
output = model(x)

# Target
target = Value(1.0)

# Loss
loss = (output - target)**2

# Backward pass
loss.backward()

# Update weights (gradient descent)
learning_rate = 0.01
for p in model.parameters():
    p.data -= learning_rate * p.grad
```

**This is a complete neural network with automatic differentiation!**

---

## Comparison: Manual vs AutoGrad

### Manual Backprop (Lesson 4)

```python
# Forward
z1 = x @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
output = softmax(z2)

# Backward (MANUAL - tedious!)
dz2 = output - y
dW2 = a1.T @ dz2
db2 = np.sum(dz2, axis=0)

da1 = dz2 @ W2.T
dz1 = da1 * relu_derivative(z1)
dW1 = x.T @ dz1
db1 = np.sum(dz1, axis=0)

# Update
W1 -= learning_rate * dW1
W2 -= learning_rate * dW2
```

### With AutoGrad

```python
# Forward (same)
z1 = x @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
output = softmax(z2)
loss = cross_entropy(output, y)

# Backward (AUTOMATIC!)
loss.backward()  # All gradients computed!

# Update
for param in model.parameters():
    param.data -= learning_rate * param.grad
```

**AutoGrad saves you from:**
- Manual gradient calculations
- Chain rule errors
- Forgetting to update a parameter
- Debugging gradient computation

---

## How GPT Uses AutoGrad

```python
# Simplified GPT training with PyTorch

# Forward pass
embeddings = embedding_layer(tokens)           # AutoGrad tracking!
for layer in transformer_layers:
    attention = multi_head_attention(embeddings)  # AutoGrad tracking!
    feedforward = mlp(attention)                  # AutoGrad tracking!
    embeddings = layer_norm(feedforward)          # AutoGrad tracking!

logits = output_projection(embeddings)         # AutoGrad tracking!
loss = cross_entropy(logits, target_tokens)    # AutoGrad tracking!

# Backward pass
loss.backward()  # Computes gradients for BILLIONS of parameters!

# Update
optimizer.step()  # Updates all parameters using their gradients
```

**Without AutoGrad:**
- Manually computing gradients for 175 billion parameters? IMPOSSIBLE!
- AutoGrad makes training GPT-scale models feasible

---

## Exercises

### Exercise 1: Verify Gradients

```python
# Implement and verify gradients for:
# f(x, y) = x^2 + y^2 + 2xy
# ∂f/∂x = 2x + 2y
# ∂f/∂y = 2y + 2x

x = Value(3.0)
y = Value(4.0)

# Implement f(x, y)
f = # YOUR CODE HERE

f.backward()

# Verify: at x=3, y=4
# ∂f/∂x should be 2*3 + 2*4 = 14
# ∂f/∂y should be 2*4 + 2*3 = 14

assert abs(x.grad - 14.0) < 1e-5, f"x.grad = {x.grad}, expected 14.0"
assert abs(y.grad - 14.0) < 1e-5, f"y.grad = {y.grad}, expected 14.0"
```

### Exercise 2: Sigmoid Activation

```python
# Implement sigmoid activation
# sigmoid(x) = 1 / (1 + e^(-x))
# derivative: sigmoid(x) * (1 - sigmoid(x))

def sigmoid(self):
    # YOUR CODE HERE
    # Hint: use self.exp()
    pass

Value.sigmoid = sigmoid

# Test
x = Value(0.0)
s = x.sigmoid()
s.backward()

# At x=0: sigmoid(0) = 0.5
# derivative = 0.5 * 0.5 = 0.25
print(f"sigmoid(0) = {s.data}")  # Should be 0.5
print(f"gradient = {x.grad}")     # Should be 0.25
```

### Exercise 3: Train XOR

```python
# Use the MLP class to train XOR function
# XOR truth table:
# [0, 0] → 0
# [0, 1] → 1
# [1, 0] → 1
# [1, 1] → 0

# Create model
model = MLP(2, [4, 1])  # 2 inputs, 4 hidden, 1 output

# Training data
X = [
    [Value(0.0), Value(0.0)],
    [Value(0.0), Value(1.0)],
    [Value(1.0), Value(0.0)],
    [Value(1.0), Value(1.0)],
]
y = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

# YOUR CODE: Train for 100 iterations
# Hint:
# 1. Forward pass for all examples
# 2. Compute loss (MSE)
# 3. Backward
# 4. Update weights
# 5. Zero gradients

for iteration in range(100):
    # YOUR CODE HERE
    pass
```

---

## Key Takeaways

1. **AutoGrad builds a computational graph** during the forward pass
2. **Each operation stores how to compute its gradient** (local derivative)
3. **Backward pass traverses the graph in reverse** using topological sort
4. **Chain rule is applied automatically** by multiplying local gradients
5. **PyTorch/TensorFlow use the same concept** for tensors instead of scalars

**You've just built the core of PyTorch!**

---

## Connection to Module 4 (Transformers)

When you build transformers:
- Attention mechanism: 30+ operations
- Layer normalization: 10+ operations
- Feed-forward network: 20+ operations

**Manual gradients?** Would take days to implement and debug!

**With AutoGrad?** Just write the forward pass, `.backward()` does the rest!

---

## Further Reading

1. **Andrej Karpathy's micrograd**: https://github.com/karpathy/micrograd
2. **PyTorch Autograd**: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
3. **Automatic Differentiation in ML**: Research paper
4. **Computational Graphs**: Deep Learning Book Chapter 6

---

## Summary

You now understand:
- ✅ How automatic differentiation works
- ✅ Why PyTorch can compute gradients automatically
- ✅ The computational graph concept
- ✅ How to implement autograd from scratch
- ✅ The foundation of modern deep learning frameworks

**Next:** Use this knowledge when learning PyTorch (Module 3.5)!

---

**Congratulations! You've demystified the "magic" behind modern deep learning frameworks!**

**This is the same technique used to train GPT, BERT, and all modern neural networks!**

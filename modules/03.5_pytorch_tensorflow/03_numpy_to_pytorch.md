# Lesson 3: Converting NumPy to PyTorch

**Migrate Your Module 3 Projects to PyTorch**

---

## Learning Objectives

By the end of this lesson, you will:
- Convert NumPy neural networks to PyTorch
- Compare performance between NumPy and PyTorch
- Understand when to use each framework
- Successfully migrate your Module 3 projects

**Time:** 3-4 hours

---

## Why Convert?

### What You Gain

```
NumPy Implementation:
✅ Understanding of fundamentals
✅ Complete control
❌ Manual gradients (error-prone)
❌ No GPU support
❌ Slower for large models

PyTorch Implementation:
✅ Automatic gradients (reliable)
✅ GPU acceleration (10-100x faster)
✅ Production-ready
✅ Ecosystem (pre-trained models, etc.)
❌ Abstract (harder to debug internals)
```

**Best of both worlds:** Learn with NumPy, build with PyTorch!

---

## Part 1: Side-by-Side Comparison

### Single Neuron (Perceptron)

#### NumPy Version (Module 3)

```python
import numpy as np

class NumpyPerceptron:
    def __init__(self, input_size):
        # Manual initialization
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0

    def forward(self, x):
        # Manual forward pass
        return np.dot(x, self.weights) + self.bias

    def backward(self, x, error):
        # Manual gradients
        grad_w = x * error
        grad_b = error
        return grad_w, grad_b

    def update(self, grad_w, grad_b, lr=0.01):
        # Manual weight update
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b

# Training
perceptron = NumpyPerceptron(input_size=2)
x = np.array([1.0, 2.0])
y_true = 3.0

for epoch in range(100):
    # Forward
    y_pred = perceptron.forward(x)
    error = y_pred - y_true

    # Backward
    grad_w, grad_b = perceptron.backward(x, error)

    # Update
    perceptron.update(grad_w, grad_b)
```

---

#### PyTorch Version (Module 3.5)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchPerceptron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Automatic initialization
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # Automatic forward pass
        return self.linear(x)

# Training
perceptron = PyTorchPerceptron(input_size=2)
optimizer = optim.SGD(perceptron.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.tensor([1.0, 2.0])
y_true = torch.tensor([3.0])

for epoch in range(100):
    # Forward
    y_pred = perceptron(x)
    loss = loss_fn(y_pred, y_true)

    # Backward (automatic!)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Line Count:**
- NumPy: ~30 lines (with backward pass)
- PyTorch: ~15 lines (backward is automatic)

**50% less code!**

---

### Comparison Table

| Aspect | NumPy | PyTorch |
|--------|-------|---------|
| **Initialization** | Manual | Automatic |
| **Forward Pass** | Manual | Automatic |
| **Backward Pass** | Manual (error-prone) | Automatic (reliable) |
| **Weight Update** | Manual | Optimizer handles it |
| **GPU Support** | ❌ None | ✅ Easy |
| **Debugging** | ✅ Full control | ⚠️ Less transparent |
| **Performance** | Good for small data | ✅ Much faster |
| **Code Length** | Longer | Shorter |

---

## Part 2: Multi-Layer Network

### NumPy MLP (Module 3)

```python
class NumpyMLP:
    def __init__(self):
        # Layer 1: 784 → 128
        self.W1 = np.random.randn(784, 128) * 0.01
        self.b1 = np.zeros(128)

        # Layer 2: 128 → 10
        self.W2 = np.random.randn(128, 10) * 0.01
        self.b2 = np.zeros(10)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        # Layer 1
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def backward(self, x, y):
        batch_size = x.shape[0]

        # Layer 2 gradients
        dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / batch_size
        db2 = np.sum(dz2, axis=0) / batch_size

        # Layer 1 gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (x.T @ dz1) / batch_size
        db1 = np.sum(dz1, axis=0) / batch_size

        return dW1, db1, dW2, db2

    def update(self, grads, lr=0.01):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

# Training
model = NumpyMLP()
for epoch in range(10):
    y_pred = model.forward(x_train)
    grads = model.backward(x_train, y_train)
    model.update(grads, lr=0.01)
```

**~60 lines of code, manual gradient calculations**

---

### PyTorch MLP (Module 3.5)

```python
class PyTorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Training
model = PyTorchMLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
```

**~15 lines of code, automatic gradients**

**75% less code!**

---

## Part 3: Step-by-Step Conversion Guide

### Step 1: Convert Data to Tensors

```python
# NumPy arrays
x_train_np = np.random.randn(1000, 784)
y_train_np = np.random.randint(0, 10, 1000)

# Convert to PyTorch tensors
x_train = torch.from_numpy(x_train_np).float()
y_train = torch.from_numpy(y_train_np).long()

# Or create directly
x_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))
```

**For .NET Devs:**
```csharp
// C# - Similar to type conversion
double[] array = new double[100];
float[] floatArray = Array.ConvertAll(array, x => (float)x);
```

---

### Step 2: Convert Model Architecture

```python
# NumPy: Manual weights
class NumpyModel:
    def __init__(self):
        self.W1 = np.random.randn(10, 5) * 0.01
        self.b1 = np.zeros(5)

# PyTorch: Use nn.Linear
class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
```

**Mapping:**
```python
# NumPy                      # PyTorch
W = np.random.randn(10, 5)  # nn.Linear(10, 5)
b = np.zeros(5)             # (bias included in Linear)
```

---

### Step 3: Convert Forward Pass

```python
# NumPy: Manual operations
def forward_numpy(x):
    z1 = x @ W1 + b1
    a1 = np.maximum(0, z1)  # ReLU
    z2 = a1 @ W2 + b2
    return softmax(z2)

# PyTorch: Use built-in functions
def forward(self, x):
    x = F.relu(self.layer1(x))
    x = self.layer2(x)
    return x
```

**Activation Mapping:**
```python
# NumPy                  # PyTorch
np.maximum(0, x)        # F.relu(x)
1 / (1 + np.exp(-x))    # torch.sigmoid(x)
# (softmax)             # F.softmax(x, dim=1)
```

---

### Step 4: Replace Backward Pass

```python
# NumPy: Manual gradients (DELETE ALL OF THIS!)
def backward(self, x, y):
    # Layer 2
    dz2 = self.a2 - y
    dW2 = self.a1.T @ dz2
    db2 = np.sum(dz2, axis=0)
    # Layer 1
    da1 = dz2 @ self.W2.T
    dz1 = da1 * (self.z1 > 0)
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)
    # ...
```

```python
# PyTorch: Automatic gradients (ONE LINE!)
loss.backward()
```

**Delete ~20-30 lines, replace with 1 line!**

---

### Step 5: Replace Weight Updates

```python
# NumPy: Manual updates (DELETE THIS!)
def update(self, lr=0.01):
    self.W1 -= lr * self.dW1
    self.b1 -= lr * self.db1
    self.W2 -= lr * self.dW2
    self.b2 -= lr * self.db2
```

```python
# PyTorch: Use optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer.step()
```

---

### Step 6: Convert Training Loop

```python
# NumPy
for epoch in range(epochs):
    # Forward
    y_pred = model.forward(x_train)

    # Compute loss
    loss = cross_entropy(y_pred, y_train)

    # Backward
    grads = model.backward(x_train, y_train)

    # Update
    model.update(grads, lr=0.01)

# PyTorch
for epoch in range(epochs):
    # Forward
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    # Backward (automatic!)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Much cleaner!**

---

## Part 4: Complete Conversion Example

### Original NumPy Code (Module 3)

```python
import numpy as np

class NumpyNetwork:
    def __init__(self):
        self.W1 = np.random.randn(2, 4) * 0.5
        self.b1 = np.zeros(4)
        self.W2 = np.random.randn(4, 1) * 0.5
        self.b2 = np.zeros(1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, x, y):
        # Output layer
        self.dz2 = self.a2 - y
        self.dW2 = self.a1.T @ self.dz2
        self.db2 = np.sum(self.dz2, axis=0)

        # Hidden layer
        self.da1 = self.dz2 @ self.W2.T
        self.dz1 = self.da1 * self.sigmoid_derivative(self.z1)
        self.dW1 = x.T @ self.dz1
        self.db1 = np.sum(self.dz1, axis=0)

    def update(self, lr=0.1):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

# Training
model = NumpyNetwork()
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

for epoch in range(10000):
    y_pred = model.forward(x_train)
    model.backward(x_train, y_train)
    model.update(lr=0.5)
```

**~50 lines**

---

### Converted PyTorch Code (Module 3.5)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

# Training
model = PyTorchNetwork()
optimizer = optim.SGD(model.parameters(), lr=0.5)
loss_fn = nn.MSELoss()

x_train = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_train = torch.tensor([[0.], [1.], [1.], [0.]])

for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
```

**~20 lines (60% reduction!)**

---

## Part 5: Performance Comparison

### Benchmarking NumPy vs PyTorch

```python
import time
import numpy as np
import torch

# Test parameters
input_size = 784
hidden_size = 256
output_size = 10
batch_size = 128
iterations = 100

# NumPy version
def benchmark_numpy():
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    x = np.random.randn(batch_size, input_size)

    start = time.time()
    for _ in range(iterations):
        # Forward
        z1 = x @ W1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2
        a2 = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)

        # Backward (simplified)
        grad = a2
        grad = grad @ W2.T

    return time.time() - start

# PyTorch CPU version
def benchmark_pytorch_cpu():
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )
    x = torch.randn(batch_size, input_size)

    start = time.time()
    for _ in range(iterations):
        output = model(x)
        output.sum().backward()
        model.zero_grad()

    return time.time() - start

# PyTorch GPU version
def benchmark_pytorch_gpu():
    if not torch.cuda.is_available():
        return None

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    ).cuda()
    x = torch.randn(batch_size, input_size).cuda()

    # Warm-up
    for _ in range(10):
        output = model(x)
        output.sum().backward()
        model.zero_grad()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        output = model(x)
        output.sum().backward()
        model.zero_grad()

    torch.cuda.synchronize()
    return time.time() - start

# Run benchmarks
print("Performance Comparison")
print("=" * 50)

numpy_time = benchmark_numpy()
print(f"NumPy:           {numpy_time:.4f} seconds")

pytorch_cpu_time = benchmark_pytorch_cpu()
print(f"PyTorch (CPU):   {pytorch_cpu_time:.4f} seconds")
print(f"Speedup vs NumPy: {numpy_time / pytorch_cpu_time:.2f}x")

pytorch_gpu_time = benchmark_pytorch_gpu()
if pytorch_gpu_time:
    print(f"PyTorch (GPU):   {pytorch_gpu_time:.4f} seconds")
    print(f"Speedup vs NumPy: {numpy_time / pytorch_gpu_time:.2f}x")
    print(f"Speedup vs CPU:   {pytorch_cpu_time / pytorch_gpu_time:.2f}x")
```

**Expected Results:**
```
Performance Comparison
==================================================
NumPy:           2.5000 seconds
PyTorch (CPU):   2.2000 seconds
Speedup vs NumPy: 1.14x

PyTorch (GPU):   0.0500 seconds
Speedup vs NumPy: 50.00x
Speedup vs CPU:   44.00x
```

**GPU is ~50x faster!**

---

## Part 6: When to Use Each

### Use NumPy When:

✅ Learning fundamentals
✅ Small experiments (< 1000 parameters)
✅ No GPU available
✅ Debugging gradient calculations
✅ Integration with non-DL code

### Use PyTorch When:

✅ Building real models (> 10K parameters)
✅ Training on large datasets
✅ GPU acceleration needed
✅ Production deployment
✅ Using pre-trained models
✅ Research and prototyping

### Best Practice:

```
Learn:     NumPy (understand the math)
           ↓
Prototype: PyTorch (fast experimentation)
           ↓
Deploy:    PyTorch (production-ready)
```

---

## Part 7: Common Conversion Patterns

### Pattern 1: Weight Initialization

```python
# NumPy
W = np.random.randn(10, 5) * 0.01
b = np.zeros(5)

# PyTorch
layer = nn.Linear(10, 5)
# Default initialization is often better!
# Or custom:
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)
```

---

### Pattern 2: Activation Functions

```python
# NumPy → PyTorch
np.maximum(0, x)                    # F.relu(x)
1 / (1 + np.exp(-x))               # torch.sigmoid(x)
np.tanh(x)                         # torch.tanh(x)
# softmax                          # F.softmax(x, dim=1)
```

---

### Pattern 3: Loss Functions

```python
# NumPy: Mean Squared Error
loss = np.mean((y_pred - y_true) ** 2)

# PyTorch
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_true)

# NumPy: Cross-Entropy (manual)
loss = -np.mean(y_true * np.log(y_pred + 1e-8))

# PyTorch (includes softmax!)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)
```

---

### Pattern 4: Gradient Descent

```python
# NumPy: Manual update
for param in [W1, b1, W2, b2]:
    param -= lr * param.grad

# PyTorch: Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.step()
```

---

## Summary

### Conversion Checklist

**Data:**
- [ ] Convert NumPy arrays to PyTorch tensors
- [ ] Ensure correct dtypes (float32 for inputs, long for labels)

**Model:**
- [ ] Replace manual weights with `nn.Linear`
- [ ] Define `__init__` and `forward` methods
- [ ] Use built-in activations (`F.relu`, etc.)

**Training:**
- [ ] Delete backward pass code
- [ ] Use optimizer (`optim.Adam`, etc.)
- [ ] Use loss function (`nn.CrossEntropyLoss`, etc.)
- [ ] Add `optimizer.zero_grad()` before backward

**Evaluation:**
- [ ] Use `model.eval()` and `torch.no_grad()`
- [ ] Convert outputs for metrics if needed

---

### Key Differences

| Concept | NumPy | PyTorch |
|---------|-------|---------|
| Arrays | `np.array` | `torch.tensor` |
| Weights | Manual | `nn.Linear` |
| Forward | Manual ops | Define `forward()` |
| Backward | Manual gradients | `loss.backward()` |
| Update | Manual | `optimizer.step()` |
| GPU | ❌ | `.to('cuda')` |

---

## Quiz

### Question 1
What's the main benefit of converting from NumPy to PyTorch?

<details>
<summary>Answer</summary>

**Automatic differentiation**: No need to manually calculate gradients (error-prone, time-consuming).

Also: GPU acceleration, easier to maintain, production-ready.

</details>

### Question 2
Can you mix NumPy and PyTorch in the same code?

<details>
<summary>Answer</summary>

Yes, but be careful:
```python
# NumPy → PyTorch
torch_tensor = torch.from_numpy(numpy_array)

# PyTorch → NumPy (CPU only!)
numpy_array = torch_tensor.cpu().numpy()
```

GPU tensors must be moved to CPU first!

</details>

### Question 3
Do you still need to understand NumPy after learning PyTorch?

<details>
<summary>Answer</summary>

**Yes!**

1. Data preprocessing often uses NumPy
2. Helps debug when PyTorch behaves unexpectedly
3. Understanding the math makes you better at deep learning
4. Many libraries (scikit-learn, pandas) use NumPy

PyTorch extends NumPy, doesn't replace it.

</details>

---

## Lab Exercise: Convert Your Module 3 Project

### Task

Pick ONE of your Module 3 projects and convert it to PyTorch:

**Options:**
1. Perceptron (simple)
2. MLP for XOR (medium)
3. MNIST classifier (challenging)

### Steps

1. **Setup**
   ```python
   import torch
   import torch.nn as nn
   import torch.optim as optim
   ```

2. **Convert model to nn.Module**
   - Define `__init__` with layers
   - Define `forward` pass

3. **Convert training loop**
   - Use optimizer
   - Use loss function
   - Delete backward pass code

4. **Test**
   - Verify same accuracy
   - Benchmark performance

5. **Compare**
   - Count lines of code
   - Measure training time
   - Document findings

### Example: MNIST Conversion

<details>
<summary>Full Solution</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Convert your NumPy model to this:
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f}')
```

**Result:** ~96% accuracy, much faster than NumPy!

</details>

---

## Next Steps

**Congratulations!** You can now:
✅ Convert NumPy neural networks to PyTorch
✅ Understand performance trade-offs
✅ Choose the right tool for the job

**Next Lesson:** TensorFlow & Keras Basics (`04_tensorflow_basics.md`)

You'll learn:
- TensorFlow fundamentals
- Keras Sequential and Functional APIs
- Comparing PyTorch vs TensorFlow
- When to use each framework

---

**Practice converting your Module 3 projects before moving on!**

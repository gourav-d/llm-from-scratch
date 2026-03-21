# Lesson 2: Building Neural Networks in PyTorch

**From Manual Implementation to nn.Module**

---

## Learning Objectives

By the end of this lesson, you will:
- Understand `nn.Module` - the base class for all models
- Use built-in layers (`nn.Linear`, activation functions)
- Build complete neural networks
- Implement training loops
- Use optimizers and loss functions
- Train a real MNIST classifier

**Time:** 5-7 hours

---

## Recap: Module 3 vs Module 3.5

### What You Built in Module 3 (NumPy)

```python
# Module 3 - Manual everything
class NeuralNetwork:
    def __init__(self):
        # Manual weight initialization
        self.W1 = np.random.randn(784, 128) * 0.01
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, 10) * 0.01
        self.b2 = np.zeros(10)

    def forward(self, x):
        # Manual forward pass
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.softmax(self.z2)

    def backward(self, x, y):
        # Manual gradients (30+ lines!)
        # ...
```

### What You'll Build in Module 3.5 (PyTorch)

```python
# Module 3.5 - Automatic everything!
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Automatic weight initialization
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        # Forward pass only (backward is automatic!)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Training
model = NeuralNetwork()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# Automatic gradients!
output = model(x)
loss = loss_fn(output, y)
loss.backward()  # ← This replaces 30+ lines!
optimizer.step()
```

**Same network, 10x less code!**

---

## Part 1: Understanding `nn.Module`

### What is nn.Module?

`nn.Module` is the base class for all neural network models in PyTorch.

**For .NET Developers:**
```csharp
// C# - Base class pattern
public class MyClass : BaseClass
{
    public MyClass() : base() { }
}

// Python - Same concept
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # Call parent constructor
```

### Basic nn.Module Template

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        # Step 1: Call parent class constructor
        super().__init__()

        # Step 2: Define layers
        self.layer1 = nn.Linear(10, 5)  # Input: 10, Output: 5
        self.layer2 = nn.Linear(5, 1)   # Input: 5, Output: 1

    def forward(self, x):
        # Step 3: Define forward pass
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
```

**Line-by-line:**
1. `class MyModel(nn.Module):` - Inherit from nn.Module
2. `super().__init__()` - Initialize parent class (REQUIRED!)
3. `self.layer1 = nn.Linear(...)` - Define layers
4. `def forward(self, x):` - Define how data flows through network
5. Return final output

**Why `super().__init__()`?** Registers layers so PyTorch can track parameters.

---

### Creating Your First Model

```python
import torch
import torch.nn as nn

# Define model
class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)  # 2 inputs, 1 output

    def forward(self, x):
        return self.layer(x)

# Create instance
model = SimpleNetwork()
print(model)
# Output:
# SimpleNetwork(
#   (layer): Linear(in_features=2, out_features=1, bias=True)
# )

# Create input
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2 samples, 2 features

# Forward pass
output = model(x)  # Calls forward() automatically!
print(output)
# tensor([[...],  # Predicted value for sample 1
#         [...]], # Predicted value for sample 2
#        grad_fn=<AddmmBackward0>)
```

**Key Points:**
- Calling `model(x)` automatically calls `forward(x)`
- Output has `grad_fn` - PyTorch tracked operations for gradients!

---

## Part 2: Built-in Layers

### nn.Linear - Fully Connected Layer

**What it does:** `output = input @ weights + bias`

```python
# Create linear layer
layer = nn.Linear(in_features=10, out_features=5)

# What's inside?
print(f"Weights shape: {layer.weight.shape}")  # [5, 10]
print(f"Bias shape: {layer.bias.shape}")       # [5]

# Forward pass
x = torch.randn(3, 10)  # 3 samples, 10 features
output = layer(x)
print(f"Output shape: {output.shape}")  # [3, 5]
```

**Equivalent NumPy (Module 3):**
```python
# NumPy version
W = np.random.randn(10, 5)
b = np.zeros(5)
output = x @ W + b

# PyTorch version
layer = nn.Linear(10, 5)
output = layer(x)  # Does the same thing!
```

**For .NET Devs:** Think of `nn.Linear` like a function that transforms data:
```csharp
// C# - Function transform
public double[] Transform(double[] input)
{
    // Multiply by weights, add bias
    return Multiply(input, weights) + bias;
}
```

---

### Activation Functions

#### ReLU (Rectified Linear Unit)

```python
import torch.nn.functional as F

# ReLU: f(x) = max(0, x)
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Option 1: Functional
output = F.relu(x)
print(output)  # tensor([0., 0., 0., 1., 2.])

# Option 2: Module
relu = nn.ReLU()
output = relu(x)
print(output)  # tensor([0., 0., 0., 1., 2.])
```

**Visual:**
```
Input:  [-2, -1,  0,  1,  2]
         ↓   ↓   ↓   ↓   ↓
ReLU:   [ 0,  0,  0,  1,  2]
        (negative → 0, positive → unchanged)
```

**When to use:** Hidden layers (most common)

---

#### Sigmoid

```python
# Sigmoid: f(x) = 1 / (1 + e^(-x))
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
output = torch.sigmoid(x)
print(output)  # tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])

# All values between 0 and 1!
```

**When to use:** Binary classification (output layer)

---

#### Softmax

```python
# Softmax: Converts to probabilities (sum = 1)
x = torch.tensor([[1.0, 2.0, 3.0]])  # Logits

output = F.softmax(x, dim=1)  # dim=1 means along columns
print(output)
# tensor([[0.0900, 0.2447, 0.6652]])
# Sum: 0.09 + 0.24 + 0.67 = 1.0 ✓
```

**When to use:** Multi-class classification (output layer)

---

### Common Activation Functions Summary

| Function | Range | Use Case | Formula |
|----------|-------|----------|---------|
| ReLU | [0, ∞) | Hidden layers | max(0, x) |
| Sigmoid | (0, 1) | Binary output | 1/(1+e^-x) |
| Tanh | (-1, 1) | Hidden layers | (e^x - e^-x)/(e^x + e^-x) |
| Softmax | [0, 1], sum=1 | Multi-class output | e^xi / Σe^xj |

---

## Part 3: Building a Complete Network

### Multi-Layer Perceptron (MLP)

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # Define layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = F.relu(x)

        # Layer 2
        x = self.layer2(x)
        x = F.relu(x)

        # Layer 3 (output)
        x = self.layer3(x)
        # Note: No activation here (done in loss function)

        return x

# Create model
model = MLP(input_size=784, hidden_size=128, output_size=10)
print(model)
```

**Output:**
```
MLP(
  (layer1): Linear(in_features=784, out_features=128, bias=True)
  (layer2): Linear(in_features=128, out_features=128, bias=True)
  (layer3): Linear(in_features=128, out_features=10, bias=True)
)
```

---

### Alternative: nn.Sequential

For simple sequential models:

```python
# Using nn.Sequential (less code!)
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

print(model)
```

**When to use:**
- `nn.Module` - Complex architectures, custom logic
- `nn.Sequential` - Simple feed-forward networks

**For .NET Devs:**
```csharp
// C# - Fluent API pattern
var pipeline = new Pipeline()
    .AddLayer(new Linear(784, 128))
    .AddActivation(new ReLU())
    .AddLayer(new Linear(128, 10));
```

---

## Part 4: Loss Functions

### What are Loss Functions?

**Loss function** measures how wrong your predictions are.

```
Goal: Minimize loss
Better predictions = Lower loss
Worse predictions = Higher loss
```

### Cross-Entropy Loss (Classification)

```python
# For multi-class classification
loss_fn = nn.CrossEntropyLoss()

# Predictions (logits - raw scores)
predictions = torch.tensor([[2.0, 1.0, 0.1]])  # 3 classes

# True label (class index)
target = torch.tensor([0])  # Correct class is 0

# Compute loss
loss = loss_fn(predictions, target)
print(f"Loss: {loss.item()}")  # Lower is better
```

**What it does:**
1. Applies softmax to predictions
2. Computes negative log-likelihood
3. Returns single loss value

**Important:** Don't apply softmax in your model! CrossEntropyLoss does it for you.

---

### Mean Squared Error (Regression)

```python
# For regression (predicting continuous values)
loss_fn = nn.MSELoss()

# Predictions
predictions = torch.tensor([2.5, 3.8, 1.2])

# True values
targets = torch.tensor([3.0, 4.0, 1.0])

# Compute loss
loss = loss_fn(predictions, targets)
print(f"Loss: {loss.item()}")  # Mean of squared differences
```

**Formula:** MSE = mean((predictions - targets)²)

---

### Common Loss Functions

| Loss Function | Use Case | PyTorch Class |
|---------------|----------|---------------|
| Cross-Entropy | Multi-class classification | `nn.CrossEntropyLoss()` |
| Binary Cross-Entropy | Binary classification | `nn.BCELoss()` |
| MSE | Regression | `nn.MSELoss()` |
| MAE | Regression (robust) | `nn.L1Loss()` |

---

## Part 5: Optimizers

### What are Optimizers?

**Optimizer** updates weights based on gradients to minimize loss.

```
Gradients tell us: "Move weights this direction"
Optimizer decides: "How far to move"
```

### Stochastic Gradient Descent (SGD)

```python
import torch.optim as optim

# Create model
model = MLP(784, 128, 10)

# Create optimizer
optimizer = optim.SGD(
    model.parameters(),  # What to optimize
    lr=0.01              # Learning rate (how far to move)
)

# Training step
optimizer.zero_grad()    # 1. Clear old gradients
loss.backward()          # 2. Compute new gradients
optimizer.step()         # 3. Update weights
```

**Line-by-line:**
1. `model.parameters()` - All weights and biases
2. `lr=0.01` - Learning rate (step size)
3. `zero_grad()` - Reset gradients to zero
4. `backward()` - Compute gradients
5. `step()` - Update weights using gradients

**For .NET Devs:**
```csharp
// C# - Similar to iterator pattern
foreach (var parameter in model.Parameters)
{
    parameter.Value -= learningRate * parameter.Gradient;
}
```

---

### Adam Optimizer (Most Popular)

```python
# Adam: Adaptive learning rates (usually better than SGD)
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001  # Lower learning rate than SGD
)
```

**Why Adam?**
- Adapts learning rate per parameter
- Usually converges faster
- Less sensitive to learning rate choice
- **Use this as default!**

---

### Common Optimizers

| Optimizer | Pros | Cons | When to Use |
|-----------|------|------|-------------|
| SGD | Simple, well-understood | Slow, sensitive to LR | Research, baseline |
| Adam | Fast, adaptive | More memory | Default choice |
| AdamW | Adam + weight decay | More hyperparameters | Large models, LLMs |
| RMSprop | Good for RNNs | Older | Recurrent networks |

---

## Part 6: The Training Loop

### Basic Training Pattern

```python
# Setup
model = MLP(784, 128, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):  # 10 epochs
    # Forward pass
    predictions = model(x_train)

    # Compute loss
    loss = loss_fn(predictions, y_train)

    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights

    # Print progress
    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

**Output:**
```
Epoch 0, Loss: 2.3026
Epoch 1, Loss: 1.8934
Epoch 2, Loss: 1.5123
...
Epoch 9, Loss: 0.2134
```

**For .NET Devs:**
```csharp
// C# - Training loop pattern
for (int epoch = 0; epoch < numEpochs; epoch++)
{
    var predictions = model.Forward(xTrain);
    var loss = lossFunction.Compute(predictions, yTrain);
    var gradients = loss.Backward();
    optimizer.UpdateWeights(gradients);
}
```

---

### Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 2. Create training data
X = torch.randn(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1]).unsqueeze(1)  # Target: sum of features

# 3. Initialize
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 4. Training loop
for epoch in range(100):
    # Forward
    predictions = model(X)
    loss = loss_fn(predictions, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")

# 5. Test
with torch.no_grad():  # No gradients needed for testing
    test_input = torch.tensor([[1.0, 2.0]])
    prediction = model(test_input)
    print(f"\nPrediction for [1.0, 2.0]: {prediction.item():.2f}")
    print(f"Expected: 3.0")
```

**Expected Output:**
```
Epoch   0, Loss: 3.4523
Epoch  10, Loss: 1.2345
Epoch  20, Loss: 0.5678
...
Epoch  90, Loss: 0.0123

Prediction for [1.0, 2.0]: 2.98
Expected: 3.0
```

---

## Part 7: Training vs Evaluation Mode

### Why Two Modes?

Some layers behave differently during training vs testing:
- **Dropout**: Randomly drops neurons during training (off during testing)
- **BatchNorm**: Uses batch statistics during training, running statistics during testing

```python
# Training mode (default)
model.train()
# Dropout is active
# BatchNorm uses batch statistics

# Evaluation mode
model.eval()
# Dropout is disabled
# BatchNorm uses running statistics
```

### Proper Evaluation Pattern

```python
# Training
model.train()
for batch in train_loader:
    optimizer.zero_grad()
    output = model(batch.x)
    loss = loss_fn(output, batch.y)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():  # Disable gradient computation
    for batch in test_loader:
        output = model(batch.x)
        # Compute accuracy, etc.
```

**Important:**
- Always call `model.eval()` before testing
- Use `torch.no_grad()` during evaluation (saves memory, faster)

---

## Part 8: MNIST Classifier Example

### Complete MNIST Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Define model
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 128)  # 784 → 128
        self.layer2 = nn.Linear(128, 64)     # 128 → 64
        self.layer3 = nn.Linear(64, 10)      # 64 → 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten image
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)  # No activation (done in loss)
        return x

# 2. Load MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)

# 3. Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# 4. Training function
def train(epoch):
    model.train()
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

# 5. Test function
def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1)  # Get predicted class
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# 6. Train and test
for epoch in range(1, 6):  # 5 epochs
    train(epoch)
    test()

# 7. Save model
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model saved!")
```

**Expected Output:**
```
Epoch 1 [0/60000] Loss: 2.3012
Epoch 1 [6400/60000] Loss: 0.4523
...
Test set: Average loss: 0.2134, Accuracy: 9345/10000 (93.45%)

Epoch 2 [0/60000] Loss: 0.1834
...
Test set: Average loss: 0.1234, Accuracy: 9612/10000 (96.12%)
...
Model saved!
```

**~96% accuracy in 5 epochs!**

---

## Summary

### Key Concepts

**nn.Module:**
- Base class for all models
- Define layers in `__init__`
- Define forward pass in `forward()`
- Must call `super().__init__()`

**Built-in Layers:**
- `nn.Linear` - Fully connected layer
- Activations: `F.relu`, `torch.sigmoid`, `F.softmax`
- Many more: `nn.Conv2d`, `nn.LSTM`, etc.

**Training Components:**
- **Loss function**: Measures error (`nn.CrossEntropyLoss`, `nn.MSELoss`)
- **Optimizer**: Updates weights (`optim.Adam`, `optim.SGD`)
- **Training loop**: forward → loss → backward → step

**Training Pattern:**
```python
optimizer.zero_grad()  # 1. Clear gradients
output = model(x)      # 2. Forward pass
loss = loss_fn(output, y)  # 3. Compute loss
loss.backward()        # 4. Compute gradients
optimizer.step()       # 5. Update weights
```

**Modes:**
- `model.train()` - Training mode
- `model.eval()` + `torch.no_grad()` - Evaluation mode

---

## Quick Reference

### Model Template

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers

    def forward(self, x):
        # Define forward pass
        return x
```

### Training Template

```python
# Setup
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()

# Evaluate
model.eval()
with torch.no_grad():
    output = model(x_test)
```

---

## Quiz

### Question 1
Why must you call `super().__init__()` in your model's `__init__`?

<details>
<summary>Answer</summary>

It initializes the parent `nn.Module` class, which:
- Registers layers so PyTorch can find them
- Enables parameter tracking
- Sets up hooks and other internal mechanisms

Without it, your model won't work properly!

</details>

### Question 2
What's the difference between `F.relu()` and `nn.ReLU()`?

<details>
<summary>Answer</summary>

- `F.relu(x)` - Functional version, stateless
- `nn.ReLU()` - Module version, can be stored in Sequential

```python
# Functional
x = F.relu(x)

# Module
relu = nn.ReLU()
x = relu(x)

# Both do the same thing!
# Use functional in forward(), module in Sequential
```

</details>

### Question 3
Why don't we apply softmax before `nn.CrossEntropyLoss()`?

<details>
<summary>Answer</summary>

`CrossEntropyLoss` applies softmax internally! Applying it twice would be wrong.

```python
# WRONG
output = F.softmax(logits, dim=1)
loss = nn.CrossEntropyLoss()(output, target)

# CORRECT
output = logits  # Raw scores
loss = nn.CrossEntropyLoss()(output, target)
```

</details>

### Question 4
What happens if you forget `optimizer.zero_grad()`?

<details>
<summary>Answer</summary>

Gradients accumulate! Each `backward()` adds to existing gradients, giving wrong results.

```python
# WRONG
loss.backward()
optimizer.step()
# Gradients keep accumulating!

# CORRECT
optimizer.zero_grad()  # Clear old gradients first
loss.backward()
optimizer.step()
```

</details>

---

## Lab Exercises

### Exercise 1: Build a Simple Network

```python
# TODO: Create a network with:
# - Input: 10 features
# - Hidden layer 1: 20 neurons, ReLU activation
# - Hidden layer 2: 10 neurons, ReLU activation
# - Output: 3 classes (no activation)

class MyNetwork(nn.Module):
    def __init__(self):
        # Your code here
        pass

    def forward(self, x):
        # Your code here
        pass

# Test your network
model = MyNetwork()
x = torch.randn(5, 10)  # 5 samples, 10 features
output = model(x)
print(f"Output shape: {output.shape}")  # Should be [5, 3]
```

<details>
<summary>Solution</summary>

```python
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)  # No activation
        return x
```

</details>

---

### Exercise 2: Train on XOR Problem

```python
# XOR dataset
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0, 1, 1, 0])  # XOR output

# TODO:
# 1. Create a 2-layer network (2 → 4 → 1)
# 2. Use BCEWithLogitsLoss (binary classification)
# 3. Train for 1000 epochs
# 4. Test predictions

# Your code here:
```

<details>
<summary>Solution</summary>

```python
# 1. Define model
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 2. Setup
model = XORNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

# 3. Train
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X).squeeze()
    loss = loss_fn(output, y.float())
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 4. Test
model.eval()
with torch.no_grad():
    predictions = torch.sigmoid(model(X)).round()
    print(f"\nPredictions: {predictions.squeeze()}")
    print(f"Actual:      {y}")
```

</details>

---

## Next Steps

**Congratulations!** You can now:
✅ Build neural networks with nn.Module
✅ Use built-in layers and activations
✅ Train models with optimizers
✅ Implement complete training loops

**Next Lesson:** Converting NumPy to PyTorch (`03_numpy_to_pytorch.md`)

You'll learn:
- Side-by-side NumPy vs PyTorch comparison
- Converting your Module 3 projects
- Performance benchmarking
- When to use each framework

---

**Practice building different network architectures before moving on!**

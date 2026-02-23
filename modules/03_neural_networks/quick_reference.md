# Neural Networks Quick Reference

One-page cheat sheet for formulas, code snippets, and debugging tips.

---

## 🧮 Core Formulas

### Forward Propagation

```
Single neuron:
z = w·x + b = Σ(wᵢxᵢ) + b
a = activation(z)

Layer (vectorized):
Z = X @ W + b          # Linear transformation
A = activation(Z)      # Non-linearity
```

### Backpropagation

```
Output layer:
dZ_L = A_L - Y                    # Gradient of loss w.r.t. final layer
dW_L = (1/m) * A_(L-1).T @ dZ_L
db_L = (1/m) * np.sum(dZ_L, axis=0)

Hidden layer:
dZ_l = (dZ_(l+1) @ W_(l+1).T) * activation'(Z_l)
dW_l = (1/m) * A_(l-1).T @ dZ_l
db_l = (1/m) * np.sum(dZ_l, axis=0)
```

### Weight Update

```
Gradient Descent:
W = W - α * dW
b = b - α * db

With Momentum:
v_W = β*v_W + (1-β)*dW
W = W - α*v_W

Adam:
m_W = β₁*m_W + (1-β₁)*dW        # First moment
v_W = β₂*v_W + (1-β₂)*(dW²)     # Second moment
m̂_W = m_W/(1-β₁ᵗ)               # Bias correction
v̂_W = v_W/(1-β₂ᵗ)
W = W - α*m̂_W/(√v̂_W + ε)
```

---

## 🎯 Activation Functions

```python
# ReLU (most common)
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Sigmoid (output layer for binary classification)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Tanh (similar to sigmoid, centered at 0)
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

# Softmax (output layer for multi-class)
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Leaky ReLU (prevents dying ReLU)
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

# GELU (used in GPT)
def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
```

**When to use:**
- Hidden layers: ReLU (default) or Leaky ReLU
- Output (binary): Sigmoid
- Output (multi-class): Softmax
- GPT-style: GELU

---

## 📊 Loss Functions

```python
# Mean Squared Error (regression)
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

# Binary Cross-Entropy (binary classification)
def binary_crossentropy(y_pred, y_true):
    epsilon = 1e-7  # Prevent log(0)
    return -np.mean(y_true * np.log(y_pred + epsilon) +
                    (1 - y_true) * np.log(1 - y_pred + epsilon))

# Categorical Cross-Entropy (multi-class)
def categorical_crossentropy(y_pred, y_true):
    epsilon = 1e-7
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

# Sparse Categorical Cross-Entropy (labels as integers)
def sparse_categorical_crossentropy(y_pred, y_true_indices):
    n = y_pred.shape[0]
    log_probs = -np.log(y_pred[np.arange(n), y_true_indices] + 1e-7)
    return np.mean(log_probs)
```

---

## 🏗️ Network Architecture

### Simple Network Class

```python
class NeuralNetwork:
    def __init__(self, layers):
        """
        layers: list of layer sizes
        Example: [784, 128, 64, 10] = input 784, hidden 128, 64, output 10
        """
        self.layers = layers
        self.weights = []
        self.biases = []

        # Initialize weights (He initialization)
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])
            b = np.zeros(layers[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """Forward propagation through all layers"""
        activations = [X]
        z_values = []

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            z_values.append(z)

            # ReLU for hidden layers, softmax for output
            if i < len(self.weights) - 1:
                a = np.maximum(0, z)  # ReLU
            else:
                a = softmax(z)        # Softmax

            activations.append(a)

        return activations, z_values
```

---

## 🎓 Training Loop Pattern

```python
# Standard training loop (use this pattern!)
def train(model, X_train, y_train, epochs, batch_size, learning_rate):
    n_samples = X_train.shape[0]
    losses = []

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward pass
            predictions, cache = model.forward(X_batch)

            # Compute loss
            loss = compute_loss(predictions, y_batch)

            # Backward pass
            gradients = model.backward(cache, y_batch)

            # Update weights
            model.update_weights(gradients, learning_rate)

        # Track progress
        epoch_loss = compute_loss(model.forward(X_train)[0], y_train)
        losses.append(epoch_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

    return losses
```

---

## 🔍 Debugging Guide

### Shape Debugging

```python
# ALWAYS print shapes when debugging!
print(f"X: {X.shape}")        # (batch, input_features)
print(f"W1: {W1.shape}")      # (input_features, hidden_units)
print(f"Z1: {Z1.shape}")      # (batch, hidden_units)
print(f"A1: {A1.shape}")      # (batch, hidden_units)
```

**Common shape issues:**
```
Error: shapes (64,128) and (64,10) not aligned
Fix: Check W shape. Should be (128, 10), not (64, 10)

Error: could not broadcast together shapes (64,10) and (128,)
Fix: Check bias shape. Should be (10,), not (128,)
```

### Learning Issues

**Symptom: Loss not decreasing**
```python
# Check 1: Learning rate
learning_rate = 0.01  # Try 0.001, 0.01, 0.1

# Check 2: Gradient flow
print(f"Gradient norm: {np.linalg.norm(dW)}")
# Should be > 0 and < 100

# Check 3: Data normalization
X = (X - X.mean()) / X.std()
```

**Symptom: Loss = NaN**
```python
# Cause: Exploding gradients
# Fix 1: Lower learning rate
learning_rate = 0.001

# Fix 2: Gradient clipping
gradients = np.clip(gradients, -1, 1)

# Fix 3: Better weight init
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

**Symptom: Overfitting (train good, val bad)**
```python
# Solution 1: L2 regularization
loss = mse_loss + lambda * np.sum(W**2)

# Solution 2: Dropout
def dropout(A, keep_prob=0.8):
    mask = np.random.rand(*A.shape) < keep_prob
    return A * mask / keep_prob

# Solution 3: More training data
# Solution 4: Simpler model (fewer layers/neurons)
```

---

## 📈 Evaluation Metrics

```python
# Accuracy
def accuracy(y_pred, y_true):
    predictions = np.argmax(y_pred, axis=1)
    return np.mean(predictions == y_true)

# Precision & Recall (binary)
def precision(y_pred, y_true):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp + 1e-7)

def recall(y_pred, y_true):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn + 1e-7)

# F1 Score
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-7)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

---

## 💾 Save/Load Models

```python
# Save model
def save_model(model, filename):
    np.savez(filename,
             weights=[w for w in model.weights],
             biases=[b for b in model.biases],
             layers=model.layers)

# Load model
def load_model(filename):
    data = np.load(filename, allow_pickle=True)
    model = NeuralNetwork(data['layers'])
    model.weights = list(data['weights'])
    model.biases = list(data['biases'])
    return model

# Usage
save_model(model, 'my_model.npz')
model = load_model('my_model.npz')
```

---

## 🎨 Visualization

```python
import matplotlib.pyplot as plt

# Plot training progress
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot predictions (MNIST)
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Pred: {predictions[i]}, True: {labels[i]}')
    ax.axis('off')
plt.show()

# Learning rate finder
learning_rates = [0.0001, 0.001, 0.01, 0.1]
for lr in learning_rates:
    model = train_model(X, y, lr=lr)
    plt.plot(model.losses, label=f'LR={lr}')
plt.legend()
plt.show()
```

---

## 🏁 Quick Start Templates

### Binary Classification

```python
# Network: input → 64 → 32 → 1 (sigmoid)
model = NeuralNetwork([n_features, 64, 32, 1])
activation_final = 'sigmoid'
loss_function = 'binary_crossentropy'
```

### Multi-Class Classification

```python
# Network: input → 128 → 64 → n_classes (softmax)
model = NeuralNetwork([n_features, 128, 64, n_classes])
activation_final = 'softmax'
loss_function = 'categorical_crossentropy'
```

### Regression

```python
# Network: input → 64 → 32 → 1 (linear)
model = NeuralNetwork([n_features, 64, 32, 1])
activation_final = 'linear'  # No activation
loss_function = 'mse'
```

---

## ⚙️ Hyperparameters

### Typical Ranges

```python
learning_rate = 0.001  # Usually 0.0001 - 0.1
batch_size = 32        # Usually 16 - 256
epochs = 100           # Usually 10 - 1000
hidden_units = 128     # Usually 32 - 512

# Adam optimizer (recommended)
beta1 = 0.9           # Momentum
beta2 = 0.999         # RMSprop
epsilon = 1e-8        # Numerical stability
```

### Learning Rate Schedule

```python
# Decay learning rate over time
def lr_schedule(epoch, initial_lr=0.01):
    return initial_lr * 0.95 ** epoch

# Or step decay
def step_decay(epoch, initial_lr=0.01, drop=0.5, epochs_drop=10):
    return initial_lr * (drop ** (epoch // epochs_drop))
```

---

## 🔧 Common Patterns

### Data Preprocessing

```python
# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Or scale to [0, 1]
X = (X - X.min()) / (X.max() - X.min())

# One-hot encode labels
def one_hot(y, n_classes):
    return np.eye(n_classes)[y]

y_encoded = one_hot(y, n_classes=10)
```

### Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

# 60% train, 20% val, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)
```

---

## 🎯 Remember

1. **Always normalize data** before training
2. **Print shapes** when debugging
3. **Start with small learning rate** (0.001)
4. **Use Adam optimizer** (best default choice)
5. **Monitor validation loss** (detect overfitting)
6. **Save best model** during training
7. **Visualize predictions** to understand failures

---

**Keep this page bookmarked for quick reference while coding!**

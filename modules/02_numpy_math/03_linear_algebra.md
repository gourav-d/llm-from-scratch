# Lesson 2.3: Linear Algebra for Neural Networks

## Why Linear Algebra?
Neural networks are **matrix operations**. Understanding this is crucial for LLMs.

## Vectors

```python
import numpy as np

# Vector = 1D array
v = np.array([1, 2, 3])

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Addition
print(v1 + v2)  # [5, 7, 9]

# Dot product (similarity measure!)
dot = v1 @ v2  # 1*4 + 2*5 + 3*6 = 32
print(dot)

# Magnitude (length)
magnitude = np.linalg.norm(v1)  # sqrt(1² + 2² + 3²) = 3.74
print(magnitude)

# Unit vector (direction only)
unit = v1 / np.linalg.norm(v1)
print(unit)
```

**In neural networks:**
- Input = vector
- Weights = vector
- Dot product = weighted sum

## Matrices

```python
# Matrix = 2D array
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3 matrix

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])  # 3x2 matrix

# Matrix multiplication
C = A @ B  # (2x3) @ (3x2) = (2x2)
print(C)
# [[58,  64],
#  [139, 154]]

# Element-wise
print(A * 2)  # Multiply all by 2

# Transpose
print(A.T)
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

**In neural networks:**
- Each layer = matrix multiplication
- Input shape: (batch_size, features)
- Weights shape: (features, neurons)
- Output shape: (batch_size, neurons)

## Matrix Dimensions

```python
# Rule: (m, n) @ (n, p) = (m, p)
# Inner dimensions must match!

A = np.random.randn(2, 3)  # 2x3
B = np.random.randn(3, 4)  # 3x4
C = A @ B  # (2,3) @ (3,4) = (2,4) ✅

# This fails:
# D = A @ A  # (2,3) @ (2,3) ❌ - inner dims don't match
```

## Common Linear Algebra Operations

### Identity Matrix

```python
# Identity matrix (like 1 in multiplication)
I = np.eye(3)
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]

A = np.array([[1, 2], [3, 4]])
print(A @ np.eye(2))  # Returns A
```

### Inverse

```python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)

print(A @ A_inv)  # Should be identity (with rounding)
# [[1., 0.],
#  [0., 1.]]
```

### Determinant

```python
A = np.array([[1, 2], [3, 4]])
det = np.linalg.det(A)
print(det)  # -2.0
```

### Solving Linear Equations

```python
# Solve: Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

x = np.linalg.solve(A, b)
print(x)  # [2. 3.]

# Verify
print(A @ x)  # [9. 8.] ✅
```

## Neural Network Example

```python
# Simple 2-layer network

# Input: 4 samples, 3 features
X = np.random.randn(4, 3)

# Layer 1: 3 → 5 neurons
W1 = np.random.randn(3, 5) * 0.01
b1 = np.zeros(5)

# Forward pass
Z1 = X @ W1 + b1  # (4,3) @ (3,5) + (5,) = (4,5)

# Activation (ReLU)
A1 = np.maximum(0, Z1)

# Layer 2: 5 → 2 neurons
W2 = np.random.randn(5, 2) * 0.01
b2 = np.zeros(2)

Z2 = A1 @ W2 + b2  # (4,5) @ (5,2) + (2,) = (4,2)

print(f"Input shape: {X.shape}")      # (4, 3)
print(f"Hidden shape: {A1.shape}")    # (4, 5)
print(f"Output shape: {Z2.shape}")    # (4, 2)
```

## Eigenvalues and Eigenvectors

```python
A = np.array([[4, -2], [1, 1]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

**Used in:** PCA, understanding network behavior

## Norms

```python
v = np.array([3, 4])

# L2 norm (Euclidean distance)
l2 = np.linalg.norm(v)  # sqrt(3² + 4²) = 5
print(l2)

# L1 norm (Manhattan distance)
l1 = np.linalg.norm(v, ord=1)  # |3| + |4| = 7
print(l1)
```

**Used in:** Regularization, gradient clipping

## Practice: Build a Perceptron

```python
import numpy as np

# Simple perceptron (1 neuron)
class Perceptron:
    def __init__(self, n_features):
        # Random weights
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

    def forward(self, X):
        # X: (n_samples, n_features)
        # weights: (n_features,)
        z = X @ self.weights + self.bias  # (n_samples,)
        # Activation (step function)
        return np.where(z > 0, 1, 0)

# Test
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
model = Perceptron(n_features=2)
predictions = model.forward(X)
print(predictions)
```

## Practice Exercises

```python
import numpy as np

# 1. Vector operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Dot: {a @ b}")
print(f"Magnitude a: {np.linalg.norm(a)}")

# 2. Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)

# 3. Transpose and shapes
X = np.random.randn(100, 784)  # 100 images, 784 pixels
W = np.random.randn(784, 10)   # 10 classes
output = X @ W
print(f"Output shape: {output.shape}")  # (100, 10)

# 4. Normalize data (zero mean, unit variance)
data = np.random.randn(1000, 5)
mean = data.mean(axis=0)
std = data.std(axis=0)
normalized = (data - mean) / std
print(f"New mean: {normalized.mean(axis=0)}")
print(f"New std: {normalized.std(axis=0)}")

# 5. Cosine similarity
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
cos_sim = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"Cosine similarity: {cos_sim}")
```

## 💡 Key Concepts

**For Neural Networks:**
- Input × Weights = Output
- Shape math: (batch, in) @ (in, out) = (batch, out)
- Transpose swaps dimensions
- Broadcasting adds bias
- Dot product = weighted sum

**Linear Algebra:**
- Vectors = 1D arrays
- Matrices = 2D arrays
- `@` for matrix mult
- `np.linalg` for advanced ops

## Module 2 Complete! ✅

**What you learned:**
- NumPy arrays and operations
- Broadcasting and vectorization
- Matrix multiplication
- Linear algebra basics

**Next Steps:**
1. Practice with exercises
2. Take quiz (when ready)
3. **Module 3: Neural Networks** - Build your first network!

You now have the math foundation for neural networks! 🚀

# NumPy Quick Reference

A one-page cheat sheet for quick lookup. Bookmark this!

---

## Import Convention
```python
import numpy as np
```

---

## Creating Arrays

```python
# From list
np.array([1, 2, 3])
np.array([[1,2], [3,4]])

# Zeros and ones
np.zeros(5)                    # [0. 0. 0. 0. 0.]
np.zeros((3, 4))               # 3x4 matrix of zeros
np.ones((2, 3))                # 2x3 matrix of ones
np.full((2, 2), 7)             # 2x2 matrix filled with 7

# Ranges
np.arange(10)                  # [0 1 2 3 4 5 6 7 8 9]
np.arange(5, 10)               # [5 6 7 8 9]
np.arange(0, 10, 2)            # [0 2 4 6 8]
np.linspace(0, 1, 5)           # 5 evenly spaced: [0. 0.25 0.5 0.75 1.]

# Random
np.random.rand(3, 3)           # Uniform [0, 1)
np.random.randn(3, 3)          # Normal distribution (mean=0, std=1)
np.random.randint(0, 10, (3,3)) # Random integers
np.random.seed(42)             # Set seed for reproducibility

# Special matrices
np.eye(3)                      # 3x3 identity matrix
np.diag([1, 2, 3])             # Diagonal matrix
```

---

## Array Properties

```python
arr.shape          # Dimensions: (rows, cols, ...)
arr.size           # Total number of elements
arr.ndim           # Number of dimensions
arr.dtype          # Data type (int64, float64, etc.)
arr.nbytes         # Memory usage in bytes
```

---

## Indexing and Slicing

```python
# 1D
arr[0]             # First element
arr[-1]            # Last element
arr[1:4]           # Slice: elements 1, 2, 3
arr[::2]           # Every 2nd element
arr[::-1]          # Reverse

# 2D
matrix[i, j]       # Element at row i, column j
matrix[0, :]       # First row
matrix[:, 0]       # First column
matrix[0:2, 1:3]   # Submatrix

# Boolean indexing
arr[arr > 5]       # Elements greater than 5
arr[(arr > 5) & (arr < 10)]  # Multiple conditions
```

---

## Reshaping

```python
arr.reshape(3, 4)          # Reshape to 3x4
arr.reshape(-1)            # Flatten to 1D
arr.flatten()              # Flatten (copy)
arr.ravel()                # Flatten (view, faster)
arr.reshape(3, -1)         # 3 rows, auto-calculate cols

arr.T                      # Transpose
arr.transpose()            # Transpose
```

---

## Array Operations (Element-wise)

```python
# Arithmetic
arr + 5                    # Add scalar
arr * 2                    # Multiply by scalar
arr1 + arr2                # Add arrays element-wise
arr1 * arr2                # Multiply element-wise
arr ** 2                   # Square each element

# Math functions
np.sqrt(arr)               # Square root
np.exp(arr)                # e^x
np.log(arr)                # Natural log
np.abs(arr)                # Absolute value
np.sin(arr), np.cos(arr)   # Trig functions

# Comparison
arr > 5                    # Boolean array
arr == arr2                # Element-wise comparison
```

---

## Matrix Operations

```python
# Dot product
a @ b                      # Matrix multiplication (Python 3.5+)
np.dot(a, b)               # Matrix multiplication
np.inner(a, b)             # Inner product

# Linear algebra
np.linalg.inv(A)           # Matrix inverse
np.linalg.det(A)           # Determinant
np.linalg.eig(A)           # Eigenvalues and eigenvectors
np.linalg.norm(v)          # Vector norm (length)
np.linalg.solve(A, b)      # Solve Ax = b
```

---

## Aggregations

```python
# Overall
arr.sum()                  # Sum all elements
arr.mean()                 # Average
arr.std()                  # Standard deviation
arr.var()                  # Variance
arr.min(), arr.max()       # Min and max
arr.argmin(), arr.argmax() # Index of min/max

# Along axis
arr.sum(axis=0)            # Sum each column
arr.sum(axis=1)            # Sum each row
arr.mean(axis=0)           # Mean of each column

# Cumulative
np.cumsum(arr)             # Cumulative sum
np.cumprod(arr)            # Cumulative product
```

---

## Stacking and Splitting

```python
# Stacking
np.vstack([a, b])          # Stack vertically (rows)
np.hstack([a, b])          # Stack horizontally (cols)
np.concatenate([a, b])     # Concatenate
np.stack([a, b])           # Stack along new axis

# Splitting
np.split(arr, 3)           # Split into 3 parts
np.vsplit(arr, 2)          # Split vertically
np.hsplit(arr, 2)          # Split horizontally
```

---

## Broadcasting Rules

Arrays are compatible when:
1. They have the same shape, OR
2. One of the dimensions is 1, OR
3. One dimension is missing (will be added)

```python
# Examples
(3, 4) + (3, 4)  →  (3, 4)  ✓
(3, 4) + (1, 4)  →  (3, 4)  ✓ (broadcast row)
(3, 4) + (3, 1)  →  (3, 4)  ✓ (broadcast column)
(3, 4) + (4,)    →  (3, 4)  ✓ (broadcast to each row)
(3, 4) + (3, 2)  →  Error!  ✗ (incompatible)
```

---

## Shape Compatibility for Matrix Multiplication

```python
# Rule: (m, n) @ (n, p) = (m, p)
#           ↑____↑
#       must match!

(2, 3) @ (3, 4)  =  (2, 4)  ✓
(5, 2) @ (2, 7)  =  (5, 7)  ✓
(3, 4) @ (5, 2)  =  Error!  ✗
```

---

## Data Types

```python
np.int32, np.int64         # Integers
np.float32, np.float64     # Floats
np.bool_                   # Boolean
np.complex64               # Complex numbers

# Convert
arr.astype(np.float64)     # Change data type
arr.astype(int)            # Convert to int
```

---

## Boolean Operations

```python
arr > 5                    # Element-wise comparison
(arr > 5) & (arr < 10)     # AND (use &, not 'and')
(arr < 5) | (arr > 10)     # OR (use |, not 'or')
~(arr > 5)                 # NOT (use ~, not 'not')

arr.any()                  # True if any element is True
arr.all()                  # True if all elements are True
```

---

## Where and Conditionals

```python
# Conditional assignment
np.where(arr > 5, 100, arr)      # If >5, set to 100, else keep
np.where(arr > 5, arr, 0)        # If >5, keep, else set to 0

# Multiple conditions
np.select([cond1, cond2], [val1, val2], default)
```

---

## Sorting

```python
np.sort(arr)               # Return sorted copy
arr.sort()                 # Sort in-place
arr.argsort()              # Indices that would sort array
np.partition(arr, k)       # Partial sort (k smallest first)
```

---

## Statistics

```python
np.median(arr)             # Median
np.percentile(arr, 75)     # 75th percentile
np.corrcoef(x, y)          # Correlation coefficient
np.cov(x, y)               # Covariance
```

---

## Set Operations

```python
np.unique(arr)             # Unique elements
np.intersect1d(a, b)       # Intersection
np.union1d(a, b)           # Union
np.setdiff1d(a, b)         # Difference
```

---

## Common Patterns

### Normalize data (0-1 range)
```python
normalized = (arr - arr.min()) / (arr.max() - arr.min())
```

### Standardize data (mean=0, std=1)
```python
standardized = (arr - arr.mean()) / arr.std()
```

### One-hot encoding
```python
def one_hot(labels, n_classes):
    return np.eye(n_classes)[labels]
```

### Softmax
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)
```

### Cosine similarity
```python
def cosine_sim(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

## Debugging Tips

```python
# Always print shapes!
print(f"Shape: {arr.shape}")

# Check for NaN or Inf
np.isnan(arr).any()
np.isinf(arr).any()

# Print with more decimals
np.set_printoptions(precision=4, suppress=True)

# Print full array (no truncation)
np.set_printoptions(threshold=np.inf)
```

---

## Common Errors and Fixes

### Shape mismatch
```python
# Error: shapes (3,4) and (5,2) not aligned
# Fix: Check inner dimensions match for @
```

### Broadcasting error
```python
# Error: operands could not be broadcast together
# Fix: Check shape compatibility rules
```

### Modifying view vs copy
```python
# Problem: Slice modifies original
arr_slice = arr[:]        # View!
arr_slice[0] = 999        # Modifies original!

# Solution:
arr_copy = arr.copy()     # Independent copy
```

---

## Performance Tips

1. **Vectorize** - avoid Python loops
2. **Use views** - `arr[:]` instead of `arr.copy()` when possible
3. **Preallocate** - create array once, fill in-place
4. **Right dtype** - use `float32` instead of `float64` if precision allows
5. **Avoid repeated shape changes** - reshape once
6. **Use `@`** for matrix mult, not nested loops
7. **Boolean indexing** is fast - use it!

---

## Neural Network Cheat Sheet

### Layer computation
```python
# Input: X (batch_size, input_features)
# Weights: W (input_features, output_neurons)
# Bias: b (output_neurons,)

output = X @ W + b   # Shape: (batch_size, output_neurons)
```

### Common activation functions
```python
# ReLU
relu = np.maximum(0, z)

# Sigmoid
sigmoid = 1 / (1 + np.exp(-z))

# Tanh
tanh = np.tanh(z)

# Softmax (for classification)
exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
softmax = exp_z / exp_z.sum(axis=1, keepdims=True)
```

### Loss functions
```python
# Mean Squared Error
mse = np.mean((predictions - targets) ** 2)

# Cross-Entropy (classification)
n = len(labels)
ce = -np.log(predictions[np.arange(n), labels]).mean()
```

---

## Resources

- **Official Docs:** https://numpy.org/doc/stable/
- **Quickstart:** https://numpy.org/doc/stable/user/quickstart.html
- **API Reference:** https://numpy.org/doc/stable/reference/

---

**Pro tip:** Keep this page open while coding. Looking up syntax is normal and good practice!

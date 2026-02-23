# Lesson 2.1: NumPy Basics

## Why NumPy?

- **Foundation of ML/AI** - All neural networks use NumPy
- **Fast** - 50-100x faster than Python lists
- **Matrix operations** - Essential for neural networks

## Installation

```bash
pip install numpy
```

## Creating Arrays

```python
import numpy as np

# From Python list
arr = np.array([1, 2, 3, 4, 5])
print(arr)  # [1 2 3 4 5]
print(type(arr))  # <class 'numpy.ndarray'>

# 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix)
# [[1 2 3]
#  [4 5 6]]

# Using built-in functions
zeros = np.zeros(5)        # [0. 0. 0. 0. 0.]
ones = np.ones((2, 3))     # 2x3 matrix of 1s
range_arr = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
linspace = np.linspace(0, 1, 5)  # 5 values from 0 to 1
random = np.random.rand(3, 3)    # 3x3 random values
```

**C# vs NumPy:**

```csharp
// C#
int[] arr = new int[] {1, 2, 3};

// NumPy
arr = np.array([1, 2, 3])
```

## Array Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.shape)   # (2, 3) - 2 rows, 3 columns
print(arr.size)    # 6 - total elements
print(arr.ndim)    # 2 - number of dimensions
print(arr.dtype)   # dtype('int64') - data type
```

## Indexing and Slicing

```python
arr = np.array([10, 20, 30, 40, 50])

# Indexing (like Python lists)
print(arr[0])    # 10
print(arr[-1])   # 50

# Slicing
print(arr[1:4])  # [20 30 40]
print(arr[:3])   # [10 20 30]
print(arr[::2])  # [10 30 50]

# 2D indexing
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[0, 0])    # 1
print(matrix[1, 2])    # 6
print(matrix[0, :])    # [1 2 3] - first row
print(matrix[:, 0])    # [1 4 7] - first column
print(matrix[0:2, 1:3]) # [[2 3], [5 6]] - submatrix
#
# Slicing in Python follows the rule: Start at the first number, but stop BEFORE the second number.
# The Rows 0:2: This tells NumPy to take Row 0 and Row 1 (it stops before Row 2).
# Resulting Rows: [1, 2, 3] and [4, 5, 6]
# The Columns 1:3: This tells NumPy to take Column 1 and Column 2 (it stops before Column 3).
# Resulting Columns: The 2nd and 3rd vertical slices.
```

## Reshaping

```python
arr = np.arange(12)  # [0 1 2 3 4 5 6 7 8 9 10 11]

# Reshape to 3x4
reshaped = arr.reshape(3, 4)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# Reshape to 2x6
reshaped = arr.reshape(2, 6)

# Flatten to 1D
flat = reshaped.flatten()  # [0 1 2 3 4 5 6 7 8 9 10 11]
```

## Basic Operations

```python
arr = np.array([1, 2, 3, 4, 5])

# Element-wise operations (vectorized!)
print(arr + 10)   # [11 12 13 14 15]
print(arr * 2)    # [ 2  4  6  8 10]
print(arr ** 2)   # [ 1  4  9 16 25]

# Array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(arr1 + arr2)  # [5 7 9]
print(arr1 * arr2)  # [ 4 10 18]

# Aggregations
print(arr.sum())   # 15
print(arr.mean())  # 3.0
print(arr.max())   # 5
print(arr.min())   # 1
print(arr.std())   # Standard deviation
```

**Python list vs NumPy:**

```python
# Python list - slow
list1 = [1, 2, 3]
list2 = [4, 5, 6]
# list1 + list2 = [1, 2, 3, 4, 5, 6] - concatenates!

# NumPy - fast, element-wise
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr1 + arr2  # [5 7 9] - adds elements!
```

## Boolean Indexing

```python
arr = np.array([10, 25, 30, 15, 40])

# Boolean condition
mask = arr > 20
print(mask)  # [False  True  True False  True]

# Filter using mask
result = arr[mask]
print(result)  # [25 30 40]

# One line
result = arr[arr > 20]
print(result)  # [25 30 40]

# Multiple conditions
result = arr[(arr > 15) & (arr < 35)]
print(result)  # [25 30]
```

## Practice

```python
import numpy as np

# Create arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.arange(0, 10, 2)  # [0 2 4 6 8]
zeros = np.zeros((3, 3))
ones = np.ones(5)

# Operations
print(arr1 * 2)
print(arr1 + arr2[:5])
print(arr1.sum())
print(arr1.mean())

# 2D operations
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.sum())        # 21
print(matrix.sum(axis=0))  # [5 7 9] - column sums
print(matrix.sum(axis=1))  # [ 6 15] - row sums

# Filtering
data = np.array([5, 15, 25, 35, 45])
filtered = data[data > 20]
print(filtered)  # [25 35 45]
```

## 💡 Key Points

- `np.array()` creates arrays
- `.shape`, `.size`, `.ndim` for properties
- Slicing like Python: `arr[start:stop:step]`
- Operations are vectorized (fast!)
- Boolean indexing for filtering

**Next:** `02_array_operations.md`

# Lesson 1: PyTorch Fundamentals

**From NumPy Arrays to PyTorch Tensors**

---

## Learning Objectives

By the end of this lesson, you will:
- Understand what PyTorch is and why it's used
- Create and manipulate PyTorch tensors
- Understand automatic differentiation (autograd)
- Move tensors between CPU and GPU
- Compare PyTorch to NumPy operations

**Time:** 4-6 hours

---

## What is PyTorch?

### Simple Explanation

**PyTorch** is a Python library for building neural networks. Think of it as "NumPy with superpowers":

```
NumPy:
- Fast arrays
- Math operations
- Manual gradients ❌

PyTorch:
- Fast tensors (like arrays)
- Math operations
- Automatic gradients ✅
- GPU acceleration ✅
- Built-in neural network layers ✅
```

### For .NET Developers

```csharp
// C# - You might use ML.NET for machine learning
var model = mlContext.Model.Load(modelPath);
var predictions = model.Transform(data);

// Python/PyTorch - Similar concept but more flexible
model = MyNeuralNetwork()
predictions = model(data)  # Forward pass
loss.backward()             # Auto gradients!
```

**Key Differences:**
- **ML.NET**: High-level, structured, .NET ecosystem
- **PyTorch**: Low-level control, dynamic, research-friendly

---

## Installing PyTorch

### Check Installation

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### Version Check Script

```python
# check_pytorch.py
import torch
import sys

print("=" * 50)
print("PyTorch Installation Check")
print("=" * 50)

# Version
print(f"\nPyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")

# CUDA (GPU) availability
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU (this is fine for learning!)")

print("\n" + "=" * 50)
print("✅ PyTorch is ready!")
print("=" * 50)
```

**Run it:**
```bash
python check_pytorch.py
```

---

## Part 1: Tensors vs Arrays

### What are Tensors?

**Tensors** are multi-dimensional arrays, just like NumPy arrays!

```
Scalar (0D):  5
Vector (1D):  [1, 2, 3]
Matrix (2D):  [[1, 2], [3, 4]]
Tensor (3D+): [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
```

**For .NET Devs:** Think of tensors like multi-dimensional arrays (`int[,,]`) but with automatic gradients.

---

### Creating Tensors

#### From Python Lists

```python
import torch
import numpy as np

# NumPy array
np_array = np.array([1, 2, 3, 4, 5])
print(f"NumPy: {np_array}")
# Output: [1 2 3 4 5]

# PyTorch tensor
pt_tensor = torch.tensor([1, 2, 3, 4, 5])
print(f"PyTorch: {pt_tensor}")
# Output: tensor([1, 2, 3, 4, 5])
```

**Line-by-line:**
1. `import torch` - Load PyTorch library
2. `torch.tensor([...])` - Create tensor from list
3. PyTorch adds "tensor(...)" wrapper when printing

**C# Equivalent:**
```csharp
// C# - Create array
int[] array = new int[] {1, 2, 3, 4, 5};
```

---

#### Common Tensor Creation Methods

```python
# 1. Zeros (all elements = 0)
zeros = torch.zeros(3, 4)  # 3x4 matrix of zeros
print(zeros)
# tensor([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])

# 2. Ones (all elements = 1)
ones = torch.ones(2, 3)  # 2x3 matrix of ones
print(ones)
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

# 3. Random values (0 to 1)
random = torch.rand(2, 2)  # 2x2 random matrix
print(random)
# tensor([[0.5234, 0.8923],
#         [0.1234, 0.9876]])

# 4. Random normal distribution (mean=0, std=1)
randn = torch.randn(3, 3)  # 3x3 normal random
print(randn)
# tensor([[-0.1234,  1.2345, -0.5678],
#         [ 0.9012, -0.3456,  0.7890],
#         [-1.2345,  0.5678,  0.1234]])

# 5. Range of values
range_tensor = torch.arange(0, 10, 2)  # Start=0, End=10, Step=2
print(range_tensor)
# tensor([0, 2, 4, 6, 8])

# 6. Evenly spaced values
linspace = torch.linspace(0, 1, 5)  # 5 values from 0 to 1
print(linspace)
# tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

**NumPy Comparison:**

```python
# NumPy                    # PyTorch
np.zeros((3, 4))          # torch.zeros(3, 4)
np.ones((2, 3))           # torch.ones(2, 3)
np.random.rand(2, 2)      # torch.rand(2, 2)
np.random.randn(3, 3)     # torch.randn(3, 3)
np.arange(0, 10, 2)       # torch.arange(0, 10, 2)
np.linspace(0, 1, 5)      # torch.linspace(0, 1, 5)
```

**Almost identical syntax!**

---

### Tensor Properties

```python
x = torch.randn(3, 4, 5)  # Create 3D tensor

# Shape (dimensions)
print(f"Shape: {x.shape}")          # torch.Size([3, 4, 5])
print(f"Size: {x.size()}")          # torch.Size([3, 4, 5]) - same as shape

# Number of dimensions
print(f"Dimensions: {x.ndim}")      # 3

# Total number of elements
print(f"Elements: {x.numel()}")     # 3 * 4 * 5 = 60

# Data type
print(f"Data type: {x.dtype}")      # torch.float32 (default)

# Device (CPU or GPU)
print(f"Device: {x.device}")        # cpu (or cuda:0 for GPU)
```

**Line-by-line:**
1. `x.shape` - Returns dimensions as `torch.Size([3, 4, 5])`
2. `x.size()` - Same as shape (method version)
3. `x.ndim` - Number of dimensions (3 in this case)
4. `x.numel()` - Total elements = 3×4×5 = 60
5. `x.dtype` - Data type (float32, int64, etc.)
6. `x.device` - Where tensor lives (CPU or GPU)

**C# Equivalent:**
```csharp
// C# multi-dimensional array
int[,,] array = new int[3, 4, 5];
Console.WriteLine($"Length: {array.Length}");      // 60
Console.WriteLine($"Rank: {array.Rank}");          // 3
Console.WriteLine($"Dimension 0: {array.GetLength(0)}");  // 3
```

---

### Data Types

```python
# Integer types
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
print(int_tensor.dtype)  # torch.int32

# Float types (most common for neural networks)
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print(float_tensor.dtype)  # torch.float32

# Boolean
bool_tensor = torch.tensor([True, False, True])
print(bool_tensor.dtype)  # torch.bool

# Convert between types
x = torch.tensor([1, 2, 3])
x_float = x.float()         # Convert to float32
x_double = x.double()       # Convert to float64
x_int = x_float.int()       # Convert back to int32
```

**Common Data Types:**

| PyTorch | NumPy | C# | Bits | Range |
|---------|-------|-----|------|-------|
| torch.float32 | np.float32 | float | 32 | ~7 decimal digits |
| torch.float64 | np.float64 | double | 64 | ~15 decimal digits |
| torch.int32 | np.int32 | int | 32 | -2B to 2B |
| torch.int64 | np.int64 | long | 64 | Very large |
| torch.bool | np.bool | bool | 1 | True/False |

**Default:** PyTorch uses `float32` by default (good for neural networks).

---

## Part 2: Tensor Operations

### Basic Math Operations

```python
# Create tensors
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
print(a + b)        # tensor([5., 7., 9.])
print(a - b)        # tensor([-3., -3., -3.])
print(a * b)        # tensor([4., 10., 18.])
print(a / b)        # tensor([0.25, 0.4, 0.5])
print(a ** 2)       # tensor([1., 4., 9.]) - square

# In-place operations (modify original tensor)
a.add_(b)           # a += b (adds b to a)
print(a)            # tensor([5., 7., 9.])
```

**In-place operations** end with `_`:
- `a.add_(b)` - Modifies `a` in place
- `a.add(b)` - Creates new tensor, `a` unchanged

**C# LINQ Equivalent:**
```csharp
// C# - Element-wise operations
var a = new[] {1.0, 2.0, 3.0};
var b = new[] {4.0, 5.0, 6.0};
var sum = a.Zip(b, (x, y) => x + y).ToArray();
```

---

### Matrix Operations

```python
# Create matrices
A = torch.tensor([[1, 2], [3, 4]])      # 2x2 matrix
B = torch.tensor([[5, 6], [7, 8]])      # 2x2 matrix

# Element-wise multiplication (Hadamard product)
print(A * B)
# tensor([[ 5, 12],
#         [21, 32]])

# Matrix multiplication (dot product)
print(torch.matmul(A, B))  # or A @ B
# tensor([[19, 22],
#         [43, 50]])

# Transpose
print(A.T)
# tensor([[1, 3],
#         [2, 4]])

# Reshape
print(A.view(1, 4))        # Reshape to 1x4
# tensor([[1, 2, 3, 4]])
```

**Matrix Multiplication Explained:**

```
A @ B = [[1*5 + 2*7,  1*6 + 2*8],     [[19, 22],
         [3*5 + 4*7,  3*6 + 4*8]]  =   [43, 50]]
```

**NumPy Comparison:**

```python
# NumPy                    # PyTorch
np.matmul(A, B)           # torch.matmul(A, B)  or  A @ B
A.T                       # A.T
A.reshape(1, 4)           # A.view(1, 4)  or  A.reshape(1, 4)
```

---

### Indexing and Slicing

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Access single element
print(x[0, 0])          # tensor(1)
print(x[1, 2])          # tensor(6)

# Access row
print(x[0])             # tensor([1, 2, 3])
print(x[1, :])          # tensor([4, 5, 6]) - same as x[1]

# Access column
print(x[:, 0])          # tensor([1, 4, 7])

# Slicing
print(x[:2, :2])        # First 2 rows, first 2 columns
# tensor([[1, 2],
#         [4, 5]])

# Boolean indexing
mask = x > 5            # Create boolean mask
print(mask)
# tensor([[False, False, False],
#         [False, False,  True],
#         [ True,  True,  True]])

print(x[mask])          # Get elements where mask is True
# tensor([6, 7, 8, 9])
```

**Exactly like NumPy!**

**C# Equivalent:**
```csharp
// C# 2D array
int[,] array = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
int element = array[0, 0];     // 1
// Note: C# doesn't have built-in slicing like Python
```

---

## Part 3: Automatic Differentiation (Autograd)

### The Magic of `requires_grad`

This is where PyTorch becomes powerful! **Automatic differentiation** means PyTorch calculates gradients for you.

**Remember Module 3?** You manually calculated gradients:

```python
# Module 3 - Manual gradients (NumPy)
def backward(x, w, grad_output):
    grad_w = x.T @ grad_output  # Manual calculation!
    grad_x = grad_output @ w.T  # Manual calculation!
    return grad_x, grad_w
```

**Module 3.5 - Automatic gradients (PyTorch):**

```python
# PyTorch - Automatic gradients
x = torch.tensor([2.0, 3.0], requires_grad=True)
w = torch.tensor([1.0, 2.0], requires_grad=True)

y = (x * w).sum()  # y = 2*1 + 3*2 = 8
y.backward()        # Compute gradients automatically!

print(x.grad)       # tensor([1., 2.]) - gradient w.r.t x
print(w.grad)       # tensor([2., 3.]) - gradient w.r.t w
```

**Line-by-line:**
1. `requires_grad=True` - Tell PyTorch "track operations on this tensor"
2. `y = (x * w).sum()` - Forward pass (PyTorch records all operations)
3. `y.backward()` - Backward pass (compute all gradients automatically!)
4. `x.grad` - Access gradient of y with respect to x (∂y/∂x)
5. `w.grad` - Access gradient of y with respect to w (∂y/∂w)

---

### How Autograd Works

**Computational Graph:**

```
Forward Pass (PyTorch records operations):

x = [2, 3]  ----\
                 * -----> z = [2, 6] ----> sum() ----> y = 8
w = [1, 2]  ----/

Backward Pass (PyTorch computes gradients):

∂y/∂x ← chain rule ← ∂y/∂z ← gradient flows back
∂y/∂w ← chain rule ← ∂y/∂z ← gradient flows back
```

**For .NET Devs:** Think of it like dependency tracking:
- Forward pass: PyTorch records "who depends on whom"
- Backward pass: PyTorch walks the graph backward, applying chain rule

---

### Simple Example

```python
# Example: y = x^2 + 3
x = torch.tensor([2.0], requires_grad=True)

# Forward pass
y = x**2 + 3
print(f"y = {y.item()}")  # y = 7.0

# Backward pass (compute gradient)
y.backward()

# Gradient: dy/dx = 2x = 2*2 = 4
print(f"Gradient: {x.grad.item()}")  # 4.0
```

**Math:**
```
y = x² + 3
dy/dx = 2x
At x=2: dy/dx = 2*2 = 4 ✓
```

---

### Multi-Variable Example

```python
# Example: z = x*y + y^2
x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)

# Forward pass
z = x * y + y**2
print(f"z = {z.item()}")  # z = 3*2 + 2^2 = 10.0

# Backward pass
z.backward()

print(f"dz/dx = {x.grad.item()}")  # 2.0
print(f"dz/dy = {y.grad.item()}")  # 7.0
```

**Math:**
```
z = xy + y²

∂z/∂x = y = 2 ✓
∂z/∂y = x + 2y = 3 + 2*2 = 7 ✓
```

---

### Important: Zero Gradients

**Gradients accumulate** in PyTorch! You must zero them between iterations.

```python
x = torch.tensor([1.0], requires_grad=True)

# First backward pass
y = x ** 2
y.backward()
print(f"First gradient: {x.grad.item()}")  # 2.0

# Second backward pass (without zeroing)
y = x ** 2
y.backward()
print(f"Accumulated gradient: {x.grad.item()}")  # 4.0 (2+2!)

# Zero gradients
x.grad.zero_()
y = x ** 2
y.backward()
print(f"After zeroing: {x.grad.item()}")  # 2.0
```

**Why?** Useful for batch gradient accumulation, but can cause bugs if forgotten!

**C# Analogy:**
```csharp
// C# - Similar to event handlers
// If you don't unsubscribe, handlers accumulate
button.Click += OnClick;  // First handler
button.Click += OnClick;  // Second handler (both will fire!)
```

---

## Part 4: GPU Acceleration

### Moving Tensors to GPU

```python
# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Create tensor on CPU (default)
x_cpu = torch.randn(1000, 1000)
print(f"Device: {x_cpu.device}")  # cpu

# Move to GPU
if torch.cuda.is_available():
    x_gpu = x_cpu.to('cuda')       # Move to GPU
    print(f"Device: {x_gpu.device}")  # cuda:0

    # Or create directly on GPU
    y_gpu = torch.randn(1000, 1000, device='cuda')

    # Compute on GPU (much faster for large tensors!)
    z_gpu = x_gpu @ y_gpu

    # Move back to CPU (if needed)
    z_cpu = z_gpu.to('cpu')
```

**Line-by-line:**
1. `torch.cuda.is_available()` - Check if GPU is available
2. `x_cpu.to('cuda')` - Copy tensor to GPU memory
3. `device='cuda'` - Create tensor directly on GPU
4. `@` - Matrix multiplication happens on GPU (fast!)
5. `.to('cpu')` - Copy back to CPU (for NumPy, plotting, etc.)

---

### Device Management Pattern

```python
# Best practice: Use device variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create tensor on appropriate device
x = torch.randn(100, 100, device=device)
y = torch.randn(100, 100, device=device)

# All operations run on the same device
z = x @ y  # Fast on GPU, works on CPU
```

**Why?** Code works on both CPU and GPU without modification!

---

### Performance Comparison

```python
import time

# Large matrix
size = 5000

# CPU computation
x_cpu = torch.randn(size, size)
y_cpu = torch.randn(size, size)

start = time.time()
z_cpu = x_cpu @ y_cpu
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f} seconds")

# GPU computation (if available)
if torch.cuda.is_available():
    x_gpu = x_cpu.to('cuda')
    y_gpu = y_cpu.to('cuda')

    # Warm-up (first run is slower)
    _ = x_gpu @ y_gpu

    # Actual timing
    start = time.time()
    z_gpu = x_gpu @ y_gpu
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start

    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")
```

**Expected Results:**
```
CPU time: 12.5000 seconds
GPU time: 0.2500 seconds
Speedup: 50.0x
```

**GPU is ~10-100x faster for large matrices!**

---

## Part 5: Converting Between NumPy and PyTorch

### NumPy to PyTorch

```python
import numpy as np
import torch

# NumPy array
np_array = np.array([[1, 2], [3, 4]])

# Convert to PyTorch tensor
tensor = torch.from_numpy(np_array)
print(tensor)
# tensor([[1, 2],
#         [3, 4]])

# Note: Shares memory! (changes affect both)
np_array[0, 0] = 999
print(tensor)  # tensor([999, 2], [3, 4]) - changed!
```

**Memory Sharing:**
- `torch.from_numpy()` - Shares memory (fast, but changes affect both)
- `torch.tensor(np_array)` - Copies memory (safe, but slower)

---

### PyTorch to NumPy

```python
# PyTorch tensor
tensor = torch.tensor([[1, 2], [3, 4]])

# Convert to NumPy array
np_array = tensor.numpy()
print(np_array)
# [[1 2]
#  [3 4]]

# For GPU tensors, must move to CPU first
if tensor.is_cuda:
    np_array = tensor.cpu().numpy()
```

**Important:** Can't convert GPU tensors directly to NumPy (NumPy only works on CPU).

---

## Summary

### Key Concepts

**Tensors:**
- Multi-dimensional arrays (like NumPy)
- Created with `torch.tensor()`, `torch.zeros()`, `torch.randn()`, etc.
- Properties: `shape`, `dtype`, `device`

**Operations:**
- Element-wise: `+`, `-`, `*`, `/`
- Matrix: `@` or `torch.matmul()`
- Indexing: Same as NumPy

**Autograd:**
- `requires_grad=True` - Track operations
- `y.backward()` - Compute gradients
- `x.grad` - Access gradients
- `x.grad.zero_()` - Reset gradients

**GPU:**
- `torch.cuda.is_available()` - Check GPU
- `.to('cuda')` - Move to GPU
- `.to('cpu')` - Move to CPU
- Much faster for large computations!

**NumPy Conversion:**
- NumPy → PyTorch: `torch.from_numpy()`
- PyTorch → NumPy: `tensor.numpy()`

---

## Quick Reference

| Operation | NumPy | PyTorch |
|-----------|-------|---------|
| Create array/tensor | `np.array([1, 2])` | `torch.tensor([1, 2])` |
| Zeros | `np.zeros((3, 4))` | `torch.zeros(3, 4)` |
| Random | `np.random.randn(2, 2)` | `torch.randn(2, 2)` |
| Matrix multiply | `A @ B` | `A @ B` |
| Transpose | `A.T` | `A.T` |
| Shape | `A.shape` | `A.shape` or `A.size()` |
| Reshape | `A.reshape(2, 3)` | `A.view(2, 3)` |
| Slice | `A[:, 0]` | `A[:, 0]` |
| To device | N/A | `A.to('cuda')` |
| Gradients | Manual | `A.backward()` |

---

## Quiz

### Question 1
What's the difference between `torch.tensor()` and `torch.Tensor()`?

<details>
<summary>Answer</summary>

- `torch.tensor()` - Creates tensor from data, infers dtype
- `torch.Tensor()` - Creates tensor with default dtype (float32)

```python
a = torch.tensor([1, 2, 3])      # dtype: int64 (inferred)
b = torch.Tensor([1, 2, 3])      # dtype: float32 (default)
```

</details>

### Question 2
What does `requires_grad=True` do?

<details>
<summary>Answer</summary>

Tells PyTorch to track all operations on this tensor so gradients can be computed later using `.backward()`.

Without it, PyTorch doesn't record operations (saves memory).

</details>

### Question 3
Why must gradients be zeroed between training iterations?

<details>
<summary>Answer</summary>

Gradients accumulate by default in PyTorch. If not zeroed, gradients from previous iterations add to current gradients, giving wrong results.

```python
optimizer.zero_grad()  # Always do this before backward!
loss.backward()
optimizer.step()
```

</details>

### Question 4
Can you convert a GPU tensor directly to NumPy?

<details>
<summary>Answer</summary>

No! You must move it to CPU first:

```python
# Wrong
np_array = gpu_tensor.numpy()  # Error!

# Correct
np_array = gpu_tensor.cpu().numpy()  # Works!
```

NumPy only works with CPU memory.

</details>

---

## Lab Exercises

### Exercise 1: Tensor Basics

```python
# TODO: Create the following tensors
# 1. A 3x3 matrix of zeros
# 2. A 2x4 matrix of ones
# 3. A 5x5 identity matrix (hint: torch.eye)
# 4. A random 3x3 matrix with values from normal distribution
# 5. A range from 0 to 20 with step 2

# Your code here:
```

<details>
<summary>Solution</summary>

```python
# 1. 3x3 zeros
zeros = torch.zeros(3, 3)

# 2. 2x4 ones
ones = torch.ones(2, 4)

# 3. 5x5 identity
identity = torch.eye(5)

# 4. 3x3 random normal
randn = torch.randn(3, 3)

# 5. Range 0 to 20, step 2
range_tensor = torch.arange(0, 21, 2)  # [0, 2, 4, ..., 20]
```

</details>

---

### Exercise 2: Tensor Operations

```python
# Given tensors
A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# TODO: Compute the following
# 1. Element-wise sum: A + B
# 2. Element-wise product: A * B
# 3. Matrix product: A @ B
# 4. Transpose of A
# 5. Mean of all elements in B

# Your code here:
```

<details>
<summary>Solution</summary>

```python
# 1. Element-wise sum
sum_result = A + B
# tensor([[ 6.,  8.],
#         [10., 12.]])

# 2. Element-wise product
product = A * B
# tensor([[ 5., 12.],
#         [21., 32.]])

# 3. Matrix product
matmul = A @ B
# tensor([[19., 22.],
#         [43., 50.]])

# 4. Transpose
transpose = A.T
# tensor([[1., 3.],
#         [2., 4.]])

# 5. Mean
mean = B.mean()
# tensor(6.5000)
```

</details>

---

### Exercise 3: Automatic Differentiation

```python
# TODO: Compute gradients for the following
# Function: y = 3x^2 + 2x + 1
# At x = 2.0

# 1. Create x as a tensor with requires_grad=True
# 2. Compute y
# 3. Compute gradient dy/dx using backward()
# 4. Verify: dy/dx = 6x + 2 = 6*2 + 2 = 14

# Your code here:
```

<details>
<summary>Solution</summary>

```python
# Step 1: Create x
x = torch.tensor([2.0], requires_grad=True)

# Step 2: Compute y
y = 3 * x**2 + 2 * x + 1
print(f"y = {y.item()}")  # 3*4 + 2*2 + 1 = 17

# Step 3: Compute gradient
y.backward()

# Step 4: Check gradient
print(f"dy/dx = {x.grad.item()}")  # 14.0 ✓

# Verify manually: dy/dx = 6x + 2 = 6*2 + 2 = 14
```

</details>

---

### Exercise 4: NumPy to PyTorch

```python
import numpy as np

# Given NumPy array
np_array = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=np.float32)

# TODO:
# 1. Convert to PyTorch tensor
# 2. Compute the mean of each row
# 3. Add 10 to all elements
# 4. Convert back to NumPy

# Your code here:
```

<details>
<summary>Solution</summary>

```python
# 1. Convert to PyTorch
tensor = torch.from_numpy(np_array)
print(tensor)

# 2. Mean of each row
row_means = tensor.mean(dim=1)  # dim=1 means along columns
print(row_means)  # tensor([2., 5., 8.])

# 3. Add 10
tensor_plus_10 = tensor + 10
print(tensor_plus_10)

# 4. Convert back to NumPy
result = tensor_plus_10.numpy()
print(result)
```

</details>

---

## Next Steps

**Congratulations!** You've learned PyTorch fundamentals:
✅ Tensors and operations
✅ Automatic differentiation
✅ GPU acceleration
✅ NumPy integration

**Next Lesson:** Building Neural Networks in PyTorch (`02_pytorch_neural_networks.md`)

You'll learn:
- `nn.Module` - Base class for all models
- Built-in layers (`nn.Linear`, `nn.Conv2d`)
- Training loop patterns
- Real MNIST classifier!

---

**Practice makes perfect! Run the examples and complete the exercises before moving on.**

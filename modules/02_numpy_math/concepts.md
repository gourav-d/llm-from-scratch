# NumPy Concepts - Visual Guide

## 🧠 Understanding NumPy Through Visuals

This guide explains NumPy concepts using ASCII diagrams and real-world analogies.

---

## 1. What is a NumPy Array?

### Python List (Scattered Memory)
```
Python List:  [ptr]───►[object: 1]
              [ptr]───────────►[object: 2]
              [ptr]─────────────────►[object: 3]
                     ↑
               Memory is SCATTERED
               Each element is a full Python object
               SLOW for numerical operations
```

### NumPy Array (Contiguous Memory)
```
NumPy Array:  [1][2][3][4][5][6][7][8]
              ↑
          Contiguous block of memory
          All same data type (e.g., int64)
          FAST - CPU can process in parallel
```

**In .NET terms:**
- Python List ≈ `List<object>` (boxed, scattered)
- NumPy Array ≈ `Span<T>` or `T[]` (contiguous, unboxed)

---

## 2. Array Dimensions (ndim)

### 0D Array (Scalar)
```
42
```
Just a single number. Shape: `()`

### 1D Array (Vector)
```
[1, 2, 3, 4, 5]
 ↑  ↑  ↑  ↑  ↑
indices: 0, 1, 2, 3, 4
```
Like a row of numbers. Shape: `(5,)`

**Real-world:** List of temperatures, test scores, stock prices

### 2D Array (Matrix)
```
      col 0  col 1  col 2
row 0  [ 1     2      3  ]
row 1  [ 4     5      6  ]
row 2  [ 7     8      9  ]

Shape: (3, 3) - 3 rows, 3 columns
```

**Real-world:**
- Spreadsheet
- Grayscale image (pixels)
- Weight matrix in neural network

### 3D Array (Tensor)
```
Think of it as layers of 2D arrays:

Layer 0:        Layer 1:        Layer 2:
[1  2  3]       [10 11 12]      [19 20 21]
[4  5  6]       [13 14 15]      [22 23 24]
[7  8  9]       [16 17 18]      [25 26 27]

Shape: (3, 3, 3)
       ↑  ↑  ↑
     layers rows cols
```

**Real-world:**
- RGB Image: (height, width, 3) - 3 color channels
- Video: (frames, height, width)
- Batch of images: (batch_size, height, width)

### 4D Array (Common in Deep Learning)
```
Batch of RGB images:

Batch 0:              Batch 1:              Batch 2:
[R channel 28x28]    [R channel 28x28]    [R channel 28x28]
[G channel 28x28]    [G channel 28x28]    [G channel 28x28]
[B channel 28x28]    [B channel 28x28]    [B channel 28x28]

Shape: (3, 28, 28, 3)
        ↑  ↑   ↑   ↑
     batch H   W  channels

Or in PyTorch/TensorFlow:
Shape: (3, 3, 28, 28)
        ↑  ↑  ↑   ↑
     batch ch  H   W
```

---

## 3. Shape and Size

```python
arr = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
```

**Visual representation:**
```
Depth 0:     Depth 1:
[1  2]       [5  6]
[3  4]       [7  8]

shape = (2, 2, 2)
         ↑  ↑  ↑
      depth rows cols

ndim = 3 (3 dimensions)
size = 8 (total elements)
```

---

## 4. Indexing and Slicing

### 1D Indexing
```
arr = [10, 20, 30, 40, 50]
       ↑   ↑   ↑   ↑   ↑
idx:   0   1   2   3   4
idx:  -5  -4  -3  -2  -1  (negative indexing)

arr[0]    = 10
arr[-1]   = 50
arr[1:4]  = [20, 30, 40]
arr[::2]  = [10, 30, 50]  (every 2nd element)
```

### 2D Indexing
```
matrix = [[10, 20, 30],
          [40, 50, 60],
          [70, 80, 90]]

matrix[1, 2] = 60
          ↑  ↑
        row col

matrix[0, :]  = [10, 20, 30]  ← whole row 0
matrix[:, 0]  = [10, 40, 70]  ← whole column 0

matrix[0:2, 1:3] = [[20, 30],  ← submatrix
                    [50, 60]]
```

### Slicing Syntax: `start:stop:step`
```
arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

arr[2:7]    = [2, 3, 4, 5, 6]     from index 2 to 6
arr[:5]     = [0, 1, 2, 3, 4]     from start to index 4
arr[5:]     = [5, 6, 7, 8, 9]     from index 5 to end
arr[::2]    = [0, 2, 4, 6, 8]     every 2nd element
arr[::-1]   = [9, 8, 7, ..., 0]   reverse!
```

---

## 5. Reshaping

### Flattening (2D → 1D)
```
Original (3x4):
[1  2  3  4]       Flattened (12,):
[5  6  7  8]   →   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
[9 10 11 12]
```

**Use case:** Feeding images into neural network input layer

### Reshaping Rules
```
arr = np.arange(12)  # [0, 1, 2, ..., 11]

reshape(3, 4):       reshape(2, 6):      reshape(4, 3):
[0  1  2  3]        [0 1 2 3 4 5]       [0  1  2]
[4  5  6  7]        [6 7 8 9 10 11]     [3  4  5]
[8  9 10 11]                            [6  7  8]
                                        [9 10 11]

Rule: Product of dimensions must equal total elements
      3 × 4 = 12 ✓
      2 × 6 = 12 ✓
      4 × 3 = 12 ✓
      3 × 5 = 15 ✗ (only have 12 elements!)
```

### Auto-dimension with -1
```python
arr.reshape(3, -1)  # 3 rows, auto-calculate columns
                    # -1 = 12/3 = 4 columns
                    # Result: (3, 4)

arr.reshape(-1, 2)  # Auto rows, 2 columns
                    # -1 = 12/2 = 6 rows
                    # Result: (6, 2)
```

---

## 6. Broadcasting

### Rule: Automatically expand smaller array to match larger one

### Example 1: Scalar Broadcasting
```
arr =    [1, 2, 3, 4]
      +          10

Broadcasting expands 10 to:
         [10, 10, 10, 10]

Result = [11, 12, 13, 14]
```

### Example 2: Vector + Matrix (Row-wise)
```
matrix =  [1  2  3]
          [4  5  6]

vector =  [10 20 30]  ← broadcast to each row

Result =  [1+10  2+20  3+30]  =  [11 22 33]
          [4+10  5+20  6+30]     [14 25 36]
```

### Example 3: Column Broadcasting
```
matrix =  [1  2  3]
          [4  5  6]

column =  [10]  ← broadcast to each column
          [20]

Result =  [1+10  2+10  3+10]  =  [11 12 13]
          [4+20  5+20  6+20]     [24 25 26]
```

### Broadcasting Rules (Dimensions Compatibility)
```
Compatible shapes (broadcasting works):
(3, 1) and (1, 4)  → result: (3, 4)
(5, 1) and (5, 3)  → result: (5, 3)
(2, 3) and (1,)    → result: (2, 3)

Incompatible (broadcasting fails):
(3, 4) and (5,)    ✗ - dimensions don't align
(2, 3) and (3, 2)  ✗ - neither can be broadcast
```

---

## 7. Vectorization (The Key to Speed!)

### Without Vectorization (Slow)
```python
# Python loop - slow
result = []
for i in range(len(arr)):
    result.append(arr[i] * 2)

CPU: Process one element → Process next → Process next...
     [1] → [2] → [3] → [4] → ...
```

### With Vectorization (Fast)
```python
# NumPy vectorized - fast!
result = arr * 2

CPU (with SIMD): Process multiple elements in parallel!
     [1, 2, 3, 4] → [2, 4, 6, 8]  (all at once!)
```

**Speed comparison:**
```
Python loop:     ████████████████████████████ (28 units)
NumPy vectorized: █ (1 unit)

50-100x faster!
```

---

## 8. Matrix Operations

### Element-wise vs Matrix Multiplication

#### Element-wise (`*`)
```
A = [1  2]    B = [5  6]
    [3  4]        [7  8]

A * B = [1×5  2×6]  =  [5   12]
        [3×7  4×8]     [21  32]
```

#### Matrix Multiplication (`@`)
```
A = [1  2]    B = [5  6]
    [3  4]        [7  8]

A @ B = [1×5+2×7  1×6+2×8]  =  [19  22]
        [3×5+4×7  3×6+4×8]     [43  50]
```

**Calculation detail:**
```
Position (0,0): 1×5 + 2×7 = 5 + 14 = 19
Position (0,1): 1×6 + 2×8 = 6 + 16 = 22
Position (1,0): 3×5 + 4×7 = 15 + 28 = 43
Position (1,1): 3×6 + 4×8 = 18 + 32 = 50
```

### Shape Compatibility for Matrix Multiplication
```
(A @ B) only works if:
     ↓
(m, n) @ (n, p) = (m, p)
     ↑____↑
   must match!

Examples:
(2, 3) @ (3, 4) = (2, 4) ✓
(5, 2) @ (2, 7) = (5, 7) ✓
(3, 4) @ (5, 2) = Error!  ✗ (4 ≠ 5)
```

---

## 9. Neural Network Forward Pass (Putting it Together!)

### Single Layer
```
Input (X):        Weights (W):      Bias (b):
[x1  x2  x3]  @   [w11  w21]   +    [b1  b2]
(1, 3)            [w12  w22]        (2,)
                  [w13  w23]
                  (3, 2)

Shape math:
(1, 3) @ (3, 2) + (2,) = (1, 2)

Result (output): [y1  y2]
```

**Step-by-step calculation:**
```
y1 = x1×w11 + x2×w12 + x3×w13 + b1
y2 = x1×w21 + x2×w22 + x3×w23 + b2
```

### Batch Processing
```
Batch of 32 samples, 784 features each:
X shape: (32, 784)

Layer 1 weights (784 → 128 neurons):
W1 shape: (784, 128)
b1 shape: (128,)

Output of layer 1:
Z1 = X @ W1 + b1
Shape: (32, 128)

This computed 32 samples × 128 neurons = 4,096 values
ALL AT ONCE using matrix multiplication!
```

---

## 10. Transpose Visualization

### Simple Transpose
```
Original:         Transposed:
[1  2  3]         [1  4]
[4  5  6]    →    [2  5]
                  [3  6]

(2, 3) → (3, 2)
```

**Rule:** Swap rows and columns
- Element at (i, j) moves to (j, i)

### Why Transpose in Neural Networks?
```
# Attention mechanism (simplified)

Query (Q): (batch, seq_len, d_k)
Key (K):   (batch, seq_len, d_k)

We want: Q @ K^T
         (batch, seq_len, d_k) @ (batch, d_k, seq_len)
                            ↑_________↑
                         K.transpose(1, 2)

Result: (batch, seq_len, seq_len) - attention scores!
```

---

## 11. Common Patterns in LLMs

### Token Embedding Lookup
```
Vocabulary: 50,000 words
Embedding dim: 768

Embedding matrix shape: (50,000, 768)

       word_id
         ↓
    [embedding_0]  ← "the"
    [embedding_1]  ← "a"
    [embedding_2]  ← "is"
    ...
    [embedding_15496] ← "hello"
    ...
    [embedding_49999] ← "zebra"

Each row is a 768-dimensional vector!
```

### Batch Processing
```
Sentence: "hello world"
Token IDs: [15496, 732]

Lookup embeddings:
embeddings[[15496, 732]]
         ↓
Result shape: (2, 768)
              [embedding for "hello"]
              [embedding for "world"]
```

### Attention Scores
```
Query × Key^T = Attention weights

(batch, seq_len, d_k) @ (batch, d_k, seq_len)
         = (batch, seq_len, seq_len)

This tells us: "Which words should pay attention to which other words?"
```

---

## 12. Common Mistakes & How to Avoid

### Mistake 1: Forgetting Parentheses in Shape
```python
❌ np.zeros(3, 3)      # Error!
✓  np.zeros((3, 3))   # Correct
```

### Mistake 2: List Concatenation vs Element-wise Addition
```python
# Python lists
list1 + list2  # Concatenates!

# NumPy arrays
arr1 + arr2    # Element-wise addition!
```

### Mistake 3: Shape Mismatch
```python
A = np.random.randn(3, 4)
B = np.random.randn(5, 2)
C = A @ B  # ❌ Error: 4 ≠ 5

# Fix: Check inner dimensions match
A = np.random.randn(3, 4)
B = np.random.randn(4, 2)  # 4 matches!
C = A @ B  # ✓ Result: (3, 2)
```

### Mistake 4: Modifying Original Array
```python
arr = np.array([1, 2, 3])
view = arr[:]        # This is a view, not a copy!
view[0] = 999        # Changes original!

# Use .copy() for independent array
copy = arr.copy()    # This is a copy
copy[0] = 999        # Original unchanged
```

---

## 13. Debugging Tips

### Always Print Shapes
```python
print(f"X shape: {X.shape}")
print(f"W shape: {W.shape}")
print(f"output shape: {(X @ W).shape}")
```

### Use Small Test Arrays
```python
# Start with small 2x2 or 3x3 matrices
# Verify logic before scaling up
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)  # Can calculate by hand to verify
```

### Visualize Your Data
```python
import matplotlib.pyplot as plt

# For 2D arrays
plt.imshow(matrix, cmap='viridis')
plt.colorbar()
plt.show()

# For 1D arrays
plt.plot(vector)
plt.show()
```

---

## Summary: Key Mental Models

1. **Arrays = Contiguous Memory** (like C arrays, not Python lists)
2. **Shape = Tuple of Dimensions** (read it like: "layers, rows, columns...")
3. **Broadcasting = Auto-expansion** (smaller array stretched to match larger)
4. **Vectorization = Parallel Processing** (all elements at once, not loops)
5. **`@` for matrices, `*` for elements** (@ is neural networks!)
6. **Reshape = Rearrange, not change data** (total elements stay same)

**Next:** Apply these concepts in `examples/` and `exercises/`!

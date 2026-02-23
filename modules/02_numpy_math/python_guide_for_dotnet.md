# NumPy for .NET Developers

## 🎯 NumPy Concepts Mapped to C#/.NET

This guide helps you understand NumPy by relating it to familiar .NET concepts.

---

## 1. Arrays and Collections

### Basic Arrays

**C#:**
```csharp
// Fixed-size array
int[] numbers = new int[] { 1, 2, 3, 4, 5 };

// Multi-dimensional array
int[,] matrix = new int[,] {
    { 1, 2, 3 },
    { 4, 5, 6 }
};

// Getting length
int length = numbers.Length;  // 5
int rows = matrix.GetLength(0);  // 2
int cols = matrix.GetLength(1);  // 3
```

**NumPy:**
```python
# 1D array
numbers = np.array([1, 2, 3, 4, 5])

# 2D array
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Getting shape
length = numbers.shape[0]  # 5
shape = matrix.shape  # (2, 3)
rows, cols = matrix.shape  # 2, 3
```

---

## 2. Memory Layout

### Contiguous Memory

**C# (Span<T> - similar to NumPy):**
```csharp
// Contiguous memory, stack-allocated
Span<int> numbers = stackalloc int[] { 1, 2, 3, 4, 5 };

// Or heap-allocated
int[] array = { 1, 2, 3, 4, 5 };
Span<int> span = array.AsSpan();

// Slicing (no copy!)
Span<int> slice = span.Slice(1, 3);  // [2, 3, 4]
```

**NumPy:**
```python
# Always contiguous memory
numbers = np.array([1, 2, 3, 4, 5])

# Slicing (returns view, no copy!)
slice = numbers[1:4]  # [2, 3, 4]
```

**Key similarity:** Both avoid copying data when slicing!

---

## 3. LINQ vs Vectorized Operations

### Element-wise Operations

**C# (LINQ):**
```csharp
int[] numbers = { 1, 2, 3, 4, 5 };

// Double each element (creates new array)
int[] doubled = numbers.Select(x => x * 2).ToArray();
// [2, 4, 6, 8, 10]

// Filter
int[] filtered = numbers.Where(x => x > 3).ToArray();
// [4, 5]

// Sum
int sum = numbers.Sum();  // 15

// Average
double avg = numbers.Average();  // 3.0
```

**NumPy:**
```python
numbers = np.array([1, 2, 3, 4, 5])

# Double each element (vectorized!)
doubled = numbers * 2
# [2, 4, 6, 8, 10]

# Filter
filtered = numbers[numbers > 3]
# [4, 5]

# Sum
sum = numbers.sum()  # 15

# Average
avg = numbers.mean()  # 3.0
```

**Key difference:** NumPy is 50-100x FASTER because operations are vectorized (run at C-speed, not Python-speed)

---

## 4. Vectorization (SIMD)

### Using System.Numerics.Vector<T>

**C# (SIMD):**
```csharp
using System.Numerics;

// Must be power of 2 length for SIMD
var a = new Vector<float>(new float[] { 1, 2, 3, 4 });
var b = new Vector<float>(new float[] { 5, 6, 7, 8 });

// Vectorized addition (parallel)
var result = a + b;  // [6, 8, 10, 12]
```

**NumPy (Always Vectorized):**
```python
# Automatically uses SIMD when possible
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = np.array([5, 6, 7, 8], dtype=np.float32)

# Vectorized addition
result = a + b  # [6, 8, 10, 12]
```

**Similarity:** Both use CPU SIMD instructions for parallelism!

---

## 5. Indexing and Slicing

### Array Access

**C#:**
```csharp
int[] arr = { 10, 20, 30, 40, 50 };

// Single element
int first = arr[0];      // 10
int last = arr[^1];      // 50 (C# 8.0+)

// Range/Slice (C# 8.0+)
int[] slice = arr[1..4]; // [20, 30, 40]
int[] from2 = arr[2..];  // [30, 40, 50]

// Multi-dimensional
int[,] matrix = { {1,2,3}, {4,5,6} };
int element = matrix[1, 2];  // 6
```

**NumPy:**
```python
arr = np.array([10, 20, 30, 40, 50])

# Single element
first = arr[0]      # 10
last = arr[-1]      # 50

# Slicing
slice = arr[1:4]    # [20, 30, 40]
from2 = arr[2:]     # [30, 40, 50]

# Multi-dimensional
matrix = np.array([[1, 2, 3], [4, 5, 6]])
element = matrix[1, 2]  # 6
```

**Very similar!** NumPy slicing inspired C# 8.0+ ranges!

---

## 6. LINQ Query Expressions vs Boolean Indexing

### Filtering

**C# (LINQ):**
```csharp
int[] scores = { 45, 67, 89, 92, 56, 78 };

// Filter with predicate
var passing = scores.Where(s => s >= 60).ToArray();
// [67, 89, 92, 78]

// Count matching
int count = scores.Count(s => s >= 80);  // 2

// Any/All
bool anyFailing = scores.Any(s => s < 60);  // true
bool allPassing = scores.All(s => s >= 60); // false
```

**NumPy (Boolean Indexing):**
```python
scores = np.array([45, 67, 89, 92, 56, 78])

# Filter with mask
passing = scores[scores >= 60]
# [67, 89, 92, 78]

# Count matching
count = (scores >= 80).sum()  # 2

# Any/All
any_failing = (scores < 60).any()  # True
all_passing = (scores >= 60).all()  # False
```

**Similarity:** Both support declarative filtering!
**Difference:** NumPy creates boolean mask first, then filters

---

## 7. IEnumerable vs NumPy Aggregations

### Statistics

**C# (LINQ):**
```csharp
using System.Linq;

double[] data = { 1.5, 2.7, 3.2, 4.8, 5.1 };

double sum = data.Sum();                    // 17.3
double average = data.Average();            // 3.46
double max = data.Max();                    // 5.1
double min = data.Min();                    // 1.5
int count = data.Count();                   // 5

// No built-in std dev - need custom
double variance = data.Select(x => Math.Pow(x - average, 2))
                      .Average();
double stdDev = Math.Sqrt(variance);
```

**NumPy:**
```python
data = np.array([1.5, 2.7, 3.2, 4.8, 5.1])

sum = data.sum()          # 17.3
average = data.mean()     # 3.46
max = data.max()          # 5.1
min = data.min()          # 1.5
count = data.size         # 5

# Built-in statistical functions
variance = data.var()     # Built-in!
std_dev = data.std()      # Built-in!
```

**NumPy advantage:** More statistical functions out of the box

---

## 8. Jagged Arrays vs Rectangular Arrays

### Multi-dimensional Arrays

**C# (Jagged - different row lengths):**
```csharp
// Jagged array (array of arrays)
int[][] jagged = new int[][] {
    new int[] { 1, 2, 3 },
    new int[] { 4, 5 },
    new int[] { 6, 7, 8, 9 }
};

// Different row lengths allowed!
```

**C# (Rectangular - fixed shape):**
```csharp
// Multi-dimensional array (fixed shape)
int[,] rectangular = new int[,] {
    { 1, 2, 3 },
    { 4, 5, 6 }
};

// All rows MUST be same length
```

**NumPy (Always Rectangular):**
```python
# NumPy arrays are always rectangular
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Cannot create jagged arrays directly
# This will fail:
# np.array([[1, 2], [3, 4, 5]])  # Error!
```

**Use case:** NumPy's rectangular requirement enables fast vectorized operations

---

## 9. Matrix Operations

### Linear Algebra

**C# (Manual Implementation):**
```csharp
// No built-in matrix multiplication!
// Need to implement or use library like Math.NET

public static double[,] MatrixMultiply(double[,] a, double[,] b) {
    int aRows = a.GetLength(0);
    int aCols = a.GetLength(1);
    int bCols = b.GetLength(1);

    var result = new double[aRows, bCols];

    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bCols; j++) {
            for (int k = 0; k < aCols; k++) {
                result[i, j] += a[i, k] * b[k, j];
            }
        }
    }
    return result;
}
```

**C# (Math.NET Numerics):**
```csharp
using MathNet.Numerics.LinearAlgebra;

var A = Matrix<double>.Build.DenseOfArray(new double[,] {
    { 1, 2 },
    { 3, 4 }
});

var B = Matrix<double>.Build.DenseOfArray(new double[,] {
    { 5, 6 },
    { 7, 8 }
});

var C = A * B;  // Matrix multiplication
```

**NumPy (Built-in):**
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise
C = A * B  # [[5, 12], [21, 32]]

# Matrix multiplication
C = A @ B  # [[19, 22], [43, 50]]
# Or: C = np.dot(A, B)
```

**NumPy advantage:** Matrix operations built-in and optimized

---

## 10. Parallel Processing

### Concurrent Operations

**C# (Parallel LINQ):**
```csharp
int[] numbers = Enumerable.Range(0, 1000000).ToArray();

// Parallel processing
var doubled = numbers.AsParallel()
                     .Select(x => x * 2)
                     .ToArray();

// Uses multiple CPU cores
```

**NumPy (Automatic SIMD):**
```python
numbers = np.arange(1000000)

# Automatically uses SIMD (single core, but very fast)
doubled = numbers * 2

# For multi-core, combine with libraries like:
# - Dask (parallel NumPy)
# - joblib (parallel processing)
```

**Difference:**
- C# PLINQ: Multi-threaded parallelism
- NumPy: SIMD parallelism (single core, but vectorized)
- Both are fast, different approaches

---

## 11. Reshaping and Transformations

### Data Shape Manipulation

**C# (Manual):**
```csharp
// Flatten 2D to 1D
int[,] matrix = { {1,2,3}, {4,5,6} };
int[] flattened = new int[6];
int index = 0;
for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
        flattened[index++] = matrix[i, j];
    }
}
// [1, 2, 3, 4, 5, 6]

// Reshape 1D to 2D
int[,] reshaped = new int[2, 3];
index = 0;
for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
        reshaped[i, j] = flattened[index++];
    }
}
```

**NumPy (Built-in):**
```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Flatten
flattened = matrix.flatten()  # [1, 2, 3, 4, 5, 6]
# Or: flattened = matrix.reshape(-1)

# Reshape
reshaped = flattened.reshape(2, 3)
# [[1, 2, 3],
#  [4, 5, 6]]
```

**NumPy advantage:** One-liners for common transformations

---

## 12. Performance Comparison

### Benchmark: Sum 1M elements

**C# (LINQ):**
```csharp
var numbers = Enumerable.Range(0, 1000000).ToArray();
var sw = Stopwatch.StartNew();
long sum = numbers.Sum();
sw.Stop();
Console.WriteLine($"C# LINQ: {sw.ElapsedMilliseconds}ms");
// ~3-5ms
```

**C# (For Loop):**
```csharp
long sum = 0;
var sw = Stopwatch.StartNew();
for (int i = 0; i < numbers.Length; i++) {
    sum += numbers[i];
}
sw.Stop();
Console.WriteLine($"C# Loop: {sw.ElapsedMilliseconds}ms");
// ~1-2ms (faster than LINQ)
```

**NumPy:**
```python
import time
numbers = np.arange(1000000)
start = time.time()
total = numbers.sum()
elapsed = (time.time() - start) * 1000
print(f"NumPy: {elapsed:.2f}ms")
# ~0.1-0.3ms (fastest!)
```

**Why NumPy is fastest:**
- Contiguous memory
- SIMD instructions
- No Python overhead (C implementation)

---

## 13. Type System

### Data Types

**C#:**
```csharp
int[] ints = { 1, 2, 3 };           // System.Int32
double[] doubles = { 1.0, 2.0 };    // System.Double
float[] floats = { 1f, 2f };        // System.Single
bool[] bools = { true, false };     // System.Boolean

// Explicit casting
double[] asDoubles = ints.Select(x => (double)x).ToArray();
```

**NumPy:**
```python
ints = np.array([1, 2, 3])              # dtype: int64
doubles = np.array([1.0, 2.0])          # dtype: float64
floats = np.array([1.0, 2.0], dtype=np.float32)
bools = np.array([True, False])         # dtype: bool

# Type conversion
as_doubles = ints.astype(np.float64)
```

**NumPy type names:**
- `np.int32` ↔ C# `int`
- `np.int64` ↔ C# `long`
- `np.float32` ↔ C# `float`
- `np.float64` ↔ C# `double`
- `np.bool_` ↔ C# `bool`

---

## 14. Common Patterns Comparison

### Pattern 1: Normalizing Data

**C#:**
```csharp
double[] data = { 1, 2, 3, 4, 5 };
double mean = data.Average();
double variance = data.Select(x => Math.Pow(x - mean, 2))
                      .Average();
double stdDev = Math.Sqrt(variance);

double[] normalized = data.Select(x => (x - mean) / stdDev)
                          .ToArray();
```

**NumPy:**
```python
data = np.array([1, 2, 3, 4, 5])
mean = data.mean()
std_dev = data.std()

normalized = (data - mean) / std_dev  # Vectorized!
```

### Pattern 2: Conditional Assignment

**C#:**
```csharp
int[] scores = { 85, 92, 78, 95, 88 };
string[] grades = scores.Select(s =>
    s >= 90 ? "A" :
    s >= 80 ? "B" :
    s >= 70 ? "C" : "F"
).ToArray();
```

**NumPy:**
```python
scores = np.array([85, 92, 78, 95, 88])
grades = np.where(scores >= 90, "A",
         np.where(scores >= 80, "B",
         np.where(scores >= 70, "C", "F")))
```

---

## 15. When to Use What?

### Use C# Collections When:
- Working with .NET objects (not just numbers)
- Need jagged arrays (different row lengths)
- Building general-purpose applications
- Type safety and IntelliSense are critical

### Use NumPy When:
- Heavy numerical computation
- Matrix/vector operations
- Machine learning / AI
- Image processing
- Scientific computing
- Need maximum speed for numerical data

### Best of Both Worlds:
Use **Math.NET Numerics** in C# for NumPy-like capabilities:
```csharp
using MathNet.Numerics.LinearAlgebra;

var A = Matrix<double>.Build.Random(100, 784);
var W = Matrix<double>.Build.Random(784, 128);
var output = A * W;  // Fast matrix multiplication
```

---

## Summary Table

| Feature | C# / .NET | NumPy |
|---------|-----------|-------|
| **Array Creation** | `new int[] {...}` | `np.array([...])` |
| **Multi-dimensional** | `int[,]` | `np.array([[]])` |
| **Slicing** | `arr[1..4]` (C# 8+) | `arr[1:4]` |
| **Element-wise ops** | LINQ `.Select()` | Vectorized `* + -` |
| **Filtering** | `.Where()` | Boolean indexing |
| **Aggregates** | `.Sum()` `.Average()` | `.sum()` `.mean()` |
| **Matrix mult** | Math.NET or manual | Built-in `@` |
| **Performance** | Good | Excellent (50-100x) |
| **Use case** | General programming | Numerical computing |

---

## Practice: Side-by-Side Examples

Try implementing these in both C# and NumPy:

1. **Filter and transform:** Get squares of even numbers from 1-100
2. **Statistics:** Calculate mean and std dev of random 1000 numbers
3. **Matrix math:** Multiply two 100x100 matrices
4. **Data processing:** Normalize image pixel values (0-255 → 0-1)

This hands-on practice will solidify the concepts!

---

## Next Steps

1. **Experiment:** Try converting your C# numerical code to NumPy
2. **Benchmark:** Compare performance for your use cases
3. **Learn More:** Explore NumPy's advanced features (broadcasting, fancy indexing)
4. **Build:** Use NumPy to implement neural network forward pass

**Remember:** NumPy is to Python what `Span<T>`, SIMD, and Math.NET combined are to C# - but even faster!

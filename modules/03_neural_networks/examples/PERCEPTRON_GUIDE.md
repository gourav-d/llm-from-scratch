# Perceptron Quick Reference Guide

## Your Confusion - ANSWERED!

### 1. Why `errors.sum()` for bias update?

```python
self.bias += self.learning_rate * errors.sum()  # Why sum?
```

**Answer:**

Bias is **not tied to any specific input feature**. It's a constant offset.

**Think of it like this:**
- **Weights** say: "If THIS feature is present, how much does it matter?"
- **Bias** says: "Overall, what's my baseline tendency?"

**The Math:**

For each example, bias contributes:
```
contribution = bias × 1  (always multiplied by 1)
```

When we update, we need to account for **all examples**:
```
bias_update = (error₁ × 1) + (error₂ × 1) + (error₃ × 1) + (error₄ × 1)
            = error₁ + error₂ + error₃ + error₄
            = errors.sum()
```

**C# Equivalent:**
```csharp
// Weights: multiply by actual input values
weights[0] += learningRate * errors.Zip(inputs, (e, x) => e * x.Feature1).Sum();
weights[1] += learningRate * errors.Zip(inputs, (e, x) => e * x.Feature2).Sum();

// Bias: multiply by 1 (constant)
bias += learningRate * errors.Sum();  // Same as multiplying each error by 1
```

---

### 2. Why do we `reshape()`?

```python
prediction = perceptron.forward(inputs.reshape(1, -1))[0]
# Why reshape?
```

**Answer:**

Matrix multiplication requires 2D arrays, but single examples are 1D.

**Visual Explanation:**

```
Original shape (2,)       →  Just a 1D array
[0, 0]

Reshaped to (1, 2)        →  A 2D matrix (1 row, 2 columns)
[[0, 0]]
```

**Why it matters:**

```python
# This FAILS:
weights = np.array([0.5, 0.3])      # Shape: (2,)
inputs = np.array([1, 0])           # Shape: (2,)
result = inputs @ weights           # Works, but not what we want for batches

# This WORKS for consistent matrix multiplication:
weights = np.array([0.5, 0.3])      # Shape: (2,)
inputs = np.array([[1, 0]])         # Shape: (1, 2) - one example
result = inputs @ weights           # Shape: (1,) - one output

# Multiple examples:
inputs = np.array([[1, 0],          # Shape: (4, 2) - four examples
                   [0, 1],
                   [1, 1],
                   [0, 0]])
result = inputs @ weights           # Shape: (4,) - four outputs
```

**The `reshape(1, -1)` breakdown:**
- `1` = I want 1 row
- `-1` = Figure out the columns automatically
- `reshape(1, -1)` on `[a, b]` → `[[a, b]]`

**C# Equivalent:**
```csharp
// 1D array → 2D array for matrix operations
double[] array1D = {1, 0};           // Can't use for matrix math
double[,] array2D = {{1, 0}};        // Can use for matrix math
```

---

### 3. Understanding the `train()` method

**The confusing code:**
```python
def train_step(self, X, y):
    predictions = self.forward(X)
    errors = y - predictions

    # This line is confusing!
    self.weights += self.learning_rate * X.T @ errors

    # This line too!
    self.bias += self.learning_rate * errors.sum()
```

**Breaking it down:**

#### What is `X.T @ errors`?

```python
X = [[0, 0],      # 4 examples, 2 features
     [0, 1],      # Shape: (4, 2)
     [1, 0],
     [1, 1]]

errors = [0, -1, 1, 0]  # 4 errors, Shape: (4,)

X.T = [[0, 0, 1, 1],    # Transposed: 2 features, 4 examples
       [0, 1, 0, 1]]    # Shape: (2, 4)

X.T @ errors = [[0, 0, 1, 1],  @  [0, -1, 1, 0]
                [0, 1, 0, 1]]

Result = [0×0 + 0×(-1) + 1×1 + 1×0,     # First weight update
          0×0 + 1×(-1) + 0×1 + 1×0]     # Second weight update
       = [1, -1]
```

**What this means:**
- First number (1): Feature 1 contributed to 1 net error → increase its weight
- Second number (-1): Feature 2 contributed to -1 net error → decrease its weight

**It's the same as doing:**
```python
weight_update = [0, 0]
for i in range(len(X)):
    weight_update[0] += errors[i] * X[i][0]  # Feature 1
    weight_update[1] += errors[i] * X[i][1]  # Feature 2

# X.T @ errors does this automatically!
```

---

## File Guide

### Which example should I start with?

1. **START HERE:** `example_01_ultra_simple.py`
   - Processes ONE example at a time
   - Shows every calculation step
   - Easiest to understand
   - Run time: ~30 seconds

2. **NEXT:** `example_01_simple_perceptron.py`
   - Processes ALL examples together (batch)
   - Explains matrix operations
   - Connects to the original code
   - Run time: ~1 minute

3. **FINALLY:** `example_01_perceptron.py` (original)
   - Complete implementation
   - Multiple examples (AND, OR, XOR)
   - Visualizations
   - Everything will make sense now!
   - Run time: ~2 minutes

---

## Running the Examples

```bash
# Ultra simple (start here!)
python modules/03_neural_networks/examples/example_01_ultra_simple.py

# Simple with batch processing
python modules/03_neural_networks/examples/example_01_simple_perceptron.py

# Full example with visualizations
python modules/03_neural_networks/examples/example_01_perceptron.py
```

---

## Matrix Operations Cheat Sheet

### Common Shapes

```python
# Single example
x = np.array([1, 0])              # Shape: (2,)     - 1D array
x = x.reshape(1, -1)              # Shape: (1, 2)   - 1 row, 2 columns

# Multiple examples (batch)
X = np.array([[1, 0],
              [0, 1]])            # Shape: (2, 2)   - 2 rows, 2 columns

# Weights
weights = np.array([0.5, 0.3])    # Shape: (2,)     - 1D array

# Predictions
predictions = X @ weights         # Shape: (2,)     - one per example
```

### Transpose

```python
X = [[a, b],
     [c, d]]
# Shape: (2, 2)

X.T = [[a, c],
       [b, d]]
# Shape: (2, 2) - rows become columns
```

### Matrix Multiplication

```python
# Shape rules: (m, n) @ (n, p) = (m, p)

(4, 2) @ (2,) = (4,)    # 4 examples, 2 features → 4 predictions
(2, 4) @ (4,) = (2,)    # Transposed features @ errors → weight updates
```

---

## Visual Summary

```
INPUT → WEIGHT → BIAS → ACTIVATION → OUTPUT
[x₁]    [w₁]      b       step(z)      ŷ
[x₂]    [w₂]

Step 1: z = (x₁×w₁ + x₂×w₂) + b
Step 2: ŷ = 1 if z > 0 else 0
Step 3: error = y - ŷ
Step 4: Update weights and bias

LEARNING HAPPENS IN STEP 4!
```

### Update Rules

```
For each weight:
  new_weight = old_weight + (learning_rate × error × input)

  - If input = 0: No change (feature wasn't active)
  - If input = 1: Change by (learning_rate × error)
  - Positive error: Increase weight
  - Negative error: Decrease weight

For bias:
  new_bias = old_bias + (learning_rate × error)

  - Always updates (bias is always "active")
  - Positive error: Increase bias (more likely to predict 1)
  - Negative error: Decrease bias (less likely to predict 1)
```

---

## Common Mistakes & Fixes

### Mistake 1: Wrong Shape
```python
# ✗ WRONG
x = np.array([1, 0])
prediction = perceptron.forward(x)  # Might fail!

# ✓ CORRECT
x = np.array([1, 0]).reshape(1, -1)
prediction = perceptron.forward(x)
```

### Mistake 2: Accessing Scalar as Array
```python
# ✗ WRONG
result = perceptron.forward(x)  # Returns array [1]
if result == 1:                 # Comparing array to int!

# ✓ CORRECT
result = perceptron.forward(x)
if result[0] == 1:              # Access the scalar value
```

### Mistake 3: Confusing Batch vs Single
```python
# Single example
x = np.array([[1, 0]])          # Shape: (1, 2)
result = perceptron.forward(x)  # Shape: (1,)

# Batch of 4 examples
X = np.array([[1, 0],
              [0, 1],
              [1, 1],
              [0, 0]])          # Shape: (4, 2)
result = perceptron.forward(X)  # Shape: (4,)
```

---

## Connection to C#/.NET

| Python | C# Equivalent | Purpose |
|--------|---------------|---------|
| `np.array([1, 0])` | `new double[] {1, 0}` | 1D array |
| `np.array([[1, 0]])` | `new double[,] {{1, 0}}` | 2D array |
| `X @ weights` | `Matrix.Multiply(X, weights)` | Matrix multiplication |
| `X.T` | `Matrix.Transpose(X)` | Transpose |
| `errors.sum()` | `errors.Sum()` | Sum (LINQ) |
| `reshape(1, -1)` | Convert `double[]` → `double[,]` | Dimension change |

---

## Still Confused?

### Quick Test: Can you answer these?

1. **Why reshape?**
   <details>
   <summary>Answer</summary>
   Matrix multiplication needs 2D arrays. reshape(1, -1) converts 1D to 2D.
   </details>

2. **Why errors.sum() for bias?**
   <details>
   <summary>Answer</summary>
   Bias is multiplied by 1 (constant) for every example, so we sum all errors.
   </details>

3. **What does X.T @ errors do?**
   <details>
   <summary>Answer</summary>
   Calculates how much each feature contributed to the total error across all examples.
   </details>

### Still need help?

Run the ultra-simple example and follow along with a pen and paper. Calculate each step manually for the first epoch. It will click!

---

## Next Steps

1. Run `example_01_ultra_simple.py` and read every line
2. Run `example_01_simple_perceptron.py` to understand batch processing
3. Run `example_01_perceptron.py` (original) - it will make sense now!
4. Try modifying the examples with your own data
5. Move on to Activation Functions (next lesson)

You've got this!

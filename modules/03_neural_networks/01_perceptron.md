# Lesson 3.1: The Perceptron - Your First Neuron

## 🎯 Learning Objectives

By the end of this lesson, you'll be able to:

- Explain what a perceptron is and how it works
- Understand the mathematical formula for a perceptron
- Implement a perceptron from scratch in NumPy
- Train a perceptron using the perceptron learning rule
- Understand the limitations of single neurons
- Connect perceptrons to modern neural networks

---

## 🤔 What is a Perceptron?

### The Simplest Possible Neural Network

A **perceptron** is a single artificial neuron. It's the building block of all neural networks, including GPT!

Think of it as a tiny decision-maker:

```
Inputs → [Perceptron] → Output (Yes/No decision)
```

### Real-World Analogy

**Hiring Decision:**

```
Inputs:
- Years of experience: 5
- Education level: Bachelor's
- Test score: 85

Perceptron weighs these factors:
- Experience: Important (weight = 0.4)
- Education: Somewhat important (weight = 0.3)
- Test score: Important (weight = 0.3)

Decision: Hire (Yes) or Don't Hire (No)
```

The perceptron learns the right weights from examples!

---

## 🌍 Real-World Examples (Detailed)

Before diving into technical details, let's understand perceptrons through everyday decisions. Think of it as a **"Yes/No" decision-making machine**!

### Example 1: Should I Go Running?

Think of the Perceptron as a simple machine that helps you decide whether to go running.

#### The Setup

**Inputs (Your Facts):**

- `x1` = Is it sunny? (1 for Yes, 0 for No)
- `x2` = Is it a weekend? (1 for Yes, 0 for No)

**Weights (How Much You Care):**

- `w1 = 0.8` (You LOVE sun, so this weight is HIGH)
- `w2 = 0.2` (You don't care much about weekends, so this is LOW)

**Threshold (Bias):**
This is how hard it is to convince you to move. If you're lazy, you need a high total score to say "Yes."

- `threshold = 0.5`

#### The Math

```
Score = (x1 × w1) + (x2 × w2)
Score = (Is_Sunny × 0.8) + (Is_Weekend × 0.2)
```

**Decision Rule:**

- If `Score > 0.5` → Output = 1 (Go Running! ✓)
- If `Score ≤ 0.5` → Output = 0 (Stay Home ✗)

#### Example Scenarios

**Scenario 1: Sunny Weekday**

```
x1=1, x2=0
Score = (1 × 0.8) + (0 × 0.2) = 0.8
0.8 > 0.5 → YES, go running! ✓
```

**Scenario 2: Rainy Weekend**

```
x1=0, x2=1
Score = (0 × 0.8) + (1 × 0.2) = 0.2
0.2 ≤ 0.5 → NO, stay home ✗
```

**Scenario 3: Rainy Weekday**

```
x1=0, x2=0
Score = (0 × 0.8) + (0 × 0.2) = 0.0
0.0 ≤ 0.5 → NO, stay home ✗
```

#### How It Learns (The Learning Rule)

The Perceptron learns by making mistakes and adjusting. It follows a simple feedback loop:

1. **GUESS** - Make a prediction
2. **CHECK** - Compare guess to actual answer (the truth)
3. **ADJUST** - Update weights based on error

**Adjustment Rules:**

- ✓ If guessed RIGHT: Change nothing
- ✗ If guessed "Yes" but should be "No": Make weights SMALLER (less likely to say yes next time)
- ✗ If guessed "No" but should be "Yes": Make weights LARGER (more enthusiastic next time)

**Formula:**

```
New_Weight = Old_Weight + (Learning_Rate × Error × Input)
Where: Error = (Actual_Answer - Guess)
```

This ensures the machine slowly nudges its "opinion" (weights) until it stops making mistakes!

---

### Example 2: Should I Go to the Beach?

Let's dive deeper with another decision to understand the complete process.

#### The Setup

**Inputs:**

- `x1` = Is it sunny? (1 for Yes, 0 for No)
- `x2` = Do I have a car? (1 for Yes, 0 for No)

**Weights:**

- `w1 = 0.9` (You care A LOT about the sun)
- `w2 = 0.3` (You don't mind taking the bus, so car is less important)

**Threshold:** `0.5`

#### The Decision (Forward Pass)

**Scenario:** It is sunny (`x1=1`), but you don't have a car (`x2=0`)

**Step 1: Multiply inputs by weights**

```
Score = (1 × 0.9) + (0 × 0.3) = 0.9 + 0.0 = 0.9
```

**Step 2: Compare to threshold**

```
0.9 > 0.5 ✓
```

**Step 3: Make decision**

```
Result: Output = 1 (Go to the beach!)
```

#### Component Breakdown

| Component          | Layman Translation | Role                                               |
| ------------------ | ------------------ | -------------------------------------------------- |
| **Input**          | Evidence           | The facts you're looking at                        |
| **Weight**         | Importance         | How much you trust that specific piece of evidence |
| **Bias/Threshold** | Difficulty         | How "hard" it is to change your mind               |
| **Learning Rule**  | Feedback           | Adjusting your "trust" in evidence after a mistake |

#### The Learning Rule (Weight Update)

"Learning" happens when the Perceptron makes a mistake.

**Formula:**

```
New_Weight = Old_Weight + [Learning_Rate × Error × Input]
```

**Where:**

- `Learning_Rate (α) = 0.1` (A small number to avoid overreacting)
- `Error = Actual_Answer - Guess`

**Example of a Correction:**

Suppose it was sunny (`x1=1`), the Perceptron said "Go" (`Guess=1`), but you had a bad time because it was too hot (`Actual=0`).

```
Error = Actual - Guess = 0 - 1 = -1  (negative error)

Update w1:
  New_w1 = Old_w1 + (Learning_Rate × Error × x1)
  New_w1 = 0.9 + (0.1 × -1 × 1)
  New_w1 = 0.9 + (-0.1)
  New_w1 = 0.8
```

**Result:** The "Sunny" weight just got SMALLER! Next time, the sun alone might not be enough to reach the threshold, making the Perceptron more "cautious."

**Why This Works:**

- Negative error (guessed too high) → Decrease weights
- Positive error (guessed too low) → Increase weights
- The machine learns from mistakes and adjusts its "opinion"!

---

### Example 3: Should I Order Pizza?

Here's another everyday decision to cement your understanding!

#### The Decision

**Should I order pizza tonight?**

**Inputs (The Facts):**

- `x1` = Is it Friday or weekend? (1 for Yes, 0 for No)
- `x2` = Am I hungry? (1 for Yes, 0 for No)

**Initial Weights (Your Preferences):**

- `w1 = 0.4` (Weekends make you slightly more likely to order)
- `w2 = 0.7` (Being hungry is MORE important than the day)

**Threshold:** `0.6` (You need a good reason to spend money on pizza!)

#### Decision Scenarios

**Scenario A: Wednesday + Very Hungry**

```
x1=0, x2=1
Score = (0 × 0.4) + (1 × 0.7) = 0.0 + 0.7 = 0.7
0.7 > 0.6 ✓
Decision: YES, order pizza! (Hunger wins)
```

**Scenario B: Saturday + Just Ate**

```
x1=1, x2=0
Score = (1 × 0.4) + (0 × 0.7) = 0.4 + 0.0 = 0.4
0.4 ≤ 0.6 ✗
Decision: NO, don't order (Not hungry enough)
```

**Scenario C: Friday + Hungry**

```
x1=1, x2=1
Score = (1 × 0.4) + (1 × 0.7) = 0.4 + 0.7 = 1.1
1.1 > 0.6 ✓✓
Decision: DEFINITELY order pizza! (Both factors align)
```

#### The Learning Process

Let's say you ordered pizza on Wednesday when hungry (Scenario A), but you regretted it because it was too expensive mid-week.

```
Actual Answer = 0 (Should NOT have ordered)
Your Guess = 1 (You did order)
Error = 0 - 1 = -1

Update weights (Learning Rate = 0.1):
  New_w1 = 0.4 + (0.1 × -1 × 0) = 0.4  (no change, wasn't weekend)
  New_w2 = 0.7 + (0.1 × -1 × 1) = 0.7 - 0.1 = 0.6
```

**Result:** The "hunger" weight decreased from 0.7 to 0.6!

Next time, being hungry alone might not be enough. The Perceptron learned to be more careful about mid-week orders.

**After Several Corrections:**

- Weekend weight might INCREASE (good experiences on weekends)
- Hunger weight might DECREASE (learned it's not the only factor)
- The threshold might ADJUST (overall decision-making changes)

#### Key Insight

The Perceptron is like your brain learning from experience:

1. **Make a decision** (forward pass)
2. **See the outcome** (compare to truth)
3. **Adjust your thinking** (update weights)
4. **Repeat** until you make good decisions consistently

This is EXACTLY how neural networks learn - just with millions of "decisions" happening at once!

---

## 📊 Visual Explanation

### The Perceptron Model

```
Inputs (x)        Weights (w)         Weighted Sum (z)      Activation    Output (y)

x₁ ────────┐
           │×w₁
x₂ ────────┤×w₂──────► Σ (sum) ──────► z = Σ(wᵢxᵢ) + b ──► Step ──► y
           │×w₃
x₃ ────────┘
           ↓
        Bias (b)
```

**Step-by-step:**

1. Multiply each input by its weight
2. Sum all weighted inputs
3. Add bias term
4. Apply activation function (step function for classical perceptron)

### Example with Numbers

```
Inputs:    x = [2, 3, 1]
Weights:   w = [0.5, -0.3, 0.2]
Bias:      b = 0.1

Step 1: Multiply
  2 × 0.5 = 1.0
  3 × -0.3 = -0.9
  1 × 0.2 = 0.2

Step 2: Sum
  z = 1.0 + (-0.9) + 0.2 = 0.3

Step 3: Add bias
  z = 0.3 + 0.1 = 0.4

Step 4: Activation (if z > 0, output 1, else 0)
  y = 1 (because 0.4 > 0)
```

---

## 🧮 The Mathematics

### Forward Pass Formula

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
  = Σ(wᵢxᵢ) + b
  = w·x + b        (dot product)
  = X @ W + b      (in NumPy)

y = step(z) where step(z) = {1 if z > 0, 0 otherwise}
```

### In NumPy (Vectorized!)

```python
# For single sample
z = np.dot(w, x) + b
# Or: z = w @ x + b

# For batch of samples (efficient!)
z = X @ w + b  # X shape: (batch_size, n_features)
               # w shape: (n_features,)
               # z shape: (batch_size,)
```

### Perceptron Learning Rule

How does the perceptron learn? By adjusting weights when it makes mistakes!

```
If prediction is correct: do nothing
If prediction is wrong: adjust weights

Update rule:
w_new = w_old + learning_rate × (y_true - y_pred) × x
b_new = b_old + learning_rate × (y_true - y_pred)
```

**Intuition:**

- If you predicted 0 but should have predicted 1: increase weights
- If you predicted 1 but should have predicted 0: decrease weights
- The size of the adjustment is proportional to the input value

---

## 💻 Code Implementation

### Basic Perceptron Class

```python
import numpy as np

class Perceptron:
    """
    A simple perceptron (single neuron)

    The foundation of all neural networks!
    """

    def __init__(self, n_features, learning_rate=0.01):
        """
        Initialize perceptron

        Args:
            n_features: Number of input features
            learning_rate: How fast to learn (typically 0.001 - 0.1)
        """
        # Initialize weights to small random values
        self.weights = np.random.randn(n_features) * 0.01

        # Initialize bias to zero
        self.bias = 0.0

        # Store learning rate
        self.learning_rate = learning_rate

    def forward(self, X):
        """
        Forward pass: make predictions

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            predictions: Binary predictions, shape (n_samples,)
        """
        # Compute weighted sum
        z = X @ self.weights + self.bias

        # Apply step activation function
        predictions = (z > 0).astype(int)

        return predictions

    def train_step(self, X, y):
        """
        One training step (one update)

        Args:
            X: Input data, shape (n_samples, n_features)
            y: True labels, shape (n_samples,)

        Returns:
            n_errors: Number of misclassifications
        """
        # Make predictions
        predictions = self.forward(X)

        # Calculate errors
        errors = y - predictions  # -1, 0, or 1

        # Update weights
        # w = w + lr * X^T @ errors
        self.weights += self.learning_rate * X.T @ errors

        # Update bias
        # b = b + lr * sum(errors)
        self.bias += self.learning_rate * errors.sum()

        # Count misclassifications
        n_errors = (errors != 0).sum()

        return n_errors

    def train(self, X, y, epochs=100):
        """
        Train the perceptron

        Args:
            X: Training data, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
            epochs: Number of training iterations

        Returns:
            errors_per_epoch: List of error counts per epoch
        """
        errors_per_epoch = []

        for epoch in range(epochs):
            n_errors = self.train_step(X, y)
            errors_per_epoch.append(n_errors)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: {n_errors} errors")

            # Stop if perfect classification
            if n_errors == 0:
                print(f"Perfect classification at epoch {epoch}!")
                break

        return errors_per_epoch
```

---

## ✨ Example Usage

### Example 1: Learning AND Gate

The AND gate is a classic binary logic problem perfect for perceptrons!

```
Truth Table:
x1  x2  →  output
0   0   →    0
0   1   →    0
1   0   →    0
1   1   →    1
```

**Code:**

```python
import numpy as np

# Training data for AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# Create and train perceptron
perceptron = Perceptron(n_features=2, learning_rate=0.1)
errors = perceptron.train(X, y, epochs=100)

# Test
print("\nTesting:")
for inputs, expected in zip(X, y):
    prediction = perceptron.forward(inputs.reshape(1, -1))[0]
    print(f"Input: {inputs} → Prediction: {prediction}, Expected: {expected}")

# Output:
# Epoch 0: 3 errors
# Epoch 10: 0 errors
# Perfect classification at epoch 11!
#
# Testing:
# Input: [0 0] → Prediction: 0, Expected: 0 ✓
# Input: [0 1] → Prediction: 0, Expected: 0 ✓
# Input: [1 0] → Prediction: 0, Expected: 0 ✓
# Input: [1 1] → Prediction: 1, Expected: 1 ✓
```

**What happened?**
The perceptron learned:

- When both inputs are 1, output 1
- Otherwise, output 0

It found weights that separate the classes!

### Example 2: Visualizing Decision Boundary

```python
import matplotlib.pyplot as plt

# Create perceptron and train on AND gate
perceptron = Perceptron(n_features=2, learning_rate=0.1)
perceptron.train(X, y, epochs=100)

# Create mesh grid
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

# Predict on mesh
Z = perceptron.forward(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolor='black', cmap='RdYlBu')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron Decision Boundary - AND Gate')
plt.show()
```

**What you'll see:**
A line dividing the space into two regions:

- Blue region: output 0
- Red region: output 1

The point [1, 1] is in the red region (output 1)!

---

## 🔍 Understanding the Learning Process

### How Weights Change During Training

**Initial state (random):**

```
w = [0.03, -0.02]
b = 0.0
```

**Example: Input [1, 1], True output: 1**

```
Step 1: Forward pass
  z = 1×0.03 + 1×(-0.02) + 0.0 = 0.01
  prediction = 1 (because z > 0)
  ✓ Correct! No update needed.

Example: Input [0, 1], True output: 0
  z = 0×0.03 + 1×(-0.02) + 0.0 = -0.02
  prediction = 0 (because z ≤ 0)
  ✓ Correct! No update needed.

Example: Input [1, 0], True output: 0
  z = 1×0.03 + 0×(-0.02) + 0.0 = 0.03
  prediction = 1 (because z > 0)
  ✗ Wrong! Should be 0

Step 2: Update weights
  error = 0 - 1 = -1
  w[0] = 0.03 + 0.1 × (-1) × 1 = -0.07
  w[1] = -0.02 + 0.1 × (-1) × 0 = -0.02
  b = 0.0 + 0.1 × (-1) = -0.1

New weights: w = [-0.07, -0.02], b = -0.1
```

After many iterations, weights converge to values that classify correctly!

---

## 🚫 Limitations of Perceptrons

### The XOR Problem

**Why perceptrons can't solve XOR:**

```
XOR Truth Table:
x1  x2  →  output
0   0   →    0
0   1   →    1
1   0   →    1
1   1   →    0
```

**Problem:** No single straight line can separate the classes!

```
Visual representation:
  x2
  ^
1 | 1    0     ← Can't draw a line to separate!
  |
0 | 0    1
  +------> x1
    0    1
```

**Solution:** Use multiple layers (you'll learn this in Lesson 3.3!)

### When Perceptrons Work

Perceptrons can solve **linearly separable** problems:

- ✅ AND gate
- ✅ OR gate
- ✅ NOT gate
- ✅ Simple binary classification
- ❌ XOR gate
- ❌ Complex patterns

**Modern use:** While single perceptrons are limited, they're the building blocks of deep networks!

---

## 🔗 Connection to Modern Neural Networks

### Perceptron vs Modern Neuron

**Classic Perceptron (1958):**

```
z = w·x + b
y = step(z)  ← Binary output (0 or 1)
```

**Modern Neuron (used in GPT):**

```
z = w·x + b
y = activation(z)  ← Continuous output with ReLU, GELU, etc.
```

**What changed:**

1. Activation function: Step → ReLU/GELU (allows gradients)
2. Learning: Perceptron rule → Backpropagation (much more powerful)
3. Architecture: Single layer → Many layers (deep learning)

**What stayed the same:**

1. The linear transformation: `z = w·x + b`
2. The concept of weighted inputs
3. The bias term

### In GPT's Feed-Forward Layers

Every neuron in GPT does this:

```python
# Exactly like perceptron, but:
# 1. Uses ReLU/GELU instead of step
# 2. Thousands of neurons per layer
# 3. Many layers stacked

z = X @ W + b
a = gelu(z)  # GELU activation (smooth version of ReLU)
```

**GPT-3 has ~96 layers, each with thousands of neurons. But each neuron is fundamentally a perceptron!**

---

## 🎓 Summary

### Key Concepts

1. **Perceptron = Weighted sum + Activation**
   - `z = w·x + b`
   - `y = activation(z)`

2. **Learning Rule: Adjust weights when wrong**
   - `w_new = w_old + lr × error × input`

3. **Limitations: Can only learn linearly separable patterns**
   - Works: AND, OR
   - Doesn't work: XOR

4. **Foundation for Deep Learning**
   - Modern neurons use same formula
   - Just better activations and learning algorithms

### What You Built

✅ A complete perceptron from scratch
✅ Training algorithm (perceptron learning rule)
✅ Successfully learned AND gate
✅ Visualized decision boundary

### Next Steps

In the next lesson, you'll learn about **activation functions** - the key to making neural networks powerful!

- Why step functions are limiting
- ReLU, Sigmoid, Tanh, Softmax
- How activation functions enable deep learning

---

## 💡 Practice Exercises

Before moving on, try these:

1. **OR Gate**: Train a perceptron on OR gate

   ```
   [0, 0] → 0
   [0, 1] → 1
   [1, 0] → 1
   [1, 1] → 1
   ```

2. **NOT Gate**: Train on NOT (one input!)

   ```
   [0] → 1
   [1] → 0
   ```

3. **XOR Gate**: Try to train on XOR (it won't work - understand why!)

4. **Modify Learning Rate**: Train AND gate with different learning rates (0.001, 0.01, 0.1, 1.0). What happens?

5. **Weight Initialization**: Try different initial weights. Does the perceptron still learn?

---

## 🔍 Debugging Tips

### Common Issues

**Problem: Perceptron doesn't learn**

```python
# Check 1: Learning rate too small?
learning_rate = 0.1  # Try increasing

# Check 2: Too few epochs?
epochs = 1000  # Try more iterations

# Check 3: Is problem linearly separable?
# Perceptrons can't solve XOR!
```

**Problem: Weights explode**

```python
# Learning rate too high
learning_rate = 0.01  # Try decreasing
```

**Problem: Prediction always same**

```python
# Check weight initialization
print(f"Weights: {perceptron.weights}")
print(f"Bias: {perceptron.bias}")

# Re-initialize if needed
perceptron.weights = np.random.randn(n_features) * 0.01
```

---

## 📚 Further Reading

### History

- **1958**: Frank Rosenblatt invents the perceptron
- **1969**: Minsky & Papert show XOR limitation
- **1980s**: Backpropagation solves the limitation
- **2020s**: Same principles power GPT!

### Why This Matters

Every neuron in GPT-4 is fundamentally a perceptron with:

- Better activation function (GELU instead of step)
- Better learning algorithm (backpropagation + Adam)
- More layers (96+ instead of 1)

You just built the atomic unit of modern AI! 🎉

---

**Next Lesson:** `02_activation_functions.md` - Making neurons non-linear!

Run `example_01_perceptron.py` to see all this code in action!

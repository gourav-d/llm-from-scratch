"""
Perceptron Exercises - Practice What You Learned

Complete these exercises to master perceptron fundamentals!
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("PERCEPTRON EXERCISES")
print("="*70)

# ==============================================================================
# EXERCISE 1: Implement OR Gate
# ==============================================================================

print("\n📝 EXERCISE 1: Train Perceptron on OR Gate")
print("-" * 70)
print("""
OR Gate Truth Table:
  x1  x2  →  y
  0   0   →  0
  0   1   →  1
  1   0   →  1
  1   1   →  1

Task: Train a perceptron to learn OR gate
""")

# TODO: Create training data for OR gate
# X_or = ...
# y_or = ...

# TODO: Create and train perceptron
# perceptron_or = Perceptron(n_features=2, learning_rate=0.1)
# perceptron_or.train(X_or, y_or, epochs=100)

# TODO: Test predictions
# print("Testing OR gate:")
# for inputs, expected in zip(X_or, y_or):
#     prediction = perceptron_or.forward(inputs.reshape(1, -1))[0]
#     print(f"Input: {inputs} → Prediction: {prediction}, Expected: {expected}")

# ==============================================================================
# EXERCISE 2: Implement NOT Gate
# ==============================================================================

print("\n📝 EXERCISE 2: Train Perceptron on NOT Gate")
print("-" * 70)
print("""
NOT Gate Truth Table:
  x  →  y
  0  →  1
  1  →  0

Task: Train a perceptron to learn NOT (single input!)
""")

# TODO: Create training data for NOT gate
# X_not = ...
# y_not = ...

# TODO: Train perceptron

# ==============================================================================
# EXERCISE 3: Learning Rate Experiment
# ==============================================================================

print("\n📝 EXERCISE 3: Effect of Learning Rate")
print("-" * 70)
print("""
Task: Train AND gate with different learning rates:
  - 0.001 (very small)
  - 0.01
  - 0.1
  - 1.0 (very large)

Compare:
  - How many epochs to converge?
  - Does it always converge?
  - Plot learning curves
""")

# TODO: Implement learning rate comparison

# ==============================================================================
# EXERCISE 4: Weight Initialization
# ==============================================================================

print("\n📝 EXERCISE 4: Effect of Weight Initialization")
print("-" * 70)
print("""
Task: Train AND gate with different initial weights:
  - Small random: np.random.randn(n) * 0.01
  - Large random: np.random.randn(n) * 1.0
  - All zeros: np.zeros(n)
  - All ones: np.ones(n)

Question: Does initialization matter?
""")

# TODO: Implement initialization experiment

# ==============================================================================
# EXERCISE 5: XOR Understanding
# ==============================================================================

print("\n📝 EXERCISE 5: Understanding Why XOR Fails")
print("-" * 70)
print("""
XOR Truth Table:
  x1  x2  →  y
  0   0   →  0
  0   1   →  1
  1   0   →  1
  1   1   →  0

Tasks:
1. Try to train perceptron on XOR
2. Plot the decision boundary
3. Explain why it fails (write in comments)
4. Sketch on paper: can you draw ONE line to separate the classes?
""")

# TODO: Implement XOR experiment

# ==============================================================================
# EXERCISE 6: Custom Dataset
# ==============================================================================

print("\n📝 EXERCISE 6: Create Your Own Linearly Separable Dataset")
print("-" * 70)
print("""
Task: Create a custom 2D dataset that IS linearly separable

Example ideas:
  - Points above y = x vs below
  - Points inside circle r < 2 vs outside r > 2 (simple case)
  - Temperature > 30 AND humidity > 70 = uncomfortable

Requirements:
  - At least 20 samples
  - 2 features (x1, x2)
  - Binary labels (0 or 1)
  - Linearly separable (perceptron CAN learn it)
""")

# TODO: Create and train on custom dataset

# ==============================================================================
# EXERCISE 7: Perceptron Limitations
# ==============================================================================

print("\n📝 EXERCISE 7: Testing Perceptron Limitations")
print("-" * 70)
print("""
Create datasets that perceptron CANNOT learn:
1. XOR (you've seen this)
2. Circle: points inside circle = 1, outside = 0
3. Checkerboard pattern

For each:
  - Create the dataset
  - Try to train perceptron
  - Show it fails
  - Explain WHY (in comments)
""")

# TODO: Implement limitation tests

# ==============================================================================
# EXERCISE 8: Visualization Challenge
# ==============================================================================

print("\n📝 EXERCISE 8: Visualize Learning Process")
print("-" * 70)
print("""
Task: Create an animated visualization showing:
  - Initial random weights
  - Decision boundary after each epoch
  - Final converged boundary

Hint: Save plots for each epoch, create GIF or side-by-side comparison
""")

# TODO: Create visualization

# ==============================================================================
# EXERCISE 9: Multi-Feature Perceptron
# ==============================================================================

print("\n📝 EXERCISE 9: Perceptron with More Features")
print("-" * 70)
print("""
Task: Create a perceptron with 5 features

Example: Student admission decision based on:
  - GPA (0-4.0)
  - Test score (0-100)
  - Extracurricular activities (0-10)
  - Recommendation letter score (0-10)
  - Interview score (0-10)

Generate synthetic data and train perceptron.
""")

# TODO: Implement multi-feature perceptron

# ==============================================================================
# EXERCISE 10: CHALLENGE - Implement NAND Gate
# ==============================================================================

print("\n📝 EXERCISE 10: CHALLENGE - NAND Gate")
print("-" * 70)
print("""
NAND Gate (NOT AND):
  x1  x2  →  y
  0   0   →  1
  0   1   →  1
  1   0   →  1
  1   1   →  0

Fun fact: NAND is universal - you can build ANY logic gate from NAND!

Task:
1. Train perceptron on NAND
2. Verify it learns correctly
3. Research: Why is NAND called "universal"?
""")

# TODO: Implement NAND gate

# ==============================================================================
# SOLUTIONS
# ==============================================================================

def show_solutions():
    """Show solutions (run after attempting exercises!)"""

    print("\n\n" + "="*70)
    input("Press Enter to see solutions... ")
    print("="*70)

    # Import perceptron from example
    class Perceptron:
        def __init__(self, n_features, learning_rate=0.01):
            self.weights = np.random.randn(n_features) * 0.01
            self.bias = 0.0
            self.learning_rate = learning_rate

        def forward(self, X):
            z = X @ self.weights + self.bias
            return (z > 0).astype(int)

        def train_step(self, X, y):
            predictions = self.forward(X)
            errors = y - predictions
            self.weights += self.learning_rate * X.T @ errors
            self.bias += self.learning_rate * errors.sum()
            return (errors != 0).sum()

        def train(self, X, y, epochs=100):
            errors_per_epoch = []
            for epoch in range(epochs):
                n_errors = self.train_step(X, y)
                errors_per_epoch.append(n_errors)
                if n_errors == 0:
                    print(f"✓ Converged at epoch {epoch}")
                    break
            return errors_per_epoch

    print("\n💡 SOLUTION 1: OR Gate")
    print("-" * 70)
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    perceptron_or = Perceptron(n_features=2, learning_rate=0.1)
    perceptron_or.train(X_or, y_or, epochs=100)
    print("Testing OR gate:")
    for inputs, expected in zip(X_or, y_or):
        pred = perceptron_or.forward(inputs.reshape(1, -1))[0]
        status = "✓" if pred == expected else "✗"
        print(f"  {inputs} → {pred} (expected {expected}) {status}")

    print("\n💡 SOLUTION 2: NOT Gate")
    print("-" * 70)
    X_not = np.array([[0], [1]])
    y_not = np.array([1, 0])
    perceptron_not = Perceptron(n_features=1, learning_rate=0.1)
    perceptron_not.train(X_not, y_not, epochs=100)
    print("Testing NOT gate:")
    for inputs, expected in zip(X_not, y_not):
        pred = perceptron_not.forward(inputs.reshape(1, -1))[0]
        status = "✓" if pred == expected else "✗"
        print(f"  {inputs[0]} → {pred} (expected {expected}) {status}")

    print("\n💡 SOLUTION 3: Learning Rate Effect")
    print("-" * 70)
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])

    learning_rates = [0.001, 0.01, 0.1, 1.0]
    for lr in learning_rates:
        p = Perceptron(n_features=2, learning_rate=lr)
        errors = p.train(X_and, y_and, epochs=100)
        epochs_to_converge = len(errors)
        print(f"LR={lr:5.3f}: Converged in {epochs_to_converge:3d} epochs")

    print("\n💡 SOLUTION 5: XOR Fails")
    print("-" * 70)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    perceptron_xor = Perceptron(n_features=2, learning_rate=0.1)
    errors = perceptron_xor.train(X_xor, y_xor, epochs=100)
    print(f"Final errors: {errors[-1]}")
    print("✗ Cannot solve XOR - not linearly separable!")
    print("\nWhy:")
    print("  No single line can separate [0,1] and [1,0] from [0,0] and [1,1]")
    print("  This requires TWO lines (or a curve) → need multiple layers!")

    print("\n💡 SOLUTION 10: NAND Gate")
    print("-" * 70)
    X_nand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_nand = np.array([1, 1, 1, 0])  # Opposite of AND
    perceptron_nand = Perceptron(n_features=2, learning_rate=0.1)
    perceptron_nand.train(X_nand, y_nand, epochs=100)
    print("Testing NAND gate:")
    for inputs, expected in zip(X_nand, y_nand):
        pred = perceptron_nand.forward(inputs.reshape(1, -1))[0]
        status = "✓" if pred == expected else "✗"
        print(f"  {inputs} → {pred} (expected {expected}) {status}")

    print("\n" + "="*70)
    print("All solutions shown!")
    print("Compare with your implementations.")
    print("="*70)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Try to solve the exercises first!")
    print("="*70)
    print("\nWhen ready, run show_solutions() to see answers.")
    print("\nOr run: python exercise_01_perceptron.py")
    print("="*70)

    # Uncomment to show solutions immediately:
    # show_solutions()

"""
Perceptron - Building Your First Neuron!

This example demonstrates the foundation of ALL neural networks:
the simple perceptron.

What you'll see:
1. How a perceptron makes decisions
2. How it learns from mistakes
3. Visualizing the decision boundary
4. Understanding limitations (XOR problem)
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("PERCEPTRON - Your First Neural Network!")
print("="*70)

# ==============================================================================
# REAL-WORLD EXAMPLES: Understanding Perceptron in Layman Terms
# ==============================================================================

print("\n" + "="*70)
print("REAL-WORLD EXAMPLES: Understanding Perceptron")
print("="*70)

print("""
Before we dive into code, let's understand how a Perceptron works using
simple everyday decisions. Think of it as a "Yes/No" decision-making machine!
""")

# ------------------------------------------------------------------------------
# Example 1: Should I Go Running?
# ------------------------------------------------------------------------------

print("\n" + "-"*70)
print("EXAMPLE 1: Should I Go Running?")
print("-"*70)

print("""
Think of the Perceptron as a simple "Yes/No" machine that helps you decide
whether to go running.

INPUTS (Your Facts):
  x1 = Is it sunny? (1 for Yes, 0 for No)
  x2 = Is it a weekend? (1 for Yes, 0 for No)

WEIGHTS (How Much You Care):
  w1 = 0.8  (You love sun, so this weight is HIGH)
  w2 = 0.2  (You don't care much about weekends, so this is LOW)

THRESHOLD (Bias):
  This is how hard it is to convince you to move. If you're lazy,
  you need a high total score to say "Yes."
  Let's say threshold = 0.5

THE MATH:
  Score = (x1 × w1) + (x2 × w2)
  Score = (Is_Sunny × 0.8) + (Is_Weekend × 0.2)

DECISION RULE:
  If Score > Threshold (0.5) → Output = 1 (Go Running!)
  If Score ≤ Threshold (0.5) → Output = 0 (Stay Home)

EXAMPLE SCENARIOS:
  Scenario 1: Sunny weekday
    x1=1, x2=0 → Score = (1×0.8) + (0×0.2) = 0.8
    0.8 > 0.5 → YES, go running! ✓

  Scenario 2: Rainy weekend
    x1=0, x2=1 → Score = (0×0.8) + (1×0.2) = 0.2
    0.2 ≤ 0.5 → NO, stay home ✗

  Scenario 3: Rainy weekday
    x1=0, x2=0 → Score = (0×0.8) + (0×0.2) = 0.0
    0.0 ≤ 0.5 → NO, stay home ✗

HOW IT LEARNS (The Learning Rule):
  The Perceptron learns by making mistakes and adjusting:

  Step 1: GUESS - Make a prediction
  Step 2: CHECK - Compare guess to actual answer (the truth)
  Step 3: ADJUST - Update weights based on error

  Adjustment Rules:
    ✓ If guessed RIGHT: Change nothing
    ✗ If guessed "Yes" but should be "No": Make weights SMALLER
                                          (less likely to say yes next time)
    ✗ If guessed "No" but should be "Yes": Make weights LARGER
                                           (more enthusiastic next time)

  Formula: New_Weight = Old_Weight + (Learning_Rate × Error × Input)
  Where Error = (Actual_Answer - Guess)

  This ensures the machine slowly nudges its "opinion" (weights)
  until it stops making mistakes!
""")

# ------------------------------------------------------------------------------
# Example 2: Should I Go to the Beach?
# ------------------------------------------------------------------------------

print("\n" + "-"*70)
print("EXAMPLE 2: Should I Go to the Beach? (Detailed)")
print("-"*70)

print("""
Let's dive deeper with another decision: "Should I go to the beach?"

SETUP:
  Inputs (x):
    x1 = Is it sunny? (1 for Yes, 0 for No)
    x2 = Do I have a car? (1 for Yes, 0 for No)

  Weights (w):
    w1 = 0.9  (You care A LOT about the sun)
    w2 = 0.3  (You don't mind taking the bus, so car is less important)

  Threshold (Bias): 0.5
    If the "score" is higher than this, you go to the beach.

THE DECISION (Forward Pass):
  Scenario: It is sunny (x1=1), but you don't have a car (x2=0)

  Step 1: Multiply inputs by weights
    Score = (1 × 0.9) + (0 × 0.3) = 0.9 + 0.0 = 0.9

  Step 2: Compare to threshold
    0.9 > 0.5 ✓

  Step 3: Make decision
    Result: Output = 1 (Go to the beach!)

COMPONENT BREAKDOWN:
  ┌─────────────┬──────────────────┬────────────────────────────┐
  │ Component   │ Layman Name      │ Role                       │
  ├─────────────┼──────────────────┼────────────────────────────┤
  │ Input       │ Evidence         │ The facts you're looking at│
  │ Weight      │ Importance       │ How much you trust that    │
  │             │                  │ specific piece of evidence │
  │ Bias/       │ Difficulty       │ How "hard" it is to change │
  │ Threshold   │                  │ your mind                  │
  │ Learning    │ Feedback         │ Adjusting your "trust" in  │
  │ Rule        │                  │ evidence after a mistake   │
  └─────────────┴──────────────────┴────────────────────────────┘

THE LEARNING RULE (Weight Update):
  "Learning" happens when the Perceptron makes a mistake.

  Formula:
    New_Weight = Old_Weight + [Learning_Rate × Error × Input]

  Where:
    Learning_Rate (α) = 0.1  (A small number to avoid overreacting)
    Error = Actual_Answer - Guess

  EXAMPLE OF A CORRECTION:
    Suppose it was sunny (x1=1), the Perceptron said "Go" (Guess=1),
    but you had a bad time because it was too hot (Actual=0).

    Error = Actual - Guess = 0 - 1 = -1 (negative error)

    Update w1:
      New_w1 = Old_w1 + (Learning_Rate × Error × x1)
      New_w1 = 0.9 + (0.1 × -1 × 1)
      New_w1 = 0.9 + (-0.1)
      New_w1 = 0.8

    Result: The "Sunny" weight just got SMALLER!
            Next time, the sun alone might not be enough to reach
            the threshold, making the Perceptron more "cautious."

  WHY THIS WORKS:
    - Negative error (guessed too high) → Decrease weights
    - Positive error (guessed too low) → Increase weights
    - The machine learns from mistakes and adjusts its "opinion"!
""")

# ------------------------------------------------------------------------------
# Example 3: Should I Order Pizza?
# ------------------------------------------------------------------------------

print("\n" + "-"*70)
print("EXAMPLE 3: Should I Order Pizza? (New Example)")
print("-"*70)

print("""
Here's another everyday decision to cement your understanding!

THE DECISION: Should I order pizza tonight?

INPUTS (The Facts):
  x1 = Is it Friday or weekend? (1 for Yes, 0 for No)
  x2 = Am I hungry? (1 for Yes, 0 for No)

INITIAL WEIGHTS (Your Preferences):
  w1 = 0.4  (Weekends make you slightly more likely to order)
  w2 = 0.7  (Being hungry is MORE important than the day)

THRESHOLD (Bias): 0.6
  You need a good reason to spend money on pizza!

THE DECISION PROCESS:
  Scenario A: It's Wednesday (x1=0) and you're very hungry (x2=1)

    Score = (0 × 0.4) + (1 × 0.7) = 0.0 + 0.7 = 0.7
    0.7 > 0.6 ✓
    Decision: YES, order pizza! (Hunger wins)

  Scenario B: It's Saturday (x1=1) but you just ate (x2=0)

    Score = (1 × 0.4) + (0 × 0.7) = 0.4 + 0.0 = 0.4
    0.4 ≤ 0.6 ✗
    Decision: NO, don't order (Not hungry enough)

  Scenario C: It's Friday (x1=1) AND you're hungry (x2=1)

    Score = (1 × 0.4) + (1 × 0.7) = 0.4 + 0.7 = 1.1
    1.1 > 0.6 ✓✓
    Decision: DEFINITELY order pizza! (Both factors align)

THE LEARNING PROCESS:
  Let's say you ordered pizza on Wednesday when hungry (Scenario A),
  but you regretted it because it was too expensive mid-week.

  Actual Answer = 0 (Should NOT have ordered)
  Your Guess = 1 (You did order)
  Error = 0 - 1 = -1

  Update weights (Learning Rate = 0.1):
    New_w1 = 0.4 + (0.1 × -1 × 0) = 0.4 (no change, wasn't weekend)
    New_w2 = 0.7 + (0.1 × -1 × 1) = 0.7 - 0.1 = 0.6

  Result: The "hunger" weight decreased from 0.7 to 0.6!
          Next time, being hungry alone might not be enough.
          The Perceptron learned to be more careful about mid-week orders.

  After several such corrections:
    - Weekend weight might INCREASE (good experiences on weekends)
    - Hunger weight might DECREASE (learned it's not the only factor)
    - The threshold might ADJUST (overall decision-making changes)

KEY INSIGHT:
  The Perceptron is like your brain learning from experience!
  - Make a decision (forward pass)
  - See the outcome (compare to truth)
  - Adjust your thinking (update weights)
  - Repeat until you make good decisions consistently

  This is EXACTLY how neural networks learn - just with millions
  of "decisions" happening at once!
""")

print("\n" + "="*70)
print("Now let's see this in action with Python code!")
print("="*70)

# ==============================================================================
# PART 1: The Perceptron Class
# ==============================================================================

class Perceptron:
    """
    A simple perceptron (single artificial neuron)

    This is the building block of GPT, BERT, and all neural networks!
    """

    def __init__(self, n_features, learning_rate=0.01):
        """
        Initialize the perceptron

        Args:
            n_features: Number of input features
            learning_rate: How fast to learn (0.001 - 0.1 typically)
        """
        # Small random weights (important!)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate

        print(f"Perceptron created!")
        print(f"  Features: {n_features}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Initial weights: {self.weights}")
        print(f"  Initial bias: {self.bias}")

    def forward(self, X):
        """
        Forward pass: make predictions

        Math: z = X @ w + b, then y = 1 if z > 0 else 0
        """
        # Compute weighted sum
        z = X @ self.weights + self.bias

        # Step activation: output 1 if positive, 0 otherwise
        predictions = (z > 0).astype(int)

        return predictions

    def train_step(self, X, y):
        """
        One training step

        Perceptron learning rule:
        - If correct: do nothing
        - If wrong: adjust weights based on error
        """
        # Predictions
        predictions = self.forward(X)

        # Errors (will be -1, 0, or 1)
        errors = y - predictions

        # Update weights: w = w + lr * X^T @ errors
        self.weights += self.learning_rate * X.T @ errors

        # Update bias: b = b + lr * sum(errors)
        self.bias += self.learning_rate * errors.sum()

        # Return number of mistakes
        n_errors = (errors != 0).sum()
        return n_errors

    def train(self, X, y, epochs=100, verbose=True):
        """Train the perceptron"""
        errors_per_epoch = []

        for epoch in range(epochs):
            n_errors = self.train_step(X, y)
            errors_per_epoch.append(n_errors)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: {n_errors} errors")

            # Perfect classification!
            if n_errors == 0:
                if verbose:
                    print(f"✓ Perfect classification at epoch {epoch}!")
                break

        return errors_per_epoch


# ==============================================================================
# PART 2: Learning AND Gate
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Learning AND Gate")
print("="*70)

print("""
AND Gate Truth Table:
  x1  x2  →  y
  0   0   →  0
  0   1   →  0
  1   0   →  0
  1   1   →  1

Only outputs 1 when BOTH inputs are 1.
""")

# Training data
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([0, 0, 0, 1])

print(f"Training data shape: {X_and.shape}")
print(f"Labels shape: {y_and.shape}\n")

# Create and train
perceptron_and = Perceptron(n_features=2, learning_rate=0.1)
errors = perceptron_and.train(X_and, y_and, epochs=100)

# Test
print("\nTesting AND Gate:")
print("-" * 40)
for inputs, expected in zip(X_and, y_and):
    prediction = perceptron_and.forward(inputs.reshape(1, -1))[0]
    status = "✓" if prediction == expected else "✗"
    print(f"Input: {inputs} → Prediction: {prediction}, Expected: {expected} {status}")

print(f"\nFinal weights: {perceptron_and.weights}")
print(f"Final bias: {perceptron_and.bias:.3f}")

# ==============================================================================
# PART 3: Learning OR Gate
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Learning OR Gate")
print("="*70)

print("""
OR Gate Truth Table:
  x1  x2  →  y
  0   0   →  0
  0   1   →  1
  1   0   →  1
  1   1   →  1

Outputs 1 when AT LEAST ONE input is 1.
""")

X_or = X_and  # Same inputs
y_or = np.array([0, 1, 1, 1])  # Different outputs

perceptron_or = Perceptron(n_features=2, learning_rate=0.1)
errors = perceptron_or.train(X_or, y_or, epochs=100)

print("\nTesting OR Gate:")
print("-" * 40)
for inputs, expected in zip(X_or, y_or):
    prediction = perceptron_or.forward(inputs.reshape(1, -1))[0]
    status = "✓" if prediction == expected else "✗"
    print(f"Input: {inputs} → Prediction: {prediction}, Expected: {expected} {status}")

# ==============================================================================
# PART 4: The XOR Problem (Why Single Layer Fails)
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: The XOR Problem")
print("="*70)

print("""
XOR (Exclusive OR) Truth Table:
  x1  x2  →  y
  0   0   →  0
  0   1   →  1
  1   0   →  1
  1   1   →  0

Outputs 1 when inputs are DIFFERENT.

⚠️  A single perceptron CANNOT learn this!
Why? It's not linearly separable (no single line can separate the classes).
""")

X_xor = X_and
y_xor = np.array([0, 1, 1, 0])

perceptron_xor = Perceptron(n_features=2, learning_rate=0.1)
errors = perceptron_xor.train(X_xor, y_xor, epochs=100, verbose=False)

print(f"Training XOR... (this will fail!)")
print(f"Errors after 100 epochs: {errors[-1]}")

print("\nTesting XOR Gate:")
print("-" * 40)
for inputs, expected in zip(X_xor, y_xor):
    prediction = perceptron_xor.forward(inputs.reshape(1, -1))[0]
    status = "✓" if prediction == expected else "✗"
    print(f"Input: {inputs} → Prediction: {prediction}, Expected: {expected} {status}")

print("\n⚠️  Cannot solve XOR with single perceptron!")
print("Solution: Use multiple layers (you'll learn in Lesson 3.3)")

# ==============================================================================
# PART 5: Understanding How It Learns
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Watching The Learning Process")
print("="*70)

# Train AND gate step-by-step
X_train = X_and
y_train = y_and

# Start fresh
perceptron_demo = Perceptron(n_features=2, learning_rate=0.1)

print("Initial state:")
print(f"  Weights: {perceptron_demo.weights}")
print(f"  Bias: {perceptron_demo.bias:.3f}")

print("\nTraining step-by-step (first 5 epochs):\n")

for epoch in range(5):
    # Make predictions
    predictions = perceptron_demo.forward(X_train)
    errors = y_train - predictions

    print(f"Epoch {epoch}:")
    print(f"  Predictions: {predictions}")
    print(f"  True labels: {y_train}")
    print(f"  Errors: {errors} (negative = predicted 1 but should be 0)")

    # Update
    n_errors = perceptron_demo.train_step(X_train, y_train)

    print(f"  New weights: {perceptron_demo.weights}")
    print(f"  New bias: {perceptron_demo.bias:.3f}")
    print(f"  Mistakes: {n_errors}\n")

    if n_errors == 0:
        print("  ✓ Learned perfectly!")
        break

# ==============================================================================
# PART 6: Visualizing Decision Boundary
# ==============================================================================

print("="*70)
print("EXAMPLE 5: Visualizing Decision Boundaries")
print("="*70)

def plot_decision_boundary(perceptron, X, y, title):
    """Plot the decision boundary learned by perceptron"""
    # Create mesh grid
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    # Predict on mesh
    Z = perceptron.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=y, s=200, edgecolor='black',
                cmap='RdYlBu', linewidth=2)

    # Plot decision line
    # Line equation: w1*x1 + w2*x2 + b = 0
    # Solving for x2: x2 = -(w1*x1 + b) / w2
    w1, w2 = perceptron.weights
    b = perceptron.bias

    if abs(w2) > 0.01:  # Avoid division by zero
        x_line = np.array([x_min, x_max])
        y_line = -(w1 * x_line + b) / w2
        plt.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary')

    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

# Plot all three
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plot_decision_boundary(perceptron_and, X_and, y_and, 'AND Gate')

plt.subplot(1, 3, 2)
plot_decision_boundary(perceptron_or, X_or, y_or, 'OR Gate')

plt.subplot(1, 3, 3)
plot_decision_boundary(perceptron_xor, X_xor, y_xor, 'XOR Gate (Failed)')

plt.tight_layout()
plt.savefig('perceptron_decision_boundaries.png', dpi=150, bbox_inches='tight')
print("✓ Saved decision boundary plots to 'perceptron_decision_boundaries.png'")
print("\nWhat you should see:")
print("  - AND: Line separates (1,1) from the rest")
print("  - OR: Line separates (0,0) from the rest")
print("  - XOR: No line can separate correctly (that's why it fails!)")

# ==============================================================================
# PART 7: Learning Curves
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 6: Learning Curves - Tracking Progress")
print("="*70)

# Train fresh perceptrons and track errors
perceptron_and_new = Perceptron(n_features=2, learning_rate=0.1)
errors_and = perceptron_and_new.train(X_and, y_and, epochs=50, verbose=False)

perceptron_or_new = Perceptron(n_features=2, learning_rate=0.1)
errors_or = perceptron_or_new.train(X_or, y_or, epochs=50, verbose=False)

perceptron_xor_new = Perceptron(n_features=2, learning_rate=0.1)
errors_xor = perceptron_xor_new.train(X_xor, y_xor, epochs=50, verbose=False)

# Plot learning curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(errors_and, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Number of Errors')
plt.title('AND Gate Learning')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(errors_or, linewidth=2, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Number of Errors')
plt.title('OR Gate Learning')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(errors_xor, linewidth=2, color='red')
plt.xlabel('Epoch')
plt.ylabel('Number of Errors')
plt.title('XOR Gate Learning (Fails)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('perceptron_learning_curves.png', dpi=150, bbox_inches='tight')
print("✓ Saved learning curves to 'perceptron_learning_curves.png'")
print("\nWhat you should see:")
print("  - AND & OR: Errors drop to 0 quickly")
print("  - XOR: Errors never reach 0 (impossible for single perceptron)")

# ==============================================================================
# PART 8: Effect of Learning Rate
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 7: Effect of Learning Rate")
print("="*70)

learning_rates = [0.001, 0.01, 0.1, 1.0]
plt.figure(figsize=(12, 3))

for i, lr in enumerate(learning_rates):
    perceptron = Perceptron(n_features=2, learning_rate=lr)
    errors = perceptron.train(X_and, y_and, epochs=50, verbose=False)

    plt.subplot(1, 4, i+1)
    plt.plot(errors, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Errors')
    plt.title(f'LR = {lr}')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 4.5)

plt.tight_layout()
plt.savefig('perceptron_learning_rates.png', dpi=150, bbox_inches='tight')
print("✓ Saved learning rate comparison to 'perceptron_learning_rates.png'")
print("\nWhat you should see:")
print("  - Too small (0.001): Slow learning")
print("  - Just right (0.01-0.1): Fast convergence")
print("  - Too large (1.0): May oscillate")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("✅ SUMMARY")
print("="*70)

print("""
What You Just Built:
1. ✅ A complete perceptron from scratch
2. ✅ Perceptron learning rule (weight updates)
3. ✅ Trained on AND and OR gates successfully
4. ✅ Discovered XOR limitation (needs multiple layers!)
5. ✅ Visualized decision boundaries
6. ✅ Tracked learning progress
7. ✅ Understood effect of learning rate

Key Insights:
- Perceptron = weighted sum + step activation
- Learning = adjust weights when wrong
- Limitation = can only learn linearly separable patterns
- Foundation = same formula used in modern neural networks!

Connection to GPT:
Every neuron in GPT-4 is fundamentally a perceptron:
  z = X @ W + b
  a = activation(z)

The only differences:
  - Activation: GELU instead of step function
  - Learning: Backpropagation instead of perceptron rule
  - Scale: Billions of neurons instead of 1!

Next Steps:
1. Read Lesson 3.2: Activation Functions
   (Why ReLU is better than step function)
2. Run example_02_activations.py
3. Complete exercise_01_perceptron.py

You've built the atomic unit of all neural networks! 🎉
""")

print("="*70)
print("Run 'python exercise_01_perceptron.py' to practice!")
print("="*70)

# Show plots
try:
    plt.show()
except:
    print("\n(Close plot windows to see all visualizations)")

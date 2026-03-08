"""
SUPER SIMPLE PERCEPTRON EXAMPLE
================================

Real-World Example: "Should I Buy Coffee Today?"

This example explains EVERY line, including:
- Why we reshape data
- Why we use errors.sum() for bias
- How training works step-by-step

Perfect for beginners!
"""

import numpy as np

print("=" * 80)
print("SIMPLE PERCEPTRON: Should I Buy Coffee Today?")
print("=" * 80)

# ==============================================================================
# PART 1: Understanding the Problem
# ==============================================================================

print("\n" + "=" * 80)
print("PART 1: The Coffee Decision Problem")
print("=" * 80)

print("""
SCENARIO: You want to train a perceptron to decide if you should buy coffee.

INPUTS (Features):
  x1 = Are you tired? (1 = Yes, 0 = No)
  x2 = Is it morning? (1 = Yes, 0 = No)

OUTPUT (Decision):
  y = 1 means "Buy Coffee"
  y = 0 means "Don't Buy Coffee"

TRAINING DATA (Your past decisions):
  Day 1: Not tired (0), Not morning (0) → Didn't buy coffee (0)
  Day 2: Not tired (0), Morning (1)     → Didn't buy coffee (0)
  Day 3: Tired (1),     Not morning (0) → Bought coffee (1)
  Day 4: Tired (1),     Morning (1)     → Bought coffee (1)

GOAL: Train the perceptron to learn your pattern!
""")

# ==============================================================================
# PART 2: Prepare the Training Data
# ==============================================================================

print("\n" + "=" * 80)
print("PART 2: Preparing Training Data")
print("=" * 80)

# Create the training data
# Each row is one training example
# Columns are: [tired?, morning?]
X_train = np.array([
    [0, 0],  # Day 1: Not tired, not morning
    [0, 1],  # Day 2: Not tired, morning
    [1, 0],  # Day 3: Tired, not morning
    [1, 1]   # Day 4: Tired, morning
])

# Labels (what you actually did)
y_train = np.array([0, 0, 1, 1])  # 0 = no coffee, 1 = bought coffee

print("Training Data (X_train):")
print(X_train)
print(f"  Shape: {X_train.shape}")
print(f"  Meaning: 4 examples, 2 features each")
print()

print("Labels (y_train):")
print(y_train)
print(f"  Shape: {y_train.shape}")
print(f"  Meaning: 4 labels (one for each example)")

# ==============================================================================
# PART 3: Initialize the Perceptron
# ==============================================================================

print("\n" + "=" * 80)
print("PART 3: Initialize the Perceptron (Random Guessing at Start)")
print("=" * 80)

# Start with random small weights
# Why random? We need a starting point for learning
# Why small? Prevents extreme initial predictions
weights = np.random.randn(2) * 0.01  # 2 weights (one per feature)
bias = 0.0                            # Start bias at zero
learning_rate = 0.1                   # How fast to learn

print(f"Initial weights: {weights}")
print(f"  w[0] controls how much 'tired' matters")
print(f"  w[1] controls how much 'morning' matters")
print()
print(f"Initial bias: {bias}")
print(f"  Bias is like a 'baseline tendency' to buy coffee")
print()
print(f"Learning rate: {learning_rate}")
print(f"  Controls how big each update step is")

# ==============================================================================
# PART 4: Understanding Forward Pass (Making Predictions)
# ==============================================================================

print("\n" + "=" * 80)
print("PART 4: Making a Prediction (Forward Pass)")
print("=" * 80)

print("""
FORMULA: prediction = 1 if (X @ weights + bias > 0) else 0

Let's break this down:
  1. X @ weights = weighted sum (matrix multiplication)
  2. + bias = add baseline tendency
  3. > 0 = step function (yes/no decision)

C# EQUIVALENT:
  var prediction = (X.DotProduct(weights) + bias > 0) ? 1 : 0;
""")

# Let's predict for the FIRST training example
print("Example: Day 1 (Not tired, Not morning)")
print("-" * 80)

# Get first training example
# We need to reshape from (2,) to (1, 2) for matrix multiplication
# Think of it like: single row → needs to be a 1-row matrix
example_1 = X_train[0]  # This is shape (2,) - just an array [0, 0]
print(f"Original example: {example_1}, shape: {example_1.shape}")

# Reshape to (1, 2) - one row, two columns
example_1_reshaped = example_1.reshape(1, -1)
print(f"Reshaped example: {example_1_reshaped}, shape: {example_1_reshaped.shape}")

print("""
WHY RESHAPE?
  Original shape (2,)   → Just a 1D array: [0, 0]
  Reshaped to (1, 2)    → A 2D matrix with 1 row: [[0, 0]]

  Matrix multiplication needs 2D arrays!
  Shape (1, 2) @ shape (2,) = valid multiplication

  C# ANALOGY:
    int[] array;           // 1D array (2,)
    int[,] matrix;         // 2D array (1, 2)
    Matrix math needs the second one!
""")

# Calculate the weighted sum
weighted_sum = example_1_reshaped @ weights + bias
print(f"\nCalculation:")
print(f"  weighted_sum = {example_1_reshaped} @ {weights} + {bias}")
print(f"  weighted_sum = {weighted_sum[0]:.4f}")

# Apply step function
prediction = 1 if weighted_sum[0] > 0 else 0
print(f"\nStep function:")
print(f"  Is {weighted_sum[0]:.4f} > 0? {weighted_sum[0] > 0}")
print(f"  Prediction: {prediction}")
print(f"  Actual label: {y_train[0]}")

# ==============================================================================
# PART 5: Understanding the Training Step (Learning from Mistakes)
# ==============================================================================

print("\n" + "=" * 80)
print("PART 5: Training Step - Learning from ALL Examples at Once")
print("=" * 80)

print("""
Now we'll predict for ALL 4 examples at once (this is called "batch processing")
and update weights based on ALL mistakes together.

This is more efficient than updating after each example!
""")

# Make predictions for ALL training examples
print("Step 1: Make predictions for all examples")
print("-" * 80)

# X_train @ weights: multiply all 4 examples by weights at once
# Shape: (4, 2) @ (2,) = (4,)
weighted_sums = X_train @ weights + bias
print(f"X_train @ weights + bias:")
print(f"  X_train shape: {X_train.shape}")
print(f"  weights shape: {weights.shape}")
print(f"  Result shape: {weighted_sums.shape}")
print(f"  Weighted sums: {weighted_sums}")

# Apply step function to all predictions
predictions = (weighted_sums > 0).astype(int)
print(f"\nPredictions (after step function):")
print(f"  {predictions}")

print(f"\nActual labels:")
print(f"  {y_train}")

# ==============================================================================
# PART 6: Calculate Errors
# ==============================================================================

print("\n" + "=" * 80)
print("PART 6: Calculate Errors (Where Did We Go Wrong?)")
print("=" * 80)

errors = y_train - predictions
print(f"Errors = Actual - Predicted")
print(f"Errors = {y_train} - {predictions}")
print(f"Errors = {errors}")

print("""
ERROR INTERPRETATION:
  error =  0  → Correct prediction!
  error =  1  → Predicted 0, should be 1 (too pessimistic, increase weights)
  error = -1  → Predicted 1, should be 0 (too optimistic, decrease weights)
""")

for i, error in enumerate(errors):
    status = "Correct!" if error == 0 else "Wrong!"
    print(f"  Example {i+1}: error = {error:2d} → {status}")

# ==============================================================================
# PART 7: Update Weights (THE LEARNING PART!)
# ==============================================================================

print("\n" + "=" * 80)
print("PART 7: Update Weights - This is Where Learning Happens!")
print("=" * 80)

print("""
WEIGHT UPDATE FORMULA:
  new_weights = old_weights + (learning_rate * X_train.T @ errors)

Let's break down X_train.T @ errors:
""")

print(f"X_train (our training data):")
print(X_train)
print(f"  Shape: {X_train.shape} (4 examples, 2 features)")

print(f"\nX_train.T (transposed - flip rows and columns):")
print(X_train.T)
print(f"  Shape: {X_train.T.shape} (2 features, 4 examples)")

print(f"\nerrors (how wrong we were):")
print(f"  {errors}")
print(f"  Shape: {errors.shape} (4 errors, one per example)")

print("""
WHY TRANSPOSE?
  We need: (2,) result (one update per weight)
  X_train.T @ errors = (2, 4) @ (4,) = (2,) ✓

  This calculates how much EACH FEATURE contributed to the errors!
""")

# Calculate the weight update
weight_update = X_train.T @ errors
print(f"\nX_train.T @ errors = {weight_update}")
print(f"  Shape: {weight_update.shape}")

print("""
WHAT THIS MEANS:
  weight_update[0] = How much to adjust the 'tired' weight
  weight_update[1] = How much to adjust the 'morning' weight

  The math automatically:
  - Sums up how much each feature contributed to ALL mistakes
  - Gives us the direction to move each weight
""")

print(f"\nDetailed calculation of weight updates:")
print(f"  For 'tired' weight:")
print(f"    X_train.T[0] @ errors = {X_train.T[0]} @ {errors}")
print(f"    = {X_train.T[0][0]}*{errors[0]} + {X_train.T[0][1]}*{errors[1]} + {X_train.T[0][2]}*{errors[2]} + {X_train.T[0][3]}*{errors[3]}")
print(f"    = {weight_update[0]}")

print(f"\n  For 'morning' weight:")
print(f"    X_train.T[1] @ errors = {X_train.T[1]} @ {errors}")
print(f"    = {X_train.T[1][0]}*{errors[0]} + {X_train.T[1][1]}*{errors[1]} + {X_train.T[1][2]}*{errors[2]} + {X_train.T[1][3]}*{errors[3]}")
print(f"    = {weight_update[1]}")

# Update weights
old_weights = weights.copy()
weights = weights + learning_rate * weight_update

print(f"\nWeight Update:")
print(f"  Old weights: {old_weights}")
print(f"  + learning_rate * weight_update:")
print(f"  + {learning_rate} * {weight_update}")
print(f"  = {learning_rate * weight_update}")
print(f"  New weights: {weights}")

# ==============================================================================
# PART 8: Update Bias (THE CONFUSING PART!)
# ==============================================================================

print("\n" + "=" * 80)
print("PART 8: Update Bias - Why errors.sum()?")
print("=" * 80)

print("""
BIAS UPDATE FORMULA:
  new_bias = old_bias + (learning_rate * errors.sum())

WHY errors.sum() and not just errors?
""")

print(f"errors: {errors}")
print(f"errors.sum(): {errors.sum()}")

print("""
INTUITION:
  - Weights are tied to FEATURES (tired, morning)
  - Bias is NOT tied to any feature - it's a constant offset

  Think of bias as: "Overall, do I tend to buy coffee?"

  - If we make MORE mistakes predicting "yes" when it should be "no":
    errors.sum() is NEGATIVE → decrease bias (be less likely to say yes)

  - If we make MORE mistakes predicting "no" when it should be "yes":
    errors.sum() is POSITIVE → increase bias (be more likely to say yes)

MATHEMATICAL REASON:
  For weights, we multiply by X: learning_rate * X.T @ errors
  For bias, think of it as multiplied by 1 (constant):
    learning_rate * [1, 1, 1, 1] @ errors
    = learning_rate * (errors[0] + errors[1] + errors[2] + errors[3])
    = learning_rate * errors.sum()

  C# ANALOGY:
    weights += learningRate * features.DotProduct(errors);  // tied to features
    bias += learningRate * errors.Sum();                    // tied to nothing (constant)
""")

old_bias = bias
bias = bias + learning_rate * errors.sum()

print(f"Bias Update:")
print(f"  Old bias: {old_bias:.4f}")
print(f"  + learning_rate * errors.sum():")
print(f"  + {learning_rate} * {errors.sum()}")
print(f"  = {learning_rate * errors.sum():.4f}")
print(f"  New bias: {bias:.4f}")

# ==============================================================================
# PART 9: Complete Training Loop
# ==============================================================================

print("\n" + "=" * 80)
print("PART 9: Complete Training Loop (Putting It All Together)")
print("=" * 80)

# Reset to start fresh
weights = np.random.randn(2) * 0.01
bias = 0.0
learning_rate = 0.1

print("Starting fresh training...")
print(f"Initial weights: {weights}")
print(f"Initial bias: {bias:.4f}\n")

# Train for multiple epochs
for epoch in range(10):
    # 1. Make predictions
    weighted_sums = X_train @ weights + bias
    predictions = (weighted_sums > 0).astype(int)

    # 2. Calculate errors
    errors = y_train - predictions
    n_mistakes = (errors != 0).sum()  # Count how many wrong

    # 3. Update weights
    weights = weights + learning_rate * (X_train.T @ errors)

    # 4. Update bias
    bias = bias + learning_rate * errors.sum()

    # Print progress
    print(f"Epoch {epoch:2d}: {n_mistakes} mistakes | weights: {weights} | bias: {bias:.3f}")

    # Stop if perfect
    if n_mistakes == 0:
        print(f"\n  Perfect! Learned the pattern in {epoch + 1} epochs!")
        break

# ==============================================================================
# PART 10: Test the Trained Perceptron
# ==============================================================================

print("\n" + "=" * 80)
print("PART 10: Testing the Trained Perceptron")
print("=" * 80)

print("Final weights:", weights)
print(f"Final bias: {bias:.3f}\n")

print("Testing all training examples:")
print("-" * 80)

for i in range(len(X_train)):
    # Get the example
    example = X_train[i].reshape(1, -1)  # Reshape for matrix multiplication

    # Make prediction
    weighted_sum = example @ weights + bias
    prediction = 1 if weighted_sum[0] > 0 else 0

    # Check correctness
    actual = y_train[i]
    status = "✓" if prediction == actual else "✗"

    # Interpret
    tired_str = "Tired" if X_train[i][0] == 1 else "Not tired"
    morning_str = "Morning" if X_train[i][1] == 1 else "Not morning"
    decision_str = "Buy coffee" if prediction == 1 else "Don't buy"

    print(f"Day {i+1}: {tired_str}, {morning_str}")
    print(f"  → Weighted sum: {weighted_sum[0]:.3f}")
    print(f"  → Prediction: {prediction} ({decision_str})")
    print(f"  → Actual: {actual} {status}\n")

# ==============================================================================
# PART 11: Test on New Data
# ==============================================================================

print("=" * 80)
print("PART 11: Predicting New Situations (Generalization)")
print("=" * 80)

print("\nLet's test situations the perceptron has never seen during training:")
print("-" * 80)

# New situation: Tired but not morning (already in training, but let's show it)
new_situations = [
    ([0, 0], "Not tired, Not morning"),
    ([1, 1], "Tired, Morning"),
    ([1, 0], "Tired, Not morning"),
    ([0, 1], "Not tired, Morning"),
]

for situation, description in new_situations:
    x_new = np.array(situation).reshape(1, -1)
    weighted_sum = x_new @ weights + bias
    prediction = 1 if weighted_sum[0] > 0 else 0
    decision = "Buy coffee" if prediction == 1 else "Don't buy coffee"

    print(f"{description}")
    print(f"  → Weighted sum: {weighted_sum[0]:.3f}")
    print(f"  → Decision: {decision}\n")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("=" * 80)
print("SUMMARY - Key Takeaways")
print("=" * 80)

print("""
1. RESHAPING:
   - reshape(1, -1) converts 1D array to 2D matrix for matrix multiplication
   - Needed because @ operator requires 2D arrays
   - C# equivalent: converting int[] to int[,]

2. BIAS UPDATE (errors.sum()):
   - Weights are tied to features (X.T @ errors)
   - Bias is a constant (not tied to any feature)
   - Sum all errors to get overall tendency adjustment
   - Like asking: "Overall, should I be more or less likely to say yes?"

3. TRAINING PROCESS:
   Step 1: Predict (forward pass)
   Step 2: Calculate errors (actual - predicted)
   Step 3: Update weights (adjust based on features)
   Step 4: Update bias (adjust overall tendency)
   Step 5: Repeat until no mistakes!

4. WHY BATCH PROCESSING:
   - Process all examples at once using matrix operations
   - More efficient than one-by-one
   - Matrix multiplication does the heavy lifting
   - Updates are averaged across all examples

5. CONNECTION TO .NET:
   - Perceptron = Like a simple if-else decision tree
   - Training = Like tuning parameters in a config file
   - Weights = Like coefficients in a linear equation
   - Matrix operations = Like LINQ operations on collections

You now understand:
- How a perceptron makes decisions
- Why we reshape data for matrix operations
- Why bias uses errors.sum()
- How the training loop works step-by-step

Next: Try the original example again - it will make much more sense now!
""")

print("=" * 80)
print("Now you're ready for more complex examples!")
print("=" * 80)

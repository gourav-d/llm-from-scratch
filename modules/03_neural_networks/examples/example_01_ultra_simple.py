"""
ULTRA SIMPLE PERCEPTRON - ONE EXAMPLE AT A TIME
================================================

This example shows training by processing ONE example at a time,
which is easier to understand than batch processing.

Real-World Example: "Will it Rain?"
"""

import numpy as np

print("=" * 80)
print("ULTRA SIMPLE PERCEPTRON: Will it Rain?")
print("=" * 80)

# ==============================================================================
# THE PROBLEM
# ==============================================================================

print("""
SCENARIO: Predict if it will rain based on two simple observations.

INPUTS:
  x1 = Are there clouds? (1 = Yes, 0 = No)
  x2 = Is the humidity high? (1 = Yes, 0 = No)

OUTPUT:
  y = 1 means "It will rain"
  y = 0 means "It won't rain"

TRAINING DATA (what actually happened):
  Day 1: No clouds (0), Low humidity (0)  → No rain (0)
  Day 2: No clouds (0), High humidity (1) → No rain (0)
  Day 3: Clouds (1),    Low humidity (0)  → Rain (1)
  Day 4: Clouds (1),    High humidity (1) → Rain (1)

Pattern: If there are clouds, it rains! (Humidity doesn't matter much)
""")

# Training data
examples = [
    {"x1": 0, "x2": 0, "y": 0, "desc": "No clouds, Low humidity → No rain"},
    {"x1": 0, "x2": 1, "y": 0, "desc": "No clouds, High humidity → No rain"},
    {"x1": 1, "x2": 0, "y": 1, "desc": "Clouds, Low humidity → Rain"},
    {"x1": 1, "x2": 1, "y": 1, "desc": "Clouds, High humidity → Rain"},
]

# ==============================================================================
# INITIALIZE THE PERCEPTRON
# ==============================================================================

print("\n" + "=" * 80)
print("INITIALIZATION")
print("=" * 80)

# Start with small random weights
w1 = 0.01  # Weight for clouds
w2 = 0.01  # Weight for humidity
bias = 0.0 # Baseline tendency
learning_rate = 0.3

print(f"w1 (clouds weight):    {w1:.3f}")
print(f"w2 (humidity weight):  {w2:.3f}")
print(f"bias:                  {bias:.3f}")
print(f"learning_rate:         {learning_rate}")

# ==============================================================================
# TRAINING - ONE EXAMPLE AT A TIME
# ==============================================================================

print("\n" + "=" * 80)
print("TRAINING - Processing Each Example One by One")
print("=" * 80)

print("""
We'll do 3 complete passes through all 4 examples.
Each pass is called an "epoch".

For EACH example, we:
  1. Make a prediction
  2. Calculate the error
  3. Update weights immediately
  4. Move to next example
""")

# Train for 3 epochs
for epoch in range(3):
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch + 1}")
    print(f"{'='*80}")

    total_mistakes = 0

    # Process each training example
    for i, example in enumerate(examples):
        print(f"\n{'-'*80}")
        print(f"Day {i+1}: {example['desc']}")
        print(f"{'-'*80}")

        # Get the input values
        x1 = example["x1"]
        x2 = example["x2"]
        actual = example["y"]

        # STEP 1: Make a prediction (Forward Pass)
        print(f"\nSTEP 1: Make Prediction")
        print(f"  Inputs: x1={x1}, x2={x2}")
        print(f"  Current weights: w1={w1:.3f}, w2={w2:.3f}, bias={bias:.3f}")

        # Calculate weighted sum
        weighted_sum = (x1 * w1) + (x2 * w2) + bias
        print(f"  Weighted sum = (x1 × w1) + (x2 × w2) + bias")
        print(f"  Weighted sum = ({x1} × {w1:.3f}) + ({x2} × {w2:.3f}) + {bias:.3f}")
        print(f"  Weighted sum = {weighted_sum:.3f}")

        # Apply step function (threshold at 0)
        prediction = 1 if weighted_sum > 0 else 0
        print(f"  Step function: Is {weighted_sum:.3f} > 0? {weighted_sum > 0}")
        print(f"  Prediction: {prediction} ({'Rain' if prediction == 1 else 'No rain'})")

        # STEP 2: Calculate error
        print(f"\nSTEP 2: Calculate Error")
        error = actual - prediction
        print(f"  Actual: {actual}")
        print(f"  Prediction: {prediction}")
        print(f"  Error = Actual - Prediction = {actual} - {prediction} = {error}")

        if error == 0:
            print(f"  Correct! No update needed.")
        elif error > 0:
            print(f"  Wrong! Predicted too low (said no rain, but it rained)")
            print(f"  → Need to INCREASE weights that were active")
        else:
            print(f"  Wrong! Predicted too high (said rain, but it didn't)")
            print(f"  → Need to DECREASE weights that were active")

        # Track mistakes
        if error != 0:
            total_mistakes += 1

        # STEP 3: Update weights (Learning!)
        print(f"\nSTEP 3: Update Weights")

        # Update rule: new_weight = old_weight + (learning_rate × error × input)
        old_w1 = w1
        old_w2 = w2
        old_bias = bias

        w1 = w1 + (learning_rate * error * x1)
        w2 = w2 + (learning_rate * error * x2)
        bias = bias + (learning_rate * error * 1)  # Bias input is always 1

        print(f"  w1 update:")
        print(f"    new_w1 = old_w1 + (learning_rate × error × x1)")
        print(f"    new_w1 = {old_w1:.3f} + ({learning_rate} × {error} × {x1})")
        print(f"    new_w1 = {old_w1:.3f} + {learning_rate * error * x1:.3f}")
        print(f"    new_w1 = {w1:.3f}")

        print(f"  w2 update:")
        print(f"    new_w2 = old_w2 + (learning_rate × error × x2)")
        print(f"    new_w2 = {old_w2:.3f} + ({learning_rate} × {error} × {x2})")
        print(f"    new_w2 = {old_w2:.3f} + {learning_rate * error * x2:.3f}")
        print(f"    new_w2 = {w2:.3f}")

        print(f"  bias update:")
        print(f"    new_bias = old_bias + (learning_rate × error × 1)")
        print(f"    new_bias = {old_bias:.3f} + ({learning_rate} × {error} × 1)")
        print(f"    new_bias = {old_bias:.3f} + {learning_rate * error:.3f}")
        print(f"    new_bias = {bias:.3f}")

        print(f"\n  KEY INSIGHT:")
        if x1 == 1 and error != 0:
            direction = "increased" if error > 0 else "decreased"
            print(f"    Clouds were present (x1=1), so w1 {direction} by {abs(learning_rate * error):.3f}")
        elif x1 == 0:
            print(f"    No clouds (x1=0), so w1 stays the same")

        if x2 == 1 and error != 0:
            direction = "increased" if error > 0 else "decreased"
            print(f"    High humidity (x2=1), so w2 {direction} by {abs(learning_rate * error):.3f}")
        elif x2 == 0:
            print(f"    Low humidity (x2=0), so w2 stays the same")

    # End of epoch summary
    print(f"\n{'='*80}")
    print(f"End of Epoch {epoch + 1}: {total_mistakes} mistakes")
    print(f"Current weights: w1={w1:.3f}, w2={w2:.3f}, bias={bias:.3f}")
    print(f"{'='*80}")

    if total_mistakes == 0:
        print(f"\n  PERFECT! The perceptron learned the pattern!")
        break

# ==============================================================================
# TESTING
# ==============================================================================

print("\n" + "=" * 80)
print("TESTING THE TRAINED PERCEPTRON")
print("=" * 80)

print(f"\nFinal weights:")
print(f"  w1 (clouds):   {w1:.3f}")
print(f"  w2 (humidity): {w2:.3f}")
print(f"  bias:          {bias:.3f}")

print(f"\nNotice: w1 is MUCH larger than w2!")
print(f"This means the perceptron learned that CLOUDS matter more than HUMIDITY.")

print(f"\nTesting all scenarios:")
print("-" * 80)

for i, example in enumerate(examples):
    x1 = example["x1"]
    x2 = example["x2"]
    actual = example["y"]

    weighted_sum = (x1 * w1) + (x2 * w2) + bias
    prediction = 1 if weighted_sum > 0 else 0

    status = "✓" if prediction == actual else "✗"
    print(f"Day {i+1}: {example['desc']}")
    print(f"  Weighted sum: {weighted_sum:.3f}")
    print(f"  Prediction: {prediction}, Actual: {actual} {status}\n")

# ==============================================================================
# COMPARISON: ONE-AT-A-TIME vs BATCH
# ==============================================================================

print("=" * 80)
print("ONE-AT-A-TIME vs BATCH TRAINING")
print("=" * 80)

print("""
What we just did: ONE-AT-A-TIME (Online Learning)
  ✓ Update weights after EACH example
  ✓ Easier to understand
  ✓ Good for streaming data
  ✗ Can be slower
  ✗ Updates can be noisy (zig-zag path)

What the original example does: BATCH Training
  ✓ Process ALL examples first
  ✓ Calculate ALL errors
  ✓ Sum up all weight updates
  ✓ Update weights ONCE per epoch
  ✓ Faster (uses matrix operations)
  ✓ Smoother learning (averaged updates)

BATCH FORMULA (from original code):
  weights += learning_rate * (X.T @ errors)
  bias += learning_rate * errors.sum()

This is EQUIVALENT to:
  for each example:
      weight_update_1 += learning_rate * error * x1
      weight_update_2 += learning_rate * error * x2
      bias_update += learning_rate * error

  weights[0] += weight_update_1
  weights[1] += weight_update_2
  bias += bias_update

The matrix version (X.T @ errors) does this automatically!

WHY errors.sum() for bias?
  Because bias is multiplied by 1 (constant) for every example:
    bias_update = (error_1 * 1) + (error_2 * 1) + (error_3 * 1) + (error_4 * 1)
                = error_1 + error_2 + error_3 + error_4
                = errors.sum()

C# ANALOGY:
  // One-at-a-time (what we just did)
  foreach (var example in examples) {
      var error = actual - prediction;
      w1 += learningRate * error * example.X1;
      w2 += learningRate * error * example.X2;
      bias += learningRate * error * 1;
  }

  // Batch (what original example does)
  var errors = actual.Zip(predictions, (a, p) => a - p);
  var w1Update = examples.Zip(errors, (ex, err) => err * ex.X1).Sum();
  var w2Update = examples.Zip(errors, (ex, err) => err * ex.X2).Sum();
  var biasUpdate = errors.Sum();

  w1 += learningRate * w1Update;
  w2 += learningRate * w2Update;
  bias += learningRate * biasUpdate;
""")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
KEY CONCEPTS YOU NOW UNDERSTAND:

1. FORWARD PASS (Prediction):
   - Multiply inputs by weights
   - Add bias
   - Apply step function (> 0 → 1, else 0)

2. ERROR CALCULATION:
   - error = actual - prediction
   - Positive error → predicted too low
   - Negative error → predicted too high
   - Zero error → perfect!

3. WEIGHT UPDATE:
   - new_weight = old_weight + (learning_rate × error × input)
   - If input was 0, weight doesn't change (it wasn't involved)
   - If input was 1, weight changes by (learning_rate × error)
   - Direction: positive error → increase, negative error → decrease

4. BIAS UPDATE:
   - new_bias = old_bias + (learning_rate × error × 1)
   - Always updates by (learning_rate × error)
   - Bias is like a "default tendency" independent of inputs

5. WHY IT WORKS:
   - Weights learn which features matter
   - Bias learns the overall tendency
   - Mistakes adjust the parameters
   - Eventually, predictions match reality

6. BATCH vs ONE-AT-A-TIME:
   - One-at-a-time: Update after each example
   - Batch: Sum all updates, then apply once
   - Batch uses matrix math (X.T @ errors) for efficiency
   - errors.sum() is the batch equivalent for bias

NOW GO BACK TO THE ORIGINAL EXAMPLE:
  The confusing parts should make sense now!
  - reshape(): Needed for matrix multiplication
  - X.T @ errors: Batch weight updates for all features
  - errors.sum(): Batch bias update
  - train() method: Just repeats the process for many epochs

You've got this!
""")

print("=" * 80)

"""
Module 02.5 - Classical Machine Learning
Example 01: Multiple Regression from Scratch

GLOSSARY
--------
Feature      : One input variable. House price example: size, bedrooms, age are features.
               Like a column in a spreadsheet. In C#: a field in your data class.

Target       : The value you want to predict. House price = the target.
               Like the return value of a function that takes features as input.

Weight       : How much each feature contributes to the prediction.
               Learned from data. C# analogy: a calibrated coefficient in a formula.

Design Matrix: Matrix X where each row = one sample, each column = one feature.
               Shape: (n_samples, n_features). Like a 2D array of training examples.

Bias Column  : A column of 1s added to X. Allows the model to learn an offset term.
               Without it, the predicted line must pass through the origin.

Normal Equation: Formula to find optimal weights in one step: W = inv(X.T @ X) @ X.T @ y
               Only works when (X.T @ X) is invertible (features not perfectly correlated).

MSE          : Mean Squared Error. Average of (prediction - actual)^2. Lower = better.

R-squared    : Fraction of variance explained. 1.0 = perfect, 0.0 = no better than mean.
"""

import numpy as np   # NumPy: array math (like System.Math + arrays in C#)

print("=" * 60)
print("Example 01: Multiple Regression from Scratch")
print("=" * 60)
print()

# ----------------------------------------------------------------
# STEP 1: Create synthetic training data
# ----------------------------------------------------------------
# In real life you would load this from a CSV file.
# Here we create fake house data to keep things simple.
# C# analogy: this is like constructing a List<HouseRecord> by hand.

print("--- Step 1: Create Training Data ---")
print()

# Each row is one house: [size_sqft, bedrooms, distance_miles]
# These are our FEATURES (inputs)
X_raw = np.array([
    [1000, 2, 10],   # House 1: small, 2br, far from city
    [1500, 3,  5],   # House 2: medium, 3br, mid-distance
    [2000, 4,  2],   # House 3: large, 4br, close to city
    [1200, 2,  8],   # House 4: smallish, 2br, fairly far
    [1800, 3,  3],   # House 5: large, 3br, close
    [2500, 5,  1],   # House 6: very large, 5br, very close
    [900,  1, 12],   # House 7: tiny, 1br, very far
    [1600, 3,  6],   # House 8: medium-large, 3br, mid-distance
], dtype=float)      # dtype=float ensures NumPy uses decimal numbers

# These are the TARGET values (house prices in dollars)
y = np.array([
    150000,   # House 1 price
    220000,   # House 2 price
    310000,   # House 3 price
    175000,   # House 4 price
    285000,   # House 5 price
    400000,   # House 6 price
    130000,   # House 7 price
    240000,   # House 8 price
], dtype=float)

print(f"X_raw shape: {X_raw.shape}")   # Should be (8, 3): 8 houses, 3 features
print(f"y shape:     {y.shape}")        # Should be (8,): 8 price targets
print()
print("First 3 rows of X_raw (size, bedrooms, distance):")
print(X_raw[:3])                        # Show first 3 rows
print()

# ----------------------------------------------------------------
# STEP 2: Build the Design Matrix (add bias column of 1s)
# ----------------------------------------------------------------
# The Normal Equation expects a column of 1s in X.
# This column lets the model learn a constant offset (bias / intercept).
# Without it, the line is forced through the origin, which is rarely correct.
#
# C# analogy:
#   var X = new double[n, features + 1];
#   for (int i = 0; i < n; i++) X[i, 0] = 1.0;  // bias column

print("--- Step 2: Add Bias Column ---")
print()

n_samples = X_raw.shape[0]                         # Number of rows (8 houses)
n_features = X_raw.shape[1]                        # Number of feature columns (3)

# np.ones creates an array of 1s
# Shape (8, 1): one column of 1s, ready to be glued to X_raw
bias_column = np.ones((n_samples, 1))              # Shape: (8, 1)

# np.hstack stacks arrays side by side (horizontally)
# Like Array.Concat in the column direction
X = np.hstack([bias_column, X_raw])               # Shape: (8, 4)

print(f"X shape after adding bias: {X.shape}")     # Should be (8, 4)
print("First 3 rows of X (bias=1, size, bedrooms, distance):")
print(X[:3])
print()

# ----------------------------------------------------------------
# STEP 3: Apply the Normal Equation to find optimal weights W
# ----------------------------------------------------------------
# Formula: W = inv(X.T @ X) @ X.T @ y
#
# Breaking it down:
#   X.T           = transpose of X, shape (4, 8)
#   X.T @ X       = matrix product, shape (4, 4)
#   inv(X.T @ X)  = matrix inverse, shape (4, 4)
#   X.T @ y       = shape (4,)
#   final result W= shape (4,) -- one weight per column of X

print("--- Step 3: Normal Equation ---")
print()

XtX = X.T @ X                          # X-transpose times X, shape (4, 4)
print("X.T @ X  (shape", XtX.shape, "):")
print(np.round(XtX, 1))               # Round for readability
print()

XtX_inv = np.linalg.inv(XtX)          # Matrix inverse; np.linalg = linear algebra tools
print("inv(X.T @ X)  (shape", XtX_inv.shape, "):")
print(np.round(XtX_inv, 8))
print()

Xty = X.T @ y                          # X-transpose times y, shape (4,)
print("X.T @ y  (shape", Xty.shape, "):")
print(np.round(Xty, 0))
print()

W = XtX_inv @ Xty                      # Final weights, shape (4,)
print("Optimal weights W:")
print(f"  w0 (bias/intercept): {W[0]:.2f}")
print(f"  w1 (size weight):    {W[1]:.2f}   <- $per sqft")
print(f"  w2 (bedrooms):       {W[2]:.2f}   <- $per bedroom")
print(f"  w3 (distance):       {W[3]:.2f}   <- $per mile from city")
print()

# ----------------------------------------------------------------
# STEP 4: Make predictions on training data
# ----------------------------------------------------------------
# Prediction = X @ W
# This multiplies each row of X by the weight vector and sums up.
# Shape: (8, 4) @ (4,) = (8,)  -- one prediction per house

print("--- Step 4: Make Predictions ---")
print()

y_pred = X @ W                          # Matrix multiply: shape (8,)

print(f"{'House':>6}  {'Actual':>10}  {'Predicted':>12}  {'Error':>10}")
print("-" * 45)
for i in range(n_samples):
    error = y_pred[i] - y[i]           # Positive = overestimate, negative = underestimate
    print(f"{i+1:>6}  {y[i]:>10.0f}  {y_pred[i]:>12.0f}  {error:>10.0f}")
print()

# ----------------------------------------------------------------
# STEP 5: Compute MSE and R-squared
# ----------------------------------------------------------------
# MSE: mean of squared errors
# R2:  how much variance the model explains

print("--- Step 5: Evaluate Model ---")
print()

errors = y_pred - y                           # Residuals (errors), shape (8,)
squared_errors = errors ** 2                  # Element-wise squaring
mse = np.mean(squared_errors)                 # Average squared error
rmse = np.sqrt(mse)                           # Root MSE, same units as y (dollars)

# R-squared calculation
ss_residual = np.sum((y_pred - y) ** 2)       # Sum of squared prediction errors
ss_total = np.sum((y - np.mean(y)) ** 2)      # Variance of y (vs. just predicting the mean)
r2 = 1 - ss_residual / ss_total              # R2: fraction of variance explained

print(f"MSE:  {mse:>15,.0f}   (lower is better)")
print(f"RMSE: {rmse:>15,.0f}   (same units as price: dollars)")
print(f"R2:   {r2:>15.4f}   (1.0 = perfect, 0.0 = useless)")
print()

# ----------------------------------------------------------------
# STEP 6: Predict a new, unseen house
# ----------------------------------------------------------------
# This is what inference looks like: take a new sample, apply the model.
# C# analogy: call PredictPrice(size, bedrooms, distance) with the
# learned weights instead of hard-coded ones.

print("--- Step 6: Predict a New House ---")
print()

# New house: 1700 sqft, 3 bedrooms, 4 miles from city
new_house = np.array([1.0, 1700.0, 3.0, 4.0])  # Note: bias=1 at front!

predicted_price = new_house @ W                  # Dot product with learned weights
print("New house: 1700 sqft, 3 bedrooms, 4 miles from city")
print(f"Predicted price: ${predicted_price:,.0f}")
print()

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print("Multiple regression finds weights W such that X @ W")
print("is as close to y as possible (minimum squared error).")
print()
print("The Normal Equation solves this in ONE step:")
print("   W = inv(X.T @ X) @ X.T @ y")
print()
print("Neural networks do the SAME thing (X @ W) as their")
print("first operation in every layer, but learn W using")
print("gradient descent instead of the Normal Equation.")
print()
print(f"Our model achieved R2 = {r2:.4f} on training data.")
print("In production you would measure R2 on a HELD-OUT test set.")

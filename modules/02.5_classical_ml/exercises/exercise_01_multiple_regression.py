"""
Module 02.5 - Classical Machine Learning
Exercise 01: Multiple Regression from Scratch

GLOSSARY
--------
Design Matrix: Matrix X where row i = features of sample i, including a leading 1 for bias.
               Shape: (n_samples, n_features + 1).

Normal Equation: W = inv(X.T @ X) @ X.T @ y
               Solves for the weights that minimize sum of squared errors.

MSE          : Mean Squared Error. mean( (y_pred - y_true)^2 ). Lower = better.

R-squared    : 1 - SS_residual / SS_total. 1.0 = perfect fit. 0.0 = useless.

Weight Vector: W[0] = bias, W[1..] = feature weights. Shape: (n_features + 1,).

HOW TO USE THIS FILE
--------------------
1. Read each EXERCISE section carefully.
2. Replace the `pass` statement with your implementation.
3. Run the file: python exercise_01_multiple_regression.py
4. Check that your output matches the EXPECTED OUTPUT comments.
"""

import numpy as np   # NumPy: array math library

print("=" * 60)
print("Exercise 01: Multiple Regression")
print("=" * 60)
print()

# ----------------------------------------------------------------
# Training data (do not modify these)
# ----------------------------------------------------------------
# Features: [size_sqft, bedrooms, age_years]
X_raw = np.array([
    [1000, 2, 20],
    [1500, 3, 10],
    [2000, 4,  5],
    [1200, 2, 15],
    [1800, 3,  8],
    [2500, 5,  2],
    [ 900, 1, 25],
    [1600, 3, 12],
], dtype=float)

# Target: house prices in dollars
y = np.array([
    145000,
    215000,
    310000,
    170000,
    275000,
    390000,
    125000,
    235000,
], dtype=float)

# ============================================================
#  EXERCISE 1
#  Topic: Build the Design Matrix
#
#  Background:
#    Multiple regression requires adding a column of 1s to X.
#    This column corresponds to the bias (intercept) term.
#    Without it, the regression line is forced through the origin.
#
#    If X_raw has shape (8, 3), the result should have shape (8, 4):
#    first column = all 1s, remaining columns = original X_raw.
#
#  Your Task:
#    Implement: add_bias_column(X) -> np.ndarray
#    Prepend a column of 1s to X.
#
#  C# Analogy:
#    // Prepend a bias column of 1s to a 2D array:
#    double[,] AddBias(double[,] X) {
#        int rows = X.GetLength(0), cols = X.GetLength(1);
#        var result = new double[rows, cols + 1];
#        for (int i = 0; i < rows; i++) {
#            result[i, 0] = 1.0;  // bias
#            for (int j = 0; j < cols; j++) result[i, j+1] = X[i, j];
#        }
#        return result;
#    }
#
#  Hint:
#    np.ones((n, 1)) creates a column of 1s with shape (n, 1).
#    np.hstack([a, b]) stacks arrays side by side.
# ============================================================

def add_bias_column(X):
    """
    Add a column of 1s as the FIRST column of X.

    Parameters:
        X : np.ndarray of shape (n_samples, n_features)

    Returns:
        np.ndarray of shape (n_samples, n_features + 1)
    """
    pass   # TODO: replace this line with your implementation


# Test Exercise 1
print("--- Exercise 1: Design Matrix ---")
X = add_bias_column(X_raw)
if X is not None:
    print(f"X_raw shape: {X_raw.shape}  -> X shape after bias: {X.shape}")
    print("First row of X:", X[0])
    # EXPECTED: X shape = (8, 4)
    # EXPECTED: First row = [1. 1000. 2. 20.]
print()

# ============================================================
#  EXERCISE 2
#  Topic: Normal Equation
#
#  Background:
#    Given design matrix X and targets y, compute the optimal weights W.
#    Formula: W = inv(X.T @ X) @ X.T @ y
#
#    Breaking it down:
#      X.T @ X      = square matrix of shape (n_features+1, n_features+1)
#      inv(...)     = matrix inverse: np.linalg.inv(...)
#      X.T @ y      = vector of shape (n_features+1,)
#      W            = vector of shape (n_features+1,)
#
#  Your Task:
#    Implement: normal_equation(X, y) -> np.ndarray
#    Use the Normal Equation to compute weights.
#
#  C# Analogy:
#    // This is like solving the linear system (X'X) * W = X'y
#    // using a matrix solver library (e.g., MathNet.Numerics in C#).
#
#  Hint:
#    np.linalg.inv(M) computes the inverse of matrix M.
#    @ is the matrix multiplication operator.
# ============================================================

def normal_equation(X, y):
    """
    Compute optimal regression weights using the Normal Equation.

    Parameters:
        X : np.ndarray of shape (n_samples, n_features)  (bias column included)
        y : np.ndarray of shape (n_samples,)

    Returns:
        W : np.ndarray of shape (n_features,)
    """
    pass   # TODO: replace with W = inv(X.T @ X) @ X.T @ y


# Test Exercise 2
print("--- Exercise 2: Normal Equation ---")
if X is not None:
    W = normal_equation(X, y)
    if W is not None:
        print("Learned weights W:")
        labels = ["bias", "size", "bedrooms", "age"]
        for i, (label, w) in enumerate(zip(labels, W)):
            print(f"  W[{i}] ({label:>10}): {w:.2f}")
        # EXPECTED: W[1] (size) should be positive (bigger house -> higher price)
        # EXPECTED: W[3] (age) should be negative (older house -> lower price)
print()

# ============================================================
#  EXERCISE 3
#  Topic: Prediction
#
#  Background:
#    Once we have weights W, prediction is a single matrix multiplication.
#    y_pred = X @ W
#    Each row of X gets dot-producted with W.
#
#  Your Task:
#    Implement: predict(X, W) -> np.ndarray
#    Returns predicted values for all rows of X.
#
#  C# Analogy:
#    double[] Predict(double[,] X, double[] W) {
#        // Each row of X dot-producted with W
#        return Enumerable.Range(0, X.GetLength(0))
#            .Select(i => Enumerable.Range(0, W.Length)
#                                   .Sum(j => X[i, j] * W[j]))
#            .ToArray();
#    }
# ============================================================

def predict(X, W):
    """
    Compute predictions for all samples.

    Parameters:
        X : np.ndarray of shape (n_samples, n_features)
        W : np.ndarray of shape (n_features,)

    Returns:
        np.ndarray of shape (n_samples,)
    """
    pass   # TODO: one line: return X @ W


# Test Exercise 3
print("--- Exercise 3: Predictions ---")
if X is not None and W is not None:
    y_pred = predict(X, W)
    if y_pred is not None:
        print(f"{'House':>6}  {'Actual Price':>14}  {'Predicted Price':>16}")
        print("-" * 42)
        for i in range(len(y)):
            print(f"{i+1:>6}  {y[i]:>14,.0f}  {y_pred[i]:>16,.0f}")
print()

# ============================================================
#  EXERCISE 4
#  Topic: MSE and R-squared
#
#  Background:
#    MSE = mean( (y_pred - y_true)^2 )
#    R2  = 1 - sum((y_pred - y_true)^2) / sum((y_true - mean(y_true))^2)
#
#    R2 = 1.0 means perfect predictions.
#    R2 = 0.0 means the model is no better than always predicting the mean.
#
#  Your Task:
#    Implement: compute_mse(y_true, y_pred) -> float
#    Implement: compute_r2(y_true, y_pred) -> float
#
#  C# Analogy:
#    double ComputeMSE(double[] truth, double[] pred) =>
#        truth.Zip(pred, (a, b) => Math.Pow(b - a, 2)).Average();
# ============================================================

def compute_mse(y_true, y_pred):
    """
    Mean Squared Error.

    Parameters:
        y_true : np.ndarray of shape (n,)
        y_pred : np.ndarray of shape (n,)

    Returns:
        float: the MSE value
    """
    pass   # TODO: return mean of squared differences


def compute_r2(y_true, y_pred):
    """
    R-squared (coefficient of determination).

    Parameters:
        y_true : np.ndarray of shape (n,)
        y_pred : np.ndarray of shape (n,)

    Returns:
        float: R2 value between 0.0 and 1.0 (can be negative if model is terrible)
    """
    pass   # TODO: return 1 - (ss_residual / ss_total)


# Test Exercise 4
print("--- Exercise 4: Metrics ---")
if y_pred is not None:
    mse = compute_mse(y, y_pred)
    r2  = compute_r2(y, y_pred)
    if mse is not None and r2 is not None:
        print(f"MSE:  {mse:>15,.0f}")
        print(f"RMSE: {np.sqrt(mse):>15,.0f}  dollars")
        print(f"R2:   {r2:>15.4f}  (1.0 = perfect)")
        # EXPECTED: R2 should be very high (> 0.99) on training data
print()

# ============================================================
#  EXERCISE 5
#  Topic: Predict a new house
#
#  Background:
#    After training, you can predict the price of any new house
#    by feeding its features through the same formula:
#      price = x_new @ W
#    where x_new must include the bias term (1 as first element).
#
#  Your Task:
#    Implement: predict_new(features, W) -> float
#    Given raw feature values (WITHOUT bias), add the bias and predict.
#
#  C# Analogy:
#    double PredictNew(double[] features, double[] W) {
#        var x = new double[] { 1.0 }.Concat(features).ToArray();
#        return x.Zip(W, (xi, wi) => xi * wi).Sum();
#    }
# ============================================================

def predict_new(features, W):
    """
    Predict price for one new house.

    Parameters:
        features : list or np.ndarray of raw feature values (NO bias)
                   e.g., [size, bedrooms, age]
        W        : np.ndarray of learned weights (includes bias at W[0])

    Returns:
        float: predicted price
    """
    pass   # TODO: prepend 1.0 to features, then dot-product with W


# Test Exercise 5
print("--- Exercise 5: New House Prediction ---")
if W is not None:
    # New house: 1750 sqft, 3 bedrooms, 7 years old
    new_features = [1750, 3, 7]
    price = predict_new(new_features, W)
    if price is not None:
        print(f"New house: {new_features[0]} sqft, {new_features[1]} bed, {new_features[2]} yrs old")
        print(f"Predicted price: ${price:,.0f}")
        # EXPECTED: somewhere in the $200,000-$280,000 range
print()

print("=" * 60)
print("All exercises complete! Check your output matches the expected ranges.")
print("=" * 60)

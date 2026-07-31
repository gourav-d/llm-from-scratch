"""
Module 02.5 - Classical Machine Learning
Example 02: Regularization from Scratch (Ridge, Lasso, Polynomial Features)

GLOSSARY
--------
Overfitting    : Model memorizes training data. Works great on training set,
                 fails on new data. C# analogy: code that only passes the
                 unit tests it was written against, breaks on real usage.

Underfitting   : Model is too simple to capture the real pattern.
                 Like a one-line function that handles only one case.

Regularization : A penalty added to the loss function that punishes large weights.
                 Forces the model to stay simple even if it could memorize data.

Lambda         : Regularization strength hyperparameter (you choose this).
                 High lambda = heavy penalty = simpler model.
                 Low lambda  = light penalty = more complex model.

Ridge (L2)     : Penalty = lambda * sum(W^2). Shrinks all weights toward zero.
                 C# analogy: a "complexity tax" on every parameter.

Lasso (L1)     : Penalty = lambda * sum(|W|). Can drive weights to exactly zero.
                 Performs automatic feature selection.

Polynomial     : Adding columns x^2, x^3, etc. to model curved relationships.
Features         Turns linear algebra into curve fitting.

Condition Num. : How close a matrix is to singular (uninvertible). High = unstable.
                 Ridge regularization always improves the condition number.
"""

import numpy as np   # NumPy: all numerical operations

print("=" * 60)
print("Example 02: Regularization (Ridge, Lasso, Polynomial)")
print("=" * 60)
print()

# ----------------------------------------------------------------
# PART 1: Ridge Regression (L2)
# ----------------------------------------------------------------
# Ridge modifies the Normal Equation by adding lambda * I to X.T @ X.
# This shrinks all weights toward zero and fixes singular matrices.

print("--- Part 1: Ridge Regression (L2) ---")
print()

# Create a simple 1D dataset with noise to illustrate overfitting
np.random.seed(42)                             # Fix random seed for reproducibility
n = 20                                         # Number of training samples

x_1d = np.linspace(-2, 2, n)                  # 20 evenly spaced x values from -2 to 2
y_true = 3 * x_1d + 5                         # True linear pattern: y = 3x + 5
noise = np.random.randn(n) * 2.0              # Gaussian noise (mean=0, std=2)
y = y_true + noise                            # Observed y = true y + noise

# Build design matrix with bias column
X = np.column_stack([np.ones(n), x_1d])       # Shape: (20, 2): [1, x]

print(f"Data shape: X={X.shape}, y={y.shape}")
print()

def ridge_regression(X, y, lam):
    """
    Compute Ridge regression weights.

    Formula: W = inv(X.T @ X + lam * I) @ X.T @ y

    Adding lam * I (identity matrix scaled by lambda) to X.T @ X:
    1. Ensures the matrix is invertible (fixes singular matrix issue)
    2. Penalizes large weights by shrinking them toward zero

    C# analogy:
        // Regularization modifies the system of equations before solving:
        // (X'X + lambda*I) * W = X'y
        // Instead of just: X'X * W = X'y
    """
    n_features = X.shape[1]                    # Number of columns in X
    I = np.eye(n_features)                     # Identity matrix (1s on diagonal, 0s elsewhere)
    # np.eye creates: [[1,0,...], [0,1,...], ...]
    # Like the multiplicative identity for matrices

    XtX_reg = X.T @ X + lam * I               # Regularized matrix: X.T @ X + lambda * I
    W = np.linalg.inv(XtX_reg) @ X.T @ y      # Solve for weights
    return W

# Try different lambda values to see the effect
lambda_values = [0.0, 0.1, 1.0, 10.0, 100.0]  # 0 = no regularization

print(f"{'Lambda':>10}  {'w0 (bias)':>12}  {'w1 (slope)':>12}")
print("-" * 40)
for lam in lambda_values:
    W = ridge_regression(X, y, lam)           # Compute weights for this lambda
    print(f"{lam:>10.1f}  {W[0]:>12.4f}  {W[1]:>12.4f}")
    # As lambda increases, w1 is pulled toward 0 (true value is ~3.0)

print()
print("Observation: true slope is 3.0. As lambda grows, the estimate is")
print("pulled toward 0 (shrinkage effect). Too high = underfitting.")
print()

# ----------------------------------------------------------------
# PART 2: Lasso via Coordinate Descent (L1)
# ----------------------------------------------------------------
# Lasso has no closed-form solution. We use iterative coordinate descent:
# Update one weight at a time while holding all others fixed.
# The soft-thresholding step is what drives weights to exactly zero.

print("--- Part 2: Lasso Regression (L1) via Coordinate Descent ---")
print()

def soft_threshold(rho, lam):
    """
    Soft thresholding: the key operation in Lasso.

    For each weight w_j, given the partial residual correlation rho_j:
      - If rho_j > lam:  set w_j = rho_j - lam   (reduce by lam)
      - If rho_j < -lam: set w_j = rho_j + lam   (increase toward 0)
      - Otherwise:       set w_j = 0              (exact zero! feature eliminated)

    C# analogy:
        double SoftThreshold(double rho, double lam) {
            if (rho > lam)   return rho - lam;
            if (rho < -lam)  return rho + lam;
            return 0.0;
        }
    """
    if rho > lam:                               # Weight should be positive
        return rho - lam                        # Shrink from positive side
    elif rho < -lam:                            # Weight should be negative
        return rho + lam                        # Shrink from negative side
    else:
        return 0.0                              # Zero it out (feature selection!)

def lasso_coordinate_descent(X, y, lam, n_iterations=1000):
    """
    Lasso regression using coordinate descent.

    At each iteration, we cycle through all weights.
    For weight j, we compute the residual ignoring feature j,
    then find the best w_j for that residual using soft thresholding.

    This guarantees weights can hit exactly 0, unlike Ridge.
    """
    n_samples, n_features = X.shape            # Shape info
    W = np.zeros(n_features)                   # Start all weights at 0

    for iteration in range(n_iterations):       # Repeat many times until convergence
        for j in range(n_features):             # Update one weight at a time
            # Compute residual WITHOUT the contribution of feature j
            # y_pred_without_j = X @ W - X[:, j] * W[j]
            partial_residual = y - X @ W + X[:, j] * W[j]
            # partial_residual is what's left to explain after all other features

            # rho_j = correlation of feature j with the partial residual
            rho_j = X[:, j] @ partial_residual / n_samples
            # This is the "ideal" value for w_j ignoring the penalty

            # Apply soft thresholding to enforce L1 penalty
            if j == 0:                          # Do not regularize the bias term
                W[j] = rho_j                   # Bias is not penalized
            else:
                W[j] = soft_threshold(rho_j, lam)  # Penalize other weights

    return W

# Compare Lasso with different lambda values
print(f"{'Lambda':>10}  {'w0 (bias)':>12}  {'w1 (slope)':>12}  Note")
print("-" * 60)
for lam in [0.0, 0.01, 0.1, 1.0, 5.0]:
    W_lasso = lasso_coordinate_descent(X, y, lam)
    note = "EXACT ZERO" if abs(W_lasso[1]) < 1e-6 else ""
    print(f"{lam:>10.2f}  {W_lasso[0]:>12.4f}  {W_lasso[1]:>12.4f}  {note}")

print()
print("Lasso can set w1 to EXACTLY 0, eliminating the feature entirely.")
print("This is feature selection -- Ridge only shrinks, never zeros.")
print()

# ----------------------------------------------------------------
# PART 3: Polynomial Features
# ----------------------------------------------------------------
# Add x^2, x^3 columns to X so linear algebra can fit curves.
# The model stays linear in W but nonlinear in x.

print("--- Part 3: Polynomial Features ---")
print()

# Create a curved dataset (true pattern: y = x^2 - x + 2 + noise)
x_curve = np.linspace(-3, 3, 30)              # 30 x values from -3 to 3
y_curve = x_curve**2 - x_curve + 2 + np.random.randn(30) * 0.5
# True quadratic pattern with some noise

def make_polynomial_features(x, degree):
    """
    Build design matrix with polynomial features up to 'degree'.

    For degree=3 and input x:
      columns: [1, x, x^2, x^3]

    C# analogy:
        var cols = new List<double[]>();
        cols.Add(Enumerable.Repeat(1.0, n).ToArray());  // bias
        for (int d = 1; d <= degree; d++) {
            cols.Add(x.Select(xi => Math.Pow(xi, d)).ToArray());
        }
    """
    columns = [np.ones(len(x))]                # Start with bias column of 1s
    for d in range(1, degree + 1):             # Add x, x^2, x^3, ..., x^degree
        columns.append(x ** d)                 # Element-wise power: each xi -> xi^d
    return np.column_stack(columns)            # Stack into matrix horizontally

# Fit with different polynomial degrees
print(f"{'Degree':>8}  {'Train MSE':>12}  Notes")
print("-" * 40)

for deg in [1, 2, 3, 6, 10]:
    X_poly = make_polynomial_features(x_curve, deg)   # Build polynomial X
    # Use Ridge with small lambda to avoid singular matrix at high degrees
    W_poly = ridge_regression(X_poly, y_curve, lam=0.001)
    y_poly_pred = X_poly @ W_poly                      # Predictions
    mse = np.mean((y_poly_pred - y_curve) ** 2)       # MSE on training data
    note = ""
    if deg < 2:
        note = "<- underfits (linear vs quadratic truth)"
    elif deg == 2:
        note = "<- near-perfect fit (matches true pattern)"
    elif deg >= 6:
        note = "<- overfitting (wiggly, memorizes noise)"
    print(f"{deg:>8}  {mse:>12.4f}  {note}")

print()
print("Degree 2 fits best because the TRUE pattern IS quadratic.")
print("Higher degrees memorize noise (overfitting) -- lower training MSE")
print("but would generalize poorly to new data.")
print()

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print("Ridge (L2): Adds lambda * sum(W^2) to loss.")
print("  - Modified Normal Eq: W = inv(X.T@X + lam*I) @ X.T @ y")
print("  - Shrinks all weights toward 0, never exact 0")
print("  - Always makes X.T@X invertible (bonus!)")
print()
print("Lasso (L1): Adds lambda * sum(|W|) to loss.")
print("  - Solved iteratively (coordinate descent)")
print("  - Can set weights to EXACTLY 0 -> feature selection")
print()
print("Polynomial Features: Add x^2, x^3, ... columns to X")
print("  - Fits curves using linear algebra")
print("  - High degree + no regularization = overfitting")
print("  - Combine with Ridge for safe curve fitting")

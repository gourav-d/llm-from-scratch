"""
Module 02.5 - Classical Machine Learning
Exercise 02: Regularization from Scratch

GLOSSARY
--------
Lambda       : Regularization strength. You choose this before training.
               High lambda -> heavy penalty -> weights shrink toward 0.
               Low lambda  -> light penalty -> behaves like plain regression.
               C# analogy: a config value passed to the constructor.

Ridge (L2)   : Penalty = lambda * sum(W^2). Shrinks all weights.
               Modified Normal Equation: W = inv(X.T@X + lam*I) @ X.T @ y

Lasso (L1)   : Penalty = lambda * sum(|W|). Can set weights to exactly 0.
               Requires iterative solver (no closed-form like Ridge).

Soft-threshold: The operation that creates Lasso's exact zeros.
               if |rho| > lam: w = rho - sign(rho)*lam  else: w = 0

Polynomial   : Expanding features to include x^2, x^3, etc.
Features       Allows fitting curves using linear algebra.
               High degree without regularization = overfitting.

Identity     : np.eye(n) creates an n x n matrix with 1s on diagonal, 0s elsewhere.
Matrix (I)     Adding lam*I to X.T@X makes it invertible and penalizes weights.

HOW TO USE THIS FILE
--------------------
1. Read each EXERCISE section.
2. Replace `pass` with your implementation.
3. Run the file and compare output to EXPECTED comments.
"""

import numpy as np   # NumPy: all array and matrix operations

print("=" * 60)
print("Exercise 02: Regularization")
print("=" * 60)
print()

# ----------------------------------------------------------------
# Shared data for all exercises (do not modify)
# ----------------------------------------------------------------
np.random.seed(42)                             # Fixed seed for reproducible results
n = 30                                         # 30 training samples

# 1D dataset: true relationship is y = 2*x + 1 + noise
x_1d = np.linspace(-3, 3, n)                  # 30 x values from -3 to 3
y_true_signal = 2 * x_1d + 1                  # True line: y = 2x + 1
y = y_true_signal + np.random.randn(n) * 1.5  # Add Gaussian noise (std=1.5)

# Design matrix with bias column: shape (30, 2)
X_lin = np.column_stack([np.ones(n), x_1d])   # [1, x] for each sample

# ============================================================
#  EXERCISE 1
#  Topic: Ridge Regression Closed-Form Solution
#
#  Background:
#    Standard Normal Equation: W = inv(X.T @ X) @ X.T @ y
#    Ridge modification adds lam * I before taking the inverse:
#      W = inv(X.T @ X + lam * I) @ X.T @ y
#
#    The identity matrix I has the same shape as X.T @ X,
#    which is (n_features, n_features).
#
#    Adding lam * I:
#      1. Ensures the matrix is always invertible (even with correlated features)
#      2. Adds a penalty that shrinks all weights toward 0
#
#  Your Task:
#    Implement: ridge_fit(X, y, lam) -> np.ndarray
#
#  C# Analogy:
#    // Add lambda to each diagonal element of X'X before solving:
#    // (X'X + lambda * I) * W = X'y
#    // Use a linear system solver (MathNet.Numerics) or matrix inverse.
#
#  Hint:
#    np.eye(k) creates a k x k identity matrix.
#    np.linalg.inv(M) computes the inverse of M.
# ============================================================

def ridge_fit(X, y, lam):
    """
    Compute Ridge regression weights.

    Parameters:
        X   : np.ndarray of shape (n_samples, n_features) with bias column
        y   : np.ndarray of shape (n_samples,)
        lam : float, regularization strength (lambda)

    Returns:
        W : np.ndarray of shape (n_features,)
    """
    pass   # TODO: W = inv(X.T @ X + lam * I) @ X.T @ y


# Test Exercise 1
print("--- Exercise 1: Ridge Regression ---")
print(f"{'Lambda':>10}  {'w0 (bias)':>12}  {'w1 (slope)':>12}")
print("-" * 40)
for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
    W_r = ridge_fit(X_lin, y, lam)
    if W_r is not None:
        print(f"{lam:>10.1f}  {W_r[0]:>12.4f}  {W_r[1]:>12.4f}")
    else:
        print(f"{lam:>10.1f}  {'(not implemented)':>25}")
# EXPECTED: true slope = 2.0, true bias = 1.0
# EXPECTED: as lambda grows, w1 shrinks toward 0
print()

# ============================================================
#  EXERCISE 2
#  Topic: Soft-Thresholding (the heart of Lasso)
#
#  Background:
#    Lasso uses a special operation called soft-thresholding
#    to update each weight during coordinate descent.
#
#    Given rho (the ideal weight without penalty) and lam (regularization):
#      if rho > lam:  return rho - lam    (move toward 0 from positive side)
#      if rho < -lam: return rho + lam    (move toward 0 from negative side)
#      else:          return 0.0          (zero it out completely!)
#
#    The "else: return 0" is what makes Lasso drive weights to EXACTLY 0.
#    Ridge never reaches exactly 0, but Lasso can.
#
#  Your Task:
#    Implement: soft_threshold(rho, lam) -> float
#
#  C# Analogy:
#    double SoftThreshold(double rho, double lam) {
#        if (rho > lam)   return rho - lam;
#        if (rho < -lam)  return rho + lam;
#        return 0.0;
#    }
# ============================================================

def soft_threshold(rho, lam):
    """
    Soft thresholding function for Lasso coordinate descent.

    Parameters:
        rho : float, the unconstrained optimal weight for one feature
        lam : float, regularization strength

    Returns:
        float: the regularized weight (can be exactly 0.0)
    """
    pass   # TODO: implement the three-case soft threshold


# Test Exercise 2
print("--- Exercise 2: Soft Thresholding ---")
test_cases = [
    (5.0, 1.0,  "should be 4.0  (5 - 1)"),
    (-5.0, 1.0, "should be -4.0 (-5 + 1)"),
    (0.5, 1.0,  "should be 0.0  (within threshold)"),
    (-0.3, 1.0, "should be 0.0  (within threshold)"),
    (0.0, 1.0,  "should be 0.0"),
]
for rho, lam, desc in test_cases:
    result = soft_threshold(rho, lam)
    print(f"  soft_threshold({rho:>5}, {lam}) = {result if result is not None else 'None':>6}   {desc}")
print()

# ============================================================
#  EXERCISE 3
#  Topic: Polynomial Feature Expansion
#
#  Background:
#    To fit curves, add x^2, x^3, ... as extra columns to X.
#    For degree=3 and input x = [x1, x2, ...]:
#      columns = [1, x, x^2, x^3]
#      shape = (n_samples, degree + 1)
#
#    After expansion, use ordinary linear regression.
#    The model is linear in W but nonlinear in x.
#
#  Your Task:
#    Implement: poly_features(x, degree) -> np.ndarray
#    Build a design matrix with columns [1, x, x^2, ..., x^degree].
#
#  C# Analogy:
#    // Build columns: [1, x, x^2, ..., x^d]
#    var cols = Enumerable.Range(0, degree + 1)
#                         .Select(d => x.Select(xi => Math.Pow(xi, d)).ToArray())
#                         .ToArray();
#
#  Hint:
#    x ** d computes element-wise power (each element of x raised to d).
#    np.column_stack([a, b, c]) stacks arrays as columns.
# ============================================================

def poly_features(x, degree):
    """
    Build polynomial design matrix.

    Parameters:
        x      : np.ndarray of shape (n_samples,) - raw input values
        degree : int, highest polynomial degree to include

    Returns:
        np.ndarray of shape (n_samples, degree + 1)
        Columns: [1, x, x^2, ..., x^degree]
    """
    pass   # TODO: build and return the polynomial design matrix


# Test Exercise 3
print("--- Exercise 3: Polynomial Features ---")
x_small = np.array([1.0, 2.0, 3.0])           # Small test array
X_poly_test = poly_features(x_small, degree=3)
if X_poly_test is not None:
    print(f"poly_features([1,2,3], degree=3) shape: {X_poly_test.shape}")
    print("Matrix:")
    print(X_poly_test)
    # EXPECTED shape: (3, 4)
    # EXPECTED rows:
    #   row 0: [1, 1, 1,  1]   (1^0, 1^1, 1^2, 1^3)
    #   row 1: [1, 2, 4,  8]   (1^0, 2^1, 2^2, 2^3)
    #   row 2: [1, 3, 9, 27]   (1^0, 3^1, 3^2, 3^3)
print()

# ============================================================
#  EXERCISE 4
#  Topic: Compare Ridge vs Plain Regression on Noisy Data
#
#  Background:
#    With noisy data, plain regression (lambda=0) may overfit.
#    Ridge regression (lambda>0) shrinks weights, reducing variance.
#
#    Steps:
#      1. Split x_1d and y into 70% train / 30% test (first 21 = train).
#      2. Fit Ridge with lambda=0 (plain) and lambda=1.0 on train.
#      3. Compute MSE on TRAINING data for both.
#      4. Compute MSE on TEST data for both.
#
#  Your Task:
#    Implement: compute_mse(y_true, y_pred) -> float
#
#  C# Analogy:
#    double MSE(double[] truth, double[] pred) =>
#        truth.Zip(pred, (a, b) => Math.Pow(b - a, 2)).Average();
# ============================================================

def compute_mse(y_true, y_pred):
    """
    Mean Squared Error = mean( (y_true - y_pred)^2 ).

    Parameters:
        y_true : np.ndarray of shape (n,)
        y_pred : np.ndarray of shape (n,)

    Returns:
        float
    """
    pass   # TODO: return the mean of squared differences


# Test Exercise 4
print("--- Exercise 4: Ridge vs Plain Regression (Train/Test MSE) ---")
split = 21                                     # First 21 samples for training
X_tr = X_lin[:split]                           # Training X (21 rows)
y_tr = y[:split]                               # Training y (21 values)
X_te = X_lin[split:]                           # Test X (9 rows)
y_te = y[split:]                               # Test y (9 values)

if ridge_fit(X_lin, y, 0.0) is not None and compute_mse(y, y) is not None:
    print(f"{'Lambda':>8}  {'Train MSE':>12}  {'Test MSE':>12}")
    print("-" * 38)
    for lam in [0.0, 1.0, 10.0]:
        W_cmp = ridge_fit(X_tr, y_tr, lam)
        tr_mse = compute_mse(y_tr, X_tr @ W_cmp)
        te_mse = compute_mse(y_te, X_te @ W_cmp)
        print(f"{lam:>8.1f}  {tr_mse:>12.2f}  {te_mse:>12.2f}")
    # EXPECTED: lambda=0 has lower train MSE but possibly higher test MSE
    # EXPECTED: lambda=1.0 or 10.0 may have higher train MSE but lower test MSE
print()

# ============================================================
#  EXERCISE 5
#  Topic: Polynomial Fit with Ridge Regularization
#
#  Background:
#    Create a curved dataset where the true pattern is y = x^2.
#    Try fitting polynomial degrees 1, 2, and 5.
#    For each degree, compute the training MSE.
#    The degree that matches the true pattern (2) should have the lowest MSE.
#    Degree 5 may have lower training MSE but is overfitting!
#
#  Your Task:
#    Call poly_features(x_curve, degree) to get the design matrix.
#    Call ridge_fit(X_poly, y_curve, lam=0.1) to get weights.
#    Call compute_mse to get the training MSE.
#    Print results for degrees 1, 2, and 5.
#
#  C# Analogy:
#    foreach (int deg in new[] { 1, 2, 5 }) {
#        var X = PolyFeatures(x, deg);
#        var W = RidgeFit(X, y, lambda: 0.1);
#        Console.WriteLine($"deg={deg} MSE={MSE(y, X*W)}");
#    }
# ============================================================

print("--- Exercise 5: Polynomial + Ridge ---")
x_curve = np.linspace(-2, 2, 30)              # 30 x values
y_curve  = x_curve ** 2 + np.random.randn(30) * 0.3  # True pattern: y = x^2 + noise

if (poly_features is not None and ridge_fit(X_lin, y, 0.0) is not None
        and compute_mse(y, y) is not None):
    print(f"{'Degree':>8}  {'Train MSE':>12}  Note")
    print("-" * 45)
    for deg in [1, 2, 5]:
        X_poly = poly_features(x_curve, deg)
        W_poly = ridge_fit(X_poly, y_curve, lam=0.1)
        y_poly_pred = X_poly @ W_poly
        mse_poly    = compute_mse(y_curve, y_poly_pred)
        note = ""
        if deg == 1:
            note = "<- underfits (linear can't fit a curve)"
        elif deg == 2:
            note = "<- should fit well (matches true pattern)"
        elif deg == 5:
            note = "<- may overfit to noise"
        print(f"{deg:>8}  {mse_poly:>12.4f}  {note}")
else:
    print("(Implement exercises 3 and 4 first)")
print()

print("=" * 60)
print("All exercises complete!")
print("Key takeaways:")
print("  Ridge: shrinks weights, always invertible, no exact zeros")
print("  Lasso: soft threshold drives weights to exactly zero")
print("  Polynomial: fit curves with linear algebra; use regularization!")
print("=" * 60)

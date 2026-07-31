"""
Module 02.5 - Classical Machine Learning
Example 05: Model Selection and Evaluation from Scratch

GLOSSARY
--------
Train/Val/Test : Three non-overlapping subsets of data.
Split            Train: fit model weights.
                 Validation: tune hyperparameters. You LOOK at this.
                 Test: final honest evaluation. Look at it ONCE, at the end.
                 C# analogy: unit tests / integration tests / QA acceptance tests.

k-Fold CV      : Split data into k equal folds. Rotate which fold is validation.
                 Average metrics across k runs. Makes better use of limited data.
                 C# analogy: a rotating QA reviewer over k review rounds.

Hyperparameter : Setting chosen BEFORE training. Lambda, max_depth, n_trees.
                 C# analogy: a configuration value passed to the constructor,
                 not learned from data.

Grid Search    : Try every combination of hyperparameter values.
                 Find the combination with lowest validation error.
                 C# analogy: nested for-loops over a config parameter space.

MAE            : Mean Absolute Error. Average of |y_pred - y_true|.
                 Robust to outliers. Units = same as y.

MSE / RMSE     : Mean/Root-Mean Squared Error. Penalizes large errors more.
                 RMSE has same units as y. Preferred when large errors are costly.

R-squared      : Fraction of variance explained. 1.0 = perfect. 0.0 = useless.

Precision      : Of all predicted positives, how many are truly positive?
                 Low precision = too many false alarms.

Recall         : Of all true positives, how many did we find?
                 Low recall = missing real positives.

F1 Score       : Harmonic mean of precision and recall. Balances both.
"""

import numpy as np   # NumPy: array math

print("=" * 60)
print("Example 05: Model Selection and Evaluation from Scratch")
print("=" * 60)
print()

# ----------------------------------------------------------------
# PART 1: Train / Validation / Test Split
# ----------------------------------------------------------------

print("--- Part 1: Data Splitting ---")
print()

# Create synthetic regression data (house prices)
np.random.seed(42)
n = 100                                        # 100 total samples

x = np.random.uniform(500, 3000, n)           # House sizes between 500 and 3000 sqft
y = 100 * x + 50000 + np.random.randn(n) * 20000   # Price = 100*size + 50000 + noise

# Shuffle indices so splits are random
indices = np.random.permutation(n)             # Random ordering of [0..99]

# Split proportions: 60% train, 20% validation, 20% test
train_end = int(0.60 * n)                      # First 60 indices for training
val_end   = int(0.80 * n)                      # Next 20 for validation (60 to 80)
# Remaining 20 are the test set (80 to 100)

train_idx = indices[:train_end]                # Training indices
val_idx   = indices[train_end:val_end]         # Validation indices
test_idx  = indices[val_end:]                  # Test indices (LOCK THESE AWAY!)

X_train = x[train_idx].reshape(-1, 1)         # Shape (60, 1): training features
y_train = y[train_idx]                         # Shape (60,): training targets
X_val   = x[val_idx].reshape(-1, 1)           # Shape (20, 1): validation features
y_val   = y[val_idx]                           # Shape (20,): validation targets
X_test  = x[test_idx].reshape(-1, 1)          # Shape (20, 1): test features (LOCKED)
y_test  = y[test_idx]                          # Shape (20,): test targets (LOCKED)

print(f"Total samples: {n}")
print(f"Training:   {len(X_train)} samples (60%)")
print(f"Validation: {len(X_val)}   samples (20%)")
print(f"Test:       {len(X_test)}   samples (20%) <- LOCKED, not used yet")
print()

# ----------------------------------------------------------------
# PART 2: Evaluation Metrics for Regression
# ----------------------------------------------------------------

print("--- Part 2: Regression Metrics ---")
print()

def add_bias(X):
    """Add a column of 1s to X for the bias term."""
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])     # Prepend column of 1s

def ridge_fit(X, y, lam=0.0):
    """Fit Ridge regression. W = inv(X.T@X + lam*I) @ X.T @ y"""
    n_feat = X.shape[1]
    I = np.eye(n_feat)                         # Identity matrix
    return np.linalg.inv(X.T @ X + lam * I) @ X.T @ y

def compute_regression_metrics(y_true, y_pred):
    """
    Compute MAE, MSE, RMSE, and R-squared.

    All return a dict for easy display and comparison.
    """
    errors      = y_pred - y_true                           # Signed errors
    abs_errors  = np.abs(errors)                            # Absolute errors
    sq_errors   = errors ** 2                               # Squared errors

    mae  = np.mean(abs_errors)                              # Mean Absolute Error
    mse  = np.mean(sq_errors)                               # Mean Squared Error
    rmse = np.sqrt(mse)                                     # Root MSE

    ss_res   = np.sum(sq_errors)                            # Sum of squared residuals
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)     # Total variance of y
    r2 = 1 - ss_res / ss_total                              # R-squared

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# Fit a simple linear model on training data
Xb_train = add_bias(X_train)                   # Add bias column
Xb_val   = add_bias(X_val)                     # Add bias column to validation too
W = ridge_fit(Xb_train, y_train, lam=0.0)      # No regularization

y_pred_train = Xb_train @ W                    # Predictions on training set
y_pred_val   = Xb_val   @ W                    # Predictions on validation set

train_metrics = compute_regression_metrics(y_train, y_pred_train)
val_metrics   = compute_regression_metrics(y_val,   y_pred_val)

print(f"{'Metric':>6}  {'Training':>12}  {'Validation':>12}")
print("-" * 35)
for key in ['MAE', 'RMSE', 'R2']:
    print(f"{key:>6}  {train_metrics[key]:>12,.0f}  {val_metrics[key]:>12,.0f}"
          if key != 'R2' else
          f"{key:>6}  {train_metrics[key]:>12.4f}  {val_metrics[key]:>12.4f}")
print()

# ----------------------------------------------------------------
# PART 3: k-Fold Cross-Validation
# ----------------------------------------------------------------

print("--- Part 3: k-Fold Cross-Validation ---")
print()

def k_fold_cross_validate(X, y, k=5, lam=0.0):
    """
    Perform k-fold cross-validation.

    Split data into k folds.
    For each fold i: train on all folds EXCEPT i, validate on fold i.
    Return the average RMSE across all k validation rounds.

    C# analogy:
        double total = 0;
        for (int fold = 0; fold < k; fold++) {
            var trainData = allData.Where((_, i) => i % k != fold);
            var valData   = allData.Where((_, i) => i % k == fold);
            var model     = Train(trainData);
            total        += Evaluate(model, valData);
        }
        return total / k;
    """
    n = len(y)
    indices = np.random.permutation(n)         # Shuffle data indices
    fold_size = n // k                         # Samples per fold

    rmse_scores = []                           # Will collect one RMSE per fold

    for fold in range(k):                      # For each of the k folds
        # Determine which indices go into validation for this fold
        val_start = fold * fold_size           # Start of validation fold
        val_end   = val_start + fold_size      # End of validation fold

        val_indices   = indices[val_start:val_end]          # This fold
        train_indices = np.concatenate([                    # All other folds
            indices[:val_start],
            indices[val_end:]
        ])

        X_fold_train = X[train_indices]         # Training portion for this round
        y_fold_train = y[train_indices]
        X_fold_val   = X[val_indices]           # Validation portion for this round
        y_fold_val   = y[val_indices]

        # Add bias and fit model
        Xb_fold_train = add_bias(X_fold_train)
        Xb_fold_val   = add_bias(X_fold_val)
        W_fold = ridge_fit(Xb_fold_train, y_fold_train, lam=lam)

        # Compute RMSE on this fold's validation data
        y_fold_pred = Xb_fold_val @ W_fold
        rmse_fold   = np.sqrt(np.mean((y_fold_pred - y_fold_val) ** 2))
        rmse_scores.append(rmse_fold)

    return np.mean(rmse_scores), np.std(rmse_scores)  # Mean and std of RMSE

# Run 5-fold CV on the training data (do not touch test set!)
mean_rmse, std_rmse = k_fold_cross_validate(X_train, y_train, k=5, lam=0.0)
print(f"5-Fold CV RMSE: {mean_rmse:,.0f} +/- {std_rmse:,.0f} dollars")
print()

# ----------------------------------------------------------------
# PART 4: Hyperparameter Tuning with Grid Search
# ----------------------------------------------------------------

print("--- Part 4: Grid Search for Best Lambda ---")
print()

lambda_grid = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # Values to try

print(f"{'Lambda':>10}  {'CV RMSE':>12}  {'CV Std':>10}")
print("-" * 38)

best_lam     = None
best_cv_rmse = float('inf')                    # Start with infinity (any real value beats it)

for lam in lambda_grid:
    cv_rmse, cv_std = k_fold_cross_validate(X_train, y_train, k=5, lam=lam)
    marker = " <- BEST" if cv_rmse < best_cv_rmse else ""

    if cv_rmse < best_cv_rmse:                 # New best found
        best_cv_rmse = cv_rmse
        best_lam     = lam

    print(f"{lam:>10.3f}  {cv_rmse:>12,.0f}  {cv_std:>10,.0f}{marker}")

print()
print(f"Best lambda from grid search: {best_lam}")
print(f"Best CV RMSE: {best_cv_rmse:,.0f}")
print()

# ----------------------------------------------------------------
# PART 5: Final evaluation on the test set (ONCE only)
# ----------------------------------------------------------------

print("--- Part 5: Final Test Set Evaluation ---")
print()
print("NOW and ONLY NOW do we look at the test set.")
print("We use the best lambda found in Part 4.")
print()

# Train on FULL training set (not cross-validation splits) with best lambda
Xb_full_train = add_bias(X_train)
Xb_test       = add_bias(X_test)
W_final = ridge_fit(Xb_full_train, y_train, lam=best_lam)

y_pred_final = Xb_test @ W_final              # Predictions on locked test set
test_metrics = compute_regression_metrics(y_test, y_pred_final)

print(f"Test MAE:  {test_metrics['MAE']:>12,.0f} dollars")
print(f"Test RMSE: {test_metrics['RMSE']:>12,.0f} dollars")
print(f"Test R2:   {test_metrics['R2']:>12.4f}")
print()

# ----------------------------------------------------------------
# PART 6: Classification Metrics (Precision, Recall, F1)
# ----------------------------------------------------------------

print("--- Part 6: Classification Metrics ---")
print()

# Create a synthetic binary classification result
y_true_cls = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0,   # True labels
                        1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
y_pred_cls = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0,   # Model predictions
                        1, 0, 0, 0, 1, 1, 1, 1, 0, 0])

# Compute confusion matrix components
TP = np.sum((y_pred_cls == 1) & (y_true_cls == 1))   # Predicted positive, truly positive
FP = np.sum((y_pred_cls == 1) & (y_true_cls == 0))   # Predicted positive, truly negative
TN = np.sum((y_pred_cls == 0) & (y_true_cls == 0))   # Predicted negative, truly negative
FN = np.sum((y_pred_cls == 0) & (y_true_cls == 1))   # Predicted negative, truly positive

accuracy  = (TP + TN) / len(y_true_cls)              # Overall fraction correct
precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Of predicted positives, how many real?
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0  # Of real positives, how many found?
f1        = (2 * precision * recall / (precision + recall)
             if (precision + recall) > 0 else 0)     # Harmonic mean of precision and recall

print("Confusion Matrix:")
print(f"  True Positives  (TP): {TP}  <- predicted spam, IS spam")
print(f"  False Positives (FP): {FP}  <- predicted spam, NOT spam (false alarm)")
print(f"  True Negatives  (TN): {TN}  <- predicted legit, IS legit")
print(f"  False Negatives (FN): {FN}  <- predicted legit, IS spam (missed!)")
print()
print(f"Accuracy:  {accuracy:.4f}  <- can be misleading with imbalanced classes!")
print(f"Precision: {precision:.4f}  <- low precision = too many false alarms")
print(f"Recall:    {recall:.4f}  <- low recall = missing real positives")
print(f"F1 Score:  {f1:.4f}  <- balanced metric, use when both matter")
print()

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()
print("Model evaluation workflow:")
print("  1. Split data: train (60%), val (20%), test (20%)")
print("  2. Use k-fold CV on training data to compare models/hyperparams")
print("  3. Grid search over hyperparameter values using CV score")
print("  4. Train final model on full training set with best hyperparams")
print("  5. Evaluate ONCE on the locked test set -> final honest score")
print()
print("Regression metrics: MAE (robust), RMSE (penalizes big errors), R2 (intuitive)")
print("Classification metrics: Accuracy (easy), Precision/Recall/F1 (honest)")
print()
print("The test set rule: look at it ONCE. If you peek during development,")
print("it becomes validation data and you lose your honest evaluation.")

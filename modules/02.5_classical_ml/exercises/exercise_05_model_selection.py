"""
Module 02.5 - Classical Machine Learning
Exercise 05: Model Selection and Evaluation from Scratch

GLOSSARY
--------
Train/Val/Test: Three non-overlapping data subsets.
Split           Train: fit weights. Val: tune hyperparams. Test: final honest eval.
                C# analogy: unit tests / integration tests / acceptance tests.
                CRITICAL: never use the test set during development.

k-Fold CV     : Split into k folds; rotate which fold is validation.
                Average metrics over k rounds = reliable performance estimate.
                C# analogy: a rotating review board over k iterations.

Shuffle       : Randomly reorder data before splitting so splits are not biased
                by the original ordering. C# analogy: list.OrderBy(_ => rand.Next()).

Accuracy      : fraction of correct predictions. correct / total.

Precision     : Of all PREDICTED positives, what fraction are truly positive?
                TP / (TP + FP). Low = too many false alarms.

Recall        : Of all TRUE positives, what fraction did we predict as positive?
                TP / (TP + FN). Low = missing real positives.

F1 Score      : Harmonic mean of precision and recall.
                2 * P * R / (P + R). Balanced metric.

TP/FP/TN/FN   : True/False Positive/Negative. The four cells of a confusion matrix.
                C# analogy: pass/fail categories in test results.

HOW TO USE THIS FILE
--------------------
1. Read each EXERCISE section.
2. Replace `pass` with your implementation.
3. Run and check output against EXPECTED comments.
"""

import numpy as np   # NumPy: array operations

print("=" * 60)
print("Exercise 05: Model Selection and Evaluation")
print("=" * 60)
print()

# ----------------------------------------------------------------
# Ridge helper (provided -- do not modify)
# ----------------------------------------------------------------
def add_bias(X):
    """Prepend a column of 1s to X."""
    return np.hstack([np.ones((X.shape[0], 1)), X])

def ridge_fit(X, y, lam=0.0):
    """Ridge regression: W = inv(X.T@X + lam*I) @ X.T @ y"""
    n_feat = X.shape[1]
    return np.linalg.inv(X.T @ X + lam * np.eye(n_feat)) @ X.T @ y

# ----------------------------------------------------------------
# Dataset (do not modify)
# ----------------------------------------------------------------
np.random.seed(123)
n = 80
x = np.random.uniform(0, 10, n)                # Feature: single number
y = 3 * x + 7 + np.random.randn(n) * 4        # True: y = 3x + 7 + noise

# ============================================================
#  EXERCISE 1
#  Topic: Train / Validation / Test Split
#
#  Background:
#    Split the data into three non-overlapping sets.
#    ALWAYS shuffle before splitting to avoid ordering bias.
#
#    Proportions to use: 60% train, 20% validation, 20% test.
#    Given n=80 samples:
#      train: first 48  (60%)
#      val:   next  16  (20%)
#      test:  last  16  (20%)
#
#    Shuffle using np.random.permutation(n) to get random order,
#    then slice the shuffled indices.
#
#  Your Task:
#    Implement: train_val_test_split(x, y, train_frac, val_frac, seed)
#    Returns: x_train, y_train, x_val, y_val, x_test, y_test
#
#  C# Analogy:
#    var shuffled = data.OrderBy(_ => rand.Next()).ToList();
#    var train = shuffled.Take((int)(n * 0.6)).ToList();
#    var val   = shuffled.Skip(train.Count).Take((int)(n * 0.2)).ToList();
#    var test  = shuffled.Skip(train.Count + val.Count).ToList();
#
#  Hint:
#    np.random.seed(seed) sets the global random seed.
#    np.random.permutation(n) returns a shuffled array of [0, 1, ..., n-1].
#    Array slicing: arr[:48] gets first 48 elements.
# ============================================================

def train_val_test_split(x, y, train_frac=0.6, val_frac=0.2, seed=42):
    """
    Split x and y into train, validation, and test sets.

    Parameters:
        x          : np.ndarray of shape (n,)
        y          : np.ndarray of shape (n,)
        train_frac : float, fraction for training
        val_frac   : float, fraction for validation
        seed       : int, random seed for shuffling

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test
        Each as np.ndarray
    """
    pass   # TODO: shuffle, then slice into three parts


# Test Exercise 1
print("--- Exercise 1: Train/Val/Test Split ---")
result = train_val_test_split(x, y, seed=42)
if result is not None:
    x_tr, y_tr, x_val, y_val, x_te, y_te = result
    print(f"Total samples: {len(x)}")
    print(f"Training:   {len(x_tr)} ({len(x_tr)/len(x)*100:.0f}%)")
    print(f"Validation: {len(x_val)} ({len(x_val)/len(x)*100:.0f}%)")
    print(f"Test:       {len(x_te)} ({len(x_te)/len(x)*100:.0f}%)")
    total = len(x_tr) + len(x_val) + len(x_te)
    print(f"Sum:        {total} (should equal {len(x)})")
    # EXPECTED: 48 + 16 + 16 = 80
else:
    x_tr, y_tr, x_val, y_val, x_te, y_te = None, None, None, None, None, None
print()

# ============================================================
#  EXERCISE 2
#  Topic: Regression Metrics (RMSE and R-squared)
#
#  Background:
#    RMSE = sqrt( mean( (y_pred - y_true)^2 ) )
#      - Same units as y (dollars, degrees, etc.)
#      - Penalizes large errors more than MAE
#
#    R2   = 1 - SS_residual / SS_total
#      where SS_residual = sum( (y_pred - y_true)^2 )
#            SS_total    = sum( (y_true - mean(y_true))^2 )
#      - R2 = 1.0: perfect predictions
#      - R2 = 0.0: model no better than predicting the mean
#      - R2 < 0:   model is worse than predicting the mean
#
#  Your Task:
#    Implement: rmse(y_true, y_pred) -> float
#    Implement: r_squared(y_true, y_pred) -> float
#
#  C# Analogy:
#    double RMSE(double[] truth, double[] pred) =>
#        Math.Sqrt(truth.Zip(pred, (a,b) => Math.Pow(b-a,2)).Average());
# ============================================================

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.

    Parameters:
        y_true : np.ndarray of shape (n,)
        y_pred : np.ndarray of shape (n,)

    Returns:
        float: RMSE value (>= 0)
    """
    pass   # TODO: sqrt(mean((y_pred - y_true)^2))


def r_squared(y_true, y_pred):
    """
    R-squared (coefficient of determination).

    Parameters:
        y_true : np.ndarray of shape (n,)
        y_pred : np.ndarray of shape (n,)

    Returns:
        float: R2 value (1.0 = perfect)
    """
    pass   # TODO: 1 - SS_residual / SS_total


# Test Exercise 2
print("--- Exercise 2: Regression Metrics ---")
# Quick sanity checks before using with real data
y_perfect = np.array([1.0, 2.0, 3.0])
y_pred_perfect = np.array([1.0, 2.0, 3.0])
y_pred_bad = np.array([3.0, 2.0, 1.0])

if rmse(y_perfect, y_pred_perfect) is not None:
    print(f"RMSE (perfect predictions): {rmse(y_perfect, y_pred_perfect):.4f}  (expected: 0.0)")
    print(f"R2   (perfect predictions): {r_squared(y_perfect, y_pred_perfect):.4f}  (expected: 1.0)")
    print(f"RMSE (bad predictions):     {rmse(y_perfect, y_pred_bad):.4f}  (expected: >0)")
    print(f"R2   (bad predictions):     {r_squared(y_perfect, y_pred_bad):.4f}  (expected: <0)")
print()

# ============================================================
#  EXERCISE 3
#  Topic: k-Fold Cross-Validation
#
#  Background:
#    k-fold CV gives a reliable estimate of model performance
#    by rotating which portion of training data is used for validation.
#
#    Algorithm for k=5:
#      Divide training data into 5 equal folds.
#      For each fold i (0, 1, 2, 3, 4):
#        - Validation = fold i
#        - Training   = all other folds
#        - Fit model, compute RMSE on validation fold
#      Return mean and std of the 5 RMSE values.
#
#  Your Task:
#    Implement: kfold_cv(x, y, k, lam) -> (mean_rmse, std_rmse)
#
#  C# Analogy:
#    double total = 0;
#    int foldSize = n / k;
#    for (int fold = 0; fold < k; fold++) {
#        var valRange = (fold*foldSize, (fold+1)*foldSize);
#        // train on everything outside valRange, validate inside
#        total += RMSE(model, valData);
#    }
#    return total / k;
#
#  Hint:
#    np.concatenate([arr1, arr2]) joins arrays end to end.
#    x[a:b] slices from index a to b (exclusive).
# ============================================================

def kfold_cv(x, y, k=5, lam=0.0):
    """
    k-Fold Cross-Validation for Ridge regression.

    Parameters:
        x   : np.ndarray of shape (n,) -- 1D feature array
        y   : np.ndarray of shape (n,)
        k   : int, number of folds
        lam : float, Ridge regularization strength

    Returns:
        (mean_rmse, std_rmse) as floats
    """
    pass   # TODO: split into k folds, rotate validation, average RMSE


# Test Exercise 3
print("--- Exercise 3: k-Fold Cross-Validation ---")
if x_tr is not None and rmse(y_perfect, y_pred_perfect) is not None:
    cv_result = kfold_cv(x_tr, y_tr, k=5, lam=0.0)
    if cv_result is not None:
        mean_r, std_r = cv_result
        print(f"5-Fold CV RMSE: {mean_r:.4f} +/- {std_r:.4f}")
        # EXPECTED: RMSE around 3-5 (given noise std=4 in data generation)
    else:
        print("kfold_cv returned None -- check implementation")
print()

# ============================================================
#  EXERCISE 4
#  Topic: Grid Search for Best Lambda
#
#  Background:
#    Grid search tries a list of hyperparameter values.
#    For each value, run k-fold CV on the training data.
#    The value with the lowest mean CV RMSE is the best.
#
#    Never use the test set for grid search!
#    Only use the training data with cross-validation.
#
#  Your Task:
#    Implement: grid_search_lambda(x_train, y_train, lambda_values, k)
#    Returns: (best_lambda, best_cv_rmse, results_dict)
#
#    results_dict: {lambda_value: (mean_rmse, std_rmse)}
#
#  C# Analogy:
#    double bestLam = 0; double bestScore = double.MaxValue;
#    foreach (double lam in lambdas) {
#        double score = CrossValidate(x, y, k, lam);
#        if (score < bestScore) { bestScore = score; bestLam = lam; }
#    }
# ============================================================

def grid_search_lambda(x_train, y_train, lambda_values, k=5):
    """
    Grid search over lambda values using k-fold cross-validation.

    Parameters:
        x_train       : np.ndarray of shape (n,)
        y_train       : np.ndarray of shape (n,)
        lambda_values : list of float, lambda candidates to try
        k             : int, number of CV folds

    Returns:
        best_lambda   : float, the lambda with lowest CV RMSE
        best_cv_rmse  : float, the lowest mean CV RMSE
        results       : dict mapping lambda -> (mean_rmse, std_rmse)
    """
    pass   # TODO: loop over lambda_values, run kfold_cv, track best


# Test Exercise 4
print("--- Exercise 4: Grid Search for Lambda ---")
lambdas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
if (x_tr is not None
        and rmse(y_perfect, y_pred_perfect) is not None
        and kfold_cv(x_tr, y_tr, k=5, lam=0.0) is not None):
    gs_result = grid_search_lambda(x_tr, y_tr, lambdas, k=5)
    if gs_result is not None:
        best_lam, best_cv, results = gs_result
        print(f"{'Lambda':>10}  {'CV RMSE':>12}  {'CV Std':>10}  Note")
        print("-" * 55)
        for lam in lambdas:
            if lam in results:
                m_r, s_r = results[lam]
                note = " <- BEST" if lam == best_lam else ""
                print(f"{lam:>10.3f}  {m_r:>12.4f}  {s_r:>10.4f}{note}")
        print(f"\nBest lambda: {best_lam}  Best CV RMSE: {best_cv:.4f}")
    else:
        print("grid_search_lambda returned None -- check implementation")
        best_lam = 0.0
else:
    print("Complete exercises 1-3 first")
    best_lam = 0.0
print()

# ============================================================
#  EXERCISE 5
#  Topic: Precision, Recall, and F1 for Binary Classification
#
#  Background:
#    Given true labels and predicted labels for a binary problem:
#
#    TP (True Positive):  predicted 1, true is 1  (correct positive)
#    FP (False Positive): predicted 1, true is 0  (false alarm)
#    TN (True Negative):  predicted 0, true is 0  (correct negative)
#    FN (False Negative): predicted 0, true is 1  (missed positive)
#
#    Precision = TP / (TP + FP)  -- of all predicted positives, how many right?
#    Recall    = TP / (TP + FN)  -- of all true positives, how many found?
#    F1        = 2*P*R / (P+R)   -- harmonic mean, balances both
#
#  Your Task:
#    Implement: classification_metrics(y_true, y_pred) -> dict
#    Returns dict with keys: 'accuracy', 'precision', 'recall', 'f1'
#
#  C# Analogy:
#    int TP = y.Zip(pred, (a,b) => a==1 && b==1 ? 1 : 0).Sum();
#    // similarly FP, TN, FN
#    double precision = TP / (double)(TP + FP);
# ============================================================

def classification_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, and F1 score.

    Parameters:
        y_true : np.ndarray of int (0 or 1), true labels
        y_pred : np.ndarray of int (0 or 1), predicted labels

    Returns:
        dict with keys: 'accuracy', 'precision', 'recall', 'f1'
        All values are floats between 0.0 and 1.0.
    """
    pass   # TODO: compute TP, FP, TN, FN then derive metrics


# Test Exercise 5
print("--- Exercise 5: Classification Metrics ---")
y_true_cls = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
                        0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
y_pred_cls = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1,
                        0, 0, 0, 1, 1, 0, 1, 1, 0, 0])

metrics = classification_metrics(y_true_cls, y_pred_cls)
if metrics is not None:
    print(f"Accuracy:  {metrics['accuracy']:.4f}  (correct / total)")
    print(f"Precision: {metrics['precision']:.4f}  (of predicted positives, how many right?)")
    print(f"Recall:    {metrics['recall']:.4f}  (of true positives, how many found?)")
    print(f"F1 Score:  {metrics['f1']:.4f}  (harmonic mean of precision and recall)")
    # EXPECTED: Check that precision + recall are both reasonable (>0.6)
    # High F1 (>0.7) means the classifier is doing well overall
print()

print("=" * 60)
print("All exercises complete!")
print()
print("Model selection workflow:")
print("  1. Split: train (60%) / val (20%) / test (20%)")
print("  2. k-Fold CV on training data to estimate performance")
print("  3. Grid search over hyperparams using CV score")
print("  4. Train final model on ALL training data with best hyperparams")
print("  5. Evaluate ONCE on the locked test set")
print()
print("Regression metrics: RMSE (penalizes big errors), R2 (intuitive fraction)")
print("Classification:     Accuracy can be misleading! Use Precision/Recall/F1.")
print("=" * 60)

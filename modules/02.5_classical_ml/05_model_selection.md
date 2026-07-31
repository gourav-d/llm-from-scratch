# Lesson 05 - Model Selection and Evaluation

## The Problem: How Do You Know If Your Model Is Good?

You trained a regression model. The predictions look great.
But wait -- you tested it on the SAME data you trained on.
Of course it looks great. The model has already "seen the answers."

This is the cardinal sin of machine learning evaluation.

You need an honest estimate of how well the model performs on **new, unseen data**.

---

## C# Analogy: Unit Tests vs. Integration Tests vs. QA

Think of model evaluation like software testing:

```
TRAINING DATA   = unit tests you write yourself.
                  You know what these test. The code passes (by design).
                  But this tells you nothing about real-world behavior.

VALIDATION DATA = integration tests run by a QA engineer on the side.
                  The engineer was NOT involved in writing the feature.
                  A more honest signal. Used to tune parameters.

TEST DATA       = UAT / production verification.
                  Stakeholders test the feature against real scenarios.
                  Only run ONCE at the very end. Never used during development.
```

If you keep looking at test results and tweaking your model, the test set
becomes "used" data and is no longer an honest estimate of future performance.

---

## The Train / Validation / Test Split

The standard approach is to divide your data into three non-overlapping sets:

```
All Data (100%)
|
+-- Training Set (60-70%):
|   Used to fit model weights.
|
+-- Validation Set (15-20%):
|   Used to tune hyperparameters (lambda, max_depth, etc.)
|   You WILL look at validation metrics during development.
|
+-- Test Set (15-20%):
|   Used ONCE at the very end to report final model performance.
|   NEVER used to make decisions about the model.
```

**Critical rule**: the test set must never influence model design.
If you tune hyperparameters based on the test set, you are cheating --
your test set is now effectively validation data.

---

## k-Fold Cross-Validation

When data is limited, using 20% for validation wastes precious training samples.
**k-Fold Cross-Validation** makes better use of data:

```
Split data into k equal groups (folds). Example: k = 5.

Fold 1:  [VALID][TRAIN][TRAIN][TRAIN][TRAIN]  -> measure error on fold 1
Fold 2:  [TRAIN][VALID][TRAIN][TRAIN][TRAIN]  -> measure error on fold 2
Fold 3:  [TRAIN][TRAIN][VALID][TRAIN][TRAIN]  -> measure error on fold 3
Fold 4:  [TRAIN][TRAIN][TRAIN][VALID][TRAIN]  -> measure error on fold 4
Fold 5:  [TRAIN][TRAIN][TRAIN][TRAIN][VALID]  -> measure error on fold 5

Final score = average error across all 5 folds
```

Every sample gets to be in the validation set exactly once.
Every sample participates in training for 4 out of 5 iterations.

This gives a much more reliable estimate of true model performance,
especially when you have fewer than a few thousand samples.

---

## Diagram: 5-Fold Cross-Validation

```
Data: [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]

Round 1: Train=[s3..s10]   Validate=[s1,s2]   -> error1
Round 2: Train=[s1,s2,s5..s10] Validate=[s3,s4] -> error2
Round 3: Train=[s1..s4,s7..s10] Validate=[s5,s6] -> error3
Round 4: Train=[s1..s6,s9,s10] Validate=[s7,s8] -> error4
Round 5: Train=[s1..s8]   Validate=[s9,s10]  -> error5

Final CV score = mean(error1, error2, error3, error4, error5)
```

---

## Hyperparameter Tuning: Grid Search

A **hyperparameter** is a setting you choose BEFORE training begins.
Unlike weights (which are learned), hyperparameters are fixed by you.

Examples:
- Ridge lambda: 0.001, 0.01, 0.1, 1.0, 10.0
- Decision tree max_depth: 3, 5, 10, None
- Random forest n_estimators: 50, 100, 500

**Grid Search** tries every combination:

```python
lambda_values = [0.001, 0.01, 0.1, 1.0, 10.0]
depth_values  = [3, 5, 10]

best_score = infinity
for lam in lambda_values:
    for depth in depth_values:
        score = cross_validate(model(lam, depth), X_train, y_train, k=5)
        if score < best_score:
            best_score = score
            best_params = (lam, depth)
```

Total combinations: 5 * 3 = 15 models trained and evaluated.
Each with 5-fold cross-validation = 75 training runs.

For large models this is expensive. In practice, **Random Search** (try random
combinations) or **Bayesian optimization** is used instead.

---

## Evaluation Metrics for Regression

| Metric | Formula                                       | Interpretation                    |
|--------|-----------------------------------------------|-----------------------------------|
| MAE    | mean(abs(y_pred - y_true))                   | Average absolute error            |
| MSE    | mean((y_pred - y_true)^2)                    | Penalizes large errors more       |
| RMSE   | sqrt(MSE)                                     | Same units as y (easier to read)  |
| R2     | 1 - SS_residual / SS_total                   | 1.0 = perfect, 0.0 = useless      |

**Which to use?**
- RMSE if large errors are especially bad (financial forecasts)
- MAE if all errors are equally bad (delivery time estimates)
- R2 for an intuitive "percentage explained" metric

---

## Evaluation Metrics for Classification

| Metric    | Formula                              | Interpretation                          |
|-----------|--------------------------------------|-----------------------------------------|
| Accuracy  | correct / total                      | Overall fraction correct                |
| Precision | TP / (TP + FP)                       | Of all predicted positives, how many real? |
| Recall    | TP / (TP + FN)                       | Of all real positives, how many found?  |
| F1-Score  | 2 * precision * recall / (prec+rec)  | Harmonic mean of precision and recall   |

**TP = True Positive, FP = False Positive, FN = False Negative**

**Example (spam detection):**
- Precision: "Of all emails flagged as spam, what fraction was really spam?"
  (Low precision = many legitimate emails wrongly blocked)
- Recall: "Of all actual spam emails, what fraction did we catch?"
  (Low recall = many spam emails slipping through)

High precision and high recall often trade off against each other.
F1-Score balances both.

---

## Bias-Variance Tradeoff

Every model's error has two components:

```
Total Error = Bias^2 + Variance + Irreducible Noise

Bias    = systematic error from wrong assumptions (underfitting)
          "The model is fundamentally too simple for the problem"

Variance = sensitivity to small changes in training data (overfitting)
           "The model memorizes noise; different training sets give
            very different models"

Irreducible Noise = randomness in the data you cannot model away
```

Diagram:
```
             High Bias                    Low Bias
             (Underfitting)               (Good range)
High      +-------------------+    +--------------------+
Variance  |  Wrong AND        |    |  Right on average  |
(Overfit) |  inconsistent     |    |  but inconsistent  |
          +-------------------+    +--------------------+

Low       +-------------------+    +--------------------+
Variance  |  Wrong but        |    |  Right AND         |
(Stable)  |  consistently     |    |  consistent        |
          |  wrong            |    |  GOAL              |
          +-------------------+    +--------------------+
```

Simple models (linear regression): high bias, low variance
Complex models (deep trees): low bias, high variance
Random Forest / Regularization: reduces variance while keeping bias low

---

## Putting It All Together: A Model Building Workflow

```
Step 1: Collect and clean data
Step 2: Split into train / validation / test  (keep test set locked away!)
Step 3: Choose a model class (linear regression, decision tree, etc.)
Step 4: Train on training set
Step 5: Evaluate on validation set (or use k-fold CV)
Step 6: Tune hyperparameters (grid search over validation performance)
Step 7: Repeat steps 4-6 until satisfied
Step 8: Run ONCE on test set to get final honest performance number
Step 9: Report test set result as your model's real-world performance
```

Do NOT go back to step 3 after seeing the test set result. If you do,
you must collect new test data.

---

## Connection to LLM Training

The same workflow applies at massive scale in LLM training:

- **Training set**: billions of tokens from web, books, code
- **Validation set**: held-out text used to track perplexity (a loss metric)
  during training and decide when to stop or adjust learning rate
- **Test set / benchmarks**: standard datasets (MMLU, HumanEval, etc.)
  used to compare different models honestly
- **Overfitting in LLMs**: a model that memorizes training data will fail
  on novel questions -- same problem, much larger scale

---

## Quiz

**Question 1:**
You train a regression model and achieve R2 = 0.95 on training data but
R2 = 0.40 on validation data. What is the most likely problem?

a) Underfitting: the model is too simple
b) Overfitting: the model memorized training data but cannot generalize
c) The validation set is too large
d) R2 cannot be used for regression

**Answer: b) Overfitting: the model memorized training data but cannot generalize**
Explanation: A large gap between training and validation performance is the
hallmark of overfitting. The model performs well on data it has seen (training)
but fails on new data (validation). Solutions: add regularization, reduce
model complexity, collect more training data.

---

**Question 2:**
You perform 5-fold cross-validation on 500 samples. In each fold, how many
samples are used for training and how many for validation?

a) 400 training, 100 validation
b) 250 training, 250 validation
c) 100 training, 400 validation
d) 500 training, 500 validation (both use all data)

**Answer: a) 400 training, 100 validation**
Explanation: With k=5, data is split into 5 equal folds of 100 samples each.
In each round, 1 fold (100 samples) is used for validation and the remaining
4 folds (400 samples) are used for training. After 5 rounds, every sample has
been in the validation set exactly once.

---

**Question 3:**
A spam filter has 95% accuracy on your test set. Your colleague says this
sounds great. But you notice 99% of your email is legitimate. What is
actually happening?

a) The model is excellent; 95% accuracy is always good
b) The model might just be predicting "not spam" for everything, which
   would give 99% accuracy. A 95% model could actually be worse than
   always saying "not spam."
c) Accuracy is the wrong metric -- we should use MSE instead
d) The test set is too small to be reliable

**Answer: b) The model might just be predicting "not spam" for everything,
which would give 99% accuracy. A 95% model could actually be worse than
always saying "not spam."**
Explanation: When data is imbalanced (99% one class), accuracy is misleading.
A model that never identifies spam would score 99% accuracy but is useless.
Use precision and recall instead: precision measures false positive rate,
recall measures how many actual spam emails you catch. For spam filtering,
you especially care about recall (missing real spam) and precision (wrongly
blocking legitimate email).

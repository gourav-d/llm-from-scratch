"""
Exercise 05: Evaluate a Fine-Tuned Model
==========================================

GOAL
----
Practice computing and interpreting evaluation metrics
for a fine-tuned classification model.

EXERCISES
---------
Exercise 1: Implement compute_accuracy()
Exercise 2: Implement compute_precision_recall_f1() for one class
Exercise 3: Build a confusion matrix and print it
Exercise 4: Write interpret_results() that gives a plain-English verdict

HOW TO RUN
----------
  python exercise_05_evaluation.py

LIBRARIES NEEDED
-----------------
  None (pure Python)
"""

print("=" * 65)
print("EXERCISE 05: Evaluate a Fine-Tuned Model")
print("=" * 65)

# ---------------------------------------------------------------------------
# PROVIDED: Simulated model predictions to evaluate
# ---------------------------------------------------------------------------

# These represent outputs from a fine-tuned support ticket classifier
# True label = what it should have predicted
# Pred label = what the model actually predicted

EVAL_DATA = [
    {"text": "App crashes on login",             "true": "BUG",     "pred": "BUG"},
    {"text": "Add dark mode option",             "true": "FEATURE", "pred": "FEATURE"},
    {"text": "API is completely down",           "true": "OUTAGE",  "pred": "OUTAGE"},
    {"text": "How do I export to CSV?",          "true": "HOW_TO",  "pred": "HOW_TO"},
    {"text": "Login throws 500 error",           "true": "BUG",     "pred": "BUG"},
    {"text": "Want a mobile app version",        "true": "FEATURE", "pred": "FEATURE"},
    {"text": "Cannot connect since 2pm",         "true": "OUTAGE",  "pred": "BUG"},     # WRONG
    {"text": "Steps to reset password",          "true": "HOW_TO",  "pred": "HOW_TO"},
    {"text": "Button clicks do nothing",         "true": "BUG",     "pred": "BUG"},
    {"text": "Add calendar integration",         "true": "FEATURE", "pred": "HOW_TO"},  # WRONG
    {"text": "Database unreachable",             "true": "OUTAGE",  "pred": "OUTAGE"},
    {"text": "How to change billing plan?",      "true": "HOW_TO",  "pred": "HOW_TO"},
    {"text": "Memory leak in background worker", "true": "BUG",     "pred": "BUG"},
    {"text": "Allow custom report filters",      "true": "FEATURE", "pred": "FEATURE"},
    {"text": "Service degraded for all users",   "true": "OUTAGE",  "pred": "OUTAGE"},
    {"text": "Null pointer on dashboard load",   "true": "BUG",     "pred": "FEATURE"}, # WRONG
    {"text": "How to invite team members?",      "true": "HOW_TO",  "pred": "HOW_TO"},
    {"text": "Add SSO / SAML support",           "true": "FEATURE", "pred": "FEATURE"},
    {"text": "App unresponsive on mobile",       "true": "BUG",     "pred": "BUG"},
    {"text": "All webhooks failing",             "true": "OUTAGE",  "pred": "OUTAGE"},
]

# Extract just the labels for convenience
TRUE_LABELS = [d["true"] for d in EVAL_DATA]   # What the model SHOULD say
PRED_LABELS = [d["pred"] for d in EVAL_DATA]   # What the model DID say
CLASSES     = ["BUG", "FEATURE", "HOW_TO", "OUTAGE"]  # All possible labels


# ===========================================================================
# EXERCISE 1: Compute Accuracy
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 1: Compute Accuracy")
print("=" * 65)

print("""
TODO: Write compute_accuracy(predictions, labels) that returns a float.

Formula:
  accuracy = number of correct predictions / total predictions

A prediction is "correct" when predictions[i] == labels[i].

HINT:
  correct = sum(1 for p, l in zip(predictions, labels) if p == l)
  return correct / len(labels)
""")


def compute_accuracy(predictions: list, labels: list) -> float:
    """
    TODO: Return fraction of correct predictions.
    predictions: list of predicted labels (strings)
    labels:      list of true labels (strings)
    Returns:     float between 0.0 and 1.0
    """
    pass  # DELETE THIS and write your implementation


acc = compute_accuracy(PRED_LABELS, TRUE_LABELS)
if acc is not None:
    print(f"Accuracy: {acc:.2%}")
    print(f"Expected: 85.00%  (17 out of 20 correct)")
    print("PASS" if abs(acc - 0.85) < 0.01 else "FAIL -- check your formula")
else:
    print("TODO: Implement compute_accuracy()")


# ===========================================================================
# EXERCISE 2: Precision, Recall, and F1 for One Class
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 2: Precision, Recall, F1 for 'BUG' class")
print("=" * 65)

print("""
TODO: Write compute_prf1(predictions, labels, target_class) that returns
a dict: {"precision": float, "recall": float, "f1": float}.

DEFINITIONS:
  True Positive  (TP): predicted target AND true label IS target
  False Positive (FP): predicted target BUT true label is NOT target
  False Negative (FN): did NOT predict target BUT true label IS target

FORMULAS:
  precision = TP / (TP + FP)   (of what we predicted BUG, how many were right?)
  recall    = TP / (TP + FN)   (of all actual BUGs, how many did we find?)
  f1        = 2 * precision * recall / (precision + recall)

HINT for counting TP:
  sum(1 for p, l in zip(predictions, labels) if p == target and l == target)
""")


def compute_prf1(predictions: list, labels: list, target_class: str) -> dict:
    """
    TODO: Compute precision, recall, F1 for a single class.
    Returns: {"precision": float, "recall": float, "f1": float}
    """
    pass  # DELETE THIS and implement


result = compute_prf1(PRED_LABELS, TRUE_LABELS, "BUG")
if result is not None:
    print(f"BUG class metrics: {result}")
    print("Expected: precision=0.857, recall=0.857, f1=0.857  (approximately)")
else:
    print("TODO: Implement compute_prf1()")


# ===========================================================================
# EXERCISE 3: Confusion Matrix
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 3: Build and Print Confusion Matrix")
print("=" * 65)

print("""
TODO: Write two functions:

  build_confusion_matrix(predictions, labels, classes) -> dict
    Returns nested dict: matrix[true_label][predicted_label] = count
    Initialize all cells to 0, then loop through predictions and labels.

  print_confusion_matrix(matrix, classes)
    Prints the matrix as an ASCII table.
    Row = true label, column = predicted label.
    Diagonal = correct predictions.

HINT for build:
  matrix = {c: {c2: 0 for c2 in classes} for c in classes}
  for pred, true in zip(predictions, labels):
      matrix[true][pred] += 1

HINT for print:
  col_width = 10
  Print a header row with all class names.
  Then print each row: class name + count for each column.
""")


def build_confusion_matrix(predictions: list, labels: list, classes: list) -> dict:
    """
    TODO: Build confusion matrix as nested dict.
    matrix[true_label][predicted_label] = count
    """
    pass  # DELETE THIS and implement


def print_confusion_matrix(matrix: dict, classes: list):
    """
    TODO: Print confusion matrix as ASCII table.
    Row = true label, Column = predicted label.
    """
    pass  # DELETE THIS and implement


matrix = build_confusion_matrix(PRED_LABELS, TRUE_LABELS, CLASSES)
if matrix is not None:
    print("Confusion Matrix:")
    print_confusion_matrix(matrix, CLASSES)
    print("""
Expected pattern:
  - Most numbers on the diagonal (correct predictions)
  - Off-diagonal numbers show where model gets confused
  - OUTAGE confused with BUG once (that was the wrong prediction)
""")
else:
    print("TODO: Implement build_confusion_matrix()")


# ===========================================================================
# EXERCISE 4: Interpret Results
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 4: Interpret Results")
print("=" * 65)

print("""
TODO: Write interpret_results(accuracy, class_metrics) that returns
a plain-English string summarizing the model's performance.

accuracy:      float (e.g., 0.85)
class_metrics: dict of {class_name: {"precision": ..., "recall": ..., "f1": ...}}

Rules for the verdict string:
  - If accuracy >= 0.90: "Excellent: ready for production"
  - If accuracy >= 0.75: "Good: acceptable for most use cases"
  - If accuracy >= 0.60: "Fair: needs improvement before production"
  - If accuracy < 0.60:  "Poor: significant issues -- collect more data"

Also find the WORST performing class (lowest F1) and mention it.

HINT:
  worst_class = min(class_metrics, key=lambda c: class_metrics[c]["f1"])
""")


def interpret_results(accuracy: float, class_metrics: dict) -> str:
    """
    TODO: Return a plain-English interpretation of the model's performance.
    """
    pass  # DELETE THIS and implement


# Build class_metrics dict from EVAL_DATA
if compute_prf1(PRED_LABELS, TRUE_LABELS, "BUG") is not None:
    class_metrics = {c: compute_prf1(PRED_LABELS, TRUE_LABELS, c) for c in CLASSES}
    verdict = interpret_results(acc or 0.85, class_metrics)
    if verdict:
        print(f"Interpretation: {verdict}")
    else:
        print("TODO: Implement interpret_results()")
else:
    print("TODO: Complete Exercises 1-3 first, then come back to Exercise 4.")


# ===========================================================================
# BONUS: Per-Class Summary Table
# ===========================================================================

print("\n" + "=" * 65)
print("BONUS: Per-Class Summary Table")
print("=" * 65)

print("""
BONUS: Write print_class_summary(predictions, labels, classes) that prints:

  Class     | Precision | Recall | F1
  ----------|-----------|--------|------
  BUG       |    0.857  |  0.857 | 0.857
  FEATURE   |    0.800  |  1.000 | 0.889
  HOW_TO    |    1.000  |  1.000 | 1.000
  OUTAGE    |    1.000  |  0.750 | 0.857
  ----------|-----------|--------|------
  AVERAGE   |    0.914  |  0.902 | 0.901

Also print: total accuracy at the bottom.
""")


def print_class_summary(predictions: list, labels: list, classes: list):
    """
    TODO: Print a per-class metrics table.
    """
    pass  # DELETE THIS and implement


print_class_summary(PRED_LABELS, TRUE_LABELS, CLASSES)


# ===========================================================================
# SOLUTION
# ===========================================================================

print("\n" + "=" * 65)
print("SOLUTION (uncomment to check)")
print("=" * 65)

"""
SOLUTION FOR EXERCISE 1:

    def compute_accuracy(predictions: list, labels: list) -> float:
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return correct / len(labels)

SOLUTION FOR EXERCISE 2:

    def compute_prf1(predictions: list, labels: list, target_class: str) -> dict:
        tp = sum(1 for p, l in zip(predictions, labels)
                 if p == target_class and l == target_class)
        fp = sum(1 for p, l in zip(predictions, labels)
                 if p == target_class and l != target_class)
        fn = sum(1 for p, l in zip(predictions, labels)
                 if p != target_class and l == target_class)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}

SOLUTION FOR EXERCISE 3:

    def build_confusion_matrix(predictions, labels, classes):
        matrix = {c: {c2: 0 for c2 in classes} for c in classes}
        for pred, true in zip(predictions, labels):
            matrix[true][pred] += 1
        return matrix

    def print_confusion_matrix(matrix, classes):
        col_w = 10
        header = "TRUE\\PRED".ljust(col_w) + "".join(c.rjust(col_w) for c in classes)
        print(header)
        print("-" * len(header))
        for true in classes:
            row = true.ljust(col_w)
            row += "".join(str(matrix[true][pred]).rjust(col_w) for pred in classes)
            print(row)

SOLUTION FOR EXERCISE 4:

    def interpret_results(accuracy, class_metrics):
        if accuracy >= 0.90:   verdict = "Excellent: ready for production"
        elif accuracy >= 0.75: verdict = "Good: acceptable for most use cases"
        elif accuracy >= 0.60: verdict = "Fair: needs improvement before production"
        else:                  verdict = "Poor: significant issues -- collect more data"

        worst = min(class_metrics, key=lambda c: class_metrics[c]["f1"])
        worst_f1 = class_metrics[worst]["f1"]
        return f"{verdict}. Accuracy={accuracy:.0%}. Worst class: {worst} (F1={worst_f1:.3f})."

SOLUTION FOR BONUS:

    def print_class_summary(predictions, labels, classes):
        print(f"{'Class':<12}| {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
        print("-" * 40)
        precisions, recalls, f1s = [], [], []
        for c in classes:
            m = compute_prf1(predictions, labels, c)
            print(f"{c:<12}| {m['precision']:>9.3f} | {m['recall']:>6.3f} | {m['f1']:>6.3f}")
            precisions.append(m['precision']); recalls.append(m['recall']); f1s.append(m['f1'])
        print("-" * 40)
        print(f"{'AVERAGE':<12}| {sum(precisions)/len(precisions):>9.3f} | {sum(recalls)/len(recalls):>6.3f} | {sum(f1s)/len(f1s):>6.3f}")
        acc = compute_accuracy(predictions, labels)
        print(f"Accuracy: {acc:.2%}")
"""

print("See SOLUTION block above.")
print("=" * 65)
print("END OF EXERCISE 05")
print("=" * 65)

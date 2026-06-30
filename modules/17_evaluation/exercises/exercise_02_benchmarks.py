"""
Exercise 02: Multiple Choice Benchmarks (MMLU, HellaSwag, ARC)
Module 17: LLM Evaluation & Benchmarks

TASKS:
  1. Implement evaluate_multiple_choice(predictions, answers) -- accuracy per benchmark
  2. Implement compute_accuracy(correct, total)              -- safe accuracy %
  3. Implement random_baseline(n_choices)                    -- expected random score
  4. Implement margin_over_random(accuracy, n_choices)       -- how much better than chance

Run:  python exercise_02_benchmarks.py
Deps: none (pure Python)
"""


# ─────────────────────────────────────────────────────────
# TASK 1: Evaluate Multiple Choice Questions
# ─────────────────────────────────────────────────────────

def evaluate_multiple_choice(predictions, answers):
    """
    Evaluate model predictions on multiple choice questions.

    Args:
        predictions: list of str, model's chosen answer letter ("A", "B", "C", or "D")
        answers:     list of str, correct answer letter for each question

    Returns:
        dict with keys:
          "correct":  int, number of correct predictions
          "total":    int, total number of questions
          "accuracy": float, fraction correct (0.0 to 1.0)

    Example:
        predictions = ["A", "B", "C", "D", "B"]
        answers     = ["A", "B", "A", "D", "C"]
        -> {"correct": 3, "total": 5, "accuracy": 0.6}

    HINT:
        correct = sum(1 for p, a in zip(predictions, answers) if p == a)
        total = len(answers)
        return {"correct": correct, "total": total, "accuracy": correct / total}
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Compute Accuracy (Safe Division)
# ─────────────────────────────────────────────────────────

def compute_accuracy(correct, total):
    """
    Compute accuracy as a percentage, handling the zero-division case.

    Args:
        correct: int, number of correct answers
        total:   int, total number of questions

    Returns:
        float: accuracy as a percentage (0.0 to 100.0)
               Returns 0.0 if total == 0.

    Example:
        compute_accuracy(3, 5)   -> 60.0
        compute_accuracy(10, 10) -> 100.0
        compute_accuracy(0, 0)   -> 0.0    (safe -- no division by zero)

    HINT:
        if total == 0: return 0.0
        return (correct / total) * 100.0
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Random Baseline
# ─────────────────────────────────────────────────────────

def random_baseline(n_choices):
    """
    Compute the expected accuracy of a random-guessing model.

    For multiple choice with n equally likely choices, random guessing
    gets 1/n accuracy on average.

    Args:
        n_choices: int, number of answer choices per question (e.g., 4 for MMLU)

    Returns:
        float: expected accuracy as a percentage (0.0 to 100.0)

    Example:
        random_baseline(4)   -> 25.0    (MMLU, HellaSwag, ARC all use 4 choices)
        random_baseline(2)   -> 50.0    (true/false)
        random_baseline(10)  -> 10.0    (10 options)

    HINT:
        return (1.0 / n_choices) * 100.0
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Margin Over Random
# ─────────────────────────────────────────────────────────

def margin_over_random(accuracy, n_choices):
    """
    Compute how much better a model is compared to random guessing.

    This tells you how much the model actually "knows" beyond luck.
    A model with accuracy = random baseline has learned nothing useful.

    Args:
        accuracy:  float, model's accuracy as a percentage (0.0 to 100.0)
        n_choices: int, number of answer choices

    Returns:
        float: accuracy - random_baseline (percentage points above random)
               Negative means model is WORSE than random.

    Example:
        margin_over_random(72.0, 4)  -> 47.0   (72 - 25 = 47 percentage points)
        margin_over_random(25.0, 4)  ->  0.0   (exactly random -- learned nothing)
        margin_over_random(20.0, 4)  -> -5.0   (worse than random!)

    HINT:
        return accuracy - random_baseline(n_choices)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 02: Multiple Choice Benchmarks")
    print("=" * 55)

    # Task 1: evaluate_multiple_choice
    print("\n--- Task 1: evaluate_multiple_choice ---")
    preds1 = ["A", "B", "C", "D", "B"]
    ans1   = ["A", "B", "A", "D", "C"]
    result1 = evaluate_multiple_choice(preds1, ans1)
    if result1 is None:
        print("  NOT IMPLEMENTED YET")
    else:
        expected = {"correct": 3, "total": 5, "accuracy": 0.6}
        ok_c = "PASS" if result1.get("correct") == 3 else "FAIL"
        ok_t = "PASS" if result1.get("total") == 5 else "FAIL"
        ok_a = "PASS" if abs(result1.get("accuracy", -1) - 0.6) < 0.001 else "FAIL"
        print(f"  {ok_c}  correct = {result1.get('correct')}  (expected 3)")
        print(f"  {ok_t}  total   = {result1.get('total')}  (expected 5)")
        print(f"  {ok_a}  accuracy = {result1.get('accuracy'):.3f}  (expected 0.600)")

    preds2 = ["A", "A", "A", "A"]
    ans2   = ["B", "B", "B", "B"]
    result2 = evaluate_multiple_choice(preds2, ans2)
    if result2 is not None:
        ok = "PASS" if result2.get("correct") == 0 else "FAIL"
        print(f"  {ok}  all wrong: correct = {result2.get('correct')}  (expected 0)")

    # Task 2: compute_accuracy
    print("\n--- Task 2: compute_accuracy ---")
    cases2 = [(3, 5, 60.0), (10, 10, 100.0), (0, 0, 0.0), (0, 4, 0.0), (1, 4, 25.0)]
    for c, t, exp in cases2:
        result = compute_accuracy(c, t)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if abs(result - exp) < 0.01 else "FAIL"
        print(f"  {status}  accuracy({c}, {t}) = {result:.1f}%  (expected {exp:.1f}%)")

    # Task 3: random_baseline
    print("\n--- Task 3: random_baseline ---")
    rb_cases = [(4, 25.0), (2, 50.0), (10, 10.0), (5, 20.0)]
    for n, exp in rb_cases:
        result = random_baseline(n)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if abs(result - exp) < 0.01 else "FAIL"
        print(f"  {status}  random_baseline({n}) = {result:.1f}%  (expected {exp:.1f}%)")

    # Task 4: margin_over_random
    print("\n--- Task 4: margin_over_random ---")
    mr_cases = [(72.0, 4, 47.0), (25.0, 4, 0.0), (20.0, 4, -5.0), (86.4, 4, 61.4)]
    for acc, n, exp in mr_cases:
        result = margin_over_random(acc, n)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if abs(result - exp) < 0.1 else "FAIL"
        print(f"  {status}  margin({acc}%, n={n}) = {result:+.1f}pp  (expected {exp:+.1f}pp)")

    # Bonus: full benchmark comparison
    print("\n--- BONUS: Benchmark Comparison Table ---")
    if (evaluate_multiple_choice(["A"], ["A"]) is not None
            and compute_accuracy(1, 1) is not None
            and random_baseline(4) is not None
            and margin_over_random(50, 4) is not None):

        bench_results = {
            "GPT-3.5 on MMLU":   (["B", "C", "A", "D", "C", "B", "A", "C"],
                                   ["B", "C", "A", "B", "C", "B", "A", "D"]),
            "GPT-4 on MMLU":     (["A", "C", "B", "D", "A", "C", "B", "A"],
                                   ["A", "C", "B", "D", "A", "C", "B", "A"]),
            "Random model":      (["A", "B", "A", "C", "B", "D", "C", "A"],
                                   ["B", "C", "A", "D", "C", "B", "B", "D"]),
        }
        print(f"\n  {'Model':<25}  {'Correct':>8}  {'Accuracy':>10}  {'Margin':>10}")
        print("  " + "-" * 60)
        rb = random_baseline(4)
        for model, (preds, ans) in bench_results.items():
            eval_result = evaluate_multiple_choice(preds, ans)
            acc_pct = compute_accuracy(eval_result["correct"], eval_result["total"])
            margin  = margin_over_random(acc_pct, 4)
            print(f"  {model:<25}  {eval_result['correct']:>8}/{eval_result['total']}  "
                  f"{acc_pct:>9.1f}%  {margin:>+9.1f}pp")
        print(f"\n  Random baseline: {rb:.1f}%")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def evaluate_multiple_choice(predictions, answers):
#     correct = sum(1 for p, a in zip(predictions, answers) if p == a)
#     total = len(answers)
#     return {"correct": correct, "total": total, "accuracy": correct / total}
#
# def compute_accuracy(correct, total):
#     if total == 0:
#         return 0.0
#     return (correct / total) * 100.0
#
# def random_baseline(n_choices):
#     return (1.0 / n_choices) * 100.0
#
# def margin_over_random(accuracy, n_choices):
#     return accuracy - random_baseline(n_choices)

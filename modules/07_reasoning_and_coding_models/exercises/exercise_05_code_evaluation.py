"""
Module 07 - Reasoning & Coding Models
Exercise 05: Code Generation Evaluation (Pass@k)

GLOSSARY
--------
HumanEval         : Standard benchmark for code generation models.
                    164 Python programming problems.
                    Each problem: function signature + docstring + test cases.
                    GPT-4 solves ~67% of problems; GPT-3.5 ~48%.
Pass@k            : The probability that AT LEAST ONE of k generated solutions
                    passes all tests for a problem.
                    Pass@1 = single-shot accuracy (model gets 1 attempt).
                    Pass@10 = with 10 attempts, does at least 1 pass?
n                 : Total solutions generated per problem (sampling budget).
c                 : Number of those solutions that pass all tests (correct count).
k                 : The k in Pass@k (how many tries we allow).
Pass@k formula    : 1 - C(n-c, k) / C(n, k)
                    = 1 - (probability ALL k picks are wrong)
Combinatorics     : C(n, k) = n! / (k! * (n-k)!) -- "n choose k"
                    Number of ways to pick k items from n.
Syntax Valid      : Code that parses without errors (AST can be built).
Code Quality      : Lines of code (LOC), cyclomatic complexity, docstring presence.
"""

import ast          # ast: parse Python code into Abstract Syntax Tree
import math         # math: for factorial and combinatorics
import re           # re: regular expressions

print("=" * 60)
print("Exercise 05: Code Generation Evaluation (Pass@k)")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Pass@k Formula
#
#  Background:
#    Pass@k = probability at least 1 correct solution in k samples.
#
#    Formula:
#      Pass@k = 1 - C(n-c, k) / C(n, k)
#
#    Where:
#      n = total solutions generated
#      c = number of correct solutions (pass all tests)
#      k = tries we allow
#      C(a, b) = combination = a! / (b! * (a-b)!)
#
#    Special cases:
#      if c == 0: Pass@k = 0.0  (no correct solutions)
#      if n < k:  undefined — raise ValueError or return None
#      if c == n: Pass@k = 1.0  (all solutions correct)
#
#    Example: n=10, c=3, k=1
#      C(7, 1) / C(10, 1) = 7/10 = 0.7 → Pass@1 = 1 - 0.7 = 0.3
#
#  Your Task:
#    Write: pass_at_k(n, c, k) -> float
#    Implement the Pass@k formula using math.comb(a, b).
#
#  C# Analogy:
#    long combinations = Factorial(a) / (Factorial(b) * Factorial(a-b));
# ============================================================

print("-" * 50)
print("EXERCISE 1: Pass@k Formula")
print("-" * 50)
print()


def pass_at_k(n, c, k):
    """
    Compute the Pass@k metric for code generation.

    Parameters:
        n (int): Total number of solutions generated.
        c (int): Number of correct solutions (pass all tests).
        k (int): Number of samples allowed.

    Returns:
        float: Pass@k probability in [0.0, 1.0].
               Returns 0.0 if c == 0.
               Returns None if n < k (undefined).
    """
    # TODO:
    # if n < k: return None    (can't pick k from fewer than k solutions)
    # if c == 0: return 0.0   (no correct solutions)
    # numerator   = math.comb(n - c, k)   <- ways to pick k wrong solutions
    # denominator = math.comb(n, k)       <- ways to pick any k solutions
    # return 1.0 - numerator / denominator
    pass  # Replace with your implementation


print(f"  {'n':>4} {'c':>4} {'k':>4}  {'Pass@k':>10}  Note")
print("  " + "-" * 50)
test_cases = [
    (10, 0,  1, "0 correct -> always 0.0"),
    (10, 3,  1, "3/10 correct, 1 try"),
    (10, 3, 10, "3/10 correct, 10 tries"),
    (10,10,  1, "all correct -> 1.0"),
    (10, 5,  5, "half correct, 5 tries"),
    ( 5,  2, 10, "k > n -> None"),
]
for n, c, k, note in test_cases:
    result = pass_at_k(n, c, k)
    result_str = f"{result:.4f}" if result is not None else "None"
    print(f"  {n:>4} {c:>4} {k:>4}  {result_str:>10}  {note}")
print()
print("  Expected: 0.0 / 0.3 / ~0.917 / 1.0 / ~0.974 / None")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Check Python Code Syntax Validity
#
#  Background:
#    Before running tests, check if generated code even parses.
#    Invalid syntax = automatic fail (0 points).
#
#    Python's ast.parse() raises SyntaxError if code is invalid.
#    A try/except around ast.parse() tells us if code is syntactically correct.
#
#  Your Task:
#    Write: is_syntax_valid(code_str) -> bool
#    Returns True if ast.parse(code_str) succeeds, False on SyntaxError.
#
#  C# Analogy:
#    try { SyntaxTree.ParseText(code); return true; }
#    catch (SyntaxException) { return false; }
# ============================================================

print("-" * 50)
print("EXERCISE 2: Syntax Validity Check")
print("-" * 50)
print()


def is_syntax_valid(code_str):
    """
    Check if a Python code string has valid syntax.

    Parameters:
        code_str (str): Python source code to check.

    Returns:
        bool: True if syntax is valid, False if SyntaxError.
    """
    # TODO:
    # try:
    #     ast.parse(code_str)
    #     return True
    # except SyntaxError:
    #     return False
    pass  # Replace with your implementation


snippets = [
    ("def add(a, b):\n    return a + b",        True,  "valid function"),
    ("def broken(\n    return 42",               False, "missing closing paren"),
    ("x = 1 + 2\ny = x * 3",                    True,  "valid multi-line"),
    ("if x > 0\n    print(x)",                  False, "missing colon after if"),
    ("class Foo:\n    pass",                     True,  "valid class"),
    ("def f():\n  return (",                     False, "unclosed paren"),
]

print(f"  {'Code (truncated)':<40} {'Valid?':>8}  {'Expected':>8}  {'OK?':>5}")
print("  " + "-" * 68)
for code, expected, desc in snippets:
    result = is_syntax_valid(code)
    preview = repr(code[:35])
    ok = "OK" if result == expected else "FAIL"
    print(f"  {preview:<40} {str(result):>8}  {str(expected):>8}  {ok:>5}  ({desc})")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Basic Code Quality Metrics
#
#  Background:
#    Even if code passes tests, we want to evaluate its quality.
#    Three simple metrics:
#
#    1. Lines of Code (LOC)
#       Total non-empty lines in the function body.
#       Fewer lines is usually better (simpler = more readable).
#
#    2. Has docstring?
#       Does the function start with a triple-quoted string?
#       Good documentation practice.
#
#    3. Has type hints?
#       Does the function signature include ":  " type annotations?
#       e.g., def add(a: int, b: int) -> int:
#       Better code quality for production.
#
#  Your Task:
#    Write: code_quality(code_str) -> dict
#    Returns: {"loc": int, "has_docstring": bool, "has_type_hints": bool}
#
#  C# Analogy:
#    int loc = code.Split('\n').Count(l => l.Trim() != "");
#    bool hasDoc = code.TrimStart().StartsWith("\"\"\"");
# ============================================================

print("-" * 50)
print("EXERCISE 3: Code Quality Metrics")
print("-" * 50)
print()


def code_quality(code_str):
    """
    Measure basic code quality metrics.

    Parameters:
        code_str (str): Python function source code.

    Returns:
        dict: {
            "loc"            : int  -- non-empty line count,
            "has_docstring"  : bool -- True if triple-quoted string at start of body,
            "has_type_hints" : bool -- True if "->" appears in the def line
        }
    """
    # TODO:
    # loc = len([l for l in code_str.split("\n") if l.strip()])
    #
    # has_docstring: check if '"""' or "'''" appears in the function body
    #   has_docstring = '"""' in code_str or "'''" in code_str
    #
    # has_type_hints: check if "->" appears on the def line
    #   def_line = [l for l in code_str.split("\n") if l.strip().startswith("def")]
    #   has_type_hints = any("->" in l for l in def_line)
    pass  # Replace with your implementation


code_high_quality = '''def add(a: int, b: int) -> int:
    """
    Add two numbers and return the result.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b
'''

code_low_quality = '''def add(a, b):
    return a + b
'''

code_medium = '''def multiply(x: float, y: float) -> float:
    return x * y
'''

for label, code in [("High quality", code_high_quality),
                     ("Low quality",  code_low_quality),
                     ("Medium",       code_medium)]:
    q = code_quality(code)
    if q:
        print(f"  {label}:")
        print(f"    LOC={q['loc']}, docstring={q['has_docstring']}, type_hints={q['has_type_hints']}")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Benchmark Summary Across Multiple Problems
#
#  Background:
#    A real evaluation runs Pass@k across ALL benchmark problems.
#    Final score = average Pass@k over all problems.
#
#    per_problem_results: list of (n, c) tuples
#      n = solutions generated per problem
#      c = correct solutions for that problem
#
#    For each problem: compute Pass@k.
#    Final score: mean of all per-problem Pass@k values.
#
#  Your Task:
#    Write: benchmark_pass_at_k(results, k) -> dict
#    results: list of (n, c) tuples
#    Returns: {"per_problem": list of float, "mean_pass_at_k": float,
#              "solved_count": int, "total": int}
#    solved_count = number of problems where Pass@k > 0.
#
#  C# Analogy:
#    double meanPassAtK = results.Select((r) => PassAtK(r.n, r.c, k)).Average();
# ============================================================

print("-" * 50)
print("EXERCISE 4: Benchmark Summary (Mean Pass@k)")
print("-" * 50)
print()


def benchmark_pass_at_k(results, k):
    """
    Compute mean Pass@k across a set of benchmark problems.

    Parameters:
        results (list of tuple): [(n, c), ...] per-problem generation results.
        k       (int)          : Number of attempts allowed (the k in Pass@k).

    Returns:
        dict: {
            "per_problem"    : list of float -- Pass@k for each problem,
            "mean_pass_at_k" : float         -- average over all problems,
            "solved_count"   : int           -- problems with Pass@k > 0,
            "total"          : int           -- total problems
        }
    """
    # TODO:
    # per_problem = []
    # for (n, c) in results:
    #     p = pass_at_k(n, c, k)
    #     per_problem.append(p if p is not None else 0.0)
    # mean = sum(per_problem) / len(per_problem)
    # solved = sum(1 for p in per_problem if p > 0)
    # return {"per_problem": per_problem, "mean_pass_at_k": mean,
    #         "solved_count": solved, "total": len(results)}
    pass  # Replace with your implementation


# Simulate 10 benchmark problems, each with 10 generated solutions
problem_results = [
    (10, 8),   # easy problem: 8/10 correct
    (10, 6),   # medium: 6/10 correct
    (10, 3),   # harder: 3/10 correct
    (10, 0),   # failed: 0/10 correct
    (10, 10),  # trivial: 10/10 correct
    (10, 5),
    (10, 2),
    (10, 7),
    (10, 1),
    (10, 0),   # another failure
]

for k_val in [1, 5, 10]:
    summary = benchmark_pass_at_k(problem_results, k=k_val)
    if summary:
        print(f"  Pass@{k_val}:")
        print(f"    Mean    : {summary['mean_pass_at_k']:.4f}  ({summary['mean_pass_at_k']*100:.1f}%)")
        print(f"    Solved  : {summary['solved_count']}/{summary['total']} problems")
        print()

print("  Insight: Pass@10 >> Pass@1 because more attempts = more chances to succeed.")
print()

print("=" * 60)
print("All exercises complete!")
print("=" * 60)

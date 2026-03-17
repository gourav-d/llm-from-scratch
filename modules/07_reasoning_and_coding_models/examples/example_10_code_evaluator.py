"""
Example 10: Code Evaluation & Testing

This example demonstrates:
1. HumanEval benchmark format
2. Pass@k metric calculation
3. Automatic test generation
4. Safe sandbox execution
5. Code quality metrics
6. Security checking

For .NET developers: This is like xUnit + code analysis + security scanning!
"""

import ast
import re
import math
import subprocess
import tempfile
import os
from typing import List, Tuple, Dict, Optional
from itertools import combinations


# ============================================================================
# PART 1: HUMANEVAL FORMAT & LOADING
# ============================================================================

class HumanEvalProblem:
    """
    Represents a single HumanEval problem
    Standard format for code generation benchmarks

    Like: LeetCode problem in Python format
    """

    def __init__(self, task_id: str, prompt: str, entry_point: str,
                 canonical_solution: str, test: str):
        """
        Args:
            task_id: Unique identifier (e.g., "HumanEval/0")
            prompt: Function signature + docstring
            entry_point: Function name to test
            canonical_solution: Reference implementation
            test: Test code to run
        """
        self.task_id = task_id
        self.prompt = prompt
        self.entry_point = entry_point
        self.canonical_solution = canonical_solution
        self.test = test

    def __repr__(self):
        return f"HumanEvalProblem({self.task_id})"


# Sample HumanEval problems (first 3 from actual dataset)
SAMPLE_PROBLEMS = [
    HumanEvalProblem(
        task_id="HumanEval/0",
        prompt='''from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
        entry_point="has_close_elements",
        canonical_solution='''    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
''',
        test='''
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
'''
    ),

    HumanEvalProblem(
        task_id="HumanEval/1",
        prompt='''from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
''',
        entry_point="separate_paren_groups",
        canonical_solution='''    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()

    return result
''',
        test='''
def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
    assert candidate('(()(())((())))') == ['(()(())((())))']
    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
'''
    ),

    HumanEvalProblem(
        task_id="HumanEval/2",
        prompt='''

def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
''',
        entry_point="truncate_number",
        canonical_solution='''    return number % 1.0
''',
        test='''
def check(candidate):
    assert candidate(3.5) == 0.5
    assert abs(candidate(1.33) - 0.33) < 1e-6
    assert abs(candidate(123.456) - 0.456) < 1e-6
'''
    ),
]


# ============================================================================
# PART 2: PASS@K METRIC CALCULATION
# ============================================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate Pass@k metric

    Formula: 1 - (C(n-c, k) / C(n, k))
    Where:
        n = total samples
        c = correct samples
        k = samples to consider

    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: Number of samples to pick

    Returns:
        Probability that at least one of k random picks is correct

    Example:
        >>> pass_at_k(10, 3, 1)  # 10 samples, 3 correct, pick 1
        0.3  # 30% chance that random pick is correct

        >>> pass_at_k(10, 3, 3)  # pick 3
        0.708  # 70.8% chance at least one is correct
    """
    # Edge case: Can't pick k wrong samples if there aren't enough
    if n - c < k:
        return 1.0

    # Edge case: No correct samples
    if c == 0:
        return 0.0

    # Calculate using combinations
    # C(n-c, k) = number of ways to pick k wrong samples
    # C(n, k) = number of ways to pick any k samples
    try:
        # Probability of picking all wrong samples
        prob_all_wrong = math.comb(n - c, k) / math.comb(n, k)

        # Pass@k = probability at least one is correct
        return 1.0 - prob_all_wrong

    except (ValueError, ZeroDivisionError):
        return 0.0


class PassAtKCalculator:
    """
    Calculates Pass@k metrics for code evaluation
    Shows success rate with different numbers of attempts

    Like: Test success rate metrics in CI/CD
    """

    def calculate(self, results: List[bool], k_values: List[int] = None) -> Dict[str, float]:
        """
        Calculate Pass@k for multiple k values

        Args:
            results: List of True/False (passed/failed) for each solution
            k_values: List of k values to compute (default: [1, 5, 10])

        Returns:
            Dictionary: {f"pass@{k}": probability}
        """
        if k_values is None:
            k_values = [1, 5, 10]

        n = len(results)  # Total samples
        c = sum(results)  # Correct samples

        metrics = {}

        for k in k_values:
            if k <= n:
                metrics[f"pass@{k}"] = pass_at_k(n, c, k)

        return metrics

    def print_results(self, metrics: Dict[str, float]):
        """Pretty print Pass@k results"""
        print("\n" + "="*50)
        print("PASS@K METRICS")
        print("="*50)

        for metric, value in sorted(metrics.items()):
            percentage = value * 100
            print(f"{metric:12s}: {percentage:6.2f}%")

        print("="*50)


# ============================================================================
# PART 3: SOLUTION EVALUATOR
# ============================================================================

class SolutionEvaluator:
    """
    Evaluates code solutions against test cases
    Runs code and checks if it passes tests

    Like: xUnit test runner in .NET
    """

    def evaluate(self, problem: HumanEvalProblem, solution: str) -> Tuple[bool, Optional[str]]:
        """
        Test if solution passes all test cases

        Args:
            problem: HumanEval problem with tests
            solution: Generated code to test

        Returns:
            (passed, error_message)
                passed: True if all tests pass
                error_message: None if passed, error details if failed
        """
        # Combine solution with test code
        # Test code expects function to be defined
        full_code = solution + "\n\n" + problem.test + "\n\ncheck(" + problem.entry_point + ")"

        # Execute in isolated namespace
        namespace = {}

        try:
            # Run the code
            # This:
            # 1. Defines the function from solution
            # 2. Defines the check() function from test
            # 3. Calls check() with the function
            exec(full_code, namespace)

            # If we get here, all assertions passed!
            return (True, None)

        except AssertionError as e:
            # One or more tests failed
            return (False, f"Test failed: {str(e)}")

        except SyntaxError as e:
            # Code has syntax errors
            return (False, f"Syntax error: {str(e)}")

        except Exception as e:
            # Runtime error (NameError, TypeError, etc.)
            return (False, f"Runtime error: {type(e).__name__}: {str(e)}")


# ============================================================================
# PART 4: SANDBOX EXECUTION
# ============================================================================

class SimpleSandbox:
    """
    Runs code in isolated subprocess for safety
    Prevents malicious code from harming system

    Like: Running tests in separate AppDomain in .NET
    """

    def __init__(self, timeout: int = 5):
        """
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout

    def execute(self, code: str, test_code: str = "") -> Tuple[bool, str]:
        """
        Execute code in sandbox (subprocess)

        Args:
            code: Code to run
            test_code: Optional test code to append

        Returns:
            (success, output_or_error)
        """
        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            # Write complete program
            f.write(code)
            if test_code:
                f.write('\n\n')
                f.write(test_code)
            temp_file = f.name

        try:
            # Run in separate process
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Check exit code
            if result.returncode == 0:
                return (True, result.stdout)
            else:
                return (False, result.stderr)

        except subprocess.TimeoutExpired:
            return (False, f"Timeout after {self.timeout} seconds (infinite loop?)")

        except Exception as e:
            return (False, f"Execution error: {str(e)}")

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass


# ============================================================================
# PART 5: AUTOMATIC TEST GENERATION
# ============================================================================

class TestGenerator:
    """
    Automatically generates test cases from docstrings
    Extracts examples and converts to assertions

    Like: Extracting XML doc examples in C#
    """

    def extract_from_docstring(self, code: str) -> List[Tuple[str, str]]:
        """
        Extract test cases from docstring examples

        Looks for doctest format:
            >>> function_call(args)
            expected_result

        Args:
            code: Function code with docstring

        Returns:
            List of (input_code, expected_output) tuples
        """
        tests = []

        # Parse code to get docstring
        try:
            tree = ast.parse(code)
        except:
            return tests

        # Find first function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    tests = self._parse_doctest(docstring)
                    break

        return tests

    def _parse_doctest(self, docstring: str) -> List[Tuple[str, str]]:
        """
        Parse doctest examples from docstring

        Format:
            >>> code
            result

        Args:
            docstring: Docstring text

        Returns:
            List of (code, result) tuples
        """
        tests = []
        lines = docstring.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Find doctest line (starts with >>>)
            if line.startswith('>>>'):
                # Extract code
                code = line[3:].strip()

                # Next line might be expected output
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()

                    # Skip if it's another >>> line
                    if not next_line.startswith('>>>') and next_line:
                        tests.append((code, next_line))

                i += 2
            else:
                i += 1

        return tests

    def generate_test_function(self, function_name: str, tests: List[Tuple[str, str]]) -> str:
        """
        Generate test function from extracted tests

        Args:
            function_name: Name of function being tested
            tests: List of (input, expected) tuples

        Returns:
            Test function code
        """
        if not tests:
            return ""

        # Build test function
        test_code = f"def test_{function_name}():\n"
        test_code += '    """Auto-generated tests from docstring"""\n'

        for input_code, expected in tests:
            # Generate assertion
            test_code += f"    assert {input_code} == {expected}\n"

        test_code += "    print('All tests passed!')\n"

        return test_code


# ============================================================================
# PART 6: CODE QUALITY METRICS
# ============================================================================

class CodeQualityAnalyzer:
    """
    Analyzes code quality beyond just correctness
    Checks readability, complexity, style

    Like: Code analyzers (Roslyn, SonarQube) in .NET
    """

    def analyze(self, code: str) -> Dict:
        """
        Analyze code and return quality metrics

        Args:
            code: Python code to analyze

        Returns:
            Dictionary of metrics:
                - lines_of_code: Non-empty, non-comment lines
                - cyclomatic_complexity: Code complexity
                - has_docstring: Has documentation?
                - has_type_hints: Has type annotations?
                - num_functions: Number of functions
                - num_comments: Number of comments
        """
        metrics = {
            "lines_of_code": 0,
            "cyclomatic_complexity": 0,
            "has_docstring": False,
            "has_type_hints": False,
            "num_functions": 0,
            "num_comments": 0,
            "avg_line_length": 0,
        }

        # Count lines
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        metrics["lines_of_code"] = len(non_empty_lines)

        # Average line length
        if non_empty_lines:
            metrics["avg_line_length"] = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)

        # Count comments
        metrics["num_comments"] = len([l for l in lines if l.strip().startswith('#')])

        # Parse AST
        try:
            tree = ast.parse(code)
        except:
            return metrics

        # Analyze functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics["num_functions"] += 1

                # Check for docstring
                if ast.get_docstring(node):
                    metrics["has_docstring"] = True

                # Check for type hints
                has_return_type = node.returns is not None
                has_param_types = any(arg.annotation for arg in node.args.args)

                if has_return_type or has_param_types:
                    metrics["has_type_hints"] = True

                # Calculate complexity
                complexity = self._calculate_complexity(node)
                metrics["cyclomatic_complexity"] = max(
                    metrics["cyclomatic_complexity"],
                    complexity
                )

        return metrics

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """
        Calculate cyclomatic complexity
        Counts independent paths through code

        Complexity = 1 (base) + number of decision points

        Args:
            node: Function AST node

        Returns:
            Complexity score (1 = simplest)
        """
        complexity = 1

        # Count decision points
        for child in ast.walk(node):
            # Each branching statement adds complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1

            # Boolean operators add complexity
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def score(self, metrics: Dict) -> float:
        """
        Calculate overall quality score (0-100)

        Args:
            metrics: Metrics from analyze()

        Returns:
            Score from 0 to 100
        """
        score = 100.0

        # Penalize high complexity
        complexity = metrics.get("cyclomatic_complexity", 0)
        if complexity > 20:
            score -= 30
        elif complexity > 10:
            score -= 20
        elif complexity > 5:
            score -= 10

        # Reward documentation
        if metrics.get("has_docstring"):
            score += 10
        else:
            score -= 15

        # Reward type hints
        if metrics.get("has_type_hints"):
            score += 10
        else:
            score -= 10

        # Penalize very long code (might be overengineered)
        if metrics.get("lines_of_code", 0) > 100:
            score -= 15
        elif metrics.get("lines_of_code", 0) > 50:
            score -= 5

        # Reward reasonable line length
        avg_len = metrics.get("avg_line_length", 0)
        if 40 <= avg_len <= 80:
            score += 5
        elif avg_len > 100:
            score -= 10

        # Reward comments
        if metrics.get("num_comments", 0) > 0:
            score += 5

        return max(0, min(100, score))


# ============================================================================
# PART 7: SECURITY CHECKER
# ============================================================================

class SecurityChecker:
    """
    Checks code for security vulnerabilities
    Prevents dangerous code patterns

    Like: Security scanning tools in .NET (SonarQube, Veracode)
    """

    def check(self, code: str) -> Tuple[bool, List[str]]:
        """
        Scan code for security issues

        Args:
            code: Python code to check

        Returns:
            (is_safe, list_of_issues)
        """
        issues = []

        # Check for dangerous imports
        if re.search(r'\bos\.system\s*\(', code):
            issues.append("CRITICAL: Uses os.system() - command injection risk")

        if re.search(r'\bsubprocess\.(call|run|Popen).*shell\s*=\s*True', code):
            issues.append("CRITICAL: Uses subprocess with shell=True - command injection risk")

        # Check for eval/exec
        if re.search(r'\beval\s*\(', code):
            issues.append("CRITICAL: Uses eval() - arbitrary code execution")

        if re.search(r'\bexec\s*\(', code):
            issues.append("CRITICAL: Uses exec() - arbitrary code execution")

        # Check for file operations
        if re.search(r'\bopen\s*\(', code):
            # Check if path is validated
            if not re.search(r'os\.path\.(abspath|realpath|normpath)', code):
                issues.append("WARNING: Uses open() without path validation - path traversal risk")

        # Check for network operations
        if re.search(r'\burllib\.(request|urlopen)', code):
            issues.append("INFO: Makes HTTP requests - verify target URLs")

        if re.search(r'\brequests\.(get|post)', code):
            issues.append("INFO: Makes HTTP requests - verify target URLs")

        # Check for SQL (simplified)
        if re.search(r'f["\'].*SELECT.*FROM', code, re.IGNORECASE):
            issues.append("CRITICAL: Uses f-string in SQL - SQL injection risk")

        if re.search(r'\.format\(.*\).*SELECT.*FROM', code, re.IGNORECASE):
            issues.append("CRITICAL: Uses .format() in SQL - SQL injection risk")

        # Check for pickle (deserialization vulnerability)
        if re.search(r'\bpickle\.loads?\s*\(', code):
            issues.append("WARNING: Uses pickle - arbitrary code execution on untrusted data")

        is_safe = len([i for i in issues if i.startswith("CRITICAL")]) == 0

        return (is_safe, issues)


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

def demo_humaneval_format():
    """Demonstrate HumanEval problem format"""
    print("\n" + "="*70)
    print("DEMO 1: HUMANEVAL FORMAT")
    print("="*70)

    problem = SAMPLE_PROBLEMS[0]

    print(f"\nProblem ID: {problem.task_id}")
    print(f"\nPrompt (what model sees):")
    print(problem.prompt)

    print(f"\nCanonical Solution (reference implementation):")
    print(problem.canonical_solution)

    print(f"\nTest Cases (hidden from model):")
    print(problem.test)


def demo_pass_at_k():
    """Demonstrate Pass@k calculation"""
    print("\n" + "="*70)
    print("DEMO 2: PASS@K METRICS")
    print("="*70)

    # Simulate: Generated 20 solutions, 6 are correct
    n = 20
    c = 6

    print(f"\nScenario:")
    print(f"  - Generated {n} solutions")
    print(f"  - {c} are correct")
    print(f"  - {n-c} are wrong")

    print(f"\nPass@k for different k values:")

    for k in [1, 3, 5, 10, 20]:
        prob = pass_at_k(n, c, k)
        print(f"  Pass@{k:2d}: {prob*100:6.2f}%")

    print("\nInterpretation:")
    print(f"  - Pass@1  = {pass_at_k(n,c,1)*100:.1f}%: Picking 1 random solution has {pass_at_k(n,c,1)*100:.1f}% success")
    print(f"  - Pass@10 = {pass_at_k(n,c,10)*100:.1f}%: Picking 10 random solutions has {pass_at_k(n,c,10)*100:.1f}% success")


def demo_solution_evaluation():
    """Demonstrate solution evaluation"""
    print("\n" + "="*70)
    print("DEMO 3: SOLUTION EVALUATION")
    print("="*70)

    problem = SAMPLE_PROBLEMS[2]  # truncate_number
    evaluator = SolutionEvaluator()

    # Correct solution
    print("\nTest Case 1: Correct Solution")
    print("-" * 60)

    correct_solution = """
def truncate_number(number: float) -> float:
    return number % 1.0
"""

    print("Solution:")
    print(correct_solution)

    passed, error = evaluator.evaluate(problem, correct_solution)
    if passed:
        print("✓ PASSED: All tests passed!")
    else:
        print(f"✗ FAILED: {error}")

    # Incorrect solution
    print("\nTest Case 2: Incorrect Solution")
    print("-" * 60)

    wrong_solution = """
def truncate_number(number: float) -> float:
    return int(number)  # Returns integer part, not decimal!
"""

    print("Solution:")
    print(wrong_solution)

    passed, error = evaluator.evaluate(problem, wrong_solution)
    if passed:
        print("✓ PASSED: All tests passed!")
    else:
        print(f"✗ FAILED: {error}")


def demo_sandbox_execution():
    """Demonstrate sandbox execution"""
    print("\n" + "="*70)
    print("DEMO 4: SANDBOX EXECUTION")
    print("="*70)

    sandbox = SimpleSandbox(timeout=2)

    # Safe code
    print("\nTest 1: Safe Code")
    print("-" * 60)

    safe_code = """
def add(a, b):
    return a + b

print(add(2, 3))
"""

    success, output = sandbox.execute(safe_code)
    print(f"Success: {success}")
    print(f"Output: {output.strip()}")

    # Infinite loop (will timeout)
    print("\nTest 2: Infinite Loop (will timeout)")
    print("-" * 60)

    infinite_loop = """
while True:
    pass
"""

    success, output = sandbox.execute(infinite_loop)
    print(f"Success: {success}")
    print(f"Output: {output}")


def demo_test_generation():
    """Demonstrate automatic test generation"""
    print("\n" + "="*70)
    print("DEMO 5: AUTOMATIC TEST GENERATION")
    print("="*70)

    generator = TestGenerator()

    code = '''
def factorial(n: int) -> int:
    """
    Calculate factorial of n.

    >>> factorial(0)
    1
    >>> factorial(1)
    1
    >>> factorial(5)
    120
    >>> factorial(10)
    3628800
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''

    print("Input code:")
    print(code)

    # Extract tests
    tests = generator.extract_from_docstring(code)

    print(f"\nExtracted {len(tests)} test cases:")
    for input_code, expected in tests:
        print(f"  {input_code} → {expected}")

    # Generate test function
    test_func = generator.generate_test_function("factorial", tests)

    print("\nGenerated test function:")
    print(test_func)


def demo_code_quality():
    """Demonstrate code quality analysis"""
    print("\n" + "="*70)
    print("DEMO 6: CODE QUALITY ANALYSIS")
    print("="*70)

    analyzer = CodeQualityAnalyzer()

    # Good code
    good_code = '''
def fibonacci(n: int) -> int:
    """
    Calculate nth Fibonacci number.

    Args:
        n: Position in sequence

    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''

    # Bad code
    bad_code = '''
def f(x):
    if x<0:
        if x<-10:
            if x<-100:
                if x<-1000:
                    return 1
                else:
                    return 2
            else:
                return 3
        else:
            return 4
    else:
        if x>10:
            return 5
        else:
            return 6
'''

    print("\n1. Good Code:")
    print("-" * 60)
    print(good_code)

    good_metrics = analyzer.analyze(good_code)
    good_score = analyzer.score(good_metrics)

    print("\nMetrics:")
    for key, value in good_metrics.items():
        print(f"  {key}: {value}")
    print(f"\nQuality Score: {good_score}/100")

    print("\n2. Bad Code:")
    print("-" * 60)
    print(bad_code)

    bad_metrics = analyzer.analyze(bad_code)
    bad_score = analyzer.score(bad_metrics)

    print("\nMetrics:")
    for key, value in bad_metrics.items():
        print(f"  {key}: {value}")
    print(f"\nQuality Score: {bad_score}/100")


def demo_security_checking():
    """Demonstrate security checking"""
    print("\n" + "="*70)
    print("DEMO 7: SECURITY CHECKING")
    print("="*70)

    checker = SecurityChecker()

    # Safe code
    safe_code = '''
def greet(name: str) -> str:
    return f"Hello, {name}!"
'''

    # Dangerous code
    dangerous_code = '''
def run_command(user_input):
    import os
    os.system(f"echo {user_input}")  # Command injection!
'''

    print("\n1. Safe Code:")
    print("-" * 60)
    print(safe_code)

    is_safe, issues = checker.check(safe_code)
    print(f"\nSafe: {is_safe}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No security issues found!")

    print("\n2. Dangerous Code:")
    print("-" * 60)
    print(dangerous_code)

    is_safe, issues = checker.check(dangerous_code)
    print(f"\nSafe: {is_safe}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")


def demo_complete_evaluation():
    """Demonstrate complete evaluation pipeline"""
    print("\n" + "="*70)
    print("DEMO 8: COMPLETE EVALUATION PIPELINE")
    print("="*70)

    problem = SAMPLE_PROBLEMS[0]
    evaluator = SolutionEvaluator()
    quality_analyzer = CodeQualityAnalyzer()
    security_checker = SecurityChecker()

    # Simulate: Model generated 10 solutions
    # (For demo, we'll use variations of the solution)
    solutions = []

    # Solution 1: Correct
    solutions.append('''
def has_close_elements(numbers, threshold):
    for i, num1 in enumerate(numbers):
        for j, num2 in enumerate(numbers):
            if i != j and abs(num1 - num2) < threshold:
                return True
    return False
''')

    # Solution 2: Correct but different approach
    solutions.append('''
def has_close_elements(numbers, threshold):
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                if abs(elem - elem2) < threshold:
                    return True
    return False
''')

    # Solution 3: Wrong (forgets to check idx != idx2)
    solutions.append('''
def has_close_elements(numbers, threshold):
    for num1 in numbers:
        for num2 in numbers:
            if abs(num1 - num2) < threshold:
                return True
    return False
''')

    # Add more solutions (mix correct and wrong)
    for i in range(7):
        if i % 2 == 0:
            solutions.append(solutions[0])  # Correct
        else:
            solutions.append(solutions[2])  # Wrong

    print(f"\nEvaluating {len(solutions)} solutions...")
    print("="*60)

    results = []
    for i, solution in enumerate(solutions, 1):
        print(f"\nSolution {i}:")

        # 1. Functional correctness
        passed, error = evaluator.evaluate(problem, solution)
        results.append(passed)

        print(f"  Correctness: {'✓ PASS' if passed else '✗ FAIL'}")
        if error:
            print(f"    Error: {error[:50]}...")

        # 2. Code quality
        metrics = quality_analyzer.analyze(solution)
        quality_score = quality_analyzer.score(metrics)
        print(f"  Quality Score: {quality_score:.1f}/100")
        print(f"  Complexity: {metrics['cyclomatic_complexity']}")

        # 3. Security
        is_safe, issues = security_checker.check(solution)
        print(f"  Security: {'✓ SAFE' if is_safe else '⚠ ISSUES'}")

    # Calculate Pass@k
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)

    calc = PassAtKCalculator()
    passk_metrics = calc.calculate(results, k_values=[1, 5, 10])
    calc.print_results(passk_metrics)

    print(f"\nSummary:")
    print(f"  Total solutions: {len(solutions)}")
    print(f"  Correct: {sum(results)}")
    print(f"  Wrong: {len(results) - sum(results)}")


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("CODE EVALUATION & TESTING EXAMPLES")
    print("Measuring Code Quality Like a Pro!")
    print("="*70)

    # Run all demos
    demo_humaneval_format()
    demo_pass_at_k()
    demo_solution_evaluation()
    demo_sandbox_execution()
    demo_test_generation()
    demo_code_quality()
    demo_security_checking()
    demo_complete_evaluation()

    print("\n" + "="*70)
    print("ALL DEMOS COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. HumanEval is the standard benchmark")
    print("  2. Pass@k measures success rate with k attempts")
    print("  3. Sandbox execution prevents malicious code")
    print("  4. Quality metrics go beyond just correctness")
    print("  5. Security checking is essential")
    print("  6. Complete evaluation = Correctness + Quality + Security")
    print("\nModule 7 Complete! You're now an LLM expert! 🎉")


if __name__ == "__main__":
    main()

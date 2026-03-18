# Project 4: Auto Test Writer

**Generate comprehensive unit tests automatically**

---

## What You'll Build

An AI that writes tests for you:
- Generates unit tests from code
- Covers edge cases automatically
- Follows testing best practices
- Saves hours of manual test writing

**Saves Time:** Writing tests manually: 30-60 min/function → **With this: 30 seconds!**

---

## Why This Is Useful

### Without This Tool
```python
# You write a function
def calculate_discount(price, discount_percent):
    return price * (1 - discount_percent / 100)

# Then spend 30 minutes writing tests... 😓
```

### With This Tool
```bash
# Auto-generate tests
test-writer generate calculate_discount

# Creates complete test file:
# ✅ Happy path tests
# ✅ Edge cases (0%, 100%, negative)
# ✅ Error cases (invalid inputs)
# ✅ Boundary conditions
# All in 30 seconds! 🚀
```

**Frequency of Use:** Weekly! Whenever you write new functions.

---

## Key Features

### 1. Auto-Generate Tests
```python
from auto_test_writer import TestWriter

writer = TestWriter()

# Your function
code = """
def validate_email(email: str) -> bool:
    '''Check if email is valid'''
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
"""

# Generate tests
tests = writer.generate_tests(code)
print(tests)
```

**Generated Output:**
```python
import pytest
from mymodule import validate_email

class TestValidateEmail:
    """Test cases for validate_email function"""

    # Happy path tests
    def test_valid_email_simple(self):
        """Test with simple valid email"""
        assert validate_email("user@example.com") == True

    def test_valid_email_with_plus(self):
        """Test email with plus sign"""
        assert validate_email("user+tag@example.com") == True

    def test_valid_email_with_subdomain(self):
        """Test email with subdomain"""
        assert validate_email("user@mail.example.com") == True

    # Edge cases
    def test_invalid_email_no_at(self):
        """Test email without @ symbol"""
        assert validate_email("userexample.com") == False

    def test_invalid_email_no_domain(self):
        """Test email without domain"""
        assert validate_email("user@") == False

    def test_invalid_email_no_tld(self):
        """Test email without TLD"""
        assert validate_email("user@example") == False

    def test_empty_string(self):
        """Test with empty string"""
        assert validate_email("") == False

    def test_none_input(self):
        """Test with None input"""
        with pytest.raises(TypeError):
            validate_email(None)

    # Boundary cases
    def test_very_long_email(self):
        """Test with very long email"""
        long_email = "a" * 100 + "@example.com"
        result = validate_email(long_email)
        # Should handle long emails gracefully

    def test_special_characters(self):
        """Test with special characters"""
        assert validate_email("user!#$%@example.com") == False
```

**AI automatically identifies:**
- ✅ Valid inputs (happy path)
- ✅ Invalid inputs (error cases)
- ✅ Edge cases (empty, None, special chars)
- ✅ Boundary conditions (very long, very short)

---

## Example Usage

### Generate Tests for File
```bash
# Auto-generate tests for entire file
test-writer generate mymodule.py

# Creates: tests/test_mymodule.py
# With tests for ALL functions!
```

### Generate Tests for Specific Function
```python
# Generate tests for one function
from auto_test_writer import TestWriter

writer = TestWriter()

# Your function
code = """
def calculate_tax(amount, rate, country="US"):
    if amount < 0:
        raise ValueError("Amount must be positive")
    if country == "US":
        return amount * rate
    elif country == "UK":
        return amount * (rate + 0.20)  # VAT
    else:
        raise ValueError(f"Unknown country: {country}")
"""

tests = writer.generate_tests(code, framework="pytest")
```

**Generated Tests:**
```python
class TestCalculateTax:
    # Happy path
    def test_us_tax(self):
        assert calculate_tax(100, 0.1, "US") == 10.0

    def test_uk_tax_with_vat(self):
        assert calculate_tax(100, 0.1, "UK") == 30.0

    def test_default_country(self):
        assert calculate_tax(100, 0.1) == 10.0

    # Error cases
    def test_negative_amount(self):
        with pytest.raises(ValueError, match="must be positive"):
            calculate_tax(-100, 0.1)

    def test_unknown_country(self):
        with pytest.raises(ValueError, match="Unknown country"):
            calculate_tax(100, 0.1, "FR")

    # Edge cases
    def test_zero_amount(self):
        assert calculate_tax(0, 0.1) == 0.0

    def test_zero_rate(self):
        assert calculate_tax(100, 0.0) == 0.0

    # Boundary cases
    def test_very_large_amount(self):
        assert calculate_tax(1e10, 0.1) > 0
```

---

## How It Works

### Step 1: Analyze Function
```
Input: Function code

AI Analysis:
- Function signature (params, types, defaults)
- Docstring (understand purpose)
- Code logic (if/else, loops, calculations)
- Error handling (raises, returns)
```

### Step 2: Identify Test Cases
```
Test Categories:
1. Happy Path
   - Typical valid inputs
   - Expected outputs

2. Edge Cases
   - Empty inputs ([], "", None)
   - Zero values
   - Very large/small values
   - Special characters

3. Error Cases
   - Invalid types
   - Out of range values
   - Missing required params

4. Boundary Conditions
   - Min/max values
   - First/last elements
   - Off-by-one scenarios
```

### Step 3: Generate Test Code
```
For each test case:
- Generate test method name
- Create test docstring
- Write assertion or exception check
- Add arrange/act/assert pattern
```

---

## Advanced Features

### 1. Increase Coverage
```python
# Target specific coverage
tests = writer.generate_tests(
    code,
    target_coverage=90  # Generate tests until 90% coverage
)
```

### 2. Different Test Frameworks
```python
# pytest
tests = writer.generate_tests(code, framework="pytest")

# unittest
tests = writer.generate_tests(code, framework="unittest")

# pytest + fixtures
tests = writer.generate_tests(code, use_fixtures=True)
```

### 3. Integration Tests
```python
# Generate integration tests
tests = writer.generate_integration_tests(
    functions=[func1, func2, func3],
    test_workflow=True
)
```

### 4. Property-Based Tests
```python
# Generate hypothesis tests
tests = writer.generate_property_tests(code)

# Output:
# @given(st.integers(), st.floats())
# def test_properties(amount, rate):
#     result = calculate_tax(amount, rate)
#     assert result >= 0  # Tax can't be negative
```

---

## Real-World Use Cases

### 1. Legacy Code
```bash
# Add tests to untested code
test-writer generate legacy_module.py --coverage 80
# Generates comprehensive tests for old code
```

### 2. TDD (Test-Driven Development)
```bash
# Write function signature, generate tests first
test-writer generate --from-signature "def process_payment(amount, card)"
# Then implement to pass tests
```

### 3. Regression Testing
```python
# Generate tests for bug fix
def fixed_function():
    # Fixed code
    pass

tests = writer.generate_regression_tests(
    before_code=buggy_code,
    after_code=fixed_code
)
# Ensures bug doesn't come back
```

### 4. API Testing
```python
# Generate tests for API endpoints
tests = writer.generate_api_tests(
    endpoint="/api/users",
    methods=["GET", "POST", "PUT", "DELETE"]
)
```

---

## C# Comparison

| C# Tool | Python Tool | This Project |
|---------|-------------|--------------|
| IntelliTest | pytest-automatically | Manual |
| Pex | Hypothesis | Property tests |
| xUnit + Moq | pytest + unittest.mock | Unit tests |
| **This tool** | - | **AI-generated!** |

---

## Project Structure

```
04_auto_test_writer/
├── README.md
├── requirements.txt
│
├── test_writer.py           # Main test generator
├── analyzers/
│   ├── function_analyzer.py # Analyze functions
│   ├── signature_parser.py  # Parse signatures
│   └── logic_analyzer.py    # Understand logic
│
├── generators/
│   ├── unit_test_generator.py      # Unit tests
│   ├── integration_test_generator.py # Integration
│   ├── property_test_generator.py   # Property-based
│   └── edge_case_finder.py         # Find edge cases
│
└── examples/
    ├── example_01_simple_function.py
    ├── example_02_class_tests.py
    ├── example_03_integration_tests.py
    └── example_04_api_tests.py
```

---

## Getting Started

```bash
cd projects/04_auto_test_writer
pip install -r requirements.txt

# Generate tests for a file
python -m test_writer generate mycode.py

# Generate for specific function
python -m test_writer function "def my_func(x): return x*2"

# Set coverage target
python -m test_writer generate mycode.py --coverage 90
```

---

## Difficulty: ⭐⭐ Intermediate

**Time Estimate:** 6-8 hours

**Prerequisites:**
- Module 7 Lessons 6, 9 (Code understanding, generation)
- Testing knowledge (pytest/unittest)
- Understanding of edge cases

---

## Success Criteria

- [x] Generates valid test code
- [x] Covers happy path + edge cases
- [x] Follows testing best practices
- [x] Tests actually run and pass
- [x] Saves time vs manual writing

---

**Stop writing boring tests manually!** ✅

Let AI handle it while you focus on building features!

"""
Example 10: Error Handling (Exceptions) for .NET Developers
This file demonstrates Python exception handling with detailed comments and C# comparisons.
"""

# ============================================
# SECTION 1: BASIC TRY/EXCEPT
# ============================================

print("=== SECTION 1: BASIC TRY/EXCEPT ===\n")

# C#:
# try {
#     int num = int.Parse(Console.ReadLine());
# }
# catch (FormatException ex) {
#     Console.WriteLine("Invalid number!");
# }

# Python:
# try:
#     num = int(input("Enter number: "))
# except ValueError:
#     print("Invalid number!")

# Without error handling - Program crashes!
print("Without error handling:")
print("  If you enter 'abc', program would crash with ValueError")

print()

# With error handling
print("With error handling:")
try:
    # This will work with valid input
    num = int("42")  # Using string instead of input for demo
    print(f"  Successfully converted: {num}")
except ValueError:
    print("  That's not a valid number!")

print()

# Example with invalid input
print("Handling invalid input:")
try:
    num = int("abc")  # This will raise ValueError
    print(f"  Successfully converted: {num}")
except ValueError:
    print("  Error: Invalid number format!")

print()

# ============================================
# SECTION 2: CATCHING MULTIPLE EXCEPTIONS
# ============================================

print("=== SECTION 2: CATCHING MULTIPLE EXCEPTIONS ===\n")

# Method 1: Multiple except blocks
print("Method 1: Multiple except blocks:")
try:
    numbers = [1, 2, 3]
    # result = 10 / 0  # ZeroDivisionError
    result = numbers[5]  # IndexError
except ZeroDivisionError:
    print("  Error: Cannot divide by zero!")
except IndexError:
    print("  Error: Index out of range!")

print()

# Method 2: Single except for multiple types
# C#:
# catch (Exception ex) when (ex is FormatException || ex is OverflowException)
# Python: except (ValueError, TypeError) as e:

print("Method 2: Single except for multiple types:")
try:
    # result = 10 / 0  # ZeroDivisionError
    result = int("abc")  # ValueError
except (ValueError, ZeroDivisionError) as e:
    print(f"  Error occurred: {e}")
    print(f"  Error type: {type(e).__name__}")

print()

# Method 3: Catch all exceptions (use carefully!)
print("Method 3: Catch all exceptions:")
try:
    result = 10 / 0
except Exception as e:
    print(f"  An error occurred: {e}")

print()
print("⚠️  Warning: Catching all exceptions can hide bugs!")
print("   Be specific when possible.")

print()

# ============================================
# SECTION 3: GETTING EXCEPTION DETAILS
# ============================================

print("=== SECTION 3: GETTING EXCEPTION DETAILS ===\n")

try:
    result = 10 / 0
except ZeroDivisionError as e:
    print("Exception details:")
    print(f"  Error type: {type(e)}")
    print(f"  Error message: {e}")
    print(f"  Error args: {e.args}")

print()

# Multiple exceptions with details
def divide(a, b):
    """Divide with detailed error handling."""
    try:
        return a / b
    except ZeroDivisionError as e:
        print(f"  ZeroDivisionError: {e}")
        return None
    except TypeError as e:
        print(f"  TypeError: {e}")
        return None

print("Testing divide function:")
print(f"divide(10, 2) = {divide(10, 2)}")
print(f"divide(10, 0) = {divide(10, 0)}")
print(f"divide(10, 'a') = {divide(10, 'a')}")

print()

# ============================================
# SECTION 4: ELSE CLAUSE
# ============================================

print("=== SECTION 4: ELSE CLAUSE ===\n")

# else: Runs if NO exception occurred
# C# doesn't have direct equivalent

print("The 'else' clause runs if NO exception occurred:")

try:
    num = int("42")  # This works
except ValueError:
    print("  Error: Invalid number!")
else:
    print(f"  Success! Number is: {num}")
    print("  This else block runs only if no exception occurred")

print()

# Example with exception
print("Example with exception:")
try:
    num = int("abc")  # This raises ValueError
except ValueError:
    print("  Error: Invalid number!")
else:
    print("  This won't print because exception occurred")

print()

# ============================================
# SECTION 5: FINALLY CLAUSE
# ============================================

print("=== SECTION 5: FINALLY CLAUSE ===\n")

# finally: ALWAYS runs, whether exception occurred or not
# C#: finally { ... } - Same concept!

print("The 'finally' clause ALWAYS runs:")

try:
    num = int("42")
    print(f"  Conversion successful: {num}")
except ValueError:
    print("  Error occurred!")
finally:
    print("  This ALWAYS runs (cleanup code goes here)")

print()

# Example with exception
print("Example with exception:")
try:
    num = int("abc")
    print(f"  Conversion successful: {num}")
except ValueError:
    print("  Error: Invalid number!")
finally:
    print("  Finally runs even when exception occurred")

print()

# Practical use - File cleanup
print("Practical use - Resource cleanup:")
file = None
try:
    # Simulate file operation
    print("  Opening file...")
    file = "simulated_file"
    # Do something with file
    print("  Processing file...")
except Exception as e:
    print(f"  Error: {e}")
finally:
    if file:
        print("  Closing file (cleanup in finally)")
        file = None

print()

# ============================================
# SECTION 6: COMPLETE TRY/EXCEPT/ELSE/FINALLY
# ============================================

print("=== SECTION 6: COMPLETE TRY/EXCEPT/ELSE/FINALLY ===\n")

def safe_divide(a, b):
    """Divide with complete error handling."""
    try:
        result = a / b
    except ZeroDivisionError:
        print(f"  Error: Cannot divide {a} by zero!")
        return None
    except TypeError:
        print(f"  Error: Invalid types ({type(a).__name__}, {type(b).__name__})")
        return None
    else:
        print(f"  Division successful: {a} / {b} = {result}")
        return result
    finally:
        print(f"  Cleanup: Division operation completed")

print("Testing safe_divide:")
print()

print("Case 1: Valid division")
result = safe_divide(10, 2)
print(f"Result: {result}")
print()

print("Case 2: Division by zero")
result = safe_divide(10, 0)
print(f"Result: {result}")
print()

print("Case 3: Invalid types")
result = safe_divide(10, "a")
print(f"Result: {result}")

print()

# ============================================
# SECTION 7: RAISING EXCEPTIONS
# ============================================

print("=== SECTION 7: RAISING EXCEPTIONS ===\n")

# C#: throw new ArgumentException("Age cannot be negative");
# Python: raise ValueError("Age cannot be negative")

def set_age(age):
    """Set age with validation."""
    if age < 0:
        raise ValueError("Age cannot be negative!")
    if age > 150:
        raise ValueError("Age seems unrealistic!")
    print(f"  Age set to {age}")
    return age

print("Testing set_age with validation:")
try:
    set_age(25)
except ValueError as e:
    print(f"  Error: {e}")

print()

try:
    set_age(-5)
except ValueError as e:
    print(f"  Error: {e}")

print()

try:
    set_age(200)
except ValueError as e:
    print(f"  Error: {e}")

print()

# Re-raising exceptions
print("Re-raising exceptions:")
try:
    result = 10 / 0
except ZeroDivisionError:
    print("  Logging error before re-raising...")
    raise  # Re-raise the same exception

print()

# ============================================
# SECTION 8: CUSTOM EXCEPTIONS
# ============================================

print("=== SECTION 8: CUSTOM EXCEPTIONS ===\n")

# C#:
# public class InvalidAgeException : Exception {
#     public InvalidAgeException(string message) : base(message) { }
# }

# Python:
class InvalidAgeError(Exception):
    """Custom exception for invalid age."""
    pass

class NegativeValueError(Exception):
    """Custom exception for negative values."""
    pass

# Use custom exception
def validate_age(age):
    """Validate age with custom exception."""
    if age < 0:
        raise InvalidAgeError("Age cannot be negative!")
    if age > 150:
        raise InvalidAgeError("Age must be 150 or less!")
    print(f"  Age {age} is valid")

print("Testing custom exception:")
try:
    validate_age(30)
except InvalidAgeError as e:
    print(f"  Error: {e}")

print()

try:
    validate_age(-5)
except InvalidAgeError as e:
    print(f"  Invalid age error: {e}")

print()

# Custom exception with additional data
class ValidationError(Exception):
    """Custom validation error with field information."""

    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

def validate_user(name, age):
    """Validate user data."""
    if not name or not name.strip():
        raise ValidationError("name", "Name cannot be empty")
    if age < 0:
        raise ValidationError("age", "Age cannot be negative")
    if age > 150:
        raise ValidationError("age", "Age must be 150 or less")
    print(f"  User '{name}' (age {age}) is valid")

print("Testing ValidationError:")
try:
    validate_user("Alice", 30)
except ValidationError as e:
    print(f"  Error: {e}")

print()

try:
    validate_user("", 30)
except ValidationError as e:
    print(f"  Validation failed on field '{e.field}': {e.message}")

print()

# ============================================
# SECTION 9: COMMON EXCEPTION TYPES
# ============================================

print("=== SECTION 9: COMMON EXCEPTION TYPES ===\n")

print("Common Python exceptions:")

# ValueError - Invalid value
print("\n1. ValueError - Invalid value:")
try:
    num = int("abc")
except ValueError as e:
    print(f"   {e}")

# TypeError - Wrong type
print("\n2. TypeError - Wrong type:")
try:
    result = "5" + 5  # Can't add string and int
except TypeError as e:
    print(f"   {e}")

# ZeroDivisionError - Division by zero
print("\n3. ZeroDivisionError - Division by zero:")
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"   {e}")

# IndexError - List index out of range
print("\n4. IndexError - Index out of range:")
try:
    numbers = [1, 2, 3]
    num = numbers[10]
except IndexError as e:
    print(f"   {e}")

# KeyError - Dictionary key not found
print("\n5. KeyError - Key not found:")
try:
    person = {"name": "Alice"}
    age = person["age"]
except KeyError as e:
    print(f"   {e}")

# FileNotFoundError - File doesn't exist
print("\n6. FileNotFoundError - File doesn't exist:")
try:
    with open("nonexistent.txt", "r") as file:
        content = file.read()
except FileNotFoundError as e:
    print(f"   {e}")

# AttributeError - Attribute doesn't exist
print("\n7. AttributeError - Attribute doesn't exist:")
try:
    num = 5
    num.append(10)  # int doesn't have append
except AttributeError as e:
    print(f"   {e}")

print()

# ============================================
# SECTION 10: PRACTICAL EXAMPLES
# ============================================

print("=== SECTION 10: PRACTICAL EXAMPLES ===\n")

# Example 1: Safe input
print("Example 1: Safe integer input:")

def get_integer(prompt, min_val=None, max_val=None):
    """Get valid integer from user with retry."""
    while True:
        try:
            value = int(input(prompt))
            if min_val is not None and value < min_val:
                print(f"  Error: Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"  Error: Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("  Error: Please enter a valid integer")

# Simulate input (using direct value for demo)
def get_integer_demo(value):
    """Demo version that doesn't require input."""
    try:
        result = int(value)
        if result < 0:
            raise ValueError("Value must be positive")
        return result
    except ValueError as e:
        print(f"  Error: {e}")
        return None

print("  Testing with '25':", get_integer_demo("25"))
print("  Testing with 'abc':", get_integer_demo("abc"))
print("  Testing with '-5':", get_integer_demo("-5"))

print()

# Example 2: Safe division calculator
print("Example 2: Safe division calculator:")

def safe_calculator(a, b, operation):
    """Perform safe mathematical operations."""
    try:
        if operation == "divide":
            return a / b
        elif operation == "multiply":
            return a * b
        elif operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except ZeroDivisionError:
        print(f"  Error: Cannot divide {a} by zero")
        return None
    except TypeError as e:
        print(f"  Error: Invalid types - {e}")
        return None
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return None

print(f"  10 / 2 = {safe_calculator(10, 2, 'divide')}")
print(f"  10 / 0 = {safe_calculator(10, 0, 'divide')}")
print(f"  10 + 'a' = {safe_calculator(10, 'a', 'add')}")

print()

# Example 3: Retry logic
print("Example 3: Retry logic:")

def fetch_data_with_retry(url, max_retries=3):
    """Simulate API call with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}...")
            # Simulate network error on first 2 attempts
            if attempt < 2:
                raise ConnectionError("Network error")
            # Success on 3rd attempt
            return {"data": "Success!", "status": 200}
        except ConnectionError as e:
            print(f"  Failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying...")
            else:
                print(f"  Max retries reached!")
                raise

result = fetch_data_with_retry("https://api.example.com")
print(f"  Result: {result}")

print()

# Example 4: File reading with fallback
print("Example 4: File reading with fallback:")

def read_config(filename, default_config=None):
    """Read config file with fallback to default."""
    try:
        with open(filename, "r") as file:
            print(f"  Config loaded from '{filename}'")
            return file.read()
    except FileNotFoundError:
        print(f"  Config file '{filename}' not found")
        if default_config:
            print(f"  Using default config")
            return default_config
        raise

default = {"theme": "dark", "lang": "en"}
config = read_config("nonexistent_config.txt", default)
print(f"  Config: {config}")

print()

# Example 5: Validation class
print("Example 5: User validation class:")

class User:
    """User class with validation."""

    def __init__(self, name, age, email):
        """Initialize user with validation."""
        self.name = self._validate_name(name)
        self.age = self._validate_age(age)
        self.email = self._validate_email(email)

    def _validate_name(self, name):
        """Validate name."""
        if not name or not name.strip():
            raise ValueError("Name cannot be empty")
        return name.strip()

    def _validate_age(self, age):
        """Validate age."""
        if not isinstance(age, int):
            raise TypeError("Age must be an integer")
        if age < 0 or age > 150:
            raise ValueError("Age must be between 0 and 150")
        return age

    def _validate_email(self, email):
        """Validate email."""
        if "@" not in email:
            raise ValueError("Invalid email format")
        return email

    def __str__(self):
        return f"User({self.name}, {self.age}, {self.email})"

# Valid user
try:
    user1 = User("Alice", 30, "alice@example.com")
    print(f"  Created: {user1}")
except (ValueError, TypeError) as e:
    print(f"  Error: {e}")

# Invalid name
try:
    user2 = User("", 30, "alice@example.com")
except ValueError as e:
    print(f"  Error creating user: {e}")

# Invalid age
try:
    user3 = User("Bob", -5, "bob@example.com")
except ValueError as e:
    print(f"  Error creating user: {e}")

# Invalid email
try:
    user4 = User("Charlie", 25, "invalid-email")
except ValueError as e:
    print(f"  Error creating user: {e}")

print()

# ============================================
# SECTION 11: BEST PRACTICES
# ============================================

print("=== SECTION 11: BEST PRACTICES ===\n")

print("✅ DO THIS:")
print()
print("1. Be specific about exceptions:")
print("   try:")
print("       num = int(input())")
print("   except ValueError:")
print("       print('Invalid number')")
print()
print("2. Use finally for cleanup:")
print("   try:")
print("       file = open('file.txt')")
print("   finally:")
print("       file.close()")
print()
print("3. Provide helpful error messages:")
print("   raise ValueError(f'Age cannot be negative (got {age})')")
print()

print("❌ DON'T DO THIS:")
print()
print("1. Don't catch all exceptions without good reason:")
print("   try:")
print("       something()")
print("   except:  # Too broad!")
print("       pass")
print()
print("2. Don't silently ignore errors:")
print("   try:")
print("       important_operation()")
print("   except Exception:")
print("       pass  # Bad! Hides errors")
print()
print("3. Don't use exceptions for control flow:")
print("   # Bad:")
print("   try:")
print("       value = my_dict[key]")
print("   except KeyError:")
print("       value = default")
print()
print("   # Better:")
print("   value = my_dict.get(key, default)")

print()

# ============================================
# SUMMARY
# ============================================

print("=== SUMMARY ===\n")

summary = """
Error Handling for .NET Developers:

BASIC SYNTAX:
  C#:
    try {
        int num = int.Parse(input);
    }
    catch (FormatException ex) {
        Console.WriteLine("Invalid number");
    }

  Python:
    try:
        num = int(input)
    except ValueError:
        print("Invalid number")

MULTIPLE EXCEPTIONS:
  Method 1: Multiple except blocks
    except ValueError:
        ...
    except TypeError:
        ...

  Method 2: Single except
    except (ValueError, TypeError) as e:
        ...

  Method 3: Catch all (use carefully!)
    except Exception as e:
        ...

EXCEPTION DETAILS:
  except ValueError as e:
      print(type(e))      # Exception type
      print(e)            # Error message
      print(e.args)       # Error arguments

ELSE CLAUSE (Python-specific):
  try:
      ...
  except ValueError:
      ...
  else:
      # Runs if NO exception occurred
      ...

FINALLY CLAUSE:
  C#: finally { ... }
  Python: finally:

  - ALWAYS runs
  - Used for cleanup (close files, etc.)

RAISING EXCEPTIONS:
  C#: throw new ArgumentException("message");
  Python: raise ValueError("message")

  Re-raise: raise

CUSTOM EXCEPTIONS:
  C#:
    public class CustomException : Exception { }

  Python:
    class CustomException(Exception):
        pass

  With data:
    class ValidationError(Exception):
        def __init__(self, field, message):
            self.field = field
            self.message = message
            super().__init__(f"{field}: {message}")

COMMON EXCEPTIONS:
  ValueError       - Invalid value
  TypeError        - Wrong type
  ZeroDivisionError - Division by zero
  IndexError       - Index out of range
  KeyError         - Dictionary key not found
  FileNotFoundError - File doesn't exist
  AttributeError   - Attribute doesn't exist

BEST PRACTICES:
  ✅ Be specific about exceptions
  ✅ Use finally for cleanup
  ✅ Provide helpful error messages
  ✅ Document what exceptions can be raised

  ❌ Don't catch all exceptions
  ❌ Don't silently ignore errors
  ❌ Don't use exceptions for control flow

C# → Python Quick Reference:
  try { }                  → try:
  catch (Exception ex) { } → except Exception as e:
  finally { }              → finally:
  throw new Exception()    → raise Exception()
  class MyEx : Exception   → class MyEx(Exception):
"""

print(summary)

print("="*60)
print("🎉 Module 1 Complete! You've learned all Python basics!")
print("="*60)

# Lesson 1.10: Error Handling (Exceptions)

## 🎯 What You'll Learn
- Understanding exceptions
- try/except blocks
- Multiple exception types
- finally and else clauses
- Raising exceptions
- Creating custom exceptions

---

## What are Exceptions?

**Exceptions** are errors that occur during program execution.

**Common exceptions:**
- `ValueError` → Invalid value
- `TypeError` → Wrong type
- `ZeroDivisionError` → Division by zero
- `FileNotFoundError` → File doesn't exist
- `KeyError` → Dictionary key not found
- `IndexError` → List index out of range

---

## Basic try/except

**Without error handling:**
```python
num = int(input("Enter a number: "))
# If user enters "abc", program crashes!
```

**With error handling:**
```python
try:
    num = int(input("Enter a number: "))
    print(f"You entered: {num}")
except ValueError:
    print("That's not a valid number!")
```

**C# Comparison:**
```csharp
// C#
try
{
    int num = int.Parse(Console.ReadLine());
}
catch (FormatException ex)
{
    Console.WriteLine("That's not a valid number!");
}

// Python
try:
    num = int(input("Enter a number: "))
except ValueError:
    print("That's not a valid number!")
```

---

## Catching Multiple Exceptions

### Method 1: Multiple except Blocks

```python
try:
    num = int(input("Enter a number: "))
    result = 10 / num
    print(f"Result: {result}")
except ValueError:
    print("Invalid number!")
except ZeroDivisionError:
    print("Cannot divide by zero!")
```

### Method 2: Single except for Multiple Types

```python
try:
    num = int(input("Enter a number: "))
    result = 10 / num
except (ValueError, ZeroDivisionError) as e:
    print(f"Error: {e}")
```

### Method 3: Catch All Exceptions

```python
try:
    num = int(input("Enter a number: "))
    result = 10 / num
except Exception as e:
    print(f"An error occurred: {e}")
```

**Warning:** Catching all exceptions can hide bugs! Be specific when possible.

---

## Getting Exception Details

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error type: {type(e)}")
    print(f"Error message: {e}")
    print(f"Error args: {e.args}")

# Output:
# Error type: <class 'ZeroDivisionError'>
# Error message: division by zero
# Error args: ('division by zero',)
```

---

## else Clause

Runs if NO exception occurred

```python
try:
    num = int(input("Enter a number: "))
except ValueError:
    print("Invalid number!")
else:
    print(f"Success! You entered: {num}")
    print("This runs only if no exception occurred")
```

**Flow:**
1. Try block runs
2. If exception → except block runs, else is skipped
3. If no exception → else block runs

---

## finally Clause

**Always** runs, whether exception occurred or not

```python
try:
    file = open("example.txt", "r")
    content = file.read()
except FileNotFoundError:
    print("File not found!")
finally:
    print("This always runs!")
    # Cleanup code goes here
```

**Common use:** Resource cleanup (close files, database connections, etc.)

```python
file = None
try:
    file = open("example.txt", "r")
    content = file.read()
except FileNotFoundError:
    print("File not found!")
finally:
    if file:
        file.close()  # Always close file
        print("File closed")
```

**Note:** Using `with` statement is better for files (auto cleanup)

---

## Complete try/except/else/finally

```python
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero!")
        return None
    except TypeError:
        print("Invalid types!")
        return None
    else:
        print("Division successful!")
        return result
    finally:
        print("Function completed")

print(divide(10, 2))    # Works
print(divide(10, 0))    # Zero division
print(divide(10, "a"))  # Type error
```

**Output:**
```
Division successful!
Function completed
5.0
Cannot divide by zero!
Function completed
None
Invalid types!
Function completed
None
```

---

## Raising Exceptions

### Raise Built-in Exception

```python
def set_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative!")
    if age > 150:
        raise ValueError("Age is too high!")
    print(f"Age set to {age}")

set_age(25)    # Age set to 25
set_age(-5)    # ValueError: Age cannot be negative!
```

### Re-raising Exceptions

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Logging error...")
    raise  # Re-raise the same exception
```

**C# Comparison:**
```csharp
// C#
if (age < 0)
    throw new ArgumentException("Age cannot be negative!");

// Python
if age < 0:
    raise ValueError("Age cannot be negative!")
```

---

## Custom Exceptions

Create your own exception types

```python
# Define custom exception
class InvalidAgeError(Exception):
    """Custom exception for invalid age"""
    pass

# Use it
def set_age(age):
    if age < 0:
        raise InvalidAgeError("Age cannot be negative!")
    if age > 150:
        raise InvalidAgeError("Age seems unrealistic!")
    print(f"Age set to {age}")

try:
    set_age(-5)
except InvalidAgeError as e:
    print(f"Invalid age error: {e}")
```

### Custom Exception with Additional Data

```python
class ValidationError(Exception):
    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

def validate_user(name, age):
    if not name:
        raise ValidationError("name", "Name cannot be empty")
    if age < 0:
        raise ValidationError("age", "Age cannot be negative")

try:
    validate_user("", 25)
except ValidationError as e:
    print(f"Validation failed on {e.field}: {e.message}")
```

---

## Common Exception Types

```python
# ValueError - Invalid value
try:
    num = int("abc")
except ValueError:
    print("Invalid number format")

# TypeError - Wrong type
try:
    result = "5" + 5  # Can't add string and int
except TypeError:
    print("Type mismatch")

# ZeroDivisionError - Division by zero
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# IndexError - List index out of range
try:
    numbers = [1, 2, 3]
    print(numbers[10])
except IndexError:
    print("Index out of range")

# KeyError - Dictionary key not found
try:
    person = {"name": "Alice"}
    print(person["age"])
except KeyError:
    print("Key not found")

# FileNotFoundError - File doesn't exist
try:
    with open("nonexistent.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("File not found")

# AttributeError - Attribute doesn't exist
try:
    num = 5
    num.append(10)  # int doesn't have append
except AttributeError:
    print("Attribute doesn't exist")
```

---

## Practical Examples

### Example 1: Safe Input

```python
def get_integer(prompt):
    """Get valid integer from user"""
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid input! Please enter a number.")

age = get_integer("Enter your age: ")
print(f"Age: {age}")
```

### Example 2: Safe Division

```python
def safe_divide(a, b):
    """Divide with error handling"""
    try:
        return a / b
    except ZeroDivisionError:
        print("Error: Division by zero!")
        return None
    except TypeError:
        print("Error: Invalid types!")
        return None

print(safe_divide(10, 2))    # 5.0
print(safe_divide(10, 0))    # Error: Division by zero!
print(safe_divide(10, "a"))  # Error: Invalid types!
```

### Example 3: File Reading with Error Handling

```python
def read_file(filename):
    """Read file with error handling"""
    try:
        with open(filename, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except PermissionError:
        print(f"Error: No permission to read '{filename}'")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

content = read_file("example.txt")
if content:
    print(content)
```

### Example 4: Validate User Input

```python
class User:
    def __init__(self, name, age, email):
        self.name = self._validate_name(name)
        self.age = self._validate_age(age)
        self.email = self._validate_email(email)

    def _validate_name(self, name):
        if not name or not name.strip():
            raise ValueError("Name cannot be empty")
        return name.strip()

    def _validate_age(self, age):
        if not isinstance(age, int):
            raise TypeError("Age must be an integer")
        if age < 0 or age > 150:
            raise ValueError("Age must be between 0 and 150")
        return age

    def _validate_email(self, email):
        if "@" not in email:
            raise ValueError("Invalid email format")
        return email

# Use it
try:
    user = User("Alice", 30, "alice@example.com")
    print("User created successfully!")
except (ValueError, TypeError) as e:
    print(f"Validation error: {e}")

try:
    user = User("", 30, "alice@example.com")  # Empty name
except ValueError as e:
    print(f"Error: {e}")
```

### Example 5: API Request with Retry

```python
import time

def fetch_data(url, max_retries=3):
    """Fetch data with retry logic"""
    for attempt in range(max_retries):
        try:
            # Simulating API call
            if attempt < 2:
                raise ConnectionError("Network error")
            return {"data": "Success!"}
        except ConnectionError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 1 second...")
                time.sleep(1)
            else:
                print("Max retries reached!")
                raise

data = fetch_data("https://api.example.com")
print(data)
```

---

## Best Practices

### ✅ Do This:

```python
# Be specific about exceptions
try:
    num = int(input("Enter number: "))
except ValueError:
    print("Invalid number")

# Use finally for cleanup
file = None
try:
    file = open("file.txt", "r")
    # Do something
finally:
    if file:
        file.close()

# Provide helpful error messages
if age < 0:
    raise ValueError(f"Age cannot be negative (got {age})")
```

### ❌ Don't Do This:

```python
# Don't catch all exceptions
try:
    something()
except:  # Too broad!
    pass

# Don't silently ignore errors
try:
    important_operation()
except Exception:
    pass  # Bad! Hides errors

# Don't use exceptions for control flow
try:
    value = my_dict[key]
except KeyError:
    value = default

# Better:
value = my_dict.get(key, default)
```

---

## 💡 Key Takeaways

1. **try/except** → Handle errors gracefully
2. **Be specific** → Catch specific exception types
3. **else** → Runs if no exception
4. **finally** → Always runs (cleanup)
5. **raise** → Throw exceptions
6. **Custom exceptions** → Inherit from Exception
7. **Don't catch all** → Be specific about what to catch
8. **Provide context** → Include helpful error messages

---

## ✏️ Practice Exercise

Create `error_handling_practice.py`:

```python
# 1. Basic try/except
try:
    num = int(input("Enter a number: "))
    result = 100 / num
    print(f"Result: {result}")
except ValueError:
    print("Invalid number!")
except ZeroDivisionError:
    print("Cannot divide by zero!")

# 2. try/except/else/finally
try:
    num = int(input("Enter a number: "))
except ValueError:
    print("Invalid input!")
else:
    print(f"You entered: {num}")
finally:
    print("Operation completed")

# 3. Custom exception
class NegativeNumberError(Exception):
    pass

def check_positive(num):
    if num < 0:
        raise NegativeNumberError("Number must be positive!")
    return num ** 2

try:
    result = check_positive(-5)
except NegativeNumberError as e:
    print(f"Error: {e}")

# 4. Safe input function
def get_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid number!")

age = get_int("Enter your age: ")
print(f"Age: {age}")

# 5. Multiple operations with error handling
def process_data(data):
    try:
        # Validate
        if not data:
            raise ValueError("Data cannot be empty")

        # Process
        result = [int(x) for x in data.split(",")]

        # Calculate
        total = sum(result)
        average = total / len(result)

        return {"total": total, "average": average}

    except ValueError as e:
        print(f"Value error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

result = process_data("10,20,30")
print(result)  # {'total': 60, 'average': 20.0}
```

**Run it:** `python error_handling_practice.py`

---

## 🤔 Quick Quiz

1. What's the difference between `except` and `finally`?
   <details>
   <summary>Answer</summary>

   - `except` runs only if exception occurs
   - `finally` always runs, whether exception occurs or not
   </details>

2. How do you raise an exception?
   <details>
   <summary>Answer</summary>

   `raise ExceptionType("message")`
   </details>

3. What's wrong with `except:` without specifying exception type?
   <details>
   <summary>Answer</summary>

   Too broad - catches ALL exceptions, even ones you don't expect. Can hide bugs!
   </details>

4. When does the `else` block run in try/except?
   <details>
   <summary>Answer</summary>

   Only when NO exception occurs in the try block
   </details>

5. How do you create a custom exception?
   <details>
   <summary>Answer</summary>

   ```python
   class MyException(Exception):
       pass
   ```
   </details>

---

## 🎉 Module 1 Complete!

Congratulations! You've finished all Python basics lessons!

**What you learned:**
- ✅ Variables and types
- ✅ Operators
- ✅ Control flow
- ✅ Functions
- ✅ Lists and tuples
- ✅ Dictionaries and sets
- ✅ Comprehensions
- ✅ Classes and OOP
- ✅ File I/O
- ✅ Error handling

**Next Steps:**
1. Take the quiz: `../../quizzes/quiz_module_1.md`
2. Do the lab: `../../labs/lab_module_1.md`
3. Update progress: `../../PROGRESS.md`
4. Ready for Module 2? Ask Claude!

**Great job! You're ready to build neural networks! 🚀**

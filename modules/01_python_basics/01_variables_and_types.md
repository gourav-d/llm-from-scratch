# Lesson 1.1: Variables and Data Types

## 🎯 What You'll Learn
- How Python variables work vs C#
- Basic data types in Python
- Type conversion

---

## Python vs C#: Key Difference

### In C# (What you know):
```csharp
// You must declare the type
int age = 25;
string name = "John";
double price = 19.99;
bool isActive = true;
```

### In Python (New way):
```python
# No type declaration needed!
age = 25
name = "John"
price = 19.99
is_active = True
```

**Key Point:** Python is **dynamically typed** - you don't declare types. Python figures it out automatically!

Think of it like using `var` in C#, but you don't even write `var`!

---

## Basic Data Types

### 1. Integers (int)
Whole numbers, like C#'s `int` or `long`

```python
# Creating integers
count = 10
negative = -5
large_number = 1000000

# Python can handle VERY large integers (no limit like int32/int64)
huge = 999999999999999999999999999
```

**Line-by-line explanation:**
- `count = 10` → Creates a variable named 'count' and assigns value 10
- No semicolon needed in Python!
- Python integers can be any size (unlike C#'s int which maxes at ~2 billion)

### 2. Floats (float)
Decimal numbers, like C#'s `double` or `float`

```python
# Creating floats
price = 19.99
temperature = -3.5
pi = 3.14159

# Scientific notation
speed_of_light = 3.0e8  # 3.0 × 10^8
```

**Explanation:**
- `price = 19.99` → Decimal number (float)
- `3.0e8` → Scientific notation, same as C#
- Python's float is like C#'s double (64-bit)

### 3. Strings (str)
Text, like C#'s `string`

```python
# Single or double quotes - both work!
name = "Alice"
message = 'Hello, World!'

# Multi-line strings (like C#'s @"..." or """...""")
long_text = """This is a
multi-line
string"""

# String concatenation
full_name = "John" + " " + "Doe"  # "John Doe"

# f-strings (like C#'s $"...")
age = 30
greeting = f"I am {age} years old"  # "I am 30 years old"
```

**Line-by-line explanation:**
- `"Alice"` or `'Alice'` → Both work! (C# only uses double quotes)
- `"""..."""` → Triple quotes for multi-line strings
- `f"I am {age} years old"` → f-string = C#'s string interpolation `$"I am {age} years old"`

**C# Comparison:**
```csharp
// C#
string greeting = $"I am {age} years old";

// Python
greeting = f"I am {age} years old"
```

### 4. Booleans (bool)
True or False, like C#'s `bool`

```python
# Booleans in Python start with capital letter!
is_valid = True
is_empty = False

# Common boolean expressions
is_greater = 10 > 5      # True
is_equal = 10 == 10      # True
is_not_equal = 10 != 5   # True
```

**Important Difference:**
- C#: `true`, `false` (lowercase)
- Python: `True`, `False` (capital T and F!)

---

## Checking Types

### In C# you might use `GetType()`:
```csharp
var x = 10;
Console.WriteLine(x.GetType());  // System.Int32
```

### In Python, use `type()`:
```python
x = 10
print(type(x))  # <class 'int'>

y = 3.14
print(type(y))  # <class 'float'>

z = "Hello"
print(type(z))  # <class 'str'>
```

**Explanation:**
- `type(x)` → Returns the type of variable x
- `print()` → Like C#'s `Console.WriteLine()`

---

## Type Conversion (Casting)

### In C# you might do:
```csharp
int x = 10;
double y = (double)x;  // Explicit cast
string s = x.ToString();
```

### In Python:
```python
# Converting to int
x = int("42")        # "42" → 42
y = int(3.14)        # 3.14 → 3 (truncates!)

# Converting to float
a = float("3.14")    # "3.14" → 3.14
b = float(10)        # 10 → 10.0

# Converting to string
s = str(42)          # 42 → "42"
t = str(3.14)        # 3.14 → "3.14"

# Converting to bool
b1 = bool(1)         # 1 → True
b2 = bool(0)         # 0 → False
b3 = bool("")        # Empty string → False
b4 = bool("text")    # Non-empty string → True
```

**Line-by-line explanation:**
- `int("42")` → Converts string "42" to integer 42
- `int(3.14)` → Converts float to int by removing decimal (not rounding!)
- `str(42)` → Converts number to string
- Empty values (0, "", None) are False, everything else is True

---

## Variable Naming Rules

### C# Convention:
```csharp
int userAge = 25;         // camelCase
string UserName = "Bob";  // PascalCase for properties
```

### Python Convention:
```python
user_age = 25       # snake_case (underscore!)
user_name = "Bob"   # snake_case

# Constants (values that don't change)
MAX_SIZE = 100      # ALL_CAPS for constants
PI = 3.14159
```

**Key Difference:**
- C# uses camelCase: `userName`
- Python uses snake_case: `user_name`

**Rules (same as C#):**
- Must start with letter or underscore
- Can contain letters, numbers, underscores
- Case sensitive (`age` ≠ `Age`)
- Can't use reserved keywords

---

## None (Python's null)

In C#, you have `null`. In Python, it's `None`:

```python
# None is like C#'s null
value = None

# Checking for None
if value is None:
    print("Value is None")

# C# equivalent:
# if (value == null)
#     Console.WriteLine("Value is null");
```

**Important:**
- Use `is None`, not `== None` (Python convention)
- `None` is capitalized

---

## Quick Reference Table

| C# Type | Python Type | Example |
|---------|-------------|---------|
| `int` | `int` | `age = 25` |
| `double`/`float` | `float` | `price = 19.99` |
| `string` | `str` | `name = "Bob"` |
| `bool` (`true`/`false`) | `bool` (`True`/`False`) | `is_valid = True` |
| `null` | `None` | `value = None` |
| `var` | (just write it) | `x = 10` |

---

## 💡 Key Takeaways

1. **No type declarations** → Python figures it out
2. **No semicolons** → Lines end automatically
3. **Use snake_case** → Not camelCase
4. **True/False capitalized** → Not true/false
5. **None instead of null** → Use `is None`
6. **f-strings for formatting** → Like C#'s $"..."

---

## ✏️ Practice Exercise

Create a file called `practice.py` and try this:

```python
# Your information
first_name = "Your Name"
last_name = "Your Last Name"
age = 25
height = 5.9
is_student = True

# Print everything
print(f"Name: {first_name} {last_name}")
print(f"Age: {age}")
print(f"Height: {height}")
print(f"Student: {is_student}")

# Check types
print(f"Type of age: {type(age)}")
print(f"Type of height: {type(height)}")
```

**How to run:**
1. Save as `practice.py`
2. Open terminal
3. Type: `python practice.py`

---

## 🤔 Quick Quiz

1. What's the Python equivalent of C#'s `string name = "Alice";`?
   <details>
   <summary>Answer</summary>

   ```python
   name = "Alice"
   ```
   No type declaration needed!
   </details>

2. How do you write `true` and `false` in Python?
   <details>
   <summary>Answer</summary>

   `True` and `False` (capitalized!)
   </details>

3. What's the output of `print(type(3.14))`?
   <details>
   <summary>Answer</summary>

   `<class 'float'>`
   </details>

4. Convert the string "123" to an integer:
   <details>
   <summary>Answer</summary>

   ```python
   num = int("123")
   ```
   </details>

---

**Next Lesson:** [02_operators.md](02_operators.md) - Learn about operators and expressions!

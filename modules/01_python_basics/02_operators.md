# Lesson 1.2: Operators and Expressions

## 🎯 What You'll Learn
- Arithmetic operators
- Comparison operators
- Logical operators
- String operations
- Operator precedence

---

## Arithmetic Operators

### Basic Math Operations

```python
# Addition
result = 10 + 5      # 15

# Subtraction
result = 10 - 5      # 5

# Multiplication
result = 10 * 5      # 50

# Division (always returns float!)
result = 10 / 3      # 3.333... (float)

# Integer Division (floor division)
result = 10 // 3     # 3 (int, drops decimal)

# Modulus (remainder)
result = 10 % 3      # 1 (10 ÷ 3 = 3 remainder 1)

# Exponentiation (power)
result = 2 ** 3      # 8 (2³)
```

**Line-by-line explanation:**
- `/` → Division, ALWAYS returns float (even 10/5 = 2.0)
- `//` → Integer division (C# equivalent: `(int)(10/3)`)
- `%` → Modulus, same as C#
- `**` → Power (C# uses `Math.Pow(2, 3)`)

**C# Comparison:**
```csharp
// C#
int result = 10 / 3;        // 3 (integer division)
double result = 10.0 / 3;   // 3.333... (float division)
double power = Math.Pow(2, 3);  // 8

// Python
result = 10 / 3       # 3.333... (always float!)
result = 10 // 3      # 3 (integer division)
result = 2 ** 3       # 8 (power operator!)
```

### Compound Assignment Operators

Same as C#!

```python
x = 10

x += 5    # x = x + 5  → 15
x -= 3    # x = x - 3  → 12
x *= 2    # x = x * 2  → 24
x /= 4    # x = x / 4  → 6.0
x //= 2   # x = x // 2 → 3
x **= 2   # x = x ** 2 → 9
```

**Note:** Python does NOT have `++` or `--` operators!

```csharp
// C# - This works
x++;
x--;

// Python - This does NOT work! ❌
x++   # ERROR!
x--   # ERROR!

// Python - Use this instead ✅
x += 1
x -= 1
```

---

## Comparison Operators

Used to compare values, returns `True` or `False`

```python
x = 10
y = 20

# Equal to
result = x == y      # False

# Not equal to
result = x != y      # True

# Greater than
result = x > y       # False

# Less than
result = x < y       # True

# Greater than or equal
result = x >= 10     # True

# Less than or equal
result = y <= 20     # True
```

**Same as C#!** Nothing new here.

### Special Comparison: Identity (`is` vs `==`)

Python has a special operator `is` to check if two variables point to the same object:

```python
# == checks if VALUES are equal
# is checks if they are the SAME OBJECT

a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)    # True (same values)
print(a is b)    # False (different objects)
print(a is c)    # True (same object!)

# For None, always use 'is'
value = None
if value is None:     # ✅ Correct
    print("It's None")

if value == None:     # ❌ Works but not Pythonic
    print("It's None")
```

**Explanation:**
- `==` → Compares values (like C#'s `==`)
- `is` → Compares if they're the same object (like C#'s `ReferenceEquals()`)
- Always use `is None`, not `== None`

---

## Logical Operators

### C# vs Python

| C# | Python | Meaning |
|---------|--------|---------|
| `&&` | `and` | Logical AND |
| `\|\|` | `or` | Logical OR |
| `!` | `not` | Logical NOT |

```python
# AND - both must be True
result = True and True      # True
result = True and False     # False

# OR - at least one must be True
result = True or False      # True
result = False or False     # False

# NOT - reverses boolean
result = not True           # False
result = not False          # True
```

**C# Comparison:**
```csharp
// C#
bool result = true && false;   // AND
bool result = true || false;   // OR
bool result = !true;           // NOT

// Python
result = True and False   # AND
result = True or False    # OR
result = not True         # NOT
```

### Combining Conditions

```python
age = 25
has_license = True

# Can drive if age >= 18 AND has license
can_drive = age >= 18 and has_license    # True

# Discount if age < 18 OR age >= 65
gets_discount = age < 18 or age >= 65    # False

# Not allowed if under 18
not_allowed = not (age >= 18)            # False
```

**Line-by-line explanation:**
- `age >= 18 and has_license` → Both conditions must be True
- `age < 18 or age >= 65` → At least one must be True
- `not (age >= 18)` → Reverses the result

---

## String Operations

### Concatenation (Joining Strings)

```python
# Using + operator
first = "Hello"
last = "World"
full = first + " " + last    # "Hello World"

# Using f-strings (better!)
name = "Alice"
age = 30
message = f"My name is {name} and I am {age}"
# "My name is Alice and I am 30"

# Multiplying strings (repeat!)
laugh = "ha" * 3    # "hahaha"
line = "-" * 10     # "----------"
```

**Line-by-line explanation:**
- `"Hello" + " " + "World"` → Joins strings together
- `f"...{variable}..."` → Inserts variable value (like C#'s $"...")
- `"ha" * 3` → Repeats string 3 times (Python-specific!)

### String Methods

```python
text = "Hello World"

# Length (like C#'s .Length)
length = len(text)           # 11

# Convert to uppercase
upper = text.upper()         # "HELLO WORLD"

# Convert to lowercase
lower = text.lower()         # "hello world"

# Replace
new_text = text.replace("World", "Python")  # "Hello Python"

# Split into list
words = text.split()         # ["Hello", "World"]

# Check if contains
has_hello = "Hello" in text  # True
has_bye = "Bye" in text      # False

# Starts with / Ends with
starts = text.startswith("Hello")   # True
ends = text.endswith("World")       # True
```

**C# Comparison:**
```csharp
// C#
string text = "Hello World";
int length = text.Length;              // Property
string upper = text.ToUpper();         // Method
bool contains = text.Contains("Hello");

// Python
text = "Hello World"
length = len(text)                     # Function!
upper = text.upper()                   # Method
contains = "Hello" in text             # Operator!
```

**Key Difference:**
- C#: `text.Length` (property)
- Python: `len(text)` (function)

---

## Membership Operators

Check if something is in a collection

```python
# in - checks if item exists
fruits = ["apple", "banana", "orange"]
has_apple = "apple" in fruits        # True
has_grape = "grape" in fruits        # False

# not in - checks if item doesn't exist
no_grape = "grape" not in fruits     # True

# Works with strings too!
has_hello = "Hello" in "Hello World"    # True
```

**C# Comparison:**
```csharp
// C#
var fruits = new List<string> {"apple", "banana"};
bool hasApple = fruits.Contains("apple");

// Python
fruits = ["apple", "banana"]
has_apple = "apple" in fruits    # More readable!
```

---

## Operator Precedence

Which operation happens first?

**Order (highest to lowest):**
1. `**` (Exponentiation)
2. `*`, `/`, `//`, `%` (Multiplication, Division)
3. `+`, `-` (Addition, Subtraction)
4. `<`, `>`, `<=`, `>=`, `==`, `!=` (Comparison)
5. `not` (Logical NOT)
6. `and` (Logical AND)
7. `or` (Logical OR)

```python
# Example
result = 2 + 3 * 4      # 14 (not 20!)
# Because: 3 * 4 = 12, then 2 + 12 = 14

# Use parentheses to be clear
result = (2 + 3) * 4    # 20

# Complex example
x = 10
y = 5
result = x > 5 and y < 10 or x == 0
# Step 1: x > 5 → True
# Step 2: y < 10 → True
# Step 3: True and True → True
# Step 4: True or False → True
```

**Tip:** When in doubt, use parentheses `()` to make it clear!

---

## 💡 Key Takeaways

1. **Division `/` always returns float** → Use `//` for integer division
2. **Power operator `**`** → C# uses `Math.Pow()`
3. **No `++` or `--`** → Use `x += 1` instead
4. **Logical operators are words** → `and`, `or`, `not` (not &&, ||, !)
5. **`in` operator** → Check if item exists in collection
6. **f-strings** → Best way to format strings

---

## ✏️ Practice Exercise

Create `operators_practice.py`:

```python
# Math operations
a = 15
b = 4

print(f"{a} + {b} = {a + b}")
print(f"{a} - {b} = {a - b}")
print(f"{a} * {b} = {a * b}")
print(f"{a} / {b} = {a / b}")       # Regular division
print(f"{a} // {b} = {a // b}")     # Integer division
print(f"{a} % {b} = {a % b}")       # Remainder
print(f"{a} ** {b} = {a ** b}")     # Power

# Comparison
x = 10
y = 20
print(f"{x} == {y}: {x == y}")
print(f"{x} < {y}: {x < y}")
print(f"{x} > {y}: {x > y}")

# Logical
age = 25
has_id = True
can_enter = age >= 21 and has_id
print(f"Can enter: {can_enter}")

# String operations
name = "Python"
print(f"Length: {len(name)}")
print(f"Uppercase: {name.upper()}")
print(f"Repeated: {name * 3}")
```

**Run it:** `python operators_practice.py`

---

## 🤔 Quick Quiz

1. What's the result of `10 / 3` in Python?
   <details>
   <summary>Answer</summary>

   `3.3333...` (float) - Division always returns float!
   </details>

2. What's the result of `10 // 3`?
   <details>
   <summary>Answer</summary>

   `3` (integer division)
   </details>

3. How do you write "x is greater than 5 AND y is less than 10" in Python?
   <details>
   <summary>Answer</summary>

   ```python
   x > 5 and y < 10
   ```
   </details>

4. What's the result of `"ha" * 3`?
   <details>
   <summary>Answer</summary>

   `"hahaha"` - String multiplication repeats the string!
   </details>

5. What does `"apple" in ["apple", "banana"]` return?
   <details>
   <summary>Answer</summary>

   `True` - The `in` operator checks membership
   </details>

---

**Next Lesson:** [03_control_flow.md](03_control_flow.md) - Learn about if/else and loops!

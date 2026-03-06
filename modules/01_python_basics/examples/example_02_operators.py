"""
Example 2: Operators and Expressions for .NET Developers
This file demonstrates Python operators with detailed comments and C# comparisons.
"""

# ============================================
# SECTION 1: ARITHMETIC OPERATORS
# ============================================

print("=== SECTION 1: ARITHMETIC OPERATORS ===\n")

# Basic arithmetic operations
a = 10
b = 3

# Addition (same as C#)
result = a + b
print(f"{a} + {b} = {result}")  # 13

# Subtraction (same as C#)
result = a - b
print(f"{a} - {b} = {result}")  # 7

# Multiplication (same as C#)
result = a * b
print(f"{a} * {b} = {result}")  # 30

# Division - IMPORTANT: Always returns float!
# C#: int result = 10 / 3;  → 3 (integer division)
# Python: result = 10 / 3   → 3.3333... (always float!)
result = a / b
print(f"{a} / {b} = {result}")  # 3.3333...
print(f"Type: {type(result)}")  # <class 'float'>

# Integer Division (floor division)
# C#: (int)(10 / 3)
# Python: 10 // 3
result = a // b
print(f"{a} // {b} = {result}")  # 3 (integer)
print(f"Type: {type(result)}")  # <class 'int'>

# Modulus (remainder) - same as C#
result = a % b
print(f"{a} % {b} = {result}")  # 1 (10 ÷ 3 = 3 remainder 1)

# Exponentiation (power)
# C#: Math.Pow(2, 3)
# Python: 2 ** 3
result = 2 ** 3
print(f"2 ** 3 = {result}")  # 8 (2³)

print()

# ============================================
# SECTION 2: COMPOUND ASSIGNMENT OPERATORS
# ============================================

print("=== SECTION 2: COMPOUND ASSIGNMENT OPERATORS ===\n")

# These work the same as C#!
x = 10
print(f"Starting value: x = {x}")

# Add and assign
x += 5  # x = x + 5
print(f"After x += 5: x = {x}")  # 15

# Subtract and assign
x -= 3  # x = x - 3
print(f"After x -= 3: x = {x}")  # 12

# Multiply and assign
x *= 2  # x = x * 2
print(f"After x *= 2: x = {x}")  # 24

# Divide and assign (result becomes float!)
x /= 4  # x = x / 4
print(f"After x /= 4: x = {x}")  # 6.0
print(f"Type: {type(x)}")  # <class 'float'>

# Integer divide and assign
x = 10
x //= 3  # x = x // 3
print(f"After x //= 3: x = {x}")  # 3

# Modulus and assign
x = 10
x %= 3  # x = x % 3
print(f"After x %= 3: x = {x}")  # 1

# Power and assign
x = 2
x **= 3  # x = x ** 3
print(f"After x **= 3: x = {x}")  # 8

print()

# IMPORTANT: Python does NOT have ++ or -- operators!
print("⚠️  IMPORTANT: No ++ or -- in Python!")
print("C#:     x++;")
print("Python: x += 1  (use this instead)")
print()

# ============================================
# SECTION 3: COMPARISON OPERATORS
# ============================================

print("=== SECTION 3: COMPARISON OPERATORS ===\n")

# These are exactly the same as C#!
x = 10
y = 20

# Equal to
result = x == y
print(f"{x} == {y}: {result}")  # False

# Not equal to
result = x != y
print(f"{x} != {y}: {result}")  # True

# Greater than
result = x > y
print(f"{x} > {y}: {result}")  # False

# Less than
result = x < y
print(f"{x} < {y}: {result}")  # True

# Greater than or equal
result = x >= 10
print(f"{x} >= 10: {result}")  # True

# Less than or equal
result = y <= 20
print(f"{y} <= 20: {result}")  # True

print()

# Identity operators: is vs ==
# C#: ReferenceEquals(a, b)
# Python: a is b
print("=== Identity Operators (is vs ==) ===\n")

# Creating different objects with same values
a = [1, 2, 3]
b = [1, 2, 3]
c = a

# == checks if VALUES are equal
print(f"a == b: {a == b}")  # True (same values)
print(f"a == c: {a == c}")  # True (same values)

# is checks if they are the SAME OBJECT
# Like C#'s ReferenceEquals()
print(f"a is b: {a is b}")  # False (different objects)
print(f"a is c: {a is c}")  # True (same object!)

# Special case: always use "is None", not "== None"
value = None
if value is None:  # ✅ Correct
    print("Value is None (correct way)")

if value == None:  # ❌ Works but not Pythonic
    print("Value is None (works but not preferred)")

print()

# ============================================
# SECTION 4: LOGICAL OPERATORS
# ============================================

print("=== SECTION 4: LOGICAL OPERATORS ===\n")

# C# uses: &&, ||, !
# Python uses: and, or, not

# AND - both conditions must be True
# C#: bool result = true && false;
# Python: result = True and False
result = True and True
print(f"True and True: {result}")  # True

result = True and False
print(f"True and False: {result}")  # False

# OR - at least one condition must be True
# C#: bool result = true || false;
# Python: result = True or False
result = True or False
print(f"True or False: {result}")  # True

result = False or False
print(f"False or False: {result}")  # False

# NOT - reverses boolean value
# C#: bool result = !true;
# Python: result = not True
result = not True
print(f"not True: {result}")  # False

result = not False
print(f"not False: {result}")  # True

print()

# Real-world example: Can drive?
age = 25
has_license = True

# Both conditions must be True
can_drive = age >= 18 and has_license
print(f"Age {age}, Has license: {has_license}")
print(f"Can drive: {can_drive}")  # True

# Discount if under 18 OR over 65
gets_discount = age < 18 or age >= 65
print(f"Gets discount: {gets_discount}")  # False

# Not allowed if under 18
not_allowed = not (age >= 18)
print(f"Not allowed: {not_allowed}")  # False

print()

# ============================================
# SECTION 5: STRING OPERATIONS
# ============================================

print("=== SECTION 5: STRING OPERATIONS ===\n")

# Concatenation using +
# C#: string full = first + " " + last;
# Python: full = first + " " + last
first = "Hello"
last = "World"
full = first + " " + last
print(f"Concatenation: {full}")  # Hello World

# Using f-strings (better!)
# C#: string message = $"My name is {name}";
# Python: message = f"My name is {name}"
name = "Alice"
age = 30
message = f"My name is {name} and I am {age}"
print(f"F-string: {message}")

# String multiplication (Python-specific!)
# This doesn't exist in C#
laugh = "ha" * 3
print(f"'ha' * 3 = {laugh}")  # hahaha

line = "-" * 40
print(f"'-' * 40 = {line}")

print()

# ============================================
# SECTION 6: STRING METHODS
# ============================================

print("=== SECTION 6: STRING METHODS ===\n")

text = "Hello World"

# Length
# C#: int length = text.Length;  (property)
# Python: length = len(text)     (function!)
length = len(text)
print(f"Length of '{text}': {length}")  # 11

# Uppercase
# C#: string upper = text.ToUpper();
# Python: upper = text.upper()
upper = text.upper()
print(f"Uppercase: {upper}")  # HELLO WORLD

# Lowercase
# C#: string lower = text.ToLower();
# Python: lower = text.lower()
lower = text.lower()
print(f"Lowercase: {lower}")  # hello world

# Replace
# C#: string newText = text.Replace("World", "Python");
# Python: new_text = text.replace("World", "Python")
new_text = text.replace("World", "Python")
print(f"Replace: {new_text}")  # Hello Python

# Split into list
# C#: string[] words = text.Split();
# Python: words = text.split()
words = text.split()
print(f"Split: {words}")  # ['Hello', 'World']

# Check if contains
# C#: bool hasHello = text.Contains("Hello");
# Python: has_hello = "Hello" in text
has_hello = "Hello" in text
print(f"Contains 'Hello': {has_hello}")  # True

has_bye = "Bye" in text
print(f"Contains 'Bye': {has_bye}")  # False

# Starts with / Ends with
# C#: bool starts = text.StartsWith("Hello");
# Python: starts = text.startswith("Hello")
starts = text.startswith("Hello")
print(f"Starts with 'Hello': {starts}")  # True

ends = text.endswith("World")
print(f"Ends with 'World': {ends}")  # True

print()

# ============================================
# SECTION 7: MEMBERSHIP OPERATORS
# ============================================

print("=== SECTION 7: MEMBERSHIP OPERATORS ===\n")

# "in" and "not in" - check if item exists in collection
# C#: bool hasApple = fruits.Contains("apple");
# Python: has_apple = "apple" in fruits

fruits = ["apple", "banana", "orange"]

# in - checks if item exists
has_apple = "apple" in fruits
print(f"'apple' in fruits: {has_apple}")  # True

has_grape = "grape" in fruits
print(f"'grape' in fruits: {has_grape}")  # False

# not in - checks if item doesn't exist
no_grape = "grape" not in fruits
print(f"'grape' not in fruits: {no_grape}")  # True

# Works with strings too!
has_hello = "Hello" in "Hello World"
print(f"'Hello' in 'Hello World': {has_hello}")  # True

print()

# ============================================
# SECTION 8: OPERATOR PRECEDENCE
# ============================================

print("=== SECTION 8: OPERATOR PRECEDENCE ===\n")

# Order of operations (highest to lowest):
# 1. ** (Exponentiation)
# 2. *, /, //, % (Multiplication, Division)
# 3. +, - (Addition, Subtraction)
# 4. Comparison operators
# 5. not (Logical NOT)
# 6. and (Logical AND)
# 7. or (Logical OR)

# Example 1: Multiplication before addition
result = 2 + 3 * 4
print(f"2 + 3 * 4 = {result}")  # 14 (not 20!)
# Because: 3 * 4 = 12, then 2 + 12 = 14

# Use parentheses to change order
result = (2 + 3) * 4
print(f"(2 + 3) * 4 = {result}")  # 20

# Example 2: Exponentiation before multiplication
result = 2 * 3 ** 2
print(f"2 * 3 ** 2 = {result}")  # 18 (not 36!)
# Because: 3 ** 2 = 9, then 2 * 9 = 18

result = (2 * 3) ** 2
print(f"(2 * 3) ** 2 = {result}")  # 36

# Example 3: Logical operators
x = 10
y = 5
result = x > 5 and y < 10 or x == 0
print(f"x > 5 and y < 10 or x == 0: {result}")  # True
# Step 1: x > 5 → True
# Step 2: y < 10 → True
# Step 3: True and True → True
# Step 4: True or False → True

print("\n💡 Tip: When in doubt, use parentheses () to make it clear!")

print()

# ============================================
# SECTION 9: PRACTICAL EXAMPLES
# ============================================

print("=== SECTION 9: PRACTICAL EXAMPLES ===\n")

# Example 1: Temperature converter
celsius = 25
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C = {fahrenheit}°F")

# Example 2: Calculate area and perimeter
length = 10
width = 5
area = length * width
perimeter = 2 * (length + width)
print(f"Rectangle {length}x{width}:")
print(f"  Area: {area}")
print(f"  Perimeter: {perimeter}")

# Example 3: Check if number is even or odd
number = 7
is_even = number % 2 == 0
is_odd = number % 2 != 0
print(f"Number {number}:")
print(f"  Is even: {is_even}")
print(f"  Is odd: {is_odd}")

# Example 4: Validate age for voting
age = 18
can_vote = age >= 18
print(f"Age {age}, Can vote: {can_vote}")

# Example 5: Check if year is leap year
# Leap year if divisible by 4, except century years must be divisible by 400
year = 2024
is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
print(f"Year {year} is leap year: {is_leap}")

print()

# ============================================
# SUMMARY
# ============================================

print("=== SUMMARY ===\n")

summary = """
Key Python Operators for .NET Developers:

ARITHMETIC:
  /  → Always returns float (C# has integer division by default)
  // → Integer division (C#: (int)(a/b))
  ** → Power operator (C#: Math.Pow())
  No ++ or -- → Use += 1 or -= 1 instead

LOGICAL:
  and → C#'s &&
  or  → C#'s ||
  not → C#'s !

COMPARISON:
  Same as C#: ==, !=, <, >, <=, >=
  is  → C#'s ReferenceEquals()
  Always use "is None", not "== None"

MEMBERSHIP:
  in     → Check if item exists (C#'s Contains())
  not in → Check if item doesn't exist

STRING:
  len(text)     → C#'s text.Length (function, not property!)
  "ha" * 3      → "hahaha" (repeat string)
  "Hi" in text  → C#'s text.Contains("Hi")

PRECEDENCE:
  ** → *, /, //, % → +, - → Comparison → not → and → or
  Use parentheses () when in doubt!

C# → Python Quick Reference:
  text.Length        → len(text)
  text.Contains(x)   → x in text
  text.ToUpper()     → text.upper()
  Math.Pow(2, 3)     → 2 ** 3
  x++                → x += 1
  true && false      → True and False
  true || false      → True or False
  !true              → not True
"""

print(summary)

print("="*60)
print("Next: example_03_control_flow.py - Learn if/else and loops!")
print("="*60)

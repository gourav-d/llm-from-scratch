"""
Example 4: Functions for .NET Developers
This file demonstrates Python functions with detailed comments and C# comparisons.
"""

# ============================================
# SECTION 1: BASIC FUNCTIONS
# ============================================

print("=== SECTION 1: BASIC FUNCTIONS ===\n")

# C#: public void Greet() { Console.WriteLine("Hello!"); }
# Python: def greet(): print("Hello!")

def greet():
    """Simple function with no parameters."""
    print("Hello, World!")

# Call the function (parentheses required!)
greet()

print()

# Function with parameter
# C#: public void Greet(string name)
# Python: def greet(name)

def greet_person(name):
    """Greet a person by name."""
    print(f"Hello, {name}!")

greet_person("Alice")
greet_person("Bob")

print()

# Function with multiple parameters
def add_numbers(a, b):
    """Add two numbers and print the result."""
    result = a + b
    print(f"{a} + {b} = {result}")

add_numbers(5, 3)
add_numbers(10, 20)

print()

# ============================================
# SECTION 2: RETURN VALUES
# ============================================

print("=== SECTION 2: RETURN VALUES ===\n")

# Single return value
# C#: public int Add(int a, int b) { return a + b; }
# Python: def add(a, b): return a + b

def add(a, b):
    """Add two numbers and return the result."""
    return a + b

result = add(5, 3)
print(f"5 + 3 = {result}")

# Can return any type (no type declaration!)
def get_info():
    """Return different types."""
    return "Alice", 30, True  # Returns a tuple!

name, age, active = get_info()  # Unpack the tuple
print(f"Name: {name}, Age: {age}, Active: {active}")

print()

# Multiple return values (Python-specific!)
# C# 7.0+: public (int min, int max) GetMinMax(List<int> numbers)
# Python: def get_min_max(numbers): return min(numbers), max(numbers)

def get_min_max(numbers):
    """Return both minimum and maximum values."""
    return min(numbers), max(numbers)

nums = [3, 7, 1, 9, 2]
minimum, maximum = get_min_max(nums)
print(f"Numbers: {nums}")
print(f"Min: {minimum}, Max: {maximum}")

print()

# Early return
def is_adult(age):
    """Check if person is adult (traditional way)."""
    if age >= 18:
        return True
    return False

# Better way (more Pythonic)
def is_adult_better(age):
    """Check if person is adult (Pythonic way)."""
    return age >= 18

print(f"Age 20 is adult: {is_adult(20)}")
print(f"Age 16 is adult: {is_adult_better(16)}")

print()

# ============================================
# SECTION 3: DEFAULT PARAMETERS
# ============================================

print("=== SECTION 3: DEFAULT PARAMETERS ===\n")

# Provide default values
# C#: public string Greet(string name, string greeting = "Hello")
# Python: def greet(name, greeting="Hello")

def greet_with_custom(name, greeting="Hello"):
    """Greet someone with a custom or default greeting."""
    return f"{greeting}, {name}!"

# Use default
print(greet_with_custom("Alice"))  # Hello, Alice!

# Override default
print(greet_with_custom("Bob", "Hi"))  # Hi, Bob!
print(greet_with_custom("Charlie", "Hey"))  # Hey, Charlie!

print()

# Multiple default parameters
def create_user(name, age=18, active=True):
    """Create a user dictionary with defaults."""
    return {
        "name": name,
        "age": age,
        "active": active
    }

user1 = create_user("Alice")
print(f"User 1: {user1}")  # Uses all defaults

user2 = create_user("Bob", 25)
print(f"User 2: {user2}")  # Overrides age

user3 = create_user("Charlie", 30, False)
print(f"User 3: {user3}")  # Overrides all

print()

# ============================================
# SECTION 4: NAMED ARGUMENTS (KEYWORD ARGUMENTS)
# ============================================

print("=== SECTION 4: NAMED ARGUMENTS ===\n")

def describe_pet(animal, name, age):
    """Describe a pet with its details."""
    return f"{name} is a {age}-year-old {animal}"

# Positional arguments (order matters)
print(describe_pet("dog", "Buddy", 3))

# Named arguments (order doesn't matter!)
print(describe_pet(name="Buddy", age=3, animal="dog"))
print(describe_pet(age=3, animal="dog", name="Buddy"))  # Same result!

# Mix positional and named (positional must come first!)
print(describe_pet("cat", age=2, name="Whiskers"))

print()

# ============================================
# SECTION 5: *args - VARIABLE POSITIONAL ARGUMENTS
# ============================================

print("=== SECTION 5: *args - VARIABLE ARGUMENTS ===\n")

# C#: public int AddAll(params int[] numbers)
# Python: def add_all(*numbers)

def add_all(*numbers):
    """Add any number of arguments."""
    total = 0
    for num in numbers:
        total += num
    return total

print(f"add_all(1, 2, 3) = {add_all(1, 2, 3)}")
print(f"add_all(10, 20, 30, 40) = {add_all(10, 20, 30, 40)}")
print(f"add_all(5) = {add_all(5)}")
print(f"add_all() = {add_all()}")  # No arguments!

print()

# Better way using built-in sum()
def add_all_better(*numbers):
    """Add any number of arguments using sum()."""
    return sum(numbers)

print(f"add_all_better(1, 2, 3, 4, 5) = {add_all_better(1, 2, 3, 4, 5)}")

print()

# Another example
def find_max(*numbers):
    """Find maximum from any number of arguments."""
    if not numbers:
        return None
    return max(numbers)

print(f"Max of 3, 7, 2, 9, 1: {find_max(3, 7, 2, 9, 1)}")
print(f"Max of 100, 50, 75: {find_max(100, 50, 75)}")

print()

# ============================================
# SECTION 6: **kwargs - VARIABLE KEYWORD ARGUMENTS
# ============================================

print("=== SECTION 6: **kwargs - VARIABLE KEYWORD ARGUMENTS ===\n")

def create_profile(**info):
    """Create profile from any number of keyword arguments."""
    print("Profile:")
    for key, value in info.items():
        print(f"  {key}: {value}")

# Can pass any named arguments!
create_profile(name="Alice", age=30, city="NYC")
print()

create_profile(name="Bob", job="Engineer", salary=80000, married=True)
print()

# Return as dictionary
def build_profile(**info):
    """Build profile dictionary from keyword arguments."""
    return info

profile = build_profile(name="Charlie", age=25, hobby="Guitar")
print(f"Profile dict: {profile}")

print()

# ============================================
# SECTION 7: COMBINING ALL PARAMETER TYPES
# ============================================

print("=== SECTION 7: COMBINING ALL PARAMETER TYPES ===\n")

# Order: positional, *args, default, **kwargs
def full_example(a, b, c=10, *args, **kwargs):
    """Demonstrate all parameter types."""
    print(f"Positional a: {a}")
    print(f"Positional b: {b}")
    print(f"Default c: {c}")
    print(f"*args: {args}")
    print(f"**kwargs: {kwargs}")

full_example(1, 2)
print()

full_example(1, 2, 3)
print()

full_example(1, 2, 3, 4, 5)
print()

full_example(1, 2, 3, 4, 5, x=10, y=20)
print()

# ============================================
# SECTION 8: LAMBDA FUNCTIONS
# ============================================

print("=== SECTION 8: LAMBDA FUNCTIONS ===\n")

# Regular function
def square_regular(x):
    """Square a number (regular function)."""
    return x * x

# Lambda function (one-line anonymous function)
# C#: Func<int, int> square = x => x * x;
# Python: square = lambda x: x * x
square = lambda x: x * x

print(f"Regular: square_regular(5) = {square_regular(5)}")
print(f"Lambda: square(5) = {square(5)}")

print()

# Lambda with multiple parameters
add = lambda a, b: a + b
print(f"add(5, 3) = {add(5, 3)}")

multiply = lambda x, y, z: x * y * z
print(f"multiply(2, 3, 4) = {multiply(2, 3, 4)}")

print()

# Lambda with sorted() - sort by length
words = ["apple", "pie", "banana", "cherry"]
sorted_by_length = sorted(words, key=lambda x: len(x))
print(f"Original: {words}")
print(f"Sorted by length: {sorted_by_length}")

print()

# Lambda with sorted() - sort tuples by second element
pairs = [(1, 5), (3, 2), (2, 8), (4, 1)]
sorted_pairs = sorted(pairs, key=lambda x: x[1])
print(f"Pairs: {pairs}")
print(f"Sorted by 2nd element: {sorted_pairs}")

print()

# Lambda with filter()
# C#: numbers.Where(x => x % 2 == 0)
# Python: filter(lambda x: x % 2 == 0, numbers)
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Numbers: {numbers}")
print(f"Evens: {evens}")

# Lambda with map()
# C#: numbers.Select(x => x * x)
# Python: map(lambda x: x * x, numbers)
squares = list(map(lambda x: x * x, numbers))
print(f"Squares: {squares}")

print()

# Note: List comprehensions are more Pythonic!
# C# LINQ equivalent: numbers.Where(x => x % 2 == 0).Select(x => x * x)
# Python (lambda): list(map(lambda x: x * x, filter(lambda x: x % 2 == 0, numbers)))
# Python (comprehension): [x * x for x in numbers if x % 2 == 0]
even_squares = [x * x for x in numbers if x % 2 == 0]
print(f"Even squares (comprehension): {even_squares}")

print()

# ============================================
# SECTION 9: SCOPE AND LIFETIME
# ============================================

print("=== SECTION 9: SCOPE AND LIFETIME ===\n")

# Local scope
def my_function():
    """Demonstrate local scope."""
    local_var = 10  # Only exists inside function
    print(f"Inside function: local_var = {local_var}")

my_function()
# print(local_var)  # Error! local_var doesn't exist here

print()

# Global scope
global_var = 100  # Global variable

def read_global():
    """Read global variable."""
    print(f"Inside function: global_var = {global_var}")

read_global()
print(f"Outside function: global_var = {global_var}")

print()

# Modifying global variables (not recommended!)
counter = 0

def increment_counter():
    """Modify global variable (requires 'global' keyword)."""
    global counter  # Declare we want to modify global
    counter += 1

print(f"Counter: {counter}")
increment_counter()
print(f"Counter after increment: {counter}")
increment_counter()
print(f"Counter after 2nd increment: {counter}")

print()

# Better approach - return values instead
def get_incremented(value):
    """Better way - return new value instead of modifying global."""
    return value + 1

my_counter = 0
print(f"My counter: {my_counter}")
my_counter = get_incremented(my_counter)
print(f"My counter after increment: {my_counter}")

print()

# ============================================
# SECTION 10: DOCSTRINGS
# ============================================

print("=== SECTION 10: DOCSTRINGS ===\n")

def calculate_area(length, width):
    """
    Calculate the area of a rectangle.

    This is a docstring - documentation for the function.
    Similar to C# XML comments.

    Parameters:
        length (float): The length of the rectangle
        width (float): The width of the rectangle

    Returns:
        float: The area of the rectangle

    Example:
        >>> calculate_area(5, 3)
        15
    """
    return length * width

area = calculate_area(5, 3)
print(f"Area: {area}")

# Access docstring
print("\nDocstring:")
print(calculate_area.__doc__)

print()

# ============================================
# SECTION 11: PRACTICAL EXAMPLES
# ============================================

print("=== SECTION 11: PRACTICAL EXAMPLES ===\n")

# Example 1: Temperature converter
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5/9

print(f"25°C = {celsius_to_fahrenheit(25):.1f}°F")
print(f"77°F = {fahrenheit_to_celsius(77):.1f}°C")

print()

# Example 2: Fibonacci number
def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print("First 10 Fibonacci numbers:")
for i in range(10):
    print(fibonacci(i), end=" ")
print()

print()

# Example 3: Palindrome checker
def is_palindrome(text):
    """Check if text is a palindrome."""
    # Remove spaces and convert to lowercase
    text = text.replace(" ", "").lower()
    return text == text[::-1]  # [::-1] reverses string

print(f"'racecar' is palindrome: {is_palindrome('racecar')}")
print(f"'hello' is palindrome: {is_palindrome('hello')}")
print(f"'A man a plan a canal Panama' is palindrome: {is_palindrome('A man a plan a canal Panama')}")

print()

# Example 4: Calculate statistics
def calculate_stats(*numbers):
    """Calculate min, max, average, and sum of numbers."""
    if not numbers:
        return None, None, None, None
    return min(numbers), max(numbers), sum(numbers)/len(numbers), sum(numbers)

nums = [5, 2, 8, 1, 9, 3]
min_val, max_val, avg_val, sum_val = calculate_stats(*nums)
print(f"Numbers: {nums}")
print(f"Min: {min_val}, Max: {max_val}, Avg: {avg_val:.2f}, Sum: {sum_val}")

print()

# Example 5: Build HTML tag
def create_tag(tag, content, **attributes):
    """Create an HTML tag with attributes."""
    attrs = " ".join([f'{key}="{value}"' for key, value in attributes.items()])
    if attrs:
        return f"<{tag} {attrs}>{content}</{tag}>"
    return f"<{tag}>{content}</{tag}>"

print(create_tag("h1", "Hello World"))
print(create_tag("a", "Click here", href="https://example.com", target="_blank"))
print(create_tag("div", "Content", class_="container", id="main"))

print()

# ============================================
# SUMMARY
# ============================================

print("=== SUMMARY ===\n")

summary = """
Python Functions for .NET Developers:

BASIC SYNTAX:
  C#:     public int Add(int a, int b) { return a + b; }
  Python: def add(a, b): return a + b

  - Use 'def' keyword
  - No return type declaration
  - No access modifiers
  - Colon : after parameters
  - Indentation for body

RETURN VALUES:
  - Single: return value
  - Multiple: return val1, val2, val3  (returns tuple)
  - Unpack: a, b, c = function()
  - No return statement → returns None

DEFAULT PARAMETERS:
  def greet(name, greeting="Hello"):
  - Provide default values
  - Must come after non-default params
  - Same as C#!

NAMED ARGUMENTS:
  function(name="Alice", age=30)
  - Order doesn't matter
  - Makes code more readable
  - Can mix with positional (positional first!)

*args (VARIABLE POSITIONAL):
  def add_all(*numbers):
  - Collects arguments into tuple
  - Like C#'s params keyword
  - Can pass any number of arguments

**kwargs (VARIABLE KEYWORD):
  def create_profile(**info):
  - Collects named arguments into dictionary
  - Very flexible for options
  - No direct C# equivalent

LAMBDA FUNCTIONS:
  C#:     Func<int, int> square = x => x * x;
  Python: square = lambda x: x * x

  - Anonymous one-line functions
  - Format: lambda params: expression
  - Great with sort, filter, map

SCOPE:
  - Local: Variables inside function
  - Global: Variables outside all functions
  - Use 'global' keyword to modify global (not recommended!)
  - Better: Return new values

DOCSTRINGS:
  \"\"\"Documentation for function.\"\"\"
  - Triple quotes
  - First thing after def
  - Like C# XML comments
  - Accessed via function.__doc__

C# → Python Quick Reference:
  public void Method()              → def method():
  public int Method(int x)          → def method(x):
  public int Method(int x = 5)      → def method(x=5):
  params int[] nums                 → *nums
  Func<int, int> f = x => x * 2     → f = lambda x: x * 2
  numbers.Where(x => x > 5)         → filter(lambda x: x > 5, numbers)
  numbers.Select(x => x * 2)        → map(lambda x: x * 2, numbers)

  But list comprehensions are more Pythonic:
  [x * 2 for x in numbers if x > 5]
"""

print(summary)

print("="*60)
print("Next: example_05_lists_tuples.py - Learn about lists!")
print("="*60)

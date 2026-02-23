"""
Example 1: Python Basics for .NET Developers
This file demonstrates fundamental Python concepts with detailed comments.
"""

# ============================================
# SECTION 1: VARIABLES AND TYPES
# ============================================

print("=== SECTION 1: VARIABLES AND TYPES ===\n")

# In C#: int age = 25;
# In Python: No type declaration needed!
age = 25
print(f"Age: {age}, Type: {type(age)}")

# In C#: string name = "Alice";
# In Python: Same concept, no type
name = "Alice"
print(f"Name: {name}, Type: {type(name)}")

# In C#: double price = 19.99;
# In Python: float
price = 19.99
print(f"Price: {price}, Type: {type(price)}")

# In C#: bool isActive = true;
# In Python: True (capitalized!)
is_active = True
print(f"Active: {is_active}, Type: {type(is_active)}")

# In C#: string greeting = $"Hello, {name}";
# In Python: f-strings
greeting = f"Hello, {name}!"
print(greeting)

print()  # Blank line

# ============================================
# SECTION 2: OPERATORS
# ============================================

print("=== SECTION 2: OPERATORS ===\n")

x = 10
y = 3

# Basic arithmetic
print(f"{x} + {y} = {x + y}")           # Addition: 13
print(f"{x} - {y} = {x - y}")           # Subtraction: 7
print(f"{x} * {y} = {x * y}")           # Multiplication: 30
print(f"{x} / {y} = {x / y}")           # Division: 3.333... (always float!)
print(f"{x} // {y} = {x // y}")         # Integer division: 3
print(f"{x} % {y} = {x % y}")           # Modulus (remainder): 1
print(f"{x} ** {y} = {x ** y}")         # Power (10³): 1000

print()

# Comparison operators
print(f"{x} > {y}: {x > y}")            # True
print(f"{x} == {y}: {x == y}")          # False
print(f"{x} != {y}: {x != y}")          # True

print()

# Logical operators (words, not symbols!)
# C#: &&, ||, !
# Python: and, or, not
has_license = True
age_18_plus = age >= 18

can_drive = age_18_plus and has_license
print(f"Can drive: {can_drive}")        # True

print()

# String operations
text = "Python"
print(f"Length: {len(text)}")           # 6
print(f"Uppercase: {text.upper()}")     # PYTHON
print(f"Repeated: {text * 3}")          # PythonPythonPython (Python-specific!)
print(f"Contains 'Py': {'Py' in text}") # True

print()

# ============================================
# SECTION 3: CONTROL FLOW
# ============================================

print("=== SECTION 3: CONTROL FLOW ===\n")

# if/elif/else (note: no parentheses, colon required, indentation!)
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

print(f"Score {score} = Grade {grade}")

print()

# for loop through range
# C#: for (int i = 0; i < 5; i++)
# Python: for i in range(5)
print("Counting 0-4:")
for i in range(5):
    print(i, end=" ")  # end=" " prints on same line
print()  # New line

print()

# for loop through list
# C#: foreach (var fruit in fruits)
# Python: for fruit in fruits
fruits = ["apple", "banana", "orange"]
print("Fruits:")
for fruit in fruits:
    print(f"- {fruit}")

print()

# for loop with index
# enumerate() gives us both index and value
print("Fruits with index:")
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")

print()

# while loop
count = 0
print("While loop (0-4):")
while count < 5:
    print(count, end=" ")
    count += 1  # Remember: no count++!
print()

print()

# ============================================
# SECTION 4: FUNCTIONS
# ============================================

print("=== SECTION 4: FUNCTIONS ===\n")


# Defining a function
# C#: public int Add(int a, int b) { return a + b; }
# Python: def add(a, b): return a + b
def add(a, b):
    """Add two numbers and return the result."""
    return a + b


result = add(5, 3)
print(f"add(5, 3) = {result}")


# Function with default parameters
def greet(name, greeting="Hello"):
    """Greet someone with a custom or default greeting."""
    return f"{greeting}, {name}!"


print(greet("Alice"))              # Uses default: "Hello, Alice!"
print(greet("Bob", "Hi"))          # Custom: "Hi, Bob!"


# Lambda function (like C#'s lambda)
# C#: Func<int, int> square = x => x * x;
# Python: square = lambda x: x * x
square = lambda x: x * x
print(f"square(5) = {square(5)}")

print()

# ============================================
# SECTION 5: DATA STRUCTURES
# ============================================

print("=== SECTION 5: DATA STRUCTURES ===\n")

# Lists (like C#'s List<T>)
numbers = [1, 2, 3, 4, 5]
print(f"List: {numbers}")
print(f"First: {numbers[0]}")      # Indexing starts at 0 (like C#)
print(f"Last: {numbers[-1]}")      # Negative indexing (Python-specific!)
print(f"Slice [1:3]: {numbers[1:3]}")  # [2, 3] - slicing

numbers.append(6)                  # Add to end
print(f"After append: {numbers}")

print()

# Dictionaries (like C#'s Dictionary<K,V>)
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

print(f"Dictionary: {person}")
print(f"Name: {person['name']}")   # Access by key
print(f"Age: {person.get('age')}")  # Alternative way

# Loop through dictionary
print("Person details:")
for key, value in person.items():
    print(f"  {key}: {value}")

print()

# ============================================
# SECTION 6: LIST COMPREHENSIONS
# ============================================

print("=== SECTION 6: LIST COMPREHENSIONS ===\n")

# Create a list of squares
# C#: var squares = numbers.Select(x => x * x).ToList();
# Python: squares = [x * x for x in numbers]
numbers = [1, 2, 3, 4, 5]
squares = [x * x for x in numbers]
print(f"Numbers: {numbers}")
print(f"Squares: {squares}")

# Filter even numbers
# C#: var evens = numbers.Where(x => x % 2 == 0).ToList();
# Python: evens = [x for x in numbers if x % 2 == 0]
evens = [x for x in numbers if x % 2 == 0]
print(f"Evens: {evens}")

print()

# ============================================
# SECTION 7: CLASSES
# ============================================

print("=== SECTION 7: CLASSES (OOP) ===\n")


class Dog:
    """A simple Dog class demonstrating OOP in Python."""

    # Constructor
    # C#: public Dog(string name, int age)
    # Python: def __init__(self, name, age)
    def __init__(self, name, age):
        self.name = name  # self = this (in C#)
        self.age = age

    # Method
    def bark(self):
        """Make the dog bark."""
        return f"{self.name} says: Woof!"

    # String representation
    # C#: override ToString()
    # Python: __str__
    def __str__(self):
        return f"Dog(name={self.name}, age={self.age})"


# Create instance
# C#: var dog = new Dog("Buddy", 3);
# Python: dog = Dog("Buddy", 3)
dog = Dog("Buddy", 3)

print(f"Dog: {dog}")
print(dog.bark())

print()

# ============================================
# SECTION 8: ERROR HANDLING
# ============================================

print("=== SECTION 8: ERROR HANDLING ===\n")

# try/except (like C#'s try/catch)
try:
    result = 10 / 0  # This will raise an error
except ZeroDivisionError as e:
    print(f"Error caught: {e}")
    result = None

print(f"Result: {result}")

# Try converting invalid string
try:
    num = int("abc")  # This will fail
except ValueError as e:
    print(f"Conversion error: {e}")

print()

# ============================================
# SUMMARY
# ============================================

print("=== SUMMARY ===\n")

summary = """
Key Python Concepts for .NET Developers:
1. No type declarations (dynamic typing)
2. No semicolons (lines end automatically)
3. Indentation defines blocks (not braces)
4. snake_case naming (not camelCase)
5. True/False capitalized
6. 'and', 'or', 'not' (not &&, ||, !)
7. No ++ or -- operators
8. f-strings for formatting
9. List comprehensions (like LINQ)
10. 'self' instead of 'this'

You're ready to build LLMs! 🚀
"""

print(summary)

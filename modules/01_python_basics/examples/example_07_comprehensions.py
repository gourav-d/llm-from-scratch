"""
Example 7: List Comprehensions for .NET Developers
This file demonstrates Python comprehensions with detailed comments and C# LINQ comparisons.
"""

# ============================================
# SECTION 1: BASIC LIST COMPREHENSIONS
# ============================================

print("=== SECTION 1: BASIC LIST COMPREHENSIONS ===\n")

# Comprehensions are Python's version of C# LINQ!
# C#: var squares = numbers.Select(x => x * x).ToList();
# Python: squares = [x * x for x in numbers]

# Traditional way (with loop)
print("Traditional way (with loop):")
squares_traditional = []
for x in range(5):
    squares_traditional.append(x * x)
print(f"Squares: {squares_traditional}")

print()

# List comprehension way (Pythonic!)
print("List comprehension way:")
squares = [x * x for x in range(5)]
print(f"Squares: {squares}")

print()

# Format: [expression for item in iterable]
# - expression: What to add to the list (x * x)
# - for item in iterable: Loop part

# More examples
numbers = [1, 2, 3, 4, 5]
print(f"Original numbers: {numbers}")

# Double all numbers
# C#: numbers.Select(x => x * 2).ToList()
# Python: [x * 2 for x in numbers]
doubled = [x * 2 for x in numbers]
print(f"Doubled: {doubled}")

# Square all numbers
squared = [x ** 2 for x in numbers]
print(f"Squared: {squared}")

print()

# String operations
words = ["hello", "world", "python"]
print(f"Original words: {words}")

# Convert to uppercase
# C#: words.Select(w => w.ToUpper()).ToList()
# Python: [w.upper() for w in words]
upper = [word.upper() for word in words]
print(f"Uppercase: {upper}")

# Get first character
first_chars = [word[0] for word in words]
print(f"First characters: {first_chars}")

# Get lengths
# C#: words.Select(w => w.Length).ToList()
# Python: [len(w) for w in words]
lengths = [len(word) for word in words]
print(f"Lengths: {lengths}")

print()

# ============================================
# SECTION 2: COMPREHENSIONS WITH CONDITIONS (FILTERING)
# ============================================

print("=== SECTION 2: COMPREHENSIONS WITH CONDITIONS ===\n")

# Format: [expression for item in iterable if condition]

# Traditional way
print("Traditional way (filter even numbers):")
evens_traditional = []
for x in range(10):
    if x % 2 == 0:
        evens_traditional.append(x)
print(f"Evens: {evens_traditional}")

print()

# Comprehension way
# C#: numbers.Where(x => x % 2 == 0).ToList()
# Python: [x for x in numbers if x % 2 == 0]
print("Comprehension way:")
evens = [x for x in range(10) if x % 2 == 0]
print(f"Evens: {evens}")

# Odd numbers
odds = [x for x in range(10) if x % 2 != 0]
print(f"Odds: {odds}")

print()

# Numbers greater than threshold
numbers = [1, 8, 3, 10, 5, 12, 2, 15]
print(f"Numbers: {numbers}")

greater_than_5 = [x for x in numbers if x > 5]
print(f"Greater than 5: {greater_than_5}")

print()

# String filtering
words = ["python", "java", "perl", "ruby", "javascript"]
print(f"Words: {words}")

# Words starting with 'p'
# C#: words.Where(w => w.StartsWith("p")).ToList()
# Python: [w for w in words if w.startswith('p')]
p_words = [w for w in words if w.startswith('p')]
print(f"Words starting with 'p': {p_words}")

# Words longer than 4 characters
long_words = [w for w in words if len(w) > 4]
print(f"Words longer than 4 chars: {long_words}")

print()

# Filter positive numbers
numbers = [-2, 3, -1, 5, -4, 8, -7, 10]
print(f"Numbers: {numbers}")

positive = [x for x in numbers if x > 0]
print(f"Positive: {positive}")

negative = [x for x in numbers if x < 0]
print(f"Negative: {negative}")

print()

# ============================================
# SECTION 3: COMBINING EXPRESSION AND CONDITION
# ============================================

print("=== SECTION 3: COMBINING EXPRESSION AND CONDITION ===\n")

# You can transform AND filter in one comprehension!
# C#: numbers.Where(x => x % 2 == 0).Select(x => x * x).ToList()
# Python: [x * x for x in numbers if x % 2 == 0]

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(f"Numbers: {numbers}")

# Square only even numbers
even_squares = [x * x for x in numbers if x % 2 == 0]
print(f"Squares of even numbers: {even_squares}")

print()

# Uppercase only long words
words = ["hi", "hello", "world", "a", "python", "code"]
print(f"Words: {words}")

long_upper = [w.upper() for w in words if len(w) > 4]
print(f"Uppercase words longer than 4 chars: {long_upper}")

print()

# Double only positive numbers
numbers = [-3, 5, -2, 8, -1, 10]
print(f"Numbers: {numbers}")

doubled_positive = [x * 2 for x in numbers if x > 0]
print(f"Doubled positive: {doubled_positive}")

print()

# ============================================
# SECTION 4: IF-ELSE IN COMPREHENSIONS
# ============================================

print("=== SECTION 4: IF-ELSE IN COMPREHENSIONS ===\n")

# Format: [expr_if_true if condition else expr_if_false for item in iterable]
# NOTE: if-else goes BEFORE the for!

# Label numbers as even or odd
# C#: numbers.Select(x => x % 2 == 0 ? "even" : "odd").ToList()
# Python: ["even" if x % 2 == 0 else "odd" for x in numbers]
numbers = [1, 2, 3, 4, 5]
print(f"Numbers: {numbers}")

labels = ["even" if x % 2 == 0 else "odd" for x in numbers]
print(f"Labels: {labels}")

print()

# Replace negative numbers with 0
numbers = [3, -1, 5, -2, 8, -4]
print(f"Original: {numbers}")

positive_only = [x if x > 0 else 0 for x in numbers]
print(f"Negative → 0: {positive_only}")

print()

# Absolute values
numbers = [-3, 5, -2, 8]
print(f"Original: {numbers}")

absolute = [x if x >= 0 else -x for x in numbers]
print(f"Absolute: {absolute}")

# Or using abs() function
absolute_builtin = [abs(x) for x in numbers]
print(f"Absolute (using abs): {absolute_builtin}")

print()

# Important distinction!
print("Important distinction:")
print()

# Filtering (if only) - goes AFTER for
# Includes only items that match condition
filtering = [x for x in numbers if x > 0]
print(f"Filtering (if only - AFTER for): {filtering}")

# Conditional expression (if-else) - goes BEFORE for
# Includes ALL items, but transforms some
conditional = [x if x > 0 else 0 for x in numbers]
print(f"Conditional (if-else - BEFORE for): {conditional}")

print()

# ============================================
# SECTION 5: NESTED LOOPS IN COMPREHENSIONS
# ============================================

print("=== SECTION 5: NESTED LOOPS IN COMPREHENSIONS ===\n")

# Traditional nested loops
print("Traditional way (nested loops):")
result_traditional = []
for x in [1, 2, 3]:
    for y in [10, 20, 30]:
        result_traditional.append((x, y))
print(f"Result: {result_traditional}")

print()

# Comprehension way
# Format: [expr for x in list1 for y in list2]
print("Comprehension way:")
result = [(x, y) for x in [1, 2, 3] for y in [10, 20, 30]]
print(f"Result: {result}")

print()

# Flatten 2D list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(f"Matrix: {matrix}")

# Traditional way
flat_traditional = []
for row in matrix:
    for num in row:
        flat_traditional.append(num)
print(f"Flat (traditional): {flat_traditional}")

# Comprehension way
# C#: matrix.SelectMany(row => row).ToList()
# Python: [num for row in matrix for num in row]
flat = [num for row in matrix for num in row]
print(f"Flat (comprehension): {flat}")

print()

# All combinations
colors = ["red", "blue"]
sizes = ["S", "M", "L"]
print(f"Colors: {colors}")
print(f"Sizes: {sizes}")

combos = [(color, size) for color in colors for size in sizes]
print(f"All combinations: {combos}")

print()

# Multiplication table
print("Multiplication table (5x5):")
mult_table = [[x * y for y in range(1, 6)] for x in range(1, 6)]
for row in mult_table:
    print(row)

print()

# ============================================
# SECTION 6: DICTIONARY COMPREHENSIONS
# ============================================

print("=== SECTION 6: DICTIONARY COMPREHENSIONS ===\n")

# Format: {key: value for item in iterable}
# C#: Enumerable.Range(0,5).ToDictionary(x => x, x => x*x)
# Python: {x: x**2 for x in range(5)}

# Create dictionary of squares
squares_dict = {x: x**2 for x in range(5)}
print(f"Squares dict: {squares_dict}")

print()

# Create from two lists using zip
keys = ["name", "age", "city"]
values = ["Alice", 30, "NYC"]
print(f"Keys: {keys}")
print(f"Values: {values}")

person = {k: v for k, v in zip(keys, values)}
print(f"Person: {person}")

print()

# Transform existing dictionary
prices = {"apple": 0.50, "banana": 0.30, "orange": 0.60}
print(f"Original prices: {prices}")

# Double all prices
doubled_prices = {fruit: price * 2 for fruit, price in prices.items()}
print(f"Doubled prices: {doubled_prices}")

# Convert to cents
cents = {fruit: int(price * 100) for fruit, price in prices.items()}
print(f"Prices in cents: {cents}")

print()

# Swap keys and values
original = {"a": 1, "b": 2, "c": 3}
print(f"Original: {original}")

swapped = {v: k for k, v in original.items()}
print(f"Swapped: {swapped}")

print()

# Filter dictionary
prices = {"apple": 0.50, "banana": 0.30, "orange": 0.60, "grape": 0.80}
print(f"All prices: {prices}")

expensive = {fruit: price for fruit, price in prices.items() if price > 0.40}
print(f"Expensive (> 0.40): {expensive}")

print()

# Count word lengths
words = ["hello", "world", "python", "code"]
print(f"Words: {words}")

word_lengths = {word: len(word) for word in words}
print(f"Word lengths: {word_lengths}")

print()

# Create index dictionary
fruits = ["apple", "banana", "orange"]
print(f"Fruits: {fruits}")

fruit_index = {fruit: i for i, fruit in enumerate(fruits)}
print(f"Fruit index: {fruit_index}")

print()

# ============================================
# SECTION 7: SET COMPREHENSIONS
# ============================================

print("=== SECTION 7: SET COMPREHENSIONS ===\n")

# Format: {expression for item in iterable}
# NOTE: Use {} but no key:value pairs (that's dict!)

# Create set of squares
squares_set = {x**2 for x in range(10)}
print(f"Squares set: {squares_set}")

print()

# Unique first letters
words = ["apple", "avocado", "banana", "blueberry", "cherry"]
print(f"Words: {words}")

first_letters = {word[0] for word in words}
print(f"Unique first letters: {first_letters}")

print()

# Unique lengths
words = ["hi", "hello", "world", "hi", "python", "hello"]
print(f"Words: {words}")

unique_lengths = {len(word) for word in words}
print(f"Unique lengths: {unique_lengths}")

print()

# Even numbers (as set)
evens_set = {x for x in range(20) if x % 2 == 0}
print(f"Even numbers (0-19): {evens_set}")

print()

# Uppercase vowels from text
text = "hello world python"
print(f"Text: {text}")

vowels = {char.upper() for char in text if char in 'aeiou'}
print(f"Unique vowels (uppercase): {vowels}")

print()

# ============================================
# SECTION 8: PRACTICAL EXAMPLES
# ============================================

print("=== SECTION 8: PRACTICAL EXAMPLES ===\n")

# Example 1: Extract numbers from strings
print("Example 1: Extract numbers from strings:")
strings = ["abc123", "xyz456", "def789"]
print(f"Strings: {strings}")

# Extract digits and convert to int
numbers = [int(''.join(c for c in s if c.isdigit())) for s in strings]
print(f"Numbers: {numbers}")

print()

# Example 2: Grade students
print("Example 2: Grade students:")
scores = [85, 92, 78, 95, 88, 76, 91]
print(f"Scores: {scores}")

# Assign grades based on score
grades = [
    "A" if s >= 90 else
    "B" if s >= 80 else
    "C" if s >= 70 else
    "F"
    for s in scores
]
print(f"Grades: {grades}")

print()

# Example 3: FizzBuzz using comprehension
print("Example 3: FizzBuzz (1-20):")
fizzbuzz = [
    "FizzBuzz" if n % 15 == 0 else
    "Fizz" if n % 3 == 0 else
    "Buzz" if n % 5 == 0 else
    str(n)
    for n in range(1, 21)
]
print(fizzbuzz)

print()

# Example 4: Transpose matrix
print("Example 4: Transpose matrix:")
matrix = [
    [1, 2, 3],
    [4, 5, 6]
]
print(f"Original matrix:")
for row in matrix:
    print(f"  {row}")

transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
print(f"Transposed:")
for row in transposed:
    print(f"  {row}")

print()

# Example 5: Parse CSV-like data
print("Example 5: Parse CSV data:")
csv_data = "name,age\nAlice,30\nBob,25\nCharlie,35"
print(f"CSV data:\n{csv_data}")
print()

lines = csv_data.split('\n')[1:]  # Skip header
people = [
    {"name": parts[0], "age": int(parts[1])}
    for line in lines
    for parts in [line.split(',')]
]
print(f"Parsed people: {people}")

print()

# Example 6: Create lookup table
print("Example 6: Create lookup table:")
students = [
    {"id": 1, "name": "Alice", "score": 95},
    {"id": 2, "name": "Bob", "score": 87},
    {"id": 3, "name": "Charlie", "score": 92}
]

# Create dictionary with id as key
lookup = {student["id"]: student for student in students}
print(f"Lookup by ID: {lookup}")

# Access by ID
print(f"Student 2: {lookup[2]}")

print()

# ============================================
# SECTION 9: WHEN TO USE COMPREHENSIONS
# ============================================

print("=== SECTION 9: WHEN TO USE COMPREHENSIONS ===\n")

print("✅ USE COMPREHENSIONS WHEN:")
print("- Creating a new list/dict/set from existing data")
print("- Simple transformation or filtering")
print("- Code is still readable (one line)")
print()

# Good examples
print("Good examples:")
print("squares = [x**2 for x in range(10)]")
print("evens = [x for x in numbers if x % 2 == 0]")
print("upper = [w.upper() for w in words]")

print()

print("❌ DON'T USE COMPREHENSIONS WHEN:")
print("- Logic is complex (hard to read)")
print("- Need multiple operations")
print("- Need error handling")
print()

# Bad example (too complex!)
print("Bad example (too complex):")
print("[x**2 if x > 0 else abs(x) if x < -5 else 0 for x in numbers if x != 3]")
print()
print("Better - use regular loop for complex logic:")
print("""
result = []
for x in numbers:
    if x != 3:
        if x > 0:
            result.append(x**2)
        elif x < -5:
            result.append(abs(x))
        else:
            result.append(0)
""")

print()

# ============================================
# SECTION 10: PERFORMANCE COMPARISON
# ============================================

print("=== SECTION 10: PERFORMANCE ===\n")

print("Comprehensions are generally FASTER than loops!")
print("They're also more readable for simple operations.")
print()

# Simple performance demonstration
import time

# Using loop
start = time.time()
squares_loop = []
for x in range(100000):
    squares_loop.append(x**2)
loop_time = time.time() - start

# Using comprehension
start = time.time()
squares_comp = [x**2 for x in range(100000)]
comp_time = time.time() - start

print(f"Loop time: {loop_time:.4f}s")
print(f"Comprehension time: {comp_time:.4f}s")
print(f"Comprehension is ~{loop_time/comp_time:.2f}x faster!")

print()

# ============================================
# SUMMARY
# ============================================

print("=== SUMMARY ===\n")

summary = """
List Comprehensions for .NET Developers:

BASIC SYNTAX:
  C#: numbers.Select(x => x * x).ToList()
  Python: [x * x for x in numbers]

  Format: [expression for item in iterable]

WITH FILTERING (if only):
  C#: numbers.Where(x => x % 2 == 0).ToList()
  Python: [x for x in numbers if x % 2 == 0]

  Format: [expression for item in iterable if condition]
  Note: if goes AFTER for

WITH IF-ELSE (conditional expression):
  C#: numbers.Select(x => x > 0 ? x : 0).ToList()
  Python: [x if x > 0 else 0 for x in numbers]

  Format: [expr_if_true if condition else expr_if_false for item in iterable]
  Note: if-else goes BEFORE for

NESTED LOOPS:
  C#: list1.SelectMany(x => list2.Select(y => (x, y))).ToList()
  Python: [(x, y) for x in list1 for y in list2]

  Flatten 2D:
  C#: matrix.SelectMany(row => row).ToList()
  Python: [item for row in matrix for item in row]

DICTIONARY COMPREHENSIONS:
  C#: Enumerable.Range(0,5).ToDictionary(x => x, x => x*x)
  Python: {x: x**2 for x in range(5)}

  Format: {key: value for item in iterable}

SET COMPREHENSIONS:
  C#: new HashSet<int>(numbers.Select(x => x * x))
  Python: {x**2 for x in numbers}

  Format: {expression for item in iterable}

WHEN TO USE:
  ✅ Simple transformations
  ✅ Filtering
  ✅ One-line operations
  ✅ Better readability

  ❌ Complex logic
  ❌ Multiple operations
  ❌ Error handling needed

C# LINQ → Python Comprehensions:
  .Select(x => expr)           → [expr for x in list]
  .Where(x => condition)       → [x for x in list if condition]
  .Select().Where()            → [expr for x in list if condition]
  .SelectMany()                → [item for sublist in list for item in sublist]
  .ToDictionary(k => k, v => v) → {k: v for ...}

PERFORMANCE:
  - Comprehensions are faster than loops
  - More readable for simple operations
  - Pythonic way to transform collections
"""

print(summary)

print("="*60)
print("Next: example_08_classes_oop.py - Learn Object-Oriented Programming!")
print("="*60)

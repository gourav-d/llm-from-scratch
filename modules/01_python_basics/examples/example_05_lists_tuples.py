"""
Example 5: Lists and Tuples for .NET Developers
This file demonstrates Python lists and tuples with detailed comments and C# comparisons.
"""

# ============================================
# SECTION 1: CREATING LISTS
# ============================================

print("=== SECTION 1: CREATING LISTS ===\n")

# C#: var numbers = new List<int> {1, 2, 3, 4, 5};
# Python: numbers = [1, 2, 3, 4, 5]

# Empty list
empty = []
print(f"Empty list: {empty}")

# List with initial values
numbers = [1, 2, 3, 4, 5]
print(f"Numbers: {numbers}")

names = ["Alice", "Bob", "Charlie"]
print(f"Names: {names}")

# Mixed types (possible but not recommended!)
mixed = [1, "hello", 3.14, True]
print(f"Mixed types: {mixed}")

# Nested lists (2D array)
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print(f"Matrix: {matrix}")

# Using list() constructor
numbers_from_range = list(range(5))  # [0, 1, 2, 3, 4]
print(f"From range: {numbers_from_range}")

chars = list("hello")  # ['h', 'e', 'l', 'l', 'o']
print(f"From string: {chars}")

print()

# ============================================
# SECTION 2: ACCESSING ELEMENTS (INDEXING)
# ============================================

print("=== SECTION 2: ACCESSING ELEMENTS ===\n")

fruits = ["apple", "banana", "orange", "grape"]

# Positive indexing (from start) - same as C#!
print(f"First element [0]: {fruits[0]}")    # apple
print(f"Second element [1]: {fruits[1]}")   # banana
print(f"Last element [3]: {fruits[3]}")     # grape

# Negative indexing (from end) - Python-specific!
# C# doesn't have this!
print(f"\nNegative indexing:")
print(f"Last element [-1]: {fruits[-1]}")      # grape
print(f"Second-to-last [-2]: {fruits[-2]}")   # orange
print(f"First element [-4]: {fruits[-4]}")    # apple

print("\nIndexing visualization:")
print("Index:     0        1         2        3")
print(f"        {fruits}")
print("Negative: -4       -3        -2       -1")

print()

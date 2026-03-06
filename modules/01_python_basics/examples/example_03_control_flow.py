"""
Example 3: Control Flow (if/else, Loops) for .NET Developers
This file demonstrates Python control flow with detailed comments and C# comparisons.
"""

# ============================================
# CRITICAL: INDENTATION IN PYTHON
# ============================================

print("=== CRITICAL: INDENTATION IN PYTHON ===\n")

print("""
⚠️  MOST IMPORTANT DIFFERENCE FROM C#:

C# uses curly braces {}:
    if (x > 5)
    {
        Console.WriteLine("Greater");
    }

Python uses INDENTATION (spaces/tabs):
    if x > 5:
        print("Greater")

RULES:
1. Use 4 spaces for each indentation level (standard)
2. Must be consistent - mixing spaces and tabs = ERROR!
3. Indentation defines code blocks (not braces!)
""")

print()

# ============================================
# SECTION 1: if/elif/else STATEMENTS
# ============================================

print("=== SECTION 1: if/elif/else STATEMENTS ===\n")

# Basic if statement
# C#: if (age >= 18) { Console.WriteLine("Adult"); }
# Python: if age >= 18: print("Adult")
age = 18

if age >= 18:
    print(f"Age {age}: You are an adult")
    print("You can vote")

print()

# if/else
# C#: if (age >= 18) { ... } else { ... }
# Python: if age >= 18: ... else: ...
age = 16

if age >= 18:
    print(f"Age {age}: You are an adult")
else:
    print(f"Age {age}: You are a minor")

print()

# if/elif/else (Multiple conditions)
# C#: if (...) else if (...) else if (...) else
# Python: if (...) elif (...) elif (...) else
score = 85

print(f"Score: {score}")
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Grade: {grade}")

print()

# Nested if statements
age = 20
has_license = True

if age >= 18:
    if has_license:
        print(f"Age {age}, Has license: You can drive!")
    else:
        print(f"Age {age}, No license: Get a license first")
else:
    print(f"Age {age}: Too young to drive")

print()

# Better way - combine conditions with 'and'
if age >= 18 and has_license:
    print("Simplified version: You can drive!")
elif age >= 18:
    print("Simplified version: You need a license")
else:
    print("Simplified version: Too young to drive")

print()

# Ternary operator (one-line if/else)
# C#: string status = age >= 18 ? "Adult" : "Minor";
# Python: status = "Adult" if age >= 18 else "Minor"
age = 20
status = "Adult" if age >= 18 else "Minor"
print(f"Age {age}: {status} (using ternary operator)")

age = 15
status = "Adult" if age >= 18 else "Minor"
print(f"Age {age}: {status} (using ternary operator)")

print()

# ============================================
# SECTION 2: for LOOPS - range()
# ============================================

print("=== SECTION 2: for LOOPS - range() ===\n")

# Loop through range (like C#'s for loop)
# C#: for (int i = 0; i < 5; i++)
# Python: for i in range(5)
print("Counting 0-4:")
for i in range(5):
    print(i, end=" ")  # end=" " prints on same line
print()  # New line

print()

# range(start, stop) - from start to stop-1
print("Counting 2-4:")
for i in range(2, 5):
    print(i, end=" ")
print()

print()

# range(start, stop, step) - with step size
print("Even numbers 0-10:")
for i in range(0, 11, 2):
    print(i, end=" ")
print()

print()

# Countdown using range
print("Countdown:")
for i in range(5, 0, -1):  # Start at 5, stop before 0, step -1
    print(i, end=" ")
print("Blast off!")

print()

# ============================================
# SECTION 3: for LOOPS - Lists
# ============================================

print("=== SECTION 3: for LOOPS - Lists ===\n")

# Loop through list (like C#'s foreach)
# C#: foreach (var fruit in fruits)
# Python: for fruit in fruits
fruits = ["apple", "banana", "orange"]

print("Fruits:")
for fruit in fruits:
    print(f"  - {fruit}")

print()

# Loop with index using enumerate()
# C#: for (int i = 0; i < fruits.Count; i++)
# Python: for i, fruit in enumerate(fruits)
print("Fruits with index:")
for i, fruit in enumerate(fruits):
    print(f"  {i}: {fruit}")

print()

# enumerate() with custom start index
print("Fruits with index starting at 1:")
for i, fruit in enumerate(fruits, start=1):
    print(f"  {i}. {fruit}")

print()

# ============================================
# SECTION 4: for LOOPS - Dictionaries
# ============================================

print("=== SECTION 4: for LOOPS - Dictionaries ===\n")

person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

# Loop through keys only
print("Keys only:")
for key in person:
    print(f"  {key}")

print()

# Loop through values only
print("Values only:")
for value in person.values():
    print(f"  {value}")

print()

# Loop through key-value pairs (best way!)
# C#: foreach (var kvp in dict)
# Python: for key, value in dict.items()
print("Key-value pairs:")
for key, value in person.items():
    print(f"  {key}: {value}")

print()

# ============================================
# SECTION 5: while LOOPS
# ============================================

print("=== SECTION 5: while LOOPS ===\n")

# Basic while loop
# C#: while (count < 5) { ... count++; }
# Python: while count < 5: ... count += 1
count = 0

print("While loop (0-4):")
while count < 5:
    print(count, end=" ")
    count += 1  # Remember: no count++!
print()

print()

# while loop with condition
number = 1
print("Powers of 2 less than 100:")
while number < 100:
    print(number, end=" ")
    number *= 2
print()

print()

# ============================================
# SECTION 6: break and continue
# ============================================

print("=== SECTION 6: break and continue ===\n")

# break - exit the loop immediately
print("Finding first number divisible by 7:")
for i in range(1, 100):
    if i % 7 == 0:
        print(f"Found: {i}")
        break  # Exit loop after finding first match
# Loop stops here, doesn't continue to 100

print()

# continue - skip to next iteration
print("Odd numbers 0-10 (skip even):")
for i in range(11):
    if i % 2 == 0:
        continue  # Skip even numbers
    print(i, end=" ")
print()

print()

# Using break in while loop
print("Searching for target in list:")
numbers = [3, 7, 2, 9, 5, 1]
target = 9
index = 0

while index < len(numbers):
    if numbers[index] == target:
        print(f"Found {target} at index {index}")
        break
    index += 1

print()

# ============================================
# SECTION 7: else with Loops (Python-specific!)
# ============================================

print("=== SECTION 7: else with Loops (Python-specific!) ===\n")

# else with for loop - runs if loop completes WITHOUT break
# This doesn't exist in C#!
print("Searching for number 4:")
numbers = [1, 3, 5, 7, 9]
search = 4

for num in numbers:
    if num == search:
        print(f"Found {search}!")
        break
else:
    # This runs ONLY if loop completes without break
    print(f"{search} not found in list")

print()

# Example with break
print("Searching for number 5:")
search = 5

for num in numbers:
    if num == search:
        print(f"Found {search}!")
        break
else:
    print(f"{search} not found in list")  # Won't print (break was used)

print()

# else with while loop
print("Checking if number is prime:")
num = 17
is_prime = True

if num < 2:
    is_prime = False
else:
    i = 2
    while i * i <= num:
        if num % i == 0:
            is_prime = False
            print(f"{num} is divisible by {i}")
            break
        i += 1
    else:
        # This runs if loop completes without break
        print(f"{num} is prime!")

print()

# ============================================
# SECTION 8: NESTED LOOPS
# ============================================

print("=== SECTION 8: NESTED LOOPS ===\n")

# Multiplication table
print("Multiplication Table (1-5):")
for i in range(1, 6):
    for j in range(1, 6):
        print(f"{i*j:3}", end=" ")  # {:3} means 3 characters wide
    print()  # New line after each row

print()

# Pattern printing
print("Triangle pattern:")
for i in range(1, 6):
    for j in range(i):
        print("*", end=" ")
    print()  # New line after each row

print()

# ============================================
# SECTION 9: COMMON PATTERNS
# ============================================

print("=== SECTION 9: COMMON PATTERNS ===\n")

# Pattern 1: Sum of numbers
print("Pattern 1: Sum of 1-10")
total = 0
for i in range(1, 11):
    total += i
print(f"Sum: {total}")

print()

# Pattern 2: Find maximum
print("Pattern 2: Find maximum")
numbers = [3, 7, 2, 9, 1, 5]
max_num = numbers[0]  # Start with first element

for num in numbers:
    if num > max_num:
        max_num = num
print(f"Numbers: {numbers}")
print(f"Maximum: {max_num}")

print()

# Pattern 3: Find minimum
print("Pattern 3: Find minimum")
min_num = numbers[0]

for num in numbers:
    if num < min_num:
        min_num = num
print(f"Minimum: {min_num}")

print()

# Pattern 4: Count items
print("Pattern 4: Count apples")
fruits = ["apple", "banana", "apple", "orange", "apple", "grape"]
apple_count = 0

for fruit in fruits:
    if fruit == "apple":
        apple_count += 1
print(f"Fruits: {fruits}")
print(f"Apples: {apple_count}")

print()

# Pattern 5: Build a list
print("Pattern 5: Squares of numbers 1-10")
squares = []
for i in range(1, 11):
    squares.append(i * i)
print(f"Squares: {squares}")

print()

# Pattern 6: Filter a list
print("Pattern 6: Even numbers only")
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = []
for num in numbers:
    if num % 2 == 0:
        evens.append(num)
print(f"Original: {numbers}")
print(f"Evens: {evens}")

print()

# ============================================
# SECTION 10: PRACTICAL EXAMPLES
# ============================================

print("=== SECTION 10: PRACTICAL EXAMPLES ===\n")

# Example 1: Grade calculator for multiple students
print("Example 1: Grade Calculator")
students = {
    "Alice": 95,
    "Bob": 87,
    "Charlie": 72,
    "Diana": 68
}

for name, score in students.items():
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"
    print(f"{name}: {score} → Grade {grade}")

print()

# Example 2: FizzBuzz (Classic programming challenge)
print("Example 2: FizzBuzz (1-20)")
for i in range(1, 21):
    if i % 15 == 0:  # Divisible by both 3 and 5
        print("FizzBuzz", end=" ")
    elif i % 3 == 0:
        print("Fizz", end=" ")
    elif i % 5 == 0:
        print("Buzz", end=" ")
    else:
        print(i, end=" ")
print()

print()

# Example 3: Find all prime numbers up to 30
print("Example 3: Prime numbers up to 30")
primes = []
for num in range(2, 31):
    is_prime = True
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            is_prime = False
            break
    if is_prime:
        primes.append(num)
print(f"Primes: {primes}")

print()

# Example 4: Reverse a string
print("Example 4: Reverse a string")
text = "Python"
reversed_text = ""
for char in text:
    reversed_text = char + reversed_text  # Add to beginning
print(f"Original: {text}")
print(f"Reversed: {reversed_text}")

print()

# Example 5: Count vowels
print("Example 5: Count vowels")
text = "Hello World"
vowels = "aeiouAEIOU"
vowel_count = 0

for char in text:
    if char in vowels:
        vowel_count += 1
print(f"Text: '{text}'")
print(f"Vowels: {vowel_count}")

print()

# ============================================
# SUMMARY
# ============================================

print("=== SUMMARY ===\n")

summary = """
Python Control Flow for .NET Developers:

INDENTATION (CRITICAL!):
  - Use 4 spaces (not tabs!)
  - Defines code blocks (not curly braces)
  - Inconsistent indentation = ERROR

if/elif/else:
  - No parentheses: if x > 5: (not if (x > 5))
  - Colon required: if x > 5:
  - elif (not "else if")
  - Ternary: value_if_true if condition else value_if_false

for LOOPS:
  - range(5)        → 0, 1, 2, 3, 4
  - range(2, 5)     → 2, 3, 4
  - range(0, 10, 2) → 0, 2, 4, 6, 8
  - for item in list:
  - for i, item in enumerate(list):
  - for key, value in dict.items():

while LOOPS:
  - while condition:
  - Use += 1 (not ++!)
  - Be careful of infinite loops!

break and continue:
  - break: Exit loop immediately
  - continue: Skip to next iteration
  - Same as C#!

else with Loops (Python-only!):
  - Runs if loop completes WITHOUT break
  - for ... else:
  - while ... else:
  - Doesn't exist in C#

C# → Python Quick Reference:
  for (int i = 0; i < 5; i++)           → for i in range(5):
  foreach (var item in list)            → for item in list:
  for (int i = 0; i < list.Count; i++)  → for i, item in enumerate(list):
  if (condition) { }                    → if condition:
  else if (condition) { }               → elif condition:
  while (condition) { }                 → while condition:
  i++                                   → i += 1
  break;                                → break
  continue;                             → continue

Common Mistakes to Avoid:
  ✗ Forgetting colon:     if x > 5 print("hi")
  ✓ Correct:              if x > 5: print("hi")

  ✗ Using parentheses:    if (x > 5):
  ✓ Better:               if x > 5:

  ✗ Inconsistent indent:  Mixed spaces and tabs
  ✓ Correct:              Always 4 spaces

  ✗ Using ++:             i++
  ✓ Correct:              i += 1
"""

print(summary)

print("="*60)
print("Next: example_04_functions.py - Learn about functions!")
print("="*60)

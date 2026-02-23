# Lesson 1.7: List Comprehensions

## 🎯 What You'll Learn
- List comprehensions (like C# LINQ)
- Dictionary comprehensions
- Set comprehensions
- When to use comprehensions vs loops

---

## What are Comprehensions?

**Comprehensions** are a concise way to create collections in Python.

Think of them as Python's version of **C# LINQ**!

**C# LINQ:**
```csharp
var squares = numbers.Select(x => x * x).ToList();
var evens = numbers.Where(x => x % 2 == 0).ToList();
```

**Python Comprehensions:**
```python
squares = [x * x for x in numbers]
evens = [x for x in numbers if x % 2 == 0]
```

---

## List Comprehensions

### Basic Syntax

**Format:** `[expression for item in iterable]`

```python
# Traditional way (with loop)
squares = []
for x in range(5):
    squares.append(x * x)
print(squares)  # [0, 1, 4, 9, 16]

# List comprehension way
squares = [x * x for x in range(5)]
print(squares)  # [0, 1, 4, 9, 16]
```

**Line-by-line explanation:**
- `x * x` → Expression (what to add to list)
- `for x in range(5)` → Loop part
- `[]` → Creates a list

**More examples:**

```python
# Double all numbers
numbers = [1, 2, 3, 4, 5]
doubled = [x * 2 for x in numbers]
print(doubled)  # [2, 4, 6, 8, 10]

# Convert to uppercase
words = ["hello", "world", "python"]
upper = [word.upper() for word in words]
print(upper)  # ['HELLO', 'WORLD', 'PYTHON']

# Get lengths
lengths = [len(word) for word in words]
print(lengths)  # [5, 5, 6]
```

---

## List Comprehensions with Conditions

### Filtering (if condition)

**Format:** `[expression for item in iterable if condition]`

```python
# Traditional way
evens = []
for x in range(10):
    if x % 2 == 0:
        evens.append(x)
print(evens)  # [0, 2, 4, 6, 8]

# Comprehension way
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]
```

**More examples:**

```python
# Numbers greater than 5
numbers = [1, 8, 3, 10, 5, 12]
greater = [x for x in numbers if x > 5]
print(greater)  # [8, 10, 12]

# Words starting with 'p'
words = ["python", "java", "perl", "ruby"]
p_words = [w for w in words if w.startswith('p')]
print(p_words)  # ['python', 'perl']

# Positive numbers only
numbers = [-2, 3, -1, 5, -4, 8]
positive = [x for x in numbers if x > 0]
print(positive)  # [3, 5, 8]
```

**C# LINQ Comparison:**
```csharp
// C#
var evens = numbers.Where(x => x % 2 == 0).ToList();
var greater = numbers.Where(x => x > 5).ToList();

// Python
evens = [x for x in numbers if x % 2 == 0]
greater = [x for x in numbers if x > 5]
```

---

## Combining Expression and Condition

```python
# Square only even numbers
numbers = [1, 2, 3, 4, 5, 6]
squares = [x * x for x in numbers if x % 2 == 0]
print(squares)  # [4, 16, 36]

# Uppercase only words longer than 4 chars
words = ["hi", "hello", "world", "a", "python"]
long_upper = [w.upper() for w in words if len(w) > 4]
print(long_upper)  # ['HELLO', 'WORLD', 'PYTHON']
```

---

## if-else in Comprehensions

**Format:** `[expr_if_true if condition else expr_if_false for item in iterable]`

```python
# Label numbers as even or odd
numbers = [1, 2, 3, 4, 5]
labels = ["even" if x % 2 == 0 else "odd" for x in numbers]
print(labels)  # ['odd', 'even', 'odd', 'even', 'odd']

# Replace negative numbers with 0
numbers = [3, -1, 5, -2, 8]
positive = [x if x > 0 else 0 for x in numbers]
print(positive)  # [3, 0, 5, 0, 8]

# Absolute values
numbers = [-3, 5, -2, 8]
absolute = [x if x >= 0 else -x for x in numbers]
print(absolute)  # [3, 5, 2, 8]
```

**Explanation:**
- When you have `if-else`, it goes BEFORE the `for`
- When you have just `if` (filtering), it goes AFTER the `for`

```python
# Filtering (if only) - goes AFTER for
[x for x in numbers if x > 0]

# Conditional expression (if-else) - goes BEFORE for
[x if x > 0 else 0 for x in numbers]
```

---

## Nested Loops in Comprehensions

```python
# Traditional nested loops
result = []
for x in [1, 2, 3]:
    for y in [10, 20, 30]:
        result.append((x, y))
print(result)
# [(1, 10), (1, 20), (1, 30), (2, 10), (2, 20), (2, 30), (3, 10), (3, 20), (3, 30)]

# Comprehension way
result = [(x, y) for x in [1, 2, 3] for y in [10, 20, 30]]
print(result)
# Same output!
```

**Practical examples:**

```python
# Flatten 2D list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# All combinations
colors = ["red", "blue"]
sizes = ["S", "M", "L"]
combos = [(color, size) for color in colors for size in sizes]
print(combos)
# [('red', 'S'), ('red', 'M'), ('red', 'L'), ('blue', 'S'), ('blue', 'M'), ('blue', 'L')]
```

---

## Dictionary Comprehensions

**Format:** `{key: value for item in iterable}`

```python
# Create dictionary from range
squares = {x: x**2 for x in range(5)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Create from two lists
keys = ["name", "age", "city"]
values = ["Alice", 30, "NYC"]
person = {k: v for k, v in zip(keys, values)}
print(person)  # {'name': 'Alice', 'age': 30, 'city': 'NYC'}

# Transform existing dictionary
prices = {"apple": 0.50, "banana": 0.30, "orange": 0.60}
doubled = {fruit: price * 2 for fruit, price in prices.items()}
print(doubled)  # {'apple': 1.0, 'banana': 0.6, 'orange': 1.2}

# Swap keys and values
original = {"a": 1, "b": 2, "c": 3}
swapped = {v: k for k, v in original.items()}
print(swapped)  # {1: 'a', 2: 'b', 3: 'c'}
```

**With conditions:**

```python
# Only high prices
prices = {"apple": 0.50, "banana": 0.30, "orange": 0.60}
expensive = {fruit: price for fruit, price in prices.items() if price > 0.40}
print(expensive)  # {'apple': 0.5, 'orange': 0.6}

# Count word lengths
words = ["hello", "world", "python"]
lengths = {word: len(word) for word in words}
print(lengths)  # {'hello': 5, 'world': 5, 'python': 6}
```

**C# LINQ Comparison:**
```csharp
// C#
var squares = Enumerable.Range(0, 5)
    .ToDictionary(x => x, x => x * x);

// Python
squares = {x: x**2 for x in range(5)}
```

---

## Set Comprehensions

**Format:** `{expression for item in iterable}`

```python
# Create set of squares
squares = {x**2 for x in range(10)}
print(squares)  # {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

# Unique first letters
words = ["apple", "avocado", "banana", "blueberry"]
first_letters = {word[0] for word in words}
print(first_letters)  # {'a', 'b'}

# Unique lengths
words = ["hi", "hello", "world", "hi", "python"]
lengths = {len(word) for word in words}
print(lengths)  # {2, 5, 6}
```

**With conditions:**

```python
# Even numbers only
evens = {x for x in range(20) if x % 2 == 0}
print(evens)  # {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}

# Uppercase vowels from string
text = "hello world"
vowels = {char.upper() for char in text if char in 'aeiou'}
print(vowels)  # {'E', 'O'}
```

---

## Practical Examples

### Example 1: Extract Numbers from Strings

```python
strings = ["abc123", "xyz456", "def789"]
numbers = [int(''.join(c for c in s if c.isdigit())) for s in strings]
print(numbers)  # [123, 456, 789]
```

### Example 2: Grade Students

```python
scores = [85, 92, 78, 95, 88, 76]
grades = ["A" if s >= 90 else "B" if s >= 80 else "C" for s in scores]
print(grades)  # ['B', 'A', 'C', 'A', 'B', 'C']
```

### Example 3: Parse CSV Data

```python
csv_data = "name,age\nAlice,30\nBob,25"
lines = csv_data.split('\n')[1:]  # Skip header
people = [{"name": parts[0], "age": int(parts[1])}
          for line in lines
          for parts in [line.split(',')]]
print(people)
# [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
```

### Example 4: Matrix Operations

```python
# Transpose matrix
matrix = [
    [1, 2, 3],
    [4, 5, 6]
]
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
print(transposed)
# [[1, 4], [2, 5], [3, 6]]
```

---

## When to Use Comprehensions

### ✅ Use Comprehensions When:

- Creating a new list/dict/set from existing data
- Simple transformation or filtering
- Code is still readable (one line)

```python
# Good - Clear and concise
squares = [x**2 for x in range(10)]
evens = [x for x in numbers if x % 2 == 0]
```

### ❌ Don't Use Comprehensions When:

- Logic is complex (hard to read)
- Need multiple operations
- Need error handling

```python
# Bad - Too complex!
result = [x**2 if x > 0 else abs(x) if x < -5 else 0
          for x in numbers if x != 3]

# Better - Use regular loop
result = []
for x in numbers:
    if x != 3:
        if x > 0:
            result.append(x**2)
        elif x < -5:
            result.append(abs(x))
        else:
            result.append(0)
```

---

## Performance

Comprehensions are generally **faster** than loops!

```python
import time

# Using loop
start = time.time()
squares = []
for x in range(1000000):
    squares.append(x**2)
print(f"Loop: {time.time() - start:.4f}s")

# Using comprehension
start = time.time()
squares = [x**2 for x in range(1000000)]
print(f"Comprehension: {time.time() - start:.4f}s")

# Comprehension is usually faster!
```

---

## 💡 Key Takeaways

1. **List comprehension** → `[expr for item in iterable]`
2. **With filter** → `[expr for item in iterable if condition]`
3. **With if-else** → `[expr_if else expr_else for item in iterable]`
4. **Dict comprehension** → `{k: v for item in iterable}`
5. **Set comprehension** → `{expr for item in iterable}`
6. **Nested loops** → `[... for x in list1 for y in list2]`
7. **Like C# LINQ** → But more concise!
8. **Keep it simple** → Don't over-complicate

---

## ✏️ Practice Exercise

Create `comprehensions_practice.py`:

```python
# 1. Basic list comprehension
numbers = list(range(1, 11))
squares = [x**2 for x in numbers]
print(f"Squares: {squares}")

# 2. Filter with comprehension
evens = [x for x in numbers if x % 2 == 0]
print(f"Evens: {evens}")

# 3. Conditional expression
labels = ["even" if x % 2 == 0 else "odd" for x in numbers]
print(f"Labels: {labels}")

# 4. String manipulation
words = ["hello", "world", "python", "coding"]
upper = [w.upper() for w in words if len(w) > 5]
print(f"Long words (upper): {upper}")

# 5. Dictionary comprehension
word_lengths = {w: len(w) for w in words}
print(f"Word lengths: {word_lengths}")

# 6. Set comprehension
first_chars = {w[0] for w in words}
print(f"First characters: {first_chars}")

# 7. Nested comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(f"Flattened: {flat}")

# 8. Complex example - filter and transform
numbers = [-3, 5, -2, 8, -7, 10]
result = [x**2 if x > 0 else 0 for x in numbers]
print(f"Squares of positives: {result}")
```

**Run it:** `python comprehensions_practice.py`

---

## 🤔 Quick Quiz

1. What's the output of `[x*2 for x in range(3)]`?
   <details>
   <summary>Answer</summary>

   `[0, 2, 4]`
   </details>

2. What's the output of `[x for x in range(10) if x % 2 == 0]`?
   <details>
   <summary>Answer</summary>

   `[0, 2, 4, 6, 8]` - Even numbers from 0 to 9
   </details>

3. How do you create a dictionary where keys are numbers 0-4 and values are their squares?
   <details>
   <summary>Answer</summary>

   `{x: x**2 for x in range(5)}`
   </details>

4. What's wrong with: `{x**2 for x in range(5): x}`?
   <details>
   <summary>Answer</summary>

   Syntax error! Should be `{x: x**2 for x in range(5)}` for dict or `{x**2 for x in range(5)}` for set
   </details>

5. Where does the `if` go for filtering vs conditional expression?
   <details>
   <summary>Answer</summary>

   - Filtering: AFTER for → `[x for x in list if x > 0]`
   - Conditional: BEFORE for → `[x if x > 0 else 0 for x in list]`
   </details>

---

**Next Lesson:** [08_classes_oop.md](08_classes_oop.md) - Learn Object-Oriented Programming in Python!

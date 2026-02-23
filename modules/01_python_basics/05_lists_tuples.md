# Lesson 1.5: Lists and Tuples

## 🎯 What You'll Learn
- Creating and using lists (like C# List<T>)
- List indexing and slicing
- List methods
- Tuples (immutable lists)
- When to use lists vs tuples

---

## Lists - Python's Dynamic Arrays

### What is a List?

Like C#'s `List<T>`, but:
- No type declaration needed
- Can contain mixed types
- Dynamic size

**C# Comparison:**
```csharp
// C#
var numbers = new List<int> {1, 2, 3, 4, 5};
var names = new List<string> {"Alice", "Bob"};

// Python
numbers = [1, 2, 3, 4, 5]
names = ["Alice", "Bob"]
```

---

## Creating Lists

```python
# Empty list
empty = []

# List with initial values
numbers = [1, 2, 3, 4, 5]
names = ["Alice", "Bob", "Charlie"]

# Mixed types (not recommended, but possible)
mixed = [1, "hello", 3.14, True]

# Nested lists (2D array)
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Using list() constructor
numbers = list(range(5))  # [0, 1, 2, 3, 4]
chars = list("hello")     # ['h', 'e', 'l', 'l', 'o']
```

**Line-by-line explanation:**
- `[]` → Creates empty list
- `[1, 2, 3]` → List with values
- `list(range(5))` → Converts range to list
- `list("hello")` → Converts string to list of characters

---

## Accessing Elements (Indexing)

```python
fruits = ["apple", "banana", "orange", "grape"]

# Positive indexing (from start)
print(fruits[0])   # apple (first element)
print(fruits[1])   # banana
print(fruits[2])   # orange
print(fruits[3])   # grape (last element)

# Negative indexing (from end) - Python-specific!
print(fruits[-1])  # grape (last element)
print(fruits[-2])  # orange (second-to-last)
print(fruits[-3])  # banana
print(fruits[-4])  # apple (first element)
```

**Explanation:**
- Index starts at 0 (like C#)
- Negative index counts from end (unique to Python!)
- `fruits[-1]` is easier than `fruits[len(fruits)-1]`

**Visual:**
```
Index:     0        1         2        3
        ["apple", "banana", "orange", "grape"]
Negative: -4       -3        -2       -1
```

---

## Slicing (Getting Sublist)

**Format:** `list[start:stop:step]`

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Basic slicing
print(numbers[2:5])    # [2, 3, 4] (from index 2 to 4)
print(numbers[0:3])    # [0, 1, 2] (first 3 elements)

# Omit start (defaults to 0)
print(numbers[:3])     # [0, 1, 2] (first 3)

# Omit stop (defaults to end)
print(numbers[5:])     # [5, 6, 7, 8, 9] (from 5 to end)

# Negative indices
print(numbers[-3:])    # [7, 8, 9] (last 3)
print(numbers[:-2])    # [0, 1, 2, 3, 4, 5, 6, 7] (all except last 2)

# With step
print(numbers[::2])    # [0, 2, 4, 6, 8] (every 2nd element)
print(numbers[1::2])   # [1, 3, 5, 7, 9] (odd indices)

# Reverse list!
print(numbers[::-1])   # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

**Key Points:**
- `start` is inclusive, `stop` is exclusive
- Omit `start` → begins at 0
- Omit `stop` → goes to end
- `step` can be negative (reverse)

**C# LINQ Comparison:**
```csharp
// C#
var numbers = Enumerable.Range(0, 10).ToList();
var slice = numbers.Skip(2).Take(3).ToList();  // [2, 3, 4]

// Python
numbers = list(range(10))
slice = numbers[2:5]  # [2, 3, 4] - Much simpler!
```

---

## Modifying Lists

### Changing Elements

```python
fruits = ["apple", "banana", "orange"]

# Change single element
fruits[1] = "grape"
print(fruits)  # ["apple", "grape", "orange"]

# Change multiple elements via slice
fruits[0:2] = ["mango", "kiwi"]
print(fruits)  # ["mango", "kiwi", "orange"]
```

### Adding Elements

```python
# append() - add to end
fruits = ["apple", "banana"]
fruits.append("orange")
print(fruits)  # ["apple", "banana", "orange"]

# insert() - add at specific position
fruits.insert(1, "grape")
print(fruits)  # ["apple", "grape", "banana", "orange"]

# extend() - add multiple elements
fruits.extend(["mango", "kiwi"])
print(fruits)  # ["apple", "grape", "banana", "orange", "mango", "kiwi"]

# + operator - concatenate lists
more_fruits = fruits + ["pear", "peach"]
print(more_fruits)
```

**Line-by-line explanation:**
- `append(item)` → Adds one item to end (like C# `Add()`)
- `insert(index, item)` → Adds at position (like C# `Insert()`)
- `extend(list)` → Adds all items from another list (like C# `AddRange()`)
- `+` → Creates new list by concatenating

**C# Comparison:**
```csharp
// C#
var fruits = new List<string> {"apple", "banana"};
fruits.Add("orange");            // append
fruits.Insert(1, "grape");       // insert
fruits.AddRange(new[] {"mango"}); // extend

// Python
fruits = ["apple", "banana"]
fruits.append("orange")
fruits.insert(1, "grape")
fruits.extend(["mango"])
```

### Removing Elements

```python
fruits = ["apple", "banana", "orange", "grape"]

# remove() - remove by value
fruits.remove("banana")
print(fruits)  # ["apple", "orange", "grape"]

# pop() - remove by index (returns removed item)
last = fruits.pop()      # Removes last
print(last)              # "grape"
print(fruits)            # ["apple", "orange"]

second = fruits.pop(1)   # Removes at index 1
print(second)            # "orange"
print(fruits)            # ["apple"]

# del - delete by index or slice
numbers = [0, 1, 2, 3, 4, 5]
del numbers[2]           # Remove index 2
print(numbers)           # [0, 1, 3, 4, 5]

del numbers[1:3]         # Remove slice
print(numbers)           # [0, 4, 5]

# clear() - remove all elements
numbers.clear()
print(numbers)           # []
```

**Explanation:**
- `remove(value)` → Removes first occurrence of value
- `pop()` → Removes and returns last item
- `pop(index)` → Removes and returns item at index
- `del list[index]` → Deletes by index
- `clear()` → Empties the list

---

## List Methods

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

# Length
print(len(numbers))         # 8

# Count occurrences
print(numbers.count(1))     # 2 (1 appears twice)

# Find index
print(numbers.index(4))     # 2 (4 is at index 2)

# Sort in place (modifies original)
numbers.sort()
print(numbers)              # [1, 1, 2, 3, 4, 5, 6, 9]

# Reverse sort
numbers.sort(reverse=True)
print(numbers)              # [9, 6, 5, 4, 3, 2, 1, 1]

# Reverse in place
numbers.reverse()
print(numbers)              # [1, 1, 2, 3, 4, 5, 6, 9]

# Get sorted copy (doesn't modify original)
original = [3, 1, 4]
sorted_copy = sorted(original)
print(original)             # [3, 1, 4] (unchanged)
print(sorted_copy)          # [1, 3, 4] (new list)

# Min, max, sum
print(min(numbers))         # 1
print(max(numbers))         # 9
print(sum(numbers))         # 31
```

---

## Checking Membership

```python
fruits = ["apple", "banana", "orange"]

# Check if item exists
print("apple" in fruits)      # True
print("grape" in fruits)      # False

# Check if item doesn't exist
print("grape" not in fruits)  # True

# Use in if statement
if "banana" in fruits:
    print("We have bananas!")
```

**C# Comparison:**
```csharp
// C#
var fruits = new List<string> {"apple", "banana"};
bool hasApple = fruits.Contains("apple");

// Python
fruits = ["apple", "banana"]
has_apple = "apple" in fruits  # More readable!
```

---

## List Copying

**IMPORTANT:** Lists are references!

```python
# This does NOT create a copy!
list1 = [1, 2, 3]
list2 = list1        # list2 points to same list!
list2.append(4)
print(list1)         # [1, 2, 3, 4] - Changed!
print(list2)         # [1, 2, 3, 4]

# Create a real copy - Method 1: slice
list1 = [1, 2, 3]
list2 = list1[:]     # Creates copy
list2.append(4)
print(list1)         # [1, 2, 3] - Unchanged!
print(list2)         # [1, 2, 3, 4]

# Create a real copy - Method 2: list()
list3 = list(list1)  # Creates copy

# Create a real copy - Method 3: copy()
list4 = list1.copy() # Creates copy
```

**C# Comparison:**
```csharp
// C#
var list1 = new List<int> {1, 2, 3};
var list2 = list1;                    // Reference!
var list3 = new List<int>(list1);     // Copy
var list4 = list1.ToList();           // Copy

// Python
list1 = [1, 2, 3]
list2 = list1        # Reference
list3 = list1[:]     # Copy
list4 = list(list1)  # Copy
```

---

## Tuples - Immutable Lists

### What is a Tuple?

Like a list, but **immutable** (cannot be changed after creation)

```python
# Create tuple
coordinates = (10, 20)
rgb = (255, 128, 0)
person = ("Alice", 30, "NYC")

# Single element tuple (comma required!)
single = (5,)      # Tuple with one element
not_tuple = (5)    # This is just an int!

# Tuple without parentheses (tuple packing)
point = 10, 20     # Same as (10, 20)
```

**Explanation:**
- Use parentheses `()` for tuples
- Use square brackets `[]` for lists
- Tuples cannot be modified (no append, remove, etc.)

### Accessing Tuple Elements

```python
person = ("Alice", 30, "NYC")

# Indexing (same as lists)
print(person[0])    # Alice
print(person[1])    # 30
print(person[-1])   # NYC

# Slicing (same as lists)
print(person[0:2])  # ("Alice", 30)

# Unpacking
name, age, city = person
print(name)         # Alice
print(age)          # 30
print(city)         # NYC
```

### Tuples are Immutable

```python
person = ("Alice", 30)

# This works with lists
# But NOT with tuples!
person[0] = "Bob"   # ❌ Error! Tuples cannot be modified
person.append(40)   # ❌ Error! No append method
```

### When to Use Tuples vs Lists

**Use Lists when:**
- Data might change (add/remove items)
- You need list methods (append, remove, etc.)
- Working with collection of similar items

**Use Tuples when:**
- Data should NOT change (immutable)
- Returning multiple values from function
- Dictionary keys (lists can't be keys!)
- Faster than lists

**Examples:**

```python
# Tuple - coordinates don't change
point = (10, 20)

# List - shopping items can change
shopping = ["milk", "bread", "eggs"]
shopping.append("cheese")  # Can add items

# Tuple - RGB color (fixed)
color = (255, 128, 0)

# List - todo items (can add/remove)
todos = ["task1", "task2"]
todos.remove("task1")  # Can remove
```

---

## Tuple Packing and Unpacking

### Packing

```python
# Pack values into tuple
person = "Alice", 30, "NYC"
print(person)  # ("Alice", 30, "NYC")
```

### Unpacking

```python
# Unpack tuple into variables
person = ("Alice", 30, "NYC")
name, age, city = person

print(name)   # Alice
print(age)    # 30
print(city)   # NYC

# Swap variables (Python trick!)
a = 5
b = 10
a, b = b, a   # Swap!
print(a)      # 10
print(b)      # 5
```

**C# Comparison:**
```csharp
// C# 7.0+ (Tuple unpacking)
var person = ("Alice", 30, "NYC");
var (name, age, city) = person;

// Swap
(a, b) = (b, a);

// Python - Same concept!
person = ("Alice", 30, "NYC")
name, age, city = person
a, b = b, a
```

---

## Common Patterns

### Looping Through Lists

```python
fruits = ["apple", "banana", "orange"]

# Loop through items
for fruit in fruits:
    print(fruit)

# Loop with index
for i in range(len(fruits)):
    print(f"{i}: {fruits[i]}")

# Loop with index and value (better!)
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
```

### List as Stack (LIFO - Last In, First Out)

```python
stack = []

# Push
stack.append(1)
stack.append(2)
stack.append(3)
print(stack)  # [1, 2, 3]

# Pop
item = stack.pop()
print(item)   # 3
print(stack)  # [1, 2]
```

### List as Queue (FIFO - First In, First Out)

```python
from collections import deque

queue = deque()

# Enqueue
queue.append(1)
queue.append(2)
queue.append(3)

# Dequeue
item = queue.popleft()
print(item)   # 1
```

---

## 💡 Key Takeaways

1. **Lists** → `[]`, mutable, like C# `List<T>`
2. **Tuples** → `()`, immutable, for fixed data
3. **Negative indexing** → `list[-1]` is last element
4. **Slicing** → `list[start:stop:step]`
5. **append()** → Add to end
6. **extend()** → Add multiple items
7. **pop()** → Remove and return
8. **in/not in** → Check membership
9. **Copying** → Use `list[:]` or `list.copy()`
10. **Unpacking** → `a, b, c = tuple`

---

## ✏️ Practice Exercise

Create `lists_tuples_practice.py`:

```python
# 1. Create and access
fruits = ["apple", "banana", "orange", "grape"]
print(f"First: {fruits[0]}")
print(f"Last: {fruits[-1]}")
print(f"First two: {fruits[:2]}")

# 2. Modify list
fruits.append("mango")
fruits.insert(1, "kiwi")
print(fruits)

# 3. Remove items
fruits.remove("banana")
last = fruits.pop()
print(f"Removed: {last}")
print(fruits)

# 4. List operations
numbers = [5, 2, 8, 1, 9]
print(f"Length: {len(numbers)}")
print(f"Sum: {sum(numbers)}")
print(f"Max: {max(numbers)}")
numbers.sort()
print(f"Sorted: {numbers}")

# 5. Slicing
nums = list(range(10))
print(f"Even indices: {nums[::2]}")
print(f"Reversed: {nums[::-1]}")

# 6. Tuple
person = ("Alice", 30, "NYC")
name, age, city = person
print(f"{name} is {age} years old from {city}")

# 7. Swap values
a = 5
b = 10
print(f"Before: a={a}, b={b}")
a, b = b, a
print(f"After: a={a}, b={b}")
```

**Run it:** `python lists_tuples_practice.py`

---

## 🤔 Quick Quiz

1. How do you access the last element of a list?
   <details>
   <summary>Answer</summary>

   `list[-1]`
   </details>

2. What does `numbers[2:5]` return for `numbers = [0,1,2,3,4,5,6]`?
   <details>
   <summary>Answer</summary>

   `[2, 3, 4]` - Start is inclusive, stop is exclusive
   </details>

3. What's the difference between `append()` and `extend()`?
   <details>
   <summary>Answer</summary>

   - `append(item)` adds one item
   - `extend(list)` adds all items from another list
   </details>

4. How do you reverse a list?
   <details>
   <summary>Answer</summary>

   `list.reverse()` or `list[::-1]`
   </details>

5. Can you modify a tuple after creation?
   <details>
   <summary>Answer</summary>

   No! Tuples are immutable.
   </details>

---

**Next Lesson:** [06_dictionaries_sets.md](06_dictionaries_sets.md) - Learn about dictionaries and sets!

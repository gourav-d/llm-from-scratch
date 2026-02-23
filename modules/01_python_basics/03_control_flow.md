# Lesson 1.3: Control Flow (if/else, Loops)

## 🎯 What You'll Learn
- if/elif/else statements
- for loops
- while loops
- break and continue
- Python's indentation rules

---

## ⚠️ CRITICAL: Indentation in Python

**THIS IS DIFFERENT FROM C#!**

### C# uses curly braces `{}`
```csharp
if (x > 5)
{
    Console.WriteLine("Greater");
}
```

### Python uses INDENTATION (spaces/tabs)
```python
if x > 5:
    print("Greater")
# No curly braces!
```

**Rules:**
1. Use **4 spaces** for each indentation level (standard)
2. **Must be consistent** - mixing spaces and tabs causes errors
3. Indentation defines code blocks

---

## if/elif/else Statements

### Basic if Statement

```python
age = 18

if age >= 18:
    print("You are an adult")
    print("You can vote")
```

**Line-by-line explanation:**
- `if age >= 18:` → Condition, ends with colon `:`
- Next line is indented by 4 spaces
- All indented lines run if condition is True
- No parentheses needed around condition!

**C# Comparison:**
```csharp
// C#
if (age >= 18)
{
    Console.WriteLine("You are an adult");
}

// Python
if age >= 18:
    print("You are an adult")
```

### if/else

```python
age = 16

if age >= 18:
    print("You are an adult")
else:
    print("You are a minor")
```

**Explanation:**
- `else:` → No condition needed, ends with `:`
- Code under `else` runs if condition is False

### if/elif/else (Multiple Conditions)

```python
score = 85

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
else:
    print("Grade: F")
```

**Line-by-line explanation:**
- `elif` → Short for "else if" (C#'s `else if`)
- Checks conditions in order from top to bottom
- Stops at first True condition
- `else` runs if all conditions are False

**C# Comparison:**
```csharp
// C#
if (score >= 90)
    Console.WriteLine("Grade: A");
else if (score >= 80)
    Console.WriteLine("Grade: B");
else
    Console.WriteLine("Grade: F");

// Python
if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
else:
    print("Grade: F")
```

### Nested if Statements

```python
age = 20
has_license = True

if age >= 18:
    if has_license:
        print("You can drive")
    else:
        print("You need a license")
else:
    print("Too young to drive")
```

**Better way - combine conditions:**
```python
age = 20
has_license = True

if age >= 18 and has_license:
    print("You can drive")
elif age >= 18:
    print("You need a license")
else:
    print("Too young to drive")
```

### Ternary Operator (One-line if/else)

```python
# Regular if/else
age = 20
if age >= 18:
    status = "Adult"
else:
    status = "Minor"

# Ternary operator (one line!)
status = "Adult" if age >= 18 else "Minor"
```

**Format:** `value_if_true if condition else value_if_false`

**C# Comparison:**
```csharp
// C#
string status = age >= 18 ? "Adult" : "Minor";

// Python
status = "Adult" if age >= 18 else "Minor"
```

---

## for Loops

### Looping Through a Range

```python
# Print numbers 0 to 4
for i in range(5):
    print(i)
# Output: 0, 1, 2, 3, 4
```

**Line-by-line explanation:**
- `for i in range(5):` → Loop variable `i`, `range(5)` creates 0,1,2,3,4
- Ends with colon `:`
- Indented code runs for each value

**range() function:**
```python
# range(stop) - from 0 to stop-1
for i in range(5):
    print(i)       # 0, 1, 2, 3, 4

# range(start, stop) - from start to stop-1
for i in range(2, 5):
    print(i)       # 2, 3, 4

# range(start, stop, step) - with step size
for i in range(0, 10, 2):
    print(i)       # 0, 2, 4, 6, 8
```

**C# Comparison:**
```csharp
// C#
for (int i = 0; i < 5; i++)
{
    Console.WriteLine(i);
}

// Python
for i in range(5):
    print(i)
```

### Looping Through a List

```python
fruits = ["apple", "banana", "orange"]

# Loop through each item
for fruit in fruits:
    print(fruit)
# Output:
# apple
# banana
# orange
```

**Explanation:**
- `for fruit in fruits:` → `fruit` takes each value from `fruits`
- Variable name can be anything: `for x in fruits:` works too
- But use meaningful names: `fruit` is better than `x`

**With Index (like C#'s for loop):**
```python
fruits = ["apple", "banana", "orange"]

# Get both index and value
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
# Output:
# 0: apple
# 1: banana
# 2: orange
```

**Line-by-line explanation:**
- `enumerate(fruits)` → Returns pairs of (index, value)
- `i, fruit` → Unpacks the pair into two variables

**C# Comparison:**
```csharp
// C#
var fruits = new List<string> {"apple", "banana"};

foreach (var fruit in fruits)
{
    Console.WriteLine(fruit);
}

// With index
for (int i = 0; i < fruits.Count; i++)
{
    Console.WriteLine($"{i}: {fruits[i]}");
}

// Python
for fruit in fruits:
    print(fruit)

# With index
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
```

### Looping Through a Dictionary

```python
person = {"name": "Alice", "age": 30, "city": "NYC"}

# Loop through keys
for key in person:
    print(key)
# Output: name, age, city

# Loop through values
for value in person.values():
    print(value)
# Output: Alice, 30, NYC

# Loop through key-value pairs
for key, value in person.items():
    print(f"{key}: {value}")
# Output:
# name: Alice
# age: 30
# city: NYC
```

**Explanation:**
- `.values()` → Gets all values
- `.items()` → Gets (key, value) pairs
- `key, value` → Unpacks the pair

---

## while Loops

Runs while condition is True

```python
count = 0

while count < 5:
    print(count)
    count += 1
# Output: 0, 1, 2, 3, 4
```

**Line-by-line explanation:**
- `while count < 5:` → Checks condition
- Loop continues as long as condition is True
- `count += 1` → Increment (remember, no `count++`!)
- Don't forget to update the counter or you'll have infinite loop!

**C# Comparison:**
```csharp
// C#
int count = 0;
while (count < 5)
{
    Console.WriteLine(count);
    count++;
}

// Python
count = 0
while count < 5:
    print(count)
    count += 1
```

### Infinite Loop (with break)

```python
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == "quit":
        break
    print(f"You entered: {user_input}")
```

**Explanation:**
- `while True:` → Loop forever
- `input()` → Gets user input (like C#'s `Console.ReadLine()`)
- `break` → Exits the loop immediately

---

## break and continue

### break - Exit the Loop

```python
# Find first number divisible by 7
for i in range(1, 100):
    if i % 7 == 0:
        print(f"Found: {i}")
        break
# Output: Found: 7
# Loop stops after finding first match
```

**Explanation:**
- `break` → Exits the loop completely
- Rest of loop doesn't run

### continue - Skip to Next Iteration

```python
# Print odd numbers only
for i in range(10):
    if i % 2 == 0:
        continue  # Skip even numbers
    print(i)
# Output: 1, 3, 5, 7, 9
```

**Explanation:**
- `continue` → Skips rest of current iteration
- Jumps to next iteration of loop

**C# Comparison:**
```csharp
// C# - Same keywords!
for (int i = 0; i < 10; i++)
{
    if (i % 2 == 0)
        continue;
    Console.WriteLine(i);
}

// Python
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)
```

### else with Loops (Python-specific!)

Python has a unique feature - `else` with loops:

```python
# Search for a number
numbers = [1, 3, 5, 7, 9]
search = 4

for num in numbers:
    if num == search:
        print("Found!")
        break
else:
    # This runs if loop completes WITHOUT break
    print("Not found!")
# Output: Not found!
```

**Explanation:**
- `else` after loop → Runs if loop completes normally
- Does NOT run if `break` was used
- Useful for search operations

**This doesn't exist in C#!**

---

## Common Patterns

### Pattern 1: Sum of Numbers
```python
# Sum of 1 to 10
total = 0
for i in range(1, 11):
    total += i
print(total)  # 55
```

### Pattern 2: Find Maximum
```python
numbers = [3, 7, 2, 9, 1]
max_num = numbers[0]

for num in numbers:
    if num > max_num:
        max_num = num
print(max_num)  # 9
```

### Pattern 3: Count Items
```python
fruits = ["apple", "banana", "apple", "orange", "apple"]
apple_count = 0

for fruit in fruits:
    if fruit == "apple":
        apple_count += 1
print(apple_count)  # 3
```

---

## 💡 Key Takeaways

1. **Indentation is CRITICAL** → Use 4 spaces
2. **Colon `:` after conditions** → if, elif, else, for, while
3. **No parentheses needed** → `if x > 5:` not `if (x > 5):`
4. **No curly braces** → Indentation defines blocks
5. **`elif` not `else if`** → Python shorthand
6. **`range()` for numbers** → Like C#'s `for (i = 0; i < n; i++)`
7. **`enumerate()` for index** → When you need both index and value
8. **`break` and `continue`** → Same as C#

---

## ✏️ Practice Exercise

Create `control_flow_practice.py`:

```python
# 1. Grade calculator
print("=== Grade Calculator ===")
score = 85

if score >= 90:
    print("A")
elif score >= 80:
    print("B")
elif score >= 70:
    print("C")
else:
    print("F")

# 2. Print even numbers 0-20
print("\n=== Even Numbers ===")
for i in range(21):
    if i % 2 == 0:
        print(i, end=" ")  # Print on same line
print()  # New line

# 3. Countdown
print("\n=== Countdown ===")
count = 5
while count > 0:
    print(count)
    count -= 1
print("Blast off!")

# 4. Find first 'a' in list
print("\n=== Find Item ===")
letters = ['x', 'y', 'a', 'b', 'c']
for i, letter in enumerate(letters):
    if letter == 'a':
        print(f"Found 'a' at index {i}")
        break
else:
    print("'a' not found")

# 5. Sum of numbers
print("\n=== Sum ===")
total = 0
for i in range(1, 11):
    total += i
print(f"Sum of 1-10: {total}")
```

**Run it:** `python control_flow_practice.py`

---

## 🤔 Quick Quiz

1. What's wrong with this code?
   ```python
   if x > 5
       print("Greater")
   ```
   <details>
   <summary>Answer</summary>

   Missing colon `:` after condition!
   ```python
   if x > 5:
       print("Greater")
   ```
   </details>

2. How do you loop from 5 to 10 (inclusive)?
   <details>
   <summary>Answer</summary>

   ```python
   for i in range(5, 11):  # 11 because stop is exclusive
       print(i)
   ```
   </details>

3. What does this output?
   ```python
   for i in range(5):
       if i == 3:
           break
       print(i)
   ```
   <details>
   <summary>Answer</summary>

   `0 1 2` - Breaks before printing 3
   </details>

4. What's the Python equivalent of C#'s `else if`?
   <details>
   <summary>Answer</summary>

   `elif`
   </details>

5. How do you loop through a list with both index and value?
   <details>
   <summary>Answer</summary>

   ```python
   for i, value in enumerate(my_list):
       print(i, value)
   ```
   </details>

---

**Next Lesson:** [04_functions.md](04_functions.md) - Learn about functions and code reusability!

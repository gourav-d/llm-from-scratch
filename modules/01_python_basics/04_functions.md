# Lesson 1.4: Functions

## 🎯 What You'll Learn
- How to define and call functions
- Parameters and arguments
- Return values
- Default parameters
- Lambda expressions (like C# lambdas)
- Scope and lifetime

---

## Defining Functions

### C# vs Python Syntax

**C#:**
```csharp
public int Add(int a, int b)
{
    return a + b;
}
```

**Python:**
```python
def add(a, b):
    return a + b
```

**Key Differences:**
- Use `def` keyword (define)
- No return type declaration
- No access modifiers (public, private)
- Colon `:` after parameters
- Indentation for function body

---

## Basic Function

```python
# Define a function
def greet():
    print("Hello, World!")

# Call the function
greet()
# Output: Hello, World!
```

**Line-by-line explanation:**
- `def greet():` → Define function named 'greet', no parameters
- Colon `:` marks start of function body
- Indented code is the function body
- `greet()` → Call the function (parentheses required!)

---

## Functions with Parameters

```python
def greet(name):
    print(f"Hello, {name}!")

# Call with argument
greet("Alice")    # Hello, Alice!
greet("Bob")      # Hello, Bob!
```

**Explanation:**
- `name` is a parameter (placeholder)
- `"Alice"` is an argument (actual value)
- No type declaration needed!

### Multiple Parameters

```python
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")

add(5, 3)     # 5 + 3 = 8
add(10, 20)   # 10 + 20 = 30
```

**C# Comparison:**
```csharp
// C#
public void Add(int a, int b)
{
    int result = a + b;
    Console.WriteLine($"{a} + {b} = {result}");
}

// Python
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")
```

---

## Return Values

### Single Return Value

```python
def add(a, b):
    return a + b

result = add(5, 3)
print(result)  # 8
```

**Explanation:**
- `return` sends value back to caller
- No return type declaration
- Can return any type

### Multiple Return Values

**Python has a cool feature - return multiple values!**

```python
def get_min_max(numbers):
    return min(numbers), max(numbers)

# Unpack the tuple
minimum, maximum = get_min_max([1, 5, 3, 9, 2])
print(f"Min: {minimum}, Max: {maximum}")
# Output: Min: 1, Max: 9
```

**Line-by-line explanation:**
- `return min(numbers), max(numbers)` → Returns a tuple (2 values)
- `minimum, maximum = ...` → Unpacks the tuple into 2 variables
- This doesn't exist in C# the same way!

**C# Equivalent (using tuple):**
```csharp
// C# 7.0+
public (int min, int max) GetMinMax(List<int> numbers)
{
    return (numbers.Min(), numbers.Max());
}

var (minimum, maximum) = GetMinMax(numbers);
```

### Early Return

```python
def is_adult(age):
    if age >= 18:
        return True
    return False

# Better way (more Pythonic)
def is_adult(age):
    return age >= 18
```

---

## Default Parameters

Provide default values for parameters

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Use default
print(greet("Alice"))           # Hello, Alice!

# Override default
print(greet("Bob", "Hi"))       # Hi, Bob!
print(greet("Charlie", "Hey"))  # Hey, Charlie!
```

**Explanation:**
- `greeting="Hello"` → Default value
- If not provided, uses "Hello"
- If provided, uses given value

**C# Comparison:**
```csharp
// C#
public string Greet(string name, string greeting = "Hello")
{
    return $"{greeting}, {name}!";
}

// Python - Same concept!
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
```

### Multiple Default Parameters

```python
def create_user(name, age=18, active=True):
    return {
        "name": name,
        "age": age,
        "active": active
    }

user1 = create_user("Alice")
# {"name": "Alice", "age": 18, "active": True}

user2 = create_user("Bob", 25)
# {"name": "Bob", "age": 25, "active": True}

user3 = create_user("Charlie", 30, False)
# {"name": "Charlie", "age": 30, "active": False}
```

**Rule:** Parameters with defaults must come AFTER parameters without defaults!

```python
# ✅ Correct
def func(a, b, c=10):
    pass

# ❌ Wrong!
def func(a, b=5, c):  # Error! Non-default after default
    pass
```

---

## Named Arguments (Keyword Arguments)

Call functions using parameter names

```python
def describe_pet(animal, name, age):
    print(f"{name} is a {age}-year-old {animal}")

# Positional arguments (order matters)
describe_pet("dog", "Buddy", 3)
# Buddy is a 3-year-old dog

# Named arguments (order doesn't matter!)
describe_pet(name="Buddy", age=3, animal="dog")
# Buddy is a 3-year-old dog

describe_pet(age=3, animal="dog", name="Buddy")
# Same result!

# Mix positional and named
describe_pet("dog", age=3, name="Buddy")
```

**Explanation:**
- Named arguments make code more readable
- Order doesn't matter with named arguments
- Can mix positional and named (positional must come first!)

---

## Variable Number of Arguments

### *args - Variable Positional Arguments

```python
def add_all(*numbers):
    total = 0
    for num in numbers:
        total += num
    return total

print(add_all(1, 2, 3))           # 6
print(add_all(10, 20, 30, 40))    # 100
print(add_all(5))                 # 5
```

**Explanation:**
- `*numbers` → Collects all arguments into a tuple
- Can pass any number of arguments
- Like C#'s `params` keyword

**C# Comparison:**
```csharp
// C#
public int AddAll(params int[] numbers)
{
    return numbers.Sum();
}

// Python
def add_all(*numbers):
    return sum(numbers)
```

### **kwargs - Variable Keyword Arguments

```python
def create_profile(**info):
    for key, value in info.items():
        print(f"{key}: {value}")

create_profile(name="Alice", age=30, city="NYC")
# Output:
# name: Alice
# age: 30
# city: NYC
```

**Explanation:**
- `**info` → Collects named arguments into a dictionary
- Can pass any number of named arguments
- Very flexible!

### Combining All Types

```python
def full_example(a, b, c=10, *args, **kwargs):
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c: {c}")
    print(f"args: {args}")
    print(f"kwargs: {kwargs}")

full_example(1, 2, 3, 4, 5, x=10, y=20)
# Output:
# a: 1
# b: 2
# c: 3
# args: (4, 5)
# kwargs: {'x': 10, 'y': 20}
```

**Order:** `positional, *args, default, **kwargs`

---

## Lambda Functions

Anonymous functions (like C# lambdas)

### Basic Lambda

```python
# Regular function
def square(x):
    return x * x

# Lambda (one-line function)
square = lambda x: x * x

print(square(5))  # 25
```

**Format:** `lambda parameters: expression`

**C# Comparison:**
```csharp
// C#
Func<int, int> square = x => x * x;

// Python
square = lambda x: x * x
```

### Lambda with Multiple Parameters

```python
# Lambda with 2 parameters
add = lambda a, b: a + b
print(add(5, 3))  # 8

# Lambda with 3 parameters
multiply = lambda x, y, z: x * y * z
print(multiply(2, 3, 4))  # 24
```

### Using Lambdas with Built-in Functions

**Sorting with custom key:**

```python
# Sort by length
words = ["apple", "pie", "banana", "cherry"]
sorted_words = sorted(words, key=lambda x: len(x))
print(sorted_words)
# ['pie', 'apple', 'banana', 'cherry']

# Sort tuples by second element
pairs = [(1, 5), (3, 2), (2, 8)]
sorted_pairs = sorted(pairs, key=lambda x: x[1])
print(sorted_pairs)
# [(3, 2), (1, 5), (2, 8)]
```

**Filtering:**

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Get even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# Get numbers > 5
greater = list(filter(lambda x: x > 5, numbers))
print(greater)  # [6, 7, 8, 9, 10]
```

**Mapping:**

```python
numbers = [1, 2, 3, 4, 5]

# Square each number
squares = list(map(lambda x: x * x, numbers))
print(squares)  # [1, 4, 9, 16, 25]

# Double each number
doubled = list(map(lambda x: x * 2, numbers))
print(doubled)  # [2, 4, 6, 8, 10]
```

**C# LINQ Comparison:**
```csharp
// C#
var numbers = new[] {1, 2, 3, 4, 5};
var evens = numbers.Where(x => x % 2 == 0);
var squares = numbers.Select(x => x * x);

// Python
numbers = [1, 2, 3, 4, 5]
evens = filter(lambda x: x % 2 == 0, numbers)
squares = map(lambda x: x * x, numbers)

// But Python list comprehensions are better!
evens = [x for x in numbers if x % 2 == 0]
squares = [x * x for x in numbers]
```

---

## Scope and Lifetime

### Local Scope

```python
def my_function():
    x = 10  # Local variable
    print(x)

my_function()  # 10
print(x)       # Error! x doesn't exist here
```

### Global Scope

```python
x = 10  # Global variable

def my_function():
    print(x)  # Can read global

my_function()  # 10
```

### Modifying Global Variables

```python
x = 10

def modify_global():
    global x  # Declare we want to modify global x
    x = 20

print(x)         # 10
modify_global()
print(x)         # 20
```

**Warning:** Modifying global variables is generally bad practice!

**Better approach - return values:**

```python
x = 10

def get_new_value(old_value):
    return old_value + 10

x = get_new_value(x)  # x = 20
```

---

## Docstrings

Document your functions (like C# XML comments)

```python
def calculate_area(length, width):
    """
    Calculate the area of a rectangle.

    Parameters:
        length (float): The length of the rectangle
        width (float): The width of the rectangle

    Returns:
        float: The area of the rectangle
    """
    return length * width

# Access docstring
print(calculate_area.__doc__)
```

**Explanation:**
- Triple quotes `"""..."""` for docstrings
- First thing after function definition
- Accessed via `function.__doc__`
- Good IDEs show docstrings as hints

---

## 💡 Key Takeaways

1. **`def` keyword** → Define functions
2. **No type declarations** → Python figures it out
3. **Return multiple values** → Returns tuple, unpack it
4. **Default parameters** → Provide default values
5. **Named arguments** → Order doesn't matter
6. **`*args`** → Variable positional arguments (tuple)
7. **`**kwargs`** → Variable keyword arguments (dict)
8. **Lambda** → `lambda x: x * 2` for simple functions
9. **Docstrings** → Document with `"""..."""`

---

## ✏️ Practice Exercise

Create `functions_practice.py`:

```python
# 1. Basic function
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))


# 2. Function with multiple parameters
def rectangle_area(length, width):
    return length * width

print(f"Area: {rectangle_area(5, 3)}")


# 3. Function with default parameter
def power(base, exponent=2):
    return base ** exponent

print(power(5))      # 25 (5^2)
print(power(5, 3))   # 125 (5^3)


# 4. Return multiple values
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)

nums = [1, 5, 3, 9, 2]
minimum, maximum, total = get_stats(nums)
print(f"Min: {minimum}, Max: {maximum}, Sum: {total}")


# 5. Variable arguments
def multiply_all(*numbers):
    result = 1
    for num in numbers:
        result *= num
    return result

print(multiply_all(2, 3, 4))  # 24


# 6. Lambda function
square = lambda x: x * x
print(f"Square of 7: {square(7)}")

# Sort by absolute value
numbers = [-5, 2, -8, 3, -1]
sorted_nums = sorted(numbers, key=lambda x: abs(x))
print(f"Sorted by absolute value: {sorted_nums}")


# 7. Filter and map
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
squares = list(map(lambda x: x ** 2, numbers))
print(f"Evens: {evens}")
print(f"Squares: {squares}")
```

**Run it:** `python functions_practice.py`

---

## 🤔 Quick Quiz

1. What keyword is used to define a function in Python?
   <details>
   <summary>Answer</summary>

   `def`
   </details>

2. How do you return multiple values from a function?
   <details>
   <summary>Answer</summary>

   ```python
   return value1, value2, value3  # Returns a tuple
   ```
   </details>

3. What's wrong with this function definition?
   ```python
   def func(a, b=5, c):
       pass
   ```
   <details>
   <summary>Answer</summary>

   Parameters with defaults must come AFTER parameters without defaults.
   Should be: `def func(a, c, b=5):`
   </details>

4. What does `*args` do?
   <details>
   <summary>Answer</summary>

   Collects variable number of positional arguments into a tuple
   </details>

5. Write a lambda function that doubles a number:
   <details>
   <summary>Answer</summary>

   ```python
   double = lambda x: x * 2
   ```
   </details>

---

**Next Lesson:** [05_lists_tuples.md](05_lists_tuples.md) - Learn about lists and tuples!

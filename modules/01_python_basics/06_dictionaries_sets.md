# Lesson 1.6: Dictionaries and Sets

## 🎯 What You'll Learn
- Dictionaries (like C# Dictionary<K,V>)
- Creating and accessing dictionaries
- Dictionary methods
- Sets (unique collections)
- Set operations

---

## Dictionaries - Key-Value Pairs

### What is a Dictionary?

Like C#'s `Dictionary<TKey, TValue>`:
- Stores key-value pairs
- Fast lookup by key
- Keys must be unique
- Unordered (before Python 3.7) / Ordered (Python 3.7+)

**C# Comparison:**
```csharp
// C#
var person = new Dictionary<string, object>
{
    {"name", "Alice"},
    {"age", 30},
    {"city", "NYC"}
};

// Python
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}
```

---

## Creating Dictionaries

```python
# Empty dictionary
empty = {}

# Dictionary with initial values
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

# Using dict() constructor
person2 = dict(name="Bob", age=25, city="LA")

# From list of tuples
pairs = [("name", "Charlie"), ("age", 35)]
person3 = dict(pairs)

print(person)   # {'name': 'Alice', 'age': 30, 'city': 'NYC'}
```

**Line-by-line explanation:**
- `{}` → Empty dictionary (curly braces!)
- `{"key": value}` → Key-value syntax
- Keys are usually strings, but can be any immutable type
- Values can be any type

---

## Accessing Dictionary Values

```python
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

# Access by key
print(person["name"])    # Alice
print(person["age"])     # 30

# Using get() - safer!
print(person.get("name"))        # Alice
print(person.get("country"))     # None (doesn't error!)
print(person.get("country", "USA"))  # USA (default value)

# Error if key doesn't exist
print(person["country"])  # ❌ KeyError!
```

**Explanation:**
- `dict["key"]` → Access value, errors if key missing
- `dict.get("key")` → Returns `None` if key missing
- `dict.get("key", default)` → Returns default if key missing

**C# Comparison:**
```csharp
// C#
var person = new Dictionary<string, object>();
var name = person["name"];           // Error if missing
person.TryGetValue("name", out var value);  // Safe

// Python
person = {"name": "Alice"}
name = person["name"]                # Error if missing
name = person.get("name")            # Safe (returns None)
```

---

## Modifying Dictionaries

### Adding and Updating

```python
person = {"name": "Alice", "age": 30}

# Add new key-value
person["city"] = "NYC"
print(person)  # {'name': 'Alice', 'age': 30, 'city': 'NYC'}

# Update existing value
person["age"] = 31
print(person)  # {'name': 'Alice', 'age': 31, 'city': 'NYC'}

# Update multiple values
person.update({"age": 32, "country": "USA"})
print(person)
# {'name': 'Alice', 'age': 32, 'city': 'NYC', 'country': 'USA'}

# Merge dictionaries (Python 3.9+)
defaults = {"theme": "dark", "lang": "en"}
settings = {"lang": "fr"}
merged = defaults | settings
print(merged)  # {'theme': 'dark', 'lang': 'fr'}
```

### Removing Items

```python
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

# Remove by key (returns value)
age = person.pop("age")
print(age)     # 30
print(person)  # {'name': 'Alice', 'city': 'NYC'}

# Remove by key (no return)
del person["city"]
print(person)  # {'name': 'Alice'}

# Remove and return last item (Python 3.7+)
person = {"name": "Alice", "age": 30}
item = person.popitem()
print(item)    # ('age', 30) - Returns tuple!

# Clear all items
person.clear()
print(person)  # {}
```

---

## Dictionary Methods

```python
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

# Get all keys
keys = person.keys()
print(keys)     # dict_keys(['name', 'age', 'city'])
print(list(keys))  # ['name', 'age', 'city']

# Get all values
values = person.values()
print(list(values))  # ['Alice', 30, 'NYC']

# Get all key-value pairs
items = person.items()
print(list(items))
# [('name', 'Alice'), ('age', 30), ('city', 'NYC')]

# Check if key exists
print("name" in person)      # True
print("country" in person)   # False

# Get number of items
print(len(person))  # 3

# Copy dictionary
person_copy = person.copy()

# Get value with default if key doesn't exist
country = person.setdefault("country", "USA")
print(country)  # USA
print(person)   # Now has 'country': 'USA'
```

---

## Looping Through Dictionaries

```python
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

# Loop through keys (default)
for key in person:
    print(key)
# Output: name, age, city

# Loop through keys (explicit)
for key in person.keys():
    print(f"{key}: {person[key]}")

# Loop through values
for value in person.values():
    print(value)
# Output: Alice, 30, NYC

# Loop through key-value pairs (best!)
for key, value in person.items():
    print(f"{key}: {value}")
# Output:
# name: Alice
# age: 30
# city: NYC
```

**C# Comparison:**
```csharp
// C#
foreach (var kvp in person)
{
    Console.WriteLine($"{kvp.Key}: {kvp.Value}");
}

// Python
for key, value in person.items():
    print(f"{key}: {value}")
```

---

## Nested Dictionaries

```python
# Dictionary of dictionaries
users = {
    "user1": {
        "name": "Alice",
        "age": 30
    },
    "user2": {
        "name": "Bob",
        "age": 25
    }
}

# Access nested values
print(users["user1"]["name"])  # Alice

# Loop through nested
for user_id, user_data in users.items():
    print(f"{user_id}:")
    for key, value in user_data.items():
        print(f"  {key}: {value}")
```

---

## Dictionary Comprehensions

Like list comprehensions, but for dictionaries!

```python
# Create dictionary from range
squares = {x: x**2 for x in range(5)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Filter while creating
evens = {x: x**2 for x in range(10) if x % 2 == 0}
print(evens)  # {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# Transform existing dictionary
person = {"name": "alice", "city": "nyc"}
uppercase = {k: v.upper() for k, v in person.items()}
print(uppercase)  # {'name': 'ALICE', 'city': 'NYC'}

# Swap keys and values
original = {"a": 1, "b": 2, "c": 3}
swapped = {v: k for k, v in original.items()}
print(swapped)  # {1: 'a', 2: 'b', 3: 'c'}
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

## Sets - Unique Collections

### What is a Set?

- Collection of **unique** elements
- Unordered (no indexing)
- Fast membership testing
- Like C#'s `HashSet<T>`

```python
# Create set
numbers = {1, 2, 3, 4, 5}
print(numbers)  # {1, 2, 3, 4, 5}

# Duplicates are automatically removed!
numbers = {1, 2, 2, 3, 3, 3}
print(numbers)  # {1, 2, 3}

# Empty set (must use set(), not {})
empty = set()     # ✅ Empty set
empty = {}        # ❌ Empty dictionary!

# Create from list (removes duplicates)
numbers = [1, 2, 2, 3, 3, 3]
unique = set(numbers)
print(unique)  # {1, 2, 3}

# Create from string
chars = set("hello")
print(chars)  # {'h', 'e', 'l', 'o'} - Only unique chars
```

**C# Comparison:**
```csharp
// C#
var numbers = new HashSet<int> {1, 2, 3, 4, 5};

// Python
numbers = {1, 2, 3, 4, 5}
```

---

## Set Operations

### Adding and Removing

```python
fruits = {"apple", "banana"}

# Add single item
fruits.add("orange")
print(fruits)  # {'apple', 'banana', 'orange'}

# Add multiple items
fruits.update(["grape", "mango"])
print(fruits)  # {'apple', 'banana', 'orange', 'grape', 'mango'}

# Remove item (errors if not found)
fruits.remove("banana")

# Remove item (no error if not found)
fruits.discard("pear")  # No error!

# Remove and return random item
item = fruits.pop()

# Clear all
fruits.clear()
```

### Set Mathematical Operations

```python
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}

# Union (all elements from both)
union = a | b
print(union)  # {1, 2, 3, 4, 5, 6, 7, 8}
# or: a.union(b)

# Intersection (elements in both)
intersection = a & b
print(intersection)  # {4, 5}
# or: a.intersection(b)

# Difference (in a but not in b)
difference = a - b
print(difference)  # {1, 2, 3}
# or: a.difference(b)

# Symmetric difference (in either but not both)
sym_diff = a ^ b
print(sym_diff)  # {1, 2, 3, 6, 7, 8}
# or: a.symmetric_difference(b)
```

**Visual:**
```
Set A: {1, 2, 3, 4, 5}
Set B: {4, 5, 6, 7, 8}

Union (A | B):          {1, 2, 3, 4, 5, 6, 7, 8}
Intersection (A & B):   {4, 5}
Difference (A - B):     {1, 2, 3}
Sym Diff (A ^ B):       {1, 2, 3, 6, 7, 8}
```

### Set Comparisons

```python
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}

# Subset (a is subset of b)
print(a <= b)  # True (all elements of a are in b)
print(a.issubset(b))  # True

# Proper subset (subset but not equal)
print(a < b)   # True

# Superset (b is superset of a)
print(b >= a)  # True
print(b.issuperset(a))  # True

# Disjoint (no common elements)
c = {7, 8, 9}
print(a.isdisjoint(c))  # True
```

---

## Practical Examples

### Remove Duplicates from List

```python
# List with duplicates
numbers = [1, 2, 2, 3, 3, 3, 4, 5, 5]

# Convert to set (removes duplicates), then back to list
unique = list(set(numbers))
print(unique)  # [1, 2, 3, 4, 5]
```

### Count Word Frequency

```python
text = "hello world hello python world"
words = text.split()

# Create frequency dictionary
frequency = {}
for word in words:
    frequency[word] = frequency.get(word, 0) + 1

print(frequency)
# {'hello': 2, 'world': 2, 'python': 1}

# Better way using Counter (next lesson will cover)
```

### Group Data

```python
# Group students by grade
students = [
    {"name": "Alice", "grade": "A"},
    {"name": "Bob", "grade": "B"},
    {"name": "Charlie", "grade": "A"},
]

by_grade = {}
for student in students:
    grade = student["grade"]
    if grade not in by_grade:
        by_grade[grade] = []
    by_grade[grade].append(student["name"])

print(by_grade)
# {'A': ['Alice', 'Charlie'], 'B': ['Bob']}
```

---

## 💡 Key Takeaways

1. **Dictionaries** → `{}` with key-value pairs
2. **Access** → `dict["key"]` or `dict.get("key")`
3. **Loop** → `for k, v in dict.items():`
4. **Check key** → `"key" in dict`
5. **Dict comprehension** → `{k: v for ...}`
6. **Sets** → `{}` with unique elements
7. **Set operations** → `|` union, `&` intersection, `-` difference
8. **Remove duplicates** → `list(set(list))`

---

## ✏️ Practice Exercise

Create `dict_set_practice.py`:

```python
# 1. Create and access dictionary
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}

print(person["name"])
print(person.get("country", "USA"))

# 2. Modify dictionary
person["age"] = 31
person["email"] = "alice@example.com"
print(person)

# 3. Loop through dictionary
for key, value in person.items():
    print(f"{key}: {value}")

# 4. Dictionary comprehension
squares = {x: x**2 for x in range(1, 6)}
print(squares)

# 5. Sets
fruits = {"apple", "banana", "orange"}
fruits.add("grape")
print(fruits)

# 6. Remove duplicates
numbers = [1, 2, 2, 3, 3, 3, 4, 5, 5]
unique = list(set(numbers))
print(unique)

# 7. Set operations
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}
print(f"Union: {a | b}")
print(f"Intersection: {a & b}")
print(f"Difference: {a - b}")

# 8. Word frequency
text = "python is great python is fun"
words = text.split()
frequency = {}
for word in words:
    frequency[word] = frequency.get(word, 0) + 1
print(frequency)
```

**Run it:** `python dict_set_practice.py`

---

## 🤔 Quick Quiz

1. How do you create an empty dictionary?
   <details>
   <summary>Answer</summary>

   `empty = {}` or `empty = dict()`
   </details>

2. What's the safe way to access a dictionary key that might not exist?
   <details>
   <summary>Answer</summary>

   `dict.get("key")` or `dict.get("key", default)`
   </details>

3. How do you loop through both keys and values?
   <details>
   <summary>Answer</summary>

   `for key, value in dict.items():`
   </details>

4. What's the difference between a set and a list?
   <details>
   <summary>Answer</summary>

   - Set: Unique elements, unordered, no indexing
   - List: Can have duplicates, ordered, has indexing
   </details>

5. How do you remove duplicates from a list?
   <details>
   <summary>Answer</summary>

   `unique = list(set(my_list))`
   </details>

---

**Next Lesson:** [07_comprehensions.md](07_comprehensions.md) - Master list/dict/set comprehensions!

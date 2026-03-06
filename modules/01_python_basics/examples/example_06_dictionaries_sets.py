"""
Example 6: Dictionaries and Sets for .NET Developers
This file demonstrates Python dictionaries and sets with detailed comments and C# comparisons.
"""

# ============================================
# SECTION 1: CREATING DICTIONARIES
# ============================================

print("=== SECTION 1: CREATING DICTIONARIES ===\n")

# C#: var person = new Dictionary<string, object> { {"name", "Alice"}, {"age", 30} };
# Python: person = {"name": "Alice", "age": 30}

# Empty dictionary
# C#: var empty = new Dictionary<string, object>();
# Python: empty = {}
empty_dict = {}
print(f"Empty dict: {empty_dict}, Type: {type(empty_dict)}")

# Dictionary with initial values
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}
print(f"Person: {person}")

print()

# Using dict() constructor
# Method 1: Keyword arguments
person2 = dict(name="Bob", age=25, city="LA")
print(f"Person2 (dict()): {person2}")

# Method 2: From list of tuples
pairs = [("name", "Charlie"), ("age", 35), ("city", "SF")]
person3 = dict(pairs)
print(f"Person3 (from tuples): {person3}")

print()

# Mixed value types
# C# requires Dictionary<string, object> for mixed types
# Python handles this automatically
mixed_dict = {
    "name": "Alice",      # string
    "age": 30,            # int
    "height": 5.6,        # float
    "active": True,       # bool
    "hobbies": ["reading", "coding"]  # list
}
print(f"Mixed types: {mixed_dict}")

print()

# ============================================
# SECTION 2: ACCESSING DICTIONARY VALUES
# ============================================

print("=== SECTION 2: ACCESSING DICTIONARY VALUES ===\n")

person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}
print(f"Person: {person}")
print()

# Access by key - Method 1: Square brackets
# C#: var name = person["name"];
# Python: name = person["name"]
print("Access by key (square brackets):")
print(f"person['name'] = {person['name']}")
print(f"person['age'] = {person['age']}")
print(f"person['city'] = {person['city']}")

print()

# Access by key - Method 2: get() (safer!)
# C#: person.TryGetValue("name", out var name);
# Python: name = person.get("name")
print("Access by key (get method - safer):")
print(f"person.get('name') = {person.get('name')}")
print(f"person.get('age') = {person.get('age')}")

# Key doesn't exist - get() returns None instead of error
print(f"person.get('country') = {person.get('country')}")  # None

# Provide default value
print(f"person.get('country', 'USA') = {person.get('country', 'USA')}")  # USA

print()

# Error when key doesn't exist (with square brackets)
print("Error handling:")
try:
    country = person["country"]  # This will raise KeyError
except KeyError as e:
    print(f"KeyError: {e}")

print()

# ============================================
# SECTION 3: MODIFYING DICTIONARIES
# ============================================

print("=== SECTION 3: MODIFYING DICTIONARIES ===\n")

person = {"name": "Alice", "age": 30}
print(f"Original: {person}")

# Add new key-value pair
# C#: person["city"] = "NYC";
# Python: person["city"] = "NYC"
person["city"] = "NYC"
print(f"After adding 'city': {person}")

# Update existing value
person["age"] = 31
print(f"After updating 'age': {person}")

print()

# update() method - Update multiple values
# C#: foreach(var kvp in updates) person[kvp.Key] = kvp.Value;
# Python: person.update(updates)
person.update({"age": 32, "country": "USA"})
print(f"After update(): {person}")

print()

# Merge dictionaries (Python 3.9+)
defaults = {"theme": "dark", "lang": "en", "notifications": True}
user_prefs = {"lang": "fr", "notifications": False}
print(f"Defaults: {defaults}")
print(f"User preferences: {user_prefs}")

merged = defaults | user_prefs  # User prefs override defaults
print(f"Merged (defaults | user_prefs): {merged}")

print()

# ============================================
# SECTION 4: REMOVING ITEMS
# ============================================

print("=== SECTION 4: REMOVING ITEMS ===\n")

person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC",
    "country": "USA"
}
print(f"Original: {person}")
print()

# pop() - Remove by key and return value
# C#: if (person.Remove("age", out var age)) { ... }
# Python: age = person.pop("age")
age = person.pop("age")
print(f"Popped 'age': {age}")
print(f"After pop('age'): {person}")

print()

# pop() with default value (if key doesn't exist)
country = person.pop("country", "Unknown")
print(f"Popped 'country': {country}")

salary = person.pop("salary", 0)  # Key doesn't exist, returns default
print(f"Popped 'salary' (doesn't exist): {salary}")
print(f"After pops: {person}")

print()

# del - Delete by key (no return value)
person = {"name": "Alice", "age": 30, "city": "NYC"}
print(f"Before del: {person}")

del person["city"]
print(f"After del person['city']: {person}")

print()

# popitem() - Remove and return last item (Python 3.7+)
person = {"name": "Alice", "age": 30, "city": "NYC"}
print(f"Before popitem: {person}")

item = person.popitem()  # Returns (key, value) tuple
print(f"Popped item: {item}")
print(f"After popitem: {person}")

print()

# clear() - Remove all items
# C#: person.Clear();
# Python: person.clear()
person = {"name": "Alice", "age": 30}
print(f"Before clear: {person}")

person.clear()
print(f"After clear(): {person}")

print()

# ============================================
# SECTION 5: DICTIONARY METHODS
# ============================================

print("=== SECTION 5: DICTIONARY METHODS ===\n")

person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}
print(f"Person: {person}")
print()

# Get all keys
# C#: person.Keys
# Python: person.keys()
keys = person.keys()
print(f"Keys: {keys}")
print(f"Keys as list: {list(keys)}")

print()

# Get all values
# C#: person.Values
# Python: person.values()
values = person.values()
print(f"Values: {values}")
print(f"Values as list: {list(values)}")

print()

# Get all key-value pairs
# C#: foreach (var kvp in person)
# Python: person.items()
items = person.items()
print(f"Items: {items}")
print(f"Items as list: {list(items)}")

print()

# Check if key exists
# C#: person.ContainsKey("name")
# Python: "name" in person
print(f"'name' in person: {'name' in person}")
print(f"'country' in person: {'country' in person}")
print(f"'country' not in person: {'country' not in person}")

print()

# Get number of items
# C#: person.Count
# Python: len(person)
print(f"Length: {len(person)}")

print()

# Copy dictionary
# C#: var copy = new Dictionary<string, object>(person);
# Python: copy = person.copy()
person_copy = person.copy()
print(f"Copy: {person_copy}")

print()

# setdefault() - Get value or set default if key doesn't exist
person = {"name": "Alice", "age": 30}
print(f"Before setdefault: {person}")

country = person.setdefault("country", "USA")
print(f"country: {country}")
print(f"After setdefault('country', 'USA'): {person}")  # 'country' added!

name = person.setdefault("name", "Unknown")
print(f"name: {name}")
print(f"After setdefault('name', 'Unknown'): {person}")  # 'name' unchanged

print()

# ============================================
# SECTION 6: LOOPING THROUGH DICTIONARIES
# ============================================

print("=== SECTION 6: LOOPING THROUGH DICTIONARIES ===\n")

person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}
print(f"Person: {person}")
print()

# Loop through keys (default)
print("Loop through keys (default):")
for key in person:
    print(f"  {key}")

print()

# Loop through keys (explicit)
print("Loop through keys (explicit):")
for key in person.keys():
    print(f"  {key}: {person[key]}")

print()

# Loop through values
print("Loop through values:")
for value in person.values():
    print(f"  {value}")

print()

# Loop through key-value pairs (best!)
# C#: foreach (var kvp in person) { ... }
# Python: for key, value in person.items():
print("Loop through key-value pairs (best way):")
for key, value in person.items():
    print(f"  {key}: {value}")

print()

# ============================================
# SECTION 7: NESTED DICTIONARIES
# ============================================

print("=== SECTION 7: NESTED DICTIONARIES ===\n")

# Dictionary of dictionaries
users = {
    "user1": {
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com"
    },
    "user2": {
        "name": "Bob",
        "age": 25,
        "email": "bob@example.com"
    }
}

print(f"Users: {users}")
print()

# Access nested values
print(f"user1 name: {users['user1']['name']}")
print(f"user2 age: {users['user2']['age']}")

print()

# Loop through nested dictionaries
print("Loop through nested dictionaries:")
for user_id, user_data in users.items():
    print(f"{user_id}:")
    for key, value in user_data.items():
        print(f"  {key}: {value}")
    print()

# ============================================
# SECTION 8: DICTIONARY COMPREHENSIONS
# ============================================

print("=== SECTION 8: DICTIONARY COMPREHENSIONS ===\n")

# Like list comprehensions, but for dictionaries!
# C#: var squares = Enumerable.Range(0, 5).ToDictionary(x => x, x => x * x);
# Python: squares = {x: x**2 for x in range(5)}

# Create dictionary from range
squares = {x: x**2 for x in range(5)}
print(f"Squares: {squares}")

# Create from two lists using zip
keys = ["name", "age", "city"]
values = ["Alice", 30, "NYC"]
person = {k: v for k, v in zip(keys, values)}
print(f"Person from zip: {person}")

print()

# Transform existing dictionary
prices = {"apple": 0.50, "banana": 0.30, "orange": 0.60}
print(f"Original prices: {prices}")

doubled = {fruit: price * 2 for fruit, price in prices.items()}
print(f"Doubled prices: {doubled}")

print()

# Filter while creating
evens = {x: x**2 for x in range(10) if x % 2 == 0}
print(f"Even squares: {evens}")

expensive = {fruit: price for fruit, price in prices.items() if price > 0.40}
print(f"Expensive fruits: {expensive}")

print()

# Swap keys and values
original = {"a": 1, "b": 2, "c": 3}
swapped = {v: k for k, v in original.items()}
print(f"Original: {original}")
print(f"Swapped: {swapped}")

print()

# Transform to uppercase
person = {"name": "alice", "city": "nyc"}
uppercase = {k: v.upper() for k, v in person.items()}
print(f"Original: {person}")
print(f"Uppercase: {uppercase}")

print()

# ============================================
# SECTION 9: SETS - UNIQUE COLLECTIONS
# ============================================

print("=== SECTION 9: SETS - UNIQUE COLLECTIONS ===\n")

# C#: var numbers = new HashSet<int> {1, 2, 3, 4, 5};
# Python: numbers = {1, 2, 3, 4, 5}

# Create set
numbers = {1, 2, 3, 4, 5}
print(f"Numbers set: {numbers}, Type: {type(numbers)}")

# Duplicates are automatically removed!
numbers_with_dupes = {1, 2, 2, 3, 3, 3, 4, 5, 5}
print(f"With duplicates: {numbers_with_dupes}")

print()

# Empty set - MUST use set(), not {}!
# {} creates an empty dictionary, not set!
empty_set = set()  # Correct
empty_dict = {}    # This is a dictionary!
print(f"Empty set: {empty_set}, Type: {type(empty_set)}")
print(f"Empty dict: {empty_dict}, Type: {type(empty_dict)}")

print()

# Create from list (removes duplicates)
numbers_list = [1, 2, 2, 3, 3, 3, 4, 5, 5]
unique = set(numbers_list)
print(f"List: {numbers_list}")
print(f"Set (unique): {unique}")

print()

# Create from string (unique characters)
chars = set("hello")
print(f"Set from 'hello': {chars}")

text = set("mississippi")
print(f"Set from 'mississippi': {text}")

print()

# ============================================
# SECTION 10: SET OPERATIONS - ADDING/REMOVING
# ============================================

print("=== SECTION 10: SET OPERATIONS - ADDING/REMOVING ===\n")

fruits = {"apple", "banana"}
print(f"Original: {fruits}")

# Add single item
# C#: fruits.Add("orange");
# Python: fruits.add("orange")
fruits.add("orange")
print(f"After add('orange'): {fruits}")

fruits.add("banana")  # Already exists, no duplicate added
print(f"After add('banana') again: {fruits}")

print()

# Add multiple items
# C#: fruits.UnionWith(new[] {"grape", "mango"});
# Python: fruits.update(["grape", "mango"])
fruits.update(["grape", "mango"])
print(f"After update(['grape', 'mango']): {fruits}")

print()

# Remove item (error if not found)
fruits.remove("banana")
print(f"After remove('banana'): {fruits}")

try:
    fruits.remove("pear")  # Error! Not in set
except KeyError as e:
    print(f"KeyError when removing 'pear': {e}")

print()

# Discard item (no error if not found)
fruits.discard("orange")
print(f"After discard('orange'): {fruits}")

fruits.discard("pear")  # No error!
print(f"After discard('pear') (not in set): {fruits}")

print()

# Pop - Remove and return arbitrary item
fruits = {"apple", "banana", "orange"}
print(f"Before pop: {fruits}")

item = fruits.pop()
print(f"Popped item: {item}")
print(f"After pop: {fruits}")

print()

# Clear - Remove all
fruits.clear()
print(f"After clear(): {fruits}")

print()

# ============================================
# SECTION 11: SET MATHEMATICAL OPERATIONS
# ============================================

print("=== SECTION 11: SET MATHEMATICAL OPERATIONS ===\n")

a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}
print(f"Set A: {a}")
print(f"Set B: {b}")
print()

# Union - All elements from both sets
# C#: a.Union(b)
# Python: a | b or a.union(b)
union = a | b
print(f"Union (A | B): {union}")
print(f"Union (A.union(B)): {a.union(b)}")

print()

# Intersection - Elements in both sets
# C#: a.Intersect(b)
# Python: a & b or a.intersection(b)
intersection = a & b
print(f"Intersection (A & B): {intersection}")
print(f"Intersection (A.intersection(B)): {a.intersection(b)}")

print()

# Difference - Elements in A but not in B
# C#: a.Except(b)
# Python: a - b or a.difference(b)
difference = a - b
print(f"Difference (A - B): {difference}")
print(f"Difference (A.difference(B)): {a.difference(b)}")

print()

# Symmetric difference - Elements in either but not both
# C#: a.SymmetricExcept(b)
# Python: a ^ b or a.symmetric_difference(b)
sym_diff = a ^ b
print(f"Symmetric Difference (A ^ B): {sym_diff}")
print(f"Symmetric Difference (A.symmetric_difference(B)): {a.symmetric_difference(b)}")

print()

# Visual representation
print("Visual representation:")
print("Set A: {1, 2, 3, 4, 5}")
print("Set B: {4, 5, 6, 7, 8}")
print()
print("Union (A | B):               {1, 2, 3, 4, 5, 6, 7, 8}")
print("Intersection (A & B):        {4, 5}")
print("Difference (A - B):          {1, 2, 3}")
print("Symmetric Diff (A ^ B):      {1, 2, 3, 6, 7, 8}")

print()

# ============================================
# SECTION 12: SET COMPARISONS
# ============================================

print("=== SECTION 12: SET COMPARISONS ===\n")

a = {1, 2, 3}
b = {1, 2, 3, 4, 5}
c = {7, 8, 9}
print(f"Set A: {a}")
print(f"Set B: {b}")
print(f"Set C: {c}")
print()

# Subset - All elements of A are in B
# C#: a.IsSubsetOf(b)
# Python: a <= b or a.issubset(b)
print(f"A <= B (A is subset of B): {a <= b}")
print(f"A.issubset(B): {a.issubset(b)}")

print()

# Proper subset - Subset but not equal
print(f"A < B (A is proper subset of B): {a < b}")

print()

# Superset - All elements of A contain B
# C#: b.IsSupersetOf(a)
# Python: b >= a or b.issuperset(a)
print(f"B >= A (B is superset of A): {b >= a}")
print(f"B.issuperset(A): {b.issuperset(a)}")

print()

# Proper superset - Superset but not equal
print(f"B > A (B is proper superset of A): {b > a}")

print()

# Disjoint - No common elements
# C#: !a.Overlaps(c)
# Python: a.isdisjoint(c)
print(f"A.isdisjoint(C) (no common elements): {a.isdisjoint(c)}")
print(f"A.isdisjoint(B) (have common elements): {a.isdisjoint(b)}")

print()

# ============================================
# SECTION 13: PRACTICAL EXAMPLES
# ============================================

print("=== SECTION 13: PRACTICAL EXAMPLES ===\n")

# Example 1: Remove duplicates from list
print("Example 1: Remove duplicates:")
numbers = [1, 2, 2, 3, 3, 3, 4, 5, 5]
unique = list(set(numbers))
print(f"Original list: {numbers}")
print(f"Unique: {unique}")

print()

# Example 2: Count word frequency
print("Example 2: Word frequency:")
text = "hello world hello python world hello"
words = text.split()

frequency = {}
for word in words:
    frequency[word] = frequency.get(word, 0) + 1

print(f"Text: {text}")
print(f"Frequency: {frequency}")

print()

# Example 3: Group students by grade
print("Example 3: Group by grade:")
students = [
    {"name": "Alice", "grade": "A"},
    {"name": "Bob", "grade": "B"},
    {"name": "Charlie", "grade": "A"},
    {"name": "David", "grade": "B"},
]

by_grade = {}
for student in students:
    grade = student["grade"]
    if grade not in by_grade:
        by_grade[grade] = []
    by_grade[grade].append(student["name"])

print(f"Students by grade: {by_grade}")

print()

# Example 4: Find common interests
print("Example 4: Common interests:")
alice_interests = {"coding", "reading", "music"}
bob_interests = {"music", "sports", "reading"}

common = alice_interests & bob_interests
print(f"Alice's interests: {alice_interests}")
print(f"Bob's interests: {bob_interests}")
print(f"Common interests: {common}")

print()

# Example 5: Unique visitors
print("Example 5: Track unique visitors:")
visitors = ["Alice", "Bob", "Alice", "Charlie", "Bob", "David"]
unique_visitors = set(visitors)
print(f"All visitors: {visitors}")
print(f"Unique visitors: {unique_visitors}")
print(f"Total unique: {len(unique_visitors)}")

print()

# ============================================
# SUMMARY
# ============================================

print("=== SUMMARY ===\n")

summary = """
Dictionaries and Sets for .NET Developers:

DICTIONARIES (like C# Dictionary<K,V>):
  - Created with curly braces: {"key": value}
  - Store key-value pairs
  - Keys must be unique
  - Fast lookup by key

CREATING:
  C#: var person = new Dictionary<string, object> {{"name", "Alice"}};
  Python: person = {"name": "Alice"}

  empty = {}
  person = {"name": "Alice", "age": 30}
  dict(name="Bob", age=25)

ACCESSING:
  C#: person["name"]           → Python: person["name"]
  C#: TryGetValue("name", out) → Python: person.get("name")

  person["name"]              # Error if key missing
  person.get("name")          # Returns None if missing
  person.get("name", "N/A")   # Returns default if missing

MODIFYING:
  person["city"] = "NYC"      # Add or update
  person.update({"age": 31})  # Update multiple

REMOVING:
  C#: person.Remove("age")    → Python: person.pop("age")
  del person["city"]          # Delete by key
  person.clear()              # Remove all

METHODS:
  C#: person.Keys             → Python: person.keys()
  C#: person.Values           → Python: person.values()
  C#: person.ContainsKey(k)   → Python: k in person
  C#: person.Count            → Python: len(person)

LOOPING:
  C#: foreach (var kvp in person)
  Python: for key, value in person.items():

DICT COMPREHENSIONS:
  C#: Enumerable.Range(0,5).ToDictionary(x => x, x => x*x)
  Python: {x: x**2 for x in range(5)}

SETS (like C# HashSet<T>):
  - Created with curly braces: {1, 2, 3}
  - Store unique elements only
  - Unordered (no indexing)
  - Fast membership testing

CREATING:
  C#: var numbers = new HashSet<int> {1, 2, 3};
  Python: numbers = {1, 2, 3}

  empty = set()               # NOT {} (that's dict!)
  numbers = {1, 2, 3}
  unique = set([1, 2, 2, 3])  # From list

OPERATIONS:
  C#: numbers.Add(4)          → Python: numbers.add(4)
  C#: numbers.Remove(2)       → Python: numbers.remove(2)
  numbers.discard(5)          # No error if missing

SET MATH:
  C#: a.Union(b)              → Python: a | b
  C#: a.Intersect(b)          → Python: a & b
  C#: a.Except(b)             → Python: a - b
  C#: a.SymmetricExcept(b)    → Python: a ^ b

SET COMPARISONS:
  C#: a.IsSubsetOf(b)         → Python: a <= b
  C#: b.IsSupersetOf(a)       → Python: b >= a
  a.isdisjoint(b)             # No common elements

COMMON PATTERNS:
  Remove duplicates: list(set(my_list))
  Count frequency: {word: frequency.get(word, 0) + 1 for word in words}
  Find common: set1 & set2
  Find differences: set1 - set2

C# → Python Quick Reference:
  new Dictionary<K,V>()        → {}
  person["key"]                → person["key"]
  person.TryGetValue()         → person.get("key")
  person.Keys                  → person.keys()
  person.Values                → person.values()
  person.ContainsKey(k)        → k in person
  new HashSet<T>()             → set() or {1, 2, 3}
  set.Add(x)                   → set.add(x)
  set.Remove(x)                → set.remove(x) or set.discard(x)
"""

print(summary)

print("="*60)
print("Next: example_07_comprehensions.py - Learn comprehensions!")
print("="*60)

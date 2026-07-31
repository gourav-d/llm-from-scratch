"""
Module 01 - Python Basics
Exercise 06: Dictionaries and Sets

GLOSSARY
--------
dict          : A MUTABLE mapping of key -> value pairs. Written with curly braces {}.
                Like C# Dictionary<K,V>. Keys must be unique and immutable (str, int, tuple).
                Example: {"name": "Alice", "age": 30}
key           : The lookup identifier in a dict. Must be unique. Like a dictionary word.
value         : The data associated with a key. Can be anything (list, dict, int...).
get()         : Safe way to access a dict value. Returns a default if key not found.
                Like C# dict.TryGetValue() or dict.GetValueOrDefault().
keys()        : Returns all keys in the dict (like C# .Keys property).
values()      : Returns all values in the dict (like C# .Values property).
items()       : Returns all (key, value) pairs as tuples. Used for iterating.
                Like C# foreach(var kvp in dict) where kvp.Key and kvp.Value.
KeyError      : Exception raised when you access a key that doesn't exist.
                Like C# KeyNotFoundException.
set           : An UNORDERED collection of UNIQUE items. No duplicates allowed.
                Like C# HashSet<T>. Written with curly braces {} but no key:value.
union         : Combines two sets (all items from both). Like SQL UNION.
intersection  : Items that appear in BOTH sets. Like SQL INTERSECT.
difference    : Items in one set but NOT the other. Like SQL EXCEPT.
"""

print("=" * 60)    # "=" repeated 60 times -- visual separator
print("Exercise 06: Dictionaries and Sets")  # exercise title
print("=" * 60)    # separator
print()            # blank line


# ============================================================
#  EXERCISE 1
#  Topic: Create a dict, add key-value pairs, check key existence
#
#  Background:
#    A dictionary maps keys to values. Perfect for looking things up.
#    In C#:
#      var person = new Dictionary<string,object>();
#      person["name"] = "Alice";
#      person["age"]  = 30;
#      bool hasKey = person.ContainsKey("name");
#
#    In Python:
#      person = {}              # empty dict (or person = dict())
#      person["name"] = "Alice" # add/update a key-value pair
#      person["age"]  = 30
#      has_key = "name" in person   # True/False -- like ContainsKey()
#
#    Create a dict with literal syntax:
#      person = {"name": "Alice", "age": 30}   # key: value pairs
#
#  Your Task:
#    1. Create an empty dict called 'student'
#    2. Add key "name" with value "Alice"
#    3. Add key "grade" with value "A"
#    4. Add key "score" with value 95
#    5. Return (student["name"], "grade" in student, "height" in student)
#       That's: the name value, whether "grade" key exists, whether "height" exists
#
#  C# Analogy:
#    var s = new Dictionary<string,object>();
#    s["name"]="Alice"; s["grade"]="A"; s["score"]=95;
#    return (s["name"], s.ContainsKey("grade"), s.ContainsKey("height"));
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 1: Create and Use a Dictionary")  # section title
print("-" * 50)    # separator
print()            # blank line


def create_student_dict():
    """
    Create a student dictionary and check for keys.

    Returns:
        tuple: (name_value, grade_key_exists, height_key_exists)
    """
    # TODO: Build the dict and return the required values
    # student = {}                     # create an empty dictionary
    # student["name"]  = "Alice"       # add key "name" with value "Alice"
    # student["grade"] = "A"           # add key "grade" with value "A"
    # student["score"] = 95            # add key "score" with value 95
    # name_value         = student["name"]       # get the value for key "name"
    # grade_key_exists   = "grade" in student    # True if "grade" key exists
    # height_key_exists  = "height" in student   # False -- we never added "height"
    # return (name_value, grade_key_exists, height_key_exists)
    pass   # replace with your code


result1 = create_student_dict()     # call the function
if result1 is not None:             # only print if something returned
    name_val, has_grade, has_height = result1   # unpack 3 values
    print(f"  student['name']       : {name_val}")       # Alice
    print(f"  'grade' in student    : {has_grade}")     # True
    print(f"  'height' in student   : {has_height}")    # False
print()
print("  Expected: 'Alice', True, False")
print()


# ============================================================
#  EXERCISE 2
#  Topic: dict.get() -- safe access with a default value
#
#  Background:
#    Accessing a key that doesn't exist raises a KeyError:
#      person["height"]   <- KeyError if "height" not in dict!
#
#    To avoid this, use dict.get(key, default):
#      person.get("height", "unknown")  -> "unknown" (no error)
#      person.get("name", "unknown")    -> "Alice"   (key exists)
#
#    This is like the null-coalescing operator in C#:
#      dict.TryGetValue("height", out var v) ? v : "unknown"
#      Or: dict.GetValueOrDefault("height", "unknown")
#
#  Your Task:
#    Given a config dict: {"host": "localhost", "port": 8080}
#    Use .get() to safely retrieve:
#    1. "host"    -- exists, should return "localhost"
#    2. "port"    -- exists, should return 8080
#    3. "timeout" -- does NOT exist, use default 30
#    4. "debug"   -- does NOT exist, use default False
#    Return all four values as a tuple.
#
#  C# Analogy:
#    config.GetValueOrDefault("host", "localhost")
#    config.GetValueOrDefault("timeout", 30)
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 2: dict.get() with Default Values")  # section title
print("-" * 50)    # separator
print()            # blank line


def safe_dict_access():
    """
    Use dict.get() to safely access dictionary values with defaults.

    Returns:
        tuple: (host, port, timeout, debug)
    """
    # TODO: Use .get() to retrieve values with defaults
    # config = {"host": "localhost", "port": 8080}   # the dictionary
    # host    = config.get("host", "unknown")         # "host" exists -> "localhost"
    # port    = config.get("port", 0)                 # "port" exists -> 8080
    # timeout = config.get("timeout", 30)             # "timeout" missing -> default 30
    # debug   = config.get("debug", False)            # "debug" missing -> default False
    # return (host, port, timeout, debug)
    pass   # replace with your code


result2 = safe_dict_access()      # call the function
if result2 is not None:           # only print if something returned
    host, port, timeout, debug = result2   # unpack 4 values
    print(f"  host    : {host}")      # "localhost"
    print(f"  port    : {port}")      # 8080
    print(f"  timeout : {timeout}")  # 30 (default)
    print(f"  debug   : {debug}")    # False (default)
print()
print("  Expected: 'localhost', 8080, 30, False")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Iterating a dict with .items()
#
#  Background:
#    .items() returns (key, value) pairs you can iterate over.
#    In C#:  foreach(var kvp in dict) { kvp.Key, kvp.Value }
#    Python: for key, value in dict.items():  # unpack key and value
#
#    You can also iterate just keys or just values:
#      for key in dict:             # iterates keys only
#      for key in dict.keys():      # same thing, more explicit
#      for value in dict.values():  # iterates values only
#
#  Your Task:
#    Given a scores dict: {"Alice": 95, "Bob": 78, "Carol": 88}
#    Iterate with .items() and build a new dict where:
#      - If score >= 90: status = "pass_with_distinction"
#      - If score >= 70: status = "pass"
#      - Otherwise:      status = "fail"
#    Return the new dict: {name: status, ...}
#
#  C# Analogy:
#    foreach(var kvp in scores) {
#        string status = kvp.Value >= 90 ? "pass_with_distinction"
#                      : kvp.Value >= 70 ? "pass" : "fail";
#        result[kvp.Key] = status;
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 3: Iterate Dict with .items()")  # section title
print("-" * 50)    # separator
print()            # blank line


def classify_scores():
    """
    Classify scores as pass_with_distinction, pass, or fail.

    Returns:
        dict: {name: status_string}
    """
    # TODO: Iterate with .items() and build a result dict
    # scores = {"Alice": 95, "Bob": 78, "Carol": 88}  # the source dict
    # result = {}                          # empty dict for results
    # for name, score in scores.items():   # .items() gives (key, value) pairs
    #     if score >= 90:                  # check highest threshold first
    #         status = "pass_with_distinction"
    #     elif score >= 70:               # between 70 and 89
    #         status = "pass"
    #     else:                           # below 70
    #         status = "fail"
    #     result[name] = status           # store the status for this name
    # return result
    pass   # replace with your code


result3 = classify_scores()       # call the function
if result3 is not None:           # only print if something returned
    for name, status in result3.items():   # iterate the returned dict
        print(f"  {name:10} -> {status}")  # :10 pads name to 10 chars
print()
print("  Expected:")
print("    Alice      -> pass_with_distinction")
print("    Bob        -> pass")
print("    Carol      -> pass")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Set operations (union, intersection, difference)
#
#  Background:
#    A set holds UNIQUE items -- no duplicates.
#    {1, 2, 3}   (note: looks like a dict but no colons!)
#    set()       creates an empty set (can't use {} -- that's an empty dict)
#
#    Converting a list to a set removes duplicates:
#      set([1,2,2,3,3,3])  -> {1, 2, 3}
#
#    Set operations:
#      a | b   or  a.union(b)          -- all items from EITHER set
#      a & b   or  a.intersection(b)   -- items in BOTH sets
#      a - b   or  a.difference(b)     -- items in a but NOT in b
#
#    In C# (HashSet):
#      setA.UnionWith(setB)
#      setA.IntersectWith(setB)
#      setA.ExceptWith(setB)
#
#  Your Task:
#    Given:
#      python_students = {"Alice", "Bob", "Carol", "Dave"}
#      java_students   = {"Bob", "Dave", "Eve", "Frank"}
#    Return:
#    (all_students, both_languages, python_only)
#
#  C# Analogy:
#    var all = new HashSet<string>(python); all.UnionWith(java);
#    var both = new HashSet<string>(python); both.IntersectWith(java);
#    var only = new HashSet<string>(python); only.ExceptWith(java);
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 4: Set Operations")  # section title
print("-" * 50)    # separator
print()            # blank line


def set_operations():
    """
    Perform union, intersection, and difference on two sets.

    Returns:
        tuple: (union_set, intersection_set, difference_set)
    """
    # TODO: Create sets and perform operations
    # python_students = {"Alice", "Bob", "Carol", "Dave"}  # set literal
    # java_students   = {"Bob", "Dave", "Eve", "Frank"}    # set literal
    # all_students    = python_students | java_students    # UNION: everyone
    # both_languages  = python_students & java_students    # INTERSECTION: both
    # python_only     = python_students - java_students    # DIFFERENCE: python not java
    # return (all_students, both_languages, python_only)
    pass   # replace with your code


result4 = set_operations()        # call the function
if result4 is not None:           # only print if something returned
    union, inter, diff = result4  # unpack 3 values
    print(f"  All students   : {sorted(union)}")   # sorted() for consistent order
    print(f"  Both languages : {sorted(inter)}")   # sorted() for consistent order
    print(f"  Python only    : {sorted(diff)}")    # sorted() for consistent order
print()
print("  Expected:")
print("    All:        Alice, Bob, Carol, Dave, Eve, Frank")
print("    Both:       Bob, Dave")
print("    Python only:Alice, Carol")
print()


# ============================================================
#  EXERCISE 5
#  Topic: Count word frequencies using a dict
#
#  Background:
#    A classic dict use case: counting occurrences.
#    For each word in a list, check if it's already in the dict.
#    If yes: increment the count. If no: set count to 1.
#
#    Efficient pattern:
#      freq = {}
#      for word in words:
#          freq[word] = freq.get(word, 0) + 1
#          # get(word, 0) returns current count OR 0 if first occurrence
#          # then +1 for this occurrence
#
#    In C#:
#      if (freq.ContainsKey(word)) freq[word]++;
#      else freq[word] = 1;
#
#  Your Task:
#    Given a list of words, return a dict mapping each unique
#    word to how many times it appears.
#    words = ["apple","banana","apple","cherry","banana","apple"]
#    Expected: {"apple": 3, "banana": 2, "cherry": 1}
#
#  C# Analogy:
#    var freq = new Dictionary<string,int>();
#    foreach(var w in words) freq[w] = freq.GetValueOrDefault(w,0) + 1;
#    return freq;
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 5: Count Word Frequencies")  # section title
print("-" * 50)    # separator
print()            # blank line


def count_words(words):
    """
    Count how many times each word appears in the list.

    Args:
        words (list): A list of strings.

    Returns:
        dict: {word: count} for each unique word.
    """
    # TODO: Build a frequency dict
    # freq = {}                            # start with an empty dict
    # for word in words:                   # loop through each word
    #     freq[word] = freq.get(word, 0) + 1  # increment count (or start at 0+1=1)
    # return freq
    pass   # replace with your code


test_words = ["apple", "banana", "apple", "cherry", "banana", "apple"]  # test data
result5 = count_words(test_words)     # call the function
if result5 is not None:               # only print if something returned
    for word, count in sorted(result5.items()):   # sorted by word for consistency
        bar = "#" * count             # draw a simple bar chart with # symbols
        print(f"  {word:10} : {count} {bar}")    # show word, count, and bar
print()
print("  Expected: apple=3, banana=2, cherry=1")
print()

print("=" * 60)                       # closing separator
print("All exercises complete!")      # completion message
print("=" * 60)                       # closing separator

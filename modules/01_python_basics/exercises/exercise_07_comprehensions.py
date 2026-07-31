"""
Module 01 - Python Basics
Exercise 07: Comprehensions

GLOSSARY
--------
list comprehension  : A compact way to build a list in one line.
                      [expression for item in iterable]
                      Like C# LINQ .Select() in one readable line.
dict comprehension  : A compact way to build a dict in one line.
                      {key_expr: value_expr for item in iterable}
                      Like C# .ToDictionary() but much shorter.
set comprehension   : A compact way to build a set in one line.
                      {expression for item in iterable}
filter condition    : An optional 'if' at the end of a comprehension.
                      [x for x in items if x > 0]  -- only include items where condition is True.
                      Like C# .Where(x => x > 0).Select(...).
LINQ equivalent     : Python comprehensions replace most LINQ chains.
                      C#: items.Where(x=>x%2==0).Select(x=>x*x).ToList()
                      Python: [x*x for x in items if x%2==0]
expression          : The value to compute for each item (goes before 'for').
iterable            : Any sequence you can loop over: list, range, string, dict, etc.
nested              : A comprehension inside another comprehension.
                      [[col for col in row] for row in matrix]
"""

print("=" * 60)    # "=" repeated 60 times -- visual separator
print("Exercise 07: Comprehensions")  # exercise title
print("=" * 60)    # separator
print()            # blank line


# ============================================================
#  EXERCISE 1
#  Topic: Basic list comprehension (select/transform)
#
#  Background:
#    A list comprehension builds a list by applying an expression
#    to every item in an iterable. General form:
#      [EXPRESSION for ITEM in ITERABLE]
#
#    Example: square each number 1-5
#      squares = [x**2 for x in range(1, 6)]
#      -> [1, 4, 9, 16, 25]
#
#    The long way (for loop version):
#      squares = []
#      for x in range(1, 6):
#          squares.append(x**2)
#
#    C# LINQ equivalent:
#      var squares = Enumerable.Range(1,5).Select(x => x*x).ToList();
#
#  Your Task:
#    Use a list comprehension to compute squares of numbers 1 through 10.
#    Return the list: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
#
#  C# Analogy:
#    Enumerable.Range(1,10).Select(x => x * x).ToList()
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 1: List Comprehension (Squares 1-10)")  # section title
print("-" * 50)    # separator
print()            # blank line


def squares_of_ten():
    """
    Compute squares of numbers 1 through 10 using a list comprehension.

    Returns:
        list: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    """
    # TODO: Use a list comprehension
    # return [x**2 for x in range(1, 11)]
    # Breakdown:
    #   x**2        -- the EXPRESSION: square each number
    #   for x       -- x is each item
    #   in range(1, 11) -- iterate 1 through 10 (11 is excluded)
    pass   # replace with your code


result1 = squares_of_ten()        # call the function
if result1 is not None:           # only print if something returned
    print(f"  Squares 1-10: {result1}")
print()
print("  Expected: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Filtered list comprehension (select + where)
#
#  Background:
#    You can add an 'if' condition to filter items:
#      [EXPRESSION for ITEM in ITERABLE if CONDITION]
#
#    Only items where CONDITION is True are included.
#
#    Example: even numbers only
#      evens = [x for x in range(1, 11) if x % 2 == 0]
#      -> [2, 4, 6, 8, 10]
#
#    Long way:
#      evens = []
#      for x in range(1, 11):
#          if x % 2 == 0:
#              evens.append(x)
#
#    C# LINQ equivalent:
#      Enumerable.Range(1,10).Where(x => x%2==0).ToList()
#
#  Your Task:
#    Use a filtered list comprehension to get all even numbers
#    from 1 to 20 (inclusive). Return the list.
#
#  C# Analogy:
#    Enumerable.Range(1,20).Where(x => x % 2 == 0).ToList()
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 2: Filtered List Comprehension (Evens Only)")  # section title
print("-" * 50)    # separator
print()            # blank line


def even_numbers():
    """
    Get all even numbers from 1 to 20 using a filtered list comprehension.

    Returns:
        list: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    """
    # TODO: Use a list comprehension with an 'if' filter
    # return [x for x in range(1, 21) if x % 2 == 0]
    # Breakdown:
    #   x           -- expression: keep the number itself (no transformation)
    #   for x       -- x is each item
    #   in range(1, 21) -- numbers 1 through 20
    #   if x % 2 == 0   -- only include if x is even (remainder = 0)
    pass   # replace with your code


result2 = even_numbers()          # call the function
if result2 is not None:           # only print if something returned
    print(f"  Even numbers 1-20: {result2}")
print()
print("  Expected: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Dict comprehension
#
#  Background:
#    A dict comprehension builds a dictionary in one line.
#    Form: {KEY_EXPR: VALUE_EXPR for ITEM in ITERABLE}
#
#    Example: map each word to its length
#      words = ["cat", "elephant", "bee"]
#      lengths = {word: len(word) for word in words}
#      -> {"cat": 3, "elephant": 8, "bee": 3}
#
#    Long way:
#      lengths = {}
#      for word in words:
#          lengths[word] = len(word)
#
#    C# LINQ equivalent:
#      words.ToDictionary(w => w, w => w.Length)
#
#  Your Task:
#    Given words = ["apple", "banana", "fig", "kiwi", "blueberry"],
#    build a dict mapping each word to its LENGTH.
#    Return the dict.
#
#  C# Analogy:
#    words.ToDictionary(w => w, w => w.Length)
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 3: Dict Comprehension (Word Lengths)")  # section title
print("-" * 50)    # separator
print()            # blank line


def word_lengths():
    """
    Build a dict mapping each word to its length.

    Returns:
        dict: {word: length_int}
    """
    # TODO: Use a dict comprehension
    # words = ["apple", "banana", "fig", "kiwi", "blueberry"]  # list of words
    # return {word: len(word) for word in words}
    # Breakdown:
    #   word        -- the KEY (each word from the list)
    #   len(word)   -- the VALUE (number of characters in word)
    #   for word in words -- iterate over each word
    pass   # replace with your code


result3 = word_lengths()          # call the function
if result3 is not None:           # only print if something returned
    for word, length in sorted(result3.items()):   # sorted for consistent display
        print(f"  {word:10} -> {length}")
print()
print("  Expected: apple=5, banana=6, blueberry=9, fig=3, kiwi=4")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Nested list comprehension -- flatten 2D list
#
#  Background:
#    You can nest comprehensions to work with 2D structures.
#    To FLATTEN a 2D list (list of lists) into a 1D list:
#
#      matrix = [[1,2,3], [4,5,6], [7,8,9]]
#      flat = [item for row in matrix for item in row]
#      -> [1, 2, 3, 4, 5, 6, 7, 8, 9]
#
#    Read it as: "give me each ITEM, for each ROW in matrix, for each ITEM in that row"
#
#    Long way:
#      flat = []
#      for row in matrix:
#          for item in row:
#              flat.append(item)
#
#    C# LINQ equivalent:
#      matrix.SelectMany(row => row).ToList()
#
#  Your Task:
#    Given matrix = [[1,2,3],[4,5,6],[7,8,9]], flatten it to a 1D list.
#    Return: [1, 2, 3, 4, 5, 6, 7, 8, 9]
#
#  C# Analogy:
#    matrix.SelectMany(row => row).ToList()
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 4: Nested Comprehension (Flatten 2D List)")  # section title
print("-" * 50)    # separator
print()            # blank line


def flatten_matrix():
    """
    Flatten a 3x3 matrix into a single 1D list.

    Returns:
        list: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    # TODO: Use a nested comprehension to flatten
    # matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 3 rows of 3 items each
    # return [item for row in matrix for item in row]
    # Breakdown:
    #   item            -- expression: keep the item itself
    #   for row in matrix -- outer loop: each row
    #   for item in row   -- inner loop: each item in that row
    # Note: the ORDER matters! Outer loop comes first.
    pass   # replace with your code


result4 = flatten_matrix()        # call the function
if result4 is not None:           # only print if something returned
    print(f"  Flattened: {result4}")
print()
print("  Matrix was: [[1,2,3],[4,5,6],[7,8,9]]")
print("  Expected:   [1, 2, 3, 4, 5, 6, 7, 8, 9]")
print()


# ============================================================
#  EXERCISE 5
#  Topic: Combined filter + transform (Where + Select in one line)
#
#  Background:
#    You can combine filtering AND transforming in one comprehension.
#    This replaces a .Where(...).Select(...) LINQ chain.
#
#    Example: squares of EVEN numbers only
#      [x**2 for x in range(1,11) if x % 2 == 0]
#      -> [4, 16, 36, 64, 100]
#
#    C# LINQ equivalent:
#      Enumerable.Range(1,10).Where(x=>x%2==0).Select(x=>x*x).ToList()
#
#    The comprehension is shorter but reads the same way:
#      "x squared, for each x from 1 to 10, only if x is even"
#
#  Your Task:
#    Given a list of words, return ONLY words that are longer than
#    4 characters, converted to UPPERCASE.
#    words = ["cat","elephant","bee","apple","fig","banana","kiwi"]
#    Expected: words with len > 4, uppercased
#
#  C# Analogy:
#    words.Where(w=>w.Length>4).Select(w=>w.ToUpper()).ToList()
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 5: Combined Filter + Transform")  # section title
print("-" * 50)    # separator
print()            # blank line


def long_words_upper():
    """
    Return words longer than 4 characters, converted to uppercase.

    Returns:
        list: Uppercase versions of words with length > 4.
    """
    # TODO: Use a comprehension with filter and transformation
    # words = ["cat", "elephant", "bee", "apple", "fig", "banana", "kiwi"]
    # return [word.upper() for word in words if len(word) > 4]
    # Breakdown:
    #   word.upper()    -- EXPRESSION: convert to uppercase
    #   for word in words -- iterate each word
    #   if len(word) > 4  -- FILTER: only words longer than 4 chars
    pass   # replace with your code


result5 = long_words_upper()      # call the function
if result5 is not None:           # only print if something returned
    print(f"  Long words (uppercased): {result5}")
print()
print("  Words tested: cat, elephant, bee, apple, fig, banana, kiwi")
print("  Kept (len>4): elephant(8), apple(5), banana(6)")
print("  Expected: ['ELEPHANT', 'APPLE', 'BANANA']")
print()

print("=" * 60)                       # closing separator
print("All exercises complete!")      # completion message
print("=" * 60)                       # closing separator

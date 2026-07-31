"""
Module 01 - Python Basics
Exercise 05: Lists and Tuples

GLOSSARY
--------
list        : An ORDERED, MUTABLE (changeable) collection. Written with square brackets [].
              Like C# List<T> or an array, but it can hold mixed types.
              Example: [1, "hello", 3.14, True]
tuple       : An ORDERED, IMMUTABLE (unchangeable) collection. Written with parentheses ().
              Like a C# readonly struct or ValueTuple. Cannot add/remove/change items.
              Example: (1, "hello", 3.14)
index       : The position of an item. Python starts at 0 (like C# arrays).
              my_list[0] = first item.   my_list[-1] = LAST item (negative indexing!).
slice       : Extract a portion of a list/tuple. Syntax: list[start:end:step].
              Like C# LINQ Skip().Take() but much shorter to write.
append()    : Add one item to the END of a list. Like C# List.Add().
pop()       : Remove and return the LAST item (or item at given index).
              Like C# List.RemoveAt() + return.
len()       : Returns the number of items in a list/tuple/string. Like C# .Count or .Length.
mutable     : Can be changed after creation. Lists are mutable.
immutable   : Cannot be changed after creation. Tuples and strings are immutable.
sorted()    : Returns a NEW sorted list without changing the original.
              Like C# LINQ .OrderBy().ToList().
"""

print("=" * 60)    # "=" repeated 60 times -- visual separator
print("Exercise 05: Lists and Tuples")  # exercise title
print("=" * 60)    # separator
print()            # blank line


# ============================================================
#  EXERCISE 1
#  Topic: Creating a list and basic operations
#
#  Background:
#    A list is like a dynamic array that can hold anything.
#    In C#: var items = new List<string>();
#    Python: items = []     (square brackets, starts empty)
#
#    Key operations:
#      items.append("apple")    # add to end -- like C# .Add()
#      items[0]                 # access by index (0 = first)
#      items[-1]                # access LAST item (unique to Python!)
#      len(items)               # number of items -- like C# .Count
#
#  Your Task:
#    1. Create an empty list called 'fruits'
#    2. Append "apple", "banana", "cherry" to it
#    3. Return a tuple: (length_of_list, last_item)
#
#  C# Analogy:
#    var fruits = new List<string>();
#    fruits.Add("apple"); fruits.Add("banana"); fruits.Add("cherry");
#    return (fruits.Count, fruits[fruits.Count - 1]);
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 1: Create a List and Use append()")  # section title
print("-" * 50)    # separator
print()            # blank line


def build_fruit_list():
    """
    Create a list of fruits and return its length and last item.

    Returns:
        tuple: (int length, str last_item)
    """
    # TODO: Build the list and return (length, last_item)
    # fruits = []               # create an empty list
    # fruits.append("apple")    # add "apple" to the end
    # fruits.append("banana")   # add "banana" to the end
    # fruits.append("cherry")   # add "cherry" to the end
    # length = len(fruits)      # len() counts items -- like C# .Count
    # last_item = fruits[-1]    # [-1] means last item (Python trick!)
    # return (length, last_item)
    pass   # replace with your code


result1 = build_fruit_list()      # call the function
if result1 is not None:           # only print if something returned
    length, last = result1        # unpack the two values
    print(f"  Length    : {length}")   # should be 3
    print(f"  Last item : {last}")     # should be "cherry"
print()
print("  Expected: length=3, last='cherry'")
print()


# ============================================================
#  EXERCISE 2
#  Topic: List slicing
#
#  Background:
#    Slicing lets you extract part of a list without a loop.
#    Syntax: list[start : end : step]
#      start = index to begin at (inclusive, default 0)
#      end   = index to stop at  (EXCLUSIVE, default end of list)
#      step  = how many to skip  (default 1)
#
#    Examples:
#      nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#      nums[2:5]    -> [2, 3, 4]          (indices 2,3,4)
#      nums[:3]     -> [0, 1, 2]          (first 3)
#      nums[7:]     -> [7, 8, 9]          (from index 7 to end)
#      nums[::2]    -> [0, 2, 4, 6, 8]   (every other item)
#      nums[::-1]   -> [9,8,7,...,0]      (REVERSED!)
#
#    C# LINQ equivalent:
#      nums.Skip(2).Take(3).ToList()  ->  [2, 3, 4]
#
#  Your Task:
#    Given nums = [0,1,2,3,4,5,6,7,8,9], return:
#    (first_three, last_three, every_other, reversed_list)
#
#  C# Analogy:
#    nums.Take(3).ToList()
#    nums.TakeLast(3).ToList()
#    nums.Where((x,i) => i%2==0).ToList()
#    nums.Reverse().ToList()
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 2: List Slicing")  # section title
print("-" * 50)    # separator
print()            # blank line


def slice_list():
    """
    Demonstrate list slicing on [0,1,2,3,4,5,6,7,8,9].

    Returns:
        tuple: (first_three, last_three, every_other, reversed_list)
    """
    # TODO: Use slicing to extract parts of the list
    # nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # create the list
    # first_three  = nums[:3]       # indices 0,1,2 -- first 3 items
    # last_three   = nums[7:]       # index 7 to end -- last 3 items
    # every_other  = nums[::2]      # step=2 means every other item
    # reversed_list = nums[::-1]    # step=-1 means go backwards (reverse)
    # return (first_three, last_three, every_other, reversed_list)
    pass   # replace with your code


result2 = slice_list()        # call the function
if result2 is not None:       # only print if something returned
    first, last, other, rev = result2   # unpack 4 values
    print(f"  First 3     : {first}")   # [0, 1, 2]
    print(f"  Last 3      : {last}")    # [7, 8, 9]
    print(f"  Every other : {other}")   # [0, 2, 4, 6, 8]
    print(f"  Reversed    : {rev}")     # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
print()
print("  Expected:")
print("    First 3:      [0, 1, 2]")
print("    Last 3:       [7, 8, 9]")
print("    Every other:  [0, 2, 4, 6, 8]")
print("    Reversed:     [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]")
print()


# ============================================================
#  EXERCISE 3
#  Topic: sorted() -- sort without modifying original
#
#  Background:
#    Python has two ways to sort a list:
#      list.sort()    -- sorts IN PLACE (modifies the original). Returns None.
#      sorted(list)   -- returns a NEW sorted list. Original is unchanged.
#
#    In C#:
#      list.Sort()                -- sorts in place (modifies)
#      list.OrderBy(x=>x).ToList() -- returns new sorted list
#
#    Optional parameter: reverse=True to sort descending.
#      sorted([3,1,2], reverse=True)  -> [3, 2, 1]
#
#  Your Task:
#    Given numbers = [5, 2, 8, 1, 9, 3], return:
#    (sorted_asc, sorted_desc, original_unchanged)
#    The original list must NOT be changed.
#
#  C# Analogy:
#    var asc  = nums.OrderBy(x=>x).ToList();
#    var desc = nums.OrderByDescending(x=>x).ToList();
#    // nums is unchanged
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 3: sorted() -- Sort Without Modifying")  # section title
print("-" * 50)    # separator
print()            # blank line


def sort_list():
    """
    Sort a list in two directions without modifying the original.

    Returns:
        tuple: (sorted_asc, sorted_desc, original)
    """
    # TODO: Use sorted() to create sorted copies
    # numbers = [5, 2, 8, 1, 9, 3]           # the original list
    # sorted_asc  = sorted(numbers)           # sorted ascending (smallest first)
    # sorted_desc = sorted(numbers, reverse=True)  # sorted descending (largest first)
    # return (sorted_asc, sorted_desc, numbers)    # original is unchanged!
    pass   # replace with your code


result3 = sort_list()         # call the function
if result3 is not None:       # only print if something returned
    asc, desc, orig = result3   # unpack 3 values
    print(f"  Original    : {orig}")   # [5, 2, 8, 1, 9, 3] -- unchanged
    print(f"  Sorted asc  : {asc}")   # [1, 2, 3, 5, 8, 9]
    print(f"  Sorted desc : {desc}")  # [9, 8, 5, 3, 2, 1]
print()
print("  Expected:")
print("    Original:   [5, 2, 8, 1, 9, 3]  (unchanged!)")
print("    Sorted asc: [1, 2, 3, 5, 8, 9]")
print("    Sorted desc:[9, 8, 5, 3, 2, 1]")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Tuples -- immutable sequences
#
#  Background:
#    A tuple is like a list, but IMMUTABLE (cannot be changed).
#    Created with parentheses () instead of square brackets [].
#    Why use tuples?
#      - For data that should not change (coordinates, RGB color, etc.)
#      - Slightly faster than lists
#      - Can be used as dictionary keys (lists cannot!)
#
#    In C#: like a record or readonly struct that you can't change.
#
#    Accessing elements is the same as a list: tuple[0], tuple[-1]
#    Tuples support slicing too.
#
#    Trying to change a tuple:
#      point = (3, 4)
#      point[0] = 10    <- TypeError: 'tuple' object does not support item assignment
#
#  Your Task:
#    1. Create a tuple called 'point' with values (3, 4) -- an (x,y) coordinate
#    2. Return (x, y, the_sum) where the_sum = x + y
#    3. Also demonstrate immutability: try to change point[0], catch the error
#
#  C# Analogy:
#    var point = (3, 4);   // ValueTuple
#    return (point.Item1, point.Item2, point.Item1 + point.Item2);
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 4: Tuples (Immutable Sequences)")  # section title
print("-" * 50)    # separator
print()            # blank line


def use_tuple():
    """
    Create and use a tuple, demonstrate immutability.

    Returns:
        tuple: (x, y, sum_of_x_y)
    """
    # TODO: Create a point tuple and return values
    # point = (3, 4)       # create a tuple with two values (x=3, y=4)
    # x = point[0]         # access first element (index 0)
    # y = point[1]         # access second element (index 1)
    # the_sum = x + y      # compute the sum
    # return (x, y, the_sum)
    pass   # replace with your code


result4 = use_tuple()          # call the function
if result4 is not None:        # only print if something returned
    x, y, total = result4      # unpack 3 values
    print(f"  x       : {x}")        # 3
    print(f"  y       : {y}")        # 4
    print(f"  x + y   : {total}")    # 7

# Demonstrate immutability: try to change a tuple element
point = (3, 4)                      # create the tuple
print()
print("  Demonstrating immutability:")
try:                                 # 'try' attempts the code block
    point[0] = 10                   # this will FAIL because tuples are immutable
except TypeError as e:               # catch the TypeError that Python raises
    print(f"  Cannot change tuple: {e}")   # show the error message
print()
print("  Expected: x=3, y=4, sum=7, then a TypeError about immutability")
print()


# ============================================================
#  EXERCISE 5
#  Topic: List of lists (2D list / matrix)
#
#  Background:
#    A 2D list is a list where each item is another list.
#    This is like a matrix or a 2D array.
#    In C#:  int[,] matrix = new int[3,3];
#    Python: matrix = [[1,2,3], [4,5,6], [7,8,9]]
#
#    Accessing elements:
#      matrix[row][col]    -- first index is the row, second is the column
#      matrix[0][0] = 1   (row 0, col 0)
#      matrix[1][2] = 6   (row 1, col 2)
#
#    In C#: matrix[0,0] = 1; (uses comma)
#    Python: matrix[0][0]    (uses two brackets)
#
#  Your Task:
#    Create a 3x3 matrix:
#      [[1, 2, 3],
#       [4, 5, 6],
#       [7, 8, 9]]
#    Return the element at row=1, col=2 and the entire middle row.
#
#  C# Analogy:
#    int[,] m = {{1,2,3},{4,5,6},{7,8,9}};
#    return (m[1,2], new[]{m[1,0],m[1,1],m[1,2]});
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 5: 2D List (Matrix)")  # section title
print("-" * 50)    # separator
print()            # blank line


def use_2d_list():
    """
    Create a 3x3 matrix and access elements.

    Returns:
        tuple: (element_at_row1_col2, middle_row)
    """
    # TODO: Create the matrix and access elements
    # matrix = [          # outer list contains 3 inner lists (rows)
    #     [1, 2, 3],      # row 0: three numbers
    #     [4, 5, 6],      # row 1: three numbers
    #     [7, 8, 9],      # row 2: three numbers
    # ]
    # element = matrix[1][2]   # row 1, col 2 -- that's the value 6
    # middle_row = matrix[1]   # the entire row at index 1 -- that's [4, 5, 6]
    # return (element, middle_row)
    pass   # replace with your code


result5 = use_2d_list()        # call the function
if result5 is not None:        # only print if something returned
    elem, row = result5        # unpack 2 values
    print(f"  matrix[1][2]  : {elem}")   # should be 6
    print(f"  middle row    : {row}")    # should be [4, 5, 6]
print()
print("  Matrix:")
print("    [[1, 2, 3],")
print("     [4, 5, 6],  <- middle row (index 1)")
print("     [7, 8, 9]]")
print("  Expected: element=6, middle_row=[4, 5, 6]")
print()

print("=" * 60)                       # closing separator
print("All exercises complete!")      # completion message
print("=" * 60)                       # closing separator

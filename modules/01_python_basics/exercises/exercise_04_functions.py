"""
Module 01 - Python Basics
Exercise 04: Functions

GLOSSARY
--------
def           : Keyword to DEFINE a function. Like 'void' or a return type in C#.
                Example: def greet(name):  is like  string Greet(string name) in C#.
parameter     : A variable listed in the function definition. The "slot" for input.
                def add(a, b):  -- 'a' and 'b' are parameters.
argument      : The actual value you pass when calling a function.
                add(3, 5)  -- 3 and 5 are arguments.
return        : Sends a value back to the caller. Same as C# return.
                A function without return (or just 'return') gives back None.
default param : A parameter with a preset value. Caller can skip it.
                def greet(name="World"):   -- like C# optional parameters.
*args         : Collects any number of positional arguments into a TUPLE.
                Like C# params keyword: void Sum(params int[] nums).
**kwargs      : Collects any number of keyword arguments into a DICT.
                Like C# Dictionary<string,object> for named parameters.
docstring     : A string right after 'def' that documents the function.
                Like XML comments (///<summary>) in C#. Use triple quotes.
recursion     : A function that calls ITSELF. Used to break problems into
                smaller identical sub-problems. Same concept as C# recursion.
"""

print("=" * 60)    # "=" repeated 60 times -- visual separator
print("Exercise 04: Functions")  # exercise title
print("=" * 60)    # separator
print()            # blank line


# ============================================================
#  EXERCISE 1
#  Topic: Simple function with parameters and return
#
#  Background:
#    In C# you declare a function with the return type first:
#      int Add(int a, int b) { return a + b; }
#
#    In Python you just use 'def' (no type declarations needed):
#      def add(a, b):         # 'def' keyword, function name, parameters
#          return a + b       # return the result (indented 4 spaces)
#
#    Calling it is the same:
#      C#:     int result = Add(3, 5);
#      Python: result = add(3, 5)
#
#  Your Task:
#    Write a function that takes two numbers 'a' and 'b' and
#    returns their sum. Use the function name 'add_numbers'.
#
#  C# Analogy:
#    int AddNumbers(int a, int b) { return a + b; }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 1: Simple function (sum of two numbers)")  # section title
print("-" * 50)    # separator
print()            # blank line


def add_numbers(a, b):
    """
    Return the sum of two numbers.

    Args:
        a: First number (int or float).
        b: Second number (int or float).

    Returns:
        The sum of a and b.
    """
    # TODO: Return the sum of a and b
    # return a + b   # the + operator adds numbers; return sends it back to caller
    pass   # replace with your code


print(f"  add_numbers(3, 5)    -> {add_numbers(3, 5)}")      # should be 8
print(f"  add_numbers(10, -4)  -> {add_numbers(10, -4)}")    # should be 6
print(f"  add_numbers(1.5, 2.5)-> {add_numbers(1.5, 2.5)}") # should be 4.0
print()
print("  Expected: 8, 6, 4.0")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Default parameter values
#
#  Background:
#    You can give a parameter a DEFAULT value. If the caller doesn't
#    pass that argument, Python uses the default.
#
#    In C#:   string Greet(string name = "World") { return $"Hello, {name}!"; }
#    Python:  def greet(name="World"):             return f"Hello, {name}!"
#
#    Calling examples:
#      greet()           -> "Hello, World!"  (uses default)
#      greet("Alice")    -> "Hello, Alice!"  (overrides default)
#      greet(name="Bob") -> "Hello, Bob!"    (keyword argument)
#
#    RULE: Default parameters must come AFTER non-default ones.
#    def greet(greeting, name="World"):  -- OK
#    def greet(name="World", greeting):  -- ERROR!
#
#  Your Task:
#    Write a function 'greet' that takes:
#      - 'greeting' (required, no default)
#      - 'name' (optional, default = "World")
#    Return the string: "{greeting}, {name}!"
#
#  C# Analogy:
#    string Greet(string greeting, string name = "World") {
#        return $"{greeting}, {name}!";
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 2: Default Parameter Values")  # section title
print("-" * 50)    # separator
print()            # blank line


def greet(greeting, name="World"):
    """
    Build a greeting string with an optional name.

    Args:
        greeting (str): The greeting word (e.g., "Hello").
        name (str): The name to greet. Defaults to "World".

    Returns:
        str: A formatted greeting like "Hello, World!"
    """
    # TODO: Return the greeting string using an f-string
    # return f"{greeting}, {name}!"  # embed greeting and name into the string
    pass   # replace with your code


print(f"  greet('Hello')           -> {greet('Hello')}")          # Hello, World!
print(f"  greet('Hello', 'Alice')  -> {greet('Hello', 'Alice')}") # Hello, Alice!
print(f"  greet('Hi', name='Bob')  -> {greet('Hi', name='Bob')}") # Hi, Bob!
print()
print("  Expected: 'Hello, World!', 'Hello, Alice!', 'Hi, Bob!'")
print()


# ============================================================
#  EXERCISE 3
#  Topic: *args -- variable number of arguments
#
#  Background:
#    Sometimes you don't know how many arguments the caller will pass.
#    *args collects ALL extra positional arguments into a TUPLE.
#
#    In C#:  int Sum(params int[] nums) { ... }
#    Python: def sum_all(*args):          -- args is a tuple of all passed values
#
#    Inside the function:
#      def sum_all(*args):
#          total = 0
#          for n in args:    # iterate over the tuple
#              total += n
#          return total
#
#    Calling examples:
#      sum_all(1, 2, 3)       -> 6   (args = (1, 2, 3))
#      sum_all(10, 20)        -> 30  (args = (10, 20))
#      sum_all(5)             -> 5   (args = (5,))
#
#  Your Task:
#    Write 'sum_all(*args)' that sums any number of values.
#    Return the total sum. Return 0 if no arguments given.
#
#  C# Analogy:
#    int SumAll(params int[] nums) {
#        int total = 0;
#        foreach(int n in nums) total += n;
#        return total;
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 3: *args (variable arguments)")  # section title
print("-" * 50)    # separator
print()            # blank line


def sum_all(*args):
    """
    Sum any number of values passed as arguments.

    Args:
        *args: Any number of numeric values.

    Returns:
        int or float: Sum of all arguments, or 0 if none given.
    """
    # TODO: Loop over args and sum them up
    # total = 0           # start the running total at zero
    # for n in args:      # args is a tuple -- iterate over each value
    #     total += n      # add each value to the running total
    # return total        # return the final sum
    pass   # replace with your code


print(f"  sum_all(1, 2, 3)       -> {sum_all(1, 2, 3)}")      # 6
print(f"  sum_all(10, 20)        -> {sum_all(10, 20)}")        # 30
print(f"  sum_all(5)             -> {sum_all(5)}")             # 5
print(f"  sum_all()              -> {sum_all()}")              # 0
print(f"  sum_all(1,2,3,4,5,6)  -> {sum_all(1,2,3,4,5,6)}")  # 21
print()
print("  Expected: 6, 30, 5, 0, 21")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Returning multiple values
#
#  Background:
#    Python functions can return MULTIPLE values easily -- just
#    separate them with commas. Python packs them into a TUPLE.
#
#    In C#, you'd use a ValueTuple or out parameters:
#      (int min, int max) FindMinMax(int[] nums) { ... }
#
#    In Python:
#      def find_min_max(numbers):
#          return min(numbers), max(numbers)   # returns a tuple
#
#    min() and max() are Python built-in functions.
#    min([3,1,4,1,5]) -> 1     (smallest value)
#    max([3,1,4,1,5]) -> 5     (largest value)
#
#  Your Task:
#    Write 'find_min_max(numbers)' that returns (min_val, max_val).
#    Also return the RANGE (max - min) as a third value.
#    Return (min_val, max_val, range_val).
#
#  C# Analogy:
#    (int min, int max, int range) = FindMinMax(nums);
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 4: Returning Multiple Values")  # section title
print("-" * 50)    # separator
print()            # blank line


def find_min_max(numbers):
    """
    Find the minimum, maximum, and range of a list of numbers.

    Args:
        numbers (list): A list of numeric values.

    Returns:
        tuple: (min_val, max_val, range_val)
    """
    # TODO: Use built-in min() and max() functions
    # min_val   = min(numbers)            # min() returns the smallest value
    # max_val   = max(numbers)            # max() returns the largest value
    # range_val = max_val - min_val       # range = max minus min
    # return (min_val, max_val, range_val)  # return all three as a tuple
    pass   # replace with your code


data = [3, 1, 4, 1, 5, 9, 2, 6]    # test list
result4 = find_min_max(data)         # call the function
if result4 is not None:              # only print if something returned
    lo, hi, rng = result4            # unpack three return values
    print(f"  Data     : {data}")
    print(f"  Min      : {lo}")      # should be 1
    print(f"  Max      : {hi}")      # should be 9
    print(f"  Range    : {rng}")     # should be 8
print()
print("  Expected: min=1, max=9, range=8")
print()


# ============================================================
#  EXERCISE 5
#  Topic: Recursive functions
#
#  Background:
#    Recursion: a function that calls ITSELF to solve a smaller
#    version of the same problem.
#
#    Classic example: factorial
#      5! = 5 * 4 * 3 * 2 * 1 = 120
#      4! = 4 * 3 * 2 * 1     = 24
#      So: 5! = 5 * 4!         <- recursion!
#
#    In Python:
#      def factorial(n):
#          if n <= 1:              # BASE CASE: stop recursing here
#              return 1
#          return n * factorial(n - 1)  # RECURSIVE CASE: call self with n-1
#
#    The BASE CASE is critical -- without it, the function calls itself
#    forever (stack overflow!). Same rule as C# recursion.
#
#    Call stack for factorial(3):
#      factorial(3) -> 3 * factorial(2)
#                          -> 2 * factorial(1)
#                                  -> 1 (base case!)
#                          -> 2 * 1 = 2
#      -> 3 * 2 = 6
#
#  Your Task:
#    Write 'factorial(n)' using recursion.
#    Return 1 for n <= 1 (base case).
#    Otherwise return n * factorial(n - 1).
#
#  C# Analogy:
#    int Factorial(int n) {
#        if (n <= 1) return 1;
#        return n * Factorial(n - 1);
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 5: Recursive Function (Factorial)")  # section title
print("-" * 50)    # separator
print()            # blank line


def factorial(n):
    """
    Compute n! (n factorial) using recursion.

    Args:
        n (int): A non-negative integer.

    Returns:
        int: n! = n * (n-1) * ... * 2 * 1. Returns 1 for n <= 1.
    """
    # TODO: Add the base case and recursive case
    # if n <= 1:                    # BASE CASE: 0! = 1, 1! = 1
    #     return 1                  # stop here, return 1
    # return n * factorial(n - 1)   # RECURSIVE CASE: n! = n * (n-1)!
    pass   # replace with your code


for i in range(1, 8):               # test factorial for 1 through 7
    result = factorial(i)           # compute factorial of i
    print(f"  factorial({i}) = {result}")   # show result
print()
print("  Expected: 1, 2, 6, 24, 120, 720, 5040")
print()

print("=" * 60)                       # closing separator
print("All exercises complete!")      # completion message
print("=" * 60)                       # closing separator

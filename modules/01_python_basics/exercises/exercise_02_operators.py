"""
Module 01 - Python Basics
Exercise 02: Operators

GLOSSARY
--------
arithmetic operator : Symbols that perform math: + (add), - (subtract), * (multiply),
                      / (divide), // (floor divide), % (modulo), ** (power).
modulo              : The % operator. Returns the REMAINDER after division.
                      10 % 3 = 1 because 10 / 3 = 3 remainder 1.
                      Like C# %  operator. Useful for checking even/odd.
floor division      : The // operator. Divides and ROUNDS DOWN to nearest whole number.
                      7 // 2 = 3 (not 3.5). No equivalent symbol in C#; use Math.Floor(7.0/2).
comparison operator : Symbols that compare two values and return True or False:
                      == (equal), != (not equal), < (less), > (greater), <= (<=), >= (>=).
                      Same as C# comparison operators.
logical operator    : 'and', 'or', 'not' -- combine boolean conditions.
                      Python uses words; C# uses symbols: && || !
short-circuit       : When Python stops evaluating as soon as the result is certain.
                      'False and X' -- Python never checks X because False and anything = False.
                      Same behavior as C# && and ||.
augmented assignment: Shorthand for updating a variable: x += 1 means x = x + 1.
                      Same as C# x += 1. Works for all arithmetic operators.
PEMDAS              : Order of operations: Parentheses, Exponents, Multiply/Divide, Add/Subtract.
                      Python follows this same order.
"""

print("=" * 60)    # "=" repeated 60 times, creates a visual separator
print("Exercise 02: Operators")  # exercise title
print("=" * 60)    # another separator line
print()            # blank line for readability


# ============================================================
#  EXERCISE 1
#  Topic: Arithmetic operators
#
#  Background:
#    Python has all the standard math operators plus two special ones:
#      +   addition           (same as C#)
#      -   subtraction        (same as C#)
#      *   multiplication     (same as C#)
#      /   division           ALWAYS returns float! (C# int/int truncates)
#      //  floor division     rounds down to int (like C# (int)(a/b))
#      %   modulo/remainder   (same as C#)
#      **  power/exponent     (C# uses Math.Pow(a, b))
#
#    Python trick: 10 / 3 = 3.333... (float)
#                  10 // 3 = 3       (int, floor)
#
#  Your Task:
#    Given a=10 and b=3, return a tuple with:
#    (a+b, a-b, a*b, a/b, a//b, a%b, a**b)
#
#  C# Analogy:
#    (a+b, a-b, a*b, (double)a/b, a/b, a%b, Math.Pow(a,b))
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 1: Arithmetic Operators")  # section title
print("-" * 50)    # separator
print()            # blank line


def arithmetic_operators(a, b):
    """
    Perform all arithmetic operations on a and b.

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        tuple: (add, subtract, multiply, divide, floor_div, modulo, power)
    """
    # TODO: Compute each result and return them as a tuple
    # add      = a + b    # addition: 10 + 3 = 13
    # subtract = a - b    # subtraction: 10 - 3 = 7
    # multiply = a * b    # multiplication: 10 * 3 = 30
    # divide   = a / b    # division: 10 / 3 = 3.333...  (always float in Python!)
    # floor_div = a // b  # floor division: 10 // 3 = 3  (drops decimal, rounds down)
    # modulo   = a % b    # remainder: 10 % 3 = 1  (10 = 3*3 + 1)
    # power    = a ** b   # exponent: 10 ** 3 = 1000  (10 cubed)
    # return (add, subtract, multiply, divide, floor_div, modulo, power)
    pass   # replace with your code


result1 = arithmetic_operators(10, 3)   # call with a=10, b=3
if result1 is not None:                 # only print if student returned something
    add, sub, mul, div, fdiv, mod, pwr = result1   # unpack all 7 values
    print(f"  10 + 3  = {add}")      # should be 13
    print(f"  10 - 3  = {sub}")      # should be 7
    print(f"  10 * 3  = {mul}")      # should be 30
    print(f"  10 / 3  = {div}")      # should be 3.3333... (float!)
    print(f"  10 // 3 = {fdiv}")     # should be 3 (floor division)
    print(f"  10 % 3  = {mod}")      # should be 1 (remainder)
    print(f"  10 ** 3 = {pwr}")      # should be 1000
print()
print("  Expected: 13, 7, 30, 3.3333, 3, 1, 1000")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Comparison operators
#
#  Background:
#    Comparison operators compare two values and return True or False.
#    They are the same symbols as C#:
#      ==  equal to              (C# ==)
#      !=  not equal to          (C# !=)
#      <   less than             (C# <)
#      >   greater than          (C# >)
#      <=  less than or equal    (C# <=)
#      >=  greater than or equal (C# >=)
#
#    WARNING: = is assignment (x = 5), == is comparison (x == 5).
#    This is the same in C# -- don't confuse them!
#
#  Your Task:
#    Given x=7 and y=10, return a tuple of 6 booleans:
#    (x==y, x!=y, x<y, x>y, x<=y, x>=y)
#
#  C# Analogy:
#    return (x==y, x!=y, x<y, x>y, x<=y, x>=y);
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 2: Comparison Operators")  # section title
print("-" * 50)    # separator
print()            # blank line


def comparison_operators(x, y):
    """
    Apply all six comparison operators to x and y.

    Args:
        x (int): First number.
        y (int): Second number.

    Returns:
        tuple: 6 booleans from comparisons.
    """
    # TODO: Compute each comparison and return as a tuple
    # eq  = x == y    # equal to: 7 == 10 -> False
    # neq = x != y    # not equal: 7 != 10 -> True
    # lt  = x < y     # less than: 7 < 10 -> True
    # gt  = x > y     # greater than: 7 > 10 -> False
    # lte = x <= y    # less than or equal: 7 <= 10 -> True
    # gte = x >= y    # greater than or equal: 7 >= 10 -> False
    # return (eq, neq, lt, gt, lte, gte)
    pass   # replace with your code


result2 = comparison_operators(7, 10)   # call with x=7, y=10
if result2 is not None:                 # only print if something returned
    eq, neq, lt, gt, lte, gte = result2   # unpack 6 booleans
    print(f"  7 == 10 -> {eq}")    # False
    print(f"  7 != 10 -> {neq}")   # True
    print(f"  7 <  10 -> {lt}")    # True
    print(f"  7 >  10 -> {gt}")    # False
    print(f"  7 <= 10 -> {lte}")   # True
    print(f"  7 >= 10 -> {gte}")   # False
print()
print("  Expected: False, True, True, False, True, False")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Logical operators (and, or, not)
#
#  Background:
#    Python uses WORDS for logical operators; C# uses SYMBOLS:
#      Python: and      C#: &&
#      Python: or       C#: ||
#      Python: not      C#: !
#
#    Truth tables:
#      True and True   -> True      (both must be True)
#      True and False  -> False
#      True or False   -> True      (at least one must be True)
#      False or False  -> False
#      not True        -> False     (flip the value)
#      not False       -> True
#
#  Your Task:
#    Given booleans a=True, b=False, return:
#    (a and b, a or b, not a, not b, a and not b)
#
#  C# Analogy:
#    return (a && b, a || b, !a, !b, a && !b);
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 3: Logical Operators")  # section title
print("-" * 50)    # separator
print()            # blank line


def logical_operators(a, b):
    """
    Apply logical operators to boolean values a and b.

    Args:
        a (bool): First boolean.
        b (bool): Second boolean.

    Returns:
        tuple: 5 boolean results.
    """
    # TODO: Compute each logical operation and return as a tuple
    # and_result   = a and b      # True AND False = False (both must be True)
    # or_result    = a or b       # True OR False  = True  (one is enough)
    # not_a        = not a        # NOT True = False (flip it)
    # not_b        = not b        # NOT False = True (flip it)
    # a_and_not_b  = a and not b  # True AND (NOT False) = True AND True = True
    # return (and_result, or_result, not_a, not_b, a_and_not_b)
    pass   # replace with your code


result3 = logical_operators(True, False)   # call with a=True, b=False
if result3 is not None:                    # only print if something returned
    r1, r2, r3, r4, r5 = result3          # unpack 5 results
    print(f"  True and False  -> {r1}")   # False
    print(f"  True or  False  -> {r2}")   # True
    print(f"  not True        -> {r3}")   # False
    print(f"  not False       -> {r4}")   # True
    print(f"  True and not F  -> {r5}")   # True
print()
print("  Expected: False, True, False, True, True")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Augmented assignment operators
#
#  Background:
#    Augmented assignment is shorthand for updating a variable.
#    Instead of writing x = x + 1, you write x += 1.
#    Same as C# -- you already know these!
#      x += 5    means x = x + 5
#      x -= 3    means x = x - 3
#      x *= 2    means x = x * 2
#      x //= 4   means x = x // 4   (floor division in-place)
#      x **= 2   means x = x ** 2   (square in-place)
#
#  Your Task:
#    Starting with x=20, apply these operations IN ORDER:
#    1. x += 5     (add 5)
#    2. x -= 3     (subtract 3)
#    3. x *= 2     (multiply by 2)
#    4. x //= 4    (floor divide by 4)
#    5. x **= 2    (square it)
#    Return x after all five steps.
#
#  C# Analogy:
#    int x = 20; x += 5; x -= 3; x *= 2; x /= 4; x = x * x; return x;
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 4: Augmented Assignment")  # section title
print("-" * 50)    # separator
print()            # blank line


def augmented_assignment():
    """
    Apply augmented assignment operators to x=20 in sequence.

    Returns:
        int: Final value of x after all operations.
    """
    # TODO: Apply each operation step by step
    # x = 20      # start with 20
    # x += 5      # x is now 25  (20 + 5)
    # x -= 3      # x is now 22  (25 - 3)
    # x *= 2      # x is now 44  (22 * 2)
    # x //= 4     # x is now 11  (44 // 4 = 11, floor division)
    # x **= 2     # x is now 121 (11 ** 2 = 121)
    # return x
    pass   # replace with your code


result4 = augmented_assignment()      # call the function
if result4 is not None:               # only print if something returned
    print(f"  Final x = {result4}")   # should be 121
print()
print("  Steps: 20 -> +5 = 25 -> -3 = 22 -> *2 = 44 -> //4 = 11 -> **2 = 121")
print("  Expected: 121")
print()


# ============================================================
#  EXERCISE 5
#  Topic: Operator precedence (PEMDAS/BODMAS)
#
#  Background:
#    Python evaluates expressions in this order (same as math class):
#      1. () Parentheses first
#      2. ** Exponents (right to left)
#      3. *, /, //, %  (left to right)
#      4. +, -          (left to right)
#    This is the same as C#.
#
#    Example: 2 + 3 * 4 = 14  (NOT 20!)
#    Because * happens before +: 3*4=12, then 2+12=14.
#    To force addition first: (2 + 3) * 4 = 20
#
#  Your Task:
#    Compute the following expression and return the result:
#      result = 2 + 3 ** 2 * 4 - 10 // 3
#    Hint: work step by step:
#      3 ** 2 = 9
#      9 * 4 = 36
#      10 // 3 = 3
#      2 + 36 - 3 = 35
#
#  C# Analogy:
#    int result = 2 + (int)Math.Pow(3,2) * 4 - 10 / 3;
#    return result;  // same answer: 35
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 5: Operator Precedence (PEMDAS)")  # section title
print("-" * 50)    # separator
print()            # blank line


def operator_precedence():
    """
    Compute an expression that tests knowledge of operator precedence.

    Returns:
        int: Result of 2 + 3 ** 2 * 4 - 10 // 3
    """
    # TODO: Just compute the expression and return it
    # Python follows PEMDAS automatically -- you don't need to add parentheses
    # result = 2 + 3 ** 2 * 4 - 10 // 3
    # return result
    pass   # replace with your code


result5 = operator_precedence()       # call the function
if result5 is not None:               # only print if something returned
    print(f"  2 + 3**2 * 4 - 10//3 = {result5}")   # should be 35
print()
print("  Step-by-step:")
print("    3 ** 2      = 9    (exponent first)")
print("    9 * 4       = 36   (multiply)")
print("    10 // 3     = 3    (floor division)")
print("    2 + 36 - 3  = 35   (add/subtract left to right)")
print("  Expected: 35")
print()

print("=" * 60)                        # closing separator
print("All exercises complete!")       # completion message
print("=" * 60)                        # closing separator

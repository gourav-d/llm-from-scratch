"""
Module 01 - Python Basics
Exercise 01: Variables and Data Types

GLOSSARY
--------
variable       : A named container for a value. Like 'int x = 5' in C#, but no type needed.
                 Python figures out the type automatically (dynamic typing).
int            : Whole numbers. Python int has no size limit (unlike C# int which caps at ~2 billion).
float          : Decimal numbers. Same as C# double (64-bit precision).
str            : Text. Like C# string. Can use single OR double quotes -- both work the same.
bool           : True or False. Capital T and F in Python! C# uses lowercase (true/false).
None           : Python's version of C# null. Means "no value here".
type()         : Built-in function that returns the type of a variable. Like C# GetType().
dynamic typing : Python automatically decides the type of a variable from the value you assign.
                 In C# you declare the type (int, string). In Python you just assign.
snake_case     : Python naming convention: my_variable_name (words joined by underscores).
                 C# uses camelCase (myVariableName) or PascalCase (MyVariableName).
f-string       : A formatted string literal. Like C# $"Hello {name}". Write f"Hello {name}".
"""

print("=" * 60)          # print() writes to the console; "=" * 60 repeats "=" sixty times
print("Exercise 01: Variables and Data Types")  # print the exercise title
print("=" * 60)          # another separator line
print()                  # blank line for readability


# ============================================================
#  EXERCISE 1
#  Topic: Creating variables of different types
#
#  Background:
#    In C# you write:  int age = 25;
#    In Python:        age = 25        (no type keyword, no semicolon!)
#    Python reads the value (25) and decides "this must be an int".
#    This automatic decision is called DYNAMIC TYPING.
#
#  Your Task:
#    Fill in the function body to:
#    1. Create an int variable called 'count' with value 42
#    2. Create a float variable called 'price' with value 9.99
#    3. Create a str variable called 'name' with value "Alice"
#    4. Create a bool variable called 'is_active' with value True
#    5. Return all four as a tuple: (count, price, name, is_active)
#
#  C# Analogy:
#    int count = 42;
#    double price = 9.99;
#    string name = "Alice";
#    bool isActive = true;
#    return (count, price, name, isActive);   // ValueTuple in C#
# ============================================================

print("-" * 50)           # separator line
print("EXERCISE 1: Create Variables")  # section header
print("-" * 50)           # separator line
print()                   # blank line


def create_variables():
    """
    Create variables of four different types and return them.

    Returns:
        tuple: (int, float, str, bool)
    """
    # TODO: Create the four variables below (remove the # to uncomment)
    # count = 42            # int: a whole number, no decimal point
    # price = 9.99          # float: a decimal number
    # name = "Alice"        # str: text surrounded by quotes
    # is_active = True      # bool: capital T! (C# uses lowercase true)
    # return (count, price, name, is_active)   # return all four as a tuple
    pass   # 'pass' is a placeholder -- Python requires something here; replace with your code


result = create_variables()    # call the function and store the result
if result is not None:         # check if the student returned something (not just pass)
    count, price, name, is_active = result  # unpack the tuple into 4 variables
    print(f"  count    : {count}  (type: {type(count).__name__})")      # show value + type name
    print(f"  price    : {price}  (type: {type(price).__name__})")      # type(x).__name__ gives "int", "float" etc.
    print(f"  name     : {name}  (type: {type(name).__name__})")        # __name__ is the class name as a string
    print(f"  is_active: {is_active}  (type: {type(is_active).__name__})")  # should show "bool"
print()
print("  Expected: 42 (int), 9.99 (float), Alice (str), True (bool)")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Type conversion (casting)
#
#  Background:
#    Sometimes you get a string "42" and need the actual number 42.
#    Python has built-in functions to convert between types:
#      int("42")    -> 42        (like C# int.Parse("42"))
#      float("3.14")-> 3.14     (like C# double.Parse("3.14"))
#      str(42)      -> "42"     (like C# 42.ToString())
#      bool(0)      -> False    (0 is falsy; anything non-zero is True)
#
#  Your Task:
#    Fill in the function body to:
#    1. Convert the string "42" to an int and store in 'a'
#    2. Convert the string "3.14" to a float and store in 'b'
#    3. Convert the number 99 to a string and store in 'c'
#    4. Convert the number 0 to a bool and store in 'd'
#    5. Return (a, b, c, d)
#
#  C# Analogy:
#    int a = int.Parse("42");
#    double b = double.Parse("3.14");
#    string c = 99.ToString();
#    bool d = Convert.ToBoolean(0);
# ============================================================

print("-" * 50)           # separator line
print("EXERCISE 2: Type Conversion")  # section header
print("-" * 50)           # separator line
print()                   # blank line


def convert_types():
    """
    Convert values between Python types.

    Returns:
        tuple: (int, float, str, bool)
    """
    # TODO: Fill in the conversions below
    # a = int("42")       # convert string "42" to integer 42
    # b = float("3.14")   # convert string "3.14" to float 3.14
    # c = str(99)         # convert integer 99 to string "99"
    # d = bool(0)         # convert 0 to bool -- 0 means False in Python
    # return (a, b, c, d)
    pass   # replace with your code


result2 = convert_types()      # call the function
if result2 is not None:        # only unpack if something was returned
    a, b, c, d = result2       # unpack the four values
    print(f"  int('42')   -> {a!r}  type: {type(a).__name__}")   # !r shows repr (quotes for strings)
    print(f"  float('3.14')-> {b!r}  type: {type(b).__name__}")
    print(f"  str(99)     -> {c!r}  type: {type(c).__name__}")   # notice the quotes around "99"
    print(f"  bool(0)     -> {d!r}  type: {type(d).__name__}")
print()
print("  Expected: 42 (int), 3.14 (float), '99' (str), False (bool)")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Checking for None
#
#  Background:
#    None is Python's null. To check if something is None, use:
#      if x is None:      (NOT x == None -- always use 'is' for None checks)
#    In C#: if (x == null)
#    'is' checks object identity (same object in memory).
#    For None there is always exactly one None object, so 'is' is correct.
#
#  Your Task:
#    Fill in the function body to:
#    1. Return True if the argument 'value' is None
#    2. Return False otherwise
#
#  C# Analogy:
#    bool IsNull(object value) { return value == null; }
# ============================================================

print("-" * 50)           # separator line
print("EXERCISE 3: Checking for None")  # section header
print("-" * 50)           # separator line
print()                   # blank line


def is_none_value(value):
    """
    Check whether the given value is None.

    Args:
        value: Any Python value.

    Returns:
        bool: True if value is None, False otherwise.
    """
    # TODO: Return True if value is None, False otherwise
    # Hint: use 'value is None'  (not ==)
    # if value is None:     # 'is' checks identity -- the only correct way for None
    #     return True       # it IS None, so return True
    # return False          # it is NOT None, so return False
    pass   # replace with your code


print(f"  is_none_value(None)  -> {is_none_value(None)}")    # should be True
print(f"  is_none_value(0)     -> {is_none_value(0)}")       # should be False (0 is not None)
print(f"  is_none_value('')    -> {is_none_value('')}")      # should be False (empty string is not None)
print(f"  is_none_value(False) -> {is_none_value(False)}")  # should be False (False is not None)
print()
print("  Expected: True, False, False, False")
print()


# ============================================================
#  EXERCISE 4
#  Topic: f-strings (formatted string literals)
#
#  Background:
#    f-strings let you embed variables inside strings.
#    In C# you write:   $"Hello, {name}! You are {age} years old."
#    In Python write:   f"Hello, {name}! You are {age} years old."
#    Same idea, just put 'f' before the opening quote instead of '$'.
#    You can also do math inside: f"Double is {age * 2}"
#
#  Your Task:
#    Fill in the function to return a greeting string:
#    "Hello, Alice! You are 30 years old. Next year you will be 31."
#    Use f-string with the given 'name' and 'age' parameters.
#    Compute next year's age inside the f-string (age + 1).
#
#  C# Analogy:
#    return $"Hello, {name}! You are {age} years old. Next year you will be {age + 1}.";
# ============================================================

print("-" * 50)           # separator line
print("EXERCISE 4: f-strings")  # section header
print("-" * 50)           # separator line
print()                   # blank line


def make_greeting(name, age):
    """
    Build a greeting string using an f-string.

    Args:
        name (str): Person's name.
        age (int): Person's current age.

    Returns:
        str: Formatted greeting message.
    """
    # TODO: Return an f-string greeting
    # The message should be:
    # "Hello, {name}! You are {age} years old. Next year you will be {age + 1}."
    # Hint: f"Hello, {name}! You are {age} years old. Next year you will be {age + 1}."
    pass   # replace with your code


greeting = make_greeting("Alice", 30)   # call with name="Alice", age=30
if greeting is not None:                # only print if student returned something
    print(f"  Result  : {greeting}")    # show the returned greeting
print()
print("  Expected: Hello, Alice! You are 30 years old. Next year you will be 31.")
print()


# ============================================================
#  EXERCISE 5
#  Topic: Constants (naming convention)
#
#  Background:
#    Python has no 'const' keyword (unlike C# const int MAX = 100).
#    Instead, developers use ALL_CAPS names as a CONVENTION to signal
#    "don't change this value".
#    Python won't stop you from changing it, but the ALL_CAPS name
#    tells other programmers: "treat this as a constant".
#
#  Your Task:
#    Fill in the function to:
#    1. Define MAX_SPEED = 100          (int constant)
#    2. Define PI = 3.14159            (float constant)
#    3. Define APP_NAME = "LLM Learner" (str constant)
#    4. Return (MAX_SPEED, PI, APP_NAME)
#
#  C# Analogy:
#    const int MAX_SPEED = 100;
#    const double PI = 3.14159;
#    const string APP_NAME = "LLM Learner";
# ============================================================

print("-" * 50)           # separator line
print("EXERCISE 5: Constants (ALL_CAPS convention)")  # section header
print("-" * 50)           # separator line
print()                   # blank line


def define_constants():
    """
    Define constants using the ALL_CAPS naming convention.

    Returns:
        tuple: (int, float, str) -- MAX_SPEED, PI, APP_NAME
    """
    # TODO: Define the three constants and return them
    # MAX_SPEED = 100             # ALL_CAPS signals "this is a constant" to other developers
    # PI = 3.14159                # float constant -- like Math.PI in C#
    # APP_NAME = "LLM Learner"   # string constant
    # return (MAX_SPEED, PI, APP_NAME)  # return all three as a tuple
    pass   # replace with your code


result5 = define_constants()       # call the function
if result5 is not None:            # only unpack if something returned
    max_speed, pi, app_name = result5   # unpack the tuple
    print(f"  MAX_SPEED : {max_speed}  (type: {type(max_speed).__name__})")   # show each constant
    print(f"  PI        : {pi}  (type: {type(pi).__name__})")
    print(f"  APP_NAME  : {app_name}  (type: {type(app_name).__name__})")
print()
print("  Expected: 100 (int), 3.14159 (float), LLM Learner (str)")
print()

print("=" * 60)                           # closing separator
print("All exercises complete!")          # final message
print("Remove the 'pass' lines and")      # hint for the student
print("uncomment the TODO code to run!")  # another hint
print("=" * 60)                           # closing separator

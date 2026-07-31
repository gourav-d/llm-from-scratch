"""
Module 01 - Python Basics
Exercise 10: Error Handling

GLOSSARY
--------
try/except    : The Python way to catch and handle errors.
                Like C# try/catch.
                try:          -- code that might fail goes here
                    risky()
                except ValueError:  -- catches only ValueError
                    handle_it()
Exception     : The base class of all Python exceptions.
                Like C# Exception base class.
                except Exception as e:  -- catches ANY exception; e is the error object.
raise         : Throw an exception intentionally.
                raise ValueError("message")
                Like C# throw new ArgumentException("message").
ValueError    : Raised when a value is wrong/unexpected (e.g., int("abc")).
                Like C# ArgumentException or FormatException.
TypeError     : Raised when an operation is applied to the wrong type.
                Like C# InvalidCastException or ArgumentException.
KeyError      : Raised when a dict key does not exist.
                Like C# KeyNotFoundException.
finally       : A block that ALWAYS runs, whether an error occurred or not.
                Like C# finally { ... }
                Use it for cleanup (close files, release resources).
assert        : A quick sanity check. assert condition, "message"
                If condition is False, raises AssertionError.
                Like C# Debug.Assert() or Guard clauses.
"""

print("=" * 60)    # "=" repeated 60 times -- visual separator
print("Exercise 10: Error Handling")  # exercise title
print("=" * 60)    # separator
print()            # blank line


# ============================================================
#  EXERCISE 1
#  Topic: Safe type conversion with try/except
#
#  Background:
#    int("42")  -> 42     (works fine)
#    int("abc") -> ValueError: invalid literal for int()
#
#    Without error handling, your program CRASHES.
#    With try/except, you can handle it gracefully:
#
#    Python:
#      try:
#          return int(text)      # try to convert
#      except ValueError:        # if it fails with ValueError...
#          return None           # ...return None instead of crashing
#
#    In C#:
#      if (int.TryParse(text, out int result)) return result;
#      else return null;
#    Or: try { return int.Parse(text); } catch (FormatException) { return null; }
#
#  Your Task:
#    Write 'safe_int(text)':
#    - Return int(text) if conversion succeeds
#    - Return None if it raises a ValueError
#
#  C# Analogy:
#    int? SafeInt(string text) {
#        if (int.TryParse(text, out int v)) return v;
#        return null;
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 1: Safe int Conversion")  # section title
print("-" * 50)    # separator
print()            # blank line


def safe_int(text):
    """
    Try to convert text to int. Return None if conversion fails.

    Args:
        text (str): A string that may or may not contain an integer.

    Returns:
        int or None: The converted integer, or None if invalid.
    """
    # TODO: Use try/except to safely convert
    # try:                    # attempt this block
    #     return int(text)    # convert text to int -- may raise ValueError
    # except ValueError:      # if ValueError is raised...
    #     return None         # ...return None instead of crashing
    pass   # replace with your code


print(f"  safe_int('42')    -> {safe_int('42')}")     # 42 (int)
print(f"  safe_int('3.14')  -> {safe_int('3.14')}")   # None (can't convert float string to int)
print(f"  safe_int('hello') -> {safe_int('hello')}")  # None
print(f"  safe_int('-7')    -> {safe_int('-7')}")      # -7
print(f"  safe_int('')      -> {safe_int('')}")        # None
print()
print("  Expected: 42, None, None, -7, None")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Safe dict access with try/except KeyError
#
#  Background:
#    Accessing a dict key that doesn't exist raises KeyError:
#      data["missing_key"]  -> KeyError!
#
#    Two safe approaches:
#    Approach A: dict.get(key, default)        -- no try needed
#    Approach B: try/except KeyError           -- handles it after the fact
#
#    This exercise practices Approach B so you understand the pattern.
#
#    Python:
#      try:
#          return data[key]
#      except KeyError:
#          return default
#
#    In C#:
#      try { return dict[key]; } catch (KeyNotFoundException) { return default; }
#      // or: dict.TryGetValue(key, out var v) ? v : default
#
#  Your Task:
#    Write 'safe_get(data, key, default_val)':
#    - Return data[key] if key exists
#    - Return default_val if KeyError is raised
#
#  C# Analogy:
#    T SafeGet<T>(Dictionary<string,T> data, string key, T defaultVal) {
#        try { return data[key]; } catch (KeyNotFoundException) { return defaultVal; }
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 2: Safe Dict Access (KeyError)")  # section title
print("-" * 50)    # separator
print()            # blank line


def safe_get(data, key, default_val):
    """
    Safely access a dict value, returning default_val if key not found.

    Args:
        data (dict): The dictionary to look up.
        key: The key to look for.
        default_val: Value to return if key is missing.

    Returns:
        The value at data[key], or default_val if KeyError.
    """
    # TODO: Use try/except KeyError
    # try:
    #     return data[key]        # try to access the key
    # except KeyError:            # key doesn't exist -- KeyError is raised
    #     return default_val      # return the fallback value
    pass   # replace with your code


config = {"host": "localhost", "port": 8080}   # test dict -- no "timeout" key
print(f"  safe_get(config, 'host', 'unknown')    -> {safe_get(config, 'host', 'unknown')}")     # localhost
print(f"  safe_get(config, 'port', 0)            -> {safe_get(config, 'port', 0)}")             # 8080
print(f"  safe_get(config, 'timeout', 30)        -> {safe_get(config, 'timeout', 30)}")         # 30
print(f"  safe_get(config, 'debug', False)       -> {safe_get(config, 'debug', False)}")        # False
print()
print("  Expected: 'localhost', 8080, 30, False")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Raising exceptions with raise
#
#  Background:
#    You can raise exceptions yourself to signal bad input.
#    Use 'raise' to throw an exception:
#      raise ValueError("Age cannot be negative")
#
#    In C#:  throw new ArgumentException("Age cannot be negative");
#
#    Convention: raise ValueError for bad values, TypeError for wrong types.
#
#    The caller then catches it with try/except:
#      try:
#          set_age(-5)
#      except ValueError as e:
#          print(f"Error: {e}")
#
#  Your Task:
#    Write 'set_age(age)':
#    - If age < 0, raise ValueError("Age cannot be negative")
#    - If age > 150, raise ValueError("Age is unrealistically large")
#    - Otherwise, return the age
#
#  C# Analogy:
#    int SetAge(int age) {
#        if (age < 0)   throw new ArgumentException("Age cannot be negative");
#        if (age > 150) throw new ArgumentException("Age is unrealistically large");
#        return age;
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 3: Raising ValueError for Bad Input")  # section title
print("-" * 50)    # separator
print()            # blank line


def set_age(age):
    """
    Validate and return age, raising ValueError for invalid input.

    Args:
        age (int): The age to validate.

    Returns:
        int: The validated age.

    Raises:
        ValueError: If age is negative or unrealistically large.
    """
    # TODO: Add validation with raise
    # if age < 0:                                        # negative age is invalid
    #     raise ValueError("Age cannot be negative")    # raise an error with a message
    # if age > 150:                                      # impossibly old
    #     raise ValueError("Age is unrealistically large")
    # return age                                         # valid age -- return it
    pass   # replace with your code


test_ages = [25, -1, 200, 0, 100]    # mix of valid and invalid ages
for age in test_ages:                 # test each age
    try:                              # try the set_age call
        result = set_age(age)         # call our function
        print(f"  set_age({age:4}) -> OK: {result}")        # success
    except ValueError as e:           # catch the ValueError
        print(f"  set_age({age:4}) -> Error: {e}")          # show the error message
print()
print("  Expected: OK for 25,0,100 -- ValueError for -1 and 200")
print()


# ============================================================
#  EXERCISE 4
#  Topic: finally block -- always runs
#
#  Background:
#    The 'finally' block ALWAYS runs -- whether an exception occurred or not.
#    Use it for cleanup: closing files, releasing connections, etc.
#
#    Pattern:
#      try:
#          risky_operation()
#      except SomeError as e:
#          handle_error(e)
#      finally:
#          cleanup()    # ALWAYS runs (even if exception not caught!)
#
#    In C#:
#      try { RiskyOp(); }
#      catch (Exception e) { HandleError(e); }
#      finally { Cleanup(); }
#
#  Your Task:
#    Write 'divide_with_cleanup(a, b)':
#    - Try to compute a / b
#    - If ZeroDivisionError (division by zero), store result = None
#    - In finally, ALWAYS append "cleanup done" to a log list
#    - Return (result, log) where log contains "cleanup done"
#
#  C# Analogy:
#    var log = new List<string>();
#    double? result = null;
#    try { result = a / b; }
#    catch (DivideByZeroException) { result = null; }
#    finally { log.Add("cleanup done"); }
#    return (result, log);
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 4: finally Block (Always Runs)")  # section title
print("-" * 50)    # separator
print()            # blank line


def divide_with_cleanup(a, b):
    """
    Divide a by b, always running cleanup in finally.

    Args:
        a (float): Numerator.
        b (float): Denominator.

    Returns:
        tuple: (result, log) where result is float or None, log contains cleanup message.
    """
    # TODO: Use try/except/finally
    # log = []               # list to track what happened
    # result = None          # default result
    # try:
    #     result = a / b         # attempt division -- may raise ZeroDivisionError
    # except ZeroDivisionError:  # catch division by zero
    #     result = None          # set result to None when division fails
    # finally:
    #     log.append("cleanup done")  # THIS ALWAYS RUNS -- even after an exception
    # return (result, log)
    pass   # replace with your code


tests = [(10, 2), (5, 0), (9, 3)]    # (10/2=5), (5/0=error), (9/3=3)
for a, b in tests:                    # test each pair
    res, log = divide_with_cleanup(a, b) if divide_with_cleanup(a, b) is not None else (None, [])
    if res is not None or (isinstance(res, type(None)) and log):  # show if we got a result
        print(f"  {a} / {b} = {res}   |  log: {log}")
print()
print("  Expected:")
print("    10/2 = 5.0    | log: ['cleanup done']")
print("    5/0  = None   | log: ['cleanup done']")
print("    9/3  = 3.0    | log: ['cleanup done']")
print()


# ============================================================
#  EXERCISE 5
#  Topic: Multiple except clauses -- handle different errors differently
#
#  Background:
#    You can catch multiple exception types separately:
#      try:
#          ...
#      except ValueError as e:
#          print(f"Value error: {e}")
#      except TypeError as e:
#          print(f"Type error: {e}")
#      except Exception as e:
#          print(f"Unexpected: {e}")   # catch-all for anything else
#
#    In C#:
#      catch (FormatException e) { ... }
#      catch (InvalidCastException e) { ... }
#      catch (Exception e) { ... }
#
#    Order matters: put more specific exceptions BEFORE generic ones.
#
#  Your Task:
#    Write 'risky_operation(value)':
#    - If value is a string: try int(value); return ("ValueError", None) if it fails
#    - If value is a list:   try value[10];  return ("IndexError", None) if it fails
#    - Otherwise:            try value + " "; return ("TypeError", None) if it fails
#    - On success:           return ("ok", result)
#
#  C# Analogy:
#    try { ... }
#    catch (FormatException)  { return ("ValueError", null); }
#    catch (IndexOutOfRangeException) { return ("IndexError", null); }
#    catch (InvalidOperationException) { return ("TypeError", null); }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 5: Multiple except Clauses")  # section title
print("-" * 50)    # separator
print()            # blank line


def risky_operation(value):
    """
    Perform different risky operations and catch different error types.

    Args:
        value: A string, list, or other value to test.

    Returns:
        tuple: (error_type_str, result) where error_type_str is "ok" on success.
    """
    # TODO: Use multiple except clauses for different error types
    # try:
    #     if isinstance(value, str):    # if it's a string, try to parse as int
    #         result = int(value)       # int("abc") raises ValueError
    #     elif isinstance(value, list): # if it's a list, try to access index 10
    #         result = value[10]        # short list raises IndexError
    #     else:                         # otherwise, try to add a string to it
    #         result = value + " "     # int + " " raises TypeError
    #     return ("ok", result)        # if we get here, no error occurred
    # except ValueError as e:          # wrong value format (e.g., "abc" -> int)
    #     return ("ValueError", None)
    # except IndexError as e:          # list index out of range
    #     return ("IndexError", None)
    # except TypeError as e:           # wrong type for operation
    #     return ("TypeError", None)
    pass   # replace with your code


test_values = ["42", "abc", [1, 2, 3], 99]   # mix of inputs
for val in test_values:                        # test each input
    res = risky_operation(val)                 # call the function
    if res is not None:                        # only print if something returned
        err_type, result = res                 # unpack the tuple
        print(f"  input={repr(val):15} -> ({err_type}, {result})")   # show result
print()
print("  Expected:")
print("    '42'       -> ('ok', 42)")
print("    'abc'      -> ('ValueError', None)")
print("    [1,2,3]    -> ('IndexError', None)")
print("    99         -> ('TypeError', None)")
print()

print("=" * 60)                       # closing separator
print("All exercises complete!")      # completion message
print("Congratulations on finishing Module 01!")   # encouragement
print("=" * 60)                       # closing separator

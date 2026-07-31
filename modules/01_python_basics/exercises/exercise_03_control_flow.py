"""
Module 01 - Python Basics
Exercise 03: Control Flow

GLOSSARY
--------
if/elif/else  : Conditional branching. 'elif' is short for 'else if'.
                Python uses indentation (4 spaces) instead of curly braces {} like C#.
indentation   : Python uses spaces/tabs to define code blocks. This is MANDATORY.
                In C# you use {} braces; in Python you use consistent spacing.
                Wrong indentation = IndentationError (a real error!).
while loop    : Repeats a block while a condition is True. Same as C# while(){}.
for loop      : Iterates over a sequence (list, range, string). Like C# foreach.
range()       : Generates a sequence of numbers. range(5) -> 0,1,2,3,4.
                range(start, stop) -> from start up to (not including) stop.
                range(start, stop, step) -> with step size.
break         : Immediately exits the loop. Same as C# break.
continue      : Skips the rest of this iteration, goes to the next. Same as C# continue.
pass          : Does nothing. A placeholder when Python requires a statement.
                No C# equivalent -- Python needs something even in empty blocks.
"""

print("=" * 60)    # "=" repeated 60 times -- visual separator
print("Exercise 03: Control Flow")  # exercise title
print("=" * 60)    # separator
print()            # blank line


# ============================================================
#  EXERCISE 1
#  Topic: if / elif / else
#
#  Background:
#    In C# you write:
#      if (x < 0) { return "negative"; }
#      else if (x == 0) { return "zero"; }
#      else { return "positive"; }
#
#    In Python:
#      if x < 0:           # colon, no parentheses needed, no braces!
#          return "negative"   # 4-space indent -- this IS the block
#      elif x == 0:        # elif = else if
#          return "zero"
#      else:
#          return "positive"
#
#    Key differences from C#:
#      - No parentheses around the condition
#      - Colon (:) at the end of each condition line
#      - Indentation (4 spaces) replaces curly braces {}
#      - 'elif' instead of 'else if'
#
#  Your Task:
#    Given a number 'n', return:
#      "negative" if n < 0
#      "zero"     if n == 0
#      "positive" if n > 0
#
#  C# Analogy:
#    string Classify(int n) {
#        if (n < 0) return "negative";
#        else if (n == 0) return "zero";
#        else return "positive";
#    }
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 1: if / elif / else")  # section title
print("-" * 50)    # separator
print()            # blank line


def classify_number(n):
    """
    Classify a number as negative, zero, or positive.

    Args:
        n (int or float): The number to classify.

    Returns:
        str: "negative", "zero", or "positive"
    """
    # TODO: Add if/elif/else to classify n
    # if n < 0:           # check if n is less than zero
    #     return "negative"   # return the label
    # elif n == 0:        # check if n is exactly zero
    #     return "zero"
    # else:               # must be positive (only option left)
    #     return "positive"
    pass   # replace with your code


print(f"  classify_number(-5)  -> {classify_number(-5)}")   # negative
print(f"  classify_number(0)   -> {classify_number(0)}")    # zero
print(f"  classify_number(7)   -> {classify_number(7)}")    # positive
print()
print("  Expected: negative, zero, positive")
print()


# ============================================================
#  EXERCISE 2
#  Topic: while loop
#
#  Background:
#    A while loop repeats a block as long as the condition is True.
#    In C#:
#      var results = new List<int>();
#      int i = 1;
#      while (i <= n) { results.Add(i); i++; }
#
#    In Python:
#      results = []        # empty list
#      i = 1               # counter variable
#      while i <= n:       # keep going while i <= n
#          results.append(i)   # add i to the list
#          i += 1          # increment (Python has no i++ operator!)
#
#    IMPORTANT: Python has no ++ operator! Use i += 1 instead.
#
#  Your Task:
#    Count from 1 to n (inclusive) using a while loop.
#    Collect each number in a list and return the list.
#    For n=5: return [1, 2, 3, 4, 5]
#
#  C# Analogy:
#    var list = new List<int>();
#    int i = 1;
#    while (i <= n) { list.Add(i); i++; }
#    return list;
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 2: while loop")  # section title
print("-" * 50)    # separator
print()            # blank line


def count_with_while(n):
    """
    Count from 1 to n using a while loop.

    Args:
        n (int): Upper limit (inclusive).

    Returns:
        list: Numbers from 1 to n.
    """
    # TODO: Use a while loop to build and return the list
    # results = []          # create an empty list (like new List<int>() in C#)
    # i = 1                 # start counter at 1
    # while i <= n:         # keep looping while i is <= n
    #     results.append(i) # add current i to the list (.append is like .Add() in C#)
    #     i += 1            # increment i (Python has no i++, use i += 1)
    # return results        # return the completed list
    pass   # replace with your code


result2 = count_with_while(5)    # call with n=5
if result2 is not None:          # only print if something returned
    print(f"  count_with_while(5) -> {result2}")   # [1, 2, 3, 4, 5]
print()
print("  Expected: [1, 2, 3, 4, 5]")
print()


# ============================================================
#  EXERCISE 3
#  Topic: for loop with range()
#
#  Background:
#    Python's for loop iterates over a sequence directly.
#    range() generates a sequence of numbers:
#      range(5)       -> 0, 1, 2, 3, 4         (starts at 0!)
#      range(1, 6)    -> 1, 2, 3, 4, 5         (start inclusive, end exclusive)
#      range(0, 10, 2)-> 0, 2, 4, 6, 8         (step by 2)
#
#    In C#:  for (int i = 1; i <= n; i++) { ... }
#    Python: for i in range(1, n+1):  # range(1, n+1) gives 1 through n
#                ...
#
#  Your Task:
#    Use a for loop with range() to compute the sum of numbers
#    from 1 to n (inclusive). Return the sum.
#    For n=5: 1+2+3+4+5 = 15
#
#  C# Analogy:
#    int total = 0;
#    for (int i = 1; i <= n; i++) { total += i; }
#    return total;
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 3: for loop with range()")  # section title
print("-" * 50)    # separator
print()            # blank line


def sum_with_for(n):
    """
    Sum numbers from 1 to n using a for loop.

    Args:
        n (int): Upper limit (inclusive).

    Returns:
        int: Sum of 1 + 2 + ... + n
    """
    # TODO: Use a for loop and range() to sum numbers 1 through n
    # total = 0                  # start the running total at zero
    # for i in range(1, n + 1): # range(1, n+1) gives 1, 2, 3, ..., n
    #     total += i             # add i to running total (total = total + i)
    # return total               # return the final sum
    pass   # replace with your code


result3 = sum_with_for(5)       # call with n=5
if result3 is not None:         # only print if something returned
    print(f"  sum_with_for(5)  -> {result3}")   # 15
    print(f"  sum_with_for(10) -> {sum_with_for(10)}")  # 55
print()
print("  Expected: 15, 55")
print()


# ============================================================
#  EXERCISE 4
#  Topic: break and continue
#
#  Background:
#    break    : Exit the loop immediately. Same as C# break.
#    continue : Skip to the next iteration. Same as C# continue.
#
#    Example -- find first even number using break:
#      for n in [1, 3, 5, 4, 7]:
#          if n % 2 == 0:    # n % 2 == 0 means n is even (no remainder)
#              found = n
#              break         # stop looping as soon as we find one
#
#    Example -- collect only even numbers using continue:
#      evens = []
#      for n in [1, 2, 3, 4, 5]:
#          if n % 2 != 0:    # if odd...
#              continue      # ...skip this iteration
#          evens.append(n)   # only even numbers reach here
#
#  Your Task:
#    Given a list of numbers, return a tuple:
#    (first_even, list_of_evens)
#    Use break to find the first even, continue to collect all evens.
#
#  C# Analogy:
#    int firstEven = -1;
#    foreach(int n in numbers) { if(n%2==0){firstEven=n; break;} }
#    var evens = numbers.Where(n => n%2==0).ToList();
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 4: break and continue")  # section title
print("-" * 50)    # separator
print()            # blank line


def find_evens(numbers):
    """
    Find the first even number (using break) and all even numbers (using continue).

    Args:
        numbers (list): List of integers.

    Returns:
        tuple: (first_even, all_evens) where first_even is int or None.
    """
    # TODO: Use break to find the first even, continue to collect all evens
    # --- Part 1: find first even using break ---
    # first_even = None           # default if no even found
    # for n in numbers:           # iterate through each number
    #     if n % 2 == 0:          # n % 2 == 0 means n is even
    #         first_even = n      # save it
    #         break               # stop immediately -- we found the first one
    #
    # --- Part 2: collect all evens using continue ---
    # all_evens = []              # empty list to collect evens
    # for n in numbers:           # iterate through each number again
    #     if n % 2 != 0:          # if n is ODD (remainder is 1)
    #         continue            # skip to next iteration (ignore odd numbers)
    #     all_evens.append(n)     # only even numbers reach this line
    #
    # return (first_even, all_evens)
    pass   # replace with your code


nums = [1, 3, 5, 4, 7, 6, 9]       # test list: odd, odd, odd, EVEN, odd, EVEN, odd
result4 = find_evens(nums)           # call the function
if result4 is not None:              # only print if something returned
    first, all_e = result4           # unpack the tuple
    print(f"  First even : {first}")      # should be 4
    print(f"  All evens  : {all_e}")      # should be [4, 6]
print()
print("  Expected: first_even=4, all_evens=[4, 6]")
print()


# ============================================================
#  EXERCISE 5
#  Topic: Nested if -- grade classifier
#
#  Background:
#    You can put if statements inside other if statements (nested if).
#    This is the same as C# nested if blocks.
#    Python uses indentation to show nesting:
#      if score >= 90:
#          grade = "A"
#      elif score >= 80:  # implicitly: score < 90 AND score >= 80
#          grade = "B"
#      ...
#
#    Note: elif is evaluated TOP TO BOTTOM.
#    Once one condition is True, the rest are SKIPPED.
#
#  Your Task:
#    Given a score (0-100), return the letter grade:
#      90-100  -> "A"
#      80-89   -> "B"
#      70-79   -> "C"
#      60-69   -> "D"
#      below 60 -> "F"
#
#  C# Analogy:
#    if (score >= 90) return "A";
#    else if (score >= 80) return "B";
#    else if (score >= 70) return "C";
#    else if (score >= 60) return "D";
#    else return "F";
# ============================================================

print("-" * 50)    # section separator
print("EXERCISE 5: Grade Classifier (if/elif/else chain)")  # section title
print("-" * 50)    # separator
print()            # blank line


def get_grade(score):
    """
    Convert a numeric score to a letter grade.

    Args:
        score (int): Score between 0 and 100.

    Returns:
        str: Letter grade "A", "B", "C", "D", or "F".
    """
    # TODO: Add if/elif/else chain to return the correct grade
    # if score >= 90:      # 90 and above is an A
    #     return "A"
    # elif score >= 80:    # 80-89 is a B (we already know it's < 90)
    #     return "B"
    # elif score >= 70:    # 70-79 is a C
    #     return "C"
    # elif score >= 60:    # 60-69 is a D
    #     return "D"
    # else:                # anything below 60 is F
    #     return "F"
    pass   # replace with your code


test_scores = [95, 85, 75, 65, 50]   # list of scores to test
for score in test_scores:             # iterate through each score
    grade = get_grade(score)          # get the grade for this score
    print(f"  Score {score} -> Grade {grade}")   # show result
print()
print("  Expected: A, B, C, D, F")
print()

print("=" * 60)                       # closing separator
print("All exercises complete!")      # completion message
print("=" * 60)                       # closing separator

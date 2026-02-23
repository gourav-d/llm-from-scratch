# Module 1 Quiz: Python Basics

Complete this quiz to test your understanding of Python basics.

---

## Section 1: Variables and Types (10 questions)

### Q1. What is the correct way to create a variable in Python?
A) `int x = 10;`
B) `var x = 10`
C) `x = 10`
D) `let x = 10`

<details>
<summary>Answer</summary>

**C) `x = 10`**

Python uses dynamic typing - no type declaration needed!
</details>

---

### Q2. How do you write boolean values in Python?
A) `true` and `false`
B) `True` and `False`
C) `TRUE` and `FALSE`
D) `1` and `0`

<details>
<summary>Answer</summary>

**B) `True` and `False`**

Python booleans are capitalized (unlike C#'s lowercase true/false)
</details>

---

### Q3. What is the output of `print(type(3.14))`?
A) `int`
B) `float`
C) `<class 'float'>`
D) `double`

<details>
<summary>Answer</summary>

**C) `<class 'float'>`**

The `type()` function returns the full type representation.
</details>

---

### Q4. What is Python's equivalent of C#'s `null`?
A) `null`
B) `None`
C) `nil`
D) `undefined`

<details>
<summary>Answer</summary>

**B) `None`**

Python uses `None` (capitalized) instead of null.
</details>

---

### Q5. Which naming convention does Python use?
A) camelCase
B) PascalCase
C) snake_case
D) kebab-case

<details>
<summary>Answer</summary>

**C) snake_case**

Python uses snake_case for variables and functions: `user_name`, not `userName`
</details>

---

### Q6. How do you convert the string "42" to an integer?
A) `(int)"42"`
B) `int("42")`
C) `"42".toInt()`
D) `Integer.parse("42")`

<details>
<summary>Answer</summary>

**B) `int("42")`**

Use the `int()` function for type conversion.
</details>

---

### Q7. What's the Python equivalent of C#'s `$"Hello {name}"`?
A) `"Hello " + name`
B) `f"Hello {name}"`
C) `"Hello ${name}"`
D) `String.Format("Hello {0}", name)`

<details>
<summary>Answer</summary>

**B) `f"Hello {name}"`**

f-strings are Python's string interpolation (the `f` prefix is required).
</details>

---

### Q8. What is the result of `int(3.99)`?
A) `4` (rounds up)
B) `3` (truncates)
C) `3.99` (no change)
D) Error

<details>
<summary>Answer</summary>

**B) `3` (truncates)**

`int()` truncates (removes) the decimal, it doesn't round!
</details>

---

### Q9. Which is the correct way to check if a value is None?
A) `if x == None:`
B) `if x is None:`
C) `if x.equals(None):`
D) Both A and B

<details>
<summary>Answer</summary>

**B) `if x is None:`**

Use `is None`, not `== None` (Pythonic way).
</details>

---

### Q10. What does `len("Hello")` return?
A) `4`
B) `5`
C) `6`
D) Error

<details>
<summary>Answer</summary>

**B) `5`**

`len()` returns the length of the string (5 characters).
</details>

---

## Section 2: Operators (10 questions)

### Q11. What is the result of `10 / 3` in Python?
A) `3`
B) `3.0`
C) `3.333...`
D) Error

<details>
<summary>Answer</summary>

**C) `3.333...`**

The `/` operator always returns a float, even for whole numbers!
</details>

---

### Q12. What is the result of `10 // 3`?
A) `3`
B) `3.0`
C) `3.333...`
D) `4`

<details>
<summary>Answer</summary>

**A) `3`**

`//` is integer division (floor division) - returns integer part only.
</details>

---

### Q13. What is `2 ** 3`?
A) `6` (2 × 3)
B) `8` (2³)
C) `5` (2 + 3)
D) Error

<details>
<summary>Answer</summary>

**B) `8` (2³)**

`**` is the power/exponent operator: 2³ = 8
</details>

---

### Q14. What is the result of `"ha" * 3`?
A) Error
B) `"hahaha"`
C) `"ha3"`
D) `9`

<details>
<summary>Answer</summary>

**B) `"hahaha"`**

String multiplication repeats the string!
</details>

---

### Q15. What is Python's AND operator?
A) `&&`
B) `and`
C) `&`
D) `AND`

<details>
<summary>Answer</summary>

**B) `and`**

Python uses words for logical operators: `and`, `or`, `not` (not symbols like &&, ||, !)
</details>

---

### Q16. What does `"apple" in ["apple", "banana"]` return?
A) `True`
B) `False`
C) `1`
D) Error

<details>
<summary>Answer</summary>

**A) `True`**

The `in` operator checks if item exists in the collection.
</details>

---

### Q17. Which is valid in Python?
A) `x++`
B) `x--`
C) `x += 1`
D) All of the above

<details>
<summary>Answer</summary>

**C) `x += 1`**

Python doesn't have `++` or `--` operators!
</details>

---

### Q18. What's the difference between `==` and `is`?
A) No difference
B) `==` compares values, `is` checks if same object
C) `is` is faster
D) `==` is deprecated

<details>
<summary>Answer</summary>

**B) `==` compares values, `is` checks if same object**

`==` checks equality, `is` checks identity (same object in memory).
</details>

---

### Q19. What is the result of `10 % 3`?
A) `3`
B) `1`
C) `3.333...`
D) `0`

<details>
<summary>Answer</summary>

**B) `1`**

`%` is modulus (remainder): 10 ÷ 3 = 3 remainder 1
</details>

---

### Q20. What is the result of `not True`?
A) `True`
B) `False`
C) `1`
D) `0`

<details>
<summary>Answer</summary>

**B) `False`**

`not` reverses the boolean value.
</details>

---

## Section 3: Control Flow (10 questions)

### Q21. What's wrong with this code?
```python
if x > 5
    print("Greater")
```
A) Missing parentheses
B) Missing colon
C) Missing semicolon
D) Nothing wrong

<details>
<summary>Answer</summary>

**B) Missing colon**

Should be: `if x > 5:`
</details>

---

### Q22. What is Python's equivalent of `else if`?
A) `elseif`
B) `else if`
C) `elif`
D) `elsif`

<details>
<summary>Answer</summary>

**C) `elif`**

Python uses `elif` as shorthand for "else if".
</details>

---

### Q23. How do you loop from 0 to 4?
A) `for i in range(4):`
B) `for i in range(5):`
C) `for i in range(0, 4):`
D) Both B

<details>
<summary>Answer</summary>

**B) `for i in range(5):`**

`range(5)` creates [0, 1, 2, 3, 4] - stop value is exclusive!
</details>

---

### Q24. What does this print?
```python
for i in range(3):
    if i == 1:
        continue
    print(i)
```
A) `0 1 2`
B) `0 2`
C) `1`
D) `0 1`

<details>
<summary>Answer</summary>

**B) `0 2`**

`continue` skips the rest of iteration, so 1 is not printed.
</details>

---

### Q25. What does this print?
```python
for i in range(5):
    if i == 3:
        break
    print(i)
```
A) `0 1 2 3 4`
B) `0 1 2`
C) `0 1 2 3`
D) `3 4`

<details>
<summary>Answer</summary>

**B) `0 1 2`**

`break` exits the loop when i equals 3 (before printing 3).
</details>

---

### Q26. How do you loop through a list with both index and value?
A) `for i, val in list:`
B) `for i, val in enumerate(list):`
C) `for i in range(len(list)):`
D) `for val in list.indexed():`

<details>
<summary>Answer</summary>

**B) `for i, val in enumerate(list):`**

`enumerate()` provides both index and value.
</details>

---

### Q27. What defines a code block in Python?
A) Curly braces `{}`
B) Parentheses `()`
C) Indentation
D) Semicolons

<details>
<summary>Answer</summary>

**C) Indentation**

Python uses indentation (4 spaces) to define code blocks, not braces!
</details>

---

### Q28. What's the ternary operator equivalent of this?
```python
if x > 5:
    result = "Big"
else:
    result = "Small"
```
A) `result = "Big" ? x > 5 : "Small"`
B) `result = "Big" if x > 5 else "Small"`
C) `result = x > 5 ? "Big" : "Small"`
D) `result = if x > 5 then "Big" else "Small"`

<details>
<summary>Answer</summary>

**B) `result = "Big" if x > 5 else "Small"`**

Python's ternary: `value_if_true if condition else value_if_false`
</details>

---

### Q29. What does `range(2, 8, 2)` produce?
A) `[2, 3, 4, 5, 6, 7]`
B) `[2, 4, 6]`
C) `[2, 4, 6, 8]`
D) `[0, 2, 4, 6]`

<details>
<summary>Answer</summary>

**B) `[2, 4, 6]`**

`range(start, stop, step)` → starts at 2, steps by 2, stops before 8.
</details>

---

### Q30. What is special about using `else` with a `for` loop?
A) It runs if loop has an error
B) It runs if loop completes without `break`
C) It runs if loop is empty
D) It's not valid in Python

<details>
<summary>Answer</summary>

**B) It runs if loop completes without `break`**

Python's unique feature - else runs only if loop wasn't broken out of.
</details>

---

## Scoring

- **25-30 correct**: Excellent! You've mastered Python basics 🌟
- **20-24 correct**: Very good! Review missed topics 👍
- **15-19 correct**: Good start! Re-read lessons and practice more 📚
- **Below 15**: Review all lessons and do more practice exercises 💪

---

## What to Do Next

1. **Review wrong answers** - Understand why you got them wrong
2. **Re-read relevant lessons** - Focus on weak areas
3. **Do the lab exercise** - Hands-on practice is crucial
4. **Try the code examples** - Experiment and modify them
5. **Move to next module** - When you score 25+

Keep learning! 🚀

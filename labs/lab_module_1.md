# Lab 1: Python Basics - Hands-On Exercises

Complete these exercises to practice what you've learned. Solutions are provided at the end.

---

## Exercise 1: Temperature Converter

**Task:** Create a program that converts Celsius to Fahrenheit.

**Formula:** F = (C × 9/5) + 32

**Requirements:**
1. Ask user for temperature in Celsius
2. Convert to Fahrenheit
3. Display both temperatures
4. Use f-strings for output

**Example Output:**
```
Enter temperature in Celsius: 25
25.0°C = 77.0°F
```

**Your Code:**
```python
# Write your code here




```

---

## Exercise 2: Grade Calculator

**Task:** Create a grade calculator based on score.

**Grading Scale:**
- A: 90-100
- B: 80-89
- C: 70-79
- D: 60-69
- F: Below 60

**Requirements:**
1. Ask user for score (0-100)
2. Use if/elif/else to determine grade
3. Display the grade with a message

**Example Output:**
```
Enter your score: 85
Your grade is: B
Great job!
```

**Your Code:**
```python
# Write your code here




```

---

## Exercise 3: Number Analyzer

**Task:** Analyze a list of numbers and provide statistics.

**Given:**
```python
numbers = [12, 45, 67, 23, 89, 34, 56, 78, 90, 11]
```

**Calculate:**
1. Sum of all numbers
2. Average (sum ÷ count)
3. Maximum number
4. Minimum number
5. Count of even numbers
6. Count of odd numbers

**Example Output:**
```
Numbers: [12, 45, 67, 23, 89, 34, 56, 78, 90, 11]
Sum: 505
Average: 50.5
Maximum: 90
Minimum: 11
Even count: 5
Odd count: 5
```

**Your Code:**
```python
# Write your code here




```

---

## Exercise 4: Multiplication Table

**Task:** Create a multiplication table for a given number.

**Requirements:**
1. Ask user for a number
2. Display multiplication table from 1 to 10
3. Format output nicely

**Example Output:**
```
Enter a number: 5
Multiplication table for 5:
5 x 1 = 5
5 x 2 = 10
5 x 3 = 15
...
5 x 10 = 50
```

**Your Code:**
```python
# Write your code here




```

---

## Exercise 5: Password Validator

**Task:** Create a password strength checker.

**Password must:**
1. Be at least 8 characters long
2. Contain at least one digit
3. Contain at least one uppercase letter
4. Contain at least one lowercase letter

**Requirements:**
1. Ask user for password
2. Check all criteria
3. Display which criteria are met/not met
4. Display if password is valid

**Example Output:**
```
Enter password: Hello123
✓ Length >= 8 characters
✓ Contains digit
✓ Contains uppercase
✓ Contains lowercase
Password is VALID!
```

**Hints:**
- `len(password)` → length
- `password.isdigit()` → has digit (use any())
- `password.isupper()` → has uppercase
- `password.islower()` → has lowercase

**Your Code:**
```python
# Write your code here




```

---

## Exercise 6: Shopping Cart

**Task:** Create a simple shopping cart system.

**Requirements:**
1. Create a menu system (add item, view cart, checkout, quit)
2. Store items and prices in a dictionary
3. Allow user to add items to cart
4. Calculate total
5. Display cart contents

**Available Items:**
```python
items = {
    "apple": 0.99,
    "banana": 0.59,
    "orange": 0.79,
    "milk": 3.49,
    "bread": 2.99
}
```

**Example Output:**
```
=== Shopping Cart ===
1. Add item
2. View cart
3. Checkout
4. Quit

Choose option: 1
Available items: apple, banana, orange, milk, bread
Enter item: apple
Apple added to cart!

Choose option: 2
Your cart:
- apple: $0.99
Total: $0.99
```

**Your Code:**
```python
# Write your code here




```

---

## Exercise 7: FizzBuzz

**Task:** Classic FizzBuzz problem.

**Rules:**
- Print numbers 1 to 100
- If divisible by 3, print "Fizz"
- If divisible by 5, print "Buzz"
- If divisible by both, print "FizzBuzz"
- Otherwise, print the number

**Example Output:**
```
1
2
Fizz
4
Buzz
Fizz
7
8
Fizz
Buzz
11
Fizz
13
14
FizzBuzz
...
```

**Your Code:**
```python
# Write your code here




```

---

# Solutions

## Solution 1: Temperature Converter

```python
# Get input from user
celsius = float(input("Enter temperature in Celsius: "))

# Convert to Fahrenheit
# F = (C × 9/5) + 32
fahrenheit = (celsius * 9/5) + 32

# Display result
print(f"{celsius}°C = {fahrenheit}°F")
```

**Explanation:**
- `float(input(...))` → Gets user input and converts to float
- `(celsius * 9/5) + 32` → Conversion formula
- f-string displays both values nicely

---

## Solution 2: Grade Calculator

```python
# Get score from user
score = int(input("Enter your score: "))

# Determine grade
if score >= 90:
    grade = "A"
    message = "Excellent!"
elif score >= 80:
    grade = "B"
    message = "Great job!"
elif score >= 70:
    grade = "C"
    message = "Good work!"
elif score >= 60:
    grade = "D"
    message = "You passed."
else:
    grade = "F"
    message = "Study harder next time."

# Display result
print(f"Your grade is: {grade}")
print(message)
```

**Explanation:**
- `int(input(...))` → Gets integer input
- `if/elif/else` → Checks score ranges from highest to lowest
- Custom message for each grade

---

## Solution 3: Number Analyzer

```python
# Given list
numbers = [12, 45, 67, 23, 89, 34, 56, 78, 90, 11]

# Calculate statistics
total = sum(numbers)              # Sum using built-in function
average = total / len(numbers)    # Average
maximum = max(numbers)            # Max using built-in function
minimum = min(numbers)            # Min using built-in function

# Count even and odd
even_count = 0
odd_count = 0
for num in numbers:
    if num % 2 == 0:
        even_count += 1
    else:
        odd_count += 1

# Display results
print(f"Numbers: {numbers}")
print(f"Sum: {total}")
print(f"Average: {average}")
print(f"Maximum: {maximum}")
print(f"Minimum: {minimum}")
print(f"Even count: {even_count}")
print(f"Odd count: {odd_count}")
```

**Explanation:**
- `sum()`, `max()`, `min()` → Built-in Python functions
- Loop through list to count even/odd
- `num % 2 == 0` → Check if even (remainder is 0)

---

## Solution 4: Multiplication Table

```python
# Get number from user
num = int(input("Enter a number: "))

# Display header
print(f"Multiplication table for {num}:")

# Loop from 1 to 10
for i in range(1, 11):
    result = num * i
    print(f"{num} x {i} = {result}")
```

**Explanation:**
- `range(1, 11)` → Numbers 1 to 10 (11 is exclusive)
- `num * i` → Calculate product
- f-string formats output nicely

---

## Solution 5: Password Validator

```python
# Get password from user
password = input("Enter password: ")

# Check criteria
has_length = len(password) >= 8
has_digit = any(char.isdigit() for char in password)
has_upper = any(char.isupper() for char in password)
has_lower = any(char.islower() for char in password)

# Display results
print("✓ Length >= 8 characters" if has_length else "✗ Length >= 8 characters")
print("✓ Contains digit" if has_digit else "✗ Contains digit")
print("✓ Contains uppercase" if has_upper else "✗ Contains uppercase")
print("✓ Contains lowercase" if has_lower else "✗ Contains lowercase")

# Check if all criteria met
if has_length and has_digit and has_upper and has_lower:
    print("Password is VALID!")
else:
    print("Password is INVALID!")
```

**Explanation:**
- `len(password) >= 8` → Check length
- `any(char.isdigit() for char in password)` → Checks if any character is a digit
- Ternary operator for checkmark/cross
- `and` combines all conditions

---

## Solution 6: Shopping Cart

```python
# Available items
items = {
    "apple": 0.99,
    "banana": 0.59,
    "orange": 0.79,
    "milk": 3.49,
    "bread": 2.99
}

# Shopping cart (empty initially)
cart = []

# Menu loop
while True:
    print("\n=== Shopping Cart ===")
    print("1. Add item")
    print("2. View cart")
    print("3. Checkout")
    print("4. Quit")

    choice = input("\nChoose option: ")

    if choice == "1":
        # Add item
        print(f"Available items: {', '.join(items.keys())}")
        item = input("Enter item: ").lower()

        if item in items:
            cart.append(item)
            print(f"{item.title()} added to cart!")
        else:
            print("Item not found!")

    elif choice == "2":
        # View cart
        if cart:
            print("\nYour cart:")
            total = 0
            for item in cart:
                price = items[item]
                print(f"- {item}: ${price:.2f}")
                total += price
            print(f"Total: ${total:.2f}")
        else:
            print("Cart is empty!")

    elif choice == "3":
        # Checkout
        if cart:
            total = sum(items[item] for item in cart)
            print(f"\nTotal: ${total:.2f}")
            print("Thank you for shopping!")
            break
        else:
            print("Cart is empty!")

    elif choice == "4":
        # Quit
        print("Goodbye!")
        break

    else:
        print("Invalid option!")
```

**Explanation:**
- `while True:` → Infinite loop for menu
- `if choice == "1":` → Handle each menu option
- `cart.append(item)` → Add to cart
- `sum(items[item] for item in cart)` → Calculate total
- `break` → Exit loop

---

## Solution 7: FizzBuzz

```python
# Loop from 1 to 100
for num in range(1, 101):
    # Check divisibility
    if num % 3 == 0 and num % 5 == 0:
        print("FizzBuzz")
    elif num % 3 == 0:
        print("Fizz")
    elif num % 5 == 0:
        print("Buzz")
    else:
        print(num)
```

**Explanation:**
- Check divisibility by both 3 and 5 first (FizzBuzz)
- Then check divisibility by 3 (Fizz)
- Then check divisibility by 5 (Buzz)
- Otherwise print the number
- Order matters! Check both first, then individual

**Alternative solution (more concise):**
```python
for num in range(1, 101):
    output = ""
    if num % 3 == 0:
        output += "Fizz"
    if num % 5 == 0:
        output += "Buzz"
    print(output if output else num)
```

---

## 🎯 Next Steps

1. **Try all exercises** without looking at solutions first
2. **Compare your solutions** with the provided ones
3. **Experiment** - modify the programs
4. **Create your own** - try building similar programs
5. **Move to Module 2** when comfortable

Great job completing Module 1! 🚀

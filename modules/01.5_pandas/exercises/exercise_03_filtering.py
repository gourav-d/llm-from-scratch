"""
=============================================================================
EXERCISE 03: Filtering, Sorting, and Grouping
=============================================================================

WHAT YOU PRACTICE
------------------
- Filter rows where salary > 50000
- Filter with multiple conditions (AND)
- Sort rows by one column
- Sort rows by multiple columns
- groupby with count
- groupby with mean salary per department

NOTE ON DATA:
  This exercise uses a simple employee dataset for clarity.
  The core pandas skills (filter, sort, groupby) are identical
  whether you use employees, news headlines, or LLM training data.

HOW TO USE THIS FILE
---------------------
1. Read each exercise description carefully.
2. Write your code in the TODO section.
3. Run: python exercise_03_filtering.py
4. Check your output.
5. Read HINTS if stuck. Check SOLUTIONS at the bottom.

LIBRARIES NEEDED
-----------------
pandas -- pip install pandas
io     -- standard library (no install)
"""

import pandas as pd
from io import StringIO

print("=" * 60)
print("EXERCISE 03: Filtering, Sorting, and Grouping")
print("=" * 60)

# ---- Setup data ----
csv_text = """name,department,salary,years_experience,performance
Alice,Engineering,85000,5,Excellent
Bob,Marketing,48000,2,Good
Carol,Engineering,92000,8,Excellent
Dave,HR,41000,1,Good
Eve,Engineering,67000,3,Good
Frank,Marketing,55000,4,Excellent
Grace,HR,38000,1,Average
Henry,Engineering,110000,12,Excellent
Iris,Marketing,62000,6,Good
Jack,HR,45000,3,Average
Kate,Engineering,78000,7,Good
Leo,Marketing,51000,4,Good
"""

df = pd.read_csv(StringIO(csv_text))

print("Employee data:")
print(df)
print(f"\nShape: {df.shape}")

# =============================================================================
# EXERCISE 1: Basic filter -- salary > 50000
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 1: Filter employees with salary > 50000")
print("-" * 60)
print("""
Create a DataFrame called 'high_earners' with employees earning more than 50000.
Print:
  a) The high_earners DataFrame (show name, department, salary)
  b) How many high earners are there?

Expected: 8 employees (Alice, Carol, Eve, Frank, Henry, Iris, Kate, Leo)
""")

# TODO a): Filter
# high_earners = ...
# print(high_earners[["name", "department", "salary"]])

# TODO b): Count
# print(f"Count: ...")


# =============================================================================
# EXERCISE 2: Multiple conditions
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 2: Multiple conditions")
print("-" * 60)
print("""
Find employees who meet ALL of these conditions:
  - Department is "Engineering"
  - Salary >= 80000
  - Performance is "Excellent"

Create a DataFrame called 'elite_engineers'.
Print the result showing name, salary, and performance.

Expected: Alice (85000), Carol (92000), Henry (110000)
""")

# TODO: Your code here
# elite_engineers = ...
# print(...)

# Bonus: Find employees in Engineering OR Marketing with >= 6 years experience
# Expected: Carol (8y), Henry (12y), Iris (6y), Kate (7y)
# print("\nBonus: Engineering or Marketing with >= 6 years:")
# bonus = ...
# print(bonus[["name", "department", "years_experience"]])


# =============================================================================
# EXERCISE 3: Sort by one column
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 3: Sort by one column")
print("-" * 60)
print("""
  a) Sort all employees by salary, highest to lowest (descending).
     Show name and salary.
     Expected first row: Henry (110000)

  b) Sort all employees by years_experience, lowest to highest (ascending).
     Show name and years_experience.
     Expected first row: Dave or Grace (1 year)
""")

# TODO a): Sort by salary descending
#

# TODO b): Sort by years ascending
#


# =============================================================================
# EXERCISE 4: Sort by multiple columns
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 4: Sort by multiple columns")
print("-" * 60)
print("""
Sort by department (A-Z) and then by salary (highest first within each department).
Show name, department, and salary.

Expected order (first few):
  Engineering: Henry (110000), Carol (92000), Alice (85000), Kate (78000), Eve (67000)
  HR: Jack (45000), Dave (41000), Grace (38000)
  Marketing: Iris (62000), Leo (51000), Frank (55000)  <-- Frank should be 55000
""")

# TODO: Sort by department ascending, salary descending
# sorted_df = ...
# print(sorted_df[["name", "department", "salary"]])


# =============================================================================
# EXERCISE 5: groupby -- count per department
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 5: groupby -- count employees per department")
print("-" * 60)
print("""
How many employees are in each department?
Use groupby("department") and count.

Expected:
  Engineering    5
  HR             3
  Marketing      4
""")

# TODO: groupby and count
# dept_count = ...
# print(dept_count)


# =============================================================================
# EXERCISE 6: groupby -- mean salary per department
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 6: groupby -- mean salary per department")
print("-" * 60)
print("""
What is the average salary in each department?
Round to 2 decimal places.

Expected (approx):
  Engineering    86400.00
  HR             41333.33
  Marketing      54000.00
""")

# TODO: groupby and mean
# avg_salary = ...
# print(avg_salary.round(2))


# =============================================================================
# EXERCISE 7: groupby -- multiple stats + sort
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 7: groupby -- multiple stats per department")
print("-" * 60)
print("""
For each department, compute:
  - employee_count : number of employees
  - avg_salary     : mean salary
  - max_salary     : highest salary
  - avg_experience : mean years_experience

Sort the result by avg_salary descending.
Round numeric columns to 1 decimal place.

Hint: df.groupby("col").agg(name=("col","func"), ...)
""")

# TODO: groupby with multiple aggregations
# dept_stats = df.groupby("department").agg(
#     employee_count=...,
#     avg_salary=...,
#     max_salary=...,
#     avg_experience=...
# )
# print(dept_stats.round(1).sort_values("avg_salary", ascending=False))


# =============================================================================
# EXERCISE 8: value_counts on performance
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 8: value_counts -- performance distribution")
print("-" * 60)
print("""
How many employees have each performance rating?
Show as counts and then as percentages.

Expected counts:
  Good        6
  Excellent   4
  Average     2

Expected percentages (normalize=True):
  Good        0.50 (50%)
  Excellent   0.33 (33%)
  Average     0.17 (17%)
""")

# TODO a): value_counts (counts)
#

# TODO b): value_counts with normalize=True (percentages)
#


print("\n" + "=" * 60)
print("All exercises complete! Check output above.")
print("Read HINTS if stuck. SOLUTIONS below.")
print("=" * 60)

# =============================================================================
# HINTS
# =============================================================================
"""
HINTS:

Exercise 1:
  - df[df["salary"] > 50000]
  - len(result) gives you the count

Exercise 2:
  - Combine conditions: (cond1) & (cond2) & (cond3)
  - Each condition needs its own parentheses
  - For OR: (cond1) | (cond2)

Exercise 3:
  - df.sort_values("salary", ascending=False)   # descending
  - df.sort_values("years_experience")           # ascending is default

Exercise 4:
  - df.sort_values(["department", "salary"], ascending=[True, False])
  - First column sorted ascending, second column sorted descending

Exercise 5:
  - df.groupby("department")["name"].count()
  - OR: df.groupby("department").size()

Exercise 6:
  - df.groupby("department")["salary"].mean()

Exercise 7:
  - df.groupby("department").agg(
        employee_count=("name",             "count"),
        avg_salary=    ("salary",           "mean"),
        max_salary=    ("salary",           "max"),
        avg_experience=("years_experience", "mean")
    )

Exercise 8:
  - df["performance"].value_counts()
  - df["performance"].value_counts(normalize=True).round(2)
"""

# =============================================================================
# SOLUTIONS (try on your own first!)
# =============================================================================

def show_solutions():
    """Run this function to see the solutions."""
    import pandas as pd
    from io import StringIO

    csv_text = """name,department,salary,years_experience,performance
Alice,Engineering,85000,5,Excellent
Bob,Marketing,48000,2,Good
Carol,Engineering,92000,8,Excellent
Dave,HR,41000,1,Good
Eve,Engineering,67000,3,Good
Frank,Marketing,55000,4,Excellent
Grace,HR,38000,1,Average
Henry,Engineering,110000,12,Excellent
Iris,Marketing,62000,6,Good
Jack,HR,45000,3,Average
Kate,Engineering,78000,7,Good
Leo,Marketing,51000,4,Good
"""
    df = pd.read_csv(StringIO(csv_text))

    print("\n\n" + "=" * 60)
    print("SOLUTIONS")
    print("=" * 60)

    print("\n--- SOLUTION 1: salary > 50000 ---")
    high_earners = df[df["salary"] > 50000]
    print(high_earners[["name", "department", "salary"]])
    print(f"Count: {len(high_earners)}")

    print("\n--- SOLUTION 2: Multiple conditions ---")
    elite_engineers = df[
        (df["department"] == "Engineering") &
        (df["salary"] >= 80000) &
        (df["performance"] == "Excellent")
    ]
    print(elite_engineers[["name", "salary", "performance"]])
    print("\nBonus: Engineering or Marketing with >= 6 years:")
    bonus = df[
        ((df["department"] == "Engineering") | (df["department"] == "Marketing")) &
        (df["years_experience"] >= 6)
    ]
    print(bonus[["name", "department", "years_experience"]])

    print("\n--- SOLUTION 3a: Sort by salary desc ---")
    print(df.sort_values("salary", ascending=False)[["name", "salary"]])

    print("\n--- SOLUTION 3b: Sort by experience asc ---")
    print(df.sort_values("years_experience")[["name", "years_experience"]])

    print("\n--- SOLUTION 4: Multi-column sort ---")
    sorted_df = df.sort_values(["department", "salary"], ascending=[True, False])
    print(sorted_df[["name", "department", "salary"]])

    print("\n--- SOLUTION 5: Count per department ---")
    dept_count = df.groupby("department")["name"].count()
    print(dept_count)

    print("\n--- SOLUTION 6: Mean salary per department ---")
    avg_salary = df.groupby("department")["salary"].mean()
    print(avg_salary.round(2))

    print("\n--- SOLUTION 7: Multiple stats per department ---")
    dept_stats = df.groupby("department").agg(
        employee_count=("name",             "count"),
        avg_salary=    ("salary",           "mean"),
        max_salary=    ("salary",           "max"),
        avg_experience=("years_experience", "mean")
    )
    print(dept_stats.round(1).sort_values("avg_salary", ascending=False))

    print("\n--- SOLUTION 8: value_counts ---")
    print(df["performance"].value_counts())
    print("\nAs percentages:")
    print(df["performance"].value_counts(normalize=True).round(2))

# Uncomment the line below to see solutions:
# show_solutions()

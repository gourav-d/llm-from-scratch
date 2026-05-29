"""
=============================================================================
EXERCISE 01: DataFrame Basics
=============================================================================

WHAT YOU PRACTICE
------------------
- Creating a DataFrame from a Python dictionary
- Accessing a single column
- Filtering rows by a value
- Checking the shape of a DataFrame
- Using head() and tail()
- Using iloc to access a specific row

HOW TO USE THIS FILE
---------------------
1. Read each exercise description carefully.
2. Write your code in the TODO section below the description.
3. Run the file: python exercise_01_dataframes.py
4. Check your output matches the expected output.
5. If stuck, read the HINTS section (search for "HINTS").
6. Check the SOLUTIONS section at the bottom (search for "SOLUTIONS").
   Try on your own BEFORE reading the solutions!

LIBRARIES NEEDED
-----------------
pandas  -- pip install pandas
"""

import pandas as pd

print("=" * 60)
print("EXERCISE 01: DataFrame Basics")
print("=" * 60)

# =============================================================================
# EXERCISE 1: Create a DataFrame from a dictionary
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 1: Create a DataFrame from a dictionary")
print("-" * 60)
print("""
Create a DataFrame called 'df' from the following data:

  title       : ["The Shining", "Alien", "Blade Runner", "Metropolis", "2001"]
  director    : ["Kubrick", "Scott", "Scott", "Lang", "Kubrick"]
  year        : [1980, 1979, 1982, 1927, 1968]
  imdb_rating : [8.4, 8.5, 8.1, 8.3, 8.3]
  genre       : ["Horror", "Sci-Fi", "Sci-Fi", "Sci-Fi", "Sci-Fi"]

Then print the whole DataFrame.
""")

# TODO: Create the DataFrame here
# data = {
#     ...
# }
# df = pd.DataFrame(data)
# print(df)


# =============================================================================
# EXERCISE 2: Access a specific column
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 2: Access a specific column")
print("-" * 60)
print("""
Using the DataFrame you created in Exercise 1:
  a) Print only the 'title' column
  b) Print only the 'imdb_rating' column
  c) Print both 'title' and 'year' columns together
""")

# TODO: Your code here
# a)

# b)

# c)


# =============================================================================
# EXERCISE 3: Filter rows by a value
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 3: Filter rows by a value")
print("-" * 60)
print("""
Using the same DataFrame:
  a) Filter to only Sci-Fi movies (genre == "Sci-Fi")
  b) Filter to movies with imdb_rating >= 8.4
  c) Filter to movies directed by "Kubrick"

Print each result.
Expected for (a): Alien, Blade Runner, Metropolis, 2001
Expected for (b): The Shining (8.4), Alien (8.5)
Expected for (c): The Shining, 2001
""")

# TODO: Your code here
# a)

# b)

# c)


# =============================================================================
# EXERCISE 4: Check the shape
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 4: Check the shape")
print("-" * 60)
print("""
Using the same DataFrame:
  a) Print df.shape and explain what the numbers mean
  b) Print the number of rows using df.shape[0]
  c) Print the number of columns using df.shape[1]
  d) Print df.dtypes and identify which columns are strings

Expected shape: (5, 5)
""")

# TODO: Your code here
# a)

# b)

# c)

# d)


# =============================================================================
# EXERCISE 5: head, tail, and iloc
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 5: head, tail, and iloc")
print("-" * 60)
print("""
Using the same DataFrame:
  a) Print the first 3 rows using head()
  b) Print the last 2 rows using tail()
  c) Print the row at position 2 using iloc
  d) Print the title of the movie at position 4 using iloc
     (hint: df.iloc[4, column_position] or df.iloc[4]["title"])

Expected for (c): Blade Runner row
Expected for (d): "2001"
""")

# TODO: Your code here
# a)

# b)

# c)

# d)


print("\n" + "=" * 60)
print("All exercises complete! Check your outputs above.")
print("If something doesn't look right, read the HINTS below,")
print("then check the SOLUTIONS at the bottom.")
print("=" * 60)

# =============================================================================
# HINTS (read these if you are stuck)
# =============================================================================
"""
HINTS:

Exercise 1:
  - A DataFrame from a dict: pd.DataFrame({"key": [list], "key2": [list]})
  - All lists must have the SAME length (5 items each here)
  - Assign to a variable: df = pd.DataFrame(data)

Exercise 2:
  - Single column: df["column_name"]
  - Multiple columns: df[["col1", "col2"]]  (double brackets = list inside [])
  - df.title also works for accessing a column (dot notation)

Exercise 3:
  - Filter with condition: df[df["column"] == value]
  - For >=: df[df["column"] >= 8.4]
  - String match: df[df["director"] == "Kubrick"]
  - The condition goes inside df[ ... ]

Exercise 4:
  - df.shape returns a tuple like (5, 5)
  - df.shape[0] is the row count (first element of tuple)
  - df.shape[1] is the column count (second element of tuple)
  - df.dtypes shows each column's type: object=string, int64=int, float64=float

Exercise 5:
  - df.head(3) returns first 3 rows
  - df.tail(2) returns last 2 rows
  - df.iloc[2] returns row at position 2 (third row, 0-indexed)
  - df.iloc[4, 0] returns row 4, column 0 (first column = "title")
  - Or: df.iloc[4]["title"] also works
"""

# =============================================================================
# SOLUTIONS (try on your own first!)
# =============================================================================

def show_solutions():
    """Run this function to see the solutions. Called at the end."""
    import pandas as pd

    print("\n\n" + "=" * 60)
    print("SOLUTIONS")
    print("=" * 60)

    # ---- Solution 1 ----
    print("\n--- SOLUTION 1: Create DataFrame ---")
    data = {
        "title":       ["The Shining", "Alien", "Blade Runner", "Metropolis", "2001"],
        "director":    ["Kubrick", "Scott", "Scott", "Lang", "Kubrick"],
        "year":        [1980, 1979, 1982, 1927, 1968],
        "imdb_rating": [8.4, 8.5, 8.1, 8.3, 8.3],
        "genre":       ["Horror", "Sci-Fi", "Sci-Fi", "Sci-Fi", "Sci-Fi"]
    }
    df = pd.DataFrame(data)
    print(df)

    # ---- Solution 2 ----
    print("\n--- SOLUTION 2a: title column ---")
    print(df["title"])

    print("\n--- SOLUTION 2b: imdb_rating column ---")
    print(df["imdb_rating"])

    print("\n--- SOLUTION 2c: title and year together ---")
    print(df[["title", "year"]])

    # ---- Solution 3 ----
    print("\n--- SOLUTION 3a: Sci-Fi movies ---")
    print(df[df["genre"] == "Sci-Fi"])

    print("\n--- SOLUTION 3b: Rating >= 8.4 ---")
    print(df[df["imdb_rating"] >= 8.4])

    print("\n--- SOLUTION 3c: Directed by Kubrick ---")
    print(df[df["director"] == "Kubrick"])

    # ---- Solution 4 ----
    print("\n--- SOLUTION 4a: shape ---")
    print(f"df.shape = {df.shape}")
    print("This means: 5 rows and 5 columns")

    print("\n--- SOLUTION 4b: row count ---")
    print(f"Row count: {df.shape[0]}")

    print("\n--- SOLUTION 4c: column count ---")
    print(f"Column count: {df.shape[1]}")

    print("\n--- SOLUTION 4d: dtypes ---")
    print(df.dtypes)
    print("String columns (dtype = object): title, director, genre")

    # ---- Solution 5 ----
    print("\n--- SOLUTION 5a: head(3) ---")
    print(df.head(3))

    print("\n--- SOLUTION 5b: tail(2) ---")
    print(df.tail(2))

    print("\n--- SOLUTION 5c: iloc[2] ---")
    print(df.iloc[2])

    print("\n--- SOLUTION 5d: title at position 4 ---")
    print(f"Title at position 4: {df.iloc[4]['title']}")

# Uncomment the line below to see solutions:
# show_solutions()

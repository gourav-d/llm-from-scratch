"""
=============================================================================
EXAMPLE 01: DataFrame Basics
=============================================================================

WHAT THIS FILE TEACHES
-----------------------
- Creating a DataFrame from a Python dictionary
- Creating a DataFrame from a list of dicts
- Inspecting shape, dtypes, head, tail, info, describe
- Accessing a column (Series)
- Accessing multiple columns
- iloc  -- row access by integer position
- loc   -- row access by label / condition

GLOSSARY
---------
DataFrame   : A table with rows and columns. Like C#'s DataTable.
Series      : A single column (a labelled list). Like DataColumn + its values.
Index       : The row label on the left side (usually 0, 1, 2 ...).
shape       : Tuple (rows, columns). df.shape returns e.g. (5, 4).
dtypes      : Data type of each column. object=string, int64=int, float64=double.
head(n)     : First n rows. Like LINQ Take(n).
tail(n)     : Last n rows. Like LINQ TakeLast(n).
info()      : Summary of columns, types, and null counts.
describe()  : Statistics: count, mean, std, min, max for numeric columns.
iloc        : Access by integer index position. df.iloc[0] = first row.
loc         : Access by label. df.loc[0, "title"] = row 0, column "title".

C# ANALOGY
-----------
C#:     DataTable + DataRow + DataColumn
pandas: DataFrame + row (Series) + column (Series)

  C#:     table.Rows[0]["title"]     -->  pandas: df.loc[0, "title"]
  C#:     table.Rows.Count           -->  pandas: df.shape[0]
  C#:     table.Columns.Count        -->  pandas: df.shape[1]
  C#:     table.Select("rating > 8") -->  pandas: df[df["rating"] > 8]

LIBRARIES
----------
pandas  : The core data library. Imported as "pd" (everyone uses this alias).
          Install with: pip install pandas
"""

import pandas as pd          # import pandas, give it the alias "pd"

print("=" * 60)
print("EXAMPLE 01: DataFrame Basics")
print("=" * 60)

# =============================================================================
# PART A: Creating DataFrames
# =============================================================================

print("\n" + "=" * 60)
print("PART A: Creating DataFrames")
print("=" * 60)

# ---- Method 1: From a dictionary ----
# Dictionary keys become column names.
# Dictionary values (lists) become the column's data.
# All lists must be the same length.

print("\n--- Method 1: DataFrame from a dictionary ---")

data = {
    "title":     ["Dune", "1917", "Joker", "Parasite", "Morbius"],
    "year":      [2021,   2019,   2019,    2019,        2022],
    "genre":     ["Sci-Fi", "War", "Thriller", "Drama", "Action"],
    "rating":    [8.0,    8.2,    8.4,     8.5,         5.2],
    "sentiment": ["positive", "positive", "positive", "positive", "negative"]
}

# pd.DataFrame() converts the dictionary into a table
df = pd.DataFrame(data)

# Print the whole table
print(df)

# ---- Method 2: From a list of dicts ----
# Each dict is one row. Keys are column names.
# This is what you get when you load a JSON file with "records" orientation.

print("\n--- Method 2: DataFrame from a list of dicts ---")

rows = [
    {"title": "Dune",     "year": 2021, "genre": "Sci-Fi",   "rating": 8.0},
    {"title": "1917",     "year": 2019, "genre": "War",      "rating": 8.2},
    {"title": "Joker",    "year": 2019, "genre": "Thriller", "rating": 8.4},
    {"title": "Parasite", "year": 2019, "genre": "Drama",    "rating": 8.5},
    {"title": "Morbius",  "year": 2022, "genre": "Action",   "rating": 5.2},
]

df2 = pd.DataFrame(rows)    # same result as Method 1

print(df2)
print("\nBoth methods produce identical results:", df.equals(df2))

# =============================================================================
# PART B: Inspecting a DataFrame
# =============================================================================

print("\n" + "=" * 60)
print("PART B: Inspecting a DataFrame")
print("=" * 60)

# ---- shape: how many rows and columns? ----
print("\n--- df.shape ---")
print(f"Shape: {df.shape}")           # (5, 5) -- 5 rows, 5 columns
print(f"Rows:    {df.shape[0]}")      # 5
print(f"Columns: {df.shape[1]}")      # 5

# ---- dtypes: what type is each column? ----
print("\n--- df.dtypes ---")
print(df.dtypes)
# object = string (text)
# int64  = 64-bit integer (like C# long)
# float64 = 64-bit float (like C# double)

# ---- head() and tail() ----
print("\n--- df.head(3) -- first 3 rows ---")
print(df.head(3))          # first 3 rows

print("\n--- df.tail(2) -- last 2 rows ---")
print(df.tail(2))          # last 2 rows

# ---- info(): column summary ----
print("\n--- df.info() -- column summary ---")
df.info()
# Shows: column name, non-null count, dtype
# "Non-Null Count" tells you how many values are NOT missing

# ---- describe(): statistics ----
print("\n--- df.describe() -- statistics for numeric columns ---")
print(df.describe())
# count: how many values (non-NaN)
# mean:  average
# std:   standard deviation (how spread out the values are)
# min:   smallest value
# 25%:   first quartile
# 50%:   median (middle value)
# 75%:   third quartile
# max:   largest value

# describe() for all columns including text
print("\n--- df.describe(include='all') -- includes text columns ---")
print(df.describe(include="all"))
# For text columns: count, unique, top (most common), freq (how many times top appears)

# =============================================================================
# PART C: Accessing Columns and Rows
# =============================================================================

print("\n" + "=" * 60)
print("PART C: Accessing Columns and Rows")
print("=" * 60)

# ---- Accessing a single column ----
print("\n--- Access single column: df['title'] ---")
titles = df["title"]       # returns a Series (a labelled list)
print(titles)
print(f"\nType of titles: {type(titles)}")  # pandas.core.series.Series

# ---- Accessing multiple columns ----
print("\n--- Access two columns: df[['title','rating']] ---")
# Double brackets! You are passing a LIST of names inside df[...]
subset = df[["title", "rating"]]
print(subset)

# ---- Column as a list ----
print("\n--- Column values as a Python list ---")
title_list = df["title"].tolist()   # .tolist() converts Series to plain Python list
print(title_list)

# ---- Accessing a single value by column name ----
print("\n--- Get value at row 0, column 'title' ---")
print(df.loc[0, "title"])           # "Dune"

# =============================================================================
# PART D: iloc vs loc
# =============================================================================

print("\n" + "=" * 60)
print("PART D: iloc vs loc")
print("=" * 60)

print("""
REMEMBER:
  iloc = Integer LOCation  --> use row NUMBER
  loc  = Label-based LOCation --> use row INDEX LABEL and COLUMN NAME
""")

# ---- iloc: by integer position ----
print("--- iloc examples ---")

print("\ndf.iloc[0]   -- first row as a Series:")
print(df.iloc[0])                  # row at position 0 (the first row)

print("\ndf.iloc[-1]  -- last row:")
print(df.iloc[-1])                 # negative index = count from the end

print("\ndf.iloc[1:3] -- rows at positions 1 and 2 (stops BEFORE 3):")
print(df.iloc[1:3])                # slice: positions 1 and 2

print("\ndf.iloc[0, 0] -- row 0, column 0 (single cell):")
print(df.iloc[0, 0])               # "Dune"

print("\ndf.iloc[0:2, 0:2] -- rows 0-1, columns 0-1:")
print(df.iloc[0:2, 0:2])           # top-left 2x2 chunk

# ---- loc: by label ----
print("\n--- loc examples ---")

print("\ndf.loc[0, 'title'] -- row label 0, column 'title':")
print(df.loc[0, "title"])           # "Dune"

print("\ndf.loc[0:2, ['title','rating']] -- rows 0,1,2 (INCLUSIVE), two columns:")
# NOTE: loc slices are INCLUSIVE on both ends (unlike iloc)
print(df.loc[0:2, ["title", "rating"]])

print("\ndf.loc[df['rating'] > 8.3, 'title'] -- titles of movies rated > 8.3:")
# loc with a boolean mask on rows and a column name
print(df.loc[df["rating"] > 8.3, "title"])

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
What you learned in this example:

  Create DataFrame:
    df = pd.DataFrame(dict)          # from a dictionary of lists
    df = pd.DataFrame(list_of_dicts) # from a list of row dicts

  Inspect:
    df.shape        --> (rows, columns)
    df.dtypes       --> column types
    df.head(n)      --> first n rows
    df.tail(n)      --> last n rows
    df.info()       --> column summary with null counts
    df.describe()   --> statistics for numeric columns

  Access columns:
    df["col"]             --> single column (Series)
    df[["col1","col2"]]   --> multiple columns (DataFrame)

  Access rows:
    df.iloc[0]            --> row by integer position
    df.iloc[1:3]          --> slice by position
    df.loc[0,"col"]       --> row by label + column name
    df.loc[mask,"col"]    --> filtered rows + column name

C# Translation:
  df.shape[0]         = table.Rows.Count
  df["col"]           = table.Columns["col"]
  df.loc[0,"title"]   = table.Rows[0]["title"]
  df.head(3)          = rows.Take(3)
""")

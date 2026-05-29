# Lesson 1.5.1: DataFrame Basics

**What is a DataFrame? How do you create one? How do you look at what is inside?**

---

## Glossary (Read This First!)

| Word | Plain-English Meaning | C# Equivalent |
|------|-----------------------|---------------|
| **DataFrame** | A table of data with rows and columns | `DataTable` |
| **Series** | A single column (a list with a label) | `DataColumn` / `IEnumerable<T>` |
| **Index** | The row number or row label on the left side | Row ID / primary key |
| **Column** | A named field in the table | `DataColumn` name |
| **dtype** | The data type of a column (int, float, string, etc.) | `SqlDbType` / C# type |
| **shape** | A tuple saying (number of rows, number of columns) | `rowCount, colCount` |
| **head()** | Show the first N rows (default 5) | `Take(5)` in LINQ |
| **tail()** | Show the last N rows (default 5) | `TakeLast(5)` in LINQ |
| **info()** | Print a summary: column names, dtypes, null counts | Like `ToString()` on a schema |
| **describe()** | Print statistics: mean, min, max, std for numeric columns | Like `DataTable.Compute()` |
| **NaN** | "Not a Number" -- a missing value placeholder | `null` for value types |
| **pd** | Short alias for pandas (everyone uses this) | `using pd = Pandas;` |

---

## Part 1 -- What is a DataFrame?

A **DataFrame** is a table. That is the whole idea.

```
+-------+-----+------------+--------+
| title | year| genre      | rating |
+-------+-----+------------+--------+
| Dune  | 2021| Sci-Fi     | 8.0    |
| 1917  | 2019| War        | 8.2    |
| Joker | 2019| Thriller   | 8.4    |
+-------+-----+------------+--------+
  ^col    ^col    ^col          ^col
```

Each vertical strip is a **column** (also called a Series).
Each horizontal strip is a **row**.
The numbers on the far left (0, 1, 2 ...) are the **index**.

### C# DataTable Analogy

```csharp
// In C# you might do this:
DataTable movies = new DataTable();
movies.Columns.Add("title",  typeof(string));
movies.Columns.Add("year",   typeof(int));
movies.Columns.Add("genre",  typeof(string));
movies.Columns.Add("rating", typeof(double));

DataRow row = movies.NewRow();
row["title"]  = "Dune";
row["year"]   = 2021;
row["genre"]  = "Sci-Fi";
row["rating"] = 8.0;
movies.Rows.Add(row);
```

```python
# In Python with pandas you do the same thing in 3 lines:
import pandas as pd

movies = pd.DataFrame({
    "title":  ["Dune", "1917", "Joker"],
    "year":   [2021, 2019, 2019],
    "genre":  ["Sci-Fi", "War", "Thriller"],
    "rating": [8.0, 8.2, 8.4]
})
```

Same result. Much less code.

---

## Part 2 -- Creating a DataFrame

### Method A: From a Dictionary

This is the most common method. The dictionary keys become column names.
The dictionary values are lists -- one list per column.

```python
import pandas as pd

# Each key = column name
# Each value = list of values in that column
data = {
    "title":  ["Dune", "1917", "Joker", "Parasite"],
    "year":   [2021, 2019, 2019, 2019],
    "genre":  ["Sci-Fi", "War", "Thriller", "Drama"],
    "rating": [8.0, 8.2, 8.4, 8.5]
}

df = pd.DataFrame(data)  # "df" is the convention name for DataFrames
print(df)
```

Output:
```
      title  year     genre  rating
0      Dune  2021    Sci-Fi     8.0
1      1917  2019       War     8.2
2     Joker  2019  Thriller     8.4
3  Parasite  2019     Drama     8.5
```

### ASCII Diagram: dict -> DataFrame

```
Dictionary (Python)        DataFrame (pandas)
-------------------        -----------------
{                          index | title    | year | genre    | rating
 "title": ["Dune",           0  | Dune     | 2021 | Sci-Fi   | 8.0
           "1917",    -->     1  | 1917     | 2019 | War      | 8.2
           "Joker"],          2  | Joker    | 2019 | Thriller | 8.4
 "year":  [2021,
           2019,
           2019],
 ...
}
```

---

### Method B: From a List of Dicts

Sometimes your data is already a list of dicts (e.g. loaded from a JSON file).
Each dict is one row.

```python
import pandas as pd

# Each dict in the list = one row
rows = [
    {"title": "Dune",     "year": 2021, "genre": "Sci-Fi",   "rating": 8.0},
    {"title": "1917",     "year": 2019, "genre": "War",      "rating": 8.2},
    {"title": "Joker",    "year": 2019, "genre": "Thriller", "rating": 8.4},
    {"title": "Parasite", "year": 2019, "genre": "Drama",    "rating": 8.5},
]

df = pd.DataFrame(rows)
print(df)
```

This produces exactly the same result as Method A.

---

## Part 3 -- Inspecting a DataFrame

Once you have a DataFrame, the first thing you do is look at it.

### shape -- How big is the table?

```python
print(df.shape)       # (4, 4) --> 4 rows, 4 columns
print(df.shape[0])    # 4 --> just the row count
print(df.shape[1])    # 4 --> just the column count
```

C# analogy:
```csharp
int rows = table.Rows.Count;
int cols = table.Columns.Count;
```

---

### dtypes -- What types are the columns?

```python
print(df.dtypes)
```

Output:
```
title      object    <-- "object" means string in pandas
year        int64    <-- 64-bit integer
genre      object    <-- string
rating    float64    <-- 64-bit float (like C# double)
dtype: object
```

C# analogy:
```csharp
foreach (DataColumn col in table.Columns)
    Console.WriteLine($"{col.ColumnName}: {col.DataType}");
```

---

### head() and tail() -- See the data

```python
print(df.head())     # first 5 rows (default)
print(df.head(2))    # first 2 rows
print(df.tail())     # last 5 rows (default)
print(df.tail(2))    # last 2 rows
```

C# LINQ analogy:
```csharp
var first2 = rows.Take(2);
var last2  = rows.TakeLast(2);
```

---

### info() -- Column summary

```python
df.info()
```

Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   title   4 non-null      object
 1   year    4 non-null      int64
 2   genre   4 non-null      object
 3   rating  4 non-null      float64
dtypes: float64(1), int64(1), object(2)
memory usage: 260.0+ bytes
```

This tells you:
- How many rows are NOT null (missing)
- What type each column is
- How much memory the DataFrame uses

---

### describe() -- Statistics at a glance

```python
print(df.describe())
```

Output:
```
            year    rating
count   4.000000  4.000000
mean 2019.500000  8.275000
std     0.577350  0.208167
min  2019.000000  8.000000
25%  2019.000000  8.150000
50%  2019.000000  8.300000
75%  2019.500000  8.425000
max  2021.000000  8.500000
```

Note: `describe()` only shows numeric columns by default.
Use `df.describe(include="all")` to include text columns.

C# analogy -- imagine writing a loop to compute min, max, mean, std
for every column and printing it. `describe()` does that in one call.

---

## Part 4 -- Accessing Columns and Rows

### Accessing a Column

```python
# Two ways to get a column (both return a Series)
titles = df["title"]     # dictionary-style (works for all column names)
titles = df.title        # attribute-style (only works if name has no spaces)

print(titles)
# 0        Dune
# 1        1917
# 2       Joker
# 3    Parasite
# Name: title, dtype: object
```

### Accessing Multiple Columns

```python
# Pass a LIST of column names inside the brackets
subset = df[["title", "rating"]]
print(subset)
```

Output:
```
      title  rating
0      Dune     8.0
1      1917     8.2
2     Joker     8.4
3  Parasite     8.5
```

### ASCII Diagram: Column Selection

```
Full DataFrame            df[["title","rating"]]
------------------        ----------------------
index | title | year |    index | title    | rating
  0   | Dune  | 2021 |      0  | Dune     | 8.0
  1   | 1917  | 2019 |  -->  1  | 1917     | 8.2
  2   | Joker | 2019 |      2  | Joker    | 8.4
  ...                        3  | Parasite | 8.5
```

---

## Part 5 -- iloc vs loc

These are the two main ways to access rows.

### iloc -- Access by Position (integer location)

`iloc` works like array indexing in C#. Use row/column numbers.

```python
# Single row by position
print(df.iloc[0])       # row at index position 0 (first row)
print(df.iloc[-1])      # last row

# Slice: rows 0 and 1
print(df.iloc[0:2])

# Row 0, Column 0
print(df.iloc[0, 0])    # "Dune"

# Row 0, Columns 0 and 2
print(df.iloc[0, [0, 2]])
```

C# analogy:
```csharp
DataRow row = table.Rows[0];   // iloc[0]
```

### loc -- Access by Label (column name or index label)

`loc` uses column names and index labels, not position numbers.

```python
# Row with index label 0, column "title"
print(df.loc[0, "title"])      # "Dune"

# Rows 0 to 2 (inclusive!), just "title" and "rating"
print(df.loc[0:2, ["title", "rating"]])
```

Important difference from iloc:
- `iloc[0:2]` means rows 0 and 1 (stops BEFORE 2)
- `loc[0:2]`  means rows 0, 1, AND 2 (includes 2!)

This is unusual -- loc slices are INCLUSIVE on both ends.

### Side-by-Side Comparison

```
              iloc (by position)      loc (by label)
              ------------------      --------------
Get row 0     df.iloc[0]              df.loc[0]
Get cell      df.iloc[0, 1]           df.loc[0, "year"]
Get rows 0-1  df.iloc[0:2]            df.loc[0:1]
```

---

## Quiz

**Question 1 (Multiple Choice)**
What does `df.shape` return?

A) The name of the DataFrame
B) A tuple of (rows, columns)
C) The column names only
D) The number of bytes used

Answer: B -- `(4, 4)` means 4 rows and 4 columns.

---

**Question 2 (Multiple Choice)**
Which of the following selects the "genre" column?

A) `df.genre()` (with parentheses)
B) `df["genre"]`
C) `df.column("genre")`
D) `df.get_column["genre"]`

Answer: B -- dictionary-style access is the most reliable.
(A would call a method, C and D do not exist in pandas.)

---

**Question 3 (Short Answer)**
You have a DataFrame with 1000 rows. You want to see the first 10 rows.
What method do you use?

Answer: `df.head(10)`

---

**Question 4 (Short Answer)**
What is the difference between `iloc` and `loc`?

Answer:
- `iloc` uses integer position (like array index). `df.iloc[0]` is row 0.
- `loc`  uses label names. `df.loc[0, "title"]` finds the value in row 0,
  column "title".

---

**Question 5 (Multiple Choice)**
What does `df.dtypes` tell you?

A) The shape of the DataFrame
B) The number of missing values
C) The data type of each column
D) The memory used by the DataFrame

Answer: C -- `df.dtypes` lists the data type of every column.

---

## Key Takeaways

1. A DataFrame is a table (like `DataTable` in C#)
2. Create from a dict (keys = columns) or a list of dicts (each dict = row)
3. Use `shape`, `dtypes`, `head()`, `tail()`, `info()`, `describe()` to inspect
4. Access columns with `df["colname"]` or `df[["col1", "col2"]]`
5. Use `iloc` for position-based access, `loc` for label-based access
6. The index is the row label on the left (usually 0, 1, 2 ...)

---

**Next Lesson:** [02_loading_files.md](02_loading_files.md)
Learn how to load CSV, JSON, and JSONL files into DataFrames.

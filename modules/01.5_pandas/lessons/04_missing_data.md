# Lesson 1.5.4: Missing Data

**What is NaN, how do you find it, and what do you do with it?**

---

## Glossary (Read This First!)

| Word | Plain-English Meaning | C# Equivalent |
|------|-----------------------|---------------|
| **NaN** | "Not a Number" -- marks a missing or unknown value | `null` for nullable types (`int?`, `double?`) |
| **None** | Python's null. Pandas converts None to NaN in numeric columns | `null` |
| **isnull()** | Returns True where a value is missing (NaN/None) | `== null` check |
| **notnull()** | Returns True where a value is present (not missing) | `!= null` check |
| **isna()** | Same as isnull() -- just an alias | Same as isnull() |
| **dropna()** | Remove rows (or columns) that contain NaN values | Filter out nulls |
| **fillna()** | Replace NaN with a specific value or strategy | `?? defaultValue` |
| **ffill** | Forward-fill: copy the last known value downward | Like carry-forward in time-series |
| **bfill** | Backward-fill: copy the next known value upward | Look-ahead fill |
| **mean()** | The average of a column | `numbers.Average()` in LINQ |
| **median()** | The middle value when sorted | No direct LINQ equivalent |
| **thresh** | Minimum number of non-NaN values required (used in dropna) | Minimum valid fields |
| **axis** | 0 = act on rows, 1 = act on columns | Dimension to operate on |

---

## Part 1 -- What is NaN?

### The Problem

Real-world data is messy. Data often has missing values:
- A user did not fill in their age
- A sensor failed and recorded nothing
- A text column has no value for certain rows
- Data was deleted or was never collected

Python represents missing numeric data as `float('nan')`.
Pandas represents it as `NaN` (which stands for "Not a Number").

### NaN vs None vs null

```
C# concept      Python concept   pandas stored as
-----------     --------------   ----------------
null (int?)     None             NaN (for numeric)
null (string)   None             None or NaN (for object/string)
```

When you load a CSV with empty cells, pandas automatically puts NaN there.

### What NaN Looks Like in a DataFrame

```python
import pandas as pd
import numpy as np          # NaN lives in numpy

data = {
    "title":     ["Dune", "1917", "Joker",        "Parasite", "Morbius"],
    "rating":    [8.0,    8.2,    float("nan"),   8.5,        5.2],
    "review":    ["Great","Epic", None,            "Brilliant",None],
    "year":      [2021,   2019,   2019,            2019,       float("nan")]
}
df = pd.DataFrame(data)
print(df)
```

Output:
```
      title  rating     review    year
0      Dune     8.0      Great  2021.0
1      1917     8.2       Epic  2019.0
2     Joker     NaN       None  2019.0
3  Parasite     8.5  Brilliant  2019.0
4   Morbius     5.2       None     NaN
```

Row 2 (Joker) has no rating. Row 4 (Morbius) has no year.
Both show as NaN.

### ASCII Diagram: NaN in a Table

```
index | title    | rating | review    | year
  0   | Dune     |  8.0   | "Great"   | 2021
  1   | 1917     |  8.2   | "Epic"    | 2019
  2   | Joker    |  NaN   | None      | 2019    <-- missing rating
  3   | Parasite |  8.5   | "Brilliant"| 2019
  4   | Morbius  |  5.2   | None      | NaN     <-- missing year
                    ^                    ^
                 missing              missing
```

---

## Part 2 -- Detecting Missing Values

### isnull() -- True Where Missing

```python
# Check the whole DataFrame for NaN
print(df.isnull())
```

Output:
```
   title  rating  review   year
0  False   False   False  False
1  False   False   False  False
2  False    True    True  False
3  False   False   False  False
4  False   False    True   True
```

### Count Missing Values Per Column

```python
# Sum of True values = count of NaN in each column
print(df.isnull().sum())
```

Output:
```
title     0
rating    1
review    2
year      1
dtype: int64
```

This is the most useful missing-data command. Run it right after loading
any file to understand how much data is missing.

### Percentage Missing Per Column

```python
# Percentage of each column that is missing
missing_pct = df.isnull().sum() / len(df) * 100
print(missing_pct.round(1))
```

Output:
```
title      0.0
rating    20.0
review    40.0
year      20.0
dtype: float64
```

40% of the review column is missing!

### notnull() -- The Opposite

```python
# True where value IS present
print(df["rating"].notnull())
# 0     True
# 1     True
# 2    False   <-- NaN here
# 3     True
# 4     True
```

Use `notnull()` to filter rows that have a value:
```python
has_rating = df[df["rating"].notnull()]
```

---

## Part 3 -- Dropping Missing Values

### dropna() -- Remove Rows with Any NaN

```python
# Drop all rows that have ANY missing value
clean = df.dropna()
print(clean)
```

Output:
```
      title  rating     review    year
0      Dune     8.0      Great  2021.0
1      1917     8.2       Epic  2019.0
3  Parasite     8.5  Brilliant  2019.0
```

Rows 2 and 4 were removed (they had at least one NaN).

### dropna(subset=...) -- Drop Only When Specific Columns Are Missing

Often you only care if certain columns are missing.

```python
# Only drop rows where "rating" is missing
clean = df.dropna(subset=["rating"])
print(clean)
```

Output:
```
      title  rating     review    year
0      Dune     8.0      Great  2021.0
1      1917     8.2       Epic  2019.0
3  Parasite     8.5  Brilliant  2019.0
4   Morbius     5.2       None     NaN
```

Row 2 (missing rating) was dropped. Row 4 kept even though year is missing.

### dropna(thresh=...) -- Keep Rows with Enough Values

`thresh` means "minimum number of non-NaN values required to keep the row."

```python
# Keep rows that have at least 3 non-NaN values (out of 4 columns)
clean = df.dropna(thresh=3)
print(clean)
```

This keeps rows that are mostly complete.

C# analogy -- imagine a rule: "If a record has more than 50% null fields, discard it."
`thresh` implements that idea.

---

## Part 4 -- Filling Missing Values

Instead of dropping rows, you can fill NaN with a reasonable value.

### fillna() with a Constant

```python
# Fill all NaN with 0
filled = df["rating"].fillna(0)

# Fill text column with a placeholder
filled = df["review"].fillna("No review available")
```

### fillna() with the Mean

For numeric columns, filling with the mean is common and sensible.

```python
# Calculate the mean of the column (ignoring NaN automatically)
mean_rating = df["rating"].mean()
print(f"Mean rating: {mean_rating}")     # mean of [8.0, 8.2, 8.5, 5.2] = 7.475

# Fill NaN with the mean
df["rating"] = df["rating"].fillna(mean_rating)
```

### fillna() with the Median

The median is the middle value when sorted. It is less sensitive to outliers.
If most ratings are 8.x and one is 1.0 (an outlier), the median is better.

```python
median_rating = df["rating"].median()
df["rating"] = df["rating"].fillna(median_rating)
```

### Forward-Fill (ffill) -- Copy Previous Value

Useful for time-series data where missing means "same as yesterday."

```python
# Fill NaN by carrying the value above it forward
df["year"] = df["year"].fillna(method="ffill")
# OR in newer pandas: df["year"] = df["year"].ffill()
```

### Backward-Fill (bfill) -- Copy Next Value

```python
# Fill NaN by pulling the value below it up
df["year"] = df["year"].fillna(method="bfill")
# OR in newer pandas: df["year"] = df["year"].bfill()
```

### ASCII Diagram: Fill Strategies

```
Original column "year":    ffill result:    bfill result:   fillna(mean):
2021                       2021             2021            2021
2019                       2019             2019            2019
NaN              -->        2019             2019            2019.5
2019                       2019             2019            2019
NaN                        2019             NaN             2019.5
                                             ^
                                    (no next value to pull from)
```

---

## Part 5 -- Filling Different Columns Differently

In practice, different columns need different strategies.

```python
import pandas as pd
import numpy as np

data = {
    "title":     ["Dune", "1917",  "Joker", "Parasite", "Morbius"],
    "rating":    [8.0,    np.nan,  np.nan,  8.5,        5.2],
    "genre":     ["Sci-Fi", "War", None,    "Drama",    None],
    "year":      [2021,   2019,    2019,    np.nan,     2022]
}
df = pd.DataFrame(data)

# Strategy 1: Numeric column -- fill with median
df["rating"] = df["rating"].fillna(df["rating"].median())

# Strategy 2: Numeric column -- fill with mean
df["year"]   = df["year"].fillna(df["year"].mean())

# Strategy 3: Text column -- fill with the most common value (mode)
df["genre"]  = df["genre"].fillna(df["genre"].mode()[0])
# .mode() returns the most common value as a Series; [0] gets the first

# Verify: no more NaN
print(df.isnull().sum())
```

C# analogy:
```csharp
// C# equivalent idea:
movie.Rating = movie.Rating ?? averageRating;
movie.Genre  = movie.Genre  ?? mostCommonGenre;
```

---

## Quiz

**Question 1 (Multiple Choice)**
What does `df.isnull().sum()` return?

A) A single number: total missing cells in the DataFrame
B) A boolean DataFrame showing where values are missing
C) A Series with the count of missing values per column
D) The row index of every missing value

Answer: C -- it counts NaN per column and returns a Series.

---

**Question 2 (Multiple Choice)**
You have a "rating" column with some NaN values. You want to remove only
the rows where rating is missing. What do you use?

A) `df.dropna()`
B) `df.dropna(subset=["rating"])`
C) `df.drop("rating")`
D) `df.fillna(subset=["rating"])`

Answer: B -- `subset=["rating"]` restricts dropna to that column.

---

**Question 3 (Short Answer)**
What is the difference between filling with mean vs filling with median?
When would you prefer the median?

Answer:
- Mean = average of all values. Pulled toward outliers.
- Median = middle value when sorted. Unaffected by extreme outliers.
- Use median when there are outliers. Example: if 9 movies rate 8.0-9.0
  and one rates 1.0 (spam), the mean gets dragged down but the median stays around 8.x.

---

**Question 4 (Short Answer)**
What does forward-fill (ffill) do?

Answer:
Forward-fill copies the last valid (non-NaN) value downward to fill the gap.
Row 3 NaN becomes whatever row 2 had. Good for time-series data.

---

**Question 5 (Multiple Choice)**
What does `df["genre"].mode()[0]` return?

A) The first value in the genre column
B) The most common (most frequent) genre value
C) The genre at index 0
D) The median genre value

Answer: B -- `mode()` returns the most frequent value(s) as a Series.
`[0]` picks the first (and usually only) mode.

---

## Key Takeaways

1. NaN is Python/pandas for "missing value" -- like `null` in C#
2. `df.isnull().sum()` is your missing-data dashboard (run it after loading)
3. `dropna()` removes rows with missing values -- use `subset=` to be selective
4. `fillna()` replaces NaN -- choose the right strategy per column:
   - Numeric: mean or median
   - Text: mode (most common) or a placeholder like "Unknown"
   - Time-series: ffill or bfill
5. Never blindly use `dropna()` on all columns -- you may delete too much data

---

**Next Lesson:** [05_export_for_llm.md](05_export_for_llm.md)
Learn how to export DataFrames to CSV, JSON, and JSONL for LLM training.

# Lesson 1.5.3: Filtering, Sorting, and Grouping

**How to query your DataFrame -- the pandas equivalent of LINQ and SQL.**

---

## Glossary (Read This First!)

| Word | Plain-English Meaning | C# / LINQ Equivalent |
|------|-----------------------|----------------------|
| **Boolean indexing** | Filtering rows by a True/False condition | `Where(x => x.Rating > 8)` |
| **Boolean mask** | A Series of True/False values, one per row | `Predicate<T>` applied to all rows |
| **loc** | Access rows by label, supports boolean masks | LINQ `.Where()` with row selection |
| **iloc** | Access rows by integer position | `list[0]`, `list[1:5]` |
| **sort_values()** | Sort DataFrame rows by one or more columns | `OrderBy()` / `ThenBy()` in LINQ |
| **ascending** | Sort from smallest to largest (A-Z, 0-9) | `OrderBy()` |
| **descending** | Sort from largest to smallest (Z-A, 9-0) | `OrderByDescending()` |
| **groupby()** | Split data into groups by column value | `GroupBy()` in LINQ |
| **agg()** | Apply aggregate functions after groupby | `.Select(g => g.Sum())` etc. |
| **value_counts()** | Count how many times each unique value appears | `GroupBy().Count()` |
| **&** | AND operator for combining boolean masks | `&&` in C# |
| **pipe (bar)** | OR operator for combining boolean masks | `\|\|` in C# |
| **~** | NOT operator -- flip True to False | `!` in C# |

---

## Part 1 -- Boolean Indexing (Filtering Rows)

### The Core Idea

In pandas, filtering works in two steps:
1. Create a True/False mask by comparing a column to a value
2. Use that mask to select only the True rows

```python
import pandas as pd

data = {
    "title":     ["Dune", "1917", "Joker", "Parasite", "Morbius"],
    "year":      [2021, 2019, 2019, 2019, 2022],
    "genre":     ["Sci-Fi", "War", "Thriller", "Drama", "Action"],
    "rating":    [8.0, 8.2, 8.4, 8.5, 5.2],
    "sentiment": ["positive", "positive", "positive", "positive", "negative"]
}
df = pd.DataFrame(data)

# Step 1: Create a boolean mask
mask = df["rating"] > 8.0     # Returns Series of True/False

print(mask)
# 0    False  (8.0 is NOT > 8.0)
# 1     True
# 2     True
# 3     True
# 4    False
# Name: rating, dtype: bool

# Step 2: Apply the mask to the DataFrame
high_rated = df[mask]
print(high_rated)
```

Output:
```
      title  year     genre  rating sentiment
1      1917  2019       War     8.2  positive
2     Joker  2019  Thriller     8.4  positive
3  Parasite  2019     Drama     8.5  positive
```

You can do this in one line (most people write it this way):
```python
high_rated = df[df["rating"] > 8.0]
```

### C# LINQ Comparison

```csharp
// C# LINQ
var highRated = movies.Where(m => m.Rating > 8.0).ToList();
```

```python
# Python pandas
high_rated = df[df["rating"] > 8.0]
```

Same result. The pandas version is slightly shorter.

---

## Part 2 -- Multiple Conditions

### AND (both conditions must be True)

Use `&` (not `and`). Each condition must be wrapped in parentheses.

```python
# Movies rated above 8.0 AND from 2019
result = df[(df["rating"] > 8.0) & (df["year"] == 2019)]
print(result)
```

Output:
```
      title  year     genre  rating sentiment
1      1917  2019       War     8.2  positive
2     Joker  2019  Thriller     8.4  positive
3  Parasite  2019     Drama     8.5  positive
```

C# equivalent:
```csharp
var result = movies.Where(m => m.Rating > 8.0 && m.Year == 2019).ToList();
```

### OR (at least one condition must be True)

Use `|` (not `or`). Again, each condition needs parentheses.

```python
# Movies that are Sci-Fi OR have rating above 8.3
result = df[(df["genre"] == "Sci-Fi") | (df["rating"] > 8.3)]
print(result)
```

Output:
```
      title  year     genre  rating sentiment
0      Dune  2021    Sci-Fi     8.0  positive
2     Joker  2019  Thriller     8.4  positive
3  Parasite  2019     Drama     8.5  positive
```

C# equivalent:
```csharp
var result = movies.Where(m => m.Genre == "Sci-Fi" || m.Rating > 8.3).ToList();
```

### NOT -- Invert a Condition

Use `~` (tilde) to flip True/False.

```python
# Movies that are NOT negative sentiment
positive = df[~(df["sentiment"] == "negative")]
# Same as: df[df["sentiment"] != "negative"]
```

### String Filtering with .str

For text columns, pandas provides `.str` helpers:

```python
# Contains a substring (case-insensitive)
result = df[df["title"].str.contains("Jo", case=False)]
# Finds "Joker"

# Starts with
result = df[df["genre"].str.startswith("Sci")]
# Finds "Sci-Fi"

# Exact match (same as ==)
result = df[df["sentiment"].str.lower() == "positive"]
```

C# analogy:
```csharp
// C#
var result = movies.Where(m => m.Title.Contains("Jo")).ToList();

// pandas
result = df[df["title"].str.contains("Jo")]
```

---

## Part 3 -- loc for Filtered Rows + Specific Columns

After filtering, you can select specific columns using `loc`.

```python
# Get just title and rating for high-rated movies
result = df.loc[df["rating"] > 8.0, ["title", "rating"]]
print(result)
```

Output:
```
      title  rating
1      1917     8.2
2     Joker     8.4
3  Parasite     8.5
```

The pattern is:
```
df.loc[ <row condition>, <list of columns> ]
          ^                  ^
          filter rows        select columns
```

C# LINQ equivalent:
```csharp
var result = movies
    .Where(m => m.Rating > 8.0)
    .Select(m => new { m.Title, m.Rating })
    .ToList();
```

---

## Part 4 -- Sorting

### sort_values -- Sort by One Column

```python
# Sort by rating, ascending (lowest first)
sorted_df = df.sort_values("rating")
print(sorted_df[["title", "rating"]])
```

Output:
```
      title  rating
4   Morbius     5.2
0      Dune     8.0
1      1917     8.2
2     Joker     8.4
3  Parasite     8.5
```

```python
# Sort by rating, descending (highest first)
sorted_df = df.sort_values("rating", ascending=False)
```

C# LINQ equivalent:
```csharp
var sorted = movies.OrderBy(m => m.Rating).ToList();        // ascending
var sorted = movies.OrderByDescending(m => m.Rating).ToList(); // descending
```

### Sort by Multiple Columns

```python
# Sort by year ascending, then by rating descending within each year
sorted_df = df.sort_values(
    ["year", "rating"],          # list of column names
    ascending=[True, False]      # True=ascending, False=descending
)
print(sorted_df[["title", "year", "rating"]])
```

Output:
```
      title  year  rating
3  Parasite  2019     8.5
2     Joker  2019     8.4
1      1917  2019     8.2
0      Dune  2021     8.0
4   Morbius  2022     5.2
```

C# LINQ equivalent:
```csharp
var sorted = movies
    .OrderBy(m => m.Year)
    .ThenByDescending(m => m.Rating)
    .ToList();
```

---

## Part 5 -- Groupby and Aggregation

### The Core Idea

`groupby()` splits the DataFrame into groups based on a column's value,
then you apply a function to each group.

```
DataFrame                     After groupby("genre")
---------                     ----------------------
genre    | rating              Group "Drama"     -> [8.5]
Sci-Fi   |  8.0        -->     Group "Sci-Fi"    -> [8.0]
War      |  8.2                Group "War"       -> [8.2]
Thriller |  8.4                Group "Thriller"  -> [8.4]
Drama    |  8.5                Group "Action"    -> [5.2]
Action   |  5.2
```

C# LINQ equivalent:
```csharp
var groups = movies.GroupBy(m => m.Genre);
foreach (var group in groups)
    Console.WriteLine($"{group.Key}: {group.Average(m => m.Rating)}");
```

### Basic groupby + mean

```python
# Average rating per genre
avg_by_genre = df.groupby("genre")["rating"].mean()
print(avg_by_genre)
```

Output:
```
genre
Action      5.2
Drama       8.5
Sci-Fi      8.0
Thriller    8.4
War         8.2
Name: rating, dtype: float64
```

### groupby + agg (multiple aggregations)

```python
# Count and mean rating per genre
result = df.groupby("genre")["rating"].agg(["count", "mean"])
print(result)
```

Output:
```
          count  mean
genre
Action        1   5.2
Drama         1   8.5
Sci-Fi        1   8.0
Thriller      1   8.4
War           1   8.2
```

### groupby + multiple columns

```python
# Group by sentiment, compute count and mean rating
result = df.groupby("sentiment").agg(
    movie_count=("title", "count"),      # count of rows
    avg_rating=("rating", "mean"),       # mean of rating column
    max_rating=("rating", "max")         # max of rating column
)
print(result)
```

Output:
```
           movie_count  avg_rating  max_rating
sentiment
negative             1        5.20        5.20
positive             4        8.275       8.50
```

---

## Part 6 -- value_counts

`value_counts()` counts how many times each unique value appears.
It is the quickest way to understand the distribution of a column.

```python
# How many movies of each genre?
print(df["genre"].value_counts())
```

Output:
```
Action      1
Drama       1
Sci-Fi      1
Thriller    1
War         1
Name: genre, dtype: int64
```

```python
# Show as percentages
print(df["sentiment"].value_counts(normalize=True))
```

Output:
```
positive    0.8
negative    0.2
Name: sentiment, dtype: float64
```

C# LINQ equivalent:
```csharp
var counts = movies
    .GroupBy(m => m.Genre)
    .Select(g => new { Genre = g.Key, Count = g.Count() })
    .OrderByDescending(x => x.Count)
    .ToList();
```

---

## Full LINQ vs pandas Side-by-Side

```
Task                   C# LINQ                                   pandas
---------------------------------------------------------------------------
Filter rows            .Where(x => x.Rating > 8)                df[df["rating"] > 8]
Select columns         .Select(x => new {x.Title,x.Year})       df[["title","year"]]
Sort ascending         .OrderBy(x => x.Rating)                  df.sort_values("rating")
Sort descending        .OrderByDescending(x => x.Rating)        df.sort_values("rating", ascending=False)
Group + count          .GroupBy(x => x.Genre)                   df.groupby("genre").size()
                        .Select(g => new{g.Key, g.Count()})
Group + average        .GroupBy(x => x.Genre)                   df.groupby("genre")["rating"].mean()
                        .Select(g => new{g.Key, g.Avg(...)})
AND condition          .Where(x => x.R > 8 && x.Y == 2019)      df[(df["r"]>8)&(df["y"]==2019)]
OR condition           .Where(x => x.R > 8 || x.G == "Sci")     df[(df["r"]>8)|(df["g"]=="Sci")]
Count per value        .GroupBy(x=>x.Genre).Count()             df["genre"].value_counts()
```

---

## Quiz

**Question 1 (Multiple Choice)**
What does `df[df["rating"] > 8.0]` do?

A) Changes all ratings to 8.0
B) Returns rows where rating is greater than 8.0
C) Deletes rows where rating is less than 8.0
D) Returns the count of rows with rating > 8.0

Answer: B -- it returns a new DataFrame with only matching rows.

---

**Question 2 (Multiple Choice)**
Why do you need parentheses around each condition when using `&`?

A) It is not required, parentheses are optional
B) Python requires parentheses for readability only
C) Operator precedence: `&` binds tighter than `>` and `==`
D) pandas requires it for performance

Answer: C -- without parentheses, Python evaluates `8.0 & df["year"]`
before the comparison, which causes an error. Always wrap each condition.

---

**Question 3 (Short Answer)**
What is the pandas equivalent of this LINQ?
`movies.OrderByDescending(m => m.Rating).Take(3)`

Answer:
```python
df.sort_values("rating", ascending=False).head(3)
```

---

**Question 4 (Multiple Choice)**
What does `df["genre"].value_counts()` return?

A) A DataFrame with genre and count columns
B) The number of unique genres
C) A Series with genre as index and count as values
D) A list of genres

Answer: C -- a Series where the index is each unique value and the
values are how many times each appeared, sorted descending.

---

**Question 5 (Short Answer)**
How do you group by sentiment and count the number of rows per group?

Answer:
```python
df.groupby("sentiment").size()
# OR
df.groupby("sentiment")["title"].count()
```

---

## Key Takeaways

1. Filter rows with `df[condition]` -- same idea as LINQ Where
2. Combine conditions with `&` (AND) and `|` (OR), NOT `and`/`or`
3. Use `.str.contains()`, `.str.startswith()` for text filtering
4. Sort with `sort_values("col")` -- add `ascending=False` to reverse
5. Sort by multiple columns: pass lists to `sort_values`
6. `groupby("col").agg(...)` = LINQ GroupBy + Select
7. `value_counts()` is the fastest way to count unique values

---

**Next Lesson:** [04_missing_data.md](04_missing_data.md)
Learn how to find and handle missing (NaN) values.

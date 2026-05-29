"""
=============================================================================
EXAMPLE 03: Filtering, Sorting, and Grouping
=============================================================================

WHAT THIS FILE TEACHES
-----------------------
- Boolean indexing: filter rows by a condition
- Multiple conditions with & (AND) and | (OR)
- str.contains() for text filtering
- sort_values() for sorting rows
- Sort by multiple columns
- groupby() + agg() for group statistics
- value_counts() for counting unique values

GLOSSARY
---------
Boolean mask  : A Series of True/False values aligned with the DataFrame rows.
                Only rows where the mask is True are kept.
Filtering     : Keeping only rows that match a condition. Like SQL WHERE clause.
&             : AND operator for masks. MUST use & not 'and'.
|             : OR operator for masks. MUST use | not 'or'.
~             : NOT operator. Flips True to False.
sort_values() : Sort rows by one or more column values.
ascending     : True = smallest first (A-Z, 0-9). False = largest first.
groupby()     : Split into groups by column value, then aggregate.
agg()         : Apply aggregate functions (mean, sum, count, etc.) to groups.
value_counts(): Count how many times each unique value appears in a column.

C# ANALOGY
-----------
pandas filtering         --> LINQ .Where(x => condition)
df[df["r"] > 8]          --> movies.Where(m => m.Rating > 8)
df.sort_values("r")      --> movies.OrderBy(m => m.Rating)
df.groupby("g")["r"].mean()--> movies.GroupBy(m=>m.Genre).Select(avg rating)
df["g"].value_counts()   --> movies.GroupBy(m=>m.Genre).Count()

LIBRARIES
----------
pandas   : Data manipulation. pip install pandas
io       : StringIO for embedded sample data. Standard library.
"""

import pandas as pd
from io import StringIO

print("=" * 60)
print("EXAMPLE 03: Filtering, Sorting, and Grouping")
print("=" * 60)

# =============================================================================
# SETUP: Load sample dataset (news headlines with categories)
# =============================================================================

# LLM-relevant sample data: news headlines and articles
csv_text = """title,category,source,rating,year,word_count
Scientists discover life on ancient Mars surface,science,BBC,9.2,2023,450
New transformer model beats GPT-4 on benchmarks,tech,TechCrunch,8.8,2024,620
Stock market hits record high after jobs report,finance,Reuters,7.1,2023,380
Python overtakes JavaScript in developer survey,tech,StackOverflow,8.5,2024,290
Climate summit reaches historic emissions deal,politics,Guardian,8.9,2023,510
Open source LLM released under MIT license,tech,GitHub,9.0,2024,340
Central bank raises interest rates again,finance,Bloomberg,7.4,2023,260
Quantum computer solves problem in minutes,science,Nature,9.5,2024,580
New study links social media to anxiety rise,health,Lancet,7.8,2023,430
Startup raises 100M for AI coding assistant,tech,TechCrunch,8.2,2024,310
Renewable energy now cheapest power source,science,Guardian,9.1,2023,490
Major data breach exposes 50M user accounts,tech,Wired,6.5,2023,340
"""

df = pd.read_csv(StringIO(csv_text))

print(f"\nLoaded {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# =============================================================================
# PART A: Basic Boolean Filtering
# =============================================================================

print("\n" + "=" * 60)
print("PART A: Basic Boolean Filtering")
print("=" * 60)

# Step 1: Create a boolean mask
# df["rating"] > 8.5 checks every row and returns True or False
mask = df["rating"] > 8.5

print("\n--- The boolean mask (True/False per row) ---")
print(mask)

# Step 2: Apply the mask to the DataFrame
high_rated = df[mask]    # keeps only rows where mask is True

print("\n--- High-rated articles (rating > 8.5) ---")
print(high_rated[["title", "rating", "category"]])
print(f"\nFound {len(high_rated)} articles")

# One-liner (most common way to write it):
high_rated_v2 = df[df["rating"] > 8.5]
print(f"\nSame result one-liner: {len(high_rated_v2)} articles")

# C# LINQ equivalent:
# var highRated = articles.Where(a => a.Rating > 8.5).ToList();

# =============================================================================
# PART B: Multiple Conditions (AND / OR)
# =============================================================================

print("\n" + "=" * 60)
print("PART B: Multiple Conditions")
print("=" * 60)

# ---- AND: both conditions must be True ----
# IMPORTANT: wrap each condition in parentheses!
# WRONG:  df[df["rating"] > 8.5 & df["category"] == "tech"]    # error!
# RIGHT:  df[(df["rating"] > 8.5) & (df["category"] == "tech")]

print("--- AND: tech articles with rating > 8.0 ---")
and_result = df[(df["category"] == "tech") & (df["rating"] > 8.0)]
print(and_result[["title", "category", "rating"]])

# C# equivalent:
# articles.Where(a => a.Category == "tech" && a.Rating > 8.0)

# ---- OR: at least one condition must be True ----
print("\n--- OR: science OR rating > 9.0 ---")
or_result = df[(df["category"] == "science") | (df["rating"] > 9.0)]
print(or_result[["title", "category", "rating"]])

# ---- NOT: invert a condition ----
print("\n--- NOT: articles that are NOT tech ---")
not_tech = df[~(df["category"] == "tech")]     # ~ flips True/False
# Same as: df[df["category"] != "tech"]
print(not_tech[["title", "category"]])

# ---- Three conditions ----
print("\n--- THREE conditions: science, rating > 9, year >= 2024 ---")
triple = df[
    (df["category"] == "science") &
    (df["rating"] > 9.0) &
    (df["year"] >= 2024)
]
print(triple[["title", "category", "rating", "year"]])

# =============================================================================
# PART C: Text Filtering with .str
# =============================================================================

print("\n" + "=" * 60)
print("PART C: Text Filtering with .str")
print("=" * 60)

# .str gives you string methods on the whole column at once

# ---- .str.contains() -- find a substring ----
print("--- Titles containing 'AI' or 'LLM' (case-insensitive) ---")
ai_articles = df[df["title"].str.contains("AI|LLM", case=False)]
# case=False = ignore upper/lower case
# "AI|LLM" = regex: AI OR LLM
print(ai_articles[["title", "category"]])

# ---- .str.startswith() ----
print("\n--- Titles starting with 'New' ---")
new_articles = df[df["title"].str.startswith("New")]
print(new_articles[["title"]])

# ---- .str.lower() for consistent comparison ----
print("\n--- Source is 'techcrunch' (case-insensitive match) ---")
tc_articles = df[df["source"].str.lower() == "techcrunch"]
print(tc_articles[["title", "source"]])

# C# equivalent:
# articles.Where(a => a.Title.Contains("AI", StringComparison.OrdinalIgnoreCase))

# =============================================================================
# PART D: Sorting
# =============================================================================

print("\n" + "=" * 60)
print("PART D: Sorting")
print("=" * 60)

# ---- Single column sort ----
print("--- Sort by rating descending (best first) ---")
sorted_desc = df.sort_values("rating", ascending=False)
print(sorted_desc[["title", "rating"]].head(5))

# C# LINQ equivalent:
# articles.OrderByDescending(a => a.Rating).Take(5)

print("\n--- Sort by word_count ascending (shortest first) ---")
sorted_asc = df.sort_values("word_count", ascending=True)
print(sorted_asc[["title", "word_count"]].head(5))

# ---- Multi-column sort ----
print("\n--- Sort by category (A-Z) then rating (highest first) ---")
multi_sort = df.sort_values(
    ["category", "rating"],       # list of columns to sort by
    ascending=[True, False]       # True=ascending for category, False=descending for rating
)
print(multi_sort[["title", "category", "rating"]])

# C# LINQ equivalent:
# articles.OrderBy(a => a.Category).ThenByDescending(a => a.Rating)

# ---- Get top N after filtering and sorting ----
print("\n--- Top 3 tech articles by rating ---")
top_tech = (
    df[df["category"] == "tech"]          # filter to tech only
    .sort_values("rating", ascending=False)  # sort highest first
    .head(3)                               # take top 3
)
print(top_tech[["title", "rating"]])

# =============================================================================
# PART E: Groupby and Aggregation
# =============================================================================

print("\n" + "=" * 60)
print("PART E: Groupby and Aggregation")
print("=" * 60)

print("""
groupby() works in 3 steps:
  1. Split: divide rows into groups based on a column value
  2. Apply: run an aggregation function on each group
  3. Combine: collect the results back into a Series or DataFrame
""")

# ---- Count rows per category ----
print("--- Count of articles per category ---")
count_by_cat = df.groupby("category")["title"].count()
print(count_by_cat)

# C# equivalent:
# articles.GroupBy(a => a.Category).Select(g => new { g.Key, Count = g.Count() })

# ---- Mean rating per category ----
print("\n--- Average rating per category ---")
avg_rating = df.groupby("category")["rating"].mean().round(2)
print(avg_rating)

# ---- Multiple aggregations at once ----
print("\n--- Multiple stats per category ---")
stats = df.groupby("category").agg(
    article_count=("title",      "count"),    # count rows
    avg_rating=   ("rating",     "mean"),     # average rating
    max_rating=   ("rating",     "max"),      # highest rating
    total_words=  ("word_count", "sum")       # sum of word counts
).round(2)
print(stats)

# ---- Group by year ----
print("\n--- Articles per year ---")
by_year = df.groupby("year").agg(
    count=       ("title",  "count"),
    avg_rating=  ("rating", "mean")
).round(2)
print(by_year)

# =============================================================================
# PART F: value_counts
# =============================================================================

print("\n" + "=" * 60)
print("PART F: value_counts")
print("=" * 60)

# value_counts() counts unique values in one column
# The result is a Series sorted by count (most common first)

print("--- How many articles per category? ---")
print(df["category"].value_counts())

print("\n--- As percentages (normalize=True) ---")
print(df["category"].value_counts(normalize=True).round(3))
# 0.333 means 33.3%

print("\n--- How many articles per source? ---")
print(df["source"].value_counts())

print("\n--- How many articles per year? ---")
print(df["year"].value_counts())

# C# equivalent:
# articles.GroupBy(a => a.Category)
#         .Select(g => new {Category = g.Key, Count = g.Count()})
#         .OrderByDescending(x => x.Count)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Filtering and querying a DataFrame:

  Basic filter:
    df[df["col"] > value]

  Multiple conditions (AND):
    df[(df["col1"] > x) & (df["col2"] == y)]

  Multiple conditions (OR):
    df[(df["col1"] > x) | (df["col2"] == y)]

  NOT:
    df[~(df["col"] == value)]

  Text filter:
    df[df["text_col"].str.contains("word", case=False)]

  Sort:
    df.sort_values("col")                          # ascending
    df.sort_values("col", ascending=False)         # descending
    df.sort_values(["col1","col2"], ascending=[True,False])

  Groupby:
    df.groupby("col")["val"].mean()                # single stat
    df.groupby("col").agg(n=("col","count"), avg=("val","mean"))

  Value counts:
    df["col"].value_counts()                       # counts
    df["col"].value_counts(normalize=True)         # percentages

C# LINQ Equivalents:
  df[mask]            = .Where(x => condition)
  sort_values()       = .OrderBy() / .OrderByDescending()
  groupby + count     = .GroupBy().Select(g => g.Count())
  groupby + mean      = .GroupBy().Select(g => g.Average())
  value_counts()      = .GroupBy(x => x).OrderByDescending(g => g.Count())
""")

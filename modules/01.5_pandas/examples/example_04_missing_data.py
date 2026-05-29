"""
=============================================================================
EXAMPLE 04: Detecting and Handling Missing Data
=============================================================================

WHAT THIS FILE TEACHES
-----------------------
- What NaN looks like in a DataFrame
- isnull() and notnull() to detect missing values
- df.isnull().sum() -- the missing-data dashboard
- dropna() -- remove rows with missing values
- dropna(subset=[...]) -- remove only when specific columns are missing
- dropna(thresh=n) -- keep rows with at least n non-null values
- fillna(value) -- replace NaN with a constant
- fillna(mean) -- replace NaN with the column average
- fillna(median) -- replace NaN with the middle value
- fillna(mode) -- replace NaN with the most common value
- ffill and bfill -- forward-fill and backward-fill

GLOSSARY
---------
NaN         : "Not a Number". Represents a missing value. Like null in C#.
isnull()    : Returns True where value is NaN. Like == null in C#.
notnull()   : Returns True where value is NOT NaN. Like != null in C#.
dropna()    : Removes rows (or columns) that contain NaN.
fillna()    : Replaces NaN with a given value or strategy.
mean        : Average of all non-NaN values.
median      : Middle value when all non-NaN values are sorted.
mode        : Most commonly occurring value.
ffill       : Forward-fill: copy previous valid value down.
bfill       : Backward-fill: copy next valid value up.
thresh      : Minimum number of non-NaN values to keep a row.
subset      : Limit dropna/fillna to specific columns only.

C# ANALOGY
-----------
NaN                  --> null (for int?, double?, string)
df["col"].fillna(0)  --> movie.Rating ?? 0
df.dropna()          --> list.Where(m => m.Rating != null)
ffill                --> "carry forward" pattern in time-series code

LIBRARIES
----------
pandas  : Data manipulation. pip install pandas
numpy   : np.nan for creating NaN values. pip install numpy
io      : StringIO for embedded data. Standard library.
"""

import pandas as pd
import numpy as np            # numpy provides np.nan
from io import StringIO

print("=" * 60)
print("EXAMPLE 04: Missing Data")
print("=" * 60)

# =============================================================================
# PART A: What NaN Looks Like
# =============================================================================

print("\n" + "=" * 60)
print("PART A: What NaN Looks Like in a DataFrame")
print("=" * 60)

# We build a DataFrame with intentional missing values
# float("nan") and np.nan are the same thing -- a missing numeric value
# None is Python's null -- pandas stores it as NaN in numeric columns

data = {
    "title":     ["Dune",       "1917",      "Joker",    "Parasite",  "Morbius",  "Inception"],
    "rating":    [8.0,           8.2,         np.nan,     8.5,         5.2,         np.nan],
    "genre":     ["Sci-Fi",     "War",        None,       "Drama",     "Action",   "Sci-Fi"],
    "year":      [2021,          2019,         2019,       2019,        np.nan,      2010],
    "word_count":[450,           np.nan,       380,        np.nan,      260,         520]
}

df = pd.DataFrame(data)

print("DataFrame with missing values:")
print(df)
print()

# NaN shows as "NaN" for numbers, "None" for strings
# (pandas displays both, and both are treated as missing)

# =============================================================================
# PART B: Detecting Missing Values
# =============================================================================

print("\n" + "=" * 60)
print("PART B: Detecting Missing Values")
print("=" * 60)

# ---- isnull(): True where missing ----
print("--- df.isnull()  (True = missing) ---")
print(df.isnull())
print()

# ---- notnull(): True where present ----
print("--- df.notnull()  (True = present) ---")
print(df.notnull())
print()

# ---- Count missing per column ----
# .sum() on a boolean Series counts how many True values there are
print("--- df.isnull().sum()  (count of missing per column) ---")
missing_counts = df.isnull().sum()
print(missing_counts)
print()

# ---- Percentage missing per column ----
print("--- Percentage of missing values per column ---")
missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
print(missing_pct)
print()

# ---- Filter rows that are complete (no NaN) ----
print("--- Rows with NO missing values ---")
complete_rows = df[df.isnull().sum(axis=1) == 0]
# axis=1 means: count NaN across columns (for each row)
print(complete_rows)

# ---- Filter rows that HAVE at least one missing value ----
print("\n--- Rows WITH at least one missing value ---")
incomplete_rows = df[df.isnull().any(axis=1)]
# .any(axis=1) = True if ANY column in the row is NaN
print(incomplete_rows)

# =============================================================================
# PART C: Dropping Missing Values
# =============================================================================

print("\n" + "=" * 60)
print("PART C: Dropping Missing Values (dropna)")
print("=" * 60)

print("""
IMPORTANT: dropna() does NOT modify the original DataFrame.
It returns a NEW DataFrame with rows removed.
To save the result, assign it back: df = df.dropna()
""")

# ---- dropna(): remove any row with at least one NaN ----
print("--- dropna() -- remove ALL rows with any NaN ---")
df_clean = df.dropna()
print(f"Before: {len(df)} rows")
print(f"After:  {len(df_clean)} rows")
print(df_clean)

# ---- dropna(subset=[...]): only remove if SPECIFIC columns are NaN ----
print("\n--- dropna(subset=['rating']) -- only remove if 'rating' is NaN ---")
df_has_rating = df.dropna(subset=["rating"])
print(f"Before: {len(df)} rows")
print(f"After:  {len(df_has_rating)} rows")
print(df_has_rating)
# Rows with missing genre or word_count are KEPT (we only care about rating)

# ---- dropna(thresh=n): keep rows with at least n non-null values ----
print("\n--- dropna(thresh=4) -- keep rows with at least 4 non-NaN values ---")
df_thresh = df.dropna(thresh=4)   # must have at least 4 non-NaN out of 5 columns
print(f"Before: {len(df)} rows")
print(f"After:  {len(df_thresh)} rows")
print(df_thresh)

# =============================================================================
# PART D: Filling Missing Values
# =============================================================================

print("\n" + "=" * 60)
print("PART D: Filling Missing Values (fillna)")
print("=" * 60)

print("""
fillna() replaces NaN with the value you specify.
It also does NOT modify original -- assign the result.
Or use inplace=True to modify in place (less recommended).
""")

# Start with a fresh copy each time to demonstrate each strategy
df_fill = df.copy()     # .copy() makes a brand new independent copy

# ---- Strategy 1: Fill with a constant ----
print("--- Strategy 1: Fill numeric column with 0 ---")
print("Before:")
print(df_fill["rating"])
df_fill["rating"] = df_fill["rating"].fillna(0)
print("After (filled with 0):")
print(df_fill["rating"])

# Reset for next demo
df_fill = df.copy()

# ---- Strategy 2: Fill with mean ----
print("\n--- Strategy 2: Fill 'rating' with mean of existing values ---")
mean_rating = df["rating"].mean()   # computes mean ignoring NaN
print(f"Mean rating (excluding NaN): {mean_rating:.2f}")

df_fill["rating"] = df_fill["rating"].fillna(mean_rating)
print("After (filled with mean):")
print(df_fill["rating"])

# Reset for next demo
df_fill = df.copy()

# ---- Strategy 3: Fill with median ----
print("\n--- Strategy 3: Fill 'word_count' with median ---")
median_wc = df["word_count"].median()   # median ignores NaN
print(f"Median word_count: {median_wc}")

df_fill["word_count"] = df_fill["word_count"].fillna(median_wc)
print("After (filled with median):")
print(df_fill["word_count"])
print()
print("WHY MEDIAN INSTEAD OF MEAN?")
print("  If most articles have 300-500 words but one has 10000 (outlier),")
print("  the mean gets pulled high by that outlier.")
print("  The median stays near 400. Median is more 'typical'.")

# Reset for next demo
df_fill = df.copy()

# ---- Strategy 4: Fill text column with mode (most common value) ----
print("\n--- Strategy 4: Fill 'genre' with mode (most common value) ---")
print("Genres before fillna:")
print(df_fill["genre"])

# .mode() returns a Series of most common values
# [0] takes the first (and usually only) mode value
mode_genre = df["genre"].mode()[0]
print(f"\nMost common genre (mode): {mode_genre}")

df_fill["genre"] = df_fill["genre"].fillna(mode_genre)
print("After (filled with mode):")
print(df_fill["genre"])

# Reset for next demo
df_fill = df.copy()

# ---- Strategy 5: Fill text column with a placeholder string ----
print("\n--- Strategy 5: Fill 'genre' with placeholder 'Unknown' ---")
df_fill["genre"] = df_fill["genre"].fillna("Unknown")
print(df_fill["genre"])

# =============================================================================
# PART E: Forward-Fill and Backward-Fill
# =============================================================================

print("\n" + "=" * 60)
print("PART E: Forward-Fill and Backward-Fill")
print("=" * 60)

print("""
Forward-fill (ffill): Copy the value from the row ABOVE.
Backward-fill (bfill): Copy the value from the row BELOW.

These are useful for time-series data -- e.g. sensor readings
where "no reading" means "same as last reading".
""")

# Create a simple time-series to demonstrate
ts_data = {
    "day":   [1, 2, 3,      4,      5, 6, 7],
    "temp":  [22.0, np.nan, np.nan, 24.0, np.nan, 23.0, np.nan]
}
ts_df = pd.DataFrame(ts_data)

print("Original time-series:")
print(ts_df)

# Forward-fill: carry the last known value forward
ts_ffill = ts_df.copy()
ts_ffill["temp"] = ts_ffill["temp"].ffill()
print("\nAfter ffill (forward-fill):")
print(ts_ffill)
# Day 2 and 3 get 22 (copied from Day 1)
# Day 5 gets 24 (copied from Day 4)
# Day 7 gets 23 (copied from Day 6)

# Backward-fill: pull the next known value backward
ts_bfill = ts_df.copy()
ts_bfill["temp"] = ts_bfill["temp"].bfill()
print("\nAfter bfill (backward-fill):")
print(ts_bfill)
# Day 2 and 3 get 24 (pulled from Day 4)
# Day 5 gets 23 (pulled from Day 6)
# Day 7 stays NaN -- no future value to pull from!

# =============================================================================
# PART F: Comparing Strategies Side by Side
# =============================================================================

print("\n" + "=" * 60)
print("PART F: Strategy Comparison")
print("=" * 60)

# Use just the rating column for comparison
ratings = pd.Series([8.0, np.nan, 8.4, np.nan, 5.2], name="rating")
print("Original ratings:")
print(ratings)
print()

strategies = {
    "fillna(0)":       ratings.fillna(0),
    "fillna(mean)":    ratings.fillna(ratings.mean()),
    "fillna(median)":  ratings.fillna(ratings.median()),
    "ffill":           ratings.ffill(),
    "bfill":           ratings.bfill(),
}

for name, result in strategies.items():
    filled_values = [f"{v:.1f}" for v in result]
    print(f"{name:20s}: {filled_values}")

print()
print("WHICH STRATEGY TO USE?")
print("  Constant (0):   Only if 0 is a valid value in context.")
print("  mean:           Good default for numeric data without outliers.")
print("  median:         Better when there are outliers.")
print("  mode:           Best for text/category columns.")
print("  ffill:          Time-series; 'same as yesterday'.")
print("  bfill:          Time-series; 'same as tomorrow'.")
print("  dropna:         When missing data rows are truly unusable.")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Key missing data commands:

  Detect:
    df.isnull().sum()                     --> count NaN per column
    df.isnull().sum() / len(df) * 100     --> % missing per column
    df[df.isnull().any(axis=1)]           --> rows with any NaN

  Drop:
    df.dropna()                           --> remove rows with any NaN
    df.dropna(subset=["col"])             --> remove only if col is NaN
    df.dropna(thresh=3)                   --> keep rows with >= 3 non-NaN

  Fill:
    df["col"].fillna(0)                   --> constant
    df["col"].fillna(df["col"].mean())    --> mean (avg)
    df["col"].fillna(df["col"].median())  --> median (middle)
    df["col"].fillna(df["col"].mode()[0]) --> mode (most common)
    df["col"].ffill()                     --> forward-fill
    df["col"].bfill()                     --> backward-fill

C# Equivalents:
  isnull()        --> == null
  notnull()       --> != null
  fillna(0)       --> value ?? 0
  dropna()        --> list.Where(x => x.Field != null)
""")

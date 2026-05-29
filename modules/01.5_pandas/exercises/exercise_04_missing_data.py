"""
=============================================================================
EXERCISE 04: Handling Missing Data
=============================================================================

WHAT YOU PRACTICE
------------------
- Detecting NaN with isnull().sum()
- Calculating % missing per column
- Dropping rows with > 50% missing values (using thresh)
- Filling numeric columns with the median
- Filling text columns with "Unknown"
- Verifying no NaN remains after filling

HOW TO USE THIS FILE
---------------------
1. Read each exercise description carefully.
2. Write your code in the TODO section.
3. Run: python exercise_04_missing_data.py
4. Check output.
5. Read HINTS if stuck. Check SOLUTIONS at the bottom.

LIBRARIES NEEDED
-----------------
pandas -- pip install pandas
numpy  -- pip install numpy
io     -- standard library
"""

import pandas as pd
import numpy as np
from io import StringIO

print("=" * 60)
print("EXERCISE 04: Missing Data")
print("=" * 60)

# ---- Setup: LLM model dataset with missing values ----
# This simulates a dataset of AI models where some fields are missing
csv_text = """model_name,organization,release_year,params_billions,context_window_k,license
GPT-4,OpenAI,2023,,128,proprietary
Claude 3 Opus,Anthropic,2024,200,,proprietary
Llama 3 70B,Meta,2024,70,8,open
Gemini Pro,Google,2023,,32,proprietary
Mistral 7B,Mistral AI,2023,7,32,open
Falcon 40B,TII,,40,,open
GPT-3.5,OpenAI,2022,175,16,proprietary
Phi-3 Mini,Microsoft,2024,3.8,128,open
,,,,,
Qwen 72B,Alibaba,2024,72,128,open
"""

df = pd.read_csv(StringIO(csv_text))

print("Raw dataset (AI language models):")
print(df)
print(f"\nShape: {df.shape}")

# =============================================================================
# EXERCISE 1: Detect NaN count per column
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 1: Count missing values per column")
print("-" * 60)
print("""
  a) Print the count of NaN per column using df.isnull().sum()
  b) Print the percentage missing per column (round to 1 decimal place)
     Formula: df.isnull().sum() / len(df) * 100

Expected results:
  model_name        : 1 missing (the blank row)
  organization      : 1 missing
  release_year      : 2 missing
  params_billions   : 3 missing (GPT-4, Gemini Pro, blank row)
  context_window_k  : 3 missing
  license           : 1 missing
""")

# TODO a): Count missing
#

# TODO b): Percentage missing
#


# =============================================================================
# EXERCISE 2: Drop the completely empty row
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 2: Drop rows with too many missing values")
print("-" * 60)
print("""
The dataset has one row that is completely empty (all NaN).
Drop rows that have more than 50% of their values missing.

The dataset has 6 columns. 50% = 3 columns.
A row must have at least 4 non-NaN values to be kept (thresh=4).

  a) Use dropna(thresh=4) to create a cleaned DataFrame called 'df_clean'
  b) Print df_clean
  c) Print the new shape

Expected: the completely empty row should be gone (shape goes from 10 to 9 rows)
""")

# TODO a): Drop rows with too many missing values
# df_clean = ...

# TODO b): Print
#

# TODO c): Shape
#


# =============================================================================
# EXERCISE 3: Fill numeric columns with median
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 3: Fill numeric columns with the median")
print("-" * 60)
print("""
Using df_clean from Exercise 2:
  a) Fill missing values in 'params_billions' with the column median
  b) Fill missing values in 'context_window_k' with the column median
  c) Fill missing values in 'release_year' with the column median

After filling, print df_clean[["model_name","params_billions","context_window_k","release_year"]]

Hint: df["col"].median() computes the median (ignores NaN)
Hint: df["col"] = df["col"].fillna(df["col"].median())
""")

# TODO a): Fill params_billions
#

# TODO b): Fill context_window_k
#

# TODO c): Fill release_year
#

# Print to verify
# print(df_clean[["model_name","params_billions","context_window_k","release_year"]])


# =============================================================================
# EXERCISE 4: Fill text columns with "Unknown"
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 4: Fill text columns with 'Unknown'")
print("-" * 60)
print("""
Using df_clean (after Exercise 3):
  a) Fill missing 'model_name' with "Unknown"
  b) Fill missing 'organization' with "Unknown"
  c) Fill missing 'license' with "Unknown"

Then print df_clean
""")

# TODO a): Fill model_name
#

# TODO b): Fill organization
#

# TODO c): Fill license
#

# Print to verify
# print(df_clean)


# =============================================================================
# EXERCISE 5: Verify no NaN remains
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 5: Verify the data is fully clean")
print("-" * 60)
print("""
After Exercises 2-4, check that no NaN values remain.

  a) Print df_clean.isnull().sum()
  b) If any column still has NaN, go back and fix it
  c) Print the final cleaned DataFrame

Expected: all columns should show 0 missing values.
""")

# TODO a): Check missing
#

# TODO b): Fix if needed
#

# TODO c): Final print
#


# =============================================================================
# EXERCISE 6: Bonus -- Compare fill strategies
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 6: BONUS -- Compare fill strategies for 'params_billions'")
print("-" * 60)
print("""
Using the original df_clean (after Exercise 2, BEFORE filling),
compare four strategies for filling 'params_billions':
  - fillna(0)
  - fillna(mean)
  - fillna(median)
  - ffill (forward-fill)

Print the params_billions column under each strategy.

Which strategy gives the most 'reasonable' result for missing model sizes?
""")

# TODO: For this exercise, reload df_clean from scratch
# Load again from csv_text, apply dropna(thresh=4)
# (so we start fresh before any filling was done)

# df_bonus = pd.read_csv(StringIO(csv_text)).dropna(thresh=4)
# params = df_bonus["params_billions"]

# Print each strategy:
# print("fillna(0):      ", ...)
# print("fillna(mean):   ", ...)
# print("fillna(median): ", ...)
# print("ffill:          ", ...)


print("\n" + "=" * 60)
print("All exercises complete! Check output above.")
print("Read HINTS below if stuck. SOLUTIONS at the bottom.")
print("=" * 60)

# =============================================================================
# HINTS
# =============================================================================
"""
HINTS:

Exercise 1:
  - df.isnull().sum()                          # count
  - (df.isnull().sum() / len(df) * 100).round(1) # percentage

Exercise 2:
  - dropna(thresh=n) keeps rows that have AT LEAST n non-NaN values
  - With 6 columns and 50% threshold: thresh = 6 * 0.5 = 3
    But "at least 50% must be present" means at least ceil(6/2)+1=4
  - df.dropna(thresh=4)

Exercise 3:
  - df["params_billions"].median()  --> computes median, ignoring NaN
  - df["params_billions"] = df["params_billions"].fillna(median_value)
  - You must assign the result back! fillna() does not modify in place.

Exercise 4:
  - df["model_name"] = df["model_name"].fillna("Unknown")
  - Same pattern for all text columns

Exercise 5:
  - df_clean.isnull().sum() should show all zeros after all fills

Exercise 6:
  - params.fillna(0)
  - params.fillna(params.mean())
  - params.fillna(params.median())
  - params.ffill()
  - Median is usually best for model sizes because of outliers
    (e.g. GPT-3.5 at 175B skews the mean upward)
"""

# =============================================================================
# SOLUTIONS (try on your own first!)
# =============================================================================

def show_solutions():
    """Run this function to see the solutions."""
    import pandas as pd
    import numpy as np
    from io import StringIO

    csv_text = """model_name,organization,release_year,params_billions,context_window_k,license
GPT-4,OpenAI,2023,,128,proprietary
Claude 3 Opus,Anthropic,2024,200,,proprietary
Llama 3 70B,Meta,2024,70,8,open
Gemini Pro,Google,2023,,32,proprietary
Mistral 7B,Mistral AI,2023,7,32,open
Falcon 40B,TII,,40,,open
GPT-3.5,OpenAI,2022,175,16,proprietary
Phi-3 Mini,Microsoft,2024,3.8,128,open
,,,,,
Qwen 72B,Alibaba,2024,72,128,open
"""
    df = pd.read_csv(StringIO(csv_text))

    print("\n\n" + "=" * 60)
    print("SOLUTIONS")
    print("=" * 60)

    print("\n--- SOLUTION 1a: Count missing ---")
    print(df.isnull().sum())

    print("\n--- SOLUTION 1b: Percentage missing ---")
    print((df.isnull().sum() / len(df) * 100).round(1))

    print("\n--- SOLUTION 2: Drop rows with >50% missing ---")
    df_clean = df.dropna(thresh=4)    # at least 4 out of 6 columns must be present
    print(df_clean)
    print(f"Shape after dropna: {df_clean.shape}")

    print("\n--- SOLUTION 3: Fill numeric with median ---")
    df_clean = df_clean.copy()       # make a copy to avoid SettingWithCopyWarning
    df_clean["params_billions"]  = df_clean["params_billions"].fillna(
        df_clean["params_billions"].median())
    df_clean["context_window_k"] = df_clean["context_window_k"].fillna(
        df_clean["context_window_k"].median())
    df_clean["release_year"]     = df_clean["release_year"].fillna(
        df_clean["release_year"].median())
    print(df_clean[["model_name","params_billions","context_window_k","release_year"]])

    print("\n--- SOLUTION 4: Fill text with 'Unknown' ---")
    df_clean["model_name"]    = df_clean["model_name"].fillna("Unknown")
    df_clean["organization"]  = df_clean["organization"].fillna("Unknown")
    df_clean["license"]       = df_clean["license"].fillna("Unknown")
    print(df_clean)

    print("\n--- SOLUTION 5: Verify no NaN ---")
    print(df_clean.isnull().sum())
    print("All zeros! Clean data confirmed.")

    print("\n--- SOLUTION 6: Compare strategies for params_billions ---")
    df_bonus = pd.read_csv(StringIO(csv_text)).dropna(thresh=4)
    params   = df_bonus["params_billions"]
    print(f"Original:      {params.tolist()}")
    print(f"fillna(0):     {params.fillna(0).tolist()}")
    print(f"fillna(mean):  {params.fillna(params.mean()).round(1).tolist()}")
    print(f"fillna(median):{params.fillna(params.median()).tolist()}")
    print(f"ffill:         {params.ffill().tolist()}")
    print("\nMedian is usually best here -- less affected by the 200B outlier (Claude 3 Opus)")

# Uncomment the line below to see solutions:
# show_solutions()

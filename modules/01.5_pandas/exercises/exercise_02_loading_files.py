"""
=============================================================================
EXERCISE 02: Loading Files (CSV, JSON, JSONL)
=============================================================================

WHAT YOU PRACTICE
------------------
- Loading CSV from a StringIO (simulated file)
- Loading JSON (list of objects) from a StringIO
- Loading JSONL line-by-line using a manual loop
- Inspecting loaded DataFrames with shape, head, isnull

HOW TO USE THIS FILE
---------------------
1. Read each exercise description carefully.
2. Write your code in the TODO section.
3. Run: python exercise_02_loading_files.py
4. Check your output.
5. If stuck, read HINTS below. Then check SOLUTIONS.

LIBRARIES NEEDED
-----------------
pandas -- pip install pandas
io     -- standard library (no install)
json   -- standard library (no install)
"""

import pandas as pd
from io import StringIO
import json

print("=" * 60)
print("EXERCISE 02: Loading Files")
print("=" * 60)

# =============================================================================
# EXERCISE 1: Load CSV from StringIO
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 1: Load CSV from StringIO")
print("-" * 60)

# This is the raw CSV text you will load
csv_text = """headline,source,category,published_year
Scientists find new species in deep ocean,BBC,science,2023
Python 3.13 released with free-threaded mode,PSF,tech,2024
New transformer beats human performance on MMLU,ArXiv,tech,2024
Record rainfall hits Western Europe in summer,Guardian,climate,2023
Open-source robot learns to walk from scratch,MIT,robotics,2024
"""

print("The CSV text to load:")
print(csv_text)

print("""
Tasks:
  a) Load the CSV text into a DataFrame called 'df_csv' using pd.read_csv and StringIO
  b) Print df_csv
  c) Print the shape
  d) Print the column names (hint: df.columns)
  e) Print the missing value count per column (df.isnull().sum())
""")

# TODO a): Load the CSV
# df_csv = ...

# TODO b): Print it
#

# TODO c): Print shape
#

# TODO d): Print column names
#

# TODO e): Print missing values
#


# =============================================================================
# EXERCISE 2: Load JSON from StringIO
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 2: Load JSON from StringIO")
print("-" * 60)

json_text = """[
    {"title": "GPT-4 Technical Report",      "authors": ["OpenAI"],         "year": 2023, "topic": "LLM"},
    {"title": "Attention Is All You Need",   "authors": ["Vaswani et al"], "year": 2017, "topic": "transformers"},
    {"title": "BERT: Pre-training of Transformers", "authors": ["Devlin et al"], "year": 2018, "topic": "LLM"},
    {"title": "Scaling Laws for Neural LMs", "authors": ["Kaplan et al"],  "year": 2020, "topic": "scaling"},
    {"title": "LLaMA: Open Foundation Models","authors": ["Touvron et al"],"year": 2023, "topic": "LLM"}
]"""

print("The JSON text to load (list of research papers):")
print(json_text[:200] + "...")

print("""
Tasks:
  a) Load the JSON text into a DataFrame called 'df_json' using pd.read_json and StringIO
  b) Print df_json
  c) Print df_json.dtypes
  d) How many papers are about "LLM" topic? Use value_counts() on the 'topic' column
""")

# TODO a): Load the JSON
# df_json = ...

# TODO b): Print it
#

# TODO c): Print dtypes
#

# TODO d): value_counts on topic
#


# =============================================================================
# EXERCISE 3: Load JSONL manually (line by line)
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 3: Load JSONL manually")
print("-" * 60)

# This is a JSONL string -- one JSON object per line, no outer [ ] brackets
jsonl_text = """{"instruction": "Translate to French.", "input": "Hello world", "output": "Bonjour le monde"}
{"instruction": "Translate to French.", "input": "Good morning", "output": "Bonjour"}
{"instruction": "Translate to French.", "input": "Thank you", "output": "Merci"}
{"instruction": "Translate to French.", "input": "How are you?", "output": "Comment allez-vous?"}
{"instruction": "Translate to French.", "input": "I love Python", "output": "J'aime Python"}"""

print("The JSONL text to load (LLM training examples):")
print(jsonl_text)

print("""
Tasks:
  a) Split the JSONL text into individual lines (hint: .strip().split("\\n"))
  b) Loop through each line and parse it with json.loads()
  c) Collect parsed dicts into a list called 'rows'
  d) Create a DataFrame called 'df_jsonl' from that list
  e) Print df_jsonl
  f) Print the number of rows
""")

# TODO a): Split into lines
# lines = ...

# TODO b) and c): Loop and parse
# rows = []
# for line in lines:
#     row = ...
#     ...

# TODO d): Create DataFrame
# df_jsonl = ...

# TODO e): Print
#

# TODO f): Print row count
#


# =============================================================================
# EXERCISE 4: Post-Load Inspection
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 4: Post-Load Inspection Checklist")
print("-" * 60)

# Load a CSV with a DELIBERATE missing value for practice
csv_with_missing = """model_name,organization,release_year,parameter_count_billions
GPT-4,OpenAI,2023,
Claude 3 Opus,Anthropic,2024,200
Llama 3 70B,Meta,2024,70
Gemini Pro,Google,2023,
Mistral 7B,Mistral AI,2023,7
"""

print("CSV with some missing values:")
print(csv_with_missing)

print("""
Tasks:
  a) Load into a DataFrame called 'df_models'
  b) Print the shape
  c) Print df_models.isnull().sum() to see which columns have missing values
  d) Print the percentage missing for each column
     hint: (df.isnull().sum() / len(df) * 100).round(1)
  e) Print only the rows where parameter_count_billions is NOT missing
     hint: df[df["column"].notnull()]
""")

# TODO a): Load
# df_models = ...

# TODO b): Shape
#

# TODO c): Missing count
#

# TODO d): Missing percentage
#

# TODO e): Rows where parameter_count_billions is not null
#


print("\n" + "=" * 60)
print("All exercises complete! Check your output above.")
print("Read HINTS if stuck. Check SOLUTIONS at the bottom.")
print("=" * 60)

# =============================================================================
# HINTS (read these if you are stuck)
# =============================================================================
"""
HINTS:

Exercise 1:
  - StringIO wraps a string so read_csv treats it as a file
  - df = pd.read_csv(StringIO(csv_text))
  - df.columns returns an Index of column names
  - df.isnull().sum() counts NaN per column

Exercise 2:
  - df = pd.read_json(StringIO(json_text))
  - By default read_json expects a list of objects (orient="records" is the default)
  - df["topic"].value_counts() counts how many times each unique topic appears

Exercise 3:
  - Step 1: lines = jsonl_text.strip().split("\\n")
  - Step 2: for line in lines: row = json.loads(line)
  - Step 3: rows.append(row)
  - Step 4: df = pd.DataFrame(rows)
  - json.loads() is for ONE line. json.load() is for a whole file.

Exercise 4:
  - Same as Exercise 1: df = pd.read_csv(StringIO(csv_with_missing))
  - Missing % formula: (df.isnull().sum() / len(df) * 100).round(1)
  - notnull() returns True where a value IS present (not missing)
  - df[df["col"].notnull()] keeps only rows where col has a value
"""

# =============================================================================
# SOLUTIONS (try on your own first!)
# =============================================================================

def show_solutions():
    """Run this function to see the solutions."""
    import pandas as pd
    from io import StringIO
    import json

    print("\n\n" + "=" * 60)
    print("SOLUTIONS")
    print("=" * 60)

    # ---- Solution 1 ----
    csv_text = """headline,source,category,published_year
Scientists find new species in deep ocean,BBC,science,2023
Python 3.13 released with free-threaded mode,PSF,tech,2024
New transformer beats human performance on MMLU,ArXiv,tech,2024
Record rainfall hits Western Europe in summer,Guardian,climate,2023
Open-source robot learns to walk from scratch,MIT,robotics,2024
"""
    print("\n--- SOLUTION 1a/b: Load CSV ---")
    df_csv = pd.read_csv(StringIO(csv_text))    # StringIO wraps the string
    print(df_csv)

    print("\n--- SOLUTION 1c: shape ---")
    print(f"Shape: {df_csv.shape}")

    print("\n--- SOLUTION 1d: column names ---")
    print(df_csv.columns.tolist())

    print("\n--- SOLUTION 1e: missing values ---")
    print(df_csv.isnull().sum())

    # ---- Solution 2 ----
    json_text = """[
        {"title": "GPT-4 Technical Report",      "authors": ["OpenAI"],         "year": 2023, "topic": "LLM"},
        {"title": "Attention Is All You Need",   "authors": ["Vaswani et al"], "year": 2017, "topic": "transformers"},
        {"title": "BERT: Pre-training",          "authors": ["Devlin et al"],  "year": 2018, "topic": "LLM"},
        {"title": "Scaling Laws for Neural LMs", "authors": ["Kaplan et al"],  "year": 2020, "topic": "scaling"},
        {"title": "LLaMA",                       "authors": ["Touvron et al"], "year": 2023, "topic": "LLM"}
    ]"""
    print("\n--- SOLUTION 2a/b: Load JSON ---")
    df_json = pd.read_json(StringIO(json_text))
    print(df_json)

    print("\n--- SOLUTION 2c: dtypes ---")
    print(df_json.dtypes)

    print("\n--- SOLUTION 2d: topic value_counts ---")
    print(df_json["topic"].value_counts())

    # ---- Solution 3 ----
    jsonl_text = """{"instruction": "Translate to French.", "input": "Hello world", "output": "Bonjour le monde"}
{"instruction": "Translate to French.", "input": "Good morning", "output": "Bonjour"}
{"instruction": "Translate to French.", "input": "Thank you", "output": "Merci"}
{"instruction": "Translate to French.", "input": "How are you?", "output": "Comment allez-vous?"}
{"instruction": "Translate to French.", "input": "I love Python", "output": "J'aime Python"}"""

    print("\n--- SOLUTION 3: Load JSONL manually ---")
    lines = jsonl_text.strip().split("\n")    # split string into list of lines
    rows = []
    for line in lines:
        row = json.loads(line)               # parse each line as a dict
        rows.append(row)
    df_jsonl = pd.DataFrame(rows)
    print(df_jsonl)
    print(f"\nRow count: {len(df_jsonl)}")

    # ---- Solution 4 ----
    csv_with_missing = """model_name,organization,release_year,parameter_count_billions
GPT-4,OpenAI,2023,
Claude 3 Opus,Anthropic,2024,200
Llama 3 70B,Meta,2024,70
Gemini Pro,Google,2023,
Mistral 7B,Mistral AI,2023,7
"""
    print("\n--- SOLUTION 4a: Load ---")
    df_models = pd.read_csv(StringIO(csv_with_missing))
    print(df_models)

    print("\n--- SOLUTION 4b: Shape ---")
    print(f"Shape: {df_models.shape}")

    print("\n--- SOLUTION 4c: Missing count ---")
    print(df_models.isnull().sum())

    print("\n--- SOLUTION 4d: Missing percentage ---")
    print((df_models.isnull().sum() / len(df_models) * 100).round(1))

    print("\n--- SOLUTION 4e: Rows where parameter_count_billions is not null ---")
    print(df_models[df_models["parameter_count_billions"].notnull()])

# Uncomment the line below to see solutions:
# show_solutions()

"""
=============================================================================
EXAMPLE 02: Loading Files (CSV, JSON, JSONL)
=============================================================================

WHAT THIS FILE TEACHES
-----------------------
- How to load CSV text into a DataFrame using StringIO
- How to load JSON (list of objects) into a DataFrame
- How to load JSONL (one object per line) into a DataFrame
- How to inspect what was loaded
- No external files needed -- all sample data is embedded here

GLOSSARY
---------
StringIO    : A fake file object built from a string.
              Like C#'s StringReader -- lets read_csv/read_json treat a
              string as if it were a real file.
CSV         : Comma-Separated Values. A plain text table.
JSON        : JavaScript Object Notation. Key-value pairs.
JSONL       : JSON Lines. One JSON object per line. No outer [ ] brackets.
read_csv()  : pandas function to load CSV into a DataFrame.
read_json() : pandas function to load JSON into a DataFrame.
lines=True  : Parameter for read_json() that enables JSONL mode.
orient      : Tells read_json() what shape the JSON has.
              "records" = list of row dicts (most common).

C# ANALOGY
-----------
StringIO in Python  -->  StringReader in C#
pd.read_csv()       -->  CsvHelper.GetRecords<T>()
pd.read_json()      -->  JsonSerializer.Deserialize<List<T>>()
JSONL loop          -->  File.ReadLines() + JsonSerializer per line

LIBRARIES
----------
pandas   : Data manipulation library. Installed with: pip install pandas
io       : Python standard library. StringIO lives here. No install needed.
json     : Python standard library. json.loads() parses a JSON string.
"""

import pandas as pd           # data manipulation
from io import StringIO       # fake file object -- like StringReader in C#
import json                   # for parsing JSONL manually

print("=" * 60)
print("EXAMPLE 02: Loading Files (CSV / JSON / JSONL)")
print("=" * 60)

# =============================================================================
# PART A: Loading CSV
# =============================================================================

print("\n" + "=" * 60)
print("PART A: Loading CSV")
print("=" * 60)

# This is what a CSV file looks like on disk.
# We embed it here as a string so we do not need a real file.
csv_text = """title,review,sentiment,score
Dune,Visually stunning and epic in scale,positive,9
1917,Gripping one-shot war masterpiece,positive,8
Joker,A dark and brilliant performance,positive,8
Parasite,Genius class satire and thriller,positive,9
Morbius,Confusing plot with weak dialogue,negative,3
Inception,Mind-bending and deeply rewarding,positive,8
"""

print("--- The raw CSV string ---")
print(csv_text)

# StringIO wraps the string so read_csv can read it like a file
# In production you would write: pd.read_csv("reviews.csv")
# Here we write: pd.read_csv(StringIO(csv_text))
df_csv = pd.read_csv(StringIO(csv_text))

print("--- Loaded DataFrame ---")
print(df_csv)

print("\n--- shape ---")
print(f"Rows: {df_csv.shape[0]}, Columns: {df_csv.shape[1]}")

print("\n--- dtypes ---")
print(df_csv.dtypes)
# title and review are "object" (pandas word for string)
# sentiment is "object"
# score is int64 (pandas detected it as an integer)

print("\n--- First 3 rows (head) ---")
print(df_csv.head(3))

print("\n--- Missing value check ---")
print(df_csv.isnull().sum())   # count NaN per column; should be all 0 here

# =============================================================================
# PART B: Loading JSON (list of objects)
# =============================================================================

print("\n" + "=" * 60)
print("PART B: Loading JSON (list of objects)")
print("=" * 60)

# This is what a JSON file looks like: an array [ ] of objects { }
json_text = """[
    {"title": "Dune",      "headline": "Sci-fi epic brings the book to life",      "category": "review"},
    {"title": "1917",      "headline": "A technical marvel about courage",          "category": "review"},
    {"title": "Joker",     "headline": "Phoenix delivers career-best performance",  "category": "review"},
    {"title": "Parasite",  "headline": "Bong Joon-ho wins the Palme d Or",         "category": "news"},
    {"title": "Inception", "headline": "Nolan s dream within a dream still amazes","category": "review"}
]"""

print("--- The raw JSON string (first 200 chars) ---")
print(json_text[:200] + "...")

# pd.read_json() reads the JSON array and converts it to a DataFrame
# orient="records" tells pandas: this JSON is a list of row-dictionaries
df_json = pd.read_json(StringIO(json_text))

print("\n--- Loaded DataFrame ---")
print(df_json)

print("\n--- dtypes ---")
print(df_json.dtypes)

print("\n--- describe(include='all') ---")
print(df_json.describe(include="all"))
# unique: how many distinct values per text column
# top: the most common value
# freq: how many times the top value appears

# =============================================================================
# PART C: Loading JSONL (one object per line)
# =============================================================================

print("\n" + "=" * 60)
print("PART C: Loading JSONL (pandas lines=True)")
print("=" * 60)

# JSONL format: no outer [ ] brackets, one JSON object per line.
# This is the most common format for LLM training data.
jsonl_text = """{"title": "Dune",      "instruction": "Summarize this review.", "input": "Stunning sci-fi visuals.", "output": "Dune is a visual masterpiece."}
{"title": "1917",      "instruction": "Summarize this review.", "input": "Gripping war drama.",           "output": "1917 is a gripping war film."}
{"title": "Joker",     "instruction": "Summarize this review.", "input": "Dark character study.",         "output": "Joker is a dark character study."}
{"title": "Parasite",  "instruction": "Summarize this review.", "input": "Brilliant social satire.",      "output": "Parasite is brilliant satire."}
{"title": "Inception", "instruction": "Summarize this review.", "input": "Complex layered storytelling.", "output": "Inception tells a complex story."}"""

print("--- The raw JSONL string ---")
print(jsonl_text)

# Method A: pd.read_json with lines=True
# lines=True tells pandas: each line is its own JSON object
df_jsonl_a = pd.read_json(StringIO(jsonl_text), lines=True)

print("\n--- Loaded with pd.read_json(lines=True) ---")
print(df_jsonl_a)
print(f"\nShape: {df_jsonl_a.shape}")

# =============================================================================
# PART D: Loading JSONL manually (line by line)
# =============================================================================

print("\n" + "=" * 60)
print("PART D: Loading JSONL manually (line-by-line loop)")
print("=" * 60)

print("""
Why use a manual loop instead of read_json(lines=True)?
  - More control: you can filter, transform, or skip lines
  - Works better with very large files (reads one line at a time)
  - Easier to debug when lines have format errors
""")

# Split the JSONL string into individual lines
lines = jsonl_text.strip().split("\n")   # strip() removes trailing whitespace

rows = []                        # empty list to collect row dicts
for i, line in enumerate(lines): # i = line number, line = the string
    row = json.loads(line)       # json.loads() parses ONE JSON string into a dict
    rows.append(row)             # add the dict to our list
    print(f"Line {i}: parsed title = '{row['title']}'")

# Convert list of dicts to DataFrame (same as Method 1 in example_01)
df_jsonl_b = pd.DataFrame(rows)

print("\n--- Loaded DataFrame (manual method) ---")
print(df_jsonl_b)

# Verify both methods give the same result
print(f"\nBoth methods match: {df_jsonl_a.equals(df_jsonl_b)}")

# =============================================================================
# PART E: Post-Load Inspection Checklist
# =============================================================================

print("\n" + "=" * 60)
print("PART E: Post-Load Inspection Checklist")
print("=" * 60)

print("""
After loading ANY file, always run these 3 commands:
""")

# Use the JSONL DataFrame as our example
df = df_jsonl_a

# 1. How much data?
print("1. df.shape  (how much data?)")
print(f"   {df.shape}   --> {df.shape[0]} rows, {df.shape[1]} columns\n")

# 2. What does it look like?
print("2. df.head(3)  (what does it look like?)")
print(df.head(3))
print()

# 3. Any missing values?
print("3. df.isnull().sum()  (any missing values?)")
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("   --> No missing values! Clean data.")
else:
    print(f"   --> WARNING: {missing.sum()} missing values found!")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Loading data into pandas:

  CSV:   df = pd.read_csv(StringIO(csv_text))
         df = pd.read_csv("file.csv")          # real file

  JSON:  df = pd.read_json(StringIO(json_text))
         df = pd.read_json("file.json")        # real file

  JSONL: df = pd.read_json(StringIO(jsonl), lines=True)
         df = pd.read_json("file.jsonl", lines=True)  # real file

  JSONL  for _, row in pd.iterrows():          # manual loop
  manual:    row = json.loads(line)            # parse each line
             rows.append(row)
         df = pd.DataFrame(rows)

Post-load checklist:
  df.shape            --> (rows, cols)
  df.head()           --> peek at data
  df.isnull().sum()   --> count missing values per column

C# Equivalents:
  StringIO            --> StringReader
  read_csv()          --> CsvHelper.GetRecords<T>()
  read_json()         --> JsonSerializer.Deserialize<List<T>>()
  json.loads(line)    --> JsonSerializer.Deserialize<T>(line)
""")

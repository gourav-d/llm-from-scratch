# Lesson 1.5.2: Loading Files

**How to read CSV, JSON, and JSONL files into a DataFrame.**

---

## Glossary (Read This First!)

| Word | Plain-English Meaning | C# Equivalent |
|------|-----------------------|---------------|
| **CSV** | Comma-Separated Values -- a plain text table format | `File.ReadAllLines()` + manual split |
| **JSON** | JavaScript Object Notation -- key-value pairs | `System.Text.Json` / `Newtonsoft.Json` |
| **JSONL** | JSON Lines -- one JSON object per line (no outer array) | Reading a file line-by-line and deserializing each |
| **read_csv()** | pandas function that reads a .csv file into a DataFrame | `SqlDataReader` or CSV parser |
| **read_json()** | pandas function that reads a .json file into a DataFrame | `JsonSerializer.Deserialize<List<T>>()` |
| **StringIO** | A fake file object built from a string (for testing) | `new StringReader(myString)` |
| **sep** | The separator character in CSV (default is comma) | Delimiter parameter in a CSV reader |
| **orient** | Tells `read_json()` what structure the JSON has | N/A (handled by schema in C#) |
| **encoding** | Character encoding of the file (usually "utf-8") | `Encoding.UTF8` |
| **header** | The first row in a CSV that contains column names | Column headers |

---

## Part 1 -- File Formats Compared

Before loading anything, understand the three formats you will use.

### Format Comparison Table

| Feature | CSV | JSON | JSONL |
|---------|-----|------|-------|
| Human readable | Yes | Yes | Yes |
| Each row a line | Yes | No | Yes |
| Supports nested data | No | Yes | Yes |
| Easy to append rows | Yes | Hard | Yes |
| Used in LLM training | Sometimes | Sometimes | Very common |
| pandas function | `read_csv()` | `read_json()` | loop + `read_json(lines=True)` |

### ASCII Diagram: The Same Data in Three Formats

```
DATA: Two movie reviews

CSV format:
-----------
title,review,sentiment
Dune,Visually stunning and epic,positive
Joker,Dark and thought-provoking,positive


JSON format (list of objects):
-------------------------------
[
    {"title": "Dune",  "review": "Visually stunning and epic",  "sentiment": "positive"},
    {"title": "Joker", "review": "Dark and thought-provoking",  "sentiment": "positive"}
]


JSONL format (one object per line, no outer brackets):
------------------------------------------------------
{"title": "Dune",  "review": "Visually stunning and epic",  "sentiment": "positive"}
{"title": "Joker", "review": "Dark and thought-provoking",  "sentiment": "positive"}
```

The JSONL format is the most popular for LLM training data because:
- You can add new rows just by appending a line
- You can read it one row at a time (memory efficient for large datasets)
- Each line is a valid, complete JSON object

---

## Part 2 -- Loading CSV

### Basic read_csv

```python
import pandas as pd

# Load a CSV file from disk
df = pd.read_csv("reviews.csv")

# Check what we loaded
print(df.head())
print(df.shape)
```

C# analogy:
```csharp
// In C# you might do something like:
string[] lines = File.ReadAllLines("reviews.csv");
// Then manually split each line by comma and build objects
// pandas does all of this automatically
```

### Useful Parameters

```python
# sep: change separator (default is comma)
df = pd.read_csv("data.tsv", sep="\t")       # tab-separated

# usecols: only load certain columns (saves memory)
df = pd.read_csv("data.csv", usecols=["title", "rating"])

# nrows: only load first N rows (great for peeking at huge files)
df = pd.read_csv("data.csv", nrows=100)

# encoding: specify character encoding
df = pd.read_csv("data.csv", encoding="utf-8")

# skiprows: skip header rows or comment lines
df = pd.read_csv("data.csv", skiprows=2)

# na_values: treat extra strings as NaN (missing)
df = pd.read_csv("data.csv", na_values=["N/A", "missing", "?"])
```

### Using StringIO (No File Needed for Testing)

`StringIO` lets you treat a Python string as if it were a file.
This is very useful in examples and tests -- no need to create actual files.

```python
import pandas as pd
from io import StringIO   # StringIO is like StringReader in C#

# Pretend this string is a CSV file
csv_text = """title,review,sentiment
Dune,Visually stunning and epic,positive
Joker,Dark and thought-provoking,positive
Inception,Mind-bending masterpiece,positive
Morbius,Confusing and dull,negative"""

# StringIO wraps the string so read_csv can read it like a file
df = pd.read_csv(StringIO(csv_text))
print(df)
```

Output:
```
       title                       review sentiment
0       Dune    Visually stunning and epic  positive
1      Joker  Dark and thought-provoking   positive
2  Inception  Mind-bending masterpiece     positive
3    Morbius           Confusing and dull  negative
```

---

## Part 3 -- Loading JSON

### JSON as a List of Objects (most common)

```python
import pandas as pd
from io import StringIO

json_text = """[
    {"title": "Dune",      "year": 2021, "rating": 8.0},
    {"title": "1917",      "year": 2019, "rating": 8.2},
    {"title": "Parasite",  "year": 2019, "rating": 8.5}
]"""

df = pd.read_json(StringIO(json_text))
print(df)
```

Output:
```
      title  year  rating
0      Dune  2021     8.0
1      1917  2019     8.2
2  Parasite  2019     8.5
```

C# analogy:
```csharp
var movies = JsonSerializer.Deserialize<List<Movie>>(jsonText);
// pandas does the same thing and gives you a DataFrame automatically
```

### orient Parameter

The `orient` parameter tells pandas what shape your JSON is.

```python
# JSON as a dict of lists (column-oriented)
json_col = """{"title": ["Dune","1917"], "year": [2021,2019]}"""
df = pd.read_json(StringIO(json_col), orient="columns")

# JSON as a list of records (row-oriented, default)
json_row = """[{"title":"Dune","year":2021},{"title":"1917","year":2019}]"""
df = pd.read_json(StringIO(json_row), orient="records")
```

---

## Part 4 -- Loading JSONL

JSONL is NOT handled by `read_json()` directly (unless you use `lines=True`).

### Method A: pandas lines=True

```python
import pandas as pd
from io import StringIO

# Each line is a separate JSON object
jsonl_text = """{"title": "Dune",     "review": "Stunning visuals", "sentiment": "positive"}
{"title": "Joker",    "review": "Brilliant acting",  "sentiment": "positive"}
{"title": "Morbius",  "review": "Weak plot",          "sentiment": "negative"}"""

df = pd.read_json(StringIO(jsonl_text), lines=True)
print(df)
```

### Method B: Manual loop (more control)

```python
import pandas as pd
import json

# Manual approach -- useful when you need to preprocess each line
lines = [
    '{"title": "Dune",    "review": "Stunning visuals",  "sentiment": "positive"}',
    '{"title": "Joker",   "review": "Brilliant acting",  "sentiment": "positive"}',
    '{"title": "Morbius", "review": "Weak plot",          "sentiment": "negative"}',
]

rows = []                         # empty list to collect rows
for line in lines:                # loop through each line
    row = json.loads(line)        # parse the JSON string into a dict
    rows.append(row)              # add the dict to our list

df = pd.DataFrame(rows)           # convert list of dicts to DataFrame
print(df)
```

C# analogy:
```csharp
var rows = new List<Dictionary<string, string>>();
foreach (string line in File.ReadAllLines("data.jsonl"))
{
    var row = JsonSerializer.Deserialize<Dictionary<string, string>>(line);
    rows.Add(row);
}
// Then convert to DataTable...
// pandas pd.DataFrame(rows) does all of this in one line
```

### ASCII Diagram: JSONL Loading

```
JSONL file on disk (or StringIO):
----------------------------------
Line 1: {"title": "Dune",    "sentiment": "positive"}
Line 2: {"title": "Joker",   "sentiment": "positive"}
Line 3: {"title": "Morbius", "sentiment": "negative"}

        json.loads() parses each line
                    |
                    v
List of dicts:
[
    {"title": "Dune",    "sentiment": "positive"},
    {"title": "Joker",   "sentiment": "positive"},
    {"title": "Morbius", "sentiment": "negative"},
]
                    |
          pd.DataFrame(rows)
                    |
                    v
DataFrame:
index | title    | sentiment
  0   | Dune     | positive
  1   | Joker    | positive
  2   | Morbius  | negative
```

---

## Part 5 -- After Loading: First Steps

After loading any file format, always do these three things:

```python
# Step 1: Check how much data you have
print(df.shape)               # (rows, columns)

# Step 2: Peek at the first few rows
print(df.head())

# Step 3: Check for missing values
print(df.isnull().sum())      # count missing values per column
```

These three lines will tell you:
- If the file loaded correctly
- If the columns are what you expected
- If there are any missing values to handle

---

## Quiz

**Question 1 (Multiple Choice)**
What function loads a CSV file into a DataFrame?

A) `pd.load_csv()`
B) `pd.read_csv()`
C) `pd.open_csv()`
D) `pd.from_csv()`

Answer: B -- `pd.read_csv()` is the correct function name.

---

**Question 2 (Multiple Choice)**
What is the difference between JSON and JSONL?

A) JSON uses commas, JSONL uses tabs
B) JSON has one object per file, JSONL has one object per line
C) JSON is for numbers, JSONL is for strings
D) They are the same format

Answer: B -- JSONL puts one JSON object on each line, no outer array.

---

**Question 3 (Short Answer)**
You want to load a CSV file but only need columns "review" and "label".
What parameter do you use?

Answer: `pd.read_csv("file.csv", usecols=["review", "label"])`
This avoids loading columns you do not need, which saves memory.

---

**Question 4 (Short Answer)**
What is `StringIO` and why do examples use it?

Answer:
`StringIO` (from `io` module) wraps a string so pandas treats it like a file.
Examples use it so you do not need to create actual files on disk.
In production code, you would just pass the real file path.

---

**Question 5 (Multiple Choice)**
Which parameter makes `read_json()` read a JSONL file?

A) `format="jsonl"`
B) `mode="lines"`
C) `lines=True`
D) `per_line=True`

Answer: C -- `pd.read_json("file.jsonl", lines=True)`

---

## Key Takeaways

1. Use `read_csv()` for CSV files
2. Use `read_json()` for JSON files
3. Use `read_json(lines=True)` or a manual loop for JSONL files
4. Use `StringIO` to test without creating real files
5. After loading: always check `shape`, `head()`, and `isnull().sum()`
6. JSONL is the standard format for LLM training datasets

---

**Next Lesson:** [03_filtering_sorting_grouping.md](03_filtering_sorting_grouping.md)
Learn to filter rows, sort data, and group for analysis.

# Module 1.5: Pandas & Data Handling

**The data toolkit every LLM engineer needs.**

---

## Why This Module Exists

Before you can train or fine-tune an LLM, you need to:
- Load datasets (CSV files, JSON files, JSONL files)
- Clean and filter rows (remove bad data)
- Reshape and group data
- Export data in LLM training formats (Alpaca, ChatML)

All of that is done with **pandas**.

Think of pandas as Excel for Python programmers.
You can load millions of rows, filter them instantly, and export them
to exactly the format a model trainer expects.

---

## What is Pandas?

**Pandas** is a Python library for working with structured data
(tables of rows and columns).

### C# Analogy

| Pandas Concept | C# Equivalent |
|----------------|---------------|
| `DataFrame`    | `DataTable` (System.Data) |
| Column         | `DataColumn` |
| Row            | `DataRow` |
| Index          | Primary key / row number |
| `read_csv()`   | `SqlDataReader` or `File.ReadAllLines()` + parse |
| `groupby()`    | LINQ `GroupBy()` |
| `fillna()`     | `?? defaultValue` (null-coalescing) |
| `to_csv()`     | `StreamWriter` + loop |

In .NET, you might use LINQ on a `List<T>` or query a `DataTable`.
In Python, you do the same thing on a `DataFrame` with much less code.

---

## Prerequisites

- Module 01 Python Basics (completed)
  - You need: variables, lists, dictionaries, for-loops, f-strings, functions
- Module 02 NumPy (helpful but not required for this module)

---

## Learning Objectives

By the end of this module you will be able to:

1. Create a DataFrame from a Python dictionary or list
2. Load CSV, JSON, and JSONL files into DataFrames
3. Filter rows using boolean conditions (like LINQ Where)
4. Sort, group, and aggregate data
5. Detect and handle missing values (NaN)
6. Export data as CSV, JSON, and JSONL
7. Build a simple LLM training dataset pipeline end-to-end

---

## Lessons

### Lesson 1.5.1: DataFrame Basics
**File:** `lessons/01_dataframes_basics.md`
- What is a DataFrame?
- Rows, columns, and the index
- Creating from dicts and lists
- shape, dtypes, head, tail, info, describe

### Lesson 1.5.2: Loading Files
**File:** `lessons/02_loading_files.md`
- read_csv, read_json
- Loading JSONL line-by-line
- File format comparison

### Lesson 1.5.3: Filtering, Sorting, Grouping
**File:** `lessons/03_filtering_sorting_grouping.md`
- Boolean indexing (like LINQ Where)
- loc vs iloc
- sort_values
- groupby + agg
- value_counts

### Lesson 1.5.4: Missing Data
**File:** `lessons/04_missing_data.md`
- What is NaN?
- isnull / notnull
- dropna vs fillna
- Fill strategies

### Lesson 1.5.5: Export for LLM Training
**File:** `lessons/05_export_for_llm.md`
- to_csv, to_json
- Writing JSONL (Alpaca format)
- End-to-end training dataset pipeline

---

## Code Examples

All in the `examples/` folder. Each file is self-contained
(no external files needed -- sample data is embedded using StringIO).

| File | What It Shows |
|------|--------------|
| `example_01_dataframes.py`  | Create, inspect, and access DataFrames |
| `example_02_loading_files.py` | Load CSV / JSON / JSONL from memory |
| `example_03_filtering_sorting.py` | Filter, sort, groupby, value_counts |
| `example_04_missing_data.py` | Detect and handle NaN values |
| `example_05_export_for_llm.py` | Build and export an LLM training dataset |

---

## Exercises

All in the `exercises/` folder. Each has TODO placeholders,
HINTS, and full SOLUTIONS commented at the bottom.

| File | What You Practice |
|------|------------------|
| `exercise_01_dataframes.py` | Create and access DataFrames |
| `exercise_02_loading_files.py` | Load CSV, JSON, JSONL |
| `exercise_03_filtering.py` | Filter, sort, groupby |
| `exercise_04_missing_data.py` | Handle NaN |
| `exercise_05_export_for_llm.py` | Build and export LLM dataset |

---

## How to Run

```bash
# Activate your virtual environment first
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# Install pandas (if not already installed)
pip install pandas

# Run any example
python modules/01.5_pandas/examples/example_01_dataframes.py
```

---

## Estimated Time

3 to 4 hours (split across 2 or 3 study sessions)

---

## Next Module

Module 02: NumPy and Mathematical Foundations

Or, if you have already done Module 02, continue to:
Module 03: Neural Network Basics

---

## Connection to LLM Work

Here is where you will use pandas later in the course:

```
Module 12 (Fine-tuning)  <-- You will build training datasets here
       ^
       |
Module 1.5 (Pandas)  <-- You learn the tools NOW
```

When you reach Module 12, you will take a raw dataset of
questions and answers, clean it, and export it in Alpaca JSONL
format so that the fine-tuning trainer can read it.
Lesson 1.5.5 (Export for LLM) is the direct preparation for that.

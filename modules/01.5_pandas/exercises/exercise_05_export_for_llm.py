"""
=============================================================================
EXERCISE 05: Exporting Data for LLM Training
=============================================================================

WHAT YOU PRACTICE
------------------
- Take a list of QA pairs and build a DataFrame
- Add a system_prompt column with a constant value
- Export as JSONL in ChatML format (role/content messages)
- Reload the exported JSONL file and verify
- Count how many rows were written

HOW TO USE THIS FILE
---------------------
1. Read each exercise description carefully.
2. Write your code in the TODO section.
3. Run: python exercise_05_export_for_llm.py
4. Check your output.
5. Read HINTS if stuck. Check SOLUTIONS at the bottom.

LIBRARIES NEEDED
-----------------
pandas  -- pip install pandas
json    -- standard library
os      -- standard library
"""

import pandas as pd
import json
import os

print("=" * 60)
print("EXERCISE 05: Export for LLM Training")
print("=" * 60)

# ---- Raw data: QA pairs about Python for .NET developers ----
# This is the kind of data you would collect to fine-tune a teaching assistant
qa_pairs = [
    {
        "question": "What is the Python equivalent of C# var?",
        "answer":   "In Python you do not need to declare types at all. Just write x = 10. Python infers the type automatically, similar to var but without even writing the keyword."
    },
    {
        "question": "How do I write a for loop in Python compared to C#?",
        "answer":   "In C# you write: for (int i = 0; i < 5; i++). In Python you write: for i in range(5). Python also supports foreach-style loops: for item in my_list."
    },
    {
        "question": "What is a list comprehension in Python?",
        "answer":   "A list comprehension is a compact way to create a list. It is like LINQ in C#. Example: squares = [x*x for x in range(5)] is the same as numbers.Select(x => x*x).ToList() in LINQ."
    },
    {
        "question": "What is None in Python?",
        "answer":   "None is Python's equivalent of null in C#. You check for it with 'if x is None:' (not '== None'). Use None when a variable has no value."
    },
    {
        "question": "What is a Python dictionary?",
        "answer":   "A Python dictionary is like Dictionary<string, object> in C#. You define it with curly braces: d = {'key': 'value'}. Access values with d['key'] or d.get('key')."
    },
    {
        "question": "What does the with statement do in Python?",
        "answer":   "The 'with' statement is like using() in C#. It automatically closes or releases resources when the block ends. For example: with open('file.txt') as f: reads the file and closes it automatically."
    },
    {
        "question": "How do I handle exceptions in Python?",
        "answer":   "Python uses try/except blocks, which work like try/catch in C#. Example: try: result = 10/0 except ZeroDivisionError as e: print(e). You can also add finally: for cleanup."
    },
    {
        "question": "What is a pandas DataFrame?",
        "answer":   "A pandas DataFrame is a table of data with rows and columns, similar to a DataTable in C# System.Data. You create one with pd.DataFrame(dict) where dict keys are column names."
    },
]

print(f"We have {len(qa_pairs)} QA pairs to process.")
print("Sample question:", qa_pairs[0]["question"])

# =============================================================================
# EXERCISE 1: Build a DataFrame from the QA pairs
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 1: Build a DataFrame from the QA pairs")
print("-" * 60)
print("""
Create a DataFrame called 'df' from the qa_pairs list.
The list is already a list of dicts, so pd.DataFrame() can take it directly.

Then:
  a) Create the DataFrame
  b) Print df.shape -- should be (8, 2)
  c) Print df.head(3)
  d) Print df.columns
""")

# TODO a): Create DataFrame
# df = ...

# TODO b): Shape
#

# TODO c): head
#

# TODO d): columns
#


# =============================================================================
# EXERCISE 2: Add a system_prompt column
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 2: Add a system_prompt column")
print("-" * 60)
print("""
Add a new column called 'system_prompt' to the DataFrame.
Set every row to the same value:
  "You are a helpful assistant teaching Python to experienced .NET developers."

Hint: In pandas you can add a column with a constant like this:
  df["new_col"] = "constant value"

Then print df[["question","system_prompt"]].head(3)
""")

# TODO: Add the column
# df["system_prompt"] = ...

# Print to verify
# print(df[["question","system_prompt"]].head(3))


# =============================================================================
# EXERCISE 3: Export as JSONL in ChatML format
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 3: Export as JSONL in ChatML format")
print("-" * 60)
print("""
Export the DataFrame to a file called "python_for_dotnet.jsonl".
Use ChatML format: each row becomes one JSON object with a "messages" list.

The messages list has 3 items:
  {"role": "system",    "content": row["system_prompt"]}
  {"role": "user",      "content": row["question"]}
  {"role": "assistant", "content": row["answer"]}

Steps:
  1. Open the file for writing
  2. Loop with iterrows()
  3. Build the messages dict for each row
  4. Write it as a JSON string + newline

After writing, print how many lines were written.
""")

output_file = "python_for_dotnet.jsonl"

# TODO: Open file and write JSONL
# with open(output_file, "w", encoding="utf-8") as f:
#     for _, row in df.iterrows():
#         record = {
#             "messages": [
#                 ...
#             ]
#         }
#         f.write(json.dumps(record) + "\n")

# Print confirmation
# print(f"Written ... rows to {output_file}")


# =============================================================================
# EXERCISE 4: Reload and verify
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 4: Reload and verify the JSONL file")
print("-" * 60)
print("""
Reload the JSONL file you wrote and verify it:
  a) Load it back using pd.read_json(output_file, lines=True)
     Call the result 'df_reload'
  b) Print the number of rows
  c) Print the column names
  d) Print the 'messages' field of the first row as pretty JSON
     Hint: print(json.dumps(df_reload.iloc[0]["messages"], indent=2))
  e) Confirm the question in row 0 matches the original qa_pairs[0]["question"]
""")

# TODO a): Reload
# df_reload = ...

# TODO b): Row count
#

# TODO c): Columns
#

# TODO d): First messages field pretty-printed
#

# TODO e): Verify match
#


# =============================================================================
# EXERCISE 5: Cleanup and summary
# =============================================================================

print("\n" + "-" * 60)
print("EXERCISE 5: Count rows and clean up the file")
print("-" * 60)
print("""
  a) Print how many training examples you created
  b) Print the number of unique questions (hint: df["question"].nunique())
  c) Delete the output file with os.remove(output_file)
  d) Print a confirmation message

Hint: nunique() counts unique values (like LINQ Distinct().Count())
""")

# TODO a): Count
#

# TODO b): Unique questions
#

# TODO c): Delete file
# os.remove(output_file)

# TODO d): Confirmation
#


# =============================================================================
# BONUS EXERCISE: Export as Alpaca format instead
# =============================================================================

print("\n" + "-" * 60)
print("BONUS: Export same data as Alpaca format")
print("-" * 60)
print("""
Now export the same qa_pairs as JSONL in Alpaca format instead of ChatML.
Alpaca format:
  {"instruction": "...", "input": "", "output": "..."}

  - instruction = the question
  - input       = "" (empty string -- no extra context needed)
  - output      = the answer

Write to "python_for_dotnet_alpaca.jsonl"
Then reload and print the first row.
Delete the file when done.
""")

alpaca_file = "python_for_dotnet_alpaca.jsonl"

# TODO: Build and export Alpaca format
# alpaca_rows = []
# for pair in qa_pairs:
#     alpaca_rows.append({
#         "instruction": ...,
#         "input":       ...,
#         "output":      ...
#     })
# alpaca_df = pd.DataFrame(alpaca_rows)
# with open(alpaca_file, "w", encoding="utf-8") as f:
#     for _, row in alpaca_df.iterrows():
#         f.write(json.dumps(row.to_dict()) + "\n")
# print(f"Alpaca JSONL written: {len(alpaca_df)} rows")
# Reload and print first row:
# verify = pd.read_json(alpaca_file, lines=True)
# print(verify.iloc[0])
# os.remove(alpaca_file)


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
  - qa_pairs is already a list of dicts, so:
    df = pd.DataFrame(qa_pairs)
  - Each dict key becomes a column name

Exercise 2:
  - Adding a constant column: df["new_col"] = "some value"
  - pandas sets every row to that value automatically

Exercise 3:
  - Open a file: with open("name.jsonl", "w", encoding="utf-8") as f:
  - Loop rows: for _, row in df.iterrows():
  - Build the messages list inside the loop
  - Serialize: json.dumps(record)
  - Write: f.write(json_string + "\\n")

Exercise 4:
  - df_reload = pd.read_json(output_file, lines=True)
  - df_reload.iloc[0]["messages"] gives you the messages list for row 0
  - json.dumps(obj, indent=2) pretty-prints it

Exercise 5:
  - len(df) gives total row count
  - df["question"].nunique() counts distinct values
  - os.remove("filename") deletes the file

Bonus:
  - Alpaca format uses "instruction", "input", "output"
  - Use row.to_dict() to convert a Series to a plain dict for json.dumps()
"""

# =============================================================================
# SOLUTIONS (try on your own first!)
# =============================================================================

def show_solutions():
    """Run this function to see the solutions."""
    import pandas as pd
    import json
    import os

    qa_pairs = [
        {"question": "What is the Python equivalent of C# var?",
         "answer":   "In Python you do not need to declare types at all. Just write x = 10."},
        {"question": "How do I write a for loop in Python compared to C#?",
         "answer":   "In C# you write: for (int i = 0; i < 5; i++). In Python: for i in range(5)."},
        {"question": "What is a list comprehension in Python?",
         "answer":   "A list comprehension is a compact way to create a list. Like LINQ Select in C#."},
        {"question": "What is None in Python?",
         "answer":   "None is Python's equivalent of null in C#. Check with 'if x is None:'."},
        {"question": "What is a Python dictionary?",
         "answer":   "A Python dictionary is like Dictionary<string, object> in C#."},
        {"question": "What does the with statement do in Python?",
         "answer":   "The with statement is like using() in C#. It auto-releases resources."},
        {"question": "How do I handle exceptions in Python?",
         "answer":   "Python uses try/except, like try/catch in C#."},
        {"question": "What is a pandas DataFrame?",
         "answer":   "A DataFrame is a table like DataTable in C#. Create with pd.DataFrame(dict)."},
    ]

    print("\n\n" + "=" * 60)
    print("SOLUTIONS")
    print("=" * 60)

    print("\n--- SOLUTION 1: Create DataFrame ---")
    df = pd.DataFrame(qa_pairs)     # list of dicts -> DataFrame
    print(f"Shape: {df.shape}")
    print(df.head(3))
    print("Columns:", df.columns.tolist())

    print("\n--- SOLUTION 2: Add system_prompt column ---")
    df["system_prompt"] = "You are a helpful assistant teaching Python to experienced .NET developers."
    print(df[["question","system_prompt"]].head(3))

    print("\n--- SOLUTION 3: Export ChatML JSONL ---")
    output_file = "python_for_dotnet.jsonl"
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "messages": [
                    {"role": "system",    "content": row["system_prompt"]},
                    {"role": "user",      "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]}
                ]
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    print(f"Written {count} rows to {output_file}")

    print("\n--- SOLUTION 4: Reload and verify ---")
    df_reload = pd.read_json(output_file, lines=True)
    print(f"Rows reloaded: {len(df_reload)}")
    print(f"Columns: {df_reload.columns.tolist()}")
    print("First row messages:")
    print(json.dumps(df_reload.iloc[0]["messages"], indent=2))
    original_q = qa_pairs[0]["question"]
    reloaded_q = df_reload.iloc[0]["messages"][1]["content"]  # user message
    print(f"\nMatch: {original_q == reloaded_q}")

    print("\n--- SOLUTION 5: Count and cleanup ---")
    print(f"Total examples: {len(df)}")
    print(f"Unique questions: {df['question'].nunique()}")
    os.remove(output_file)
    print(f"Deleted {output_file}")

    print("\n--- BONUS: Alpaca format ---")
    alpaca_file = "python_for_dotnet_alpaca.jsonl"
    alpaca_rows = []
    for pair in qa_pairs:
        alpaca_rows.append({
            "instruction": pair["question"],
            "input":       "",
            "output":      pair["answer"]
        })
    alpaca_df = pd.DataFrame(alpaca_rows)
    with open(alpaca_file, "w", encoding="utf-8") as f:
        for _, row in alpaca_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    print(f"Alpaca JSONL written: {len(alpaca_df)} rows")
    verify = pd.read_json(alpaca_file, lines=True)
    print(verify.iloc[0])
    os.remove(alpaca_file)
    print(f"Deleted {alpaca_file}")

# Uncomment the line below to see solutions:
# show_solutions()

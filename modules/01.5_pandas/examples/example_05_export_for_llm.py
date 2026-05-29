"""
=============================================================================
EXAMPLE 05: Exporting Data for LLM Training
=============================================================================

WHAT THIS FILE TEACHES
-----------------------
- Exporting a DataFrame to CSV with to_csv()
- Exporting a DataFrame to JSON with to_json()
- Writing JSONL using a manual loop (the standard for LLM training)
- Building an Alpaca-format training dataset from raw data
- Building a ChatML-format training dataset from QA pairs
- Reloading the exported file to verify it was written correctly
- End-to-end pipeline: raw data -> clean -> reshape -> export

GLOSSARY
---------
to_csv()       : Write a DataFrame to a CSV file.
to_json()      : Write a DataFrame to a JSON file.
JSONL          : JSON Lines -- one JSON object per line. No outer [ ] array.
Alpaca format  : LLM training format with fields: instruction, input, output.
ChatML format  : LLM training format with a list of messages (role + content).
index=False    : Do not write the row numbers (0,1,2) as a column in the file.
orient=        : Shape of the JSON: "records"=list of row dicts (most common).
lines=True     : Write one JSON object per line (JSONL mode).
iterrows()     : Iterate over DataFrame rows. Gives (index, Series) pairs.
to_dict()      : Convert a pandas Series (row) to a plain Python dict.
json.dumps()   : Serialize a Python dict to a JSON string.

C# ANALOGY
-----------
to_csv()                -->  StreamWriter + foreach loop
to_json(orient=records) -->  JsonSerializer.Serialize(list)
JSONL loop              -->  foreach row: writer.WriteLine(JsonSerializer.Serialize(row))
iterrows()              -->  foreach (DataRow row in table.Rows)
to_dict()               -->  new Dictionary<string,object> from DataRow

CONNECTION TO MODULE 12
------------------------
In Module 12 (Fine-tuning) you will:
  1. Take a dataset like the one built here
  2. Pass it to a training script (e.g. trl, axolotl, or llama_factory)
  3. The trainer reads each JSONL line and teaches the model
This example is DIRECT preparation for that module.

LIBRARIES
----------
pandas   : Data manipulation. pip install pandas
json     : Serialize dicts to JSON strings. Standard library.
io       : StringIO for embedded data. Standard library.
os       : File operations. Standard library.
"""

import pandas as pd
import json
import os
from io import StringIO

print("=" * 60)
print("EXAMPLE 05: Exporting Data for LLM Training")
print("=" * 60)

# =============================================================================
# PART A: Exporting to CSV
# =============================================================================

print("\n" + "=" * 60)
print("PART A: Exporting to CSV")
print("=" * 60)

# Sample movie review data
data = {
    "title":     ["Dune",      "1917",     "Joker",    "Parasite", "Inception"],
    "review":    ["Visually stunning epic", "Gripping war film",
                  "Dark brilliant acting", "Genius satire thriller",
                  "Mind-bending adventure"],
    "sentiment": ["positive", "positive", "positive", "positive", "positive"],
    "score":     [9, 8, 8, 9, 9]
}
df = pd.DataFrame(data)

print("DataFrame to export:")
print(df)

# Write to CSV
output_csv = "movies_example.csv"
df.to_csv(output_csv,
          index=False,         # do NOT write row numbers as a column
          encoding="utf-8")    # utf-8 is standard for most use cases

print(f"\nWritten to: {output_csv}")

# Read back to verify
df_reloaded = pd.read_csv(output_csv)
print("\nReloaded from CSV:")
print(df_reloaded)
print(f"\nShapes match: {df.shape == df_reloaded.shape}")   # should be True

# Clean up
os.remove(output_csv)
print(f"Deleted {output_csv}")

# =============================================================================
# PART B: Exporting to JSON
# =============================================================================

print("\n" + "=" * 60)
print("PART B: Exporting to JSON")
print("=" * 60)

output_json = "movies_example.json"

# orient="records" = list of row dicts (most human-readable)
# indent=2 = pretty-print with 2-space indentation
df.to_json(output_json, orient="records", indent=2)

print(f"Written to: {output_json}")

# Read back and show content
with open(output_json, "r", encoding="utf-8") as f:
    content = f.read()
print("\nFile content (first 400 chars):")
print(content[:400])

# Reload as DataFrame
df_json = pd.read_json(output_json)
print(f"\nReloaded: {len(df_json)} rows, {len(df_json.columns)} columns")

# Clean up
os.remove(output_json)
print(f"Deleted {output_json}")

# =============================================================================
# PART C: Exporting to JSONL (Alpaca Format)
# =============================================================================

print("\n" + "=" * 60)
print("PART C: Exporting to JSONL -- Alpaca Format")
print("=" * 60)

print("""
Alpaca format is the most common format for instruction-tuning LLMs.
Each row has exactly three fields:
  "instruction" : what the model is asked to do
  "input"       : the data to work with (can be empty string)
  "output"      : the expected correct answer

Example:
  {"instruction": "Summarize this review.", "input": "...", "output": "..."}
""")

# Raw review data
raw_reviews = {
    "title":   ["Dune",      "1917",          "Joker",          "Parasite",        "Inception"],
    "review":  ["Visually stunning and epic",  "Gripping one-shot war film",
                "Dark and brilliant",           "Genius social satire",
                "Mind-bending layered story"],
    "stars":   [5, 5, 5, 5, 5]
}
raw_df = pd.DataFrame(raw_reviews)

print("Raw data:")
print(raw_df)

# --- Build Alpaca-format rows ---
alpaca_rows = []

for _, row in raw_df.iterrows():    # iterrows() gives (index, row_as_Series)
    alpaca_row = {
        "instruction": "Write a one-sentence summary of this movie review.",
        "input":       row["review"],               # the review is the input
        "output":      f'{row["title"]} is a must-watch film.'  # expected output
    }
    alpaca_rows.append(alpaca_row)   # collect each row dict

alpaca_df = pd.DataFrame(alpaca_rows)   # convert list of dicts to DataFrame

print("\nAlpaca-format DataFrame:")
print(alpaca_df)

# --- Export as JSONL ---
output_alpaca = "alpaca_training.jsonl"

with open(output_alpaca, "w", encoding="utf-8") as f:
    for _, row in alpaca_df.iterrows():       # iterate over rows
        line_dict = row.to_dict()             # convert Series to plain Python dict
        json_line = json.dumps(line_dict)     # serialize dict to JSON string
        f.write(json_line + "\n")             # write the line + newline character

print(f"\nWritten {len(alpaca_df)} training examples to: {output_alpaca}")

# --- Show the file contents ---
print("\nFile contents:")
with open(output_alpaca, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(f"  Line {i+1}: {line.strip()[:80]}...")

# --- Reload to verify ---
reload_df = pd.read_json(output_alpaca, lines=True)
print(f"\nReloaded: {len(reload_df)} rows")
print("Columns:", list(reload_df.columns))
print("\nFirst row:")
print(reload_df.iloc[0])

# Clean up
os.remove(output_alpaca)
print(f"\nDeleted {output_alpaca}")

# =============================================================================
# PART D: Exporting to JSONL -- ChatML Format
# =============================================================================

print("\n" + "=" * 60)
print("PART D: Exporting to JSONL -- ChatML Format")
print("=" * 60)

print("""
ChatML format is used for chat-based fine-tuning.
Each row has a list of messages with role and content.
Roles: "system", "user", "assistant"

This matches how modern chat models (GPT-4, Claude, Llama 3) are fine-tuned.
""")

# QA pairs about LLM concepts
qa_pairs = [
    {
        "question": "What is tokenization in NLP?",
        "answer":   "Tokenization is the process of splitting text into smaller units called tokens, such as words or subwords, which are then converted to numbers for the model."
    },
    {
        "question": "What is a transformer architecture?",
        "answer":   "A transformer is a neural network architecture based on the attention mechanism. It processes all tokens in parallel instead of sequentially, making it very efficient."
    },
    {
        "question": "What is fine-tuning an LLM?",
        "answer":   "Fine-tuning is the process of continuing to train a pre-trained language model on a smaller, task-specific dataset to specialize its behaviour."
    },
    {
        "question": "What does context window mean?",
        "answer":   "The context window is the maximum number of tokens a model can read and remember at once. For example, GPT-4 has a context window of up to 128,000 tokens."
    },
    {
        "question": "What is the difference between parameters and tokens?",
        "answer":   "Parameters are the learned weights inside the model (e.g. 7 billion). Tokens are the pieces of text the model reads. These are completely different things."
    },
]

system_prompt = "You are a helpful assistant that explains LLM and AI concepts clearly and simply."

# Build ChatML format: each row has a "messages" list
chatml_rows = []

for pair in qa_pairs:
    row = {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]}
        ]
    }
    chatml_rows.append(row)

# Export as JSONL
output_chatml = "chatml_training.jsonl"

with open(output_chatml, "w", encoding="utf-8") as f:
    for row in chatml_rows:
        f.write(json.dumps(row) + "\n")    # serialize the nested dict to JSON

print(f"Written {len(chatml_rows)} chat examples to: {output_chatml}")

# --- Show the file contents ---
print("\nFile contents (each line = one training conversation):")
with open(output_chatml, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # Parse it back to show it nicely
        obj = json.loads(line)
        user_msg = obj["messages"][1]["content"]    # the user turn
        assistant_msg = obj["messages"][2]["content"][:50]  # first 50 chars of answer
        print(f"  Example {i+1}: Q: {user_msg[:50]}... A: {assistant_msg}...")

# Clean up
os.remove(output_chatml)
print(f"\nDeleted {output_chatml}")

# =============================================================================
# PART E: End-to-End Pipeline
# =============================================================================

print("\n" + "=" * 60)
print("PART E: End-to-End Pipeline -- Raw Data to LLM Training File")
print("=" * 60)

print("""
This is the FULL pipeline you will use in Module 12.
Steps:
  1. Load raw data (here embedded as CSV)
  2. Clean: remove missing values, filter bad rows
  3. Build training format (Alpaca)
  4. Export as JSONL
  5. Verify by reloading
""")

# --- Step 1: Raw data ---
csv_raw = """headline,category,source,quality
Scientists discover water on Mars,science,Nature,high
New Python 4.0 released with major speedups,tech,PSF,high
Stock market hits record high,finance,Reuters,high
AI model writes code better than junior devs,tech,ArXiv,high
Local sports team wins championship,sports,ESPN,low
Quantum computer cracks encryption in seconds,tech,Nature,high
New study links coffee to longer life,health,Lancet,high
Celebrity spotted at airport with new partner,celebrity,TMZ,low
Open source LLM beats GPT-4 on coding tasks,tech,HuggingFace,high
Climate change accelerates glacier melting,science,NOAA,high
"""

raw_df = pd.read_csv(StringIO(csv_raw))
print(f"Step 1 - Raw data: {len(raw_df)} rows")
print(raw_df)

# --- Step 2: Clean ---
# Remove rows with any missing values
raw_df = raw_df.dropna()
# Keep only high-quality articles
raw_df = raw_df[raw_df["quality"] == "high"]
# Remove very short headlines
raw_df = raw_df[raw_df["headline"].str.len() > 20]

print(f"\nStep 2 - After cleaning: {len(raw_df)} rows")

# --- Step 3: Build Alpaca format ---
def make_alpaca_row(row):
    """Convert one raw row into an Alpaca training example."""
    return {
        "instruction": "Classify the category of this news headline.",
        "input":       row["headline"],    # the headline text is the input
        "output":      row["category"]     # the category is the expected answer
    }

# Apply to every row
training_rows = [make_alpaca_row(r) for _, r in raw_df.iterrows()]
train_df = pd.DataFrame(training_rows)

print(f"\nStep 3 - Training DataFrame ({len(train_df)} examples):")
print(train_df)

# --- Step 4: Export ---
output_path = "news_headlines_training.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for _, row in train_df.iterrows():
        f.write(json.dumps(row.to_dict()) + "\n")

print(f"\nStep 4 - Written to: {output_path}")

# --- Step 5: Verify ---
verify_df = pd.read_json(output_path, lines=True)
print(f"\nStep 5 - Verification:")
print(f"  Rows written:   {len(train_df)}")
print(f"  Rows reloaded:  {len(verify_df)}")
print(f"  Columns:        {list(verify_df.columns)}")
print(f"  All rows match: {len(train_df) == len(verify_df)}")
print("\nFirst training example:")
print(json.dumps(verify_df.iloc[0].to_dict(), indent=2))

# Clean up
os.remove(output_path)
print(f"\nDeleted {output_path}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Exporting for LLM training:

  CSV:
    df.to_csv("file.csv", index=False)

  JSON (pretty):
    df.to_json("file.json", orient="records", indent=2)

  JSONL (Alpaca / ChatML):
    with open("file.jsonl", "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\\n")

  Alpaca format keys:
    "instruction"  -- what to do
    "input"        -- data to process (can be empty "")
    "output"       -- expected answer

  ChatML format keys:
    "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
    ]

  End-to-end pipeline:
    1. Load raw data (CSV / JSON / JSONL)
    2. Clean (dropna, filter, fillna)
    3. Reshape (build instruction/input/output columns)
    4. Export as JSONL
    5. Verify by reloading

CONNECTION TO MODULE 12:
  The JSONL file you produce here is the direct input to fine-tuning.
  Module 12 will use this exact pipeline with a real dataset.

C# Equivalents:
  iterrows()    --> foreach DataRow in table.Rows
  to_dict()     --> new Dictionary<string,object> from DataRow
  json.dumps()  --> JsonSerializer.Serialize(dict)
  to_csv()      --> StreamWriter line-by-line
""")

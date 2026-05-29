# Lesson 1.5.5: Exporting Data for LLM Training

**How to turn a DataFrame into training data that a model trainer can read.**

---

## Glossary (Read This First!)

| Word | Plain-English Meaning | C# Equivalent |
|------|-----------------------|---------------|
| **to_csv()** | Write a DataFrame to a CSV file on disk | `StreamWriter` + loop |
| **to_json()** | Write a DataFrame to a JSON file on disk | `JsonSerializer.Serialize()` |
| **JSONL** | One JSON object per line. No outer array. | Write lines to StreamWriter |
| **Alpaca format** | A JSONL format with fields: instruction, input, output | A specific schema/DTO |
| **ChatML format** | A JSONL format with a list of messages per row | Another schema/DTO |
| **Fine-tuning** | Teaching a pre-trained model new behaviour using your data | Transfer learning + custom data |
| **instruction** | What you are asking the model to do | The prompt / user request |
| **input** | Optional context or data to go with the instruction | Extra body content |
| **output** | The expected correct answer from the model | The label / target response |
| **system_prompt** | Instructions that set the model's role or behaviour | Global config / system message |
| **index=False** | Tell to_csv() not to write the row numbers as a column | Skip auto-increment ID column |
| **orient="records"** | Tell to_json() to write one object per row | Record-per-row serialization |
| **lines=True** | Tell to_json() to write one line per record (JSONL mode) | Line-by-line output |

---

## Part 1 -- Why This Matters for LLMs

When you fine-tune an LLM in Module 12, the training script will read
a file from disk and train on it row by row.

That file must be in a specific format. The two most common formats are:

### Alpaca Format (instruction tuning)

```json
{"instruction": "Summarize this review.", "input": "The movie was amazing.", "output": "Positive review."}
{"instruction": "Classify sentiment.",    "input": "Boring film.",            "output": "Negative."}
```

### ChatML Format (chat fine-tuning)

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Summarize this review."}, {"role": "assistant", "content": "Positive review."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Classify: Boring film."}, {"role": "assistant", "content": "Negative."}]}
```

Your job as a data engineer: take raw data, clean it with pandas,
and produce one of these files.

---

## Part 2 -- Exporting to CSV

```python
import pandas as pd

data = {
    "title":     ["Dune", "1917", "Joker"],
    "rating":    [8.0, 8.2, 8.4],
    "sentiment": ["positive", "positive", "positive"]
}
df = pd.DataFrame(data)

# Write to CSV
df.to_csv("movies.csv", index=False)
# index=False: do NOT write row numbers (0,1,2) as the first column
# Without index=False you get an ugly unnamed first column

# Read it back to verify
df2 = pd.read_csv("movies.csv")
print(df2)
```

### Useful Parameters for to_csv

```python
df.to_csv("movies.csv",
    index=False,            # no row numbers column
    encoding="utf-8",       # character encoding
    sep=",",                # separator (default is comma)
    columns=["title","rating"]  # only write these columns
)
```

---

## Part 3 -- Exporting to JSON

### List of Records (most human-readable)

```python
# Write as a JSON array of objects
df.to_json("movies.json", orient="records", indent=2)
```

Result in file:
```json
[
  {
    "title": "Dune",
    "rating": 8.0,
    "sentiment": "positive"
  },
  {
    "title": "1917",
    "rating": 8.2,
    "sentiment": "positive"
  }
]
```

C# analogy:
```csharp
string json = JsonSerializer.Serialize(movies, new JsonSerializerOptions { WriteIndented = true });
File.WriteAllText("movies.json", json);
```

---

## Part 4 -- Exporting to JSONL (for LLM training)

### Method A: pandas to_json with lines=True

```python
df.to_json("movies.jsonl", orient="records", lines=True)
```

Result:
```
{"title":"Dune","rating":8.0,"sentiment":"positive"}
{"title":"1917","rating":8.2,"sentiment":"positive"}
{"title":"Joker","rating":8.4,"sentiment":"positive"}
```

### Method B: Manual loop (more control over the structure)

```python
import json

with open("movies.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():         # iterrows() gives (index, Series) pairs
        line = row.to_dict()             # convert row to Python dict
        f.write(json.dumps(line) + "\n") # write as JSON + newline
```

C# analogy:
```csharp
using var writer = new StreamWriter("movies.jsonl");
foreach (var row in rows)
{
    string line = JsonSerializer.Serialize(row);
    writer.WriteLine(line);
}
```

---

## Part 5 -- Building an LLM Training Dataset (Alpaca Format)

This is the end-to-end pipeline you will use in Module 12.

### Step 1: Start with Raw Data

```python
import pandas as pd

# Raw data: movie reviews
raw_data = {
    "title":   ["Dune", "1917",      "Joker",             "Morbius"],
    "review":  ["Stunning visuals", "Gripping war epic", "Dark masterpiece", "Confusing plot"],
    "stars":   [5, 5, 5, 2]
}
raw_df = pd.DataFrame(raw_data)
```

### Step 2: Clean the Data

```python
# Remove rows with missing values
raw_df = raw_df.dropna()

# Keep only positive reviews (for a "good writing" training set)
raw_df = raw_df[raw_df["stars"] >= 4]
```

### Step 3: Build the Alpaca Structure

```python
# Create a new DataFrame in Alpaca format
training_rows = []

for _, row in raw_df.iterrows():
    # Each row becomes one training example
    training_row = {
        "instruction": "Write a one-sentence summary of this movie review.",
        "input":       row["review"],         # the raw review is the input
        "output":      f'{row["title"]} is worth watching.'  # expected output
    }
    training_rows.append(training_row)

train_df = pd.DataFrame(training_rows)
print(train_df)
```

Output:
```
                                     instruction               input                       output
0  Write a one-sentence summary...  Stunning visuals         Dune is worth watching.
1  Write a one-sentence summary...  Gripping war epic        1917 is worth watching.
2  Write a one-sentence summary...  Dark masterpiece         Joker is worth watching.
```

### Step 4: Export as JSONL

```python
import json

with open("training_data.jsonl", "w", encoding="utf-8") as f:
    for _, row in train_df.iterrows():
        line = row.to_dict()
        f.write(json.dumps(line) + "\n")

print(f"Wrote {len(train_df)} training examples to training_data.jsonl")
```

### Step 5: Verify by Reloading

```python
reload_df = pd.read_json("training_data.jsonl", lines=True)
print(f"Reloaded {len(reload_df)} rows")
print(reload_df.head())
```

---

## Part 6 -- Building a ChatML Training Dataset

ChatML format is used for chat-based fine-tuning (e.g. OpenAI format).

```python
import pandas as pd
import json

qa_pairs = [
    ("What is tokenization?",     "Tokenization splits text into smaller pieces called tokens."),
    ("What is a transformer?",    "A transformer is a neural network architecture using attention."),
    ("What is fine-tuning?",      "Fine-tuning trains a pre-trained model on your specific data."),
]

system_prompt = "You are a helpful assistant that teaches LLM concepts clearly."

chatml_rows = []
for question, answer in qa_pairs:
    row = {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer}
        ]
    }
    chatml_rows.append(row)

# Write to JSONL
with open("chatml_training.jsonl", "w", encoding="utf-8") as f:
    for row in chatml_rows:
        f.write(json.dumps(row) + "\n")

print(f"Wrote {len(chatml_rows)} chat examples to chatml_training.jsonl")
```

### ASCII Diagram: End-to-End Pipeline

```
Raw Data (CSV / JSON / list)
         |
         v
  pd.DataFrame(raw_data)
         |
         v
  Clean: dropna(), filter, fillna()
         |
         v
  Reshape: build instruction/input/output fields
         |
         v
  Export: json.dumps() line by line -> .jsonl file
         |
         v
  LLM Training Script (Module 12) reads the .jsonl file
```

---

## Part 7 -- The Complete Pipeline in One Place

```python
import pandas as pd
import json

# --- 1. Raw data ---------------------------------------------------------
raw = [
    {"headline": "Scientists discover water on Mars",     "category": "science"},
    {"headline": "Stock market hits all-time high",        "category": "finance"},
    {"headline": "New Python release improves performance","category": "tech"},
    {"headline": "Local team wins championship",           "category": "sports"},
]
df = pd.DataFrame(raw)

# --- 2. Clean ------------------------------------------------------------
df = df.dropna()
df = df[df["headline"].str.len() > 10]    # remove very short headlines

# --- 3. Build training format (Alpaca) -----------------------------------
def make_training_row(row):
    return {
        "instruction": "Classify the category of this news headline.",
        "input":       row["headline"],
        "output":      row["category"]
    }

train_df = pd.DataFrame([make_training_row(r) for _, r in df.iterrows()])

# --- 4. Export -----------------------------------------------------------
output_path = "news_training.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for _, row in train_df.iterrows():
        f.write(json.dumps(row.to_dict()) + "\n")

print(f"Pipeline complete. {len(train_df)} examples written to {output_path}")

# --- 5. Verify -----------------------------------------------------------
verify = pd.read_json(output_path, lines=True)
print(verify)
```

---

## Quiz

**Question 1 (Multiple Choice)**
What does `index=False` do in `df.to_csv("file.csv", index=False)`?

A) Removes all column headers from the output
B) Prevents writing the row number column (0, 1, 2 ...) to the file
C) Sorts the DataFrame before writing
D) Writes only the index column

Answer: B -- without `index=False` you get an ugly unnamed first column
that is just the row numbers 0, 1, 2 ...

---

**Question 2 (Multiple Choice)**
What format does `df.to_json(orient="records", lines=True)` produce?

A) CSV
B) A JSON array on one line
C) JSONL (one JSON object per line)
D) XML

Answer: C -- this produces JSONL, one object per line.

---

**Question 3 (Short Answer)**
What are the three required fields in Alpaca format?

Answer: `instruction`, `input`, `output`
- instruction: what to do (e.g. "classify sentiment")
- input: the data to process (e.g. the review text)
- output: the expected answer (e.g. "positive")

---

**Question 4 (Short Answer)**
How do you iterate over rows in a DataFrame?

Answer: `for index, row in df.iterrows():`
`row` is a pandas Series. Use `row["column_name"]` to access values.
Or `row.to_dict()` to convert the row to a plain Python dict.

---

**Question 5 (Multiple Choice)**
Why is JSONL preferred over JSON for large training datasets?

A) JSONL files are compressed automatically
B) You can read JSONL one line at a time without loading the whole file
C) JSONL supports more data types
D) JSON cannot store text strings

Answer: B -- JSONL lets you stream the file line by line. A 50GB JSON
array would need to be fully loaded into memory first. JSONL reads one row at a time.

---

## Key Takeaways

1. Use `to_csv(index=False)` to export clean CSV
2. Use `to_json(orient="records")` for pretty JSON
3. Use `to_json(orient="records", lines=True)` or a manual loop for JSONL
4. Alpaca format = {instruction, input, output} -- the most common fine-tuning format
5. ChatML format = {messages: [{role, content}, ...]} -- for chat fine-tuning
6. The pipeline: load raw data -> clean -> reshape -> export as JSONL
7. This pipeline is exactly what you will use in Module 12 (Fine-tuning)

---

**Module Complete!**
Run all 5 example files and complete all 5 exercises before moving on.
Next: Module 02 (NumPy) or Module 03 (Neural Networks) if already done.

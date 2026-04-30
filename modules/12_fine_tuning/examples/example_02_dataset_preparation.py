"""
Module 12 - Fine-Tuning LLMs: Example 02
==========================================
TOPIC: Building and Validating a Fine-Tuning Dataset

=== GLOSSARY ===

JSONL (JSON Lines):
    A file format where each LINE is a separate valid JSON object.
    Used heavily in LLM training because you can read one line at a time
    without loading the whole file into memory.
    C# analogy: Like StreamReader reading one JSON object per line
    instead of parsing one giant array.

Train/Validation/Test Split:
    Dividing your dataset into three non-overlapping parts.
    - Train set   : the model LEARNS from this (e.g., 70%)
    - Val set     : tune hyperparameters, check for overfitting (e.g., 15%)
    - Test set    : final unbiased evaluation, touch only ONCE (e.g., 15%)
    C# analogy: Like separating unit tests / integration tests / acceptance tests.

Data Leakage:
    When information from the test set accidentally "leaks" into training.
    This gives falsely high accuracy - the model has seen the test data.
    C# analogy: Like using production data in your unit tests and then
    claiming your tests passed in production. Misleading!

Instruction-Response Format (Alpaca style):
    A specific text template used to fine-tune instruction-following models.
    Format:
        ### Instruction:
        [what you want the model to do]
        ### Input:
        [optional extra context]
        ### Response:
        [the correct answer]
    Named after the Stanford Alpaca paper that popularized this format.

Tokenization:
    The process of breaking text into smaller pieces (tokens) that a model
    can process numerically.
    C# analogy: Like splitting a string into tokens for a parser,
    but each token becomes an integer ID.

Character-level tokenizer:
    The simplest possible tokenizer: each character becomes one token.
    'H','e','l','l','o' -> [72, 101, 108, 108, 111] (ASCII values).
    Real LLMs use sub-word tokenizers (like BPE) but the principle is the same.

Truncation:
    Cutting a token sequence to a maximum length.
    Models have a fixed context window (max tokens they can process).
    Any text beyond that limit must be cut off.
    C# analogy: Like string.Substring(0, maxLength).

Class distribution:
    How many examples exist per class in the dataset.
    Imbalanced distribution (e.g., 90% BUG, 3% OUTAGE) causes problems
    because the model rarely sees minority classes.
    C# analogy: Like checking if your test suite has coverage for all code paths.

Shuffle:
    Randomly reorder the dataset before splitting.
    Prevents bias from the order in which data was collected.
    C# analogy: Like calling list.OrderBy(_ => Guid.NewGuid()) before splitting.

=== PART A: Classification dataset in JSONL format ===
=== PART B: Instruction-response format, tokenization, truncation ===

Run with: python example_02_dataset_preparation.py
"""

# ============================================================
# IMPORTS
# ============================================================

import json      # json: built-in library for JSON serialization/deserialization
                 # C# analogy: System.Text.Json or Newtonsoft.Json

import os        # os: built-in library for file/directory operations
                 # C# analogy: System.IO namespace

import random    # random: built-in library for random number generation
                 # C# analogy: System.Random

from collections import Counter  # Counter: counts occurrences in a list
                                  # C# analogy: Dictionary<T, int> with auto-increment

# ============================================================
# PART A: CLASSIFICATION DATASET IN JSONL FORMAT
# ============================================================

print("=" * 60)
print("MODULE 12 - DATASET PREPARATION")
print("=" * 60)
print()
print("--- PART A: Classification Dataset (JSONL Format) ---")
print()

# ----------------------------------------------------------
# A.1 - DEFINE RAW SUPPORT TICKET DATA
# ----------------------------------------------------------
# Each record is a Python dict (key-value pairs).
# C# analogy: List<Dictionary<string, string>> or List<TicketRecord>
#
# Categories:
#   BUG      - something is broken
#   FEATURE  - request for new functionality
#   OUTAGE   - system is down / emergency
#   HOW_TO   - user asking how to do something

RAW_TICKETS = [
    # --- BUG examples ---
    {"text": "The login button does not work after entering password", "label": "BUG"},
    {"text": "Getting NullReferenceException when opening dashboard",   "label": "BUG"},
    {"text": "Export to CSV crashes the application",                   "label": "BUG"},
    {"text": "Dashboard shows wrong data for last month",               "label": "BUG"},
    {"text": "Email notifications are not being sent",                  "label": "BUG"},
    {"text": "Filter dropdown resets on page reload",                   "label": "BUG"},
    {"text": "Chart is missing after the latest update",               "label": "BUG"},

    # --- FEATURE examples ---
    {"text": "Please add dark mode to the app",                         "label": "FEATURE"},
    {"text": "Would love to see bulk export functionality",             "label": "FEATURE"},
    {"text": "Can we get keyboard shortcuts for common actions",        "label": "FEATURE"},
    {"text": "Add support for multi-factor authentication",             "label": "FEATURE"},
    {"text": "We need an API endpoint for this report",                 "label": "FEATURE"},
    {"text": "Please allow custom date ranges in the filter",           "label": "FEATURE"},

    # --- OUTAGE examples ---
    {"text": "URGENT: Production system is completely down",            "label": "OUTAGE"},
    {"text": "All users cannot log in - service appears offline",       "label": "OUTAGE"},
    {"text": "Critical: database unreachable, all transactions failing","label": "OUTAGE"},
    {"text": "Website returns 503 error for everyone",                  "label": "OUTAGE"},
    {"text": "EMERGENCY: data pipeline stopped processing",             "label": "OUTAGE"},

    # --- HOW_TO examples ---
    {"text": "How do I export my report as a PDF",                      "label": "HOW_TO"},
    {"text": "What is the shortcut to switch between views",            "label": "HOW_TO"},
    {"text": "How can I share a dashboard with a colleague",            "label": "HOW_TO"},
    {"text": "Where do I update my notification preferences",           "label": "HOW_TO"},
    {"text": "How do I reset my API key",                               "label": "HOW_TO"},
    {"text": "Can you explain what the variance column means",          "label": "HOW_TO"},
]

print(f"Total raw tickets: {len(RAW_TICKETS)}")
print()

# Count how many samples per class using Counter
# C# analogy: tickets.GroupBy(t => t.Label).ToDictionary(g => g.Key, g => g.Count())
label_counts = Counter(ticket["label"] for ticket in RAW_TICKETS)  # generator expression counts labels
print("Class distribution in raw data:")
for label, count in sorted(label_counts.items()):  # sorted() alphabetically by label name
    bar = "#" * count   # simple ASCII bar chart using # characters
    print(f"  {label:8s}: {count:2d}  {bar}")
print()

# ----------------------------------------------------------
# A.2 - SHUFFLE THE DATA BEFORE SPLITTING
# ----------------------------------------------------------
# Always shuffle BEFORE splitting so classes aren't grouped together.
# Without shuffle, all OUTAGE examples might end up only in the test set.

random.seed(99)              # fixed seed for reproducibility
                             # C# analogy: new Random(99)

shuffled = RAW_TICKETS[:]    # make a COPY of the list (so original is unchanged)
                             # C# analogy: var shuffled = new List<Ticket>(rawTickets);
random.shuffle(shuffled)     # shuffle IN PLACE (modifies the list)
                             # C# analogy: shuffled.Sort((_, __) => Random.Next(-1, 2))

print("Data has been shuffled (seed=99 for reproducibility).")
print()

# ----------------------------------------------------------
# A.3 - TRAIN / VAL / TEST SPLIT FUNCTION
# ----------------------------------------------------------
# We write this as a reusable function.
# C# analogy: Like a static utility method SplitDataset<T>(...).

def split_dataset(data, train_ratio=0.70, val_ratio=0.15):
    """
    Split a list of records into train, val, and test sets.

    Parameters:
        data       : list of records (any dicts)
        train_ratio: fraction of data for training (default 70%)
        val_ratio  : fraction for validation (default 15%)
                     test_ratio is inferred as 1 - train - val

    Returns:
        (train_list, val_list, test_list)

    C# analogy:
        static (List<T> train, List<T> val, List<T> test)
        SplitDataset<T>(List<T> data, float trainRatio, float valRatio)
    """
    total = len(data)                           # total number of samples

    train_end = int(total * train_ratio)        # index where training set ends
    val_end   = int(total * (train_ratio + val_ratio))  # index where val set ends

    # Slice the list into three parts using Python slice notation
    # C# analogy: data.GetRange(0, trainEnd)
    train_set = data[:train_end]          # from index 0 up to (not including) train_end
    val_set   = data[train_end:val_end]   # from train_end up to val_end
    test_set  = data[val_end:]            # from val_end to the end

    return train_set, val_set, test_set   # return three separate lists

# Call the function to split our shuffled data
train_data, val_data, test_data = split_dataset(shuffled)  # tuple unpacking (like C# deconstruct)

print("Dataset split results:")
print(f"  Total   : {len(RAW_TICKETS):3d} samples")
print(f"  Train   : {len(train_data):3d} samples  ({len(train_data)/len(RAW_TICKETS)*100:.0f}%)")
print(f"  Val     : {len(val_data):3d} samples  ({len(val_data)/len(RAW_TICKETS)*100:.0f}%)")
print(f"  Test    : {len(test_data):3d} samples  ({len(test_data)/len(RAW_TICKETS)*100:.0f}%)")
print()

# Show class distribution in each split
# This verifies no class accidentally ended up entirely in one split
for split_name, split in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
    counts = Counter(r["label"] for r in split)  # count labels in this split
    dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))  # format as one line
    print(f"  {split_name:5s} class dist: {dist_str}")

print()

# ----------------------------------------------------------
# A.4 - SAVE TO JSONL FORMAT
# ----------------------------------------------------------
# JSONL = one JSON object per line.
# This is the standard format for fine-tuning datasets.

def save_jsonl(records, filepath):
    """
    Save a list of dicts to a .jsonl file (one JSON object per line).
    C# analogy:
        using (var writer = new StreamWriter(filepath)) {
            foreach (var rec in records) writer.WriteLine(JsonSerializer.Serialize(rec));
        }
    """
    with open(filepath, "w", encoding="utf-8") as f:  # open file for writing
        for record in records:                          # iterate over each dict
            line = json.dumps(record)                   # convert dict to JSON string
            f.write(line + "\n")                        # write the line + newline character

def load_jsonl(filepath):
    """
    Load a .jsonl file and return a list of dicts.
    C# analogy:
        File.ReadAllLines(path).Select(line => JsonSerializer.Deserialize<T>(line)).ToList()
    """
    records = []                                        # start with empty list
    with open(filepath, "r", encoding="utf-8") as f:   # open file for reading
        for line in f:                                  # iterate line by line
            line = line.strip()                         # remove trailing newline/whitespace
            if line:                                    # skip empty lines
                records.append(json.loads(line))        # parse JSON string back to dict
    return records                                      # return populated list

# Define output paths
# C# analogy: string outputDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data");
output_dir = os.path.join(os.path.dirname(__file__), "data")  # same folder as this script + /data

# Create the output directory if it doesn't exist
# C# analogy: Directory.CreateDirectory(outputDir);
os.makedirs(output_dir, exist_ok=True)  # exist_ok=True means don't error if it already exists

# Build file paths
train_path = os.path.join(output_dir, "train.jsonl")  # C# analogy: Path.Combine(outputDir, "train.jsonl")
val_path   = os.path.join(output_dir, "val.jsonl")
test_path  = os.path.join(output_dir, "test.jsonl")

# Save all three splits
save_jsonl(train_data, train_path)   # write training set
save_jsonl(val_data,   val_path)     # write validation set
save_jsonl(test_data,  test_path)    # write test set

print(f"JSONL files saved to: {output_dir}")
print(f"  {train_path}")
print(f"  {val_path}")
print(f"  {test_path}")
print()

# ----------------------------------------------------------
# A.5 - RELOAD AND VERIFY
# ----------------------------------------------------------
# Load back from disk to confirm round-trip works correctly.
train_loaded = load_jsonl(train_path)   # reload from file
print("Reload verification: first 3 records from train.jsonl:")
for i, record in enumerate(train_loaded[:3]):  # slice first 3 records
    print(f"  [{i}] {record}")                 # print each dict
print()

# Show raw content of the JSONL file (first 3 lines)
print("Raw JSONL file content (first 3 lines of train.jsonl):")
with open(train_path, "r", encoding="utf-8") as f:  # open for reading
    for i, line in enumerate(f):                    # iterate lines
        if i >= 3:                                  # stop after 3 lines
            break
        print(f"  {line.rstrip()}")                 # print line without trailing newline
print()
print("Notice: each line is a complete, self-contained JSON object.")
print("You can add/remove samples without re-parsing the whole file.")
print()
print("=" * 60)

# ============================================================
# PART B: INSTRUCTION-RESPONSE FORMAT + TOKENIZATION
# ============================================================

print()
print("--- PART B: Instruction-Response Format + Tokenization ---")
print()

# ----------------------------------------------------------
# B.1 - DEFINE THE ALPACA-STYLE PROMPT TEMPLATE
# ----------------------------------------------------------
# The Alpaca prompt template was introduced by Stanford researchers.
# It tells the model:
#   (a) what task to perform (instruction)
#   (b) the input to process
#   (c) where the response should go
#
# During fine-tuning:
#   - The WHOLE text (instruction + input + response) is shown to the model.
#   - The model is trained to predict the RESPONSE part.
#   - At inference, you give instruction+input and ask model to complete.

ALPACA_TEMPLATE = (
    "### Instruction:\n"       # section header for the task description
    "{instruction}\n"          # {instruction} is a placeholder (like C# string.Format)
    "\n"                       # blank line separator
    "### Input:\n"             # section header for the user input
    "{input}\n"                # {input} placeholder
    "\n"
    "### Response:\n"          # section header for the expected answer
    "{response}"               # {response} placeholder (no newline at end)
)

def apply_template(instruction, input_text, response):
    """
    Fill in the Alpaca template with actual values.
    C# analogy: string.Format(ALPACA_TEMPLATE, instruction, inputText, response)
    or string interpolation: $"### Instruction:\n{instruction}\n..."
    """
    return ALPACA_TEMPLATE.format(   # .format() fills in the {} placeholders
        instruction=instruction,
        input=input_text,
        response=response,
    )

# Convert our classification dataset to instruction format
INSTRUCTION = "Classify the following support ticket into one of: BUG, FEATURE, OUTAGE, HOW_TO"

# Build instruction-format examples from our training data
# C# analogy: trainData.Select(r => ApplyTemplate(INSTRUCTION, r["text"], r["label"])).ToList()
instruction_examples = []                           # start with empty list
for record in train_data:                           # iterate over training records
    formatted = apply_template(                     # fill in the template
        instruction=INSTRUCTION,
        input_text=record["text"],                  # ticket text goes in {input}
        response=record["label"],                   # label goes in {response}
    )
    instruction_examples.append(formatted)          # add to list

print(f"Converted {len(instruction_examples)} training records to instruction format.")
print()
print("Example of one formatted record:")
print("-" * 40)
print(instruction_examples[0])                      # print the first formatted example
print("-" * 40)
print()

# ----------------------------------------------------------
# B.2 - CHARACTER-LEVEL TOKENIZER
# ----------------------------------------------------------
# A tokenizer converts text to a list of integer IDs.
# We build the simplest possible version: one integer per character.
#
# Real LLMs use BPE (Byte Pair Encoding) or SentencePiece,
# but character-level tokenization shows the same CONCEPT.
#
# C# analogy:
#   int[] Tokenize(string text) => text.Select(c => (int)c).ToArray();

def build_vocabulary(texts):
    """
    Scan all texts and collect every unique character.
    Returns:
        char_to_id : dict mapping character -> integer ID
        id_to_char : dict mapping integer ID -> character
    C# analogy:
        Dictionary<char, int> charToId = ...;
        Dictionary<int, char> idToChar = ...;
    """
    all_chars = set()                               # set auto-deduplicates
    for text in texts:                              # loop over all text samples
        for ch in text:                             # loop over each character
            all_chars.add(ch)                       # add to set (duplicates ignored)

    # Sort for consistent ordering (deterministic vocabulary)
    sorted_chars = sorted(all_chars)               # sort alphabetically

    # Build forward mapping: char -> int
    # enumerate() gives (index, value) pairs, like C# Select((ch, i) => ...)
    char_to_id = {ch: idx for idx, ch in enumerate(sorted_chars)}

    # Build reverse mapping: int -> char
    id_to_char = {idx: ch for idx, ch in enumerate(sorted_chars)}

    return char_to_id, id_to_char                  # return both dicts

def tokenize(text, char_to_id):
    """
    Convert a text string to a list of integer token IDs.
    C# analogy: text.Select(c => charToId[c]).ToList()
    Unknown characters are skipped (replaced with nothing).
    """
    return [char_to_id[ch] for ch in text if ch in char_to_id]
    # list comprehension: build a list of IDs for each char that exists in vocab

def detokenize(token_ids, id_to_char):
    """
    Convert a list of token IDs back to a text string.
    C# analogy: string.Concat(tokenIds.Select(id => idToChar[id]))
    """
    return "".join(id_to_char.get(tid, "?") for tid in token_ids)
    # .get(key, default) is like dict.GetValueOrDefault(key, "?") in C#
    # "".join() concatenates a list of chars with no separator

# Build vocabulary from all instruction examples
char_to_id, id_to_char = build_vocabulary(instruction_examples)

vocab_size = len(char_to_id)   # number of unique characters in our data
print(f"Vocabulary size: {vocab_size} unique characters")
print()

# Show a slice of the vocabulary
print("Sample vocabulary entries (char -> ID):")
sample_chars = list(char_to_id.items())[:10]   # first 10 entries
for ch, idx in sample_chars:
    display = repr(ch)    # repr() shows special chars like '\n' as literal text
    print(f"  {display:6s} -> {idx}")
print("  ...")
print()

# Tokenize the first example
first_text = instruction_examples[0]                     # get first formatted text
tokens = tokenize(first_text, char_to_id)                # convert to token IDs
print(f"Original text length   : {len(first_text)} characters")
print(f"Tokenized length       : {len(tokens)} tokens (same for character-level)")
print(f"First 20 token IDs     : {tokens[:20]}")
print()

# Verify round-trip: tokenize then detokenize should give original text back
reconstructed = detokenize(tokens, id_to_char)           # convert IDs back to text
match = reconstructed == first_text                      # check equality
print(f"Round-trip check (tokenize -> detokenize = original): {match}")
print()

# ----------------------------------------------------------
# B.3 - MAX LENGTH TRUNCATION
# ----------------------------------------------------------
# Real models have a maximum context length (e.g., GPT-2 = 1024 tokens).
# Any sequence longer than MAX_LEN must be truncated (cut off).
# This is a critical step - if you skip it, training will crash.

MAX_LEN = 80   # small value for demonstration (real LLMs use 512-4096+)

def truncate_tokens(token_ids, max_len):
    """
    Cut the token sequence to max_len tokens.
    Returns the truncated list.
    C# analogy: tokenIds.Take(maxLen).ToList()
    """
    return token_ids[:max_len]   # Python list slicing: take first max_len elements

print(f"Applying max-length truncation (MAX_LEN = {MAX_LEN}):")
print()
for i, text in enumerate(instruction_examples[:3]):   # show first 3 examples
    tokens_full = tokenize(text, char_to_id)           # full token list
    tokens_trunc = truncate_tokens(tokens_full, MAX_LEN)  # truncated token list
    was_truncated = len(tokens_full) > MAX_LEN         # boolean flag
    status = "TRUNCATED" if was_truncated else "OK"
    print(f"  Example {i}: {len(tokens_full):4d} tokens -> {len(tokens_trunc):4d} tokens [{status}]")

print()
print(f"NOTE: With MAX_LEN={MAX_LEN}, most examples are truncated.")
print("In real fine-tuning you use larger MAX_LEN (512, 1024, 2048...).")
print()
print("=" * 60)

# ----------------------------------------------------------
# B.4 - DATA LEAKAGE DEMO
# ----------------------------------------------------------
# This section shows WHAT data leakage looks like and HOW to prevent it.

print()
print("--- B.4: Data Leakage Demo ---")
print()

# ---- WHAT LEAKAGE LOOKS LIKE ----
# Imagine we have a small dataset and we accidentally put the SAME
# samples in both training and test sets.

all_samples = [
    "Login crashes on submit",         # sample A
    "Add dark mode please",            # sample B
    "System is down for all users",    # sample C
    "How do I export data",            # sample D
    "Null pointer in payment service", # sample E
]

print("WRONG WAY (data leakage):")
print("-" * 40)

# BAD: slicing AFTER shuffling doesn't prevent leakage if done incorrectly.
# Here we demonstrate putting the same sample in both train and test.
leaky_train = all_samples[:4]   # first 4 samples -> train
leaky_test  = all_samples[2:]   # last 3 samples -> test  (overlaps with train!)

# Check for overlap
# set() creates an unordered unique collection (like HashSet<string> in C#)
overlap = set(leaky_train) & set(leaky_test)   # & = intersection (items in BOTH sets)

print(f"  Train samples : {leaky_train}")
print(f"  Test samples  : {leaky_test}")
print(f"  OVERLAP found : {overlap}")  # sample C and D appear in both!
print()
print("  Problem: Model was trained ON these test samples.")
print("  Test accuracy will be ARTIFICIALLY HIGH (model just memorized them).")
print()

print("CORRECT WAY (no leakage):")
print("-" * 40)

# GOOD: shuffle FIRST, then split with strict indices and NO overlap.
all_samples_copy = all_samples[:]    # copy the list
random.seed(42)                      # set seed for reproducibility
random.shuffle(all_samples_copy)     # shuffle randomly

split_idx = 3                        # train gets first 3, test gets remaining 2
clean_train = all_samples_copy[:split_idx]   # non-overlapping slice
clean_test  = all_samples_copy[split_idx:]   # non-overlapping slice

# Check overlap
clean_overlap = set(clean_train) & set(clean_test)   # intersection

print(f"  Shuffled order: {all_samples_copy}")
print(f"  Train samples : {clean_train}")
print(f"  Test samples  : {clean_test}")
print(f"  Overlap       : {clean_overlap if clean_overlap else 'NONE (correct!)'}")
print()
print("  No overlap = honest evaluation. Model has never seen test samples.")
print()

# More subtle leakage: time-based data
# Example: if tickets from Jan-Oct are train, and Nov-Dec are test,
# leakage occurs if you normalize/scale using stats from ALL months.
print("SUBTLE LEAKAGE (normalization leak):")
print("-" * 40)

# Simulate ticket lengths from two time periods
all_lengths = [45, 78, 23, 90, 34, 55, 110, 67, 29, 88]  # all ticket lengths
train_lengths = all_lengths[:7]   # first 7 = training period
test_lengths  = all_lengths[7:]   # last 3 = test period

# BAD: computing mean/std from ALL data including test
bad_mean = sum(all_lengths) / len(all_lengths)     # uses test data stats!
bad_std  = (sum((x - bad_mean)**2 for x in all_lengths) / len(all_lengths)) ** 0.5

print(f"  All lengths  : {all_lengths}")

# GOOD: compute mean/std from TRAINING data only, apply to test
good_mean = sum(train_lengths) / len(train_lengths)   # only from training!
good_std  = (sum((x - good_mean)**2 for x in train_lengths) / len(train_lengths)) ** 0.5

print(f"  WRONG normalization (using all data)  : mean={bad_mean:.1f}, std={bad_std:.1f}")
print(f"  CORRECT normalization (train data only): mean={good_mean:.1f}, std={good_std:.1f}")
print()
print("  Rule: compute ALL statistics from TRAINING data.")
print("  Apply those same statistics to val/test without re-computing.")
print()

# ----------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------

print("=" * 60)
print("SUMMARY - Fine-Tuning Dataset Preparation Checklist")
print("=" * 60)
print()
print("  [1] Collect labeled examples for your task")
print("  [2] Check class distribution - avoid severe imbalance")
print("  [3] Shuffle the dataset with a fixed random seed")
print("  [4] Split into train / val / test (no overlap!)")
print("  [5] Save each split as a JSONL file")
print("  [6] Format as instruction-response pairs if needed")
print("  [7] Tokenize using your chosen tokenizer")
print("  [8] Apply max-length truncation")
print("  [9] Compute normalization stats from TRAIN ONLY")
print(" [10] Never touch test set until final evaluation")
print()
print("C#/.NET analogy:")
print("  JSONL file     = StreamReader reading rows from a log file")
print("  Train/val/test = unit tests / integration tests / acceptance tests")
print("  Data leakage   = using production data in your unit tests")
print("  Tokenization   = parsing text to a token stream (like a lexer)")
print()
print("Done! Run example_03_lora_from_scratch.py next.")

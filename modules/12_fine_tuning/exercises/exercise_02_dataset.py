"""
Module 12 - Fine-Tuning LLMs
Exercise 02: Dataset Preparation

GLOSSARY
--------
JSONL           : JSON Lines format -- one JSON object per line in a text file.
                  Like a CSV where each row is a full JSON object instead of
                  comma-separated values. Easy to stream line by line.
Prompt Template : A fill-in-the-blank string used to format training examples.
                  Like a C# string.Format() or $"..." interpolation, but for LLMs.
Alpaca Template : A popular prompt format used to fine-tune instruction LLMs.
                  Named after the Stanford Alpaca project.
Train/Val/Test  : Three splits of a dataset.
                  Train  = data the model learns from.
                  Val    = data to tune hyperparameters and detect overfitting.
                  Test   = data held out until final evaluation (never peeked at).
Shuffle         : Randomise the order of examples before splitting.
                  Prevents order bias (e.g., if data is sorted by label).
Label           : The correct answer for a training example.
                  Like the "expected" value in a unit test assertion.
Classification  : Assigning one of N fixed category labels to an input.
                  Like an if/else chain: "does this belong to bucket A, B, or C?"
"""

import json     # Python's built-in JSON library (like Newtonsoft.Json in C#)
import random   # Built-in random module for shuffling
import os       # Built-in OS module for file path operations

print("=" * 60)
print("Exercise 02: Dataset Preparation")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Building a Labelled Classification Dataset
#
#  Background:
#    Fine-tuning needs labelled examples. For a support ticket classifier,
#    we need examples tagged as BUG, FEATURE, or HOW_TO.
#    Each example is a dict: {"text": "...", "label": "..."}
#
#  Your Task:
#    Write: build_classification_dataset() -> list[dict]
#    Returns exactly 30 examples:
#      - 10 labelled "BUG"
#      - 10 labelled "FEATURE"
#      - 10 labelled "HOW_TO"
#    Each dict must have keys "text" and "label".
#    Use realistic support ticket text (not just "bug 1", "bug 2", etc.).
#
#  C# Analogy:
#    Like creating a List<TicketDto> with hardcoded test data for unit tests.
#    Each TicketDto has a Text property and a Label property.
# ============================================================

print("-" * 50)
print("EXERCISE 1: Build a Labelled Dataset")
print("-" * 50)
print()

def build_classification_dataset():
    """
    Create a list of 30 labelled support ticket examples.
    10 each of BUG, FEATURE, and HOW_TO.

    Returns:
        list[dict]: Each dict has "text" (str) and "label" (str).
    """
    # TODO: Complete this function.
    # Create 10 BUG examples, 10 FEATURE examples, 10 HOW_TO examples.
    # Each is a dict: {"text": "...", "label": "BUG"} etc.
    # Use realistic text (e.g. "App crashes when I click save button")
    pass  # Replace with your implementation


# Test your function
dataset = build_classification_dataset()
if dataset:
    print(f"  Total examples  : {len(dataset)}")
    # Count per label
    from collections import Counter          # Counter: like LINQ GroupBy().Count() in C#
    label_counts = Counter(item["label"] for item in dataset)  # Count each label
    for label, count in sorted(label_counts.items()):
        print(f"  {label:<12}: {count} examples")
    print()
    print("  First 3 examples:")
    for item in dataset[:3]:                 # Slice: like LINQ .Take(3)
        print(f"    [{item['label']:>7}] {item['text']}")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Save and Load JSONL Files
#
#  Background:
#    JSONL is the standard format for LLM training datasets.
#    Each line is a complete JSON object (one training example).
#    Example file content:
#      {"text": "app crashes", "label": "BUG"}
#      {"text": "add dark mode", "label": "FEATURE"}
#      ...
#
#  Your Task:
#    Write: save_as_jsonl(data, filepath)
#           load_from_jsonl(filepath) -> list[dict]
#    save_as_jsonl: write each dict as one JSON line to the file.
#    load_from_jsonl: read the file back and return a list of dicts.
#    Test by saving the Exercise 1 dataset and loading it back.
#    Verify the loaded data matches the original.
#
#  C# Analogy:
#    Like File.WriteAllLines() where each line is JsonSerializer.Serialize(obj),
#    and File.ReadAllLines() + JsonSerializer.Deserialize() on load.
# ============================================================

print("-" * 50)
print("EXERCISE 2: Save and Load JSONL")
print("-" * 50)
print()

def save_as_jsonl(data, filepath):
    """
    Save a list of dicts to a JSONL file.
    Each dict becomes one line in the file.

    Parameters:
        data     (list[dict]): The data to save.
        filepath (str)       : Path to the output .jsonl file.
    """
    # TODO: Open the file for writing.
    # For each dict in data: write json.dumps(item) + "\n"
    pass  # Replace with your implementation


def load_from_jsonl(filepath):
    """
    Load a JSONL file into a list of dicts.
    Each line in the file becomes one dict in the list.

    Parameters:
        filepath (str): Path to the .jsonl file.

    Returns:
        list[dict]: All examples loaded from the file.
    """
    # TODO: Open the file for reading.
    # For each line: if not empty, do json.loads(line.strip()) and append.
    pass  # Replace with your implementation


# Test save and load
if dataset:
    test_path = "test_dataset.jsonl"                 # Temporary file path
    save_as_jsonl(dataset, test_path)                # Save the dataset

    if os.path.exists(test_path):                    # Confirm file was created
        loaded = load_from_jsonl(test_path)          # Load it back
        match  = (loaded == dataset) if loaded else False  # Check they match
        print(f"  Saved to     : {test_path}")
        print(f"  Lines written: {len(dataset)}")
        print(f"  Lines loaded : {len(loaded) if loaded else 0}")
        print(f"  Data matches : {match}")
        os.remove(test_path)                         # Clean up temp file
    else:
        print("  File was not created (function not implemented yet).")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Applying a Prompt Template
#
#  Background:
#    LLMs are often fine-tuned with structured prompts.
#    The Alpaca template is very popular:
#
#      ### Instruction:
#      {instruction}
#
#      ### Input:
#      {input}
#
#      ### Response:
#      {output}
#
#    The {placeholders} are filled with real values from each example.
#    This teaches the LLM what format to expect and produce.
#
#  Your Task:
#    Write: apply_prompt_template(example, template_str) -> str
#    Replace {key} placeholders in template_str with values from example dict.
#    Use Python's str.format_map() or a manual loop.
#    Test with the Alpaca template and a sample example.
#
#  C# Analogy:
#    Like string.Format("Hello {0}", name) but using named keys instead of
#    positional indexes: string.Format("Hello {name}", new { name = "Alice" })
# ============================================================

print("-" * 50)
print("EXERCISE 3: Apply Prompt Template")
print("-" * 50)
print()

# The Alpaca fine-tuning prompt template
ALPACA_TEMPLATE = (
    "### Instruction:\n"
    "{instruction}\n\n"
    "### Input:\n"
    "{input}\n\n"
    "### Response:\n"
    "{output}"
)

# A sample training example with instruction, input, and output fields
sample_example = {
    "instruction": "Classify this support ticket into BUG, FEATURE, or HOW_TO.",
    "input"      : "The app freezes whenever I try to export a PDF.",
    "output"     : "BUG",
}

def apply_prompt_template(example, template_str):
    """
    Fill in a prompt template with values from an example dict.

    Parameters:
        example      (dict): Keys match the {placeholders} in the template.
        template_str (str) : Template string with {key} placeholders.

    Returns:
        str: The filled-in prompt string.
    """
    # TODO: Replace {key} placeholders with example[key] values.
    # Easiest approach: return template_str.format_map(example)
    # format_map is like .format(**kwargs) but takes a dict directly.
    pass  # Replace with your implementation


filled = apply_prompt_template(sample_example, ALPACA_TEMPLATE)
if filled:
    print("  Filled Alpaca template:")
    print()
    for line in filled.split("\n"):              # Print each line indented
        print(f"    {line}")
    print()
else:
    print("  (Function not implemented yet)")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Splitting a Dataset into Train / Val / Test
#
#  Background:
#    Before training, we split our dataset into three parts:
#      - Train (80%): Used to update model weights.
#      - Val   (10%): Used to check for overfitting during training.
#      - Test  (10%): Used ONCE at the very end to report final accuracy.
#    IMPORTANT: Always shuffle before splitting to avoid order bias!
#
#  Your Task:
#    Write: split_dataset(data, train=0.8, val=0.1, test=0.1)
#    Returns: (train_list, val_list, test_list)
#    Requirements:
#      - Assert that train + val + test == 1.0 (within floating point tolerance)
#      - Shuffle the data before splitting (use random.shuffle)
#      - Use deterministic shuffle (random.seed(42) before shuffling)
#
#  C# Analogy:
#    Like LINQ .Shuffle().Take(80%).Skip(0) for train,
#    .Skip(80%).Take(10%) for val, .Skip(90%) for test.
#    (LINQ doesn't have Shuffle, but you get the idea.)
# ============================================================

print("-" * 50)
print("EXERCISE 4: Split Dataset into Train / Val / Test")
print("-" * 50)
print()

def split_dataset(data, train=0.8, val=0.1, test=0.1):
    """
    Shuffle and split data into train, validation, and test sets.

    Parameters:
        data  (list): All examples to split.
        train (float): Fraction for training (e.g., 0.8).
        val   (float): Fraction for validation (e.g., 0.1).
        test  (float): Fraction for test (e.g., 0.1).

    Returns:
        (list, list, list): (train_data, val_data, test_data)
    """
    # TODO: Complete this function.
    # Step 1: Assert that train + val + test is approximately 1.0
    #         Use: assert abs((train + val + test) - 1.0) < 1e-9, "Splits must sum to 1.0"
    # Step 2: Make a copy of data (don't mutate the original)
    # Step 3: random.seed(42), then random.shuffle(data_copy)
    # Step 4: Compute split indices using int(len * fraction)
    # Step 5: Slice and return the three parts
    pass  # Replace with your implementation


if dataset:
    train_split, val_split, test_split = split_dataset(dataset) or ([], [], [])

    if train_split:
        print(f"  Total examples : {len(dataset)}")
        print(f"  Train          : {len(train_split)} ({len(train_split)/len(dataset)*100:.0f}%)")
        print(f"  Val            : {len(val_split)}  ({len(val_split)/len(dataset)*100:.0f}%)")
        print(f"  Test           : {len(test_split)}  ({len(test_split)/len(dataset)*100:.0f}%)")
        print()

        # Verify no overlap between splits
        train_texts = {item["text"] for item in train_split}  # Set of texts in train
        val_texts   = {item["text"] for item in val_split}
        test_texts  = {item["text"] for item in test_split}
        overlap = train_texts & val_texts & test_texts       # Intersection (should be empty)
        print(f"  Overlap between splits: {len(overlap)} (should be 0)")
    else:
        print("  (Function not implemented yet)")
print()


# ============================================================
#  SOLUTIONS  (commented out -- try it yourself first!)
# ============================================================

"""
# ---- SOLUTION: Exercise 1 ----

def build_classification_dataset():
    bug_examples = [
        {"text": "App crashes when I click the save button",         "label": "BUG"},
        {"text": "Login page shows blank screen on mobile",          "label": "BUG"},
        {"text": "Export to PDF fails with an error message",        "label": "BUG"},
        {"text": "Search results are not loading after filter",      "label": "BUG"},
        {"text": "Notifications are not appearing on the dashboard", "label": "BUG"},
        {"text": "User profile picture does not upload correctly",   "label": "BUG"},
        {"text": "Date picker shows wrong month after navigation",   "label": "BUG"},
        {"text": "Deleted records still appear in the list",         "label": "BUG"},
        {"text": "Password reset email is not being received",       "label": "BUG"},
        {"text": "Graph displays incorrect data after refresh",      "label": "BUG"},
    ]
    feature_examples = [
        {"text": "Please add dark mode to the application",          "label": "FEATURE"},
        {"text": "Allow users to export data as Excel spreadsheet",  "label": "FEATURE"},
        {"text": "Add support for two-factor authentication",        "label": "FEATURE"},
        {"text": "Request for bulk delete on the records page",      "label": "FEATURE"},
        {"text": "Add keyboard shortcuts for common actions",        "label": "FEATURE"},
        {"text": "Allow custom themes and colour palettes",          "label": "FEATURE"},
        {"text": "Request API access for third-party integration",   "label": "FEATURE"},
        {"text": "Add a timeline view to the project board",         "label": "FEATURE"},
        {"text": "Allow drag and drop to reorder list items",        "label": "FEATURE"},
        {"text": "Support multiple languages in the user interface", "label": "FEATURE"},
    ]
    howto_examples = [
        {"text": "How do I reset my account password?",              "label": "HOW_TO"},
        {"text": "How can I invite team members to my workspace?",   "label": "HOW_TO"},
        {"text": "Where do I find my API key?",                      "label": "HOW_TO"},
        {"text": "How do I export my data to CSV?",                  "label": "HOW_TO"},
        {"text": "Can you explain how the billing cycle works?",     "label": "HOW_TO"},
        {"text": "How to set up email notifications?",               "label": "HOW_TO"},
        {"text": "What is the difference between admin and viewer?", "label": "HOW_TO"},
        {"text": "How do I archive a completed project?",            "label": "HOW_TO"},
        {"text": "How to connect my Google account to the app?",     "label": "HOW_TO"},
        {"text": "How do I change my notification preferences?",     "label": "HOW_TO"},
    ]
    return bug_examples + feature_examples + howto_examples


# ---- SOLUTION: Exercise 2 ----

def save_as_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:   # Open file for writing
        for item in data:                               # Loop over all examples
            f.write(json.dumps(item) + "\n")            # Write one JSON line

def load_from_jsonl(filepath):
    results = []                                        # List to collect loaded items
    with open(filepath, "r", encoding="utf-8") as f:   # Open file for reading
        for line in f:                                  # Read line by line
            line = line.strip()                         # Remove whitespace/newline
            if line:                                    # Skip empty lines
                results.append(json.loads(line))        # Parse JSON and add to list
    return results


# ---- SOLUTION: Exercise 3 ----

def apply_prompt_template(example, template_str):
    return template_str.format_map(example)    # Fill all {key} placeholders from dict


# ---- SOLUTION: Exercise 4 ----

def split_dataset(data, train=0.8, val=0.1, test=0.1):
    assert abs((train + val + test) - 1.0) < 1e-9, "Splits must sum to 1.0"
    data_copy = list(data)                     # Copy to avoid mutating original
    random.seed(42)                            # Deterministic shuffle
    random.shuffle(data_copy)                  # Shuffle the copy
    n         = len(data_copy)                 # Total number of examples
    n_train   = int(n * train)                 # Number of training examples
    n_val     = int(n * val)                   # Number of validation examples
    # Test gets whatever is left (avoids off-by-one from rounding)
    train_data = data_copy[:n_train]
    val_data   = data_copy[n_train : n_train + n_val]
    test_data  = data_copy[n_train + n_val:]
    return train_data, val_data, test_data
"""

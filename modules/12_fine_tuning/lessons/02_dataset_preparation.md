# Lesson 02: Dataset Preparation

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain the three formats for fine-tuning datasets
2. Build a training dataset in the instruction-response format
3. Split data into train/validation/test sets
4. Identify and fix common dataset quality problems
5. Calculate how many training examples you need

---

## GLOSSARY

```
Training Dataset:
  The labeled examples the model learns from.
  Format: (input, expected_output) pairs.
  More data = generally better performance (up to a point).

Validation Dataset:
  A held-out portion of data used to MONITOR training progress.
  The model NEVER trains on this data.
  Used to detect overfitting (train loss down, val loss up = overfitting).
  Typically 10-20% of total data.

Test Dataset:
  Another held-out portion used for FINAL evaluation after training.
  Only look at test results ONCE, at the very end.
  Never used during training or hyperparameter tuning.
  Typically 10% of total data.

Train/Val/Test Split:
  Standard: 80% train, 10% validation, 10% test.
  Small datasets: 90% train, 5% val, 5% test.

Instruction-Response Format:
  The most common format for fine-tuning instruction-following models.
  Each example: {"instruction": "...", "input": "...", "output": "..."}
  Used by: Alpaca, LLaMA-Instruct, Mistral-Instruct.

JSONL (JSON Lines):
  A file format where each line is a valid JSON object.
  Standard format for fine-tuning datasets.
  Easy to load line by line (memory-efficient for large datasets).
  Example: {"text": "..."}\n{"text": "..."}\n

Prompt Template:
  A fixed format string used to structure each training example.
  CRITICAL: must be identical during training AND inference.
  Example: "### Instruction:\n{instruction}\n\n### Response:\n{output}"

Data Leakage:
  When test data accidentally appears in training data.
  The model "cheats" -- appears to perform well but fails on truly new data.
  Fix: split data BEFORE any preprocessing.

Label Quality:
  How accurate and consistent your training labels are.
  Low quality labels -> model learns wrong patterns.
  Rule: if you can't agree on the label yourself, the model won't learn it.

Tokenization:
  Converting text to tokens before training.
  Max sequence length: most models accept 512-2048 tokens per example.
  Examples longer than max length are truncated (cut off).
```

---

## Part 1: The Three Dataset Formats

### Format 1: Classification (Input -> Label)
```json
{"text": "App crashes when I log in.",      "label": "BUG"}
{"text": "Please add a dark mode option.",  "label": "FEATURE_REQUEST"}
{"text": "Server down for 2 hours.",        "label": "OUTAGE"}
{"text": "How do I reset my password?",     "label": "HOW_TO"}
```

Use when: simple classification tasks (sentiment, category, intent).

### Format 2: Completion (Prompt -> Continuation)
```json
{"prompt": "Translate to French: Hello world",  "completion": "Bonjour le monde"}
{"prompt": "Summarize: [long article text]",     "completion": "Short one-sentence summary."}
{"prompt": "Fix this bug: def add(a,b): a+b",   "completion": "def add(a, b): return a + b"}
```

Use when: text completion, translation, summarization, code fixing.

### Format 3: Instruction-Response (Most Flexible)
```json
{
  "instruction": "Classify this support ticket into one of: BUG, FEATURE, OUTAGE, HOW_TO",
  "input": "App crashes when I try to log in with my Google account.",
  "output": "BUG"
}
```

Use when: instruction-following models (like ChatGPT). Most powerful format.
The "instruction" field is the task. "input" is the data. "output" is the answer.

---

## Part 2: The Prompt Template

The prompt template combines instruction + input into a single string for the model.
**The SAME template must be used for BOTH training and inference.**

```
TRAINING EXAMPLE:

  Raw data:
    instruction: "Classify this support ticket: BUG, FEATURE, OUTAGE, or HOW_TO"
    input:       "App crashes on login."
    output:      "BUG"

  After applying prompt template:
    "### Instruction:
    Classify this support ticket: BUG, FEATURE, OUTAGE, or HOW_TO

    ### Input:
    App crashes on login.

    ### Response:
    BUG"

  The model learns: when it sees this format, output the label.

INFERENCE (after training):
  You send:
    "### Instruction:
    Classify this support ticket: BUG, FEATURE, OUTAGE, or HOW_TO

    ### Input:
    Users can't connect to the API.

    ### Response:"
  (Note: "### Response:" is included but empty -- model fills it in)

  Model outputs: "OUTAGE"
```

### Common Prompt Templates

```python
# Template 1: Alpaca-style (used by LLaMA fine-tunes)
ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# Template 2: Simple classification
CLASSIFICATION_TEMPLATE = """Task: {instruction}
Text: {input}
Label: {output}"""

# Template 3: Chat format (used by ChatGPT-style models)
CHAT_TEMPLATE = """<|user|>
{instruction}: {input}
<|assistant|>
{output}"""
```

---

## Part 3: How Many Examples Do You Need?

This is the most common question. The honest answer: it depends.

```
RULE OF THUMB:
  Classification (2-10 labels):     100 - 1,000 examples per class
  Classification (10-100 labels):   500 - 5,000 examples per class
  Instruction following:            1,000 - 50,000 total examples
  Domain expert chatbot:            10,000 - 100,000 total examples
  Replicating ChatGPT-level:        500,000+ examples

MINIMUM VIABLE:
  You can fine-tune with as few as 50-100 examples and see improvement.
  Quality matters more than quantity.
  100 clean, correct examples > 10,000 noisy, inconsistent examples.
```

### The Quality vs Quantity Tradeoff

```
BAD DATASET (1000 examples, low quality):
  {"input": "app broken", "label": "BUG"}
  {"input": "app broken", "label": "FEATURE"}   <- same input, different label!
  {"input": "not working", "label": "HOW_TO"}   <- ambiguous classification

GOOD DATASET (100 examples, high quality):
  {"input": "Login button throws NullPointerException", "label": "BUG"}
  {"input": "Add export to PDF feature", "label": "FEATURE"}
  {"input": "How do I change my email address?", "label": "HOW_TO"}

The 100-example good dataset will perform BETTER.
```

---

## Part 4: The Train/Val/Test Split

```python
import random

def split_dataset(data: list, train_pct: float = 0.8,
                  val_pct: float = 0.1, test_pct: float = 0.1) -> tuple:
    """
    Split data into train, validation, and test sets.
    data:      list of all examples
    train_pct: fraction for training (default 80%)
    val_pct:   fraction for validation (default 10%)
    test_pct:  fraction for test (default 10%)
    Returns:   (train, val, test) lists
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-9, "Percentages must sum to 1.0"

    random.shuffle(data)               # Shuffle BEFORE splitting to avoid ordering bias

    n = len(data)
    train_end = int(n * train_pct)     # Where training data ends
    val_end   = train_end + int(n * val_pct)  # Where validation ends

    train = data[:train_end]           # First 80%
    val   = data[train_end:val_end]    # Next 10%
    test  = data[val_end:]             # Last 10%

    return train, val, test
```

### Why Keep Them Separate?

```
TRAINING SET (80%):
  The model LEARNS from this.
  Every example updates the model weights.

VALIDATION SET (10%):
  The model MONITORS on this during training.
  Used to decide: "Should I stop training now?"
  (If val loss starts rising, stop -- overfitting has started)

TEST SET (10%):
  The model is EVALUATED on this only ONCE, at the very end.
  Gives an honest estimate of real-world performance.
  If you look at test results mid-training and adjust -> data leakage!
```

---

## Part 5: Data Quality Checklist

Before training, check your dataset for these common problems:

```
CHECKLIST:

  CONSISTENCY
  [ ] Same input always maps to same output (no contradictions)
  [ ] Labels are spelled consistently ("BUG" not sometimes "Bug" or "bug")
  [ ] Same prompt template used for all examples

  BALANCE
  [ ] Class distribution is not severely skewed
      (If 95% of data is "BUG" and 5% is "FEATURE", model ignores "FEATURE")
  [ ] At least 50 examples per class (for classification)

  QUALITY
  [ ] Labels are correct (spot-check 10% manually)
  [ ] Inputs are realistic (match what you will send at inference time)
  [ ] No HTML, strange characters, or truncated text

  SPLITS
  [ ] Data split BEFORE any preprocessing
  [ ] No overlap between train/val/test sets
  [ ] Similar distributions across all splits

  FORMAT
  [ ] All examples in same format (JSONL or consistent dict structure)
  [ ] Prompt template is documented and will be reused at inference
  [ ] Long examples truncated consistently (not randomly cut mid-sentence)
```

---

## Part 6: C#/.NET Analogy

```
Fine-tuning dataset = a unit test suite for the model.

Training set = the test cases the model practices on.
Validation set = a small regression suite run after each epoch.
Test set = the final acceptance test you run before shipping.

Data leakage = having production data accidentally in your unit tests.
              The tests "pass" but the code fails in real scenarios.

Overfitting = model memorizes the test cases instead of understanding the logic.
              Passes all tests, fails on any new input.

JSONL format = like writing one JSON object per line in a log file.
              Easy to append, easy to read line by line (streaming).
              In C#: equivalent to reading a file line by line with JsonSerializer.Deserialize<T>(line).
```

---

## Key Takeaways

1. Three dataset formats: Classification (input -> label), Completion (prompt -> text), Instruction-Response (most flexible).

2. Prompt template = critical. Must be IDENTICAL during training and inference.

3. Quantity rule of thumb: 100-1000 per class for classification, 1000-50000 for instruction following.

4. Quality > quantity. 100 clean examples outperform 10,000 noisy ones.

5. Always split: 80% train, 10% validation, 10% test. Shuffle BEFORE splitting.

6. Validate dataset before training: consistency, balance, correct labels.

---

## Next

Lesson 03: LoRA and PEFT
  - Why can't we just fine-tune all the weights?
  - What is LoRA and how does it work?
  - How does LoRA reduce memory usage by 100x?
  - How to implement a LoRA layer from scratch.

# Lesson 5: Large Dataset Streaming

## The Data Problem

LLaMA 3 trained on 15 trillion tokens.
At 2 bytes per token (using byte-pair encoding), that is:

```
15,000,000,000,000 tokens × 2 bytes = 30,000,000,000,000 bytes = 30 TB
```

30 terabytes of text data.

No single machine has 30 TB of RAM.
Even downloading 30 TB of data would take weeks and cost hundreds of dollars in storage.

The solution: **streaming datasets** — process data as it arrives, never store it all locally.

---

## What Is Dataset Streaming?

Instead of loading a dataset into memory:

```
TRADITIONAL (non-streaming):
  1. Download entire dataset to disk    ← weeks, terabytes of storage
  2. Load entire dataset into RAM       ← impossible for large datasets
  3. Iterate over it during training    ← fast, but steps 1-2 blocked you

STREAMING:
  1. Connect to dataset source          ← instant
  2. Fetch one batch at a time          ← as needed, on demand
  3. Tokenize and train as you go       ← no waiting, no downloading everything
```

Streaming is like watching a YouTube video vs downloading the whole file first.

---

## HuggingFace Datasets Library

The standard tool for dataset streaming in the LLM world is HuggingFace `datasets`.

Install:
```bash
pip install datasets
```

Basic usage comparison:

```python
from datasets import load_dataset

# NON-STREAMING: downloads and loads everything
dataset = load_dataset("c4", "en", split="train")
# This would download ~750 GB. Don't do this at home.

# STREAMING: loads nothing, just opens a connection
dataset = load_dataset("c4", "en", split="train", streaming=True)
# Instant. No download. Returns an IterableDataset.
```

The streaming version returns an `IterableDataset` — a lazy iterator.
Nothing is downloaded until you iterate.

---

## IterableDataset vs Dataset

```
+------------------------------------------------------------------+
|  DATASET (non-streaming)         ITERABLEDATASET (streaming)     |
+------------------------------------------------------------------+
|                                                                  |
|  Type: Dataset (in-memory)       Type: IterableDataset (lazy)    |
|  Load: everything upfront        Load: one example at a time     |
|  Index: dataset[42]  ✓           Index: dataset[42]  ✗           |
|  Shuffle: true shuffle ✓         Shuffle: buffer shuffle ✓       |
|  Repeatable: yes ✓               Repeatable: yes (re-iterate) ✓  |
|  Download: full dataset           Download: only what you use    |
|  RAM: O(dataset size)            RAM: O(buffer size)             |
|  Best for: < 10 GB datasets      Best for: TB-scale datasets     |
+------------------------------------------------------------------+
```

---

## Core Operations on IterableDataset

### 1. Iterating

```python
dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

for i, example in enumerate(dataset):
    print(example["text"][:100])   # first 100 chars of article
    if i >= 4:
        break                       # only look at 5 examples

# Output:
# Anarchism is a political philosophy...
# Autism is a neurodevelopmental disorder...
# A (named a /eɪ/, plural As) is the first letter...
```

### 2. Mapping (Tokenization)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )

# Map applies tokenize lazily — nothing runs until you iterate
tokenized = dataset.map(tokenize, remove_columns=["text", "url", "title"])
```

### 3. Filtering

```python
# Only keep examples with text longer than 500 characters
filtered = dataset.filter(lambda ex: len(ex["text"]) > 500)
```

### 4. Shuffling (Buffer Shuffle)

Full shuffle is impossible on a streaming dataset (you don't have all the data).
Instead, a **buffer shuffle** is used:

```
BUFFER SHUFFLE:

  1. Fill a buffer with N examples from the stream
  2. Randomly pick examples from the buffer to yield
  3. Refill empty slots from the stream
  4. Repeat

  buffer_size=10000: holds 10,000 examples, shuffles within that window
  buffer_size=1000000: holds 1M examples, better shuffle quality, more RAM
```

```python
# Shuffle with a buffer of 10,000 examples
shuffled = dataset.shuffle(seed=42, buffer_size=10_000)
```

### 5. Batching

```python
# Group into batches for training
batched = tokenized.batch(batch_size=32)

for batch in batched:
    input_ids = batch["input_ids"]   # shape: [32, 1024]
    # ... training step ...
```

---

## Chaining Operations

Operations chain together, all lazily:

```python
dataset = (
    load_dataset("c4", "en", split="train", streaming=True)
    .filter(lambda ex: len(ex["text"]) > 200)       # skip short texts
    .shuffle(seed=42, buffer_size=10_000)             # buffer shuffle
    .map(tokenize, remove_columns=["text", "url"])    # tokenize on-the-fly
    .batch(batch_size=32)                             # batch for training
)

# NOTHING has been downloaded or processed yet.
# The chain of operations is stored as a recipe.

# Processing starts only when you iterate:
for batch in dataset:
    train_step(batch)
    # Each iteration: fetches examples, filters, shuffles, tokenizes, batches
```

This is the **lazy evaluation** pattern.

---

## Interleaving Multiple Datasets

You can mix multiple datasets with controlled probabilities:

```python
from datasets import interleave_datasets

# Load multiple streaming datasets
wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
books = load_dataset("bookcorpus", split="train", streaming=True)
code  = load_dataset("codeparrot/github-code", split="train", streaming=True)

# Mix: 60% wiki, 30% books, 10% code
mixed = interleave_datasets(
    [wiki, books, code],
    probabilities=[0.6, 0.3, 0.1],
    seed=42
)
```

This is how real LLMs create diverse training data mixtures.
LLaMA 3's training mix: web crawl + code + Wikipedia + books + scientific papers.

---

## Practical Training Loop with Streaming

```python
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

# Setup
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_and_chunk(examples, chunk_size=512):
    tokens = tokenizer(
        examples["text"],
        truncation=False,  # don't truncate — we'll chunk below
        return_tensors=None
    )
    # Concatenate all token sequences and chunk into fixed-size pieces
    all_tokens = []
    for ids in tokens["input_ids"]:
        all_tokens.extend(ids)

    # Split into chunks
    chunks = []
    for i in range(0, len(all_tokens) - chunk_size, chunk_size):
        chunks.append(all_tokens[i : i + chunk_size])

    return {"input_ids": chunks}

# Streaming pipeline
dataset = (
    load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    .map(tokenize_and_chunk, batched=True, remove_columns=["text", "url", "title"])
    .shuffle(seed=42, buffer_size=5_000)
    .batch(batch_size=8)
)

# Training loop
model = MyGPTModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step, batch in enumerate(dataset):
    input_ids = torch.tensor(batch["input_ids"])
    targets = input_ids.clone()

    logits = model(input_ids)
    loss = cross_entropy(logits, targets)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

---

## Resuming Training (Checkpointing with Streaming)

Streaming datasets have no random access — you cannot jump to "position 1,000,000".
To resume training after a crash, you need to skip forward:

```python
# Resume training from step 50,000
RESUME_STEP = 50_000

for step, batch in enumerate(dataset):
    if step < RESUME_STEP:
        continue  # skip examples we already trained on
    # ... training step ...
```

This works but is slow for large skips. Better approach: use `.skip()`:

```python
# Skip first 50,000 batches efficiently
dataset = dataset.skip(RESUME_STEP)

for step, batch in enumerate(dataset):
    actual_step = RESUME_STEP + step
    # ... training step ...
```

---

## Popular Free Streaming Datasets

These datasets are on HuggingFace Hub and can be streamed for free:

| Dataset | Size | Content | Best For |
|---------|------|---------|---------|
| `c4` "en" | 750 GB | Cleaned Common Crawl web text | General pretraining |
| `wikimedia/wikipedia` | ~20 GB | All Wikipedia articles | Factual knowledge |
| `bookcorpus` | ~5 GB | Books text | Long-range coherence |
| `EleutherAI/pile` | 825 GB | Diverse web + books + code | Research pretraining |
| `codeparrot/github-code` | 250 GB | Code from GitHub | Code models |
| `HuggingFaceFW/fineweb` | 15 TB | High-quality web text | State-of-art pretraining |
| `OpenHermes-2.5` | ~1 GB | Instruction tuning data | Fine-tuning |

---

## C# Analogy: IEnumerable and yield return

```csharp
// HuggingFace IterableDataset is EXACTLY like C# IEnumerable<T> with yield return

// NON-STREAMING (like loading entire List<T>):
List<TrainingExample> dataset = LoadAllData("training_data.bin");  // 750 GB in RAM!
foreach (var example in dataset) { Train(example); }

// STREAMING (like IEnumerable with yield return):
IEnumerable<TrainingExample> StreamDataset(string path)
{
    using var reader = new StreamReader(path);
    string line;
    while ((line = reader.ReadLine()) != null)
    {
        yield return Parse(line);  // one at a time, no buffering
    }
}

foreach (var example in StreamDataset("data.txt"))
{
    Train(example);  // processes one line, then fetches next
}

// HuggingFace streaming is the same:
// load_dataset(..., streaming=True) returns something like StreamDataset()
// .map() is like .Select()
// .filter() is like .Where()
// .batch() is like .Chunk() (LINQ .Chunk() in .NET 6+)
// .shuffle(buffer_size=N) is like loading N items, shuffling, taking one, refilling

// The Python pipeline:
// dataset.filter(...).map(...).shuffle(...).batch(32)
//
// Is equivalent to C# LINQ:
// streamDataset.Where(...).Select(...).ShuffleBuffered(10000).Chunk(32)
```

---

## Quiz Questions

**Q1**: What is the main advantage of `streaming=True` in `load_dataset`?
        a) It makes loading faster by using multiple threads
        b) It allows training without downloading the entire dataset
        c) It automatically shuffles the data perfectly
        d) It reduces the size of each example by compression

**Q2**: Why can't you do `dataset[42]` on a streaming IterableDataset?
        a) Python doesn't support index access on custom iterators
        b) The dataset has no index — data arrives in a stream and is not stored
        c) Only the first 10 examples are accessible
        d) You need to call `.collect()` first

**Q3**: Buffer shuffle on a streaming dataset with buffer_size=10000 means:
        a) The entire dataset is shuffled perfectly using 10000 iterations
        b) 10000 examples are held in memory and shuffled within that window
        c) Shuffling is applied in chunks of 10000 examples with perfect randomness
        d) Only the first 10000 examples are shuffled, the rest are in order

*(Answers: Q1=b, Q2=b, Q3=b)*

---

## Key Takeaways

1. TB-scale datasets cannot be downloaded or loaded into RAM — streaming is required
2. HuggingFace `datasets` library: `load_dataset(..., streaming=True)` → IterableDataset
3. IterableDataset is lazy — nothing downloaded until you iterate
4. Operations chain lazily: `.filter()`, `.map()`, `.shuffle()`, `.batch()`
5. Buffer shuffle: hold N examples in RAM, shuffle within that window
6. `interleave_datasets()` mixes multiple datasets with probability weights
7. Resume training with `.skip(N)` to fast-forward past already-trained examples
8. C# analogy: `IEnumerable<T>` with `yield return` + LINQ operators

---

*Next: Lesson 6 — Knowledge Distillation (compressing a 70B model into a 7B model)*

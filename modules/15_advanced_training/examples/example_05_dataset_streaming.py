"""
Example 05: Dataset Streaming — Lazy Pipeline Simulation
Module 15: Advanced LLM Training

Simulates a HuggingFace-style streaming dataset pipeline in pure Python.
No actual HuggingFace datasets library needed — we build the concept from scratch.

Demonstrates:
  - Lazy evaluation (nothing computed until iterated)
  - Chaining filter → map → shuffle → batch
  - Buffer shuffle behavior
  - Memory usage: streaming vs loading everything

Run:  python example_05_dataset_streaming.py
Deps: none (pure Python)
"""

import random
import time
import sys
from typing import Iterator, Callable, Any


def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


# ─────────────────────────────────────────────────────────
# ITERABLEDATASET: Lazy Pipeline
# ─────────────────────────────────────────────────────────

class IterableDataset:
    """
    Simulates HuggingFace IterableDataset.
    Operations are lazy — nothing executes until you iterate.
    This is the streaming dataset pattern.
    """

    def __init__(self, source: Iterator):
        self._source = source
        self._ops = []    # list of (operation_type, args) — applied lazily

    def filter(self, fn: Callable) -> "IterableDataset":
        """Keep only examples where fn(example) is True."""
        new_ds = IterableDataset(self._source)
        new_ds._ops = self._ops + [("filter", fn)]
        return new_ds

    def map(self, fn: Callable, remove_columns: list = None) -> "IterableDataset":
        """Transform each example with fn."""
        new_ds = IterableDataset(self._source)
        new_ds._ops = self._ops + [("map", fn, remove_columns or [])]
        return new_ds

    def shuffle(self, seed: int = 42, buffer_size: int = 1000) -> "IterableDataset":
        """Buffer shuffle: fill buffer, yield random items, refill."""
        new_ds = IterableDataset(self._source)
        new_ds._ops = self._ops + [("shuffle", seed, buffer_size)]
        return new_ds

    def batch(self, batch_size: int) -> "IterableDataset":
        """Group examples into batches."""
        new_ds = IterableDataset(self._source)
        new_ds._ops = self._ops + [("batch", batch_size)]
        return new_ds

    def skip(self, n: int) -> "IterableDataset":
        """Skip first n examples."""
        new_ds = IterableDataset(self._source)
        new_ds._ops = self._ops + [("skip", n)]
        return new_ds

    def __iter__(self):
        """Apply all operations lazily when iteration starts."""
        # Start with raw source
        gen = iter(self._source)

        # Apply each operation in sequence
        for op in self._ops:
            if op[0] == "filter":
                _, fn = op
                gen = (ex for ex in gen if fn(ex))

            elif op[0] == "map":
                _, fn, remove_cols = op
                def _map(g, f, remove):
                    for ex in g:
                        result = f(ex)
                        for col in remove:
                            result.pop(col, None)
                        yield result
                gen = _map(gen, fn, remove_cols)

            elif op[0] == "shuffle":
                _, seed, buffer_size = op
                gen = _buffer_shuffle(gen, buffer_size=buffer_size, seed=seed)

            elif op[0] == "batch":
                _, batch_size = op
                gen = _batcher(gen, batch_size=batch_size)

            elif op[0] == "skip":
                _, n = op
                def _skip(g, num):
                    for i, ex in enumerate(g):
                        if i >= num:
                            yield ex
                gen = _skip(gen, n)

        yield from gen


def _buffer_shuffle(source: Iterator, buffer_size: int, seed: int) -> Iterator:
    """
    Buffer shuffle: fill buffer with buffer_size items,
    randomly yield one item, refill from source.

    This approximates true shuffling without loading all data.
    """
    rng = random.Random(seed)
    buffer = []

    # Fill initial buffer
    for item in source:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            break

    while buffer:
        # Pick a random item from buffer
        idx = rng.randrange(len(buffer))
        yield buffer[idx]
        buffer[idx] = buffer[-1]   # replace with last item
        buffer.pop()

        # Try to refill from source
        try:
            buffer.append(next(source))
        except StopIteration:
            pass   # source exhausted, drain buffer


def _batcher(source: Iterator, batch_size: int) -> Iterator:
    """Group items into fixed-size batches."""
    batch = []
    for item in source:
        batch.append(item)
        if len(batch) >= batch_size:
            yield _collate(batch)
            batch = []
    if batch:
        yield _collate(batch)   # yield final partial batch


def _collate(examples: list) -> dict:
    """Merge list of dicts into dict of lists (like PyTorch DataLoader collate)."""
    if not examples:
        return {}
    keys = examples[0].keys()
    return {k: [ex[k] for ex in examples] for k in keys}


# ─────────────────────────────────────────────────────────
# FAKE DATA SOURCE (simulates a large remote dataset)
# ─────────────────────────────────────────────────────────

def fake_wikipedia_stream(n_articles: int = 100, seed: int = 0) -> Iterator:
    """
    Simulate a streaming Wikipedia dataset.
    In real use: load_dataset("wikipedia", streaming=True)
    Here: generate fake articles on demand.
    """
    rng = random.Random(seed)
    topics = [
        "Quantum mechanics", "Byzantine Empire", "Python programming",
        "Machine learning", "French Revolution", "DNA replication",
        "Solar system", "World War II", "Thermodynamics", "Roman Empire",
    ]
    for i in range(n_articles):
        topic = topics[i % len(topics)]
        word_count = rng.randint(50, 500)
        # Simulate article text length (not real text)
        text = f"Article about {topic}. " * (word_count // 4)
        yield {
            "id": i,
            "title": f"{topic} - Article {i}",
            "text": text,
            "url": f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}_{i}",
            "word_count": word_count,
        }


# ─────────────────────────────────────────────────────────
# SIMPLE TOKENIZER SIMULATION
# ─────────────────────────────────────────────────────────

def simple_tokenize(example: dict, max_length: int = 64) -> dict:
    """Fake tokenizer: split on spaces, truncate, pad."""
    words = example["text"].split()[:max_length]
    token_ids = [hash(w) % 50000 for w in words]   # fake token IDs
    # Pad to max_length
    token_ids = token_ids + [0] * (max_length - len(token_ids))
    return {
        "input_ids": token_ids[:max_length],
        "text": example["text"],   # kept for display; remove_columns removes this
        "title": example["title"],
    }


# ─────────────────────────────────────────────────────────
# DEMO 1: Basic Streaming Pipeline
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: Basic Streaming Pipeline")

print("\nBuilding pipeline (nothing executes yet)...")

dataset = IterableDataset(fake_wikipedia_stream(n_articles=50))
pipeline = (
    dataset
    .filter(lambda ex: ex["word_count"] > 100)         # skip short articles
    .map(simple_tokenize, remove_columns=["url", "id"]) # tokenize (fake)
    .shuffle(seed=42, buffer_size=10)                   # buffer shuffle
    .batch(batch_size=4)                                # group into batches
)

print("Pipeline built. No data fetched yet.")
print("\nIterating (data flows lazily)...\n")

for batch_num, batch in enumerate(pipeline):
    titles = batch["title"]
    token_ids = batch["input_ids"]
    print(f"  Batch {batch_num}: {len(titles)} examples")
    print(f"    First title:   {titles[0][:50]}")
    print(f"    Token ids[0]:  {token_ids[0][:8]} ...")
    if batch_num >= 2:
        print(f"  (stopping after 3 batches for demo)")
        break


# ─────────────────────────────────────────────────────────
# DEMO 2: Buffer Shuffle — Understanding the Approximation
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: Buffer Shuffle Behavior")

print("\nGenerating sequence 0..19, shuffled with different buffer sizes:")

def make_ordered_stream(n: int):
    for i in range(n):
        yield {"value": i}

print(f"\n{'Buffer Size':>14} {'Output Order'}")
print("-" * 60)

for buf_size in [1, 5, 10, 20]:
    ds = IterableDataset(make_ordered_stream(20))
    shuffled = ds.shuffle(seed=42, buffer_size=buf_size)
    values = [ex["value"] for ex in shuffled]
    print(f"{buf_size:>14}   {values}")

print("""
LESSON:
  buffer_size=1   → no shuffle (always picks index 0 from size-1 buffer)
  buffer_size=5   → partial shuffle within 5-item windows
  buffer_size=20  → perfect shuffle (entire dataset in buffer)
  buffer_size=N   → true shuffle (need N items in RAM: defeats streaming)

In practice: buffer_size=10000 to 1,000,000 gives good quality.
HuggingFace recommendation: buffer_size=1,000,000 for pretraining.
""")


# ─────────────────────────────────────────────────────────
# DEMO 3: Memory Usage — Streaming vs Loading All
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: Memory Usage — Streaming vs Loading All")

print("""
Comparing memory required for different dataset sizes:

  LOAD ALL:    memory = dataset_size × avg_example_size
  STREAMING:   memory = buffer_size × avg_example_size

Assumptions:
  - Average tokenized example: 512 tokens × 4 bytes = 2 KB
  - Buffer size: 10,000 examples
""")

example_size_kb = 512 * 4 / 1024   # 2 KB per example
buffer_size = 10_000

print(f"{'Dataset Size':>16} {'Load All (GB)':>16} {'Streaming (MB)':>16} {'Savings':>10}")
print("-" * 62)

for n_examples, label in [
    (100_000, "100K"),
    (1_000_000, "1M"),
    (10_000_000, "10M"),
    (100_000_000, "100M"),
    (1_000_000_000, "1B"),
    (15_000_000_000, "15B (LLaMA 3 scale)"),
]:
    load_all_gb = (n_examples * example_size_kb) / (1024 * 1024)
    streaming_mb = (buffer_size * example_size_kb) / 1024
    savings = load_all_gb * 1024 / streaming_mb
    print(f"{label:>16} {load_all_gb:>16.1f} {streaming_mb:>16.1f} {savings:>9.0f}x")

print(f"\nWith streaming + buffer_size=10,000:")
print(f"  Always ~{(buffer_size * example_size_kb)/1024:.0f} MB in RAM regardless of dataset size.")


# ─────────────────────────────────────────────────────────
# DEMO 4: Resume Training with Skip
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: Resuming Training with .skip()")

print("\nSimulating a training run that was interrupted at step 15:")

TOTAL_STEPS = 30
INTERRUPTED_AT = 15

full_stream = IterableDataset(fake_wikipedia_stream(n_articles=100))
full_pipeline = (
    full_stream
    .filter(lambda ex: ex["word_count"] > 50)
    .batch(batch_size=3)
)

print(f"\n  Training for {INTERRUPTED_AT} steps, then 'crash'...")
training_log = []
for step, batch in enumerate(full_pipeline):
    if step >= INTERRUPTED_AT:
        break
    training_log.append(batch["title"][0][:35])

print(f"  Crashed at step {INTERRUPTED_AT}. Resuming...")

# Resume: recreate pipeline and skip to where we left off
resume_stream = IterableDataset(fake_wikipedia_stream(n_articles=100))
resume_pipeline = (
    resume_stream
    .filter(lambda ex: ex["word_count"] > 50)
    .batch(batch_size=3)
    .skip(INTERRUPTED_AT)   # fast-forward past already-trained batches
)

resume_log = []
for step, batch in enumerate(resume_pipeline):
    actual_step = INTERRUPTED_AT + step
    resume_log.append(batch["title"][0][:35])
    if step >= 4:
        break

print(f"\n  Last 3 steps before crash:")
for t in training_log[-3:]:
    print(f"    {t}")

print(f"\n  First 3 steps after resume:")
for t in resume_log[:3]:
    print(f"    {t}")

print("\n  First example after resume should continue where crash left off.")
print("  (Same seed + same filter = same order = correct resume point)")


# ─────────────────────────────────────────────────────────
# DEMO 5: Mixing Multiple Datasets
# ─────────────────────────────────────────────────────────

print_section("DEMO 5: Interleaving Multiple Datasets (Probability Mixing)")

def interleave_datasets_by_prob(datasets: list, probabilities: list, seed: int = 42):
    """
    Interleave multiple datasets using probability weights.
    Similar to HuggingFace interleave_datasets().
    """
    rng = random.Random(seed)
    iters = [iter(ds) for ds in datasets]
    exhausted = [False] * len(datasets)

    while not all(exhausted):
        # Pick dataset based on probability (ignore exhausted ones)
        available = [(i, p) for i, p in enumerate(probabilities) if not exhausted[i]]
        if not available:
            break
        indices, probs = zip(*available)
        total = sum(probs)
        normalized = [p / total for p in probs]

        r = rng.random()
        cumulative = 0
        chosen = indices[-1]
        for idx, prob in zip(indices, normalized):
            cumulative += prob
            if r <= cumulative:
                chosen = idx
                break

        try:
            yield next(iters[chosen])
        except StopIteration:
            exhausted[chosen] = True


# Create three "different" datasets
wiki_stream = list({"source": "wikipedia", "id": i} for i in range(20))
books_stream = list({"source": "books", "id": i} for i in range(20))
code_stream = list({"source": "code", "id": i} for i in range(20))

mixed = list(interleave_datasets_by_prob(
    [iter(wiki_stream), iter(books_stream), iter(code_stream)],
    probabilities=[0.6, 0.3, 0.1],
    seed=42
))

from collections import Counter
source_counts = Counter(ex["source"] for ex in mixed[:30])
print(f"\nFirst 30 examples from 60%/30%/10% mix:")
print(f"  wikipedia: {source_counts['wikipedia']:>3} examples  (target: 60% = 18)")
print(f"  books:     {source_counts['books']:>3} examples  (target: 30% = 9)")
print(f"  code:      {source_counts['code']:>3} examples  (target: 10% = 3)")
print(f"\nData mixture sample: {[ex['source'][0].upper() for ex in mixed[:15]]}")
print(f"  (W=Wikipedia, B=Books, C=Code)")
print("""
LLaMA 3 training mix (approximate):
  Web crawl:   ~80% (CommonCrawl, FineWeb)
  Code:        ~8%  (GitHub)
  Wikipedia:   ~4%  (factual knowledge)
  Books:       ~4%  (long-form reasoning)
  Scientific:  ~4%  (ArXiv, papers)
""")

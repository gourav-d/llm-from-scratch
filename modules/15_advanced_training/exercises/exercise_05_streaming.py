"""
Exercise 05: Dataset Streaming — Build a Lazy Tokenization Pipeline
Module 15: Advanced LLM Training

TASKS:
  1. Implement lazy_filter() — yield examples matching a condition
  2. Implement lazy_map() — transform each example lazily
  3. Implement lazy_batch() — group examples into fixed-size batches
  4. Implement buffer_shuffle() — shuffle within a sliding buffer
  5. Build a full pipeline and verify lazy evaluation

Run:  python exercise_05_streaming.py
Deps: none (pure Python)
"""

import random
from typing import Iterator, Callable


# ─────────────────────────────────────────────────────────
# TASK 1: Lazy Filter
# ─────────────────────────────────────────────────────────

def lazy_filter(source: Iterator, predicate: Callable) -> Iterator:
    """
    Yield only examples where predicate(example) is True.
    Lazy: only processes examples when the caller iterates.

    Args:
        source:    any iterator of examples
        predicate: function that returns True/False for each example

    Yields:
        Examples where predicate is True

    HINT:
        for example in source:
            if predicate(example):
                yield example
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Lazy Map
# ─────────────────────────────────────────────────────────

def lazy_map(source: Iterator, transform: Callable) -> Iterator:
    """
    Apply transform to each example lazily.

    Args:
        source:    any iterator of examples
        transform: function to apply to each example

    Yields:
        Transformed examples

    HINT:
        for example in source:
            yield transform(example)
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Lazy Batch
# ─────────────────────────────────────────────────────────

def lazy_batch(source: Iterator, batch_size: int) -> Iterator:
    """
    Group examples into fixed-size batches.
    Last batch may be smaller if dataset size is not divisible by batch_size.

    Args:
        source:     any iterator of examples
        batch_size: number of examples per batch

    Yields:
        Lists of batch_size examples (last batch may be shorter)

    HINT:
        batch = []
        for example in source:
            batch.append(example)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch   # don't forget the last partial batch!
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Buffer Shuffle
# ─────────────────────────────────────────────────────────

def buffer_shuffle(source: Iterator, buffer_size: int, seed: int = 42) -> Iterator:
    """
    Approximate shuffle using a fixed-size buffer.

    Algorithm:
      1. Fill buffer with buffer_size items from source
      2. Pick a random item from buffer, yield it
      3. Replace that slot with next item from source
      4. Repeat until source is exhausted
      5. Then yield remaining buffer items in random order

    Args:
        source:      any iterator of examples
        buffer_size: number of items to hold in memory
        seed:        random seed for reproducibility

    Yields:
        Examples in shuffled order

    HINT:
        rng = random.Random(seed)
        buffer = []

        # Fill initial buffer
        for item in source:
            buffer.append(item)
            if len(buffer) >= buffer_size:
                break

        while buffer:
            idx = rng.randrange(len(buffer))
            yield buffer[idx]
            buffer[idx] = buffer[-1]
            buffer.pop()
            try:
                buffer.append(next(source))
            except StopIteration:
                pass
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# HELPER: Fake dataset generator
# ─────────────────────────────────────────────────────────

def fake_text_dataset(n: int, seed: int = 0) -> Iterator:
    """Generate n fake text examples. Simulates a remote streaming dataset."""
    rng = random.Random(seed)
    topics = ["machine learning", "history", "science", "art", "sports"]
    for i in range(n):
        length = rng.randint(20, 200)
        topic = topics[i % len(topics)]
        yield {
            "id": i,
            "text": f"Article {i} about {topic}. " * (length // 10),
            "length": length,
            "topic": topic,
        }


def simple_tokenize(example: dict, max_len: int = 32) -> dict:
    """Fake tokenizer: split on spaces, take first max_len tokens."""
    words = example["text"].split()[:max_len]
    token_ids = [abs(hash(w)) % 50000 for w in words]
    return {
        "id": example["id"],
        "topic": example["topic"],
        "token_ids": token_ids,
        "n_tokens": len(token_ids),
    }


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 60)
    print("  Exercise 05: Dataset Streaming")
    print("=" * 60)

    # Test 1: lazy_filter
    print("\n--- Test 1: lazy_filter ---")
    source = iter(range(20))
    result = lazy_filter(source, lambda x: x % 3 == 0)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        filtered = list(result)
        expected = [0, 3, 6, 9, 12, 15, 18]
        if filtered == expected:
            print(f"  PASS  filtered = {filtered}")
        else:
            print(f"  FAIL  got {filtered}, expected {expected}")

    # Test 2: lazy_map
    print("\n--- Test 2: lazy_map ---")
    source = iter([1, 2, 3, 4, 5])
    result = lazy_map(source, lambda x: x ** 2)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        mapped = list(result)
        expected = [1, 4, 9, 16, 25]
        if mapped == expected:
            print(f"  PASS  mapped = {mapped}")
        else:
            print(f"  FAIL  got {mapped}, expected {expected}")

    # Test 3: lazy_batch
    print("\n--- Test 3: lazy_batch ---")
    source = iter(range(10))
    result = lazy_batch(source, batch_size=3)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        batches = list(result)
        if len(batches) == 4 and batches[0] == [0, 1, 2] and batches[-1] == [9]:
            print(f"  PASS  {len(batches)} batches: {batches}")
        else:
            print(f"  FAIL  got {batches}")
            print(f"         expected [[0,1,2],[3,4,5],[6,7,8],[9]]")

    # Test 4: buffer_shuffle
    print("\n--- Test 4: buffer_shuffle ---")
    source = iter(range(20))
    result = buffer_shuffle(source, buffer_size=5, seed=42)
    if result is None:
        print("  NOT IMPLEMENTED YET")
    else:
        shuffled = list(result)
        if len(shuffled) == 20 and sorted(shuffled) == list(range(20)):
            print(f"  PASS  all 20 items present, order changed")
            print(f"  First 10: {shuffled[:10]}")
        else:
            print(f"  FAIL  got {shuffled}")
            print(f"         should contain all numbers 0-19")

    # Test 5: Full Pipeline
    print("\n--- Test 5: Full Pipeline ---")

    if (lazy_filter(iter([1, 2]), lambda x: True) is not None and
            lazy_map(iter([1]), lambda x: x) is not None and
            lazy_batch(iter([1, 2]), 1) is not None):

        print("\nBuilding pipeline (lazy — nothing executes yet)...")

        # Check that building the pipeline doesn't consume data
        data_stream = fake_text_dataset(n=100)

        class CountingIterator:
            """Wrapper to count how many items were consumed."""
            def __init__(self, source):
                self._source = source
                self.consumed = 0
            def __iter__(self):
                return self
            def __next__(self):
                item = next(self._source)
                self.consumed += 1
                return item

        counting_source = CountingIterator(iter(fake_text_dataset(n=100)))

        # Build pipeline
        pipeline = lazy_filter(counting_source, lambda ex: ex["length"] > 50)
        pipeline = lazy_map(pipeline, lambda ex: simple_tokenize(ex, max_len=16))
        pipeline = buffer_shuffle(pipeline, buffer_size=10, seed=42)
        pipeline = lazy_batch(pipeline, batch_size=4)

        consumed_before = counting_source.consumed
        print(f"  Items consumed before iteration: {consumed_before}  (should be 0)")

        # Now iterate — data should flow
        batch_count = 0
        total_examples = 0
        for batch in pipeline:
            batch_count += 1
            total_examples += len(batch)
            if batch_count <= 2:
                topics = [ex["topic"] for ex in batch]
                print(f"  Batch {batch_count}: {len(batch)} examples, topics={topics}")

        print(f"\n  Total batches: {batch_count}")
        print(f"  Total examples: {total_examples}")
        print(f"  Items consumed from source: {counting_source.consumed}")

        if consumed_before == 0:
            print(f"\n  PASS  lazy evaluation confirmed (0 items consumed before iteration)")
        else:
            print(f"\n  FAIL  {consumed_before} items consumed before iteration (should be 0)")

    # BONUS: Memory comparison
    print("\n--- BONUS: Memory Usage: Streaming vs Loading All ---")
    EXAMPLE_SIZE_BYTES = 16 * 4   # 16 tokens × 4 bytes
    BUFFER_SIZE = 1000

    print(f"\n  Buffer size: {BUFFER_SIZE:,} examples")
    print(f"  Example size: {EXAMPLE_SIZE_BYTES} bytes (16 tokens, int32)")
    print(f"\n  {'N Examples':>15} {'Load All (MB)':>15} {'Streaming (KB)':>16} {'Savings':>10}")
    print("  " + "-" * 60)

    for n in [10_000, 100_000, 1_000_000, 100_000_000, 15_000_000_000]:
        load_mb = (n * EXAMPLE_SIZE_BYTES) / 1024**2
        stream_kb = (BUFFER_SIZE * EXAMPLE_SIZE_BYTES) / 1024
        savings = load_mb * 1024 / stream_kb
        label = f"{n:>15,}"
        print(f"  {label} {load_mb:>15.1f} {stream_kb:>16.1f} {savings:>9.0f}x")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def lazy_filter(source, predicate):
#     for example in source:
#         if predicate(example):
#             yield example
#
# def lazy_map(source, transform):
#     for example in source:
#         yield transform(example)
#
# def lazy_batch(source, batch_size):
#     batch = []
#     for example in source:
#         batch.append(example)
#         if len(batch) >= batch_size:
#             yield batch
#             batch = []
#     if batch:
#         yield batch
#
# def buffer_shuffle(source, buffer_size, seed=42):
#     rng = random.Random(seed)
#     buffer = []
#     source = iter(source)
#     for item in source:
#         buffer.append(item)
#         if len(buffer) >= buffer_size:
#             break
#     while buffer:
#         idx = rng.randrange(len(buffer))
#         yield buffer[idx]
#         buffer[idx] = buffer[-1]
#         buffer.pop()
#         try:
#             buffer.append(next(source))
#         except StopIteration:
#             pass

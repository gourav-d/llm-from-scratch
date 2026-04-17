"""
=============================================================================
EXERCISE 3: nanoGPT — Hands-On Practice
=============================================================================

HOW TO USE THIS FILE
---------------------
Each exercise has:
  1. A description of what to do
  2. A code skeleton with TODO comments
  3. A hint (read only if stuck)
  4. A solution (ONLY look after trying yourself!)

Run the file as-is first to see what's already working.
Then fill in each TODO section and re-run to test.

WHAT YOU NEED
  - Python 3.8+
  - numpy  (pip install numpy)
  - No PyTorch needed for this exercise

=============================================================================
"""

import numpy as np
from collections import defaultdict

# =============================================================================
# Setup: Shared code used by all exercises
# =============================================================================

# Training text — you will build models that learn from this
TRAINING_TEXT = (
    "to be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune "
    "or to take arms against a sea of troubles "
)

# Character-level vocabulary
CHARS       = sorted(set(TRAINING_TEXT))
VOCAB_SIZE  = len(CHARS)
CHAR_TO_IDX = {ch: i for i, ch in enumerate(CHARS)}
IDX_TO_CHAR = {i: ch for i, ch in enumerate(CHARS)}

# Helper function: encode text to numbers
def encode(text):
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]

# Helper function: decode numbers back to text
def decode(indices):
    return ''.join(IDX_TO_CHAR.get(i, '?') for i in indices)

ENCODED = encode(TRAINING_TEXT)

print("=" * 60)
print("EXERCISE 3: nanoGPT — Hands-On Practice")
print("=" * 60)
print(f"\nTraining text length : {len(TRAINING_TEXT)} characters")
print(f"Vocabulary size      : {VOCAB_SIZE} unique characters")
print(f"Vocabulary           : {CHARS}\n")

# =============================================================================
# EXERCISE 1: Build a unigram model
# =============================================================================
# DIFFICULTY: Easy
#
# WHAT IS A UNIGRAM?
#   A unigram model ignores context entirely.
#   It just asks: "across the whole training text, which characters appear most?"
#   Then it always samples from that distribution.
#   It is like a bag — it knows what letters exist and how common they are,
#   but has no idea what came before.
#
# YOUR TASK:
#   Build a unigram probability distribution from ENCODED.
#   Then use it to generate 50 characters of text.
#
# EXPECTED: output should contain the same characters as the training text
#           but in essentially random order (no real words).
# =============================================================================

print("=" * 60)
print("EXERCISE 1: Unigram Model (no context)")
print("=" * 60)

def build_unigram(encoded, vocab_size):
    """
    Count how often each character appears and return a probability array.

    encoded    : list of integer token indices (e.g. [5, 2, 1, 3, ...])
    vocab_size : number of unique characters

    Returns    : numpy array of shape (vocab_size,) where
                 result[i] = probability that character i appears
    """
    # Step 1: count how many times each index appears
    counts = np.zeros(vocab_size)           # start all counts at 0

    # TODO: Loop through `encoded` and increment counts[idx] for each idx
    # HINT: a simple for loop, like: for idx in encoded: ...
    for idx in encoded:
        # --- YOUR CODE HERE ---
        pass                                # replace this line with your count code

    # Step 2: add 1 to each count (smoothing — avoids zero probabilities)
    counts = counts + 1

    # Step 3: divide each count by the total to get probabilities
    # TODO: compute probabilities from counts so they sum to 1
    # HINT: probs = counts / counts.sum()
    probs = None   # replace None with your calculation

    return probs


def generate_unigram(probs, length=50):
    """
    Generate `length` characters by sampling from the unigram distribution.

    probs  : probability array from build_unigram
    length : how many characters to generate

    Returns: generated string
    """
    result = []

    for _ in range(length):
        # TODO: sample a random character index using np.random.choice
        # HINT: np.random.choice(len(probs), p=probs)
        idx = None   # replace None with your sample code

        # Convert index back to character
        result.append(IDX_TO_CHAR[idx])

    return ''.join(result)


# --- Run Exercise 1 ---
unigram_probs = build_unigram(ENCODED, VOCAB_SIZE)

if unigram_probs is not None and None not in [unigram_probs]:
    generated_unigram = generate_unigram(unigram_probs, length=50)
    print("\nUnigram generated text (50 chars):")
    print(f"  '{generated_unigram}'")
    print("  (random-looking but uses correct character frequencies)")
else:
    print("  [Complete the TODO sections above to see output]")

"""
---- SOLUTION (only look after trying!) ----

def build_unigram(encoded, vocab_size):
    counts = np.zeros(vocab_size)
    for idx in encoded:
        counts[idx] += 1
    counts = counts + 1
    probs = counts / counts.sum()
    return probs

def generate_unigram(probs, length=50):
    result = []
    for _ in range(length):
        idx = np.random.choice(len(probs), p=probs)
        result.append(IDX_TO_CHAR[idx])
    return ''.join(result)
"""

# =============================================================================
# EXERCISE 2: Experiment with context size
# =============================================================================
# DIFFICULTY: Medium
#
# In Example 3, we built a context-aware model with context_size=4.
# A larger context size means the model looks further back — which generally
# produces more realistic text, but needs more training data to work well.
#
# YOUR TASK:
#   Run the context-aware model from example_03 with THREE different context sizes:
#     context_size = 1  (look at only the last 1 character)
#     context_size = 4  (look at the last 4 characters)
#     context_size = 8  (look at the last 8 characters)
#   Generate 60 characters from each and compare the quality.
#
# EXPECTED: larger context = more words from the training text appear in output.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 2: Experiment with Context Size")
print("=" * 60)

def build_context_model(encoded, vocab_size, context_size):
    """
    Build a context-aware model.

    encoded      : list of token indices
    vocab_size   : number of unique tokens
    context_size : how many previous tokens to look at

    Returns      : dict mapping context tuple -> probability array
    """
    counts = defaultdict(lambda: np.zeros(vocab_size))

    # Walk through the encoded text
    for i in range(len(encoded) - context_size):
        # The context is the previous context_size characters
        context    = tuple(encoded[i : i + context_size])

        # The next character after the context
        next_char  = encoded[i + context_size]

        # Record that this next_char followed this context
        counts[context][next_char] += 1

    # Convert counts to probabilities
    probs = {}
    for ctx, cnt in counts.items():
        total       = cnt.sum()
        probs[ctx]  = (cnt + 1) / (total + vocab_size)   # smoothed

    return probs


def generate_context(model_probs, start_text, context_size, num_chars=60):
    """
    Generate text using a context-aware model.

    model_probs  : dict from build_context_model
    start_text   : seed text (must be >= context_size characters)
    context_size : must match the model
    num_chars    : number of NEW characters to generate

    Returns      : full text including the seed
    """
    # TODO: Implement this function.
    #
    # Steps:
    #   1. Convert start_text to a list of token indices using encode()
    #   2. For each step (up to num_chars):
    #      a. Take the LAST context_size indices as the current context tuple
    #      b. Look up context in model_probs; if not found, use uniform probs
    #      c. Sample the next index using np.random.choice(VOCAB_SIZE, p=probs)
    #      d. Append the sampled index to the sequence
    #   3. Decode the full sequence and return it
    #
    # HINT: Use encode() and decode() defined at the top of this file.

    sequence = encode(start_text)   # start with the encoded seed

    for _ in range(num_chars):
        # TODO: take last context_size tokens as the context tuple
        context = None              # replace with: tuple(sequence[-context_size:])

        # TODO: look up probabilities for this context
        if context in model_probs:
            row = None              # replace with: model_probs[context]
        else:
            row = None              # replace with: np.ones(VOCAB_SIZE) / VOCAB_SIZE

        # TODO: sample next token index
        next_idx = None             # replace with: np.random.choice(VOCAB_SIZE, p=row)

        # TODO: append to sequence
        # sequence.append(next_idx)

    return decode(sequence)


# --- Run Exercise 2 ---
seed = "to be"

print(f"\nSeed text: '{seed}'")
print()
for ctx_size in [1, 4, 8]:
    model = build_context_model(ENCODED, VOCAB_SIZE, ctx_size)
    result = generate_context(model, seed, ctx_size, num_chars=60)
    print(f"  context_size={ctx_size}: '{result}'")

print()
print("Question: Which context size produces the most readable text?")
print("Why? (Think about how much context the model uses to make decisions)")

"""
---- SOLUTION (only look after trying!) ----

def generate_context(model_probs, start_text, context_size, num_chars=60):
    sequence = encode(start_text)
    for _ in range(num_chars):
        context = tuple(sequence[-context_size:])
        if context in model_probs:
            row = model_probs[context]
        else:
            row = np.ones(VOCAB_SIZE) / VOCAB_SIZE
        next_idx = np.random.choice(VOCAB_SIZE, p=row)
        sequence.append(next_idx)
    return decode(sequence)
"""

# =============================================================================
# EXERCISE 3: Add top-k sampling to generation
# =============================================================================
# DIFFICULTY: Medium
#
# WHAT IS TOP-K?
#   Instead of sampling from ALL characters, only sample from the top k
#   most-probable characters. This prevents unlikely (weird) characters
#   from being chosen.
#
# YOUR TASK:
#   Modify the generate_with_topk() function below to:
#     1. Given a probability array, find the top-k entries
#     2. Set all other entries to 0
#     3. Re-normalize so the k entries sum to 1
#     4. Sample from those k entries
#
# Then compare output with k=1, k=3, k=10 using a context_size=4 model.
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 3: Add Top-k Sampling")
print("=" * 60)

def top_k_filter(probs, k):
    """
    Keep only the top-k probabilities. Set all others to zero.
    Re-normalize so the result sums to 1.

    probs : numpy array of probabilities (sums to 1)
    k     : how many top entries to keep
    Returns: filtered probability array (also sums to 1)
    """
    # TODO: implement top-k filtering
    #
    # HINT (step by step):
    #   1. Find the k-th largest value:
    #      sorted_probs = np.sort(probs)[::-1]  # sort descending
    #      threshold = sorted_probs[k - 1]       # value at position k-1
    #
    #   2. Zero out all values below the threshold:
    #      filtered = np.where(probs >= threshold, probs, 0.0)
    #
    #   3. Re-normalize:
    #      filtered = filtered / filtered.sum()
    #
    #   4. Return filtered

    filtered = probs.copy()   # start with original (replace with your code)

    # --- YOUR CODE HERE ---

    return filtered


def generate_topk(model_probs, start_text, context_size, k, num_chars=60):
    """Generate text using the context model with top-k sampling."""
    sequence = encode(start_text)

    for _ in range(num_chars):
        context = tuple(sequence[-context_size:])

        if context in model_probs:
            raw_probs = model_probs[context]
        else:
            raw_probs = np.ones(VOCAB_SIZE) / VOCAB_SIZE

        # Apply top-k filter
        filtered_probs = top_k_filter(raw_probs, k=k)

        # Sample
        next_idx = np.random.choice(VOCAB_SIZE, p=filtered_probs)
        sequence.append(next_idx)

    return decode(sequence)


# --- Run Exercise 3 ---
model_4 = build_context_model(ENCODED, VOCAB_SIZE, context_size=4)

print(f"\nSeed: 'to be', using context_size=4")
for k in [1, 3, 10]:
    result = generate_topk(model_4, "to be", context_size=4, k=k, num_chars=60)
    print(f"  k={k:>2}: '{result}'")

print()
print("Question: What happens at k=1? What happens at k=10?")
print("Why does k=1 produce the same text every time?")

"""
---- SOLUTION ----

def top_k_filter(probs, k):
    sorted_probs = np.sort(probs)[::-1]
    threshold    = sorted_probs[k - 1]
    filtered     = np.where(probs >= threshold, probs, 0.0)
    total        = filtered.sum()
    if total > 0:
        filtered = filtered / total
    else:
        filtered = np.ones(len(probs)) / len(probs)
    return filtered
"""

# =============================================================================
# EXERCISE 4 (Bonus): Train on your own text
# =============================================================================
# DIFFICULTY: Easy (mostly changing one variable)
#
# YOUR TASK:
#   Replace CUSTOM_TEXT below with any text you like, then run the cell.
#   Good options:
#     - A nursery rhyme you know
#     - A few lines from a song
#     - A short paragraph from a book
#     - Python code (watch it try to write code!)
#
#   Then look at the generated output and ask:
#     - Does the model capture the style of your text?
#     - What context size works best?
# =============================================================================

print("\n" + "=" * 60)
print("EXERCISE 4 (Bonus): Train on Your Own Text")
print("=" * 60)

# TODO: Replace this with your own text!
CUSTOM_TEXT = (
    "jack and jill went up the hill to fetch a pail of water "
    "jack fell down and broke his crown and jill came tumbling after "
    "up jack got and home did trot as fast as he could caper "
    "he went to bed to mend his head with vinegar and brown paper "
)

# Build vocabulary from custom text
custom_chars    = sorted(set(CUSTOM_TEXT))
custom_vocab    = len(custom_chars)
custom_c2i      = {ch: i for i, ch in enumerate(custom_chars)}
custom_i2c      = {i: ch for i, ch in enumerate(custom_chars)}
custom_encoded  = [custom_c2i[ch] for ch in CUSTOM_TEXT]

# Build and sample from model
custom_model = build_context_model(custom_encoded, custom_vocab, context_size=4)

seed_char = "jack"
seed_idx  = [custom_c2i[ch] for ch in seed_char if ch in custom_c2i]

sequence = list(seed_idx)
for _ in range(80):
    ctx = tuple(sequence[-4:])
    if ctx in custom_model:
        row = custom_model[ctx]
    else:
        row = np.ones(custom_vocab) / custom_vocab
    next_idx = np.random.choice(custom_vocab, p=row)
    sequence.append(next_idx)

generated_custom = ''.join(custom_i2c[i] for i in sequence)
print(f"\nCustom text model — starting with '{seed_char}':")
print(f"  '{generated_custom}'")
print()
print("TRY: Change CUSTOM_TEXT to Python code and use seed_char='def '")

print("\n" + "=" * 60)
print("Exercise 3 Complete!")
print()
print("KEY TAKEAWAYS:")
print("  - Unigram: just character frequencies — completely random order")
print("  - Bigram/context: captures short patterns — recognizable fragments")
print("  - Larger context = better, but needs more training data")
print("  - Top-k prevents weird characters from being chosen")
print("  - Real GPT does the same thing, but with ATTENTION instead of")
print("    a lookup table — attention learns WHICH past tokens matter most")
print("=" * 60)

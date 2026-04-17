"""
=============================================================================
EXAMPLE 3: nanoGPT - Building a Tiny Language Model from Scratch
=============================================================================

GLOSSARY (read this first, before looking at any code)
-------------------------------------------------------
Language Model : A program that predicts the NEXT word/character given
                 the ones before it. Like autocomplete on your phone.

Token          : A piece of text the model works with.
                 Could be a character, a word, or a sub-word.
                 e.g. "hello" -> characters: ['h','e','l','l','o']

Vocabulary     : The complete list of unique tokens the model knows.
                 e.g. for character-level: ['a','b','c',...,'z',' ']

Embedding      : A list of numbers that represents a token.
                 e.g. 'h' might be represented as [0.2, -0.1, 0.5]
                 Why? Numbers are what math can work with.

Logits         : Raw scores for each possible next token.
                 Higher score = more likely to be next.

Softmax        : Converts raw scores -> probabilities that add up to 1.
                 e.g. [2.0, 1.0, 0.5] -> [0.59, 0.24, 0.16]

Loss           : A number that measures how WRONG the model is.
                 Low loss = model is predicting well.
                 High loss = model is guessing poorly.

Training       : Adjusting the model's numbers (weights) to reduce loss.

=============================================================================
PART A: The Simplest Possible Language Model (Bigram)
=============================================================================

What is a Bigram Model?
  - Bigram = "two characters"
  - It predicts the next character based ONLY on the current character.
  - This is the simplest possible language model.
  - GPT does the same thing, but looks at MANY previous characters.

We will teach this model ONE sentence:
  "hello"

After training, it should know:
  h -> e  (h is usually followed by e)
  e -> l  (e is usually followed by l)
  l -> l  (l can be followed by l)
  l -> o  (l can also be followed by o)

=============================================================================
"""

import numpy as np  # numpy: the math library (like System.Math in C#)

print("=" * 60)
print("PART A: Simplest Language Model (Bigram)")
print("=" * 60)

# =============================================================================
# STEP 1: Prepare the text and build a vocabulary
# =============================================================================

# Our training text (intentionally tiny so you can follow every step)
text = "hello"

print(f"\nTraining text: '{text}'")
print(f"Length: {len(text)} characters")

# Build vocabulary: find all unique characters
# sorted() puts them in alphabetical order so results are consistent
chars = sorted(set(text))         # set() removes duplicates
vocab_size = len(chars)           # how many unique characters we have

print(f"\nUnique characters found: {chars}")
print(f"Vocabulary size: {vocab_size}")

# Create lookup dictionaries
# char_to_idx: converts a character to a number
# e.g. 'e' -> 1, 'h' -> 2, 'l' -> 3, 'o' -> 4
char_to_idx = {ch: i for i, ch in enumerate(chars)}

# idx_to_char: converts a number back to a character
# e.g. 1 -> 'e', 2 -> 'h', 3 -> 'l', 4 -> 'o'
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"\nCharacter -> Number mapping: {char_to_idx}")
print(f"Number -> Character mapping: {idx_to_char}")

# =============================================================================
# STEP 2: Convert text to numbers
# =============================================================================

# Encode: turn the text into a list of numbers
encoded = [char_to_idx[ch] for ch in text]   # list comprehension (like LINQ Select)
print(f"\n'hello' encoded as numbers: {encoded}")
# e.g. [2, 1, 3, 3, 4]  (h=2, e=1, l=3, l=3, o=4)

# =============================================================================
# STEP 3: Create training pairs
# =============================================================================

# For a bigram model:
#   Input (x):  current character
#   Target (y): NEXT character
#
# From "hello":
#   'h' -> 'e'  (see 'h', predict 'e')
#   'e' -> 'l'  (see 'e', predict 'l')
#   'l' -> 'l'  (see 'l', predict 'l')
#   'l' -> 'o'  (see 'l', predict 'o')

print("\nTraining pairs (input -> target):")
for i in range(len(encoded) - 1):              # stop 1 before the end
    x = encoded[i]                             # current character (as number)
    y = encoded[i + 1]                         # next character (as number)
    print(f"  '{idx_to_char[x]}' ({x}) -> '{idx_to_char[y]}' ({y})")

# =============================================================================
# STEP 4: The Bigram "Model" - a lookup table
# =============================================================================

# The simplest possible model: a table of counts
# counts[i][j] = "how often does character i appear right before character j?"
#
# C# equivalent: int[,] counts = new int[vocab_size, vocab_size];

counts = np.zeros((vocab_size, vocab_size))    # start everything at 0

# Fill the table by counting what follows what
for i in range(len(encoded) - 1):
    x = encoded[i]                             # current character index
    y = encoded[i + 1]                         # next character index
    counts[x][y] += 1                          # record that y followed x

print("\nCount table (rows=current char, cols=next char):")
print("     ", end="")
for ch in chars:
    print(f"{ch:>5}", end="")                  # print column headers
print()
for i, ch in enumerate(chars):
    print(f"{ch:>4} ", end="")                 # print row label
    for j in range(vocab_size):
        print(f"{int(counts[i][j]):>5}", end="")
    print()

# =============================================================================
# STEP 5: Convert counts to probabilities
# =============================================================================

# Probabilities: for each character, what is the chance each character follows?
# We do this by dividing each row by its row total.
#
# Example: row for 'l' has counts [0, 0, 1, 1]
#          total = 2
#          probabilities = [0, 0, 0.5, 0.5]  -> 50% chance of 'l', 50% of 'o'

# Add 1 to every count to avoid division by zero (called "smoothing")
probs = (counts + 1) / (counts + 1).sum(axis=1, keepdims=True)
# axis=1 means sum across columns (each row)
# keepdims=True keeps the shape so division works correctly

print("\nProbability table (rows=current char, cols=next char):")
print("        ", end="")
for ch in chars:
    print(f"{ch:>7}", end="")
print()
for i, ch in enumerate(chars):
    print(f"{ch:>6}  ", end="")
    for j in range(vocab_size):
        print(f"{probs[i][j]:>7.3f}", end="")
    print()

# =============================================================================
# STEP 6: Generate text using the model
# =============================================================================

def generate_bigram(probs, idx_to_char, start_char, num_chars=10):
    """
    Generate text using the bigram model.

    start_char : the character to start with (e.g. 'h')
    num_chars  : how many characters to generate
    """
    # Convert start character to index
    current_idx = char_to_idx[start_char]
    result = [start_char]                      # start with the given character

    for _ in range(num_chars - 1):
        # Look up the probability row for the current character
        row = probs[current_idx]               # e.g. if current is 'h': [0, 0.8, 0.05, 0.05, 0.1]

        # np.random.choice: pick a random index, weighted by probabilities
        # This is like rolling a weighted dice
        next_idx = np.random.choice(len(row), p=row)

        # Convert index back to character
        next_char = idx_to_char[next_idx]
        result.append(next_char)               # add to output

        current_idx = next_idx                 # move forward

    return ''.join(result)                     # join list of chars into one string

print("\nGenerating text starting with 'h':")
for trial in range(5):                         # run 5 times to see variety
    generated = generate_bigram(probs, idx_to_char, start_char='h', num_chars=8)
    print(f"  Trial {trial + 1}: '{generated}'")

print("\nNote: it learned that 'h' is usually followed by 'e', etc.")
print("But it doesn't look at more than 1 previous character (that's the GPT advantage)")

# =============================================================================
print("\n" + "=" * 60)
print("PART B: nanoGPT - Looking at Multiple Previous Characters")
print("=" * 60)
# =============================================================================

"""
PART B: What makes GPT better than Bigram?

The bigram model looks at ONLY the previous 1 character.
GPT looks at ALL previous characters using ATTENTION.

Example:
  Text: "The bank of the river..."
  Bigram sees: 'r' after 'e' (in "river") — no context
  GPT sees:    all of "The bank of the" -> knows "bank" means riverbank, not money bank

In this part, we build a simplified version that considers
a WINDOW of previous characters (called the "context").

We train it on a nursery rhyme so you can see it actually learn something!
"""

print("\nTraining text: a nursery rhyme")
print("The model will learn to continue it after seeing a few words.")
print()

# Training text — short enough to run fast, but long enough to be interesting
training_text = (
    "mary had a little lamb little lamb little lamb "
    "mary had a little lamb its fleece was white as snow "
    "and everywhere that mary went mary went mary went "
    "and everywhere that mary went the lamb was sure to go"
)

print(f"Training text ({len(training_text)} chars):")
print(f"  '{training_text[:60]}...'")

# -------------------------------------------------------------------------
# Step B1: Build vocabulary from the training text
# -------------------------------------------------------------------------

chars_b = sorted(set(training_text))          # all unique characters
vocab_size_b = len(chars_b)                   # number of unique chars
char_to_idx_b = {ch: i for i, ch in enumerate(chars_b)}
idx_to_char_b = {i: ch for i, ch in enumerate(chars_b)}
encoded_b = [char_to_idx_b[ch] for ch in training_text]

print(f"\nVocabulary ({vocab_size_b} unique chars): {''.join(chars_b)!r}")

# -------------------------------------------------------------------------
# Step B2: Context-aware model
# -------------------------------------------------------------------------

# CONTEXT SIZE = how many previous characters we look at
# Bigram: context_size = 1  (look at 1 previous char)
# GPT-3:  context_size = 2048 (look at 2048 previous tokens!)
# Ours:   context_size = 4   (look at 4 previous chars)

context_size = 4          # how many chars to look back

print(f"\nContext size: {context_size}")
print("(GPT looks at the previous N characters/tokens to decide what comes next)")
print()

# Build a table: context -> count of what follows
# A context is a tuple of N characters, e.g. ('m', 'a', 'r', 'y')
from collections import defaultdict   # defaultdict: like Dictionary with a default value

# context_counts[context][next_char_idx] = count of times this happened
context_counts = defaultdict(lambda: np.zeros(vocab_size_b))

# Walk through the text and record context -> next character
for i in range(len(encoded_b) - context_size):
    # The context is the previous `context_size` characters
    context = tuple(encoded_b[i : i + context_size])   # e.g. (12, 0, 17, 24)

    # The target is the character AFTER the context
    next_char_idx = encoded_b[i + context_size]

    # Count it
    context_counts[context][next_char_idx] += 1

print(f"Learned {len(context_counts)} unique contexts from the training text")

# Convert counts to probabilities (same as Part A)
context_probs = {}
for ctx, cnt in context_counts.items():
    total = cnt.sum()
    if total > 0:
        context_probs[ctx] = (cnt + 1) / (cnt.sum() + vocab_size_b)   # smoothed
    else:
        context_probs[ctx] = np.ones(vocab_size_b) / vocab_size_b     # uniform if unseen

# -------------------------------------------------------------------------
# Step B3: Generate text using context
# -------------------------------------------------------------------------

def generate_context(context_probs, char_to_idx, idx_to_char,
                     start_text, context_size, num_chars=50):
    """
    Generate text using the context-aware model.

    start_text   : seed text to start with (must be >= context_size chars)
    context_size : how many previous chars to look at
    num_chars    : how many NEW characters to generate
    """
    # Start with the seed text (encoded as numbers)
    result = [char_to_idx.get(ch, 0) for ch in start_text]

    for _ in range(num_chars):
        # Take the last `context_size` characters as the current context
        ctx = tuple(result[-context_size:])

        if ctx in context_probs:
            # We've seen this context before — use learned probabilities
            row = context_probs[ctx]
        else:
            # Unseen context — guess equally
            row = np.ones(len(idx_to_char)) / len(idx_to_char)

        # Sample the next character
        next_idx = np.random.choice(len(row), p=row)
        result.append(next_idx)                    # add to the sequence

    # Convert all indices back to characters and join
    return ''.join(idx_to_char[i] for i in result)

# Try it out
print("\n--- Generated text (context-aware model) ---")
seed = "mary"
for trial in range(3):
    generated = generate_context(
        context_probs, char_to_idx_b, idx_to_char_b,
        start_text=seed,
        context_size=context_size,
        num_chars=60
    )
    print(f"\nTrial {trial + 1} (seed='{seed}'):")
    print(f"  {generated}")

# -------------------------------------------------------------------------
# Step B4: Show the difference
# -------------------------------------------------------------------------

print("\n" + "-" * 60)
print("SUMMARY: Bigram vs Context-aware vs Real GPT")
print("-" * 60)
print("""
  Bigram model (Part A):
    - Looks at: 1 previous character
    - Math:     a 5x5 probability table
    - Result:   somewhat random, short-range patterns

  Context model (Part B):
    - Looks at: 4 previous characters
    - Math:     a dictionary of probability tables
    - Result:   better, can repeat phrases from training text

  Real GPT:
    - Looks at: thousands of previous tokens
    - Math:     ATTENTION mechanism (learns which parts matter most)
    - Result:   coherent, long-range understanding

  The attention mechanism is what lets GPT look back 2048 tokens
  and figure out WHICH ones are relevant — not just the last N.
  That is what Lesson 3 is really about!
""")

print("=" * 60)
print("Example 3 complete!")
print("Next: Run exercise_03_nanogpt.py to practice")
print("=" * 60)

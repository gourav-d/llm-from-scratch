"""
=============================================================================
PROJECT 3: Smart Autocomplete - Build Your Own Text Completion System
=============================================================================

GLOSSARY (read this first, before looking at any code)
-------------------------------------------------------
Autocomplete    : A system that predicts what you will type next.
                  You see it in Google Search, VS Code, and phone keyboards.

Bigram          : A pair of consecutive words, e.g. "the quick" or "fox jumps".
                  "bi" = two. A bigram model looks at ONE previous word to predict
                  the next word.
                  C# analogy: like a Dictionary<string, Dictionary<string, int>>
                              where the outer key is the current word, the inner
                              key is the next word, and the value is the count.

Frequency       : How often something appears. Word frequency = count of that
                  word in your training text.

Vocabulary      : The complete set of unique words the model knows.
                  Like a Dictionary<string, int> mapping word -> index number.

Token           : A unit of text the model processes. Here each word is one token.
                  In real LLMs, tokens are parts of words (e.g. "running" -> ["run","ning"]).

Embedding       : A list of numbers that represents a word as a point in space.
                  Words with similar meanings end up close together.
                  e.g. "king" -> [0.2, -0.5, 1.1, 0.8]
                  C# analogy: like a float[] array that IS the word in math-space.

Linear Layer    : A math operation: output = input * weights + bias.
                  Also called "fully connected" or "dense" layer.
                  C# analogy: like a function y = m*x + b from school algebra,
                              but for arrays instead of single numbers.

Logits          : Raw scores from the model for every possible next word.
                  NOT probabilities yet. Any number, positive or negative.

Softmax         : Converts logits into probabilities that sum to 1.
                  Formula: exp(x_i) / sum(exp(x_j) for all j)

Loss            : A single number measuring how WRONG the model is right now.
                  We want to make this number as small as possible (minimize loss).
                  C# analogy: like a "score" in a game, but lower = better.

Cross-Entropy   : A specific loss function for classification problems.
                  Punishes the model more when it is very confident and very wrong.

Backpropagation : The algorithm that figures out HOW MUCH each weight contributed
                  to the loss. Then we nudge each weight in the right direction.
                  Short form: "backprop". This is how neural networks learn.

Optimizer       : Code that updates the weights based on what backprop computed.
                  Adam is the most popular optimizer (Adaptive Moment estimation).
                  C# analogy: like an automatic "parameter tuner" for your model.

Epoch           : One full pass through ALL your training data.
                  After 100 epochs, the model has seen the data 100 times.

Greedy          : Always pick the word with the HIGHEST probability. No randomness.

Temperature     : Divides logits before softmax. Controls how "surprising" output is.
                  temperature < 1 -> focused, repetitive.
                  temperature > 1 -> creative, random.

Top-k           : Only consider the k most likely next words. Ignore all others.

Top-p           : Only consider the smallest group of words whose probabilities
                  add up to p (e.g. 0.9 = 90%). Called "nucleus sampling".

=============================================================================
PROJECT OVERVIEW
=============================================================================

In this project you will build TWO versions of autocomplete:

  PART A - Frequency-Based (no ML)
  ---------------------------------
  Count how often each word follows another word in the training text.
  When asked for a completion, return the most frequent next word.
  This is how autocomplete worked in the 1990s and early 2000s.

  PART B - ML-Powered (tiny neural network)
  ------------------------------------------
  Train a tiny word-level neural network on the same text.
  Use ALL 4 generation strategies from Lesson 5:
    1. Greedy
    2. Temperature sampling
    3. Top-k sampling
    4. Top-p (nucleus) sampling

  By comparing both, you will understand WHAT ML adds over simple counting.

TRAINING TEXT (hard-coded):
  "the quick brown fox jumps over the lazy dog"
  "the fox ran quickly through the forest"
  "the dog chased the fox around the garden"
  "the quick cat sat on the mat in the garden"

=============================================================================
"""

# =============================================================================
# IMPORTS
# We import only standard Python + numpy in Part A.
# PyTorch is imported in Part B inside a try/except so the file still runs
# even if PyTorch is not installed.
# C# analogy: like conditional "using" statements (not possible in C#, but
#             Python lets you try to import and catch the error gracefully).
# =============================================================================

import random          # built-in Python module for random number generation
import numpy as np     # numerical computing library (like Math class on steroids)
from collections import defaultdict  # dict that creates missing keys automatically

# Fix the random seed so results are repeatable every time you run this file.
# C# analogy: like new Random(42) in C# -- same seed = same sequence.
random.seed(42)
np.random.seed(42)

# =============================================================================
# SHARED TRAINING TEXT
# Both Part A and Part B use exactly this text.
# We put it here at the top so it is defined once and reused everywhere.
# =============================================================================

# Each string is one "sentence" of training data
TRAINING_SENTENCES = [
    "the quick brown fox jumps over the lazy dog",   # classic pangram
    "the fox ran quickly through the forest",         # introduces new words
    "the dog chased the fox around the garden",       # more dog/fox pairs
    "the quick cat sat on the mat in the garden",    # adds cat/mat/on/in
]

# Join all sentences into one long string, then split on spaces to get words.
# Python note: " ".join(list) is like String.Join(" ", list) in C#.
# .split() with no argument splits on any whitespace.
ALL_WORDS_FLAT = " ".join(TRAINING_SENTENCES).split()   # flat list of every word


# =============================================================================
# PRINT HELPERS
# We use simple ASCII characters only (cp1252-safe).
# NO Unicode arrows, box-drawing, or special characters.
# =============================================================================

def section(title):
    """Print a bold section header using = characters."""
    print()
    print("=" * 60)             # 60 equals signs as a divider
    print(title)                 # print the title text
    print("=" * 60)             # another 60 equals signs

def subsection(title):
    """Print a smaller sub-section header using - characters."""
    print()
    print(title)                 # print the title
    print("-" * len(title))     # underline with dashes (same length as title)


# =============================================================================
# =============================================================================
#
#  PART A: FREQUENCY-BASED AUTOCOMPLETE (NO MACHINE LEARNING)
#
# =============================================================================
# =============================================================================

section("PART A: Frequency-Based Autocomplete (No Machine Learning)")

print("""
HOW IT WORKS (no code yet):
  1. Read every consecutive word pair in the training text.
  2. Store counts: after "the", how often does "quick" appear? "fox"? "dog"?
  3. For any word you type, look up its most frequent follower.

This is a BIGRAM MODEL: it only looks at the 1 previous word.
A trigram model would look at 2 previous words, etc.

C# analogy:
  Dictionary<string, Dictionary<string, int>> bigrams;
  bigrams["the"]["quick"] = 2;
  bigrams["the"]["fox"]   = 3;
  ...
""")

# =============================================================================
# STEP A1: BUILD THE BIGRAM TABLE
# A bigram table maps: word -> {next_word: count}
# =============================================================================

subsection("Step A1: Building the Bigram Table")

# defaultdict(lambda: defaultdict(int)) means:
#   - If we access a key that does not exist, create an empty inner dict.
#   - If we access an inner key that does not exist, start count at 0.
# C# analogy: like using ConcurrentDictionary with GetOrAdd().
bigram_counts = defaultdict(lambda: defaultdict(int))

# Loop through every consecutive pair of words in the training text.
# zip(ALL_WORDS_FLAT, ALL_WORDS_FLAT[1:]) pairs each word with the next word.
# C# analogy: for (int i = 0; i < words.Length - 1; i++) { ... }
for current_word, next_word in zip(ALL_WORDS_FLAT, ALL_WORDS_FLAT[1:]):
    bigram_counts[current_word][next_word] += 1   # increment the count for this pair

# Print the table so we can see what was learned
print("Bigram counts (how often word2 follows word1):")
print()

# Print a header row for the table
print(f"  {'Word':<12} {'Follows word':<15} {'Count':>5}")  # f-string for alignment
print("  " + "-" * 35)                                       # separator line

# Sort the outer keys alphabetically for readable output
for word in sorted(bigram_counts.keys()):                    # sorted() = alphabetical order
    for next_w, count in sorted(bigram_counts[word].items(), # sort inner dict by count
                                key=lambda x: -x[1]):        # lambda sorts descending by count
        print(f"  {word:<12} {next_w:<15} {count:>5}")       # print one row

# =============================================================================
# STEP A2: THE PREDICTION FUNCTION
# Given the last word typed, return the most likely next word.
# =============================================================================

subsection("Step A2: Predict the Most Likely Next Word")

def predict_next_word(word, bigram_table, top_n=3):
    """
    Given a word, return the top_n most likely next words.

    Parameters:
      word         - the last word typed by the user (string)
      bigram_table - our defaultdict of counts
      top_n        - how many suggestions to return (default 3)

    Returns:
      A list of (next_word, count) tuples, sorted by count descending.
      Returns an empty list if the word was never seen in training.

    C# analogy:
      Like a method List<(string, int)> PredictNext(string word, int topN)
    """
    if word not in bigram_table:        # word never appeared in training text
        return []                        # return empty list (no suggestions)

    # Get the inner dictionary of {next_word: count} for this word
    followers = bigram_table[word]

    # Sort the followers by count, descending.
    # .items() gives (key, value) pairs -- like .ToDictionary() in LINQ.
    # sorted() with key=lambda x: -x[1] sorts by the count (index 1), descending.
    sorted_followers = sorted(followers.items(), key=lambda x: -x[1])

    # Return only the top_n results (slice the list)
    # C# analogy: .Take(top_n) in LINQ
    return sorted_followers[:top_n]

# Test the prediction function on a few words
test_words = ["the", "fox", "quick", "garden", "cat"]   # words to test

print()
print("Autocomplete suggestions (frequency-based):")
print()

for test_word in test_words:                             # loop over each test word
    suggestions = predict_next_word(test_word, bigram_counts, top_n=3)  # get suggestions
    if suggestions:                                      # if we found any suggestions
        # Format the suggestions as "word (count)" strings
        # C# analogy: String.Join(", ", suggestions.Select(s => $"{s.Item1} ({s.Item2})"))
        suggestion_text = ", ".join(f"{w} ({c})" for w, c in suggestions)
        print(f"  '{test_word}' -> {suggestion_text}")
    else:                                                # no suggestions found
        print(f"  '{test_word}' -> [no suggestion -- word not in training data]")

# =============================================================================
# STEP A3: COMPLETE A PHRASE
# We chain predictions together to complete a multi-word phrase.
# =============================================================================

subsection("Step A3: Complete a Phrase (Chained Predictions)")

def complete_phrase_freq(start_word, bigram_table, num_words=5):
    """
    Generate a completion by chaining bigram predictions.

    At each step:
      - Take the last word generated.
      - Look up its most frequent follower (greedy).
      - Append that word and repeat.

    Parameters:
      start_word   - the word to start from (string)
      bigram_table - our bigram count table
      num_words    - how many additional words to generate

    Returns:
      A list of words (the generated continuation).
    """
    generated = [start_word]                             # start with the seed word

    current = start_word                                 # track the current word
    for _ in range(num_words):                           # _ means "I don't need the loop variable"
        suggestions = predict_next_word(current, bigram_table, top_n=1)  # get top-1 suggestion
        if not suggestions:                              # if no suggestion found, stop
            break                                        # exit the loop early
        next_w = suggestions[0][0]                       # get just the word (ignore count)
        generated.append(next_w)                         # add to our growing sentence
        current = next_w                                 # move forward one step

    return generated                                     # return the full list of words

print()
print("Generating completions (always picks most frequent next word):")
print()

# Test with different seed words
seed_tests = ["the", "fox", "quick", "dog"]             # different starting points

for seed in seed_tests:                                  # loop over each seed
    result = complete_phrase_freq(seed, bigram_counts, num_words=5)  # generate
    # Join list into a sentence string
    sentence = " ".join(result)                          # ["the","fox"] -> "the fox"
    print(f"  Seed: '{seed}' -> \"{sentence}\"")

print()
print("OBSERVATION:")
print("  The frequency model always generates the same output for the same input.")
print("  It has NO randomness and NO memory beyond the last 1 word.")
print("  This is the limitation that ML fixes.")


# =============================================================================
# =============================================================================
#
#  PART B: ML-POWERED AUTOCOMPLETE (TINY NEURAL NETWORK)
#
# =============================================================================
# =============================================================================

section("PART B: ML-Powered Autocomplete (Neural Network)")

print("""
HOW IT WORKS (concept before code):
  1. Build a vocabulary: assign each unique word an integer ID.
  2. Create a tiny neural network:
       - Embedding layer: turns a word ID into a list of numbers (a vector).
       - Linear layer:    turns that vector into scores for every word in vocab.
  3. Train it: feed word pairs (input word -> target next word),
               compute loss, do backpropagation, update weights.
  4. Generate text using all 4 sampling strategies.

WHY IS THIS BETTER THAN FREQUENCY COUNTING?
  - The network learns PATTERNS, not just exact word pairs it has seen.
  - Embeddings let it know that "fox" and "cat" are similar (both animals).
  - It can generalize to word combinations it never saw in training.
  - Temperature, top-k, top-p give us CONTROL over creativity.
""")

# =============================================================================
# STEP B1: BUILD THE VOCABULARY
# Map each unique word to an integer index and back.
# =============================================================================

subsection("Step B1: Build Vocabulary")

# Get all unique words from the training text.
# set() removes duplicates -- like HashSet<string> in C#.
# sorted() puts them in alphabetical order (for consistent IDs across runs).
unique_words = sorted(set(ALL_WORDS_FLAT))              # sorted unique words

# vocab: maps word (string) -> integer index
# C# analogy: Dictionary<string, int> vocab = new();
vocab = {word: idx for idx, word in enumerate(unique_words)}  # dict comprehension

# reverse_vocab: maps integer index -> word (for decoding predictions)
# C# analogy: Dictionary<int, string> reverseVocab = new();
reverse_vocab = {idx: word for word, idx in vocab.items()}     # inverted mapping

# Size of our vocabulary (how many unique words we have)
vocab_size = len(vocab)                                  # integer count

print(f"  Unique words found: {vocab_size}")
print()
print("  Word-to-ID mapping (vocabulary):")
print()

# Print 3 words per line to keep output compact
words_per_row = 3                                        # how many to print per line
word_list = list(vocab.items())                          # convert dict to list of tuples

for i in range(0, len(word_list), words_per_row):        # step through with stride=3
    row = word_list[i:i + words_per_row]                 # slice 3 items at a time
    # Format each as "word=ID" and join with spaces
    row_text = "   ".join(f"{w}={idx:2d}" for w, idx in row)
    print(f"    {row_text}")                              # print the row

# Convert training text to integer token IDs.
# This is how all real LLMs store text: as integers, not strings.
# C# analogy: int[] tokenIds = words.Select(w => vocab[w]).ToArray();
token_ids = [vocab[w] for w in ALL_WORDS_FLAT]           # list of integers

print()
print(f"  Training text as token IDs (first 15):")
print(f"    {token_ids[:15]}")                           # show first 15 for brevity

# =============================================================================
# STEP B2: BUILD TRAINING PAIRS
# Each training pair is (input_word_id, target_next_word_id).
# =============================================================================

subsection("Step B2: Build Training Pairs")

# A training pair: given token_ids[i], predict token_ids[i+1].
# This is called "next-word prediction" or "causal language modelling".
# C# analogy: List<(int input, int target)> pairs = new();
input_ids  = token_ids[:-1]   # all tokens EXCEPT the last one (these are inputs)
target_ids = token_ids[1:]    # all tokens EXCEPT the first one (these are targets)

print(f"  Total training pairs: {len(input_ids)}")
print()
print("  First 5 training pairs:")
print()
print(f"  {'Input word':<15} {'Target (next) word':<20} {'IDs'}")
print("  " + "-" * 50)
for i in range(5):                                       # show first 5 pairs
    in_word  = reverse_vocab[input_ids[i]]               # decode integer back to word
    out_word = reverse_vocab[target_ids[i]]              # decode target word
    print(f"  {in_word:<15} {out_word:<20} ({input_ids[i]:2d} -> {target_ids[i]:2d})")

# =============================================================================
# STEP B3: DEFINE SOFTMAX (shared by both PyTorch-free and PyTorch paths)
# =============================================================================

def softmax_np(x):
    """
    Convert raw scores (logits) to probabilities using numpy.
    Subtracting the max before exp() prevents numeric overflow.
    This is the same formula as in Example 5.

    C# analogy: like a static helper method that takes double[] and returns double[].
    """
    e = np.exp(x - x.max())      # subtract max for numerical stability (prevents inf)
    return e / e.sum()            # divide each by sum so all values add up to 1.0

# =============================================================================
# STEP B4: DEFINE SAMPLING STRATEGIES (numpy only, no PyTorch needed)
# These mirror the strategies from Example 5 / Lesson 5.
# =============================================================================

subsection("Step B4: Sampling Strategy Functions")

def greedy_pick(logits):
    """
    Strategy 1 - Greedy: always pick the word with the highest score.
    No randomness. Same input always gives same output.

    C# analogy: Array.IndexOf(logits, logits.Max())
    """
    return int(np.argmax(logits))      # index of the maximum value in logits

def temperature_pick(logits, temperature):
    """
    Strategy 2 - Temperature sampling: divide logits by temperature, then sample.

    temperature < 1.0  -> more focused (top word more likely)
    temperature = 1.0  -> unchanged (model's own probabilities)
    temperature > 1.0  -> more random (all words become more equal)

    C# analogy: like scaling a probability distribution before drawing from it.
    """
    if temperature <= 0:                           # temperature of 0 = pure greedy
        return greedy_pick(logits)                 # avoid division by zero
    scaled = logits / temperature                  # divide ALL logits by temperature
    probs  = softmax_np(scaled)                    # convert scaled logits to probabilities
    # np.random.choice picks a random index weighted by probs
    # C# analogy: like WeightedRandom.Next(probs)
    return int(np.random.choice(len(probs), p=probs))

def top_k_pick(logits, k, temperature=1.0):
    """
    Strategy 3 - Top-k sampling: only consider the k most likely words.
    All other words get probability 0.

    Steps:
      1. Apply temperature.
      2. Find the k-th largest logit value (the threshold).
      3. Set all logits below threshold to -infinity (excluded from softmax).
      4. Sample from the remaining k words.

    C# analogy: like LINQ .OrderByDescending().Take(k) then sample from those.
    """
    k = min(k, len(logits))                         # k cannot exceed vocab size
    scaled = logits / max(temperature, 1e-8)         # apply temperature (avoid div by 0)
    sorted_logits = np.sort(scaled)[::-1]            # sort descending
    threshold     = sorted_logits[k - 1]             # value of k-th largest
    # Where logit >= threshold, keep it; otherwise replace with -infinity
    # -infinity becomes 0 after softmax, so those words are never picked
    filtered = np.where(scaled >= threshold, scaled, -np.inf)
    probs    = softmax_np(filtered)                  # re-normalize among survivors
    probs    = np.nan_to_num(probs, nan=0.0)         # replace NaN with 0
    if probs.sum() == 0:                             # safety: if all zeros, use uniform
        probs = np.ones(len(probs)) / len(probs)
    return int(np.random.choice(len(probs), p=probs))

def top_p_pick(logits, p, temperature=1.0):
    """
    Strategy 4 - Top-p (nucleus) sampling: keep the smallest group of words
    whose cumulative probability reaches p.

    Steps:
      1. Apply temperature.
      2. Sort words by probability, highest first.
      3. Accumulate probabilities until the running total >= p.
      4. That group is the "nucleus". Sample from it.

    WHY BETTER THAN TOP-K?
      Top-k always uses exactly k words.
      Top-p uses as many words as needed -- fewer when confident, more when unsure.

    C# analogy: like stopping a foreach loop when a running total crosses a threshold.
    """
    scaled    = logits / max(temperature, 1e-8)      # apply temperature
    sort_idx  = np.argsort(scaled)[::-1]             # indices sorted by logit, descending
    sorted_l  = scaled[sort_idx]                     # logits in descending order
    sorted_p  = softmax_np(sorted_l)                 # probabilities in that order
    cumul     = np.cumsum(sorted_p)                  # cumulative sum: [0.4, 0.7, 0.85, ...]
    # Find the first index where cumulative probability >= p
    # +1 to include the word that pushes us over the threshold
    cutoff    = int(np.searchsorted(cumul, p)) + 1
    cutoff    = min(cutoff, len(logits))             # cannot exceed vocab size

    # Build a new logit array: keep top-cutoff words, set rest to -infinity
    filtered  = np.full_like(scaled, -np.inf)        # start with all -infinity
    filtered[sort_idx[:cutoff]] = sorted_l[:cutoff]  # restore the nucleus words

    probs     = softmax_np(filtered)                 # final probabilities for nucleus
    probs     = np.nan_to_num(probs, nan=0.0)        # replace NaN with 0
    if probs.sum() == 0:                             # safety fallback
        probs = np.ones(len(probs)) / len(probs)
    return int(np.random.choice(len(probs), p=probs))

print("  All 4 sampling functions defined:")
print("    1. greedy_pick(logits)")
print("    2. temperature_pick(logits, temperature)")
print("    3. top_k_pick(logits, k, temperature)")
print("    4. top_p_pick(logits, p, temperature)")

# =============================================================================
# STEP B5: NUMPY FALLBACK MODEL (runs even without PyTorch)
# A minimal bigram embedding model implemented in plain numpy.
# =============================================================================

subsection("Step B5: Numpy Fallback Model (no PyTorch needed)")

print("""
  This is a tiny word embedding model built from scratch using only numpy.
  It is NOT as powerful as PyTorch but it is enough to demonstrate the concepts.

  Architecture:
    Input: word integer ID (e.g. 5 for "fox")
    Embedding layer: ID -> vector of EMBED_DIM numbers
    Linear layer: vector -> scores for every word in vocab
    Output: logits (one score per vocabulary word)

  C# analogy:
    class TinyModel {
        float[,] embeddings;   // [vocabSize, embedDim]
        float[,] weights;      // [embedDim, vocabSize]
        float[]  bias;         // [vocabSize]

        float[] Forward(int wordId) {
            var emb = embeddings[wordId];           // lookup embedding
            return MatMul(emb, weights) + bias;     // linear layer
        }
    }
""")

EMBED_DIM   = 16     # size of each word embedding vector (16 numbers per word)
LEARN_RATE  = 0.05   # learning rate: how big each weight update step is
NUM_EPOCHS  = 300    # number of full passes through the training data

# Initialize weights randomly (small values to start near 0)
# np.random.randn gives numbers from a standard normal distribution (mean=0, std=1)
# We multiply by 0.1 to keep initial values small and avoid exploding gradients.
np_embeddings = np.random.randn(vocab_size, EMBED_DIM) * 0.1   # shape: [vocab, embed_dim]
np_weights    = np.random.randn(EMBED_DIM, vocab_size) * 0.1   # shape: [embed_dim, vocab]
np_bias       = np.zeros(vocab_size)                             # shape: [vocab] -- start at 0

def np_forward(word_id):
    """
    Forward pass: given a word ID, compute logits for every possible next word.

    Steps:
      1. Look up the embedding for this word (row from embeddings matrix).
      2. Multiply embedding by weights matrix (linear layer).
      3. Add bias.

    C# analogy: like calling model.Forward(wordId) which returns float[vocabSize].
    """
    emb    = np_embeddings[word_id]            # get embedding vector for this word
    logits = emb @ np_weights + np_bias        # @ is matrix multiply; + adds bias
    return logits                              # shape: [vocab_size]

def np_loss_and_grad(word_id, target_id):
    """
    Compute cross-entropy loss and gradients for one training pair.

    Cross-entropy loss = -log(probability of the correct word)
    The lower this value, the better the model is at predicting target_id.

    Also computes gradients (how to adjust each weight to reduce loss).
    Gradients tell us: "nudge this weight UP or DOWN, and by how much?"

    C# analogy: like calling loss.Backward() in PyTorch, but written out manually.
    """
    emb    = np_embeddings[word_id]            # embedding for the input word
    logits = emb @ np_weights + np_bias        # forward pass (compute scores)
    probs  = softmax_np(logits)                # convert scores to probabilities

    # Cross-entropy loss: -log(probability of the correct target word)
    # If model assigns 90% to the correct word: loss = -log(0.9) = 0.105 (good)
    # If model assigns  5% to the correct word: loss = -log(0.05) = 3.0  (bad)
    loss   = -np.log(probs[target_id] + 1e-9)  # 1e-9 prevents log(0) = -infinity

    # --- GRADIENT COMPUTATION ---
    # Gradient of loss w.r.t. logits (before softmax):
    # For the target word: (probability - 1)
    # For all other words: (probability - 0) = probability
    # This is the standard formula for softmax + cross-entropy combined.
    d_logits              = probs.copy()       # start with all probabilities
    d_logits[target_id]  -= 1.0               # subtract 1 from the correct word's probability

    # Gradient of loss w.r.t. the weights matrix:
    # d_weights = outer product of embedding and d_logits
    # "outer product" means: for each pair (emb[i], d_logit[j]), compute emb[i]*d_logit[j]
    # C# analogy: result[i,j] = emb[i] * d_logits[j]
    d_weights = np.outer(emb, d_logits)        # shape: [embed_dim, vocab_size]

    # Gradient of loss w.r.t. bias: same as d_logits (adding bias has gradient 1)
    d_bias    = d_logits                        # shape: [vocab_size]

    # Gradient of loss w.r.t. the embedding for this word:
    # d_emb = d_logits @ weights.T   (chain rule through the linear layer)
    d_emb     = d_logits @ np_weights.T        # shape: [embed_dim]

    return loss, d_emb, d_weights, d_bias       # return loss and all gradients

# --- TRAINING LOOP ---
print(f"  Training numpy model for {NUM_EPOCHS} epochs...")
print(f"  Vocab size:  {vocab_size} words")
print(f"  Embed dim:   {EMBED_DIM}")
print(f"  Train pairs: {len(input_ids)}")
print()

for epoch in range(NUM_EPOCHS):                   # loop over epochs
    total_loss = 0.0                               # accumulate loss for this epoch

    # Shuffle training pairs each epoch so the model does not memorize order.
    # C# analogy: like calling list.Shuffle() at the start of each epoch.
    indices = list(range(len(input_ids)))          # list of indices [0, 1, 2, ...]
    random.shuffle(indices)                        # shuffle in-place

    for idx in indices:                            # loop over each training pair
        w_in   = input_ids[idx]                   # input word ID
        w_tgt  = target_ids[idx]                  # target (next) word ID

        # Compute loss and gradients for this training pair
        loss, d_emb, d_weights, d_bias = np_loss_and_grad(w_in, w_tgt)

        total_loss += loss                         # add to epoch total

        # --- GRADIENT DESCENT WEIGHT UPDATE ---
        # Subtract (learning_rate * gradient) from each parameter.
        # This nudges the weights in the direction that reduces loss.
        # C# analogy: weights -= learningRate * gradient;
        np_embeddings[w_in] -= LEARN_RATE * d_emb        # update embedding for this word
        np_weights          -= LEARN_RATE * d_weights    # update linear layer weights
        np_bias             -= LEARN_RATE * d_bias        # update bias

    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:                     # every 50th epoch
        avg_loss = total_loss / len(input_ids)     # average loss per training pair
        print(f"  Epoch {epoch+1:>3}/{NUM_EPOCHS}  |  avg loss: {avg_loss:.4f}")

print()
print("  Numpy model training complete!")

# =============================================================================
# STEP B6: PYTORCH MODEL (if available)
# This is the "real" version using PyTorch. Skipped gracefully if not installed.
# =============================================================================

subsection("Step B6: PyTorch Model (used if PyTorch is installed)")

# try/except for imports is the Python way to handle optional dependencies.
# C# analogy: like checking Assembly.LoadFrom() and catching FileNotFoundException.
try:
    import torch                       # core PyTorch library
    import torch.nn as nn              # neural network building blocks
    import torch.optim as optim        # optimizers (Adam, SGD, etc.)

    PYTORCH_AVAILABLE = True           # flag to use later
    print("  PyTorch found! Will use PyTorch model for generation.")
    print(f"  PyTorch version: {torch.__version__}")

except ImportError:
    PYTORCH_AVAILABLE = False          # flag: PyTorch not installed
    print("  PyTorch not found. Using numpy fallback model.")
    print("  (Install with: pip install torch)")


if PYTORCH_AVAILABLE:
    # -------------------------------------------------------------------------
    # Define the PyTorch model as a class.
    # In PyTorch, models inherit from nn.Module.
    # C# analogy: like inheriting from a base class "NeuralNetworkBase".
    # -------------------------------------------------------------------------

    class TinyWordModel(nn.Module):
        """
        A tiny word-level language model.

        Architecture:
          Embedding layer -> Linear layer -> Logits

        Input:  word ID (integer)
        Output: logit scores for every word in vocabulary

        C# analogy:
          class TinyWordModel : NeuralNetworkBase {
              EmbeddingLayer embeddings;
              LinearLayer linear;
              ...
          }
        """

        def __init__(self, vocab_size, embed_dim):
            """
            Constructor: define the layers of the model.
            __init__ is Python's constructor -- like TinyWordModel() in C#.
            super().__init__() calls the parent class constructor (nn.Module).
            """
            super().__init__()                              # must call parent constructor

            # Embedding layer: maps word IDs to dense vectors.
            # Like a lookup table: vocab_size rows, embed_dim columns.
            # C# analogy: float[,] embeddings = new float[vocabSize, embedDim];
            self.embedding = nn.Embedding(vocab_size, embed_dim)

            # Linear layer: maps embed_dim inputs to vocab_size outputs.
            # This layer has (embed_dim * vocab_size) + vocab_size learnable parameters.
            # C# analogy: float[,] weights; float[] bias;
            self.linear = nn.Linear(embed_dim, vocab_size)

        def forward(self, word_id_tensor):
            """
            Forward pass: compute logits for the given input word.
            PyTorch calls this automatically when you write model(input).
            C# analogy: like implementing an interface method Forward().
            """
            emb    = self.embedding(word_id_tensor)   # lookup embedding for word
            logits = self.linear(emb)                 # linear transformation
            return logits                             # raw scores, not probabilities

    # -------------------------------------------------------------------------
    # Instantiate and train the PyTorch model
    # -------------------------------------------------------------------------

    PT_EMBED_DIM  = 16      # same embedding size as the numpy model
    PT_EPOCHS     = 300     # number of training epochs
    PT_LEARN_RATE = 0.05    # learning rate for Adam optimizer

    # Create model instance
    # C# analogy: var model = new TinyWordModel(vocabSize, embedDim);
    pt_model = TinyWordModel(vocab_size, PT_EMBED_DIM)

    # Loss function: CrossEntropyLoss is perfect for classification (which word is next?)
    # It combines softmax + negative log likelihood internally.
    # C# analogy: var lossFunction = new CrossEntropyLoss();
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: tracks momentum and adapts the learning rate automatically.
    # Much better than plain gradient descent for most tasks.
    # C# analogy: var optimizer = new AdamOptimizer(model.Parameters, learningRate);
    optimizer = optim.Adam(pt_model.parameters(), lr=PT_LEARN_RATE)

    # Convert our Python lists to PyTorch tensors.
    # A tensor is like a numpy array but PyTorch can track gradients on it.
    # C# analogy: like converting List<int> to a GPU-compatible float[].
    input_tensor  = torch.tensor(input_ids,  dtype=torch.long)   # long = int64
    target_tensor = torch.tensor(target_ids, dtype=torch.long)

    print()
    print(f"  Training PyTorch model for {PT_EPOCHS} epochs...")

    for epoch in range(PT_EPOCHS):                    # loop over all epochs
        # optimizer.zero_grad() clears gradients from the previous step.
        # PyTorch ACCUMULATES gradients by default, so we must clear them each step.
        # C# analogy: like resetting an accumulator to 0 before each iteration.
        optimizer.zero_grad()

        # Forward pass: compute logits for ALL training pairs at once (batch).
        # This is more efficient than looping one pair at a time.
        logits = pt_model(input_tensor)               # shape: [num_pairs, vocab_size]

        # Compute loss: how wrong are the logits compared to the target words?
        loss   = criterion(logits, target_tensor)     # scalar loss value

        # Backward pass: compute gradients of loss w.r.t. all parameters.
        # PyTorch builds a "computation graph" during forward pass and
        # walks it backwards here to find gradients automatically.
        # C# analogy: like calling loss.Backward() -- PyTorch does the calculus for you.
        loss.backward()

        # Update weights: optimizer uses the gradients to adjust all parameters.
        optimizer.step()

        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:>3}/{PT_EPOCHS}  |  loss: {loss.item():.4f}")

    print()
    print("  PyTorch model training complete!")

    # Function to get logits from PyTorch model (returns numpy array for compatibility)
    def get_logits_pt(word_id):
        """
        Run a forward pass through the PyTorch model and return numpy logits.
        torch.no_grad() tells PyTorch: do not track gradients (saves memory at inference).
        .detach() detaches the tensor from the computation graph.
        .numpy() converts PyTorch tensor to numpy array.
        C# analogy: like calling model.Predict(wordId) with inference-only mode.
        """
        with torch.no_grad():                             # disable gradient tracking
            tensor = torch.tensor([word_id], dtype=torch.long)  # wrap ID in tensor
            logits = pt_model(tensor)                     # forward pass
            return logits[0].detach().numpy()             # return as numpy array

    USE_PYTORCH = True    # flag: use PyTorch logits for generation

else:
    # PyTorch not available -- we will use the numpy model instead.
    USE_PYTORCH = False

    def get_logits_pt(word_id):
        """Fallback: use numpy model when PyTorch is not available."""
        return np_forward(word_id)                        # call numpy forward pass

# Unified function: always returns numpy logits regardless of backend.
# This lets the generation code below work with EITHER backend.
def get_logits(word_id):
    """
    Get logit scores for the next word, given the current word ID.
    Automatically uses PyTorch model if available, numpy model otherwise.
    C# analogy: like calling an interface method that dispatches to the right impl.
    """
    if USE_PYTORCH if PYTORCH_AVAILABLE else False:       # check which backend to use
        return get_logits_pt(word_id)                     # PyTorch path
    else:
        return np_forward(word_id)                        # numpy path

# =============================================================================
# STEP B7: TEXT GENERATION WITH ALL 4 STRATEGIES
# =============================================================================

subsection("Step B7: Generate Text Using All 4 Strategies")

def generate_ml(seed_word, num_words, strategy, **kwargs):
    """
    Generate text from the ML model using a chosen strategy.

    Parameters:
      seed_word  - starting word (must be in vocabulary)
      num_words  - how many additional words to generate
      strategy   - one of: 'greedy', 'temperature', 'top_k', 'top_p'
      **kwargs   - extra arguments for the strategy (temperature, k, p)
                   C# analogy: like params object[] in C#.

    Returns:
      generated text as a single string.
    """
    # Check if the seed word is in our vocabulary
    if seed_word not in vocab:                            # if unknown word
        return f"[Unknown word: '{seed_word}' not in vocabulary]"

    current_id = vocab[seed_word]                        # convert seed word to ID
    words      = [seed_word]                             # start the output list

    for _ in range(num_words):                           # generate num_words more words
        logits = get_logits(current_id)                  # get scores from model

        # Pick next word ID using the chosen strategy
        if strategy == "greedy":
            next_id = greedy_pick(logits)                # always highest score

        elif strategy == "temperature":
            temp    = kwargs.get("temperature", 1.0)     # get temperature from kwargs
            next_id = temperature_pick(logits, temp)     # sample with temperature

        elif strategy == "top_k":
            k       = kwargs.get("k", 3)                 # get k from kwargs
            temp    = kwargs.get("temperature", 1.0)
            next_id = top_k_pick(logits, k, temp)        # sample from top-k

        elif strategy == "top_p":
            p       = kwargs.get("p", 0.9)               # get p from kwargs
            temp    = kwargs.get("temperature", 1.0)
            next_id = top_p_pick(logits, p, temp)        # nucleus sampling

        else:
            next_id = greedy_pick(logits)                # default to greedy

        next_word = reverse_vocab[next_id]               # decode ID back to word
        words.append(next_word)                          # add to output
        current_id = next_id                             # advance to next word

    return " ".join(words)                               # join list into sentence

# --- Run all 4 strategies and display results side by side ---

print()
print("Generating 6-word completions from the ML model:")
print()

# List of (label, strategy, kwargs) tuples to test
strategies_to_test = [
    ("Greedy",               "greedy",      {}),
    ("Temperature = 0.5",   "temperature", {"temperature": 0.5}),
    ("Temperature = 1.5",   "temperature", {"temperature": 1.5}),
    ("Top-k (k=3, t=0.8)",  "top_k",       {"k": 3,  "temperature": 0.8}),
    ("Top-p (p=0.8, t=0.8)","top_p",       {"p": 0.8,"temperature": 0.8}),
]

# Test each seed word with all strategies
seed_words = ["the", "fox", "dog"]                       # starting words

for seed in seed_words:                                  # loop over seed words
    print(f"  Seed word: '{seed}'")
    print("  " + "-" * 50)
    for label, strategy, kwargs in strategies_to_test:   # loop over strategies
        np.random.seed(7)                                # same seed for fair comparison
        text = generate_ml(seed, num_words=6, strategy=strategy, **kwargs)
        print(f"  {label:<26}: {text}")
    print()

# =============================================================================
# PART C: SIDE-BY-SIDE COMPARISON
# Frequency model vs ML model on the same seeds
# =============================================================================

section("PART C: Comparison - Frequency vs ML Model")

print("""
Both models learned from the same 4 sentences.
Let's compare their outputs side by side.

KEY DIFFERENCES to look for:
  - Frequency model: deterministic, always the same output
  - ML model with greedy: also deterministic, but via learned weights
  - ML model with temperature/top-k/top-p: has controlled randomness
""")

print(f"  {'Seed':<8} {'Method':<30} {'Generated Text'}")
print("  " + "-" * 70)

for seed in ["the", "fox"]:                              # test two seeds
    # Frequency model output (always same -- no seed needed)
    freq_result = complete_phrase_freq(seed, bigram_counts, num_words=5)
    freq_text   = " ".join(freq_result)                  # join to string
    print(f"  {seed:<8} {'Frequency (bigram)':<30} {freq_text}")

    # ML model: greedy (also deterministic)
    np.random.seed(42)
    ml_greedy = generate_ml(seed, num_words=5, strategy="greedy")
    print(f"  {seed:<8} {'ML model (greedy)':<30} {ml_greedy}")

    # ML model: temperature = 0.8 (some randomness)
    np.random.seed(42)
    ml_temp = generate_ml(seed, num_words=5, strategy="temperature",
                          temperature=0.8)
    print(f"  {seed:<8} {'ML model (temp=0.8)':<30} {ml_temp}")

    # ML model: top-p nucleus sampling
    np.random.seed(42)
    ml_topp = generate_ml(seed, num_words=5, strategy="top_p",
                          p=0.85, temperature=0.8)
    print(f"  {seed:<8} {'ML model (top-p=0.85)':<30} {ml_topp}")

    print()   # blank line between seeds

# =============================================================================
# PART D: DECISION GUIDE
# When to use each strategy in a real project
# =============================================================================

section("PART D: Strategy Decision Guide")

print("""
  Use this guide when building your own autocomplete system:

  What are you building?
  |
  +-- Phone keyboard autocomplete
  |     -> Frequency bigram model (fast, no GPU needed)
  |        OR top-k (k=3) with low temperature (0.3)
  |
  +-- Code completion (VS Code, Copilot style)
  |     -> temperature=0.2, top_k=10
  |        (precision first; creativity is dangerous in code)
  |
  +-- Search engine autocomplete (Google style)
  |     -> Frequency trigram OR greedy ML
  |        (user expects the most common/correct completion)
  |
  +-- Chatbot (customer service, Q&A)
  |     -> temperature=0.8, top_p=0.9
  |        (natural and varied but not too wild)
  |        THIS IS WHAT CHATGPT USES
  |
  +-- Creative writing assistant
  |     -> temperature=1.2, top_p=0.95
  |        (surprising word choices welcome)
  |
  +-- Factual summary / document generation
        -> temperature=0.3, top_p=0.9
           (accurate, small variety, avoids hallucination)

  Real-world settings:
    ChatGPT        : top_p=0.9, temperature ~ 0.7
    GitHub Copilot : temperature ~ 0.2, top_k=10
    Google Search  : frequency model + query logs
""")

# =============================================================================
# SUMMARY
# =============================================================================

section("PROJECT 3 SUMMARY")

print("""
  What you built in this project:
  --------------------------------
  PART A - Frequency Autocomplete:
    - Counted word pairs (bigrams) from training text
    - Stored counts in a nested dictionary
    - Predicted next words by looking up the most frequent follower
    - Chained predictions to complete phrases

  PART B - ML Autocomplete:
    - Built a vocabulary (word -> integer ID mapping)
    - Created training pairs (input word -> target next word)
    - Trained a tiny embedding + linear model
      * Numpy version: manual backpropagation, gradient descent
      * PyTorch version (if available): automatic backpropagation with Adam
    - Generated text using all 4 strategies from Lesson 5:
        1. Greedy: always highest probability
        2. Temperature: control randomness by scaling logits
        3. Top-k: only sample from top k words
        4. Top-p: sample from the nucleus (cumulative probability >= p)

  Key insight:
    Both models learn "the fox" and "the quick" from the training text.
    The frequency model MEMORIZES exact pairs.
    The ML model learns PATTERNS via weights that can generalize.

  C# analogy recap:
    - bigram_counts  ->  Dictionary<string, Dictionary<string, int>>
    - vocab          ->  Dictionary<string, int>
    - Embedding      ->  float[,] lookup table
    - Linear layer   ->  y = x * W + b (matrix multiply)
    - Softmax        ->  Converts scores to probabilities summing to 1
    - Loss           ->  Measures how wrong the model is (lower = better)
    - Optimizer      ->  Adjusts weights to reduce loss (like auto-tuning)
""")

# =============================================================================
# QUIZ QUESTIONS
# (Test your understanding -- try to answer before reading the answers!)
# =============================================================================

section("QUIZ QUESTIONS")

print("""
  Q1. In the frequency-based model, if the word "the" was followed by
      "fox" 3 times and "quick" 2 times in training, what word will the
      greedy frequency model always predict after "the"?

      A) quick
      B) fox
      C) dog
      D) It depends on randomness

  A1. B) fox. The frequency model always picks the most frequent follower.
      "fox" appeared 3 times after "the", which is more than "quick" (2 times).
      There is no randomness in greedy selection.

  -----------------------------------------------------------------------

  Q2. Temperature controls how "random" the output is.
      Which temperature setting would make the model pick
      UNEXPECTED words more often?

      A) temperature = 0.1
      B) temperature = 1.0
      C) temperature = 2.0
      D) temperature = 0.0

  A2. C) temperature = 2.0. High temperature flattens the probability
      distribution, making unlikely words more probable.
      temperature = 0.1 would make the model very focused (almost greedy).
      temperature = 0.0 would be pure greedy (deterministic).

  -----------------------------------------------------------------------

  Q3. What is the key ADVANTAGE of top-p sampling over top-k sampling?

      A) Top-p is always faster to compute
      B) Top-p uses a fixed number of candidates, making it more consistent
      C) Top-p adapts the nucleus size based on the model's confidence,
         using fewer words when confident and more words when uncertain
      D) Top-p always produces better quality text than top-k

  A3. C) Top-p adapts dynamically. When the model is very confident
      (one word has 80% probability), a small nucleus of 1-2 words covers p=0.9.
      When the model is uncertain (many words share probability equally),
      a larger nucleus is needed. Top-k always uses exactly k words
      regardless of confidence level.

  -----------------------------------------------------------------------

  BONUS Q4. In the numpy model, what does the EMBEDDING layer do?
            Why can't we just pass the integer word ID directly into
            the linear layer?

  A4. The embedding layer converts a word ID (a single integer like 5)
      into a dense vector of numbers (e.g. [0.2, -0.4, 1.1, ...]).
      We cannot pass raw integers into a linear layer because:
        - A linear layer does math: output = ID * weights
        - That would give different "importance" to the word just based
          on its arbitrary ID number (word ID 10 is not 2x "more" than ID 5)
        - Embeddings let the model learn a meaningful representation for
          each word, where similar words (fox, cat) can end up with
          similar embedding vectors.
      C# analogy: embedding is like mapping an enum value to a rich object
                  rather than doing arithmetic on the enum integer directly.
""")

print("=" * 60)
print("Project 3 complete!")
print("Next: project_04 (coming soon)")
print("=" * 60)

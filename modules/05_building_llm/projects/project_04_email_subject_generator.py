"""
=============================================================================
PROJECT 4: Email Subject Line Generator
=============================================================================

GLOSSARY (read this first, before looking at any code)
-------------------------------------------------------
Bigram          : A pair of consecutive items (words here, chars in Part B).
                  "bi" = two. A bigram model uses ONE previous item to predict
                  the next item.
                  C# analogy: Dictionary<string, Dictionary<string, int>>
                              where outer key = current word,
                              inner key = next word, value = count.

Frequency Model : A model that learns by counting how often things appear
                  together. No math, no gradients. Just counting.
                  C# analogy: like a Dictionary<string, int> of tallies.

Vocabulary      : The complete set of unique characters (or words) the model
                  knows. Anything outside the vocabulary is unknown.
                  C# analogy: HashSet<char> or Dictionary<char, int>.

Token           : The smallest unit of text the model works with.
                  In Part A tokens are WORDS. In Part B tokens are CHARACTERS.

Character-Level : Working with individual characters instead of words.
                  "50%" becomes ['5','0','%']. Simpler vocabulary, slower
                  to learn patterns that span many characters.

Embedding       : A list of numbers that represents a token.
                  The model LEARNS these numbers during training.
                  e.g. 'a' might become [0.3, -0.1, 0.8, ...]
                  C# analogy: like a float[] that IS the token in math-space.

Logits          : Raw scores from the model for every possible next token.
                  NOT probabilities yet. Can be positive or negative.
                  C# analogy: float[] rawScores before any normalization.

Softmax         : Converts logits into probabilities that sum to 1.0.
                  Formula: exp(x_i) / sum(exp(x_j) for all j)
                  C# analogy: normalizing a score array so all values sum to 1.

Temperature     : A number that divides logits before softmax.
                  temp < 1.0  = more focused, predictable output.
                  temp = 1.0  = model's own raw probabilities unchanged.
                  temp > 1.0  = more random, creative, surprising output.
                  C# analogy: like a "creativity dial" on a random number generator.

Cross-Entropy   : A loss function for classification tasks (which token is next?).
                  Punishes the model MORE when it is confident AND wrong.
                  Cross-entropy loss = -log(probability of correct answer).
                  C# analogy: a scoring penalty function; lower is better.

Loss            : A single number measuring how WRONG the model is right now.
                  We try to make this as small as possible over training.
                  C# analogy: like a "score" in a game, but lower = better.

Backpropagation : The algorithm that works out how much each weight in the
                  model contributed to the loss. Then we nudge weights to
                  reduce the loss. Called "backprop" for short.
                  C# analogy: like automatic differentiation -- PyTorch does
                              this for you with .backward().

Gradient        : The direction and size of the nudge we apply to each weight.
                  Gradient = derivative of loss with respect to that weight.
                  C# analogy: like the slope of a curve telling you which way
                              to step to go downhill faster.

Optimizer       : Code that applies gradients to update weights.
                  Adam is the most popular optimizer (Adaptive Moment estimation).
                  C# analogy: an automatic "weight tuner" that uses gradients.

Epoch           : One complete pass through ALL training data.
                  After 200 epochs the model has seen every character pair
                  200 times.

nanoGPT-style   : A tiny GPT-like model that works at the character level.
                  Uses the same building blocks as GPT (embedding + linear layer)
                  but with NO attention and NO transformer blocks. Just the
                  bare minimum to demonstrate the concept.

Sampling        : Choosing the next token by drawing from a probability
                  distribution instead of always picking the highest score.
                  This introduces useful randomness (creativity).

Nucleus/Top-p   : A sampling strategy. Keep only the smallest group of tokens
                  whose probabilities add up to p (e.g. 0.9 = 90%).
                  Draw the next token from that group only.

=============================================================================
PROJECT OVERVIEW
=============================================================================

Real-world use case:
  Marketing teams write email campaigns. One of the hardest parts is the
  SUBJECT LINE -- the 6-10 words the reader sees in their inbox before
  deciding to open or delete the email.

  A subject line generator can:
    - Produce 20 variations in seconds (instead of a human spending 30 min)
    - Test different tones: urgent, friendly, exclusive, mysterious
    - Feed into A/B testing tools to find which line gets more opens

  Real companies that use LLMs for this:
    - Mailchimp (subject line assistant)
    - HubSpot (AI email writer)
    - Salesforce Einstein (predictive subject lines)

What you will build:
  PART A -- Frequency-Based Generator (No Machine Learning)
  ---------------------------------------------------------
  Count how often each WORD follows another word in the training subject lines.
  Generate new subject lines by chaining the most-likely-next-word predictions.
  This is how autocomplete worked in the 1990s and early 2000s.

  PART B -- nanoGPT-Style Char-Level Model (Machine Learning)
  ------------------------------------------------------------
  Train a tiny character-level language model (inspired by nanoGPT) on the
  same subject lines. Generate new lines using temperature sampling.
  Show clearly how temperature affects creativity vs coherence.

TRAINING DATA (hard-coded, 15 real-style marketing subject lines):
  50% off your next order today only
  Don't miss our biggest sale of the year
  Your free gift is waiting inside
  Exclusive offer just for you this week
  Last chance to save on summer deals
  New arrivals you will love this season
  Limited time offer ends at midnight
  Get 20% off when you shop today
  We miss you here is a special discount
  Your account has a reward waiting now
  Top picks selected just for you today
  Flash sale starts now up to 70% off
  Free shipping on all orders this weekend
  Hurry offer expires in 24 hours only
  Members only early access starts today

=============================================================================
"""

# =============================================================================
# IMPORTS
# We use only standard Python + numpy here.
# PyTorch is imported inside a try/except in Part B so this file runs even
# if PyTorch is not installed.
# C# analogy: conditional "using" statements (impossible in C#, but Python
#             lets you TRY to import and catch the failure gracefully).
# =============================================================================

import random                        # built-in module: random number generation
import numpy as np                   # numerical computing (like System.Math on steroids)
from collections import defaultdict  # dict that auto-creates missing keys

# Fix random seeds so you get the same output every run.
# C# analogy: new Random(42) gives the same sequence each time.
random.seed(42)                      # fix Python's built-in random
np.random.seed(42)                   # fix numpy's random

# =============================================================================
# TRAINING DATA
# 15 marketing-style email subject lines, hard-coded as Python strings.
# C# analogy: string[] trainingSubjects = { "50% off ...", "Don't miss ..." };
# =============================================================================

SUBJECT_LINES = [
    "50% off your next order today only",
    "Don't miss our biggest sale of the year",
    "Your free gift is waiting inside",
    "Exclusive offer just for you this week",
    "Last chance to save on summer deals",
    "New arrivals you will love this season",
    "Limited time offer ends at midnight",
    "Get 20% off when you shop today",
    "We miss you here is a special discount",
    "Your account has a reward waiting now",
    "Top picks selected just for you today",
    "Flash sale starts now up to 70% off",
    "Free shipping on all orders this weekend",
    "Hurry offer expires in 24 hours only",
    "Members only early access starts today",
]

# =============================================================================
# PRINT HELPER FUNCTIONS
# ALL output uses only ASCII characters (cp1252-safe).
# No Unicode arrows, box-drawing, or special symbols above ASCII 127.
# C# analogy: static void Section(string title) { Console.WriteLine(...); }
# =============================================================================

def section(title):
    """Print a bold section header using = separators."""
    print()                          # blank line before
    print("=" * 60)                  # 60 equals signs
    print(title)                     # title text
    print("=" * 60)                  # 60 equals signs again

def subsection(title):
    """Print a smaller sub-section header using - separators."""
    print()                          # blank line before
    print(title)                     # title text
    print("-" * len(title))          # underline with dashes (same length)


# =============================================================================
# =============================================================================
#
#  PART A: FREQUENCY-BASED SUBJECT LINE GENERATOR
#          (No Machine Learning -- just counting words)
#
# =============================================================================
# =============================================================================

section("PART A: Frequency-Based Subject Line Generator (No ML)")

print("""
HOW IT WORKS (concept before any code):
  1. Split each subject line into individual words.
  2. For every consecutive word pair, count how often word2 follows word1.
  3. Also count how often each word appears at the START of a line.
  4. To generate a new subject line:
       a. Pick a start word (based on frequency of line-starting words).
       b. Look up the most-likely next word after the current word.
       c. Keep going until we have enough words.
  5. Print 5 generated subject lines.

This is a WORD-LEVEL BIGRAM MODEL.
  - It only looks 1 word back to decide the next word.
  - No math, no gradients, no GPU needed.
  - Very fast but very limited: it can only produce patterns it has seen.

C# analogy for the data structures we will build:
  Dictionary<string, Dictionary<string, int>> bigramCounts;
    // bigramCounts["offer"]["ends"] = 1  (offer was followed by ends once)

  Dictionary<string, int> startWordCounts;
    // startWordCounts["Your"] = 2  ("Your" started 2 subject lines)
""")

# =============================================================================
# STEP A1: BUILD THE WORD BIGRAM TABLE AND START-WORD TABLE
# =============================================================================

subsection("Step A1: Build Word Bigram Table and Start-Word Table")

# defaultdict(lambda: defaultdict(int)) means:
#   Outer dict: if key (word) is missing, create an empty inner defaultdict.
#   Inner dict: if key (next_word) is missing, start count at 0.
# C# analogy: like using ConcurrentDictionary.GetOrAdd() for both levels.
word_bigram = defaultdict(lambda: defaultdict(int))  # word -> {next_word: count}

# Track which words appear at the START of a subject line and how often.
# C# analogy: Dictionary<string, int> startCounts = new();
start_counts = defaultdict(int)                      # word -> count of line-starts

for line in SUBJECT_LINES:                           # loop over each subject line
    words = line.split()                             # split on spaces to get word list
    # .split() with no argument splits on any whitespace -- like str.Split() in C#

    if not words:                                    # skip empty lines (safety check)
        continue                                     # move to next line

    start_counts[words[0]] += 1                      # record this line's first word

    # zip(words, words[1:]) pairs each word with the word that follows it.
    # C# analogy: for (int i = 0; i < words.Length - 1; i++) { ... }
    for current_w, next_w in zip(words, words[1:]):  # loop consecutive pairs
        word_bigram[current_w][next_w] += 1          # increment pair count

# Print the start-word distribution so the student can see what was learned
print()
print("Words that START a subject line (and how often):")
print()

# Sort by count descending so most-common start words appear first
# sorted() with key=lambda sorts by the second element of each tuple (the count)
sorted_starts = sorted(start_counts.items(), key=lambda pair: -pair[1])

for word, count in sorted_starts:                   # loop sorted start words
    bar = "#" * count                               # ASCII bar chart using # symbols
    print(f"  {word:<12} {count:>2}  {bar}")        # formatted output

print()
print(f"Total unique starting words: {len(start_counts)}")
print(f"Total word bigrams learned:  {sum(len(v) for v in word_bigram.values())}")

# =============================================================================
# STEP A2: THE GREEDY NEXT-WORD PREDICTION FUNCTION
# =============================================================================

subsection("Step A2: Predict the Most Likely Next Word")

def predict_next_word_freq(current_word, bigram_table):
    """
    Given the current word, return the most frequent next word.

    Parameters:
      current_word  - the last word in the sequence (string)
      bigram_table  - our nested defaultdict of pair counts

    Returns:
      The next word as a string, or None if current_word was never seen.

    C# analogy:
      string? PredictNext(string currentWord,
                          Dictionary<string, Dictionary<string, int>> table)
    """
    if current_word not in bigram_table:             # word never seen in training
        return None                                  # no prediction possible

    # Get the inner dict {next_word: count} for this word
    followers = bigram_table[current_word]           # dict of word -> count

    # max() with key finds the key with the highest value.
    # key=lambda pair: pair[1] means "sort by the count (second element)".
    # C# analogy: followers.OrderByDescending(p => p.Value).First().Key
    best_next = max(followers.items(), key=lambda pair: pair[1])[0]

    return best_next                                 # return just the word string

# =============================================================================
# STEP A3: SAMPLE A START WORD (WEIGHTED BY FREQUENCY)
# =============================================================================

subsection("Step A3: Sample a Weighted Start Word")

def sample_start_word(start_counts_dict):
    """
    Pick a starting word at random, but weighted by how often it starts a line.
    Words that start more subject lines are more likely to be picked.

    This is called WEIGHTED SAMPLING.
    C# analogy: like rolling a weighted die (not all sides equal).

    Parameters:
      start_counts_dict - dict mapping word -> count of line starts

    Returns:
      A start word (string).
    """
    words_list  = list(start_counts_dict.keys())     # all possible start words
    counts_list = list(start_counts_dict.values())   # their corresponding counts

    # Convert counts to probabilities (divide each count by total count).
    # This gives us a proper probability distribution summing to 1.0.
    total = sum(counts_list)                         # total number of line starts
    probs = [c / total for c in counts_list]         # list of probabilities

    # np.random.choice picks one item from words_list weighted by probs.
    # C# analogy: like a weighted random selector.
    chosen = np.random.choice(words_list, p=probs)   # pick one start word

    return str(chosen)                               # return as Python string

# =============================================================================
# STEP A4: GENERATE A COMPLETE SUBJECT LINE
# =============================================================================

subsection("Step A4: Generate Subject Lines by Chaining Predictions")

def generate_subject_freq(bigram_table, start_counts_dict,
                          min_words=4, max_words=9):
    """
    Generate one email subject line using the frequency bigram model.

    Algorithm:
      1. Sample a start word (weighted by frequency).
      2. Predict the next word greedily (most frequent follower).
      3. Repeat until we hit max_words or no next word exists.
      4. Return the generated line as a string.

    Parameters:
      bigram_table       - word bigram count table
      start_counts_dict  - word -> line-start count
      min_words          - minimum length of generated line (default 4)
      max_words          - maximum length of generated line (default 9)

    Returns:
      A string containing the generated subject line.

    C# analogy:
      string GenerateSubject(... int minWords = 4, int maxWords = 9)
    """
    generated = []                                   # list to build the line

    current = sample_start_word(start_counts_dict)  # pick a start word
    generated.append(current)                        # add start word to line

    for _ in range(max_words - 1):                  # generate up to max_words-1 more
        next_w = predict_next_word_freq(current, bigram_table)  # greedy prediction

        if next_w is None:                           # no next word known -- stop
            break                                    # exit the loop

        generated.append(next_w)                     # add word to line
        current = next_w                             # advance to the new word

        if len(generated) >= max_words:              # reached max length -- stop
            break                                    # exit the loop

    # Capitalise the first letter of the generated line (like a real subject)
    # C# analogy: char.ToUpper(line[0]) + line[1..]
    line = " ".join(generated)                       # join list into a single string
    if line:                                         # if line is not empty
        line = line[0].upper() + line[1:]            # capitalise first character

    return line                                      # return the finished line

# Generate and print 5 subject lines
print()
print("Generating 5 subject lines using the frequency bigram model:")
print()
print("  (These are generated by chaining the most-likely-next-word predictions)")
print()

np.random.seed(42)                                   # reset seed for reproducibility

for i in range(5):                                   # generate 5 subject lines
    line = generate_subject_freq(word_bigram, start_counts)  # generate one line
    print(f"  {i+1}. {line}")                        # print with numbering

print()
print("OBSERVATION (Part A):")
print("  The frequency model always picks the MOST COMMON next word.")
print("  This means it can get stuck in a loop if two words point to each other.")
print("  It has no memory beyond the last 1 word (it is a bigram, not trigram).")
print("  All creativity is limited to which START WORD is sampled randomly.")


# =============================================================================
# =============================================================================
#
#  PART B: nanoGPT-STYLE CHAR-LEVEL MODEL
#          (Machine Learning with character-level generation)
#
# =============================================================================
# =============================================================================

section("PART B: nanoGPT-Style Character-Level Generator (ML)")

print("""
HOW IT WORKS (concept before any code):
  Instead of words, we now work with INDIVIDUAL CHARACTERS.
  The model learns: given character X, what character usually comes next?

  This is "nanoGPT-style" because:
    - It is character-level (not word-level)
    - It uses an Embedding layer + Linear layer (same as real GPT)
    - It trains by minimising cross-entropy loss (same as real GPT)
    - It generates by sampling from a softmax distribution (same as real GPT)
    - The BIG difference: real GPT also has an Attention mechanism that lets
      it look at MANY previous characters. Our model only looks at 1 (bigram).
      Adding attention would make this a full GPT -- that is Module 04!

  WHY CHARACTER-LEVEL FOR THIS PROJECT?
    - Simpler vocabulary (just letters, digits, spaces, punctuation)
    - You can clearly see how temperature affects creativity
    - A word-level model needs more training data to learn well
    - Character-level works on short data like subject lines

  TEMPERATURE SAMPLING EXPLAINED:
    - temp=0.2  -> Very focused. Almost always picks the most likely char.
                   Output: predictable, similar to training data.
    - temp=0.8  -> Balanced. Some creativity. Our default for this project.
                   Output: new but still coherent subject-line-like text.
    - temp=1.5  -> Very creative. Many unlikely chars get a chance.
                   Output: surprising, sometimes nonsensical, fun to read.

  C# analogy for the full pipeline:
    1. char[] vocab = GetUniqueChars(trainingText);         // vocabulary
    2. Dictionary<char,int> charToId = BuildMapping(vocab);  // char -> ID
    3. float[,] embeddings = new float[vocabSize, embedDim]; // lookup table
    4. float[,] weights    = new float[embedDim, vocabSize]; // linear layer
    5. Train(pairs, epochs, lr);                            // backprop loop
    6. string output = Generate(seedChar, temperature);     // sample loop
""")

# =============================================================================
# STEP B1: BUILD THE CHARACTER VOCABULARY
# =============================================================================

subsection("Step B1: Build Character Vocabulary from Subject Lines")

# Join all subject lines into one long training string.
# "\n".join(list) is like String.Join("\n", list) in C#.
# The newline character acts as a sentence boundary -- the model learns
# that after certain characters a new line (and new subject) can begin.
training_text = "\n".join(SUBJECT_LINES)            # one big string

# Get all unique characters in the training text.
# set() removes duplicates. sorted() gives consistent ordering across runs.
# C# analogy: new SortedSet<char>(trainingText.ToCharArray())
chars = sorted(set(training_text))                  # list of unique chars

vocab_size_chars = len(chars)                       # integer: how many unique chars

# Build char-to-ID mapping (encode): char -> integer index
# C# analogy: Dictionary<char, int> charToId = new();
char_to_id = {ch: idx for idx, ch in enumerate(chars)}   # dict comprehension

# Build ID-to-char mapping (decode): integer index -> char
# C# analogy: Dictionary<int, char> idToChar = new();
id_to_char = {idx: ch for ch, idx in char_to_id.items()} # inverted mapping

print()
print(f"Training text length:     {len(training_text)} characters")
print(f"Unique characters (vocab): {vocab_size_chars}")
print()
print("Character vocabulary (ID : character):")
print()

# Print the vocab in 8 columns so it fits on screen
columns = 8                                         # characters per row
for i, ch in enumerate(chars):                      # loop each character
    display = repr(ch)                              # repr shows \n as '\\n' etc.
    print(f"  {i:3d}:{display:<6}", end="")         # print without newline
    if (i + 1) % columns == 0:                      # every 8th item
        print()                                     # start a new row

print()                                             # final newline after last row

# =============================================================================
# STEP B2: ENCODE TRAINING TEXT AND BUILD CHARACTER BIGRAM PAIRS
# =============================================================================

subsection("Step B2: Encode Text and Build Training Pairs")

# Convert each character in the training text to its integer ID.
# C# analogy: int[] tokenIds = trainingText.Select(c => charToId[c]).ToArray();
token_ids_chars = [char_to_id[ch] for ch in training_text]  # list of ints

# Build training pairs: (input_char_id, target_next_char_id)
# Each pair teaches: "when you see char X, predict char Y next."
# C# analogy: List<(int input, int target)> pairs = new();
input_char_ids  = token_ids_chars[:-1]              # all IDs except the last
target_char_ids = token_ids_chars[1:]               # all IDs except the first

print()
print(f"Total training pairs (character bigrams): {len(input_char_ids)}")
print()
print("First 8 training pairs (input char -> target char):")
print()
print(f"  {'Input char':<14} {'Target char':<14} {'IDs'}")
print("  " + "-" * 40)

for i in range(8):                                  # show first 8 pairs
    in_ch  = id_to_char[input_char_ids[i]]          # decode int to char
    out_ch = id_to_char[target_char_ids[i]]         # decode int to char
    in_disp  = repr(in_ch)                          # safe display (shows \n)
    out_disp = repr(out_ch)
    print(f"  {in_disp:<14} {out_disp:<14} "
          f"({input_char_ids[i]:3d} -> {target_char_ids[i]:3d})")

# =============================================================================
# STEP B3: SOFTMAX HELPER (numpy, used by both numpy and PyTorch paths)
# =============================================================================

def softmax_np(x):
    """
    Convert raw logit scores to probabilities using numpy.

    Why subtract max first?
      Without it, exp(large_number) can overflow to infinity.
      Subtracting the max before calling exp() keeps numbers small.
      Mathematically this does NOT change the result (the max cancels out).

    C# analogy: static float[] Softmax(float[] x) { ... }
    """
    shifted = x - x.max()                           # subtract max for numeric stability
    exps    = np.exp(shifted)                        # e^x for each element
    return exps / exps.sum()                         # divide so all values sum to 1.0

# =============================================================================
# STEP B4: NUMPY FALLBACK MODEL
# Runs even if PyTorch is not installed.
# This is a manual implementation of: Embedding -> Linear -> Logits
# =============================================================================

subsection("Step B4: Define the nanoGPT-Style Model (numpy version)")

print("""
  Architecture (same for numpy and PyTorch versions):

    Input: one character ID (integer, e.g. 34 for 'o')
      |
      v
    Embedding layer [vocab_size x embed_dim]
      Looks up a row from the embedding table for this character.
      Each character has its own learned vector of EMBED_DIM numbers.
      C# analogy: float[] embedding = embedTable[charId];
      |
      v
    Linear layer [embed_dim x vocab_size]
      Multiplies the embedding vector by a weight matrix + adds bias.
      Result is a score for EVERY character in the vocabulary.
      C# analogy: float[] logits = MatMul(embedding, weights) + bias;
      |
      v
    Logits [vocab_size]
      One score per character. Higher score = more likely to be next.
      Convert to probabilities with softmax, then sample.
""")

EMBED_DIM_CHARS  = 32    # number of numbers per character embedding vector
LEARN_RATE_CHARS = 0.05  # how big each weight update step is (learning rate)
NUM_EPOCHS_CHARS = 200   # how many times to loop through all training pairs

# Initialise weights with small random values.
# Small values (x0.1) prevent exploding gradients at the start.
# np.random.randn gives numbers from standard normal distribution (mean=0, std=1).
# C# analogy: float[,] W = RandomSmall(vocabSize, embedDim);
np_emb   = np.random.randn(vocab_size_chars, EMBED_DIM_CHARS) * 0.1  # [vocab, embed]
np_W     = np.random.randn(EMBED_DIM_CHARS, vocab_size_chars) * 0.1  # [embed, vocab]
np_b     = np.zeros(vocab_size_chars)                                  # [vocab] bias

def np_char_forward(char_id):
    """
    Forward pass: given a character ID, compute logit scores for every character.

    Steps:
      1. Look up the embedding row for this character.
      2. Multiply embedding by weight matrix (linear transformation).
      3. Add bias.

    Returns:
      numpy array of shape [vocab_size_chars] -- one score per character.

    C# analogy: float[] Forward(int charId) { ... }
    """
    emb    = np_emb[char_id]             # get embedding vector: shape [embed_dim]
    logits = emb @ np_W + np_b           # @ = matrix multiply; shape [vocab_size]
    return logits                        # raw scores

def np_char_loss_and_grad(char_id, target_id):
    """
    Compute cross-entropy loss and gradients for one character pair.

    Cross-entropy loss formula:
      loss = -log( probability of the correct target character )
      If model gives 80% to correct char: loss = -log(0.8) = 0.22  (good)
      If model gives  5% to correct char: loss = -log(0.05) = 3.0  (bad)

    Then compute gradients using the standard softmax + cross-entropy
    combined gradient formula.

    Returns:
      (loss, d_emb, d_W, d_b) -- loss value and gradients for each parameter.

    C# analogy: (float loss, float[] dEmb, float[,] dW, float[] dB)
                ComputeLossAndGrad(int charId, int targetId)
    """
    emb    = np_emb[char_id]             # lookup embedding: shape [embed_dim]
    logits = emb @ np_W + np_b           # forward pass: shape [vocab_size]
    probs  = softmax_np(logits)          # convert to probabilities

    # Cross-entropy loss: -log(prob of correct answer)
    # 1e-9 prevents log(0) which would be negative infinity
    loss   = -np.log(probs[target_id] + 1e-9)

    # Gradient of loss with respect to logits (combined softmax+CE formula):
    #   For the correct target: gradient = prob - 1
    #   For all other chars:   gradient = prob
    d_logits             = probs.copy()  # start with all probabilities
    d_logits[target_id] -= 1.0          # subtract 1 from the correct char's prob

    # Gradient for the weight matrix:
    # d_W = outer product of embedding and d_logits
    # outer product: d_W[i,j] = emb[i] * d_logits[j]
    # C# analogy: result[i,j] = emb[i] * dLogits[j] (double loop)
    d_W = np.outer(emb, d_logits)        # shape [embed_dim, vocab_size]

    # Gradient for bias: same as d_logits (adding bias has gradient = 1)
    d_b = d_logits                       # shape [vocab_size]

    # Gradient for the embedding of this character:
    # chain rule through the linear layer: d_emb = d_logits @ W.T
    d_emb = d_logits @ np_W.T           # shape [embed_dim]

    return loss, d_emb, d_W, d_b        # return all four values

# =============================================================================
# STEP B5: TRAIN THE NUMPY MODEL
# =============================================================================

subsection("Step B5: Train the numpy Character Model")

print(f"  Vocab size:   {vocab_size_chars} unique characters")
print(f"  Embed dim:    {EMBED_DIM_CHARS}")
print(f"  Train pairs:  {len(input_char_ids)}")
print(f"  Epochs:       {NUM_EPOCHS_CHARS}")
print(f"  Learn rate:   {LEARN_RATE_CHARS}")
print()
print(f"  Training numpy model... (this may take a moment)")
print()

for epoch in range(NUM_EPOCHS_CHARS):              # loop over all epochs

    total_loss = 0.0                               # reset loss accumulator

    # Shuffle pairs each epoch so model does not memorise order.
    # C# analogy: list.Shuffle() before each epoch.
    indices = list(range(len(input_char_ids)))     # list [0, 1, 2, ...]
    random.shuffle(indices)                        # shuffle in-place

    for idx in indices:                            # loop every training pair

        c_in  = input_char_ids[idx]                # input character ID
        c_tgt = target_char_ids[idx]               # target (next) character ID

        # Compute loss and gradients for this pair
        loss, d_emb, d_W, d_b = np_char_loss_and_grad(c_in, c_tgt)

        total_loss += loss                         # accumulate loss

        # Gradient descent update: parameter -= learning_rate * gradient
        # This "nudges" each parameter in the direction that reduces loss.
        # C# analogy: param -= learningRate * gradient;
        np_emb[c_in] -= LEARN_RATE_CHARS * d_emb  # update this char's embedding
        np_W         -= LEARN_RATE_CHARS * d_W     # update weight matrix
        np_b         -= LEARN_RATE_CHARS * d_b     # update bias

    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:                     # every 50th epoch
        avg = total_loss / len(input_char_ids)     # average loss per pair
        print(f"  Epoch {epoch+1:>3}/{NUM_EPOCHS_CHARS}  |  avg loss: {avg:.4f}")

print()
print("  numpy model training complete.")

# =============================================================================
# STEP B6: PyTorch MODEL (used if PyTorch is installed, graceful fallback if not)
# =============================================================================

subsection("Step B6: PyTorch nanoGPT-Style Model (if available)")

# try/except ImportError is the Python way to handle optional dependencies.
# C# analogy: try { Assembly.Load("torch"); } catch (FileNotFoundException) { }
try:
    import torch                         # core PyTorch library
    import torch.nn as nn                # neural network building blocks
    import torch.optim as optim          # optimisers (Adam, SGD, etc.)

    PYTORCH_AVAILABLE = True             # flag for later use
    print(f"  PyTorch found (version {torch.__version__}). Using PyTorch model.")

except ImportError:                      # PyTorch not installed
    PYTORCH_AVAILABLE = False            # flag: use numpy fallback
    print("  PyTorch not found. Using numpy model for generation.")
    print("  Install with: pip install torch")

if PYTORCH_AVAILABLE:                    # only define class if PyTorch is available

    # In PyTorch, every neural network is a CLASS that inherits from nn.Module.
    # C# analogy: class NanoGptChar : NeuralNetworkBase { ... }
    class NanoGptCharModel(nn.Module):
        """
        A nanoGPT-inspired character-level bigram language model.

        Architecture: Embedding -> Linear -> Logits
        This is the simplest possible GPT-style model:
          - Same embedding lookup as real GPT
          - Same linear projection as real GPT
          - Missing: multi-head attention and transformer blocks
            (those are covered in Module 04 -- Transformers)

        C# analogy:
          class NanoGptCharModel : NeuralNetworkBase {
              EmbeddingLayer embeddings;   // [vocabSize, embedDim]
              LinearLayer    linear;       // [embedDim, vocabSize]
          }
        """

        def __init__(self, vocab_size, embed_dim):
            """
            Constructor: define layers.
            super().__init__() MUST be called first -- it sets up PyTorch internals.
            C# analogy: base() call in a constructor.
            """
            super().__init__()                            # call nn.Module constructor

            # nn.Embedding: a lookup table of shape [vocab_size, embed_dim].
            # Given an integer ID, it returns the corresponding row (a vector).
            # C# analogy: float[,] embTable = new float[vocabSize, embedDim];
            self.embedding = nn.Embedding(vocab_size, embed_dim)

            # nn.Linear: a linear transformation y = x * W^T + b.
            # Input size: embed_dim. Output size: vocab_size.
            # C# analogy: float[] Linear(float[] x) { return MatMul(x, W) + b; }
            self.linear = nn.Linear(embed_dim, vocab_size)

        def forward(self, char_id_tensor):
            """
            Forward pass: compute logits for the given input character ID.
            PyTorch calls this automatically when you do model(input).
            C# analogy: override float[] Forward(int[] charIds) { ... }
            """
            emb    = self.embedding(char_id_tensor)   # lookup embedding: [batch, embed]
            logits = self.linear(emb)                 # linear layer: [batch, vocab]
            return logits                             # raw scores

    # --- Instantiate and train the PyTorch model ---

    PT_EMBED  = 32      # embedding dimension (same as numpy model)
    PT_EPOCHS = 200     # training epochs
    PT_LR     = 0.05    # learning rate for Adam

    # Create model instance
    # C# analogy: var model = new NanoGptCharModel(vocabSize, embedDim);
    pt_model = NanoGptCharModel(vocab_size_chars, PT_EMBED)

    # nn.CrossEntropyLoss combines softmax + negative log likelihood.
    # It expects raw logits (NOT softmax probabilities) as input.
    # C# analogy: var lossFn = new CrossEntropyLoss();
    criterion = nn.CrossEntropyLoss()

    # Adam optimiser: adapts the learning rate per parameter automatically.
    # Better than plain gradient descent for most deep learning tasks.
    # C# analogy: var optimizer = new AdamOptimizer(model.Parameters, lr);
    optimizer = optim.Adam(pt_model.parameters(), lr=PT_LR)

    # Convert Python lists to PyTorch tensors.
    # torch.long = int64, required for embedding layer input.
    # C# analogy: like converting List<int> to a GPU-compatible array.
    in_tensor  = torch.tensor(input_char_ids,  dtype=torch.long)  # input IDs
    tgt_tensor = torch.tensor(target_char_ids, dtype=torch.long)  # target IDs

    print()
    print(f"  Training PyTorch model for {PT_EPOCHS} epochs...")
    print()

    for epoch in range(PT_EPOCHS):                    # loop over epochs

        optimizer.zero_grad()                         # clear old gradients
        # WHY clear gradients? PyTorch ACCUMULATES them by default.
        # If we do not clear, gradients from the previous step add up incorrectly.
        # C# analogy: like resetting a running total to 0 each loop iteration.

        logits = pt_model(in_tensor)                  # forward pass: [N, vocab_size]
        loss   = criterion(logits, tgt_tensor)        # compute loss (scalar)

        loss.backward()                               # backprop: compute all gradients
        # PyTorch builds a "computation graph" during forward pass and
        # walks it backwards to find each parameter's gradient automatically.
        # C# analogy: the framework does the calculus for you.

        optimizer.step()                              # update all weights using gradients

        if (epoch + 1) % 50 == 0:                    # print every 50 epochs
            print(f"  Epoch {epoch+1:>3}/{PT_EPOCHS}  |  loss: {loss.item():.4f}")

    print()
    print("  PyTorch model training complete.")

    def get_char_logits(char_id):
        """
        Get logits from the PyTorch model as a numpy array.

        torch.no_grad() disables gradient tracking during inference.
        This saves memory and speeds things up -- we do not need gradients
        when we are just generating (not training).
        C# analogy: model.Eval() or inference-only mode.
        """
        with torch.no_grad():                                     # no gradient tracking
            t = torch.tensor([char_id], dtype=torch.long)        # wrap ID in tensor
            logits = pt_model(t)                                  # forward pass
            return logits[0].detach().numpy()                     # return as numpy

    USE_PYTORCH = True                                            # use PyTorch for generation

else:
    # PyTorch not available -- fall back to the numpy model.
    USE_PYTORCH = False

    def get_char_logits(char_id):
        """Fallback: use numpy model when PyTorch is not available."""
        return np_char_forward(char_id)                           # numpy forward pass

def get_logits_unified(char_id):
    """
    Unified logit function: dispatches to PyTorch or numpy based on availability.
    All generation code below calls THIS function -- it does not care which backend.
    C# analogy: calling an interface method that dispatches to the right implementation.
    """
    if USE_PYTORCH:                                               # PyTorch path
        return get_char_logits(char_id)                          # PyTorch logits
    else:                                                         # numpy path
        return np_char_forward(char_id)                          # numpy logits

# =============================================================================
# STEP B7: TEMPERATURE SAMPLING FUNCTION
# =============================================================================

subsection("Step B7: Temperature Sampling")

print("""
  Temperature sampling steps:
    1. Get logit scores from the model (one score per character).
    2. Divide ALL logits by the temperature value.
    3. Apply softmax to convert to probabilities.
    4. Sample one character index from that probability distribution.

  Effect of temperature:
    temp = 0.2  Divides by a small number, making scores MORE extreme.
                The highest score becomes MUCH higher than the rest.
                Result: almost always picks the most likely character.

    temp = 0.8  Divides by a medium number. Moderate smoothing.
                Result: mostly picks likely chars, with some surprises.

    temp = 1.5  Divides by a large number, flattening the distribution.
                All characters get more similar probabilities.
                Result: picks unusual characters more often.
""")

def temperature_sample(logits, temperature):
    """
    Pick the next character using temperature sampling.

    Parameters:
      logits      - numpy array of raw scores, shape [vocab_size]
      temperature - float controlling randomness
                    lower = focused, higher = creative

    Returns:
      Integer: the index of the sampled next character.

    C# analogy:
      int TemperatureSample(float[] logits, float temperature)
    """
    if temperature <= 0.0:                           # temperature of 0 = pure greedy
        return int(np.argmax(logits))                # just pick the max (no randomness)

    scaled = logits / temperature                    # divide ALL scores by temperature
    probs  = softmax_np(scaled)                      # convert to probabilities

    # np.random.choice picks one index weighted by probs.
    # probs must sum to exactly 1.0 (softmax guarantees this).
    # C# analogy: like a weighted dice roll over vocab_size faces.
    return int(np.random.choice(len(probs), p=probs))

# =============================================================================
# STEP B8: GENERATE SUBJECT LINES WITH THE CHAR MODEL
# =============================================================================

subsection("Step B8: Generate Character-Level Subject Lines")

def generate_subject_char(seed_char, num_chars, temperature):
    """
    Generate a sequence of characters starting from seed_char.

    Algorithm:
      1. Start with seed_char.
      2. Get logits from the model for the current character.
      3. Sample the next character using temperature.
      4. Append to output, advance to next character, repeat.
      5. Stop at num_chars or when a newline character is generated.

    Parameters:
      seed_char   - first character to start generation from (string of length 1)
      num_chars   - maximum characters to generate
      temperature - float controlling creativity

    Returns:
      The generated text as a string (may be multi-word like a subject line).

    C# analogy:
      string GenerateSubject(char seedChar, int numChars, float temperature)
    """
    if seed_char not in char_to_id:                  # seed character not in vocab
        return f"[Unknown char: '{seed_char}']"      # return error message

    current_id = char_to_id[seed_char]               # convert char to integer ID
    output     = [seed_char]                         # start output list with seed

    for _ in range(num_chars):                       # generate up to num_chars more
        logits   = get_logits_unified(current_id)    # get scores from model
        next_id  = temperature_sample(logits, temperature)  # sample next char
        next_ch  = id_to_char[next_id]               # decode ID back to character

        if next_ch == "\n":                          # newline = end of subject line
            break                                    # stop generation here

        output.append(next_ch)                       # add character to output
        current_id = next_id                         # advance to next character

    result = "".join(output)                         # join char list into string
    if result:                                       # if result is not empty
        result = result[0].upper() + result[1:]      # capitalise first char

    return result                                    # return finished subject line

# =============================================================================
# STEP B9: DEMONSTRATE TEMPERATURE EFFECT
# Generate 5 subject lines at each of 3 temperature settings
# =============================================================================

subsection("Step B9: Temperature Comparison -- 5 Lines Each")

# Seed characters to start generation from.
# We use common starting characters from the training data.
seed_chars = ["5", "D", "Y", "E", "L", "N", "G", "W", "T", "F", "H", "M"]

# Filter to only seeds that exist in vocab (safety)
valid_seeds = [s for s in seed_chars if s in char_to_id]  # list of valid seeds

# Temperature settings to compare
temperature_settings = [
    (0.2, "FOCUSED  (temp=0.2)  -- predictable, close to training data"),
    (0.8, "BALANCED (temp=0.8)  -- some creativity, default for marketing"),
    (1.5, "CREATIVE (temp=1.5)  -- surprising, may be incoherent"),
]

for temp_value, temp_label in temperature_settings:     # loop over 3 temperatures

    print()
    print(f"  {temp_label}")
    print("  " + "-" * 55)

    np.random.seed(10)                                  # same seed for fair comparison

    for i in range(5):                                  # generate 5 lines per setting
        seed = valid_seeds[i % len(valid_seeds)]        # cycle through valid seeds
        line = generate_subject_char(                   # generate one line
            seed_char=seed,
            num_chars=50,                               # allow up to 50 characters
            temperature=temp_value                      # use this temperature
        )
        print(f"  {i+1}. {line}")

print()
print("=" * 60)
print("TEMPERATURE OBSERVATIONS:")
print("=" * 60)
print("""
  temp=0.2  Lines look very similar to the training subjects.
            The model is confident and repeats common patterns.
            e.g. repeating "your" or "offer" frequently.

  temp=0.8  Lines feel like plausible new marketing subjects.
            Some fresh word combinations that were not in training.
            This is the SWEET SPOT for a real email generator.

  temp=1.5  Lines may start sensibly but drift into unusual character
            combinations. The model is exploring more of the space.
            Useful for brainstorming wild, attention-grabbing ideas.
""")

# =============================================================================
# SUMMARY
# =============================================================================

section("PROJECT 4 SUMMARY")

print("""
  What you built:
  ---------------
  PART A - Frequency Bigram (Word Level):
    - Counted word pairs from 15 marketing subject lines
    - Stored counts in a nested dictionary (bigram table)
    - Generated 5 new subject lines by chaining greedy predictions
    - Started each line by sampling a weighted start word

  PART B - nanoGPT-Style Char Level (Machine Learning):
    - Built a character vocabulary (unique chars as integer IDs)
    - Created (input char, target char) training pairs
    - Trained a tiny Embedding + Linear model
        numpy version : manual backprop and gradient descent
        PyTorch version : automatic backprop with Adam optimiser
    - Generated subject lines using temperature sampling
    - Compared output at temp=0.2, temp=0.8, and temp=1.5

  Key Concepts Demonstrated:
  --------------------------
  - Frequency models memorise exact pairs; ML models learn patterns
  - Character-level vocab is small but needs many chars per word
  - Temperature controls the creativity vs coherence trade-off
  - The same embedding + linear architecture powers real GPT models
    (plus attention and more layers, covered in Module 04)

  Real-World Connection:
  ----------------------
  - Mailchimp, HubSpot, Salesforce all use LLMs for subject lines
  - Real tools use temp ~ 0.7-0.9 for marketing copy generation
  - A/B testing tools then pick the subject line with the highest
    open rate from a set of generated candidates

  C# Analogy Recap:
  -----------------
  - word_bigram     ->  Dictionary<string, Dictionary<string, int>>
  - char_to_id      ->  Dictionary<char, int>
  - np_emb          ->  float[,] embTable (vocab x embedDim)
  - np_W            ->  float[,] weights  (embedDim x vocab)
  - softmax_np      ->  float[] Softmax(float[] scores)
  - temperature     ->  float creativityDial (lower = safe, higher = wild)
  - cross-entropy   ->  float Penalty(float correctProb)  [lower = better]
  - epoch           ->  int fullPassesThroughData
""")

# =============================================================================
# QUIZ QUESTIONS
# (Try to answer these before reading the answers!)
# =============================================================================

section("QUIZ QUESTIONS")

print("""
  Q1. In Part A, the frequency bigram model always picks the MOST COMMON
      next word. What is the main drawback of this greedy approach?

      A) It is too slow to run on a laptop
      B) It can get stuck in a loop and never generates new word combinations
      C) It requires a GPU to compute
      D) It cannot handle punctuation

  A1. B) It can get stuck in a loop.
         If "offer" is most often followed by "ends", and "ends" is most
         often followed by "offer", the model loops forever: "offer ends
         offer ends offer ...". Greedy selection has no randomness to escape
         such loops. Temperature sampling in Part B fixes this.

  -----------------------------------------------------------------------

  Q2. In Part B, what does temperature=0.2 do to the probability
      distribution before sampling?

      A) It makes all characters equally likely (uniform distribution)
      B) It makes the most likely character even MORE dominant
      C) It makes unlikely characters more likely than likely characters
      D) It doubles the learning rate during training

  A2. B) It makes the most likely character even MORE dominant.
         Dividing logits by 0.2 is like multiplying them by 5.
         This stretches the gap between high and low scores.
         After softmax, the top character gets a probability very close
         to 1.0, while all others become nearly 0.
         Think of it as "sharpening" the distribution.

  -----------------------------------------------------------------------

  Q3. The nanoGPT-style model in Part B is a BIGRAM model at the character
      level. What is the key capability that is MISSING compared to a real
      GPT model, and which module covers it?

      A) GPT uses words instead of characters -- covered in Module 02
      B) GPT uses multi-head attention to look at many previous tokens,
         not just the last one -- covered in Module 04 (Transformers)
      C) GPT uses a different loss function -- covered in Module 06
      D) GPT uses a larger vocabulary -- covered in Module 03

  A3. B) Multi-head attention.
         Our model only looks at the LAST character to predict the next one
         (that is the definition of a bigram model).
         A real GPT uses attention to look at ALL previous characters in the
         context window simultaneously. This lets it understand long-range
         patterns like "this email is about a sale" from the first 3 words,
         and use that context 20 characters later. Module 04 builds the
         full transformer with attention from scratch.
""")

print("=" * 60)
print("Project 4 complete!")
print("Module 05 projects done: 1, 2, 3, 4")
print("=" * 60)

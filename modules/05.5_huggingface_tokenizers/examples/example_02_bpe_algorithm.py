"""
Module 05.5 - HuggingFace Tokenizers
Example 02: BPE Algorithm From Scratch

GLOSSARY
--------
BPE          : Byte Pair Encoding. A text compression algorithm adapted for NLP.
               Builds a subword vocabulary by iteratively merging frequent character pairs.
               C# analogy: like LZW compression that builds a dictionary from training data.

Merge Rule   : A pair of tokens (A, B) -> new token AB.
               E.g., ('e', 'r') -> 'er' means whenever 'e' is followed by 'r', merge them.
               Stored as an ordered list: rule 1 applied first, rule 2 second, etc.

Corpus       : The text we train on. BPE learns merge rules from the corpus.
               C# analogy: the training dataset.

Frequency    : How many times a token pair appears next to each other in the corpus.
               The most frequent pair is merged at each step.

Vocabulary   : The set of all known tokens after training.
               Starts as individual characters, grows by one new token per merge.

Pre-tokenize : Split raw text into word-level units BEFORE BPE runs.
               BPE merges happen WITHIN words, not across word boundaries.
               C# analogy: Regex.Split() before applying your actual parser.

This file implements BPE completely from scratch. No library needed.
"""

from collections import defaultdict    # defaultdict: like Dictionary<K,V> but auto-initializes
                                       # defaultdict(int) -> any missing key starts at 0
                                       # C# analogy: dict.GetValueOrDefault(key, 0)
import re                              # re: regular expressions

print("=" * 60)
print("Example 02: BPE Algorithm (Implemented From Scratch)")
print("=" * 60)
print()


# ============================================================
# PART A: Understanding Pair Frequency Counting
# ============================================================

print("PART A: Counting Pair Frequencies")
print("-" * 50)
print()

def get_vocab_from_corpus(corpus):
    """
    Convert a corpus (list of sentences) into a word-frequency dictionary.
    Each word is represented as a tuple of characters + end-of-word marker.

    BPE classic implementation uses </w> to mark word boundaries.
    Example: "low" -> ('l', 'o', 'w', '</w>')

    Why </w>? So the tokenizer knows where words end.
    "low" + "</w>" and "lower" are different: "lower" does NOT have </w> after "low".

    C# analogy:
      var wordFreqs = corpus
        .SelectMany(s => s.ToLower().Split())
        .GroupBy(w => w)
        .ToDictionary(g => Tuple.Create(g.Key.ToCharArray()), g => g.Count());

    Args:
        corpus (list[str]): list of text strings

    Returns:
        dict: {tuple_of_chars: frequency} e.g., {('l','o','w','</w>'): 5}
    """
    word_freq = defaultdict(int)                    # auto-initializes missing keys to 0

    for sentence in corpus:                         # loop over each sentence
        words = sentence.lower().split()            # lowercase and split on whitespace
        for word in words:                          # loop over each word
            # Convert word to tuple of characters + end-of-word marker
            char_tuple = tuple(list(word) + ["</w>"])
            # list(word) = ['l', 'o', 'w']
            # + ["</w>"] = ['l', 'o', 'w', '</w>']
            # tuple() converts list to tuple (tuples are hashable, can be dict keys)
            word_freq[char_tuple] += 1              # increment frequency count

    return dict(word_freq)                          # convert defaultdict to regular dict


# Small training corpus
corpus = [
    "low low low low low",      # "low" appears 5 times
    "lower lower",               # "lower" appears 2 times
    "newest newest newest",      # "newest" appears 3 times
    "widest widest",             # "widest" appears 2 times
]

vocab = get_vocab_from_corpus(corpus)

print("Initial vocabulary (before any merges):")
print("Each word is split into characters + </w> end marker.")
print()
for word_tuple, freq in sorted(vocab.items(), key=lambda x: -x[1]):  # sort by frequency
    word_str = " ".join(word_tuple)    # join tuple elements with spaces for display
    print(f"  {word_str:<30} frequency: {freq}")
    # :<30 = left-justify in a field of width 30 (for aligned output)
print()


# ============================================================
# PART B: Finding the Most Frequent Pair
# ============================================================

print("PART B: Finding the Most Frequent Adjacent Pair")
print("-" * 50)
print()

def get_pair_frequencies(vocab):
    """
    Count how often each adjacent pair of tokens appears across the vocabulary.

    For each word, look at every consecutive pair of tokens.
    Weight by the word's frequency (a word appearing 5 times contributes 5x to each pair).

    Example:
      Word ('l','o','w','</w>') with freq=5:
        pairs: ('l','o')=+5, ('o','w')=+5, ('w','</w>')=+5

    C# analogy:
      var pairs = vocab.SelectMany(entry =>
        Enumerable.Range(0, entry.Key.Length - 1)
          .Select(i => (entry.Key[i], entry.Key[i+1]))
          .Select(pair => (pair, entry.Value))
      ).GroupBy(x => x.pair).ToDictionary(g => g.Key, g => g.Sum(x => x.Value));

    Args:
        vocab (dict): {tuple_of_tokens: frequency}

    Returns:
        dict: {(token_a, token_b): frequency}
    """
    pairs = defaultdict(int)            # accumulate pair frequencies

    for word_tuple, freq in vocab.items():    # loop: (word_as_token_tuple, word_frequency)
        symbols = list(word_tuple)            # convert tuple to list for indexing

        for i in range(len(symbols) - 1):    # range(n-1) = 0, 1, ..., n-2
                                              # we look at pairs: symbols[i] and symbols[i+1]
            pair = (symbols[i], symbols[i + 1])   # create pair tuple
            pairs[pair] += freq               # add word's frequency to this pair's count

    return dict(pairs)                        # return regular dict


# Count pairs in our initial vocabulary
pairs = get_pair_frequencies(vocab)

# Sort by frequency, highest first
sorted_pairs = sorted(pairs.items(), key=lambda x: -x[1])
# .items() = (key, value) pairs
# key=lambda x: -x[1] = sort by value descending (negate for descending sort)

print("Top 10 adjacent pairs by frequency:")
for (tok_a, tok_b), freq in sorted_pairs[:10]:  # [:10] = first 10 items
    print(f"  ('{tok_a}', '{tok_b}') -> frequency: {freq}")

# Find the best pair to merge
best_pair = sorted_pairs[0][0]    # [0] = first item (most frequent), [0] = the pair tuple
print()
print(f"Best pair to merge: {best_pair} (frequency: {sorted_pairs[0][1]})")
print()


# ============================================================
# PART C: Performing a Merge
# ============================================================

print("PART C: Performing a Merge Step")
print("-" * 50)
print()

def merge_vocab(vocab, best_pair):
    """
    Apply one merge: replace all occurrences of best_pair with the merged token.

    For each word in the vocabulary:
    - Scan through token sequence
    - Wherever (best_pair[0], best_pair[1]) appears consecutively, replace with merged

    C# analogy: string.Replace("er", "er_merged") but operating on token sequences.

    Args:
        vocab (dict): current {tuple_of_tokens: frequency}
        best_pair (tuple): (token_a, token_b) pair to merge

    Returns:
        dict: new vocabulary with best_pair merged everywhere it appears
    """
    new_vocab = {}                          # new vocabulary after this merge

    merged_token = best_pair[0] + best_pair[1]  # e.g., ('e','s') -> 'es'
    # String concatenation: just join the two tokens together

    for word_tuple, freq in vocab.items():  # loop over all words
        symbols = list(word_tuple)          # convert tuple to list (mutable)
        new_symbols = []                    # list to build merged version

        i = 0                               # current position in symbols
        while i < len(symbols):             # scan from left to right
            # Check if current symbol and next symbol form the best pair
            if (i < len(symbols) - 1 and   # not at the last position
                    symbols[i] == best_pair[0] and     # current matches first of pair
                    symbols[i + 1] == best_pair[1]):   # next matches second of pair
                new_symbols.append(merged_token)        # add merged token instead
                i += 2                      # skip both symbols (we merged them)
            else:
                new_symbols.append(symbols[i])          # keep this symbol as-is
                i += 1                      # move to next symbol

        new_vocab[tuple(new_symbols)] = freq    # store updated word with same frequency

    return new_vocab                        # return new vocabulary


# Perform the first merge
print(f"Merging pair: {best_pair} -> '{best_pair[0] + best_pair[1]}'")
print()
print("Before merge:")
for word_tuple, freq in sorted(vocab.items(), key=lambda x: -x[1]):
    print(f"  {' '.join(word_tuple):<35} (freq: {freq})")

new_vocab = merge_vocab(vocab, best_pair)

print()
print("After merge:")
for word_tuple, freq in sorted(new_vocab.items(), key=lambda x: -x[1]):
    print(f"  {' '.join(word_tuple):<35} (freq: {freq})")

print()


# ============================================================
# PART D: Full BPE Training Loop
# ============================================================

print("PART D: Full BPE Training (10 Merge Steps)")
print("-" * 50)
print()

def train_bpe(corpus, num_merges):
    """
    Run the complete BPE training algorithm.

    Algorithm:
    1. Start with character-level vocabulary
    2. Repeat num_merges times:
       a. Count pair frequencies across all words
       b. Find most frequent pair
       c. Merge that pair into new token
       d. Record merge rule
    3. Return: final vocabulary + ordered list of merge rules

    C# analogy:
      This is like LZW compression dictionary building:
      var dictionary = new List<string>();
      // Initialize with all possible single characters
      // Then iteratively find common sequences and add them

    Args:
        corpus (list[str]): Training corpus
        num_merges (int): Number of merge operations to perform

    Returns:
        tuple: (final_vocab_dict, list_of_merge_rules)
    """
    # Step 1: Initialize vocabulary with character-level representation
    vocab = get_vocab_from_corpus(corpus)       # start with char-level vocab

    # Collect all the merge rules (ordered list)
    merge_rules = []                            # list to store (pair, merged_token) tuples

    print(f"Starting BPE training with {num_merges} merge steps...")
    print()

    # Step 2: Iteratively merge most frequent pair
    for step in range(1, num_merges + 1):       # step 1, 2, 3, ..., num_merges
        # Count all adjacent pairs in current vocabulary
        pairs = get_pair_frequencies(vocab)

        if not pairs:                           # if no pairs left (all single tokens)
            print(f"  No more pairs to merge at step {step}. Stopping.")
            break

        # Find the most frequent pair
        best_pair = max(pairs, key=pairs.get)  # max() with key= finds max by that function
        # pairs.get is the function that returns the value for a given key
        # So this finds the key with the highest value (frequency)

        best_freq = pairs[best_pair]           # get frequency of best pair
        merged = best_pair[0] + best_pair[1]  # the new merged token

        # Record this merge rule
        merge_rules.append((best_pair, merged))

        # Apply merge to vocabulary
        vocab = merge_vocab(vocab, best_pair)

        # Print progress
        print(f"  Step {step:2d}: merge {best_pair} -> '{merged}' (freq={best_freq})")

    print()
    return vocab, merge_rules


# Train BPE on our small corpus
final_vocab, merge_rules = train_bpe(corpus, num_merges=10)


# ============================================================
# PART E: Extracting the Token Set from Final Vocabulary
# ============================================================

print("PART E: Final Vocabulary Tokens")
print("-" * 50)
print()

def build_token_set(final_vocab):
    """
    Extract all unique tokens that appear in the final vocabulary.

    After all merges, words are represented as tuples of tokens.
    We collect all unique tokens across all words.

    C# analogy:
      var tokenSet = finalVocab.Keys
        .SelectMany(wordTuple => wordTuple)
        .Distinct()
        .OrderBy(t => t)
        .ToHashSet();

    Args:
        final_vocab (dict): {tuple_of_tokens: frequency} after training

    Returns:
        set: all unique token strings
    """
    token_set = set()                           # set: like HashSet<string> in C#
    for word_tuple in final_vocab.keys():       # loop over each word (as token tuple)
        for token in word_tuple:                # loop over each token in the word
            token_set.add(token)                # add to set (set ignores duplicates)
    return token_set


tokens = build_token_set(final_vocab)
sorted_tokens = sorted(tokens)              # sort for consistent display

# Assign IDs (simple index-based assignment)
token_vocab = {tok: idx for idx, tok in enumerate(["[PAD]", "[UNK]"] + sorted_tokens)}
# enumerate() gives (index, value) pairs
# Dict comprehension: {value: index} for each (index, value) pair

print(f"Total unique tokens after {len(merge_rules)} merges: {len(token_vocab)}")
print()
print("Final token vocabulary:")
for token, token_id in sorted(token_vocab.items(), key=lambda x: x[1]):  # sort by ID
    print(f"  '{token}' -> ID {token_id}")

print()


# ============================================================
# PART F: BPE Encoding (Inference)
# ============================================================

print("PART F: BPE Encoding - Applying Merge Rules to New Text")
print("-" * 50)
print()

def bpe_encode(text, merge_rules, token_vocab):
    """
    Encode text using trained BPE merge rules (inference phase).

    Steps:
    1. Pre-tokenize: split on whitespace, add </w> end marker
    2. Apply merge rules in training order (first rule first)
    3. Look up each final token in vocabulary to get ID

    C# analogy:
      This is like applying a substitution cipher in order.
      Apply rule 1 everywhere, then apply rule 2 everywhere, etc.

    Args:
        text (str): New text to tokenize
        merge_rules (list): Ordered list of (pair, merged_token) from training
        token_vocab (dict): {token_string: token_id}

    Returns:
        tuple: (list of token strings, list of token IDs)
    """
    # Step 1: Pre-tokenize into words + add end-of-word marker
    words = text.lower().split()                        # split on whitespace
    word_tokens = []                                    # list to collect all word tokens

    for word in words:                                  # process each word
        # Start with character-level representation
        symbols = list(word) + ["</w>"]                # chars + end marker
        # symbols = ['l', 'o', 'w', '</w>'] for word "low"

        # Step 2: Apply each merge rule in training order
        for (pair_a, pair_b), merged in merge_rules:   # unpack each merge rule
            i = 0
            new_symbols = []                            # build merged symbol list
            while i < len(symbols):
                if (i < len(symbols) - 1 and           # not at end
                        symbols[i] == pair_a and        # current matches pair first
                        symbols[i + 1] == pair_b):      # next matches pair second
                    new_symbols.append(merged)          # replace with merged token
                    i += 2                              # skip both
                else:
                    new_symbols.append(symbols[i])      # keep as-is
                    i += 1
            symbols = new_symbols                       # update symbol list for next rule

        word_tokens.extend(symbols)                     # add this word's tokens to list

    # Step 3: Convert tokens to IDs
    ids = []
    for token in word_tokens:
        if token in token_vocab:                        # if token is in vocabulary
            ids.append(token_vocab[token])
        else:                                           # unknown token
            ids.append(token_vocab.get("[UNK]", 1))    # use [UNK] ID (default: 1)

    return word_tokens, ids                             # return strings and IDs


# Test encoding on the training words and some variations
test_texts = [
    "low",          # should be in vocab (frequent in training)
    "lower",        # should split into ["low", "er</w>"] or similar
    "newest",       # should split based on learned merges
    "widest",       # should split based on learned merges
    "fast",         # NOT in training data -> character-level fallback
]

print("BPE Encoding Results (applying learned merge rules):")
print()
for text in test_texts:
    try:
        tok_strings, tok_ids = bpe_encode(text, merge_rules, token_vocab)
        known = "in-vocab" if all(t in token_vocab for t in tok_strings) else "has-UNK"
        print(f"  '{text}':")
        print(f"    tokens: {tok_strings}")
        print(f"    IDs:    {tok_ids} ({known})")
    except Exception as e:
        print(f"  '{text}': ERROR - {e}")
    print()


# ============================================================
# PART G: ASCII Visualization of BPE Training Steps
# ============================================================

print("PART G: BPE Training Step Visualization")
print("-" * 50)
print()

print("Tracing BPE on tiny corpus 'aaabdaaabac':")
print()

# Classic BPE paper example
tiny_corpus = ["aaabdaaabac"]
tiny_vocab = get_vocab_from_corpus(tiny_corpus)

print("Initial (character-level):")
for wt, freq in tiny_vocab.items():
    print(f"  {' '.join(wt)} (freq={freq})")

tiny_merges = []
for step_num in range(1, 4):            # do 3 merge steps
    pairs = get_pair_frequencies(tiny_vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)    # most frequent pair
    merged = best[0] + best[1]
    tiny_merges.append((best, merged))
    tiny_vocab = merge_vocab(tiny_vocab, best)

    print()
    print(f"Step {step_num}: merge {best} -> '{merged}'")
    for wt, freq in tiny_vocab.items():
        print(f"  {' '.join(wt)} (freq={freq})")

print()
print("Merge rules learned:")
for i, ((a, b), m) in enumerate(tiny_merges, 1):  # enumerate starting at 1
    print(f"  Rule {i}: ('{a}', '{b}') -> '{m}'")


print()
print("=" * 60)
print("Example 02 Complete!")
print()
print("Key Takeaways:")
print("  1. BPE starts with characters, merges most frequent pair each step")
print("  2. </w> marker shows word boundaries so 'low' != 'lower'")
print("  3. Each merge adds one new token to the vocabulary")
print("  4. Training: iterate merges. Inference: apply merge rules in order.")
print("  5. GPT-2 does this 50,000 times on 40GB of text -> 50,257 tokens")
print("=" * 60)

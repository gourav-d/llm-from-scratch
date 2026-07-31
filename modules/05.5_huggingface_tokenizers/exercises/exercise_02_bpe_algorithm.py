"""
Module 05.5 - HuggingFace Tokenizers
Exercise 02: BPE Algorithm

GLOSSARY
--------
BPE          : Byte Pair Encoding. Builds a subword vocabulary by merging
               the most frequent adjacent token pair, iteratively.
               C# analogy: Like LZW compression building a symbol table.

Pair         : Two adjacent tokens in a word sequence.
               ('e','r') means token 'e' is immediately followed by token 'r'.
               C# analogy: a Tuple<string, string> representing adjacent tokens.

Frequency    : Number of times a pair appears in the corpus (weighted by word count).
               The most frequent pair is merged at each BPE training step.

Merge Rule   : (pair_a, pair_b) -> merged_token.
               Merge rules are applied IN ORDER during encoding.
               C# analogy: an ordered List<(string, string, string)> of substitution rules.

Corpus Vocab : After pre-tokenizing a corpus, each word is a tuple of chars.
               E.g., "low" appears 5x -> {('l','o','w','</w>'): 5}
               C# analogy: Dictionary<ImmutableArray<string>, int>

</w>          : End-of-word marker appended to each word's character sequence.
               Distinguishes "low" (has </w>) from "lower" (no </w> after 'w').
               C# analogy: a sentinel value (like '\0' in C strings).
"""

from collections import defaultdict    # defaultdict: auto-initializes missing keys

print("=" * 60)
print("Exercise 02: BPE Algorithm")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Build the initial character-level vocabulary
#
#  Background:
#    BPE starts with a character-level vocabulary.
#    Each word from the corpus is represented as a tuple of characters + </w>.
#    "low" appears 3 times -> key: ('l','o','w','</w>'), value: 3
#    "lower" appears 2 times -> key: ('l','o','w','e','r','</w>'), value: 2
#
#  Your Task:
#    Complete build_initial_vocab(corpus) that returns a dict of
#    {character_tuple: word_frequency}.
#
#  C# Analogy:
#    var vocab = corpus
#      .SelectMany(s => s.ToLower().Split())
#      .GroupBy(w => w)
#      .ToDictionary(
#        g => ImmutableArray.Create(g.Key.Select(c => c.ToString()).Append("</w>").ToArray()),
#        g => g.Count()
#      );
# ============================================================

def build_initial_vocab(corpus):
    """
    Build the character-level vocabulary from a corpus.

    For each word in the corpus:
    1. Split into individual characters
    2. Append the end-of-word marker "</w>"
    3. Convert to a tuple (tuples are hashable, can be dict keys)
    4. Count how many times each word appears

    Example:
      corpus = ["low low lower"]
      "low" appears 2 times: key = ('l','o','w','</w>'), value = 2
      "lower" appears 1 time: key = ('l','o','w','e','r','</w>'), value = 1

    Args:
        corpus (list[str]): List of text strings

    Returns:
        dict: {char_tuple: frequency}
    """
    vocab = defaultdict(int)            # missing keys start at 0

    for sentence in corpus:             # loop over each sentence
        words = sentence.lower().split()  # lowercase + split on whitespace
        for word in words:              # loop over each word
            if word:                    # skip empty strings
                # TODO: Create a tuple of characters + "</w>" marker
                # Hint: list(word) gives ['l', 'o', 'w'] for "low"
                # Hint: + ["</w>"] adds the end marker
                # Hint: tuple(...) converts list to hashable tuple
                key = None              # replace None
                # YOUR CODE HERE
                pass                    # remove this

                vocab[key] += 1         # increment count

    return dict(vocab)                  # convert to regular dict


# Test Exercise 1
print("EXERCISE 1: Build Initial Character-Level Vocabulary")
print("-" * 40)
try:
    corpus = [
        "low low low low low",          # "low" x 5
        "lower lower",                   # "lower" x 2
        "newest newest newest",          # "newest" x 3
        "widest widest",                 # "widest" x 2
    ]

    initial_vocab = build_initial_vocab(corpus)
    print(f"  Vocabulary ({len(initial_vocab)} unique words):")
    for word_tuple, freq in sorted(initial_vocab.items(), key=lambda x: -x[1]):
        print(f"    {' '.join(word_tuple):<30} freq={freq}")

    # Verify specific entries
    expected_low = ('l', 'o', 'w', '</w>')
    expected_lower = ('l', 'o', 'w', 'e', 'r', '</w>')
    assert expected_low in initial_vocab, "('l','o','w','</w>') should be in vocab"
    assert initial_vocab[expected_low] == 5, "'low' should appear 5 times"
    print(f"\n  PASS: 'low' correctly has frequency 5")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure to build the char_tuple correctly with </w> at end.")

print()


# ============================================================
#  EXERCISE 2
#  Topic: Count adjacent pair frequencies
#
#  Background:
#    Given the vocabulary {char_tuple: frequency}, count how often
#    each adjacent pair of tokens co-occurs (weighted by word frequency).
#
#    For word ('l','o','w','</w>') with frequency 5:
#      pairs: ('l','o')+=5, ('o','w')+=5, ('w','</w>')+=5
#
#    The pair with the highest total count is merged next.
#
#  C# Analogy:
#    var pairCounts = vocab
#      .SelectMany(entry =>
#        Enumerable.Range(0, entry.Key.Length - 1)
#          .Select(i => (pair: (entry.Key[i], entry.Key[i+1]), freq: entry.Value)))
#      .GroupBy(x => x.pair)
#      .ToDictionary(g => g.Key, g => g.Sum(x => x.freq));
# ============================================================

def count_pair_frequencies(vocab):
    """
    Count how often each adjacent pair of tokens appears in the vocabulary.
    Weight counts by word frequency (a word appearing 5x contributes 5 to each pair).

    Args:
        vocab (dict): {token_tuple: word_frequency}

    Returns:
        dict: {(token_a, token_b): total_frequency}
    """
    pairs = defaultdict(int)            # pair -> total count

    for word_tuple, freq in vocab.items():    # loop: (token tuple, word frequency)
        symbols = list(word_tuple)            # convert tuple to list for indexing

        # TODO: Loop over consecutive pairs (symbols[i], symbols[i+1])
        # Add freq to the count for each pair
        # Hint: range(len(symbols) - 1) gives indices 0, 1, ..., n-2
        # YOUR CODE HERE
        pass                                  # remove this

    return dict(pairs)


# Test Exercise 2
print("EXERCISE 2: Count Adjacent Pair Frequencies")
print("-" * 40)
try:
    initial_vocab = build_initial_vocab([
        "low low low low low",
        "lower lower",
        "newest newest newest",
        "widest widest",
    ])

    pairs = count_pair_frequencies(initial_vocab)

    # Sort by frequency, descending
    sorted_pairs = sorted(pairs.items(), key=lambda x: -x[1])

    print(f"  Top 10 pairs by frequency:")
    for (a, b), freq in sorted_pairs[:10]:
        print(f"    ('{a}', '{b}') -> {freq}")

    # Verify the most frequent pair
    best_pair = sorted_pairs[0][0]
    best_freq = sorted_pairs[0][1]
    print(f"\n  Most frequent pair: {best_pair} (freq={best_freq})")
    print(f"  This pair will be merged in Step 1 of BPE training.")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure count_pair_frequencies sums freq for each pair correctly.")

print()


# ============================================================
#  EXERCISE 3
#  Topic: Perform one BPE merge step
#
#  Background:
#    Merging (A, B) means: everywhere A is immediately followed by B,
#    replace both with the single token AB.
#
#    Before merge of ('e','s'):
#      ('n','e','w','e','s','t','</w>') freq=3
#      ('w','i','d','e','s','t','</w>') freq=2
#
#    After merge of ('e','s') -> 'es':
#      ('n','e','w','es','t','</w>') freq=3
#      ('w','i','d','es','t','</w>') freq=2
#
#  C# Analogy:
#    // Replace all occurrences of (A, B) pair with merged token AB
#    var newVocab = vocab.ToDictionary(
#      entry => MergePairInSequence(entry.Key, pairA, pairB, merged),
#      entry => entry.Value
#    );
# ============================================================

def apply_bpe_merge(vocab, pair_to_merge):
    """
    Apply one BPE merge: replace all occurrences of pair_to_merge with merged token.

    Scan each word tuple from left to right.
    When we find pair_to_merge[0] followed by pair_to_merge[1], replace with merged.

    Args:
        vocab (dict): Current {token_tuple: frequency}
        pair_to_merge (tuple): (token_a, token_b) to merge

    Returns:
        tuple: (new_vocab, merged_token_string)
    """
    pair_a, pair_b = pair_to_merge          # unpack the pair
    merged = pair_a + pair_b                # concatenate: "e" + "s" = "es"
    new_vocab = {}                          # build new vocabulary

    for word_tuple, freq in vocab.items():
        symbols = list(word_tuple)          # mutable copy
        new_symbols = []                    # result after merging
        i = 0

        while i < len(symbols):
            # TODO: Check if symbols[i] and symbols[i+1] form the pair
            # If yes: append merged, advance by 2
            # If no: append symbols[i], advance by 1
            # Handle the case where i is the last symbol (no i+1 exists)!
            # YOUR CODE HERE
            pass                            # remove this

        new_vocab[tuple(new_symbols)] = freq

    return new_vocab, merged


# Test Exercise 3
print("EXERCISE 3: Apply One BPE Merge Step")
print("-" * 40)
try:
    initial_vocab = build_initial_vocab([
        "low low low low low",
        "lower lower",
        "newest newest newest",
        "widest widest",
    ])

    # Count pairs to find the best merge
    pairs = count_pair_frequencies(initial_vocab)
    best_pair = max(pairs, key=pairs.get)   # find most frequent pair

    print(f"  Best pair to merge: {best_pair}")
    print()
    print("  Before merge:")
    for wt, freq in sorted(initial_vocab.items(), key=lambda x: -x[1]):
        print(f"    {' '.join(wt):<35} freq={freq}")

    # Apply merge
    new_vocab, merged_token = apply_bpe_merge(initial_vocab, best_pair)

    print()
    print(f"  After merging {best_pair} -> '{merged_token}':")
    for wt, freq in sorted(new_vocab.items(), key=lambda x: -x[1]):
        print(f"    {' '.join(wt):<35} freq={freq}")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure the merge scan loop correctly handles the last position.")

print()


# ============================================================
#  EXERCISE 4
#  Topic: Run the full BPE training loop
#
#  Background:
#    BPE training repeats:
#    1. Count pair frequencies
#    2. Find most frequent pair
#    3. Merge it
#    4. Record the merge rule
#    ...until the vocabulary reaches target size.
#
#    The "vocabulary" here means the token set (all unique tokens that appear
#    across all word representations), not the word frequency dict.
#
#  C# Analogy:
#    while (tokenSet.Count < vocabSize) {
#        var best = PairWithMaxFrequency(vocab);
#        vocab = ApplyMerge(vocab, best);
#        mergeRules.Add(best);
#        tokenSet.Add(best.merged);
#    }
# ============================================================

def run_bpe_training(corpus, num_merges):
    """
    Run the full BPE training algorithm for the specified number of merges.

    Args:
        corpus (list[str]): Training text
        num_merges (int): Number of merge steps to perform

    Returns:
        tuple: (final_vocab, merge_rules)
               final_vocab: {token_tuple: frequency}
               merge_rules: list of ((pair_a, pair_b), merged_token) in order
    """
    # Step 1: Initialize character-level vocabulary
    vocab = build_initial_vocab(corpus)
    merge_rules = []                        # ordered list of merge rules

    print(f"  Starting BPE training: {num_merges} merges on {len(vocab)} unique words")

    for step in range(1, num_merges + 1):
        # TODO 1: Count pair frequencies
        pairs = None                        # replace None
        # YOUR CODE HERE
        pass                                # remove this

        # TODO 2: Check if there are any pairs left
        if not pairs:
            print(f"  Stopping at step {step}: no more pairs.")
            break

        # TODO 3: Find the best pair (highest frequency)
        best_pair = None                    # replace None
        # Hint: max(pairs, key=pairs.get)
        # YOUR CODE HERE
        pass                                # remove this

        best_freq = pairs[best_pair]

        # TODO 4: Apply the merge
        vocab, merged = apply_bpe_merge(vocab, best_pair)
        # YOUR CODE HERE - apply_bpe_merge already written above!
        pass                                # remove this (after filling in above)

        # Record the merge rule
        merge_rules.append((best_pair, merged))

        print(f"  Step {step:2d}: merge {best_pair[0]!r:8s}+{best_pair[1]!r:8s}"
              f" -> {merged!r:<12s} (freq={best_freq})")

    return vocab, merge_rules


# Test Exercise 4
print("EXERCISE 4: Full BPE Training Loop")
print("-" * 40)
try:
    corpus = [
        "low low low low low",
        "lower lower",
        "newest newest newest",
        "widest widest",
    ]

    print("  Running BPE training for 10 steps...")
    final_vocab, merge_rules = run_bpe_training(corpus, num_merges=10)

    print()
    print(f"  Learned {len(merge_rules)} merge rules:")
    for i, ((a, b), merged) in enumerate(merge_rules, 1):
        print(f"    Rule {i:2d}: ('{a}', '{b}') -> '{merged}'")

    print()
    print(f"  Final vocabulary state ({len(final_vocab)} word entries):")
    for wt, freq in sorted(final_vocab.items(), key=lambda x: -x[1]):
        print(f"    {' '.join(wt)}")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Check that run_bpe_training calls count_pair_frequencies and apply_bpe_merge.")

print()


# ============================================================
#  EXERCISE 5
#  Topic: Apply BPE to encode new text
#
#  Background:
#    Once trained, BPE encoding applies merge rules in order to new text.
#    Start with character-level representation, apply rule 1, then rule 2, etc.
#
#    If a word was seen during training, it will typically reduce to 1-2 tokens.
#    If a word was NOT seen, it will be split into whatever pieces match.
#
#  C# Analogy:
#    // Apply substitution rules in order
#    foreach (var rule in mergeRules) {
#        symbols = ApplyRule(symbols, rule.PairA, rule.PairB, rule.Merged);
#    }
# ============================================================

def bpe_encode(text, merge_rules):
    """
    Encode text using trained BPE merge rules.

    For each word:
    1. Start with characters + </w>
    2. Apply each merge rule in order
    3. Collect final tokens

    Args:
        text (str): Input text to tokenize
        merge_rules (list): Ordered list of ((pair_a, pair_b), merged) from training

    Returns:
        list[str]: Final token strings (before ID lookup)
    """
    all_tokens = []

    for word in text.lower().split():       # process each word
        # Start with character-level representation
        syms = list(word) + ["</w>"]       # e.g., "low" -> ['l','o','w','</w>']

        # TODO: Apply each merge rule in order
        # For each rule ((pair_a, pair_b), merged):
        #   scan syms left to right
        #   wherever syms[i]==pair_a and syms[i+1]==pair_b: replace with merged, skip 2
        #   else: keep syms[i], advance 1
        for (pair_a, pair_b), merged in merge_rules:
            # YOUR CODE HERE
            pass                            # remove this and add the scan loop

        all_tokens.extend(syms)             # add this word's tokens to result

    return all_tokens


# Test Exercise 5
print("EXERCISE 5: Encoding New Text with Trained BPE")
print("-" * 40)
try:
    # First train the model
    corpus = [
        "low low low low low",
        "lower lower",
        "newest newest newest",
        "widest widest",
    ]
    _, merge_rules_for_encode = run_bpe_training(corpus, num_merges=8)

    print()
    print("  Encoding test words using trained merge rules:")
    test_words = ["low", "lower", "newest", "widest", "newest"]

    for word in test_words:
        tokens = bpe_encode(word, merge_rules_for_encode)
        print(f"    '{word}' -> {tokens}")

    print()
    print("  Encoding out-of-vocabulary word:")
    unk_word = "slowest"
    unk_tokens = bpe_encode(unk_word, merge_rules_for_encode)
    print(f"    '{unk_word}' -> {unk_tokens}")
    print(f"    (splits into known pieces even though 'slowest' was not in training!)")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Check that bpe_encode applies the merge rules correctly in order.")


print()
print("=" * 60)
print("Exercise 02 Complete (or attempted)!")
print()
print("Expected outputs (approximate):")
print("  Ex1: vocab has 4 word entries, 'low' has freq=5")
print("  Ex2: most frequent pair should be ('e','s') or ('s','t') with freq=5")
print("  Ex3: ('e','s') merged to 'es', word tuples now contain 'es'")
print("  Ex4: 10 merge rules learned, showing BPE building up 'es','est','wi','wid'...")
print("  Ex5: 'low' -> ['low</w>'] or ['low', '</w>'], 'lower' splits correctly")
print("=" * 60)

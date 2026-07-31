"""
Module 05.5 - HuggingFace Tokenizers
Example 01: Tokenization Basics

GLOSSARY
--------
Token        : A piece of text that the model treats as one unit.
               Can be a full word ("cat"), part of a word ("##ning"), or punctuation.
               C# analogy: like an enum value representing a text chunk.

Vocabulary   : The complete set of all known tokens. A lookup table: token -> ID.
               GPT-2 vocabulary has 50,257 entries.
               C# analogy: Dictionary<string, int>

Token ID     : The integer index of a token in the vocabulary.
               Model input is always a list of token IDs (integers), not strings.
               C# analogy: the int key in Dictionary<string, int>

Special Token: Reserved tokens with special meaning: [PAD]=padding, [UNK]=unknown.
               C# analogy: reserved keywords in C# (class, void, return)

Subword      : A piece of a word. "unhappy" = "un" + "happy".
               Subword tokenization balances vocabulary size vs sequence length.

This example has NO library dependencies - runs with pure Python only.
"""

import re          # re: regular expressions, like System.Text.RegularExpressions in C#
import string      # string: constants like string.punctuation (!"#$%&'...)

print("=" * 60)    # print a separator line of 60 "=" characters
print("Example 01: Tokenization Basics")
print("=" * 60)
print()            # blank line for readability


# ============================================================
# PART A: Why Word-Level Tokenization Has Problems
# ============================================================

print("PART A: Word-Level Tokenization - Showing the Problems")
print("-" * 50)
print()

# Build a tiny word-level vocabulary from a sample corpus
# corpus: the text data we learn from (like a training dataset)
corpus = [                              # corpus = list of sentences
    "I love cats",
    "I love dogs",
    "Cats and dogs are pets",
    "Running cats are fast",
    "A runner runs fast",
]

# Tokenize each sentence by splitting on whitespace (simplest possible method)
# This is the "word-level" approach
all_words = []                          # start with empty list
for sentence in corpus:                 # loop over each sentence in corpus
    words = sentence.lower().split()    # lowercase, then split on spaces
    all_words.extend(words)             # add all words to our list
# all_words now contains every word from every sentence (with duplicates)

# Build vocabulary: unique words, sorted alphabetically, each gets an ID
unique_words = sorted(set(all_words))   # set() removes duplicates, sorted() alphabetizes
vocab = {}                              # empty dictionary to hold vocab (token -> ID)
for idx, word in enumerate(unique_words):  # enumerate gives (index, value)
    vocab[word] = idx                   # map each word to its index

print("Word-Level Vocabulary:")
for token, token_id in vocab.items():  # .items() gives (key, value) pairs
    print(f"  '{token}' -> {token_id}")

print()
print("Problems with word-level tokenization:")
print()

# Problem 1: "cats" and "cat" are different tokens (model can't share info)
# cats_id = vocab.get("cats")  -> gets ID for "cats"
# cat_id  = vocab.get("cat")   -> ERROR: "cat" is not in vocab!
cats_in_vocab = "cats" in vocab        # check if "cats" is in vocabulary
cat_in_vocab  = "cat"  in vocab        # check if "cat" is in vocabulary
print(f"  Problem 1 - Related words are separate tokens:")
print(f"  'cats' in vocab: {cats_in_vocab}")   # True
print(f"  'cat'  in vocab: {cat_in_vocab}")    # False! "cat" was never seen

# Problem 2: "running" and "runner" and "runs" are all different tokens
running_in_vocab = "running" in vocab  # True (seen in corpus)
runner_in_vocab  = "runner"  in vocab  # True (seen in corpus)
runs_in_vocab    = "runs"    in vocab  # True (seen in corpus)
print()
print(f"  Problem 2 - Morphology ignored (all treated as unrelated):")
print(f"  'running' -> ID {vocab.get('running', 'NOT FOUND')}")
print(f"  'runner'  -> ID {vocab.get('runner', 'NOT FOUND')}")
print(f"  'runs'    -> ID {vocab.get('runs', 'NOT FOUND')}")
print(f"  These are 3 separate unrelated entries! Model must learn each independently.")

# Problem 3: new words not in vocab become [UNK]
UNK_TOKEN = "[UNK]"                    # [UNK] = unknown token sentinel
new_words = ["skateboarding", "neural", "transformer"]  # words not seen during training
print()
print(f"  Problem 3 - Unknown words become [UNK]:")
for word in new_words:
    if word in vocab:                  # if word is in vocabulary
        token_id = vocab[word]         # look up its ID
    else:                              # if word is NOT in vocabulary
        token_id = -1                  # -1 represents [UNK] (would be a real ID)
        print(f"  '{word}' -> [UNK] (lost forever! model sees nothing useful)")

print()


# ============================================================
# PART B: Character-Level Tokenization - Another Approach
# ============================================================

print("PART B: Character-Level Tokenization - Too Granular")
print("-" * 50)
print()

def char_tokenize(text):
    """
    Tokenize text into individual characters.

    C# analogy:
      text.ToCharArray().Select(c => c.ToString()).ToList()

    Args:
        text (str): Input text to tokenize

    Returns:
        list[str]: List of single-character strings
    """
    return list(text)               # list("abc") = ['a', 'b', 'c'] in Python
                                    # list() on a string gives one element per character


def build_char_vocab(texts):
    """
    Build a character-level vocabulary from a list of texts.

    Returns a dictionary mapping each character to a unique integer ID.
    Special tokens are added at the beginning (IDs 0, 1, 2).

    Args:
        texts (list[str]): list of text strings to learn vocab from

    Returns:
        dict: {char: int} vocabulary
    """
    all_chars = set()               # set: like HashSet<char> in C#, no duplicates
    for text in texts:              # loop over each text
        for char in text:           # loop over each character in the text
            all_chars.add(char)     # add character to the set

    # Create vocab with special tokens first
    vocab = {
        "[PAD]": 0,                 # [PAD] gets ID 0 (padding token)
        "[UNK]": 1,                 # [UNK] gets ID 1 (unknown character)
        "[CLS]": 2,                 # [CLS] gets ID 2 (start of sequence)
    }

    # Add all characters found in the texts
    for idx, char in enumerate(sorted(all_chars)):  # sorted for reproducibility
        vocab[char] = idx + 3       # +3 because IDs 0,1,2 are taken by special tokens

    return vocab                    # return the completed vocabulary dictionary


# Test character tokenization
sample_texts = ["cat", "cats", "bat", "running"]
char_vocab = build_char_vocab(sample_texts)  # build vocab from all characters

print("Character Vocabulary (subset):")
# Show first 15 entries for readability
for token, token_id in list(char_vocab.items())[:15]:  # [:15] = first 15 items
    print(f"  '{token}' -> {token_id}")

print()

# Tokenize "cat" and "cats" character by character
word1 = "cat"
word2 = "cats"

tokens1 = char_tokenize(word1)         # ['c', 'a', 't']
tokens2 = char_tokenize(word2)         # ['c', 'a', 't', 's']

ids1 = [char_vocab.get(c, char_vocab["[UNK]"]) for c in tokens1]  # look up each char
ids2 = [char_vocab.get(c, char_vocab["[UNK]"]) for c in tokens2]  # look up each char
# .get(key, default) = returns char_vocab["[UNK]"] if char not found

print(f"Char tokenize '{word1}': {tokens1} -> IDs: {ids1}")
print(f"Char tokenize '{word2}': {tokens2} -> IDs: {ids2}")

# Demonstrate shared prefix (cat and cats share 'c','a','t')
print()
print("Good news: 'cat' and 'cats' share the tokens c,a,t!")
print("Bad news: long text creates very long sequences:")

long_text = "unhappiness is a state of mind"
char_tokens = char_tokenize(long_text)
print(f"  Text: '{long_text}'")
print(f"  Character tokens: {len(char_tokens)} tokens")
print(f"  Word tokens would be: {len(long_text.split())} tokens")
print(f"  Character level is {len(char_tokens)//len(long_text.split())}x longer!")

print()


# ============================================================
# PART C: Subword Tokenization - The Best of Both Worlds
# ============================================================

print("PART C: Subword Tokenization - Conceptual Demo")
print("-" * 50)
print()

# This is a simplified manual demonstration of the subword concept.
# Real subword tokenizers use BPE (covered in Example 02).

# Hand-crafted mini subword vocabulary (like what BPE would learn)
# This shows the CONCEPT: common words whole, rare words split
subword_vocab = {
    # Special tokens (always first)
    "[PAD]" : 0,
    "[UNK]" : 1,
    "[CLS]" : 2,
    "[SEP]" : 3,

    # Common complete words (appear frequently)
    "the"   : 4,
    "is"    : 5,
    "a"     : 6,
    "of"    : 7,

    # Common word stems (appear in many forms)
    "run"   : 8,
    "cat"   : 9,
    "happi" : 10,    # "happy" with BPE spelling convention

    # Common suffixes and prefixes (subwords!)
    "un"    : 11,    # prefix: "un-happy", "un-known"
    "ness"  : 12,    # suffix: "happi-ness", "dark-ness"
    "ing"   : 13,    # suffix: "run-ning", "walk-ing"
    "s"     : 14,    # suffix: "cat-s", "run-s"
    "er"    : 15,    # suffix: "run-ner" -> "run"+"er"
    "re"    : 16,    # prefix: "re-run"
    "ly"    : 17,    # suffix: "quick-ly"

    # Common endings
    "."     : 18,
    ","     : 19,
    "!"     : 20,
}

# Reverse vocab for decoding (ID -> token string)
# In C#: var reverseVocab = vocab.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
id_to_token = {v: k for k, v in subword_vocab.items()}  # dict comprehension (swaps k and v)


def simple_subword_encode(text, vocab):
    """
    Very simplified subword tokenizer.
    Splits text into tokens by checking longest match in vocabulary.

    This is a greedy approach: try to match the longest possible token first.
    Real BPE applies merge rules; this demonstrates the same end result.

    Args:
        text (str): Input text to tokenize
        vocab (dict): {token_string: token_id} vocabulary

    Returns:
        tuple: (list of token strings, list of token IDs)
    """
    text_lower = text.lower()       # lowercase for simpler matching
    tokens = []                     # list to collect token strings
    i = 0                           # current position in the string

    while i < len(text_lower):      # while there are still characters to process
        matched = False             # track if we found a match at position i

        # Try to match the longest token possible (greedy approach)
        # Try length 6, then 5, then 4, ... down to 1
        for length in range(6, 0, -1):      # range(6, 0, -1) = [6, 5, 4, 3, 2, 1]
            candidate = text_lower[i:i+length]  # slice of text starting at i with given length
            if candidate in vocab:              # check if this slice is a known token
                tokens.append(candidate)        # add the matched token
                i += length                     # advance position by matched length
                matched = True                  # we found a match
                break                           # stop trying shorter lengths

        if not matched:             # if no token matched (character not in vocab)
            tokens.append("[UNK]")  # add [UNK] token for the unknown character
            i += 1                  # advance by 1 character

    # Convert token strings to IDs using vocabulary lookup
    ids = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
    # vocab.get(key, default): like dict.GetValueOrDefault() in C#
    # If token not found, use [UNK] ID

    return tokens, ids              # return both strings and IDs


# Test subword tokenization on several words
test_words = ["run", "running", "runner", "unhappiness", "cats", "cat"]

print("Subword Tokenization Results:")
print()
for word in test_words:
    tok_strings, tok_ids = simple_subword_encode(word, subword_vocab)
    print(f"  '{word}' -> tokens: {tok_strings} -> IDs: {tok_ids}")
    # Shows how common words stay whole, rare words split into known pieces

print()

# Demonstrate shared token advantage
print("Shared token advantage:")
print("  'run'    and 'running' and 'runner' all contain token 'run' (ID=8)")
print("  The model learns ONE vector for 'run' and reuses it across all forms!")
print("  In word-level: model would need 3 separate vectors, no shared info.")

print()


# ============================================================
# PART D: Special Tokens in Practice
# ============================================================

print("PART D: Special Tokens")
print("-" * 50)
print()

# Demonstrate how special tokens are added to sequences
# Different model families use different conventions

def bert_encode(text, vocab):
    """
    Simulate BERT-style encoding with [CLS] and [SEP] tokens.
    BERT always wraps input with [CLS] at start and [SEP] at end.

    C# analogy: like adding XML root tags around content
      "<CLS>" + content + "</SEP>"

    Args:
        text (str): Input text
        vocab (dict): Vocabulary with [CLS] and [SEP] defined

    Returns:
        list[int]: Token IDs including special tokens
    """
    tokens, ids = simple_subword_encode(text, vocab)  # first encode the text

    cls_id = vocab["[CLS]"]             # get the [CLS] token ID
    sep_id = vocab["[SEP]"]             # get the [SEP] token ID

    # BERT format: [CLS] + text_tokens + [SEP]
    full_ids = [cls_id] + ids + [sep_id]    # + concatenates lists in Python
    full_tokens = ["[CLS]"] + tokens + ["[SEP]"]

    return full_tokens, full_ids


# Demonstrate BERT encoding
text1 = "the cat is running"
bert_tokens, bert_ids = bert_encode(text1, subword_vocab)

print("BERT-style encoding:")
print(f"  Text: '{text1}'")
print(f"  Tokens: {bert_tokens}")
print(f"  IDs:    {bert_ids}")
print()

# Show what happens with padding (when batching multiple sequences)
text2 = "run"                           # shorter sequence
bert_tokens2, bert_ids2 = bert_encode(text2, subword_vocab)

pad_id = subword_vocab["[PAD]"]         # get the [PAD] token ID (= 0)
max_len = max(len(bert_ids), len(bert_ids2))  # longest sequence length

# Pad the shorter sequence to match the longer one
ids1_padded = bert_ids + [pad_id] * (max_len - len(bert_ids))   # add PAD tokens
ids2_padded = bert_ids2 + [pad_id] * (max_len - len(bert_ids2)) # add PAD tokens
# [pad_id] * N creates a list of N copies of pad_id

# Attention mask: 1 for real tokens, 0 for padding
mask1 = [1] * len(bert_ids) + [0] * (max_len - len(bert_ids))
mask2 = [1] * len(bert_ids2) + [0] * (max_len - len(bert_ids2))

print("Padding two sequences to the same length (for batch processing):")
print()
print(f"  Seq 1 '{text1}': {ids1_padded}")
print(f"  Mask 1:          {mask1}   (1=real, 0=padding)")
print()
print(f"  Seq 2 '{text2}': {ids2_padded}")
print(f"  Mask 2:          {mask2}   (1=real, 0=padding)")
print()
print("  The model reads mask to ignore PAD positions during attention!")

print()


# ============================================================
# PART E: ASCII Visualization of the Full Pipeline
# ============================================================

print("PART E: Full Tokenization Pipeline (ASCII Diagram)")
print("-" * 50)
print()

def visualize_pipeline(text, vocab):
    """
    Print a visual diagram of the tokenization pipeline.

    Shows: Raw Text -> Pre-tokenize -> Subword Split -> Lookup -> IDs

    Args:
        text (str): Input text to visualize
        vocab (dict): Vocabulary to use
    """
    print(f"  Input text: \"{text}\"")
    print()
    print("  Step 1 - Pre-tokenize (split on spaces):")
    words = text.lower().split()            # split on whitespace
    print(f"    {words}")
    print()
    print("  Step 2 - Subword tokenize each word:")
    all_tokens = []
    for word in words:
        toks, _ = simple_subword_encode(word, vocab)  # tokenize each word
        print(f"    '{word}' -> {toks}")
        all_tokens.extend(toks)             # add to flat list
    print()
    print("  Step 3 - Add special tokens (BERT style):")
    full_tokens = ["[CLS]"] + all_tokens + ["[SEP]"]
    print(f"    {full_tokens}")
    print()
    print("  Step 4 - Convert to IDs:")
    full_ids = [vocab.get(t, vocab["[UNK]"]) for t in full_tokens]
    print(f"    {full_ids}")
    print()
    print("  Step 5 - This integer list is what the model receives!")
    print()


visualize_pipeline("unhappiness is running", subword_vocab)

print()
print("=" * 60)
print("Example 01 Complete!")
print()
print("Key Takeaways:")
print("  1. Word-level: big vocab, no morphology, unknown words -> [UNK]")
print("  2. Char-level: tiny vocab, but sequences too long")
print("  3. Subword: balance - common words whole, rare words split")
print("  4. Special tokens: [CLS], [SEP], [PAD] have fixed IDs in vocab")
print("  5. The model ONLY receives integers (token IDs), never raw text")
print("=" * 60)

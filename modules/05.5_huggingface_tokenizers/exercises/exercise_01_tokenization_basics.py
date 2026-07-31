"""
Module 05.5 - HuggingFace Tokenizers
Exercise 01: Tokenization Basics

GLOSSARY
--------
Token        : A piece of text the model processes as one unit.
               C# analogy: an enum value representing a text chunk.

Vocabulary   : Dictionary<string, int> mapping each token to its unique integer ID.
               GPT-2 has 50,257 entries.

Token ID     : The integer index of a token in the vocabulary.
               Models receive lists of integers, never raw text.

Encoding     : Converting text to a list of token IDs.
               C# analogy: JsonSerializer.Serialize()

Decoding     : Converting a list of token IDs back to text.
               C# analogy: JsonSerializer.Deserialize()

Special Token: Reserved token with a specific role: [PAD], [UNK], [CLS], [SEP].
               C# analogy: reserved keyword in C# (class, void, return).

Attention Mask: A list of 1s and 0s. 1 = real token, 0 = padding.
                Tells the model which positions to attend to.
                C# analogy: bool[] mask = tokens.Select(t => t != PAD).ToArray()

Try to complete each exercise before looking at the expected output!
"""

print("=" * 60)
print("Exercise 01: Tokenization Basics")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Build a character-level vocabulary
#
#  Background:
#    Character tokenizers split text into individual characters.
#    Each unique character gets an integer ID.
#    Special tokens are always added FIRST (IDs 0, 1, 2...).
#
#    Example:
#      text = "hello"
#      chars = sorted(set("hello")) = ['e', 'h', 'l', 'o']
#      vocab = {"[PAD]":0, "[UNK]":1, "e":2, "h":3, "l":4, "o":5}
#      encode("hello") = [3, 2, 4, 4, 5]
#
#  Your Task:
#    Complete build_char_vocab(texts) and char_encode(text, vocab).
#
#  C# Analogy:
#    var allChars = string.Concat(texts).Distinct().OrderBy(c => c).ToList();
#    var vocab = new Dictionary<string, int> {{"[PAD]", 0}, {"[UNK]", 1}};
#    for (int i = 0; i < allChars.Count; i++) vocab[allChars[i].ToString()] = i + 2;
# ============================================================

def build_char_vocab(texts):
    """
    Build a character-level vocabulary from a list of text strings.

    The vocabulary maps each unique character to a unique integer ID.
    Special tokens are added first: [PAD]=0, [UNK]=1.
    Then characters are added in sorted order starting from ID 2.

    Args:
        texts (list[str]): list of strings to collect characters from

    Returns:
        dict: {char: int} e.g., {"[PAD]":0, "[UNK]":1, "a":2, "b":3, ...}
    """
    # TODO 1: Collect all unique characters from all texts
    # Hint: use a set to collect unique chars
    # Hint: loop over texts, then over characters in each text
    all_chars = set()                                   # collect unique chars here
    # YOUR CODE HERE
    pass                                                # remove this line

    # TODO 2: Create vocab starting with special tokens
    vocab = {"[PAD]": 0, "[UNK]": 1}                  # special tokens first

    # TODO 3: Add sorted characters starting from ID 2
    # Hint: sorted(all_chars) gives alphabetical order
    # Hint: enumerate(sorted_chars) gives (index, char) pairs
    # YOUR CODE HERE
    pass                                                # remove this line

    return vocab


def char_encode(text, vocab):
    """
    Encode text as a list of character-level token IDs.
    Characters not in vocab are mapped to [UNK] (ID 1).

    Args:
        text (str): Input text to encode
        vocab (dict): Character vocabulary {char: int}

    Returns:
        list[int]: List of character IDs
    """
    # TODO 4: Convert each character to its ID using vocab.get()
    # Hint: vocab.get(char, vocab["[UNK]"]) returns [UNK] ID if char not found
    # YOUR CODE HERE
    pass                                                # remove this


def char_decode(ids, vocab):
    """
    Decode a list of character IDs back to a text string.

    Args:
        ids (list[int]): Token IDs to decode
        vocab (dict): Character vocabulary {char: int}

    Returns:
        str: Reconstructed text
    """
    # TODO 5: Build reverse vocabulary (ID -> char) and join characters
    # Hint: {v: k for k, v in vocab.items()} swaps keys and values
    # Hint: "".join(list_of_chars) joins chars without spaces
    # YOUR CODE HERE
    pass                                                # remove this


# Test Exercise 1
print("EXERCISE 1: Character-Level Vocabulary")
print("-" * 40)
try:
    sample_texts = ["hello", "world", "cat"]
    char_vocab = build_char_vocab(sample_texts)

    print(f"  Vocabulary ({len(char_vocab)} tokens):")
    for token, token_id in sorted(char_vocab.items(), key=lambda x: x[1]):
        print(f"    '{token}' -> {token_id}")

    encoded = char_encode("hello", char_vocab)
    decoded = char_decode(encoded, char_vocab)
    print(f"\n  encode('hello'): {encoded}")
    print(f"  decode({encoded}): '{decoded}'")
    print(f"  Round-trip correct: {decoded == 'hello'}")

    # Test unknown character handling
    encoded_unk = char_encode("xyz!", char_vocab)
    print(f"  encode('xyz!'): {encoded_unk}")
    print(f"  (! is not in vocab, should be [UNK] ID=1)")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure to remove 'pass' and fill in the TODO sections.")

print()


# ============================================================
#  EXERCISE 2
#  Topic: Build a word-level vocabulary and tokenizer
#
#  Background:
#    Word-level tokenization splits on whitespace.
#    Each unique word becomes one token.
#    Unknown words (not seen during vocab building) map to [UNK].
#
#    C# Analogy:
#      var words = corpus.SelectMany(s => s.Split()).Distinct().ToList();
#      var vocab = words.Select((w, i) => (w, i+2))
#                       .ToDictionary(x => x.w, x => x.Item2);
#      // +2 because [PAD]=0 and [UNK]=1
# ============================================================

def build_word_vocab(corpus):
    """
    Build a word-level vocabulary from a list of sentences.

    Pre-process: lowercase all text before building vocabulary.
    Add [PAD]=0 and [UNK]=1 as special tokens.
    Assign IDs to words in sorted order starting from 2.

    Args:
        corpus (list[str]): List of training sentences

    Returns:
        dict: {word: int} vocabulary
    """
    # TODO 1: Collect all unique words from corpus
    # Hint: sentence.lower().split() gives lowercase words
    # Hint: use a set to collect unique words
    all_words = set()
    # YOUR CODE HERE
    pass                                                # remove this

    # TODO 2: Build vocab with special tokens first
    vocab = {"[PAD]": 0, "[UNK]": 1}
    # YOUR CODE HERE - add sorted words starting from ID 2
    pass                                                # remove this

    return vocab


def word_encode(text, vocab):
    """
    Encode text as word-level token IDs.
    Pre-process: lowercase the text.
    Unknown words map to [UNK].

    Args:
        text (str): Input text
        vocab (dict): Word vocabulary

    Returns:
        list[int]: Token IDs
    """
    words = text.lower().split()                        # lowercase + split
    # TODO: map each word to its ID (use [UNK] for unknown words)
    # YOUR CODE HERE
    pass                                                # remove this


# Test Exercise 2
print("EXERCISE 2: Word-Level Vocabulary")
print("-" * 40)
try:
    corpus = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "cats and dogs are pets",
    ]

    word_vocab = build_word_vocab(corpus)
    print(f"  Vocabulary size: {len(word_vocab)} tokens")

    test_text = "the cat chased a mouse"
    encoded = word_encode(test_text, word_vocab)
    print(f"  Text: '{test_text}'")
    print(f"  Encoded: {encoded}")
    print(f"  Note: 'chased' and 'mouse' are [UNK] (not in training corpus)")
    print(f"  [UNK] ID is 1, so you should see some 1s in the output")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure to remove 'pass' and fill in the TODO sections.")

print()


# ============================================================
#  EXERCISE 3
#  Topic: Add special tokens and create attention masks
#
#  Background:
#    BERT-style models wrap every input with [CLS] at start and [SEP] at end.
#    When batching multiple sequences, they must all be the same length.
#    Shorter sequences are padded with [PAD] tokens.
#    The attention mask tells the model which positions are real (1) vs padding (0).
#
#    C# Analogy:
#      var paddedIds = ids.Concat(Enumerable.Repeat(PAD_ID, padLen)).ToArray();
#      var mask = ids.Select(_ => 1).Concat(Enumerable.Repeat(0, padLen)).ToArray();
# ============================================================

def add_special_tokens(ids, cls_id, sep_id):
    """
    Wrap a list of token IDs with [CLS] at the start and [SEP] at the end.

    [CLS] and [SEP] are BERT's special tokens:
    - [CLS] (Classification token): Always first. The model uses this
      token's output for classification tasks.
    - [SEP] (Separator): Always last. Marks end of sequence (or segment).

    Args:
        ids (list[int]): Original token IDs
        cls_id (int): ID of [CLS] token
        sep_id (int): ID of [SEP] token

    Returns:
        list[int]: [cls_id] + ids + [sep_id]
    """
    # TODO: Prepend cls_id and append sep_id
    # Hint: [cls_id] + ids + [sep_id] concatenates three lists
    # YOUR CODE HERE
    pass                                                # remove this


def pad_and_mask(batch_ids, pad_id):
    """
    Pad all sequences in a batch to the same length.
    Return padded IDs and attention masks.

    The attention mask is 1 for real tokens and 0 for padding.
    This tells self-attention to ignore padded positions.

    Args:
        batch_ids (list[list[int]]): List of token ID sequences
        pad_id (int): ID to use for padding

    Returns:
        tuple: (padded_ids, attention_masks)
               Both are list of lists, all with the same length.
    """
    # TODO 1: Find the maximum sequence length in the batch
    # Hint: max(len(ids) for ids in batch_ids)
    max_len = None                                      # replace None
    # YOUR CODE HERE
    pass                                                # remove this

    padded_ids = []
    attention_masks = []

    for ids in batch_ids:
        pad_count = max_len - len(ids)                  # how many [PAD] to add

        # TODO 2: Create padded sequence (original ids + PAD tokens)
        padded = None                                   # replace None
        # YOUR CODE HERE - hint: ids + [pad_id] * pad_count
        pass                                            # remove this

        # TODO 3: Create attention mask (1 for real, 0 for padding)
        mask = None                                     # replace None
        # YOUR CODE HERE - hint: [1]*len(ids) + [0]*pad_count
        pass                                            # remove this

        padded_ids.append(padded)
        attention_masks.append(mask)

    return padded_ids, attention_masks


# Test Exercise 3
print("EXERCISE 3: Special Tokens and Attention Masks")
print("-" * 40)
try:
    # Use the vocab from Exercise 2 (or build a new one)
    sample_vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "hello": 4, "world": 5}
    cls_id = sample_vocab["[CLS]"]
    sep_id = sample_vocab["[SEP]"]
    pad_id = sample_vocab["[PAD]"]

    # Sequences of different lengths
    seq1 = [4, 5]               # ["hello", "world"] -> IDs [4, 5]
    seq2 = [4]                  # ["hello"] -> ID [4]
    seq3 = [5, 4, 5]            # ["world", "hello", "world"] -> [5, 4, 5]

    # Add special tokens
    seq1_special = add_special_tokens(seq1, cls_id, sep_id)
    seq2_special = add_special_tokens(seq2, cls_id, sep_id)
    seq3_special = add_special_tokens(seq3, cls_id, sep_id)

    print(f"  seq1 after special tokens: {seq1_special}")
    print(f"  seq2 after special tokens: {seq2_special}")
    print(f"  seq3 after special tokens: {seq3_special}")

    # Pad the batch
    batch = [seq1_special, seq2_special, seq3_special]
    padded, masks = pad_and_mask(batch, pad_id)

    print()
    print(f"  After padding (all same length={len(padded[0])}):")
    for i, (p, m) in enumerate(zip(padded, masks)):
        print(f"    Seq {i+1} IDs:  {p}")
        print(f"    Seq {i+1} mask: {m}  (1=real token, 0=padding)")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure to remove 'pass' and fill in the TODO sections.")

print()


# ============================================================
#  EXERCISE 4
#  Topic: Build a simple subword vocabulary manually
#
#  Background:
#    Subword tokenization splits rare/unknown words into known pieces.
#    The vocabulary contains both whole words and word fragments.
#    "unhappiness" -> ["un", "happi", "ness"] if those are in vocab.
#
#    C# Analogy:
#      // Try longest match first (greedy)
#      int i = 0;
#      while (i < text.Length) {
#          for (int len = maxTokenLen; len > 0; len--) {
#              if (vocab.ContainsKey(text.Substring(i, len))) { ... }
#          }
#      }
# ============================================================

SUBWORD_VOCAB = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
    "un": 4, "happi": 5, "ness": 6, "happy": 7,
    "run": 8, "ning": 9, "cat": 10, "s": 11,
    "the": 12, "is": 13, "a": 14,
    "ing": 15, "er": 16, "re": 17, "ly": 18,
    ".": 19, ",": 20,
}


def subword_encode(text, vocab):
    """
    Encode text using a subword vocabulary.
    Use greedy longest-match: try longest possible token first.
    Unknown characters map to [UNK].

    Example:
      text = "unhappiness"
      Try "unhappiness" (length 11) -> not in vocab
      Try "unhappines"  (length 10) -> not in vocab
      ...
      Try "un" (length 2) -> IN VOCAB! Add "un", continue from position 2.
      Try "happines" -> not in vocab
      ...
      Try "happi" (length 5) -> IN VOCAB! Add "happi", continue from position 7.
      ...
      Add "ness" -> done.
      Result: ["un", "happi", "ness"] -> IDs [4, 5, 6]

    Args:
        text (str): Input text to encode
        vocab (dict): Subword vocabulary {token: int}

    Returns:
        list[int]: Token IDs
    """
    text_lower = text.lower()               # lowercase for matching
    tokens = []                             # token IDs to collect
    i = 0                                   # current position in text

    while i < len(text_lower):
        # TODO: Greedy longest-match algorithm
        # Try decreasing lengths from len(text_lower)-i down to 1
        # If a match found: add its ID to tokens, advance i by match length
        # If no match: add [UNK] ID, advance i by 1

        matched = False
        max_len = min(10, len(text_lower) - i)          # max token length to try

        for length in range(max_len, 0, -1):            # try longest first
            candidate = text_lower[i:i+length]          # slice of text
            # TODO: check if candidate is in vocab, if so add its ID and break
            # YOUR CODE HERE
            pass                                        # remove this

        if not matched:
            tokens.append(vocab["[UNK]"])               # unknown character
            i += 1

    return tokens


# Test Exercise 4
print("EXERCISE 4: Subword Encoding (Greedy Longest-Match)")
print("-" * 40)
try:
    test_inputs = [
        "unhappiness",
        "running",
        "cats",
        "the cat is running",
    ]

    for text in test_inputs:
        ids = subword_encode(text, SUBWORD_VOCAB)
        # Decode back to strings for display
        id_to_token = {v: k for k, v in SUBWORD_VOCAB.items()}
        tokens = [id_to_token.get(id, "[?]") for id in ids]
        print(f"  '{text}' -> {tokens} -> IDs {ids}")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure to remove 'pass' and fill in the TODO sections.")

print()


# ============================================================
#  EXERCISE 5
#  Topic: Compute vocabulary coverage statistics
#
#  Background:
#    A good vocabulary should cover most tokens in the test corpus.
#    "Coverage" = percentage of tokens that are NOT [UNK].
#    Low coverage means the vocabulary is too small or domain mismatch.
#
#    C# Analogy:
#      double coverage = tokenIds.Count(id => id != UNK_ID) / (double)tokenIds.Count * 100;
# ============================================================

def compute_coverage(texts, encode_fn, vocab):
    """
    Compute vocabulary coverage: what fraction of tokens are known (not [UNK])?

    Coverage = (total_tokens - unk_tokens) / total_tokens * 100

    Args:
        texts (list[str]): List of text strings to test
        encode_fn (callable): Function that takes (text, vocab) and returns list[int]
        vocab (dict): The vocabulary being tested

    Returns:
        dict: {
            "total_tokens": int,
            "unk_tokens": int,
            "coverage_pct": float,
        }
    """
    total = 0                               # total token count
    unk_count = 0                           # [UNK] token count
    unk_id = vocab.get("[UNK]", 1)         # get the [UNK] ID

    for text in texts:
        ids = encode_fn(text, vocab)        # encode the text

        # TODO: Count total tokens and UNK tokens
        # YOUR CODE HERE
        pass                                # remove this

    # TODO: Compute coverage percentage
    # coverage_pct = (1 - unk_count/total) * 100 if total > 0 else 0
    coverage_pct = None                     # replace None
    # YOUR CODE HERE
    pass                                    # remove this

    return {
        "total_tokens" : total,
        "unk_tokens"   : unk_count,
        "coverage_pct" : coverage_pct,
    }


# Test Exercise 5
print("EXERCISE 5: Vocabulary Coverage Analysis")
print("-" * 40)
try:
    # Test texts (mix of in-vocabulary and out-of-vocabulary words)
    in_domain_texts = [
        "the cat is running",
        "unhappiness is real",
        "running cats are happy",
    ]

    out_of_domain_texts = [
        "artificial intelligence transforms industries",
        "neural networks optimize parameters",
        "gradient descent minimizes loss functions",
    ]

    in_stats  = compute_coverage(in_domain_texts,  subword_encode, SUBWORD_VOCAB)
    out_stats = compute_coverage(out_of_domain_texts, subword_encode, SUBWORD_VOCAB)

    print("  In-domain texts (related to our vocab):")
    print(f"    Total tokens: {in_stats['total_tokens']}")
    print(f"    [UNK] tokens: {in_stats['unk_tokens']}")
    print(f"    Coverage:     {in_stats['coverage_pct']:.1f}%")

    print()
    print("  Out-of-domain texts (ML terminology, not in our vocab):")
    print(f"    Total tokens: {out_stats['total_tokens']}")
    print(f"    [UNK] tokens: {out_stats['unk_tokens']}")
    print(f"    Coverage:     {out_stats['coverage_pct']:.1f}%")

    print()
    print("  Observation: lower coverage on out-of-domain text shows")
    print("  why domain-specific tokenizers matter!")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure to remove 'pass' and fill in the TODO sections.")


print()
print("=" * 60)
print("Exercise 01 Complete (or attempted)!")
print()
print("Expected outputs:")
print("  Ex1: vocab has ~15 chars, encode/decode round-trips correctly")
print("  Ex2: word vocab has ~14 words, 'chased'/'mouse' map to [UNK]=1")
print("  Ex3: sequences wrapped with [CLS]=2 and [SEP]=3, padded to length 5")
print("  Ex4: 'unhappiness' -> ['un','happi','ness'], IDs [4,5,6]")
print("  Ex5: in-domain coverage > 50%, out-domain coverage < 30%")
print("=" * 60)

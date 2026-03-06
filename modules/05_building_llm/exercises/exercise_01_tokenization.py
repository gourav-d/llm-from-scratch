"""
Exercise 1: Tokenization Practice

Complete these exercises to practice building tokenizers!
Solutions are provided at the bottom (commented out).
"""

import re
from collections import Counter

# ==============================================================================
# Exercise 1: Build a Character Tokenizer
# ==============================================================================

print("="*70)
print("Exercise 1: Build a Character Tokenizer")
print("="*70)

"""
Build a simple character-level tokenizer that:
1. Builds vocabulary from a list of texts
2. Encodes text to token IDs
3. Decodes token IDs back to text

Hint: Use a dictionary to map characters to IDs
"""

# TODO: Implement CharTokenizer class
class CharTokenizer:
    def __init__(self):
        # Initialize your tokenizer
        pass

    def build_vocab(self, texts):
        # Build vocabulary from texts
        pass

    def encode(self, text):
        # Convert text to token IDs
        pass

    def decode(self, token_ids):
        # Convert token IDs to text
        pass

# Test your tokenizer
# texts = ["hello", "world"]
# tokenizer = CharTokenizer()
# tokenizer.build_vocab(texts)
# print(tokenizer.encode("hello"))

# ==============================================================================
# Exercise 2: Build Vocabulary with Frequency Cutoff
# ==============================================================================

print("\n" + "="*70)
print("Exercise 2: Build Vocabulary with Frequency Cutoff")
print("="*70)

"""
Create a function that builds a vocabulary but only includes words
that appear at least min_freq times.

Hint: Use Counter to count word frequencies
"""

# TODO: Implement build_vocabulary function
def build_vocabulary(texts, min_freq=2):
    """
    Build vocabulary from texts, only including words with freq >= min_freq

    Args:
        texts: List of text strings
        min_freq: Minimum frequency for a word to be included

    Returns:
        Dictionary mapping words to IDs
    """
    pass

# Test your function
# corpus = ["the cat sat", "the dog sat", "the bird flew", "a cat meowed"]
# vocab = build_vocabulary(corpus, min_freq=2)
# print(f"Vocabulary: {vocab}")

# ==============================================================================
# Exercise 3: Implement Encode and Decode with <UNK> Token
# ==============================================================================

print("\n" + "="*70)
print("Exercise 3: Encode/Decode with Unknown Token")
print("="*70)

"""
Implement encode and decode functions that handle unknown words
by using a special <UNK> token.

Hint: Check if word exists in vocab, otherwise use <UNK>
"""

# TODO: Implement these functions
def encode_with_unk(text, word_to_id, unk_token="<UNK>"):
    """
    Encode text to token IDs, using <UNK> for unknown words

    Args:
        text: Input text string
        word_to_id: Dictionary mapping words to IDs
        unk_token: Token to use for unknown words

    Returns:
        List of token IDs
    """
    pass

def decode_from_ids(token_ids, id_to_word):
    """
    Decode token IDs back to text

    Args:
        token_ids: List of token IDs
        id_to_word: Dictionary mapping IDs to words

    Returns:
        Decoded text string
    """
    pass

# Test your functions
# vocab = {"<UNK>": 0, "the": 1, "cat": 2, "sat": 3}
# text = "the cat sat"
# ids = encode_with_unk(text, vocab)
# print(f"Encoded: {ids}")
# decoded = decode_from_ids(ids, {v: k for k, v in vocab.items()})
# print(f"Decoded: {decoded}")

# ==============================================================================
# Exercise 4: Add Special Tokens
# ==============================================================================

print("\n" + "="*70)
print("Exercise 4: Handle Special Tokens")
print("="*70)

"""
Extend your encoder to add special tokens:
- <BOS> at the beginning of the sequence
- <EOS> at the end of the sequence

Hint: Add these tokens to the vocabulary first, then use them in encoding
"""

# TODO: Implement encode_with_special_tokens
def encode_with_special_tokens(text, word_to_id, add_special=True):
    """
    Encode text with optional special tokens

    Args:
        text: Input text string
        word_to_id: Dictionary mapping words to IDs
        add_special: Whether to add <BOS> and <EOS> tokens

    Returns:
        List of token IDs
    """
    pass

# Test your function
# vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "the": 4, "cat": 5}
# text = "the cat"
# ids = encode_with_special_tokens(text, vocab, add_special=True)
# print(f"With special tokens: {ids}")

# ==============================================================================
# BONUS Exercise 5: Simple BPE Merge Step
# ==============================================================================

print("\n" + "="*70)
print("BONUS Exercise 5: Implement One BPE Merge Step")
print("="*70)

"""
Implement a single step of the BPE algorithm:
1. Find the most frequent adjacent pair of tokens
2. Merge that pair into a single token
3. Update the vocabulary

This is simplified - real BPE is more complex!
"""

# TODO: Implement bpe_merge_step
def bpe_merge_step(word_freqs):
    """
    Perform one BPE merge step

    Args:
        word_freqs: Dictionary mapping tuples of tokens to frequencies
                   Example: {('h', 'e', 'l', 'l', 'o'): 5}

    Returns:
        Updated word_freqs after merging most frequent pair
        Most frequent pair that was merged
    """
    pass

# Test your function
# words = {
#     ('l', 'o', 'w'): 5,
#     ('l', 'o', 'w', 'e', 'r'): 2,
#     ('n', 'e', 'w', 'e', 'r'): 6,
#     ('w', 'i', 'd', 'e', 'r'): 3
# }
# new_words, merged_pair = bpe_merge_step(words)
# print(f"Merged pair: {merged_pair}")
# print(f"Updated vocabulary: {new_words}")

# ==============================================================================
# SOLUTIONS
# ==============================================================================

print("\n" + "="*70)
print("Solutions are below (scroll down)")
print("="*70)

"""
# SOLUTION TO EXERCISE 1: Character Tokenizer

class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def build_vocab(self, texts):
        # Collect unique characters
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)

        # Build mappings
        for i, char in enumerate(sorted(unique_chars)):
            self.char_to_id[char] = i
            self.id_to_char[i] = char

        self.vocab_size = len(unique_chars)

    def encode(self, text):
        return [self.char_to_id[char] for char in text if char in self.char_to_id]

    def decode(self, token_ids):
        return ''.join([self.id_to_char[id] for id in token_ids if id in self.id_to_char])

# Test
texts = ["hello", "world"]
tokenizer = CharTokenizer()
tokenizer.build_vocab(texts)
print(f"Vocabulary size: {tokenizer.vocab_size}")
encoded = tokenizer.encode("hello")
print(f"Encoded 'hello': {encoded}")
decoded = tokenizer.decode(encoded)
print(f"Decoded: '{decoded}'")


# SOLUTION TO EXERCISE 2: Build Vocabulary with Frequency Cutoff

def build_vocabulary(texts, min_freq=2):
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)

    # Build vocabulary with frequency cutoff
    vocab = {"<PAD>": 0, "<UNK>": 1}
    current_id = 2

    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = current_id
            current_id += 1

    return vocab

# Test
corpus = ["the cat sat", "the dog sat", "the bird flew", "a cat meowed"]
vocab = build_vocabulary(corpus, min_freq=2)
print(f"Vocabulary: {vocab}")
print(f"'the' appears {sum(1 for text in corpus for word in text.split() if word == 'the')} times - included")
print(f"'bird' appears 1 time - excluded")


# SOLUTION TO EXERCISE 3: Encode/Decode with <UNK>

def encode_with_unk(text, word_to_id, unk_token="<UNK>"):
    words = text.lower().split()
    unk_id = word_to_id.get(unk_token, 1)  # Default UNK ID is 1
    return [word_to_id.get(word, unk_id) for word in words]

def decode_from_ids(token_ids, id_to_word):
    return ' '.join([id_to_word.get(id, "<UNK>") for id in token_ids])

# Test
vocab = {"<UNK>": 0, "the": 1, "cat": 2, "sat": 3}
text = "the cat sat on mat"  # "on" and "mat" are unknown
ids = encode_with_unk(text, vocab)
print(f"Encoded '{text}': {ids}")
decoded = decode_from_ids(ids, {v: k for k, v in vocab.items()})
print(f"Decoded: '{decoded}'")


# SOLUTION TO EXERCISE 4: Special Tokens

def encode_with_special_tokens(text, word_to_id, add_special=True):
    words = text.lower().split()
    unk_id = word_to_id.get("<UNK>", 1)

    # Encode words
    token_ids = [word_to_id.get(word, unk_id) for word in words]

    # Add special tokens if requested
    if add_special:
        bos_id = word_to_id.get("<BOS>", 2)
        eos_id = word_to_id.get("<EOS>", 3)
        token_ids = [bos_id] + token_ids + [eos_id]

    return token_ids

# Test
vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "the": 4, "cat": 5}
text = "the cat"
ids_with_special = encode_with_special_tokens(text, vocab, add_special=True)
ids_without_special = encode_with_special_tokens(text, vocab, add_special=False)
print(f"With special tokens: {ids_with_special}")
print(f"Without special tokens: {ids_without_special}")


# BONUS SOLUTION TO EXERCISE 5: BPE Merge Step

def bpe_merge_step(word_freqs):
    # Count all adjacent pairs
    pair_counts = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq

    if not pair_counts:
        return word_freqs, None

    # Find most frequent pair
    best_pair = pair_counts.most_common(1)[0][0]

    # Merge this pair in all words
    new_word_freqs = {}
    for word, freq in word_freqs.items():
        # Merge the pair in this word
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                # Merge!
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq

    return new_word_freqs, best_pair

# Test
words = {
    ('l', 'o', 'w'): 5,
    ('l', 'o', 'w', 'e', 'r'): 2,
    ('n', 'e', 'w', 'e', 'r'): 6,
    ('w', 'i', 'd', 'e', 'r'): 3
}
print(f"Original words: {words}")
new_words, merged_pair = bpe_merge_step(words)
print(f"\\nMerged pair: {merged_pair}")
print(f"Updated words: {new_words}")
print(f"\\nNotice that '{merged_pair[0]}' + '{merged_pair[1]}' → '{merged_pair[0] + merged_pair[1]}'")
"""

print("\n" + "="*70)
print("Exercise 1 Complete! Now try Exercise 2.")
print("="*70)

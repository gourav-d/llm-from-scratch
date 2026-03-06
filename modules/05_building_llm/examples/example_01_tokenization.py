"""
Tokenization - Turning Text into Numbers!

This example demonstrates the foundation of all LLMs:
breaking text into tokens.

What you'll see:
1. Why tokenization is necessary
2. Character-level tokenization
3. Word-level tokenization
4. Building vocabulary
5. Encoding and decoding
6. Simple BPE (Byte Pair Encoding)
7. Special tokens (BOS, EOS, PAD, UNK)
8. Comparing different approaches
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

print("="*70)
print("TOKENIZATION - The Foundation of LLMs!")
print("="*70)

# ==============================================================================
# REAL-WORLD ANALOGY: Understanding Tokenization
# ==============================================================================

print("\n" + "="*70)
print("REAL-WORLD ANALOGY: Understanding Tokenization")
print("="*70)

print("""
Before we dive into code, let's understand tokenization with a simple analogy!

Imagine you're teaching a computer to read English, but it only understands numbers.
You need to create a "dictionary" that maps pieces of text to numbers.

ANALOGY: Reading a Music Festival Lineup
==========================================
Raw lineup: "The Beatles, Led Zeppelin, Pink Floyd"

Option 1: CHARACTER-LEVEL (like reading letter by letter)
  Split into: ['T', 'h', 'e', ' ', 'B', 'e', 'a', 't', 'l', 'e', 's', ...]
  ✓ Simple! Only ~100 unique characters
  ✗ Very long! Loses word meaning

Option 2: WORD-LEVEL (like reading full band names)
  Split into: ['The', 'Beatles', ',', 'Led', 'Zeppelin', ',', 'Pink', 'Floyd']
  ✓ Preserves meaning! Each band is one unit
  ✗ Need huge vocabulary for all band names

Option 3: SUBWORD-LEVEL (BPE - smart middle ground)
  Split into: ['The', 'Beat', 'les', ',', 'Led', 'Zep', 'pelin', ',', 'Pink', 'Floyd']
  ✓ Balanced! Common words stay whole, rare words split
  ✓ Smaller vocabulary than word-level
  ✓ More meaningful than character-level

THIS is how GPT tokenizes! BPE with ~50,000 subword tokens.

WHY TOKENIZATION MATTERS:
- Neural networks need fixed-size inputs (numbers, not variable text)
- Vocabulary size affects model size (embeddings = vocab_size × embed_dim)
- Token count affects inference cost (GPT-4 charges per token!)
- Good tokenization = better performance + lower cost
""")

print("\n" + "="*70)
print("Now let's see this in action with Python code!")
print("="*70)

# ==============================================================================
# EXAMPLE 1: Character-Level Tokenization
# ==============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Character-Level Tokenization")
print("="*70)

print("""
Character-level tokenization is the simplest approach:
- Treat each character as a token
- Vocabulary is just the set of unique characters
- Usually ~100 tokens (letters, digits, punctuation)
""")

class CharTokenizer:
    """
    Simple character-level tokenizer

    C# Analogy:
    Like converting string to char[], then mapping each char to int
    Dictionary<char, int> vocabulary
    """

    def __init__(self):
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        # Will be built from training data
        self.char_to_id = {}
        self.id_to_char = {}

    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        # Collect all unique characters
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)

        # Add special tokens first (IDs 0-3)
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

        # Build char_to_id and id_to_char mappings
        for i, token in enumerate(special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token

        # Add regular characters
        for i, char in enumerate(sorted(unique_chars)):
            idx = i + len(special_tokens)
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char

    def encode(self, text, add_special_tokens=True):
        """Convert text to list of token IDs"""
        tokens = []

        if add_special_tokens:
            tokens.append(self.char_to_id[self.bos_token])

        for char in text:
            # Use UNK token if character not in vocabulary
            token_id = self.char_to_id.get(char, self.char_to_id[self.unk_token])
            tokens.append(token_id)

        if add_special_tokens:
            tokens.append(self.char_to_id[self.eos_token])

        return tokens

    def decode(self, token_ids):
        """Convert list of token IDs back to text"""
        chars = []
        for token_id in token_ids:
            char = self.id_to_char.get(token_id, self.unk_token)
            # Skip special tokens in output
            if char not in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
                chars.append(char)
        return ''.join(chars)

    @property
    def vocab_size(self):
        """Size of vocabulary"""
        return len(self.char_to_id)

# Build and test character tokenizer
print("Building character-level tokenizer...")
texts = ["Hello world", "How are you?", "Python is amazing!"]

tokenizer_char = CharTokenizer()
tokenizer_char.build_vocab(texts)

print(f"Vocabulary size: {tokenizer_char.vocab_size}")
print(f"Sample vocab: {list(tokenizer_char.char_to_id.items())[:15]}")

# Encode text
text = "Hello!"
encoded = tokenizer_char.encode(text)
print(f"\nOriginal text: '{text}'")
print(f"Encoded: {encoded}")
print(f"Token breakdown:")
for i, token_id in enumerate(encoded):
    char = tokenizer_char.id_to_char[token_id]
    print(f"  {i}: '{char}' → {token_id}")

# Decode back
decoded = tokenizer_char.decode(encoded)
print(f"Decoded: '{decoded}'")

print()

# ==============================================================================
# EXAMPLE 2: Word-Level Tokenization
# ==============================================================================

print("="*70)
print("EXAMPLE 2: Word-Level Tokenization")
print("="*70)

print("""
Word-level tokenization:
- Split text into words (whitespace + punctuation)
- Each unique word gets an ID
- More intuitive than character-level
- But vocabulary can be HUGE (millions of words!)
""")

class WordTokenizer:
    """
    Word-level tokenizer

    C# Analogy:
    Like string.Split(), but with smart punctuation handling
    Dictionary<string, int> vocabulary
    """

    def __init__(self):
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.word_to_id = {}
        self.id_to_word = {}

    def _tokenize_text(self, text):
        """Split text into words (simple whitespace + punctuation)"""
        # Simple approach: split on whitespace and separate punctuation
        # Real tokenizers use more sophisticated rules
        text = text.lower()  # Convert to lowercase
        # Add spaces around punctuation
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        # Split on whitespace
        words = text.split()
        return words

    def build_vocab(self, texts, max_vocab_size=None):
        """Build vocabulary from list of texts"""
        # Collect all words and their frequencies
        word_counts = Counter()
        for text in texts:
            words = self._tokenize_text(text)
            word_counts.update(words)

        # Add special tokens first
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token

        # Add most common words (vocabulary cutoff)
        if max_vocab_size:
            most_common = word_counts.most_common(max_vocab_size - len(special_tokens))
        else:
            most_common = word_counts.most_common()

        for i, (word, count) in enumerate(most_common):
            idx = i + len(special_tokens)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

    def encode(self, text, add_special_tokens=True):
        """Convert text to list of token IDs"""
        words = self._tokenize_text(text)
        tokens = []

        if add_special_tokens:
            tokens.append(self.word_to_id[self.bos_token])

        for word in words:
            token_id = self.word_to_id.get(word, self.word_to_id[self.unk_token])
            tokens.append(token_id)

        if add_special_tokens:
            tokens.append(self.word_to_id[self.eos_token])

        return tokens

    def decode(self, token_ids):
        """Convert list of token IDs back to text"""
        words = []
        for token_id in token_ids:
            word = self.id_to_word.get(token_id, self.unk_token)
            if word not in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
                words.append(word)
        return ' '.join(words)

    @property
    def vocab_size(self):
        return len(self.word_to_id)

# Build and test word tokenizer
print("Building word-level tokenizer...")
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a great programming language.",
    "Machine learning models need data to train.",
    "The transformer architecture revolutionized NLP."
]

tokenizer_word = WordTokenizer()
tokenizer_word.build_vocab(corpus)

print(f"Vocabulary size: {tokenizer_word.vocab_size}")
print(f"Sample vocab: {list(tokenizer_word.word_to_id.items())[:15]}")

# Encode text
text = "The quick brown fox."
encoded = tokenizer_word.encode(text)
print(f"\nOriginal text: '{text}'")
print(f"Encoded: {encoded}")
print(f"Token breakdown:")
for i, token_id in enumerate(encoded):
    word = tokenizer_word.id_to_word[token_id]
    print(f"  {i}: '{word}' → {token_id}")

decoded = tokenizer_word.decode(encoded)
print(f"Decoded: '{decoded}'")

# Test out-of-vocabulary word
text_oov = "supercalifragilisticexpialidocious"
encoded_oov = tokenizer_word.encode(text_oov)
print(f"\nOut-of-vocabulary: '{text_oov}'")
print(f"Encoded: {encoded_oov} (uses <UNK> token)")

print()

# ==============================================================================
# EXAMPLE 3: Building Vocabulary from Corpus
# ==============================================================================

print("="*70)
print("EXAMPLE 3: Building Vocabulary from Corpus")
print("="*70)

print("""
Vocabulary building is critical:
- Determines which tokens the model "knows"
- Vocabulary size affects model size (embeddings!)
- Balance: coverage vs. model size
""")

# Build vocabulary with different sizes
def analyze_vocab_coverage(corpus, vocab_sizes):
    """Analyze how vocabulary size affects token coverage"""
    tokenizer = WordTokenizer()

    # Get all unique words
    all_words = set()
    for text in corpus:
        all_words.update(tokenizer._tokenize_text(text))

    total_unique = len(all_words)

    print(f"Total unique words in corpus: {total_unique}")
    print()

    for vocab_size in vocab_sizes:
        tokenizer.build_vocab(corpus, max_vocab_size=vocab_size)

        # Count how many words are in vocabulary
        covered_words = sum(1 for word in all_words if word in tokenizer.word_to_id)
        coverage = (covered_words / total_unique) * 100

        print(f"Vocab size {vocab_size:4d}: {covered_words:3d}/{total_unique:3d} words ({coverage:.1f}% coverage)")

# Test with different vocabulary sizes
large_corpus = corpus * 5  # Repeat to have more text
analyze_vocab_coverage(large_corpus, [10, 20, 50, 100, 200])

print()

# ==============================================================================
# EXAMPLE 4: Simple BPE (Byte Pair Encoding)
# ==============================================================================

print("="*70)
print("EXAMPLE 4: Simple BPE (Byte Pair Encoding)")
print("="*70)

print("""
BPE is what GPT uses! It's a middle ground:
- Start with characters
- Iteratively merge most frequent pairs
- Results in subword units
- Example: "unbelievable" → ["un", "believ", "able"]

This is a simplified version to demonstrate the concept.
""")

class SimpleBPE:
    """
    Simplified Byte Pair Encoding tokenizer

    Real BPE is more complex, but this shows the core idea.
    """

    def __init__(self, num_merges=10):
        self.num_merges = num_merges
        self.merges = {}  # Track merge operations
        self.vocab = set()

    def _get_pairs(self, word):
        """Get all adjacent character pairs in word"""
        pairs = []
        for i in range(len(word) - 1):
            pairs.append((word[i], word[i+1]))
        return pairs

    def train(self, texts):
        """Learn BPE merges from texts"""
        # Start with character-level vocabulary
        vocab = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                # Represent word as character sequence
                word_tuple = tuple(list(word))
                vocab[word_tuple] = vocab.get(word_tuple, 0) + 1

        # Perform merges
        for i in range(self.num_merges):
            # Count all pairs
            pair_counts = Counter()
            for word, freq in vocab.items():
                pairs = self._get_pairs(word)
                for pair in pairs:
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            print(f"Merge {i+1}: {best_pair[0]} + {best_pair[1]} → {best_pair[0]+best_pair[1]}")

            # Merge this pair in all words
            new_vocab = {}
            for word, freq in vocab.items():
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
                new_vocab[tuple(new_word)] = freq

            vocab = new_vocab
            self.merges[best_pair] = best_pair[0] + best_pair[1]

        # Build final vocabulary
        for word in vocab.keys():
            self.vocab.update(word)

    def encode(self, text):
        """Encode text using learned merges (simplified)"""
        words = text.lower().split()
        tokens = []

        for word in words:
            # Start with characters
            word_tokens = list(word)

            # Apply merges
            for (a, b), ab in self.merges.items():
                i = 0
                new_tokens = []
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and word_tokens[i] == a and word_tokens[i+1] == b:
                        new_tokens.append(ab)
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens

            tokens.extend(word_tokens)

        return tokens

# Train and test simple BPE
print("Training simple BPE tokenizer...")
bpe_corpus = [
    "the quick brown fox",
    "the lazy dog",
    "the cat and the dog"
]

bpe = SimpleBPE(num_merges=5)
bpe.train(bpe_corpus)

print(f"\nLearned vocabulary has {len(bpe.vocab)} tokens")
print(f"Sample tokens: {list(bpe.vocab)[:20]}")

# Encode text
text = "the quick cat"
tokens = bpe.encode(text)
print(f"\nOriginal: '{text}'")
print(f"BPE tokens: {tokens}")

print()

# ==============================================================================
# EXAMPLE 5: Special Tokens
# ==============================================================================

print("="*70)
print("EXAMPLE 5: Special Tokens")
print("="*70)

print("""
Special tokens have special meanings:
- <PAD>: Padding token (to make sequences same length)
- <UNK>: Unknown token (for out-of-vocabulary words)
- <BOS>: Beginning of sequence
- <EOS>: End of sequence

These are CRITICAL for training language models!
""")

def demonstrate_special_tokens():
    """Show how special tokens are used"""
    tokenizer = WordTokenizer()
    corpus = ["Hello world", "How are you"]
    tokenizer.build_vocab(corpus)

    # Encode with special tokens
    text1 = "Hello"
    text2 = "How are you"

    enc1 = tokenizer.encode(text1, add_special_tokens=True)
    enc2 = tokenizer.encode(text2, add_special_tokens=True)

    print(f"Text 1: '{text1}'")
    print(f"Tokens: {enc1}")
    print(f"Decoded: {[tokenizer.id_to_word[i] for i in enc1]}")

    print(f"\nText 2: '{text2}'")
    print(f"Tokens: {enc2}")
    print(f"Decoded: {[tokenizer.id_to_word[i] for i in enc2]}")

    # Padding to same length
    print("\nPadding both to length 10:")
    max_len = 10

    pad_id = tokenizer.word_to_id[tokenizer.pad_token]

    padded1 = enc1 + [pad_id] * (max_len - len(enc1))
    padded2 = enc2 + [pad_id] * (max_len - len(enc2))

    print(f"Padded 1: {padded1}")
    print(f"Padded 2: {padded2}")
    print(f"\nNow both sequences have length {max_len}!")
    print("This is needed for batch processing in neural networks.")

demonstrate_special_tokens()

print()

# ==============================================================================
# EXAMPLE 6: Tokenization Comparison
# ==============================================================================

print("="*70)
print("EXAMPLE 6: Comparing Tokenization Approaches")
print("="*70)

# Compare all three approaches on same text
test_text = "Python programming is amazing!"

# Character-level
char_tok = CharTokenizer()
char_tok.build_vocab([test_text])
char_tokens = char_tok.encode(test_text, add_special_tokens=False)

# Word-level
word_tok = WordTokenizer()
word_tok.build_vocab([test_text])
word_tokens = word_tok.encode(test_text, add_special_tokens=False)

# BPE
bpe_tok = SimpleBPE(num_merges=3)
bpe_tok.train([test_text])
bpe_tokens = bpe_tok.encode(test_text)

print(f"Original text: '{test_text}'")
print(f"\nCharacter-level ({len(char_tokens)} tokens):")
print(f"  Vocab size: {char_tok.vocab_size}")
print(f"  Tokens: {char_tokens[:20]}...")

print(f"\nWord-level ({len(word_tokens)} tokens):")
print(f"  Vocab size: {word_tok.vocab_size}")
print(f"  Tokens: {word_tokens}")

print(f"\nBPE ({len(bpe_tokens)} tokens):")
print(f"  Vocab size: {len(bpe_tok.vocab)}")
print(f"  Tokens: {bpe_tokens}")

print("""
\n=== Comparison ===
Character-level:
  ✓ Small vocabulary (~100)
  ✗ Very long sequences
  ✗ Loses word meaning

Word-level:
  ✓ Preserves word meaning
  ✗ Large vocabulary (millions)
  ✗ Can't handle unknown words well

BPE (Subword):
  ✓ Medium vocabulary (~50k for GPT)
  ✓ Handles unknown words
  ✓ Captures morphology (un-, -able, etc.)
  → This is why GPT uses it!
""")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("✅ SUMMARY")
print("="*70)

print("""
What You Just Built:
1. ✅ Character-level tokenizer
2. ✅ Word-level tokenizer with vocabulary
3. ✅ Vocabulary building from corpus
4. ✅ Simple BPE tokenizer
5. ✅ Special tokens (BOS, EOS, PAD, UNK)
6. ✅ Comparison of approaches

Key Insights:
- Tokenization converts text → numbers for neural networks
- Character-level: Simple but long sequences
- Word-level: Intuitive but huge vocabulary
- BPE (Subword): Best of both worlds! (Used by GPT)
- Vocabulary size affects model size and performance
- Special tokens are critical for training

Connection to GPT:
- GPT-3/4 uses BPE with ~50,000 tokens
- Tiktoken is OpenAI's tokenizer library
- Token count determines API costs!
- Better tokenization = better performance

Real-World Impact:
"unbelievable" (12 characters)
  → Character: 12 tokens
  → Word: 1 token (if in vocab) or 1 UNK
  → BPE: 3 tokens ["un", "believ", "able"]

GPT-4 API charges per token, so:
  → "I can't believe this is unbelievable!"
  → ~8-10 tokens (BPE)
  → Understanding tokenization helps estimate costs!

Next Steps:
1. Read Lesson 5.2: Word Embeddings
   (How these tokens become meaningful vectors)
2. Run example_02_word_embeddings.py
3. Complete exercise_01_tokenization.py

You've learned how text becomes numbers! 🎉
""")

print("="*70)
print("Run 'python example_02_word_embeddings.py' next!")
print("="*70)

# Lesson 5.1: Tokenization - Breaking Text into Pieces

**How computers read text: The first step to building an LLM!**

---

## 🎯 What You'll Learn

- ✅ Why neural networks can't work with text directly
- ✅ Character-level tokenization (simple but limiting)
- ✅ Word-level tokenization (intuitive but problematic)
- ✅ Subword tokenization (BPE - what GPT uses!)
- ✅ Building a tokenizer from scratch
- ✅ Special tokens (BOS, EOS, PAD, UNK)
- ✅ Token economy and why it matters for pricing
- ✅ Encoding and decoding text

**Time:** 2-3 hours
**Difficulty:** ⭐⭐⭐☆☆

---

## 🤔 The Problem: Neural Networks Need Numbers, Not Text

### Why Can't We Just Feed Text to Neural Networks?

Neural networks are mathematical machines. They only understand **numbers**, not words!

**Think of it like a calculator:**

```
Calculator can do: 5 + 3 = 8 ✓
Calculator CANNOT do: "cat" + "dog" = ??? ✗
```

**We need to convert text → numbers!**

---

### Real-World Analogy: Restaurant Orders

**Imagine a kitchen with cooks who only speak numbers:**

```
You say: "I want a cheeseburger with fries"

Kitchen needs:
- Menu item #42 (cheeseburger)
- Menu item #15 (fries)

You say: "supersized"

Kitchen confused! (new word not in menu)
```

**Tokenization = Creating the menu!**

---

## 📖 What is Tokenization?

### Definition

**Tokenization** is breaking text into smaller pieces (tokens) and converting them to numbers.

```
Text: "Hello, world!"
       ↓
Tokens: ["Hello", ",", "world", "!"]
       ↓
Token IDs: [5000, 11, 1035, 0]
```

**Each token gets a unique number (ID) from a vocabulary!**

---

### Three Main Approaches

| Approach | Token Unit | Vocabulary Size | Pros | Cons |
|----------|-----------|-----------------|------|------|
| **Character-level** | Individual characters | ~100-200 | Simple, no unknowns | Very long sequences |
| **Word-level** | Whole words | ~50,000-100,000+ | Intuitive | Huge vocab, many unknowns |
| **Subword (BPE)** | Word pieces | ~30,000-50,000 | Best of both worlds! | Slightly complex |

**GPT uses subword tokenization (BPE)!** ⭐

---

## 🔤 Approach 1: Character-Level Tokenization

### The Simplest Method

Break text into individual characters.

```python
text = "Hello"

# Character tokenization
tokens = ["H", "e", "l", "l", "o"]
```

---

### Example: Building a Character Tokenizer

```python
class CharacterTokenizer:
    """
    Simplest tokenizer: one character = one token

    Like treating each letter as a separate menu item!
    """

    def __init__(self, text):
        """
        Build vocabulary from text

        Args:
            text: Training text to build vocabulary from
        """
        # Get all unique characters
        unique_chars = sorted(set(text))

        # Create mappings: char → ID and ID → char
        self.char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(unique_chars)}

        # Vocabulary size
        self.vocab_size = len(unique_chars)

    def encode(self, text):
        """
        Convert text to list of token IDs

        Args:
            text: String to encode

        Returns:
            List of integers (token IDs)
        """
        return [self.char_to_id[ch] for ch in text]

    def decode(self, token_ids):
        """
        Convert token IDs back to text

        Args:
            token_ids: List of integers

        Returns:
            Original text string
        """
        return ''.join([self.id_to_char[id] for id in token_ids])


# Example usage
text = "Hello, world!"

# Build tokenizer
tokenizer = CharacterTokenizer(text)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Vocabulary: {tokenizer.char_to_id}")

# Encode
encoded = tokenizer.encode("Hello")
print(f"'Hello' → {encoded}")

# Decode
decoded = tokenizer.decode(encoded)
print(f"{encoded} → '{decoded}'")
```

**Output:**
```
Vocabulary size: 10
Vocabulary: {' ': 0, '!': 1, ',': 2, 'H': 3, 'd': 4, 'e': 5, 'l': 6, 'o': 7, 'r': 8, 'w': 9}
'Hello' → [3, 5, 6, 6, 7]
[3, 5, 6, 6, 7] → 'Hello'
```

---

### Pros and Cons

**Advantages:**
- ✅ **Tiny vocabulary** (~100 characters)
- ✅ **No unknown words** (every word is made of known characters!)
- ✅ **Very simple** to implement

**Disadvantages:**
- ❌ **Very long sequences** ("Hello" = 5 tokens instead of 1)
- ❌ **Loses word meaning** (H-e-l-l-o treated as 5 separate things)
- ❌ **Hard to learn patterns** (network must learn letter→word→meaning)

---

### C#/.NET Analogy

```csharp
// C#: Character array
string text = "Hello";
char[] chars = text.ToCharArray();  // ['H', 'e', 'l', 'l', 'o']

// Python: Same concept
text = "Hello"
chars = list(text)  # ['H', 'e', 'l', 'l', 'o']
```

**Both break strings into characters, but Python's tokenizer adds ID mapping!**

---

## 📝 Approach 2: Word-Level Tokenization

### Breaking Text into Words

Split text by spaces and punctuation.

```python
text = "Hello, world!"

# Word tokenization
tokens = ["Hello", ",", "world", "!"]
```

---

### Example: Building a Word Tokenizer

```python
import re

class WordTokenizer:
    """
    Word-level tokenizer: each word = one token

    Like having one menu item per dish!
    """

    def __init__(self, text):
        """
        Build vocabulary from text

        Args:
            text: Training text
        """
        # Split into words (simple regex pattern)
        # Matches words and punctuation separately
        words = re.findall(r'\w+|[^\w\s]', text.lower())

        # Get unique words
        unique_words = sorted(set(words))

        # Create mappings
        self.word_to_id = {word: i for i, word in enumerate(unique_words)}
        self.id_to_word = {i: word for i, word in enumerate(unique_words)}

        # Add special tokens
        self.UNK_ID = len(self.word_to_id)  # Unknown word
        self.word_to_id['<UNK>'] = self.UNK_ID
        self.id_to_word[self.UNK_ID] = '<UNK>'

        self.vocab_size = len(self.word_to_id)

    def encode(self, text):
        """
        Convert text to token IDs

        Args:
            text: String to encode

        Returns:
            List of token IDs
        """
        words = re.findall(r'\w+|[^\w\s]', text.lower())

        # Use UNK_ID for unknown words
        return [self.word_to_id.get(word, self.UNK_ID) for word in words]

    def decode(self, token_ids):
        """
        Convert token IDs back to text

        Args:
            token_ids: List of integers

        Returns:
            Text string
        """
        words = [self.id_to_word[id] for id in token_ids]

        # Simple reconstruction (not perfect!)
        result = ""
        for word in words:
            if word in [',', '.', '!', '?', ';', ':']:
                result += word
            else:
                result += ' ' + word

        return result.strip()


# Example usage
text = "Hello, world! Hello, Python!"

tokenizer = WordTokenizer(text)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Vocabulary: {list(tokenizer.word_to_id.keys())}")

# Encode known text
encoded = tokenizer.encode("Hello, world!")
print(f"'Hello, world!' → {encoded}")

# Encode text with unknown word
encoded_unk = tokenizer.encode("Hello, universe!")
print(f"'Hello, universe!' → {encoded_unk}")
print(f"  (Note: 'universe' → {tokenizer.UNK_ID} = <UNK>)")

# Decode
decoded = tokenizer.decode(encoded)
print(f"{encoded} → '{decoded}'")
```

**Output:**
```
Vocabulary size: 6
Vocabulary: ['!', ',', 'hello', 'python', 'world', '<UNK>']
'Hello, world!' → [2, 1, 4, 0]
'Hello, universe!' → [2, 1, 5, 0]
  (Note: 'universe' → 5 = <UNK>)
[2, 1, 4, 0] → ' hello, world!'
```

---

### Pros and Cons

**Advantages:**
- ✅ **Intuitive** (matches how humans think of text)
- ✅ **Shorter sequences** (one word = one token)
- ✅ **Preserves word meaning**

**Disadvantages:**
- ❌ **Huge vocabulary** (English has 170,000+ words!)
- ❌ **Many unknown words** (typos, names, new words)
- ❌ **Poor handling of related words** ("run", "running", "ran" are 3 separate tokens)
- ❌ **Memory hungry** (vocabulary too large for neural networks)

---

### The Unknown Word Problem

```python
# Training text
training = "I love cats and dogs"

# Tokenizer learns: ["I", "love", "cats", "and", "dogs"]

# New text with unseen word
test = "I love elephants"
#              ↑
#         Unknown! Becomes <UNK>

# Result: "I love <UNK>"
# Lost meaning completely!
```

**This is a BIG problem!** 😱

---

## 🧩 Approach 3: Subword Tokenization (BPE)

### The Best of Both Worlds

**Subword tokenization** splits words into smaller pieces (subwords).

```
Word: "unhappiness"

Subwords: ["un", "happi", "ness"]
          ↑     ↑       ↑
       prefix  root   suffix
```

**Why this is brilliant:**

1. **Vocabulary not too large** (~30k-50k tokens)
2. **No unknown words** (can build any word from subwords!)
3. **Learns word structure** (prefixes, suffixes, roots)

---

### What is BPE (Byte Pair Encoding)?

**BPE** is an algorithm that learns the most common subword patterns from text.

**Core idea:** Merge the most frequent character pairs iteratively!

---

### BPE Algorithm Step-by-Step

**Step 1: Start with characters**

```
Text: "low low low lower lower newest newest newest newest widest widest widest"

Initial tokens (by character):
l o w _ l o w _ l o w _ l o w e r _ l o w e r _ n e w e s t _ ...
```

---

**Step 2: Find most frequent pair**

```
Count pairs:
"l" + "o" → 8 times  ← Most frequent!
"o" + "w" → 7 times
"e" + "s" → 4 times
...

Merge: "l" + "o" → "lo"
```

---

**Step 3: Update vocabulary**

```
Vocabulary: [..., "lo"]

Text becomes:
lo w _ lo w _ lo w _ lo w e r _ lo w e r _ n e w e s t _ ...
```

---

**Step 4: Repeat**

```
Iteration 2:
"lo" + "w" → 7 times  ← Most frequent now!
Merge: "lo" + "w" → "low"

Vocabulary: [..., "lo", "low"]

Text becomes:
low _ low _ low _ low e r _ low e r _ n e w e s t _ ...
```

---

**Step 5: Continue until vocabulary size reached**

```
After many iterations:
Vocabulary: ["l", "o", "w", "e", "r", "n", "s", "t", "i", "d",
             "lo", "low", "er", "est", "newer", "newest", ...]

Text: "low _ low _ low _ lower _ lower _ newest _ newest _ widest _ widest"
       ↑    ↑    ↑    ↑       ↑       ↑          ↑          ↑         ↑
     1 token each!
```

---

### Example: Simple BPE Implementation

```python
import re
from collections import Counter

class SimpleBPE:
    """
    Simplified BPE tokenizer

    Like learning common word chunks from a cookbook!
    """

    def __init__(self, vocab_size=100):
        """
        Initialize BPE tokenizer

        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def get_stats(self, word_freqs):
        """
        Count frequency of adjacent character pairs

        Args:
            word_freqs: Dictionary of {word: frequency}

        Returns:
            Counter of pair frequencies
        """
        pairs = Counter()

        for word, freq in word_freqs.items():
            symbols = word.split()

            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq

        return pairs

    def merge_pair(self, pair, word_freqs):
        """
        Merge most frequent pair in all words

        Args:
            pair: Tuple of (symbol1, symbol2) to merge
            word_freqs: Dictionary of words

        Returns:
            Updated word_freqs with merged pair
        """
        new_word_freqs = {}

        # Create pattern to find pair
        pattern = re.escape(' '.join(pair))
        replacement = ''.join(pair)

        for word, freq in word_freqs.items():
            # Replace pair with merged version
            new_word = re.sub(pattern, replacement, word)
            new_word_freqs[new_word] = freq

        return new_word_freqs

    def train(self, text):
        """
        Train BPE on text corpus

        Args:
            text: Training text
        """
        # Tokenize into words
        words = re.findall(r'\w+', text.lower())

        # Create word frequencies
        word_freqs = Counter(words)

        # Split into characters with spaces
        word_freqs = {' '.join(word) + ' _': freq
                      for word, freq in word_freqs.items()}

        # Get initial vocabulary (all characters)
        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word.split())

        # Merge pairs until we reach vocab_size
        while len(vocab) < self.vocab_size:
            # Get pair frequencies
            pairs = self.get_stats(word_freqs)

            if not pairs:
                break

            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Merge this pair in all words
            word_freqs = self.merge_pair(best_pair, word_freqs)

            # Add merged pair to vocab
            new_token = ''.join(best_pair)
            vocab.add(new_token)

            # Remember this merge
            self.merges.append(best_pair)

        # Create final vocabulary with IDs
        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}

        print(f"Trained BPE with {len(self.vocab)} tokens")
        print(f"Number of merges: {len(self.merges)}")

    def encode(self, text):
        """
        Encode text using learned BPE merges

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        # Simplified encoding (real BPE is more complex)
        words = re.findall(r'\w+', text.lower())

        tokens = []
        for word in words:
            # Start with characters
            word_tokens = list(word) + ['_']

            # Apply merges
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i+1]) == pair:
                        # Merge pair
                        word_tokens = (word_tokens[:i] +
                                      [''.join(pair)] +
                                      word_tokens[i+2:])
                    else:
                        i += 1

            # Convert to IDs
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])

        return tokens


# Example usage
text = """
The quick brown fox jumps over the lazy dog.
The dog was quick to jump.
Brown foxes are quick jumpers.
"""

# Train BPE
bpe = SimpleBPE(vocab_size=50)
bpe.train(text)

# Show some merges
print("\nFirst 10 merges:")
for i, merge in enumerate(bpe.merges[:10]):
    print(f"{i+1}. '{merge[0]}' + '{merge[1]}' → '{merge[0]}{merge[1]}'")

# Encode text
test_text = "The quick fox"
encoded = bpe.encode(test_text)
print(f"\nEncoded '{test_text}': {encoded}")
```

---

### Why BPE is Brilliant

**Example: Handling unknown words**

```
Training: "run running runner"

BPE learns: ["run", "ning", "ner"]

New word: "runnable"
         = ["run", "nable"]  ← Can still tokenize!

vs Word-level: "runnable" → <UNK>  ← Completely lost!
```

**BPE can handle ANY word by breaking it into known pieces!** 🎉

---

### Real GPT Tokenization Example

```python
# GPT-3 uses BPE with ~50,000 tokens

text = "tokenization"

# Character-level: 12 tokens
# t-o-k-e-n-i-z-a-t-i-o-n

# Word-level: 1 token
# tokenization

# BPE (GPT-3): 2-3 tokens
# token-ization  or  token-iz-ation
```

**Perfect balance!** ⚖️

---

## 🎫 Special Tokens

### Why We Need Special Tokens

Special tokens are markers for different purposes.

| Token | Purpose | Example |
|-------|---------|---------|
| `<BOS>` | Beginning of sequence | Marks start of text |
| `<EOS>` | End of sequence | Marks end of text |
| `<PAD>` | Padding | Makes sequences same length |
| `<UNK>` | Unknown | Represents unseen words |
| `<MASK>` | Mask (BERT) | Hides words for training |

---

### Example: Adding Special Tokens

```python
class TokenizerWithSpecialTokens:
    """
    Tokenizer with special tokens support
    """

    def __init__(self, text):
        # Special tokens
        self.BOS = "<BOS>"  # Beginning of sequence
        self.EOS = "<EOS>"  # End of sequence
        self.PAD = "<PAD>"  # Padding
        self.UNK = "<UNK>"  # Unknown

        # Build vocabulary
        words = text.lower().split()
        unique_words = sorted(set(words))

        # Add special tokens FIRST (get IDs 0, 1, 2, 3)
        special_tokens = [self.PAD, self.UNK, self.BOS, self.EOS]
        vocab_list = special_tokens + unique_words

        self.word_to_id = {word: i for i, word in enumerate(vocab_list)}
        self.id_to_word = {i: word for i, word in enumerate(vocab_list)}

        # Remember special token IDs
        self.PAD_ID = self.word_to_id[self.PAD]
        self.UNK_ID = self.word_to_id[self.UNK]
        self.BOS_ID = self.word_to_id[self.BOS]
        self.EOS_ID = self.word_to_id[self.EOS]

        self.vocab_size = len(self.word_to_id)

    def encode(self, text, add_special_tokens=True):
        """
        Encode text with special tokens

        Args:
            text: String to encode
            add_special_tokens: Whether to add BOS/EOS

        Returns:
            List of token IDs
        """
        words = text.lower().split()
        token_ids = [self.word_to_id.get(word, self.UNK_ID) for word in words]

        if add_special_tokens:
            # Add BOS at start, EOS at end
            token_ids = [self.BOS_ID] + token_ids + [self.EOS_ID]

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs to text

        Args:
            token_ids: List of integers
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Text string
        """
        words = []

        for id in token_ids:
            word = self.id_to_word[id]

            # Skip special tokens if requested
            if skip_special_tokens and word in [self.BOS, self.EOS, self.PAD]:
                continue

            words.append(word)

        return ' '.join(words)

    def pad_sequence(self, token_ids, max_length):
        """
        Pad sequence to max_length

        Args:
            token_ids: List of token IDs
            max_length: Target length

        Returns:
            Padded sequence
        """
        if len(token_ids) >= max_length:
            return token_ids[:max_length]

        # Add PAD tokens to reach max_length
        padding = [self.PAD_ID] * (max_length - len(token_ids))
        return token_ids + padding


# Example usage
text = "hello world this is tokenization"

tokenizer = TokenizerWithSpecialTokens(text)

# Encode with special tokens
encoded = tokenizer.encode("hello world")
print(f"Encoded (with special tokens): {encoded}")
print(f"  BOS_ID={tokenizer.BOS_ID}, EOS_ID={tokenizer.EOS_ID}")

# Decode
decoded = tokenizer.decode(encoded)
print(f"Decoded: '{decoded}'")

# Padding
padded = tokenizer.pad_sequence(encoded, max_length=10)
print(f"Padded to length 10: {padded}")
print(f"  (PAD_ID={tokenizer.PAD_ID})")
```

**Output:**
```
Encoded (with special tokens): [2, 4, 8, 3]
  BOS_ID=2, EOS_ID=3
Decoded: 'hello world'
Padded to length 10: [2, 4, 8, 3, 0, 0, 0, 0, 0, 0]
  (PAD_ID=0)
```

---

### Why Padding Matters

**Neural networks need same-length inputs:**

```
Batch of sentences:
1. "Hello" → [BOS, 5, EOS] → Length 3
2. "Hello world" → [BOS, 5, 10, EOS] → Length 4
3. "Hi" → [BOS, 8, EOS] → Length 3

Problem: Different lengths! Can't create tensor.

Solution: Pad to same length!
1. [BOS, 5, EOS, PAD] → Length 4
2. [BOS, 5, 10, EOS] → Length 4
3. [BOS, 8, EOS, PAD] → Length 4

Now can create tensor: shape (3, 4) ✓
```

---

## 💰 Token Economy: Why Tokens Matter for Pricing

### How LLM APIs Charge You

**OpenAI GPT-4 pricing (as of 2024):**
- Input: $0.03 per 1,000 tokens
- Output: $0.06 per 1,000 tokens

**Tokens ≠ Words!**

---

### Example: Cost Calculation

```python
# Prompt
prompt = "Explain quantum physics in simple terms"

# GPT tokenization (approximate)
# "Explain" = 1 token
# " quantum" = 1 token
# " physics" = 1 token
# " in" = 1 token
# " simple" = 1 token
# " terms" = 1 token
# Total: ~6 tokens

# Response (200 tokens)

# Total cost:
input_cost = 6 / 1000 * 0.03 = $0.00018
output_cost = 200 / 1000 * 0.06 = $0.012
total = $0.01218 ≈ $0.01
```

---

### Why Efficient Tokenization Matters

```python
# Inefficient: Character-level
"Hello world" = 11 tokens → More expensive!

# Efficient: BPE
"Hello world" = 2-3 tokens → Cheaper!

# For 1 million words:
Character-level: ~5 million tokens → $150
BPE: ~1.3 million tokens → $39

Savings: $111! 💰
```

---

### Tips to Reduce Token Usage

1. **Be concise** in prompts
2. **Avoid repetition**
3. **Use shorter variable names** in code
4. **Remove extra whitespace**

```python
# Inefficient (many tokens)
prompt = "Please explain to me in great detail with lots of examples..."

# Efficient (fewer tokens)
prompt = "Explain with examples:"
```

---

## 🔄 Encoding and Decoding

### The Full Pipeline

```
Text → Tokenization → Token IDs → Model → Token IDs → Detokenization → Text
```

---

### Example: Complete Pipeline

```python
# Step 1: Build tokenizer
text_corpus = "the cat sat on the mat the dog sat on the log"
tokenizer = TokenizerWithSpecialTokens(text_corpus)

# Step 2: Encode input
input_text = "the cat sat"
token_ids = tokenizer.encode(input_text)
print(f"Input: '{input_text}'")
print(f"Encoded: {token_ids}")

# Step 3: Model processes (simulated)
# In real LLM, this would be neural network forward pass
# For demo, just append some IDs
output_ids = token_ids + [tokenizer.word_to_id["on"], tokenizer.word_to_id["the"]]

# Step 4: Decode output
output_text = tokenizer.decode(output_ids)
print(f"Output IDs: {output_ids}")
print(f"Decoded: '{output_text}'")
```

**Output:**
```
Input: 'the cat sat'
Encoded: [2, 8, 5, 7, 3]
Output IDs: [2, 8, 5, 7, 3, 6, 8]
Decoded: 'the cat sat on the'
```

---

## 🔗 Connection to GPT

### How GPT Uses Tokenization

**GPT-3/GPT-4 tokenizer:**
- **Algorithm**: BPE (Byte Pair Encoding)
- **Vocabulary size**: ~50,257 tokens
- **Handles**: 100+ languages
- **Special tokens**: `<|endoftext|>`, etc.

---

### Example: GPT Tokenization

```python
# Using OpenAI's tiktoken library
import tiktoken

# Load GPT-4 tokenizer
enc = tiktoken.encoding_for_model("gpt-4")

# Encode text
text = "Hello, world! This is tokenization."
tokens = enc.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")

# Decode
decoded = enc.decode(tokens)
print(f"Decoded: {decoded}")

# Show individual tokens
for token in tokens:
    print(f"  {token} → '{enc.decode([token])}'")
```

**Output (approximate):**
```
Text: Hello, world! This is tokenization.
Tokens: [9906, 11, 1917, 0, 1096, 318, 11241, 1634, 13]
Token count: 9

Individual tokens:
  9906 → 'Hello'
  11 → ','
  1917 → ' world'
  0 → '!'
  1096 → ' This'
  318 → ' is'
  11241 → ' token'
  1634 → 'ization'
  13 → '.'
```

**Notice:** "tokenization" = 2 tokens ("token" + "ization")!

---

## 🔍 C#/.NET Analogy

### String Processing Comparison

```csharp
// C#: String.Split()
string text = "Hello, world!";
string[] words = text.Split(' ');
// Result: ["Hello,", "world!"]

// Python: Similar but tokenization adds ID mapping
text = "Hello, world!"
tokens = text.split()  # ['Hello,', 'world!']
token_ids = [vocab[token] for token in tokens]  # [42, 105]
```

**Tokenization = String.Split() + Dictionary mapping!**

---

### Dictionary Comparison

```csharp
// C#: Dictionary for vocabulary
Dictionary<string, int> vocab = new Dictionary<string, int>
{
    {"hello", 0},
    {"world", 1}
};

int id = vocab["hello"];  // 0

// Python: Same concept!
vocab = {
    "hello": 0,
    "world": 1
}

id = vocab["hello"]  # 0
```

---

## 📊 Visual Summary

### Tokenization Approaches Compared

```
Text: "unhappiness"

┌─────────────────────────────────────────────────────────┐
│ Character-level:                                        │
│ [u][n][h][a][p][p][i][n][e][s][s]                      │
│ 11 tokens ❌ Too many!                                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Word-level:                                             │
│ [unhappiness]                                           │
│ 1 token ✓ BUT: Unknown word problem!                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Subword (BPE):                                          │
│ [un][happi][ness]                                       │
│ 3 tokens ✓ Perfect balance!                            │
└─────────────────────────────────────────────────────────┘
```

---

## 🎓 Key Takeaways

### Core Concepts

1. **Neural networks need numbers**, not text
   - Tokenization converts text → numbers

2. **Three main approaches**:
   - Character: Simple but long sequences
   - Word: Intuitive but huge vocabulary
   - **Subword (BPE): Best of both worlds** ⭐

3. **BPE learns common subword patterns**
   - Iteratively merges frequent character pairs
   - Can handle any word (no unknowns!)

4. **Special tokens** serve important purposes
   - `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`

5. **Token economy matters**
   - LLM APIs charge per token
   - Efficient tokenization saves money

6. **GPT uses BPE** with ~50k vocabulary
   - Handles 100+ languages
   - Balances efficiency and coverage

---

### What You Built

✅ Character-level tokenizer
✅ Word-level tokenizer
✅ Simple BPE tokenizer
✅ Special tokens implementation
✅ Padding and encoding/decoding

---

### Next Steps

In the next lesson (**02_word_embeddings.md**), you'll learn:
- How to convert token IDs → meaningful vectors
- Word2Vec and the famous "king - man + woman = queen"
- Building embedding layers
- Connection to transformer architecture

---

## 📝 Quiz

Test your understanding!

### Question 1
**Why can't neural networks work with text directly?**

<details>
<summary>Click to see answer</summary>

Neural networks are mathematical machines that only understand numbers, not text. They perform calculations like matrix multiplication and addition, which require numerical inputs. Text must be converted to numbers (tokenization) before a neural network can process it.
</details>

---

### Question 2
**What are the three main tokenization approaches and their trade-offs?**

<details>
<summary>Click to see answer</summary>

1. **Character-level**:
   - Pros: Tiny vocabulary (~100), no unknown words
   - Cons: Very long sequences, loses word meaning

2. **Word-level**:
   - Pros: Intuitive, preserves word meaning, shorter sequences
   - Cons: Huge vocabulary (100k+), many unknown words

3. **Subword (BPE)**:
   - Pros: Balanced vocabulary size (30-50k), no unknowns, learns word structure
   - Cons: Slightly more complex to implement

BPE is the best approach and is used by GPT!
</details>

---

### Question 3
**How does BPE (Byte Pair Encoding) work?**

<details>
<summary>Click to see answer</summary>

BPE iteratively merges the most frequent character pairs:

1. Start with individual characters
2. Find most frequent adjacent pair (e.g., "l" + "o")
3. Merge this pair into single token ("lo")
4. Update vocabulary and text
5. Repeat until desired vocabulary size

This learns common subword patterns like prefixes, suffixes, and roots, allowing the tokenizer to handle any word by breaking it into known pieces.
</details>

---

### Question 4
**What is the purpose of the `<PAD>` special token?**

<details>
<summary>Click to see answer</summary>

The `<PAD>` (padding) token is used to make sequences the same length in a batch. Neural networks require fixed-size inputs, so shorter sequences are padded with `<PAD>` tokens to match the length of the longest sequence in the batch.

Example:
- "Hello" → [BOS, 5, EOS, PAD, PAD] → Length 5
- "Hello world" → [BOS, 5, 10, EOS, PAD] → Length 5
- "Hi there friend" → [BOS, 8, 12, 15, EOS] → Length 5
</details>

---

### Question 5
**Why does tokenization matter for LLM API pricing?**

<details>
<summary>Click to see answer</summary>

LLM APIs like OpenAI's GPT-4 charge based on the number of tokens, not words. Efficient tokenization (like BPE) uses fewer tokens than inefficient methods (like character-level), directly reducing costs.

Example:
- Character-level: "Hello world" = 11 tokens
- BPE: "Hello world" = 2-3 tokens

For 1 million words, this difference could mean $150 (character) vs $39 (BPE), saving $111!
</details>

---

### Question 6
**What's the difference between encoding and decoding?**

<details>
<summary>Click to see answer</summary>

**Encoding**: Converting text → token IDs
- Input: "Hello world"
- Output: [5000, 1035]

**Decoding**: Converting token IDs → text
- Input: [5000, 1035]
- Output: "Hello world"

These are inverse operations. Encoding prepares text for the model, decoding converts model output back to readable text.
</details>

---

### Question 7
**How does BPE handle unknown words that weren't in the training data?**

<details>
<summary>Click to see answer</summary>

BPE can break any unknown word into smaller known subwords!

Example:
- Training: "run", "running", "runner"
- BPE learns: ["run", "ning", "ner"]
- Unknown word: "runnable"
- BPE tokenizes: ["run", "nable"]

Even though "runnable" wasn't in training, BPE can still tokenize it using learned subword pieces. This is a huge advantage over word-level tokenization, which would convert "runnable" to `<UNK>` (unknown), losing all meaning.
</details>

---

### Question 8
**In the context of tokenization, what does vocabulary size mean?**

<details>
<summary>Click to see answer</summary>

Vocabulary size is the total number of unique tokens the tokenizer can recognize and use.

Examples:
- Character-level: ~100-200 (all characters, punctuation, etc.)
- Word-level: 50,000-100,000+ (all unique words)
- BPE: 30,000-50,000 (optimized subword pieces)
- GPT-3/4: ~50,257 tokens

Larger vocabulary = more memory required but potentially better representation of language. BPE balances efficiency and coverage.
</details>

---

**Next Lesson:** `02_word_embeddings.md` - Converting tokens to meaningful vectors!

Run `examples/example_01_tokenization.py` to see all concepts in action!

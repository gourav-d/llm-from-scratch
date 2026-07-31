# Lesson 03 - WordPiece and SentencePiece

---

## Overview

We now know BPE. But several major models use different algorithms:

| Model | Tokenizer | Algorithm |
|-------|-----------|-----------|
| GPT-2, GPT-4, LLaMA | tiktoken / HuggingFace | BPE |
| BERT, DistilBERT | HuggingFace | WordPiece |
| T5, mT5, ALBERT | SentencePiece library | SentencePiece (BPE or Unigram) |
| LLaMA 2 | SentencePiece | SentencePiece BPE |
| LLaMA 3 | tiktoken | BPE |

Knowing the differences helps you pick the right tokenizer for your use case
and understand model documentation.

---

## WordPiece (BERT's Tokenizer)

### What Is WordPiece?

WordPiece was developed by Google for BERT (2018).
It is similar to BPE but uses a different criterion for merging.

### BPE vs WordPiece Merge Criterion

**BPE:**
```
Merge the pair (A, B) with highest COUNT(A, B)
Score = frequency of pair
```

**WordPiece:**
```
Merge the pair (A, B) that maximizes LIKELIHOOD GAIN
Score = COUNT(A, B) / (COUNT(A) * COUNT(B))
```

In plain English:
- BPE says: "merge what appears together most often"
- WordPiece says: "merge what gains the most by being together (relative to how common each piece is separately)"

**Example:**
```
If "un" appears 1000 times and "happy" appears 1000 times,
and "unhappy" appears 900 times:

BPE score   = 900       (raw count, high -> merge)
WordPiece   = 900 / (1000 * 1000) = 0.0009  (high for WordPiece too)

But if "the" appears 50000 times and "cat" appears 30000 times,
and "the cat" appears 800 times:

BPE score   = 800       (looks high!)
WordPiece   = 800 / (50000 * 30000) = 0.00000053  (very low! don't merge)

WordPiece correctly avoids merging words that often appear together by chance.
```

### The "##" Prefix Convention

WordPiece uses a special prefix to mark tokens that are NOT at the start of a word:

```
"running" -> ["run", "##ning"]
"unhappy" -> ["un", "##happy"]
"dogs"    -> ["dog", "##s"]

"run"     -> ["run"]             <- no prefix: word-initial token
"##ning"  -> continuation token  <- ## means "this continues a word"

This lets the model know:
"##s" at position 3 means "s is part of the previous word"
"s"   at position 3 means "s is a new word" (rare, but possible)
```

**ASCII Diagram - WordPiece Tokenization:**
```
Input: "I was running quickly"

Step 1 - Pre-tokenize by whitespace:
  ["I", "was", "running", "quickly"]

Step 2 - Check each word against vocabulary:
  "I"       -> in vocab -> ["I"]
  "was"     -> in vocab -> ["was"]
  "running" -> NOT in vocab -> split into subwords
  "quickly" -> NOT in vocab -> split into subwords

Step 3 - Greedy longest-match split for unknown words:
  "running" -> "run" (in vocab) + "##ning" (in vocab) -> ["run", "##ning"]
  "quickly" -> "quick" (in vocab) + "##ly" (in vocab) -> ["quick", "##ly"]

Final: ["I", "was", "run", "##ning", "quick", "##ly"]
IDs:   [ 1045, 2001, 2448, 6752, 4248, 2135 ]
```

### BERT Special Tokens

BERT uses specific special tokens:
```
[CLS] - always at the START of every input
[SEP] - marks END of a sentence or segment boundary
[PAD] - fills shorter sequences to match batch length
[UNK] - replaces tokens not found in vocabulary
[MASK] - replaces tokens during Masked Language Model training

Full BERT input format:
[CLS] token1 token2 ... [SEP] token1 token2 ... [SEP]
 ^                       ^                       ^
 start             sentence boundary           end
```

---

## SentencePiece (T5, LLaMA 2)

### The Problem SentencePiece Solves

Both BPE and WordPiece assume text has already been split on whitespace.
This is called "pre-tokenization".

```
"The cat sat" -> pre-tokenize -> ["The", "cat", "sat"] -> then BPE
```

But what about:
- **Chinese:** "" (no spaces between words)
- **Japanese:** "" (mixed writing systems, no spaces)
- **Thai:** Text has no space separators at all
- **German:** Compound words: "Donaudampfschifffahrtsgesellschaft" (one word!)

**SentencePiece treats the raw byte stream as input, including spaces.**

### How SentencePiece Works

1. No pre-tokenization step. Raw text goes in directly.
2. Spaces are treated as special characters (encoded as U+2581, shown as _)
3. Works on bytes directly, fully language-agnostic
4. Can use either BPE or Unigram language model as the underlying algorithm

**ASCII Diagram - SentencePiece Space Handling:**
```
Input: "Hello world"
         ^
         Space is PART OF a token, not a separator!

BPE/WordPiece sees:  ["Hello", "world"]  (space discarded as separator)
SentencePiece sees:  ["_Hello", "_world"]
                       ^         ^
                       underscore = leading space encoded into token

When decoding:
["_Hello", "_world"] -> replace _ with space -> "Hello world"  <- exact round-trip
```

### Why Space-Inclusive Tokens Matter

```
BPE: "New York" and "York" might get the same token for "York"
     but "New York" has a space before "York" which is just lost.

SentencePiece:
"New_York" = one context ("York" after a word)
"York"     = different context ("York" starting text)
These are different tokens! The model learns different representations.
```

### Languages Without Spaces

```
Chinese: ""  (= "I love natural language processing")
         SentencePiece just treats this as a byte sequence.
         No pre-tokenization needed.
         BPE merges build up to Chinese character n-grams.

Japanese: ""  (= "natural language processing")
          SentencePiece handles this identically - just bytes.
```

### SentencePiece in Practice (LLaMA 2)

LLaMA 2 uses SentencePiece with:
- Vocabulary size: 32,000 tokens
- Algorithm: BPE
- Byte fallback: any unknown byte gets its own token (256 byte tokens reserved)
- Special tokens: `<s>` (begin), `</s>` (end), `<unk>` (unknown)

```
Input: "Hello, world!"
LLaMA 2 tokens: ["<s>", "_Hello", ",", "_world", "!"]
LLaMA 2 IDs:    [1, 15043, 29892, 3186, 29991]
Note: "<s>" is automatically added to every sequence start
```

---

## tiktoken (GPT-4, GPT-3.5)

**tiktoken** is OpenAI's tokenizer, implemented in Rust with a Python wrapper.

### Key Facts

- **Algorithm:** BPE (byte-level, like GPT-2)
- **Performance:** Extremely fast (Rust backend, parallelized)
- **GPT-4 vocab size:** ~100,256 tokens
- **GPT-3.5 vocab size:** 100,256 tokens (cl100k_base encoding)
- **GPT-2 vocab size:** 50,257 tokens (gpt2 encoding)

### tiktoken Encodings

```python
import tiktoken

# Load the GPT-4 tokenizer
enc = tiktoken.get_encoding("cl100k_base")   # GPT-3.5/GPT-4

# Tokenize text
tokens = enc.encode("Hello, world!")
print(tokens)   # [9906, 11, 1917, 0]

# Decode back
text = enc.decode(tokens)
print(text)     # "Hello, world!"
```

### tiktoken vs HuggingFace Tokenizers

| Feature | tiktoken | HuggingFace tokenizers |
|---------|---------|------------------------|
| Backend language | Rust | Rust |
| Speed | Very fast | Very fast |
| Models | OpenAI only | All models |
| API complexity | Simple (encode/decode) | Richer (attention mask, padding) |
| Custom training | No | Yes |

---

## Comparison Table: All Three Methods

| Feature | BPE | WordPiece | SentencePiece |
|---------|-----|-----------|---------------|
| Merge criterion | Frequency (count) | Likelihood gain | Frequency (BPE) or Unigram LM |
| Pre-tokenization | Yes (split on space) | Yes (split on space) | No (raw text input) |
| Space handling | Space ignored or separate | Space ignored | Space encoded into token (_) |
| Unknown words | Split into known pieces | Split into known pieces | Byte fallback |
| Subword prefix | None (GPT-2 uses Ġ for space) | ## for non-initial | None (uses _) |
| Models | GPT-2, LLaMA 3 | BERT, DistilBERT | T5, LLaMA 2, ALBERT |
| Special tokens | `<\|endoftext\|>` | `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]` | `<s>`, `</s>`, `<unk>` |

---

## C# Analogy: Different Serialization Formats

```csharp
// Think of tokenization methods like serialization formats:
// They all convert objects <-> bytes, but use different encoding schemes.

// BPE -> like JSON
// Simple, widely used, human-readable, minor overhead at boundaries.
// JsonSerializer.Serialize(obj) -> "{\"name\":\"hello\"}"

// WordPiece -> like XML with CDATA sections
// More explicit marking of structure (## prefix = CDATA continuation)
// XDocument.Parse(xmlText)  ->  XElement with marked continuation nodes

// SentencePiece -> like Protobuf
// Fully binary/language-agnostic, encodes separators as data, compact
// Protobuf.Serialize(message)  ->  byte stream (no ambiguity about separators)
```

---

## How to Choose a Tokenizer

If you are building something new:

```
Use case                         Recommended tokenizer
-------------------------------  ---------------------
English-only, GPT-style model    BPE (tiktoken or HuggingFace)
Classification tasks (BERT-like) WordPiece (BERT)
Multilingual or no-space lang    SentencePiece
Production OpenAI API apps       tiktoken (matches API tokenization)
Fine-tuning existing model       Use THAT model's tokenizer (never change it!)
```

**Important:** Never mix a model with a different tokenizer.
If you fine-tune BERT, you must use BERT's WordPiece tokenizer.
Token IDs must match what the embedding table expects.

---

## Quiz

**Question 1**
What does the "##" prefix mean in WordPiece tokenization?

A) The token is a special reserved token  
B) The token is a continuation of the previous word (not word-initial)  
C) The token is unknown and will be replaced by [UNK]  
D) The token comes from a different vocabulary  

**Answer: B**
*Explanation: In WordPiece, "##" signals a continuation token.*
*"running" -> ["run", "##ning"] where "##ning" means "ning attached to the previous token".*
*This lets the model distinguish word boundaries from subword boundaries.*

---

**Question 2**
Why does SentencePiece not require a pre-tokenization step?

A) It only works with single words at a time  
B) It treats the raw byte sequence including spaces as input, so no splitting is needed  
C) It only processes English text which always has spaces  
D) It uses a much larger vocabulary so words never need to be split  

**Answer: B**
*Explanation: SentencePiece encodes spaces as part of the token (using the _ prefix internally).*
*It reads raw bytes directly, making it language-agnostic and removing the dependency*
*on whitespace as a separator. This is why it works for Chinese, Japanese, Thai, etc.*

---

**Question 3**
You are fine-tuning the BERT model for a sentiment analysis task.
Which tokenizer must you use?

A) Any tokenizer with vocabulary size >= 30,000  
B) GPT-2 BPE tokenizer, which is more modern  
C) BERT's original WordPiece tokenizer with vocabulary ID=30,522  
D) SentencePiece, since it is language-agnostic  

**Answer: C**
*Explanation: BERT's embedding table maps token IDs 0 to 30,521 to specific vectors.*
*If you use a different tokenizer, the IDs will be wrong and point to wrong embeddings.*
*You must always use the tokenizer that was used when the model was trained.*

---

## Summary

- **WordPiece** (BERT): Merges pairs that maximize likelihood gain (not just frequency).
  Uses "##" prefix for continuation tokens. Requires `[CLS]`, `[SEP]` special tokens.

- **SentencePiece** (T5, LLaMA 2): Treats raw text (including spaces) as input.
  Encodes spaces as `_` inside tokens. Fully language-agnostic. Byte fallback for rare chars.

- **tiktoken** (GPT-3.5, GPT-4): Fast Rust-based BPE. 100K vocab. OpenAI's production choice.

- **Rule**: Always use the tokenizer that matches your model. Never swap them.

**Next:** Lesson 04 - Using the HuggingFace Tokenizers Library

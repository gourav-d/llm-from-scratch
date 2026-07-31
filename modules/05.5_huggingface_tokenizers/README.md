# Module 05.5 - HuggingFace Tokenizers
## (Optional Module - Between M05 and M06)

---

## What Is This Module?

This is an **optional deep-dive** into tokenization.

You already know from M05 that:
- Text must be converted to token IDs before a model can read it
- A vocabulary maps tokens (text pieces) to integer IDs
- GPT-2 uses 50,257 tokens

This module explains **HOW that conversion works under the hood**, and teaches
you to use the HuggingFace `tokenizers` library for real-world tokenization tasks.

---

## Prerequisites

Before starting this module, make sure you have completed:

| Module | Topic | Why Needed |
|--------|-------|------------|
| M05 - Building LLM | Token IDs, vocab, embeddings | Core vocabulary concepts |
| M02 - NumPy | Arrays, integers | Token ID lists are integer arrays |
| M01 - Python Basics | Dictionaries, loops | Vocab = dictionary |

---

## What You Will Learn

```
1. Tokenization Basics
   - Why word-level and character-level tokenization fail
   - How subword tokenization solves both problems
   - What a vocabulary is (and why size matters)
   - Special tokens: [CLS], [SEP], [PAD], [UNK], [MASK]

2. The BPE Algorithm (Byte Pair Encoding)
   - How BPE was invented (text compression, 1994)
   - Step-by-step BPE training with a worked example
   - How GPT-2's tokenizer was built using BPE
   - Implementing BPE from scratch in Python

3. WordPiece and SentencePiece
   - How BERT's WordPiece differs from BPE
   - How SentencePiece (T5, LLaMA) handles any language
   - tiktoken: OpenAI's fast tokenizer
   - Comparison table of all methods

4. HuggingFace Tokenizers Library
   - Installing and importing the library
   - AutoTokenizer: one class for all models
   - encode() and decode()
   - Padding and attention masks

5. Training a Custom Tokenizer
   - When and why you need a custom tokenizer
   - BpeTrainer and WordPieceTrainer
   - Saving and loading your tokenizer
```

---

## C# Developer Summary

If you are coming from C#, here is the mental model:

```
C# Concept                  Python/HuggingFace Equivalent
--------------------------  ---------------------------------
Dictionary<string, int>     Vocabulary (token -> ID lookup)
JsonSerializer.Serialize    tokenizer.encode()  (text -> IDs)
JsonSerializer.Deserialize  tokenizer.decode()  (IDs -> text)
Regex.Split()               Pre-tokenizer (splits raw text)
string.Intern()             Token deduplication in vocab
IEnumerable<string>         List of tokens
int[]                       List of token IDs
```

---

## Install

The examples in this module are written to work **with or without** the HuggingFace
tokenizers library. Core concepts are implemented in pure Python.

To install the optional HuggingFace library (recommended):

```bash
# Activate your virtual environment first
# Windows:
venv\Scripts\activate

# Install the tokenizers library
pip install tokenizers

# Also install transformers for AutoTokenizer
pip install transformers

# Verify installation
python -c "import tokenizers; print('tokenizers version:', tokenizers.__version__)"
```

If installation fails or you prefer to skip it, all examples have pure Python
fallback implementations that demonstrate the same concepts.

---

## File List

```
05.5_huggingface_tokenizers/
|
+-- README.md                          <- You are here
|
+-- 01_tokenization_basics.md          <- What is tokenization and why it matters
+-- 02_bpe_algorithm.md                <- Byte Pair Encoding explained step by step
+-- 03_wordpiece_sentencepiece.md      <- BERT and T5 tokenizer variants
+-- 04_huggingface_tokenizers_library.md  <- Using the HF library
+-- 05_training_custom_tokenizer.md    <- Train your own tokenizer
|
+-- examples/
|   +-- example_01_tokenization_basics.py     <- Word/char/subword comparison
|   +-- example_02_bpe_algorithm.py           <- BPE from scratch
|   +-- example_03_tokenizer_comparison.py    <- BPE vs WordPiece vs SentencePiece
|   +-- example_04_huggingface_tokenizer.py   <- HuggingFace API walkthrough
|   +-- example_05_train_custom_tokenizer.py  <- Train BPE on custom corpus
|
+-- exercises/
    +-- exercise_01_tokenization_basics.py    <- Build simple tokenizers
    +-- exercise_02_bpe_algorithm.py          <- Implement BPE merge steps
    +-- exercise_03_tokenizer_comparison.py   <- Compare methods on same text
    +-- exercise_04_huggingface_tokenizer.py  <- Practice HF API
    +-- exercise_05_train_custom_tokenizer.py <- Train domain tokenizer
```

---

## How to Run

```bash
# Run any example
python modules/05.5_huggingface_tokenizers/examples/example_01_tokenization_basics.py

# Run exercises (try to fill in the TODO sections first!)
python modules/05.5_huggingface_tokenizers/exercises/exercise_01_tokenization_basics.py
```

---

## Learning Path

```
Read 01_tokenization_basics.md
        |
        v
Run example_01_tokenization_basics.py
        |
        v
Try exercise_01_tokenization_basics.py
        |
        v
Read 02_bpe_algorithm.md
        |
        v
Run example_02_bpe_algorithm.py   (implements BPE from scratch!)
        |
        v
...continue for lessons 03, 04, 05...
```

---

## Key Takeaways (Read This First)

1. **Tokenization is lossy** - "unhappiness" becomes ["un", "happi", "ness"].
   The model learns from patterns across millions of such splits.

2. **Vocabulary size is a design choice** - Larger = better coverage, more memory.
   Typical: 30,000 to 50,000 tokens.

3. **BPE is the dominant method** - GPT-2, GPT-4, LLaMA all use BPE variants.

4. **HuggingFace makes it easy** - AutoTokenizer loads any model's tokenizer
   with one line. You rarely need to build from scratch in production.

5. **Training is cheap** - Training a tokenizer on text is fast (no GPU needed).
   It only requires raw text, no labels.

---

*This module was created as part of the "Learn LLM from Scratch" course for .NET developers.*

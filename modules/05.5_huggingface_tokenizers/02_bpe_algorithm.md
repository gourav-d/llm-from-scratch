# Lesson 02 - The BPE Algorithm (Byte Pair Encoding)

---

## What Is BPE?

**BPE** stands for **Byte Pair Encoding**.

It was originally invented in 1994 as a **text compression algorithm**.
The idea: find the most common pair of adjacent bytes, replace them with a new symbol,
repeat until you cannot compress further.

In 2016, researchers adapted BPE for NLP tokenization. Instead of compressing files,
we use it to build a subword vocabulary from a text corpus.

**C# Analogy:**
```csharp
// BPE is like building a compression dictionary from training data.
// Imagine you are LZW-compressing a C# source file:
// The compressor learns that "public " appears a lot, so it assigns it a short code.
// BPE does the same with text: "ing" appears often, so it gets its own token.

// Training phase: analyze corpus, find frequent pairs, build merge table
// Inference phase: apply merge table to tokenize new text
```

---

## The Core Idea

Start with individual characters. Merge the most frequent adjacent pair.
Repeat until vocabulary is the size you want.

**Before BPE (character-level):**
```
"low", "lower", "newest", "widest"
l o w     l o w e r     n e w e s t     w i d e s t
```

**After BPE training (learned merges):**
```
First merge: "e" + "s" -> "es"  (most frequent adjacent pair)
Second merge: "es" + "t" -> "est"
Third merge: "l" + "o" -> "lo"
...
Final: "low", "lo" + "w" + "er", "n" + "ew" + "est", "w" + "id" + "est"
```

---

## Step-by-Step BPE Training Example

Let us trace through BPE on a tiny corpus.

**Input corpus:**
```
"aaabdaaabac"
```

**Step 0 - Start with character vocabulary:**
```
Characters: a, b, c, d
Input:      a a a b d a a a b a c
Pairs:      (a,a), (a,b), (b,d), (d,a), (a,a), (a,b), (b,a), (a,c)
```

**Step 1 - Count all adjacent pairs:**
```
Pair   Count
(a,a)    3     <- most frequent!
(a,b)    2
(b,d)    1
(d,a)    1
(b,a)    1
(a,c)    1
```

**Step 2 - Merge the most frequent pair: (a,a) -> "aa"**
```
Before: a  a  a  b  d  a  a  a  b  a  c
After:  aa a  b  d  aa a  b  a  c
                         ^
                    (merged second "aa" from positions 6-7)
Note: scanning left to right, non-overlapping merges
```

**Step 3 - Count pairs again:**
```
Pair    Count
(aa,a)    2   <- now most frequent
(a,b)     2   <- tied!
(aa,b)    1
(b,d)     1
...
```
*(When tied, pick alphabetically or by first occurrence)*

**Step 4 - Merge (aa,a) -> "aaa":**
```
Before: aa a  b  d  aa a  b  a  c
After:  aaa b  d  aaa b  a  c
```

**Continue until vocabulary size is reached.**

---

## BPE on Real Text (How GPT-2 Was Built)

GPT-2 BPE tokenizer:
- Trained on WebText corpus (Reddit links, ~40GB)
- Target vocabulary: **50,257 tokens**
- Started with: 256 byte-level characters (works for any Unicode via UTF-8 bytes)
- Applied 50,000 merge rules

**ASCII Diagram - BPE Training Flow:**
```
TRAINING PHASE (happens once, on large corpus)
================================================
Raw Corpus (40GB of text)
        |
        | Split into characters / bytes
        v
[h][e][l][l][o][ ][w][o][r][l][d]...
        |
        | Count all adjacent pairs
        v
Pair frequencies: {('e','r'):84231, ('i','n'):76540, ...}
        |
        | Merge top pair -> new token
        v
Merge rule 1: ('e','r') -> 'er'
Merge rule 2: ('i','n') -> 'in'
Merge rule 3: ('er','s') -> 'ers'
...
Merge rule 50000: ('world','s') -> 'worlds'
        |
        v
VOCABULARY (50,257 tokens)  +  MERGE RULES LIST
    Saved to disk as tokenizer files
```

---

## BPE Inference (Tokenizing New Text)

Once trained, applying BPE to new text uses the learned merge rules in order:

```
INFERENCE PHASE (happens every time we tokenize text)
=======================================================
Input: "lower"

Step 0 - Start with characters:
  [l][o][w][e][r]

Step 1 - Apply merge rule 1 if applicable: ('l','o') -> 'lo'?
  If ('l','o') is in merge rules: [lo][w][e][r]

Step 2 - Apply merge rule 2: ('lo','w') -> 'low'?
  If ('lo','w') is in merge rules: [low][e][r]

Step 3 - Apply merge rule 3: ('e','r') -> 'er'?
  If ('e','r') is in merge rules: [low][er]

Step 4 - Apply remaining rules... no more apply.
  Result: ["low", "er"]

Token IDs: [2221, 263]  (looked up in vocabulary)
```

The key insight: **merge rules are applied in training order** (rule 1 first, then rule 2, etc.).
This determinism ensures the same text always tokenizes the same way.

---

## Why "Byte" in Byte Pair Encoding?

Modern BPE (GPT-2 and later) operates on **UTF-8 bytes** rather than characters.

**Problem with character-level BPE:**
- Chinese, Arabic, emoji have thousands of characters
- You would need huge initial vocabulary for all Unicode characters

**Solution - Byte-level BPE:**
- UTF-8 encodes every character as 1-4 bytes
- Bytes only have 256 possible values (0-255)
- Start vocabulary = 256 bytes (covers ALL text in any language!)
- Merge from there

```
ASCII 'A'   = byte 65  -> already in vocabulary
Chinese ''  = bytes [228, 184, 173]  -> three vocabulary entries initially
                                         then BPE may merge them into one token
```

---

## GPT-2 Tokenizer in Practice

Let us see what GPT-2's tokenizer actually produces:

```
Text: "Hello, world!"

GPT-2 tokens:    ["Hello", ",", " world", "!"]
GPT-2 token IDs: [15496, 11, 995, 0]

Text: "unhappiness"
GPT-2 tokens:    ["un", "happiness"]
GPT-2 token IDs: [403, 34140]

Text: "ChatGPT"
GPT-2 tokens:    ["Chat", "G", "PT"]
GPT-2 token IDs: [41, 38, 11571]

Note: spaces are PART of the token!
" world" (with space) is one token, not " " + "world"
This is a GPT-2 convention: spaces attach to the following word.
```

---

## Training BPE vs. Applying BPE

This distinction is critical:

| Phase | What Happens | Frequency |
|-------|-------------|-----------|
| **Training** | Analyze corpus, find merge rules, build vocabulary | Once, before deployment |
| **Inference** | Apply fixed merge rules to new text | Every time text is tokenized |

In production, you **never re-train** the tokenizer. You ship the merge rules file
(`merges.txt`) and the vocabulary file (`vocab.json`) with the model.

**C# Analogy:**
```csharp
// Training = building the compression dictionary:
var compressor = new BpeCompressor();
compressor.Train(hugeCorpus, targetVocabSize: 50257);
compressor.Save("tokenizer_files/");   // save merge rules + vocab

// Inference = applying the dictionary:
var tokenizer = BpeCompressor.Load("tokenizer_files/");
int[] ids = tokenizer.Encode("Hello world");    // fast, deterministic
string text = tokenizer.Decode(ids);             // reversible
```

---

## Advantages and Limitations of BPE

**Advantages:**
- Handles any word (even unseen ones) by splitting into known pieces
- Small vocabulary (30K-50K) works for large corpora
- Deterministic and reversible (you can always decode back to original text)
- Language-agnostic when byte-level BPE is used

**Limitations:**
- Tokenization is sensitive to whitespace (leading space matters!)
- Numbers are split oddly: "12345" -> ["123", "45"] or ["1", "2", "3", "4", "5"]
- Languages with no spaces (Chinese, Japanese) need pre-processing
- Merge rules must be applied in order (sequential, hard to parallelize training)

---

## Merge Rule File Example (Real GPT-2 Format)

The file `merges.txt` for GPT-2 looks like this:
```
#version: 0.2
Ġ t
Ġ a
h e
i n
r e
o n
Ġ the
e r
...
```

Each line is one merge rule: `A B` means "merge token A and token B into AB".
The `Ġ` symbol represents a leading space (it is GPT-2's encoding of " ").

---

## Quiz

**Question 1**
What does BPE start with before any merges have been applied?

A) A vocabulary of full English words  
B) Individual characters or bytes (one token per character/byte)  
C) A vocabulary trained on Wikipedia  
D) Random token assignments  

**Answer: B**
*Explanation: BPE always starts at the character or byte level.*
*It builds up to multi-character tokens through repeated merges.*

---

**Question 2**
How does BPE decide which pair to merge next?

A) It picks the pair that appears in the fewest documents  
B) It picks the pair that has the highest alphabetical order  
C) It picks the most frequently occurring adjacent pair in the corpus  
D) It picks the pair that creates the longest new token  

**Answer: C**
*Explanation: At each step, BPE counts all adjacent pairs and merges the most frequent one.*
*This greedily reduces the total number of tokens needed to represent the corpus.*

---

**Question 3**
Once BPE is trained and merge rules are saved, what happens during tokenization of new text?

A) BPE re-trains on the new text to find optimal splits  
B) The merge rules are applied in the original training order to the new text  
C) The new text is split by spaces only  
D) BPE looks for exact word matches in the vocabulary  

**Answer: B**
*Explanation: BPE inference applies the saved merge rules in order.*
*Rule 1 is checked first, then rule 2, and so on. This is deterministic and fast.*

---

## Summary

- BPE = Byte Pair Encoding, originally a compression algorithm
- Training: count pairs, merge most frequent, repeat until vocab size reached
- Inference: apply merge rules in training order to new text
- GPT-2 uses byte-level BPE with 50,257 vocabulary size and 50,000 merge rules
- Merge rules are saved to disk; inference never re-trains
- BPE handles unknown words by splitting into known subword pieces

**Next:** Lesson 03 - WordPiece and SentencePiece: BERT and T5's approaches

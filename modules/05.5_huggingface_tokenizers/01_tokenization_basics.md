# Lesson 01 - Tokenization Basics

---

## What Is Tokenization?

**Tokenization** is the process of breaking raw text into smaller pieces called **tokens**.
A neural language model cannot read strings. It only reads integers.
Tokenization converts a string into a list of integers that the model processes.

```
Raw Text     "The cat sat."
     |
     | tokenize
     v
Tokens       ["The", " cat", " sat", "."]
     |
     | look up in vocabulary
     v
Token IDs    [464, 3797, 2425, 13]
     |
     | embed
     v
Vectors      [ [0.1, 0.3, ...], [0.9, 0.2, ...], ... ]   <- model input
```

Think of it like this: the model is a function that takes a list of integers.
The tokenizer is the translator between human text and that list of integers.

**C# Analogy:**
```csharp
// In C# you might convert an enum to an int for serialization:
enum Token { Cat = 3797, Sat = 2425, Period = 13 }
int[] tokenIds = text.Split(' ').Select(w => (int)Enum.Parse<Token>(w)).ToArray();
// Tokenization is essentially this, but with 50,000+ possible values.
```

---

## Why Tokenization Is Hard

You might think: "just split on spaces!" Let us try that.

```
"The cat sat."     -> ["The", "cat", "sat."]
```

Problems immediately:
- "sat." and "sat" are different tokens even though the word is the same
- "cat" and "cats" are different tokens (no shared meaning)
- "unhappiness" becomes one token even though "un", "happy", "ness" carry meaning
- For Chinese text ("") spaces do not even exist between words

We need a smarter approach.

---

## Method 1: Word-Level Tokenization

**Idea:** Split on whitespace and punctuation. Each unique word is one token.

```
Corpus: "I like cats. I like dogs. Cats and dogs are pets."
Vocab:  { "I":0, "like":1, "cats":2, ".":3, "dogs":4, "Cats":5, "and":6, "are":7, "pets":8 }
```

**Pros:**
- Simple to understand
- Each token has clear meaning

**Cons (why it fails in practice):**

| Problem | Example |
|---------|---------|
| Huge vocabulary | English has 170,000+ words. Each needs an entry. |
| Unknown words | "ChatGPT" was not in any 2010 corpus. |
| Morphology ignored | "run", "runs", "running", "runner" are 4 separate tokens with no shared information |
| Rare words | Words that appear < 5 times get mapped to [UNK] (unknown) token |
| Memory explosion | Embedding table grows with vocab: 170,000 tokens x 768 dimensions = 1.3 billion floats |

**ASCII Diagram - Word-Level Failure:**
```
"running" -> [running_id]        <- one token, one vector
"runner"  -> [runner_id]         <- completely different token
"run"     -> [run_id]            <- model cannot tell these are related!

With BPE (subword):
"running" -> ["run", "##ning"]   <- shares "run" token with "runner"
"runner"  -> ["run", "##ner"]    <- model LEARNS shared meaning of "run"
```

---

## Method 2: Character-Level Tokenization

**Idea:** Every character is a token. Vocabulary = 26 letters + punctuation + digits = ~100 tokens.

```
"cat"  -> ['c', 'a', 't']  -> [3, 1, 20]
"cats" -> ['c', 'a', 't', 's'] -> [3, 1, 20, 19]
```

**Pros:**
- No unknown words (any text can be represented)
- Tiny vocabulary (< 200 tokens)
- Morphology partially captured ("cat" and "cats" share 'c','a','t')

**Cons (why it fails in practice):**

| Problem | Example |
|---------|---------|
| Very long sequences | "Hello world" = 11 tokens instead of 2 |
| Memory cost | Attention is O(n^2). Longer sequences = much more computation |
| Meaning fragmented | 'c','a','t' has less meaning than "cat" |
| Training difficulty | Model must learn to combine letters into words, then words into concepts |

**ASCII Diagram - Sequence Length Problem:**
```
Word-level:    "The quick brown fox" -> 4 tokens  (short, fast attention)
Char-level:    "The quick brown fox" -> 19 tokens (long, slow attention)
Subword (BPE): "The quick brown fox" -> 5 tokens  (balanced!)
```

---

## Method 3: Subword Tokenization (The Winner)

**Idea:** Common words stay as one token. Rare/long words are split into known pieces.

```
"unhappiness"  ->  ["un", "happi", "ness"]
"running"      ->  ["run", "ning"]
"cat"          ->  ["cat"]              <- common word, stays whole
"ChatGPT"      ->  ["Chat", "G", "PT"] <- unknown word, split into known pieces
```

**Why this is the best of both worlds:**

| Goal | Solution |
|------|---------|
| Handle unknown words | Split into known subword pieces |
| Keep vocab small | 30,000 to 50,000 tokens (not 170,000+) |
| Capture morphology | "run" appears in "running", "runner" as shared piece |
| Short sequences | Common words are single tokens, not character-by-character |

---

## Vocabulary: The Lookup Table

The **vocabulary** (or vocab) is a dictionary that maps every known token to an integer ID.

```
Vocabulary example (tiny, 10 tokens):
{
    "[PAD]"  : 0,    <- padding token (fills empty space)
    "[UNK]"  : 1,    <- unknown token (anything not in vocab)
    "[CLS]"  : 2,    <- start of sequence (BERT)
    "[SEP]"  : 3,    <- separator / end (BERT)
    "un"     : 4,    <- common prefix
    "happi"  : 5,    <- word stem
    "ness"   : 6,    <- common suffix
    "cat"    : 7,
    "the"    : 8,
    "."      : 9,
}
```

**C# Analogy:**
```csharp
// Vocabulary is exactly a Dictionary<string, int>
var vocab = new Dictionary<string, int>
{
    { "[PAD]", 0 },
    { "[UNK]", 1 },
    { "un",    4 },
    { "happi", 5 },
    { "ness",  6 },
    { "cat",   7 },
};

// Encoding is just a lookup
string[] tokens = new[] { "un", "happi", "ness" };
int[] ids = tokens.Select(t => vocab.GetValueOrDefault(t, vocab["[UNK]"])).ToArray();
// Result: [4, 5, 6]
```

---

## Special Tokens

Special tokens are **reserved entries** in the vocabulary with specific roles.
Every model family uses slightly different special tokens.

| Token | ID (typical) | Purpose | Model |
|-------|-------------|---------|-------|
| `[PAD]` | 0 | Padding: fill short sequences to match batch length | BERT |
| `[UNK]` | 1 | Unknown: any word not in vocabulary | BERT, many |
| `[CLS]` | 2 | Classification: start of input sequence | BERT |
| `[SEP]` | 3 | Separator: marks end of sentence or segment | BERT |
| `[MASK]` | 4 | Mask: token replaced for training objective | BERT |
| `<\|endoftext\|>` | 50256 | End of document | GPT-2 |
| `<s>` | 1 | Start of sequence | LLaMA |
| `</s>` | 2 | End of sequence | LLaMA |

**C# Analogy:**
```csharp
// Special tokens are like reserved keywords or sentinel values
// Similar to how string.Empty is reserved, or null has special meaning
// [PAD] = default value in padding context
// [UNK] = catch-all for unrecognized input (like a default case in switch)
```

---

## Token IDs: The Final Numbers

Token IDs are the integer indices into the vocabulary table.

**ASCII Diagram - Full Pipeline:**
```
Input:    "unhappiness is real"

Step 1 - Pre-tokenize (split into rough word units):
          ["unhappiness", " is", " real"]

Step 2 - Subword tokenize (split rare words into vocab pieces):
          ["un", "happi", "ness", " is", " re", "al"]

Step 3 - Look up in vocabulary (token -> ID):
          [ 142,   7831,  2923,   318,  302,  282]

Step 4 - Add special tokens (if model requires):
          [2, 142, 7831, 2923, 318, 302, 282, 3]
           ^                                   ^
          [CLS]                              [SEP]

This integer list is what the model receives.
```

---

## Vocabulary Size: Why It Matters

Different models use different vocabulary sizes:

| Model | Vocab Size | Method |
|-------|-----------|--------|
| GPT-2 | 50,257 | BPE |
| GPT-4 | ~100,000 | BPE (tiktoken) |
| BERT | 30,522 | WordPiece |
| LLaMA 3 | 128,000 | BPE (tiktoken) |
| T5 | 32,000 | SentencePiece |

**Rule of thumb:**
- Larger vocab = fewer tokens per sentence = faster attention
- Larger vocab = bigger embedding table = more memory
- 32,000 - 50,000 is the sweet spot for English
- 100,000+ needed for multilingual models

---

## Quiz

**Question 1**
What is the main problem with word-level tokenization?

A) It creates too few tokens  
B) The vocabulary becomes huge and rare words are marked [UNK]  
C) It only works for English  
D) It cannot handle punctuation  

**Answer: B**
*Explanation: English has 170,000+ words. Word-level tokenization needs an entry for each.*
*Rare words that the model has not seen are replaced by the [UNK] token, losing all meaning.*

---

**Question 2**
What does the vocabulary (vocab) store?

A) The text of the entire training corpus  
B) A mapping from token strings to integer IDs  
C) The weights of the embedding layer  
D) A list of special tokens only  

**Answer: B**
*Explanation: The vocabulary is essentially a Dictionary<string, int> in C# terms.*
*It maps each known token (like "run" or "##ning") to a unique integer index.*

---

**Question 3**
A model receives the input [464, 3797, 2425, 13]. What are these numbers?

A) Word frequencies from training data  
B) Embedding vector dimensions  
C) Token IDs - integers that index into the vocabulary  
D) Layer weights in the neural network  

**Answer: C**
*Explanation: Models receive lists of integers. Each integer is a token ID.*
*The embedding layer converts each ID into a vector. The model never sees raw text.*

---

## Summary

- Tokenization converts raw text into a list of integer token IDs
- Word-level tokenization fails: huge vocabulary, no morphology sharing, unknown words
- Character-level tokenization fails: sequences too long, meaning fragmented
- Subword tokenization (BPE, WordPiece, SentencePiece) is the standard approach
- Vocabulary is a Dictionary<string, int> mapping token -> ID
- Special tokens ([CLS], [SEP], [PAD], [UNK]) have reserved roles
- Token IDs are the actual integers the model receives as input

**Next:** Lesson 02 - The BPE Algorithm: how subword vocabularies are built

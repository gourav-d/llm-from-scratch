# Lesson 04 - The HuggingFace Tokenizers Library

---

## What Is the HuggingFace Tokenizers Library?

HuggingFace (`huggingface.co`) is the dominant open-source ML platform.
They maintain two Python packages relevant to tokenization:

| Package | Purpose |
|---------|---------|
| `tokenizers` | Low-level: build, train, and run tokenizers. Rust backend. Very fast. |
| `transformers` | High-level: load any pretrained model AND its tokenizer with one line. |

For most practical work, you use `transformers` (which calls `tokenizers` internally).
For custom tokenizer training, you use `tokenizers` directly.

**Install:**
```bash
pip install tokenizers transformers
```

**C# Analogy:**
```csharp
// tokenizers  = System.Text.Json (the raw serialization engine)
// transformers = Newtonsoft.Json (higher-level API built on top)

// Both do JSON, but transformers (Newtonsoft) has more convenience methods.
// Similarly, you usually use transformers.AutoTokenizer in Python,
// which calls tokenizers internally.
```

---

## Architecture Overview

```
                    Your Python Code
                          |
                          | import
                          v
                 transformers.AutoTokenizer     <- High-level API
                          |
                          | delegates to
                          v
              tokenizers.Tokenizer              <- Mid-level API
                          |
                          | implemented in
                          v
                    Rust (tokenizers-rs)         <- Low-level, fast
```

The Rust backend makes HuggingFace tokenizers 100-1000x faster than pure Python implementations.

---

## AutoTokenizer: One Class for All Models

`AutoTokenizer` is the Swiss Army knife. Give it a model name, it loads the right tokenizer.

```python
from transformers import AutoTokenizer  # import AutoTokenizer from transformers

# Load BERT's WordPiece tokenizer
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load GPT-2's BPE tokenizer
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

# Load LLaMA 2's SentencePiece tokenizer (requires HuggingFace login for gated models)
# llama_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Same API, different tokenizer underneath!
```

`from_pretrained("model-name")` downloads the tokenizer config files from HuggingFace Hub
and caches them locally at `~/.cache/huggingface/`.

---

## encode(): Text -> Token IDs

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Basic encode: returns a list of token IDs
ids = tokenizer.encode("Hello, world!")
print(ids)    # [101, 7592, 1010, 2088, 999, 102]
#              ^                              ^
#             [CLS]                        [SEP]

# Decode: IDs back to text
text = tokenizer.decode(ids)
print(text)   # "[CLS] hello, world! [SEP]"
```

**What BERT's tokenizer does automatically:**
- Lowercases text (bert-base-UNCASED: always lowercase)
- Adds [CLS] at start (ID = 101)
- Adds [SEP] at end (ID = 102)
- Returns Python list of ints

**C# Analogy:**
```csharp
// encode() is like Serialize, decode() is like Deserialize
var tokenizer = TokenizerFactory.Load("bert-base-uncased");

// Serialize: text -> token IDs
int[] ids = tokenizer.Encode("Hello, world!");

// Deserialize: token IDs -> text
string text = tokenizer.Decode(ids);
```

---

## tokenize(): Text -> Token Strings (Without IDs)

Sometimes you want to see the tokens before converting to IDs:

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# tokenize returns the actual token strings
tokens = tokenizer.tokenize("I was running quickly")
print(tokens)
# ['i', 'was', 'running', 'quickly']
# Note: bert-base-UNCASED lowercases everything

tokenizer2 = AutoTokenizer.from_pretrained("gpt2")
tokens2 = tokenizer2.tokenize("I was running quickly")
print(tokens2)
# ['I', 'Ġwas', 'Ġrunning', 'Ġquickly']
# Note: GPT-2 uses Ġ to represent a leading space
```

---

## convert_tokens_to_ids() and convert_ids_to_tokens()

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Step 1: text to tokens (strings)
tokens = tokenizer.tokenize("unhappiness is real")
print(tokens)   # ['un', '##happiness', 'is', 'real']
#                          ^ ## means continuation

# Step 2: tokens to IDs
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)      # [4895, 9237, 2003, 2613]

# Reverse: IDs to tokens
tokens_back = tokenizer.convert_ids_to_tokens(ids)
print(tokens_back)  # ['un', '##happiness', 'is', 'real']

# And back to text
text = tokenizer.convert_tokens_to_string(tokens_back)
print(text)    # "unhappiness is real"
```

---

## __call__(): The Full Pipeline

The recommended way to tokenize for model input is to call the tokenizer directly.
This returns a dictionary with everything the model needs:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Call the tokenizer like a function (uses __call__ method)
output = tokenizer("Hello, how are you?")

print(output)
# {
#   'input_ids':      [101, 7592, 1010, 2129, 2024, 2017, 1029, 102],
#   'token_type_ids': [0,   0,    0,    0,    0,    0,    0,    0  ],
#   'attention_mask': [1,   1,    1,    1,    1,    1,    1,    1  ]
# }

# input_ids: the token IDs (what the model reads)
# token_type_ids: which sentence each token belongs to (0=first, 1=second)
# attention_mask: 1=real token, 0=padding (explained below)
```

**C# Analogy:**
```csharp
// The output is like a DTO (Data Transfer Object):
public class TokenizerOutput
{
    public int[] InputIds      { get; set; }   // token IDs
    public int[] TokenTypeIds  { get; set; }   // sentence segment IDs
    public int[] AttentionMask { get; set; }   // real vs padding mask
}
TokenizerOutput output = tokenizer.Call("Hello, how are you?");
```

---

## Attention Mask: Real Tokens vs Padding

When processing a batch of sentences, they must all be the same length.
Shorter sentences get padded. The attention mask tells the model which tokens are real.

```
Batch of 3 sentences:
  "Hi"                 -> 3 tokens  (with [CLS] and [SEP])
  "Hello world"        -> 4 tokens
  "I love Python"      -> 5 tokens

After padding to max length (5):
  "Hi"           -> [101, 7632, 102,  0,    0  ]   <- 0 = [PAD] token
  "Hello world"  -> [101, 7592, 2088, 102,  0  ]
  "I love Python"-> [101, 1045, 2293, 21966,102]

Attention mask:
  "Hi"           -> [1,   1,    1,    0,    0  ]   <- 0 = ignore this token
  "Hello world"  -> [1,   1,    1,    1,    0  ]
  "I love Python"-> [1,   1,    1,    1,    1  ]

The model uses the attention mask so it does NOT attend to padding tokens.
```

**ASCII Diagram - Padding and Attention Mask:**
```
Sentence:    [CLS] [Hello] [world]  [PAD]  [PAD]
Attention:     1      1       1       0      0
                                      ^      ^
                             Model ignores these!
                             They are just filler, not real words.
```

---

## Batch Tokenization with Padding and Truncation

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize multiple sentences at once
sentences = [
    "Short sentence.",
    "This is a much longer sentence with more words.",
    "Medium length here.",
]

# padding=True: pad all to the same length
# truncation=True: cut off sequences longer than max_length
# max_length: maximum number of tokens (model's context window limit)
# return_tensors="pt": return PyTorch tensors instead of Python lists
output = tokenizer(
    sentences,
    padding=True,          # pad shorter sequences
    truncation=True,       # truncate sequences longer than max_length
    max_length=20,         # BERT can handle up to 512
    return_tensors="pt",   # "pt" for PyTorch, "tf" for TensorFlow, "np" for NumPy
)

print("input_ids shape:", output['input_ids'].shape)    # (3, 20) - 3 sentences, 20 tokens each
print("attention_mask:")
print(output['attention_mask'])    # 1s for real tokens, 0s for padding
```

---

## Special Tokens: Control and Customization

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Check what special tokens this tokenizer uses
print("Vocab size:", tokenizer.vocab_size)          # 30522 for BERT
print("CLS token:", tokenizer.cls_token)            # "[CLS]"
print("CLS token ID:", tokenizer.cls_token_id)      # 101
print("SEP token:", tokenizer.sep_token)            # "[SEP]"
print("PAD token:", tokenizer.pad_token)            # "[PAD]"
print("PAD token ID:", tokenizer.pad_token_id)      # 0
print("UNK token:", tokenizer.unk_token)            # "[UNK]"
print("MASK token:", tokenizer.mask_token)          # "[MASK]"

# Encode WITHOUT adding special tokens (sometimes useful)
ids_no_special = tokenizer.encode("Hello world", add_special_tokens=False)
print("Without special tokens:", ids_no_special)   # [7592, 2088]

# Encode WITH special tokens (default)
ids_with_special = tokenizer.encode("Hello world", add_special_tokens=True)
print("With special tokens:", ids_with_special)    # [101, 7592, 2088, 102]
```

---

## Word-to-Token Mapping

Sometimes you need to know which token corresponds to which original word.
This is useful for NER (Named Entity Recognition) and similar tasks.

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "unhappiness is real"
encoding = tokenizer(text, return_offsets_mapping=True)

print(encoding['input_ids'])         # [101, 4895, 9237, 2003, 2613, 102]
print(encoding['offset_mapping'])    # [(0,0), (0,2), (2,11), (12,14), (15,19), (0,0)]
#                                      ^ [CLS]         ^ "unhappiness" spans chars 0-11

# word_ids() tells you which word each token came from
word_ids = encoding.word_ids()
print(word_ids)    # [None, 0, 0, 1, 2, None]
#                    ^              ^       ^
#                  [CLS]     "is"=word 1  [SEP]
#                   word 0 = "unhappiness" (spans 2 tokens: indices 1 and 2)
```

---

## Fast vs Slow Tokenizers

HuggingFace has two implementations of most tokenizers:

| Type | Backend | Speed | Extra Features |
|------|---------|-------|----------------|
| Fast tokenizer | Rust (tokenizers library) | Very fast | offset_mapping, word_ids() |
| Slow tokenizer | Pure Python | Slow | Basic encode/decode only |

```python
# AutoTokenizer loads Fast by default
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer))     # BertTokenizerFast

# Load slow version explicitly
from transformers import BertTokenizer
slow_tok = BertTokenizer.from_pretrained("bert-base-uncased")
print(type(slow_tok))      # BertTokenizer (without Fast)

# Always use Fast (default) unless you have a specific reason not to
```

---

## Saving and Loading a Custom Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Save to a local directory
tokenizer.save_pretrained("./my_tokenizer/")
# Creates files: tokenizer_config.json, vocab.txt, tokenizer.json, special_tokens_map.json

# Load it back
from transformers import AutoTokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer/")

# Verify it works
ids = loaded_tokenizer.encode("Hello world")
print(ids)    # same as before: [101, 7592, 2088, 102]
```

---

## Quiz

**Question 1**
What does `tokenizer("some text", return_tensors="pt")` return?

A) A Python string of token strings  
B) A dictionary with "input_ids", "attention_mask", and possibly "token_type_ids" as PyTorch tensors  
C) A single integer token ID  
D) A JSON file saved to disk  

**Answer: B**
*Explanation: Calling a tokenizer returns a BatchEncoding dictionary.*
*With return_tensors="pt", the values are PyTorch tensors.*
*The keys are "input_ids" (token IDs), "attention_mask" (1=real, 0=pad),*
*and "token_type_ids" (sentence segment IDs, for BERT-style models).*

---

**Question 2**
Why is an attention mask needed when tokenizing a batch of sentences?

A) The model needs to know which tokens are special tokens ([CLS], [SEP])  
B) Shorter sentences must be padded to match the longest sentence; the mask tells the model which positions are real tokens versus padding  
C) The attention mask improves tokenization speed  
D) The attention mask prevents the tokenizer from truncating long sentences  

**Answer: B**
*Explanation: Neural networks process fixed-size batches. All sequences must be the same length.*
*Padding tokens (ID=0) are added to short sequences. The attention mask (1=real, 0=pad)*
*tells the self-attention mechanism to ignore padded positions.*

---

**Question 3**
What is the difference between `tokenizer.tokenize()` and `tokenizer.encode()`?

A) tokenize() is faster; encode() is more accurate  
B) tokenize() returns token strings; encode() returns integer token IDs  
C) tokenize() only works for BERT; encode() works for all models  
D) They are identical; different names for the same function  

**Answer: B**
*Explanation: tokenize() returns a list of string tokens (["run", "##ning"]).*
*encode() returns a list of integer IDs ([2448, 6752]).*
*Use tokenize() to inspect what the tokenizer does. Use encode() to prepare model input.*

---

## Summary

- `transformers.AutoTokenizer.from_pretrained("model-name")` loads the right tokenizer
- `tokenizer.encode(text)` -> list of token IDs
- `tokenizer.decode(ids)` -> back to text
- `tokenizer(text, padding=True, truncation=True)` -> full model-ready input dict
- Attention mask: 1 for real tokens, 0 for padding
- Fast tokenizers (Rust backend) are the default and support extra features like offset_mapping
- Save tokenizer with `save_pretrained()`, load back with `from_pretrained()`

**Next:** Lesson 05 - Training a Custom Tokenizer

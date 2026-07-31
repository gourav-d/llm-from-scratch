# Lesson 05 - Training a Custom Tokenizer

---

## When Do You Need a Custom Tokenizer?

Most of the time you should use the tokenizer that came with the pretrained model.
**Do not change it when fine-tuning.**

But there are cases where you must train a new tokenizer from scratch:

| Scenario | Why Custom Tokenizer? |
|----------|----------------------|
| Building a new model from scratch | No existing tokenizer to use |
| Domain-specific text (medical, legal, code) | General tokenizer splits domain terms poorly |
| Low-resource language | General tokenizer has few tokens for that language |
| Non-Latin script (Arabic, Korean, etc.) | General tokenizer may not handle script well |
| Efficiency on specific domain | Custom tokenizer = fewer tokens = faster training |

**Example: Medical domain problem with general tokenizer:**
```
GPT-2 tokenizes: "electrocardiogram"
Result:  ["elect", "ro", "card", "io", "gram"]  <- 5 tokens, awkward split

Custom medical tokenizer (trained on medical text):
Result:  ["electro", "cardio", "gram"]  <- 3 tokens, medically meaningful
```

**C# Analogy:**
```csharp
// It is like training a custom spell-checker dictionary for your domain.
// A general spell-checker knows "cat" but not "cardiomyopathy".
// A medical spell-checker is trained on medical textbooks -> knows the full word.

// General tokenizer = standard English dictionary
// Custom medical tokenizer = medical dictionary built from medical literature
```

---

## Training Data Requirements

Training a tokenizer is different from training a model:

| Aspect | Tokenizer Training | Model Training |
|--------|-------------------|----------------|
| Input | Raw text files | Text + labels (or unsupervised) |
| GPU needed? | No (CPU only!) | Yes (strongly recommended) |
| Time | Minutes | Hours to weeks |
| Memory | Low (streams data) | High (loads batches) |
| Output | vocab.json + merges.txt | model weights (.bin or .safetensors) |

You can train a tokenizer on a **laptop** in **minutes**.

---

## The Training API

HuggingFace `tokenizers` library provides trainers:

| Trainer | Tokenizer Type | Models that use it |
|---------|----------------|-------------------|
| `BpeTrainer` | Byte Pair Encoding | GPT-style |
| `WordPieceTrainer` | WordPiece | BERT-style |
| `UnigramTrainer` | Unigram LM | ALBERT, some T5 variants |

---

## Step-by-Step: Training a BPE Tokenizer

### Step 1 - Prepare Training Text

```python
# Write sample training text to files (one sentence per line)
training_text = [
    "The patient was diagnosed with acute myocardial infarction.",
    "Electrocardiogram showed ST elevation in leads II, III, and aVF.",
    "The cardiologist recommended immediate percutaneous coronary intervention.",
    "Blood pressure was 140/90 mmHg on admission.",
    "Troponin levels were elevated at 2.5 ng/mL.",
]

# Save to file (tokenizer trains from files)
with open("training_corpus.txt", "w") as f:
    for line in training_text:
        f.write(line + "\n")
```

### Step 2 - Choose and Configure the Tokenizer

```python
from tokenizers import Tokenizer                         # base Tokenizer class
from tokenizers.models import BPE                        # BPE model (the core algorithm)
from tokenizers.trainers import BpeTrainer               # trainer: manages the training loop
from tokenizers.pre_tokenizers import Whitespace         # pre-tokenizer: splits on whitespace first

# Create an empty BPE tokenizer (no vocabulary yet)
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))           # [UNK] token for unknown words
# BPE() creates an empty BPE model
# unk_token="[UNK]": any token not in vocab maps to this

# Set the pre-tokenizer: splits raw text before BPE algorithm runs
tokenizer.pre_tokenizer = Whitespace()
# Whitespace: splits on spaces and punctuation
# Result: "Hello, world" -> ["Hello", ",", "world"]
```

### Step 3 - Configure the Trainer

```python
# Create the trainer with settings
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    # special_tokens: list of reserved tokens added to vocabulary FIRST
    # Their IDs will be 0, 1, 2, 3, 4 in this order

    vocab_size=8000,
    # vocab_size: how many tokens the final vocabulary will have
    # 8000 is small (domain-specific corpus)
    # 32000 is typical (general purpose)
    # 50000+ for multilingual

    min_frequency=2,
    # min_frequency: a token must appear at least this many times
    # Filters out typos and very rare domain-specific strings
    # Set to 1 to include everything (useful for tiny corpora)
)
```

### Step 4 - Train

```python
# Train on the corpus files
files = ["training_corpus.txt"]       # list of file paths to train on

tokenizer.train(files, trainer)       # runs BPE training on all files
# This: reads files, counts character pairs, merges 8000-5 times
#       (8000 vocab size minus 5 special tokens = 7995 merge steps)

print("Vocabulary size:", tokenizer.get_vocab_size())   # 8000 (or less if corpus is small)
```

### Step 5 - Test the Tokenizer

```python
# Test encoding
output = tokenizer.encode("electrocardiogram")

print("Tokens:", output.tokens)       # list of token strings
print("IDs:", output.ids)            # list of token IDs
```

### Step 6 - Save the Tokenizer

```python
# Save to directory (creates tokenizer.json)
tokenizer.save("./my_medical_tokenizer.json")

# Load back
from tokenizers import Tokenizer
loaded = Tokenizer.from_file("./my_medical_tokenizer.json")

# Verify
result = loaded.encode("myocardial infarction")
print(result.tokens)    # should produce medically-meaningful subwords
```

---

## Setting Vocabulary Size

Choosing the right vocabulary size:

```
Small domain corpus (<10MB text):     vocab_size = 4,000 - 8,000
Medium domain corpus (10MB-1GB):      vocab_size = 16,000 - 32,000
Large general corpus (10GB+):         vocab_size = 32,000 - 50,000
Multilingual corpus (100GB+):         vocab_size = 100,000+

Too small: many words get split into lots of pieces -> slow training, longer sequences
Too large: many rare tokens rarely learned -> wasted embedding table space
```

**ASCII Diagram - Vocabulary Size Effect:**
```
vocab_size = 1000 (tiny):
"running" -> ["r", "u", "n", "n", "i", "n", "g"]   <- 7 tokens! Too fragmented.

vocab_size = 8000 (small domain):
"running" -> ["run", "##ning"]                       <- 2 tokens. Good!

vocab_size = 50000 (GPT-2):
"running" -> ["running"]                             <- 1 token! Whole word.
```

---

## Training a WordPiece Tokenizer (BERT-Style)

```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

# Create an empty WordPiece tokenizer
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# WordPieceTrainer: same arguments as BpeTrainer
trainer = WordPieceTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=30000,        # typical for WordPiece (BERT uses 30,522)
    min_frequency=2,
    continuing_subword_prefix="##",   # use ## for continuation tokens (BERT convention)
)

# Train and save
tokenizer.train(["training_corpus.txt"], trainer)
tokenizer.save("./my_wordpiece_tokenizer.json")
```

---

## Adding Post-Processing (Special Tokens)

For BERT-style models, you want [CLS] and [SEP] added automatically:

```python
from tokenizers.processors import TemplateProcessing

# Configure post-processor to add [CLS] and [SEP] automatically
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    # single: template for one sentence
    # $A = placeholder for the actual tokens of sentence A

    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    # pair: template for two sentences (e.g., question + context)
    # $B = placeholder for sentence B tokens
    # :1 = assign token_type_id=1 to these tokens (BERT convention)

    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),   # (token string, token ID)
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# Now encoding automatically adds [CLS] and [SEP]
output = tokenizer.encode("Hello world")
print(output.tokens)    # ['[CLS]', 'hello', 'world', '[SEP]']
```

---

## Loading the Custom Tokenizer with transformers

Once saved, you can use your custom tokenizer with HuggingFace `transformers`:

```python
from transformers import PreTrainedTokenizerFast

# Wrap your tokenizers.Tokenizer in a transformers-compatible class
custom_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="./my_medical_tokenizer.json",   # path to saved tokenizer
    unk_token="[UNK]",     # tell transformers which token is [UNK]
    cls_token="[CLS]",     # tell transformers which token is [CLS]
    sep_token="[SEP]",     # tell transformers which token is [SEP]
    pad_token="[PAD]",     # tell transformers which token is [PAD]
    mask_token="[MASK]",   # tell transformers which token is [MASK]
)

# Now you can use it exactly like any transformers tokenizer
output = custom_tokenizer(
    "electrocardiogram",
    padding=True,
    truncation=True,
    return_tensors="pt",
)
print(output['input_ids'])
```

---

## Training from a Generator (Memory-Efficient)

For large corpora that do not fit in memory:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=32000, special_tokens=["[UNK]"])

# Use a generator function instead of loading all text into RAM
def get_training_data():
    """Yields lines of text one at a time. Memory-efficient for large files."""
    file_paths = ["corpus_part1.txt", "corpus_part2.txt", "corpus_part3.txt"]
    for file_path in file_paths:                     # iterate over each file
        with open(file_path, "r", encoding="utf-8") as f:  # open file
            for line in f:                           # iterate over each line
                yield line.strip()                   # yield line without newline

# train_from_iterator: accepts any Python generator/iterator
tokenizer.train_from_iterator(get_training_data(), trainer=trainer)
# This processes one line at a time, never loading the full corpus into RAM.
```

---

## Full Example: Building a Code Tokenizer

A practical example: train a tokenizer specialized for Python code.

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel  # ByteLevel for code (handles all chars)

# ByteLevel pre-tokenizer: best for code because:
# - Code has all kinds of special characters: {, }, (, ), =, +=, ==, !=
# - ByteLevel handles them all without needing whitespace splitting
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
# add_prefix_space=False: do not add a space before the first word

# Sample Python code corpus
sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()
"""

# Save code to file
with open("python_code_corpus.txt", "w") as f:
    f.write(sample_code * 100)    # repeat to create bigger training set

# Train with larger vocab (code has many unique identifiers)
trainer = BpeTrainer(
    vocab_size=16000,
    min_frequency=1,              # include all tokens (corpus is small)
    special_tokens=["<|endoftext|>", "<|pad|>"],
)

tokenizer.train(["python_code_corpus.txt"], trainer)
tokenizer.save("./python_code_tokenizer.json")

print("Code tokenizer vocabulary size:", tokenizer.get_vocab_size())

# Test
result = tokenizer.encode("def fibonacci(n):")
print("Code tokens:", result.tokens)    # should split code meaningfully
```

---

## Quiz

**Question 1**
Why is training a tokenizer different from training a neural network model?

A) Tokenizer training requires labeled data; model training uses unlabeled text  
B) Tokenizer training requires no GPU, takes minutes, and only needs raw text;
   model training requires GPU, hours to weeks, and much more computation  
C) Tokenizer training can only run once; model training can be run repeatedly  
D) Tokenizer training needs more memory than model training  

**Answer: B**
*Explanation: Tokenizer training is statistical (count pairs, merge). It is CPU-bound and fast.*
*Model training involves matrix multiplications with billions of parameters, requiring GPU.*
*You can train a tokenizer on a laptop in minutes from raw text with no labels.*

---

**Question 2**
When should you train a custom tokenizer instead of using an existing one?

A) Every time you fine-tune a model, even on general English text  
B) When your text domain has specialized vocabulary that is poorly handled
   by general tokenizers (medical, legal, code, non-English language)  
C) Whenever your training corpus is larger than 1GB  
D) When you want to use the HuggingFace tokenizers library  

**Answer: B**
*Explanation: Train a custom tokenizer when the domain is specialized.*
*"electrocardiogram" split into 5 general-purpose pieces is inefficient.*
*A custom medical tokenizer trained on medical text learns the full term as one token.*
*For general English fine-tuning, ALWAYS reuse the original model's tokenizer.*

---

**Question 3**
What are the two files that define a BPE tokenizer (the minimum needed to reproduce it)?

A) model.bin and config.yaml  
B) vocab.json (token->ID mapping) and merges.txt (merge rules in order)  
C) tokenizer.py and requirements.txt  
D) training_data.txt and hyperparameters.json  

**Answer: B**
*Explanation: A BPE tokenizer is fully defined by:*
*1. vocab.json: the dictionary mapping every token string to its integer ID*
*2. merges.txt: the ordered list of merge rules (rule 1 applied first, etc.)*
*With these two files, anyone can reproduce the exact same tokenization.*
*HuggingFace also saves a tokenizer.json that combines both into one file.*

---

## Summary

- Train a custom tokenizer when: building a new model, working with specialized text,
  or handling languages poorly covered by general tokenizers
- Training needs only raw text, no labels, no GPU, runs in minutes
- HuggingFace `BpeTrainer` and `WordPieceTrainer` handle the training loop
- Key hyperparameters: `vocab_size` (8K-100K), `min_frequency` (filter rare tokens)
- Save with `tokenizer.save()`, load with `Tokenizer.from_file()`
- Wrap in `PreTrainedTokenizerFast` to use with `transformers` models
- For memory efficiency on large corpora, use `train_from_iterator()` with a generator

---

## Module 05.5 Complete!

You have now learned:
1. Why tokenization matters and how subword methods solve word/character-level problems
2. How BPE builds a vocabulary through iterative pair merging
3. How WordPiece and SentencePiece differ from BPE
4. How to use HuggingFace AutoTokenizer for any pretrained model
5. How to train a custom tokenizer for your domain

**Next Module:** M06 - Training and Fine-tuning (applies everything you know about tokenization!)

"""
=============================================================================
EXAMPLE 01 (PyTorch Version): Tokenization → PyTorch Tensors
=============================================================================

GLOSSARY
---------
Tensor       : PyTorch's version of a NumPy array.
               C# analogy: think of it as float[] or int[] but smarter —
               it can run on GPU and track gradients automatically.

torch.tensor : A function that creates a Tensor from a Python list.
               e.g.  torch.tensor([4, 7, 2])  →  tensor([4, 7, 2])

nn.Embedding : A PyTorch layer that acts like a lookup table.
               Input : token IDs  (integers)
               Output: dense vectors (floats)
               It is the FIRST layer of every transformer / GPT model.

pad_sequence : A utility that takes a list of tensors of DIFFERENT lengths
               and pads the shorter ones so they all have the same length.
               Why needed? Neural networks process batches in parallel —
               all sequences in a batch must be the same size.

dtype        : The number type inside a tensor.
               torch.long  = 64-bit integer  (used for token IDs)
               torch.float = 32-bit decimal  (used for embeddings)

=============================================================================
HOW THIS EXAMPLE CONNECTS TO THE NumPy VERSION
=============================================================================

In example_01_tokenization.py we built:
  CharTokenizer, WordTokenizer, SimpleBPE  (all in pure Python / NumPy)

Those classes still work exactly the same way here.
The NEW thing in this file is what happens AFTER tokenization:

  text  →  token IDs (integers)  →  torch.tensor  →  nn.Embedding  →  vectors

The first arrow (text → IDs) is the tokenizer's job.
Everything from torch.tensor onward is PyTorch.

=============================================================================
"""

import torch                        # the main PyTorch library
import torch.nn as nn               # neural-network building blocks
import re                           # Python's regex module (same as Regex in C#)
from collections import Counter     # counts how often each item appears

print("=" * 65)
print("TOKENIZATION + PyTorch Tensors")
print("=" * 65)

# =============================================================================
# STEP 1: Same tokenizers as before (unchanged from NumPy version)
# =============================================================================

# We copy CharTokenizer and WordTokenizer here so the file is self-contained.
# They produce plain Python lists of integers — same as before.

class CharTokenizer:
    """Character-level tokenizer (identical to NumPy version)."""

    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.char_to_id = {}
        self.id_to_char = {}

    def build_vocab(self, texts):
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token
        for i, char in enumerate(sorted(unique_chars)):
            idx = i + len(special_tokens)
            self.char_to_id[char] = idx
            self.id_to_char[idx] = char

    def encode(self, text, add_special_tokens=True):
        tokens = []
        if add_special_tokens:
            tokens.append(self.char_to_id[self.bos_token])
        for char in text:
            token_id = self.char_to_id.get(char, self.char_to_id[self.unk_token])
            tokens.append(token_id)
        if add_special_tokens:
            tokens.append(self.char_to_id[self.eos_token])
        return tokens

    def decode(self, token_ids):
        special = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        return ''.join(
            self.id_to_char.get(tid, self.unk_token)
            for tid in token_ids
            if self.id_to_char.get(tid) not in special
        )

    @property
    def vocab_size(self):
        return len(self.char_to_id)


class WordTokenizer:
    """Word-level tokenizer (identical to NumPy version)."""

    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.word_to_id = {}
        self.id_to_word = {}

    def _tokenize_text(self, text):
        text = text.lower()
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        return text.split()

    def build_vocab(self, texts, max_vocab_size=None):
        word_counts = Counter()
        for text in texts:
            word_counts.update(self._tokenize_text(text))
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        most_common = (word_counts.most_common(max_vocab_size - len(special_tokens))
                       if max_vocab_size else word_counts.most_common())
        for i, (word, _) in enumerate(most_common):
            idx = i + len(special_tokens)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

    def encode(self, text, add_special_tokens=True):
        words = self._tokenize_text(text)
        tokens = []
        if add_special_tokens:
            tokens.append(self.word_to_id[self.bos_token])
        for word in words:
            tokens.append(self.word_to_id.get(word, self.word_to_id[self.unk_token]))
        if add_special_tokens:
            tokens.append(self.word_to_id[self.eos_token])
        return tokens

    def decode(self, token_ids):
        special = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        return ' '.join(
            self.id_to_word.get(tid, self.unk_token)
            for tid in token_ids
            if self.id_to_word.get(tid) not in special
        )

    @property
    def vocab_size(self):
        return len(self.word_to_id)


# =============================================================================
# STEP 2: Tokenize some text — still returns plain Python lists
# =============================================================================

print("\n--- STEP 2: Tokenize text (same as NumPy version) ---")

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a great programming language.",
    "Machine learning models need data to train.",
]

tokenizer = WordTokenizer()
tokenizer.build_vocab(corpus)

sentences = [
    "The quick brown fox.",
    "Python is great.",
    "Machine learning needs data.",
]

# encode() still returns a plain Python list of ints (no PyTorch yet)
encoded_sentences = [tokenizer.encode(s) for s in sentences]

print(f"Vocabulary size : {tokenizer.vocab_size}")
for sent, ids in zip(sentences, encoded_sentences):
    print(f"  '{sent}'  ->  {ids}")

# =============================================================================
# STEP 3: Convert token ID lists to PyTorch Tensors
# =============================================================================

print("\n--- STEP 3: Convert to PyTorch Tensors ---")
print("""
Why do we need tensors?
  The tokenizer output is a Python list like [2, 5, 3, 8].
  PyTorch layers (like nn.Embedding) ONLY accept tensors, not plain lists.
  Conversion is one line: torch.tensor(list, dtype=torch.long)

  dtype=torch.long  →  64-bit integer
  This is required for token IDs because nn.Embedding expects integers.

  C# analogy: casting int[] to a typed array that the GPU understands.
""")

# torch.tensor() converts a Python list to a PyTorch Tensor.
# dtype=torch.long means each value is stored as a 64-bit integer.
# Token IDs must be integers (you can't look up row 2.5 in a table!).
tensor_sent1 = torch.tensor(encoded_sentences[0], dtype=torch.long)
tensor_sent2 = torch.tensor(encoded_sentences[1], dtype=torch.long)
tensor_sent3 = torch.tensor(encoded_sentences[2], dtype=torch.long)

print(f"Sentence 1 as tensor : {tensor_sent1}")
print(f"  shape  : {tensor_sent1.shape}")   # shape = how many tokens
print(f"  dtype  : {tensor_sent1.dtype}")   # should be torch.int64 (=torch.long)

# =============================================================================
# STEP 4: Padding — make all sequences the same length
# =============================================================================

print("\n--- STEP 4: Padding sequences to equal length ---")
print("""
Problem:
  Sentence 1 has 6 tokens, Sentence 2 has 5, Sentence 3 has 6.
  Neural networks process sequences in BATCHES.
  A batch must be a rectangular 2-D tensor: (batch_size, seq_len).
  All rows must have the SAME length.

Solution: PADDING
  Append <PAD> (token ID = 0) to the shorter sequences until all are equal.

  C# analogy: like string.PadRight(maxLength, ' ') but for arrays of numbers.
""")

# pad_sequence pads a list of 1-D tensors so they all reach the longest length.
# batch_first=True  → output shape is (batch_size, seq_len)
#                     (default is seq_len, batch_size — less intuitive)
# padding_value=0   → fill with token ID 0, which is <PAD>
from torch.nn.utils.rnn import pad_sequence   # utility inside PyTorch

padded = pad_sequence(
    [tensor_sent1, tensor_sent2, tensor_sent3],
    batch_first=True,        # we want shape (3, seq_len), not (seq_len, 3)
    padding_value=0          # 0 = <PAD> token ID in our vocabulary
)

print(f"Original lengths : {[len(s) for s in encoded_sentences]}")
print(f"After padding shape : {padded.shape}")
print("Padded batch:")
print(padded)
print("\nRows are sentences. Trailing 0s are <PAD> tokens.")

# =============================================================================
# STEP 5: nn.Embedding — the FIRST layer of every transformer
# =============================================================================

print("\n--- STEP 5: nn.Embedding — from token IDs to vectors ---")
print("""
nn.Embedding is a big learnable lookup table.

  Input : a tensor of token IDs, e.g. tensor([[2, 5, 3, 0], ...])
  Output: a tensor of vectors,   e.g. shape (batch, seq_len, embed_dim)

  Internally it is just a 2-D matrix:
      shape = (vocab_size, embed_dim)
  Each row is the embedding vector for one token.

  When you pass token ID 5, PyTorch returns row 5 of the matrix.
  These rows are LEARNED during training via backpropagation.

  C# analogy: Dictionary<int, float[]> but as a 2-D array
              where you do table[token_id] to get the vector.
""")

embed_dim = 8   # each token will be represented as a vector of 8 numbers

# nn.Embedding(num_embeddings, embedding_dim)
#   num_embeddings = number of rows (one per token in vocabulary)
#   embedding_dim  = number of columns (size of each vector)
embedding_layer = nn.Embedding(
    num_embeddings=tokenizer.vocab_size,  # one row per token
    embedding_dim=embed_dim,              # each token = 8 numbers
    padding_idx=0                         # token 0 (<PAD>) always stays all zeros
)

print(f"Embedding table shape : {embedding_layer.weight.shape}")
print(f"  rows    = vocab_size  = {tokenizer.vocab_size}")
print(f"  columns = embed_dim   = {embed_dim}")

# Pass the padded batch through the embedding layer.
# PyTorch automatically looks up the vector for each token ID.
# No loop needed — it processes the entire batch at once!
embedded = embedding_layer(padded)   # shape: (3, seq_len, 8)

print(f"\nInput  shape : {padded.shape}    (batch=3, seq_len={padded.shape[1]})")
print(f"Output shape : {embedded.shape}  (batch=3, seq_len={padded.shape[1]}, embed_dim=8)")
print("\nFirst sentence, first token embedding vector:")
print(embedded[0, 0])   # sentence 0, token position 0, all 8 dimensions
print("\nPAD token embedding (should be all zeros):")
print(embedded[0, -1])  # the last position of sentence 0 might be PAD

# =============================================================================
# STEP 6: Attention mask — tell the model to IGNORE padding
# =============================================================================

print("\n--- STEP 6: Attention Mask ---")
print("""
Problem:
  We padded short sentences with 0s (<PAD>).
  But the model should NOT try to learn from PAD tokens.
  PAD tokens are just filler — they carry no meaning.

Solution: ATTENTION MASK
  A tensor of 1s and 0s:
    1 = "real token, pay attention to this"
    0 = "padding token, IGNORE this"

  The transformer uses this mask during the attention step.

  C# analogy: like a bool[] IsValid where true=process, false=skip.
""")

# Create attention mask: 1 where token != 0 (PAD), else 0.
# padded != 0  →  returns True/False tensor
# .long()      →  converts True→1, False→0
attention_mask = (padded != 0).long()

print("Padded token IDs:")
print(padded)
print("\nAttention mask (1=real token, 0=PAD):")
print(attention_mask)

# =============================================================================
# STEP 7: Summary diagram
# =============================================================================

print("\n" + "=" * 65)
print("SUMMARY: Text → Token IDs → Tensors → Embeddings")
print("=" * 65)
print("""
  "Python is great."          <- raw text
       |
       | tokenizer.encode()
       v
  [2, 12, 7, 3]               <- token IDs (plain Python list)
       |
       | torch.tensor(..., dtype=torch.long)
       v
  tensor([2, 12, 7, 3])       <- PyTorch tensor
       |
       | pad_sequence()        (makes all sequences same length)
       v
  tensor([[2, 12,  7, 3, 0],
          [2,  5, 13, 8, 0],   <- padded batch  (0 = PAD)
          [2,  9,  4, 6, 3]])
       |
       | nn.Embedding()        (lookup: each ID -> vector of 8 floats)
       v
  tensor of shape (3, 5, 8)   <- batch x seq_len x embed_dim
  (one 8-dimensional vector per token per sentence)
       |
       v
  Transformer / GPT layers     <- this is where modules 04 and beyond come in!

Key PyTorch tools used:
  torch.tensor()         converts Python list → Tensor
  pad_sequence()         pads sequences to equal length in a batch
  nn.Embedding()         lookup table: token ID → float vector
  attention_mask         tells the model which tokens are real vs. padding
""")

print("=" * 65)
print("Run example_02_embeddings_pytorch.py next!")
print("=" * 65)

"""
Module 05.5 - HuggingFace Tokenizers
Example 04: Using the HuggingFace Tokenizers Library

GLOSSARY
--------
AutoTokenizer    : The "magic" class that loads any model's tokenizer automatically.
                   Given a model name, it figures out the right tokenizer class.
                   C# analogy: like an abstract factory pattern (TokenizerFactory.Create(modelName))

encode()         : Convert text -> list of token IDs.
                   C# analogy: JsonSerializer.Serialize()

decode()         : Convert list of token IDs -> text.
                   C# analogy: JsonSerializer.Deserialize()

BatchEncoding    : The dictionary returned by calling a tokenizer.
                   Contains: input_ids, attention_mask, token_type_ids.
                   C# analogy: a DTO (Data Transfer Object) with three arrays.

input_ids        : The token IDs. What the model actually reads (integers, not strings).

attention_mask   : 1 for real tokens, 0 for padding tokens.
                   Tells model which positions to attend to and which to ignore.

padding          : Adding [PAD] tokens to make all sequences the same length.
                   Needed because neural networks process fixed-size batches.
                   C# analogy: Array.Resize() to fill shorter arrays with defaults.

truncation       : Cutting sequences that are too long (longer than max_length).
                   Needed because model has a fixed context window (e.g., 512 tokens).

return_tensors   : Format of output. "pt"=PyTorch tensor, "np"=NumPy array, None=Python list.

This example shows HuggingFace library usage WITH a pure Python fallback.
If the library is not installed, we use a minimal simulation.
"""

print("=" * 60)
print("Example 04: HuggingFace Tokenizers Library")
print("=" * 60)
print()

# ---- Try to import HuggingFace libraries ----
# This is the recommended pattern for optional dependencies
try:
    from transformers import AutoTokenizer    # high-level tokenizer API
    HF_AVAILABLE = True                       # library is installed
    print("[OK] transformers library available")
except ImportError:
    HF_AVAILABLE = False                      # library is NOT installed
    print("[INFO] transformers not installed. Using pure Python simulation.")
    print("[INFO] Install with: pip install transformers")

print()


# ============================================================
# PART A: Pure Python Simulation (Always Runs)
# ============================================================

print("PART A: Simulated Tokenizer API (Pure Python)")
print("-" * 50)
print()

class SimpleTokenizer:
    """
    A minimal tokenizer that simulates the HuggingFace AutoTokenizer API.

    This is for learning purposes only - it shows what the real tokenizer does
    using Python data structures, without any ML library.

    Real AutoTokenizer: loads pretrained BPE/WordPiece rules from disk.
    This simulator: uses a hand-crafted vocabulary for demonstration.

    C# analogy:
      Like a minimal ITokenizer mock used in unit tests:
      public class MockTokenizer : ITokenizer { ... }
    """

    def __init__(self):
        """Initialize with a small hand-crafted vocabulary."""

        # The vocabulary: maps token strings to integer IDs
        # Real BERT vocab has 30,522 entries loaded from vocab.txt
        # We have a tiny demo vocabulary
        self._vocab = {
            "[PAD]"    : 0,        # padding token - ID 0 is always PAD in BERT
            "[UNK]"    : 100,      # unknown token
            "[CLS]"    : 101,      # classification token (start of input)
            "[SEP]"    : 102,      # separator token (end of segment)
            "[MASK]"   : 103,      # mask token (for masked language modeling)

            # Common words (simulate a subset of BERT's vocab)
            "hello"    : 7592,
            "world"    : 2088,
            "the"      : 1996,
            "cat"      : 4937,
            "dog"      : 3899,
            "run"      : 2448,
            "##ning"   : 6752,     # ## = continuation token (part of same word)
            "##s"      : 1116,     # continuation: e.g., "cat" + "##s" = "cats"
            "quick"    : 4248,
            "##ly"     : 2135,
            "python"   : 18750,
            "is"       : 2003,
            "great"    : 2307,
            "i"        : 1045,
            "love"     : 2293,
            "."        : 1012,
            ","        : 1010,
            "!"        : 999,
            "un"       : 4895,     # prefix: "un-happy"
            "##happy"  : 8378,
        }

        # Reverse vocabulary: maps IDs back to token strings
        # In C#: vocab.ToDictionary(kvp => kvp.Value, kvp => kvp.Key)
        self._id_to_token = {v: k for k, v in self._vocab.items()}

        # Special token IDs (easy access)
        self.cls_token_id = self._vocab["[CLS]"]    # 101
        self.sep_token_id = self._vocab["[SEP]"]    # 102
        self.pad_token_id = self._vocab["[PAD]"]    # 0
        self.unk_token_id = self._vocab["[UNK]"]    # 100

        # Special token strings
        self.cls_token  = "[CLS]"
        self.sep_token  = "[SEP]"
        self.pad_token  = "[PAD]"
        self.unk_token  = "[UNK]"
        self.mask_token = "[MASK]"

        # Vocabulary size
        self.vocab_size = len(self._vocab)          # number of entries in vocab


    def tokenize(self, text):
        """
        Convert text to a list of token strings (without adding special tokens).
        This is the tokenize step - returns strings, not IDs.

        C# analogy: text.Split(' ').Select(w => FindTokenInVocab(w)).ToList()

        Args:
            text (str): Input text

        Returns:
            list[str]: Token strings (e.g., ["hello", ",", "world"])
        """
        words = text.lower().split()        # lowercase and split on whitespace
        tokens = []                         # collect token strings

        for word in words:
            # Remove trailing punctuation and handle it separately
            punct = ""
            if word and word[-1] in ".,!?;:":   # check if last char is punctuation
                punct = word[-1]                  # save the punctuation
                word = word[:-1]                  # remove it from word

            # Check whole word
            if word in self._vocab:
                tokens.append(word)               # word is in vocab as-is

            # Try "un" prefix split
            elif word.startswith("un") and "##" + word[2:] not in self._vocab:
                tokens.append("un")               # "un" prefix
                suffix = word[2:]                 # rest of word
                if "##" + suffix in self._vocab:
                    tokens.append("##" + suffix)  # look for continuation piece
                elif suffix in self._vocab:
                    tokens.append(suffix)
                else:
                    tokens.append(self.unk_token)  # give up -> [UNK]

            # Try common word + "##s" suffix
            elif word.endswith("s") and word[:-1] in self._vocab:
                tokens.append(word[:-1])          # stem without 's'
                tokens.append("##s")              # continuation suffix

            # Try word + "##ning" or "##ly"
            elif word.endswith("ning") and word[:-4] in self._vocab:
                tokens.append(word[:-4])          # stem
                tokens.append("##ning")
            elif word.endswith("ly") and word[:-2] in self._vocab:
                tokens.append(word[:-2])          # stem
                tokens.append("##ly")

            else:
                tokens.append(self.unk_token)      # unknown word -> [UNK]

            # Add punctuation as separate token
            if punct and punct in self._vocab:
                tokens.append(punct)

        return tokens                              # return list of token strings


    def convert_tokens_to_ids(self, tokens):
        """
        Convert a list of token strings to a list of integer IDs.

        C# analogy: tokens.Select(t => vocab.GetValueOrDefault(t, unknownId)).ToList()

        Args:
            tokens (list[str]): Token strings

        Returns:
            list[int]: Token IDs
        """
        return [self._vocab.get(t, self.unk_token_id) for t in tokens]
        # .get(key, default): return value if key exists, else return default
        # unk_token_id is the fallback for unknown tokens


    def encode(self, text, add_special_tokens=True):
        """
        Convert text directly to token IDs (combines tokenize + convert to IDs).
        Also adds [CLS] and [SEP] by default (add_special_tokens=True).

        C# analogy: JsonSerializer.Serialize(text) -> byte array -> int[]

        Args:
            text (str): Input text
            add_special_tokens (bool): Whether to add [CLS] and [SEP]

        Returns:
            list[int]: Token IDs
        """
        tokens = self.tokenize(text)                    # get token strings
        ids = self.convert_tokens_to_ids(tokens)        # convert to IDs

        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
            # Prepend [CLS]=101 and append [SEP]=102

        return ids                                       # return list of ints


    def decode(self, ids, skip_special_tokens=True):
        """
        Convert token IDs back to text.

        C# analogy: JsonSerializer.Deserialize<string>(tokenIds)

        Args:
            ids (list[int]): Token IDs to decode
            skip_special_tokens (bool): Whether to remove [CLS], [SEP], [PAD]

        Returns:
            str: Reconstructed text
        """
        tokens = []                                      # collect token strings
        special_ids = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        # set of IDs we may want to skip

        for token_id in ids:                             # loop over each ID
            if skip_special_tokens and token_id in special_ids:
                continue                                  # skip this special token

            token = self._id_to_token.get(token_id, "[UNK]")  # look up token string
            tokens.append(token)                          # add to list

        # Join tokens into text (remove ## and add spaces appropriately)
        text = ""
        for token in tokens:
            if token.startswith("##"):                    # continuation token
                text += token[2:]                         # attach directly (no space)
            elif text:                                    # not the first token
                text += " " + token                       # add space before
            else:
                text += token                             # first token, no space

        return text                                       # return reconstructed string


    def __call__(self, text_or_list, padding=False, truncation=False,
                 max_length=512, add_special_tokens=True):
        """
        Main tokenizer call. Handles single text or list of texts.
        Returns a dictionary with input_ids and attention_mask.

        C# analogy:
          TokenizerOutput tokenizer.Call(string text) {
            return new TokenizerOutput { InputIds = ..., AttentionMask = ... };
          }

        Args:
            text_or_list: Single string or list of strings
            padding (bool): Pad all sequences to the same length
            truncation (bool): Truncate sequences longer than max_length
            max_length (int): Maximum sequence length
            add_special_tokens (bool): Add [CLS] and [SEP]

        Returns:
            dict: {"input_ids": [...], "attention_mask": [...]}
        """
        # Normalize input: always work with a list
        if isinstance(text_or_list, str):               # isinstance() = is operator in C#
            texts = [text_or_list]                      # wrap single string in list
            single = True                               # flag: we got one string
        else:
            texts = text_or_list                        # already a list
            single = False

        # Encode each text to get token IDs
        all_ids = []
        for text in texts:
            ids = self.encode(text, add_special_tokens=add_special_tokens)

            if truncation and len(ids) > max_length:    # cut off if too long
                ids = ids[:max_length - 1] + [self.sep_token_id]  # keep [SEP] at end

            all_ids.append(ids)                         # add to list

        # Determine max length in this batch
        if padding:
            batch_max_len = max(len(ids) for ids in all_ids)   # longest sequence
        else:
            batch_max_len = None                                 # no padding needed

        # Build output with padding and attention masks
        padded_ids = []
        attention_masks = []

        for ids in all_ids:
            # Create attention mask BEFORE padding (1 for real tokens)
            mask = [1] * len(ids)                       # 1 = attend to this token

            if padding and batch_max_len is not None:
                pad_len = batch_max_len - len(ids)      # how much padding needed
                ids = ids + [self.pad_token_id] * pad_len    # add PAD tokens
                mask = mask + [0] * pad_len                  # 0 = ignore PAD tokens

            padded_ids.append(ids)
            attention_masks.append(mask)

        # If input was a single string, unwrap the lists
        if single:
            result = {
                "input_ids"      : padded_ids[0],       # [0] = first (only) item
                "attention_mask" : attention_masks[0],
            }
        else:
            result = {
                "input_ids"      : padded_ids,           # list of lists
                "attention_mask" : attention_masks,
            }

        return result                                    # return the dictionary


# ============================================================
# PART B: Demo the Simulated Tokenizer
# ============================================================

print("Creating our simulated tokenizer...")
tokenizer = SimpleTokenizer()                           # create our simulated tokenizer

print(f"  Vocab size: {tokenizer.vocab_size} tokens")
print()

# Demo 1: tokenize()
print("--- tokenize() ---")
text = "the quick cat is running."
tokens = tokenizer.tokenize(text)
print(f"  Input:  '{text}'")
print(f"  Tokens: {tokens}")
print()

# Demo 2: convert_tokens_to_ids()
print("--- convert_tokens_to_ids() ---")
ids_from_tokens = tokenizer.convert_tokens_to_ids(tokens)
print(f"  Tokens: {tokens}")
print(f"  IDs:    {ids_from_tokens}")
print()

# Demo 3: encode() (with and without special tokens)
print("--- encode() ---")
text2 = "i love python"
ids_with = tokenizer.encode(text2, add_special_tokens=True)
ids_without = tokenizer.encode(text2, add_special_tokens=False)
print(f"  Text: '{text2}'")
print(f"  With special tokens:    {ids_with}")
print(f"    [CLS]=101 at start, [SEP]=102 at end")
print(f"  Without special tokens: {ids_without}")
print()

# Demo 4: decode()
print("--- decode() ---")
decoded = tokenizer.decode(ids_with, skip_special_tokens=True)
print(f"  IDs:     {ids_with}")
print(f"  Decoded: '{decoded}'")
print()

# Demo 5: __call__() - main tokenizer interface
print("--- __call__() - full pipeline ---")
single_output = tokenizer("hello world", padding=False)
print(f"  Single text output:")
print(f"    input_ids:      {single_output['input_ids']}")
print(f"    attention_mask: {single_output['attention_mask']}")
print()

# Demo 6: Batch processing with padding
print("--- Batch Tokenization with Padding ---")
sentences = [
    "hello world",
    "the quick cat",
    "i love python is great",
]

batch_output = tokenizer(sentences, padding=True, truncation=True, max_length=20)
print("  Batch (3 sentences, padded to same length):")
print()
for i, (sent, ids, mask) in enumerate(zip(sentences,
                                          batch_output["input_ids"],
                                          batch_output["attention_mask"])):
    print(f"  Sentence {i+1}: '{sent}'")
    print(f"    input_ids:      {ids}")
    print(f"    attention_mask: {mask}")
    real_count = sum(mask)          # count the 1s (real tokens)
    pad_count  = mask.count(0)      # count the 0s (padding)
    print(f"    Real tokens: {real_count}, Padding: {pad_count}")
    print()


# ============================================================
# PART C: Real HuggingFace Tokenizer (If Available)
# ============================================================

print("PART C: Real HuggingFace AutoTokenizer")
print("-" * 50)
print()

if HF_AVAILABLE:
    print("Loading BERT tokenizer from HuggingFace...")
    real_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print(f"[OK] Loaded! Vocab size: {real_tokenizer.vocab_size}")
    print()

    # Repeat the same demos with the real tokenizer
    text_demo = "I love Python programming"

    print(f"Text: '{text_demo}'")
    print()

    # tokenize()
    real_tokens = real_tokenizer.tokenize(text_demo)
    print(f"  tokenize()  : {real_tokens}")

    # encode()
    real_ids = real_tokenizer.encode(text_demo)
    print(f"  encode()    : {real_ids}")

    # decode()
    real_decoded = real_tokenizer.decode(real_ids, skip_special_tokens=True)
    print(f"  decode()    : '{real_decoded}'")

    # __call__() batch
    batch_texts = [
        "Short text.",
        "This is a much longer text with more words and tokens.",
    ]
    real_batch = real_tokenizer(
        batch_texts,
        padding=True,           # pad shorter to match longer
        truncation=True,        # cut if longer than max_length
        max_length=30,          # maximum 30 tokens
    )
    print()
    print("  Batch encoding with padding=True, truncation=True, max_length=30:")
    print(f"  input_ids shapes: {[len(row) for row in real_batch['input_ids']]} tokens each")
    print(f"  Seq 1 input_ids:      {real_batch['input_ids'][0]}")
    print(f"  Seq 1 attention_mask: {real_batch['attention_mask'][0]}")
    print(f"  Seq 2 input_ids:      {real_batch['input_ids'][1]}")
    print(f"  Seq 2 attention_mask: {real_batch['attention_mask'][1]}")

    # Show special token properties
    print()
    print("  Special token info:")
    print(f"    cls_token:  '{real_tokenizer.cls_token}'  ID={real_tokenizer.cls_token_id}")
    print(f"    sep_token:  '{real_tokenizer.sep_token}'  ID={real_tokenizer.sep_token_id}")
    print(f"    pad_token:  '{real_tokenizer.pad_token}'  ID={real_tokenizer.pad_token_id}")
    print(f"    unk_token:  '{real_tokenizer.unk_token}'  ID={real_tokenizer.unk_token_id}")
    print(f"    mask_token: '{real_tokenizer.mask_token}' ID={real_tokenizer.mask_token_id}")

else:
    print("(Skipping - transformers library not installed)")
    print()
    print("The simulated tokenizer above shows the SAME API as the real one.")
    print("Install: pip install transformers")
    print("Then: from transformers import AutoTokenizer")
    print("      tok = AutoTokenizer.from_pretrained('bert-base-uncased')")


print()
print("=" * 60)
print("Example 04 Complete!")
print()
print("Key Takeaways:")
print("  1. tokenize() -> token strings, encode() -> token IDs, decode() -> text")
print("  2. __call__() returns a dict: input_ids, attention_mask, (token_type_ids)")
print("  3. padding=True makes all sequences the same length (adds [PAD])")
print("  4. attention_mask: 1 for real, 0 for padding")
print("  5. truncation=True cuts sequences longer than max_length")
print("  6. AutoTokenizer.from_pretrained() loads the right tokenizer for any model")
print("=" * 60)

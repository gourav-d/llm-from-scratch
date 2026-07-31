"""
Module 05.5 - HuggingFace Tokenizers
Exercise 04: HuggingFace Tokenizers Library

GLOSSARY
--------
AutoTokenizer    : The "smart" tokenizer loader. Given a model name, it picks
                   the right tokenizer class automatically.
                   C# analogy: like an abstract factory pattern.

encode()         : text -> list[int]. Converts string to token IDs.
                   C# analogy: JsonSerializer.Serialize()

decode()         : list[int] -> text. Converts token IDs back to string.
                   C# analogy: JsonSerializer.Deserialize()

BatchEncoding    : Return type of calling a tokenizer. A dict with:
                   "input_ids": list of token ID lists
                   "attention_mask": list of 1/0 lists
                   C# analogy: a DTO (Data Transfer Object).

padding          : Filling shorter sequences with [PAD] tokens so all sequences
                   in a batch have the same length.
                   Required for batch processing in neural networks.

truncation       : Cutting off sequences that exceed max_length.
                   Necessary because models have a fixed context window.

attention_mask   : 1 for real tokens, 0 for padding tokens.
                   The self-attention mechanism uses this to ignore padding.
                   C# analogy: bool[] mask = ids.Select(id => id != PAD_ID).ToArray()

This exercise uses a simulated tokenizer that matches the HuggingFace API.
If HuggingFace is installed, it also shows real results for comparison.
"""

print("=" * 60)
print("Exercise 04: HuggingFace Tokenizers API")
print("=" * 60)
print()

# Try to import HuggingFace
try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
    print("[OK] transformers library available")
except ImportError:
    HF_AVAILABLE = False
    print("[INFO] transformers not installed. Using simulation.")

print()


# ============================================================
# Simulated Tokenizer (Mini HuggingFace API)
# (This is the same as Example 04 but used here as the test target)
# ============================================================

class SimulatedTokenizer:
    """A minimal tokenizer that simulates the HuggingFace API for exercises."""

    def __init__(self):
        self._vocab = {
            "[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103,
            "hello": 7592, "world": 2088, "the": 1996, "cat": 4937,
            "dog": 3899, "run": 2448, "##ning": 6752, "##s": 1116,
            "quick": 4248, "##ly": 2135, "python": 18750, "is": 2003,
            "great": 2307, "i": 1045, "love": 2293, ".": 1012, ",": 1010, "!": 999,
            "fast": 4019, "slow": 4808, "cats": 8870, "dogs": 4516,
            "machine": 3698, "learn": 4553, "##ing": 3995,
            "natural": 3019, "language": 2653,
        }
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self.cls_token_id = 101; self.sep_token_id = 102
        self.pad_token_id = 0;   self.unk_token_id = 100
        self.cls_token = "[CLS]"; self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"; self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"; self.mask_token_id = 103
        self.vocab_size = len(self._vocab)

    def tokenize(self, text):
        words = text.lower().split()
        tokens = []
        for word in words:
            punct = ""
            if word and word[-1] in ".,!?":
                punct = word[-1]; word = word[:-1]
            if word in self._vocab:
                tokens.append(word)
            elif word.endswith("ning") and word[:-4] in self._vocab:
                tokens.append(word[:-4]); tokens.append("##ning")
            elif word.endswith("ing") and word[:-3] in self._vocab:
                tokens.append(word[:-3]); tokens.append("##ing")
            elif word.endswith("ly") and word[:-2] in self._vocab:
                tokens.append(word[:-2]); tokens.append("##ly")
            elif word.endswith("s") and word[:-1] in self._vocab:
                tokens.append(word[:-1]); tokens.append("##s")
            else:
                tokens.append(self.unk_token)
            if punct and punct in self._vocab:
                tokens.append(punct)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self._vocab.get(t, self.unk_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self._id_to_token.get(i, "[UNK]") for i in ids]

    def encode(self, text, add_special_tokens=True):
        ids = self.convert_tokens_to_ids(self.tokenize(text))
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        skip_ids = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        tokens = [self._id_to_token.get(i, "[UNK]") for i in ids
                  if not (skip_special_tokens and i in skip_ids)]
        text = ""
        for tok in tokens:
            text = text + (tok[2:] if tok.startswith("##") else
                           (" " + tok if text else tok))
        return text

    def __call__(self, text_or_list, padding=False, truncation=False,
                 max_length=512, add_special_tokens=True):
        single = isinstance(text_or_list, str)
        texts = [text_or_list] if single else text_or_list
        all_ids = []
        for t in texts:
            ids = self.encode(t, add_special_tokens=add_special_tokens)
            if truncation and len(ids) > max_length:
                ids = ids[:max_length-1] + [self.sep_token_id]
            all_ids.append(ids)
        max_len = max(len(i) for i in all_ids) if padding else None
        padded, masks = [], []
        for ids in all_ids:
            mask = [1] * len(ids)
            if padding and max_len:
                pad_n = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_n
                mask = mask + [0] * pad_n
            padded.append(ids); masks.append(mask)
        result = {"input_ids": padded[0] if single else padded,
                  "attention_mask": masks[0] if single else masks}
        return result


# ============================================================
#  EXERCISE 1
#  Topic: Tokenize text and inspect tokens
#
#  Background:
#    tokenize() returns token strings (["run", "##ning"]).
#    encode() returns token IDs ([101, 2448, 6752, 102]).
#    decode() converts IDs back to text.
#    These three operations are the core of any tokenizer.
#
#  C# Analogy:
#    var tokens = tokenizer.Tokenize(text);       // strings
#    var ids = tokenizer.Encode(text);             // integers
#    var original = tokenizer.Decode(ids);         // back to string
# ============================================================

def inspect_tokenization(text, tokenizer):
    """
    Fully inspect how a tokenizer processes a given text.
    Show: tokens, token IDs, special token positions, and reconstruction.

    Args:
        text (str): Input text
        tokenizer: A tokenizer object with tokenize(), encode(), decode() methods

    Returns:
        dict: {
            "tokens"           : list[str],  - token strings (no special tokens)
            "ids_no_special"   : list[int],  - IDs without [CLS]/[SEP]
            "ids_with_special" : list[int],  - IDs with [CLS] and [SEP]
            "decoded"          : str,        - text reconstructed from IDs
        }
    """
    # TODO 1: Get token strings (no special tokens)
    tokens = None               # replace None
    # Hint: tokenizer.tokenize(text)
    # YOUR CODE HERE
    pass                        # remove this

    # TODO 2: Get IDs without special tokens
    ids_no_special = None       # replace None
    # Hint: tokenizer.encode(text, add_special_tokens=False)
    # YOUR CODE HERE
    pass                        # remove this

    # TODO 3: Get IDs with special tokens (default behavior)
    ids_with_special = None     # replace None
    # Hint: tokenizer.encode(text) - add_special_tokens=True is the default
    # YOUR CODE HERE
    pass                        # remove this

    # TODO 4: Decode back to text (with skip_special_tokens=True)
    decoded = None              # replace None
    # Hint: tokenizer.decode(ids_with_special, skip_special_tokens=True)
    # YOUR CODE HERE
    pass                        # remove this

    return {
        "tokens"           : tokens,
        "ids_no_special"   : ids_no_special,
        "ids_with_special" : ids_with_special,
        "decoded"          : decoded,
    }


# Test Exercise 1
print("EXERCISE 1: Inspect Tokenization")
print("-" * 40)
try:
    tok = SimulatedTokenizer()

    test_texts = [
        "hello world",
        "the cat is running quickly",
        "i love machine learning",
    ]

    for text in test_texts:
        result = inspect_tokenization(text, tok)
        if result and result["tokens"] is not None:
            print(f"  Text: '{text}'")
            print(f"    tokens:           {result['tokens']}")
            print(f"    ids (no special): {result['ids_no_special']}")
            print(f"    ids (w/ special): {result['ids_with_special']}")
            print(f"    decoded:          '{result['decoded']}'")
            cls_at_start = (result['ids_with_special'][0] == tok.cls_token_id)
            sep_at_end   = (result['ids_with_special'][-1] == tok.sep_token_id)
            print(f"    [CLS] at start: {cls_at_start}, [SEP] at end: {sep_at_end}")
            print()

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Fill in the 4 TODOs in inspect_tokenization().")

print()


# ============================================================
#  EXERCISE 2
#  Topic: Batch tokenization with padding
#
#  Background:
#    Neural networks process batches of examples simultaneously.
#    All examples in a batch must be the same length.
#    Shorter sequences are padded with [PAD] tokens.
#    The attention mask tells the model which positions are real.
#
#    Calling tokenizer(list_of_texts, padding=True) does this automatically.
#
#  C# Analogy:
#    var padded = texts.Select(t => tokenizer.Encode(t)).ToList();
#    int maxLen = padded.Max(ids => ids.Length);
#    padded = padded.Select(ids => ids.Concat(Repeat(PAD_ID, maxLen-ids.Length)));
#    var masks = padded.Select(ids => ids.Select(id => id != PAD_ID ? 1 : 0));
# ============================================================

def batch_tokenize(texts, tokenizer, max_length=20):
    """
    Tokenize a list of texts with padding and truncation.
    Returns the full batch encoding dictionary.

    Args:
        texts (list[str]): Multiple input strings
        tokenizer: Tokenizer with __call__ method
        max_length (int): Maximum sequence length

    Returns:
        dict: {"input_ids": list of lists, "attention_mask": list of lists}
              All inner lists have the same length (padded).
    """
    # TODO: Call the tokenizer with padding=True, truncation=True, max_length
    # Hint: result = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
    result = None                   # replace None
    # YOUR CODE HERE
    pass                            # remove this

    return result


def verify_batch_properties(batch_output):
    """
    Verify that a batch encoding has the expected properties:
    1. All sequences have the same length
    2. attention_mask has same shape as input_ids
    3. attention_mask contains only 1s and 0s
    4. [PAD] positions have mask=0

    Args:
        batch_output (dict): {"input_ids": ..., "attention_mask": ...}

    Returns:
        dict: {"all_same_length": bool, "mask_shape_ok": bool, "mask_values_ok": bool}
    """
    input_ids = batch_output.get("input_ids", [])
    masks     = batch_output.get("attention_mask", [])

    # TODO 1: Check all sequences have the same length
    # Hint: len(set(len(seq) for seq in input_ids)) == 1 means all same length
    all_same_length = None          # replace with True or False
    # YOUR CODE HERE
    pass                            # remove this

    # TODO 2: Check mask shape matches input_ids shape
    # Hint: compare lengths of input_ids and masks, then compare inner lengths
    mask_shape_ok = None            # replace with True or False
    # YOUR CODE HERE
    pass                            # remove this

    # TODO 3: Check mask values are only 0 or 1
    # Hint: all(v in (0, 1) for seq in masks for v in seq)
    mask_values_ok = None           # replace with True or False
    # YOUR CODE HERE
    pass                            # remove this

    return {
        "all_same_length" : all_same_length,
        "mask_shape_ok"   : mask_shape_ok,
        "mask_values_ok"  : mask_values_ok,
    }


# Test Exercise 2
print("EXERCISE 2: Batch Tokenization with Padding")
print("-" * 40)
try:
    tok = SimulatedTokenizer()

    sentences = [
        "hello",
        "the cat is running",
        "i love python is great",
    ]

    batch = batch_tokenize(sentences, tok, max_length=15)

    if batch is not None:
        print(f"  Batch of {len(sentences)} sentences:")
        for i, (sent, ids, mask) in enumerate(zip(sentences,
                                                    batch["input_ids"],
                                                    batch["attention_mask"])):
            print(f"  [{i}] '{sent}'")
            print(f"       input_ids:      {ids}")
            print(f"       attention_mask: {mask}")
            real_count = sum(mask)
            pad_count  = mask.count(0)
            print(f"       real={real_count} tokens, padding={pad_count} tokens")
            print()

        print("  Verifying batch properties:")
        props = verify_batch_properties(batch)
        for prop_name, prop_val in props.items():
            status = "PASS" if prop_val else "FAIL"
            print(f"    {status}: {prop_name} = {prop_val}")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Fill in batch_tokenize() and verify_batch_properties().")

print()


# ============================================================
#  EXERCISE 3
#  Topic: Inspect special token information
#
#  Background:
#    Every tokenizer has special tokens with fixed IDs.
#    These IDs are determined when the tokenizer is created/trained.
#    You can inspect them via properties: cls_token_id, sep_token_id, etc.
#
#  C# Analogy:
#    var specialTokenInfo = new Dictionary<string, int> {
#        { tokenizer.ClsToken, tokenizer.ClsTokenId },
#        { tokenizer.SepToken, tokenizer.SepTokenId },
#        ...
#    };
# ============================================================

def get_special_token_info(tokenizer):
    """
    Collect information about all special tokens.

    Returns a dict mapping token strings to their IDs.
    Should include: [CLS], [SEP], [PAD], [UNK], [MASK].

    Args:
        tokenizer: A tokenizer object

    Returns:
        dict: {token_string: token_id}
    """
    # TODO: Collect special token name -> ID mappings
    # Access: tokenizer.cls_token, tokenizer.cls_token_id, etc.
    # YOUR CODE HERE
    special_tokens = {}
    # Fill in cls, sep, pad, unk, mask token -> id pairs
    pass                                    # remove this - add the entries

    return special_tokens


# Test Exercise 3
print("EXERCISE 3: Special Token Information")
print("-" * 40)
try:
    tok = SimulatedTokenizer()
    info = get_special_token_info(tok)

    if info:
        print(f"  Special tokens ({len(info)} found):")
        for token, token_id in info.items():
            print(f"    '{token}' -> ID {token_id}")

        # Verify specific values
        expected = {"[CLS]": 101, "[SEP]": 102, "[PAD]": 0}
        for token, expected_id in expected.items():
            actual_id = info.get(token)
            status = "PASS" if actual_id == expected_id else "FAIL"
            print(f"  {status}: {token} has ID {actual_id} (expected {expected_id})")
    else:
        print("  [INFO] special_tokens is empty - fill in the TODO section.")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Access tokenizer.cls_token, tokenizer.cls_token_id, etc.")

print()


# ============================================================
#  EXERCISE 4
#  Topic: Truncation behavior
#
#  Background:
#    Models have a maximum context length (e.g., BERT: 512 tokens).
#    Texts longer than max_length must be truncated.
#    Truncation should keep [CLS] at start and [SEP] at end.
#    The middle tokens are what gets cut off.
#
#  C# Analogy:
#    if (ids.Length > maxLength) {
#        ids = ids.Take(maxLength - 1).Append(SEP_ID).ToArray();
#    }
# ============================================================

def truncate_encoding(text, tokenizer, max_length):
    """
    Tokenize text with truncation to max_length.
    Verify that the output has exactly max_length tokens.

    Args:
        text (str): Input text
        tokenizer: Tokenizer with __call__ method
        max_length (int): Maximum number of tokens

    Returns:
        dict with "input_ids" and "attention_mask", both of length max_length
    """
    # TODO: Call tokenizer with truncation=True and max_length
    # YOUR CODE HERE
    result = None                       # replace None
    pass                                # remove this

    return result


# Test Exercise 4
print("EXERCISE 4: Truncation")
print("-" * 40)
try:
    tok = SimulatedTokenizer()

    # Long text that will exceed max_length
    long_text = "hello world the cat is running quickly. the dog is fast. i love python."
    max_len = 8

    result = truncate_encoding(long_text, tok, max_length=max_len)

    if result is not None:
        ids = result["input_ids"]
        mask = result["attention_mask"]

        print(f"  Text: '{long_text[:50]}...'")
        print(f"  max_length: {max_len}")
        print(f"  input_ids (length={len(ids)}):      {ids}")
        print(f"  attention_mask (length={len(mask)}): {mask}")

        # Verify truncation
        correct_len = len(ids) == max_len
        has_cls = ids[0] == tok.cls_token_id if ids else False
        has_sep = ids[-1] == tok.sep_token_id if ids else False

        print(f"  Length is {max_len}: {'PASS' if correct_len else 'FAIL'}")
        print(f"  Starts with [CLS]: {'PASS' if has_cls else 'FAIL'}")
        print(f"  Ends with [SEP]: {'PASS' if has_sep else 'FAIL'}")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Call tokenizer with truncation=True, max_length=max_length.")

print()


# ============================================================
#  EXERCISE 5
#  Topic: Using real HuggingFace tokenizer (if available)
#
#  Background:
#    The simulated tokenizer above has the same API as the real AutoTokenizer.
#    If transformers is installed, this exercise uses the REAL BERT tokenizer.
#    You will see that all the same methods work identically.
#
#  C# Analogy:
#    // The API contract (interface) is the same whether you use
#    // the mock in unit tests or the real implementation in production.
#    ITokenizer tok = environment.IsTest() ? new MockTokenizer() : new BertTokenizer();
#    var ids = tok.Encode("Hello world");  // identical call in both cases
# ============================================================

print("EXERCISE 5: Using Real HuggingFace Tokenizer")
print("-" * 40)
print()

if HF_AVAILABLE:
    # TODO: Load the real BERT tokenizer and repeat Exercise 1's inspection
    print("  Loading real BERT tokenizer...")
    try:
        # TODO: Load real tokenizer
        # Hint: AutoTokenizer.from_pretrained("bert-base-uncased")
        real_tok = None                 # replace None
        # YOUR CODE HERE
        pass                            # remove this

        if real_tok is not None:
            print(f"  Loaded! Vocab size: {real_tok.vocab_size}")
            print()

            test_text = "I love natural language processing"

            # TODO: Reuse inspect_tokenization from Exercise 1
            result = inspect_tokenization(test_text, real_tok)
            if result and result["tokens"] is not None:
                print(f"  Text: '{test_text}'")
                print(f"  tokens:    {result['tokens']}")
                print(f"  ids:       {result['ids_with_special']}")
                print(f"  decoded:   '{result['decoded']}'")
                print()
                print("  Compare to our simulated tokenizer:")
                sim_result = inspect_tokenization(test_text, SimulatedTokenizer())
                print(f"  Simulated tokens: {sim_result['tokens']}")
                print(f"  (Real BERT has 30,522 tokens; simulation has {SimulatedTokenizer().vocab_size})")

    except Exception as e:
        print(f"  [ERROR loading BERT]: {e}")
        print(f"  Try: pip install transformers")

else:
    print("  (HuggingFace not installed - showing what the code would do)")
    print()
    print("  If transformers were installed:")
    print("  real_tok = AutoTokenizer.from_pretrained('bert-base-uncased')")
    print("  result   = inspect_tokenization('I love natural language processing', real_tok)")
    print("  print(result['tokens'])")
    print("  # ['i', 'love', 'natural', 'language', 'processing']")
    print()
    print("  The API is identical to our SimulatedTokenizer.")
    print("  That's the point of the simulation - learn the API before needing the library!")


print()
print("=" * 60)
print("Exercise 04 Complete (or attempted)!")
print()
print("Expected outputs:")
print("  Ex1: tokens=['run','##ning'], ids=[101,...,102], decoded='running'")
print("  Ex2: 3 sequences all padded to same length, masks correct")
print("  Ex3: [CLS]=101, [SEP]=102, [PAD]=0, [UNK]=100, [MASK]=103")
print("  Ex4: output has exactly max_length=8 tokens, starts [CLS], ends [SEP]")
print("  Ex5: real BERT gives same API, slightly different token splits")
print("=" * 60)

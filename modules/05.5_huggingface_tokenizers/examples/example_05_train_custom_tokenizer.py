"""
Module 05.5 - HuggingFace Tokenizers
Example 05: Training a Custom Tokenizer

GLOSSARY
--------
Training corpus  : Raw text used to train the tokenizer. No labels needed.
                   C# analogy: the "training data" for a spell-check dictionary.

BpeTrainer       : HuggingFace class that runs BPE training given a corpus.
                   Reads text, counts pairs, runs merges, builds vocabulary.
                   C# analogy: like a builder pattern for a compression dictionary.

vocab_size       : Total number of tokens in the final vocabulary.
                   Larger = fewer splits per word but more memory.
                   Typical: 8K for small domain, 32K for general, 100K for multilingual.

min_frequency    : Minimum times a token must appear to be kept.
                   Filters out typos and very rare tokens.
                   C# analogy: like a frequency threshold in a word-count filter.

special_tokens   : Reserved token strings added to vocabulary with fixed IDs.
                   Must be in the vocab before training so their IDs are predictable.

Trainer          : Manages the training loop. You give it corpus files + settings.
                   It calls the algorithm (BPE/WordPiece), counts frequencies,
                   runs merge steps, and builds the final vocabulary.

This example trains a custom BPE tokenizer on a domain-specific corpus.
It includes pure Python fallback that demonstrates the concept without any library.
"""

import os                          # os: filesystem operations (like System.IO in C#)
import json                        # json: read/write JSON files (like JsonSerializer in C#)
import tempfile                    # tempfile: create temporary files and directories

print("=" * 60)
print("Example 05: Training a Custom Tokenizer")
print("=" * 60)
print()

# ---- Try to import HuggingFace tokenizers library ----
try:
    from tokenizers import Tokenizer                # base Tokenizer class
    from tokenizers.models import BPE               # BPE algorithm
    from tokenizers.trainers import BpeTrainer      # BPE training manager
    from tokenizers.pre_tokenizers import Whitespace # pre-tokenizer
    HF_TOKENIZERS_AVAILABLE = True
    print("[OK] tokenizers library available")
except ImportError:
    HF_TOKENIZERS_AVAILABLE = False
    print("[INFO] tokenizers library not installed. Using pure Python simulation.")
    print("[INFO] Install with: pip install tokenizers")

print()


# ============================================================
# PART A: Prepare Training Corpus
# ============================================================

print("PART A: Creating Training Corpus")
print("-" * 50)
print()

# We'll use a small domain-specific corpus about finance
# In real use, this would be millions of lines from domain-specific text
TRAINING_CORPUS = [
    # Finance domain sentences - specialized vocabulary
    "The balance sheet shows total assets of fifty million dollars.",
    "Accounts receivable increased by fifteen percent this quarter.",
    "The general ledger contains all financial transactions.",
    "Amortization of goodwill affected earnings per share significantly.",
    "The accounts payable team processed three hundred invoices.",
    "Revenue recognition follows the accrual accounting principle.",
    "Depreciation expense was recorded on fixed assets quarterly.",
    "The trial balance verifies debits equal credits in all accounts.",
    "Cash flow from operations exceeded capital expenditures this year.",
    "Intercompany eliminations removed duplicate revenue entries.",
    "The consolidation process merged subsidiary financials.",
    "Deferred tax liability relates to timing differences in recognition.",
    "The audit committee reviewed internal controls over financial reporting.",
    "Working capital management improved the current ratio significantly.",
    "Accounts receivable aging showed thirty percent past due invoices.",
    "The reconciliation process matched bank statements to general ledger.",
    "Amortization schedules for intangible assets span five to twenty years.",
    "Currency translation adjustments affected other comprehensive income.",
    "The impairment test compared carrying value to recoverable amount.",
    "Capital lease obligations were reclassified under new accounting standards.",
    # Add more variety for BPE to learn useful subwords
    "receivable payable account accounts accounting accountant accountants",
    "depreciation amortization reconciliation elimination consolidation",
    "financial financing financed finance finances financially",
    "quarter quarterly quartile four fourth",
    "revenue revenues revenued revenuer",
    "asset assets liability liabilities equity equities",
]

print(f"Training corpus: {len(TRAINING_CORPUS)} sentences")
print("Sample sentences:")
for sent in TRAINING_CORPUS[:3]:       # show first 3 sentences
    print(f"  '{sent[:70]}...'")       # truncate long lines for display
print()

# Write corpus to a temporary file (tokenizer trains from files)
corpus_file = "finance_corpus.txt"     # file path to save corpus

with open(corpus_file, "w", encoding="utf-8") as f:
    for sentence in TRAINING_CORPUS:   # write each sentence to file
        f.write(sentence + "\n")       # \n = newline after each sentence

print(f"Corpus written to: {corpus_file}")
print(f"  File size: {os.path.getsize(corpus_file)} bytes")
print()


# ============================================================
# PART B: Pure Python BPE Training (Always Runs)
# ============================================================

print("PART B: Training BPE Tokenizer in Pure Python")
print("-" * 50)
print()

# This is the full BPE training algorithm implemented in pure Python
# Same logic as in Example 02, now packaged cleanly for training

from collections import defaultdict    # defaultdict: like ConcurrentDictionary in C#


def build_word_freq_table(corpus_lines):
    """
    Convert text lines into a word-frequency table.
    Each word is represented as a tuple of characters + end-of-word marker.

    The </w> marker tells BPE where word boundaries are.
    Without it: "low" + "er" might merge incorrectly with "ower" from "flower".

    C# analogy:
      var wordFreqs = corpus
        .SelectMany(line => line.ToLower().Split())
        .GroupBy(w => w)
        .ToDictionary(g => string.Join(" ", g.Key.ToCharArray()) + " </w>", g => g.Count());

    Args:
        corpus_lines (list[str]): Lines of text

    Returns:
        dict: {char_tuple: frequency}
    """
    freq = defaultdict(int)                             # word -> frequency count
    for line in corpus_lines:                           # loop over each line
        words = line.lower().split()                    # lowercase + split on spaces
        for word in words:                              # loop over each word
            if word:                                    # skip empty strings
                # Represent as tuple of chars + </w> end marker
                key = tuple(list(word) + ["</w>"])     # e.g., "cat" -> ('c','a','t','</w>')
                freq[key] += 1                          # increment count
    return dict(freq)                                   # return regular dict


def count_pairs(vocab):
    """Count all adjacent pairs in vocab weighted by frequency."""
    pairs = defaultdict(int)
    for word_tuple, freq in vocab.items():
        syms = list(word_tuple)                         # convert tuple to list
        for i in range(len(syms) - 1):                 # iterate pairs
            pairs[(syms[i], syms[i+1])] += freq        # weight by word frequency
    return dict(pairs)


def apply_merge(vocab, pair_to_merge):
    """Apply one merge: replace all occurrences of pair with merged token."""
    merged = pair_to_merge[0] + pair_to_merge[1]       # "e" + "r" = "er"
    new_vocab = {}

    for word_tuple, freq in vocab.items():
        syms = list(word_tuple)
        new_syms = []
        i = 0
        while i < len(syms):
            if (i < len(syms) - 1 and
                    syms[i] == pair_to_merge[0] and
                    syms[i+1] == pair_to_merge[1]):
                new_syms.append(merged)                 # merge the pair
                i += 2
            else:
                new_syms.append(syms[i])               # keep as-is
                i += 1
        new_vocab[tuple(new_syms)] = freq

    return new_vocab, merged                            # return new vocab and merged token


def train_bpe_tokenizer(corpus_lines, vocab_size=100, min_frequency=1):
    """
    Train a BPE tokenizer on the given corpus.

    Full BPE training algorithm:
    1. Build initial character-level vocabulary
    2. Find most frequent adjacent pair
    3. Merge that pair into a new token
    4. Record merge rule
    5. Repeat until vocabulary reaches target size

    C# analogy:
      var tokenizer = new BpeTokenizer();
      tokenizer.Train(corpusLines, vocabSize: 100);
      // Internally: while(vocab.Count < vocabSize) { MergeMostFrequentPair(); }

    Args:
        corpus_lines (list[str]): Lines of training text
        vocab_size (int): Target vocabulary size
        min_frequency (int): Minimum frequency to keep a token

    Returns:
        tuple: (final_vocab_dict, list_of_merge_rules)
    """
    print(f"  Starting BPE training...")
    print(f"  Target vocab size: {vocab_size}")
    print(f"  Min frequency: {min_frequency}")
    print()

    # Step 1: Build initial vocabulary (character-level)
    word_freq = build_word_freq_table(corpus_lines)

    # Collect all individual character tokens (starting vocabulary)
    char_tokens = set()
    for word_tuple in word_freq.keys():
        for char in word_tuple:
            char_tokens.add(char)

    # Assign IDs to special tokens + character tokens
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    all_initial_tokens = special_tokens + sorted(char_tokens)

    token_to_id = {tok: idx for idx, tok in enumerate(all_initial_tokens)}
    current_vocab_size = len(token_to_id)

    print(f"  Initial vocabulary: {current_vocab_size} tokens "
          f"({len(special_tokens)} special + {len(char_tokens)} chars)")

    # Step 2: BPE merge loop
    merge_rules = []                                    # ordered list of merge rules
    num_merges_to_do = vocab_size - current_vocab_size  # how many merges needed

    for step in range(num_merges_to_do):
        pairs = count_pairs(word_freq)                  # count all adjacent pairs

        if not pairs:
            print(f"  No more pairs to merge at step {step+1}. Stopping.")
            break

        # Filter by minimum frequency
        pairs = {p: f for p, f in pairs.items() if f >= min_frequency}
        if not pairs:
            print(f"  No pairs with frequency >= {min_frequency}. Stopping.")
            break

        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        best_freq = pairs[best_pair]

        # Apply merge
        word_freq, merged_token = apply_merge(word_freq, best_pair)
        merge_rules.append((best_pair, merged_token))   # record the rule
        token_to_id[merged_token] = len(token_to_id)    # assign new ID

        if (step + 1) % 10 == 0 or step < 5:           # print first 5 and every 10th
            print(f"  Step {step+1:3d}: merge {best_pair[0]!r:10s} + {best_pair[1]!r:10s}"
                  f" -> {merged_token!r:<15s} (freq={best_freq})")

    print()
    print(f"  Training complete!")
    print(f"  Final vocabulary size: {len(token_to_id)} tokens")
    print(f"  Merge rules learned: {len(merge_rules)}")

    return token_to_id, merge_rules


def bpe_encode_text(text, merge_rules, token_to_id):
    """
    Encode text using trained BPE tokenizer.
    Apply merge rules in training order.

    Args:
        text (str): Text to encode
        merge_rules (list): Ordered merge rules from training
        token_to_id (dict): Final vocabulary

    Returns:
        tuple: (token_strings, token_ids)
    """
    all_tokens = []

    for word in text.lower().split():                   # process each word
        syms = list(word) + ["</w>"]                    # char-level + end marker

        for (pair_a, pair_b), merged in merge_rules:    # apply each rule in order
            i = 0
            new_syms = []
            while i < len(syms):
                if (i < len(syms) - 1 and
                        syms[i] == pair_a and
                        syms[i+1] == pair_b):
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms

        all_tokens.extend(syms)

    ids = [token_to_id.get(t, token_to_id.get("[UNK]", 1)) for t in all_tokens]
    return all_tokens, ids


# Train the pure Python BPE tokenizer
print("Training pure Python BPE tokenizer on finance corpus...")
py_vocab, py_merge_rules = train_bpe_tokenizer(
    TRAINING_CORPUS,
    vocab_size=80,              # small: just for demonstration
    min_frequency=1,            # include all pairs (corpus is tiny)
)

# Test encoding
print()
print("Testing pure Python BPE tokenizer:")
test_words = [
    "accounts receivable",
    "depreciation",
    "financial reporting",
    "amortization",
    "ledger",
]

for text in test_words:
    toks, ids = bpe_encode_text(text, py_merge_rules, py_vocab)
    print(f"  '{text}' -> {toks}")

print()


# ============================================================
# PART C: HuggingFace BPE Training (If Available)
# ============================================================

print("PART C: HuggingFace BpeTrainer")
print("-" * 50)
print()

if HF_TOKENIZERS_AVAILABLE:
    print("Training BPE tokenizer using HuggingFace tokenizers library...")
    print()

    # Step 1: Create empty BPE tokenizer
    hf_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # BPE(unk_token="[UNK]"): creates empty BPE model
    # Any token not in vocab will be mapped to [UNK]

    # Step 2: Set pre-tokenizer (splits raw text into words before BPE runs)
    hf_tokenizer.pre_tokenizer = Whitespace()
    # Whitespace(): splits on whitespace and punctuation
    # "Hello, world" -> ["Hello", ",", "world"]

    # Step 3: Configure trainer
    trainer = BpeTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        # special_tokens: reserved tokens added FIRST with IDs 0,1,2,3,4

        vocab_size=100,
        # vocab_size: target vocabulary size (small for our tiny corpus)

        min_frequency=1,
        # min_frequency: token must appear at least this many times
        # Set to 1 for tiny corpora (every token is included)

        show_progress=False,
        # show_progress: whether to show tqdm progress bar
    )

    # Step 4: Train on corpus files
    print("  Running BpeTrainer on finance_corpus.txt...")
    hf_tokenizer.train([corpus_file], trainer)          # pass list of file paths
    print(f"  Done! Vocabulary size: {hf_tokenizer.get_vocab_size()} tokens")
    print()

    # Step 5: Test the tokenizer
    print("  Testing HuggingFace BPE tokenizer:")
    test_texts_hf = [
        "accounts receivable",
        "depreciation amortization",
        "financial reporting",
        "general ledger reconciliation",
    ]

    for text in test_texts_hf:
        output = hf_tokenizer.encode(text)              # encode text
        tokens = output.tokens                           # list of token strings
        ids = output.ids                                 # list of token IDs

        print(f"  '{text}'")
        print(f"    tokens: {tokens}")
        print(f"    IDs:    {ids}")
        print()

    # Step 6: Save the tokenizer
    save_path = "finance_tokenizer.json"                # save file path
    hf_tokenizer.save(save_path)                        # saves everything to one JSON file
    print(f"  Tokenizer saved to: {save_path}")
    print(f"  File size: {os.path.getsize(save_path)} bytes")
    print()

    # Step 7: Load it back
    print("  Loading saved tokenizer...")
    loaded_tokenizer = Tokenizer.from_file(save_path)  # load from JSON file
    print(f"  Loaded! Vocab size: {loaded_tokenizer.get_vocab_size()}")

    # Verify loaded tokenizer gives same results
    verify_text = "accounts receivable ledger"
    original_result = hf_tokenizer.encode(verify_text).tokens
    loaded_result   = loaded_tokenizer.encode(verify_text).tokens

    print(f"  Verify same output after load:")
    print(f"    Original: {original_result}")
    print(f"    Loaded:   {loaded_result}")
    print(f"    Match: {original_result == loaded_result}")
    print()

    # Clean up save file
    if os.path.exists(save_path):
        os.remove(save_path)                            # delete the file

else:
    print("(Skipping HuggingFace training - tokenizers library not installed)")
    print()
    print("To use HuggingFace BpeTrainer, install: pip install tokenizers")
    print()
    print("Code that would run:")
    print("  from tokenizers import Tokenizer")
    print("  from tokenizers.models import BPE")
    print("  from tokenizers.trainers import BpeTrainer")
    print("  from tokenizers.pre_tokenizers import Whitespace")
    print()
    print("  tokenizer = Tokenizer(BPE(unk_token='[UNK]'))")
    print("  tokenizer.pre_tokenizer = Whitespace()")
    print("  trainer = BpeTrainer(vocab_size=1000, special_tokens=['[UNK]','[PAD]'])")
    print("  tokenizer.train(['corpus.txt'], trainer)")
    print("  tokenizer.save('my_tokenizer.json')")


# ============================================================
# PART D: Vocabulary Size Comparison
# ============================================================

print("PART D: Effect of Vocabulary Size on Tokenization")
print("-" * 50)
print()

# Demonstrate how different vocab sizes affect tokenization
# We'll simulate by varying how aggressively we merge

def demo_vocab_size_effect():
    """
    Show how vocabulary size affects tokenization quality.
    Small vocab -> more tokens per word. Large vocab -> fewer tokens.
    """
    test_word = "depreciation"
    print(f"Word: '{test_word}'")
    print()

    # Simulate different vocab sizes by stopping BPE at different points
    configs = [
        (5,  "tiny (5 merges)"),     # very early stop
        (15, "small (15 merges)"),    # some merges
        (30, "medium (30 merges)"),   # more merges
    ]

    for max_merges, label in configs:
        # Use only the first N merge rules
        limited_rules = py_merge_rules[:max_merges]     # [:N] = first N items

        toks, ids = bpe_encode_text(test_word, limited_rules, py_vocab)
        print(f"  {label}: {len(toks)} tokens -> {toks}")

    print()
    print("Observation:")
    print("  Fewer merges (small vocab) = more tokens per word = longer sequences")
    print("  More merges (large vocab)  = fewer tokens per word = shorter sequences")
    print("  Trade-off: vocabulary size vs sequence length")


demo_vocab_size_effect()


# ============================================================
# PART E: Special Tokens and Their Importance
# ============================================================

print("PART E: Understanding Special Tokens in Custom Tokenizers")
print("-" * 50)
print()

print("Special tokens are ALWAYS added first to the vocabulary.")
print("Their IDs are FIXED and must match between tokenizer and model.")
print()

special_token_table = [
    ("[PAD]",  0, "Padding. Fills shorter sequences in a batch. Model ignores these."),
    ("[UNK]",  1, "Unknown. Any token not in vocabulary maps here."),
    ("[CLS]",  2, "Classification token. First token of every BERT input."),
    ("[SEP]",  3, "Separator. Marks end of a sentence or segment."),
    ("[MASK]", 4, "Mask. Replaced during masked language model training."),
]

print(f"  {'Token':<10} {'ID':<5} {'Purpose'}")
print("  " + "-" * 60)
for token, token_id, purpose in special_token_table:
    print(f"  {token:<10} {token_id:<5} {purpose}")

print()
print("C# analogy:")
print("  Special tokens are like reserved keywords in C#.")
print("  Just as 'class', 'void', 'return' cannot be variable names,")
print("  [CLS], [SEP], [PAD] are reserved in the tokenizer vocabulary.")
print("  Their IDs are hardcoded in the model architecture code.")
print()


# ============================================================
# PART F: When to Use Custom vs Pretrained Tokenizer
# ============================================================

print("PART F: Custom vs Pretrained Tokenizer - Decision Guide")
print("-" * 50)
print()

decision_cases = [
    ("Fine-tuning BERT for classification",
     "PRETRAINED (BERT's WordPiece)",
     "Model expects specific token IDs. Changing tokenizer breaks the model."),

    ("Building GPT-2 from scratch",
     "PRETRAINED (GPT-2 BPE or similar)",
     "Unless your data is very different, reuse existing high-quality tokenizer."),

    ("Medical report summarization",
     "CUSTOM (trained on medical corpus)",
     "'electrocardiogram' -> 5 tokens in general BPE vs 1-2 in medical BPE."),

    ("Python code generation model",
     "CUSTOM (trained on code corpus)",
     "Keywords 'def', 'class', '__init__' should be single tokens."),

    ("Multilingual model (100+ languages)",
     "CUSTOM (large multilingual BPE)",
     "Need tokens for every language's characters. 100K+ vocab size needed."),

    ("Legal document analysis",
     "CUSTOM (trained on legal corpus)",
     "'notwithstanding', 'hereinafter' are common in law, rare in general English."),
]

print("Decision: Custom tokenizer or pretrained tokenizer?")
print()
for case, decision, reason in decision_cases:
    print(f"  Use case: {case}")
    print(f"  Decision: {decision}")
    print(f"  Reason:   {reason}")
    print()


# Clean up the corpus file we created
if os.path.exists(corpus_file):
    os.remove(corpus_file)             # delete temporary file
    print(f"Cleaned up: {corpus_file}")


print()
print("=" * 60)
print("Example 05 Complete!")
print()
print("Key Takeaways:")
print("  1. Tokenizer training needs only raw text, no labels, no GPU")
print("  2. BpeTrainer runs the BPE loop: count pairs -> merge -> repeat")
print("  3. vocab_size controls the trade-off: larger = fewer tokens/word")
print("  4. Save: tokenizer.save('file.json') | Load: Tokenizer.from_file('file.json')")
print("  5. Custom tokenizer: needed for domain-specific or new-language work")
print("  6. Fine-tuning: NEVER change the tokenizer - use the model's original")
print("=" * 60)

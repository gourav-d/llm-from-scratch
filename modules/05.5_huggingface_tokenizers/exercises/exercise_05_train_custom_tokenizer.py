"""
Module 05.5 - HuggingFace Tokenizers
Exercise 05: Training a Custom Tokenizer

GLOSSARY
--------
Training corpus  : Raw text used to train the tokenizer. No labels needed.
                   C# analogy: the data used to build a custom spell-check dictionary.

BpeTrainer       : The class that runs BPE training. You configure it, then call .train().
                   C# analogy: a Builder pattern for constructing a compression dictionary.

vocab_size       : Number of tokens in the final vocabulary.
                   Larger -> fewer splits per word. Smaller -> more splits.
                   Typical: 8K small domain, 32K general, 100K+ multilingual.

min_frequency    : A token pair must appear at least this many times to be merged.
                   Filters out typos and rare junk tokens.
                   C# analogy: frequency filter in a word-count HashSet.

Coverage         : Fraction of tokens that are NOT [UNK].
                   100% means the tokenizer knows every token in the test set.
                   Low coverage means the vocabulary is too small or wrong domain.

Save/Load        : Tokenizer config is saved to JSON; loaded without retraining.
                   C# analogy: serialize/deserialize configuration to disk.
"""

import os                           # os: filesystem operations
from collections import defaultdict # defaultdict: auto-initializes missing keys

print("=" * 60)
print("Exercise 05: Training a Custom Tokenizer")
print("=" * 60)
print()

# Try HuggingFace tokenizers library
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    HF_AVAILABLE = True
    print("[OK] tokenizers library available")
except ImportError:
    HF_AVAILABLE = False
    print("[INFO] tokenizers not installed. Using pure Python implementation.")
    print("[INFO] Install: pip install tokenizers")

print()


# ============================================================
# Helper functions (from Example 02 - reused here)
# ============================================================

def _build_word_freq(corpus):
    """Build character-level word frequency table from corpus."""
    freq = defaultdict(int)
    for line in corpus:
        for word in line.lower().split():
            if word:
                freq[tuple(list(word) + ["</w>"])] += 1
    return dict(freq)


def _count_pairs(vocab):
    """Count adjacent pairs in vocabulary."""
    pairs = defaultdict(int)
    for word_tuple, freq in vocab.items():
        syms = list(word_tuple)
        for i in range(len(syms) - 1):
            pairs[(syms[i], syms[i+1])] += freq
    return dict(pairs)


def _apply_merge(vocab, pair):
    """Apply one merge to vocabulary."""
    merged = pair[0] + pair[1]
    new_vocab = {}
    for word_tuple, freq in vocab.items():
        syms = list(word_tuple)
        new_syms = []
        i = 0
        while i < len(syms):
            if i < len(syms)-1 and syms[i] == pair[0] and syms[i+1] == pair[1]:
                new_syms.append(merged)
                i += 2
            else:
                new_syms.append(syms[i])
                i += 1
        new_vocab[tuple(new_syms)] = freq
    return new_vocab, merged


# ============================================================
#  EXERCISE 1
#  Topic: Prepare and analyze training corpus
#
#  Background:
#    Before training a tokenizer, analyze the corpus to understand
#    what tokens will be learned.
#    Key metrics: total words, unique words, word frequency distribution.
#
#    For domain-specific tokenizers, knowing the most common words helps
#    predict which will become single tokens vs split into pieces.
#
#  C# Analogy:
#    var wordCounts = corpus
#      .SelectMany(s => s.ToLower().Split())
#      .GroupBy(w => w)
#      .ToDictionary(g => g.Key, g => g.Count())
#      .OrderByDescending(kv => kv.Value);
# ============================================================

# Sample domain corpus: legal text
LEGAL_CORPUS = [
    "The defendant shall appear before the court on the specified date.",
    "Pursuant to the contract, all parties must fulfill their obligations.",
    "The plaintiff filed a motion for summary judgment with the court.",
    "Notwithstanding the foregoing, the agreement shall remain in effect.",
    "The arbitration clause supersedes all prior agreements between parties.",
    "The indemnification provision protects the company from liability.",
    "Breach of contract results in damages payable to the injured party.",
    "The jurisdiction clause specifies which court shall hear disputes.",
    "Force majeure clauses excuse performance during extraordinary events.",
    "The statute of limitations bars claims filed after the prescribed period.",
    "Parties agree to confidentiality regarding all proprietary information.",
    "The governing law provision determines which jurisdiction applies.",
    "Default on payment obligations triggers the acceleration clause.",
    "The arbitrator shall render a decision within thirty business days.",
    "Intellectual property rights shall be assigned upon contract completion.",
    "The liquidated damages clause estimates harm from contract breach.",
    "Parties waive the right to trial by jury for all disputes arising.",
    "The warranty period extends for twelve months from delivery date.",
    "Termination for convenience allows either party to end the agreement.",
    "The limitation of liability clause caps recoverable damages at contract value.",
]


def analyze_corpus(corpus):
    """
    Analyze the training corpus and return statistics.

    Compute:
    - total word count (with duplicates)
    - unique word count
    - top-10 most frequent words
    - vocabulary that will initially be character-level

    Args:
        corpus (list[str]): List of training sentences

    Returns:
        dict: {
            "total_words"   : int,
            "unique_words"  : int,
            "top_10_words"  : list of (word, count) tuples, sorted descending,
            "char_vocab_size": int (number of unique characters),
        }
    """
    # TODO 1: Count word frequencies
    word_counts = defaultdict(int)
    for sentence in corpus:
        words = sentence.lower().split()        # lowercase + split
        for word in words:
            if word:                            # skip empty strings
                # TODO: increment count for this word
                # YOUR CODE HERE
                pass                            # remove this

    # TODO 2: Compute statistics
    total_words    = None                       # replace with sum of all counts
    unique_words   = None                       # replace with len(word_counts)
    # Hint: sum(word_counts.values()) for total

    # TODO 3: Find top 10 words by frequency
    # Hint: sorted(word_counts.items(), key=lambda x: -x[1])[:10]
    top_10_words = None                         # replace with sorted list
    # YOUR CODE HERE
    pass                                        # remove this

    # TODO 4: Count unique characters in corpus
    all_chars = set()
    for sentence in corpus:
        for char in sentence.lower():
            if char != " ":                     # skip spaces (they become </w>)
                all_chars.add(char)
    char_vocab_size = len(all_chars)

    return {
        "total_words"    : total_words,
        "unique_words"   : unique_words,
        "top_10_words"   : top_10_words,
        "char_vocab_size": char_vocab_size,
    }


# Test Exercise 1
print("EXERCISE 1: Corpus Analysis")
print("-" * 40)
try:
    stats = analyze_corpus(LEGAL_CORPUS)

    if stats["total_words"] is not None:
        print(f"  Total words:      {stats['total_words']}")
        print(f"  Unique words:     {stats['unique_words']}")
        print(f"  Unique characters:{stats['char_vocab_size']} (BPE starts here)")
        print()
        print("  Top 10 most frequent words:")
        if stats["top_10_words"]:
            for word, count in stats["top_10_words"]:
                print(f"    '{word}' x {count}")
    else:
        print("  [INFO] Fill in the TODO sections to compute statistics.")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Check analyze_corpus() implementation.")

print()


# ============================================================
#  EXERCISE 2
#  Topic: Train a pure Python BPE tokenizer
#
#  Background:
#    Using the helper functions above, train BPE on the legal corpus.
#    The training loop:
#    1. Build character vocabulary
#    2. Count pairs
#    3. Merge most frequent pair
#    4. Record merge rule
#    5. Repeat num_merges times
#
#  C# Analogy:
#    var tokenizer = new CustomBpeTokenizer();
#    tokenizer.Train(legalCorpus, vocabSize: 150);
#    // Same pattern as any ML training loop: initialize, iterate, record.
# ============================================================

def train_pure_python_bpe(corpus, vocab_size=150, min_frequency=1, verbose=True):
    """
    Train a BPE tokenizer using pure Python helpers.

    Args:
        corpus (list[str]): Training sentences
        vocab_size (int): Target vocabulary size
        min_frequency (int): Minimum pair frequency to merge
        verbose (bool): Print progress

    Returns:
        tuple: (token_to_id, merge_rules)
               token_to_id: {token_string: int_id}
               merge_rules: list of ((pair_a, pair_b), merged_token) in order
    """
    # Step 1: Build initial char vocabulary
    word_freq = _build_word_freq(corpus)

    # Collect all initial characters
    initial_chars = set()
    for word_tuple in word_freq.keys():
        for char in word_tuple:
            initial_chars.add(char)

    # Build initial token->ID mapping
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    sorted_chars = sorted(initial_chars)
    all_initial = special_tokens + sorted_chars

    token_to_id = {tok: idx for idx, tok in enumerate(all_initial)}
    # All special tokens + all chars are already in vocab

    num_merges = vocab_size - len(token_to_id)  # how many merges to do
    merge_rules = []

    if verbose:
        print(f"  Initial vocab: {len(token_to_id)} tokens "
              f"({len(special_tokens)} special + {len(sorted_chars)} chars)")
        print(f"  Target vocab:  {vocab_size} tokens ({num_merges} merges needed)")
        print()

    # TODO: Training loop
    for step in range(1, num_merges + 1):
        # TODO 1: Count pair frequencies
        pairs = None                        # replace None
        # YOUR CODE HERE
        pass                                # remove this

        if not pairs:
            if verbose:
                print(f"  No more pairs at step {step}. Stopping.")
            break

        # TODO 2: Filter by min_frequency
        # Hint: {p: f for p, f in pairs.items() if f >= min_frequency}
        pairs = None                        # replace None
        # YOUR CODE HERE
        pass                                # remove this

        if not pairs:
            if verbose:
                print(f"  No pairs with freq >= {min_frequency}. Stopping.")
            break

        # TODO 3: Find best pair (highest frequency)
        best_pair = None                    # replace None
        # Hint: max(pairs, key=pairs.get)
        # YOUR CODE HERE
        pass                                # remove this

        best_freq = pairs[best_pair]

        # TODO 4: Apply the merge (returns new word_freq dict and merged token)
        word_freq, merged_tok = None, None  # replace Nones
        # Hint: _apply_merge(word_freq, best_pair)
        # YOUR CODE HERE
        pass                                # remove this (unpack the tuple)

        # Record merge rule and add new token to vocab
        merge_rules.append((best_pair, merged_tok))
        token_to_id[merged_tok] = len(token_to_id)

        if verbose and (step <= 5 or step % 10 == 0):
            print(f"  Step {step:3d}: {best_pair[0]!r:10s}+{best_pair[1]!r:10s}"
                  f" -> {merged_tok!r:<15s} (freq={best_freq})")

    if verbose:
        print(f"  Training complete. Final vocab size: {len(token_to_id)}")

    return token_to_id, merge_rules


# Test Exercise 2
print("EXERCISE 2: Pure Python BPE Training on Legal Corpus")
print("-" * 40)
try:
    print("  Training BPE tokenizer...")
    token_to_id, merge_rules = train_pure_python_bpe(
        LEGAL_CORPUS,
        vocab_size=150,
        min_frequency=1,
        verbose=True,
    )
    print()
    print(f"  Learned {len(merge_rules)} merge rules")
    print()
    if len(merge_rules) >= 5:
        print("  First 5 merge rules (what BPE learned first):")
        for i, ((a, b), merged) in enumerate(merge_rules[:5], 1):
            print(f"    Rule {i}: ('{a}' + '{b}') -> '{merged}'")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Check the 4 TODOs inside the training loop.")

print()


# ============================================================
#  EXERCISE 3
#  Topic: Measure vocabulary coverage on in-domain vs out-of-domain text
#
#  Background:
#    A tokenizer trained on legal text should tokenize legal text well
#    (high coverage, few [UNK]) but may struggle with medical text
#    (different vocabulary -> more [UNK] or more subword splits).
#
#    Coverage = (total_tokens - UNK_tokens) / total_tokens * 100
#
#  C# Analogy:
#    double coverage = tokenIds
#      .Where(id => id != UNK_ID)
#      .Count() / (double)tokenIds.Count * 100;
# ============================================================

def bpe_encode_with_ids(text, merge_rules, token_to_id):
    """
    Encode text using trained BPE merge rules, returning token IDs.

    Args:
        text (str): Input text
        merge_rules (list): Ordered merge rules from training
        token_to_id (dict): Final vocabulary {token: id}

    Returns:
        list[int]: Token IDs
    """
    unk_id = token_to_id.get("[UNK]", 1)
    all_ids = []
    for word in text.lower().split():
        syms = list(word) + ["</w>"]
        for (pair_a, pair_b), merged in merge_rules:
            i = 0
            new_syms = []
            while i < len(syms):
                if (i < len(syms)-1 and syms[i] == pair_a and syms[i+1] == pair_b):
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms
        all_ids.extend([token_to_id.get(s, unk_id) for s in syms])
    return all_ids


def measure_coverage(test_texts, merge_rules, token_to_id):
    """
    Measure what fraction of tokens are known (not [UNK]).

    For each text:
    1. Encode it using the trained tokenizer
    2. Count total tokens and [UNK] tokens
    3. Return coverage statistics

    Args:
        test_texts (list[str]): Texts to evaluate
        merge_rules (list): Trained merge rules
        token_to_id (dict): Vocabulary

    Returns:
        dict: {
            "total"    : int,
            "unknown"  : int,
            "coverage" : float (0-100),
        }
    """
    unk_id = token_to_id.get("[UNK]", 1)
    total = 0
    unknown = 0

    for text in test_texts:
        ids = bpe_encode_with_ids(text, merge_rules, token_to_id)
        # TODO: Add len(ids) to total and count UNK occurrences
        # Hint: ids.count(unk_id) gives number of UNK tokens
        # YOUR CODE HERE
        pass                                    # remove this

    # TODO: Compute coverage percentage
    coverage = None                             # replace None
    if total > 0:
        # YOUR CODE HERE
        pass                                    # replace this

    return {"total": total, "unknown": unknown, "coverage": coverage}


# Test Exercise 3
print("EXERCISE 3: Vocabulary Coverage Analysis")
print("-" * 40)
try:
    # We need the trained tokenizer from Exercise 2
    # Run training again quietly if needed
    tok_id, merges = train_pure_python_bpe(LEGAL_CORPUS, vocab_size=150,
                                           min_frequency=1, verbose=False)

    # In-domain: legal text (similar to training data)
    in_domain = [
        "the court shall issue its judgment within sixty days",
        "breach of contract entitles the plaintiff to damages",
        "the arbitration clause governs all disputes between parties",
    ]

    # Out-of-domain: medical text (very different vocabulary)
    out_domain = [
        "the patient underwent cardiac catheterization successfully",
        "electrocardiogram revealed atrial fibrillation requiring treatment",
        "troponin levels elevated indicating myocardial injury",
    ]

    in_stats  = measure_coverage(in_domain, merges, tok_id)
    out_stats = measure_coverage(out_domain, merges, tok_id)

    print("  In-domain (legal) text:")
    if in_stats["coverage"] is not None:
        print(f"    Total tokens:   {in_stats['total']}")
        print(f"    Unknown tokens: {in_stats['unknown']}")
        print(f"    Coverage:       {in_stats['coverage']:.1f}%")

    print()
    print("  Out-of-domain (medical) text:")
    if out_stats["coverage"] is not None:
        print(f"    Total tokens:   {out_stats['total']}")
        print(f"    Unknown tokens: {out_stats['unknown']}")
        print(f"    Coverage:       {out_stats['coverage']:.1f}%")

    print()
    if in_stats["coverage"] and out_stats["coverage"]:
        diff = in_stats["coverage"] - out_stats["coverage"]
        print(f"  Coverage difference: {diff:.1f}%")
        print(f"  Legal tokenizer works better on legal text!")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Check measure_coverage() TODOs.")

print()


# ============================================================
#  EXERCISE 4
#  Topic: Train with HuggingFace BpeTrainer (if available)
#
#  Background:
#    The HuggingFace tokenizers library provides the same BPE algorithm
#    but much faster (Rust backend) and with more features.
#    API: Tokenizer(BPE()) -> set pre_tokenizer -> BpeTrainer -> .train()
#
#  C# Analogy:
#    var tokenizer = new HuggingFaceTokenizer(new BpeModel());
#    tokenizer.PreTokenizer = new WhitespacePreTokenizer();
#    var trainer = new BpeTrainer { VocabSize = 500 };
#    tokenizer.Train(new[] { "corpus.txt" }, trainer);
# ============================================================

print("EXERCISE 4: HuggingFace BpeTrainer (if installed)")
print("-" * 40)
print()

if HF_AVAILABLE:
    # Write corpus to file (HF trainer reads from files)
    corpus_path = "legal_corpus_ex05.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in LEGAL_CORPUS:
            f.write(line + "\n")

    print(f"  Corpus written to {corpus_path} ({len(LEGAL_CORPUS)} lines)")
    print()

    # TODO 1: Create an empty BPE tokenizer
    # Hint: Tokenizer(BPE(unk_token="[UNK]"))
    hf_tokenizer = None                     # replace None
    # YOUR CODE HERE
    pass                                    # remove this

    if hf_tokenizer is not None:
        # TODO 2: Set the pre-tokenizer to Whitespace()
        # Hint: hf_tokenizer.pre_tokenizer = Whitespace()
        # YOUR CODE HERE
        pass                                # remove this

        # TODO 3: Create a BpeTrainer
        # Parameters:
        #   special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        #   vocab_size=500
        #   min_frequency=1
        #   show_progress=False
        hf_trainer = None                   # replace None
        # YOUR CODE HERE
        pass                                # remove this

        if hf_trainer is not None:
            # TODO 4: Train the tokenizer on the corpus file
            # Hint: hf_tokenizer.train([corpus_path], hf_trainer)
            # YOUR CODE HERE
            pass                            # remove this

            final_vocab_size = hf_tokenizer.get_vocab_size()
            print(f"  HuggingFace BPE trained! Vocab size: {final_vocab_size}")
            print()

            # Test encoding
            test_texts_hf = [
                "the court shall hear the dispute",
                "breach of contract damages",
                "arbitration clause governs",
            ]

            print("  HuggingFace tokenizer output:")
            for text in test_texts_hf:
                output = hf_tokenizer.encode(text)
                print(f"    '{text}'")
                print(f"    tokens: {output.tokens}")
                print()

            # Compare with pure Python version
            print("  Comparing HF vs Pure Python tokenizer:")
            compare_text = "the court shall issue judgment"
            hf_tokens = hf_tokenizer.encode(compare_text).tokens

            py_tok_id, py_merges = train_pure_python_bpe(
                LEGAL_CORPUS, vocab_size=150, min_frequency=1, verbose=False
            )
            py_tokens_ids = bpe_encode_with_ids(compare_text, py_merges, py_tok_id)
            id_to_tok = {v: k for k, v in py_tok_id.items()}
            py_tokens = [id_to_tok.get(i, "[UNK]") for i in py_tokens_ids]

            print(f"  Text: '{compare_text}'")
            print(f"  HF tokenizer:  {hf_tokens}")
            print(f"  Pure Python:   {py_tokens}")
            print(f"  (Results differ because HF uses different vocab_size and pre-tokenizer)")

    # Cleanup
    if os.path.exists(corpus_path):
        os.remove(corpus_path)

else:
    print("  (Skipping - tokenizers library not installed)")
    print()
    print("  Install with: pip install tokenizers")
    print()
    print("  Code that would run:")
    print("  from tokenizers import Tokenizer")
    print("  from tokenizers.models import BPE")
    print("  from tokenizers.trainers import BpeTrainer")
    print("  from tokenizers.pre_tokenizers import Whitespace")
    print()
    print("  hf_tokenizer = Tokenizer(BPE(unk_token='[UNK]'))")
    print("  hf_tokenizer.pre_tokenizer = Whitespace()")
    print("  trainer = BpeTrainer(vocab_size=500, special_tokens=['[PAD]','[UNK]','[CLS]'])")
    print("  hf_tokenizer.train(['legal_corpus.txt'], trainer)")
    print("  output = hf_tokenizer.encode('breach of contract')")
    print("  print(output.tokens)  # ['breach', 'of', 'cont', '##ract']")

print()


# ============================================================
#  EXERCISE 5
#  Topic: Design decisions for custom tokenizer
#
#  Background:
#    When training a custom tokenizer, you make several design choices:
#    - vocab_size: how many tokens?
#    - min_frequency: filter rare tokens?
#    - special tokens: which ones do you need?
#    - algorithm: BPE or WordPiece?
#
#    This exercise asks you to analyze the trade-offs and fill in
#    a design recommendation for a given scenario.
#
#  C# Analogy:
#    Like choosing between compression algorithms:
#    - Small vocab (DEFLATE) vs large vocab (LZMA)
#    - Speed vs compression ratio
# ============================================================

def design_tokenizer(scenario):
    """
    Return tokenizer design recommendations for a given scenario.

    For each scenario, fill in:
    - algorithm: "BPE" or "WordPiece"
    - vocab_size: an integer
    - min_frequency: an integer
    - reason: a short explanation string

    Args:
        scenario (str): One of "medical", "code", "multilingual", "general"

    Returns:
        dict: {"algorithm": str, "vocab_size": int, "min_frequency": int, "reason": str}
    """
    # TODO: Return appropriate recommendations for each scenario
    # Think about: domain size, character diversity, typical word lengths

    if scenario == "medical":
        # Medical text: specialized vocabulary, English only, ~10MB of text
        return {
            "algorithm"    : None,          # "BPE" or "WordPiece"?
            "vocab_size"   : None,          # 8000? 32000? 100000?
            "min_frequency": None,          # 1? 2? 5?
            "reason"       : None,          # explain your choices
        }

    elif scenario == "code":
        # Python/Java code: lots of special chars ({}, (), ==, +=), identifiers
        return {
            "algorithm"    : None,          # ByteLevel BPE is common for code
            "vocab_size"   : None,          # identifiers vary a lot
            "min_frequency": None,
            "reason"       : None,
        }

    elif scenario == "multilingual":
        # 50 languages: needs to cover diverse scripts and character sets
        return {
            "algorithm"    : None,          # SentencePiece BPE handles all languages
            "vocab_size"   : None,          # needs to be large for 50 languages
            "min_frequency": None,
            "reason"       : None,
        }

    elif scenario == "general":
        # General English: Wikipedia + books, standard NLP tasks
        return {
            "algorithm"    : None,          # either BPE or WordPiece works
            "vocab_size"   : None,          # GPT-2 uses 50257, BERT uses 30522
            "min_frequency": None,
            "reason"       : None,
        }

    else:
        return {"error": f"Unknown scenario: {scenario}"}


# Test Exercise 5
print("EXERCISE 5: Tokenizer Design Decisions")
print("-" * 40)

scenarios = ["medical", "code", "multilingual", "general"]

for scenario in scenarios:
    design = design_tokenizer(scenario)
    print(f"  Scenario: {scenario.upper()}")

    if "error" in design:
        print(f"    ERROR: {design['error']}")
    elif design["algorithm"] is None:
        print(f"    [TODO] Fill in design_tokenizer('{scenario}')")
    else:
        print(f"    Algorithm:     {design['algorithm']}")
        print(f"    Vocab size:    {design['vocab_size']:,}")
        print(f"    Min frequency: {design['min_frequency']}")
        print(f"    Reason:        {design['reason']}")
    print()


print("=" * 60)
print("Exercise 05 Complete (or attempted)!")
print()
print("Expected outputs:")
print("  Ex1: total_words~200, unique_words~100, top words: 'the','of','contract'")
print("  Ex2: ~100 merge rules learned, first merges: 'th','he','er','on'...")
print("  Ex3: legal coverage > 90%, medical coverage < 50%")
print("  Ex4: HF tokenizer tokenizes legal terms into fewer pieces than pure Python")
print("  Ex5: medical->BPE/8K, code->BPE/16K, multilingual->BPE/100K, general->BPE/32K")
print("=" * 60)

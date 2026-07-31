"""
Module 05.5 - HuggingFace Tokenizers
Exercise 03: Tokenizer Comparison

GLOSSARY
--------
BPE          : Byte Pair Encoding. Space is PART of the token (" world" = one token).
               C# analogy: like keeping spaces inside serialized string values.

WordPiece    : BERT's method. Spaces are DISCARDED. Continuation tokens get "##" prefix.
               C# analogy: like XML tokens where separators are stripped.

SentencePiece: Spaces encoded as "_" prefix in tokens. Truly language-agnostic.
               C# analogy: like URL encoding where spaces become underscores.

Token Count  : Number of tokens a text produces with a given tokenizer.
               Fewer tokens = shorter sequence = faster attention computation.

Continuation : A token that continues the previous word (not word-initial).
               In WordPiece: "##ning" means "ning" continues previous word.
               C# analogy: like a StringBuilder.Append() vs new StringBuilder().

Round-trip   : Encode then decode should recover the original text.
               Not always perfect: some tokenizers normalize (lowercase, strip spaces).
               C# analogy: JsonSerializer.Deserialize(JsonSerializer.Serialize(obj)) == obj
"""

print("=" * 60)
print("Exercise 03: Tokenizer Comparison")
print("=" * 60)
print()

# Try to import HuggingFace
try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
    print("[OK] transformers library available")
except ImportError:
    HF_AVAILABLE = False
    print("[INFO] transformers not installed. Using pure Python simulations.")
    print("[INFO] Install: pip install transformers")

print()


# ============================================================
# Helper: Pure Python Tokenizer Simulations
# (These provide consistent test behavior whether or not HF is installed)
# ============================================================

def simulate_bpe(text):
    """
    Simulate GPT-2 style BPE tokenization.
    Space is part of the FOLLOWING token (attached, not separate).
    Common words stay whole; less common words may split by suffix.
    """
    tokens = []
    words = text.split(" ")                             # split on spaces

    for i, word in enumerate(words):
        if not word:                                    # skip empty strings
            continue
        prefix = "" if i == 0 else " "                 # space before all but first word
        word_lower = word.lower()

        # Simulate what BPE would learn: common suffixes become separate tokens
        known_suffixes = ["ing", "ness", "er", "ed", "ly", "tion", "ment"]
        split_done = False
        for suffix in known_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 1:
                stem = word[:len(word)-len(suffix)]     # word without suffix
                tokens.append(prefix + stem)            # stem (with leading space)
                tokens.append(suffix)                   # suffix as separate token
                split_done = True
                break
        if not split_done:
            tokens.append(prefix + word)                # whole word with space prefix

    return [t for t in tokens if t]                     # filter empty strings


def simulate_wordpiece(text):
    """
    Simulate BERT WordPiece tokenization.
    Spaces are discarded. Continuation pieces get "##" prefix.
    Common suffixes get "##" prefix; word stems stay as-is.
    """
    tokens = []
    words = text.lower().split()                        # split + lowercase, spaces gone

    for word in words:
        known_suffixes = ["ing", "ness", "er", "ed", "ly", "tion", "ment", "s"]
        split_done = False
        for suffix in known_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                stem = word[:len(word)-len(suffix)]
                tokens.append(stem)                     # stem (no prefix)
                tokens.append("##" + suffix)            # continuation with ##
                split_done = True
                break
        if not split_done:
            tokens.append(word)                         # whole word, no prefix

    return tokens


def simulate_sentencepiece(text):
    """
    Simulate SentencePiece tokenization.
    Spaces become "_" prefix on the following token.
    All tokens are otherwise the same as character content.
    """
    tokens = []
    words = text.split(" ")                             # split but keep track of spaces

    for i, word in enumerate(words):
        if not word:
            continue
        if i == 0:
            tokens.append(word.lower())                 # first word: no space prefix
        else:
            tokens.append("_" + word.lower())           # subsequent words: _ prefix

    return tokens


# ============================================================
#  EXERCISE 1
#  Topic: Compare token counts across tokenizer types
#
#  Background:
#    Different tokenizers produce different numbers of tokens for the same text.
#    Fewer tokens = shorter sequence = faster model computation (attention is O(n^2)).
#    The "right" tokenizer for your domain produces shorter sequences on your data.
#
#  Your Task:
#    Complete compare_token_counts(text, tokenizers) that returns a dict
#    mapping each tokenizer name to its token count for the given text.
#
#  C# Analogy:
#    var counts = tokenizers.ToDictionary(
#      t => t.Name,
#      t => t.Tokenize(text).Length
#    );
# ============================================================

def compare_token_counts(text, tokenizers):
    """
    Tokenize the same text with multiple tokenizers and count the tokens.

    Args:
        text (str): Input text to tokenize
        tokenizers (dict): {name: callable} where callable(text) -> list of tokens

    Returns:
        dict: {name: token_count}
    """
    # TODO: For each tokenizer, call it on the text and record the count
    # Hint: len(tokenizer_fn(text)) gives the token count
    results = {}
    for name, tokenizer_fn in tokenizers.items():
        # YOUR CODE HERE
        pass                                        # remove this

    return results


# Test Exercise 1
print("EXERCISE 1: Compare Token Counts Across Tokenizers")
print("-" * 40)
try:
    tokenizers = {
        "BPE (simulated)"         : simulate_bpe,
        "WordPiece (simulated)"   : simulate_wordpiece,
        "SentencePiece (simulated)": simulate_sentencepiece,
    }

    test_texts = [
        "I love running in the morning",
        "Electrocardiogram results were abnormal",
        "The quick brown fox jumps over the lazy dog",
        "Transformers use self-attention mechanisms",
    ]

    print("  Token count comparison:")
    print(f"  {'Text':<45} {'BPE':>6} {'WP':>6} {'SP':>6}")
    print("  " + "-" * 65)

    for text in test_texts:
        counts = compare_token_counts(text, tokenizers)
        bpe_count = counts.get("BPE (simulated)", "?")
        wp_count  = counts.get("WordPiece (simulated)", "?")
        sp_count  = counts.get("SentencePiece (simulated)", "?")
        print(f"  {text[:43]:<45} {bpe_count:>6} {wp_count:>6} {sp_count:>6}")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure to call tokenizer_fn(text) and store len() of result.")

print()


# ============================================================
#  EXERCISE 2
#  Topic: Demonstrate the "##" prefix in WordPiece
#
#  Background:
#    WordPiece marks continuation tokens with "##".
#    "running" -> ["run", "##ning"]
#    This lets the model know:
#    - "run" is a word-initial piece
#    - "##ning" continues the previous token (no space before it)
#
#    The "##" is removed during decoding: ["run", "##ning"] -> "running"
#
#  Your Task:
#    Complete decode_wordpiece(tokens) that joins tokens correctly.
#    - If a token starts with "##", remove "##" and join to previous (no space)
#    - Otherwise, add a space before the token (except for the first token)
#
#  C# Analogy:
#    var sb = new StringBuilder();
#    foreach (var token in tokens) {
#        if (token.StartsWith("##")) sb.Append(token.Substring(2));
#        else { if (sb.Length > 0) sb.Append(' '); sb.Append(token); }
#    }
# ============================================================

def decode_wordpiece(tokens):
    """
    Decode a list of WordPiece tokens back to a string.

    Rules:
    - If token starts with "##": remove the "##" and join directly to previous token
    - Otherwise: if not the first token, add a space before it

    Args:
        tokens (list[str]): WordPiece tokens (may include "##" continuation markers)

    Returns:
        str: Decoded text
    """
    result = ""                             # accumulate decoded text here

    for token in tokens:
        # TODO: Handle "##" tokens vs regular tokens
        # YOUR CODE HERE
        pass                                # remove this

    return result


# Test Exercise 2
print("EXERCISE 2: WordPiece Decode with ## Markers")
print("-" * 40)
try:
    # Test cases: various WordPiece token sequences
    test_cases = [
        (["run", "##ning", "cat", "##s"],           "running cats"),
        (["un", "##happi", "##ness"],                "unhappiness"),
        (["the", "quick", "brown", "fox"],           "the quick brown fox"),
        (["transform", "##er", "##s"],               "transformers"),
    ]

    print("  WordPiece decoding test cases:")
    all_pass = True
    for tokens, expected in test_cases:
        decoded = decode_wordpiece(tokens)
        status = "PASS" if decoded == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {status}: {tokens}")
        print(f"       -> '{decoded}' (expected: '{expected}')")
        print()

    if all_pass:
        print("  All test cases passed!")
    else:
        print("  Some test cases failed. Check ## handling logic.")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure to handle both '##' and regular tokens.")

print()


# ============================================================
#  EXERCISE 3
#  Topic: Demonstrate SentencePiece space encoding
#
#  Background:
#    SentencePiece encodes spaces as "_" prefix.
#    " hello" (space + hello) becomes "_hello".
#    Decoding: replace "_" with " ", then strip leading space.
#
#    This preserves spacing information in the tokens themselves.
#    BPE keeps spaces; WordPiece loses spaces; SentencePiece encodes as "_".
#
#  C# Analogy:
#    var decoded = string.Concat(tokens.Select(t => t.Replace("_", " "))).TrimStart();
# ============================================================

def encode_sentencepiece_style(text):
    """
    Encode text in SentencePiece style: spaces become "_" prefix on next word.

    Example:
      "Hello world"  -> ["Hello", "_world"]
      "I love cats"  -> ["I", "_love", "_cats"]

    Args:
        text (str): Raw input text

    Returns:
        list[str]: Tokens with "_" for space prefix
    """
    # TODO: Split on spaces, add "_" prefix to all words except the first
    tokens = []
    words = text.split(" ")                 # split on spaces

    for i, word in enumerate(words):
        if not word:                        # skip empty strings from double-spaces
            continue
        # TODO: first word: no prefix, subsequent words: "_" prefix
        # YOUR CODE HERE
        pass                                # remove this

    return tokens


def decode_sentencepiece_style(tokens):
    """
    Decode SentencePiece-style tokens back to text.
    Replace "_" prefix with a space character.

    Example:
      ["Hello", "_world"]  -> "Hello world"
      ["I", "_love", "_cats"] -> "I love cats"

    Args:
        tokens (list[str]): SentencePiece-style tokens

    Returns:
        str: Decoded text
    """
    # TODO: Join tokens, replacing "_" with " ", then strip leading space
    # Hint: "".join(tokens) first, then .replace("_", " ").strip()
    # YOUR CODE HERE
    pass                                    # remove this


# Test Exercise 3
print("EXERCISE 3: SentencePiece Space Encoding/Decoding")
print("-" * 40)
try:
    test_texts = [
        "Hello world",
        "I love Python programming",
        "The quick brown fox",
    ]

    print("  SentencePiece encode/decode round-trips:")
    for text in test_texts:
        encoded = encode_sentencepiece_style(text)
        decoded = decode_sentencepiece_style(encoded)
        round_trip_ok = decoded == text

        print(f"  Original:  '{text}'")
        print(f"  Encoded:    {encoded}")
        print(f"  Decoded:   '{decoded}'")
        print(f"  Round-trip: {'PASS' if round_trip_ok else 'FAIL'}")
        print()

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Check that encode adds '_' prefix and decode replaces '_' with space.")

print()


# ============================================================
#  EXERCISE 4
#  Topic: Tokenizer round-trip fidelity check
#
#  Background:
#    A tokenizer should be able to reconstruct the original text from token IDs.
#    This "round-trip" fidelity is important for text generation.
#    Different tokenizers have different fidelity:
#    - BPE: spaces preserved (high fidelity)
#    - WordPiece: lowercase + no spaces (lower fidelity without post-processing)
#    - SentencePiece: spaces encoded as _ (high fidelity after decode)
#
#  C# Analogy:
#    bool fidelityOk = originalText == Decode(Encode(originalText));
# ============================================================

def check_round_trip(text, encode_fn, decode_fn):
    """
    Check if encoding then decoding recovers the original text.

    Args:
        text (str): Original text
        encode_fn (callable): text -> list of tokens
        decode_fn (callable): list of tokens -> text

    Returns:
        dict: {"encoded": tokens, "decoded": str, "fidelity": float (0-1)}
    """
    encoded = encode_fn(text)               # text -> tokens
    decoded = decode_fn(encoded)            # tokens -> text

    # Compute simple character-level fidelity
    # (how many characters match between original and decoded)
    # TODO: compute fidelity as fraction of characters that match
    # Simple approach: check if they are equal
    # Advanced: Levenshtein distance (we use simple equality here)

    # TODO: Return the result dict
    # fidelity should be 1.0 if decoded == text.lower().strip(), 0.0 otherwise
    # (WordPiece lowercases, so compare to lowercase)

    # YOUR CODE HERE
    pass                                    # remove this - return the dict


# Simple decode functions for BPE and SentencePiece
def decode_bpe_style(tokens):
    """Decode BPE style tokens: join directly (spaces are inside tokens)."""
    return "".join(tokens).lstrip()         # join and remove leading space


# Test Exercise 4
print("EXERCISE 4: Round-Trip Fidelity Check")
print("-" * 40)
try:
    test_texts = [
        "Running is good for health",
        "The cat sat on the mat",
        "Python programming is fun",
    ]

    tokenizer_pairs = [
        ("BPE",          simulate_bpe,          decode_bpe_style),
        ("WordPiece",    simulate_wordpiece,     decode_wordpiece),
        ("SentencePiece",simulate_sentencepiece, decode_sentencepiece_style),
    ]

    print("  Round-trip fidelity check:")
    print()

    for text in test_texts[:2]:             # test on 2 sentences for brevity
        print(f"  Text: '{text}'")
        for name, enc_fn, dec_fn in tokenizer_pairs:
            result = check_round_trip(text, enc_fn, dec_fn)
            if result is not None:
                fid = result.get("fidelity", "?")
                dec = result.get("decoded", "?")
                print(f"    {name:<15} decoded='{dec}' fidelity={fid}")
        print()

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure check_round_trip returns a dict with 'encoded','decoded','fidelity'.")

print()


# ============================================================
#  EXERCISE 5
#  Topic: Find the best tokenizer for a given domain
#
#  Background:
#    Different tokenizers produce different token counts for different domains.
#    For code: common programming keywords should be single tokens.
#    For medical: domain terms should not be over-split.
#    For general English: any modern tokenizer works well.
#
#    "Best" = fewest tokens (shorter sequences = faster model computation)
#    BUT also: tokens should be meaningful (not random character splits).
#
#  C# Analogy:
#    var bestTokenizer = tokenizers
#      .OrderBy(t => t.Tokenize(domainText).Average(seq => seq.Count()))
#      .First();
# ============================================================

def find_best_tokenizer(domain_corpus, tokenizers):
    """
    Find which tokenizer produces the fewest tokens on average for a corpus.

    Steps:
    1. For each tokenizer, tokenize all texts in the corpus.
    2. Compute average token count per text.
    3. Return the tokenizer name with the lowest average count.

    Args:
        domain_corpus (list[str]): Domain-specific text samples
        tokenizers (dict): {name: callable}

    Returns:
        tuple: (best_name, avg_counts_dict)
               avg_counts_dict: {name: avg_token_count}
    """
    avg_counts = {}

    for name, tok_fn in tokenizers.items():
        # TODO 1: Tokenize all texts and compute average token count
        total_tokens = 0
        for text in domain_corpus:
            tokens = tok_fn(text)           # tokenize this text
            # TODO: add len(tokens) to total_tokens
            # YOUR CODE HERE
            pass                            # remove this

        # TODO 2: Compute average
        avg = None                          # replace None
        if len(domain_corpus) > 0:
            # YOUR CODE HERE
            pass                            # remove this

        avg_counts[name] = avg

    # TODO 3: Find the name with the minimum average count
    # Hint: min(avg_counts, key=avg_counts.get) finds key with min value
    best_name = None                        # replace None
    # YOUR CODE HERE
    pass                                    # remove this

    return best_name, avg_counts


# Test Exercise 5
print("EXERCISE 5: Best Tokenizer for a Domain")
print("-" * 40)
try:
    # Medical domain corpus
    medical_corpus = [
        "electrocardiogram showed normal sinus rhythm",
        "the cardiologist recommended coronary angiography",
        "troponin levels indicated myocardial infarction",
        "echocardiogram revealed left ventricular dysfunction",
        "anticoagulation therapy was initiated immediately",
    ]

    # General English corpus
    general_corpus = [
        "the cat sat on the mat",
        "I love to run in the park",
        "she went to the store yesterday",
        "he is a very fast runner",
        "the dog barked at the mailman",
    ]

    tokenizers = {
        "BPE (simulated)"         : simulate_bpe,
        "WordPiece (simulated)"   : simulate_wordpiece,
        "SentencePiece (simulated)": simulate_sentencepiece,
    }

    print("  Medical domain:")
    best_med, avg_med = find_best_tokenizer(medical_corpus, tokenizers)
    if best_med is not None:
        for name, avg in sorted(avg_med.items(), key=lambda x: x[1] or 999):
            print(f"    {name:<35} avg tokens: {avg:.1f if avg else '?'}")
        print(f"    Best: {best_med}")

    print()
    print("  General English:")
    best_gen, avg_gen = find_best_tokenizer(general_corpus, tokenizers)
    if best_gen is not None:
        for name, avg in sorted(avg_gen.items(), key=lambda x: x[1] or 999):
            print(f"    {name:<35} avg tokens: {avg:.1f if avg else '?'}")
        print(f"    Best: {best_gen}")

except Exception as e:
    print(f"  [ERROR] {e}")
    print(f"  Make sure to sum token counts and divide by corpus size.")


print()
print("=" * 60)
print("Exercise 03 Complete (or attempted)!")
print()
print("Expected outputs:")
print("  Ex1: BPE and SentPiece counts are similar; WordPiece may differ for long words")
print("  Ex2: 'run' + '##ning' -> 'running' (no space, ## removed)")
print("  Ex3: 'Hello world' -> ['Hello','_world'] -> 'Hello world' (round-trip OK)")
print("  Ex4: BPE and SentPiece have high fidelity; WordPiece lowercases")
print("  Ex5: Results vary - the best tokenizer depends on word length in domain")
print("=" * 60)

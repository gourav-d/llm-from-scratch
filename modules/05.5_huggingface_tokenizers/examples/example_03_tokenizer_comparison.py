"""
Module 05.5 - HuggingFace Tokenizers
Example 03: Comparing BPE, WordPiece, and SentencePiece

GLOSSARY
--------
BPE          : Byte Pair Encoding. GPT-2/GPT-4/LLaMA tokenizer type.
               Merges most frequent character pairs to build subword vocabulary.
               Space handling: space is PART of the following token ("_hello").

WordPiece    : BERT's tokenizer algorithm.
               Like BPE but selects merges based on likelihood gain, not raw count.
               Continuation tokens marked with ##: "running" -> ["run", "##ning"]

SentencePiece: Algorithm by Google. Treats raw bytes including spaces as input.
               No pre-tokenization step needed. Language-agnostic.
               Space token shown as _ (underscore prefix in vocab).

tiktoken     : OpenAI's fast BPE tokenizer (Rust backend).
               Used by GPT-3.5, GPT-4, GPT-4o, LLaMA 3.
               Very fast, byte-level BPE with ~100K vocabulary.

This example demonstrates all three approaches with:
1. Pure Python simulations (always work, no library needed)
2. HuggingFace library usage (if installed)

Differences are shown side-by-side on the same test texts.
"""

print("=" * 60)
print("Example 03: Tokenizer Comparison (BPE vs WordPiece vs SentencePiece)")
print("=" * 60)
print()

# Try to import optional libraries
# If not installed, we fall back to pure Python simulation

try:
    from transformers import AutoTokenizer    # high-level HuggingFace API
    HF_TRANSFORMERS_AVAILABLE = True          # flag: transformers is installed
    print("[OK] transformers library is available (using real tokenizers)")
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False         # flag: transformers NOT installed
    print("[INFO] transformers not installed. Using pure Python simulations.")
    print("[INFO] To install: pip install transformers")

print()


# ============================================================
# PART A: Pure Python Simulations of All Three Methods
# ============================================================

print("PART A: Simulating All Three Methods (Pure Python)")
print("-" * 50)
print()

# ---- Simulated BPE Tokenizer ----
# This mimics GPT-2 style BPE:
# - Spaces attach to the FOLLOWING word (not separate tokens)
# - Common words are single tokens, rare words split into pieces

def simulated_bpe_tokenize(text):
    """
    Simulate GPT-2 style BPE tokenization.

    Key characteristics of GPT-2 BPE:
    - Space is part of the token: " hello" is one token (with leading space)
    - Written as "Gello" in GPT-2 vocab (G = special space character)
    - We simulate this by keeping spaces attached to the next word
    - Common word endings get their own tokens: "ing", "er", "ed", "s"

    C# analogy:
      This is like a custom string parser that splits on specific patterns.
      Regex.Split(@"(?=\s)") to keep spaces with following words.

    Args:
        text (str): Input text to tokenize

    Returns:
        list[str]: List of simulated BPE tokens
    """
    # Pre-defined "learned" merges (simulating what BPE training would produce)
    # In real BPE: these would come from 50,000 merge rules learned on 40GB of text
    # Here: we hand-craft a small set for demonstration
    known_tokens = {
        # Common whole words (would be single tokens in real GPT-2 BPE)
        "the", "is", "a", "of", "and", "to", "in", "that",
        "run", "cat", "dog", "happy", "quick",

        # Common subword pieces (suffixes/prefixes learned by BPE)
        "ing", "er", "ed", "ly", "un", "re", "ness", "tion",

        # Common letter combinations
        "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd"
    }

    # Simple word splitting (GPT-2 uses complex regex but concept is same)
    words = []                                  # list to collect tokens
    # Split while keeping track of spaces (spaces go WITH following word)
    parts = text.split(" ")                     # split on spaces
    for i, part in enumerate(parts):            # enumerate gives (index, part)
        if i == 0:                              # first word: no leading space
            prefix = ""
        else:                                   # subsequent words: add space prefix
            prefix = " "                        # space attaches to THIS word (GPT-2 style)

        # For demonstration: check if word is known, else split character-by-character
        word_lower = part.lower()
        if word_lower in known_tokens:
            words.append(prefix + part)         # whole word as one token
        elif len(part) > 3:                     # longer unknown words: try to find suffix
            # Check for known suffixes
            found_suffix = False
            for suffix in ["ing", "ness", "tion", "er", "ed", "ly"]:
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix):
                    stem = part[:-len(suffix)]  # everything before the suffix
                    words.append(prefix + stem) # stem as one token
                    words.append(suffix)         # suffix as another token
                    found_suffix = True
                    break
            if not found_suffix:
                words.append(prefix + part)     # fallback: whole word as token
        else:
            words.append(prefix + part)         # short word: keep whole

    return [w for w in words if w]             # filter out empty strings


# ---- Simulated WordPiece Tokenizer ----
# This mimics BERT style:
# - Spaces separate words (pre-tokenization)
# - Continuation tokens get ## prefix
# - Unknown characters become [UNK]

def simulated_wordpiece_tokenize(text):
    """
    Simulate BERT-style WordPiece tokenization.

    Key characteristics:
    - Split text on whitespace first (pre-tokenize)
    - For each word: check if in vocab. If not, split into longest matching pieces.
    - Continuation pieces (not word-start) get "##" prefix.
    - Spaces are DISCARDED (not part of tokens).

    C# analogy:
      Like XML token parsing where each continuation element has an attribute:
      <token continuation="true">ning</token> is displayed as "##ning"

    Args:
        text (str): Input text to tokenize

    Returns:
        list[str]: List of simulated WordPiece tokens (with ## markers)
    """
    # Simulated vocabulary (what BERT's 30,522 vocab would contain)
    vocab = {
        # Common whole words
        "the", "is", "a", "of", "and", "to", "in", "that",
        "run", "cat", "dog", "quick", "i", "was",
        "happy", "sad",

        # Word-initial pieces (no ## prefix)
        "un", "re", "pre", "dis",

        # Continuation pieces (## prefix in real BERT)
        "running", "cats", "dogs",   # some words whole
        "##ning", "##s", "##er", "##ed", "##ly", "##ness",
        "##ing", "##tion", "##ment",

        # Character fallbacks
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    }

    words = text.lower().split()            # pre-tokenize: split on whitespace
    all_tokens = []                         # collect all tokens

    for word in words:
        # Try to tokenize this word using greedy longest-match
        tokens = wordpiece_split(word, vocab)
        all_tokens.extend(tokens)           # add to result list

    return all_tokens


def wordpiece_split(word, vocab):
    """
    Split a single word into WordPiece tokens using greedy longest-match.

    Algorithm:
    1. Try to match the longest prefix of the remaining text.
    2. If no match found, return ['[UNK]'].
    3. For each subsequent piece (after the first), add "##" prefix before looking up.

    Args:
        word (str): Single word to split
        vocab (set): Set of known token strings

    Returns:
        list[str]: Tokens (with ## markers for continuations)
    """
    if word in vocab:                       # check if whole word is in vocab
        return [word]                       # return as single token

    tokens = []                             # list to collect pieces
    start = 0                               # current start position in word

    while start < len(word):
        end = len(word)                     # start trying from longest possible
        found = False                       # did we find a match?

        while start < end:
            piece = word[start:end]         # slice from start to end
            if start > 0:                   # not the first piece?
                piece = "##" + piece        # add ## prefix for continuation tokens

            if piece in vocab:              # check if this piece is in vocab
                tokens.append(piece)        # add matched piece
                start = end                 # advance start to after the match
                found = True
                break                       # try to match next piece from start
            else:
                end -= 1                    # try shorter slice

        if not found:                       # no vocab match found at all
            return ["[UNK]"]               # return [UNK] for the whole word

    return tokens


# ---- Simulated SentencePiece Tokenizer ----
# Key difference: spaces are encoded as part of tokens (using _ prefix)
# No pre-tokenization needed

def simulated_sentencepiece_tokenize(text):
    """
    Simulate SentencePiece tokenization.

    Key characteristics:
    - Input text is processed as a raw stream (no pre-splitting on whitespace)
    - Spaces are represented as the "_" character INSIDE tokens
    - " hello" becomes "_hello" (underscore at start means there was a space)
    - Any byte sequence can be tokenized (truly language-agnostic)

    C# analogy:
      Like URL encoding where " " becomes "%20":
      string.Replace(" ", "_").Split('_').Select(w => "_" + w)
      Except we keep the underscore as part of the token.

    Args:
        text (str): Raw input text

    Returns:
        list[str]: SentencePiece-style tokens with _ for spaces
    """
    # Replace spaces with the SentencePiece space symbol (we use _ for display)
    # In real SentencePiece: uses the Unicode character U+2581 (Lower One Eighth Block)
    # We use simple "_" for clarity
    text_with_markers = text.replace(" ", " _")
    # " " + "_" = the start of a new word with space marker
    # "hello world" -> "hello _world"

    # Split into rough units
    parts = text_with_markers.split()   # split on whitespace (the actual spaces)
    tokens = []

    for part in parts:
        if part.startswith("_"):        # this piece was preceded by a space
            tokens.append(part)         # keep with _ prefix: "_world"
        else:                           # first word (no preceding space)
            tokens.append(part)         # keep as-is

    # Simulate subword splitting for long tokens
    result = []
    for token in tokens:
        if len(token) <= 6:             # short tokens: keep whole
            result.append(token)
        else:                           # longer tokens: split at midpoint
            mid = len(token) // 2       # integer division for midpoint
            result.append(token[:mid])  # first half
            result.append(token[mid:])  # second half

    return result


# Test all three simulated tokenizers on the same texts
print("Comparing simulated tokenizers on the same texts:")
print()

test_sentences = [
    "running cats are unhappy",
    "the quick brown fox",
    "pre-processing data",
]

for sentence in test_sentences:
    bpe_tokens = simulated_bpe_tokenize(sentence)
    wp_tokens  = simulated_wordpiece_tokenize(sentence)
    sp_tokens  = simulated_sentencepiece_tokenize(sentence)

    print(f"Text: '{sentence}'")
    print(f"  BPE        : {bpe_tokens}")
    print(f"  WordPiece  : {wp_tokens}")
    print(f"  SentPiece  : {sp_tokens}")
    print()

print()


# ============================================================
# PART B: Key Difference - Space Handling
# ============================================================

print("PART B: Space Handling - The Core Difference")
print("-" * 50)
print()

# This is the most important practical difference between the three methods
print("The biggest practical difference: how spaces are handled.")
print()

# Show the three conventions using a simple example
text = "Hello world"
print(f"Input text: '{text}'")
print()

# BPE (GPT-2 style): space becomes part of the FOLLOWING token
print("BPE (GPT-2 style):")
print("  'Hello' = first token, no leading space")
print("  ' world' = second token, WITH leading space")
print("  Written in vocab file as: 'Hello' and 'Gworld' (G=space character)")
print("  Our simulation: ['Hello', ' world']")
print()

# WordPiece (BERT style): spaces are discarded during pre-tokenization
print("WordPiece (BERT style):")
print("  'Hello' = first token")
print("  'world' = second token (space was thrown away!)")
print("  The model CANNOT tell if 'world' had a space before it")
print("  Our simulation: ['hello', 'world']")
print()

# SentencePiece: space is encoded as _ prefix
print("SentencePiece style:")
print("  '_Hello' = first token (no leading space in _Hello? depends on config)")
print("  '_world' = second token with _ meaning 'space before this word'")
print("  The _ is part of the token ID - model can always reconstruct spacing")
print("  Our simulation: ['Hello', '_world']")
print()

# ASCII Diagram showing the difference
print("ASCII Diagram - How the same text looks in each format:")
print()
print("  Text:         'Hello world'")
print("  BPE:          ['Hello'] [' world']")
print("                              ^")
print("                          space is inside the token")
print()
print("  WordPiece:    ['Hello'] ['world']")
print("                  ^--- space between these is just GONE")
print()
print("  SentPiece:    ['Hello'] ['_world']")
print("                              ^")
print("                          underscore MEANS 'space was here'")
print()


# ============================================================
# PART C: Real HuggingFace Tokenizers (If Available)
# ============================================================

print("PART C: Real HuggingFace Tokenizers (if library available)")
print("-" * 50)
print()

if HF_TRANSFORMERS_AVAILABLE:
    # Use real GPT-2 and BERT tokenizers to show actual differences

    print("Loading real GPT-2 (BPE) tokenizer...")
    try:
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")  # load GPT-2 tokenizer
        print("[OK] GPT-2 tokenizer loaded")
    except Exception as e:
        print(f"[WARN] Could not load GPT-2: {e}")
        gpt2_tokenizer = None

    print("Loading real BERT (WordPiece) tokenizer...")
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # load BERT
        print("[OK] BERT tokenizer loaded")
    except Exception as e:
        print(f"[WARN] Could not load BERT: {e}")
        bert_tokenizer = None

    print()

    if gpt2_tokenizer and bert_tokenizer:
        # Compare on the same texts
        comparison_texts = [
            "running cats are unhappy",
            "electrocardiogram",
            "Hello, world!",
            "I love Python programming",
        ]

        print("Real tokenizer comparison:")
        print()
        for text in comparison_texts:
            # GPT-2 BPE tokens
            gpt2_tokens = gpt2_tokenizer.tokenize(text)
            gpt2_ids = gpt2_tokenizer.encode(text)

            # BERT WordPiece tokens
            bert_tokens = bert_tokenizer.tokenize(text)
            bert_ids = bert_tokenizer.encode(text, add_special_tokens=False)

            print(f"Text: '{text}'")
            print(f"  GPT-2 (BPE)      tokens: {gpt2_tokens}")
            print(f"  GPT-2 (BPE)      IDs:    {gpt2_ids}")
            print(f"  BERT  (WordPiece) tokens: {bert_tokens}")
            print(f"  BERT  (WordPiece) IDs:    {bert_ids}")
            print(f"  GPT-2 uses {len(gpt2_tokens)} tokens, BERT uses {len(bert_tokens)} tokens")
            print()

else:
    print("(Skipping - transformers library not installed)")
    print()
    print("To see real tokenizer output, install:")
    print("  pip install transformers")
    print()
    print("Then these calls would work:")
    print("  gpt2 = AutoTokenizer.from_pretrained('gpt2')")
    print("  bert = AutoTokenizer.from_pretrained('bert-base-uncased')")
    print("  gpt2.tokenize('running cats')  # ['running', 'Gcats']")
    print("  bert.tokenize('running cats')  # ['running', 'cats']")


# ============================================================
# PART D: Comparison Table Summary
# ============================================================

print("PART D: Feature Comparison Table")
print("-" * 50)
print()

# Print a formatted comparison table
print(f"{'Feature':<25} {'BPE':<20} {'WordPiece':<20} {'SentencePiece':<20}")
print("-" * 85)

# Define rows as tuples: (feature name, bpe, wordpiece, sentencepiece)
comparison_rows = [
    ("Merge criterion",     "Frequency (count)",  "Likelihood gain",    "Frequency (BPE variant)"),
    ("Pre-tokenize",        "Yes (on whitespace)", "Yes (on whitespace)","No (raw bytes)"),
    ("Space handling",      "Space in token",      "Space discarded",    "Space as _ prefix"),
    ("Subword prefix",      "None (uses Gsp)",     "## for non-initial", "None (uses _ for sp)"),
    ("Unknown words",       "Split to pieces",     "Split to pieces",    "Byte fallback"),
    ("Language",            "Best for English",    "Best for English",   "Any language"),
    ("Main models",         "GPT-2, LLaMA 3",     "BERT, DistilBERT",  "T5, LLaMA 2, ALBERT"),
    ("Typical vocab size",  "50,000 - 100,000",    "30,000 - 32,000",   "32,000 - 64,000"),
]

for row in comparison_rows:
    feature, bpe_val, wp_val, sp_val = row          # unpack tuple
    print(f"  {feature:<23} {bpe_val:<20} {wp_val:<20} {sp_val:<20}")

print()


# ============================================================
# PART E: Decode Comparison
# ============================================================

print("PART E: Decoding (Tokens -> Text) Comparison")
print("-" * 50)
print()

# Show how decoding works differently for each method
print("Decoding is method-specific because the tokens look different.")
print()

# BPE decode: remove Ġ (or space markers), concatenate
bpe_example_tokens = ["The", " quick", " brown", " fox"]
bpe_decoded = "".join(bpe_example_tokens)       # just concatenate (spaces inside tokens)
print(f"BPE tokens:    {bpe_example_tokens}")
print(f"BPE decoded:   '{bpe_decoded}'")
print()

# WordPiece decode: join with spaces (except for ## tokens, which join directly)
wp_example_tokens = ["The", "quick", "brown", "fox"]
wp_example_tokens2 = ["run", "##ning", "cat", "##s"]  # with continuation markers

def decode_wordpiece(tokens):
    """Decode WordPiece tokens: remove ## markers, join with spaces."""
    result = ""                                 # start with empty string
    for i, token in enumerate(tokens):         # loop with index
        if token.startswith("##"):              # continuation token
            result += token[2:]                 # strip ## and join directly (no space)
        elif i == 0:                            # first token
            result += token                     # no leading space
        else:                                   # regular token, not first
            result += " " + token               # add space before
    return result


wp_decoded  = decode_wordpiece(wp_example_tokens)
wp_decoded2 = decode_wordpiece(wp_example_tokens2)
print(f"WordPiece tokens:    {wp_example_tokens}")
print(f"WordPiece decoded:   '{wp_decoded}'")
print()
print(f"WordPiece tokens:    {wp_example_tokens2}")
print(f"WordPiece decoded:   '{wp_decoded2}'  (## removed, no space before ##)")
print()

# SentencePiece decode: replace _ with space
sp_example_tokens = ["_The", "_quick", "_brown", "_fox"]
sp_decoded = "".join(t.replace("_", " ") for t in sp_example_tokens).strip()
print(f"SentPiece tokens:    {sp_example_tokens}")
print(f"SentPiece decoded:   '{sp_decoded}'  (_ becomes space)")
print()


print("=" * 60)
print("Example 03 Complete!")
print()
print("Key Takeaways:")
print("  1. All three methods produce different token representations")
print("  2. Biggest difference: how spaces are handled")
print("     BPE: space in token | WordPiece: space gone | SentPiece: _ prefix")
print("  3. WordPiece uses ## for continuation pieces (BERT convention)")
print("  4. SentencePiece is language-agnostic (best for multilingual)")
print("  5. NEVER mix a model with the wrong tokenizer!")
print("=" * 60)

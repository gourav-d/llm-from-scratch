# small_tokenizer_standalone.py
# Trains a SentencePiece BPE tokenizer on a text file.
# Extracted from the Udemy "Small LLM" course notebook (small_tokenizer_official.ipynb).
# Runs without Jupyter Notebook.
#
# Usage:
#   python small_tokenizer_standalone.py
#   python small_tokenizer_standalone.py --vocab_size 8192 --input mydata.txt --model_prefix my_tokenizer
#
# Requirements:
#   pip install sentencepiece
#
# Files produced:
#   <model_prefix>.model  -- the tokenizer binary (load this to encode/decode)
#   <model_prefix>.vocab  -- human-readable vocabulary list

# Uncomment to install if needed:
# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece", "-q"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "ipdb", "-q"])

import argparse
import os
import sys
import requests
import zipfile
import io

import sentencepiece as spm


# ============================================================
# CONFIGURATION (command-line overrides these defaults)
# ============================================================
DEFAULT_VOCAB_SIZE = 4096          # Size of the vocabulary you wish to have
DEFAULT_INPUT_FILE = "wiki.txt"    # Training data file (plain text, one doc per line is fine)
DEFAULT_MODEL_PREFIX = "test_wiki_tokenizer"  # Output file name prefix
FILES_URL = "https://ideami.com/llm_train"    # URL to download wiki.txt + encoded_data.pt


def download_files_if_needed():
    """Download wiki.txt and encoded_data.pt if not already present."""
    # If you are running this online (for example at Google Colab),
    # make sure you have the support files on the same folder.
    # Otherwise run this function to download them.
    #
    # NOTE: Downloading will take a while, be patient.
    # You can refresh your folder from time to time to see when the files
    # have been created.
    if not os.path.exists("encoded_data.pt"):
        print("Downloading files using Python...")
        response = requests.get(FILES_URL)
        zipfile.ZipFile(io.BytesIO(response.content)).extractall(".")
        print("Download complete.")
    else:
        print("Files already downloaded. (Delete encoded_data.pt to re-download.)")


def train_tokenizer(input_file, model_prefix, vocab_size):
    """Train a SentencePiece BPE tokenizer on the given text file."""
    print(f"Training tokenizer on '{input_file}' with vocab_size={vocab_size}...")

    spm.SentencePieceTrainer.train(
        input=input_file,
        # pick the name for your trained tokenizer
        # This creates two files: <model_prefix>.model and <model_prefix>.vocab
        model_prefix=model_prefix,

        # model_type: algorithm used to build the tokenizer.
        # Options: "bpe", "unigram", "word", "char"
        # BPE = Byte Pair Encoding: iteratively merges the most frequent
        # character pairs until the desired vocabulary size is reached.
        # BPE splits rare words into common subword units, so the model
        # can generalize to unseen words.
        model_type="bpe",

        # vocab_size: total number of unique tokens in the vocabulary.
        # 4096 = 2^12. Small for a demo. GPT-2 uses 50257, LLaMA uses 32000.
        # Larger vocab = fewer tokens per sentence = faster training,
        # but needs more data to learn all token meanings.
        vocab_size=vocab_size,

        self_test_sample_size=0,    # Number of sentences to use for self-test (0 = skip)

        input_format="text",        # Input is plain text (not TSV/JSON)

        # character_coverage: proportion of characters in the training corpus
        # to include when building the tokenizer model.
        # Important for languages with large character sets (Japanese, Chinese).
        # 0.995 means the tokenizer includes the most frequent 99.5% of characters.
        # The remaining 0.5% of less frequent characters are treated as unknown.
        # Helps manage vocabulary size and improves efficiency.
        character_coverage=0.995,

        num_threads=os.cpu_count(),  # Use all available CPU cores for speed

        # split_digits: if True, "2024" becomes ["2", "0", "2", "4"]
        # Good for arithmetic and dates -- model sees individual digits, not mystery tokens
        split_digits=True,

        # allow_whitespace_only_pieces: if True, spaces can be their own tokens.
        # Helps preserve exact spacing when decoding back to text.
        allow_whitespace_only_pieces=True,

        # byte_fallback: if True, characters not in the vocabulary are encoded
        # as byte-level tokens (e.g., <0xC3><0xA9> for "e" with accent).
        # This means there is NO true unknown token -- every character can be encoded.
        # Without this, unusual characters become a single <unk> token, losing information.
        byte_fallback=True,

        # unk_surface: what the unknown token looks like when printed/decoded.
        # The default is a Unicode "flower" character (rare enough to not appear in text).
        # \342\201\207 is the octal encoding of that character.
        unk_surface=r" \342\201\207 ",

        # normalization_rule_name: controls text normalization before tokenization.
        # "identity" means NO normalization -- preserve the text exactly as-is.
        # Other options: "nmt_nfkc" (Unicode normalization, lowercasing, etc.)
        # We use "identity" so the tokenizer learns the actual casing/punctuation of the data.
        normalization_rule_name="identity"
    )

    print("Tokenizer training completed.")
    print(f"Files created: {model_prefix}.model and {model_prefix}.vocab")


def validate_tokenizer(model_prefix):
    """Load the trained tokenizer and test that encode/decode round-trips correctly."""
    print("\n--- Validating Tokenizer ---")

    # Load the trained model file
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    # Print vocabulary size (should match the vocab_size you requested)
    actual_vocab_size = sp.get_piece_size()
    print(f"SentencePiece vocab_size: {actual_vocab_size}")

    # Create simple encode/decode helper functions
    # encode: string -> list of integer token IDs
    # decode: list of integer token IDs -> string
    # These are Python lambda functions (like C# Func<string, List<int>> and Func<List<int>, string>)
    encode = lambda s: sp.Encode(s)
    decode = lambda l: sp.Decode(l)

    # Test with a sample sentence
    test_sentence = "What is a healthy dish that includes strawberry?"
    encoded = encode(test_sentence)
    decoded = decode(encoded)

    print(f"\nTest sentence:  {test_sentence}")
    print(f"Encoded tokens: {encoded}")
    print(f"Decoded back:   {decoded}")

    # Check round-trip accuracy
    if decoded == test_sentence:
        print("Encode/decode round-trip: PASS")
    else:
        print(f"Encode/decode round-trip: MISMATCH")
        print(f"  Expected: {repr(test_sentence)}")
        print(f"  Got:      {repr(decoded)}")

    return sp


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece BPE tokenizer on a text file."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Vocabulary size (default: {DEFAULT_VOCAB_SIZE})"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"Input text file to train on (default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default=DEFAULT_MODEL_PREFIX,
        help=f"Output model name prefix (default: {DEFAULT_MODEL_PREFIX})"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download wiki.txt from ideami.com if not present"
    )
    args = parser.parse_args()

    # Optionally download the dataset
    if args.download:
        download_files_if_needed()

    # Check that the input file exists before training
    if not os.path.exists(args.input):
        print(f"ERROR: Input file '{args.input}' not found.")
        print("Run with --download to fetch wiki.txt, or provide your own text file with --input.")
        sys.exit(1)

    # Train the tokenizer
    train_tokenizer(
        input_file=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size
    )

    # Validate encode/decode works
    validate_tokenizer(model_prefix=args.model_prefix)

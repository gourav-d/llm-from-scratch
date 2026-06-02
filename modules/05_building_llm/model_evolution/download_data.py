# =============================================================================
# download_data.py
# =============================================================================
# PURPOSE: Download ~10MB of text data so we have something to train on.
#
# WHY DO WE NEED DATA?
#   Our model learns by reading millions of characters and figuring out
#   patterns — "after 'th', you often see 'e' or 'i'", etc.
#   More data = more patterns = better model.
#   10MB of text ≈ about 10 million characters. That's plenty for a
#   tiny model like ours.
#
# C# ANALOGY:
#   Think of this like a setup script that downloads a NuGet package's
#   sample data, or like a database seeder that fills your DB before
#   you run your app for the first time.
#
# STRATEGY:
#   1. Try HuggingFace "datasets" library first (clean Wikipedia text)
#   2. If that fails, fall back to free Project Gutenberg books
# =============================================================================

import os           # For creating directories and checking file existence
import sys          # For reading command-line arguments and writing to stderr

# --- Figure out where THIS script lives so paths always work ---
# os.path.abspath(__file__) = full path to this .py file
# os.path.dirname(...)      = the folder containing this .py file
# C# analogy: Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The folder where we'll save corpus.txt
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# The final output file all training scripts will read from
OUTPUT_FILE = os.path.join(DATA_DIR, "corpus.txt")

# How many lines to take from WikiText-103 (≈200k lines ≈ 10-15 MB)
MAX_LINES = 200_000


def download_via_huggingface():
    """
    Download WikiText-103 using the HuggingFace 'datasets' library.

    WHAT IS WIKITEXT-103?
      A dataset of clean, well-written Wikipedia articles.
      "103" refers to ~103 million tokens (words/subwords).
      It's one of the standard benchmarks for language model research.
      We only take the first 200k lines — about 10MB.

    WHY WIKIPEDIA TEXT?
      Wikipedia articles are written in clear, consistent English.
      Good sentence structure means our model learns real patterns,
      not random internet noise.

    RETURNS:
      True if successful, False if the library isn't installed.
    """
    try:
        # Try to import the datasets library (pip install datasets)
        # C# analogy: like a using statement — if the DLL isn't there, this throws
        from datasets import load_dataset

        print("  HuggingFace 'datasets' library found. Downloading WikiText-103...")
        print("  (This may take a minute on first run — it downloads ~500MB)")

        # load_dataset returns a DatasetDict — like a Dictionary<string, Dataset>
        # split="train" means we only want the training portion
        # The dataset has three splits: "train", "validation", "test"
        dataset = load_dataset(
            "wikitext",            # dataset name on HuggingFace Hub
            "wikitext-103-raw-v1", # the specific version (raw = not pre-tokenized)
            split="train",         # only the training split
            trust_remote_code=False
        )

        print(f"  Full dataset has {len(dataset):,} lines. Taking first {MAX_LINES:,}...")

        # dataset["text"] returns a list of strings — one per article/line
        # We take only the first MAX_LINES to keep file size manageable
        # C# analogy: dataset.Take(MAX_LINES).Select(row => row["text"])
        lines = dataset["text"][:MAX_LINES]

        # Join all lines with newline — creates one big string
        # C# analogy: String.Join("\n", lines)
        text = "\n".join(lines)

        return text

    except ImportError:
        # The datasets library is not installed
        print("  HuggingFace 'datasets' not installed. Falling back to Gutenberg...")
        return None

    except Exception as e:
        # Something else went wrong (network error, etc.)
        print(f"  HuggingFace download failed: {e}")
        print("  Falling back to Project Gutenberg...")
        return None


def download_via_gutenberg():
    """
    Download free books from Project Gutenberg as a fallback.

    WHAT IS PROJECT GUTENBERG?
      A free library of over 70,000 books whose copyright has expired.
      Includes classics like Shakespeare, Dickens, Tolstoy, etc.
      Perfect for training — high-quality, free, publicly available.
      Website: https://www.gutenberg.org

    WHY SHAKESPEARE + WAR AND PEACE?
      Shakespeare: Rich vocabulary, diverse sentence structures, ~5MB
      War and Peace: Very long novel, consistent prose style, ~3MB
      Together they give us ~8-10MB of clean English text.

    RETURNS:
      Combined text string from both books.
    """
    import urllib.request  # Built-in Python module for downloading files
                           # C# analogy: System.Net.Http.HttpClient

    # List of books to download
    # Each entry: (book name for display, URL)
    books = [
        (
            "Complete Works of Shakespeare",
            "https://www.gutenberg.org/files/100/100-0.txt"
        ),
        (
            "War and Peace (Tolstoy)",
            "https://www.gutenberg.org/files/2600/2600-0.txt"
        ),
    ]

    all_text = []  # We'll collect text from each book here
                   # C# analogy: var allText = new List<string>();

    for book_name, url in books:
        # Each iteration: book_name = "Complete Works...", url = "https://..."
        # C# analogy: foreach (var (bookName, url) in books)
        try:
            print(f"  Downloading: {book_name}")
            print(f"    URL: {url}")

            # urllib.request.urlopen opens a URL like a file stream
            # C# analogy: httpClient.GetStreamAsync(url)
            with urllib.request.urlopen(url, timeout=30) as response:
                # Read all bytes from the response
                # C# analogy: await response.Content.ReadAsByteArrayAsync()
                raw_bytes = response.read()

            # Decode bytes to string
            # The files use UTF-8 encoding; errors="replace" substitutes
            # unreadable characters with ? instead of crashing
            # C# analogy: Encoding.UTF8.GetString(bytes)
            text = raw_bytes.decode("utf-8", errors="replace")

            print(f"    Downloaded {len(text):,} characters")
            all_text.append(text)  # Add this book's text to our collection

        except Exception as e:
            print(f"  WARNING: Could not download {book_name}: {e}")
            # Continue to next book even if one fails
            continue

    if not all_text:
        # Both downloads failed — nothing to work with
        raise RuntimeError(
            "Could not download any training data.\n"
            "Please check your internet connection or install datasets:\n"
            "  pip install datasets"
        )

    # Combine all books with a separator between them
    # "\n\n" between books gives a clear break
    combined = "\n\n".join(all_text)
    return combined


def save_text(text):
    """
    Save the text to data/corpus.txt and print the file size.

    WHY SAVE TO A FILE?
      All training scripts read from corpus.txt.
      This way you only download once — like caching in C#.
    """
    # Create the data/ directory if it doesn't exist yet
    # exist_ok=True means "don't crash if it already exists"
    # C# analogy: Directory.CreateDirectory(dataDir)
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"\n  Saving to: {OUTPUT_FILE}")

    # Open file for writing in UTF-8 encoding
    # "w" = write mode (creates new file or overwrites existing)
    # encoding="utf-8" = standard text encoding
    # errors="replace" = replace any unencodable characters instead of crashing
    # C# analogy: File.WriteAllText(path, text, Encoding.UTF8)
    with open(OUTPUT_FILE, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)  # Write the entire text at once

    # Get file size in bytes, then convert to MB for display
    # os.path.getsize returns bytes — divide by 1_000_000 for MB
    file_size_bytes = os.path.getsize(OUTPUT_FILE)
    file_size_mb = file_size_bytes / 1_000_000  # Convert bytes → megabytes

    print(f"  File size : {file_size_mb:.1f} MB  ({file_size_bytes:,} bytes)")
    print(f"  Characters: {len(text):,}")
    print(f"  Lines     : {text.count(chr(10)):,}")


def main():
    """
    Main function — orchestrates the download process.
    Called when you run: python download_data.py
    """
    print("=" * 60)
    print("  MODEL EVOLUTION LAB — Data Downloader")
    print("=" * 60)
    print()

    # Check if corpus.txt already exists — no need to re-download
    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / 1_000_000
        print(f"  corpus.txt already exists ({size_mb:.1f} MB).")
        print("  Delete it and re-run this script to re-download.")
        print()
        return  # Early exit — nothing to do

    print("  Attempting download...")
    print()

    # --- Strategy 1: HuggingFace datasets ---
    text = download_via_huggingface()

    # --- Strategy 2: Project Gutenberg fallback ---
    if text is None:
        # HuggingFace didn't work, try Gutenberg
        text = download_via_gutenberg()

    # --- Save to disk ---
    save_text(text)

    print()
    print("  Done! corpus.txt is ready.")
    print()
    print("  Next step: python step_00_bigram.py")
    print("=" * 60)


# =============================================================================
# if __name__ == "__main__": PATTERN EXPLAINED
# =============================================================================
# In Python, every .py file is a "module". When you import a module, Python
# runs all the top-level code in it.
#
# The variable __name__ is:
#   - "__main__"  when you run this file directly:  python download_data.py
#   - "download_data" when someone imports it:       import download_data
#
# This pattern means: "only run main() if THIS file was executed directly,
# not if it was imported by another file."
#
# C# analogy: like a static void Main(string[] args) — the entry point.
# =============================================================================
if __name__ == "__main__":
    main()

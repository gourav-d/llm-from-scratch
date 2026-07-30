"""
chunker.py -- Split source files into overlapping text chunks.

WHY DO WE CHUNK?
----------------
LLMs can only read a limited amount of text at once (their "context window").
A codebase might have millions of characters -- way too much to send all at once.
So we split it into small pieces (chunks) and only send the RELEVANT pieces.

WHY OVERLAP?
------------
Imagine cutting a document every 1500 characters exactly.
You might cut a function definition in half:

  Chunk 1: "def calculate_tax(amount, rate):\n    # apply discount\n    discou"
  Chunk 2: "nted = amount * (1 - discount)\n    return discounted * rate"

The second chunk loses context about what the function does.
Overlap fixes this: each chunk shares 200 chars with the previous one,
so function names, class names, etc. appear in both adjacent chunks.

C#/.NET ANALOGY:
----------------
This is like a sliding window algorithm over a string:

  // C# equivalent concept
  for (int i = 0; i < text.Length; i += (chunkSize - overlap)) {
      string chunk = text.Substring(i, Math.Min(chunkSize, text.Length - i));
      chunks.Add(chunk);
  }
"""

import os                   # for file path operations
from typing import List     # for type hints (like List<T> in C#)


def read_file_safe(file_path: str) -> str:
    """
    Read a file and return its text content.
    Tries UTF-8 first, then falls back to latin-1.

    Why two encodings?
    - UTF-8: modern standard, handles all languages
    - latin-1: handles old files, Windows files, never fails (every byte is valid)

    C# equivalent: File.ReadAllText(path, Encoding.UTF8) with a catch block
    """
    # Try reading as UTF-8 first (most source files are UTF-8)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, fall back to latin-1 (never fails, handles any byte)
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            # If even latin-1 fails (e.g. binary file), return empty string
            return ""


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split a long string into overlapping chunks.

    Parameters:
        text       : the full file content as a string
        chunk_size : max characters per chunk (e.g. 1500)
        overlap    : characters shared between consecutive chunks (e.g. 200)

    Returns:
        List of string chunks

    Example with chunk_size=10, overlap=3:
        text = "ABCDEFGHIJKLMNOP"
        chunks = ["ABCDEFGHIJ", "HIJKLMNOPQ", "NOPQ"]
                              ^^^              ^^^
                              3-char overlap   3-char overlap
    """
    chunks = []     # will hold our result chunks (like List<string> in C#)
    start = 0       # index of where current chunk begins

    # Keep creating chunks until we've covered the whole text
    while start < len(text):
        # End of this chunk: either chunk_size chars ahead, or end of text
        end = min(start + chunk_size, len(text))

        # Extract this chunk from the text
        chunk = text[start:end]

        # Only add non-empty chunks (skip blank files or whitespace-only chunks)
        if chunk.strip():
            chunks.append(chunk)

        # Move start forward by (chunk_size - overlap)
        # This is what creates the sliding window effect
        start += chunk_size - overlap

        # Safety check: if overlap >= chunk_size, we'd loop forever
        # (start would never advance). This breaks the infinite loop.
        if chunk_size <= overlap:
            break

    return chunks


def chunk_file(file_path: str, chunk_size: int, overlap: int) -> List[dict]:
    """
    Read a file and return a list of chunk dictionaries.

    Each chunk dict contains:
        - "text"      : the actual chunk content
        - "source"    : the file path this chunk came from
        - "chunk_idx" : which chunk number this is (0, 1, 2, ...)

    This metadata is stored alongside the vector in ChromaDB
    so we can show the user WHERE the answer came from.

    C# analogy: this is like a DTO (Data Transfer Object) --
    a small class that bundles the data we want to store.

    Example output:
    [
        {"text": "def foo():\n    pass", "source": "src/utils.py", "chunk_idx": 0},
        {"text": "    pass\n\ndef bar():", "source": "src/utils.py", "chunk_idx": 1},
    ]
    """
    # Read the raw text from disk
    text = read_file_safe(file_path)

    # If file is empty or unreadable, return nothing
    if not text.strip():
        return []

    # Split into overlapping chunks
    raw_chunks = chunk_text(text, chunk_size, overlap)

    # Wrap each chunk in a dict with source metadata
    result = []
    for idx, chunk in enumerate(raw_chunks):
        result.append({
            "text": chunk,              # the text content
            "source": file_path,        # WHERE this came from
            "chunk_idx": idx,           # which chunk within the file
        })

    return result


def walk_repo(repo_path: str, allowed_extensions: set, skip_dirs: set) -> List[str]:
    """
    Walk a directory tree and return all indexable file paths.

    Parameters:
        repo_path          : root folder to start from
        allowed_extensions : set of file extensions to include (e.g. {".py", ".md"})
        skip_dirs          : set of directory names to skip (e.g. {"venv", ".git"})

    Returns:
        List of absolute file paths

    C# analogy:
        Directory.GetFiles(repoPath, "*.*", SearchOption.AllDirectories)
        .Where(f => allowedExtensions.Contains(Path.GetExtension(f)))
        .Where(f => !f.Split(Path.DirectorySeparatorChar).Any(d => skipDirs.Contains(d)))
    """
    file_paths = []     # list to collect matching file paths

    # os.walk() is like Directory.EnumerateFiles with recursion in C#
    # It yields (current_dir, list_of_subdirs, list_of_files) for each folder
    for root, dirs, files in os.walk(repo_path):

        # IMPORTANT: modify dirs IN-PLACE to skip unwanted subdirectories.
        # os.walk() checks this list before descending, so removing a name
        # here prevents walking that entire subtree.
        # In C#: you'd use a custom recursive method that skips folders.
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for filename in files:
            # Get the file extension: "utils.py" -> ".py"
            ext = os.path.splitext(filename)[1].lower()

            # Only include files with allowed extensions
            if ext in allowed_extensions:
                # Build the full absolute path
                full_path = os.path.join(root, filename)
                file_paths.append(full_path)

    return file_paths

"""
reindex.py -- Phase 5: Smart diff-based re-indexer.

WHAT THIS FILE DOES:
---------------------
Instead of re-indexing the ENTIRE codebase every time you change a few files,
this script detects which files have changed and only re-embeds those.

HOW IT DETECTS CHANGES:
------------------------
We store a "file manifest" -- a JSON file that records each indexed file's
path and its MD5 hash (a fingerprint of its content).

When you run reindex.py:
1. Walk the repo to get current files + compute their MD5 hashes
2. Compare against the saved manifest
3. Find: NEW files (not in manifest), CHANGED files (hash differs), DELETED files
4. Re-index only new + changed files
5. Remove deleted files from ChromaDB
6. Update the manifest

WHY MD5 FOR FILE COMPARISON?
------------------------------
MD5 is fast and produces a fixed-length "fingerprint" of a file.
If even one character changes, the MD5 changes completely.
We only need fingerprinting, not cryptographic security, so MD5 is fine.

C# analogy:
  // Get file fingerprint
  using var md5 = MD5.Create();
  using var stream = File.OpenRead(path);
  byte[] hash = md5.ComputeHash(stream);
  string fingerprint = Convert.ToHexString(hash);

  // Compare against saved manifest
  if (!manifest.ContainsKey(path) || manifest[path] != fingerprint) {
      filesToReindex.Add(path);
  }

Run with:
    python reindex.py
"""

import os
import sys
import json
import hashlib
from tqdm import tqdm
import chromadb

import config
from utils.chunker import walk_repo, chunk_file
from utils.embedder import get_embedding, check_ollama_running, check_model_available
from indexer import make_chunk_id, get_or_create_collection


# Path to the manifest file -- stored next to the ChromaDB folder
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "file_manifest.json")


def compute_file_hash(file_path: str) -> str:
    """
    Compute the MD5 hash of a file's contents.

    Returns a 32-character hex string like "d41d8cd98f00b204e9800998ecf8427e".
    Returns empty string if the file cannot be read.

    C# analogy:
        using var md5 = MD5.Create();
        using var stream = File.OpenRead(filePath);
        return Convert.ToHexString(md5.ComputeHash(stream));
    """
    hash_obj = hashlib.md5()

    try:
        # Read file in binary mode -- works for any file type
        # "rb" = read binary (in C#: FileMode.Open with BinaryReader)
        with open(file_path, "rb") as f:
            # Read in 65536-byte chunks to handle large files without loading all into memory
            # In C#: ReadAsync with a buffer
            for chunk in iter(lambda: f.read(65536), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception:
        return ""


def load_manifest() -> dict:
    """
    Load the file manifest from disk.

    The manifest is a JSON file: { "file_path": "md5_hash", ... }

    Returns empty dict if the manifest doesn't exist yet.

    C# analogy:
        return File.Exists(manifestPath)
            ? JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(manifestPath))
            : new Dictionary<string, string>();
    """
    if not os.path.exists(MANIFEST_PATH):
        return {}

    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(manifest: dict):
    """
    Save the file manifest to disk.

    C# analogy:
        File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest));
    """
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def compute_diff(current_files: list, manifest: dict) -> dict:
    """
    Compare current repo files against the manifest to find changes.

    Parameters:
        current_files : list of file paths currently in the repo
        manifest      : dict of {file_path: md5_hash} from last index

    Returns:
        Dict with:
            - "new"     : files not in manifest (need indexing)
            - "changed" : files whose MD5 changed (need re-indexing)
            - "deleted" : files in manifest but no longer in repo (need removal)
            - "unchanged": files with same MD5 (skip)

    C# analogy:
        var currentSet = new HashSet<string>(currentFiles);
        var manifestKeys = new HashSet<string>(manifest.Keys);
        var deleted = manifestKeys.Except(currentSet).ToList();
        var newFiles = currentSet.Except(manifestKeys).ToList();
        var changed = currentSet.Intersect(manifestKeys)
            .Where(f => ComputeHash(f) != manifest[f])
            .ToList();
    """
    current_set = set(current_files)        # for O(1) lookup
    manifest_set = set(manifest.keys())     # files that were indexed before

    diff = {
        "new": [],
        "changed": [],
        "deleted": [],
        "unchanged": [],
    }

    # Deleted files: in manifest but no longer in repo
    for path in manifest_set - current_set:
        diff["deleted"].append(path)

    # New and changed files
    for path in current_files:
        current_hash = compute_file_hash(path)

        if path not in manifest:
            # New file -- not indexed before
            diff["new"].append(path)
        elif manifest[path] != current_hash:
            # Changed file -- content different from last index
            diff["changed"].append(path)
        else:
            # Unchanged -- skip
            diff["unchanged"].append(path)

    return diff


def remove_file_from_index(collection, file_path: str):
    """
    Remove all chunks belonging to a file from ChromaDB.

    When a file is deleted or significantly changed, we remove its old chunks
    from the database so stale data doesn't appear in search results.

    ChromaDB lets us query by metadata, then delete by IDs.

    C# analogy:
        var idsToDelete = db.Chunks.Where(c => c.Source == filePath).Select(c => c.Id).ToList();
        db.Chunks.RemoveRange(idsToDelete);
        db.SaveChanges();
    """
    try:
        # Find all chunks from this file using a metadata filter
        results = collection.get(
            where={"source": file_path},     # filter by source metadata
            include=["metadatas"],
        )

        # Get the IDs of matching chunks
        ids_to_delete = results["ids"]

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)

    except Exception as e:
        print(f"  WARN: Could not remove chunks for {file_path}: {e}")


def index_file(collection, file_path: str) -> int:
    """
    Index a single file: chunk, embed, upsert into ChromaDB.

    Returns the number of chunks stored.
    """
    # Chunk the file
    chunks = chunk_file(file_path, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        return 0

    ids = []
    texts = []
    embeddings = []
    metadatas = []

    for chunk in chunks:
        chunk_id = make_chunk_id(chunk["source"], chunk["chunk_idx"], chunk["text"])
        embedding = get_embedding(chunk["text"])

        ids.append(chunk_id)
        texts.append(chunk["text"])
        embeddings.append(embedding)
        metadatas.append({
            "source": chunk["source"],
            "chunk_idx": chunk["chunk_idx"],
        })

    # upsert = insert or update (handles both new and changed files)
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def run_reindex():
    """
    Main re-index function: compute diff, process only changed files.
    """
    print("=" * 60)
    print("  Chat with Codebase -- Smart Re-indexer")
    print("=" * 60)

    # Pre-flight checks
    print("\nChecking setup...")
    if not check_ollama_running():
        print("ERROR: Ollama is not running.")
        sys.exit(1)

    if not check_model_available(config.EMBEDDING_MODEL):
        print(f"ERROR: Model '{config.EMBEDDING_MODEL}' not found. Run: ollama pull {config.EMBEDDING_MODEL}")
        sys.exit(1)

    print("  [OK] Ollama ready")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = get_or_create_collection(client)
    print(f"  [OK] ChromaDB: {collection.count()} chunks currently indexed")

    # Load existing manifest
    manifest = load_manifest()
    print(f"  [OK] Manifest: {len(manifest)} files tracked")

    # Walk current repo
    print(f"\nScanning repo: {config.REPO_PATH}")
    current_files = walk_repo(config.REPO_PATH, config.ALLOWED_EXTENSIONS, config.SKIP_DIRS)
    print(f"Found {len(current_files)} files in repo")

    # Compute diff
    print("\nComputing diff...")
    diff = compute_diff(current_files, manifest)

    print(f"  New files:       {len(diff['new'])}")
    print(f"  Changed files:   {len(diff['changed'])}")
    print(f"  Deleted files:   {len(diff['deleted'])}")
    print(f"  Unchanged files: {len(diff['unchanged'])} (skipping)")

    # If nothing changed, we're done
    if not diff["new"] and not diff["changed"] and not diff["deleted"]:
        print("\nNo changes detected. Index is up to date.")
        return

    total_chunks = 0

    # Process DELETED files: remove from ChromaDB + manifest
    if diff["deleted"]:
        print(f"\nRemoving {len(diff['deleted'])} deleted files...")
        for file_path in tqdm(diff["deleted"], desc="Removing", unit="file"):
            remove_file_from_index(collection, file_path)
            manifest.pop(file_path, None)   # remove from manifest

    # Process CHANGED files: remove old chunks, re-index
    if diff["changed"]:
        print(f"\nRe-indexing {len(diff['changed'])} changed files...")
        for file_path in tqdm(diff["changed"], desc="Re-indexing changed", unit="file"):
            # Remove stale chunks first
            remove_file_from_index(collection, file_path)
            # Re-embed and store
            count = index_file(collection, file_path)
            total_chunks += count
            # Update manifest with new hash
            manifest[file_path] = compute_file_hash(file_path)

    # Process NEW files: index and add to manifest
    if diff["new"]:
        print(f"\nIndexing {len(diff['new'])} new files...")
        for file_path in tqdm(diff["new"], desc="Indexing new", unit="file"):
            count = index_file(collection, file_path)
            total_chunks += count
            manifest[file_path] = compute_file_hash(file_path)

    # Save the updated manifest
    save_manifest(manifest)

    # Summary
    print("\n" + "=" * 60)
    print("  Re-index Complete!")
    print("=" * 60)
    print(f"  Files removed:   {len(diff['deleted'])}")
    print(f"  Files updated:   {len(diff['changed'])}")
    print(f"  Files added:     {len(diff['new'])}")
    print(f"  New chunks:      {total_chunks}")
    print(f"  Total in DB:     {collection.count()}")
    print(f"  Manifest saved:  {MANIFEST_PATH}")


if __name__ == "__main__":
    run_reindex()

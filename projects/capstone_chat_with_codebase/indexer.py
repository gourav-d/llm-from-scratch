"""
indexer.py -- Phase 1: Index your codebase into ChromaDB.

WHAT THIS FILE DOES:
--------------------
1. Walk every source file in your repo
2. Split each file into overlapping text chunks
3. Convert each chunk into an embedding vector (via Ollama)
4. Store chunk text + vector + metadata in ChromaDB

Run this once before you can start chatting:
    python indexer.py

HOW ChromaDB WORKS:
-------------------
ChromaDB is a local vector database -- like SQL Server, but instead of
storing rows and querying by WHERE clause, it stores vectors and queries
by "which vectors are closest to this query vector?"

Conceptual mapping to SQL:
  CREATE TABLE codebase (id TEXT, text TEXT, source TEXT, vector FLOAT[768])
  INSERT INTO codebase VALUES (id, text, source, embedding(text))

  -- Then at query time:
  SELECT text, source FROM codebase
  ORDER BY cosine_distance(vector, embedding(question))
  LIMIT 5

ChromaDB does all of this automatically -- we just call collection.add() and collection.query().

WHAT IS cosine DISTANCE?
-------------------------
It measures the angle between two vectors.
0.0 = identical direction = very similar meaning
1.0 = opposite direction = very different meaning

Think of it like: if two arrows point in the same direction, the texts mean the same thing.
"""

import os
import sys
import json
import hashlib          # for generating unique IDs from content
from tqdm import tqdm   # progress bar (like a for loop with a visual counter)
import chromadb         # vector database

import config
from utils.chunker import walk_repo, chunk_file
from utils.embedder import get_embedding, check_ollama_running, check_model_available


def make_chunk_id(file_path: str, chunk_idx: int, text: str) -> str:
    """
    Generate a unique, stable ID for a chunk.

    WHY?: ChromaDB requires a unique string ID for every stored item.
    If we re-index, we want the same chunk to get the same ID so ChromaDB
    can UPDATE it rather than create a duplicate.

    We hash (file_path + chunk_idx + first 100 chars of text) to get a stable ID.
    MD5 is fine here -- we just need uniqueness, not security.

    C# analogy:
        string id = Convert.ToHexString(MD5.HashData(
            Encoding.UTF8.GetBytes(filePath + chunkIdx + text[:100])
        ));
    """
    # Combine file path + chunk index + text preview into one string
    raw = f"{file_path}::{chunk_idx}::{text[:100]}"

    # MD5 hash gives us a fixed-length hex string (32 chars)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def preflight_checks() -> bool:
    """
    Check that Ollama is running and required models are available.
    Returns True if all checks pass, False otherwise.

    C# analogy: a startup health check method.
    """
    print("Running pre-flight checks...")

    # Check 1: Is Ollama running?
    if not check_ollama_running():
        print("ERROR: Ollama is not running.")
        print("  Start it: open the Ollama app, or run 'ollama serve' in a terminal.")
        return False
    print("  [OK] Ollama is running")

    # Check 2: Is the embedding model available?
    if not check_model_available(config.EMBEDDING_MODEL):
        print(f"ERROR: Embedding model '{config.EMBEDDING_MODEL}' not found.")
        print(f"  Pull it: ollama pull {config.EMBEDDING_MODEL}")
        return False
    print(f"  [OK] Embedding model '{config.EMBEDDING_MODEL}' is available")

    # Check 3: Does the repo path exist?
    if not os.path.isdir(config.REPO_PATH):
        print(f"ERROR: REPO_PATH does not exist: {config.REPO_PATH}")
        print("  Edit config.py and set REPO_PATH to your codebase folder.")
        return False
    print(f"  [OK] Repo path exists: {config.REPO_PATH}")

    return True


def get_or_create_collection(client: chromadb.Client):
    """
    Get or create the ChromaDB collection.

    A ChromaDB "collection" is like a table in a relational database.
    We store all our chunks in one collection called COLLECTION_NAME.

    Parameters:
        client : the ChromaDB client (database connection)

    Returns:
        The ChromaDB collection object (our "table")
    """
    # get_or_create_collection: creates it if it doesn't exist, opens it if it does
    # cosine distance is best for embeddings -- compares meaning, not magnitude
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # use cosine distance for similarity
    )
    return collection


def index_repo(force_reindex: bool = False) -> dict:
    """
    Main indexing function -- walk the repo, chunk files, embed, store.

    Parameters:
        force_reindex : if True, delete existing collection and start fresh.
                        if False, skip files that are already indexed.

    Returns:
        A summary dict with counts of files processed, chunks stored, etc.

    Steps:
        1. Connect to ChromaDB
        2. Walk the repo to find all source files
        3. For each file: chunk it, embed each chunk, store in ChromaDB
        4. Return a summary

    C# analogy: a repository seeding method with upsert semantics.
    """
    # -------------------------------------------------------------------------
    # STEP 1: Connect to ChromaDB
    # -------------------------------------------------------------------------
    print(f"\nConnecting to ChromaDB at: {config.CHROMA_DB_PATH}")

    # PersistentClient saves data to disk at the given path.
    # Without this, data would be lost when the program exits.
    # C# analogy: new SqlConnection(connectionString)
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    # If force_reindex, delete the collection and start fresh
    if force_reindex:
        print("Force re-index: deleting existing collection...")
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass    # collection might not exist yet, that's fine

    # Get or create our collection
    collection = get_or_create_collection(client)
    print(f"Collection '{config.COLLECTION_NAME}' ready. "
          f"Current count: {collection.count()} chunks")

    # -------------------------------------------------------------------------
    # STEP 2: Walk the repo to find all source files
    # -------------------------------------------------------------------------
    print(f"\nScanning repo: {config.REPO_PATH}")
    file_paths = walk_repo(
        config.REPO_PATH,
        config.ALLOWED_EXTENSIONS,
        config.SKIP_DIRS,
    )
    print(f"Found {len(file_paths)} files to index.")

    # -------------------------------------------------------------------------
    # STEP 3: For each file, chunk + embed + store
    # -------------------------------------------------------------------------
    stats = {
        "files_found": len(file_paths),
        "files_processed": 0,
        "files_skipped": 0,
        "chunks_added": 0,
        "chunks_updated": 0,
        "errors": 0,
    }

    # tqdm wraps the file list and shows a progress bar in the terminal
    # C# analogy: like a for loop with Console.Write($"\r{i}/{total}")
    for file_path in tqdm(file_paths, desc="Indexing files", unit="file"):
        try:
            # Split this file into chunks
            chunks = chunk_file(
                file_path,
                config.CHUNK_SIZE,
                config.CHUNK_OVERLAP,
            )

            # Skip empty files
            if not chunks:
                stats["files_skipped"] += 1
                continue

            # Prepare batch data for ChromaDB
            # ChromaDB add() accepts lists -- we send all chunks from one file at once
            ids = []            # unique ID for each chunk
            texts = []          # the raw text of each chunk
            embeddings = []     # the vector for each chunk
            metadatas = []      # extra info: source file path, chunk index

            for chunk in chunks:
                # Generate a unique, stable ID for this chunk
                chunk_id = make_chunk_id(
                    chunk["source"],
                    chunk["chunk_idx"],
                    chunk["text"],
                )

                # Convert chunk text to embedding vector
                embedding = get_embedding(chunk["text"])

                ids.append(chunk_id)
                texts.append(chunk["text"])
                embeddings.append(embedding)

                # Metadata stored alongside the vector.
                # ChromaDB metadata values must be str, int, float, or bool.
                metadatas.append({
                    "source": chunk["source"],       # file path
                    "chunk_idx": chunk["chunk_idx"], # position in file
                })

            # Store (or update) all chunks from this file in ChromaDB.
            # upsert = insert if new, update if ID already exists.
            # C# analogy: context.AddOrUpdate(entities)
            collection.upsert(
                ids=ids,
                documents=texts,        # "documents" = the text ChromaDB stores
                embeddings=embeddings,  # the vectors
                metadatas=metadatas,    # the extra metadata
            )

            stats["files_processed"] += 1
            stats["chunks_added"] += len(chunks)

        except Exception as e:
            # Don't let one bad file stop the whole indexing
            print(f"\nWARN: Could not index {file_path}: {e}")
            stats["errors"] += 1

    return stats


def main():
    """
    Entry point when running: python indexer.py
    """
    print("=" * 60)
    print("  Chat with Codebase -- Indexer")
    print("=" * 60)

    # Run pre-flight checks before doing any work
    if not preflight_checks():
        sys.exit(1)     # exit with error code 1 (like Environment.Exit(1) in C#)

    # Ask user if they want a full re-index or an update
    print("\nIndex options:")
    print("  [1] Update -- add/update changed files (default)")
    print("  [2] Full re-index -- delete everything and start fresh")
    choice = input("Choose [1/2]: ").strip()

    force_reindex = (choice == "2")

    # Run the indexing
    print()
    stats = index_repo(force_reindex=force_reindex)

    # Print summary
    print("\n" + "=" * 60)
    print("  Indexing Complete!")
    print("=" * 60)
    print(f"  Files found:     {stats['files_found']}")
    print(f"  Files indexed:   {stats['files_processed']}")
    print(f"  Files skipped:   {stats['files_skipped']}")
    print(f"  Chunks stored:   {stats['chunks_added']}")
    print(f"  Errors:          {stats['errors']}")
    print()
    print("Next step: run 'python chat.py' to start chatting!")


# Standard Python entry point guard.
# In C#: this is like the static void Main() method.
# __name__ == "__main__" is True only when THIS file is run directly.
# If another file imports indexer.py, this block is skipped.
if __name__ == "__main__":
    main()

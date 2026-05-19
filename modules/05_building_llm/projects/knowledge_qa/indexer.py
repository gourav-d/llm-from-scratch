"""
=============================================================================
PROJECT: Personal Knowledge Base Q&A  --  Phase 2 of 5
FILE   : indexer.py
=============================================================================

WHAT THIS FILE DOES
--------------------
Reads all .md notes, splits them into small chunks (paragraphs),
embeds each chunk using the model trained in Phase 1,
and saves everything to index.json.

index.json is the "database" the retriever searches.
Run embedder.py BEFORE this file.
Run this file ONCE (or re-run whenever notes change).

SKILLS REUSED FROM MODULE 05
------------------------------
  example_01_tokenization_pytorch.py  -->  text -> token IDs concept
  example_02_embeddings_pytorch.py    -->  encode_chunk (mean pooling)

HOW TO RUN
-----------
  python indexer.py

OUTPUT
-------
  saved/index.json   -- list of {text, source_file, vector} for every chunk

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

NOTES_DIR  = Path(__file__).parent.parent.parent   # = modules/05_building_llm/
SAVED_DIR  = Path(__file__).parent / "saved"
CHUNK_SIZE = 300    # characters per chunk (bigger = more context per chunk)
MIN_CHUNK  = 50     # skip chunks shorter than this (too small to be useful)

# =============================================================================
# STEP 1: Load the trained embedding model from Phase 1
# =============================================================================

print("=" * 60)
print("PHASE 2: Indexing Your Notes")
print("=" * 60)

# Load vocabulary (saved by embedder.py)
vocab_path = SAVED_DIR / "vocab.json"
if not vocab_path.exists():
    print("ERROR: saved/vocab.json not found.")
    print("Run embedder.py first!")
    exit(1)

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

char_to_idx = vocab_data["char_to_idx"]
vocab_size  = vocab_data["vocab_size"]
embed_dim   = vocab_data["embed_dim"]

print(f"Vocabulary loaded: {vocab_size} chars, embed_dim={embed_dim}")


# Rebuild the EmbeddingModel class (must match embedder.py exactly)
# In a real project this would be in a shared models.py file.
# Here we repeat it so each file is self-contained and easy to read.
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x      = self.emb(idx)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def encode_chunk(self, chunk_text, char_to_idx):
        ids = torch.tensor(
            [char_to_idx.get(ch, 0) for ch in chunk_text],
            dtype=torch.long
        )
        if len(ids) == 0:
            return torch.zeros(self.emb.embedding_dim)
        vecs = self.emb(ids)
        return vecs.mean(dim=0)


# Load trained weights into the model
# model.load_state_dict() = load saved weights from file
# C# analogy: deserialize weights from a binary file
model = EmbeddingModel(vocab_size, embed_dim)
model_path = SAVED_DIR / "embedding_model.pt"

if not model_path.exists():
    print("ERROR: saved/embedding_model.pt not found.")
    print("Run embedder.py first!")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()   # set to eval mode: disables dropout, etc.
               # C# analogy: set model to read-only / inference mode

print(f"Embedding model loaded from: {model_path}")

# =============================================================================
# STEP 2: Read all .md files
# =============================================================================

def load_notes(notes_dir):
    """Return list of (filename, full_text) tuples."""
    notes = []
    for md_file in sorted(Path(notes_dir).rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            notes.append((md_file.name, content))
        except Exception as e:
            print(f"  Warning: {md_file.name}: {e}")
    return notes

print(f"\nReading notes from: {NOTES_DIR}")
notes = load_notes(NOTES_DIR)
print(f"Files found: {len(notes)}")
for name, content in notes:
    print(f"  - {name}  ({len(content):,} chars)")

# =============================================================================
# STEP 3: Split each file into chunks
# =============================================================================
# A "chunk" is a paragraph or section of text.
# Smaller chunks = more precise retrieval but less context.
# Larger chunks = more context but less precise.
# 300 characters is a good starting point.

def split_into_chunks(text, chunk_size, min_chunk):
    """
    Split text into overlapping chunks of ~chunk_size characters.

    Strategy: split on paragraph boundaries (double newline).
    If a paragraph is too long, split it into smaller pieces.

    Returns: list of strings
    """
    chunks = []

    # Split on blank lines (paragraph breaks)
    # '\\n\\n' = double newline = paragraph separator
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para = para.strip()
        if len(para) < min_chunk:
            continue    # skip very short paragraphs

        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # Paragraph too long: split into pieces with 50-char overlap
            # Overlap = end of chunk A appears at start of chunk B
            # Why overlap? So retrieval does not miss context at boundaries.
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                piece = para[start:end]
                if len(piece) >= min_chunk:
                    chunks.append(piece)
                start += chunk_size - 50    # 50-char overlap

    return chunks


print(f"\nChunking notes (target chunk size: {CHUNK_SIZE} chars)...")
all_chunks = []   # list of {"text": "...", "source": "filename.md"}

for filename, content in notes:
    chunks = split_into_chunks(content, CHUNK_SIZE, MIN_CHUNK)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "source": filename})

print(f"Total chunks created: {len(all_chunks)}")

# =============================================================================
# STEP 4: Embed every chunk
# =============================================================================
# For each chunk: run encode_chunk() -> get a vector -> store it.
# This is the most time-consuming step. On a CPU with 3000 steps it takes ~1 min.

print(f"\nEmbedding {len(all_chunks)} chunks...")
print("(This may take a minute...)")

index = []   # final list of {text, source, vector}

for i, chunk_dict in enumerate(all_chunks):
    # encode_chunk: char sequence -> mean-pooled embedding vector
    vec = model.encode_chunk(chunk_dict["text"], char_to_idx)

    # .tolist() converts torch tensor to plain Python list (JSON-serializable)
    # C# analogy: .ToArray() or .ToList()
    index.append({
        "text":   chunk_dict["text"],
        "source": chunk_dict["source"],
        "vector": vec.tolist(),
    })

    if (i + 1) % 100 == 0 or (i + 1) == len(all_chunks):
        print(f"  Embedded {i + 1:4d} / {len(all_chunks)}")

# =============================================================================
# STEP 5: Save index to disk
# =============================================================================

index_path = SAVED_DIR / "index.json"
with open(index_path, "w", encoding="utf-8") as f:
    json.dump(index, f, ensure_ascii=False, indent=2)

print(f"\nSaved index to: {index_path}")
print(f"Index contains: {len(index)} chunks")
print(f"Each entry has: text, source filename, {embed_dim}-dimensional vector")

# Quick sanity check: show first 3 entries
print("\nSample index entries:")
for entry in index[:3]:
    preview = entry["text"][:80].replace("\n", " ")
    print(f"  [{entry['source']}] {preview}...")

print("\n" + "=" * 60)
print("Phase 2 DONE.")
print("Next step: run  python retriever.py  to test search")
print("           run  python app.py        to use the full app")
print("=" * 60)

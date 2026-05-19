"""
=============================================================================
PROJECT: Personal Knowledge Base Q&A  --  Phase 1 of 5
FILE   : embedder.py
=============================================================================

WHAT THIS FILE DOES
--------------------
Reads all your .md notes, trains a character-level embedding model on them,
and saves the trained model + vocabulary to disk.

Every other file (indexer, retriever, generator, app) depends on this.
Run this FIRST, run it ONCE.

SKILLS REUSED FROM MODULE 05
------------------------------
  example_02_embeddings_pytorch.py  -->  nn.Embedding, F.cosine_similarity
  example_03_bigram_pytorch.py      -->  training loop: zero_grad/backward/step

HOW TO RUN
-----------
  python embedder.py

OUTPUT
-------
  saved/embedding_model.pt   -- trained model weights
  saved/vocab.json           -- character vocabulary + settings

=============================================================================
"""

import torch                        # core PyTorch
import torch.nn as nn               # neural network layers
import torch.nn.functional as F     # standalone functions
import json                         # for saving vocabulary to disk
from pathlib import Path            # cross-platform file paths (better than strings)

# =============================================================================
# CONFIGURATION  --  change these if needed
# =============================================================================

# Where your .md notes live (relative to this file's location)
# Path(__file__).parent = the folder containing this script
# .parent.parent        = go up two levels to project root
NOTES_DIR   = Path(__file__).parent.parent.parent   # = modules/05_building_llm/

# Where to save the trained model + vocab
SAVED_DIR   = Path(__file__).parent / "saved"

EMBED_DIM   = 32     # size of each character vector (32 numbers per char)
TRAIN_STEPS = 3000   # how many training steps (more = better embeddings)
BATCH_SIZE  = 32     # how many training examples per step
CONTEXT_LEN = 16     # how many characters to look at at once
LEARNING_RATE = 1e-3

# =============================================================================
# STEP 1: Load all .md notes into one big string
# =============================================================================

def load_notes(notes_dir):
    """
    Read every .md file in notes_dir (and subfolders).
    Combine all text into one big string.

    Returns: (combined_text, list_of_file_paths_found)
    """
    all_text   = ""
    file_paths = []

    # Path.rglob("*.md") = find all .md files recursively
    # C# analogy: Directory.GetFiles(path, "*.md", SearchOption.AllDirectories)
    for md_file in sorted(Path(notes_dir).rglob("*.md")):
        try:
            # encoding='utf-8', errors='ignore' = skip characters we can't read
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            all_text += content + "\n"
            file_paths.append(str(md_file.name))
        except Exception as e:
            print(f"  Warning: could not read {md_file.name}: {e}")

    return all_text, file_paths


print("=" * 60)
print("PHASE 1: Training the Embedding Model")
print("=" * 60)
print(f"\nLooking for notes in: {NOTES_DIR}")

text, files_found = load_notes(NOTES_DIR)

if not text:
    print("ERROR: No .md files found. Check NOTES_DIR path above.")
    exit(1)

print(f"Files found : {len(files_found)}")
for f in files_found:
    print(f"  - {f}")
print(f"Total chars : {len(text):,}")

# =============================================================================
# STEP 2: Build character vocabulary
# =============================================================================
# Same as every other module 05 example: sorted unique chars -> int IDs

chars       = sorted(set(text))           # all unique characters in our notes
vocab_size  = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}    # char -> int
idx_to_char = {i: ch for i, ch in enumerate(chars)}    # int -> char

print(f"\nVocabulary  : {vocab_size} unique characters")
print(f"Sample chars: {chars[:20]}...")

# Encode the entire text as a tensor of integer token IDs
data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
print(f"Encoded data: {len(data):,} token IDs")

# =============================================================================
# STEP 3: Define the embedding model
# =============================================================================

class EmbeddingModel(nn.Module):
    """
    Character-level embedding model.

    Architecture:
      character IDs
           |
           v
      nn.Embedding (vocab_size, embed_dim)   <-- the lookup table we care about
           |
           v
      nn.Linear (embed_dim, vocab_size)      <-- prediction head (discarded after training)
           |
           v
      logits: score for each possible next character

    Why train on next-char prediction?
      The embedding layer learns vectors that help predict what comes next.
      Characters appearing in similar contexts get similar vectors.
      This makes the embeddings useful for finding similar text chunks later.

      After training, we use ONLY the nn.Embedding part.
      The prediction head is just the training tool -- we throw it away.

    C# analogy:
      Like training a model then stripping off the output layer,
      keeping only the hidden representation layer for feature extraction.
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()                              # always call parent first

        # The lookup table: vocab_size rows, embed_dim columns
        # Each row = one character's learned vector
        # Same as example_02_embeddings_pytorch.py
        self.emb  = nn.Embedding(vocab_size, embed_dim)

        # Prediction head: maps embed_dim -> vocab_size scores
        # Used during training only
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """
        idx     : token IDs, shape (batch, seq_len)
        targets : next-char IDs, shape (batch, seq_len) -- optional
        returns : (logits, loss)
        """
        x      = self.emb(idx)          # (batch, seq_len, embed_dim)
        logits = self.head(x)           # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # F.cross_entropy expects (N, C) and (N,)
            # Flatten batch and seq dimensions together
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),    # (batch*seq, vocab_size)
                targets.view(-1)                # (batch*seq,)
            )

        return logits, loss

    @torch.no_grad()
    def encode_chunk(self, chunk_text, char_to_idx):
        """
        Convert a text chunk to a SINGLE vector using mean pooling.

        Mean pooling: embed every character, then average all vectors.
        Result: one vector that represents the whole chunk.

        C# analogy:
          float[] Encode(string chunk) {
              var vecs = chunk.Select(ch => EmbedChar(ch));
              return vecs.Aggregate((a, b) => a.Zip(b, (x, y) => x + y).ToArray())
                         .Select(x => x / chunk.Length).ToArray();
          }

        Why mean pooling?
          Simple. Fast. Works well enough for finding similar chunks.
          More advanced: use CLS token (like BERT). But this is our version.
        """
        # Convert each character to its ID (use 0 for unknown chars)
        ids = torch.tensor(
            [char_to_idx.get(ch, 0) for ch in chunk_text],
            dtype=torch.long
        )

        if len(ids) == 0:
            return torch.zeros(self.emb.embedding_dim)

        vecs = self.emb(ids)        # (seq_len, embed_dim) -- one vector per char
        return vecs.mean(dim=0)     # (embed_dim,) -- average across all chars


# =============================================================================
# STEP 4: Train the model
# =============================================================================

model     = EmbeddingModel(vocab_size, EMBED_DIM)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {n_params:,}")
print(f"Training for {TRAIN_STEPS} steps...")
print()

# Training loop -- same pattern as example_03_bigram_pytorch.py
for step in range(TRAIN_STEPS):
    # Pick random starting positions in the data
    # torch.randint(high, size) = BATCH_SIZE random integers from 0 to high
    starts = torch.randint(len(data) - CONTEXT_LEN - 1, (BATCH_SIZE,))

    # Build batch: x = input chars, y = next chars (shifted by 1)
    x_batch = torch.stack([data[s     : s + CONTEXT_LEN    ] for s in starts])
    y_batch = torch.stack([data[s + 1 : s + CONTEXT_LEN + 1] for s in starts])

    # Forward + backward + step -- the 3-line training loop from module 05
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0 or step == TRAIN_STEPS - 1:
        print(f"  Step {step:4d}: loss = {loss.item():.4f}")

print(f"\nTraining complete. Final loss: {loss.item():.4f}")
print("(Lower loss = model learned character patterns better)")

# =============================================================================
# STEP 5: Save model + vocabulary to disk
# =============================================================================

# Create the saved/ folder if it does not exist
SAVED_DIR.mkdir(parents=True, exist_ok=True)

# Save model weights
# torch.save: serializes a Python object to disk
# model.state_dict(): returns all learned weights as a dictionary
model_path = SAVED_DIR / "embedding_model.pt"
torch.save(model.state_dict(), model_path)
print(f"\nSaved model to    : {model_path}")

# Save vocabulary as JSON (so indexer.py can rebuild the char_to_idx map)
vocab_data = {
    "char_to_idx": char_to_idx,
    "idx_to_char": {str(k): v for k, v in idx_to_char.items()},  # JSON needs string keys
    "vocab_size":  vocab_size,
    "embed_dim":   EMBED_DIM,
}
vocab_path = SAVED_DIR / "vocab.json"
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=2)
print(f"Saved vocabulary to: {vocab_path}")

print("\n" + "=" * 60)
print("Phase 1 DONE.")
print("Next step: run  python indexer.py")
print("=" * 60)

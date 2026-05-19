"""
=============================================================================
PROJECT: Personal Knowledge Base Q&A  --  Phase 3 of 5
FILE   : retriever.py
=============================================================================

WHAT THIS FILE DOES
--------------------
Given a question string, finds the TOP-K most relevant chunks
from the index built in Phase 2, using cosine similarity.

This IS the search engine of the app.

SKILLS REUSED FROM MODULE 05
------------------------------
  example_02_embeddings_pytorch.py  -->  F.cosine_similarity (exact same call)

HOW TO RUN (standalone test)
------------------------------
  python retriever.py

WHAT TO EXPECT
---------------
Runs a few test searches against your notes and shows top matches.

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

SAVED_DIR = Path(__file__).parent / "saved"
TOP_K     = 3      # how many chunks to return per query
MIN_SCORE = 0.3    # minimum similarity score (below this = "not found")

# =============================================================================
# STEP 1: Load vocabulary, model, and index
# =============================================================================

def load_vocab(saved_dir):
    """Load char_to_idx mapping and model settings from vocab.json."""
    with open(saved_dir / "vocab.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_index(saved_dir):
    """
    Load index.json and convert stored vectors back to PyTorch tensors.

    index.json stores vectors as plain Python lists (JSON format).
    We convert them back to tensors for fast cosine similarity math.

    Returns:
      texts   : list of chunk text strings
      sources : list of source filenames
      vectors : 2D tensor of shape (num_chunks, embed_dim)
                One row per chunk. Ready for batch similarity computation.
    """
    with open(saved_dir / "index.json", "r", encoding="utf-8") as f:
        index = json.load(f)

    texts   = [entry["text"]   for entry in index]
    sources = [entry["source"] for entry in index]

    # Stack all vectors into one 2D tensor
    # torch.stack: combines a list of 1D tensors into a 2D tensor
    # C# analogy: converting List<float[]> into float[,]
    vectors = torch.tensor(
        [entry["vector"] for entry in index],
        dtype=torch.float
    )   # shape: (num_chunks, embed_dim)

    return texts, sources, vectors


# Rebuild EmbeddingModel (same as embedder.py and indexer.py)
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
            loss = F.cross_entropy(logits.view(-1, self.emb.num_embeddings), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def encode_chunk(self, chunk_text, char_to_idx):
        ids = torch.tensor(
            [char_to_idx.get(ch, 0) for ch in chunk_text],
            dtype=torch.long
        )
        if len(ids) == 0:
            return torch.zeros(self.emb.embedding_dim)
        return self.emb(ids).mean(dim=0)


def load_everything(saved_dir):
    """Load model + vocabulary + index. Returns (model, char_to_idx, texts, sources, vectors)."""
    for needed in ["vocab.json", "embedding_model.pt", "index.json"]:
        if not (saved_dir / needed).exists():
            print(f"ERROR: {saved_dir / needed} not found.")
            print("Run embedder.py then indexer.py first!")
            exit(1)

    vocab_data  = load_vocab(saved_dir)
    char_to_idx = vocab_data["char_to_idx"]
    vocab_size  = vocab_data["vocab_size"]
    embed_dim   = vocab_data["embed_dim"]

    model = EmbeddingModel(vocab_size, embed_dim)
    model.load_state_dict(torch.load(saved_dir / "embedding_model.pt", map_location="cpu"))
    model.eval()

    texts, sources, vectors = load_index(saved_dir)
    return model, char_to_idx, texts, sources, vectors


# =============================================================================
# STEP 2: The retrieval function
# =============================================================================

def retrieve(question, model, char_to_idx, texts, sources, vectors, top_k=TOP_K, min_score=MIN_SCORE):
    """
    Find the most relevant chunks for a given question.

    HOW IT WORKS:
      1. Embed the question into a vector (same encode_chunk method)
      2. Compute cosine similarity between question vector and every chunk vector
      3. Sort by similarity score (highest = most similar)
      4. Return top_k results

    WHY COSINE SIMILARITY?
      It measures the ANGLE between two vectors, not their length.
      Two chunks about "embeddings" will have vectors pointing in a similar
      direction, even if one chunk is long and one is short.
      Score = 1.0 means identical direction. Score = 0.0 means unrelated.

      Same math as example_02_embeddings_pytorch.py Part 3.

    Parameters:
      question : str  -- the user's question
      top_k    : int  -- how many results to return
      min_score: float -- minimum score threshold (below = "not found")

    Returns:
      list of dicts: [{text, source, score}]  sorted by score descending
    """

    # Embed the question -- same method used to embed every chunk
    question_vec = model.encode_chunk(question, char_to_idx)   # shape: (embed_dim,)

    # Expand question_vec to match vectors shape for batch computation
    # unsqueeze(0) adds a batch dimension: (embed_dim,) -> (1, embed_dim)
    # expand(len(texts), -1) repeats it for every chunk: (num_chunks, embed_dim)
    # C# analogy: making num_chunks copies of the same row in a matrix
    q_expanded = question_vec.unsqueeze(0).expand(len(texts), -1)

    # Compute cosine similarity between question and EVERY chunk at once
    # F.cosine_similarity(a, b, dim=1): compare row by row
    # Returns a 1D tensor of scores: one per chunk
    # Same call as example_02_embeddings_pytorch.py Part 3!
    scores = F.cosine_similarity(q_expanded, vectors, dim=1)   # shape: (num_chunks,)

    # torch.topk: find top_k highest scores and their positions
    # C# analogy: .OrderByDescending(s => s).Take(top_k)
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(texts)))

    results = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        if score >= min_score:
            results.append({
                "text":   texts[idx],
                "source": sources[idx],
                "score":  round(score, 4),
            })

    return results


# =============================================================================
# STEP 3: Test the retriever (runs when you execute this file directly)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3: Testing the Retriever")
    print("=" * 60)

    print("\nLoading model + index...")
    model, char_to_idx, texts, sources, vectors = load_everything(SAVED_DIR)
    print(f"Index loaded: {len(texts)} chunks from your notes")
    print(f"Vector shape: {vectors.shape}  (chunks x embed_dim)")

    # Run some test queries
    test_questions = [
        "what is nn.Embedding?",
        "how does cross entropy loss work?",
        "what is the training loop?",
        "explain cosine similarity",
        "what is backpropagation?",
    ]

    print("\n" + "-" * 60)
    print("TEST SEARCHES")
    print("-" * 60)

    for question in test_questions:
        print(f"\nQuestion: '{question}'")
        results = retrieve(question, model, char_to_idx, texts, sources, vectors)

        if not results:
            print("  No relevant chunks found (all scores below threshold).")
        else:
            for i, r in enumerate(results):
                preview = r["text"][:100].replace("\n", " ")
                print(f"  [{i+1}] score={r['score']:.3f}  [{r['source']}]")
                print(f"       {preview}...")

    print("\n" + "=" * 60)
    print("Phase 3 DONE.")
    print("Next step: run  python app.py  to use the full Q&A app")
    print("=" * 60)

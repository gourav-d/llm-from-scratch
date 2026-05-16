"""
=============================================================================
EXAMPLE 02 (PyTorch Version): Word Embeddings with nn.Embedding
=============================================================================

GLOSSARY
---------
nn.Embedding     : PyTorch's built-in embedding lookup table.
                   Replaces our manual NumPy EmbeddingLayer.
                   Automatically tracks gradients — weights update during training.

F.cosine_similarity : PyTorch function that computes cosine similarity
                      between two tensors. Same math as our manual formula,
                      but handles batches and edge cases automatically.

torch.no_grad()  : A context manager that tells PyTorch "don't track gradients
                   for code inside this block". Use it when you are just
                   inspecting values, not training.
                   C# analogy: like a read-only scope.

detach()         : Detaches a tensor from the gradient-tracking graph.
                   Gives you a plain tensor you can convert to NumPy.
                   Required before calling .numpy() on a tensor.

.numpy()         : Converts a PyTorch tensor to a NumPy array.
                   Needed for matplotlib (which doesn't know about tensors).

=============================================================================
HOW THIS CONNECTS TO THE NumPy VERSION (example_02_word_embeddings.py)
=============================================================================

NumPy version:
  self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.1
  def forward(self, token_ids):
      return self.embeddings[token_ids]     # manual lookup

PyTorch version:
  self.emb = nn.Embedding(vocab_size, embed_dim)
  output   = self.emb(token_ids_tensor)    # same lookup, but tracked!

The output is IDENTICAL math. The difference is that PyTorch tracks every
operation so it can automatically compute gradients during training.

=============================================================================
"""

import torch                        # core PyTorch
import torch.nn as nn               # neural-network layers
import torch.nn.functional as F     # standalone functions (no learnable weights)
import numpy as np                  # still used for PCA + matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("=" * 65)
print("WORD EMBEDDINGS — PyTorch Version")
print("=" * 65)

# =============================================================================
# PART 1: One-Hot Encoding (same concept, shown with PyTorch tensors)
# =============================================================================

print("\n--- PART 1: One-Hot Encoding with PyTorch ---")
print("""
Same concept as NumPy version, but using PyTorch's built-in function.
F.one_hot(tensor, num_classes) converts integer IDs to one-hot vectors.
Result dtype is torch.long (integers).  Cast to float for math.
""")

vocab = ["cat", "dog", "bird", "fish", "python", "java", "rust"]

# Token IDs for the three words we want to compare
cat_id    = torch.tensor(vocab.index("cat"))     # scalar tensor containing 0
dog_id    = torch.tensor(vocab.index("dog"))     # scalar tensor containing 1
python_id = torch.tensor(vocab.index("python"))  # scalar tensor containing 4

# F.one_hot creates one-hot vectors automatically
# num_classes = vocabulary size (number of possible tokens)
cat_oh    = F.one_hot(cat_id,    num_classes=len(vocab)).float()  # .float() for math
dog_oh    = F.one_hot(dog_id,    num_classes=len(vocab)).float()
python_oh = F.one_hot(python_id, num_classes=len(vocab)).float()

print(f"One-hot 'cat'    : {cat_oh}")
print(f"One-hot 'dog'    : {dog_oh}")
print(f"One-hot 'python' : {python_oh}")

# Dot product similarity: torch.dot replaces np.dot
# For one-hot vectors, dot product is always 0 (no overlap unless same word)
print(f"\nSimilarity 'cat' vs 'dog'    : {torch.dot(cat_oh, dog_oh).item():.0f}")
print(f"Similarity 'cat' vs 'python' : {torch.dot(cat_oh, python_oh).item():.0f}")
print("Both 0 — one-hot has no notion of similarity!")

# =============================================================================
# PART 2: Dense Embeddings with nn.Embedding
# =============================================================================

print("\n--- PART 2: Dense Embeddings with nn.Embedding ---")
print("""
nn.Embedding is PyTorch's official embedding layer.
It replaces our manual EmbeddingLayer class from the NumPy version.

  nn.Embedding(num_embeddings, embedding_dim)
    num_embeddings = vocabulary size (how many rows in the table)
    embedding_dim  = vector size per token (how many columns)

  Internally it is:
    weight = torch.randn(vocab_size, embed_dim) * small_value

  The weight matrix is LEARNABLE — it updates during training.
  In NumPy we had to update it manually. PyTorch does it automatically.
""")

embed_dim = 5   # small so we can print it

# Create the embedding layer
# vocab_size rows, embed_dim columns — one vector per token
emb_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_dim)

print(f"Embedding table shape: {emb_layer.weight.shape}")
print(f"  {len(vocab)} rows (one per token) x {embed_dim} columns (embed dim)")

# Retrieve embeddings for specific tokens
# We use torch.no_grad() because we are just inspecting, not training
with torch.no_grad():
    # emb_layer() accepts a tensor of token IDs
    # For a single token, wrap the ID in a 1-element tensor
    cat_emb    = emb_layer(torch.tensor([cat_id]))     # shape: (1, 5)
    dog_emb    = emb_layer(torch.tensor([dog_id]))
    python_emb = emb_layer(torch.tensor([python_id]))

print(f"\nDense embedding for 'cat'    : {cat_emb}")
print(f"Dense embedding for 'dog'    : {dog_emb}")
print(f"Dense embedding for 'python' : {python_emb}")
print(f"\nAll are compact {embed_dim}-dimensional vectors (not {len(vocab)}-dimensional like one-hot!)")

# =============================================================================
# PART 3: Cosine Similarity with F.cosine_similarity
# =============================================================================

print("\n--- PART 3: Cosine Similarity ---")
print("""
NumPy version: we wrote cosine_similarity() manually.
PyTorch version: F.cosine_similarity() handles it.

  F.cosine_similarity(x1, x2, dim=1)
    x1, x2 = tensors of the SAME shape
    dim=1   = compute similarity along dimension 1 (the vector dimension)
    output  = scalar value between -1 and +1

  Same formula: cos(theta) = (A . B) / (||A|| * ||B||)
  But PyTorch is vectorized — it handles edge cases like zero vectors.
""")

with torch.no_grad():
    # F.cosine_similarity expects tensors of shape (N, D) or (D,)
    # Our embeddings are shape (1, 5) — that works with dim=1
    sim_cat_dog    = F.cosine_similarity(cat_emb, dog_emb, dim=1)
    sim_cat_python = F.cosine_similarity(cat_emb, python_emb, dim=1)
    sim_dog_python = F.cosine_similarity(dog_emb, python_emb, dim=1)

# .item() extracts the Python float from a 1-element tensor
print(f"Cosine similarity 'cat' vs 'dog'    : {sim_cat_dog.item():.3f}")
print(f"Cosine similarity 'cat' vs 'python' : {sim_cat_python.item():.3f}")
print(f"Cosine similarity 'dog' vs 'python' : {sim_dog_python.item():.3f}")
print("(Random init — values change every run. After training they become meaningful.)")

# =============================================================================
# PART 4: Trained Embeddings (Simulated) — same as NumPy version
# =============================================================================

print("\n--- PART 4: Simulated Trained Embeddings ---")
print("""
We now create embeddings with HAND-CRAFTED values to simulate what a
trained model learns. Dimensions represent [royalty, gender_male, age_adult].

We store them directly in nn.Embedding.weight so PyTorch can work with them.
This shows how to LOAD pre-trained embeddings into PyTorch.
""")

vocab_sem = ["king", "queen", "prince", "princess", "man", "woman", "boy", "girl"]
embed_dim_sem = 3   # [royalty, gender_male, age_adult]

# Hand-crafted embedding values — same as NumPy version
pretrained_vectors = torch.tensor([
    [ 0.9,  0.9,  0.9],   # king     [royal,  male,   adult]
    [ 0.9, -0.9,  0.9],   # queen    [royal,  female, adult]
    [ 0.8,  0.8,  0.3],   # prince   [royal,  male,   young]
    [ 0.8, -0.8,  0.3],   # princess [royal,  female, young]
    [ 0.0,  0.9,  0.9],   # man      [common, male,   adult]
    [ 0.0, -0.9,  0.9],   # woman    [common, female, adult]
    [ 0.0,  0.8,  0.3],   # boy      [common, male,   young]
    [ 0.0, -0.8,  0.3],   # girl     [common, female, young]
], dtype=torch.float)

# Create nn.Embedding and load pre-trained vectors
# from_pretrained() sets the weights AND freezes them (freeze=True by default)
# freeze=True means these weights won't change during training
emb_sem = nn.Embedding.from_pretrained(pretrained_vectors, freeze=True)

print(f"Embedding table shape: {emb_sem.weight.shape}")
print(f"  {len(vocab_sem)} words x {embed_dim_sem} dimensions")
print()

# Look up embeddings for all words
ids_all = torch.arange(len(vocab_sem))   # tensor([0, 1, 2, ..., 7])
with torch.no_grad():
    all_emb = emb_sem(ids_all)           # shape: (8, 3)

for word, vec in zip(vocab_sem, all_emb):
    print(f"  {word:10s}: {vec.tolist()}")

# Compute pairwise similarities
print("\nSimilarities (trained/simulated embeddings):")
pairs = [("king", "queen"), ("king", "man"), ("queen", "woman"), ("prince", "princess")]
for w1, w2 in pairs:
    id1 = torch.tensor([vocab_sem.index(w1)])
    id2 = torch.tensor([vocab_sem.index(w2)])
    with torch.no_grad():
        v1 = emb_sem(id1)   # shape (1, 3)
        v2 = emb_sem(id2)   # shape (1, 3)
        sim = F.cosine_similarity(v1, v2, dim=1).item()
    print(f"  {w1:10s} <-> {w2:10s} : {sim:.3f}")

# =============================================================================
# PART 5: Word Analogies — king - man + woman = queen (in PyTorch)
# =============================================================================

print("\n--- PART 5: Word Analogy — king - man + woman = ? ---")
print("""
Same concept as NumPy version, but using PyTorch tensor arithmetic.
torch tensors support +, -, *, / directly (just like NumPy arrays).
""")

with torch.no_grad():
    king_vec  = emb_sem(torch.tensor([vocab_sem.index("king")]))    # (1, 3)
    man_vec   = emb_sem(torch.tensor([vocab_sem.index("man")]))     # (1, 3)
    woman_vec = emb_sem(torch.tensor([vocab_sem.index("woman")]))   # (1, 3)

    # Tensor arithmetic — same as NumPy: +, - work element-wise
    result_vec = king_vec - man_vec + woman_vec    # (1, 3)

    print(f"king    : {king_vec.squeeze().tolist()}")   # .squeeze() removes the batch dim
    print(f"man     : {man_vec.squeeze().tolist()}")
    print(f"woman   : {woman_vec.squeeze().tolist()}")
    print(f"result  : {result_vec.squeeze().tolist()}")

    # Find which word is closest to result_vec
    all_vecs = emb_sem(ids_all)    # (8, 3) — all embeddings

    # Expand result_vec to match shape (8, 3) for batch similarity
    # repeat(8, 1) makes 8 copies of the (1, 3) vector along dim 0
    result_expanded = result_vec.repeat(8, 1)              # (8, 3)

    similarities = F.cosine_similarity(result_expanded, all_vecs, dim=1)  # (8,)

    # Exclude king, man, woman from the result
    exclude = {"king", "man", "woman"}
    best_sim = -2.0
    best_word = None
    for i, word in enumerate(vocab_sem):
        if word in exclude:
            continue
        if similarities[i].item() > best_sim:
            best_sim = similarities[i].item()
            best_word = word

print(f"\nking - man + woman  =  '{best_word}'  (similarity: {best_sim:.3f})")
print("Expected: 'queen'")

# =============================================================================
# PART 6: Visualizing Embeddings in 2D (same as NumPy version)
# =============================================================================

print("\n--- PART 6: Visualization (PCA to 2D) ---")
print("""
matplotlib doesn't understand PyTorch tensors directly.
We must convert to NumPy first using:
    tensor.detach().numpy()

  detach() : remove the tensor from gradient tracking
  .numpy() : convert to NumPy array

After that, PCA and matplotlib work exactly as before.
""")

with torch.no_grad():
    vecs_np = all_emb.numpy()   # convert (8, 3) tensor to NumPy array

pca = PCA(n_components=2)
vecs_2d = pca.fit_transform(vecs_np)

plt.figure(figsize=(8, 6))
plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], s=100, alpha=0.6, color='steelblue')
for i, word in enumerate(vocab_sem):
    plt.annotate(word, (vecs_2d[i, 0], vecs_2d[i, 1]), fontsize=12, ha='center')

plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("Word Embeddings (PyTorch) — 2D Visualization")
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig("word_embeddings_pytorch_2d.png", dpi=150, bbox_inches="tight")
print("Saved: word_embeddings_pytorch_2d.png")
print("Royal words (king/queen/prince/princess) should cluster together.")

# =============================================================================
# PART 7: Batch Embedding Lookup — how GPT uses it every forward pass
# =============================================================================

print("\n--- PART 7: Batch Embedding Lookup (how GPT uses it) ---")
print("""
In a real GPT model, the embedding layer receives an entire BATCH of
sequences at once — not one token at a time.

Input shape : (batch_size, seq_len)      — integers (token IDs)
Output shape: (batch_size, seq_len, dim) — floats (embedding vectors)

PyTorch handles the 2D input automatically.
""")

# Simulate a batch of 3 sentences, each 5 tokens long
batch_token_ids = torch.tensor([
    [2, 4, 1, 6, 0],    # sentence 1 (5 token IDs)
    [3, 5, 2, 0, 0],    # sentence 2 (last two are PAD)
    [1, 2, 3, 4, 5],    # sentence 3
], dtype=torch.long)

# Use a fresh embedding layer with dim=6 for this demo
batch_emb_layer = nn.Embedding(num_embeddings=8, embedding_dim=6, padding_idx=0)

with torch.no_grad():
    batch_output = batch_emb_layer(batch_token_ids)   # (3, 5, 6)

print(f"Input  shape : {batch_token_ids.shape}  (3 sentences, 5 tokens each)")
print(f"Output shape : {batch_output.shape}  (3 sentences, 5 tokens, 6 dims each)")
print("\nEmbedding for sentence 0, token position 0:")
print(batch_output[0, 0])
print("\nEmbedding for sentence 1, position 3 (PAD token — should be all zeros):")
print(batch_output[1, 3])

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("SUMMARY: NumPy vs PyTorch Embeddings")
print("=" * 65)
print("""
  NumPy (example_02):                  PyTorch (this file):
  ----------------------------          ----------------------------
  np.random.randn(V, D) * 0.1          nn.Embedding(V, D)
  embeddings[token_ids]                 emb_layer(token_id_tensor)
  np.dot(v1, v2) / norms               F.cosine_similarity(v1, v2)
  Manual weight update                  Automatic via loss.backward()
  CPU only                              CPU or GPU (model.to('cuda'))
  Can't track gradients                 Tracks gradients automatically

  The MATH is identical.
  PyTorch just handles memory, gradients, and GPU for you.

New PyTorch tools learned:
  nn.Embedding(V, D)              learnable lookup table
  nn.Embedding.from_pretrained()  load existing vectors
  F.cosine_similarity(a, b, dim)  cosine similarity on tensors
  tensor.detach().numpy()         convert tensor → NumPy for plotting
  torch.no_grad()                 skip gradient tracking (inference/inspect)
""")

print("=" * 65)
print("Run example_03_bigram_pytorch.py next!")
print("=" * 65)

try:
    plt.show()
except Exception:
    pass

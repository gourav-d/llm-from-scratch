"""
=============================================================================
EXAMPLE 03D: Trigram Model -- Extending Context from 1 to 2 Tokens
=============================================================================

GLOSSARY
---------
N-gram         : A model that predicts the next token given N-1 previous tokens.
                   Bigram  (N=2): predict next token given 1 previous token
                   Trigram (N=3): predict next token given 2 previous tokens
                   4-gram  (N=4): predict next token given 3 previous tokens
                   GPT     (N=?): predict next token given ALL previous tokens

Context window : How many previous tokens the model looks at.
                   Bigram context  = 1   (sees "meeting",   predicts next)
                   Trigram context = 2   (sees "the meeting", predicts next)
                   GPT-4 context   = 128,000 tokens

Concatenation  : How we represent 2-token context for the model.
                   Bigram:  embed(token_A)
                   Trigram: concat(embed(token_A), embed(token_B))
                   We join the two embeddings into one longer vector.

nn.Linear      : A fully-connected layer. Applies: output = input @ weight + bias
                 C# analogy: matrix multiplication + offset.
                 Learns to combine two token embeddings into next-token scores.

=============================================================================
THE BIG IDEA: CONTEXT WINDOW
=============================================================================

This is the single most important concept on the path from bigram to GPT.

  Bigram:   "the" -> ?
            Model knows only "the". Could be "the meeting", "the cat", anything.

  Trigram:  "the meeting" -> ?
            Model knows TWO words. "is" is now much more likely than in bigram.

  GPT:      "The 9am meeting with the sales team is" -> ?
            Model knows the ENTIRE sentence. "scheduled" is now very probable.

Each step up increases the context window -- and the accuracy.
The transformer's attention mechanism is how GPT handles unlimited context.

=============================================================================
ARCHITECTURE CHANGE: BIGRAM vs TRIGRAM
=============================================================================

BIGRAM:
  Input:  1 token ID    (e.g., 5)
  Embed:  embed(5)      shape: (embed_dim,)
  Output: scores        shape: (vocab_size,)
  Model:  nn.Embedding  (lookup table only)

TRIGRAM:
  Input:  2 token IDs   (e.g., [5, 7])
  Embed:  embed(5), embed(7)  each shape: (embed_dim,)
  Concat: [embed(5), embed(7)]   shape: (2 * embed_dim,)
  Linear: matrix multiply    shape: (vocab_size,)
  Model:  nn.Embedding + nn.Linear

The nn.Linear layer learns HOW to combine two token embeddings
to make a good prediction. It learns relationships like:
  "the" + "meeting" together -> "is" is very likely
  "the" + "project" together -> "is" or "has" is likely

=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("EXAMPLE 03D: Trigram -- 2-Token Context Window")
print("=" * 60)

# =============================================================================
# STEP 1: Training data (same as 03b, so comparison is fair)
# =============================================================================

print("""
STEP 1: Training Data
Using the SAME office-phrase sentences as example_03b.
This lets us directly compare bigram vs trigram quality.
""")

sentences = [
    "the budget is approved today",
    "the budget is rejected today",
    "the client is happy today",
    "the client is waiting today",
    "the deadline is tomorrow",
    "the deadline is today",
    "the team is busy today",
    "the team is ready today",
    "the server is down today",
    "the server is running today",
    "the feature is done today",
    "the feature is broken today",
    "the release is scheduled tomorrow",
    "the release is delayed tomorrow",
    "the deployment is complete today",
    "the deployment is running today",
    "the ticket is closed today",
    "the ticket is open today",
    "the review is pending today",
    "the review is approved today",
]

print(f"Sentences: {len(sentences)}")

# =============================================================================
# STEP 2: Vocabulary (same process as bigram)
# =============================================================================

all_words = [w for s in sentences for w in s.split()]
vocab = sorted(set(all_words))
vocab_size = len(vocab)
w2i = {w: i for i, w in enumerate(vocab)}   # word -> ID
i2w = {i: w for i, w in enumerate(vocab)}   # ID -> word

print(f"Vocab size: {vocab_size} unique words")
print(f"Words: {vocab}")

# =============================================================================
# STEP 3: Build TRIGRAM training pairs
# =============================================================================

print("""
STEP 3: Trigram Training Pairs
Unlike bigram (1 input -> 1 target), trigram has:
  2 inputs  -> 1 target

From "the budget is approved today":
  ("the", "budget")  -> "is"
  ("budget", "is")   -> "approved"
  ("is", "approved") -> "today"

We need 2 input IDs and 1 target ID per training example.
""")

x1_list = []   # first  token of the 2-token context
x2_list = []   # second token of the 2-token context
y_list  = []   # target token (what follows the 2-token context)

for sentence in sentences:
    words = sentence.split()
    # Need at least 3 words to form one trigram training example
    for i in range(len(words) - 2):
        x1_list.append(w2i[words[i]])      # first context word
        x2_list.append(w2i[words[i + 1]]) # second context word
        y_list.append( w2i[words[i + 2]]) # target word

x1_train = torch.tensor(x1_list, dtype=torch.long)
x2_train = torch.tensor(x2_list, dtype=torch.long)
y_train  = torch.tensor(y_list,  dtype=torch.long)

print(f"Total trigram training pairs: {len(y_train)}")
print("\nFirst 6 trigram pairs (word1, word2) -> next_word:")
for i in range(6):
    w1  = i2w[x1_train[i].item()]
    w2  = i2w[x2_train[i].item()]
    tgt = i2w[y_train[i].item()]
    print(f"  ('{w1}', '{w2}') -> '{tgt}'")

# =============================================================================
# STEP 4: Define the Trigram Model
# =============================================================================

print("""
STEP 4: TrigramModel Architecture
Two embedding lookups, then concatenate, then one Linear layer.

  token_1 ID  --embed--> vector_1  (shape: embed_dim)
  token_2 ID  --embed--> vector_2  (shape: embed_dim)
                concat-> combined  (shape: 2 * embed_dim)
                linear-> logits    (shape: vocab_size)

The nn.Linear layer is the new part vs bigram.
It learns: "given these two token embeddings together, what comes next?"
""")

EMBED_DIM = 32   # size of each token's embedding vector
# C# analogy: each word maps to a float[32] array (its "meaning vector")

class TrigramModel(nn.Module):
    """
    Predicts next token given 2 previous tokens.
    Context window = 2 (vs 1 for bigram).

    Architecture:
      embed1 + embed2 -> concatenate -> Linear -> logits
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        # Single embedding table for BOTH input positions
        # Both tokens share the same embedding space
        # Row i = embedding vector for token i (shape: embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Linear layer: maps (2 * embed_dim) -> vocab_size
        # This is the "prediction head" that combines both token embeddings
        # nn.Linear(in_features, out_features) applies: output = input @ W + b
        self.linear = nn.Linear(embed_dim * 2, vocab_size)

    def forward(self, x1, x2, targets=None):
        """
        x1      : first  context token IDs, shape (N,)
        x2      : second context token IDs, shape (N,)
        targets : target token IDs,         shape (N,)  -- optional
        """
        # Look up embeddings for each token -- shape: (N, embed_dim)
        emb1 = self.embedding(x1)
        emb2 = self.embedding(x2)

        # Concatenate along feature dimension:
        # (N, embed_dim) + (N, embed_dim) -> (N, 2 * embed_dim)
        # C# analogy: Enumerable.Concat(emb1_row, emb2_row).ToArray()
        combined = torch.cat([emb1, emb2], dim=1)

        # Linear layer produces logits (raw scores per vocab token)
        # (N, 2 * embed_dim) -> (N, vocab_size)
        logits = self.linear(combined)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def predict_top_k(self, word1, word2, k):
        """
        Given a 2-word context, return top-k likely next words.

        word1, word2 : string tokens (not IDs)
        k            : number of suggestions
        """
        if word1 not in w2i or word2 not in w2i:
            return []
        x1 = torch.tensor([w2i[word1]], dtype=torch.long)
        x2 = torch.tensor([w2i[word2]], dtype=torch.long)
        logits, _ = self(x1, x2)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], k)
        return [(i2w[top_indices[i].item()], top_probs[i].item()) for i in range(k)]

    @torch.no_grad()
    def generate_sentence(self, start_word1, start_word2, length):
        """
        Generate a sentence given the first TWO seed words.
        Each step the 2-token context slides forward by 1.
        """
        if start_word1 not in w2i or start_word2 not in w2i:
            return "(start words not in vocab)"
        words = [start_word1, start_word2]
        for _ in range(length - 2):
            x1 = torch.tensor([w2i[words[-2]]], dtype=torch.long)  # second-to-last
            x2 = torch.tensor([w2i[words[-1]]], dtype=torch.long)  # last
            logits, _ = self(x1, x2)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs[0], num_samples=1).item()
            words.append(i2w[next_id])
        return " ".join(words)


trigram_model = TrigramModel(vocab_size, EMBED_DIM)
n_params = sum(p.numel() for p in trigram_model.parameters())
print(f"\nTrigramModel created!")
print(f"  Vocab size     : {vocab_size}")
print(f"  Embedding dim  : {EMBED_DIM}")
print(f"  Embedding table: {vocab_size} x {EMBED_DIM} = {vocab_size * EMBED_DIM} values")
print(f"  Linear layer   : ({EMBED_DIM * 2}) -> {vocab_size} = {EMBED_DIM * 2 * vocab_size} values")
print(f"  Total params   : {n_params}")

# =============================================================================
# STEP 5: Train the Trigram Model
# =============================================================================

optimizer = torch.optim.AdamW(trigram_model.parameters(), lr=0.05)

print("\nTraining for 800 steps...")
for step in range(800):
    logits, loss = trigram_model(x1_train, x2_train, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"  Step {step:3d}: loss = {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.4f}")

# =============================================================================
# STEP 6: Compare Bigram vs Trigram predictions
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Bigram vs Trigram -- Side by Side")
print("=" * 60)
print("""
The key advantage of trigram: it uses 2-word context.

  Bigram  asks: "given 'is', what comes next?"
  Trigram asks: "given 'budget is', what comes next?"

With more context, the model can be more precise.
""")

# --- Train a bigram model on the same data for comparison ---
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def predict_top_k(self, word, k):
        if word not in w2i:
            return []
        idx = torch.tensor([w2i[word]], dtype=torch.long)
        logits, _ = self(idx)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], k)
        return [(i2w[top_indices[i].item()], top_probs[i].item()) for i in range(k)]

# Build bigram training pairs
bx = torch.tensor([w2i[s.split()[i]]     for s in sentences for i in range(len(s.split())-1)], dtype=torch.long)
by = torch.tensor([w2i[s.split()[i+1]]   for s in sentences for i in range(len(s.split())-1)], dtype=torch.long)

bigram_model = BigramModel(vocab_size)
opt_b = torch.optim.AdamW(bigram_model.parameters(), lr=0.05)
for step in range(500):
    _, loss_b = bigram_model(bx, by)
    opt_b.zero_grad()
    loss_b.backward()
    opt_b.step()

print(f"(Bigram trained for comparison. Final loss: {loss_b.item():.4f})\n")

# --- Side-by-side comparison ---
comparisons = [
    ("budget", "is"),
    ("server", "is"),
    ("release", "is"),
    ("review", "is"),
    ("the", "deadline"),
]

print(f"{'Context':<25} {'Bigram (1 token context)':<35} {'Trigram (2 token context)'}")
print("-" * 90)
for w1, w2 in comparisons:
    # Bigram only sees the LAST word (w2)
    bigram_sug = bigram_model.predict_top_k(w2, k=3)
    bigram_str = "  ".join(f"'{t}'({p*100:.0f}%)" for t, p in bigram_sug)

    # Trigram sees BOTH words (w1 + w2)
    trigram_sug = trigram_model.predict_top_k(w1, w2, k=3)
    trigram_str = "  ".join(f"'{t}'({p*100:.0f}%)" for t, p in trigram_sug)

    context = f"'{w1}' '{w2}'"
    print(f"{context:<25} {bigram_str:<35} {trigram_str}")

print("""
WHAT TO NOTICE:
  Bigram sees only 'is' -> can suggest anything that follows 'is'.
  Trigram sees 'budget is' -> should strongly favor 'approved' or 'rejected'
  Trigram sees 'server is' -> should strongly favor 'down' or 'running'

  With 2 words of context, predictions become domain-specific.
""")

# =============================================================================
# STEP 7: Generate sentences with 2-word seeds
# =============================================================================

print("=" * 60)
print("STEP 7: Generate Sentences (trigram, sliding context window)")
print("=" * 60)
print("""
Generation with trigram:
  Seed: "the server"
  Step 1: context = ("the", "server") -> predict "is"
  Step 2: context = ("server", "is")  -> predict "down" or "running"
  Step 3: context = ("is", "down")    -> predict "today"
  Result: "the server is down today"

The context window SLIDES forward each step.
This is the same sliding window idea inside GPT -- just much larger.
""")

seeds = [
    ("the", "server"),
    ("the", "budget"),
    ("the", "release"),
    ("the", "review"),
]

for w1, w2 in seeds:
    for trial in range(2):
        sentence = trigram_model.generate_sentence(w1, w2, length=5)
        print(f"  Seed ('{w1}','{w2}') trial {trial+1}: {sentence}")
    print()

# =============================================================================
# STEP 8: The Path from Trigram to GPT
# =============================================================================

print("=" * 60)
print("STEP 8: The Path from Trigram to GPT")
print("=" * 60)
print("""
  Model       | Context  | How context is handled
  ------------|----------|-------------------------------
  Bigram      | 1 token  | Single embedding lookup
  Trigram     | 2 tokens | Concat 2 embeddings + Linear
  4-gram      | 3 tokens | Concat 3 embeddings + Linear
  N-gram      | N-1 tok  | Concat N-1 embeddings + Linear
  GPT (small) | 512 tok  | Attention over all 512 tokens
  GPT-4       | 128K tok | Attention over all 128K tokens

PROBLEM with N-gram as N grows:
  Trigram: 2 * embed_dim -> vocab_size  (small Linear layer)
  10-gram: 9 * embed_dim -> vocab_size  (big Linear, still OK)
  1000-gram: 999 * embed_dim -> vocab_size  (huge, but still possible)

  The real problem: you need to have seen EVERY possible N-gram.
  "the meeting is scheduled" is a 4-gram.
  If training data never had this exact 4-gram, the model guesses randomly.
  N-grams don't GENERALIZE -- they just memorize patterns.

WHAT ATTENTION SOLVES:
  Instead of concatenating fixed-position embeddings,
  attention DYNAMICALLY weights which tokens to focus on.
  It can learn: "the token 'server' 3 positions back is important
                 for predicting what comes after 'is'"
  And it generalizes to sequences it has never seen before.

  That leap from concat+linear -> attention is what example_04_gpt_pytorch.py covers.
""")

print("=" * 60)
print("Next: example_03e_perplexity.py -- score sentences with bigram probabilities")
print("=" * 60)

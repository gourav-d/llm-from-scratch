"""
=============================================================================
EXAMPLE 03B: Word-Level Bigram -- Phone Keyboard Autocomplete
=============================================================================

GLOSSARY
---------
Word-level tokenization : Split text into WORDS, not characters.
                          "the cat sat" -> ["the", "cat", "sat"]
                          Vocab = all unique WORDS (not letters).
                          C# analogy: sentence.Split(' ')

Word-level bigram       : Predict the next WORD given the current WORD.
                          Previous examples predicted the next CHARACTER.
                          This predicts the next WORD.

vocab_size              : Number of unique WORDS (not characters).
                          Sentence "the cat sat" has vocab_size = 3, not 11.

nn.Embedding            : Same learnable table as before.
                          Now each ROW = one WORD (not one character).
                          Row for "the" holds scores for all possible next words.

logits                  : Raw unnormalized scores from the model.
                          One score per word in the vocabulary.

top-k                   : Instead of one prediction, show the top 3 most
                          likely next words. This is exactly what your phone
                          keyboard shows above the keys.

=============================================================================
HOW THIS DIFFERS FROM example_03_bigram_pytorch.py
=============================================================================

  example_03_bigram_pytorch.py          THIS FILE
  ----------------------------          ----------------------------------
  Token    = single character           Token    = whole word
  Vocab    : e, h, l, o  (4 tokens)    Vocab    : the, meeting, is, ... (N)
  Predicts : next CHARACTER             Predicts : next WORD
  Use case : character-level writing    Use case : word autocomplete

Real-world picture:
  You type "the meeting is"  on your phone.
  Phone keyboard shows: "scheduled"  "cancelled"  "done"
  That is a word-level bigram (simplified).

  GPT does the same thing, just with billions of words in training data
  and attention layers that remember more than one word back.

=============================================================================
"""

import torch                        # core PyTorch library
import torch.nn as nn               # neural network building blocks
import torch.nn.functional as F     # standalone functions (softmax, cross_entropy)

print("=" * 60)
print("PART A: Word-Level Bigram on Office Phrases")
print("=" * 60)

print("""
Scenario: You are building a word autocomplete feature for a work chat app.
Users type short phrases about meetings, tasks, and schedules.
The model learns: after "the" -> likely "meeting" or "report"
                  after "is"  -> likely "scheduled" or "done"
""")

# =============================================================================
# STEP 1: Training sentences (word-level, not char-level)
# =============================================================================

# Six short office-style sentences
# Each sentence is a list of words (already split on spaces)
sentences = [
    "the meeting is scheduled tomorrow",
    "the meeting is cancelled today",
    "the report is due tomorrow",
    "the report is complete today",
    "the project is delayed tomorrow",
    "the project is done today",
]

print("Training sentences:")
for s in sentences:
    print(f"  {s}")

# =============================================================================
# STEP 2: Build word vocabulary
# =============================================================================

print("\n--- STEP 2: Word Vocabulary ---")
print("""
Character bigram: vocab = individual letters  (e, h, l, o ...)
Word bigram:      vocab = individual words    (the, meeting, is ...)

We collect all unique words, sort them, assign IDs.
Same idea as char_to_idx -- just words instead of letters.
""")

# Flatten all sentences into one big list of words
# C# analogy: sentences.SelectMany(s => s.Split(' ')).Distinct().OrderBy(w => w)
all_words = [word for sentence in sentences for word in sentence.split()]

# Get unique words, sorted so IDs are consistent across runs
vocab = sorted(set(all_words))
vocab_size = len(vocab)

# Build lookup dictionaries
word_to_idx = {w: i for i, w in enumerate(vocab)}   # word  -> integer ID
idx_to_word = {i: w for i, w in enumerate(vocab)}   # integer ID -> word

print(f"Total words in training data : {len(all_words)}")
print(f"Unique words (vocab size)    : {vocab_size}")
print(f"Word -> ID mapping:")
for w, i in word_to_idx.items():
    print(f"  '{w}' -> {i}")

# =============================================================================
# STEP 3: Encode sentences into (input, target) training pairs
# =============================================================================

print("\n--- STEP 3: Training Pairs (Word -> Next Word) ---")
print("""
Same idea as character bigram:
  Input  = current word
  Target = next word

From sentence "the meeting is scheduled tomorrow":
  'the'       -> 'meeting'
  'meeting'   -> 'is'
  'is'        -> 'scheduled'
  'scheduled' -> 'tomorrow'

We do this for ALL sentences and collect all pairs.
""")

# Build lists of (current_word_id, next_word_id) pairs
x_list = []   # input word IDs
y_list = []   # target word IDs

for sentence in sentences:
    words = sentence.split()          # split sentence into list of words
    for i in range(len(words) - 1):   # iterate each consecutive word pair
        x_list.append(word_to_idx[words[i]])       # current word ID
        y_list.append(word_to_idx[words[i + 1]])   # next word ID

# Convert to PyTorch tensors (dtype=long because these are integer IDs)
x_train = torch.tensor(x_list, dtype=torch.long)
y_train = torch.tensor(y_list, dtype=torch.long)

print(f"Total training pairs: {len(x_train)}")
print("\nFirst 6 pairs (word -> next word):")
for i in range(6):
    src = idx_to_word[x_train[i].item()]   # .item() extracts Python int from tensor
    tgt = idx_to_word[y_train[i].item()]
    print(f"  '{src}' ({x_train[i].item()}) -> '{tgt}' ({y_train[i].item()})")

# =============================================================================
# STEP 4: Define the Bigram Model (identical structure to char-level)
# =============================================================================

print("\n--- STEP 4: BigramModel (same class, bigger vocab) ---")
print("""
The model is IDENTICAL to example_03_bigram_pytorch.py.
Only vocab_size is different (words not chars).

This is the key insight: the neural network does not care whether
tokens are characters or words. It just sees integer IDs.

  Char bigram:  vocab_size = 4   (e, h, l, o)
  Word bigram:  vocab_size = 9   (cancelled, complete, done, ...)

Same nn.Module. Same training loop. Same generation logic.
""")

class BigramModel(nn.Module):
    """
    Predicts the next token (character OR word) given the current token.
    The model is a single nn.Embedding table:
      - Row i = scores for all possible next tokens when current token is i
    """

    def __init__(self, vocab_size):
        super().__init__()   # required: connects this class to PyTorch internals

        # One embedding table: shape (vocab_size, vocab_size)
        # Row i holds scores for: "what word comes after word i?"
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx     : input token IDs,  shape (N,)
        targets : target token IDs, shape (N,)  -- optional, only during training
        returns : (logits, loss)
        """
        logits = self.table(idx)   # look up row for each input ID -- shape (N, vocab_size)

        loss = None
        if targets is not None:
            # cross_entropy needs logits shape (N, C) and targets shape (N,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()   # turn off gradient tracking -- not needed during prediction
    def predict_top_k(self, word_idx, k, idx_to_word):
        """
        Given a word ID, return the top-k most likely next words.
        This is exactly what phone keyboard shows above the typing area.

        word_idx  : integer ID of the current word
        k         : how many suggestions to return (e.g. 3)
        returns   : list of (word, probability) tuples, sorted by probability
        """
        # Build a (1,) tensor from the single word ID
        idx_tensor = torch.tensor([word_idx], dtype=torch.long)

        # Forward pass -- no targets needed (we just want predictions, not loss)
        logits, _ = self(idx_tensor)   # logits shape: (1, vocab_size)

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)   # shape: (1, vocab_size), values sum to 1.0

        # Get top-k probabilities and their indices
        # torch.topk returns (values, indices) of the k largest elements
        top_probs, top_indices = torch.topk(probs[0], k)   # probs[0]: squeeze batch dim

        # Build a readable list: [(word, probability), ...]
        results = []
        for i in range(k):
            word  = idx_to_word[top_indices[i].item()]   # .item() converts tensor -> Python int
            prob  = top_probs[i].item()                   # probability as Python float
            results.append((word, prob))

        return results   # already sorted highest -> lowest by torch.topk


# =============================================================================
# STEP 5: Create model and train
# =============================================================================

model = BigramModel(vocab_size)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model created!")
print(f"  Vocab size       : {vocab_size}  (words, not chars)")
print(f"  Table shape      : {model.table.weight.shape}  ({vocab_size} x {vocab_size})")
print(f"  Trainable params : {n_params}")
print()

# AdamW = better optimizer than SGD for language tasks
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

print("Training for 300 steps...")
for step in range(300):
    logits, loss = model(x_train, y_train)   # forward pass
    optimizer.zero_grad()                    # clear old gradients
    loss.backward()                          # compute new gradients (automatic)
    optimizer.step()                         # update weights

    if step % 75 == 0:
        print(f"  Step {step:3d}: loss = {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.4f}")

# =============================================================================
# STEP 6: Phone keyboard autocomplete demo
# =============================================================================

print("\n--- STEP 6: Autocomplete Demo ---")
print("""
After training, ask the model:
  "What are the top 3 words likely to follow THIS word?"

That is phone keyboard autocomplete.
""")

# Test a few words from our vocabulary
demo_words = ["the", "is", "meeting", "report"]

for word in demo_words:
    if word not in word_to_idx:
        print(f"  '{word}' not in vocabulary, skipping")
        continue

    word_id = word_to_idx[word]
    suggestions = model.predict_top_k(word_id, k=3, idx_to_word=idx_to_word)

    # Format: "the" --> "meeting" (42%)  "report" (35%)  "project" (23%)
    suggestion_str = "  ".join(f"'{w}' ({p*100:.0f}%)" for w, p in suggestions)
    print(f"  '{word}' -->  {suggestion_str}")

# =============================================================================
# STEP 7: Generate a full sentence word by word
# =============================================================================

print("\n--- STEP 7: Generate a Full Sentence ---")
print("""
Same autoregressive loop as before, but word-by-word instead of char-by-char.

  Step 1: start with "the"
  Step 2: model predicts next word, sample from distribution
  Step 3: append predicted word, use it as new input
  Step 4: repeat until desired length
""")

@torch.no_grad()   # generation does not need gradients
def generate_sentence(model, start_word, num_words, word_to_idx, idx_to_word):
    """
    Generate a sentence word by word starting from start_word.
    Same logic as char-level generate(), just words instead of chars.
    """
    if start_word not in word_to_idx:
        return f"('{start_word}' not in vocabulary)"

    current_id = word_to_idx[start_word]      # integer ID of start word
    result = [start_word]                      # start the word list

    for _ in range(num_words - 1):
        # Forward pass for single word
        idx_tensor = torch.tensor([current_id], dtype=torch.long)
        logits, _ = model(idx_tensor)                 # logits shape: (1, vocab_size)

        probs = F.softmax(logits, dim=-1)             # convert to probabilities

        # Sample next word from the probability distribution
        # torch.multinomial = weighted random pick (not always the top word)
        # This gives variety: model picks "scheduled" 60% of time but "done" 20%, etc.
        next_id = torch.multinomial(probs[0], num_samples=1).item()

        result.append(idx_to_word[next_id])   # add predicted word to sentence
        current_id = next_id                   # next iteration starts from this word

    return " ".join(result)   # join word list back into a sentence string


print("3 generated sentences starting with 'the':")
for trial in range(3):
    sentence = generate_sentence(model, "the", num_words=5,
                                 word_to_idx=word_to_idx, idx_to_word=idx_to_word)
    print(f"  Trial {trial + 1}: '{sentence}'")

# =============================================================================
# PART B: Bigger Dataset -- More Realistic Autocomplete
# =============================================================================

print("\n" + "=" * 60)
print("PART B: Larger Dataset -- Richer Autocomplete Suggestions")
print("=" * 60)

print("""
Part A: 6 sentences, 9 unique words, very repetitive.
Part B: 20 sentences, richer vocabulary, more realistic patterns.

The architecture does NOT change. Only the training data grows.
More data -> model sees more word patterns -> better predictions.
""")

# Larger set of short work-related phrases
sentences_b = [
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

# --- Build vocabulary for Part B ---
all_words_b   = [w for s in sentences_b for w in s.split()]
vocab_b       = sorted(set(all_words_b))
vocab_size_b  = len(vocab_b)
w2i           = {w: i for i, w in enumerate(vocab_b)}   # word -> ID
i2w           = {i: w for i, w in enumerate(vocab_b)}   # ID -> word

# --- Build training pairs ---
xb_list, yb_list = [], []
for sentence in sentences_b:
    words = sentence.split()
    for i in range(len(words) - 1):
        xb_list.append(w2i[words[i]])
        yb_list.append(w2i[words[i + 1]])

x_b = torch.tensor(xb_list, dtype=torch.long)
y_b = torch.tensor(yb_list, dtype=torch.long)

print(f"Training sentences : {len(sentences_b)}")
print(f"Unique words       : {vocab_size_b}")
print(f"Training pairs     : {len(x_b)}")

# --- Train ---
model_b    = BigramModel(vocab_size_b)
opt_b      = torch.optim.AdamW(model_b.parameters(), lr=0.05)

print("\nTraining for 500 steps...")
for step in range(500):
    logits_b, loss_b = model_b(x_b, y_b)
    opt_b.zero_grad()
    loss_b.backward()
    opt_b.step()
    if step % 125 == 0:
        print(f"  Step {step:3d}: loss = {loss_b.item():.4f}")

print(f"\nFinal loss: {loss_b.item():.4f}")

# --- Demo autocomplete ---
print("\nAutocomplete suggestions (top 3):")
for word in ["the", "is", "deadline", "release"]:
    if word not in w2i:
        continue
    suggestions = model_b.predict_top_k(w2i[word], k=3, idx_to_word=i2w)
    parts = "  ".join(f"'{w}' ({p*100:.0f}%)" for w, p in suggestions)
    print(f"  '{word}' -->  {parts}")

# --- Generate sentences ---
print("\n3 generated sentences (Part B, starting with 'the'):")
for trial in range(3):
    sentence = generate_sentence(model_b, "the", num_words=5,
                                 word_to_idx=w2i, idx_to_word=i2w)
    print(f"  Trial {trial + 1}: '{sentence}'")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY: Char-Level vs Word-Level Bigram")
print("=" * 60)
print("""
  example_03_bigram_pytorch.py         THIS FILE (example_03b)
  ----------------------------         -----------------------------------
  Token  = 1 character                 Token  = 1 word
  Vocab  = letters + punctuation       Vocab  = all unique words
  Output = next character              Output = next word (top-k choices)
  Use    = text character generation   Use    = keyboard autocomplete

The model class is IDENTICAL. The training loop is IDENTICAL.
Only the tokenization step is different:
  char-level: text -> list of characters -> IDs
  word-level: text -> text.split()       -> IDs

Key PyTorch tool learned in this file:
  torch.topk(tensor, k)   -- return k largest values + their indices
                             C# analogy: .OrderByDescending(x => x).Take(k)

The bigram model does ONE thing:
  Given token i, output scores for every possible next token.
  Whether 'token' means a letter or a word is YOUR choice.
  The neural network does not care.

Phone keyboard, email autocomplete, search suggestions --
all started as variations of this same idea.
""")

print("=" * 60)
print("Next: example_04_gpt_pytorch.py")
print("      Same foundation + attention layers = GPT")
print("=" * 60)

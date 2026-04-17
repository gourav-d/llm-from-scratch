"""
=============================================================================
PROJECT 1: Shakespeare Text Generator
=============================================================================

GLOSSARY (read this first, before looking at any code)
-------------------------------------------------------
Character-level model : A language model that works one character at a time,
                        not one word at a time. It learns patterns like
                        "after 'q' usually comes 'u'". Simpler than word-level
                        but still surprisingly powerful.

Bigram               : A pair of two things that appear together.
                        In an "bigram model", we look at ONE character and
                        predict the NEXT character.
                        "bi" = two, "gram" = unit of text.

Vocabulary           : The complete set of unique characters (or words) the
                        model knows. For a character-level model this is just
                        all the distinct characters in the training text.

Training             : The process of adjusting a model's internal numbers
                        (weights) so it gets better at its task.
                        C# analogy: like calibrating a PID controller until
                        it hits the target value.

Weight / Parameter   : A number inside the model that gets adjusted during
                        training. The model learns BY changing these numbers.
                        C# analogy: like a field on a class whose value the
                        optimizer keeps tweaking.

Loss                 : A single number that measures how WRONG the model is.
                        Lower loss = better predictions. Training tries to
                        minimize this number.
                        C# analogy: the return value of a fitness function
                        that we are trying to drive to zero.

Gradient             : The direction and amount we should nudge each weight
                        to reduce the loss. Points "uphill" on the loss
                        surface, so we move in the OPPOSITE direction.
                        C# analogy: the slope of a curve at a point; we
                        step down the slope.

Learning Rate        : How big a step we take when following the gradient.
                        Too big -> model overshoots.
                        Too small -> training is very slow.
                        C# analogy: the step size in a binary search.

Epoch                : One full pass through ALL the training data.
                        Like reading the whole book once.

Softmax              : A function that turns a list of raw scores into
                        probabilities that all add up to 1.
                        e.g. [2.0, 1.0, 0.5] -> [0.59, 0.24, 0.16]

Cross-Entropy Loss   : The standard loss function for classification.
                        It measures how surprised the model is by the
                        correct answer. Lower = less surprised = better.

Embedding            : A small list of numbers that represents a token.
                        Instead of "character 42", we use [0.3, -0.1, 0.7].
                        The model learns what numbers work best.
                        C# analogy: a learned feature vector, like a
                        dictionary value that gets updated during training.

Logits               : The raw output scores from the model before softmax.
                        One score per vocabulary character.

One-Hot Encoding     : Representing a character as a list of 0s with a single
                        1 at the position of that character.
                        e.g. vocab=['a','b','c'], 'b' -> [0, 1, 0]
                        C# analogy: a flags enum with exactly one flag set.

ReLU                 : Rectified Linear Unit. A simple activation function:
                        ReLU(x) = x if x > 0, else 0.
                        Adds non-linearity so the network can learn curves,
                        not just straight lines.
                        C# analogy: Math.Max(0, x)

Hidden Layer         : A layer of neurons BETWEEN the input and output.
                        It lets the model learn more complex patterns.

PyTorch              : A Python library for deep learning. Handles gradients
                        automatically (autograd). Like ASP.NET Core for
                        neural networks - manages the plumbing for you.

nn.Module            : PyTorch base class for all neural network layers.
                        C# analogy: abstract base class that every custom
                        layer must inherit from.

Optimizer            : The algorithm that updates the weights using gradients.
                        Adam is a popular choice (Adaptive Moment Estimation).
                        C# analogy: a smart step-size controller.

=============================================================================
PROJECT OVERVIEW
=============================================================================

Goal: Train a real (tiny) language model on Shakespeare text, then use it
      to generate new Shakespeare-like text character by character.

We do this in two parts:

  PART A - Pure NumPy bigram model
    * No external ML libraries needed (only numpy)
    * Train a character-level bigram on a Shakespeare snippet
    * Generate 50 characters of "Shakespeare-like" text
    * You will see every single math step

  PART B - Two-layer PyTorch model (upgraded)
    * Requires PyTorch (graceful fallback if not installed)
    * Same Shakespeare snippet + a few more lines
    * Adds a hidden layer so the model can learn richer patterns
    * Trains with real gradient descent (not just counting)
    * Generates 50 characters of output

Training data used in both parts (hard-coded, no files needed):
    "To be, or not to be, that is the question:
     Whether tis nobler in the mind to suffer
     The slings and arrows of outrageous fortune,
     Or to take arms against a sea of troubles"

=============================================================================
"""

# =============================================================================
# IMPORTS  (libraries we need)
# =============================================================================

import numpy as np          # numpy: numerical computing (like System.Math + arrays)
import random               # random: for seeding so results are reproducible

# Try to import PyTorch. If it is not installed we fall back to NumPy only.
# C# analogy: try { Assembly.Load("torch"); } catch (FileNotFoundException) {}
try:
    import torch                        # main PyTorch library
    import torch.nn as nn               # building blocks (layers, loss functions)
    import torch.nn.functional as F     # stateless functions (softmax, relu, etc.)
    TORCH_AVAILABLE = True              # flag: PyTorch IS available
except ImportError:
    TORCH_AVAILABLE = False             # flag: PyTorch is NOT available

# =============================================================================
# SHARED TRAINING TEXT  (hard-coded - no external files needed)
# =============================================================================

# This is our training data. Both Part A and Part B will learn from it.
# We call it a "corpus" (Latin for "body of text").
SHAKESPEARE = (
    "To be, or not to be, that is the question:\n"
    "Whether tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles"
)

# =============================================================================
print("=" * 60)
print("PROJECT 1: Shakespeare Text Generator")
print("=" * 60)
print()
print("Training corpus:")
print("-" * 40)
print(SHAKESPEARE)           # show the student what we are training on
print("-" * 40)
print(f"Total characters in corpus: {len(SHAKESPEARE)}")
print()

# =============================================================================
# =============================================================================
#  PART A: SIMPLEST VERSION - BIGRAM MODEL (NumPy only)
# =============================================================================
# =============================================================================

print("=" * 60)
print("PART A: Bigram Model (NumPy only, no PyTorch)")
print("=" * 60)
print()

# -----------------------------------------------------------------------------
# What a bigram model does:
#   Step 1: Count how often each character follows each other character.
#   Step 2: Turn those counts into probabilities.
#   Step 3: To generate text: start with one character, look up the
#           probabilities for the next character, pick one at random,
#           repeat.
#
# C# analogy: it is like a Dictionary<char, Dictionary<char, double>>
#             where the inner value is the probability.
# -----------------------------------------------------------------------------

print("STEP A1: Build vocabulary (list of unique characters)")
print("-" * 40)

# sorted() -> returns a sorted list
# set()    -> removes duplicates (like HashSet<char> in C#)
# Together: get every unique character in alphabetical order
chars_a = sorted(set(SHAKESPEARE))     # e.g. [' ', '\n', ',', ':', 'T', 'W', ...]

vocab_size_a = len(chars_a)            # how many unique characters we have

# char_to_idx: maps each character to a unique integer index
# C# analogy: Dictionary<char, int>
# enumerate(chars_a) -> yields (0,'a'), (1,'b'), ... so we swap them
char_to_idx_a = {ch: i for i, ch in enumerate(chars_a)}

# idx_to_char: the reverse mapping, integer back to character
# C# analogy: Dictionary<int, char>
idx_to_char_a = {i: ch for i, ch in enumerate(chars_a)}

print(f"Unique characters found : {vocab_size_a}")
print(f"Characters              : {repr(''.join(chars_a))}")
print()

# -----------------------------------------------------------------------------
print("STEP A2: Encode the text as a list of integers")
print("-" * 40)
# -----------------------------------------------------------------------------

# List comprehension: for each character in SHAKESPEARE, look up its index.
# C# analogy: SHAKESPEARE.Select(ch => char_to_idx_a[ch]).ToList()
encoded_a = [char_to_idx_a[ch] for ch in SHAKESPEARE]

print(f"First 20 chars of text    : {repr(SHAKESPEARE[:20])}")
print(f"First 20 encoded integers : {encoded_a[:20]}")
print()

# -----------------------------------------------------------------------------
print("STEP A3: Count bigram frequencies")
print("-" * 40)
# -----------------------------------------------------------------------------

# counts_a[i][j] = "how many times did character j appear right after character i?"
# This is the CORE of the bigram model.
# Shape: (vocab_size x vocab_size), all zeros to start.
# C# analogy: float[,] counts_a = new float[vocab_size_a, vocab_size_a];
counts_a = np.zeros((vocab_size_a, vocab_size_a), dtype=np.float64)

# Walk through every consecutive pair in the encoded text
for pos in range(len(encoded_a) - 1):   # stop one before the end (no "next" at the last char)
    current = encoded_a[pos]             # index of the current character
    nxt     = encoded_a[pos + 1]         # index of the character that follows
    counts_a[current][nxt] += 1          # increment that cell in the table

print(f"Count table shape: {counts_a.shape}  (rows=current char, cols=next char)")
print(f"Total pairs counted: {int(counts_a.sum())}")
print()

# -----------------------------------------------------------------------------
print("STEP A4: Convert counts to probabilities")
print("-" * 40)
# -----------------------------------------------------------------------------

# For each row (current character), divide by the row total.
# We add 1 to every cell first ("add-1 smoothing" / "Laplace smoothing").
# WHY? If a character never appeared in training, its count is 0, which would
# give probability 0 and cause division-by-zero or log(0) crashes.
# Adding 1 pretends we saw every pair at least once.
smoothed = counts_a + 1.0               # add 1 to every cell

# sum each ROW (axis=1); keepdims keeps shape (vocab x 1) so division broadcasts
row_totals = smoothed.sum(axis=1, keepdims=True)

# Divide each cell by its row total -> probabilities
probs_a = smoothed / row_totals         # each row now adds up to 1.0

print("Each row of probs_a is a probability distribution over the next character.")
print(f"Row sum check (first row): {probs_a[0].sum():.6f}  (should be 1.0)")
print()

# -----------------------------------------------------------------------------
print("STEP A5: Generate 50 characters of Shakespeare-like text")
print("-" * 40)
# -----------------------------------------------------------------------------

# Seed the random number generator so results are reproducible.
# C# analogy: new Random(seed)
np.random.seed(42)

def generate_bigram_a(probs, idx_to_char, char_to_idx, start_char, num_chars):
    """
    Generate text using the bigram probability table.

    probs       : 2D numpy array, shape (vocab_size, vocab_size)
    idx_to_char : dict mapping index -> character
    char_to_idx : dict mapping character -> index
    start_char  : the character we begin generating from
    num_chars   : total number of characters to output (including start_char)
    """
    # Look up the starting character's index
    current_idx = char_to_idx[start_char]   # e.g. 'T' -> some integer

    result = [start_char]                   # list to collect generated characters

    for _ in range(num_chars - 1):          # we already have 1 char, so loop N-1 times
        row = probs[current_idx]            # get probabilities for this character

        # np.random.choice: pick one index at random, weighted by probabilities
        # p=row means more probable characters are picked more often
        # C# analogy: like a weighted random picker from a List<(char, double weight)>
        next_idx = np.random.choice(len(row), p=row)

        next_char = idx_to_char[next_idx]   # convert index back to character

        result.append(next_char)            # add to output list

        current_idx = next_idx              # advance: this char is now "current"

    return ''.join(result)                  # join list of chars into one string

# Generate text starting with 'T' (first letter of our corpus)
generated_a = generate_bigram_a(
    probs_a,            # the probability table we built
    idx_to_char_a,      # index -> char mapping
    char_to_idx_a,      # char -> index mapping
    start_char='T',     # begin with 'T' (like "To be...")
    num_chars=50        # generate 50 characters total
)

print("Generated text (50 characters):")
print("  " + generated_a)
print()
print("Note: the output looks scrambled - that is EXPECTED.")
print("A bigram model only looks at ONE previous character.")
print("It learned letter frequencies but not real words.")
print()

# Show a quick summary of what the model learned
print("Top 5 most likely characters to follow 'T':")
t_idx = char_to_idx_a['T']              # row index for 'T'
t_row = probs_a[t_idx]                  # probability distribution for 'T'
top5_indices = np.argsort(t_row)[::-1][:5]   # indices of top 5 probabilities (descending)
for rank, idx in enumerate(top5_indices):    # loop with a counter (like for (int i=0; ...))
    ch = idx_to_char_a[idx]             # character at this index
    prob = t_row[idx]                   # its probability
    print(f"  Rank {rank+1}: '{repr(ch)}' -> probability {prob:.3f}")
print()

print("PART A complete!")
print()

# =============================================================================
# =============================================================================
#  PART B: UPGRADED VERSION - 2-LAYER MODEL WITH PYTORCH
# =============================================================================
# =============================================================================

print("=" * 60)
print("PART B: 2-Layer Character Model (PyTorch)")
print("=" * 60)
print()

# Check if PyTorch is available. If not, show a friendly message.
if not TORCH_AVAILABLE:
    print("PyTorch is NOT installed on this machine.")
    print("Part B requires PyTorch. To install it, open a terminal and run:")
    print()
    print("  pip install torch")
    print()
    print("After installing, re-run this file to see Part B in action.")
    print()
    print("WHAT Part B would do (even without PyTorch):")
    print("  1. Build a vocabulary from a slightly longer Shakespeare snippet.")
    print("  2. Represent each character as a learned embedding vector.")
    print("  3. Pass that through a hidden layer (ReLU activation).")
    print("  4. Produce probabilities over the vocabulary (softmax).")
    print("  5. Train using gradient descent for many epochs.")
    print("  6. Generate 50 characters of output.")
    print()
    print("=" * 60)
    print("PROJECT 1 complete (Part B skipped - PyTorch not installed)")
    print("=" * 60)

else:
    # =========================================================================
    # PyTorch IS available. Run the full Part B.
    # =========================================================================

    print("PyTorch is available! Running 2-layer model.")
    print()

    # -------------------------------------------------------------------------
    # PART B TRAINING TEXT: same Shakespeare snippet as Part A.
    # Using the same text keeps the comparison fair.
    # -------------------------------------------------------------------------

    SHAKESPEARE_B = SHAKESPEARE     # re-use the same hard-coded text from above

    print("Training text (same Shakespeare snippet):")
    print("-" * 40)
    print(SHAKESPEARE_B)
    print("-" * 40)
    print(f"Total characters: {len(SHAKESPEARE_B)}")
    print()

    # -------------------------------------------------------------------------
    print("STEP B1: Build vocabulary")
    print("-" * 40)
    # -------------------------------------------------------------------------

    # Same approach as Part A: find unique chars, build index mappings
    chars_b       = sorted(set(SHAKESPEARE_B))              # unique chars sorted
    vocab_size_b  = len(chars_b)                            # number of unique chars
    char_to_idx_b = {ch: i for i, ch in enumerate(chars_b)} # char -> int
    idx_to_char_b = {i: ch for i, ch in enumerate(chars_b)} # int -> char

    print(f"Vocabulary size: {vocab_size_b} unique characters")
    print()

    # -------------------------------------------------------------------------
    print("STEP B2: Encode the full training text as integers")
    print("-" * 40)
    # -------------------------------------------------------------------------

    # Convert every character to its integer index
    # C# analogy: SHAKESPEARE_B.Select(c => char_to_idx_b[c]).ToArray()
    encoded_b = [char_to_idx_b[ch] for ch in SHAKESPEARE_B]

    # Convert the list to a PyTorch tensor (the PyTorch equivalent of a numpy array)
    # dtype=torch.long -> 64-bit integer (required for embedding lookup indices)
    # C# analogy: long[] tensor = encoded_b.ToArray()  (but with GPU support)
    data_tensor = torch.tensor(encoded_b, dtype=torch.long)

    print(f"Encoded tensor shape: {data_tensor.shape}")
    print(f"First 10 values     : {data_tensor[:10].tolist()}")
    print()

    # -------------------------------------------------------------------------
    print("STEP B3: Create training pairs (context -> next char)")
    print("-" * 40)
    # -------------------------------------------------------------------------

    # CONTEXT_SIZE: how many previous characters the model looks at.
    # Part A looked at 1 (bigram).
    # Here we look at 4 (a small improvement, but still simple).
    CONTEXT_SIZE = 4    # look at the last 4 characters to predict the next one

    # Build input (X) and target (Y) tensors from the data.
    #
    # Example with CONTEXT_SIZE=2 and text "hello":
    #   X (input context)   Y (next char)
    #   [h, e]           -> l
    #   [e, l]           -> l
    #   [l, l]           -> o
    #
    X_list = []     # will hold the context windows (each is CONTEXT_SIZE integers)
    Y_list = []     # will hold the corresponding next characters

    for i in range(len(encoded_b) - CONTEXT_SIZE):     # slide a window across the text
        context = encoded_b[i : i + CONTEXT_SIZE]       # CONTEXT_SIZE chars (input)
        target  = encoded_b[i + CONTEXT_SIZE]           # the char right after (label)
        X_list.append(context)                          # add to input list
        Y_list.append(target)                           # add to target list

    # Convert Python lists to PyTorch tensors
    # X shape: (num_examples, CONTEXT_SIZE)  - each row is one context window
    # Y shape: (num_examples,)               - each value is the next char index
    X = torch.tensor(X_list, dtype=torch.long)   # input contexts
    Y = torch.tensor(Y_list, dtype=torch.long)   # target next characters

    print(f"Context size (chars looked back): {CONTEXT_SIZE}")
    print(f"Number of training examples     : {X.shape[0]}")
    print(f"X shape: {X.shape}   (rows=examples, cols=context chars)")
    print(f"Y shape: {Y.shape}   (one target per example)")
    print()

    # -------------------------------------------------------------------------
    print("STEP B4: Define the 2-layer neural network model")
    print("-" * 40)
    # -------------------------------------------------------------------------

    # Architecture overview:
    #
    #   INPUT: a window of CONTEXT_SIZE character indices
    #          e.g. [12, 0, 17, 24]   (4 integers)
    #          |
    #          v
    #   LAYER 0 (Embedding): each integer index -> a dense vector of EMBED_DIM numbers
    #          e.g. 12 -> [0.3, -0.1, 0.7, 0.2]  (EMBED_DIM=8 numbers)
    #          After embedding all 4 chars: we have CONTEXT_SIZE x EMBED_DIM numbers
    #          We FLATTEN these into one long vector: CONTEXT_SIZE * EMBED_DIM numbers
    #          |
    #          v
    #   LAYER 1 (Hidden, Linear + ReLU):
    #          input:  (CONTEXT_SIZE * EMBED_DIM) numbers
    #          output: HIDDEN_DIM numbers (e.g. 64)
    #          ReLU clips negatives to 0 (adds non-linearity)
    #          |
    #          v
    #   LAYER 2 (Output, Linear):
    #          input:  HIDDEN_DIM numbers (e.g. 64)
    #          output: vocab_size numbers (raw scores = logits)
    #          |
    #          v
    #   Softmax -> probabilities over all vocabulary characters
    #
    # C# analogy: think of this as a pipeline of transforms, each one
    #             reshaping the data into something more useful.

    EMBED_DIM  = 16    # size of each character's embedding vector
    HIDDEN_DIM = 64    # number of neurons in the hidden layer

    class TwoLayerCharModel(nn.Module):
        """
        A simple 2-layer character-level language model.

        Inherits from nn.Module - the PyTorch base class for all models.
        C# analogy: class TwoLayerCharModel : NeuralNetworkBase { ... }
        """

        def __init__(self, vocab_size, embed_dim, context_size, hidden_dim):
            """
            __init__: constructor. Sets up all the layers.
            C# analogy: public TwoLayerCharModel(int vocabSize, ...) { ... }

            vocab_size   : number of unique characters (size of our vocabulary)
            embed_dim    : how many numbers represent each character embedding
            context_size : how many previous characters we look at
            hidden_dim   : number of neurons in the hidden layer
            """
            super().__init__()          # MUST call parent constructor first
                                        # C# analogy: base() in the constructor chain

            self.vocab_size   = vocab_size    # store for use in forward()
            self.embed_dim    = embed_dim     # store for use in forward()
            self.context_size = context_size  # store for use in forward()

            # LAYER 0: Embedding table
            # Maps integer token IDs -> dense vectors of size embed_dim
            # Think of it as a lookup table with learnable values.
            # C# analogy: float[,] embeddingTable = new float[vocab_size, embed_dim];
            self.embedding = nn.Embedding(vocab_size, embed_dim)

            # LAYER 1: Hidden (fully connected, linear)
            # Input size: context_size * embed_dim  (all embedded chars, flattened)
            # Output size: hidden_dim               (e.g. 64 neurons)
            # C# analogy: a matrix multiply: output = W1 * input + b1
            self.hidden = nn.Linear(context_size * embed_dim, hidden_dim)

            # LAYER 2: Output (fully connected, linear)
            # Input size: hidden_dim     (output of hidden layer)
            # Output size: vocab_size    (one score per character in vocabulary)
            # C# analogy: output = W2 * input + b2
            self.output = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            """
            forward: defines the computation (the "forward pass").
            PyTorch calls this automatically when you do model(input).
            C# analogy: override object Compute(object input) { ... }

            x : input tensor, shape (batch_size, context_size)
                each row is a context window of CONTEXT_SIZE character indices
            """
            # Step 1: Embedding lookup
            # For each character index in x, look up its embedding vector.
            # x shape:    (batch_size, context_size)
            # emb shape:  (batch_size, context_size, embed_dim)
            emb = self.embedding(x)         # look up embeddings for all chars

            # Step 2: Flatten the embeddings into one long vector per example
            # From (batch_size, context_size, embed_dim)
            # To   (batch_size, context_size * embed_dim)
            # C# analogy: Flatten a 2D array into a 1D array
            batch_size = emb.shape[0]       # number of examples in this batch
            flat = emb.view(batch_size, -1) # -1 means "calculate this dimension automatically"

            # Step 3: Hidden layer (Linear) + ReLU activation
            # Linear: applies W1 @ flat + b1  (@ = matrix multiplication)
            # ReLU:   clips negatives to 0 (adds non-linearity so model learns curves)
            # C# analogy: var h = hidden.Compute(flat); h = Math.Max(0, h);
            h = torch.relu(self.hidden(flat))   # shape: (batch_size, hidden_dim)

            # Step 4: Output layer (Linear) -> raw scores (logits)
            # No activation here - the loss function (CrossEntropyLoss) applies
            # softmax internally.
            logits = self.output(h)         # shape: (batch_size, vocab_size)

            return logits                   # return raw scores for each char in vocab

    # Instantiate the model
    model = TwoLayerCharModel(
        vocab_size   = vocab_size_b,    # how many unique characters
        embed_dim    = EMBED_DIM,       # 16 numbers per character embedding
        context_size = CONTEXT_SIZE,    # look at 4 previous chars
        hidden_dim   = HIDDEN_DIM       # 64 neurons in the hidden layer
    )

    # Count total trainable parameters (weights the model learns)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model architecture:")
    print(f"  Embedding layer : {vocab_size_b} chars x {EMBED_DIM} dims")
    print(f"  Hidden layer    : {CONTEXT_SIZE * EMBED_DIM} -> {HIDDEN_DIM} neurons (ReLU)")
    print(f"  Output layer    : {HIDDEN_DIM} -> {vocab_size_b} (logits)")
    print(f"  Total trainable parameters: {total_params:,}")
    print()

    # -------------------------------------------------------------------------
    print("STEP B5: Set up the optimizer and loss function")
    print("-" * 40)
    # -------------------------------------------------------------------------

    # LEARNING RATE: how big a step we take each time we update weights
    LEARNING_RATE = 0.01

    # OPTIMIZER: Adam is a smart version of gradient descent.
    # It adapts the step size automatically per parameter.
    # C# analogy: a feedback controller that tunes itself.
    # model.parameters() gives Adam the list of all weights to update.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # LOSS FUNCTION: CrossEntropyLoss measures how wrong the model is.
    # It combines softmax + negative log likelihood internally.
    # C# analogy: a scoring function we want to minimize.
    loss_fn = nn.CrossEntropyLoss()

    print(f"Optimizer     : Adam  (learning rate = {LEARNING_RATE})")
    print(f"Loss function : CrossEntropyLoss")
    print()

    # -------------------------------------------------------------------------
    print("STEP B6: Train the model")
    print("-" * 40)
    # -------------------------------------------------------------------------

    NUM_EPOCHS  = 500       # number of complete passes through the training data
    PRINT_EVERY = 100       # print a status update every N epochs

    print(f"Training for {NUM_EPOCHS} epochs...")
    print()

    # Set the model to training mode (enables things like Dropout if we had it)
    # C# analogy: model.IsTraining = true;
    model.train()

    for epoch in range(1, NUM_EPOCHS + 1):   # epochs count from 1 to NUM_EPOCHS

        # --- FORWARD PASS ---
        # Feed ALL training examples through the model in one go (full-batch)
        # X shape: (num_examples, CONTEXT_SIZE)
        # logits shape: (num_examples, vocab_size)
        logits = model(X)       # model.__call__(X) which calls model.forward(X)

        # --- COMPUTE LOSS ---
        # CrossEntropyLoss expects:
        #   logits: (num_examples, num_classes)  <- this is our logits
        #   targets: (num_examples,)              <- this is Y
        loss = loss_fn(logits, Y)   # single number measuring how wrong we are

        # --- BACKWARD PASS (compute gradients) ---
        optimizer.zero_grad()   # clear gradients from the previous step
                                # C# analogy: reset an accumulator to zero
        loss.backward()         # compute gradients for all parameters
                                # C# analogy: run the derivative calculation automatically

        # --- UPDATE WEIGHTS ---
        optimizer.step()        # adjust each weight using its gradient
                                # C# analogy: apply the correction to each parameter

        # Print progress every PRINT_EVERY epochs
        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{NUM_EPOCHS}  |  Loss: {loss.item():.4f}")
            # .item() converts a PyTorch scalar tensor to a plain Python float

    print()
    print("Training complete!")
    print()

    # -------------------------------------------------------------------------
    print("STEP B7: Generate 50 characters of Shakespeare-like text")
    print("-" * 40)
    # -------------------------------------------------------------------------

    def generate_pytorch_model(model, char_to_idx, idx_to_char,
                               start_text, context_size, num_chars):
        """
        Generate text one character at a time using the trained PyTorch model.

        model        : the trained TwoLayerCharModel instance
        char_to_idx  : dict, character -> integer index
        idx_to_char  : dict, integer index -> character
        start_text   : seed string (must have at least context_size characters)
        context_size : how many previous chars the model looks at
        num_chars    : number of NEW characters to generate
        """
        model.eval()    # switch to evaluation mode (no gradient tracking)
                        # C# analogy: model.IsTraining = false;

        # Encode the seed text into a list of integer indices
        context = [char_to_idx.get(ch, 0) for ch in start_text[-context_size:]]
        # .get(ch, 0): if character is not in vocabulary, use index 0 (unknown)
        # [-context_size:]: take only the LAST context_size characters of seed

        generated = list(start_text)    # start result with the seed text

        with torch.no_grad():           # tell PyTorch: no need to track gradients
                                        # C# analogy: using (var scope = noGradScope) { }
            for _ in range(num_chars):  # generate one character at a time

                # Convert current context to a tensor
                # unsqueeze(0) adds a batch dimension: (context_size,) -> (1, context_size)
                # C# analogy: wrap a 1D array in a 2D array with one row
                ctx_tensor = torch.tensor([context], dtype=torch.long)

                # Forward pass: get logits for the next character
                logits = model(ctx_tensor)          # shape: (1, vocab_size)

                # Convert logits to probabilities using softmax
                # dim=-1 means apply softmax along the last dimension (vocab_size)
                probs = torch.softmax(logits, dim=-1)   # shape: (1, vocab_size)

                # Sample from the probability distribution
                # torch.multinomial: pick an index weighted by probabilities
                # C# analogy: weighted random choice from an array
                next_idx = torch.multinomial(probs, num_samples=1).item()
                # .item() converts the 1-element tensor to a plain Python int

                next_char = idx_to_char[next_idx]   # convert index -> character

                generated.append(next_char)         # add to our result

                # Slide the context window forward: drop oldest, add new char
                context = context[1:] + [next_idx]
                # context[1:]      : remove the first (oldest) character
                # + [next_idx]     : append the newly generated character

        return ''.join(generated)       # join list of chars into a single string

    # Seed text: use the first CONTEXT_SIZE characters of the corpus
    seed_text = SHAKESPEARE_B[:CONTEXT_SIZE]    # e.g. "To b"

    print(f"Seed text: '{seed_text}'")
    print()

    # Set random seeds for reproducibility
    torch.manual_seed(42)       # PyTorch random seed (C# analogy: new Random(42))
    random.seed(42)             # Python built-in random seed

    generated_b = generate_pytorch_model(
        model         = model,
        char_to_idx   = char_to_idx_b,
        idx_to_char   = idx_to_char_b,
        start_text    = seed_text,
        context_size  = CONTEXT_SIZE,
        num_chars     = 50          # generate 50 NEW characters
    )

    print("Generated text (seed + 50 new characters):")
    print("  " + generated_b)
    print()

    # -------------------------------------------------------------------------
    # Show how loss changed from start to end of training
    # -------------------------------------------------------------------------
    print("What does lower loss mean?")
    print("  The model started by guessing randomly (high loss).")
    print("  After training, it has learned which characters tend")
    print("  to follow which others in Shakespeare's writing (lower loss).")
    print("  The generated text still looks messy because the model is tiny")
    print("  and the training set is only a few lines long.")
    print("  Real GPT is trained on billions of characters for days!")
    print()

    print("=" * 60)
    print("PART B complete!")
    print("=" * 60)
    print()

# =============================================================================
# FINAL SUMMARY: Part A vs Part B comparison
# =============================================================================

print("=" * 60)
print("SUMMARY: Part A vs Part B")
print("=" * 60)
print()
print("  Part A (Bigram / NumPy):")
print("    - No external ML library (just numpy)")
print("    - Looks at: 1 previous character")
print("    - Model: a probability table (manual counting)")
print("    - Training: count + divide (no gradient descent)")
print("    - Generates: mostly random-looking output")
print()
print("  Part B (2-layer / PyTorch):")
print("    - Uses PyTorch for automatic gradient computation")
print("    - Looks at: 4 previous characters (via embedding + hidden layer)")
print("    - Model: learned weights (not just a table)")
print("    - Training: gradient descent over 500 epochs")
print("    - Generates: slightly more structured output")
print()
print("  Real GPT:")
print("    - Looks at: thousands of previous tokens")
print("    - Uses self-attention (Transformers) instead of a fixed window")
print("    - Trained on billions of characters for days on many GPUs")
print("    - Generates: coherent, high-quality text")
print()
print("  The core IDEA is the same at every scale:")
print("    predict the next character -> measure error -> adjust weights")
print()
print("=" * 60)
print("Project 1 complete! Next: project_02 (coming soon).")
print("=" * 60)

# =============================================================================
# =============================================================================
#  QUIZ QUESTIONS
#  (Try to answer before reading the answers below.)
# =============================================================================
# =============================================================================

# -----------------------------------------------------------------------
# QUESTION 1 (Multiple choice):
#
#   In Part A (the bigram model), how does the model decide which
#   character to generate next?
#
#   A) It runs backpropagation and updates weights.
#   B) It looks at a table of probabilities built from character counts.
#   C) It uses the Transformer attention mechanism.
#   D) It picks the character with the highest embedding value.
#
# ANSWER: B
#   The bigram model does NOT use gradient descent at all. It simply
#   counts how often each character follows each other character, then
#   divides by the total to get probabilities. Generation means looking
#   up the row for the current character and sampling from it.
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# QUESTION 2 (Short answer):
#
#   What is "add-1 smoothing" (also called Laplace smoothing) and why
#   do we use it in Part A before converting counts to probabilities?
#
# ANSWER:
#   Add-1 smoothing means we add 1 to every cell in the count table
#   BEFORE dividing to compute probabilities. We do this because some
#   character pairs may never appear in the training text, giving them
#   a count of 0. If we computed probability = 0 / total = 0, and the
#   model tried to sample from a row containing zeros, or if we tried
#   to compute log(0) for a loss, we would get a crash (division by
#   zero or negative infinity). Adding 1 pretends every pair was seen
#   at least once, which avoids zero probabilities entirely.
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# QUESTION 3 (Concept):
#
#   Part A uses "counting" to train; Part B uses "gradient descent".
#   Explain in simple terms what gradient descent does differently,
#   and why it matters for scaling to larger models.
#
# ANSWER:
#   Counting (Part A) works by tallying up which character follows
#   which, then converting to probabilities. It is fast and simple,
#   but it ONLY works for lookup-table style models. You cannot
#   "count" your way to training an embedding or a hidden layer.
#
#   Gradient descent (Part B) works by:
#     1. Running the model forward to get a prediction.
#     2. Measuring the error (loss) between prediction and correct answer.
#     3. Computing the gradient: for each weight, how much does the loss
#        change if we nudge that weight? (This is what loss.backward() does.)
#     4. Moving each weight slightly in the direction that REDUCES the loss
#        (that is what optimizer.step() does).
#     5. Repeating many times (epochs) until the loss is small.
#
#   Gradient descent scales to models with billions of parameters because
#   every weight gets an independent gradient signal telling it exactly
#   which direction to move. You could never "count" your way through
#   billions of embedding dimensions.
# -----------------------------------------------------------------------

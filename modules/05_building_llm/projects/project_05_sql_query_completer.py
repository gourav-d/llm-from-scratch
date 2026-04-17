"""
=============================================================================
PROJECT 5: SQL Query Completer - Auto-finish SQL as You Type
=============================================================================

GLOSSARY (read this first, before looking at any code)
-------------------------------------------------------
Prefix          : The text the user has ALREADY typed.
                  If you typed "SELECT", that is the prefix.
                  The completer guesses what should come AFTER it.
                  C# analogy: like string.StartsWith() -- you match the
                              beginning of a candidate string.

Completion      : The suggested full query that the model fills in for you.
                  e.g. prefix "SELECT" -> completion "SELECT * FROM users WHERE id = 1"
                  This is exactly what VS Code IntelliSense does for C# code.

Template        : A hard-coded example query we store up front.
                  Part A uses templates directly; Part B learns patterns from them.
                  C# analogy: like a List<string> of known code snippets.

Corpus          : The full collection of text the model is trained on.
                  Here our corpus is the 20 SQL templates.
                  In real LLMs the corpus is terabytes of internet text.

Character-level : Processing text one CHARACTER at a time instead of one word.
                  Our Part B model sees 'S','E','L','E','C','T' individually.
                  Advantage: works for any vocabulary including SQL keywords.
                  C# analogy: iterating over string as char[] instead of string[].

Vocabulary      : The complete set of UNIQUE characters the model knows.
                  Built by scanning every character in the corpus.
                  C# analogy: like HashSet<char> built from all training text.

Index / Token ID: An integer assigned to each unique character.
                  The model works with numbers, not characters.
                  C# analogy: like Dictionary<char, int> mapping char -> int.

Embedding       : A row of numbers representing one character (or word).
                  The model LEARNS good numbers through training.
                  e.g. character 'S' might become [0.3, -0.1, 0.8, ...]
                  C# analogy: like float[] that IS the character in math-space.

Weight Matrix   : A 2D array of numbers the model learns during training.
                  Also called a "parameter matrix" or "layer".
                  C# analogy: like float[,] or a 2D jagged array float[][].

Forward Pass    : Running input data THROUGH the model to get a prediction.
                  Goes input -> hidden layer -> output.
                  C# analogy: calling a method that returns a result.

Logits          : Raw output scores from the model for every possible character.
                  NOT probabilities yet. Any positive or negative number.
                  Higher logit = model thinks that character is more likely.

Softmax         : Converts logits into probabilities that sum to exactly 1.0.
                  Formula: prob[i] = exp(logit[i]) / sum(exp(logit[j]) for all j)
                  C# analogy: normalizing a score array so values add up to 1.

Loss            : A number measuring how WRONG the model is right now.
                  We want to minimize this during training.
                  C# analogy: like a penalty score in a game -- lower is better.

Cross-Entropy   : The specific loss formula used for character prediction.
                  Loss = -log(probability assigned to the CORRECT character).
                  If model gives correct char 90% prob -> small loss.
                  If model gives correct char 1% prob  -> huge loss.

Backpropagation : Algorithm that figures out which weights caused the error
                  and by HOW MUCH. Then we nudge each weight to reduce loss.
                  Short form: "backprop". This is how neural networks learn.
                  C# analogy: like an automatic "undo" that tells each variable
                              how much it hurt the final answer.

Gradient        : The direction and size of the nudge for each weight.
                  A positive gradient means "increase this weight raises loss --
                  so DECREASE it to reduce loss."

Learning Rate   : How big each weight update step is.
                  Too large -> overshoots, model unstable.
                  Too small -> trains forever, may not converge.
                  C# analogy: like a step size in a binary search.

Epoch           : One complete pass through ALL training data.
                  After 200 epochs the model has seen every character 200 times.

Context Window  : How many previous characters the model can "see" to predict
                  the next one. Here we use 8 characters of context.
                  Real GPT-4 uses 128,000 tokens of context.

Greedy Decoding : Always pick the character with the HIGHEST probability.
                  No randomness. Fast and deterministic.
                  C# analogy: always take the if-branch with highest score.

Temperature     : A number that controls how "creative" the output is.
                  Divide logits by temperature BEFORE softmax.
                  temperature = 1.0 -> normal probabilities
                  temperature < 1.0 -> more focused (boring but safe)
                  temperature > 1.0 -> more random (creative but risky)

nanoGPT         : A tiny, educational version of GPT (Generative Pre-trained
                  Transformer). Created by Andrej Karpathy. Uses PyTorch.
                  Our Part B is inspired by nanoGPT but even smaller.

PyTorch         : The most popular deep learning library in Python.
                  Provides tensors (like numpy arrays) with automatic gradient
                  computation. "Autograd" does backprop for you automatically.
                  C# analogy: like a math library where every operation also
                              records HOW to undo itself, so training is automatic.

Tensor          : A PyTorch array. A 1D tensor is like a list, 2D is a matrix,
                  3D or higher is like a multi-dimensional array.
                  C# analogy: like Array / jagged array, but GPU-aware.

nn.Module       : The base class for every neural network in PyTorch.
                  You inherit from it and define forward() to describe your model.
                  C# analogy: like implementing an interface with a Compute() method.

=============================================================================
PROJECT OVERVIEW
=============================================================================

What is a SQL Query Completer?
  When you use an IDE like VS Code, DataGrip, or SQL Server Management Studio,
  you start typing a query and the tool SUGGESTS how to finish it.
  That suggestion could come from:
    (a) Simple prefix matching on known patterns -- fast, no ML (Part A)
    (b) A trained language model that learned SQL grammar -- smarter (Part B)

This project builds BOTH versions so you can compare them.

Why does this matter for LLMs?
  GitHub Copilot, ChatGPT Code Interpreter, and Cursor all use LLM-based
  completion under the hood. You are building a toy version of that.

C# Connection:
  As a .NET developer you already know SQL and C#. This project bridges:
    SQL keywords    -> training data / vocabulary
    C# Dictionary   -> Python dict / defaultdict
    C# List<string> -> Python list
    C# string.StartsWith() -> Python str.startswith()
    C# LINQ         -> Python list comprehensions and filter()

PART A: No ML. Prefix-match on 20 hard-coded SQL templates.
PART B: Tiny char-level nanoGPT trained on those same 20 templates.
        Try/except protects against missing PyTorch: falls back to bigram.

=============================================================================
"""

# =============================================================================
# IMPORTS
# Standard Python modules only in Part A.
# PyTorch imported inside a try/except in Part B (graceful fallback).
# C# analogy: like conditional "using" statements -- Python allows this at
#             runtime, C# does not.
# =============================================================================

import random          # built-in: random number generation
import math            # built-in: exp(), log(), sqrt() -- like Math class in C#
import numpy as np     # numerical arrays, used in the numpy bigram fallback

# Set random seeds so the output is repeatable every time you run this file.
# C# analogy: like new Random(42) -- same seed always gives the same sequence.
random.seed(42)
np.random.seed(42)


# =============================================================================
# HELPER FUNCTIONS -- PRINT FORMATTING
# We only use ASCII characters (cp1252-safe).
# NO Unicode, NO arrows, NO box-drawing characters.
# =============================================================================

def section(title):
    """Print a large section header using = signs."""
    print()                          # blank line before header
    print("=" * 60)                  # 60 = signs as top border
    print(title)                     # the title text
    print("=" * 60)                  # 60 = signs as bottom border

def subsection(title):
    """Print a smaller sub-header using - signs."""
    print()                          # blank line before
    print(title)                     # the title text
    print("-" * len(title))          # dashes the same length as the title


# =============================================================================
# TRAINING DATA
# 20 SQL query templates, hard-coded as plain strings.
# Both Part A and Part B use this exact list.
# C# analogy: like a static readonly string[] field on a helper class.
# =============================================================================

SQL_TEMPLATES = [
    "SELECT * FROM users WHERE id = 1",
    "SELECT name FROM customers WHERE age > 18",
    "SELECT * FROM orders WHERE status = active",
    "SELECT COUNT(*) FROM products WHERE price < 100",
    "SELECT id, name FROM employees WHERE department = sales",
    "INSERT INTO users VALUES (1, John, admin)",
    "INSERT INTO orders VALUES (101, customer1, pending)",
    "UPDATE users SET status = active WHERE id = 1",
    "UPDATE products SET price = 50 WHERE category = books",
    "DELETE FROM sessions WHERE expired = true",
    "SELECT * FROM users ORDER BY name ASC",
    "SELECT * FROM orders ORDER BY date DESC",
    "WHERE status = active AND created > 2024",
    "WHERE price BETWEEN 10 AND 100",
    "WHERE name LIKE John AND role = admin",
    "SELECT AVG(price) FROM products GROUP BY category",
    "SELECT SUM(amount) FROM orders GROUP BY customer",
    "JOIN customers ON orders.customer_id = customers.id",
    "LEFT JOIN products ON orders.product_id = products.id",
    "SELECT * FROM users LIMIT 10 OFFSET 20",
]

# Total count -- useful for output messages
NUM_TEMPLATES = len(SQL_TEMPLATES)      # should be 20


# =============================================================================
# =============================================================================
#
#  PART A: PREFIX-MATCHING COMPLETER (NO MACHINE LEARNING)
#
# =============================================================================
# =============================================================================

section("PART A: Prefix-Matching Completer (No Machine Learning)")

print("""
HOW IT WORKS (plain English, no code yet):
  1. Store all 20 SQL templates in a list.
  2. When the user types a prefix (e.g. "SELECT"), loop through the list.
  3. Any template whose beginning matches the prefix is a candidate.
  4. Return up to N candidates, ranked by length (shortest first).

This is exactly how IDE "quick fix" suggestion lists work:
  - Visual Studio suggests method names that START WITH what you typed.
  - SQL Server Management Studio suggests tables that match your prefix.

C# analogy:
  var templates = new List<string>{ ... };
  var matches = templates.Where(t => t.StartsWith(prefix)).ToList();
  // That is Python list comprehension: [t for t in templates if t.startswith(prefix)]
""")

# =============================================================================
# STEP A1: DEFINE THE COMPLETER FUNCTION
# =============================================================================

subsection("Step A1: Prefix-Matching Function")

def get_completions(prefix, templates, max_results=5):
    """
    Return up to max_results templates that start with the given prefix.

    Parameters:
      prefix      -- the text the user has typed so far (e.g. "SELECT")
      templates   -- the list of all known SQL query strings
      max_results -- how many suggestions to return at most

    Returns:
      A list of matching template strings, sorted shortest first.

    C# analogy:
      List<string> GetCompletions(string prefix, List<string> templates, int maxResults)
      {
          return templates
              .Where(t => t.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
              .OrderBy(t => t.Length)
              .Take(maxResults)
              .ToList();
      }
    """
    # Convert prefix to uppercase for case-insensitive matching.
    # C# analogy: prefix.ToUpper()
    prefix_upper = prefix.upper()

    # List comprehension: build a new list containing only matching templates.
    # [t for t in templates if ...] is Python's version of LINQ .Where().
    # t.upper() converts the template to uppercase before comparing.
    # .startswith() is exactly like C# string.StartsWith().
    matches = [t for t in templates if t.upper().startswith(prefix_upper)]

    # Sort matches by length: shorter completions appear first.
    # key=len tells sorted() to use the length of each string as the sort key.
    # C# analogy: .OrderBy(t => t.Length)
    matches_sorted = sorted(matches, key=len)

    # Return only the first max_results items.
    # Python slice [start:stop] -- [:max_results] means from index 0 to max_results.
    # C# analogy: .Take(maxResults).ToList()
    return matches_sorted[:max_results]


# =============================================================================
# STEP A2: RUN 5 DEMO COMPLETIONS
# =============================================================================

subsection("Step A2: Demo Completions")

# Define 5 demo prefixes to test.
# These simulate what a developer might type in an IDE.
demo_prefixes = [
    "SELECT",           # most common SQL keyword
    "WHERE",            # filtering clause
    "INSERT",           # insert statements
    "UPDATE",           # update statements
    "LEFT JOIN",        # join clause
]

# Loop through each demo prefix and show its completions.
# C# analogy: foreach (var prefix in demoPrefixes) { ... }
for prefix in demo_prefixes:                            # iterate over each prefix
    print()                                             # blank line for spacing
    print("You typed: [" + prefix + "]")                # show what the user typed
    print("Suggestions:")                               # header for suggestions

    # Call our completer function to get matching templates.
    results = get_completions(prefix, SQL_TEMPLATES, max_results=5)

    if results:                                         # if the list is not empty
        for i, suggestion in enumerate(results):        # enumerate gives index + value
            # Print each suggestion with a number prefix.
            # str(i+1) converts integer to string (C# analogy: (i+1).ToString())
            print("  " + str(i + 1) + ". " + suggestion)
    else:                                               # if no matches found
        print("  (no matches found for this prefix)")

print()
print("=" * 60)
print("Part A complete. No ML used. Pure string matching.")
print("This is how autocomplete worked before neural networks.")
print("=" * 60)


# =============================================================================
# =============================================================================
#
#  PART B: CHAR-LEVEL nanoGPT TRAINED ON SQL TEMPLATES
#
# =============================================================================
# =============================================================================

section("PART B: Char-Level nanoGPT Trained on SQL Templates")

print("""
HOW IT WORKS (plain English first):
  Instead of simple prefix-matching, we train a tiny neural network.
  The network reads the 20 SQL templates CHARACTER BY CHARACTER.
  It learns: "after 'SELEC', the next character is very likely 'T'".
             "after 'WHER', the next character is very likely 'E'".

  After training, we GENERATE text by:
    1. Feed the prefix characters into the model.
    2. Model outputs probabilities for every possible next character.
    3. Pick a character (greedy OR with temperature for variety).
    4. Append it and repeat until we have generated enough characters.

  Architecture (our tiny model):
    Input   : sequence of character IDs (integers)
    Embed   : map each ID to a small float vector (embedding)
    Hidden  : one linear layer with ReLU activation
    Output  : one linear layer -> logits for every character in vocabulary

  C# analogy:
    class TinyGPT {
        float[,] embedding;   // vocab_size x embed_dim
        float[,] W_hidden;    // (context * embed_dim) x hidden_dim
        float[]  b_hidden;    // hidden_dim
        float[,] W_output;    // hidden_dim x vocab_size
        float[]  b_output;    // vocab_size
    }

STRATEGY COMPARISON:
  Greedy        : always pick the highest-probability character. Deterministic.
  Temperature   : divide logits by T before softmax.
                  T=0.7 -> more focused than greedy (but still random).
""")

# =============================================================================
# STEP B1: BUILD THE CORPUS FROM SQL TEMPLATES
# =============================================================================

subsection("Step B1: Build the Corpus and Vocabulary")

# Join all 20 templates into one long string separated by newline characters.
# "\n".join(list) -- like String.Join("\n", list) in C#.
# Every newline acts as a "sentence boundary" signal for the model.
CORPUS = "\n".join(SQL_TEMPLATES)      # one long string, all templates combined

# Get every UNIQUE character in the corpus.
# set(string) in Python creates a set of unique elements -- like HashSet<char> in C#.
# sorted() converts the set to a sorted list.
VOCAB = sorted(set(CORPUS))           # e.g. [' ', '*', ',', '.', '0', '1', ...]

VOCAB_SIZE = len(VOCAB)               # how many unique characters exist

# Build two lookup dictionaries:
#   char_to_idx : character -> integer index  (like Dictionary<char, int> in C#)
#   idx_to_char : integer index -> character  (like Dictionary<int, char> in C#)
char_to_idx = {ch: i for i, ch in enumerate(VOCAB)}   # dict comprehension
idx_to_char = {i: ch for i, ch in enumerate(VOCAB)}   # reverse lookup

print("Corpus length (chars) : " + str(len(CORPUS)))       # total characters
print("Vocabulary size       : " + str(VOCAB_SIZE))         # unique chars
print("Sample vocabulary     : " + "".join(VOCAB[:20]))     # first 20 chars

# =============================================================================
# STEP B2: ENCODE THE CORPUS AS A LIST OF INTEGERS
# =============================================================================

subsection("Step B2: Encode Corpus to Integer IDs")

# Convert every character in the corpus to its index number.
# List comprehension: [char_to_idx[c] for c in CORPUS]
# C# analogy: corpus.Select(c => charToIdx[c]).ToArray()
encoded_corpus = [char_to_idx[c] for c in CORPUS]     # list of int IDs

print("First 40 character IDs: " + str(encoded_corpus[:40]))


# =============================================================================
# STEP B3: ATTEMPT TO IMPORT PYTORCH
# If PyTorch is installed, use a real neural network (nanoGPT-style).
# If not, fall back to a numpy bigram model.
# C# analogy: try { ... } catch (DllNotFoundException) { ... }
# =============================================================================

subsection("Step B3: Check for PyTorch")

try:
    import torch                                   # main PyTorch package
    import torch.nn as nn                          # neural network module (like System.Math)
    import torch.nn.functional as F                # activation functions, softmax, etc.
    TORCH_AVAILABLE = True                         # flag: we can use real ML
    print("PyTorch found. Version: " + torch.__version__)
    print("Using neural network model (nanoGPT-style).")

except ImportError:                                # PyTorch not installed
    TORCH_AVAILABLE = False                        # flag: fall back to numpy
    print("PyTorch NOT found.")
    print("Falling back to numpy bigram model.")
    print("To install PyTorch: pip install torch")


# =============================================================================
# HELPER: ENCODE / DECODE STRINGS
# These helpers convert between text and integer lists.
# Used by both the PyTorch model and the numpy fallback.
# =============================================================================

def encode(text):
    """
    Convert a string of characters to a list of integer IDs.
    Characters not in the vocabulary are SKIPPED (unknown chars ignored).
    C# analogy: text.Select(c => charToIdx.TryGetValue(c, out int id) ? id : -1)
                     .Where(id => id >= 0).ToList()
    """
    result = []                                    # empty list to collect IDs
    for c in text:                                 # loop over each character
        if c in char_to_idx:                       # only include known characters
            result.append(char_to_idx[c])          # append the integer ID
    return result                                  # return list of IDs

def decode(indices):
    """
    Convert a list of integer IDs back to a human-readable string.
    C# analogy: string.Concat(indices.Select(i => idxToChar[i]))
    """
    # "".join(list) glues characters together with no separator.
    # C# analogy: string.Concat(charList)
    return "".join(idx_to_char[i] for i in indices)     # return decoded string


# =============================================================================
# =============================================================================
#  BRANCH A: PYTORCH NEURAL NETWORK
# =============================================================================
# =============================================================================

if TORCH_AVAILABLE:

    # =========================================================================
    # STEP B4 (PyTorch): DEFINE THE MODEL
    # =========================================================================

    subsection("Step B4 (PyTorch): Define the nanoGPT Model")

    # Hyperparameters -- these are tuning knobs for the model.
    # C# analogy: like static readonly const fields in a Config class.
    CONTEXT_LEN  = 8      # how many previous characters the model sees at once
    EMBED_DIM    = 32     # size of each character embedding vector
    HIDDEN_DIM   = 64     # number of neurons in the hidden layer
    LEARNING_RATE = 0.01  # how big each gradient step is
    NUM_EPOCHS   = 300    # how many times we loop over the full training data
    GENERATE_LEN = 40     # how many characters to generate per completion

    print("Hyperparameters:")
    print("  Context length : " + str(CONTEXT_LEN) + " characters")
    print("  Embed dim      : " + str(EMBED_DIM))
    print("  Hidden dim     : " + str(HIDDEN_DIM))
    print("  Learning rate  : " + str(LEARNING_RATE))
    print("  Epochs         : " + str(NUM_EPOCHS))

    class TinySQLGPT(nn.Module):
        """
        A tiny character-level language model for SQL completion.

        Architecture:
          1. Embedding layer   : each char ID -> a vector of EMBED_DIM floats
          2. Flatten           : concatenate all context embeddings into one vector
          3. Hidden linear     : (CONTEXT_LEN * EMBED_DIM) -> HIDDEN_DIM
          4. ReLU activation   : zero out negative values (adds non-linearity)
          5. Output linear     : HIDDEN_DIM -> VOCAB_SIZE (one score per char)

        C# analogy:
          class TinySQLGPT {
              Embedding embed;       // char ID -> float[EMBED_DIM]
              Linear hiddenLayer;    // flat input -> float[HIDDEN_DIM]
              Linear outputLayer;    // float[HIDDEN_DIM] -> float[VOCAB_SIZE]
              float[] Forward(int[] context) { ... }
          }
        """

        def __init__(self, vocab_size, embed_dim, context_len, hidden_dim):
            """
            Constructor: define all learnable layers.
            C# analogy: public TinySQLGPT() { ... }
            """
            # super().__init__() calls the nn.Module constructor.
            # Required in every PyTorch model.
            # C# analogy: base() call in a derived class constructor.
            super().__init__()

            # Embedding table: maps each character ID to a vector.
            # Shape: (vocab_size, embed_dim)
            # C# analogy: float[vocab_size, embed_dim] -- row = char, col = feature
            self.embedding = nn.Embedding(vocab_size, embed_dim)

            # Hidden linear layer: input is context_len character embeddings
            # concatenated together.
            # Input size  = context_len * embed_dim
            # Output size = hidden_dim
            # C# analogy: float[context_len * embed_dim, hidden_dim] weight matrix
            self.hidden = nn.Linear(context_len * embed_dim, hidden_dim)

            # Output linear layer: projects hidden state to vocabulary scores.
            # Input size  = hidden_dim
            # Output size = vocab_size  (one logit per possible next character)
            self.output_layer = nn.Linear(hidden_dim, vocab_size)

            # Store context_len so forward() can use it.
            self.context_len = context_len

        def forward(self, x):
            """
            Forward pass: given a batch of context windows, predict next char.

            x : tensor of shape (batch_size, context_len) -- integer IDs
            Returns: logits of shape (batch_size, vocab_size)

            C# analogy: float[,] Forward(int[,] x) { ... }
            """
            # Step 1: Embed each character ID.
            # self.embedding(x) looks up each integer in the embedding table.
            # Output shape: (batch_size, context_len, embed_dim)
            embedded = self.embedding(x)

            # Step 2: Flatten the context embeddings into one long vector.
            # view(batch_size, -1) reshapes to (batch_size, context_len * embed_dim).
            # -1 means "figure out this dimension automatically".
            # C# analogy: Array.Reshape() or Buffer.BlockCopy to flatten a 2D array.
            flat = embedded.view(embedded.shape[0], -1)

            # Step 3: Pass through hidden linear layer, then ReLU activation.
            # ReLU replaces negative numbers with 0. Adds non-linearity.
            # C# analogy: Math.Max(0, value) applied element-wise.
            hidden_out = F.relu(self.hidden(flat))

            # Step 4: Project to vocabulary size to get raw scores (logits).
            # Shape: (batch_size, vocab_size)
            logits = self.output_layer(hidden_out)

            return logits                          # return raw un-normalized scores

    # =========================================================================
    # STEP B5 (PyTorch): CREATE TRAINING DATA (context, target) PAIRS
    # =========================================================================

    subsection("Step B5 (PyTorch): Build Training Pairs")

    # Convert encoded corpus to a PyTorch tensor.
    # torch.tensor() wraps a Python list into a PyTorch tensor.
    # dtype=torch.long means 64-bit integers -- required for embedding lookups.
    # C# analogy: int[] corpusTensor = encodedCorpus.ToArray();
    corpus_tensor = torch.tensor(encoded_corpus, dtype=torch.long)

    # Build a list of (context_window, target_character) pairs.
    # context_window : the CONTEXT_LEN characters BEFORE position i
    # target_char    : the character AT position i (what we want to predict)
    contexts = []       # list to collect context windows (each is a list of IDs)
    targets  = []       # list to collect target character IDs

    # Slide a window of length CONTEXT_LEN through the entire encoded corpus.
    # C# analogy: for (int i = CONTEXT_LEN; i < encodedCorpus.Length; i++) { ... }
    for i in range(CONTEXT_LEN, len(encoded_corpus)):       # start after first window
        ctx = encoded_corpus[i - CONTEXT_LEN : i]           # slice: last CONTEXT_LEN chars
        tgt = encoded_corpus[i]                             # the character to predict
        contexts.append(ctx)                                # add context to list
        targets.append(tgt)                                 # add target to list

    # Convert lists to PyTorch tensors.
    # torch.tensor(list_of_lists) creates a 2D tensor automatically.
    X = torch.tensor(contexts, dtype=torch.long)    # shape: (N, CONTEXT_LEN)
    Y = torch.tensor(targets,  dtype=torch.long)    # shape: (N,)

    print("Training samples : " + str(X.shape[0]))          # number of pairs
    print("Context shape    : " + str(tuple(X.shape)))       # (N, CONTEXT_LEN)
    print("Target shape     : " + str(tuple(Y.shape)))       # (N,)

    # =========================================================================
    # STEP B6 (PyTorch): INSTANTIATE MODEL AND OPTIMIZER
    # =========================================================================

    subsection("Step B6 (PyTorch): Create Model and Optimizer")

    # Create an instance of our tiny model.
    # C# analogy: var model = new TinySQLGPT(...);
    model = TinySQLGPT(
        vocab_size   = VOCAB_SIZE,     # how many unique characters
        embed_dim    = EMBED_DIM,      # embedding vector size
        context_len  = CONTEXT_LEN,    # characters of context
        hidden_dim   = HIDDEN_DIM,     # neurons in hidden layer
    )

    # Count total number of learnable parameters (weights + biases).
    # p.numel() returns number of elements in a parameter tensor.
    # sum() adds them all up.
    total_params = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters: " + str(total_params))
    print("(GPT-3 has 175 billion -- we have " + str(total_params) + "!)")

    # Adam optimizer: updates model weights after each batch.
    # Adam stands for Adaptive Moment Estimation.
    # C# analogy: like an automatic "weight tuner" that uses gradient history.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # =========================================================================
    # STEP B7 (PyTorch): TRAIN THE MODEL
    # =========================================================================

    subsection("Step B7 (PyTorch): Training Loop")

    print("Training for " + str(NUM_EPOCHS) + " epochs...")
    print("(Each epoch = one full pass through all training pairs)")
    print()

    for epoch in range(NUM_EPOCHS):                          # loop over epochs

        # Forward pass: run all X through the model to get predictions.
        # logits shape: (N, VOCAB_SIZE) -- one score per char per sample
        logits = model(X)

        # Compute cross-entropy loss.
        # F.cross_entropy(logits, targets) does softmax + negative log likelihood.
        # It tells us: on average, how wrong is the model right now?
        loss = F.cross_entropy(logits, Y)

        # Backward pass: compute gradients (how much each weight caused the loss).
        optimizer.zero_grad()          # clear old gradients (must do before backprop)
        loss.backward()                # compute new gradients via backpropagation
        optimizer.step()               # update weights using the computed gradients

        # Print progress every 50 epochs so we can see training happening.
        # epoch+1 because epoch starts at 0 but we want to display 1-based.
        if (epoch + 1) % 50 == 0:                           # modulo check
            print("  Epoch " + str(epoch + 1).rjust(4) +   # right-justify number
                  " / " + str(NUM_EPOCHS) +
                  "   loss = " + "{:.4f}".format(loss.item()))  # 4 decimal places

    print()
    print("Training complete!")

    # =========================================================================
    # STEP B8 (PyTorch): GENERATION FUNCTIONS
    # =========================================================================

    subsection("Step B8 (PyTorch): Generation Functions")

    def generate_greedy(prefix_text, max_new_chars=GENERATE_LEN):
        """
        Generate characters one at a time, always picking the
        most probable next character (greedy decoding).

        prefix_text    : the SQL prefix the user typed
        max_new_chars  : how many NEW characters to generate

        Returns: the prefix + generated continuation as a string.

        C# analogy: like always taking the branch with the highest score,
                    no randomness involved.
        """
        model.eval()                               # put model in eval mode (no dropout)

        # Encode the prefix into a list of integer IDs.
        ids = encode(prefix_text)                  # list of int IDs

        # Pad or truncate to exactly CONTEXT_LEN characters.
        # If prefix is shorter, pad with 0s on the LEFT.
        # If prefix is longer, keep only the LAST CONTEXT_LEN characters.
        if len(ids) < CONTEXT_LEN:                 # prefix too short
            ids = [0] * (CONTEXT_LEN - len(ids)) + ids    # left-pad with zeros
        else:                                      # prefix too long or exact
            ids = ids[-CONTEXT_LEN:]               # keep last CONTEXT_LEN items

        generated = list(prefix_text)              # start result with the prefix text

        for _ in range(max_new_chars):             # generate one char at a time
            # Build a tensor of shape (1, CONTEXT_LEN) -- batch size 1.
            ctx = torch.tensor([ids], dtype=torch.long)   # wrap in list for batch dim

            with torch.no_grad():                  # no gradient needed during inference
                logits = model(ctx)                # shape: (1, VOCAB_SIZE)

            # Get the index of the highest logit (most probable char).
            # .argmax() returns the position of the maximum value.
            # C# analogy: Array.IndexOf(logits, logits.Max())
            next_id = logits[0].argmax().item()    # .item() converts tensor to Python int

            # Decode the integer back to a character and append.
            next_char = idx_to_char[next_id]       # look up character
            generated.append(next_char)             # add to output

            # Slide the context window: drop oldest char, add the new one.
            ids = ids[1:] + [next_id]              # shift left, append new ID

        return "".join(generated)                  # join char list into a string

    def generate_temperature(prefix_text, temperature=0.7, max_new_chars=GENERATE_LEN):
        """
        Generate characters using temperature sampling.

        Divide logits by temperature before softmax.
        temperature < 1.0 -> more focused (less random)
        temperature > 1.0 -> more creative (more random)
        temperature = 1.0 -> equivalent to raw softmax probabilities

        C# analogy: like a dice roll where you can "bias" the die
                    toward or away from the most likely outcome.
        """
        model.eval()                               # eval mode

        ids = encode(prefix_text)                  # encode prefix

        # Pad or truncate to CONTEXT_LEN.
        if len(ids) < CONTEXT_LEN:
            ids = [0] * (CONTEXT_LEN - len(ids)) + ids
        else:
            ids = ids[-CONTEXT_LEN:]

        generated = list(prefix_text)              # start with prefix

        for _ in range(max_new_chars):
            ctx = torch.tensor([ids], dtype=torch.long)

            with torch.no_grad():
                logits = model(ctx)                # shape: (1, VOCAB_SIZE)

            # TEMPERATURE: divide logits by temperature before softmax.
            # Lower T -> probabilities more "peaked" (confident).
            # Higher T -> probabilities more "flat" (random).
            scaled_logits = logits[0] / temperature    # element-wise division

            # Softmax converts scaled logits to a probability distribution.
            # probabilities will sum to 1.0.
            probs = F.softmax(scaled_logits, dim=-1)   # shape: (VOCAB_SIZE,)

            # Sample one index from the probability distribution.
            # torch.multinomial picks a random index weighted by probabilities.
            # C# analogy: weighted random selection from a list.
            next_id = torch.multinomial(probs, num_samples=1).item()

            next_char = idx_to_char[next_id]
            generated.append(next_char)

            ids = ids[1:] + [next_id]              # slide context window

        return "".join(generated)

    # =========================================================================
    # STEP B9 (PyTorch): GENERATE COMPLETIONS FOR 3 SQL PREFIXES
    # =========================================================================

    subsection("Step B9 (PyTorch): SQL Completions - Greedy vs Temperature=0.7")

    # Three SQL prefixes to test -- chosen to exercise different SQL keywords.
    test_prefixes = ["SELECT", "WHERE", "INSERT"]

    for prefix in test_prefixes:                       # loop over each test prefix
        print()
        print("=" * 60)
        print("PREFIX: [" + prefix + "]")
        print("=" * 60)

        # Greedy completion -- deterministic, always the same output.
        greedy_result = generate_greedy(prefix)
        print("Greedy  : " + greedy_result)

        # Temperature=0.7 -- slightly random, may vary each run.
        temp_result = generate_temperature(prefix, temperature=0.7)
        print("Temp0.7 : " + temp_result)

        print()
        print("NOTE: Greedy always picks the MOST LIKELY char.")
        print("NOTE: Temp=0.7 is more focused than random but not fully greedy.")


# =============================================================================
# =============================================================================
#  BRANCH B: NUMPY BIGRAM FALLBACK (no PyTorch)
# =============================================================================
# =============================================================================

else:       # PyTorch not available -- use numpy bigram instead

    # =========================================================================
    # STEP B4 (Numpy): BUILD A CHARACTER-LEVEL BIGRAM MODEL
    # =========================================================================

    subsection("Step B4 (Numpy): Character Bigram Fallback Model")

    print("""
Numpy Bigram Model (no neural network):
  Count how often character B follows character A in the corpus.
  Store counts in a 2D matrix: bigram_matrix[A][B] = count.
  To predict the next character after A, find the column with the highest count.

  C# analogy:
    int[,] bigramMatrix = new int[vocabSize, vocabSize];
    bigramMatrix[charAIdx, charBIdx]++;  // count each consecutive pair
""")

    # Create a 2D numpy array of shape (VOCAB_SIZE, VOCAB_SIZE) filled with zeros.
    # Row index = current character, Column index = next character.
    # C# analogy: int[,] bigramMatrix = new int[VOCAB_SIZE, VOCAB_SIZE];
    bigram_matrix = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float32)

    # Slide through every consecutive pair in the corpus.
    # C# analogy: for (int i = 0; i < encodedCorpus.Length - 1; i++) { ... }
    for i in range(len(encoded_corpus) - 1):         # stop one before the end
        current_id = encoded_corpus[i]               # current character ID
        next_id    = encoded_corpus[i + 1]           # next character ID
        bigram_matrix[current_id, next_id] += 1      # increment count

    # Convert raw counts to probabilities by dividing each row by its row sum.
    # keepdims=True preserves the 2D shape for broadcasting.
    # Avoid division by zero by clipping the denominator at 1.
    row_sums = bigram_matrix.sum(axis=1, keepdims=True)      # sum each row
    row_sums = np.maximum(row_sums, 1)                        # avoid /0
    bigram_probs = bigram_matrix / row_sums                   # normalize to probs

    print("Bigram matrix shape: " + str(bigram_matrix.shape))
    print("(rows = current char, cols = next char, values = probability)")

    # =========================================================================
    # STEP B5 (Numpy): GENERATION FUNCTIONS
    # =========================================================================

    subsection("Step B5 (Numpy): Generation Functions")

    def generate_greedy_numpy(prefix_text, max_new_chars=40):
        """
        Greedy generation using the numpy bigram model.
        Always pick the next character with the highest probability.
        Uses only the LAST character of context (bigram = 1 char context).
        """
        ids = encode(prefix_text)                    # encode prefix

        if not ids:                                  # if encoding returned nothing
            ids = [0]                                # default to first vocab char

        generated = list(prefix_text)                # start with prefix

        for _ in range(max_new_chars):               # generate one char at a time
            current_id = ids[-1]                     # use the last known character
            probs_row  = bigram_probs[current_id]    # row = probabilities for next char
            next_id    = int(np.argmax(probs_row))   # greedy: pick highest prob
            next_char  = idx_to_char[next_id]        # decode to character
            generated.append(next_char)              # add to result
            ids.append(next_id)                      # extend context

        return "".join(generated)                    # join to string

    def generate_temperature_numpy(prefix_text, temperature=0.7, max_new_chars=40):
        """
        Temperature sampling using the numpy bigram model.
        Raise each probability to the power (1/temperature) then renormalize.
        Lower temperature -> more focused. Higher -> more random.
        """
        ids = encode(prefix_text)

        if not ids:
            ids = [0]

        generated = list(prefix_text)

        for _ in range(max_new_chars):
            current_id = ids[-1]                     # last character
            probs_row  = bigram_probs[current_id].copy()  # copy to avoid mutation

            # Apply temperature: raise probs to power (1/T).
            # Lower T -> sharper distribution (more confident).
            # Higher T -> flatter distribution (more random).
            # C# analogy: probs[i] = Math.Pow(probs[i], 1.0 / temperature)
            probs_row  = np.power(probs_row, 1.0 / temperature)
            row_sum    = probs_row.sum()             # compute new sum

            if row_sum == 0:                         # all zeros edge case
                next_id = int(np.argmax(bigram_probs[current_id]))
            else:
                probs_row /= row_sum                 # renormalize to sum=1
                # np.random.choice picks an index weighted by probs.
                # C# analogy: weighted random selection.
                next_id = int(np.random.choice(len(probs_row), p=probs_row))

            next_char = idx_to_char[next_id]
            generated.append(next_char)
            ids.append(next_id)

        return "".join(generated)

    # =========================================================================
    # STEP B6 (Numpy): GENERATE COMPLETIONS FOR 3 SQL PREFIXES
    # =========================================================================

    subsection("Step B6 (Numpy): SQL Completions - Greedy vs Temperature=0.7")

    test_prefixes = ["SELECT", "WHERE", "INSERT"]

    for prefix in test_prefixes:
        print()
        print("=" * 60)
        print("PREFIX: [" + prefix + "]")
        print("=" * 60)

        greedy_result = generate_greedy_numpy(prefix)
        print("Greedy  : " + greedy_result)

        temp_result = generate_temperature_numpy(prefix, temperature=0.7)
        print("Temp0.7 : " + temp_result)

        print()
        print("NOTE: Bigram only looks at 1 previous char (limited context).")
        print("NOTE: Neural model (PyTorch) uses " + str(8) + " chars of context.")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section("PROJECT SUMMARY")

print("""
What you built in this project:
  Part A - Prefix-Matching Completer (no ML)
    - Hard-coded 20 SQL templates in a Python list
    - Filtered with str.startswith() -- like C# string.StartsWith()
    - Sorted by length using sorted(key=len) -- like LINQ OrderBy(t => t.Length)
    - This is how autocomplete worked BEFORE machine learning

  Part B - Char-Level nanoGPT (with ML)
    - Built a vocabulary of unique characters from the SQL corpus
    - Encoded every character as an integer ID
    - Trained a tiny neural network to predict the next character
    - Generated completions using two strategies:
        Greedy     : always pick the most probable next character
        Temp=0.7   : sample from a sharpened probability distribution

Key Concepts Reinforced:
  - Vocabulary / tokenization : turning text into numbers the model can use
  - Training loop             : forward pass, compute loss, backprop, update weights
  - Greedy vs. temperature    : the tradeoff between safe and creative outputs
  - Prefix matching           : the non-ML baseline that ML tries to beat

Real-World Connection:
  - GitHub Copilot, ChatGPT, and VS Code IntelliSense all use ideas from Part B.
  - The difference is scale: they have billions of parameters, we have hundreds.
  - Part A is still used today for fast, low-latency keyword suggestions.

C# / SQL Analogies Used:
  - Python list            -- C# List<string>
  - dict                   -- C# Dictionary<K,V>
  - str.startswith()       -- C# string.StartsWith()
  - list comprehension     -- C# LINQ .Where() + .Select()
  - sorted(key=...)        -- C# LINQ .OrderBy()
  - try/except ImportError -- C# try/catch(DllNotFoundException)
  - nn.Module              -- C# interface with Forward() method
  - tensor                 -- C# multi-dimensional array (Array / float[,])
""")

print("=" * 60)
print("PROJECT 5 COMPLETE")
print("=" * 60)


# =============================================================================
# =============================================================================
#
#  QUIZ QUESTIONS
#  (Answers provided below each question in the comments)
#
# =============================================================================
# =============================================================================

# QUIZ QUESTION 1 (Multiple Choice)
# ----------------------------------
# In Part A, which Python method is used to check if a template starts
# with the user's prefix?
#
#   A) str.contains()
#   B) str.startswith()
#   C) str.find()
#   D) str.match()
#
# ANSWER: B -- str.startswith()
#   This is exactly like C# string.StartsWith(prefix, StringComparison.OrdinalIgnoreCase).
#   It returns True if the string begins with the given prefix, False otherwise.
#
# WHY the others are wrong:
#   A) str.contains() does not exist in Python (use 'in' operator instead)
#   C) str.find() returns the INDEX of the substring, not a True/False
#   D) str.match() is a regex method on re module objects, not str


# QUIZ QUESTION 2 (Multiple Choice)
# ----------------------------------
# What does "temperature" do in text generation?
#
#   A) It warms up the GPU before training starts.
#   B) It controls the learning rate during training.
#   C) It divides the logits before softmax to control randomness.
#   D) It determines how many characters to generate.
#
# ANSWER: C -- divides logits before softmax to control randomness.
#   temperature < 1.0 -> probabilities become more peaked -> less random (focused)
#   temperature > 1.0 -> probabilities become flatter   -> more random (creative)
#   temperature = 1.0 -> standard softmax, no change.
#
# Real-world use: ChatGPT uses temperature internally.
# When you want factual, safe output: use low temperature (0.2 - 0.5).
# When you want creative writing: use higher temperature (0.8 - 1.2).


# QUIZ QUESTION 3 (Short Answer)
# --------------------------------
# Explain the difference between Part A and Part B in one sentence each.
# Then explain: which one would you use in a production SQL IDE and why?
#
# ANSWER (Part A):
#   Part A uses simple prefix matching on a hard-coded list of templates --
#   no learning, no probabilities, just string.StartsWith() in a loop.
#   C# equivalent: templates.Where(t => t.StartsWith(prefix)).ToList()
#
# ANSWER (Part B):
#   Part B trains a tiny neural network to learn SQL character patterns,
#   then generates new completions character by character using learned probabilities.
#
# ANSWER (production choice):
#   In a real SQL IDE you would use BOTH:
#     - Part A for fast keyword/table-name lookup (instant, no GPU needed)
#     - Part B (or a larger LLM) for intelligent multi-line query completion
#   This hybrid approach is exactly what tools like GitHub Copilot do:
#   fast lexical matching for imports/keywords + LLM for complex suggestions.
#
# SQL analogy: Part A is like an index scan (fast lookup by key).
#              Part B is like a full-table scan with ML scoring (smarter, slower).


# BONUS QUIZ QUESTION 4 (Stretch)
# ---------------------------------
# If you increase the training epochs from 300 to 1000, what would you expect?
#
# ANSWER:
#   The loss should decrease further and the model should memorize the SQL templates
#   more accurately. With only 20 short templates (a tiny corpus), the model will
#   likely OVERFIT -- meaning it memorizes the training data perfectly but cannot
#   generalize to new SQL patterns it has never seen.
#   Overfitting is like a student who memorizes exam answers but cannot solve
#   new problems. With more diverse training data the model would generalize better.
#   C# analogy: like a unit test that only checks one specific input -- passes for
#               that input but gives no confidence for other inputs.

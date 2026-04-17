"""
=============================================================================
PROJECT 7: Company Name Generator - Build a Startup Name Brainstormer
=============================================================================

GLOSSARY (read this first, before looking at any code)
-------------------------------------------------------
Character-level model : A language model that works one CHARACTER at a time
                        (not one word). It reads "A-n-t-h-r-o-p-i-c" and
                        learns patterns like "after 'th' often comes 'r'".
                        Simpler than word-level, but surprisingly powerful
                        for learning naming conventions.

Corpus              : The body of text we train on. Here it is a list of
                      real tech company names. Latin for "body of text".

Vocabulary          : The complete set of unique characters the model knows.
                      For our company names, this includes a-z and A-Z.
                      C# analogy: HashSet<char> built from all training text.

Token               : A single unit the model processes. Here each token is
                      one character. In real LLMs tokens are sub-words.

Embedding           : A small list of numbers (a vector) that represents a
                      character. Instead of storing "character 5" the model
                      stores [0.3, -0.1, 0.7]. These numbers are LEARNED.
                      C# analogy: a float[] that gets updated during training.

Logits              : The raw output scores from the model, one per vocab
                      character. These are NOT probabilities yet. Any number.

Softmax             : A function that converts logits -> probabilities.
                      All outputs are between 0 and 1, and they sum to 1.
                      Formula: exp(x_i) / sum(exp(x_j) for all j)

Temperature         : A number that controls how "creative" generation is.
                      Divide logits by temperature BEFORE softmax.
                      temp < 1.0  -> top character dominates (boring but safe)
                      temp = 1.0  -> model's original probabilities (balanced)
                      temp > 1.0  -> all characters become more equal (wild)
                      C# analogy: like turning up the randomness dial.

Greedy              : Always pick the character with the HIGHEST probability.
                      No randomness. Like always ordering the same meal.

Training            : The process of adjusting the model's internal numbers
                      (weights) so it gets better at predicting the next
                      character. We measure error, compute gradients, and
                      nudge the weights in the right direction.
                      C# analogy: like a feedback loop that auto-tunes params.

Loss                : A single number measuring how WRONG the model is.
                      Lower loss = better predictions. We want to minimize it.
                      C# analogy: return value of a fitness function.

Gradient            : For each weight, how much does the loss change if we
                      nudge that weight slightly? Points "uphill" on the error
                      surface, so we move in the OPPOSITE direction.
                      C# analogy: the slope/derivative of the loss curve.

Epoch               : One complete pass through ALL the training data.
                      Like reading the whole list of company names once.

Adam Optimizer      : A popular algorithm that adjusts weights using gradients.
                      "Adaptive Moment Estimation". Better than simple gradient
                      descent because it adapts the step size per weight.
                      C# analogy: a PID controller that self-tunes.

CrossEntropyLoss    : The standard loss function for classification tasks.
                      Measures how surprised the model is by the correct answer.
                      Lower = less surprised = better at predicting next char.

nn.Module           : PyTorch base class for all neural network layers.
                      C# analogy: abstract base class every layer must inherit.

Context Window      : How many previous characters the model looks at when
                      deciding what comes next. Here we use 4 characters.
                      GPT-4 can look at ~128,000 characters back.

torch.no_grad()     : Tells PyTorch NOT to track gradients during this block.
                      We do this at generation time to save memory/speed.
                      C# analogy: using (var scope = ReadOnlyScope()) { }

Prefix / Root / Suffix : Parts of a company name.
                      Prefix  = beginning syllable, e.g. "Super", "Open"
                      Root    = core word or syllable, e.g. "base", "scale"
                      Suffix  = ending, e.g. "ify", "io", "ai"

=============================================================================
PROJECT OVERVIEW
=============================================================================

Real-world use cases this project demonstrates:

  1. Startup Founders  - Need 10 name ideas before registering a domain.
                         Checking dozens of possibilities is tedious.
                         A name generator gives instant options to evaluate.

  2. Domain Registrars - Sites like Namecheap show alternative names when
                         your first choice is taken. This AI approach can
                         suggest alternatives that FEEL like real tech names.

  3. Branding Agencies - Agencies create name lists for clients. An AI tool
                         generates hundreds of candidates instantly, then
                         humans filter to the best ones.

We will build this in TWO parts:

  PART A - Random combination generator (no ML at all)
  ----------------------------------------------------
  Hard-code lists of prefixes, roots, and suffixes common in tech names.
  Randomly combine them to produce 10 candidate names.
  Fast, simple, and useful - but ignores learned patterns.

  PART B - Character-level nanoGPT trained on real company names
  ---------------------------------------------------------------
  Use PyTorch (with graceful fallback to NumPy if not installed).
  Train a tiny model on 40 real company names.
  Generate 10 new names using THREE temperature settings:
    - Greedy     (temperature=0, equivalent) -> boring but safe
    - temp=0.8   -> creative and plausible
    - temp=1.5   -> wild and unusual
  Display all 3 side by side so you can see the temperature effect clearly.

=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import random       # random: built-in Python library for random number generation
                    # C# analogy: System.Random

import numpy as np  # numpy: fast math/array library (like System.Math + arrays)
                    # C# analogy: a super-powered version of double[]

# Try to import PyTorch. If it is not installed we fall back to a NumPy model.
# C# analogy: try { Assembly.Load("torch"); } catch (FileNotFoundException) { ... }
try:
    import torch                        # main PyTorch library (tensor computation)
    import torch.nn as nn               # neural network building blocks (layers)
    import torch.nn.functional as F     # stateless functions: softmax, relu, etc.
    TORCH_AVAILABLE = True              # flag so we can branch later
except ImportError:
    TORCH_AVAILABLE = False             # PyTorch not installed; use NumPy fallback

# =============================================================================
# TRAINING DATA: 40 real tech company names (hard-coded, no files needed)
# =============================================================================

# This list IS our training corpus.
# The model will read every character in every name and learn:
#   "what characters tend to follow what other characters in tech company names?"
# C# analogy: List<string> containing our training examples.

COMPANY_NAMES = [
    "Anthropic",     # AI safety company
    "Openai",        # GPT company (note: lowercase 'ai' on purpose for pattern learning)
    "Deepmind",      # Google's AI research lab
    "Mistral",       # French AI company
    "Cohere",        # enterprise NLP company
    "Inflection",    # conversational AI company
    "Stability",     # Stable Diffusion creators
    "Hugging",       # Hugging Face (open-source AI hub)
    "Runway",        # video generation AI
    "Jasper",        # AI writing tool
    "Notion",        # productivity / notes
    "Linear",        # project management
    "Vercel",        # frontend deployment
    "Supabase",      # open-source Firebase alternative
    "Render",        # cloud hosting
    "Planetscale",   # serverless MySQL
    "Neon",          # serverless Postgres
    "Turso",         # distributed SQLite
    "Convex",        # real-time backend
    "Liveblocks",    # collaborative tech
    "Resend",        # email for developers
    "Loops",         # email automation
    "Postmark",      # transactional email
    "Sendgrid",      # email API
    "Twilio",        # communication APIs
    "Stripe",        # payments
    "Plaid",         # fintech data API
    "Brex",          # corporate cards / fintech
    "Rippling",      # HR + IT platform
    "Gusto",         # payroll and HR
    "Lattice",       # performance management
    "Leapsome",      # employee engagement
    "Greenhouse",    # recruiting software
    "Lever",         # applicant tracking
    "Workday",       # enterprise HR
    "Salesforce",    # CRM platform
    "Hubspot",       # inbound marketing
    "Intercom",      # customer messaging
    "Zendesk",       # customer support
    "Freshdesk",     # help desk software
]

# =============================================================================
# PRINT HEADER
# =============================================================================

print("=" * 60)
print("PROJECT 7: Company Name Generator")
print("=" * 60)
print()
print("Training corpus: " + str(len(COMPANY_NAMES)) + " real tech company names")
print("-" * 40)
for i, name in enumerate(COMPANY_NAMES):   # enumerate gives index + value (like foreach with counter)
    print("  " + str(i + 1).rjust(2) + ". " + name)   # rjust(2) = right-justify in 2 chars
print("-" * 40)
print()

# =============================================================================
# =============================================================================
#  PART A: RANDOM COMBINATION GENERATOR (No Machine Learning)
# =============================================================================
# =============================================================================

print("=" * 60)
print("PART A: Random Combination Generator (No ML)")
print("=" * 60)
print()
print("Approach: combine hard-coded prefix + root + suffix lists.")
print("No training, no model, just random assembly.")
print("This is the 'naive baseline' before we bring in ML.")
print()

# -----------------------------------------------------------------------------
# Hard-coded word parts from real tech company naming patterns
# -----------------------------------------------------------------------------

# PREFIXES: syllables that appear at the START of tech company names.
# Chosen by manually observing patterns in real company names.
# C# analogy: string[] prefixes = { "Open", "Deep", ... };
PREFIXES = [
    "Open",     # OpenAI, OpenSea
    "Deep",     # DeepMind, DeepL
    "Super",    # Superhuman, SuperScale
    "Meta",     # Meta (Facebook), MetaBase
    "Neo",      # NeoVim, NeoFS
    "Hyper",    # HyperDX, Hyperscaler
    "Fast",     # FastAPI, FastMail
    "Data",     # Databricks, DataRobot
    "Cloud",    # Cloudflare, CloudSmith
    "Smart",    # SmartThings, SmartHR
    "Auto",     # AutoGPT, AutoML
    "Pro",      # ProtonMail, ProCore
    "True",     # TrueNorth, TrueCaller
    "Alpha",    # AlphaCode, AlphaSense
]

# ROOTS: the core word or syllable in the middle of the name.
# These are common in tech branding: short, punchy, memorable.
ROOTS = [
    "scale",    # Planetscale, Cloudscale
    "base",     # Supabase, Codebase
    "stack",    # FullStack, DevStack
    "flow",     # Airflow, Workflow
    "forge",    # GitForge, Cloudforge
    "sync",     # Notion Sync, FileSync
    "grid",     # Sendgrid, DataGrid
    "node",     # Node.js, EdgeNode
    "loop",     # Loops, DevLoop
    "lab",      # GitLab, CodeLab
    "hub",      # GitHub, HubSpot
    "core",     # Hardcore, Salescore
    "mind",     # DeepMind, ThinkMind
    "link",     # Interlink, DataLink
    "block",    # Liveblocks, BlockChain
]

# SUFFIXES: endings that make names sound like tech companies.
# Notice: many real names end in -ify, -ly, -io, -ai, -er.
SUFFIXES = [
    "ify",      # Spotify, Cloudify
    "ly",       # Grammarly, Bitly
    "io",       # GitHub.io, Linear.io
    "ai",       # Cohere AI, Jasper AI
    "er",       # Render, Lever
    "ent",      # Inflect, Gradient
    "al",       # Mistral, Lateral
    "ic",       # Anthropic, Elastic
    "on",       # Notion, Proton
    "ity",      # Stability, Clarity
    "ance",     # Performance, Balance
    "ix",       # Salesfix, FlowIx
    "ex",       # Convex, Vertex
    "el",       # Vercel, Kernel
]

# Seed the random number generator so results are reproducible.
# C# analogy: new Random(seed) -- same seed = same sequence every run.
random.seed(99)

print("Generating 10 random company names by combining prefix + root + suffix:")
print("-" * 40)
print()

# Generate 10 names by randomly picking one item from each list.
for i in range(10):                         # loop 10 times (i goes 0..9)
    prefix  = random.choice(PREFIXES)       # random.choice: pick one item from a list
    root    = random.choice(ROOTS)          # same for root
    suffix  = random.choice(SUFFIXES)       # same for suffix

    # Combine them: prefix stays as-is, root and suffix are lowercase
    # capitalize() makes the FIRST letter uppercase, rest lowercase
    name = prefix + root.capitalize() + suffix   # e.g. "Deep" + "Scale" + "ify" = "DeepScaleify"

    print("  " + str(i + 1).rjust(2) + ". " + name)   # print with a number

print()
print("Observations about Part A:")
print("  + Very fast: no training needed, instant results")
print("  + Controllable: you choose which prefixes/roots/suffixes to include")
print("  - Names feel mechanical: they do not capture real naming rhythm")
print("  - Cannot learn subtler patterns (e.g. 'Anthropic' does not fit prefix+root+suffix)")
print("  - Vocabulary is frozen: adding new patterns requires hand-editing lists")
print()
print("PART A complete!")
print()

# =============================================================================
# =============================================================================
#  PART B: CHAR-LEVEL NANOGPT TRAINED ON REAL COMPANY NAMES
# =============================================================================
# =============================================================================

print("=" * 60)
print("PART B: Character-Level nanoGPT on Real Company Names")
print("=" * 60)
print()

# =============================================================================
# STEP B1: Build the training corpus string
# =============================================================================
# We join all 40 names with newline characters.
# The newline '\n' acts as a "name boundary" token.
# When the model generates a '\n', it signals "end of name".
# C# analogy: string.Join("\n", COMPANY_NAMES)

print("STEP B1: Build the training corpus string")
print("-" * 40)

CORPUS = "\n".join(COMPANY_NAMES)   # join all names, one per line

print("Corpus (first 80 chars shown): " + repr(CORPUS[:80]) + "...")
print("Total corpus length          : " + str(len(CORPUS)) + " characters")
print()

# =============================================================================
# STEP B2: Build vocabulary (list of unique characters)
# =============================================================================
# sorted(set(text)) = get unique chars and sort them alphabetically.
# C# analogy: new SortedSet<char>(CORPUS).ToList()

print("STEP B2: Build vocabulary (unique characters)")
print("-" * 40)

CHARS          = sorted(set(CORPUS))      # sorted list of unique characters in corpus
VOCAB_SIZE     = len(CHARS)               # number of unique characters

# char_to_idx: character -> integer index
# C# analogy: Dictionary<char, int>
# enumerate gives (0,'A'), (1,'B'), ... we swap to get {'A':0, 'B':1, ...}
char_to_idx    = {ch: i for i, ch in enumerate(CHARS)}

# idx_to_char: integer index -> character (reverse mapping)
# C# analogy: Dictionary<int, char>
idx_to_char    = {i: ch for i, ch in enumerate(CHARS)}

print("Unique characters : " + repr("".join(CHARS)))
print("Vocabulary size   : " + str(VOCAB_SIZE))
print()

# =============================================================================
# STEP B3: Encode the corpus as a list of integers
# =============================================================================
# Every character is replaced by its index in char_to_idx.
# e.g. 'A' -> 0, 'n' -> 27, etc.
# C# analogy: CORPUS.Select(c => char_to_idx[c]).ToList()

print("STEP B3: Encode the corpus as integers")
print("-" * 40)

ENCODED = [char_to_idx[ch] for ch in CORPUS]   # list comprehension: map each char to its index

print("First 20 chars    : " + repr(CORPUS[:20]))
print("First 20 encoded  : " + str(ENCODED[:20]))
print()

# =============================================================================
# STEP B4: Build training pairs (context -> next character)
# =============================================================================
# CONTEXT_SIZE = how many previous characters the model looks at.
# We use 4. (Bigram uses 1; GPT-4 uses ~128,000)
#
# Example with CONTEXT_SIZE=3 and text "abc":
#   Input (context)   Target (next char)
#   ['A','n','t']  -> 'h'
#   ['n','t','h']  -> 'r'
#   ['t','h','r']  -> 'o'

print("STEP B4: Build training pairs (context -> next char)")
print("-" * 40)

CONTEXT_SIZE = 4    # look back this many characters to predict the next one

X_list = []    # will hold input contexts (each is a list of CONTEXT_SIZE integers)
Y_list = []    # will hold the corresponding target character indices

for i in range(len(ENCODED) - CONTEXT_SIZE):    # slide a window across the encoded corpus
    context = ENCODED[i : i + CONTEXT_SIZE]      # take CONTEXT_SIZE chars as input
    target  = ENCODED[i + CONTEXT_SIZE]          # the char immediately after is the label
    X_list.append(context)                       # add to input list
    Y_list.append(target)                        # add to label list

print("Context size            : " + str(CONTEXT_SIZE) + " characters")
print("Number of training pairs: " + str(len(X_list)))
print()
print("First 3 training pairs:")
for k in range(3):                                     # show the first 3 examples
    ctx_chars = [idx_to_char[c] for c in X_list[k]]   # convert indices back to chars
    tgt_char  = idx_to_char[Y_list[k]]                 # convert target index to char
    print("  Context " + repr("".join(ctx_chars)) + " -> Target " + repr(tgt_char))
print()

# =============================================================================
# HELPER: softmax (used in both PyTorch and NumPy paths)
# =============================================================================
# Converts raw scores (any numbers) into probabilities that sum to 1.
# We subtract max(x) first for numerical stability (avoids exp overflow).
# C# analogy: a utility method used by both model versions.

def softmax_numpy(x):
    """
    Convert a 1-D numpy array of raw scores into probabilities.

    x : 1-D numpy array of any floating-point numbers (logits)
    Returns: numpy array of the same shape, values in (0,1), sum = 1.0
    """
    e = np.exp(x - x.max())   # subtract max before exp to avoid very large numbers
    return e / e.sum()         # divide each by the total so they sum to 1

# =============================================================================
# =============================================================================
# BRANCH: Use PyTorch if available, else fall back to NumPy
# =============================================================================
# =============================================================================

if not TORCH_AVAILABLE:

    # =========================================================================
    # NUMPY FALLBACK: bigram-style model without PyTorch
    # =========================================================================
    # When PyTorch is not installed we build a CONTEXT-AWARE probability table.
    # It is more powerful than a pure bigram (it looks at CONTEXT_SIZE chars),
    # but it is NOT a neural network - just a glorified lookup table.

    print("PyTorch is NOT installed.")
    print("Falling back to a context-aware NumPy probability table.")
    print("To install PyTorch, open a terminal and run:  pip install torch")
    print()

    # -------------------------------------------------------------------------
    # STEP B5 (NumPy): Build context probability table
    # -------------------------------------------------------------------------
    # For every unique context (tuple of CONTEXT_SIZE char indices) we have seen,
    # record how many times each character followed it.
    # C# analogy: Dictionary<int[], Dictionary<int, int>> (counts per context)

    print("STEP B5: Building context probability table (NumPy fallback)")
    print("-" * 40)

    from collections import defaultdict   # defaultdict: dict with automatic default values
                                          # C# analogy: Dictionary with GetOrAdd behaviour

    # context_counts[context_tuple][next_char_idx] = how many times this happened
    # defaultdict(lambda: np.zeros(VOCAB_SIZE)) means: if a key is missing,
    # automatically create a numpy array of zeros.
    context_counts = defaultdict(lambda: np.zeros(VOCAB_SIZE, dtype=np.float64))

    for i in range(len(ENCODED) - CONTEXT_SIZE):   # slide window across corpus
        ctx  = tuple(ENCODED[i : i + CONTEXT_SIZE])  # tuple = hashable key for dict
        nxt  = ENCODED[i + CONTEXT_SIZE]             # next character index
        context_counts[ctx][nxt] += 1                # increment that bucket

    # Convert counts -> probabilities (with add-1 smoothing to avoid zeros)
    context_probs = {}     # will hold: context_tuple -> probability array

    for ctx, counts in context_counts.items():   # iterate over all learned contexts
        smoothed = counts + 1.0                  # add 1 to every count (Laplace smoothing)
        context_probs[ctx] = smoothed / smoothed.sum()   # normalise to probabilities

    print("Unique contexts learned: " + str(len(context_probs)))
    print()

    # -------------------------------------------------------------------------
    # STEP B6 (NumPy): Generation with temperature
    # -------------------------------------------------------------------------

    def generate_name_numpy(context_probs, char_to_idx, idx_to_char,
                            seed_text, context_size, max_chars, temperature):
        """
        Generate a company name using the NumPy context-probability table.

        context_probs : dict mapping context tuple -> probability array
        char_to_idx   : dict char -> int index
        idx_to_char   : dict int index -> char
        seed_text     : string to begin generation from (at least context_size chars)
        context_size  : how many previous chars to look at
        max_chars     : maximum characters to generate before stopping
        temperature   : float controlling creativity
                        0.01 = near-greedy, 0.8 = balanced, 1.5 = wild
        """
        # Encode the seed text into integer indices
        # .get(ch, 0) = if the char is not in vocabulary, use index 0 (safest fallback)
        context = [char_to_idx.get(ch, 0) for ch in seed_text[-context_size:]]
        # [-context_size:] = take only the LAST context_size characters of the seed

        generated = list(seed_text)   # start the output with the seed

        for _ in range(max_chars):    # generate up to max_chars new characters

            ctx_key = tuple(context)  # convert list to tuple so it works as dict key

            if ctx_key in context_probs:
                raw_probs = context_probs[ctx_key]    # learned distribution for this context
            else:
                raw_probs = np.ones(VOCAB_SIZE) / VOCAB_SIZE   # unseen context: uniform distribution

            # Apply temperature: divide logits (log-scale scores) by temperature.
            # We work in log-space, then exponentiate back.
            # log(prob) / temperature is equivalent to logit / temperature.
            log_probs    = np.log(raw_probs + 1e-10)   # add tiny value to avoid log(0)
            scaled_logs  = log_probs / temperature      # scale by temperature
            scaled_probs = softmax_numpy(scaled_logs)   # convert back to probabilities

            # Sample one character index, weighted by scaled_probs
            # C# analogy: a weighted random picker
            next_idx = np.random.choice(VOCAB_SIZE, p=scaled_probs)

            next_char = idx_to_char[next_idx]   # convert index back to character

            if next_char == "\n":               # newline = end-of-name signal
                break                           # stop generating this name

            generated.append(next_char)         # add the character to our output

            context = context[1:] + [next_idx]  # slide context window: drop oldest, add newest

        return "".join(generated)   # join list of characters into one string

    # -------------------------------------------------------------------------
    # STEP B7 (NumPy): Generate 10 names at three temperature settings
    # -------------------------------------------------------------------------

    print("STEP B6: Generating names at 3 temperature settings (NumPy fallback)")
    print("-" * 40)
    print()

    # Seed characters to start generation from (must be in the vocabulary)
    # We try a few different starting characters to get variety.
    SEED_CHARS = ["A", "S", "C", "D", "T", "P", "N", "G", "L", "R"]

    # Temperature settings we will compare
    TEMP_GREEDY = 0.01   # near-zero temperature = almost always picks the top character
    TEMP_MEDIUM = 0.8    # creative but still mostly plausible names
    TEMP_WILD   = 1.5    # high temperature = very random, unusual combinations

    np.random.seed(42)   # fix seed for reproducibility (C# analogy: new Random(42))

    print("  " + "Greedy (t=0.01)".ljust(22) + "Balanced (t=0.8)".ljust(22) + "Wild (t=1.5)")
    print("  " + "-" * 65)

    for i, seed_char in enumerate(SEED_CHARS):   # loop over the 10 seed characters

        # Check if the seed character is in our vocabulary; skip if not
        if seed_char not in char_to_idx:
            print("  " + str(i + 1).rjust(2) + ". [seed '" + seed_char + "' not in vocab, skipping]")
            continue   # move to the next iteration of the loop

        # Generate a name at each of the three temperature settings
        name_greedy = generate_name_numpy(
            context_probs, char_to_idx, idx_to_char,
            seed_text    = seed_char,    # start with this character
            context_size = CONTEXT_SIZE,
            max_chars    = 14,           # limit length to keep names realistic
            temperature  = TEMP_GREEDY
        )

        name_medium = generate_name_numpy(
            context_probs, char_to_idx, idx_to_char,
            seed_text    = seed_char,
            context_size = CONTEXT_SIZE,
            max_chars    = 14,
            temperature  = TEMP_MEDIUM
        )

        name_wild = generate_name_numpy(
            context_probs, char_to_idx, idx_to_char,
            seed_text    = seed_char,
            context_size = CONTEXT_SIZE,
            max_chars    = 14,
            temperature  = TEMP_WILD
        )

        # Format with fixed column widths for a side-by-side table
        col1 = name_greedy.ljust(22)    # ljust(22) = left-align in 22 characters
        col2 = name_medium.ljust(22)
        print("  " + str(i + 1).rjust(2) + ". " + col1 + col2 + name_wild)

    print()

else:

    # =========================================================================
    # PYTORCH PATH: Full neural network (char-level GPT-style model)
    # =========================================================================

    print("PyTorch is available! Running the full neural network model.")
    print()

    # -------------------------------------------------------------------------
    # STEP B5 (PyTorch): Define the neural network architecture
    # -------------------------------------------------------------------------
    # Architecture diagram:
    #
    #   INPUT: 4 character indices, e.g. [0, 13, 19, 7]
    #          |
    #          v
    #   EMBEDDING LAYER: each index -> a vector of EMBED_DIM numbers
    #          e.g. [0,13,19,7] -> 4 x 16 = 64 numbers
    #          then FLATTEN to one long vector: (CONTEXT_SIZE * EMBED_DIM,)
    #          |
    #          v
    #   HIDDEN LAYER (Linear + ReLU):
    #          input : CONTEXT_SIZE * EMBED_DIM = 64 numbers
    #          output: HIDDEN_DIM = 128 numbers
    #          ReLU clips negatives to 0 (adds non-linearity)
    #          |
    #          v
    #   OUTPUT LAYER (Linear):
    #          input : HIDDEN_DIM = 128 numbers
    #          output: VOCAB_SIZE numbers (one raw score per character)
    #          |
    #          v
    #   SOFTMAX -> probabilities over all vocab characters
    #
    # C# analogy: a pipeline of transforms, each reshaping the data.

    print("STEP B5: Define the neural network model (PyTorch)")
    print("-" * 40)

    EMBED_DIM  = 16    # how many numbers represent each character's embedding
    HIDDEN_DIM = 128   # number of neurons in the hidden layer

    class CompanyNameModel(nn.Module):
        """
        A tiny character-level language model for generating company names.

        Inherits from nn.Module - the PyTorch base class for all models.
        C# analogy: class CompanyNameModel : NeuralNetworkBase { ... }
        """

        def __init__(self, vocab_size, embed_dim, context_size, hidden_dim):
            """
            Constructor: set up all the layers.
            C# analogy: public CompanyNameModel(int vocabSize, ...) { ... }

            vocab_size   : number of unique characters in the vocabulary
            embed_dim    : size of each character's embedding vector
            context_size : how many previous characters we look at
            hidden_dim   : number of neurons in the hidden layer
            """
            super().__init__()      # MUST call parent constructor
                                    # C# analogy: base() in the constructor chain

            self.vocab_size   = vocab_size    # store for use in forward()
            self.embed_dim    = embed_dim     # store for use in forward()
            self.context_size = context_size  # store for use in forward()

            # EMBEDDING TABLE: maps integer char indices -> dense vectors
            # Shape: (vocab_size, embed_dim)
            # Think of it as a lookup table of learnable float arrays.
            # C# analogy: float[,] embeddingTable = new float[vocab_size, embed_dim];
            self.embedding = nn.Embedding(vocab_size, embed_dim)

            # HIDDEN LAYER: linear transformation with ReLU
            # Input:  CONTEXT_SIZE * embed_dim numbers (flattened embeddings)
            # Output: hidden_dim numbers
            # C# analogy: applies y = W * x + b where W and b are learned
            self.hidden = nn.Linear(context_size * embed_dim, hidden_dim)

            # OUTPUT LAYER: linear transformation to vocab scores
            # Input:  hidden_dim numbers
            # Output: vocab_size scores (one per character)
            self.output = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            """
            Forward pass: defines the computation when we call model(input).
            PyTorch calls this automatically. Do not call forward() directly.
            C# analogy: override object Compute(Tensor input) { ... }

            x : input tensor, shape (batch_size, context_size)
                each row is a context window of CONTEXT_SIZE char indices
            Returns: logits tensor, shape (batch_size, vocab_size)
            """
            # STEP 1: Embedding lookup
            # For each integer index, look up its embedding vector.
            # x shape:   (batch_size, context_size)
            # emb shape: (batch_size, context_size, embed_dim)
            emb = self.embedding(x)         # look up embeddings for all chars in context

            # STEP 2: Flatten embeddings into one long vector per example
            # From (batch_size, context_size, embed_dim)
            # To   (batch_size, context_size * embed_dim)
            # C# analogy: Flatten a jagged array into a single row
            batch_size = emb.shape[0]       # number of examples in this batch
            flat = emb.view(batch_size, -1) # -1 tells PyTorch to calculate this dimension

            # STEP 3: Hidden layer (Linear) + ReLU activation
            # torch.relu(y): replaces all negative values with 0
            # C# analogy: Math.Max(0, value)   applied to every element
            h = torch.relu(self.hidden(flat))   # shape: (batch_size, hidden_dim)

            # STEP 4: Output layer -> logits (raw scores, no activation)
            # CrossEntropyLoss will apply softmax internally, so we stop here.
            logits = self.output(h)     # shape: (batch_size, vocab_size)

            return logits               # return raw scores

    # Convert training data to PyTorch tensors
    # dtype=torch.long -> 64-bit integer (required for embedding index lookups)
    # C# analogy: long[] X_array = X_list.SelectMany(row => row).ToArray();
    X_tensor = torch.tensor(X_list, dtype=torch.long)   # shape: (num_examples, CONTEXT_SIZE)
    Y_tensor = torch.tensor(Y_list, dtype=torch.long)   # shape: (num_examples,)

    # Instantiate the model with our hyperparameters
    model = CompanyNameModel(
        vocab_size   = VOCAB_SIZE,      # how many unique characters
        embed_dim    = EMBED_DIM,       # 16 numbers per character embedding
        context_size = CONTEXT_SIZE,    # look back 4 characters
        hidden_dim   = HIDDEN_DIM       # 128 neurons in the hidden layer
    )

    # Count total trainable parameters
    # p.numel() = number of elements in a parameter tensor (numel = number of elements)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model architecture:")
    print("  Embedding layer : " + str(VOCAB_SIZE) + " chars x " + str(EMBED_DIM) + " dims")
    print("  Hidden layer    : " + str(CONTEXT_SIZE * EMBED_DIM) + " -> " + str(HIDDEN_DIM) + " (ReLU)")
    print("  Output layer    : " + str(HIDDEN_DIM) + " -> " + str(VOCAB_SIZE) + " (logits)")
    print("  Total trainable parameters: " + str(total_params))
    print()

    # -------------------------------------------------------------------------
    # STEP B6 (PyTorch): Set up optimizer and loss function
    # -------------------------------------------------------------------------

    print("STEP B6: Set up optimizer and loss function")
    print("-" * 40)

    LEARNING_RATE = 0.005   # how big a step to take each time we update weights
                            # too big = model overshoots; too small = very slow training

    # Adam optimizer: smarter than plain gradient descent.
    # It tracks a momentum and an adaptive step size per weight.
    # model.parameters() = all the weights the optimizer should update.
    # C# analogy: an auto-tuning step-size controller.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # CrossEntropyLoss: measures how surprised the model is by the correct answer.
    # Internally it applies softmax + negative log likelihood.
    # C# analogy: a scoring function we want to minimise.
    loss_fn = nn.CrossEntropyLoss()

    print("Optimizer     : Adam  (learning rate = " + str(LEARNING_RATE) + ")")
    print("Loss function : CrossEntropyLoss")
    print()

    # -------------------------------------------------------------------------
    # STEP B7 (PyTorch): Train the model
    # -------------------------------------------------------------------------

    print("STEP B7: Train the model")
    print("-" * 40)

    NUM_EPOCHS  = 800    # number of full passes through the training data
    PRINT_EVERY = 200    # print progress every N epochs

    print("Training for " + str(NUM_EPOCHS) + " epochs...")
    print()

    # model.train(): switch to training mode
    # C# analogy: model.IsTraining = true;
    model.train()

    for epoch in range(1, NUM_EPOCHS + 1):   # epoch counts from 1 to NUM_EPOCHS inclusive

        # --- FORWARD PASS ---
        # Feed ALL training examples through the model at once (full-batch training).
        # logits shape: (num_examples, VOCAB_SIZE)
        logits = model(X_tensor)    # calls model.forward(X_tensor) internally

        # --- COMPUTE LOSS ---
        # CrossEntropyLoss expects:
        #   logits:  (num_examples, num_classes)  <- shape of our logits
        #   targets: (num_examples,)               <- shape of Y_tensor
        loss = loss_fn(logits, Y_tensor)   # single float: how wrong the model is

        # --- BACKWARD PASS ---
        optimizer.zero_grad()   # clear gradients from the PREVIOUS step
                                # if we skip this, gradients accumulate and explode
                                # C# analogy: reset an accumulator to zero

        loss.backward()         # compute gradient for every weight
                                # (how much does each weight contribute to the loss?)
                                # C# analogy: auto-compute all partial derivatives

        # --- UPDATE WEIGHTS ---
        optimizer.step()        # move each weight slightly in the direction that
                                # reduces the loss
                                # C# analogy: apply the computed correction to each param

        # Print a progress line every PRINT_EVERY epochs
        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print("  Epoch " + str(epoch).rjust(4) + "/" + str(NUM_EPOCHS) +
                  "   Loss: " + str(round(loss.item(), 4)))
            # .item() converts a PyTorch 0-dim tensor to a plain Python float

    print()
    print("Training complete!")
    print()

    # -------------------------------------------------------------------------
    # STEP B8 (PyTorch): Generation function with temperature
    # -------------------------------------------------------------------------

    def generate_name_pytorch(model, char_to_idx, idx_to_char,
                               seed_char, context_size, max_chars, temperature):
        """
        Generate a company name using the trained PyTorch model.

        model        : the trained CompanyNameModel instance
        char_to_idx  : dict mapping character to integer index
        idx_to_char  : dict mapping integer index to character
        seed_char    : single character to begin generation from
        context_size : how many previous chars the model looks at
        max_chars    : maximum new characters to generate
        temperature  : float controlling creativity
                       near 0 = near-greedy, 0.8 = balanced, 1.5 = wild

        Returns: a string (the generated name, without the seed char's newline)
        """
        model.eval()    # switch to evaluation mode (disables dropout etc.)
                        # C# analogy: model.IsTraining = false;

        # Encode the seed character; use index 0 if not in vocabulary
        seed_idx = char_to_idx.get(seed_char, 0)

        # Build an initial context: pad with zeros on the left if needed.
        # e.g. seed='A', context_size=4 -> [0, 0, 0, seed_idx]
        context = [0] * (context_size - 1) + [seed_idx]   # list of context_size ints

        generated = [seed_char]    # start the output with the seed character

        with torch.no_grad():      # tell PyTorch: no gradients needed here
                                   # this saves memory and speeds up generation
            for _ in range(max_chars):    # generate up to max_chars characters

                # Convert current context list to a 2-D tensor: shape (1, context_size)
                # unsqueeze(0) adds the batch dimension (1 example in our batch)
                # C# analogy: wrap a 1-D array in a 2-D array with one row
                ctx_tensor = torch.tensor([context], dtype=torch.long)

                # Forward pass: get raw scores for the next character
                logits = model(ctx_tensor)       # shape: (1, VOCAB_SIZE)

                # Take the first (only) row and convert to numpy for manipulation
                logits_np = logits[0].numpy()    # shape: (VOCAB_SIZE,)

                # Apply temperature by dividing the logits
                # temperature near 0  -> top score dominates massively (near-greedy)
                # temperature = 1.0   -> original distribution unchanged
                # temperature = 1.5   -> scores squashed together (more random)
                scaled = logits_np / max(temperature, 1e-6)   # avoid division by zero

                # Convert scaled logits to probabilities
                probs = softmax_numpy(scaled)    # shape: (VOCAB_SIZE,)

                # Sample one character index according to probabilities
                # C# analogy: a weighted random integer picker
                next_idx = np.random.choice(VOCAB_SIZE, p=probs)

                next_char = idx_to_char[next_idx]   # convert index back to character

                if next_char == "\n":   # newline = end-of-name signal
                    break               # stop generating this name

                generated.append(next_char)         # add character to output

                # Slide the context window forward:
                # drop the oldest character (context[1:]) and add the new one
                context = context[1:] + [next_idx]

        return "".join(generated)   # join list into a single string

    # -------------------------------------------------------------------------
    # STEP B9 (PyTorch): Generate 10 names at 3 temperature settings
    # -------------------------------------------------------------------------

    print("STEP B8: Generate 10 company names at 3 temperature settings")
    print("-" * 40)
    print()
    print("Temperature effect:")
    print("  Greedy (t~0) -> safe, deterministic, boringly repetitive")
    print("  t = 0.8      -> creative but still plausible tech-sounding names")
    print("  t = 1.5      -> wild, unusual, sometimes nonsensical names")
    print()

    # Seed characters: one for each of the 10 names we will generate.
    # These are the first characters our model will extend.
    SEED_CHARS = ["A", "S", "C", "D", "T", "P", "N", "G", "L", "R"]

    TEMP_GREEDY = 0.01   # effectively greedy: top character almost always wins
    TEMP_MEDIUM = 0.8    # sweet spot: creative yet plausible
    TEMP_WILD   = 1.5    # high entropy: expect weird but memorable names

    # Fix random seeds for reproducibility
    torch.manual_seed(42)   # PyTorch random seed (C# analogy: new Random(42))
    np.random.seed(42)      # NumPy random seed

    # Print column header row
    header_col1 = "Greedy (t~0)".ljust(22)    # ljust = left-justify in N chars
    header_col2 = "Balanced (t=0.8)".ljust(22)
    print("  " + header_col1 + header_col2 + "Wild (t=1.5)")
    print("  " + "-" * 65)

    for i, seed_char in enumerate(SEED_CHARS):   # loop over all 10 seed characters

        # Check the seed character is in our vocabulary
        if seed_char not in char_to_idx:
            print("  " + str(i + 1).rjust(2) + ". ['" + seed_char + "' not in vocab - skipping]")
            continue   # skip to the next seed

        # Generate one name per temperature setting
        name_greedy = generate_name_pytorch(
            model, char_to_idx, idx_to_char,
            seed_char    = seed_char,
            context_size = CONTEXT_SIZE,
            max_chars    = 14,          # cap at 14 chars to keep names realistic
            temperature  = TEMP_GREEDY
        )

        name_medium = generate_name_pytorch(
            model, char_to_idx, idx_to_char,
            seed_char    = seed_char,
            context_size = CONTEXT_SIZE,
            max_chars    = 14,
            temperature  = TEMP_MEDIUM
        )

        name_wild = generate_name_pytorch(
            model, char_to_idx, idx_to_char,
            seed_char    = seed_char,
            context_size = CONTEXT_SIZE,
            max_chars    = 14,
            temperature  = TEMP_WILD
        )

        # Format into fixed-width columns for a clean side-by-side table
        col1 = name_greedy.ljust(22)    # pad with spaces to 22 chars wide
        col2 = name_medium.ljust(22)
        print("  " + str(i + 1).rjust(2) + ". " + col1 + col2 + name_wild)

    print()

# =============================================================================
# SHARED: Temperature explanation (printed regardless of PyTorch availability)
# =============================================================================

print("=" * 60)
print("HOW TEMPERATURE WORKS - VISUAL EXPLANATION")
print("=" * 60)
print()
print("Imagine the model sees 'Str' and must pick the next character.")
print("Its raw scores (logits) might be:")
print()
print("  'i' -> 3.0   (seen in 'Stripe', 'Strip')")
print("  'e' -> 2.0   (seen in 'Strength')")
print("  'a' -> 1.0   (less common after 'Str')")
print("  'o' -> 0.5   (rare)")
print("  'u' -> 0.2   (rare)")
print()

# Calculate and display how temperature reshapes probabilities
example_logits = np.array([3.0, 2.0, 1.0, 0.5, 0.2])   # fake logits
example_chars  = ["i", "e", "a", "o", "u"]               # corresponding characters

print("  Char   Logit   t=0.1    t=0.8    t=1.0    t=1.5")
print("  " + "-" * 52)

for idx_c, (ch, logit) in enumerate(zip(example_chars, example_logits)):
    row = "  " + ch.rjust(4) + "   " + str(round(logit, 1)).rjust(5) + "   "
    for temp in [0.1, 0.8, 1.0, 1.5]:
        scaled_logits = example_logits / temp      # scale all logits by temperature
        probs_temp    = softmax_numpy(scaled_logits)   # convert to probabilities
        prob_this_ch  = probs_temp[idx_c]          # probability of this character
        row += str(round(prob_this_ch * 100, 1)).rjust(5) + "%  "
    print(row)

print()
print("Key observations:")
print("  At t=0.1: 'i' gets ~100% probability. Almost always chosen. Boring.")
print("  At t=0.8: 'i' gets ~75%. 'e' gets ~20%. Variety but still plausible.")
print("  At t=1.0: model's original probabilities. Balanced.")
print("  At t=1.5: 'i' gets ~45%. Even 'u' gets ~10%. Wild, unexpected choices.")
print()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("=" * 60)
print("SUMMARY: Part A vs Part B")
print("=" * 60)
print()
print("PART A - Random Combination:")
print("  Method  : Hard-coded prefix + root + suffix lists")
print("  Training: None (no data needed)")
print("  Quality : Names feel assembled, not organic")
print("  Control : Easy (just edit the lists)")
print("  Speed   : Instant")
print()
print("PART B - Character-Level Model:")
print("  Method  : Char-level neural network trained on real names")
print("  Training: Learns from 40 real company names")
print("  Quality : Names have the 'rhythm' of real tech company names")
print("  Control : Via temperature (0 = safe, 1.5 = wild)")
print("  Speed   : Needs a training step first, then fast generation")
print()
print("Real-world pipeline (what branding agencies actually do):")
print("  1. Generate hundreds of candidates with temp=0.8 (balanced)")
print("  2. Filter by domain availability and trademark search")
print("  3. Human curators shortlist the best 10-20")
print("  4. Client picks from the shortlist")
print()
print("What you learned in this project:")
print("  - Characters can be encoded as integers and fed to a neural net")
print("  - A model trained on text learns the 'style' of that text")
print("  - Temperature is a single knob that trades safety for creativity")
print("  - The same generation technique powers real GPT-based products")
print()
print("=" * 60)
print("Project 7 complete!")
print("Next project: project_08 (coming soon)")
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
#   In Part B, we split the corpus into (context, target) pairs.
#   CONTEXT_SIZE is set to 4. If the corpus contains the word "Stripe",
#   which of the following is a valid training pair?
#
#   A) Context = ['S','t','r','i']    Target = 'p'
#   B) Context = ['S','t','r']        Target = 'i'
#   C) Context = ['S','t','r','i','p'] Target = 'e'
#   D) Context = ['r','i','p','e']    Target = 'S'
#
# ANSWER: A
#
#   With CONTEXT_SIZE=4, the input is always EXACTLY 4 characters and the
#   target is the SINGLE character immediately after those 4.
#   'S','t','r','i' are the first 4 chars of "Stripe", and 'p' follows.
#   Option B has only 3 context chars (wrong length).
#   Option C has 5 context chars (too many).
#   Option D is looking backwards (the model predicts FORWARD, not backward).
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# QUESTION 2 (Short answer):
#
#   Part A (random combination) is fast and requires no training data.
#   Why is Part B (neural network) often more useful for branding despite
#   requiring training and being slower to set up?
#
# ANSWER:
#   Part A can only produce combinations of its hard-coded lists. It cannot
#   produce names like "Anthropic" or "Mistral" because those do not follow a
#   simple prefix+root+suffix pattern. The lists are frozen: adding new naming
#   styles requires manual editing by a human.
#
#   Part B learns the RHYTHM and CHARACTER PATTERNS of real tech names
#   automatically from data. It picks up patterns like:
#     - Short names tend to end in vowels or simple consonants
#     - Capital letters appear at the start
#     - Certain letter combinations (like 'str', 'pl', 'gr') are common
#   Because it learns from data, you can easily re-train it on a different
#   domain (e.g., pharmaceutical names, fashion brands) just by swapping the
#   training list, with no manual rule-writing. That flexibility is the core
#   advantage of learned models over hard-coded rules.
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# QUESTION 3 (Concept):
#
#   Explain in simple terms what happens to the probability distribution
#   when temperature is set to 0.01 (near zero) vs 1.5 (high).
#   Use the following analogy to frame your answer:
#     "probabilities as heights of bars on a bar chart."
#
# ANSWER:
#   Think of each character's probability as the height of a bar on a chart.
#   Before temperature is applied, the bars have some natural heights from
#   the model (e.g., 'i' is tall, 'u' is short).
#
#   Temperature near 0 (e.g., 0.01):
#     The logits are DIVIDED by a tiny number (0.01), making them MUCH larger.
#     After softmax, the tallest bar becomes almost 100% and all other bars
#     shrink to nearly zero. The chart looks like one huge spike.
#     Result: the model almost always picks the same (most likely) character.
#     Generation is deterministic and repetitive -- all "Greedy" names look
#     the same, like Anthropic -> Anthropic -> Anthropic every time.
#
#   High temperature (e.g., 1.5):
#     The logits are divided by 1.5, making them SMALLER and CLOSER TOGETHER.
#     After softmax, the bars all become more similar in height.
#     The chart looks like a nearly flat distribution.
#     Result: even unlikely characters get a meaningful chance of being picked.
#     Generation is unpredictable and creative -- sometimes producing
#     surprisingly catchy names, sometimes producing nonsense.
#
#   In practice: temperature=0.8 is the sweet spot for name generation.
#   The top choices still win most of the time, but there is enough variety
#   to produce diverse and interesting outputs across 10 tries.
# -----------------------------------------------------------------------

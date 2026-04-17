"""
=============================================================================
PROJECT 2: Custom Chatbot - Build Your Own Q&A Bot
=============================================================================

GLOSSARY (read this first, before looking at any code)
-------------------------------------------------------
Chatbot         : A program that has a "conversation" with a user by reading
                  their input and producing a reply.

Rule-Based      : A system that follows explicit IF/ELSE rules written by a
                  human. No learning happens -- the rules are fixed forever.
                  Think of it like a big switch-statement in C#.

Machine Learning (ML) : A system that LEARNS patterns from data instead of
                  following hand-written rules.

Character-Level Model : An ML model that reads and predicts one CHARACTER at
                  a time (a, b, c...) rather than whole words. Very simple,
                  great for learning the idea.

Training        : Showing the model many examples so it can adjust its
                  internal numbers (weights) to give better answers.

Weights / Parameters : The numbers inside a neural network that get adjusted
                  during training. Like the "settings" of the model.

Loss            : A number that measures how wrong the model's prediction is.
                  Lower loss = better model. We try to minimize this.

Softmax         : A math function that turns a list of raw numbers into
                  probabilities that sum to 1.0. Think of it as normalizing
                  a list of scores into percentages.

Embedding       : Turning a character (like 'a') into a list of numbers that
                  the model can do math on.

Temperature     : Controls how "creative" (random) the model is when picking
                  the next character. Low = safe, High = wild.

Vocabulary      : The full set of unique characters the model knows about.

One-Hot Encoding: Representing a character as a list of zeros with a single 1
                  at the position of that character in the vocabulary.
                  Like a bitmask in C#.

PyTorch         : A popular Python library for building neural networks.
                  Think of it as a framework like ASP.NET but for ML.

Tensor          : PyTorch's version of a multi-dimensional array. Like a
                  List<List<double>> but with GPU support and auto-gradients.

Gradient        : The direction and size of the adjustment we need to make to
                  the weights to reduce the loss. Calculus in action.

Backpropagation : The algorithm that calculates gradients by working backwards
                  through the network. Automatic in PyTorch.

Optimizer       : The code that actually updates the weights using the
                  gradients. We use Adam, a popular choice.

=============================================================================
PART A: Rule-Based Chatbot (No Machine Learning)
=============================================================================

OVERVIEW:
  In Part A we build the simplest possible chatbot.
  It looks for KEYWORDS in what the user types, then returns a pre-written
  answer from a dictionary. No math, no ML, no surprises.

  C# analogy: This is like a Dictionary<string, string> lookup with a
  series of string.Contains() checks.

PART B OVERVIEW:
  In Part B we train a tiny character-level language model on a small FAQ
  dataset. The model learns which characters tend to follow other characters
  in the training text, then generates new text character by character.

HOW TO RUN:
  python project_02_custom_chatbot.py

  No external files needed. Everything is self-contained.

=============================================================================
"""

# ===========================================================================
# IMPORTS
# ===========================================================================

import random          # Built-in module for random number generation
import string          # Built-in module with string constants (letters, etc.)
import math            # Built-in module for math functions like exp()

# We try to import PyTorch. If it is not installed, we fall back to a
# pure-numpy (then pure-math) implementation. This is called "graceful
# degradation" -- the program keeps working even when a library is missing.
#
# C# analogy: This is like a try/catch around Assembly.Load(), then
# falling back to a simpler implementation.

try:
    import torch                          # Main PyTorch module
    import torch.nn as nn                 # Neural network building blocks
    import torch.optim as optim           # Optimizers (Adam, SGD, etc.)
    PYTORCH_AVAILABLE = True              # Flag: PyTorch loaded successfully
    print("[INFO] PyTorch found. Part B will use the PyTorch model.")
except ImportError:                       # Runs if PyTorch is NOT installed
    PYTORCH_AVAILABLE = False             # Flag: fall back to numpy/math
    print("[INFO] PyTorch not found. Part B will use the numpy fallback.")

# Try numpy as a secondary option for the fallback
try:
    import numpy as np                    # Numerical computing library
    NUMPY_AVAILABLE = True                # Flag: numpy loaded successfully
except ImportError:
    NUMPY_AVAILABLE = False               # Flag: numpy not installed either

# ===========================================================================
# SHARED CONSTANTS
# ===========================================================================

# This is the FAQ dataset used in Part B as training data.
# It is a plain string -- no file needed.
# Topic: Python programming basics (fitting for our student!)
#
# C# analogy: This is like a const string in C#.

FAQ_TEXT = """
Q: What is Python?
A: Python is a programming language that is easy to read and write.
Q: What is a variable?
A: A variable stores a value, like a named box that holds data.
Q: What is a function?
A: A function is a reusable block of code that does a specific job.
Q: What is a list?
A: A list stores many values in order, like an array in C#.
Q: What is a loop?
A: A loop repeats a block of code many times without writing it over and over.
Q: What is a class?
A: A class is a blueprint for creating objects, just like in C#.
Q: What is a dictionary?
A: A dictionary maps keys to values, like a Dictionary<K,V> in C#.
Q: What is indentation?
A: Indentation defines code blocks in Python, replacing curly braces in C#.
"""

# ===========================================================================
# SEPARATOR HELPER
# ===========================================================================

def print_separator(title=""):
    # Print a visual line separator, optionally with a centered title.
    # We use "=" repeated 60 times -- plain ASCII, cp1252-safe.
    # C# analogy: Console.WriteLine(new string('=', 60));
    print()                                      # Blank line before
    print("=" * 60)                              # 60 equals signs
    if title:                                    # If a title was provided
        print("  " + title)                      # Print it indented
        print("=" * 60)                          # Another separator line
    print()                                      # Blank line after


# ===========================================================================
# PART A: RULE-BASED CHATBOT
# ===========================================================================

print_separator("PART A: Rule-Based Chatbot")

# ---- A1. Build the Knowledge Base (a dictionary of keyword -> response) ----

# A Python dictionary maps KEYS to VALUES.
# Here the key is a keyword we look for in the user's message.
# The value is the response we send back.
#
# C# analogy: var responses = new Dictionary<string, string>();

RESPONSES = {
    # keyword (string)   :   response (string)
    "python"             : "Python is a high-level, easy-to-read programming language.",
    "variable"           : "A variable stores a value. In Python: x = 10 (no type needed!).",
    "function"           : "A function groups reusable code. In Python: def my_func(): ...",
    "list"               : "A list holds many values: my_list = [1, 2, 3]. Like a C# array.",
    "loop"               : "A loop repeats code. Python has 'for' and 'while' loops.",
    "class"              : "A class is a blueprint for objects. Python OOP is similar to C#.",
    "dictionary"         : "A dict maps keys to values: {'name': 'Alice'}. Like C# Dictionary.",
    "indentation"        : "Python uses indentation (spaces) to define blocks, not curly braces.",
    "hello"              : "Hello! Ask me anything about Python basics.",
    "hi"                 : "Hi there! What would you like to know about Python?",
    "help"               : "I can answer questions about: python, variable, function, list, loop, class, dictionary, indentation.",
}

# ---- A2. Default response when no keyword matches ----

DEFAULT_RESPONSE = "Sorry, I don't know that yet. Try asking about: python, variable, function, list, loop, class, dictionary, or indentation."

# ---- A3. The chatbot function ----

def rule_based_response(user_message):
    """
    Look for a known keyword in the user's message and return a response.

    Parameters
    ----------
    user_message : str
        The text typed by the user.

    Returns
    -------
    str
        The chatbot's reply.

    C# analogy:
        public string GetResponse(string userMessage) { ... }
    """

    # Convert the message to lowercase so "Python" and "python" both match.
    # C# analogy: userMessage.ToLower()
    lowered = user_message.lower()

    # Loop through every key in our RESPONSES dictionary.
    # C# analogy: foreach (var key in responses.Keys)
    for keyword in RESPONSES:           # keyword = each key in the dict
        if keyword in lowered:          # Check if this keyword is in the message
            return RESPONSES[keyword]   # Return the matching response

    # If no keyword matched, return the default response.
    return DEFAULT_RESPONSE

# ---- A4. Demonstrate a few conversation turns ----

print("Rule-Based Chatbot Demo")
print("-" * 40)                         # 40 dashes for a sub-separator

# Hard-coded sample questions (simulating user input)
# C# analogy: string[] sampleQuestions = { ... };
sample_questions = [
    "What is Python?",
    "Tell me about a variable",
    "How does a function work?",
    "What is a list?",
    "I want to know about loops",
    "What is a class?",
    "Explain dictionary please",
    "What about indentation?",
    "What is the meaning of life?",    # This one has no keyword -- tests default
]

# Loop through each sample question and print the bot's response.
# C# analogy: foreach (string question in sampleQuestions)
for question in sample_questions:       # question = one question at a time
    response = rule_based_response(question)   # Get the chatbot's reply
    print("You : " + question)                 # Print what "you" said
    print("Bot : " + response)                 # Print the bot's reply
    print("-" * 40)                            # Visual separator between turns

print()
print("[Part A Complete]")
print("Notice: the bot can ONLY answer questions that contain known keywords.")
print("It has no understanding -- just keyword matching.")


# ===========================================================================
# PART B: ML-POWERED CHATBOT (Character-Level Language Model)
# ===========================================================================

print_separator("PART B: ML-Powered Chatbot (Character-Level Model)")

print("We will now train a tiny character-level model on the FAQ dataset.")
print("The model learns which characters tend to follow other characters,")
print("then generates new text character by character.")
print()

# ---- B1. Prepare the training data ----

# Use the FAQ_TEXT defined at the top of this file.
# We treat the entire string as our "corpus" (body of training text).
training_text = FAQ_TEXT.strip()        # Remove leading/trailing whitespace

# Build the vocabulary: the set of unique characters in the training text.
# Python's set() removes duplicates. sorted() gives a stable order.
# C# analogy: new SortedSet<char>(trainingText.ToCharArray())
vocab = sorted(set(training_text))      # List of unique characters, sorted

vocab_size = len(vocab)                 # How many unique characters we have

print("Training text length : " + str(len(training_text)) + " characters")
print("Vocabulary size      : " + str(vocab_size) + " unique characters")
print()

# Build lookup tables: character -> integer index, and integer -> character.
# We need integers because neural networks work with numbers, not letters.
#
# C# analogy:
#   var charToIdx = new Dictionary<char, int>();
#   var idxToChar = new Dictionary<int, char>();

char_to_idx = {ch: i for i, ch in enumerate(vocab)}   # char -> index
idx_to_char = {i: ch for i, ch in enumerate(vocab)}   # index -> char

# Convert the entire training text into a list of integer indices.
# C# analogy: trainingText.Select(c => charToIdx[c]).ToList()
data_indices = [char_to_idx[ch] for ch in training_text]   # List of ints

# ---- B2. Hyperparameters ----
# Hyperparameters are settings WE choose before training. Not learned.
# C# analogy: const int CONTEXT_SIZE = 10;

CONTEXT_SIZE  = 10     # How many previous characters we look at to predict the next one
HIDDEN_SIZE   = 64     # Number of neurons in the hidden layer of our network
EMBED_DIM     = 16     # Size of each character's embedding vector
LEARNING_RATE = 0.01   # How big each weight update step is (too big = unstable)
EPOCHS        = 300    # How many full passes over the training data
GENERATE_LEN  = 100    # How many characters to generate when we test the model
TEMPERATURE   = 0.8    # Controls randomness: lower = more predictable output

print("Hyperparameters:")
print("  Context size  : " + str(CONTEXT_SIZE))
print("  Hidden size   : " + str(HIDDEN_SIZE))
print("  Embed dim     : " + str(EMBED_DIM))
print("  Learning rate : " + str(LEARNING_RATE))
print("  Epochs        : " + str(EPOCHS))
print("  Generate len  : " + str(GENERATE_LEN))
print("  Temperature   : " + str(TEMPERATURE))
print()


# ===========================================================================
# BRANCH: PyTorch version vs Fallback version
# ===========================================================================

if PYTORCH_AVAILABLE:
    # -----------------------------------------------------------------------
    # PART B -- PyTorch Implementation
    # -----------------------------------------------------------------------

    print("Using PyTorch implementation.")
    print("-" * 40)

    # ---- B3 (PyTorch). Build the model ----
    #
    # Our model has three layers:
    #   1. Embedding layer : turns each character index into a vector of floats
    #   2. Hidden layer    : a linear layer that processes the embedded context
    #   3. Output layer    : produces a score for each character in the vocab
    #
    # C# analogy: This is like a class with a Forward() method that chains
    # matrix multiplications together.

    class TinyCharModel(nn.Module):
        """
        A tiny character-level language model.

        Architecture:
          Embedding -> Flatten -> Linear -> ReLU -> Linear -> LogSoftmax

        C# analogy: a class that inherits from a base NeuralNetwork class
        and overrides the Forward() method.
        """

        def __init__(self, vocab_size, embed_dim, context_size, hidden_size):
            """
            Constructor. Sets up the layers.

            C# analogy: public TinyCharModel(int vocabSize, ...) { ... }
            """
            super().__init__()   # Always call the parent constructor in PyTorch

            # Embedding layer: maps each character index to a vector.
            # Input:  integer in range [0, vocab_size)
            # Output: tensor of shape (embed_dim,)
            self.embedding = nn.Embedding(vocab_size, embed_dim)

            # Hidden (fully-connected) layer.
            # Input size  = context_size * embed_dim  (we flatten all context embeddings)
            # Output size = hidden_size
            self.hidden = nn.Linear(context_size * embed_dim, hidden_size)

            # Output layer. Maps hidden layer to one score per vocab character.
            self.output = nn.Linear(hidden_size, vocab_size)

            # ReLU activation: sets negative values to 0. Adds non-linearity.
            self.relu = nn.ReLU()

        def forward(self, x):
            """
            Forward pass: given a context tensor x, predict the next character.

            Parameters
            ----------
            x : torch.Tensor of shape (batch, context_size)
                Integer indices of the context characters.

            Returns
            -------
            torch.Tensor of shape (batch, vocab_size)
                Log-probabilities for each character being next.
            """
            emb = self.embedding(x)              # Shape: (batch, context, embed_dim)
            emb = emb.view(emb.size(0), -1)      # Flatten: (batch, context * embed_dim)
            hidden = self.relu(self.hidden(emb)) # Apply hidden layer + ReLU
            out = self.output(hidden)            # Apply output layer
            return out                           # Raw scores (logits)

    # Create the model instance.
    # C# analogy: var model = new TinyCharModel(...);
    model = TinyCharModel(vocab_size, EMBED_DIM, CONTEXT_SIZE, HIDDEN_SIZE)

    # Create the optimizer. Adam adjusts learning rates automatically.
    # C# analogy: var optimizer = new AdamOptimizer(model.Parameters, LEARNING_RATE);
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loss function: CrossEntropy measures how wrong our predictions are.
    # Lower = better. We minimize this during training.
    criterion = nn.CrossEntropyLoss()

    # ---- B4 (PyTorch). Build training examples ----
    #
    # Each training example is:
    #   - input  : CONTEXT_SIZE consecutive character indices
    #   - target : the character index that comes RIGHT AFTER the context
    #
    # Example (context_size=3): "hel" -> 'l'

    X_list = []    # List of input contexts (each is a list of CONTEXT_SIZE ints)
    y_list = []    # List of target character indices

    # Slide a window of size CONTEXT_SIZE through the data.
    # i is the start of each window.
    for i in range(len(data_indices) - CONTEXT_SIZE):
        context = data_indices[i : i + CONTEXT_SIZE]   # CONTEXT_SIZE chars
        target  = data_indices[i + CONTEXT_SIZE]       # The next char
        X_list.append(context)                         # Add to inputs
        y_list.append(target)                          # Add to targets

    # Convert Python lists to PyTorch tensors (the format PyTorch needs).
    # C# analogy: converting List<int[]> to a 2D matrix type
    X_tensor = torch.tensor(X_list, dtype=torch.long)   # Shape: (N, CONTEXT_SIZE)
    y_tensor = torch.tensor(y_list, dtype=torch.long)   # Shape: (N,)

    print("Training examples : " + str(len(X_list)))
    print()

    # ---- B5 (PyTorch). Training loop ----

    print("Training the model ...")
    print("-" * 40)

    for epoch in range(EPOCHS):                    # Loop EPOCHS times
        model.train()                              # Put model in training mode

        optimizer.zero_grad()                      # Clear old gradients
        predictions = model(X_tensor)              # Forward pass
        loss = criterion(predictions, y_tensor)    # Compute loss
        loss.backward()                            # Backpropagation
        optimizer.step()                           # Update weights

        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            print("  Epoch " + str(epoch + 1).rjust(3) +
                  " / " + str(EPOCHS) +
                  "  |  Loss: " + str(round(loss.item(), 4)))

    print()
    print("Training complete!")
    print()

    # ---- B6 (PyTorch). Text generation ----

    def generate_text_pytorch(model, seed_text, length, temperature):
        """
        Generate new text character-by-character using the trained model.

        Parameters
        ----------
        model       : TinyCharModel  -- the trained model
        seed_text   : str            -- starting text to prime the generator
        length      : int            -- how many new characters to generate
        temperature : float          -- randomness control (lower = safer)

        Returns
        -------
        str  -- the seed_text followed by the generated characters
        """
        model.eval()                       # Put model in evaluation mode (no gradients)

        # Pad or truncate the seed so it is exactly CONTEXT_SIZE characters long.
        # If seed is shorter, prepend spaces. If longer, take the last CONTEXT_SIZE chars.
        if len(seed_text) < CONTEXT_SIZE:
            seed_text = seed_text.rjust(CONTEXT_SIZE)   # Right-justify, pad with spaces
        else:
            seed_text = seed_text[-CONTEXT_SIZE:]        # Take last CONTEXT_SIZE chars

        # Convert the seed characters to integer indices.
        context = [char_to_idx.get(ch, 0) for ch in seed_text]   # List of ints

        generated = seed_text    # Start with the seed text

        with torch.no_grad():                        # No gradient tracking during inference
            for _ in range(length):                  # Generate 'length' characters
                x = torch.tensor([context], dtype=torch.long)   # Shape: (1, CONTEXT_SIZE)
                logits = model(x)                    # Get raw scores; shape: (1, vocab_size)
                logits = logits[0]                   # Remove batch dimension; shape: (vocab_size,)

                # Apply temperature: divide logits to control randomness.
                # Lower temperature -> more confident (less random) choices.
                logits = logits / temperature

                # Convert logits to probabilities using softmax.
                # exp(x) / sum(exp(x)) for each element.
                probs = torch.softmax(logits, dim=0)   # Shape: (vocab_size,)

                # Sample the next character index from the probability distribution.
                # torch.multinomial picks one index weighted by probabilities.
                next_idx = torch.multinomial(probs, 1).item()   # A single integer

                # Convert index back to character.
                next_char = idx_to_char[next_idx]

                generated += next_char               # Append to output string

                # Slide the context window: drop oldest char, add new one.
                context = context[1:] + [next_idx]

        return generated    # Return the full generated string

    # Generate some sample output
    print("Generated text (seeded with 'Q: '):")
    print("-" * 40)
    generated = generate_text_pytorch(model, "Q: ", GENERATE_LEN, TEMPERATURE)
    print(generated)
    print()

    # ---- B7 (PyTorch). Simple chatbot interface ----

    def ml_chatbot_pytorch(user_prompt, length=80):
        """
        Use the ML model to generate a response to a user prompt.

        Note: this model is TINY and trained on very little data.
        Its responses will be imperfect, but this demonstrates the concept.

        Parameters
        ----------
        user_prompt : str  -- the user's question/input
        length      : int  -- how many characters to generate

        Returns
        -------
        str  -- the model's generated continuation
        """
        # Use the first CONTEXT_SIZE chars of the prompt as the seed.
        seed = user_prompt[:CONTEXT_SIZE] if len(user_prompt) >= CONTEXT_SIZE else user_prompt
        return generate_text_pytorch(model, seed, length, TEMPERATURE)

    print("ML Chatbot Demo (PyTorch)")
    print("-" * 40)
    print("Note: responses are generated character-by-character from patterns")
    print("learned in the training text. Quality is limited by model size.")
    print()

    # A few demo prompts
    demo_prompts = [
        "Q: What is",
        "A: Python",
        "Q: What is a",
    ]

    for prompt in demo_prompts:           # Loop through each demo prompt
        response = ml_chatbot_pytorch(prompt)    # Generate a response
        print("Prompt   : " + prompt)            # Show the prompt
        print("Response : " + response)          # Show the generated output
        print("-" * 40)                          # Visual separator

# -----------------------------------------------------------------------
# FALLBACK: No PyTorch -- use pure Python/math
# -----------------------------------------------------------------------
else:
    print("Using pure Python / math fallback (no PyTorch).")
    print("-" * 40)

    # ---- B3 (Fallback). Build character bigram frequency table ----
    #
    # A bigram is a pair of characters that appear next to each other.
    # We count how often each character follows each other character.
    # Then we use these counts as probabilities.
    #
    # This is simpler than a neural network but still "learns" from data.
    # C# analogy: Dictionary<char, Dictionary<char, int>> bigrams;

    # bigram_counts[i][j] = how many times character j follows character i
    # We use a list of lists (2D array).
    bigram_counts = []                             # Outer list
    for i in range(vocab_size):                    # One row per character
        bigram_counts.append([0] * vocab_size)     # Each row: count for each next-char

    # Fill in the counts by scanning the training data.
    for i in range(len(data_indices) - 1):         # Stop one before the end
        current_char_idx = data_indices[i]          # Index of current character
        next_char_idx    = data_indices[i + 1]      # Index of next character
        bigram_counts[current_char_idx][next_char_idx] += 1   # Increment count

    # Convert raw counts to probabilities (divide each row by its row total).
    # C# analogy: Normalize each row so it sums to 1.0
    bigram_probs = []                              # Will hold float probabilities
    for row in bigram_counts:                      # For each character's row
        total = sum(row)                           # Total count for this character
        if total == 0:                             # If this char never appeared
            probs = [1.0 / vocab_size] * vocab_size    # Uniform distribution
        else:
            probs = [count / total for count in row]   # Normalize to probabilities
        bigram_probs.append(probs)                 # Add to probability table

    print("Bigram table built from " + str(len(data_indices)) + " characters.")
    print()

    # ---- B4 (Fallback). Text generation using the bigram table ----

    def sample_index(probs, temperature):
        """
        Sample one index from a probability list, applying temperature.

        Parameters
        ----------
        probs       : list of float  -- probabilities for each index
        temperature : float          -- controls randomness

        Returns
        -------
        int  -- the chosen index
        """
        # Apply temperature: raise each prob to power (1/temperature).
        # Lower temperature -> sharper distribution -> less random.
        adjusted = [p ** (1.0 / temperature) for p in probs]   # Adjust probs
        total    = sum(adjusted)                                 # Renormalize
        adjusted = [p / total for p in adjusted]                # Normalize to sum=1

        # Cumulative sum to turn probabilities into ranges [0, 1].
        rand_val = random.random()     # Random float in [0, 1)
        cumulative = 0.0               # Running total
        for i, prob in enumerate(adjusted):    # Loop through each probability
            cumulative += prob                 # Add to running total
            if rand_val < cumulative:          # If we've passed the random value
                return i                       # Return this index
        return len(probs) - 1                  # Safety: return last index

    def generate_text_fallback(seed_text, length, temperature):
        """
        Generate new text using the bigram probability table.

        Parameters
        ----------
        seed_text   : str    -- starting character(s)
        length      : int    -- how many characters to generate
        temperature : float  -- randomness control

        Returns
        -------
        str  -- generated text
        """
        # Use the last character of the seed as our starting point.
        if seed_text and seed_text[-1] in char_to_idx:    # If last char is known
            current_idx = char_to_idx[seed_text[-1]]      # Get its index
        else:
            current_idx = random.randint(0, vocab_size - 1)  # Random start

        generated = seed_text    # Start with the seed text

        for _ in range(length):                            # Generate 'length' chars
            row_probs = bigram_probs[current_idx]          # Probabilities for next char
            next_idx  = sample_index(row_probs, temperature)   # Sample next index
            next_char = idx_to_char[next_idx]              # Convert to character
            generated += next_char                         # Append to output
            current_idx = next_idx                         # Advance to next character

        return generated    # Return full generated string

    # Generate sample output
    print("Generated text (seeded with 'Q: '):")
    print("-" * 40)
    generated = generate_text_fallback("Q: ", GENERATE_LEN, TEMPERATURE)
    print(generated)
    print()

    # ---- B5 (Fallback). Simple chatbot interface ----

    def ml_chatbot_fallback(user_prompt, length=80):
        """
        Use the bigram model to generate a response to a user prompt.

        Parameters
        ----------
        user_prompt : str  -- the user's input
        length      : int  -- how many characters to generate

        Returns
        -------
        str  -- the generated continuation
        """
        return generate_text_fallback(user_prompt, length, TEMPERATURE)

    print("ML Chatbot Demo (Bigram Fallback)")
    print("-" * 40)
    print("Note: the bigram model is very simple -- it only looks at ONE")
    print("previous character to predict the next. Coherence will be limited.")
    print()

    demo_prompts = [
        "Q: What is",
        "A: Python",
        "Q: What is a",
    ]

    for prompt in demo_prompts:               # Loop through demo prompts
        response = ml_chatbot_fallback(prompt)     # Generate a response
        print("Prompt   : " + prompt)              # Show the prompt
        print("Response : " + response)            # Show the generated output
        print("-" * 40)                            # Visual separator


# ===========================================================================
# COMPARISON SUMMARY
# ===========================================================================

print_separator("Summary: Rule-Based vs ML Chatbot")

print("RULE-BASED CHATBOT:")
print("  + Easy to understand and debug")
print("  + Always gives predictable answers")
print("  + No training data needed")
print("  - Can only answer questions you programmed in advance")
print("  - Cannot handle unexpected phrasing")
print("  - Does not improve over time")
print()
print("ML-POWERED CHATBOT:")
print("  + Learns patterns from data")
print("  + Can generalize (handle unseen inputs)")
print("  + Improves with more data and training")
print("  - Needs training data and compute time")
print("  - Harder to debug (black box behaviour)")
print("  - Can produce nonsense (especially when tiny like ours)")
print()
print("REAL-WORLD CHATBOTS (like ChatGPT):")
print("  -> Trained on billions of characters / tokens")
print("  -> Use Transformer architecture (not just 1-layer networks)")
print("  -> Use Reinforcement Learning from Human Feedback (RLHF)")
print("  -> Still build on the same core ideas you just learned here!")


# ===========================================================================
# QUIZ QUESTIONS
# ===========================================================================

print_separator("Quiz Questions")

print("Answer the following questions to test your understanding.")
print("Answers are provided in the comments at the bottom of this file.")
print()
print("Q1: What is the main difference between a rule-based chatbot")
print("    and an ML-powered chatbot?")
print()
print("Q2: In the ML model, what does 'temperature' control?")
print("    (a) How fast the model trains")
print("    (b) How random the generated output is")
print("    (c) The size of the vocabulary")
print("    (d) The number of training epochs")
print()
print("Q3: What is a 'bigram' model?")
print("    (a) A model that uses two hidden layers")
print("    (b) A model that looks at two characters at a time to predict the next one")
print("    (c) A model trained on two datasets")
print("    (d) A model with two outputs")
print()
print("Q4: In Python, what does 'try/except ImportError' do?")
print("    (Compare to C# to help your answer.)")
print()
print("Q5: Why do we convert characters to integer indices before feeding")
print("    them into a neural network?")

# ===========================================================================
# QUIZ ANSWERS (in comments so student must think first!)
# ===========================================================================

# ---------------------------------------------------------------------------
# ANSWER 1:
#   A rule-based chatbot follows hand-written IF/ELSE rules (keyword matching).
#   It cannot learn or improve. An ML chatbot LEARNS patterns from data and can
#   generalize to new inputs it has never seen before.
#
# ANSWER 2:
#   (b) Temperature controls how random (creative) the generated output is.
#   Low temperature -> the model picks the most likely characters -> safe, repetitive.
#   High temperature -> the model picks less likely characters -> creative, but may be nonsense.
#
# ANSWER 3:
#   (b) A bigram model looks at ONE previous character and predicts the NEXT
#   character based on how often that pair appeared in the training data.
#   "Bi" means two -- two characters at a time.
#
# ANSWER 4:
#   try/except ImportError in Python is like try/catch (TypeLoadException) in C#.
#   It tries to load a library (PyTorch in our case). If the library is not
#   installed, the ImportError is caught and we fall back to a simpler approach
#   instead of crashing. This is called "graceful degradation."
#
# ANSWER 5:
#   Neural networks are mathematical functions. They can only do math on NUMBERS,
#   not letters. By converting each character to an integer index, we give the
#   network something it can multiply, add, and compute gradients on.
#   The Embedding layer then turns that integer into a vector of floats.
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  Project 2 complete. Well done!")
print("=" * 60)
print()

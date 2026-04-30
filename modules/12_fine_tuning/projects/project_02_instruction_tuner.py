# =============================================================================
# MODULE 12 - PROJECT 02: INSTRUCTION-RESPONSE FINE-TUNER
# =============================================================================
# Title   : Instruction-Following Fine-Tuner (Alpaca Format)
# Goal    : Teach a tiny character-level language model to follow instructions
#           by fine-tuning it to classify support tickets using the Alpaca
#           prompt template format.
# What you
# will    : 1. Build a dataset of 25 examples in Alpaca instruction format
# build   : 2. Write a tokenizer and a loss-masking helper
#           3. Build a tiny NumPy-only language model
#           4. Train it with loss masking so it learns only the RESPONSE part
#           5. Run inference: give an instruction + ticket, get a class label
#           6. Show a Before vs After comparison table
#           7. See what happens when you use a different template at inference
# How to  :   python project_02_instruction_tuner.py
# run     :
# Dependencies: Python 3.10+ and NumPy only. No PyTorch, no APIs.
# =============================================================================

# =============================================================================
# GLOSSARY
# (Read every definition below before looking at the code.)
# =============================================================================
#
# Instruction Tuning
#   A type of fine-tuning where the model is trained on pairs of
#   (instruction, response) rather than plain text.
#   Instead of "predict the next word in a novel", it learns to
#   "follow a task description and produce the correct answer".
#   C# analogy: teaching a method to implement a given interface contract.
#
# Alpaca Format
#   A specific prompt template made popular by Stanford's Alpaca project.
#   Every example follows the same three-section structure:
#       ### Instruction:  <what to do>
#       ### Input:        <the data to act on>
#       ### Response:     <the correct answer>
#   Consistency is critical -- if you change the template between training
#   and inference the model will produce garbage.
#   C# analogy: a strongly-typed request/response DTO structure.
#
# Prompt Template
#   The fixed text that wraps your data so the model knows where each part
#   begins and ends. The model learns the MEANING of these markers.
#   C# analogy: a string.Format("{0}...{1}...{2}") wrapper.
#
# Instruction / Input / Output (the three Alpaca fields)
#   instruction : What task the model must perform (e.g. "Classify this ticket").
#   input       : The raw data the instruction applies to (the ticket text).
#   output      : The correct response the model should produce (e.g. "BUG").
#
# Response Token
#   The marker "### Response:" in the template. Everything AFTER this marker
#   is what the model must learn to generate. Everything BEFORE it is context.
#
# Loss Masking
#   During training, we only compute the error (loss) on the RESPONSE tokens.
#   The instruction and input tokens are "masked out" -- their loss is set to 0.
#   Why? Because we don't want to punish the model for not memorising the
#   instruction wording; we only care that it produces the right answer.
#   C# analogy: selectively applying validation rules only to certain fields
#   in a form, ignoring read-only fields.
#
# Template Consistency
#   The template used at inference time MUST be byte-for-byte identical to the
#   one used during training. Even a single extra space breaks things.
#   C# analogy: a serialization contract -- both sides must use the same schema.
#
# =============================================================================
# ASCII DIAGRAM -- THE ALPACA TEMPLATE
# =============================================================================
#
#  Full training example (output is INCLUDED):
#  +----------------------------------------------------------+
#  |  ### Instruction:                                        |
#  |  Classify the support ticket into one of:                |
#  |  BUG, FEATURE, HOW_TO, OUTAGE                           |
#  |                                                          |
#  |  ### Input:                                              |
#  |  App crashes when I upload a file larger than 10 MB.    |
#  |                                                          |
#  |  ### Response:                                           |  <-- marker
#  |  BUG                                                     |  <-- only these
#  +----------------------------------------------------------+     chars get loss
#
#  Inference example (output is EMPTY, model must generate it):
#  +----------------------------------------------------------+
#  |  ### Instruction:                                        |
#  |  Classify the support ticket into one of:                |
#  |  BUG, FEATURE, HOW_TO, OUTAGE                           |
#  |                                                          |
#  |  ### Input:                                              |
#  |  App crashes when I upload a file larger than 10 MB.    |
#  |                                                          |
#  |  ### Response:                                           |
#  |  ???  <-- model generates from here                      |
#  +----------------------------------------------------------+
#
# =============================================================================

import numpy as np     # NumPy: numerical arrays; like float[] / int[] in C#
import json            # json: read and write JSON; like System.Text.Json in C#
import random          # random: shuffle data; like System.Random in C#

# Fix the random seed so results are the same every run.
# C# analogy: new Random(42) -- deterministic seed.
np.random.seed(42)
random.seed(42)

# =============================================================================
# PART 1: ALPACA DATASET
# =============================================================================
# We create 25 support ticket examples.
# Each example is a Python dict with three keys: instruction, input, output.
# C# analogy: List<AlpacaExample> where AlpacaExample is a record with three
#             string properties.
# =============================================================================

print("=" * 70)
print("PART 1: ALPACA DATASET")
print("=" * 70)

# The shared instruction string -- every example uses EXACTLY this text.
# Changing even one character here would break template consistency.
INSTRUCTION_TEXT = (
    "Classify the support ticket into one of: BUG, FEATURE, HOW_TO, OUTAGE"
)

# 25 labelled support tickets in Alpaca format.
# C# analogy: var dataset = new List<AlpacaExample> { ... };
DATASET = [
    # ----- BUG examples (6) -----
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "App crashes when I upload a file larger than 10 MB.",
        "output": "BUG",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Login button does nothing after entering correct credentials.",
        "output": "BUG",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Report export produces a blank PDF every time.",
        "output": "BUG",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Dashboard shows negative values in the revenue chart.",
        "output": "BUG",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Error 500 appears when saving user profile changes.",
        "output": "BUG",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Search results page crashes on the second page of results.",
        "output": "BUG",
    },
    # ----- FEATURE examples (6) -----
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Please add dark mode to the dashboard.",
        "output": "FEATURE",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Can you add CSV export for all data tables?",
        "output": "FEATURE",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Would love a keyboard shortcut to submit forms.",
        "output": "FEATURE",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Please support two-factor authentication login.",
        "output": "FEATURE",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Add the ability to schedule reports to run weekly.",
        "output": "FEATURE",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Allow bulk delete on the user management page.",
        "output": "FEATURE",
    },
    # ----- HOW_TO examples (7) -----
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "How do I reset my password?",
        "output": "HOW_TO",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Where can I find the API documentation?",
        "output": "HOW_TO",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "How do I add a new team member to my account?",
        "output": "HOW_TO",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Can you explain how to set up webhooks?",
        "output": "HOW_TO",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "How do I change the default currency in billing?",
        "output": "HOW_TO",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "What steps are needed to migrate data from the old system?",
        "output": "HOW_TO",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "How do I enable email notifications for new invoices?",
        "output": "HOW_TO",
    },
    # ----- OUTAGE examples (6) -----
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Your entire platform is down. Nobody can log in right now.",
        "output": "OUTAGE",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "All users in our office cannot reach the service since 9am.",
        "output": "OUTAGE",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Website is completely unreachable for all our customers.",
        "output": "OUTAGE",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "Entire API is returning 503 for every endpoint since an hour.",
        "output": "OUTAGE",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "No one on our team can access the portal. Is there an outage?",
        "output": "OUTAGE",
    },
    {
        "instruction": INSTRUCTION_TEXT,
        "input": "System has been completely unavailable for the last 30 minutes.",
        "output": "OUTAGE",
    },
]

# Print a few examples so you can see the structure.
print("Sample dataset entries (first 3 shown):")
for i, ex in enumerate(DATASET[:3]):           # loop over first 3 examples
    print(f"  Example {i+1}:")                  # show index number
    print(f"    instruction : {ex['instruction'][:50]}...")  # trim for display
    print(f"    input       : {ex['input']}")   # full ticket text
    print(f"    output      : {ex['output']}")  # label: BUG/FEATURE/HOW_TO/OUTAGE
print()

# -------------------------------------------------------------------------
# How to save dataset as JSONL (JSON Lines format)
# JSONL = one JSON object per line. Common format for fine-tuning datasets.
# C# analogy: each line is like JsonSerializer.Serialize(item) + newline.
# We only PRINT the lines here; we do not write to disk.
# -------------------------------------------------------------------------
print("What the JSONL file would look like (first 2 lines):")
for ex in DATASET[:2]:                         # loop over first 2 examples
    print(json.dumps(ex))                       # serialize dict -> JSON string
print()

# -------------------------------------------------------------------------
# apply_template() -- converts an Alpaca dict into the full prompt string.
# C# analogy: string.Format() or an interpolated string template.
#
# Parameters:
#   example      : dict with keys instruction, input, output
#   include_output: if True, append the answer (for training)
#                   if False, leave response blank (for inference)
# -------------------------------------------------------------------------
def apply_template(example, include_output=True):
    # Build the "prefix" -- everything up to and including "### Response:\n"
    prefix = (
        "### Instruction:\n"        # section header for the task description
        + example["instruction"]    # the task text
        + "\n\n"                    # blank line separator (MUST stay consistent)
        + "### Input:\n"            # section header for the data
        + example["input"]          # the ticket text
        + "\n\n"                    # blank line separator
        + "### Response:\n"         # section header for the answer -- key marker
    )
    if include_output:              # training mode: add the correct answer
        return prefix + example["output"]   # e.g. "...### Response:\nBUG"
    else:                           # inference mode: leave blank, model fills in
        return prefix               # e.g. "...### Response:\n"

# Show a rendered template for the first example.
print("Full rendered template (training, include_output=True):")
print("-" * 50)
print(apply_template(DATASET[0], include_output=True))   # full training text
print("-" * 50)
print()
print("Inference template (include_output=False):")
print("-" * 50)
print(apply_template(DATASET[0], include_output=False))  # prompt only
print("-" * 50)
print()

# =============================================================================
# PART 2: TOKENIZATION (CHARACTER-LEVEL)
# =============================================================================
# We use a character-level tokenizer: each character is one token.
# This is simpler than word-level and needs no external library.
# C# analogy: converting a string to char[], then mapping each char to an int.
#
# Why character-level?
#   - No vocabulary file needed
#   - Works with any text
#   - Easy to understand
#   - Downside: long sequences (we keep examples short)
# =============================================================================

print("=" * 70)
print("PART 2: TOKENIZATION")
print("=" * 70)

class SimpleTokenizer:
    """
    Character-level tokenizer.
    Builds a vocabulary from ALL unique characters seen in the training data.
    C# analogy: a class with a Dictionary<char, int> for encoding and
                Dictionary<int, char> for decoding.
    """

    def __init__(self, texts):
        # texts: a list of strings to build the vocabulary from
        # We collect all unique characters across every text.
        # C# analogy: texts.SelectMany(t => t).Distinct()
        all_chars = set()                       # a Python set = no duplicates
        for text in texts:                      # iterate over each training text
            for ch in text:                     # iterate over each character
                all_chars.add(ch)               # add char to the set

        # Sort the characters so the vocabulary is deterministic (same order
        # every run).  C# analogy: .OrderBy(c => c)
        vocab = sorted(all_chars)               # sorted list of unique chars

        # char_to_id maps each character to an integer index.
        # C# analogy: Dictionary<char, int>
        self.char_to_id = {ch: idx for idx, ch in enumerate(vocab)}

        # id_to_char maps each integer back to a character.
        # C# analogy: Dictionary<int, char>
        self.id_to_char = {idx: ch for idx, ch in enumerate(vocab)}

        # vocab_size is the total number of unique characters.
        # C# analogy: vocab.Count
        self.vocab_size = len(vocab)

    def encode(self, text):
        """
        Convert a string into a list of integer token IDs.
        C# analogy: text.Select(c => char_to_id[c]).ToList()
        """
        return [self.char_to_id[ch] for ch in text]   # list comprehension

    def decode(self, ids):
        """
        Convert a list of integer token IDs back to a string.
        C# analogy: string.Join("", ids.Select(i => id_to_char[i]))
        """
        return "".join(self.id_to_char[i] for i in ids)   # join chars


# Build the vocabulary from all training templates (output INCLUDED).
# We pass every rendered template so the tokenizer sees all characters.
all_training_texts = [
    apply_template(ex, include_output=True) for ex in DATASET
]                                               # list of 25 full template strings

tokenizer = SimpleTokenizer(all_training_texts) # build vocab

print(f"Vocabulary size: {tokenizer.vocab_size} unique characters")
print(f"First 20 chars in vocab: {list(tokenizer.char_to_id.keys())[:20]}")
print()

# -------------------------------------------------------------------------
# Loss Masking -- find where "### Response:" starts in the token sequence.
# We ONLY compute gradient/loss on the tokens AFTER this marker.
# -------------------------------------------------------------------------

# The exact response marker string -- must match the template exactly.
RESPONSE_MARKER = "### Response:\n"            # 16 characters

def find_response_start(token_ids, tokenizer):
    """
    Scan the token ID sequence and return the index of the FIRST token
    that belongs to the RESPONSE (i.e., the character right after the
    "### Response:\n" marker).

    Returns:
        int: the index in token_ids where the response starts.
             Returns len(token_ids) if the marker is not found
             (meaning: mask everything, produce no loss).

    C# analogy:
        int idx = text.IndexOf("### Response:\n") + markerLength;
        then map that character offset to token index.
    """
    marker_ids = tokenizer.encode(RESPONSE_MARKER)  # encode the marker string
    marker_len = len(marker_ids)                     # number of tokens in marker

    # Slide a window of length marker_len over token_ids looking for a match.
    # C# analogy: for (int i = 0; i <= ids.Count - markerLen; i++) { ... }
    for i in range(len(token_ids) - marker_len + 1):  # slide window
        # Check if the window at position i matches the marker.
        if token_ids[i : i + marker_len] == marker_ids:  # window == marker
            return i + marker_len               # response starts right after marker
    return len(token_ids)                       # marker not found: mask everything


# Demonstrate loss masking on the first example.
demo_text   = apply_template(DATASET[0], include_output=True)  # full string
demo_tokens = tokenizer.encode(demo_text)                      # encode to ints
resp_start  = find_response_start(demo_tokens, tokenizer)      # find split point

print("Loss masking demo (first example):")
print(f"  Full template length : {len(demo_text)} characters")
print(f"  Number of tokens     : {len(demo_tokens)}")
print(f"  Response starts at   : token index {resp_start}")
print(f"  Masked tokens (no loss): indices 0 to {resp_start - 1}  "
      f"= '{tokenizer.decode(demo_tokens[:resp_start])[-30:]}'...")
print(f"  Active tokens (get loss): indices {resp_start} to end  "
      f"= '{tokenizer.decode(demo_tokens[resp_start:])}'")
print()

# =============================================================================
# PART 3: BASE MODEL (BEFORE TRAINING)
# =============================================================================
# TinyLanguageModel is a minimal character-level language model.
#
# Architecture (very simple):
#   1. Embedding layer : maps each token ID to a dense vector (like a lookup table)
#   2. Linear layer    : projects the embedding to logits over the vocabulary
#
# This is NOT a transformer. It ignores context (no attention). It just learns
# to predict the next character purely from the CURRENT character's embedding.
# That is intentionally simple so training is fast and results are visible.
#
# C# analogy: a class with two float[,] weight matrices and a Predict() method.
# =============================================================================

print("=" * 70)
print("PART 3: BASE MODEL BEHAVIOR (BEFORE TRAINING)")
print("=" * 70)

# Hyper-parameters -- small values so training is fast.
EMBED_DIM  = 16    # size of each character embedding vector
HIDDEN_DIM = 32    # size of the hidden projection layer
VOCAB_SIZE = tokenizer.vocab_size   # total unique characters

class TinyLanguageModel:
    """
    A tiny character-level language model using NumPy.

    Layers:
      embedding : shape (vocab_size, embed_dim)
                  Each row is the embedding for one character.
                  C# analogy: float[vocabSize, embedDim] lookup table.
      W_hidden  : shape (embed_dim, hidden_dim)
                  Projects embedding to a hidden representation.
                  C# analogy: float[embedDim, hiddenDim] weight matrix.
      W_out     : shape (hidden_dim, vocab_size)
                  Projects hidden representation to vocabulary logits.
                  C# analogy: float[hiddenDim, vocabSize] weight matrix.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        # vocab_size : total unique characters in our vocabulary
        # embed_dim  : how many numbers represent each character
        # hidden_dim : intermediate layer size

        self.vocab_size = vocab_size    # store for later use
        self.embed_dim  = embed_dim     # store for later use
        self.hidden_dim = hidden_dim    # store for later use

        # Initialize embedding matrix with small random numbers.
        # np.random.randn draws from a standard normal distribution (mean=0, std=1).
        # Multiplying by 0.01 keeps values small so gradients don't explode early.
        # C# analogy: new float[vocabSize, embedDim] filled with small randoms.
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.01

        # Hidden weight matrix: transforms embedding -> hidden representation.
        self.W_hidden = np.random.randn(embed_dim, hidden_dim) * 0.01

        # Output weight matrix: transforms hidden -> vocabulary logits.
        self.W_out = np.random.randn(hidden_dim, vocab_size) * 0.01

    def forward(self, token_id):
        """
        Given ONE token ID, compute logits over the entire vocabulary.
        Returns a 1-D array of shape (vocab_size,).

        Steps:
          1. Look up the embedding for this token ID.
          2. Multiply by W_hidden to get a hidden vector.
          3. Apply ReLU activation (clamp negatives to 0).
          4. Multiply by W_out to get vocabulary logits.

        C# analogy:
          float[] emb    = embedding[tokenId, ..];
          float[] hidden = ReLU(emb @ W_hidden);
          float[] logits = hidden @ W_out;
        """
        emb    = self.embedding[token_id]          # shape: (embed_dim,)
        hidden = np.dot(emb, self.W_hidden)        # shape: (hidden_dim,)
        hidden = np.maximum(0, hidden)             # ReLU: set negatives to 0
        logits = np.dot(hidden, self.W_out)        # shape: (vocab_size,)
        return logits                              # raw scores, not probabilities

    def predict_next_char(self, token_id, temperature=1.0):
        """
        Given the current token ID, predict the most likely next character.

        temperature: controls randomness.
          1.0 = normal,  <1 = more confident,  >1 = more random.
          We use temperature=0.1 at inference for near-greedy decoding.

        Returns the predicted next character as a string.
        """
        logits = self.forward(token_id)            # get raw scores
        # Divide by temperature to scale the logits before softmax.
        logits = logits / temperature              # scaled logits
        # Softmax: convert logits to probabilities.
        # exp(logit) / sum(exp(all logits))
        # Subtract max for numerical stability (avoids overflow in exp).
        logits -= np.max(logits)                   # stability trick
        probs = np.exp(logits)                     # exponentiate
        probs /= np.sum(probs)                     # normalise to sum = 1.0
        # Sample from the probability distribution.
        # np.random.choice picks an index weighted by probs.
        # C# analogy: weighted random selection from an array.
        next_id = np.random.choice(self.vocab_size, p=probs)  # sample
        return tokenizer.id_to_char[next_id]       # convert id back to char


def generate_response(model, instruction, inp, max_tokens=10, temperature=0.1):
    """
    Run inference: feed the instruction + input prompt, then let the model
    generate characters one at a time up to max_tokens.

    Parameters:
      model       : a TinyLanguageModel instance
      instruction : the task instruction string
      inp         : the ticket text (the 'input' field)
      max_tokens  : how many characters to generate after "### Response:\n"
      temperature : controls randomness (lower = more deterministic)

    Returns:
      generated string (the model's predicted response)
    """
    # Build the inference prompt (no output included).
    prompt = apply_template(
        {"instruction": instruction, "input": inp, "output": ""},
        include_output=False,           # leave response blank
    )
    prompt_ids = tokenizer.encode(prompt)  # tokenise the prompt

    # The model only uses the LAST token as context (no attention/memory).
    # We start generation from the last token of the prompt.
    current_id = prompt_ids[-1]            # last token of the prompt

    generated = []                         # list to accumulate generated chars
    for _ in range(max_tokens):            # generate up to max_tokens chars
        next_char = model.predict_next_char(current_id, temperature)  # predict
        generated.append(next_char)        # collect the char
        # Encode the new char to get its ID for the next step.
        # If the char is not in vocab, stop early.
        if next_char not in tokenizer.char_to_id:  # unknown char guard
            break
        current_id = tokenizer.char_to_id[next_char]  # next context token
    return "".join(generated)             # join chars into string


# Instantiate the model BEFORE training.
model = TinyLanguageModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)

# Show 3 "before training" predictions -- they will be random/garbage.
print("Model predictions BEFORE training (random weights):")
before_examples = DATASET[:3]              # use first 3 examples
before_outputs  = []                       # store for comparison later
for ex in before_examples:                 # loop over 3 examples
    pred = generate_response(
        model,
        ex["instruction"],
        ex["input"],
        max_tokens=8,                      # up to 8 chars (labels are short)
        temperature=0.1,                   # low temperature = less random
    )
    before_outputs.append(pred)            # save for Part 6 comparison
    print(f"  Ticket : {ex['input'][:50]}")   # show ticket text
    print(f"  True   : {ex['output']}")        # show correct label
    print(f"  Pred   : {pred!r}")              # show model's garbage output
    print()

# =============================================================================
# PART 4: FINE-TUNING WITH LOSS MASKING
# =============================================================================
# Training loop:
#   For each epoch:
#     For each example in the dataset:
#       1. Render the full template (instruction + input + output).
#       2. Tokenise it.
#       3. Find where the response starts (loss_mask_start).
#       4. For each RESPONSE token position:
#            - Run forward pass on the PREVIOUS token.
#            - Compute cross-entropy loss between predicted and true next token.
#            - Compute gradient and update weights.
#
# We use Stochastic Gradient Descent (SGD): update weights after each token.
# C# analogy: a nested foreach with weight updates inside the inner loop.
#
# Cross-Entropy Loss (for one position):
#   loss = -log( probability of the correct token )
#   If prob=1.0 (perfect prediction): loss = 0
#   If prob=0.01 (bad prediction):    loss = ~4.6
# =============================================================================

print("=" * 70)
print("PART 4: FINE-TUNING WITH LOSS MASKING")
print("=" * 70)

LEARNING_RATE = 0.05    # how big each weight update step is; C# analogy: step size
NUM_EPOCHS    = 30      # how many full passes through the dataset


def train_one_step(model, token_ids, loss_mask_start, lr):
    """
    Train the model on ONE full tokenised example.

    Only tokens at positions >= loss_mask_start contribute to the loss.
    (Positions before that are the instruction/input -- we ignore their loss.)

    Parameters:
      model            : TinyLanguageModel to update
      token_ids        : list of int token IDs for the full template
      loss_mask_start  : index where the response begins
      lr               : learning rate

    Returns:
      float: average loss over the response tokens
    """
    total_loss = 0.0        # accumulate loss over response tokens
    n_active   = 0          # count of response tokens processed

    # Iterate over every adjacent pair (current_token -> next_token).
    # We predict next_token given current_token.
    # C# analogy: for (int i = 0; i < ids.Count - 1; i++) { ... }
    for i in range(len(token_ids) - 1):    # stop 1 before the end
        current_id = token_ids[i]           # current character token
        target_id  = token_ids[i + 1]      # the character we want to predict

        # Only compute loss if we are in the RESPONSE part.
        # i+1 because the PREDICTION for position i is the token at i+1.
        if i + 1 < loss_mask_start:        # still in instruction/input region
            continue                        # skip: no loss, no update

        # --- Forward pass ---
        emb    = model.embedding[current_id]        # (embed_dim,)
        hidden = np.dot(emb, model.W_hidden)        # (hidden_dim,)
        hidden_relu = np.maximum(0, hidden)         # ReLU activation
        logits = np.dot(hidden_relu, model.W_out)   # (vocab_size,)

        # --- Softmax to get probabilities ---
        logits -= np.max(logits)            # numerical stability
        exp_logits = np.exp(logits)         # exponentiate
        probs = exp_logits / np.sum(exp_logits)     # normalise

        # --- Cross-entropy loss on the correct token ---
        # -log(prob of the correct next token)
        correct_prob = probs[target_id]             # probability of true answer
        loss = -np.log(correct_prob + 1e-9)         # add 1e-9 to avoid log(0)
        total_loss += loss                          # accumulate
        n_active   += 1                             # count active positions

        # --- Backward pass (manual gradient computation) ---
        # Gradient of cross-entropy + softmax w.r.t. logits:
        #   d_loss/d_logit[j] = prob[j] - 1 if j==target else prob[j]
        # This is the standard closed-form result for softmax + cross-entropy.
        d_logits = probs.copy()                     # start with probabilities
        d_logits[target_id] -= 1.0                  # subtract 1 from correct class

        # Gradient w.r.t. W_out:  hidden_relu.T @ d_logits  (outer product)
        # hidden_relu shape: (hidden_dim,)
        # d_logits   shape: (vocab_size,)
        # grad_W_out shape: (hidden_dim, vocab_size)
        grad_W_out = np.outer(hidden_relu, d_logits)    # outer product

        # Gradient w.r.t. hidden (before W_out):
        # d_hidden = d_logits @ W_out.T
        d_hidden = np.dot(model.W_out, d_logits)        # (hidden_dim,)

        # Gradient through ReLU: zero out positions where hidden was <= 0.
        d_hidden_relu = d_hidden * (hidden > 0)         # element-wise mask

        # Gradient w.r.t. W_hidden: emb.T @ d_hidden_relu  (outer product)
        # emb          shape: (embed_dim,)
        # d_hidden_relu shape: (hidden_dim,)
        # grad_W_hidden shape: (embed_dim, hidden_dim)
        grad_W_hidden = np.outer(emb, d_hidden_relu)    # outer product

        # Gradient w.r.t. embedding row for current_id.
        # d_emb = d_hidden_relu @ W_hidden.T
        d_emb = np.dot(model.W_hidden, d_hidden_relu)   # (embed_dim,)

        # --- SGD weight updates: weight = weight - lr * gradient ---
        model.W_out             -= lr * grad_W_out      # update output weights
        model.W_hidden          -= lr * grad_W_hidden   # update hidden weights
        model.embedding[current_id] -= lr * d_emb       # update this token's emb

    # Return average loss; if no active tokens, return 0.
    return total_loss / max(n_active, 1)


# Main training loop.
for epoch in range(1, NUM_EPOCHS + 1):          # epochs 1..30
    epoch_loss = 0.0                            # total loss for this epoch
    random.shuffle(DATASET)                     # shuffle examples each epoch

    for ex in DATASET:                          # loop over all 25 examples
        # Render the full template with the correct output.
        full_text  = apply_template(ex, include_output=True)
        token_ids  = tokenizer.encode(full_text)        # tokenise
        resp_start = find_response_start(token_ids, tokenizer)  # find mask boundary
        # Train on this example; accumulate loss.
        loss = train_one_step(model, token_ids, resp_start, LEARNING_RATE)
        epoch_loss += loss                      # add to epoch total

    avg_loss = epoch_loss / len(DATASET)        # average over 25 examples

    # Print progress every 5 epochs.
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:3d} / {NUM_EPOCHS}  |  avg train loss: {avg_loss:.4f}")

print()

# =============================================================================
# PART 5: INFERENCE (AFTER TRAINING)
# =============================================================================
# We now use the fine-tuned model to classify 5 support tickets.
# The model has never seen the test examples below.
# =============================================================================

print("=" * 70)
print("PART 5: INFERENCE (FINE-TUNED MODEL)")
print("=" * 70)

# 5 new test examples (not in training data).
test_examples = [
    {"input": "Nothing works, whole system down for all users.",
     "true_label": "OUTAGE"},
    {"input": "How can I export my data to Excel?",
     "true_label": "HOW_TO"},
    {"input": "Please add Slack integration to notifications.",
     "true_label": "FEATURE"},
    {"input": "Clicking Save throws a null reference error.",
     "true_label": "BUG"},
    {"input": "How do I upgrade my subscription plan?",
     "true_label": "HOW_TO"},
]

def run_inference(model, instruction, inp, max_tokens=10, temperature=0.1):
    """
    Public inference function.
    Builds the inference prompt, generates up to max_tokens characters.
    Returns the generated response string.
    """
    return generate_response(model, instruction, inp, max_tokens, temperature)


print("Inference results:")
after_outputs = []                          # store for comparison
for ex in test_examples:                    # loop over test examples
    pred = run_inference(
        model,
        INSTRUCTION_TEXT,                   # same instruction as training
        ex["input"],
        max_tokens=10,                      # labels are <= 7 chars
        temperature=0.1,
    )
    after_outputs.append(pred)              # save result
    # Show input, true label, and model prediction.
    print(f"  Ticket : {ex['input']}")
    print(f"  True   : {ex['true_label']}")
    print(f"  Pred   : {pred!r}")
    print()

# =============================================================================
# PART 6: BEFORE vs AFTER COMPARISON TABLE
# =============================================================================
# Show the first 3 training examples:
#   Input | True Label | Before Training Output | After Training Output
# =============================================================================

print("=" * 70)
print("PART 6: BEFORE vs AFTER COMPARISON")
print("=" * 70)

# We already have before_outputs (from Part 3) and before_examples.
# Now compute after_outputs for those same examples.
comparison_after = []                       # re-run inference after training
for ex in before_examples:                  # same 3 examples as Part 3
    pred = run_inference(
        model,
        ex["instruction"],
        ex["input"],
        max_tokens=10,
        temperature=0.1,
    )
    comparison_after.append(pred)           # save prediction

# Print a formatted table.
# Column widths chosen to fit within 70 chars.
col1 = 30   # Ticket column width
col2 = 8    # True label column width
col3 = 8    # Before column width
col4 = 8    # After column width

# Print table header.
print(f"  {'Ticket':<{col1}} {'True':<{col2}} {'Before':<{col3}} {'After':<{col4}}")
print("  " + "-" * (col1 + col2 + col3 + col4 + 3))   # separator line

# Print one row per example.
for i, ex in enumerate(before_examples):   # loop over 3 examples
    ticket_short = ex["input"][:col1 - 3] + "..." if len(ex["input"]) > col1 - 3 else ex["input"]
    before = before_outputs[i][:col3]      # trim to column width
    after  = comparison_after[i][:col4]    # trim to column width
    true_l = ex["output"][:col2]           # trim to column width
    print(f"  {ticket_short:<{col1}} {true_l:<{col2}} {before:<{col3}} {after:<{col4}}")

print()

# =============================================================================
# PART 7: TEMPLATE CONSISTENCY WARNING
# =============================================================================
# What happens if we use a DIFFERENT template at inference time?
# We demonstrate by swapping "### Response:\n" for "### Answer:\n".
# The model was NEVER trained on this marker, so it produces garbage.
# =============================================================================

print("=" * 70)
print("PART 7: TEMPLATE CONSISTENCY WARNING")
print("=" * 70)

# Bad template: uses "### Answer:" instead of "### Response:".
def apply_bad_template(example):
    """
    An INCORRECT template -- uses the wrong section header for the response.
    Shows what happens when inference template != training template.
    """
    return (
        "### Instruction:\n"
        + example["instruction"]
        + "\n\n"
        + "### Input:\n"
        + example["input"]
        + "\n\n"
        + "### Answer:\n"            # WRONG: should be "### Response:\n"
    )

# A helper that generates from an ARBITRARY prompt string (not via apply_template).
def generate_from_prompt(model, prompt, max_tokens=10, temperature=0.1):
    """
    Generate characters from any raw prompt string.
    Used here to show the broken template scenario.
    """
    prompt_ids  = tokenizer.encode(prompt)  # tokenise
    current_id  = prompt_ids[-1]            # start from last token
    generated   = []                        # accumulate output
    for _ in range(max_tokens):             # generate up to max_tokens
        next_char = model.predict_next_char(current_id, temperature)
        generated.append(next_char)
        if next_char not in tokenizer.char_to_id:  # unknown char guard
            break
        current_id = tokenizer.char_to_id[next_char]
    return "".join(generated)

# Take one example and run it with the broken template.
demo_ex       = DATASET[0]                  # first example (BUG ticket)
correct_pred  = run_inference(              # correct template
    model, demo_ex["instruction"], demo_ex["input"], max_tokens=8, temperature=0.1
)
bad_prompt    = apply_bad_template(demo_ex)             # wrong template string
bad_pred      = generate_from_prompt(model, bad_prompt, max_tokens=8, temperature=0.1)

print("Demonstration: same ticket, two different templates")
print()
print("  [CORRECT TEMPLATE]")
print(f"  Prompt ends with : '### Response:\\n'")
print(f"  Model output     : {correct_pred!r}")
print()
print("  [WRONG TEMPLATE -- uses '### Answer:' instead of '### Response:']")
print(f"  Prompt ends with : '### Answer:\\n'")
print(f"  Model output     : {bad_pred!r}")
print()
print("  *** LESSON: Template at inference MUST match template at training. ***")
print("  *** Even one changed word causes the model to produce garbage.     ***")
print()

# Why does this happen? Explain:
print("  Why?")
print("  During training the model learned: AFTER seeing '### Response:' tokens,")
print("  predict the class label (BUG, FEATURE, ...). It NEVER saw '### Answer:'")
print("  so it has no idea what should come after that marker.")
print("  C# analogy: calling method.Invoke() on a delegate with the wrong")
print("  signature -- the runtime doesn't know what to do.")
print()

# =============================================================================
# PART 8: KEY TAKEAWAYS
# =============================================================================

print("=" * 70)
print("PART 8: KEY TAKEAWAYS")
print("=" * 70)

# Each lesson is a simple numbered sentence.
lessons = [
    "1. Instruction tuning teaches the model to FOLLOW DIRECTIONS, not just "
    "predict text. The Alpaca format (Instruction / Input / Response) is a "
    "widely used template for this.",

    "2. LOSS MASKING is critical: only penalise the model for wrong RESPONSE "
    "tokens. Masking the instruction tokens prevents the model wasting effort "
    "memorising prompts instead of learning the task.",

    "3. TEMPLATE CONSISTENCY: the prompt template used at inference must be "
    "byte-for-byte identical to the one used at training time. Any change "
    "breaks the model. Store your template in a single constant (INSTRUCTION_TEXT).",

    "4. Character-level tokenisation is simple and dependency-free, but "
    "real LLMs use subword tokenisers (BPE, SentencePiece) that balance "
    "vocabulary size and sequence length much more efficiently.",

    "5. This tiny NumPy model has NO attention, NO context window -- it only "
    "looks at one character at a time. Real transformers maintain context "
    "across the whole sequence, which is why they can follow complex "
    "multi-sentence instructions.",
]

for lesson in lessons:          # print each lesson
    print()
    # Word-wrap manually at ~65 chars for readability.
    words  = lesson.split()     # split into words
    line   = "  "               # start with indentation
    for word in words:          # build lines word by word
        if len(line) + len(word) + 1 > 68:   # would exceed line width
            print(line)         # print current line
            line = "  " + word  # start new line with this word
        else:
            line += (" " if len(line) > 2 else "") + word  # append word
    if line.strip():            # print any remaining text
        print(line)

print()
print("=" * 70)
print("END OF PROJECT 02: INSTRUCTION-RESPONSE FINE-TUNER")
print("=" * 70)

# =============================================================================
# PART B: PyTorch SKETCH (commented out)
# =============================================================================
# Below is a sketch of how this project would look in PyTorch.
# It is commented out so this file runs with NumPy only.
# Read it to understand the direction real LLM fine-tuning takes.
# C# analogy: this is like pseudo-code with actual syntax you can activate later.
# =============================================================================

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
#
#
# class AlpacaDataset(Dataset):
#     """
#     PyTorch Dataset for Alpaca-format instruction tuning.
#     C# analogy: implements IEnumerable<(Tensor input, Tensor target, int mask_start)>
#     """
#     def __init__(self, data, tokenizer, max_len=256):
#         self.samples = []
#         for ex in data:
#             full_text   = apply_template(ex, include_output=True)
#             ids         = tokenizer.encode(full_text)[:max_len]  # truncate
#             resp_start  = find_response_start(ids, tokenizer)
#             self.samples.append((ids, resp_start))
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         ids, resp_start = self.samples[idx]
#         # input_ids  = all tokens except last (we predict the next one)
#         # target_ids = all tokens except first (shifted by 1)
#         input_ids  = torch.tensor(ids[:-1], dtype=torch.long)
#         target_ids = torch.tensor(ids[1:],  dtype=torch.long)
#         return input_ids, target_ids, resp_start
#
#
# class TransformerLM(nn.Module):
#     """
#     A minimal GPT-style transformer language model.
#     C# analogy: a class with nn.Embedding, nn.TransformerEncoder, nn.Linear.
#     """
#     def __init__(self, vocab_size, embed_dim=64, n_heads=2, n_layers=2, max_len=256):
#         super().__init__()
#         self.embedding  = nn.Embedding(vocab_size, embed_dim)   # token embeddings
#         self.pos_embed  = nn.Embedding(max_len, embed_dim)      # position embeddings
#         encoder_layer   = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=n_heads, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#         self.lm_head    = nn.Linear(embed_dim, vocab_size)      # output projection
#
#     def forward(self, token_ids):
#         # token_ids: (batch, seq_len)
#         B, T       = token_ids.shape
#         positions  = torch.arange(T, device=token_ids.device)   # [0,1,...,T-1]
#         x          = self.embedding(token_ids) + self.pos_embed(positions)  # combine
#         # Causal mask: token i can only attend to tokens 0..i
#         mask       = nn.Transformer.generate_square_subsequent_mask(T)
#         x          = self.transformer(x, mask=mask)             # apply transformer
#         logits     = self.lm_head(x)                            # (B, T, vocab_size)
#         return logits
#
#
# def train_pytorch(model, dataset, epochs=3, lr=3e-4):
#     """
#     Training loop with loss masking in PyTorch.
#     C# analogy: foreach epoch { foreach batch { forward; maskedLoss; backward; step; } }
#     """
#     loader    = DataLoader(dataset, batch_size=4, shuffle=True)
#     optimiser = torch.optim.Adam(model.parameters(), lr=lr)   # Adam optimiser
#     loss_fn   = nn.CrossEntropyLoss(reduction='none')         # per-token loss
#
#     model.train()                                             # set training mode
#     for epoch in range(1, epochs + 1):
#         epoch_loss = 0.0
#         for input_ids, target_ids, resp_start in loader:
#             optimiser.zero_grad()                             # clear gradients
#             logits = model(input_ids)                         # (B, T, V) forward
#             B, T, V = logits.shape
#             # Reshape for cross-entropy: (B*T, V) vs (B*T,)
#             loss_per_token = loss_fn(
#                 logits.reshape(-1, V), target_ids.reshape(-1)
#             )                                                 # (B*T,) per-token loss
#             # Build the loss mask: 1.0 for response tokens, 0.0 for prompt tokens.
#             # resp_start is a scalar per sample; we build a (B, T) mask.
#             mask = torch.zeros(B, T)                          # start with all zeros
#             for b in range(B):                                # per-sample
#                 mask[b, resp_start[b]:] = 1.0                # 1 from response start
#             mask = mask.reshape(-1)                           # flatten to (B*T,)
#             # Apply mask: zero out loss for prompt tokens.
#             masked_loss = (loss_per_token * mask).sum() / mask.sum()  # mean over active
#             masked_loss.backward()                            # compute gradients
#             optimiser.step()                                  # update weights
#             epoch_loss += masked_loss.item()
#         print(f"Epoch {epoch} loss: {epoch_loss / len(loader):.4f}")
#
#
# # Usage:
# # pt_model   = TransformerLM(vocab_size=tokenizer.vocab_size)
# # pt_dataset = AlpacaDataset(DATASET, tokenizer)
# # train_pytorch(pt_model, pt_dataset, epochs=3)

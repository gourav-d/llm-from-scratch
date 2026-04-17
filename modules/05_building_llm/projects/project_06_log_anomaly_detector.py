"""
=============================================================================
PROJECT 6: Log Anomaly Detector - Find Unusual Logs with ML
=============================================================================

GLOSSARY (read this first, before looking at any code)
-------------------------------------------------------
Log Line             : A single text entry written by a server or application
                       that records what happened and when.
                       C# analogy: one call to ILogger.LogInformation(...).

Anomaly              : Something unusual, unexpected, or suspicious.
                       In logs, an anomaly might be an error, a security
                       breach, or a system running out of resources.

Rule-Based Detection : Flagging anomalies by checking against a hand-crafted
                       list of rules (e.g. "if line contains ERROR, flag it").
                       Fast, but you must predict every bad thing in advance.
                       C# analogy: a giant if/else if chain or switch statement.

ML-Based Detection   : Using a machine-learning model to detect anomalies
                       automatically, without writing explicit rules.
                       The model learns what "normal" looks like, then flags
                       anything that does not fit that pattern.

Loss                 : A number that measures how SURPRISED the model is by
                       a piece of text.
                       Low loss  = model expected this text (it looks normal).
                       High loss = model did NOT expect this (it looks unusual).
                       C# analogy: imagine a unit test assertion score --
                       a score of 0 means everything matched expectations,
                       a high score means something was very wrong.

Character-Level Model: A language model that reads text one character at a
                       time (like reading letter-by-letter, not word-by-word).
                       Simpler to build from scratch, no tokenizer needed.

Bigram Model         : A model that predicts the NEXT character based only on
                       the CURRENT character. "bi" = two, so it looks at pairs.
                       C# analogy: a Dictionary<char, Dictionary<char, float>>
                       that stores how often char B follows char A.

Vocabulary           : The complete set of unique characters the model knows.
                       If training data contains only 'a','b','c', those three
                       are the vocabulary.
                       C# analogy: a HashSet<char> of all known characters.

Training             : The process of updating a model's numbers (weights)
                       using training data so the model gets better predictions.
                       C# analogy: calling Fit() on a calibration engine.

Weights / Parameters : The numbers inside a model that are adjusted during
                       training. The model "learns" by changing these numbers.
                       C# analogy: private fields on a class that the optimizer
                       keeps updating.

Softmax              : A math function that converts raw scores into
                       probabilities (all between 0 and 1, summing to 1).
                       e.g. [2.0, 1.0, 0.5] becomes roughly [0.59, 0.24, 0.17].

Cross-Entropy Loss   : The most common loss formula for text models.
                       Measures how many "bits of surprise" the model felt.
                       Higher = more surprised = text looks more anomalous.

Epoch                : One full pass through ALL the training data.
                       Training for 5 epochs = reading all normal logs 5 times.

Learning Rate        : How large a step the optimizer takes when updating
                       weights. Too large: overshoots. Too small: too slow.
                       C# analogy: the step size in a binary search.

Gradient             : The direction to nudge each weight to reduce loss.
                       We always move OPPOSITE to the gradient (downhill).

PyTorch              : A Python deep-learning library that calculates gradients
                       automatically. We use it in Part B.
                       C# analogy: ML.NET, but much more widely used in research.

NumPy                : A Python library for fast array (matrix) math.
                       C# analogy: MathNet.Numerics or a float[] with helpers.

Fallback             : If PyTorch is not installed, we switch to a simpler
                       NumPy implementation automatically (try/except pattern).

DevOps               : Development + Operations. Engineers who keep production
                       systems running. They are the main users of log anomaly
                       detectors in real companies.

=============================================================================
PROJECT OVERVIEW
=============================================================================

Goal: Build TWO anomaly detectors for server log lines, then compare them.

Real-World Context:
  DevOps teams at companies like Netflix, LinkedIn, and Microsoft deal with
  millions of log lines every day. Writing rules to catch every possible
  problem is impossible. Instead, they train a model on NORMAL logs, then
  flag any log line the model finds "surprising" (high loss).
  This project shows you exactly how that works.

PART A - Rule-Based Detector (no ML):
  * Hard-code a list of suspicious keywords (ERROR, CRITICAL, etc.)
  * Scan each test log line for those keywords
  * Flag the line if any keyword is found
  * Fast and simple, but brittle

PART B - ML-Based Detector using Loss:
  * Train a character-level bigram model ONLY on NORMAL log lines
  * The model learns the patterns of normal text (character by character)
  * Then compute the loss on each TEST log line
  * High loss means the model was surprised -> likely anomalous
  * Uses PyTorch if available, falls back to NumPy if not

Side-by-Side Comparison:
  * See both results next to each other for every test line
  * Understand WHY loss is a useful signal, not just a training metric

Training Data (10 normal log lines, hard-coded -- no files needed):
  INFO lines from a healthy server: logins, requests, cache hits, jobs.

Test Data (10 mixed log lines, hard-coded):
  Some normal, some with ERROR/CRITICAL/FATAL/WARNING prefixes and messages
  that the model has never seen before.
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import math       # standard Python math library (like System.Math in C#)
import random     # for reproducible randomness (like System.Random in C#)

# Try to import PyTorch for Part B.
# If it is not installed, we set a flag and fall back to NumPy.
# C# analogy: like checking if an optional NuGet package is available at runtime
try:
    import torch                          # main PyTorch module
    import torch.nn as nn                 # neural network building blocks
    import torch.optim as optim           # optimizers (Adam, SGD, etc.)
    TORCH_AVAILABLE = True                # flag: PyTorch IS available
except ImportError:
    TORCH_AVAILABLE = False               # flag: PyTorch NOT available

import numpy as np   # NumPy: array math library, always available as fallback

# =============================================================================
# HARD-CODED DATA
# =============================================================================

# NORMAL training log lines (10 lines from a healthy, boring server day).
# The ML model will ONLY see these during training.
# C# analogy: this is like List<string> trainingData = new List<string> { ... }
NORMAL_LOGS = [
    "INFO 2024-01-15 08:00:01 Server started on port 8080",
    "INFO 2024-01-15 08:00:05 Database connection established",
    "INFO 2024-01-15 08:00:10 User login successful for user123",
    "INFO 2024-01-15 08:00:15 Request GET /api/users completed in 45ms",
    "INFO 2024-01-15 08:00:20 Request POST /api/orders completed in 120ms",
    "INFO 2024-01-15 08:00:25 Cache hit for key user_profile_456",
    "INFO 2024-01-15 08:00:30 User logout successful for user123",
    "INFO 2024-01-15 08:00:35 Scheduled job cleanup ran successfully",
    "INFO 2024-01-15 08:00:40 Request GET /api/products completed in 30ms",
    "INFO 2024-01-15 08:00:45 Email notification sent to admin",
]

# TEST log lines (10 lines: mix of normal and anomalous).
# These are NOT shown to the model during training.
# C# analogy: like IEnumerable<string> testData in a unit test.
TEST_LOGS = [
    "INFO 2024-01-15 08:01:00 Request GET /api/users completed in 42ms",
    "ERROR 2024-01-15 08:01:05 Database connection refused after 3 retries",
    "INFO 2024-01-15 08:01:10 User login successful for user789",
    "WARNING 2024-01-15 08:01:15 Memory usage at 95 percent capacity",
    "INFO 2024-01-15 08:01:20 Request POST /api/orders completed in 115ms",
    "CRITICAL 2024-01-15 08:01:25 Disk space below 1 percent on volume C",
    "INFO 2024-01-15 08:01:30 Cache hit for key user_profile_789",
    "ERROR 2024-01-15 08:01:35 Unauthorized access attempt from IP 192.168.1.99",
    "INFO 2024-01-15 08:01:40 Scheduled job cleanup ran successfully",
    "FATAL 2024-01-15 08:01:45 Out of memory exception in payment service",
]

# =============================================================================
# PART A: RULE-BASED ANOMALY DETECTION (No Machine Learning)
# =============================================================================
#
# How it works:
#   1. We define a list of "anomaly keywords" (ERROR, CRITICAL, etc.)
#   2. For each test log line, we check if any keyword appears in the line.
#   3. If yes -> ANOMALY.  If no -> NORMAL.
#
# Analogy: like a spam filter that flags emails containing "FREE MONEY".
# C# analogy: string.Contains() in a loop over a list of bad keywords.
# =============================================================================

# List of keywords that strongly suggest something went wrong.
# If ANY of these appears in a log line, we flag it as anomalous.
# C# analogy: string[] anomalyKeywords = { "ERROR", "CRITICAL", ... };
ANOMALY_KEYWORDS = [
    "ERROR",        # something failed
    "CRITICAL",     # serious failure
    "FATAL",        # unrecoverable crash
    "WARNING",      # potential problem (we flag warnings too)
    "EXCEPTION",    # an exception was thrown
    "REFUSED",      # connection or request was refused
    "UNAUTHORIZED", # security: someone accessed something they should not
    "FAILED",       # an operation failed
    "TIMEOUT",      # request took too long
    "OUT OF MEMORY",# the process ran out of RAM
]


def rule_based_detect(log_line):
    """
    Check one log line against the anomaly keyword list.

    Parameters:
        log_line (str): A single log line to check.
                        C# analogy: string logLine

    Returns:
        (bool, str): A tuple of (is_anomaly, reason).
                     C# analogy: (bool isAnomaly, string reason) value tuple
    """
    # Convert the log line to uppercase so our comparison is case-insensitive.
    # C# analogy: logLine.ToUpper()
    upper_line = log_line.upper()

    # Loop through every keyword in our anomaly list.
    # C# analogy: foreach (string keyword in anomalyKeywords)
    for keyword in ANOMALY_KEYWORDS:
        # Check if this keyword appears anywhere in the log line.
        # C# analogy: upper_line.Contains(keyword)
        if keyword in upper_line:
            # Found a match! Return True (is anomaly) and which keyword matched.
            return True, keyword   # like return (true, keyword) in C#

    # No keyword matched, so this line looks normal.
    return False, "none"           # like return (false, "none") in C#


def run_part_a():
    """
    Run the rule-based anomaly detector on all test log lines and print results.
    """
    # Print a big separator so the output is easy to read.
    print("=" * 60)                        # 60 equals signs as a divider
    print("PART A: Rule-Based Anomaly Detection")
    print("=" * 60)
    print()

    # Explain what rule-based means in plain English.
    print("Method: Scan each log line for suspicious keywords.")
    print("Keywords we flag:", ", ".join(ANOMALY_KEYWORDS))
    print()

    # Print a header row for our results table.
    # rjust / ljust are Python string methods for padding text to a fixed width.
    # C# analogy: string.PadRight() and string.PadLeft()
    print("{:<10} {:<55} {}".format("RESULT", "LOG LINE (truncated)", "KEYWORD"))
    print("-" * 80)   # 80 dashes as a sub-divider

    # results_a will store each line's rule-based result so we can compare later.
    # C# analogy: List<(bool isAnomaly, string reason)> resultsA = new();
    results_a = []

    # Loop through every test log line.
    # enumerate() gives us an index i AND the value line at the same time.
    # C# analogy: for (int i = 0; i < TEST_LOGS.Length; i++)
    for i, line in enumerate(TEST_LOGS):
        # Call our rule-based detector for this line.
        is_anomaly, keyword = rule_based_detect(line)

        # Store the result so Part C (comparison) can use it.
        results_a.append(is_anomaly)       # like resultsA.Add(isAnomaly)

        # Build a short label: "ANOMALY" or "normal".
        label = "ANOMALY" if is_anomaly else "normal"

        # Truncate the log line to 54 chars so the table stays tidy.
        # C# analogy: line.Substring(0, Math.Min(line.Length, 54))
        short_line = line[:54]

        # Print one result row.
        # The format string left-justifies (:<10) and (:<55) each column.
        print("{:<10} {:<55} {}".format(label, short_line, keyword))

    print()   # blank line after the table
    return results_a   # return results for later comparison


# =============================================================================
# PART B: ML-BASED ANOMALY DETECTION USING LOSS
# =============================================================================
#
# THE KEY IDEA (read carefully!):
#
#   We train a tiny language model ONLY on the 10 normal log lines.
#   The model learns what "normal" characters look like in sequence.
#
#   Then we feed each TEST log line through the model and compute the loss.
#
#   Loss = how SURPRISED was the model by this sequence of characters?
#
#   Normal line   -> model has seen similar patterns -> LOW loss
#   Anomalous line -> model has NOT seen these patterns -> HIGH loss
#
#   We pick a THRESHOLD loss value. Lines above that threshold are anomalies.
#
# This is exactly how real anomaly detectors work at companies like Netflix.
# C# analogy: instead of "does this string match our rules?", we ask
#             "how different is this input from everything we trained on?"
#
# BIGRAM MODEL RECAP:
#   For every pair of consecutive characters (a, b) in training data:
#     count[(a, b)] += 1
#   To predict: given char a, the next char b is sampled proportional to
#     count[(a, b)] / sum(count[(a, *)])
#   Loss for a sequence = -average log probability of each next character.
# =============================================================================


# ===========================================================================
# Part B -- NumPy fallback (always available)
# ===========================================================================

class BigramModelNumpy:
    """
    A character-level bigram language model built with NumPy only.
    No PyTorch needed.

    Internally stores a matrix of counts:
      counts[i, j] = how often character j followed character i in training.

    C# analogy: a class with a float[,] countMatrix field.
    """

    def __init__(self, vocab_size):
        """
        Set up an empty count matrix of shape (vocab_size x vocab_size).

        Parameters:
            vocab_size (int): number of unique characters in the vocabulary.
                              C# analogy: int vocabSize
        """
        # counts is a 2-D array: rows = "current char", cols = "next char".
        # We start with all zeros (no observations yet).
        # C# analogy: float[,] counts = new float[vocabSize, vocabSize];
        self.counts = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    def train(self, sequences):
        """
        Count how often each (current_char, next_char) pair appears.

        Parameters:
            sequences (list of list of int): each inner list is one log line
                                             encoded as integer indices.
        """
        # Loop through every encoded log line.
        for seq in sequences:                  # like foreach (var seq in sequences)
            # Loop through pairs of consecutive characters in this line.
            # zip(seq, seq[1:]) gives (seq[0], seq[1]), (seq[1], seq[2]), ...
            # C# analogy: for (int k = 0; k < seq.Count - 1; k++)
            for current_idx, next_idx in zip(seq, seq[1:]):
                # Increment the count for this (current, next) pair.
                # C# analogy: counts[currentIdx, nextIdx]++
                self.counts[current_idx, next_idx] += 1

    def compute_loss(self, sequence):
        """
        Compute the average cross-entropy loss for one encoded sequence.

        Loss formula for one step:
          loss = -log(probability_of_next_char)
        Average over all steps in the sequence.

        Parameters:
            sequence (list of int): one log line as character indices.

        Returns:
            float: average loss for this sequence.
        """
        # Accumulate total loss across all character pairs in the sequence.
        total_loss = 0.0    # C# analogy: double totalLoss = 0.0;
        steps = 0           # count how many (current, next) pairs we saw

        for current_idx, next_idx in zip(sequence, sequence[1:]):
            # Get the row of counts for the current character.
            # This tells us how often each "next char" followed "current char".
            row = self.counts[current_idx]   # NumPy row slice, like row = counts[i, :]

            # Compute the total count for this row (sum of all next-char counts).
            row_sum = row.sum()              # like row.Sum() in LINQ

            if row_sum == 0:
                # This character was never seen in training.
                # Assign maximum surprise: loss of 10.0 (very high).
                # C# analogy: like throwing a KeyNotFoundException in a Dictionary
                total_loss += 10.0
            else:
                # Probability of next_idx given current_idx.
                # C# analogy: float prob = counts[currentIdx, nextIdx] / rowSum;
                prob = row[next_idx] / row_sum

                if prob == 0:
                    # This transition was never seen in training -> max surprise.
                    total_loss += 10.0
                else:
                    # Cross-entropy: -log(probability).
                    # log of a small probability is very negative.
                    # Negating it gives a large positive loss (high surprise).
                    # C# analogy: totalLoss += -Math.Log(prob);
                    total_loss += -math.log(prob)

            steps += 1    # we processed one more character pair

        # Return average loss (total divided by number of steps).
        # If sequence is only 1 char long, there are no pairs -> return 0.
        if steps == 0:
            return 0.0                   # no pairs to evaluate

        return total_loss / steps        # average loss per character pair


# ===========================================================================
# Part B -- PyTorch version (used if PyTorch is installed)
# ===========================================================================

def build_pytorch_bigram_model(vocab_size):
    """
    Build a tiny PyTorch bigram model as a single embedding layer.

    A single nn.Embedding of shape (vocab_size, vocab_size) learns a
    score (logit) for every (current_char, next_char) pair.
    It is mathematically equivalent to the NumPy count table, but trained
    with gradient descent instead of simple counting.

    Parameters:
        vocab_size (int): number of unique characters.

    Returns:
        nn.Module: the PyTorch model.
    """
    # nn.Embedding(num_embeddings, embedding_dim) creates a lookup table.
    # Here we use vocab_size x vocab_size so each character maps to a
    # score vector over all possible next characters.
    # C# analogy: new float[vocabSize, vocabSize] but learnable.
    model = nn.Embedding(vocab_size, vocab_size)
    return model


def train_pytorch_model(model, sequences, epochs=50, lr=0.1):
    """
    Train the PyTorch bigram model on encoded log line sequences.

    Parameters:
        model     : the nn.Embedding model from build_pytorch_bigram_model()
        sequences : list of list of int (encoded log lines)
        epochs    : how many full passes through the data (default 50)
        lr        : learning rate (default 0.1)
    """
    # Adam optimizer: updates model weights to reduce loss.
    # C# analogy: like an intelligent step-size controller for gradient descent.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # nn.CrossEntropyLoss computes the cross-entropy between predicted
    # logits and the true next character.
    # reduction='mean' means it averages the loss over all positions.
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Training loop: repeat for the requested number of epochs.
    for epoch in range(epochs):            # like for (int epoch = 0; epoch < epochs; epoch++)
        # Zero out any leftover gradients from the previous step.
        # C# analogy: reset all gradient accumulators to 0.
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0)     # accumulate batch loss

        for seq in sequences:              # loop over each encoded log line
            if len(seq) < 2:              # skip lines with fewer than 2 chars
                continue

            # Convert the Python list to a PyTorch LongTensor (integer tensor).
            # C# analogy: long[] inputArray = seq.ToArray();
            seq_tensor = torch.tensor(seq, dtype=torch.long)

            # Input:  every character EXCEPT the last one (we want to predict next)
            # Target: every character EXCEPT the first one (the "next" character)
            inputs  = seq_tensor[:-1]   # chars 0..N-2
            targets = seq_tensor[1:]    # chars 1..N-1

            # Forward pass: look up embedding scores for each input character.
            # logits shape: (sequence_length, vocab_size)
            # C# analogy: float[,] logits = model.Forward(inputs);
            logits = model(inputs)

            # Compute cross-entropy loss between logits and target characters.
            # C# analogy: double loss = criterion.Compute(logits, targets);
            loss = criterion(logits, targets)

            total_loss = total_loss + loss    # accumulate

        # Backpropagation: compute gradients of total_loss w.r.t. all weights.
        # C# analogy: like calling .Backward() to propagate error signals.
        total_loss.backward()

        # Optimizer step: nudge the weights in the direction that reduces loss.
        # C# analogy: update all weights using their computed gradients.
        optimizer.step()

    # Training is complete. The model now "knows" normal log patterns.


def compute_pytorch_loss(model, sequence):
    """
    Compute the average cross-entropy loss for one encoded sequence using PyTorch.

    Parameters:
        model    : trained PyTorch model
        sequence : list of int (one encoded log line)

    Returns:
        float: average loss for this sequence.
    """
    # torch.no_grad() disables gradient tracking (we are NOT training here).
    # C# analogy: like calling model.Eval() to switch to inference mode.
    with torch.no_grad():
        if len(sequence) < 2:
            return 0.0    # nothing to evaluate

        # Convert the sequence list to a LongTensor.
        seq_tensor = torch.tensor(sequence, dtype=torch.long)

        inputs  = seq_tensor[:-1]    # all chars except last
        targets = seq_tensor[1:]     # all chars except first

        # Forward pass: get logit scores for each input character.
        logits = model(inputs)

        # Compute cross-entropy loss.
        criterion = nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(logits, targets)

        # .item() extracts the Python float from the PyTorch tensor.
        # C# analogy: (float)loss
        return loss.item()


# ===========================================================================
# Shared utilities: build vocabulary, encode text
# ===========================================================================

def build_vocabulary(text_lines):
    """
    Build a vocabulary from a list of text lines.
    Returns:
        char_to_idx : dict mapping each unique character to an integer index.
                      C# analogy: Dictionary<char, int>
        idx_to_char : dict mapping each integer index back to its character.
                      C# analogy: Dictionary<int, char>
        vocab_size  : number of unique characters (int).
    """
    # Collect all unique characters from ALL lines combined.
    # We join all lines with a newline, then use set() to remove duplicates.
    # C# analogy: new HashSet<char>(string.Join("\n", lines))
    all_chars = set("".join(text_lines))

    # Sort the characters so the vocabulary is consistent across runs.
    # C# analogy: all_chars.OrderBy(c => c).ToList()
    sorted_chars = sorted(all_chars)

    # Build char -> index mapping.
    # enumerate() gives (0, 'char0'), (1, 'char1'), ...
    # C# analogy: sorted_chars.Select((c, i) => (c, i)).ToDictionary(...)
    char_to_idx = {ch: idx for idx, ch in enumerate(sorted_chars)}

    # Build index -> char mapping (the reverse lookup).
    # C# analogy: char_to_idx.ToDictionary(kv => kv.Value, kv => kv.Key)
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

    # The number of unique characters.
    vocab_size = len(sorted_chars)    # like sorted_chars.Count

    return char_to_idx, idx_to_char, vocab_size


def encode_line(line, char_to_idx):
    """
    Convert a string into a list of integer indices using the vocabulary.

    Unknown characters (not seen during training) get mapped to index 0
    as a safe default (like an "unknown token").

    Parameters:
        line        : the log line string.
        char_to_idx : Dictionary<char, int> vocabulary mapping.

    Returns:
        list of int: the encoded sequence.
        C# analogy: int[] encoded = line.Select(c => charToIdx.GetValueOrDefault(c, 0)).ToArray();
    """
    # .get(ch, 0) returns the index of ch, or 0 if ch is not in the vocabulary.
    # C# analogy: charToIdx.TryGetValue(ch, out int idx) ? idx : 0
    return [char_to_idx.get(ch, 0) for ch in line]


def run_part_b():
    """
    Run the ML-based anomaly detector on all test log lines and print results.
    Returns a list of (loss_value, is_anomaly) tuples for comparison.
    """
    print("=" * 60)
    print("PART B: ML-Based Anomaly Detection (Loss as Signal)")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Step 1: Build vocabulary from NORMAL logs only.
    # The model must ONLY learn what normal looks like.
    # ------------------------------------------------------------------
    print("Step 1: Building vocabulary from normal training logs...")

    # Call our helper function to get the char-to-index mapping.
    char_to_idx, idx_to_char, vocab_size = build_vocabulary(NORMAL_LOGS)

    # Tell the student how many unique characters we found.
    print("  Vocabulary size (unique characters):", vocab_size)
    print()

    # ------------------------------------------------------------------
    # Step 2: Encode ALL normal log lines into integer sequences.
    # ------------------------------------------------------------------
    print("Step 2: Encoding normal log lines as integer sequences...")

    # List of encoded sequences: each element is a list of ints.
    # C# analogy: List<int[]> encodedTraining = normalLogs.Select(Encode).ToList();
    encoded_training = [encode_line(line, char_to_idx) for line in NORMAL_LOGS]

    print("  Encoded", len(encoded_training), "normal log lines.")
    print()

    # ------------------------------------------------------------------
    # Step 3: Train the model (PyTorch or NumPy depending on availability).
    # ------------------------------------------------------------------
    if TORCH_AVAILABLE:
        # PyTorch path: gradient-descent-trained embedding model.
        print("Step 3: Training PyTorch bigram model for 50 epochs...")
        print("  (PyTorch found -- using gradient descent)")
        print()

        # Seed the random number generator for reproducibility.
        # C# analogy: new Random(42)
        torch.manual_seed(42)

        # Build the tiny embedding model.
        model_pt = build_pytorch_bigram_model(vocab_size)

        # Train the model on normal log sequences.
        train_pytorch_model(model_pt, encoded_training, epochs=50, lr=0.1)

        print("  PyTorch training complete.")
        print()

        # Helper: decide how to call the loss function.
        # C# analogy: Func<List<int>, float> computeLoss = ...
        def get_loss(encoded_seq):
            return compute_pytorch_loss(model_pt, encoded_seq)

    else:
        # NumPy fallback path: count-based bigram model.
        print("Step 3: Training NumPy bigram model (counting character pairs)...")
        print("  (PyTorch not found -- using NumPy count-based bigram)")
        print()

        # Create and train the NumPy bigram model.
        model_np = BigramModelNumpy(vocab_size)
        model_np.train(encoded_training)    # just counts character pairs

        print("  NumPy training complete.")
        print()

        # Helper: compute loss using the NumPy model.
        def get_loss(encoded_seq):
            return model_np.compute_loss(encoded_seq)

    # ------------------------------------------------------------------
    # Step 4: Compute loss for each TEST log line.
    # ------------------------------------------------------------------
    print("Step 4: Computing loss for each test log line...")
    print()

    # Collect all loss scores so we can compute a threshold.
    # C# analogy: List<float> allLosses = new();
    all_losses = []

    for line in TEST_LOGS:              # loop over every test line
        enc = encode_line(line, char_to_idx)    # encode it to ints
        loss_val = get_loss(enc)                # compute loss
        all_losses.append(loss_val)             # store it

    # ------------------------------------------------------------------
    # Step 5: Choose a threshold.
    # Lines with loss ABOVE the threshold are flagged as anomalies.
    #
    # Simple rule: threshold = mean_loss + 0.5 * std_dev_loss
    # This is similar to how real anomaly detectors work (z-score thresholding).
    # C# analogy: like computing a control limit on a Statistical Process Control chart.
    # ------------------------------------------------------------------
    mean_loss   = sum(all_losses) / len(all_losses)    # average loss
    # Standard deviation: measures how spread out the losses are.
    # C# analogy: Math.Sqrt(losses.Select(l => Math.Pow(l - mean, 2)).Average())
    variance    = sum((l - mean_loss) ** 2 for l in all_losses) / len(all_losses)
    std_loss    = math.sqrt(variance)

    # Threshold: mean + 0.5 standard deviations above the mean.
    threshold = mean_loss + 0.5 * std_loss

    print("  Loss statistics across all test lines:")
    print("    Average loss : {:.4f}".format(mean_loss))
    print("    Std deviation: {:.4f}".format(std_loss))
    print("    Threshold    : {:.4f}  (mean + 0.5 * std)".format(threshold))
    print()

    # ------------------------------------------------------------------
    # Step 6: Print the ML-based detection results.
    # ------------------------------------------------------------------
    print("{:<10} {:<8} {:<55}".format("RESULT", "LOSS", "LOG LINE (truncated)"))
    print("-" * 80)

    # Store results for the side-by-side comparison in Part C.
    # C# analogy: List<(bool isAnomaly, float loss)> resultsB = new();
    results_b = []

    for i, (line, loss_val) in enumerate(zip(TEST_LOGS, all_losses)):
        # Flag as anomaly if loss is above the threshold.
        is_anomaly = loss_val > threshold      # C# analogy: bool isAnomaly = lossVal > threshold;

        # Build a short label.
        label = "ANOMALY" if is_anomaly else "normal"

        # Truncate the log line for display.
        short_line = line[:54]

        # Print one result row with the loss value to 4 decimal places.
        print("{:<10} {:<8.4f} {:<55}".format(label, loss_val, short_line))

        # Store the result tuple.
        results_b.append((loss_val, is_anomaly))    # like resultsB.Add((lossVal, isAnomaly))

    print()
    return results_b, threshold    # return for the comparison section


# =============================================================================
# PART C: SIDE-BY-SIDE COMPARISON
# =============================================================================

def run_comparison(results_a, results_b, threshold):
    """
    Print a side-by-side comparison of rule-based vs ML-based detection.

    Parameters:
        results_a : list of bool    (rule-based: True = anomaly)
        results_b : list of (float, bool) (ML: loss value and is_anomaly flag)
        threshold : float           (ML loss threshold used)
    """
    print("=" * 60)
    print("COMPARISON: Rule-Based  vs  ML-Based (Loss)")
    print("=" * 60)
    print()

    # Explain the comparison columns.
    print("Columns:")
    print("  Rule  : ANOMALY = keyword matched | normal = no keyword")
    print("  ML    : ANOMALY = loss > {:.2f}    | normal = loss <= {:.2f}".format(
        threshold, threshold))
    print("  Loss  : higher = model more surprised = more likely anomaly")
    print()

    # Print a column header row.
    print("{:<12} {:<12} {:<8} {}".format(
        "RULE-BASED", "ML-BASED", "LOSS", "LOG LINE"))
    print("-" * 90)

    # Loop through every test line and print both results side by side.
    for i, line in enumerate(TEST_LOGS):
        # Get rule-based result for this line.
        rule_label = "ANOMALY" if results_a[i] else "normal"

        # Get ML-based result for this line.
        ml_loss, ml_is_anomaly = results_b[i]
        ml_label = "ANOMALY" if ml_is_anomaly else "normal"

        # Truncate log line for display.
        short_line = line[:54]

        # Print one comparison row.
        print("{:<12} {:<12} {:<8.4f} {}".format(
            rule_label, ml_label, ml_loss, short_line))

    print()

    # Count how many lines each method flagged.
    rule_flags = sum(1 for r in results_a if r)                        # count Trues
    ml_flags   = sum(1 for _, ml_anom in results_b if ml_anom)        # count Trues

    print("Total anomalies flagged:")
    print("  Rule-based : {}".format(rule_flags))
    print("  ML-based   : {}".format(ml_flags))
    print()

    # Check where the two methods DISAGREE.
    print("Lines where the two methods DISAGREE:")
    disagreements = 0    # counter for disagreements

    for i, line in enumerate(TEST_LOGS):
        rule_anom = results_a[i]
        _, ml_anom = results_b[i]

        if rule_anom != ml_anom:            # C# analogy: if (ruleAnom != mlAnom)
            disagreements += 1
            rule_str = "ANOMALY" if rule_anom else "normal"
            ml_str   = "ANOMALY" if ml_anom   else "normal"
            print("  Line {}: Rule={} | ML={}".format(i + 1, rule_str, ml_str))
            print("    {}".format(line[:80]))   # show the log line

    if disagreements == 0:
        print("  None: both methods agreed on every line.")

    print()


# =============================================================================
# LEARNING SUMMARY
# =============================================================================

def print_summary():
    """
    Print a plain-English summary of what we learned in this project.
    """
    print("=" * 60)
    print("SUMMARY: What Did We Learn?")
    print("=" * 60)
    print()
    print("1. RULE-BASED DETECTION")
    print("   - Simple and fast: keyword matching.")
    print("   - Pro: easy to understand and explain.")
    print("   - Con: requires you to predict every possible anomaly keyword.")
    print("   - Con: misses NEW types of anomalies you never wrote rules for.")
    print()
    print("2. ML-BASED DETECTION (Loss as Signal)")
    print("   - Train ONLY on normal logs.")
    print("   - Loss = model surprise. Low loss = normal. High loss = anomaly.")
    print("   - Pro: can catch anomalies you never explicitly defined.")
    print("   - Pro: adapts to your specific system's normal patterns.")
    print("   - Con: choosing the right threshold requires experimentation.")
    print("   - Con: needs enough training data to define 'normal' well.")
    print()
    print("3. REAL-WORLD USAGE")
    print("   - Netflix, LinkedIn, and Microsoft use loss-based anomaly")
    print("     detection in production monitoring systems.")
    print("   - As a .NET developer, you might use ML-based log analysis")
    print("     via Azure Anomaly Detector or custom PyTorch models.")
    print()
    print("4. KEY INSIGHT")
    print("   Loss is not just a 'training metric' to minimize.")
    print("   Loss is a USEFUL SIGNAL at inference time too.")
    print("   High loss tells you: 'I have never seen anything like this.'")
    print("   That surprise IS the anomaly detection mechanism.")
    print()


# =============================================================================
# QUIZ QUESTIONS
# =============================================================================
#
# QUIZ 1 (Multiple Choice):
#   When a language model produces HIGH loss on a log line, it means:
#   A) The model is very confident the line is normal.
#   B) The model is very surprised by the log line -- it looks unusual.
#   C) The model has been trained on too much data.
#   D) The log line contains a syntax error.
#
#   ANSWER: B
#   Explanation: Loss measures model surprise. High loss = the model never
#   saw these character patterns during training. If training only used normal
#   logs, high loss signals something outside the normal distribution.
#
# -------------------------------------------------------------------------
#
# QUIZ 2 (Short Answer):
#   Why do we train the ML model ONLY on normal log lines, and NOT on
#   anomalous ones?
#
#   ANSWER:
#   Because we want the model to learn what "normal" looks like.
#   If we also trained on anomalous lines, the model would learn to expect
#   anomalies too, so its loss would be low for both normal AND anomalous lines.
#   By training only on normal data, any deviation from normal (anomalous line)
#   will produce high loss because the model never learned to expect it.
#
# -------------------------------------------------------------------------
#
# QUIZ 3 (Concept Check):
#   A rule-based detector flags a log line that contains the word "WARNING".
#   An ML-based detector gives the same line a LOSS of 2.1 (below threshold 2.5).
#   The ML detector says it is NORMAL. Which detector is right?
#   How would you investigate the disagreement?
#
#   ANSWER:
#   Both could be "right" depending on context. The rule-based detector uses
#   an explicit rule (WARNING = bad). The ML-based detector says: "this WARNING
#   line looks similar in character patterns to the training data."
#   To investigate:
#   1. Check if WARNING lines appear in your training data (they don't here,
#      so the ML model might be fooled by shared characters like "2024-01-15").
#   2. You could lower the threshold to be more sensitive.
#   3. You could add WARNING lines to the vocabulary of rule-based keywords.
#   4. You could train on more diverse data so the model better separates classes.
#   Real-world systems often combine BOTH approaches (ensemble detection).
#
# =============================================================================

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main function: runs Part A, Part B, and the comparison in order.
    C# analogy: static void Main(string[] args)
    """
    # Set a fixed random seed for reproducibility.
    # C# analogy: new Random(42) to get the same sequence every run.
    random.seed(42)
    np.random.seed(42)

    # Print a welcome banner.
    print()
    print("=" * 60)
    print("PROJECT 6: Log Anomaly Detector")
    print("Training data : 10 normal INFO log lines")
    print("Test data     : 10 mixed log lines (some anomalous)")
    if TORCH_AVAILABLE:
        print("ML backend    : PyTorch (gradient descent bigram model)")
    else:
        print("ML backend    : NumPy (count-based bigram model -- fallback)")
    print("=" * 60)
    print()

    # ---------- PART A ----------
    results_a = run_part_a()       # rule-based detection, returns list of bool

    # ---------- PART B ----------
    results_b, threshold = run_part_b()    # ML detection, returns (loss, bool) pairs

    # ---------- COMPARISON ----------
    run_comparison(results_a, results_b, threshold)

    # ---------- SUMMARY ----------
    print_summary()


# Standard Python idiom: only call main() if this script is run directly.
# C# analogy: if (Environment.GetCommandLineArgs()[0] == thisAssembly) { Main(); }
if __name__ == "__main__":
    main()

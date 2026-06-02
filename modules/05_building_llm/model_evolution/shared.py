# =============================================================================
# shared.py
# =============================================================================
# PURPOSE: Common utilities used by ALL step_XX files.
#
# WHY A SHARED FILE?
#   Every step (bigram, MLP, attention, GPT...) needs the same boilerplate:
#     - Load training data from disk
#     - Build a character vocabulary
#     - Encode text to integers, decode back to text
#     - Sample random batches for training
#     - Measure loss on validation set
#     - Generate text samples
#     - Run the training loop
#
#   Instead of copy-pasting these 200 lines into every file, we put them
#   here once and import them.
#
# C# ANALOGY:
#   Think of this as a static utility class, like:
#     public static class DataUtils { ... }
#     public static class TrainingUtils { ... }
#   You'd put these in a shared project/library that all other projects reference.
#
# HOW TO USE:
#   from shared import load_data, build_vocab, encode, decode, get_batch, ...
# =============================================================================

import os       # File system operations (paths, directories)
import time     # Measuring elapsed training time
import torch    # PyTorch — the deep learning framework
                # C# analogy: like using ML.NET, but far more widely used in research

# --- Path to the training corpus ---
# __file__ = path to THIS shared.py file
# We navigate to data/corpus.txt relative to this file's location
# This way the path works no matter what directory you run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(SCRIPT_DIR, "data", "corpus.txt")


# =============================================================================
# FUNCTION: load_data
# =============================================================================
def load_data(max_chars=10_000_000):
    """
    Load corpus.txt from disk and return it as a single string.

    WHY max_chars?
      10 million characters is plenty for our small models.
      Loading more would slow training without much benefit.
      On a CPU, 10MB trains in 5-10 minutes per step.

    C# ANALOGY:
      Like File.ReadAllText(path).Substring(0, maxChars)
      — read the file, then take only the first N characters.

    PARAMETERS:
      max_chars (int): Maximum number of characters to load. Default 10M.

    RETURNS:
      str: The raw text content.
    """
    # Check if the corpus file exists before trying to open it
    if not os.path.exists(CORPUS_PATH):
        # Tell the user exactly what command to run — friendly error message
        raise FileNotFoundError(
            f"Training data not found at:\n  {CORPUS_PATH}\n\n"
            "Please run first:\n  python download_data.py"
        )

    print(f"  Loading data from: {CORPUS_PATH}")

    # Open the file in read mode with UTF-8 encoding
    # "r" = read mode (text mode, not binary)
    # errors="replace" = replace unreadable characters with ? instead of crashing
    # C# analogy: File.ReadAllText(path, Encoding.UTF8)
    with open(CORPUS_PATH, "r", encoding="utf-8", errors="replace") as f:
        text = f.read(max_chars)  # read(N) reads at most N characters

    print(f"  Loaded {len(text):,} characters")
    return text


# =============================================================================
# FUNCTION: build_vocab
# =============================================================================
def build_vocab(text):
    """
    Build a character-level vocabulary from the training text.

    WHAT IS A VOCABULARY?
      A vocabulary maps characters <-> integers so our model can work with
      numbers instead of letters. Neural networks only understand numbers!

      Example:
        ' ' -> 0      'a' -> 1      'b' -> 2     ...    'z' -> 27
        0   -> ' '    1   -> 'a'    2   -> 'b'   ...    27  -> 'z'

    WHY CHARACTER-LEVEL?
      Simpler than word-level or subword (BPE) tokenization.
      vocab_size is small (~100 chars vs ~50,000 words).
      Perfect for learning the fundamentals.

    C# ANALOGY:
      char2idx is like Dictionary<char, int>   — encoding lookup
      idx2char is like List<char>              — decoding lookup (index = position)

    PARAMETERS:
      text (str): The full training corpus.

    RETURNS:
      tuple: (char2idx, idx2char)
        char2idx: dict mapping character -> integer index
        idx2char: list where idx2char[i] gives the character at index i
    """
    # sorted(set(text)) gives us a sorted list of all unique characters
    # set() removes duplicates — like a HashSet<char> in C#
    # sorted() puts them in consistent alphabetical order
    chars = sorted(set(text))  # e.g., [' ', '!', '"', ..., 'z', '{', '}']

    vocab_size = len(chars)    # How many unique characters exist

    # Build char -> int mapping
    # Dictionary comprehension: { char: index for each (index, char) pair }
    # C# analogy: chars.Select((c, i) => new { c, i }).ToDictionary(x => x.c, x => x.i)
    char2idx = {ch: idx for idx, ch in enumerate(chars)}

    # idx2char is just the list itself — the index IS the lookup key
    # idx2char[0] = first char, idx2char[1] = second char, etc.
    idx2char = chars  # Already a list, already indexed

    print(f"  Vocabulary size: {vocab_size} unique characters")
    return char2idx, idx2char


# =============================================================================
# FUNCTION: encode
# =============================================================================
def encode(text, char2idx):
    """
    Convert a string of text into a list of integer token IDs.

    EXAMPLE:
      encode("hello", char2idx) -> [23, 14, 27, 27, 30]
      (exact numbers depend on vocab)

    C# ANALOGY:
      Like text.Select(c => char2idx.TryGetValue(c, out var id) ? id : -1)
                .Where(id => id >= 0)
                .ToList()

    PARAMETERS:
      text (str): The string to encode.
      char2idx (dict): Character -> index mapping from build_vocab().

    RETURNS:
      list[int]: A list of integer token IDs.
    """
    # List comprehension: for every character in text, look up its integer ID
    # .get(ch, None) returns None if the character isn't in our vocab
    # We skip unknown characters (filter out None values)
    # C# analogy: text.Select(c => char2idx.GetValueOrDefault(c, -1)).Where(i => i >= 0)
    return [char2idx[ch] for ch in text if ch in char2idx]


# =============================================================================
# FUNCTION: decode
# =============================================================================
def decode(ids, idx2char):
    """
    Convert a list of integer token IDs back into a text string.

    EXAMPLE:
      decode([23, 14, 27, 27, 30], idx2char) -> "hello"

    C# ANALOGY:
      Like string.Concat(ids.Select(i => idx2char[i]))

    PARAMETERS:
      ids (list[int]): The list of integer IDs to decode.
      idx2char (list): Index -> character mapping from build_vocab().

    RETURNS:
      str: The decoded text string.
    """
    # "".join([...]) concatenates a list of characters into a string
    # C# analogy: String.Concat(ids.Select(i => idx2char[i]))
    return "".join(idx2char[i] for i in ids)


# =============================================================================
# FUNCTION: get_batch
# =============================================================================
def get_batch(data, batch_size, block_size, device):
    """
    Sample a random mini-batch of (input, target) pairs from the data.

    WHAT IS A BATCH?
      Instead of training on one example at a time (slow), we train on
      batch_size examples simultaneously (fast, better gradient estimates).
      C# analogy: like parallel processing multiple work items at once.

    WHAT IS block_size?
      The number of tokens the model sees at once ("context window").
      block_size=64 means the model sees 64 characters and predicts the 65th.

    THE SHIFT-BY-ONE TRICK:
      x (input)  = data[i   : i+block_size]    <- model SEES these
      y (target) = data[i+1 : i+block_size+1]  <- model must PREDICT these

      VISUAL EXAMPLE (block_size=4):
        data = [5, 3, 7, 2, 8, 1, 4, 6]
        x    = [5, 3, 7, 2]   <- what model sees
        y    = [3, 7, 2, 8]   <- what model must predict (shifted right by 1)

        Position 0: given 5, predict 3
        Position 1: given 5,3, predict 7
        Position 2: given 5,3,7, predict 2
        Position 3: given 5,3,7,2, predict 8

      So ONE sequence of length 4 actually gives us 4 training examples!
      C# analogy: like a sliding window of size block_size over the data array.

    PARAMETERS:
      data (torch.LongTensor): 1D tensor of all token IDs.
      batch_size (int): Number of sequences per batch.
      block_size (int): Length of each sequence (context window size).
      device (str): "cpu" or "cuda" — where to put the tensors.

    RETURNS:
      tuple: (x, y) where x and y are both shape (batch_size, block_size)
    """
    # data has shape (N,) — N total tokens in our dataset
    n = len(data)

    # Pick batch_size random starting positions
    # Each position i means: use data[i : i+block_size] as one training example
    # We stop at n - block_size so we don't go past the end of data
    # torch.randint(low, high, size) is like Random.Next(low, high) in C#
    # but generates multiple random numbers at once
    ix = torch.randint(n - block_size, (batch_size,))  # shape: (batch_size,)

    # Stack batch_size sequences into a 2D tensor
    # torch.stack takes a list of 1D tensors and stacks them into a 2D tensor
    # C# analogy: like converting List<int[]> into a 2D int[,] array
    x = torch.stack([data[i    : i + block_size    ] for i in ix])  # (B, T)
    y = torch.stack([data[i+1  : i + block_size + 1] for i in ix])  # (B, T)

    # Move tensors to the correct device (CPU or GPU)
    # .to(device) is like converting between memory spaces
    # C# analogy: there's no direct equivalent, but think of it as
    # choosing whether to store data in RAM vs GPU VRAM
    x, y = x.to(device), y.to(device)

    return x, y  # Both shapes: (batch_size, block_size)


# =============================================================================
# FUNCTION: split_data
# =============================================================================
def split_data(text, char2idx, train_frac=0.9):
    """
    Encode the text and split into train / validation sets.

    WHY SPLIT?
      We train on the training set and measure performance on the validation set.
      The model never "sees" the validation set during training.
      This tells us if the model is learning patterns vs. just memorizing.
      C# analogy: like having a test project separate from your main project.

    SPLIT RATIO:
      90% train, 10% validation (standard practice for language models).
      If text has 10M chars: 9M train, 1M validation.

    PARAMETERS:
      text (str): Full text corpus.
      char2idx (dict): Character -> index mapping.
      train_frac (float): Fraction to use for training (default 0.9 = 90%).

    RETURNS:
      tuple: (train_data, val_data) as torch.LongTensor tensors
    """
    # Encode the entire text to a list of integers
    all_ids = encode(text, char2idx)  # e.g., [5, 3, 7, 2, 8, 1, ...]

    # Convert to a PyTorch tensor of type long (64-bit integer)
    # torch.tensor() wraps a Python list into a PyTorch tensor
    # dtype=torch.long means int64 — needed for embedding lookups
    # C# analogy: like converting List<int> to long[] array
    data = torch.tensor(all_ids, dtype=torch.long)

    # Calculate the split point
    # int(...) truncates to integer — like (int)(...) in C#
    split = int(train_frac * len(data))  # e.g., 90% of 10M = 9M

    # Slice the tensor at the split point
    # data[:split]  = first 90%  = training data
    # data[split:]  = last 10%   = validation data
    # C# analogy: like arr[..split] and arr[split..] with C# 8 range syntax
    train_data = data[:split]
    val_data   = data[split:]

    print(f"  Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")
    return train_data, val_data


# =============================================================================
# FUNCTION: estimate_loss
# =============================================================================
@torch.no_grad()   # This decorator turns off gradient tracking for speed
                   # C# analogy: like using a read-only snapshot — no tracking overhead
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters=100):
    """
    Estimate average loss on both train and validation sets.

    WHY NOT JUST USE TRAINING LOSS?
      The loss we print during training is noisy — it's just one batch.
      For a more accurate estimate, we average over eval_iters batches.

    WHY SET model.eval()?
      Some layers behave differently during training vs evaluation.
      Dropout (used in step_04+) randomly zeros activations during training
      but should be disabled during evaluation for consistent results.
      C# analogy: like a flag that puts the system in "read-only mode".

    @torch.no_grad() EXPLAINED:
      During training, PyTorch tracks all operations to compute gradients.
      This uses extra memory and time. During evaluation we don't need gradients,
      so we disable tracking for speed. Like turning off logging in production.

    PARAMETERS:
      model: The neural network model (any of our step_XX models).
      train_data, val_data: Tensors of token IDs.
      batch_size, block_size: Batch configuration.
      device: "cpu" or "cuda".
      eval_iters (int): How many batches to average over.

    RETURNS:
      dict: {"train": avg_train_loss, "val": avg_val_loss}
    """
    results = {}  # Will hold {"train": float, "val": float}
                  # C# analogy: var results = new Dictionary<string, float>();

    model.eval()  # Switch model to evaluation mode (disables dropout, etc.)

    # Evaluate on both splits
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        # Collect eval_iters loss values, then average them
        losses = torch.zeros(eval_iters)  # Pre-allocate a tensor of zeros
                                           # C# analogy: new float[evalIters]

        for k in range(eval_iters):
            # Get a random batch from this split
            xb, yb = get_batch(split_data, batch_size, block_size, device)

            # Forward pass: compute loss
            # We don't need the logits here, just the loss
            _logits, loss = model(xb, yb)  # _ prefix = "I'm ignoring this value"
                                            # C# analogy: (_, var loss) = model(xb, yb)

            # .item() converts a single-element tensor to a Python float
            # C# analogy: like calling .Value on a Nullable<float>
            losses[k] = loss.item()

        # Average all the individual losses
        # .mean() computes the arithmetic mean of all elements
        results[split_name] = losses.mean().item()

    model.train()  # Switch back to training mode (re-enables dropout, etc.)

    return results  # e.g., {"train": 2.341, "val": 2.389}


# =============================================================================
# FUNCTION: generate_text
# =============================================================================
def generate_text(model, char2idx, idx2char, device, seed="The ", max_new=300, block_size=64):
    """
    Generate text from the model, starting from a seed string.

    WHAT IS TEXT GENERATION?
      We give the model a starting sequence ("The ") and ask it to predict
      what comes next, one character at a time. Then we feed the prediction
      back in as input for the next step.

      Like autocomplete, but we keep going for max_new characters.

    TEMPERATURE SAMPLING (temperature=0.8):
      Instead of always picking the most likely next character (greedy),
      we sample randomly from the probability distribution.

      Temperature controls "creativity":
        temperature=1.0  → original probabilities (balanced)
        temperature<1.0  → sharper distribution (less random, more focused)
        temperature>1.0  → flatter distribution (more random, more creative)

      We use 0.8: slightly focused but still varied. Good for readable output.

      C# analogy: like a weighted random choice — not always picking the
      most likely option, but weighted toward likely options.

    PARAMETERS:
      model: The trained model.
      char2idx, idx2char: Vocabulary mappings.
      device (str): "cpu" or "cuda".
      seed (str): Starting text to condition generation on. Default "The ".
      max_new (int): Number of new characters to generate.
      block_size (int): Model's context window size.

    RETURNS:
      str: The generated text (seed + generated characters).
    """
    model.eval()  # Evaluation mode — disable dropout

    # Encode the seed string to a list of integers
    seed_ids = encode(seed, char2idx)

    # Convert to tensor, add a batch dimension of size 1
    # seed_ids = [5, 23, 14, ...]  (a 1D list)
    # After unsqueeze(0): shape becomes (1, len(seed_ids))
    # We need the batch dimension because the model expects shape (B, T)
    # C# analogy: like wrapping a single item in a single-element array
    context = torch.tensor(seed_ids, dtype=torch.long).unsqueeze(0).to(device)
    # context shape: (1, len(seed_ids))

    generated_ids = seed_ids.copy()  # Start with the seed, then append to this

    with torch.no_grad():  # No gradient tracking needed for generation
        for _ in range(max_new):  # Generate max_new characters one at a time
            # Crop context to the last block_size tokens
            # The model can only look back block_size tokens
            # If context is longer, we throw away the oldest part
            # C# analogy: like using a Queue<int> with a max capacity
            context_cropped = context[:, -block_size:]  # (1, min(T, block_size))

            # Forward pass: get logits (raw scores for each vocab token)
            # We pass targets=None because we're generating, not computing loss
            logits, _loss = model(context_cropped)  # logits shape: (1, T, vocab_size)

            # We only care about the LAST token's prediction
            # logits[:, -1, :] = last time step's scores = (1, vocab_size)
            # C# analogy: like taking the Last() element of a sequence
            logits = logits[:, -1, :]  # (1, vocab_size)

            # Apply temperature: divide logits before softmax
            # Lower temperature → sharper probabilities → less random
            temperature = 0.8
            logits = logits / temperature  # Scale the raw scores

            # Convert logits to probabilities using softmax
            # softmax turns arbitrary numbers into a probability distribution
            # (all values between 0 and 1, sum = 1)
            # C# analogy: like normalizing a List<double> so they sum to 1
            probs = torch.softmax(logits, dim=-1)  # (1, vocab_size)

            # Sample one token from the probability distribution
            # torch.multinomial picks one index, weighted by probabilities
            # num_samples=1 means pick one character
            # C# analogy: like a weighted random number generator
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # Add the new token to our running context for the next step
            # torch.cat concatenates tensors along a dimension
            # dim=1 = concatenate along the time/sequence dimension
            context = torch.cat([context, next_id], dim=1)  # (1, T+1)

            # Also save the ID so we can decode later
            generated_ids.append(next_id.item())  # .item() converts tensor to Python int

    model.train()  # Restore training mode

    # Decode the full list of IDs back to a string
    return decode(generated_ids, idx2char)


# =============================================================================
# FUNCTION: run_training
# =============================================================================
def run_training(model_name, model, train_data, val_data, char2idx, idx2char, config):
    """
    Run the full training loop for any model.

    WHAT IS A TRAINING LOOP?
      1. Sample a batch of data
      2. Forward pass: run data through model, compute loss
      3. Backward pass: compute gradients (which direction to adjust weights)
      4. Optimizer step: update weights to reduce loss
      5. Repeat thousands of times

      C# analogy: like a retry loop that keeps adjusting parameters until
      an objective function (the loss) is minimized.

    WHAT IS LOSS?
      Loss = how wrong the model is. Lower = better.
      We use cross-entropy loss, which measures how surprised the model is
      by the actual next character.
      Perfect model: loss ≈ 0
      Random model: loss ≈ log(vocab_size) ≈ 4.6 for 100-char vocab

    WHAT IS AN OPTIMIZER?
      The optimizer decides HOW to update weights based on gradients.
      We use AdamW — the standard choice for transformers.
      C# analogy: like a gradient-descent algorithm that also adapts its
      learning rate per parameter.

    CONFIG KEYS:
      batch_size (int)     : examples per batch
      block_size (int)     : context window size
      max_iters (int)      : total training steps
      eval_interval (int)  : how often to print loss
      lr (float)           : learning rate (step size for weight updates)
      device (str)         : "cpu" or "cuda"

    RETURNS:
      float: Final validation loss.
    """
    # --- Unpack config ---
    # C# analogy: like reading from IConfiguration or a settings dictionary
    batch_size    = config["batch_size"]
    block_size    = config["block_size"]
    max_iters     = config["max_iters"]
    eval_interval = config["eval_interval"]
    lr            = config["lr"]
    device        = config["device"]

    # --- Move model to device ---
    # This sends all model parameters (weights) to CPU or GPU
    # C# analogy: like allocating memory on the device
    model = model.to(device)

    # --- Count trainable parameters ---
    # sum(p.numel() for p in model.parameters() if p.requires_grad)
    # .numel() = number of elements in a tensor
    # p.requires_grad = True means this parameter will be updated during training
    # C# analogy: like reflection to count all public properties that have setters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")
    print(f"  Device    : {device}")
    print()

    # --- Generate a sample BEFORE training ---
    # This shows the student what untrained gibberish looks like
    print("--- Sample BEFORE training ---")
    sample = generate_text(model, char2idx, idx2char, device, block_size=block_size)
    print(sample[:300])  # Print first 300 characters of generated text
    print()

    # --- Set up the optimizer ---
    # AdamW = Adam optimizer with Weight Decay (L2 regularization)
    # lr = learning rate: how big each update step is
    # Too high: training is unstable (loss goes up and down wildly)
    # Too low : training is slow (takes forever to converge)
    # lr=3e-4 (0.0003) is the classic "safe" value for transformers
    # C# analogy: AdamW is like a smarter version of gradient descent
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- Training loop ---
    print("--- Training ---")
    start_time = time.time()  # Record start time for elapsed time display

    final_val_loss = float("inf")  # Will be updated during training

    for step in range(max_iters):
        # ---- Evaluate periodically ----
        # We don't evaluate every step — that's too slow.
        # Instead, evaluate every eval_interval steps.
        if step % eval_interval == 0:
            # estimate_loss averages over 100 batches for a stable reading
            losses = estimate_loss(
                model, train_data, val_data,
                batch_size, block_size, device
            )
            elapsed = time.time() - start_time  # How many seconds since we started

            # Print a formatted status line
            print(
                f"  step {step:5d}"                             # Right-aligned 5-char step number
                f" | train loss {losses['train']:.4f}"          # 4 decimal places
                f" | val loss {losses['val']:.4f}"
                f" | {elapsed:.1f}s"
            )
            final_val_loss = losses["val"]  # Save the latest val loss

        # ---- Get a training batch ----
        xb, yb = get_batch(train_data, batch_size, block_size, device)

        # ---- Forward pass ----
        # Run the input through the model to get logits and loss
        _logits, loss = model(xb, yb)

        # ---- Backward pass ----
        # Zero the gradients from the previous step
        # If we don't zero them, gradients accumulate (wrong!)
        # C# analogy: like resetting accumulators before a new calculation
        optimizer.zero_grad(set_to_none=True)  # set_to_none=True is slightly faster

        # Compute gradients via backpropagation
        # This calculates d(loss)/d(weight) for every weight in the model
        # C# analogy: no direct equivalent — this is automatic differentiation
        loss.backward()

        # Update weights using the computed gradients
        # AdamW adjusts each weight in the direction that reduces loss
        optimizer.step()

    # Print final loss after training completes
    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time:.1f}s")

    # --- Generate samples AFTER training ---
    # Let the student see how much the model improved
    print()
    print("--- Sample AFTER training ---")
    for i in range(2):  # Print 2 different samples
        sample = generate_text(model, char2idx, idx2char, device, block_size=block_size)
        print(f"  [Sample {i+1}]")
        print(f"  {sample[:300]}")
        print()

    return final_val_loss

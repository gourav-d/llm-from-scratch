# Udemy Small LLM Deep Dive
# Line-by-Line Explanation with C#/.NET Analogies

Source notebooks:
- `jupyterNotebooks/small_tokenizer_official.ipynb` -- trains a SentencePiece tokenizer
- `jupyterNotebooks/small_llm_official.ipynb` -- trains a 19M-parameter GPT model

Standalone Python versions (run without Jupyter):
- `jupyterNotebooks/small_tokenizer_standalone.py`
- `jupyterNotebooks/small_llm_standalone.py`

---

## Table of Contents

- Part A: SentencePiece Tokenizer -- Every Parameter Explained
- Part B: Architecture Parameters -- WHY These Numbers?
- Part C: GPT Class Line-by-Line
- Part D: Block, ForwardLayer, Multihead, Head Classes
- Part E: Production Training Details
- Part F: bfloat16 and torch.compile

---

## Part A: SentencePiece Tokenizer -- Every Parameter Explained

### What is SentencePiece?

Our earlier modules covered the BPE algorithm from scratch (building it step-by-step in Python).
SentencePiece is Google's production-ready tokenizer library that implements BPE (and other algorithms)
with industrial-strength features. It is what LLaMA, T5, and many other real models use.

In C# terms: our scratch BPE is like writing your own JSON parser for learning.
SentencePiece is like using `System.Text.Json` -- same concept, battle-tested implementation.

### Training Call (from the notebook):

```python
spm.SentencePieceTrainer.train(
    input='wiki.txt',
    model_prefix="test_wiki_tokenizer",
    model_type="bpe",
    vocab_size=4096,
    self_test_sample_size=0,
    input_format="text",
    character_coverage=0.995,
    num_threads=os.cpu_count(),
    split_digits=True,
    allow_whitespace_only_pieces=True,
    byte_fallback=True,
    unk_surface=r" \342\201\207 ",
    normalization_rule_name="identity"
)
```

Let's go through EVERY parameter:

---

### `input='wiki.txt'`

**What:** The file containing training text. SentencePiece reads this to learn what character pairs
appear most frequently, which determines what gets merged into tokens.

**Analogy:** Like a training dataset for any machine learning model -- the tokenizer "learns"
from this data what patterns are common in the language.

**Practical note:** The larger and more representative your text, the better the tokenizer.
Training on English Wikipedia gives you a tokenizer good at English Wikipedia topics.
If you train on Python code instead, you get a code tokenizer.

---

### `model_prefix="test_wiki_tokenizer"`

**What:** The base name for the output files. This call creates:
- `test_wiki_tokenizer.model` -- the binary model file (load this to tokenize new text)
- `test_wiki_tokenizer.vocab` -- a human-readable list of all vocabulary pieces

**Analogy:** Like a `FileInfo` base path in C# -- you specify the prefix and the library
adds the extensions for each output file type.

---

### `model_type="bpe"`

**What:** The algorithm used to build the tokenizer. Options:
- `"bpe"` (Byte Pair Encoding): starts with characters, repeatedly merges most-frequent pairs
- `"unigram"`: starts with a large vocabulary, prunes less useful pieces
- `"word"`: simple word-level tokenization (no subword splitting)
- `"char"`: one character per token (no merging at all)

**Why BPE?**
BPE is the most common choice for LLMs. It handles rare words well by splitting them
into common subword pieces:
- "unhappiness" -> ["un", "happy", "ness"] (three common pieces)
- vs word-level which would make "unhappiness" a single token or unknown

**C# analogy:** Choosing an algorithm is like choosing a sort algorithm. BPE is the QuickSort
of tokenization -- widely used, well-understood, good performance.

---

### `vocab_size=4096`

**What:** Total number of unique tokens the tokenizer will have.

**Why 4096?**
- 4096 = 2^12 -- a power of 2, which is GPU-friendly (memory alignment)
- Small for a demo model. Real models use much larger vocabularies:
  - GPT-2: 50,257 tokens
  - LLaMA 2: 32,000 tokens
  - GPT-4: ~100,000 tokens (estimated)

**Tradeoff:**
- Larger vocab = fewer tokens per sentence = faster training per batch, but harder to learn meanings
- Smaller vocab = more tokens per sentence = slower, but each token is simpler to learn

**C# analogy:** Like choosing the size of a fixed-size cache. Too small and you miss many words;
too large and you waste memory on rare items.

---

### `character_coverage=0.995`

**What:** The fraction of unique characters in the training corpus to include in the vocabulary.
0.995 means "include the most frequent 99.5% of characters. The rarest 0.5% are treated as unknown."

**Why not 1.0?**
Some characters appear only once or twice (obscure Unicode, typos). Including ALL of them
wastes vocabulary slots on useless rare characters. 99.5% coverage includes all practically
important characters while ignoring extreme outliers.

**Practical example:**
If your text has 10,000 unique characters and you set coverage=0.995, the tokenizer includes
the 9,950 most common characters and treats the other 50 as unknown (or byte_fallback handles them).

**C# analogy:** Like setting a minimum frequency threshold when building a word frequency dictionary.
You keep words that appear at least N times; very rare words get a special "other" bucket.

---

### `num_threads=os.cpu_count()`

**What:** How many CPU cores to use during training.
`os.cpu_count()` returns the total number of logical CPU cores on your machine.

**Why:** Tokenizer training involves counting character pair frequencies across the entire text file.
This is embarrassingly parallel -- more cores = proportionally faster training.

**C# analogy:** Like setting `ParallelOptions.MaxDegreeOfParallelism = Environment.ProcessorCount`
in a `Parallel.ForEach` call.

---

### `split_digits=True`

**What:** If True, numbers are split into individual digits.
"2024" becomes ["2", "0", "2", "4"] -- four separate tokens.

**Why is this important?**
Without digit splitting, "2024" might become a single token that the model never saw during training.
"2023" and "2024" would look completely unrelated. With digit splitting:
- Every year is just four digit tokens
- Model learns arithmetic relationships between digits
- Handles any number, no matter how large

**C# analogy:** Like separating a date string "20240315" into individual characters
before processing, so the parser handles any date, not just dates it saw before.

---

### `allow_whitespace_only_pieces=True`

**What:** If True, spaces can be their own tokens.
A sequence of spaces like "   " can become a single token (or several space tokens).

**Why:** Preserves exact spacing in code and structured text.
Python code cares about indentation -- "    def" (4 spaces + def) should tokenize
in a way that preserves the 4 spaces when decoded back.

Without this: spaces get awkwardly attached to surrounding words, making decoding imprecise.

---

### `byte_fallback=True`

**What:** If True, any character NOT in the vocabulary is encoded as individual byte tokens
(e.g., <0x41> for 'A', <0xC3><0xA9> for 'e' with accent).

**Why this is important:**
Without byte_fallback, any character outside the vocab becomes a single <UNK> (unknown) token.
Information is LOST -- you cannot decode back to the original text.

With byte_fallback: every possible character can be encoded (since all bytes 0x00-0xFF are
represented). The vocabulary includes 256 byte tokens as a fallback. No true unknown tokens.

**C# analogy:** Like using UTF-8 encoding with a fallback to literal byte escaping.
Even if you can't decode a character, you can still represent its raw bytes.

---

### `unk_surface=r" \342\201\207 "`

**What:** When an unknown token appears in decoded output, what string to show.
The default is a rare Unicode character (a flower symbol) that is unlikely to appear in text.
`\342\201\207` is octal encoding for that character (octal is base-8 number notation).

**Why octal?** SentencePiece uses octal escapes in this specific field (historical convention).

**Practical impact:** If the model generates an <UNK> token, you see this character in the output.
Since it's rare, it acts as a visible marker that something went wrong (rare word or encoding issue).

---

### `normalization_rule_name="identity"`

**What:** Controls text normalization before tokenization. "identity" means NO normalization.
Other options:
- `"nmt_nfkc"`: Unicode normalization, lowercasing, and some cleaning (good for multilingual)
- `"nfkc"`: Unicode normalization only
- `"nmt_nfkc_cf"`: Like nmt_nfkc but also case-folded (everything lowercase)

**Why "identity" for this model?**
We want the tokenizer to learn the actual text as-is, preserving:
- Capitalization: "Dog" and "dog" as different tokens
- Punctuation: "Hello!" and "Hello" as different sequences
- Mixed case: "McDonald's" stays "McDonald's"

If we normalized (e.g., lowercased everything), the model would lose information about
proper nouns, start-of-sentence capitalization, etc.

**C# analogy:** Like `StringComparison.Ordinal` (exact match) vs `StringComparison.OrdinalIgnoreCase`
(normalized match). We want exact here.

---

## Part B: Architecture Parameters -- WHY These Numbers?

```python
# ARCHITECTURE PARAMETERS
batch_size = 8
context = 512
embed_size = 384
n_layers = 7
n_heads = 7
BIAS = True

# HYPERPARAMETERS
lr = 3e-4
dropout = 0.05
weight_decay = 0.01
grad_clip = 1.0
```

---

### `batch_size = 8`

**What:** How many training examples to process in parallel in one gradient update step.

**Why 8?**
- With 4GB GPU: batch_size=8 fits in memory
- With 24GB GPU: batch_size=128 is possible (faster training)
- Each "example" is a sequence of 512 tokens -- that's 512 * 8 = 4096 tokens per batch

**Effect of changing:**
- Double batch_size -> roughly half as many gradient steps needed, but 2x memory
- Too small (1-2) -> noisy gradient estimates, unstable training
- Too large -> better gradient estimates, but diminishing returns past ~128

**C# analogy:** Like the `BulkInsert` batch size in Entity Framework. Processing 8 rows at once
vs 1 row at a time -- the database (GPU) is more efficient in bulk.

---

### `context = 512`

**What:** The maximum sequence length -- how many tokens the model can "see" at once.
Also called "context window" or "sequence length".

**Why 512?**
Memory for the attention mechanism scales as O(context^2):
- 512 tokens: 512 * 512 = 262,144 values in the attention matrix (manageable)
- 1024 tokens: 1024 * 1024 = 1,048,576 (4x more memory)
- GPT-4 uses 128K context, which requires FlashAttention and other tricks

**Effect:** A model with context=512 can only use the previous 512 tokens when predicting
the next token. Longer context = can maintain longer "memory" of the conversation.

---

### `embed_size = 384`

**What:** The dimensionality of the token embedding vectors. Each token is represented
as a list of 384 numbers.

**Why 384?**
- GPT-2 Small uses 768
- GPT-3 uses 12288
- 384 is half of GPT-2 Small -- half the parameters, still enough for a demo
- Must be divisible by n_heads: 384 / 7 = 54.86 -> floor to 54 per head

**Effect:** More dimensions = richer representation = more capacity to learn
complex patterns, but more parameters and slower training.

**C# analogy:** Like the rank (number of features) in a feature vector for machine learning.
A 384-dimensional embedding is like a float[384] array for each token.

---

### `n_layers = 7`

**What:** Number of stacked Transformer blocks.

**Scaling:**
- GPT-2 Small: 12 layers
- GPT-2 Large: 36 layers
- GPT-3: 96 layers
- GPT-4: estimated 120+ layers

**Why 7?**
Odd choice -- allows n_heads=7 to divide evenly. Each block adds ~2.7M parameters.
7 blocks * ~2.7M = ~19M total (target for this demo model).

**Effect:** More layers = deeper reasoning, can learn more abstract patterns.
The "depth" of a network is what makes it "deep learning".

**C# analogy:** Like chaining LINQ operations. Each `.Where().Select().GroupBy()` is a layer
that transforms the data. More chained operations = more complex transformation.

---

### `n_heads = 7`

**What:** Number of parallel attention heads in each Transformer block.

**Why 7?**
Must divide embed_size: 384 / 7 = 54 (integer division, slight loss).
7 is an unconventional choice -- GPT-2 uses 12. Chosen here to pair with 7 layers.

**Effect:** Each head attends to different relationship patterns simultaneously.
More heads = model can focus on multiple relationship types at once.

**Analogy:** Like having 7 domain experts each reviewing code from different perspectives
(security, performance, readability, correctness, style, documentation, testing).
Each expert (head) gives their assessment; the model combines all 7 views.

---

### `BIAS = True`

**What:** Whether Linear (fully-connected) layers include a bias term.

**Formula with bias:**    `output = input @ weight + bias`
**Formula without bias:** `output = input @ weight`

**Modern trend:** Some newer models (like LLaMA) remove biases. The thought is:
- LayerNorm already has scale and shift parameters (which do what bias does)
- Removing bias slightly reduces parameters and speeds up computation
- Empirically similar quality

**Why True here?** Conservative choice -- the original GPT-2 uses bias.

---

### `lr = 3e-4` (0.0003)

**What:** The initial learning rate -- how large a step to take when updating weights.

**Why 3e-4?**
Andrej Karpathy's nanoGPT uses this. It is the sweet spot for AdamW with transformer models:
- Too high (1e-2): unstable, loss oscillates or diverges
- Too low (1e-6): training is technically correct but takes forever
- 3e-4: fast learning without instability

The scheduler then decays this to 3e-5 over training (cosine annealing).

---

### `dropout = 0.05`

**What:** Fraction of neurons to randomly "switch off" each forward pass during training.

**Why 0.05 (5%)?**
Small model + small dataset = low dropout. If you drop too many neurons, the small model
cannot learn effectively. 5% adds just enough regularization without impeding learning.

Larger models on more data use higher dropout:
- GPT-2 (1.5B params): dropout=0.1 (10%)
- BERT: dropout=0.1
- Research shows 5-10% is typical for transformers

**See `09_missing_topics.md` for a full dedicated explanation of Dropout.**

---

### `weight_decay = 0.01`

**What:** L2 regularization coefficient. Adds a penalty to the loss proportional to the
magnitude of the weights: `total_loss = cross_entropy_loss + weight_decay * sum(weights^2)`

**WHY apply it only to weight matrices (not biases)?**
See Part E -- this is a crucial production detail covered in the AdamW parameter groups section.

**Effect:** Prevents individual weights from growing too large. Acts like a "gravity" pulling
all weights toward zero, which prevents overfitting.

---

### `grad_clip = 1.0`

**What:** Maximum allowed norm of the gradient vector across all parameters.

**What is gradient norm?**
All gradients for all parameters (tens of millions of numbers) are treated as a single vector.
The "norm" is the length of that vector (like Euclidean distance in millions of dimensions).
If this length exceeds grad_clip=1.0, all gradients are scaled down proportionally.

**Why do gradients explode?**
In deep networks, gradients are multiplied through many layers during backpropagation.
If any weight matrix has eigenvalues > 1, gradients grow exponentially ("exploding gradients").
Clipping provides a safety net.

---

## Part C: GPT Class Line-by-Line

```
INPUT: Sequence of token IDs (integers)
  |
  v
[Token Embedding]    -- each token ID -> 384-dim vector
  +
[Position Embedding] -- each position (0..511) -> 384-dim vector
  |
  v
[Block 1]  -- attention + FFN
[Block 2]
...
[Block 7]
  |
  v
[LayerNorm]          -- normalize final representations
  |
  v
[Linear: 384 -> 4096] -- score each token in vocabulary
  |
  v
OUTPUT: Logits (raw scores for next token prediction)
```

### `__init__` method

```python
def __init__(self):
    super().__init__()

    # nn.Embedding(num_embeddings, embedding_dim)
    # Like Dictionary<int, float[]> with 4096 entries of size 384 each
    self.embeddings = nn.Embedding(vocab_size, embed_size)  # (4096, 384)

    # Same structure but for positions 0..511
    # The model needs to know WHERE each token is (position matters!)
    self.positions = nn.Embedding(context, embed_size)      # (512, 384)

    # [Block(n_heads) for _ in range(n_layers)] creates a Python LIST of 7 Block objects
    # *[...] unpacks the list into separate arguments
    # nn.Sequential chains them: output of Block1 feeds into Block2, etc.
    self.blocks = nn.Sequential(*[Block(n_heads) for _ in range(n_layers)])

    # LayerNorm applied after all blocks
    self.ln = nn.LayerNorm(embed_size)

    # Final linear layer: maps each position's 384-dim vector to 4096 scores
    # These scores = how likely is each vocabulary token to come NEXT
    self.final_linear = nn.Linear(embed_size, vocab_size, bias=BIAS)

    # Apply weight initialization to ALL sub-modules
    # self.apply() is like a recursive Visitor pattern in C#:
    # it calls _init_weights(module) for every Linear, Embedding, LayerNorm in the model
    self.apply(self._init_weights)
```

### `_init_weights` -- Why Initialize Weights Carefully?

**The problem with random initialization:**

If all weights are drawn from a standard normal distribution (mean=0, std=1), then after passing
through many layers, the variance of activations grows exponentially. After 7 layers of 384-dim
linear transforms, the numbers become huge -> NaN (not a number) -> training fails.

**The solution (GPT-2 paper):** Use std=0.02 for all weight matrices and embeddings.

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        # Normal distribution, mean=0, std=0.02
        # Each weight starts small and slightly random
        # The network learns to push weights to useful values during training
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Bias starts at exactly zero: neutral, no initial preference
        # The optimizer will adjust bias values during training
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        # Same small initialization for embedding tables
        # All tokens start with similar small vectors; training pushes them apart
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

**C# analogy:** In game development, spawning enemies with random initial positions
in a small area (std=0.02) vs randomly across the entire world (std=1.0).
Small initial spread = stable start.

### `forward` -- Shape Tracking

The shapes are the KEY to understanding what is happening. Follow each tensor:

```python
def forward(self, input, targets=None):
    BS, SL = input.shape
    # input = (8, 512)  -- 8 sequences, each 512 token IDs
    # BS = 8, SL = 512

    emb = self.embeddings(input)
    # For each of the 8*512 = 4096 token IDs, look up the 384-dim embedding vector
    # Result: (8, 512, 384) -- each token is now a 384-number vector

    pos = self.positions(torch.arange(SL, device=device))
    # torch.arange(512) = [0, 1, 2, ..., 511] -- position indices
    # Look up position embeddings for each position
    # Result: (512, 384) -- one 384-vector per position
    # (NO batch dimension -- same position embeddings for all sequences in the batch)

    x = emb + pos
    # PyTorch broadcasting: pos (512, 384) is broadcast to match emb (8, 512, 384)
    # Like adding the same vector to every row in a 3D array
    # Result: (8, 512, 384) -- each token has meaning + position combined

    x = self.blocks(x)
    # Pass through 7 transformer blocks in sequence
    # Each block outputs same shape as input (residual connections preserve shape)
    # Result: (8, 512, 384) -- enriched representations

    x = self.ln(x)
    # Final layer normalization -- stabilizes the output of the last block
    # Result: (8, 512, 384) -- same shape, normalized

    logits = self.final_linear(x)
    # Linear: for each of the 8*512 positions, map 384-dim -> 4096-dim
    # These 4096 numbers are the "logits": raw scores for each vocabulary token
    # Higher score = model thinks this is more likely to be the next token
    # Result: (8, 512, 4096)
```

### Cross Entropy Loss

```python
if targets is not None:
    BS, SL, VS = logits.shape           # (8, 512, 4096)
    logits = logits.view(BS*SL, VS)     # reshape to (4096, 4096) -- flatten batch+sequence
    targets = targets.view(BS*SL)       # reshape to (4096,)
    loss = F.cross_entropy(logits, targets)
```

**WHY reshape?**
`F.cross_entropy` expects 2D input: (N, num_classes). We have (BS, SL, VS) = 3D.
By reshaping to (BS*SL, VS) we treat each position in each sequence as a separate example.
4096 positions * 8 batch = 4096 classification problems in one call.

**What is cross entropy?**
- The model outputs 4096 raw scores for the next token
- Softmax converts these to probabilities (4096 values that sum to 1.0)
- Cross entropy = -log(probability of the CORRECT next token)
- If model is 100% sure of the right answer: probability=1.0, loss = -log(1.0) = 0
- If model is wrong: probability near 0, loss = -log(small) = large number
- Training MINIMIZES this loss -> model learns to assign high probability to correct tokens

### `generate` -- Autoregressive Generation

```python
def generate(self, input, max=500):
    for _ in range(max):
        # Sliding window: keep only last 'context' tokens
        # If input grows beyond 512 tokens, slice off the oldest ones
        # (The model has a fixed context window -- cannot attend to more than 512 tokens)
        input = input[:, -context:]

        # Forward pass (no targets -> no loss calculation)
        logits, _ = self(input)         # (1, input_len, 4096)

        # Take only the LAST position's predictions
        # The last position predicts what comes AFTER the entire sequence
        logits = logits[:, -1, :]       # (1, 4096)

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)  # (1, 4096) -- all values 0..1, sum to 1

        # Sample the next token from the probability distribution
        # multinomial = weighted random draw (tokens with higher probability chosen more often)
        # This is better than argmax (greedy): argmax always picks the same highest-probability
        # token, leading to repetitive text. Sampling gives variety.
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Append the new token to the input sequence and repeat
        input = torch.cat((input, next_token), dim=1)

    return input  # full sequence including generated tokens
```

---

## Part D: Block, ForwardLayer, Multihead, Head Classes

### Block -- The Transformer Block

```
Input x (BS, SL, 384)
    |
    +---> LayerNorm1 -> MultiHeadAttention -> + ---> output1 (BS, SL, 384)
    |                                          |
    +-----------------------------------------+  <-- residual connection 1

    |
    +---> LayerNorm2 -> ForwardLayer ---------> + ---> output2 (BS, SL, 384)
    |                                           |
    +-------------------------------------------+  <-- residual connection 2

Output = output2
```

**Pre-Norm design (this notebook) vs Post-Norm (original 2017 paper):**

```python
# POST-NORM (original "Attention is All You Need" paper, 2017):
x = self.ln1(x + self.ma(x))       # LayerNorm AFTER attention+residual

# PRE-NORM (GPT-2, GPT-3, LLaMA -- this notebook):
x = x + self.ma(self.ln1(x))       # LayerNorm BEFORE attention
```

**Why Pre-Norm is better:**
- Post-Norm: gradients pass through LayerNorm during backprop, can cause instability in deep networks
- Pre-Norm: gradients bypass LayerNorm via the residual connection, flow more directly to early layers
- At 7+ layers, Pre-Norm trains more stably
- Modern consensus: Pre-Norm is preferred for decoder-only LLMs

---

### ForwardLayer -- The Feed-Forward Network

```
Input (BS, SL, 384)
    |
    v
Linear: 384 -> 2304  (6x expansion)
    |
    v
GELU activation
    |
    v
Linear: 2304 -> 384  (compress back)
    |
    v
Dropout(0.05)
    |
    v
Output (BS, SL, 384)
```

**Why 6x expansion (2304 = 6*384)?**
Our earlier module used 4x (the original paper's choice). This notebook uses 6x.
More expansion = more "working memory" for computation within the FFN.
Recent models (LLaMA) use 8/3 * embed_size with the SwiGLU activation.
6x is a middle ground -- more powerful than 4x, less extreme than 8/3x.

**GELU vs ReLU:**

```
ReLU:   f(x) = max(0, x)     -- hard cutoff at 0. Gradient is 0 or 1.
GELU:   f(x) = x * P(X <= x) -- smooth weighted activation
```

```
       |
 1.0   |         ///////
       |       //
 0.5   |     /
       |   /
 0.0   |/_______________
   -3  -2  -1   0   1   2
         ReLU (hard edge at 0)

       |
 1.0   |         ///////
       |       //
 0.5   |     /
       |  //
 0.0   +/-              <- smooth transition near 0
   -3  -2  -1   0   1   2
         GELU (smooth edge)
```

**Why GELU for LLMs?**
The smooth transition near 0 allows small gradients even for slightly negative values.
ReLU kills all negative values (gradient=0) -- no learning signal for those units.
GELU's smooth transition = better gradient flow = better training for language models.

---

### Head -- Single Attention Head

```
Input x (BS, SL, 384)
    |
    +-> Queries linear (384->54) -> Q (BS, SL, 54)
    +-> Keys    linear (384->54) -> K (BS, SL, 54)
    +-> Values  linear (384->54) -> V (BS, SL, 54)

Q @ K.T / sqrt(54)  ->  attention weights raw (BS, SL, SL)
    |
    mask upper triangle with -inf (causal masking)
    |
    softmax -> attention weights (BS, SL, SL)  [each row sums to 1.0]
    |
    dropout
    |
attention weights @ V -> output (BS, SL, 54)
```

**Q, K, V explained with an analogy:**
Imagine a library catalog system:
- Query (Q): "I'm looking for books about WWII aviation" -- what this token is searching for
- Key (K): "This book is about aviation history 1939-1945" -- what each token describes itself as
- Value (V): "Here is the actual content of this book" -- the actual information to share

When Q and K match well (high dot product), that token gets a large attention weight,
meaning its Value contributes more to the output.

**Scaling by sqrt(head_size):**
Without scaling: as head_size grows, dot products Q@K become large -> softmax saturates
(one near-1, rest near-0) -> one-hot attention -> gradients vanish.
Dividing by sqrt(54) = 7.35 keeps the dot products in a range where softmax gives
more spread-out probabilities.

**Causal masking:**
```
Position 0 can attend to: [0]
Position 1 can attend to: [0, 1]
Position 2 can attend to: [0, 1, 2]
...
Position 511 can attend to: [0, 1, ..., 511]
```
The lower triangular matrix enforces this -- each token only sees PAST tokens.
This is essential for generation: you can't know what comes later when predicting next token.

---

## Part E: Production Training Details

These details are NOT covered in our earlier modules but are CRITICAL for real training.

### Weight Decay Parameter Groups

**The problem:**
Weight decay (L2 regularization) penalizes large weights by adding `weight_decay * sum(w^2)`
to the loss. This pulls weights toward zero during training, preventing overfitting.

BUT: not all parameters SHOULD be pulled toward zero:
- **Bias terms:** Just offset values. Forcing toward 0 hurts the model's ability to represent
  non-zero baselines.
- **LayerNorm scale/shift:** These learn to scale and shift normalized values. Forcing toward 0
  breaks normalization.
- **Weight matrices (Linear, Embedding):** These represent feature transformations and
  BENEFIT from regularization.

**Solution: Two parameter groups with different weight_decay settings**

```python
# Get all trainable parameters as a dictionary: {parameter_name: tensor}
p_dict = {p_name: p for p_name, p in model.named_parameters() if p.requires_grad}

# Weight matrices are 2D (rows x cols). Also Embedding (2D), multi-head projection (2D).
# p.dim() returns the number of dimensions: weight matrix = 2, bias vector = 1
weight_decay_p = [p for n, p in p_dict.items() if p.dim() >= 2]

# Bias vectors are 1D. LayerNorm scale/shift are 1D.
no_weight_decay_p = [p for n, p in p_dict.items() if p.dim() < 2]

# Two groups, each with their own weight_decay setting
optimizer_groups = [
    {'params': weight_decay_p, 'weight_decay': weight_decay},   # 0.01
    {'params': no_weight_decay_p, 'weight_decay': 0.0},          # 0.0
]
optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.99))
```

**C# analogy:** Like applying a property tax rate to real estate but not to cash savings.
Both are assets, but the regularization (tax) makes sense for large static holdings (weights)
but not for liquid working capital (biases).

---

### AdamW Betas Explained

```python
optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.99))
```

**beta1 = 0.9 (momentum)**

AdamW keeps a running average of past gradients: `m = 0.9 * m_prev + 0.1 * gradient`
This is like momentum in physics -- the optimizer "remembers" the direction it was going.
If gradients consistently point the same direction, momentum builds up -> faster convergence.
If gradients oscillate, momentum dampens the oscillation.

**beta2 = 0.99 (variance tracking)**

AdamW keeps a running average of squared gradients: `v = 0.99 * v_prev + 0.01 * gradient^2`
This estimates how "noisy" or "large" each parameter's gradient tends to be.
Parameters with consistent small gradients (scale/bias layers) get larger effective LR.
Parameters with large noisy gradients get smaller effective LR.
This adaptive per-parameter learning rate is AdaM (Adaptive Moments).

**Adam vs AdamW:**
- Adam: weight decay is applied INSIDE the moment estimates (wrong -- it interacts with m and v)
- AdamW: weight decay applied DIRECTLY to weights BEFORE the Adam update (correct)
- "W" = "decoupled Weight decay" (from the 2019 paper by Loshchilov & Hutter)

**Note:** This notebook uses beta2=0.99, while standard Adam uses beta2=0.999.
Lower beta2 = more responsive to recent gradient changes. Good for this small dataset.

---

### Cosine Annealing LR Scheduler

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_iters, eta_min=lr/10)
```

**Visual:**

```
LR
3e-4 |\.
     | \..
     |    \....
2e-4 |         \......
     |                \.......
1e-4 |                         \..........
     |                                     \...............
3e-5 |                                                     \__________
     +-----------------------------------------------------------> iterations
     0                  50000                    100000
```

**How it works:**
`LR(t) = eta_min + 0.5 * (lr - eta_min) * (1 + cos(pi * t / train_iters))`

- At t=0: LR = lr = 3e-4 (start high)
- At t=50000: LR = (3e-4 + 3e-5) / 2 = ~1.65e-4 (halfway down)
- At t=100000: LR = 3e-5 = lr/10 (finish low)

**Why cosine?**
- Smooth (no sudden drops)
- Starts high (fast learning of rough structure)
- Ends low (fine-tuning details)
- The cosine shape is gentler than linear decay and avoids abrupt steps

---

### Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
```

This is called AFTER `loss.backward()` (compute gradients) and BEFORE `optimizer.step()` (apply them).

**What it does:**
Compute the total norm of all gradients: `total_norm = sqrt(sum(g^2 for all g in gradients))`
If `total_norm > max_norm=1.0`: multiply all gradients by `max_norm / total_norm`
Result: total norm is now exactly 1.0. Direction is preserved. Only magnitude is clipped.

**Different from our module's manual implementation:**
Our earlier implementation (`03_training_gpt.md`) showed a per-parameter clipping.
`nn.utils.clip_grad_norm_` clips the GLOBAL norm across ALL parameters together.
This is more principled -- it preserves the relative magnitude between parameter gradients.

---

### Checkpoint Save/Load

**Save:**
```python
torch.save({
    'model_state_dict': model.state_dict(),           # all weight tensors
    'optimizer_state_dict': optimizer.state_dict(),   # momentum, variance, step count
    'loss': best_val_loss,                            # best loss achieved
    'iteration': i,                                   # current training step
}, checkpoint_dir + checkpoint_fn)
```

**Load:**
```python
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
iteration = checkpoint['iteration']
```

**Why save optimizer state?**
AdamW maintains per-parameter state:
- `m`: gradient momentum (exponential moving average of gradients)
- `v`: gradient variance (exponential moving average of squared gradients)
- `step`: how many updates have been applied

Without restoring this state: the optimizer "forgets" its momentum and variance estimates.
Training effectively restarts in terms of optimizer behavior, wasting the warm-up period.

**C# analogy:** Like serializing and deserializing a search algorithm's internal state
(e.g., a Monte Carlo tree with visited nodes). You can stop and resume without losing progress.

---

### Wandb Logging

**What is Weights & Biases (wandb)?**
Weights & Biases is an experiment tracking platform for machine learning.
Think of it as "Application Insights + Azure Monitor" but for ML training runs.

**What it tracks:**
- Training loss over time
- Validation loss over time
- Learning rate changes
- GPU memory usage
- Time per iteration

**Why it matters:**
Without logging, you have to watch the console output and write it down manually.
With wandb, you get interactive dashboards, can compare multiple runs, and share results.

```python
if wandb_log:
    wandb.log({
        "loss/train": l['train'],
        "loss/val": l['eval'],
        "lr": scheduler.get_last_lr()[0],
    }, step=i)
```

**How to enable:**
1. `pip install wandb`
2. Set `wandb_log = True` in the script config
3. Run the script -- it will prompt you for your API key (free at wandb.ai)

---

## Part F: bfloat16 and torch.compile

### bfloat16 -- Why Use 16-bit Floats?

**Standard floating point formats:**

```
float32 (single precision):
  1 bit sign | 8 bits exponent | 23 bits mantissa
  Total: 32 bits
  Range: ~1.2e-38 to ~3.4e+38
  Precision: ~7 decimal digits

float16 (half precision):
  1 bit sign | 5 bits exponent | 10 bits mantissa
  Total: 16 bits
  Range: ~6e-8 to ~65504   <-- NARROW RANGE (problem!)
  Precision: ~3 decimal digits

bfloat16 (Brain Float 16, developed by Google):
  1 bit sign | 8 bits exponent | 7 bits mantissa
  Total: 16 bits
  Range: ~1.2e-38 to ~3.4e+38  <-- SAME RANGE as float32
  Precision: ~2-3 decimal digits  <-- less precise than float32
```

**Why bfloat16 over float16?**
float16's narrow range causes "overflow" (NaN) when gradients or activations exceed 65504.
This happens frequently in LLM training without careful scaling.
bfloat16 keeps float32's exponent range (8 bits), so it can represent the same magnitude
of numbers. It only sacrifices mantissa bits (precision), which is acceptable for gradients.

**Why 16-bit at all?**
- 2x less GPU memory: a model that needs 8GB in float32 only needs 4GB in bfloat16
- Faster matrix multiply: modern GPUs have dedicated bfloat16 hardware units (NVIDIA Tensor Cores)
- Training speed: 1.5-2x faster on Ampere+ GPUs

**C# analogy:** float (32-bit) vs a hypothetical type that keeps float's range but rounds
to 2 decimal places. You lose some precision but save memory and get faster hardware support.

**Usage in code:**
```python
dtype = torch.bfloat16
model = model.to(dtype)    # convert all model parameters to bfloat16
model = model.to(device)   # move to GPU
```

---

### TF32 -- The Hidden Performance Boost

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**What is TF32?**
TF32 = TensorFloat-32. A 19-bit format used INTERNALLY by NVIDIA Ampere GPUs (RTX 30xx, A100+)
for matrix multiplications:
- 1 sign + 8 exponent + 10 mantissa (internally -- NOT a PyTorch dtype you set)
- Range: same as float32 (8-bit exponent)
- Precision: similar to float16 (10-bit mantissa)

**What these flags do:**
When enabled, CUDA automatically uses TF32 instead of float32 for matrix multiply ops.
The inputs/outputs are still float32 -- TF32 is only used for the intermediate computation.
This gives ~3x speedup on Ampere GPUs with minimal accuracy impact.

**Why disabled by default?**
PyTorch disables TF32 by default for reproducibility. Different hardware gives slightly
different results due to TF32's reduced precision. For research requiring bit-exact results,
this matters. For LLM training, the tiny accuracy difference is irrelevant.

---

### torch.compile

```python
if compile:
    model = torch.compile(model)
```

**What it does:**
PyTorch 2.0+ feature. The first time the model runs, Python compiles the model's operations
into optimized native CUDA code (similar to how C# JIT compiles to machine code, but for GPU).

**Benefits:**
- Fuses multiple operations (e.g., LayerNorm + Linear -> single kernel)
- Eliminates Python overhead from the forward pass
- Reduces memory bandwidth (fused ops don't write intermediate results to GPU memory)
- Typical speedup: 10-30% on compatible hardware

**Why disabled by default in this notebook?**
- Requires PyTorch 2.0+ (older installs don't have it)
- Compilation takes ~2-3 minutes the first time (cold start)
- Incompatible with some debugging tools (ipdb, profilers)
- Some model architectures trigger compilation errors

**C# analogy:** Like Ahead-of-Time (AOT) compilation in .NET 7+ with `dotnet publish -r <RID>`.
Instead of JIT-compiling at runtime, you compile to native code ahead of time for better
performance, at the cost of a longer initial compilation step.

---

## Summary: What This Notebook Teaches vs Our Module

| Topic | This Notebook | Our Module |
|---|---|---|
| Tokenizer training | SentencePiece BPE (production) | BPE from scratch (educational) |
| Weight initialization | GPT-2 style (std=0.02, explained) | Inline 0.02 (unexplained) |
| AdamW param groups | Yes -- weight matrices vs bias | No |
| Cosine LR schedule | Yes -- actual PyTorch code | Named only, no code |
| Gradient clipping | nn.utils.clip_grad_norm_ | Manual implementation |
| Checkpoint save/load | Full -- model + optimizer state | Conceptual only |
| Wandb logging | Yes | Not mentioned |
| bfloat16 | Yes | Not mentioned |
| torch.compile | Yes | Not mentioned |
| Attention theory | 3 lines of code | 3 full documents |
| RLHF/Alignment | Not mentioned | Dedicated M06 + M13 modules |
| Dropout theory | Used (0.05) | Covered in M03/M06 |
| GAN/VAE/Flow/Diffusion | Not mentioned | M03, M16 modules |

**Verdict:** Learn the CONCEPTS from our module, then use this notebook for PRODUCTION code patterns.
The two complement each other: theory + implementation.

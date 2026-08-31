# small_llm_standalone.py
# A 19-million-parameter GPT language model trained on wiki.txt.
# Extracted from the Udemy "Small LLM" course notebook (small_llm_official.ipynb).
# Runs without Jupyter Notebook.
#
# Typical LLMs need many GPUs and millions of dollars to train.
# This model trains with a single GPU and little GPU memory.
# Results are not like ChatGPT, but good enough to see the LLM learn
# to go from random text to actual words and phrases.
# GPT-3 has 175 Billion parameters. GPT-4 has many more.
# This model has only 19 Million parameters with default settings.
#
# Usage:
#   python small_llm_standalone.py               # train from scratch
#   python small_llm_standalone.py --inference   # interactive inference mode
#   python small_llm_standalone.py --load        # load checkpoint and continue training
#
# Requirements (install once):
#   pip install torch sentencepiece tqdm
#   pip install wandb  (optional -- only needed if WANDB_LOG=True)
#   pip install ipdb   (optional -- only needed for debugging)
#
# Files needed:
#   wiki_tokenizer.model  -- SentencePiece tokenizer (from small_tokenizer_standalone.py)
#   encoded_data.pt       -- pre-tokenized dataset (auto-downloaded if missing)
#   wiki.txt              -- raw text dataset (auto-downloaded if missing)
#
# GPU note:
#   For GOOGLE COLAB and similar: select a GPU in Runtime menus. Do not use CPU -- too slow.
#   Locally: your GPU is selected automatically if available.

# Uncomment to install if needed:
# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece", "tqdm", "-q"])

import os
import sys
import argparse
from tqdm import tqdm                    # progress bar library
from datetime import datetime            # for naming wandb runs
import platform
import shutil
import requests
import zipfile
import io

import torch                             # PyTorch -- main deep learning library
import torch.nn as nn                   # neural network modules (layers, loss functions)
from torch.nn import functional as F    # functional API (softmax, cross_entropy, etc.)

import sentencepiece as spm              # tokenizer library

# These lines improve performance on Ampere Architecture GPUs (e.g. A100, RTX 3090/4090)
# TF32 is a 19-bit format that speeds up matrix multiplications on these GPUs
# with minimal accuracy loss. Disabled by default for reproducibility, we enable it here.
torch.backends.cuda.matmul.allow_tf32 = True   # enable TF32 for matrix multiply
torch.backends.cudnn.allow_tf32 = True          # enable TF32 for convolutions
torch.cuda.empty_cache()                        # free any leftover GPU memory before we start


# ================================================================
# CONFIGURATION -- Edit these values to change the model/training
# ================================================================

# ARCHITECTURE PARAMETERS
batch_size = 8      # How many samples to train at once.
                    # 8 = good for 4GB GPU. 128 = good for 24GB GPU.
                    # More = faster training but needs more GPU memory.

context = 512       # Sequence length: how many tokens the model sees at once.
                    # Memory for attention scales as O(context^2), so higher = much more memory.
                    # 512 is a good compromise for limited resources.

embed_size = 384    # Embedding dimension: each token is represented as a 384-number vector.
                    # More dimensions = richer representations but slower + more memory.
                    # GPT-2 small uses 768, GPT-3 uses 12288.

n_layers = 7        # Number of stacked Transformer blocks.
                    # More layers = deeper reasoning. GPT-2 small=12, GPT-3=96.
                    # 7 is manageable for a small GPU.

n_heads = 7         # Number of attention heads in each block.
                    # Must divide embed_size evenly: 384 / 7 = 54.9 -> truncated to 54 per head.
                    # More heads = model looks at relationships from more "perspectives".

BIAS = True         # Whether Linear layers use a bias term.
                    # True = standard. False = slightly faster, used in some modern models.

# HYPERPARAMETERS
lr = 3e-4           # Initial learning rate (0.0003). Classic value for transformer training.
                    # Karpathy uses this for nanoGPT. Too high = unstable, too low = slow.

dropout = 0.05      # Dropout fraction: randomly zero out this fraction of neurons each step.
                    # 5% is small because the model itself is small. Prevents overfitting.
                    # Higher values (0.1-0.2) for larger models or smaller datasets.

weight_decay = 0.01 # L2 regularization: penalizes large weights to prevent overfitting.
                    # Applied ONLY to weight matrices (2D+), NOT to biases or LayerNorm params.
                    # 0.01 is the standard value for GPT-style training.

grad_clip = 1.0     # Gradient clipping: cap the gradient vector norm at this value.
                    # Prevents "exploding gradients" that destabilize training.
                    # If total gradient norm > 1.0, scale it down proportionally.

# TRAINING PARAMETERS
train_iters = 100000    # Maximum number of training iterations (gradient update steps).
eval_interval = 50      # How often (in steps) to evaluate performance and print loss.
eval_iters = 3          # Number of batches to average when evaluating loss.
compile = False         # If True: compile model with torch.compile for 10-30% speedup.
                        # Requires PyTorch 2.0+. Disable if you get errors.
load_pretrained = False # If True: load a saved checkpoint before training.

checkpoint_dir = "models/"      # Directory to save checkpoints in.
checkpoint_fn = "latest.pt"     # Filename for checkpoint saved during training.
checkpoint_load_fn = "latest.pt"  # Filename for checkpoint to load when load_pretrained=True.
                                  # Try "llm2.pt" to load a pretrained model at loss ~2.31.

dtype = torch.bfloat16  # Internal data type for the model.
                        # bfloat16 = 16-bit float with same range as float32 but less precision.
                        # Uses 2x less GPU memory and runs faster on modern GPUs.
                        # Range matters more than precision for neural networks.

# MODE
inference = False   # If True: load checkpoint and run in interactive generation mode.
                    # If False: train the model.

# DEVICE -- automatically choose GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# LOGGING -- Weights & Biases experiment tracking (like Application Insights for ML)
wandb_log = False       # Set to True to enable wandb logging. Requires: pip install wandb
wandb_project = "test"
wandb_run_name = "test-run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# FILES
files_url = "https://ideami.com/llm_train"  # Download URL for dataset + tokenizer files
tokenizer_model_file = "wiki_tokenizer.model"  # Path to trained SentencePiece tokenizer


# ================================================================
# SETUP -- Initialize wandb if enabled
# ================================================================
if wandb_log:
    import wandb
    # First time: wandb will ask for your API key (find it at wandb.ai/settings#api)
    wandb.init(project=wandb_project, name=wandb_run_name)


# ================================================================
# DATA -- Download and load training data
# ================================================================
def download_files_if_needed():
    """Download wiki.txt, tokenizer, and encoded_data.pt if not present."""
    if not os.path.exists("encoded_data.pt"):
        print("Downloading files using Python (this may take a while)...")
        response = requests.get(files_url)
        zipfile.ZipFile(io.BytesIO(response.content)).extractall(".")
        print("Download complete.")
    else:
        print("Files already downloaded. (Delete encoded_data.pt to re-download.)")


# ================================================================
# DATA LOADING
# ================================================================

# Load the trained SentencePiece tokenizer
sp = None
vocab_size = None
encode = None
decode = None

def setup_tokenizer():
    global sp, vocab_size, encode, decode
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model_file)
    vocab_size = sp.get_piece_size()           # Get the vocabulary size of our tokenizer
    print(f"Tokenizer vocab_size: {vocab_size}")
    encode = lambda s: sp.Encode(s)            # string -> list of int token IDs
    decode = lambda l: sp.Decode(l)            # list of int token IDs -> string
    print(decode(encode("Encoding Decoding functions ready")))  # test it works


def load_data():
    """Load or create the tokenized dataset, split into train/val."""
    global train_data, val_data

    if os.path.exists("encoded_data.pt"):
        # Load pre-tokenized data if already saved (saves time on re-runs)
        print("Loading saved encoded data...")
        data = torch.load("encoded_data.pt")
    else:
        # Read raw text and tokenize it (slow for large files, so we save the result)
        print("Encoding data (this may take a while)...")
        with open("wiki.txt", "r", encoding="utf-8") as f:
            text = f.read()
        data = torch.tensor(encode(text), dtype=torch.long)  # tensor of token IDs
        torch.save(data, "encoded_data.pt")   # save for next time

    data_size = len(data)  # total number of tokens in the dataset
    spl = int(0.9 * data_size)          # split at 90% train / 10% validation
    train_data = data[:spl]             # first 90% = training data
    val_data = data[spl:]               # last 10% = validation data
    print(
        f"Total data: {data_size/1e6:.2f}M | "
        f"Training: {len(train_data)/1e6:.2f}M | "
        f"Validation: {len(val_data)/1e6:.2f}M tokens"
    )
    return train_data, val_data


# Will be set by load_data()
train_data = None
val_data = None


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def get_batch(split):
    """Return a random batch of input (x) and target (y) tensors.

    For language modeling: targets are inputs shifted by 1 position.
    If x[i] = [T1, T2, T3, T4], then y[i] = [T2, T3, T4, T5].
    The model predicts the NEXT token at each position.
    """
    # BS = Batch Size / SL = Sequence Length (context)
    data = train_data if split == "train" else val_data
    inds = torch.randint(len(data) - context, (batch_size,))   # (BS,) random start positions
    x = torch.stack([data[i: i + context] for i in inds])       # (BS, SL) inputs
    y = torch.stack([data[i + 1: i + context + 1] for i in inds])  # (BS, SL) targets (shifted by 1)
    x, y = x.to(device), y.to(device)  # move to GPU if available
    return x, y


@torch.no_grad()  # decorator: disable gradient calculation for efficiency during evaluation
def calculate_loss():
    """Evaluate average train and validation loss over eval_iters batches.

    @torch.no_grad() means PyTorch does not track operations for backpropagation.
    This saves memory and speeds up the evaluation pass.
    Like a "read-only" mode for the model.
    """
    out = {}
    model.eval()  # switch to evaluation mode (disables dropout)
    for split in ["train", "eval"]:
        l = torch.zeros(eval_iters)        # tensor to collect loss values
        for i in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)          # run model, get loss
            l[i] = loss                    # store this iteration's loss
        out[split] = l.mean().item()       # average loss -> Python float
    model.train()  # switch back to training mode (re-enables dropout)
    return out


@torch.no_grad()
def generate_sample(input_text):
    """Generate new text starting from the given input string.

    Tokenize input -> run model -> sample next token -> repeat.
    """
    t1 = torch.tensor(encode(input_text), dtype=torch.long, device=device)  # tokenize: string -> tensor of IDs
    t1 = t1[None, :]            # add batch dimension: (token_count,) -> (1, token_count)
    newgen = model.generate(t1, max=64)[0].tolist()  # generate up to 64 new tokens, take first batch item
    result = decode(newgen)     # convert token IDs back to text
    print(f"{result}")


def load_checkpoint(path):
    """Load a previously saved training checkpoint.

    Restores: model weights, optimizer state (momentum/variance), iteration number, loss.
    IMPORTANT: We restore optimizer state too, not just model weights.
    AdamW maintains per-parameter momentum and variance estimates.
    Without restoring these, training restarts from scratch momentum-wise.
    """
    print("LLM - Loading model checkpoint...")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])           # restore model weights
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])   # restore optimizer state
    iteration = checkpoint["iteration"]    # what iteration did we save at?
    loss = checkpoint["loss"]              # what was the loss at that point?
    print(f"Loaded iteration {iteration} with loss {loss}")
    return iteration, loss


# ================================================================
# MODEL CLASSES
# ================================================================

class Head(nn.Module):
    """Single attention head.

    Detects and reinforces patterns in relationships between members of sequence.
    Each head learns to focus on DIFFERENT kinds of relationships:
    - one head might learn subject-verb agreement
    - another might learn pronoun references
    - another might learn positional patterns

    In C#, this is like having multiple specialized IComparer implementations,
    each sorting/scoring relationships between tokens differently.
    """
    # BS = Batch Size / SL = Sequence Length

    def __init__(self, head_size):
        super().__init__()
        # Q, K, V projections: each maps from embed_size (384) to head_size (54)
        # These are learnable linear transforms -- like weight matrices in a C# Matrix class
        self.queries = nn.Linear(embed_size, head_size, bias=BIAS)  # Query: "what am I looking for?" (384 -> 54)
        self.keys = nn.Linear(embed_size, head_size, bias=BIAS)     # Key:   "what do I contain?"    (384 -> 54)
        self.values = nn.Linear(embed_size, head_size, bias=BIAS)   # Value: "what do I actually say?" (384 -> 54)

        # Causal mask: lower-triangular matrix of 1s.
        # Position i can only attend to positions 0..i (cannot see future tokens).
        # register_buffer: stores tensor as part of model state (moves to GPU with model.to(device))
        # but is NOT a learnable parameter (not updated by optimizer).
        self.register_buffer("tril", torch.tril(torch.ones(context, context)))  # (SL, SL)

        self.dropout = nn.Dropout(dropout)  # randomly zero out attention weights during training

    def forward(self, x):
        BS, SL, VS = x.shape         # BS=batch, SL=sequence length, VS=embed_size (384)
        q = self.queries(x)          # (BS, SL, 54) -- query vectors for each position
        k = self.keys(x)             # (BS, SL, 54) -- key vectors for each position
        v = self.values(x)           # (BS, SL, 54) -- value vectors for each position

        # Attention weights: how much should each position attend to each other position?
        # q @ k.T = dot product between queries and keys for every pair (i, j)
        # * k.shape[-1]**-0.5 = divide by sqrt(head_size) to prevent softmax saturation.
        # Without scaling, dot products get large -> softmax gives one near-1 and rest near-0
        # -> gradients vanish. Scaling keeps values in a reasonable range.
        attn_w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (BS, SL, SL)

        # Apply causal mask: set upper triangle to -infinity.
        # After softmax, -inf -> 0, so position i cannot attend to positions i+1, i+2...
        # This is the "autoregressive" property: model can only use past context.
        attn_w = attn_w.masked_fill(self.tril[:SL, :SL] == 0, float("-inf"))  # (BS, SL, SL)

        attn_w = F.softmax(attn_w, dim=-1)   # convert to probabilities along each row (BS, SL, SL)
        attn_w = self.dropout(attn_w)         # randomly drop some attention connections during training

        # Weighted sum of value vectors: attended positions contribute more to output
        x = attn_w @ v  # (BS, SL, 54) -- context-enriched representation for each position
        return x


class Multihead(nn.Module):
    """Multi-head attention: runs multiple attention heads in parallel, then combines results.

    Why multiple heads? Each head specializes in different relationship patterns.
    Running them in parallel and combining gives a richer representation.
    Like having multiple domain experts each give an opinion, then taking a weighted vote.
    """

    def __init__(self, n_heads, head_size):
        super().__init__()
        # Create n_heads Head instances -- each is a separate attention mechanism
        # nn.ModuleList: like List<IModule> in C# -- proper parameter registration
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])  # n_heads=7 heads

        # Projection: combine all head outputs back to embed_size
        # head_size * n_heads = 54 * 7 = 378 (slightly less than 384 due to integer division)
        # This projects 378 -> 384 to maintain consistent dimensions through the model
        self.combine = nn.Linear(head_size * n_heads, embed_size, bias=BIAS)  # (378, 384)

        self.dropout = nn.Dropout(dropout)  # regularization after combining heads

    def forward(self, x):
        # BS = Batch Size / SL = Sequence Length
        # x is (BS, SL, 384)
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        # Each head outputs (BS, SL, 54). cat along last dimension -> (BS, SL, 378)
        x = self.combine(x)    # project back to embed_size: (BS, SL, 384)
        x = self.dropout(x)    # apply dropout
        return x


class ForwardLayer(nn.Module):
    """Feed-Forward Network (FFN) within each transformer block.

    After attention "communicates" relationships between tokens,
    the FFN applies computation to each token independently.
    Analogy: attention = group meeting (tokens talk to each other).
             FFN = individual desk work (each token processes what it learned).

    Architecture: Linear -> GELU -> Linear -> Dropout
    Expands by 6x then compresses back: 384 -> 2304 -> 384
    Why 6x? More expansion = more computational capacity. Original paper used 4x.
    Udemy uses 6x for better performance.
    """

    def __init__(self, embed_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_size, 6 * embed_size, bias=BIAS),  # expand: 384 -> 2304
            nn.GELU(),                                           # activation: smooth version of ReLU
            # GELU (Gaussian Error Linear Unit): unlike ReLU which cuts off at 0,
            # GELU has a smooth curve near 0. This helps gradients flow better.
            # Used in GPT-2, BERT, and most modern LLMs. ReLU is older/simpler.
            nn.Linear(6 * embed_size, embed_size, bias=BIAS),  # compress back: 2304 -> 384
            nn.Dropout(dropout)                                  # regularization: drop 5% of neurons
            # Dropout is placed AFTER the compression, before being added back via residual connection
        )

    def forward(self, x):
        x = self.network(x)  # run through expand -> activate -> compress -> dropout
        return x


class Block(nn.Module):
    """A single Transformer block.

    A transformer block combines:
    1. Multi-head attention (communication: tokens share information)
    2. Feed-forward network (computation: each token processes what it learned)

    Both use residual connections: output = input + sub_layer(normalize(input))
    This allows gradients to flow directly through the addition, preventing vanishing gradients
    in deep networks. GPT-3 has 96 of these blocks stacked.

    PRE-NORM design: normalize BEFORE the sub-layer (not after as in the original 2017 paper).
    Pre-norm is more stable for deep networks and used in GPT-2, GPT-3, LLaMA, etc.
    """

    def __init__(self, n_heads):
        super().__init__()
        head_size = embed_size // n_heads   # split embedding dimensions among heads: 384 // 7 = 54
        self.ma = Multihead(n_heads, head_size)   # multi-head attention sub-layer
        self.feed_forward = ForwardLayer(embed_size)  # FFN sub-layer
        self.ln1 = nn.LayerNorm(embed_size)   # normalization before attention
        self.ln2 = nn.LayerNorm(embed_size)   # normalization before FFN

        # LayerNorm normalizes inputs across the features for each data point independently.
        # It subtracts the mean and divides by the standard deviation, followed by scaling and shifting.
        # More computationally intensive than RMSNorm but offers greater flexibility.
        # C# analogy: like normalizing values to a 0-1 range, but per-sample not per-batch.

    def forward(self, x):
        # Pre-norm attention with residual:
        # 1. Normalize x (stabilize inputs to attention)
        # 2. Run through multi-head attention
        # 3. ADD original x (residual connection -- skip-connection)
        x = x + self.ma(self.ln1(x))            # (BS, SL, 384)

        # Pre-norm FFN with residual:
        # Same pattern: normalize -> FFN -> add residual
        x = x + self.feed_forward(self.ln2(x))  # (BS, SL, 384)
        return x


class GPT(nn.Module):
    """The complete GPT language model.

    Architecture:
    Input token IDs -> Token Embedding + Positional Embedding
                    -> 7 Transformer Blocks
                    -> LayerNorm
                    -> Linear (output projection to vocab)
                    -> Softmax -> Sample next token

    19 Million parameters with default configuration.
    Trained on 1 GPU with 4-24 GB memory.

    In C# terms: this is a class that implements a sequence-to-sequence transformer.
    nn.Module is the base class (like inheriting from a framework base class in C#).
    """

    def __init__(self):
        super().__init__()

        # Token embedding: maps each vocabulary ID to a dense vector
        # Like a Dictionary<int, float[]> in C# with vocab_size=4096 entries, each of size embed_size=384
        self.embeddings = nn.Embedding(vocab_size, embed_size)  # (4096, 384) lookup table

        # Positional embedding: maps each position (0 to context-1) to a vector
        # Tells the model WHERE each token is in the sequence (position 0, 1, 2, ... 511)
        # Without this, "dog bites man" and "man bites dog" would look the same (order-blind).
        self.positions = nn.Embedding(context, embed_size)      # (512, 384) lookup table

        # Stack of n_layers transformer blocks, chained sequentially
        # nn.Sequential: like a pipeline -- output of one block is input of next
        # Like IEnumerable<ITransformerBlock> applied in sequence in C#
        self.blocks = nn.Sequential(*[Block(n_heads) for _ in range(n_layers)])  # 7 blocks

        # Final layer normalization before output projection
        self.ln = nn.LayerNorm(embed_size)  # (384,)

        # Output linear layer: maps embed_size back to vocab_size scores (logits)
        # These scores become probabilities after softmax.
        # Like a multi-class classifier with vocab_size=4096 classes in C#.
        self.final_linear = nn.Linear(embed_size, vocab_size, bias=BIAS)  # (384, 4096)

        # Initialize all weights using the _init_weights method
        # self.apply() visits every sub-module recursively and calls _init_weights on each
        # Like a recursive Visitor pattern in C#
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights for stable training.

        WHY special initialization?
        With random weights from standard normal (mean=0, std=1), the variance of
        activations GROWS through each layer. After 7 layers it explodes, causing NaN.
        Using std=0.02 keeps activations in a controlled range at the start of training.
        This is the GPT-2 paper's recommendation.

        WHY zeros for bias?
        Bias starts neutral (no preference). The network learns to shift it during training.
        Starting at zero prevents any bias in initial predictions.
        """
        if isinstance(module, nn.Linear):
            # Initialize weight matrices with normal distribution: mean=0, std=0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Initialize bias parameters to 0 (neutral starting point)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights with normal distribution: mean=0, std=0.02
            # Same reasoning: small weights = stable gradients at initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input, targets=None):
        """Forward pass: given input token IDs, produce logits (and optionally compute loss).

        Shapes (with default config, batch_size=8, context=512, embed_size=384, vocab_size=4096):
        """
        loss = None
        BS, SL = input.shape                                    # (8, 512) -- Batch Size, Sequence Length

        emb = self.embeddings(input)                            # (8, 512, 384) -- token meaning vectors
        pos = self.positions(torch.arange(SL, device=device))  # (512, 384) -- position vectors
        x = emb + pos                                           # (8, 512, 384) -- meaning + position combined

        x = self.blocks(x)                                      # (8, 512, 384) -- through 7 transformer blocks
        x = self.ln(x)                                          # (8, 512, 384) -- normalized
        logits = self.final_linear(x)                           # (8, 512, 4096) -- raw scores for each token

        # Calculate loss if we have targets (during training, not during generation)
        if targets is not None:
            # Cross Entropy loss: measures how wrong our predictions are.
            #
            # Information: -log p(x)  (rare events = more information)
            # Entropy: average information in a distribution: -sum(p(x) * log p(x))
            # CrossEntropy: compares predicted distribution p to true distribution q:
            #               -sum(q(x) * log p(x))
            # For LLMs: true labels are 1 for the correct token, 0 for all others.
            # So cross entropy simplifies to: -log(predicted_probability_of_correct_token)
            # We want to MAXIMIZE the probability of the correct next token.
            # Equivalently: MINIMIZE the negative log probability = cross entropy loss.

            BS, SL, VS = logits.shape                  # (8, 512, 4096)
            logits = logits.view(BS * SL, VS)          # reshape: (4096, 4096) -- needed by F.cross_entropy
            targets = targets.view(BS * SL)             # reshape: (4096,)
            loss = F.cross_entropy(logits, targets)     # compute loss: scalar tensor

        return logits, loss

    def generate(self, input, max=500):
        """Autoregressively generate tokens one at a time.

        At each step:
        1. Run the model on current input (up to context length)
        2. Take logits at the LAST position (the next-token prediction)
        3. Convert to probabilities with softmax
        4. Sample the next token from the probability distribution
        5. Append to input and repeat

        WHY multinomial sampling instead of greedy (argmax)?
        Greedy always picks the most likely next token -> repetitive, boring text.
        Sampling introduces controlled randomness -> more creative, diverse text.
        """
        for _ in range(max):
            input = input[:, -context:]    # keep only last 'context' tokens (sliding window)
            logits, _ = self(input)        # (1, input_len, 4096) -- forward pass
            logits = logits[:, -1, :]      # (1, 4096) -- only last position's predictions
            probs = F.softmax(logits, dim=-1)  # (1, 4096) -- convert to probabilities
            next_token = torch.multinomial(probs, num_samples=1)  # sample one token
            input = torch.cat((input, next_token), dim=1)  # append to sequence
        return input


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train or run a small GPT language model.")
    parser.add_argument("--inference", action="store_true", help="Run in interactive inference mode")
    parser.add_argument("--load", action="store_true", help="Load checkpoint and continue training")
    parser.add_argument("--download", action="store_true", help="Download data files if not present")
    args = parser.parse_args()

    # Override config flags from CLI
    if args.inference:
        inference = True
    if args.load:
        load_pretrained = True

    # Check GPU
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: Running on CPU. Training will be VERY slow.")
        print("Use a GPU (or Google Colab with GPU runtime) for practical training.")

    # Download data if needed
    if args.download:
        download_files_if_needed()

    # Setup tokenizer (must be done before defining model, since we need vocab_size)
    if not os.path.exists(tokenizer_model_file):
        print(f"ERROR: Tokenizer file '{tokenizer_model_file}' not found.")
        print("Run small_tokenizer_standalone.py first, or use --download to fetch pre-trained files.")
        sys.exit(1)

    setup_tokenizer()

    # Load data
    train_data, val_data = load_data()

    # Create the model
    model = GPT()                       # instantiate GPT
    model = model.to(dtype)             # convert to bfloat16
    model = model.to(device)            # move to GPU (or CPU)

    # Optionally compile the model for faster execution (PyTorch 2.0+)
    if compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Print parameter count
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.1f} Million")

    # ----------------------------------------------------------------
    # OPTIMIZER SETUP
    # ----------------------------------------------------------------
    # Separate parameters into two groups: weight matrices and everything else.
    # WHY? Weight decay (L2 regularization) is beneficial for weight matrices
    # to prevent overfitting, but HARMFUL for bias/LayerNorm parameters.
    # Biases are just offsets -- forcing them toward 0 hurts performance.
    #
    # C# analogy: like applying a tax on large stock holdings but not on pocket change.

    # Dictionary of all trainable parameters: {name: parameter_tensor}
    p_dict = {p_name: p for p_name, p in model.named_parameters() if p.requires_grad}

    # 2D+ parameters (weight matrices in Linear layers, Embedding tables):
    # apply weight_decay regularization
    weight_decay_p = [p for n, p in p_dict.items() if p.dim() >= 2]

    # 1D parameters (bias vectors, LayerNorm scale/shift):
    # NO weight decay
    no_weight_decay_p = [p for n, p in p_dict.items() if p.dim() < 2]

    # Two parameter groups with different weight_decay settings
    optimizer_groups = [
        {"params": weight_decay_p, "weight_decay": weight_decay},
        {"params": no_weight_decay_p, "weight_decay": 0.0},
    ]

    # AdamW optimizer:
    # Adam maintains exponential moving averages of gradients (m) and squared gradients (v).
    # AdamW fixes Adam's weight decay implementation -- applies it DIRECTLY to weights,
    # not inside the moment estimates. (See "Decoupled Weight Decay Regularization" paper.)
    #
    # betas=(0.9, 0.99):
    #   beta1=0.9: use 90% of previous gradient direction (momentum)
    #   beta2=0.99: smooth 99% of squared gradient history (variance)
    optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.99))

    # Cosine annealing learning rate scheduler:
    # LR starts at lr (3e-4) and decreases to lr/10 (3e-5) following a cosine curve.
    # WHY cosine? Smooth decay (not abrupt steps). Model makes big updates early in training,
    # and fine-grained adjustments near the end. Cosine decay is used in GPT-3 and most LLMs.
    #
    # LR over training:
    #   Iter 0       : lr = 3e-4    (start)
    #   Iter 50000   : lr = ~1.65e-4 (halfway down cosine curve)
    #   Iter 100000  : lr = 3e-5    (eta_min, minimum lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_iters, eta_min=lr / 10)

    start_iteration = 0
    best_val_loss = float("inf")  # track best validation loss for checkpoint saving

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # OPTIONALLY LOAD A PRETRAINED CHECKPOINT
    # ----------------------------------------------------------------
    if os.path.exists(f"{checkpoint_dir}/{checkpoint_load_fn}") and load_pretrained:
        start_iteration, loss = load_checkpoint(checkpoint_dir + checkpoint_load_fn)
        best_val_loss = loss

    # ----------------------------------------------------------------
    # INFERENCE MODE -- interactive generation, no training
    # ----------------------------------------------------------------
    if inference:
        model.eval()  # disable dropout for inference
        print("\n=== INFERENCE MODE ===")
        print("Type text to continue, or 'q' to quit.\n")
        while True:
            qs = input("Enter text (q to quit) >>> ")
            if qs == "":
                continue
            if qs == "q":
                break
            generate_sample(qs)
        # Exit after inference -- do not proceed to training loop
        sys.exit(0)

    # ----------------------------------------------------------------
    # TRAINING LOOP
    # ----------------------------------------------------------------
    print(f"\n=== STARTING TRAINING ===")
    print(f"Training for {train_iters} iterations, evaluating every {eval_interval} steps.")
    print(f"Checkpoints saved to: {checkpoint_dir}{checkpoint_fn}")

    try:
        for i in tqdm(range(start_iteration, train_iters)):

            # Get a fresh batch of training data
            xb, yb = get_batch("train")

            # Forward pass: run the model, compute loss
            logits, loss = model(xb, yb)

            # Evaluate and print progress periodically
            if i % eval_interval == 0 or i == train_iters - 1:
                l = calculate_loss()
                print(f"\nStep {i}: train loss={l['train']:.4f} | val loss={l['eval']:.4f}")

                # Quick generation test -- watch the model improve over training
                generate_sample("The mountain in my city is")

                # Save checkpoint if validation loss improved
                if l["eval"] < best_val_loss:
                    best_val_loss = l["eval"]
                    print(f"[CHECKPOINT] Saving with loss: {best_val_loss:.4f}")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),           # all model weights
                            "optimizer_state_dict": optimizer.state_dict(),   # optimizer momentum/variance
                            "loss": best_val_loss,                            # best loss achieved
                            "iteration": i,                                   # training step
                        },
                        checkpoint_dir + checkpoint_fn,
                    )

                # Log to Weights & Biases if enabled
                if wandb_log:
                    wandb.log(
                        {
                            "loss/train": l["train"],
                            "loss/val": l["eval"],
                            "lr": scheduler.get_last_lr()[0],
                        },
                        step=i,
                    )

            # Backward pass and weight update:

            # 1. Zero out gradients from previous step
            #    set_to_none=True is faster than setting to zero
            optimizer.zero_grad(set_to_none=True)

            # 2. Backpropagate: compute gradients of loss w.r.t. all parameters
            loss.backward()

            # 3. Gradient clipping: prevent exploding gradients
            #    If the total gradient norm exceeds grad_clip (1.0), scale all gradients down.
            #    This stabilizes training, especially for large models or large learning rates.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            # 4. Update model parameters using computed gradients
            optimizer.step()

            # 5. Update learning rate according to cosine annealing schedule
            scheduler.step()

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving state...")

    finally:
        # Always release GPU memory when done (even if interrupted)
        torch.cuda.empty_cache()
        print("GPU memory released.")

    # Clean up wandb
    if wandb_log:
        wandb.finish()

    torch.cuda.empty_cache()
    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {checkpoint_dir}{checkpoint_fn}")

# Code based on Udemy course by Javier Ideami (ideami.com)
# Adapted for standalone Python by the Learn LLM from Scratch course

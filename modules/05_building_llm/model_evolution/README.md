# Model Evolution Lab

**Start with a 10-line bigram model. End with a Mini-Qwen equivalent.**

This folder takes you on a hands-on journey through the evolution of language models — from the simplest possible character predictor (a lookup table!) all the way to the modern architecture used by LLaMA, Mistral, and Qwen. Every step adds exactly ONE new idea, so you can see what each concept actually contributes.

---

## Who this is for

You are a .NET / C# developer learning Python and LLMs simultaneously. Every file in this lab:
- Comments **every single line** of code
- Gives a **C# analogy** for every new concept
- Prints a clear **before/after training** sample so you can see learning happen
- Runs in **5-10 minutes on CPU** — no GPU needed

---

## Quick Start

```
Step 1: Install dependencies
    pip install torch datasets requests

Step 2: Download training data (~10MB)
    python download_data.py

Step 3: Run each step in order
    python step_00_bigram.py
    python step_01_mlp_context.py
    python step_02_single_attention.py
    python step_03_multi_head.py
    python step_04_gpt_nano.py
    python step_05_gqa.py
    python step_06_kv_cache.py
    python step_07_rope.py
```

**Always run from this folder:**
```
cd modules/05_building_llm/model_evolution
python download_data.py
```

---

## The Evolution Table

| Step | File | What's New | Expected Val Loss | CPU Train Time |
|------|------|------------|-------------------|----------------|
| 0 | `step_00_bigram.py` | Lookup table only — no context | ~2.4 | 1 min |
| 1 | `step_01_mlp_context.py` | Context window (16 chars) | ~2.0 | 3 min |
| 2 | `step_02_single_attention.py` | Single-head attention (Q, K, V) | ~1.7 | 5 min |
| 3 | `step_03_multi_head.py` | 4 attention heads + FeedForward | ~1.5 | 7 min |
| 4 | `step_04_gpt_nano.py` | 3 stacked blocks + LayerNorm + residual | ~1.3 | 10 min |
| 5 | `step_05_gqa.py` | Group-Query Attention (50% less KV memory) | ~1.3 | 10 min |
| 6 | `step_06_kv_cache.py` | KV Cache (5-10x faster generation) | ~1.3 | 10 min |
| 7 | `step_07_rope.py` | RoPE positional embeddings | ~1.2 | 10 min |

Lower val loss = better model. Watch it improve as you add each concept!

---

## File Descriptions

### `download_data.py`
Downloads ~10MB of training text to `data/corpus.txt`. Tries HuggingFace WikiText-103 first; falls back to Project Gutenberg (Shakespeare + War and Peace) if the `datasets` library isn't installed. Run this once before any step file.

### `shared.py`
A utility module imported by all step files. Contains: data loading, vocabulary building, encoding/decoding text to integers, batch sampling, loss estimation, text generation, and the training loop. You do not run this directly — it is a helper library.

### `step_00_bigram.py`
The simplest possible model: a lookup table. Given one character, predict the next. No memory of anything else. Uses a single `nn.Embedding(vocab_size, vocab_size)` — that is literally it.

### `step_01_mlp_context.py`
Adds a context window of 16 characters. Embeds all 16, concatenates them into a flat vector, runs through two Linear layers with ReLU. Shows that context matters enormously even with a simple MLP.

### `step_02_single_attention.py`
The most important conceptual step. Introduces the Query-Key-Value attention mechanism: each token asks "which other tokens are relevant to me?" and gathers their information using learned relevance scores. Also introduces positional embedding.

### `step_03_multi_head.py`
Runs 4 attention heads in parallel — each learning a different type of relationship (syntax, semantics, reference, position). Adds a FeedForward layer after attention for per-token processing.

### `step_04_gpt_nano.py`
The full GPT architecture in miniature. Stacks 3 TransformerBlocks, adds residual connections (prevents vanishing gradients), layer normalization (stabilizes training), and dropout (prevents overfitting). This IS GPT-2 architecture — just smaller numbers.

### `step_05_gqa.py`
Group-Query Attention: use fewer K,V heads than Q heads. Multiple Q heads share the same K and V. Saves KV cache memory at inference time with minimal quality loss. Used by LLaMA 2 70B, LLaMA 3, Mistral, Qwen.

### `step_06_kv_cache.py`
Makes generation fast. Without cache: O(T²) — recomputes all K,V every step. With cache: O(T) — compute K,V once per token, reuse forever. The benchmark at the end shows the speedup in tokens/sec.

### `step_07_rope.py`
Replaces the learned positional embedding with Rotary Positional Embeddings. Instead of adding a position vector to token embeddings, RoPE rotates the Query and Key vectors by position-dependent angles. The dot product Q·K then naturally encodes relative position. Better extrapolation beyond training length. The final model is architecturally equivalent to modern open-source LLMs.

---

## How to Read Each File

Every step file follows the same structure:

```python
# ============================================================
# Big comment block at the top explaining:
#   - What is new in this step
#   - Why it matters
#   - C# analogy
#   - ASCII architecture diagram
# ============================================================

CONFIG = { ... }           # All hyperparameters in one place — easy to tweak

class SomeLayer(nn.Module):
    """Docstring with explanation"""
    def __init__(self, ...):
        # Comment on EVERY line
        ...
    def forward(self, x, ...):
        # Comment on EVERY line
        ...

class FullModel(nn.Module):
    ...

if __name__ == "__main__":
    # This block only runs when you execute the file directly
    # (not when another file imports it)
    # Prints a banner, trains, benchmarks, prints next step
```

### The `if __name__ == "__main__":` Pattern

In Python, every `.py` file is a "module". The variable `__name__` is:
- `"__main__"` when you run the file directly: `python step_04_gpt_nano.py`
- `"step_04_gpt_nano"` when another file imports it: `import step_04_gpt_nano`

This pattern prevents training from starting accidentally when a file is imported. It is the Python equivalent of `static void Main(string[] args)` in C#.

---

## What Each Concept Teaches

**Bigram (step 0):** The absolute baseline. Bigram = two characters. The model only uses the current character to predict the next. No history, no context. Despite being trivial, it learns something — some character pairs are far more common than others.

**MLP Context (step 1):** Context is everything in language. "t" alone could be followed by anything; "th" is almost certainly followed by "e" or "i". The MLP concatenates the last 16 character embeddings and processes them together. A direct demonstration of why context windows exist.

**Single Attention (step 2):** The core idea of all modern LLMs. Instead of treating all context positions equally (MLP does this), attention learns which past tokens are relevant *right now*. A noun 10 positions back might be critical for a pronoun resolution; attention can learn to focus there. The Query-Key-Value framework gives every token a way to "ask" about and "answer" other tokens.

**Multi-Head Attention (step 3):** Language has multiple types of structure simultaneously. One head might track syntax (which word is the subject), another tracks topic (is this about technology or nature), another tracks coreference (what does "it" refer to). Four parallel attention computations — each with different learned weights — gives the model four different perspectives on the same input.

**NanoGPT (step 4):** Stacking transformer blocks is how you get a deep model. But deeper networks suffer from vanishing gradients — signals fade as they travel backward through many layers. Residual connections (adding the input to the output: `x = x + transform(x)`) solve this by providing a direct gradient highway. Layer normalization keeps activation values from exploding or collapsing. This is the real GPT architecture.

**Group-Query Attention (step 5):** At inference time, fast generation requires storing all past K and V tensors in a "KV cache" (see step 6). For a 70-billion-parameter model with 80 attention heads, that is an enormous amount of memory. GQA reduces this by having multiple Query heads share the same Key and Value heads. LLaMA 2 70B uses 8 KV heads for 64 Query heads — an 8× reduction in KV cache size.

**KV Cache (step 6):** Without cache, generating token T requires recomputing K and V for all T past tokens — that is O(T²) total work. With cache, you compute K and V exactly once per token and store them. Generating token T requires only computing K and V for the one new token and a single matrix multiply. Total work is O(T). Every production LLM uses this — it is not optional for real-world deployment.

**RoPE (step 7):** Learned positional embeddings (steps 2-6) have two problems: they break at sequences longer than block_size, and they only learn absolute positions (not relative distances). RoPE encodes position by rotating Q and K vectors. Because rotation is a mathematical operation, the dot product Q·K naturally depends on the *relative* distance between tokens — without any explicit training for relative positions. LLaMA, Mistral, Qwen, Gemma, and Phi all use RoPE.

---

## Connection to Real Models

| This Lab | Real Model | Scale Difference |
|---|---|---|
| `step_04_gpt_nano.py` | GPT-2 small (124M params) | Same architecture, 100× bigger |
| `step_04_gpt_nano.py` | GPT-3 (175B params) | Same architecture, 100,000× bigger |
| `step_05_gqa.py` | LLaMA 2 70B | GQA: 64 Q heads, 8 KV heads |
| `step_05_gqa.py` | Mistral 7B | GQA: 32 Q heads, 8 KV heads |
| `step_06_kv_cache.py` | All production LLMs | Every deployed LLM uses KV cache |
| `step_07_rope.py` | LLaMA 2/3 | RoPE base=10000, exact same formula |
| `step_07_rope.py` | Qwen2 | RoPE + GQA + SwiGLU (next step up) |
| `step_07_rope.py` | Gemma 2 | RoPE + GQA + interleaved local/global attention |

The only differences between your `step_07_rope.py` and LLaMA 3 8B are:
- LLaMA uses 32 layers (you use 3)
- LLaMA uses n_embd=4096 (you use 128)
- LLaMA uses SwiGLU activation instead of ReLU
- LLaMA uses RMSNorm instead of LayerNorm
- LLaMA uses BPE tokenization (subwords) instead of character-level

The core logic — GQA, RoPE, residual connections, stacked transformer blocks — is identical.

---

## Troubleshooting

**`FileNotFoundError: data/corpus.txt not found`**
Run `python download_data.py` first. The data file must exist before any step file runs.

**`ModuleNotFoundError: No module named 'torch'`**
Install PyTorch: `pip install torch`
If that fails on Windows: visit https://pytorch.org and use the install selector.

**`ModuleNotFoundError: No module named 'shared'`**
You must run step files from the `model_evolution/` directory:
```
cd modules/05_building_llm/model_evolution
python step_00_bigram.py
```

**`UnicodeDecodeError` during download**
The download functions use `errors="replace"` to handle encoding issues automatically. If you still see this, try deleting `data/corpus.txt` and running `python download_data.py` again.

**Training loss does not decrease**
- Check that `data/corpus.txt` is not empty: it should be several MB
- Try increasing `max_iters` in CONFIG (e.g., from 3000 to 5000)
- Make sure you are running the right file and not an earlier step

**Training is very slow**
These configs are tuned for CPU. On a modern CPU (Intel i5/i7, Apple M1+):
- Steps 0-1: should take under 3 minutes
- Steps 2-7: should take 5-12 minutes each
If it takes much longer, reduce `max_iters` in CONFIG at the top of the file.

**`CUDA out of memory`**
The configs are sized for CPU. If you have a GPU but it is small, change `device` in CONFIG:
```python
CONFIG = { ..., "device": "cpu" }
```

**Generated text looks like random characters**
This is expected early in training. Sample BEFORE training = pure gibberish (that is the point — to show the baseline). After training, you should see recognizable character patterns (English-ish words) even if they do not make full sense.

**`assert n_heads % n_kv_heads == 0` error (step 5-7)**
The number of KV heads must divide evenly into the number of Q heads.
`N_HEADS=4, N_KV_HEADS=2` works (4 % 2 == 0).
`N_HEADS=4, N_KV_HEADS=3` does not work (4 % 3 != 0).

---

## Key Vocabulary

| Term | Plain English | C# Analogy |
|---|---|---|
| **Token** | One unit of input (one character in our case) | One `char` element |
| **Vocabulary** | Set of all unique tokens | `HashSet<char>` |
| **Embedding** | Learned vector representation of a token | `float[]` lookup in a `Dictionary<char, float[]>` |
| **Logits** | Raw unnormalized scores before softmax | Scores before `.Normalize()` |
| **Softmax** | Convert scores to probabilities (sum to 1) | Normalize a `float[]` so values sum to 1.0 |
| **Cross-entropy loss** | How wrong the model is | `-Math.Log(probability_of_correct_answer)` |
| **Batch** | Multiple training examples processed together | `Parallel.For` over N work items |
| **Block size** | Context window size (how many tokens the model sees) | Input array length |
| **Forward pass** | Running data through the model | Calling a `Func<Tensor, Tensor>` |
| **Backward pass** | Computing gradients (which way to adjust weights) | Automatic differentiation |
| **Gradient** | Direction to change weights to reduce loss | Derivative of the loss function |
| **Learning rate** | How big each weight update step is | Step size in gradient descent |
| **Residual** | Adding input to output: `x = x + transform(x)` | `x += transform(x)` (the `+=` operator) |
| **Layer norm** | Normalize each token's vector to mean=0, std=1 | Normalize a `float[]` to standard deviation 1 |
| **Dropout** | Randomly zero out neurons during training | Randomly skip 10% of code paths |
| **KV cache** | Store past Keys and Values to avoid recomputation | `Dictionary<int, (K, V)>` memoization |
| **RoPE** | Encode position by rotating Q and K vectors | Multiplying by a rotation matrix indexed by position |
| **GQA** | Multiple Q heads sharing fewer K,V heads | `static readonly` K,V shared across head instances |

# Lesson 02: GGUF Format and llama.cpp

## Glossary (Read This First!)

| Term | Plain English Definition |
|------|--------------------------|
| **GGUF** | GPT-Generated Unified Format. A file format for storing quantized LLMs. Successor to GGML. |
| **llama.cpp** | An open-source C++ library that runs LLMs efficiently on CPU and GPU. Created by Georgi Gerganov in 2023. |
| **GGML** | The older format before GGUF. You may see it mentioned in old tutorials. It is replaced by GGUF. |
| **llama-cpp-python** | Python bindings for llama.cpp. Lets you call llama.cpp from Python. |
| **Tensor** | A multi-dimensional array. The weight matrices in an LLM are tensors. |
| **Metadata** | Data about data. GGUF files store model metadata (name, architecture, vocab, etc.) alongside the weights. |
| **K-quant** | A family of GGUF quantization methods that use different precisions for different parts of the model. |
| **Memory mapping (mmap)** | A technique to load only the parts of a file you need, rather than the whole file. GGUF supports this. |
| **Context window** | The maximum number of tokens the model can see at once. 4K, 8K, 32K, 128K are common sizes. |
| **KV-cache** | Key-Value cache. Stores intermediate computations so the model does not re-compute them for each new token. |
| **Prompt template** | The specific format a model was trained to expect. Different models need different templates. |
| **Token** | The atomic unit an LLM works with. A word, a sub-word, or a character, depending on the tokenizer. |
| **Tokenizer** | Converts text to tokens (integers) for the model, and tokens back to text for output. |
| **Inference engine** | Software that runs a model to produce output. llama.cpp is an inference engine. |

---

## Part 1: The Problem GGUF Solves

Before GGUF existed (early 2023), sharing and running quantized LLMs was a mess.

```
+------------------------------------------------------------------+
|  THE PROBLEM BEFORE GGUF                                         |
+------------------------------------------------------------------+
|                                                                  |
|  Multiple incompatible formats:                                  |
|   - PyTorch .pt / .bin files (huge, fp32 or fp16 only)          |
|   - HuggingFace safetensors (better than .pt, still not tiny)   |
|   - GGML (early llama.cpp format, no metadata, fragile)         |
|   - Various custom INT8/INT4 formats (each tool different)       |
|                                                                  |
|  Problems:                                                       |
|   - No standard way to store quantized weights                   |
|   - Tokenizer stored separately, easy to mismatch               |
|   - No model metadata -- you had to know the architecture        |
|   - Updating the format broke all existing models (GGML issue)  |
|   - Could not partially load the file                            |
|                                                                  |
|  GGUF fixed all of this.                                         |
+------------------------------------------------------------------+
```

---

## Part 2: What Is Inside a GGUF File?

A GGUF file is self-contained. It has everything needed to run the model.

```
+==================================================================+
|  GGUF FILE STRUCTURE                                             |
+==================================================================+
|                                                                  |
|  [1] MAGIC NUMBER + VERSION                                      |
|      4 bytes: "GGUF"                                             |
|      4 bytes: version number                                     |
|      Purpose: identify this as a valid GGUF file                |
|                                                                  |
+==================================================================+
|                                                                  |
|  [2] METADATA (KEY-VALUE STORE)                                  |
|      Number of metadata entries                                  |
|      For each entry:                                             |
|        - key (string, e.g., "llama.context_length")              |
|        - type (e.g., uint32, string, array)                      |
|        - value (e.g., 4096)                                      |
|                                                                  |
|      Examples of metadata stored:                                |
|        general.architecture = "llama"                            |
|        general.name = "LLaMA v2 7B"                              |
|        llama.context_length = 4096                               |
|        llama.embedding_length = 4096                             |
|        llama.block_count = 32                                    |
|        llama.attention.head_count = 32                           |
|        tokenizer.ggml.model = "llama"                            |
|        tokenizer.ggml.tokens = [list of all 32000 tokens]       |
|        tokenizer.ggml.scores = [token frequencies]              |
|        tokenizer.chat_template = "<s>[INST]..."                  |
|                                                                  |
+==================================================================+
|                                                                  |
|  [3] TENSOR INDEX                                                |
|      List of all tensors in the model                            |
|      For each tensor:                                            |
|        - name (e.g., "blk.0.attn_q.weight")                     |
|        - shape (e.g., [4096, 4096])                              |
|        - quantization type (e.g., Q4_K)                         |
|        - file offset (where in the file to find the data)        |
|                                                                  |
+==================================================================+
|                                                                  |
|  [4] TENSOR DATA                                                 |
|      The actual quantized weight bytes, back-to-back.            |
|      Each tensor's bytes are aligned to 32 bytes for fast access.|
|                                                                  |
+==================================================================+
```

### Why This Structure Is Smart

The key insight: **the tensor index comes BEFORE the data.**

This means llama.cpp can:
1. Read just the metadata + index (fast, small)
2. Know exactly where every tensor is in the file
3. Load only the tensors it needs right now (memory mapping)
4. Not load the whole 4 GB file into RAM at startup

C# analogy:
```csharp
// GGUF is like a well-organized ZIP file with a manifest:
//
// ZIP (traditional):
//   You must decompress everything to find anything.
//   Read all bytes sequentially.
//
// GGUF:
//   Read the header (small, fast).
//   See exactly where "blk.5.attn_q.weight" is at byte offset 1,234,567.
//   Jump directly there. Read just that tensor. Done.
//
// This is like a database index vs. a full table scan.
// The index (GGUF header) makes random access fast.
```

---

## Part 3: GGUF Quantization Types Explained

GGUF supports many quantization types. Here are the ones you will encounter:

```
+------------------------------------------------------------------+
|  GGUF QUANTIZATION TYPE REFERENCE                                |
+------------------------------------------------------------------+
|                                                                  |
|  Q4_0  (4-bit, simple)                                           |
|    - Each block of 32 weights: 1 fp16 scale + 32 INT4 weights    |
|    - Oldest format, smallest size, lower quality                  |
|    - Memory: ~2.3 GB for 7B model                               |
|                                                                  |
|  Q4_1  (4-bit, with min value)                                   |
|    - Like Q4_0 but adds a minimum value offset per block         |
|    - Slightly larger, handles asymmetric distributions better    |
|    - Memory: ~2.5 GB for 7B model                               |
|                                                                  |
|  Q4_K_S  (4-bit, K-quant, small)                                 |
|    - K-quant: uses 6-bit quantization for attention layers       |
|    - Other layers: 4-bit                                         |
|    - Better quality than Q4_0 at similar size                    |
|    - Memory: ~2.5 GB for 7B model                               |
|                                                                  |
|  Q4_K_M  (4-bit, K-quant, medium) -- RECOMMENDED FOR MOST USES  |
|    - Uses 8-bit quantization for some layers                     |
|    - Best quality-to-size tradeoff in 4-bit range                |
|    - Memory: ~2.7 GB for 7B model                               |
|                                                                  |
|  Q5_K_M  (5-bit, K-quant, medium)                               |
|    - 5 bits per weight (not common, but good middle ground)      |
|    - Quality between Q4_K_M and Q8_0                            |
|    - Memory: ~3.4 GB for 7B model                               |
|                                                                  |
|  Q8_0  (8-bit, simple)                                           |
|    - Each block of 32 weights: 1 fp32 scale + 32 INT8 weights    |
|    - Very close to fp16 quality                                  |
|    - Memory: ~7.2 GB for 7B model                               |
|                                                                  |
|  F16  (16-bit float, no quantization)                            |
|    - Pure fp16. No integer quantization.                         |
|    - Best quality. Largest size.                                 |
|    - Memory: ~13.5 GB for 7B model                              |
|                                                                  |
+------------------------------------------------------------------+

QUICK DECISION GUIDE:
  Have < 6 GB RAM?  --> Q4_K_M
  Have 8 GB RAM?    --> Q4_K_M or Q5_K_M
  Have 16 GB RAM?   --> Q8_0 or F16
  Quality critical? --> Q8_0
```

### What "K-quant" Means

K-quant (K for "quantization type K") is a smarter quantization strategy:

```
+------------------------------------------------------------------+
|  K-QUANT KEY INSIGHT                                             |
+------------------------------------------------------------------+
|                                                                  |
|  Observation: Not all layers in an LLM are equally sensitive.    |
|                                                                  |
|  SENSITIVE LAYERS (use higher precision):                        |
|    - Attention Q, K, V weight matrices                           |
|    - First and last layers of the network                        |
|    - Embedding layer                                             |
|                                                                  |
|  LESS SENSITIVE LAYERS (can use lower precision):                |
|    - Feed-forward network layers in middle blocks                |
|    - Many of the projection layers                               |
|                                                                  |
|  K-QUANT STRATEGY:                                               |
|    Sensitive layers:     use 6-bit or 8-bit                      |
|    Other layers:         use 4-bit                               |
|    Result: Better quality at almost the same size as naive Q4.   |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 4: Where to Get GGUF Files

The main source is HuggingFace, in repositories maintained by Bartowski and TheBloke.

```
+------------------------------------------------------------------+
|  WHERE TO FIND GGUF FILES                                        |
+------------------------------------------------------------------+
|                                                                  |
|  HuggingFace (huggingface.co):                                   |
|    Search: "model-name GGUF"                                     |
|    Popular uploaders:                                            |
|      - bartowski/Llama-3.2-1B-Instruct-GGUF                     |
|      - bartowski/Mistral-7B-Instruct-v0.3-GGUF                  |
|      - TheBloke (older models, still widely used)                |
|                                                                  |
|  Ollama Library (ollama.com/library):                            |
|    Curated list of ready-to-use models                           |
|    Ollama downloads GGUF files automatically                     |
|                                                                  |
|  FILE NAMING PATTERN:                                            |
|    {model-name}-{version}-{quant}.gguf                           |
|    Examples:                                                     |
|      Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf                     |
|      mistral-7b-instruct-v0.3-Q8_0.gguf                         |
|      phi-3-mini-4k-instruct-Q4_K_M.gguf                         |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 5: llama.cpp -- What It Is and Why It Exists

```
+------------------------------------------------------------------+
|  LLAMA.CPP ORIGIN STORY                                          |
+------------------------------------------------------------------+
|                                                                  |
|  January 2023: Meta released LLaMA model weights.                |
|  January 2023 (4 days later): Georgi Gerganov released           |
|                               llama.cpp on GitHub.               |
|                                                                  |
|  What Gerganov did:                                              |
|  - Took the LLaMA architecture (Python/PyTorch)                  |
|  - Rewrote it in pure C++ with no dependencies                   |
|  - Added quantization (INT4, INT8)                               |
|  - Optimized for CPU inference using BLAS (fast linear algebra)  |
|  - Added Apple Metal support (run on Mac GPU for free)           |
|  - Added CUDA support (optional NVIDIA GPU acceleration)         |
|                                                                  |
|  RESULT:                                                         |
|  - LLaMA-7B running at 10-20 tokens/second on a 2019 MacBook.   |
|  - No Python. No PyTorch. No cloud. No internet.                 |
|  - Just C++ and a GGUF file.                                     |
|                                                                  |
|  Today llama.cpp supports 50+ model architectures.               |
|  It is the foundation of Ollama (which is a wrapper around it).  |
+------------------------------------------------------------------+
```

### llama.cpp Architecture

```
+------------------------------------------------------------------+
|  HOW LLAMA.CPP WORKS INTERNALLY                                  |
+------------------------------------------------------------------+
|                                                                  |
|  1. Load GGUF file                                               |
|     Read metadata: context size, architecture, tokenizer.        |
|     Memory-map the weight data (don't load all into RAM yet).   |
|                                                                  |
|  2. Tokenize input                                               |
|     The tokenizer (BPE/SentencePiece) converts text to integers. |
|     "Hello world" -> [15043, 3186] (token IDs)                   |
|                                                                  |
|  3. Allocate KV-cache                                            |
|     Pre-allocate memory for key-value pairs.                     |
|     Size = context_length x num_layers x 2 x head_size           |
|     For 7B model at 4K context: ~1 GB                            |
|                                                                  |
|  4. Forward pass (per token generated)                           |
|     Load relevant weight tensors (dequantize INT4 -> fp16).      |
|     Compute attention using KV-cache.                            |
|     Compute feed-forward layers.                                  |
|     Get logits (probability distribution over all tokens).       |
|                                                                  |
|  5. Sampling                                                      |
|     Pick next token from logits.                                 |
|     (Greedy: highest probability. Temperature: add randomness.)  |
|                                                                  |
|  6. Detokenize                                                    |
|     Convert token ID back to text. Stream it to output.          |
|                                                                  |
|  7. Repeat steps 4-6 until stop token or max length              |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 6: Running a Model with llama-cpp-python

The `llama-cpp-python` package wraps llama.cpp in Python.

### Installation

```bash
# Basic install (CPU only):
pip install llama-cpp-python

# With Metal (Apple Silicon GPU acceleration):
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# With CUDA (NVIDIA GPU):
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Note: compilation takes 2-5 minutes. This is normal.
```

### Basic Usage

```python
from llama_cpp import Llama

# Load the model from a GGUF file
# n_ctx = context window size (how many tokens it can "see")
# n_threads = CPU threads to use (use your physical core count)
# n_gpu_layers = how many layers to run on GPU (0 = CPU only)
model = Llama(
    model_path="./models/mistral-7b-instruct-v0.3-Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=0
)

# Simple completion (raw text continuation)
output = model(
    "The capital of France is",
    max_tokens=50,
    stop=["\n"],
    echo=True
)

print(output["choices"][0]["text"])
# -> "The capital of France is Paris, which..."

# Chat completion (instruction-following format)
response = model.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"}
    ],
    max_tokens=100
)

print(response["choices"][0]["message"]["content"])
# -> "2 + 2 equals 4."
```

### Streaming Output

```python
from llama_cpp import Llama

model = Llama(
    model_path="./models/mistral-7b-instruct-Q4_K_M.gguf",
    n_ctx=4096
)

# stream=True makes output appear token by token (like ChatGPT)
stream = model.create_chat_completion(
    messages=[{"role": "user", "content": "Tell me a short joke."}],
    max_tokens=100,
    stream=True
)

# Print each token as it arrives, without newline
for chunk in stream:
    delta = chunk["choices"][0]["delta"]
    if "content" in delta:
        print(delta["content"], end="", flush=True)

print()  # final newline
```

---

## Part 7: Important Parameters Explained

```
+------------------------------------------------------------------+
|  KEY PARAMETERS WHEN LOADING A GGUF MODEL                        |
+------------------------------------------------------------------+
|                                                                  |
|  n_ctx (context window)                                          |
|    Default: 512                                                  |
|    What it does: Sets how many tokens the model can "remember"   |
|    Tradeoff: Larger = more memory (KV-cache grows linearly)      |
|    Typical values: 2048, 4096, 8192                              |
|    Rule: Set to what your use case needs. Bigger costs RAM.      |
|                                                                  |
|  n_threads (CPU threads)                                         |
|    Default: OS default (often 1)                                 |
|    What it does: Parallelizes matrix operations across CPU cores  |
|    Rule: Set to your physical core count, not logical (no HT).   |
|    Example: 8-core CPU -> n_threads=8                            |
|                                                                  |
|  n_gpu_layers (GPU offloading)                                   |
|    Default: 0 (CPU only)                                         |
|    What it does: Offloads N layers to GPU VRAM                   |
|    Rule: Set to max (-1) if model fits in VRAM                   |
|    n_gpu_layers=32 means "offload 32 transformer blocks to GPU"  |
|    Mixed: some layers on GPU, remaining on CPU (if VRAM is low)  |
|                                                                  |
|  n_batch (tokens processed at once)                              |
|    Default: 512                                                  |
|    What it does: How many tokens are processed in parallel       |
|    Affects: Speed of prompt processing (prefill phase)           |
|    Rule: Larger = faster prefill, more memory.                   |
|                                                                  |
|  temperature (sampling temperature)                              |
|    Default: 0.8                                                  |
|    Range: 0.0 to 2.0                                             |
|    0.0 = deterministic (always picks most likely token)          |
|    1.0 = standard randomness                                     |
|    2.0 = very random / creative (often incoherent)              |
|                                                                  |
+------------------------------------------------------------------+
```

C# analogy:
```csharp
// n_ctx is like the size of a sliding window buffer in C#:
//
// var contextWindow = new CircularBuffer<Token>(capacity: 4096);
// When full, oldest token is dropped to make room for new ones.
//
// n_threads is like Parallel.For with a degree-of-parallelism:
// var options = new ParallelOptions { MaxDegreeOfParallelism = 8 };
//
// temperature is like injecting randomness into selection:
// temperature = 0 -> always pick best option (greedy)
// temperature > 0 -> sample from probability distribution
// Higher temperature = more randomness = more "creative" output
```

---

## Part 8: Prompt Templates -- The Hidden Gotcha

Different models expect different prompt formats.
Using the wrong format gives bad results even with a good model.

```
+------------------------------------------------------------------+
|  PROMPT TEMPLATE COMPARISON                                      |
+------------------------------------------------------------------+
|                                                                  |
|  MODEL: LLaMA-2-Chat                                             |
|  Template:                                                       |
|    <s>[INST] <<SYS>>                                             |
|    {system_message}                                              |
|    <</SYS>>                                                      |
|    {user_message} [/INST]                                        |
|                                                                  |
|  MODEL: Mistral-Instruct                                         |
|  Template:                                                       |
|    <s>[INST] {user_message} [/INST]                              |
|                                                                  |
|  MODEL: ChatML (used by many models)                             |
|  Template:                                                       |
|    <|im_start|>system                                            |
|    {system_message}<|im_end|>                                    |
|    <|im_start|>user                                              |
|    {user_message}<|im_end|>                                      |
|    <|im_start|>assistant                                         |
|                                                                  |
|  MODEL: LLaMA-3-Instruct                                         |
|  Template:                                                       |
|    <|begin_of_text|><|start_header_id|>system<|end_header_id|>   |
|    {system_message}<|eot_id|><|start_header_id|>user             |
|    <|end_header_id|>{user_message}<|eot_id|>                     |
|    <|start_header_id|>assistant<|end_header_id|>                 |
|                                                                  |
+------------------------------------------------------------------+

GOOD NEWS: llama-cpp-python's create_chat_completion() handles this
           automatically when using models with a chat_template in their
           GGUF metadata (most modern models).

WHEN YOU NEED TO CARE: Only when using raw model() completions
                        (not create_chat_completion()).
```

---

## Part 9: GGUF File Size Verification

Before loading a model, verify it downloaded correctly:

```python
import os
import hashlib

def check_gguf_file(filepath):
    """Check that a GGUF file looks valid before loading."""

    # Check file exists
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return False

    # Check file size (at minimum a few hundred MB for any real model)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

    if size_mb < 100:
        print("WARNING: File seems too small. May be corrupted or incomplete.")
        return False

    # Check GGUF magic number (first 4 bytes should be "GGUF")
    with open(filepath, "rb") as f:
        magic = f.read(4)

    if magic == b"GGUF":
        print("Magic number: OK (valid GGUF file)")
        return True
    else:
        print(f"ERROR: Invalid magic number: {magic}. Not a GGUF file.")
        return False

# Usage:
check_gguf_file("./models/mistral-7b-instruct-Q4_K_M.gguf")
```

---

## Part 10: Reading GGUF Metadata in Python

You can inspect a GGUF file's metadata without fully loading it:

```python
# Install: pip install gguf
from gguf import GGUFReader

def inspect_gguf(filepath):
    """Print key metadata from a GGUF file."""
    reader = GGUFReader(filepath)

    print("=== GGUF Model Information ===")
    print()

    # Print all metadata fields
    for field_name, field in reader.fields.items():
        # Skip the tokenizer vocab (too long to print)
        if "tokens" in field_name or "scores" in field_name:
            continue

        # Get the value based on its type
        if field.types[0].name == "STRING":
            value = str(bytes(field.parts[-1]), encoding="utf-8")
        elif hasattr(field.parts[-1], "tolist"):
            value = field.parts[-1].tolist()
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
        else:
            value = field.parts[-1]

        print(f"  {field_name}: {value}")

    # Print tensor information
    print("\n=== Tensors ===")
    print(f"Total tensors: {len(reader.tensors)}")

    # Show first few tensors
    for tensor in list(reader.tensors)[:5]:
        print(f"  {tensor.name}: shape={tensor.shape}, type={tensor.tensor_type.name}")

    print("  ...")

# Usage:
inspect_gguf("./models/mistral-7b-instruct-Q4_K_M.gguf")
```

Example output:
```
=== GGUF Model Information ===
  general.architecture: llama
  general.name: Mistral-7B-Instruct-v0.3
  llama.context_length: 32768
  llama.embedding_length: 4096
  llama.block_count: 32
  llama.attention.head_count: 32
  llama.rope.freq_base: 1000000.0
  tokenizer.ggml.model: llama

=== Tensors ===
Total tensors: 291
  token_embd.weight: shape=[32768, 4096], type=Q4_K
  blk.0.attn_norm.weight: shape=[4096], type=F32
  blk.0.ffn_down.weight: shape=[4096, 14336], type=Q4_K
  blk.0.ffn_gate.weight: shape=[14336, 4096], type=Q4_K
  blk.0.ffn_up.weight: shape=[14336, 4096], type=Q4_K
  ...
```

---

## Summary

```
+------------------------------------------------------------------+
|  LESSON 02 SUMMARY                                               |
+------------------------------------------------------------------+
|                                                                  |
|  1. GGUF Format                                                  |
|     Self-contained file: metadata + tokenizer + weights.         |
|     Versioned format that won't break when llama.cpp updates.    |
|     Memory-mappable for efficient loading.                       |
|                                                                  |
|  2. GGUF Quantization Types                                      |
|     Q4_K_M: best quality-size tradeoff for 4-bit models.         |
|     Q8_0: high quality, 2x larger than Q4_K_M.                  |
|     K-quant: uses different precision for sensitive layers.      |
|                                                                  |
|  3. llama.cpp                                                    |
|     C++ inference engine. Fast on CPU and GPU.                   |
|     Foundation of Ollama and many other tools.                   |
|                                                                  |
|  4. llama-cpp-python                                             |
|     Python bindings for llama.cpp.                               |
|     Llama() class loads and runs GGUF models.                    |
|     create_chat_completion() for instruction-following models.   |
|                                                                  |
|  5. Key Parameters                                               |
|     n_ctx: context window. n_threads: CPU cores.                 |
|     n_gpu_layers: GPU offloading. temperature: randomness.       |
|                                                                  |
|  6. Prompt Templates                                             |
|     Each model has a specific format it expects.                 |
|     create_chat_completion() handles this automatically.         |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Quiz Questions

1. What does GGUF stand for? What replaced it (i.e., what did GGUF replace)?

2. List the 4 main sections inside a GGUF file in order.
   Why does the tensor index come before the tensor data?

3. You have a laptop with 8 GB RAM. Which GGUF quantization would you choose
   for a 7B parameter model, and why?

4. What is the difference between Q4_0 and Q4_K_M?
   What does the "K" in Q4_K_M stand for?

5. What is llama.cpp and who created it? Why is it significant?

6. You set n_gpu_layers=0 but n_threads=16. Where does the model run?
   You then set n_gpu_layers=-1. What changes?

7. Why might you get poor results from a model even though you loaded it correctly?
   (Hint: think about how different models expect their input formatted.)

8. What does a temperature of 0.0 mean for text generation?
   What happens at temperature 2.0?

---

*Next lesson: TorchAO -- PyTorch-native quantization and optimization.*
*File: lessons/03_torchao.md*

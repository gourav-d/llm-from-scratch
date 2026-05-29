# =============================================================================
# Module 14 - Deploying LLMs
# Example 02: GGUF Format and llama.cpp
# =============================================================================
#
# WHAT THIS FILE TEACHES:
#   - What a GGUF file contains (structure walkthrough)
#   - How to validate a GGUF file before loading
#   - How to load and chat with a GGUF model via llama-cpp-python
#   - Key parameters: n_ctx, n_threads, n_gpu_layers, temperature
#
# GLOSSARY:
#   GGUF          - GPT-Generated Unified Format. Self-contained model file.
#                   Contains weights + tokenizer + metadata all in one file.
#   llama.cpp     - C++ library that runs GGUF models fast on CPU and GPU.
#   llama-cpp-python - Python wrapper around llama.cpp.
#   magic number  - First bytes of a file that identify its type.
#                   Like a file signature. GGUF starts with bytes: G G U F
#   n_ctx         - Context window: how many tokens the model can see at once
#   n_threads     - CPU cores to use for inference
#   n_gpu_layers  - How many transformer blocks to run on GPU (0 = CPU only)
#   temperature   - Randomness of sampling (0=deterministic, 1=normal, 2=chaotic)
#
# C# ANALOGY:
#   GGUF is like a self-contained NuGet package that includes:
#     - The compiled DLL (weights)
#     - The manifest (metadata: architecture, context size, etc.)
#     - The resource files (tokenizer vocabulary)
#   No external dependencies needed -- just the one file.
#
# REQUIREMENTS:
#   Part A: No external libraries (uses only Python built-in 'struct' and 'os')
#   Part B: pip install llama-cpp-python
#           AND you need a GGUF model file downloaded locally.
#
# HOW TO GET A GGUF FILE (free):
#   Option 1 - Ollama (easiest):
#     ollama pull llama3.2:1b
#     Files stored in: C:\Users\{you}\.ollama\models\blobs\
#
#   Option 2 - Download direct:
#     huggingface.co -> search "llama-3.2-1b-instruct GGUF" -> download Q4_K_M file
#
# =============================================================================

import os       # os = operating system utilities (file paths, file size, etc.)
import struct   # struct = read binary data in specific formats (like C structs)

# =============================================================================
# PART A: GGUF FILE STRUCTURE (No external libraries)
#
# We inspect a GGUF file's header to understand its structure.
# Works even without llama-cpp-python installed.
# =============================================================================

print("=" * 60)
print("PART A: GGUF File Structure")
print("=" * 60)

# ---------------------------------------------------------
# GGUF Magic Number Check
#
# Every valid GGUF file starts with exactly 4 bytes: 0x47 0x47 0x55 0x46
# Which in ASCII is: G G U F
# This is called the "magic number" -- a file fingerprint.
# C# analogy: Like checking the first bytes of a ZIP file (PK 0x03 0x04)
# ---------------------------------------------------------

def check_gguf_magic(filepath):
    """
    Read the first few bytes of a file to check if it is a valid GGUF file.

    Parameters:
        filepath: path to the file to check
    Returns:
        True if valid GGUF, False otherwise
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    # Get file size for information
    size_bytes = os.path.getsize(filepath)      # Returns file size in bytes
    size_mb = size_bytes / (1024 * 1024)        # Convert bytes to megabytes
    size_gb = size_bytes / (1024 * 1024 * 1024) # Convert bytes to gigabytes

    print(f"\nFile: {filepath}")
    print(f"Size: {size_mb:.1f} MB ({size_gb:.2f} GB)")

    # Open file in binary mode ("rb" = read binary)
    # C# analogy: FileStream(path, FileMode.Open, FileAccess.Read)
    with open(filepath, "rb") as f:
        # Read first 8 bytes (magic number + version)
        header = f.read(8)

    if len(header) < 8:
        print("ERROR: File too small to be a valid GGUF file.")
        return False

    # First 4 bytes = magic number (should be "GGUF" in ASCII)
    # struct.unpack("4s", ...) reads 4 bytes as a string
    # C# analogy: BitConverter.ToString(bytes, 0, 4)
    magic = header[:4]   # Python slice: bytes 0 to 3

    if magic == b"GGUF":   # b"GGUF" = bytes literal, not a string
        print("Magic number: GGUF -- Valid GGUF file!")
    else:
        print(f"Magic number: {magic} -- NOT a valid GGUF file.")
        return False

    # Bytes 4-7 = version number (uint32 little-endian)
    # struct.unpack("I", ...) reads 4 bytes as unsigned int (little-endian)
    # "<" means little-endian (least significant byte first, common on Intel CPUs)
    # C# analogy: BitConverter.ToUInt32(bytes, 4) -- little-endian by default on Windows
    version = struct.unpack("<I", header[4:8])[0]   # [0] gets first (only) result
    print(f"GGUF version: {version}")

    if version < 2:
        print("WARNING: Old GGUF version. May not work with current tools.")
    else:
        print("Version OK (2 or higher is current standard).")

    return True


# ---------------------------------------------------------
# Show the conceptual GGUF structure with ASCII diagram
# ---------------------------------------------------------

def show_gguf_structure():
    """Print an ASCII diagram of GGUF file structure."""
    print("\nGGUF File Structure:")
    print("-" * 50)
    print("[1] HEADER (8 bytes)")
    print("    Magic: 'GGUF' (4 bytes)")
    print("    Version: 2 or 3  (4 bytes)")
    print("")
    print("[2] TENSOR COUNT + METADATA COUNT (16 bytes)")
    print("    n_tensors: how many weight tensors")
    print("    n_kv:      how many metadata entries")
    print("")
    print("[3] METADATA KEY-VALUE PAIRS")
    print("    Each entry: key (string) + type + value")
    print("    Examples:")
    print("      general.architecture = 'llama'")
    print("      llama.context_length = 4096")
    print("      llama.embedding_length = 4096")
    print("      llama.block_count = 32")
    print("      tokenizer.ggml.tokens = [list of all 32000 token strings]")
    print("      tokenizer.chat_template = '<s>[INST] ...'")
    print("")
    print("[4] TENSOR INDEX")
    print("    For each tensor:")
    print("      name   (e.g., 'blk.0.attn_q.weight')")
    print("      shape  (e.g., [4096, 4096])")
    print("      quant  (e.g., Q4_K)")
    print("      offset (byte position in section [5])")
    print("")
    print("[5] TENSOR DATA (the actual quantized weights)")
    print("    Each tensor's bytes, aligned to 32-byte boundaries")
    print("    Aligned = padded so each tensor starts at a multiple of 32")
    print("-" * 50)
    print("\nKey insight: Index [4] comes BEFORE data [5].")
    print("This lets llama.cpp jump directly to any tensor without")
    print("reading the whole file. Like a database index vs full scan.")
    print("C# analogy: Dictionary<string, long> tensorIndex = ...")
    print("           fileStream.Seek(tensorIndex['blk.5.attn_q.weight'], ...)")


show_gguf_structure()

# ---------------------------------------------------------
# Show quantization type naming explained
# ---------------------------------------------------------

def show_quant_names():
    """Explain GGUF quantization naming conventions."""
    print("\nGGUF Quantization Types (what the file name tells you):")
    print("-" * 50)

    # Dictionary: quant type name -> explanation
    # In C#: Dictionary<string, string>
    quant_types = {
        "Q4_0":   "INT4, simple scale per 32 weights. Smallest, lowest quality.",
        "Q4_1":   "INT4 + min-value offset. Slightly better than Q4_0.",
        "Q4_K_S": "INT4 K-quant, Small. Attention layers use 6-bit. Better quality.",
        "Q4_K_M": "INT4 K-quant, Medium. BEST QUALITY/SIZE for 4-bit. Use this.",
        "Q5_K_M": "INT5 K-quant, Medium. Between Q4 and Q8. Good if you have RAM.",
        "Q8_0":   "INT8, simple scale per 32 weights. Near fp16 quality. 2x Q4 size.",
        "F16":    "fp16 (no integer quantization). Best quality. Largest size.",
    }

    for name, desc in quant_types.items():   # .items() gives (key, value) pairs
        print(f"  {name:<12} {desc}")   # :<12 = left-aligned in 12 chars

    print("\nRECOMMENDATION:")
    print("  Less than 8 GB RAM  -> Q4_K_M")
    print("  8 to 16 GB RAM      -> Q5_K_M or Q8_0")
    print("  16+ GB RAM          -> Q8_0 or F16")
    print("  Quality critical    -> Q8_0")


show_quant_names()

# Try to check a GGUF file if path is provided
# Update this path to point to an actual GGUF file on your machine
gguf_path = r"C:\path\to\your\model.gguf"   # <-- change this path
if os.path.exists(gguf_path):
    check_gguf_magic(gguf_path)
else:
    print(f"\nSkipping file check (file not found: {gguf_path})")
    print("Update 'gguf_path' to point to a real .gguf file to test.")

# =============================================================================
# PART B: LOADING AND CHATTING WITH A GGUF MODEL
#
# Uses llama-cpp-python to load a real GGUF model and run inference.
# Install: pip install llama-cpp-python
#
# You MUST have a GGUF file on your machine to run this part.
# The script will work without it but skip the actual inference.
# =============================================================================

print("\n" + "=" * 60)
print("PART B: Llama-cpp-python: Load and Chat with GGUF Model")
print("=" * 60)

# Try to import llama-cpp-python
# If not installed, we show what the code WOULD do (educational mode)
try:
    from llama_cpp import Llama    # The main class from llama-cpp-python
    LLAMA_CPP_AVAILABLE = True
    print("llama-cpp-python is installed.")
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("llama-cpp-python not installed.")
    print("Install: pip install llama-cpp-python")
    print("(Showing code explanation in educational mode)")

# Path to your GGUF model file
# Common small models to download for learning:
#   - Phi-3 mini (3.8B, Q4_K_M) -- ~2.2 GB
#   - LLaMA 3.2 1B (Q4_K_M)     -- ~0.8 GB  (smallest useful model)
#   - Mistral 7B (Q4_K_M)        -- ~4.1 GB  (good general purpose)
MODEL_PATH = r"C:\path\to\your\model-Q4_K_M.gguf"   # <-- change this path

def demonstrate_llama_cpp():
    """
    Show how to load and use a GGUF model with llama-cpp-python.
    Runs only if llama-cpp-python is installed AND model file exists.
    """
    if not LLAMA_CPP_AVAILABLE:
        print("\nCode that WOULD run if llama-cpp-python was installed:")
        print("-" * 40)
        print("from llama_cpp import Llama")
        print("")
        print("# Load the GGUF model file")
        print("# n_ctx=2048:        model can see 2048 tokens at once")
        print("# n_threads=4:       use 4 CPU cores for matrix math")
        print("# n_gpu_layers=0:    run entirely on CPU (no GPU needed)")
        print("# verbose=False:     hide llama.cpp debug output")
        print("model = Llama(")
        print("    model_path='your-model-Q4_K_M.gguf',")
        print("    n_ctx=2048,")
        print("    n_threads=4,")
        print("    n_gpu_layers=0,")
        print("    verbose=False")
        print(")")
        print("")
        print("# create_chat_completion: handles prompt template automatically")
        print("# (Each model has its own format -- llama.cpp reads it from GGUF)")
        print("response = model.create_chat_completion(")
        print("    messages=[")
        print("        {'role': 'system', 'content': 'You are a helpful assistant.'},")
        print("        {'role': 'user',   'content': 'What is quantization?'}")
        print("    ],")
        print("    max_tokens=200,")
        print("    temperature=0.7   # 0=deterministic, 1=normal, 2=very random")
        print(")")
        print("")
        print("# Get the text response")
        print("reply = response['choices'][0]['message']['content']")
        print("print(reply)")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"\nModel file not found: {MODEL_PATH}")
        print("Update MODEL_PATH to point to a real GGUF file.")
        return

    print(f"\nLoading model: {MODEL_PATH}")
    print("(This may take 5-30 seconds depending on model size and RAM speed)")

    # Load the GGUF model
    # n_ctx: context window size (2048 tokens = can see 2048 tokens history)
    # n_threads: CPU cores to use (set to your physical core count)
    # n_gpu_layers: 0 means CPU only; set higher if you have a GPU
    # verbose: False hides the llama.cpp loading output
    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,        # 2048 tokens context window
        n_threads=4,       # Use 4 CPU cores (change to match your CPU)
        n_gpu_layers=0,    # 0 = CPU only. -1 = all layers on GPU.
        verbose=False      # Suppress llama.cpp debug output
    )

    print("Model loaded successfully!")
    print("-" * 40)

    # ---------------------------------------------------------
    # Chat completion (the standard way to talk to instruct models)
    # ---------------------------------------------------------

    print("\nTest 1: Basic chat question")
    # create_chat_completion takes a list of messages (same format as OpenAI)
    # "role" can be "system", "user", or "assistant"
    response = model.create_chat_completion(
        messages=[
            {
                "role": "system",
                # System message sets the model's behavior
                "content": "You are a helpful assistant. Answer in 2 sentences max."
            },
            {
                "role": "user",
                # User message is the actual question
                "content": "What is model quantization in one sentence?"
            }
        ],
        max_tokens=100,    # Maximum number of tokens to generate
        temperature=0.3,   # Low temperature = more focused/deterministic answers
        stop=["\n\n"]      # Stop generating if we see a double newline
    )

    # Extract the text from the response structure
    # response is a dict with 'choices' list, each choice has 'message'
    # C# analogy: response.Choices[0].Message.Content
    reply = response["choices"][0]["message"]["content"]
    print(f"Model reply: {reply}")

    # ---------------------------------------------------------
    # Streaming output (token by token, like ChatGPT's typing effect)
    # ---------------------------------------------------------

    print("\nTest 2: Streaming response (watch tokens appear one by one)")
    print("Response: ", end="")    # end="" means no newline after "Response: "

    # stream=True makes the function return an iterator instead of waiting
    # Each item in the iterator is one token chunk
    stream = model.create_chat_completion(
        messages=[
            {"role": "user", "content": "Count from 1 to 5 slowly."}
        ],
        max_tokens=50,
        temperature=0.1,
        stream=True    # KEY: this enables token-by-token streaming
    )

    # Iterate over chunks as they are generated
    for chunk in stream:
        # Each chunk has choices[0].delta (the new token content)
        delta = chunk["choices"][0]["delta"]

        # "content" key only exists when there is new text
        # (Some chunks are just status updates with no content)
        if "content" in delta:
            # flush=True forces the output to appear immediately
            # Without flush=True, output might be buffered and appear all at once
            print(delta["content"], end="", flush=True)

    print()   # Final newline after all tokens are printed

    # ---------------------------------------------------------
    # Show model parameters (what parameters the GGUF reports)
    # ---------------------------------------------------------

    print("\nModel metadata from GGUF file:")
    # model.metadata is a dict of all key-value pairs in the GGUF header
    interesting_keys = [
        "general.architecture",   # Model type: llama, mistral, phi, etc.
        "general.name",           # Model's name
        "llama.context_length",   # Max context the model supports
        "llama.embedding_length", # Dimension of token embeddings (d_model)
        "llama.block_count",      # Number of transformer blocks (depth)
        "llama.attention.head_count",  # Number of attention heads
    ]

    for key in interesting_keys:
        # .get() returns None if key doesn't exist (safe dictionary lookup)
        # C# analogy: dict.TryGetValue(key, out value)
        value = model.metadata.get(key, "not found")
        print(f"  {key}: {value}")


# Run the demonstration
demonstrate_llama_cpp()

# =============================================================================
# PARAMETER REFERENCE
# =============================================================================

print("\n" + "=" * 60)
print("Key Parameter Reference for Llama()")
print("=" * 60)
print("")
print("n_ctx (context window)")
print("  Default: 512")
print("  What: How many tokens the model 'remembers'")
print("  Tradeoff: Larger = more RAM (KV-cache grows linearly)")
print("  Recommended: 2048 for chat, 4096 for long documents")
print("")
print("n_threads (CPU cores)")
print("  Default: 1 (too slow!)")
print("  What: Parallel CPU computation for matrix math")
print("  Recommended: set to your PHYSICAL core count (not logical)")
print("  Find your core count: Task Manager -> Performance -> CPU -> Cores")
print("")
print("n_gpu_layers (GPU offloading)")
print("  Default: 0 (CPU only)")
print("  What: Number of transformer blocks to put on GPU")
print("  0   = CPU only (slowest, works on any machine)")
print("  32  = offload 32 blocks to GPU (faster if VRAM allows)")
print("  -1  = offload ALL layers to GPU (fastest, needs full VRAM)")
print("  Mixed: e.g., 20 layers GPU + 12 layers CPU (if partial VRAM)")
print("")
print("temperature (randomness)")
print("  Default: 0.8")
print("  0.0 = deterministic (always pick most likely token)")
print("  0.3 = focused, factual answers")
print("  0.7 = balanced creativity")
print("  1.0 = standard randomness")
print("  2.0 = very random (often incoherent)")
print("")
print("=" * 60)
print("Key Takeaways:")
print("=" * 60)
print("1. GGUF files are self-contained: weights + tokenizer + metadata.")
print("2. Magic bytes 'GGUF' identify the file format.")
print("3. Q4_K_M is the best choice for local laptops with limited RAM.")
print("4. llama-cpp-python wraps llama.cpp with a Python-friendly API.")
print("5. create_chat_completion handles prompt templates automatically.")

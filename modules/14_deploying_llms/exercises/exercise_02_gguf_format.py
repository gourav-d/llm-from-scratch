# =============================================================================
# Module 14 - Deploying LLMs
# Exercise 02: GGUF Format and llama.cpp
# =============================================================================
#
# INSTRUCTIONS:
#   Complete each TODO section.
#   Part A requires no special libraries (just Python built-ins).
#   Part B requires: pip install llama-cpp-python
#   You need a GGUF model file for Part B tasks 3-4.
#
# WHAT YOU PRACTICE:
#   - Reading GGUF file structure (magic bytes, version)
#   - Choosing the right quantization type for a hardware constraint
#   - Writing llama-cpp-python loading code
#   - Building a simple multi-turn chat loop
#
# =============================================================================

import os
import struct

print("=" * 60)
print("Exercise 02: GGUF Format")
print("=" * 60)

# =============================================================================
# TASK 1: GGUF File Validator
#
# Write a function that checks if a file is a valid GGUF file.
# A valid GGUF file must:
#   - Exist on disk
#   - Be at least 1 MB in size (any real model is much larger)
#   - Start with the magic bytes: G G U F  (0x47 0x47 0x55 0x46)
#   - Have a version of 2 or higher (versions 1 and below are obsolete)
# =============================================================================

print("\n--- Task 1: GGUF File Validator ---")

def validate_gguf(filepath):
    """
    Validate that a file is a usable GGUF model file.

    Returns a dict with:
        valid:   bool    -- True if the file passes all checks
        reason:  str     -- Description of first failure, or "OK"
        size_mb: float   -- File size in MB (0 if file not found)
        version: int     -- GGUF version number (0 if unreadable)

    TODO: Implement all 4 checks described above.
    """
    result = {"valid": False, "reason": "", "size_mb": 0.0, "version": 0}

    # TODO: Check 1 -- does the file exist?
    if not os.path.exists(filepath):
        result["reason"] = "File not found"
        return result   # Return early -- no point checking further

    # TODO: Check 2 -- is it at least 1 MB?
    size_bytes = os.path.getsize(filepath)
    result["size_mb"] = size_bytes / (1024 * 1024)
    if result["size_mb"] < 1.0:
        result["reason"] = f"File too small ({result['size_mb']:.1f} MB). Not a real model."
        return result

    # TODO: Check 3 -- does it start with the GGUF magic bytes?
    # Open in binary mode ("rb"), read first 8 bytes
    with open(filepath, "rb") as f:
        header = f.read(8)

    magic = header[:4]   # First 4 bytes
    if magic != b"GGUF":
        result["reason"] = f"Invalid magic: {magic!r}. Expected b'GGUF'."
        return result

    # TODO: Check 4 -- is the version at least 2?
    # Bytes 4-7 are the version as a little-endian uint32
    # Hint: struct.unpack("<I", header[4:8])[0]
    version = None   # TODO
    result["version"] = version if version is not None else 0

    if version is not None and version < 2:
        result["reason"] = f"Obsolete GGUF version {version}. Need version 2+."
        return result

    # All checks passed
    result["valid"] = True
    result["reason"] = "OK"
    return result


# Test with a dummy file (we create a fake one for testing)
def create_test_gguf(path, magic=b"GGUF", version=3, size_padding=2048):
    """Create a minimal fake GGUF file for testing."""
    with open(path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", version))
        f.write(b"\x00" * size_padding)   # Padding to make it > 1 MB

# Test 1: Valid GGUF
valid_path = "test_valid.gguf"
create_test_gguf(valid_path, magic=b"GGUF", version=3, size_padding=2 * 1024 * 1024)
r = validate_gguf(valid_path)
print(f"Valid GGUF:   valid={r['valid']}, reason='{r['reason']}', version={r['version']}, size={r['size_mb']:.1f}MB")
os.remove(valid_path)

# Test 2: Wrong magic
bad_magic_path = "test_bad_magic.gguf"
create_test_gguf(bad_magic_path, magic=b"BADM", version=3, size_padding=2 * 1024 * 1024)
r = validate_gguf(bad_magic_path)
print(f"Bad magic:    valid={r['valid']}, reason='{r['reason']}'")
os.remove(bad_magic_path)

# Test 3: File not found
r = validate_gguf("nonexistent.gguf")
print(f"Missing file: valid={r['valid']}, reason='{r['reason']}'")

# Test 4: Old version
old_path = "test_old_version.gguf"
create_test_gguf(old_path, magic=b"GGUF", version=1, size_padding=2 * 1024 * 1024)
r = validate_gguf(old_path)
print(f"Old version:  valid={r['valid']}, reason='{r['reason']}'")
os.remove(old_path)

# =============================================================================
# TASK 2: Quantization Advisor
#
# Given available RAM and quality requirements, recommend a GGUF quant type.
# =============================================================================

print("\n--- Task 2: Quantization Advisor ---")

def recommend_quantization(model_params_billions, available_ram_gb, quality="balanced"):
    """
    Recommend a GGUF quantization type given hardware and quality needs.

    Parameters:
        model_params_billions: float -- model size in billions (e.g., 7.0 for 7B)
        available_ram_gb:      float -- RAM available on the machine
        quality: str           -- "best", "balanced", or "smallest"

    Returns:
        dict with keys:
            recommended: str  -- e.g., "Q4_K_M"
            model_size_gb: float  -- estimated size with this quantization
            fits_in_ram: bool
            reason: str

    Size estimates (bytes per param):
        Q4_K_M: 0.55 bytes  (overhead from scale factors)
        Q5_K_M: 0.70 bytes
        Q8_0:   1.10 bytes  (overhead from scale factors)
        F16:    2.00 bytes

    TODO: Implement this function.
    Logic:
        1. Calculate size for each quant type
        2. If quality="best", prefer Q8_0 if it fits, else Q5_K_M, else Q4_K_M
        3. If quality="balanced", prefer Q4_K_M if it fits, else error
        4. If quality="smallest", always use Q4_K_M
        5. Return recommended type, estimated size, and whether it fits
    """
    # Bytes per parameter for each quantization type
    bytes_per_param = {
        "Q4_K_M": 0.55,
        "Q5_K_M": 0.70,
        "Q8_0":   1.10,
        "F16":    2.00,
    }

    params = model_params_billions * 1_000_000_000   # Convert B to raw count

    # Calculate model size in GB for each quant type
    sizes = {}
    for quant, bpp in bytes_per_param.items():
        sizes[quant] = (params * bpp) / (1024 ** 3)

    # TODO: Implement the recommendation logic
    recommended = None
    reason = None

    if quality == "best":
        # TODO: Try Q8_0 first, then Q5_K_M, then Q4_K_M
        pass
    elif quality == "balanced":
        # TODO: Use Q4_K_M if it fits, otherwise suggest quality="smallest"
        pass
    elif quality == "smallest":
        # TODO: Always use Q4_K_M
        pass

    if recommended is None:
        recommended = "Q4_K_M"
        reason = "Default choice"

    model_size_gb = sizes[recommended]
    fits = model_size_gb <= available_ram_gb

    return {
        "recommended": recommended,
        "model_size_gb": round(model_size_gb, 2),
        "fits_in_ram": fits,
        "reason": reason or ""
    }


# Test your advisor
test_scenarios = [
    (7.0,  8.0,  "best"),       # 7B on 8GB machine, want quality
    (7.0,  8.0,  "balanced"),   # 7B on 8GB machine, balanced
    (70.0, 16.0, "smallest"),   # 70B on 16GB machine (tight!)
    (7.0,  64.0, "best"),       # 7B on 64GB server
]

print("\nQuantization recommendations:")
for params_b, ram_gb, quality in test_scenarios:
    r = recommend_quantization(params_b, ram_gb, quality)
    fits_str = "fits" if r["fits_in_ram"] else "DOES NOT FIT"
    print(f"  {params_b:.0f}B model, {ram_gb:.0f}GB RAM, quality={quality!r}:")
    print(f"    -> {r['recommended']} ({r['model_size_gb']:.1f}GB, {fits_str})")
    print(f"    Reason: {r['reason']}")

# =============================================================================
# TASK 3: Write the llama-cpp-python Loading Code
#
# Fill in the blanks to write correct llama-cpp-python usage.
# This is a code-writing exercise (no execution needed if library not installed).
# =============================================================================

print("\n--- Task 3: llama-cpp-python Code ---")

print("""
Fill in the blanks (____) to complete the code:

from llama_cpp import ____           # 1. What class do you import?

model = ____(                        # 2. What class do you instantiate?
    model_path="mistral-7b-Q4_K_M.gguf",
    n_ctx=____,                      # 3. Context window. Good default?
    n_threads=____,                  # 4. For an 8-core CPU, what value?
    n_gpu_layers=____,               # 5. CPU only, no GPU?
    verbose=____                     # 6. Hide debug output?
)

response = model.create_chat_completion(
    messages=[
        {"role": "____", "content": "You are a helpful assistant."},  # 7. System role name?
        {"role": "____", "content": "What is quantization?"}           # 8. User role name?
    ],
    max_tokens=200,
    temperature=____,                # 9. Focused, factual answer? (low value)
    stream=____                      # 10. Wait for full response, not streaming?
)

reply = response["____"][0]["____"]["____"]  # 11. How to extract the text?
""")

print("Answers:")
print("  1. Llama")
print("  2. Llama")
print("  3. 2048 (or 4096 for longer conversations)")
print("  4. 8 (one thread per physical core)")
print("  5. 0")
print("  6. False")
print("  7. system")
print("  8. user")
print("  9. 0.1 to 0.3 (lower = more deterministic)")
print(" 10. False")
print(" 11. response['choices'][0]['message']['content']")

# =============================================================================
# TASK 4: Multi-Turn Chat Loop (runs only if llama-cpp-python is installed)
#
# Build a chat loop that maintains conversation history.
# Multi-turn = the model remembers previous messages in the conversation.
# =============================================================================

print("\n--- Task 4: Multi-Turn Chat Loop ---")

MODEL_PATH = r"C:\path\to\your\model-Q4_K_M.gguf"   # <-- update this path

def run_chat_loop(model_path):
    """
    Run a simple multi-turn chat with a GGUF model.

    The conversation history grows with each turn so the model
    remembers what was said earlier (until context window fills).

    TODO: Complete the conversation history management.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed. Skipping Task 4.")
        print("Install: pip install llama-cpp-python")
        return

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Update MODEL_PATH to point to a GGUF file.")
        return

    print(f"Loading: {model_path}")
    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False
    )
    print("Model loaded. Type 'quit' to exit.\n")

    # TODO: Initialize the conversation history list
    # It should start with a system message
    conversation_history = [
        # TODO: Add a system message here
        # {"role": "system", "content": "You are a helpful assistant. Be concise."}
    ]

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:   # Skip empty inputs
            continue

        # TODO: Append the user's message to conversation_history
        # {"role": "user", "content": user_input}

        # Call the model with the FULL history (not just the latest message)
        response = model.create_chat_completion(
            messages=conversation_history,   # TODO: is this right?
            max_tokens=300,
            temperature=0.7
        )

        # Extract the reply
        reply = response["choices"][0]["message"]["content"]

        # TODO: Append the assistant's reply to conversation_history
        # {"role": "assistant", "content": reply}

        print(f"Assistant: {reply}\n")

        # Show history length (useful to see context window filling up)
        print(f"[History: {len(conversation_history)} messages]")


run_chat_loop(MODEL_PATH)

# =============================================================================
# HINTS
# =============================================================================

print("\n" + "=" * 60)
print("HINTS")
print("=" * 60)
print("""
Task 1 (validate_gguf):
  version = struct.unpack("<I", header[4:8])[0]
  if version < 2: ... return result

Task 2 (recommend_quantization):
  if quality == "best":
      for quant in ["Q8_0", "Q5_K_M", "Q4_K_M"]:
          if sizes[quant] <= available_ram_gb:
              recommended = quant
              reason = f"{quant} fits in RAM with best quality"
              break
  elif quality == "balanced":
      if sizes["Q4_K_M"] <= available_ram_gb:
          recommended = "Q4_K_M"
          reason = "Best quality-size tradeoff"
  elif quality == "smallest":
      recommended = "Q4_K_M"
      reason = "Smallest model that still gives useful results"

Task 4 (multi-turn chat):
  conversation_history = [{"role": "system", "content": "You are helpful."}]
  # Each turn:
  conversation_history.append({"role": "user", "content": user_input})
  # After model responds:
  conversation_history.append({"role": "assistant", "content": reply})
""")

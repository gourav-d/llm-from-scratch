# Lesson 03: TorchAO -- PyTorch-Native Quantization

## Glossary (Read This First!)

| Term | Plain English Definition |
|------|--------------------------|
| **TorchAO** | "Torch Architecture Optimization." PyTorch's official library for quantization, sparsity, and kernel optimization. Developed and maintained by PyTorch team. |
| **Kernel** | In GPU/CPU programming, a kernel is a function that runs on many data elements in parallel. Matrix multiplication is done by a highly optimized kernel. |
| **Kernel fusion** | Combining multiple operations (e.g., add + multiply + activate) into a single kernel call. Reduces memory round-trips = faster. |
| **Eager mode** | PyTorch's default mode. Operations run immediately as you write them. Easy to debug. |
| **torch.compile** | PyTorch's JIT compiler. Takes eager-mode code and compiles it to optimized machine code. Combined with TorchAO for maximum speed. |
| **Linear layer** | The core operation in a transformer: output = input @ weight + bias. This is the most quantized layer. |
| **Activation** | The output of a layer before it becomes input to the next. Different from "activation functions." |
| **Affine quantization** | Quantization with a scale and zero point: q = round(x / scale) + zero_point. |
| **Symmetric quantization** | Quantization centered at zero. Scale only, no zero point. Simpler math. |
| **Dynamic quantization** | Compute scale factors at runtime based on actual values. Slower but more accurate. |
| **Static quantization** | Pre-compute scale factors once (during calibration). Faster at inference time. |
| **Weight-only quantization** | Only the weight matrices are quantized. Activations stay in fp16. |
| **CUDA** | NVIDIA's parallel computing platform. Needed for GPU acceleration in PyTorch. |
| **Tensor Core** | Special hardware in NVIDIA GPUs (Volta and later) that accelerates matrix multiplication in fp16/bf16/INT8. |

---

## Part 1: Why TorchAO Exists

Before TorchAO, PyTorch's quantization story was fragmented:

```
+------------------------------------------------------------------+
|  PYTORCH QUANTIZATION HISTORY                                    |
+------------------------------------------------------------------+
|                                                                  |
|  2019: torch.quantization (legacy)                               |
|    - Only supported specific model types                         |
|    - Hard to use, needed significant model rewriting             |
|    - INT8 only. No INT4. No bf16.                                |
|                                                                  |
|  2022: torch.ao (early architecture optimization)                |
|    - Better API, but still complex                               |
|    - Not many users                                              |
|                                                                  |
|  2023-2024: TorchAO (torchao)                                    |
|    - Complete rewrite from scratch                               |
|    - Simple API: one function call to quantize                   |
|    - Supports INT4, INT8, fp8, and more                          |
|    - Works natively with torch.compile                           |
|    - Developed by Meta's PyTorch team                            |
|    - Actively maintained (unlike legacy torch.quantization)      |
|                                                                  |
|  WHY USE TORCHAO INSTEAD OF GGUF/llama.cpp?                      |
|    - You have a PyTorch model (not a standard architecture)       |
|    - You need GPU acceleration with CUDA (not just CPU)          |
|    - You want to quantize during training (QAT)                  |
|    - You are building a Python-native serving pipeline            |
|    - You need to customize the quantization behavior             |
|                                                                  |
+------------------------------------------------------------------+
```

C# analogy:
```csharp
// TorchAO is like Microsoft.ML.OnnxRuntime's optimization passes,
// but for PyTorch models.
//
// When you run a .NET app in Release mode, the JIT compiler:
//   - Inlines small methods (kernel fusion)
//   - Uses SIMD instructions for loops (vectorization)
//   - Eliminates dead code (pruning)
//
// TorchAO does the equivalent for neural network inference:
//   - Fuses attention + FFN operations into single kernels
//   - Uses Tensor Core instructions (like SIMD for matrices)
//   - Reduces weight precision (fewer bytes to load from memory)
```

---

## Part 2: Installing TorchAO

```bash
# Install with pip (requires PyTorch already installed)
pip install torchao

# Or install PyTorch + TorchAO together:
pip install torch torchao

# Verify installation:
python -c "import torchao; print(torchao.__version__)"

# Note: Some quantization types require CUDA.
# CPU-only machines can still use INT8 and some INT4 modes.
```

---

## Part 3: TorchAO Core API

TorchAO's main API is designed to be simple.
Quantize a model in one function call.

### The quantize_() Function

```python
import torch
from torchao.quantization import quantize_, int8_weight_only, int4_weight_only

# Load your model (any PyTorch nn.Module)
# For this example, use a small transformer-like model
model = MyTransformerModel()
model.eval()  # Always set to eval mode before quantizing for inference

# Option 1: INT8 weight-only quantization
# Weights stored as INT8. Activations stay in fp16.
# 4x memory reduction vs fp32. Minimal quality loss.
quantize_(model, int8_weight_only())

# Option 2: INT4 weight-only quantization
# Weights stored as INT4. 8x memory reduction vs fp32.
# Small quality loss, significant speed gain.
quantize_(model, int4_weight_only())

# Option 3: INT8 dynamic activation quantization
# Both weights AND activations quantized to INT8.
# Scale factors for activations computed at runtime.
from torchao.quantization import int8_dynamic_activation_int8_weight
quantize_(model, int8_dynamic_activation_int8_weight())

# After quantization, run inference normally
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
with torch.no_grad():
    output = model(input_ids)
```

### What quantize_() Does Internally

```
+------------------------------------------------------------------+
|  WHAT quantize_() DOES UNDER THE HOOD                           |
+------------------------------------------------------------------+
|                                                                  |
|  1. Walk through all layers of the model.                        |
|                                                                  |
|  2. Find all nn.Linear layers.                                   |
|     (These contain the weight matrices we want to quantize.)     |
|                                                                  |
|  3. For each nn.Linear:                                          |
|     a. Compute scale factor(s) for the weight matrix.            |
|     b. Quantize the weights (fp32 -> INT8 or INT4).              |
|     c. Replace the nn.Linear module with a custom module         |
|        that stores INT weights but dequantizes before compute.   |
|                                                                  |
|  4. Non-linear layers (norms, embeddings, output head)           |
|     are left at their original precision.                        |
|                                                                  |
|  NOTE: quantize_() modifies the model IN-PLACE.                  |
|  (That is what the trailing underscore _ means in Python/PyTorch)|
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 4: Quantization Types in TorchAO

### INT8 Weight-Only

```python
from torchao.quantization import quantize_, int8_weight_only

quantize_(model, int8_weight_only())

# What it does:
# - Each Linear layer's weight: fp32/fp16 -> INT8
# - Scale factor: one per output channel (per row of weight matrix)
# - At inference: weights are dequantized to fp16 for matrix multiply
#
# Memory savings: ~4x vs fp32, ~2x vs fp16
# Speed:          Similar to fp16 (dequant overhead is small)
# Quality:        Minimal difference (<0.5% on most benchmarks)
# Best for:       Memory-constrained servers, moderate speed needs
```

### INT4 Weight-Only

```python
from torchao.quantization import quantize_, int4_weight_only

# group_size: quantize in groups of N weights (default 32)
# Smaller group_size = better quality, slightly more overhead
quantize_(model, int4_weight_only(group_size=32))

# What it does:
# - Each weight group of 32: compute scale + zero_point
# - Store weights as INT4 (two weights per byte)
# - At inference: dequantize group -> fp16, then compute
#
# Memory savings: ~8x vs fp32, ~4x vs fp16
# Speed:          Often FASTER than fp16 (less data to load from memory)
# Quality:        Noticeable on long-form generation, fine for most tasks
# Best for:       Memory-constrained local deployment
```

### INT8 Dynamic Activation Quantization

```python
from torchao.quantization import (
    quantize_,
    int8_dynamic_activation_int8_weight
)

quantize_(model, int8_dynamic_activation_int8_weight())

# What it does:
# - Weights: quantized to INT8 once (static)
# - Activations: quantized to INT8 at runtime (dynamic)
# - Uses INT8 x INT8 matrix multiply (very fast on modern hardware)
#
# Memory savings: ~4x vs fp32 for weights
# Speed:          Often 2-4x faster than fp32 on CPU (uses INT8 SIMD)
# Quality:        Similar to weight-only INT8
# Best for:       CPU inference where speed is critical
```

---

## Part 5: Combining with torch.compile

The full performance gain from TorchAO requires `torch.compile()`.
This compiles the model into optimized machine code.

```python
import torch
from torchao.quantization import quantize_, int4_weight_only

# Step 1: Load model
model = MyLLM().cuda()  # Move to GPU
model.eval()

# Step 2: Quantize
quantize_(model, int4_weight_only())

# Step 3: Compile
# mode="max-autotune" finds the fastest kernel for your specific GPU
# Takes 1-5 minutes on first run. Cached after that.
model = torch.compile(model, mode="max-autotune")

# Step 4: Run inference (compiled + quantized = maximum speed)
input_ids = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
with torch.no_grad():
    output = model(input_ids)
```

### What torch.compile Does

```
+------------------------------------------------------------------+
|  torch.compile OPTIMIZATION PIPELINE                             |
+------------------------------------------------------------------+
|                                                                  |
|  Your Model (Python/PyTorch eager mode)                          |
|         |                                                        |
|         v                                                        |
|  [TorchDynamo]                                                   |
|    Traces the Python code as it runs.                            |
|    Captures the computation graph.                               |
|    Handles Python control flow (if/for/while).                   |
|         |                                                        |
|         v                                                        |
|  [TorchInductor]                                                 |
|    Takes the computation graph.                                  |
|    Fuses operations (e.g., linear + gelu + linear into one).     |
|    Generates CUDA kernels or CPU vectorized code.                |
|    Finds the optimal kernel for your specific GPU model.         |
|         |                                                        |
|         v                                                        |
|  Compiled binary (saved in cache)                                |
|    Next run: skip compilation, use cached binary.                |
|    10-100x faster compilation on subsequent calls.               |
|                                                                  |
+------------------------------------------------------------------+
```

C# analogy:
```csharp
// torch.compile is like the .NET JIT, but you trigger it manually:
//
// Normal Python (like interpreted .NET):
//   Python bytecode -> Python interpreter -> CPU
//   Every operation goes through Python overhead
//
// torch.compile (like compiled .NET + NGEN):
//   Python code -> TorchDynamo -> TorchInductor -> Native CUDA/CPU binary
//   No Python overhead during inference
//   Optimized specifically for YOUR hardware
//
// The difference can be 2-10x faster inference.
```

---

## Part 6: Measuring the Impact of Quantization

Always measure before and after quantization. Do not trust claims -- benchmark.

```python
import torch
import time
from torchao.quantization import quantize_, int4_weight_only

def measure_model(model, input_ids, runs=50, warmup=10):
    """Measure throughput and memory of a model."""
    model.eval()

    # Warmup runs (let the GPU/JIT settle)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)

    # Synchronize GPU before timing
    if input_ids.is_cuda:
        torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_ids)

    if input_ids.is_cuda:
        torch.cuda.synchronize()
    end = time.perf_counter()

    # Report
    elapsed_ms = (end - start) * 1000 / runs
    memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if input_ids.is_cuda else 0

    return elapsed_ms, memory_mb

# Load model
model = MyLLM().eval()
input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # Sample input

# Baseline: fp32
t_fp32, m_fp32 = measure_model(model, input_ids)
print(f"fp32:  {t_fp32:.2f} ms/inference, {m_fp32:.1f} MB")

# Apply quantization
quantize_(model, int4_weight_only())
t_int4, m_int4 = measure_model(model, input_ids)
print(f"INT4:  {t_int4:.2f} ms/inference, {m_int4:.1f} MB")

# Report speedup and memory savings
print(f"Speedup: {t_fp32/t_int4:.1f}x")
print(f"Memory reduction: {m_fp32/m_int4:.1f}x")
```

---

## Part 7: Checking Model Size Before and After

```python
import torch
from torchao.quantization import quantize_, int8_weight_only

def model_size_mb(model):
    """Calculate total parameter memory in MB."""
    total_bytes = 0
    for param in model.parameters():
        # param.element_size() = bytes per element (4 for fp32, 1 for INT8)
        # param.nelement() = total number of elements
        total_bytes += param.nelement() * param.element_size()
    return total_bytes / (1024 * 1024)

model = MyLLM()

# Before quantization
size_before = model_size_mb(model)
print(f"Before quantization: {size_before:.1f} MB")

# Quantize
quantize_(model, int8_weight_only())

# After quantization
size_after = model_size_mb(model)
print(f"After INT8 quantization: {size_after:.1f} MB")
print(f"Compression ratio: {size_before/size_after:.1f}x")
```

---

## Part 8: Saving and Loading Quantized Models

Quantized models can be saved and loaded just like regular PyTorch models:

```python
import torch
from torchao.quantization import quantize_, int8_weight_only

# Quantize the model
model = MyLLM()
quantize_(model, int8_weight_only())

# Save using torch.save (saves the full model including quantization state)
torch.save(model.state_dict(), "model_int8.pt")

# Or save the whole model object (easier to load, larger file):
torch.save(model, "model_int8_full.pt")

# Load the state dict (requires knowing the model architecture):
new_model = MyLLM()
quantize_(new_model, int8_weight_only())  # Must quantize FIRST to match architecture
new_model.load_state_dict(torch.load("model_int8.pt"))
new_model.eval()

# Load the full model object (easier, but model class must be importable):
loaded_model = torch.load("model_int8_full.pt")
loaded_model.eval()
```

---

## Part 9: When to Use TorchAO vs GGUF

```
+------------------------------------------------------------------+
|  TORCHAO vs GGUF DECISION GUIDE                                  |
+------------------------------------------------------------------+
|                                                                  |
|  USE TORCHAO WHEN:                                               |
|                                                                  |
|  - You have a custom PyTorch model (not LLaMA/Mistral/etc.)      |
|    Your model is not supported by llama.cpp.                     |
|                                                                  |
|  - You need GPU inference with CUDA.                             |
|    TorchAO + CUDA + torch.compile = maximum GPU performance.     |
|                                                                  |
|  - You are building a Python application.                        |
|    Native PyTorch integration. No C++ bindings needed.           |
|                                                                  |
|  - You want to quantize during fine-tuning (QAT).               |
|    TorchAO supports quantization-aware training.                 |
|                                                                  |
|  - You need precise control over which layers are quantized.     |
|    TorchAO is programmable. GGUF is an opaque file format.       |
|                                                                  |
|  USE GGUF/OLLAMA WHEN:                                           |
|                                                                  |
|  - Your model is a standard architecture (LLaMA, Mistral, etc.) |
|    The GGUF file already exists. Just download and run.          |
|                                                                  |
|  - You want CPU inference with maximum portability.              |
|    llama.cpp works on any OS with no GPU required.               |
|                                                                  |
|  - You want the simplest possible setup.                         |
|    Ollama: one command to download and run.                      |
|                                                                  |
|  - You want to share a quantized model with others.              |
|    GGUF is the standard format for sharing quantized LLMs.       |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 10: TorchAO Quantization for HuggingFace Models

TorchAO integrates with HuggingFace transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torchao.quantization import quantize_, int4_weight_only

# Load a standard HuggingFace model
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # 1B model, manageable size
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Start in fp16
    device_map="auto"           # Let HuggingFace choose CPU or GPU
)

# Apply TorchAO INT4 quantization
model.eval()
quantize_(model, int4_weight_only())

# Optional: compile for extra speed (skip if you hit errors)
# model = torch.compile(model, mode="reduce-overhead")

# Run inference exactly as before -- the API does not change
inputs = tokenizer("What is the capital of France?", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## Summary

```
+------------------------------------------------------------------+
|  LESSON 03 SUMMARY                                               |
+------------------------------------------------------------------+
|                                                                  |
|  1. TorchAO                                                      |
|     PyTorch's official quantization library.                     |
|     Simple API: quantize_(model, method())                       |
|                                                                  |
|  2. Main Quantization Types                                      |
|     int8_weight_only(): 4x memory reduction, minimal quality loss|
|     int4_weight_only(): 8x memory reduction, small quality loss  |
|     int8_dynamic_activation: weights + activations in INT8       |
|                                                                  |
|  3. torch.compile                                                |
|     Compile quantized model for maximum performance.             |
|     First call: slow (1-5 min). After: uses cached binary.       |
|                                                                  |
|  4. Measuring Impact                                             |
|     Always benchmark: latency, memory, quality.                  |
|     Speedup is typically 2-4x for INT4 on GPU.                   |
|                                                                  |
|  5. TorchAO vs GGUF                                              |
|     TorchAO: custom models, GPU, Python-native, programmable.    |
|     GGUF: standard models, CPU-friendly, shareable, simple.      |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Quiz Questions

1. What does TorchAO stand for? Who develops it?

2. In Python/PyTorch, what does a trailing underscore `_` in a function name
   mean (like `quantize_(model, ...)`)?

3. You have a custom transformer model you trained yourself.
   Why would you use TorchAO instead of exporting to GGUF?

4. What is the difference between `int8_weight_only()` and
   `int8_dynamic_activation_int8_weight()`?
   When would each be faster?

5. What does `torch.compile()` do, and why does it take several minutes
   on the first call but is fast afterwards?

6. You want maximum inference speed on an NVIDIA GPU with your custom model.
   Write the three lines of code (quantize, compile, run) that achieve this.

7. What is kernel fusion? Give an example of operations that would be fused.

8. Name two reasons you would choose GGUF/Ollama over TorchAO.
   Name two reasons you would choose TorchAO over GGUF/Ollama.

---

*Next lesson: ONNX format -- framework-agnostic model deployment.*
*File: lessons/04_onnx_export.md*

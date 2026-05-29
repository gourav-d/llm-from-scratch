# Lesson 04: ONNX Export -- Framework-Agnostic Deployment

## Glossary (Read This First!)

| Term | Plain English Definition |
|------|--------------------------|
| **ONNX** | Open Neural Network Exchange. A file format for storing neural networks in a way any framework can read. |
| **ONNX Runtime (ORT)** | Microsoft's high-performance inference engine for ONNX models. Runs on CPU, GPU, mobile, edge devices. |
| **Computation graph** | A representation of a neural network as a directed graph. Nodes are operations (multiply, add, etc.), edges are data flowing between them. |
| **Operator** | A single operation in an ONNX graph. Examples: MatMul, Add, Softmax, LayerNorm. |
| **Opset** | ONNX Operator Set version. Newer opsets add new operators. You must specify which opset you are targeting. |
| **Dynamic shapes** | When a model can accept inputs of varying sizes (e.g., different sequence lengths). Opposite of fixed/static shapes. |
| **Execution provider** | In ONNX Runtime, a backend that runs computations. CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider, etc. |
| **Graph optimization** | Simplifying the ONNX graph before inference. Fold constants, fuse operations, eliminate dead nodes. |
| **Quantization (ONNX)** | ONNX has its own quantization tools (onnxruntime-tools) that quantize ONNX graphs to INT8. |
| **Shape inference** | Figuring out the shapes (dimensions) of every tensor in the graph. Required before optimization. |
| **Intermediate representation (IR)** | ONNX is an IR -- a format between the original code and the final machine code. |
| **Protobuf** | Protocol Buffers. Google's serialization format. ONNX files are stored as Protobuf. |
| **torch.onnx.export** | PyTorch's built-in function to export a PyTorch model to ONNX format. |
| **Tracing** | ONNX export works by "tracing" -- running the model with sample input and recording every operation. |

---

## Part 1: What Is ONNX and Why Does It Exist?

```
+------------------------------------------------------------------+
|  THE PROBLEM ONNX SOLVES                                         |
+------------------------------------------------------------------+
|                                                                  |
|  BEFORE ONNX (the fragmentation problem):                        |
|                                                                  |
|   PyTorch model ---> runs only in Python + PyTorch               |
|   TensorFlow model -> runs only in Python + TensorFlow           |
|   JAX model -------> runs only in Python + JAX                  |
|                                                                  |
|   If you trained in PyTorch but need to run in:                  |
|     - C++ application (game engine, embedded system)             |
|     - JavaScript browser (ONNX.js)                               |
|     - iOS/Android app (Core ML, TFLite)                          |
|     - Azure ML, TensorRT, OpenVINO inference engines             |
|   ... you had to rewrite or retrain the model.                   |
|                                                                  |
|  ONNX (the solution):                                            |
|                                                                  |
|   PyTorch model   ---> ONNX file ---> ONNX Runtime (anywhere)   |
|   TensorFlow model ---> ONNX file ---> same ONNX Runtime         |
|   JAX model ---------> ONNX file ---> same ONNX Runtime         |
|                                                                  |
|   Train in any framework. Deploy anywhere.                       |
|   No framework dependency in production.                         |
|                                                                  |
+------------------------------------------------------------------+
```

### The ONNX Ecosystem

```
+------------------------------------------------------------------+
|  WHAT CAN RUN ONNX MODELS?                                       |
+------------------------------------------------------------------+
|                                                                  |
|  ONNX Runtime (Microsoft) -- the main inference engine           |
|    - Windows, Linux, macOS                                       |
|    - Python, C#, C++, Java, JavaScript                           |
|    - CPU, CUDA (NVIDIA), DirectML (Windows GPU), CoreML (Apple)  |
|    - Azure ML uses ONNX Runtime internally                        |
|                                                                  |
|  TensorRT (NVIDIA)                                               |
|    - Parses ONNX graphs and compiles to NVIDIA GPU code          |
|    - Maximum performance on NVIDIA hardware                       |
|                                                                  |
|  OpenVINO (Intel)                                                |
|    - Parses ONNX graphs for Intel CPU/GPU inference              |
|    - Edge devices (Intel Movidius, Raspberry Pi)                 |
|                                                                  |
|  Web browsers                                                    |
|    - ONNX.js: runs in any browser (no server needed)             |
|    - WebNN API: hardware-accelerated in browser                  |
|                                                                  |
|  Mobile                                                          |
|    - iOS: Core ML (convert ONNX -> CoreML format)                |
|    - Android: ONNX Runtime Mobile                                |
|                                                                  |
+------------------------------------------------------------------+
```

C# analogy:
```csharp
// ONNX is like a compiled .dll that any .NET runtime can load.
//
// You built your library in C# 12.
// The .dll can run on .NET 8, .NET Framework 4.8, Mono, Unity.
// The source code is C# but the compiled output is cross-platform.
//
// Similarly:
// You trained your model in PyTorch (your "source code").
// The .onnx file can run on ONNX Runtime, TensorRT, OpenVINO.
// The training framework is PyTorch but the deployment is cross-platform.
//
// Better yet:
// ONNX Runtime has a C# SDK (Microsoft.ML.OnnxRuntime on NuGet)!
// You can call your Python-trained model directly from C# in production.
// This is a real production pattern at many companies.
```

---

## Part 2: The ONNX File Format

ONNX models are stored as Protobuf files with a defined schema.

```
+------------------------------------------------------------------+
|  ONNX FILE STRUCTURE                                             |
+------------------------------------------------------------------+
|                                                                  |
|  ModelProto (the top-level object)                               |
|  |--> ir_version: 8              (ONNX IR version)              |
|  |--> opset_imports: [17]        (operator set version used)     |
|  |--> model_version: 1           (your model version)           |
|  |--> doc_string: "My LLM"       (description)                   |
|  |--> graph: GraphProto          (the computation graph)         |
|       |--> name: "main_graph"                                    |
|       |--> input: [ValueInfoProto]   (graph inputs, with shapes) |
|       |--> output: [ValueInfoProto]  (graph outputs, with shapes)|
|       |--> initializer: [TensorProto] (weights stored here!)     |
|       |--> node: [NodeProto]         (operations)                |
|            |--> op_type: "MatMul"                               |
|            |--> input: ["input", "weight"]                       |
|            |--> output: ["matmul_output"]                        |
|            |--> attribute: [...]                                 |
|                                                                  |
+------------------------------------------------------------------+
```

### Key Concepts: Initializers and Nodes

```
INITIALIZERS = weight tensors stored in the file
  - Your model's trained weights
  - Stored as TensorProto (name, data_type, dims, raw_data)
  - Can be large (the whole model)

NODES = operations in the computation graph
  - MatMul, Add, Softmax, LayerNorm, Transpose, etc.
  - Each node takes named tensor inputs and produces named outputs
  - Together they form a DAG (Directed Acyclic Graph)

GRAPH = initializers + nodes, connected by named tensors
  Data flows from input node through operations to output node
  Every intermediate tensor has a name (like a variable)

Example flow for a simple linear layer (y = x @ W + b):
  Node 1: MatMul
    inputs:  ["x", "weight_W"]   (x from user, W from initializer)
    outputs: ["matmul_out"]
  Node 2: Add
    inputs:  ["matmul_out", "bias_b"]
    outputs: ["y"]
```

---

## Part 3: Exporting a PyTorch Model to ONNX

### Basic Export

```python
import torch
import torch.onnx

# A simple model to demonstrate export
class TinyTransformerBlock(torch.nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.GELU(),
            torch.nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# Create model and set to eval mode
model = TinyTransformerBlock()
model.eval()

# Create dummy input that matches the shape your model expects
# Shape: (batch_size, sequence_length, d_model)
dummy_input = torch.randn(1, 16, 128)

# Export to ONNX
torch.onnx.export(
    model,                      # The PyTorch model
    dummy_input,                # Example input (used for tracing)
    "tiny_transformer.onnx",    # Output file path
    export_params=True,         # Store weights inside the ONNX file
    opset_version=17,           # ONNX opset version (17 = good modern choice)
    do_constant_folding=True,   # Fold constant operations for optimization
    input_names=["input"],      # Name the input tensor
    output_names=["output"],    # Name the output tensor
    dynamic_axes={              # Allow variable sequence length
        "input":  {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"}
    }
)

print("Export successful: tiny_transformer.onnx")
```

### Understanding dynamic_axes

```
+------------------------------------------------------------------+
|  STATIC VS DYNAMIC SHAPES                                        |
+------------------------------------------------------------------+
|                                                                  |
|  WITHOUT dynamic_axes:                                           |
|    Model is exported for EXACTLY the input shape used for tracing|
|    dummy_input shape: (1, 16, 128)                               |
|    Model can ONLY accept: (1, 16, 128)                           |
|    Different sequence length? ERROR.                             |
|    Different batch size? ERROR.                                  |
|                                                                  |
|  WITH dynamic_axes:                                              |
|    "input":  {0: "batch_size", 1: "seq_len"}                     |
|    Means: axis 0 and axis 1 can be any size                      |
|    Model can accept: (2, 32, 128) or (1, 5, 128) or (8, 512, 128)|
|    Axis 2 (d_model=128) is fixed (not in dynamic_axes).          |
|                                                                  |
|  FOR LLMs: Always make batch_size and seq_len dynamic.           |
|  Sequence length varies per user input. Batch size varies        |
|  depending on whether you are doing batched inference.           |
|                                                                  |
+------------------------------------------------------------------+
```

C# analogy:
```csharp
// Static shapes are like a method with fixed-size arrays:
void ProcessTokens(float[1,16,128] tokens) { }  // Only accepts exactly 1,16,128

// Dynamic shapes are like a method with variable-length inputs:
void ProcessTokens(float[,,] tokens) { }         // Accepts any size
// Or in modern C#:
void ProcessTokens(ReadOnlySpan<float> tokens, int batchSize, int seqLen, int dModel) { }
```

---

## Part 4: Verifying the ONNX Model

Always verify the exported model before using it:

```python
import onnx

# Load the model from disk
model = onnx.load("tiny_transformer.onnx")

# Check model is well-formed (valid ONNX graph)
try:
    onnx.checker.check_model(model)
    print("Model is valid!")
except onnx.checker.ValidationError as e:
    print(f"Model is invalid: {e}")

# Print model info
print("\n=== Model Info ===")
print(f"IR Version: {model.ir_version}")
print(f"Opset: {model.opset_import[0].version}")

# Print graph inputs
print("\n=== Inputs ===")
for inp in model.graph.input:
    shape = [dim.dim_param if dim.dim_param else dim.dim_value
             for dim in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {shape}")

# Print graph outputs
print("\n=== Outputs ===")
for out in model.graph.output:
    shape = [dim.dim_param if dim.dim_param else dim.dim_value
             for dim in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {shape}")

# Count parameters
total_params = sum(
    1
    for init in model.graph.initializer
    for _ in [1]
)
print(f"\nTotal weight tensors: {total_params}")
```

---

## Part 5: Running Inference with ONNX Runtime

Once exported, run the model with ONNX Runtime (no PyTorch needed):

```python
import numpy as np
import onnxruntime as ort

# Create inference session
# Choose execution providers in order of preference
session = ort.InferenceSession(
    "tiny_transformer.onnx",
    providers=[
        "CUDAExecutionProvider",    # Use NVIDIA GPU if available
        "CPUExecutionProvider"      # Fall back to CPU
    ]
)

# Check which provider is actually being used
print("Active provider:", session.get_providers()[0])

# Get input details
for inp in session.get_inputs():
    print(f"Input: {inp.name}, shape: {inp.shape}, type: {inp.type}")

# Create test input as NumPy array
# Note: ONNX Runtime uses NumPy, not PyTorch tensors
test_input = np.random.randn(1, 16, 128).astype(np.float32)

# Run inference
# Input is a dict: {input_name: numpy_array}
outputs = session.run(
    output_names=None,     # None = return all outputs
    input_feed={"input": test_input}
)

# Output is a list of NumPy arrays
print(f"Output shape: {outputs[0].shape}")
print(f"Output dtype: {outputs[0].dtype}")
```

### Comparing PyTorch and ONNX Runtime Outputs

You should always verify the ONNX output matches the PyTorch output:

```python
import torch
import numpy as np
import onnxruntime as ort

# Original PyTorch model
model = TinyTransformerBlock()
model.eval()

# Test input
test_input = np.random.randn(1, 8, 128).astype(np.float32)
torch_input = torch.from_numpy(test_input)

# PyTorch inference
with torch.no_grad():
    torch_output = model(torch_input).numpy()

# ONNX Runtime inference
session = ort.InferenceSession("tiny_transformer.onnx")
ort_output = session.run(None, {"input": test_input})[0]

# Compare outputs
max_diff = np.max(np.abs(torch_output - ort_output))
mean_diff = np.mean(np.abs(torch_output - ort_output))

print(f"Max absolute difference: {max_diff:.8f}")
print(f"Mean absolute difference: {mean_diff:.8f}")

# Typical acceptable tolerance
if max_diff < 1e-4:
    print("PASS: Outputs match within tolerance.")
else:
    print("WARNING: Large discrepancy between PyTorch and ONNX Runtime.")
```

---

## Part 6: ONNX Graph Optimization

ONNX Runtime can optimize the graph before inference:

```python
import onnxruntime as ort
from onnxruntime.transformers import optimizer

# Option 1: Let ONNX Runtime optimize automatically
# (happens at session creation time)
sess_options = ort.SessionOptions()

# Optimization level:
# ORT_DISABLE_ALL:     No optimization
# ORT_ENABLE_BASIC:    Constant folding, dead node elimination
# ORT_ENABLE_EXTENDED: More aggressive optimizations
# ORT_ENABLE_ALL:      All optimizations including hardware-specific
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Optional: save the optimized graph to disk
sess_options.optimized_model_filepath = "tiny_transformer_optimized.onnx"

session = ort.InferenceSession(
    "tiny_transformer.onnx",
    sess_options=sess_options
)

# Option 2: Use the transformer-specific optimizer
# (designed specifically for BERT/GPT-style models)
# pip install onnxruntime-tools
optimized_model = optimizer.optimize_model(
    "tiny_transformer.onnx",
    model_type="gpt2",         # or "bert", "t5", etc.
    num_heads=4,
    hidden_size=128
)
optimized_model.save_model_to_file("tiny_transformer_opt.onnx")
```

### What Graph Optimization Does

```
+------------------------------------------------------------------+
|  ONNX GRAPH OPTIMIZATIONS EXPLAINED                              |
+------------------------------------------------------------------+
|                                                                  |
|  CONSTANT FOLDING                                                |
|    Before: Mul(x, Constant(2.0)) -> Add(result, Constant(1.0))   |
|    After:  Result is pre-computed: Mul(x, Constant(2.0)) + 1.0   |
|    Savings: Fewer operations at runtime                          |
|                                                                  |
|  OPERATOR FUSION                                                 |
|    Before: LayerNorm = Reduce -> Sub -> Pow -> Reduce -> Sqrt ... |
|            (8-10 separate operations)                            |
|    After:  FusedLayerNorm (1 operation, custom kernel)           |
|    Savings: Fewer GPU kernel launches, less memory bandwidth     |
|                                                                  |
|  DEAD NODE ELIMINATION                                           |
|    Some operations produce values that are never used.           |
|    The optimizer identifies and removes them.                    |
|                                                                  |
|  ATTENTION FUSION                                                |
|    Before: Q, K, V projections + MatMul + Scale + Softmax + ...  |
|            (many separate operations)                            |
|    After:  FusedAttention (1 fused kernel using Flash Attention) |
|    Savings: 10-50% speedup for transformer models               |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 7: INT8 Quantization of ONNX Models

ONNX Runtime has its own quantization tools:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization: scale factors computed at runtime
# Fast to apply. No calibration dataset needed.
quantize_dynamic(
    model_input="tiny_transformer.onnx",
    model_output="tiny_transformer_int8.onnx",
    weight_type=QuantType.QInt8   # Quantize weights to INT8
)

print("INT8 quantized model saved.")

# Compare sizes
import os
fp32_size = os.path.getsize("tiny_transformer.onnx") / 1024 / 1024
int8_size = os.path.getsize("tiny_transformer_int8.onnx") / 1024 / 1024
print(f"fp32: {fp32_size:.2f} MB")
print(f"INT8: {int8_size:.2f} MB")
print(f"Compression: {fp32_size/int8_size:.1f}x")

# Run the quantized model
session = ort.InferenceSession("tiny_transformer_int8.onnx")
test_input = np.random.randn(1, 8, 128).astype(np.float32)
output = session.run(None, {"input": test_input})[0]
print(f"INT8 output shape: {output.shape}")
```

---

## Part 8: ONNX in C# (The .NET Connection!)

This is relevant to you as a .NET developer.
ONNX Runtime has an official C# SDK:

```csharp
// Install: dotnet add package Microsoft.ML.OnnxRuntime

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Load the ONNX model
using var session = new InferenceSession("tiny_transformer.onnx");

// Prepare input as a DenseTensor (like a NumPy array in C#)
// Shape: batch=1, seq_len=8, d_model=128
var inputData = new float[1 * 8 * 128];
var random = new Random();
for (int i = 0; i < inputData.Length; i++)
    inputData[i] = (float)random.NextGaussian();

var inputTensor = new DenseTensor<float>(inputData, new[] { 1, 8, 128 });

// Create input dictionary
var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("input", inputTensor)
};

// Run inference
using var results = session.Run(inputs);

// Get output
var output = results.First().AsTensor<float>();
Console.WriteLine($"Output shape: {string.Join(", ", output.Dimensions.ToArray())}");
Console.WriteLine($"First value: {output[0, 0, 0]}");
```

```
+------------------------------------------------------------------+
|  WHY THIS MATTERS FOR .NET DEVELOPERS                            |
+------------------------------------------------------------------+
|                                                                  |
|  Real production pattern:                                        |
|                                                                  |
|  1. Data Scientists train model in Python (PyTorch, TensorFlow). |
|  2. Export to ONNX.                                              |
|  3. .NET backend loads ONNX Runtime.                             |
|  4. Your C# ASP.NET Core API calls the ONNX model directly.      |
|  5. No Python in production. No Flask. No FastAPI.               |
|  6. Native .NET performance + .NET deployment tooling.           |
|                                                                  |
|  This is how many enterprises deploy ML in .NET shops.           |
|  Azure ML exports models as ONNX automatically.                  |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 9: Limitations of ONNX for LLMs

ONNX is not the best choice for large LLMs. Here is why:

```
+------------------------------------------------------------------+
|  ONNX LIMITATIONS FOR LLMs                                       |
+------------------------------------------------------------------+
|                                                                  |
|  PROBLEM 1: KV-cache state management                            |
|    LLMs maintain a KV-cache between token generations.           |
|    ONNX is designed for stateless operations.                    |
|    Workaround: pass KV-cache as explicit inputs/outputs.         |
|    Result: very complex graphs with hundreds of I/O tensors.     |
|                                                                  |
|  PROBLEM 2: Dynamic sequence lengths                             |
|    ONNX handles dynamic shapes, but some optimizations break.    |
|    Some operators behave differently at compile-time vs runtime. |
|                                                                  |
|  PROBLEM 3: File size                                            |
|    A 7B fp16 ONNX model is ~14 GB. Multiple files needed.        |
|    GGUF stores the same model in ~4 GB with INT4 quantization.   |
|                                                                  |
|  PROBLEM 4: Export complexity                                     |
|    Large models with custom operations may fail to export.       |
|    Some PyTorch operations have no ONNX equivalent.              |
|                                                                  |
|  WHEN ONNX IS STILL THE RIGHT CHOICE FOR LLMs:                   |
|    - Smaller models (< 1B parameters)                            |
|    - Enterprise .NET/Java/C++ environments                       |
|    - Azure ML or other ONNX-native platforms                     |
|    - When standardization across frameworks is required          |
|    - Edge deployment (mobile, IoT devices)                       |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 10: ONNX vs GGUF vs TorchAO Summary

```
+------------------------------------------------------------------+
|  THREE DEPLOYMENT FORMATS COMPARED                               |
+------------------------------------------------------------------+
|                                                                  |
|              | GGUF        | ONNX         | TorchAO              |
|  ------------|-------------|--------------|-------------------   |
|  Format type | File format | File format  | In-memory (Python)   |
|  Quantized?  | Yes (INT4)  | Optional     | Yes (INT4/INT8)      |
|  Framework   | llama.cpp   | ONNX Runtime | PyTorch              |
|  Languages   | C++, Python | Any language | Python only          |
|  Best for    | Consumer    | Enterprise   | Research/custom      |
|              | laptops,    | cross-lang,  | models, GPU serving  |
|              | Ollama      | Azure ML     |                      |
|  LLM support | Excellent   | Difficult    | Excellent (PyTorch)  |
|  Portability | Medium      | Excellent    | Low (Python/PyTorch) |
|  Ease of use | Very easy   | Moderate     | Easy (for PyTorch)   |
|  .NET use    | Via Ollama  | Native SDK!  | Not applicable       |
|              | REST API    |              |                      |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Summary

```
+------------------------------------------------------------------+
|  LESSON 04 SUMMARY                                               |
+------------------------------------------------------------------+
|                                                                  |
|  1. What ONNX Is                                                 |
|     Framework-agnostic neural network format.                    |
|     Train in PyTorch. Run in any language or platform.           |
|                                                                  |
|  2. ONNX File Structure                                          |
|     Computation graph: nodes (operations) + initializers (weights)|
|     Stored as Protobuf. Opset version controls available ops.    |
|                                                                  |
|  3. Exporting from PyTorch                                       |
|     torch.onnx.export(). Tracing-based.                          |
|     Use dynamic_axes for variable sequence lengths.              |
|                                                                  |
|  4. ONNX Runtime                                                 |
|     Runs ONNX models. Execution providers for CPU/GPU/Mobile.    |
|     Graph optimization can fuse operations for speed.            |
|                                                                  |
|  5. .NET Integration                                             |
|     Microsoft.ML.OnnxRuntime NuGet package.                      |
|     Call Python-trained models natively from C#.                 |
|                                                                  |
|  6. Limitations for Large LLMs                                   |
|     KV-cache state is awkward. File sizes large.                 |
|     Better for small models and enterprise cross-language use.   |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Quiz Questions

1. What does ONNX stand for? What problem was it created to solve?

2. What are "dynamic axes" in ONNX export, and why are they important for LLMs?

3. You are a C# backend developer. A data scientist hands you an ONNX file.
   What NuGet package do you need, and roughly how would you call the model?

4. What is the difference between `dynamic` and `static` quantization in ONNX Runtime?
   Which requires a calibration dataset?

5. What is "operator fusion" in ONNX graph optimization?
   Give an example of operations that might be fused.

6. Why is ONNX harder to use with large LLMs compared to small models?
   Name two specific problems.

7. Compare GGUF, ONNX, and TorchAO in one sentence each.
   Then name the best use case for each.

8. You exported a model and the ONNX output differs slightly from PyTorch.
   Is this expected? What is an acceptable tolerance level?

---

*Next lesson: Ollama and FastAPI -- serve your local LLM over HTTP.*
*File: lessons/05_ollama_fastapi.md*

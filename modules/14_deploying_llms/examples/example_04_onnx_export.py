# =============================================================================
# Module 14 - Deploying LLMs
# Example 04: ONNX Export -- Framework-Agnostic Deployment
# =============================================================================
#
# WHAT THIS FILE TEACHES:
#   - What ONNX is and why it lets models run anywhere
#   - How to export a PyTorch model to ONNX format
#   - How to load and run an ONNX model with ONNX Runtime
#   - How to verify ONNX output matches PyTorch output
#   - Why ONNX is important for .NET developers
#
# GLOSSARY:
#   ONNX          - Open Neural Network Exchange. A file format for neural
#                   networks that any framework or language can read and run.
#   ONNX Runtime  - Microsoft's high-performance inference engine for ONNX.
#                   Runs on CPU, GPU, mobile, edge devices. Works in C# too!
#   computation graph - A neural network represented as a graph.
#                   Nodes = operations (MatMul, Add, Softmax).
#                   Edges = data flowing between operations.
#   opset         - ONNX Operator Set version. Like an API version.
#                   opset 17 is a good modern choice (stable, widely supported).
#   dynamic axes  - Marks certain tensor dimensions as variable-size.
#                   "batch_size" and "seq_len" should be dynamic for LLMs.
#   tracing       - How ONNX export works: it runs the model with sample input
#                   and records every operation performed.
#   execution provider - Which hardware runs the model in ONNX Runtime.
#                   CPUExecutionProvider, CUDAExecutionProvider, etc.
#
# C# ANALOGY:
#   ONNX is like a compiled .dll that any .NET runtime can load.
#   - You train in PyTorch (like writing C# source code)
#   - You export to ONNX (like compiling to a .dll)
#   - ONNX Runtime loads the .dll in C#, Java, or Python (cross-platform)
#
#   BONUS: ONNX Runtime has an official C# NuGet package!
#   Install: dotnet add package Microsoft.ML.OnnxRuntime
#   You can call Python-trained models directly from C# in production!
#
# REQUIREMENTS:
#   Part A: numpy only (no PyTorch -- shows the concept)
#   Part B: pip install torch onnx onnxruntime
#
# =============================================================================

import numpy as np    # numpy for Part A
import os             # os for file operations

# =============================================================================
# PART A: ONNX CONCEPTS WITHOUT LIBRARIES
#
# Understand what ONNX is and what makes it powerful.
# =============================================================================

print("=" * 60)
print("PART A: ONNX Concepts and Structure")
print("=" * 60)

# ---------------------------------------------------------
# Show the ONNX ecosystem with ASCII diagram
# ---------------------------------------------------------

def show_onnx_ecosystem():
    """Print the ONNX ecosystem diagram."""
    print("\nThe ONNX Ecosystem:")
    print("-" * 50)
    print("")
    print("TRAINING (any framework):")
    print("  PyTorch model ---|")
    print("  TensorFlow model +--> ONNX file (.onnx)")
    print("  JAX model     ---|")
    print("")
    print("DEPLOYMENT (any language or platform):")
    print("  ONNX file ---> ONNX Runtime (Python, C#, Java, C++)")
    print("  ONNX file ---> TensorRT (NVIDIA GPU, maximum speed)")
    print("  ONNX file ---> OpenVINO (Intel CPU/GPU)")
    print("  ONNX file ---> ONNX.js  (browser, no server needed)")
    print("  ONNX file ---> Core ML  (iPhone/iPad via conversion)")
    print("  ONNX file ---> ORT Mobile (Android)")
    print("")
    print("KEY BENEFIT: Train ONCE, deploy EVERYWHERE.")
    print("No framework dependency in production.")
    print("C# production backend can call a PyTorch-trained model directly.")
    print("-" * 50)


def show_onnx_structure():
    """Print the ONNX file structure."""
    print("\nONNX File Internal Structure (Protobuf format):")
    print("-" * 50)
    print("ModelProto")
    print("  ir_version: 8            (ONNX IR version)")
    print("  opset_imports: [17]      (operator set version)")
    print("  graph: GraphProto")
    print("    inputs:")
    print("      'input': shape=[batch_size, seq_len, d_model]  float32")
    print("    outputs:")
    print("      'output': shape=[batch_size, seq_len, d_model] float32")
    print("    initializers: (weight tensors stored here)")
    print("      'layers.0.weight': [256, 256]  float32")
    print("      'layers.0.bias':   [256]        float32")
    print("      'layers.1.weight': [256, 256]  float32")
    print("      ... (all weights)")
    print("    nodes: (operations)")
    print("      Node: op=MatMul")
    print("        inputs:  ['input_reshaped', 'layers.0.weight']")
    print("        outputs: ['matmul_0_out']")
    print("      Node: op=Add")
    print("        inputs:  ['matmul_0_out', 'layers.0.bias']")
    print("        outputs: ['linear_0_out']")
    print("      Node: op=Relu")
    print("        inputs:  ['linear_0_out']")
    print("        outputs: ['relu_0_out']")
    print("      ... (all operations in order)")
    print("-" * 50)
    print("")
    print("ONNX is a computation GRAPH not a procedure.")
    print("Like a dataflow diagram: data flows through nodes.")
    print("C# analogy: Expression<Func<T>> instead of a Func<T>.")
    print("You can inspect, optimize, and transform the graph.")


def show_dynamic_axes():
    """Explain why dynamic axes matter."""
    print("\nWhy Dynamic Axes Matter for LLMs:")
    print("-" * 50)
    print("")
    print("WITHOUT dynamic axes:")
    print("  Model exported with dummy_input shape: (1, 16, 256)")
    print("  Model can ONLY accept inputs of shape (1, 16, 256)")
    print("  User sends 32 tokens? ERROR.")
    print("  User sends 2 samples in a batch? ERROR.")
    print("")
    print("WITH dynamic axes:")
    print("  dynamic_axes = {")
    print("      'input':  {0: 'batch_size', 1: 'seq_len'},")
    print("      'output': {0: 'batch_size', 1: 'seq_len'}")
    print("  }")
    print("  Axis 0 (batch_size) = variable: 1, 2, 4, 8, ...")
    print("  Axis 1 (seq_len)    = variable: 5, 16, 32, 128, ...")
    print("  Axis 2 (d_model=256) = FIXED (not in dynamic_axes)")
    print("")
    print("FOR LLMs: ALWAYS make batch_size and seq_len dynamic.")
    print("Users will send different length messages!")
    print("-" * 50)


show_onnx_ecosystem()
show_onnx_structure()
show_dynamic_axes()

# =============================================================================
# PART B: REAL ONNX EXPORT WITH PYTORCH
#
# Export a small transformer-like model to ONNX,
# then run it with ONNX Runtime and verify outputs match.
# =============================================================================

print("\n" + "=" * 60)
print("PART B: Export PyTorch Model to ONNX")
print("=" * 60)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print("PyTorch installed. Version:", torch.__version__)
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install: pip install torch")

# Check if onnx is available
try:
    import onnx
    ONNX_AVAILABLE = True
    print("onnx installed. Version:", onnx.__version__)
except ImportError:
    ONNX_AVAILABLE = False
    print("onnx not installed. Install: pip install onnx")

# Check if onnxruntime is available
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
    print("onnxruntime installed. Version:", ort.__version__)
except ImportError:
    ORT_AVAILABLE = False
    print("onnxruntime not installed. Install: pip install onnxruntime")

if not TORCH_AVAILABLE:
    print("\nShowing ONNX export code (educational mode, PyTorch not installed):")
    print("-" * 40)
    print("")
    print("# 1. Define a simple model")
    print("class TinyTransformer(nn.Module):")
    print("    def __init__(self, d_model=128, n_layers=2):")
    print("        super().__init__()")
    print("        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) * n_layers])")
    print("        self.norm = nn.LayerNorm(d_model)")
    print("    def forward(self, x):")
    print("        for layer in self.layers:")
    print("            x = torch.relu(layer(x))")
    print("        return self.norm(x)")
    print("")
    print("# 2. Export to ONNX")
    print("model = TinyTransformer()")
    print("model.eval()")
    print("dummy = torch.randn(1, 16, 128)  # (batch, seq_len, d_model)")
    print("torch.onnx.export(")
    print("    model,")
    print("    dummy,")
    print("    'tiny_transformer.onnx',")
    print("    export_params=True,    # Include weights in the file")
    print("    opset_version=17,      # ONNX operator set version 17")
    print("    do_constant_folding=True,  # Pre-compute constant expressions")
    print("    input_names=['input'],")
    print("    output_names=['output'],")
    print("    dynamic_axes={")
    print("        'input':  {0: 'batch_size', 1: 'seq_len'},  # variable dims")
    print("        'output': {0: 'batch_size', 1: 'seq_len'}")
    print("    }")
    print(")")
    print("")
    print("# 3. Run with ONNX Runtime (no PyTorch needed!)")
    print("session = ort.InferenceSession('tiny_transformer.onnx')")
    print("test_input = np.random.randn(1, 16, 128).astype(np.float32)")
    print("outputs = session.run(None, {'input': test_input})")
    print("print(outputs[0].shape)  # -> (1, 16, 128)")

else:
    # ---------------------------------------------------------
    # Define a small model for the demonstration
    # ---------------------------------------------------------

    class TinyTransformer(nn.Module):
        """
        Small transformer-like model for ONNX export demonstration.
        Has:
          - Two linear layers (feed-forward)
          - One layer normalization
          - ReLU activation
        This is a simplified version of one transformer block.
        """
        def __init__(self, d_model=128, n_layers=2):
            super().__init__()   # Call nn.Module's constructor

            # ModuleList: a list of layers that PyTorch tracks
            # C# analogy: List<Layer> where PyTorch knows each item has parameters
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model)    # Square linear: 128 -> 128
                for _ in range(n_layers)
            ])

            # LayerNorm: normalizes each token's vector to have mean=0, std=1
            # Stabilizes training and inference
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            """
            x: input tensor of shape (batch_size, seq_len, d_model)
            returns: tensor of same shape
            """
            # Pass through each linear layer with ReLU activation
            for layer in self.layers:
                x = torch.relu(layer(x))   # ReLU: max(0, x) per element
            return self.norm(x)            # Normalize each token vector


    # Create model and set to eval mode
    torch.manual_seed(42)   # Fix random seed for reproducible weights
    model = TinyTransformer(d_model=128, n_layers=2)
    model.eval()   # Switch off dropout and training-specific behavior

    BATCH = 1        # Batch size for the dummy input (used for tracing)
    SEQ_LEN = 16     # Sequence length for the dummy input
    D_MODEL = 128    # Must match the model's d_model

    # Dummy input: required for ONNX export's tracing mechanism
    # ONNX export works by running the model once with this input
    # and recording every operation. The values don't matter -- the shape does.
    dummy_input = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    print(f"\nModel: TinyTransformer(d_model={D_MODEL}, n_layers=2)")
    print(f"Dummy input shape: {dummy_input.shape}")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {param_count:,}")

    # ---------------------------------------------------------
    # Export to ONNX
    # ---------------------------------------------------------

    onnx_path = "tiny_transformer.onnx"   # Output file name

    print(f"\nExporting to ONNX: {onnx_path}")

    torch.onnx.export(
        model,              # The PyTorch model to export
        dummy_input,        # Example input (used for tracing)
        onnx_path,          # Output file path
        export_params=True, # Store trained weights inside the ONNX file
        opset_version=17,   # ONNX opset 17: stable, supported by most tools
        do_constant_folding=True,   # Pre-compute constant expressions (faster inference)
        input_names=["input"],      # Name the input tensor (any name is fine)
        output_names=["output"],    # Name the output tensor
        dynamic_axes={              # Mark which dimensions are variable
            "input":  {0: "batch_size", 1: "seq_len"},   # batch and seq can change
            "output": {0: "batch_size", 1: "seq_len"}    # output follows input shape
        }
    )

    # Check the file was created and its size
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Export successful!")
    print(f"File size: {file_size_mb:.2f} MB")

    # ---------------------------------------------------------
    # Verify the ONNX model is valid
    # ---------------------------------------------------------

    if ONNX_AVAILABLE:
        print("\nVerifying ONNX model...")

        # Load the model from disk
        onnx_model = onnx.load(onnx_path)

        # check_model validates the graph structure (shapes, operator types, etc.)
        try:
            onnx.checker.check_model(onnx_model)
            print("Model validation: PASSED")
        except onnx.checker.ValidationError as e:
            print(f"Model validation FAILED: {e}")

        # Print model metadata
        print(f"ONNX IR version: {onnx_model.ir_version}")
        print(f"Opset version:   {onnx_model.opset_import[0].version}")

        # Print input information
        print("\nInputs:")
        for inp in onnx_model.graph.input:
            # dim.dim_param = dynamic dimension name (e.g., "batch_size")
            # dim.dim_value = static dimension value (e.g., 128)
            shape = [
                dim.dim_param if dim.dim_param else dim.dim_value
                for dim in inp.type.tensor_type.shape.dim
            ]
            print(f"  {inp.name}: shape={shape}")

        # Print output information
        print("Outputs:")
        for out in onnx_model.graph.output:
            shape = [
                dim.dim_param if dim.dim_param else dim.dim_value
                for dim in out.type.tensor_type.shape.dim
            ]
            print(f"  {out.name}: shape={shape}")

        # Count weight tensors
        n_initializers = len(onnx_model.graph.initializer)
        print(f"\nWeight tensors (initializers): {n_initializers}")

        # Count operations
        n_nodes = len(onnx_model.graph.node)
        print(f"Operations (nodes): {n_nodes}")

        # Show operation types used in the graph
        op_types = set(node.op_type for node in onnx_model.graph.node)
        # set() removes duplicates -- each op type appears only once
        print(f"Operation types: {sorted(op_types)}")
        # sorted() alphabetizes the set for consistent display

    # ---------------------------------------------------------
    # Run inference with ONNX Runtime (no PyTorch needed!)
    # ---------------------------------------------------------

    if ORT_AVAILABLE:
        print("\nRunning inference with ONNX Runtime...")

        # Create an inference session
        # providers = list of execution backends in priority order
        # CUDAExecutionProvider: use NVIDIA GPU if available
        # CPUExecutionProvider: always available, fall back to this
        session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Which provider is actually being used?
        active_provider = session.get_providers()[0]
        print(f"Active execution provider: {active_provider}")

        # Show input info from the session
        for inp in session.get_inputs():
            print(f"Input: {inp.name}, shape: {inp.shape}, type: {inp.type}")

        # Create test input as NumPy array
        # NOTE: ONNX Runtime uses NumPy arrays, not PyTorch tensors
        # This is what lets it run without PyTorch installed
        test_input = np.random.randn(1, 16, 128).astype(np.float32)
        # .astype(np.float32) ensures we use 32-bit float (ONNX expects float32)

        # Run inference
        # input_feed: dict mapping input name to numpy array
        # output_names=None: return ALL outputs
        outputs = session.run(
            output_names=None,             # Return all outputs
            input_feed={"input": test_input}  # Match the name we set in export
        )

        # outputs is a list of numpy arrays (one per model output)
        print(f"ONNX Runtime output shape: {outputs[0].shape}")
        print(f"ONNX Runtime output dtype: {outputs[0].dtype}")

        # ---------------------------------------------------------
        # Compare PyTorch output vs ONNX Runtime output
        # ---------------------------------------------------------

        print("\nComparing PyTorch output vs ONNX Runtime output...")

        # Run the same input through PyTorch
        torch_input = torch.from_numpy(test_input)   # Convert numpy to torch tensor
        with torch.no_grad():
            torch_output = model(torch_input).numpy()  # Convert back to numpy for comparison

        # ONNX Runtime output
        ort_output = outputs[0]

        # Compute absolute differences
        max_diff = np.max(np.abs(torch_output - ort_output))
        mean_diff = np.mean(np.abs(torch_output - ort_output))

        print(f"Max absolute difference:  {max_diff:.8f}")
        print(f"Mean absolute difference: {mean_diff:.8f}")

        # 1e-5 (0.00001) is the typical tolerance for ONNX export accuracy
        # Small differences come from float rounding in different math libraries
        TOLERANCE = 1e-5
        if max_diff < TOLERANCE:
            print(f"PASS: Outputs match within tolerance ({TOLERANCE})")
        else:
            print(f"WARNING: Difference exceeds tolerance. Max diff: {max_diff:.2e}")

        # ---------------------------------------------------------
        # Test that dynamic axes work (different sequence lengths)
        # ---------------------------------------------------------

        print("\nTesting dynamic axes (different input shapes):")

        test_shapes = [
            (1, 5, 128),    # Short sequence
            (2, 16, 128),   # Batch of 2
            (1, 64, 128),   # Long sequence
        ]

        for shape in test_shapes:
            test_in = np.random.randn(*shape).astype(np.float32)
            # *shape unpacks the tuple: (1, 5, 128) becomes 1, 5, 128
            try:
                out = session.run(None, {"input": test_in})
                print(f"  Input {shape} -> Output {out[0].shape}  OK")
            except Exception as e:
                print(f"  Input {shape} -> ERROR: {e}")

        # Clean up the ONNX file
        os.remove(onnx_path)
        print(f"\nCleaned up: {onnx_path}")

# =============================================================================
# C# INTEGRATION CODE EXAMPLE
# (Shows how you'd call this model from a .NET application)
# =============================================================================

print("\n" + "=" * 60)
print("C# Integration: Calling ONNX from .NET")
print("=" * 60)
print("")
print("// Step 1: Add NuGet package")
print("// dotnet add package Microsoft.ML.OnnxRuntime")
print("")
print("// Step 2: Use in C# (same model, no Python needed in production!)")
print("""
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Load the ONNX model file
using var session = new InferenceSession("tiny_transformer.onnx");

// Prepare input data
// Shape: (batch=1, seq_len=16, d_model=128)
int batchSize = 1, seqLen = 16, dModel = 128;
var data = new float[batchSize * seqLen * dModel];
// ... fill data array with your token embeddings ...

// Create a tensor from the float array
var inputTensor = new DenseTensor<float>(data, new[] { batchSize, seqLen, dModel });

// Create the input dictionary
var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("input", inputTensor)
};

// Run inference -- no Python, no PyTorch, no GPU required!
using var results = session.Run(inputs);

// Get the output tensor
var output = results.First().AsTensor<float>();
Console.WriteLine($"Output shape: [{string.Join(", ", output.Dimensions.ToArray())}]");
""")

print("REAL PRODUCTION PATTERN:")
print("  1. Data Scientists train model in Python (PyTorch)")
print("  2. Export to .onnx file")
print("  3. Check .onnx file into your repo or artifact store")
print("  4. C# ASP.NET Core API loads .onnx at startup")
print("  5. Each HTTP request runs model.Run(input)")
print("  6. NO Python in production. All .NET. Azure-deployable.")
print("")
print("This is how many enterprise teams deploy ML in .NET shops.")
print("Azure ML exports models as ONNX automatically.")

print("\n" + "=" * 60)
print("Key Takeaways:")
print("=" * 60)
print("1. ONNX = 'Write Once, Run Anywhere' for neural networks.")
print("2. torch.onnx.export() traces the model to capture the graph.")
print("3. dynamic_axes lets the model accept variable batch size and seq length.")
print("4. ONNX Runtime works in Python, C#, Java, C++ -- no framework needed.")
print("5. Always verify ONNX output matches PyTorch output before deploying.")
print("6. For .NET teams: Microsoft.ML.OnnxRuntime is the production path.")

# =============================================================================
# Module 14 - Deploying LLMs
# Exercise 04: ONNX Export
# =============================================================================
#
# INSTRUCTIONS:
#   Complete each TODO section.
#   Task 1: numpy only (concept questions, no libraries)
#   Task 2-4: pip install torch onnx onnxruntime
#
# WHAT YOU PRACTICE:
#   - Understanding ONNX structure (multiple choice)
#   - Exporting a PyTorch model with correct parameters
#   - Verifying export correctness by comparing outputs
#   - Identifying and fixing a common dynamic axes mistake
#
# =============================================================================

import numpy as np
import os

print("=" * 60)
print("Exercise 04: ONNX Export")
print("=" * 60)

# =============================================================================
# TASK 1: ONNX Concept Questions
# (No code required -- fill in the answers in the print statements)
# =============================================================================

print("\n--- Task 1: ONNX Concept Questions ---")

questions = {
    "Q1": (
        "What does ONNX stand for?",
        # TODO: fill in the answer below
        "TODO: your answer here"
    ),
    "Q2": (
        "Why are 'dynamic axes' important when exporting an LLM?",
        "TODO: your answer here"
    ),
    "Q3": (
        "Name THREE languages/platforms that can run an ONNX model.",
        "TODO: your answer here"
    ),
    "Q4": (
        "What NuGet package lets you run ONNX models in C#?",
        "TODO: your answer here"
    ),
    "Q5": (
        "What are 'initializers' in an ONNX graph?",
        "TODO: your answer here"
    ),
    "Q6": (
        "What is 'operator fusion' and why does it speed up inference?",
        "TODO: your answer here"
    ),
}

for qid, (question, answer) in questions.items():
    print(f"\n{qid}: {question}")
    print(f"  Answer: {answer}")

print("\nExpected answers:")
print("  Q1: Open Neural Network Exchange")
print("  Q2: LLMs receive inputs of varying lengths (different user messages).")
print("      Without dynamic axes, model only accepts the exact shape used during export.")
print("  Q3: Python (ONNX Runtime), C# (.NET), Java, C++, JavaScript, iOS, Android")
print("  Q4: Microsoft.ML.OnnxRuntime")
print("  Q5: The trained weight tensors stored inside the ONNX file.")
print("      Like the DLL's embedded resources -- data that comes with the model.")
print("  Q6: Combining multiple ops (e.g. LayerNorm = 8 ops) into one kernel call.")
print("      Saves memory round-trips between GPU and CPU. Often 10-50% speedup.")

# =============================================================================
# TASK 2: Export a Model to ONNX (Fill in the Blanks)
#
# Complete the torch.onnx.export() call below.
# Then run the file to verify the export worked.
# =============================================================================

print("\n--- Task 2: Export a PyTorch Model to ONNX ---")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Showing expected code.")

if TORCH_AVAILABLE:

    class TextEncoder(nn.Module):
        """
        Simple text encoder: embedding -> 2 linear layers -> output.
        Typical first part of a sentence encoder or small LLM.
        """
        def __init__(self, vocab_size=1000, d_model=64, n_layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(d_model)

        def forward(self, token_ids):
            """
            token_ids: (batch_size, seq_len)   integer tensor
            returns:   (batch_size, seq_len, d_model)  float tensor
            """
            x = self.embed(token_ids)
            for layer in self.layers:
                x = torch.relu(layer(x))
            return self.norm(x)


    torch.manual_seed(42)
    model = TextEncoder(vocab_size=1000, d_model=64, n_layers=2)
    model.eval()

    # Dummy input for tracing (shape determines what gets exported)
    # Shape: (batch_size=1, seq_len=10)
    dummy_ids = torch.randint(0, 1000, (1, 10))

    onnx_path = "text_encoder.onnx"

    # TODO: Complete the export call
    # Fill in each ____ with the correct value
    torch.onnx.export(
        model,          # The model to export
        dummy_ids,      # Dummy input for tracing
        onnx_path,      # Output file path

        export_params=True,        # Include weights in the file (always True for deployment)

        # TODO: What opset version? (use 17 -- modern and well-supported)
        opset_version=None,        # TODO: replace None with 17

        do_constant_folding=True,  # Pre-compute constants for faster inference

        # TODO: Name the inputs and outputs (use descriptive names)
        input_names=None,          # TODO: replace with ["token_ids"]
        output_names=None,         # TODO: replace with ["embeddings"]

        # TODO: Make batch_size and seq_len dynamic
        # Axis 0 = batch_size, Axis 1 = seq_len
        dynamic_axes=None,         # TODO: fill in the dict
        # Hint:
        # {
        #     "token_ids":   {0: "batch_size", 1: "seq_len"},
        #     "embeddings":  {0: "batch_size", 1: "seq_len"}
        # }
    )

    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"Export succeeded! File: {onnx_path} ({size_mb:.2f} MB)")
    else:
        print("Export may have failed. Check for errors above.")

# =============================================================================
# TASK 3: Verify ONNX Output Matches PyTorch Output
#
# After exporting, run the same input through both PyTorch and ONNX Runtime.
# The outputs should be almost identical (difference < 0.0001).
# =============================================================================

print("\n--- Task 3: Verify ONNX Output ---")

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("onnxruntime not installed. Install: pip install onnxruntime")

if TORCH_AVAILABLE and ORT_AVAILABLE and os.path.exists("text_encoder.onnx"):

    def verify_onnx_vs_pytorch(model, onnx_path, test_input_np):
        """
        Run test_input through both PyTorch model and ONNX Runtime.
        Print max and mean absolute difference.
        Return True if difference is within tolerance.

        Parameters:
            model:         PyTorch nn.Module (in eval mode)
            onnx_path:     str -- path to the exported ONNX file
            test_input_np: numpy int32 array of shape (batch, seq_len)

        Returns:
            bool -- True if max difference < 0.001
        """
        # TODO: Run through PyTorch
        # Hint: torch.from_numpy(test_input_np), then model(), then .numpy()
        torch_input = None    # Convert numpy to torch tensor
        with torch.no_grad():
            torch_output = None   # Run model, get output
        torch_output_np = None    # Convert to numpy


        # TODO: Run through ONNX Runtime
        # Hint:
        #   session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        #   outputs = session.run(None, {"token_ids": test_input_np})
        #   ort_output = outputs[0]
        session = None    # TODO: create session
        ort_output = None # TODO: run inference

        if torch_output_np is None or ort_output is None:
            print("TODO not complete yet.")
            return False

        # Compute differences
        max_diff = np.max(np.abs(torch_output_np - ort_output))
        mean_diff = np.mean(np.abs(torch_output_np - ort_output))

        print(f"Max difference:  {max_diff:.8f}")
        print(f"Mean difference: {mean_diff:.8f}")

        TOLERANCE = 1e-4
        if max_diff < TOLERANCE:
            print(f"PASS: Outputs match (tolerance {TOLERANCE})")
            return True
        else:
            print(f"WARNING: Difference exceeds tolerance. Check export settings.")
            return False


    # Test with a different sequence length than the dummy input (tests dynamic axes)
    test_input = np.random.randint(0, 1000, size=(2, 15), dtype=np.int64)
    # Shape (2, 15): batch=2, seq_len=15 (different from dummy_input's batch=1, seq_len=10)
    print(f"Test input shape: {test_input.shape}")

    verify_onnx_vs_pytorch(model, "text_encoder.onnx", test_input)

    # Clean up
    if os.path.exists("text_encoder.onnx"):
        os.remove("text_encoder.onnx")
        print("Cleaned up text_encoder.onnx")

else:
    print("Expected result:")
    print("  Max difference:  < 0.00001")
    print("  Mean difference: < 0.000001")
    print("  PASS: Outputs match (tolerance 0.0001)")

# =============================================================================
# TASK 4: Fix the Dynamic Axes Bug
#
# The code below exports a model WITHOUT dynamic axes.
# The exported model only accepts inputs of exactly (1, 8, 64).
# Fix it so it accepts any batch size and sequence length.
# =============================================================================

print("\n--- Task 4: Fix the Dynamic Axes Bug ---")

if TORCH_AVAILABLE:

    class BuggyModel(nn.Module):
        """Simple model for the dynamic axes exercise."""
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64)

        def forward(self, x):
            return self.linear(x)


    buggy_model = BuggyModel()
    buggy_model.eval()

    dummy = torch.randn(1, 8, 64)   # Fixed shape

    # BUGGY EXPORT (no dynamic axes):
    torch.onnx.export(
        buggy_model,
        dummy,
        "buggy_model.onnx",
        opset_version=17,
        input_names=["x"],
        output_names=["y"]
        # Notice: no dynamic_axes parameter!
    )

    # Test: try a different shape -- this should FAIL or give wrong results
    if ORT_AVAILABLE and os.path.exists("buggy_model.onnx"):
        session_buggy = ort.InferenceSession("buggy_model.onnx",
                                              providers=["CPUExecutionProvider"])

        try:
            wrong_shape = np.random.randn(2, 16, 64).astype(np.float32)   # Different shape
            out = session_buggy.run(None, {"x": wrong_shape})
            print(f"Buggy model with shape (2,16,64): output shape = {out[0].shape}")
            print("(ONNX Runtime may reshape automatically OR error -- depends on version)")
        except Exception as e:
            print(f"Buggy model failed with shape (2,16,64): {type(e).__name__}")

        os.remove("buggy_model.onnx")

    # TODO: Fix the export by adding dynamic_axes
    print("\nFixed export code:")
    print("torch.onnx.export(")
    print("    buggy_model,")
    print("    dummy,")
    print("    'fixed_model.onnx',")
    print("    opset_version=17,")
    print("    input_names=['x'],")
    print("    output_names=['y'],")
    print("    # TODO: add dynamic_axes here to allow any batch_size and seq_len")
    print("    dynamic_axes=None  # <-- replace this!")
    print(")")

    # TODO: Write the correct fixed export (uncomment and fix):
    # torch.onnx.export(
    #     buggy_model,
    #     dummy,
    #     "fixed_model.onnx",
    #     opset_version=17,
    #     input_names=["x"],
    #     output_names=["y"],
    #     dynamic_axes={
    #         "x": {TODO},   # Which axes should be dynamic?
    #         "y": {TODO}
    #     }
    # )

    # If you fix it, test here:
    if os.path.exists("fixed_model.onnx"):
        if ORT_AVAILABLE:
            session_fixed = ort.InferenceSession("fixed_model.onnx",
                                                  providers=["CPUExecutionProvider"])
            test_shapes = [(1, 8, 64), (2, 16, 64), (4, 32, 64)]
            for shape in test_shapes:
                x = np.random.randn(*shape).astype(np.float32)
                try:
                    out = session_fixed.run(None, {"x": x})
                    print(f"Fixed model: input {shape} -> output {out[0].shape}  OK")
                except Exception as e:
                    print(f"Fixed model: input {shape} FAILED: {e}")
        os.remove("fixed_model.onnx")

# =============================================================================
# HINTS
# =============================================================================

print("\n" + "=" * 60)
print("HINTS")
print("=" * 60)
print("""
Task 2 (export call):
  opset_version=17
  input_names=["token_ids"]
  output_names=["embeddings"]
  dynamic_axes={
      "token_ids":  {0: "batch_size", 1: "seq_len"},
      "embeddings": {0: "batch_size", 1: "seq_len"}
  }

Task 3 (verify):
  torch_input = torch.from_numpy(test_input_np.astype(np.int64))
  with torch.no_grad():
      torch_output = model(torch_input).numpy()

  session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
  outputs = session.run(None, {"token_ids": test_input_np})
  ort_output = outputs[0]

Task 4 (fix dynamic axes):
  dynamic_axes={
      "x": {0: "batch_size", 1: "seq_len"},
      "y": {0: "batch_size", 1: "seq_len"}
  }
  Note: axis 2 (d_model=64) is NOT dynamic -- it is always 64.
""")

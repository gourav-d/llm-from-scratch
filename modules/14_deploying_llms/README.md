# Module 14: Deploying LLMs

## What Is This Module About?

You have built an LLM from scratch.
You have trained it. Fine-tuned it. Aligned it.

But right now, your model lives inside a Python script on your laptop.
Nobody else can use it.

**This module teaches you how to take a trained LLM and put it in front of real users.**

That means:
- Making the model small enough to run on normal hardware (quantization)
- Packaging it in standard formats others can use (GGUF, ONNX)
- Running it locally without internet or cloud bills (Ollama)
- Building an API so applications can call it (FastAPI)

> Key Insight: A 70-billion parameter model in fp32 needs 280 GB of RAM.
> That same model in INT4 needs only 35 GB.
> Quantization is what makes local LLMs possible.

---

## Why Does This Matter?

Most tutorials show you how to TRAIN models using cloud GPUs.
But real applications need models that run:

- On a developer's laptop (no cloud cost)
- In production servers (cheap, fast, predictable)
- Without sending user data to external APIs (privacy)
- Offline, in air-gapped environments (security)

This module bridges the gap between "model file on disk" and "working application."

---

## Prerequisites

Before starting this module, you should have completed:

| Module | Why It Is Required |
|--------|-------------------|
| Module 01 - Python Basics | Python syntax, functions, file I/O |
| Module 02 - NumPy | Understanding of tensors and numerical data |
| Module 03 - Neural Networks | How weights and parameters work |
| Module 04 - Transformers | The architecture you will be deploying |
| Module 05 - Building LLM | GPT model structure, forward pass |
| Module 06 - Training Loop | You know what a trained model file contains |

You do NOT need Module 12 or 13 completed first.
This module is about deployment, not training techniques.

---

## The Deployment Problem Visualized

```
+========================================================================+
|                    THE GAP WE ARE CROSSING                            |
+========================================================================+
|                                                                        |
|  What you have now:                                                    |
|  +-----------------------------------------------------------------+   |
|  |  model.pt (PyTorch weights)                                     |   |
|  |  Size: 28 GB (7B params x 4 bytes per float32)                  |   |
|  |  Runs on: GPU with 40+ GB VRAM                                  |   |
|  |  Access: python train.py  (just you, on your machine)           |   |
|  +-----------------------------------------------------------------+   |
|                                                                        |
|  What you want:                                                        |
|  +-----------------------------------------------------------------+   |
|  |  model.gguf (quantized, compressed)                             |   |
|  |  Size: 4 GB (INT4 quantization)                                 |   |
|  |  Runs on: Any laptop with 8 GB RAM                              |   |
|  |  Access: http://localhost:8000/chat  (anyone, any device)       |   |
|  +-----------------------------------------------------------------+   |
|                                                                        |
|  This module teaches you how to cross this gap.                       |
|                                                                        |
+========================================================================+
```

---

## The 5-Step Deployment Pipeline

Every real deployment follows some version of this sequence:

```
+--------------------------------------------------------------------+
|                   DEPLOYMENT PIPELINE                              |
+--------------------------------------------------------------------+
|                                                                    |
|  Step 1: QUANTIZE                                                  |
|  Trained model (fp32, huge) --> Quantized model (INT4/INT8, small) |
|                                                                    |
|  Step 2: PACKAGE                                                   |
|  PyTorch weights --> Standard format (GGUF or ONNX)               |
|  GGUF: for local use with llama.cpp / Ollama                       |
|  ONNX: for cross-platform production deployment                    |
|                                                                    |
|  Step 3: SERVE                                                     |
|  Packaged model --> Running server (Ollama or custom server)       |
|  Ollama: dead-simple local serving                                 |
|  Custom: full control with FastAPI                                 |
|                                                                    |
|  Step 4: API                                                       |
|  Running server --> HTTP endpoints                                 |
|  POST /chat   --> returns model response                           |
|  GET /health  --> confirms server is alive                         |
|                                                                    |
|  Step 5: CLIENT                                                    |
|  HTTP endpoints --> Application (web app, CLI, Slack bot, etc.)   |
|                                                                    |
+--------------------------------------------------------------------+
```

---

## Module Structure

### Lessons (5 total)

| # | File | Topic |
|---|------|-------|
| 1 | lessons/01_quantization_concepts.md | What quantization is, INT4/INT8/bf16/fp16, size vs quality tradeoffs |
| 2 | lessons/02_gguf_format.md | GGUF format, llama.cpp, how to run a quantized model locally |
| 3 | lessons/03_torchao.md | TorchAO library, PyTorch-native quantization and kernel optimization |
| 4 | lessons/04_onnx_export.md | ONNX format, exporting models, running across frameworks |
| 5 | lessons/05_ollama_fastapi.md | Ollama for local serving, FastAPI for building an HTTP API around it |

### Examples (5 total) -- coming soon

| # | File | What You Will Build |
|---|------|---------------------|
| 1 | examples/example_01_quantization_demo.py | Simulate INT8 quantization manually in NumPy |
| 2 | examples/example_02_gguf_runner.py | Load and run a GGUF model with llama-cpp-python |
| 3 | examples/example_03_torchao_quantize.py | Apply TorchAO quantization to a small PyTorch model |
| 4 | examples/example_04_onnx_export.py | Export a tiny transformer to ONNX and run inference |
| 5 | examples/example_05_fastapi_server.py | Full FastAPI server wrapping an Ollama model |

### Exercises (5 total) -- coming soon

| # | File | What You Will Practice |
|---|------|------------------------|
| 1 | exercises/exercise_01_quantization.py | Implement your own quantize/dequantize functions |
| 2 | exercises/exercise_02_model_formats.py | Compare model sizes in different formats |
| 3 | exercises/exercise_03_torchao_practice.py | Quantize and benchmark a PyTorch layer |
| 4 | exercises/exercise_04_onnx_inference.py | Run ONNX model and verify outputs match PyTorch |
| 5 | exercises/exercise_05_api_design.py | Design and test a full model serving API |

---

## What You Will Learn

By the end of this module, you will be able to:

1. Explain what quantization is and why it dramatically reduces model size
2. Understand the difference between fp32, fp16, bf16, INT8, and INT4
3. Describe what GGUF format is and why it was created
4. Run a local LLM using Ollama in under 5 minutes
5. Use TorchAO to quantize a PyTorch model
6. Export a model to ONNX format for cross-platform deployment
7. Build a FastAPI server that serves LLM responses over HTTP
8. Make intelligent decisions about which deployment approach to use

---

## Key Terms to Know Before Starting

Do not worry if these are unfamiliar now. Each lesson has a full glossary.

- **Quantization**: Compressing model weights from high-precision numbers to lower-precision numbers
- **fp32**: 32-bit floating point. Default PyTorch dtype. 4 bytes per number.
- **fp16 / bf16**: 16-bit floating point. 2 bytes per number. Half the memory of fp32.
- **INT8 / INT4**: 8-bit and 4-bit integers. Even smaller. Fast on modern hardware.
- **GGUF**: A file format for quantized LLMs. Used by llama.cpp and Ollama.
- **llama.cpp**: Open-source C++ library for fast LLM inference on CPU and GPU.
- **Ollama**: A tool that makes running GGUF models dead simple (one command).
- **ONNX**: Open Neural Network Exchange. Framework-agnostic model format.
- **TorchAO**: PyTorch's official quantization and optimization library.
- **FastAPI**: Modern Python web framework for building HTTP APIs.
- **Inference**: Running a trained model to get predictions (vs. training = updating weights).
- **Latency**: Time from request to response. Lower is better.
- **Throughput**: Requests per second. Higher is better.

---

## Hardware Reality Check

```
+-----------------------------------------------------------------------+
|  HARDWARE REQUIREMENTS BY MODEL SIZE AND QUANTIZATION                 |
+-----------------------------------------------------------------------+
|                                                                       |
|  Model Size  | fp32       | fp16/bf16  | INT8      | INT4             |
|  ------------|------------|------------|-----------|------------------  |
|  1B params   | 4 GB RAM   | 2 GB RAM   | 1 GB RAM  | 0.5 GB RAM       |
|  3B params   | 12 GB RAM  | 6 GB RAM   | 3 GB RAM  | 1.5 GB RAM       |
|  7B params   | 28 GB RAM  | 14 GB RAM  | 7 GB RAM  | 3.5 GB RAM       |
|  13B params  | 52 GB RAM  | 26 GB RAM  | 13 GB RAM | 6.5 GB RAM       |
|  70B params  | 280 GB RAM | 140 GB RAM | 70 GB RAM | 35 GB RAM        |
|                                                                       |
|  WHAT THIS MEANS:                                                     |
|  - fp32 70B model: needs 4 x A100 80GB GPUs (very expensive)         |
|  - INT4 7B model: runs on a MacBook Pro with 16 GB RAM (free!)        |
|                                                                       |
|  Quantization is not a trick. It is the reason local LLMs exist.     |
+-----------------------------------------------------------------------+
```

---

## C# Analogy: Why Deployment Is Hard

```csharp
// In .NET, you have experienced this problem in a different form:
//
// Development:
//   var model = new MLModel();           // Works on your dev machine
//   model.Load("my_trained_model.zip"); // 2 GB file, no problem
//
// Production:
//   - Azure App Service: 14 GB RAM limit, model won't fit
//   - Azure Functions: 1.5 GB memory limit
//   - Client machine: 8 GB total RAM, other apps running
//
// Solution in .NET ML.NET:
//   - Model compression (smaller format)
//   - ONNX export (run anywhere, even in JS with ONNX Runtime)
//   - Quantization (float32 weights -> int8 weights)
//
// LLM deployment solves the SAME problem, just at much larger scale.
// A .NET model might be 50 MB. An LLM is 28,000 MB.
// The techniques are analogous. The scale is radically different.
```

---

## Choosing Your Deployment Approach

```
+-----------------------------------------------------------------------+
|  DECISION TREE: WHICH APPROACH TO USE?                               |
+-----------------------------------------------------------------------+
|                                                                       |
|  Is your model a standard architecture (LLaMA, Mistral, etc.)?       |
|   YES --> Use Ollama. Simplest. Done in minutes.                      |
|   NO  --> Continue below.                                             |
|                                                                       |
|  Do you need to run on CPU only (no GPU available)?                   |
|   YES --> GGUF + llama.cpp or ONNX Runtime                            |
|   NO  --> Continue below.                                             |
|                                                                       |
|  Do you need to serve many users simultaneously?                      |
|   YES --> FastAPI + vLLM or TGI (Text Generation Inference)           |
|   NO  --> Continue below.                                             |
|                                                                       |
|  Do you need to deploy to mobile or edge devices?                     |
|   YES --> ONNX + INT4 quantization                                    |
|   NO  --> Continue below.                                             |
|                                                                       |
|  Are you doing research or local development?                         |
|   YES --> Ollama is still the answer. It is that easy.                |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## How to Use This Module

1. Read the lessons in order (01 through 05)
2. Lesson 01 gives the foundation (quantization) -- do NOT skip it
3. Lessons 02-05 are somewhat independent after Lesson 01
4. Install Ollama before starting Lesson 05 (installation instructions in that lesson)
5. Run the examples as you go -- these lessons are very hands-on

---

## Tools to Install (Do This Now)

```bash
# Core Python packages
pip install torch torchao onnx onnxruntime fastapi uvicorn

# For GGUF / llama.cpp
pip install llama-cpp-python

# Ollama -- install from https://ollama.com
# Then run in terminal:
ollama pull mistral        # 7B model, ~4 GB download
# or
ollama pull phi3:mini      # 3.8B model, ~2 GB download (smaller, good for testing)
```

---

## Recommended Reading (Optional)

- "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (Dettmers et al., 2022)
- "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022)
- llama.cpp GitHub: github.com/ggerganov/llama.cpp
- Ollama documentation: ollama.com/docs
- ONNX Runtime documentation: onnxruntime.ai

---

*Module 14 of the Learn LLM from Scratch course.*
*For a .NET developer learning Python and Large Language Models.*

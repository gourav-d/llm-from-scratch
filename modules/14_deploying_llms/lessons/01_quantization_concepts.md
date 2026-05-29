# Lesson 01: Quantization Concepts

## Glossary (Read This First!)

Every term used in this lesson is defined here.
Do not skip this section.

| Term | Plain English Definition |
|------|--------------------------|
| **Quantization** | The process of converting high-precision numbers (like float32) to lower-precision numbers (like int8). This makes the model smaller and faster. |
| **Precision** | How many bits are used to store a single number. More bits = more precision = more accuracy, but also more memory. |
| **fp32** | 32-bit floating point. Default in most ML frameworks. 4 bytes per number. Example: 3.14159265358979 |
| **fp16** | 16-bit floating point. Half the size of fp32. 2 bytes per number. Example: 3.14 (less precise) |
| **bf16** | Brain Float 16. A special 16-bit format designed by Google. Same range as fp32 but less precision. Very stable for training. |
| **INT8** | 8-bit integer. 1 byte per number. Integers only (-128 to 127). 4x smaller than fp32. |
| **INT4** | 4-bit integer. Half a byte per number. Very small range (-8 to 7). 8x smaller than fp32. |
| **Weight** | A single number in a neural network model. GPT-2 has 117 million weights. LLaMA-7B has 7 billion weights. |
| **Model size** | Total memory to store all weights. 7B weights x 4 bytes (fp32) = 28 GB. |
| **Inference** | Running a trained model to get output (not training, not updating weights). |
| **Post-Training Quantization (PTQ)** | Quantizing a model AFTER training is complete. No re-training needed. Fast but may lose some accuracy. |
| **Quantization-Aware Training (QAT)** | Training the model WITH quantization simulation. More accurate but needs re-training. |
| **Scale factor** | A multiplier used to map floating point values into integer range during quantization. |
| **Zero point** | An offset used in quantization to handle asymmetric distributions. |
| **Calibration** | Running a small dataset through the model to figure out the best scale factors for quantization. |
| **Dequantization** | Converting back from INT8 to fp32 for computation. Often happens on-the-fly during inference. |

---

## Part 1: Why Model Size Is the Problem

### The Math of Model Memory

Every weight in a neural network is stored as a number.
The precision of that number determines how much memory it takes.

```
+------------------------------------------------------------------+
|  MEMORY PER WEIGHT BY DATA TYPE                                  |
+------------------------------------------------------------------+
|                                                                  |
|  Data Type  | Bits | Bytes | Range                               |
|  -----------|------|-------|------------------------------------  |
|  fp64       |  64  |   8   | Huge range, very precise            |
|  fp32       |  32  |   4   | 3.4e38 range, 7 decimal digits      |
|  bf16       |  16  |   2   | Same range as fp32, less precision  |
|  fp16       |  16  |   2   | 65,504 max, 3 decimal digits        |
|  INT8       |   8  |   1   | -128 to 127 (integers only)         |
|  INT4       |   4  |  0.5  | -8 to 7 (integers only)             |
|                                                                  |
+------------------------------------------------------------------+

MODEL SIZE FORMULA:
  Total Memory = Number of Parameters x Bytes per Parameter

EXAMPLES:
  GPT-2 (117M params)      x fp32 (4 bytes) =  468 MB
  GPT-2 (117M params)      x INT8 (1 byte)  =  117 MB
  LLaMA-7B (7B params)     x fp32 (4 bytes) =   28 GB
  LLaMA-7B (7B params)     x INT4 (0.5 byte)=  3.5 GB
  LLaMA-70B (70B params)   x fp32 (4 bytes) =  280 GB
  LLaMA-70B (70B params)   x INT4 (0.5 byte)=   35 GB
```

C# analogy:
```csharp
// In C#, you know about different numeric types:
double d = 3.14159265358979;  // 64-bit, 8 bytes (like fp64)
float  f = 3.14f;             // 32-bit, 4 bytes (like fp32)
short  s = 314;               // 16-bit, 2 bytes (like INT16)
byte   b = 127;               //  8-bit, 1 byte  (like INT8)

// A List<double> with 7 billion elements = 56 GB
// A List<byte>   with 7 billion elements =  7 GB
//
// Quantization is converting that List<double> into List<byte>
// while trying to preserve the meaning of the values.
```

---

## Part 2: How Floating Point Numbers Work

To understand WHY quantization works, you need to understand what
you are trading away.

### The Structure of a Float32

Every fp32 number is stored in 32 bits, split into 3 parts:

```
+------------------------------------------------------------------+
|  fp32 NUMBER STRUCTURE (32 bits)                                 |
+------------------------------------------------------------------+
|                                                                  |
|  Sign  |  Exponent (8 bits)  |  Mantissa (23 bits)              |
|  [0]   |  [01111100]         |  [01000000000000000000000]       |
|                                                                  |
|  Sign:     1 bit  -> positive or negative                        |
|  Exponent: 8 bits -> size/scale of the number (the "zoom level") |
|  Mantissa: 23 bits -> the actual digits (the "precision")        |
|                                                                  |
|  Together they can represent: -3.4e38 to +3.4e38                 |
|  With about 7 significant decimal digits of precision.           |
|                                                                  |
+------------------------------------------------------------------+

fp16 NUMBER STRUCTURE (16 bits):
  Sign: 1 bit
  Exponent: 5 bits  (LESS range: max ~65,504)
  Mantissa: 10 bits (LESS precision: ~3 decimal digits)

bf16 NUMBER STRUCTURE (16 bits):
  Sign: 1 bit
  Exponent: 8 bits  (SAME range as fp32! max ~3.4e38)
  Mantissa: 7 bits  (LESS precision: ~2 decimal digits)
```

### Why bf16 Is Special for LLMs

bf16 was designed specifically for neural networks:

```
+------------------------------------------------------------------+
|  bf16 vs fp16 COMPARISON                                         |
+------------------------------------------------------------------+
|                                                                  |
|              | bf16        | fp16                                 |
|  ------------|-------------|-----------------------------------   |
|  Total bits  | 16          | 16                                  |
|  Exponent    | 8 bits      | 5 bits                              |
|  Range       | Same as fp32| Only up to 65,504                   |
|  Precision   | Low         | Medium                              |
|                                                                  |
|  PROBLEM WITH fp16:                                              |
|  During training, gradients can be very large or very small.     |
|  fp16's limited range (max 65,504) causes overflow:              |
|    gradient = 100,000 -> fp16 says: ERROR (out of range)         |
|                                                                  |
|  WHY bf16 IS BETTER FOR TRAINING:                                |
|  Same large range as fp32 means no overflow.                     |
|  Lower precision is fine for gradients.                          |
|                                                                  |
|  In practice:                                                    |
|    Training: use bf16 (stable, half the memory of fp32)          |
|    Inference: use fp16 or INT8 or INT4 (even smaller)            |
+------------------------------------------------------------------+
```

---

## Part 3: INT8 Quantization in Detail

INT8 is the most common deployment quantization level.
It cuts memory by 4x vs fp32 with minimal quality loss.

### The Core Idea: Mapping a Range to 256 Buckets

An INT8 number can hold values from -128 to 127.
That is only 256 possible values.

A fp32 weight might be any value, say between -2.5 and +2.5.
We need to map that continuous range to 256 discrete buckets.

```
+------------------------------------------------------------------+
|  INT8 QUANTIZATION: MAPPING CONTINUOUS TO DISCRETE               |
+------------------------------------------------------------------+
|                                                                  |
|  Original fp32 weights for one layer:                            |
|  [-2.5, -1.3, 0.0, 0.7, 1.2, 2.1, 2.5]                         |
|                                                                  |
|  Step 1: Find the range                                          |
|    min_val = -2.5                                                |
|    max_val = +2.5                                                |
|    range   = max_val - min_val = 5.0                             |
|                                                                  |
|  Step 2: Compute the scale factor                                |
|    scale = range / (2^8 - 1) = 5.0 / 255 = 0.01961              |
|                                                                  |
|  Step 3: Quantize each weight                                    |
|    quantized = round(weight / scale)                             |
|    -2.5 / 0.01961 = -127.5 -> round -> -128 (INT8)              |
|    -1.3 / 0.01961 =  -66.3 -> round ->  -66 (INT8)              |
|     0.0 / 0.01961 =    0.0 -> round ->    0 (INT8)              |
|     0.7 / 0.01961 =   35.7 -> round ->   36 (INT8)              |
|     2.5 / 0.01961 =  127.5 -> round ->  127 (INT8)              |
|                                                                  |
|  Step 4: To recover approximate original values (dequantize):    |
|    original ≈ quantized x scale                                  |
|    -128 x 0.01961 = -2.5098... (close to original -2.5)         |
|      36 x 0.01961 =  0.7060... (close to original 0.7)          |
|                                                                  |
|  Notice: Small precision loss is introduced. This is acceptable   |
|  for inference because neural networks are robust to small noise. |
+------------------------------------------------------------------+
```

### The Scale Factor and Zero Point

The formula above is called "symmetric quantization."
There is also "asymmetric quantization" which uses a zero point:

```python
# Symmetric quantization (simpler, slightly less accurate):
scale = max(abs(weights)) / 127
quantized = round(weight / scale)

# Asymmetric quantization (better for one-sided distributions like ReLU):
scale = (max_val - min_val) / 255
zero_point = round(-min_val / scale)
quantized = round(weight / scale) + zero_point

# To dequantize (recover approximate original):
original ≈ (quantized - zero_point) * scale
```

C# analogy:
```csharp
// Imagine you need to store a temperature reading.
// Actual temperature: -25.7 degrees to +42.3 degrees
// You only have a byte (0-255) to store it.
//
// Mapping: scale each degree to ~0.265 of a byte
// 0 bytes   = -25.7°  (coldest possible)
// 255 bytes = +42.3°  (hottest possible)
// 127 bytes = +7.9°   (middle value)
//
// Original: -10.5°   -> stored as byte: 57
// Recover:  57 * 0.265 - 25.7 = 15.1 - 25.7 = -10.6° (small error!)
//
// This is EXACTLY what INT8 quantization does with neural network weights.
// The "error" is small enough that the model still works well.
```

---

## Part 4: INT4 Quantization

INT4 pushes further: only 4 bits per weight, values from -8 to 7.
That is only 16 possible values for each weight.

```
+------------------------------------------------------------------+
|  INT4 vs INT8: THE TRADEOFF                                      |
+------------------------------------------------------------------+
|                                                                  |
|              | INT4            | INT8                            |
|  ------------|-----------------|------------------------------   |
|  Bits        | 4               | 8                               |
|  Values      | 16 (-8 to 7)    | 256 (-128 to 127)               |
|  Memory      | 0.5 bytes       | 1 byte                          |
|  vs fp32     | 8x smaller      | 4x smaller                      |
|  Quality     | Noticeable loss | Minimal loss                    |
|              | on some tasks   | on most tasks                   |
|                                                                  |
|  WHEN INT4 IS USED:                                              |
|  - Memory is the primary constraint (laptop, phone)             |
|  - Speed matters more than tiny quality differences              |
|  - GGUF Q4 files (most popular Ollama downloads)                 |
|                                                                  |
|  TRICK: Group quantization                                       |
|  Instead of one scale factor per layer, use one per 32 weights.  |
|  More scale factors = better approximation despite 4-bit buckets.|
|                                                                  |
+------------------------------------------------------------------+
```

### Group Quantization (Key Technique for INT4)

```
WITHOUT GROUP QUANTIZATION (bad):
  Entire 7B weight matrix uses 1 scale factor.
  Most weights are near 0. Very few are large.
  The 16 INT4 buckets don't fit the distribution well.

WITH GROUP QUANTIZATION (good, used in GGUF):
  Every 32 weights get their own scale factor.
  Each group of 32 is independently quantized.
  The 16 buckets fit each local distribution much better.
  
  7B weights / 32 = 218 million groups
  Each group has 1 fp16 scale factor (2 bytes overhead)
  218M groups x 2 bytes = 436 MB of scale factors
  7B weights x 0.5 bytes = 3.5 GB of INT4 weights
  Total: ~4 GB  (vs 28 GB fp32 -- 7x smaller!)
```

---

## Part 5: The Quality-Size Tradeoff

Quantization always involves a tradeoff between model size and output quality.

```
+------------------------------------------------------------------+
|  QUANTIZATION QUALITY TRADEOFF                                   |
+------------------------------------------------------------------+
|                                                                  |
|  Higher quality                                                  |
|       ^                                                          |
|       |  fp32  o                                                 |
|       |                                                          |
|       |         bf16 o                                           |
|       |                                                          |
|       |              fp16 o                                      |
|       |                                                          |
|       |                   INT8 o                                 |
|       |                                                          |
|       |                          INT4 o                          |
|       |                                                          |
|       |                                  INT2 o (usually bad)   |
|       +-------------------------------------------> Smaller     |
|                                                                  |
|  PRACTICAL EXPERIENCE (LLaMA-7B):                               |
|  fp32:  Perfect quality. 28 GB.                                  |
|  bf16:  Identical quality. 14 GB.                                |
|  INT8:  Barely noticeable difference. 7 GB.                      |
|  INT4:  Small but measurable difference. 3.5 GB.                 |
|  INT2:  Significant quality degradation. 1.75 GB.                |
|                                                                  |
|  RECOMMENDATION:                                                 |
|  INT8 for servers (quality matters, have RAM)                    |
|  INT4 for local use (memory is the constraint)                   |
|  bf16 for training and fine-tuning (stable, not too large)       |
+------------------------------------------------------------------+
```

### Why Neural Networks Are Robust to Quantization

This seems counterintuitive. Why does reducing from fp32 to INT4
(from millions of possible values to just 16) not completely break the model?

```
+------------------------------------------------------------------+
|  WHY NEURAL NETWORKS TOLERATE QUANTIZATION                       |
+------------------------------------------------------------------+
|                                                                  |
|  REASON 1: Redundancy                                            |
|  Neural networks have millions of parameters that together       |
|  represent a concept. A small error in one weight is averaged    |
|  out by all the others. It is like a crowd vote -- one wrong     |
|  vote does not change the outcome.                               |
|                                                                  |
|  REASON 2: Most weights are small                                |
|  In a trained LLM, most weights are clustered near 0.            |
|  The high-precision mantissa bits in fp32 mostly encode tiny     |
|  differences in small numbers. These tiny differences have       |
|  almost no effect on the final output.                           |
|                                                                  |
|  REASON 3: Outputs are probability distributions                 |
|  LLMs output probability distributions over tokens.              |
|  If "Paris" has probability 0.82 in fp32 and 0.79 in INT4,      |
|  the model still picks "Paris" both times. The decision is       |
|  the same even though the number is slightly different.          |
|                                                                  |
|  REASON 4: The model can learn to be quantization-friendly       |
|  With quantization-aware training (QAT), the model is exposed    |
|  to quantization noise during training and learns to compensate. |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 6: Post-Training Quantization (PTQ) vs QAT

There are two ways to quantize a model:

```
+------------------------------------------------------------------+
|  PTQ vs QAT COMPARISON                                           |
+------------------------------------------------------------------+
|                                                                  |
|  POST-TRAINING QUANTIZATION (PTQ)                                |
|  ------------------------------------------------------------------  |
|  When:    After training is complete                             |
|  How:     Take the trained fp32 weights, convert to INT8/INT4   |
|  Needs:   A small "calibration dataset" (a few hundred examples) |
|           to figure out the best scale factors                   |
|  Time:    Minutes to hours                                       |
|  Quality: Good for INT8, decent for INT4                         |
|  Use:     This is what llama.cpp, GGUF, and most tools do        |
|                                                                  |
|  QUANTIZATION-AWARE TRAINING (QAT)                               |
|  ------------------------------------------------------------------  |
|  When:    During training (or fine-tuning)                       |
|  How:     Simulate INT8/INT4 quantization in the forward pass    |
|           Let gradients flow through the fake quantization       |
|           Model learns to work despite quantization noise        |
|  Needs:   Full training infrastructure                           |
|  Time:    Same as training (hours to days)                       |
|  Quality: Better than PTQ, especially for INT4                   |
|  Use:     High-quality production models from Google, Apple, etc.|
|                                                                  |
|  FOR THIS COURSE: We focus on PTQ.                               |
|  PTQ is simpler, works without retraining, and produces          |
|  results good enough for almost every real use case.             |
|                                                                  |
+------------------------------------------------------------------+
```

C# analogy:
```csharp
// PTQ is like compressing an existing DLL:
//   You have YourApp.dll (large, high-precision)
//   A compression tool reads it and creates YourApp.compressed.dll
//   No source code needed. No recompilation. Fast.
//   Works well but slightly less optimal than native compilation.
//
// QAT is like recompiling with -O3 optimization from the start:
//   You modify the training code to account for quantization.
//   The model is built knowing it will be quantized.
//   Better result, but requires the full training pipeline.
```

---

## Part 7: Calibration

PTQ needs calibration to determine the best scale factors.

```
+------------------------------------------------------------------+
|  WHAT CALIBRATION DOES                                           |
+------------------------------------------------------------------+
|                                                                  |
|  PROBLEM:                                                        |
|  To quantize a layer's weights to INT8, you need to know:        |
|  "What is the range of values this layer normally sees?"         |
|                                                                  |
|  You cannot just look at the weights themselves.                 |
|  The ACTIVATIONS (intermediate computations) also matter.        |
|  Activation values depend on the input data.                     |
|                                                                  |
|  CALIBRATION PROCESS:                                            |
|  1. Take 512 representative examples from your target domain     |
|  2. Run them through the model in fp32                           |
|  3. Record the min/max values in each layer's activations        |
|  4. Use those min/max values to set scale factors                |
|                                                                  |
|  BAD CALIBRATION = wrong scale factors = bigger quantization error|
|  GOOD CALIBRATION = scale factors fit the data = small error     |
|                                                                  |
|  RULE: Calibration data should look like your actual use case.   |
|  If you deploy a legal document chatbot, calibrate on            |
|  legal documents -- not random Wikipedia articles.               |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 8: GPTQ -- The Algorithm That Made INT4 Practical

Standard INT4 quantization produces poor quality.
GPTQ (2022) fixed this with a smarter algorithm.

```
+------------------------------------------------------------------+
|  GPTQ: THE KEY IDEA                                              |
+------------------------------------------------------------------+
|                                                                  |
|  NAIVE INT4: Quantize all weights independently.                 |
|  Each weight gets rounded to the nearest INT4 value.             |
|  Error accumulates. Quality suffers.                             |
|                                                                  |
|  GPTQ:                                                           |
|  1. Quantize one weight at a time.                               |
|  2. After quantizing a weight, compute the error introduced.     |
|  3. Distribute that error to the remaining weights in the row.   |
|     (The remaining weights absorb the rounding error.)           |
|  4. This means each subsequent weight can "correct" for          |
|     the errors made by previous quantizations.                   |
|                                                                  |
|  RESULT: INT4 GPTQ is nearly as good as INT8 naive quantization. |
|                                                                  |
|  This is the algorithm used in most GGUF quantized models.       |
|  When you download "llama-7b-q4_k_m.gguf" you are getting       |
|  a model quantized with a GPTQ-style algorithm.                  |
|                                                                  |
+------------------------------------------------------------------+
```

You do not need to implement GPTQ yourself.
Tools like llama.cpp, AutoGPTQ, and TorchAO do this for you.
But you should know WHY INT4 GGUF models are as good as they are.

---

## Part 9: Quantization Naming Conventions

When you look at model repositories (HuggingFace, Ollama),
you will see names like these. Here is what they mean:

```
+------------------------------------------------------------------+
|  COMMON QUANTIZATION NAMING CONVENTIONS                          |
+------------------------------------------------------------------+
|                                                                  |
|  PyTorch / TorchAO naming:                                       |
|    int8_weight_only    = INT8 weights, fp16 activations          |
|    int4_weight_only    = INT4 weights, fp16 activations          |
|    fp8_weight_only     = fp8 weights (newer, hardware-specific)  |
|                                                                  |
|  GGUF / llama.cpp naming (Q-format):                             |
|    Q4_0    = INT4, simple quantization, smallest size            |
|    Q4_1    = INT4 with better scale factors (slightly larger)    |
|    Q4_K_M  = INT4, K-quant method, Medium size variant           |
|    Q4_K_S  = INT4, K-quant method, Small variant                 |
|    Q5_K_M  = INT5, K-quant method, Medium (between Q4 and Q8)   |
|    Q8_0    = INT8, highest quality in GGUF, 2x larger than Q4   |
|    F16     = fp16, no integer quantization, highest quality      |
|                                                                  |
|  RECOMMENDATION FOR STARTERS:                                    |
|    Q4_K_M  -> Best quality/size tradeoff for INT4 models         |
|    Q8_0    -> If you have RAM to spare and want better quality   |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 10: Mixed Precision -- The Real-World Approach

In production, models rarely use a single data type everywhere.
Instead they use different precisions for different parts.

```
+------------------------------------------------------------------+
|  MIXED PRECISION IN MODERN LLMs                                  |
+------------------------------------------------------------------+
|                                                                  |
|  TRAINING (typical):                                             |
|    Weights stored in:       fp32                                 |
|    Computations in:         bf16 (2x faster on modern GPUs)      |
|    Gradient accumulation:   fp32 (prevents underflow)            |
|    Optimizer state (Adam):  fp32 (needs precision for updates)   |
|                                                                  |
|  INFERENCE (typical for local deployment):                       |
|    Weights stored in:       INT4 or INT8                         |
|    Dequantized to:          fp16 for matrix multiplication       |
|    KV-cache stored in:      fp16 or INT8                         |
|    Final logits:            fp32                                 |
|                                                                  |
|  WHY DEQUANTIZE FOR COMPUTATION?                                 |
|    INT4 x INT4 matrix multiply is tricky hardware.               |
|    Most GPUs are optimized for fp16 matrix multiply.             |
|    So: load INT4 weights, dequantize to fp16, compute in fp16.  |
|    Memory savings from INT4 storage, speed from fp16 compute.   |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Summary

Here is what you learned in this lesson:

```
+------------------------------------------------------------------+
|  LESSON 01 SUMMARY                                               |
+------------------------------------------------------------------+
|                                                                  |
|  1. Model Size Problem                                           |
|     7B params x fp32 = 28 GB. Too large for most hardware.      |
|     Quantization solves this.                                    |
|                                                                  |
|  2. Data Types                                                   |
|     fp32: 4 bytes, default. bf16: 2 bytes, stable for training. |
|     INT8: 1 byte, 4x smaller. INT4: 0.5 bytes, 8x smaller.     |
|                                                                  |
|  3. How INT8 Quantization Works                                  |
|     Map fp32 range to 256 buckets using scale factor.           |
|     Small precision loss. Model still works because of redundancy.|
|                                                                  |
|  4. PTQ vs QAT                                                   |
|     PTQ: quantize after training (fast, easy, good enough).      |
|     QAT: quantize during training (better quality, harder).      |
|                                                                  |
|  5. GPTQ                                                         |
|     Smart error-correction algorithm that makes INT4 viable.     |
|     Used in most GGUF files you will download.                   |
|                                                                  |
|  6. Quality Tradeoff                                             |
|     bf16 ≈ fp32 quality. INT8 barely noticeable. INT4 small loss.|
|     For local use, INT4 is almost always the right choice.       |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Quiz Questions

1. A model has 7 billion parameters stored in fp32. How many gigabytes is the model file?
   Show your calculation.

2. What is the difference between fp16 and bf16? Why is bf16 preferred for training?

3. INT8 can only represent 256 different values (-128 to 127).
   Explain how a weight that was originally 1.837 in fp32 can be stored in INT8.
   What information do you need to recover the approximate original value?

4. What is the difference between PTQ and QAT? When would you choose each?

5. Why do neural networks tolerate quantization errors?
   Give at least two reasons.

6. What does GPTQ do differently from naive INT4 quantization?

7. You see a GGUF file named "mistral-7b-instruct-Q4_K_M.gguf".
   What do each of those parts of the name tell you?

8. In mixed-precision inference, weights are stored as INT4 but dequantized to fp16
   for computation. Why not just compute in INT4?

---

*Next lesson: GGUF format and how to run quantized models locally with llama.cpp.*
*File: lessons/02_gguf_format.md*

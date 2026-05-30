# Lesson 1: Chinchilla Scaling Laws

## What Problem Does This Solve?

Imagine you have a budget of $1 million to train an LLM.
You have two choices:

- **Option A**: Train a 70 billion parameter model on 300 billion tokens
- **Option B**: Train a 7 billion parameter model on 1.4 trillion tokens

Same compute cost. Very different results.

Which is better?

Before 2022, most researchers assumed bigger models were always better.
GPT-3 used 175 billion parameters and trained on 300 billion tokens.

Then in 2022, researchers at DeepMind published a paper that changed everything.

---

## The Chinchilla Paper

**Paper**: "Training Compute-Optimal Large Language Models"
**Authors**: Hoffmann et al. (DeepMind), 2022
**Model name**: Chinchilla (named after the animal, DeepMind liked animal names)

The key finding:

> **For a fixed compute budget, you should scale model size and training tokens equally.**

GPT-3 (175B params, 300B tokens) was massively undertrained.
Chinchilla (70B params, 1.4T tokens) matched or beat Gopher (280B params, 300B tokens)
using **4 times fewer parameters** and the **same compute budget**.

---

## What Are FLOPs?

Before the math, you need to understand FLOPs.

**FLOP** = Floating Point Operation = one arithmetic operation (add, multiply, etc.)

Training one step on a transformer uses approximately:

```
FLOPs per forward pass ≈ 2 × N × D
```

Where:
- **N** = number of model parameters
- **D** = number of training tokens

The "2" comes from: one multiply + one add per parameter per token = 2 operations.

Full training (forward + backward) uses approximately 6× instead of 2×:

```
Total FLOPs ≈ 6 × N × D
```

This formula is called the **Chinchilla compute formula**.

---

## The Scaling Law Formula

```
+==================================================================+
|  CHINCHILLA COMPUTE FORMULA                                      |
+==================================================================+
|                                                                  |
|  C = 6 × N × D                                                   |
|                                                                  |
|  Where:                                                          |
|    C = total compute budget (in FLOPs)                           |
|    N = number of model parameters                                |
|    D = number of training tokens                                 |
|                                                                  |
|  To train compute-optimally:                                     |
|    N_optimal ≈ (C / 12)^0.5     (optimal model size)            |
|    D_optimal ≈ (C / 0.33)^0.5   (optimal training tokens)       |
|                                                                  |
|  Simplified rule of thumb:                                       |
|    D_optimal ≈ 20 × N_optimal                                    |
|    (train each parameter on about 20 tokens)                     |
|                                                                  |
+==================================================================+
```

---

## The Rule of 20

The most memorable result from Chinchilla:

```
+------------------------------------------------------------------+
|  RULE OF 20                                                      |
|                                                                  |
|  For compute-optimal training:                                   |
|  number of training tokens = 20 × number of parameters          |
|                                                                  |
|  Examples:                                                       |
|  1B parameter model  → train on 20B tokens                      |
|  7B parameter model  → train on 140B tokens                     |
|  70B parameter model → train on 1.4T tokens                     |
|                                                                  |
|  This is a MINIMUM for good performance.                         |
|  LLaMA 3 trained on 15T tokens (70B model) = 214 tokens/param.  |
|  More data beyond 20x usually still helps.                       |
+------------------------------------------------------------------+
```

---

## Visual: The Compute Tradeoff

```
                  SAME COMPUTE BUDGET
                        |
         +--------------+----------------+
         |                               |
  Big Model                         Small Model
  Few Tokens                        Many Tokens
         |                               |
  70B params                        7B params
  300B tokens                       1.4T tokens
         |                               |
  Undertrained                    Compute-Optimal
  (GPT-3 style)                   (Chinchilla style)
         |                               |
  Each parameter sees                Each parameter sees
  only 1.7 tokens avg                 200 tokens avg
         |                               |
  Worse performance                 Better performance


LESSON: More data beats more parameters (up to the 20x rule).
```

---

## Real-World Numbers

| Model | Params | Training Tokens | Tokens/Param | Optimal? |
|-------|--------|-----------------|--------------|----------|
| GPT-3 | 175B | 300B | 1.7 | No (undertrained) |
| Gopher | 280B | 300B | 1.1 | No (undertrained) |
| Chinchilla | 70B | 1.4T | 20 | Yes |
| LLaMA 1 | 7B | 1T | 143 | More than optimal |
| LLaMA 3 | 8B | 15T | 1875 | Way beyond 20x |

**Why does LLaMA 3 train on 15T tokens for an 8B model?**

Because compute-optimal is for ONE training run.
If you are going to run the model billions of times after training,
it pays to train more so inference is faster (smaller model, same quality).

This is called the **inference-optimal** or **deployment-optimal** perspective.

---

## Kaplan vs Chinchilla

There were scaling laws BEFORE Chinchilla. You may see both cited.

| | Kaplan (OpenAI, 2020) | Chinchilla (DeepMind, 2022) |
|--|---|---|
| Key finding | Model size matters most | Model size AND tokens matter equally |
| Recommendation | Scale model, keep data constant | Scale both model and data together |
| Error | Underestimated data importance | Corrected the error |
| Influence | Led to GPT-3 | Led to LLaMA, Mistral, all modern LLMs |

Modern LLM training follows Chinchilla, not Kaplan.

---

## C# Analogy: Right-Sizing a System

```csharp
// Imagine you have a $10,000/month Azure budget.
// You have two options for a web service:
//
// Option A: 100 vCPU machine, 10,000 requests/month of training data
// Option B: 10 vCPU machine, 100,000 requests/month of training data
//
// For an ML.NET model, Option B almost always wins.
// A smaller model with MORE training examples generalizes better.
//
// Chinchilla is the mathematical proof of this intuition.
//
// In .NET terms:
//   Model parameters = number of weights = "complexity budget"
//   Training tokens = training examples × sequence length = "experience"
//
// A person with 10 years of experience at ONE company
// is often less skilled than someone with 3 years at FIVE companies.
// More diverse experience (tokens) often beats raw intelligence (parameters).
```

---

## How to Use Chinchilla Laws in Practice

```
STEP 1: Decide your compute budget
        Example: 1 × A100 GPU × 30 days
        A100 does ~312 TFLOP/s = 312 × 10^12 FLOP/s
        30 days = 30 × 24 × 3600 = 2.59 × 10^6 seconds
        Total budget C = 312 × 10^12 × 2.59 × 10^6 = 8 × 10^20 FLOPs

STEP 2: Find optimal model size
        N_optimal ≈ sqrt(C / 12) = sqrt(8×10^20 / 12) ≈ 8.2 × 10^9 ≈ 8B params

STEP 3: Find optimal token count
        D_optimal = 20 × N = 20 × 8B = 160B tokens

STEP 4: Choose a model and dataset
        - Build or use an 8B architecture
        - Find or download 160B tokens of text data
        - Train until tokens exhausted
```

---

## Quiz Questions

**Q1**: A model has 13 billion parameters. According to Chinchilla's Rule of 20,
        how many training tokens is compute-optimal?
        a) 13 billion tokens
        b) 130 billion tokens
        c) 260 billion tokens
        d) 1.3 trillion tokens

**Q2**: GPT-3 has 175B parameters and was trained on 300B tokens.
        Is this compute-optimal?
        a) Yes, it follows the Rule of 20
        b) No — it needs about 3.5 trillion tokens to be compute-optimal
        c) No — it has too many parameters
        d) Yes — more parameters are always better

**Q3**: What does C = 6 × N × D mean?
        a) C is cost in dollars, N is nodes, D is dataset size
        b) C is compute in FLOPs, N is parameters, D is training tokens
        c) C is context length, N is network depth, D is data batches
        d) C is CPU count, N is neurons, D is dropout rate

*(Answers: Q1=c, Q2=b, Q3=b)*

---

## Key Takeaways

1. Compute budget (FLOPs) = 6 × parameters × training tokens
2. Rule of 20: train each parameter on ~20 tokens minimum
3. GPT-3 and Gopher were undertrained — too many params, too few tokens
4. Chinchilla proved equal scaling of size and data beats scaling size alone
5. Modern LLMs (LLaMA 3) train far beyond 20x for deployment efficiency
6. Bigger is NOT always better — more data often beats more parameters

---

*Next: Lesson 2 — Mixed Precision Training (how to halve your memory usage)*

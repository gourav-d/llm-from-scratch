# Lesson 6: Knowledge Distillation

## The Problem: Large Models Are Expensive

A 70B parameter model:
- Needs 35–140 GB RAM to run (depending on quantization)
- Takes 100–500ms per response
- Costs $0.01–$0.10 per 1000 tokens in cloud APIs

A 7B parameter model:
- Needs 4–14 GB RAM
- Takes 10–50ms per response
- Costs 10× less to run

Can we get the **quality of the 70B model** in the **body of a 7B model**?

That is exactly what **knowledge distillation** does.

---

## What Is Knowledge Distillation?

Knowledge distillation (Hinton et al., 2015) trains a **small model (student)**
to mimic the behavior of a **large model (teacher)**.

```
+------------------------------------------------------------------+
|  TEACHER-STUDENT FRAMEWORK                                       |
+------------------------------------------------------------------+
|                                                                  |
|  TEACHER MODEL:                                                  |
|    - Large (70B parameters)                                      |
|    - Slow and expensive                                          |
|    - Already trained, weights frozen                             |
|    - Produces probability distributions over vocabulary          |
|                                                                  |
|  STUDENT MODEL:                                                  |
|    - Small (7B parameters)                                       |
|    - Fast and cheap                                              |
|    - Being trained                                               |
|    - Learns to match teacher's distributions                     |
|                                                                  |
|  TRAINING SIGNAL:                                                |
|    - Not just "predict the right answer" (hard label)           |
|    - "Produce distributions similar to teacher" (soft label)    |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Hard Labels vs Soft Labels

This is the key concept that makes distillation powerful.

**Hard labels** (standard training):
```
Input: "The sky is ___"
Correct answer: "blue"

Hard label = [0, 0, 0, 1, 0, 0, ...]
              vocab indices
              position of "blue" = 1, everything else = 0

Information content: just "the correct answer is blue"
```

**Soft labels** (teacher's probability distribution):
```
Input: "The sky is ___"
Teacher's output distribution:

  "blue"   : 0.65
  "clear"  : 0.18
  "dark"   : 0.07
  "grey"   : 0.04
  "light"  : 0.02
  ... (all 50,000 vocab tokens, most near 0)

Information content:
  - "blue" is most likely
  - BUT "clear" is almost as reasonable (19% chance)
  - "dark" is also plausible (clouds/night)
  - "grey" slightly possible
  - This captures the STRUCTURE of language relationships
```

Soft labels contain far more information than hard labels.
The student learns not just what is correct, but **how similar different answers are**.

---

## Temperature Scaling

The teacher's distribution is often very peaked (one token has 99% probability).
To make the soft labels more useful (more spread out), we use **temperature scaling**.

```
+------------------------------------------------------------------+
|  TEMPERATURE SCALING                                             |
+------------------------------------------------------------------+
|                                                                  |
|  Standard softmax:      softmax(logits)                          |
|  Temperature softmax:   softmax(logits / T)                      |
|                                                                  |
|  T = 1.0: standard distribution (may be very peaked)            |
|  T = 2.0: softer distribution (more spread out)                  |
|  T = 5.0: very soft distribution (almost uniform)               |
|                                                                  |
|  Example with T=4:                                               |
|                                                                  |
|  T=1 (peaked):  blue:0.98, clear:0.01, dark:0.005, ...          |
|  T=4 (soft):    blue:0.48, clear:0.22, dark:0.15, grey:0.08, .. |
|                                                                  |
|  Higher T → student learns more about relationships              |
|  Lower T  → student focuses more on "the right answer"          |
|                                                                  |
|  Typical value: T = 2 to 5 for distillation                     |
+------------------------------------------------------------------+
```

**Key**: You use the SAME temperature T in both teacher and student during distillation.
For inference, you set T=1 (back to standard softmax).

---

## The Distillation Loss Function

The total loss combines two objectives:

```
Loss_total = α × Loss_distillation + (1 - α) × Loss_ce

Where:
  Loss_distillation = KL(teacher_soft_probs || student_soft_probs)
                    = how different student is from teacher (at temperature T)

  Loss_ce           = CrossEntropy(student_logits, true_hard_labels)
                    = standard language modeling loss (at temperature T=1)

  α = mixing weight (typically 0.5 to 0.9)
  KL = Kullback-Leibler divergence (measures distribution difference)
```

**Intuition**:
- `Loss_distillation` says: "be like the teacher"
- `Loss_ce` says: "also be correct on the ground truth"
- `α` controls which matters more

---

## KL Divergence Explained

KL divergence measures how different two probability distributions are.

```
+------------------------------------------------------------------+
|  KL DIVERGENCE                                                   |
+------------------------------------------------------------------+
|                                                                  |
|  KL(P || Q) = sum over all tokens: P(x) × log(P(x) / Q(x))     |
|                                                                  |
|  Where:                                                          |
|    P = teacher's distribution (target)                           |
|    Q = student's distribution (being optimized)                 |
|                                                                  |
|  KL = 0:   distributions are identical (student = teacher)      |
|  KL > 0:   distributions differ (higher = more different)       |
|  KL = inf: student assigns 0 to something teacher gives P > 0  |
|                                                                  |
|  Goal: minimize KL → make student match teacher                 |
|                                                                  |
+------------------------------------------------------------------+
```

In PyTorch:
```python
import torch.nn.functional as F

# KL divergence loss (expects log-probabilities for input)
kl_loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),  # log Q
    F.softmax(teacher_logits / T, dim=-1),       # P
    reduction="batchmean"
) * (T ** 2)  # scale by T^2 to compensate for temperature division
```

The `T²` scaling keeps the gradient magnitude consistent regardless of temperature.

---

## Full Distillation Training Step

```python
import torch
import torch.nn.functional as F

def distillation_step(teacher, student, batch, T=4.0, alpha=0.7):
    """
    One training step with knowledge distillation.

    teacher: large pretrained model (frozen, in eval mode)
    student: small model being trained
    batch:   dict with 'input_ids' and 'labels'
    T:       temperature for soft labels
    alpha:   weight for distillation loss vs hard label loss
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Get teacher predictions (no gradient needed, teacher is frozen)
    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits  # shape: [B, seq_len, vocab]

    # Get student predictions (gradient tracked here)
    student_logits = student(input_ids).logits      # shape: [B, seq_len, vocab]

    # --- Loss 1: Distillation (soft labels from teacher) ---
    # Compute soft distributions at temperature T
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)    # P
    student_log_soft = F.log_softmax(student_logits / T, dim=-1)  # log Q

    # KL divergence: how far student is from teacher
    loss_distill = F.kl_div(
        student_log_soft,
        teacher_soft,
        reduction="batchmean"
    ) * (T ** 2)  # T^2 scaling is critical!

    # --- Loss 2: Standard cross-entropy (hard labels) ---
    # Reshape for cross_entropy: [B*seq_len, vocab] vs [B*seq_len]
    B, S, V = student_logits.shape
    loss_ce = F.cross_entropy(
        student_logits.view(B * S, V),
        labels.view(B * S)
    )

    # --- Combined loss ---
    loss = alpha * loss_distill + (1 - alpha) * loss_ce

    return loss, loss_distill.item(), loss_ce.item()
```

---

## Types of Distillation

There are several variations of distillation:

```
+------------------------------------------------------------------+
|  DISTILLATION TYPES                                              |
+------------------------------------------------------------------+
|                                                                  |
|  1. OUTPUT DISTILLATION (classic)                                |
|     Student mimics teacher's final token probabilities.          |
|     Simple. Works well for most cases.                           |
|     This is what we implemented above.                           |
|                                                                  |
|  2. FEATURE DISTILLATION                                         |
|     Student mimics teacher's intermediate hidden states.         |
|     More powerful but requires same architecture dimensions.    |
|     Example: TinyBERT matches BERT's attention patterns.        |
|                                                                  |
|  3. SELF-DISTILLATION                                            |
|     Teacher and student are the SAME model.                      |
|     Earlier checkpoint teaches later checkpoint.                |
|     Useful for regularization.                                   |
|                                                                  |
|  4. SEQUENCE-LEVEL DISTILLATION                                  |
|     Teacher generates data; student trains on that data.        |
|     Teacher doesn't need to be available during training.       |
|     Example: generate 1M responses with GPT-4, fine-tune GPT-2. |
|     This is how many open-source instruct models were created.  |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Real-World Examples

| Student | Teacher | Result |
|---------|---------|--------|
| DistilBERT (66M) | BERT (110M) | 40% smaller, 60% faster, 97% accuracy |
| DistilGPT-2 (82M) | GPT-2 (117M) | 33% smaller, similar quality |
| TinyLlama (1.1B) | LLaMA 2 (7B) | 6× smaller, surprisingly capable |
| Phi-2 (2.7B) | Multiple teachers | Outperforms many 7B models |
| Qwen2.5-0.5B | Qwen2.5-72B | Runs on phones, reasonable quality |

Microsoft's Phi series is famous for using distillation + curated data
to create small models that punch far above their weight.

---

## Sequence-Level Distillation: Building Instruct Models

This technique deserves special attention — it is how most cheap instruct models are made:

```
+------------------------------------------------------------------+
|  SEQUENCE-LEVEL DISTILLATION (data-free teacher)                |
+------------------------------------------------------------------+
|                                                                  |
|  Step 1: Run expensive teacher (GPT-4) on many prompts           |
|    prompt: "Explain photosynthesis to a 10-year-old"             |
|    teacher response: "Plants make food from sunlight..."         |
|    Store (prompt, response) pair                                 |
|                                                                  |
|  Step 2: Generate thousands/millions of such pairs               |
|    This is called "synthetic data generation"                    |
|                                                                  |
|  Step 3: Fine-tune a small model on (prompt, response) pairs     |
|    The small model learns to produce GPT-4 style responses        |
|    This is standard SFT (Module 12), not "real" distillation    |
|                                                                  |
|  Cost: one-time teacher inference cost                           |
|  Benefit: cheap, fast model with GPT-4 style behavior           |
|                                                                  |
|  Example: Alpaca, Vicuna, OpenHermes — all use this approach    |
+------------------------------------------------------------------+
```

---

## C# Analogy: Mentorship and Code Review

```csharp
// Knowledge distillation is like a junior developer (student)
// learning from a senior developer (teacher).
//
// HARD LABEL approach (textbook learning):
// "The correct pattern for a repository is:
//  public interface IRepository<T> { ... }"
// Junior learns the answer: "use interfaces"
// But doesn't understand WHY or WHEN.
//
// SOFT LABEL approach (mentorship):
// Senior developer reviews junior's PR.
// Senior says: "This code works (correct answer),
//   but I would consider dependency injection (0.6 confidence),
//   or a factory pattern (0.3 confidence),
//   or even your approach (0.1 confidence)."
//
// Junior now learns the SPACE of solutions, not just one answer.
// They understand that multiple approaches are valid with different tradeoffs.
//
// Temperature = how much the senior "softens" their opinion.
// T=1: "Use DI. Period." (overconfident)
// T=4: "DI is best, factory is good, yours works too" (nuanced)
//
// This is exactly what temperature does to soft labels.
```

---

## When to Use Distillation

```
+------------------------------------------------------------------+
|  USE DISTILLATION WHEN:                                          |
+------------------------------------------------------------------+
|                                                                  |
|  You have a large, accurate model but need a fast one            |
|  → Distill to 4x-10x smaller model, keep most quality           |
|                                                                  |
|  You need to run on mobile/edge devices                          |
|  → Distill down to 1B or even 100M parameters                   |
|                                                                  |
|  You want a specialized model for one domain                     |
|  → Teacher: general 70B model                                    |
|  → Student: 7B model that's an expert at your specific task      |
|                                                                  |
|  You want to compress API costs                                   |
|  → Fine-tune GPT-4 responses into a small self-hosted model      |
|                                                                  |
|  DO NOT USE DISTILLATION WHEN:                                   |
|  - Teacher and student are similar size (little to gain)         |
|  - You need the absolute best performance                        |
|  - The task is novel (teacher may not be reliable)               |
+------------------------------------------------------------------+
```

---

## Quiz Questions

**Q1**: What is the main advantage of soft labels over hard labels in distillation?
        a) Soft labels make training faster
        b) Soft labels encode the teacher's knowledge about similar answers
        c) Soft labels reduce the vocabulary size
        d) Soft labels are easier to compute

**Q2**: Temperature T=4 in distillation:
        a) Makes the distribution more peaked (focused on one answer)
        b) Slows down the student's learning
        c) Makes the distribution softer (more spread across multiple tokens)
        d) Quadruples the student model's size

**Q3**: KL divergence in distillation loss measures:
        a) How much the student's parameters differ from the teacher's
        b) How different the student's probability distribution is from the teacher's
        c) The size difference between teacher and student models
        d) How fast the student is compared to the teacher

*(Answers: Q1=b, Q2=c, Q3=b)*

---

## Key Takeaways

1. Distillation trains a small student to mimic a large teacher
2. Soft labels (teacher's probability distribution) contain more signal than hard labels
3. Temperature T softens the distribution → higher T = more information transferred
4. Loss = α × KL(teacher || student) + (1-α) × CrossEntropy with true labels
5. T² scaling keeps gradient magnitude consistent across temperatures
6. DistilBERT: 40% smaller, 60% faster, 97% BERT quality — the canonical example
7. Sequence-level distillation: use teacher to generate data, fine-tune student on it
8. Most cheap instruct models (Alpaca, Vicuna) use sequence-level distillation from GPT-4

---

## Module 15 Complete!

You have now learned the six core techniques for advanced LLM training:

| Technique | What It Solves |
|-----------|---------------|
| Chinchilla Laws | How much to train — optimal model size vs token count |
| Mixed Precision | Memory and speed — bf16/fp16 halves VRAM usage |
| Flash Attention | Long contexts — O(N) memory instead of O(N²) |
| Gradient Checkpointing | Activation memory — recompute instead of store |
| ZeRO Optimizer | Multi-GPU efficiency — eliminate redundant state |
| Dataset Streaming | Massive datasets — train without downloading |
| Knowledge Distillation | Model compression — 70B quality in a 7B model |

These techniques are used in every serious LLM training run.
LLaMA, Mistral, Qwen, Phi — all of them use all of these.

---

*Module 15 complete. Next: Module 16 — Modern LLM Architectures (Diffusion LMs, MTP)*

# Lesson 01: What Is Fine-Tuning?

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain what fine-tuning is in plain English
2. Describe what changes inside a model during fine-tuning
3. Compare fine-tuning vs prompting vs training from scratch
4. List three real-world use cases for fine-tuning
5. Explain why fine-tuning works (transfer learning)

---

## GLOSSARY

```
Pre-trained Model:
  A model already trained on a massive dataset (billions of web pages, books, code).
  Examples: GPT-4, LLaMA, BERT, DistilBERT.
  These cost millions of dollars and months to train.
  You do NOT retrain these -- you fine-tune them.

Fine-Tuning:
  Continuing to train a pre-trained model on a SMALLER, TASK-SPECIFIC dataset.
  The model's weights are updated slightly to improve performance on your task.
  Like taking an expert and sending them to a specialist course.

Weights (Parameters):
  The numbers inside the model that store its "knowledge."
  Pre-trained model: weights already set to encode language understanding.
  Fine-tuning: adjusts these weights slightly for your task.

Transfer Learning:
  Reusing knowledge from one task to help with another.
  A model trained on all human text already understands language structure,
  grammar, facts, and reasoning. You transfer this to your specific task.
  In C#: like inheriting from a base class and overriding only what you need.

Base Model:
  The original pre-trained model before fine-tuning.
  LLaMA-3-8B, Mistral-7B, GPT-2 are common base models for fine-tuning.

Fine-Tuned Model:
  The result after fine-tuning. Has new task-specific behavior.
  Still uses the base model's language understanding.

Epoch:
  One full pass through the training dataset.
  Fine-tuning typically needs only 1-5 epochs (vs thousands for training from scratch).

Learning Rate:
  How much to change weights per training step.
  Fine-tuning uses a VERY SMALL learning rate (e.g., 1e-4 or 2e-5).
  Large learning rate = overwrites original knowledge (bad).

Catastrophic Forgetting:
  When fine-tuning destroys the model's original knowledge.
  Happens when: learning rate too high, too many epochs, bad data.
  Fix: small learning rate, few epochs, LoRA (Lesson 03).
```

---

## Part 1: What Happens Inside the Model During Fine-Tuning

From Module 05 you learned that a model is just a big function: input text -> output text.
Internally it is billions of numbers (weights) arranged in layers.

```
PRE-TRAINED MODEL (before fine-tuning):
  Layer 1 weights: [0.23, -0.41, 0.88, ...]  <- learned from internet text
  Layer 2 weights: [0.71,  0.12, -0.33, ...]
  ...
  (7 billion numbers for LLaMA-7B)

These weights encode: grammar, facts, reasoning, code patterns, etc.
The model "knows" things because these numbers encode patterns from training data.

FINE-TUNING STEP 1: Show the model your data
  Example: {"input": "Classify ticket: App crashes on login", "output": "BUG"}

FINE-TUNING STEP 2: Model makes a prediction
  Model says: "I think the answer is: 'This appears to be a technical issue...'"
  (Wrong! We wanted just "BUG")

FINE-TUNING STEP 3: Calculate the error (loss)
  Expected: "BUG"
  Got:      "This appears to be a technical issue..."
  Loss:     Very high (far from correct)

FINE-TUNING STEP 4: Backpropagation adjusts weights
  Layer 1 weights: [0.23, -0.41, 0.88, ...] -> [0.231, -0.409, 0.882, ...]
  (Tiny changes -- using a very small learning rate)

REPEAT thousands of times with your dataset:
  The model gradually learns to output "BUG" for bug reports,
  "FEATURE REQUEST" for feature requests, etc.
```

The key insight: **you do not start from zero**. The model already understands language.
You are just nudging it in the right direction.

---

## Part 2: Fine-Tuning vs Other Approaches

### Approach 1: Prompting (Zero Examples)
```
Prompt: "Classify this support ticket: 'App crashes on login'"
LLM:    "This could be a bug, or possibly a configuration issue.
         It may relate to authentication. I would suggest checking..."

Problem: Too verbose, inconsistent format, sometimes wrong.
```

### Approach 2: Few-Shot Prompting (Examples in Prompt)
```
Prompt: "Classify tickets. Examples:
  'Server is down' -> OUTAGE
  'Add dark mode' -> FEATURE
  'App crashes on login' -> ?"
LLM:    "BUG"

Better! But: uses precious context window space.
Every call includes these examples = expensive + slower.
```

### Approach 3: Fine-Tuning (Your Data Baked In)
```
After fine-tuning:
  Input:  "App crashes on login"
  Output: "BUG"

No examples in the prompt. Model learned the pattern from training data.
Result: consistent, fast, no wasted context window.
```

### Approach 4: Training from Scratch
```
Start with random weights -> train on all internet text -> train on your data
Cost: millions of dollars, months of compute, team of 50+ engineers.
Only OpenAI, Google, Meta, Anthropic do this.
You will NEVER need to do this.
```

### Decision Table

```
WHEN TO USE EACH APPROACH:

Situation                               Best Approach
----------------------------------------------
Just testing an idea quickly            Prompting
Need consistent output format           Fine-tuning
Data changes frequently                 RAG + prompting (Module 10/11)
Need domain expertise baked in          Fine-tuning
Building a product for many users       Fine-tuning (cheaper per call)
Limited labeled training data (<100)    Prompting or few-shot
Large labeled dataset (>1000)           Fine-tuning
Need the model to reason differently    Fine-tuning + RLHF/DPO
```

---

## Part 3: Transfer Learning -- Why Fine-Tuning Works

Fine-tuning works because of TRANSFER LEARNING.

The pre-trained model has already learned:
  - How language works (grammar, syntax, semantics)
  - Facts about the world (history, science, code)
  - How to reason step by step
  - Code patterns across many languages

All of this transfers to your task FOR FREE.

```
ANALOGY (C# developer learning Python):
  You already know: OOP, loops, functions, debugging, testing, APIs.
  Learning Python = transfer that knowledge, learn new syntax.
  You do NOT need to re-learn what a loop is.

  Similarly:
  Pre-trained LLM already knows: language, reasoning, facts.
  Fine-tuning = apply that to YOUR task.
  It does NOT need to re-learn what language is.
```

### What Actually Changes in the Weights

```
BEFORE fine-tuning (pre-trained on internet):
  Model knows: "A support ticket with 'crash' likely relates to software bugs."
  But outputs: Long, verbose, conversational explanations.

AFTER fine-tuning on your labeled data:
  Model learns: "For this task, output just the category label."
  It learns the FORMAT and STYLE you want.
  It does NOT learn new facts -- it already knew the facts.
  It learned how to PRESENT the answer.
```

Most fine-tuning is about teaching the model:
1. The output FORMAT you expect
2. The STYLE appropriate for your use case
3. Domain-specific TERMINOLOGY and classifications
4. How to REFUSE or handle edge cases

---

## Part 4: Real-World Fine-Tuning Examples

### Example 1: Sentiment Analysis (Simple)
```
Dataset:
  {"text": "Great product! Love it!",      "label": "POSITIVE"}
  {"text": "Broken after one day.",         "label": "NEGATIVE"}
  {"text": "It's okay, nothing special.",   "label": "NEUTRAL"}
  (x 10,000 examples)

Fine-tuning result:
  Input:  "This app is absolutely terrible."
  Output: "NEGATIVE"
  (Instant, no prompt engineering needed, consistent format)
```

### Example 2: Instruction Following (Medium)
```
Dataset (instruction-response pairs):
  {"instruction": "Summarize in one sentence", "input": "Long article...", "output": "Short summary."}
  {"instruction": "Translate to French",       "input": "Hello world",     "output": "Bonjour le monde"}
  (x 50,000 examples -- this is how ChatGPT was made from GPT-3)

Result: Model follows natural language instructions.
```

### Example 3: Domain Expert (Complex)
```
Dataset (medical, legal, financial Q&A):
  {"question": "What is the half-life of ibuprofen?", "answer": "Approximately 2 hours."}
  {"question": "What is a contra account?", "answer": "An account that reduces..."}
  (x 100,000 domain-specific examples)

Result: Model that answers domain questions accurately with correct terminology.
```

### C# Analogy

```csharp
// Pre-trained model = a fully-implemented base class
public abstract class LanguageModel {
    // Already implemented: understand text, reason, generate language
    public virtual string Generate(string input) {
        return GeneralPurposeResponse(input);
    }
}

// Fine-tuning = override specific behaviors
public class SupportTicketClassifier : LanguageModel {
    // Fine-tuned to override the general response with a specific one
    public override string Generate(string input) {
        // Weights have been adjusted to return: "BUG", "FEATURE", "OUTAGE"
        return ClassifyTicket(input);
    }
}
```

---

## Part 5: What Can Go Wrong

```
PROBLEM 1: Catastrophic Forgetting
  Model "forgets" its original knowledge during fine-tuning.
  Cause: learning rate too high, too many epochs.
  Fix: small learning rate (1e-4 to 2e-5), few epochs (1-3), LoRA (Lesson 03).

PROBLEM 2: Overfitting
  Model memorizes your training data exactly.
  Train loss: very low (looks good).
  Validation loss: rising (model fails on new data).
  Fix: more training data, early stopping, regularization.

PROBLEM 3: Bad Training Data
  "Garbage in, garbage out."
  Inconsistent labels, wrong answers, biased data.
  Result: model learns the wrong thing confidently.
  Fix: clean and validate your dataset (Lesson 02).

PROBLEM 4: Wrong Format
  Training data in one format, inference data in different format.
  Example: trained on "Question: {q} Answer: {a}" but inference sends just "{q}".
  Result: model confused, poor performance.
  Fix: use consistent prompt template (Lesson 02).
```

---

## Key Takeaways

1. Fine-tuning = continue training a pre-trained model on your task-specific data.

2. You only update the weights SLIGHTLY -- the base knowledge is preserved.

3. Fine-tuning is about teaching FORMAT and STYLE, not new facts.

4. Transfer learning: the model already knows language; you just redirect it.

5. Three pitfalls: catastrophic forgetting, overfitting, bad data.

6. Most real-world NLP products use fine-tuning, not training from scratch.

---

## Next

Lesson 02: Dataset Preparation
  - What format should training data be in?
  - How many examples do you need?
  - What is the instruction-response format?
  - How to split data into train/validation/test sets?

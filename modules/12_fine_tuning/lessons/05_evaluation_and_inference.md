# Lesson 05: Evaluation and Inference

## Learning Objectives

By the end of this lesson, you will be able to:
1. Choose the right evaluation metric for your fine-tuning task
2. Calculate accuracy, F1, and perplexity
3. Run inference on a fine-tuned model
4. Compare base model vs fine-tuned model side by side
5. Identify when to stop fine-tuning or collect more data

---

## GLOSSARY

```
Inference:
  Running the model to get predictions on NEW data (not training data).
  You load the fine-tuned model, give it an input, and get the output.
  This is what the end user experiences.

Evaluation Metrics:
  Numbers that tell you how good the fine-tuned model is.
  Different tasks need different metrics.
  Never use just one metric -- use 2-3 together.

Accuracy:
  Fraction of predictions that are exactly correct.
  Good for: balanced classification (equal samples per class).
  Formula: (correct predictions) / (total predictions)
  Range: 0.0 to 1.0 (higher is better)

Precision:
  Of everything the model labeled as "BUG", what fraction was actually a BUG?
  Formula: true_positives / (true_positives + false_positives)
  Measures: how much you can trust a positive prediction.

Recall:
  Of all actual BUGs in the data, what fraction did the model find?
  Formula: true_positives / (true_positives + false_negatives)
  Measures: how many true cases did the model catch.

F1 Score:
  The harmonic mean of precision and recall.
  Formula: 2 * (precision * recall) / (precision + recall)
  Use when: classes are imbalanced (more BUGs than FEATURES).
  Range: 0.0 to 1.0 (higher is better)

Perplexity:
  Measures how "surprised" the model is by test data.
  Lower perplexity = model finds the data more predictable = better.
  Formula: exp(average cross-entropy loss)
  Use for: generative tasks (summarization, translation).

BLEU Score:
  Measures overlap between model output and reference text.
  Used for: translation, summarization.
  Range: 0 to 100 (higher is better). Human translations score ~50-60.

Baseline:
  A simple reference point to compare against.
  Examples: "always predict the most common class", "original base model".
  If your fine-tuned model does not beat the baseline, something is wrong.

A/B Comparison:
  Running the same prompts through BOTH base and fine-tuned model.
  Side-by-side comparison shows if fine-tuning actually helped.
```

---

## Part 1: Choosing the Right Metric

```
TASK                        METRIC(S)
-------------------------------------------------------------------
Binary classification        Accuracy, F1 (if imbalanced)
Multi-class classification   Accuracy, F1 (macro average)
Text generation              Perplexity, BLEU (if reference exists)
Summarization                ROUGE score, human eval
Code generation              Pass@k (does the code run?)
Question answering           Exact match, F1 over tokens
Chatbot / instruction        Human preference (hard to automate)

GENERAL RULE:
  Simple classification: accuracy + F1
  Generation: perplexity + human eval
  Production: define business metric (click rate, task completion, etc.)
```

---

## Part 2: Computing Evaluation Metrics

```python
def compute_accuracy(predictions: list, labels: list) -> float:
    """
    Simple accuracy: what fraction of predictions are correct?
    predictions: list of predicted labels (strings or ints)
    labels:      list of true labels
    Returns: float 0.0 to 1.0
    """
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(labels)

def compute_f1(predictions: list, labels: list, target_class) -> dict:
    """
    Compute precision, recall, and F1 for a specific class.
    predictions:   predicted labels
    labels:        true labels
    target_class:  the class we care about (e.g., "BUG")
    Returns: {"precision": ..., "recall": ..., "f1": ...}
    """
    # True Positive: predicted target_class AND it was correct
    tp = sum(1 for p, l in zip(predictions, labels)
             if p == target_class and l == target_class)

    # False Positive: predicted target_class but was wrong
    fp = sum(1 for p, l in zip(predictions, labels)
             if p == target_class and l != target_class)

    # False Negative: should have predicted target_class but did not
    fn = sum(1 for p, l in zip(predictions, labels)
             if p != target_class and l == target_class)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

def compute_perplexity(loss: float) -> float:
    """
    Convert cross-entropy loss to perplexity.
    loss:    average cross-entropy loss on test set
    Returns: perplexity (lower is better)
    """
    import math
    return math.exp(loss)

# Example usage:
preds  = ["BUG", "FEATURE", "BUG", "HOW_TO", "OUTAGE"]
actual = ["BUG", "FEATURE", "OUTAGE", "HOW_TO", "OUTAGE"]

acc = compute_accuracy(preds, actual)
f1  = compute_f1(preds, actual, target_class="BUG")

print(f"Accuracy: {acc:.2%}")         # 4/5 correct = 80%
print(f"F1 for BUG: {f1}")
```

---

## Part 3: The Confusion Matrix

For classification tasks, always print a confusion matrix:

```python
def confusion_matrix(predictions: list, labels: list) -> dict:
    """
    Count correct and incorrect predictions per class.
    Returns nested dict: matrix[true_label][predicted_label] = count
    """
    classes = sorted(set(labels) | set(predictions))  # All unique classes
    matrix = {c: {c2: 0 for c2 in classes} for c in classes}  # Initialize all to 0

    for pred, true in zip(predictions, labels):
        matrix[true][pred] += 1    # Increment: true label row, predicted label column

    return matrix

def print_confusion_matrix(matrix: dict):
    """Print confusion matrix in a readable format."""
    classes = sorted(matrix.keys())
    col_width = max(len(c) for c in classes) + 2

    # Header
    print("TRUE\\PRED " + "".join(f"{c:>{col_width}}" for c in classes))
    print("-" * (col_width * (len(classes) + 1)))

    # Rows
    for true in classes:
        row = f"{true:<10}" + "".join(
            f"{matrix[true][pred]:>{col_width}}" for pred in classes
        )
        print(row)

# Example:
preds  = ["BUG", "BUG", "FEATURE", "FEATURE", "OUTAGE",  "HOW_TO"]
actual = ["BUG", "OUTAGE", "FEATURE", "BUG",    "OUTAGE", "HOW_TO"]

cm = confusion_matrix(preds, actual)
print_confusion_matrix(cm)
```

Output:
```
TRUE\PRED    BUG  FEATURE  HOW_TO  OUTAGE
-------------------------------------------
BUG            1        0       0       1   <- 1 BUG correctly classified, 1 misclassified as OUTAGE
FEATURE        1        1       0       0   <- 1 correct, 1 wrong
HOW_TO         0        0       1       0   <- Perfect!
OUTAGE         0        0       0       1   <- 1 correct
```

Read the matrix: rows = true labels, columns = predicted labels.
**Diagonal = correct predictions.** Off-diagonal = mistakes.

---

## Part 4: Running Inference on a Fine-Tuned Model

```python
def run_inference(model, tokenizer, prompt_template: str, examples: list) -> list:
    """
    Run the fine-tuned model on new examples.
    model:           the fine-tuned model
    tokenizer:       converts text to tokens
    prompt_template: SAME template used during training
    examples:        list of {"instruction": ..., "input": ...} dicts
    Returns:         list of predicted output strings
    """
    predictions = []

    for example in examples:
        # Build the prompt EXACTLY as during training (same template!)
        prompt = prompt_template.format(
            instruction=example["instruction"],
            input=example["input"],
            output=""    # Leave output empty -- model will fill this in
        )

        # Convert text to tokens
        tokens = tokenizer.encode(prompt, return_tensors="pt")

        # Generate (model predicts next tokens until it stops)
        with torch.no_grad():              # No gradients during inference
            output_tokens = model.generate(
                tokens,
                max_new_tokens=50,         # Max 50 new tokens for the response
                temperature=0.1,           # Low temperature = more deterministic
                do_sample=True,            # Use sampling (vs greedy)
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the response tokens back to text
        output_text = tokenizer.decode(
            output_tokens[0][len(tokens[0]):],  # Only the new tokens (not the prompt)
            skip_special_tokens=True
        )

        predictions.append(output_text.strip())

    return predictions
```

---

## Part 5: A/B Comparison -- Base vs Fine-Tuned

This is the most important evaluation: does fine-tuning actually help?

```
SETUP:
  base_model:       original pre-trained model (no fine-tuning)
  finetuned_model:  our fine-tuned version

COMPARE ON 20-50 test examples:

TEST 1: Support ticket classification
---------------------------------------------------------------------------
Input:    "Users getting 401 errors when calling the payment API"

Base model output:
  "A 401 error indicates an authentication issue. This could be caused
   by expired API keys, incorrect credentials, or..."
  (Long, verbose, not in the format we wanted)

Fine-tuned model output:
  "BUG"
  (Exact format, correct classification, instant)

WINNER: Fine-tuned model

---------------------------------------------------------------------------
TEST 2: Knowledge question
---------------------------------------------------------------------------
Input:    "What are the main causes of the French Revolution?"

Base model output:
  "The French Revolution (1789-1799) was caused by: social inequality,
   financial crisis, Enlightenment ideas, weak leadership of Louis XVI..."
  (Good detailed answer using general knowledge)

Fine-tuned model output:
  (Depends on fine-tuning -- if it overfit on support tickets,
   it might give a strange or wrong answer here)

WARNING: Fine-tuned model may be WORSE at general tasks.
This is the trade-off. Fine-tuning for one task can hurt performance on others.
```

### Measuring the Tradeoff

```python
def compare_models(base_model, finetuned_model, test_data: list) -> dict:
    """
    Compare base vs fine-tuned model on the test set.
    Returns accuracy and example comparisons.
    """
    base_preds   = []
    tuned_preds  = []
    true_labels  = []

    for example in test_data:
        base_pred  = base_model.predict(example["input"])
        tuned_pred = finetuned_model.predict(example["input"])
        true_label = example["output"]

        base_preds.append(base_pred)
        tuned_preds.append(tuned_pred)
        true_labels.append(true_label)

    base_acc  = compute_accuracy(base_preds, true_labels)
    tuned_acc = compute_accuracy(tuned_preds, true_labels)

    return {
        "base_accuracy":    base_acc,
        "tuned_accuracy":   tuned_acc,
        "improvement":      tuned_acc - base_acc,
        "examples": [
            {
                "input": ex["input"],
                "true":  true_labels[i],
                "base":  base_preds[i],
                "tuned": tuned_preds[i],
            }
            for i, ex in enumerate(test_data[:5])  # Show first 5 examples
        ]
    }
```

---

## Part 6: When to Stop Fine-Tuning and What to Do Next

```
SCENARIO 1: Fine-tuned model is much better than base
  val_accuracy: base=42%, tuned=91%
  Action: SUCCESS! Deploy the fine-tuned model.

SCENARIO 2: Fine-tuned model is slightly better
  val_accuracy: base=65%, tuned=72%
  Action: Consider more training data. Is 72% good enough for your use case?

SCENARIO 3: No improvement
  val_accuracy: base=65%, tuned=66%
  Possible causes:
    - Dataset too small (need more examples)
    - Wrong prompt template (mismatch train vs eval)
    - Data quality problems (inconsistent labels)
    - Learning rate wrong (too high = forgot everything, too low = did not learn)
  Action: Debug systematically. Check each cause.

SCENARIO 4: Fine-tuned model is WORSE
  val_accuracy: base=65%, tuned=50%
  Cause: Catastrophic forgetting (LR too high or too many epochs)
  Action: Reduce learning_rate by 10x, reduce epochs to 1-2, try LoRA.

DIAGNOSTIC FLOWCHART:
  No improvement? -> Check dataset format matches training template
  Model worse?    -> Learning rate too high, use LoRA
  Overfitting?    -> More data, fewer epochs, early stopping
  Still bad?      -> Try a bigger base model, or collect more/better data
```

---

## Key Takeaways

1. Choose metrics that match your task: accuracy+F1 for classification, perplexity for generation.

2. Always print a confusion matrix for classification -- shows where the model confuses classes.

3. Inference: use the EXACT SAME prompt template as training. Slight difference = broken model.

4. A/B comparison is the most honest evaluation: compare base vs fine-tuned on the same inputs.

5. Fine-tuning specializes the model -- it may get worse at general tasks. That is the tradeoff.

6. If no improvement: check data format, quality, and learning rate before giving up.

---

## Module 12 Summary

You now know all the core concepts:

```
Lesson 01: What is fine-tuning (transfer learning, base model, adapted model)
Lesson 02: Dataset preparation (formats, splits, quality checklist)
Lesson 03: LoRA / PEFT (efficient fine-tuning, low-rank matrices)
Lesson 04: Training loop (epochs, batches, early stopping, LR schedule)
Lesson 05: Evaluation (metrics, confusion matrix, A/B comparison)
```

Now work through the examples and exercises!

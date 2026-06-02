# Module 17 — Python Guide

## Python Features Used in This Module

---

## 1. math.exp() and math.log()

Used to convert between loss and perplexity.

```python
import math

loss = 3.5
perplexity = math.exp(loss)       # exp(3.5) = 33.1
print(f"Perplexity: {perplexity:.1f}")

# Reverse: loss from perplexity
ppl = 33.1
loss_back = math.log(ppl)         # ln(33.1) = 3.5
```

**C# equivalent:**
```csharp
double perplexity = Math.Exp(loss);
double lossBack = Math.Log(perplexity);  // natural log
```

Note: `math.log()` in Python = natural log (ln). `math.log(x, 10)` = log base 10.

---

## 2. collections.Counter — count occurrences

Used in BLEU/ROUGE to count n-gram frequencies.

```python
from collections import Counter

tokens = ["the", "cat", "sat", "on", "the", "mat"]
counts = Counter(tokens)
# Counter({'the': 2, 'cat': 1, 'sat': 1, 'on': 1, 'mat': 1})

print(counts["the"])    # 2
print(counts["dog"])    # 0 (no KeyError — returns 0 for missing keys)

# Count bigrams
bigrams = list(zip(tokens, tokens[1:]))
# [('the', 'cat'), ('cat', 'sat'), ('sat', 'on'), ('on', 'the'), ('the', 'mat')]
bigram_counts = Counter(bigrams)
```

**C# equivalent:**
```csharp
var counts = tokens.GroupBy(t => t).ToDictionary(g => g.Key, g => g.Count());
```

---

## 3. zip() for n-gram generation

Used to generate bigrams and trigrams from a word list.

```python
words = ["the", "cat", "sat", "on", "the", "mat"]

# Bigrams: zip current list with itself shifted by 1
bigrams = list(zip(words, words[1:]))
# [('the', 'cat'), ('cat', 'sat'), ...]

# Trigrams: zip with shifts of 1 and 2
trigrams = list(zip(words, words[1:], words[2:]))
# [('the', 'cat', 'sat'), ('cat', 'sat', 'on'), ...]

# General n-gram function
def get_ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))

print(get_ngrams(words, 2))  # bigrams
print(get_ngrams(words, 3))  # trigrams
```

**C# equivalent:**
```csharp
var bigrams = words.Zip(words.Skip(1), (a, b) => (a, b)).ToList();
```

---

## 4. min() / max() with default

Used in BLEU clipping (cap matched count to reference count).

```python
reference_count = {"the": 2, "cat": 1}
generated_count = {"the": 5, "cat": 1, "dog": 2}

# BLEU clips: matched count = min(generated, reference)
clipped = {
    word: min(generated_count[word], reference_count.get(word, 0))
    for word in generated_count
}
# {"the": 2 (clipped from 5), "cat": 1, "dog": 0}

total_clipped = sum(clipped.values())   # 3
total_generated = sum(generated_count.values())  # 8
precision = total_clipped / total_generated  # 0.375
```

`.get(word, 0)` returns `0` instead of raising `KeyError` if key is missing.

---

## 5. json module — reading evaluation results

`lm-eval` outputs results as JSON. You'll read and compare them.

```python
import json

# Load results file
with open("results/mistral_eval.json", "r") as f:
    results = json.load(f)

# Navigate nested dict
mmlu_score = results["results"]["mmlu"]["acc,none"]
hellaswag_score = results["results"]["hellaswag"]["acc_norm,none"]

print(f"MMLU:      {mmlu_score * 100:.1f}%")
print(f"HellaSwag: {hellaswag_score * 100:.1f}%")

# Compare two model results
def compare_models(file_a, file_b, tasks):
    with open(file_a) as f: results_a = json.load(f)
    with open(file_b) as f: results_b = json.load(f)
    
    for task in tasks:
        score_a = results_a["results"][task]["acc,none"] * 100
        score_b = results_b["results"][task]["acc,none"] * 100
        diff = score_b - score_a
        print(f"{task:20s}: {score_a:.1f}% → {score_b:.1f}% ({diff:+.1f}%)")
```

**C# equivalent:**
```csharp
using System.Text.Json;
var json = File.ReadAllText("results.json");
var results = JsonSerializer.Deserialize<Dictionary<string, object>>(json);
```

---

## 6. subprocess — running lm-eval from Python

For automating evaluation runs.

```python
import subprocess
import json

def run_evaluation(model_name, tasks, num_fewshot=5, output_path="results.json"):
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name}",
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--output_path", output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error:", result.stderr)
        return None
    
    with open(output_path) as f:
        return json.load(f)

# Usage
scores = run_evaluation(
    model_name="mistralai/Mistral-7B-v0.1",
    tasks=["mmlu", "hellaswag", "gsm8k"],
    num_fewshot=5
)
```

**Why subprocess?** lm-eval is a CLI tool. `subprocess.run()` lets you call CLI tools
from Python — like `Process.Start()` in C#.

---

## 7. matplotlib — plotting loss curves

Used to visualize training progress.

```python
import matplotlib.pyplot as plt

train_losses = [4.2, 3.8, 3.3, 2.9, 2.6, 2.3, 2.1, 1.9, 1.8, 1.75]
val_losses   = [4.3, 3.9, 3.5, 3.1, 2.9, 2.7, 2.6, 2.55, 2.52, 2.51]
epochs = list(range(1, len(train_losses) + 1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
ax1.plot(epochs, train_losses, label="Train Loss", color="blue")
ax1.plot(epochs, val_losses, label="Val Loss", color="orange")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training vs Validation Loss")
ax1.legend()
ax1.grid(True)

# Perplexity curves
import math
train_ppl = [math.exp(l) for l in train_losses]
val_ppl   = [math.exp(l) for l in val_losses]

ax2.plot(epochs, train_ppl, label="Train PPL", color="blue")
ax2.plot(epochs, val_ppl,   label="Val PPL",   color="orange")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Perplexity")
ax2.set_title("Perplexity")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
```

`plt.subplots(1, 2)` creates 2 side-by-side plots — `ax1` and `ax2` are independent axes.

---

## 8. evaluate library — HuggingFace metrics

One library for BLEU, ROUGE, BERTScore.

```python
from evaluate import load

# BLEU
bleu = load("bleu")
result = bleu.compute(
    predictions=["the cat sat on a rug"],
    references=[["the cat sat on the mat"]]   # list of reference lists
)
print(f"BLEU: {result['bleu']:.3f}")

# ROUGE
rouge = load("rouge")
result = rouge.compute(
    predictions=["the cat sat on a rug"],
    references=["the cat sat on the mat"]     # single reference strings
)
print(f"ROUGE-1: {result['rouge1']:.3f}")
print(f"ROUGE-2: {result['rouge2']:.3f}")
print(f"ROUGE-L: {result['rougeL']:.3f}")

# BERTScore (downloads BERT model on first run)
bertscore = load("bertscore")
result = bertscore.compute(
    predictions=["The car is going fast"],
    references=["The vehicle is moving quickly"],
    lang="en"
)
print(f"BERTScore F1: {result['f1'][0]:.3f}")
```

**Install:** `pip install evaluate rouge_score bert_score`

---

## 9. f-string formatting for results tables

Used to print clean comparison tables.

```python
benchmarks = {
    "MMLU":       {"before": 62.5, "after": 61.8},
    "GSM8K":      {"before": 52.2, "after": 71.4},
    "HumanEval":  {"before": 30.5, "after": 30.1},
    "TruthfulQA": {"before": 56.0, "after": 55.2},
}

print(f"{'Benchmark':<15} {'Before':>8} {'After':>8} {'Δ':>8}")
print("-" * 43)

for name, scores in benchmarks.items():
    delta = scores["after"] - scores["before"]
    indicator = "✓" if delta > 0 else ("✗" if delta < -1 else "~")
    print(f"{name:<15} {scores['before']:>7.1f}% {scores['after']:>7.1f}% {delta:>+7.1f}% {indicator}")
```

Output:
```
Benchmark       Before    After        Δ
-------------------------------------------
MMLU             62.5%    61.8%    -0.7% ~
GSM8K            52.2%    71.4%   +19.2% ✓
HumanEval        30.5%    30.1%    -0.4% ~
TruthfulQA       56.0%    55.2%    -0.8% ~
```

**Format spec `:<15`:** left-align, width 15.
**Format spec `:>8`:** right-align, width 8.
**Format spec `:>+7.1f`:** right-align, show sign (+/-), 1 decimal place.

---

## Quick Reference

| Python Feature | C# Equivalent | Used For |
|---------------|---------------|----------|
| `math.exp(x)` | `Math.Exp(x)` | Loss → Perplexity |
| `math.log(x)` | `Math.Log(x)` | Perplexity → Loss |
| `Counter(list)` | `.GroupBy().ToDictionary()` | n-gram frequency counts |
| `zip(a, a[1:])` | `a.Zip(a.Skip(1))` | Generate bigrams |
| `dict.get(k, 0)` | `dict.GetValueOrDefault(k, 0)` | Safe dict access |
| `json.load(f)` | `JsonSerializer.Deserialize()` | Read eval results |
| `subprocess.run()` | `Process.Start()` | Run lm-eval CLI |
| `f"{val:<15}"` | `val.PadRight(15)` | Aligned table output |
| `f"{val:>+7.1f}"` | `val.ToString("+0.0").PadLeft(7)` | Signed float formatting |

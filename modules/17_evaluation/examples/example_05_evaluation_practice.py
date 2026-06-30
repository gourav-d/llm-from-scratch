"""
Example 05: Evaluation in Practice -- Leaderboards, Pitfalls, Comparison
Module 17: LLM Evaluation & Benchmarks

Run:  python example_05_evaluation_practice.py
Deps: none (pure Python)
"""

import math

print("=" * 60)
print("  Example 05: Evaluation in Practice")
print("=" * 60)


# ─────────────────────────────────────────────────────────
# DEMO 1: Reading an Open LLM Leaderboard
# ─────────────────────────────────────────────────────────

print("\n--- DEMO 1: Reading an Open LLM Leaderboard ---")
print()
print("  Real data from HuggingFace Open LLM Leaderboard (approximate, 2024).")
print("  Rows = models. Columns = benchmark scores (%).")
print()

models = [
    # (name,                 params, MMLU,  HellaSwag, ARC-C, TruthfulQA)
    ("LLaMA-3-70B-Instruct",  "70B", 82.0,  88.0,     87.6,  62.8),
    ("LLaMA-3-8B-Instruct",    "8B", 66.6,  82.0,     79.7,  44.0),
    ("Mistral-7B-Instruct-v2", "7B", 62.5,  81.0,     60.0,  56.0),
    ("Phi-3-mini-4k",         "3.8B",68.8,  78.9,     68.6,  56.5),
    ("Gemma-7B-IT",            "7B", 64.3,  80.6,     61.3,  33.0),
    ("Random-4-choice",        "N/A", 25.0,  25.0,    25.0,  25.0),
]

cols = ["MMLU", "HellaSwag", "ARC-C", "TruthfulQA", "Avg"]
print(f"  {'Model':<28}  {'Params':>6}  " + "  ".join(f"{c:>10}" for c in cols))
print("  " + "-" * 90)

for name, params, mmlu, hellaswag, arc, tqa in models:
    avg = (mmlu + hellaswag + arc + tqa) / 4
    row = f"{mmlu:>10.1f}  {hellaswag:>10.1f}  {arc:>10.1f}  {tqa:>10.1f}  {avg:>8.1f}"
    print(f"  {name:<28}  {params:>6}  {row}")

print()
print("  Reading tips:")
print("  1. Always compare same parameter count (7B vs 7B, not 7B vs 70B).")
print("  2. TruthfulQA is consistently lower -- factuality is hard for all models.")
print("  3. 'Avg' hides which task a model excels at -- look at individual scores.")
print("  4. Phi-3 (3.8B) beats Gemma-7B on MMLU -- efficiency matters.")


# ─────────────────────────────────────────────────────────
# DEMO 2: Before vs After Fine-Tuning
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 2: Interpreting Fine-Tuning Results ---")
print()
print("  Scenario: fine-tune Mistral-7B on math dataset (target: GSM8K improvement).")
print()

before = {"MMLU": 62.5, "HellaSwag": 81.0, "ARC-C": 60.0, "GSM8K": 52.2, "TruthfulQA": 56.0}
after  = {"MMLU": 61.8, "HellaSwag": 80.2, "ARC-C": 59.5, "GSM8K": 71.4, "TruthfulQA": 54.8}

print(f"  {'Benchmark':<14}  {'Before':>8}  {'After':>8}  {'Delta':>8}  {'Verdict'}")
print("  " + "-" * 58)

for bench in before:
    b = before[bench]
    a = after[bench]
    delta = a - b
    if bench == "GSM8K":
        verdict = "TARGET -- excellent gain!" if delta > 10 else "Improved"
    elif abs(delta) < 1:
        verdict = "Stable"
    elif delta < 0 and abs(delta) < 2:
        verdict = "Slight regression (acceptable)"
    elif delta < -2:
        verdict = "WARNING: notable regression"
    else:
        verdict = "Improved"
    sign = "+" if delta >= 0 else ""
    print(f"  {bench:<14}  {b:>8.1f}  {a:>8.1f}  {sign}{delta:>7.1f}  {verdict}")

print()
print("  Decision: Fine-tuning worked. GSM8K improved +19.2%.")
print("  Small regressions on general benchmarks (<2%) are acceptable trade-off.")
print("  This is called the 'specialization-generalization trade-off'.")


# ─────────────────────────────────────────────────────────
# DEMO 3: Common Pitfalls
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 3: Common Evaluation Pitfalls ---")
print()

# Pitfall 1: 0-shot vs Few-shot comparison
print("  PITFALL 1: Comparing 0-shot vs 5-shot results.")
print()
gpt3_hellaswag = [("0-shot", 33.7), ("1-shot", 54.7), ("5-shot", 78.5), ("10-shot", 79.3)]
print("  GPT-3 on HellaSwag:")
print(f"  {'Setting':>8}  {'Accuracy':>10}  {'Note'}")
print("  " + "-" * 40)
for setting, acc in gpt3_hellaswag:
    note = ""
    if setting == "0-shot":
        note = "<-- barely above random!"
    elif setting == "5-shot":
        note = "<-- standard benchmark"
    print(f"  {setting:>8}  {acc:>10.1f}%  {note}")

print()
print("  0-shot vs 5-shot is NOT a fair comparison.")
print("  Always report the number of shots used.")

# Pitfall 2: Data contamination
print()
print()
print("  PITFALL 2: Training data contamination.")
print()

def contamination_impact_demo():
    """Show how contamination inflates scores."""
    # Simulate: model performance on clean vs contaminated subset
    clean_score = 72.3
    contaminated_score = 91.8  # artificially high because model memorized answers

    print("  If MMLU test questions appear in your training data:")
    print(f"    Clean eval subset:         {clean_score}% (real capability)")
    print(f"    Contaminated eval subset:  {contaminated_score}% (memorized!)")
    print(f"    Inflation:                 +{contaminated_score - clean_score:.1f}%")
    print()
    print("  This is why lm-evaluation-harness includes contamination checks.")
    print("  Rule: always verify your training data does NOT contain benchmark questions.")

contamination_impact_demo()

# Pitfall 3: Single benchmark
print()
print()
print("  PITFALL 3: Trusting a single benchmark score.")
print()

claims = [
    ("Beats GPT-4 on GSM8K!", {"GSM8K": 93, "MMLU": 48, "TruthfulQA": 31, "HumanEval": 22}),
    ("Best model ever!",      {"GSM8K": 50, "MMLU": 50, "TruthfulQA": 50, "HumanEval": 50}),
    ("Balanced model",        {"GSM8K": 79, "MMLU": 72, "TruthfulQA": 58, "HumanEval": 65}),
]

for claim, scores in claims:
    avg = sum(scores.values()) / len(scores)
    print(f"  Claim: '{claim}'")
    for bench, score in scores.items():
        bar = "#" * (score // 5)
        print(f"    {bench:<12}: {score:>3}%  {bar}")
    print(f"    Average: {avg:.0f}%")
    if max(scores.values()) - min(scores.values()) > 40:
        print("    RED FLAG: huge variance -- model specialized, not generally good!")
    print()

print("  Always report a full benchmark SUITE, not just one number.")
print("  Cherry-picking benchmarks is the most common way to mislead.")

# Summary
print()
print()
print("  EVALUATION CHECKLIST:")
print("  [ ] Training: train loss and val loss both logged")
print("  [ ] Perplexity on held-out validation set")
print("  [ ] MMLU + HellaSwag + ARC-C (general capability)")
print("  [ ] Task-specific: GSM8K (math) / HumanEval (code) / TruthfulQA (factuality)")
print("  [ ] Text quality: BLEU/ROUGE/BERTScore if generating free-form text")
print("  [ ] Shot count reported (0-shot or N-shot)")
print("  [ ] Contamination check performed")
print("  [ ] Regression check: did fine-tuning hurt other benchmarks?")

"""
Example 02: General Benchmarks -- MMLU, HellaSwag, ARC
Module 17: LLM Evaluation & Benchmarks

Run:  python example_02_general_benchmarks.py
Deps: none (pure Python)
"""

import random
import math

print("=" * 60)
print("  Example 02: General Benchmarks (MMLU, HellaSwag, ARC)")
print("=" * 60)


# ─────────────────────────────────────────────────────────
# DEMO 1: MMLU-Style Multiple Choice Evaluation
# ─────────────────────────────────────────────────────────

print("\n--- DEMO 1: MMLU-Style Evaluation ---")
print()
print("  MMLU = Massive Multitask Language Understanding")
print("  57 academic subjects, 4-choice multiple choice, 25% random baseline.")
print()

# Simulated MMLU questions and "model" answers
mmlu_questions = [
    {
        "subject":   "Mathematics",
        "question":  "What is the derivative of sin(x)?",
        "choices":   ["A: sin(x)", "B: cos(x)", "C: -sin(x)", "D: -cos(x)"],
        "answer":    "B",
        "model_out": "B",   # correct
    },
    {
        "subject":   "History",
        "question":  "In what year did World War II end?",
        "choices":   ["A: 1943", "B: 1944", "C: 1945", "D: 1946"],
        "answer":    "C",
        "model_out": "C",   # correct
    },
    {
        "subject":   "Biology",
        "question":  "What molecule carries genetic information in cells?",
        "choices":   ["A: RNA", "B: ATP", "C: DNA", "D: Protein"],
        "answer":    "C",
        "model_out": "A",   # wrong -- model chose RNA
    },
    {
        "subject":   "Computer Science",
        "question":  "What is the time complexity of binary search?",
        "choices":   ["A: O(n)", "B: O(n^2)", "C: O(log n)", "D: O(1)"],
        "answer":    "C",
        "model_out": "C",   # correct
    },
    {
        "subject":   "Medicine",
        "question":  "Which organ produces insulin?",
        "choices":   ["A: Liver", "B: Kidney", "C: Pancreas", "D: Stomach"],
        "answer":    "C",
        "model_out": "A",   # wrong -- model chose Liver
    },
    {
        "subject":   "Philosophy",
        "question":  "Who wrote 'Critique of Pure Reason'?",
        "choices":   ["A: Descartes", "B: Hume", "C: Kant", "D: Locke"],
        "answer":    "C",
        "model_out": "C",   # correct
    },
]

def evaluate_multiple_choice(questions):
    """Evaluate model on multiple choice questions, return accuracy per subject."""
    correct_by_subject = {}
    total_by_subject   = {}

    for q in questions:
        subj = q["subject"]
        if subj not in correct_by_subject:
            correct_by_subject[subj] = 0
            total_by_subject[subj]   = 0

        total_by_subject[subj] += 1
        if q["model_out"] == q["answer"]:
            correct_by_subject[subj] += 1

    results = {}
    for subj in total_by_subject:
        results[subj] = {
            "correct": correct_by_subject[subj],
            "total":   total_by_subject[subj],
            "accuracy": correct_by_subject[subj] / total_by_subject[subj],
        }
    return results

print("  Questions shown:")
for q in mmlu_questions:
    mark = "PASS" if q["model_out"] == q["answer"] else "FAIL"
    print(f"    [{mark}] [{q['subject']:>18}] Model={q['model_out']}  Answer={q['answer']}")

results = evaluate_multiple_choice(mmlu_questions)
correct_total = sum(r["correct"] for r in results.values())
total_total   = sum(r["total"]   for r in results.values())
overall_acc   = correct_total / total_total

print()
print(f"  Overall accuracy:  {correct_total}/{total_total} = {overall_acc:.1%}")
print(f"  Random baseline:   1/4 = 25.0%")
print(f"  Above random?      {'Yes' if overall_acc > 0.25 else 'No'}")
print(f"  Margin over random: +{(overall_acc - 0.25)*100:.1f}%")


# ─────────────────────────────────────────────────────────
# DEMO 2: HellaSwag -- Common Sense Completion
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 2: HellaSwag -- Common Sense Completion ---")
print()
print("  HellaSwag: pick the most natural continuation of a paragraph.")
print("  Wrong answers are adversarially filtered -- plausible but wrong.")
print()

hellaswag_examples = [
    {
        "context":  "She found a stray cat in the park. She decided to take it home.",
        "choices": [
            "A: She drove her car to the bank to make a deposit.",
            "B: She made a bed for it using an old blanket.",
            "C: She finished her quarterly financial report.",
            "D: She called the fire department.",
        ],
        "answer":    "B",
        "model_out": "B",
    },
    {
        "context":  "He was hungry after a long workout at the gym.",
        "choices": [
            "A: He sat down and wrote a poem about nature.",
            "B: He calculated his taxes for the year.",
            "C: He headed to the kitchen and made a protein shake.",
            "D: He started reading a book about art history.",
        ],
        "answer":    "C",
        "model_out": "C",
    },
    {
        "context":  "The chef sliced the vegetables and heated the pan.",
        "choices": [
            "A: He began writing an email to his colleagues.",
            "B: He added oil and began stir-frying the vegetables.",
            "C: He organized the items by color and size.",
            "D: He went to the living room to watch television.",
        ],
        "answer":    "B",
        "model_out": "A",   # wrong
    },
]

for i, ex in enumerate(hellaswag_examples, 1):
    mark = "PASS" if ex["model_out"] == ex["answer"] else "FAIL"
    print(f"  Example {i} [{mark}]:")
    print(f"    Context: {ex['context']}")
    print(f"    Model chose: {ex['model_out']}  |  Correct: {ex['answer']}")
    if ex["model_out"] != ex["answer"]:
        print(f"    Wrong choice: {ex['choices'][ord(ex['model_out']) - ord('A')]}")
        print(f"    Right answer: {ex['choices'][ord(ex['answer']) - ord('A')]}")
    print()

hs_correct = sum(1 for e in hellaswag_examples if e["model_out"] == e["answer"])
hs_acc     = hs_correct / len(hellaswag_examples)
print(f"  HellaSwag accuracy: {hs_correct}/{len(hellaswag_examples)} = {hs_acc:.1%}")


# ─────────────────────────────────────────────────────────
# DEMO 3: Benchmark Comparison Table
# ─────────────────────────────────────────────────────────

print("\n\n--- DEMO 3: Benchmark Scores for Real Models ---")
print()

# Real published benchmark scores (approximate, 2024)
benchmark_data = {
    "GPT-3.5":       {"MMLU": 70.0, "HellaSwag": 85.5, "ARC-C": 79.7, "TruthfulQA": 47.0},
    "GPT-4":         {"MMLU": 86.4, "HellaSwag": 95.3, "ARC-C": 96.3, "TruthfulQA": 59.0},
    "LLaMA-3-8B":    {"MMLU": 66.6, "HellaSwag": 82.0, "ARC-C": 79.7, "TruthfulQA": 44.0},
    "LLaMA-3-70B":   {"MMLU": 82.0, "HellaSwag": 88.0, "ARC-C": 87.6, "TruthfulQA": 62.8},
    "Mistral-7B":    {"MMLU": 62.5, "HellaSwag": 81.0, "ARC-C": 60.0, "TruthfulQA": 56.0},
    "Random (4-choice)": {"MMLU": 25.0, "HellaSwag": 25.0, "ARC-C": 25.0, "TruthfulQA": 25.0},
}

benchmarks = ["MMLU", "HellaSwag", "ARC-C", "TruthfulQA"]
print(f"  {'Model':<22}  " + "  ".join(f"{b:>12}" for b in benchmarks) + "  {'Avg':>8}")
print("  " + "-" * 80)

for model, scores in benchmark_data.items():
    vals = [scores[b] for b in benchmarks]
    avg  = sum(vals) / len(vals)
    row  = "  ".join(f"{v:>12.1f}" for v in vals)
    print(f"  {model:<22}  {row}  {avg:>8.1f}")

print()
print("  Note: Always compare models at similar parameter counts (7B vs 7B).")
print("  GPT-4 dominates on hard reasoning (ARC-C 96%) but TruthfulQA is hard for all.")

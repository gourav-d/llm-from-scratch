"""
Module 07 - Reasoning & Coding Models
Exercise 01: Chain-of-Thought Reasoning

GLOSSARY
--------
Chain-of-Thought  : Prompting technique that makes the model show its reasoning
(CoT)               step-by-step before giving a final answer.
                    Like showing your work on a math exam.
                    Result: models make far fewer reasoning mistakes.
Zero-shot CoT     : Add "Let's think step by step." to the prompt.
                    No examples needed — the magic phrase triggers reasoning.
Few-shot CoT      : Provide 2-3 worked examples BEFORE asking the question.
                    Model learns the reasoning format from the examples.
Step              : One piece of reasoning in a CoT chain.
                    Example: "Step 1: Each ticket costs $12."
Reasoning Chain   : The full sequence of steps from question to answer.
                    Length matters: more steps = more complex reasoning.
Answer Extraction : Pulling the final numerical/text answer out of a CoT response.
                    Typically after "Answer:" or "Therefore:" or "= ".
Direct Prompting  : Just asking "What is X?" with no step-by-step guidance.
                    Fast but fails on multi-step problems.
"""

import re   # re: regular expressions for text parsing (like Regex in C#)

print("=" * 60)
print("Exercise 01: Chain-of-Thought Reasoning")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Count Reasoning Steps in a CoT Response
#
#  Background:
#    When a model uses Chain-of-Thought, it outputs numbered steps.
#    Common patterns:
#      "Step 1: ..."
#      "Step 2: ..."
#    We want to count how many steps are in a response.
#
#    More steps = more detailed reasoning.
#    Counting steps tells us if the model actually reasoned or just guessed.
#
#  Your Task:
#    Write: count_steps(response) -> int
#    Count lines that start with "Step N:" (case-insensitive).
#    Use re.findall() with pattern r"(?i)step\s+\d+:"
#
#  C# Analogy:
#    Like counting Regex.Matches(response, @"Step\s+\d+:").Count
# ============================================================

print("-" * 50)
print("EXERCISE 1: Count Reasoning Steps")
print("-" * 50)
print()


def count_steps(response):
    """
    Count how many numbered steps appear in a CoT response.

    Parameters:
        response (str): The model's full response text.

    Returns:
        int: Number of "Step N:" occurrences found.
    """
    # TODO:
    # Use re.findall(r"(?i)step\s+\d+:", response) to find all step markers.
    # Return len() of that list.
    pass  # Replace with your implementation


response_good = """
Step 1: Each ticket costs $12.
Step 2: There are 5 people going.
Step 3: Total cost = 5 × $12 = $60.
Step 4: They each contribute $60 / 5 = $12.
Answer: $12 per person.
"""

response_no_cot = "The answer is $12."

response_partial = """
Let me think...
Step 1: Find the total.
The total is $60, so each person pays $12.
"""

s1 = count_steps(response_good)
s2 = count_steps(response_no_cot)
s3 = count_steps(response_partial)

if s1 is not None:
    print(f"  Full CoT (4 steps) -> count = {s1}  (expected: 4)")
    print(f"  No CoT             -> count = {s2}  (expected: 0)")
    print(f"  Partial CoT        -> count = {s3}  (expected: 1)")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Extract Final Answer from CoT Response
#
#  Background:
#    CoT responses follow a pattern where the final answer appears
#    after keywords like "Answer:", "Therefore:", or "= ".
#
#    We want to extract just the answer, not the full reasoning.
#    This is used to evaluate whether the model got the right answer.
#
#    Strategy: search for "Answer:" and return everything after it.
#    If not found, return the last non-empty line (fallback).
#
#  Your Task:
#    Write: extract_answer(response) -> str
#    1. Search for "Answer:" (case-insensitive) in the response.
#    2. If found, return the text after "Answer:" stripped of whitespace.
#    3. If not found, return the last non-empty line (fallback).
#
#  C# Analogy:
#    Like finding the index of "Answer:" and doing
#    response.Substring(answerIdx + "Answer:".Length).Trim()
# ============================================================

print("-" * 50)
print("EXERCISE 2: Extract Final Answer from CoT")
print("-" * 50)
print()


def extract_answer(response):
    """
    Extract the final answer from a CoT response.

    Parameters:
        response (str): Full model response with reasoning steps.

    Returns:
        str: The final answer text (stripped of whitespace).
    """
    # TODO:
    # 1. Use re.search(r"(?i)answer:\s*(.*)", response) to find "Answer: ..."
    # 2. If match found: return match.group(1).strip()
    # 3. Else: return the last non-empty line
    #    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    #    return lines[-1] if lines else ""
    pass  # Replace with your implementation


r1 = """
Step 1: 3 apples × $2 = $6.
Step 2: 2 oranges × $3 = $6.
Step 3: Total = $6 + $6 = $12.
Answer: $12
"""

r2 = """
Let me think step by step.
Total tickets = 3 × 5 = 15.
Therefore: 15 tickets total.
"""

r3 = "The capital of France is Paris."   # No Answer: or Therefore:

a1 = extract_answer(r1)
a2 = extract_answer(r2)
a3 = extract_answer(r3)

if a1 is not None:
    print(f"  Has 'Answer:'    -> '{a1}'  (expected: '$12')")
    print(f"  Has 'Therefore:' -> '{a2}'  (expected: '15 tickets total.')")
    print(f"  No marker        -> '{a3}'  (expected: last line = 'The capital...')")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Compare CoT vs Direct Accuracy
#
#  Background:
#    CoT prompting improves accuracy on multi-step problems.
#    We can measure this by comparing:
#      - Direct answers (no reasoning shown)
#      - CoT answers (step-by-step reasoning)
#
#    accuracy = correct / total  (as a percentage)
#    improvement = cot_accuracy - direct_accuracy
#
#  Your Task:
#    Write: accuracy_comparison(direct_results, cot_results) -> dict
#    direct_results: list of bools (True = correct, False = wrong)
#    cot_results:    list of bools (same length)
#    Returns: {"direct_pct": float, "cot_pct": float, "improvement": float}
#    All values in percentage (0-100).
#
#  C# Analogy:
#    double acc = results.Count(r => r) / (double)results.Count * 100;
# ============================================================

print("-" * 50)
print("EXERCISE 3: CoT vs Direct Accuracy")
print("-" * 50)
print()


def accuracy_comparison(direct_results, cot_results):
    """
    Compare accuracy of direct answering vs Chain-of-Thought.

    Parameters:
        direct_results (list of bool): True if direct answer was correct.
        cot_results    (list of bool): True if CoT answer was correct.

    Returns:
        dict: {
            "direct_pct"  : float -- % correct without CoT,
            "cot_pct"     : float -- % correct with CoT,
            "improvement" : float -- percentage point improvement
        }
    """
    # TODO:
    # direct_pct  = 100 * sum(direct_results) / len(direct_results)
    # cot_pct     = 100 * sum(cot_results)    / len(cot_results)
    # improvement = cot_pct - direct_pct
    pass  # Replace with your implementation


# Simulated results: 10 math problems
direct = [True, False, False, True, False, False, True, False, False, True]
cot    = [True, True,  True,  True, False, True,  True, True,  False, True]

r = accuracy_comparison(direct, cot)
if r:
    print(f"  Direct accuracy : {r['direct_pct']:.1f}%  (expected: 40.0%)")
    print(f"  CoT accuracy    : {r['cot_pct']:.1f}%  (expected: 80.0%)")
    print(f"  Improvement     : +{r['improvement']:.1f} percentage points")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Build a Zero-Shot CoT Prompt
#
#  Background:
#    Zero-shot CoT: Append "Let's think step by step." to any question.
#    This simple addition dramatically improves reasoning on math/logic tasks.
#
#    The prompt format is:
#      "<question>
#       Let's think step by step."
#
#    Few-shot CoT adds examples BEFORE the question:
#      "<example_q>
#       <example_cot_answer>
#
#       <question>
#       Let's think step by step."
#
#  Your Task:
#    Write: zero_shot_cot_prompt(question) -> str
#    Appends the CoT trigger phrase to the question.
#
#    Write: few_shot_cot_prompt(question, examples) -> str
#    examples: list of (question_str, answer_str) tuples
#    Prepends examples, then appends the question + CoT trigger.
#
#  C# Analogy:
#    Like building a string with $"Q: {question}\nA: Let's think step by step."
# ============================================================

print("-" * 50)
print("EXERCISE 4: Build CoT Prompts")
print("-" * 50)
print()


def zero_shot_cot_prompt(question):
    """
    Build a zero-shot CoT prompt by appending the trigger phrase.

    Parameters:
        question (str): The question to answer.

    Returns:
        str: Prompt with "Let's think step by step." appended.
    """
    # TODO:
    # Return f"Q: {question}\nA: Let's think step by step."
    pass  # Replace with your implementation


def few_shot_cot_prompt(question, examples):
    """
    Build a few-shot CoT prompt with worked examples.

    Parameters:
        question (str)       : The target question.
        examples (list)      : List of (question_str, cot_answer_str) tuples.

    Returns:
        str: Prompt with examples prepended then question + CoT trigger.
    """
    # TODO:
    # For each (q, a) in examples, add "Q: {q}\nA: {a}\n\n"
    # Then append "Q: {question}\nA: Let's think step by step."
    pass  # Replace with your implementation


q = "If a train travels 60 mph for 2.5 hours, how far does it go?"

prompt_zero = zero_shot_cot_prompt(q)
if prompt_zero is not None:
    print("  Zero-shot CoT prompt:")
    print("  " + "\n  ".join(prompt_zero.split("\n")))
    print()

examples = [
    ("A car goes 30 mph for 2 hours. How far?",
     "Step 1: distance = speed × time = 30 × 2 = 60 miles.\nAnswer: 60 miles."),
]

prompt_few = few_shot_cot_prompt(q, examples)
if prompt_few is not None:
    print("  Few-shot CoT prompt (first 200 chars):")
    print("  " + prompt_few[:200].replace("\n", "\n  "))
print()

print("=" * 60)
print("All exercises complete!")
print("=" * 60)

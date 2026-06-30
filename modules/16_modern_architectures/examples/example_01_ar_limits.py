"""
Example 01: Auto-Regressive Generation Limits
Module 16: Modern LLM Architectures

Demonstrates the three core limitations of auto-regressive models:
  1. Sequential generation -- cannot parallelize at inference
  2. No revision -- wrong tokens stay wrong
  3. Left-to-right bias -- cannot use right context

Run:  python example_01_ar_limits.py
Deps: none (pure Python)
"""

import time
import random


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def print_section(title):
    print("\n" + "=" * 60)
    print("  " + title)
    print("=" * 60)


def fake_forward_pass(tokens, sleep_ms=0.05):
    """
    Simulates one transformer forward pass.
    In a real model, this runs millions of matrix multiplications.
    Here we just sleep to simulate compute time.
    Returns a fake next-token prediction.
    """
    time.sleep(sleep_ms / 1000)  # simulate compute delay (tiny)
    return len(tokens) % 10     # fake token id


# ─────────────────────────────────────────────────────────
# DEMO 1: Sequential Generation -- Forward Pass Count
# ─────────────────────────────────────────────────────────

print_section("DEMO 1: Sequential Generation -- Forward Pass Counter")

print("""
Auto-regressive generation requires ONE forward pass per token.
Generating N tokens = N forward passes. They cannot run in parallel
because each step depends on the output of the previous step.
""")

def ar_generate(prompt_tokens, num_tokens_to_generate):
    """
    Simulate AR generation step by step.
    Returns (all_tokens, forward_pass_count).
    """
    tokens = list(prompt_tokens)          # start from the prompt
    forward_passes = 0                    # count how many model calls we make

    print(f"  Prompt: {tokens}")

    for step in range(num_tokens_to_generate):
        new_token = fake_forward_pass(tokens)  # one full model forward pass
        forward_passes += 1
        tokens.append(new_token)
        print(f"  Step {step+1:>3}: forward pass #{forward_passes} → new token {new_token} → tokens so far: {tokens}")

    return tokens, forward_passes


prompt = [10, 20, 30]     # 3 prompt tokens
result, n_passes = ar_generate(prompt, num_tokens_to_generate=5)

print(f"\n  Result: {result}")
print(f"  Total forward passes needed: {n_passes}")
print(f"  Generated {len(result) - len(prompt)} tokens in {n_passes} sequential steps")
print("""
  Key insight: step 3 cannot start until step 2 finishes.
  Step 2 cannot start until step 1 finishes.
  This is inherently sequential -- like a for-loop with data dependencies.

  C# analogy: like a chain of await calls where each depends on the previous:
    var t1 = await GenerateToken(prompt);
    var t2 = await GenerateToken(prompt + t1);
    var t3 = await GenerateToken(prompt + t1 + t2);
    // Cannot use Task.WhenAll -- they are not independent.
""")


# ─────────────────────────────────────────────────────────
# DEMO 2: Scaling Problem -- N tokens needs N passes
# ─────────────────────────────────────────────────────────

print_section("DEMO 2: Scaling -- Forward Passes vs Tokens Generated")

print(f"\n  {'Tokens to generate':>20} {'Forward passes needed':>22}")
print("  " + "-" * 44)

for n in [1, 10, 50, 100, 500, 1000, 4096]:
    # Each token needs exactly 1 forward pass in AR generation
    print(f"  {n:>20} {n:>22}")

print("""
  AR: 1 token = 1 forward pass. Always. No batching possible at inference.

  Diffusion alternative: generate ALL tokens in T=50 steps (parallel per step)
    4096 tokens with T=50: only 50 forward passes -- 82x fewer than AR!
  (Trade-off: each diffusion step is slightly more expensive per pass.)
""")


# ─────────────────────────────────────────────────────────
# DEMO 3: No Revision -- The Wrong Token Problem
# ─────────────────────────────────────────────────────────

print_section("DEMO 3: No Revision -- Wrong Tokens Stay Wrong")

print("""
Once an AR model generates a wrong token, all future tokens are conditioned
on that mistake. The model cannot go back and fix it.
""")

def ar_generate_with_mistake(sentence_words, mistake_at_position, wrong_word, correct_word):
    """
    Simulate AR generation that makes a mistake at a given position.
    Shows how the mistake propagates through the rest of the generation.
    """
    generated = []

    for i, word in enumerate(sentence_words):
        if i == mistake_at_position:
            # Model makes a mistake here
            chosen = wrong_word
            tag = " <-- WRONG (model confident but incorrect)"
        else:
            chosen = word
            tag = ""

        generated.append(chosen)
        context = " ".join(generated)
        print(f"  Step {i+1}: context = '{context}'{tag}")

    return generated


print("  Correct sentence: 'The capital of France is Paris'")
print("  AR model generates token by token:")
print()

result = ar_generate_with_mistake(
    sentence_words=["The", "capital", "of", "France", "is", "Paris"],
    mistake_at_position=5,
    wrong_word="Berlin",
    correct_word="Paris"
)

print(f"\n  Final output: '{' '.join(result)}'")
print("""
  Problem: "Berlin" is generated. The model cannot go back.
  Next tokens will continue from "...France is Berlin..." as if it were correct.
  The error COMPOUNDS -- each subsequent token is conditioned on the mistake.

  This is why hallucinations in GPT-style models are hard to fix:
  the model cannot revise. It can only continue.

  Diffusion models CAN revise -- they update all positions at each step.
  A wrong token at step T can be corrected at step T+1.
""")


# ─────────────────────────────────────────────────────────
# DEMO 4: Left-to-Right Bias -- Missing Right Context
# ─────────────────────────────────────────────────────────

print_section("DEMO 4: Left-to-Right Bias -- Missing Right Context")

print("""
AR models only see LEFT context when predicting each token.
Right context (future words) is hidden by the causal mask.

Example: fill in [BLANK] in "The ___ is very fast."
  Left context only:  "The" → ???   (could be anything)
  Full context:       "The ___ is very fast" → "car", "train", "cheetah", etc.

The word "fast" on the RIGHT is a huge clue -- but AR cannot use it.
""")

def show_context_available(sentence, blank_position):
    """
    Show what context an AR model sees vs what a bidirectional model sees.
    """
    words = sentence.split()
    words[blank_position] = "[BLANK]"

    print(f"  Sentence: {' '.join(words)}")
    print()

    # AR model: only sees left context
    left_context = words[:blank_position]
    right_context = words[blank_position + 1:]

    print(f"  AR model sees:              '{' '.join(left_context) if left_context else '(nothing)'}'")
    print(f"  AR model is blind to:       '{' '.join(right_context)}'")
    print(f"  Bidirectional model sees:   '{' '.join(left_context)} [BLANK] {' '.join(right_context)}'")
    print()

    # Show how much information is lost
    left_words = len(left_context)
    right_words = len(right_context)
    total = left_words + right_words
    pct_blind = right_words / total * 100 if total > 0 else 0
    print(f"  AR model is BLIND to {right_words}/{total} words ({pct_blind:.0f}% of context)")


show_context_available("The car is very fast indeed", blank_position=1)
show_context_available("Despite the heavy rain the match continued", blank_position=4)
show_context_available("The patient was carefully monitored by doctors", blank_position=2)

print("""
  BERT (bidirectional) solves this -- but BERT cannot generate text.
  Diffusion models also use FULL bidirectional context at each step
  because all positions are visible (only some are masked).
""")


# ─────────────────────────────────────────────────────────
# DEMO 5: Summary Comparison
# ─────────────────────────────────────────────────────────

print_section("DEMO 5: AR vs Diffusion vs MTP -- Side by Side")

print("""
  Property               AR (GPT)         Diffusion LM       MTP
  ──────────────────────────────────────────────────────────────────────
  Generation order       Left to right    All at once         Left to right
  Forward passes (N tok) N passes         T passes (T<<N)     N passes
  Can revise tokens?     No               Yes                 No
  Uses right context?    No               Yes (all visible)   No
  Training complexity    Simple           Moderate            Simple+
  Drop-in improvement?   N/A (baseline)   Full redesign       Yes (add heads)
  Best for               Everything       Long parallel gen   Code, math, reasoning
  Production use 2025    Dominant         Emerging            Growing (Meta LLaMA)
""")

print("  Conclusion: AR is fast to train and simple, but has real limits.")
print("  Diffusion and MTP are different approaches to overcoming those limits.")

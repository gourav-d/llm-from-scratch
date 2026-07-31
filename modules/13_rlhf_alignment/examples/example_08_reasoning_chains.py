"""
=============================================================================
MODULE 13 - EXAMPLE 08: Reasoning Chains — How Thinking Emerges from RL
=============================================================================

WHAT YOU WILL LEARN:
  - How to parse and score reasoning chain responses
  - Why longer thinking correlates with higher accuracy (test-time compute)
  - How backtracking and self-verification emerge from reward signals
  - How to measure the "quality" of a reasoning chain
  - What the DeepSeek-R1 training output looks like

C# ANALOGY:
  Reasoning chains are like debug logs that the model writes to itself.
  The model learns: "When I write out my steps, I make fewer mistakes."
  Just like a developer who adds logging statements catches bugs faster.

  Training signal:
    Response with debug logs + correct answer = reward 1.2
    Response without logs + correct answer    = reward 1.0
    Response with logs + wrong answer         = reward 0.2
    Response without logs + wrong answer      = reward 0.0

  Model learns: always add debug logs (reasoning).

=============================================================================

PART A: Pure Python — no external libraries required
Run with: python example_08_reasoning_chains.py

=============================================================================
"""

import re       # regular expressions for parsing reasoning chains
import math     # math functions for test problems


# =============================================================================
# SECTION 1: PARSING REASONING CHAIN RESPONSES
# =============================================================================

def parse_response(response: str) -> dict:
    """
    Parse a reasoning chain response into its components.

    Expected format:
      <think>
        [reasoning steps here]
      </think>
      [final answer here]

    Parameters:
      response : the full model response string

    Returns:
      dict with:
        "has_thinking"    : bool — did the model use <think> tags?
        "thinking"        : str  — content inside <think>...</think>
        "answer"          : str  — content after </think>
        "thinking_length" : int  — number of characters in thinking
        "num_steps"       : int  — estimated number of reasoning steps
        "has_backtrack"   : bool — did the model reconsider mid-reasoning?
        "has_verification": bool — did the model verify its answer?
    """
    result = {
        "has_thinking":     False,
        "thinking":         "",
        "answer":           "",
        "thinking_length":  0,
        "num_steps":        0,
        "has_backtrack":    False,
        "has_verification": False,
    }

    # Check if response uses <think> tags
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    # re.DOTALL makes '.' match newlines too (multiline thinking)

    if think_match:
        result["has_thinking"]    = True
        result["thinking"]        = think_match.group(1).strip()
        result["thinking_length"] = len(result["thinking"])

        # Count reasoning steps: lines that start with numbers or "Step" or "-"
        lines = result["thinking"].split('\n')
        step_lines = [l for l in lines
                      if re.match(r'^\s*(Step\s+\d|[\d]+[.)]\s|\-\s)', l)]
        result["num_steps"] = max(len(step_lines), 1)   # at least 1 step if thinking exists

        # Detect backtracking: phrases like "wait", "actually", "no", "let me reconsider"
        backtrack_phrases = [
            "wait", "actually", "let me reconsider", "that's wrong",
            "i made an error", "let me redo", "no,", "hmm,",
            "let me try again", "that doesn't seem right"
        ]
        thinking_lower = result["thinking"].lower()
        result["has_backtrack"] = any(phrase in thinking_lower for phrase in backtrack_phrases)

        # Detect self-verification: phrases like "let me verify", "checking", "to confirm"
        verify_phrases = [
            "let me verify", "let me check", "verify:", "checking:",
            "to confirm", "double-check", "let me double", "indeed"
        ]
        result["has_verification"] = any(phrase in thinking_lower for phrase in verify_phrases)

        # Extract answer (everything after </think>)
        after_think = response.split("</think>")[-1].strip()
        result["answer"] = after_think

    else:
        # No thinking tags — whole response is the answer
        result["answer"] = response.strip()

    return result


def score_reasoning_quality(parsed: dict) -> float:
    """
    Score the quality of a reasoning chain beyond just correctness.

    This is NOT the RLVR reward — it's a diagnostic tool.
    It measures HOW the model reasoned, not just WHAT it answered.

    Higher quality reasoning:
      - Uses thinking tags
      - Has multiple steps
      - Shows backtracking (caught potential errors)
      - Includes verification

    Parameters:
      parsed : output of parse_response()

    Returns:
      float in [0.0, 1.0]
    """
    score = 0.0

    # Did the model use <think> tags at all?
    if parsed["has_thinking"]:
        score += 0.3    # base credit for using structured thinking

    # More steps = more thorough reasoning (up to a point)
    # 1 step: minimal, 5+ steps: thorough
    step_score = min(parsed["num_steps"] / 5.0, 1.0) * 0.3   # up to 0.3
    score += step_score

    # Backtracking = model caught a potential error
    if parsed["has_backtrack"]:
        score += 0.2

    # Verification = model double-checked its answer
    if parsed["has_verification"]:
        score += 0.2

    return min(score, 1.0)   # cap at 1.0


# =============================================================================
# SECTION 2: EXAMPLES OF REASONING CHAIN EVOLUTION
# =============================================================================

def show_reasoning_examples():
    """
    Show how reasoning chains evolve during training.

    Early in training: short or no reasoning.
    Late in training: extended, verified reasoning.
    """
    print("=" * 65)
    print("REASONING CHAIN EVOLUTION DURING RLVR TRAINING")
    print("=" * 65)

    prompt = "A store sells apples for $0.50 each. If you buy 12 apples, \nhow much do you pay in total?"
    correct_answer = "6"

    print(f"\nPrompt: {prompt}")
    print(f"Correct answer: ${correct_answer}.00\n")

    # Simulate different stages of training
    responses = [
        ("Early training (step 100)",
         "I don't know"),

        ("Early training (step 500)",
         "6"),

        ("Mid training (step 2000)",
         "The total is $6.00"),

        ("Mid training (step 5000)",
         "<think>\n"
         "Price per apple = $0.50\n"
         "Number of apples = 12\n"
         "Total = 0.50 * 12 = 6\n"
         "</think>\n"
         "6"),

        ("Late training (step 20000)",
         "<think>\n"
         "Step 1: Identify what we know.\n"
         "  - Price per apple = $0.50\n"
         "  - Quantity = 12 apples\n"
         "\n"
         "Step 2: Calculate total cost.\n"
         "  Total = price × quantity\n"
         "  Total = $0.50 × 12\n"
         "  Total = $6.00\n"
         "\n"
         "Step 3: Verify my answer.\n"
         "  Let me check: 12 × 0.5 = 6. Yes, that's correct.\n"
         "  Alternative: 12 × 50 cents = 600 cents = $6.00. Confirmed.\n"
         "</think>\n"
         "6"),

        ("Late training — hard problem, backtracking",
         "<think>\n"
         "Step 1: 0.50 * 12...\n"
         "Hmm, let me be careful here. 0.5 * 12 = ...\n"
         "Wait, I almost wrote 0.5 * 12 = 5. That's wrong.\n"
         "Actually: 0.5 * 10 = 5, and 0.5 * 2 = 1, so 0.5 * 12 = 6.\n"
         "\n"
         "Let me verify: 6 / 12 = 0.5. Yes, $0.50 per apple. Correct.\n"
         "</think>\n"
         "6"),
    ]

    for stage, response in responses:
        parsed = parse_response(response)
        quality = score_reasoning_quality(parsed)

        # Compute RLVR reward
        from example_06_rlvr import total_reward
        reward = total_reward(response, correct_answer)

        print(f"--- {stage} ---")
        print(f"Response: {response[:80]}{'...' if len(response) > 80 else ''}")
        print(f"  Thinking: {'Yes' if parsed['has_thinking'] else 'No':3s} | "
              f"Steps: {parsed['num_steps']:2d} | "
              f"Backtrack: {'Yes' if parsed['has_backtrack'] else 'No':3s} | "
              f"Verify: {'Yes' if parsed['has_verification'] else 'No':3s}")
        print(f"  RLVR reward: {reward:.1f} | Reasoning quality: {quality:.2f}")
        print()


# =============================================================================
# SECTION 3: TEST-TIME COMPUTE SCALING
# =============================================================================

def simulate_test_time_compute():
    """
    Demonstrate how more thinking tokens -> higher accuracy.

    In real reasoning models, harder problems get more tokens.
    We simulate this with a probability model:
      - Without thinking: 60% chance of correct answer
      - With 50-token thinking: 70% chance
      - With 200-token thinking: 80% chance
      - With 1000-token thinking: 90% chance
    """
    print("=" * 65)
    print("TEST-TIME COMPUTE SCALING")
    print("=" * 65)
    print("More thinking tokens = higher accuracy on hard problems\n")

    import random
    random.seed(42)

    # Thinking budget levels to compare
    budgets = [
        ("No thinking",    0,    0.60),
        ("Short (50)",     50,   0.70),
        ("Medium (200)",   200,  0.80),
        ("Long (1000)",    1000, 0.88),
        ("Very long (4k)", 4000, 0.93),
    ]

    num_problems = 1000   # simulate solving this many math problems

    print(f"Simulating {num_problems} math problems per budget level:\n")
    print(f"{'Budget':20s} | {'Tokens':8s} | {'Accuracy':10s} | {'Correct':8s}")
    print("-" * 55)

    for label, tokens, base_accuracy in budgets:
        # Simulate answers: each problem is correct with base_accuracy probability
        correct = sum(1 for _ in range(num_problems)
                      if random.random() < base_accuracy)
        accuracy = correct / num_problems

        # ASCII bar chart
        bar = "#" * int(accuracy * 30)

        print(f"{label:20s} | {tokens:8d} | {accuracy:9.1%} | {bar}")

    print()
    print("Key insight: You can 'buy' accuracy by allocating more thinking tokens.")
    print("Easy questions: use short thinking budget (fast, cheap).")
    print("Hard questions: use long thinking budget (slow, accurate).")


# =============================================================================
# SECTION 4: REWARD SIGNAL DRIVES REASONING EMERGENCE
# =============================================================================

def explain_emergence():
    """
    Show WHY reasoning chains emerge from RLVR training.

    The model doesn't "know" to reason. It learns that reasoning
    correlates with higher rewards. So it generates more reasoning.
    """
    print("=" * 65)
    print("WHY REASONING EMERGES FROM RLVR TRAINING")
    print("=" * 65)

    print("""
The model sees millions of training examples like these:

  Prompt: "What is 17 * 8?"

  Response A:  "100"              reward = 0.0  (wrong)
  Response B:  "136"              reward = 1.0  (correct, no thinking)
  Response C:  "<think>            reward = 1.2  (correct + format bonus)
                17 * 8:
                17 * 8 = 17 * 4 * 2 = 68 * 2 = 136
                </think> 136"

  The model learns the pattern:
    "Short answer, wrong"        -> reward 0.0
    "Short answer, right"        -> reward 1.0
    "Thinking + right answer"    -> reward 1.2   <-- HIGHEST

  After millions of such examples, the model develops:
    1. Use <think> tags (format reward)
    2. Write more steps when unsure (more thinking = more correct)
    3. Backtrack when it detects uncertainty (catches errors)
    4. Verify final answer (reduces careless mistakes)

  None of this was explicitly programmed.
  It ALL emerged from the reward signal.

  This is what DeepSeek called the "aha moment":
    "We observe that the model has developed an interesting
    self-reflection behavior where it reconsiders its approach.
    This is an aha moment not just for the model but for us."
    -- DeepSeek-R1 paper, January 2025
""")


# =============================================================================
# SECTION 5: MEASURE HOW TRAINING IMPROVES REASONING
# =============================================================================

def simulate_training_progress():
    """
    Simulate how reasoning quality metrics improve over RLVR training.
    """
    print("=" * 65)
    print("REASONING QUALITY IMPROVEMENT OVER TRAINING")
    print("=" * 65)

    import random
    random.seed(0)

    # At each training step, simulate the model's response distribution
    # We track: % using thinking, avg steps, % backtracking, % verifying

    checkpoints = [0, 1000, 5000, 10000, 50000]

    # These represent the model's evolving behavior at each checkpoint
    # (based on typical RLVR training curves from literature)
    checkpoint_stats = {
        0:     {"think": 0.10, "steps": 1.2, "backtrack": 0.02, "verify": 0.01},
        1000:  {"think": 0.30, "steps": 2.1, "backtrack": 0.08, "verify": 0.05},
        5000:  {"think": 0.60, "steps": 3.5, "backtrack": 0.20, "verify": 0.18},
        10000: {"think": 0.80, "steps": 5.2, "backtrack": 0.35, "verify": 0.40},
        50000: {"think": 0.92, "steps": 8.1, "backtrack": 0.55, "verify": 0.68},
    }

    print(f"\n{'Step':8s} | {'Uses <think>':12s} | {'Avg steps':10s} | "
          f"{'Backtracks':10s} | {'Verifies':10s}")
    print("-" * 60)

    for step in checkpoints:
        stats = checkpoint_stats[step]
        print(f"{step:8d} | {stats['think']:11.0%} | {stats['steps']:10.1f} | "
              f"{stats['backtrack']:9.0%} | {stats['verify']:9.0%}")

    print()
    print("All metrics increase monotonically — the model learns to reason")
    print("more thoroughly over time, driven only by the RLVR reward signal.")


# =============================================================================
# SECTION 6: ANSWERS TO LESSON SELF-CHECK
# =============================================================================

def print_lesson_answers():
    print("\n" + "=" * 65)
    print("LESSON 08 SELF-CHECK ANSWERS")
    print("=" * 65)

    answers = [
        ("Q1: Why does extended thinking improve accuracy?",
         "Writing steps makes each step's output visible and checkable. "
         "The model can condition step N on the explicit output of step N-1. "
         "More tokens = more opportunities to self-correct before committing to answer."),

        ("Q2: What is the DeepSeek 'aha moment'?",
         "During RLVR training (no human labels, no reasoning examples), "
         "R1-Zero spontaneously started writing long reasoning chains, "
         "backtracking when wrong, and verifying its answers. "
         "DeepSeek did not program this — it emerged from the reward signal."),

        ("Q3: How does backtracking emerge from RLVR?",
         "Responses that backtrack AND get the right answer get reward = 1.0+. "
         "Responses that don't backtrack and get it wrong get reward = 0.0. "
         "Model sees: 'backtracking + correct' correlates with high reward. "
         "So it learns: when uncertain, try backtracking."),

        ("Q4: What is test-time compute scaling?",
         "The observation that allocating more tokens to thinking (at inference) "
         "consistently improves accuracy on hard problems. "
         "You can trade compute for accuracy without retraining the model."),

        ("Q5: What format does DeepSeek-R1 use?",
         "<think>...reasoning...</think> followed by the clean final answer. "
         "The <think> block is the chain-of-thought. "
         "Content after </think> is shown to the user."),
    ]

    for q, a in answers:
        print(f"\n{q}")
        print(f"   {a}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("MODULE 13 - EXAMPLE 08: REASONING CHAINS")

    show_reasoning_examples()
    simulate_test_time_compute()
    explain_emergence()
    simulate_training_progress()
    print_lesson_answers()

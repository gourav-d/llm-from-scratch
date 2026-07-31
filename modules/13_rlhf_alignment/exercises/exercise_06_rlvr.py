"""
=============================================================================
MODULE 13 - EXERCISE 06: RLVR Reward Functions
=============================================================================

YOUR TASK:
  Implement four reward functions used in RLVR training.
  Each function takes a model response and returns a score.

RULES:
  - Do NOT import any external libraries (just Python stdlib)
  - Read each function's docstring carefully
  - The tests at the bottom will check your work

RUN WITH:  python exercise_06_rlvr.py
=============================================================================
"""

import re


# =============================================================================
# EXERCISE 1: Extract Answer
# =============================================================================

def extract_final_answer(response: str) -> str:
    """
    Extract the final numeric answer from a model response.

    The response might be in any of these formats:
      "42"                              -> "42"
      "The answer is 42"               -> "42"
      "<think>...</think> 42"          -> "42"
      "\\boxed{42}"                    -> "42"
      "I think it's 42."               -> "42"

    Rules:
      1. If response contains </think>, take everything after it, strip whitespace
      2. Else if response contains \\boxed{X}, return X
      3. Else find the last number in the response

    Parameters:
      response : full model response string

    Returns:
      extracted answer as a string, or "" if nothing found

    HINTS:
      - Use response.split("</think>") to split on the closing tag
      - Use re.search(r'\\\\boxed\\{([^}]+)\\}', response) for boxed format
      - Use re.findall(r'-?\\d+\\.?\\d*', response) to find all numbers
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 2: Correctness Reward
# =============================================================================

def correctness_reward(response: str, correct_answer: str) -> float:
    """
    Return 1.0 if the response contains the correct answer, 0.0 otherwise.

    Steps:
      1. Extract the answer using extract_final_answer()
      2. Convert both to float for numeric comparison
      3. Return 1.0 if they match within tolerance 0.001
      4. Return 0.0 otherwise

    Parameters:
      response       : model response string
      correct_answer : the known correct answer as a string (e.g. "42")

    Returns:
      1.0 if correct, 0.0 if wrong

    HINTS:
      - Use float() to convert strings to numbers
      - Use abs(a - b) < 0.001 for numeric comparison
      - Wrap float() in try/except in case the string isn't a valid number
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Format Reward
# =============================================================================

def format_reward(response: str) -> float:
    """
    Return 0.2 if the response uses <think>...</think> tags, 0.0 otherwise.

    BOTH the opening AND closing tag must be present.
    A response with only <think> but no </think> gets 0.0.

    Parameters:
      response : model response string

    Returns:
      0.2 if both <think> and </think> present, else 0.0
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Total Reward
# =============================================================================

def total_reward(response: str, correct_answer: str) -> float:
    """
    Compute total RLVR reward = correctness_reward + format_reward.

    This is the reward signal used to train the model.

    Possible values:
      0.0 = wrong, no thinking
      0.2 = wrong, but used thinking (small credit for trying)
      1.0 = correct, no thinking
      1.2 = correct AND used thinking  <-- best possible

    Parameters:
      response       : model response
      correct_answer : correct answer

    Returns:
      float in [0.0, 1.2]
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# TESTS — DO NOT MODIFY BELOW THIS LINE
# =============================================================================

def run_tests():
    print("=" * 55)
    print("EXERCISE 06 TESTS")
    print("=" * 55)
    passed = 0
    total = 0

    def check(name, got, expected):
        nonlocal passed, total
        total += 1
        ok = (got == expected) if isinstance(expected, str) else (abs(got - expected) < 0.01)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {name}")
        if not ok:
            print(f"         Expected: {expected!r}")
            print(f"         Got:      {got!r}")

    # --- extract_final_answer ---
    check("extract: plain number",
          extract_final_answer("42"), "42")
    check("extract: after </think>",
          extract_final_answer("<think>stuff</think> 42"), "42")
    check("extract: boxed",
          extract_final_answer("\\boxed{42}"), "42")
    check("extract: 'The answer is X'",
          extract_final_answer("The answer is 42"), "42")
    check("extract: nothing",
          extract_final_answer("I have no idea"), "")

    # --- correctness_reward ---
    check("correct: exact match",
          correctness_reward("42", "42"), 1.0)
    check("correct: with think tags",
          correctness_reward("<think>math</think> 42", "42"), 1.0)
    check("correct: float tolerance",
          correctness_reward("6.0", "6"), 1.0)
    check("correct: wrong answer",
          correctness_reward("41", "42"), 0.0)
    check("correct: no answer",
          correctness_reward("I don't know", "42"), 0.0)

    # --- format_reward ---
    check("format: both tags present",
          format_reward("<think>reasoning</think> 42"), 0.2)
    check("format: only opening tag",
          format_reward("<think>reasoning 42"), 0.0)
    check("format: no tags",
          format_reward("42"), 0.0)
    check("format: empty think block",
          format_reward("<think></think> 42"), 0.2)

    # --- total_reward ---
    check("total: correct + format",
          total_reward("<think>work</think> 42", "42"), 1.2)
    check("total: correct + no format",
          total_reward("42", "42"), 1.0)
    check("total: wrong + format",
          total_reward("<think>work</think> 99", "42"), 0.2)
    check("total: wrong + no format",
          total_reward("99", "42"), 0.0)

    print(f"\n{passed}/{total} tests passed")
    if passed == total:
        print("All tests passed!")
    else:
        print("Some tests failed. Re-read the docstrings and try again.")


if __name__ == "__main__":
    run_tests()

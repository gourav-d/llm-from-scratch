"""
=============================================================================
MODULE 13 - EXAMPLE 06: RLVR (Reinforcement Learning with Verifiable Rewards)
=============================================================================

WHAT YOU WILL LEARN:
  - What makes a reward "verifiable" vs a learned reward model
  - How to write reward functions for math and code tasks
  - Why verifiable rewards cannot be gamed (no reward hacking)
  - How format rewards encourage chain-of-thought reasoning
  - How reward signals differ across task types

C# ANALOGY:
  RLVR is like a CI/CD pipeline that grades your code automatically:
    1. You submit code (model generates a response)
    2. Unit tests run automatically (verifiable reward checker)
    3. Pass = reward 1.0, Fail = reward 0.0
    4. No human reviewer needed

  In contrast, RLHF is like code review by a human:
    1. You submit code
    2. A human reads it and gives it a score 1-10
    3. Expensive, slow, subjective

=============================================================================

PART A: Pure Python/NumPy — no external libraries required
Run with: python example_06_rlvr.py

=============================================================================
"""

import re       # regular expressions — for extracting answers from text
import math     # math functions — for generating test problems


# =============================================================================
# SECTION 1: VERIFIABLE REWARD FUNCTIONS
# =============================================================================
# These are the "checkers" in RLVR.
# Each one takes a model response and returns a score.
# The score is computed automatically — no human needed.

def extract_answer_from_text(text: str) -> str:
    """
    Extract the final numeric answer from a model response.

    The model might write:
      "<think>5 * 7 = 35</think> 35"
      "The answer is 35."
      "35"
      "\\boxed{35}"

    We want to extract "35" from all of these.

    Parameters:
      text : the model's full response string

    Returns:
      the extracted answer as a string, or "" if nothing found
    """
    # Strategy 1: look for content inside \boxed{} (LaTeX math notation)
    # This is the standard format for math competition answers
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()   # .group(1) = first capture group = content inside {}

    # Strategy 2: look for text AFTER </think> tag (if model showed reasoning)
    # Everything after </think> is the clean answer
    think_split = text.split("</think>")
    if len(think_split) > 1:
        # Take the part after </think>, strip whitespace
        return think_split[-1].strip()

    # Strategy 3: look for "The answer is X" or "= X" at the end
    patterns = [
        r'[Tt]he answer is\s*([\d.]+)',   # "The answer is 42"
        r'=\s*([\d.]+)\s*$',              # "= 42" at end of string
        r'^([\d.]+)\s*$',                 # just a number on its own line
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match:
            return match.group(1).strip()

    # Strategy 4: if all else fails, take the last token that looks like a number
    tokens = text.split()
    for token in reversed(tokens):     # reversed() = iterate from end to start
        cleaned = token.strip('.,!?')
        if re.match(r'^-?[\d.]+$', cleaned):   # matches integers and decimals
            return cleaned

    return ""   # could not extract an answer


def math_reward(response: str, correct_answer: str) -> float:
    """
    Compute reward for a math problem response.

    Reward:
      1.0 = correct answer
      0.0 = wrong answer

    Parameters:
      response       : the model's full response text
      correct_answer : the known correct answer (string)

    Returns:
      float reward between 0.0 and 1.0
    """
    # Extract the answer the model gave
    extracted = extract_answer_from_text(response)

    # Handle empty extraction
    if not extracted:
        return 0.0

    # Normalize: strip whitespace, convert to float for numeric comparison
    try:
        model_num = float(extracted.replace(',', ''))    # remove commas in large numbers
        correct_num = float(correct_answer.replace(',', ''))

        # Allow tiny floating point differences (0.001 tolerance)
        # C# analogy: Math.Abs(a - b) < epsilon
        if abs(model_num - correct_num) < 0.001:
            return 1.0
        else:
            return 0.0

    except ValueError:
        # Could not convert to float — do string comparison
        # This handles non-numeric answers like "prime" or "undefined"
        if extracted.strip().lower() == correct_answer.strip().lower():
            return 1.0
        return 0.0


def format_reward(response: str) -> float:
    """
    Compute a small bonus reward for using the correct reasoning format.

    We reward the model for:
      1. Using <think>...</think> tags to show its work
      2. Putting the final answer in \\boxed{} notation

    This encourages the model to structure its reasoning.

    Returns:
      0.2 if format is correct
      0.0 otherwise

    Note: format reward is SMALL compared to correctness reward.
    This prevents the model from focusing on format over correctness.
    """
    has_think_open  = "<think>"  in response    # opened a think block
    has_think_close = "</think>" in response    # closed a think block

    # Only award format bonus if BOTH tags are present
    # (we don't want to reward half-formed responses)
    if has_think_open and has_think_close:
        return 0.2

    return 0.0


def total_reward(response: str, correct_answer: str) -> float:
    """
    Combined reward = correctness + format bonus.

    This is what RLVR actually uses to train the model.

    Total possible = 1.2 (correct answer + correct format)

    Parameters:
      response       : full model response
      correct_answer : known correct answer

    Returns:
      float in range [0.0, 1.2]
    """
    corr = math_reward(response, correct_answer)
    fmt  = format_reward(response)
    return corr + fmt


# =============================================================================
# SECTION 2: CODE TASK REWARD
# =============================================================================
# For code tasks, the reward comes from running test cases.
# This is the most "hack-proof" reward — tests are deterministic.

def code_reward(generated_code: str, test_cases: list) -> float:
    """
    Compute reward for a code generation response.

    Runs each test case and returns the fraction that pass.

    Parameters:
      generated_code : the Python code string the model wrote
      test_cases     : list of (input, expected_output) tuples

    Returns:
      float in [0.0, 1.0] = fraction of tests that passed

    C# analogy:
      Like running NUnit tests and returning (passed / total) as the score.
    """
    if not generated_code or not test_cases:
        return 0.0

    passed = 0      # count of tests that passed

    for test_input, expected_output in test_cases:
        try:
            # Create a local execution environment
            # C# analogy: like using Roslyn's scripting API to compile and run a snippet
            namespace = {}

            # Execute the generated code in the namespace
            # This is safe for toy examples — in production, use a sandbox
            exec(generated_code, namespace)    # exec() runs a string as Python code

            # Find the function the code defined
            # We assume the function is named 'solution'
            if 'solution' not in namespace:
                continue   # no solution function found — this test fails

            solution_fn = namespace['solution']

            # Call the function with the test input
            actual = solution_fn(test_input)

            # Check if output matches expected
            if actual == expected_output:
                passed += 1

        except Exception:
            # Code threw an error — this test fails (reward = 0 for this test)
            pass

    # Return fraction of tests passed
    # 0/5 = 0.0 (all wrong), 5/5 = 1.0 (all correct)
    return passed / len(test_cases)


# =============================================================================
# SECTION 3: COMPARE RLHF vs RLVR REWARD SIGNALS
# =============================================================================

class SimulatedHumanRewardModel:
    """
    Simulates an RLHF reward model — a neural network trained on human preferences.

    In real RLHF, this would be a large neural network.
    Here we simulate it with a scoring heuristic to illustrate its weaknesses.
    """

    def score(self, response: str) -> float:
        """
        Score a response the way a (flawed) human reward model might.

        This reward model has been "fooled" — it gives higher scores to responses
        that SOUND confident and detailed, even if they're wrong.

        This is an example of reward hacking.
        """
        score = 0.5   # baseline score

        # Heuristic 1: longer responses score higher
        # (reward model learned: longer = more effort = better)
        length_bonus = min(0.3, len(response) / 500)   # up to +0.3
        score += length_bonus

        # Heuristic 2: confident language scores higher
        # (reward model learned: confident = sure = better)
        confidence_words = ["definitely", "certainly", "clearly", "obviously", "the answer is"]
        for word in confidence_words:
            if word.lower() in response.lower():
                score += 0.05

        # Heuristic 3: structured formatting scores higher
        # (reward model learned: bullet points = well-organized = better)
        if "•" in response or "1." in response:
            score += 0.1

        return min(1.0, score)   # cap at 1.0


def demonstrate_reward_hacking():
    """
    Show how RLHF reward models can be gamed vs RLVR which cannot.

    A model could learn to write confident, verbose, formatted responses
    that SCORE WELL but give the WRONG ANSWER.
    """
    human_rm = SimulatedHumanRewardModel()

    correct_answer = "12"
    prompt = "What is 15% of 80?"

    # Compare three responses

    # Response A: Correct, short, no reasoning shown
    response_a = "12"

    # Response B: Wrong, but verbose and confident (REWARD HACKED)
    response_b = (
        "Certainly! Let me clearly work through this step by step:\n"
        "• First, I will definitely convert 15% to a decimal.\n"
        "• 15% = 0.15\n"
        "• Then multiply by 80.\n"
        "• 0.15 × 80 = 11\n"   # WRONG ANSWER
        "The answer is obviously 11."
    )

    # Response C: Correct, with reasoning format
    response_c = "<think>15% = 0.15. 0.15 × 80 = 12.</think> 12"

    print("=" * 60)
    print("REWARD COMPARISON: RLHF vs RLVR")
    print("=" * 60)
    print(f"\nPrompt: {prompt}")
    print(f"Correct answer: {correct_answer}\n")

    for label, response in [("A (correct, short)", response_a),
                             ("B (wrong, verbose)", response_b),
                             ("C (correct, formatted)", response_c)]:

        rlhf_score = human_rm.score(response)
        rlvr_score = total_reward(response, correct_answer)

        print(f"Response {label}:")
        print(f"  Preview: {response[:60]}...")
        print(f"  RLHF reward (human model): {rlhf_score:.2f}")
        print(f"  RLVR reward (verifiable):  {rlvr_score:.2f}")
        print()

    print("KEY INSIGHT:")
    print("  RLHF reward model gives Response B (WRONG) a higher score than A (CORRECT).")
    print("  RLVR reward is 0.0 for B — correct answers cannot be gamed.")


# =============================================================================
# SECTION 4: SIMULATE RLVR TRAINING LOOP (SIMPLIFIED)
# =============================================================================

class SimpleLMSimulator:
    """
    Simulates a language model that improves through RLVR training.

    In real training this would be a transformer. Here we simulate
    a model that generates responses with varying quality, improving
    over time as training rewards correct behavior.

    The "model" here is just a probability distribution over response types:
      - correct with thinking (best)
      - correct without thinking
      - wrong with thinking
      - wrong without thinking (worst)
    """

    def __init__(self):
        # Initial probabilities for each response type
        # At the start, the model has no strong preference
        # These are like logits — they'll change as we "train"
        self.probs = {
            "correct_with_think": 0.15,    # rare at start
            "correct_no_think":   0.35,    # somewhat common
            "wrong_with_think":   0.10,    # rare
            "wrong_no_think":     0.40,    # most common at start
        }

    def generate_response(self, correct_answer: str) -> str:
        """
        Simulate generating a response by sampling from the probability distribution.
        """
        import random
        roll = random.random()   # random float in [0.0, 1.0)

        # Cumulative probability: pick whichever bucket the roll falls in
        cumulative = 0.0
        for response_type, prob in self.probs.items():
            cumulative += prob
            if roll < cumulative:
                # Generate a response of this type
                if response_type == "correct_with_think":
                    return f"<think>Let me compute this carefully. The answer is {correct_answer}.</think> {correct_answer}"
                elif response_type == "correct_no_think":
                    return correct_answer
                elif response_type == "wrong_with_think":
                    wrong = str(int(correct_answer) + 3)   # deliberately wrong
                    return f"<think>Hmm, I think the answer is {wrong}.</think> {wrong}"
                else:
                    wrong = str(int(correct_answer) + 3)
                    return wrong

        return correct_answer  # fallback

    def update(self, response: str, reward: float):
        """
        Simulate RLVR training update: increase probability of rewarded response types.

        In real training: backpropagation + gradient descent.
        Here: just nudge the probabilities in the right direction.
        """
        learning_rate = 0.01   # how fast to update

        # Determine which response type this was
        has_think = "<think>" in response and "</think>" in response
        extracted = extract_answer_from_text(response)

        # Compute correctness (we don't have the real answer here — simplified)
        is_correct = reward >= 1.0

        if is_correct and has_think:
            response_type = "correct_with_think"
        elif is_correct and not has_think:
            response_type = "correct_no_think"
        elif not is_correct and has_think:
            response_type = "wrong_with_think"
        else:
            response_type = "wrong_no_think"

        # If reward is high: increase this type's probability
        # If reward is low:  decrease this type's probability
        adjustment = learning_rate * (reward - 0.5)   # positive if reward > 0.5

        self.probs[response_type] += adjustment

        # Re-normalize so all probabilities sum to 1.0
        # (like softmax normalization)
        total = sum(self.probs.values())
        for k in self.probs:
            self.probs[k] = max(0.001, self.probs[k] / total)   # max(0.001) prevents 0


def run_rlvr_training_simulation():
    """
    Run a mini RLVR training simulation and show how the model improves.
    """
    import random
    random.seed(42)   # fixed seed for reproducibility

    model = SimpleLMSimulator()

    # Training data: simple math problems
    # Format: (problem_text, correct_answer)
    training_problems = [
        ("What is 10 + 5?",  "15"),
        ("What is 3 × 7?",   "21"),
        ("What is 20 - 8?",  "12"),
        ("What is 100 / 4?", "25"),
    ]

    num_epochs = 50      # number of training passes over the data
    step_count = 0

    print("=" * 60)
    print("RLVR TRAINING SIMULATION")
    print("=" * 60)

    # Track performance over time
    # We'll sample at epochs 0, 10, 25, 50 to show improvement
    checkpoints = [0, 10, 25, 49]

    for epoch in range(num_epochs):
        for problem_text, correct_answer in training_problems:
            # 1. Generate response
            response = model.generate_response(correct_answer)

            # 2. Compute reward using verifiable checker
            reward = total_reward(response, correct_answer)

            # 3. Update model (in reality: GRPO gradient update)
            model.update(response, reward)

            step_count += 1

        # Print stats at checkpoints
        if epoch in checkpoints:
            print(f"\n--- Epoch {epoch + 1} ---")
            print(f"Response type probabilities:")
            for rtype, prob in sorted(model.probs.items()):
                bar = "#" * int(prob * 40)   # ASCII bar chart
                print(f"  {rtype:30s} {prob:.3f} {bar}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("The model learned to prefer 'correct_with_think' responses")
    print("because they have the highest reward (1.0 + 0.2 format bonus).")


# =============================================================================
# SECTION 5: ANSWERS TO LESSON SELF-CHECK
# =============================================================================

def print_lesson_answers():
    print("\n" + "=" * 60)
    print("LESSON 06 SELF-CHECK ANSWERS")
    print("=" * 60)
    answers = [
        ("Q1: Verifiable vs preference reward?",
         "Verifiable: automated checker (correct/wrong). "
         "Preference: human labels which response is better. "
         "Verifiable is objective; preference is subjective."),

        ("Q2: Why not RLVR for creative writing?",
         "No automatic way to verify if a creative story is 'correct'. "
         "There is no ground truth to compare against."),

        ("Q3: Why does RLVR avoid reward hacking?",
         "The checker is deterministic. A correct answer IS correct. "
         "The model cannot write a response that 'looks' correct "
         "— the checker actually verifies the answer."),

        ("Q4: Two components of total reward?",
         "correctness_reward (1.0 if correct, 0.0 if wrong) + "
         "format_reward (0.2 bonus for using <think> tags)."),

        ("Q5: DeepSeek-R1 aha moment?",
         "Mid-training, R1-Zero spontaneously started writing "
         "long reasoning chains, backtracking when wrong, and "
         "verifying its answers — without being explicitly trained "
         "to do any of this. It emerged from the reward signal alone."),
    ]
    for q, a in answers:
        print(f"\n{q}")
        print(f"   {a}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("MODULE 13 - EXAMPLE 06: RLVR")
    print("=" * 60)

    # --- Demo 1: Reward functions ---
    print("\n[1] MATH REWARD FUNCTION")
    test_responses = [
        ("12",                                        "12",  "Direct correct answer"),
        ("<think>15% = 0.15, 0.15*80=12</think> 12", "12",  "Correct with thinking"),
        ("\\boxed{12}",                               "12",  "LaTeX boxed format"),
        ("The answer is 11",                          "12",  "Wrong answer"),
        ("I'm not sure but maybe 12?",                "12",  "Vague but extractable"),
    ]

    for response, answer, label in test_responses:
        r = total_reward(response, answer)
        extracted = extract_answer_from_text(response)
        print(f"  {label:35s} | extracted='{extracted:4s}' | reward={r:.1f}")

    # --- Demo 2: Code reward ---
    print("\n[2] CODE REWARD FUNCTION")
    code_correct = """
def solution(n):
    return n * n
"""
    code_wrong = """
def solution(n):
    return n + n
"""
    tests = [(2, 4), (3, 9), (4, 16), (5, 25)]
    print(f"  Correct code reward: {code_reward(code_correct, tests):.2f}")
    print(f"  Wrong code reward:   {code_reward(code_wrong,   tests):.2f}")

    # --- Demo 3: Reward hacking demo ---
    print()
    demonstrate_reward_hacking()

    # --- Demo 4: Training simulation ---
    print()
    run_rlvr_training_simulation()

    # --- Answers ---
    print_lesson_answers()

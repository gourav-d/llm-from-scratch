"""
Example 4: Process Supervision & Reasoning Traces

This example demonstrates how process supervision works and how to build
Process Reward Models (PRMs) that evaluate reasoning step-by-step.

For .NET developers:
- Process supervision = Unit testing each method in a class
- Outcome supervision = Only testing the final return value
- PRM = Automated code reviewer that checks each line

Author: Learn LLM from Scratch
Module: 7 - Reasoning & Coding Models
Lesson: 4 - Process Supervision
"""

import numpy as np
import random
from typing import List, Dict, Tuple
from collections import defaultdict


# ============================================================================
# PART 1: Understanding the Problem
# ============================================================================

def example_01_outcome_vs_process():
    """
    Example 1: Show the difference between outcome and process supervision.

    This is like the difference between:
    - Multiple choice test (outcome) vs Show-your-work test (process)
    """
    print("=" * 70)
    print("EXAMPLE 1: Outcome vs Process Supervision")
    print("=" * 70)

    question = "What is 8 + 7?"
    correct_answer = "15"

    # Three different student responses
    responses = [
        {
            'name': 'Student A',
            'reasoning': ['8 + 7 = 15'],
            'answer': '15'
        },
        {
            'name': 'Student B',
            'reasoning': [
                'I need to add 8 + 7',
                '8 + 7 = 8 + (2 + 5)',
                '= (8 + 2) + 5',
                '= 10 + 5',
                '= 15'
            ],
            'answer': '15'
        },
        {
            'name': 'Student C (cheating!)',
            'reasoning': [
                'I know 8 + 7 = 20 - 5',  # WRONG reasoning!
                'So the answer is 15'
            ],
            'answer': '15'
        }
    ]

    print(f"\nQuestion: {question}")
    print(f"Correct Answer: {correct_answer}\n")

    # Outcome Supervision: Only check final answer
    print("-" * 70)
    print("OUTCOME SUPERVISION (Traditional)")
    print("-" * 70)

    for response in responses:
        is_correct = response['answer'] == correct_answer
        reward = 1.0 if is_correct else 0.0

        print(f"\n{response['name']}:")
        print(f"  Answer: {response['answer']}")
        print(f"  Reward: {reward} {'✓' if is_correct else '✗'}")

    print("\n⚠️  PROBLEM: All three get the same reward!")
    print("   Student C used wrong reasoning but got lucky!")

    # Process Supervision: Check each step
    print("\n" + "-" * 70)
    print("PROCESS SUPERVISION (o1-style)")
    print("-" * 70)

    def score_step(step):
        """Simple step scoring (in reality, use trained PRM)"""
        # Check for wrong reasoning patterns
        if '8 + 7 = 20' in step or '8 + 7 = 16' in step:
            return 0.1  # Wrong calculation

        # Check for good reasoning patterns
        good_patterns = [
            'I need to',
            'break down',
            '8 + 7 = 15',
            '10 + 5 = 15',
            'Therefore'
        ]

        if any(pattern in step for pattern in good_patterns):
            return 0.9  # Good reasoning

        return 0.6  # Neutral

    for response in responses:
        print(f"\n{response['name']}:")
        print(f"  Steps:")

        step_scores = []
        for i, step in enumerate(response['reasoning'], 1):
            score = score_step(step)
            step_scores.append(score)
            status = '✓' if score >= 0.7 else '✗'
            print(f"    {i}. {step:50s} {status} (score: {score:.2f})")

        avg_score = np.mean(step_scores)
        print(f"  Overall Quality: {avg_score:.2f}")

    print("\n✅ BENEFIT: Student C now gets lower score!")
    print("   Process supervision catches wrong reasoning!")
    print()


# ============================================================================
# PART 2: Building a Process Reward Model
# ============================================================================

class ProcessRewardModel:
    """
    A Process Reward Model that scores individual reasoning steps.

    In C# terms: This is like StyleCop or ReSharper analyzing
    each line of code and giving quality scores.
    """

    def __init__(self):
        """Initialize the PRM with scoring rules."""
        # In a real o1, this would be a trained neural network
        # For learning, we use rule-based scoring

        self.math_operations = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '×': lambda a, b: a * b,
            '/': lambda a, b: a / b if b != 0 else None,
            '÷': lambda a, b: a / b if b != 0 else None
        }

    def score_step(
        self,
        question: str,
        previous_steps: List[str],
        current_step: str,
        step_type: str = 'math'
    ) -> float:
        """
        Score a single reasoning step.

        Args:
            question: The original problem
            previous_steps: Previous reasoning steps
            current_step: Current step to score
            step_type: 'math', 'logic', or 'general'

        Returns:
            Score between 0.0 (wrong) and 1.0 (perfect)
        """
        if step_type == 'math':
            return self._score_math_step(question, previous_steps, current_step)
        elif step_type == 'logic':
            return self._score_logic_step(question, previous_steps, current_step)
        else:
            return self._score_general_step(question, previous_steps, current_step)

    def _score_math_step(
        self,
        question: str,
        previous_steps: List[str],
        current_step: str
    ) -> float:
        """Score a mathematical reasoning step."""
        import re

        # Pattern 1: Equation (e.g., "8 + 7 = 15")
        eq_pattern = r'(\d+\.?\d*)\s*([\+\-\*/×÷])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)'
        match = re.search(eq_pattern, current_step)

        if match:
            left = float(match.group(1))
            op = match.group(2)
            right = float(match.group(3))
            claimed_result = float(match.group(4))

            # Calculate actual result
            if op in self.math_operations:
                actual_result = self.math_operations[op](left, right)

                if actual_result is not None:
                    # Check if calculation is correct
                    if abs(actual_result - claimed_result) < 0.001:
                        return 0.95  # Correct calculation
                    else:
                        return 0.10  # Wrong calculation

        # Pattern 2: Good reasoning phrases
        good_phrases = [
            'let me think',
            'i need to',
            'first',
            'next',
            'then',
            'therefore',
            'so',
            'this means'
        ]

        if any(phrase in current_step.lower() for phrase in good_phrases):
            return 0.75  # Good structure

        # Pattern 3: Stating the problem
        if 'what is' in current_step.lower() or 'calculate' in current_step.lower():
            return 0.70  # Understanding the problem

        return 0.60  # Neutral (not wrong, but not particularly good)

    def _score_logic_step(
        self,
        question: str,
        previous_steps: List[str],
        current_step: str
    ) -> float:
        """Score a logical reasoning step."""
        # Check for logical fallacies
        fallacies = [
            'everyone knows',
            'obviously',
            'it must be true',
            'all .* always',
            'never'
        ]

        import re
        if any(re.search(pattern, current_step.lower()) for pattern in fallacies):
            return 0.30  # Likely flawed logic

        # Check for good logical connectives
        good_logic = [
            'if .* then',
            'because',
            'since',
            'it follows that',
            'we can conclude',
            'given that',
            'assuming',
            'premise'
        ]

        if any(re.search(pattern, current_step.lower()) for pattern in good_logic):
            return 0.85  # Good logical structure

        return 0.60  # Neutral

    def _score_general_step(
        self,
        question: str,
        previous_steps: List[str],
        current_step: str
    ) -> float:
        """Score a general reasoning step."""
        # Check for empty or too short
        if len(current_step.strip()) < 5:
            return 0.30

        # Check for coherence with previous steps
        if previous_steps:
            # Simple coherence check
            transition_words = ['therefore', 'so', 'thus', 'next', 'then']
            if any(word in current_step.lower() for word in transition_words):
                return 0.75  # Builds on previous reasoning

        return 0.60  # Neutral


def example_02_process_reward_model():
    """
    Example 2: Using a Process Reward Model to score reasoning steps.
    """
    print("=" * 70)
    print("EXAMPLE 2: Process Reward Model in Action")
    print("=" * 70)

    prm = ProcessRewardModel()

    question = "What is 12 + 8?"

    # Good reasoning
    good_steps = [
        "I need to add 12 + 8",
        "12 + 8 = 20",
        "Therefore, the answer is 20"
    ]

    # Bad reasoning
    bad_steps = [
        "I need to add 12 + 8",
        "12 + 8 = 25",  # WRONG!
        "Therefore, the answer is 25"
    ]

    print(f"\nQuestion: {question}\n")

    # Score good reasoning
    print("Good Reasoning:")
    print("-" * 70)
    total_good = 0
    for i, step in enumerate(good_steps):
        score = prm.score_step(question, good_steps[:i], step, step_type='math')
        total_good += score
        print(f"Step {i+1}: {step}")
        print(f"  Score: {score:.2f} {'✓' if score >= 0.7 else '✗'}\n")

    avg_good = total_good / len(good_steps)
    print(f"Average Quality: {avg_good:.2f}\n")

    # Score bad reasoning
    print("Bad Reasoning:")
    print("-" * 70)
    total_bad = 0
    for i, step in enumerate(bad_steps):
        score = prm.score_step(question, bad_steps[:i], step, step_type='math')
        total_bad += score
        print(f"Step {i+1}: {step}")
        print(f"  Score: {score:.2f} {'✓' if score >= 0.7 else '✗'}\n")

    avg_bad = total_bad / len(bad_steps)
    print(f"Average Quality: {avg_bad:.2f}\n")

    print(f"Quality Difference: {avg_good - avg_bad:.2f}")
    print("✅ PRM successfully identifies better reasoning!\n")


# ============================================================================
# PART 3: Creating Training Data with Reasoning Traces
# ============================================================================

def create_reasoning_trace(
    question: str,
    steps: List[str],
    step_labels: List[bool]
) -> Dict:
    """
    Create a training example with labeled reasoning steps.

    Args:
        question: The problem to solve
        steps: List of reasoning steps
        step_labels: List of True/False for each step

    Returns:
        Dictionary in format for training
    """
    return {
        'question': question,
        'reasoning_trace': [
            {
                'step_number': i + 1,
                'text': step,
                'correct': label
            }
            for i, (step, label) in enumerate(zip(steps, step_labels))
        ],
        'final_answer': steps[-1] if steps else None
    }


def example_03_training_data():
    """
    Example 3: Creating training data with reasoning traces.

    This is like creating unit test cases where you mark
    each assertion as passing or failing.
    """
    print("=" * 70)
    print("EXAMPLE 3: Creating Training Data with Reasoning Traces")
    print("=" * 70)

    # Example 1: Math problem with all correct steps
    trace1 = create_reasoning_trace(
        question="What is 15% of 80?",
        steps=[
            "15% means 15/100",           # ✓
            "So I calculate (15/100) × 80", # ✓
            "15 × 80 = 1200",             # ✓
            "1200 / 100 = 12",            # ✓
            "Therefore, 15% of 80 is 12"  # ✓
        ],
        step_labels=[True, True, True, True, True]
    )

    # Example 2: Math problem with an error
    trace2 = create_reasoning_trace(
        question="What is 20% of 50?",
        steps=[
            "20% means 20/100",           # ✓
            "So I calculate (20/100) × 50", # ✓
            "20 × 50 = 1500",             # ✗ WRONG! Should be 1000
            "1500 / 100 = 15",            # ✗ Wrong because previous was wrong
            "Therefore, 20% of 50 is 15"  # ✗ Wrong final answer
        ],
        step_labels=[True, True, False, False, False]
    )

    # Example 3: Logic problem
    trace3 = create_reasoning_trace(
        question="If all birds can fly, and penguins are birds, can penguins fly?",
        steps=[
            "The premise states 'all birds can fly'",  # ✓
            "The premise states 'penguins are birds'", # ✓
            "However, this creates a logical issue",   # ✓
            "In reality, penguins cannot fly",         # ✓
            "The first premise is false",              # ✓
            "Answer: No, penguins cannot fly"          # ✓
        ],
        step_labels=[True, True, True, True, True, True]
    )

    dataset = [trace1, trace2, trace3]

    print("\nTraining Dataset with Reasoning Traces:\n")

    for i, trace in enumerate(dataset, 1):
        print(f"Example {i}:")
        print(f"  Question: {trace['question']}")
        print(f"  Reasoning:")

        for step_data in trace['reasoning_trace']:
            status = '✓' if step_data['correct'] else '✗'
            print(f"    {step_data['step_number']}. {step_data['text']:50s} {status}")

        correct_count = sum(1 for s in trace['reasoning_trace'] if s['correct'])
        total_count = len(trace['reasoning_trace'])
        print(f"  Quality: {correct_count}/{total_count} steps correct")
        print()

    print("💡 This data can be used to:")
    print("   1. Train a Process Reward Model (PRM)")
    print("   2. Fine-tune an LLM with RL using step-by-step rewards")
    print("   3. Evaluate reasoning quality")
    print()


# ============================================================================
# PART 4: Training with Process Supervision
# ============================================================================

class MockLLM:
    """
    Simple mock LLM for demonstration.
    In reality, use your GPT model from Module 6.
    """

    def __init__(self):
        self.quality = 0.5  # Start with mediocre quality

    def generate_reasoning(self, question: str) -> List[str]:
        """Generate reasoning steps for a question."""
        # In reality, this would use your actual LLM
        # For demo, return somewhat random reasoning

        templates = [
            f"I need to solve: {question}",
            "Let me break this down step by step",
            "First, I'll identify the key information",
            "Next, I'll perform the calculation",
            "Therefore, the answer is..."
        ]

        # Quality affects how many good steps we generate
        num_steps = random.randint(3, 5)
        steps = random.sample(templates, min(num_steps, len(templates)))

        return steps

    def update_with_reward(self, question: str, steps: List[str], reward: float):
        """Update model based on reward."""
        # In reality, this would use RL (PPO, etc.)
        # For demo, just adjust quality parameter
        learning_rate = 0.1
        self.quality += learning_rate * (reward - self.quality)
        self.quality = max(0.0, min(1.0, self.quality))


def example_04_training_loop():
    """
    Example 4: Training with process supervision.

    Shows how models learn from step-by-step feedback.
    """
    print("=" * 70)
    print("EXAMPLE 4: Training with Process Supervision")
    print("=" * 70)

    # Initialize
    model = MockLLM()
    prm = ProcessRewardModel()

    # Training data
    training_examples = [
        {
            'question': 'What is 10 + 5?',
            'correct_steps': [
                'I need to add 10 + 5',
                '10 + 5 = 15',
                'Therefore, the answer is 15'
            ]
        },
        {
            'question': 'What is 20 - 8?',
            'correct_steps': [
                'I need to subtract 8 from 20',
                '20 - 8 = 12',
                'Therefore, the answer is 12'
            ]
        }
    ]

    print("\nTraining for 5 epochs...\n")

    for epoch in range(5):
        total_reward = 0

        for example in training_examples:
            question = example['question']

            # Generate reasoning
            generated_steps = model.generate_reasoning(question)

            # Score each step
            step_scores = []
            for i, step in enumerate(generated_steps):
                score = prm.score_step(
                    question,
                    generated_steps[:i],
                    step,
                    step_type='math'
                )
                step_scores.append(score)

            # Calculate average reward
            avg_reward = np.mean(step_scores) if step_scores else 0.0
            total_reward += avg_reward

            # Update model
            model.update_with_reward(question, generated_steps, avg_reward)

        avg_reward = total_reward / len(training_examples)
        print(f"Epoch {epoch + 1}: Average Reward = {avg_reward:.3f}, "
              f"Model Quality = {model.quality:.3f}")

    print("\n✅ Model improved through process supervision!")
    print("   (In reality, this would be much more complex)")
    print()


# ============================================================================
# PART 5: Comparison - Outcome vs Process Supervision
# ============================================================================

def example_05_comparison():
    """
    Example 5: Direct comparison of outcome vs process supervision.
    """
    print("=" * 70)
    print("EXAMPLE 5: Outcome vs Process Supervision Comparison")
    print("=" * 70)

    test_cases = [
        {
            'question': 'What is 6 + 9?',
            'reasoning': [
                'I need to add 6 + 9',
                '6 + 9 = 15'
            ],
            'correct_answer': '15'
        },
        {
            'question': 'What is 6 + 9?',
            'reasoning': [
                '6 + 9 = 20 - 5',  # Wrong reasoning!
                'So the answer is 15'
            ],
            'correct_answer': '15'
        },
        {
            'question': 'What is 6 + 9?',
            'reasoning': [
                'I need to add 6 + 9',
                '6 + 9 = 14'  # Wrong answer!
            ],
            'correct_answer': '15'
        }
    ]

    prm = ProcessRewardModel()

    print("\nEvaluating 3 responses to the same question:\n")

    for i, test in enumerate(test_cases, 1):
        print(f"Response {i}:")
        print(f"  Question: {test['question']}")
        print(f"  Reasoning:")

        for step in test['reasoning']:
            print(f"    - {step}")

        # Outcome supervision
        final_answer = test['reasoning'][-1].split()[-1]
        outcome_correct = final_answer == test['correct_answer']
        outcome_reward = 1.0 if outcome_correct else 0.0

        # Process supervision
        step_scores = []
        for j, step in enumerate(test['reasoning']):
            score = prm.score_step(
                test['question'],
                test['reasoning'][:j],
                step,
                step_type='math'
            )
            step_scores.append(score)

        process_reward = np.mean(step_scores)

        print(f"\n  Outcome Supervision Reward: {outcome_reward:.2f}")
        print(f"  Process Supervision Reward: {process_reward:.2f}")
        print(f"  Difference: {abs(outcome_reward - process_reward):.2f}")
        print()

    print("📊 Analysis:")
    print("   Response 1: Both methods agree (good reasoning, correct answer)")
    print("   Response 2: Outcome says ✓, Process says ✗ (caught wrong reasoning!)")
    print("   Response 3: Both methods agree (poor reasoning, wrong answer)")
    print("\n✅ Process supervision provides more nuanced feedback!")
    print()


# ============================================================================
# PART 6: Real-World Application
# ============================================================================

def example_06_real_world_scenario():
    """
    Example 6: Real-world scenario - tutoring system.

    Shows how process supervision helps build better educational AI.
    """
    print("=" * 70)
    print("EXAMPLE 6: Real-World Application - Math Tutoring System")
    print("=" * 70)

    class MathTutor:
        """AI tutor that uses process supervision to help students."""

        def __init__(self):
            self.prm = ProcessRewardModel()

        def evaluate_student_work(
            self,
            question: str,
            student_steps: List[str]
        ) -> Dict:
            """Evaluate student's work step-by-step."""
            feedback = []
            step_scores = []

            for i, step in enumerate(student_steps):
                score = self.prm.score_step(
                    question,
                    student_steps[:i],
                    step,
                    step_type='math'
                )
                step_scores.append(score)

                # Generate feedback
                if score >= 0.8:
                    feedback.append(f"✓ Step {i+1}: Excellent!")
                elif score >= 0.6:
                    feedback.append(f"~ Step {i+1}: Okay, but could be clearer")
                else:
                    feedback.append(f"✗ Step {i+1}: Check this step - something's not right")

            return {
                'step_scores': step_scores,
                'average_score': np.mean(step_scores),
                'feedback': feedback,
                'needs_help': any(score < 0.6 for score in step_scores)
            }

    # Create tutor
    tutor = MathTutor()

    # Student submission
    question = "What is 25% of 60?"

    student_work = [
        "25% means 25/100",
        "So I need (25/100) × 60",
        "25 × 60 = 1500",
        "1500 / 100 = 15",
        "Answer: 15"
    ]

    print(f"\nQuestion: {question}\n")
    print("Student's Work:")
    for i, step in enumerate(student_work, 1):
        print(f"  {i}. {step}")

    print("\n" + "-" * 70)
    print("Tutor's Evaluation:")
    print("-" * 70)

    result = tutor.evaluate_student_work(question, student_work)

    for feedback_line in result['feedback']:
        print(f"  {feedback_line}")

    print(f"\nOverall Score: {result['average_score']:.0%}")

    if result['needs_help']:
        print("⚠️  Recommendation: Review with teacher")
    else:
        print("✅ Great work! Understanding demonstrated")

    print("\n💡 Benefits of Process Supervision in Education:")
    print("   - Identifies exactly where student struggles")
    print("   - Gives specific, actionable feedback")
    print("   - Rewards correct reasoning even with calc errors")
    print("   - Helps students learn the process, not just memorize")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print("PROCESS SUPERVISION & REASONING TRACES")
    print("Module 7, Lesson 4 - Examples")
    print("=" * 70)
    print("\nThis demonstrates the key technique behind OpenAI o1's reliability!\n")

    # Run examples
    example_01_outcome_vs_process()
    input("Press Enter to continue to Example 2...")

    example_02_process_reward_model()
    input("Press Enter to continue to Example 3...")

    example_03_training_data()
    input("Press Enter to continue to Example 4...")

    example_04_training_loop()
    input("Press Enter to continue to Example 5...")

    example_05_comparison()
    input("Press Enter to continue to Example 6...")

    example_06_real_world_scenario()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Takeaways:

1. Outcome Supervision = Only check final answer
   - Simple but can reward wrong reasoning

2. Process Supervision = Check every step
   - More complex but learns correct reasoning

3. Process Reward Models (PRMs) score individual steps
   - Like automated code reviewer
   - Can be rule-based or trained neural networks

4. Training with process supervision requires:
   - Human-annotated reasoning traces
   - Step-by-step scoring
   - More expensive but more reliable

5. Real-world applications:
   - Educational AI (tutoring systems)
   - Medical diagnosis (verify reasoning)
   - Legal analysis (check logic)
   - Any high-stakes reasoning task

You now understand how o1 achieves reliable reasoning! 🎉
    """)


if __name__ == "__main__":
    main()

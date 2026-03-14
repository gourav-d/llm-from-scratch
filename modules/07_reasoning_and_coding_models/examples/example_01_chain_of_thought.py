"""
Example 1: Chain-of-Thought (CoT) Prompting

This example demonstrates how to implement Chain-of-Thought prompting
to improve reasoning on complex tasks.

Concepts demonstrated:
- Few-shot CoT prompting
- Zero-shot CoT prompting
- Comparing with vs. without CoT
- Math reasoning with step-by-step thinking

Author: Your Name
Module: 7.1 - Chain-of-Thought
"""

import numpy as np
from typing import List, Dict, Tuple


# ============================================================================
# SECTION 1: Simple CoT Implementation
# ============================================================================

class ChainOfThoughtPrompter:
    """
    Implements Chain-of-Thought prompting for better reasoning.

    This class wraps a base language model (your GPT from Module 6)
    and adds reasoning capabilities through prompting.

    Think of this like a C# decorator pattern:
    ```csharp
    [AddReasoning]
    public class EnhancedGPT : GPT {
        // Adds reasoning before generating
    }
    ```
    """

    def __init__(self, base_model):
        """
        Initialize CoT prompter.

        Args:
            base_model: Your GPT model from Module 6
        """
        self.base_model = base_model
        self.few_shot_examples = []

    def add_example(self, question: str, reasoning_steps: List[str], answer: str):
        """
        Add a few-shot example showing step-by-step reasoning.

        This is like providing XML doc examples in C#:
        /// <example>
        /// Step 1: ...
        /// Step 2: ...
        /// </example>

        Args:
            question: The example question
            reasoning_steps: List of reasoning steps
            answer: Final answer
        """
        example = {
            'question': question,
            'reasoning': reasoning_steps,
            'answer': answer
        }
        self.few_shot_examples.append(example)
        print(f"✓ Added example: {question[:50]}...")

    def format_few_shot_prompt(self, question: str) -> str:
        """
        Create a prompt with few-shot examples.

        Like C# string interpolation:
        var prompt = $"Example 1:\nQ: {q1}\nA: {a1}\n\nYour question:\nQ: {question}";

        Args:
            question: The question to answer

        Returns:
            Formatted prompt with examples
        """
        parts = []

        # Add each example
        for i, example in enumerate(self.few_shot_examples, 1):
            parts.append(f"Example {i}:")
            parts.append(f"Q: {example['question']}")
            parts.append("A: Let me think step by step:")

            # Add reasoning steps
            for step_num, step in enumerate(example['reasoning'], 1):
                parts.append(f"   {step_num}. {step}")

            # Add final answer
            parts.append(f"   Therefore: {example['answer']}")
            parts.append("")  # Blank line

        # Add the actual question
        parts.append("Now your turn:")
        parts.append(f"Q: {question}")
        parts.append("A: Let me think step by step:")

        return "\n".join(parts)

    def generate_with_cot(self, question: str, max_length: int = 200) -> Dict:
        """
        Generate answer using Chain-of-Thought reasoning.

        Args:
            question: Question to answer
            max_length: Maximum tokens to generate

        Returns:
            Dictionary with reasoning and answer
        """
        # Create prompt with examples
        prompt = self.format_few_shot_prompt(question)

        # Generate response
        # NOTE: Replace this with your actual GPT model from Module 6
        response = self._mock_generate(prompt, max_length)

        # Parse reasoning steps and answer
        steps, answer = self._parse_response(response)

        return {
            'question': question,
            'reasoning_steps': steps,
            'answer': answer,
            'full_response': response
        }

    def _parse_response(self, response: str) -> Tuple[List[str], str]:
        """
        Parse the response to extract reasoning steps and final answer.

        Like parsing structured text in C#:
        var lines = response.Split('\n');
        var steps = lines.Where(l => l.Contains("Step")).ToList();
        """
        lines = response.strip().split('\n')
        steps = []
        answer = None

        for line in lines:
            line = line.strip()

            # Look for numbered steps
            if line and (line[0].isdigit() or 'Step' in line):
                steps.append(line)

            # Look for final answer
            if 'therefore' in line.lower() or 'answer:' in line.lower():
                answer = line

        return steps, answer

    def _mock_generate(self, prompt: str, max_length: int) -> str:
        """
        Mock generation for demonstration.
        In real usage, replace with: self.base_model.generate(prompt)
        """
        # This is a placeholder - your GPT will generate real responses
        return """
        1. Identify what we know
        2. Set up the equation
        3. Solve step by step
        4. Verify the answer
        Therefore: The answer is correct.
        """


# ============================================================================
# SECTION 2: Zero-Shot CoT (Simpler!)
# ============================================================================

class ZeroShotCoT:
    """
    Zero-shot Chain-of-Thought prompting.

    No examples needed - just add the magic phrase!

    Like a C# extension method:
    public static string ThinkStepByStep(this GPT model, string question) {
        return model.Generate(question + "\nLet's think step by step:");
    }
    """

    def __init__(self, base_model):
        self.base_model = base_model
        # The magic trigger phrase!
        self.cot_trigger = "Let's think step by step:"

    def generate_with_thinking(self, question: str) -> Dict:
        """
        Generate answer with automatic step-by-step thinking.

        Just adds "Let's think step by step" and the model reasons!

        Args:
            question: Question to answer

        Returns:
            Dictionary with reasoning and answer
        """
        # Add magic phrase
        prompt = f"{question}\n{self.cot_trigger}"

        # Generate (mock for now)
        response = self._mock_generate(prompt)

        # Parse
        steps, answer = self._extract_reasoning(response)

        return {
            'question': question,
            'reasoning_steps': steps,
            'answer': answer,
            'method': 'zero-shot-cot'
        }

    def _extract_reasoning(self, response: str) -> Tuple[List[str], str]:
        """Extract reasoning steps from response."""
        lines = response.strip().split('\n')
        steps = [line.strip() for line in lines if line.strip()]
        answer = steps[-1] if steps else "No answer"
        return steps[:-1], answer

    def _mock_generate(self, prompt: str) -> str:
        """Mock generation - replace with your GPT."""
        return """
        First, let's understand what we're looking for.
        Next, we'll break down the problem.
        Then, we'll solve each part.
        Finally, we'll combine for the answer.
        The final answer is 42.
        """


# ============================================================================
# SECTION 3: Math Reasoning System
# ============================================================================

class MathReasoningSystem:
    """
    Complete math reasoning system using CoT.

    Solves math problems with step-by-step explanations.
    """

    def __init__(self, base_model):
        self.base_model = base_model
        self.cot = ChainOfThoughtPrompter(base_model)

        # Add math-specific examples
        self._add_math_examples()

    def _add_math_examples(self):
        """Add few-shot examples for math problems."""

        # Example 1: Simple arithmetic
        self.cot.add_example(
            question="What is 25 + 37?",
            reasoning_steps=[
                "Break into tens and ones",
                "Tens: 20 + 30 = 50",
                "Ones: 5 + 7 = 12",
                "Total: 50 + 12 = 62"
            ],
            answer="62"
        )

        # Example 2: Word problem
        self.cot.add_example(
            question="Sarah has 12 apples. She gives away 1/3 of them. How many does she have left?",
            reasoning_steps=[
                "Sarah starts with 12 apples",
                "She gives away 1/3 of 12",
                "1/3 of 12 = 12 ÷ 3 = 4 apples",
                "Remaining: 12 - 4 = 8 apples"
            ],
            answer="8 apples"
        )

        # Example 3: Multi-step
        self.cot.add_example(
            question="A book costs $20. It's on sale for 25% off. Then there's 10% tax. What's the final price?",
            reasoning_steps=[
                "Original price: $20",
                "25% discount: 0.25 × $20 = $5",
                "Sale price: $20 - $5 = $15",
                "10% tax: 0.10 × $15 = $1.50",
                "Final price: $15 + $1.50 = $16.50"
            ],
            answer="$16.50"
        )

    def solve(self, problem: str) -> Dict:
        """
        Solve a math problem with reasoning.

        Args:
            problem: Math problem in natural language

        Returns:
            Solution with reasoning steps
        """
        result = self.cot.generate_with_cot(problem)

        # Add verification
        result['verified'] = self._verify_math_reasoning(result['reasoning_steps'])

        return result

    def _verify_math_reasoning(self, steps: List[str]) -> bool:
        """
        Basic verification of math reasoning.

        Checks:
        - Multiple steps (not just guessing)
        - Contains math keywords
        - Has numbers and operations
        """
        if len(steps) < 2:
            return False

        # Check for math keywords
        math_keywords = ['add', 'subtract', 'multiply', 'divide', '=', '+', '-', '×', '÷', 'calculate']
        has_math = any(
            any(kw in step.lower() for kw in math_keywords)
            for step in steps
        )

        return has_math


# ============================================================================
# SECTION 4: Demonstration
# ============================================================================

def demonstrate_cot():
    """
    Demonstrate Chain-of-Thought prompting.

    This function shows how to use CoT in practice.
    """
    print("=" * 60)
    print("Chain-of-Thought Prompting Demo")
    print("=" * 60)
    print()

    # Mock model (replace with your GPT from Module 6)
    mock_model = None

    # -------------------------------------------------------------------------
    # Demo 1: Zero-Shot CoT
    # -------------------------------------------------------------------------
    print("DEMO 1: Zero-Shot CoT (No Examples)")
    print("-" * 60)

    zero_shot = ZeroShotCoT(mock_model)
    question = "If a train travels 60 mph for 2.5 hours, how far does it go?"

    result = zero_shot.generate_with_thinking(question)

    print(f"Question: {result['question']}")
    print("\nReasoning:")
    for i, step in enumerate(result['reasoning_steps'], 1):
        print(f"  {i}. {step}")
    print(f"\nAnswer: {result['answer']}")
    print()

    # -------------------------------------------------------------------------
    # Demo 2: Few-Shot CoT
    # -------------------------------------------------------------------------
    print("\nDEMO 2: Few-Shot CoT (With Examples)")
    print("-" * 60)

    math_solver = MathReasoningSystem(mock_model)
    problem = "A pizza has 8 slices. If you eat 3/4 of the pizza, how many slices did you eat?"

    result = math_solver.solve(problem)

    print(f"Problem: {result['question']}")
    print("\nReasoning Steps:")
    for i, step in enumerate(result['reasoning_steps'], 1):
        print(f"  {i}. {step}")
    print(f"\nAnswer: {result['answer']}")
    print(f"Verified: {'✓ Yes' if result['verified'] else '✗ No'}")
    print()

    # -------------------------------------------------------------------------
    # Demo 3: Comparison
    # -------------------------------------------------------------------------
    print("\nDEMO 3: With vs. Without CoT")
    print("-" * 60)

    print("Without CoT:")
    print("  Question: What is 23 × 47?")
    print("  Answer: 1,081")
    print("  (Might be wrong, no way to verify)")
    print()

    print("With CoT:")
    print("  Question: What is 23 × 47?")
    print("  Reasoning:")
    print("    1. Break down: 23 × 47 = 23 × (40 + 7)")
    print("    2. Calculate: 23 × 40 = 920")
    print("    3. Calculate: 23 × 7 = 161")
    print("    4. Add: 920 + 161 = 1,081")
    print("  Answer: 1,081 ✓ (Verified through steps!)")
    print()

    print("=" * 60)
    print("Key Takeaway: CoT improves accuracy and provides verification!")
    print("=" * 60)


# ============================================================================
# SECTION 5: Practical Tips
# ============================================================================

def practical_tips():
    """
    Tips for using Chain-of-Thought in real applications.
    """
    print("\n" + "=" * 60)
    print("Practical Tips for Using CoT")
    print("=" * 60)
    print()

    tips = [
        ("When to use CoT", [
            "✓ Math problems",
            "✓ Logic puzzles",
            "✓ Multi-step reasoning",
            "✓ Tasks requiring verification",
            "✓ Complex decision-making"
        ]),

        ("When NOT to use CoT", [
            "✗ Simple factual questions ('What's the capital of France?')",
            "✗ Creative writing (reasoning not needed)",
            "✗ Time-critical applications (adds latency)",
            "✗ Token-limited scenarios (uses more tokens)"
        ]),

        ("Best Practices", [
            "• Start with zero-shot, add examples if needed",
            "• Verify reasoning steps programmatically",
            "• Use temperature 0.7-0.8 for balanced creativity",
            "• Parse outputs carefully to extract steps",
            "• Combine with self-consistency for critical tasks"
        ]),

        ("C#/.NET Analogies", [
            "CoT = Debug logging / verbose mode",
            "Few-shot examples = XML documentation examples",
            "Reasoning steps = Stack trace / call hierarchy",
            "Verification = Unit test assertions"
        ])
    ]

    for title, items in tips:
        print(f"{title}:")
        for item in items:
            print(f"  {item}")
        print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    """
    Run demonstrations and examples.

    To use with your actual GPT from Module 6:
    1. Import your GPT model
    2. Load trained weights
    3. Replace mock_model with your GPT instance
    4. Run this file
    """

    print("\n" + "🧠 " * 20)
    print("Module 7.1: Chain-of-Thought Prompting")
    print("🧠 " * 20 + "\n")

    # Run demonstrations
    demonstrate_cot()

    # Show practical tips
    practical_tips()

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("  1. Integrate with your GPT from Module 6")
    print("  2. Try different types of problems")
    print("  3. Experiment with prompting techniques")
    print("  4. Move to Lesson 2: Self-Consistency")
    print("=" * 60)
    print()

    # Example integration with your GPT
    print("\nTo integrate with your GPT model:")
    print("-" * 60)
    print("""
# Import your GPT
import sys
sys.path.append('../06_training_finetuning')
from example_01_complete_gpt import GPT, GPTConfig

# Load model
config = GPTConfig(vocab_size=50257, max_seq_len=256, embed_dim=512, n_layers=6, n_heads=8)
gpt = GPT(config)
# gpt.load_weights('path/to/weights.pth')

# Use CoT
math_solver = MathReasoningSystem(gpt)
result = math_solver.solve("Your math problem here")
print(result)
    """)

# Lesson 7.1: Chain-of-Thought (CoT) Prompting

## 🎯 Learning Objectives

By the end of this lesson, you'll be able to:

- Explain why regular LLMs fail at complex reasoning
- Understand Chain-of-Thought (CoT) prompting
- Implement few-shot CoT prompting
- Implement zero-shot CoT ("Let's think step by step")
- Build a reasoning system that shows its work
- Evaluate reasoning quality
- Connect CoT to how OpenAI o1 works

---

## 🤔 What is Chain-of-Thought?

### The Problem: LLMs Can't Reason (Without Help)

**Real example - GPT-3 failing:**

```
Question: "Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?"

GPT-3 (without CoT):
"Roger has 11 tennis balls." ✗ WRONG!
```

**Why did it fail?**
- Didn't break down the problem
- Didn't show intermediate steps
- Just guessed the final answer

---

### The Solution: Make the Model "Show Its Work"

**Same question with Chain-of-Thought:**

```
Question: "Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?"

GPT-3 (with CoT):
"Let me think step by step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans of tennis balls
3. Each can has 3 tennis balls
4. So he gets: 2 cans × 3 balls = 6 balls
5. Total balls = 5 (original) + 6 (new) = 11 balls

Answer: Roger has 11 tennis balls." ✓ CORRECT!
```

**Why it worked:**
- Broke problem into steps
- Showed intermediate calculations
- Verified the logic
- Got the right answer!

---

## 🌍 Real-World Analogy

Think of Chain-of-Thought like this:

### Without CoT: Guessing the Answer
```
You're taking a math test and just write down answers:

Problem 1: "23 × 47 = ?"
Your answer: "1,081"

Teacher: "How did you get this?"
You: "I just knew it."

Grade: ✗ (Even if correct, no partial credit if wrong!)
```

### With CoT: Showing Your Work
```
You're taking a math test and show all steps:

Problem 1: "23 × 47 = ?"
Your work:
  Step 1: Break down 47 = 40 + 7
  Step 2: 23 × 40 = 920
  Step 3: 23 × 7 = 161
  Step 4: Total = 920 + 161 = 1,081

Grade: ✓ (Full credit! Plus partial credit if you make a mistake)
```

**Chain-of-Thought is teaching the AI to "show its work"!**

---

## 📚 How Chain-of-Thought Works

### The Breakthrough Discovery (2022)

Researchers discovered that adding "Let's think step by step" dramatically improves accuracy:

**Before (2021):**
```
Q: Complex math problem
A: [Wrong answer]
Accuracy: 17%
```

**After (2022 - with CoT):**
```
Q: Complex math problem
A: Let's think step by step:
   Step 1: ...
   Step 2: ...
   Step 3: ...
   Final answer: [Correct!]
Accuracy: 78%
```

**That's 4.5x better with just a few words!**

---

### Two Types of Chain-of-Thought

#### 1. Few-Shot CoT (Show Examples)

**You provide examples of step-by-step reasoning:**

```
# Example 1 (you provide):
Q: "If you have 3 apples and buy 2 more, how many apples do you have?"
A: "Let me think:
    - Start with 3 apples
    - Buy 2 more apples
    - Total = 3 + 2 = 5 apples"

# Example 2 (you provide):
Q: "A shirt costs $20. There's a 10% discount. What's the final price?"
A: "Let me think:
    - Original price: $20
    - Discount: 10% of $20 = $2
    - Final price: $20 - $2 = $18"

# Now your actual question:
Q: "If a book costs $15 and there's a 20% discount, what do you pay?"
A: "Let me think:  ← Model learned to reason!
    - Original price: $15
    - Discount: 20% of $15 = $3
    - Final price: $15 - $3 = $12"
```

**Analogy to C#/.NET:**
Think of this like providing XML documentation examples:
```csharp
/// <example>
/// <code>
/// var calculator = new Calculator();
/// int result = calculator.Add(2, 3);  // Returns 5
/// </code>
/// </example>
```

The examples teach the model what format to follow!

---

#### 2. Zero-Shot CoT (Just Ask to Think)

**You don't provide examples, just add "Let's think step by step":**

```
Q: "If a train travels 60 miles in 1 hour, how far will it travel in 3.5 hours?"
Let's think step by step:

A: "Okay, let's break this down:
    - Speed = 60 miles per hour
    - Time = 3.5 hours
    - Distance = Speed × Time
    - Distance = 60 × 3.5 = 210 miles

    Answer: 210 miles"
```

**Amazingly, this simple phrase triggers reasoning!**

---

## 💻 Implementation in Python

### Part 1: Building Few-Shot CoT

Let's implement this step by step. Remember, we're enhancing your GPT from Module 6!

```python
import numpy as np
from typing import List, Tuple

class ChainOfThoughtPrompting:
    """
    Chain-of-Thought prompting for improved reasoning.

    This class wraps your GPT model and adds reasoning capabilities.
    Think of it like a decorator in C# that enhances functionality.
    """

    def __init__(self, base_model):
        """
        Initialize CoT prompting system.

        Args:
            base_model: Your GPT model from Module 6

        C# equivalent:
            public ChainOfThoughtPrompting(GPTModel baseModel) {
                this.baseModel = baseModel;
            }
        """
        self.base_model = base_model
        self.few_shot_examples = []  # Like List<Example> in C#

    def add_example(self, question: str, reasoning_steps: List[str], answer: str):
        """
        Add a few-shot example of step-by-step reasoning.

        Args:
            question: The question
            reasoning_steps: List of reasoning steps
            answer: Final answer

        Example:
            cot.add_example(
                question="What is 5 + 3 × 2?",
                reasoning_steps=[
                    "Follow order of operations (PEMDAS)",
                    "First: 3 × 2 = 6",
                    "Then: 5 + 6 = 11"
                ],
                answer="11"
            )
        """
        # Format the example
        example = {
            'question': question,
            'reasoning': reasoning_steps,  # Like List<string> in C#
            'answer': answer
        }
        self.few_shot_examples.append(example)  # Like List.Add() in C#

    def format_few_shot_prompt(self, question: str) -> str:
        """
        Create a prompt with examples showing step-by-step reasoning.

        This is like string interpolation in C#:
            var prompt = $"Q: {question}\nA: Let me think...";

        Args:
            question: The question to answer

        Returns:
            Formatted prompt with examples
        """
        prompt_parts = []  # Like StringBuilder in C#

        # Add each example
        for example in self.few_shot_examples:
            # Question
            prompt_parts.append(f"Q: {example['question']}")

            # Reasoning steps
            prompt_parts.append("A: Let me think step by step:")
            for i, step in enumerate(example['reasoning'], 1):
                prompt_parts.append(f"   {i}. {step}")

            # Final answer
            prompt_parts.append(f"   Answer: {example['answer']}")
            prompt_parts.append("")  # Blank line

        # Add the actual question
        prompt_parts.append(f"Q: {question}")
        prompt_parts.append("A: Let me think step by step:")

        # Join all parts
        # Like string.Join("\n", list) in C#
        return "\n".join(prompt_parts)

    def generate_with_cot(self, question: str, max_steps: int = 10) -> dict:
        """
        Generate an answer using Chain-of-Thought reasoning.

        Args:
            question: The question to answer
            max_steps: Maximum reasoning steps to generate

        Returns:
            Dictionary with reasoning steps and final answer

        C# equivalent return type:
            public class CoTResult {
                public List<string> ReasoningSteps { get; set; }
                public string Answer { get; set; }
            }
        """
        # Create the prompt
        prompt = self.format_few_shot_prompt(question)

        # Generate response
        response = self.base_model.generate(
            prompt=prompt,
            max_length=200,  # Enough for reasoning steps
            temperature=0.7,  # Balanced creativity
            top_p=0.9
        )

        # Parse the response to extract steps and answer
        steps, answer = self._parse_cot_response(response)

        return {
            'question': question,
            'reasoning_steps': steps,
            'answer': answer,
            'full_response': response
        }
```

**Line-by-line explanation:**

```python
self.few_shot_examples = []
```
- Creates an empty list to store examples
- Like `List<Example> examples = new List<Example>();` in C#
- Each example shows the model HOW to reason

```python
'reasoning': reasoning_steps,
```
- Stores the list of steps
- Like a `List<string>` property in a C# class
- This is the "show your work" part!

```python
prompt_parts.append(f"   {i}. {step}")
```
- Uses f-string (Python's string interpolation)
- Like `$"   {i}. {step}"` in C#
- Formats each step with a number

---

### Part 2: Zero-Shot CoT (Simpler!)

```python
class ZeroShotCoT:
    """
    Zero-shot Chain-of-Thought prompting.

    No examples needed - just add magic words!
    Like a static method in C# that doesn't need instance data.
    """

    def __init__(self, base_model):
        self.base_model = base_model
        # The magic phrase that triggers reasoning!
        self.cot_trigger = "Let's think step by step:"

    def generate_with_thinking(self, question: str) -> dict:
        """
        Generate answer with automatic step-by-step thinking.

        Args:
            question: The question to answer

        Returns:
            Dictionary with reasoning and answer
        """
        # Create prompt with magic trigger phrase
        prompt = f"{question}\n{self.cot_trigger}"

        # Generate reasoning + answer
        response = self.base_model.generate(
            prompt=prompt,
            max_length=300,
            temperature=0.7
        )

        # Parse response
        steps, answer = self._extract_reasoning_and_answer(response)

        return {
            'question': question,
            'reasoning_steps': steps,
            'answer': answer,
            'full_response': response
        }

    def _extract_reasoning_and_answer(self, response: str) -> Tuple[List[str], str]:
        """
        Extract reasoning steps and final answer from response.

        This is like parsing structured text in C#:
            var parts = response.Split('\n');
            var steps = parts.Where(p => p.Contains("Step")).ToList();
        """
        lines = response.strip().split('\n')
        steps = []
        answer = None

        for line in lines:
            line = line.strip()

            # Look for numbered steps or keywords
            if any(keyword in line.lower() for keyword in ['step', 'first', 'then', 'next', 'finally']):
                steps.append(line)

            # Look for final answer
            if 'answer:' in line.lower() or 'therefore' in line.lower():
                answer = line

        return steps, answer
```

**Why this works:**

The phrase "Let's think step by step:" was discovered to trigger reasoning in LLMs. It's like a magic spell that activates a reasoning mode!

**Analogy to C#/.NET:**
```csharp
// Without trigger
var result = Calculate(problem);  // Might fail

// With trigger (like verbose mode)
var result = Calculate(problem, showSteps: true);  // Shows work!
```

---

## 🧪 Complete Example: Math Problem Solver

Let's build a complete math reasoning system using CoT:

```python
class MathReasoningSystem:
    """
    A math problem solver that shows its reasoning.

    Like a C# calculator class, but it explains its work:
    public class SmartCalculator {
        public (List<string> Steps, double Answer) Solve(string problem) { ... }
    }
    """

    def __init__(self, base_model):
        self.base_model = base_model
        self.cot = ChainOfThoughtPrompting(base_model)

        # Add math examples
        self._add_math_examples()

    def _add_math_examples(self):
        """Add few-shot examples for math problems."""

        # Example 1: Basic arithmetic
        self.cot.add_example(
            question="What is 23 + 45?",
            reasoning_steps=[
                "Break down: 23 + 45",
                "Add tens: 20 + 40 = 60",
                "Add ones: 3 + 5 = 8",
                "Total: 60 + 8 = 68"
            ],
            answer="68"
        )

        # Example 2: Word problem
        self.cot.add_example(
            question="Alice has 3 times as many apples as Bob. Together they have 24 apples. How many does each have?",
            reasoning_steps=[
                "Let Bob's apples = x",
                "Alice's apples = 3x (three times as many)",
                "Together: x + 3x = 24",
                "Simplify: 4x = 24",
                "Solve: x = 6",
                "Bob has 6, Alice has 3 × 6 = 18"
            ],
            answer="Bob: 6 apples, Alice: 18 apples"
        )

        # Example 3: Multi-step
        self.cot.add_example(
            question="A shirt costs $25. It's on sale for 20% off. Tax is 10%. What's the final price?",
            reasoning_steps=[
                "Original price: $25",
                "20% off means: 20% of $25 = $5",
                "Sale price: $25 - $5 = $20",
                "Tax: 10% of $20 = $2",
                "Final price: $20 + $2 = $22"
            ],
            answer="$22"
        )

    def solve(self, problem: str) -> dict:
        """
        Solve a math problem with step-by-step reasoning.

        Args:
            problem: Math problem in natural language

        Returns:
            Dictionary with reasoning steps and answer
        """
        # Use Chain-of-Thought to solve
        result = self.cot.generate_with_cot(problem)

        # Add verification (bonus!)
        result['verified'] = self._verify_reasoning(result['reasoning_steps'])

        return result

    def _verify_reasoning(self, steps: List[str]) -> bool:
        """
        Verify that reasoning steps are logical.

        Simple checks:
        - Are there multiple steps? (not just guessing)
        - Do steps follow logically?
        - Is there a clear answer?

        In production, you'd have more sophisticated checks!
        """
        if len(steps) < 2:
            return False  # Too few steps, probably guessing

        # Check for mathematical keywords
        math_keywords = ['calculate', 'add', 'subtract', 'multiply', 'divide', '=', '+', '-', '×', '÷']
        has_math = any(any(kw in step.lower() for kw in math_keywords) for step in steps)

        return has_math
```

---

## 🧪 Using the Math Reasoning System

```python
# Example usage
from modules.module_06.example_01_complete_gpt import GPT, GPTConfig

# Load your GPT model
config = GPTConfig(vocab_size=50257, max_seq_len=256, embed_dim=512, n_layers=6, n_heads=8)
gpt = GPT(config)
# gpt.load_weights('path/to/your/trained/gpt.pth')  # If you have trained weights

# Create reasoning system
math_solver = MathReasoningSystem(gpt)

# Solve a problem
problem = "If a car travels 60 mph for 2.5 hours, how far does it go?"
result = math_solver.solve(problem)

# Display results
print(f"Problem: {result['question']}\n")
print("Reasoning:")
for i, step in enumerate(result['reasoning_steps'], 1):
    print(f"  {i}. {step}")
print(f"\nAnswer: {result['answer']}")
print(f"Verified: {result['verified']} ✓" if result['verified'] else "Verified: ✗")
```

**Expected output:**
```
Problem: If a car travels 60 mph for 2.5 hours, how far does it go?

Reasoning:
  1. Speed = 60 miles per hour
  2. Time = 2.5 hours
  3. Distance = Speed × Time
  4. Distance = 60 × 2.5
  5. Calculate: 60 × 2.5 = 150 miles

Answer: 150 miles
Verified: ✓
```

---

## 📊 Comparing With vs. Without CoT

Let's see the dramatic difference:

```python
def compare_with_without_cot(gpt_model, problems: List[str]):
    """
    Compare accuracy with and without Chain-of-Thought.

    C# equivalent:
    public void CompareAccuracy(GPTModel model, List<string> problems) {
        var withoutCoT = problems.Select(p => SolveDirectly(p));
        var withCoT = problems.Select(p => SolveWithCoT(p));
        CompareResults(withoutCoT, withCoT);
    }
    """
    results = {
        'without_cot': [],
        'with_cot': []
    }

    for problem in problems:
        # WITHOUT CoT - direct answer
        direct_answer = gpt_model.generate(
            prompt=problem,
            max_length=50,
            temperature=0.7
        )
        results['without_cot'].append(direct_answer)

        # WITH CoT - reasoning first
        cot_prompt = f"{problem}\nLet's think step by step:"
        cot_answer = gpt_model.generate(
            prompt=cot_prompt,
            max_length=200,
            temperature=0.7
        )
        results['with_cot'].append(cot_answer)

    return results

# Test on math problems
problems = [
    "What is 137 + 289?",
    "If a book costs $15 with 25% off, what do you pay?",
    "A rectangle is 8cm by 5cm. What's the perimeter?"
]

comparison = compare_with_without_cot(gpt, problems)

# Display comparison
for i, problem in enumerate(problems):
    print(f"\nProblem {i+1}: {problem}")
    print(f"Without CoT: {comparison['without_cot'][i]}")
    print(f"With CoT: {comparison['with_cot'][i]}")
```

**Typical results:**
```
Problem 1: What is 137 + 289?
Without CoT: "426" ✓ (simple, might work)
With CoT: "Step 1: Add hundreds: 100 + 200 = 300
          Step 2: Add tens: 30 + 80 = 110
          Step 3: Add ones: 7 + 9 = 16
          Step 4: Total: 300 + 110 + 16 = 426" ✓ (verified!)

Problem 2: If a book costs $15 with 25% off, what do you pay?
Without CoT: "$12.50" ✗ (wrong!)
With CoT: "Step 1: Original price: $15
          Step 2: Discount: 25% of $15 = $3.75
          Step 3: Final price: $15 - $3.75 = $11.25" ✓ (correct!)
```

---

## 🎯 When to Use Chain-of-Thought

### ✅ Use CoT For:
- **Math problems** - Requires step-by-step calculation
- **Logic puzzles** - Need to track multiple conditions
- **Multi-step reasoning** - Planning, analysis
- **Verification needed** - Important decisions
- **Teaching/explaining** - Need to show the process

### ❌ Don't Use CoT For:
- **Simple questions** - "What's the capital of France?" (just "Paris")
- **Creative writing** - "Write a poem" (reasoning not needed)
- **Fast responses** - When you need quick answers
- **Short outputs** - When token limit is strict

---

## 🔬 Advanced: Self-Consistency with CoT

**Next-level technique:** Generate multiple reasoning paths and vote!

```python
class SelfConsistentCoT:
    """
    Generate multiple reasoning paths and pick the most consistent answer.

    Like running multiple test cases in C#:
    var results = Enumerable.Range(1, 5).Select(_ => Solve(problem));
    var mostCommon = results.GroupBy(r => r).OrderByDescending(g => g.Count()).First();
    """

    def __init__(self, base_model, n_samples: int = 5):
        self.base_model = base_model
        self.n_samples = n_samples  # Number of reasoning paths to generate
        self.cot = ZeroShotCoT(base_model)

    def solve_with_consistency(self, question: str) -> dict:
        """
        Generate multiple reasoning paths and vote on the answer.
        """
        # Generate multiple solutions
        solutions = []
        for _ in range(self.n_samples):
            result = self.cot.generate_with_thinking(question)
            solutions.append(result)

        # Count answers (majority vote)
        # Like: var counts = answers.GroupBy(a => a).ToDictionary(g => g.Key, g => g.Count());
        answer_counts = {}
        for sol in solutions:
            ans = sol['answer']
            answer_counts[ans] = answer_counts.get(ans, 0) + 1

        # Pick most common answer
        best_answer = max(answer_counts, key=answer_counts.get)

        # Find the best reasoning path for that answer
        best_solution = next(s for s in solutions if s['answer'] == best_answer)

        return {
            'question': question,
            'all_solutions': solutions,
            'final_answer': best_answer,
            'confidence': answer_counts[best_answer] / self.n_samples,
            'reasoning': best_solution['reasoning_steps']
        }

# Usage
sc_cot = SelfConsistentCoT(gpt, n_samples=5)
result = sc_cot.solve_with_consistency("What is 23 × 47?")

print(f"Question: {result['question']}")
print(f"Final Answer: {result['final_answer']}")
print(f"Confidence: {result['confidence']:.0%} (appeared in {int(result['confidence'] * 5)}/5 samples)")
print("\nBest Reasoning:")
for step in result['reasoning']:
    print(f"  {step}")
```

---

## 🎓 Connection to OpenAI o1

**How o1 likely uses CoT:**

```
User question: "Solve complex problem"
      ↓
o1 Internal Process:
1. Generate reasoning tokens (not shown to user)
   - "Let me think about this..."
   - "First, I should..."
   - "Wait, that doesn't work..."
   - "Let me try another approach..."

2. Verify reasoning (process supervision)
   - Check each step for validity
   - Prune bad reasoning paths
   - Continue good paths

3. Synthesize final answer
   - Combine best reasoning
   - Format for user
      ↓
Output: Polished answer + (optionally) reasoning trace
```

**You just learned the foundation of how o1 works!**

---

## ✅ Quiz Questions

Test your understanding:

1. **What is Chain-of-Thought prompting?**
   - A) Making models think faster
   - B) Teaching models to show step-by-step reasoning
   - C) A new model architecture
   - D) A fine-tuning technique

2. **What's the "magic phrase" for zero-shot CoT?**
   - A) "Think carefully"
   - B) "Let's think step by step"
   - C) "Show your work"
   - D) "Explain your answer"

3. **When should you use CoT?**
   - A) Always, for every question
   - B) Never, it's too slow
   - C) For complex reasoning tasks
   - D) Only for math problems

4. **What's the main benefit of showing reasoning steps?**
   - A) Makes output longer
   - B) Increases accuracy on complex problems
   - C) Makes the model run faster
   - D) Reduces token usage

**Answers:** 1-B, 2-B, 3-C, 4-B

---

## 🛠️ Hands-On Exercise

**Build your own CoT system:**

```python
# Exercise: Create a Logic Reasoning System
class LogicReasoningSystem:
    """
    Solve logic puzzles using Chain-of-Thought.

    Your task: Implement this class!
    """

    def __init__(self, base_model):
        # TODO: Initialize with your GPT model
        # TODO: Set up CoT prompting
        pass

    def solve_logic_puzzle(self, puzzle: str) -> dict:
        """
        Solve a logic puzzle step by step.

        Example puzzle:
        "Three friends: Alice, Bob, Carol.
         - Alice is taller than Bob
         - Bob is taller than Carol
         - Who is the tallest?"

        Expected output:
        {
            'puzzle': "...",
            'reasoning_steps': [
                "Alice > Bob (Alice taller than Bob)",
                "Bob > Carol (Bob taller than Carol)",
                "Therefore: Alice > Bob > Carol",
                "Alice is the tallest"
            ],
            'answer': "Alice"
        }
        """
        # TODO: Implement this!
        # Hints:
        # 1. Use few-shot CoT with logic examples
        # 2. Parse the response to extract steps
        # 3. Verify the logic is sound
        pass

# Test your implementation
puzzle = """
Three people: Alice, Bob, Carol
Rules:
- Only one person tells the truth
- Alice says: "Bob is lying"
- Bob says: "Carol is lying"
- Carol says: "Alice is lying"
Who tells the truth?
"""

# Your code here:
# logic_solver = LogicReasoningSystem(gpt)
# result = logic_solver.solve_logic_puzzle(puzzle)
# print(result)
```

---

## 📝 Summary

**What you learned:**

1. **Chain-of-Thought = "Show your work"**
   - Dramatically improves reasoning accuracy
   - Makes AI explain its thinking
   - Foundation of models like o1

2. **Two approaches:**
   - Few-shot: Provide examples of reasoning
   - Zero-shot: Just add "Let's think step by step"

3. **When to use:**
   - Complex reasoning tasks
   - Math problems
   - Logic puzzles
   - Any task requiring verification

4. **Implementation:**
   - Wrap your GPT model
   - Add reasoning triggers
   - Parse step-by-step output

**C#/.NET connections:**
- CoT examples = XML documentation examples
- Few-shot learning = Providing sample code
- Reasoning steps = Logging/tracing in debug mode
- Self-consistency = Running multiple test cases

---

## 🚀 Next Steps

**You've mastered Chain-of-Thought!**

Next lesson: **Self-Consistency** - Making reasoning even more reliable by generating multiple paths and voting.

**Continue to:** `02_self_consistency.md`

---

**You just learned the secret behind o1's reasoning abilities!** 🎉

**This is cutting-edge AI - and you built it from scratch!** 💪

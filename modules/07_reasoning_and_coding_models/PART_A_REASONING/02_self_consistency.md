# Lesson 7.2: Self-Consistency & Ensemble Reasoning

## 🎯 Learning Objectives

By the end of this lesson, you'll be able to:

- Understand why a single reasoning path can fail
- Implement self-consistency with multiple reasoning paths
- Use majority voting to select the best answer
- Calculate confidence scores for answers
- Build ensemble reasoning systems
- Know when to use self-consistency vs. simple CoT
- Understand how o1 likely uses multiple reasoning attempts

---

## 🤔 What is Self-Consistency?

### The Problem: One Reasoning Path Can Be Wrong

Even with Chain-of-Thought, a single reasoning path might make mistakes:

**Example - Chain-of-Thought failing:**

```
Question: "What is 237 × 486?"

Path 1 (CoT):
Step 1: Break down 486 = 400 + 80 + 6
Step 2: 237 × 400 = 94,800
Step 3: 237 × 80 = 18,960
Step 4: 237 × 6 = 1,442  ← ERROR! Should be 1,422
Step 5: Sum = 94,800 + 18,960 + 1,442 = 115,202 ✗ WRONG!
```

**The model made ONE mistake in step 4, and got the wrong answer!**

---

### The Solution: Try Multiple Times, Vote on Answer

**Self-Consistency = Generate multiple reasoning paths, pick the most common answer**

```
Question: "What is 237 × 486?"

Path 1: ... → 115,202 ✗
Path 2: ... → 115,182 ✓
Path 3: ... → 115,182 ✓
Path 4: ... → 115,182 ✓
Path 5: ... → 115,200 ✗

Majority vote: 115,182 (appears 3/5 times = 60% confidence) ✓ CORRECT!
```

**Even though 2 paths failed, the majority got it right!**

---

## 🌍 Real-World Analogy

Think of Self-Consistency like this:

### Single CoT: Asking One Expert
```
You have a difficult medical question.

Scenario 1: Ask one doctor
Doctor A: "I think it's condition X"

Problem: What if they're wrong? No second opinion!
Risk: High
```

### Self-Consistency: Asking Multiple Experts
```
You have a difficult medical question.

Scenario 2: Ask five doctors
Doctor A: "I think it's condition X"
Doctor B: "I think it's condition Y"  ← Different!
Doctor C: "I think it's condition Y"
Doctor D: "I think it's condition Y"
Doctor E: "I think it's condition X"

Majority says: Condition Y (3/5 doctors = 60% confidence)
Confidence: Medium-High
Risk: Lower

You trust the majority opinion!
```

**Self-Consistency is like getting multiple expert opinions!**

---

## 📚 How Self-Consistency Works

### The Algorithm (Simple Version)

```
DEFINITION: Self-Consistency
A reasoning technique where you:
1. Generate multiple different reasoning paths (using Chain-of-Thought)
2. Each path arrives at an answer independently
3. Count how many times each answer appears
4. Pick the answer that appears most often (majority vote)
5. Use the frequency as a confidence score
```

**Step-by-Step Process:**

```
Input: Question

Step 1: Generate Path 1 with CoT → Answer A
Step 2: Generate Path 2 with CoT → Answer B
Step 3: Generate Path 3 with CoT → Answer A
Step 4: Generate Path 4 with CoT → Answer A
Step 5: Generate Path 5 with CoT → Answer C

Step 6: Count answers
  - Answer A: 3 votes (60%)
  - Answer B: 1 vote (20%)
  - Answer C: 1 vote (20%)

Step 7: Pick majority → Answer A
Step 8: Confidence = 60%

Output: Answer A with 60% confidence
```

**Like voting in C#:**
```csharp
// C# equivalent
var answers = new List<string> { "A", "B", "A", "A", "C" };
var mostCommon = answers
    .GroupBy(a => a)
    .OrderByDescending(g => g.Count())
    .First();
// mostCommon.Key = "A"
// mostCommon.Count() = 3
// Confidence = 3 / 5 = 60%
```

---

### Why This Works

**Statistical Principle:**

```
DEFINITION: Law of Large Numbers
When you repeat something many times, random errors cancel out
and the true answer emerges.

Example:
- Flip a coin 5 times: might get 4 heads (80%) ← Random variation
- Flip a coin 1000 times: get ~500 heads (50%) ← True probability

Same with reasoning:
- 1 reasoning path: might make a mistake
- 5 reasoning paths: mistakes are random, correct answer appears most
```

**Think of it like unit testing in C#:**
```csharp
// Run the same test multiple times with slight variations
[Theory]
[InlineData(1)]
[InlineData(2)]
[InlineData(3)]
[InlineData(4)]
[InlineData(5)]
public void TestReasoningPath(int seed)
{
    var result = SolveWithDifferentRandomness(problem, seed);
    // Most should agree on the correct answer
}
```

---

## 💻 Implementation in Python

### Part 1: Basic Self-Consistency

```python
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter  # For counting answers

class SelfConsistencyReasoning:
    """
    Self-Consistency reasoning with majority voting.

    This enhances Chain-of-Thought by generating multiple reasoning paths
    and picking the most common answer.

    C# analogy:
    public class SelfConsistencyReasoning {
        private GPTModel baseModel;
        private int numSamples;

        public (string Answer, double Confidence) Solve(string question) {
            var answers = GenerateMultiplePaths(question);
            return MajorityVote(answers);
        }
    }
    """

    def __init__(self, base_model, cot_system, n_samples: int = 5):
        """
        Initialize Self-Consistency system.

        Args:
            base_model: Your GPT model from Module 6
            cot_system: Chain-of-Thought system from Lesson 1
            n_samples: Number of reasoning paths to generate (default: 5)

        DEFINITION - n_samples:
        The number of different reasoning attempts to generate.
        - More samples = higher confidence but slower
        - Typical: 3-10 samples
        - Production: 5-20 samples

        Like running multiple test iterations:
        public SelfConsistency(GPTModel model, int iterations = 5) {
            this.iterations = iterations;
        }
        """
        self.base_model = base_model
        self.cot = cot_system  # From Lesson 1
        self.n_samples = n_samples

    def generate_multiple_paths(self, question: str) -> List[Dict]:
        """
        Generate multiple independent reasoning paths.

        DEFINITION - Independent paths:
        Each path is generated separately with slight randomness,
        so they might take different approaches but should reach
        the same correct answer.

        Like running the same function with different random seeds:
        var results = Enumerable.Range(1, 5)
            .Select(seed => SolveWithSeed(problem, seed))
            .ToList();

        Args:
            question: The question to answer

        Returns:
            List of reasoning results, each containing:
            - question: The original question
            - reasoning_steps: List of reasoning steps taken
            - answer: The final answer
            - temperature: Randomness used for this path
        """
        paths = []  # Like List<ReasoningPath> in C#

        # Generate each path with slight variation
        for i in range(self.n_samples):
            # Vary temperature slightly for diversity
            # DEFINITION - Temperature:
            # Controls randomness in generation (0.0 = deterministic, 1.0 = very random)
            # We vary it slightly (0.7-0.9) so each path might reason differently
            temperature = 0.7 + (i * 0.05)  # 0.7, 0.75, 0.8, 0.85, 0.9

            # Generate one reasoning path using CoT
            result = self.cot.generate_with_cot(
                question=question,
                temperature=temperature  # Slight randomness for diversity
            )

            # Store this path
            result['path_id'] = i + 1  # Track which path this is
            result['temperature'] = temperature
            paths.append(result)

        return paths

    def extract_answer(self, reasoning_result: Dict) -> str:
        """
        Extract the final answer from a reasoning result.

        DEFINITION - Answer extraction:
        Parse the reasoning output to find just the final answer,
        ignoring all the reasoning steps.

        Example:
        Input: {
            'reasoning_steps': ['Step 1...', 'Step 2...'],
            'answer': 'The answer is 42'
        }
        Output: '42'

        C# equivalent:
        private string ExtractAnswer(ReasoningResult result) {
            return result.Answer.Replace("The answer is ", "").Trim();
        }
        """
        answer = reasoning_result.get('answer', '')

        # Clean up the answer
        # Remove common prefixes
        for prefix in ['The answer is ', 'Answer: ', 'Therefore, ', 'So, ']:
            if answer.startswith(prefix):
                answer = answer[len(prefix):]

        # Trim whitespace
        answer = answer.strip()

        return answer

    def majority_vote(self, paths: List[Dict]) -> Dict:
        """
        Perform majority voting on the answers.

        DEFINITION - Majority voting:
        Count how many times each answer appears and pick the most common one.

        Example:
        Answers: ['42', '43', '42', '42', '44']
        Counts: {'42': 3, '43': 1, '44': 1}
        Winner: '42' (appears 3/5 = 60%)

        C# equivalent:
        private (string Answer, double Confidence) MajorityVote(List<string> answers) {
            var groups = answers.GroupBy(a => a)
                               .OrderByDescending(g => g.Count())
                               .ToList();
            var winner = groups.First();
            return (winner.Key, (double)winner.Count() / answers.Count);
        }

        Args:
            paths: List of reasoning paths with answers

        Returns:
            Dictionary containing:
            - final_answer: The winning answer
            - confidence: How often it appeared (0.0 to 1.0)
            - vote_counts: How many votes each answer got
            - all_paths: All the reasoning paths generated
        """
        # Extract all answers
        answers = [self.extract_answer(path) for path in paths]

        # Count occurrences
        # Python's Counter is like C#'s GroupBy + Count
        # counter = { 'answer1': 3, 'answer2': 1, 'answer3': 1 }
        answer_counts = Counter(answers)

        # Find the most common answer
        # Like: var mostCommon = groups.OrderByDescending(g => g.Count()).First()
        most_common_answer, vote_count = answer_counts.most_common(1)[0]

        # Calculate confidence
        # DEFINITION - Confidence:
        # The percentage of paths that agreed on this answer
        # 0.0 = no agreement, 1.0 = all paths agree
        confidence = vote_count / len(paths)

        # Find the best reasoning path for the winning answer
        best_path = next(p for p in paths if self.extract_answer(p) == most_common_answer)

        return {
            'question': paths[0]['question'],
            'final_answer': most_common_answer,
            'confidence': confidence,
            'vote_counts': dict(answer_counts),  # Convert Counter to dict
            'all_paths': paths,
            'best_reasoning': best_path['reasoning_steps'],
            'n_samples': len(paths)
        }

    def solve_with_consistency(self, question: str) -> Dict:
        """
        Solve a question using self-consistency.

        This is the main method that combines everything:
        1. Generate multiple reasoning paths
        2. Extract answers from each path
        3. Vote on the most common answer
        4. Return result with confidence score

        Args:
            question: The question to answer

        Returns:
            Complete result with answer, confidence, and all reasoning paths
        """
        # Step 1: Generate multiple reasoning paths
        print(f"Generating {self.n_samples} reasoning paths...")
        paths = self.generate_multiple_paths(question)

        # Step 2: Vote on the best answer
        print("Performing majority vote...")
        result = self.majority_vote(paths)

        return result
```

**Line-by-line explanation of key parts:**

```python
temperature = 0.7 + (i * 0.05)
```
- **What it does:** Varies the randomness for each path
- **Why:** So each path might reason slightly differently
- **C# analogy:** Like using different random seeds
- **Values:** Path 1 uses 0.7, Path 2 uses 0.75, etc.

```python
answer_counts = Counter(answers)
```
- **What it does:** Counts how many times each answer appears
- **Counter:** Python's built-in class for counting (like Dictionary<string, int>)
- **Example:** `Counter(['A', 'B', 'A', 'A', 'C'])` → `{'A': 3, 'B': 1, 'C': 1}`
- **C# equivalent:** `answers.GroupBy(a => a).ToDictionary(g => g.Key, g => g.Count())`

```python
confidence = vote_count / len(paths)
```
- **What it does:** Calculates percentage agreement
- **Example:** 3 votes out of 5 paths = 3/5 = 0.6 = 60%
- **Higher is better:** 100% means all paths agreed!

---

## 🧪 Complete Example: Self-Consistent Math Solver

```python
class SelfConsistentMathSolver:
    """
    Math solver using self-consistency for higher accuracy.

    Like having multiple calculators check the same problem:
    public class RedundantCalculator {
        public (double Answer, double Confidence) Calculate(string problem) {
            var results = RunMultipleCalculations(problem, times: 5);
            return MajorityVote(results);
        }
    }
    """

    def __init__(self, base_model, n_samples: int = 5):
        from modules.module_07.PART_A_REASONING.lesson_01_cot import ChainOfThoughtPrompting

        self.base_model = base_model
        self.cot = ChainOfThoughtPrompting(base_model)
        self.sc = SelfConsistencyReasoning(base_model, self.cot, n_samples)

        # Add math examples for CoT
        self._setup_math_examples()

    def _setup_math_examples(self):
        """Add math examples for few-shot CoT."""
        self.cot.add_example(
            question="What is 45 + 78?",
            reasoning_steps=[
                "Break down: 45 + 78",
                "Add tens: 40 + 70 = 110",
                "Add ones: 5 + 8 = 13",
                "Total: 110 + 13 = 123"
            ],
            answer="123"
        )

        self.cot.add_example(
            question="A shirt costs $30 with 20% off. What's the final price?",
            reasoning_steps=[
                "Original price: $30",
                "Discount: 20% of $30 = $6",
                "Final price: $30 - $6 = $24"
            ],
            answer="$24"
        )

    def solve(self, problem: str, show_all_paths: bool = False) -> None:
        """
        Solve a math problem with self-consistency.

        Args:
            problem: Math problem to solve
            show_all_paths: Whether to display all reasoning paths
        """
        print(f"Problem: {problem}\n")
        print("=" * 60)

        # Solve with self-consistency
        result = self.sc.solve_with_consistency(problem)

        # Display all paths if requested
        if show_all_paths:
            print("\nAll Reasoning Paths:")
            print("-" * 60)
            for path in result['all_paths']:
                answer = self.sc.extract_answer(path)
                print(f"\nPath {path['path_id']} (temp={path['temperature']:.2f}):")
                for step in path['reasoning_steps']:
                    print(f"  {step}")
                print(f"  → Answer: {answer}")
            print("\n" + "=" * 60)

        # Display vote counts
        print("\nVoting Results:")
        print("-" * 60)
        for answer, count in result['vote_counts'].items():
            percentage = (count / result['n_samples']) * 100
            bar = "█" * count + "░" * (result['n_samples'] - count)
            print(f"{answer:20s} : {bar} ({count}/{result['n_samples']} = {percentage:.0f}%)")

        # Display final answer
        print("\n" + "=" * 60)
        print(f"Final Answer: {result['final_answer']}")
        print(f"Confidence: {result['confidence']:.0%}")

        # Interpret confidence
        if result['confidence'] >= 0.8:
            print("Confidence Level: HIGH - Very reliable!")
        elif result['confidence'] >= 0.6:
            print("Confidence Level: MEDIUM - Fairly reliable")
        else:
            print("Confidence Level: LOW - Be cautious!")

        print("=" * 60)
```

---

## 🧪 Using Self-Consistent Math Solver

```python
from modules.module_06.example_01_complete_gpt import GPT, GPTConfig

# Load your GPT model
config = GPTConfig(vocab_size=50257, max_seq_len=256, embed_dim=512, n_layers=6, n_heads=8)
gpt = GPT(config)

# Create self-consistent solver
solver = SelfConsistentMathSolver(gpt, n_samples=5)

# Solve a problem
problem = "What is 237 × 486?"
solver.solve(problem, show_all_paths=True)
```

**Expected output:**

```
Problem: What is 237 × 486?

============================================================
Generating 5 reasoning paths...
Performing majority vote...

All Reasoning Paths:
------------------------------------------------------------

Path 1 (temp=0.70):
  Break down 486 = 400 + 80 + 6
  237 × 400 = 94,800
  237 × 80 = 18,960
  237 × 6 = 1,422
  Sum = 94,800 + 18,960 + 1,422 = 115,182
  → Answer: 115,182

Path 2 (temp=0.75):
  Use standard multiplication
  237 × 486 = 115,182
  → Answer: 115,182

Path 3 (temp=0.80):
  Break down 237 = 200 + 30 + 7
  200 × 486 = 97,200
  30 × 486 = 14,580
  7 × 486 = 3,402
  Sum = 97,200 + 14,580 + 3,402 = 115,182
  → Answer: 115,182

Path 4 (temp=0.85):
  237 × 486 = 115,200  ← ERROR in this path
  → Answer: 115,200

Path 5 (temp=0.90):
  Break down using distribution
  237 × (500 - 14)
  237 × 500 = 118,500
  237 × 14 = 3,318
  118,500 - 3,318 = 115,182
  → Answer: 115,182

============================================================

Voting Results:
------------------------------------------------------------
115,182              : ████░ (4/5 = 80%)
115,200              : █░░░░ (1/5 = 20%)

============================================================
Final Answer: 115,182
Confidence: 80%
Confidence Level: HIGH - Very reliable!
============================================================
```

---

## 📊 Confidence Levels

**Understanding confidence scores:**

```
CONFIDENCE INTERPRETATION:

100% (5/5 votes):
  "All paths agree - extremely reliable"
  Use case: High-stakes decisions

80% (4/5 votes):
  "Strong agreement - very reliable"
  Use case: Production systems

60% (3/5 votes):
  "Majority agrees - fairly reliable"
  Use case: Most applications

40% (2/5 votes):
  "Weak agreement - be cautious"
  Use case: Might need human review

20% (1/5 votes):
  "No consensus - unreliable"
  Use case: Generate more paths or flag for review
```

**C# analogy:**
```csharp
public enum ConfidenceLevel
{
    VeryHigh,  // >= 80%
    High,      // >= 60%
    Medium,    // >= 40%
    Low        // < 40%
}

public ConfidenceLevel GetConfidenceLevel(double confidence)
{
    return confidence switch
    {
        >= 0.8 => ConfidenceLevel.VeryHigh,
        >= 0.6 => ConfidenceLevel.High,
        >= 0.4 => ConfidenceLevel.Medium,
        _      => ConfidenceLevel.Low
    };
}
```

---

## 🎯 When to Use Self-Consistency

### ✅ Use Self-Consistency For:

**High-stakes decisions:**
- Medical diagnosis suggestions
- Financial calculations
- Legal reasoning
- Safety-critical systems

**Complex reasoning:**
- Multi-step math problems
- Logic puzzles
- Strategic planning
- Code generation (want multiple approaches)

**When accuracy > speed:**
- Offline analysis
- Batch processing
- Research applications

### ❌ Don't Use Self-Consistency For:

**Simple questions:**
- "What's 2 + 2?" (overkill!)
- "What's the capital of France?"
- Any question with obvious answer

**Speed-critical applications:**
- Real-time chat
- Interactive UIs
- When latency matters

**Low-resource environments:**
- Mobile devices
- Edge computing
- Limited API quota

---

## 🔬 Advanced: Weighted Voting

**Not all paths are equal!** Some paths might be more reliable than others.

```python
class WeightedSelfConsistency:
    """
    Self-consistency with weighted voting based on reasoning quality.

    DEFINITION - Weighted voting:
    Instead of counting each path equally (1 vote each),
    assign higher weight to better-quality reasoning.

    Example:
    Path 1: Clear, detailed reasoning → Weight 2.0
    Path 2: Vague reasoning → Weight 0.5
    Path 3: Clear reasoning → Weight 2.0

    C# analogy:
    public class WeightedVote {
        public double Weight { get; set; }
        public string Answer { get; set; }
    }

    var totalWeight = votes.GroupBy(v => v.Answer)
                          .Select(g => new { Answer = g.Key, Weight = g.Sum(v => v.Weight) })
                          .OrderByDescending(x => x.Weight)
                          .First();
    """

    def __init__(self, base_model, cot_system, n_samples: int = 5):
        self.base_model = base_model
        self.cot = cot_system
        self.n_samples = n_samples
        self.sc = SelfConsistencyReasoning(base_model, cot_system, n_samples)

    def score_reasoning_quality(self, reasoning_steps: List[str]) -> float:
        """
        Score the quality of reasoning (0.0 to 2.0).

        DEFINITION - Reasoning quality:
        How good the reasoning is based on:
        - Number of steps (more detailed = better)
        - Use of mathematical operations
        - Logical connectors ("therefore", "because")
        - Verification steps

        Higher score = more trustworthy path

        Args:
            reasoning_steps: List of reasoning steps

        Returns:
            Quality score (0.5 = poor, 1.0 = normal, 2.0 = excellent)
        """
        score = 1.0  # Start with neutral score

        # More steps = more detailed reasoning
        if len(reasoning_steps) >= 5:
            score += 0.3  # Bonus for detail
        elif len(reasoning_steps) <= 2:
            score -= 0.3  # Penalty for vagueness

        # Check for mathematical rigor
        math_keywords = ['calculate', '=', '+', '-', '×', '÷', 'sum', 'total']
        math_count = sum(
            any(kw in step.lower() for kw in math_keywords)
            for step in reasoning_steps
        )
        score += min(math_count * 0.1, 0.5)  # Up to +0.5 bonus

        # Check for logical connectors
        logic_keywords = ['therefore', 'because', 'so', 'thus', 'hence']
        has_logic = any(
            any(kw in step.lower() for kw in logic_keywords)
            for step in reasoning_steps
        )
        if has_logic:
            score += 0.2

        # Clamp score between 0.5 and 2.0
        return max(0.5, min(2.0, score))

    def weighted_majority_vote(self, paths: List[Dict]) -> Dict:
        """
        Perform weighted majority voting.

        Each path gets a weight based on reasoning quality,
        then we count weighted votes instead of simple votes.
        """
        # Score each path
        weighted_answers = []
        for path in paths:
            answer = self.sc.extract_answer(path)
            weight = self.score_reasoning_quality(path['reasoning_steps'])
            weighted_answers.append({
                'answer': answer,
                'weight': weight,
                'path': path
            })

        # Count weighted votes
        # Like: var groups = answers.GroupBy(a => a.Answer)
        #                          .Select(g => new { Answer = g.Key, TotalWeight = g.Sum(a => a.Weight) })
        answer_weights = {}
        for wa in weighted_answers:
            ans = wa['answer']
            answer_weights[ans] = answer_weights.get(ans, 0.0) + wa['weight']

        # Find winner
        best_answer = max(answer_weights, key=answer_weights.get)
        total_weight = sum(answer_weights.values())
        confidence = answer_weights[best_answer] / total_weight

        # Get best path for winner
        best_path = next(wa['path'] for wa in weighted_answers if wa['answer'] == best_answer)

        return {
            'question': paths[0]['question'],
            'final_answer': best_answer,
            'confidence': confidence,
            'answer_weights': answer_weights,
            'weighted_answers': weighted_answers,
            'best_reasoning': best_path['reasoning_steps'],
            'total_weight': total_weight
        }

    def solve(self, question: str) -> Dict:
        """Solve with weighted self-consistency."""
        paths = self.sc.generate_multiple_paths(question)
        return self.weighted_majority_vote(paths)

# Usage
weighted_sc = WeightedSelfConsistency(gpt, cot, n_samples=5)
result = weighted_sc.solve("What is 123 × 456?")
print(f"Answer: {result['final_answer']}")
print(f"Weighted Confidence: {result['confidence']:.0%}")
```

---

## ✅ Quiz Questions

Test your understanding:

1. **What is Self-Consistency?**
   - A) A new model architecture
   - B) Generating multiple reasoning paths and voting on the answer
   - C) Making the model more confident
   - D) A fine-tuning technique

2. **Why does Self-Consistency work?**
   - A) It makes the model faster
   - B) Random errors cancel out, correct answer appears most often
   - C) It uses more GPU memory
   - D) It makes reasoning simpler

3. **What does a 60% confidence score mean?**
   - A) The model is 60% sure of its answer
   - B) 60% of reasoning paths agreed on this answer
   - C) The answer is 60% correct
   - D) You should use 60% of the answer

4. **When should you NOT use Self-Consistency?**
   - A) High-stakes decisions
   - B) Complex reasoning
   - C) Simple questions or speed-critical apps
   - D) Math problems

5. **What's the typical number of samples (n_samples)?**
   - A) Always 2
   - B) 3-10 for most applications
   - C) 100+ always
   - D) Just 1 is enough

**Answers:** 1-B, 2-B, 3-B, 4-C, 5-B

---

## 🛠️ Hands-On Exercise

**Build a self-consistent logic puzzle solver:**

```python
# Exercise: Implement Self-Consistent Logic Solver
class SelfConsistentLogicSolver:
    """
    Solve logic puzzles using self-consistency.

    Your task: Complete this implementation!
    """

    def __init__(self, base_model, n_samples: int = 7):
        # TODO: Initialize CoT and Self-Consistency
        pass

    def solve_puzzle(self, puzzle: str, show_voting: bool = True) -> Dict:
        """
        Solve a logic puzzle with multiple reasoning attempts.

        Example puzzle:
        "Three students: Alice, Bob, Carol
         - Alice scored higher than Bob
         - Carol scored lower than Bob
         - Who scored highest?"

        Expected: Multiple reasoning paths, then majority vote
        """
        # TODO: Implement using self-consistency
        # Hints:
        # 1. Generate n_samples reasoning paths
        # 2. Extract answer from each path
        # 3. Perform majority voting
        # 4. Return result with confidence
        pass

# Test your implementation
puzzle = """
Four people: Alice, Bob, Carol, Dave
Clues:
- Alice is older than Bob
- Carol is younger than Bob
- Dave is younger than Carol
- Who is the oldest?
"""

# solver = SelfConsistentLogicSolver(gpt, n_samples=7)
# result = solver.solve_puzzle(puzzle, show_voting=True)
# print(f"Answer: {result['final_answer']}")
# print(f"Confidence: {result['confidence']:.0%}")
```

**Challenge:** Can you get 100% confidence (all paths agree)?

---

## 📝 Summary

**What you learned:**

1. **Self-Consistency = Multiple Expert Opinions**
   - Generate multiple reasoning paths
   - Vote on the most common answer
   - Get confidence score from agreement level

2. **Why it works:**
   - Random errors cancel out
   - Correct answer appears most frequently
   - Higher confidence = more reliable

3. **Key concepts:**
   - **Majority voting:** Pick the most common answer
   - **Confidence:** Percentage of paths that agreed
   - **Weighted voting:** Give better reasoning more weight

4. **Implementation:**
   - Generate paths with varied temperature
   - Count answers using Counter (like GroupBy in C#)
   - Calculate confidence as vote_count / total_paths

**C#/.NET connections:**
- Majority voting = GroupBy + OrderByDescending
- Multiple paths = Running tests with different seeds
- Confidence score = Percentage calculation
- Weighted voting = Sum of weights per group

**Trade-offs:**
- **Pros:** Higher accuracy, confidence scores, error resilience
- **Cons:** Slower (5x calls to model), more expensive, not always needed

---

## 🚀 Next Steps

**You've mastered Self-Consistency!**

You now understand:
- Why one reasoning path isn't enough
- How to generate multiple paths
- Majority voting and confidence
- When to use this technique

**Next lesson:** **Tree-of-Thoughts** - Going beyond linear reasoning to explore multiple solution strategies like a search algorithm!

**Continue to:** `03_tree_of_thoughts.md`

---

**You're building the same techniques used in OpenAI o1!** 🎉

**Self-Consistency is how AI systems gain reliability!** 💪

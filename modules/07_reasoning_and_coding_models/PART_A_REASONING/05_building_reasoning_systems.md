# Lesson 7.5: Building Reasoning Systems (o1-like)

## 🎯 Learning Objectives

By the end of this lesson, you'll be able to:

- Understand the complete architecture of reasoning models like o1
- Implement "thinking tokens" and reasoning phases
- Build search and verification loops
- Scale test-time compute for better reasoning
- Create a mini-o1 system from scratch
- Deploy reasoning systems in production
- Understand why o1 is slower but smarter than GPT-4

**This is the culmination of everything you've learned - you'll build your own o1!**

---

## 🤔 What Makes o1 Different?

### The Revolution: Test-Time Compute

**Traditional LLMs (GPT-3, GPT-4):**
```
User: "Solve this hard math problem..."

GPT-4 thinks for: 0.5 seconds
GPT-4 response: "The answer is 42" (might be wrong)

Why so fast? → Fixed compute per token
```

**Reasoning Models (o1, o3):**
```
User: "Solve this hard math problem..."

o1 thinks for: 30 seconds (or more!)
o1 internal: [Generates 1000s of reasoning steps]
           [Verifies each step]
           [Backtracks when wrong]
           [Explores multiple paths]
o1 response: "Here's my reasoning... [shows work] ...The answer is 42" ✓

Why so slow? → Uses MORE compute to think harder!
```

**Key insight:** o1 trades speed for accuracy by "thinking longer"

---

## 🌍 Real-World Analogy

### GPT-4: Quick Mental Math

```
Teacher: "What's 23 × 47?"
Student (GPT-4): [Thinks for 2 seconds]
                "Umm... 1,081?" (might be right, might be wrong)
```

### o1: Careful Written Work

```
Teacher: "What's 23 × 47?"
Student (o1): [Takes 30 seconds, writes everything down]

My work:
Step 1: Break down 47 = 40 + 7
Step 2: 23 × 40 = 920
Step 3: 23 × 7 = 161
Step 4: Add them: 920 + 161
Step 5: Check: Does 1,081 make sense? Yes!
Answer: 1,081 ✓

[Confident and correct because they showed their work!]
```

**o1 is the student who always shows their work and double-checks!**

---

## 📚 The o1 Architecture

### Three Core Components

```
┌─────────────────────────────────────────────┐
│           OpenAI o1 System                  │
│                                             │
│  ┌────────────────────────────────────┐    │
│  │  1. THINKING PHASE                 │    │
│  │     - Generate many reasoning steps│    │
│  │     - Explore different approaches │    │
│  │     - Internal, not shown to user  │    │
│  └────────────────────────────────────┘    │
│                    ↓                        │
│  ┌────────────────────────────────────┐    │
│  │  2. VERIFICATION PHASE             │    │
│  │     - Check each step with PRM     │    │
│  │     - Backtrack if wrong           │    │
│  │     - Search for best path         │    │
│  └────────────────────────────────────┘    │
│                    ↓                        │
│  ┌────────────────────────────────────┐    │
│  │  3. ANSWER PHASE                   │    │
│  │     - Synthesize final answer      │    │
│  │     - Show relevant reasoning      │    │
│  │     - Explain solution to user     │    │
│  └────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

Let's build each component!

---

## 💻 Component 1: Thinking Phase

### What is the Thinking Phase?

**The model generates internal reasoning that the user doesn't see (initially).**

```python
class ThinkingPhase:
    """
    Generate internal reasoning steps.

    C# analogy: Like private methods that do the work
    before returning a public result.
    """

    def __init__(self, base_model, max_thinking_steps=1000):
        self.model = base_model
        self.max_thinking_steps = max_thinking_steps

        # Special tokens
        self.THINK_START = "<think>"
        self.THINK_END = "</think>"
        self.ANSWER_START = "<answer>"
        self.ANSWER_END = "</answer>"

    def generate_thoughts(self, question):
        """
        Generate internal reasoning steps.

        Returns:
            thoughts: List of reasoning steps (internal)
            confidence: How confident model is in reasoning
        """
        thoughts = []
        confidence_scores = []

        # Start thinking
        current_context = f"{self.THINK_START}\nQuestion: {question}\n"

        for step in range(self.max_thinking_steps):
            # Generate next thought
            thought = self.model.generate_next_token_sequence(
                current_context,
                max_tokens=50,
                temperature=0.7  # Some randomness for exploration
            )

            thoughts.append(thought)
            current_context += thought + "\n"

            # Check if model has reached conclusion
            if self.has_reached_conclusion(thought):
                break

            # Prevent infinite loops
            if self.is_repeating(thoughts):
                break

        return thoughts, confidence_scores

    def has_reached_conclusion(self, thought):
        """Check if reasoning has reached a conclusion."""
        conclusion_markers = [
            "therefore",
            "thus",
            "in conclusion",
            "the answer is",
            "final answer"
        ]
        return any(marker in thought.lower() for marker in conclusion_markers)

    def is_repeating(self, thoughts):
        """Check if model is stuck in a loop."""
        if len(thoughts) < 3:
            return False

        # Check last 3 thoughts for repetition
        last_three = thoughts[-3:]
        return len(set(last_three)) == 1  # All the same


# Example usage
if __name__ == "__main__":
    thinking = ThinkingPhase(base_model)

    question = "What is 15% of 240?"

    thoughts, _ = thinking.generate_thoughts(question)

    print("Internal reasoning:")
    for i, thought in enumerate(thoughts, 1):
        print(f"  {i}. {thought}")

    # Example output:
    # Internal reasoning:
    #   1. I need to find 15% of 240
    #   2. 15% means 15/100
    #   3. So I calculate (15/100) × 240
    #   4. 15 × 240 = 3600
    #   5. 3600 / 100 = 36
    #   6. Therefore, 15% of 240 is 36
```

---

## 💻 Component 2: Verification Phase

### Using Process Reward Model to Verify

```python
class VerificationPhase:
    """
    Verify reasoning steps and backtrack if needed.

    C# analogy: Like unit tests that verify each method
    before committing to the result.
    """

    def __init__(self, process_reward_model, quality_threshold=0.7):
        self.prm = process_reward_model
        self.threshold = quality_threshold

    def verify_reasoning(self, question, thoughts):
        """
        Verify each reasoning step.

        Returns:
            verified_thoughts: Only high-quality reasoning steps
            verification_scores: Score for each step
        """
        verified_thoughts = []
        verification_scores = []

        for i, thought in enumerate(thoughts):
            # Score this step
            score = self.prm.score_step(
                question,
                verified_thoughts,  # Previous verified steps
                thought
            )

            verification_scores.append(score)

            # Only keep high-quality steps
            if score >= self.threshold:
                verified_thoughts.append(thought)
            else:
                # Low quality step detected!
                print(f"⚠️  Step {i+1} rejected (score: {score:.2f})")
                print(f"   Content: {thought}")

                # Could implement backtracking here
                # For now, just skip the bad step

        return verified_thoughts, verification_scores

    def should_backtrack(self, verification_scores, window=3):
        """
        Check if recent steps are consistently low quality.
        """
        if len(verification_scores) < window:
            return False

        recent_scores = verification_scores[-window:]
        avg_recent = sum(recent_scores) / len(recent_scores)

        return avg_recent < 0.5  # Poor recent reasoning


# Example usage
def verify_example():
    from lesson_04_process_supervision import SimpleProcessRewardModel

    prm = SimpleProcessRewardModel()
    verifier = VerificationPhase(prm, quality_threshold=0.7)

    question = "What is 8 + 7?"

    # Simulated thoughts (some good, some bad)
    thoughts = [
        "I need to add 8 + 7",                    # Good
        "8 + 7 = 20 - 5",                          # Bad reasoning!
        "Wait, let me recalculate",                # Good (self-correction)
        "8 + 7 = 15",                              # Good
        "Therefore the answer is 15"               # Good
    ]

    verified, scores = verifier.verify_reasoning(question, thoughts)

    print("\nVerification results:")
    for i, (thought, score) in enumerate(zip(thoughts, scores)):
        status = "✓" if score >= 0.7 else "✗"
        print(f"{status} Step {i+1} (score: {score:.2f}): {thought}")

    print(f"\nKept {len(verified)}/{len(thoughts)} steps")
```

---

## 💻 Component 3: Search and Exploration

### Beam Search for Reasoning

**Problem:** There might be multiple ways to solve a problem.
**Solution:** Explore multiple reasoning paths simultaneously!

```python
class ReasoningSearcher:
    """
    Explore multiple reasoning paths in parallel.

    C# analogy: Like trying multiple algorithmic approaches
    in parallel and picking the best one.
    """

    def __init__(self, base_model, prm, beam_width=5):
        self.model = base_model
        self.prm = prm
        self.beam_width = beam_width

    def beam_search_reasoning(self, question, max_steps=20):
        """
        Explore multiple reasoning paths simultaneously.

        beam_width = 5 means we keep top 5 reasoning paths at each step.
        """
        # Start with single path
        beams = [{
            'thoughts': [],
            'score': 1.0,
            'complete': False
        }]

        for step in range(max_steps):
            if all(beam['complete'] for beam in beams):
                break  # All beams have finished

            new_beams = []

            for beam in beams:
                if beam['complete']:
                    new_beams.append(beam)
                    continue

                # Generate multiple candidate next steps
                candidates = self.model.generate_n_continuations(
                    question,
                    beam['thoughts'],
                    n=self.beam_width
                )

                # Score each candidate
                for candidate in candidates:
                    thought_score = self.prm.score_step(
                        question,
                        beam['thoughts'],
                        candidate
                    )

                    # Calculate cumulative score
                    # Average of all steps so far
                    new_score = (
                        (beam['score'] * len(beam['thoughts']) + thought_score) /
                        (len(beam['thoughts']) + 1)
                    )

                    new_beam = {
                        'thoughts': beam['thoughts'] + [candidate],
                        'score': new_score,
                        'complete': self.is_complete(candidate)
                    }
                    new_beams.append(new_beam)

            # Keep only top beam_width beams
            beams = sorted(new_beams, key=lambda x: x['score'], reverse=True)
            beams = beams[:self.beam_width]

        # Return best beam
        best_beam = max(beams, key=lambda x: x['score'])
        return best_beam['thoughts'], best_beam['score']

    def is_complete(self, thought):
        """Check if reasoning is complete."""
        completion_markers = [
            'therefore',
            'thus',
            'the answer is',
            'in conclusion'
        ]
        return any(marker in thought.lower() for marker in completion_markers)


# Example: Compare single path vs beam search
def compare_search_methods():
    """
    Show difference between single path and beam search.
    """
    print("=" * 60)
    print("SINGLE PATH REASONING (like GPT-4)")
    print("=" * 60)

    single_path = [
        "8 + 7 = 15",  # Direct answer, no reasoning
    ]
    print("Steps:", len(single_path))
    print("Reasoning:", single_path[0])

    print("\n" + "=" * 60)
    print("BEAM SEARCH REASONING (like o1)")
    print("=" * 60)

    # Beam search explores multiple paths:
    beam_1 = [
        "I need to add 8 + 7",
        "8 + 7 = 15"
    ]

    beam_2 = [
        "I can use the fact that 8 + 7 = 8 + (2 + 5)",
        "= (8 + 2) + 5",
        "= 10 + 5",
        "= 15"
    ]

    beam_3 = [
        "I can count: 8, 9, 10, 11, 12, 13, 14, 15",
        "So 8 + 7 = 15"
    ]

    print("Explored 3 different reasoning paths:")
    print("\nPath 1 (score: 0.75):", beam_1)
    print("Path 2 (score: 0.92):", beam_2)  # ← Best!
    print("Path 3 (score: 0.68):", beam_3)

    print("\n→ Selected Path 2 (highest score)")
```

---

## 🧠 Complete o1-Style System

### Putting It All Together

```python
import numpy as np
from typing import List, Dict, Tuple

class O1ReasoningSystem:
    """
    Complete reasoning system inspired by OpenAI o1.

    This combines:
    - Chain-of-Thought (Lesson 7.1)
    - Self-Consistency (Lesson 7.2)
    - Tree-of-Thoughts (Lesson 7.3)
    - Process Supervision (Lesson 7.4)
    - Everything in this lesson!
    """

    def __init__(
        self,
        base_model,
        process_reward_model,
        max_thinking_steps=100,
        beam_width=5,
        quality_threshold=0.7
    ):
        """
        Initialize o1-style reasoning system.

        Args:
            base_model: Your GPT model from Module 6
            process_reward_model: PRM from Lesson 7.4
            max_thinking_steps: Maximum internal reasoning steps
            beam_width: Number of parallel reasoning paths
            quality_threshold: Minimum step quality (0-1)
        """
        self.base_model = base_model
        self.prm = process_reward_model
        self.max_thinking_steps = max_thinking_steps
        self.beam_width = beam_width
        self.quality_threshold = quality_threshold

        # Components
        self.thinking = ThinkingPhase(base_model, max_thinking_steps)
        self.verifier = VerificationPhase(process_reward_model, quality_threshold)
        self.searcher = ReasoningSearcher(base_model, process_reward_model, beam_width)

    def solve(self, question: str, show_reasoning: bool = True) -> Dict:
        """
        Solve a problem with full o1-style reasoning.

        Args:
            question: Problem to solve
            show_reasoning: Whether to show internal reasoning

        Returns:
            Dict with:
                - answer: Final answer
                - reasoning_steps: Internal reasoning
                - confidence: Confidence score (0-1)
                - thinking_time: Simulated thinking time
        """
        import time
        start_time = time.time()

        print(f"🤔 Question: {question}")
        print(f"⏳ Thinking... (exploring {self.beam_width} paths)")
        print()

        # PHASE 1: THINKING - Explore multiple reasoning paths
        best_thoughts, best_score = self.searcher.beam_search_reasoning(
            question,
            max_steps=self.max_thinking_steps
        )

        if show_reasoning:
            print("🧠 Internal Reasoning:")
            for i, thought in enumerate(best_thoughts, 1):
                print(f"   {i}. {thought}")
            print()

        # PHASE 2: VERIFICATION - Double-check the reasoning
        verified_thoughts, scores = self.verifier.verify_reasoning(
            question,
            best_thoughts
        )

        avg_score = np.mean(scores) if scores else 0.0

        if show_reasoning:
            print(f"✅ Verification: {len(verified_thoughts)}/{len(best_thoughts)} steps verified")
            print(f"📊 Average quality: {avg_score:.2f}")
            print()

        # PHASE 3: ANSWER - Synthesize final answer
        final_answer = self.synthesize_answer(verified_thoughts)

        thinking_time = time.time() - start_time

        print(f"💡 Final Answer: {final_answer}")
        print(f"⏱️  Thinking time: {thinking_time:.2f}s")
        print(f"🎯 Confidence: {avg_score:.0%}")
        print()

        return {
            'answer': final_answer,
            'reasoning_steps': verified_thoughts,
            'confidence': avg_score,
            'thinking_time': thinking_time,
            'total_steps_explored': len(best_thoughts)
        }

    def synthesize_answer(self, reasoning_steps: List[str]) -> str:
        """
        Extract final answer from reasoning steps.

        In a real system, this would use the LLM to generate
        a clean final answer based on the reasoning.
        """
        if not reasoning_steps:
            return "Unable to determine answer"

        # Simple extraction from last step
        last_step = reasoning_steps[-1]

        # Look for answer patterns
        import re
        patterns = [
            r'(?:answer is|equals?|=)\s*([0-9.]+)',
            r'therefore\s+(.+)',
            r'(?:final answer|conclusion):\s*(.+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, last_step.lower())
            if match:
                return match.group(1).strip()

        # Fallback: return last step
        return last_step

    def solve_with_self_consistency(
        self,
        question: str,
        n_samples: int = 5
    ) -> Dict:
        """
        Use self-consistency: solve multiple times and vote.

        This combines o1 reasoning with self-consistency from Lesson 7.2!
        """
        print(f"🎲 Generating {n_samples} independent solutions...")
        print()

        solutions = []
        all_reasoning = []

        for i in range(n_samples):
            print(f"--- Solution {i+1}/{n_samples} ---")
            result = self.solve(question, show_reasoning=False)
            solutions.append(result['answer'])
            all_reasoning.append(result['reasoning_steps'])
            print()

        # Vote on most common answer
        from collections import Counter
        vote_counts = Counter(solutions)
        final_answer, vote_count = vote_counts.most_common(1)[0]

        consensus = vote_count / n_samples

        print("=" * 60)
        print("📊 SELF-CONSISTENCY RESULTS")
        print("=" * 60)
        print(f"Answers generated: {solutions}")
        print(f"Most common: {final_answer} ({vote_count}/{n_samples} = {consensus:.0%})")
        print(f"Consensus level: {'High ✓' if consensus >= 0.6 else 'Low ⚠️'}")

        return {
            'answer': final_answer,
            'all_answers': solutions,
            'consensus': consensus,
            'all_reasoning': all_reasoning
        }


# ============================================================
# EXAMPLE USAGE
# ============================================================

def demo_o1_system():
    """
    Demonstrate complete o1-style reasoning system.
    """
    from lesson_04_process_supervision import SimpleProcessRewardModel

    # Initialize components
    # (In real system, these would be actual trained models)
    base_model = MockLLM()  # Placeholder
    prm = SimpleProcessRewardModel()

    # Create o1 system
    o1 = O1ReasoningSystem(
        base_model=base_model,
        process_reward_model=prm,
        max_thinking_steps=50,
        beam_width=3,
        quality_threshold=0.7
    )

    # Example 1: Simple math
    print("=" * 70)
    print("EXAMPLE 1: SIMPLE MATH")
    print("=" * 70)

    result1 = o1.solve("What is 15% of 240?")

    # Example 2: Word problem
    print("\n" + "=" * 70)
    print("EXAMPLE 2: WORD PROBLEM")
    print("=" * 70)

    question2 = """
    Sarah has $50. She buys 3 books at $12 each.
    How much money does she have left?
    """

    result2 = o1.solve(question2)

    # Example 3: With self-consistency
    print("\n" + "=" * 70)
    print("EXAMPLE 3: WITH SELF-CONSISTENCY")
    print("=" * 70)

    result3 = o1.solve_with_self_consistency(
        "What is 23 × 47?",
        n_samples=3
    )


class MockLLM:
    """
    Mock LLM for demonstration purposes.
    In real system, use your GPT model from Module 6.
    """

    def generate_next_token_sequence(self, context, max_tokens=50, temperature=0.7):
        """Generate next reasoning step."""
        # In reality, this would use your actual LLM
        # For demo, return canned responses
        mock_responses = [
            "I need to calculate this step by step",
            "Let me break down the problem",
            "First, I'll identify the key numbers",
            "Next, I'll perform the calculation",
            "The result is..."
        ]
        import random
        return random.choice(mock_responses)

    def generate_n_continuations(self, question, thoughts, n=5):
        """Generate n possible next steps."""
        return [
            self.generate_next_token_sequence(f"{question}\n{thoughts}")
            for _ in range(n)
        ]


if __name__ == "__main__":
    demo_o1_system()
```

---

## 🚀 Scaling Test-Time Compute

### The Key Innovation of o1

**Traditional scaling:**
```
More powerful model = More training compute
→ GPT-3 (175B params) → GPT-4 (1.7T params)
→ Better but expensive to train
```

**o1's scaling:**
```
More thinking time = More test-time compute
→ Same model, but thinks longer on hard problems
→ Doesn't require retraining!
```

### Implementation

```python
class AdaptiveReasoningSystem(O1ReasoningSystem):
    """
    System that adapts thinking time based on problem difficulty.

    Easy problems → Quick answer
    Hard problems → More thinking
    """

    def estimate_difficulty(self, question: str) -> float:
        """
        Estimate problem difficulty (0 = easy, 1 = hard).

        In reality, would use a trained classifier.
        """
        # Simple heuristics
        difficulty = 0.0

        # Length indicates complexity
        difficulty += min(len(question) / 200, 0.3)

        # Keywords indicate difficulty
        hard_keywords = ['prove', 'calculate', 'derive', 'complex', 'multiple']
        keyword_count = sum(1 for kw in hard_keywords if kw in question.lower())
        difficulty += min(keyword_count * 0.15, 0.4)

        # Math symbols indicate difficulty
        math_symbols = ['∫', '∑', '√', '×', '÷', '^']
        symbol_count = sum(1 for sym in math_symbols if sym in question)
        difficulty += min(symbol_count * 0.1, 0.3)

        return min(difficulty, 1.0)

    def adaptive_solve(self, question: str) -> Dict:
        """
        Solve with adaptive thinking time based on difficulty.
        """
        # Estimate difficulty
        difficulty = self.estimate_difficulty(question)

        print(f"📊 Estimated difficulty: {difficulty:.0%}")

        # Adjust parameters based on difficulty
        if difficulty < 0.3:
            # Easy problem - quick solve
            self.max_thinking_steps = 10
            self.beam_width = 1
            print("⚡ Quick mode: Minimal thinking")

        elif difficulty < 0.7:
            # Medium problem - standard solve
            self.max_thinking_steps = 50
            self.beam_width = 3
            print("🧠 Standard mode: Normal thinking")

        else:
            # Hard problem - deep thinking
            self.max_thinking_steps = 200
            self.beam_width = 5
            print("🔬 Deep mode: Extended thinking")

        print()

        # Solve with adapted parameters
        return self.solve(question)


# Example usage
def demo_adaptive_reasoning():
    """
    Show how reasoning adapts to problem difficulty.
    """
    from lesson_04_process_supervision import SimpleProcessRewardModel

    prm = SimpleProcessRewardModel()
    adaptive_o1 = AdaptiveReasoningSystem(
        base_model=MockLLM(),
        process_reward_model=prm
    )

    # Easy problem
    print("=" * 70)
    print("EASY PROBLEM")
    print("=" * 70)
    adaptive_o1.adaptive_solve("What is 2 + 2?")

    # Medium problem
    print("\n" + "=" * 70)
    print("MEDIUM PROBLEM")
    print("=" * 70)
    adaptive_o1.adaptive_solve("Calculate 15% of 240")

    # Hard problem
    print("\n" + "=" * 70)
    print("HARD PROBLEM")
    print("=" * 70)
    adaptive_o1.adaptive_solve(
        "Prove that the sum of angles in a triangle equals 180 degrees"
    )
```

---

## 📊 Comparison: GPT-4 vs o1

### Architecture Differences

| Aspect | GPT-4 | o1 |
|--------|-------|-----|
| **Thinking phase** | No internal reasoning | Yes, extensive |
| **Verification** | No step verification | PRM checks each step |
| **Search** | Single path | Beam search (multiple paths) |
| **Compute** | Fixed per token | Adaptive (harder problems = more compute) |
| **Speed** | Fast (0.5s) | Slow (5-60s) |
| **Accuracy** | Good | Better on reasoning tasks |
| **Cost** | Low | Higher (more tokens) |
| **Best for** | General tasks | Math, coding, logic |

### When to Use Each

**Use GPT-4 when:**
- Speed is critical
- Problem is straightforward
- Cost must be minimized
- Creative writing or general chat

**Use o1 when:**
- Accuracy is critical
- Problem requires reasoning
- Willing to wait for better answer
- Math, science, coding, logic problems

---

## 🎯 Production Deployment

### Optimizations for Real-World Use

```python
class ProductionReasoningSystem(O1ReasoningSystem):
    """
    Production-ready reasoning system with optimizations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Caching
        self.reasoning_cache = {}

        # Monitoring
        self.metrics = {
            'total_questions': 0,
            'avg_thinking_time': 0,
            'avg_confidence': 0,
            'cache_hits': 0
        }

    def solve_cached(self, question: str, **kwargs) -> Dict:
        """
        Solve with caching for repeated questions.
        """
        # Check cache
        cache_key = self.get_cache_key(question)

        if cache_key in self.reasoning_cache:
            self.metrics['cache_hits'] += 1
            print("💾 Cache hit! Returning cached answer.")
            return self.reasoning_cache[cache_key]

        # Not in cache - solve normally
        result = self.solve(question, **kwargs)

        # Cache result
        self.reasoning_cache[cache_key] = result

        # Update metrics
        self.update_metrics(result)

        return result

    def get_cache_key(self, question: str) -> str:
        """Generate cache key for question."""
        import hashlib
        return hashlib.md5(question.lower().encode()).hexdigest()

    def update_metrics(self, result: Dict):
        """Update performance metrics."""
        n = self.metrics['total_questions']

        # Running averages
        self.metrics['avg_thinking_time'] = (
            (self.metrics['avg_thinking_time'] * n + result['thinking_time']) /
            (n + 1)
        )
        self.metrics['avg_confidence'] = (
            (self.metrics['avg_confidence'] * n + result['confidence']) /
            (n + 1)
        )
        self.metrics['total_questions'] += 1

    def get_metrics(self) -> Dict:
        """Return performance metrics."""
        hit_rate = (
            self.metrics['cache_hits'] / max(self.metrics['total_questions'], 1)
        )

        return {
            **self.metrics,
            'cache_hit_rate': hit_rate
        }

    def solve_async(self, question: str):
        """
        Solve asynchronously for long-running problems.

        In real system, would use async/await.
        """
        import threading

        result_container = {}

        def solve_thread():
            result_container['result'] = self.solve(question)
            result_container['done'] = True

        result_container['done'] = False
        thread = threading.Thread(target=solve_thread)
        thread.start()

        return result_container  # Poll 'done' to check completion
```

---

## 🧪 Evaluation and Benchmarks

### How to Measure Reasoning Quality

```python
class ReasoningEvaluator:
    """
    Evaluate reasoning system performance.
    """

    def __init__(self, reasoning_system):
        self.system = reasoning_system

    def evaluate_on_benchmark(self, benchmark_data: List[Dict]) -> Dict:
        """
        Evaluate on standard benchmarks.

        benchmark_data format:
        [
            {
                'question': '...',
                'correct_answer': '...',
                'category': 'math' | 'logic' | 'science'
            },
            ...
        ]
        """
        results = {
            'total': len(benchmark_data),
            'correct': 0,
            'by_category': {},
            'avg_confidence': 0.0,
            'avg_thinking_time': 0.0
        }

        for item in benchmark_data:
            result = self.system.solve(item['question'], show_reasoning=False)

            # Check correctness
            is_correct = self.check_answer(
                result['answer'],
                item['correct_answer']
            )

            if is_correct:
                results['correct'] += 1

            # Track by category
            category = item.get('category', 'general')
            if category not in results['by_category']:
                results['by_category'][category] = {'total': 0, 'correct': 0}

            results['by_category'][category]['total'] += 1
            if is_correct:
                results['by_category'][category]['correct'] += 1

            # Accumulate metrics
            results['avg_confidence'] += result['confidence']
            results['avg_thinking_time'] += result['thinking_time']

        # Calculate averages
        results['accuracy'] = results['correct'] / results['total']
        results['avg_confidence'] /= results['total']
        results['avg_thinking_time'] /= results['total']

        # Calculate per-category accuracy
        for category, stats in results['by_category'].items():
            stats['accuracy'] = stats['correct'] / stats['total']

        return results

    def check_answer(self, generated: str, correct: str) -> bool:
        """Check if generated answer matches correct answer."""
        # Normalize answers
        gen_normalized = self.normalize_answer(generated)
        correct_normalized = self.normalize_answer(correct)

        return gen_normalized == correct_normalized

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        import re
        # Remove whitespace, lowercase, remove punctuation
        answer = answer.lower().strip()
        answer = re.sub(r'[^\w\s]', '', answer)
        answer = re.sub(r'\s+', ' ', answer)
        return answer

    def print_report(self, results: Dict):
        """Print evaluation report."""
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)
        print(f"Total questions: {results['total']}")
        print(f"Correct answers: {results['correct']}")
        print(f"Overall accuracy: {results['accuracy']:.1%}")
        print(f"Average confidence: {results['avg_confidence']:.1%}")
        print(f"Average thinking time: {results['avg_thinking_time']:.2f}s")

        print("\nBy Category:")
        for category, stats in results['by_category'].items():
            print(f"  {category.capitalize()}: {stats['accuracy']:.1%} "
                  f"({stats['correct']}/{stats['total']})")


# Example benchmark
def create_sample_benchmark():
    """Create sample benchmark for testing."""
    return [
        {
            'question': 'What is 15% of 80?',
            'correct_answer': '12',
            'category': 'math'
        },
        {
            'question': 'If all A are B, and all B are C, are all A also C?',
            'correct_answer': 'yes',
            'category': 'logic'
        },
        # Add more...
    ]
```

---

## 🎯 Key Takeaways

### What You Learned

1. **o1 Architecture has 3 phases:**
   - Thinking: Generate internal reasoning (not shown to user)
   - Verification: Check each step with PRM
   - Answer: Synthesize final answer from reasoning

2. **Test-time compute scaling:**
   - Traditional: Better model = more training compute
   - o1: Same model, more thinking time on hard problems
   - Can adapt compute based on difficulty

3. **Search strategies:**
   - Beam search: Explore multiple reasoning paths
   - Verification: Keep only high-quality steps
   - Backtracking: Go back when reasoning goes wrong

4. **Production considerations:**
   - Caching for repeated questions
   - Async solving for long problems
   - Metrics and monitoring
   - Cost vs quality tradeoffs

5. **When to use o1-style reasoning:**
   - Complex math, logic, science problems
   - When accuracy > speed
   - Problems that benefit from step-by-step thinking

---

## 🚀 Building Your Own o1

### Step-by-Step Guide

**Week 1: Core Components**
```
1. Implement ThinkingPhase class
2. Implement VerificationPhase class
3. Integrate with your GPT model from Module 6
4. Test on simple math problems
```

**Week 2: Search and Verification**
```
1. Implement ReasoningSearcher with beam search
2. Train or adapt Process Reward Model
3. Add backtracking logic
4. Test on harder problems
```

**Week 3: Complete System**
```
1. Combine all components into O1ReasoningSystem
2. Add self-consistency voting
3. Implement adaptive difficulty scaling
4. Evaluate on benchmarks
```

**Week 4: Production Ready**
```
1. Add caching and optimization
2. Implement async solving
3. Add monitoring and metrics
4. Deploy and test in production
```

---

## 📚 Further Reading

- OpenAI o1 System Card (official documentation)
- "Let's Verify Step by Step" (Lightman et al., 2023)
- "Tree of Thoughts" (Yao et al., 2023)
- "Chain-of-Thought Prompting" (Wei et al., 2022)
- o1 blog post on OpenAI website

---

## 🎉 Congratulations!

**You now understand how to build o1-style reasoning systems!**

You've learned:
- ✅ Chain-of-Thought prompting
- ✅ Self-consistency and voting
- ✅ Tree-of-Thoughts search
- ✅ Process supervision and PRMs
- ✅ Complete o1 architecture
- ✅ Production deployment strategies

**You can now build AI that thinks step-by-step and verifies its reasoning!**

---

## 🔜 What's Next?

**Part B: Coding Models (Lessons 6-10)**
- Learn how GitHub Copilot works
- Build code completion engines
- Train models on code
- Create your own mini-Copilot!

**Or start building:**
- Your own o1-style system
- Math reasoning solver
- Logic puzzle solver
- Any application that needs reliable reasoning!

---

**The future of AI is reasoning models - and now you know how to build them!** 🚀

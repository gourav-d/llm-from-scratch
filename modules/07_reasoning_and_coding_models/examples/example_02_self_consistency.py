"""
Example 2: Self-Consistency Reasoning

This example demonstrates self-consistency reasoning where multiple
reasoning paths are generated and the most common answer is selected.

WHAT THIS SHOWS:
- Generating multiple independent reasoning paths
- Majority voting on answers
- Confidence score calculation
- Weighted voting based on reasoning quality

COMPARISON TO C#:
Like running multiple test cases and taking the majority result:
var results = Enumerable.Range(1, 5).Select(_ => Solve(problem, seed++));
var mostCommon = results.GroupBy(r => r).OrderByDescending(g => g.Count()).First();

Author: LLM Learning Module 7
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


class MockGPT:
    """
    Mock GPT model for demonstration purposes.
    Replace with your actual GPT model from Module 6.

    C# equivalent:
    public class MockGPT : IGPTModel {
        public string Generate(string prompt, ...) {
            return MockResponse(prompt);
        }
    }
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(self, prompt: str, max_length: int = 100,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Mock generation - returns varied responses for demonstration.
        In production, this would call your actual GPT model.
        """
        # Simulate different reasoning paths based on temperature
        if "237 × 486" in prompt or "237 * 486" in prompt:
            # Simulate occasional errors
            if self.rng.random() < 0.2:  # 20% chance of error
                return """
                Step 1: Break down 486 = 400 + 80 + 6
                Step 2: 237 × 400 = 94,800
                Step 3: 237 × 80 = 18,960
                Step 4: 237 × 6 = 1,442
                Step 5: Sum = 115,202
                Answer: 115,202
                """
            else:
                return """
                Step 1: 237 × 486
                Step 2: Break down: 237 × (500 - 14)
                Step 3: 237 × 500 = 118,500
                Step 4: 237 × 14 = 3,318
                Step 5: 118,500 - 3,318 = 115,182
                Answer: 115,182
                """

        return "Let me think step by step:\nStep 1: ...\nAnswer: [result]"


class SelfConsistencyReasoning:
    """
    Self-Consistency reasoning system.

    Generates multiple reasoning paths and uses majority voting
    to select the most reliable answer.

    DEFINITION: Self-Consistency
    A technique where you:
    1. Generate N different reasoning paths (with slight randomness)
    2. Extract the final answer from each path
    3. Count how many times each answer appears
    4. Pick the answer that appears most often
    5. Use frequency as confidence score

    C# analogy:
    public class SelfConsistencyReasoning {
        private int nSamples;

        public (string Answer, double Confidence) Solve(string question) {
            var answers = GenerateMultiplePaths(question, nSamples);
            return MajorityVote(answers);
        }
    }
    """

    def __init__(self, base_model, n_samples: int = 5):
        """
        Initialize Self-Consistency system.

        Args:
            base_model: Your GPT model (or MockGPT for testing)
            n_samples: Number of reasoning paths to generate

        GUIDELINE: Typical n_samples
        - Quick testing: 3-5
        - Production: 5-10
        - High-stakes: 10-20
        """
        self.base_model = base_model
        self.n_samples = n_samples

    def generate_multiple_paths(self, question: str) -> List[Dict]:
        """
        Generate multiple independent reasoning paths.

        Each path uses slightly different temperature (randomness)
        so they explore different approaches.

        Like running the same function with different random seeds:
        var results = Enumerable.Range(0, 5)
            .Select(i => SolveWithSeed(problem, seed: i))
            .ToList();

        Returns:
            List of dictionaries, each containing:
            - reasoning: The full reasoning text
            - answer: Extracted final answer
            - temperature: Randomness used
            - path_id: Which path this is (1, 2, 3, ...)
        """
        paths = []

        print(f"Generating {self.n_samples} reasoning paths...")

        for i in range(self.n_samples):
            # Vary temperature for diversity
            # 0.7, 0.75, 0.8, 0.85, 0.9
            temperature = 0.7 + (i * 0.05)

            print(f"  Path {i+1} (temp={temperature:.2f})...", end=" ")

            # Generate reasoning
            prompt = f"{question}\nLet's think step by step:"
            reasoning = self.base_model.generate(
                prompt=prompt,
                max_length=200,
                temperature=temperature
            )

            # Extract answer from reasoning
            answer = self._extract_answer(reasoning)

            paths.append({
                'path_id': i + 1,
                'temperature': temperature,
                'reasoning': reasoning,
                'answer': answer
            })

            print(f"Answer: {answer}")

        print()
        return paths

    def _extract_answer(self, reasoning: str) -> str:
        """
        Extract the final answer from reasoning text.

        Looks for patterns like:
        - "Answer: 42"
        - "The answer is 42"
        - "Therefore, 42"

        C# equivalent:
        private string ExtractAnswer(string reasoning) {
            var match = Regex.Match(reasoning, @"Answer:\s*(.+)");
            return match.Success ? match.Groups[1].Value.Trim() : "";
        }
        """
        lines = reasoning.strip().split('\n')

        for line in reversed(lines):  # Start from end
            line = line.strip()

            # Look for answer markers
            if any(marker in line.lower() for marker in ['answer:', 'therefore', 'result:']):
                # Remove markers
                for marker in ['Answer:', 'answer:', 'Therefore,', 'therefore,', 'Result:', 'result:']:
                    line = line.replace(marker, '')

                return line.strip()

        # If no marker found, return last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()

        return ""

    def majority_vote(self, paths: List[Dict]) -> Dict:
        """
        Perform majority voting on answers.

        Counts how many times each answer appears and picks the winner.

        Example:
        Answers: ['42', '43', '42', '42', '44']
        Counts: Counter({'42': 3, '43': 1, '44': 1})
        Winner: '42' (60% confidence)

        C# equivalent:
        private (string Answer, double Confidence) MajorityVote(List<ReasoningPath> paths) {
            var answers = paths.Select(p => p.Answer).ToList();
            var groups = answers.GroupBy(a => a)
                               .OrderByDescending(g => g.Count())
                               .ToList();
            var winner = groups.First();
            return (winner.Key, (double)winner.Count() / answers.Count);
        }
        """
        # Extract all answers
        answers = [path['answer'] for path in paths]

        # Count occurrences
        # Python Counter is like C#'s GroupBy + Count
        answer_counts = Counter(answers)

        # Find most common
        most_common_answer, vote_count = answer_counts.most_common(1)[0]

        # Calculate confidence
        confidence = vote_count / len(paths)

        # Find best reasoning path for winning answer
        best_path = next(p for p in paths if p['answer'] == most_common_answer)

        return {
            'question': "Question here",  # Would be passed in real implementation
            'final_answer': most_common_answer,
            'confidence': confidence,
            'vote_counts': dict(answer_counts),
            'all_paths': paths,
            'best_reasoning': best_path['reasoning'],
            'n_samples': len(paths)
        }

    def solve_with_consistency(self, question: str, show_paths: bool = False) -> Dict:
        """
        Main method: Solve using self-consistency.

        1. Generate multiple reasoning paths
        2. Vote on most common answer
        3. Return result with confidence
        """
        # Generate paths
        paths = self.generate_multiple_paths(question)

        # Optional: Show all paths
        if show_paths:
            print("\nAll Reasoning Paths:")
            print("=" * 60)
            for path in paths:
                print(f"\nPath {path['path_id']} (temp={path['temperature']:.2f}):")
                print(path['reasoning'])
                print(f"→ Answer: {path['answer']}")
            print("=" * 60 + "\n")

        # Perform majority voting
        result = self.majority_vote(paths)
        result['question'] = question  # Add question to result

        return result

    def display_result(self, result: Dict):
        """
        Pretty-print the self-consistency result.

        Shows:
        - All vote counts
        - Final answer
        - Confidence level
        - Visual representation
        """
        print("\n" + "=" * 60)
        print("SELF-CONSISTENCY RESULTS")
        print("=" * 60)

        print(f"\nQuestion: {result['question']}\n")

        # Display voting
        print("Voting Results:")
        print("-" * 60)
        for answer, count in sorted(result['vote_counts'].items(),
                                    key=lambda x: x[1], reverse=True):
            percentage = (count / result['n_samples']) * 100
            # Visual bar: ████░░░
            bar = "█" * count + "░" * (result['n_samples'] - count)
            print(f"{answer:30s} : {bar} ({count}/{result['n_samples']} = {percentage:.0f}%)")

        print("\n" + "-" * 60)
        print(f"Final Answer: {result['final_answer']}")
        print(f"Confidence: {result['confidence']:.1%}")

        # Interpret confidence
        if result['confidence'] >= 0.8:
            level = "VERY HIGH - Extremely reliable"
            emoji = "✓✓✓"
        elif result['confidence'] >= 0.6:
            level = "HIGH - Very reliable"
            emoji = "✓✓"
        elif result['confidence'] >= 0.4:
            level = "MEDIUM - Fairly reliable"
            emoji = "✓"
        else:
            level = "LOW - Be cautious"
            emoji = "⚠"

        print(f"Confidence Level: {level} {emoji}")
        print("=" * 60 + "\n")


class WeightedSelfConsistency(SelfConsistencyReasoning):
    """
    Enhanced self-consistency with weighted voting.

    Not all reasoning paths are equally good!
    This gives more weight to higher-quality reasoning.

    Example:
    Path 1: Detailed reasoning → Weight 2.0
    Path 2: Vague reasoning → Weight 0.5
    Path 3: Detailed reasoning → Weight 2.0

    Total weight for answer = sum of weights

    C# analogy:
    public class WeightedVote {
        public string Answer { get; set; }
        public double Weight { get; set; }
    }

    var totalWeight = votes.GroupBy(v => v.Answer)
                          .Select(g => new {
                              Answer = g.Key,
                              Weight = g.Sum(v => v.Weight)
                          })
                          .OrderByDescending(x => x.Weight)
                          .First();
    """

    def score_reasoning_quality(self, reasoning: str) -> float:
        """
        Score the quality of reasoning (0.5 to 2.0).

        Higher score = better reasoning = more trustworthy

        Criteria:
        - Number of steps (more detailed = better)
        - Mathematical operations
        - Logical connectors ("therefore", "because")

        Returns:
            Quality score (0.5 = poor, 1.0 = average, 2.0 = excellent)
        """
        score = 1.0  # Start neutral

        # Count reasoning steps
        steps = reasoning.count('Step') + reasoning.count('\n')
        if steps >= 5:
            score += 0.3  # Bonus for detail
        elif steps <= 2:
            score -= 0.3  # Penalty for vagueness

        # Check for math
        math_symbols = ['×', '÷', '+', '-', '=', '*', '/']
        math_count = sum(reasoning.count(sym) for sym in math_symbols)
        score += min(math_count * 0.1, 0.5)  # Up to +0.5

        # Check for logical connectors
        logic_words = ['therefore', 'because', 'so', 'thus', 'hence']
        if any(word in reasoning.lower() for word in logic_words):
            score += 0.2

        # Clamp to valid range
        return max(0.5, min(2.0, score))

    def weighted_majority_vote(self, paths: List[Dict]) -> Dict:
        """
        Perform weighted majority voting.

        Each path gets a weight based on its quality,
        then we sum weights per answer instead of counting votes.
        """
        # Score each path
        for path in paths:
            path['quality_score'] = self.score_reasoning_quality(path['reasoning'])

        # Calculate weighted votes per answer
        answer_weights = {}
        for path in paths:
            answer = path['answer']
            weight = path['quality_score']
            answer_weights[answer] = answer_weights.get(answer, 0.0) + weight

        # Find winner
        best_answer = max(answer_weights, key=answer_weights.get)
        total_weight = sum(answer_weights.values())
        confidence = answer_weights[best_answer] / total_weight

        # Get best path
        best_path = max(
            (p for p in paths if p['answer'] == best_answer),
            key=lambda p: p['quality_score']
        )

        return {
            'question': "Question here",
            'final_answer': best_answer,
            'confidence': confidence,
            'answer_weights': answer_weights,
            'all_paths': paths,
            'best_reasoning': best_path['reasoning'],
            'n_samples': len(paths),
            'weighted': True
        }

    def solve_with_consistency(self, question: str, show_paths: bool = False) -> Dict:
        """Override to use weighted voting."""
        paths = self.generate_multiple_paths(question)

        if show_paths:
            print("\nAll Reasoning Paths (with quality scores):")
            print("=" * 60)
            for path in paths:
                quality = self.score_reasoning_quality(path['reasoning'])
                print(f"\nPath {path['path_id']} (temp={path['temperature']:.2f}, quality={quality:.2f}):")
                print(path['reasoning'])
                print(f"→ Answer: {path['answer']}")
            print("=" * 60 + "\n")

        result = self.weighted_majority_vote(paths)
        result['question'] = question
        return result


def demo_basic_self_consistency():
    """
    Demonstrate basic self-consistency reasoning.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Self-Consistency")
    print("=" * 70 + "\n")

    # Create mock model
    gpt = MockGPT(seed=42)

    # Create self-consistency system
    sc = SelfConsistencyReasoning(gpt, n_samples=5)

    # Solve a problem
    question = "What is 237 × 486?"
    result = sc.solve_with_consistency(question, show_paths=False)

    # Display results
    sc.display_result(result)

    print("\nBest Reasoning Path:")
    print("-" * 60)
    print(result['best_reasoning'])
    print("-" * 60)


def demo_weighted_self_consistency():
    """
    Demonstrate weighted self-consistency.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Weighted Self-Consistency")
    print("=" * 70 + "\n")

    gpt = MockGPT(seed=123)
    wsc = WeightedSelfConsistency(gpt, n_samples=5)

    question = "What is 237 × 486?"
    result = wsc.solve_with_consistency(question, show_paths=False)

    # Display weighted results
    print("\n" + "=" * 60)
    print("WEIGHTED VOTING RESULTS")
    print("=" * 60)

    print(f"\nQuestion: {question}\n")

    print("Answer Weights (sum of quality scores):")
    print("-" * 60)
    for answer, weight in sorted(result['answer_weights'].items(),
                                 key=lambda x: x[1], reverse=True):
        percentage = (weight / sum(result['answer_weights'].values())) * 100
        print(f"{answer:30s} : {weight:.2f} ({percentage:.1f}%)")

    print("\n" + "-" * 60)
    print(f"Final Answer: {result['final_answer']}")
    print(f"Weighted Confidence: {result['confidence']:.1%}")
    print("=" * 60 + "\n")


def compare_regular_vs_weighted():
    """
    Compare regular vs weighted self-consistency.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Comparison - Regular vs Weighted Voting")
    print("=" * 70 + "\n")

    gpt = MockGPT(seed=999)
    question = "What is 237 × 486?"

    # Regular self-consistency
    print("Running REGULAR self-consistency...")
    sc = SelfConsistencyReasoning(gpt, n_samples=5)
    regular_result = sc.solve_with_consistency(question)

    # Reset model with same seed
    gpt = MockGPT(seed=999)

    # Weighted self-consistency
    print("\nRunning WEIGHTED self-consistency...")
    wsc = WeightedSelfConsistency(gpt, n_samples=5)
    weighted_result = wsc.solve_with_consistency(question)

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\nRegular voting:")
    print(f"  Answer: {regular_result['final_answer']}")
    print(f"  Confidence: {regular_result['confidence']:.1%}")
    print(f"\nWeighted voting:")
    print(f"  Answer: {weighted_result['final_answer']}")
    print(f"  Confidence: {weighted_result['confidence']:.1%}")

    print("\nKey difference:")
    print("  Regular: All paths count equally (1 vote each)")
    print("  Weighted: Better reasoning gets more weight")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    """
    Run all demonstrations.

    To use with your actual GPT model:

    from modules.module_06.example_01_complete_gpt import GPT, GPTConfig

    config = GPTConfig(vocab_size=50257, max_seq_len=256,
                      embed_dim=512, n_layers=6, n_heads=8)
    gpt = GPT(config)
    # gpt.load_weights('path/to/your/model.pth')

    sc = SelfConsistencyReasoning(gpt, n_samples=5)
    result = sc.solve_with_consistency("Your question here")
    sc.display_result(result)
    """

    print("\n" + "🎓" * 35)
    print("SELF-CONSISTENCY REASONING EXAMPLES")
    print("🎓" * 35)

    # Run demonstrations
    demo_basic_self_consistency()
    demo_weighted_self_consistency()
    compare_regular_vs_weighted()

    print("\n" + "✓" * 35)
    print("All demonstrations complete!")
    print("✓" * 35 + "\n")

    print("KEY TAKEAWAYS:")
    print("1. Self-consistency generates multiple reasoning paths")
    print("2. Majority voting picks the most common answer")
    print("3. Confidence = agreement percentage")
    print("4. Weighted voting gives better reasoning more influence")
    print("5. Higher n_samples = better reliability but slower")
    print("\nNext: Try with your own GPT model from Module 6!")

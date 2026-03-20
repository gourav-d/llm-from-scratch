# Lesson 8: Prompt Optimization

**Systematically improve prompts for better results, lower cost, and faster responses**

---

## 🎯 Learning Objectives

After this lesson, you will:
- Measure prompt performance objectively
- Use DSPy for automatic optimization
- A/B test prompt variations
- Optimize for cost, latency, and quality
- Build a prompt optimization pipeline

**Time:** 90 minutes

---

## 📖 What is Prompt Optimization?

### The Problem

You have a working prompt, but:
- ❌ Results are inconsistent
- ❌ Too expensive (too many tokens)
- ❌ Too slow (multiple calls needed)
- ❌ Not accurate enough

### The Solution

**Prompt optimization** = Systematically improving prompts using data and metrics

Like C# performance optimization:
```csharp
// Before optimization
public List<T> SlowMethod() {
    // Works but slow
}

// After profiling and optimization
public List<T> FastMethod() {
    // Same results, 10x faster
}
```

---

## 📊 Measuring Prompt Performance

### Key Metrics

1. **Quality Metrics**
   - Accuracy (correct vs incorrect)
   - Precision/Recall (for classification)
   - F1 Score
   - Human evaluation scores

2. **Cost Metrics**
   - Tokens per request
   - Cost per 1000 requests
   - Total monthly spend

3. **Latency Metrics**
   - Response time (ms)
   - Time to first token (streaming)
   - End-to-end latency

4. **Reliability Metrics**
   - Success rate
   - Parse error rate
   - Retry rate

### Setting Up Measurement

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class PromptMetrics:
    """
    Track prompt performance metrics.

    C#/.NET equivalent:
    public class PromptMetrics
    {
        public double Accuracy { get; set; }
        public int TokenCount { get; set; }
        public double Cost { get; set; }
        public double LatencyMs { get; set; }
    }
    """
    accuracy: float
    token_count: int
    cost: float
    latency_ms: float
    success_rate: float
    parse_error_rate: float

    @property
    def quality_score(self) -> float:
        """Composite quality score."""
        return self.accuracy * self.success_rate * (1 - self.parse_error_rate)

    @property
    def efficiency_score(self) -> float:
        """Cost-adjusted quality score."""
        return self.quality_score / (self.cost + 0.001)  # Avoid division by zero


class PromptEvaluator:
    """Evaluate prompts systematically."""

    def __init__(self, test_cases: list):
        """
        Initialize with test cases.

        Args:
            test_cases: List of (input, expected_output) tuples
        """
        self.test_cases = test_cases

    def evaluate(self, prompt_template: str) -> PromptMetrics:
        """
        Evaluate a prompt template.

        Returns:
            Metrics for this prompt
        """
        correct = 0
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0.0
        successes = 0
        parse_errors = 0

        for input_data, expected_output in self.test_cases:
            start_time = time.time()

            # Create prompt from template
            prompt = prompt_template.format(input=input_data)

            # Call LLM
            try:
                response = call_llm(prompt)
                latency = (time.time() - start_time) * 1000  # ms

                # Parse response
                parsed = parse_response(response)

                # Check correctness
                if parsed == expected_output:
                    correct += 1
                successes += 1

                # Track metrics
                total_tokens += count_tokens(prompt + response)
                total_cost += calculate_cost(prompt, response)
                total_latency += latency

            except ParseError:
                parse_errors += 1
            except Exception:
                pass  # Count as failure

        n = len(self.test_cases)
        return PromptMetrics(
            accuracy=correct / n,
            token_count=int(total_tokens / n),
            cost=total_cost / n,
            latency_ms=total_latency / n,
            success_rate=successes / n,
            parse_error_rate=parse_errors / n
        )
```

---

## 🔧 Manual Optimization Techniques

### Technique 1: Reduce Prompt Length

**Before:**
```
You are a highly skilled and experienced senior software engineer with
over 15 years of professional experience working at top tech companies
including Google, Amazon, and Microsoft. You have deep expertise in
multiple programming languages including Python, Java, C++, and JavaScript.
You are known for writing clean, efficient, well-documented code that
follows all industry best practices and design patterns.

Your task is to carefully review and analyze the following code snippet.
Please examine it thoroughly and provide detailed feedback on any issues,
problems, or areas for improvement you can identify. Be comprehensive
and specific in your analysis.

Code: {code}
```

**After (50% shorter, same quality):**
```
You are a senior software engineer reviewing code.

Analyze this code for:
- Bugs
- Security issues
- Performance problems
- Best practices violations

Code: {code}
```

**Savings:** ~50% fewer tokens, ~50% lower cost, faster response

### Technique 2: Optimize Few-Shot Examples

**Before (3 examples):**
```
Example 1: [long example]
Example 2: [long example]
Example 3: [long example]

Now classify: {input}
```

**Test different numbers:**
```python
results = {}
for num_examples in [0, 1, 2, 3, 5]:
    prompt = create_prompt_with_n_examples(num_examples)
    metrics = evaluate(prompt)
    results[num_examples] = metrics

# Find optimal number
optimal = max(results.items(), key=lambda x: x[1].efficiency_score)
```

**Finding:** Often 1-2 examples are enough, 3+ adds cost without benefit

### Technique 3: Simplify Output Format

**Before (verbose JSON):**
```json
{
  "analysis_result": {
    "classification": {
      "category": "spam",
      "confidence_level": 0.95,
      "reasoning": "This message contains..."
    }
  }
}
```

**After (minimal JSON):**
```json
{"category": "spam", "confidence": 0.95}
```

**Savings:** ~60% fewer output tokens

---

## 🤖 Automatic Optimization with DSPy

### What is DSPy?

**DSPy** = Framework for programming (not prompting) with LLMs

- Automatically optimizes prompts
- Compiles programs to optimized prompts
- Like compiler optimization for prompts

### Installing DSPy

```bash
pip install dspy-ai
```

### Basic DSPy Example

```python
import dspy

# Configure LLM
lm = dspy.OpenAI(model="gpt-4o-mini")
dspy.settings.configure(lm=lm)

# Define signature (input/output spec)
class Classify(dspy.Signature):
    """Classify email as spam or not spam."""
    email = dspy.InputField(desc="email text")
    category = dspy.OutputField(desc="spam or not_spam")

# Create predictor
classifier = dspy.Predict(Classify)

# Use it
result = classifier(email="Get rich quick!")
print(result.category)  # "spam"
```

### DSPy Optimization

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# 1. Define task
class EmailClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought('email -> category, confidence')

    def forward(self, email):
        return self.classify(email=email)

# 2. Prepare training data
trainset = [
    dspy.Example(email="Get rich quick!", category="spam", confidence="high").with_inputs('email'),
    dspy.Example(email="Meeting at 2pm", category="not_spam", confidence="high").with_inputs('email'),
    # ... more examples
]

# 3. Define metric
def accuracy_metric(example, prediction, trace=None):
    return example.category == prediction.category

# 4. Optimize
optimizer = BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=3)
optimized_classifier = optimizer.compile(EmailClassifier(), trainset=trainset)

# 5. Use optimized version
result = optimized_classifier(email="Limited time offer!")
print(result.category, result.confidence)
```

**What DSPy does:**
- Automatically selects best examples
- Optimizes prompt structure
- Tunes chain-of-thought reasoning
- Reduces unnecessary tokens

---

## 🧪 A/B Testing Prompts

### Setting Up A/B Tests

```python
class PromptVariant:
    """
    A prompt variant for A/B testing.

    Like C# experimentation framework.
    """

    def __init__(self, name: str, template: str):
        self.name = name
        self.template = template
        self.metrics = []

    def test(self, inputs: list) -> PromptMetrics:
        """Test this variant on inputs."""
        evaluator = PromptEvaluator(inputs)
        metrics = evaluator.evaluate(self.template)
        self.metrics.append(metrics)
        return metrics


class ABTester:
    """Run A/B tests on prompt variants."""

    def __init__(self):
        self.variants = {}

    def add_variant(self, name: str, template: str):
        """Add a prompt variant to test."""
        self.variants[name] = PromptVariant(name, template)

    def run_test(self, test_cases: list, traffic_split: dict = None):
        """
        Run A/B test.

        Args:
            test_cases: Test data
            traffic_split: Dict of variant_name -> percentage (0.0-1.0)

        Returns:
            Results for each variant
        """
        if traffic_split is None:
            # Equal split
            split = 1.0 / len(self.variants)
            traffic_split = {name: split for name in self.variants}

        results = {}

        for name, variant in self.variants.items():
            # Allocate test cases based on traffic split
            num_cases = int(len(test_cases) * traffic_split[name])
            variant_cases = test_cases[:num_cases]

            # Run test
            metrics = variant.test(variant_cases)
            results[name] = metrics

        return results

    def get_winner(self, results: dict, metric: str = "efficiency_score") -> str:
        """
        Determine winning variant.

        Args:
            results: Results from run_test()
            metric: Metric to optimize for

        Returns:
            Name of winning variant
        """
        return max(results.items(), key=lambda x: getattr(x[1], metric))[0]


# Usage
tester = ABTester()

tester.add_variant("control", "Classify this email: {input}")
tester.add_variant("with_role", "You are an email expert. Classify: {input}")
tester.add_variant("with_cot", "You are an email expert. Think step-by-step and classify: {input}")

results = tester.run_test(test_cases)
winner = tester.get_winner(results, metric="efficiency_score")

print(f"Winner: {winner}")
for name, metrics in results.items():
    print(f"{name}: accuracy={metrics.accuracy:.2f}, cost=${metrics.cost:.4f}")
```

---

## 💡 Optimization Strategies

### Strategy 1: Cost Optimization

**Goal:** Reduce cost while maintaining quality

**Tactics:**
1. **Use cheaper models for simple tasks**
   ```python
   # Before: GPT-4 for everything
   model = "gpt-4o"  # $0.03/1K tokens

   # After: GPT-4o-mini for simple tasks
   if task_complexity == "simple":
       model = "gpt-4o-mini"  # $0.15/1M tokens (20x cheaper!)
   else:
       model = "gpt-4o"
   ```

2. **Reduce prompt length**
   - Remove unnecessary words
   - Shorter role descriptions
   - Minimal examples

3. **Optimize output format**
   - JSON instead of prose
   - Shorter field names
   - Remove explanations if not needed

### Strategy 2: Latency Optimization

**Goal:** Faster responses

**Tactics:**
1. **Reduce tokens**
   - Shorter prompts = faster generation
   - Each token adds ~50ms

2. **Use streaming**
   ```python
   # Start processing as soon as first tokens arrive
   for chunk in call_llm_stream(prompt):
       process_partial_result(chunk)
   ```

3. **Parallel requests**
   ```python
   # Process multiple inputs concurrently
   import asyncio

   async def process_batch(inputs):
       tasks = [call_llm_async(prompt) for prompt in inputs]
       return await asyncio.gather(*tasks)
   ```

4. **Caching**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def cached_llm_call(prompt: str):
       """Cache common prompts."""
       return call_llm(prompt)
   ```

### Strategy 3: Quality Optimization

**Goal:** Higher accuracy

**Tactics:**
1. **Add chain-of-thought** (when worth the cost)
2. **Better examples** (few-shot learning)
3. **Clearer instructions**
4. **Output validation & retry**

### Strategy 4: Multi-Objective Optimization

**Goal:** Balance quality, cost, and latency

```python
def optimization_score(metrics: PromptMetrics, weights: dict) -> float:
    """
    Calculate weighted score across multiple objectives.

    Args:
        metrics: Prompt metrics
        weights: Dict of metric -> weight (must sum to 1.0)

    Example:
        weights = {
            "quality": 0.5,    # 50% weight on quality
            "cost": 0.3,       # 30% weight on cost
            "latency": 0.2     # 20% weight on latency
        }
    """
    # Normalize metrics (0-1 scale)
    norm_quality = metrics.quality_score
    norm_cost = 1.0 / (1.0 + metrics.cost)  # Lower cost = higher score
    norm_latency = 1.0 / (1.0 + metrics.latency_ms / 1000)  # Lower latency = higher score

    # Weighted sum
    score = (
        weights.get("quality", 0.0) * norm_quality +
        weights.get("cost", 0.0) * norm_cost +
        weights.get("latency", 0.0) * norm_latency
    )

    return score
```

---

## 🚀 Production Optimization Pipeline

### Complete Optimization Workflow

```python
class PromptOptimizationPipeline:
    """
    End-to-end prompt optimization pipeline.

    Workflow:
    1. Collect baseline metrics
    2. Generate variants
    3. A/B test variants
    4. Select winner
    5. Monitor in production
    6. Iterate
    """

    def __init__(self, test_cases: list):
        self.test_cases = test_cases
        self.baseline = None
        self.variants = []
        self.winner = None

    def step1_establish_baseline(self, current_prompt: str):
        """Measure current prompt performance."""
        evaluator = PromptEvaluator(self.test_cases)
        self.baseline = evaluator.evaluate(current_prompt)
        print(f"Baseline - Accuracy: {self.baseline.accuracy:.2%}, Cost: ${self.baseline.cost:.4f}")

    def step2_generate_variants(self, base_prompt: str) -> list:
        """Generate prompt variants to test."""
        variants = []

        # Variant 1: Shorter version
        short = optimize_for_length(base_prompt)
        variants.append(("short", short))

        # Variant 2: With CoT
        with_cot = add_chain_of_thought(base_prompt)
        variants.append(("with_cot", with_cot))

        # Variant 3: Better examples
        better_examples = optimize_examples(base_prompt)
        variants.append(("better_examples", better_examples))

        # Variant 4: Structured output
        structured = add_structured_output(base_prompt)
        variants.append(("structured", structured))

        self.variants = variants
        return variants

    def step3_ab_test(self):
        """Run A/B test on all variants."""
        tester = ABTester()

        # Add baseline as control
        tester.add_variant("control", self.baseline)

        # Add all variants
        for name, template in self.variants:
            tester.add_variant(name, template)

        # Run test
        results = tester.run_test(self.test_cases)

        return results

    def step4_select_winner(self, results: dict, optimization_goal: str = "efficiency"):
        """Select best variant based on goal."""
        if optimization_goal == "quality":
            metric = "quality_score"
        elif optimization_goal == "cost":
            metric = "cost"  # Lower is better, need to invert
        elif optimization_goal == "latency":
            metric = "latency_ms"  # Lower is better
        else:  # efficiency (default)
            metric = "efficiency_score"

        self.winner = max(results.items(), key=lambda x: getattr(x[1], metric))
        return self.winner

    def step5_validate_improvement(self):
        """Ensure winner is actually better than baseline."""
        name, metrics = self.winner

        improvement = {
            "accuracy": (metrics.accuracy - self.baseline.accuracy) / self.baseline.accuracy,
            "cost": (self.baseline.cost - metrics.cost) / self.baseline.cost,
            "latency": (self.baseline.latency_ms - metrics.latency_ms) / self.baseline.latency_ms
        }

        print(f"\nImprovement over baseline:")
        print(f"  Accuracy: {improvement['accuracy']:+.1%}")
        print(f"  Cost: {improvement['cost']:+.1%}")
        print(f"  Latency: {improvement['latency']:+.1%}")

        return improvement

    def step6_deploy_gradually(self, winner_template: str):
        """
        Gradual rollout of winner.

        Deploy to small percentage first, monitor, then increase.
        """
        rollout_stages = [
            (5, "Deploy to 5% of traffic"),
            (25, "Deploy to 25% of traffic"),
            (50, "Deploy to 50% of traffic"),
            (100, "Full deployment")
        ]

        for percentage, description in rollout_stages:
            print(f"\n{description}")
            # Monitor metrics at each stage
            # If metrics degrade, rollback
            input("Press Enter when ready for next stage...")
```

---

## ✅ Summary

### Key Takeaways

1. **Measure Everything**
   - Quality (accuracy)
   - Cost (tokens, $)
   - Latency (ms)
   - Reliability (success rate)

2. **Optimization Techniques**
   - Manual: Shorten, simplify, optimize examples
   - Automatic: DSPy, A/B testing
   - Multi-objective: Balance quality/cost/latency

3. **DSPy Framework**
   - Automatic prompt optimization
   - Like compiler for prompts
   - Great for complex tasks

4. **A/B Testing**
   - Test multiple variants
   - Measure objectively
   - Choose winner based on metrics

5. **Production Pipeline**
   - Baseline → Variants → Test → Select → Deploy → Monitor → Iterate

### When to Optimize

| Scenario | Priority | Technique |
|----------|----------|-----------|
| High cost | HIGH | Reduce tokens, use smaller model |
| Low accuracy | HIGH | Add CoT, better examples, DSPy |
| Slow responses | MEDIUM | Shorten prompt, parallel requests |
| Inconsistent | MEDIUM | Structured outputs, validation |

---

## 📝 Practice Exercises

1. **Measure baseline:**
   - Pick a current prompt
   - Measure quality, cost, latency
   - Document metrics

2. **Manual optimization:**
   - Create 3 variants
   - Reduce length by 30%
   - Test and compare

3. **DSPy optimization:**
   - Install DSPy
   - Optimize a classification task
   - Compare before/after

4. **A/B test:**
   - Set up testing framework
   - Run test on 5 variants
   - Select winner

5. **Full pipeline:**
   - Complete 6-step optimization
   - Deploy gradually
   - Monitor production metrics

---

**Next Lesson:** Lesson 9 - Prompt Security

**Estimated time:** 75 minutes

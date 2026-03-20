# Lesson 5: Chain-of-Thought (CoT) Prompting

**Make LLMs think step-by-step for better reasoning and accuracy**

---

## 🎯 Learning Objectives

After this lesson, you will:
- Understand chain-of-thought prompting
- Apply CoT to complex reasoning tasks
- Use zero-shot and few-shot CoT
- Combine CoT with other techniques
- Know when to use CoT vs direct prompting

**Time:** 75 minutes

---

## 📖 What is Chain-of-Thought?

### The Problem: Direct Answers Can Be Wrong

**Without CoT:**
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 tennis balls each.
   How many tennis balls does he have now?

A: 11 tennis balls.
❌ WRONG (Should be 5 + 2×3 = 11, but let's say AI makes mistake)
```

**With CoT:**
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 tennis balls each.
   How many tennis balls does he have now?

Think step by step:
A: Let's break this down:
   1. Roger starts with 5 tennis balls
   2. He buys 2 cans
   3. Each can has 3 tennis balls
   4. So he gets: 2 cans × 3 balls = 6 new balls
   5. Total = 5 + 6 = 11 tennis balls

Answer: 11 tennis balls ✅ CORRECT (with visible reasoning)
```

### The Concept

**Chain-of-Thought (CoT)** = Making the AI show its work

Like in C#, instead of:
```csharp
// Direct return
public int Calculate(int x, int y) {
    return x + y * 2;  // Why? Not clear!
}
```

Do this:
```csharp
// Show reasoning
public int Calculate(int x, int y) {
    // Step 1: Multiply y by 2
    int doubled = y * 2;

    // Step 2: Add x
    int result = x + doubled;

    // Step 3: Return result
    return result;  // Clear reasoning path!
}
```

---

## 🔑 Why Chain-of-Thought Works

### 1. **Breaks Down Complexity**

Complex problems → Smaller, manageable steps

```
Bad: "Analyze this data"
Good: "Analyze this data by:
      1. Identifying patterns
      2. Calculating statistics
      3. Drawing conclusions
      4. Recommending actions"
```

### 2. **Reduces Errors**

LLMs make fewer mistakes when thinking step-by-step

### 3. **Provides Explainability**

You can see WHERE the AI went wrong

### 4. **Improves Confidence**

Better reasoning → More reliable answers

---

## 💻 CoT Prompting Techniques

### Technique 1: Zero-Shot CoT (Simplest)

**Pattern:**
```
{question}

Let's think step by step:
```

**Example:**
```
What is 15% of 240?

Let's think step by step:
```

**Result:**
```
1. 15% means 15/100 = 0.15
2. 0.15 × 240 = ?
3. 0.15 × 240 = 36

Answer: 36
```

**When to use:** Quick, no examples needed

### Technique 2: Few-Shot CoT (More Accurate)

**Pattern:**
```
Example 1:
Q: {question1}
A: {step-by-step answer1}

Example 2:
Q: {question2}
A: {step-by-step answer2}

Now answer:
Q: {actual question}
A:
```

**Example:**
```
Example:
Q: If a store sells apples at $2 each and oranges at $3 each,
   how much for 4 apples and 3 oranges?

A: Let's calculate step by step:
   1. Apples: 4 × $2 = $8
   2. Oranges: 3 × $3 = $9
   3. Total: $8 + $9 = $17

Now answer:
Q: If books cost $15 each and magazines cost $5 each,
   how much for 2 books and 4 magazines?
```

**When to use:** More complex tasks, need higher accuracy

### Technique 3: Structured CoT

**Pattern:**
```
{question}

Analyze using this framework:
1. {step 1 description}
2. {step 2 description}
3. {step 3 description}
...
```

**Example:**
```
Should we invest in Project A or Project B?

Analyze using this framework:
1. Initial investment required
2. Expected ROI over 3 years
3. Risk factors
4. Strategic alignment
5. Final recommendation with reasoning
```

**When to use:** Business decisions, complex analysis

---

## 🎯 CoT Patterns for Different Tasks

### Pattern 1: Mathematical Reasoning

```
Problem: {math problem}

Solve step by step:
1. Identify what we're looking for
2. List the given information
3. Determine the formula or approach
4. Perform calculations
5. Verify the answer makes sense
```

### Pattern 2: Logical Reasoning

```
Scenario: {logical problem}

Reason through this:
1. State the premises
2. Identify what we need to prove/find
3. Apply logical rules
4. Draw intermediate conclusions
5. Reach final conclusion
```

### Pattern 3: Code Debugging

```
Code with error: {code}
Error message: {error}

Debug step by step:
1. Read the error message carefully
2. Identify the line number
3. Understand what the code is trying to do
4. Identify the root cause
5. Propose a fix
6. Explain why the fix works
```

### Pattern 4: Data Analysis

```
Data: {dataset}
Question: {analysis question}

Analyze step by step:
1. Understand the data structure
2. Identify relevant columns/fields
3. Apply appropriate statistical methods
4. Calculate metrics
5. Interpret results
6. Draw conclusions
```

### Pattern 5: Decision Making

```
Decision: {decision to make}
Context: {context}

Decide using this framework:
1. Define the problem clearly
2. List all options
3. Criteria for evaluation
4. Pros and cons of each option
5. Weight the criteria
6. Make recommendation
7. Justify the choice
```

---

## 🔧 Advanced CoT Techniques

### 1. Self-Consistency

Run CoT multiple times, pick most common answer:

```python
def self_consistent_cot(question: str, num_runs: int = 5):
    """
    Generate multiple CoT reasoning paths.

    C#/.NET: Like running tests multiple times to verify consistency
    """
    answers = []

    for i in range(num_runs):
        prompt = f"{question}\n\nLet's think step by step:"
        answer = call_llm(prompt, temperature=0.7)  # Some randomness
        answers.append(extract_final_answer(answer))

    # Return most common answer
    from collections import Counter
    return Counter(answers).most_common(1)[0][0]
```

### 2. Progressive CoT

Break extremely complex problems into phases:

```
Phase 1: Understand the problem
- Rephrase in your own words
- Identify key information
- State what we need to find

Phase 2: Plan the approach
- What method should we use?
- What steps are needed?
- Any potential pitfalls?

Phase 3: Execute
- Perform calculations/analysis
- Show all work
- Check each step

Phase 4: Verify
- Does the answer make sense?
- Check against edge cases
- Final validation
```

### 3. Least-to-Most Prompting

Start with simplest sub-problem, build up:

```
Problem: [Complex problem]

Let's solve this from simplest to most complex:

First, let's solve: [Easiest sub-problem]
[Solution]

Next, building on that: [Slightly harder]
[Solution]

Finally, the full problem: [Original problem]
[Solution]
```

### 4. Socratic CoT

Ask leading questions instead of giving steps:

```
Problem: {problem}

Before solving, ask yourself:
- What am I trying to find?
- What information do I have?
- What formulas or methods apply?
- What assumptions am I making?
- How can I verify my answer?

Now solve with these questions in mind:
```

---

## 💡 Real-World Examples

### Example 1: Business Decision

```
Decision: Should we launch Product X in Market Y?

Think through this systematically:

1. Market Analysis
   - Market size: What's the TAM?
   - Competition: Who are the competitors?
   - Customer needs: What problems does Product X solve?

2. Financial Viability
   - Development cost: How much to build?
   - Marketing cost: How much to promote?
   - Expected revenue: What's the potential?
   - Break-even point: When do we profit?

3. Strategic Fit
   - Company vision: Does this align?
   - Resources: Do we have capabilities?
   - Timing: Is now the right time?

4. Risk Assessment
   - What could go wrong?
   - How likely are these risks?
   - Can we mitigate them?

5. Recommendation
   Based on the above analysis...
   [Final decision with justification]
```

### Example 2: Code Architecture

```
Task: Design a user authentication system

Design step by step:

1. Requirements
   - What features needed? (login, signup, password reset)
   - What security level? (standard, high-security, multi-factor)
   - What scale? (100 users, 1M users, 100M users)

2. Component Design
   - Authentication service
   - Token management
   - Session handling
   - Database schema

3. Security Considerations
   - Password hashing (bcrypt, scrypt?)
   - Token security (JWT? Session cookies?)
   - Rate limiting
   - Encryption

4. Scalability
   - How to handle growth?
   - Caching strategy?
   - Database optimization?

5. Implementation Plan
   - Phase 1: Basic auth
   - Phase 2: OAuth integration
   - Phase 3: MFA

6. Validation
   - Unit tests
   - Integration tests
   - Security audit
```

### Example 3: Data Science Problem

```
Problem: Predict customer churn

Solve step by step:

1. Problem Definition
   - What is "churn"? (no purchase in 90 days?)
   - Why predict it? (retention campaigns)
   - Success metric? (accuracy, recall, precision?)

2. Data Exploration
   - What data available? (purchase history, demographics, behavior)
   - Data quality? (missing values, outliers)
   - Sample size? (enough for ML?)

3. Feature Engineering
   - Which features matter? (recency, frequency, monetary value)
   - Create new features? (days_since_last_purchase, avg_order_value)
   - Handle categorical data? (one-hot encoding)

4. Model Selection
   - Try logistic regression (baseline)
   - Try random forest (non-linear patterns)
   - Try gradient boosting (best performance?)

5. Training & Validation
   - Split data (train: 70%, val: 15%, test: 15%)
   - Cross-validation
   - Hyperparameter tuning

6. Evaluation
   - Which model performs best?
   - Feature importance?
   - Business impact?

7. Deployment
   - How to serve predictions?
   - Monitoring & retraining?
```

---

## 📊 When to Use CoT

### ✅ Use CoT When:

1. **Problem is complex**
   - Multiple steps required
   - Easy to make mistakes
   - Examples: Math, logic, multi-step reasoning

2. **Accuracy is critical**
   - Wrong answers have consequences
   - Need to verify reasoning
   - Examples: Financial calculations, medical advice, legal analysis

3. **Need explainability**
   - Must justify the answer
   - Stakeholders need to understand
   - Examples: Business decisions, audits

4. **Learning/teaching**
   - Want to see the reasoning process
   - Educational context
   - Examples: Tutoring, documentation

### ❌ Don't Use CoT When:

1. **Simple tasks**
   - One-step answers
   - Examples: "What is 2+2?", "Define Python"

2. **Creative tasks**
   - No "correct" reasoning path
   - Examples: Creative writing, brainstorming

3. **Speed is critical**
   - CoT adds latency (more tokens)
   - Examples: Real-time chat, quick queries

4. **Factual recall**
   - Just need a fact
   - Examples: "What is the capital of France?"

---

## 🔬 Measuring CoT Effectiveness

### Metrics to Track

```python
class CoTMetrics:
    """
    Track CoT prompting performance.

    C#/.NET: Like performance counters
    """

    def __init__(self):
        self.correct_answers = 0
        self.total_questions = 0
        self.avg_reasoning_steps = 0
        self.token_usage = 0

    def accuracy(self) -> float:
        """Calculate accuracy rate."""
        return self.correct_answers / self.total_questions if self.total_questions > 0 else 0

    def compare_with_direct(self, direct_accuracy: float) -> dict:
        """
        Compare CoT vs direct prompting.

        Returns improvement metrics.
        """
        cot_accuracy = self.accuracy()
        improvement = cot_accuracy - direct_accuracy

        return {
            "cot_accuracy": cot_accuracy,
            "direct_accuracy": direct_accuracy,
            "improvement": improvement,
            "worth_it": improvement > 0.05  # 5% improvement threshold
        }
```

### A/B Testing

```python
# Test same question with and without CoT

# Without CoT (Direct)
direct_prompt = "What is 25% of 80?"

# With CoT
cot_prompt = """
What is 25% of 80?

Let's think step by step:
"""

# Compare:
# - Accuracy
# - Token usage
# - Latency
# - Cost
```

---

## 🚀 Production Tips

### 1. Template for CoT

```python
class CoTTemplates:
    """Reusable CoT prompt templates."""

    MATH_COT = """
Problem: {problem}

Solve step by step:
1. Identify what we're looking for
2. List the given information
3. Determine the approach
4. Perform calculations
5. State the final answer

Solution:
"""

    ANALYSIS_COT = """
Analyze: {subject}

Think through this systematically:
1. Current situation
2. Key factors
3. Implications
4. Recommendations

Analysis:
"""

    @classmethod
    def math(cls, problem: str) -> str:
        return cls.MATH_COT.format(problem=problem)

    @classmethod
    def analyze(cls, subject: str) -> str:
        return cls.ANALYSIS_COT.format(subject=subject)
```

### 2. Extract Reasoning Steps

```python
import re

def extract_reasoning_and_answer(response: str) -> dict:
    """
    Parse CoT response into reasoning steps and final answer.

    C#/.NET: Like parsing structured text with regex
    """
    # Extract numbered steps
    steps = re.findall(r'\d+\.\s*(.+)', response)

    # Extract final answer (usually after "Answer:", "Result:", etc.)
    answer_match = re.search(r'(?:Answer|Result|Conclusion):\s*(.+)', response, re.IGNORECASE)
    final_answer = answer_match.group(1) if answer_match else ""

    return {
        "reasoning_steps": steps,
        "final_answer": final_answer,
        "num_steps": len(steps)
    }
```

### 3. Validate Reasoning

```python
def validate_cot_reasoning(response: str) -> bool:
    """
    Check if response actually contains step-by-step reasoning.

    Returns:
        True if response shows reasoning, False otherwise
    """
    indicators = [
        r'\d+\.',  # Numbered steps
        r'first|second|third|next|then|finally',  # Sequence words
        r'step \d+',  # "Step 1", "Step 2", etc.
        r'let\'s|we can|we need to'  # Reasoning language
    ]

    matches = sum(1 for pattern in indicators if re.search(pattern, response, re.IGNORECASE))

    # Should have at least 2 reasoning indicators
    return matches >= 2
```

---

## ✅ Summary

### Key Takeaways

1. **CoT = Show Your Work**
   - Make AI think step-by-step
   - Like commenting your code

2. **Two Main Types**
   - Zero-shot: "Let's think step by step"
   - Few-shot: Provide example reasoning

3. **Use Cases**
   - Complex problems
   - High accuracy needed
   - Explainability required
   - Teaching/learning

4. **Benefits**
   - Better accuracy
   - Fewer errors
   - Visible reasoning
   - Debuggable

5. **Tradeoffs**
   - More tokens = higher cost
   - Slower responses
   - Not needed for simple tasks

### C#/.NET Comparison

| Concept | C#/.NET Equivalent |
|---------|-------------------|
| CoT | Method with commented steps |
| Zero-shot CoT | Debug.WriteLine() at each step |
| Few-shot CoT | Example code with comments |
| Self-consistency | Running tests multiple times |
| Validation | Unit tests for logic |

---

## 📝 Practice Exercises

1. **Convert direct to CoT:**
   - Take 5 direct prompts
   - Add step-by-step reasoning
   - Compare results

2. **Create CoT templates:**
   - Math problems
   - Business decisions
   - Code debugging
   - Data analysis

3. **Measure improvement:**
   - Test with/without CoT
   - Calculate accuracy difference
   - Analyze when CoT helps most

4. **Advanced:**
   - Implement self-consistency
   - Build reasoning validator
   - Create CoT template library

---

## 🔗 Related Concepts

- **Module 7:** Reasoning models (GPT-4 with CoT built-in)
- **Lesson 6:** Tree of Thoughts (CoT on steroids)
- **Lesson 8:** Prompt optimization (when to use CoT)

---

**Next Lesson:** Lesson 6 - Tree of Thoughts

**Estimated time:** 75 minutes

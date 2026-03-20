# Lesson 6: Tree of Thoughts (ToT)

**Explore multiple reasoning paths simultaneously for complex problem-solving**

---

## 🎯 Learning Objectives

After this lesson, you will:
- Understand Tree of Thoughts framework
- Apply ToT to complex problems
- Implement ToT programmatically
- Know when ToT is worth the cost
- Combine ToT with other techniques

**Time:** 60 minutes

---

## 📖 What is Tree of Thoughts?

### Evolution of Reasoning

**Direct Prompting:**
```
Problem → Answer
```

**Chain-of-Thought:**
```
Problem → Step 1 → Step 2 → Step 3 → Answer
(Single path)
```

**Tree of Thoughts:**
```
Problem →  Step 1a → Step 2a → Answer A
        →  Step 1b → Step 2b → Answer B
        →  Step 1c → Step 2c → Answer C

Evaluate all paths, choose best
(Multiple paths explored)
```

### The Concept

**ToT** = Explore multiple reasoning paths, evaluate, and choose the best

Like in C#, instead of:
```csharp
// Single approach
public Solution Solve(Problem problem) {
    return FirstApproachThatComesToMind(problem);
}
```

Do this:
```csharp
// Multiple approaches
public Solution Solve(Problem problem) {
    var approach1 = SolveWithMethodA(problem);
    var approach2 = SolveWithMethodB(problem);
    var approach3 = SolveWithMethodC(problem);

    return EvaluateAndSelectBest(approach1, approach2, approach3);
}
```

---

## 🌳 ToT Framework

### Core Components

1. **Thought Decomposition**
   - Break problem into decision points
   - Identify where choices matter

2. **Thought Generation**
   - Generate multiple options at each point
   - Explore different approaches

3. **State Evaluation**
   - Assess each path's promise
   - Score intermediate solutions

4. **Search Strategy**
   - BFS: Breadth-first (explore all options at each level)
   - DFS: Depth-first (go deep on promising paths)
   - Beam search: Keep top K paths

---

## 💻 ToT Prompting Pattern

### Basic ToT Template

```
Problem: {problem}

Let's explore multiple approaches:

Approach 1: {method 1}
Reasoning:
- Step 1:
- Step 2:
- Conclusion:

Approach 2: {method 2}
Reasoning:
- Step 1:
- Step 2:
- Conclusion:

Approach 3: {method 3}
Reasoning:
- Step 1:
- Step 2:
- Conclusion:

Now evaluate each approach:
- Approach 1: [Score/Assessment]
- Approach 2: [Score/Assessment]
- Approach 3: [Score/Assessment]

Best approach: [Choice with justification]
Final answer: [Answer from best approach]
```

---

## 🎯 ToT for Different Problems

### Example 1: Algorithm Design

```
Problem: Sort a large dataset efficiently

Explore different sorting algorithms:

Option 1: QuickSort
Pros:
- Average O(n log n)
- In-place sorting
- Cache-friendly

Cons:
- Worst case O(n²)
- Not stable

Score: 8/10 for general use

Option 2: Merge Sort
Pros:
- Guaranteed O(n log n)
- Stable sort
- Good for linked lists

Cons:
- Requires O(n) extra space
- Not in-place

Score: 7/10 for general use

Option 3: Heap Sort
Pros:
- Guaranteed O(n log n)
- In-place sorting
- No worst case issues

Cons:
- Not stable
- Poor cache locality

Score: 6/10 for general use

Evaluation:
For a large random dataset with space constraints,
QuickSort is best. However, if stability is required,
Merge Sort is the choice.

Recommendation: QuickSort with median-of-three pivot selection
```

### Example 2: Business Strategy

```
Decision: How to enter new market?

Strategy 1: Aggressive Launch
- Massive marketing campaign
- Loss-leader pricing
- Quick market share grab

Outcomes:
+ Fast brand awareness
+ First-mover advantage
- High burn rate
- Price war risk

Risk: High | Reward: High | Timeline: 6 months

Strategy 2: Partnership Approach
- Partner with established player
- Share resources and market access
- Joint go-to-market

Outcomes:
+ Lower risk
+ Credibility through partner
- Profit sharing
- Less control

Risk: Medium | Reward: Medium | Timeline: 9 months

Strategy 3: Stealth Entry
- Targeted pilot in one region
- Learn and iterate
- Gradual expansion

Outcomes:
+ Lower cost
+ Learning opportunity
- Slow growth
- Competitor advantage

Risk: Low | Reward: Medium | Timeline: 18 months

Evaluation:
Given current cash position and market dynamics,
Strategy 2 (Partnership) offers best risk/reward balance.

Chosen Strategy: Partnership with Brand X in Region Y
```

---

## 🔧 Implementing ToT Programmatically

### Pattern 1: Simple ToT

```python
def tree_of_thoughts_simple(problem: str, num_paths: int = 3):
    """
    Simple ToT implementation.

    C#/.NET equivalent:
    public List<Solution> TreeOfThoughts(string problem, int numPaths = 3)
    """

    # Step 1: Generate multiple thought paths
    paths = []
    for i in range(num_paths):
        prompt = f"""
Problem: {problem}

Solve using approach {i+1}. Think step-by-step with this method.
"""
        solution = call_llm(prompt, temperature=0.9)  # Higher temp for diversity
        paths.append(solution)

    # Step 2: Evaluate each path
    evaluations = []
    for path in paths:
        eval_prompt = f"""
Evaluate this solution on a scale of 1-10:
- Correctness
- Completeness
- Efficiency

Solution:
{path}

Score (just the number):
"""
        score = float(call_llm(eval_prompt, temperature=0.0))
        evaluations.append(score)

    # Step 3: Select best path
    best_idx = evaluations.index(max(evaluations))
    return paths[best_idx]
```

### Pattern 2: Structured ToT

```python
class ThoughtNode:
    """
    Represents a thought in the tree.

    C#/.NET: Like a tree node class
    public class ThoughtNode
    {
        public string Content { get; set; }
        public double Score { get; set; }
        public List<ThoughtNode> Children { get; set; }
    }
    """

    def __init__(self, content: str, score: float = 0.0):
        self.content = content
        self.score = score
        self.children = []

    def add_child(self, child):
        self.children.append(child)


def tree_of_thoughts_structured(problem: str, depth: int = 3, breadth: int = 3):
    """
    Build thought tree with BFS.

    Args:
        problem: Problem to solve
        depth: How many steps to explore
        breadth: How many options at each step
    """

    # Root node
    root = ThoughtNode(content=problem)

    # Current level nodes
    current_level = [root]

    # Build tree level by level
    for level in range(depth):
        next_level = []

        for node in current_level:
            # Generate `breadth` possible next thoughts
            for i in range(breadth):
                thought_prompt = f"""
Current thinking: {node.content}

Generate next logical step (option {i+1}):
"""
                next_thought = call_llm(thought_prompt, temperature=0.8)

                # Evaluate this thought
                eval_prompt = f"""
Rate this reasoning step from 0 to 1:
{next_thought}

Score:
"""
                score = float(call_llm(eval_prompt, temperature=0.0))

                # Add to tree
                child = ThoughtNode(content=next_thought, score=score)
                node.add_child(child)
                next_level.append(child)

        # Keep only top K nodes for next level (beam search)
        K = breadth
        next_level.sort(key=lambda n: n.score, reverse=True)
        current_level = next_level[:K]

    # Find best path from root to leaf
    return find_best_path(root)


def find_best_path(root: ThoughtNode) -> list:
    """
    DFS to find highest-scoring path from root to leaf.

    C#/.NET: Like traversing a tree with LINQ
    """
    if not root.children:
        return [root]

    best_child_path = max(
        (find_best_path(child) for child in root.children),
        key=lambda path: sum(node.score for node in path)
    )

    return [root] + best_child_path
```

---

## 📊 When to Use ToT

### ✅ Use ToT When:

1. **High-stakes decisions**
   - Wrong answer is costly
   - Examples: Investment decisions, architecture choices

2. **Creative problem-solving**
   - Multiple valid approaches
   - Examples: Design, brainstorming, strategy

3. **Complex reasoning**
   - Multi-step with branching
   - Examples: Game playing, planning, optimization

4. **Uncertainty**
   - Not clear which approach is best upfront
   - Need to explore options

### ❌ Don't Use ToT When:

1. **Simple problems**
   - One obvious solution
   - ToT is overkill

2. **Low latency required**
   - ToT is slow (multiple LLM calls)
   - Examples: Real-time apps

3. **High cost sensitivity**
   - ToT uses many tokens
   - Can be 10-30x more expensive than CoT

4. **Single correct answer**
   - Factual questions
   - Examples: "What is the capital of France?"

---

## 🚀 ToT Strategies

### Strategy 1: Breadth-First Search (BFS)

Explore all options at each level before going deeper:

```python
def tot_bfs(problem: str, max_depth: int = 3):
    """
    BFS approach: Explore all branches equally.

    Like C# queue-based BFS:
    var queue = new Queue<Node>();
    """
    from collections import deque

    queue = deque([(problem, 0)])  # (state, depth)
    solutions = []

    while queue:
        state, depth = queue.popleft()

        if depth == max_depth:
            solutions.append(state)
            continue

        # Generate next thoughts
        next_thoughts = generate_next_thoughts(state)

        for thought in next_thoughts:
            queue.append((thought, depth + 1))

    # Evaluate and return best
    return evaluate_and_select_best(solutions)
```

### Strategy 2: Depth-First Search (DFS)

Go deep on promising paths first:

```python
def tot_dfs(problem: str, max_depth: int = 3):
    """
    DFS approach: Go deep on best path.

    Like C# recursive DFS:
    void DFS(Node node, int depth)
    """
    def dfs_helper(state: str, depth: int):
        if depth == max_depth:
            return [state]

        # Generate and score next thoughts
        next_thoughts = generate_next_thoughts(state)
        scored = [(thought, score_thought(thought)) for thought in next_thoughts]

        # Sort by score, take best
        scored.sort(key=lambda x: x[1], reverse=True)
        best_thought = scored[0][0]

        # Recurse on best path
        return [state] + dfs_helper(best_thought, depth + 1)

    return dfs_helper(problem, 0)
```

### Strategy 3: Beam Search

Keep top K paths at each level:

```python
def tot_beam_search(problem: str, beam_width: int = 3, max_depth: int = 3):
    """
    Beam search: Keep top K paths.

    Balances breadth and depth.
    """
    beams = [(problem, 0.0)]  # (state, cumulative_score)

    for depth in range(max_depth):
        candidates = []

        for state, score in beams:
            # Generate next thoughts
            next_thoughts = generate_next_thoughts(state)

            for thought in next_thoughts:
                thought_score = score_thought(thought)
                candidates.append((thought, score + thought_score))

        # Keep top K
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    # Return best final state
    return beams[0][0]
```

---

## 💡 Real-World Example: Code Review

```
Code Review with ToT:

Problem: Review this pull request for potential issues

Angle 1: Security Perspective
- Check for SQL injection
- Validate input sanitization
- Review authentication/authorization
Assessment: Found 2 HIGH severity issues

Angle 2: Performance Perspective
- Analyze query efficiency
- Check for N+1 problems
- Review caching strategy
Assessment: Found 1 MEDIUM issue (missing index)

Angle 3: Maintainability Perspective
- Code readability
- Documentation quality
- Test coverage
Assessment: Good overall, minor improvements suggested

Angle 4: Business Logic Perspective
- Correctness of implementation
- Edge cases handling
- Error handling
Assessment: 1 CRITICAL bug in edge case

Synthesize:
Priority order:
1. CRITICAL: Fix business logic bug
2. HIGH: Address security issues
3. MEDIUM: Add database index
4. LOW: Improve documentation

Recommendation: Block merge until 1 & 2 are fixed
```

---

## 🔬 Measuring ToT Effectiveness

```python
class ToTMetrics:
    """Track ToT performance."""

    def __init__(self):
        self.tot_correct = 0
        self.cot_correct = 0
        self.direct_correct = 0
        self.total = 0
        self.tot_cost = 0.0
        self.cot_cost = 0.0
        self.direct_cost = 0.0

    def compare_approaches(self) -> dict:
        """
        Compare ToT vs CoT vs Direct.

        Returns improvement metrics.
        """
        return {
            "tot_accuracy": self.tot_correct / self.total,
            "cot_accuracy": self.cot_correct / self.total,
            "direct_accuracy": self.direct_correct / self.total,
            "tot_cost_per_query": self.tot_cost / self.total,
            "cot_cost_per_query": self.cot_cost / self.total,
            "direct_cost_per_query": self.direct_cost / self.total,
            "tot_worth_it": self.is_tot_worth_it()
        }

    def is_tot_worth_it(self) -> bool:
        """
        Is ToT worth the extra cost?

        ToT worth it if:
        - Accuracy improvement > 10% AND
        - Cost increase < 10x
        """
        accuracy_gain = (self.tot_correct - self.cot_correct) / self.total
        cost_ratio = self.tot_cost / self.cot_cost if self.cot_cost > 0 else float('inf')

        return accuracy_gain > 0.10 and cost_ratio < 10.0
```

---

## ✅ Summary

### Key Takeaways

1. **ToT = Multiple Reasoning Paths**
   - Explore different approaches
   - Evaluate and select best

2. **Three Search Strategies**
   - BFS: Explore all equally
   - DFS: Go deep on best
   - Beam: Balanced approach

3. **Use Cases**
   - High-stakes decisions
   - Creative problem-solving
   - Complex reasoning with uncertainty

4. **Tradeoffs**
   - Much more expensive (10-30x CoT)
   - Much slower (multiple LLM calls)
   - Higher accuracy for complex problems

5. **Implementation**
   - Can be implemented programmatically
   - Use beam search for efficiency
   - Track metrics to justify cost

### When to Use What?

| Problem Type | Approach | Why |
|--------------|----------|-----|
| Simple fact | Direct | Fast, cheap |
| Math problem | CoT | Accuracy matters |
| Complex decision | ToT | Multiple valid paths |
| Creative task | ToT | Explore possibilities |
| Real-time query | Direct/CoT | Speed matters |

---

## 📝 Practice Exercises

1. **Implement simple ToT:**
   - Generate 3 paths
   - Evaluate each
   - Select best

2. **Compare ToT vs CoT:**
   - Same problem
   - Measure accuracy, cost, time
   - Determine when ToT is worth it

3. **Build ToT template:**
   - For your domain
   - With evaluation criteria
   - Test on real problems

4. **Advanced:**
   - Implement beam search
   - Build ToT metrics tracker
   - Create cost-benefit analyzer

---

**Next Lesson:** Lesson 7 - Structured Outputs

**Estimated time:** 60 minutes

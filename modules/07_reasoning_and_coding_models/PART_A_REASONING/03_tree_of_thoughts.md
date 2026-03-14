# Lesson 7.3: Tree-of-Thoughts (ToT)

## 🎯 Learning Objectives

By the end of this lesson, you'll be able to:

- Understand the limitations of linear reasoning (Chain-of-Thought)
- Build tree-based reasoning structures
- Implement search algorithms (BFS and DFS) for reasoning
- Evaluate and prune bad reasoning branches
- Solve complex problems that require exploration
- Connect ToT to game-playing AI (like chess engines)
- Understand how advanced AI systems explore solution spaces

---

## 🤔 What is Tree-of-Thoughts?

### The Problem: Linear Reasoning Can Get Stuck

**Chain-of-Thought is linear** - it follows one path from start to finish:

```
Problem → Step 1 → Step 2 → Step 3 → Answer

Example:
"How do I get from A to B?"

CoT Path:
Step 1: Go north
Step 2: Go east
Step 3: Oops, hit a wall! ✗

STUCK! Can't backtrack or try different approaches.
```

**What if Step 1 was wrong? CoT can't explore alternatives!**

---

### The Solution: Think Like a Tree, Not a Line

**Tree-of-Thoughts explores multiple possibilities at each step:**

```
DEFINITION: Tree-of-Thoughts (ToT)
A reasoning approach where the AI:
1. Generates multiple possible next steps (branches)
2. Evaluates each possibility
3. Explores the most promising branches
4. Backtracks if a branch fails
5. Continues until finding the best solution

Think of it like a chess player thinking several moves ahead!
```

**Visual comparison:**

```
Chain-of-Thought (Linear):
Problem → Step 1 → Step 2 → Step 3 → Answer
            |
         (stuck if wrong!)

Tree-of-Thoughts (Branching):
                Problem
                   |
        ┌──────────┼──────────┐
        ↓          ↓          ↓
     Idea A     Idea B     Idea C
        |          |          |
    ┌───┼───┐  ┌──┼──┐      ✗ (dead end, prune)
    ↓   ↓   ↓  ↓     ↓
   A1  A2  A3  B1    B2
   ✗   ✓   ✗   ✗     ✓
      (found!)      (also found!)

Result: Compare A2 vs B2, pick the best!
```

**Like exploring a maze with multiple paths!**

---

## 🌍 Real-World Analogy

Think of Tree-of-Thoughts like planning a trip:

### Linear Thinking (CoT):
```
You're planning a trip from New York to San Francisco.

Linear approach:
"I'll drive west."
→ Start driving
→ Hit the Rocky Mountains
→ Car breaks down
→ FAILED! Can't backtrack or try a different route.
```

### Tree Thinking (ToT):
```
You're planning a trip from New York to San Francisco.

Tree approach:
Consider multiple options:
├─ Option A: Drive (5 days)
│  ├─ Route A1: Through Chicago → Too slow ✗
│  ├─ Route A2: Through Denver → Possible ✓
│  └─ Route A3: Through Dallas → Possible ✓
│
├─ Option B: Fly (5 hours)
│  ├─ Airline B1: Direct flight → Best! ✓✓✓
│  └─ Airline B2: Two stops → Slower ✗
│
└─ Option C: Train (4 days)
   └─ Only one route → Possible ✓

Evaluate all paths:
- Best: B1 (fly direct) - 5 hours ✓✓✓
- Backup: A2 or A3 (drive through Denver/Dallas)

You pick the best option after exploring all possibilities!
```

**Tree-of-Thoughts = Exploring all options before committing!**

---

## 📚 How Tree-of-Thoughts Works

### Core Concepts

```
DEFINITION: Thought
A partial solution or intermediate step in the reasoning process.

Example for "Solve 8-puzzle":
- Thought 1: "Move tile left"
- Thought 2: "Move tile down"
- Thought 3: "Move tile right"

Each thought leads to a new state of the problem.
```

```
DEFINITION: State
The current situation after applying a thought.

Example:
Initial state:  [1 2 3]
                [4 5 6]
                [7 8 _]

After "move tile down":
New state:      [1 2 3]
                [4 5 _]
                [7 8 6]
```

```
DEFINITION: Branch
A sequence of thoughts forming one possible solution path.

Example:
Branch 1: Move left → Move down → Move right → GOAL! ✓
Branch 2: Move up → Move left → STUCK ✗
```

```
DEFINITION: Pruning
Stopping exploration of a branch that won't lead to a good solution.

Example:
If a branch leads to the same state we've seen before,
prune it (stop exploring) to save time.

C# analogy:
if (visitedStates.Contains(newState)) {
    return; // Don't explore this path
}
```

---

### The ToT Algorithm

**High-level process:**

```
Algorithm: Tree-of-Thoughts

Input: Problem to solve
Output: Best solution

1. Start with initial state (the problem)
2. Generate possible next thoughts (actions to try)
3. For each thought:
   a. Apply thought → get new state
   b. Evaluate: Is this state promising?
   c. If promising: Add to exploration queue
   d. If bad: Prune (ignore this branch)
4. Pick most promising state
5. Repeat steps 2-4 until solution found
6. Return best solution

This is like Breadth-First Search (BFS) or Depth-First Search (DFS)!
```

**C# analogy (like graph search):**
```csharp
public class TreeOfThoughts {
    public Solution Solve(Problem problem) {
        var queue = new Queue<State>();
        queue.Enqueue(problem.InitialState);
        var visited = new HashSet<State>();

        while (queue.Count > 0) {
            var state = queue.Dequeue();

            if (state.IsGoal()) {
                return state.Solution;
            }

            foreach (var thought in GenerateThoughts(state)) {
                var newState = ApplyThought(state, thought);

                if (IsPromising(newState) && !visited.Contains(newState)) {
                    queue.Enqueue(newState);
                    visited.Add(newState);
                }
            }
        }

        return null; // No solution found
    }
}
```

---

## 💻 Implementation in Python

### Part 1: Basic Tree-of-Thoughts Framework

```python
from typing import List, Dict, Optional, Set
from collections import deque
import numpy as np

class ThoughtNode:
    """
    Represents one node (state) in the reasoning tree.

    DEFINITION - Node:
    A point in the tree representing a partial solution.
    Contains:
    - The current state
    - The thought that led here
    - The parent node (where we came from)
    - The score (how good this state is)

    C# equivalent:
    public class ThoughtNode {
        public string State { get; set; }
        public string Thought { get; set; }
        public ThoughtNode Parent { get; set; }
        public double Score { get; set; }
        public List<ThoughtNode> Children { get; set; }
    }
    """

    def __init__(self, state: str, thought: str = "", parent: Optional['ThoughtNode'] = None, score: float = 0.0):
        """
        Initialize a thought node.

        Args:
            state: The current state (e.g., "x = 5, y = 10")
            thought: The reasoning that led to this state (e.g., "Set x = 5")
            parent: The previous node in the path (None for root)
            score: How promising this state is (0.0 to 1.0)

        C# analogy:
        public ThoughtNode(string state, string thought = "", ThoughtNode parent = null, double score = 0.0) {
            this.State = state;
            this.Thought = thought;
            this.Parent = parent;
            this.Score = score;
            this.Children = new List<ThoughtNode>();
        }
        """
        self.state = state
        self.thought = thought
        self.parent = parent
        self.score = score
        self.children = []  # Like List<ThoughtNode> in C#
        self.depth = 0 if parent is None else parent.depth + 1

    def add_child(self, child: 'ThoughtNode'):
        """Add a child node."""
        self.children.append(child)

    def get_path(self) -> List[str]:
        """
        Get the full reasoning path from root to this node.

        Returns:
            List of thoughts that led to this state

        C# equivalent:
        public List<string> GetPath() {
            var path = new List<string>();
            var current = this;
            while (current.Parent != null) {
                path.Insert(0, current.Thought);
                current = current.Parent;
            }
            return path;
        }
        """
        path = []
        current = self
        while current.parent is not None:
            if current.thought:  # Don't include empty root thought
                path.insert(0, current.thought)  # Insert at beginning
            current = current.parent
        return path

    def __repr__(self):
        """String representation for debugging."""
        return f"Node(state='{self.state[:30]}...', score={self.score:.2f}, depth={self.depth})"


class TreeOfThoughts:
    """
    Tree-of-Thoughts reasoning system.

    This implements a search algorithm over possible reasoning paths,
    like BFS/DFS for finding the best solution.

    C# analogy:
    public class TreeOfThoughts {
        private GPTModel model;
        private int maxDepth;
        private int branchingFactor;

        public Solution Search(Problem problem) {
            // Explore reasoning tree using BFS or DFS
        }
    }
    """

    def __init__(self, base_model, max_depth: int = 5, branching_factor: int = 3, search_method: str = 'bfs'):
        """
        Initialize Tree-of-Thoughts system.

        Args:
            base_model: Your GPT model from Module 6
            max_depth: Maximum depth of the tree (how many steps)
            branching_factor: How many branches to explore at each step
            search_method: 'bfs' (breadth-first) or 'dfs' (depth-first)

        DEFINITION - Max depth:
        The maximum number of reasoning steps allowed.
        Example: max_depth=5 means up to 5 steps in the reasoning chain.

        DEFINITION - Branching factor:
        How many different thoughts to generate at each step.
        Example: branching_factor=3 means try 3 different ideas at each node.

        DEFINITION - BFS vs DFS:
        - BFS (Breadth-First Search): Explore all options at current level before going deeper
          Good for: Finding shortest solution
        - DFS (Depth-First Search): Go deep into one path before trying others
          Good for: Finding any solution quickly
        """
        self.base_model = base_model
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.search_method = search_method.lower()

    def generate_thoughts(self, state: str, problem: str, num_thoughts: int) -> List[str]:
        """
        Generate possible next thoughts (actions) from current state.

        DEFINITION - Thought generation:
        Ask the model: "Given the current state, what are the possible next steps?"

        Example:
        State: "x = 5"
        Problem: "Find x + y = 10"
        Generated thoughts:
        1. "Since x = 5 and x + y = 10, solve for y"
        2. "Substitute x = 5 into equation"
        3. "Calculate 10 - 5 = y"

        C# equivalent:
        private List<string> GenerateThoughts(string state, string problem, int count) {
            var prompt = $"Current state: {state}\nProblem: {problem}\nWhat are {count} possible next steps?";
            var response = model.Generate(prompt);
            return ParseThoughtsFromResponse(response);
        }

        Args:
            state: Current reasoning state
            problem: The original problem
            num_thoughts: How many thoughts to generate

        Returns:
            List of possible next thoughts
        """
        # Create prompt asking for next possible steps
        prompt = f"""
Problem: {problem}

Current state: {state}

Generate {num_thoughts} possible next reasoning steps.
Format: One step per line, numbered.

1."""

        # Generate thoughts
        response = self.base_model.generate(
            prompt=prompt,
            max_length=150,
            temperature=0.8,  # Higher temperature for diversity
            top_p=0.9
        )

        # Parse thoughts from response
        thoughts = self._parse_thoughts(response, num_thoughts)
        return thoughts

    def _parse_thoughts(self, response: str, expected_count: int) -> List[str]:
        """
        Parse numbered thoughts from model response.

        Example input:
        "1. Try approach A
         2. Try approach B
         3. Try approach C"

        Output: ["Try approach A", "Try approach B", "Try approach C"]

        C# equivalent:
        private List<string> ParseThoughts(string response, int expectedCount) {
            return response.Split('\n')
                          .Where(line => Regex.IsMatch(line, @"^\d+\."))
                          .Select(line => Regex.Replace(line, @"^\d+\.\s*", ""))
                          .Take(expectedCount)
                          .ToList();
        }
        """
        lines = response.strip().split('\n')
        thoughts = []

        for line in lines:
            line = line.strip()
            # Look for numbered lines: "1. ...", "2. ...", etc.
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                thought = line.lstrip('0123456789.-) ').strip()
                if thought:
                    thoughts.append(thought)

        # Return requested number of thoughts
        return thoughts[:expected_count]

    def evaluate_state(self, state: str, problem: str) -> float:
        """
        Evaluate how promising a state is (0.0 to 1.0).

        DEFINITION - State evaluation:
        Assign a score indicating how likely this state will lead to a solution.
        - 1.0 = Very promising, likely to solve the problem
        - 0.5 = Uncertain
        - 0.0 = Dead end, won't lead to solution

        Like heuristic function in A* search!

        Example:
        Problem: "Find x where x² = 16"
        State 1: "x = 4" → Score: 0.9 (looks correct!)
        State 2: "x = 7" → Score: 0.2 (probably wrong)

        C# equivalent:
        private double EvaluateState(string state, string problem) {
            var prompt = $"Problem: {problem}\nState: {state}\nHow likely is this to lead to the solution? (0-1):";
            var response = model.Generate(prompt);
            return ParseScore(response);
        }

        Args:
            state: Current reasoning state
            problem: Original problem

        Returns:
            Score from 0.0 (bad) to 1.0 (excellent)
        """
        # Ask model to evaluate this state
        prompt = f"""
Problem: {problem}

Current reasoning state: {state}

How promising is this state for solving the problem?
Rate from 0.0 (dead end) to 1.0 (very promising).
Just give a number between 0 and 1.

Score:"""

        response = self.base_model.generate(
            prompt=prompt,
            max_length=10,
            temperature=0.3  # Lower temperature for consistent evaluation
        )

        # Extract score
        try:
            score = float(response.strip().split()[0])
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except:
            score = 0.5  # Default if parsing fails

        return score

    def is_goal_state(self, state: str, problem: str) -> bool:
        """
        Check if this state solves the problem.

        DEFINITION - Goal state:
        A state that represents a complete solution to the problem.

        Example:
        Problem: "Find x where x + 5 = 10"
        State: "x = 5, verified: 5 + 5 = 10 ✓" → Is goal state!

        C# equivalent:
        private bool IsGoalState(string state, string problem) {
            var prompt = $"Problem: {problem}\nState: {state}\nIs this a complete solution? (yes/no)";
            var response = model.Generate(prompt).ToLower();
            return response.Contains("yes");
        }

        Args:
            state: Current state
            problem: Original problem

        Returns:
            True if state solves the problem
        """
        # Ask model if this is a complete solution
        prompt = f"""
Problem: {problem}

Current state: {state}

Is this a complete solution to the problem? Answer only 'yes' or 'no'.

Answer:"""

        response = self.base_model.generate(
            prompt=prompt,
            max_length=5,
            temperature=0.1  # Very low temperature for consistent yes/no
        )

        return 'yes' in response.lower()

    def breadth_first_search(self, problem: str, initial_state: str = "") -> Optional[ThoughtNode]:
        """
        Explore reasoning tree using Breadth-First Search.

        DEFINITION - Breadth-First Search (BFS):
        Explore all nodes at depth 1, then all at depth 2, etc.

        Visualization:
        Level 0:           Root
                          /  |  \
        Level 1:        A   B   C     ← Explore all of these first
                       /|   |   |\
        Level 2:      D E   F   G H   ← Then explore all of these
                      |
        Level 3:      I                ← Then this

        Guarantees: Finds shallowest (shortest) solution

        C# equivalent:
        private ThoughtNode BreadthFirstSearch(string problem, string initialState) {
            var queue = new Queue<ThoughtNode>();
            queue.Enqueue(new ThoughtNode(initialState));

            while (queue.Count > 0) {
                var node = queue.Dequeue();
                if (IsGoal(node)) return node;

                foreach (var child in ExpandNode(node)) {
                    queue.Enqueue(child);
                }
            }
            return null;
        }

        Args:
            problem: The problem to solve
            initial_state: Starting state (empty if starting from scratch)

        Returns:
            Goal node if found, None otherwise
        """
        # Create root node
        root = ThoughtNode(state=initial_state if initial_state else "Starting to solve problem")

        # BFS uses a queue (FIFO - First In First Out)
        # Like Queue<ThoughtNode> in C#
        queue = deque([root])
        visited = set()  # Track visited states to avoid cycles

        print(f"Starting BFS for: {problem}\n")

        while queue:
            # Dequeue first node
            current = queue.popleft()  # Remove from front (FIFO)

            print(f"Exploring depth {current.depth}: {current.state[:50]}...")

            # Check if this is the goal
            if self.is_goal_state(current.state, problem):
                print(f"\n✓ Solution found at depth {current.depth}!")
                return current

            # Don't go deeper than max_depth
            if current.depth >= self.max_depth:
                continue

            # Skip if we've seen this state
            if current.state in visited:
                continue
            visited.add(current.state)

            # Generate possible next thoughts
            thoughts = self.generate_thoughts(
                state=current.state,
                problem=problem,
                num_thoughts=self.branching_factor
            )

            # Create child nodes for each thought
            for thought in thoughts:
                # New state = current state + this thought
                new_state = f"{current.state}\n{thought}"

                # Evaluate how promising this state is
                score = self.evaluate_state(new_state, problem)

                # Create child node
                child = ThoughtNode(
                    state=new_state,
                    thought=thought,
                    parent=current,
                    score=score
                )
                current.add_child(child)

                # Add to queue if promising enough (prune bad branches)
                if score >= 0.3:  # Threshold for pruning
                    queue.append(child)  # Add to back (FIFO)
                else:
                    print(f"  ✗ Pruned (score {score:.2f}): {thought[:40]}...")

        print("\n✗ No solution found within depth limit")
        return None

    def depth_first_search(self, problem: str, initial_state: str = "") -> Optional[ThoughtNode]:
        """
        Explore reasoning tree using Depth-First Search.

        DEFINITION - Depth-First Search (DFS):
        Go deep into one path before exploring others.

        Visualization:
        Start:           Root
                          |
        Go deep:          A      ← Explore this path fully first
                          |
                          D      ← Keep going deep
                          |
                          I      ← Until end, then backtrack

        Then try:        Root
                          |
                          B      ← Now explore this path
                          |
                          F

        Faster: Finds a solution quickly (but might not be shortest)

        C# equivalent:
        private ThoughtNode DepthFirstSearch(string problem, string initialState) {
            var stack = new Stack<ThoughtNode>();
            stack.Push(new ThoughtNode(initialState));

            while (stack.Count > 0) {
                var node = stack.Pop();
                if (IsGoal(node)) return node;

                foreach (var child in ExpandNode(node)) {
                    stack.Push(child);  // LIFO - Last In First Out
                }
            }
            return null;
        }

        Args:
            problem: The problem to solve
            initial_state: Starting state

        Returns:
            Goal node if found, None otherwise
        """
        # Create root node
        root = ThoughtNode(state=initial_state if initial_state else "Starting to solve problem")

        # DFS uses a stack (LIFO - Last In First Out)
        # Like Stack<ThoughtNode> in C#
        stack = [root]
        visited = set()

        print(f"Starting DFS for: {problem}\n")

        while stack:
            # Pop last node (LIFO)
            current = stack.pop()  # Remove from end

            print(f"Exploring depth {current.depth}: {current.state[:50]}...")

            # Check if goal
            if self.is_goal_state(current.state, problem):
                print(f"\n✓ Solution found at depth {current.depth}!")
                return current

            # Depth limit
            if current.depth >= self.max_depth:
                continue

            # Skip visited
            if current.state in visited:
                continue
            visited.add(current.state)

            # Generate thoughts
            thoughts = self.generate_thoughts(
                state=current.state,
                problem=problem,
                num_thoughts=self.branching_factor
            )

            # Add children (in reverse order for DFS)
            for thought in reversed(thoughts):  # Reverse for consistent ordering
                new_state = f"{current.state}\n{thought}"
                score = self.evaluate_state(new_state, problem)

                child = ThoughtNode(
                    state=new_state,
                    thought=thought,
                    parent=current,
                    score=score
                )
                current.add_child(child)

                # Add to stack if promising
                if score >= 0.3:
                    stack.append(child)  # Add to end (LIFO)
                else:
                    print(f"  ✗ Pruned: {thought[:40]}...")

        print("\n✗ No solution found")
        return None

    def solve(self, problem: str, initial_state: str = "") -> Optional[Dict]:
        """
        Solve a problem using Tree-of-Thoughts.

        Args:
            problem: The problem to solve
            initial_state: Optional starting state

        Returns:
            Dictionary with solution and reasoning path
        """
        # Choose search method
        if self.search_method == 'bfs':
            goal_node = self.breadth_first_search(problem, initial_state)
        else:  # dfs
            goal_node = self.depth_first_search(problem, initial_state)

        if goal_node is None:
            return None

        # Extract solution path
        path = goal_node.get_path()

        return {
            'problem': problem,
            'solution': goal_node.state,
            'reasoning_path': path,
            'depth': goal_node.depth,
            'score': goal_node.score
        }
```

**Key line-by-line explanations:**

```python
queue = deque([root])
```
- **What:** Creates a double-ended queue (pronounced "deck")
- **Why:** Efficient for BFS (add to back, remove from front)
- **C# equivalent:** `var queue = new Queue<ThoughtNode>();`
- **deque:** Python's efficient queue/stack implementation

```python
current = queue.popleft()
```
- **What:** Remove and return first element (FIFO - First In First Out)
- **Why:** BFS explores in order of depth
- **C# equivalent:** `var current = queue.Dequeue();`

```python
stack.pop()
```
- **What:** Remove and return last element (LIFO - Last In First Out)
- **Why:** DFS goes deep before wide
- **C# equivalent:** `var current = stack.Pop();`

```python
if score >= 0.3:
```
- **What:** Pruning threshold
- **Why:** Don't explore branches unlikely to succeed
- **Saves:** Lots of computation by avoiding dead ends
- **Like:** Early return in C# to avoid unnecessary work

---

## 🧪 Complete Example: Game of 24 Solver

```python
class GameOf24Solver:
    """
    Solve the Game of 24 using Tree-of-Thoughts.

    GAME RULES:
    Given 4 numbers, use +, -, ×, ÷ to make 24.
    Each number used exactly once.

    Example: [3, 3, 8, 8]
    Solution: 8 ÷ (3 - 8 ÷ 3) = 24

    This is perfect for ToT because:
    - Multiple possible operations to try at each step
    - Need to explore different combinations
    - Some paths lead to dead ends
    - Need to backtrack and try different approaches
    """

    def __init__(self, base_model):
        self.base_model = base_model
        self.tot = TreeOfThoughts(
            base_model=base_model,
            max_depth=4,  # 4 numbers → 3 operations → depth 4
            branching_factor=4,  # Try multiple operations at each step
            search_method='bfs'  # BFS to find shortest solution
        )

    def solve(self, numbers: List[int]) -> Optional[Dict]:
        """
        Solve Game of 24.

        Args:
            numbers: List of 4 numbers

        Returns:
            Solution dict or None
        """
        problem = f"Use the numbers {numbers} with operations +, -, ×, ÷ to make 24. Each number used exactly once."
        initial_state = f"Available numbers: {numbers}"

        print(f"\n{'='*60}")
        print(f"GAME OF 24: {numbers}")
        print(f"{'='*60}\n")

        result = self.tot.solve(problem, initial_state)

        if result:
            print(f"\n{'='*60}")
            print("SOLUTION FOUND!")
            print(f"{'='*60}")
            print("\nReasoning Path:")
            for i, step in enumerate(result['reasoning_path'], 1):
                print(f"{i}. {step}")
            print(f"\nFinal: {result['solution']}")
            print(f"{'='*60}\n")

        return result

# Usage
from modules.module_06.example_01_complete_gpt import GPT, GPTConfig

config = GPTConfig(vocab_size=50257, max_seq_len=256, embed_dim=512, n_layers=6, n_heads=8)
gpt = GPT(config)

solver = GameOf24Solver(gpt)
result = solver.solve([3, 3, 8, 8])
```

**Expected output:**

```
============================================================
GAME OF 24: [3, 3, 8, 8]
============================================================

Starting BFS for: Use the numbers [3, 3, 8, 8]...

Exploring depth 0: Available numbers: [3, 3, 8, 8]...
Exploring depth 1: Try 8 ÷ 3 = 8/3...
Exploring depth 1: Try 8 - 3 = 5...
Exploring depth 1: Try 3 × 8 = 24...  ← Too early, not using all numbers
  ✗ Pruned (score 0.2): Try 3 + 3 = 6...
Exploring depth 2: Use 8/3 with remaining 3, 8...
Exploring depth 2: Calculate 3 - 8/3 = 9/3 - 8/3 = 1/3...
Exploring depth 3: Calculate 8 ÷ (1/3) = 8 × 3 = 24...

✓ Solution found at depth 3!

============================================================
SOLUTION FOUND!
============================================================

Reasoning Path:
1. Calculate 8 ÷ 3 = 8/3 (using first 8 and first 3)
2. Calculate 3 - 8/3 = 1/3 (using second 3)
3. Calculate 8 ÷ (1/3) = 24 (using second 8)

Final: 8 ÷ (3 - 8 ÷ 3) = 24 ✓
============================================================
```

---

## 📊 BFS vs DFS Comparison

```
BREADTH-FIRST SEARCH (BFS):
Queue (FIFO): First In, First Out

       Root
      / | \
    A   B  C    ← Explore ALL at depth 1
   /|   |   |\
  D E   F   G H ← Then ALL at depth 2

Order: Root → A → B → C → D → E → F → G → H

Pros:
✓ Finds shortest solution
✓ Guaranteed to find solution if exists
✓ Good when solution is near the top

Cons:
✗ Uses more memory (stores all nodes at current level)
✗ Slower if solution is deep
```

```
DEPTH-FIRST SEARCH (DFS):
Stack (LIFO): Last In, First Out

       Root
        |
        A      ← Go deep into this path
        |
        D      ← Keep going until end
        |
        I      ← Then backtrack

Order: Root → A → D → I → (backtrack) → E → (backtrack) → B → F

Pros:
✓ Uses less memory (only stores current path)
✓ Faster if solution is deep
✓ Good for "find any solution"

Cons:
✗ Might not find shortest solution
✗ Can get stuck in deep branches
```

**When to use which:**

```
Use BFS when:
- Want shortest/optimal solution
- Solution is likely near the top
- Memory is available
Example: Game of 24 (shortest solution is elegant)

Use DFS when:
- Any solution is acceptable
- Solution is likely deep
- Memory is limited
Example: Maze solving (just need to find exit)
```

**C# analogy:**
```csharp
// BFS - like level-order tree traversal
public void BreadthFirst() {
    var queue = new Queue<Node>();
    queue.Enqueue(root);
    while (queue.Count > 0) {
        var node = queue.Dequeue();  // FIFO
        Process(node);
        foreach (var child in node.Children) {
            queue.Enqueue(child);
        }
    }
}

// DFS - like pre-order tree traversal
public void DepthFirst() {
    var stack = new Stack<Node>();
    stack.Push(root);
    while (stack.Count > 0) {
        var node = stack.Pop();  // LIFO
        Process(node);
        foreach (var child in node.Children) {
            stack.Push(child);
        }
    }
}
```

---

## 🎯 When to Use Tree-of-Thoughts

### ✅ Use ToT For:

**Exploration problems:**
- Game playing (chess, Go, puzzles)
- Planning (trip planning, project planning)
- Optimization (find best solution among many)
- Constraint satisfaction (Sudoku, N-Queens)

**Backtracking needed:**
- When initial approach might fail
- Need to try multiple strategies
- Dead ends are common

**Complex reasoning:**
- Multi-step problems with many possible approaches
- When order of steps matters
- When need to verify each step

### ❌ Don't Use ToT For:

**Simple linear problems:**
- "What is 2 + 2?" (overkill!)
- "Translate this text" (no branching needed)
- Straightforward calculations

**When speed matters:**
- Real-time applications
- Simple question answering
- When first answer is usually correct

**Resource-constrained:**
- Limited compute budget
- Many API calls expensive
- Memory limitations

---

## 🔬 Advanced: Best-First Search

**Combine BFS with evaluation scores:**

```python
import heapq

class BestFirstToT(TreeOfThoughts):
    """
    Best-First Search: Always explore most promising node first.

    DEFINITION - Best-First Search:
    Like BFS, but instead of exploring in order,
    always explore the node with the highest score.

    Uses a priority queue (heap) instead of regular queue.

    C# equivalent:
    var priorityQueue = new PriorityQueue<ThoughtNode, double>();
    priorityQueue.Enqueue(node, -node.Score);  // Negative for max-heap
    var best = priorityQueue.Dequeue();  // Always gets highest-scored node
    """

    def best_first_search(self, problem: str, initial_state: str = "") -> Optional[ThoughtNode]:
        """
        Explore most promising nodes first.

        Always picks the node with highest evaluation score.
        Like A* search in pathfinding!
        """
        root = ThoughtNode(state=initial_state if initial_state else "Starting")

        # Priority queue: (priority, node)
        # Python's heapq is a min-heap, so use negative score for max-heap
        heap = [(-root.score, id(root), root)]  # (negative score, unique id, node)
        visited = set()

        print(f"Starting Best-First Search for: {problem}\n")

        while heap:
            # Get node with highest score (most promising)
            neg_score, _, current = heapq.heappop(heap)
            score = -neg_score

            print(f"Exploring (score={score:.2f}, depth={current.depth}): {current.state[:50]}...")

            # Check goal
            if self.is_goal_state(current.state, problem):
                print(f"\n✓ Solution found!")
                return current

            # Depth/visited checks
            if current.depth >= self.max_depth or current.state in visited:
                continue
            visited.add(current.state)

            # Generate and evaluate children
            thoughts = self.generate_thoughts(current.state, problem, self.branching_factor)

            for thought in thoughts:
                new_state = f"{current.state}\n{thought}"
                child_score = self.evaluate_state(new_state, problem)

                if child_score >= 0.3:  # Pruning threshold
                    child = ThoughtNode(new_state, thought, current, child_score)
                    current.add_child(child)

                    # Add to priority queue (negative for max-heap)
                    heapq.heappush(heap, (-child_score, id(child), child))

        print("\n✗ No solution found")
        return None

# Usage
tot = BestFirstToT(gpt, max_depth=5, branching_factor=3)
result = tot.best_first_search("Solve 8-puzzle", "Initial state: [[1,2,3],[4,5,6],[7,8,0]]")
```

---

## ✅ Quiz Questions

Test your understanding:

1. **What is the main advantage of Tree-of-Thoughts over Chain-of-Thought?**
   - A) It's faster
   - B) It can explore multiple solution paths and backtrack
   - C) It uses less memory
   - D) It doesn't need a language model

2. **What does "branching factor" mean?**
   - A) How many trees to build
   - B) How many possible next steps to explore at each node
   - C) How deep the tree can go
   - D) How fast the algorithm runs

3. **What's the difference between BFS and DFS?**
   - A) BFS is always faster
   - B) BFS explores level-by-level, DFS goes deep first
   - C) DFS always finds the best solution
   - D) They're the same algorithm

4. **What does "pruning" mean in ToT?**
   - A) Making the tree shorter
   - B) Stopping exploration of unpromising branches
   - C) Deleting old nodes
   - D) Trimming the input

5. **When should you use ToT?**
   - A) For all problems always
   - B) Only for math
   - C) For complex problems needing exploration and backtracking
   - D) Never, CoT is better

**Answers:** 1-B, 2-B, 3-B, 4-B, 5-C

---

## 🛠️ Hands-On Exercise

**Build a Sudoku solver using Tree-of-Thoughts:**

```python
# Exercise: Implement Sudoku Solver with ToT
class SudokuToTSolver:
    """
    Solve Sudoku puzzles using Tree-of-Thoughts.

    Sudoku is perfect for ToT because:
    - Multiple possible numbers to try in each cell
    - Need to backtrack when constraints violated
    - Some choices lead to dead ends
    - Need systematic exploration

    Your task: Complete this implementation!
    """

    def __init__(self, base_model):
        # TODO: Initialize ToT system
        # Hint: Use DFS (goes deep into one solution path)
        pass

    def solve(self, puzzle: List[List[int]]) -> Optional[List[List[int]]]:
        """
        Solve a Sudoku puzzle.

        Args:
            puzzle: 9x9 grid with 0 for empty cells

        Returns:
            Solved puzzle or None

        Example:
        puzzle = [
            [5,3,0,0,7,0,0,0,0],
            [6,0,0,1,9,5,0,0,0],
            ...
        ]
        """
        # TODO: Implement using ToT
        # Hints:
        # 1. State = current board configuration
        # 2. Thought = "Try number N in cell (row, col)"
        # 3. Evaluate = Check if constraints satisfied
        # 4. Goal = All cells filled, all constraints met
        pass

# Test
# puzzle = [[5,3,0,...], [6,0,0,...], ...]
# solver = SudokuToTSolver(gpt)
# solution = solver.solve(puzzle)
```

---

## 📝 Summary

**What you learned:**

1. **Tree-of-Thoughts = Exploring multiple solution paths**
   - Build a tree of possible reasoning steps
   - Evaluate each branch
   - Prune bad branches
   - Find best solution through search

2. **Search algorithms:**
   - **BFS:** Level-by-level, finds shortest solution
   - **DFS:** Depth-first, finds any solution quickly
   - **Best-First:** Always explore most promising node

3. **Key concepts:**
   - **Thought:** One possible reasoning step
   - **State:** Current situation after applying thoughts
   - **Branch:** Sequence of thoughts forming a path
   - **Pruning:** Stopping exploration of bad branches
   - **Evaluation:** Scoring how promising a state is

4. **When to use:**
   - Complex problems with multiple approaches
   - Need to explore and backtrack
   - Optimization and constraint satisfaction
   - Game playing and planning

**C#/.NET connections:**
- Tree search = Graph traversal algorithms
- BFS = Queue-based traversal (FIFO)
- DFS = Stack-based traversal (LIFO)
- Best-First = Priority Queue / Heap
- Pruning = Early return / continue
- State = Object with properties

**Complexity:**
- More sophisticated than Chain-of-Thought
- More computational cost (explores multiple paths)
- Better for problems that need it

---

## 🚀 Next Steps

**You've mastered Tree-of-Thoughts!**

You now understand:
- How to explore solution spaces systematically
- BFS vs DFS search strategies
- Evaluating and pruning branches
- Solving complex problems that need exploration

**Next lesson:** **Process Supervision** - How to train models to generate better reasoning by rewarding correct steps, not just correct answers!

**Continue to:** `04_process_supervision.md`

---

**You're learning the same techniques used in advanced AI systems and game-playing engines!** 🎉

**Tree-of-Thoughts is how AI solves complex problems that need strategic thinking!** 💪

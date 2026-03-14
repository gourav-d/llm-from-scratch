"""
Example 3: Tree-of-Thoughts Reasoning

This example demonstrates Tree-of-Thoughts (ToT) reasoning where the AI
explores multiple solution paths using search algorithms (BFS/DFS).

WHAT THIS SHOWS:
- Building a tree of reasoning steps
- Breadth-First Search (BFS) for shortest solution
- Depth-First Search (DFS) for any solution
- Evaluating and pruning bad branches
- Backtracking when stuck

COMPARISON TO C#:
Like graph/tree search algorithms:
public Solution BFS(Problem problem) {
    var queue = new Queue<State>();
    while (queue.Count > 0) {
        var state = queue.Dequeue();
        if (IsGoal(state)) return state.Solution;
        foreach (var child in Expand(state)) queue.Enqueue(child);
    }
}

Author: LLM Learning Module 7
"""

from typing import List, Dict, Optional, Set
from collections import deque
import heapq


class ThoughtNode:
    """
    Represents one node (state) in the reasoning tree.

    DEFINITION: Node
    A point in the tree containing:
    - Current state (where we are)
    - Thought that led here (how we got here)
    - Parent node (where we came from)
    - Children nodes (where we can go)
    - Score (how promising this is)

    C# equivalent:
    public class ThoughtNode {
        public string State { get; set; }
        public string Thought { get; set; }
        public ThoughtNode Parent { get; set; }
        public List<ThoughtNode> Children { get; set; }
        public double Score { get; set; }
        public int Depth { get; set; }
    }
    """

    def __init__(self, state: str, thought: str = "",
                 parent: Optional['ThoughtNode'] = None, score: float = 0.5):
        self.state = state
        self.thought = thought
        self.parent = parent
        self.score = score
        self.children = []
        self.depth = 0 if parent is None else parent.depth + 1

    def add_child(self, child: 'ThoughtNode'):
        """Add a child node to this node."""
        self.children.append(child)

    def get_path(self) -> List[str]:
        """
        Get the full reasoning path from root to this node.

        Returns list of thoughts that led here.

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
            if current.thought:
                path.insert(0, current.thought)
            current = current.parent
        return path

    def __repr__(self):
        return f"Node(depth={self.depth}, score={self.score:.2f}, state='{self.state[:30]}...')"

    def __lt__(self, other):
        """For heap comparison (used in Best-First Search)."""
        return self.score > other.score  # Higher score = higher priority


class MockGPT:
    """
    Mock GPT for demonstration.
    Replace with your actual GPT from Module 6.
    """

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str, max_length: int = 100,
                 temperature: float = 0.7) -> str:
        self.call_count += 1

        # Simulate thought generation
        if "next steps" in prompt.lower() or "next reasoning" in prompt.lower():
            return """
            1. Try approach A
            2. Try approach B
            3. Try approach C
            """

        # Simulate evaluation
        if "how promising" in prompt.lower() or "rate" in prompt.lower():
            # Simulate varied scores
            import random
            return str(random.uniform(0.3, 0.9))

        # Simulate goal check
        if "complete solution" in prompt.lower():
            if "final" in prompt or "answer" in prompt:
                return "yes"
            return "no"

        return "Response"


class TreeOfThoughts:
    """
    Tree-of-Thoughts reasoning system.

    Explores multiple reasoning paths using search algorithms.

    ANALOGY: Like a chess player thinking several moves ahead,
    exploring different strategies before choosing the best.

    C# equivalent:
    public class TreeOfThoughts {
        private GPTModel model;
        private int maxDepth;
        private int branchingFactor;

        public Solution Search(Problem problem) {
            return BreadthFirstSearch(problem);  // or DepthFirstSearch
        }
    }
    """

    def __init__(self, base_model, max_depth: int = 4,
                 branching_factor: int = 3, search_method: str = 'bfs'):
        """
        Initialize Tree-of-Thoughts system.

        Args:
            base_model: Your GPT model
            max_depth: Maximum reasoning depth (how many steps)
            branching_factor: How many options to try at each step
            search_method: 'bfs' (breadth-first) or 'dfs' (depth-first)

        DEFINITION - Max depth:
        Maximum number of reasoning steps allowed.
        Example: max_depth=4 means up to 4 steps in reasoning chain.

        DEFINITION - Branching factor:
        How many different ideas to explore at each node.
        Example: branching_factor=3 means try 3 options at each step.
        """
        self.base_model = base_model
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.search_method = search_method.lower()
        self.nodes_explored = 0
        self.nodes_pruned = 0

    def generate_thoughts(self, state: str, problem: str, num_thoughts: int) -> List[str]:
        """
        Generate possible next thoughts from current state.

        This asks the model: "What are the possible next steps?"

        Example:
        State: "x = 5"
        Problem: "Find x + y = 10"
        Generated thoughts:
        1. "Substitute x = 5 into equation"
        2. "Solve for y: y = 10 - x"
        3. "Calculate y = 10 - 5 = 5"

        Returns:
            List of possible next thoughts
        """
        prompt = f"""
Problem: {problem}
Current state: {state}

Generate {num_thoughts} possible next reasoning steps.
List them numbered 1, 2, 3, etc.

Steps:"""

        response = self.base_model.generate(prompt, max_length=150, temperature=0.8)

        # Parse thoughts from response
        thoughts = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                thought = line.lstrip('0123456789.-) ').strip()
                if thought:
                    thoughts.append(thought)

        return thoughts[:num_thoughts]

    def evaluate_state(self, state: str, problem: str) -> float:
        """
        Evaluate how promising a state is.

        Returns score from 0.0 (dead end) to 1.0 (very promising).

        This is like a heuristic function in A* search!

        C# equivalent:
        private double EvaluateState(string state, string problem) {
            var prompt = $"How promising is this state? (0-1): {state}";
            var response = model.Generate(prompt);
            return double.Parse(response);
        }
        """
        prompt = f"""
Problem: {problem}
State: {state}

How promising is this state for solving the problem?
Rate 0.0 (dead end) to 1.0 (very promising).
Just give a number.

Score:"""

        response = self.base_model.generate(prompt, max_length=10, temperature=0.3)

        try:
            score = float(response.strip().split()[0])
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default if parsing fails

    def is_goal_state(self, state: str, problem: str) -> bool:
        """
        Check if this state solves the problem.

        Returns True if this is a complete solution.

        C# equivalent:
        private bool IsGoalState(string state, string problem) {
            var prompt = $"Is this a complete solution? {state}";
            return model.Generate(prompt).Contains("yes");
        }
        """
        prompt = f"""
Problem: {problem}
State: {state}

Is this a complete solution? Answer 'yes' or 'no'.

Answer:"""

        response = self.base_model.generate(prompt, max_length=5, temperature=0.1)
        return 'yes' in response.lower()

    def breadth_first_search(self, problem: str, initial_state: str = "") -> Optional[ThoughtNode]:
        """
        Breadth-First Search: Explore level by level.

        DEFINITION: BFS
        Explores all nodes at depth 1, then all at depth 2, etc.
        Guarantees shortest solution.

        Visualization:
               Root
              / | \
            A   B  C     ← Explore all of level 1
           /|   |   |\
          D E   F   G H  ← Then all of level 2

        Uses Queue (FIFO - First In First Out)

        C# equivalent:
        public ThoughtNode BFS(string problem) {
            var queue = new Queue<ThoughtNode>();
            queue.Enqueue(root);

            while (queue.Count > 0) {
                var node = queue.Dequeue();
                if (IsGoal(node)) return node;
                foreach (var child in Expand(node))
                    queue.Enqueue(child);
            }
            return null;
        }
        """
        print(f"\n{'='*60}")
        print("BREADTH-FIRST SEARCH")
        print(f"{'='*60}\n")

        # Create root
        root = ThoughtNode(state=initial_state or "Starting")

        # BFS uses Queue (FIFO)
        queue = deque([root])
        visited = set()

        print(f"Problem: {problem}")
        print(f"Max depth: {self.max_depth}")
        print(f"Branching factor: {self.branching_factor}\n")

        while queue:
            # Dequeue (remove from front)
            current = queue.popleft()
            self.nodes_explored += 1

            print(f"[Depth {current.depth}] Exploring: {current.state[:40]}...")

            # Check if goal
            if self.is_goal_state(current.state, problem):
                print(f"\n✓ SOLUTION FOUND at depth {current.depth}!")
                print(f"Nodes explored: {self.nodes_explored}")
                print(f"Nodes pruned: {self.nodes_pruned}")
                return current

            # Don't exceed max depth
            if current.depth >= self.max_depth:
                continue

            # Skip visited states
            if current.state in visited:
                continue
            visited.add(current.state)

            # Generate child thoughts
            thoughts = self.generate_thoughts(current.state, problem,
                                              self.branching_factor)

            # Expand children
            for thought in thoughts:
                new_state = f"{current.state}\n→ {thought}"
                score = self.evaluate_state(new_state, problem)

                child = ThoughtNode(new_state, thought, current, score)
                current.add_child(child)

                # Pruning: Skip unpromising branches
                if score >= 0.3:  # Pruning threshold
                    queue.append(child)  # Add to back (FIFO)
                    print(f"  [Score {score:.2f}] Added: {thought[:35]}...")
                else:
                    self.nodes_pruned += 1
                    print(f"  [Score {score:.2f}] PRUNED: {thought[:35]}...")

        print(f"\n✗ No solution found within depth {self.max_depth}")
        print(f"Nodes explored: {self.nodes_explored}")
        print(f"Nodes pruned: {self.nodes_pruned}")
        return None

    def depth_first_search(self, problem: str, initial_state: str = "") -> Optional[ThoughtNode]:
        """
        Depth-First Search: Go deep into one path first.

        DEFINITION: DFS
        Explores one path completely before trying others.
        Faster to find any solution (but might not be shortest).

        Visualization:
               Root
                |
                A      ← Go deep
                |
                D      ← Keep going
                |
                I      ← Until end, then backtrack

        Uses Stack (LIFO - Last In First Out)

        C# equivalent:
        public ThoughtNode DFS(string problem) {
            var stack = new Stack<ThoughtNode>();
            stack.Push(root);

            while (stack.Count > 0) {
                var node = stack.Pop();
                if (IsGoal(node)) return node;
                foreach (var child in Expand(node))
                    stack.Push(child);
            }
            return null;
        }
        """
        print(f"\n{'='*60}")
        print("DEPTH-FIRST SEARCH")
        print(f"{'='*60}\n")

        root = ThoughtNode(state=initial_state or "Starting")

        # DFS uses Stack (LIFO)
        stack = [root]
        visited = set()

        print(f"Problem: {problem}")
        print(f"Max depth: {self.max_depth}")
        print(f"Branching factor: {self.branching_factor}\n")

        while stack:
            # Pop (remove from end)
            current = stack.pop()
            self.nodes_explored += 1

            print(f"[Depth {current.depth}] Exploring: {current.state[:40]}...")

            # Check if goal
            if self.is_goal_state(current.state, problem):
                print(f"\n✓ SOLUTION FOUND at depth {current.depth}!")
                print(f"Nodes explored: {self.nodes_explored}")
                print(f"Nodes pruned: {self.nodes_pruned}")
                return current

            # Depth/visited checks
            if current.depth >= self.max_depth or current.state in visited:
                continue
            visited.add(current.state)

            # Generate thoughts
            thoughts = self.generate_thoughts(current.state, problem,
                                              self.branching_factor)

            # Expand (reverse order for consistent behavior)
            for thought in reversed(thoughts):
                new_state = f"{current.state}\n→ {thought}"
                score = self.evaluate_state(new_state, problem)

                child = ThoughtNode(new_state, thought, current, score)
                current.add_child(child)

                # Pruning
                if score >= 0.3:
                    stack.append(child)  # Add to end (LIFO)
                    print(f"  [Score {score:.2f}] Added: {thought[:35]}...")
                else:
                    self.nodes_pruned += 1
                    print(f"  [Score {score:.2f}] PRUNED: {thought[:35]}...")

        print(f"\n✗ No solution found")
        print(f"Nodes explored: {self.nodes_explored}")
        print(f"Nodes pruned: {self.nodes_pruned}")
        return None

    def best_first_search(self, problem: str, initial_state: str = "") -> Optional[ThoughtNode]:
        """
        Best-First Search: Always explore most promising node first.

        DEFINITION: Best-First Search
        Like BFS but uses priority queue.
        Always picks the highest-scored node to explore next.

        Uses Heap (priority queue)

        C# equivalent:
        public ThoughtNode BestFirstSearch(string problem) {
            var pq = new PriorityQueue<ThoughtNode, double>();
            pq.Enqueue(root, -root.Score);  // Negative for max-heap

            while (pq.Count > 0) {
                var node = pq.Dequeue();
                if (IsGoal(node)) return node;
                foreach (var child in Expand(node))
                    pq.Enqueue(child, -child.Score);
            }
            return null;
        }
        """
        print(f"\n{'='*60}")
        print("BEST-FIRST SEARCH")
        print(f"{'='*60}\n")

        root = ThoughtNode(state=initial_state or "Starting", score=0.5)

        # Priority queue (heap) - Python's heapq is min-heap
        # We want max-heap (highest score first), so use negative scores
        heap = [(-root.score, id(root), root)]
        visited = set()

        print(f"Problem: {problem}")
        print(f"Max depth: {self.max_depth}\n")

        while heap:
            # Pop highest-scored node
            neg_score, _, current = heapq.heappop(heap)
            score = -neg_score
            self.nodes_explored += 1

            print(f"[Score {score:.2f}, Depth {current.depth}] {current.state[:40]}...")

            # Check if goal
            if self.is_goal_state(current.state, problem):
                print(f"\n✓ SOLUTION FOUND!")
                print(f"Nodes explored: {self.nodes_explored}")
                return current

            # Checks
            if current.depth >= self.max_depth or current.state in visited:
                continue
            visited.add(current.state)

            # Generate thoughts
            thoughts = self.generate_thoughts(current.state, problem,
                                              self.branching_factor)

            for thought in thoughts:
                new_state = f"{current.state}\n→ {thought}"
                child_score = self.evaluate_state(new_state, problem)

                if child_score >= 0.3:
                    child = ThoughtNode(new_state, thought, current, child_score)
                    current.add_child(child)

                    # Add to heap with negative score (for max-heap behavior)
                    heapq.heappush(heap, (-child_score, id(child), child))
                    print(f"  [Score {child_score:.2f}] Added: {thought[:35]}...")

        print(f"\n✗ No solution found")
        return None

    def solve(self, problem: str, initial_state: str = "") -> Optional[Dict]:
        """
        Solve problem using configured search method.

        Returns:
            Dictionary with solution and reasoning path, or None
        """
        # Reset counters
        self.nodes_explored = 0
        self.nodes_pruned = 0

        # Choose search method
        if self.search_method == 'bfs':
            goal = self.breadth_first_search(problem, initial_state)
        elif self.search_method == 'dfs':
            goal = self.depth_first_search(problem, initial_state)
        else:  # best-first
            goal = self.best_first_search(problem, initial_state)

        if goal is None:
            return None

        # Extract solution
        path = goal.get_path()

        return {
            'problem': problem,
            'solution': goal.state,
            'reasoning_path': path,
            'depth': goal.depth,
            'score': goal.score,
            'nodes_explored': self.nodes_explored,
            'nodes_pruned': self.nodes_pruned,
            'search_method': self.search_method
        }


def display_solution(result: Optional[Dict]):
    """Pretty-print the solution."""
    if result is None:
        print("\n" + "="*60)
        print("NO SOLUTION FOUND")
        print("="*60)
        return

    print("\n" + "="*60)
    print("SOLUTION")
    print("="*60)

    print(f"\nProblem: {result['problem']}")
    print(f"\nSearch method: {result['search_method'].upper()}")
    print(f"Solution depth: {result['depth']}")
    print(f"Nodes explored: {result['nodes_explored']}")
    print(f"Nodes pruned: {result['nodes_pruned']}")

    print(f"\nReasoning Path ({len(result['reasoning_path'])} steps):")
    print("-"*60)
    for i, step in enumerate(result['reasoning_path'], 1):
        print(f"{i}. {step}")

    print("\n" + "="*60)


def demo_bfs():
    """Demonstrate Breadth-First Search."""
    print("\n" + "🌳"*30)
    print("DEMO 1: Breadth-First Search (BFS)")
    print("🌳"*30)

    gpt = MockGPT()
    tot = TreeOfThoughts(gpt, max_depth=3, branching_factor=3, search_method='bfs')

    problem = "Find the value of x where x² - 5x + 6 = 0"
    result = tot.solve(problem, "Need to solve quadratic equation")

    display_solution(result)


def demo_dfs():
    """Demonstrate Depth-First Search."""
    print("\n" + "🌳"*30)
    print("DEMO 2: Depth-First Search (DFS)")
    print("🌳"*30)

    gpt = MockGPT()
    tot = TreeOfThoughts(gpt, max_depth=3, branching_factor=3, search_method='dfs')

    problem = "Find the value of x where x² - 5x + 6 = 0"
    result = tot.solve(problem, "Need to solve quadratic equation")

    display_solution(result)


def demo_best_first():
    """Demonstrate Best-First Search."""
    print("\n" + "🌳"*30)
    print("DEMO 3: Best-First Search (Most Promising First)")
    print("🌳"*30)

    gpt = MockGPT()
    tot = TreeOfThoughts(gpt, max_depth=3, branching_factor=3, search_method='best')

    problem = "Find the value of x where x² - 5x + 6 = 0"
    result = tot.solve(problem, "Need to solve quadratic equation")

    display_solution(result)


def compare_search_methods():
    """Compare all three search methods."""
    print("\n" + "📊"*30)
    print("COMPARISON: BFS vs DFS vs Best-First")
    print("📊"*30)

    problem = "Find optimal solution to puzzle"

    results = {}
    for method in ['bfs', 'dfs', 'best']:
        print(f"\n{'─'*60}")
        print(f"Testing {method.upper()}...")
        print(f"{'─'*60}")

        gpt = MockGPT()
        tot = TreeOfThoughts(gpt, max_depth=3, branching_factor=3,
                            search_method=method)

        result = tot.solve(problem, "Initial puzzle state")
        results[method] = result

    # Compare
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print(f"\n{'Method':<15} {'Found':<10} {'Depth':<10} {'Explored':<15} {'Pruned':<10}")
    print("-"*60)

    for method, result in results.items():
        if result:
            print(f"{method.upper():<15} {'Yes':<10} {result['depth']:<10} "
                  f"{result['nodes_explored']:<15} {result['nodes_pruned']:<10}")
        else:
            print(f"{method.upper():<15} {'No':<10} {'-':<10} {'-':<15} {'-':<10}")

    print("\nKey Differences:")
    print("  BFS: Explores level-by-level (shortest path)")
    print("  DFS: Goes deep first (any path)")
    print("  Best-First: Follows most promising nodes (smart path)")
    print("="*60)


if __name__ == "__main__":
    """
    Run Tree-of-Thoughts demonstrations.

    To use with your actual GPT:

    from modules.module_06.example_01_complete_gpt import GPT, GPTConfig

    config = GPTConfig(vocab_size=50257, max_seq_len=256,
                      embed_dim=512, n_layers=6, n_heads=8)
    gpt = GPT(config)

    tot = TreeOfThoughts(gpt, max_depth=5, branching_factor=4,
                        search_method='bfs')
    result = tot.solve("Your problem here")
    display_solution(result)
    """

    print("\n" + "🎓"*35)
    print("TREE-OF-THOUGHTS REASONING EXAMPLES")
    print("🎓"*35)

    # Run demonstrations
    demo_bfs()
    demo_dfs()
    demo_best_first()
    compare_search_methods()

    print("\n" + "✓"*35)
    print("All demonstrations complete!")
    print("✓"*35 + "\n")

    print("KEY TAKEAWAYS:")
    print("1. ToT explores multiple reasoning paths like a tree")
    print("2. BFS guarantees shortest solution (level-by-level)")
    print("3. DFS finds any solution quickly (depth-first)")
    print("4. Best-First explores most promising nodes first")
    print("5. Pruning removes unpromising branches")
    print("6. Backtracking allows trying different approaches")
    print("\nNext: Apply ToT to complex problems like puzzles!")

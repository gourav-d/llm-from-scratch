"""
Module 07 - Reasoning & Coding Models
Exercise 03: Tree-of-Thoughts Reasoning

GLOSSARY
--------
Tree-of-Thoughts  : Reasoning technique where the model explores MULTIPLE reasoning
(ToT)               paths in a tree structure — like a chess engine exploring moves.
                    Unlike CoT (one path), ToT branches and backtracks.
Node              : One step/state in the reasoning tree.
                    Has: state (current position), score (how promising), depth.
Root Node         : The starting problem. Depth = 0.
Leaf Node         : A node with no children. Either a solution or a dead end.
Depth             : How many reasoning steps from root to this node.
                    Root = depth 0. Its children = depth 1. And so on.
Score             : 0-1 float rating how promising a node is.
                    1.0 = looks like a solution path.
                    0.0 = dead end.
BFS               : Breadth-First Search. Explore all nodes at depth 1 first,
(Breadth-First)     then all at depth 2, etc. Finds shortest path.
                    Like FIFO queue (Queue<T> in C#).
DFS               : Depth-First Search. Follow one path as deep as possible,
(Depth-First)       then backtrack. Can find solutions quickly.
                    Like recursive stack (Stack<T> in C#).
Pruning           : Cutting off branches with low score (< threshold).
                    Don't explore dead ends. Saves compute.
"""

print("=" * 60)
print("Exercise 03: Tree-of-Thoughts Reasoning")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Build a Thought Node
#
#  Background:
#    A ThoughtNode represents one step in the reasoning tree.
#    Key properties:
#      - state: current reasoning state (string)
#      - depth: how deep in the tree (root = 0)
#      - score: how promising (0.0 to 1.0)
#      - children: list of child nodes (empty at creation)
#
#    Computing depth:
#      If parent is None -> depth = 0  (root node)
#      Else              -> depth = parent.depth + 1
#
#  Your Task:
#    Complete the ThoughtNode class:
#    - __init__: set state, score, depth (from parent), children=[]
#    - add_child: append child to self.children, return child
#    - is_leaf: return True if no children
#
#  C# Analogy:
#    class ThoughtNode {
#        public string State;
#        public double Score;
#        public int Depth;
#        public List<ThoughtNode> Children = new();
#        public bool IsLeaf => Children.Count == 0;
#    }
# ============================================================

print("-" * 50)
print("EXERCISE 1: ThoughtNode Class")
print("-" * 50)
print()


class ThoughtNode:
    """
    One node (step) in the Tree-of-Thoughts reasoning tree.

    Attributes:
        state    (str)  : Description of the current reasoning state.
        score    (float): How promising this path looks (0.0 to 1.0).
        depth    (int)  : Distance from root node (root = 0).
        children (list) : Child ThoughtNode objects branching from here.
    """

    def __init__(self, state, score=0.5, parent=None):
        """
        Initialize a thought node.

        Parameters:
            state  (str)       : Current reasoning state.
            score  (float)     : Promising-ness score (default 0.5).
            parent (ThoughtNode): Parent node, or None if root.
        """
        # TODO:
        # self.state    = state
        # self.score    = score
        # self.children = []
        # self.depth    = 0 if parent is None else parent.depth + 1
        pass  # Replace with your implementation

    def add_child(self, child):
        """
        Add a child node and return it.

        Parameters:
            child (ThoughtNode): Child to add.

        Returns:
            ThoughtNode: The child (for chaining).
        """
        # TODO: self.children.append(child); return child
        pass  # Replace with your implementation

    def is_leaf(self):
        """
        Return True if this node has no children.

        Returns:
            bool: True = leaf node (no children).
        """
        # TODO: return len(self.children) == 0
        pass  # Replace with your implementation


# Build a small tree and verify
root = ThoughtNode("Start: solve 2 + 3 × 4", score=0.5)

if hasattr(root, 'depth') and root.depth is not None:
    child1 = root.add_child(ThoughtNode("Try: compute 2+3 first -> 5×4=20 (wrong order)", score=0.2, parent=root))
    child2 = root.add_child(ThoughtNode("Try: compute 3×4 first -> 12, then 2+12=14 (correct)", score=0.9, parent=root))

    if child1 is not None:
        grandchild = child1.add_child(ThoughtNode("Dead end: order of operations violated", score=0.1, parent=child1))

        print(f"  Root depth  : {root.depth}    (expected: 0)")
        print(f"  Child depth : {child1.depth}   (expected: 1)")
        print(f"  Grandchild  : {grandchild.depth}   (expected: 2)")
        print(f"  Root is leaf? {root.is_leaf()}  (expected: False)")
        print(f"  Child2 leaf?  {child2.is_leaf()}  (expected: True — no children added)")
        print(f"  Root children: {len(root.children)}  (expected: 2)")
print()


# ============================================================
#  EXERCISE 2
#  Topic: Score Nodes and Prune Low-Scoring Branches
#
#  Background:
#    After generating candidate thoughts, we score each one (0-1).
#    Low-scoring nodes are pruned (removed) to save compute.
#    Only promising branches are expanded further.
#
#    Pruning rule: keep nodes where score >= threshold.
#    Typical threshold: 0.5 (discard bottom half).
#
#  Your Task:
#    Write: prune_nodes(nodes, threshold) -> list
#    nodes: list of ThoughtNode objects
#    Returns only nodes with score >= threshold.
#    Sort surviving nodes by score descending (best first).
#
#  C# Analogy:
#    nodes.Where(n => n.Score >= threshold)
#         .OrderByDescending(n => n.Score)
#         .ToList()
# ============================================================

print("-" * 50)
print("EXERCISE 2: Prune Low-Scoring Nodes")
print("-" * 50)
print()


def prune_nodes(nodes, threshold=0.5):
    """
    Keep only nodes scoring at or above the threshold.

    Parameters:
        nodes     (list of ThoughtNode): Candidate nodes.
        threshold (float)              : Minimum score to keep (default 0.5).

    Returns:
        list of ThoughtNode: Surviving nodes sorted by score descending.
    """
    # TODO:
    # 1. Filter: kept = [n for n in nodes if n.score >= threshold]
    # 2. Sort:   kept.sort(key=lambda n: n.score, reverse=True)
    # 3. Return: kept
    pass  # Replace with your implementation


candidates = [
    ThoughtNode("Path A: direct calculation",      score=0.9),
    ThoughtNode("Path B: estimate then refine",    score=0.6),
    ThoughtNode("Path C: wrong formula",           score=0.2),
    ThoughtNode("Path D: similar to A, valid",     score=0.75),
    ThoughtNode("Path E: very uncertain",          score=0.1),
    ThoughtNode("Path F: exactly at threshold",    score=0.5),
]

kept = prune_nodes(candidates, threshold=0.5)

if kept is not None:
    print(f"  Input: {len(candidates)} nodes")
    print(f"  Kept (score >= 0.5): {len(kept)} nodes  (expected: 4)")
    print()
    print(f"  {'State':<40} {'Score':>7}")
    print("  " + "-" * 50)
    for n in kept:
        print(f"  {n.state:<40} {n.score:>7.2f}")
print()


# ============================================================
#  EXERCISE 3
#  Topic: BFS — Breadth-First Search Through Thought Tree
#
#  Background:
#    BFS explores all nodes level-by-level.
#    Use a queue (FIFO): dequeue a node, enqueue its children.
#
#    Returns all nodes in BFS order (breadth-first).
#    This lets us find the shallowest (fastest) solution path.
#
#    BFS order for this tree:
#      root -> child1, child2 -> grandchildren of child1, grandchildren of child2
#
#  Your Task:
#    Write: bfs_traverse(root_node) -> list of ThoughtNode
#    Returns all nodes in breadth-first order.
#    Use collections.deque as the queue.
#
#  C# Analogy:
#    Queue<ThoughtNode> q = new(); q.Enqueue(root);
#    while (q.Count > 0) { var n = q.Dequeue(); yield return n;
#                           n.Children.ForEach(c => q.Enqueue(c)); }
# ============================================================

print("-" * 50)
print("EXERCISE 3: BFS Traversal of Thought Tree")
print("-" * 50)
print()

from collections import deque   # deque: efficient queue (FIFO) for BFS


def bfs_traverse(root_node):
    """
    Return all nodes in breadth-first order.

    Parameters:
        root_node (ThoughtNode): The root of the reasoning tree.

    Returns:
        list of ThoughtNode: All nodes, level by level.
    """
    # TODO:
    # visited = []
    # queue   = deque([root_node])
    # while queue:
    #     node = queue.popleft()          # dequeue from front
    #     visited.append(node)
    #     for child in node.children:
    #         queue.append(child)         # enqueue children at back
    # return visited
    pass  # Replace with your implementation


# Build a small tree
r = ThoughtNode("Root",    score=0.5)
c1 = r.add_child(ThoughtNode("Child-1",  score=0.8, parent=r))
c2 = r.add_child(ThoughtNode("Child-2",  score=0.3, parent=r))
gc1 = c1.add_child(ThoughtNode("GC-1-1", score=0.9, parent=c1)) if c1 else None
gc2 = c1.add_child(ThoughtNode("GC-1-2", score=0.4, parent=c1)) if c1 else None

if gc1 is not None:
    traversal = bfs_traverse(r)
    if traversal is not None:
        print(f"  BFS order ({len(traversal)} nodes):")
        for i, node in enumerate(traversal):
            print(f"    [{i}] depth={node.depth}  state='{node.state}'  score={node.score}")
        print()
        print("  Expected order: Root, Child-1, Child-2, GC-1-1, GC-1-2")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Find Best Leaf Node
#
#  Background:
#    After building and exploring the tree, we want the BEST solution.
#    The best solution is the leaf node with the highest score.
#    (Leaf = no children = terminal state = a proposed answer)
#
#    Steps:
#      1. Traverse all nodes (BFS or DFS)
#      2. Filter to keep only leaf nodes
#      3. Return the leaf with the highest score
#
#  Your Task:
#    Write: best_leaf(root_node) -> ThoughtNode or None
#    Returns the leaf node with the highest score.
#    If no leaves exist (tree has only root), return root.
#
#  C# Analogy:
#    allNodes.Where(n => n.IsLeaf()).MaxBy(n => n.Score)
# ============================================================

print("-" * 50)
print("EXERCISE 4: Find Best Leaf Node")
print("-" * 50)
print()


def best_leaf(root_node):
    """
    Find the leaf node with the highest score in the tree.

    Parameters:
        root_node (ThoughtNode): Root of the reasoning tree.

    Returns:
        ThoughtNode: The leaf with the highest score.
    """
    # TODO:
    # 1. Collect all nodes via BFS (reuse bfs_traverse if it works,
    #    or inline the BFS loop here).
    # 2. Filter to leaves: [n for n in all_nodes if n.is_leaf()]
    # 3. If no leaves, return root_node.
    # 4. Return max(leaves, key=lambda n: n.score)
    pass  # Replace with your implementation


# Build tree with known scores
root2   = ThoughtNode("Problem: schedule 3 tasks", score=0.5)
path_a  = root2.add_child(ThoughtNode("Plan A: task 1 first", score=0.6, parent=root2))
path_b  = root2.add_child(ThoughtNode("Plan B: task 2 first", score=0.7, parent=root2))

if path_a and path_b:
    sol_a1 = path_a.add_child(ThoughtNode("A→ finish: total=10h", score=0.55, parent=path_a))
    sol_a2 = path_a.add_child(ThoughtNode("A→ finish: total=8h",  score=0.85, parent=path_a))
    sol_b1 = path_b.add_child(ThoughtNode("B→ finish: total=9h",  score=0.70, parent=path_b))

    winner = best_leaf(root2)
    if winner is not None:
        print(f"  Best leaf: '{winner.state}'")
        print(f"  Score    : {winner.score}")
        print(f"  Depth    : {winner.depth}")
        print()
        print("  Expected: 'A→ finish: total=8h'  (score=0.85, highest leaf)")
print()

print("=" * 60)
print("All exercises complete!")
print("=" * 60)

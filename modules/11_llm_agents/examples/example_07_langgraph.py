"""
Example 07: LangGraph -- State Machines for Agent Workflows
============================================================

GLOSSARY
--------
StateGraph:
  The core LangGraph class. You define nodes (steps) and edges (transitions).
  At runtime it executes like a state machine.
  C# analogy: like StateMachine<T> or Azure Durable Functions orchestration.

Node:
  A function in the graph: receives state dict, returns partial state update.
  LangGraph merges the returned dict into the existing state automatically.

Edge:
  A connection between nodes. Normal edges always go A -> B.
  Conditional edges call a function to decide the next node.

State:
  A shared dict that flows through all nodes, accumulating results.
  C# analogy: like IOrchestrationContext -- shared across all activities.

END:
  Special constant. Route to END to stop the graph.

Checkpointing:
  LangGraph saves state after every node. Enables pause/resume.
  C# analogy: like Durable Functions replay/event-sourcing.

WHAT THIS EXAMPLE SHOWS
------------------------
Part A: Build a minimal 2-node graph (research -> write)
Part B: Add a conditional edge (review node with retry loop)
Part C: Simulate checkpointing (pause + resume)
Part D: Show graph execution trace

LIBRARIES NEEDED
----------------
  None (pure Python stdlib -- we simulate LangGraph without the real library)
  Real project: pip install langgraph langchain-openai
"""

import json    # For pretty-printing state
import time    # For simulating delays

print("=" * 65)
print("EXAMPLE 07: LangGraph -- State Machines for Agent Workflows")
print("=" * 65)


# ==============================================================================
# LANGGRAPH SIMULATOR
# We simulate LangGraph's StateGraph without installing the real library.
# This lets you understand the concepts without extra dependencies.
# The API is intentionally similar so migration to real LangGraph is easy.
# ==============================================================================

END = "__END__"    # Sentinel constant: route to this to stop the graph


class SimulatedStateGraph:
    """
    Simulates LangGraph's StateGraph class.

    In real LangGraph:
      from langgraph.graph import StateGraph, END
      graph = StateGraph(AgentState)

    C# analogy: like a StateMachine<TState> where you define states and transitions.
    """

    def __init__(self, state_schema: dict):
        """
        state_schema: the initial state template (defines all keys)
        In real LangGraph this would be a TypedDict class.
        """
        self.initial_state     = state_schema.copy()  # Template for fresh state
        self.nodes             = {}                    # name -> function
        self.normal_edges      = {}                    # from_node -> to_node
        self.conditional_edges = {}                    # from_node -> (router_fn, mapping)
        self.entry_point       = None                  # Which node to start at
        self.checkpoints       = {}                    # thread_id -> state snapshots
        self.execution_log     = []                    # Trace of what ran

    def add_node(self, name: str, fn):
        """
        Register a node (step) in the graph.
        fn: function(state: dict) -> dict  (returns partial state update)
        """
        self.nodes[name] = fn

    def add_edge(self, from_node: str, to_node: str):
        """Add a normal (unconditional) edge from one node to another."""
        self.normal_edges[from_node] = to_node

    def add_conditional_edges(self, from_node: str, router_fn, mapping: dict):
        """
        Add a conditional edge.
        router_fn: (state) -> str  (returns a key from mapping)
        mapping:   dict mapping router output -> next node name (or END)
        """
        self.conditional_edges[from_node] = (router_fn, mapping)

    def set_entry_point(self, node_name: str):
        """Set which node to start execution from."""
        self.entry_point = node_name

    def compile(self, checkpointer=None):
        """
        Compile the graph into a runnable app.
        checkpointer: if provided, enables pause/resume.
        Returns a CompiledGraph that has an .invoke() method.
        """
        return CompiledGraph(self, checkpointer)


class InMemoryCheckpointer:
    """
    Simulates LangGraph's MemorySaver.
    Stores graph state snapshots in a dict, keyed by thread_id.

    Real LangGraph: from langgraph.checkpoint.memory import MemorySaver
    C# analogy: like IOrchestrationStore in Durable Functions.
    """

    def __init__(self):
        self._store = {}    # thread_id -> list of (node_name, state) tuples

    def save(self, thread_id: str, node_name: str, state: dict):
        """Save state after a node executes."""
        if thread_id not in self._store:
            self._store[thread_id] = []
        self._store[thread_id].append({
            "node":  node_name,
            "state": state.copy()    # Save a snapshot (not a reference)
        })

    def load_latest(self, thread_id: str):
        """Load the most recent checkpoint for a thread."""
        if thread_id not in self._store or not self._store[thread_id]:
            return None
        return self._store[thread_id][-1]   # Return the last checkpoint

    def get_history(self, thread_id: str):
        """Return all checkpoints for a thread (full history)."""
        return self._store.get(thread_id, [])


class CompiledGraph:
    """
    The runnable version of a StateGraph.
    Created by StateGraph.compile().
    Has .invoke() to run the graph.
    """

    def __init__(self, graph: SimulatedStateGraph, checkpointer):
        self.graph        = graph
        self.checkpointer = checkpointer

    def invoke(self, initial_state, config: dict = None, interrupt_before: list = None):
        """
        Run the graph.

        initial_state: starting state dict (or None to resume from checkpoint)
        config:        {"configurable": {"thread_id": "..."}} for checkpointing
        interrupt_before: list of node names to pause before executing

        Returns final state.
        """
        interrupt_before = interrupt_before or []
        thread_id = (config or {}).get("configurable", {}).get("thread_id")

        # Determine starting state and node
        if initial_state is None and thread_id and self.checkpointer:
            # Resume from checkpoint
            checkpoint = self.checkpointer.load_latest(thread_id)
            if checkpoint:
                state        = checkpoint["state"]
                current_node = checkpoint["node"]
                print(f"  [Graph] Resuming from checkpoint: node='{current_node}'")
                # Find the next node after the checkpoint
                current_node = self._get_next_node(current_node, state)
            else:
                print("  [Graph] No checkpoint found. Starting fresh.")
                state        = self.graph.initial_state.copy()
                current_node = self.graph.entry_point
        else:
            # Fresh start
            state        = {**self.graph.initial_state, **(initial_state or {})}
            current_node = self.graph.entry_point

        self.graph.execution_log = []    # Clear previous execution log

        # Execute nodes until we reach END
        max_steps = 20    # Safety limit to prevent infinite loops
        steps     = 0

        while current_node != END and current_node is not None and steps < max_steps:
            steps += 1

            # Check for interrupt BEFORE executing this node
            if current_node in interrupt_before:
                print(f"  [Graph] INTERRUPTED before node: '{current_node}'")
                if thread_id and self.checkpointer:
                    # Save the interrupt point so we can resume here
                    self.checkpointer.save(thread_id, f"__interrupt_before_{current_node}__", state)
                return state    # Return current state -- caller will resume later

            print(f"\n  [Graph] Executing node: '{current_node}'")
            self.graph.execution_log.append(f"NODE: {current_node}")

            # Execute the node function: fn(state) -> partial state update
            node_fn      = self.graph.nodes[current_node]
            state_update = node_fn(state)          # Run the node

            # Merge the partial update into the full state
            # (Only update the keys the node returned -- leave others unchanged)
            state.update(state_update)

            # Save checkpoint after this node (if checkpointing enabled)
            if thread_id and self.checkpointer:
                self.checkpointer.save(thread_id, current_node, state)

            # Determine the next node
            current_node = self._get_next_node(current_node, state)

        if steps >= max_steps:
            print(f"  [Graph] WARNING: Reached maximum steps ({max_steps}). Stopping.")

        print(f"\n  [Graph] Reached END. Total nodes executed: {steps}")
        return state    # Return the final state

    def _get_next_node(self, current_node: str, state: dict) -> str:
        """
        Determine the next node after current_node.
        Checks conditional edges first, then normal edges.
        """
        # Check conditional edges (dynamic routing based on state)
        if current_node in self.graph.conditional_edges:
            router_fn, mapping = self.graph.conditional_edges[current_node]
            decision     = router_fn(state)    # Call the router function
            next_node    = mapping.get(decision, END)    # Look up in mapping
            self.graph.execution_log.append(f"  ROUTE: {current_node} -> {next_node} (decision='{decision}')")
            print(f"  [Graph] Conditional route: '{current_node}' -> '{next_node}' (decision='{decision}')")
            return next_node

        # Check normal edges (always go to the same next node)
        if current_node in self.graph.normal_edges:
            next_node = self.graph.normal_edges[current_node]
            self.graph.execution_log.append(f"  EDGE: {current_node} -> {next_node}")
            return next_node

        # No edge defined -- stop
        return END


# ==============================================================================
# PART A: Minimal 2-Node Graph (Research -> Write)
# ==============================================================================

print("\n" + "=" * 65)
print("PART A: Minimal 2-Node Graph (Research -> Write)")
print("=" * 65)

print("""
Simplest possible LangGraph: two nodes, one normal edge.

  [research] -> [write] -> END

State flows through both nodes, accumulating results.
C# analogy: like two sequential Durable Functions Activities.
""")

# Define the state schema (initial values for all keys)
initial_state_ab = {
    "task":     "",          # The task to work on
    "research": "",          # Will be filled by research node
    "draft":    "",          # Will be filled by write node
    "steps":    []           # Log of what happened (grows as we add to it)
}

# ---- Node Functions ----

def research_node(state: dict) -> dict:
    """
    Look up information for the task.
    Receives full state, returns ONLY the keys it changed.
    LangGraph will merge this into the full state.
    """
    task = state["task"]
    print(f"    [research_node] Task: '{task}'")

    # Simulated research (in production: call a web search tool or vector DB)
    research_results = {
        "python frameworks": "Top Python frameworks: Django (full-featured), FastAPI (modern/fast), Flask (lightweight).",
        "machine learning":  "ML basics: supervised learning, unsupervised learning, neural networks, gradient descent.",
        "transformers":      "Transformers use self-attention mechanism. Encoder processes input, decoder generates output.",
    }

    # Find matching research
    result = "No specific research found."
    for key, value in research_results.items():
        if key in task.lower():
            result = value
            break

    print(f"    [research_node] Research complete: {result[:60]}...")
    # Return PARTIAL update -- only the keys this node changed
    return {
        "research": result,
        "steps":    state["steps"] + ["research_done"]   # Append to the list
    }


def write_node(state: dict) -> dict:
    """
    Write a draft based on the research in state.
    """
    task     = state["task"]
    research = state["research"]

    print(f"    [write_node] Writing draft based on research...")
    draft = f"# {task.title()}\n\n## Summary\n{research}\n\n## Conclusion\nThis covers the key points about {task}."
    print(f"    [write_node] Draft written ({len(draft)} chars).")

    return {
        "draft": draft,
        "steps": state["steps"] + ["write_done"]
    }


# Build and run the minimal graph
graph_ab = SimulatedStateGraph(initial_state_ab)
graph_ab.add_node("research", research_node)
graph_ab.add_node("write",    write_node)
graph_ab.add_edge("research", "write")     # Normal edge: always go research -> write
graph_ab.add_edge("write", END)            # Normal edge: always go write -> END
graph_ab.set_entry_point("research")

app_ab = graph_ab.compile()

print("Running graph with task: 'Python frameworks'")
final_state = app_ab.invoke({"task": "python frameworks"})

print("\nFINAL STATE:")
print(f"  task:     {final_state['task']}")
print(f"  research: {final_state['research'][:80]}...")
print(f"  draft:    {final_state['draft'][:100]}...")
print(f"  steps:    {final_state['steps']}")


# ==============================================================================
# PART B: Conditional Edge (Review + Retry Loop)
# ==============================================================================

print("\n" + "=" * 65)
print("PART B: Conditional Edge -- Review with Retry Loop")
print("=" * 65)

print("""
A more realistic graph: research -> write -> review.
If review passes: END.
If review fails: loop back to research (try again).
Maximum 3 attempts.

  [research] -> [write] -> [review] --pass--> END
                    ^                  |
                    |                  | fail (and attempts < 3)
                    +------------------+

C# analogy: like a Durable Functions orchestration with ContinueAsNew()
            for retry logic, counting attempts in the state.
""")

initial_state_c = {
    "task":     "",
    "research": "",
    "draft":    "",
    "score":    0,     # Quality score from review node (0-100)
    "attempts": 0,     # How many times we've tried
    "approved": False, # Whether the draft was approved
    "steps":    []
}

def research_node_c(state: dict) -> dict:
    """Research node -- simulates improving research on retry."""
    attempt  = state["attempts"] + 1    # This is attempt number N
    task     = state["task"]
    print(f"    [research_node] Attempt #{attempt} -- researching '{task}'")

    # Simulate: research gets better with each attempt
    quality_additions = ["", " More detail: uses async/await.", " Expert note: production-ready."]
    idx    = min(attempt - 1, len(quality_additions) - 1)
    result = f"Framework research for '{task}'.{quality_additions[idx]}"

    return {
        "research": result,
        "attempts": attempt,    # Track how many times we've tried
        "steps":    state["steps"] + [f"research_attempt_{attempt}"]
    }

def write_node_c(state: dict) -> dict:
    """Write node -- draft quality improves each attempt."""
    attempt  = state["attempts"]
    research = state["research"]
    print(f"    [write_node] Writing draft (attempt #{attempt})...")

    # Longer draft = higher score from reviewer
    base   = f"# Draft (Attempt {attempt})\n\n{research}"
    extras = "\n\n## Details\nMore comprehensive content." * (attempt - 1)   # More content each retry
    draft  = base + extras

    return {
        "draft": draft,
        "steps": state["steps"] + [f"write_attempt_{attempt}"]
    }

def review_node_c(state: dict) -> dict:
    """Review node -- score based on draft quality."""
    draft    = state["draft"]
    attempt  = state["attempts"]
    print(f"    [review_node] Reviewing draft (attempt #{attempt})...")

    # Score: longer + more structured = higher score
    score = min(100, len(draft.split()) * 3 + (20 * attempt))   # Improves with attempts
    approved = score >= 70

    print(f"    [review_node] Score: {score}/100. Approved: {approved}")
    return {
        "score":    score,
        "approved": approved,
        "steps":    state["steps"] + [f"review_attempt_{attempt}_score_{score}"]
    }

def route_after_review(state: dict) -> str:
    """
    Conditional router: decides what happens after review.
    Returns a string that maps to a node name (or END).
    """
    if state["approved"]:
        return "approved"      # Route to END via "approved" mapping key
    elif state["attempts"] >= 3:
        return "give_up"       # Too many attempts -> also END
    else:
        return "retry"         # Try again from research


# Build the graph with retry loop
graph_c = SimulatedStateGraph(initial_state_c)
graph_c.add_node("research", research_node_c)
graph_c.add_node("write",    write_node_c)
graph_c.add_node("review",   review_node_c)

graph_c.add_edge("research", "write")
graph_c.add_edge("write",    "review")

# Conditional edge from review: approved -> END, give_up -> END, retry -> research
graph_c.add_conditional_edges(
    "review",             # From this node
    route_after_review,   # Call this function
    {
        "approved": END,        # If approved: stop
        "give_up":  END,        # If too many attempts: stop
        "retry":    "research"  # If retry: go back to research
    }
)

graph_c.set_entry_point("research")
app_c = graph_c.compile()

print("Running retry graph with task: 'FastAPI web framework'")
final_c = app_c.invoke({"task": "FastAPI web framework"})
print(f"\nFINAL STATE:")
print(f"  attempts: {final_c['attempts']}")
print(f"  score:    {final_c['score']}")
print(f"  approved: {final_c['approved']}")
print(f"  steps:    {final_c['steps']}")


# ==============================================================================
# PART C: Checkpointing -- Pause and Resume
# ==============================================================================

print("\n" + "=" * 65)
print("PART C: Checkpointing -- Pause and Resume")
print("=" * 65)

print("""
Checkpointing saves state after every node.
This allows: pause, resume, and audit of every step.

Simulating:
  1. Run graph until research node completes (then simulate a "crash")
  2. Resume from the checkpoint (skip research, continue from write)

C# analogy: exactly like Azure Durable Functions replay:
  the orchestration state is persisted after every Activity call.
  If the process crashes, the orchestration replays from persisted state.
""")

# Build a simple graph with checkpointing
initial_state_cp = {
    "task":     "",
    "research": "",
    "draft":    "",
    "steps":    []
}

checkpointer = InMemoryCheckpointer()

# Simple 2-node graph (research -> write)
graph_cp = SimulatedStateGraph(initial_state_cp)
graph_cp.add_node("research", research_node)   # Reuse from Part A
graph_cp.add_node("write",    write_node)
graph_cp.add_edge("research", "write")
graph_cp.add_edge("write", END)
graph_cp.set_entry_point("research")

app_cp = graph_cp.compile(checkpointer=checkpointer)

# Run the graph with a thread_id (like a session ID)
config = {"configurable": {"thread_id": "session_001"}}

print("=== Run 1: Full execution with checkpointing ===")
result1 = app_cp.invoke({"task": "python frameworks"}, config=config)
print(f"\nRun 1 completed. Draft created: {result1['draft'][:60]}...")

# Show checkpoints that were saved
history = checkpointer.get_history("session_001")
print(f"\nCheckpoints saved ({len(history)} total):")
for i, cp in enumerate(history):
    print(f"  [{i+1}] After node '{cp['node']}' -> state keys: {list(cp['state'].keys())}")

# Show the latest checkpoint
latest = checkpointer.load_latest("session_001")
print(f"\nLatest checkpoint: after node '{latest['node']}'")
print(f"  research: {latest['state']['research'][:60]}...")
print(f"  draft:    {latest['state']['draft'][:60]}...")


# ==============================================================================
# PART D: Execution Trace
# ==============================================================================

print("\n" + "=" * 65)
print("PART D: Execution Trace and Graph Visualization")
print("=" * 65)

print("""
LangGraph records which nodes ran and how it routed between them.
This is the execution_log -- useful for debugging.

Real LangGraph: app.get_graph().draw_ascii()
Here: we print our simulated log.
""")

# Run the retry graph and print its execution trace
graph_trace = SimulatedStateGraph(initial_state_c)
graph_trace.add_node("research", research_node_c)
graph_trace.add_node("write",    write_node_c)
graph_trace.add_node("review",   review_node_c)
graph_trace.add_edge("research", "write")
graph_trace.add_edge("write",    "review")
graph_trace.add_conditional_edges("review", route_after_review,
                                  {"approved": END, "give_up": END, "retry": "research"})
graph_trace.set_entry_point("research")

app_trace = graph_trace.compile()
print("Running graph to collect execution trace...")
final_trace = app_trace.invoke({"task": "transformers"})

print("\nEXECUTION TRACE:")
for line in graph_trace.execution_log:
    print(f"  {line}")

print(f"\nFinal state: attempts={final_trace['attempts']}, approved={final_trace['approved']}, score={final_trace['score']}")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("SUMMARY - LangGraph")
print("=" * 65)
print("""
WHAT WE BUILT:
  Part A: Minimal 2-node graph (research -> write -> END)
  Part B: Conditional edge with retry loop (review gates the workflow)
  Part C: Checkpointing (save/restore state after every node)
  Part D: Execution trace (see what ran and how routing decisions were made)

KEY PATTERNS:
  1. State = shared dict that flows through ALL nodes
  2. Nodes = functions: (state) -> partial state update (only changed keys)
  3. Normal edges = always go A -> B
  4. Conditional edges = router function returns a key -> lookup in mapping
  5. Checkpoints = saved state after every node (enables resume)
  6. Retry loop = conditional edge routes back to earlier node

C# ANALOGY RECAP:
  StateGraph         = StateMachine<T> or Durable Functions orchestration
  Node               = IActivity<TInput, TOutput>
  Conditional edge   = switch statement inside orchestration
  Checkpointing      = Durable Functions event-sourcing / replay
  State dict         = IOrchestrationContext input bag

REAL LANGGRAPH MIGRATION:
  pip install langgraph langchain-openai
  from langgraph.graph import StateGraph, END
  from langgraph.checkpoint.memory import MemorySaver
  # Replace SimulatedStateGraph -> StateGraph (same API!)
  # Replace InMemoryCheckpointer -> MemorySaver (same API!)
""")

print("=" * 65)
print("END OF EXAMPLE 07")
print("=" * 65)

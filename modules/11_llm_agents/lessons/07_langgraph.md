# Lesson 07: LangGraph -- State Machines for Agent Workflows

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain what LangGraph is and why it exists
2. Describe nodes, edges, and conditional routing in a StateGraph
3. Explain what "state" is in LangGraph and how it flows through the graph
4. Describe checkpointing and why it enables pause/resume
5. Compare LangGraph to a plain ReAct loop

---

## GLOSSARY

```
LangGraph:
  A Python library for building agent workflows as graphs (nodes + edges).
  From LangChain Inc. (2024). Key idea: instead of a simple loop, you define
  a state machine where each step is a node and the routing between steps is
  an edge. Supports complex workflows: parallel branches, cycles, human-in-the-loop.
  C# analogy: like Windows Workflow Foundation (WF) or Azure Durable Functions --
  workflow state is explicit and persisted.

StateGraph:
  The core LangGraph class. You add nodes and edges to it.
  At runtime it executes like a state machine: start -> node A -> node B -> END.
  C# analogy: like StateMachine<TState> or IOrchestrationContext in Durable Functions.

Node:
  A single step in the graph. It's just a Python function:
    def my_node(state: dict) -> dict:
        # read from state, do work, return updated state
  The function receives the current state and returns a (partial) updated state.
  C# analogy: like an IActivity<T> in Durable Functions, or a workflow step/task.

Edge:
  A connection between nodes. Tells the graph: "after node A, go to node B".
  Two types:
    Normal edge: always go from A to B.
    Conditional edge: run a function that returns a string; route to the
                      node with that name (or END).
  C# analogy: like a transition in WF or routing logic in a pipeline.

State:
  A shared dictionary (or TypedDict) that all nodes read from and write to.
  It travels through the entire graph, accumulating results at each step.
  C# analogy: like an orchestration context/input bag in Durable Functions,
  or a shared Dictionary<string, object> passed through a pipeline.

TypedDict:
  A Python dictionary with defined key types (type hints).
  Used to define the shape of the graph's state.
  C# analogy: like a strongly-typed DTO or record class.

Conditional Edge:
  An edge where routing depends on the current state.
  You provide a function: (state) -> "node_name" or END.
  Example: if state["answer"] is good -> END, else -> retry_node.
  C# analogy: like a switch statement in a workflow activity or
  Activity.ContinueAsNew() with a condition.

END:
  A special constant in LangGraph meaning "stop the graph here".
  When an edge routes to END, the graph execution stops and returns the state.

Checkpointing:
  LangGraph can save the graph's state after every node execution.
  This lets you: pause a workflow, resume it later, or replay from any step.
  C# analogy: like Durable Functions' replay/event sourcing -- the orchestration
  state is stored durably so it survives restarts.

Human-in-the-loop:
  A workflow pattern where the graph pauses and waits for human input before
  continuing. Made easy by LangGraph's checkpointing: save state, wait, resume.
  C# analogy: like a Durable Functions human approval gate.

Subgraph:
  A full LangGraph graph used as a single node inside another graph.
  Enables modular, nested workflows.
  C# analogy: like calling a child workflow from an orchestration.
```

---

## Part 1: Why Not Just Use a Loop?

In Lesson 03 you learned the ReAct loop:

```
while not done:
    thought = llm.think(state)
    action  = llm.decide_action(thought)
    result  = tool.run(action)
    state.update(result)
```

This works for SIMPLE agents. Problems appear at scale:

```
PROBLEM 1: No explicit workflow structure
  A loop has no map of what steps are possible.
  Hard to add branches: "if this type of task, do these steps; else, do those."
  Hard to review or visualize what the agent actually does.

PROBLEM 2: Hard to add parallelism
  A plain loop is sequential: step 1 -> step 2 -> step 3.
  What if steps 2 and 3 can run at the same time?
  Adding parallelism to a loop requires complex threading code.

PROBLEM 3: No resume after failure
  If the agent crashes at step 7 of 10, you restart from step 1.
  For long-running tasks (hours of work), this is unacceptable.
  A graph with checkpointing can resume from step 7.

PROBLEM 4: Hard to add human approval gates
  "After the agent drafts a PR, wait for a human to approve before merging."
  In a plain loop: hard. In LangGraph: one line (interrupt_before=["merge_node"]).

SOLUTION: LangGraph models the workflow as a state machine.
  Every step = a node.
  Every possible transition = an edge.
  The full workflow is visible, testable, and resumable.
```

---

## Part 2: Core Concepts -- Nodes, Edges, State

### The State

```python
from typing import TypedDict, List

# Define the "shape" of state that flows through the graph
# C# analogy: like a DTO record -- strongly typed
class AgentState(TypedDict):
    messages:  List[str]   # Conversation history (grows as we add messages)
    task:      str          # The current task being worked on
    result:    str          # Final result (empty until done)
    attempts:  int          # How many times we've tried
    approved:  bool         # Has a human approved the result?
```

### Nodes

```python
# A node is just a function: receives state, returns partial state update
# LangGraph merges the returned dict into the existing state automatically.
# C# analogy: like an IActivity that receives + returns a context bag

def research_node(state: AgentState) -> dict:
    """Look up information for the task."""
    task = state["task"]
    # ... call LLM or tool here ...
    research = f"[Research results for: {task}]"
    # Return ONLY what changed -- LangGraph merges this into the full state
    return {"messages": state["messages"] + [f"Research: {research}"]}

def write_node(state: AgentState) -> dict:
    """Write a response based on research."""
    last_message = state["messages"][-1]   # Last message = research results
    result = f"[Draft based on: {last_message}]"
    return {
        "result":   result,
        "attempts": state["attempts"] + 1,
        "messages": state["messages"] + [f"Draft: {result}"]
    }

def review_node(state: AgentState) -> dict:
    """Check if the result is good enough."""
    result = state["result"]
    is_good = len(result) > 50    # Simulated quality check
    return {"approved": is_good}
```

### Edges

```python
from langgraph.graph import StateGraph, END

# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("research", research_node)
graph.add_node("write",    write_node)
graph.add_node("review",   review_node)

# Add normal edges (always go A -> B)
graph.add_edge("research", "write")    # After research, always go to write
graph.add_edge("write",    "review")   # After write, always go to review

# Add conditional edge (routing depends on state)
def route_after_review(state: AgentState) -> str:
    """Decide: are we done or do we try again?"""
    if state["approved"]:
        return END                # Done! Stop the graph.
    elif state["attempts"] >= 3:
        return END                # Tried 3 times, give up
    else:
        return "research"         # Not good enough, try again from research

graph.add_conditional_edges(
    "review",               # From this node
    route_after_review,     # Call this function to decide the next node
    {
        "research": "research",   # "research" string -> route to research node
        END: END                  # END constant -> stop the graph
    }
)

# Set entry point
graph.set_entry_point("research")

# Compile into a runnable
app = graph.compile()
```

---

## Part 3: Running the Graph

```python
# Create initial state
initial_state = AgentState(
    messages=[],
    task="Write a summary of Python web frameworks",
    result="",
    attempts=0,
    approved=False
)

# Run the graph -- returns the final state after all nodes complete
final_state = app.invoke(initial_state)

print(final_state["result"])    # The final output
print(final_state["attempts"])  # How many tries it took
```

```
WHAT HAPPENS INSIDE:
  1. Graph starts at "research" node (entry point)
  2. research_node runs -> updates messages
  3. Normal edge -> write_node runs -> updates result, attempts, messages
  4. Normal edge -> review_node runs -> updates approved
  5. Conditional edge -> route_after_review(state) called
     -> if approved: END (stop)
     -> else if attempts >= 3: END (stop)
     -> else: "research" (loop back, try again)
  6. Graph returns final state when it reaches END

C# analogy: like a Durable Functions orchestration:
  context.CallActivityAsync("Research", input)
  context.CallActivityAsync("Write", state)
  context.CallActivityAsync("Review", state)
  if (!state.Approved && state.Attempts < 3)
      context.ContinueAsNew(state)   // loop back
```

---

## Part 4: Visualization -- The Graph Diagram

```
EXAMPLE: Research -> Write -> Review graph with retry loop

  START
    |
    v
 [research] <---------+
    |                  |
    v                  |
  [write]             | "retry" edge (if not approved)
    |                  |
    v                  |
 [review] ------------+
    |
    | (if approved OR attempts >= 3)
    v
   END

This graph can loop! Unlike a plain pipeline, it revisits nodes.
LangGraph tracks state across loops automatically.
C# analogy: ContinueAsNew() in Durable Functions -- restart orchestration with new state.
```

---

## Part 5: Checkpointing -- Pause and Resume

```python
from langgraph.checkpoint.memory import MemorySaver

# Create a checkpointer (saves state in memory; use SqliteSaver for disk)
checkpointer = MemorySaver()

# Compile with checkpointing enabled
app = graph.compile(checkpointer=checkpointer)

# Run with a thread_id (like a session ID)
config = {"configurable": {"thread_id": "session_42"}}
state  = app.invoke(initial_state, config=config)

# --- Agent can be stopped here ---
# --- Even if your program restarts ---

# Resume later with the same thread_id -- picks up from where it left off
resumed_state = app.invoke(None, config=config)
# None means "don't start over -- resume from checkpoint"
```

```
HOW CHECKPOINTING WORKS:
  After every node execution, LangGraph saves:
    - Current state (all fields)
    - Which node just ran
    - Which node is next
  Stored in MemorySaver (RAM) or SqliteSaver (disk) or custom backend.

  C# analogy: exactly like Azure Durable Functions -- the framework
  serializes orchestration state after every Activity, so the process
  can restart from the same point after a crash or scale-out event.

HUMAN-IN-THE-LOOP with checkpointing:
  app = graph.compile(
      checkpointer=checkpointer,
      interrupt_before=["review"]   # Pause BEFORE the review node
  )
  state = app.invoke(initial_state, config=config)
  # Graph runs research + write, then PAUSES before review
  # Human checks the draft in state["result"]
  # Human approves: app.invoke(None, config=config)  -> continues from review
```

---

## Part 6: Parallel Branches

LangGraph supports parallel node execution (fan-out / fan-in):

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class ParallelState(TypedDict):
    task:         str
    research_a:   str    # Result from branch A
    research_b:   str    # Result from branch B
    final_result: str

# Two independent research nodes (run in parallel)
def branch_a(state):
    return {"research_a": f"[Branch A results for: {state['task']}]"}

def branch_b(state):
    return {"research_b": f"[Branch B results for: {state['task']}]"}

# Merge node (runs AFTER both branches complete)
def merge(state):
    combined = state["research_a"] + "\n" + state["research_b"]
    return {"final_result": f"[Merged: {combined}]"}

graph = StateGraph(ParallelState)
graph.add_node("branch_a", branch_a)
graph.add_node("branch_b", branch_b)
graph.add_node("merge",    merge)

# Fan-out: both branches start after START
graph.set_entry_point("branch_a")    # Simplified; real parallel needs special syntax
graph.add_edge("branch_a", "merge")
graph.add_edge("branch_b", "merge")
graph.add_edge("merge", END)
```

```
C# analogy:
  Parallel branches = Task.WhenAll([branchA, branchB])
  Merge node = the continuation after Task.WhenAll completes
```

---

## Part 7: LangGraph vs Plain ReAct Loop

```
+-----------------+---------------------------+--------------------------------+
| Feature         | Plain ReAct Loop (L03)    | LangGraph                      |
+-----------------+---------------------------+--------------------------------+
| Workflow        | Implicit (inside loop)    | Explicit graph (nodes + edges) |
| Visualization   | Hard to see steps         | graph.get_graph().draw_ascii() |
| Branching       | if/else in the loop       | Conditional edges              |
| Parallelism     | Manual threading          | Built-in fan-out/fan-in        |
| Resume/restart  | Not supported             | Checkpointing (MemorySaver)    |
| Human-in-loop   | Hard to add              | interrupt_before=[node]        |
| Debugging       | Print statements          | State visible at every node    |
| Best for        | Simple agents, fast POCs  | Production, complex workflows  |
+-----------------+---------------------------+--------------------------------+

Rule of thumb:
  - Prototype / simple agent -> plain ReAct loop (less code)
  - Production / complex multi-step -> LangGraph (explicit, resumable)
```

---

## Key Takeaways

1. LangGraph models agent workflows as state machines: nodes (steps) + edges (transitions).

2. State is a shared dict that flows through all nodes, accumulating results.

3. Conditional edges enable dynamic routing: decide the next node based on current state.

4. Cycles are supported: nodes can loop back (retry / self-correction patterns).

5. Checkpointing saves state after every node — enables pause, resume, human approval gates.

6. LangGraph vs ReAct loop: use LangGraph for production workflows that need explicit structure, branching, or resumability.

7. C# analogy: LangGraph ≈ Azure Durable Functions (explicit state, resumable, visual workflow).

---

*Next: Lesson 08 — Building a Multi-Agent System with MCP Tools + LangGraph Routing*

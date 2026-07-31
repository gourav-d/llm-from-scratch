# Lesson 08: Building a Multi-Agent System with MCP + LangGraph

## Learning Objectives

By the end of this lesson, you will be able to:
1. Describe how MCP tools plug into a LangGraph agent
2. Explain the Orchestrator → Specialist routing pattern in LangGraph
3. Understand how checkpointing enables human-in-the-loop in multi-agent systems
4. Explain why this architecture dominates real-world LLM deployments (2024-2025)
5. Sketch a LangGraph + MCP architecture for a given problem

---

## GLOSSARY

```
Tool Node:
  A LangGraph node whose only job is to call a tool (MCP or local) and
  return the result into state. Keeps tool-call logic separate from LLM logic.
  C# analogy: like a dedicated Activity in Durable Functions just for external calls.

ToolMessage:
  A message type in LangChain/LangGraph representing a tool's response.
  Goes into the messages list so the LLM can "see" the tool result.
  C# analogy: like a typed event in an event-sourced system.

create_react_agent:
  A LangGraph helper that builds a ReAct-style agent over a list of tools.
  Returns a compiled graph. Replaces writing nodes + edges manually for simple agents.
  C# analogy: like a factory method that returns a pre-configured orchestration.

Router Agent (Supervisor):
  An orchestrator agent that reads a task and decides: which specialist agent
  or which tool should handle it? Returns a routing decision (a string: agent name).
  C# analogy: like a CQRS command dispatcher or a Mediator.Route() call.

Specialist Agent:
  A LangGraph subgraph compiled to handle ONE category of task.
  Examples: CodeAgent (runs code tools), ResearchAgent (uses web search + DB).
  C# analogy: like an ICommandHandler<T> -- handles one command type.

Shared State (Multi-Agent):
  In a multi-agent LangGraph, ALL agents share the same state dict.
  The router writes "next_agent" to state; the subgraph reads "task" from state.
  C# analogy: like a shared IOrchestrationContext passed between child orchestrations.

Interrupt Point:
  A node where LangGraph pauses and waits for external input (human or another system).
  Set with interrupt_before=["node_name"] at compile time.
  C# analogy: like a WaitForExternalEvent call in Durable Functions.

Agent Handoff:
  The pattern where the router writes the next agent's name into state,
  the conditional edge reads it, and routes to that agent's subgraph.
  C# analogy: like calling context.CallSubOrchestratorAsync("AgentName", input).
```

---

## Part 1: The Full Architecture

```
USER QUERY
     |
     v
+--------------------+
| ROUTER (Supervisor)|  <- LangGraph node
| LLM decides:       |  <- Reads state["task"], outputs state["next_agent"]
|   which agent?     |
+--------------------+
     |
     | (conditional edge reads state["next_agent"])
     |
     +----------+----------+----------+
     |          |          |          |
     v          v          v          v
[CodeAgent] [ResearchAgent] [MathAgent] [END]
     |          |          |
     v          v          v
  MCP Tools  MCP Tools  stdlib
  (exec_py)  (web_search  (math.eval)
  (lint_py)   db_query)
     |          |          |
     v          v          v
  result -> state["result"]
     |          |          |
     +----------+----------+
                |
                v
+--------------------+
| ROUTER again?      |  <- Is task fully done? Route to END or another agent.
+--------------------+
```

```
HOW IT MAPS TO C#:
  Router      = IMediator.Send() -- routes request to correct handler
  Agents      = IRequestHandler<T> -- handles specific request type
  MCP tools   = IHttpClientFactory + named clients -- each tool is a service call
  State       = IOrchestrationContext -- shared across all sub-orchestrations
  Checkpoint  = Durable Functions replay -- state persisted, resumable
```

---

## Part 2: Wiring MCP Tools into LangGraph

LangGraph agents call tools through the messages list. The pattern:

```
AGENT NODE (LLM thinks + decides which tool to call)
     |
     | -> writes ToolCall to state["messages"]
     v
TOOL NODE (executes the tool call, returns result)
     |
     | -> writes ToolMessage to state["messages"]
     v
AGENT NODE (LLM reads tool result, decides next step)
     |
     | -> if done: route to END
     | -> if more tools needed: route back to TOOL NODE
```

```python
# Pseudocode -- real code in example_08.py

# Step 1: Define tools as Python functions (or wrap MCP tools)
def search_web(query: str) -> str:
    """Search the web and return top results."""
    # In production: calls mcp-server-brave-search via MCP protocol
    return f"[Web results for: {query}]"

def run_python(code: str) -> str:
    """Execute Python code and return output."""
    # In production: calls mcp-server-python-exec via MCP protocol
    return f"[Output: executed {len(code)} chars of code]"

# Step 2: Bind tools to a model (creates a model that knows about these tools)
tools  = [search_web, run_python]
model  = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# Step 3: Build the graph
# create_react_agent() builds the Agent + ToolNode graph automatically
from langgraph.prebuilt import create_react_agent
agent_graph = create_react_agent(model, tools)

# That's it! The graph handles:
#   - LLM decides which tool to call
#   - Tool executes
#   - Result feeds back to LLM
#   - LLM decides: done or call another tool?
```

---

## Part 3: The Router + Specialist Pattern

For multi-agent routing, we build two layers:

```python
# Layer 1: Specialist agents (each handles one domain)

# ResearchAgent: uses web search + database tools
research_tools = [search_web, query_database]
research_model  = ChatOpenAI(model="gpt-4o").bind_tools(research_tools)
research_agent  = create_react_agent(research_model, research_tools)

# CodeAgent: uses code execution + linting tools
code_tools  = [run_python, lint_code]
code_model  = ChatOpenAI(model="gpt-4o").bind_tools(code_tools)
code_agent  = create_react_agent(code_model, code_tools)

# Layer 2: Router (supervisor) -- decides which specialist to call
def router_node(state: AgentState) -> dict:
    """LLM reads the task and decides which specialist to use."""
    task = state["task"]
    # LLM call to decide routing
    decision = llm.invoke(f"""
        Task: {task}
        Available agents: research_agent, code_agent, FINISH
        Which agent should handle this? Reply with EXACTLY one of those names.
    """)
    return {"next_agent": decision.content.strip()}


# Layer 3: Subgraph wrappers (run the specialist agent, write result to state)
def run_research_agent(state: AgentState) -> dict:
    result = research_agent.invoke({"messages": [state["task"]]})
    return {"result": result["messages"][-1].content}

def run_code_agent(state: AgentState) -> dict:
    result = code_agent.invoke({"messages": [state["task"]]})
    return {"result": result["messages"][-1].content}


# Layer 4: Build the supervisor graph
graph = StateGraph(AgentState)
graph.add_node("router",          router_node)
graph.add_node("research_agent",  run_research_agent)
graph.add_node("code_agent",      run_code_agent)

graph.set_entry_point("router")

# After each specialist, go back to router (maybe more work needed)
graph.add_edge("research_agent", "router")
graph.add_edge("code_agent",     "router")

# Router conditional edge: which specialist? or done?
def route(state: AgentState) -> str:
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "research_agent": return "research_agent"
    if next_agent == "code_agent":     return "code_agent"
    return END

graph.add_conditional_edges("router", route)

app = graph.compile(checkpointer=MemorySaver())
```

---

## Part 4: Human-in-the-Loop Approval Gate

```python
# Add a human approval gate BEFORE the code agent runs
# (Don't auto-execute code without human review)

app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["code_agent"]   # Pause before running code
)

config = {"configurable": {"thread_id": "user_42"}}

# Run until the interrupt point
state = app.invoke({"task": "Write and run a script that lists all files"}, config)
# Graph runs: router -> routes to code_agent -> PAUSES

# Human reviews what the code agent would do
print("Agent wants to run code. Current plan:", state["messages"][-1])

# Human approves -> continue
state = app.invoke(None, config)    # None = resume from checkpoint
# Graph continues: code_agent runs -> router -> END

print("Final result:", state["result"])
```

```
C# ANALOGY for the approval gate:
  // Durable Functions human approval gate
  var approvalTask = context.WaitForExternalEvent<bool>("HumanApproval");
  bool approved = await approvalTask;
  if (approved)
      await context.CallActivityAsync("RunCode", input);

LangGraph version is even simpler:
  interrupt_before=["code_agent"]
  -> Graph pauses. State is saved. Resume when human says go.
  No WaitForExternalEvent plumbing needed.
```

---

## Part 5: Why This Architecture Dominates (2024-2025)

```
CURRENT REAL-WORLD PATTERNS (as of 2025):

  Claude Code (Anthropic):
    Uses MCP for tools (filesystem, GitHub, database)
    Uses a ReAct-style loop internally
    Human approves destructive actions (interrupt gates)

  GitHub Copilot Workspace:
    LangGraph-style orchestration
    Specialist agents for: planning, coding, testing, PR creation
    Human-in-the-loop at key steps

  Enterprise AI Platforms (Blackline, ServiceNow, Salesforce):
    Router (supervisor) dispatches to domain agents
    Each domain agent has MCP-connected tools (CRM, ERP, database)
    Checkpointing enables long-running multi-day workflows

  Why this combination wins:
  +-------------------+------------------------------------------+
  | MCP               | Any tool, any language, plug-and-play    |
  | LangGraph         | Explicit workflow, resumable, auditable  |
  | Together          | Scalable enterprise AI architecture      |
  +-------------------+------------------------------------------+

EMPLOYABILITY NOTE:
  65% of LLM engineering job postings (2025) require:
    - Experience with agent frameworks (LangGraph, AutoGen, CrewAI)
    - Understanding of tool protocols (MCP, function calling)
    - Knowledge of human-in-the-loop patterns
  This lesson + examples + exercises cover all three.
```

---

## Part 6: Debugging Multi-Agent LangGraph Systems

```
COMMON ISSUES AND FIXES:

  Issue: Agent loops forever
  Fix:   Add a max_iterations counter to state.
         Conditional edge: if state["iterations"] > 10 -> END

  Issue: Agent calls wrong specialist
  Fix:   Improve router prompt. Add few-shot examples.
         Or: add a validation node after router that checks the decision.

  Issue: Tool result too large for context
  Fix:   Truncate in the tool node. Return "first 2000 chars + ...".
         Or: use a summarizer agent between tool and main agent.

  Issue: Cannot see what agent did
  Fix:   state["messages"] has the full trace. Print it.
         Or: app.get_state(config).values shows full state at any point.

  Issue: Want to retry a failed node
  Fix:   Wrap node in try/except. Write error to state["error"].
         Conditional edge: if state["error"] -> retry_node, else -> next_node.

C# analogy:
  Loops    = set maxRetries in Polly / set maxReplays in Durable Functions
  Tracing  = Application Insights traces / structured logging per activity
  Retry    = Polly retry policy / ContinueAsNew with retry state
```

---

## Part 7: Full System Diagram -- MCP + LangGraph

```
+----------------------------+
|  USER: "Find all open PRs  |
|  with failing tests and    |
|  post a summary to Slack"  |
+----------------------------+
             |
             v
   +------------------+
   | LangGraph App    |
   |                  |
   | +-------------+  |
   | | ROUTER NODE |  |   <- LLM decides: github_agent first
   | +-------------+  |
   |        |         |
   |        v         |
   | +-------------+  |
   | | GITHUB AGENT|  |   <- Calls MCP: github_mcp_server
   | |             |--|--> tools/call: list_prs  -> [{pr: 42, tests: "FAIL"}]
   | |             |--|--> tools/call: get_pr_details -> {...}
   | +-------------+  |
   |        |         |
   |  (back to router)|
   |        |         |
   |        v         |
   | +-------------+  |
   | | ROUTER NODE |  |   <- LLM decides: slack_agent next
   | +-------------+  |
   |        |         |
   |        v         |
   | +-------------+  |
   | | SLACK AGENT |  |   <- Calls MCP: slack_mcp_server
   | |             |--|--> tools/call: post_message -> "Message sent"
   | +-------------+  |
   |        |         |
   |  (back to router)|
   |        |         |
   |        v         |
   | +-------------+  |
   | | ROUTER NODE |  |   <- LLM decides: FINISH (task complete)
   | +-------------+  |
   |        |         |
   |        v         |
   |       END        |
   +------------------+
             |
             v
   "Found 3 PRs with failing tests. Summary posted to #dev-alerts."
```

---

## Key Takeaways

1. MCP + LangGraph is the dominant production pattern for 2024-2025 LLM agents.

2. MCP provides the tools (any language, any server, discovered at runtime).

3. LangGraph provides the workflow (nodes, edges, state, checkpointing).

4. Router pattern: supervisor LLM reads task → decides which specialist agent → routes via conditional edge.

5. Human-in-the-loop: `interrupt_before=["node"]` pauses graph; `app.invoke(None, config)` resumes.

6. Debugging: `state["messages"]` is the full trace; `app.get_state(config)` shows current state.

7. Employability: this architecture appears in 65% of 2025 LLM engineering job postings.

---

*Next: Examples and Exercises to build and test this yourself!*

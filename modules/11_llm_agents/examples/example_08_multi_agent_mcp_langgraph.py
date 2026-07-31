"""
Example 08: Multi-Agent System with MCP + LangGraph
====================================================

GLOSSARY
--------
Router (Supervisor):
  An orchestrator agent (LLM or rules-based) that reads a task and
  decides which specialist agent should handle it.
  C# analogy: CQRS command dispatcher / MediatR.

Specialist Agent:
  A focused agent that handles ONE category of task.
  Uses tools appropriate for its domain.
  C# analogy: IRequestHandler<T> -- handles one command type.

Handoff:
  Router writes "next_agent" to state; conditional edge routes there.
  C# analogy: context.CallSubOrchestratorAsync("AgentName", input).

Interrupt Gate:
  Graph pauses before a specified node, waits for human approval.
  C# analogy: context.WaitForExternalEvent<bool>("ApprovalEvent").

Tool Node:
  A graph node whose only job is to execute a tool call and return the result.
  Keeps tool logic separate from LLM decision logic.

WHAT THIS EXAMPLE SHOWS
------------------------
Part A: Router + two specialist agents (research + code)
Part B: Agents using MCP tools (from Example 06 servers)
Part C: Human-in-the-loop approval gate (interrupt before code execution)
Part D: Full end-to-end run with trace

LIBRARIES NEEDED
----------------
  None (pure Python stdlib)
  Builds on SimulatedMCPServer/Client from example_06
  Builds on SimulatedStateGraph from example_07
"""

import json    # For displaying structured data
import re      # For parsing tasks

print("=" * 65)
print("EXAMPLE 08: Multi-Agent MCP + LangGraph")
print("=" * 65)


# ==============================================================================
# REUSE: Copy the simulator classes (condensed versions)
# In a real project these would be imported from example_06 and example_07.
# ==============================================================================

END = "__END__"

class SimulatedMCPServer:
    def __init__(self, name):
        self.server_name     = name
        self.tools_registry  = {}
    def register_tool(self, schema, handler):
        self.tools_registry[schema["name"]] = {"schema": schema, "handler": handler}
    def handle_request(self, method, params):
        if method == "tools/list":
            return {"result": {"tools": [e["schema"] for e in self.tools_registry.values()]}}
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {})
            if name not in self.tools_registry:
                return {"error": f"Tool '{name}' not found"}
            try:
                result = self.tools_registry[name]["handler"](args)
                return {"result": {"content": [{"type": "text", "text": str(result)}]}}
            except Exception as e:
                return {"error": str(e)}

class SimulatedMCPClient:
    def __init__(self):
        self.servers   = {}
        self.all_tools = []
    def connect(self, server):
        self.servers[server.server_name] = server
        resp = server.handle_request("tools/list", {})
        for tool in resp["result"]["tools"]:
            self.all_tools.append({"server": server.server_name, "schema": tool})
    def call_tool(self, tool_name, arguments):
        for entry in self.all_tools:
            if entry["schema"]["name"] == tool_name:
                server   = self.servers[entry["server"]]
                response = server.handle_request("tools/call", {"name": tool_name, "arguments": arguments})
                if "error" in response:
                    return f"ERROR: {response['error']}"
                return response["result"]["content"][0]["text"]
        return f"ERROR: Tool '{tool_name}' not found"
    def list_tools(self):
        return [f"{e['server']}/{e['schema']['name']}" for e in self.all_tools]

class SimulatedStateGraph:
    def __init__(self, schema):
        self.initial_state     = schema.copy()
        self.nodes             = {}
        self.normal_edges      = {}
        self.conditional_edges = {}
        self.entry_point       = None
        self.trace             = []    # Execution trace
    def add_node(self, name, fn):
        self.nodes[name] = fn
    def add_edge(self, a, b):
        self.normal_edges[a] = b
    def add_conditional_edges(self, a, fn, mapping):
        self.conditional_edges[a] = (fn, mapping)
    def set_entry_point(self, name):
        self.entry_point = name
    def compile(self, interrupt_before=None):
        return CompiledGraph(self, interrupt_before or [])

class CompiledGraph:
    def __init__(self, graph, interrupt_before):
        self.graph            = graph
        self.interrupt_before = interrupt_before
        self._paused_state    = None    # Saved state when interrupted
        self._resume_node     = None    # Node to resume at

    def invoke(self, initial_state, config=None):
        if initial_state is None and self._paused_state is not None:
            # Resume from interrupt
            state        = self._paused_state
            current_node = self._resume_node
            self._paused_state = None
            self._resume_node  = None
            print(f"\n  [Graph] Resuming from interrupt at node: '{current_node}'")
        else:
            state        = {**self.graph.initial_state, **(initial_state or {})}
            current_node = self.graph.entry_point
            self.graph.trace = []

        max_steps = 25
        steps     = 0

        while current_node != END and current_node is not None and steps < max_steps:
            steps += 1

            if current_node in self.interrupt_before:
                print(f"\n  [Graph] --- INTERRUPTED before: '{current_node}' ---")
                print(f"  [Graph] Waiting for human approval. Call invoke(None) to resume.")
                self._paused_state = state
                self._resume_node  = current_node
                return state    # Return paused state

            print(f"\n  [Graph] >>> Node: '{current_node}'")
            self.graph.trace.append(f"NODE: {current_node}")

            update       = self.graph.nodes[current_node](state)
            state.update(update)

            # Determine next node
            if current_node in self.graph.conditional_edges:
                fn, mapping = self.graph.conditional_edges[current_node]
                decision    = fn(state)
                next_node   = mapping.get(decision, END)
                self.graph.trace.append(f"  ROUTE: '{current_node}' -> '{next_node}' ('{decision}')")
                print(f"  [Graph] Route decision: '{decision}' -> '{next_node}'")
                current_node = next_node
            elif current_node in self.graph.normal_edges:
                current_node = self.graph.normal_edges[current_node]
            else:
                current_node = END

        if steps >= max_steps:
            print(f"  [Graph] WARNING: max steps reached.")
        print(f"\n  [Graph] === COMPLETED ({steps} nodes executed) ===")
        return state


# ==============================================================================
# SET UP MCP SERVERS
# ==============================================================================

print("\n--- Setting up MCP Servers ---")

# ---- Research Server (web search + knowledge base) ----
research_server = SimulatedMCPServer("research-server")

_knowledge_base = {
    "python":     "Python is a high-level language with simple syntax. Popular for AI/ML, web (FastAPI, Django), data science.",
    "fastapi":    "FastAPI is a modern Python web framework. Auto-generates OpenAPI docs. Uses type hints for validation. Async-native.",
    "langchain":  "LangChain is a framework for building LLM applications. Provides chains, agents, memory, and tool integration.",
    "langgraph":  "LangGraph is a library for building stateful, multi-actor workflows with LLMs. Uses nodes and edges.",
    "mcp":        "Model Context Protocol: open standard for agent-tool connections. Write tools once, use with any agent.",
    "transformers": "Transformers use self-attention. Input tokens -> embeddings -> multi-head attention -> output. BERT encoder-only, GPT decoder-only.",
}

def web_search(args):
    """Simulated web search tool."""
    query = args["query"].lower()
    for key, info in _knowledge_base.items():
        if key in query:
            return f"[Search results for '{args['query']}']\n{info}"
    return f"[Search results for '{args['query']}']\nNo specific results found."

def query_database(args):
    """Simulated database query tool."""
    table  = args.get("table", "unknown")
    filter_val = args.get("filter", "")
    return f"[DB query: SELECT * FROM {table} WHERE topic LIKE '%{filter_val}%' -- 3 rows returned]"

research_server.register_tool({
    "name": "web_search",
    "description": "Search the web for information on a topic. Returns top results.",
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"]
    }
}, web_search)

research_server.register_tool({
    "name": "query_database",
    "description": "Query the knowledge database for structured information.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "table":  {"type": "string", "description": "Table to query"},
            "filter": {"type": "string", "description": "Filter keyword"}
        },
        "required": ["table"]
    }
}, query_database)

# ---- Code Server (code execution + linting) ----
code_server = SimulatedMCPServer("code-server")

_exec_log = []    # Track what was "executed"

def run_python_code(args):
    """Simulated Python code execution."""
    code    = args["code"]
    context = args.get("context", "")
    _exec_log.append({"code": code, "context": context})
    # Simulate execution: count lines, detect obvious issues
    lines   = code.strip().split("\n")
    n_lines = len(lines)
    has_print = "print" in code
    output = f"[Executed {n_lines} lines of Python code]\n"
    if has_print:
        output += "[stdout: Hello from Python!]\n"
    output += f"[Exit code: 0]"
    return output

def lint_python_code(args):
    """Simulated Python linter."""
    code    = args["code"]
    issues  = []
    if "import *" in code:
        issues.append("W0401: Wildcard import -- avoid 'from module import *'")
    if "eval(" in code:
        issues.append("W0123: Use of eval() -- security risk")
    if len(code) > 1000:
        issues.append("C0301: Line too long")
    if not issues:
        return "[Lint passed: no issues found]"
    return "[Lint issues found]\n" + "\n".join(issues)

code_server.register_tool({
    "name": "run_python",
    "description": "Execute Python code and return stdout/stderr. Use for running scripts.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "code":    {"type": "string", "description": "Python code to execute"},
            "context": {"type": "string", "description": "Description of what this code does"}
        },
        "required": ["code"]
    }
}, run_python_code)

code_server.register_tool({
    "name": "lint_python",
    "description": "Lint Python code for style and security issues.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to lint"}
        },
        "required": ["code"]
    }
}, lint_python_code)

print("MCP Servers ready: research-server (2 tools), code-server (2 tools)")


# ==============================================================================
# PART A: Router + Specialist Agents
# ==============================================================================

print("\n" + "=" * 65)
print("PART A: Router + Two Specialist Agents")
print("=" * 65)

print("""
Architecture:
  [router] -> [research_agent] -> [router] -> ... -> END
           -> [code_agent]     -> [router] -> ...

Router reads state["task"] and writes state["next_agent"].
Conditional edge reads state["next_agent"] and routes to the right agent.
After each agent completes, we return to the router to check if more work needed.
""")

# Build MCP clients for each specialist agent
research_client = SimulatedMCPClient()
research_client.connect(research_server)    # Research agent gets research tools

code_client = SimulatedMCPClient()
code_client.connect(code_server)            # Code agent gets code tools

print(f"Research agent tools: {research_client.list_tools()}")
print(f"Code agent tools:     {code_client.list_tools()}")

# State schema for the multi-agent graph
initial_state = {
    "task":            "",     # The user's request
    "next_agent":      "",     # Router's decision: which agent next
    "research_result": "",     # Output from research agent
    "code_result":     "",     # Output from code agent
    "final_answer":    "",     # Combined final answer
    "steps_done":      [],     # Log of completed steps
    "iteration":       0       # Safety counter to prevent infinite loops
}

# ---- Router Node ----
def router_node(state: dict) -> dict:
    """
    Reads the task and current state, decides which agent to use next.
    In production: an LLM reads the task and chooses.
    Here: keyword-based routing (same decision logic as an LLM).
    """
    task      = state["task"].lower()
    iteration = state["iteration"]
    done_steps = state["steps_done"]

    print(f"    [router] Task: '{state['task']}' | Iteration: {iteration}")
    print(f"    [router] Already done: {done_steps}")

    # Simple routing logic (in production: LLM call)
    if iteration >= 3:
        next_agent = "FINISH"       # Safety: too many iterations
    elif "research" not in done_steps and any(w in task for w in ["explain", "what is", "how does", "describe"]):
        next_agent = "research_agent"
    elif "code" not in done_steps and any(w in task for w in ["write code", "run", "execute", "script"]):
        next_agent = "code_agent"
    elif "research" not in done_steps:
        next_agent = "research_agent"   # Default: research first
    else:
        next_agent = "FINISH"           # We have research, we're done

    print(f"    [router] Decision: next_agent='{next_agent}'")
    return {
        "next_agent": next_agent,
        "iteration":  iteration + 1
    }

# ---- Research Specialist Node ----
def research_agent_node(state: dict) -> dict:
    """
    Research specialist: searches web + database using MCP tools.
    Uses the research_client (connected to research-server).
    """
    task = state["task"]
    print(f"    [research_agent] Researching: '{task}'")

    # Call web search tool via MCP client
    search_result = research_client.call_tool("web_search", {"query": task})
    print(f"    [research_agent] Search result: {search_result[:80]}...")

    # Call database tool for additional structured info
    db_result = research_client.call_tool("query_database", {"table": "knowledge", "filter": task.split()[0]})
    print(f"    [research_agent] DB result: {db_result[:60]}...")

    combined = f"[Research for: '{task}']\n{search_result}\n{db_result}"

    return {
        "research_result": combined,
        "steps_done":      state["steps_done"] + ["research"]
    }

# ---- Code Specialist Node ----
def code_agent_node(state: dict) -> dict:
    """
    Code specialist: writes, lints, and runs code using MCP tools.
    Uses the code_client (connected to code-server).
    """
    task = state["task"]
    print(f"    [code_agent] Generating code for: '{task}'")

    # Generate a simple code snippet based on the task
    code = f"""
# Generated code for: {task}
def solution():
    print("Executing solution for: {task}")
    result = 42    # Simulated computation
    return result

if __name__ == "__main__":
    output = solution()
    print(f"Result: {{output}}")
""".strip()

    # Lint the code before running
    lint_result = code_client.call_tool("lint_python", {"code": code})
    print(f"    [code_agent] Lint: {lint_result}")

    # Run the code
    exec_result = code_client.call_tool("run_python", {"code": code, "context": task})
    print(f"    [code_agent] Execution: {exec_result}")

    combined = f"[Code for: '{task}']\n{code}\n\nLint: {lint_result}\nOutput: {exec_result}"

    return {
        "code_result": combined,
        "steps_done":  state["steps_done"] + ["code"]
    }

# ---- Finalize Node ----
def finalize_node(state: dict) -> dict:
    """Combine all specialist results into a final answer."""
    parts = []
    if state["research_result"]:
        parts.append(f"RESEARCH:\n{state['research_result'][:200]}")
    if state["code_result"]:
        parts.append(f"CODE:\n{state['code_result'][:200]}")
    final = "\n\n".join(parts) if parts else "No results collected."
    print(f"    [finalize] Combined {len(parts)} specialist results.")
    return {"final_answer": final}

# ---- Routing Function ----
def route_from_router(state: dict) -> str:
    """Read state["next_agent"] and return the routing key."""
    return state.get("next_agent", "FINISH")


# ---- Build the Graph ----
graph = SimulatedStateGraph(initial_state)
graph.add_node("router",          router_node)
graph.add_node("research_agent",  research_agent_node)
graph.add_node("code_agent",      code_agent_node)
graph.add_node("finalize",        finalize_node)

graph.set_entry_point("router")

# After each specialist, return to router
graph.add_edge("research_agent", "router")
graph.add_edge("code_agent",     "router")

# Router conditional: route to the right specialist (or finalize)
graph.add_conditional_edges(
    "router",
    route_from_router,
    {
        "research_agent": "research_agent",
        "code_agent":     "code_agent",
        "FINISH":         "finalize",   # When done, go to finalize
    }
)

graph.add_edge("finalize", END)

app = graph.compile()

# Run with a research task
print("\n=== Run 1: Research task ===")
result1 = app.invoke({"task": "explain how FastAPI works"})
print(f"\nFinal answer (first 300 chars):\n{result1['final_answer'][:300]}...")
print(f"Steps taken: {result1['steps_done']}")


# ==============================================================================
# PART B: Human-in-the-Loop (Interrupt Before Code Execution)
# ==============================================================================

print("\n" + "=" * 65)
print("PART B: Human-in-the-Loop -- Interrupt Gate")
print("=" * 65)

print("""
Before the code agent runs, we want human approval.
Pattern:
  1. Graph runs: router -> decides "code_agent" -> PAUSES
  2. Human sees the task and approves (or rejects)
  3. Graph resumes: code_agent runs -> finalize -> END

C# analogy:
  context.WaitForExternalEvent<bool>("HumanApproval")
  -> workflow pauses, waits for external signal
  -> resumes when signal arrives
""")

# Build same graph but with interrupt before code_agent
app_with_gate = graph.compile(interrupt_before=["code_agent"])

print("=== Run 2: Code task with approval gate ===")
print("\n--- Step 1: Run until interrupt ---")
paused_state = app_with_gate.invoke({"task": "write code to compute fibonacci numbers"})

print(f"\n[Human] Paused. Current state:")
print(f"  task:      {paused_state['task']}")
print(f"  next_agent: {paused_state['next_agent']}")
print(f"  steps_done: {paused_state['steps_done']}")

# Simulate human decision
print("\n[Human] Reviewing the task before code runs...")
user_approved = True    # Simulated: human approves
print(f"[Human] Decision: {'APPROVED' if user_approved else 'REJECTED'}")

if user_approved:
    print("\n--- Step 2: Resuming after human approval ---")
    final_state = app_with_gate.invoke(None)    # None = resume from paused state
    print(f"\nFinal answer (first 200 chars):\n{final_state['final_answer'][:200]}...")
    print(f"Steps taken: {final_state['steps_done']}")
else:
    print("[Human] Task rejected. Workflow cancelled.")


# ==============================================================================
# PART C: Full End-to-End Run with Trace
# ==============================================================================

print("\n" + "=" * 65)
print("PART C: Full End-to-End Run with Execution Trace")
print("=" * 65)

print("Running a complex task that needs both research and code...")

app_full = graph.compile()    # No interrupt for this demo
final_full = app_full.invoke({"task": "explain transformers and write a demo script"})

print("\n=== EXECUTION TRACE ===")
for line in graph.trace:
    print(f"  {line}")

print(f"\n=== FINAL ANSWER ===")
print(final_full["final_answer"][:500])
print(f"\nSteps completed: {final_full['steps_done']}")
print(f"Total iterations: {final_full['iteration']}")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("SUMMARY - Multi-Agent MCP + LangGraph")
print("=" * 65)
print("""
WHAT WE BUILT:
  - 2 MCP servers: research-server (web_search, query_database)
                   code-server (run_python, lint_python)
  - 2 specialist agents: ResearchAgent + CodeAgent (each uses its MCP client)
  - 1 Router node: keyword routing (in production: LLM call)
  - 1 LangGraph: router -> specialists -> router -> finalize -> END
  - Human-in-the-loop: interrupt_before=["code_agent"]

KEY PATTERNS:
  1. Router writes "next_agent" to state
     Conditional edge reads it and routes to the right specialist
  2. After each specialist: return to router (might need more agents)
  3. Specialists use MCP tools via SimulatedMCPClient
  4. interrupt_before=["code_agent"] pauses for human review
  5. invoke(None) resumes from the paused state

REAL MIGRATION:
  pip install langgraph langchain-openai langchain-community mcp
  Replace SimulatedMCPServer  -> real MCP servers (npm packages or Python)
  Replace SimulatedMCPClient  -> use langchain_mcp or manual MCP connection
  Replace SimulatedStateGraph -> from langgraph.graph import StateGraph
  Replace routing keywords    -> use an LLM to make routing decisions

C# ANALOGY RECAP:
  Router        = IMediator.Send() / CQRS dispatcher
  Specialist    = IRequestHandler<T> -- handles one domain
  MCP tools     = IHttpClientFactory named clients
  Interrupt     = context.WaitForExternalEvent<bool>()
  Graph state   = IOrchestrationContext

EMPLOYABILITY:
  This architecture (LangGraph + MCP + human-in-the-loop) is in
  65% of LLM engineering job descriptions (2025).
  Mastering it puts you ahead of most applicants.
""")

print("=" * 65)
print("END OF EXAMPLE 08")
print("=" * 65)

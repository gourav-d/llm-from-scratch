"""
=============================================================================
MODULE 11 - EXERCISE 08: Multi-Agent System with Router
=============================================================================

YOUR TASK:
  Build a multi-agent system:
    - A router that decides which specialist to use
    - A research specialist (uses a knowledge lookup tool)
    - A math specialist (uses a calculator tool)
    - A LangGraph-style state machine that connects them

RULES:
  - Only Python stdlib
  - Read each docstring carefully
  - Tests at the bottom verify your work

RUN WITH:  python exercise_08_multi_agent.py
=============================================================================
"""

import re


# =============================================================================
# CONSTANTS
# =============================================================================

END = "__END__"


# =============================================================================
# EXERCISE 1: Implement the Router
# =============================================================================

def router_node(state: dict) -> dict:
    """
    Reads state["task"] and decides which specialist agent to use next.
    Writes the decision to state["next_agent"].

    Routing rules (check in this order):
      1. If state["task"] is empty:
           next_agent = "FINISH"

      2. If any of these words in task (case-insensitive):
           "calculate", "compute", "add", "subtract", "multiply",
           "divide", "sum", "square", "math", "+", "-", "*", "/"
           AND "math" not already in state["done_agents"]:
           next_agent = "math_agent"

      3. If any of these words in task (case-insensitive):
           "explain", "what is", "describe", "tell me about",
           "how does", "information about"
           AND "research" not already in state["done_agents"]:
           next_agent = "research_agent"

      4. Otherwise:
           next_agent = "FINISH"

    Also increments state["router_calls"] by 1.

    Parameters:
      state: dict with keys "task" (str), "done_agents" (list), "router_calls" (int)

    Returns:
      dict with "next_agent" (str) and "router_calls" (int)

    EXAMPLES:
      task="calculate 5 + 3", done_agents=[] -> next_agent="math_agent"
      task="explain Python",  done_agents=[] -> next_agent="research_agent"
      task="",                done_agents=[] -> next_agent="FINISH"
      task="calculate 5 + 3", done_agents=["math"] -> next_agent="FINISH"
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 2: Research Specialist Node
# =============================================================================

KNOWLEDGE_BASE = {
    "python":     "Python is a high-level, interpreted programming language. Known for readability and large ecosystem.",
    "fastapi":    "FastAPI is a modern Python web framework. Fast, async, auto-generates OpenAPI docs.",
    "llm":        "LLMs (Large Language Models) are neural networks trained on massive text. Examples: GPT-4, Claude.",
    "mcp":        "MCP (Model Context Protocol) is a standard for connecting AI agents to external tools.",
    "langgraph":  "LangGraph is a library for building stateful agent workflows using nodes and edges.",
    "transformer":"Transformers use self-attention mechanism. Form the backbone of modern LLMs.",
}

def research_agent_node(state: dict) -> dict:
    """
    Look up information about the task in KNOWLEDGE_BASE.

    Steps:
      1. task = state["task"].lower()
      2. Search KNOWLEDGE_BASE: for each key, check if key in task
      3. If found: result = KNOWLEDGE_BASE[key]
         If not found: result = f"No information found about: {state['task']}"
      4. Append "research" to done_agents
      5. Return updated research_result and done_agents

    Parameters:
      state: dict with "task" (str) and "done_agents" (list)

    Returns:
      dict with "research_result" (str) and "done_agents" (list)

    EXAMPLE:
      state = {"task": "explain Python", "done_agents": []}
      output = {"research_result": "Python is a high-level...", "done_agents": ["research"]}
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Math Specialist Node
# =============================================================================

def math_agent_node(state: dict) -> dict:
    """
    Evaluate a math expression found in the task.

    Steps:
      1. Extract numbers from state["task"] using:
           re.findall(r'-?\d+\.?\d*', state["task"])
      2. Detect operation from task (case-insensitive):
           "add" or "sum" or "plus" or "+"         -> a + b
           "subtract" or "minus" or "-"             -> a - b
           "multiply" or "times" or "*"             -> a * b
           "divide" or "divided by" or "/"          -> a / b  (handle div by zero)
           "square" or "squared"                    -> n * n  (uses first number only)
      3. If fewer than 2 numbers found for binary ops: result = "Need two numbers"
      4. For "square": use just the first number
      5. result = str(computed_value)  (e.g., "15.0" or "49.0")
      6. If no operation matched: result = f"Cannot parse math from: {task}"
      7. Append "math" to done_agents

    Parameters:
      state: dict with "task" (str) and "done_agents" (list)

    Returns:
      dict with "math_result" (str) and "done_agents" (list)

    EXAMPLES:
      "add 10 and 5"        -> math_result = "15.0"
      "subtract 3 from 9"   -> math_result = "6.0"
      "multiply 4 and 7"    -> math_result = "28.0"
      "divide 10 by 2"      -> math_result = "5.0"
      "square of 9"         -> math_result = "81.0"
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Build and Run the Multi-Agent Graph
# =============================================================================

def run_multi_agent_graph(task: str) -> dict:
    """
    Build and execute a multi-agent graph for the given task.

    Graph structure:
      router -> math_agent    -> router -> ... -> finalize -> END
             -> research_agent -> router -> ...

    Steps:
      1. Define initial state:
           {
             "task":             task,
             "next_agent":       "",
             "research_result":  "",
             "math_result":      "",
             "done_agents":      [],
             "router_calls":     0,
             "final_answer":     ""
           }

      2. Define finalize_node(state):
           Combines research_result and math_result into final_answer.
           If research_result: include it.
           If math_result: include it.
           Return {"final_answer": combined_string}

      3. Define route_from_router(state) -> str:
           return state.get("next_agent", "FINISH")

      4. Build the graph using the run_graph_with_conditions helper:
           nodes:
             "router":          router_node
             "math_agent":      math_agent_node
             "research_agent":  research_agent_node
             "finalize":        finalize_node

           normal_edges:
             "math_agent":      "router"
             "research_agent":  "router"
             "finalize":        END

           conditional_edges:
             "router": (route_from_router, {
                            "math_agent":      "math_agent",
                            "research_agent":  "research_agent",
                            "FINISH":          "finalize"
                        })

           entry_point: "router"

      5. Run the graph and return final state.

    HINT: You can reuse run_graph_with_conditions from exercise_07,
          or implement a simple version inline.
    """
    # YOUR CODE HERE

    def finalize_node(state: dict) -> dict:
        """Combine specialist results into final_answer."""
        parts = []
        if state.get("research_result"):
            parts.append(f"Research: {state['research_result']}")
        if state.get("math_result"):
            parts.append(f"Math result: {state['math_result']}")
        return {"final_answer": "\n".join(parts) if parts else "No results."}

    def route_from_router(state: dict) -> str:
        return state.get("next_agent", "FINISH")

    # YOUR CODE HERE: build and run the graph
    pass


# =============================================================================
# HELPER: Simple graph runner (mini version of exercise_07's run_graph_with_conditions)
# =============================================================================

def _run_graph(nodes, normal_edges, conditional_edges, entry, initial_state, max_steps=20):
    """Simple graph runner. You may use this inside run_multi_agent_graph."""
    state   = {**initial_state}
    current = entry
    steps   = 0

    while current != END and current is not None and steps < max_steps:
        fn = nodes.get(current)
        if fn is None:
            break
        update = fn(state)
        state.update(update)
        steps += 1

        if current in conditional_edges:
            router_fn, mapping = conditional_edges[current]
            decision = router_fn(state)
            current  = mapping.get(decision, END)
        elif current in normal_edges:
            current = normal_edges[current]
        else:
            current = END

    return state


# =============================================================================
# TESTS -- DO NOT MODIFY BELOW THIS LINE
# =============================================================================

def run_tests():
    print("=" * 55)
    print("EXERCISE 08 TESTS")
    print("=" * 55)
    passed = 0
    total  = 0

    def check(name, got, expected):
        nonlocal passed, total
        total += 1
        if isinstance(expected, bool):
            ok = (got == expected)
        elif isinstance(expected, int):
            ok = (got == expected)
        elif isinstance(expected, str):
            ok = expected in str(got)
        elif isinstance(expected, list):
            ok = (sorted(str(x) for x in got) == sorted(str(x) for x in expected))
        else:
            ok = abs(float(got) - float(expected)) < 0.01
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {name}")
        if not ok:
            print(f"         Expected: {expected!r}")
            print(f"         Got:      {got!r}")

    # --- router_node ---
    r1 = router_node({"task": "calculate 5 + 3", "done_agents": [], "router_calls": 0})
    check("router: math task -> math_agent",     r1["next_agent"], "math_agent")
    check("router: increments router_calls",      r1["router_calls"], 1)

    r2 = router_node({"task": "explain Python",  "done_agents": [], "router_calls": 0})
    check("router: research task -> research",    r2["next_agent"], "research_agent")

    r3 = router_node({"task": "",                "done_agents": [], "router_calls": 0})
    check("router: empty task -> FINISH",         r3["next_agent"], "FINISH")

    r4 = router_node({"task": "calculate 5+3",   "done_agents": ["math"], "router_calls": 2})
    check("router: math done -> FINISH",          r4["next_agent"], "FINISH")

    r5 = router_node({"task": "explain LLM",     "done_agents": ["research"], "router_calls": 1})
    check("router: research done -> FINISH",      r5["next_agent"], "FINISH")

    # --- research_agent_node ---
    res1 = research_agent_node({"task": "explain Python", "done_agents": []})
    check("research: Python found",          "Python" in res1["research_result"], True)
    check("research: done_agents updated",   "research" in res1["done_agents"], True)

    res2 = research_agent_node({"task": "explain basketball", "done_agents": []})
    check("research: not found message",     "No information" in res2["research_result"], True)

    res3 = research_agent_node({"task": "describe MCP", "done_agents": []})
    check("research: MCP found",             "MCP" in res3["research_result"] or "Model Context" in res3["research_result"], True)

    # --- math_agent_node ---
    m1 = math_agent_node({"task": "add 10 and 5", "done_agents": []})
    check("math: add 10+5=15",               "15" in m1["math_result"], True)
    check("math: done_agents updated",       "math" in m1["done_agents"], True)

    m2 = math_agent_node({"task": "subtract 3 from 9", "done_agents": []})
    check("math: subtract 9-3=6",            "6" in m2["math_result"], True)

    m3 = math_agent_node({"task": "multiply 4 and 7", "done_agents": []})
    check("math: multiply 4*7=28",           "28" in m3["math_result"], True)

    m4 = math_agent_node({"task": "square of 8", "done_agents": []})
    check("math: square of 8=64",            "64" in m4["math_result"], True)

    m5 = math_agent_node({"task": "divide 10 by 2", "done_agents": []})
    check("math: divide 10/2=5",             "5" in m5["math_result"], True)

    # --- run_multi_agent_graph ---
    gr1 = run_multi_agent_graph("explain FastAPI")
    if gr1 is not None:
        check("graph: research task has final_answer",  bool(gr1.get("final_answer")), True)
        check("graph: research result populated",       bool(gr1.get("research_result")), True)
        check("graph: research in done_agents",         "research" in gr1.get("done_agents", []), True)

    gr2 = run_multi_agent_graph("calculate 6 + 4")
    if gr2 is not None:
        check("graph: math task has final_answer",  bool(gr2.get("final_answer")), True)
        check("graph: math result populated",       bool(gr2.get("math_result")), True)
        check("graph: math in done_agents",         "math" in gr2.get("done_agents", []), True)
        check("graph: math result is 10",           "10" in str(gr2.get("math_result", "")), True)

    print(f"\n{passed}/{total} tests passed")
    if passed == total:
        print("All tests passed!")
    else:
        print("Some tests failed. Re-read the docstrings and try again.")


if __name__ == "__main__":
    run_tests()

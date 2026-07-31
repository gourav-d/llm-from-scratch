"""
=============================================================================
MODULE 11 - EXERCISE 07: LangGraph State Machine
=============================================================================

YOUR TASK:
  Build a LangGraph-style state machine simulation from scratch.
  You'll implement nodes, edges, conditional routing, and execution.

RULES:
  - Only Python stdlib
  - Read each docstring carefully
  - Tests at the bottom verify your work

RUN WITH:  python exercise_07_langgraph.py
=============================================================================
"""


# =============================================================================
# CONSTANTS
# =============================================================================

END = "__END__"    # Route to this to stop the graph


# =============================================================================
# EXERCISE 1: Node Functions
# =============================================================================

def normalize_text_node(state: dict) -> dict:
    """
    A graph node that normalizes the "text" field in state.

    Rules:
      - Convert state["text"] to lowercase
      - Strip leading/trailing whitespace
      - Store result back in state["text"]
      - Append "normalize" to state["steps"] list

    Parameters:
      state: dict with at least "text" (str) and "steps" (list) keys

    Returns:
      dict with updated "text" and "steps" keys ONLY
      (do NOT return the full state -- only what changed)

    EXAMPLE:
      input:  {"text": "  Hello World  ", "steps": []}
      output: {"text": "hello world", "steps": ["normalize"]}
    """
    # YOUR CODE HERE
    pass


def count_words_node(state: dict) -> dict:
    """
    A graph node that counts words in state["text"].

    Rules:
      - Split state["text"] by whitespace: text.split()
      - Store the count in state["word_count"]
      - Append "count_words" to state["steps"]

    Returns:
      dict with "word_count" (int) and "steps" (list) keys ONLY

    EXAMPLE:
      input:  {"text": "hello world", "word_count": 0, "steps": ["normalize"]}
      output: {"word_count": 2, "steps": ["normalize", "count_words"]}
    """
    # YOUR CODE HERE
    pass


def label_node(state: dict) -> dict:
    """
    A graph node that assigns a label based on word count.

    Rules:
      - word_count == 0:  label = "empty"
      - word_count <= 3:  label = "short"
      - word_count <= 10: label = "medium"
      - word_count > 10:  label = "long"
      - Append "label" to state["steps"]

    Returns:
      dict with "label" (str) and "steps" (list) ONLY

    EXAMPLE:
      input:  {"word_count": 2, "steps": [...]}
      output: {"label": "short", "steps": [..., "label"]}
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 2: Graph Runner
# =============================================================================

def run_graph(nodes: dict, edges: dict, entry: str, initial_state: dict) -> dict:
    """
    Execute a simple graph (nodes + normal edges only).

    Parameters:
      nodes:         dict mapping node_name -> function
      edges:         dict mapping node_name -> next_node_name (or END)
      entry:         name of the starting node
      initial_state: starting state dict

    Algorithm:
      1. state = initial_state.copy()
      2. current = entry
      3. While current != END and current is not None:
           a. Get the node function: fn = nodes[current]
           b. Call it: update = fn(state)
           c. Merge: state.update(update)
           d. Get next node: current = edges.get(current, END)
      4. Return final state

    SAFETY: if a node name is not in the nodes dict, set current = END

    EXAMPLE:
      nodes = {"a": fn_a, "b": fn_b}
      edges = {"a": "b", "b": END}
      entry = "a"
      result = run_graph(nodes, edges, entry, {"text": "Hello World", ...})
      # fn_a runs, then fn_b runs, then END
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Conditional Routing
# =============================================================================

def run_graph_with_conditions(
    nodes: dict,
    normal_edges: dict,
    conditional_edges: dict,
    entry: str,
    initial_state: dict,
    max_steps: int = 20
) -> dict:
    """
    Execute a graph with BOTH normal edges and conditional edges.

    Parameters:
      nodes:             dict mapping node_name -> function
      normal_edges:      dict mapping node_name -> next_node_name
      conditional_edges: dict mapping node_name -> (router_fn, mapping_dict)
                         router_fn(state) -> string key
                         mapping_dict maps that string to a node_name or END
      entry:             starting node name
      initial_state:     starting state dict
      max_steps:         safety limit (default 20)

    Algorithm:
      1. state = initial_state.copy(), current = entry, steps = 0
      2. While current != END and steps < max_steps:
           a. fn = nodes.get(current) -- if None, break
           b. update = fn(state) -- run the node
           c. state.update(update)
           d. steps += 1
           e. If current in conditional_edges:
                router_fn, mapping = conditional_edges[current]
                decision = router_fn(state)
                current  = mapping.get(decision, END)
              Elif current in normal_edges:
                current = normal_edges[current]
              Else:
                current = END
      3. Return final state

    EXAMPLE with retry loop:
      def review_router(state):
          return "pass" if state["score"] >= 70 else "retry"

      conditional_edges = {
          "review": (review_router, {"pass": END, "retry": "process"})
      }
      # After "review" node: if score >= 70 -> END, else -> "process" (retry)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Build a Complete Pipeline
# =============================================================================

def build_text_analysis_pipeline() -> dict:
    """
    Build and return a graph configuration for a 3-node text analysis pipeline:

    Nodes:
      "normalize"  -> normalize_text_node
      "count"      -> count_words_node
      "label"      -> label_node

    Normal edges:
      "normalize" -> "count"
      "count"     -> "label"
      "label"     -> END

    Entry point: "normalize"

    Return a dict with keys:
      "nodes":  the nodes dict
      "edges":  the normal_edges dict
      "entry":  the entry point string

    USAGE (will be called in tests):
      config = build_text_analysis_pipeline()
      result = run_graph(config["nodes"], config["edges"], config["entry"], state)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# TESTS -- DO NOT MODIFY BELOW THIS LINE
# =============================================================================

def run_tests():
    print("=" * 55)
    print("EXERCISE 07 TESTS")
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
            ok = (got == expected)
        elif isinstance(expected, list):
            ok = (got == expected)
        else:
            ok = abs(float(got) - float(expected)) < 0.01
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {name}")
        if not ok:
            print(f"         Expected: {expected!r}")
            print(f"         Got:      {got!r}")

    # --- normalize_text_node ---
    s1 = normalize_text_node({"text": "  Hello World  ", "steps": []})
    check("normalize: lowercase + strip",    s1["text"], "hello world")
    check("normalize: steps appended",       s1["steps"], ["normalize"])
    check("normalize: only text+steps keys", set(s1.keys()) <= {"text", "steps"}, True)

    s2 = normalize_text_node({"text": "  PYTHON  ", "steps": ["a"]})
    check("normalize: existing steps kept",  s2["steps"], ["a", "normalize"])

    # --- count_words_node ---
    s3 = count_words_node({"text": "hello world foo", "word_count": 0, "steps": []})
    check("count: 3 words",                  s3["word_count"], 3)
    check("count: steps appended",           s3["steps"], ["count_words"])

    s4 = count_words_node({"text": "", "word_count": 0, "steps": []})
    check("count: empty text = 0 words",     s4["word_count"], 0)

    # --- label_node ---
    check("label: 0 words -> empty",    label_node({"word_count": 0,  "steps": []})["label"], "empty")
    check("label: 2 words -> short",    label_node({"word_count": 2,  "steps": []})["label"], "short")
    check("label: 5 words -> medium",   label_node({"word_count": 5,  "steps": []})["label"], "medium")
    check("label: 15 words -> long",    label_node({"word_count": 15, "steps": []})["label"], "long")

    # --- run_graph ---
    initial = {"text": "  HELLO WORLD  ", "word_count": 0, "label": "", "steps": []}
    config = build_text_analysis_pipeline()
    if config is not None:
        result = run_graph(config["nodes"], config["edges"], config["entry"], initial)
        check("pipeline: text normalized",    result["text"], "hello world")
        check("pipeline: word count = 2",     result["word_count"], 2)
        check("pipeline: label = short",      result["label"], "short")
        check("pipeline: 3 steps",            len(result["steps"]), 3)
        check("pipeline: steps order",        result["steps"], ["normalize", "count_words", "label"])
    else:
        for _ in range(5):
            check("build_text_analysis_pipeline: returned None", False, True)

    # --- run_graph_with_conditions: retry loop ---
    attempt_count = [0]    # Track how many times process_node runs

    def process_node(state):
        attempt_count[0] += 1
        # Score improves each attempt: 50 on first, 75 on second
        score = 50 + (25 * (attempt_count[0] - 1))
        return {"score": min(score, 100), "steps": state["steps"] + [f"process_{attempt_count[0]}"]}

    def review_node(state):
        passed_review = state["score"] >= 70
        return {"approved": passed_review, "steps": state["steps"] + ["review"]}

    def review_router(state):
        return "pass" if state["approved"] else "retry"

    retry_initial = {"score": 0, "approved": False, "steps": []}
    retry_nodes   = {"process": process_node, "review": review_node}
    retry_normal  = {"process": "review"}
    retry_cond    = {"review": (review_router, {"pass": END, "retry": "process"})}

    retry_result = run_graph_with_conditions(
        retry_nodes, retry_normal, retry_cond, "process", retry_initial
    )

    check("retry: final score >= 70",     retry_result["score"] >= 70, True)
    check("retry: approved = True",       retry_result["approved"], True)
    check("retry: 2 attempts to pass",    attempt_count[0], 2)

    # --- run_graph_with_conditions: max_steps safety ---
    attempt_count2 = [0]
    def infinite_node(state):
        attempt_count2[0] += 1
        return {"steps": state["steps"] + ["loop"]}
    def always_retry(state):
        return "retry"

    loop_result = run_graph_with_conditions(
        {"process": infinite_node},
        {},
        {"process": (always_retry, {"retry": "process"})},
        "process",
        {"steps": []},
        max_steps=5
    )
    check("max_steps: stops at 5",    attempt_count2[0], 5)

    print(f"\n{passed}/{total} tests passed")
    if passed == total:
        print("All tests passed!")
    else:
        print("Some tests failed. Re-read the docstrings and try again.")


if __name__ == "__main__":
    run_tests()

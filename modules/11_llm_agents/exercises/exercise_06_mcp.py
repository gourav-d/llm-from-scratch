"""
=============================================================================
MODULE 11 - EXERCISE 06: MCP Protocol
=============================================================================

YOUR TASK:
  Build a simple MCP server simulation from scratch.
  You'll implement the server, register tools, and call them.

RULES:
  - Only Python stdlib (no external libraries)
  - Read each docstring carefully
  - Tests at the bottom verify your work

RUN WITH:  python exercise_06_mcp.py
=============================================================================
"""

import json


# =============================================================================
# EXERCISE 1: Build an MCP Server
# =============================================================================

class MCPServer:
    """
    An MCP server that holds a registry of tools and handles two request types:
      - "tools/list" : return all tool schemas
      - "tools/call" : execute a tool by name

    The registry should map tool name -> {"schema": {...}, "handler": fn}.

    WHAT TO IMPLEMENT:
      __init__:       set self.tools_registry = {}
      register_tool:  store the schema and handler in the registry
      handle_request: route to _list_tools or _call_tool based on method

    C# analogy: like a minimal gRPC service that exposes named operations.
    """

    def __init__(self):
        """Initialize with an empty tools registry."""
        # YOUR CODE HERE
        pass

    def register_tool(self, schema: dict, handler):
        """
        Register a tool.

        schema:  dict with at least a "name" key
        handler: callable(arguments: dict) -> str result

        Store as: self.tools_registry[name] = {"schema": schema, "handler": handler}
        """
        # YOUR CODE HERE
        pass

    def handle_request(self, method: str, params: dict) -> dict:
        """
        Handle an MCP protocol request.

        Rules:
          - "tools/list":  return {"result": {"tools": [list of schemas]}}
          - "tools/call":  call the handler and return
                           {"result": {"content": [{"type": "text", "text": str(result)}]}}
          - Unknown tool:  return {"error": "Tool 'name' not found"}
          - Unknown method: return {"error": "Unknown method: method_name"}

        HINTS:
          - For "tools/list": iterate self.tools_registry.values() and collect ["schema"]
          - For "tools/call": params["name"] = tool name, params["arguments"] = args dict
          - Wrap handler call in try/except; on exception return {"error": str(e)}
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# EXERCISE 2: Register Tools on the Server
# =============================================================================

def build_math_server() -> MCPServer:
    """
    Create and return an MCPServer with three tools registered:

    Tool 1: "add"
      description: "Add two numbers and return the sum."
      parameters:  a (number, required), b (number, required)
      handler:     returns a + b

    Tool 2: "subtract"
      description: "Subtract b from a and return the result."
      parameters:  a (number, required), b (number, required)
      handler:     returns a - b

    Tool 3: "square"
      description: "Return the square of a number."
      parameters:  n (number, required)
      handler:     returns n * n

    Each schema must have keys: "name", "description", "inputSchema"
    inputSchema must have "type": "object" and "properties" and "required" keys.

    EXAMPLE schema for "add":
      {
        "name": "add",
        "description": "Add two numbers and return the sum.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"}
          },
          "required": ["a", "b"]
        }
      }

    HINTS:
      - Create an MCPServer instance
      - Call server.register_tool(schema, handler) three times
      - Return the server
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: MCP Client
# =============================================================================

class MCPClient:
    """
    A simple MCP client that connects to one or more servers,
    discovers all tools, and can call them by name.

    WHAT TO IMPLEMENT:
      __init__:   set self.servers = {} and self.all_tools = []
      connect:    call server's tools/list, store each tool in all_tools
      call_tool:  find which server has the tool, call it, return the text result
      tool_names: return a list of all tool names (just the names, not full schemas)
    """

    def __init__(self):
        """Initialize with empty servers dict and empty all_tools list."""
        # YOUR CODE HERE
        pass

    def connect(self, server: MCPServer):
        """
        Connect to a server and discover its tools.

        Steps:
          1. Store the server: self.servers[server.server_name] = server
          2. Call server.handle_request("tools/list", {})
          3. For each tool in response["result"]["tools"]:
               self.all_tools.append({"server": server.server_name, "schema": tool})
        """
        # YOUR CODE HERE
        pass

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Call a tool by name.

        Steps:
          1. Find the tool in self.all_tools (match on entry["schema"]["name"])
          2. Get the server: self.servers[entry["server"]]
          3. Call server.handle_request("tools/call", {"name": tool_name, "arguments": arguments})
          4. If "error" in response: return "ERROR: " + response["error"]
          5. Else: return response["result"]["content"][0]["text"]

        If no tool with that name exists: return "ERROR: Tool 'name' not found"
        """
        # YOUR CODE HERE
        pass

    def tool_names(self) -> list:
        """
        Return a list of all available tool names (strings).

        HINTS:
          - Iterate self.all_tools
          - Each entry has entry["schema"]["name"]
        """
        # YOUR CODE HERE
        pass


# =============================================================================
# EXERCISE 4: Use the Client to Solve Tasks
# =============================================================================

def solve_with_mcp(client: MCPClient, task: str) -> str:
    """
    Use the MCP client to solve a math task.

    Parse the task string and call the right tool:
      - "add X and Y"        -> call_tool("add", {"a": X, "b": Y})
      - "subtract Y from X"  -> call_tool("subtract", {"a": X, "b": Y})
      - "square of X"        -> call_tool("square", {"n": X})

    Return the tool's result as a string.
    If no matching pattern: return "Unknown task: <task>"

    PARSING:
      - Use re.findall(r'-?\d+\.?\d*', task) to extract numbers as strings
      - Convert to float: float(nums[0])
      - "add" / "sum" / "plus" in task.lower()      -> add
      - "subtract" / "minus" in task.lower()         -> subtract
      - "square" / "squared" in task.lower()         -> square

    EXAMPLE:
      solve_with_mcp(client, "add 10 and 5")     -> "15.0" (from tool result)
      solve_with_mcp(client, "square of 7")       -> "49.0"
      solve_with_mcp(client, "subtract 3 from 9") -> "6.0"
    """
    import re
    # YOUR CODE HERE
    pass


# =============================================================================
# TESTS -- DO NOT MODIFY BELOW THIS LINE
# =============================================================================

def run_tests():
    print("=" * 55)
    print("EXERCISE 06 TESTS")
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
            # For strings: check if expected is a substring (flexible matching)
            ok = (expected in str(got)) if expected else (str(got) == "")
        elif isinstance(expected, list):
            ok = sorted(got) == sorted(expected)
        else:
            ok = abs(float(got) - float(expected)) < 0.01
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {name}")
        if not ok:
            print(f"         Expected: {expected!r}")
            print(f"         Got:      {got!r}")

    # --- MCPServer: __init__ ---
    server = MCPServer()
    check("server: init has tools_registry", hasattr(server, "tools_registry"), True)
    check("server: registry starts empty",   len(server.tools_registry), 0)

    # --- MCPServer: register_tool ---
    server.server_name = "test-server"
    server.register_tool(
        {"name": "ping", "description": "Returns pong.", "inputSchema": {"type": "object", "properties": {}}},
        lambda args: "pong"
    )
    check("server: tool registered",         "ping" in server.tools_registry, True)
    check("server: schema stored",           "schema" in server.tools_registry["ping"], True)
    check("server: handler stored",          "handler" in server.tools_registry["ping"], True)

    # --- MCPServer: handle_request tools/list ---
    resp = server.handle_request("tools/list", {})
    check("server: tools/list returns result", "result" in resp, True)
    check("server: tools list has 1 tool",     len(resp["result"]["tools"]), 1)
    check("server: first tool name is 'ping'", resp["result"]["tools"][0]["name"], "ping")

    # --- MCPServer: handle_request tools/call ---
    call_resp = server.handle_request("tools/call", {"name": "ping", "arguments": {}})
    check("server: tools/call success",       "result" in call_resp, True)
    check("server: tools/call content text",  call_resp["result"]["content"][0]["text"], "pong")

    # --- MCPServer: unknown tool ---
    err_resp = server.handle_request("tools/call", {"name": "unknown", "arguments": {}})
    check("server: unknown tool returns error", "error" in err_resp, True)

    # --- MCPServer: unknown method ---
    unk_resp = server.handle_request("tools/perform", {})
    check("server: unknown method returns error", "error" in unk_resp, True)

    # --- build_math_server ---
    math_server = build_math_server()
    check("math server: has 3 tools",    len(math_server.tools_registry), 3)
    check("math server: has 'add'",      "add"      in math_server.tools_registry, True)
    check("math server: has 'subtract'", "subtract" in math_server.tools_registry, True)
    check("math server: has 'square'",   "square"   in math_server.tools_registry, True)

    add_resp = math_server.handle_request("tools/call", {"name": "add", "arguments": {"a": 10, "b": 5}})
    check("math server: add(10,5)=15", add_resp["result"]["content"][0]["text"], "15")

    sub_resp = math_server.handle_request("tools/call", {"name": "subtract", "arguments": {"a": 10, "b": 3}})
    check("math server: subtract(10,3)=7", sub_resp["result"]["content"][0]["text"], "7")

    sq_resp = math_server.handle_request("tools/call", {"name": "square", "arguments": {"n": 6}})
    check("math server: square(6)=36", sq_resp["result"]["content"][0]["text"], "36")

    # --- MCPClient ---
    math_server.server_name = "math-server"
    client = MCPClient()
    client.connect(math_server)

    check("client: 3 tools discovered",  len(client.tool_names()), 3)
    check("client: 'add' in tool names", "add" in client.tool_names(), True)

    add_result = client.call_tool("add", {"a": 20, "b": 22})
    check("client: call_tool add(20,22)=42", "42" in str(add_result), True)

    err_result = client.call_tool("nonexistent", {})
    check("client: unknown tool returns ERROR", "ERROR" in err_result, True)

    # --- solve_with_mcp ---
    math_server.server_name = "math-server"
    client2 = MCPClient()
    client2.connect(math_server)

    r1 = solve_with_mcp(client2, "add 10 and 5")
    check("solve: add 10 and 5 = 15",     "15" in str(r1), True)

    r2 = solve_with_mcp(client2, "square of 9")
    check("solve: square of 9 = 81",      "81" in str(r2), True)

    r3 = solve_with_mcp(client2, "subtract 4 from 10")
    check("solve: subtract 4 from 10 = 6", "6" in str(r3), True)

    print(f"\n{passed}/{total} tests passed")
    if passed == total:
        print("All tests passed!")
    else:
        print("Some tests failed. Re-read the docstrings and try again.")


if __name__ == "__main__":
    run_tests()

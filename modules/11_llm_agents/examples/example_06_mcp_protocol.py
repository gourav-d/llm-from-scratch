"""
Example 06: MCP -- Model Context Protocol
==========================================

GLOSSARY
--------
MCP (Model Context Protocol):
  An open standard for how AI agents discover and call external tools.
  Like USB-C for AI: write a tool once as an MCP server, any agent uses it.

MCP Server:
  A program that exposes tools over the MCP protocol.
  Responds to: tools/list (what tools exist?) and tools/call (run this tool).

MCP Client:
  The agent that connects to MCP servers and calls their tools.
  We simulate one here with a plain Python class.

Tool Schema:
  A JSON dict describing a tool: name, description, parameters.
  The agent reads schemas to discover what tools are available.

Tool Discovery:
  Client asks server "what tools do you have?" Server responds with schemas.
  No hardcoding: agent learns about tools at runtime.

WHAT THIS EXAMPLE SHOWS
------------------------
Part A: Simulate an MCP server with tool schemas and tool execution
Part B: Simulate an MCP client that discovers tools and calls them
Part C: Simulate multiple MCP servers (filesystem + github + calculator)
Part D: Show MCP vs function-calling comparison

LIBRARIES NEEDED
----------------
  None (pure Python stdlib)
"""

import json      # For pretty-printing JSON structures
import re        # For simple text parsing

print("=" * 65)
print("EXAMPLE 06: MCP -- Model Context Protocol")
print("=" * 65)


# ==============================================================================
# PART A: Simulated MCP Server
# ==============================================================================

print("\n" + "=" * 65)
print("PART A: Simulated MCP Server")
print("=" * 65)

print("""
An MCP server responds to two types of requests:
  1. tools/list  -> returns a list of tool schemas
  2. tools/call  -> executes one tool and returns the result

We simulate this with a Python class instead of a real network server.
The structure of requests and responses is IDENTICAL to the real MCP protocol.
""")


class SimulatedMCPServer:
    """
    Simulates an MCP server.
    In production: this would be a real subprocess or HTTP server.
    Here: a Python class with the same interface.

    C# analogy: like a mock IService or a stub in unit tests.
                Also like a WCF service or gRPC server that exposes named operations.
    """

    def __init__(self, server_name: str):
        # server_name: identifies this server (e.g., "calculator", "github")
        self.server_name = server_name

        # tools_registry: maps tool name -> (schema dict, handler function)
        # Each entry: {"name": ..., "description": ..., "inputSchema": ...}
        self.tools_registry = {}

    def register_tool(self, schema: dict, handler):
        """
        Register a tool with its schema and handler function.

        schema:  dict with keys: name, description, inputSchema
        handler: function(arguments: dict) -> str result

        C# analogy: like registering a command handler in MediatR:
          services.AddTransient<IRequestHandler<MyCommand, string>, MyHandler>()
        """
        name = schema["name"]           # Extract the tool name from schema
        self.tools_registry[name] = {   # Store schema + handler together
            "schema":  schema,
            "handler": handler
        }

    def handle_request(self, method: str, params: dict) -> dict:
        """
        Handle an MCP protocol request.

        method: "tools/list" or "tools/call"
        params: depends on the method
        Returns: dict with "result" key (success) or "error" key (failure)

        C# analogy: like a controller action that routes to the right service method.
        """
        if method == "tools/list":
            # Return all tool schemas (no params needed)
            schemas = [entry["schema"] for entry in self.tools_registry.values()]
            return {"result": {"tools": schemas}}   # MCP protocol response format

        elif method == "tools/call":
            # Execute a specific tool
            tool_name  = params.get("name")         # Which tool to call
            arguments  = params.get("arguments", {}) # Arguments for the tool

            if tool_name not in self.tools_registry:
                # Tool not found -- return MCP error format
                return {"error": f"Tool '{tool_name}' not found on server '{self.server_name}'"}

            handler = self.tools_registry[tool_name]["handler"]  # Get the function

            try:
                result_text = handler(arguments)    # Call the actual tool function
                # MCP protocol: content is a list of objects with "type" and "text"
                return {
                    "result": {
                        "content": [{"type": "text", "text": str(result_text)}]
                    }
                }
            except Exception as e:
                return {"error": f"Tool execution failed: {str(e)}"}

        else:
            return {"error": f"Unknown method: {method}"}


# ---- Build a Calculator MCP Server ----

calc_server = SimulatedMCPServer("calculator-server")

# Register the "add" tool
calc_server.register_tool(
    schema={
        "name":        "add",
        "description": "Add two numbers together. Returns the sum.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]  # Both parameters are required
        }
    },
    handler=lambda args: args["a"] + args["b"]   # Simple addition
)

# Register the "multiply" tool
calc_server.register_tool(
    schema={
        "name":        "multiply",
        "description": "Multiply two numbers. Returns the product.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    },
    handler=lambda args: args["a"] * args["b"]   # Simple multiplication
)

# Register the "power" tool
def power_handler(args):
    """Raises base to the power of exponent."""
    return args["base"] ** args["exponent"]

calc_server.register_tool(
    schema={
        "name":        "power",
        "description": "Raise a number to a power. Returns base^exponent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "base":     {"type": "number", "description": "The base number"},
                "exponent": {"type": "number", "description": "The exponent"}
            },
            "required": ["base", "exponent"]
        }
    },
    handler=power_handler
)

# Test: tools/list (tool discovery)
print("Client -> Server: tools/list")
discovery_response = calc_server.handle_request("tools/list", {})
tools = discovery_response["result"]["tools"]  # Extract the tools list
print(f"Server -> Client: {len(tools)} tools available:")
for tool in tools:
    # Print each tool's name and description
    params = list(tool["inputSchema"]["properties"].keys())
    print(f"  Tool: {tool['name']}({', '.join(params)}) -- {tool['description']}")

# Test: tools/call (tool execution)
print("\nClient -> Server: tools/call 'add' with a=15, b=27")
call_response = calc_server.handle_request("tools/call", {
    "name":      "add",
    "arguments": {"a": 15, "b": 27}
})
result_text = call_response["result"]["content"][0]["text"]  # Extract text from MCP response
print(f"Server -> Client: {result_text}")

print("\nClient -> Server: tools/call 'power' with base=2, exponent=10")
call_response2 = calc_server.handle_request("tools/call", {
    "name":      "power",
    "arguments": {"base": 2, "exponent": 10}
})
print(f"Server -> Client: {call_response2['result']['content'][0]['text']}")

# Test: error case
print("\nClient -> Server: tools/call 'nonexistent_tool'")
error_response = calc_server.handle_request("tools/call", {"name": "nonexistent_tool", "arguments": {}})
print(f"Server -> Client: {error_response}")


# ==============================================================================
# PART B: Simulated MCP Client (Agent)
# ==============================================================================

print("\n" + "=" * 65)
print("PART B: Simulated MCP Client (Agent)")
print("=" * 65)

print("""
An MCP client:
  1. Connects to one or more MCP servers
  2. Discovers all available tools (tools/list)
  3. Chooses the right tool for a task (simulated LLM decision)
  4. Calls the tool (tools/call)
  5. Uses the result to generate a response

C# analogy: like an HttpClient that auto-discovers API endpoints
            from a Swagger/OpenAPI spec and calls them dynamically.
""")


class SimulatedMCPClient:
    """
    Simulates an MCP client (the agent's tool-calling layer).

    In production: this connects to real MCP servers via stdio or HTTP.
    Here: we pass SimulatedMCPServer objects directly.

    C# analogy: like an HttpClient factory that connects to multiple microservices
                and discovers their capabilities from their OpenAPI specs.
    """

    def __init__(self):
        # servers: maps server_name -> SimulatedMCPServer instance
        self.servers = {}
        # all_tools: flat list of (server_name, tool_schema) for all connected servers
        self.all_tools = []

    def connect(self, server: SimulatedMCPServer):
        """
        Connect to an MCP server and discover its tools.
        This is the "Initialize + tools/list" step.
        """
        print(f"  [Client] Connecting to: {server.server_name}")

        # Store the server connection
        self.servers[server.server_name] = server

        # Discover all tools from this server
        response   = server.handle_request("tools/list", {})   # Ask: what tools?
        tools      = response["result"]["tools"]                 # Get the list

        # Register each tool, tagged with which server it came from
        for tool in tools:
            self.all_tools.append({
                "server": server.server_name,    # Which server hosts this tool
                "schema": tool                   # The tool's full schema
            })
            print(f"    Discovered tool: {server.server_name}/{tool['name']}")

    def list_all_tools(self):
        """Show all tools from all connected servers."""
        print(f"\n  [Client] All available tools ({len(self.all_tools)} total):")
        for entry in self.all_tools:
            tool = entry["schema"]
            print(f"    {entry['server']}/{tool['name']}: {tool['description'][:60]}")

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Call a tool by name. The client finds which server has it and calls it.

        C# analogy: like a service locator pattern:
          var handler = _serviceLocator.Resolve<IToolHandler>(toolName);
          return await handler.ExecuteAsync(arguments);
        """
        # Find the server that has this tool
        for entry in self.all_tools:
            if entry["schema"]["name"] == tool_name:
                server_name = entry["server"]               # Found the server
                server      = self.servers[server_name]     # Get the server object

                # Call the tool via MCP protocol
                response = server.handle_request("tools/call", {
                    "name":      tool_name,
                    "arguments": arguments
                })

                if "error" in response:
                    return f"ERROR: {response['error']}"

                # Extract the text result from MCP content format
                return response["result"]["content"][0]["text"]

        return f"ERROR: No server has a tool named '{tool_name}'"

    def simulate_agent_task(self, task: str) -> str:
        """
        Simulate an agent solving a task using available tools.
        In production: an LLM reads the task + tool schemas and decides which to call.
        Here: simple keyword matching (same logic an LLM would use).
        """
        print(f"\n  [Agent] Task: {task}")
        task_lower = task.lower()

        # Simulated LLM decision: which tool fits this task?
        if "add" in task_lower or "sum" in task_lower or "plus" in task_lower:
            # Parse numbers from the task (e.g., "add 15 and 27")
            numbers = re.findall(r'-?\d+\.?\d*', task)   # Find all numbers
            if len(numbers) >= 2:
                a, b = float(numbers[0]), float(numbers[1])
                print(f"  [Agent] Chose tool: 'add'  arguments: a={a}, b={b}")
                result = self.call_tool("add", {"a": a, "b": b})
                return f"The sum of {a} and {b} is {result}."

        elif "multiply" in task_lower or "times" in task_lower or "product" in task_lower:
            numbers = re.findall(r'-?\d+\.?\d*', task)
            if len(numbers) >= 2:
                a, b = float(numbers[0]), float(numbers[1])
                print(f"  [Agent] Chose tool: 'multiply'  arguments: a={a}, b={b}")
                result = self.call_tool("multiply", {"a": a, "b": b})
                return f"{a} times {b} equals {result}."

        elif "power" in task_lower or "squared" in task_lower or "cubed" in task_lower:
            numbers = re.findall(r'-?\d+\.?\d*', task)
            if len(numbers) >= 2:
                base, exp = float(numbers[0]), float(numbers[1])
                print(f"  [Agent] Chose tool: 'power'  arguments: base={base}, exponent={exp}")
                result = self.call_tool("power", {"base": base, "exponent": exp})
                return f"{base} to the power of {exp} is {result}."

        return f"No suitable tool found for task: '{task}'"


# Build the client and connect it to our calculator server
client = SimulatedMCPClient()
client.connect(calc_server)      # Connect + discover tools
client.list_all_tools()          # Show what we discovered

# Solve some tasks
print()
tasks = [
    "Add 42 and 58",
    "Multiply 7 and 8",
    "What is 2 to the power of 16?",
]

for task in tasks:
    answer = client.simulate_agent_task(task)
    print(f"  [Agent] Answer: {answer}")


# ==============================================================================
# PART C: Multiple MCP Servers
# ==============================================================================

print("\n" + "=" * 65)
print("PART C: Multiple MCP Servers (Filesystem + GitHub)")
print("=" * 65)

print("""
In real setups, one agent connects to MANY MCP servers.
Each server adds a new set of tools to the agent's repertoire.

C# analogy: like a composite HttpClient that aggregates multiple
            API services, each with their own Swagger spec.
""")

# ---- Simulated Filesystem MCP Server ----
fs_server = SimulatedMCPServer("filesystem-server")

# Simulated in-memory filesystem
_fake_filesystem = {
    "/project/main.py":      "print('Hello World')\n# Main entry point",
    "/project/config.json":  '{"version": "1.0", "debug": true}',
    "/project/README.md":    "# My Project\nThis is a sample project.",
    "/project/utils.py":     "def add(a, b): return a + b"
}

def read_file(args):
    """Read a file from the simulated filesystem."""
    path = args["path"]
    if path in _fake_filesystem:
        return _fake_filesystem[path]
    return f"Error: File not found: {path}"

def list_directory(args):
    """List files in a directory."""
    directory = args["path"]
    files = [f for f in _fake_filesystem.keys() if f.startswith(directory)]
    return "\n".join(files) if files else "Empty directory"

def write_file(args):
    """Write content to a file."""
    path    = args["path"]
    content = args["content"]
    _fake_filesystem[path] = content
    return f"Written {len(content)} bytes to {path}"

# Register filesystem tools
fs_server.register_tool({
    "name": "read_file",
    "description": "Read the contents of a file. Returns file content as text.",
    "inputSchema": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Absolute file path"}},
        "required": ["path"]
    }
}, read_file)

fs_server.register_tool({
    "name": "list_directory",
    "description": "List all files in a directory.",
    "inputSchema": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory path to list"}},
        "required": ["path"]
    }
}, list_directory)

fs_server.register_tool({
    "name": "write_file",
    "description": "Write text content to a file. Creates or overwrites.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path":    {"type": "string", "description": "Absolute file path"},
            "content": {"type": "string", "description": "Text content to write"}
        },
        "required": ["path", "content"]
    }
}, write_file)

# ---- Simulated GitHub MCP Server ----
github_server = SimulatedMCPServer("github-server")

# Simulated GitHub data
_fake_issues = [
    {"id": 1, "title": "Fix login bug",     "status": "open",   "author": "alice"},
    {"id": 2, "title": "Add dark mode",     "status": "open",   "author": "bob"},
    {"id": 3, "title": "Update README",     "status": "closed", "author": "charlie"},
]
_fake_prs = [
    {"id": 101, "title": "Fix typo in docs",   "status": "open",   "author": "alice"},
    {"id": 102, "title": "Add unit tests",     "status": "merged", "author": "bob"},
]

def list_issues(args):
    """List open GitHub issues."""
    status_filter = args.get("status", "open")
    issues = [i for i in _fake_issues if i["status"] == status_filter]
    return json.dumps(issues, indent=2)

def create_issue(args):
    """Create a new GitHub issue."""
    new_id = max(i["id"] for i in _fake_issues) + 1
    issue  = {"id": new_id, "title": args["title"], "status": "open", "author": "agent"}
    _fake_issues.append(issue)
    return f"Issue #{new_id} created: '{args['title']}'"

def list_prs(args):
    """List pull requests."""
    return json.dumps(_fake_prs, indent=2)

github_server.register_tool({
    "name": "list_issues",
    "description": "List GitHub issues filtered by status.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "Filter by status",
                "enum": ["open", "closed"],
                "default": "open"
            }
        }
    }
}, list_issues)

github_server.register_tool({
    "name": "create_issue",
    "description": "Create a new GitHub issue with a title and description.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Issue title"},
            "body":  {"type": "string", "description": "Issue description (optional)"}
        },
        "required": ["title"]
    }
}, create_issue)

github_server.register_tool({
    "name": "list_prs",
    "description": "List all pull requests in the repository.",
    "inputSchema": {
        "type": "object",
        "properties": {}
    }
}, list_prs)

# Build a NEW client and connect to BOTH servers
multi_client = SimulatedMCPClient()
multi_client.connect(calc_server)     # Connect to calculator server
multi_client.connect(fs_server)       # Connect to filesystem server
multi_client.connect(github_server)   # Connect to github server
multi_client.list_all_tools()

# Directly call tools from different servers
print("\n--- Cross-Server Tool Calls ---")

print("\n[Call 1] Read a file via filesystem-server:")
file_content = multi_client.call_tool("read_file", {"path": "/project/main.py"})
print(f"  Content: {file_content}")

print("\n[Call 2] List open GitHub issues via github-server:")
issues = multi_client.call_tool("list_issues", {"status": "open"})
print(f"  Issues: {issues}")

print("\n[Call 3] Create a new GitHub issue via github-server:")
new_issue = multi_client.call_tool("create_issue", {"title": "Implement MCP client", "body": "Track MCP integration work"})
print(f"  Result: {new_issue}")

print("\n[Call 4] Add numbers via calculator-server:")
sum_result = multi_client.call_tool("add", {"a": 100, "b": 200})
print(f"  Result: {sum_result}")


# ==============================================================================
# PART D: MCP vs Function Calling Comparison
# ==============================================================================

print("\n" + "=" * 65)
print("PART D: MCP vs Function Calling")
print("=" * 65)

print("""
FUNCTION CALLING (Lesson 02 style):
  - Tools are defined as dicts IN YOUR AGENT CODE
  - Agent calls Python functions directly (same process)
  - No discovery: you hardcode the tool list when building the agent
  - Best for: simple agents, tools written in the same language

  Example:
    tools = [{"name": "add", "description": "...", "parameters": {...}}]
    agent = MyAgent(tools=tools)    # Tools baked in at construction time

MCP (this lesson):
  - Tools are defined in SEPARATE SERVER PROCESSES
  - Agent connects at runtime and DISCOVERS available tools
  - Any language can host the server (Python, Rust, Go, TypeScript)
  - Best for: enterprise setups, reusable tools, multi-team environments

  Example:
    client = MCPClient()
    client.connect("github-server")      # Server running in separate process
    client.connect("database-server")    # Another server
    # Client now has all tools -- discovered at runtime, not hardcoded
""")

# Side-by-side comparison
print("COMPARISON TABLE:")
print("-" * 70)
table = [
    ("Tool location",    "In your code",          "Separate server process"),
    ("Discovery",        "Hardcoded tool list",    "tools/list at runtime"),
    ("Language",         "Same as agent",          "Any language"),
    ("Reusability",      "One agent only",         "Any agent can connect"),
    ("Transport",        "Function call",          "stdio or HTTP+SSE"),
    ("Best for",         "Simple apps / POC",      "Enterprise / production"),
    ("Real example",     "ChatGPT plugins",        "Claude Code + GitHub MCP"),
]
print(f"  {'Feature':<20} {'Function Calling':<25} {'MCP'}")
print(f"  {'-'*20} {'-'*25} {'-'*25}")
for feature, fc, mcp in table:
    print(f"  {feature:<20} {fc:<25} {mcp}")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("SUMMARY - MCP Protocol")
print("=" * 65)
print("""
WHAT WE BUILT:
  - SimulatedMCPServer: registers tools with schemas, executes tools
  - SimulatedMCPClient: connects to servers, discovers tools, calls them
  - Three servers: calculator, filesystem, github
  - One client connected to all three -- 9 tools total, all discovered

KEY PATTERNS:
  1. tools/list  -> server returns all tool schemas (discovery)
  2. tools/call  -> server executes one tool, returns result
  3. Schema format: name + description + inputSchema (JSON Schema)
  4. One client, many servers: M+N instead of M*N integrations
  5. Tool result format: {"content": [{"type": "text", "text": "..."}]}

REAL-WORLD USE:
  - Claude Code uses MCP for: filesystem, GitHub, databases, web search
  - Install a server: npx -y @modelcontextprotocol/server-github
  - Configure in Claude Code settings -> Tools -> Add MCP Server
  - Claude Code auto-discovers all tools from configured servers

C# ANALOGY RECAP:
  MCP Server     = gRPC/WCF service that exposes named operations
  MCP Client     = gRPC/WCF client with auto-generated client from schema
  tools/list     = reflection or Swagger discovery
  tools/call     = typed remote method invocation
  Multiple servers = composite pattern over multiple microservices
""")

print("=" * 65)
print("END OF EXAMPLE 06")
print("=" * 65)

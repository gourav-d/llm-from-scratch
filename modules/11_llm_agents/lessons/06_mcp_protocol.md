# Lesson 06: MCP -- Model Context Protocol

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain what MCP is and why it was created
2. Describe the MCP server / client model
3. Explain tool schemas and how tools are discovered
4. List three real-world MCP server examples
5. Explain why MCP matters for agent interoperability

---

## GLOSSARY

```
MCP (Model Context Protocol):
  An open standard that defines HOW an AI agent connects to external tools.
  Created by Anthropic in 2024. Like USB-C for AI tools:
  any MCP-compatible agent + any MCP-compatible tool = they just work together.

MCP Server:
  A program that exposes tools, resources, and prompts over the MCP protocol.
  Examples: a database server, a GitHub server, a filesystem server, a calculator server.
  Think of it as a "plugin host" -- it runs in the background and waits for requests.
  C# analogy: like a gRPC service or a WCF service that exposes specific operations.

MCP Client:
  The AI agent (or application) that connects to MCP servers and calls their tools.
  Examples: Claude, Claude Code, any LLM agent that supports MCP.
  C# analogy: like a gRPC client or a REST client that consumes a service.

Tool Schema:
  A JSON description of a tool: its name, what it does, and what parameters it takes.
  The agent reads tool schemas to know WHICH tools are available and HOW to call them.
  C# analogy: like an interface definition or Swagger/OpenAPI spec.

Tool Discovery:
  The process of the client asking the server: "What tools do you have?"
  The server responds with a list of tool schemas.
  The agent then knows which tools it can use -- no hardcoding needed.

Resource:
  In MCP, a "resource" is data exposed by the server (files, database rows, API responses).
  Agents can read resources WITHOUT calling a tool.
  Like a GET endpoint that returns data directly.
  C# analogy: like an IQueryable or read-only repository.

Prompt:
  In MCP, a "prompt" is a pre-made prompt template the server provides.
  Agents can use these to get context-specific instructions.
  C# analogy: like a template string or T4 template stored server-side.

Transport:
  How MCP messages move between client and server.
  Two options: stdio (standard input/output, same machine) or HTTP+SSE (network).
  stdio is most common for local tools.
  C# analogy: like the binding in WCF -- how data physically moves.

Interoperability:
  The ability for different tools and agents to work together without custom integrations.
  Before MCP: every agent needed custom code for every tool.
  After MCP: write a tool once as an MCP server; any agent can use it.
```

---

## Part 1: The Problem MCP Solves

Before MCP, connecting agents to tools was a mess:

```
BEFORE MCP -- the "M x N" problem:

  3 agents (Claude, GPT, Gemini) x 5 tools (GitHub, Database, Calendar, Slack, Files)
  = 15 custom integrations to write and maintain

  ClaudeAgent  --custom--> GitHubConnector
  ClaudeAgent  --custom--> DatabaseConnector
  ClaudeAgent  --custom--> CalendarConnector
  GPTAgent     --custom--> GitHubConnector    (different code!)
  GPTAgent     --custom--> DatabaseConnector  (different code!)
  ... and so on for 15 total

  Every new tool = write N new integrations (one per agent).
  Every new agent = write M new integrations (one per tool).
  PAIN: Different APIs, different auth, different error handling.
```

```
AFTER MCP -- the "M + N" solution:

  Each tool publishes ONE MCP server.
  Each agent has ONE MCP client.

  GitHubServer  (MCP server)  -+-> MCP Protocol
  DatabaseServer (MCP server) -+-> MCP Protocol
  CalendarServer (MCP server) -+-> MCP Protocol

  ClaudeAgent  (MCP client)  <--> MCP Protocol
  GPTAgent     (MCP client)  <--> MCP Protocol
  GeminiAgent  (MCP client)  <--> MCP Protocol

  3 agents + 5 tools = 8 integrations total. Not 15.
  Add a new tool? Write ONE MCP server. All agents get it for free.
  Add a new agent? Support MCP once. All tools work immediately.
```

---

## Part 2: How MCP Works -- The Flow

```
MCP CONNECTION LIFECYCLE:

  1. STARTUP
     Client (agent) starts the MCP server as a subprocess
     OR connects to it over HTTP.

  2. INITIALIZE
     Client sends: {"method": "initialize", "params": {...}}
     Server responds: {"result": {"protocolVersion": "2024-11-05",
                                  "serverInfo": {"name": "github-mcp"}}}
     They shake hands and agree on protocol version.

  3. TOOL DISCOVERY
     Client sends: {"method": "tools/list"}
     Server responds: {"result": {"tools": [
       {
         "name": "create_issue",
         "description": "Create a GitHub issue",
         "inputSchema": {
           "type": "object",
           "properties": {
             "title":  {"type": "string", "description": "Issue title"},
             "body":   {"type": "string", "description": "Issue description"},
             "repo":   {"type": "string", "description": "Repository name"}
           },
           "required": ["title", "repo"]
         }
       },
       {
         "name": "list_prs",
         "description": "List open pull requests",
         "inputSchema": { "type": "object", "properties": {"repo": {"type": "string"}}}
       }
     ]}}

  4. TOOL CALL
     Client sends: {"method": "tools/call",
                    "params": {"name": "create_issue",
                               "arguments": {"title": "Fix bug", "repo": "myapp"}}}
     Server executes the tool and responds:
     {"result": {"content": [{"type": "text", "text": "Issue #42 created."}]}}

  5. REPEAT steps 3-4 as many times as needed.

  6. SHUTDOWN
     Client sends: {"method": "notifications/cancelled"} or just closes the connection.
```

---

## Part 3: Tool Schema in Detail

A tool schema tells the agent EXACTLY what a tool does and how to call it.

```json
{
  "name": "run_sql_query",

  "description": "Execute a read-only SQL query on the database.
                  Returns results as a list of rows.
                  Only SELECT queries are allowed.",

  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "SQL SELECT query to execute. No INSERT/UPDATE/DELETE."
      },
      "database": {
        "type": "string",
        "description": "Database name. Options: 'sales', 'inventory', 'users'",
        "enum": ["sales", "inventory", "users"]
      },
      "max_rows": {
        "type": "integer",
        "description": "Maximum rows to return (default 100, max 1000)",
        "default": 100
      }
    },
    "required": ["query", "database"]
  }
}
```

```
WHY SCHEMAS MATTER:
  The agent reads this schema and knows:
  - Name: run_sql_query (what to call in a tools/call request)
  - Description: what it does and what it CANNOT do (read-only)
  - Parameters: "query" (required string), "database" (required enum), "max_rows" (optional int)

  The agent can now call this tool correctly WITHOUT hardcoded knowledge of it.
  This is the "discovery" part: the agent figures out tools at runtime, not compile time.

  C# analogy: like reading a Swagger/OpenAPI spec at runtime and auto-generating
  a typed HTTP client from it. Or like reflection to discover available methods.
```

---

## Part 4: Resources and Prompts

MCP has two more features beyond tools:

### Resources -- Read Data Directly
```
Resources are like read-only files or database views the server exposes.
Agents access them with: resources/read {"uri": "github://myrepo/README.md"}

Examples:
  filesystem://  /home/user/project/config.json
  github://      anthropics/claude-code/README.md
  db://          sales/customers/top100

C# analogy: like IQueryable<T> -- data you can read but not directly execute.
Use when: you want the agent to see data WITHOUT calling an action tool.
```

### Prompts -- Server-Provided Templates
```
Prompts are pre-written instructions the server provides.
Agents call: prompts/get {"name": "code_review_prompt", "arguments": {"language": "Python"}}
Server returns: a ready-made system prompt for reviewing Python code.

Use when: you want consistent, context-specific instructions that live server-side.
C# analogy: like a T4 template or a localized string resource.
```

---

## Part 5: Real-World MCP Servers

Anthropic and the community have published many ready-to-use MCP servers:

```
OFFICIAL MCP SERVERS (open source, available on GitHub):

  mcp-server-filesystem
    Tools: read_file, write_file, list_directory, move_file, search_files
    Use case: agent reads/writes files on your computer
    C# analogy: System.IO wrapper exposed over MCP

  mcp-server-github
    Tools: create_issue, list_prs, get_file_contents, create_pull_request
    Use case: agent manages GitHub repos, creates issues, reviews PRs
    C# analogy: Octokit.NET client wrapped as MCP

  mcp-server-postgres
    Tools: query (SELECT), describe_table, list_tables
    Use case: agent queries your PostgreSQL database
    C# analogy: Dapper or EF Core query execution over MCP

  mcp-server-slack
    Tools: post_message, list_channels, get_thread
    Use case: agent sends Slack messages, reads channels
    C# analogy: Slack.NetStandard wrapped as MCP

  mcp-server-brave-search
    Tools: search (web search)
    Use case: agent searches the web for current information
    C# analogy: HttpClient to Brave Search API wrapped as MCP

  Claude Code itself IS an MCP client:
    - Open Claude Code, configure MCP servers in settings
    - Claude can now use GitHub, filesystem, databases as tools
    - No custom code needed -- just plug in the MCP server
```

---

## Part 6: Writing a Simple MCP Server (Concept)

A minimal Python MCP server looks like this (conceptual -- real code in example_06):

```python
# Minimal MCP server concept
# Real library: pip install mcp

from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("my-calculator")

# Declare a tool
@app.list_tools()
async def list_tools():
    return [
        {
            "name": "add",
            "description": "Add two numbers",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    ]

# Handle tool calls
@app.call_tool()
async def call_tool(name, arguments):
    if name == "add":
        return arguments["a"] + arguments["b"]

# Start the server (listens on stdin/stdout)
async def main():
    async with stdio_server() as streams:
        await app.run(*streams)
```

```
KEY CONCEPTS from above:
  @app.list_tools()  -> handler that returns all tool schemas
  @app.call_tool()   -> handler that executes a tool by name
  stdio_server()     -> uses stdin/stdout as transport (local machine)

  C# analogy:
    Server     = ASP.NET Core app or gRPC service
    list_tools = [HttpGet("tools")] controller action
    call_tool  = [HttpPost("tools/{name}")] controller action
    stdio      = IPC named pipes (local communication)
```

---

## Part 7: MCP vs Function Calling

You already learned function calling in Lesson 02. How is MCP different?

```
+------------------+-----------------------------+--------------------------------+
| Feature          | Function Calling (L02)      | MCP                            |
+------------------+-----------------------------+--------------------------------+
| Where tools live | Hardcoded in your code      | In a separate server process   |
| Discovery        | You pass tool list manually | Server advertises tools itself |
| Reusability      | One agent only              | Any agent can connect          |
| Languages        | Same as your agent          | Server can be ANY language     |
| Transport        | In-process function call    | stdio or HTTP+SSE              |
| Real-world use   | Simple apps                 | Enterprise, multi-team setups  |
| Examples         | "My agent has a calculator" | "Claude Code uses GitHub MCP"  |
+------------------+-----------------------------+--------------------------------+

Function calling = tools baked into your agent code.
MCP = tools published as independent services, any agent can use them.

Think of it this way:
  Function calling: like embedding a library directly (DLL reference)
  MCP: like calling a microservice (network call, language-agnostic)
```

---

## Key Takeaways

1. MCP solves the M×N integration problem: write tools once, use with any agent.

2. Flow: Initialize → Discover Tools → Call Tools → Get Results.

3. Tool schemas are JSON descriptions: name, description, inputSchema (parameters).

4. Resources = read data. Prompts = pre-made templates. Tools = executable actions.

5. Real MCP servers exist for: filesystem, GitHub, PostgreSQL, Slack, web search.

6. MCP vs function calling: function calling is baked in; MCP is an independent service.

7. Claude Code is already an MCP client — you can add MCP servers to it today.

---

*Next: Lesson 07 — LangGraph: State Machines for Agent Workflows*

"""
Example 02: Tool Use -- Multiple Tools in Action
=================================================

GLOSSARY
--------
Tool Registry:
  A dictionary that maps tool names to Python functions.
  The agent looks up tools here by name.
  Like a DI container in C#: register tools, resolve by name.

Tool Definition:
  A dictionary describing a tool: name, description, parameters.
  The LLM reads this to decide WHEN and HOW to use each tool.

Tool Selection:
  The LLM's decision about WHICH tool to call for a given step.
  It reads the tool descriptions and picks the most relevant one.

Multi-Tool Agent:
  An agent with access to many tools. It picks the right one per step.
  More powerful than a single-tool agent.

Calculator Tool:
  Evaluates math expressions. For any arithmetic or algebra.

Search Tool:
  Finds information from the web (or a simulated database here).

File Reader Tool:
  Reads the contents of a text file.

String Tool:
  Manipulates strings: count words, reverse, uppercase, etc.

WHAT THIS EXAMPLE SHOWS
------------------------
Part A: An agent with 4 tools -- it selects the right tool per question
Part B: Demonstrates parallel tool calling (two tools at once)

LIBRARIES NEEDED
-----------------
  None (pure Python, no external packages)
"""

import math      # For math operations in calculator
import os        # For file operations in file reader

print("=" * 65)
print("EXAMPLE 02: Tool Use -- Multiple Tools in Action")
print("=" * 65)


# ==============================================================================
# PART A: Agent with Multiple Tools
# ==============================================================================

print("\n" + "=" * 65)
print("PART A: Agent Selects the Right Tool")
print("=" * 65)

print("""
We will build an agent with 4 tools:
  1. calculator  -- for math
  2. search_web  -- for facts
  3. read_file   -- for reading text files
  4. string_tool -- for text manipulation

The agent reads the user's question and picks the CORRECT tool.
This is the core skill of a real LLM agent.
""")


# -----------------------------------------------------------------------
# Tool 1: Calculator
# -----------------------------------------------------------------------

def calculator(expression: str) -> str:
    """
    Evaluates a math expression.
    expression: Python math expression, e.g., "sqrt(144)" or "2**10"
    Returns: the numeric result as a string

    C# analogy: like calling a calculator service:
      var result = calculatorService.Evaluate("sqrt(144)");
    """
    try:
        # Allow only math module functions (no file access, no network)
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_env)   # Evaluate safely
        return f"{result}"                             # Return result as string
    except Exception as e:
        return f"Calculator error: {e}"


# -----------------------------------------------------------------------
# Tool 2: Web Search (simulated)
# -----------------------------------------------------------------------

def search_web(query: str) -> str:
    """
    Searches for information (simulated for learning -- no real internet).
    query: what to search for
    Returns: simulated search result

    In production: would call a real search API (Bing, Google, Serper, Tavily).
    """
    # Our "fake internet" -- a dictionary of facts
    fake_web = {
        "python creator":           "Python was created by Guido van Rossum, first released in 1991.",
        "speed of light":           "The speed of light in a vacuum is 299,792,458 meters per second.",
        "mount everest height":     "Mount Everest is 8,848.86 meters (29,031.7 feet) above sea level.",
        "water boiling point":      "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
        "earth population":         "Earth's population is approximately 8.1 billion people (2024).",
        "microsoft founded":        "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
        "machine learning":         "Machine learning is a subset of AI where systems learn from data without explicit programming.",
        "transformer architecture": "Transformers use self-attention to process sequences in parallel. Introduced in 'Attention Is All You Need' (2017).",
    }

    query_lower = query.lower()                    # Lowercase for matching
    for key, value in fake_web.items():            # Check each entry
        if key in query_lower or any(word in query_lower for word in key.split()):
            return value                           # Return matching result

    return f"No web results found for: '{query}'"


# -----------------------------------------------------------------------
# Tool 3: File Reader
# -----------------------------------------------------------------------

def read_file(path: str) -> str:
    """
    Reads the contents of a text file.
    path: the file path (relative or absolute)
    Returns: file contents, or an error message if file not found
    """
    try:
        with open(path, "r", encoding="utf-8") as f:   # Open file for reading
            contents = f.read()                         # Read all content
        return contents[:500] + ("..." if len(contents) > 500 else "")  # Limit to 500 chars
    except FileNotFoundError:
        return f"File not found: '{path}'"
    except Exception as e:
        return f"File read error: {e}"


# -----------------------------------------------------------------------
# Tool 4: String Tool
# -----------------------------------------------------------------------

def string_tool(operation: str, text: str) -> str:
    """
    Performs text manipulation operations.
    operation: what to do -- "count_words", "uppercase", "lowercase",
               "reverse", "word_count", "char_count"
    text:      the text to process
    Returns:   the result as a string

    C# analogy: like calling string.ToUpper(), string.Split(), etc.
    """
    ops = {
        "count_words":  lambda t: str(len(t.split())),           # Count words
        "uppercase":    lambda t: t.upper(),                      # ALL CAPS
        "lowercase":    lambda t: t.lower(),                      # all lowercase
        "reverse":      lambda t: t[::-1],                        # Reverse the string
        "char_count":   lambda t: str(len(t)),                    # Count characters
        "word_list":    lambda t: str(t.split()),                  # List of words
    }

    if operation not in ops:
        return f"Unknown operation: '{operation}'. Valid: {list(ops.keys())}"

    return ops[operation](text)                                   # Call the right operation


# -----------------------------------------------------------------------
# Tool Definitions -- what the LLM reads to decide which tool to use
# -----------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": "Evaluates a mathematical expression. Use for ANY arithmetic, "
                       "geometry, algebra, or number computation. Input must be a "
                       "Python math expression. Examples: '2+2', 'sqrt(144)', '2**10'.",
        "parameters": {
            "expression": "Python math expression string. Example: 'sqrt(1764)'"
        }
    },
    {
        "name": "search_web",
        "description": "Searches for factual information. Use when you need facts, "
                       "historical data, scientific constants, or general knowledge. "
                       "Do NOT use for math -- use calculator for that.",
        "parameters": {
            "query": "What to search for. Be specific. Example: 'speed of light in meters'"
        }
    },
    {
        "name": "read_file",
        "description": "Reads the contents of a text file. Use when the user asks "
                       "about the contents of a specific file.",
        "parameters": {
            "path": "File path. Example: 'data/notes.txt'"
        }
    },
    {
        "name": "string_tool",
        "description": "Manipulates text strings. Use for: counting words, "
                       "uppercasing, lowercasing, reversing text, counting characters.",
        "parameters": {
            "operation": "One of: count_words, uppercase, lowercase, reverse, char_count, word_list",
            "text": "The text to process"
        }
    },
]

# Tool Registry: maps names to real Python functions
TOOL_REGISTRY = {
    "calculator": calculator,
    "search_web": search_web,
    "read_file":  read_file,
    "string_tool": string_tool,
}


# -----------------------------------------------------------------------
# Smarter FakeLLM -- selects the correct tool based on the question
# -----------------------------------------------------------------------

class ToolSelectorLLM:
    """
    A smarter FakeLLM that mimics how a real LLM selects tools.
    It reads the tool definitions (like a real LLM does) and picks
    the most appropriate tool for each question.

    In real life: Claude or GPT reads the tool descriptions and
    decides which one to call. This class mimics that decision logic.
    """

    def decide(self, question: str) -> dict:
        """
        Given a user question, decide which tool to use and what args to pass.
        Returns a dict: {"tool": "...", "args": {...}}
        """
        q = question.lower()    # Lowercase for easy matching

        # Math questions -> calculator
        math_keywords = ["calculate", "what is", "compute", "sqrt", "square root",
                         "power", "**", "multiply", "divide", "plus", "minus",
                         "how many meters", "how many km", "fahrenheit", "celsius"]
        if any(kw in q for kw in math_keywords):
            # Extract the expression part (after "calculate" or "what is")
            expr = question
            for prefix in ["calculate", "what is", "compute"]:
                if prefix in q:
                    expr = question.lower().split(prefix)[-1].strip()
                    break
            return {
                "tool": "calculator",
                "args": {"expression": expr.replace("^", "**")}   # ^ -> ** for Python
            }

        # File questions -> read_file
        if "read" in q and ("file" in q or ".txt" in q or ".md" in q):
            # Extract file path -- look for quoted string or last word
            import re
            quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", question)
            if quoted:
                path = quoted[0][0] or quoted[0][1]
            else:
                path = question.split()[-1]    # Guess: last word is the path
            return {"tool": "read_file", "args": {"path": path}}

        # String operations -> string_tool
        string_ops = {"uppercase": "uppercase", "lowercase": "lowercase",
                      "reverse": "reverse", "count words": "count_words",
                      "count characters": "char_count", "word list": "word_list"}
        for phrase, op in string_ops.items():
            if phrase in q:
                # Extract the text to operate on (after the operation keyword)
                text = question.split(phrase)[-1].strip().strip("'\"")
                return {"tool": "string_tool", "args": {"operation": op, "text": text}}

        # Default: search for facts
        return {"tool": "search_web", "args": {"query": question}}


# -----------------------------------------------------------------------
# Test the Tool Selector
# -----------------------------------------------------------------------

print("Testing Tool Selector (which tool for which question?):\n")

selector = ToolSelectorLLM()

test_questions = [
    "What is sqrt(1764)?",
    "Calculate 2 ** 10",
    "Who created Python?",
    "What is the speed of light?",
    "Uppercase hello world",
    "Count words in this is a test sentence",
    "What is the height of Mount Everest?",
]

for question in test_questions:
    decision = selector.decide(question)      # LLM decides which tool
    tool_name = decision["tool"]
    args = decision["args"]

    # Actually call the tool
    result = TOOL_REGISTRY[tool_name](**args)

    print(f"Q: {question}")
    print(f"   Tool selected: {tool_name}")
    print(f"   Args: {args}")
    print(f"   Result: {result}")
    print()


# ==============================================================================
# PART B: Parallel Tool Calling
# ==============================================================================

print("=" * 65)
print("PART B: Parallel Tool Calling")
print("=" * 65)

print("""
Sometimes a question needs MULTIPLE tools, and the results are independent.
Instead of calling them one by one, we can call them in PARALLEL.

Example: "What is sqrt(144) AND who created Python?"
  - sqrt(144) needs calculator
  - "who created Python" needs search_web
  - These are INDEPENDENT -- we can run both at the same time

C# analogy: Task.WhenAll() -- run tasks in parallel, wait for all to finish.
""")

import concurrent.futures   # Python's parallel task executor (like Task.WhenAll in C#)

def run_tools_in_parallel(tool_calls: list) -> list:
    """
    Execute multiple tool calls simultaneously.
    tool_calls: list of {"tool": "...", "args": {...}} dicts
    Returns: list of {"tool": "...", "result": "..."} dicts

    C# equivalent:
      var tasks = toolCalls.Select(call => Task.Run(() => Execute(call)));
      var results = await Task.WhenAll(tasks);
    """

    def execute_one(call: dict) -> dict:
        """Execute a single tool call and return the result."""
        tool_name = call["tool"]
        args = call["args"]

        if tool_name not in TOOL_REGISTRY:
            result = f"Error: Tool '{tool_name}' not found."
        else:
            result = TOOL_REGISTRY[tool_name](**args)    # Call the tool

        return {"tool": tool_name, "args": args, "result": result}

    # Run all tool calls simultaneously
    # ThreadPoolExecutor uses multiple threads -- like Task.Run() in C#
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all tasks at once, then wait for all of them
        futures = [executor.submit(execute_one, call) for call in tool_calls]  # Start all
        results = [f.result() for f in futures]        # Wait for each to finish

    return results


# Test parallel tool calling
parallel_tasks = [
    {"tool": "calculator", "args": {"expression": "sqrt(144)"}},      # Math task
    {"tool": "search_web", "args": {"query": "who created Python"}},  # Fact task
    {"tool": "string_tool", "args": {"operation": "uppercase", "text": "hello world"}},  # String task
]

print("Calling 3 tools in parallel:")
for task in parallel_tasks:
    print(f"  - {task['tool']}({task['args']})")
print()

results = run_tools_in_parallel(parallel_tasks)

print("Results (all computed simultaneously):")
for r in results:
    print(f"  {r['tool']}: {r['result']}")
print()

print("""
In a real agent, parallel tool calling means:
  - Multiple searches happen at the same time
  - The agent gets all results before deciding the next step
  - Much faster for complex questions!

Real LLMs (Claude, GPT-4) support this natively -- they can output
multiple tool calls in a single response.
""")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("=" * 65)
print("SUMMARY - Tool Use")
print("=" * 65)

print("""
WHAT WE BUILT:
  - 4 tools: calculator, search_web, read_file, string_tool
  - A tool selector that picks the right tool for each question
  - Parallel tool execution (multiple tools at once)

KEY LESSONS:
  1. Tool descriptions matter. The LLM reads them to decide which to call.
  2. One question can need multiple independent tools -> run in parallel.
  3. The tool registry maps names (strings) to actual Python functions.
  4. Always validate tool inputs and handle errors gracefully.

C# ANALOGY:
  Tool registry = Dictionary<string, Func<Dictionary<string,string>, string>>
  Parallel calls = Task.WhenAll() with tool Task.Run() calls
  Tool definitions = XML doc comments + interface definitions

REAL WORLD:
  In production, tools can call:
  - APIs (Stripe, Twilio, Salesforce)
  - Databases (SQL, NoSQL)
  - Files (read/write)
  - Code runners (Python REPL, Jupyter)
  - Vector databases (from Module 10)
  - External agents (multi-agent, next lesson)

NEXT EXAMPLE (03):
  Full ReAct loop -- multi-step reasoning with tool use.
  The agent solves problems that require 3-5 tool calls in sequence.
""")

print("=" * 65)
print("END OF EXAMPLE 02")
print("=" * 65)

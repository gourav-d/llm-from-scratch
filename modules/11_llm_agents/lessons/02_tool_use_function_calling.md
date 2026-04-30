# Lesson 02: Tool Use and Function Calling

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain how an LLM decides which tool to call
2. Describe the structure of a tool definition (name, description, parameters)
3. Trace the full flow from user request to tool call to final answer
4. Write a tool definition in Python
5. Explain what "function calling" means in the context of LLM APIs

---

## GLOSSARY

```
Function Calling:
  A feature in modern LLM APIs (OpenAI, Anthropic) where the LLM can
  output a STRUCTURED decision: "Call this function with these arguments."
  Instead of writing free text, the LLM writes: {"tool": "calculator", "args": {"expression": "2^32"}}
  Your code then ACTUALLY calls that function and returns the result.

Tool Definition:
  A description of a function the LLM can call. Includes:
    name:        what it is called
    description: what it does (THIS IS WHAT THE LLM READS!)
    parameters:  what inputs it needs (name, type, description for each)

Tool Registry:
  A dictionary of all available tools. The agent looks up tools here.
  Like a DI container in C# -- you register tools, then resolve them by name.

Tool Call:
  When the LLM decides to use a tool. It outputs the tool name and arguments.
  Your code intercepts this, calls the real function, and returns the result.

Tool Result:
  What the tool returns after executing. Sent back to the LLM as context
  for the next step of reasoning.

Structured Output:
  When an LLM returns JSON or another structured format instead of plain text.
  Function calling is a specific kind of structured output: tool name + args.

Schema:
  The formal description of what a tool expects. Like a C# interface definition.
  In Python: usually a dictionary describing parameter names and types.
```

---

## Part 1: How the LLM Decides What Tool to Call

The LLM does NOT magically know which tools exist.
You must TELL it what tools are available by including their descriptions in the prompt.

Here is what the LLM actually "sees":

```
SYSTEM PROMPT (what you send to the LLM):
================================================
You are a helpful assistant. You have access to these tools:

Tool 1: calculator
  Description: Evaluates a mathematical expression and returns the result.
               Use this when the user asks you to compute something.
  Parameters:
    expression (string): A mathematical expression like "2 + 2" or "sqrt(16)"

Tool 2: search_web
  Description: Searches the internet for current information.
               Use this when the user asks about recent events or facts you may not know.
  Parameters:
    query (string): The search query string

Tool 3: read_file
  Description: Reads the contents of a local text file.
  Parameters:
    path (string): The file path, e.g., "data/report.txt"

When you need to use a tool, respond ONLY with JSON in this format:
{
  "tool": "tool_name",
  "args": {"param1": "value1", "param2": "value2"}
}

When you have enough information and do not need a tool, respond with:
{
  "tool": "final_answer",
  "answer": "Your complete answer here."
}
================================================

USER MESSAGE: "What is the square root of 1764?"
```

The LLM reads this and decides:
"The user wants a calculation. I have a calculator tool. I should use it."
It outputs:
```json
{
  "tool": "calculator",
  "args": {"expression": "sqrt(1764)"}
}
```

Your code then ACTUALLY calls `calculator(expression="sqrt(1764)")`, gets `42.0`,
and sends that result back to the LLM. The LLM then writes the final answer.

---

## Part 2: The Full Tool Call Flow

```
Step 1: User asks a question
        User: "What is the square root of 1764?"

Step 2: You send the question + tool descriptions to the LLM
        [Tool list] + "What is the square root of 1764?"

Step 3: LLM outputs a tool call (JSON, not English!)
        {"tool": "calculator", "args": {"expression": "sqrt(1764)"}}

Step 4: Your code intercepts the JSON and calls the real function
        result = calculator(expression="sqrt(1764)")
        result == 42.0

Step 5: You send the result BACK to the LLM as context
        "Tool result: 42.0"

Step 6: LLM generates the final answer
        "The square root of 1764 is 42."
```

In diagram form:

```
User                  Your Code               LLM
 |                        |                    |
 |--- "sqrt(1764)?" ---->>|                    |
 |                        |-- [tools + q] -->> |
 |                        |                    |-- thinks...
 |                        |                    |
 |                        |<-- {tool: calc} ---|
 |                        |                    |
 |                        |-- runs calc() --   |
 |                        |   result = 42.0    |
 |                        |                    |
 |                        |-- "result: 42.0" ->|
 |                        |                    |-- thinks...
 |                        |                    |
 |                        |<-- "sqrt(1764)=42" |
 |                        |                    |
 |<--- "sqrt(1764)=42" ---|                    |
```

---

## Part 3: Writing Tool Definitions in Python

A tool definition describes what the tool does.
The LLM reads this description to decide whether and how to call the tool.

```python
# Tool definitions -- a list of dictionaries
# Each dictionary tells the LLM about one tool

TOOLS = [
    {
        "name": "calculator",
        # CRITICAL: The description must be clear. The LLM decides based on this!
        "description": "Evaluates a mathematical expression. Use for any arithmetic, "
                       "algebra, or math computation. Input should be a valid Python math expression.",
        "parameters": {
            "expression": {
                "type": "string",
                "description": "A Python math expression. Examples: '2 + 2', 'sqrt(16)', '2**10'"
            }
        }
    },
    {
        "name": "search_web",
        "description": "Searches the internet for current information. Use when you need "
                       "up-to-date facts, recent news, or information you might not know.",
        "parameters": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific. Example: 'Python 3.12 new features'"
            }
        }
    },
    {
        "name": "read_file",
        "description": "Reads and returns the contents of a text file.",
        "parameters": {
            "path": {
                "type": "string",
                "description": "The file path relative to the project root. Example: 'data/report.txt'"
            }
        }
    }
]
```

### C# Analogy: Like a Service Interface

```csharp
// Tool definition in C# is like an interface + XML doc comment:

/// <summary>
/// Evaluates a mathematical expression.
/// Use for any arithmetic, algebra, or math computation.
/// </summary>
/// <param name="expression">A C# math expression, e.g., "Math.Sqrt(1764)"</param>
public interface ICalculatorTool {
    double Calculate(string expression);
}
```

The XML doc comment is EXACTLY like the tool description -- both tell the CALLER
(human or LLM) how to use the function correctly.

---

## Part 4: The Tool Registry and Executor

After the LLM decides which tool to call, your code needs to:
1. Find the real Python function for that tool
2. Call it with the LLM's arguments
3. Return the result to the LLM

```python
import math     # For math calculations

# ---------------------------------------------------------------
# Step 1: Write the actual Python functions (the REAL tools)
# ---------------------------------------------------------------

def calculator(expression: str) -> str:
    """
    Actually evaluates a math expression.
    expression: a string like "sqrt(1764)" or "2 + 2"
    Returns: the result as a string
    """
    try:
        # Safe evaluation: only allow math functions
        # eval() runs Python code -- NEVER run user input through eval() in production!
        # For learning purposes, we restrict it to math module functions
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def search_web(query: str) -> str:
    """
    Simulates a web search. In production, this would call a real search API.
    query: the search string
    Returns: simulated search results
    """
    # In a real agent, this would call Google, Bing, or DuckDuckGo API
    # For learning, we return fake results based on keywords
    fake_results = {
        "python frameworks": "Top Python frameworks: Django (web), FastAPI (API), Flask (micro), Pandas (data), PyTorch (ML)",
        "microsoft ceo":     "Satya Nadella is the CEO of Microsoft since 2014.",
        "population paris":  "Paris city population: approximately 2.16 million (2024). Metro: 12 million.",
        "population london": "London city population: approximately 8.9 million (2024). Metro: 14 million.",
    }
    query_lower = query.lower()
    for key, value in fake_results.items():
        if key in query_lower:
            return value
    return f"Search results for '{query}': No results found in simulation."

def read_file(path: str) -> str:
    """
    Reads the contents of a file.
    path: the file path
    Returns: file contents as a string
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{path}' not found."
    except Exception as e:
        return f"Error reading file: {e}"

# ---------------------------------------------------------------
# Step 2: The Tool Registry -- maps tool names to Python functions
# ---------------------------------------------------------------
# In C#, this is like: Dictionary<string, Func<Dictionary<string,string>, string>>
# Or a service locator pattern with a key-to-handler mapping

TOOL_REGISTRY = {
    "calculator": calculator,     # name -> actual function
    "search_web": search_web,
    "read_file":  read_file,
}

# ---------------------------------------------------------------
# Step 3: The Tool Executor -- calls the right tool with right args
# ---------------------------------------------------------------

def execute_tool(tool_name: str, args: dict) -> str:
    """
    Looks up the tool by name and calls it with the given arguments.
    tool_name: which tool to call (must match a key in TOOL_REGISTRY)
    args:      a dictionary of argument names to values
    Returns:   the tool's output as a string
    """
    if tool_name not in TOOL_REGISTRY:
        return f"Error: Tool '{tool_name}' not found."

    tool_function = TOOL_REGISTRY[tool_name]   # Get the Python function
    return tool_function(**args)               # Call it with the args dict unpacked
                                               # **args unpacks {"expression": "sqrt(16)"} as expression="sqrt(16)"
```

---

## Part 5: Why Tool Descriptions Matter So Much

The LLM cannot run your code to understand what a tool does.
It ONLY reads the description to decide when and how to call it.

### Bad Tool Description (LLM will misuse it)
```python
{
    "name": "calc",
    "description": "Does math stuff",          # Too vague!
    "parameters": {
        "x": {"type": "string", "description": "input"}  # What kind of input??
    }
}
```

The LLM might call this with `x = "What is 2+2?"` (wrong) instead of `x = "2+2"` (correct).

### Good Tool Description (LLM uses it correctly)
```python
{
    "name": "calculator",
    "description": "Evaluates a mathematical expression and returns the numeric result. "
                   "Use this for ANY arithmetic, algebra, or math problem. "
                   "Do NOT use this for string operations or non-math tasks.",
    "parameters": {
        "expression": {
            "type": "string",
            "description": "A valid Python math expression. "
                           "Examples: '2 + 2', 'sqrt(16)', '2**10', '(5 * 3) / 2'"
        }
    }
}
```

Rule of thumb: Write descriptions as if explaining the tool to a very literal-minded person.
Be specific. Include examples. State what NOT to use it for.

---

## Part 6: Parallel Tool Calls

Modern LLMs can call multiple tools AT THE SAME TIME if the tasks are independent.

```
User: "What is sqrt(144) AND what is the population of Paris?"

LLM response (parallel calls):
[
  {"tool": "calculator", "args": {"expression": "sqrt(144)"}},
  {"tool": "search_web", "args": {"query": "population of Paris 2024"}}
]

Your code runs BOTH simultaneously:
  Result 1: "12.0"
  Result 2: "Paris city population: 2.16 million"

LLM generates final answer:
  "sqrt(144) = 12. The population of Paris is approximately 2.16 million."
```

This is like running parallel Tasks in C#:
```csharp
var tasks = new Task<string>[] {
    Task.Run(() => calculator.Evaluate("sqrt(144)")),
    Task.Run(() => searchEngine.Search("population of Paris 2024"))
};
await Task.WhenAll(tasks);
string[] results = tasks.Select(t => t.Result).ToArray();
```

---

## Key Takeaways

1. Tool calling = LLM outputs JSON with tool name + args. Your code runs the real function.

2. The LLM decides WHICH tool to call by reading the tool DESCRIPTION (not the code!).

3. Tool definitions need: name, description, and parameter descriptions.

4. The flow: User question -> [tools + question] to LLM -> LLM outputs JSON ->
   you call the function -> result back to LLM -> LLM writes final answer.

5. Write descriptions clearly. Vague descriptions = wrong tool calls.

6. LLMs can call multiple tools in parallel for independent sub-tasks.

---

## Next

Lesson 03: The ReAct Pattern
  - What is the Reason-Act loop?
  - How does the agent combine thinking and doing?
  - How do we implement a multi-step reasoning loop from scratch?

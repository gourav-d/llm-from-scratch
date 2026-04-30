"""
Example 01: Your First Agent -- Simple Step Planner
====================================================

GLOSSARY
--------
Agent:
  An LLM that can plan and take actions to achieve a goal.
  This example uses a SIMULATED LLM (no API key needed).
  The simulation shows the exact same patterns used in real agents.

FakeLLM:
  A class that pretends to be an LLM. It returns predefined responses
  based on keywords. This lets us learn agent patterns without needing
  an actual API key or internet connection.

Plan:
  A list of steps the agent will take to achieve the goal.
  Like a shopping list -- decide what to do, then do each item.

Step:
  One action in the plan. Could be: search, calculate, read, etc.

Goal:
  The task the user wants the agent to accomplish.

WHAT THIS EXAMPLE SHOWS
------------------------
Part A: A simple step-by-step planner using a FakeLLM (no API key needed)
Part B: Shows how to swap in a REAL LLM (Anthropic Claude) -- requires API key

WHY SIMULATED FIRST?
  You can run Part A immediately without setting up any accounts.
  It shows EXACTLY how a real agent works, just without internet.
  Once you understand the pattern, Part B shows the real thing.

LIBRARIES NEEDED
-----------------
  None for Part A (uses only Python built-ins)
  anthropic  (pip install anthropic) for Part B -- requires ANTHROPIC_API_KEY
"""

import json        # For parsing JSON responses from the LLM
import os          # For reading environment variables (API keys)

print("=" * 65)
print("EXAMPLE 01: Your First Agent")
print("=" * 65)


# ==============================================================================
# PART A: Simulated Agent (No API Key Needed)
# ==============================================================================

print("\n" + "=" * 65)
print("PART A: Simulated Agent")
print("=" * 65)

print("""
We will build an agent that:
1. Receives a GOAL from the user
2. Plans the steps needed to achieve it
3. Executes each step (using simulated tools)
4. Returns a final answer

The FakeLLM simulates what a real LLM would output.
The patterns here are IDENTICAL to real LLM agents.
""")


# -----------------------------------------------------------------------
# Step 1: Define the FakeLLM (simulates an LLM's responses)
# -----------------------------------------------------------------------

class FakeLLM:
    """
    Simulates an LLM by returning predefined responses.
    In real code, this would call the Anthropic or OpenAI API.
    The interface (inputs/outputs) is identical to a real LLM call.

    C# analogy: A test mock (ILlmService mock) used in unit tests.
    """

    def think(self, goal: str, history: list) -> dict:
        """
        Given a goal and conversation history, decide the next step.
        Returns a dict with: thought, tool name, and tool args.

        goal:    the user's original request
        history: list of past {thought, action, observation} dicts
        Returns: {"thought": "...", "tool": "...", "args": {...}}
        """

        # Count how many steps have been taken already
        step_number = len(history) + 1

        # ---- Simulated decision logic (mimics what a real LLM would do) ----

        # First step: make a plan
        if step_number == 1 and "population" in goal.lower():
            return {
                "thought": "The user wants population info. I'll search for it.",
                "tool": "search_web",
                "args": {"query": "population of Paris 2024"}
            }

        # Second step: after getting Paris population, get London's
        if step_number == 2 and "paris" in str(history).lower():
            return {
                "thought": "I have Paris population. Now I need London's population.",
                "tool": "search_web",
                "args": {"query": "population of London 2024"}
            }

        # Third step: compare and answer
        if step_number == 3:
            return {
                "thought": "I have both populations now. I can write the final comparison.",
                "tool": "final_answer",
                "args": {
                    "answer": "Paris has a city population of about 2.16 million "
                              "(12M metro area), while London has about 8.9 million "
                              "(14M metro area). London is roughly 4x larger."
                }
            }

        # Calculator goal
        if step_number == 1 and "calculate" in goal.lower():
            import re
            numbers = re.findall(r'\d+', goal)    # Find numbers in the goal
            if len(numbers) >= 2:
                expression = " + ".join(numbers)  # Simple: add them
            else:
                expression = goal.split("calculate")[-1].strip()   # Take the math part
            return {
                "thought": "The user wants a calculation. I'll use the calculator tool.",
                "tool": "calculator",
                "args": {"expression": expression}
            }

        if step_number == 2 and "calculator" in str(history[0]).lower():
            return {
                "thought": "I have the calculation result. I can answer now.",
                "tool": "final_answer",
                "args": {"answer": f"The result is: {history[-1].get('observation', 'unknown')}"}
            }

        # Fallback: if we get confused, just answer
        return {
            "thought": "I have gathered enough information to answer.",
            "tool": "final_answer",
            "args": {"answer": "Based on my research, I can provide the following answer."}
        }


# -----------------------------------------------------------------------
# Step 2: Define the Tools (the things the agent can DO)
# -----------------------------------------------------------------------

def search_web(query: str) -> str:
    """
    Simulates a web search.
    In production: would call Google API, Bing API, or DuckDuckGo.
    query: what to search for
    Returns: simulated search result string
    """
    # Fake search database -- maps keywords to results
    search_database = {
        "population of paris":   "Paris city population: approximately 2.16 million (2024). Metro area: 12 million.",
        "population of london":  "London city population: approximately 8.9 million (2024). Metro area: 14 million.",
        "python frameworks":     "Top Python web frameworks: Django, FastAPI, Flask, Tornado, Starlette.",
        "ceo of microsoft":      "Satya Nadella is the CEO of Microsoft since February 2014.",
        "anthropic claude":      "Claude is an AI assistant made by Anthropic, founded in 2021.",
    }

    query_lower = query.lower()                        # Make case-insensitive
    for key, result in search_database.items():        # Check each entry
        if key in query_lower:                         # If key found in query
            return result                              # Return the fake result

    return f"No results found for: '{query}'"         # Default if nothing matches


def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression.
    expression: a Python math expression like "2 + 2" or "sqrt(16)"
    Returns: the result as a string
    """
    import math                                       # Import math for sqrt, etc.
    try:
        # Allow only math functions (safety measure)
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_env)   # Evaluate safely
        return str(result)                            # Convert result to string
    except Exception as e:
        return f"Calculator error: {e}"              # Return error if expression is invalid


# -----------------------------------------------------------------------
# Step 3: The Tool Registry (maps names to functions)
# -----------------------------------------------------------------------

TOOLS = {
    "search_web": search_web,      # "search_web" -> the search_web() function
    "calculator": calculator,      # "calculator" -> the calculator() function
}


# -----------------------------------------------------------------------
# Step 4: The Agent -- the loop that connects everything
# -----------------------------------------------------------------------

class SimpleAgent:
    """
    A basic agent that:
    1. Asks the LLM what to do next
    2. Calls the appropriate tool
    3. Records the result
    4. Repeats until the LLM says "final_answer"

    C# analogy: A while-loop that calls services until a StopCondition is met.
    """

    def __init__(self, llm, tools: dict, max_steps: int = 5):
        """
        llm:       the language model (real or fake)
        tools:     dictionary of tool_name -> function
        max_steps: stop after this many steps (safety limit)
        """
        self.llm = llm                 # The brain
        self.tools = tools             # The available actions
        self.max_steps = max_steps     # Safety limit
        self.history = []              # Scratchpad: all steps taken so far

    def run(self, goal: str) -> str:
        """
        Run the agent on a goal.
        goal: the user's request
        Returns: the final answer string
        """
        print(f"\nGOAL: {goal}")
        print("-" * 50)

        self.history = []              # Start fresh for each new goal

        for step in range(1, self.max_steps + 1):    # Up to max_steps iterations
            print(f"\n[Step {step}]")

            # ---- THINK ----
            # Ask the LLM: given the goal and history so far, what next?
            decision = self.llm.think(goal, self.history)

            thought = decision.get("thought", "")              # Why doing this?
            tool_name = decision.get("tool", "final_answer")  # What to do?
            args = decision.get("args", {})                    # With what args?

            print(f"  THOUGHT:     {thought}")
            print(f"  TOOL:        {tool_name}")
            print(f"  ARGS:        {args}")

            # ---- CHECK IF DONE ----
            if tool_name == "final_answer":
                answer = args.get("answer", "No answer provided.")
                print(f"\nFINAL ANSWER: {answer}")
                return answer              # Done! Return the answer.

            # ---- ACT ----
            # Call the real tool function
            if tool_name in self.tools:
                observation = self.tools[tool_name](**args)   # ** unpacks dict as kwargs
            else:
                observation = f"Error: Tool '{tool_name}' does not exist."

            print(f"  OBSERVATION: {observation}")

            # ---- RECORD ----
            # Add this step to history so the LLM can see it next time
            self.history.append({
                "step": step,
                "thought": thought,
                "tool": tool_name,
                "args": args,
                "observation": observation
            })

        # If we used all steps without finishing, return a partial answer
        return "Maximum steps reached. Could not complete the task."


# -----------------------------------------------------------------------
# Run the Agent on some goals
# -----------------------------------------------------------------------

print("\nCreating agent with FakeLLM...")
fake_llm = FakeLLM()                              # The simulated brain
agent = SimpleAgent(fake_llm, TOOLS, max_steps=5) # Create the agent

# Test 1: Population comparison
print("\n" + "=" * 65)
print("TEST 1: Compare city populations")
print("=" * 65)
result1 = agent.run("Compare the population of Paris and London")
print(f"\nAgent returned: {result1}")

# Test 2: Calculation
print("\n" + "=" * 65)
print("TEST 2: Calculate something")
print("=" * 65)
result2 = agent.run("Calculate 2 ** 10")
print(f"\nAgent returned: {result2}")


# -----------------------------------------------------------------------
# Show the full scratchpad (what the agent was "thinking")
# -----------------------------------------------------------------------

print("\n" + "=" * 65)
print("SCRATCHPAD (Agent's Working Memory)")
print("=" * 65)
print("""
The scratchpad below shows what the agent recorded during its run.
In a real agent, the LLM reads this entire scratchpad before each step.
This is how it knows what has already been done.
""")
print(json.dumps(agent.history, indent=2))


# ==============================================================================
# PART B: Real LLM (Anthropic Claude) -- Requires API Key
# ==============================================================================

print("\n" + "=" * 65)
print("PART B: Real LLM Integration (Anthropic Claude)")
print("=" * 65)

print("""
To use a REAL LLM instead of the FakeLLM:

1. Install the Anthropic SDK:
   pip install anthropic

2. Set your API key:
   Windows: set ANTHROPIC_API_KEY=your-key-here
   Mac/Linux: export ANTHROPIC_API_KEY=your-key-here

3. Get a free API key at: https://console.anthropic.com/

The code below shows how to replace FakeLLM with a real Claude call.
It follows the EXACT same interface -- only the LLM changes.
""")

ANTHROPIC_AVAILABLE = False                         # Set to True if you have a key
try:
    import anthropic                                # Try to import the SDK
    if os.environ.get("ANTHROPIC_API_KEY"):         # Check if API key is set
        ANTHROPIC_AVAILABLE = True
        print("Anthropic SDK found and API key detected. Running real LLM demo...")
    else:
        print("Anthropic SDK found but no ANTHROPIC_API_KEY set. Skipping real LLM demo.")
except ImportError:
    print("anthropic package not installed. Run: pip install anthropic")


if ANTHROPIC_AVAILABLE:
    class ClaudeLLM:
        """
        A real LLM using Anthropic's Claude API.
        Has the SAME interface as FakeLLM -- only the implementation differs.
        This is the Strategy Pattern in C#:
          ILlmStrategy: both FakeLLM and ClaudeLLM implement it.
        """

        def __init__(self):
            self.client = anthropic.Anthropic()     # Create the API client

        def think(self, goal: str, history: list) -> dict:
            """
            Sends the goal + history to Claude and gets a structured JSON response.
            """
            # Build the conversation history for Claude
            # We include all past steps so Claude knows what has been done
            history_text = ""
            for step in history:
                history_text += f"\nStep {step['step']}:\n"
                history_text += f"  Thought: {step['thought']}\n"
                history_text += f"  Tool: {step['tool']}\n"
                history_text += f"  Observation: {step['observation']}\n"

            # The full prompt: instructions + goal + history
            prompt = f"""You are a helpful agent solving a goal step by step.
Available tools: search_web(query), calculator(expression), final_answer(answer)

Goal: {goal}

Steps taken so far:{history_text if history_text else " None yet."}

What is the next step? Respond ONLY with valid JSON:
{{"thought": "your reasoning", "tool": "tool_name", "args": {{"param": "value"}}}}

If you have enough information, use:
{{"thought": "I can answer now.", "tool": "final_answer", "args": {{"answer": "complete answer here"}}}}"""

            # Call Claude
            message = self.client.messages.create(
                model="claude-haiku-4-5-20251001",    # Use a fast, cheap model for demos
                max_tokens=256,                        # Short responses are fine for step decisions
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text.strip()  # Get the text response

            # Parse JSON from Claude's response
            try:
                # Find the JSON block in the response (Claude sometimes adds explanation)
                start = response_text.find("{")         # Find start of JSON
                end = response_text.rfind("}") + 1      # Find end of JSON
                json_str = response_text[start:end]     # Extract just the JSON
                return json.loads(json_str)             # Parse it
            except Exception as e:
                # If parsing fails, return a safe fallback
                return {
                    "thought": f"Parse error: {e}. Raw: {response_text[:100]}",
                    "tool": "final_answer",
                    "args": {"answer": "Could not parse LLM response."}
                }

    # Run with real Claude
    claude_llm = ClaudeLLM()
    real_agent = SimpleAgent(claude_llm, TOOLS, max_steps=5)

    print("\nRunning with real Claude LLM:")
    real_result = real_agent.run("What is the population of Paris?")
    print(f"\nReal Claude returned: {real_result}")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("SUMMARY - Your First Agent")
print("=" * 65)

print("""
WHAT WE BUILT:
  A simple agent with:
  - A brain (FakeLLM or real Claude)
  - Two tools (search_web, calculator)
  - A loop: Think -> Act -> Observe -> repeat until done

KEY PATTERNS:
  1. The agent has a max_steps limit to prevent infinite loops.
  2. The scratchpad records every step. The LLM reads it before deciding next.
  3. "final_answer" tool signals the agent is done.
  4. FakeLLM and ClaudeLLM have the SAME interface -- swap one for the other.

C# ANALOGY:
  The agent is like a BackgroundService that:
  - Injects a brain (ILlmService) and tools (IToolRegistry)
  - Loops until a stop condition (final_answer) is reached
  - Records all steps for audit and debugging

NEXT EXAMPLE (02):
  More tools -- adding: file reader, web scraper, code runner.
  We will see how the agent chooses the RIGHT tool for each situation.
""")

print("=" * 65)
print("END OF EXAMPLE 01")
print("=" * 65)

"""
Example 03: Full ReAct Agent (Reason + Act Loop)
=================================================

GLOSSARY
--------
ReAct:
  Reason + Act. A pattern where the agent alternates between:
  THINKING about what to do and DOING it (calling a tool).
  Each step has: Thought + Action + Observation.

Thought:
  The agent's internal reasoning. "I need X first, then Y."
  The agent explains WHY before it acts.

Action:
  The tool call. "I will call search_web('Python creator')."

Observation:
  What the tool returned. The result of the action.

Scratchpad:
  The growing record of all Thought/Action/Observation triples.
  The agent reads it before deciding the next step.
  Like a detective's notepad -- everything written down.

Multi-Step Task:
  A task that requires several tool calls, where each step's result
  informs what to do next. Cannot be solved in one shot.

Stopping Condition:
  When the agent decides "I have enough info" and calls final_answer.
  Without this, the agent would loop forever.

WHAT THIS EXAMPLE SHOWS
------------------------
Part A: Full ReAct loop with detailed step tracing
Part B: A harder multi-step problem requiring 4+ steps
Part C: Error handling -- what happens when a tool fails

LIBRARIES NEEDED
-----------------
  None (pure Python)
"""

import json      # For pretty-printing the scratchpad

print("=" * 65)
print("EXAMPLE 03: Full ReAct Agent")
print("=" * 65)


# ==============================================================================
# TOOLS (same as Example 02, compact versions)
# ==============================================================================

import math

def calculator(expression: str) -> str:
    """Evaluates a math expression. Returns result as string."""
    try:
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_env)
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"

def search_web(query: str) -> str:
    """Searches for information (simulated)."""
    database = {
        "paris population":         "Paris city: 2.16 million. Metro area: 12 million.",
        "london population":        "London city: 8.9 million. Metro area: 14 million.",
        "python creator":           "Guido van Rossum, released Python in 1991.",
        "speed of light":           "299,792,458 m/s in a vacuum.",
        "mount everest":            "8,848.86 meters above sea level.",
        "microsoft founded":        "1975, by Bill Gates and Paul Allen.",
        "anthropic founded":        "2021, by Dario Amodei and others, including ex-OpenAI team.",
        "gpt-4 release":            "GPT-4 released by OpenAI in March 2023.",
        "transformer paper":        "Attention is All You Need, published 2017 by Vaswani et al. at Google.",
        "richest person 2024":      "Elon Musk or Jeff Bezos depending on market conditions (approx $200B each).",
        "world cup 2022 winner":    "Argentina won the 2022 FIFA World Cup, defeating France on penalties.",
        "python latest version":    "Python 3.12 was released in October 2023. Python 3.13 in October 2024.",
        "numpy creator":            "Travis Oliphant created NumPy (2006), building on Numeric by Jim Hugunin.",
    }

    q = query.lower()
    for key, value in database.items():
        if any(word in q for word in key.split()):
            return value
    return f"No results for: '{query}'"

def get_current_date() -> str:
    """Returns the current date."""
    import datetime
    return datetime.date.today().strftime("%B %d, %Y")   # e.g., "April 30, 2026"

def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Converts between units.
    value:     the number to convert
    from_unit: original unit (meters, km, miles, kg, pounds, celsius, fahrenheit)
    to_unit:   target unit
    Returns:   converted value with unit label
    """
    conversions = {
        ("meters", "feet"):      lambda x: x * 3.28084,
        ("feet", "meters"):      lambda x: x / 3.28084,
        ("km", "miles"):         lambda x: x * 0.621371,
        ("miles", "km"):         lambda x: x / 0.621371,
        ("kg", "pounds"):        lambda x: x * 2.20462,
        ("pounds", "kg"):        lambda x: x / 2.20462,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("meters", "km"):        lambda x: x / 1000,
        ("km", "meters"):        lambda x: x * 1000,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"Cannot convert from {from_unit} to {to_unit}"

TOOLS = {
    "calculator":     calculator,
    "search_web":     search_web,
    "get_date":       lambda: get_current_date(),          # No args needed
    "unit_converter": unit_converter,
}


# ==============================================================================
# PART A: Full ReAct Loop with Detailed Tracing
# ==============================================================================

print("\n" + "=" * 65)
print("PART A: Full ReAct Loop -- Step by Step")
print("=" * 65)

print("""
We will trace through a full ReAct loop for a multi-step problem.
You will see every Thought, Action, and Observation.
This is exactly what happens inside a real LLM agent.
""")


class ReActAgent:
    """
    A full ReAct (Reason + Act) agent.
    For each step, the agent:
      1. THINKS: What do I need to do next?
      2. ACTS:   Calls a tool.
      3. OBSERVES: Records the result.
    Repeats until the goal is achieved or max_steps is reached.

    C# analogy:
      Like a state machine where each state is a (Thought, Action, Observation) triple.
      The state machine runs until reaching a terminal state (final_answer).
    """

    def __init__(self, tools: dict, max_steps: int = 8, verbose: bool = True):
        """
        tools:     dict of tool_name -> function
        max_steps: stop after this many steps
        verbose:   if True, print each step as it happens
        """
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self.scratchpad = []         # Full record: list of step dicts

    def _print_step(self, step_num: int, thought: str, action: str, args: dict, obs: str):
        """Helper to print one step nicely."""
        if not self.verbose:
            return
        print(f"\n  [Step {step_num}]")
        print(f"  THOUGHT:     {thought}")
        print(f"  ACTION:      {action}({args})")
        print(f"  OBSERVATION: {obs[:120]}{'...' if len(obs) > 120 else ''}")

    def run(self, goal: str, script: list) -> str:
        """
        Run the agent with a pre-defined script of steps.
        In a real agent, the LLM generates each step dynamically.
        Here, we use a script so you can see exactly what happens.

        goal:   the user's task
        script: list of {"thought": ..., "tool": ..., "args": ...} dicts
        Returns: final answer
        """
        print(f"\nGOAL: {goal}")
        print("=" * 55)

        self.scratchpad = []    # Clear scratchpad for new goal

        for step_num, step in enumerate(script, start=1):
            if step_num > self.max_steps:
                return f"Max steps ({self.max_steps}) reached."

            thought   = step["thought"]
            tool_name = step["tool"]
            args      = step.get("args", {})

            # Check for final answer
            if tool_name == "final_answer":
                answer = args.get("answer", "")
                if self.verbose:
                    print(f"\n  [Step {step_num}]")
                    print(f"  THOUGHT:  {thought}")
                    print(f"  DONE!     Final answer ready.")
                    print(f"\n  ANSWER: {answer}")
                return answer

            # Call the tool
            if tool_name in self.tools:
                tool_fn = self.tools[tool_name]
                if args:
                    observation = tool_fn(**args)   # Call with arguments
                else:
                    observation = tool_fn()          # Call with no arguments
            else:
                observation = f"Error: Tool '{tool_name}' not found."

            # Record the step
            record = {
                "step": step_num,
                "thought": thought,
                "tool": tool_name,
                "args": args,
                "observation": observation
            }
            self.scratchpad.append(record)

            # Print this step
            self._print_step(step_num, thought, tool_name, args, observation)

        return "Loop ended without final answer."


agent = ReActAgent(TOOLS, max_steps=8, verbose=True)


# Test 1: Population comparison (needs 2 searches + comparison)
print("\n--- TEST 1: Population Comparison ---")
script_1 = [
    {
        "thought": "I need the population of Paris. Let me search for it.",
        "tool": "search_web",
        "args": {"query": "Paris population 2024"}
    },
    {
        "thought": "Got Paris data. Now I need London's population.",
        "tool": "search_web",
        "args": {"query": "London population 2024"}
    },
    {
        "thought": "I have both. London is 8.9M city, Paris is 2.16M city. "
                   "Let me calculate the ratio.",
        "tool": "calculator",
        "args": {"expression": "8.9 / 2.16"}
    },
    {
        "thought": "Ratio is about 4.12. London is ~4x larger. I can answer now.",
        "tool": "final_answer",
        "args": {
            "answer": "London (8.9M) has about 4x the city population of Paris (2.16M). "
                      "Their metro areas are closer: London 14M vs Paris 12M."
        }
    }
]
answer1 = agent.run("Compare the populations of Paris and London", script_1)


# Test 2: Unit conversion chain (needs search + convert + calculate)
print("\n" + "=" * 65)
print("\n--- TEST 2: Unit Conversion Chain ---")
script_2 = [
    {
        "thought": "I need the height of Mount Everest to convert it to feet.",
        "tool": "search_web",
        "args": {"query": "Mount Everest height meters"}
    },
    {
        "thought": "Everest is 8,848.86 meters. Let me convert to feet.",
        "tool": "unit_converter",
        "args": {"value": 8848.86, "from_unit": "meters", "to_unit": "feet"}
    },
    {
        "thought": "Got feet. Now let me also get the result in miles for context.",
        "tool": "unit_converter",
        "args": {"value": 8848.86, "from_unit": "meters", "to_unit": "km"}
    },
    {
        "thought": "I have all three units. I can give a complete answer now.",
        "tool": "final_answer",
        "args": {
            "answer": "Mount Everest is 8,848.86 meters = approximately 29,031 feet = 8.85 km above sea level."
        }
    }
]
answer2 = agent.run("How tall is Mount Everest in feet and km?", script_2)


# ==============================================================================
# PART B: Harder Multi-Step Problem
# ==============================================================================

print("\n" + "=" * 65)
print("PART B: Harder Multi-Step Problem (4+ Steps)")
print("=" * 65)

print("""
Goal: "What year was Python created, and how many years ago was that from today?"

This requires:
1. Search for Python creation year
2. Get today's date
3. Calculate the difference
4. Return a complete answer
""")

script_3 = [
    {
        "thought": "I need to know when Python was created. Searching...",
        "tool": "search_web",
        "args": {"query": "Python creator and first release year"}
    },
    {
        "thought": "Python was first released in 1991. Now I need today's date.",
        "tool": "get_date",
        "args": {}
    },
    {
        "thought": "I have both: Python = 1991, today = 2026. "
                   "Let me calculate how many years ago that was.",
        "tool": "calculator",
        "args": {"expression": "2026 - 1991"}
    },
    {
        "thought": "35 years. I have everything needed to answer.",
        "tool": "final_answer",
        "args": {
            "answer": "Python was created by Guido van Rossum, first released in 1991. "
                      "That was 35 years ago as of 2026."
        }
    }
]

agent3 = ReActAgent(TOOLS, max_steps=8, verbose=True)
answer3 = agent3.run(
    "What year was Python created and how many years ago was that?",
    script_3
)


# ==============================================================================
# PART C: Error Handling
# ==============================================================================

print("\n" + "=" * 65)
print("PART C: Error Handling -- What Happens When a Tool Fails?")
print("=" * 65)

print("""
Real agents encounter errors. Tools fail. Bad inputs happen.
A robust agent recovers from errors and tries a different approach.

This shows how the scratchpad helps with error recovery:
  - The agent calls a tool
  - The tool returns an error
  - The agent reads the error in its scratchpad
  - The agent tries a different approach
""")

class RobustReActAgent(ReActAgent):
    """
    Extended ReAct agent with error detection and recovery.
    """

    def _is_error(self, observation: str) -> bool:
        """Checks if a tool returned an error."""
        return observation.startswith("ERROR") or observation.startswith("Error") or \
               observation.startswith("No results") or "not found" in observation.lower()

    def run_with_recovery(self, goal: str, script: list, recovery_script: list) -> str:
        """
        Run the agent. If any tool returns an error, switch to the recovery script.
        goal:             the user's task
        script:           primary plan
        recovery_script:  fallback plan if something goes wrong
        """
        print(f"\nGOAL (with error handling): {goal}")
        print("=" * 55)

        self.scratchpad = []
        using_recovery = False

        current_script = script
        step_num = 0

        for step in current_script:
            step_num += 1
            thought   = step["thought"]
            tool_name = step["tool"]
            args      = step.get("args", {})

            if tool_name == "final_answer":
                answer = args.get("answer", "")
                print(f"\n  [Step {step_num}] DONE: {answer}")
                return answer

            if tool_name in self.tools:
                tool_fn = self.tools[tool_name]
                observation = tool_fn(**args) if args else tool_fn()
            else:
                observation = f"Error: Tool '{tool_name}' not found."

            print(f"\n  [Step {step_num}]")
            print(f"  THOUGHT:     {thought}")
            print(f"  ACTION:      {tool_name}({args})")
            print(f"  OBSERVATION: {observation}")

            # Check for error -- switch to recovery plan if needed
            if self._is_error(observation) and not using_recovery:
                print(f"\n  *** ERROR DETECTED! Switching to recovery plan... ***")
                using_recovery = True
                current_script = recovery_script
                step_num = 0        # Reset step counter for the recovery plan
                continue            # Start the recovery plan from the beginning

            self.scratchpad.append({
                "step": step_num,
                "thought": thought,
                "tool": tool_name,
                "args": args,
                "observation": observation
            })

        return "Could not complete the task."


# Test error recovery
primary_plan = [
    {
        "thought": "Let me search for this fact.",
        "tool": "search_web",
        "args": {"query": "this query will fail and return no results xyzabc123"}  # Will fail
    },
    {
        "thought": "Got results, will answer.",
        "tool": "final_answer",
        "args": {"answer": "Primary plan answer."}
    }
]

recovery_plan = [
    {
        "thought": "The search failed. Let me try a different approach -- use the calculator.",
        "tool": "calculator",
        "args": {"expression": "42 * 2"}
    },
    {
        "thought": "I switched to a fallback. I'll give a recovery answer.",
        "tool": "final_answer",
        "args": {"answer": "Primary search failed. Recovery: 42 * 2 = 84. (Recovery plan executed successfully.)"}
    }
]

robust_agent = RobustReActAgent(TOOLS, max_steps=10, verbose=True)
recovery_result = robust_agent.run_with_recovery(
    "What is the answer to a failing query?",
    primary_plan,
    recovery_plan
)

print(f"\nFinal result: {recovery_result}")


# ==============================================================================
# SHOW THE FULL SCRATCHPAD
# ==============================================================================

print("\n" + "=" * 65)
print("FINAL SCRATCHPAD (from last run)")
print("=" * 65)
print("""
The scratchpad below is what the LLM reads at each step.
It is the agent's working memory for the current task.
Every Thought + Action + Observation is stored here.
""")
print(json.dumps(robust_agent.scratchpad, indent=2))


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("SUMMARY - ReAct Agent")
print("=" * 65)

print("""
WHAT WE BUILT:
  - Full ReAct agent with Thought/Action/Observation tracing
  - Multi-step problem solving (4+ tool calls)
  - Error detection and recovery

KEY PATTERNS:
  1. ReAct loop: Thought -> Action -> Observation -> repeat
  2. Scratchpad grows with each step; LLM reads it all each time
  3. "final_answer" tool is the stopping condition
  4. Always handle tool errors -- try a recovery plan
  5. Always set max_steps to prevent infinite loops

THE SCRATCHPAD IS KEY:
  Without it, the agent forgets what it has done.
  With it, each step builds on all previous results.
  It is the agent's short-term working memory.

REAL WORLD:
  Real LLMs (Claude, GPT-4) generate the Thought/Action JSON dynamically.
  They read the scratchpad text and decide the next step on their own.
  The loop is identical -- only the LLM call replaces our FakeLLM.

NEXT EXAMPLE (04):
  Memory agent -- the agent REMEMBERS between sessions using:
  - Short-term memory (conversation buffer)
  - Long-term memory (vector database from Module 10)
""")

print("=" * 65)
print("END OF EXAMPLE 03")
print("=" * 65)

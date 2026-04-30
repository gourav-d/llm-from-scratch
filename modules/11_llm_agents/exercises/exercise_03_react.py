"""
Exercise 03: Implement the ReAct Loop
======================================

GOAL
----
Build a full ReAct (Reason + Act) agent from scratch.
Practice writing the Thought -> Action -> Observation loop.

EXERCISES
---------
Exercise 1: Complete the ReactAgent.step() method
Exercise 2: Build a 3-step ReAct sequence for a multi-step question
Exercise 3: Add a "stuck" detection (agent repeating the same action)
Exercise 4: Implement a ReAct agent that uses 4 different tools

HOW TO RUN
----------
  python exercise_03_react.py

LIBRARIES NEEDED
-----------------
  None (pure Python)
"""

import math

print("=" * 65)
print("EXERCISE 03: Implement the ReAct Loop")
print("=" * 65)


# ===========================================================================
# PROVIDED TOOLS
# ===========================================================================

def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    try:
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        return str(eval(expression, {"__builtins__": {}}, safe_env))
    except Exception as e:
        return f"Error: {e}"

def search_web(query: str) -> str:
    """Simulated web search."""
    db = {
        "python release year":   "Python 1.0 released January 1994. Python 3.0 released December 2008.",
        "speed of light":        "Speed of light: 299,792,458 m/s in a vacuum.",
        "earth radius":          "Earth's mean radius: 6,371 km.",
        "population india":      "India population: approximately 1.44 billion (2024). Largest in the world.",
        "population china":      "China population: approximately 1.41 billion (2024).",
        "bitcoin price":         "Bitcoin price varies. As of 2024: roughly $60,000-$70,000 USD.",
        "transformer paper":     "Attention Is All You Need, Vaswani et al., Google, 2017.",
        "founders of google":    "Larry Page and Sergey Brin founded Google in 1998.",
        "founders of microsoft": "Bill Gates and Paul Allen founded Microsoft in 1975.",
    }
    q = query.lower()
    for key, value in db.items():
        if any(word in q for word in key.split()):
            return value
    return f"No search results for: '{query}'"

def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Converts between common units."""
    convs = {
        ("km", "miles"):   lambda x: x * 0.621371,
        ("miles", "km"):   lambda x: x / 0.621371,
        ("kg", "lbs"):     lambda x: x * 2.20462,
        ("lbs", "kg"):     lambda x: x / 2.20462,
        ("m", "ft"):       lambda x: x * 3.28084,
        ("ft", "m"):       lambda x: x / 3.28084,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in convs:
        return f"Cannot convert {from_unit} to {to_unit}"
    result = convs[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"

TOOLS = {
    "calculator":   calculator,
    "search_web":   search_web,
    "unit_convert": unit_convert,
}


# ===========================================================================
# EXERCISE 1: Complete ReactAgent.step()
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 1: Complete ReactAgent.step()")
print("=" * 65)

print("""
TODO: Complete the step() method in ReactAgent.

  step() should:
  1. Take a "decision" dict with keys: "thought", "tool", "args"
  2. Print: "[Step N] THOUGHT: ..."
  3. If tool == "final_answer": return (True, args["answer"])
     (True means "done", the second value is the answer)
  4. Otherwise:
     a. Call the tool from self.tools using args
     b. Print: "         ACTION: tool(args)"
     c. Print: "         OBSERVATION: result"
     d. Add to self.scratchpad: {"step": N, "thought": ..., "tool": ..., "result": ...}
     e. Return (False, result)
     (False means "not done yet")

  Return type: tuple[bool, str]
    bool: True if done, False if not done
    str:  the final answer (if done) or the observation (if not done)
""")


class ReactAgent:
    """
    ReAct agent: Reason + Act.
    Each step: Thought -> Action -> Observation.
    """

    def __init__(self, tools: dict, max_steps: int = 10):
        self.tools = tools
        self.max_steps = max_steps
        self.scratchpad = []
        self._step_count = 0

    def step(self, decision: dict) -> tuple:
        """
        Execute one step: Thought + Action + Observation.
        decision: {"thought": "...", "tool": "...", "args": {...}}
        Returns: (is_done: bool, value: str)
          - (True,  final_answer) if tool == "final_answer"
          - (False, observation)  otherwise

        TODO: Implement this method.
        """
        self._step_count += 1
        n = self._step_count

        thought   = decision.get("thought", "")
        tool_name = decision.get("tool", "")
        args      = decision.get("args", {})

        print(f"\n[Step {n}] THOUGHT: {thought}")

        # TODO: check if tool_name == "final_answer" and return (True, answer) if so
        # TODO: call self.tools[tool_name](**args) to get the observation
        # TODO: print the action and observation
        # TODO: append to self.scratchpad
        # TODO: return (False, observation)

        pass  # DELETE THIS and write your implementation

    def run(self, goal: str, script: list) -> str:
        """
        Run the agent through a pre-defined script.
        Returns the final answer.
        """
        print(f"\nGOAL: {goal}")
        print("=" * 55)

        self.scratchpad = []
        self._step_count = 0

        for decision in script:
            if self._step_count >= self.max_steps:
                return "Max steps reached."

            is_done, value = self.step(decision)

            if is_done:
                print(f"\nFINAL ANSWER: {value}")
                return value

        return "Script ended without final answer."


# Test your implementation
agent = ReactAgent(TOOLS, max_steps=10)

script_test = [
    {
        "thought": "I need to find the speed of light.",
        "tool":    "search_web",
        "args":    {"query": "speed of light"}
    },
    {
        "thought": "Got the speed. Let me convert it to km/s (divide by 1000).",
        "tool":    "calculator",
        "args":    {"expression": "299792458 / 1000"}
    },
    {
        "thought": "Done. Speed of light = 299,792 km/s. I can answer now.",
        "tool":    "final_answer",
        "args":    {"answer": "The speed of light is 299,792,458 m/s or approximately 299,792 km/s."}
    },
]

result = agent.run("What is the speed of light in km/s?", script_test)


# ===========================================================================
# EXERCISE 2: Write a 3-Step Script
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 2: Write Your Own 3-Step Script")
print("=" * 65)

print("""
TODO: Write a 3-step ReAct script to answer:
  "How many miles is the Earth's radius?"

Steps needed:
  Step 1: Search for Earth's radius in km
  Step 2: Convert the radius from km to miles using unit_convert
  Step 3: Provide the final answer

Write the script as a list of dicts:
  [
    {"thought": "...", "tool": "search_web", "args": {"query": "..."}},
    {"thought": "...", "tool": "unit_convert", "args": {"value": ..., "from_unit": "km", "to_unit": "miles"}},
    {"thought": "...", "tool": "final_answer", "args": {"answer": "..."}},
  ]

HINT: Earth's radius is 6,371 km.
""")

# TODO: Write your script here
earth_radius_script = [
    # Step 1: Search
    # {"thought": "...", "tool": "search_web", "args": {"query": "earth radius"}},

    # Step 2: Convert
    # {"thought": "...", "tool": "unit_convert", "args": {"value": 6371.0, "from_unit": "km", "to_unit": "miles"}},

    # Step 3: Answer
    # {"thought": "...", "tool": "final_answer", "args": {"answer": "..."}},
]

if earth_radius_script:
    agent2 = ReactAgent(TOOLS, max_steps=10)
    result2 = agent2.run("How many miles is the Earth's radius?", earth_radius_script)
    print(f"\nResult: {result2}")
else:
    print("TODO: Fill in earth_radius_script above and re-run.")


# ===========================================================================
# EXERCISE 3: Stuck Detection
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 3: Detect When Agent is Stuck")
print("=" * 65)

print("""
A stuck agent calls the SAME tool with the SAME args twice in a row.
This happens when the tool keeps failing and the agent keeps retrying.

TODO: Add a is_stuck() method to ReactAgent that:
  1. Looks at the last 2 entries in self.scratchpad
  2. Returns True if BOTH have the same tool AND same args
  3. Returns False otherwise (or if fewer than 2 steps taken)

Then modify run() to call is_stuck() after each step and stop if stuck.
""")


class ReactAgentV2(ReactAgent):
    """Extended ReactAgent with stuck detection."""

    def is_stuck(self) -> bool:
        """
        TODO: Return True if the agent is repeating the same action.
        Check the last 2 scratchpad entries.
        Two entries are "stuck" if they have identical "tool" AND "args" values.
        """
        pass  # DELETE THIS and write your implementation

    def run(self, goal: str, script: list) -> str:
        """Override run() to add stuck detection."""
        print(f"\nGOAL (with stuck detection): {goal}")
        print("=" * 55)

        self.scratchpad = []
        self._step_count = 0

        for decision in script:
            if self._step_count >= self.max_steps:
                return "Max steps reached."

            is_done, value = self.step(decision)

            if is_done:
                print(f"\nFINAL ANSWER: {value}")
                return value

            # TODO: Check if stuck and break if so
            # HINT: if self.is_stuck(): print("STUCK! Stopping."); break

        return "Script ended without final answer."


# Test stuck detection with a script that repeats the same action
stuck_script = [
    {"thought": "Searching...", "tool": "search_web", "args": {"query": "failing query xyz"}},
    {"thought": "Searching again...", "tool": "search_web", "args": {"query": "failing query xyz"}},  # Same!
    {"thought": "Found it!", "tool": "final_answer", "args": {"answer": "Should not reach here."}},
]

agent_v2 = ReactAgentV2(TOOLS, max_steps=10)
result3 = agent_v2.run("Test stuck detection", stuck_script)
print(f"Result: {result3}")


# ===========================================================================
# EXERCISE 4: 4-Tool ReAct Chain
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 4: 4-Tool ReAct Chain")
print("=" * 65)

print("""
TODO: Write a script for this question:
  "How many miles would you travel if you ran the circumference of the Earth?"

Steps needed:
  1. Search for Earth's radius in km
  2. Calculate circumference: 2 * pi * radius_km  (use calculator)
  3. Convert circumference from km to miles
  4. Final answer

HINT:
  - Earth radius: 6,371 km
  - Circumference = 2 * pi * r  (in Python: 2 * 3.14159 * 6371)
  - You have all 3 tools: search_web, calculator, unit_convert
""")

# TODO: Write your 4-step script here
circumference_script = [
    # Step 1: {"thought": "...", "tool": "search_web", "args": {"query": "earth radius"}},
    # Step 2: {"thought": "...", "tool": "calculator", "args": {"expression": "2 * 3.14159 * 6371"}},
    # Step 3: {"thought": "...", "tool": "unit_convert", "args": {"value": 40030.17, "from_unit": "km", "to_unit": "miles"}},
    # Step 4: {"thought": "...", "tool": "final_answer", "args": {"answer": "..."}},
]

if circumference_script:
    agent4 = ReactAgentV2(TOOLS, max_steps=10)
    result4 = agent4.run("How many miles is Earth's circumference?", circumference_script)
    print(f"\nResult: {result4}")
else:
    print("TODO: Fill in circumference_script above and re-run.")


# ===========================================================================
# SOLUTION
# ===========================================================================

print("\n" + "=" * 65)
print("SOLUTION (uncomment to check)")
print("=" * 65)

"""
SOLUTION FOR EXERCISE 1 -- step() method:

    def step(self, decision: dict) -> tuple:
        self._step_count += 1
        n = self._step_count

        thought   = decision.get("thought", "")
        tool_name = decision.get("tool", "")
        args      = decision.get("args", {})

        print(f"\\n[Step {n}] THOUGHT: {thought}")

        if tool_name == "final_answer":
            return (True, args.get("answer", ""))

        if tool_name not in self.tools:
            observation = f"Error: Tool '{tool_name}' not found."
        else:
            observation = self.tools[tool_name](**args)

        print(f"         ACTION:      {tool_name}({args})")
        print(f"         OBSERVATION: {observation[:100]}")

        self.scratchpad.append({
            "step": n, "thought": thought, "tool": tool_name,
            "args": args, "observation": observation
        })

        return (False, observation)

SOLUTION FOR EXERCISE 2:

    earth_radius_script = [
        {"thought": "I need Earth's radius in km.", "tool": "search_web",
         "args": {"query": "earth radius km"}},
        {"thought": "Earth radius is 6371 km. Converting to miles.",
         "tool": "unit_convert",
         "args": {"value": 6371.0, "from_unit": "km", "to_unit": "miles"}},
        {"thought": "Got miles. Can answer now.",
         "tool": "final_answer",
         "args": {"answer": "Earth's radius is 6,371 km = approximately 3,958.76 miles."}},
    ]

SOLUTION FOR EXERCISE 3 -- is_stuck():

    def is_stuck(self) -> bool:
        if len(self.scratchpad) < 2:
            return False
        last = self.scratchpad[-1]
        prev = self.scratchpad[-2]
        return last["tool"] == prev["tool"] and last["args"] == prev["args"]

SOLUTION FOR EXERCISE 4:

    circumference_script = [
        {"thought": "I need Earth's radius.",
         "tool": "search_web", "args": {"query": "earth radius"}},
        {"thought": "Radius is 6371 km. Calculating circumference: 2 * pi * r.",
         "tool": "calculator", "args": {"expression": "2 * 3.14159265 * 6371"}},
        {"thought": "Circumference is ~40030 km. Converting to miles.",
         "tool": "unit_convert",
         "args": {"value": 40030.17, "from_unit": "km", "to_unit": "miles"}},
        {"thought": "Got miles. I can answer now.",
         "tool": "final_answer",
         "args": {"answer": "Earth's circumference is about 40,030 km or 24,874 miles."}},
    ]
"""

print("See SOLUTION block above.")
print("=" * 65)
print("END OF EXERCISE 03")
print("=" * 65)

"""
Exercise 01: Build a Basic Step Planner Agent
=============================================

GOAL
----
Build an agent from scratch that:
1. Receives a goal (string)
2. Plans steps to achieve it (using a simple planner)
3. Executes each step
4. Returns a final answer

This is about understanding the LOOP that makes an agent work.

EXERCISES
---------
Exercise 1: Complete the StepPlannerAgent class
Exercise 2: Add a new tool (date_tool) and test it
Exercise 3: Add step logging (print each step's result)
Exercise 4: Add a max_steps limit and test what happens when it's hit

HOW TO RUN
----------
  python exercise_01_basic_agent.py

LIBRARIES NEEDED
-----------------
  None (pure Python)
"""

import math
import datetime

print("=" * 65)
print("EXERCISE 01: Basic Step Planner Agent")
print("=" * 65)

# ===========================================================================
# PROVIDED TOOLS (use these in your exercises)
# ===========================================================================

def calculator(expression: str) -> str:
    """Evaluates a math expression. Returns result string."""
    try:
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_env)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_fact(topic: str) -> str:
    """Returns a fact about a topic (simulated web search)."""
    facts = {
        "python":     "Python was created by Guido van Rossum, released 1991.",
        "java":       "Java was created by James Gosling at Sun Microsystems in 1995.",
        "csharp":     "C# was created by Anders Hejlsberg at Microsoft, released 2000.",
        "html":       "HTML was created by Tim Berners-Lee in 1991.",
        "javascript": "JavaScript was created by Brendan Eich at Netscape in 1995.",
    }
    return facts.get(topic.lower(), f"No fact found for '{topic}'.")

# ===========================================================================
# EXERCISE 1: Complete the StepPlannerAgent
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 1: Complete the StepPlannerAgent")
print("=" * 65)

print("""
Complete the TODO sections below.
The agent should:
  1. Loop until it finds a "final_answer" step or hits max_steps
  2. For each step: call the correct tool and record the result
  3. Return the final answer

HINT:
  - Use TOOLS[step["tool"]](**step["args"]) to call the tool
  - Use steps[i]["args"] to get the arguments for step i
""")


class StepPlannerAgent:
    """
    A basic agent that executes a pre-defined list of steps.
    Your job: complete the run() method.
    """

    def __init__(self, tools: dict, max_steps: int = 5):
        """
        tools:     dict of tool_name -> function
        max_steps: stop after this many steps
        """
        self.tools = tools
        self.max_steps = max_steps
        self.log = []    # Record of all steps taken

    def run(self, goal: str, steps: list) -> str:
        """
        Execute the steps to achieve the goal.
        goal:  the user's request
        steps: list of dicts, each with: {"tool": "...", "args": {...}}
               special: {"tool": "final_answer", "args": {"answer": "..."}}

        TODO: Implement this method.

        The method should:
          1. Clear self.log (start fresh)
          2. Loop through steps (up to max_steps)
          3. For each step:
             a. If step["tool"] == "final_answer", return step["args"]["answer"]
             b. Otherwise, look up the tool in self.tools
             c. Call the tool with step["args"] as keyword arguments
             d. Print: f"Step {i}: {tool_name}({args}) -> {result}"
             e. Append to self.log: {"step": i, "tool": tool_name, "result": result}
          4. If we run out of steps, return "No final answer provided."
        """
        self.log = []    # Clear log

        # TODO: Write the loop here
        # HINT: for i, step in enumerate(steps, start=1):
        #           if i > self.max_steps: break  <- safety check
        #           tool_name = step["tool"]
        #           args = step.get("args", {})
        #           if tool_name == "final_answer": return args.get("answer", "")
        #           result = self.tools[tool_name](**args)
        #           print and log the result

        pass  # DELETE THIS LINE when you implement the method

        return "No final answer provided."


# Test your implementation
TOOLS = {
    "calculator": calculator,
    "get_fact":   get_fact,
}

# Test steps -- your agent should execute these and return the final answer
test_steps = [
    {"tool": "get_fact",    "args": {"topic": "python"}},
    {"tool": "calculator",  "args": {"expression": "1991 + 35"}},
    {"tool": "final_answer","args": {"answer": "Python was released in 1991. 35 years later = 2026."}},
]

agent = StepPlannerAgent(TOOLS, max_steps=5)
result = agent.run("When was Python created?", test_steps)
print(f"\nResult: {result}")

# EXPECTED OUTPUT:
# Step 1: get_fact(topic=python) -> Python was created by...
# Step 2: calculator(expression=1991 + 35) -> 2026
# Result: Python was released in 1991. 35 years later = 2026.


# ===========================================================================
# EXERCISE 2: Add a New Tool
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 2: Add a date_tool")
print("=" * 65)

print("""
TODO:
  1. Write a function called date_tool() that returns today's date as a string
     Format: "April 30, 2026"
     HINT: import datetime; datetime.date.today().strftime("%B %d, %Y")

  2. Add "date_tool": date_tool to the TOOLS_EX2 dictionary

  3. Run the provided test steps and check they work
""")

def date_tool() -> str:
    """
    TODO: Return today's date as a formatted string.
    Example return value: "April 30, 2026"
    """
    pass  # DELETE THIS and replace with your implementation


TOOLS_EX2 = {
    "calculator": calculator,
    "get_fact":   get_fact,
    # TODO: Add date_tool here
}

test_steps_ex2 = [
    {"tool": "date_tool",    "args": {}},
    {"tool": "get_fact",     "args": {"topic": "python"}},
    {"tool": "final_answer", "args": {"answer": "Python fact + today's date retrieved."}},
]

agent_ex2 = StepPlannerAgent(TOOLS_EX2, max_steps=5)
result_ex2 = agent_ex2.run("What is today's date and tell me about Python?", test_steps_ex2)
print(f"\nResult: {result_ex2}")


# ===========================================================================
# EXERCISE 3: Step Logging
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 3: Formatted Step Log")
print("=" * 65)

print("""
TODO:
  After running the agent (from Exercise 1), print a nicely formatted log
  of all steps taken. Use the agent.log list.

  The log should look like:
    ============ STEP LOG ============
    Step 1 | Tool: get_fact    | Result: Python was...
    Step 2 | Tool: calculator  | Result: 2026
    ==================================

  HINT: iterate over agent.log (list of dicts with keys: step, tool, result)
""")

# Run the agent first (from Exercise 1, after you implement it)
agent_ex3 = StepPlannerAgent(TOOLS, max_steps=5)
agent_ex3.run("Test for logging", test_steps)

# TODO: Print the formatted log from agent_ex3.log
# HINT:
# print("=" * 35)
# print("STEP LOG")
# print("=" * 35)
# for entry in agent_ex3.log:
#     print(f"Step {entry['step']} | Tool: {entry['tool']:15s} | Result: {entry['result'][:40]}")
# print("=" * 35)


# ===========================================================================
# EXERCISE 4: Max Steps Limit
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 4: Max Steps Safety Limit")
print("=" * 65)

print("""
TODO:
  1. Create a StepPlannerAgent with max_steps=2
  2. Run it with a script that has 5 steps
  3. Verify it stops after 2 steps and does NOT crash

  This tests that your safety limit works correctly.
  Agents can run forever without a max_steps limit!
""")

# 5 steps, but agent should only run 2
long_steps = [
    {"tool": "get_fact",    "args": {"topic": "python"}},
    {"tool": "get_fact",    "args": {"topic": "java"}},
    {"tool": "get_fact",    "args": {"topic": "csharp"}},
    {"tool": "get_fact",    "args": {"topic": "html"}},
    {"tool": "final_answer","args": {"answer": "Got all facts."}},
]

# TODO: Create agent with max_steps=2 and run it
# Expected: only 2 steps executed, then "No final answer provided." returned
# agent_limited = StepPlannerAgent(TOOLS, max_steps=2)
# result_limited = agent_limited.run("Get many facts", long_steps)
# print(f"Result with max_steps=2: {result_limited}")
# print(f"Steps actually taken: {len(agent_limited.log)}")
# Expected output: Steps actually taken: 2


# ===========================================================================
# SOLUTION (uncomment to check your answers)
# ===========================================================================

print("\n" + "=" * 65)
print("SOLUTION (uncomment to check)")
print("=" * 65)

"""
SOLUTION FOR EXERCISE 1 -- run() method:

    def run(self, goal: str, steps: list) -> str:
        self.log = []

        print(f"\\nGOAL: {goal}")
        print("-" * 40)

        for i, step in enumerate(steps, start=1):
            if i > self.max_steps:
                print(f"Max steps ({self.max_steps}) reached!")
                break

            tool_name = step["tool"]
            args = step.get("args", {})

            if tool_name == "final_answer":
                answer = args.get("answer", "")
                print(f"Step {i}: FINAL ANSWER: {answer}")
                return answer

            if tool_name not in self.tools:
                result = f"Error: Tool '{tool_name}' not found."
            else:
                result = self.tools[tool_name](**args)

            print(f"Step {i}: {tool_name}({args}) -> {result[:60]}")
            self.log.append({"step": i, "tool": tool_name, "result": result})

        return "No final answer provided."

SOLUTION FOR EXERCISE 2 -- date_tool():

    def date_tool() -> str:
        return datetime.date.today().strftime("%B %d, %Y")

    TOOLS_EX2 = {
        "calculator": calculator,
        "get_fact":   get_fact,
        "date_tool":  date_tool,
    }

SOLUTION FOR EXERCISE 3 -- log printing:

    print("=" * 35)
    print("STEP LOG")
    print("=" * 35)
    for entry in agent_ex3.log:
        print(f"Step {entry['step']} | Tool: {entry['tool']:15s} | Result: {str(entry['result'])[:40]}")
    print("=" * 35)

SOLUTION FOR EXERCISE 4 -- max_steps limit:

    agent_limited = StepPlannerAgent(TOOLS, max_steps=2)
    result_limited = agent_limited.run("Get many facts", long_steps)
    print(f"Result with max_steps=2: {result_limited}")
    print(f"Steps actually taken: {len(agent_limited.log)}")
"""

print("See SOLUTION block above (uncomment to check your work).")
print("=" * 65)
print("END OF EXERCISE 01")
print("=" * 65)

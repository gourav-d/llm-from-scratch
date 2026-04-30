# Lesson 03: The ReAct Pattern

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain what ReAct stands for and why the name makes sense
2. Trace through a multi-step ReAct loop with a real example
3. Identify the Thought, Action, and Observation parts of each step
4. Explain why ReAct is better than asking the LLM all at once
5. Implement a basic ReAct loop in Python

---

## GLOSSARY

```
ReAct:
  Stands for Reason + Act.
  A pattern where the LLM alternates between THINKING about what to do
  and DOING it (calling a tool). Named after a research paper from 2022.

Thought:
  The LLM's internal reasoning step. It explains WHY it is doing something.
  Example: "I need to find the CEO of Microsoft first, before looking up net worth."
  Thoughts are NOT shown to the user -- they are part of the agent's internal loop.

Action:
  The tool call the LLM makes after thinking.
  Example: search_web("CEO of Microsoft 2024")

Observation:
  The result of the action. What the tool returned.
  Example: "Satya Nadella is the CEO of Microsoft since 2014."

Step:
  One full cycle: Thought + Action + Observation.
  A complex task may need 5, 10, or more steps.

Max Steps (max_iterations):
  A safety limit. The agent stops after N steps even if it has not finished.
  Prevents infinite loops. Always set this when building agents.

Scratchpad:
  The agent's working memory for the current task.
  A growing text/list of all Thoughts + Actions + Observations so far.
  Each new step sees everything that came before.

Stopping Condition:
  The point when the agent decides "I have enough information to answer."
  It outputs a final_answer instead of another tool call.

Chain-of-Thought (CoT):
  From Module 07: making the LLM think step by step in plain text.
  ReAct is CoT PLUS the ability to actually DO things (call tools).
```

---

## Part 1: The Problem with Single-Shot Answers

From Module 07, you learned that Chain-of-Thought (CoT) helps LLMs reason:

```
User:    "A store has 50 apples. 12 are sold. Then 30 more arrive. How many now?"
LLM CoT: "Step 1: Start with 50. Step 2: 50 - 12 = 38. Step 3: 38 + 30 = 68."
Answer:  "68 apples"
```

CoT works for REASONING PROBLEMS.
But what if the LLM needs EXTERNAL INFORMATION?

```
User:  "What is the current price of Apple stock times the number of iPhones sold in 2023?"

LLM without tools:
  Step 1: "I think Apple stock is around $180 (but this might be outdated)."
  Step 2: "I think iPhone sales in 2023 were about 200 million."
  Answer: "$180 * 200M = $36 billion"

Problem: Both numbers might be wrong! The LLM is guessing from training data.
```

ReAct solves this by allowing the LLM to LOOK THINGS UP in real time.

---

## Part 2: What ReAct Looks Like

Let us trace through a ReAct loop for:
"What is the current price of Apple stock times iPhone units sold in 2023?"

```
STEP 1:
  THOUGHT:    "I need two pieces of information: Apple stock price and iPhone 2023 sales.
               Let me start with the stock price."
  ACTION:     search_web("Apple AAPL stock price today")
  OBSERVATION: "Apple (AAPL) stock price: $189.30 (as of market close today)"

STEP 2:
  THOUGHT:    "Good. I have the stock price: $189.30. Now I need iPhone 2023 sales."
  ACTION:     search_web("iPhone units sold 2023 annual")
  OBSERVATION: "Apple sold approximately 234 million iPhones in fiscal year 2023."

STEP 3:
  THOUGHT:    "I have both numbers: $189.30 and 234 million.
               Let me calculate the product."
  ACTION:     calculator("189.30 * 234000000")
  OBSERVATION: "44296200000.0"

STEP 4:
  THOUGHT:    "The calculation is done: ~$44.3 billion.
               I now have all the information I need to answer."
  ACTION:     final_answer(...)
  ANSWER:     "Apple stock at $189.30 x 234 million iPhones sold in 2023
               = approximately $44.3 billion total."
```

Each step has THREE parts:
  - THOUGHT:      Why am I doing this?
  - ACTION:       What am I doing? (tool name + args)
  - OBSERVATION:  What did I get back?

---

## Part 3: Why ReAct is Better Than a Single Call

### Option A: Single LLM Call (No ReAct)
```
You ask: "Calculate Apple stock price * 2023 iPhone sales"
LLM guesses both numbers from training data (possibly outdated or wrong)
Result: An answer that sounds confident but may be completely wrong
```

### Option B: ReAct Loop
```
Step 1: Look up current stock price (live, accurate)
Step 2: Look up 2023 iPhone sales (verified, from news)
Step 3: Calculate with real numbers (precise)
Result: Accurate, traceable, trustworthy
```

### The Key Insight
In ReAct, each OBSERVATION becomes part of the context for the NEXT THOUGHT.
The agent builds understanding incrementally -- like a detective gathering clues.

```
C# analogy: Like a workflow in Temporal or Azure Durable Functions:
  - Each step is a separate activity
  - Steps are executed in sequence
  - Each step's output is passed to the next
  - The workflow can be paused, resumed, or retried
```

---

## Part 4: How to Build a ReAct Loop

The core loop is simple:

```python
def react_agent(goal: str, tools: dict, max_steps: int = 10) -> str:
    """
    A basic ReAct agent loop.
    goal:       the user's question or task
    tools:      dictionary of tool_name -> function
    max_steps:  safety limit to prevent infinite loops
    Returns:    the final answer as a string
    """

    scratchpad = []          # Grows with each step -- all T/A/O recorded here
    step_count = 0           # How many steps we have taken

    while step_count < max_steps:                   # Safety limit
        step_count += 1

        # ------- THINK -------
        # Ask the LLM: "Given the goal and everything so far, what is next?"
        # The LLM reads the scratchpad and decides the next action
        thought, action_name, action_args = llm_think(goal, scratchpad)

        # Record the thought
        scratchpad.append({"role": "thought", "content": thought})

        # ------- CHECK FOR DONE -------
        if action_name == "final_answer":
            return action_args.get("answer", "")    # Agent is done!

        # ------- ACT -------
        if action_name not in tools:
            observation = f"Error: Tool '{action_name}' not found."
        else:
            observation = tools[action_name](**action_args)   # Call the real tool

        # ------- OBSERVE -------
        # Record the action and its result in the scratchpad
        scratchpad.append({
            "role": "action",
            "tool": action_name,
            "args": action_args,
            "result": observation
        })

    # If we hit max_steps without finishing, return what we have
    return "Max steps reached. Partial answer based on gathered information."
```

---

## Part 5: The Scratchpad in Detail

The scratchpad is the agent's "working memory" for the current task.
It grows with each step and the LLM reads ALL of it before each decision.

After 2 steps, the scratchpad looks like this:
```
[
  {
    "role": "thought",
    "content": "I need the Apple stock price first."
  },
  {
    "role": "action",
    "tool": "search_web",
    "args": {"query": "Apple AAPL stock price today"},
    "result": "Apple (AAPL) stock price: $189.30"
  },
  {
    "role": "thought",
    "content": "Got stock price. Now I need iPhone 2023 sales."
  },
  {
    "role": "action",
    "tool": "search_web",
    "args": {"query": "iPhone units sold 2023"},
    "result": "Apple sold 234 million iPhones in fiscal 2023."
  }
]
```

When the LLM is asked for the NEXT step, it sees all of this.
It knows what it has already tried and what it already knows.
This prevents the agent from repeating itself.

---

## Part 6: The LLM Prompt for ReAct

Here is what the prompt to the LLM looks like at each step:

```
SYSTEM:
  You are a helpful assistant that solves problems step by step.
  You have access to these tools: [tool list with descriptions]

  For each step, respond with JSON in EXACTLY this format:
  {
    "thought": "Why I am doing this step",
    "tool":    "tool_name",
    "args":    {"param": "value"}
  }

  When you have enough information to answer, respond with:
  {
    "thought": "I now have all the information needed.",
    "tool":    "final_answer",
    "args":    {"answer": "Your complete answer here."}
  }

USER:
  Goal: What is Apple stock price * iPhone 2023 sales?

  Steps taken so far:
  THOUGHT: I need the Apple stock price first.
  ACTION: search_web({"query": "Apple AAPL stock price today"})
  OBSERVATION: Apple (AAPL) stock price: $189.30

  THOUGHT: Got stock price. Now I need iPhone 2023 sales.
  ACTION: search_web({"query": "iPhone units sold 2023"})
  OBSERVATION: Apple sold 234 million iPhones in fiscal 2023.

  What should I do next?
```

The LLM reads this and responds with the next JSON step.

---

## Part 7: Stopping Conditions and Safety

### Always Set a Max Steps Limit
```python
MAX_STEPS = 10   # Agent stops after 10 steps even if not done

# Why? Without this, the agent can loop forever if:
# - A tool keeps failing
# - The LLM gets confused and repeats steps
# - The goal is impossible
```

### Watch for Signs of Looping
```python
# Check if the last two actions are identical (agent is stuck)
if len(scratchpad) >= 4:
    last_action = scratchpad[-1]
    prev_action = scratchpad[-3]
    if last_action == prev_action:
        return "Agent appears stuck. Stopping."
```

### Log Every Step (for debugging)
```python
print(f"Step {step_count}: THOUGHT: {thought}")
print(f"Step {step_count}: ACTION: {action_name}({action_args})")
print(f"Step {step_count}: OBSERVATION: {observation[:100]}...")  # First 100 chars
```

---

## Part 8: Comparison to What You Have Learned

| Concept       | Module | What it does                    | ReAct relation             |
|---------------|--------|---------------------------------|----------------------------|
| CoT           | 07     | Think step by step              | ReAct adds TOOL CALLS      |
| Transformer   | 04     | Predicts next token             | The LLM inside the agent   |
| Embeddings    | 05     | Encode meaning as numbers       | Used in memory (Lesson 04) |
| RAG           | 10     | Retrieve relevant documents     | One of the agent's tools   |
| Prompt Eng.   | 08     | Write effective prompts         | Critical for ReAct prompts |

ReAct builds on EVERYTHING you have learned so far.

---

## Key Takeaways

1. ReAct = Reason (think) + Act (call tools). Each step: Thought + Action + Observation.

2. The scratchpad grows with each step. The LLM reads it all before deciding the next step.

3. ReAct > single LLM call because it uses REAL data from tools instead of guessing.

4. Always set max_steps to prevent infinite loops.

5. The LLM writes JSON (thought + tool + args). Your code runs the tool. Result goes back to LLM.

6. ReAct is CoT (from Module 07) + tool calls. If you understand CoT, you understand ReAct.

---

## Next

Lesson 04: Memory and State
  - How does the agent remember things between conversations?
  - Short-term memory (context window) vs long-term memory (vector database)
  - How to store and retrieve memories efficiently

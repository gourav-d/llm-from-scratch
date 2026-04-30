# Lesson 01: What Are LLM Agents?

## Learning Objectives

By the end of this lesson, you will be able to:
1. Define what an LLM agent is in plain English
2. Explain the difference between a chatbot and an agent
3. Name the four core components of an agent
4. Describe the agent loop (the cycle of thinking and acting)
5. Give two real-world examples of agents

---

## GLOSSARY

Read this section BEFORE the rest of the lesson.

```
Agent:
  An LLM that can plan, decide, and take actions to achieve a goal.
  It does not just answer questions -- it actively does things.
  Example: "Research the top 3 Python frameworks" -> searches the web,
           reads pages, compares them, writes a report. All automatically.

Chatbot:
  An LLM that only generates text replies. It cannot take actions.
  Example: You ask "What is Python?" -> it answers from its training data.
  It cannot search the web, run code, or call any external service.

Tool:
  A function the agent is allowed to call.
  Examples: calculator(), search_web(), read_file(), send_email()
  The agent DECIDES which tool to use and with what inputs.

Planning:
  The agent's ability to break a big goal into smaller steps.
  "To answer this question, I first need to X, then Y, then Z."
  This is the agent THINKING before it acts.

Action:
  Something the agent actually does (calls a tool, generates text).
  As opposed to just thinking.

Observation:
  The result of an action. The agent SEES what happened and uses
  that information to plan the next step.

Agent Loop:
  The cycle: Think -> Act -> Observe -> Think -> Act -> Observe -> ...
  The agent keeps looping until the goal is achieved.

LLM (Large Language Model):
  The "brain" inside the agent. It does the thinking, planning, and
  reasoning. The agent is the LLM PLUS the ability to use tools.

Autonomy:
  How much the agent can do without asking the user for help.
  A fully autonomous agent does everything on its own.
  A semi-autonomous agent checks with the user at key steps.
```

---

## Part 1: The Problem with Plain Chatbots

You have learned about transformers (Module 04) and how GPT works (Module 05).
A plain chatbot is just a text-in, text-out system:

```
User Input    ->    LLM    ->    Text Output
"What is        (brain)       "Python is a
 Python?"                      programming..."
```

This works great for:
  - Answering factual questions (if the LLM knows the answer)
  - Writing, summarizing, translating text
  - Explaining code

But it FAILS for:
  - Questions that require up-to-date information ("What happened in the news today?")
  - Tasks that require doing things ("Send an email to John")
  - Multi-step problems ("Research, compare, decide, and book the cheapest flight")
  - Long tasks that exceed the LLM's context window

The LLM is like a brilliant professor locked in a room with no phone.
They can answer questions brilliantly -- but only from memory.
They cannot look anything up, send messages, or interact with the outside world.

---

## Part 2: Enter the Agent

An agent gives the LLM access to TOOLS (the ability to do things).

```
                        +----------------+
                        |   TOOLS        |
                        |                |
                        |  calculator()  |
                        |  search_web()  |
                        |  read_file()   |
                        |  send_email()  |
                        +----------------+
                               ^  |
                               |  | (results)
                               |  v
User Input    ->    LLM    ->  Decision: "I need to search the web"
"What is the   (brain)         -> Calls: search_web("cheapest Python course")
 cheapest                      -> Gets: [list of results]
 Python                        -> Thinks: "I now have the info"
 course?"                      -> Outputs: "The cheapest course is..."
```

The LLM still does the THINKING.
The tools do the DOING.
The agent is the LLM + tools + the loop that connects them.

---

## Part 3: The Four Core Components

Every agent has these four parts:

```
+----------------------------------------------------------+
|                        AGENT                             |
|                                                          |
|  1. BRAIN (LLM)                                         |
|     The language model that does reasoning.              |
|     Reads the goal, plans steps, decides what to call.   |
|                                                          |
|  2. TOOLS                                               |
|     Functions the LLM can call.                          |
|     Calculator, web search, database, file system, APIs. |
|                                                          |
|  3. MEMORY                                              |
|     What the agent knows and remembers.                  |
|     Short-term: current conversation (context window).   |
|     Long-term: vector database (persists forever).       |
|                                                          |
|  4. PLANNER / LOOP                                       |
|     The engine that drives the cycle:                    |
|     Think -> Act -> Observe -> Think -> Act -> ...       |
+----------------------------------------------------------+
```

### C# Analogy

Think of an agent like a microservices application:

```csharp
// Plain chatbot: just one method
string Answer(string question) {
    return model.GenerateText(question);   // That's it
}

// Agent: orchestrates multiple services
string SolveGoal(string goal) {
    var plan = model.Plan(goal);           // Brain plans steps
    foreach (var step in plan) {
        if (step.NeedsTool) {
            var result = tools[step.Tool].Execute(step.Args);  // Tools act
            memory.Store(result);           // Memory stores result
            plan = model.UpdatePlan(memory.GetAll());          // Brain re-plans
        }
    }
    return model.GenerateFinalAnswer(memory.GetAll());
}
```

---

## Part 4: The Agent Loop

The agent runs in a loop until it reaches an answer:

```
GOAL: "What is the population of Paris and how does it compare to London?"

LOOP ITERATION 1:
  THINK:    "I need the population of Paris. I'll search for it."
  ACT:      search_web("population of Paris 2024")
  OBSERVE:  "Paris population: approximately 2.16 million (city), 12 million (metro)"

LOOP ITERATION 2:
  THINK:    "I have Paris data. Now I need London's population."
  ACT:      search_web("population of London 2024")
  OBSERVE:  "London population: approximately 8.9 million (city), 14 million (metro)"

LOOP ITERATION 3:
  THINK:    "I have both numbers. I can now compare them and answer."
  ACT:      (no tool needed -- just generate text)
  OBSERVE:  "I have enough information"

FINAL ANSWER:
  "Paris has a city population of about 2.16 million (12M metro area),
   while London has about 8.9 million (14M metro area).
   London's city population is roughly 4x larger than Paris's city proper,
   though both metro areas are similar in size."
```

---

## Part 5: Types of Agents

### Simple Agent (Single Step)
  - Gets a question
  - Calls ONE tool
  - Returns the answer

```
User: "What is 2^32?"
Agent: calls calculator(2**32) -> 4294967296
Agent: "2^32 = 4,294,967,296"
```

### Multi-Step Agent (ReAct)
  - Gets a complex goal
  - Breaks into steps
  - Calls multiple tools in sequence
  - Each result informs the next step

```
User: "Find the CEO of Microsoft and his net worth"
Step 1: search("CEO of Microsoft") -> "Satya Nadella"
Step 2: search("Satya Nadella net worth") -> "$1 billion approx"
Answer: "Satya Nadella is CEO of Microsoft. Net worth: ~$1B"
```

### Autonomous Agent (Long-Running)
  - Given a complex project goal
  - Works for minutes, hours, or longer
  - Makes hundreds of decisions
  - May check in with humans at key points
  - Examples: Devin (AI programmer), AutoGPT

### Multi-Agent System
  - Multiple specialized agents working together
  - Orchestrator agent delegates to specialists
  - Example: Research agent -> Writer agent -> Editor agent
  - Each specialist is optimized for its role

---

## Part 6: Real-World Examples

### GitHub Copilot (Code Assistant Agent)
```
Goal: "Add a unit test for the Login function"

Step 1: read_file("login.cs")                    <- Reads your code
Step 2: analyze(code)                            <- Understands the function
Step 3: search("C# unit test best practices")    <- Looks up patterns (if needed)
Step 4: generate_test(login_code, patterns)      <- Writes the test
Step 5: run_tests()                              <- Verifies it passes
```

### Customer Support Agent
```
Goal: "User says their order hasn't arrived after 2 weeks"

Step 1: lookup_order(user_id="12345")            <- Finds their order
Step 2: check_shipping_status(order_id="9876")   <- Checks tracking
Step 3: If delayed > 10 days: escalate_to_refund() <- Takes action
Step 4: send_email(user, "Your refund has been processed")
```

### C# .NET Analogy: Background Worker Service
```
An agent is like a BackgroundService in .NET that:
  - Has a goal (the work item)
  - Has dependencies (tools = injected services via DI)
  - Loops until the goal is done (the agent loop)
  - Can be cancelled or paused (human-in-the-loop)

public class AgentService : BackgroundService {
    private readonly IToolRegistry _tools;
    private readonly ILlmBrain _brain;
    private readonly IMemory _memory;

    protected override async Task ExecuteAsync(CancellationToken ct) {
        while (!goalAchieved && !ct.IsCancellationRequested) {
            var action = await _brain.PlanNextStep(_memory.GetHistory());
            var result = await _tools.Execute(action);
            _memory.Store(result);
            goalAchieved = _brain.IsGoalComplete(_memory.GetHistory());
        }
    }
}
```

---

## Part 7: Risks and Limitations

Agents are powerful but have real limitations:

```
RISK 1: Infinite loops
  The agent keeps calling tools and never finishes.
  Fix: Set a maximum number of steps (e.g., max_iterations=10).

RISK 2: Tool misuse
  The agent calls the wrong tool or with wrong arguments.
  Fix: Clear tool descriptions + validation of arguments.

RISK 3: Accumulated errors
  An error in step 3 causes wrong results in steps 4, 5, 6...
  Fix: Error handling in each tool + agent checks results.

RISK 4: Cost (real APIs)
  Each LLM call costs money. A long agent loop can be expensive.
  Fix: Limit steps, use cheaper models for simple sub-tasks.

RISK 5: Trust and safety
  What if the agent calls send_email() or delete_file() incorrectly?
  Fix: Human-in-the-loop for dangerous actions + confirmation prompts.
```

---

## Key Takeaways

1. A chatbot just answers. An agent PLANS and ACTS.

2. Every agent has 4 parts: Brain (LLM), Tools, Memory, and Loop.

3. The agent loop: Think -> Act -> Observe -> (repeat) -> Final Answer.

4. Agents can be simple (1 tool call) or complex (hundreds of steps).

5. Real risks: infinite loops, tool misuse, accumulated errors, cost.

6. Always limit the number of steps and validate tool calls.

---

## Next

Lesson 02: Tool Use and Function Calling
  - How does the LLM decide WHICH tool to call?
  - How does it pass the right arguments?
  - How does the tool result get back to the LLM?
  - How do we define our own tools?

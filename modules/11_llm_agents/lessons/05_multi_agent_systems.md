# Lesson 05: Multi-Agent Systems

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain when a single agent is not enough
2. Describe the Orchestrator-Worker pattern
3. Name three real-world multi-agent architectures
4. Explain agent communication (how agents talk to each other)
5. List the tradeoffs of single-agent vs multi-agent systems

---

## GLOSSARY

```
Orchestrator Agent:
  The "manager" agent. Receives a big goal, breaks it into sub-tasks,
  and delegates each sub-task to a specialist agent.
  It collects the results and produces the final output.
  Like a project manager: doesn't do the work, directs who does.

Specialist (Worker) Agent:
  A focused agent built for ONE specific task.
  Examples: ResearchAgent, WriterAgent, CodeAgent, ReviewerAgent.
  Simpler and more reliable than a do-everything generalist agent.
  Like a software engineer: does one thing very well.

Agent Communication:
  How agents pass information between each other.
  Usually via structured messages (JSON) or shared state (a shared memory/queue).

Shared Memory:
  A common data store that multiple agents can read and write.
  Agents use it to communicate results without talking directly.
  Like a shared database table or a message queue (RabbitMQ, Azure Service Bus).

Message Passing:
  Agents send structured messages directly to each other.
  Orchestrator sends: {"task": "research", "topic": "Python frameworks", "for_agent": "ResearchAgent"}
  Worker sends back: {"result": "Top frameworks: Django, FastAPI, Flask...", "agent": "ResearchAgent"}

Parallel Agents:
  Multiple specialist agents working simultaneously on independent sub-tasks.
  Like running multiple Tasks in parallel in C#.

Sequential Agents:
  Agents that work one after another. Output of Agent 1 = input of Agent 2.
  Like a pipeline: Research -> Write -> Edit -> Review.

Handoff:
  When one agent finishes and passes its output to the next agent.
  The next agent picks up from where the previous one left off.
```

---

## Part 1: When One Agent Is Not Enough

A single agent handling a complex task has problems:

### Problem 1: Context Window Overload
```
Task: "Analyze 50 research papers and write a comprehensive report"

Single agent:
  - Reads 50 papers into context -> 500,000 tokens needed
  - Most LLMs cap at 200,000 tokens
  - FAILS: too much data to process at once

Multi-agent solution:
  - 50 specialist agents, each reads 1 paper
  - Each writes a 1-paragraph summary
  - Orchestrator collects 50 summaries (small!) and writes the report
  - Total tokens per agent call: reasonable
```

### Problem 2: Conflicting Roles
```
Task: "Write code, then review the same code for bugs"

Single agent:
  - Writes code (tends to be optimistic about its own work)
  - Reviews its own code (hard to spot your own mistakes)
  - Result: may miss bugs it wrote itself

Multi-agent solution:
  - CodeAgent: writes the code
  - ReviewAgent: reviews the code (fresh perspective, no attachment to it)
  - Result: better quality, like a real code review
```

### Problem 3: Specialization vs Generalization
```
Task: "Research Python, translate the summary to Spanish, post it to Twitter"

A single general agent:
  - OK at all three tasks
  - Not excellent at any

Three specialist agents:
  - ResearchAgent: excellent at research (tuned for accuracy)
  - TranslationAgent: excellent at translation (tuned for fluency)
  - SocialMediaAgent: excellent at writing posts (tuned for engagement)
  - Each can use different models and tools optimized for its role
```

---

## Part 2: The Orchestrator-Worker Pattern

This is the most common multi-agent pattern.

```
              +--------------------+
              |   USER / GOAL      |
              +--------------------+
                        |
                        v
              +--------------------+
              |   ORCHESTRATOR     |  <- "The Manager"
              |   AGENT            |
              |   - Reads the goal |
              |   - Plans steps    |
              |   - Delegates work |
              |   - Collects output|
              +--------------------+
               /        |         \
              /         |          \
             v          v           v
    +----------+  +----------+  +----------+
    | RESEARCH |  | WRITER   |  | REVIEWER |
    | AGENT    |  | AGENT    |  | AGENT    |
    | - Search |  | - Draft  |  | - Check  |
    | - Verify |  | - Format |  | - Polish |
    +----------+  +----------+  +----------+
```

### How the Orchestrator Works

```python
# Simplified orchestrator logic (pseudocode)
def orchestrator_agent(user_goal: str) -> str:
    """
    Receives a big goal and delegates it to specialists.
    """
    # Step 1: Plan -- what sub-tasks do we need?
    plan = llm_plan(f"""
        Goal: {user_goal}
        Available agents: ResearchAgent, WriterAgent, ReviewerAgent
        Break this goal into sub-tasks, one per agent.
    """)
    # plan = [{"task": "research Python frameworks", "agent": "ResearchAgent"},
    #         {"task": "write article draft", "agent": "WriterAgent"},
    #         {"task": "review and polish draft", "agent": "ReviewerAgent"}]

    # Step 2: Execute each sub-task
    results = {}
    for step in plan:
        agent_name = step["agent"]         # Which specialist?
        task = step["task"]               # What to do?
        context = results                  # What does it need to know from prior steps?

        # Call the specialist agent with the task + prior results
        result = AGENT_REGISTRY[agent_name].run(task, context)
        results[step["task"]] = result     # Store result for next agents

    # Step 3: Combine results into final output
    final = llm_combine(user_goal, results)
    return final
```

---

## Part 3: Three Multi-Agent Architectures

### Architecture 1: Pipeline (Sequential)
```
Agents work in sequence. Each agent's output feeds into the next.

Research -> Write -> Edit -> Translate -> Publish

Pro:  Simple. Easy to debug. Each step is clear.
Con:  Slow (one at a time). If one agent fails, the whole pipeline stops.

C# analogy: Method chaining or IEnumerable.Select() pipeline
  results.Research().Write().Edit().Translate().Publish()
```

### Architecture 2: Map-Reduce (Parallel + Combine)
```
Orchestrator splits work into N parallel jobs.
N agents work simultaneously.
Orchestrator collects all results and combines them.

                 Orchestrator
                /     |      \
          Agent1   Agent2   Agent3    <- Work in PARALLEL
          (Part1)  (Part2)  (Part3)
                \     |      /
                 Orchestrator          <- Combine results

Pro:  Fast (parallel). Scales to large tasks.
Con:  Combining results can be tricky. Orchestrator has complex logic.

C# analogy: Task.WhenAll() + aggregation:
  var tasks = parts.Select(p => Task.Run(() => agent.Process(p)));
  var results = await Task.WhenAll(tasks);
  return combiner.Merge(results);
```

### Architecture 3: Mixture of Experts (Router)
```
A router agent decides which specialist is best for this specific question.

User question:
  "Translate this to French" -> Router -> TranslationAgent
  "Fix this bug"             -> Router -> CodeAgent
  "What is 2+2?"             -> Router -> MathAgent

Pro:  Each specialist is very good at its task. Fast (no unnecessary steps).
Con:  Router can make wrong decisions. Need to handle routing errors.

C# analogy: A factory pattern or strategy pattern:
  IAgentStrategy agent = _agentFactory.GetBestAgent(userQuery);
  return agent.Process(userQuery);
```

---

## Part 4: Agent Communication

How do agents pass information to each other?

### Option A: Direct Message Passing
```python
# Orchestrator calls a specialist directly and waits for the result
research_result = research_agent.run(
    task="Find top Python web frameworks",
    context={}
)
# research_result = "Django, FastAPI, Flask, Tornado..."

writer_result = writer_agent.run(
    task="Write a comparison article",
    context={"research": research_result}   # Pass prior result as context
)
```

### Option B: Shared Blackboard (Shared State)
```python
# All agents read from and write to a shared dictionary
blackboard = {}                              # Shared state

# Research agent writes its findings
research_agent.run(task="research", state=blackboard)
# -> blackboard["research"] = "Django, FastAPI, Flask..."

# Writer agent reads the research and writes a draft
writer_agent.run(task="write", state=blackboard)
# -> blackboard["draft"] = "Python Web Frameworks: A Comparison..."

# Reviewer agent reads the draft and writes feedback
reviewer_agent.run(task="review", state=blackboard)
# -> blackboard["feedback"] = "Good structure. Add code examples."
```

### Option C: Message Queue (Decoupled)
```python
# Agents communicate through a queue (like Azure Service Bus or RabbitMQ)
# More complex but scales to many agents

queue.publish("research_task", {"query": "Python frameworks"})
# ResearchAgent picks it up, does the work, publishes results
queue.publish("research_done", {"result": "Django, FastAPI..."})
# WriterAgent is subscribed, picks up the result, writes the article
```

---

## Part 5: Real-World Multi-Agent Systems

### System 1: Devin (AI Software Engineer)
```
Goal: "Build a REST API for user authentication"

Orchestrator Agent: Plans the whole project
  |
  v
ArchitectAgent: Designs the API structure (routes, models, database)
  |
  v
CodeAgent: Writes the code (Python FastAPI + SQLAlchemy)
  |
  v
TestAgent: Writes unit and integration tests
  |
  v
ReviewAgent: Reviews code for security issues (SQL injection, auth bugs)
  |
  v
DocumentationAgent: Writes README and API docs
```

### System 2: Customer Support Pipeline
```
User complaint arrives
  |
  v
ClassifierAgent: "Is this a billing, technical, or shipping issue?"
  |
  v
[Billing? -> BillingAgent] [Technical? -> TechAgent] [Shipping? -> ShippingAgent]
  |
  v
ResolutionAgent: Crafts the response using the specialist's findings
  |
  v
QualityAgent: Checks response for tone, accuracy, policy compliance
  |
  v
Send response to user
```

---

## Part 6: Multi-Agent vs Single Agent

| Factor              | Single Agent          | Multi-Agent                    |
|---------------------|-----------------------|--------------------------------|
| Complexity          | Simple to build       | Complex to coordinate          |
| Best for            | Simple, focused tasks | Large, complex, parallel tasks |
| Context window use  | All in one window     | Distributed across agents      |
| Debugging           | Easy (one trace)      | Harder (multiple traces)       |
| Cost                | Lower (one LLM call)  | Higher (many LLM calls)        |
| Speed               | Sequential            | Can be parallel                |
| Specialization      | General purpose       | Each agent is an expert        |
| Failure handling    | One failure = done    | Other agents can continue      |

**Rule of thumb:** Start with a single agent. Add more agents when you have a specific problem
that a specialist would solve better.

---

## Key Takeaways

1. Multi-agent systems are needed for: tasks too large for one context window,
   tasks needing conflicting roles (write AND review), or tasks needing true specialization.

2. Orchestrator pattern: one manager agent + many specialist worker agents.

3. Three architectures: Pipeline (sequential), Map-Reduce (parallel+combine), Router (expert selection).

4. Agents communicate via: direct calls, shared state (blackboard), or message queues.

5. Start simple: single agent first. Add agents when you hit real limitations.

6. Cost warning: multi-agent = many LLM calls = higher API cost. Design carefully.

---

## Module 11 Summary

You now know all the core concepts:

```
Lesson 01: What agents are (brain + tools + memory + loop)
Lesson 02: How tool calling works (tool definitions, the flow, JSON responses)
Lesson 03: ReAct pattern (Thought -> Action -> Observation -> repeat)
Lesson 04: Memory (short-term buffer vs long-term vector DB)
Lesson 05: Multi-agent systems (orchestrator, workers, communication)
```

Next: Work through the examples to see all of this in running Python code!

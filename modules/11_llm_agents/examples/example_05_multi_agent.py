"""
Example 05: Multi-Agent System -- Orchestrator + Specialists
=============================================================

GLOSSARY
--------
Orchestrator:
  The manager agent. Receives a big goal, breaks it into sub-tasks,
  and assigns each sub-task to the right specialist agent.
  Collects all results and produces the final output.

Specialist Agent:
  A focused agent designed for ONE type of task.
  Examples: ResearchAgent, WriterAgent, MathAgent, SummaryAgent.
  Simpler, more reliable, and more accurate than a do-everything agent.

Pipeline:
  Agents working in sequence. Output of A goes into B, B into C.
  Like a factory assembly line.

Parallel Execution:
  Multiple agents working at the same time on independent sub-tasks.
  Like running multiple Tasks simultaneously in C#.

Blackboard:
  A shared dictionary where all agents read and write their results.
  Each agent "publishes" its output for other agents to use.
  In C# : like a shared Dictionary<string, object> passed by reference.

Agent Message:
  A structured dict passed between agents:
  {"task": "...", "context": {...}, "for_agent": "..."}

Routing:
  Deciding WHICH specialist agent handles a given task.
  The orchestrator does this based on task type.

WHAT THIS EXAMPLE SHOWS
------------------------
Part A: Simple pipeline -- three agents in sequence (Research -> Write -> Review)
Part B: Parallel execution -- multiple agents at once
Part C: Orchestrator with dynamic routing

LIBRARIES NEEDED
-----------------
  None (pure Python)
  concurrent.futures (standard library -- for parallel execution in Part B)
"""

import json                      # For pretty-printing
import concurrent.futures        # For running agents in parallel (standard library)

print("=" * 65)
print("EXAMPLE 05: Multi-Agent System")
print("=" * 65)


# ==============================================================================
# BASE AGENT CLASS
# ==============================================================================

class BaseAgent:
    """
    All specialist agents inherit from this base class.
    They share a common interface: agent.run(task, context) -> result

    C# analogy: like an abstract base class or IAgent interface:
      public interface IAgent {
          string Run(string task, Dictionary<string, string> context);
      }
    """

    def __init__(self, name: str):
        """name: the agent's display name (e.g., "ResearchAgent")"""
        self.name = name

    def run(self, task: str, context: dict = None) -> str:
        """
        Execute the task and return a string result.
        task:    what the agent should do
        context: results from previous agents (shared blackboard data)
        Returns: this agent's output as a string
        """
        raise NotImplementedError("Subclasses must implement run()")  # Must override

    def __str__(self):
        """Returns the agent's name when printed."""
        return self.name


# ==============================================================================
# SPECIALIST AGENTS
# ==============================================================================

class ResearchAgent(BaseAgent):
    """
    Specialist: researches facts on a topic.
    Like a research assistant -- finds information, does not write articles.
    """

    def __init__(self):
        super().__init__("ResearchAgent")   # Call parent constructor

        # Simulated knowledge base (in production: web search + vector DB)
        self.knowledge = {
            "python":    "Python is a high-level, interpreted programming language created by "
                         "Guido van Rossum in 1991. Known for: readable syntax, large ecosystem, "
                         "widely used in AI/ML, web dev (Django/FastAPI), data science (Pandas/NumPy).",
            "fastapi":   "FastAPI is a modern Python web framework for building APIs quickly. "
                         "Key features: automatic OpenAPI docs, data validation with Pydantic, "
                         "async support, very fast (comparable to Node.js). Released 2018.",
            "django":    "Django is a full-featured Python web framework. Follows 'batteries included' "
                         "philosophy. Has built-in admin panel, ORM, auth system. Good for complex apps.",
            "flask":     "Flask is a lightweight Python web micro-framework. Minimal, flexible, easy "
                         "to start with. No ORM or admin panel included -- you add what you need.",
            "agents":    "LLM Agents are AI systems that can plan and take actions to achieve goals. "
                         "They use tools (search, calculator, APIs), memory, and reasoning loops (ReAct). "
                         "Examples: GitHub Copilot, Devin, AutoGPT, Claude Computer Use.",
            "llm":       "Large Language Models (LLMs) are neural networks trained on massive text datasets. "
                         "They learn to predict the next token. Examples: GPT-4, Claude, Gemini, Llama. "
                         "Used for: chat, code generation, summarization, reasoning.",
        }

    def run(self, task: str, context: dict = None) -> str:
        """Research a topic and return structured findings."""
        context = context or {}    # Default to empty dict if None
        task_lower = task.lower()

        print(f"\n  [{self.name}] Researching: {task}")

        # Find matching knowledge
        found_facts = []
        for topic, facts in self.knowledge.items():
            if topic in task_lower:
                found_facts.append(f"[{topic.upper()}] {facts}")

        if found_facts:
            result = "\n".join(found_facts)
        else:
            result = f"Research findings for '{task}': No specific data found in knowledge base."

        print(f"  [{self.name}] Found {len(found_facts)} relevant fact(s).")
        return result


class WriterAgent(BaseAgent):
    """
    Specialist: writes clear, structured content based on research.
    Takes raw facts and turns them into readable articles or summaries.
    """

    def __init__(self):
        super().__init__("WriterAgent")

    def run(self, task: str, context: dict = None) -> str:
        """Write an article based on provided research context."""
        context = context or {}

        print(f"\n  [{self.name}] Writing: {task}")

        research = context.get("research", "No research provided.")
        topic    = context.get("topic", task)

        # Simulated writing: format the research into a readable article
        lines = [
            f"# {topic}",
            "",
            "## Overview",
            research[:300] + ("..." if len(research) > 300 else ""),
            "",
            "## Key Points",
        ]

        # Extract bullet points from research (split on ". " and add as bullets)
        sentences = research.replace("\n", " ").split(". ")
        for sentence in sentences[:4]:    # Take first 4 sentences as key points
            if len(sentence.strip()) > 20:
                lines.append(f"- {sentence.strip()}.")

        lines.append("")
        lines.append("## Conclusion")
        lines.append(f"This provides a solid foundation for understanding {topic}.")

        article = "\n".join(lines)
        print(f"  [{self.name}] Article written ({len(article)} chars).")
        return article


class ReviewerAgent(BaseAgent):
    """
    Specialist: reviews and improves written content.
    Checks for clarity, completeness, and provides a quality score.
    """

    def __init__(self):
        super().__init__("ReviewerAgent")

    def run(self, task: str, context: dict = None) -> str:
        """Review a draft and provide feedback + quality score."""
        context = context or {}

        print(f"\n  [{self.name}] Reviewing draft...")

        draft = context.get("draft", "")
        if not draft:
            return "Error: No draft provided for review."

        # Simulated review logic
        word_count = len(draft.split())
        has_sections = "#" in draft           # Check for headings
        has_bullets = "-" in draft            # Check for bullet points
        is_long_enough = word_count > 30      # Minimum length check

        # Build feedback
        feedback_items = []
        score = 0

        if has_sections:
            feedback_items.append("Good: Has clear sections with headings.")
            score += 30
        else:
            feedback_items.append("Missing: Should have section headings for clarity.")

        if has_bullets:
            feedback_items.append("Good: Uses bullet points for key information.")
            score += 20
        else:
            feedback_items.append("Suggestion: Add bullet points to highlight key facts.")

        if is_long_enough:
            feedback_items.append(f"Good: Adequate length ({word_count} words).")
            score += 30
        else:
            feedback_items.append(f"Warning: Too short ({word_count} words). Expand the content.")

        if "introduction" in draft.lower() or "overview" in draft.lower():
            feedback_items.append("Good: Has an introduction/overview section.")
            score += 10

        if "conclusion" in draft.lower():
            feedback_items.append("Good: Has a conclusion section.")
            score += 10

        review = f"QUALITY SCORE: {score}/100\n\nFEEDBACK:\n"
        for item in feedback_items:
            review += f"  - {item}\n"
        review += f"\nSUGGESTION: {'Publish as-is' if score >= 70 else 'Revise before publishing'}"

        print(f"  [{self.name}] Review complete. Score: {score}/100")
        return review


class MathAgent(BaseAgent):
    """
    Specialist: handles all mathematical computations.
    Fast and accurate for any numeric task.
    """

    def __init__(self):
        super().__init__("MathAgent")

    def run(self, task: str, context: dict = None) -> str:
        """Compute a mathematical expression."""
        import math

        print(f"\n  [{self.name}] Computing: {task}")

        # Extract the expression (everything after "calculate" or "compute")
        expr = task
        for prefix in ["calculate", "compute", "evaluate", "what is"]:
            if prefix in task.lower():
                expr = task.lower().split(prefix)[-1].strip()
                break

        # Clean up the expression
        expr = expr.replace("^", "**")       # ^ -> Python exponent operator
        expr = expr.replace("x", "*")        # x -> Python multiply operator
        expr = expr.strip()

        try:
            safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
            result = eval(expr, {"__builtins__": {}}, safe_env)
            output = f"Result: {expr} = {result}"
            print(f"  [{self.name}] {output}")
            return output
        except Exception as e:
            error = f"Cannot evaluate '{expr}': {e}"
            print(f"  [{self.name}] Error: {error}")
            return error


class SummaryAgent(BaseAgent):
    """
    Specialist: creates concise summaries of long text.
    """

    def __init__(self):
        super().__init__("SummaryAgent")

    def run(self, task: str, context: dict = None) -> str:
        """Summarize text provided in context."""
        context = context or {}

        print(f"\n  [{self.name}] Summarizing...")

        text = context.get("text", task)    # Summarize from context or the task itself

        # Simulated summarization: extract first sentence of each paragraph
        paragraphs = text.split("\n\n")                  # Split into paragraphs
        key_sentences = []
        for para in paragraphs:
            sentences = para.split(". ")                 # Split into sentences
            if sentences and sentences[0].strip():
                key_sentences.append(sentences[0].strip())  # Take first sentence

        summary = "SUMMARY: " + " | ".join(key_sentences[:3])  # Join first 3 key sentences
        print(f"  [{self.name}] Summary: {len(summary)} chars")
        return summary


# ==============================================================================
# PART A: Sequential Pipeline
# ==============================================================================

print("\n" + "=" * 65)
print("PART A: Sequential Pipeline (Research -> Write -> Review)")
print("=" * 65)

print("""
Pipeline: Three agents working in sequence.
  Agent 1 (ResearchAgent) -> output fed to Agent 2 (WriterAgent)
  Agent 2 output fed to Agent 3 (ReviewerAgent)

Like a conveyor belt: each station hands work to the next.
C# analogy: method chaining or IEnumerable pipeline.
""")


def run_pipeline(topic: str) -> dict:
    """
    Run the Research -> Write -> Review pipeline for a topic.
    Returns a dict with all intermediate and final results.
    """
    print(f"Starting pipeline for topic: '{topic}'")
    print("-" * 45)

    # Shared blackboard: all agents read from and write to this
    blackboard = {"topic": topic}     # Start with just the topic

    # Agent 1: Research
    researcher = ResearchAgent()
    research_result = researcher.run(
        task=f"Research everything about {topic}",
        context=blackboard
    )
    blackboard["research"] = research_result   # Save to blackboard

    # Agent 2: Write (reads research from blackboard)
    writer = WriterAgent()
    draft_result = writer.run(
        task=f"Write an article about {topic}",
        context=blackboard                     # Gets "research" from blackboard
    )
    blackboard["draft"] = draft_result         # Save draft to blackboard

    # Agent 3: Review (reads draft from blackboard)
    reviewer = ReviewerAgent()
    review_result = reviewer.run(
        task=f"Review the article about {topic}",
        context=blackboard                     # Gets "draft" from blackboard
    )
    blackboard["review"] = review_result       # Save review to blackboard

    return blackboard                          # Return everything


result = run_pipeline("Python web frameworks")

print("\n" + "=" * 45)
print("PIPELINE RESULTS:")
print("=" * 45)
print(f"\nFINAL DRAFT (first 400 chars):\n{result['draft'][:400]}...")
print(f"\nREVIEW:\n{result['review']}")


# ==============================================================================
# PART B: Parallel Execution
# ==============================================================================

print("\n" + "=" * 65)
print("PART B: Parallel Agent Execution")
print("=" * 65)

print("""
Some tasks are INDEPENDENT -- they do not need each other's results.
These can run in PARALLEL (at the same time).

Example: Research Python AND research FastAPI at the same time.
  - Research #1 and Research #2 are independent
  - Run both simultaneously
  - Combine results when both finish

C# analogy: Task.WhenAll() -- run tasks in parallel, await all.
""")


def run_agents_in_parallel(tasks: list) -> list:
    """
    Run multiple agents simultaneously.
    tasks: list of (agent, task_string, context_dict) tuples
    Returns: list of result strings, in the same order as tasks
    """

    def run_one(args):
        """Run a single agent (called by each thread)."""
        agent, task, context = args         # Unpack the tuple
        return agent.run(task, context)     # Run the agent

    # Run all tasks in parallel using a thread pool
    # ThreadPoolExecutor = a pool of worker threads (like Task.Run() in C#)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = executor.map(run_one, tasks)    # Start all tasks simultaneously
        results = list(futures)                    # Wait for all and collect results

    return results


# Run research on multiple topics in parallel
parallel_research_tasks = [
    (ResearchAgent(), "Research Python programming language", {}),
    (ResearchAgent(), "Research FastAPI framework",           {}),
    (ResearchAgent(), "Research LLM agents and AI",          {}),
    (MathAgent(),     "Calculate 2 ** 32",                   {}),
]

print("Running 4 agents in PARALLEL:")
for agent, task, ctx in parallel_research_tasks:
    print(f"  - {agent.name}: {task[:50]}")

import time
start_time = time.time()
parallel_results = run_agents_in_parallel(parallel_research_tasks)
elapsed = time.time() - start_time

print(f"\nAll 4 agents finished in {elapsed:.2f} seconds.")
print("\nResults:")
for i, (task_tuple, result) in enumerate(zip(parallel_research_tasks, parallel_results), 1):
    agent, task, _ = task_tuple
    print(f"\n  Result {i} ({agent.name}):")
    print(f"  {result[:150]}...")


# ==============================================================================
# PART C: Orchestrator with Dynamic Routing
# ==============================================================================

print("\n" + "=" * 65)
print("PART C: Orchestrator with Dynamic Routing")
print("=" * 65)

print("""
The Orchestrator receives a user goal and ROUTES each sub-task
to the most appropriate specialist agent.

It acts as the "manager":
  - Breaks the goal into steps
  - Decides which agent handles each step
  - Passes results between agents
  - Produces the final combined output
""")


class OrchestratorAgent:
    """
    The master agent that coordinates all specialist agents.
    Breaks a goal into sub-tasks and assigns each to the right specialist.

    C# analogy: like a Mediator pattern or CQRS command dispatcher --
    receives a request and routes it to the correct handler.
    """

    def __init__(self):
        self.name = "OrchestratorAgent"

        # Register all available specialist agents
        self.agents = {
            "research":  ResearchAgent(),    # Use for: factual lookups, knowledge
            "write":     WriterAgent(),      # Use for: article/content writing
            "review":    ReviewerAgent(),    # Use for: quality checking, feedback
            "math":      MathAgent(),        # Use for: calculations, numbers
            "summarize": SummaryAgent(),     # Use for: condensing long text
        }

    def _route(self, task: str) -> str:
        """
        Decide which specialist agent should handle a given task.
        In production: an LLM reads the task and picks the agent.
        Here: simple keyword matching (same logic as a real LLM would use).
        """
        task_lower = task.lower()

        if any(w in task_lower for w in ["calculate", "compute", "math", "how many", "+", "-", "*", "/"]):
            return "math"
        if any(w in task_lower for w in ["write", "create article", "draft", "compose"]):
            return "write"
        if any(w in task_lower for w in ["review", "check", "evaluate quality", "score"]):
            return "review"
        if any(w in task_lower for w in ["summarize", "summary", "condense", "brief"]):
            return "summarize"
        # Default: research
        return "research"

    def execute(self, goal: str, sub_tasks: list) -> dict:
        """
        Execute a multi-step goal by routing sub-tasks to specialists.
        goal:      the overall objective
        sub_tasks: list of task strings to execute in order
        Returns:   blackboard dict with all results
        """
        print(f"\nORCHESTRATOR: Goal = '{goal}'")
        print(f"ORCHESTRATOR: {len(sub_tasks)} sub-tasks to execute")
        print("-" * 50)

        blackboard = {"goal": goal}     # Shared state, starts with just the goal

        for i, task in enumerate(sub_tasks, 1):
            agent_type = self._route(task)          # Decide which agent to use
            agent = self.agents[agent_type]         # Get the agent

            print(f"\n[Sub-task {i}/{len(sub_tasks)}] -> Routed to {agent.name}")
            print(f"  Task: {task}")

            result = agent.run(task, context=blackboard)   # Run with full blackboard
            blackboard[f"step_{i}_{agent_type}"] = result  # Save with unique key

        # Final: summarize everything into a coherent answer
        print("\n[ORCHESTRATOR] Combining all results...")
        all_results = "\n\n".join([
            f"Step {k}: {v}"
            for k, v in blackboard.items()
            if k.startswith("step_")
        ])
        blackboard["final_output"] = f"Goal: {goal}\n\nResults:\n{all_results}"

        return blackboard


# Test the orchestrator with a complex multi-task goal
orchestrator = OrchestratorAgent()

result = orchestrator.execute(
    goal="Create a comprehensive Python learning guide",
    sub_tasks=[
        "Research Python programming language",
        "Research LLM agents and AI",
        "Calculate 2 ** 16 to show Python's power for computation",
        "Write an article about Python for beginners",
        "Review the written draft for quality",
        "Summarize the key findings",
    ]
)

print("\n" + "=" * 65)
print("ORCHESTRATOR FINAL OUTPUT")
print("=" * 65)
print(result.get("final_output", "No output")[:600] + "...")


# ==============================================================================
# DIAGRAM: The Full Multi-Agent Architecture
# ==============================================================================

print("\n" + "=" * 65)
print("ARCHITECTURE DIAGRAM")
print("=" * 65)

print("""
Here is the full multi-agent architecture we built:

User / Goal
    |
    v
+-------------------+
| OrchestratorAgent |  <- Breaks goal into steps, routes each step
+-------------------+
   |    |    |    |
   v    v    v    v
[R]  [W]  [Rev] [Math]  <- Specialist agents
Research Write Review Calc
   |    |    |    |
   v    v    v    v
+-------------------+
|  Shared Blackboard|  <- All agents read/write here
+-------------------+
         |
         v
    Final Answer

In C# terms:
  Orchestrator  = IMediator or command dispatcher
  Specialists   = ICommandHandler<T> implementations
  Blackboard    = Dictionary<string, object> passed by reference
  Parallel run  = Task.WhenAll()
""")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("=" * 65)
print("SUMMARY - Multi-Agent System")
print("=" * 65)

print("""
WHAT WE BUILT:
  - 5 specialist agents: Research, Writer, Reviewer, Math, Summary
  - Sequential pipeline (Part A): agents in sequence, results flow forward
  - Parallel execution (Part B): independent agents run simultaneously
  - Orchestrator (Part C): routes tasks to the right specialist

KEY PATTERNS:
  1. All agents share the same interface: agent.run(task, context) -> str
  2. Blackboard = shared state dictionary (agents read + write)
  3. Orchestrator decides routing (which agent for which task)
  4. Parallel execution: concurrent.futures.ThreadPoolExecutor
  5. Sequential pipeline: blackboard passed from agent to agent

REAL-WORLD APPLICATIONS:
  - Customer support: Classifier -> Specialist -> Responder -> QualityCheck
  - Code generation: Architect -> Coder -> Tester -> Reviewer
  - Content creation: Researcher -> Writer -> Editor -> Translator
  - Data analysis: DataFetcher -> Analyzer -> Visualizer -> Reporter

C# ANALOGY:
  Sequential pipeline = method chaining or IEnumerable pipeline
  Parallel agents     = Task.WhenAll() with multiple Task.Run() calls
  Orchestrator        = Mediator pattern / CQRS dispatcher
  Blackboard          = shared Dictionary<string, object>

MODULE 11 COMPLETE!
  You now know:
  01. What agents are (brain + tools + memory + loop)
  02. How tool calling works (definitions, registry, execution)
  03. ReAct pattern (Thought -> Action -> Observation -> repeat)
  04. Memory (short-term buffer + long-term vector store)
  05. Multi-agent systems (pipeline, parallel, orchestrator)

  Next: Exercises to practice building these yourself!
""")

print("=" * 65)
print("END OF EXAMPLE 05")
print("=" * 65)

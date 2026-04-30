"""
Exercise 05: Build a Multi-Agent Pipeline
==========================================

GOAL
----
Practice building a multi-agent orchestration system where
multiple specialist agents work together to solve a complex task.

EXERCISES
---------
Exercise 1: Complete the BaseAgent interface
Exercise 2: Build a TranslatorAgent specialist
Exercise 3: Build an OrchestratorAgent that routes tasks
Exercise 4: Run a full Research -> Write -> Translate pipeline

HOW TO RUN
----------
  python exercise_05_orchestration.py

LIBRARIES NEEDED
-----------------
  None (pure Python)
"""

print("=" * 65)
print("EXERCISE 05: Build a Multi-Agent Pipeline")
print("=" * 65)


# ===========================================================================
# EXERCISE 1: Complete the BaseAgent Interface
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 1: Complete BaseAgent")
print("=" * 65)

print("""
TODO: Complete the BaseAgent class.

  It should have:
  - __init__(self, name: str): store self.name and self.result_history = []
  - run(self, task: str, context: dict = None) -> str:
      Should raise NotImplementedError (forces subclasses to implement it)
  - log_result(self, task: str, result: str):
      Should append {"task": task, "result": result} to self.result_history
  - __str__(self): return self.name

This is like an abstract base class in C#:
  abstract class BaseAgent {
      public abstract string Run(string task, Dictionary<string,object> context);
  }
""")


class BaseAgent:
    """
    Abstract base class for all specialist agents.
    TODO: Complete the __init__, run, log_result, and __str__ methods.
    """

    def __init__(self, name: str):
        """TODO: Store name and initialize result_history as empty list."""
        pass  # DELETE THIS and implement

    def run(self, task: str, context: dict = None) -> str:
        """TODO: Raise NotImplementedError with message 'Subclasses must implement run()'"""
        pass  # DELETE THIS and implement

    def log_result(self, task: str, result: str):
        """TODO: Append {"task": task, "result": result} to self.result_history"""
        pass  # DELETE THIS and implement

    def __str__(self):
        """TODO: Return self.name"""
        pass  # DELETE THIS and implement


# Test BaseAgent (should raise NotImplementedError)
print("Testing BaseAgent:")
try:
    b = BaseAgent("TestAgent")
    b.run("test task")
    print("ERROR: Should have raised NotImplementedError!")
except NotImplementedError as e:
    print(f"Good! BaseAgent.run() raised NotImplementedError: {e}")
except Exception:
    print("TODO: Implement BaseAgent first.")


# ===========================================================================
# PROVIDED SPECIALIST AGENTS
# (Study these -- they show the pattern you'll use in Exercise 2)
# ===========================================================================

class ResearchAgent(BaseAgent):
    """Researches facts about a topic."""

    def __init__(self):
        super().__init__("ResearchAgent")    # Call BaseAgent.__init__
        self.knowledge = {
            "python":    "Python: high-level language, created 1991 by Guido van Rossum. "
                         "Widely used for AI, web dev, automation.",
            "csharp":    "C#: object-oriented language by Microsoft, created 2000. "
                         "Used for .NET apps, games (Unity), enterprise software.",
            "ai agents": "AI Agents: LLMs that can plan and use tools to achieve goals. "
                         "Use ReAct pattern: Thought -> Action -> Observation -> repeat.",
            "fastapi":   "FastAPI: Python web framework for building APIs fast. "
                         "Auto-generates OpenAPI docs. Used for microservices and AI backends.",
        }

    def run(self, task: str, context: dict = None) -> str:
        context = context or {}
        task_lower = task.lower()

        for topic, info in self.knowledge.items():
            if topic in task_lower:
                result = f"Research on '{topic}': {info}"
                self.log_result(task, result)    # Log for history
                return result

        result = f"No specific research found for: '{task}'"
        self.log_result(task, result)
        return result


class WriterAgent(BaseAgent):
    """Writes articles and summaries from research."""

    def __init__(self):
        super().__init__("WriterAgent")

    def run(self, task: str, context: dict = None) -> str:
        context = context or {}
        research = context.get("research", "No research provided.")
        topic    = context.get("topic", "the topic")

        article = f"""ARTICLE: {topic.title()}

Introduction:
{research[:200]}...

Key Takeaways:
- {research.split('.')[0]}.
- This technology is widely used in industry.
- Further learning is recommended for advanced use cases.

Conclusion:
Understanding {topic} is valuable for modern software development.
"""
        self.log_result(task, article)
        return article


# ===========================================================================
# EXERCISE 2: Build TranslatorAgent
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 2: Build TranslatorAgent")
print("=" * 65)

print("""
TODO: Build a TranslatorAgent that translates text.
  It should inherit from BaseAgent.

  run(self, task: str, context: dict = None) -> str:
    1. Get the text from context["text"] (or use task as the text if no context)
    2. Get the target language from context["language"] (default: "spanish")
    3. Look up the translation from self.translations
    4. If not found, return: f"[Simulated {language} translation of: {text[:50]}]"
    5. Call self.log_result(task, result) and return result

  self.translations should be a dict mapping:
    ("hello", "spanish")     -> "Hola"
    ("hello", "french")      -> "Bonjour"
    ("hello", "german")      -> "Hallo"
    ("goodbye", "spanish")   -> "Adios"
    ("goodbye", "french")    -> "Au revoir"
    ("python", "spanish")    -> "Python (mismo nombre en espanol)"
    ("thank you", "spanish") -> "Gracias"
    ("thank you", "french")  -> "Merci"

HINT:
  Look up: (text.lower(), language.lower()) in self.translations
""")


class TranslatorAgent(BaseAgent):
    """
    TODO: Build this specialist agent.
    """

    def __init__(self):
        # TODO: Call super().__init__("TranslatorAgent")
        # TODO: Build self.translations dictionary
        pass  # DELETE THIS and implement

    def run(self, task: str, context: dict = None) -> str:
        """
        TODO: Translate text from context["text"] to context["language"].
        """
        pass  # DELETE THIS and implement


# Test TranslatorAgent
print("Testing TranslatorAgent:")
translator = TranslatorAgent()

if hasattr(translator, 'run') and translator.run.__code__.co_code != BaseAgent.run.__code__.co_code:
    tests = [
        ({"text": "hello", "language": "spanish"},  "Expected: Hola"),
        ({"text": "goodbye", "language": "french"},  "Expected: Au revoir"),
        ({"text": "python", "language": "spanish"},  "Expected: Python translation"),
        ({"text": "some unknown text", "language": "japanese"}, "Expected: [Simulated...]"),
    ]
    for ctx, hint in tests:
        result = translator.run(f"Translate '{ctx['text']}' to {ctx['language']}", context=ctx)
        print(f"  {hint}")
        print(f"  -> {result}")
        print()
else:
    print("TODO: Implement TranslatorAgent first.")


# ===========================================================================
# EXERCISE 3: Build OrchestratorAgent
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 3: Build OrchestratorAgent")
print("=" * 65)

print("""
TODO: Build an OrchestratorAgent that:
  1. Has access to all three specialists: research, writer, translator
  2. Has a route(task: str) -> str method that returns the agent type:
       - "research"   if task contains: "research", "find", "what is", "look up"
       - "write"      if task contains: "write", "article", "draft", "compose"
       - "translate"  if task contains: "translate", "spanish", "french", "german"
       - "research"   (default fallback)

  3. Has an execute(goal: str, sub_tasks: list) -> dict method that:
       - Creates an empty blackboard dict
       - For each sub_task string in sub_tasks:
           a. Routes it to the correct agent
           b. Runs the agent: agent.run(sub_task, context=blackboard)
           c. Stores result in blackboard with key f"step_{i}_{agent_type}"
           d. Prints: "[Orchestrator] Step i: routed to AgentName"
       - Returns the blackboard
""")


class OrchestratorAgent:
    """
    TODO: Build this orchestrator.
    """

    def __init__(self):
        # TODO: Initialize self.agents with all three specialists
        # self.agents = {
        #     "research":   ResearchAgent(),
        #     "write":      WriterAgent(),
        #     "translate":  TranslatorAgent(),
        # }
        pass  # DELETE THIS and implement

    def route(self, task: str) -> str:
        """
        TODO: Return the agent type (string key) for the given task.
        """
        pass  # DELETE THIS and implement

    def execute(self, goal: str, sub_tasks: list) -> dict:
        """
        TODO: Execute each sub-task by routing to the correct agent.
        """
        pass  # DELETE THIS and implement


# Test OrchestratorAgent routing
print("Testing OrchestratorAgent routing:")
orch = OrchestratorAgent()

if hasattr(orch, 'route') and orch.route:
    routing_tests = [
        ("Research everything about Python",     "Expected: research"),
        ("Write an article about AI agents",     "Expected: write"),
        ("Translate hello to Spanish",           "Expected: translate"),
        ("Find information about FastAPI",       "Expected: research"),
        ("Draft a summary of the results",       "Expected: write"),
    ]
    for task, hint in routing_tests:
        try:
            result = orch.route(task)
            print(f"  '{task[:45]}...' -> {result}  ({hint})")
        except Exception as e:
            print(f"  Error: {e}")
else:
    print("TODO: Implement OrchestratorAgent first.")


# ===========================================================================
# EXERCISE 4: Full Pipeline
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 4: Full Research -> Write -> Translate Pipeline")
print("=" * 65)

print("""
TODO: Using the OrchestratorAgent, run a full pipeline for:
  Goal: "Create a Spanish article about Python"

  Sub-tasks (in order):
    1. "Research Python programming language"
    2. "Write an article about Python"
    3. "Translate the article to Spanish"

  After running, print:
    - All blackboard keys
    - The final article (blackboard["step_2_write"] or similar key)
    - The translation result

HINT:
  After execute(), print blackboard.keys() to see what was stored.
  Then print the relevant values.
""")

# TODO: Run the pipeline
# orch4 = OrchestratorAgent()
# blackboard = orch4.execute(
#     goal="Create a Spanish article about Python",
#     sub_tasks=[
#         "Research Python programming language",
#         "Write an article about Python",
#         "Translate hello to Spanish",
#     ]
# )
#
# print("\nBlackboard keys:", list(blackboard.keys()))
# for key, value in blackboard.items():
#     if key != "goal":
#         print(f"\n{key}:\n{str(value)[:200]}...")

print("TODO: Implement the pipeline above (uncomment the code).")


# ===========================================================================
# BONUS: Parallel Pipeline
# ===========================================================================

print("\n" + "=" * 65)
print("BONUS: Parallel Multi-Agent Execution")
print("=" * 65)

print("""
BONUS CHALLENGE (optional):
  Run research on Python AND C# simultaneously using parallel agents.
  Both research tasks are independent so they can run at the same time.

  Use concurrent.futures.ThreadPoolExecutor to run them in parallel.
  Then pass both results to the WriterAgent to write a comparison.

HINT from Example 05:
  import concurrent.futures
  with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(agent.run, task, {}) for agent, task in pairs]
      results = [f.result() for f in futures]
""")


# ===========================================================================
# SOLUTION
# ===========================================================================

print("\n" + "=" * 65)
print("SOLUTION (uncomment to check)")
print("=" * 65)

"""
SOLUTION FOR EXERCISE 1:

    class BaseAgent:
        def __init__(self, name: str):
            self.name = name
            self.result_history = []

        def run(self, task: str, context: dict = None) -> str:
            raise NotImplementedError("Subclasses must implement run()")

        def log_result(self, task: str, result: str):
            self.result_history.append({"task": task, "result": result})

        def __str__(self):
            return self.name

SOLUTION FOR EXERCISE 2:

    class TranslatorAgent(BaseAgent):
        def __init__(self):
            super().__init__("TranslatorAgent")
            self.translations = {
                ("hello", "spanish"):     "Hola",
                ("hello", "french"):      "Bonjour",
                ("hello", "german"):      "Hallo",
                ("goodbye", "spanish"):   "Adios",
                ("goodbye", "french"):    "Au revoir",
                ("python", "spanish"):    "Python (mismo nombre en espanol)",
                ("thank you", "spanish"): "Gracias",
                ("thank you", "french"):  "Merci",
            }

        def run(self, task: str, context: dict = None) -> str:
            context = context or {}
            text     = context.get("text", task)
            language = context.get("language", "spanish")
            key = (text.lower(), language.lower())
            if key in self.translations:
                result = self.translations[key]
            else:
                result = f"[Simulated {language} translation of: {text[:50]}]"
            self.log_result(task, result)
            return result

SOLUTION FOR EXERCISE 3:

    class OrchestratorAgent:
        def __init__(self):
            self.agents = {
                "research":  ResearchAgent(),
                "write":     WriterAgent(),
                "translate": TranslatorAgent(),
            }

        def route(self, task: str) -> str:
            t = task.lower()
            if any(w in t for w in ["translate", "spanish", "french", "german"]):
                return "translate"
            if any(w in t for w in ["write", "article", "draft", "compose"]):
                return "write"
            return "research"

        def execute(self, goal: str, sub_tasks: list) -> dict:
            blackboard = {"goal": goal}
            for i, task in enumerate(sub_tasks, 1):
                agent_type = self.route(task)
                agent = self.agents[agent_type]
                print(f"[Orchestrator] Step {i}: routed to {agent.name}")
                result = agent.run(task, context=blackboard)
                blackboard[f"step_{i}_{agent_type}"] = result
            return blackboard
"""

print("See SOLUTION block above.")
print("=" * 65)
print("END OF EXERCISE 05")
print("=" * 65)

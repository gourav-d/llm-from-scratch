"""
Project 01: Personal Assistant Agent
======================================

WHAT THIS BUILDS
-----------------
A fully functional personal assistant agent that:
- Remembers who you are (long-term memory)
- Answers questions using multiple tools (search, calculator, date)
- Tracks your tasks (to-do list)
- Maintains conversation history (short-term memory)
- Handles a full interactive chat session

LEARNING GOALS
--------------
- Combine all Module 11 concepts: tools + ReAct + memory
- See how a real assistant (like Claude or Siri) works internally
- Practice building a useful end-to-end agent

HOW TO RUN
----------
  python project_01_personal_assistant.py

HOW IT WORKS
------------
Part A: Simple demo (runs automatically, no user input needed)
Part B: Interactive mode (you type questions, agent responds)
        Set INTERACTIVE = True to enable

LIBRARIES NEEDED
-----------------
  None (pure Python)
"""

import math
import datetime
import json

print("=" * 65)
print("PROJECT 01: Personal Assistant Agent")
print("=" * 65)

# Set to True to have a real conversation with the agent
INTERACTIVE = False


# ==============================================================================
# TOOLS
# ==============================================================================

def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    try:
        safe = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        return str(eval(expression, {"__builtins__": {}}, safe))
    except Exception as e:
        return f"Error: {e}"

def get_current_datetime() -> str:
    """Returns the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")    # e.g., "Wednesday, April 30, 2026 at 10:30 AM"

def search_knowledge(query: str) -> str:
    """Searches built-in knowledge base."""
    knowledge = {
        "python":      "Python: high-level language, 1991, by Guido van Rossum. Great for AI, web, data.",
        "csharp":      "C#: Microsoft language, 2000. Runs on .NET. OOP, strongly typed.",
        "dotnet":      ".NET: Microsoft framework for building apps. Supports C#, F#, VB.NET.",
        "ai":          "AI (Artificial Intelligence): computer systems that simulate human intelligence.",
        "llm":         "LLM: Large Language Model. Neural network trained on text. Examples: GPT, Claude.",
        "machine learning": "ML: Teaching computers to learn from data without explicit programming.",
        "blackline":   "BlackLine: Financial technology company. Specializes in accounting automation.",
        "react":       "ReAct: Reason + Act pattern for LLM agents. Think -> Tool -> Observe -> repeat.",
        "transformer": "Transformer: Neural network architecture using attention. Powers GPT, BERT, Claude.",
        "vector db":   "Vector database: stores embeddings for similarity search. Examples: ChromaDB, Pinecone.",
    }
    q = query.lower()
    for key, value in knowledge.items():
        if key in q or any(word in q for word in key.split()):
            return value
    return f"No specific knowledge found for '{query}'. I know about: {', '.join(knowledge.keys()[:5])}..."

AVAILABLE_TOOLS = {
    "calculator":         calculator,
    "get_datetime":       lambda: get_current_datetime(),
    "search_knowledge":   search_knowledge,
}


# ==============================================================================
# MEMORY SYSTEM
# ==============================================================================

class ConversationBuffer:
    """Short-term: last N messages."""

    def __init__(self, max_size: int = 10):
        self.messages = []
        self.max_size = max_size

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_size:
            self.messages.pop(0)    # Drop oldest

    def get_text(self) -> str:
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.messages)

    def clear(self):
        self.messages = []

    def __len__(self):
        return len(self.messages)


class LongTermMemory:
    """Long-term: persistent facts with keyword retrieval."""

    SAVE_TRIGGERS = [
        "my name is", "i am", "i work", "i prefer", "i like",
        "i completed", "i learned", "my goal", "i am a", "i live"
    ]

    def __init__(self):
        self.facts = []

    def should_save(self, text: str) -> bool:
        return any(t in text.lower() for t in self.SAVE_TRIGGERS)

    def save(self, fact: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        self.facts.append({"fact": fact, "date": timestamp})

    def recall(self, query: str, top_k: int = 3) -> list:
        qwords = set(query.lower().split())
        scored = []
        for f in self.facts:
            fwords = set(f["fact"].lower().split())
            score = len(qwords & fwords)
            if score > 0:
                scored.append((score, f["fact"]))
        scored.sort(reverse=True)
        return [fact for _, fact in scored[:top_k]]

    def all_facts(self) -> list:
        return [f["fact"] for f in self.facts]


class TaskList:
    """A simple to-do list manager."""

    def __init__(self):
        self.tasks = []
        self._next_id = 1

    def add_task(self, title: str) -> str:
        task = {"id": self._next_id, "title": title, "done": False}
        self.tasks.append(task)
        self._next_id += 1
        return f"Task #{task['id']} added: '{title}'"

    def complete_task(self, task_id: int) -> str:
        for task in self.tasks:
            if task["id"] == task_id:
                task["done"] = True
                return f"Task #{task_id} marked as done."
        return f"Task #{task_id} not found."

    def list_tasks(self) -> str:
        if not self.tasks:
            return "No tasks yet."
        lines = ["Your tasks:"]
        for t in self.tasks:
            status = "[x]" if t["done"] else "[ ]"
            lines.append(f"  {status} #{t['id']}: {t['title']}")
        return "\n".join(lines)

    def pending_count(self) -> int:
        return sum(1 for t in self.tasks if not t["done"])


# ==============================================================================
# THE PERSONAL ASSISTANT AGENT
# ==============================================================================

class PersonalAssistant:
    """
    A fully-featured personal assistant with:
    - Multi-tool support (calculator, datetime, knowledge search)
    - Short-term memory (conversation history)
    - Long-term memory (persistent user facts)
    - Task management (to-do list)

    This is the COMPLETE agent pattern from Module 11 in one class.
    """

    def __init__(self, assistant_name: str = "Alex"):
        self.name = assistant_name
        self.tools = AVAILABLE_TOOLS
        self.short_term = ConversationBuffer(max_size=10)
        self.long_term  = LongTermMemory()
        self.tasks      = TaskList()

        # Greet on first run
        print(f"\nAssistant '{self.name}' initialized.")
        print(f"Available tools: {list(self.tools.keys())}")
        print(f"Memory systems: short-term (buffer) + long-term (facts)")

    def _decide_tool(self, message: str) -> tuple:
        """
        Decide which tool (if any) to use for this message.
        Returns (tool_name, args) or (None, None) if no tool needed.
        """
        msg = message.lower()

        # Date/time questions
        if any(w in msg for w in ["date", "time", "today", "now", "day", "year"]):
            return ("get_datetime", {})

        # Math questions
        math_signals = ["calculate", "compute", "what is", "+", "-", "*", "/",
                        "sqrt", "**", "percent", "how much", "how many"]
        if any(s in msg for s in math_signals):
            # Try to extract the math expression
            for prefix in ["calculate", "compute", "what is"]:
                if prefix in msg:
                    expr = message.split(prefix, 1)[-1].strip()
                    expr = expr.rstrip("?!.")
                    return ("calculator", {"expression": expr})
            # Fallback: search instead
            return ("search_knowledge", {"query": message})

        # Knowledge questions
        knowledge_signals = ["what is", "tell me", "explain", "describe", "how does",
                              "who created", "define", "what are"]
        if any(s in msg for s in knowledge_signals):
            topic = message
            for prefix in ["what is", "tell me about", "explain", "describe"]:
                if prefix in msg:
                    topic = message.lower().split(prefix, 1)[-1].strip().rstrip("?!")
                    break
            return ("search_knowledge", {"query": topic})

        # Task management
        if "add task" in msg or "remind me" in msg or "todo" in msg:
            return ("add_task_special", {"title": message})

        return (None, None)

    def _handle_task_commands(self, message: str) -> str:
        """Handle task management commands."""
        msg = message.lower()

        if "add task" in msg or "remind me to" in msg:
            task_title = message
            for prefix in ["add task:", "add task", "remind me to"]:
                if prefix in msg:
                    task_title = message.lower().split(prefix, 1)[-1].strip()
                    break
            return self.tasks.add_task(task_title)

        if "list tasks" in msg or "show tasks" in msg or "my tasks" in msg:
            return self.tasks.list_tasks()

        if "complete task" in msg or "done task" in msg or "finish task" in msg:
            import re
            numbers = re.findall(r'\d+', message)
            if numbers:
                return self.tasks.complete_task(int(numbers[0]))
            return "Please specify a task number. Example: 'complete task 1'"

        return None    # Not a task command

    def _generate_response(self, message: str, memories: list, tool_result: str) -> str:
        """Generate a contextual response based on memories and tool results."""
        msg = message.lower()

        # If we have a tool result, use it
        if tool_result:
            if "date" in msg or "time" in msg or "today" in msg:
                return f"It is currently {tool_result}."
            if any(c in message for c in ["+", "-", "*", "/", "**"]) or "calculate" in msg:
                return f"The result is: {tool_result}"
            return f"Here is what I found: {tool_result}"

        # Personal questions using memory
        if "name" in msg or "who am i" in msg:
            for mem in memories:
                if "name" in mem.lower():
                    return f"Based on my notes: {mem}"

        if any(w in msg for w in ["work", "job", "company"]):
            for mem in memories:
                if "work" in mem.lower():
                    return f"From my memory: {mem}"

        if "remember" in msg or "recall" in msg:
            if memories:
                return f"Yes! I recall: {'; '.join(memories[:2])}"
            return "I don't have specific memories about that yet. Tell me and I'll remember!"

        if memories:
            return f"Based on what I know about you: {memories[0]}. How can I help further?"

        # Default friendly responses
        defaults = {
            "hello": f"Hello! I'm {self.name}, your personal assistant. How can I help?",
            "hi":    f"Hi there! I'm {self.name}. What can I do for you today?",
            "thanks": "You're welcome! Let me know if you need anything else.",
            "thank you": "Happy to help! Is there anything else you'd like?",
            "bye":   f"Goodbye! Come back anytime. Your memories are saved.",
            "help":  "I can: answer questions, do math, tell the time, search my knowledge, manage your tasks. Just ask!",
        }
        for keyword, response in defaults.items():
            if keyword in msg:
                return response

        return f"I can help with math, facts, tasks, and questions about your goals. What would you like?"

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        This is the main agent loop in one method.
        """
        print(f"\n[YOU]  {user_message}")

        # Step 1: Save important facts to long-term memory
        if self.long_term.should_save(user_message):
            self.long_term.save(user_message)
            print(f"       (Saved to memory)")

        # Step 2: Recall relevant memories
        memories = self.long_term.recall(user_message, top_k=2)

        # Step 3: Add to short-term buffer
        self.short_term.add("user", user_message)

        # Step 4: Check for task management commands first
        task_response = self._handle_task_commands(user_message)
        if task_response:
            response = task_response
        else:
            # Step 5: Decide and call a tool if needed
            tool_name, args = self._decide_tool(user_message)
            tool_result = None

            if tool_name and tool_name != "add_task_special":
                if tool_name in self.tools:
                    tool_result = self.tools[tool_name](**args) if args else self.tools[tool_name]()
                    print(f"       [Tool: {tool_name}] -> {tool_result[:80]}")

            # Step 6: Generate response
            response = self._generate_response(user_message, memories, tool_result)

        # Step 7: Add response to short-term
        self.short_term.add("assistant", response)

        print(f"[{self.name}] {response}")
        return response

    def show_status(self):
        """Print the current state of the assistant."""
        print("\n" + "=" * 55)
        print("ASSISTANT STATUS")
        print("=" * 55)
        print(f"Name:               {self.name}")
        print(f"Short-term messages: {len(self.short_term)}")
        print(f"Long-term facts:     {len(self.long_term.facts)}")
        print(f"Pending tasks:       {self.tasks.pending_count()}")

        if self.long_term.facts:
            print("\nLong-term memories:")
            for fact in self.long_term.all_facts():
                print(f"  - {fact}")

        print(f"\nTask list:")
        print(self.tasks.list_tasks())


# ==============================================================================
# PART A: Automated Demo
# ==============================================================================

print("\n" + "=" * 65)
print("PART A: Automated Demo")
print("=" * 65)

assistant = PersonalAssistant(assistant_name="Alex")

# Demo conversation
demo_messages = [
    "Hello!",
    "My name is Gourav and I work at Blackline.",
    "I am a .NET developer learning Python.",
    "What is today's date?",
    "Calculate 2 ** 16",
    "What is Python?",
    "What is a transformer in AI?",
    "Add task: Complete Module 11 exercises",
    "Add task: Build the personal assistant project",
    "List tasks",
    "What is my name?",
    "Where do I work?",
    "Complete task 1",
    "List tasks",
    "Thanks!",
]

print("\nRunning automated demo conversation...\n")
for message in demo_messages:
    assistant.chat(message)

assistant.show_status()


# ==============================================================================
# PART B: Interactive Mode
# ==============================================================================

print("\n" + "=" * 65)
print("PART B: Interactive Mode")
print("=" * 65)

if not INTERACTIVE:
    print("""
To talk to the assistant yourself:
  1. Open this file
  2. Change: INTERACTIVE = False
  3. To:     INTERACTIVE = True
  4. Run again: python project_01_personal_assistant.py

The assistant will remember facts you share across the conversation.
Type 'quit' or 'exit' to stop.
""")
else:
    print("Interactive mode active. Type 'quit' to exit.\n")
    interactive_assistant = PersonalAssistant(assistant_name="Alex")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            interactive_assistant.chat("bye")
            interactive_assistant.show_status()
            break

        interactive_assistant.chat(user_input)


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("SUMMARY - Personal Assistant Agent")
print("=" * 65)

print("""
WHAT WE BUILT:
  A complete personal assistant with:
  - 3 tools: calculator, datetime, knowledge search
  - Short-term memory (last 10 messages)
  - Long-term memory (auto-saves personal facts)
  - Task management (add, complete, list tasks)
  - Context-aware responses (uses memories to personalize)

PATTERNS USED:
  - ReAct (decide tool -> call tool -> use result in response)
  - Short-term + long-term memory (from Lesson 04)
  - Tool registry and routing (from Lesson 02)
  - Single-agent loop (from Lesson 01)

IN PRODUCTION:
  Replace:
  - FakeLLM decisions with real Claude/GPT calls
  - Keyword-based tool selection with LLM function calling
  - In-memory lists with a real database (SQLite, PostgreSQL)
  - Keyword-recall with ChromaDB vector search (Module 10)

NEXT PROJECT (02):
  Research Agent -- uses vector DB for persistent knowledge storage.
  Builds on Module 10 (ChromaDB) + Module 11 (ReAct + memory).
""")

print("=" * 65)
print("END OF PROJECT 01")
print("=" * 65)

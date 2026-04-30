"""
Exercise 04: Add Memory to an Agent
=====================================

GOAL
----
Practice building both types of agent memory:
  - Short-term: conversation buffer
  - Long-term: persistent fact store with recall

EXERCISES
---------
Exercise 1: Complete ConversationBuffer.add() with max size enforcement
Exercise 2: Complete LongTermMemory.remember() and recall()
Exercise 3: Build a MemoryAgent that uses both memory types
Exercise 4: Test memory across "sessions" (simulate session restart)

HOW TO RUN
----------
  python exercise_04_memory.py

LIBRARIES NEEDED
-----------------
  None (pure Python)
"""

import datetime

print("=" * 65)
print("EXERCISE 04: Add Memory to an Agent")
print("=" * 65)


# ===========================================================================
# EXERCISE 1: ConversationBuffer
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 1: Complete ConversationBuffer")
print("=" * 65)

print("""
TODO: Complete ConversationBuffer.add() so that:
  1. It appends {"role": role, "content": content} to self.messages
  2. If len(self.messages) > self.max_size, it removes the OLDEST message (index 0)
  3. Print a notification when a message is dropped:
       print(f"[Buffer] Dropped oldest: '{content_of_dropped[:30]}...'")

Also complete:
  - get_as_text(): format messages as "ROLE: content\\n" joined string
  - clear(): set self.messages = []
""")


class ConversationBuffer:
    """Stores the most recent N messages."""

    def __init__(self, max_size: int = 5):
        """max_size: how many messages to keep"""
        self.messages = []
        self.max_size = max_size

    def add(self, role: str, content: str):
        """
        TODO: Add a message. Drop the oldest if over max_size.
        role:    "user" or "assistant"
        content: the message text
        """
        pass  # DELETE THIS and implement

    def get_all(self) -> list:
        """Return a copy of all messages."""
        return list(self.messages)

    def get_as_text(self) -> str:
        """
        TODO: Format all messages as a readable string.
        Each line: "ROLE: content"
        Join lines with newline character.
        """
        pass  # DELETE THIS and implement

    def clear(self):
        """TODO: Clear all messages."""
        pass  # DELETE THIS and implement

    def __len__(self):
        return len(self.messages)


# Test your ConversationBuffer
print("Testing ConversationBuffer (max_size=3):")
buf = ConversationBuffer(max_size=3)

test_messages = [
    ("user",      "Hello, my name is Gourav."),
    ("assistant", "Hi Gourav! How can I help?"),
    ("user",      "I am learning Python and LLMs."),
    ("assistant", "Great! Python is a wonderful language for AI."),  # Should drop first message
    ("user",      "What was my first message?"),                       # Should drop second
]

for role, content in test_messages:
    print(f"\nAdding [{role}]: {content[:50]}")
    buf.add(role, content)
    print(f"Buffer size: {len(buf)}")

print("\nBuffer contents:")
if buf.get_as_text:
    print(buf.get_as_text())

print("\nClearing buffer...")
buf.clear()
print(f"Buffer size after clear: {len(buf)}")


# ===========================================================================
# EXERCISE 2: LongTermMemory
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 2: Complete LongTermMemory")
print("=" * 65)

print("""
TODO: Complete LongTermMemory.remember() and recall().

  remember(fact: str, tags: list = None):
    - Create a dict: {"fact": fact, "timestamp": <now>, "tags": tags or []}
    - Append it to self.memories
    - Print: f"[Memory] Saved: '{fact[:50]}'"

  recall(query: str, top_k: int = 3) -> list:
    - For each memory, count how many words from the query appear in the fact text
    - Sort memories by that count (highest first)
    - Return the top_k fact strings (just the "fact" field, not the whole dict)
    - Only return memories with at least 1 word in common with the query

HINT for recall:
  query_words = set(query.lower().split())
  for mem in self.memories:
      fact_words = set(mem["fact"].lower().split())
      overlap = len(query_words & fact_words)   # & = intersection (common elements)
""")


class LongTermMemory:
    """Stores facts permanently with keyword-based recall."""

    def __init__(self):
        self.memories = []    # List of {"fact": ..., "timestamp": ..., "tags": [...]}

    def remember(self, fact: str, tags: list = None):
        """
        TODO: Save a fact to memory.
        """
        pass  # DELETE THIS and implement

    def recall(self, query: str, top_k: int = 3) -> list:
        """
        TODO: Return the top_k most relevant facts for the query.
        """
        pass  # DELETE THIS and implement

    def count(self) -> int:
        return len(self.memories)

    def get_all_facts(self) -> list:
        return [m["fact"] for m in self.memories]


# Test LongTermMemory
print("Testing LongTermMemory:")
ltm = LongTermMemory()

facts_to_save = [
    ("User's name is Gourav Dwivedi.",              ["name", "user"]),
    ("User works at Blackline as a .NET developer.", ["work", "job", "dotnet"]),
    ("User is learning Python and LLMs.",            ["learning", "python"]),
    ("User prefers C# analogies in explanations.",   ["preference", "csharp"]),
    ("User completed Module 10 on Vector Databases.", ["progress", "module10"]),
]

print("\nSaving facts:")
for fact, tags in facts_to_save:
    ltm.remember(fact, tags=tags)

print(f"\nTotal memories: {ltm.count()}")

print("\nRecall tests:")
recall_queries = [
    "What is the user's name?",
    "Where does the user work?",
    "What is the user's preferred learning style?",
    "What modules has the user completed?",
]

for query in recall_queries:
    results = ltm.recall(query, top_k=2)
    print(f"\n  Query: {query}")
    if results:
        for r in results:
            print(f"    -> {r}")
    else:
        print("    -> No results found.")


# ===========================================================================
# EXERCISE 3: Build MemoryAgent
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 3: Build MemoryAgent")
print("=" * 65)

print("""
TODO: Complete the MemoryAgent class.

  It should have:
  - self.short_term = ConversationBuffer(max_size=6)
  - self.long_term  = LongTermMemory()

  chat(user_message: str) -> str should:
    1. Check if the message contains a "save trigger" (words like "my name is",
       "i work at", "i prefer", "i am a", "i completed")
       If yes: call self.long_term.remember(user_message)

    2. Retrieve relevant memories: self.long_term.recall(user_message, top_k=2)

    3. Add the user message to short-term: self.short_term.add("user", user_message)

    4. Generate a response (use the _respond() method provided below)

    5. Add the response to short-term: self.short_term.add("assistant", response)

    6. Return the response
""")


class MemoryAgent:
    """Agent with short-term + long-term memory."""

    SAVE_TRIGGERS = ["my name is", "i work at", "i am a", "i prefer",
                     "i completed", "i like", "i learned"]

    def __init__(self):
        # TODO: Initialize self.short_term and self.long_term
        pass  # DELETE THIS and add initialization

    def _should_save(self, message: str) -> bool:
        """Returns True if the message contains a fact worth saving."""
        msg_lower = message.lower()
        return any(trigger in msg_lower for trigger in self.SAVE_TRIGGERS)

    def _respond(self, user_message: str, memories: list) -> str:
        """Generate a simulated response. PROVIDED -- do not modify."""
        msg = user_message.lower()

        if "name" in msg and memories:
            for m in memories:
                if "name" in m.lower():
                    return f"Yes, I remember -- {m}"

        if any(w in msg for w in ["work", "job", "company"]) and memories:
            for m in memories:
                if "work" in m.lower() or "blackline" in m.lower():
                    return f"I have this in my notes: {m}"

        if "remember" in msg or "recall" in msg:
            if memories:
                return f"I remember: {'; '.join(memories[:2])}"
            return "I don't have specific memories about that yet."

        if memories:
            return f"Based on what I know about you: {memories[0]}"

        return f"I don't have information about that. You can tell me and I'll remember!"

    def chat(self, user_message: str) -> str:
        """
        TODO: Process a message using both memory types.
        Follow the steps described in the exercise description above.
        """
        pass  # DELETE THIS and implement

    def status(self):
        """Print memory status. PROVIDED -- do not modify."""
        print(f"\n[Memory Status]")
        print(f"  Short-term messages: {len(self.short_term)}")
        print(f"  Long-term facts:     {self.long_term.count()}")


# Test MemoryAgent
print("\nTesting MemoryAgent:")
agent = MemoryAgent()

if hasattr(agent, 'chat') and agent.chat is not None:
    # Session 1: tell the agent things to remember
    print("\n--- Session 1: Introduction ---")
    agent.chat("Hi, my name is Gourav Dwivedi.")
    agent.chat("I work at Blackline as a senior developer.")
    agent.chat("I prefer code examples with C# comparisons.")

    agent.status()

    # Session 2: test if it remembers
    print("\n--- Session 2: Test Recall ---")
    if hasattr(agent, 'short_term'):
        agent.short_term.clear()    # Simulate new session (clear short-term)
        print("(Cleared short-term memory -- new session)")

    agent.chat("What is my name?")
    agent.chat("Where do I work?")
    agent.chat("Do you remember my preferences?")


# ===========================================================================
# EXERCISE 4: Multi-Session Test
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 4: Multi-Session Memory Test")
print("=" * 65)

print("""
TODO: Simulate 3 separate sessions.
  In each session, clear the short-term memory (to simulate app restart).
  Long-term memory should persist across all sessions.

  Session 1: Tell the agent your name and programming background.
  Session 2: Clear short-term. Ask the agent your name. Did it remember?
  Session 3: Clear short-term. Tell it about a new skill. Ask about all your skills.

  After all 3 sessions, print all long-term memories.

HINT:
  agent.short_term.clear()   <- clears short-term (simulates restart)
  agent.long_term persists   <- does NOT clear between sessions

Write your test code below (replace the TODOs):
""")

# TODO: Run 3 sessions here
# Example:
# agent4 = MemoryAgent()
#
# print("--- Session 1 ---")
# agent4.chat("My name is Gourav.")
# agent4.chat("I am a .NET developer.")
#
# print("\\n--- Session 2 (new session) ---")
# agent4.short_term.clear()
# agent4.chat("What is my name?")
#
# print("\\n--- Session 3 (another new session) ---")
# agent4.short_term.clear()
# agent4.chat("I learned vector databases this week.")
# agent4.chat("What have I learned recently?")
#
# print("\\n--- All Long-Term Memories ---")
# for fact in agent4.long_term.get_all_facts():
#     print(f"  - {fact}")


# ===========================================================================
# SOLUTION
# ===========================================================================

print("\n" + "=" * 65)
print("SOLUTION (uncomment to check)")
print("=" * 65)

"""
SOLUTION FOR EXERCISE 1 -- ConversationBuffer:

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_size:
            dropped = self.messages.pop(0)
            print(f"[Buffer] Dropped oldest: '{dropped['content'][:30]}...'")

    def get_as_text(self) -> str:
        lines = [f"{m['role'].upper()}: {m['content']}" for m in self.messages]
        return "\\n".join(lines)

    def clear(self):
        self.messages = []

SOLUTION FOR EXERCISE 2 -- LongTermMemory:

    def remember(self, fact: str, tags: list = None):
        self.memories.append({
            "fact": fact,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "tags": tags or []
        })
        print(f"[Memory] Saved: '{fact[:50]}'")

    def recall(self, query: str, top_k: int = 3) -> list:
        query_words = set(query.lower().split())
        scored = []
        for mem in self.memories:
            fact_words = set(mem["fact"].lower().split())
            tag_words  = set(" ".join(mem["tags"]).lower().split())
            overlap = len(query_words & (fact_words | tag_words))
            if overlap > 0:
                scored.append((overlap, mem["fact"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:top_k]]

SOLUTION FOR EXERCISE 3 -- MemoryAgent:

    def __init__(self):
        self.short_term = ConversationBuffer(max_size=6)
        self.long_term  = LongTermMemory()

    def chat(self, user_message: str) -> str:
        print(f"\\n[USER] {user_message}")

        if self._should_save(user_message):
            self.long_term.remember(user_message)

        memories = self.long_term.recall(user_message, top_k=2)
        self.short_term.add("user", user_message)
        response = self._respond(user_message, memories)
        self.short_term.add("assistant", response)

        print(f"[AGENT] {response}")
        return response
"""

print("See SOLUTION block above.")
print("=" * 65)
print("END OF EXERCISE 04")
print("=" * 65)

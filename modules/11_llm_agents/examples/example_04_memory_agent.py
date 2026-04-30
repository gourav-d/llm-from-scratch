"""
Example 04: Memory Agent -- Remembering Across Conversations
============================================================

GLOSSARY
--------
Short-Term Memory:
  The current conversation messages stored in a buffer.
  Fast, in RAM, lost when session ends.
  Like a whiteboard that gets erased at end of day.

Long-Term Memory:
  Facts stored persistently using a vector database (or simple file here).
  Survives between sessions. Can hold unlimited facts.
  Like a notebook that never gets erased.

ConversationBuffer:
  A list that holds the last N messages.
  When it fills up, oldest messages are dropped.
  In C#: like Queue<Message> with a max-size Enqueue policy.

MemoryStore:
  A simple in-memory store for this example.
  In production: would use ChromaDB (Module 10) + real embeddings.
  Here we use a simple list + keyword search to keep things clear.

Memory Retrieval:
  Finding relevant past memories for the current question.
  Here: keyword matching (simple).
  In production: vector similarity search (Module 10).

Episodic Memory:
  Memory of specific events. "On Monday, user asked about Python."
  Stored with a timestamp.

Fact Memory:
  Memory of general facts the user told the agent.
  "User's name is Gourav." "User prefers dark mode."

WHAT THIS EXAMPLE SHOWS
------------------------
Part A: Conversation buffer (short-term memory) in action
Part B: Long-term memory store with save + recall
Part C: Full memory agent that uses BOTH types

LIBRARIES NEEDED
-----------------
  None (pure Python built-ins only)
"""

import json          # For pretty printing
import datetime      # For timestamps on memories

print("=" * 65)
print("EXAMPLE 04: Memory Agent")
print("=" * 65)


# ==============================================================================
# PART A: Short-Term Memory (Conversation Buffer)
# ==============================================================================

print("\n" + "=" * 65)
print("PART A: Short-Term Memory (Conversation Buffer)")
print("=" * 65)

print("""
Short-term memory = the list of recent messages.
The agent always reads the last N messages before responding.
When the buffer is full, the OLDEST message is dropped.

This is how ChatGPT "remembers" what you said earlier in the same conversation.
""")


class ConversationBuffer:
    """
    Stores the most recent messages from the current conversation.
    When it reaches max_messages, the oldest message is dropped.

    C# equivalent:
      A Queue<Message> where Enqueue() also calls Dequeue() when full.

      public void Add(Message msg) {
          _queue.Enqueue(msg);
          if (_queue.Count > _maxSize) _queue.Dequeue();
      }
    """

    def __init__(self, max_messages: int = 6):
        """
        max_messages: how many messages to keep (older ones are dropped)
        """
        self.messages = []            # List of {"role": "...", "content": "..."} dicts
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        """
        Add a new message to the buffer.
        role:    "user", "assistant", or "system"
        content: the message text
        """
        self.messages.append({"role": role, "content": content})  # Add to the end

        # If over the limit, drop the oldest message (index 0)
        if len(self.messages) > self.max_messages:
            dropped = self.messages.pop(0)    # Remove from the front
            print(f"   [Buffer full -- dropped oldest: '{dropped['content'][:40]}...']")

    def get_all(self) -> list:
        """Return all messages (copy to prevent modification)."""
        return list(self.messages)    # Return a copy

    def get_context_text(self) -> str:
        """Format messages as a readable conversation string."""
        lines = []
        for msg in self.messages:
            role_label = msg["role"].upper()
            lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        """Erase all messages (start fresh)."""
        self.messages = []

    def __len__(self):
        """Allow len(buffer) to return the number of messages."""
        return len(self.messages)


# Test the conversation buffer
print("Creating buffer with max_messages=4 (small for demo):")
buffer = ConversationBuffer(max_messages=4)

conversations = [
    ("user",      "Hi! My name is Gourav and I work at Blackline."),
    ("assistant", "Hello Gourav! Nice to meet you. How can I help?"),
    ("user",      "I am a .NET developer learning Python."),
    ("assistant", "Great! Python has a lot in common with C#. What would you like to learn?"),
    ("user",      "Can you explain list comprehensions?"),          # Buffer will be full here
    ("assistant", "Sure! list comprehensions are like LINQ in C#: [x*2 for x in range(5)]"),
]

print()
for role, content in conversations:
    print(f"Adding [{role}]: {content[:60]}...")
    buffer.add(role, content)
    print(f"  Buffer now has {len(buffer)} messages.")

print("\nCurrent buffer contents:")
print(buffer.get_context_text())

print("""
Notice: the first message ("Hi! My name is Gourav...") was dropped
when the buffer reached its limit. That is the limitation of short-term memory.
Solution: save important facts to LONG-TERM memory before they fall off.
""")


# ==============================================================================
# PART B: Long-Term Memory Store
# ==============================================================================

print("=" * 65)
print("PART B: Long-Term Memory Store")
print("=" * 65)

print("""
Long-term memory stores IMPORTANT FACTS permanently.
The agent decides which facts to save (not every message -- just key info).
When answering a question, the agent RETRIEVES relevant memories.

In production: ChromaDB + real embeddings (Module 10).
Here: simple list + keyword search (so you can focus on the CONCEPT).
""")


class SimpleLongTermMemory:
    """
    A simple long-term memory store.
    Stores facts with timestamps.
    Retrieves facts by keyword matching.

    In production: replace this with ChromaDB from Module 10.
    The interface (remember/recall) stays the same.

    C# analogy:
      Like a simple Dictionary<string, MemoryEntry> where
      retrieval is text search instead of key lookup.
    """

    def __init__(self):
        self.memories = []    # List of {"fact": ..., "timestamp": ..., "tags": [...]}

    def remember(self, fact: str, tags: list = None):
        """
        Save a fact to long-term memory.
        fact:  the fact to remember (plain English sentence)
        tags:  keywords that help find this memory later (optional)
        """
        if tags is None:
            tags = []

        memory = {
            "fact":      fact,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),  # When saved
            "tags":      tags                                                  # Keywords
        }
        self.memories.append(memory)   # Add to the store
        print(f"  [MEMORY SAVED] {fact[:60]}...")

    def recall(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve the most relevant memories for a query.
        query: the current question or context
        top_k: max number of memories to return

        Here we use simple keyword overlap.
        In production: vector similarity search (cosine similarity from Module 10).

        Returns: list of relevant fact strings
        """
        query_words = set(query.lower().split())    # Words in the query

        # Score each memory by how many query words it contains
        scored = []
        for mem in self.memories:
            fact_words = set(mem["fact"].lower().split())     # Words in the fact
            tag_words  = set(" ".join(mem["tags"]).lower().split())  # Words in tags

            # Count overlap between query words and fact+tag words
            overlap = len(query_words & (fact_words | tag_words))   # & = intersection

            if overlap > 0:                          # Only include memories with any overlap
                scored.append((overlap, mem["fact"]))  # (score, fact text)

        # Sort by score (highest overlap first) and return top_k
        scored.sort(key=lambda x: x[0], reverse=True)    # Sort descending by score
        return [fact for score, fact in scored[:top_k]]   # Return just the fact text

    def get_all(self) -> list:
        """Return all stored memories."""
        return [m["fact"] for m in self.memories]

    def count(self) -> int:
        """How many memories are stored."""
        return len(self.memories)


# Test long-term memory
print("\nSaving facts to long-term memory:")
ltm = SimpleLongTermMemory()

ltm.remember("User's name is Gourav Dwivedi.",              tags=["name", "user", "gourav"])
ltm.remember("User works at Blackline as a .NET developer.", tags=["work", "job", "blackline", "dotnet"])
ltm.remember("User prefers code examples with C# analogies.", tags=["preference", "teaching", "csharp"])
ltm.remember("User is learning Python and LLMs from scratch.", tags=["learning", "python", "llm"])
ltm.remember("User completed Module 10 (Vector Databases).",  tags=["progress", "module10", "completed"])

print(f"\nTotal memories stored: {ltm.count()}")

print("\nRetrieving relevant memories:")
test_queries = [
    "What is the user's name?",
    "What programming background does the user have?",
    "What learning topics is the user interested in?",
    "What has the user completed so far?",
]

for query in test_queries:
    results = ltm.recall(query, top_k=2)
    print(f"\n  Query: {query}")
    if results:
        for fact in results:
            print(f"    -> {fact}")
    else:
        print("    -> No relevant memories found.")


# ==============================================================================
# PART C: Full Memory Agent (Short-Term + Long-Term)
# ==============================================================================

print("\n" + "=" * 65)
print("PART C: Full Memory Agent")
print("=" * 65)

print("""
The full memory agent uses BOTH types of memory:
  - Short-term: reads recent conversation messages before responding
  - Long-term: saves important facts; retrieves them when relevant

This is how a real personal assistant works:
  - Remembers what you said earlier today (short-term)
  - Remembers things you told it weeks ago (long-term)
""")


class MemoryAgent:
    """
    An agent with both short-term and long-term memory.

    On each turn:
    1. Retrieves relevant long-term memories
    2. Reads short-term conversation history
    3. Responds using both sources
    4. Decides what new facts to save to long-term memory
    """

    def __init__(self, buffer_size: int = 10):
        """
        buffer_size: max messages in short-term buffer
        """
        self.short_term = ConversationBuffer(max_messages=buffer_size)  # Short-term
        self.long_term  = SimpleLongTermMemory()                         # Long-term

        # Facts that trigger saving to long-term memory
        self.save_triggers = [
            "my name is", "i am a", "i work at", "i prefer", "i like",
            "i completed", "i learned", "my goal is", "i am from"
        ]

    def _should_save(self, message: str) -> bool:
        """
        Decide if a user message contains a fact worth saving to long-term memory.
        Returns True if the message contains a personal fact.
        """
        msg_lower = message.lower()
        return any(trigger in msg_lower for trigger in self.save_triggers)

    def _extract_tags(self, fact: str) -> list:
        """Extract simple tags from a fact sentence."""
        # Remove common words, keep the meaningful ones
        stopwords = {"my", "i", "am", "is", "a", "an", "the", "at", "in", "to", "of"}
        words = fact.lower().split()
        tags = [w.strip(".,!?") for w in words if w not in stopwords and len(w) > 2]
        return tags[:5]    # Keep max 5 tags

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        Uses both short-term and long-term memory.

        user_message: what the user said
        Returns: the agent's response
        """
        print(f"\n[USER] {user_message}")

        # Step 1: Check if this message has a fact to save to long-term memory
        if self._should_save(user_message):
            tags = self._extract_tags(user_message)
            self.long_term.remember(user_message, tags=tags)

        # Step 2: Retrieve relevant long-term memories for this message
        relevant_memories = self.long_term.recall(user_message, top_k=3)

        # Step 3: Add user message to short-term buffer
        self.short_term.add("user", user_message)

        # Step 4: Build context (what the agent "knows" right now)
        context_parts = []

        if relevant_memories:
            context_parts.append("RELEVANT MEMORIES:")
            for mem in relevant_memories:
                context_parts.append(f"  - {mem}")

        context_parts.append("\nRECENT CONVERSATION:")
        context_parts.append(self.short_term.get_context_text())

        full_context = "\n".join(context_parts)

        # Step 5: Generate a response (simulated here -- real agent calls LLM)
        response = self._simulated_response(user_message, relevant_memories)

        # Step 6: Add response to short-term buffer
        self.short_term.add("assistant", response)

        print(f"[AGENT] {response}")

        if relevant_memories:
            print(f"  (Used memories: {relevant_memories[:1]})")  # Show first memory used

        return response

    def _simulated_response(self, user_message: str, memories: list) -> str:
        """
        Simulates an LLM response using the user message and retrieved memories.
        In production: this would call Claude/GPT with the full context.
        """
        msg_lower = user_message.lower()

        # If asking about name and we have it in memory
        if "name" in msg_lower and memories:
            for mem in memories:
                if "name is" in mem.lower():
                    name = mem.split("name is")[-1].strip().rstrip(".")
                    return f"Your name is {name}, as I have recorded in my memory."

        # If asking about work/job
        if any(w in msg_lower for w in ["work", "job", "company"]) and memories:
            for mem in memories:
                if "work" in mem.lower() or "blackline" in mem.lower():
                    return f"According to my notes: {mem}"

        # If asking about progress/completed
        if any(w in msg_lower for w in ["completed", "done", "progress", "finished"]) and memories:
            for mem in memories:
                if "completed" in mem.lower() or "module" in mem.lower():
                    return f"I remember: {mem}"

        # If asking "do you remember"
        if "remember" in msg_lower or "recall" in msg_lower:
            if memories:
                return f"Yes! I remember these relevant facts: {'; '.join(memories[:2])}"
            return "I don't have any relevant memories for that question yet."

        # Generic response
        if memories:
            return f"Based on what I know about you: {memories[0]}. How can I help?"
        return "I don't have specific information about that. Tell me more!"

    def show_memory_status(self):
        """Print the current state of both memory stores."""
        print("\n--- Memory Status ---")
        print(f"Short-term messages: {len(self.short_term)}")
        print(f"Long-term facts:     {self.long_term.count()}")
        print("\nAll long-term facts:")
        for i, fact in enumerate(self.long_term.get_all(), 1):
            print(f"  {i}. {fact}")


# Simulate a conversation with the memory agent
print("\nSimulating a conversation with the memory agent:\n")
memory_agent = MemoryAgent(buffer_size=6)

# Session 1: User introduces themselves
print("--- SESSION 1: Introduction ---")
memory_agent.chat("Hi, my name is Gourav and I am a .NET developer.")
memory_agent.chat("I work at Blackline, a financial software company.")
memory_agent.chat("I prefer code examples with C# comparisons when learning Python.")
memory_agent.chat("I completed Module 10 about Vector Databases.")

print("\n")
memory_agent.show_memory_status()

# Session 2: User comes back and tests memory recall
print("\n--- SESSION 2: Testing Memory Recall ---")
memory_agent.short_term.clear()       # Clear short-term (simulating new session)
print("(Short-term memory cleared -- simulating new session)")

memory_agent.chat("What is my name?")
memory_agent.chat("Where do I work?")
memory_agent.chat("What have I completed in my learning?")
memory_agent.chat("Do you remember my preferences?")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("SUMMARY - Memory Agent")
print("=" * 65)

print("""
WHAT WE BUILT:
  - ConversationBuffer: short-term memory (last N messages)
  - SimpleLongTermMemory: fact storage with keyword retrieval
  - MemoryAgent: uses BOTH, saves new facts, recalls relevant ones

KEY INSIGHTS:
  1. Short-term: fast, limited, lost when session ends.
  2. Long-term: persistent, unlimited, needs retrieval step.
  3. Agent decides WHAT to save (not every message).
  4. Retrieval uses similarity (vector search in production, keywords here).
  5. Clear separation: short-term = context window, long-term = database.

PRODUCTION UPGRADE:
  Replace SimpleLongTermMemory with ChromaDB (Module 10):
  - ltm.remember() -> chromadb_collection.add(embedding, fact)
  - ltm.recall()   -> chromadb_collection.query(embed(query))
  Same interface, much better search quality.

C# ANALOGY:
  Short-term = Queue<Message> in RAM (IMemoryCache in ASP.NET Core)
  Long-term  = a proper database (SQL or vector DB) with full-text search

NEXT EXAMPLE (05):
  Multi-agent system -- multiple specialized agents working together.
  An orchestrator that delegates tasks to expert workers.
""")

print("=" * 65)
print("END OF EXAMPLE 04")
print("=" * 65)

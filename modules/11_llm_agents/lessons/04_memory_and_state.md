# Lesson 04: Memory and State

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain the difference between short-term and long-term agent memory
2. Describe what happens when the context window fills up
3. Implement a simple conversation buffer (short-term memory)
4. Explain how a vector database provides long-term memory (links to Module 10)
5. Describe when to use each type of memory

---

## GLOSSARY

```
Context Window:
  The maximum amount of text the LLM can read in one call.
  Like RAM -- it is fast but limited in size.
  GPT-4: 128,000 tokens. Claude: 200,000 tokens.
  Everything the agent needs to reason about MUST fit in the context window.

Token:
  Roughly 0.75 words. "Hello world" = 2 tokens. One page of text = ~750 tokens.
  Every call to the LLM uses tokens -- both input (what you send) and output (what it says).

Short-Term Memory:
  The current conversation history stored in the context window.
  Fast to access, but limited in size and lost when the session ends.
  Like a human's working memory -- only holds what is happening right now.

Long-Term Memory:
  A persistent store of past knowledge (vector database, file, SQL database).
  Survives between sessions. Can store millions of facts.
  Slower to access (requires retrieval/search step).
  Like a human's episodic memory -- can remember things from weeks ago.

Memory Retrieval:
  Finding the relevant memories for the current situation.
  For vector DBs: similarity search (find memories similar to the current query).
  For SQL: exact lookup (find memory WHERE user_id = X).

Summarization:
  When the context window is getting full, the agent compresses old messages
  into a shorter summary to free up space while keeping key information.

Episodic Memory:
  Memory of specific past events. "Last Tuesday you asked me about Python decorators."
  Stored as facts in a vector database.

Semantic Memory:
  General knowledge. "Python decorators are like C# attributes."
  Also stored in the vector DB, but represents learned facts, not specific events.

Buffer:
  A list that holds the last N messages. Older messages are dropped when it fills up.
  The simplest form of short-term memory management.
```

---

## Part 1: The Problem Without Memory

Without memory, every conversation starts fresh.

```
Session 1 (Monday):
  User:  "My name is Gourav and I work at Blackline."
  Agent: "Nice to meet you, Gourav!"

Session 2 (Tuesday):
  User:  "What is my name?"
  Agent: "I'm sorry, I don't know your name." <- FORGETS EVERYTHING
```

This is like talking to someone with amnesia every single day.
They are perfectly smart during the conversation, but remember nothing afterward.

For a useful agent, we need memory that persists.

---

## Part 2: The Two Types of Memory

```
                AGENT MEMORY
                     |
         +-----------+-----------+
         |                       |
   SHORT-TERM                LONG-TERM
   (Context Window)           (External Store)
         |                       |
   - Current session          - Persists forever
   - Last N messages           - Vector database (Module 10)
   - Fast (already in RAM)    - SQL database
   - Lost when session ends   - File system
   - Limited: ~200K tokens    - Unlimited size
                               - Requires search step
```

### When to Use Each

| Situation                              | Memory Type    |
|----------------------------------------|----------------|
| User's last 5 messages                | Short-term     |
| What user said in a previous session  | Long-term      |
| Tool call results from this session   | Short-term     |
| User's preferences (always use dark mode) | Long-term   |
| Current step in a multi-step task     | Short-term     |
| Historical data for analysis          | Long-term      |

---

## Part 3: Short-Term Memory (Conversation Buffer)

The simplest implementation: a list of message dictionaries.

```python
class ConversationBuffer:
    """
    Stores the last N messages from the current conversation.
    Like a queue -- when it fills up, oldest messages are dropped.

    C# analogy: Queue<Message> with a max size, where Dequeue happens
                automatically when Enqueue would exceed the limit.
    """

    def __init__(self, max_messages: int = 10):
        """
        max_messages: how many messages to keep. Older ones are dropped.
        """
        self.messages = []           # List of message dicts: [{"role": "user", "content": "..."}]
        self.max_messages = max_messages  # Maximum number of messages to keep

    def add(self, role: str, content: str):
        """
        Add a new message to the buffer.
        role:    "user", "assistant", or "tool"
        content: the message text
        """
        self.messages.append({"role": role, "content": content})  # Add to end

        # If we have too many messages, remove the oldest one
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)       # Remove the first (oldest) message
                                       # C#: messages.RemoveAt(0)

    def get_all(self) -> list:
        """Return all messages as a list (for sending to the LLM)."""
        return self.messages.copy()    # Return a copy to prevent external modification

    def get_as_text(self) -> str:
        """Format all messages as readable text (for debugging)."""
        lines = []
        for msg in self.messages:
            lines.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(lines)        # Join all lines with newlines

    def clear(self):
        """Remove all messages (start fresh)."""
        self.messages = []
```

### The Context Window Problem

When the buffer exceeds the LLM's context window, you have two options:

**Option A: Truncation** (simple but lossy)
```python
# Keep only the last N messages
while token_count(messages) > MAX_TOKENS:
    messages.pop(0)           # Drop oldest message
```
Problem: You lose information permanently.

**Option B: Summarization** (smarter)
```python
# When buffer is getting full, summarize old messages into a compact form
old_messages = messages[:5]          # The 5 oldest messages
summary = llm_summarize(old_messages)  # LLM condenses them into 1-2 sentences
messages = [{"role": "summary", "content": summary}] + messages[5:]  # Replace with summary
```
This preserves the key information while freeing up space.

---

## Part 4: Long-Term Memory (Vector Database)

From Module 10, you learned that a vector database stores information as vectors
and lets you find similar information quickly.

For agent memory, we use the SAME concept:
  - Each memory = a text fact stored with its vector embedding
  - To retrieve relevant memories: embed the current question, find similar vectors
  - Results: the most relevant past facts, returned for the LLM to use

```python
# HOW LONG-TERM MEMORY WORKS (pseudocode -- full implementation in Example 04)

class LongTermMemory:
    """
    Stores facts permanently. Survives between sessions.
    Uses a vector database for semantic retrieval.
    """

    def remember(self, fact: str):
        """
        Store a new fact permanently.
        fact: a string describing something to remember
        Examples: "User's name is Gourav"
                  "User prefers dark mode in the IDE"
                  "User mentioned they work at Blackline on 2026-04-30"
        """
        # Convert the fact text into a vector (embedding)
        vector = embedding_model.encode(fact)

        # Store: vector + original text in the vector database
        vector_db.add(
            id=generate_unique_id(),
            vector=vector,
            text=fact
        )

    def recall(self, query: str, top_k: int = 3) -> list:
        """
        Find the most relevant memories for the current question.
        query: the current context (what is happening now)
        top_k: how many memories to retrieve
        Returns: list of the most relevant past facts
        """
        # Convert the query into a vector
        query_vector = embedding_model.encode(query)

        # Search for the most similar stored memories
        results = vector_db.search(query_vector, top_k=top_k)

        # Return just the text of each result
        return [r["text"] for r in results]
```

### Memory Flow

```
New information arrives (e.g., "My name is Gourav")
  |
  v
Agent decides this is worth remembering
  |
  v
LongTermMemory.remember("User's name is Gourav")
  -> text -> embedding -> store in vector DB
  |
  (session ends... hours/days pass... new session starts)
  |
  v
New question: "What is my name?"
  |
  v
LongTermMemory.recall("What is the user's name?")
  -> embed query -> search vector DB -> find "User's name is Gourav"
  |
  v
LLM receives retrieved memory + user question
  |
  v
LLM answers: "Your name is Gourav."
```

---

## Part 5: What to Store in Memory

Not everything needs to go into long-term memory.
The agent should be selective.

### Store (Long-Term)
```
- User preferences: "User prefers concise answers"
- User context: "User is a .NET developer learning Python"
- Important facts: "User's project deadline is 2026-05-15"
- Past decisions: "User chose PyTorch over TensorFlow for their project"
```

### Do NOT Store (Keep in Short-Term Only)
```
- Routine conversation turns ("Hello", "Thanks", "Got it")
- Intermediate calculation steps (they are only relevant right now)
- Tool call results (unless they contain important facts)
- Error messages (unless they reveal a persistent issue)
```

### Rule of Thumb
Ask: "Would a human assistant need to remember this in 3 months?"
If yes -> long-term memory. If no -> short-term only.

---

## Part 6: Memory in C# Terms

```csharp
// Short-term memory: like a List<Message> with a max size
// (Clears when the application restarts)
public class ConversationBuffer {
    private readonly Queue<Message> _messages = new();
    private readonly int _maxSize;

    public void Add(Message msg) {
        _messages.Enqueue(msg);
        if (_messages.Count > _maxSize) {
            _messages.Dequeue();         // Drop oldest
        }
    }

    public IEnumerable<Message> GetAll() => _messages;
}

// Long-term memory: like a database with semantic search
// (Persists across application restarts)
public class LongTermMemory {
    private readonly IVectorDatabase _db;      // Vector DB (ChromaDB equivalent)
    private readonly IEmbeddingService _embed;  // Turns text into float[]

    public async Task Remember(string fact) {
        float[] vector = await _embed.Encode(fact);     // Convert to vector
        await _db.Upsert(Guid.NewGuid().ToString(), vector, fact);  // Store
    }

    public async Task<string[]> Recall(string query, int topK = 3) {
        float[] queryVector = await _embed.Encode(query);
        var results = await _db.Search(queryVector, topK);
        return results.Select(r => r.Text).ToArray();
    }
}
```

---

## Part 7: Memory Types Comparison Table

| Feature           | Short-Term (Buffer)    | Long-Term (Vector DB)      |
|-------------------|------------------------|----------------------------|
| Speed             | Instant (in RAM)       | Fast (indexed search)      |
| Capacity          | ~200K tokens           | Unlimited                  |
| Persistence       | Current session only   | Forever                    |
| Retrieval method  | All messages in order  | Semantic similarity search |
| When to use       | Current conversation   | Facts to keep forever      |
| C# equivalent     | Queue<Message>         | SQL/NoSQL database         |
| Implementation    | List of dicts          | ChromaDB + embeddings      |

---

## Key Takeaways

1. Without memory, every conversation starts fresh. Agents need memory to be truly useful.

2. Short-term memory = the context window. Fast, limited, lost when session ends.

3. Long-term memory = vector database (ChromaDB from Module 10). Persistent, unlimited.

4. Short-term: keep recent messages. Long-term: store important facts forever.

5. When context fills up: truncate (lossy) or summarize (smarter).

6. Retrieval from long-term memory = semantic search: embed the query, find similar vectors.

---

## Next

Lesson 05: Multi-Agent Systems
  - What happens when one agent is not enough?
  - How do multiple agents work together?
  - Orchestrator + specialist patterns
  - When to use multi-agent vs single agent

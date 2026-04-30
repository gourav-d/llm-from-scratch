"""
Project 02: Research Agent with Vector Memory
===============================================

WHAT THIS BUILDS
-----------------
A research agent that:
1. Accepts a research topic from the user
2. Searches multiple "sources" (simulated)
3. Stores all findings in a vector-like memory store
4. Answers follow-up questions by retrieving relevant findings
5. Generates a research report combining all findings

This bridges Module 10 (Vector Databases) and Module 11 (Agents).

LEARNING GOALS
--------------
- Use ReAct pattern for multi-step research
- Apply vector-like retrieval for memory (connects to Module 10)
- Build a complete research pipeline from scratch
- See how RAG (Retrieve-Augmented Generation) agents work

HOW TO RUN
----------
  python project_02_research_agent.py

STRUCTURE
---------
Part A: Research Agent demo (automated, no input needed)
Part B: Interactive Q&A based on research findings
        Set INTERACTIVE = True to enable

LIBRARIES NEEDED
-----------------
  None (pure Python -- vector math done with basic lists)

NOTE ON CHROMADB:
  Production version would use ChromaDB from Module 10.
  This version uses a simple cosine similarity implemented in pure Python
  to keep the focus on the AGENT pattern, not the DB setup.
"""

import math
import json
import datetime

print("=" * 65)
print("PROJECT 02: Research Agent with Vector Memory")
print("=" * 65)

INTERACTIVE = False


# ==============================================================================
# SIMPLE VECTOR MEMORY (ChromaDB-like, no external library)
# ==============================================================================

class SimpleVectorMemory:
    """
    A simplified vector memory store.
    Stores text + a simple TF-based "vector" for similarity search.
    This mimics what ChromaDB does (Module 10), using only Python built-ins.

    In production: replace this entire class with ChromaDB.
    The interface (add, query) stays exactly the same.
    """

    def __init__(self):
        self.entries = []    # List of {"id": ..., "text": ..., "vector": [...], "metadata": {...}}
        self._next_id = 1

    def _text_to_vector(self, text: str) -> dict:
        """
        Convert text to a simple word-frequency 'vector'.
        This is a very basic version of TF (term frequency).
        Real embeddings from Module 10 are much better.
        Returns: a dict of {word: frequency}
        """
        words = text.lower().split()
        freq = {}
        for word in words:
            clean = word.strip(".,!?;:\"'()[]")   # Remove punctuation
            if len(clean) > 2:                     # Skip short words
                freq[clean] = freq.get(clean, 0) + 1
        return freq

    def _similarity(self, vec1: dict, vec2: dict) -> float:
        """
        Compute similarity between two word-frequency vectors.
        Uses a simplified cosine similarity.
        Returns: float 0.0 to 1.0
        """
        if not vec1 or not vec2:
            return 0.0

        # Find common words
        common_words = set(vec1.keys()) & set(vec2.keys())
        if not common_words:
            return 0.0

        # Dot product: sum of (freq in vec1 * freq in vec2) for common words
        dot = sum(vec1[w] * vec2[w] for w in common_words)

        # Magnitudes
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)    # Cosine similarity

    def add(self, text: str, metadata: dict = None):
        """
        Store a text document in memory.
        text:     the text to store
        metadata: optional dict with extra info (source, date, topic, etc.)
        """
        entry = {
            "id":       self._next_id,
            "text":     text,
            "vector":   self._text_to_vector(text),   # Convert to searchable vector
            "metadata": metadata or {},
            "added_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        self.entries.append(entry)
        self._next_id += 1

    def query(self, question: str, top_k: int = 3) -> list:
        """
        Find the most relevant stored texts for a question.
        question: the search query
        top_k:    how many results to return
        Returns:  list of (similarity_score, text, metadata) tuples
        """
        query_vector = self._text_to_vector(question)

        scored = []
        for entry in self.entries:
            sim = self._similarity(query_vector, entry["vector"])
            scored.append((sim, entry["text"], entry["metadata"]))

        # Sort by similarity (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return only results with similarity > 0 (relevant results only)
        return [(s, t, m) for s, t, m in scored[:top_k] if s > 0]

    def count(self) -> int:
        return len(self.entries)


# ==============================================================================
# RESEARCH SOURCES (Simulated)
# ==============================================================================

# In production: replace with real web search API (Bing, Google, Serper, Tavily)
KNOWLEDGE_DATABASE = {
    "machine learning": [
        "Machine learning is a subset of AI where systems learn from data without explicit programming. "
        "Key types: supervised learning (labeled data), unsupervised (unlabeled), reinforcement (reward signals).",

        "Popular machine learning algorithms include: Linear Regression (predict values), "
        "Decision Trees (classify), Random Forest (ensemble of trees), Neural Networks (deep learning), "
        "Support Vector Machines (classification).",

        "Machine learning workflow: collect data, clean data, choose algorithm, train model, "
        "evaluate performance, deploy. The data quality is often more important than the algorithm choice.",
    ],
    "transformers": [
        "Transformer architecture was introduced in 'Attention Is All You Need' (Vaswani et al., 2017, Google). "
        "Uses self-attention mechanism instead of RNNs. Processes all tokens in parallel.",

        "Key components of Transformers: Self-Attention (tokens attend to all other tokens), "
        "Multi-Head Attention (multiple attention patterns simultaneously), "
        "Positional Encoding (inject position information), Feed-Forward layers (MLP per position).",

        "Transformers power all modern LLMs: GPT (OpenAI), BERT (Google), T5, "
        "Claude (Anthropic), LLaMA (Meta), Gemini (Google). They scale excellently with data and compute.",
    ],
    "llm agents": [
        "LLM Agents are AI systems that use language models to plan and take actions. "
        "Core components: Brain (LLM), Tools (functions to call), Memory, and Planning loop.",

        "ReAct (Reason + Act) is the dominant agent pattern. Loop: Thought -> Action -> Observation -> repeat. "
        "Agent keeps a scratchpad of all steps. Stops when confident enough to answer.",

        "Popular agent frameworks: LangChain, AutoGen (Microsoft), CrewAI, Claude Claude with tools, "
        "OpenAI Assistants API, Semantic Kernel (Microsoft). All implement the same core patterns.",

        "Agent memory types: Short-term (context window, ~200K tokens), "
        "Long-term (vector database, unlimited, persists forever). "
        "ChromaDB, Pinecone, Weaviate, pgvector are popular vector databases for agent memory.",
    ],
    "python": [
        "Python is a high-level, interpreted programming language created by Guido van Rossum, "
        "first released in 1991. Known for its readable syntax and large ecosystem.",

        "Python is dominant in AI and machine learning due to libraries like: "
        "NumPy (math), Pandas (data), Matplotlib (charts), Scikit-learn (ML), "
        "PyTorch and TensorFlow (deep learning).",

        "Python for web development: Django (full-featured, batteries-included), "
        "FastAPI (modern, fast, auto-docs), Flask (minimalist, flexible). "
        "All are excellent choices depending on project complexity.",
    ],
    "rag": [
        "RAG (Retrieval-Augmented Generation) combines a retrieval system with an LLM. "
        "Step 1: Embed the query. Step 2: Find similar documents in vector DB. "
        "Step 3: Add documents to prompt. Step 4: LLM generates answer using retrieved context.",

        "RAG solves the LLM knowledge cutoff problem. Instead of relying on training data alone, "
        "the model retrieves current, relevant information at query time. "
        "Used in: enterprise chatbots, document Q&A, code search, legal research.",

        "RAG vs Fine-tuning: RAG is preferred when data changes frequently (live databases, docs). "
        "Fine-tuning is better for teaching the model a new skill or style. "
        "Many production systems use both: fine-tuned model + RAG for live data.",
    ],
}


def search_topic(topic: str, num_results: int = 2) -> list:
    """
    Search for information about a topic.
    Returns a list of text findings.
    """
    topic_lower = topic.lower()
    results = []

    for key, texts in KNOWLEDGE_DATABASE.items():
        if key in topic_lower or any(word in topic_lower for word in key.split()):
            results.extend(texts[:num_results])
            if results:
                break

    if not results:
        results = [f"No specific information found about '{topic}' in the knowledge base."]

    return results[:num_results]


# ==============================================================================
# RESEARCH AGENT
# ==============================================================================

class ResearchAgent:
    """
    A multi-step research agent that:
    1. Plans the research (what to search for)
    2. Searches multiple sources
    3. Stores findings in vector memory
    4. Answers questions by retrieving relevant findings
    5. Generates a research report

    This is ReAct + RAG combined.
    """

    def __init__(self):
        self.memory = SimpleVectorMemory()    # Where findings are stored
        self.research_log = []                # Complete log of all research steps
        self.current_topic = None             # What we're currently researching

    def _plan_research(self, topic: str) -> list:
        """
        Plan what sub-topics to research for a given main topic.
        In production: call an LLM to plan the research.
        Here: rule-based planning based on keywords.
        """
        base_searches = [topic]    # Always search the main topic

        # Add related sub-topic searches
        related = {
            "machine learning": ["ml algorithms", "ml workflow", "ml vs deep learning"],
            "transformer":      ["attention mechanism", "transformer components", "llm models"],
            "llm agent":        ["agent memory", "agent tools", "react pattern"],
            "python":           ["python for ai", "python web frameworks"],
            "rag":              ["rag vs fine tuning", "vector database for rag"],
        }

        topic_lower = topic.lower()
        for key, subs in related.items():
            if key in topic_lower:
                base_searches.extend(subs[:2])    # Add first 2 related topics
                break

        return base_searches

    def research(self, topic: str) -> str:
        """
        Conduct full research on a topic.
        Uses ReAct loop: Plan -> Search -> Store -> Repeat.
        Returns a summary of what was found.
        """
        self.current_topic = topic
        self.memory = SimpleVectorMemory()    # Fresh memory for new research
        self.research_log = []

        print(f"\n{'='*55}")
        print(f"RESEARCHING: {topic}")
        print(f"{'='*55}")

        # Step 1: Plan
        search_queries = self._plan_research(topic)
        print(f"\n[Plan] Will search {len(search_queries)} queries:")
        for q in search_queries:
            print(f"  - {q}")

        # Step 2: Search and Store
        total_findings = 0
        for i, query in enumerate(search_queries, 1):
            print(f"\n[Step {i}] Searching: '{query}'")

            findings = search_topic(query, num_results=2)

            for j, finding in enumerate(findings):
                # Store in vector memory with metadata
                self.memory.add(finding, metadata={
                    "source":  f"search:{query}",
                    "query":   query,
                    "finding": j + 1,
                    "step":    i,
                })
                total_findings += 1
                print(f"  Stored finding {total_findings}: {finding[:80]}...")

            self.research_log.append({
                "step":     i,
                "query":    query,
                "findings": findings
            })

        print(f"\n[Done] Research complete. Stored {total_findings} findings.")
        return f"Research on '{topic}' complete. {total_findings} findings stored in memory."

    def ask(self, question: str) -> str:
        """
        Answer a question based on stored research findings.
        Retrieves relevant findings from memory (RAG pattern).
        """
        if self.memory.count() == 0:
            return "No research has been done yet. Call research(topic) first."

        print(f"\n[Q&A] Question: {question}")

        # Retrieve relevant findings (RAG retrieval step)
        relevant = self.memory.query(question, top_k=3)

        if not relevant:
            return f"Could not find relevant research for: '{question}'"

        print(f"[Q&A] Retrieved {len(relevant)} relevant findings.")

        # Build the answer from retrieved findings
        answer_parts = []
        for rank, (score, text, metadata) in enumerate(relevant, 1):
            print(f"  Finding {rank} (similarity: {score:.3f}): {text[:60]}...")
            answer_parts.append(text)

        # Synthesize into a final answer
        combined = " | ".join(answer_parts[:2])    # Use top 2 findings
        answer = f"Based on the research: {combined[:400]}..."

        return answer

    def generate_report(self) -> str:
        """
        Generate a structured research report from all stored findings.
        """
        if not self.research_log:
            return "No research done. Run research(topic) first."

        lines = [
            f"# Research Report: {self.current_topic}",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}",
            f"Total findings: {self.memory.count()}",
            "",
            "## Summary",
        ]

        # Add top findings for each query
        for entry in self.research_log:
            lines.append(f"\n### {entry['query'].title()}")
            for finding in entry["findings"]:
                lines.append(f"- {finding[:150]}...")

        lines.append("\n## Conclusion")
        lines.append(
            f"This report covered {len(self.research_log)} research queries "
            f"on the topic of '{self.current_topic}'. "
            f"For deeper analysis, use the ask() method with specific questions."
        )

        return "\n".join(lines)


# ==============================================================================
# PART A: Research Demo
# ==============================================================================

print("\n" + "=" * 65)
print("PART A: Research Agent Demo")
print("=" * 65)

agent = ResearchAgent()

# Research 1: Machine Learning
print("\n--- RESEARCH 1: Machine Learning ---")
result = agent.research("machine learning")
print(f"\nResearch result: {result}")

print("\n--- Asking questions based on research ---")
questions = [
    "What are the types of machine learning?",
    "What algorithms are used in machine learning?",
    "How does the machine learning workflow work?",
]

for q in questions:
    answer = agent.ask(q)
    print(f"\nQ: {q}")
    print(f"A: {answer[:200]}...")

print("\n--- Generating Research Report ---")
report = agent.generate_report()
print(report[:600] + "...")


# Research 2: LLM Agents
print("\n" + "=" * 65)
print("\n--- RESEARCH 2: LLM Agents ---")
result2 = agent.research("llm agents")
print(f"\nResearch result: {result2}")

print("\n--- Asking follow-up questions ---")
followups = [
    "What is the ReAct pattern?",
    "What tools do LLM agents use?",
    "How does agent memory work?",
]

for q in followups:
    answer = agent.ask(q)
    print(f"\nQ: {q}")
    print(f"A: {answer[:200]}...")


# ==============================================================================
# PART B: Interactive Research
# ==============================================================================

print("\n" + "=" * 65)
print("PART B: Interactive Research Mode")
print("=" * 65)

if not INTERACTIVE:
    print("""
To use interactive research mode:
  1. Open this file
  2. Set INTERACTIVE = True
  3. Run again: python project_02_research_agent.py

Commands:
  research <topic>  -- Research a new topic
  ask <question>    -- Ask about the research
  report            -- Generate a research report
  status            -- Show memory status
  quit              -- Exit
""")
else:
    print("Interactive Research Mode. Commands: research <topic>, ask <question>, report, status, quit")

    interactive_agent = ResearchAgent()

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        elif user_input.lower().startswith("research "):
            topic = user_input[9:].strip()
            print(interactive_agent.research(topic))
        elif user_input.lower().startswith("ask "):
            question = user_input[4:].strip()
            print(interactive_agent.ask(question))
        elif user_input.lower() == "report":
            print(interactive_agent.generate_report())
        elif user_input.lower() == "status":
            print(f"Findings in memory: {interactive_agent.memory.count()}")
            print(f"Research queries done: {len(interactive_agent.research_log)}")
        else:
            print("Commands: research <topic>, ask <question>, report, status, quit")


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("SUMMARY - Research Agent")
print("=" * 65)

print("""
WHAT WE BUILT:
  A research agent that:
  - Plans research (what sub-topics to search)
  - Executes multi-step research (ReAct loop)
  - Stores findings in vector-like memory
  - Answers questions by retrieving relevant findings (RAG)
  - Generates a structured research report

PATTERNS USED:
  ReAct loop (Plan -> Search -> Store -> Answer)
  RAG (Retrieve-Augmented Generation from Module 10)
  Vector similarity search (simplified cosine similarity)
  Multi-step research (each query informs the next)

UPGRADE TO PRODUCTION:
  1. Replace search_topic() with real web search:
       pip install tavily-python    (or use SerpAPI, Bing Search API)

  2. Replace SimpleVectorMemory with ChromaDB (from Module 10):
       import chromadb
       from sentence_transformers import SentenceTransformer
       chroma_client = chromadb.Client()
       collection = chroma_client.create_collection("research")
       # Then: collection.add() and collection.query()

  3. Replace rule-based planning with real LLM planning:
       response = claude.messages.create(
           model="claude-haiku-4-5-20251001",
           messages=[{"role": "user", "content": f"Plan research queries for: {topic}"}]
       )

NEXT PROJECT (03):
  Code Review Agent -- automated code quality analysis.
  Uses the same agent pattern but specialized for software engineering.
""")

print("=" * 65)
print("END OF PROJECT 02")
print("=" * 65)

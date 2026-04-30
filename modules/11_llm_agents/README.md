# Module 11: LLM Agents

## What You Will Learn

In this module you will learn how to turn a passive LLM (one that just answers questions)
into an ACTIVE AGENT (one that plans, uses tools, and takes actions to solve problems).

---

## Why Agents Matter

A regular LLM chatbot:
  - You ask a question
  - It answers from memory
  - Done

An LLM Agent:
  - You give it a GOAL ("research the top 3 Python frameworks and write a summary")
  - It breaks the goal into steps
  - It CALLS tools (search the web, read files, run code)
  - It uses tool results to form the next step
  - It keeps going until the goal is achieved

This is the difference between a passive encyclopedia and an active assistant.

---

## Real-World Agents You Already Know

| Agent                  | What it does                                      |
|------------------------|---------------------------------------------------|
| GitHub Copilot         | Reads your code, suggests completions, runs tests |
| Devin (AI Developer)   | Plans features, writes code, debugs, opens PRs    |
| Claude Computer Use    | Controls a computer: clicks, types, browses       |
| ChatGPT with plugins   | Searches the web, runs Python, calls APIs         |
| AutoGPT                | Breaks any goal into tasks, executes autonomously |

All of these are built on the same core pattern you will learn here.

---

## Module Structure

```
11_llm_agents/
+-- lessons/
|   +-- 01_what_are_agents.md          <- What agents are and how they work
|   +-- 02_tool_use_function_calling.md <- How LLMs call external tools
|   +-- 03_react_pattern.md            <- The Reason-Act loop
|   +-- 04_memory_and_state.md         <- How agents remember things
|   +-- 05_multi_agent_systems.md      <- Multiple agents working together
|
+-- examples/
|   +-- example_01_simple_agent.py     <- Your first agent (no tools needed)
|   +-- example_02_tool_use.py         <- Agent that calls calculator + search
|   +-- example_03_react_agent.py      <- Full ReAct loop from scratch
|   +-- example_04_memory_agent.py     <- Agent with short and long-term memory
|   +-- example_05_multi_agent.py      <- Orchestrator + specialist agents
|
+-- exercises/
|   +-- exercise_01_basic_agent.py     <- Build a simple step planner
|   +-- exercise_02_tools.py           <- Add tools to an agent
|   +-- exercise_03_react.py           <- Implement ReAct loop
|   +-- exercise_04_memory.py          <- Add memory to an agent
|   +-- exercise_05_orchestration.py   <- Build a multi-agent pipeline
|
+-- projects/
    +-- project_01_personal_assistant.py  <- Full personal assistant agent
    +-- project_02_research_agent.py      <- Research agent with vector memory
    +-- project_03_code_review_agent.py   <- Automated code review agent
```

---

## Learning Path

```
Lesson 01: What Are Agents?
  |
  v
Lesson 02: Tools & Function Calling
  |
  v
Lesson 03: ReAct Pattern (Reason + Act loop)
  |
  v
Lesson 04: Memory & State
  |
  v
Lesson 05: Multi-Agent Systems
  |
  v
Examples 01-05 (run and study)
  |
  v
Exercises 01-05 (build yourself)
  |
  v
Projects 01-03 (capstone)
```

---

## Prerequisites

Before starting this module, you should have completed:
  - Module 04: Transformers (understand how LLMs work)
  - Module 05: Building Your LLM (understand text generation)
  - Module 08: Prompt Engineering (understand how to talk to LLMs)
  - Module 10: Vector Databases (needed for memory agent and research agent)

---

## Libraries Used

```
No external libraries required for most examples (they use simulation).

For projects using real LLM APIs:
  pip install anthropic          <- Anthropic (Claude) API client
  pip install openai             <- OpenAI (GPT) API client (optional)

For memory projects:
  pip install chromadb           <- Vector database (from Module 10)
  pip install sentence-transformers <- For generating real embeddings
```

All simulated examples run with ZERO external API keys.
The projects show how to connect to real APIs.

---

## Key Concepts

```
Agent:
  An LLM that can PLAN and TAKE ACTIONS to achieve a goal.
  Not just answer questions -- actually DO things.

Tool:
  A function the agent can call. Examples: calculator, web search,
  file reader, database query, API call.

ReAct Pattern:
  Reason -> Act -> Observe -> Reason -> Act -> Observe -> ... -> Answer
  The agent THINKS about what to do, DOES it, SEES the result, then thinks again.

Memory:
  Short-term: the current conversation (what was said in this session)
  Long-term:  a vector database of past knowledge (persists between sessions)

Orchestrator:
  A master agent that breaks a big task into smaller sub-tasks
  and assigns them to specialist agents.
```

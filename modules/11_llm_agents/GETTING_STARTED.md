# Getting Started -- Module 11: LLM Agents

## Quick Setup

No external libraries needed for most examples.
All examples run with pure Python.

```bash
# Navigate to this module
cd modules/11_llm_agents

# Run any example
python examples/example_01_simple_agent.py
python examples/example_02_tool_use.py
python examples/example_03_react_agent.py
python examples/example_04_memory_agent.py
python examples/example_05_multi_agent.py

# Run exercises (complete the TODOs)
python exercises/exercise_01_basic_agent.py
python exercises/exercise_02_tools.py
python exercises/exercise_03_react.py
python exercises/exercise_04_memory.py
python exercises/exercise_05_orchestration.py

# Run projects (full applications)
python projects/project_01_personal_assistant.py
python projects/project_02_research_agent.py
python projects/project_03_code_review_agent.py
```

## Optional: Real LLM API (for Part B sections)

```bash
# Install Anthropic SDK
pip install anthropic

# Set API key (Windows)
set ANTHROPIC_API_KEY=your-key-here

# Get a free key at: https://console.anthropic.com/
```

## Recommended Order

1. Read all 5 lessons first (lessons/ folder)
2. Run examples 01-05 to see the patterns in action
3. Complete exercises 01-05 (fill in the TODOs)
4. Build projects 01-03 (full applications)

## What Each File Does

| File                              | Teaches                              |
|-----------------------------------|--------------------------------------|
| lessons/01_what_are_agents.md     | Core concepts: brain, tools, loop    |
| lessons/02_tool_use_function_calling.md | How LLMs call tools            |
| lessons/03_react_pattern.md       | Thought -> Action -> Observation     |
| lessons/04_memory_and_state.md    | Short-term + long-term memory        |
| lessons/05_multi_agent_systems.md | Orchestrator + specialists           |
| examples/01_simple_agent.py       | Your first agent loop                |
| examples/02_tool_use.py           | Multiple tools + parallel calls      |
| examples/03_react_agent.py        | Full ReAct loop with tracing         |
| examples/04_memory_agent.py       | Conversation buffer + fact store     |
| examples/05_multi_agent.py        | Pipeline, parallel, orchestrator     |
| exercises/01-05                   | Practice -- complete the TODOs       |
| projects/01_personal_assistant.py | Full assistant with memory + tasks   |
| projects/02_research_agent.py     | ReAct + RAG research system          |
| projects/03_code_review_agent.py  | Multi-agent code analyzer            |

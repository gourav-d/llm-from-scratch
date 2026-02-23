# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a **"Learn LLM from Scratch"** educational project for a .NET developer learning Python and Large Language Models simultaneously.

## Student Context
- **Background**: .NET developer with no Python experience
- **Goal**: Learn to build LLMs from scratch while mastering Python
- **Learning Style**: Prefers simple explanations, visual aids, hands-on labs, and structured quizzes

## Teaching Approach

### Code Explanations
- Explain EVERY line of Python code in detail
- Compare Python concepts to C#/.NET equivalents when relevant
- Use layman language, avoid jargon without explanation
- Break down complex concepts into simple steps
- Use bullet points and simple formatting over long paragraphs

### Documentation Standards
- Document everything, including small examples
- Add diagrams using ASCII art or mermaid syntax where helpful
- Explain all libraries and their purposes before using them
- Include practical examples that relate to real-world scenarios

### Learning Structure
Each module should contain:
1. **Concept Overview** - What and why
2. **Python Basics** - Language features used
3. **Code Examples** - With line-by-line explanations
4. **Visual Aids** - Diagrams where applicable
5. **Quiz Questions** - Multiple choice and short answer
6. **Lab Exercises** - Hands-on coding with solutions
7. **Summary** - Key takeaways

### Python for .NET Developers
When introducing Python concepts, relate them to C#:
- Classes and OOP → Similar but simpler syntax
- List comprehensions → LINQ equivalents
- Decorators → Attributes
- duck typing → var/dynamic
- Packages → NuGet packages
- Virtual environments → Project-specific dependencies

## Project Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Jupyter for interactive learning
pip install jupyter notebook
```

### Running Code
```bash
# Run Python scripts
python module_name/example.py

# Run Jupyter notebooks (recommended for learning)
jupyter notebook

# Run specific module
python -m module_name
```

### Testing
```bash
# Run all tests
python -m pytest

# Run specific module tests
python -m pytest tests/test_module_name.py

# Run with verbose output
python -m pytest -v
```

## Repository Structure

```
/
├── modules/              # Learning modules in sequence
│   ├── 01_python_basics/
│   ├── 02_numpy_fundamentals/
│   ├── 03_neural_networks/
│   ├── 04_transformers/
│   ├── 05_building_llm/
│   └── 06_training_finetuning/
├── labs/                 # Hands-on exercises
├── quizzes/             # Quiz questions and answers
├── diagrams/            # Visual learning aids
├── references/          # Additional reading materials
├── projects/            # Final capstone projects
└── utils/               # Reusable helper code

Each module contains:
- README.md           # Module overview and learning objectives
- concepts.md         # Theoretical explanations
- python_guide.md     # Python language features explained
- examples/           # Code examples with explanations
- exercises/          # Practice problems with solutions
```

## Architecture Overview

### Learning Path Progression
1. **Python Fundamentals** → Basic syntax, data structures, OOP
2. **NumPy & Math** → Arrays, linear algebra, matrix operations
3. **Neural Network Basics** → Perceptrons, activation functions, backpropagation
4. **Deep Learning Foundations** → MLPs, training loops, optimization
5. **Transformer Architecture** → Attention mechanism, encoder-decoder
6. **LLM Construction** → Tokenization, embeddings, GPT-style models
7. **Training & Fine-tuning** → Data preparation, training strategies, evaluation

### Key Libraries Used
- **NumPy**: Numerical computing (matrix operations)
- **Matplotlib**: Plotting and visualization
- **Jupyter**: Interactive notebooks for experimentation
- **PyTorch** (later modules): Deep learning framework
- **tiktoken**: Tokenization library

### Free Tools Used
- Python 3.10+
- Jupyter Notebook
- VS Code (recommended) or any text editor
- Git for version control
- Google Colab (optional, for GPU access)

## When Helping the Student

### Always:
- Explain WHY, not just WHAT
- Provide C#/.NET analogies when introducing Python concepts
- Break down complex code into digestible chunks
- Include diagrams for architectural concepts
- Create quizzes that reinforce learning
- Provide lab solutions with detailed explanations
- Use concrete examples before abstractions

### Never:
- Assume Python knowledge
- Skip over "obvious" parts
- Use unexplained technical jargon
- Provide code without line-by-line explanations
- Rush through mathematical concepts

### Question Handling
- If student asks "how does X work?", explain the concept, show code, and relate to .NET
- If student is stuck, provide hints first, then progressively more detailed help
- Encourage experimentation and learning from errors

## Progress Tracking
The student should complete modules sequentially. Each module builds on previous knowledge. Mark completion in PROGRESS.md after finishing quizzes and labs.

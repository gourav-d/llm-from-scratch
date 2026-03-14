# Module 7 Projects

**Hands-on projects to apply your reasoning and coding model knowledge**

---

## 🎯 Overview

This folder contains 5 complete projects that demonstrate advanced LLM applications:

**Part A: Reasoning Projects**
1. Math Reasoning System
2. Logic Puzzle Solver

**Part B: Coding Projects**
3. Code Completion Engine (Mini-Copilot)
4. Bug Detection System
5. Unit Test Generator

Each project integrates multiple concepts from Module 7 and provides real-world applications.

---

## 📚 Project 1: Math Reasoning System

**Lessons Used:** 1, 2, 3, 5

### What You'll Build
A complete math problem solver that:
- Solves word problems with step-by-step reasoning
- Uses Chain-of-Thought for transparency
- Verifies answers through multiple reasoning paths
- Handles algebra, geometry, and arithmetic

### Example Usage
```python
from math_reasoning_system import MathReasoner

reasoner = MathReasoner(your_gpt_model)

problem = """
A train leaves Station A at 60 mph heading east.
Another train leaves Station B (180 miles east of A) at 40 mph heading west.
When do they meet?
"""

solution = reasoner.solve(problem)
# Shows full reasoning trace and verified answer
```

### Skills Practiced
- Chain-of-Thought prompting
- Self-consistency verification
- Parsing mathematical expressions
- Step verification

### Difficulty: ⭐⭐ Intermediate
### Time: 4-6 hours

---

## 📚 Project 2: Logic Puzzle Solver

**Lessons Used:** 1, 2, 3

### What You'll Build
An AI that solves logic puzzles using:
- Tree-of-Thoughts for exploring solution paths
- Constraint satisfaction
- Backtracking when paths fail
- Clear explanation of reasoning

### Example Usage
```python
from logic_puzzle_solver import LogicSolver

puzzle = """
Three people: Alice, Bob, Carol
Clues:
- Only one person tells the truth
- Alice says: "Bob is lying"
- Bob says: "Carol is lying"
- Carol says: "Alice is lying"
Who tells the truth?
"""

solver = LogicSolver(your_gpt_model)
solution = solver.solve(puzzle)
# Shows reasoning tree and correct answer
```

### Skills Practiced
- Tree-of-Thoughts search
- Constraint reasoning
- Logical deduction
- Solution verification

### Difficulty: ⭐⭐⭐ Advanced
### Time: 5-7 hours

---

## 📚 Project 3: Code Completion Engine (Mini-Copilot)

**Lessons Used:** 6, 7, 8, 9

### What You'll Build
Your own GitHub Copilot-like tool that:
- Completes code from comments or partial code
- Understands code context using AST
- Generates multiple suggestions
- Ranks by quality and syntax validity

### Example Usage
```python
from code_completion_engine import CodeCompleter

completer = CodeCompleter(your_gpt_model)

prefix = '''
def fibonacci(n):
    """Calculate nth Fibonacci number using memoization"""
'''

suggestions = completer.complete(prefix, n_suggestions=3)
# Returns ranked code completions
```

### Skills Practiced
- Code tokenization with AST
- Fill-in-the-middle (FIM) generation
- Syntax validation
- Code ranking

### Difficulty: ⭐⭐⭐ Advanced
### Time: 6-8 hours

---

## 📚 Project 4: Bug Detection System

**Lessons Used:** 6, 7, 9, 10

### What You'll Build
Automated bug detector that:
- Analyzes code for common bugs
- Explains why something is a bug
- Suggests fixes
- Handles multiple languages

### Example Usage
```python
from bug_detector import BugDetector

detector = BugDetector(your_gpt_model)

code = """
def divide_numbers(a, b):
    return a / b  # Bug: No zero check!
"""

bugs = detector.find_bugs(code)
# Returns:
# [{
#   'line': 2,
#   'type': 'ZeroDivisionError',
#   'severity': 'High',
#   'explanation': 'Division by zero not handled',
#   'fix': 'Add: if b == 0: raise ValueError(...)'
# }]
```

### Skills Practiced
- Code analysis
- Pattern recognition
- AST traversal
- Bug classification

### Difficulty: ⭐⭐⭐ Advanced
### Time: 5-7 hours

---

## 📚 Project 5: Unit Test Generator

**Lessons Used:** 6, 9, 10

### What You'll Build
Automatic test generator that:
- Analyzes function signatures
- Generates comprehensive test cases
- Covers edge cases
- Follows testing best practices

### Example Usage
```python
from test_generator import TestGenerator

generator = TestGenerator(your_gpt_model)

function_code = """
def is_palindrome(s: str) -> bool:
    '''Check if string is a palindrome'''
    return s == s[::-1]
"""

tests = generator.generate_tests(function_code)
# Returns complete pytest test file
```

### Skills Practiced
- Code understanding
- Test case design
- Edge case identification
- Code generation

### Difficulty: ⭐⭐ Intermediate
### Time: 4-6 hours

---

## 🎯 Recommended Order

### Path 1: Reasoning First
1. Math Reasoning System (easier start)
2. Logic Puzzle Solver (build on reasoning)
3. Then move to coding projects

### Path 2: Coding First
1. Unit Test Generator (easier coding project)
2. Code Completion Engine
3. Bug Detection System
4. Then do reasoning projects

### Path 3: Complete (Recommended)
1. Math Reasoning System
2. Unit Test Generator
3. Code Completion Engine
4. Logic Puzzle Solver
5. Bug Detection System

---

## 📁 Project Structure

Each project folder contains:

```
project_name/
├── README.md           # Detailed project description
├── requirements.txt    # Python dependencies
├── main.py            # Main implementation
├── utils.py           # Helper functions
├── examples/          # Example usage
│   ├── example_1.py
│   └── example_2.py
├── tests/             # Unit tests
│   └── test_main.py
├── data/              # Sample data (if needed)
│   └── sample_inputs.txt
└── SOLUTION.md        # Complete solution (check after trying!)
```

---

## 🚀 Getting Started

### Prerequisites
✅ Completed Module 7 Lessons 1-10 (or at least the relevant lessons)
✅ Working GPT model from Module 6
✅ Python 3.8+
✅ Basic understanding of the concepts

### Setup
```bash
# Navigate to a project
cd projects/math_reasoning_system

# Install dependencies
pip install -r requirements.txt

# Read the README
cat README.md

# Try the examples
python examples/example_1.py

# Implement the solution
# (Try yourself first before checking SOLUTION.md!)
```

---

## 💡 Tips for Success

### 1. Try Before Looking at Solutions
```
❌ Don't: Open SOLUTION.md immediately
✓ Do: Read README, understand requirements, try implementing
✓ If stuck: Check examples, re-read lessons
✓ Still stuck: Look at partial solution hints
✓ Finally: Compare with full solution
```

### 2. Test Incrementally
```python
# Don't build everything at once
# Build and test piece by piece:

# Step 1: Basic functionality
def basic_feature():
    pass
# Test it!

# Step 2: Add enhancement
def enhanced_feature():
    pass
# Test it!

# Step 3: Integration
# Test everything together
```

### 3. Use Your GPT Model
```python
# Don't use mock/dummy models for projects
# Load your actual trained GPT:

import sys
sys.path.append('../../06_training_finetuning')
from example_01_complete_gpt import GPT, GPTConfig

config = GPTConfig(...)
gpt = GPT(config)
gpt.load_weights('path/to/your/trained/model.pth')

# Now use it in your project!
```

### 4. Iterate and Improve
```
Version 1: Basic working version
  ↓
Test with examples
  ↓
Version 2: Add error handling
  ↓
Test edge cases
  ↓
Version 3: Optimize performance
  ↓
Version 4: Polish UX
```

---

## 📊 Project Difficulty Matrix

| Project | Lessons | Difficulty | Time | Concepts |
|---------|---------|------------|------|----------|
| Math Reasoner | 1,2,3,5 | ⭐⭐ | 4-6h | CoT, Self-Consistency |
| Logic Solver | 1,2,3 | ⭐⭐⭐ | 5-7h | ToT, Search |
| Code Completer | 6,7,8,9 | ⭐⭐⭐ | 6-8h | AST, FIM, Generation |
| Bug Detector | 6,7,9,10 | ⭐⭐⭐ | 5-7h | Code Analysis, Testing |
| Test Generator | 6,9,10 | ⭐⭐ | 4-6h | Code Understanding |

---

## ✅ Success Criteria

You've successfully completed a project when:

- [ ] Core functionality works as specified
- [ ] Examples run without errors
- [ ] Tests pass (if provided)
- [ ] You understand every line of code
- [ ] You can explain the design decisions
- [ ] You can extend it with new features

---

## 🎓 Learning Outcomes

### After Math Reasoning System:
✓ Can implement CoT in production
✓ Understand reasoning verification
✓ Handle multi-step problems
✓ Parse and validate reasoning traces

### After Logic Puzzle Solver:
✓ Can implement tree search algorithms
✓ Understand constraint satisfaction
✓ Handle complex logical reasoning
✓ Debug reasoning failures

### After Code Completion Engine:
✓ Can build practical coding tools
✓ Understand FIM training
✓ Parse and generate code
✓ Rank code by quality

### After Bug Detection System:
✓ Can analyze code programmatically
✓ Identify common bug patterns
✓ Suggest fixes automatically
✓ Build code quality tools

### After Unit Test Generator:
✓ Can generate tests automatically
✓ Identify edge cases
✓ Understand test design
✓ Build developer tools

---

## 🔄 What's Next?

After completing these projects:

1. **Combine Projects**
   - Build a complete developer assistant (Copilot + Bug Detector + Test Generator)
   - Build a reasoning system that codes (Math Reasoner + Code Generator)

2. **Extend Functionality**
   - Add more programming languages
   - Support more complex reasoning
   - Improve accuracy and speed
   - Add web UI

3. **Deploy**
   - Package as VS Code extension
   - Create CLI tool
   - Build web service
   - Share on GitHub!

4. **Module 8**
   - Production deployment
   - Scaling and optimization
   - Monitoring and observability

---

## 📚 Additional Resources

### For Reasoning Projects
- Research papers in `references/reasoning_papers/`
- Example reasoning traces in `data/examples/`
- Benchmark datasets for testing

### For Coding Projects
- HumanEval dataset
- Code examples in multiple languages
- AST documentation links
- Testing frameworks guide

---

## 🎉 Conclusion

These projects are your chance to build real, useful AI applications!

**Remember:**
- Start simple, iterate
- Test thoroughly
- Learn from mistakes
- Have fun building!

**You're creating tools that could actually help people code and reason better!** 🚀

---

**Ready to build? Choose a project and dive in!** 💪

**Questions? Check the lesson files or SOLUTION.md after trying!**

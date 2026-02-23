# Quick Start Guide

Get started with learning LLMs from scratch in 5 minutes!

---

## Step 1: Install Python (5 minutes)

### Windows:
1. Go to: https://www.python.org/downloads/
2. Download Python 3.10 or higher
3. Run installer
4. **IMPORTANT:** Check "Add Python to PATH"
5. Click "Install Now"

### Verify Installation:
Open Command Prompt (cmd) and type:
```bash
python --version
```

You should see something like: `Python 3.10.x`

---

## Step 2: Set Up Your Environment (2 minutes)

### Open Command Prompt in this folder:
1. Open File Explorer
2. Navigate to this folder
3. Click on the address bar
4. Type `cmd` and press Enter

### Create virtual environment:
```bash
python -m venv venv
```

### Activate it:
```bash
venv\Scripts\activate
```

You'll see `(venv)` at the beginning of your command prompt.

### Install dependencies:
```bash
pip install -r requirements.txt
```

This might take a few minutes - it's downloading libraries.

---

## Step 3: Run Your First Python Program! (1 minute)

### Create a test file:
Create a file called `hello.py` with this content:

```python
# My first Python program!
print("Hello, Python!")

# Variables (no type declaration!)
name = "Your Name"
age = 25

# f-strings (like C#'s $"...")
message = f"My name is {name} and I am {age} years old"
print(message)

# Simple math
x = 10
y = 3

print(f"{x} + {y} = {x + y}")
print(f"{x} / {y} = {x / y}")       # Division (always float!)
print(f"{x} // {y} = {x // y}")     # Integer division
print(f"{x} ** {y} = {x ** y}")     # Power (10³)
```

### Run it:
```bash
python hello.py
```

**Expected Output:**
```
Hello, Python!
My name is Your Name and I am 25 years old
10 + 3 = 13
10 / 3 = 3.3333333333333335
10 // 3 = 3
10 ** 3 = 1000
```

🎉 Congratulations! You just ran your first Python program!

---

## Step 4: Start Learning! (∞ time)

### Option A: Read the lessons (Recommended for beginners)
1. Open `modules/01_python_basics/README.md`
2. Read lessons in order (01 → 10)
3. Try the code examples
4. Do the exercises

### Option B: Interactive learning with Jupyter
1. Install Jupyter: `pip install jupyter notebook`
2. Start Jupyter: `jupyter notebook`
3. Browser will open automatically
4. Create new notebook and start coding!

---

## Learning Path

```
Week 1: Python Basics
├── Variables and Types
├── Operators
├── Control Flow
├── Functions
└── Data Structures

Week 2: NumPy & Math
├── Arrays
├── Matrix Operations
└── Linear Algebra

Week 3: Neural Networks
├── What is a Neural Network?
├── Forward Propagation
└── Backpropagation

Weeks 4-8: Build Your LLM!
```

---

## Useful Commands

### Python:
```bash
python script.py          # Run a Python script
python                    # Start Python interpreter
exit()                    # Exit Python interpreter
```

### Virtual Environment:
```bash
venv\Scripts\activate     # Activate (Windows)
deactivate                # Deactivate
```

### Jupyter Notebook:
```bash
jupyter notebook          # Start Jupyter
Ctrl + C                  # Stop Jupyter (in terminal)
```

### Install Packages:
```bash
pip install package_name      # Install a package
pip install -r requirements.txt   # Install all dependencies
pip list                      # List installed packages
```

---

## File Structure

```
your-folder/
├── README.md                 ← Start here!
├── QUICK_START.md           ← You are here
├── PROGRESS.md              ← Track your progress
├── CLAUDE.md                ← Guide for Claude Code
├── requirements.txt         ← Python dependencies
│
├── modules/                 ← Learning modules
│   └── 01_python_basics/
│       ├── README.md
│       ├── 01_variables_and_types.md
│       ├── 02_operators.md
│       ├── 03_control_flow.md
│       └── ...
│
├── quizzes/                 ← Test your knowledge
│   └── quiz_module_1.md
│
└── labs/                    ← Hands-on exercises
    └── lab_module_1.md
```

---

## Tips for .NET Developers

### Things That Are Different:
- **No type declarations** → `name = "Bob"` not `string name = "Bob";`
- **No semicolons** → Lines end automatically
- **Indentation matters!** → Use 4 spaces, not braces `{}`
- **snake_case** → `user_name` not `userName`
- **True/False** → Capitalized, not lowercase
- **No `++` or `--`** → Use `x += 1`

### Things That Are Similar:
- if/else, for, while → Same logic, different syntax
- Functions → Similar concept
- Classes → Similar OOP
- Lists → Like `List<T>`
- Dictionaries → Like `Dictionary<K,V>`

---

## Getting Help

### Within this course:
- Read the lessons carefully
- Try the examples
- Do the quizzes and labs
- Use Claude Code to ask questions!

### Online resources:
- Official Python Docs: https://docs.python.org/3/
- Python for .NET developers: Search "Python for C# developers"

---

## Common Issues

### Issue: "python is not recognized"
**Solution:** Add Python to PATH
1. Find where Python is installed
2. Add to System Environment Variables
3. Restart Command Prompt

### Issue: "No module named X"
**Solution:** Install the module
```bash
pip install X
```

### Issue: "IndentationError"
**Solution:** Check your indentation
- Use 4 spaces per level
- Don't mix spaces and tabs
- Use a proper code editor (VS Code recommended)

### Issue: Virtual environment not activating
**Solution:**
```bash
# Try this instead:
venv\Scripts\activate.bat
```

---

## Next Steps

1. ✅ Python installed
2. ✅ Environment set up
3. ✅ First program run
4. → Start `modules/01_python_basics/01_variables_and_types.md`
5. → Do quizzes and labs
6. → Track progress in `PROGRESS.md`
7. → Build your first LLM!

Let's go! 🚀

---

## Questions?

Use Claude Code to ask any questions:
- "Explain what X means"
- "Show me an example of Y"
- "What's the difference between X and Y?"
- "Help me debug this code"

Claude is here to help you learn! 🤖

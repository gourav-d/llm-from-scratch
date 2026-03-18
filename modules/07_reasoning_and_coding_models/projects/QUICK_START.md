# Quick Start Guide - Module 7 Projects

**Get started with real-world projects in 5 minutes!**

---

## What's Available RIGHT NOW

✅ **Project 1: AI Code Reviewer** - COMPLETE & READY
✅ **Project 2: Smart Bug Debugger** - COMPLETE & READY

🚧 **Project 3: Semantic Code Search** - Coming soon
🚧 **Project 4: Auto Test Writer** - Coming soon
🚧 **Project 5: Code Quality Analyzer** - Coming soon

---

## Try Project 1: AI Code Reviewer

### Installation (30 seconds)

```bash
cd projects/01_ai_code_reviewer

# Install dependencies
pip install colorama rich
# Note: Full dependencies in requirements.txt, but these work for demo
```

### Run the Example (30 seconds)

```bash
python examples/example_01_basic.py
```

**What you'll see:**
- ✅ Detects SQL injection vulnerability
- ✅ Finds off-by-one error
- ✅ Identifies empty except blocks
- ✅ Explains each issue with Chain-of-Thought reasoning
- ✅ Suggests fixes with code examples

### Try It On Your Own Code (1 minute)

```python
# Create test.py
from ai_code_reviewer import CodeReviewer

# Your code to review
my_code = """
def login(username, password):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    user = db.execute(query)
    if user.password == password:
        return True
    return False
"""

reviewer = CodeReviewer()
issues = reviewer.review_code(my_code)
print(reviewer.format_report())

# Run it!
# python test.py
```

**Output:**
```
🔴 CRITICAL: Potential SQL injection vulnerability
Line 2 | Category: security

Problem:
query = "SELECT * FROM users WHERE name = '" + username + "'"

Reasoning:
Step 1: Analyzing the code
   The code constructs an SQL query using string concatenation...

Step 2: Identifying the risk
   When user input is directly concatenated into SQL queries...

Suggestion:
Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', [user_id])
```

---

## Try Project 2: Smart Bug Debugger

### Installation (30 seconds)

```bash
cd projects/02_smart_debugger

# Install dependencies
pip install colorama rich
```

### Run the Example (30 seconds)

```bash
python examples/example_01_basic_error.py
```

**What you'll see:**
- ✅ Analyzes 4 common errors (IndexError, ZeroDivisionError, etc.)
- ✅ Chain-of-Thought reasoning for each
- ✅ Multiple fix suggestions
- ✅ Clear explanations

### Try It On Your Bug (1 minute)

```python
# Create debug_test.py
from smart_debugger import SmartDebugger

debugger = SmartDebugger()

# Your buggy code
code = """
def calculate_discount(price, percent):
    return price - (price * percent / 100)
"""

# The error you got
error = "TypeError: unsupported operand type(s) for -: 'str' and 'float'"

# Get AI analysis
analysis = debugger.analyze_error(error, code)
print(analysis)

# Run it!
# python debug_test.py
```

**Output:**
```
🔍 SMART DEBUGGER ANALYSIS
==========================================

Error Type: TypeError
Message: unsupported operand type(s) for -: 'str' and 'float'

CHAIN-OF-THOUGHT REASONING:
----------------------------------------
Step 1: Understanding the error
   TypeError: Operation not supported for the given types
   Message: unsupported operand type(s) for -: 'str' and 'float'

Step 2: Common causes of TypeError
   - Wrong number of arguments to function
   - Wrong type passed to function
   - Calling non-callable object
   - Unsupported operation for type

Step 3: Impact assessment
   Severity: MEDIUM
   This will crash the program if not handled

ROOT CAUSE:
----------------------------------------
A TypeError occurred. See the reasoning steps above for likely causes.

FIX SUGGESTIONS:
----------------------------------------

Option 1:
**Guard clause** (Recommended)
```python
if index < len(list):
    item = list[index]
```
Explanation: Check bounds before accessing

Option 2:
**Add error handling**
```python
try:
    # Your code here
except TypeError as e:
    logger.error(f'Error: {e}')
    # Handle gracefully
```
Explanation: Catch and handle the error gracefully
```

---

## Real-World Usage

### Use Case 1: Pre-Commit Hook

```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
cd projects/01_ai_code_reviewer
python -m ai_code_reviewer review $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
```

### Use Case 2: Exception Handler

```python
# Add to your main.py
import sys
import traceback
from smart_debugger import SmartDebugger

def handle_exception(exc_type, exc_value, exc_traceback):
    debugger = SmartDebugger()
    trace = ''.join(traceback.format_tb(exc_traceback))
    analysis = debugger.analyze_error(str(exc_value), None, trace)
    print(analysis)

sys.excepthook = handle_exception
```

### Use Case 3: Code Review Script

```python
# review_pr.py
from ai_code_reviewer import CodeReviewer
import subprocess

# Get changed files
result = subprocess.run(['git', 'diff', 'origin/main', '--name-only'],
                       capture_output=True, text=True)
files = result.stdout.strip().split('\n')

reviewer = CodeReviewer()
for file in files:
    if file.endswith('.py'):
        with open(file) as f:
            code = f.read()
        issues = reviewer.review_code(code, file)
        if issues:
            print(f"\n{'='*80}")
            print(f"Issues in {file}:")
            print(reviewer.format_report())

# Run before creating PR:
# python review_pr.py
```

---

## Next Steps

### Today
1. ✅ Run both examples
2. ✅ Try on your own code
3. ✅ Find at least 1 real issue

### This Week
1. ✅ Integrate into your workflow
2. ✅ Add pre-commit hook
3. ✅ Review all your Python files

### This Month
1. ✅ Customize the rules
2. ✅ Add more security patterns
3. ✅ Share with your team

---

## Customization Examples

### Add Custom Security Rule

```python
# In ai_code_reviewer.py, add to _load_rules():

self.security_patterns["api_key_exposed"] = {
    "pattern": r'api[_-]?key\s*=\s*[\'"][^\'"]{20,}[\'"]',
    "message": "Hardcoded API key detected",
    "severity": IssueSeverity.CRITICAL,
}
```

### Add Custom Bug Pattern

```python
self.bug_patterns["mutable_default"] = {
    "pattern": r'def\s+\w+\([^)]*=\s*\[\s*\]',
    "message": "Mutable default argument (dangerous!)",
    "severity": IssueSeverity.HIGH,
}
```

### Add Custom Error Explanation

```python
# In smart_debugger.py, add to _load_error_patterns():

self.error_patterns["ModuleNotFoundError"] = {
    "common_causes": [
        "Package not installed (pip install missing)",
        "Typo in import statement",
        "Virtual environment not activated",
    ],
    "explanation": "Python can't find the module you're trying to import"
}
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'ai_code_reviewer'"

**Solution:**
```bash
# Make sure you're in the right directory
cd projects/01_ai_code_reviewer

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/projects/01_ai_code_reviewer"
```

### "No issues found" when you know there are issues

**Possible reasons:**
1. Pattern not in the database → Add custom pattern
2. Code is actually good! → Verify manually
3. Language not Python → Currently only supports Python

### Want more detailed analysis?

**Enable AI model (optional):**
```python
# If you have a trained GPT model from Module 6
reviewer = CodeReviewer(model=your_gpt_model)
```

---

## Performance Tips

### Speed Up Analysis

```python
# Review only changed lines (faster)
reviewer = CodeReviewer()
issues = reviewer.review_code(code, incremental=True)

# Skip low-priority checks
reviewer = CodeReviewer(check_style=False, check_todos=False)
```

### Parallel Processing

```python
# Review multiple files in parallel
from concurrent.futures import ThreadPoolExecutor

files = ['file1.py', 'file2.py', 'file3.py']

def review_file(filename):
    with open(filename) as f:
        code = f.read()
    reviewer = CodeReviewer()
    return reviewer.review_code(code, filename)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(review_file, files))
```

---

## Feedback & Contributions

### Found a bug?
Open an issue with the code that caused the problem

### Have an idea?
Suggest new patterns or features

### Want to contribute?
PRs welcome! Add more:
- Security patterns
- Bug patterns
- Error explanations
- Fix templates

---

## What's Next?

### Coming Soon (March 2026)
- ✅ Project 3: Semantic Code Search
- ✅ Project 4: Auto Test Writer
- ✅ Project 5: Code Quality Analyzer

### Future Enhancements
- 🔜 Support more languages (JavaScript, C#, Java)
- 🔜 VS Code extension
- 🔜 GitHub Action
- 🔜 Web UI
- 🔜 Team dashboards

---

## Success Stories

**Use these tools to:**
- ✅ Catch bugs before they reach production
- ✅ Learn security best practices
- ✅ Improve code quality
- ✅ Save time debugging
- ✅ Mentor junior developers

---

**Ready to level up your development workflow?** 🚀

Start with Project 1 or 2 today and see the difference!

**Questions?** Check the individual project READMEs or open an issue.

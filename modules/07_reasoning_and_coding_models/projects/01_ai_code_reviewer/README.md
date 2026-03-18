# Project 1: AI Code Reviewer

**Production-ready code review assistant for daily development work**

---

## What You'll Build

A complete AI-powered code reviewer that:
- Reviews code changes (like a senior developer)
- Checks for bugs, security issues, and best practices
- Provides explanations using Chain-of-Thought reasoning
- Suggests improvements with examples
- Works with Git diffs and PR reviews

---

## Why This Is Useful

**Real-World Use Cases:**
- Review your own code before committing
- Automate PR reviews in your team
- Learn best practices from AI feedback
- Catch bugs before they reach production
- Improve code quality consistently

**Frequency of Use:** Daily! Every time you write code.

---

## Features

### 1. Code Quality Review
```python
reviewer.review_code(code_snippet)
# Returns: style issues, bugs, improvements
```

### 2. Security Analysis
```python
reviewer.check_security(code_snippet)
# Returns: SQL injection, XSS, hardcoded secrets, etc.
```

### 3. Bug Detection
```python
reviewer.find_bugs(code_snippet)
# Returns: null checks, edge cases, race conditions
```

### 4. Best Practice Suggestions
```python
reviewer.suggest_improvements(code_snippet)
# Returns: better patterns, refactorings
```

### 5. Git Diff Review
```python
reviewer.review_diff(git_diff)
# Reviews only changed lines
```

---

## Example Usage

### Basic Code Review

```python
from ai_code_reviewer import CodeReviewer

# Initialize with your LLM
reviewer = CodeReviewer(model="gpt-4")

# Review a code snippet
code = """
def get_user(user_id):
    user = db.query("SELECT * FROM users WHERE id = " + user_id)
    return user
"""

review = reviewer.review_code(code, language="python")

# Output:
# CRITICAL: SQL Injection vulnerability!
# Reasoning: Concatenating user input directly into SQL query allows
# attackers to inject malicious SQL commands.
#
# Fix: Use parameterized queries
# Suggestion:
# def get_user(user_id):
#     user = db.query("SELECT * FROM users WHERE id = ?", [user_id])
#     return user
```

### Git Diff Review

```python
# Review a Git diff
diff = """
diff --git a/app.py b/app.py
@@ -10,3 +10,5 @@
-    return user.password == password
+    return user.password_hash == hash_password(password)
"""

review = reviewer.review_diff(diff)
# Gives feedback on the changes
```

### Integrate with CI/CD

```python
# In your GitHub Actions or Jenkins pipeline
from ai_code_reviewer import CodeReviewer
import sys

reviewer = CodeReviewer()
files_changed = get_git_diff()  # Your CI tool
review_result = reviewer.review_diff(files_changed)

if review_result.has_critical_issues():
    print(review_result.format_for_pr_comment())
    sys.exit(1)  # Fail the build
```

---

## What You'll Learn

### From Module 7

**Reasoning (Lessons 1-5):**
- Chain-of-Thought for explaining issues
- Self-consistency for validating findings
- Process supervision for review quality

**Code Understanding (Lessons 6-10):**
- Code tokenization and AST parsing
- Pattern matching for bug detection
- Code quality metrics

### C# to Python

| C# Concept | Python Equivalent | Used For |
|------------|-------------------|----------|
| Roslyn | AST parsing | Code analysis |
| CodeAnalysis | ast module | Syntax tree |
| LINQ | List comprehensions | Filtering issues |
| Regex | re module | Pattern matching |

---

## Project Structure

```
01_ai_code_reviewer/
├── README.md                    # This file
├── requirements.txt             # Dependencies
│
├── ai_code_reviewer.py          # Main implementation
├── reviewers/
│   ├── base_reviewer.py         # Base class
│   ├── security_reviewer.py     # Security checks
│   ├── bug_reviewer.py          # Bug detection
│   ├── style_reviewer.py        # Code style
│   └── best_practice_reviewer.py # Best practices
│
├── utils/
│   ├── ast_parser.py            # Code parsing
│   ├── git_utils.py             # Git diff handling
│   └── formatting.py            # Output formatting
│
├── examples/
│   ├── example_01_basic.py      # Basic review
│   ├── example_02_security.py   # Security scan
│   ├── example_03_git_diff.py   # Review PR
│   └── example_04_cli.py        # CLI usage
│
├── tests/
│   ├── test_reviewer.py         # Unit tests
│   ├── test_security.py         # Security tests
│   └── test_integration.py      # End-to-end tests
│
└── data/
    ├── rules/                   # Review rules
    │   ├── security_rules.json
    │   ├── bug_patterns.json
    │   └── style_rules.json
    └── test_cases/              # Sample code to review
        ├── vulnerable_code.py
        └── good_code.py
```

---

## Getting Started

### Step 1: Install Dependencies

```bash
cd projects/01_ai_code_reviewer
pip install -r requirements.txt
```

### Step 2: Try Examples

```bash
# Basic code review
python examples/example_01_basic.py

# Security scan
python examples/example_02_security.py

# Review a Git diff
python examples/example_03_git_diff.py
```

### Step 3: Use as CLI Tool

```bash
# Review a file
python -m ai_code_reviewer review myfile.py

# Review Git diff
python -m ai_code_reviewer diff HEAD~1

# Review current changes
git diff | python -m ai_code_reviewer diff --stdin
```

### Step 4: Integrate into Your Workflow

```bash
# Add pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
python -m ai_code_reviewer diff --staged
EOF

chmod +x .git/hooks/pre-commit
```

---

## Implementation Guide

### Core Components

**1. Code Parser (AST-based)**
```python
# Uses Python's ast module to parse code
# Similar to Roslyn in C#
import ast

tree = ast.parse(code)
# Analyze the syntax tree
```

**2. Review Engine (Chain-of-Thought)**
```python
# Uses CoT to explain issues
prompt = f"""
Analyze this code:
{code}

Think step by step:
1. What does this code do?
2. Are there any bugs?
3. Are there security issues?
4. Can it be improved?
"""
```

**3. Pattern Matching**
```python
# Detect common issues
PATTERNS = {
    "sql_injection": r"query.*\+.*user",
    "hardcoded_password": r"password\s*=\s*['\"]",
    "todo_comment": r"#\s*TODO",
}
```

---

## Review Categories

### 1. Security Issues (CRITICAL)
- SQL Injection
- XSS vulnerabilities
- Hardcoded credentials
- Insecure random numbers
- Path traversal
- Command injection

### 2. Bugs (HIGH)
- Null pointer exceptions
- Array out of bounds
- Division by zero
- Race conditions
- Resource leaks
- Infinite loops

### 3. Code Quality (MEDIUM)
- Unused variables
- Dead code
- Complex functions
- Code duplication
- Poor naming

### 4. Style Issues (LOW)
- Formatting inconsistencies
- Missing docstrings
- Line length
- Import order

---

## Advanced Features

### 1. Custom Rules

```python
# Add your own review rules
reviewer.add_rule({
    "name": "No console.log",
    "pattern": r"console\.log",
    "severity": "medium",
    "message": "Remove debug statements",
    "suggestion": "Use proper logging library"
})
```

### 2. Team Standards

```python
# Configure for your team
reviewer.configure({
    "max_function_length": 50,
    "max_complexity": 10,
    "required_docstrings": True,
    "style_guide": "pep8",
})
```

### 3. Auto-Fix

```python
# Automatically fix simple issues
review = reviewer.review_code(code)
fixed_code = reviewer.auto_fix(code, review.fixable_issues)
```

---

## Performance Tips

1. **Cache AST parsing** - Don't parse the same file twice
2. **Parallel reviews** - Review multiple files concurrently
3. **Incremental analysis** - Only review changed lines
4. **Smart batching** - Batch similar checks together

---

## Success Criteria

You've completed this project when:

- [x] Can review Python code and find issues
- [x] Detects security vulnerabilities
- [x] Provides clear explanations with CoT
- [x] Works with Git diffs
- [x] Can be used as CLI tool
- [x] All tests pass

---

## Real-World Integration

### GitHub Actions

```yaml
# .github/workflows/code-review.yml
name: AI Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run AI Code Review
        run: |
          pip install -r projects/01_ai_code_reviewer/requirements.txt
          python -m ai_code_reviewer diff origin/main
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
python -m ai_code_reviewer diff --staged --fail-on-critical
```

### VS Code Extension (Future)

```json
{
  "name": "ai-code-reviewer",
  "command": "python -m ai_code_reviewer review ${file}"
}
```

---

## Difficulty: ⭐⭐⭐ Advanced

**Time Estimate:** 6-8 hours

**Prerequisites:**
- Module 7 Lessons 1-10 (especially 1, 4, 6, 7, 9, 10)
- Understanding of AST parsing
- Basic Git knowledge

---

## Next Steps

After completing this project:

1. **Extend to more languages** (JavaScript, C#, Java)
2. **Add ML-based bug detection** (train on known bugs)
3. **Build PR comment bot** (automatically comment on PRs)
4. **Create VS Code extension** (real-time feedback)
5. **Add team analytics** (track code quality trends)

---

**This is a tool you'll use EVERY DAY in your development workflow!** 🚀

Ready to build? Let's start coding!

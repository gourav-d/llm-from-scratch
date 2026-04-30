"""
Project 03: Automated Code Review Agent
==========================================

WHAT THIS BUILDS
-----------------
A multi-agent code review system that:
1. Analyzes code for bugs and potential issues
2. Checks code quality and style
3. Suggests improvements with explanations
4. Generates a structured review report

This is a MULTI-AGENT system:
  - AnalyzerAgent: finds bugs and issues
  - StyleAgent:    checks code style and naming
  - SuggestionAgent: generates improvement ideas
  - ReviewOrchestrator: coordinates all agents, produces final report

LEARNING GOALS
--------------
- Apply multi-agent orchestration (Lesson 05)
- Use specialist agents with focused responsibilities
- See how coding tools like GitHub Copilot work internally
- Build a useful developer tool from scratch

HOW TO RUN
----------
  python project_03_code_review_agent.py

LIBRARIES NEEDED
-----------------
  None (pure Python)
"""

import re        # Regular expressions for code analysis
import json      # For structured output

print("=" * 65)
print("PROJECT 03: Automated Code Review Agent")
print("=" * 65)


# ==============================================================================
# CODE SAMPLES TO REVIEW
# ==============================================================================

# These are Python code snippets with intentional issues for the agent to find
CODE_SAMPLES = {
    "calculator_buggy": '''
def calculate(a, b, op):
    if op == "add":
        return a + b
    elif op == "subtract":
        return a - b
    elif op == "multiply":
        return a * b
    elif op == "divide":
        return a / b  # BUG: no division by zero check!

def main():
    x = 10
    y = 0
    result = calculate(x, y, "divide")
    print("Result:", result)
''',

    "user_login_issues": '''
import hashlib

def login(username, password):
    # BUG: MD5 is cryptographically weak -- use bcrypt or argon2
    hashed = hashlib.md5(password.encode()).hexdigest()
    users = {"admin": "21232f297a57a5a743894a0e4a801fc3"}  # "admin" hashed with MD5
    if username in users and users[username] == hashed:
        return True
    return False  # STYLE: could also just return username in users and users[username] == hashed

def get_user_data(user_id):
    # BUG: SQL injection vulnerability!
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    # This should use parameterized queries: cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return query
''',

    "data_processor_good": '''
def process_items(items: list) -> list:
    """
    Process a list of items and return only the valid ones.
    items: list of item dicts with 'value' key
    Returns: list of items where value > 0
    """
    if not items:       # Handle empty list
        return []

    results = []
    for item in items:
        if not isinstance(item, dict):     # Type check
            continue
        value = item.get("value", None)    # Safe key access
        if value is not None and value > 0:
            results.append(item)

    return results

def calculate_average(numbers: list) -> float:
    """Calculate average of a list of numbers."""
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)
''',
}


# ==============================================================================
# SPECIALIST AGENTS
# ==============================================================================

class BaseReviewAgent:
    """Base class for all code review specialist agents."""

    def __init__(self, name: str):
        self.name = name
        self.findings = []    # Issues found

    def analyze(self, code: str, context: dict = None) -> list:
        """
        Analyze code and return a list of findings.
        Each finding: {"severity": "critical/warning/info", "message": "...", "line": N or None}
        """
        raise NotImplementedError

    def __str__(self):
        return self.name


class BugAnalyzerAgent(BaseReviewAgent):
    """
    Finds potential bugs and runtime errors in code.
    Focus: crashes, wrong logic, security vulnerabilities.
    """

    def __init__(self):
        super().__init__("BugAnalyzerAgent")

    def analyze(self, code: str, context: dict = None) -> list:
        """Check for common bugs."""
        self.findings = []
        lines = code.split("\n")    # Split into individual lines

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()    # Remove leading/trailing spaces

            # Check: division by zero risk
            if "/ " in line and "0" in line and "# BUG" not in line:
                if re.search(r'/\s*\w+', line):    # Has division
                    # Check if there's no zero check before it
                    # Look at surrounding lines for a zero check
                    nearby = lines[max(0, i-3):i]
                    has_zero_check = any("!= 0" in l or "== 0" in l or "ZeroDivision" in l
                                         for l in nearby)
                    if not has_zero_check:
                        self.findings.append({
                            "severity": "warning",
                            "message":  f"Line {i}: Division without zero check. If denominator is 0, this will crash.",
                            "line":     i,
                            "code":     stripped
                        })

            # Check: bare division that could cause ZeroDivisionError
            if re.search(r'return\s+\w+\s*/\s*\w+', line):
                self.findings.append({
                    "severity": "warning",
                    "message":  f"Line {i}: Division in return statement. Consider adding zero check.",
                    "line":     i,
                    "code":     stripped
                })

            # Check: SQL injection
            if "SELECT" in line.upper() and ("+" in line or "%" in line) and '"' in line:
                self.findings.append({
                    "severity": "critical",
                    "message":  f"Line {i}: Possible SQL injection! String concatenation in SQL query. Use parameterized queries.",
                    "line":     i,
                    "code":     stripped
                })

            # Check: MD5 usage (weak cryptography)
            if "md5" in line.lower() and "password" in code.lower():
                self.findings.append({
                    "severity": "critical",
                    "message":  f"Line {i}: MD5 is cryptographically weak for password hashing. Use bcrypt, argon2, or scrypt.",
                    "line":     i,
                    "code":     stripped
                })

            # Check: bare except (catches everything including KeyboardInterrupt)
            if stripped == "except:" or stripped.startswith("except:"):
                self.findings.append({
                    "severity": "warning",
                    "message":  f"Line {i}: Bare 'except:' catches ALL exceptions including KeyboardInterrupt. Use 'except Exception:'",
                    "line":     i,
                    "code":     stripped
                })

            # Check: eval() usage (security risk)
            if "eval(" in line and "safe" not in line.lower():
                self.findings.append({
                    "severity": "warning",
                    "message":  f"Line {i}: eval() can execute arbitrary code. Ensure input is trusted and sanitized.",
                    "line":     i,
                    "code":     stripped
                })

        return self.findings


class StyleCheckerAgent(BaseReviewAgent):
    """
    Checks code style and best practices.
    Focus: naming, comments, code organization.
    """

    def __init__(self):
        super().__init__("StyleCheckerAgent")

    def analyze(self, code: str, context: dict = None) -> list:
        """Check code style."""
        self.findings = []
        lines = code.split("\n")

        # Check: function docstrings
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Function defined but no docstring on next non-empty line?
            if stripped.startswith("def ") and i < len(lines):
                next_lines = [l.strip() for l in lines[i:i+2]]
                has_docstring = any(l.startswith('"""') or l.startswith("'''")
                                    or l.startswith('"') for l in next_lines)
                if not has_docstring:
                    func_name = re.search(r'def\s+(\w+)', stripped)
                    if func_name:
                        self.findings.append({
                            "severity": "info",
                            "message":  f"Line {i}: Function '{func_name.group(1)}()' has no docstring. "
                                        "Add a docstring explaining what it does.",
                            "line":     i,
                            "code":     stripped
                        })

            # Check: magic numbers (unexplained constants in code)
            magic_match = re.search(r'[^=!<>]\b([2-9][0-9]+|[0-9]{4,})\b', stripped)
            if magic_match and not stripped.startswith("#") and "import" not in stripped:
                num = magic_match.group(1)
                self.findings.append({
                    "severity": "info",
                    "message":  f"Line {i}: Magic number '{num}'. Consider defining it as a named constant for readability.",
                    "line":     i,
                    "code":     stripped
                })

            # Check: very long lines
            if len(line) > 100 and not stripped.startswith("#"):
                self.findings.append({
                    "severity": "info",
                    "message":  f"Line {i}: Line is {len(line)} chars (recommended max: 79-100). Consider breaking it up.",
                    "line":     i,
                    "code":     stripped[:60] + "..."
                })

            # Check: single-letter variable names (except common ones like i, x, y, n)
            single_letter = re.findall(r'\b([a-zA-Z])\b\s*=', stripped)
            bad_names = [c for c in single_letter if c.lower() not in list("inxyzkt")]
            for name in bad_names[:1]:    # Report first occurrence only
                self.findings.append({
                    "severity": "info",
                    "message":  f"Line {i}: Variable '{name}' is a single letter. Use a descriptive name.",
                    "line":     i,
                    "code":     stripped
                })

        # Check: overall function presence
        functions = re.findall(r'def\s+(\w+)', code)
        if not functions:
            self.findings.append({
                "severity": "info",
                "message":  "No functions found. Consider organizing code into functions for reusability.",
                "line":     None,
                "code":     None
            })

        return self.findings


class SuggestionAgent(BaseReviewAgent):
    """
    Generates improvement suggestions based on findings from other agents.
    Synthesizes bugs + style issues into actionable recommendations.
    """

    def __init__(self):
        super().__init__("SuggestionAgent")

    def analyze(self, code: str, context: dict = None) -> list:
        """Generate suggestions based on code and prior findings."""
        self.findings = []
        context = context or {}

        prior_findings = context.get("all_findings", [])

        # Count severity levels from prior findings
        critical_count = sum(1 for f in prior_findings if f.get("severity") == "critical")
        warning_count  = sum(1 for f in prior_findings if f.get("severity") == "warning")

        # High-level recommendations
        if critical_count > 0:
            self.findings.append({
                "severity": "critical",
                "message":  f"URGENT: {critical_count} critical security/safety issue(s) found. "
                            "Fix these before deploying to production.",
                "line":     None,
                "code":     None
            })

        if warning_count > 0:
            self.findings.append({
                "severity": "warning",
                "message":  f"{warning_count} warning(s) found. Review and fix to improve reliability.",
                "line":     None,
                "code":     None
            })

        # Code-specific suggestions
        if "sql" in code.lower() or "query" in code.lower():
            self.findings.append({
                "severity": "info",
                "message":  "SUGGESTION: Use parameterized queries (cursor.execute(sql, params)) for all SQL. "
                            "Never build SQL strings with user input.",
                "line":     None,
                "code":     "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
            })

        if "password" in code.lower() or "hash" in code.lower():
            self.findings.append({
                "severity": "info",
                "message":  "SUGGESTION: For password hashing, use bcrypt: "
                            "import bcrypt; hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())",
                "line":     None,
                "code":     "pip install bcrypt; bcrypt.hashpw(password.encode(), bcrypt.gensalt())"
            })

        if "/ " in code and "0" in code:
            self.findings.append({
                "severity": "info",
                "message":  "SUGGESTION: Always guard divisions: "
                            "if denominator != 0: result = numerator / denominator",
                "line":     None,
                "code":     "result = a / b if b != 0 else None"
            })

        # General best practices
        if not self.findings:
            self.findings.append({
                "severity": "info",
                "message":  "GENERAL: Consider adding type hints to function parameters and return values. "
                            "Example: def process(items: list[dict]) -> list[dict]:",
                "line":     None,
                "code":     None
            })

        return self.findings


# ==============================================================================
# REVIEW ORCHESTRATOR
# ==============================================================================

class CodeReviewOrchestrator:
    """
    Orchestrates the full code review pipeline.
    Coordinates BugAnalyzerAgent, StyleCheckerAgent, and SuggestionAgent.
    Produces a structured review report.
    """

    def __init__(self):
        self.bug_analyzer  = BugAnalyzerAgent()
        self.style_checker = StyleCheckerAgent()
        self.suggester     = SuggestionAgent()

        self.last_review = None    # Store the most recent review

    def _format_severity_bar(self, severity: str) -> str:
        """Return a text label for the severity level."""
        labels = {
            "critical": "[CRITICAL]",
            "warning":  "[WARNING ]",
            "info":     "[ INFO   ]",
        }
        return labels.get(severity, "[UNKNOWN ]")

    def review(self, code: str, code_name: str = "code") -> dict:
        """
        Run the full code review pipeline.
        code:      the Python code to review (string)
        code_name: a label for the code (for the report)
        Returns:   review dict with all findings and summary
        """
        print(f"\n{'='*55}")
        print(f"REVIEWING: {code_name}")
        print(f"{'='*55}")

        blackboard = {
            "code_name":    code_name,
            "code":         code,
            "all_findings": [],
        }

        # Step 1: Bug Analysis
        print(f"\n[{self.bug_analyzer.name}] Analyzing for bugs...")
        bug_findings = self.bug_analyzer.analyze(code, blackboard)
        blackboard["all_findings"].extend(bug_findings)
        print(f"  Found {len(bug_findings)} bug-related issue(s).")

        # Step 2: Style Check
        print(f"\n[{self.style_checker.name}] Checking code style...")
        style_findings = self.style_checker.analyze(code, blackboard)
        blackboard["all_findings"].extend(style_findings)
        print(f"  Found {len(style_findings)} style issue(s).")

        # Step 3: Suggestions (reads all_findings from blackboard)
        print(f"\n[{self.suggester.name}] Generating suggestions...")
        suggestions = self.suggester.analyze(code, blackboard)
        blackboard["suggestions"] = suggestions
        print(f"  Generated {len(suggestions)} suggestion(s).")

        # Build summary
        all_issues = blackboard["all_findings"]
        critical = [f for f in all_issues if f["severity"] == "critical"]
        warnings = [f for f in all_issues if f["severity"] == "warning"]
        infos    = [f for f in all_issues if f["severity"] == "info"]

        score = 100
        score -= len(critical) * 25     # Each critical = -25 points
        score -= len(warnings) * 10     # Each warning = -10 points
        score -= len(infos)    * 2      # Each info = -2 points
        score = max(0, score)           # Minimum score is 0

        blackboard["summary"] = {
            "total_issues":   len(all_issues),
            "critical_count": len(critical),
            "warning_count":  len(warnings),
            "info_count":     len(infos),
            "score":          score,
            "verdict":        "PASS" if score >= 70 else "NEEDS WORK" if score >= 40 else "FAIL"
        }

        self.last_review = blackboard
        return blackboard

    def print_report(self, review: dict):
        """Print a formatted review report."""
        print("\n" + "=" * 55)
        print(f"CODE REVIEW REPORT: {review['code_name']}")
        print("=" * 55)

        summary = review["summary"]
        verdict_symbols = {"PASS": "OK", "NEEDS WORK": "WARN", "FAIL": "FAIL"}
        verdict_label = verdict_symbols.get(summary["verdict"], "???")

        print(f"\nOVERALL SCORE: {summary['score']}/100")
        print(f"VERDICT:       [{verdict_label}] {summary['verdict']}")
        print(f"Total issues:  {summary['total_issues']}")
        print(f"  Critical:    {summary['critical_count']}")
        print(f"  Warnings:    {summary['warning_count']}")
        print(f"  Info/Style:  {summary['info_count']}")

        print("\n" + "-" * 55)
        print("ISSUES FOUND:")
        print("-" * 55)

        all_findings = review["all_findings"]
        if not all_findings:
            print("  No issues found!")
        else:
            for finding in all_findings:
                label = self._format_severity_bar(finding["severity"])
                line_info = f"Line {finding['line']}: " if finding.get("line") else ""
                print(f"\n  {label} {line_info}{finding['message']}")
                if finding.get("code"):
                    print(f"             Code: {finding['code'][:80]}")

        print("\n" + "-" * 55)
        print("SUGGESTIONS:")
        print("-" * 55)
        for sug in review.get("suggestions", []):
            label = self._format_severity_bar(sug["severity"])
            print(f"\n  {label} {sug['message']}")
            if sug.get("code"):
                print(f"             Fix:  {sug['code'][:80]}")

        print("\n" + "=" * 55)


# ==============================================================================
# RUN THE CODE REVIEW AGENT
# ==============================================================================

orchestrator = CodeReviewOrchestrator()

# Review all code samples
for sample_name, code in CODE_SAMPLES.items():
    print(f"\n{'#'*65}")
    print(f"# Sample: {sample_name}")
    print(f"{'#'*65}")
    print("Code to review:")
    print(code)

    review_result = orchestrator.review(code, code_name=sample_name)
    orchestrator.print_report(review_result)


# ==============================================================================
# LIVE REVIEW: Review custom code
# ==============================================================================

print("\n" + "=" * 65)
print("LIVE REVIEW: Testing a New Code Snippet")
print("=" * 65)

custom_code = '''
def send_email(to, subject, body):
    # Missing: input validation, no error handling
    import smtplib
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.sendmail("from@example.com", to, f"Subject: {subject}\\n\\n{body}")

def get_discount(price, percent):
    discount = price * percent / 100
    final = price - discount
    return final

class UserManager:
    def check_user(self, user_id):
        sql = "SELECT * FROM users WHERE id=" + user_id  # SQL injection!
        return sql
'''

print("Custom code:")
print(custom_code)

custom_review = orchestrator.review(custom_code, "custom_snippet")
orchestrator.print_report(custom_review)


# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 65)
print("SUMMARY - Code Review Agent")
print("=" * 65)

print("""
WHAT WE BUILT:
  A 3-agent code review system:
  - BugAnalyzerAgent:   finds security issues, crashes, bad patterns
  - StyleCheckerAgent:  checks naming, docstrings, line length
  - SuggestionAgent:    synthesizes improvements from all findings
  - CodeReviewOrchestrator: coordinates agents, generates report

PATTERNS USED:
  - Orchestrator + specialists (Lesson 05, Example 05)
  - Shared blackboard (findings passed between agents)
  - Pipeline: Bugs -> Style -> Suggestions -> Report
  - Scoring system (100 - penalties per issue type)

WHAT IT CAUGHT:
  - Division by zero (crash)
  - SQL injection (security critical)
  - MD5 password hashing (security critical)
  - Missing docstrings (style)
  - Magic numbers (style)
  - Long lines (style)

UPGRADE TO PRODUCTION:
  1. Replace regex analysis with AST parsing:
       import ast; tree = ast.parse(code)
       # Walk the AST to find actual code structure

  2. Add more checks: type hint coverage, test coverage, complexity metrics

  3. Use a real LLM for natural language explanations:
       llm.generate(f"Explain this bug: {finding['message']}")

  4. Integrate with GitHub via GitHub API:
       Post review comments directly on a pull request

C# ANALOGY:
  This is like a Roslyn Analyzer in .NET -- static code analysis
  that runs against your code and reports issues. Same pattern:
  visitor that walks the syntax tree and reports diagnostics.

MODULE 11 COMPLETE!
  You have finished all 5 lessons, 5 examples, 5 exercises, and 3 projects.
  You can now:
    - Build single-agent ReAct systems
    - Add short-term and long-term memory
    - Orchestrate multi-agent pipelines
    - Connect agents to tools and external data
    - Build practical applications (assistant, research, code review)
""")

print("=" * 65)
print("END OF PROJECT 03")
print("=" * 65)

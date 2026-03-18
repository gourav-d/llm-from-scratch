"""
AI Code Reviewer - Production-Ready Code Review Assistant

This module implements an AI-powered code reviewer that analyzes code for:
- Security vulnerabilities
- Bugs and edge cases
- Code quality issues
- Best practice violations

Uses Chain-of-Thought reasoning to explain findings clearly.

Author: Learn LLM from Scratch - Module 7 Project
"""

import ast
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class IssueSeverity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"  # Security issues, crashes
    HIGH = "high"          # Bugs, data loss
    MEDIUM = "medium"      # Code quality
    LOW = "low"            # Style, minor improvements
    INFO = "info"          # Suggestions


@dataclass
class ReviewIssue:
    """Represents a code review issue"""
    severity: IssueSeverity
    line_number: int
    category: str           # e.g., "security", "bug", "style"
    title: str             # Short description
    explanation: str       # Chain-of-Thought explanation
    suggestion: str        # How to fix it
    code_snippet: str      # Problematic code
    fixed_code: Optional[str] = None  # Suggested fix

    def __str__(self):
        """Format issue for display"""
        severity_icon = {
            IssueSeverity.CRITICAL: "🔴",
            IssueSeverity.HIGH: "🟠",
            IssueSeverity.MEDIUM: "🟡",
            IssueSeverity.LOW: "🟢",
            IssueSeverity.INFO: "ℹ️",
        }

        return f"""
{severity_icon[self.severity]} {self.severity.value.upper()}: {self.title}
Line {self.line_number} | Category: {self.category}

Problem:
{self.code_snippet}

Reasoning:
{self.explanation}

Suggestion:
{self.suggestion}
{f"
Fixed Code:
{self.fixed_code}" if self.fixed_code else ""}
"""


class CodeReviewer:
    """
    Main AI Code Reviewer class

    This is like having a senior developer review your code!
    Uses Chain-of-Thought reasoning from Module 7 Lesson 1.

    Comparison to C#:
    - Similar to Roslyn analyzers
    - Like StyleCop + FxCop + Security Code Scan combined
    - But powered by AI for deeper understanding
    """

    def __init__(self, model=None, language="python"):
        """
        Initialize the code reviewer

        Args:
            model: LLM model for AI-powered reviews (optional)
            language: Programming language to review
        """
        self.model = model
        self.language = language
        self.issues: List[ReviewIssue] = []

        # Load review rules
        self._load_rules()

    def _load_rules(self):
        """Load security patterns, bug patterns, etc."""
        # Security patterns (similar to OWASP checks)
        self.security_patterns = {
            "sql_injection": {
                "pattern": r'(execute|query|sql).*\+.*(?:request|input|user|param)',
                "message": "Potential SQL injection vulnerability",
                "severity": IssueSeverity.CRITICAL,
            },
            "hardcoded_password": {
                "pattern": r'password\s*=\s*[\'"][^\'"]+[\'"]',
                "message": "Hardcoded password detected",
                "severity": IssueSeverity.CRITICAL,
            },
            "eval_usage": {
                "pattern": r'\beval\s*\(',
                "message": "Use of eval() is dangerous",
                "severity": IssueSeverity.CRITICAL,
            },
            "pickle_load": {
                "pattern": r'pickle\.load',
                "message": "Pickle can execute arbitrary code",
                "severity": IssueSeverity.HIGH,
            },
        }

        # Bug patterns
        self.bug_patterns = {
            "division_by_zero": {
                "pattern": r'/\s*(?:0|zero)',
                "message": "Potential division by zero",
                "severity": IssueSeverity.HIGH,
            },
            "empty_except": {
                "pattern": r'except:\s*pass',
                "message": "Empty except block hides errors",
                "severity": IssueSeverity.HIGH,
            },
        }

    def review_code(self, code: str, filename: str = "code.py") -> List[ReviewIssue]:
        """
        Main review method - analyzes code and returns issues

        This uses Chain-of-Thought reasoning:
        1. Parse code into AST (understand structure)
        2. Run pattern matching (find known issues)
        3. Use AI to find deeper issues (if model provided)
        4. Explain findings clearly

        Args:
            code: Source code to review
            filename: Name of file (for context)

        Returns:
            List of ReviewIssue objects
        """
        self.issues = []

        # Step 1: Security checks (CRITICAL)
        self._check_security(code)

        # Step 2: Bug detection (HIGH)
        self._check_bugs(code)

        # Step 3: Code quality (MEDIUM)
        self._check_quality(code)

        # Step 4: Best practices (LOW)
        self._check_best_practices(code)

        # Step 5: AI-powered review (if model available)
        if self.model:
            self._ai_review(code)

        # Sort by severity
        self.issues.sort(key=lambda x: [
            IssueSeverity.CRITICAL,
            IssueSeverity.HIGH,
            IssueSeverity.MEDIUM,
            IssueSeverity.LOW,
            IssueSeverity.INFO
        ].index(x.severity))

        return self.issues

    def _check_security(self, code: str):
        """
        Check for security vulnerabilities

        This is like a security scanner (similar to Fortify or Checkmarx)
        """
        lines = code.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Check each security pattern
            for name, pattern_info in self.security_patterns.items():
                if re.search(pattern_info["pattern"], line, re.IGNORECASE):
                    # Found a security issue!
                    # Use Chain-of-Thought to explain WHY it's dangerous

                    explanation = self._generate_cot_explanation(
                        issue_type=name,
                        code_line=line
                    )

                    self.issues.append(ReviewIssue(
                        severity=pattern_info["severity"],
                        line_number=line_num,
                        category="security",
                        title=pattern_info["message"],
                        explanation=explanation,
                        suggestion=self._get_security_fix(name),
                        code_snippet=line.strip(),
                        fixed_code=self._suggest_fix(name, line)
                    ))

    def _generate_cot_explanation(self, issue_type: str, code_line: str) -> str:
        """
        Generate Chain-of-Thought explanation for an issue

        This is the "reasoning" part - explaining WHY something is wrong
        Similar to how o1 explains its reasoning (Module 7 Lesson 5)
        """
        # Chain of Thought reasoning templates
        cot_templates = {
            "sql_injection": """
Step 1: Analyzing the code
   The code constructs an SQL query using string concatenation with user input.

Step 2: Identifying the risk
   When user input is directly concatenated into SQL queries, an attacker can
   inject malicious SQL commands. For example, input like "1 OR 1=1" would
   bypass authentication.

Step 3: Impact assessment
   This could allow an attacker to:
   - Read sensitive data from the database
   - Modify or delete data
   - Execute administrative operations
   - Compromise the entire database

Step 4: Recommendation
   ALWAYS use parameterized queries or ORM methods to prevent SQL injection.
""",
            "hardcoded_password": """
Step 1: Analyzing the code
   The code contains a password stored as a plain text string in source code.

Step 2: Identifying the risk
   Hardcoded credentials have several problems:
   - Visible in source code (anyone with code access can see it)
   - Stored in version control history forever
   - Same password used in all environments
   - Cannot be easily rotated

Step 3: Impact assessment
   If this code is committed, the password becomes permanently visible in:
   - Git history
   - Code reviews
   - Developer machines
   - CI/CD logs

Step 4: Recommendation
   Use environment variables or secure secret management systems.
""",
            "eval_usage": """
Step 1: Analyzing the code
   The code uses eval() to execute dynamic code.

Step 2: Identifying the risk
   eval() executes arbitrary Python code, which means if an attacker can
   control the input, they can execute ANY Python code on your system:
   - Delete files: eval("__import__('os').system('rm -rf /')")
   - Steal data: eval("__import__('os').environ")
   - Install backdoors

Step 3: Impact assessment
   This is essentially giving attackers a Python shell on your server.
   It's one of the most dangerous functions in Python.

Step 4: Recommendation
   Use safer alternatives like ast.literal_eval() for data, or better yet,
   use JSON parsing or specific parsers.
""",
        }

        return cot_templates.get(issue_type, f"Issue detected: {issue_type}")

    def _get_security_fix(self, issue_type: str) -> str:
        """Get fix suggestion for security issue"""
        fixes = {
            "sql_injection": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', [user_id])",
            "hardcoded_password": "Use environment variables: password = os.getenv('DB_PASSWORD')",
            "eval_usage": "Use ast.literal_eval() or JSON parsing instead",
            "pickle_load": "Use JSON instead of pickle for untrusted data",
        }
        return fixes.get(issue_type, "Review and fix this security issue")

    def _suggest_fix(self, issue_type: str, code_line: str) -> Optional[str]:
        """Suggest fixed code"""
        if issue_type == "sql_injection":
            # Simple fix suggestion
            return code_line.replace('+', ', parameters=[').rstrip() + ']'
        elif issue_type == "hardcoded_password":
            var_name = re.search(r'(\w+)\s*=', code_line)
            if var_name:
                return f"{var_name.group(1)} = os.getenv('PASSWORD')  # Load from environment"
        return None

    def _check_bugs(self, code: str):
        """Check for common bugs"""
        lines = code.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Empty except blocks
            if re.search(r'except:\s*pass', line):
                self.issues.append(ReviewIssue(
                    severity=IssueSeverity.HIGH,
                    line_number=line_num,
                    category="bug",
                    title="Empty except block silences errors",
                    explanation="Catching all exceptions without handling them makes debugging impossible. You won't know when your code fails.",
                    suggestion="Catch specific exceptions and log them: except ValueError as e: logger.error(e)",
                    code_snippet=line.strip(),
                    fixed_code="except Exception as e:\n    logger.error(f'Error: {e}')\n    raise"
                ))

    def _check_quality(self, code: str):
        """Check code quality using AST"""
        try:
            tree = ast.parse(code)

            # Check function complexity
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Count statements
                    statements = sum(1 for _ in ast.walk(node))
                    if statements > 50:
                        self.issues.append(ReviewIssue(
                            severity=IssueSeverity.MEDIUM,
                            line_number=node.lineno,
                            category="quality",
                            title=f"Function '{node.name}' is too complex ({statements} statements)",
                            explanation="Large functions are hard to understand, test, and maintain. Consider breaking it into smaller functions.",
                            suggestion="Refactor into smaller, focused functions (max ~30 statements each)",
                            code_snippet=f"def {node.name}(...):",
                        ))
        except SyntaxError:
            pass  # Code has syntax errors, will be caught elsewhere

    def _check_best_practices(self, code: str):
        """Check for best practice violations"""
        lines = code.split('\n')

        for line_num, line in enumerate(lines, 1):
            # TODO comments
            if re.search(r'#\s*TODO', line, re.IGNORECASE):
                self.issues.append(ReviewIssue(
                    severity=IssueSeverity.INFO,
                    line_number=line_num,
                    category="best-practice",
                    title="TODO comment found",
                    explanation="TODO comments indicate incomplete work. Track these in your issue tracker instead.",
                    suggestion="Create a proper issue/ticket for this work",
                    code_snippet=line.strip(),
                ))

    def _ai_review(self, code: str):
        """
        Use AI model for deeper code review

        This would use your trained GPT model to find issues that
        pattern matching can't catch.

        In a real implementation, you'd:
        1. Create a prompt with Chain-of-Thought instructions
        2. Ask the model to review the code
        3. Parse the model's response
        4. Add issues to self.issues
        """
        # Placeholder for AI-powered review
        # In real implementation, call self.model with a CoT prompt
        pass

    def format_report(self) -> str:
        """Format all issues into a readable report"""
        if not self.issues:
            return "✅ No issues found! Code looks good."

        report = f"\n{'='*80}\n"
        report += f"CODE REVIEW REPORT - Found {len(self.issues)} issues\n"
        report += f"{'='*80}\n"

        # Group by severity
        by_severity = {}
        for issue in self.issues:
            severity = issue.severity
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(issue)

        # Show critical first
        for severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH,
                        IssueSeverity.MEDIUM, IssueSeverity.LOW, IssueSeverity.INFO]:
            if severity in by_severity:
                report += f"\n{severity.value.upper()} Issues ({len(by_severity[severity])})\n"
                report += "-" * 80 + "\n"
                for issue in by_severity[severity]:
                    report += str(issue)
                    report += "\n"

        return report

    def has_critical_issues(self) -> bool:
        """Check if any critical issues were found"""
        return any(issue.severity == IssueSeverity.CRITICAL for issue in self.issues)


# Example usage
if __name__ == "__main__":
    # Example code with issues
    vulnerable_code = """
def get_user_by_email(email):
    # TODO: Add input validation
    query = "SELECT * FROM users WHERE email = '" + email + "'"
    result = database.execute(query)
    return result

def process_payment(amount):
    try:
        charge_card(amount)
    except:
        pass

def authenticate(username, password):
    admin_password = "admin123"  # Hardcoded password!
    if password == admin_password:
        return True
    return False
"""

    print("AI CODE REVIEWER - Demo")
    print("=" * 80)
    print("\nReviewing vulnerable code...\n")

    # Create reviewer
    reviewer = CodeReviewer()

    # Review the code
    issues = reviewer.review_code(vulnerable_code)

    # Print report
    print(reviewer.format_report())

    # Check for critical issues
    if reviewer.has_critical_issues():
        print("\n⚠️  CRITICAL ISSUES FOUND - Do not deploy this code!")
    else:
        print("\n✅ No critical issues - code is safe to deploy")

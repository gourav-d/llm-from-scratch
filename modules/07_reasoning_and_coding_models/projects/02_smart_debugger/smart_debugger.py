"""
Smart Bug Debugger - AI-Powered Debugging Assistant

Uses Chain-of-Thought reasoning to analyze errors and suggest fixes.

Author: Learn LLM from Scratch - Module 7 Project
"""

import re
import traceback
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass
class DebugAnalysis:
    """Result of debugging analysis"""
    error_type: str
    error_message: str
    root_cause: str
    reasoning_steps: List[str]  # Chain-of-Thought steps
    fix_suggestions: List[str]
    code_snippet: Optional[str] = None
    explanation: str = ""

    def __str__(self):
        """Format analysis for display"""
        output = "\n" + "="*80 + "\n"
        output += "🔍 SMART DEBUGGER ANALYSIS\n"
        output += "="*80 + "\n\n"

        output += f"Error Type: {self.error_type}\n"
        output += f"Message: {self.error_message}\n\n"

        output += "CHAIN-OF-THOUGHT REASONING:\n"
        output += "-"*80 + "\n"
        for i, step in enumerate(self.reasoning_steps, 1):
            output += f"Step {i}: {step}\n\n"

        output += "ROOT CAUSE:\n"
        output += "-"*80 + "\n"
        output += f"{self.root_cause}\n\n"

        if self.fix_suggestions:
            output += "FIX SUGGESTIONS (ranked by best practice):\n"
            output += "-"*80 + "\n"
            for i, fix in enumerate(self.fix_suggestions, 1):
                output += f"\nOption {i}:\n{fix}\n"

        return output


class SmartDebugger:
    """
    AI-powered debugger that uses Chain-of-Thought reasoning

    Similar to C#:
    - Like Visual Studio's Exception Helper, but AI-powered
    - Explains errors like a senior developer would
    - Uses reasoning from Module 7 Lesson 1
    """

    def __init__(self):
        """Initialize the debugger"""
        self.error_patterns = self._load_error_patterns()
        self.fix_templates = self._load_fix_templates()

    def _load_error_patterns(self) -> Dict:
        """Load common error patterns and their explanations"""
        return {
            "IndexError": {
                "common_causes": [
                    "Accessing index >= len(list)",
                    "Using len(list) as index (should be len(list)-1)",
                    "Loop counter off by one",
                    "Empty list access",
                ],
                "explanation": "Trying to access an array index that doesn't exist"
            },
            "KeyError": {
                "common_causes": [
                    "Dictionary key doesn't exist",
                    "Typo in key name",
                    "Missing data from API/database",
                ],
                "explanation": "Trying to access a dictionary key that doesn't exist"
            },
            "AttributeError": {
                "common_causes": [
                    "Object is None (null reference)",
                    "Object doesn't have that attribute",
                    "Typo in attribute name",
                    "Wrong object type",
                ],
                "explanation": "Trying to access an attribute that doesn't exist"
            },
            "TypeError": {
                "common_causes": [
                    "Wrong number of arguments to function",
                    "Wrong type passed to function",
                    "Calling non-callable object",
                    "Unsupported operation for type",
                ],
                "explanation": "Operation not supported for the given types"
            },
            "ZeroDivisionError": {
                "common_causes": [
                    "Dividing by zero",
                    "Empty list in average calculation",
                    "Missing validation",
                ],
                "explanation": "Attempting to divide by zero"
            },
        }

    def _load_fix_templates(self) -> Dict:
        """Load fix templates for common errors"""
        return {
            "IndexError": [
                {
                    "name": "Guard clause",
                    "template": "if index < len(list):\n    item = list[index]",
                    "explanation": "Check bounds before accessing",
                },
                {
                    "name": "Use negative indexing",
                    "template": "item = list[-1]  # Last item",
                    "explanation": "Python supports negative indices",
                },
            ],
            "KeyError": [
                {
                    "name": "Use get() with default",
                    "template": "value = dict.get(key, default_value)",
                    "explanation": "Returns default if key missing",
                },
                {
                    "name": "Check key exists",
                    "template": "if key in dict:\n    value = dict[key]",
                    "explanation": "Explicit check before access",
                },
            ],
            "AttributeError": [
                {
                    "name": "None check",
                    "template": "if obj is not None:\n    obj.attribute",
                    "explanation": "Check for None before accessing",
                },
                {
                    "name": "Use getattr()",
                    "template": "value = getattr(obj, 'attr', default)",
                    "explanation": "Safe attribute access with default",
                },
            ],
        }

    def analyze_error(
        self,
        error_message: str,
        code: Optional[str] = None,
        stack_trace: Optional[str] = None
    ) -> DebugAnalysis:
        """
        Main analysis method - uses Chain-of-Thought reasoning

        Args:
            error_message: The error message
            code: Source code where error occurred
            stack_trace: Full stack trace

        Returns:
            DebugAnalysis with reasoning and suggestions
        """
        # Step 1: Parse error type and message
        error_type, parsed_message = self._parse_error(error_message)

        # Step 2: Build Chain-of-Thought reasoning
        reasoning_steps = self._generate_reasoning_chain(
            error_type,
            parsed_message,
            code,
            stack_trace
        )

        # Step 3: Determine root cause
        root_cause = self._find_root_cause(
            error_type,
            code,
            reasoning_steps
        )

        # Step 4: Generate fix suggestions
        fix_suggestions = self._generate_fixes(
            error_type,
            code,
            root_cause
        )

        return DebugAnalysis(
            error_type=error_type,
            error_message=parsed_message,
            root_cause=root_cause,
            reasoning_steps=reasoning_steps,
            fix_suggestions=fix_suggestions,
            code_snippet=code,
        )

    def _parse_error(self, error_message: str) -> Tuple[str, str]:
        """Parse error type and message from error string"""
        # Try to extract error type
        match = re.search(r'(\w+Error):\s*(.+)', error_message)
        if match:
            return match.group(1), match.group(2)

        # Fallback
        return "Unknown", error_message

    def _generate_reasoning_chain(
        self,
        error_type: str,
        message: str,
        code: Optional[str],
        stack_trace: Optional[str]
    ) -> List[str]:
        """
        Generate Chain-of-Thought reasoning steps

        This is the core reasoning from Module 7 Lesson 1
        """
        steps = []

        # Step 1: Understand the error
        pattern_info = self.error_patterns.get(error_type, {})
        explanation = pattern_info.get(
            "explanation",
            "An error occurred during execution"
        )

        steps.append(
            f"Understanding the error\n"
            f"   {error_type}: {explanation}\n"
            f"   Message: {message}"
        )

        # Step 2: Analyze the context
        if code:
            steps.append(
                f"Analyzing the code\n"
                f"   Looking at where the error occurred..."
            )

            # Simple code analysis
            if error_type == "IndexError":
                if "len(" in code and "[" in code:
                    steps.append(
                        f"Pattern identified\n"
                        f"   Code uses len() to calculate index\n"
                        f"   This is a common off-by-one error"
                    )
            elif error_type == "ZeroDivisionError":
                if "/" in code:
                    steps.append(
                        f"Pattern identified\n"
                        f"   Code performs division without zero check"
                    )

        # Step 3: Common causes
        if error_type in self.error_patterns:
            causes = self.error_patterns[error_type]["common_causes"]
            steps.append(
                f"Common causes of {error_type}\n" +
                "\n".join(f"   - {cause}" for cause in causes)
            )

        # Step 4: Impact analysis
        severity = "HIGH" if error_type in ["IndexError", "KeyError", "AttributeError"] else "MEDIUM"
        steps.append(
            f"Impact assessment\n"
            f"   Severity: {severity}\n"
            f"   This will crash the program if not handled"
        )

        return steps

    def _find_root_cause(
        self,
        error_type: str,
        code: Optional[str],
        reasoning_steps: List[str]
    ) -> str:
        """Determine the root cause of the error"""
        if not code:
            return f"The code raised a {error_type}. Without seeing the code, the most likely causes are: " + \
                   ", ".join(self.error_patterns.get(error_type, {}).get("common_causes", []))

        # Analyze code for specific patterns
        if error_type == "IndexError":
            if "len(" in code and "[" in code:
                return (
                    "The code uses len() to calculate an index, which causes an off-by-one error. "
                    "Arrays are zero-indexed, so valid indices are 0 to len(array)-1. "
                    "Using len(array) as an index accesses one past the end."
                )

        if error_type == "ZeroDivisionError":
            if "/" in code and "if" not in code:
                return (
                    "The code performs division without checking if the divisor is zero. "
                    "This commonly happens with calculations involving counts or averages "
                    "when the input might be empty."
                )

        if error_type == "AttributeError":
            if "None" in str(code) or not code.strip():
                return (
                    "The object is None (null). This typically means the object was never "
                    "initialized, or a function returned None instead of an object."
                )

        # Default root cause
        return f"A {error_type} occurred. See the reasoning steps above for likely causes."

    def _generate_fixes(
        self,
        error_type: str,
        code: Optional[str],
        root_cause: str
    ) -> List[str]:
        """Generate fix suggestions"""
        fixes = []

        if error_type in self.fix_templates:
            for template in self.fix_templates[error_type]:
                fix = f"**{template['name']}** (Recommended)\n"
                fix += f"```python\n{template['template']}\n```\n"
                fix += f"Explanation: {template['explanation']}"
                fixes.append(fix)

        # Add generic best practice
        fixes.append(
            f"**Add error handling**\n"
            f"```python\n"
            f"try:\n"
            f"    # Your code here\n"
            f"except {error_type} as e:\n"
            f"    logger.error(f'Error: {{e}}')\n"
            f"    # Handle gracefully\n"
            f"```\n"
            f"Explanation: Catch and handle the error gracefully"
        )

        return fixes

    def explain_stack_trace(self, stack_trace: str) -> str:
        """Convert complex stack trace into plain English"""
        lines = stack_trace.strip().split('\n')

        explanation = "STACK TRACE EXPLANATION:\n"
        explanation += "="*80 + "\n\n"

        explanation += "Reading from bottom to top (reverse chronological order):\n\n"

        # Parse stack trace
        for i, line in enumerate(reversed(lines)):
            if 'File "' in line:
                # Extract filename and line number
                match = re.search(r'File "([^"]+)", line (\d+)', line)
                if match:
                    file, line_num = match.groups()
                    explanation += f"{i+1}. In {file}, line {line_num}\n"

            elif line.strip() and not line.startswith('Traceback'):
                # Code line or error message
                explanation += f"   → {line.strip()}\n\n"

        return explanation


# Example usage
if __name__ == "__main__":
    print("SMART DEBUGGER - Demo")
    print("="*80)

    debugger = SmartDebugger()

    # Example 1: IndexError
    print("\n\nExample 1: IndexError\n")
    code1 = """
def get_last_item(items):
    index = len(items)
    return items[index]
"""
    error1 = "IndexError: list index out of range"

    analysis1 = debugger.analyze_error(error1, code1)
    print(analysis1)

    # Example 2: ZeroDivisionError
    print("\n\nExample 2: ZeroDivisionError\n")
    code2 = """
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
"""
    error2 = "ZeroDivisionError: division by zero"

    analysis2 = debugger.analyze_error(error2, code2)
    print(analysis2)

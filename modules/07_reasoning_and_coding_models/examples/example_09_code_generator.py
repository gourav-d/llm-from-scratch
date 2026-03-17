"""
Example 09: Code Generation & Completion (Mini-Copilot)

This example demonstrates:
1. Natural language to code generation
2. Docstring to implementation
3. Context-aware code completion
4. Building a mini-Copilot system
5. Syntax validation and ranking

For .NET developers: This is like building IntelliSense + GitHub Copilot!
"""

import re
import ast
from typing import List, Tuple, Dict, Optional


# ============================================================================
# PART 1: SIMPLE CODE GENERATOR (Template-Based)
# ============================================================================

class TemplateCodeGenerator:
    """
    Generates code using template matching
    Simple but effective for common patterns

    Like: Visual Studio code snippets but AI-powered
    """

    def __init__(self):
        """Initialize with common templates"""
        # Map action words to operators
        # Example: "add" → "+"
        self.actions = {
            "add": ("+", "add"),
            "adds": ("+", "add"),
            "subtract": ("-", "subtract"),
            "subtracts": ("-", "subtract"),
            "multiply": ("*", "multiply"),
            "multiplies": ("*", "multiply"),
            "divide": ("/", "divide"),
            "divides": ("/", "divide"),
            "power": ("**", "power"),
            "modulo": ("%", "modulo")
        }

        # Code templates
        # {name}, {param1}, {param2}, {operator} are placeholders
        self.templates = {
            "binary_operation": """def {name}({param1}, {param2}):
    \"\"\"
    {description}

    Args:
        {param1}: First operand
        {param2}: Second operand

    Returns:
        Result of {operation}
    \"\"\"
    return {param1} {operator} {param2}""",

            "unary_operation": """def {name}({param}):
    \"\"\"
    {description}

    Args:
        {param}: Input value

    Returns:
        {return_desc}
    \"\"\"
    return {operation}({param})"""
        }

    def generate(self, description: str) -> str:
        """
        Generate code from natural language

        Args:
            description: Natural language like "function that adds two numbers"

        Returns:
            Generated Python code
        """
        print(f"\n{'='*60}")
        print(f"Template Generator: {description}")
        print(f"{'='*60}")

        # Parse description
        words = description.lower().split()

        # Find action word
        operator = None
        func_name = None
        operation = None

        for word in words:
            if word in self.actions:
                operator, func_name = self.actions[word]
                operation = word
                break

        if not operator:
            return "# Error: Could not understand the action"

        # Determine parameter names from context
        param1 = "a"
        param2 = "b"

        # Check for specific mentions of parameters
        if "two numbers" in description or "two values" in description:
            param1, param2 = "x", "y"
        elif "items" in description:
            param1, param2 = "item1", "item2"

        # Generate code from template
        code = self.templates["binary_operation"].format(
            name=func_name,
            param1=param1,
            param2=param2,
            operator=operator,
            description=description,
            operation=operation
        )

        print("Generated code:")
        print(code)

        return code


# ============================================================================
# PART 2: DOCSTRING PARSER
# ============================================================================

class DocstringParser:
    """
    Parses Python docstrings to extract metadata
    Understands what function should do from documentation

    Like: Reading XML doc comments in C#
    """

    def parse(self, docstring: str) -> Dict:
        """
        Extract structured information from docstring

        Args:
            docstring: Raw docstring text (triple-quoted string)

        Returns:
            Dictionary with:
                - description: What function does
                - args: List of parameters
                - returns: Return type and description
                - raises: Exceptions raised
        """
        metadata = {
            "description": "",
            "args": [],
            "returns": None,
            "raises": []
        }

        if not docstring:
            return metadata

        lines = docstring.strip().split('\n')

        # First non-empty line is short description
        for line in lines:
            if line.strip():
                metadata["description"] = line.strip()
                break

        # Parse sections
        current_section = None

        for line in lines[1:]:
            stripped = line.strip()

            # Detect sections
            if stripped.startswith("Args:"):
                current_section = "args"
                continue
            elif stripped.startswith("Returns:"):
                current_section = "returns"
                continue
            elif stripped.startswith("Raises:"):
                current_section = "raises"
                continue

            # Parse based on current section
            if current_section == "args" and stripped:
                # Format: "param_name (type): description"
                # Example: "n (int): The number to process"
                match = re.match(r'(\w+)\s*\(([^)]+)\):\s*(.+)', stripped)
                if match:
                    metadata["args"].append({
                        "name": match.group(1),
                        "type": match.group(2),
                        "description": match.group(3)
                    })

            elif current_section == "returns" and stripped:
                # Format: "type: description"
                # Example: "int: The factorial result"
                match = re.match(r'([^:]+):\s*(.+)', stripped)
                if match:
                    metadata["returns"] = {
                        "type": match.group(1).strip(),
                        "description": match.group(2).strip()
                    }
                else:
                    # Sometimes just description
                    if not metadata["returns"]:
                        metadata["returns"] = {
                            "type": "Any",
                            "description": stripped
                        }

            elif current_section == "raises" and stripped:
                # Format: "ExceptionType: description"
                match = re.match(r'(\w+):\s*(.+)', stripped)
                if match:
                    metadata["raises"].append({
                        "exception": match.group(1),
                        "description": match.group(2)
                    })

        return metadata


# ============================================================================
# PART 3: CONTEXT GATHERER
# ============================================================================

class ContextGatherer:
    """
    Gathers context for code completion
    Looks at surrounding code to understand what to suggest

    Like: IntelliSense analyzing your current file in Visual Studio
    """

    def gather_context(self, file_content: str, cursor_position: Tuple[int, int]) -> Dict:
        """
        Collect context around cursor position

        Args:
            file_content: Full file content as string
            cursor_position: (line_number, column_number) - 0-indexed

        Returns:
            Dictionary with:
                - before: Lines before cursor
                - after: Lines after cursor
                - imports: Import statements
                - functions: Function names
                - classes: Class names
                - variables: Variable names in scope
        """
        lines = file_content.split('\n')
        cursor_line, cursor_col = cursor_position

        context = {
            "before": [],
            "after": [],
            "imports": [],
            "functions": [],
            "classes": [],
            "variables": []
        }

        # Get surrounding lines
        # Before: Up to 20 lines before cursor
        start_line = max(0, cursor_line - 20)
        context["before"] = lines[start_line:cursor_line]

        # After: Up to 10 lines after cursor
        context["after"] = lines[cursor_line + 1:cursor_line + 11]

        # Extract imports
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                context["imports"].append(stripped)

        # Parse AST to find definitions
        # AST = Abstract Syntax Tree (code structure)
        try:
            tree = ast.parse(file_content)

            # Walk through all nodes in the tree
            for node in ast.walk(tree):
                # Find function definitions
                if isinstance(node, ast.FunctionDef):
                    context["functions"].append(node.name)

                # Find class definitions
                elif isinstance(node, ast.ClassDef):
                    context["classes"].append(node.name)

                # Find variable assignments
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            context["variables"].append(target.id)

        except SyntaxError:
            # File might have incomplete code (common during typing)
            # That's okay - we'll work with what we have
            pass

        return context

    def build_prompt(self, file_content: str, cursor_position: Tuple[int, int]) -> str:
        """
        Build prompt for model from context
        Includes code before cursor

        Args:
            file_content: Full file
            cursor_position: Where cursor is

        Returns:
            Prompt string (code before cursor)
        """
        lines = file_content.split('\n')
        cursor_line, cursor_col = cursor_position

        # Get all lines up to cursor
        prompt_lines = lines[:cursor_line]

        # Add partial line up to cursor column
        if cursor_line < len(lines):
            current_line = lines[cursor_line][:cursor_col]
            if current_line:
                prompt_lines.append(current_line)

        return '\n'.join(prompt_lines)


# ============================================================================
# PART 4: SYNTAX VALIDATOR
# ============================================================================

class SyntaxValidator:
    """
    Validates syntax of generated code
    Catches errors before showing to user

    Like: Roslyn compiler diagnostics in C#
    """

    def validate(self, code: str, language: str = "python") -> Tuple[bool, Optional[str]]:
        """
        Check if code is syntactically valid

        Args:
            code: Code to validate
            language: Programming language (default: python)

        Returns:
            (is_valid, error_message)
                is_valid: True if valid, False otherwise
                error_message: None if valid, error description if invalid
        """
        if language == "python":
            return self._validate_python(code)
        else:
            return (False, f"Unsupported language: {language}")

    def _validate_python(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python syntax using AST parser

        Args:
            code: Python code

        Returns:
            (is_valid, error_message)
        """
        try:
            # Try parsing code into AST
            # If successful, syntax is valid
            ast.parse(code)
            return (True, None)

        except SyntaxError as e:
            # Syntax error - capture details
            error_msg = f"Line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n  {e.text.strip()}"
            return (False, error_msg)

        except Exception as e:
            # Other parse errors
            return (False, f"Parse error: {str(e)}")

    def auto_fix(self, code: str) -> str:
        """
        Attempt to automatically fix common syntax errors

        Args:
            code: Potentially broken code

        Returns:
            Fixed code (or original if can't fix)
        """
        is_valid, error = self.validate(code)

        if is_valid:
            return code

        # Try common fixes in order
        fixes = [
            self._fix_missing_colon,
            self._fix_missing_parenthesis,
            self._fix_missing_quotes,
        ]

        for fix_func in fixes:
            fixed_code = fix_func(code)
            is_valid, _ = self.validate(fixed_code)

            if is_valid:
                print(f"  ✓ Auto-fixed using {fix_func.__name__}")
                return fixed_code

        # Couldn't fix
        return code

    def _fix_missing_colon(self, code: str) -> str:
        """Add missing colons after if/for/def/class/while"""
        lines = code.split('\n')
        fixed = []

        keywords = ('def ', 'class ', 'if ', 'elif ', 'else',
                   'for ', 'while ', 'try', 'except', 'finally', 'with ')

        for line in lines:
            stripped = line.strip()

            # Check if line needs a colon
            if any(stripped.startswith(kw) for kw in keywords):
                if not stripped.endswith(':') and not stripped.endswith('\\'):
                    line = line + ':'

            fixed.append(line)

        return '\n'.join(fixed)

    def _fix_missing_parenthesis(self, code: str) -> str:
        """Add missing closing parentheses"""
        # Count opening vs closing
        open_count = code.count('(')
        close_count = code.count(')')

        if open_count > close_count:
            code += ')' * (open_count - close_count)

        return code

    def _fix_missing_quotes(self, code: str) -> str:
        """Add missing closing quotes"""
        # Count single quotes
        single_quotes = code.count("'")
        if single_quotes % 2 != 0:
            code += "'"

        # Count double quotes
        double_quotes = code.count('"')
        if double_quotes % 2 != 0:
            code += '"'

        return code


# ============================================================================
# PART 5: COMPLETION RANKER
# ============================================================================

class CompletionRanker:
    """
    Ranks code completion candidates
    Shows best suggestions first

    Like: IntelliSense ranking in Visual Studio
    """

    def __init__(self):
        self.validator = SyntaxValidator()

    def rank_candidates(
        self,
        candidates: List[Tuple[str, float]],
        context: Dict
    ) -> List[str]:
        """
        Score and rank completion candidates

        Args:
            candidates: List of (code, probability) tuples
            context: Context from ContextGatherer

        Returns:
            Sorted list of candidates (best first)
        """
        scored = []

        for code, prob in candidates:
            score = 0.0

            # Factor 1: Model confidence (40% weight)
            # Higher probability from model = better
            score += prob * 40

            # Factor 2: Syntax validity (20 points)
            is_valid, _ = self.validator.validate(code)
            if is_valid:
                score += 20

            # Factor 3: Uses existing variables (20% weight)
            var_score = self._variable_usage_score(code, context)
            score += var_score * 20

            # Factor 4: Style consistency (20% weight)
            style_score = self._style_score(code, context)
            score += style_score * 20

            scored.append((code, score, prob, is_valid))

        # Sort by score (descending = best first)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return just the code
        return [code for code, score, prob, valid in scored]

    def _variable_usage_score(self, code: str, context: Dict) -> float:
        """
        Score based on using existing variables

        Args:
            code: Generated code
            context: Available context

        Returns:
            Score from 0.0 to 1.0
        """
        # Extract existing variable names
        existing_vars = set(context.get("variables", []))

        # Also check variables in "before" lines
        for line in context.get("before", []):
            match = re.match(r'\s*(\w+)\s*=', line)
            if match:
                existing_vars.add(match.group(1))

        if not existing_vars:
            return 1.0  # No variables to check against

        # Find variable references in generated code
        # Pattern: word boundaries around identifiers
        used_vars = set(re.findall(r'\b([a-z_]\w*)\b', code.lower()))

        # Calculate overlap
        overlap = used_vars & existing_vars

        return len(overlap) / len(existing_vars)

    def _style_score(self, code: str, context: Dict) -> float:
        """
        Score based on style consistency

        Args:
            code: Generated code
            context: Existing code context

        Returns:
            Score from 0.0 to 1.0
        """
        score = 0.0

        before_lines = context.get("before", [])

        if not before_lines:
            return 1.0

        # Check 1: Indentation style (tabs vs spaces)
        context_indent = self._detect_indentation(before_lines)
        code_indent = self._detect_indentation([code])

        if context_indent == code_indent or code_indent == "unknown":
            score += 0.5

        # Check 2: Naming convention (snake_case vs camelCase)
        context_naming = self._detect_naming_style(before_lines)
        code_naming = self._detect_naming_style([code])

        if context_naming == code_naming or code_naming == "unknown":
            score += 0.5

        return score

    def _detect_indentation(self, lines: List[str]) -> str:
        """Detect if code uses tabs or spaces"""
        for line in lines:
            if line.startswith('\t'):
                return 'tabs'
            elif line.startswith('    '):  # 4 spaces
                return 'spaces'
        return 'unknown'

    def _detect_naming_style(self, lines: List[str]) -> str:
        """Detect naming convention (snake_case vs camelCase)"""
        text = '\n'.join(lines)

        # Count snake_case identifiers
        snake_count = len(re.findall(r'\b[a-z]+_[a-z]+\w*\b', text))

        # Count camelCase identifiers
        camel_count = len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]+\b', text))

        if snake_count > camel_count:
            return 'snake_case'
        elif camel_count > snake_count:
            return 'camelCase'
        else:
            return 'unknown'


# ============================================================================
# PART 6: SIMPLE CODE GENERATOR (Simulated)
# ============================================================================

class SimpleCodeGenerator:
    """
    Simulates code generation using patterns
    (In real system, this would use a transformer model)

    For demonstration purposes - shows the interface
    """

    def __init__(self):
        """Initialize with simple pattern matching"""
        self.patterns = {
            "factorial": self._generate_factorial,
            "fibonacci": self._generate_fibonacci,
            "prime": self._generate_prime,
            "sort": self._generate_sort,
            "reverse": self._generate_reverse,
        }

    def generate_code(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Generate code from prompt
        (Simplified version - real one uses transformer model)

        Args:
            prompt: Code prefix to complete
            max_length: Max tokens (unused in this simple version)
            temperature: Randomness (unused in this simple version)

        Returns:
            Generated code
        """
        # Check for known patterns in prompt
        prompt_lower = prompt.lower()

        for pattern, generator in self.patterns.items():
            if pattern in prompt_lower:
                return generator(prompt)

        # Default: return simple completion
        return self._generate_default(prompt)

    def _generate_factorial(self, prompt: str) -> str:
        """Generate factorial function"""
        # Check if function is already defined
        if "def " in prompt:
            # Just add implementation
            return prompt + """
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""
        else:
            return prompt + """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""

    def _generate_fibonacci(self, prompt: str) -> str:
        """Generate fibonacci function"""
        return prompt + """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)"""

    def _generate_prime(self, prompt: str) -> str:
        """Generate prime checking function"""
        return prompt + """
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True"""

    def _generate_sort(self, prompt: str) -> str:
        """Generate sorting code"""
        return prompt + """
    return sorted(items)"""

    def _generate_reverse(self, prompt: str) -> str:
        """Generate reverse code"""
        return prompt + """
    return items[::-1]"""

    def _generate_default(self, prompt: str) -> str:
        """Default completion when no pattern matches"""
        # Just add a simple pass statement
        return prompt + "\n    pass"


# ============================================================================
# PART 7: MINI-COPILOT (Complete System)
# ============================================================================

class MiniCopilot:
    """
    Mini version of GitHub Copilot
    Complete code completion system

    Combines all components:
    - Context gathering
    - Code generation
    - Candidate ranking
    - Syntax validation
    """

    def __init__(self):
        """Initialize all components"""
        print("\n" + "="*60)
        print("MINI-COPILOT INITIALIZING...")
        print("="*60)

        self.generator = SimpleCodeGenerator()
        self.context_gatherer = ContextGatherer()
        self.ranker = CompletionRanker()
        self.validator = SyntaxValidator()

        print("✓ Code generator loaded")
        print("✓ Context gatherer ready")
        print("✓ Ranker initialized")
        print("✓ Validator ready")
        print("\nMini-Copilot ready! 🚀")

    def complete(
        self,
        file_content: str,
        cursor_position: Tuple[int, int],
        num_candidates: int = 3
    ) -> List[str]:
        """
        Generate code completions for cursor position

        Args:
            file_content: Full file content
            cursor_position: (line, column) where cursor is
            num_candidates: How many suggestions to return

        Returns:
            List of completion suggestions (best first)
        """
        print("\n" + "="*60)
        print("GENERATING COMPLETIONS...")
        print("="*60)

        # Step 1: Gather context
        print("\n[Step 1] Gathering context...")
        context = self.context_gatherer.gather_context(file_content, cursor_position)

        print(f"  ├─ Imports: {len(context['imports'])}")
        print(f"  ├─ Functions: {len(context['functions'])}")
        print(f"  ├─ Classes: {len(context['classes'])}")
        print(f"  └─ Variables: {len(context['variables'])}")

        # Step 2: Build prompt
        print("\n[Step 2] Building prompt...")
        prompt = self.context_gatherer.build_prompt(file_content, cursor_position)
        print(f"  └─ Prompt length: {len(prompt)} characters")

        # Step 3: Generate candidates with varying creativity
        print(f"\n[Step 3] Generating {num_candidates} candidates...")
        candidates = []

        for i in range(num_candidates):
            # Vary temperature for diversity
            temp = 0.3 + (i * 0.2)  # 0.3, 0.5, 0.7, ...

            completion = self.generator.generate_code(
                prompt,
                temperature=temp
            )

            # Extract new code (after prompt)
            new_code = completion[len(prompt):].strip()

            # Simulated probability (higher for lower temperature)
            probability = 1.0 - (temp * 0.3)

            candidates.append((new_code, probability))
            print(f"  ├─ Candidate {i+1}: temp={temp:.2f}, prob={probability:.2f}")

        print(f"  └─ Generated {len(candidates)} candidates")

        # Step 4: Rank candidates
        print("\n[Step 4] Ranking candidates...")
        ranked = self.ranker.rank_candidates(candidates, context)
        print(f"  └─ Ranked by quality score")

        # Step 5: Validate syntax
        print("\n[Step 5] Validating syntax...")
        valid_completions = []

        for i, completion in enumerate(ranked, 1):
            # Test with surrounding context
            test_code = '\n'.join(context["before"]) + '\n' + completion

            is_valid, error = self.validator.validate(test_code)

            if is_valid:
                valid_completions.append(completion)
                print(f"  ├─ ✓ Candidate {i}: Valid")
            else:
                print(f"  ├─ ✗ Candidate {i}: {error}")

                # Try auto-fix
                fixed = self.validator.auto_fix(test_code)
                if fixed != test_code:
                    fixed_new = fixed[len('\n'.join(context["before"])) + 1:]
                    valid_completions.append(fixed_new)

        # Return top candidates
        result = valid_completions[:num_candidates]

        print(f"\n[Result] Returning {len(result)} completions")
        print("="*60 + "\n")

        return result


# ============================================================================
# DEMONSTRATION & TESTING
# ============================================================================

def demo_template_generator():
    """Demonstrate template-based code generation"""
    print("\n" + "="*70)
    print("DEMO 1: TEMPLATE-BASED CODE GENERATION")
    print("="*70)

    generator = TemplateCodeGenerator()

    # Test cases
    test_cases = [
        "function that adds two numbers",
        "function that multiplies two values",
        "function that divides two items",
    ]

    for description in test_cases:
        code = generator.generate(description)
        print()


def demo_docstring_parser():
    """Demonstrate docstring parsing"""
    print("\n" + "="*70)
    print("DEMO 2: DOCSTRING PARSING")
    print("="*70)

    parser = DocstringParser()

    # Test docstring
    docstring = """
    Calculate the factorial of a number.

    Args:
        n (int): The number to calculate factorial for

    Returns:
        int: The factorial result

    Raises:
        ValueError: If n is negative
    """

    print("\nInput docstring:")
    print(docstring)

    print("\nParsed metadata:")
    metadata = parser.parse(docstring)

    print(f"  Description: {metadata['description']}")
    print(f"  Arguments:")
    for arg in metadata['args']:
        print(f"    - {arg['name']} ({arg['type']}): {arg['description']}")

    if metadata['returns']:
        print(f"  Returns: {metadata['returns']['type']} - {metadata['returns']['description']}")

    if metadata['raises']:
        print(f"  Raises:")
        for exc in metadata['raises']:
            print(f"    - {exc['exception']}: {exc['description']}")


def demo_context_gatherer():
    """Demonstrate context gathering"""
    print("\n" + "="*70)
    print("DEMO 3: CONTEXT GATHERING")
    print("="*70)

    gatherer = ContextGatherer()

    # Sample file
    file_content = """
import math
import sys

def helper_function(x):
    return x * 2

class DataProcessor:
    def __init__(self):
        self.data = []

    def process(self, items):
        result = []
        for item in items:
            # CURSOR HERE (line 14, column 12)

        return result
"""

    cursor_position = (14, 12)  # Line 14, column 12

    print(f"\nFile content:\n{file_content}")
    print(f"\nCursor at: Line {cursor_position[0]}, Column {cursor_position[1]}")

    context = gatherer.gather_context(file_content, cursor_position)

    print("\nGathered context:")
    print(f"  Imports: {context['imports']}")
    print(f"  Functions: {context['functions']}")
    print(f"  Classes: {context['classes']}")
    print(f"  Variables: {context['variables']}")
    print(f"  Lines before cursor: {len(context['before'])}")
    print(f"  Lines after cursor: {len(context['after'])}")


def demo_syntax_validator():
    """Demonstrate syntax validation"""
    print("\n" + "="*70)
    print("DEMO 4: SYNTAX VALIDATION & AUTO-FIX")
    print("="*70)

    validator = SyntaxValidator()

    # Test cases
    test_cases = [
        ("Valid code", "def add(a, b):\n    return a + b"),
        ("Missing colon", "def add(a, b)\n    return a + b"),
        ("Missing paren", "result = calculate(x, y"),
        ("Missing quote", "name = 'John"),
    ]

    for name, code in test_cases:
        print(f"\n{name}:")
        print(f"  Code: {code.replace(chr(10), '\\n')}")

        is_valid, error = validator.validate(code)
        print(f"  Valid: {is_valid}")

        if not is_valid:
            print(f"  Error: {error}")

            # Try auto-fix
            fixed = validator.auto_fix(code)
            if fixed != code:
                print(f"  Fixed: {fixed.replace(chr(10), '\\n')}")


def demo_mini_copilot():
    """Demonstrate complete Mini-Copilot system"""
    print("\n" + "="*70)
    print("DEMO 5: MINI-COPILOT (COMPLETE SYSTEM)")
    print("="*70)

    copilot = MiniCopilot()

    # Test case 1: Complete factorial function
    print("\n\nTest Case 1: Complete factorial function")
    print("-" * 60)

    file1 = """
def factorial(n):
    \"\"\"Calculate factorial of n.\"\"\"
"""

    cursor1 = (2, 4)  # After docstring

    print(f"Input code:")
    print(file1)
    print(f"Cursor at: Line {cursor1[0]}, Column {cursor1[1]}")

    completions1 = copilot.complete(file1, cursor1, num_candidates=3)

    print("\nSuggested completions:")
    for i, completion in enumerate(completions1, 1):
        print(f"\n{i}. {completion}")

    # Test case 2: Complete prime checker
    print("\n\nTest Case 2: Complete prime checker")
    print("-" * 60)

    file2 = """
def is_prime(n):
    \"\"\"Check if n is prime.\"\"\"
"""

    cursor2 = (2, 4)

    print(f"Input code:")
    print(file2)

    completions2 = copilot.complete(file2, cursor2, num_candidates=3)

    print("\nSuggested completions:")
    for i, completion in enumerate(completions2, 1):
        print(f"\n{i}. {completion}")


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print("CODE GENERATION & COMPLETION EXAMPLES")
    print("Building Mini-Copilot from Scratch!")
    print("="*70)

    # Run all demos
    demo_template_generator()
    demo_docstring_parser()
    demo_context_gatherer()
    demo_syntax_validator()
    demo_mini_copilot()

    print("\n" + "="*70)
    print("ALL DEMOS COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Template matching: Fast but limited")
    print("  2. Context is crucial for good completions")
    print("  3. Syntax validation prevents bad suggestions")
    print("  4. Ranking ensures best suggestions first")
    print("  5. Complete system = Context + Generation + Ranking + Validation")
    print("\nYou now understand how GitHub Copilot works! 🎉")


if __name__ == "__main__":
    main()

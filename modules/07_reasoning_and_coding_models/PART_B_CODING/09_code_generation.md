# Lesson 9: Code Generation & Completion (Building Mini-Copilot!)

**Part B: Coding Models - Lesson 9 of 10**
**Difficulty:** Advanced
**Time Required:** 4-5 hours
**Prerequisites:** Lessons 6 (Tokenization), 7 (Embeddings), 8 (Training)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Natural Language to Code](#natural-language-to-code)
3. [Docstring to Implementation](#docstring-to-implementation)
4. [Code Completion Strategies](#code-completion-strategies)
5. [Building Mini-Copilot](#building-mini-copilot)
6. [Advanced Techniques](#advanced-techniques)
7. [Evaluation & Quality](#evaluation-quality)
8. [Quiz & Exercises](#quiz-exercises)
9. [Summary](#summary)

---

## Introduction

### What is Code Generation?

**Code generation** is the process of automatically creating source code from various inputs:
- **Natural language** descriptions ("Create a function that sorts a list")
- **Docstrings** or comments (function signature → implementation)
- **Partial code** (complete the rest)
- **Examples** (learn pattern, generate similar code)

**Think of it like:**
- **C# Analogy:** Visual Studio's IntelliSense on steroids - not just autocomplete, but writing entire functions!
- **Real-world:** GitHub Copilot, Tabnine, Amazon CodeWhisperer

---

### Why is Code Generation Challenging?

Code generation is **much harder** than text generation because:

1. **Syntax Must Be Perfect**
   - Text: Minor grammar errors are tolerable
   - Code: One missing semicolon breaks everything!

2. **Semantic Correctness Matters**
   - Text: "The cat sat on the dog" is grammatically correct
   - Code: Syntactically correct code can still be logically wrong

3. **Context is Critical**
   - Need to understand surrounding code
   - Variable names, function signatures, imports
   - Project structure and conventions

4. **Multiple Valid Solutions**
   - Many ways to implement the same functionality
   - Need to choose the "best" one

**ASCII Diagram: Code Generation Challenges**
```
Natural Language: "Sort a list"
         ↓
    [Code Generator]
         ↓
   Multiple Options:
   ┌─────────────────────────────────────┐
   │ Option 1: list.sort()               │ ← Built-in
   │ Option 2: sorted(list)              │ ← Functional
   │ Option 3: Bubble sort from scratch  │ ← Manual
   │ Option 4: Quick sort algorithm      │ ← Efficient
   └─────────────────────────────────────┘
         ↓
   Which is best? → Depends on context!
```

---

### Applications of Code Generation

**Real-World Uses:**

1. **IDE Autocomplete** - Complete code as you type (Copilot)
2. **Boilerplate Generation** - Create classes, CRUD operations
3. **Test Generation** - Auto-generate unit tests
4. **Code Translation** - Convert between languages (C# → Python)
5. **Documentation to Code** - Implement from specs
6. **Bug Fixing** - Suggest fixes for errors

**C# Developer Perspective:**
- Like Resharper code generation
- Similar to T4 templates, but AI-powered
- Roslyn analyzers + automatic fixes, but more intelligent

---

## Natural Language to Code

### Understanding the Problem

**Goal:** Convert natural language descriptions → executable code

**Example:**
```
Input:  "Create a function that calculates factorial"
Output: def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)
```

---

### Approach 1: Template Matching

**Concept:** Recognize common patterns and fill templates

**How it works:**
1. **Parse** natural language for keywords
2. **Match** to known templates
3. **Extract** parameters
4. **Fill** template with parameters

**Example:**

```python
# Step 1: Parse input
input_text = "Create a function that adds two numbers"

# Step 2: Identify pattern
# Keywords: "function", "adds", "two numbers"
# Pattern: "function that [action] [operands]"

# Step 3: Match template
template = """
def {function_name}({param1}, {param2}):
    return {param1} {operator} {param2}
"""

# Step 4: Extract parameters
function_name = "add"          # From "adds"
param1, param2 = "a", "b"      # From "two numbers"
operator = "+"                  # From "adds"

# Step 5: Generate code
result = template.format(
    function_name=function_name,
    param1=param1,
    param2=param2,
    operator=operator
)
```

**Result:**
```python
def add(a, b):
    return a + b
```

**Pros:**
- ✅ Simple and fast
- ✅ Predictable output
- ✅ Works for common patterns

**Cons:**
- ❌ Limited to predefined templates
- ❌ Can't handle novel requests
- ❌ Brittle to variations in phrasing

**C# Comparison:**
```csharp
// Similar to code snippets in Visual Studio
// Type "prop" + Tab → generates property

// Template:
public {type} {name} { get; set; }

// Result:
public string Name { get; set; }
```

---

### Approach 2: Seq2Seq (Sequence-to-Sequence)

**Concept:** Treat as translation problem: English → Python

**Architecture:**
```
Natural Language → [Encoder] → Context Vector → [Decoder] → Code
```

**How it works:**

1. **Encoder** reads natural language word-by-word
2. Creates a **context vector** (semantic representation)
3. **Decoder** generates code token-by-token

**Example:**

```python
import torch
import torch.nn as nn

class Seq2SeqCodeGenerator(nn.Module):
    """
    Sequence-to-sequence model for code generation
    Similar to machine translation models
    """

    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        # Encoder: Processes natural language
        # Like reading a book and understanding the plot
        self.encoder = nn.LSTM(
            input_size=vocab_size,    # Size of vocabulary
            hidden_size=hidden_size,   # Size of memory
            num_layers=2,              # Depth of network
            batch_first=True
        )

        # Decoder: Generates code
        # Like writing a summary of the book
        self.decoder = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        # Output layer: Predicts next token
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, natural_language, code_so_far):
        """
        Generate code from natural language

        Args:
            natural_language: Tokenized NL input [batch, seq_len]
            code_so_far: Partially generated code [batch, seq_len]

        Returns:
            next_token_probs: Probabilities for next token
        """
        # Step 1: Encode natural language
        # encoder_output: what was said at each step
        # hidden: overall understanding (context vector)
        encoder_output, (hidden, cell) = self.encoder(natural_language)

        # Step 2: Decode to code
        # Start decoder with encoder's understanding
        decoder_output, _ = self.decoder(code_so_far, (hidden, cell))

        # Step 3: Predict next token
        # For each position, predict what comes next
        next_token_probs = self.fc(decoder_output)

        return next_token_probs
```

**Line-by-Line Explanation:**

- `nn.LSTM`: **Long Short-Term Memory** network - remembers context over long sequences
  - **C# Analogy:** Like a `List<T>` that forgets old items when full

- `input_size=vocab_size`: Size of input vectors (one-hot encoded tokens)

- `hidden_size`: How much information to remember
  - Larger = more memory, but slower

- `num_layers=2`: Stack LSTMs for deeper understanding
  - **C# Analogy:** Like nested function calls, each adding more processing

- `batch_first=True`: Input shape is [batch, sequence, features]
  - **C# Analogy:** Like `List<List<int>>` instead of `int[][]`

- `encoder_output, (hidden, cell)`:
  - `encoder_output`: Representation at each time step
  - `hidden`: Final hidden state (summary)
  - `cell`: Memory state (what to remember)

**Pros:**
- ✅ Can handle novel inputs
- ✅ Learns from data, not templates
- ✅ Generalizes to unseen examples

**Cons:**
- ❌ Requires large training data
- ❌ Can generate syntactically incorrect code
- ❌ Slower than template matching

---

### Approach 3: Transformer-Based (Modern Approach)

**Concept:** Use pre-trained models (like GPT, Codex)

**Architecture:**
```
Natural Language Prompt → [Transformer LLM] → Generated Code
```

**How it works:**

1. **Pre-training:** Model learns from billions of lines of code
2. **Fine-tuning:** Specialize on code generation tasks
3. **Prompting:** Give examples or instructions
4. **Generation:** Sample tokens autoregressively

**Example:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class TransformerCodeGenerator:
    """
    Uses pre-trained transformer (like GPT) for code generation
    This is how GitHub Copilot works!
    """

    def __init__(self, model_name="Salesforce/codegen-350M-mono"):
        """
        Initialize with pre-trained code model

        Args:
            model_name: HuggingFace model identifier
                - codegen: Salesforce's code generation model
                - 350M: 350 million parameters
                - mono: Single language (Python)
        """
        # Load tokenizer (converts text ↔ numbers)
        # Like a dictionary: word → ID
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load pre-trained model
        # Already knows how to code from training!
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Move to GPU if available (much faster)
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate_code(self, prompt, max_length=100, temperature=0.7):
        """
        Generate code from natural language prompt

        Args:
            prompt: Natural language description
            max_length: Maximum tokens to generate
            temperature: Randomness (0=deterministic, 1=creative)
                - Low (0.1-0.3): Safe, predictable
                - Medium (0.5-0.7): Balanced
                - High (0.8-1.0): Creative, risky

        Returns:
            Generated code as string
        """
        # Step 1: Convert text to token IDs
        # "def add(a, b):" → [123, 456, 789, ...]
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt"  # Return PyTorch tensor
        ).to(self.device)

        # Step 2: Generate tokens autoregressively
        # Start with prompt, predict next token, add it, repeat
        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,       # Stop after this many tokens
            temperature=temperature,      # Control randomness
            do_sample=True,              # Sample (don't just pick max)
            top_p=0.95,                  # Nucleus sampling (top 95% cumulative prob)
            pad_token_id=self.tokenizer.eos_token_id  # Padding token
        )

        # Step 3: Convert token IDs back to text
        # [123, 456, 789, ...] → "def add(a, b):\n    return a + b"
        generated_code = self.tokenizer.decode(
            output_ids[0],               # First (and only) batch item
            skip_special_tokens=True     # Remove <PAD>, <EOS>, etc.
        )

        return generated_code
```

**Line-by-Line Explanation:**

- `AutoModelForCausalLM`: **Causal Language Model** - predicts next token
  - **C# Analogy:** Like autocomplete in IDE, but much smarter

- `from_pretrained(model_name)`: Load pre-trained weights
  - Model already "knows" how to code from training
  - **C# Analogy:** Like using NuGet package instead of building from scratch

- `self.device`: Run on GPU if available
  - GPU: 10-100x faster than CPU for neural networks
  - **C# Analogy:** Like using parallel LINQ (`AsParallel()`)

- `tokenizer.encode()`: Text → Token IDs
  - "def" → 123, "add" → 456, etc.

- `return_tensors="pt"`: Return PyTorch tensors (not NumPy)

- `model.generate()`: Auto-regressive generation
  - Predict next token, add to sequence, repeat
  - Like writing code one character at a time

- `temperature`: Controls randomness
  - 0.0: Always pick most likely token (boring)
  - 1.0: Sample according to probabilities (creative)
  - **C# Analogy:** Like `Random.Next()` with weighted probabilities

- `top_p=0.95`: **Nucleus sampling**
  - Only consider tokens in top 95% cumulative probability
  - Filters out unlikely/nonsensical tokens

- `skip_special_tokens=True`: Remove `<|endoftext|>`, `<PAD>`, etc.

**Pros:**
- ✅ State-of-the-art quality
- ✅ Handles complex requests
- ✅ Learns from massive code corpus
- ✅ Few-shot learning (give examples)

**Cons:**
- ❌ Requires powerful hardware (GPU)
- ❌ Large model size (350M-175B parameters)
- ❌ Can generate plausible but wrong code
- ❌ May require API access (GPT-4, Codex)

---

## Docstring to Implementation

### The Problem

**Goal:** Generate function body from signature + docstring

**Example:**
```python
# Input:
def calculate_fibonacci(n):
    """
    Calculate the nth Fibonacci number.

    Args:
        n (int): Position in Fibonacci sequence

    Returns:
        int: The nth Fibonacci number
    """
    # TODO: Implement this function
    pass

# Desired Output:
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)
```

---

### Parsing Docstrings

**Step 1:** Extract key information from docstring

```python
import re
import ast

class DocstringParser:
    """
    Parses Python docstrings to extract function metadata
    Helps understand what function should do
    """

    def parse(self, docstring):
        """
        Extract structured information from docstring

        Args:
            docstring: Raw docstring text

        Returns:
            dict: Parsed metadata (description, args, returns)
        """
        metadata = {
            "description": "",
            "args": [],
            "returns": None
        }

        if not docstring:
            return metadata

        lines = docstring.strip().split('\n')

        # First line is usually short description
        metadata["description"] = lines[0].strip()

        # Parse Args section
        # Example: "    n (int): Position in sequence"
        in_args_section = False
        in_returns_section = False

        for line in lines[1:]:
            line = line.strip()

            # Detect "Args:" section
            if line.startswith("Args:"):
                in_args_section = True
                in_returns_section = False
                continue

            # Detect "Returns:" section
            if line.startswith("Returns:"):
                in_returns_section = True
                in_args_section = False
                continue

            # Parse argument
            # Format: "param_name (type): description"
            if in_args_section and line:
                match = re.match(r'(\w+)\s*\((\w+)\):\s*(.+)', line)
                if match:
                    metadata["args"].append({
                        "name": match.group(1),   # Parameter name
                        "type": match.group(2),   # Type hint
                        "description": match.group(3)  # What it does
                    })

            # Parse return type
            if in_returns_section and line:
                match = re.match(r'(\w+):\s*(.+)', line)
                if match:
                    metadata["returns"] = {
                        "type": match.group(1),
                        "description": match.group(2)
                    }

        return metadata
```

**Line-by-Line Explanation:**

- `re.match(pattern, string)`: Regular expression matching
  - **C# Analogy:** Like `Regex.Match()` in System.Text.RegularExpressions

- `r'(\w+)\s*\((\w+)\):\s*(.+)'`: Regex pattern
  - `(\w+)`: Capture word characters (parameter name)
  - `\s*`: Optional whitespace
  - `\((\w+)\)`: Type in parentheses
  - `:\s*`: Colon and whitespace
  - `(.+)`: Description (rest of line)

- Groups: `match.group(1)`, `match.group(2)`, etc.
  - Access captured parts of regex
  - **C# Analogy:** Like `Match.Groups[1].Value`

---

### Generating Implementation

**Step 2:** Use metadata to generate code

```python
class DocstringToCode:
    """
    Generates function implementation from docstring
    Uses transformer model + docstring parsing
    """

    def __init__(self, model):
        """
        Args:
            model: Pre-trained code generation model
        """
        self.model = model
        self.parser = DocstringParser()

    def generate_implementation(self, function_signature, docstring):
        """
        Generate function body from signature + docstring

        Args:
            function_signature: "def func_name(args):"
            docstring: Triple-quoted docstring

        Returns:
            Complete function code
        """
        # Parse docstring to understand intent
        metadata = self.parser.parse(docstring)

        # Create prompt for model
        # Include signature + docstring as context
        prompt = f'''{function_signature}
    """
    {docstring}
    """
    '''

        # Generate implementation
        # Model sees signature + docstring, completes the function
        generated = self.model.generate_code(
            prompt,
            max_length=200,
            temperature=0.3  # Low temperature = more conservative
        )

        return generated
```

**Example Usage:**

```python
# Create generator
generator = DocstringToCode(model)

# Function signature
signature = "def is_prime(n):"

# Docstring
docstring = """
Check if a number is prime.

Args:
    n (int): Number to check

Returns:
    bool: True if prime, False otherwise
"""

# Generate implementation
result = generator.generate_implementation(signature, docstring)
print(result)
```

**Output:**
```python
def is_prime(n):
    """
    Check if a number is prime.

    Args:
        n (int): Number to check

    Returns:
        bool: True if prime, False otherwise
    """
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
```

**C# Comparison:**
```csharp
// In C#, XML documentation comments
/// <summary>
/// Check if a number is prime.
/// </summary>
/// <param name="n">Number to check</param>
/// <returns>True if prime, False otherwise</returns>
public bool IsPrime(int n)
{
    // Implementation generated from XML doc
    if (n < 2) return false;
    for (int i = 2; i <= Math.Sqrt(n); i++)
    {
        if (n % i == 0) return false;
    }
    return true;
}
```

---

## Code Completion Strategies

### Types of Completion

**1. Single-Line Completion**
```python
# User types:
result = sorted(my_list, key=lambda x: x.

# Completion suggests:
result = sorted(my_list, key=lambda x: x.name)
```

**2. Multi-Line Completion**
```python
# User types:
def factorial(n):

# Completion suggests:
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

**3. Fill-in-the-Middle (FIM)**
```python
# User has:
def process_data(data):
    # Validate input
    [CURSOR]
    # Process data
    return result

# Completion fills the gap:
def process_data(data):
    # Validate input
    if not data:
        raise ValueError("Data cannot be empty")
    # Process data
    return result
```

---

### Context Gathering

**Key Principle:** More context = better completions

**What to gather:**

1. **Current file** - surrounding code
2. **Imports** - available libraries
3. **Function signatures** - what's defined
4. **Variable names** - what exists in scope
5. **Related files** - similar code

```python
class ContextGatherer:
    """
    Gathers context for code completion
    Like looking around before suggesting next word
    """

    def gather_context(self, file_content, cursor_position):
        """
        Collect relevant context for completion

        Args:
            file_content: Full file as string
            cursor_position: Where cursor is (line, column)

        Returns:
            dict: Context information
        """
        lines = file_content.split('\n')
        cursor_line, cursor_col = cursor_position

        context = {
            "before": [],   # Lines before cursor
            "after": [],    # Lines after cursor
            "imports": [],  # Import statements
            "functions": [], # Defined functions
            "classes": [],  # Defined classes
            "variables": [] # Variables in scope
        }

        # Get code before cursor (up to 20 lines)
        start_line = max(0, cursor_line - 20)
        context["before"] = lines[start_line:cursor_line]

        # Get code after cursor (up to 10 lines)
        context["after"] = lines[cursor_line + 1:cursor_line + 11]

        # Extract imports
        for line in lines:
            if line.strip().startswith(('import ', 'from ')):
                context["imports"].append(line.strip())

        # Parse AST to find function/class definitions
        # AST = Abstract Syntax Tree (code structure)
        try:
            tree = ast.parse(file_content)

            for node in ast.walk(tree):
                # Find function definitions
                if isinstance(node, ast.FunctionDef):
                    context["functions"].append(node.name)

                # Find class definitions
                elif isinstance(node, ast.ClassDef):
                    context["classes"].append(node.name)
        except:
            pass  # Ignore parse errors (incomplete code)

        return context
```

**Line-by-Line Explanation:**

- `ast.parse(code)`: Parse Python code into AST
  - **C# Analogy:** Like Roslyn syntax trees in C#
  - **What it does:** Converts code string → structured tree

- `ast.walk(tree)`: Visit all nodes in AST
  - **C# Analogy:** Like traversing a TreeView control

- `isinstance(node, ast.FunctionDef)`: Check node type
  - **C# Analogy:** Like `node is FunctionDeclarationSyntax` in Roslyn

- `node.name`: Get function/class name
  - AST nodes have properties for each code element

**ASCII Diagram: Context Window**
```
File Content:
┌─────────────────────────────┐
│ import numpy as np          │ ← Imports (always include)
│ import pandas as pd         │
├─────────────────────────────┤
│ def helper_function():      │
│     ...                     │
│                             │
│ class DataProcessor:        │ ← Before context (20 lines)
│     def __init__(self):     │
│         self.data = []      │
│                             │
│     def process(self):      │
│         result = []         │
│         for item in data:   │
│             [CURSOR HERE]   │ ← Cursor position
│                             │
│         return result       │ ← After context (10 lines)
│                             │
│     def save(self):         │
│         ...                 │
└─────────────────────────────┘
```

---

### Ranking Completion Candidates

**Problem:** Model generates multiple candidates - which to show?

**Ranking Factors:**

1. **Confidence Score** - Model's probability
2. **Syntax Validity** - Can it be parsed?
3. **Type Consistency** - Does it match expected types?
4. **Style Consistency** - Matches existing code style?
5. **Contextual Relevance** - Uses available variables/imports?

```python
class CompletionRanker:
    """
    Ranks code completion candidates
    Shows best suggestions first
    """

    def rank_candidates(self, candidates, context):
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
            score = 0

            # Factor 1: Model confidence (0-1)
            score += prob * 40  # 40% weight

            # Factor 2: Syntax validity (0 or 20 points)
            if self.is_syntactically_valid(code):
                score += 20

            # Factor 3: Uses existing variables (0-20 points)
            score += self.variable_usage_score(code, context) * 20

            # Factor 4: Style consistency (0-20 points)
            score += self.style_score(code, context) * 20

            scored.append((code, score))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        return [code for code, score in scored]

    def is_syntactically_valid(self, code):
        """Check if code can be parsed without errors"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def variable_usage_score(self, code, context):
        """
        Score based on using existing variables
        Higher if code uses variables already in scope
        """
        # Extract variable names from code before cursor
        existing_vars = set()
        for line in context["before"]:
            # Simple pattern: "var_name = ..."
            match = re.match(r'\s*(\w+)\s*=', line)
            if match:
                existing_vars.add(match.group(1))

        # Check if generated code uses these variables
        used_vars = set(re.findall(r'\b(\w+)\b', code))
        overlap = used_vars & existing_vars  # Intersection

        if not existing_vars:
            return 1.0

        return len(overlap) / len(existing_vars)

    def style_score(self, code, context):
        """
        Score based on style consistency
        E.g., indentation, naming conventions
        """
        score = 0.0

        # Check indentation (tabs vs spaces)
        if context["before"]:
            # Most common indentation in context
            context_indent = self._detect_indentation(context["before"])
            code_indent = self._detect_indentation([code])

            if context_indent == code_indent:
                score += 0.5

        # Check naming convention (snake_case vs camelCase)
        context_style = self._detect_naming_style(context["before"])
        code_style = self._detect_naming_style([code])

        if context_style == code_style:
            score += 0.5

        return score

    def _detect_indentation(self, lines):
        """Detect if code uses tabs or spaces"""
        for line in lines:
            if line.startswith('\t'):
                return 'tabs'
            elif line.startswith('    '):
                return 'spaces'
        return 'unknown'

    def _detect_naming_style(self, lines):
        """Detect naming convention"""
        snake_case_count = len(re.findall(r'\b[a-z]+_[a-z]+\b', '\n'.join(lines)))
        camel_case_count = len(re.findall(r'\b[a-z]+[A-Z][a-z]+\b', '\n'.join(lines)))

        if snake_case_count > camel_case_count:
            return 'snake_case'
        elif camel_case_count > snake_case_count:
            return 'camelCase'
        else:
            return 'unknown'
```

**Scoring Example:**
```
Candidate 1: "result.append(item)"
├─ Model confidence: 0.85 → 34 points
├─ Syntax valid: Yes → 20 points
├─ Uses existing vars: 2/2 → 20 points
└─ Style consistent: Yes → 20 points
Total: 94 points ✅ Best!

Candidate 2: "result.add(item)"
├─ Model confidence: 0.60 → 24 points
├─ Syntax valid: Yes → 20 points
├─ Uses existing vars: 2/2 → 20 points
└─ Style consistent: Yes → 20 points
Total: 84 points

Candidate 3: "result << item"
├─ Model confidence: 0.40 → 16 points
├─ Syntax valid: No → 0 points
├─ Uses existing vars: 1/2 → 10 points
└─ Style consistent: No → 0 points
Total: 26 points ❌ Worst
```

---

## Building Mini-Copilot

### Architecture Overview

**Complete System Components:**

```
┌─────────────────────────────────────────────────────┐
│                  MINI-COPILOT                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐      ┌──────────────┐            │
│  │   Context    │      │  Code Model  │            │
│  │   Gatherer   │─────▶│  (Codegen)   │            │
│  └──────────────┘      └──────────────┘            │
│         │                      │                    │
│         │                      ▼                    │
│         │             ┌──────────────┐             │
│         │             │  Candidate   │             │
│         │             │  Generator   │             │
│         │             └──────────────┘             │
│         │                      │                    │
│         │                      ▼                    │
│         │             ┌──────────────┐             │
│         └────────────▶│    Ranker    │             │
│                       └──────────────┘             │
│                               │                     │
│                               ▼                     │
│                      ┌──────────────┐              │
│                      │   Validator  │              │
│                      └──────────────┘              │
│                               │                     │
│                               ▼                     │
│                      ┌──────────────┐              │
│                      │  Top K Best  │              │
│                      │  Completions │              │
│                      └──────────────┘              │
└─────────────────────────────────────────────────────┘
```

---

### Implementation

```python
class MiniCopilot:
    """
    Mini version of GitHub Copilot
    Complete code completion system

    Features:
    - Context-aware completion
    - Multiple candidate generation
    - Intelligent ranking
    - Syntax validation
    """

    def __init__(self, model_name="Salesforce/codegen-350M-mono"):
        """
        Initialize Mini-Copilot

        Args:
            model_name: HuggingFace model to use
        """
        # Core components
        self.generator = TransformerCodeGenerator(model_name)
        self.context_gatherer = ContextGatherer()
        self.ranker = CompletionRanker()

        print(f"Mini-Copilot initialized with {model_name}")

    def complete(self, file_content, cursor_position, num_candidates=5):
        """
        Generate code completions for cursor position

        Args:
            file_content: Full file content as string
            cursor_position: (line, column) tuple
            num_candidates: How many completions to generate

        Returns:
            List of suggested completions (best first)
        """
        print(f"\n{'='*60}")
        print(f"MINI-COPILOT: Generating completions...")
        print(f"{'='*60}")

        # Step 1: Gather context
        print("\n[Step 1] Gathering context...")
        context = self.context_gatherer.gather_context(
            file_content,
            cursor_position
        )

        print(f"  - Found {len(context['imports'])} imports")
        print(f"  - Found {len(context['functions'])} functions")
        print(f"  - Found {len(context['classes'])} classes")

        # Step 2: Build prompt from context
        print("\n[Step 2] Building prompt...")
        prompt = self._build_prompt(file_content, cursor_position, context)
        print(f"  - Prompt length: {len(prompt)} characters")

        # Step 3: Generate multiple candidates
        print(f"\n[Step 3] Generating {num_candidates} candidates...")
        candidates = []

        for i in range(num_candidates):
            # Generate with different temperatures for diversity
            temp = 0.3 + (i * 0.15)  # 0.3, 0.45, 0.6, 0.75, 0.9

            completion = self.generator.generate_code(
                prompt,
                max_length=len(prompt) + 50,  # Generate 50 more tokens
                temperature=temp
            )

            # Extract only the new part (after prompt)
            new_code = completion[len(prompt):].strip()

            # Calculate probability (simplified)
            # In real system, model returns log probabilities
            probability = 1.0 - (temp / 2.0)  # Higher temp = lower base prob

            candidates.append((new_code, probability))
            print(f"  Candidate {i+1}: temp={temp:.2f}, prob={probability:.2f}")

        # Step 4: Rank candidates
        print("\n[Step 4] Ranking candidates...")
        ranked = self.ranker.rank_candidates(candidates, context)

        # Step 5: Validate and filter
        print("\n[Step 5] Validating syntax...")
        valid_completions = []

        for completion in ranked[:num_candidates]:
            # Try parsing completion with surrounding context
            test_code = '\n'.join(context["before"]) + '\n' + completion

            try:
                ast.parse(test_code)
                valid_completions.append(completion)
                print(f"  ✓ Valid: {completion[:50]}...")
            except SyntaxError:
                print(f"  ✗ Invalid: {completion[:50]}...")

        # If no valid completions, return best candidate anyway
        if not valid_completions:
            valid_completions = [ranked[0]]

        print(f"\n[Result] Returning {len(valid_completions)} completions")
        print(f"{'='*60}\n")

        return valid_completions

    def _build_prompt(self, file_content, cursor_position, context):
        """
        Build prompt for model from context
        Includes code before cursor
        """
        cursor_line, cursor_col = cursor_position
        lines = file_content.split('\n')

        # Get code up to cursor
        before_cursor = '\n'.join(lines[:cursor_line])
        current_line = lines[cursor_line][:cursor_col]

        prompt = before_cursor
        if current_line:
            prompt += '\n' + current_line

        return prompt
```

**Line-by-Line Explanation:**

- `num_candidates=5`: Generate multiple options for diversity
  - Why: One model run might miss the best solution

- `temp = 0.3 + (i * 0.15)`: Varying temperatures
  - First candidate: conservative (temp=0.3)
  - Last candidate: creative (temp=0.9)
  - Gets diverse suggestions

- `completion[len(prompt):]`: Extract new code
  - Model returns prompt + completion
  - We only want the new part

- `test_code = context + completion`: Test validity
  - Need surrounding context to parse correctly
  - Isolated snippet might not parse

**Example Usage:**

```python
# Create Mini-Copilot
copilot = MiniCopilot()

# Sample file
file_content = '''
import math

def calculate_area(radius):
    """Calculate circle area."""

'''

# Cursor after the docstring
cursor_position = (4, 4)  # Line 4, column 4

# Get completions
completions = copilot.complete(file_content, cursor_position)

# Show results
for i, completion in enumerate(completions, 1):
    print(f"\nCompletion {i}:")
    print(completion)
```

**Expected Output:**
```
Completion 1:
    return math.pi * radius ** 2

Completion 2:
    area = math.pi * (radius * radius)
    return area

Completion 3:
    return math.pi * math.pow(radius, 2)
```

---

## Advanced Techniques

### 1. Beam Search for Code

**Concept:** Explore multiple promising paths simultaneously

**Normal Generation (Greedy):**
```
Start: "def add("
  → Pick best: "a"
    → Pick best: ", "
      → Pick best: "b"
        → Result: "def add(a, b):"
```

**Beam Search (Width=3):**
```
Start: "def add("
  → Keep top 3:
     1. "a" (prob: 0.8)
     2. "x" (prob: 0.1)
     3. "num" (prob: 0.1)

  → Expand each:
     From "a": ", b" (0.8*0.9=0.72)
     From "x": ", y" (0.1*0.8=0.08)
     From "num": "1, num2" (0.1*0.7=0.07)

  → Keep top 3 overall:
     1. "a, b" (0.72) ← Best path!
     2. "x, y" (0.08)
     3. "num1, num2" (0.07)
```

**Implementation:**

```python
def beam_search_generate(model, prompt, beam_width=3, max_length=50):
    """
    Generate code using beam search
    Keeps multiple hypotheses, picks best overall

    Args:
        model: Language model
        prompt: Starting text
        beam_width: How many paths to keep
        max_length: Maximum tokens

    Returns:
        Best completion
    """
    import torch

    # Start with prompt
    # beams = [(tokens, cumulative_log_prob)]
    beams = [(prompt, 0.0)]

    for step in range(max_length):
        new_beams = []

        # Expand each beam
        for tokens, score in beams:
            # Get next token probabilities
            # (In real code, call model here)
            next_token_probs = model.get_next_token_probs(tokens)

            # Keep top-k tokens for this beam
            top_k = next_token_probs.topk(beam_width)

            for token_id, prob in zip(top_k.indices, top_k.values):
                new_tokens = tokens + [token_id]
                new_score = score + torch.log(prob)  # Log probability
                new_beams.append((new_tokens, new_score))

        # Keep only top beam_width beams overall
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

        # Stop if all beams end with EOS token
        if all(b[0][-1] == EOS_TOKEN for b in beams):
            break

    # Return best beam
    best_tokens, best_score = beams[0]
    return best_tokens
```

**C# Analogy:**
```csharp
// Like keeping top K items in priority queue
var beams = new PriorityQueue<(List<int> tokens, double score)>();

foreach (var (tokens, score) in currentBeams)
{
    var nextProbs = model.GetNextTokenProbs(tokens);
    var topK = nextProbs.OrderByDescending(p => p.Value).Take(beamWidth);

    foreach (var (tokenId, prob) in topK)
    {
        var newTokens = tokens.Append(tokenId).ToList();
        var newScore = score + Math.Log(prob);
        beams.Enqueue((newTokens, newScore));
    }
}

// Keep only top beamWidth
currentBeams = beams.DequeueRange(beamWidth);
```

---

### 2. Nucleus (Top-P) Sampling

**Concept:** Sample from smallest set of tokens with cumulative probability ≥ p

**Example:**
```
Token probabilities:
  "return": 0.50
  "result": 0.25
  "output": 0.10
  "value":  0.08
  "data":   0.05
  "x":      0.02

With top-p = 0.9:
  Cumulative: 0.50 → 0.75 → 0.85 → 0.93 ✓
  Sample from: {"return", "result", "output", "value"}
  Exclude: {"data", "x"} (too unlikely)
```

**Why it helps:**
- Filters out nonsense tokens
- Keeps reasonable options
- Adaptive (set size changes based on confidence)

**Implementation:**

```python
def nucleus_sampling(probs, p=0.95):
    """
    Sample from top-p nucleus

    Args:
        probs: Token probabilities (tensor)
        p: Cumulative probability threshold

    Returns:
        Sampled token index
    """
    # Sort probabilities descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Calculate cumulative probabilities
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff: first index where cumulative >= p
    cutoff_index = torch.where(cumulative >= p)[0][0]

    # Keep only top-p tokens
    top_p_probs = sorted_probs[:cutoff_index + 1]
    top_p_indices = sorted_indices[:cutoff_index + 1]

    # Renormalize to sum to 1
    top_p_probs = top_p_probs / top_p_probs.sum()

    # Sample from this distribution
    sampled_index = torch.multinomial(top_p_probs, num_samples=1)

    # Map back to original index
    token_id = top_p_indices[sampled_index]

    return token_id
```

---

### 3. Constrained Decoding

**Concept:** Force generation to follow rules

**Example Constraints:**
1. **Syntax:** Must be valid Python
2. **Type:** Must return int, not string
3. **Style:** Must use snake_case
4. **API:** Must use allowed libraries only

**Implementation:**

```python
class ConstrainedDecoder:
    """
    Generates code that satisfies constraints
    Rejects invalid tokens during generation
    """

    def __init__(self, model):
        self.model = model

    def generate_with_constraints(self, prompt, constraints):
        """
        Generate code satisfying constraints

        Args:
            prompt: Starting code
            constraints: List of constraint functions

        Returns:
            Valid generated code
        """
        current = prompt
        max_attempts = 100

        for _ in range(max_attempts):
            # Get next token probabilities
            probs = self.model.get_next_token_probs(current)

            # Filter out tokens that violate constraints
            valid_tokens = []
            for token_id in range(len(probs)):
                # Try adding this token
                test_code = current + self.model.decode([token_id])

                # Check all constraints
                if all(constraint(test_code) for constraint in constraints):
                    valid_tokens.append(token_id)

            if not valid_tokens:
                break  # Can't continue without violating constraints

            # Sample from valid tokens only
            valid_probs = probs[valid_tokens]
            valid_probs = valid_probs / valid_probs.sum()  # Renormalize

            chosen_idx = torch.multinomial(valid_probs, num_samples=1)
            chosen_token = valid_tokens[chosen_idx]

            current += self.model.decode([chosen_token])

            # Check if done
            if chosen_token == EOS_TOKEN:
                break

        return current

# Example constraints
def must_be_valid_python(code):
    """Constraint: Code must be syntactically valid"""
    try:
        ast.parse(code)
        return True
    except:
        return False

def must_use_snake_case(code):
    """Constraint: Variables must use snake_case"""
    # Check if any camelCase variables
    if re.search(r'\b[a-z]+[A-Z][a-z]+\b', code):
        return False
    return True

# Usage
decoder = ConstrainedDecoder(model)
result = decoder.generate_with_constraints(
    prompt="def calculate_total(items):\n    ",
    constraints=[must_be_valid_python, must_use_snake_case]
)
```

---

## Evaluation & Quality

### Syntax Validation

**Goal:** Ensure generated code is syntactically correct

```python
class SyntaxValidator:
    """
    Validates syntax of generated code
    Catches errors before showing to user
    """

    def validate(self, code, language="python"):
        """
        Check if code is syntactically valid

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            (is_valid, error_message)
        """
        if language == "python":
            return self._validate_python(code)
        else:
            return (False, f"Unsupported language: {language}")

    def _validate_python(self, code):
        """Validate Python syntax"""
        try:
            # Try parsing code
            ast.parse(code)
            return (True, None)

        except SyntaxError as e:
            # Syntax error details
            return (False, f"SyntaxError at line {e.lineno}: {e.msg}")

        except Exception as e:
            return (False, f"Parse error: {str(e)}")

    def auto_fix(self, code):
        """
        Attempt to automatically fix syntax errors
        Simple heuristics only
        """
        is_valid, error = self.validate(code)

        if is_valid:
            return code

        # Try common fixes
        fixes = [
            self._fix_missing_colon,
            self._fix_indentation,
            self._fix_missing_parenthesis,
        ]

        for fix_func in fixes:
            fixed_code = fix_func(code)
            is_valid, _ = self.validate(fixed_code)

            if is_valid:
                return fixed_code

        # Can't fix automatically
        return code

    def _fix_missing_colon(self, code):
        """Add missing colons after if/for/def/class"""
        lines = code.split('\n')
        fixed = []

        for line in lines:
            # Check if line should end with colon
            stripped = line.strip()
            if (stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while '))
                and not stripped.endswith(':')):
                line = line + ':'

            fixed.append(line)

        return '\n'.join(fixed)

    def _fix_indentation(self, code):
        """Fix common indentation issues"""
        # This is complex - simplified version
        # Real implementation would use AST
        return code

    def _fix_missing_parenthesis(self, code):
        """Add missing closing parentheses"""
        # Count open/close parens
        open_count = code.count('(')
        close_count = code.count(')')

        if open_count > close_count:
            code += ')' * (open_count - close_count)

        return code
```

**Example:**
```python
validator = SyntaxValidator()

# Valid code
code1 = "def add(a, b):\n    return a + b"
is_valid, error = validator.validate(code1)
print(f"Valid: {is_valid}")  # True

# Invalid code (missing colon)
code2 = "def add(a, b)\n    return a + b"
is_valid, error = validator.validate(code2)
print(f"Valid: {is_valid}")  # False
print(f"Error: {error}")     # SyntaxError at line 1: invalid syntax

# Auto-fix
fixed = validator.auto_fix(code2)
print(f"Fixed: {fixed}")     # def add(a, b):\n    return a + b
```

---

### Semantic Validation

**Goal:** Check if code does what it's supposed to do

**Techniques:**

1. **Type Checking** - Does it return correct type?
2. **Test Cases** - Does it pass example inputs?
3. **Static Analysis** - Any logical errors?

```python
def semantic_validation(code, test_cases):
    """
    Validate code semantics using test cases

    Args:
        code: Generated function code
        test_cases: List of (input, expected_output) tuples

    Returns:
        (passes_all, failed_cases)
    """
    import sys
    from io import StringIO

    # Execute code in isolated namespace
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return (False, [f"Execution error: {e}"])

    # Find the function (assume first function defined)
    func = None
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith('_'):
            func = obj
            break

    if not func:
        return (False, ["No function found"])

    # Run test cases
    failed = []
    for inputs, expected in test_cases:
        try:
            # Call function
            if isinstance(inputs, tuple):
                result = func(*inputs)
            else:
                result = func(inputs)

            # Check result
            if result != expected:
                failed.append(f"Input: {inputs}, Expected: {expected}, Got: {result}")

        except Exception as e:
            failed.append(f"Input: {inputs}, Error: {e}")

    return (len(failed) == 0, failed)

# Example
code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

test_cases = [
    (0, 1),
    (1, 1),
    (5, 120),
    (10, 3628800)
]

passes, failures = semantic_validation(code, test_cases)
print(f"Passes all tests: {passes}")
if not passes:
    for failure in failures:
        print(f"  - {failure}")
```

---

## Quiz & Exercises

### Quiz Questions

**Question 1: Code Generation Approaches**

Which approach is BEST for generating code from natural language in a production system?

A) Template matching
B) Seq2Seq with LSTM
C) Pre-trained transformer (e.g., Codex)
D) Random generation

<details>
<summary>Answer</summary>

**C) Pre-trained transformer**

**Explanation:**
- **Template matching**: Too rigid, only handles known patterns
- **Seq2Seq LSTM**: Requires lots of training data, limited context
- **Pre-trained transformer**: State-of-the-art, handles novel inputs, large context
- **Random**: Not a real approach!

In production, pre-trained models (Codex, CodeGen) are best because:
1. Already trained on billions of lines of code
2. Can handle diverse requests
3. Support few-shot learning
4. Large context windows (2048+ tokens)

</details>

---

**Question 2: Context Gathering**

When generating code completions, what context is MOST important?

A) Imports at top of file
B) Code immediately before cursor
C) Code immediately after cursor
D) All of the above

<details>
<summary>Answer</summary>

**D) All of the above**

**Explanation:**

Each provides crucial information:

- **Imports**: What libraries are available
  ```python
  import pandas as pd  # Now can suggest pd.DataFrame()
  ```

- **Code before cursor**: Current scope, variables
  ```python
  data = load_data()
  # Cursor here - can suggest operations on 'data'
  ```

- **Code after cursor**: Intent, what comes next
  ```python
  # Cursor here
  return result  # Must define 'result' before this!
  ```

**Best practice:** Gather context from all sources, weighted by proximity to cursor.

</details>

---

**Question 3: Ranking Candidates**

You have 3 completion candidates:

```python
# Candidate A: result.append(item)  (prob: 0.9, valid syntax)
# Candidate B: result.add(item)     (prob: 0.8, valid syntax)
# Candidate C: result << item       (prob: 0.7, invalid syntax)
```

Which should be ranked FIRST?

A) Candidate A (highest probability)
B) Candidate B (middle ground)
C) Candidate C (shortest code)
D) Can't determine without more context

<details>
<summary>Answer</summary>

**A) Candidate A**

**Explanation:**

Ranking factors:
1. **Syntax validity** (eliminates C)
2. **Model confidence** (A: 0.9 > B: 0.8)
3. **Correctness** (lists use `.append()`, not `.add()`)

**Scoring:**
- A: Syntax ✓, High prob ✓, Correct method ✓ → **Best**
- B: Syntax ✓, Med prob ✓, Wrong method ✗ → Second
- C: Syntax ✗ → **Eliminated**

**Note:** `result.add()` doesn't exist for Python lists (that's a Set method).

</details>

---

**Question 4: Fill-in-the-Middle (FIM)**

What makes FIM training crucial for code completion?

A) Allows completing code in the middle of a function
B) Improves model's general language understanding
C) Makes training faster
D) Reduces model size

<details>
<summary>Answer</summary>

**A) Allows completing code in the middle of a function**

**Explanation:**

**Standard training (left-to-right):**
```python
def calculate(x):
    [CURSOR] ???  # Can't complete here!
    return result
```
Model only learns to predict next token given previous tokens.

**FIM training:**
```python
<PREFIX> def calculate(x):
<SUFFIX> return result
<MIDDLE> [Generate this part]
```

Model learns to generate middle section given prefix AND suffix!

**Why it matters:**
- Real coding: Often edit middle of functions
- Standard completion: Only works at end of file
- FIM: Works anywhere!

**C# Analogy:**
```csharp
public int Calculate(int x)
{
    // [CURSOR] - Need to complete here!
    return result;
}
```

GitHub Copilot uses FIM extensively!

</details>

---

### Practice Exercises

**Exercise 1: Build a Simple Template Matcher**

Create a template matcher that generates functions from natural language.

**Requirements:**
- Handle patterns: "function that [action] [operands]"
- Support actions: add, subtract, multiply, divide
- Generate valid Python code

**Example:**
```python
input: "function that multiplies two numbers"
output:
def multiply(a, b):
    return a * b
```

**Starter Code:**
```python
class TemplateCodeGenerator:
    def generate(self, description):
        # TODO: Parse description
        # TODO: Match template
        # TODO: Return code
        pass

# Test
gen = TemplateCodeGenerator()
print(gen.generate("function that adds two numbers"))
```

<details>
<summary>Solution</summary>

```python
class TemplateCodeGenerator:
    def __init__(self):
        # Map action words to operators
        self.actions = {
            "add": "+",
            "adds": "+",
            "subtract": "-",
            "subtracts": "-",
            "multiply": "*",
            "multiplies": "*",
            "divide": "/",
            "divides": "/"
        }

    def generate(self, description):
        # Parse description
        words = description.lower().split()

        # Find action word
        operator = None
        function_name = None

        for word in words:
            if word in self.actions:
                operator = self.actions[word]
                function_name = word if word.endswith('s') else word
                break

        if not operator:
            return "# Error: Unknown action"

        # Generate code
        code = f"""def {function_name}(a, b):
    return a {operator} b"""

        return code

# Test
gen = TemplateCodeGenerator()
print(gen.generate("function that adds two numbers"))
print()
print(gen.generate("function that multiplies two values"))
```

**Output:**
```python
def adds(a, b):
    return a + b

def multiplies(a, b):
    return a * b
```

</details>

---

**Exercise 2: Implement Context Gatherer**

Build a context gatherer that extracts information for code completion.

**Requirements:**
- Extract imports
- Find function definitions
- Get code before/after cursor

**Example:**
```python
file_content = """
import math

def helper():
    pass

def main():
    x = 10
    # CURSOR HERE
    print(x)
"""

cursor = (6, 4)  # Line 6, col 4
context = gather_context(file_content, cursor)
```

<details>
<summary>Solution</summary>

See `example_09_code_generator.py` for complete implementation!

</details>

---

## Summary

### Key Takeaways

1. **Code Generation is Challenging**
   - Syntax must be perfect
   - Semantics matter
   - Context is critical

2. **Multiple Approaches**
   - Template matching: Fast but limited
   - Seq2Seq: Flexible but data-hungry
   - Transformers: State-of-the-art

3. **Context is King**
   - Gather imports, functions, variables
   - Use code before AND after cursor
   - AST parsing helps

4. **Generate Multiple Candidates**
   - Vary temperature for diversity
   - Rank by confidence + validity
   - Show top K suggestions

5. **Validation is Essential**
   - Syntax checking (AST)
   - Semantic checking (tests)
   - Auto-fix common errors

6. **FIM is Critical**
   - Enables mid-function completion
   - Used by all modern code assistants
   - Requires special training

---

### C#/.NET Comparisons

| Python/Copilot | C#/.NET Equivalent |
|---|---|
| `ast.parse()` | Roslyn syntax trees |
| Template matching | Code snippets |
| Context gathering | IntelliSense context |
| FIM training | Mid-statement completion |
| Beam search | Priority queue |
| Validation | Compiler diagnostics |

---

### What You've Learned

After this lesson, you can:

- ✅ Generate code from natural language
- ✅ Implement docstring-to-code conversion
- ✅ Build context-aware completion
- ✅ Create a mini-Copilot system
- ✅ Rank completion candidates
- ✅ Validate generated code
- ✅ Apply advanced techniques (beam search, nucleus sampling)

---

### Next Steps

**Lesson 10:** Code Evaluation & Testing
- HumanEval benchmark
- Pass@k metrics
- Test generation
- Measuring code quality

---

### Additional Resources

**Papers:**
- Codex (OpenAI, 2021)
- CodeGen (Salesforce, 2022)
- InCoder (Facebook, 2022)
- AlphaCode (DeepMind, 2022)

**Tools:**
- GitHub Copilot
- Tabnine
- Amazon CodeWhisperer
- Replit Ghostwriter

**Libraries:**
- `transformers` (HuggingFace)
- `ast` (Python AST parsing)
- `tree-sitter` (Multi-language parsing)

---

**Congratulations!** You've learned how to build a mini-Copilot!

**Module Progress:** 90% (9/10 lessons complete)

**Next:** Lesson 10 - Code Evaluation & Testing

---

**Created:** March 17, 2026
**Author:** Learn LLM from Scratch Project
**For:** .NET Developers Learning AI

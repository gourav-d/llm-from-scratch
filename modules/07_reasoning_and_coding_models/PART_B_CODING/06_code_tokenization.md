# Lesson 7.6: Code Representation & Tokenization

## 🎯 Learning Objectives

By the end of this lesson, you'll understand:

- Why code needs special tokenization (different from natural language)
- Character-level vs token-level vs AST-based tokenization
- How to parse code into Abstract Syntax Trees (AST)
- Using TreeSitter for multi-language parsing
- Building code-specific vocabularies
- How GitHub Copilot tokenizes code
- Preparing code data for training

**This is the foundation for building Copilot-like code generation!**

---

## 🤔 What is Code Tokenization?

### The Problem: Code is NOT Natural Language

**Natural language tokenization:**
```
Input: "Hello, world! How are you?"

GPT Tokens: ["Hello", ",", " world", "!", " How", " are", " you", "?"]

Works fine! ✓
```

**Naive code tokenization (treating code like text):**
```python
Input: "def calculate_sum(a, b):\n    return a + b"

Tokens: ["def", " calculate", "_", "sum", "(", "a", ",", " b", ")"]

Problems:
- "calculate_sum" split into 3 tokens ✗
- Loses function name semantics ✗
- Indentation lost ✗
- No structural understanding ✗
```

**Why this fails:**
- Code has **syntax** (grammar rules)
- Code has **structure** (functions, classes, blocks)
- Code has **semantics** (meaning beyond words)
- Whitespace/indentation **matters** in Python
- Variable names should stay together

---

## 🌍 Real-World Analogy

### Natural Language vs Code

**Natural language is like prose:**
```
"The quick brown fox jumps over the lazy dog."

- Order matters, but flexible
- Context from surrounding words
- Tolerate small errors ("the quik fox")
- No strict grammar rules
```

**Code is like music notation:**
```python
def play_note(note, duration):
    speaker.play(note, duration)

- Exact syntax required (like musical notation)
- Structure matters (indentation = rhythm)
- Variable names = specific instruments
- One wrong character = syntax error (out of tune!)
```

**Code tokenization is like reading sheet music:**
- Need to understand measures (code blocks)
- Notes within measures (statements)
- Instruments (variable names)
- Timing/rhythm (indentation, structure)

---

## 📚 Three Approaches to Code Tokenization

### Approach 1: Character-Level Tokenization

**Idea:** Treat code as a sequence of characters

```python
code = "def add(a, b):"

Character tokens: ['d', 'e', 'f', ' ', 'a', 'd', 'd', '(', 'a', ',', ' ', 'b', ')', ':']
```

**Pros:**
- ✅ Simple to implement
- ✅ No vocabulary limits (any code works)
- ✅ Works for any programming language

**Cons:**
- ✗ Very long sequences (inefficient)
- ✗ No semantic understanding
- ✗ Hard to learn patterns (e.g., "def" as 3 separate chars)

**When to use:**
- Simple code generation tasks
- Limited training data
- Learning code syntax from scratch

---

### Approach 2: Token-Level Tokenization (BPE/WordPiece)

**Idea:** Use Byte-Pair Encoding to create code-aware tokens

```python
code = "def calculate_sum(a, b):"

BPE Tokens: ['def', ' calculate', '_sum', '(', 'a', ',', ' b', ')', ':']
```

**How BPE works for code:**
1. Start with character vocabulary
2. Find frequent character pairs: "de" + "f" = "def"
3. Merge into single token: "def"
4. Repeat for code-specific patterns: "calculate_sum" → ["calculate", "_sum"]

**Pros:**
- ✅ Balances vocabulary size and sequence length
- ✅ Learns code-specific patterns (function names, keywords)
- ✅ Handles unknown code (falls back to chars)

**Cons:**
- ✗ Still treats code as text (no structure)
- ✗ May split important identifiers poorly
- ✗ Doesn't understand code semantics

**When to use:**
- Most modern code models (Copilot, CodeGen)
- Large-scale training
- Need to handle multiple languages

**This is what GitHub Copilot uses!**

---

### Approach 3: AST-Based Tokenization

**Idea:** Parse code into Abstract Syntax Tree (structure-aware)

```python
code = "def add(a, b):\n    return a + b"

AST Structure:
FunctionDef
├── name: "add"
├── args:
│   ├── arg: "a"
│   └── arg: "b"
└── body:
    └── Return
        └── BinOp
            ├── left: "a"
            ├── op: Add
            └── right: "b"

Tokens: ['FunctionDef', 'add', 'args', 'a', 'b', 'Return', 'BinOp', 'a', 'Add', 'b']
```

**Pros:**
- ✅ Understands code structure
- ✅ Semantically meaningful
- ✅ Easier to learn code patterns
- ✅ Can verify syntax correctness

**Cons:**
- ✗ Requires language-specific parser
- ✗ Can't handle syntax errors
- ✗ More complex to implement
- ✗ Loses some surface-level info (formatting)

**When to use:**
- Code understanding tasks (search, analysis)
- Bug detection
- Code refactoring
- Semantic code completion

---

## 💻 Implementation: BPE Tokenization for Code

### Simple Code Tokenizer

```python
import re
from collections import Counter
from typing import List, Dict

class CodeTokenizer:
    """
    BPE-based tokenizer optimized for code.

    C# analogy: Like Roslyn's SyntaxToken, but for neural models.
    """

    def __init__(self, vocab_size: int = 1000):
        """
        Initialize code tokenizer.

        Args:
            vocab_size: Size of vocabulary to build
        """
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = {}  # BPE merge rules

    def train(self, code_samples: List[str]):
        """
        Train tokenizer on code samples.

        Args:
            code_samples: List of code strings
        """
        # Step 1: Pre-tokenize (split on code boundaries)
        all_tokens = []
        for code in code_samples:
            tokens = self._pre_tokenize(code)
            all_tokens.extend(tokens)

        # Step 2: Build character vocabulary
        vocab = set()
        for token in all_tokens:
            vocab.update(token)

        # Special tokens
        vocab.update(['<PAD>', '<UNK>', '<BOS>', '<EOS>'])

        # Step 3: Learn BPE merges
        print(f"Learning BPE merges from {len(all_tokens)} tokens...")
        current_vocab_size = len(vocab)

        while current_vocab_size < self.vocab_size:
            # Count pairs
            pair_counts = Counter()
            for token in all_tokens:
                for i in range(len(token) - 1):
                    pair = (token[i], token[i + 1])
                    pair_counts[pair] += 1

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]

            # Merge this pair
            new_token = ''.join(best_pair)
            self.merges[best_pair] = new_token

            # Update tokens
            all_tokens = [self._apply_merge(token, best_pair, new_token)
                          for token in all_tokens]

            vocab.add(new_token)
            current_vocab_size += 1

            if current_vocab_size % 100 == 0:
                print(f"  Vocab size: {current_vocab_size}")

        # Step 4: Build token<->id mappings
        self.token_to_id = {token: i for i, token in enumerate(sorted(vocab))}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

        print(f"✓ Trained tokenizer with {len(self.token_to_id)} tokens")

    def _pre_tokenize(self, code: str) -> List[str]:
        """
        Pre-tokenize code (split on natural boundaries).

        This preserves code structure better than naive splitting.
        """
        # Pattern that splits on:
        # - Whitespace
        # - Operators (+, -, *, /, =, etc.)
        # - Punctuation ((), {}, [], etc.)
        # But keeps them as separate tokens

        pattern = r'(\s+|[+\-*/=<>!]=?|[(){}\[\],;:.]|[\w_]+)'
        tokens = re.findall(pattern, code)

        # Convert to list of characters for BPE
        return [list(token) for token in tokens if token.strip()]

    def _apply_merge(self, token: List[str], pair: tuple, new_token: str) -> List[str]:
        """Apply a BPE merge to a token."""
        result = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
                result.append(new_token)
                i += 2
            else:
                result.append(token[i])
                i += 1
        return result

    def encode(self, code: str) -> List[int]:
        """
        Encode code string to token IDs.

        Args:
            code: Code string to encode

        Returns:
            List of token IDs
        """
        # Pre-tokenize
        tokens = self._pre_tokenize(code)

        # Apply BPE merges
        for pair, merged in self.merges.items():
            tokens = [self._apply_merge(token, pair, merged) for token in tokens]

        # Flatten and convert to IDs
        flat_tokens = []
        for token_list in tokens:
            flat_tokens.extend(token_list)

        # Map to IDs
        ids = []
        for token in flat_tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id['<UNK>'])

        return ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to code string.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded code string
        """
        tokens = [self.id_to_token.get(id_, '<UNK>') for id_ in token_ids]
        return ''.join(tokens)


# Example usage
if __name__ == "__main__":
    # Training data
    code_samples = [
        "def add(a, b):\n    return a + b",
        "def multiply(x, y):\n    return x * y",
        "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total"
    ]

    # Train tokenizer
    tokenizer = CodeTokenizer(vocab_size=500)
    tokenizer.train(code_samples)

    # Test encoding
    test_code = "def subtract(a, b):\n    return a - b"

    print(f"\nOriginal code:\n{test_code}")

    encoded = tokenizer.encode(test_code)
    print(f"\nEncoded: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"\nDecoded:\n{decoded}")
```

---

## 🔬 Abstract Syntax Trees (AST)

### What is an AST?

An AST represents code as a tree structure that captures **meaning**, not just syntax.

```python
code = "x = 5 + 3"

AST:
Assignment
├── target: "x"
└── value:
    BinOp
    ├── left: Constant(5)
    ├── op: Add
    └── right: Constant(3)
```

### Generating ASTs in Python

```python
import ast
import json

def code_to_ast(code: str) -> ast.AST:
    """Parse Python code into AST."""
    return ast.parse(code)

def ast_to_dict(node: ast.AST) -> dict:
    """Convert AST to dictionary for visualization."""
    result = {
        'type': node.__class__.__name__
    }

    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            result[field] = [ast_to_dict(item) if isinstance(item, ast.AST) else item
                            for item in value]
        elif isinstance(value, ast.AST):
            result[field] = ast_to_dict(value)
        else:
            result[field] = value

    return result

# Example
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

tree = code_to_ast(code)
ast_dict = ast_to_dict(tree)

print(json.dumps(ast_dict, indent=2))

# Output shows complete function structure:
# {
#   "type": "Module",
#   "body": [
#     {
#       "type": "FunctionDef",
#       "name": "fibonacci",
#       "args": {...},
#       "body": [...]
#     }
#   ]
# }
```

### Why ASTs Matter for Code Models

**1. Structure-Aware Learning**
```python
# These are semantically the same:
def add(a, b): return a + b
def add(x, y): return x + y

# AST captures this equivalence!
# Both have identical structure, just different variable names
```

**2. Syntax Validation**
```python
# This has syntax error:
def add(a, b
    return a + b

# AST parsing fails → Model knows this is invalid
```

**3. Code Transformations**
```python
# Can transform AST to rename variables:
Old: def foo(x): return x * 2
New: def foo(value): return value * 2

# Just change AST node, regenerate code!
```

---

## 🛠️ TreeSitter: Universal Code Parser

### What is TreeSitter?

**TreeSitter** is a parser generator that creates ASTs for multiple languages.

**Used by:**
- GitHub (code navigation)
- Neovim (syntax highlighting)
- GitHub Copilot (code understanding)

### Using TreeSitter

```python
from tree_sitter import Language, Parser

# Note: In reality, you'd need to build language files first
# This is pseudocode showing the concept

class TreeSitterCodeParser:
    """
    Parse code using TreeSitter (multi-language).

    C# analogy: Like Roslyn but works for Python, JavaScript, etc.
    """

    def __init__(self, language_name: str):
        """
        Initialize parser for specific language.

        Args:
            language_name: 'python', 'javascript', 'java', etc.
        """
        # Load language grammar
        # self.language = Language('build/languages.so', language_name)
        # self.parser = Parser()
        # self.parser.set_language(self.language)
        pass

    def parse(self, code: str):
        """
        Parse code into AST.

        Returns:
            TreeSitter Tree object
        """
        # tree = self.parser.parse(bytes(code, 'utf8'))
        # return tree
        pass

    def extract_functions(self, tree):
        """
        Extract all function definitions from AST.

        Returns:
            List of function nodes
        """
        # Query: (function_definition name: (identifier) @func-name)
        # functions = tree.root_node.query(query)
        # return functions
        pass


# Example of what TreeSitter enables
"""
Input (Python):
def add(a, b):
    return a + b

TreeSitter output:
(module
  (function_definition
    name: (identifier) @func-name
    parameters: (parameters
      (identifier) @param
      (identifier) @param)
    body: (block
      (return_statement
        (binary_operator
          left: (identifier)
          operator: "+"
          right: (identifier))))))
"""
```

### Benefits of TreeSitter

1. **Multi-Language Support**
   - Python, JavaScript, Java, C++, Go, Rust...
   - Same API for all languages

2. **Error Recovery**
   - Can parse incomplete/broken code
   - Important for code completion!

3. **Incremental Parsing**
   - Only re-parse changed parts
   - Fast for large codebases

4. **Query Language**
   - Find specific patterns in code
   - Extract functions, classes, variables

---

## 🎯 Code-Specific Tokenization Strategies

### Strategy 1: Preserve Identifiers

**Problem:** BPE might split variable names poorly

```python
# Bad tokenization:
calculate_sum → ["calc", "ulate", "_s", "um"]  # Lost meaning!

# Good tokenization:
calculate_sum → ["calculate_sum"]  # Keep together!
```

**Solution:** Pre-tokenize on identifier boundaries

```python
def smart_pre_tokenize(code: str) -> List[str]:
    """
    Pre-tokenize code preserving identifiers.
    """
    import re

    # Pattern matches:
    # - Complete identifiers (word characters + underscores)
    # - Operators
    # - Whitespace
    # - Punctuation
    pattern = r'(\w+|[+\-*/=<>!]=?|[ \t\n]|[(){}\[\],;:.])'

    tokens = re.findall(pattern, code)
    return [t for t in tokens if t.strip()]  # Remove empty

# Example
code = "calculate_sum(a, b)"
tokens = smart_pre_tokenize(code)
# Result: ["calculate_sum", "(", "a", ",", "b", ")"]
# ✓ Identifier kept together!
```

---

### Strategy 2: Whitespace Handling

**Problem:** Python needs indentation; other languages don't care

```python
# Python - indentation matters
def foo():
    return 42  # 4 spaces = inside function

# JavaScript - indentation optional
function foo() {
return 42;  # Works fine!
}
```

**Solution:** Language-specific whitespace tokens

```python
def tokenize_with_whitespace(code: str, language: str) -> List[str]:
    """
    Tokenize code with language-appropriate whitespace handling.
    """
    if language == 'python':
        # Preserve indentation
        tokens = []
        for line in code.split('\n'):
            # Count leading spaces
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                tokens.append(f'<INDENT:{indent}>')

            tokens.extend(smart_pre_tokenize(line.lstrip()))
            tokens.append('<NEWLINE>')

        return tokens
    else:
        # For C-like languages, whitespace is just separation
        return smart_pre_tokenize(code)
```

---

### Strategy 3: Special Tokens for Code

**Add code-specific special tokens:**

```python
special_tokens = {
    '<INDENT>': 'Indentation marker',
    '<DEDENT>': 'Dedentation marker',
    '<NEWLINE>': 'Line break',
    '<COMMENT>': 'Start of comment',
    '<STRING>': 'String literal',
    '<NUMBER>': 'Numeric literal',
    '<FUNC>': 'Function definition',
    '<CLASS>': 'Class definition'
}
```

**Example:**

```python
code = """
def calculate(x):  # Helper function
    return x * 2
"""

Tokens with special markers:
[
    '<FUNC>', 'calculate', '(', 'x', ')', ':',
    '<COMMENT>', 'Helper function',
    '<NEWLINE>', '<INDENT>',
    'return', 'x', '*', '<NUMBER>', '2',
    '<DEDENT>'
]
```

---

## 📊 Comparison: Different Tokenization Approaches

| Approach | Vocab Size | Seq Length | Semantic | Multi-Lang | Used By |
|----------|------------|------------|----------|------------|---------|
| Character | ~100 | Very Long | ✗ | ✓ | Simple models |
| BPE | 10k-50k | Medium | ~  | ✓ | Copilot, CodeGen |
| WordPiece | 10k-50k | Medium | ~ | ✓ | Code T5 |
| AST-based | Varies | Short | ✓ | ✗ | Code analyzers |
| Hybrid (BPE+AST) | 10k-50k | Medium | ✓ | ✓ | GraphCodeBERT |

---

## 🔧 Building Production Code Tokenizer

### Following Copilot's Approach

```python
class CopilotStyleTokenizer:
    """
    Tokenizer similar to GitHub Copilot.

    Combines BPE with code-aware preprocessing.
    """

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.bpe_tokenizer = CodeTokenizer(vocab_size)

        # Special tokens for code
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<INDENT>': 4,
            '<DEDENT>': 5,
            '<NEWLINE>': 6,
            '<MASK>': 7  # For fill-in-the-middle
        }

    def train(self, code_files: List[str]):
        """
        Train on code files.

        Args:
            code_files: Paths to code files
        """
        code_samples = []

        for file_path in code_files:
            with open(file_path, 'r') as f:
                code = f.read()
                # Preprocess
                preprocessed = self.preprocess_code(code)
                code_samples.append(preprocessed)

        # Train BPE on preprocessed code
        self.bpe_tokenizer.train(code_samples)

    def preprocess_code(self, code: str) -> str:
        """
        Preprocess code before tokenization.

        - Normalize whitespace
        - Mark indentation
        - Preserve string literals
        """
        import re

        lines = code.split('\n')
        processed_lines = []

        for line in lines:
            # Detect indentation
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                line = f'<INDENT:{indent}>' + line.lstrip()

            # Mark string literals
            line = re.sub(r'"([^"]*)"', r'<STR>"\1"<ENDSTR>', line)
            line = re.sub(r"'([^']*)'", r"<STR>'\1'<ENDSTR>", line)

            processed_lines.append(line)

        return '\n'.join(processed_lines)

    def tokenize_for_training(self, code: str) -> List[int]:
        """
        Tokenize code for model training.

        Returns:
            List of token IDs
        """
        # Preprocess
        preprocessed = self.preprocess_code(code)

        # Encode with BPE
        token_ids = self.bpe_tokenizer.encode(preprocessed)

        # Add BOS and EOS
        token_ids = [self.special_tokens['<BOS>']] + token_ids + [self.special_tokens['<EOS>']]

        return token_ids
```

---

## 🎯 Key Takeaways

### What You Learned

1. **Code ≠ Natural Language**
   - Code has syntax, structure, and semantics
   - Needs specialized tokenization
   - Identifiers and whitespace are critical

2. **Three Main Approaches**
   - Character-level: Simple but inefficient
   - BPE/WordPiece: Best balance (used by Copilot)
   - AST-based: Structure-aware but complex

3. **Abstract Syntax Trees**
   - Represent code as meaningful structure
   - Enable syntax validation
   - Support code transformations

4. **TreeSitter**
   - Universal multi-language parser
   - Used by GitHub and Copilot
   - Handles incomplete/broken code

5. **Production Strategies**
   - Preserve identifier boundaries
   - Handle language-specific whitespace
   - Add code-specific special tokens
   - Preprocess before tokenization

---

## 🧪 Practice Exercises

### Exercise 1: Build Simple Code Tokenizer

**Task:** Implement character-level tokenizer for Python code

```python
def char_tokenize(code: str) -> List[int]:
    """
    Your implementation here.

    Hint: Create char-to-id mapping
    """
    pass

# Test
code = "def add(a, b):\n    return a + b"
tokens = char_tokenize(code)
```

### Exercise 2: Compare Tokenization Methods

**Task:** Tokenize the same code with different methods and compare

```python
code = """
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count
"""

# Try:
# 1. Character-level
# 2. Word-level (split on whitespace)
# 3. BPE-based
# Compare: sequence length, semantic preservation
```

### Exercise 3: Parse Code with AST

**Task:** Extract all function names from Python file

```python
import ast

def extract_function_names(code: str) -> List[str]:
    """
    Extract all function names using AST.

    Hint: Use ast.parse() and ast.walk()
    """
    pass
```

---

## 🔗 Connection to Copilot

### How GitHub Copilot Tokenizes Code

**Copilot's approach (simplified):**

1. **Pre-tokenization**
   - Split on code boundaries (keywords, operators, identifiers)
   - Preserve function/variable names

2. **BPE Encoding**
   - ~50,000 token vocabulary
   - Learned from billions of lines of code
   - Optimized for common code patterns

3. **Special Tokens**
   - `<|endoftext|>` - End of file
   - `<|fim|>` - Fill-in-the-middle marker
   - Language-specific markers

4. **Context Window**
   - ~2048 tokens of context
   - Includes current file + nearby files
   - Smart context selection (imports, related functions)

---

## 🚀 Next Steps

Now that you understand code tokenization, you're ready for:

1. **Lesson 7:** Code Embeddings & Understanding
   - Semantic code search
   - Code similarity metrics
   - Cross-language understanding

2. **Lesson 8:** Training on Code (Codex-style)
   - Building training datasets
   - Fill-in-the-middle training
   - Multi-language training

3. **Building Your Own:**
   - Code completion engine
   - Code search system
   - Syntax highlighter

---

## 📚 Further Reading

**Papers:**
- "Evaluating Large Language Models Trained on Code" (Chen et al., 2021) - Codex
- "CodeBERT: A Pre-Trained Model for Programming and Natural Languages" (2020)
- "GraphCodeBERT: Pre-training Code Representations with Data Flow" (2021)

**Tools:**
- TreeSitter: https://tree-sitter.github.io/
- SentencePiece (BPE implementation): https://github.com/google/sentencepiece
- Python AST documentation: https://docs.python.org/3/library/ast.html

---

**You now understand how to tokenize code properly!** 🎉

**Next lesson:** We'll learn how to create embeddings that understand code semantics!

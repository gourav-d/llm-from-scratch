"""
Example 6: Code Representation & Tokenization

This example demonstrates how to tokenize code properly for training
code generation models like GitHub Copilot.

For .NET developers:
- Code tokenization = Like Roslyn's SyntaxToken
- AST = Like Roslyn's SyntaxTree
- BPE = Like building a custom code vocabulary

Author: Learn LLM from Scratch
Module: 7 - Reasoning & Coding Models
Lesson: 6 - Code Tokenization
"""

import re
import ast
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set


# ============================================================================
# PART 1: Character-Level Tokenization
# ============================================================================

def example_01_character_tokenization():
    """
    Example 1: Character-level tokenization (simplest approach).

    Like reading code one character at a time.
    """
    print("=" * 70)
    print("EXAMPLE 1: Character-Level Tokenization")
    print("=" * 70)

    code = "def add(a, b):\n    return a + b"

    print(f"Original code:")
    print(code)
    print()

    # Character-level tokenization
    char_tokens = list(code)

    print(f"Character tokens ({len(char_tokens)} tokens):")
    print(char_tokens[:50])  # Show first 50
    print()

    # Build character vocabulary
    char_vocab = sorted(set(char_tokens))

    print(f"Character vocabulary (size: {len(char_vocab)}):")
    print(char_vocab)
    print()

    # Map characters to IDs
    char_to_id = {char: idx for idx, char in enumerate(char_vocab)}
    id_to_char = {idx: char for char, idx in char_to_id.items()}

    # Encode
    encoded = [char_to_id[char] for char in char_tokens]

    print(f"Encoded (first 20 IDs): {encoded[:20]}")
    print()

    # Decode
    decoded = ''.join([id_to_char[id_] for id_ in encoded])

    print(f"Decoded code:")
    print(decoded)
    print()

    print("✅ Character tokenization works!")
    print("⚠️  But sequence is very long (inefficient)\n")


# ============================================================================
# PART 2: Word-Level Tokenization (Naive)
# ============================================================================

def example_02_word_tokenization():
    """
    Example 2: Naive word-level tokenization.

    Shows why treating code like natural language fails.
    """
    print("=" * 70)
    print("EXAMPLE 2: Naive Word-Level Tokenization")
    print("=" * 70)

    code = "def calculate_sum(a, b):\n    return a + b"

    print(f"Original code:")
    print(code)
    print()

    # Naive approach: split on whitespace
    naive_tokens = code.split()

    print(f"Naive tokenization ({len(naive_tokens)} tokens):")
    for i, token in enumerate(naive_tokens):
        print(f"  {i}: '{token}'")
    print()

    print("❌ Problems:")
    print("   - 'calculate_sum(a,' is one token (wrong!)")
    print("   - Lost structure (parentheses, colons)")
    print("   - Indentation lost")
    print()

    # Better approach: regex-based splitting
    pattern = r'(\w+|[+\-*/=<>!]=?|[(){}\[\],;:.]|[ \t\n]+)'
    better_tokens = [t for t in re.findall(pattern, code) if t.strip()]

    print(f"Better tokenization ({len(better_tokens)} tokens):")
    for i, token in enumerate(better_tokens):
        print(f"  {i}: '{token}'")
    print()

    print("✅ Much better!")
    print("   - Operators separated")
    print("   - Identifiers preserved")
    print("   - Structure maintained\n")


# ============================================================================
# PART 3: BPE (Byte-Pair Encoding) Tokenization
# ============================================================================

class SimpleBPETokenizer:
    """
    Simple BPE tokenizer for code.

    This is the approach used by GitHub Copilot!
    """

    def __init__(self, vocab_size: int = 500):
        self.vocab_size = vocab_size
        self.merges = {}  # Merge rules
        self.vocab = set()

    def train(self, code_samples: List[str]):
        """Train BPE on code samples."""
        print(f"Training BPE tokenizer (target vocab size: {self.vocab_size})...")

        # Step 1: Pre-tokenize all samples
        all_words = []
        for code in code_samples:
            words = self._pre_tokenize(code)
            all_words.extend(words)

        print(f"  Pre-tokenized into {len(all_words)} words")

        # Step 2: Initialize vocabulary with characters
        for word in all_words:
            self.vocab.update(word)

        initial_vocab_size = len(self.vocab)
        print(f"  Initial vocabulary size: {initial_vocab_size}")

        # Step 3: Learn merges
        num_merges = 0
        current_words = all_words.copy()

        while len(self.vocab) < self.vocab_size:
            # Count pairs
            pair_counts = self._count_pairs(current_words)

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            new_symbol = ''.join(best_pair)

            # Save merge rule
            self.merges[best_pair] = new_symbol
            self.vocab.add(new_symbol)

            # Apply merge to all words
            current_words = [self._merge_pair(word, best_pair, new_symbol)
                           for word in current_words]

            num_merges += 1

            if num_merges % 50 == 0:
                print(f"    Learned {num_merges} merges, vocab size: {len(self.vocab)}")

        print(f"  ✓ Learned {num_merges} merge rules")
        print(f"  ✓ Final vocabulary size: {len(self.vocab)}\n")

    def _pre_tokenize(self, code: str) -> List[Tuple[str]]:
        """Pre-tokenize code into words."""
        pattern = r'(\w+|[+\-*/=<>!]=?|[(){}\[\],;:.])'
        tokens = re.findall(pattern, code)
        # Convert each token to tuple of characters
        return [tuple(token) for token in tokens if token.strip()]

    def _count_pairs(self, words: List[Tuple[str]]) -> Counter:
        """Count adjacent symbol pairs in all words."""
        pairs = Counter()
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += 1
        return pairs

    def _merge_pair(self, word: Tuple[str], pair: Tuple[str], new_symbol: str) -> Tuple[str]:
        """Merge a specific pair in a word."""
        result = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                result.append(new_symbol)
                i += 2
            else:
                result.append(word[i])
                i += 1
        return tuple(result)

    def encode(self, code: str) -> List[str]:
        """Encode code into tokens."""
        words = self._pre_tokenize(code)

        # Apply all merges
        for pair, new_symbol in self.merges.items():
            words = [self._merge_pair(word, pair, new_symbol) for word in words]

        # Flatten
        tokens = []
        for word in words:
            tokens.extend(word)

        return tokens


def example_03_bpe_tokenization():
    """
    Example 3: BPE tokenization (like GitHub Copilot).
    """
    print("=" * 70)
    print("EXAMPLE 3: BPE Tokenization (Copilot-Style)")
    print("=" * 70)

    # Training data
    code_samples = [
        "def add(a, b):\n    return a + b",
        "def multiply(x, y):\n    return x * y",
        "def calculate_sum(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total",
        "def subtract(a, b):\n    return a - b",
        "def divide(x, y):\n    if y != 0:\n        return x / y\n    return None"
    ]

    print("Training on code samples:")
    for i, code in enumerate(code_samples[:2], 1):
        print(f"\n{i}. {code[:50]}...")
    print(f"... and {len(code_samples) - 2} more\n")

    # Train tokenizer
    tokenizer = SimpleBPETokenizer(vocab_size=200)
    tokenizer.train(code_samples)

    # Test on new code
    test_code = "def power(base, exponent):\n    return base ** exponent"

    print(f"Test code:")
    print(test_code)
    print()

    tokens = tokenizer.encode(test_code)

    print(f"BPE tokens ({len(tokens)} tokens):")
    print(tokens)
    print()

    print("🔍 Observations:")
    print("   - Common patterns merged (e.g., 'def', 'return')")
    print("   - Much shorter than character-level")
    print("   - Preserves code structure")
    print("   - This is how Copilot works!\n")


# ============================================================================
# PART 4: Abstract Syntax Tree (AST)
# ============================================================================

def example_04_ast_parsing():
    """
    Example 4: Parse code into Abstract Syntax Tree.

    Shows structure-aware code representation.
    """
    print("=" * 70)
    print("EXAMPLE 4: Abstract Syntax Tree (AST)")
    print("=" * 70)

    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    print("Python code:")
    print(code)
    print()

    # Parse into AST
    tree = ast.parse(code)

    print("AST Structure:")
    print("-" * 70)

    # Pretty print AST
    def print_ast(node, indent=0):
        """Recursively print AST."""
        prefix = "  " * indent
        print(f"{prefix}{node.__class__.__name__}", end="")

        # Print important attributes
        if hasattr(node, 'name'):
            print(f" (name: {node.name})", end="")
        if hasattr(node, 'id'):
            print(f" (id: {node.id})", end="")
        if isinstance(node, ast.Constant):
            print(f" (value: {node.value})", end="")

        print()

        # Recurse on children
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        print_ast(item, indent + 1)
            elif isinstance(value, ast.AST):
                print_ast(value, indent + 1)

    print_ast(tree)
    print()

    print("✅ AST captures code structure:")
    print("   - Function definition")
    print("   - Conditional logic")
    print("   - Recursive calls")
    print("   - All semantically meaningful!\n")


def example_05_ast_information_extraction():
    """
    Example 5: Extract information from AST.

    Shows practical use of AST parsing.
    """
    print("=" * 70)
    print("EXAMPLE 5: Extracting Information from AST")
    print("=" * 70)

    code = """
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count

def find_maximum(values):
    max_val = values[0]
    for val in values:
        if val > max_val:
            max_val = val
    return max_val

class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""

    print("Analyzing this code:")
    print(code[:150] + "...\n")

    # Parse
    tree = ast.parse(code)

    # Extract function names
    functions = []
    classes = []
    variables = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            variables.add(node.id)

    print("📊 Extracted Information:")
    print(f"   Functions: {functions}")
    print(f"   Classes: {classes}")
    print(f"   Variables: {sorted(variables)}")
    print()

    print("💡 This enables:")
    print("   - Code navigation (jump to definition)")
    print("   - Symbol search")
    print("   - Refactoring tools")
    print("   - Code completion suggestions\n")


# ============================================================================
# PART 5: Code-Specific Tokenization Strategies
# ============================================================================

def example_06_identifier_preservation():
    """
    Example 6: Preserving identifiers in tokenization.

    Shows importance of keeping function/variable names together.
    """
    print("=" * 70)
    print("EXAMPLE 6: Identifier Preservation")
    print("=" * 70)

    code = "calculate_total_sum(user_input_values)"

    print(f"Code: {code}\n")

    # Bad tokenization: naive split
    bad_pattern = r'(.{1,5})'  # Split every 5 chars
    bad_tokens = re.findall(bad_pattern, code)

    print("❌ Bad tokenization (character-based):")
    print(f"   {bad_tokens}")
    print("   Problem: Lost semantic meaning of identifiers!\n")

    # Good tokenization: preserve identifiers
    good_pattern = r'(\w+|[()])'
    good_tokens = re.findall(good_pattern, code)

    print("✅ Good tokenization (identifier-aware):")
    print(f"   {good_tokens}")
    print("   ✓ Function name preserved: 'calculate_total_sum'")
    print("   ✓ Variable name preserved: 'user_input_values'")
    print("   ✓ Semantics maintained!\n")


def example_07_whitespace_handling():
    """
    Example 7: Handling indentation and whitespace.

    Critical for Python where indentation = structure.
    """
    print("=" * 70)
    print("EXAMPLE 7: Whitespace & Indentation Handling")
    print("=" * 70)

    python_code = """
def calculate(x):
    if x > 0:
        return x * 2
    else:
        return 0
"""

    print("Python code (indentation matters!):")
    print(python_code)

    # Tokenize with indentation markers
    def tokenize_with_indent(code: str) -> List[str]:
        """Tokenize Python preserving indentation."""
        tokens = []
        lines = code.split('\n')

        for line in lines:
            if not line.strip():
                continue

            # Count indentation
            indent = len(line) - len(line.lstrip())

            if indent > 0:
                tokens.append(f'<INDENT:{indent}>')

            # Tokenize line content
            pattern = r'(\w+|[+\-*/=<>!]=?|[():,])'
            line_tokens = re.findall(pattern, line.strip())
            tokens.extend(line_tokens)

            tokens.append('<NEWLINE>')

        return tokens

    tokens = tokenize_with_indent(python_code)

    print("Tokens with indentation markers:")
    for i, token in enumerate(tokens):
        if token.startswith('<INDENT'):
            print(f"  {i}: {token} ← Indentation marker")
        elif token == '<NEWLINE>':
            print(f"  {i}: {token}")
        else:
            print(f"  {i}: {token}", end="  ")
        if (i + 1) % 5 == 0 and not token == '<NEWLINE>':
            print()

    print("\n✅ Indentation preserved!")
    print("   Model can learn Python's structure from tokens\n")


# ============================================================================
# PART 6: Complete Code Tokenizer (Production-Style)
# ============================================================================

class ProductionCodeTokenizer:
    """
    Production-quality code tokenizer.

    Combines best practices:
    - BPE for efficiency
    - Identifier preservation
    - Whitespace handling
    - Special tokens
    """

    def __init__(self):
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<INDENT>': 4,
            '<DEDENT>': 5,
            '<NEWLINE>': 6,
            '<MASK>': 7
        }

        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        self.next_id = len(self.special_tokens)

    def train(self, code_samples: List[str]):
        """Train tokenizer on code samples."""
        # Build vocabulary from code
        all_tokens = []

        for code in code_samples:
            tokens = self._smart_tokenize(code)
            all_tokens.extend(tokens)

        # Count token frequencies
        token_counts = Counter(all_tokens)

        # Add most common tokens to vocabulary
        for token, count in token_counts.most_common(1000):
            if token not in self.token_to_id:
                self.token_to_id[token] = self.next_id
                self.id_to_token[self.next_id] = token
                self.next_id += 1

    def _smart_tokenize(self, code: str) -> List[str]:
        """
        Smart tokenization preserving:
        - Identifiers
        - Operators
        - Indentation
        - Structure
        """
        tokens = []
        lines = code.split('\n')

        for line in lines:
            if not line.strip():
                continue

            # Handle indentation
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                tokens.append(f'<INDENT:{indent}>')

            # Tokenize line
            # Pattern preserves:
            # - Words/identifiers (\w+)
            # - Operators ([+\-*/...])
            # - Punctuation ([(){}...])
            pattern = r'(\w+|[+\-*/=<>!]=?|[(){}\[\],;:.]|"[^"]*"|\'[^\']*\')'
            line_tokens = re.findall(pattern, line.strip())
            tokens.extend(line_tokens)

            tokens.append('<NEWLINE>')

        return tokens

    def encode(self, code: str) -> List[int]:
        """Encode code to token IDs."""
        tokens = self._smart_tokenize(code)

        # Add BOS and EOS
        tokens = ['<BOS>'] + tokens + ['<EOS>']

        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id['<UNK>'])

        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to code."""
        tokens = [self.id_to_token.get(id_, '<UNK>') for id_ in token_ids]

        # Remove special tokens
        tokens = [t for t in tokens if not (t.startswith('<') and t.endswith('>'))]

        # Reconstruct code (simple version)
        return ' '.join(tokens)


def example_08_production_tokenizer():
    """
    Example 8: Production-quality code tokenizer.
    """
    print("=" * 70)
    print("EXAMPLE 8: Production Code Tokenizer")
    print("=" * 70)

    # Training data
    training_code = [
        "def add(a, b):\n    return a + b",
        "def multiply(x, y):\n    return x * y",
        "class Calculator:\n    def __init__(self):\n        self.result = 0",
    ]

    print("Training tokenizer on code samples...")
    tokenizer = ProductionCodeTokenizer()
    tokenizer.train(training_code)

    print(f"✓ Vocabulary size: {len(tokenizer.token_to_id)}\n")

    # Test
    test_code = "def power(base, exp):\n    return base ** exp"

    print(f"Test code:")
    print(test_code)
    print()

    # Encode
    encoded = tokenizer.encode(test_code)

    print(f"Encoded ({len(encoded)} tokens): {encoded[:20]}...")
    print()

    # Decode
    decoded = tokenizer.decode(encoded)

    print(f"Decoded: {decoded}")
    print()

    print("✅ Production tokenizer features:")
    print("   ✓ Preserves identifiers")
    print("   ✓ Handles indentation")
    print("   ✓ Special tokens for structure")
    print("   ✓ Efficient encoding/decoding\n")


# ============================================================================
# PART 7: Comparison of Methods
# ============================================================================

def example_09_tokenization_comparison():
    """
    Example 9: Compare different tokenization methods.
    """
    print("=" * 70)
    print("EXAMPLE 9: Tokenization Methods Comparison")
    print("=" * 70)

    test_code = "def calculate_sum(numbers):\n    total = sum(numbers)\n    return total"

    print(f"Test code:")
    print(test_code)
    print()

    # Method 1: Character-level
    char_tokens = list(test_code)

    # Method 2: Word-level
    word_pattern = r'(\w+|[+\-*/=<>!]=?|[(){}\[\],;:.])'
    word_tokens = re.findall(word_pattern, test_code)

    # Method 3: Smart (with indentation)
    tokenizer = ProductionCodeTokenizer()
    tokenizer.train([test_code])
    smart_tokens = tokenizer._smart_tokenize(test_code)

    # Compare
    print("📊 Comparison:")
    print("-" * 70)
    print(f"{'Method':<25} {'Tokens':<10} {'Vocab Size':<15} {'Semantics'}")
    print("-" * 70)
    print(f"{'Character-level':<25} {len(char_tokens):<10} {len(set(char_tokens)):<15} {'Low'}")
    print(f"{'Word-level':<25} {len(word_tokens):<10} {len(set(word_tokens)):<15} {'Medium'}")
    print(f"{'Smart (Production)':<25} {len(smart_tokens):<10} {len(set(smart_tokens)):<15} {'High'}")
    print()

    print("💡 Trade-offs:")
    print("   Character: Simple but inefficient")
    print("   Word: Better but loses structure")
    print("   Smart: Best for code (used in production)")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples."""
    print("\n")
    print("=" * 70)
    print("CODE TOKENIZATION & REPRESENTATION")
    print("Module 7, Lesson 6 - Examples")
    print("=" * 70)
    print("\nThis demonstrates how GitHub Copilot tokenizes code!\n")

    # Run examples
    example_01_character_tokenization()
    input("Press Enter to continue to Example 2...")

    example_02_word_tokenization()
    input("Press Enter to continue to Example 3...")

    example_03_bpe_tokenization()
    input("Press Enter to continue to Example 4...")

    example_04_ast_parsing()
    input("Press Enter to continue to Example 5...")

    example_05_ast_information_extraction()
    input("Press Enter to continue to Example 6...")

    example_06_identifier_preservation()
    input("Press Enter to continue to Example 7...")

    example_07_whitespace_handling()
    input("Press Enter to continue to Example 8...")

    example_08_production_tokenizer()
    input("Press Enter to continue to Example 9...")

    example_09_tokenization_comparison()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
🎉 You now understand code tokenization!

Key Concepts:
1. Code ≠ Natural Language
   - Has syntax, structure, semantics
   - Needs specialized tokenization

2. Three Main Approaches:
   - Character-level: Simple, inefficient
   - BPE/WordPiece: Best balance (Copilot uses this!)
   - AST-based: Structure-aware, complex

3. Critical Features:
   - Preserve identifiers (function/variable names)
   - Handle indentation (Python)
   - Add special tokens (<INDENT>, <NEWLINE>)
   - Code-aware vocabulary

4. Abstract Syntax Trees (AST):
   - Represent code as meaningful structure
   - Enable syntax validation
   - Support code transformations
   - Used for code analysis

5. Production Strategies:
   - BPE with ~50k vocabulary
   - Smart pre-tokenization
   - Language-specific handling
   - Special tokens for structure

How GitHub Copilot Works:
┌────────────────────────────────────┐
│  Your Code                         │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  Smart Pre-tokenization            │
│  - Preserve identifiers            │
│  - Mark indentation                │
│  - Handle operators                │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  BPE Encoding                      │
│  - ~50k token vocabulary           │
│  - Learned from billions of lines  │
│  - Optimized for code patterns     │
└──────────────┬─────────────────────┘
               ↓
┌────────────────────────────────────┐
│  Token IDs → Model → Predictions   │
└────────────────────────────────────┘

Next Steps:
1. Lesson 7: Code Embeddings
   - Semantic code search
   - Code similarity
   - Cross-language understanding

2. Lesson 8: Training on Code
   - Building datasets
   - Fill-in-the-middle training
   - Multi-language training

3. Build Your Own:
   - Code completion engine
   - Syntax highlighter
   - Code search system

You're ready to build Copilot-like systems! 🚀
    """)


if __name__ == "__main__":
    main()

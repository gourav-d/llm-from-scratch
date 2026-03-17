# Lesson 8: Training Models on Code (Codex-style)

## 📚 What You'll Learn

In this lesson, you'll learn how to **train language models specifically on code**, just like OpenAI did with Codex (the model behind GitHub Copilot). This is different from training on regular text!

**Think of it like this:** Training on code is like teaching someone a programming language. They need to learn syntax, patterns, common libraries, and how to write idiomatic code.

---

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand why code needs special training techniques
2. ✅ Know what Fill-in-the-Middle (FIM) training is and why it's critical
3. ✅ Learn how to prepare code datasets (like GitHub, Stack Overflow)
4. ✅ Implement data augmentation for code
5. ✅ Build a mini code generation model
6. ✅ Fine-tune models for specific programming languages

**Time:** 4-5 hours
**Difficulty:** Advanced
**Prerequisites:** Lesson 6 (Code Tokenization), Lesson 7 (Code Embeddings)

---

## 📖 Part 1: Why Code Training Is Different

### Challenge #1: Code Has Structure

**Problem:** Code isn't just text - it has syntax, semantics, and structure!

```python
# This is valid code:
def add(x, y):
    return x + y

# This is NOT valid (syntax error):
def add(x, y)
    return x + y  # Missing colon!
```

**Solution:** Training data must include valid code with correct syntax.

### Challenge #2: Code Has Context

**Problem:** Understanding code requires understanding surrounding context.

```python
# To autocomplete this line:
def calculate_total(prices):
    tax_rate = 0.08
    subtotal = sum(prices)
    # Next line should be: tax = subtotal * tax_rate

# The model needs to understand:
# - Function purpose (calculate_total)
# - Available variables (tax_rate, subtotal)
# - Expected return (total with tax)
```

**Solution:** Use Fill-in-the-Middle (FIM) training!

### Challenge #3: Code Completion vs Text Completion

**Text Completion (Normal):**
```
"The cat sat on the ___"
→ "mat" (predict next word)
```

**Code Completion (Special):**
```python
def process_data(data):
    # Cursor is HERE - need to complete the function!
    # Model should suggest relevant code based on:
    # 1. Function name
    # 2. Parameter types
    # 3. Common patterns
```

### C# Analogy

In C#, this is like IntelliSense:
```csharp
public class Calculator {
    public int Add(int x, int y) {
        // IntelliSense suggests: return x + y;
        // Based on:
        // - Method name (Add)
        // - Parameters (x, y)
        // - Return type (int)
    }
}
```

---

## 📖 Part 2: Fill-in-the-Middle (FIM) Training

### What Is FIM?

**Fill-in-the-Middle** is a training technique where the model learns to complete code **in the middle** of a file, not just at the end.

**Why FIM?** Because developers don't just write code top-to-bottom. They:
- Insert code in the middle of functions
- Add new methods between existing ones
- Fill in function bodies after writing signatures

### How FIM Works

**Normal Training (Left-to-Right):**
```python
Input:  "def add(x, y):"
Target: "return x + y"

Model learns: Given prefix, predict next tokens
```

**FIM Training (Fill Middle):**
```python
Prefix:  "def add(x, y):"
Suffix:  "return result"
Middle:  "result = x + y\n"

Model learns: Given prefix AND suffix, predict middle!
```

### FIM Format

**Special tokens mark prefix, middle, suffix:**

```
<fim_prefix>def add(x, y):<fim_suffix>return result<fim_middle>result = x + y
```

**In training:**
1. Take complete code sample
2. Randomly split into (prefix, middle, suffix)
3. Rearrange as: prefix → suffix → middle
4. Train model to predict middle given prefix + suffix

### Example

**Original code:**
```python
def calculate_total(prices):
    tax_rate = 0.08
    subtotal = sum(prices)
    tax = subtotal * tax_rate
    total = subtotal + tax
    return total
```

**FIM split:**
```python
# Prefix (before cursor):
def calculate_total(prices):
    tax_rate = 0.08

# Middle (what we want to predict):
    subtotal = sum(prices)
    tax = subtotal * tax_rate

# Suffix (after cursor):
    total = subtotal + tax
    return total
```

**Training format:**
```
<fim_prefix>def calculate_total(prices):
    tax_rate = 0.08
<fim_suffix>    total = subtotal + tax
    return total
<fim_middle>    subtotal = sum(prices)
    tax = subtotal * tax_rate

```

---

## 📖 Part 3: Data Preparation

### Where to Get Code Data?

| Source | Size | Quality | License |
|--------|------|---------|---------|
| **GitHub** | Huge (billions of lines) | Variable | Check repo licenses |
| **Stack Overflow** | Large (millions) | High (upvoted) | CC BY-SA |
| **Kaggle** | Medium | High | Varies |
| **CodeSearchNet** | Large | Curated | MIT |
| **The Stack** | Massive (6TB+) | Good | Permissive only |

### Data Cleaning Pipeline

```
Raw Code
    ↓
Remove duplicates
    ↓
Filter by language
    ↓
Remove generated/minified code
    ↓
Filter low-quality code
    ↓
Tokenize
    ↓
Create training samples
    ↓
Apply FIM transformation
    ↓
Ready for training!
```

### Quality Filters

**1. Remove Auto-Generated Code:**
```python
# Indicators of generated code:
# - "Auto-generated by..."
# - "Do not edit"
# - Minified code (no whitespace)
# - Very long lines (>500 chars)
```

**2. Filter by Metrics:**
```python
def is_good_quality(code):
    """
    Check if code is good quality for training.
    """
    # Too short (not useful)
    if len(code) < 50:
        return False

    # Too long (probably generated)
    if len(code) > 10000:
        return False

    # Check for actual code content
    lines = code.split('\n')

    # Too many blank lines
    blank_ratio = sum(1 for l in lines if not l.strip()) / len(lines)
    if blank_ratio > 0.5:
        return False

    # Has function definitions
    has_functions = 'def ' in code or 'class ' in code
    if not has_functions:
        return False

    return True
```

**3. Language Detection:**
```python
# Detect programming language
if code.startswith('def ') or 'import ' in code:
    language = 'python'
elif code.startswith('public class'):
    language = 'java'
elif code.startswith('using System'):
    language = 'csharp'
```

---

## 📖 Part 4: Data Augmentation for Code

### Why Augment Code?

**Goal:** Create more diverse training samples from limited data.

### Technique 1: Variable Renaming

```python
# Original
def add(x, y):
    return x + y

# Augmented (different names, same logic)
def add(a, b):
    return a + b

def add(num1, num2):
    return num1 + num2
```

**In C#:**
```csharp
// Original
public int Add(int x, int y) => x + y;

// Augmented
public int Add(int a, int b) => a + b;
```

### Technique 2: Code Formatting Variations

```python
# Variation 1: Compact
def add(x, y): return x + y

# Variation 2: Expanded
def add(x, y):
    result = x + y
    return result

# Variation 3: Comments
def add(x, y):
    # Add two numbers
    return x + y
```

### Technique 3: Equivalent Implementations

```python
# Version 1: List comprehension
squares = [x**2 for x in range(10)]

# Version 2: Loop
squares = []
for x in range(10):
    squares.append(x**2)

# Version 3: Map
squares = list(map(lambda x: x**2, range(10)))
```

**All three do the same thing!** Training on all versions helps the model learn multiple approaches.

### Technique 4: Docstring Variations

```python
# Version 1: Short docstring
def add(x, y):
    """Add two numbers."""
    return x + y

# Version 2: Detailed docstring
def add(x, y):
    """
    Add two numbers together.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of x and y
    """
    return x + y
```

---

## 📖 Part 5: Training Objectives

### Objective 1: Causal Language Modeling (CLM)

**Goal:** Predict next token given previous tokens (standard language modeling).

```python
Input:  "def add(x, y):"
Target: "return x + y"

Loss: CrossEntropy(predicted_tokens, actual_tokens)
```

**Use case:** Code generation from scratch

### Objective 2: Fill-in-the-Middle (FIM)

**Goal:** Predict middle section given prefix and suffix.

```python
Input:  "<fim_prefix>def add(x, y):<fim_suffix>return result<fim_middle>"
Target: "result = x + y\n"

Loss: CrossEntropy(predicted_middle, actual_middle)
```

**Use case:** Code completion (like Copilot)

### Objective 3: Masked Language Modeling (MLM)

**Goal:** Predict masked tokens (like BERT).

```python
Input:  "def add(x, y): return [MASK] + [MASK]"
Target: "def add(x, y): return x + y"

Loss: Only on masked tokens
```

**Use case:** Code understanding and embeddings

### Objective 4: Multi-Task Learning

**Combine all three!**

```python
Training batch:
- 60% CLM (next-token prediction)
- 30% FIM (fill middle)
- 10% MLM (masked tokens)

Result: Model good at generation AND completion!
```

---

## 📖 Part 6: Multi-Language Training

### Why Train on Multiple Languages?

**Benefits:**
1. **Transfer learning:** Patterns learned in Python help with JavaScript
2. **Broader applications:** One model for all languages
3. **Better generalization:** Learns programming concepts, not just syntax

### Language-Specific Tokens

```python
# Add language identifier
<python>def add(x, y):
    return x + y
</python>

<javascript>function add(x, y) {
    return x + y;
}
</javascript>

<csharp>public int Add(int x, int y) {
    return x + y;
}
</csharp>
```

### Data Balancing

**Problem:** GitHub has much more JavaScript than Haskell!

**Solution:** Oversample rare languages, undersample common ones.

```python
language_distribution = {
    'python': 0.25,      # 25% of training data
    'javascript': 0.25,
    'java': 0.15,
    'csharp': 0.10,
    'go': 0.10,
    'rust': 0.05,
    'cpp': 0.05,
    'ruby': 0.05,
}
```

---

## 📖 Part 7: Fine-Tuning Strategies

### Approach 1: Domain-Specific Fine-Tuning

**Goal:** Adapt general code model to specific domain.

```python
# General model knows:
def add(x, y): return x + y

# After fine-tuning on data science code:
def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def train_model(X, y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
```

### Approach 2: Repository-Specific Fine-Tuning

**Goal:** Learn your codebase patterns.

```python
# Your company's coding style:
class DataProcessor:
    """Process data using our standard pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__)

    def process(self, data: DataFrame) -> DataFrame:
        self.logger.info("Processing data...")
        # ... company-specific logic
```

### Approach 3: Language-Specific Fine-Tuning

**Goal:** Become expert in one language.

```python
# Before fine-tuning (generic Python):
def process(data):
    return data

# After fine-tuning (idiomatic Python):
def process(data: List[Dict]) -> List[Dict]:
    """Process data using Pythonic patterns."""
    return [
        {k: v.lower() if isinstance(v, str) else v
         for k, v in item.items()}
        for item in data
    ]
```

---

## 📖 Part 8: Training Hyperparameters

### Key Hyperparameters for Code

| Parameter | Value (Code Models) | Reasoning |
|-----------|---------------------|-----------|
| **Context Length** | 2048-8192 tokens | Code needs long context |
| **Learning Rate** | 1e-4 to 5e-5 | Lower for fine-tuning |
| **Batch Size** | 8-32 | Larger batches for stability |
| **FIM Ratio** | 0.3 (30%) | Balance with CLM |
| **Temperature** | 0.2-0.8 | Lower for more deterministic code |
| **Top-p** | 0.9-0.95 | Nucleus sampling for diversity |

### Context Length

**Why longer?** Code files can be large!

```python
# Model with 512 tokens can't see full context:
class MyClass:
    def __init__(self):
        ...  # 100 lines of code

    def method1(self):
        ...  # 100 lines

    def method2(self):  # ← Model might not remember __init__!
        ...
```

**Solution:** Use 2K-8K context length (or more for modern models).

### Warmup Steps

```python
# Learning rate schedule for code training
def get_lr(step, warmup_steps=1000, max_lr=5e-5):
    if step < warmup_steps:
        # Warmup: gradually increase learning rate
        return max_lr * (step / warmup_steps)
    else:
        # Decay: gradually decrease
        return max_lr * 0.99 ** ((step - warmup_steps) / 1000)
```

---

## 📖 Part 9: Evaluation Metrics

### How do we know if the model is good?

### Metric 1: Perplexity

**What it measures:** How "surprised" the model is by the test code.

```python
# Lower perplexity = better

perplexity = exp(cross_entropy_loss)

# Example:
# Perplexity = 5   → Very good!
# Perplexity = 50  → Okay
# Perplexity = 500 → Bad
```

### Metric 2: Exact Match

**What it measures:** Percentage of times model predicts EXACTLY the right code.

```python
def exact_match(predicted, actual):
    """
    Check if predicted code exactly matches actual code.
    """
    # Strip whitespace
    pred = predicted.strip()
    act = actual.strip()

    return pred == act

# Example:
# Predicted: "return x + y"
# Actual:    "return x + y"
# Result:    100% match ✓
```

### Metric 3: BLEU Score

**What it measures:** N-gram overlap (from machine translation).

```python
# BLEU score: 0 to 100
# Higher = more similar

# Predicted: "return x + y"
# Actual:    "return a + b"
# BLEU:      ~75 (different variable names)
```

### Metric 4: CodeBLEU

**What it measures:** BLEU + code structure (syntax tree).

**Better for code than regular BLEU!**

```python
# Considers:
# - Token overlap (like BLEU)
# - AST similarity (code structure)
# - Data flow (variable dependencies)
```

### Metric 5: Pass@k

**What it measures:** Generate k samples, how many execute correctly?

```python
# Example:
problem = "Write a function to add two numbers"
test_cases = [
    (1, 2, 3),
    (5, 7, 12),
    (-1, 1, 0),
]

# Generate 10 solutions
for i in range(10):
    code = model.generate(problem)
    if passes_all_tests(code, test_cases):
        success_count += 1

pass_at_10 = success_count / 10
# If 7 out of 10 pass: Pass@10 = 70%
```

---

## 📖 Part 10: Advanced Techniques

### 1. Instruction Tuning for Code

**Goal:** Make model follow instructions.

```python
# Training format:
instruction = "Write a function to calculate factorial"
code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
"""

# Model learns: instruction → code
```

### 2. Code Infilling with Context

**Goal:** Better completions using surrounding code.

```python
# Context-aware completion:
class Calculator:
    def add(self, x, y):
        return x + y

    def multiply(self, x, y):
        # Cursor here - suggest: return x * y
        # Model knows we're in Calculator class
```

### 3. Retrieval-Augmented Generation

**Goal:** Use similar code from codebase.

```python
# When completing:
def process_user_data(user):
    # 1. Search codebase for similar functions
    # 2. Use them as context
    # 3. Generate completion

# Example retrieved context:
"""
def process_admin_data(admin):
    validate_permissions(admin)
    return admin.data
"""

# Suggested completion:
"""
def process_user_data(user):
    validate_permissions(user)
    return user.data
"""
```

### 4. Self-Training with Execution Feedback

**Goal:** Learn from code that actually runs!

```python
# 1. Generate code
generated_code = model.generate(prompt)

# 2. Execute it
try:
    exec(generated_code)
    # Success! Use as positive training example
    positive_examples.append(generated_code)
except:
    # Failed! Use as negative example
    negative_examples.append(generated_code)

# 3. Fine-tune on successful examples
```

---

## 🎓 Key Takeaways

### What We Learned

1. **Code training** requires special techniques (FIM, multi-language, data quality)
2. **Fill-in-the-Middle (FIM)** is critical for code completion
3. **Data preparation** is crucial: clean, filter, augment
4. **Multi-task learning** combines generation, completion, and understanding
5. **Evaluation** uses perplexity, exact match, BLEU, and Pass@k

### Training Pipeline

```
1. Collect code from GitHub/Stack Overflow
2. Clean and filter (remove generated, low-quality)
3. Tokenize using code-aware tokenizer
4. Apply FIM transformation (30% of samples)
5. Train with multi-task objective
6. Evaluate on held-out code
7. Fine-tune for specific domains
```

### FIM vs CLM

| Aspect | CLM (Standard) | FIM (Code) |
|--------|---------------|------------|
| **Direction** | Left-to-right only | Can fill middle |
| **Use Case** | Generation from scratch | Code completion |
| **Training** | Predict next token | Predict middle given prefix+suffix |
| **Example** | Generate full function | Complete partial code |

### C# Comparison

**Python:**
```python
# FIM training sample
<fim_prefix>def add(x, y):<fim_suffix>return result<fim_middle>result = x + y
```

**C# equivalent concept:**
```csharp
// IntelliSense completion
public int Add(int x, int y)
{
    // Cursor here - suggest: return x + y;
}
```

---

## 🧪 Quiz Time!

### Question 1: Multiple Choice

**What is Fill-in-the-Middle (FIM) training?**

A) Training that only uses the middle of code files
B) Training that predicts middle code given prefix and suffix
C) Training that ignores the beginning and end of files
D) Training that splits code into three equal parts

<details>
<summary>Click for answer</summary>

**Answer: B**

FIM training teaches the model to predict the middle section of code when given what comes before (prefix) and after (suffix). This is essential for code completion tools like GitHub Copilot.
</details>

### Question 2: Multiple Choice

**Why is FIM important for code completion?**

A) It's faster than regular training
B) It uses less memory
C) Developers often insert code in the middle, not just at the end
D) It requires less training data

<details>
<summary>Click for answer</summary>

**Answer: C**

In real development, programmers frequently add code in the middle of files - between functions, inside methods, etc. FIM training prepares the model for this realistic use case.
</details>

### Question 3: Short Answer

**What are three sources of training data for code models?**

<details>
<summary>Click for answer</summary>

**Answer:**

1. **GitHub**: Billions of lines of open-source code
2. **Stack Overflow**: High-quality code snippets with explanations
3. **CodeSearchNet/The Stack**: Curated datasets specifically for training

(Other valid answers: Kaggle, company repositories, coding competition sites)
</details>

### Question 4: True/False

**True or False: You should include auto-generated and minified code in your training data.**

<details>
<summary>Click for answer</summary>

**Answer: FALSE**

Auto-generated and minified code should be filtered out because:
- Auto-generated code often has weird patterns
- Minified code has no useful structure or formatting
- Both can teach the model bad habits

Include only human-written, well-formatted code.
</details>

---

## 🎯 Practice Exercises

### Exercise 1: Implement FIM Transformation

**Task:** Create a function that transforms code into FIM format.

```python
def create_fim_sample(code):
    """
    Convert code into FIM training format.

    Args:
        code: Complete code snippet

    Returns:
        (prefix, middle, suffix) tuple
    """
    # TODO: Implement this!
    # 1. Split code at random position
    # 2. Create another split
    # 3. Return (prefix, middle, suffix)
    pass
```

<details>
<summary>Solution</summary>

```python
import random

def create_fim_sample(code):
    lines = code.split('\n')

    if len(lines) < 3:
        # Too short for FIM
        return None

    # Choose two random split points
    split1 = random.randint(1, len(lines) - 2)
    split2 = random.randint(split1 + 1, len(lines) - 1)

    # Create prefix, middle, suffix
    prefix = '\n'.join(lines[:split1])
    middle = '\n'.join(lines[split1:split2])
    suffix = '\n'.join(lines[split2:])

    return prefix, middle, suffix

# Example usage:
code = """def calculate_total(prices):
    tax_rate = 0.08
    subtotal = sum(prices)
    tax = subtotal * tax_rate
    return subtotal + tax"""

prefix, middle, suffix = create_fim_sample(code)
print(f"Prefix: {prefix}")
print(f"Middle: {middle}")
print(f"Suffix: {suffix}")
```
</details>

### Exercise 2: Code Quality Filter

**Task:** Implement a function to filter low-quality code.

```python
def is_good_quality(code):
    """
    Check if code is suitable for training.

    Should filter out:
    - Too short code
    - Too many blank lines
    - Auto-generated code
    - Minified code

    Returns:
        bool: True if good quality
    """
    # TODO: Implement quality checks
    pass
```

<details>
<summary>Solution</summary>

```python
def is_good_quality(code):
    # Too short
    if len(code.strip()) < 50:
        return False

    # Too long (probably generated)
    if len(code) > 10000:
        return False

    lines = code.split('\n')

    # Too many blank lines
    non_blank = [l for l in lines if l.strip()]
    if len(non_blank) / len(lines) < 0.3:
        return False

    # Check for auto-generated markers
    auto_generated_markers = [
        'auto-generated',
        'do not edit',
        'generated by',
        'autogenerated'
    ]

    code_lower = code.lower()
    if any(marker in code_lower for marker in auto_generated_markers):
        return False

    # Check for minified code (very long lines)
    max_line_length = max(len(l) for l in lines)
    if max_line_length > 500:
        return False

    # Must have some code structure
    has_structure = any(keyword in code for keyword in ['def ', 'class ', 'function ', 'public '])
    if not has_structure:
        return False

    return True
```
</details>

---

## 🚀 Next Steps

### What's Next?

**In Lesson 9**, we'll learn **Code Generation & Completion**:
- Natural language to code
- Docstring to implementation
- Multi-line completion strategies
- Building mini-Copilot!

### Further Reading

1. **Codex Paper:** "Evaluating Large Language Models Trained on Code"
2. **InCoder:** "A Generative Model for Code Infilling and Synthesis"
3. **CodeGen:** "An Open Large Language Model for Code"
4. **AlphaCode:** "Competition-Level Code Generation"

### Practice Projects

1. Build a FIM data preprocessor
2. Create a code quality filter for a dataset
3. Fine-tune a small model on Python code
4. Implement code augmentation techniques
5. Build a training data pipeline

---

## 📝 Summary

### Training Pipeline Overview

```
┌──────────────────────────────────────────────────────────┐
│             Code Model Training Pipeline                  │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  1. Data Collection                                       │
│     ├── GitHub repositories                               │
│     ├── Stack Overflow                                    │
│     └── Curated datasets                                  │
│                                                            │
│  2. Data Cleaning                                         │
│     ├── Remove duplicates                                 │
│     ├── Filter auto-generated                             │
│     └── Quality checks                                    │
│                                                            │
│  3. Data Augmentation                                     │
│     ├── Variable renaming                                 │
│     ├── Format variations                                 │
│     └── Equivalent implementations                        │
│                                                            │
│  4. Tokenization                                          │
│     └── Code-aware tokenizer                              │
│                                                            │
│  5. FIM Transformation (30%)                              │
│     └── <fim_prefix>...<fim_suffix>...<fim_middle>       │
│                                                            │
│  6. Training                                              │
│     ├── Multi-task objective                              │
│     ├── Long context (2K-8K tokens)                       │
│     └── Multi-language support                            │
│                                                            │
│  7. Evaluation                                            │
│     ├── Perplexity                                        │
│     ├── Exact match                                       │
│     ├── CodeBLEU                                          │
│     └── Pass@k                                            │
│                                                            │
│  8. Fine-Tuning                                           │
│     ├── Domain-specific                                   │
│     ├── Repository-specific                               │
│     └── Language-specific                                 │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

### Key Metrics

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| **Perplexity** | Model confidence | < 10 |
| **Exact Match** | Perfect predictions | > 30% |
| **CodeBLEU** | Structure + tokens | > 60 |
| **Pass@k** | Code correctness | > 50% |

### Python Concepts Used

- File I/O for reading code datasets
- Random sampling for FIM splits
- String manipulation for tokenization
- List comprehensions for data filtering
- Regular expressions for pattern matching

---

**Congratulations!** You now understand how to train models on code! 🎉

**Next lesson:** Code Generation & Completion (Building mini-Copilot)

---

**Remember:** The key to training great code models is high-quality data and FIM training. Together, they enable powerful code completion!

**Created:** March 16, 2026
**Module:** 07 - Reasoning and Coding Models
**Part:** B - Coding Models
**Lesson:** 8 of 10

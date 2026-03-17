# Module 7 Quick Reference

**Quick lookup for Reasoning & Coding Models concepts**

---

## 📚 Part A: Reasoning Models

### Chain-of-Thought (CoT)

**Concept:** Make LLMs show step-by-step reasoning

**Few-Shot CoT:**
```python
# Provide examples of reasoning
prompt = """
Q: 5 + 3 × 2 = ?
A: Let's think:
   1. Order of operations: multiply first
   2. 3 × 2 = 6
   3. 5 + 6 = 11
   Answer: 11

Q: 7 + 4 × 3 = ?
A: Let's think:
"""
```

**Zero-Shot CoT:**
```python
prompt = "Question\nLet's think step by step:"
```

**When to use:** Math, logic, multi-step reasoning

---

### Self-Consistency

**Concept:** Generate multiple reasoning paths, vote for best answer

**Formula:**
```
Generate N solutions → Count answers → Pick majority
```

**Implementation:**
```python
solutions = [generate_cot(question) for _ in range(5)]
answers = [s['answer'] for s in solutions]
best = most_common(answers)  # Majority vote
```

**When to use:** High-stakes decisions, verification needed

---

### Tree-of-Thoughts (ToT)

**Concept:** Explore multiple reasoning branches like a search tree

**Structure:**
```
            [Problem]
           /    |    \
        [A]   [B]   [C]
        / \    |    / \
      [A1][A2][B1][C1][C2]
       ✓   ✗   ✗   ✗   ✓
```

**Implementation:**
```python
def tot_search(problem, max_depth=3):
    # Breadth-first search
    paths = [initial_state]
    for depth in range(max_depth):
        new_paths = []
        for path in paths:
            children = generate_next_thoughts(path)
            scored = score_thoughts(children)
            new_paths.extend(top_k(scored, k=3))
        paths = new_paths
    return best_path(paths)
```

**When to use:** Complex planning, game playing, optimization

---

### Process Supervision

**Concept:** Reward correct reasoning steps, not just final answer

**Comparison:**
```
Outcome Supervision:
  Reward: Final answer is correct ✓

Process Supervision:
  Step 1: ✓ reward
  Step 2: ✓ reward
  Step 3: ✗ no reward (wrong step)
  Step 4: ✓ reward
```

**Training:**
```python
for step in reasoning_steps:
    reward = verify_step(step)  # Reward each step
    update_model(step, reward)
```

**When to use:** Training reasoning models, building reliable AI

---

### Building Reasoning Systems (o1-like)

**Architecture:**
```python
class ReasoningLLM:
    def generate(self, question):
        # Phase 1: Think (hidden)
        thoughts = []
        while not solved:
            thought = self.generate_thought()
            thoughts.append(thought)
            if self.verify(thoughts):
                break

        # Phase 2: Answer (visible)
        answer = self.synthesize(thoughts)
        return answer, thoughts
```

**Key components:**
- Internal reasoning tokens
- Verification loop
- Process supervision
- Search over thought space

---

## 📚 Part B: Coding Models

### Code Tokenization

**Character-level:**
```python
"def hello():" → ['d','e','f',' ','h','e','l','l','o','(',')']
```

**Token-level:**
```python
"def hello():" → ['def', ' ', 'hello', '(', ')']
```

**AST-aware:**
```python
"def hello():" →
{
    'type': 'FunctionDef',
    'name': 'hello',
    'args': [],
    'body': []
}
```

**When to use:** AST-aware for better understanding

---

### Code Embeddings (Lesson 7)

**Purpose:** Represent code as vectors for semantic search

**Types:**
- **Token-level:** Each token → vector
- **Line-level:** Each line → vector
- **Function-level:** Entire function → vector (best for search)

**Cosine Similarity:**
```python
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude

# Result:
# 1.0  = identical
# 0.9+ = very similar
# 0.5  = somewhat similar
# 0.0  = unrelated
```

**Semantic Code Search:**
```python
# 1. Embed all code in database
code_embeddings = [embed(code) for code in codebase]

# 2. Embed query
query_embedding = embed("function that sorts an array")

# 3. Find similar
similarities = [cosine_similarity(query_embedding, emb)
                for emb in code_embeddings]
top_results = argsort(similarities)[-5:]  # Top 5
```

**Applications:**
- Code search (like GitHub)
- Duplicate detection
- Code recommendations
- Bug pattern finding

---

### Training on Code (Lesson 8)

**Data Preparation Pipeline:**
```
Raw Code → Clean → Filter → Augment → Tokenize → FIM → Training
```

**Quality Filters:**
```python
def is_good_quality(code):
    # Remove if:
    if len(code) < 50: return False           # Too short
    if len(code) > 10000: return False        # Too long
    if "auto-generated" in code: return False # Generated
    if max_line_length > 500: return False    # Minified
    return True
```

**Data Augmentation:**
```python
# Variable renaming
"def add(x, y)" → "def add(a, b)"

# Format variations
"def add(x, y): return x + y"  # Compact
→ "def add(x, y):\n    return x + y"  # Expanded

# Docstring variations
Add different docstring styles
```

**Fill-in-the-Middle (FIM):**
```python
# Original code
code = """
def add(a, b):
    result = a + b
    return result
"""

# FIM format
prefix = "def add(a, b):"
middle = "\n    result = a + b"
suffix = "\n    return result"

# Training sample
fim_sample = f"{prefix_token}{prefix}{suffix_token}{suffix}{middle_token}{middle}"
```

**Training Objectives:**
- **CLM (60%):** Standard left-to-right generation
- **FIM (30%):** Fill-in-the-middle completion
- **MLM (10%):** Masked token prediction

**Evaluation Metrics:**
```python
# Perplexity (lower = better)
perplexity = exp(cross_entropy_loss)
# Good: < 10, Bad: > 50

# Exact Match (higher = better)
exact_match = (predicted == actual)
# Good: > 30%

# Pass@k (higher = better)
pass_at_k = successful_samples / k
# Good: > 50% at k=10
```

---

### Code Generation (Lesson 9)

**Three Approaches:**

**1. Template Matching (Simple):**
```python
templates = {
    "add": "def {name}(a, b): return a + b",
    "sort": "def {name}(items): return sorted(items)"
}

# "function that adds numbers" → template["add"]
```

**2. Seq2Seq (Medium):**
```python
# Treat as translation: English → Python
encoder_output = encoder(natural_language)
code = decoder(encoder_output)
```

**3. Transformer (Best):**
```python
# Pre-trained model (like Codex)
code = model.generate(
    prompt="# Create function that sorts list\ndef",
    temperature=0.3  # Lower = more conservative
)
```

**Mini-Copilot Architecture:**
```
┌─────────────────────────────────┐
│      Context Gatherer           │
│  ├─ Imports                     │
│  ├─ Functions                   │
│  └─ Variables                   │
├─────────────────────────────────┤
│      Code Generator             │
│  └─ Generate N candidates       │
├─────────────────────────────────┤
│      Ranker                     │
│  ├─ Confidence (40%)            │
│  ├─ Syntax valid (20%)          │
│  ├─ Uses variables (20%)        │
│  └─ Style match (20%)           │
├─────────────────────────────────┤
│      Validator                  │
│  ├─ Parse AST                   │
│  ├─ Check syntax                │
│  └─ Auto-fix errors             │
└─────────────────────────────────┘
```

**Key Techniques:**

**Beam Search:**
```python
# Keep top K paths instead of just best
beams = [(prompt, 0.0)]  # (text, score)
for step in range(max_length):
    new_beams = []
    for text, score in beams:
        # Generate next tokens
        candidates = model.next_tokens(text, k=beam_width)
        for token, prob in candidates:
            new_beams.append((text + token, score + log(prob)))
    beams = sorted(new_beams, key=lambda x: x[1])[:beam_width]
```

**Nucleus (Top-P) Sampling:**
```python
# Sample from smallest set with cumulative prob ≥ p
sorted_probs = sort(probs, descending=True)
cumsum = cumulative_sum(sorted_probs)
cutoff = where(cumsum >= p)[0]  # First index ≥ p
sample_from(sorted_probs[:cutoff])
```

**When to use:**
- Natural language to code
- Docstring to implementation
- Code completion (single/multi-line)
- Fill-in-the-middle

---

### Code Evaluation (Lesson 10)

**HumanEval Format:**
```python
{
    'task_id': 'HumanEval/0',
    'prompt': '''def has_close_elements(numbers, threshold):
        """Check if any two numbers are closer than threshold."""
    ''',
    'entry_point': 'has_close_elements',
    'test': '''
def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) == False
    assert candidate([1.0, 2.8, 3.0], 0.3) == True
    '''
}
```

**Pass@k Metric:**
```python
def pass_at_k(n, c, k):
    """
    n = total samples
    c = correct samples
    k = samples to pick

    Returns: Probability ≥1 correct in k picks
    """
    # Probability all k picks are wrong
    prob_all_wrong = comb(n-c, k) / comb(n, k)

    # Pass@k = probability at least one correct
    return 1.0 - prob_all_wrong

# Example:
# 20 solutions, 5 correct
pass_at_1  = 5/20 = 0.25 = 25%
pass_at_10 = 0.708 = 70.8%  # Much better!
```

**Evaluation Pipeline:**
```
Solution → [Functional Test] → [Quality Check] → [Security Scan] → Score
              Pass@k             Complexity         Vulnerabilities
```

**Sandbox Execution:**
```python
# Subprocess (simple)
result = subprocess.run(
    ['python', temp_file],
    capture_output=True,
    timeout=5  # Prevent infinite loops
)

# Docker (production)
container.run(
    image='python:3.10',
    command=['python', '-c', code],
    mem_limit='128m',      # Limit memory
    network_disabled=True  # No network access
)
```

**Quality Metrics:**
```python
metrics = {
    'cyclomatic_complexity': 5,    # Lower = simpler (good)
    'lines_of_code': 20,
    'has_docstring': True,         # Good!
    'has_type_hints': True,        # Good!
    'num_comments': 3
}

# Complexity guidelines:
# 1-10:  Simple (good)
# 11-20: Moderate
# 21+:   Complex (refactor!)
```

**Security Checks:**
```python
dangers = [
    r'os\.system',           # Command injection
    r'subprocess.*shell=True', # Command injection
    r'\beval\(',             # Code injection
    r'\bexec\(',             # Code injection
    r'f["\'].*SELECT.*FROM',  # SQL injection
]

for pattern in dangers:
    if re.search(pattern, code):
        return UNSAFE
```

**When to use:**
- Benchmarking code models
- Automated testing
- Code quality gates
- Security scanning

---

## 🎯 When to Use What

### Reasoning Techniques

| Technique | Use Case | Speed | Accuracy |
|-----------|----------|-------|----------|
| **CoT** | Math, logic | Fast | High |
| **Self-Consistency** | High-stakes | Slow | Very High |
| **ToT** | Complex planning | Very Slow | Highest |
| **Process Supervision** | Training | N/A | Best |

### Coding Techniques

| Technique | Use Case | Complexity |
|-----------|----------|------------|
| **Character-level** | Simple, any language | Low |
| **Token-level** | Fast, efficient | Medium |
| **AST-aware** | Best understanding | High |
| **FIM** | Code completion | Medium |

---

## 💻 Code Templates

### CoT Template
```python
class ChainOfThought:
    def __init__(self, model):
        self.model = model

    def generate(self, question):
        prompt = f"{question}\nLet's think step by step:"
        return self.model.generate(prompt)
```

### Self-Consistency Template
```python
def self_consistent_solve(question, n=5):
    solutions = [cot_generate(question) for _ in range(n)]
    answers = [s['answer'] for s in solutions]
    return most_common(answers)
```

### ToT Template
```python
def tot_solve(problem, branching_factor=3, depth=4):
    queue = [initial_state(problem)]
    for _ in range(depth):
        new_states = []
        for state in queue:
            children = generate_children(state)
            scored = [(score(c), c) for c in children]
            top = sorted(scored)[:branching_factor]
            new_states.extend([s for _, s in top])
        queue = new_states
    return best_solution(queue)
```

### Code Generation Template
```python
def generate_code(description):
    candidates = []
    for _ in range(10):
        code = model.generate(description)
        if is_valid_syntax(code):
            candidates.append(code)
    return max(candidates, key=lambda c: score(c))
```

### Code Embedding Template (Lesson 7)
```python
class CodeSearchEngine:
    def __init__(self, embedder):
        self.embedder = embedder
        self.codebase = []
        self.embeddings = []

    def index(self, code_snippets):
        self.codebase = code_snippets
        self.embeddings = [self.embedder.embed(c) for c in code_snippets]

    def search(self, query, top_k=5):
        query_emb = self.embedder.embed(query)
        similarities = [cosine_similarity(query_emb, emb)
                       for emb in self.embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.codebase[i], similarities[i]) for i in top_indices]
```

### FIM Transformation Template (Lesson 8)
```python
def create_fim_sample(code):
    lines = code.split('\n')
    if len(lines) < 3:
        return None

    # Random split points
    split1 = random.randint(1, len(lines) - 2)
    split2 = random.randint(split1 + 1, len(lines))

    # Extract parts
    prefix = '\n'.join(lines[:split1])
    middle = '\n'.join(lines[split1:split2])
    suffix = '\n'.join(lines[split2:])

    # Format for training
    return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}"
```

### Mini-Copilot Template (Lesson 9)
```python
class MiniCopilot:
    def complete(self, file_content, cursor_pos):
        # 1. Gather context
        context = self.gather_context(file_content, cursor_pos)

        # 2. Build prompt
        prompt = file_content[:cursor_pos]

        # 3. Generate candidates
        candidates = []
        for i in range(5):
            code = self.model.generate(prompt, temp=0.3 + i*0.15)
            candidates.append((code, self.score(code, context)))

        # 4. Rank and validate
        ranked = sorted(candidates, key=lambda x: x[1], reverse=True)
        valid = [c for c, s in ranked if self.is_valid(c)]

        return valid[:3]  # Top 3 suggestions
```

### Pass@k Calculator Template (Lesson 10)
```python
def evaluate_model(model, problems):
    """Evaluate on HumanEval with Pass@k"""
    results = []

    for problem in problems:
        # Generate N solutions
        solutions = [model.generate(problem['prompt'])
                    for _ in range(100)]

        # Test each solution
        correct = sum(1 for s in solutions
                     if test_passes(s, problem['test']))

        # Calculate Pass@k
        n, c = len(solutions), correct
        passk = {
            'pass@1': pass_at_k(n, c, 1),
            'pass@10': pass_at_k(n, c, 10),
            'pass@100': pass_at_k(n, c, 100)
        }
        results.append(passk)

    # Average across all problems
    return {
        k: np.mean([r[k] for r in results])
        for k in ['pass@1', 'pass@10', 'pass@100']
    }
```

---

## 📊 Performance Metrics

### Reasoning Metrics

**Accuracy Improvement:**
- Baseline (no CoT): ~17%
- With CoT: ~78%
- With Self-Consistency: ~85%
- With ToT: ~90%+

### Coding Metrics

**Pass@k scores (HumanEval):**
- GPT-3: pass@1 = 0%
- Codex: pass@1 = 28.8%, pass@100 = 72.3%
- GPT-4: pass@1 = 67.0%, pass@100 = 90.2%
- Code Llama 34B: pass@1 = 53.7%

**Code Quality Scores:**
- Cyclomatic Complexity: 1-10 (good), 11-20 (ok), 21+ (refactor)
- Documentation: Docstring + type hints = +20 points
- Security: No vulnerabilities = critical

---

## 🔧 Debugging Tips

### Reasoning Issues

**Problem:** Model not showing steps
**Solution:** Add more few-shot examples

**Problem:** Steps are illogical
**Solution:** Use process supervision to reward valid steps only

**Problem:** Wrong final answer
**Solution:** Use self-consistency to verify

### Coding Issues

**Problem:** Syntax errors
**Solution:** Parse AST and filter invalid code

**Problem:** Wrong logic
**Solution:** Generate multiple samples, run tests

**Problem:** Slow generation
**Solution:** Use FIM for targeted completion

---

## 📚 Research Papers

### Must-Read (Reasoning)
1. Chain-of-Thought Prompting (Wei et al., 2022)
2. Tree of Thoughts (Yao et al., 2023)
3. Let's Verify Step by Step (Lightman et al., 2023)

### Must-Read (Coding)
1. Evaluating Code Models (Chen et al., 2021)
2. Code Llama (Meta, 2023)
3. InCoder (Fried et al., 2022)

---

## 🎓 Key Takeaways

### Reasoning
1. **CoT = Show your work** → Better accuracy
2. **Multiple paths** → More reliable (self-consistency)
3. **Search** → Best for complex problems (ToT)
4. **Process > Outcome** → Better training
5. **Test-time compute** → Scale reasoning by thinking longer

### Coding
1. **AST > Tokens** → Better understanding
2. **FIM** → Better completion (critical for Copilot!)
3. **Generate Many** → Filter & rank (use Pass@10, not Pass@1)
4. **Test Everything** → Verify correctness
5. **Context is King** → More context = better completions
6. **Security First** → Always sandbox untrusted code
7. **Quality Matters** → Beyond correctness (complexity, style, docs)

---

## ✅ Module 7 Complete!

**All 10 Lessons Covered:**

**Part A - Reasoning:**
1. Chain-of-Thought ✅
2. Self-Consistency ✅
3. Tree-of-Thoughts ✅
4. Process Supervision ✅
5. Building o1-like Systems ✅

**Part B - Coding:**
6. Code Tokenization ✅
7. Code Embeddings ✅
8. Training on Code ✅
9. Code Generation ✅
10. Code Evaluation ✅

**You can now:**
- Build o1-style reasoning systems
- Create Mini-Copilot code generators
- Evaluate code with HumanEval & Pass@k
- Train models on code with FIM
- Deploy production AI systems

---

**This is your quick reference - bookmark this page!** 📖

**For detailed explanations, see the full lessons in PART_A_REASONING and PART_B_CODING folders.**

**Last Updated:** March 17, 2026

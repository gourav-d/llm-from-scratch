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

### Code Embeddings

**Purpose:** Semantic search for code

**Example:**
```python
# Query
query_embedding = embed("sort an array")

# Find similar code
similarities = cosine_similarity(query_embedding, code_database)
results = top_k(similarities, k=5)

# Results:
# 1. def sort_array(arr): return sorted(arr)
# 2. def bubble_sort(arr): ...
# 3. def quick_sort(arr): ...
```

---

### Fill-in-the-Middle (FIM) Training

**Traditional (left-to-right):**
```python
"def add(a, b):\n    return a + b"
```

**FIM:**
```python
Prefix:  "def add(a, b):\n"
Suffix:  "\n    return result"
Middle:  "    result = a + b"  ← Model learns to fill
```

**Why:** Enables code completion in the middle!

---

### Code Generation

**Process:**
```
Natural language → Tokenize → Generate → Parse → Validate → Return
```

**Example:**
```python
def generate_code(description):
    # 1. Generate multiple candidates
    candidates = model.generate_n(description, n=10)

    # 2. Filter syntactically valid
    valid = [c for c in candidates if is_valid_syntax(c)]

    # 3. Rank by likelihood
    ranked = rank_by_probability(valid)

    return ranked[0]
```

---

### HumanEval Benchmark

**Format:**
```python
{
    'task_id': 'HumanEval/0',
    'prompt': 'def has_close_elements(numbers, threshold):',
    'canonical_solution': '...',
    'test': 'def check():\n    assert has_close_elements...'
}
```

**Metrics:**
- **pass@1:** First attempt success rate
- **pass@10:** Success in top 10 attempts
- **pass@100:** Success in top 100 attempts

**Calculation:**
```python
pass_at_k = (successful_samples / total_samples) at k attempts
```

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

---

## 📊 Performance Metrics

### Reasoning Metrics

**Accuracy Improvement:**
- Baseline (no CoT): ~17%
- With CoT: ~78%
- With Self-Consistency: ~85%
- With ToT: ~90%+

### Coding Metrics

**pass@k scores (HumanEval):**
- GPT-3: pass@1 = 0%
- Codex: pass@1 = 28.8%
- GPT-4: pass@1 = 67.0%
- Code Llama 34B: pass@1 = 53.7%

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

### Coding
1. **AST > Tokens** → Better understanding
2. **FIM** → Better completion
3. **Generate Many** → Filter & rank
4. **Test Everything** → Verify correctness

---

**This is your quick reference - bookmark this page!** 📖

**For detailed explanations, see the full lessons in PART_A_REASONING and PART_B_CODING folders.**

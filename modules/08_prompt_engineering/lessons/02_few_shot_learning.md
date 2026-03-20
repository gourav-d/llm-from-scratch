# Lesson 2: Few-Shot Learning

**Learn to teach AI through examples - the most powerful prompting technique**

---

## 🎯 Learning Objectives

After this lesson, you will:
- ✅ Understand what few-shot learning is and why it's powerful
- ✅ Know when to use few-shot vs zero-shot prompting
- ✅ Select effective examples for your tasks
- ✅ Structure few-shot prompts correctly
- ✅ Understand the trade-offs (cost, context length, accuracy)
- ✅ Build reusable few-shot patterns

**Time:** 3-4 hours
**Difficulty:** Intermediate
**Prerequisites:** Lesson 1 (Zero-Shot Prompting)

---

## 📚 What is Few-Shot Learning?

### Definition

**Few-shot learning** = Teaching the model by showing it **examples** of the task, then asking it to perform the same task on new data.

```python
# Zero-shot: Tell the model what to do
prompt = "Extract the name and email from this text"

# Few-shot: Show the model examples first
prompt = """
Extract name and email in format: {name: X, email: Y}

Example 1:
Input: "Contact John at john@email.com"
Output: {name: "John", email: "john@email.com"}

Example 2:
Input: "Reach out to Sarah (sarah.jones@company.com)"
Output: {name: "Sarah", email: "sarah.jones@company.com"}

Now extract from:
Input: "Get in touch with Mike via mike.wilson@startup.io"
Output:
"""
```

### Why "Few-Shot"?

The term comes from machine learning research:
- **Zero-shot**: No examples (0 shots)
- **One-shot**: One example (1 shot)
- **Few-shot**: A few examples (typically 2-10 shots)
- **Many-shot**: Many examples (10+ shots)

---

## 🔑 Why Few-Shot Learning is Powerful

### The Power of Examples

**Humans learn from examples:**
```
"Be concise" ← Abstract instruction (hard to follow)

vs.

"Like this: 'The project succeeds.' Not: 'The project has achieved
a state of successful completion.'" ← Concrete example (easy to follow)
```

**LLMs are the same!** Examples are often more effective than instructions.

### Real-World Impact

```python
# Without examples (zero-shot)
prompt = "Classify sentiment as positive, negative, or neutral"
accuracy = 70%  # Inconsistent definitions

# With examples (few-shot)
prompt = """
Classify sentiment:

"I love this product!" → positive
"This is terrible." → negative
"It's okay." → neutral

Classify: "Not bad for the price."
"""
accuracy = 95%  # Model learns your exact criteria
```

**Result: 25 percentage point improvement just by adding examples!**

---

## 🎨 Anatomy of a Few-Shot Prompt

### Template Structure

```
[INSTRUCTION]
Brief description of the task

[EXAMPLES]
Example 1:
Input: {example_input_1}
Output: {example_output_1}

Example 2:
Input: {example_input_2}
Output: {example_output_2}

[Additional examples...]

[ACTUAL TASK]
Now do the same:
Input: {actual_input}
Output:
```

### Example: Data Extraction

```python
# Complete few-shot prompt
prompt = """
Extract structured information from customer reviews.

Example 1:
Input: "Great product! Fast shipping. Highly recommend. 5 stars!"
Output: {
  "sentiment": "positive",
  "rating": 5,
  "mentions_shipping": true,
  "would_recommend": true
}

Example 2:
Input: "Product broke after 2 days. Waste of money. 1 star."
Output: {
  "sentiment": "negative",
  "rating": 1,
  "mentions_shipping": false,
  "would_recommend": false
}

Example 3:
Input: "It's okay. Does the job. Average quality."
Output: {
  "sentiment": "neutral",
  "rating": 3,
  "mentions_shipping": false,
  "would_recommend": false
}

Now extract from this review:
Input: "Excellent value! Arrived quickly and works perfectly."
Output:
"""
```

**This teaches the model your exact format and criteria!**

---

## 💡 When to Use Few-Shot vs Zero-Shot

### Decision Matrix

| Scenario | Best Approach | Why |
|----------|--------------|-----|
| Simple, common task | Zero-shot | Model already knows how |
| Complex or unusual task | Few-shot | Examples clarify expectations |
| Specific output format | Few-shot | Show exact structure |
| Ambiguous criteria | Few-shot | Examples define boundaries |
| Quick one-off task | Zero-shot | Faster, cheaper |
| Production system | Few-shot | More consistent results |

### When Zero-Shot is Better

```python
# Simple translation - zero-shot works fine
"Translate 'Hello' to Spanish"
→ "Hola"  ✅

# Common summarization - zero-shot works
"Summarize this article in 3 sentences"
→ Good enough for most cases ✅
```

### When Few-Shot is Better

```python
# Complex extraction with specific format
"Extract entities from text"
→ Inconsistent format ❌

# WITH examples showing exact format
→ Perfect consistency ✅

# Domain-specific classification
"Classify this support ticket"
→ May use wrong categories ❌

# WITH examples of each category
→ Uses your exact taxonomy ✅
```

---

## 🎯 Selecting Good Examples

### Example Selection Criteria

**1. Representative**
```python
# Bad: All similar examples
examples = [
    "I love it!",
    "Amazing!",
    "Best ever!"
]  # Only shows extreme positives

# Good: Diverse examples
examples = [
    "I love it!",           # Extreme positive
    "Pretty good",          # Moderate positive
    "It's okay",            # Neutral
    "Not great",            # Moderate negative
    "Terrible product"      # Extreme negative
]  # Covers the spectrum
```

**2. Clear and Unambiguous**
```python
# Bad: Ambiguous example
"This is interesting" → positive  # Why positive? Could be neutral

# Good: Clear example
"This exceeded my expectations!" → positive  # Clearly positive
```

**3. Edge Cases Included**
```python
# Include tricky cases in examples
examples = [
    "Not bad" → neutral,              # Double negative
    "Could be worse" → neutral,       # Backhanded
    "I don't hate it" → neutral,      # Negative framing
]  # Teaches model to handle complexity
```

**4. Match Task Complexity**
```python
# Bad: Simple examples for complex task
Task: Extract name, company, title, email, phone
Example: "John at Acme" → Only shows name and company

# Good: Examples matching full complexity
Example: "John Smith, CEO of Acme Corp, john@acme.com, 555-1234"
→ Shows all fields the model needs to extract
```

---

## 🔢 How Many Examples?

### The Sweet Spot

**Research findings:**
- **1-2 examples**: Big improvement over zero-shot
- **3-5 examples**: Optimal for most tasks
- **6-10 examples**: Diminishing returns
- **10+ examples**: Usually better to fine-tune

### Practical Guidelines

```python
# Simple binary classification
examples_needed = 2  # One for each class

# Multi-class classification (5 classes)
examples_needed = 5-10  # 1-2 per class

# Complex extraction task
examples_needed = 3-5  # Show variety of cases

# Extremely unusual task
examples_needed = 5-10  # Model needs more guidance
```

### Cost vs Accuracy Trade-off

```python
# Example token costs (approximate)
0 examples: 50 tokens  (cheapest)
2 examples: 200 tokens
5 examples: 500 tokens
10 examples: 1000 tokens (expensive)

# Accuracy improvement
0 → 2 examples: +20-30% accuracy  (best ROI)
2 → 5 examples: +10-15% accuracy  (good ROI)
5 → 10 examples: +5% accuracy     (diminishing returns)
```

**Recommendation:** Start with 3 examples, add more only if needed.

---

## 🎨 Few-Shot Patterns

### Pattern 1: Classification

```python
CLASSIFICATION_TEMPLATE = """
Classify the following text into one of these categories: {categories}

{examples}

Now classify:
Text: {text}
Category:
"""

# Usage
prompt = CLASSIFICATION_TEMPLATE.format(
    categories="spam, not_spam",
    examples="""
Text: "WIN FREE IPHONE NOW!!!" → spam
Text: "Meeting scheduled for 3pm" → not_spam
Text: "CLICK HERE FOR PRIZES" → spam
""",
    text="You've won a million dollars!"
)
```

### Pattern 2: Data Extraction

```python
EXTRACTION_TEMPLATE = """
Extract information in this exact format:
{format_description}

{examples}

Now extract from:
{input_text}
"""

# Usage
prompt = EXTRACTION_TEMPLATE.format(
    format_description='{"name": str, "age": int, "city": str}',
    examples="""
Input: "Sarah, 28, lives in Boston"
Output: {"name": "Sarah", "age": 28, "city": "Boston"}

Input: "John from NYC is 35 years old"
Output: {"name": "John", "age": 35, "city": "NYC"}
""",
    input_text="Meet Alice, she's 42 and from Seattle"
)
```

### Pattern 3: Transformation

```python
TRANSFORMATION_TEMPLATE = """
Transform the input using the following style:

{examples}

Now transform:
{input_text}
"""

# Usage
prompt = TRANSFORMATION_TEMPLATE.format(
    examples="""
Input: "The quick brown fox jumps"
Output: "the_quick_brown_fox_jumps"

Input: "Hello World Example"
Output: "hello_world_example"
""",
    input_text="Convert To Snake Case"
)
```

### Pattern 4: Question Answering

```python
QA_TEMPLATE = """
Answer questions based on the given context.

{examples}

Context: {context}
Question: {question}
Answer:
"""

# Usage
prompt = QA_TEMPLATE.format(
    examples="""
Context: "The Eiffel Tower is in Paris, France. Built in 1889."
Question: "Where is the Eiffel Tower?"
Answer: "Paris, France"

Context: "Python was created by Guido van Rossum in 1991."
Question: "Who created Python?"
Answer: "Guido van Rossum"
""",
    context="The capital of Japan is Tokyo. Population: 14 million.",
    question="What is the population of Tokyo?"
)
```

---

## ⚖️ Few-Shot vs Fine-Tuning

### When to Use Each

```python
# Few-Shot Learning: Quick, flexible, no training needed
Use when:
- Need results immediately
- Task changes frequently
- Have few examples (<100)
- Don't want infrastructure complexity

# Fine-Tuning: Better performance, requires training
Use when:
- Have lots of examples (1000+)
- Task is stable and well-defined
- Need maximum accuracy
- Cost per inference matters (fine-tuned models can be cheaper at scale)
```

### Comparison

| Aspect | Few-Shot | Fine-Tuning |
|--------|----------|-------------|
| **Setup time** | Seconds | Hours to days |
| **Examples needed** | 2-10 | 1000+ |
| **Accuracy** | Good (80-90%) | Best (95-99%) |
| **Cost per call** | Higher (longer prompts) | Lower |
| **Flexibility** | High (change anytime) | Low (need to retrain) |
| **Infrastructure** | None | Training pipeline needed |

**Often the best approach: Start with few-shot, fine-tune if needed later!**

---

## 🚫 Common Mistakes

### Mistake 1: Examples Too Similar

```python
# Bad: All examples basically the same
examples = [
    "I love this!" → positive,
    "This is great!" → positive,
    "Amazing product!" → positive
]  # Doesn't teach model the boundaries

# Good: Diverse examples showing edge cases
examples = [
    "I love this!" → positive,
    "It's fine" → neutral,
    "Disappointing" → negative,
    "Not the worst" → neutral,  # Tricky case
]
```

### Mistake 2: Inconsistent Format

```python
# Bad: Different formats in examples
examples = """
Input: "text1" → positive
"text2": negative
text3 = neutral
"""  # Model gets confused

# Good: Consistent format
examples = """
Input: "text1" → positive
Input: "text2" → negative
Input: "text3" → neutral
"""  # Clear pattern
```

### Mistake 3: Too Many Examples

```python
# Bad: 20 examples in prompt
# - Wastes tokens (expensive)
# - Model may focus on wrong patterns
# - Slower response time

# Good: 3-5 carefully chosen examples
# - Covers key variations
# - Cost-effective
# - Fast responses
```

### Mistake 4: Examples Don't Match Task

```python
# Bad: Training examples don't match real data
Training: "Extract from: 'Name: John, Age: 30'"
Real data: "John (30 years old) from Boston"
# Format mismatch causes errors

# Good: Examples match real data format
Training examples use same format as real inputs
```

---

## 🎓 Advanced Few-Shot Techniques

### Technique 1: Chain-of-Thought in Examples

```python
# Show reasoning in examples
prompt = """
Solve math word problems. Show your work.

Example:
Problem: "Sarah has 3 apples. She buys 2 more. How many total?"
Solution:
Step 1: Starting amount = 3 apples
Step 2: Additional amount = 2 apples
Step 3: Total = 3 + 2 = 5 apples
Answer: 5 apples

Now solve:
Problem: "A store had 15 items. Sold 7. How many remain?"
Solution:
"""
```

### Technique 2: Progressive Examples

```python
# Start simple, build complexity
prompt = """
Convert to snake_case:

# Simple example
"Hello" → "hello"

# Two words
"Hello World" → "hello_world"

# Multiple words with numbers
"Hello World 123" → "hello_world_123"

# Complex with special characters
"Hello-World_Test" → "hello_world_test"

Now convert: "ThisIsMyVariable"
"""
```

### Technique 3: Contrastive Examples

```python
# Show what TO do and what NOT to do
prompt = """
Generate professional email subject lines.

✅ Good examples:
"Q4 Results Available" (clear, concise)
"Action Required: Update Profile" (specific)

❌ Bad examples:
"Important!!!" (vague, excessive punctuation)
"Hey there :)" (too casual)

Now generate a subject line for: monthly team update
"""
```

### Technique 4: Multi-Turn Examples

```python
# Show conversation-style examples
prompt = """
Helpful assistant that asks clarifying questions.

Example 1:
User: "I need a report"
Assistant: "What type of report? (sales, inventory, analytics)"
User: "Sales report"
Assistant: "Which time period? (daily, weekly, monthly)"

Example 2:
User: "Schedule a meeting"
Assistant: "Who should attend?"
User: "The whole team"
Assistant: "Which day works best? (Mon-Fri)"

Now handle:
User: "I want to order"
Assistant:
"""
```

---

## 🔬 Experimenting with Few-Shot

### Testing Different Example Counts

```python
# Test with different numbers of examples
for num_examples in [1, 2, 3, 5, 10]:
    prompt = build_prompt(examples[:num_examples])
    result = llm.generate(prompt)
    accuracy = evaluate(result)
    cost = calculate_cost(prompt)

    print(f"{num_examples} examples: "
          f"Accuracy={accuracy}%, Cost=${cost}")

# Find sweet spot: minimal examples for target accuracy
```

### A/B Testing Examples

```python
# Test different example sets
example_set_a = [easy_examples]
example_set_b = [hard_examples]
example_set_c = [diverse_examples]

results = {
    'set_a': test_with_examples(example_set_a),
    'set_b': test_with_examples(example_set_b),
    'set_c': test_with_examples(example_set_c),
}

best_set = max(results, key=lambda x: results[x]['accuracy'])
```

---

## 💰 Cost Optimization

### Token Usage

```python
# Few-shot prompts use more tokens
zero_shot_prompt = "Classify sentiment: {text}"  # ~10 tokens

few_shot_prompt = """
Classify sentiment:
"I love it" → positive
"It's okay" → neutral
"I hate it" → negative

Classify: {text}
"""  # ~40 tokens (4x more expensive!)
```

### Optimization Strategies

**1. Cache Examples**
```python
# Store examples in system message (cached by API)
system_message = """
You classify sentiment. Use these examples:
{examples}
"""  # Cached, not charged repeatedly

user_message = "Classify: {text}"  # Only this is charged each time
```

**2. Use Fewer Examples**
```python
# Start with 2 examples
if accuracy < target:
    add_more_examples()
# Only use what you need
```

**3. Shorter Examples**
```python
# Verbose examples
"I really love this product so much!" → positive

# Concise examples (same information)
"Love it!" → positive  # 60% fewer tokens
```

---

## 📊 Measuring Few-Shot Performance

### Key Metrics

```python
def evaluate_few_shot_prompt(prompt_template, test_cases):
    results = {
        'accuracy': 0,
        'consistency': 0,
        'avg_tokens': 0,
        'avg_cost': 0,
        'avg_latency': 0
    }

    for test_case in test_cases:
        prompt = prompt_template.format(**test_case)

        # Measure
        start = time.time()
        response = llm.generate(prompt)
        latency = time.time() - start

        # Evaluate
        correct = response == test_case['expected']
        tokens = count_tokens(prompt + response)
        cost = calculate_cost(tokens)

        # Aggregate
        results['accuracy'] += correct
        results['avg_tokens'] += tokens
        results['avg_cost'] += cost
        results['avg_latency'] += latency

    # Average
    n = len(test_cases)
    results['accuracy'] /= n
    results['avg_tokens'] /= n
    results['avg_cost'] /= n
    results['avg_latency'] /= n

    return results
```

---

## ✅ Lesson Checklist

After completing this lesson, you should be able to:

- [ ] Explain what few-shot learning is
- [ ] Know when to use few-shot vs zero-shot
- [ ] Select 3-5 representative examples for a task
- [ ] Structure few-shot prompts correctly
- [ ] Understand the cost vs accuracy trade-off
- [ ] Avoid common few-shot mistakes
- [ ] Apply few-shot patterns to real tasks
- [ ] Optimize few-shot prompts for cost and performance

---

## 🚀 Next Steps

1. **Practice**: Open `examples/example_02_few_shot.py`
2. **Experiment**: Try different numbers of examples
3. **Exercise**: Complete `exercises/exercise_02_few_shot.py`
4. **Apply**: Create few-shot prompts for your work tasks
5. **Move On**: When comfortable, proceed to Lesson 3: Prompt Templates

---

## 💡 Key Takeaways

1. **Examples > Instructions** - Models learn better from examples than descriptions
2. **3-5 examples is the sweet spot** - More isn't always better
3. **Quality > Quantity** - Diverse, representative examples beat many similar ones
4. **Show the format** - Examples teach output structure perfectly
5. **Start few-shot, fine-tune later** - Get results fast, optimize later if needed
6. **Cache when possible** - Use system messages to reduce costs
7. **Test and iterate** - Find your optimal example count and selection

---

## 📚 Additional Resources

### Reading
- [Language Models are Few-Shot Learners (GPT-3 paper)](https://arxiv.org/abs/2005.14165)
- [What Makes Good In-Context Examples for GPT-3?](https://arxiv.org/abs/2101.06804)

### Tools
- Example selectors in LangChain
- DSPy for automatic example selection

### Practice
- Build a classification task with few-shot
- Compare zero-shot vs few-shot accuracy
- Test different example counts

---

**Remember:** Few-shot learning is often the secret weapon that makes the difference between a "meh" prompt and an amazing one. Master this, and you'll be ahead of 90% of prompt engineers!

**Next:** [Lesson 3: Prompt Templates](./03_prompt_templates.md)

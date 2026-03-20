# Lesson 1: Zero-Shot Prompting

**Learn to get great results without providing examples**

---

## 🎯 Learning Objectives

After this lesson, you will:
- ✅ Understand what zero-shot prompting is
- ✅ Write clear, specific prompts that get accurate results
- ✅ Know when to use zero-shot vs few-shot
- ✅ Master the key components of effective prompts
- ✅ Avoid common prompting mistakes

**Time:** 2-3 hours
**Difficulty:** Beginner
**Prerequisites:** Basic understanding of LLMs

---

## 📚 What is Zero-Shot Prompting?

### Definition

**Zero-shot prompting** = Giving the model a task with **no examples**, just instructions.

```python
# Zero-shot: No examples provided
prompt = "Translate 'Hello' to French"
# Model uses its training to figure it out
```

### C# / .NET Comparison

Think of it like calling a method:

```csharp
// In C#, you call a method with parameters
string result = translator.Translate(
    text: "Hello",
    targetLanguage: "French"
);

// In prompt engineering, you "call" the LLM with instructions
string prompt = "Translate 'Hello' to French";
string result = llm.Complete(prompt);
```

**Key difference:** The LLM interprets natural language instructions, not typed parameters!

---

## 🔑 Key Components of Zero-Shot Prompts

### 1. Task Description (What to do)

```python
# Bad: Vague
prompt = "Do something with this data"

# Good: Specific
prompt = "Analyze the following sales data and identify the top 3 trends"
```

### 2. Context (Background information)

```python
# Without context
prompt = "Should we increase the price?"
# Model doesn't know what product, market, situation

# With context
prompt = """
You are a pricing analyst for a SaaS company.
Product: Project management tool
Current price: $10/month
Market: Small businesses
Competitors: $15-$20/month

Question: Should we increase the price?
"""
# Now the model can give informed advice
```

### 3. Format Specification (How to respond)

```python
# Bad: No format
prompt = "Analyze this customer review"
# Result: Unpredictable format

# Good: Specified format
prompt = """
Analyze this customer review.

Provide response in this format:
- Sentiment: [Positive/Negative/Neutral]
- Key issues: [Bullet points]
- Recommended action: [One sentence]
"""
# Result: Consistent, parseable format
```

### 4. Constraints (What NOT to do)

```python
# Without constraints
prompt = "Summarize this article"
# Result: Might be too long, too short, wrong focus

# With constraints
prompt = """
Summarize this article.

Constraints:
- Maximum 3 sentences
- Focus on business implications only
- Use simple language, no jargon
- Do not include opinions
"""
# Result: Exactly what you need
```

---

## 🎨 The Anatomy of a Good Zero-Shot Prompt

### Template

```
[ROLE/CONTEXT]
You are a [specific role] with expertise in [domain].

[TASK]
[Clear description of what to do]

[INPUT]
[The actual data/text to process]

[CONSTRAINTS]
- [Constraint 1]
- [Constraint 2]
- [Constraint 3]

[FORMAT]
Provide output in the following format:
[Specific format description]
```

### Example: Email Summarization

```python
# Poor prompt
prompt = "Summarize this email"

# Excellent prompt
prompt = """
You are an executive assistant helping busy executives
process their inbox efficiently.

Task: Summarize the following email.

Email:
{email_content}

Constraints:
- Maximum 2 sentences
- Focus on actionable items only
- Indicate urgency (High/Medium/Low)

Format:
Summary: [2 sentence summary]
Action required: [Yes/No]
Urgency: [High/Medium/Low]
Deadline: [If mentioned]
"""
```

**Result quality difference:** 3/10 → 9/10 just from better structure!

---

## 💡 Best Practices

### 1. Be Specific, Not Generic

```python
# Generic (Bad)
"Write a function to process data"

# Specific (Good)
"""
Write a Python function that:
1. Takes a list of dictionaries as input
2. Filters items where 'status' == 'active'
3. Returns a sorted list by 'created_date' descending
4. Includes error handling for missing keys
5. Has type hints and a docstring
"""
```

### 2. Use Clear Language

```python
# Ambiguous (Bad)
"Make it better"

# Clear (Good)
"""
Improve this code by:
1. Adding error handling for edge cases
2. Adding type hints
3. Improving variable names for clarity
4. Adding docstrings
"""
```

### 3. Specify the Output Format

```python
# Unspecified (Bad)
"What's the weather?"

# Specified (Good)
"""
What's the current weather in Tokyo?

Respond in JSON format:
{
  "temperature": "number in Celsius",
  "conditions": "string",
  "humidity": "percentage",
  "recommendation": "what to wear"
}
"""
```

### 4. Include Constraints

```python
# No constraints (Bad)
"Explain quantum computing"

# With constraints (Good)
"""
Explain quantum computing to a 10-year-old.

Constraints:
- Use only simple words (elementary school level)
- Maximum 5 sentences
- Use an analogy
- No technical jargon
"""
```

### 5. Provide Context When Needed

```python
# No context (Bad)
"Is this a good deal?"

# With context (Good)
"""
Context: I'm a small business owner looking to buy
accounting software for my 10-person company.

Product: QuickBooks Online
Price: $30/month
Features: Invoicing, expense tracking, basic reports

Competitors:
- Wave: Free but limited
- FreshBooks: $50/month with more features
- Xero: $40/month

Question: Is QuickBooks a good deal for my situation?
"""
```

---

## ⚖️ Zero-Shot vs Few-Shot: When to Use Each

### Use Zero-Shot When:
✅ Task is straightforward
✅ Model has been trained on similar tasks
✅ You want quick results
✅ Examples are hard to create

### Use Few-Shot When:
✅ Task is complex or unusual
✅ Specific format is critical
✅ Model needs to understand patterns
✅ Consistency is important

### Examples

```python
# GOOD for zero-shot (simple, common task)
"Translate 'Good morning' to Spanish"

# BETTER with few-shot (complex, specific format)
"Extract entities from text in our company's specific format"
```

---

## 🚫 Common Mistakes

### Mistake 1: Too Vague

```python
# Bad
prompt = "Help me with marketing"

# Good
prompt = """
You are a digital marketing expert specializing in B2B SaaS.

Create a 3-month content marketing strategy for:
- Product: AI-powered analytics tool
- Target: Data scientists and analysts
- Goal: Generate 50 qualified leads/month
- Budget: $5,000/month

Include:
1. Content types and topics
2. Distribution channels
3. KPIs to track
4. Expected timeline
"""
```

### Mistake 2: Assuming Model Knows Context

```python
# Bad (assumes model knows what "it" is)
prompt = "Is it good?"

# Good (provides all context)
prompt = """
Product: iPhone 15 Pro
Price: $999
Competing products: Samsung Galaxy S24 ($899), Google Pixel 8 Pro ($899)

Question: Is the iPhone 15 Pro a good value compared to competitors?
Consider: performance, camera, ecosystem, long-term support.
"""
```

### Mistake 3: No Output Format

```python
# Bad (unpredictable format)
prompt = "Analyze this code"

# Good (structured output)
prompt = """
Analyze the following Python code.

Provide analysis in this structure:

1. Code Quality (1-10): [score]
   - Issues: [bullet points]

2. Performance (1-10): [score]
   - Bottlenecks: [bullet points]

3. Security (1-10): [score]
   - Vulnerabilities: [bullet points]

4. Recommendations:
   - [Prioritized list]

Code:
{code_here}
"""
```

### Mistake 4: Multiple Unclear Tasks

```python
# Bad (too many vague tasks)
prompt = "Analyze, summarize, and provide recommendations"

# Good (clear, ordered tasks)
prompt = """
Perform the following analysis on the sales data:

Step 1: Identify top 3 trends
Step 2: Calculate month-over-month growth
Step 3: Highlight anomalies (>2 std deviations)
Step 4: Provide 3 actionable recommendations

Format each step clearly with headers.
"""
```

---

## 🎯 Practice Examples

### Example 1: Code Review

**Bad Prompt:**
```python
prompt = "Review this code"
```

**Good Prompt:**
```python
prompt = """
You are a senior Python developer conducting a code review.

Review the following code for:
1. Correctness - Does it work as intended?
2. Style - Does it follow PEP 8?
3. Performance - Any inefficiencies?
4. Security - Any vulnerabilities?
5. Maintainability - Is it readable and well-structured?

For each issue found, provide:
- Severity: [Critical/High/Medium/Low]
- Location: [Line number or function]
- Issue: [Description]
- Fix: [Specific recommendation]

Code:
{code}

Format as a structured report with sections for each category.
"""
```

### Example 2: Data Analysis

**Bad Prompt:**
```python
prompt = "What do you see in this data?"
```

**Good Prompt:**
```python
prompt = """
You are a data analyst specializing in e-commerce.

Analyze the following sales data:
{data}

Provide:

1. Summary Statistics:
   - Total sales
   - Average order value
   - Number of transactions

2. Trends:
   - Top 3 patterns observed
   - Any concerning trends

3. Insights:
   - 3 actionable business insights
   - Confidence level for each (High/Medium/Low)

4. Recommendations:
   - 3 specific actions to take
   - Expected impact of each

Use simple, non-technical language suitable for executives.
"""
```

### Example 3: Content Generation

**Bad Prompt:**
```python
prompt = "Write a blog post about AI"
```

**Good Prompt:**
```python
prompt = """
You are a technical content writer for a developer audience.

Write a blog post introduction (150-200 words) about:
Topic: Prompt Engineering for Developers
Audience: Software engineers with 2-5 years experience
Tone: Professional but conversational
Goal: Hook readers and make them want to learn more

Requirements:
- Start with a relatable problem
- Include 1-2 statistics
- End with a clear value proposition
- Avoid hype and buzzwords
- Use active voice

Format:
- 2-3 paragraphs
- Each paragraph: 2-4 sentences
- Include a compelling hook in first sentence
"""
```

---

## 🔬 Experimentation Guide

### Test Different Temperatures

```python
# Temperature affects creativity vs consistency

# Temperature 0 (Deterministic)
prompt = "List 3 benefits of exercise"
# Result: Same answer every time (good for consistency)

# Temperature 0.7 (Balanced)
prompt = "List 3 benefits of exercise"
# Result: Varied but sensible (good for creative tasks)

# Temperature 1.0+ (Creative)
prompt = "List 3 benefits of exercise"
# Result: Highly varied, sometimes unusual (good for brainstorming)
```

### Test Different Phrasings

```python
# Test variations
prompts = [
    "Summarize this article",
    "Provide a brief summary of this article",
    "Create a concise summary of the following article",
    "TL;DR this article",
]

# Run all, compare results
# Pick the one that works best for your use case
```

---

## 📊 Measuring Success

### How to Know if Your Prompt is Good

1. **Consistency**: Run 10 times, get similar quality?
2. **Accuracy**: Meets requirements?
3. **Efficiency**: Shortest prompt that works?
4. **Robustness**: Works with different inputs?

### Quick Test Framework

```python
def test_prompt(prompt_template, test_cases):
    results = []
    for test_case in test_cases:
        prompt = prompt_template.format(**test_case)
        result = llm.complete(prompt)

        # Evaluate
        score = evaluate_result(result, test_case['expected'])
        results.append(score)

    # Overall metrics
    avg_score = sum(results) / len(results)
    consistency = std_dev(results)  # Lower is better

    return {
        'avg_score': avg_score,
        'consistency': consistency,
        'pass_rate': sum(1 for r in results if r > 0.8) / len(results)
    }
```

---

## 🎁 Practical Templates

### Template 1: Analysis Task

```python
ANALYSIS_TEMPLATE = """
You are a {domain} expert.

Analyze the following {content_type}:
{content}

Provide:
1. Summary: [2-3 sentences]
2. Key findings: [3-5 bullet points]
3. Insights: [2-3 actionable insights]
4. Recommendations: [2-3 specific actions]

Format as structured sections with headers.
"""
```

### Template 2: Content Generation

```python
CONTENT_TEMPLATE = """
You are a {role} writing for {audience}.

Create {content_type} about: {topic}

Requirements:
- Length: {length}
- Tone: {tone}
- Style: {style}
- Include: {must_include}
- Avoid: {must_avoid}

Format:
{format_specification}
"""
```

### Template 3: Code Task

```python
CODE_TEMPLATE = """
You are an expert {language} developer.

Task: {task_description}

Requirements:
- Follow {style_guide}
- Include {requirements}
- Handle edge cases: {edge_cases}
- Performance: {performance_requirements}

Provide:
1. Implementation
2. Usage example
3. Test cases
4. Complexity analysis
"""
```

---

## ✅ Lesson Checklist

After completing this lesson, you should be able to:

- [ ] Explain what zero-shot prompting is
- [ ] List the 4 key components of a good prompt
- [ ] Write prompts that include role, task, constraints, and format
- [ ] Know when to use zero-shot vs few-shot
- [ ] Avoid the 4 common mistakes
- [ ] Create prompts using the templates provided
- [ ] Test and evaluate prompt quality

---

## 🚀 Next Steps

1. **Practice**: Open `examples/example_01_zero_shot.py`
2. **Experiment**: Try the templates with your own use cases
3. **Exercise**: Complete `exercises/exercise_01_zero_shot.py`
4. **Apply**: Rewrite 3 prompts you use at work
5. **Move On**: When comfortable, proceed to Lesson 2: Few-Shot Learning

---

## 📚 Additional Resources

### Reading
- [OpenAI: Prompt Engineering Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic: Introduction to Prompt Design](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design)

### Tools
- OpenAI Playground: platform.openai.com/playground
- Anthropic Console: console.anthropic.com

### Practice
- Try rewriting prompts you've used before
- Compare old vs new results
- Build a personal prompt library

---

## 💡 Key Takeaways

1. **Specificity is power** - The more specific your prompt, the better the result
2. **Structure matters** - Role, task, constraints, format = consistent results
3. **Context is king** - Give the model all information it needs
4. **Test and iterate** - First prompt is rarely the best prompt
5. **Simple is often better** - Don't over-complicate if unnecessary

---

**Remember:** Even this lesson's content was generated using good prompting techniques! The skills you learn here are what we use to create high-quality AI outputs!

**Next:** [Lesson 2: Few-Shot Learning](./02_few_shot_learning.md)

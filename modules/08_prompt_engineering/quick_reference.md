# Prompt Engineering Quick Reference

**Your cheat sheet for writing great prompts fast!**

---

## 🎯 The Golden Template

```
[ROLE] You are a {specific role} with expertise in {domain}.

[TASK] {Clear description of what to do}

[INPUT]
{The actual data/content to process}

[CONSTRAINTS]
- {Specific limitation 1}
- {Specific limitation 2}
- {What to avoid}

[FORMAT]
Provide output in this format:
{Exact format specification}
```

---

## ⚡ Quick Patterns

### Pattern 1: Simple Task
```
{Task} the following {content_type}:
{content}
```

### Pattern 2: Structured Output
```
{Task} and format as:
- Field 1: {description}
- Field 2: {description}

Input: {content}
```

### Pattern 3: Analysis
```
You are a {role}.
Analyze {content} for:
1. {Aspect 1}
2. {Aspect 2}
3. {Aspect 3}

Provide structured report.
```

### Pattern 4: Generation
```
You are a {role} writing for {audience}.
Create {content_type} about {topic}.
- Tone: {tone}
- Length: {length}
- Must include: {requirements}
```

---

## 🔑 Key Components Checklist

Before sending your prompt, check:

- [ ] **Role/Context**: Did I specify who the AI should act as?
- [ ] **Task**: Is it crystal clear what I want?
- [ ] **Input**: Did I provide all necessary data?
- [ ] **Constraints**: Did I specify what NOT to do?
- [ ] **Format**: Did I specify how to structure the output?
- [ ] **Examples**: (If few-shot) Did I provide good examples?

---

## 💡 Magic Phrases

### For Reasoning
- "Let's think step by step"
- "Before answering, consider..."
- "Show your work"
- "Explain your reasoning"

### For Accuracy
- "Double-check your answer"
- "Verify the calculation"
- "Be precise and specific"
- "Cite sources when possible"

### For Creativity
- "Think outside the box"
- "Generate 5 different approaches"
- "Be creative but practical"
- "Consider unconventional solutions"

### For Structured Output
- "Format as JSON"
- "Use bullet points"
- "Create a table with columns: {columns}"
- "Follow this exact schema"

---

## 🎨 Role Templates

### Expert Roles
```
You are a senior {profession} with {X} years of experience in {domain}.
```

### Specific Personas
```
You are {Famous Person}, known for {characteristic}.
Apply your {approach} to {task}.
```

### Multi-Role
```
You are both a {role1} and {role2}.
Consider both {perspective1} and {perspective2}.
```

---

## 📊 Output Format Examples

### JSON
```json
Return valid JSON:
{
  "field1": "type",
  "field2": number,
  "field3": ["array", "of", "strings"]
}
```

### Markdown Table
```markdown
| Column1 | Column2 | Column3 |
|---------|---------|---------|
| value   | value   | value   |
```

### Structured List
```
1. Category 1:
   - Point A
   - Point B
2. Category 2:
   - Point C
   - Point D
```

### Report Format
```
SUMMARY:
[2-3 sentences]

FINDINGS:
1. [Finding 1]
2. [Finding 2]

RECOMMENDATIONS:
- [Action 1]
- [Action 2]
```

---

## 🚫 Common Mistakes & Fixes

### ❌ Too Vague → ✅ Specific
```
Bad:  "Summarize this"
Good: "Summarize in 3 bullet points, max 20 words each, focusing on key actions"
```

### ❌ No Context → ✅ Full Context
```
Bad:  "Is this good?"
Good: "You are a {role}. Evaluate {item} against {criteria}. Context: {background}"
```

### ❌ Unclear Format → ✅ Exact Format
```
Bad:  "List the items"
Good: "List items as JSON array: ['item1', 'item2']"
```

### ❌ Multiple Tasks → ✅ Ordered Steps
```
Bad:  "Analyze, summarize, and recommend"
Good: "Step 1: Analyze for {aspects}. Step 2: Summarize in {format}. Step 3: Recommend {specifics}"
```

---

## 🎯 Task-Specific Quick Templates

### Email Writing
```
You are a professional communicator.
Write a {tone} email to {recipient} about {topic}.
- Purpose: {purpose}
- Key points: {points}
- Length: {length}
- Call to action: {cta}
```

### Code Review
```
You are a senior {language} developer.
Review this code for:
1. Correctness
2. Performance
3. Security
4. Style

Provide: severity, issue, recommendation for each finding.
```

### Data Analysis
```
You are a data analyst.
Analyze {data_type} and provide:
1. Summary statistics
2. Top 3 insights
3. Trends and patterns
4. Actionable recommendations

Format as structured report.
```

### Content Summarization
```
Summarize the following {content_type} in {length}.
Audience: {audience}
Focus on: {key_aspects}
Exclude: {what_to_skip}
```

### Translation
```
Translate to {language}.
- Preserve: {what_to_preserve}
- Tone: {tone}
- Audience: {audience}
- Technical terms: {how_to_handle}
```

---

## 🔧 Advanced Techniques (Quick Reference)

### Chain-of-Thought
```
{Question}

Let's solve this step by step:
1. First, {step 1}
2. Then, {step 2}
3. Finally, {step 3}
```

### Few-Shot Pattern
```
Task: {task_description}

Example 1:
Input: {input1}
Output: {output1}

Example 2:
Input: {input2}
Output: {output2}

Now do:
Input: {actual_input}
Output:
```

### Self-Consistency
```
Generate 3 different approaches to {problem}.
For each approach:
1. Reasoning
2. Solution
3. Confidence (Low/Medium/High)

Then recommend the best approach.
```

---

## 📏 Temperature Settings

```
Temperature 0.0:
- Most deterministic
- Use for: factual tasks, consistency needed
- Example: data extraction, classification

Temperature 0.3-0.5:
- Slightly varied
- Use for: analysis, summarization
- Example: code review, reports

Temperature 0.7:
- Balanced creativity
- Use for: content generation, general tasks
- Example: emails, articles

Temperature 1.0+:
- Highly creative
- Use for: brainstorming, creative writing
- Example: marketing copy, story generation
```

---

## 🛡️ Security Quick Checks

### Input Validation
```python
# Before sending user input
- Check for prompt injection patterns
- Sanitize special characters
- Limit input length
- Remove suspicious instructions
```

### Output Validation
```python
# After receiving response
- Verify format matches expected
- Check for PII leakage
- Validate against schema
- Ensure no malicious content
```

---

## 💰 Cost Optimization

### Token Reduction
```
Bad:  Long, verbose prompt
Good: Concise, specific prompt

Bad:  Repeated context in every call
Good: System message once + short user messages
```

### Model Selection
```
Simple tasks  → Use cheaper model (gpt-4o-mini, claude-haiku)
Complex tasks → Use smarter model (gpt-4o, claude-sonnet)
```

### Caching
```python
# Cache identical prompts
# Cache expensive computations
# Reuse results when possible
```

---

## 📈 Testing Prompts

### Quick Test
```python
# Test with 3-5 examples
# Check:
1. Accuracy - Correct results?
2. Format - Consistent structure?
3. Completeness - All required info?
```

### A/B Testing
```python
# Compare two prompts
prompt_a = "..."
prompt_b = "..."

# Run on same test set
# Measure: accuracy, consistency, cost
# Pick winner
```

---

## 🎓 Learning Progression

### Beginner (Week 1)
- Zero-shot prompting
- Basic templates
- Clear instructions
- Format specification

### Intermediate (Week 2)
- Few-shot learning
- Role prompting
- Chain-of-thought
- Structured outputs

### Advanced (Week 3+)
- Prompt optimization
- Security measures
- Cost optimization
- Production patterns

---

## 🔗 Quick Links

### Tools
- [OpenAI Playground](https://platform.openai.com/playground)
- [Claude Console](https://console.anthropic.com)
- [Promptfoo](https://promptfoo.dev)

### Documentation
- [OpenAI Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Learn Prompting](https://learnprompting.org)

---

## 💡 Pro Tips

1. **Start simple** - Add complexity only if needed
2. **Be specific** - Vague prompts = vague results
3. **Iterate** - First prompt is rarely the best
4. **Test** - Always validate with real data
5. **Measure** - Track what works
6. **Save** - Build your prompt library
7. **Share** - Learn from community

---

## 🚀 Emergency Prompt Improver

If your prompt isn't working, try adding:

1. **Role**: "You are a {specific expert}"
2. **Examples**: Show 1-2 examples
3. **Format**: "Format output as {specific structure}"
4. **Constraints**: "Do NOT {what to avoid}"
5. **Steps**: "Let's think step by step"

**Usually fixes 80% of issues!**

---

**Print this page and keep it next to your keyboard!** 📄

**For detailed explanations, see the full lessons in the module.**

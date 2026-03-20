# Lesson 4: Role and System Prompting

**Master the art of shaping AI personality and behavior through roles**

---

## 🎯 Learning Objectives

After this lesson, you will:
- Understand role prompting vs system prompts
- Create effective role definitions
- Use system prompts to control behavior
- Combine roles with other techniques
- Apply roles in production scenarios

**Time:** 60 minutes

---

## 📖 What is Role Prompting?

### The Concept

**Role prompting** = Telling the AI who it should be

Instead of:
```
"Analyze this data"
❌ Generic, unpredictable results
```

Use:
```
"You are a senior data scientist with 15 years of experience in retail analytics.
Analyze this sales data..."
✅ Focused, expert-level response
```

### Why Roles Matter

1. **Sets expectations** - AI knows what perspective to take
2. **Improves quality** - More expert, targeted responses
3. **Increases consistency** - Same role = same style
4. **Reduces ambiguity** - Clear context for the task

---

## 🎭 Types of Roles

### 1. Professional Roles

Expertise-based personas:

```
You are a:
- Senior software engineer specializing in distributed systems
- Financial analyst with CFA certification
- Legal expert specializing in contract law
- Medical doctor specializing in cardiology
- Marketing strategist with B2B SaaS experience
```

**When to use:** Need domain expertise, technical knowledge

### 2. Functional Roles

Task-oriented personas:

```
You are a:
- Helpful teaching assistant
- Objective code reviewer
- Concise summarizer
- Creative brainstorming partner
- Critical thinking coach
```

**When to use:** Need specific behavior or output style

### 3. Character Roles

Personality-based personas:

```
You are:
- A patient tutor who explains concepts to beginners
- A meticulous editor who catches every mistake
- An enthusiastic motivational coach
- A skeptical devil's advocate
- A friendly customer service representative
```

**When to use:** Need specific tone or interaction style

### 4. Composite Roles

Combining multiple attributes:

```
You are a senior Python developer AND technical writer with expertise in:
- Writing clean, production-ready code
- Creating clear, comprehensive documentation
- Teaching complex concepts to beginners
- Following PEP 8 and industry best practices
```

**When to use:** Complex tasks requiring multiple skills

---

## 🔧 Role Prompting Techniques

### Technique 1: Simple Role Assignment

**Pattern:**
```
You are a {role}.
{task}
```

**Example:**
```
You are a professional email editor.
Edit the following email for clarity and professionalism.
```

### Technique 2: Detailed Role Definition

**Pattern:**
```
You are a {role} with {qualifications}.
You specialize in {specialty}.
Your approach is {approach}.

{task}
```

**Example:**
```
You are a senior security engineer with CISSP certification.
You specialize in web application security and penetration testing.
Your approach is thorough, methodical, and focused on practical fixes.

Review this code for security vulnerabilities.
```

### Technique 3: Role with Context

**Pattern:**
```
You are a {role} working at {organization}.
Your target audience is {audience}.
Your goal is to {goal}.

{task}
```

**Example:**
```
You are a content strategist working at a fintech startup.
Your target audience is small business owners with no financial background.
Your goal is to educate without overwhelming or using jargon.

Create a blog post explaining investment diversification.
```

### Technique 4: Role with Constraints

**Pattern:**
```
You are a {role}.
You MUST {constraint1}.
You MUST NOT {constraint2}.
You should always {guideline}.

{task}
```

**Example:**
```
You are a customer support agent.
You MUST be empathetic and solution-focused.
You MUST NOT make promises the company can't keep.
You should always acknowledge the customer's frustration first.

Respond to this complaint about delayed shipping.
```

---

## ⚙️ System Prompts

### What are System Prompts?

**System prompts** are persistent instructions that apply to the entire conversation.

```python
# In API calls (OpenAI, Anthropic):
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]
```

### System vs User Prompts

| Aspect | System Prompt | User Prompt |
|--------|--------------|-------------|
| **Persistence** | Applies to all messages | Single interaction |
| **Priority** | Higher weight | Normal weight |
| **Visibility** | Hidden from user | Visible to user |
| **Use Case** | Behavior guidelines | Specific requests |

**C#/.NET Analogy:**
```csharp
// System prompt = Class-level configuration
public class AIAssistant
{
    private readonly string _systemPrompt = "You are helpful.";  // Applies to all methods

    public string Process(string userMessage) { }  // User prompt = method parameter
}
```

### Effective System Prompts

**Pattern:**
```
You are {role}.

Core responsibilities:
- {responsibility1}
- {responsibility2}

Guidelines:
- {guideline1}
- {guideline2}

Constraints:
- {constraint1}
- {constraint2}
```

**Example:**
```
You are a senior software engineering mentor.

Core responsibilities:
- Help developers write better code
- Explain concepts clearly
- Provide actionable feedback

Guidelines:
- Always explain the "why" behind recommendations
- Use code examples when possible
- Be encouraging and constructive

Constraints:
- Do not write code for the user; guide them to solve it
- Do not assume expertise level; ask clarifying questions
- Focus on teaching, not just providing answers
```

---

## 🎯 Role Prompting Best Practices

### 1. Be Specific

❌ **Vague:**
```
You are an expert.
```

✅ **Specific:**
```
You are a PostgreSQL database administrator with 10 years of experience
optimizing queries for high-traffic e-commerce applications.
```

### 2. Include Relevant Context

❌ **No Context:**
```
You are a teacher.
Explain Python functions.
```

✅ **With Context:**
```
You are a computer science teacher instructing high school students
who have basic programming knowledge but are new to functions.
Explain Python functions using simple analogies.
```

### 3. Set Clear Expectations

❌ **Unclear:**
```
You are helpful.
```

✅ **Clear:**
```
You are a helpful teaching assistant who:
- Breaks down complex topics into simple steps
- Uses analogies and real-world examples
- Checks understanding before moving forward
- Encourages questions and exploration
```

### 4. Combine with Other Techniques

```
You are a senior code reviewer.  ← Role

Review the following Python code.  ← Task

Check for:  ← Constraints
- Security vulnerabilities
- Performance issues
- PEP 8 compliance

Return findings as JSON.  ← Format
```

---

## 💡 Real-World Examples

### Example 1: Customer Support

**Scenario:** Automated customer service chatbot

```python
system_prompt = """
You are a customer support representative for TechCorp, a software company.

Your role:
- Resolve customer issues efficiently
- Maintain a friendly, professional tone
- Escalate complex issues to human agents

Guidelines:
- Always start by acknowledging the customer's concern
- Ask clarifying questions before providing solutions
- Provide step-by-step instructions
- Thank the customer for their patience

Constraints:
- Do not make commitments about refunds (escalate to supervisor)
- Do not share internal company information
- Do not argue with customers; stay solution-focused
- Maximum response length: 150 words

Available actions:
- Troubleshooting steps
- Documentation links
- Escalation to human agent
"""
```

### Example 2: Code Review Bot

**Scenario:** Automated pull request reviewer

```python
system_prompt = """
You are a senior software engineer reviewing pull requests.

Expertise:
- Python best practices (PEP 8, typing, testing)
- Security (OWASP Top 10, injection attacks)
- Performance optimization
- Code maintainability

Review criteria:
1. Correctness: Does the code work as intended?
2. Security: Any vulnerabilities?
3. Performance: Any obvious inefficiencies?
4. Readability: Is the code clear and well-documented?
5. Testing: Are there adequate tests?

Output format:
- List issues by severity (CRITICAL, HIGH, MEDIUM, LOW)
- Reference specific line numbers
- Suggest concrete improvements
- Highlight what's done well (positive feedback)

Style:
- Constructive and educational
- Explain WHY, not just WHAT
- Provide code examples for suggestions
"""
```

### Example 3: Educational Tutor

**Scenario:** AI tutor for learning programming

```python
system_prompt = """
You are a patient programming tutor specializing in Python.

Teaching philosophy:
- Socratic method: Guide students to discover answers
- Growth mindset: Mistakes are learning opportunities
- Hands-on: Encourage experimentation

Student level: {beginner/intermediate/advanced}
Current topic: {topic}

Teaching approach:
1. Check current understanding with questions
2. Introduce concepts with simple analogies
3. Show code examples, explain line by line
4. Provide practice exercises
5. Review and reinforce

Constraints:
- Don't just give answers; ask leading questions
- Use simple language, avoid jargon
- Relate to real-world scenarios
- Celebrate progress and effort

If student is stuck:
- Break problem into smaller steps
- Provide hints, not full solutions
- Encourage debugging strategies
"""
```

### Example 4: Data Analyst

**Scenario:** Automated data analysis assistant

```python
system_prompt = """
You are a senior data analyst specializing in business intelligence.

Analysis framework:
1. Understand the business question
2. Examine the data (types, quality, completeness)
3. Apply appropriate statistical methods
4. Generate insights with confidence levels
5. Recommend actions based on findings

Output structure:
- Executive Summary (2-3 sentences)
- Key Findings (top 3-5 insights)
- Statistical Evidence (with confidence levels)
- Visualizations (describe what charts to create)
- Recommendations (specific, actionable)
- Caveats and Limitations

Data quality checks:
- Missing values?
- Outliers?
- Sample size adequate?
- Biases in data collection?

Communication style:
- Clear, concise language
- Numbers with context
- Visual descriptions for non-technical audience
- Highlight business impact, not just statistics
"""
```

---

## 🔄 Role + Other Techniques

### Combining Roles with Few-Shot Learning

```
You are a senior Python developer and code reviewer.

Here are examples of good code reviews:

Example 1:
Code: [bad code]
Review: [constructive feedback]

Example 2:
Code: [another example]
Review: [feedback]

Now review this code:
[code to review]
```

### Combining Roles with Chain-of-Thought

```
You are a logic and reasoning expert.

Think through this problem step by step:
1. Identify the given information
2. Determine what we need to find
3. Consider possible approaches
4. Choose the best approach and explain why
5. Solve step by step
6. Verify the answer

Problem: [problem statement]
```

### Combining Roles with Templates

```python
ROLE_TEMPLATE = """
You are a {role} with expertise in {domain}.

Your task: {task}

Approach:
{approach}

Constraints:
{constraints}

Output format: {format}
"""

# Usage:
prompt = ROLE_TEMPLATE.format(
    role="financial analyst",
    domain="portfolio management",
    task="Evaluate this investment portfolio",
    approach="Risk-adjusted returns, diversification analysis",
    constraints="Focus on long-term growth, risk tolerance: moderate",
    format="Markdown with sections"
)
```

---

## ⚡ Advanced Role Techniques

### 1. Persona Switching

Switch roles mid-conversation for different perspectives:

```
First, as a software engineer, evaluate this architecture.

Now, as a security expert, identify potential vulnerabilities.

Finally, as a cost analyst, estimate infrastructure costs.
```

### 2. Role Hierarchies

Create role authority levels:

```
You are a junior developer seeking code review.
I am a senior architect. Ask me questions about the design.
```

### 3. Multi-Agent Roles

Simulate multiple perspectives:

```
Debate this design decision from three perspectives:

1. As a performance engineer: Focus on speed and efficiency
2. As a security expert: Focus on vulnerabilities and threats
3. As a UX designer: Focus on user experience and usability
```

### 4. Evolving Roles

Roles that change with context:

```python
def get_role(user_level: str) -> str:
    """
    Adapt role based on user expertise.

    C#/.NET: Like strategy pattern
    """
    roles = {
        "beginner": "patient tutor explaining basics",
        "intermediate": "peer reviewer providing detailed feedback",
        "advanced": "consulting expert discussing edge cases"
    }
    return f"You are a {roles[user_level]}."
```

---

## 🧪 Testing Role Effectiveness

### Metrics to Track

1. **Response Quality**
   - Are responses more expert/appropriate?
   - Consistent with role?

2. **User Satisfaction**
   - Do users get better answers?
   - Fewer follow-up questions?

3. **Consistency**
   - Same role = same style?
   - Predictable behavior?

### A/B Testing Roles

```python
# Test different role definitions
role_a = "You are a helpful assistant."
role_b = "You are a senior technical support engineer with 10 years of experience."

# Compare results with same task
task = "Help me debug this error: AttributeError"

# Which role produces better responses?
```

---

## 📊 Common Pitfalls

### ❌ Pitfall 1: Overly Complex Roles

**Bad:**
```
You are a senior principal staff distinguished architect fellow
with PhD in computer science and 30 years of experience across
15 programming languages and 50 frameworks...
```

**Good:**
```
You are a senior software architect specializing in cloud-native applications.
```

### ❌ Pitfall 2: Conflicting Roles

**Bad:**
```
You are a concise summarizer who provides comprehensive, detailed analysis.
```

**Good:**
```
You are a data analyst who provides concise summaries with key insights.
```

### ❌ Pitfall 3: Role Without Purpose

**Bad:**
```
You are a friendly robot.
[Task has nothing to do with friendliness]
```

**Good:**
```
You are a customer service agent.  ← Role matches task
Handle this customer complaint.
```

### ❌ Pitfall 4: Role as Instruction

**Bad:**
```
You are someone who writes code without bugs.
```

**Good:**
```
You are a senior developer who follows best practices.
Write production-ready code with proper error handling.  ← Instruction separate from role
```

---

## ✅ Summary

### Key Takeaways

1. **Roles Shape Behavior**
   - Who the AI "is" affects how it responds
   - More specific = better results

2. **Role Types**
   - Professional (expertise)
   - Functional (behavior)
   - Character (personality)
   - Composite (multiple attributes)

3. **System vs User**
   - System prompts = persistent configuration
   - User prompts = specific requests

4. **Best Practices**
   - Be specific and relevant
   - Include necessary context
   - Set clear expectations
   - Combine with other techniques

5. **Avoid Pitfalls**
   - Not too complex
   - No conflicting attributes
   - Role should match task
   - Separate role from instructions

### C#/.NET Comparison

| Concept | C#/.NET Equivalent |
|---------|-------------------|
| Role | Class inheritance/interface |
| System prompt | Class-level configuration |
| User prompt | Method parameter |
| Role switching | Strategy pattern |
| Composite role | Multiple interface implementation |

---

## 📝 Practice Exercises

1. **Create roles for these tasks:**
   - Email classification
   - Code documentation
   - Data analysis
   - Content creation

2. **Write system prompts for:**
   - Customer service chatbot
   - Code review assistant
   - Educational tutor
   - Data analyst

3. **Compare results:**
   - Same task with different roles
   - Generic vs specific roles
   - Simple vs composite roles

4. **A/B test:**
   - Two role definitions
   - Measure which performs better
   - Iterate and improve

---

## 🔗 Related Concepts

- **Lesson 2:** Few-shot learning (role + examples)
- **Lesson 5:** Chain-of-thought (role + reasoning)
- **Lesson 9:** Security (role-based injection attacks)

---

**Next Lesson:** Lesson 5 - Chain-of-Thought Prompting

**Estimated time:** 75 minutes

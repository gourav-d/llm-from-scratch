# Lesson 3: Prompt Templates

**Learn to create reusable, scalable prompts that work across different scenarios**

---

## 🎯 Learning Objectives

After this lesson, you will be able to:
- Create reusable prompt templates
- Use variables and placeholders effectively
- Build template libraries for common tasks
- Apply DRY principle to prompts
- Scale prompts across your organization

**Time:** 60-90 minutes

---

## 📖 What Are Prompt Templates?

### The Problem

Without templates:
```python
# Every time you need to summarize, you write a new prompt:
prompt1 = "Summarize this article about AI..."
prompt2 = "Summarize this article about climate..."
prompt3 = "Summarize this article about economics..."
# ❌ Repetitive, inconsistent, hard to maintain
```

### The Solution

With templates:
```python
# One template, infinite uses:
template = """
You are a professional editor.
Summarize the following article about {topic} in exactly {num_sentences} sentences.
Focus on {focus_area}.

Article: {article_text}
"""

# Reuse with different inputs:
prompt1 = template.format(topic="AI", num_sentences=3, focus_area="key findings", article_text=article1)
prompt2 = template.format(topic="climate", num_sentences=5, focus_area="policy implications", article_text=article2)
# ✅ Consistent, maintainable, scalable
```

---

## 🔑 Why Templates Matter

### 1. **Consistency**
- Same structure every time
- Predictable results
- Easier to debug

### 2. **Maintainability**
- Fix once, apply everywhere
- Version control your prompts
- Team collaboration

### 3. **Scalability**
- Process thousands of inputs
- A/B test prompt variations
- Build prompt libraries

### 4. **Cost Efficiency**
- Test once, reuse forever
- No manual prompt writing
- Reduced token usage through optimization

---

## 🏗️ Template Structure

### Basic Template Anatomy

```python
template = """
{system_role}                    # 1. WHO is the AI?

{task_description}               # 2. WHAT to do?

{constraints}                    # 3. RULES and limits

{format_specification}           # 4. OUTPUT structure

{input_placeholder}              # 5. USER data
"""
```

### C#/.NET Comparison

Templates are like **string interpolation** in C#:

```csharp
// C# string interpolation
var greeting = $"Hello, {name}! You have {count} messages.";

// Python f-string (similar concept)
greeting = f"Hello, {name}! You have {count} messages."

// Prompt template (same idea, larger scale)
template = "You are {role}. Analyze {data} and provide {output}."
```

---

## 💻 Template Types

### 1. **Simple String Templates** (Python str.format)

```python
template = "You are a {role}. {task}"

# Usage
prompt = template.format(
    role="data analyst",
    task="Analyze sales trends"
)
```

**Pros:** Simple, built-in, no dependencies
**Cons:** Basic, limited features

### 2. **F-String Templates** (Python 3.6+)

```python
role = "teacher"
subject = "math"

prompt = f"""
You are a {role} specializing in {subject}.
Create a lesson plan for topic: {topic}.
"""
```

**Pros:** Clean syntax, fast
**Cons:** Variables must be in scope, less reusable

### 3. **Template Classes** (Python Template)

```python
from string import Template

template = Template("""
You are a $role specializing in $domain.
Analyze: $input
""")

# Usage
prompt = template.substitute(
    role="engineer",
    domain="security",
    input=user_data
)
```

**Pros:** Safe substitution, prevents injection
**Cons:** Different syntax ($var instead of {var})

### 4. **LangChain PromptTemplate** (Recommended)

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["role", "task", "constraints"],
    template="""
You are a {role}.

Task: {task}

Constraints:
{constraints}

Provide your analysis.
"""
)

# Usage with validation
prompt = template.format(
    role="financial advisor",
    task="Evaluate investment portfolio",
    constraints="Risk level: Conservative\nTime horizon: 10 years"
)
```

**Pros:** Validation, composition, LangChain integration
**Cons:** Requires installation

---

## 📝 Template Best Practices

### 1. **Use Descriptive Variable Names**

❌ **Bad:**
```python
template = "Analyze {x} and return {y}"
```

✅ **Good:**
```python
template = "Analyze {input_data} and return {output_format}"
```

### 2. **Provide Defaults**

```python
def create_summary_prompt(
    text: str,
    max_sentences: int = 3,
    tone: str = "professional"
) -> str:
    """
    C#/.NET developers: This is like optional parameters!

    In C#:
    string CreateSummaryPrompt(
        string text,
        int maxSentences = 3,
        string tone = "professional"
    )
    """
    template = """
Summarize in exactly {max_sentences} sentences.
Tone: {tone}

Text: {text}
"""
    return template.format(
        text=text,
        max_sentences=max_sentences,
        tone=tone
    )
```

### 3. **Validate Inputs**

```python
def create_classification_prompt(category: str, text: str) -> str:
    """
    Validate inputs before creating prompt.

    Like C# data validation attributes:
    [Required]
    [StringLength(100)]
    """
    valid_categories = ["spam", "important", "social", "promotions"]

    if category not in valid_categories:
        raise ValueError(f"Invalid category. Must be one of: {valid_categories}")

    if not text or len(text) < 10:
        raise ValueError("Text must be at least 10 characters")

    template = """
Classify this email as: {category}
Explain your reasoning.

Email: {text}
"""
    return template.format(category=category, text=text)
```

### 4. **Document Your Templates**

```python
class EmailTemplates:
    """
    Collection of email-related prompt templates.

    Like C# XML documentation:
    /// <summary>
    /// Collection of email-related prompt templates
    /// </summary>
    """

    @staticmethod
    def classify_email(email_text: str, categories: List[str]) -> str:
        """
        Create a prompt to classify emails.

        Args:
            email_text: The email content to classify
            categories: Valid categories (e.g., ["spam", "important"])

        Returns:
            Formatted prompt string

        Example:
            >>> prompt = EmailTemplates.classify_email(
            ...     "Get rich quick!",
            ...     ["spam", "legitimate"]
            ... )
        """
        template = """
You are an email classification expert.

Categories: {categories}
Classify this email into exactly one category.

Email: {email_text}

Return JSON: {{"category": "spam", "confidence": 0.95}}
"""
        return template.format(
            categories=", ".join(categories),
            email_text=email_text
        )
```

---

## 🏭 Building a Template Library

### Example: Analysis Templates

```python
class AnalysisTemplates:
    """
    Reusable templates for data analysis tasks.

    C#/.NET equivalent: Static helper class
    public static class AnalysisTemplates
    """

    # Template 1: SWOT Analysis
    SWOT_ANALYSIS = """
You are a business strategy consultant.

Perform a SWOT analysis of:
{subject}

Context:
{context}

Return as JSON:
{{
  "strengths": ["list"],
  "weaknesses": ["list"],
  "opportunities": ["list"],
  "threats": ["list"]
}}
"""

    # Template 2: Trend Analysis
    TREND_ANALYSIS = """
You are a data analyst specializing in {domain}.

Analyze trends in the following data:
{data}

Time period: {time_period}

Provide:
1. Key trends (3-5)
2. Statistical significance
3. Predictions for next {forecast_period}
4. Confidence levels

Format: {output_format}
"""

    # Template 3: Comparative Analysis
    COMPARATIVE_ANALYSIS = """
You are an analytical expert.

Compare and contrast:
A: {option_a}
B: {option_b}

Criteria:
{criteria}

Provide:
- Side-by-side comparison table
- Pros and cons for each
- Recommendation with justification

Format as Markdown table.
"""

    @classmethod
    def swot(cls, subject: str, context: str = "") -> str:
        """Generate SWOT analysis prompt."""
        return cls.SWOT_ANALYSIS.format(
            subject=subject,
            context=context or "General business context"
        )

    @classmethod
    def trend(cls,
              data: str,
              domain: str = "business",
              time_period: str = "last quarter",
              forecast_period: str = "3 months",
              output_format: str = "Markdown") -> str:
        """Generate trend analysis prompt."""
        return cls.TREND_ANALYSIS.format(
            domain=domain,
            data=data,
            time_period=time_period,
            forecast_period=forecast_period,
            output_format=output_format
        )
```

### Usage:

```python
# Easy to use across your codebase
prompt1 = AnalysisTemplates.swot(
    subject="Our mobile app",
    context="Launching in competitive market"
)

prompt2 = AnalysisTemplates.trend(
    data=sales_data,
    domain="retail sales",
    time_period="Q4 2025"
)
```

---

## 🔧 Advanced Template Techniques

### 1. **Template Composition**

Combine smaller templates into larger ones:

```python
class PromptComponents:
    """
    Modular prompt components.

    Like LEGO blocks - combine to build complex prompts!
    """

    # Component 1: Role definitions
    ROLES = {
        "analyst": "You are a senior data analyst with 10 years of experience.",
        "teacher": "You are an expert teacher skilled at explaining complex topics.",
        "reviewer": "You are a meticulous code reviewer focused on quality.",
    }

    # Component 2: Constraints
    CONSTRAINTS = {
        "concise": "Keep response under 100 words.",
        "detailed": "Provide comprehensive explanation with examples.",
        "structured": "Use numbered lists and clear sections.",
    }

    # Component 3: Output formats
    FORMATS = {
        "json": "Return as valid JSON only, no additional text.",
        "markdown": "Format as Markdown with headers and lists.",
        "bullet_points": "Use bullet points for each item.",
    }

    @classmethod
    def build(cls, role: str, task: str, constraint: str, format: str) -> str:
        """
        Compose a prompt from components.

        Like C# method chaining or builder pattern!
        """
        return f"""
{cls.ROLES[role]}

Task: {task}

Constraints:
- {cls.CONSTRAINTS[constraint]}
- {cls.FORMATS[format]}

Provide your response below.
"""

# Usage:
prompt = PromptComponents.build(
    role="analyst",
    task="Analyze sales trends",
    constraint="detailed",
    format="markdown"
)
```

### 2. **Conditional Templates**

Templates that adapt based on input:

```python
def create_adaptive_prompt(
    task: str,
    complexity: str = "medium",
    include_examples: bool = False
) -> str:
    """
    Template adapts based on parameters.

    C#/.NET: Like conditional compilation with #if directives
    """

    # Base template
    template = "Perform the following task:\n{task}\n\n"

    # Add complexity-specific instructions
    if complexity == "simple":
        template += "Provide a brief, straightforward answer.\n"
    elif complexity == "medium":
        template += "Provide a balanced explanation with key details.\n"
    else:  # complex
        template += "Provide comprehensive analysis with multiple perspectives.\n"

    # Optionally include examples
    if include_examples:
        template += "\nInclude 2-3 concrete examples.\n"

    template += "\nConstraints:\n- Be accurate\n- Stay focused\n"

    return template.format(task=task)
```

### 3. **Template Inheritance**

```python
class BasePrompt:
    """Base template with common structure."""

    BASE_TEMPLATE = """
{role}

Task: {task}

{specific_instructions}

Constraints:
{constraints}

Output format: {format}
"""

    def render(self, **kwargs) -> str:
        """Render the template with provided values."""
        return self.BASE_TEMPLATE.format(**kwargs)


class EmailPrompt(BasePrompt):
    """Specialized prompt for email tasks."""

    def classify(self, email: str, categories: List[str]) -> str:
        """Email classification prompt."""
        return self.render(
            role="You are an email classification expert.",
            task="Classify the email below into one category.",
            specific_instructions=f"Valid categories: {', '.join(categories)}",
            constraints="- Choose exactly one category\n- Provide confidence score",
            format="JSON with category and confidence"
        ) + f"\n\nEmail:\n{email}"


class CodePrompt(BasePrompt):
    """Specialized prompt for code tasks."""

    def review(self, code: str, language: str) -> str:
        """Code review prompt."""
        return self.render(
            role=f"You are a senior {language} developer and code reviewer.",
            task="Review the following code for issues.",
            specific_instructions="Check for: bugs, security issues, best practices",
            constraints="- Provide specific line numbers\n- Suggest improvements",
            format="Markdown with sections for each issue"
        ) + f"\n\n```{language}\n{code}\n```"
```

---

## 📊 Template Performance

### Measuring Template Quality

```python
class TemplateMetrics:
    """
    Track template performance.

    C#/.NET: Like performance counters or Application Insights
    """

    def __init__(self):
        self.usage_count = {}
        self.success_rate = {}
        self.avg_response_quality = {}

    def record_usage(self, template_name: str, success: bool, quality_score: float):
        """Record template usage and results."""
        if template_name not in self.usage_count:
            self.usage_count[template_name] = 0
            self.success_rate[template_name] = []
            self.avg_response_quality[template_name] = []

        self.usage_count[template_name] += 1
        self.success_rate[template_name].append(1 if success else 0)
        self.avg_response_quality[template_name].append(quality_score)

    def get_best_template(self) -> str:
        """
        Find the best performing template.

        Like C# LINQ:
        var bestTemplate = templates.OrderByDescending(t => t.Quality).First();
        """
        best_template = None
        best_score = 0

        for template_name in self.usage_count:
            avg_quality = sum(self.avg_response_quality[template_name]) / len(self.avg_response_quality[template_name])
            if avg_quality > best_score:
                best_score = avg_quality
                best_template = template_name

        return best_template
```

---

## 🎯 Real-World Examples

### Example 1: Customer Support Template

```python
class SupportTemplates:
    """Customer support prompt templates."""

    RESPONSE_TEMPLATE = """
You are a helpful customer support agent for {company}.

Customer issue:
{customer_message}

Context:
- Customer tier: {customer_tier}
- Previous issues: {previous_issues}
- Product: {product}

Provide a response that:
1. Acknowledges the issue
2. Provides a solution or next steps
3. Maintains a {tone} tone
4. Includes relevant documentation links if available

Keep response under {max_words} words.
"""

    @classmethod
    def create_response(cls,
                       company: str,
                       customer_message: str,
                       customer_tier: str = "standard",
                       previous_issues: int = 0,
                       product: str = "main product",
                       tone: str = "friendly and professional",
                       max_words: int = 150) -> str:
        return cls.RESPONSE_TEMPLATE.format(
            company=company,
            customer_message=customer_message,
            customer_tier=customer_tier,
            previous_issues=f"{previous_issues} previous tickets" if previous_issues > 0 else "No previous issues",
            product=product,
            tone=tone,
            max_words=max_words
        )
```

### Example 2: Content Generation Template

```python
class ContentTemplates:
    """Content creation templates."""

    BLOG_POST_TEMPLATE = """
You are a content writer for {blog_name}, targeting {target_audience}.

Create a blog post about: {topic}

Requirements:
- Length: {word_count} words
- Tone: {tone}
- Include: {required_elements}
- SEO keywords: {keywords}
- Structure: {structure}

Brand voice guidelines:
{brand_voice}

Output format: Markdown
"""

    @classmethod
    def blog_post(cls,
                  topic: str,
                  blog_name: str = "Tech Insights",
                  target_audience: str = "software developers",
                  word_count: int = 800,
                  tone: str = "informative and engaging",
                  keywords: str = "",
                  structure: str = "Introduction, Main Points, Conclusion") -> str:

        required_elements = "statistics, examples, actionable takeaways"
        brand_voice = "Clear, technical but accessible, focus on practical value"

        return cls.BLOG_POST_TEMPLATE.format(
            blog_name=blog_name,
            target_audience=target_audience,
            topic=topic,
            word_count=word_count,
            tone=tone,
            required_elements=required_elements,
            keywords=keywords or "Not specified",
            structure=structure,
            brand_voice=brand_voice
        )
```

---

## 🚀 Production Tips

### 1. **Version Your Templates**

```python
class TemplateRegistry:
    """
    Centralized template registry with versioning.

    C#/.NET: Like a dependency injection container
    """

    templates = {
        "email_classifier_v1": "...",
        "email_classifier_v2": "...",  # Improved version
        "summarizer_v1": "...",
    }

    @classmethod
    def get(cls, template_name: str, version: int = None) -> str:
        """Get template by name and version."""
        if version:
            key = f"{template_name}_v{version}"
        else:
            # Get latest version
            versions = [k for k in cls.templates if k.startswith(template_name)]
            key = max(versions) if versions else template_name

        return cls.templates.get(key, "Template not found")
```

### 2. **A/B Test Templates**

```python
import random

def get_template_variant(base_template: str, variants: List[str]) -> str:
    """
    Randomly select template variant for A/B testing.

    C#/.NET: Like feature flags or experimentation frameworks
    """
    all_templates = [base_template] + variants
    return random.choice(all_templates)

# Usage:
variant_a = "Summarize in 3 sentences..."
variant_b = "Create a brief 3-sentence summary..."
variant_c = "Provide a concise 3-sentence overview..."

template = get_template_variant(variant_a, [variant_b, variant_c])
```

### 3. **Cache Rendered Templates**

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_prompt(template_name: str, **kwargs) -> str:
    """
    Cache frequently used prompts.

    C#/.NET: Like MemoryCache or distributed caching
    """
    template = TemplateRegistry.get(template_name)
    # Sort kwargs to make cache key consistent
    sorted_kwargs = tuple(sorted(kwargs.items()))
    return template.format(**dict(sorted_kwargs))
```

---

## ✅ Summary

### Key Takeaways

1. **Templates = Reusable Prompts**
   - Write once, use many times
   - Consistent results
   - Easy to maintain

2. **Template Types**
   - Simple: str.format(), f-strings
   - Advanced: LangChain, custom classes

3. **Best Practices**
   - Descriptive variable names
   - Provide defaults
   - Validate inputs
   - Document thoroughly

4. **Build Libraries**
   - Organize by domain (email, code, analysis)
   - Use classes for structure
   - Version your templates

5. **Production Ready**
   - Version control
   - A/B testing
   - Caching
   - Metrics

### C#/.NET Comparison

| Python Concept | C#/.NET Equivalent |
|----------------|-------------------|
| `str.format()` | `string.Format()` |
| f-strings | `$"string interpolation"` |
| Template class | Custom template engine |
| `@lru_cache` | `MemoryCache` |
| Class methods | Static methods |
| kwargs | Named parameters |

---

## 📝 Practice Exercises

1. **Create a Template Library**
   - Build 5 templates for your domain
   - Organize into a class
   - Add documentation

2. **Template Composition**
   - Create modular components
   - Build complex prompts from simple parts
   - Test different combinations

3. **A/B Testing**
   - Write 3 variations of one template
   - Test with same input
   - Compare results

4. **Production Template**
   - Add versioning
   - Add caching
   - Add metrics
   - Add validation

---

## 🔗 Related Concepts

- **Module 1:** Python basics (string formatting)
- **Module 9:** RAG (using templates with retrieved data)
- **Module 10:** LangChain (advanced template composition)

---

## 📚 Additional Resources

- LangChain PromptTemplate docs
- Python string formatting guide
- Template design patterns
- Best practices for production AI

---

**Next Lesson:** Lesson 4 - Role & System Prompting

**Estimated time:** 60 minutes

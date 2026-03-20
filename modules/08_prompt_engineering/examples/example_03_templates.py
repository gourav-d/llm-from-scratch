"""
Example 3: Prompt Templates in Action
=====================================

This example demonstrates how to create and use reusable prompt templates.

LEARNING OBJECTIVES:
- Create template classes for different tasks
- Use template composition
- Build a template library
- Compare template approaches

WHAT YOU'LL SEE:
1. Simple string templates
2. LangChain templates
3. Custom template classes
4. Template library organization
5. Real-world template examples

C#/.NET PERSPECTIVE:
Templates are like string.Format() or $"interpolation" but at scale.
We're building reusable components like you would with C# classes!

RUN: python example_03_templates.py
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*70)
print("EXAMPLE 3: Prompt Templates - Building Reusable Prompts")
print("="*70)

# ============================================================================
# SECTION 1: Simple String Templates
# ============================================================================

print("\n" + "="*70)
print("SECTION 1: Simple String Templates (str.format)")
print("="*70)

print("""
The simplest way to create templates is using Python's built-in str.format().

In C#:
string template = "Hello, {0}! You have {1} messages.";
string result = string.Format(template, name, count);

In Python:
template = "Hello, {name}! You have {count} messages."
result = template.format(name=name, count=count)
""")

# Example 1.1: Basic email classification template
email_classification_template = """
You are an email classification expert.

Task: Classify the following email into exactly one category.

Categories: {categories}

Email:
{email_text}

Return JSON: {{"category": "...", "confidence": 0.95}}
"""

print("\n📧 Email Classification Template:")
print(email_classification_template)

# Use the template
categories = "spam, important, social, promotions"
email = "LIMITED TIME OFFER! Get 50% off now! Click here!!!"

prompt = email_classification_template.format(
    categories=categories,
    email_text=email
)

print("\n✅ Rendered Prompt:")
print("-" * 70)
print(prompt)

# ============================================================================
# SECTION 2: F-String Templates
# ============================================================================

print("\n" + "="*70)
print("SECTION 2: F-String Templates (Modern Python)")
print("="*70)

print("""
F-strings (Python 3.6+) provide clean syntax for templates.

In C#:
var greeting = $"Hello, {name}!";  // String interpolation

In Python:
greeting = f"Hello, {name}!"  // F-string
""")

# Example 2.1: Dynamic code review template
def create_code_review_prompt(code: str, language: str, focus: str = "all") -> str:
    """
    Create code review prompt using f-strings.

    Args:
        code: Code to review
        language: Programming language
        focus: What to focus on (security, performance, readability, all)

    Returns:
        Formatted prompt string
    """
    focus_areas = {
        "security": "security vulnerabilities and potential exploits",
        "performance": "performance issues and optimization opportunities",
        "readability": "code clarity and maintainability",
        "all": "bugs, security, performance, and best practices"
    }

    focus_description = focus_areas.get(focus, focus_areas["all"])

    # F-string template
    prompt = f"""
You are a senior {language} developer and code reviewer.

Review the following code for {focus_description}.

Provide:
1. Issues found (with severity: HIGH, MEDIUM, LOW)
2. Specific line numbers
3. Suggested improvements
4. Explanation of why changes are needed

Code:
```{language}
{code}
```

Format as Markdown.
"""
    return prompt


# Test the f-string template
sample_code = """
def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
"""

prompt = create_code_review_prompt(sample_code, "python", "performance")
print("\n🔍 Code Review Template (F-String):")
print("-" * 70)
print(prompt)

# ============================================================================
# SECTION 3: Template Classes
# ============================================================================

print("\n" + "="*70)
print("SECTION 3: Template Classes (Best Practice)")
print("="*70)

print("""
Template classes provide structure, validation, and reusability.

This is like creating a C# static helper class:
public static class EmailTemplates
{
    public static string Classify(string email, List<string> categories) { }
}
""")


class EmailTemplates:
    """
    Collection of email-related prompt templates.

    C#/.NET equivalent:
    public static class EmailTemplates { }
    """

    # Template constants
    CLASSIFICATION_TEMPLATE = """
You are an email classification expert specialized in {domain}.

Task: Classify the following email into exactly one category.

Valid categories:
{categories}

Classification criteria:
{criteria}

Email:
---
{email_text}
---

Return as JSON:
{{
  "category": "category_name",
  "confidence": 0.95,
  "reasoning": "Brief explanation"
}}
"""

    SUMMARY_TEMPLATE = """
You are a professional email assistant.

Summarize the following email in exactly {num_sentences} sentences.

Focus on:
- Main purpose
- Action items
- Deadlines or urgency

Email:
{email_text}

Format: {num_sentences} numbered sentences.
"""

    RESPONSE_TEMPLATE = """
You are a {role} responding to an email.

Original email:
{original_email}

Context:
{context}

Compose a response that:
1. {requirement_1}
2. {requirement_2}
3. {requirement_3}

Tone: {tone}
Max length: {max_words} words

Draft response:
"""

    @classmethod
    def classify(cls,
                email_text: str,
                categories: List[str],
                domain: str = "general business",
                criteria: str = None) -> str:
        """
        Create email classification prompt.

        Args:
            email_text: Email content to classify
            categories: List of valid categories
            domain: Domain expertise (e.g., "customer support", "sales")
            criteria: Custom classification criteria

        Returns:
            Formatted prompt string

        Example:
            >>> prompt = EmailTemplates.classify(
            ...     "Get rich quick!",
            ...     ["spam", "legitimate"]
            ... )
        """
        if not criteria:
            criteria = "\n".join([f"- {cat}: Emails related to {cat}" for cat in categories])

        return cls.CLASSIFICATION_TEMPLATE.format(
            domain=domain,
            categories="\n".join([f"- {cat}" for cat in categories]),
            criteria=criteria,
            email_text=email_text
        )

    @classmethod
    def summarize(cls, email_text: str, num_sentences: int = 3) -> str:
        """Create email summary prompt."""
        return cls.SUMMARY_TEMPLATE.format(
            email_text=email_text,
            num_sentences=num_sentences
        )

    @classmethod
    def respond(cls,
                original_email: str,
                role: str = "professional email correspondent",
                context: str = "",
                tone: str = "professional and friendly",
                max_words: int = 150) -> str:
        """Create email response prompt."""
        return cls.RESPONSE_TEMPLATE.format(
            role=role,
            original_email=original_email,
            context=context or "No additional context",
            requirement_1="Address all points raised",
            requirement_2="Provide clear next steps",
            requirement_3="Maintain appropriate tone",
            tone=tone,
            max_words=max_words
        )


# Test template class
print("\n📧 Using EmailTemplates Class:")
print("-" * 70)

spam_email = "URGENT! You've won $1,000,000! Click here now!!!"
categories = ["spam", "important", "informational"]

prompt = EmailTemplates.classify(
    email_text=spam_email,
    categories=categories,
    domain="email security"
)

print(prompt)

# ============================================================================
# SECTION 4: Template Library (Real-World)
# ============================================================================

print("\n" + "="*70)
print("SECTION 4: Complete Template Library")
print("="*70)

print("""
A production-ready template library with multiple domains.

In C#, this would be organized like:
namespace PromptTemplates
{
    public static class AnalysisTemplates { }
    public static class ContentTemplates { }
    public static class CodeTemplates { }
}
""")


class AnalysisTemplates:
    """Data analysis prompt templates."""

    SWOT_ANALYSIS = """
You are a business strategy consultant with expertise in {industry}.

Perform a comprehensive SWOT analysis of:
{subject}

Context and background:
{context}

Consider:
- Internal factors (Strengths, Weaknesses)
- External factors (Opportunities, Threats)
- Market conditions
- Competitive landscape

Return as structured JSON:
{{
  "strengths": ["strength 1", "strength 2", ...],
  "weaknesses": ["weakness 1", ...],
  "opportunities": ["opportunity 1", ...],
  "threats": ["threat 1", ...],
  "summary": "Overall assessment",
  "recommendations": ["action 1", "action 2", ...]
}}
"""

    TREND_ANALYSIS = """
You are a {domain} analyst specializing in trend identification.

Analyze the following data for trends:

{data}

Time period: {time_period}
Analysis depth: {depth}

Provide:
1. **Key Trends** (3-5 major patterns)
   - Trend description
   - Statistical significance
   - Impact assessment

2. **Predictions** for next {forecast_period}
   - Expected outcomes
   - Confidence levels (%)
   - Supporting evidence

3. **Recommendations**
   - Actionable steps
   - Priority levels
   - Expected ROI

Format: {output_format}
"""

    @classmethod
    def swot(cls,
            subject: str,
            industry: str = "general business",
            context: str = "") -> str:
        """Generate SWOT analysis prompt."""
        return cls.SWOT_ANALYSIS.format(
            industry=industry,
            subject=subject,
            context=context or "No additional context provided"
        )

    @classmethod
    def trends(cls,
              data: str,
              domain: str = "business",
              time_period: str = "past 12 months",
              depth: str = "comprehensive",
              forecast_period: str = "6 months",
              output_format: str = "Markdown") -> str:
        """Generate trend analysis prompt."""
        return cls.TREND_ANALYSIS.format(
            domain=domain,
            data=data,
            time_period=time_period,
            depth=depth,
            forecast_period=forecast_period,
            output_format=output_format
        )


class ContentTemplates:
    """Content generation templates."""

    BLOG_POST = """
You are a content writer for {publication}.

Target audience: {audience}
Content goal: {goal}

Create a blog post about:
{topic}

Requirements:
- Length: {word_count} words (+/- 10%)
- Tone: {tone}
- Include: {required_elements}
- SEO keywords: {keywords}

Structure:
{structure}

Brand voice:
{brand_voice}

Output as Markdown with proper headers.
"""

    SOCIAL_MEDIA = """
You are a social media manager for {brand}.

Platform: {platform}
Campaign: {campaign}

Create {num_posts} posts about:
{topic}

Guidelines:
- Character limit: {char_limit}
- Include hashtags: {use_hashtags}
- Call-to-action: {cta}
- Tone: {tone}

Return as JSON array of posts.
"""

    @classmethod
    def blog_post(cls,
                 topic: str,
                 publication: str = "Tech Blog",
                 audience: str = "developers",
                 goal: str = "educate and engage",
                 word_count: int = 800,
                 tone: str = "professional yet conversational",
                 keywords: str = "") -> str:
        """Generate blog post creation prompt."""

        structure = """
1. **Introduction** (10%)
   - Hook
   - Context
   - Preview of main points

2. **Main Content** (70%)
   - Key point 1 with examples
   - Key point 2 with examples
   - Key point 3 with examples

3. **Conclusion** (20%)
   - Summary
   - Call to action
   - Next steps
"""

        return cls.BLOG_POST.format(
            publication=publication,
            audience=audience,
            goal=goal,
            topic=topic,
            word_count=word_count,
            tone=tone,
            required_elements="statistics, examples, actionable takeaways",
            keywords=keywords or "Not specified",
            structure=structure,
            brand_voice="Clear, technical but accessible, practical value"
        )


# Test the template library
print("\n📊 SWOT Analysis Template Example:")
print("-" * 70)

swot_prompt = AnalysisTemplates.swot(
    subject="Our new AI-powered chatbot product",
    industry="SaaS technology",
    context="Launching in competitive market with established players"
)

print(swot_prompt[:500] + "...\n")

print("\n📝 Blog Post Template Example:")
print("-" * 70)

blog_prompt = ContentTemplates.blog_post(
    topic="Getting Started with Prompt Engineering",
    publication="AI Developers Blog",
    audience="software developers new to AI",
    word_count=1200
)

print(blog_prompt[:500] + "...\n")

# ============================================================================
# SECTION 5: Template Composition
# ============================================================================

print("\n" + "="*70)
print("SECTION 5: Template Composition (Modular Building)")
print("="*70)

print("""
Build complex prompts from smaller, reusable components.

Like LEGO blocks or C# extension methods:
public static class PromptBuilder
{
    public static Prompt WithRole(this Prompt p, string role) { }
    public static Prompt WithConstraints(this Prompt p, string constraints) { }
}
""")


class PromptComponents:
    """Modular prompt components for composition."""

    # Reusable roles
    ROLES = {
        "analyst": "You are a senior data analyst with 10 years of experience in statistical analysis and business intelligence.",
        "teacher": "You are an expert teacher skilled at breaking down complex topics into simple, understandable concepts.",
        "reviewer": "You are a meticulous code reviewer focused on security, performance, and maintainability.",
        "writer": "You are a professional technical writer with expertise in creating clear, engaging documentation.",
        "consultant": "You are a business consultant specializing in strategy and operational improvement.",
    }

    # Reusable constraints
    CONSTRAINTS = {
        "concise": "- Keep response under 100 words\n- Focus on key points only\n- No unnecessary details",
        "detailed": "- Provide comprehensive explanation\n- Include examples and evidence\n- Cover edge cases",
        "structured": "- Use numbered lists\n- Clear sections with headers\n- Logical flow",
        "technical": "- Use precise terminology\n- Include technical details\n- Reference standards/best practices",
    }

    # Reusable formats
    FORMATS = {
        "json": "Return ONLY valid JSON, no additional text before or after.",
        "markdown": "Format as Markdown with proper headers (##), lists, and code blocks.",
        "bullet_points": "Use bullet points (•) for each main item.",
        "numbered_list": "Use numbered list (1., 2., 3.) for sequential items.",
    }

    @classmethod
    def build(cls,
             role_key: str,
             task: str,
             constraint_keys: List[str],
             format_key: str,
             additional_context: str = "") -> str:
        """
        Compose prompt from components.

        Args:
            role_key: Key from ROLES dict
            task: Main task description
            constraint_keys: List of keys from CONSTRAINTS dict
            format_key: Key from FORMATS dict
            additional_context: Any additional context

        Returns:
            Composed prompt

        Example:
            >>> prompt = PromptComponents.build(
            ...     role_key="analyst",
            ...     task="Analyze sales trends",
            ...     constraint_keys=["detailed", "structured"],
            ...     format_key="markdown"
            ... )
        """
        # Build constraints section
        constraints = "\n".join([cls.CONSTRAINTS[key] for key in constraint_keys])

        # Compose final prompt
        prompt = f"""{cls.ROLES[role_key]}

Task:
{task}

Constraints:
{constraints}

{additional_context}

Output format:
{cls.FORMATS[format_key]}

Provide your response below.
"""
        return prompt


# Test composition
print("\n🔨 Composed Prompt Example:")
print("-" * 70)

composed_prompt = PromptComponents.build(
    role_key="analyst",
    task="Analyze the quarterly sales data and identify key trends",
    constraint_keys=["detailed", "structured"],
    format_key="markdown",
    additional_context="Focus on year-over-year growth and regional performance."
)

print(composed_prompt)

# ============================================================================
# SECTION 6: Template Validation
# ============================================================================

print("\n" + "="*70)
print("SECTION 6: Template Validation")
print("="*70)

print("""
Validate template inputs before using them.

In C#, this is like data annotations:
[Required]
[StringLength(100, MinimumLength = 10)]
[RegularExpression(@"^[a-zA-Z ]+$")]
""")


class ValidatedTemplate:
    """Template with input validation."""

    @staticmethod
    def validate_non_empty(value: str, field_name: str):
        """Validate string is not empty."""
        if not value or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")

    @staticmethod
    def validate_length(value: str, field_name: str, min_len: int, max_len: int):
        """Validate string length."""
        length = len(value)
        if length < min_len:
            raise ValueError(f"{field_name} must be at least {min_len} characters (got {length})")
        if length > max_len:
            raise ValueError(f"{field_name} must be at most {max_len} characters (got {length})")

    @staticmethod
    def validate_choice(value: str, field_name: str, valid_choices: List[str]):
        """Validate value is in allowed choices."""
        if value not in valid_choices:
            raise ValueError(f"{field_name} must be one of: {', '.join(valid_choices)}")

    @classmethod
    def create_classification_prompt(cls,
                                     text: str,
                                     categories: List[str],
                                     domain: str = "general") -> str:
        """
        Create validated classification prompt.

        Raises:
            ValueError: If validation fails
        """
        # Validate inputs
        cls.validate_non_empty(text, "text")
        cls.validate_length(text, "text", min_len=10, max_len=5000)

        if not categories or len(categories) < 2:
            raise ValueError("Must provide at least 2 categories")

        valid_domains = ["general", "email", "customer_support", "content"]
        cls.validate_choice(domain, "domain", valid_domains)

        # Build template
        template = f"""
You are a {domain} classification expert.

Categories: {', '.join(categories)}

Classify the following text into exactly one category.

Text:
{text}

Return JSON: {{"category": "...", "confidence": 0.95}}
"""
        return template


# Test validation
print("\n✅ Testing Template Validation:")
print("-" * 70)

try:
    # Valid input
    prompt = ValidatedTemplate.create_classification_prompt(
        text="This is a sample text for classification with sufficient length.",
        categories=["positive", "negative", "neutral"],
        domain="general"
    )
    print("✅ Valid input accepted")
    print(prompt[:200] + "...\n")

    # Invalid input (too short)
    prompt = ValidatedTemplate.create_classification_prompt(
        text="Too short",
        categories=["cat1", "cat2"]
    )
except ValueError as e:
    print(f"❌ Validation caught error: {e}\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: Template Best Practices")
print("="*70)

summary = """
✅ KEY TAKEAWAYS:

1. **Start Simple**
   - Use str.format() or f-strings for basic templates
   - Add complexity only when needed

2. **Organize into Classes**
   - Group related templates
   - Use class methods for clean API
   - Like C# static helper classes

3. **Validate Inputs**
   - Check for required fields
   - Validate data types and ranges
   - Provide clear error messages

4. **Compose When Possible**
   - Build complex from simple
   - Reuse components
   - Maintain single responsibility

5. **Production Ready**
   - Version your templates
   - Track performance
   - A/B test variations
   - Cache when appropriate

📊 COMPARISON: Template Approaches

| Approach | Use Case | Pros | Cons |
|----------|----------|------|------|
| str.format() | Simple, one-off | Easy, built-in | Basic features |
| f-strings | Variables in scope | Clean syntax | Less reusable |
| Template classes | Production code | Structure, validation | More code |
| Composition | Complex prompts | Modular, flexible | Can be complex |

🎯 NEXT STEPS:
1. Build your own template library for your domain
2. Practice composition for complex prompts
3. Add validation to prevent errors
4. Move to Lesson 4: Role & System Prompting
"""

print(summary)

print("\n" + "="*70)
print("END OF EXAMPLE 3")
print("="*70)
print("\nYou now know how to build production-ready prompt templates!")
print("Practice by creating templates for your own use cases.")
print("\nNext: example_04_roles.py - Role and System Prompting")

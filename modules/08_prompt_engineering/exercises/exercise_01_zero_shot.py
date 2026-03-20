"""
Exercise 1: Zero-Shot Prompting Practice
=========================================

This exercise helps you practice writing effective zero-shot prompts.

LEARNING OBJECTIVES:
- Apply the 4-component prompt structure (Role, Task, Constraints, Format)
- Practice writing clear, specific prompts
- Compare prompt quality and results
- Understand the impact of each component

SETUP:
1. Install dependencies: pip install openai anthropic python-dotenv
2. Set up your .env file with API keys:
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here

INSTRUCTIONS:
- Complete each exercise function
- Run the tests to check your work
- Compare your prompts with the sample solutions
- Experiment and iterate!

C#/.NET PERSPECTIVE:
This is like writing unit tests - you define expected behavior
through clear specifications!
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try importing OpenAI
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    LLM_AVAILABLE = True
except Exception as e:
    print(f"⚠️  OpenAI not available: {e}")
    print("   You can still complete the exercises without running them!")
    LLM_AVAILABLE = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def call_llm(prompt: str, temperature: float = 0.7) -> str:
    """
    Helper function to call the LLM with your prompt.

    In C#, this is like:
    string CallLlm(string prompt, float temperature = 0.7)
    {
        // Call API and return response
    }
    """
    if not LLM_AVAILABLE:
        return "[LLM not configured - check your API key]"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using cheaper model for exercises
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def evaluate_prompt(prompt: str) -> Dict[str, any]:
    """
    Evaluate a prompt based on the 4-component structure.
    Returns a score and feedback.

    Similar to C#:
    Dictionary<string, object> EvaluatePrompt(string prompt)
    """
    score = 0
    feedback = []

    # Check for role
    role_keywords = ["you are", "act as", "as a", "you're a"]
    if any(keyword in prompt.lower() for keyword in role_keywords):
        score += 25
        feedback.append("✅ Has role definition")
    else:
        feedback.append("❌ Missing role definition (add 'You are a...')")

    # Check for clear task
    if len(prompt) > 50:  # Basic length check
        score += 25
        feedback.append("✅ Has task description")
    else:
        feedback.append("❌ Task description too brief")

    # Check for constraints
    constraint_keywords = ["must", "should", "do not", "avoid", "limit", "only", "exactly"]
    if any(keyword in prompt.lower() for keyword in constraint_keywords):
        score += 25
        feedback.append("✅ Has constraints")
    else:
        feedback.append("❌ Missing constraints")

    # Check for format specification
    format_keywords = ["format:", "output format:", "structure:", "json", "bullet points",
                       "numbered list", "table", "markdown"]
    if any(keyword in prompt.lower() for keyword in format_keywords):
        score += 25
        feedback.append("✅ Has format specification")
    else:
        feedback.append("❌ Missing format specification")

    return {
        "score": score,
        "feedback": feedback,
        "passed": score >= 75
    }


# ============================================================================
# EXERCISE 1: EMAIL CLASSIFICATION
# ============================================================================

def exercise_1_email_classification() -> str:
    """
    TASK: Write a prompt to classify customer emails.

    REQUIREMENTS:
    - Role: Customer service AI assistant
    - Task: Classify emails into categories (question, complaint, praise, request)
    - Constraints: Must provide confidence score, only use listed categories
    - Format: JSON with category and confidence

    EXAMPLE EMAIL:
    "I love your product! It's made my life so much easier.
     Just wanted to say thanks!"

    YOUR TASK: Complete the prompt below
    """

    # TODO: Write your prompt here
    prompt = """
    # WRITE YOUR PROMPT HERE
    # Include all 4 components: Role, Task, Constraints, Format
    # Delete this placeholder text and replace with your prompt
    """

    # Evaluate your prompt
    evaluation = evaluate_prompt(prompt)
    print("\n" + "="*60)
    print("EXERCISE 1: Email Classification")
    print("="*60)
    print(f"Your Prompt:\n{prompt}\n")
    print(f"Score: {evaluation['score']}/100")
    for item in evaluation['feedback']:
        print(f"  {item}")

    if evaluation['passed']:
        print("\n✅ PASSED! Your prompt has good structure.")
        if LLM_AVAILABLE:
            print("\nTesting with sample email...")
            sample_email = "I love your product! It's made my life so much easier. Just wanted to say thanks!"
            result = call_llm(prompt + f"\n\nEmail: {sample_email}")
            print(f"LLM Response:\n{result}")
    else:
        print("\n❌ NEEDS IMPROVEMENT. Review the feedback above.")
        print("\nHINT: Look at example_01_zero_shot.py for reference!")

    return prompt


# ============================================================================
# EXERCISE 2: CODE DOCUMENTATION
# ============================================================================

def exercise_2_code_documentation() -> str:
    """
    TASK: Write a prompt to generate code documentation.

    REQUIREMENTS:
    - Role: Senior software engineer and technical writer
    - Task: Write docstring for a Python function
    - Constraints: Follow Google style, include examples, max 10 lines
    - Format: Python docstring format

    SAMPLE FUNCTION:
    def calculate_discount(price: float, percentage: float) -> float:
        return price * (1 - percentage / 100)

    YOUR TASK: Complete the prompt below
    """

    # TODO: Write your prompt here
    prompt = """
    # WRITE YOUR PROMPT HERE
    # Remember: Role, Task, Constraints, Format
    """

    evaluation = evaluate_prompt(prompt)
    print("\n" + "="*60)
    print("EXERCISE 2: Code Documentation")
    print("="*60)
    print(f"Your Prompt:\n{prompt}\n")
    print(f"Score: {evaluation['score']}/100")
    for item in evaluation['feedback']:
        print(f"  {item}")

    if evaluation['passed']:
        print("\n✅ PASSED!")
        if LLM_AVAILABLE:
            print("\nTesting with sample function...")
            sample_code = """
def calculate_discount(price: float, percentage: float) -> float:
    return price * (1 - percentage / 100)
"""
            result = call_llm(prompt + f"\n\nFunction:\n{sample_code}")
            print(f"LLM Response:\n{result}")
    else:
        print("\n❌ NEEDS IMPROVEMENT.")

    return prompt


# ============================================================================
# EXERCISE 3: DATA EXTRACTION
# ============================================================================

def exercise_3_data_extraction() -> str:
    """
    TASK: Write a prompt to extract structured data from text.

    REQUIREMENTS:
    - Role: Data extraction specialist
    - Task: Extract person name, company, email, phone from text
    - Constraints: Return "null" if not found, validate email format
    - Format: JSON with specific fields

    SAMPLE TEXT:
    "Hi, I'm John Smith from Acme Corp. You can reach me at
     john.smith@acme.com or call (555) 123-4567."

    YOUR TASK: Complete the prompt below
    """

    # TODO: Write your prompt here
    prompt = """
    # WRITE YOUR PROMPT HERE
    """

    evaluation = evaluate_prompt(prompt)
    print("\n" + "="*60)
    print("EXERCISE 3: Data Extraction")
    print("="*60)
    print(f"Your Prompt:\n{prompt}\n")
    print(f"Score: {evaluation['score']}/100")
    for item in evaluation['feedback']:
        print(f"  {item}")

    if evaluation['passed']:
        print("\n✅ PASSED!")
        if LLM_AVAILABLE:
            sample_text = "Hi, I'm John Smith from Acme Corp. You can reach me at john.smith@acme.com or call (555) 123-4567."
            result = call_llm(prompt + f"\n\nText: {sample_text}")
            print(f"LLM Response:\n{result}")
    else:
        print("\n❌ NEEDS IMPROVEMENT.")

    return prompt


# ============================================================================
# EXERCISE 4: SENTIMENT ANALYSIS
# ============================================================================

def exercise_4_sentiment_analysis() -> str:
    """
    TASK: Write a prompt for sentiment analysis.

    REQUIREMENTS:
    - Role: Sentiment analysis expert
    - Task: Analyze sentiment of product reviews
    - Constraints: Score from -1 (negative) to +1 (positive), explain reasoning
    - Format: JSON with score, label, and explanation

    YOUR TASK: Write a complete prompt
    """

    # TODO: Write your prompt here
    prompt = """
    # WRITE YOUR PROMPT HERE
    """

    evaluation = evaluate_prompt(prompt)
    print("\n" + "="*60)
    print("EXERCISE 4: Sentiment Analysis")
    print("="*60)
    print(f"Your Prompt:\n{prompt}\n")
    print(f"Score: {evaluation['score']}/100")
    for item in evaluation['feedback']:
        print(f"  {item}")

    return prompt


# ============================================================================
# EXERCISE 5: TEXT SUMMARIZATION
# ============================================================================

def exercise_5_text_summarization() -> str:
    """
    TASK: Write a prompt for article summarization.

    REQUIREMENTS:
    - Role: Professional editor
    - Task: Summarize news articles
    - Constraints: Exactly 3 sentences, preserve key facts, neutral tone
    - Format: Plain text with sentence numbers

    YOUR TASK: Write a complete prompt
    """

    # TODO: Write your prompt here
    prompt = """
    # WRITE YOUR PROMPT HERE
    """

    evaluation = evaluate_prompt(prompt)
    print("\n" + "="*60)
    print("EXERCISE 5: Text Summarization")
    print("="*60)
    print(f"Your Prompt:\n{prompt}\n")
    print(f"Score: {evaluation['score']}/100")
    for item in evaluation['feedback']:
        print(f"  {item}")

    return prompt


# ============================================================================
# BONUS CHALLENGE: COMPARE TEMPERATURES
# ============================================================================

def bonus_temperature_experiment():
    """
    BONUS: Experiment with temperature settings.

    Run the same prompt with different temperatures and observe:
    - temperature=0.0 (deterministic, consistent)
    - temperature=0.7 (balanced)
    - temperature=1.5 (creative, varied)

    This is like adjusting randomness in C# Random class!
    """
    if not LLM_AVAILABLE:
        print("⚠️  LLM not available for bonus challenge")
        return

    prompt = """You are a creative writer. Write a one-sentence story about a robot."""

    print("\n" + "="*60)
    print("BONUS: Temperature Experiment")
    print("="*60)
    print(f"Prompt: {prompt}\n")

    for temp in [0.0, 0.7, 1.5]:
        result = call_llm(prompt, temperature=temp)
        print(f"Temperature {temp}: {result}\n")


# ============================================================================
# SAMPLE SOLUTIONS
# ============================================================================

def show_sample_solutions():
    """
    Show sample solutions for reference.

    IMPORTANT: Try the exercises first before looking at these!
    """

    print("\n" + "="*60)
    print("SAMPLE SOLUTIONS (Don't peek until you've tried!)")
    print("="*60)

    print("\nEXERCISE 1 - Sample Solution:")
    print("""
You are a customer service AI assistant specialized in email classification.

Task: Analyze the following customer email and classify it into exactly one category.

Categories (choose one):
- question: Customer asking for information
- complaint: Customer expressing dissatisfaction
- praise: Customer expressing satisfaction or thanks
- request: Customer asking for something to be done

Constraints:
- Provide a confidence score between 0 and 1
- Only use the categories listed above
- Base your decision on the email content only

Output Format (JSON):
{
  "category": "praise",
  "confidence": 0.95,
  "reasoning": "Brief explanation"
}
    """)

    print("\nEXERCISE 2 - Sample Solution:")
    print("""
You are a senior software engineer and technical writer with expertise in Python documentation.

Task: Write a comprehensive docstring for the provided Python function.

Constraints:
- Follow Google style docstring format
- Include: description, Args, Returns, and Examples sections
- Maximum 10 lines
- Include at least one usage example
- Use proper type hints

Output Format: Python docstring in triple quotes
    """)

    print("\nEXERCISE 3 - Sample Solution:")
    print("""
You are a data extraction specialist with expertise in parsing unstructured text.

Task: Extract contact information from the provided text.

Fields to extract:
- name: Person's full name
- company: Company name
- email: Email address (validate format)
- phone: Phone number

Constraints:
- Return "null" for any field not found in the text
- Validate email format (must contain @ and domain)
- Preserve original formatting of phone numbers
- Do not make assumptions or infer missing data

Output Format (JSON):
{
  "name": "John Smith",
  "company": "Acme Corp",
  "email": "john.smith@acme.com",
  "phone": "(555) 123-4567"
}
    """)


# ============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """
    Main function to run all exercises.

    In C#, this would be:
    static void Main(string[] args)
    """

    print("="*60)
    print("MODULE 8 - EXERCISE 1: Zero-Shot Prompting Practice")
    print("="*60)
    print("\nThis exercise helps you practice the 4-component prompt structure:")
    print("1. ROLE - Who is the AI?")
    print("2. TASK - What should it do?")
    print("3. CONSTRAINTS - What are the limits?")
    print("4. FORMAT - How should output look?")
    print("\nComplete each exercise, then check the sample solutions.")
    print("\nPress Enter to continue...")
    input()

    # Run exercises
    exercise_1_email_classification()
    input("\nPress Enter for Exercise 2...")

    exercise_2_code_documentation()
    input("\nPress Enter for Exercise 3...")

    exercise_3_data_extraction()
    input("\nPress Enter for Exercise 4...")

    exercise_4_sentiment_analysis()
    input("\nPress Enter for Exercise 5...")

    exercise_5_text_summarization()

    # Bonus challenge
    if LLM_AVAILABLE:
        print("\n\nWould you like to try the bonus temperature experiment? (y/n): ", end="")
        if input().lower() == 'y':
            bonus_temperature_experiment()

    # Show solutions
    print("\n\nWould you like to see sample solutions? (y/n): ", end="")
    if input().lower() == 'y':
        show_sample_solutions()

    print("\n" + "="*60)
    print("EXERCISE COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Review your prompts and compare with sample solutions")
    print("2. Try improving prompts that scored < 100")
    print("3. Experiment with different phrasings")
    print("4. Move on to Lesson 2: Few-Shot Learning")
    print("\nKeep practicing - prompt engineering is a skill!")


if __name__ == "__main__":
    main()

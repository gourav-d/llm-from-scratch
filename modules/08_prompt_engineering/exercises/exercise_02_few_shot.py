"""
Exercise 2: Few-Shot Learning Practice

Complete these exercises to master few-shot prompting.

Instructions:
1. Read each exercise description
2. Complete the TODO sections
3. Run the script to test your solutions
4. Compare with the provided solutions at the end
"""

import os
import sys
from typing import List, Dict
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from examples.example_01_zero_shot import get_llm_client, call_llm
except ImportError:
    print("Please ensure example_01_zero_shot.py is available")
    sys.exit(1)

load_dotenv()


# ==============================================================================
# EXERCISE 1: Email Classification
# ==============================================================================

def exercise_1_email_classification():
    """
    Exercise 1: Create a few-shot prompt to classify emails as:
    - urgent (needs immediate action)
    - important (needs action this week)
    - normal (can wait)
    - spam (unsolicited)

    TODO: Create 4-5 examples showing each category
    """
    print("\n" + "=" * 70)
    print("EXERCISE 1: EMAIL CLASSIFICATION")
    print("=" * 70)

    provider, client = get_llm_client()

    # Test emails
    test_emails = [
        "URGENT: Server is down! All customers affected!",
        "Weekly team meeting scheduled for Friday",
        "Congratulations! You've won $1,000,000!!!",
        "Reminder: Project deadline is Monday",
    ]

    # TODO: Create your few-shot prompt here
    # Hint: Include 1-2 examples for each category
    prompt_template = """
Classify emails as: urgent, important, normal, or spam

# TODO: Add your examples here
# Example format:
# Email: "..."
# Category: ...

Now classify:
Email: "{email}"
Category:
"""

    # YOUR CODE HERE:
    # Replace the above template with your complete few-shot prompt
    # Include examples for each category

    print("Testing your prompt on sample emails:\n")
    for email in test_emails:
        prompt = prompt_template.format(email=email)
        result = call_llm(prompt, provider, client, temperature=0.0)
        print(f"Email: {email[:50]}...")
        print(f"Classification: {result}\n")

    print("💡 Check: Did your examples help classify all emails correctly?")
    print("Expected: urgent, normal, spam, important")


# ==============================================================================
# EXERCISE 2: Code Comment Generation
# ==============================================================================

def exercise_2_code_comments():
    """
    Exercise 2: Create few-shot examples to generate code comments.

    TODO: Provide examples of good code comments for different code patterns
    """
    print("\n" + "=" * 70)
    print("EXERCISE 2: CODE COMMENT GENERATION")
    print("=" * 70)

    provider, client = get_llm_client()

    # Code to comment
    code = """
def calculate_discount(price, customer_type):
    if customer_type == "premium":
        return price * 0.8
    elif customer_type == "regular":
        return price * 0.9
    return price
"""

    # TODO: Create few-shot examples showing how to comment code
    prompt = """
Generate clear, concise code comments.

# TODO: Add 2-3 examples of code with good comments

Now generate comments for:
{code}

Commented version:
""".format(code=code)

    # YOUR CODE HERE:
    # Create examples showing:
    # 1. Function-level comments
    # 2. Inline comments for complex logic
    # 3. Brief, descriptive style

    print("Generated comments:\n")
    result = call_llm(prompt, provider, client, temperature=0.3)
    print(result)

    print("\n💡 Check: Are the comments clear and helpful?")


# ==============================================================================
# EXERCISE 3: Data Validation
# ==============================================================================

def exercise_3_data_validation():
    """
    Exercise 3: Create few-shot examples for validating input data.

    TODO: Show examples of valid and invalid inputs with explanations
    """
    print("\n" + "=" * 70)
    print("EXERCISE 3: DATA VALIDATION")
    print("=" * 70)

    provider, client = get_llm_client()

    # Test inputs
    test_inputs = [
        "john@email.com",
        "invalid-email",
        "test@domain.co.uk",
        "@nodomain.com",
    ]

    # TODO: Create few-shot prompt showing valid/invalid email patterns
    prompt_template = """
Validate email addresses. Return: valid or invalid with reason.

# TODO: Add examples of valid and invalid emails

Now validate:
Email: "{email}"
Result:
"""

    # YOUR CODE HERE:
    # Create examples showing:
    # 1. Valid emails with different formats
    # 2. Invalid emails with specific issues

    print("Validating emails:\n")
    for email in test_inputs:
        prompt = prompt_template.format(email=email)
        result = call_llm(prompt, provider, client, temperature=0.0)
        print(f"{email}: {result}\n")

    print("💡 Check: Did it correctly identify valid and invalid emails?")


# ==============================================================================
# EXERCISE 4: Summarization Style
# ==============================================================================

def exercise_4_summarization_style():
    """
    Exercise 4: Use few-shot to teach a specific summarization style.

    TODO: Create examples showing your desired summary style
    """
    print("\n" + "=" * 70)
    print("EXERCISE 4: CUSTOM SUMMARIZATION STYLE")
    print("=" * 70)

    provider, client = get_llm_client()

    article = """
    Artificial Intelligence (AI) has transformed various industries in recent years.
    From healthcare to finance, AI applications are helping businesses make better
    decisions and improve efficiency. Machine learning algorithms can analyze vast
    amounts of data quickly, identifying patterns that humans might miss. However,
    challenges remain, including data privacy concerns and the need for skilled
    professionals. Despite these obstacles, experts predict continued growth in
    AI adoption across all sectors.
    """

    # TODO: Create few-shot examples showing "executive summary" style
    # Requirements:
    # - Start with key finding
    # - Use bullet points
    # - Include impact/implications
    # - Max 3 bullets

    prompt = """
Create executive summaries in this style:

# TODO: Add 2 examples showing your summary style

Now summarize:
Article: {article}

Summary:
""".format(article=article)

    # YOUR CODE HERE:
    # Show examples with:
    # - Key finding upfront
    # - Bullet format
    # - Business implications

    print("Generated summary:\n")
    result = call_llm(prompt, provider, client, temperature=0.3)
    print(result)

    print("\n💡 Check: Does it follow your example style?")


# ==============================================================================
# EXERCISE 5: Error Message Improvement
# ==============================================================================

def exercise_5_error_messages():
    """
    Exercise 5: Improve technical error messages for end users.

    TODO: Show examples of converting technical errors to friendly messages
    """
    print("\n" + "=" * 70)
    print("EXERCISE 5: USER-FRIENDLY ERROR MESSAGES")
    print("=" * 70)

    provider, client = get_llm_client()

    technical_errors = [
        "NullPointerException at line 42",
        "Connection timeout after 30000ms",
        "Invalid token: JWT signature verification failed",
    ]

    # TODO: Create few-shot examples converting technical → friendly
    prompt_template = """
Convert technical error messages to user-friendly explanations.

# TODO: Add 2-3 examples

Now convert:
Technical: "{error}"
User-friendly:
"""

    # YOUR CODE HERE:
    # Show examples that:
    # 1. Explain what happened
    # 2. Suggest what to do
    # 3. Avoid technical jargon

    print("Converting error messages:\n")
    for error in technical_errors:
        prompt = prompt_template.format(error=error)
        result = call_llm(prompt, provider, client, temperature=0.3)
        print(f"Technical: {error}")
        print(f"Friendly: {result}\n")

    print("💡 Check: Are the messages clear and actionable?")


# ==============================================================================
# CHALLENGE EXERCISE: Optimal Example Selection
# ==============================================================================

def challenge_optimal_examples():
    """
    CHALLENGE: Find the minimum number of examples needed for good accuracy.

    Task: Classify product reviews as positive, negative, or neutral.
    Test with 1, 2, 3, and 5 examples to find the sweet spot.
    """
    print("\n" + "=" * 70)
    print("CHALLENGE: FIND OPTIMAL EXAMPLE COUNT")
    print("=" * 70)

    provider, client = get_llm_client()

    # Example pool
    all_examples = [
        ('Review: "Best product ever!" → positive', 1),
        ('Review: "Complete waste of money" → negative', 2),
        ('Review: "It\'s okay, does the job" → neutral', 3),
        ('Review: "Not bad for the price" → neutral', 4),
        ('Review: "Exceeded my expectations!" → positive', 5),
    ]

    # Test review
    test_review = "Decent quality, nothing spectacular"
    expected = "neutral"

    print(f"Test review: '{test_review}'")
    print(f"Expected: {expected}\n")

    # TODO: Test with different numbers of examples
    for num_examples in [1, 2, 3, 5]:
        # YOUR CODE HERE:
        # 1. Build prompt with first N examples
        # 2. Test with the review
        # 3. Check if it matches expected result
        # 4. Note the accuracy and cost trade-off

        examples_text = "\n".join([ex[0] for ex in all_examples[:num_examples]])

        prompt = f"""
Classify review sentiment:

{examples_text}

Review: "{test_review}"
Sentiment:
"""

        result = call_llm(prompt, provider, client, temperature=0.0)
        correct = expected in result.lower()

        print(f"With {num_examples} example(s): {result} "
              f"{'✅' if correct else '❌'}")

    print("\n💡 Question: What's the minimum examples needed for accuracy?")


# ==============================================================================
# SOLUTIONS (Don't peek until you've tried!)
# ==============================================================================

def show_solutions():
    """
    Show solution examples for each exercise.
    """
    print("\n" + "=" * 70)
    print("SOLUTIONS")
    print("=" * 70)

    print("""
EXERCISE 1 SOLUTION:
-------------------
Classify emails as: urgent, important, normal, or spam

Examples:

Email: "URGENT: Server down, all users affected!"
Category: urgent

Email: "Reminder: Board meeting next Monday at 10am"
Category: important

Email: "Newsletter: Top 10 productivity tips"
Category: normal

Email: "You've won a free iPad! Click here now!!!"
Category: spam

Email: "FYI: New parking policy starts next month"
Category: normal


EXERCISE 2 SOLUTION:
-------------------
Generate clear, concise code comments.

Example 1:
def add_numbers(a, b):
    return a + b

Commented:
def add_numbers(a, b):
    \"\"\"Add two numbers and return the result.\"\"\"
    return a + b

Example 2:
def process_order(items, discount):
    total = sum(item.price for item in items)
    if discount > 0:
        total = total * (1 - discount)
    return total

Commented:
def process_order(items, discount):
    \"\"\"Calculate order total with optional discount.

    Args:
        items: List of items in order
        discount: Discount rate (0.0 to 1.0)
    \"\"\"
    total = sum(item.price for item in items)
    # Apply discount if provided
    if discount > 0:
        total = total * (1 - discount)
    return total


EXERCISE 3 SOLUTION:
-------------------
Validate email addresses.

Examples:

Email: "user@example.com"
Result: valid - standard format

Email: "john.doe@company.co.uk"
Result: valid - includes subdomain

Email: "invalid@"
Result: invalid - missing domain

Email: "@nodomain.com"
Result: invalid - missing username

Email: "no-at-sign.com"
Result: invalid - missing @ symbol


EXERCISE 4 SOLUTION:
-------------------
Create executive summaries:

Example:
Article: "Cloud computing adoption has grown 45% this year. Companies
cite cost savings and scalability as key benefits. Main challenges
include security concerns and migration complexity."

Summary:
• Key finding: Cloud adoption up 45% YoY driven by cost and scalability
• Benefits: Reduced infrastructure costs, easier scaling
• Challenges: Security and migration remain top concerns


EXERCISE 5 SOLUTION:
-------------------
Convert technical errors to user-friendly messages.

Example 1:
Technical: "Database connection failed: errno 111"
User-friendly: "We're having trouble connecting to our servers.
Please check your internet connection and try again in a moment."

Example 2:
Technical: "AuthenticationException: Invalid credentials"
User-friendly: "The email or password you entered is incorrect.
Please double-check and try again."

Example 3:
Technical: "MemoryError: Cannot allocate array"
User-friendly: "This file is too large to process.
Please try a smaller file or contact support for help."


KEY PATTERNS:
------------
1. Use 3-5 diverse examples
2. Show edge cases in examples
3. Keep examples concise but clear
4. Maintain consistent format across examples
5. Include examples that cover the full range of outputs
""")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """
    Run all exercises
    """
    print("\n" + "=" * 70)
    print("FEW-SHOT LEARNING EXERCISES")
    print("=" * 70)
    print("\nComplete each exercise by filling in the TODO sections.")
    print("Don't look at the solutions until you've tried!\n")

    try:
        # Run exercises
        choice = input("Run all exercises? (y/n): ").lower()

        if choice == 'y':
            exercise_1_email_classification()
            exercise_2_code_comments()
            exercise_3_data_validation()
            exercise_4_summarization_style()
            exercise_5_error_messages()
            challenge_optimal_examples()

        # Offer to show solutions
        show_sol = input("\nShow solutions? (y/n): ").lower()
        if show_sol == 'y':
            show_solutions()

        print("\n" + "=" * 70)
        print("GREAT JOB!")
        print("=" * 70)
        print("""
You've practiced:
✅ Multi-class classification with few-shot
✅ Teaching specific output formats
✅ Handling edge cases with examples
✅ Finding optimal example counts
✅ Creating reusable few-shot patterns

Next: Move on to Lesson 3 (Prompt Templates) to make these patterns reusable!
""")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Completed the setup from example_01")
        print("2. API key configured properly")


if __name__ == "__main__":
    main()

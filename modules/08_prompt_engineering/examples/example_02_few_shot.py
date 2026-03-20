"""
Example 2: Few-Shot Learning

This example demonstrates the power of teaching models through examples.

Prerequisites:
- Completed example_01_zero_shot.py
- OpenAI API key or Anthropic API key or Ollama installed
"""

import os
import sys
from typing import List, Dict
from dotenv import load_dotenv

# Add parent directory to path to import from example_01
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import from example_01
try:
    from examples.example_01_zero_shot import get_llm_client, call_llm
except ImportError:
    # If import fails, define locally
    print("Note: Importing LLM functions from example_01")

# Load environment variables
load_dotenv()


def example_1_zero_shot_vs_few_shot():
    """
    Example 1: Direct comparison of zero-shot vs few-shot
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: ZERO-SHOT VS FEW-SHOT COMPARISON")
    print("=" * 70)

    provider, client = get_llm_client()

    # Test text
    text = "This product is not bad, could be better though"

    # ZERO-SHOT: May misinterpret "not bad"
    print("\n--- ZERO-SHOT APPROACH ---")
    zero_shot_prompt = f"""
Classify the sentiment of this review as positive, negative, or neutral.

Review: {text}
Sentiment:
"""
    print(f"Prompt:\n{zero_shot_prompt}")

    zero_shot_result = call_llm(zero_shot_prompt, provider, client, temperature=0.0)
    print(f"\nResult: {zero_shot_result}")
    print("❌ Problem: May misinterpret \"not bad\" as positive instead of neutral")

    # FEW-SHOT: Examples clarify how to handle edge cases
    print("\n--- FEW-SHOT APPROACH ---")
    few_shot_prompt = f"""
Classify the sentiment of reviews as positive, negative, or neutral.

Examples:

Review: "I absolutely love this product! Best purchase ever!"
Sentiment: positive

Review: "This is terrible. Waste of money."
Sentiment: negative

Review: "It's okay. Does what it's supposed to do."
Sentiment: neutral

Review: "Not bad for the price, could be better"
Sentiment: neutral

Review: "Doesn't work at all. Very disappointed."
Sentiment: negative

Now classify:
Review: {text}
Sentiment:
"""
    print(f"Prompt:\n{few_shot_prompt}")

    few_shot_result = call_llm(few_shot_prompt, provider, client, temperature=0.0)
    print(f"\nResult: {few_shot_result}")
    print("✅ Benefit: Examples teach model to handle edge cases like \"not bad\"")


def example_2_data_extraction():
    """
    Example 2: Structured data extraction with few-shot
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: DATA EXTRACTION WITH FEW-SHOT")
    print("=" * 70)

    provider, client = get_llm_client()

    # Contact information to extract
    text = "Please contact Sarah Johnson at sjohnson@techcorp.com or call 555-0123"

    # FEW-SHOT: Show exact format
    prompt = """
Extract contact information in this exact JSON format.

Example 1:
Input: "Reach John at john.doe@email.com, phone: 555-1234"
Output: {
  "name": "John",
  "email": "john.doe@email.com",
  "phone": "555-1234"
}

Example 2:
Input: "Contact Alice Smith (alice@company.org)"
Output: {
  "name": "Alice Smith",
  "email": "alice@company.org",
  "phone": null
}

Example 3:
Input: "Call Mike on 555-5678"
Output: {
  "name": "Mike",
  "email": null,
  "phone": "555-5678"
}

Now extract from:
Input: "%s"
Output:
""" % text

    print(f"Prompt:\n{prompt}\n")

    result = call_llm(prompt, provider, client, temperature=0.0)
    print(f"Result:\n{result}\n")
    print("✅ Few-shot examples ensure consistent JSON format")

    # Try to parse as JSON
    try:
        import json
        parsed = json.loads(result)
        print(f"✅ Successfully parsed: {parsed}")
    except:
        print("⚠️  Note: Response may need cleaning to be valid JSON")


def example_3_optimal_example_count():
    """
    Example 3: Finding the optimal number of examples
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: OPTIMAL NUMBER OF EXAMPLES")
    print("=" * 70)

    provider, client = get_llm_client()

    task_text = "Convert to snake_case: HelloWorldTest"

    # Base examples pool
    all_examples = [
        ('Input: "Hello" → Output: "hello"', 1),
        ('Input: "HelloWorld" → Output: "hello_world"', 2),
        ('Input: "TestCase" → Output: "test_case"', 3),
        ('Input: "MyVariable" → Output: "my_variable"', 4),
        ('Input: "HTTPSConnection" → Output: "https_connection"', 5),
    ]

    print("\nTesting with different numbers of examples:\n")

    # Test with 1, 2, 3, and 5 examples
    for num_examples in [1, 2, 3, 5]:
        examples_text = "\n".join([ex[0] for ex in all_examples[:num_examples]])

        prompt = f"""
Convert to snake_case format.

{examples_text}

Now convert:
{task_text}
"""

        print(f"--- WITH {num_examples} EXAMPLE(S) ---")
        result = call_llm(prompt, provider, client, temperature=0.0)
        print(f"Result: {result}")

        # Estimate token count (rough approximation)
        token_count = len(prompt.split())
        print(f"Approx tokens: {token_count}")
        print()

    print("💡 Insight: Usually 2-3 examples give best cost/accuracy balance")


def example_4_classification_task():
    """
    Example 4: Multi-class classification with few-shot
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: MULTI-CLASS CLASSIFICATION")
    print("=" * 70)

    provider, client = get_llm_client()

    # Support ticket to classify
    ticket = "My login isn't working and I can't access my account"

    prompt = """
Classify support tickets into categories: technical, billing, account, or feedback.

Examples:

Ticket: "I was charged twice for my subscription"
Category: billing

Ticket: "The app keeps crashing when I try to upload files"
Category: technical

Ticket: "I want to update my email address"
Category: account

Ticket: "Great product! Would love to see dark mode added"
Category: feedback

Ticket: "Can't connect to the database server"
Category: technical

Ticket: "Need a refund for last month"
Category: billing

Now classify:
Ticket: "%s"
Category:
""" % ticket

    print(f"Prompt:\n{prompt}\n")

    result = call_llm(prompt, provider, client, temperature=0.0)
    print(f"Result: {result}\n")
    print("✅ Examples for each category ensure accurate classification")


def example_5_chain_of_thought_few_shot():
    """
    Example 5: Few-shot with chain-of-thought reasoning
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: FEW-SHOT WITH CHAIN-OF-THOUGHT")
    print("=" * 70)

    provider, client = get_llm_client()

    problem = "A store had 25 items. They sold 8 in the morning and 5 in the afternoon. How many remain?"

    prompt = """
Solve math word problems. Show your reasoning step by step.

Example 1:
Problem: "Sarah has 10 apples. She gives 3 to John. How many does she have left?"
Solution:
Step 1: Sarah starts with 10 apples
Step 2: She gives away 3 apples
Step 3: Remaining = 10 - 3 = 7 apples
Answer: 7 apples

Example 2:
Problem: "A box contains 20 candies. Mike eats 5 and shares 6 with friends. How many are left?"
Solution:
Step 1: Start with 20 candies
Step 2: Mike eats 5, so 20 - 5 = 15 remain
Step 3: He shares 6, so 15 - 6 = 9 remain
Answer: 9 candies

Now solve:
Problem: "%s"
Solution:
""" % problem

    print(f"Prompt:\n{prompt}\n")

    result = call_llm(prompt, provider, client, temperature=0.0)
    print(f"Result:\n{result}\n")
    print("✅ Examples with reasoning teach model to show its work")


def example_6_format_transformation():
    """
    Example 6: Format transformation with few-shot
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: FORMAT TRANSFORMATION")
    print("=" * 70)

    provider, client = get_llm_client()

    input_text = "John Smith, 555-1234, john@email.com"

    prompt = """
Convert contact info to vCard format.

Example 1:
Input: "Alice Johnson, 555-5678, alice@company.com"
Output:
BEGIN:VCARD
FN:Alice Johnson
TEL:555-5678
EMAIL:alice@company.com
END:VCARD

Example 2:
Input: "Bob Wilson, 555-9999, bob.w@startup.io"
Output:
BEGIN:VCARD
FN:Bob Wilson
TEL:555-9999
EMAIL:bob.w@startup.io
END:VCARD

Now convert:
Input: "%s"
Output:
""" % input_text

    print(f"Prompt:\n{prompt}\n")

    result = call_llm(prompt, provider, client, temperature=0.0)
    print(f"Result:\n{result}\n")
    print("✅ Examples teach exact output format, even for structured text")


def example_7_edge_cases():
    """
    Example 7: Teaching edge cases through examples
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: HANDLING EDGE CASES")
    print("=" * 70)

    provider, client = get_llm_client()

    # Test with tricky input
    text = "I don't not like this product"  # Double negative

    prompt = """
Classify sentiment, handling negations carefully.

Examples:

Text: "I love this product"
Sentiment: positive
Reasoning: Clear positive statement

Text: "I don't like this product"
Sentiment: negative
Reasoning: Negation makes it negative

Text: "This is not bad"
Sentiment: neutral
Reasoning: Double negative (not + bad) = neutral/slight positive

Text: "I can't say I hate it"
Sentiment: neutral
Reasoning: Negated negative = neutral

Text: "Couldn't be happier!"
Sentiment: positive
Reasoning: Negation emphasizing extreme positive

Now classify:
Text: "%s"
Sentiment:
Reasoning:
""" % text

    print(f"Prompt:\n{prompt}\n")

    result = call_llm(prompt, provider, client, temperature=0.0)
    print(f"Result:\n{result}\n")
    print("✅ Examples teach model to handle complex negations")


def example_8_progressive_complexity():
    """
    Example 8: Progressive examples (simple to complex)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: PROGRESSIVE COMPLEXITY")
    print("=" * 70)

    provider, client = get_llm_client()

    input_text = "Re: RE: FWD: Meeting notes - URGENT!!!"

    prompt = """
Clean email subject lines by removing prefixes and excessive punctuation.

Simple example:
"Hello" → "Hello"

With prefix:
"Re: Meeting" → "Meeting"

Multiple prefixes:
"Re: FWD: Update" → "Update"

With punctuation:
"URGENT!!!" → "URGENT"

Complex case:
"Re: RE: FWD: Project Update - IMPORTANT!!!" → "Project Update - IMPORTANT"

Now clean:
"%s"
""" % input_text

    print(f"Prompt:\n{prompt}\n")

    result = call_llm(prompt, provider, client, temperature=0.0)
    print(f"Result:\n{result}\n")
    print("✅ Progressive examples build from simple to complex cases")


def compare_example_quality():
    """
    Bonus: Compare impact of good vs bad examples
    """
    print("\n" + "=" * 70)
    print("BONUS: GOOD VS BAD EXAMPLES")
    print("=" * 70)

    provider, client = get_llm_client()

    task = "Classify: 'The movie was okay, nothing special'"

    # BAD EXAMPLES: Too similar, don't show spectrum
    print("\n--- WITH BAD EXAMPLES (all similar) ---")
    bad_prompt = """
Classify sentiment:

"Amazing movie!" → positive
"Best film ever!" → positive
"Loved it!" → positive

Classify: "%s"
""" % task

    bad_result = call_llm(bad_prompt, provider, client, temperature=0.0)
    print(f"Result: {bad_result}")
    print("❌ Problem: Examples don't show the full range")

    # GOOD EXAMPLES: Diverse, showing spectrum
    print("\n--- WITH GOOD EXAMPLES (diverse) ---")
    good_prompt = """
Classify sentiment:

"Best movie I've ever seen!" → positive
"Pretty good, enjoyed it" → positive
"It was okay, nothing special" → neutral
"Didn't like it much" → negative
"Worst film ever!" → negative

Classify: "%s"
""" % task

    good_result = call_llm(good_prompt, provider, client, temperature=0.0)
    print(f"Result: {good_result}")
    print("✅ Benefit: Diverse examples show the full spectrum")


def main():
    """
    Run all examples
    """
    print("\n" + "=" * 70)
    print("FEW-SHOT LEARNING EXAMPLES")
    print("=" * 70)
    print("\nThese examples show how providing examples dramatically")
    print("improves model performance and consistency!")

    try:
        # Run all examples
        example_1_zero_shot_vs_few_shot()
        example_2_data_extraction()
        example_3_optimal_example_count()
        example_4_classification_task()
        example_5_chain_of_thought_few_shot()
        example_6_format_transformation()
        example_7_edge_cases()
        example_8_progressive_complexity()
        compare_example_quality()

        # Summary
        print("\n" + "=" * 70)
        print("KEY TAKEAWAYS")
        print("=" * 70)
        print("""
1. ✅ Few-shot > zero-shot for complex tasks
2. ✅ 2-3 examples often enough for simple tasks
3. ✅ 3-5 examples is the sweet spot for most tasks
4. ✅ Examples teach format better than instructions
5. ✅ Include edge cases in your examples
6. ✅ Progressive examples (simple → complex) work well
7. ✅ Quality > quantity: diverse examples beat many similar ones

Next Steps:
1. Try modifying the examples
2. Test with your own tasks
3. Compare zero-shot vs few-shot accuracy
4. Complete: exercises/exercise_02_few_shot.py
""")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up an LLM provider (OpenAI/Anthropic/Ollama)")
        print("2. Installed required packages: pip install -r requirements.txt")
        print("3. Set API key in .env file or environment variable")


if __name__ == "__main__":
    main()

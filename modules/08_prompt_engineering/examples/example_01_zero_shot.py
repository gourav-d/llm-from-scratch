"""
Example 1: Zero-Shot Prompting

This example demonstrates the difference between bad and good zero-shot prompts.

Prerequisites:
- OpenAI API key or Anthropic API key or Ollama installed
- Set environment variable: OPENAI_API_KEY or ANTHROPIC_API_KEY
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_llm_client():
    """
    Get an LLM client based on available API keys.
    Tries OpenAI, Anthropic, then Ollama in that order.
    """
    # Try OpenAI first
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            print("✅ Using OpenAI GPT-4o-mini")
            return ("openai", client)
        except ImportError:
            print("⚠️  OpenAI package not installed. Run: pip install openai")

    # Try Anthropic second
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from anthropic import Anthropic
            client = Anthropic()
            print("✅ Using Anthropic Claude")
            return ("anthropic", client)
        except ImportError:
            print("⚠️  Anthropic package not installed. Run: pip install anthropic")

    # Try Ollama last (local, free)
    try:
        import ollama
        # Test if Ollama is running
        ollama.list()
        print("✅ Using Ollama (local)")
        return ("ollama", None)
    except Exception:
        print("⚠️  Ollama not available. Install from ollama.ai")

    raise Exception(
        "No LLM provider available. Please set up:\n"
        "- OpenAI API key (export OPENAI_API_KEY=...)\n"
        "- Anthropic API key (export ANTHROPIC_API_KEY=...)\n"
        "- Or install Ollama (https://ollama.ai)"
    )


def call_llm(prompt: str, provider: str, client, temperature: float = 0.7) -> str:
    """
    Call the LLM with the given prompt.

    Args:
        prompt: The prompt text
        provider: "openai", "anthropic", or "ollama"
        client: The client object (or None for Ollama)
        temperature: Sampling temperature (0.0-1.0)

    Returns:
        The LLM response as a string
    """
    if provider == "openai":
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheaper, faster model
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content

    elif provider == "anthropic":
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Cheaper, faster model
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    elif provider == "ollama":
        import ollama
        response = ollama.generate(
            model="llama3.1",  # Or any model you have installed
            prompt=prompt,
            options={"temperature": temperature}
        )
        return response['response']

    else:
        raise ValueError(f"Unknown provider: {provider}")


def example_1_bad_vs_good():
    """
    Example 1: Compare bad prompt vs good prompt
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: BAD PROMPT VS GOOD PROMPT")
    print("=" * 70)

    provider, client = get_llm_client()

    # Sample email to summarize
    email = """
    Hi team,

    I wanted to update you on the Q4 product launch. We've completed the beta
    testing phase with 250 users, and the feedback has been overwhelmingly
    positive (4.7/5 average rating).

    However, we've identified a critical bug in the payment processing system
    that needs to be fixed before launch. The engineering team estimates 2 weeks
    for the fix and thorough testing.

    This means we'll need to push the launch date from Dec 15 to Dec 29. I know
    this is during the holidays, but we can't ship with this bug.

    I need approval from leadership by Friday to proceed with the revised timeline.

    Best,
    Sarah
    """

    # BAD PROMPT: Too vague
    print("\n--- BAD PROMPT (Vague) ---")
    bad_prompt = "Summarize this email"
    print(f"Prompt: {bad_prompt}\n")

    bad_response = call_llm(bad_prompt + "\n\n" + email, provider, client)
    print(f"Response:\n{bad_response}\n")
    print("❌ Problems: Unpredictable format, no focus, may miss key details")

    # GOOD PROMPT: Specific, structured
    print("\n--- GOOD PROMPT (Specific & Structured) ---")
    good_prompt = """
You are an executive assistant helping busy executives process their inbox.

Task: Summarize the following email for an executive.

Email:
{email}

Provide summary in this format:
- Main topic: [One sentence]
- Key points: [2-3 bullet points]
- Action required: [Yes/No]
- Urgency: [High/Medium/Low]
- Deadline: [If mentioned]
- Recommended response: [Brief suggestion]
""".format(email=email)

    print(f"Prompt:\n{good_prompt}\n")

    good_response = call_llm(good_prompt, provider, client)
    print(f"Response:\n{good_response}\n")
    print("✅ Benefits: Consistent format, clear focus, actionable")


def example_2_adding_constraints():
    """
    Example 2: Effect of adding constraints
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: ADDING CONSTRAINTS")
    print("=" * 70)

    provider, client = get_llm_client()

    topic = "artificial intelligence"

    # WITHOUT CONSTRAINTS
    print("\n--- WITHOUT CONSTRAINTS ---")
    prompt_no_constraints = f"Explain {topic}"
    print(f"Prompt: {prompt_no_constraints}\n")

    response_no_constraints = call_llm(prompt_no_constraints, provider, client)
    print(f"Response:\n{response_no_constraints}\n")
    print("❌ Problems: May be too long, too technical, or unfocused")

    # WITH CONSTRAINTS
    print("\n--- WITH CONSTRAINTS ---")
    prompt_with_constraints = f"""
Explain {topic} to a 10-year-old.

Constraints:
- Maximum 5 sentences
- Use only simple words (elementary school level)
- Include one analogy
- No technical jargon
- Make it fun and engaging
"""
    print(f"Prompt:\n{prompt_with_constraints}\n")

    response_with_constraints = call_llm(prompt_with_constraints, provider, client)
    print(f"Response:\n{response_with_constraints}\n")
    print("✅ Benefits: Perfect length, appropriate level, engaging")


def example_3_output_format():
    """
    Example 3: Specifying output format
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: SPECIFYING OUTPUT FORMAT")
    print("=" * 70)

    provider, client = get_llm_client()

    data = "The quick brown fox jumps over the lazy dog"

    # NO FORMAT SPECIFIED
    print("\n--- NO FORMAT SPECIFIED ---")
    prompt_no_format = f"Analyze this sentence: {data}"
    print(f"Prompt: {prompt_no_format}\n")

    response_no_format = call_llm(prompt_no_format, provider, client)
    print(f"Response:\n{response_no_format}\n")
    print("❌ Problems: Unpredictable structure, hard to parse programmatically")

    # JSON FORMAT SPECIFIED
    print("\n--- JSON FORMAT SPECIFIED ---")
    prompt_json = f"""
Analyze the following sentence and return ONLY valid JSON:

Sentence: {data}

Return this exact structure:
{{
  "word_count": <number>,
  "character_count": <number>,
  "has_punctuation": <boolean>,
  "sentiment": "positive" | "negative" | "neutral",
  "complexity": "simple" | "medium" | "complex",
  "key_words": [<array of strings>]
}}
"""
    print(f"Prompt:\n{prompt_json}\n")

    response_json = call_llm(prompt_json, provider, client, temperature=0.0)
    print(f"Response:\n{response_json}\n")
    print("✅ Benefits: Consistent structure, machine-readable, easy to parse")

    # Try to parse as JSON
    try:
        import json
        parsed = json.loads(response_json)
        print(f"✅ Successfully parsed as JSON: {parsed}")
    except json.JSONDecodeError as e:
        print(f"⚠️  Warning: Response is not valid JSON: {e}")


def example_4_role_prompting():
    """
    Example 4: Effect of role/persona
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: ROLE/PERSONA PROMPTING")
    print("=" * 70)

    provider, client = get_llm_client()

    question = "How can I improve my public speaking skills?"

    # NO ROLE
    print("\n--- NO ROLE (Generic) ---")
    prompt_no_role = question
    print(f"Prompt: {prompt_no_role}\n")

    response_no_role = call_llm(prompt_no_role, provider, client)
    print(f"Response:\n{response_no_role}\n")

    # WITH EXPERT ROLE
    print("\n--- WITH EXPERT ROLE ---")
    prompt_with_role = f"""
You are a professional public speaking coach with 20 years of experience
training executives and TED speakers.

Question: {question}

Provide 3 specific, actionable techniques that can be practiced immediately.
"""
    print(f"Prompt:\n{prompt_with_role}\n")

    response_with_role = call_llm(prompt_with_role, provider, client)
    print(f"Response:\n{response_with_role}\n")
    print("✅ Benefits: More specific, expert-level advice, actionable")


def example_5_temperature_effects():
    """
    Example 5: Temperature effects on creativity
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: TEMPERATURE EFFECTS")
    print("=" * 70)

    provider, client = get_llm_client()

    prompt = "Name 3 creative uses for a paper clip"

    # Temperature 0.0 (Deterministic)
    print("\n--- TEMPERATURE 0.0 (Deterministic) ---")
    response_low = call_llm(prompt, provider, client, temperature=0.0)
    print(f"Response:\n{response_low}\n")
    print("→ Most predictable, same answer every time")

    # Temperature 0.7 (Balanced)
    print("\n--- TEMPERATURE 0.7 (Balanced) ---")
    response_medium = call_llm(prompt, provider, client, temperature=0.7)
    print(f"Response:\n{response_medium}\n")
    print("→ Good balance of creativity and coherence")

    # Temperature 1.5 (Highly Creative)
    print("\n--- TEMPERATURE 1.5 (Highly Creative) ---")
    response_high = call_llm(prompt, provider, client, temperature=1.5)
    print(f"Response:\n{response_high}\n")
    print("→ Most creative, but may be unusual or inconsistent")


def main():
    """
    Run all examples
    """
    print("\n" + "=" * 70)
    print("ZERO-SHOT PROMPTING EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate the impact of good prompting techniques.")
    print("Watch how the same task produces dramatically different results!")

    try:
        # Run all examples
        example_1_bad_vs_good()
        example_2_adding_constraints()
        example_3_output_format()
        example_4_role_prompting()
        example_5_temperature_effects()

        # Summary
        print("\n" + "=" * 70)
        print("KEY TAKEAWAYS")
        print("=" * 70)
        print("""
1. ✅ Specific prompts > vague prompts
2. ✅ Constraints improve quality and consistency
3. ✅ Specified formats = predictable, parseable outputs
4. ✅ Roles/personas enhance expertise level
5. ✅ Temperature controls creativity vs consistency

Next: Try modifying these prompts and see what happens!
Then complete: exercises/exercise_01_zero_shot.py
""")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up an LLM provider (OpenAI/Anthropic/Ollama)")
        print("2. Installed required packages: pip install -r requirements.txt")
        print("3. Set API key in .env file or environment variable")


if __name__ == "__main__":
    main()

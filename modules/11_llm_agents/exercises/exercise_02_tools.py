"""
Exercise 02: Build and Register Custom Tools
=============================================

GOAL
----
Practice writing tools and registering them in a tool registry.
Then build an agent that picks the right tool for each question.

EXERCISES
---------
Exercise 1: Write a temperature_converter tool
Exercise 2: Write a word_counter tool
Exercise 3: Build a tool registry with all tools + auto-select the right one
Exercise 4: Handle tool errors gracefully

HOW TO RUN
----------
  python exercise_02_tools.py

LIBRARIES NEEDED
-----------------
  None (pure Python)
"""

import math

print("=" * 65)
print("EXERCISE 02: Build and Register Custom Tools")
print("=" * 65)


# ===========================================================================
# EXERCISE 1: Temperature Converter Tool
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 1: temperature_converter tool")
print("=" * 65)

print("""
TODO: Write a function called temperature_converter().

  Signature: temperature_converter(value: float, direction: str) -> str

  direction can be:
    "c_to_f"  -> Celsius to Fahrenheit (formula: (C * 9/5) + 32)
    "f_to_c"  -> Fahrenheit to Celsius (formula: (F - 32) * 5/9)
    "c_to_k"  -> Celsius to Kelvin     (formula: C + 273.15)

  Should return a formatted string like: "100 C = 212.00 F"
  If direction is unknown, return: "Unknown direction: 'xyz'"

HINT:
  Use a dict to map direction strings to lambda functions.
  Example:
    conversions = {
        "c_to_f": lambda v: v * 9/5 + 32,
        ...
    }
""")

def temperature_converter(value: float, direction: str) -> str:
    """
    Converts temperature between Celsius, Fahrenheit, and Kelvin.
    TODO: Implement this function.
    """
    pass  # DELETE THIS and write your implementation


# Test your implementation
print("Testing temperature_converter:")
tests = [
    (100.0, "c_to_f"),     # Expected: "100.0 C = 212.00 F"
    (32.0,  "f_to_c"),     # Expected: "32.0 F = 0.00 C"
    (0.0,   "c_to_k"),     # Expected: "0.0 C = 273.15 K"
    (0.0,   "unknown"),    # Expected: "Unknown direction: 'unknown'"
]

for value, direction in tests:
    if temperature_converter is not None:
        result = temperature_converter(value, direction)
        print(f"  temperature_converter({value}, '{direction}') -> {result}")


# ===========================================================================
# EXERCISE 2: Word Counter Tool
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 2: word_counter tool")
print("=" * 65)

print("""
TODO: Write a function called word_counter().

  Signature: word_counter(text: str, count_type: str) -> str

  count_type can be:
    "words"      -> count number of words
    "chars"      -> count number of characters (including spaces)
    "chars_no_space" -> count chars excluding spaces
    "sentences"  -> count number of sentences (split by "." "!" "?")
    "unique"     -> count unique words (case-insensitive)

  Should return a formatted string like: "Word count: 15"

HINT:
  words = text.split()
  sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
  unique = set(text.lower().split())
""")

def word_counter(text: str, count_type: str) -> str:
    """
    Counts various text metrics.
    TODO: Implement this function.
    """
    pass  # DELETE THIS and write your implementation


# Test your implementation
print("Testing word_counter:")
sample = "Hello world. How are you? I am fine! Python is great."

test_counts = [
    ("words",          "Word count: 10"),
    ("chars",          "Character count: 53"),
    ("chars_no_space", "Character count (no spaces): 44"),
    ("sentences",      "Sentence count: 4"),
    ("unique",         "Unique words: 10"),
]

for count_type, expected_hint in test_counts:
    if word_counter is not None:
        result = word_counter(sample, count_type)
        print(f"  word_counter(text, '{count_type}') -> {result}  (expected ~{expected_hint})")


# ===========================================================================
# EXERCISE 3: Smart Tool Registry (Auto-Select)
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 3: Smart Tool Registry with Auto-Selection")
print("=" * 65)

print("""
TODO:
  1. Build a TOOLS dictionary with these tools:
       "calculator":           (use the one below)
       "temperature_converter": (from Exercise 1)
       "word_counter":         (from Exercise 2)
       "get_fact":             (use the one below)

  2. Write a function pick_tool(question: str) -> tuple[str, dict]
     It should return: (tool_name, args_dict)
     Rules:
       - If question contains "temperature", "celsius", "fahrenheit", "kelvin": use temperature_converter
       - If question contains "words", "count", "characters", "sentences": use word_counter
       - If question contains math operators (+, -, *, /, ^, sqrt, **): use calculator
       - Otherwise: use get_fact

  3. Run the provided test questions and verify the right tool is selected each time
""")

def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    try:
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        return str(eval(expression, {"__builtins__": {}}, safe_env))
    except Exception as e:
        return f"Error: {e}"

def get_fact(topic: str) -> str:
    """Returns a fact about a programming language."""
    facts = {
        "python": "Python: created by Guido van Rossum, 1991.",
        "java":   "Java: created by James Gosling, 1995.",
        "csharp": "C#: created by Anders Hejlsberg at Microsoft, 2000.",
    }
    return facts.get(topic.lower(), f"No fact found for '{topic}'.")


# TODO: Build the tools dictionary
# TOOLS_EX3 = {
#     "calculator":            calculator,
#     "temperature_converter": temperature_converter,
#     "word_counter":          word_counter,
#     "get_fact":              get_fact,
# }

def pick_tool(question: str) -> tuple:
    """
    TODO: Implement this function.
    Analyzes the question and returns (tool_name, args_dict).

    HINT: Use 'if ... in question.lower():' checks
    For temperature: extract the number and direction from the question
    For word_counter: extract the text to count
    For calculator: extract the math expression
    For get_fact: use the last word of the question as the topic
    """
    pass  # DELETE THIS and write your implementation


# Test questions
test_questions = [
    "Convert 100 celsius to fahrenheit",      # -> temperature_converter
    "How many words are in this text: Hello world my name is Gourav",  # -> word_counter
    "Calculate sqrt(256)",                     # -> calculator
    "Tell me about Python",                    # -> get_fact
    "What is 37 celsius in fahrenheit?",       # -> temperature_converter
]

print("\nTesting pick_tool():")
for question in test_questions:
    if pick_tool is not None:
        result = pick_tool(question)
        print(f"\n  Q: {question}")
        print(f"     pick_tool() returned: {result}")


# ===========================================================================
# EXERCISE 4: Error Handling
# ===========================================================================

print("\n" + "=" * 65)
print("EXERCISE 4: Tool Error Handling")
print("=" * 65)

print("""
TODO: Write a function called safe_tool_call() that:
  1. Takes: tool_name (str), args (dict), tools (dict)
  2. Tries to call tools[tool_name](**args)
  3. If the tool is not found, returns: "Tool 'name' not found."
  4. If the tool raises an exception, returns: "Tool error: <error message>"
  5. Otherwise, returns the tool result

This is how production agents handle tool failures gracefully.
""")

def safe_tool_call(tool_name: str, args: dict, tools: dict) -> str:
    """
    TODO: Implement this function.
    Safely calls a tool and handles all possible errors.
    """
    pass  # DELETE THIS and write your implementation


# Test error handling
print("Testing safe_tool_call():")

test_tools = {"calculator": calculator}

call_tests = [
    ("calculator", {"expression": "sqrt(144)"},  "Should work: returns '12.0'"),
    ("calculator", {"expression": "1/0"},         "Should return: 'Tool error: ...'"),
    ("unknown",    {"expression": "2+2"},          "Should return: 'Tool not found'"),
    ("calculator", {"expression": "invalid"},      "Should return: error message"),
]

for tool, args, note in call_tests:
    if safe_tool_call is not None:
        result = safe_tool_call(tool, args, test_tools)
        print(f"\n  {note}")
        print(f"  safe_tool_call('{tool}', {args}) -> {result}")


# ===========================================================================
# SOLUTION (uncomment to check)
# ===========================================================================

print("\n" + "=" * 65)
print("SOLUTION (uncomment to check)")
print("=" * 65)

"""
SOLUTION FOR EXERCISE 1:

    def temperature_converter(value: float, direction: str) -> str:
        conversions = {
            "c_to_f": (lambda v: v * 9/5 + 32,  "C", "F"),
            "f_to_c": (lambda v: (v - 32) * 5/9, "F", "C"),
            "c_to_k": (lambda v: v + 273.15,      "C", "K"),
        }
        if direction not in conversions:
            return f"Unknown direction: '{direction}'"
        fn, from_unit, to_unit = conversions[direction]
        result = fn(value)
        return f"{value} {from_unit} = {result:.2f} {to_unit}"

SOLUTION FOR EXERCISE 2:

    def word_counter(text: str, count_type: str) -> str:
        if count_type == "words":
            return f"Word count: {len(text.split())}"
        elif count_type == "chars":
            return f"Character count: {len(text)}"
        elif count_type == "chars_no_space":
            return f"Character count (no spaces): {len(text.replace(' ', ''))}"
        elif count_type == "sentences":
            sentences = [s.strip() for s in text.replace("!",".").replace("?",".").split(".") if s.strip()]
            return f"Sentence count: {len(sentences)}"
        elif count_type == "unique":
            unique = set(text.lower().split())
            return f"Unique words: {len(unique)}"
        return f"Unknown count_type: '{count_type}'"

SOLUTION FOR EXERCISE 3:

    TOOLS_EX3 = {
        "calculator": calculator,
        "temperature_converter": temperature_converter,
        "word_counter": word_counter,
        "get_fact": get_fact,
    }

    def pick_tool(question: str) -> tuple:
        q = question.lower()
        if any(w in q for w in ["celsius", "fahrenheit", "kelvin", "temperature"]):
            import re
            numbers = re.findall(r'[-0-9.]+', question)
            value = float(numbers[0]) if numbers else 0.0
            if "celsius" in q and "fahrenheit" in q: direction = "c_to_f"
            elif "fahrenheit" in q and "celsius" in q: direction = "f_to_c"
            elif "celsius" in q and "kelvin" in q: direction = "c_to_k"
            elif "celsius" in q: direction = "c_to_f"
            else: direction = "c_to_f"
            return ("temperature_converter", {"value": value, "direction": direction})
        elif any(w in q for w in ["words", "count", "characters", "sentences", "unique"]):
            text_part = question.split(":", 1)[-1].strip() if ":" in question else question
            count_type = "words"
            for ct in ["chars_no_space", "chars", "sentences", "unique", "words"]:
                if ct.replace("_", " ") in q or ct in q:
                    count_type = ct
                    break
            return ("word_counter", {"text": text_part, "count_type": count_type})
        elif any(c in q for c in ["+", "-", "*", "/", "^", "sqrt", "**", "calculate"]):
            expr = q.split("calculate")[-1].strip() if "calculate" in q else q
            return ("calculator", {"expression": expr})
        else:
            topic = question.split()[-1].rstrip("?.,!").lower()
            return ("get_fact", {"topic": topic})

SOLUTION FOR EXERCISE 4:

    def safe_tool_call(tool_name: str, args: dict, tools: dict) -> str:
        if tool_name not in tools:
            return f"Tool '{tool_name}' not found. Available: {list(tools.keys())}"
        try:
            return tools[tool_name](**args)
        except Exception as e:
            return f"Tool error: {e}"
"""

print("See SOLUTION block above (uncomment to check your work).")
print("=" * 65)
print("END OF EXERCISE 02")
print("=" * 65)

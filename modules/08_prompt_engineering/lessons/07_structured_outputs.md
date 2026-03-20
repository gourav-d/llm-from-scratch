# Lesson 7: Structured Outputs

**Get reliable, machine-readable responses from LLMs every time**

---

## 🎯 Learning Objectives

After this lesson, you will:
- Generate structured outputs (JSON, XML, CSV)
- Use function calling APIs
- Validate and parse LLM responses
- Handle schema evolution
- Build production-ready structured prompts

**Time:** 75 minutes

---

## 📖 Why Structured Outputs?

### The Problem: Unstructured Responses

**Without structure:**
```
User: "Extract customer info from this email"
LLM: "The customer's name is John Smith and he works at Acme Corp.
      His email is john@acme.com and he mentioned being interested in
      our enterprise plan..."
❌ Hard to parse, unreliable format
```

**With structure:**
```
User: "Extract customer info as JSON"
LLM: {
  "name": "John Smith",
  "company": "Acme Corp",
  "email": "john@acme.com",
  "interest": "enterprise plan"
}
✅ Machine-readable, consistent, reliable
```

### C#/.NET Analogy

```csharp
// Unstructured (like string output)
string result = "Name: John, Age: 30, City: NYC";
// ❌ Need to parse with regex, error-prone

// Structured (like strongly-typed object)
var result = new Customer {
    Name = "John",
    Age = 30,
    City = "NYC"
};
// ✅ Type-safe, compile-time checks
```

---

## 📋 Common Output Formats

### 1. JSON (Most Popular)

**Use when:** Need structured data for APIs, databases, frontend

```
Return as JSON:
{
  "field1": "value",
  "field2": 123,
  "nested": {
    "key": "value"
  },
  "array": ["item1", "item2"]
}
```

**Advantages:**
- Universal format
- Easy to parse
- Supports nesting
- Type-safe

### 2. XML

**Use when:** Legacy systems, specific industry standards

```
Return as XML:
<customer>
  <name>John Smith</name>
  <email>john@example.com</email>
  <orders>
    <order id="1">Product A</order>
    <order id="2">Product B</order>
  </orders>
</customer>
```

### 3. CSV

**Use when:** Tabular data, Excel import, data analysis

```
Return as CSV:
name,email,order_count,total_value
John Smith,john@example.com,5,1250.00
Jane Doe,jane@example.com,3,890.50
```

### 4. Markdown Tables

**Use when:** Human-readable but structured

```
Return as Markdown table:
| Name | Email | Orders |
|------|-------|--------|
| John | john@example.com | 5 |
| Jane | jane@example.com | 3 |
```

### 5. YAML

**Use when:** Configuration files, human-readable structured data

```
Return as YAML:
customer:
  name: John Smith
  email: john@example.com
  orders:
    - product: Product A
      quantity: 2
    - product: Product B
      quantity: 1
```

---

## 🎯 JSON Output Patterns

### Pattern 1: Simple JSON

```
Extract information from this text as JSON.

Required fields:
- name (string)
- email (string)
- phone (string or null)

Return ONLY valid JSON, no additional text.

Text: [input]
```

### Pattern 2: JSON with Schema

```
Extract data according to this JSON schema:

{
  "type": "object",
  "required": ["name", "email"],
  "properties": {
    "name": {"type": "string"},
    "email": {"type": "string", "format": "email"},
    "phone": {"type": "string", "pattern": "^\\+?[1-9]\\d{1,14}$"},
    "age": {"type": "integer", "minimum": 0, "maximum": 150"}
  }
}

Text: [input]

Return JSON matching this schema exactly.
```

### Pattern 3: Nested JSON

```
Analyze this product review and return structured JSON:

{
  "product": {
    "name": "...",
    "category": "..."
  },
  "review": {
    "rating": 1-5,
    "sentiment": "positive|negative|neutral",
    "aspects": [
      {
        "aspect": "quality|price|service",
        "sentiment": "positive|negative",
        "quote": "exact quote from review"
      }
    ]
  },
  "customer": {
    "verified_purchase": true|false,
    "helpful_votes": number
  }
}
```

---

## 🔧 Best Practices for Structured Outputs

### 1. Be Explicit About Format

❌ **Vague:**
```
Return the data in a structured format.
```

✅ **Explicit:**
```
Return ONLY valid JSON. No markdown formatting, no additional text before or after.
Start with { and end with }.
```

### 2. Provide Example Output

```
Extract customer data as JSON following this exact format:

Example:
{
  "name": "John Smith",
  "email": "john@example.com",
  "interested_in": ["product A", "product B"]
}

Now extract from:
[actual input]
```

### 3. Specify Data Types

```
Return JSON with these exact types:
- id: integer
- name: string
- price: float (two decimal places)
- in_stock: boolean
- tags: array of strings
- created_at: ISO 8601 date string
```

### 4. Handle Missing Data

```
If a field is not found in the input:
- Use null for optional fields
- Use appropriate default values
- Do NOT omit required fields

Example:
{
  "name": "John Smith",
  "email": null,  ← not found
  "phone": null   ← not found
}
```

---

## ⚡ Function Calling (OpenAI/Anthropic)

### What is Function Calling?

Structured way to get LLMs to fill in function parameters.

**C#/.NET Analogy:**
```csharp
// You define a function signature
public void SendEmail(string to, string subject, string body) { }

// LLM fills in the parameters based on user input
SendEmail(
    to: "john@example.com",
    subject: "Meeting Reminder",
    body: "Don't forget our 2pm meeting"
)
```

### OpenAI Function Calling

```python
import openai

# Define function schema
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. San Francisco"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
]

# Call LLM
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What's the weather in London in Celsius?"}
    ],
    functions=functions,
    function_call="auto"
)

# LLM returns structured function call:
# {
#     "name": "get_weather",
#     "arguments": {
#         "location": "London",
#         "unit": "celsius"
#     }
# }
```

### Anthropic Tool Use

```python
from anthropic import Anthropic

client = Anthropic()

tools = [
    {
        "name": "get_stock_price",
        "description": "Get current stock price for a symbol",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                }
            },
            "required": ["symbol"]
        }
    }
]

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's Apple's stock price?"}
    ]
)

# Claude returns structured tool use:
# {
#     "type": "tool_use",
#     "name": "get_stock_price",
#     "input": {"symbol": "AAPL"}
# }
```

---

## 🛡️ Validation & Error Handling

### Pattern 1: JSON Validation

```python
import json
from jsonschema import validate, ValidationError

def parse_llm_json(response: str, schema: dict) -> dict:
    """
    Parse and validate LLM JSON response.

    C#/.NET: Like JSON deserialization with validation
    var obj = JsonSerializer.Deserialize<Customer>(json);
    """
    try:
        # Extract JSON (handle markdown code blocks)
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:-3]  # Remove ```json and ```
        elif json_str.startswith("```"):
            json_str = json_str[3:-3]

        # Parse JSON
        data = json.loads(json_str)

        # Validate against schema
        validate(instance=data, schema=schema)

        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}")
```

### Pattern 2: Retry with Feedback

```python
def get_structured_response_with_retry(
    prompt: str,
    schema: dict,
    max_retries: int = 3
) -> dict:
    """
    Retry if validation fails, providing feedback to LLM.

    Like C# retry logic with Polly library.
    """
    for attempt in range(max_retries):
        response = call_llm(prompt)

        try:
            return parse_llm_json(response, schema)
        except ValueError as e:
            if attempt < max_retries - 1:
                # Add error feedback to prompt
                prompt += f"\n\nPrevious response was invalid: {e}\nPlease correct and return valid JSON."
            else:
                raise

    raise ValueError("Failed after max retries")
```

### Pattern 3: Fallback to Parsing

```python
import re

def extract_json_fallback(response: str) -> dict:
    """
    Fallback parser if LLM includes extra text.

    Finds JSON in the response even if surrounded by other text.
    """
    # Try to find JSON object
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json.loads(json_str)

    # Try to find JSON array
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json.loads(json_str)

    raise ValueError("No JSON found in response")
```

---

## 💡 Real-World Examples

### Example 1: Customer Data Extraction

```python
CUSTOMER_SCHEMA = {
    "type": "object",
    "required": ["name", "email"],
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string", "format": "email"},
        "company": {"type": "string"},
        "phone": {"type": ["string", "null"]},
        "interests": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}

prompt = f"""
Extract customer information from this email and return as JSON.

JSON Schema:
{json.dumps(CUSTOMER_SCHEMA, indent=2)}

Rules:
- email MUST be valid email format
- interests should be array of strings
- phone can be null if not found
- Return ONLY valid JSON

Email:
---
Hi, I'm John Smith from Acme Corp. You can reach me at john.smith@acme.com.
I'm interested in your Enterprise plan and API access.
---

JSON:
"""
```

### Example 2: Multi-Record Extraction

```python
prompt = """
Extract all products mentioned in this description as a JSON array.

Format:
[
  {
    "name": "product name",
    "price": numeric price,
    "category": "category",
    "in_stock": true|false
  }
]

Description:
Our store offers the iPhone 15 Pro for $999 in the Electronics category (in stock),
the Samsung Galaxy S24 for $899 in Electronics (out of stock),
and the Sony WH-1000XM5 headphones for $399 in Audio (in stock).

Return ONLY the JSON array, no additional text.
"""
```

### Example 3: Sentiment Analysis with Aspects

```python
prompt = """
Analyze this product review and return structured sentiment analysis:

{
  "overall_rating": 1-5,
  "overall_sentiment": "positive|negative|neutral|mixed",
  "aspects": [
    {
      "aspect": "quality|price|service|delivery|etc",
      "sentiment": "positive|negative|neutral",
      "score": 1-5,
      "supporting_quote": "exact quote from review"
    }
  ],
  "summary": "one sentence summary",
  "would_recommend": true|false
}

Review:
"The product quality is excellent and worth every penny. However, the delivery
took 3 weeks which was frustrating. Customer service was helpful when I called
to inquire. Overall, I'm satisfied with my purchase."

Return JSON only:
"""
```

---

## 🚀 Production Patterns

### Pattern 1: Typed Responses (Python)

```python
from typing import List, Optional
from pydantic import BaseModel, EmailStr, field_validator

class Customer(BaseModel):
    """
    Strongly-typed customer model.

    C#/.NET equivalent:
    public class Customer
    {
        [Required]
        public string Name { get; set; }

        [EmailAddress]
        public string Email { get; set; }
    }
    """
    name: str
    email: EmailStr
    company: Optional[str] = None
    phone: Optional[str] = None
    interests: List[str] = []

    @field_validator('phone')
    def validate_phone(cls, v):
        if v and not re.match(r'^\+?[1-9]\d{1,14}$', v):
            raise ValueError('Invalid phone format')
        return v

# Usage:
def extract_customer(email_text: str) -> Customer:
    """Extract customer info with type safety."""
    prompt = create_extraction_prompt(email_text, Customer)
    response = call_llm(prompt)
    json_data = parse_llm_json(response)

    # Pydantic validates and parses
    return Customer(**json_data)
```

### Pattern 2: Schema Evolution

```python
class SchemaVersions:
    """
    Manage schema versions for backwards compatibility.

    Like C# API versioning.
    """

    CUSTOMER_V1 = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"}
        }
    }

    CUSTOMER_V2 = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "company": {"type": "string"},  # Added in v2
            "phone": {"type": "string"}     # Added in v2
        }
    }

    @staticmethod
    def migrate_v1_to_v2(v1_data: dict) -> dict:
        """Migrate old data to new schema."""
        return {
            **v1_data,
            "company": None,
            "phone": None
        }
```

### Pattern 3: Streaming Structured Outputs

```python
import json

def stream_json_objects(prompt: str):
    """
    Stream JSON objects as they're generated.

    Useful for large arrays of results.
    """
    buffer = ""

    for chunk in call_llm_stream(prompt):
        buffer += chunk

        # Try to extract complete JSON objects
        while '{' in buffer and '}' in buffer:
            start = buffer.index('{')
            end = buffer.index('}', start) + 1

            try:
                obj = json.loads(buffer[start:end])
                yield obj  # Successfully parsed object
                buffer = buffer[end:]
            except json.JSONDecodeError:
                # Incomplete object, keep buffering
                break
```

---

## ✅ Summary

### Key Takeaways

1. **Structured = Machine-Readable**
   - JSON, XML, CSV for reliability
   - Like strongly-typed objects in C#

2. **Be Explicit**
   - Specify exact format
   - Provide schemas and examples
   - Handle missing data

3. **Function Calling**
   - OpenAI and Anthropic support it
   - Best for API integration
   - Type-safe parameter extraction

4. **Validation is Critical**
   - Always validate LLM output
   - Retry with feedback on errors
   - Use typed models (Pydantic)

5. **Production Patterns**
   - Schema versioning
   - Error handling
   - Type safety

### When to Use What?

| Use Case | Format | Why |
|----------|--------|-----|
| API responses | JSON | Standard, parseable |
| Data export | CSV | Excel, analysis |
| Documentation | Markdown | Human + machine |
| Config files | YAML | Readable structure |
| Legacy systems | XML | Industry standard |

---

## 📝 Practice Exercises

1. **Create extraction prompts:**
   - JSON for contact info
   - CSV for product catalog
   - XML for orders

2. **Build validation:**
   - JSON schema validator
   - Retry logic with feedback
   - Typed models with Pydantic

3. **Function calling:**
   - Define function schemas
   - Test with OpenAI/Anthropic
   - Handle responses

4. **Production system:**
   - Schema versioning
   - Migration logic
   - Full validation pipeline

---

**Next Lesson:** Lesson 8 - Prompt Optimization

**Estimated time:** 60 minutes

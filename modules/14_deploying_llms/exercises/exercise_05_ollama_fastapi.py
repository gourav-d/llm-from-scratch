# =============================================================================
# Module 14 - Deploying LLMs
# Exercise 05: Ollama + FastAPI
# =============================================================================
#
# INSTRUCTIONS:
#   Complete each TODO section.
#   Task 1-2: pip install requests  (+ Ollama running)
#   Task 3-5: pip install fastapi uvicorn pydantic ollama
#
# WHAT YOU PRACTICE:
#   - Calling Ollama with raw HTTP requests
#   - Building a FastAPI server with Pydantic validation
#   - Adding streaming responses to a FastAPI endpoint
#   - Writing a rate limiter
#   - Putting it all together: full local LLM API
#
# HOW TO TEST YOUR FastAPI SERVER:
#   1. Run: uvicorn exercise_05_ollama_fastapi:app --reload --port 8001
#   2. Open: http://localhost:8001/docs  (Swagger UI -- test live!)
#   3. Or use curl / requests from another terminal
#
# =============================================================================

import json
import time
import os
from collections import defaultdict

print("=" * 60)
print("Exercise 05: Ollama + FastAPI")
print("=" * 60)

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:1b"   # Change to a model you have pulled

# =============================================================================
# TASK 1: Build an Ollama Client Class
#
# Wrap the Ollama HTTP API in a clean Python class.
# The class should handle: connection errors, model-not-found errors, timeouts.
# =============================================================================

print("\n--- Task 1: OllamaClient Class ---")

class OllamaClient:
    """
    A clean wrapper around Ollama's HTTP API.

    Usage:
        client = OllamaClient()
        reply = client.chat("What is Python?")
        print(reply)
    """

    def __init__(self, base_url=OLLAMA_URL, default_model=DEFAULT_MODEL):
        self.base_url = base_url
        self.default_model = default_model

    def is_running(self):
        """
        Check if Ollama server is running.
        Returns True if reachable, False otherwise.

        TODO: Make a GET request to {self.base_url}/api/tags with timeout=2.
        Return True if status_code == 200, False on any exception.
        """
        try:
            import requests
            # TODO: GET request to /api/tags
            response = None   # TODO
            return response is not None and response.status_code == 200
        except Exception:
            return False

    def list_models(self):
        """
        Return list of downloaded model names.

        TODO: GET /api/tags, parse JSON, return list of name strings.
        Returns [] if Ollama is not running.
        """
        try:
            import requests
            response = None   # TODO: GET {self.base_url}/api/tags
            if response is None or response.status_code != 200:
                return []
            data = response.json()
            # TODO: extract and return list of model names
            # Hint: [m["name"] for m in data.get("models", [])]
            return []   # TODO
        except Exception:
            return []

    def chat(self, message, model=None, system=None, temperature=0.7, max_tokens=200):
        """
        Send a single message to Ollama and return the reply text.

        Parameters:
            message:     str -- the user's message
            model:       str -- model name (uses self.default_model if None)
            system:      str -- optional system prompt
            temperature: float -- 0.0 to 2.0
            max_tokens:  int -- maximum tokens to generate

        Returns:
            str -- the model's reply, or raises RuntimeError on failure

        TODO: Build the messages list, POST to /api/chat, return reply text.
        """
        import requests

        model = model or self.default_model   # Use default if not specified

        # TODO: Build messages list (add system message if provided, then user message)
        messages = []
        if system:
            pass   # TODO: append {"role": "system", "content": system}
        # TODO: append {"role": "user", "content": message}

        # TODO: POST to {self.base_url}/api/chat with the request body
        # Body: {"model": model, "messages": messages, "stream": False,
        #        "options": {"temperature": temperature, "num_predict": max_tokens}}
        try:
            response = None   # TODO
        except Exception as e:
            raise RuntimeError(f"Cannot reach Ollama at {self.base_url}: {e}")

        if response is None or response.status_code != 200:
            status = response.status_code if response else "no response"
            raise RuntimeError(f"Ollama returned error: HTTP {status}")

        # TODO: parse and return the reply text
        # Hint: response.json()["message"]["content"]
        return None   # TODO

    def stream_chat(self, message, model=None, temperature=0.7):
        """
        Stream a response token by token.

        Yields each token string as it arrives.
        The caller prints or processes each token.

        TODO: POST to /api/chat with stream=True.
        Iterate over response lines, parse each JSON chunk, yield the token.

        Hint: use response.iter_lines() for streaming HTTP.
        Each line is a JSON object: {"message": {"content": "token"}, ...}
        The last line has "done": true.
        """
        import requests

        model = model or self.default_model
        messages = [{"role": "user", "content": message}]

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": True,
                      "options": {"temperature": temperature}},
                stream=True,    # Tell requests: don't buffer the body
                timeout=60
            )
        except Exception as e:
            raise RuntimeError(f"Cannot reach Ollama: {e}")

        for line in response.iter_lines():
            if line:
                try:
                    # TODO: parse the JSON line
                    chunk = None   # Hint: json.loads(line.decode("utf-8"))
                    if chunk is None:
                        continue

                    # TODO: get the token text
                    token = None   # Hint: chunk.get("message", {}).get("content", "")

                    # TODO: yield the token if it is not empty
                    # if token: yield token

                    # Check if we are done
                    if chunk.get("done", False):
                        break
                except Exception:
                    continue


# Test OllamaClient
client = OllamaClient()
print(f"Ollama running: {client.is_running()}")
models = client.list_models()
print(f"Available models: {models if models else '(none -- Ollama not running)'}")

if client.is_running() and models:
    try:
        print("\nTesting chat():")
        reply = client.chat("What is 1+1? Answer in one word.", max_tokens=10)
        print(f"Reply: {reply}")

        print("\nTesting stream_chat():")
        print("Stream: ", end="")
        for token in client.stream_chat("Say hello.", temperature=0.1):
            print(token, end="", flush=True)
        print()
    except RuntimeError as e:
        print(f"Error: {e}")

# =============================================================================
# TASK 2: Prompt Template Builder
#
# Different models need different prompt formats.
# Build a function that wraps a user message in the correct template.
# =============================================================================

print("\n--- Task 2: Prompt Template Builder ---")

def build_prompt(user_message, system_message=None, template="chatml"):
    """
    Wrap a user message in a model-specific prompt template.

    Supported templates:
        "chatml" (Phi, Qwen, many models):
            <|im_start|>system
            {system}<|im_end|>
            <|im_start|>user
            {user}<|im_end|>
            <|im_start|>assistant

        "llama2" (LLaMA-2-Chat):
            <s>[INST] <<SYS>>
            {system}
            <</SYS>>
            {user} [/INST]

        "mistral" (Mistral-Instruct, no system prompt support):
            <s>[INST] {user} [/INST]

        "plain" (raw models, no special tokens):
            User: {user}
            Assistant:

    Parameters:
        user_message:   str -- the user's question
        system_message: str or None -- system instructions
        template:       str -- one of "chatml", "llama2", "mistral", "plain"

    Returns:
        str -- the formatted prompt string

    TODO: Implement all 4 templates.
    """
    system = system_message or "You are a helpful assistant."

    if template == "chatml":
        # TODO: build ChatML format
        return None   # TODO

    elif template == "llama2":
        # TODO: build LLaMA-2 format
        return None   # TODO

    elif template == "mistral":
        # NOTE: Mistral-Instruct does not support system prompts in raw mode
        # Just wrap the user message
        return None   # TODO

    elif template == "plain":
        return None   # TODO

    else:
        raise ValueError(f"Unknown template: {template}")


# Test all templates
test_msg = "What is Python?"
test_sys = "You are a concise assistant."

for tmpl in ["chatml", "llama2", "mistral", "plain"]:
    result = build_prompt(test_msg, test_sys, tmpl)
    print(f"\nTemplate '{tmpl}':")
    if result:
        print(result)
    else:
        print("  TODO not complete yet.")

# =============================================================================
# TASK 3: FastAPI Server with Pydantic Validation
#
# Build a FastAPI server with:
#   - GET  /health   -- returns {"status": "ok"}
#   - POST /chat     -- accepts ChatRequest, returns ChatResponse
#   - Pydantic validation on all fields
# =============================================================================

print("\n--- Task 3: FastAPI Server ---")

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from typing import Optional
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Install: pip install fastapi uvicorn")

if FASTAPI_AVAILABLE:

    # TODO: Define ChatRequest using Pydantic BaseModel
    # Fields:
    #   message:        str, required, min_length=1, max_length=5000
    #   model:          str, default=DEFAULT_MODEL
    #   system_prompt:  Optional[str], default=None
    #   temperature:    float, default=0.7, ge=0.0, le=2.0
    #   max_tokens:     int, default=200, ge=1, le=2000
    class ChatRequest(BaseModel):
        # TODO: add all fields
        message: str = Field(..., min_length=1)   # Start here, add the rest
        # model: ...
        # system_prompt: ...
        # temperature: ...
        # max_tokens: ...
        pass


    # TODO: Define ChatResponse using Pydantic BaseModel
    # Fields:
    #   reply:       str -- the model's response
    #   model:       str -- which model was used
    #   latency_ms:  float -- how long the request took
    class ChatResponse(BaseModel):
        # TODO: add all fields
        pass


    # Create the FastAPI app
    # This 'app' variable is what uvicorn looks for when you run:
    # uvicorn exercise_05_ollama_fastapi:app --reload
    app = FastAPI(
        title="LLM Exercise API",
        description="Exercise 05: Local LLM API with FastAPI + Ollama",
        version="1.0.0"
    )


    @app.get("/health")
    def health():
        """
        Health check endpoint.
        TODO: Return {"status": "ok", "ollama": "connected"} if Ollama is running,
              or {"status": "ok", "ollama": "not available"} if not.
        """
        # TODO: check if ollama is running (use client.is_running())
        return {"status": "ok"}   # TODO: improve this


    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest):
        """
        Chat with the local LLM.
        FastAPI automatically validates the request body using ChatRequest.
        FastAPI automatically serializes the return value using ChatResponse.

        TODO: Use OllamaClient to call the model and return a ChatResponse.
        """
        start = time.perf_counter()

        # TODO: call client.chat() with the request fields
        # Handle the case where Ollama is not available (raise HTTPException 503)
        try:
            reply = None   # TODO: client.chat(request.message, ...)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        elapsed_ms = (time.perf_counter() - start) * 1000

        # TODO: return a ChatResponse with reply, model, latency_ms
        return ChatResponse(
            reply=reply or "TODO: implement",
            model=DEFAULT_MODEL,
            latency_ms=round(elapsed_ms, 2)
        )

    print("FastAPI app created!")
    print("To run: uvicorn exercise_05_ollama_fastapi:app --reload --port 8001")
    print("Then visit: http://localhost:8001/docs")

else:
    print("Showing expected ChatRequest schema:")
    print("  message:       str (required, 1-5000 chars)")
    print("  model:         str (default: llama3.2:1b)")
    print("  system_prompt: str or None (optional)")
    print("  temperature:   float (0.0-2.0, default 0.7)")
    print("  max_tokens:    int (1-2000, default 200)")

# =============================================================================
# TASK 4: Rate Limiter
#
# Complete the rate limiter function below.
# It should allow MAX_REQUESTS requests per WINDOW_SECONDS from each IP.
# If the limit is exceeded, raise HTTP 429.
# =============================================================================

print("\n--- Task 4: Rate Limiter ---")

request_timestamps = defaultdict(list)   # ip -> [timestamp, ...]
MAX_REQUESTS = 5
WINDOW_SECONDS = 10

def rate_limit(client_ip: str, max_req: int = MAX_REQUESTS, window: int = WINDOW_SECONDS):
    """
    Enforce a sliding window rate limit.

    Steps:
      1. Get current time (time.time())
      2. Remove old timestamps that are outside the window
         (keep only timestamps where timestamp > now - window)
      3. If count >= max_req, raise HTTPException(status_code=429, ...)
      4. Append current timestamp to the list for this IP

    Parameters:
        client_ip: str -- the client's IP address
        max_req:   int -- max requests per window
        window:    int -- window duration in seconds

    Raises:
        HTTPException(429) if rate limit exceeded
        (import HTTPException from fastapi for this to work)
    """
    now = time.time()

    # TODO: Step 2 -- remove timestamps outside the window
    request_timestamps[client_ip] = None   # TODO: filter list

    # TODO: Step 3 -- check if over limit
    # if len(...) >= max_req:
    #     raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # TODO: Step 4 -- record this request
    # request_timestamps[client_ip].append(now)


# Test the rate limiter (without HTTP, just the logic)
def test_rate_limiter():
    """Simulate rapid requests from one IP and verify rate limiting works."""
    test_ip = "192.168.1.1"
    request_timestamps.clear()   # Reset for clean test

    print(f"\nRate limit: {MAX_REQUESTS} requests per {WINDOW_SECONDS} seconds")

    blocked_at = None
    for i in range(MAX_REQUESTS + 3):
        try:
            rate_limit(test_ip)
            count = len(request_timestamps[test_ip])
            print(f"  Request {i+1}: ALLOWED (count={count})")
        except Exception as e:
            if blocked_at is None:
                blocked_at = i + 1
            print(f"  Request {i+1}: BLOCKED ({type(e).__name__})")

    if blocked_at == MAX_REQUESTS + 1:
        print(f"Correct! Blocked starting at request {blocked_at}.")
    else:
        print(f"TODO not complete yet (should block at request {MAX_REQUESTS + 1}).")


test_rate_limiter()

# =============================================================================
# TASK 5: Streaming Endpoint (Conceptual)
#
# Write the FastAPI streaming endpoint code (no execution needed).
# Fill in the blanks in the code below.
# =============================================================================

print("\n--- Task 5: Streaming Endpoint ---")

streaming_code = """
from fastapi.responses import StreamingResponse
import json

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    '''Stream tokens one by one as the model generates them.'''

    def generate():
        '''Generator: yields Server-Sent Events (SSE) format.'''
        try:
            # Call OllamaClient.stream_chat() to get tokens
            # TODO: iterate over client.stream_chat(request.message, ...)
            for token in ____:
                if token:
                    # TODO: format each token as SSE
                    # SSE format: "data: {json_string}\\n\\n"
                    data = ____   # json.dumps({"token": token})
                    yield f"____"    # f"data: {data}\\n\\n"

            # TODO: signal end of stream
            yield ____    # "data: [DONE]\\n\\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\\n\\n"
            yield "data: [DONE]\\n\\n"

    return StreamingResponse(
        generate(),
        media_type="____",    # "text/event-stream"
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
"""

print(streaming_code)

print("Fill in the blanks:")
print("  Blank 1 (iterate):  client.stream_chat(request.message, request.model)")
print("  Blank 2 (data):     json.dumps({'token': token})")
print("  Blank 3 (yield):    f'data: {data}\\n\\n'")
print("  Blank 4 (done):     'data: [DONE]\\n\\n'")
print("  Blank 5 (type):     'text/event-stream'")

# =============================================================================
# HINTS
# =============================================================================

print("\n" + "=" * 60)
print("HINTS")
print("=" * 60)
print("""
Task 1 (OllamaClient):
  is_running():
    response = requests.get(f"{self.base_url}/api/tags", timeout=2)
    return response.status_code == 200

  list_models():
    response = requests.get(f"{self.base_url}/api/tags", timeout=5)
    return [m["name"] for m in response.json().get("models", [])]

  chat():
    messages = []
    if system: messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    response = requests.post(f"{self.base_url}/api/chat", json={
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }, timeout=60)
    return response.json()["message"]["content"]

  stream_chat():
    chunk = json.loads(line.decode("utf-8"))
    token = chunk.get("message", {}).get("content", "")
    if token: yield token

Task 2 (prompt templates):
  chatml:
    result = ""
    if system_message:
        result += f"<|im_start|>system\\n{system}<|im_end|>\\n"
    result += f"<|im_start|>user\\n{user_message}<|im_end|>\\n<|im_start|>assistant\\n"
    return result

  llama2:
    return f"<s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n{user_message} [/INST]"

  mistral:
    return f"<s>[INST] {user_message} [/INST]"

  plain:
    return f"User: {user_message}\\nAssistant:"

Task 4 (rate limiter):
  request_timestamps[client_ip] = [
      t for t in request_timestamps[client_ip] if t > now - window
  ]
  if len(request_timestamps[client_ip]) >= max_req:
      raise HTTPException(status_code=429, detail="Rate limit exceeded")
  request_timestamps[client_ip].append(now)
""")

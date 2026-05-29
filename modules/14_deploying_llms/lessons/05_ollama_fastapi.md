# Lesson 05: Ollama + FastAPI -- Serving Your Local LLM

## Glossary (Read This First!)

| Term | Plain English Definition |
|------|--------------------------|
| **Ollama** | A tool that makes downloading and running local LLMs as simple as one command. Wraps llama.cpp internally. |
| **FastAPI** | A modern Python web framework for building HTTP APIs. Fast to write, automatically generates documentation, validates inputs. |
| **HTTP API** | An interface that lets software talk to your model over a network. Your model is the server. Any client (browser, app, Slack bot) can call it. |
| **Endpoint** | A specific URL that accepts requests. `/chat`, `/health`, `/models` are endpoints. |
| **Request body** | The JSON data sent to your API with a POST request. Contains the user's message, settings, etc. |
| **Response** | What your API sends back. Usually JSON with the model's output. |
| **Middleware** | Code that runs between receiving a request and sending a response. Used for logging, auth, CORS, etc. |
| **CORS** | Cross-Origin Resource Sharing. Browser security feature. You need to configure it if your frontend is on a different domain. |
| **Streaming response** | Sending tokens one by one as they are generated, instead of waiting for the complete response. Like ChatGPT's typing effect. |
| **Pydantic** | Python library for data validation. FastAPI uses it to validate request/response shapes. |
| **uvicorn** | An ASGI web server. Used to run FastAPI applications in production. |
| **REST** | Representational State Transfer. Architectural style for HTTP APIs. Stateless, uses standard HTTP methods (GET, POST, etc.). |
| **JSON** | JavaScript Object Notation. The standard format for API request and response data. |
| **Health check** | A simple endpoint (GET /health) that confirms your server is running. Used by load balancers and monitoring. |
| **Rate limiting** | Restricting how many requests a client can make per minute. Prevents abuse. |
| **OpenAI-compatible API** | Ollama can speak the same API as OpenAI. Any code written for OpenAI's API works with Ollama with just a URL change. |

---

## Part 1: Ollama -- The Simplest Possible LLM Deployment

```
+------------------------------------------------------------------+
|  WHAT OLLAMA IS                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  Ollama = CLI tool + local server + model library                |
|                                                                  |
|  Internally:                                                     |
|    - Downloads GGUF files from Ollama's model library            |
|    - Runs llama.cpp as the inference engine                      |
|    - Starts a local HTTP server at http://localhost:11434         |
|    - Exposes an OpenAI-compatible REST API                        |
|                                                                  |
|  What you get in 2 commands:                                     |
|    $ ollama serve          (start the server)                    |
|    $ ollama pull mistral   (download 7B model, ~4 GB)            |
|                                                                  |
|  Then ANY of these work:                                         |
|    - Chat in terminal:     ollama run mistral                    |
|    - Call from Python:     requests.post(...)                    |
|    - Call from C#:         HttpClient.PostAsync(...)             |
|    - Use OpenAI SDK:       openai.ChatCompletion.create(...)     |
|                                                                  |
+------------------------------------------------------------------+
```

### Installing Ollama

```bash
# Windows:
# Download from https://ollama.com and run the installer.
# Ollama installs as a Windows service that auto-starts.

# Mac:
# Download from https://ollama.com or:
brew install ollama

# Linux:
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation:
ollama --version

# Start the server (if not auto-started):
ollama serve
# Server runs at: http://localhost:11434
```

### Essential Ollama Commands

```bash
# Pull a model (downloads GGUF file)
ollama pull mistral          # Mistral 7B -- good general purpose
ollama pull llama3.2:1b      # LLaMA 3.2 1B -- tiny, fast
ollama pull phi3:mini        # Phi-3 mini 3.8B -- very capable for size
ollama pull codellama        # Code-specialized model

# List downloaded models
ollama list

# Run a model interactively (chat in terminal)
ollama run mistral
>>> Tell me a joke.

# Remove a model (free up disk space)
ollama rm mistral

# Show model details (architecture, context length, etc.)
ollama show mistral
```

---

## Part 2: Calling Ollama from Python

Ollama's server accepts HTTP requests. Call it from Python two ways.

### Method 1: Direct HTTP (no special library)

```python
import requests
import json

OLLAMA_URL = "http://localhost:11434"

def chat(message, model="mistral", system=None):
    """Send a message to Ollama and get a response."""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False       # False = wait for complete response
        }
    )

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.status_code} {response.text}")

    return response.json()["message"]["content"]

# Usage
reply = chat("What is the capital of France?")
print(reply)

# With a system prompt
reply = chat(
    "What should I know about Paris?",
    system="You are a brief, factual travel guide. Answer in 2-3 sentences."
)
print(reply)
```

### Method 2: Official Ollama Python Library

```python
# Install: pip install ollama
import ollama

# Simple chat
response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": "Why is the sky blue?"}]
)
print(response["message"]["content"])

# Streaming (token by token, like ChatGPT's typing effect)
stream = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": "Tell me about Paris."}],
    stream=True
)

print("Response: ", end="")
for chunk in stream:
    content = chunk["message"]["content"]
    print(content, end="", flush=True)
print()

# List available models
models = ollama.list()
for m in models["models"]:
    print(f"  {m['name']} ({m['size'] / 1e9:.1f} GB)")
```

### Method 3: OpenAI-Compatible API

Ollama speaks the same API as OpenAI. This is huge: existing code just works.

```python
# Install: pip install openai
from openai import OpenAI

# Point the OpenAI client at your local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Your local Ollama
    api_key="ollama"                        # Any non-empty string works
)

# Exact same code as OpenAI API -- just different URL!
response = client.chat.completions.create(
    model="mistral",                       # Local model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 17 x 23?"}
    ],
    temperature=0.1
)

print(response.choices[0].message.content)
```

```
+------------------------------------------------------------------+
|  WHY OPENAI-COMPATIBLE API MATTERS                               |
+------------------------------------------------------------------+
|                                                                  |
|  SCENARIO:                                                       |
|  You built an application using OpenAI's API.                    |
|  Now you want to switch to a local Ollama model to:              |
|    - Avoid API costs                                             |
|    - Keep data private (no data leaves your machine)             |
|    - Work offline                                                |
|                                                                  |
|  CHANGE REQUIRED: 2 lines of code.                               |
|                                                                  |
|  Before:                                                         |
|    client = OpenAI(api_key="sk-...")                             |
|    model = "gpt-4"                                               |
|                                                                  |
|  After:                                                          |
|    client = OpenAI(base_url="http://localhost:11434/v1",         |
|                    api_key="ollama")                              |
|    model = "mistral"                                             |
|                                                                  |
|  Everything else stays identical.                                |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Part 3: FastAPI Basics

FastAPI is the modern standard for Python HTTP APIs.

C# analogy:
```csharp
// FastAPI is Python's equivalent of ASP.NET Core Minimal APIs:
//
// ASP.NET Core Minimal API:
// var app = WebApplication.Create();
// app.MapPost("/chat", (ChatRequest req) => {
//     var response = model.Generate(req.Message);
//     return new ChatResponse { Content = response };
// });
// app.Run();
//
// FastAPI equivalent:
// app = FastAPI()
// @app.post("/chat")
// def chat(req: ChatRequest):
//     response = model.generate(req.message)
//     return ChatResponse(content=response)
//
// Key similarities:
//   - Route decorators (MapPost vs @app.post)
//   - Automatic JSON serialization
//   - Automatic request validation
//   - Built-in OpenAPI/Swagger docs (/docs endpoint)
//   - Async support
```

### A Minimal FastAPI Server

```python
# Install: pip install fastapi uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="My LLM API", version="1.0.0")

# Pydantic model = request body schema (like a C# DTO/record)
class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7

class ChatResponse(BaseModel):
    reply: str
    model: str

@app.get("/health")
def health_check():
    """Returns 200 OK if the server is running."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message and get a reply from the LLM."""
    # For now, return a dummy response
    return ChatResponse(
        reply=f"You said: {request.message}",
        model="echo"
    )

# Run with: uvicorn lesson_05_server:app --reload
# Docs at: http://localhost:8000/docs
```

Run the server:
```bash
uvicorn my_server:app --reload --host 0.0.0.0 --port 8000
```

---

## Part 4: Full FastAPI + Ollama Server

Now combine FastAPI and Ollama into a real LLM API:

```python
# full_llm_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import ollama
import time

app = FastAPI(
    title="Local LLM API",
    description="REST API wrapping a local Ollama model",
    version="1.0.0"
)

# CORS: allow web browsers from any origin to call this API
# In production, replace "*" with your frontend's specific URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===== Request/Response Models =====

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message", min_length=1, max_length=10000)
    model: str = Field(default="mistral", description="Ollama model name")
    system_prompt: Optional[str] = Field(default=None, description="Optional system instructions")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=1, le=4096)

class ChatResponse(BaseModel):
    reply: str
    model: str
    tokens_generated: int
    latency_ms: float

class ModelInfo(BaseModel):
    name: str
    size_gb: float

# ===== Endpoints =====

@app.get("/health")
def health():
    """Health check -- used by load balancers and monitoring."""
    try:
        # Try to reach Ollama to confirm it's running
        ollama.list()
        return {"status": "ok", "ollama": "connected"}
    except Exception as e:
        return {"status": "degraded", "ollama": str(e)}


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    """List all models available in local Ollama."""
    try:
        models_data = ollama.list()
        return [
            ModelInfo(
                name=m["name"],
                size_gb=m["size"] / 1e9
            )
            for m in models_data["models"]
        ]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Ollama: {e}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Send a message to a local LLM and get a response.
    The model runs locally -- no data leaves your machine.
    """
    # Build message list
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.message})

    # Call Ollama
    start_time = time.perf_counter()
    try:
        response = ollama.chat(
            model=request.model,
            messages=messages,
            options={
                "temperature": request.temperature,
                "num_predict": request.max_tokens
            }
        )
    except ollama.ResponseError as e:
        # Ollama returned an error (e.g., model not found)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Ollama server unreachable
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    reply_text = response["message"]["content"]

    return ChatResponse(
        reply=reply_text,
        model=request.model,
        tokens_generated=len(reply_text.split()),  # approximate
        latency_ms=round(elapsed_ms, 2)
    )
```

---

## Part 5: Streaming Responses

Streaming shows tokens as they are generated (like ChatGPT's typing effect).
This greatly improves perceived performance.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import json

app = FastAPI()

class StreamRequest(BaseModel):
    message: str
    model: str = "mistral"

@app.post("/chat/stream")
def chat_stream(request: StreamRequest):
    """
    Stream tokens one by one as the model generates them.
    Uses Server-Sent Events (SSE) format.
    """

    def generate():
        """Generator function: yields tokens as they arrive from Ollama."""
        stream = ollama.chat(
            model=request.model,
            messages=[{"role": "user", "content": request.message}],
            stream=True
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                # SSE format: "data: {json}\n\n"
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"

        # Signal end of stream
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",    # SSE content type
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"      # Disable nginx buffering
        }
    )
```

### Consuming the Streaming API from Python

```python
import requests
import json

def stream_chat(message, model="mistral"):
    """Call the streaming endpoint and print tokens as they arrive."""

    response = requests.post(
        "http://localhost:8000/chat/stream",
        json={"message": message, "model": model},
        stream=True    # Tell requests to not buffer the response
    )

    print("Response: ", end="")
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]   # Remove "data: " prefix
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                print(chunk["token"], end="", flush=True)
    print()   # Final newline

# Usage
stream_chat("Tell me about the Roman Empire in 3 sentences.")
```

---

## Part 6: Adding Authentication

A simple API key check to prevent unauthorized access:

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

app = FastAPI()
security = HTTPBearer()

# Store valid API keys (in production, use a database or secrets manager)
VALID_API_KEYS = {
    os.environ.get("API_KEY_1", "dev-key-12345"),
    os.environ.get("API_KEY_2", "test-key-67890")
}

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency function: validates the Bearer token.
    FastAPI calls this automatically for protected endpoints.
    """
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return credentials.credentials

# Protected endpoint -- requires valid API key
@app.post("/chat")
def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    # api_key is the validated key (you can log it for audit trail)
    return {"reply": "Protected response", "authenticated_as": api_key[:8] + "..."}

# Unprotected endpoint -- anyone can call
@app.get("/health")
def health():
    return {"status": "ok"}
```

C# analogy:
```csharp
// FastAPI's Depends() is like ASP.NET Core's [Authorize] attribute
// plus a custom policy handler:
//
// [Authorize(Policy = "ApiKeyPolicy")]
// [HttpPost("chat")]
// public IActionResult Chat([FromBody] ChatRequest request) { ... }
//
// FastAPI's dependency injection via Depends() is equivalent to
// registering a middleware or policy handler that runs before the endpoint.
```

---

## Part 7: Rate Limiting

Prevent users from overloading your server:

```python
from fastapi import FastAPI, HTTPException
from collections import defaultdict
import time

app = FastAPI()

# Simple in-memory rate limiter
# In production, use Redis for distributed rate limiting
request_counts = defaultdict(list)  # ip_address -> [timestamp, timestamp, ...]

RATE_LIMIT = 10      # Max requests per window
WINDOW_SECONDS = 60  # Time window

def check_rate_limit(client_ip: str):
    """Allow max 10 requests per minute per IP address."""
    now = time.time()
    window_start = now - WINDOW_SECONDS

    # Keep only requests within the current window
    request_counts[client_ip] = [
        t for t in request_counts[client_ip]
        if t > window_start
    ]

    # Check if over limit
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT} requests per {WINDOW_SECONDS}s."
        )

    # Record this request
    request_counts[client_ip].append(now)


@app.post("/chat")
def chat(request: ChatRequest, req: Request):
    # Get client IP from the request
    client_ip = req.client.host

    # Check rate limit before doing any LLM work
    check_rate_limit(client_ip)

    # ... rest of chat logic
    return {"reply": "..."}
```

---

## Part 8: Logging Requests

Essential for debugging and monitoring:

```python
from fastapi import FastAPI, Request
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llm_api")

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware: runs for every request, before and after the endpoint."""
    start_time = time.perf_counter()

    # Log incoming request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host}"
    )

    # Call the actual endpoint
    response = await call_next(request)

    # Log response details
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"Response: {response.status_code} in {elapsed_ms:.1f}ms"
    )

    return response
```

---

## Part 9: The Complete Picture

Now you can see how all the pieces fit together:

```
+================================================================+
|                 COMPLETE DEPLOYMENT ARCHITECTURE               |
+================================================================+
|                                                                |
|  USER'S BROWSER / APP / CLI                                    |
|    POST http://your-server:8000/chat                           |
|    Body: {"message": "...", "model": "mistral"}                |
|         |                                                      |
|         v                                                      |
|  FASTAPI SERVER (your_server.py)                               |
|    |--> Middleware: log request, check rate limit              |
|    |--> Dependency: verify API key                             |
|    |--> Endpoint: /chat                                        |
|         |--> Validate request with Pydantic                    |
|         |--> Call Ollama Python library                        |
|              |                                                 |
|              v                                                 |
|  OLLAMA SERVER (http://localhost:11434)                        |
|    |--> Find GGUF file for requested model                     |
|    |--> Load GGUF via llama.cpp                                |
|    |--> Tokenize input                                         |
|    |--> Run inference (CPU or GPU)                             |
|    |--> Stream or return tokens                                |
|         |                                                      |
|         v (back up the chain)                                  |
|  FASTAPI SERVER                                                |
|    |--> Package response as JSON                               |
|    |--> Middleware: log response, add headers                  |
|    |--> Send HTTP 200 response                                 |
|         |                                                      |
|         v                                                      |
|  USER'S BROWSER / APP / CLI                                    |
|    Receives JSON response or streams tokens                    |
|                                                                |
+================================================================+
```

---

## Part 10: Production Considerations

Things to think about before deploying to production:

```
+------------------------------------------------------------------+
|  PRODUCTION CHECKLIST FOR LLM API                                |
+------------------------------------------------------------------+
|                                                                  |
|  SECURITY                                                        |
|  [ ] API key authentication on all sensitive endpoints           |
|  [ ] HTTPS (TLS) -- never serve over plain HTTP in production    |
|  [ ] Rate limiting per user/IP                                   |
|  [ ] Input validation (max length, sanitization)                |
|  [ ] No user data logged if privacy-sensitive                    |
|                                                                  |
|  PERFORMANCE                                                     |
|  [ ] Ollama runs on GPU if available (faster responses)          |
|  [ ] Chose smallest model that meets quality needs               |
|  [ ] Streaming responses (better perceived performance)          |
|  [ ] Request queuing (if many concurrent users)                  |
|                                                                  |
|  RELIABILITY                                                     |
|  [ ] Health check endpoint for monitoring                        |
|  [ ] Graceful error messages (no stack traces to users)          |
|  [ ] Timeout handling (LLM calls can take 10-60 seconds)         |
|  [ ] Retry logic for transient Ollama failures                   |
|                                                                  |
|  MONITORING                                                      |
|  [ ] Request/response logging                                    |
|  [ ] Latency tracking (p50, p95, p99)                            |
|  [ ] Error rate tracking                                         |
|  [ ] Token count tracking (for cost estimation)                  |
|                                                                  |
|  WHAT THIS COURSE'S CAPSTONE PROJECT NEEDS:                      |
|  You will build this exact pattern for the "Chat with Codebase"  |
|  capstone. Ollama runs CodeLlama. FastAPI serves the RAG pipeline|
|  (M10/M11). ChromaDB stores the code vectors (M10).             |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Summary

```
+------------------------------------------------------------------+
|  LESSON 05 SUMMARY                                               |
+------------------------------------------------------------------+
|                                                                  |
|  1. Ollama                                                       |
|     One-command LLM download and run.                            |
|     Serves OpenAI-compatible API at localhost:11434.             |
|     ollama pull, ollama run, ollama list.                        |
|                                                                  |
|  2. Calling Ollama from Python                                   |
|     ollama library: simplest Python interface.                   |
|     OpenAI SDK: change 2 lines, all code works.                  |
|     Raw HTTP: requests.post() to /api/chat.                      |
|                                                                  |
|  3. FastAPI Basics                                               |
|     Route decorators (@app.post("/chat")).                       |
|     Pydantic models for request validation.                      |
|     Automatic /docs page with interactive testing.               |
|                                                                  |
|  4. FastAPI + Ollama                                             |
|     FastAPI handles HTTP layer. Ollama handles LLM layer.        |
|     Add auth, rate limiting, logging as middleware.              |
|                                                                  |
|  5. Streaming                                                    |
|     StreamingResponse + Server-Sent Events.                      |
|     Tokens appear one by one like ChatGPT.                       |
|                                                                  |
|  6. Production                                                   |
|     Security, performance, reliability, monitoring all matter.   |
|     Capstone project will use this exact architecture.           |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Quiz Questions

1. What does Ollama do? What inference engine does it use internally?

2. You have an existing application that calls OpenAI's API.
   How many lines of code do you need to change to switch to Ollama? What are they?

3. What is Pydantic and why does FastAPI use it?

4. In the streaming endpoint, what does `yield` do?
   How is this different from `return`?

5. You set up a FastAPI server but get a CORS error in the browser.
   What middleware do you need to add?

6. What is the purpose of a health check endpoint?
   What should it check to be useful?

7. Describe in order what happens when a user sends a POST request to
   your `/chat` endpoint in the complete deployment architecture.
   Name at least 5 steps.

8. Your LLM API is occasionally slow (10-60 seconds per request).
   What three things could you do to improve the user experience
   without necessarily making the LLM faster?

---

## Module 14 Complete!

You have covered the full deployment pipeline:

```
+------------------------------------------------------------------+
|  MODULE 14 COMPLETE -- WHAT YOU NOW KNOW                         |
+------------------------------------------------------------------+
|                                                                  |
|  Lesson 01: Quantization                                         |
|    Why models are huge. How INT8/INT4 shrinks them.              |
|    PTQ vs QAT. GPTQ. Quality tradeoffs.                          |
|                                                                  |
|  Lesson 02: GGUF Format                                          |
|    Self-contained model file. llama.cpp inference.               |
|    Q4_K_M vs Q8_0. Prompt templates. llama-cpp-python.           |
|                                                                  |
|  Lesson 03: TorchAO                                              |
|    PyTorch-native quantization. quantize_() API.                 |
|    torch.compile for maximum GPU performance.                    |
|                                                                  |
|  Lesson 04: ONNX                                                 |
|    Framework-agnostic deployment. ONNX Runtime.                  |
|    Cross-language including C#. Graph optimization.              |
|                                                                  |
|  Lesson 05: Ollama + FastAPI                                     |
|    Simplest local LLM deployment. REST API wrapper.              |
|    Streaming, auth, rate limiting, logging.                      |
|                                                                  |
|  NEXT MODULE: Module 14.5 -- Streamlit/Gradio UI                 |
|  Build a chat UI on top of your FastAPI server.                  |
|  This unlocks the Capstone project.                              |
|                                                                  |
+------------------------------------------------------------------+
```

---

*End of Module 14: Deploying LLMs.*
*Next: Module 14.5 (optional) -- Streamlit/Gradio chat UI.*
*Or: Module 15 -- Advanced LLM Training.*

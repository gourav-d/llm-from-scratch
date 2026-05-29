# =============================================================================
# Module 14 - Deploying LLMs
# Example 05: Ollama + FastAPI -- Serving Your Local LLM
# =============================================================================
#
# WHAT THIS FILE TEACHES:
#   - How to call Ollama from Python (3 methods: requests, ollama lib, OpenAI)
#   - How to build a FastAPI server wrapping a local LLM
#   - How streaming responses work (token by token like ChatGPT)
#   - How to add auth, rate limiting, and logging
#
# GLOSSARY:
#   Ollama      - Tool that downloads and runs GGUF models locally.
#                 Wraps llama.cpp. Starts an HTTP server at localhost:11434.
#   FastAPI     - Python web framework for building REST APIs.
#                 Like ASP.NET Core Minimal APIs, but Python.
#   endpoint    - A URL that accepts requests. /chat, /health, /models.
#   Pydantic    - Python data validation library. Like C# records with validation.
#                 FastAPI uses Pydantic to validate request bodies automatically.
#   streaming   - Sending tokens one by one as generated, not waiting for full response.
#                 Gives the "ChatGPT typing" effect.
#   SSE         - Server-Sent Events. HTTP streaming standard for one-way data push.
#   CORS        - Cross-Origin Resource Sharing. Browser security feature.
#                 You need it configured if browser and API are on different origins.
#   uvicorn     - ASGI web server. Runs FastAPI applications.
#   middleware  - Code that runs for every request (before endpoint logic).
#                 Used for logging, auth, rate limiting.
#
# C# ANALOGY:
#   FastAPI @app.post("/chat")  is like:
#   [HttpPost("chat")] in ASP.NET Core
#
#   FastAPI Pydantic BaseModel  is like:
#   C# record or class with [Required] / [Range] data annotations
#
#   FastAPI Depends()           is like:
#   ASP.NET Core [Authorize] + custom policy handler
#
# REQUIREMENTS:
#   Part A: pip install requests  (standard HTTP calls to Ollama)
#   Part B: pip install fastapi uvicorn ollama
#   Ollama installed: https://ollama.com
#   Model pulled:     ollama pull llama3.2:1b  (or any model you have)
#
# TO RUN THIS FILE:
#   Part A: python example_05_ollama_fastapi.py
#           (requires Ollama running: ollama serve)
#
#   Part B server: uvicorn example_05_ollama_fastapi:create_app --factory --reload
#           (runs the FastAPI server on http://localhost:8000)
#           (visit http://localhost:8000/docs for interactive testing!)
#
# =============================================================================

import json      # json: parse and create JSON data
import time      # time: measure how long operations take
import os        # os: environment variables, system info

# =============================================================================
# PART A: CALLING OLLAMA FROM PYTHON (3 METHODS)
#
# Ollama runs a local HTTP server at http://localhost:11434
# You can call it three ways:
#   Method 1: raw HTTP requests (no special library)
#   Method 2: ollama Python library (simplest)
#   Method 3: OpenAI SDK (drop-in replacement for OpenAI API)
# =============================================================================

print("=" * 60)
print("PART A: Three Ways to Call Ollama from Python")
print("=" * 60)

OLLAMA_URL = "http://localhost:11434"    # Default Ollama server address
DEFAULT_MODEL = "llama3.2:1b"           # Change to a model you have downloaded

# ---------------------------------------------------------
# Helper: Check if Ollama is running
# ---------------------------------------------------------

def check_ollama_running():
    """
    Try to reach the Ollama server.
    Returns True if running, False if not.
    """
    try:
        import requests   # requests library for HTTP calls
        # GET /api/tags returns list of downloaded models
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        # timeout=3: give up after 3 seconds (don't hang forever)
        return response.status_code == 200    # 200 = OK
    except Exception:
        # Any error (connection refused, timeout, etc.) = Ollama not running
        return False


def check_model_available(model_name):
    """
    Check if a specific model is downloaded in Ollama.
    Returns True if the model is available.
    """
    try:
        import requests
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        data = response.json()    # Parse JSON response into Python dict

        # data["models"] is a list of model info dicts
        # Each model has a "name" key
        available_names = [m["name"] for m in data.get("models", [])]
        # .get("models", []) = return models key, or empty list if not found

        return model_name in available_names   # True if our model is in the list
    except Exception:
        return False


# Check if Ollama is available
ollama_running = check_ollama_running()

if not ollama_running:
    print("\nOllama is not running or not installed.")
    print("To run Part A examples:")
    print("  1. Download Ollama from https://ollama.com")
    print("  2. Run: ollama serve")
    print("  3. Run: ollama pull llama3.2:1b")
    print("\nShowing code in educational mode instead.")
    print("-" * 40)
    EDUCATIONAL_MODE = True
else:
    model_available = check_model_available(DEFAULT_MODEL)
    if model_available:
        print(f"\nOllama is running. Model '{DEFAULT_MODEL}' is available.")
        EDUCATIONAL_MODE = False
    else:
        print(f"\nOllama is running but model '{DEFAULT_MODEL}' is not downloaded.")
        print(f"Run: ollama pull {DEFAULT_MODEL}")
        EDUCATIONAL_MODE = True


# ---------------------------------------------------------
# METHOD 1: Raw HTTP Requests
#
# Talk directly to Ollama's HTTP API using the 'requests' library.
# No special Ollama library needed -- just standard HTTP calls.
# C# equivalent: HttpClient.PostAsync(url, content)
# ---------------------------------------------------------

def method1_raw_http(message, model=DEFAULT_MODEL):
    """
    Call Ollama using raw HTTP requests.
    This works from any language that can make HTTP requests.
    """
    print("\nMethod 1: Raw HTTP request to Ollama")

    try:
        import requests

        # POST /api/chat: send a chat message, get a response
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",   # Ollama's chat endpoint
            json={                       # json= automatically sets Content-Type: application/json
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "stream": False          # False = wait for complete response (not streaming)
            },
            timeout=60   # LLM calls can take up to 60 seconds for longer responses
        )

        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code}")
            print(response.text)
            return None

        # Parse the JSON response
        data = response.json()
        # data["message"]["content"] = the model's reply text
        reply = data["message"]["content"]
        return reply

    except ImportError:
        print("requests library not installed. Run: pip install requests")
        return None
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None


if not EDUCATIONAL_MODE:
    reply = method1_raw_http("What is 2+2? Answer in one word.")
    if reply:
        print(f"Response: {reply}")
else:
    print("\nMethod 1 code (educational mode):")
    print("import requests")
    print("response = requests.post(")
    print("    'http://localhost:11434/api/chat',")
    print("    json={")
    print("        'model': 'llama3.2:1b',")
    print("        'messages': [{'role': 'user', 'content': 'What is 2+2?'}],")
    print("        'stream': False")
    print("    }")
    print(")")
    print("reply = response.json()['message']['content']")


# ---------------------------------------------------------
# METHOD 2: Ollama Python Library
#
# The official Python library for Ollama.
# More convenient than raw HTTP for Python applications.
# ---------------------------------------------------------

def method2_ollama_library(message, model=DEFAULT_MODEL):
    """
    Call Ollama using the official ollama Python library.
    Simpler code than raw HTTP. Handles error cases automatically.
    """
    print("\nMethod 2: Ollama Python library")

    try:
        import ollama   # pip install ollama

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": message}
            ]
        )

        # response is a dict with "message" key containing the AI's reply
        return response["message"]["content"]

    except ImportError:
        print("ollama library not installed. Run: pip install ollama")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def method2_streaming(message, model=DEFAULT_MODEL):
    """
    Streaming version: tokens appear one by one as they are generated.
    This is the "ChatGPT typing effect".
    """
    print("\nMethod 2 (Streaming): Tokens appear as they are generated")

    try:
        import ollama

        # stream=True returns an iterator instead of waiting for the full response
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": message}],
            stream=True   # KEY: enables streaming
        )

        print("Response: ", end="")   # end="" = no newline, tokens will follow on same line
        full_response = ""   # Collect the full response for returning

        for chunk in stream:
            # Each chunk contains a small piece of the response
            token = chunk["message"]["content"]   # The new token text
            if token:
                print(token, end="", flush=True)  # flush=True: display immediately
                full_response += token

        print()   # Final newline after all tokens

        return full_response

    except ImportError:
        print("ollama library not installed. Run: pip install ollama")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


if not EDUCATIONAL_MODE:
    reply = method2_ollama_library("Name one planet. One word only.")
    if reply:
        print(f"Response: {reply}")

    method2_streaming("Count 1, 2, 3. Just those three numbers.")
else:
    print("\nMethod 2 code (educational mode):")
    print("import ollama")
    print("response = ollama.chat(")
    print("    model='llama3.2:1b',")
    print("    messages=[{'role': 'user', 'content': 'Your question here'}]")
    print(")")
    print("print(response['message']['content'])")


# ---------------------------------------------------------
# METHOD 3: OpenAI-Compatible API
#
# Ollama speaks the same API as OpenAI's API.
# Just change the base_url from api.openai.com to localhost:11434/v1
# ALL existing OpenAI code works with Ollama with this one change!
# ---------------------------------------------------------

def method3_openai_compatible(message, model=DEFAULT_MODEL):
    """
    Call Ollama using the OpenAI Python SDK.
    SAME CODE as calling OpenAI -- just different URL!
    """
    print("\nMethod 3: OpenAI-compatible API (same code as OpenAI!)")

    try:
        from openai import OpenAI   # pip install openai

        # Point the OpenAI client at your local Ollama server
        # base_url: instead of api.openai.com, we use localhost
        # api_key: any non-empty string (Ollama doesn't check it)
        client = OpenAI(
            base_url=f"{OLLAMA_URL}/v1",   # /v1 = OpenAI-compatible path
            api_key="ollama-local"          # Required by the client, ignored by Ollama
        )

        # IDENTICAL to OpenAI API call:
        response = client.chat.completions.create(
            model=model,             # Local model name instead of "gpt-4"
            messages=[
                {"role": "user", "content": message}
            ],
            temperature=0.1,        # Low = more deterministic
            max_tokens=50
        )

        return response.choices[0].message.content

    except ImportError:
        print("openai library not installed. Run: pip install openai")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


if not EDUCATIONAL_MODE:
    reply = method3_openai_compatible("What color is the sky? One word.")
    if reply:
        print(f"Response: {reply}")
else:
    print("\nMethod 3 code (educational mode):")
    print("from openai import OpenAI")
    print("client = OpenAI(")
    print("    base_url='http://localhost:11434/v1',  # <-- only change!")
    print("    api_key='ollama'")
    print(")")
    print("response = client.chat.completions.create(")
    print("    model='llama3.2:1b',  # local model, not 'gpt-4'")
    print("    messages=[{'role': 'user', 'content': 'Your question'}]")
    print(")")


print("\nSummary: Three Methods to Call Ollama")
print("-" * 40)
print("Method 1 (requests): Universal. Works from any HTTP client.")
print("Method 2 (ollama):   Pythonic. Best for Python-only projects.")
print("Method 3 (openai):   Drop-in. Swap OpenAI for local model, zero code changes.")

# =============================================================================
# PART B: FASTAPI SERVER WRAPPING OLLAMA
#
# Build a proper REST API in front of Ollama.
# Add: validation, error handling, logging, rate limiting, streaming.
# =============================================================================

print("\n" + "=" * 60)
print("PART B: FastAPI Server Wrapping Ollama")
print("=" * 60)

# Check if FastAPI is available
try:
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    from typing import Optional
    import uvicorn
    FASTAPI_AVAILABLE = True
    print("FastAPI and uvicorn installed.")
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")


if not FASTAPI_AVAILABLE:
    print("\nShowing FastAPI server code (educational mode):")
    print("Install dependencies: pip install fastapi uvicorn ollama")
    print("-" * 40)
    print("")
    print("# Request body schema (Pydantic = C# record with validation)")
    print("class ChatRequest(BaseModel):")
    print("    message: str = Field(min_length=1, max_length=10000)")
    print("    model: str = 'llama3.2:1b'")
    print("    temperature: float = Field(default=0.7, ge=0.0, le=2.0)")
    print("    max_tokens: int = Field(default=200, ge=1, le=4096)")
    print("")
    print("# Create FastAPI app")
    print("app = FastAPI(title='Local LLM API')")
    print("")
    print("# Health check endpoint (no auth required)")
    print("@app.get('/health')")
    print("def health():")
    print("    return {'status': 'ok'}")
    print("")
    print("# Chat endpoint")
    print("@app.post('/chat')")
    print("def chat(request: ChatRequest):")
    print("    response = ollama.chat(")
    print("        model=request.model,")
    print("        messages=[{'role': 'user', 'content': request.message}]")
    print("    )")
    print("    return {'reply': response['message']['content']}")
    print("")
    print("# Run with: uvicorn example_05_ollama_fastapi:app --reload")
    print("# Docs at: http://localhost:8000/docs  (Swagger UI, try it live!)")

else:
    # ---------------------------------------------------------
    # Request and Response Models (Pydantic)
    #
    # These define the shape of JSON the API accepts and returns.
    # Pydantic validates automatically -- if user sends wrong type,
    # FastAPI returns a 422 error with helpful message.
    # C# analogy: DTOs with [Required], [Range], [MaxLength] attributes.
    # ---------------------------------------------------------

    class ChatRequest(BaseModel):
        """Request body for the /chat endpoint."""
        # Field() adds metadata: description, constraints
        # ... means the field is REQUIRED (no default value)
        message: str = Field(
            ...,                         # Required field
            description="User's message",
            min_length=1,                # Must have at least 1 character
            max_length=10000             # No essays please (model has context limits)
        )
        model: str = Field(
            default=DEFAULT_MODEL,       # Use our default model if not specified
            description="Ollama model name (must be downloaded)"
        )
        system_prompt: Optional[str] = Field(
            default=None,                # Optional: not required
            description="System instructions for the model's behavior"
        )
        temperature: float = Field(
            default=0.7,
            ge=0.0,  # ge = greater than or equal to (minimum value)
            le=2.0,  # le = less than or equal to (maximum value)
            description="Randomness: 0=deterministic, 1=normal, 2=very random"
        )
        max_tokens: int = Field(
            default=200,
            ge=1,
            le=4096,
            description="Maximum tokens to generate"
        )


    class ChatResponse(BaseModel):
        """Response body for the /chat endpoint."""
        reply: str              # The model's response text
        model: str              # Which model was used
        latency_ms: float       # How long the request took (in milliseconds)


    class ModelInfo(BaseModel):
        """Info about one available model."""
        name: str
        size_gb: float


    # ---------------------------------------------------------
    # Simple rate limiter (in-memory)
    #
    # Prevents one client from sending too many requests per minute.
    # In production: use Redis for shared rate limiting across server instances.
    # C# analogy: like a SlidingWindowRateLimiter in ASP.NET Core
    # ---------------------------------------------------------

    # Dict: IP address -> list of request timestamps
    # defaultdict creates an empty list for new keys automatically
    from collections import defaultdict
    request_counts = defaultdict(list)   # {ip: [timestamp1, timestamp2, ...]}

    RATE_LIMIT = 20        # Max requests per window
    WINDOW_SECONDS = 60    # Window duration in seconds

    def check_rate_limit(client_ip: str):
        """
        Raise HTTP 429 if client has made too many requests recently.
        Called at the start of each request.
        """
        now = time.time()              # Current time as Unix timestamp
        window_start = now - WINDOW_SECONDS   # Start of the rate limit window

        # Remove old timestamps (outside the current window)
        request_counts[client_ip] = [
            t for t in request_counts[client_ip]   # List comprehension filter
            if t > window_start                     # Keep only recent timestamps
        ]

        # Check if client is over the limit
        if len(request_counts[client_ip]) >= RATE_LIMIT:
            raise HTTPException(
                status_code=429,   # 429 = Too Many Requests (HTTP standard)
                detail=f"Rate limit exceeded: max {RATE_LIMIT} requests per {WINDOW_SECONDS}s"
            )

        # Record this request
        request_counts[client_ip].append(now)


    # ---------------------------------------------------------
    # Create the FastAPI application
    # ---------------------------------------------------------

    def create_app() -> FastAPI:
        """
        Create and configure the FastAPI application.
        Returns the app so uvicorn can import it.
        """
        app = FastAPI(
            title="Local LLM API",
            description="REST API wrapper for a local Ollama model. No data leaves your machine.",
            version="1.0.0"
        )

        # CORS middleware: allows web browsers to call this API from other origins
        # Example: React app at http://localhost:3000 calling our API at :8000
        # In production: replace "*" with your specific frontend URL
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],      # Allow all origins (change in production!)
            allow_methods=["*"],      # Allow all HTTP methods (GET, POST, etc.)
            allow_headers=["*"]       # Allow all headers
        )

        # ---------------------------------------------------------
        # Request logging middleware
        #
        # This code runs for EVERY request, before and after the endpoint.
        # C# analogy: IMiddleware or UseBefore/UseAfter in ASP.NET Core
        # ---------------------------------------------------------

        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        logger = logging.getLogger("llm_api")

        @app.middleware("http")   # Registers this function as middleware
        async def log_requests(request: Request, call_next):
            """
            Log each request with method, path, client IP, and response time.
            'async def' means this function can pause while waiting (non-blocking).
            'call_next' is a function we call to run the actual endpoint.
            """
            start = time.perf_counter()   # perf_counter: high-precision timer

            # Log the incoming request
            logger.info(
                f"IN  {request.method} {request.url.path} from {request.client.host}"
            )

            # Call the actual endpoint and wait for its response
            # 'await' pauses here until call_next completes (non-blocking wait)
            response = await call_next(request)

            elapsed_ms = (time.perf_counter() - start) * 1000   # Convert to ms
            logger.info(
                f"OUT {response.status_code} in {elapsed_ms:.0f}ms"
            )

            return response

        # ---------------------------------------------------------
        # Endpoints
        # ---------------------------------------------------------

        @app.get("/health")
        def health():
            """
            Health check: returns 200 if server and Ollama are running.
            Used by load balancers and monitoring tools (like Prometheus).
            """
            try:
                import requests as req
                # Try to reach Ollama
                r = req.get(f"{OLLAMA_URL}/api/tags", timeout=2)
                if r.status_code == 200:
                    return {"status": "ok", "ollama": "connected"}
                else:
                    return {"status": "degraded", "ollama": f"HTTP {r.status_code}"}
            except Exception as e:
                # Return 200 even if Ollama is down (server itself is healthy)
                # Let the client decide what to do with the "degraded" status
                return {"status": "ok", "ollama": f"unreachable: {e}"}


        @app.get("/models", response_model=list[ModelInfo])
        def list_models():
            """List all models currently downloaded in Ollama."""
            try:
                import requests as req
                response = req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
                data = response.json()

                return [
                    ModelInfo(
                        name=m["name"],
                        size_gb=round(m.get("size", 0) / 1e9, 2)
                        # .get("size", 0): use 0 if "size" key not present
                    )
                    for m in data.get("models", [])
                ]
            except Exception as e:
                # 503 = Service Unavailable (Ollama is not running)
                raise HTTPException(status_code=503, detail=f"Cannot reach Ollama: {e}")


        @app.post("/chat", response_model=ChatResponse)
        def chat(request: ChatRequest, req: Request):
            """
            Send a message to the local LLM and get a response.

            FastAPI automatically:
            - Parses the JSON request body into a ChatRequest object
            - Validates all field constraints (min_length, ge/le, etc.)
            - Returns 422 if validation fails (with a helpful error message)
            - Serializes the ChatResponse back to JSON
            """
            # Rate limit check (raises 429 if over limit)
            check_rate_limit(req.client.host)

            # Build the messages list for Ollama
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.message})

            # Call Ollama and measure latency
            start_time = time.perf_counter()

            try:
                import ollama as ollama_lib
                response = ollama_lib.chat(
                    model=request.model,
                    messages=messages,
                    options={
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens   # num_predict = max tokens
                    }
                )
            except Exception as e:
                # If the model isn't found, Ollama gives a specific error
                if "not found" in str(e).lower():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model '{request.model}' not found. Run: ollama pull {request.model}"
                    )
                # Any other Ollama error
                raise HTTPException(
                    status_code=503,
                    detail=f"LLM service unavailable: {e}"
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return ChatResponse(
                reply=response["message"]["content"],
                model=request.model,
                latency_ms=round(elapsed_ms, 2)
            )


        @app.post("/chat/stream")
        def chat_stream(request: ChatRequest, req: Request):
            """
            Streaming version: returns tokens one by one as generated.
            Uses Server-Sent Events (SSE) format.

            The client receives a stream of lines:
              data: {"token": "Hello"}
              data: {"token": " world"}
              data: [DONE]
            """
            # Rate limit check
            check_rate_limit(req.client.host)

            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.message})

            def generate():
                """
                Generator function: yields SSE-formatted chunks.
                'yield' pauses here and sends one chunk, then continues.
                C# analogy: IAsyncEnumerable<string> with yield return.
                """
                try:
                    import ollama as ollama_lib
                    stream = ollama_lib.chat(
                        model=request.model,
                        messages=messages,
                        options={"temperature": request.temperature},
                        stream=True   # Enable token-by-token streaming
                    )

                    for chunk in stream:
                        token = chunk["message"]["content"]
                        if token:
                            # SSE format: "data: {json}\n\n"
                            # The double newline \n\n signals end of one event
                            data = json.dumps({"token": token})   # Convert dict to JSON string
                            yield f"data: {data}\n\n"

                    # Signal that we are done generating
                    yield "data: [DONE]\n\n"

                except Exception as e:
                    # Even in streaming, we can signal errors via SSE
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),                          # The generator function
                media_type="text/event-stream",      # SSE content type (standard)
                headers={
                    "Cache-Control": "no-cache",     # Don't cache streaming responses
                    "Connection": "keep-alive",       # Keep HTTP connection open
                    "X-Accel-Buffering": "no"         # Disable nginx buffering (shows tokens live)
                }
            )


        return app   # Return the configured app


    # Show server start instructions
    print("\nFastAPI server created successfully!")
    print("")
    print("To start the server:")
    print("  uvicorn example_05_ollama_fastapi:create_app --factory --reload --port 8000")
    print("")
    print("Then:")
    print("  - Open http://localhost:8000/docs for Swagger UI (test the API live!)")
    print("  - GET  http://localhost:8000/health         (check server status)")
    print("  - GET  http://localhost:8000/models         (list downloaded models)")
    print("  - POST http://localhost:8000/chat           (chat, JSON body)")
    print("  - POST http://localhost:8000/chat/stream    (streaming chat)")
    print("")
    print("Example curl command to test /chat:")
    print('  curl -X POST http://localhost:8000/chat \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"message": "What is Python?", "max_tokens": 50}\'')


# =============================================================================
# CALLING THE STREAMING API FROM A CLIENT
# =============================================================================

print("\n" + "=" * 60)
print("How to consume the streaming /chat/stream endpoint:")
print("=" * 60)
print("")
print("import requests")
print("import json")
print("")
print("def stream_chat(message):")
print("    response = requests.post(")
print("        'http://localhost:8000/chat/stream',")
print("        json={'message': message},")
print("        stream=True    # Tell requests: don't buffer the response")
print("    )")
print("")
print("    print('Response: ', end='')")
print("    for line in response.iter_lines():")
print("        if line:")
print("            line = line.decode('utf-8')   # bytes -> str")
print("            if line.startswith('data: '):")
print("                data = line[6:]           # Remove 'data: ' prefix")
print("                if data == '[DONE]':")
print("                    break")
print("                chunk = json.loads(data)  # Parse JSON")
print("                print(chunk['token'], end='', flush=True)")
print("    print()   # Final newline")
print("")
print("stream_chat('Tell me a fact about Python.')")

# =============================================================================
# COMPLETE ARCHITECTURE DIAGRAM
# =============================================================================

print("\n" + "=" * 60)
print("Complete Architecture: User -> FastAPI -> Ollama -> LLM")
print("=" * 60)
print("")
print("USER (browser/app/curl)")
print("  |")
print("  | POST /chat {message: 'Hello'}")
print("  v")
print("FASTAPI SERVER (your_server.py on port 8000)")
print("  |- Middleware: log request, check rate limit")
print("  |- Route: validate ChatRequest with Pydantic")
print("  |- Call: ollama.chat(model, messages, options)")
print("  |")
print("  v")
print("OLLAMA SERVER (localhost:11434)")
print("  |- Find GGUF file for the requested model")
print("  |- Load model via llama.cpp (if not cached in memory)")
print("  |- Tokenize input")
print("  |- Run inference (CPU or GPU)")
print("  |- Return generated tokens")
print("  |")
print("  v (back up the chain)")
print("FASTAPI SERVER")
print("  |- Package result as ChatResponse JSON")
print("  |- Middleware: log response, add headers")
print("  |- Send HTTP 200")
print("  |")
print("  v")
print("USER receives: {'reply': 'Hello!', 'model': 'llama3.2:1b', 'latency_ms': 450.0}")

print("\n" + "=" * 60)
print("Key Takeaways:")
print("=" * 60)
print("1. Ollama makes running local LLMs trivial: one command to download and run.")
print("2. FastAPI = ASP.NET Core Minimal APIs in Python: routes, validation, docs.")
print("3. Pydantic BaseModel = C# records with automatic JSON validation.")
print("4. Streaming shows tokens as they are generated (much better UX).")
print("5. OpenAI-compatible API = swap OpenAI for Ollama with 2 line changes.")
print("6. FastAPI auto-generates Swagger docs at /docs (try it in browser!).")
print("7. Rate limiting + logging + CORS = production-ready API skeleton.")

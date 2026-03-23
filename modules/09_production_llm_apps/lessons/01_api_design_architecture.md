# Lesson 1: API Design & Architecture for LLM Applications

**Learn to build production-ready REST APIs for AI applications**

---

## Table of Contents
1. [Introduction](#introduction)
2. [Why API Design Matters](#why-api-design-matters)
3. [FastAPI Fundamentals](#fastapi-fundamentals)
4. [Request/Response Patterns](#requestresponse-patterns)
5. [Authentication & Authorization](#authentication--authorization)
6. [Error Handling](#error-handling)
7. [Streaming Responses](#streaming-responses)
8. [API Versioning](#api-versioning)
9. [Complete Example](#complete-example)
10. [Exercises](#exercises)

---

## Introduction

### What You'll Learn

By the end of this lesson, you will:
- Design RESTful APIs for LLM applications
- Build production APIs with FastAPI
- Implement authentication with JWT
- Handle streaming responses
- Validate requests with Pydantic
- Follow API best practices

### Time Required
**8-10 hours** (including hands-on practice)

### Prerequisites
- Python basics (Module 1)
- Understanding of HTTP and REST concepts
- Familiarity with LLM APIs (Module 5)

---

## Why API Design Matters

### The Problem: Quick Prototype vs Production

**Quick prototype (NOT production-ready):**
```python
# app.py - DO NOT USE IN PRODUCTION
from flask import Flask, request
import openai

app = Flask(__name__)
openai.api_key = "sk-hardcoded-key"  # ❌ Security risk!

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']  # ❌ No validation!
    response = openai.ChatCompletion.create(  # ❌ No error handling!
        model="gpt-4",
        messages=[{"role": "user", "content": message}]
    )
    return {"response": response.choices[0].message.content}

# ❌ No rate limiting
# ❌ No authentication
# ❌ No logging
# ❌ No monitoring
# ❌ Synchronous (blocks on slow requests)
```

**Production-ready API:**
```python
# app.py - Production-grade
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import logging
from datetime import datetime
import os

# ✅ Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# ✅ Structured logging
logger = logging.getLogger(__name__)

# ✅ Request validation with Pydantic
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(500, ge=1, le=MAX_TOKENS)

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    tokens_used: int
    timestamp: datetime

# ✅ FastAPI app with metadata
app = FastAPI(
    title="LLM Chat API",
    description="Production-ready LLM API",
    version="1.0.0",
    docs_url="/docs",  # Auto-generated Swagger docs
    redoc_url="/redoc"  # Alternative docs
)

# ✅ Authentication
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    # Verify JWT token
    token = credentials.credentials
    # ... token validation logic
    return token

# ✅ Async endpoint with proper error handling
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token: str = Depends(verify_token)
):
    try:
        # ✅ Input validation (automatic via Pydantic)
        logger.info(f"Chat request: {request.message[:50]}...")

        # ✅ Async LLM call (non-blocking)
        response = await async_llm_call(request)

        # ✅ Structured response
        return ChatResponse(
            response=response.content,
            conversation_id=request.conversation_id or generate_id(),
            tokens_used=response.usage.total_tokens,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        # ✅ Proper error handling
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ✅ Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow()}
```

**Key Differences:**

| Aspect | Prototype | Production |
|--------|-----------|-----------|
| **Security** | Hardcoded keys | Environment variables |
| **Validation** | None | Pydantic models |
| **Errors** | Crashes | Graceful handling |
| **Performance** | Blocking | Async/await |
| **Auth** | None | JWT tokens |
| **Docs** | None | Auto-generated Swagger |
| **Monitoring** | None | Structured logging |
| **Scalability** | 1 request at a time | 1000s concurrent |

---

## FastAPI Fundamentals

### Why FastAPI?

FastAPI is the modern Python web framework designed for building APIs:

**Advantages:**
- ✅ **Fast**: Built on Starlette (async) and Pydantic
- ✅ **Auto-docs**: Swagger UI and ReDoc out of the box
- ✅ **Type-safe**: Full type hints and validation
- ✅ **Async**: Native async/await support
- ✅ **Easy**: Intuitive and developer-friendly
- ✅ **Production**: Used by Microsoft, Uber, Netflix

**Comparison to C# ASP.NET Core:**

```python
# FastAPI (Python)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items")
async def create_item(item: Item):
    return {"name": item.name, "price": item.price}
```

```csharp
// ASP.NET Core (C#)
using Microsoft.AspNetCore.Mvc;

[ApiController]
[Route("[controller]")]
public class ItemsController : ControllerBase
{
    public class Item
    {
        public string Name { get; set; }
        public float Price { get; set; }
    }

    [HttpPost]
    public IActionResult CreateItem([FromBody] Item item)
    {
        return Ok(new { name = item.Name, price = item.Price });
    }
}
```

**Similarities:**
- Both use attributes/decorators for routing
- Both have automatic model binding
- Both support async operations
- Both generate OpenAPI specs

**Key Differences:**
- FastAPI is more concise (less boilerplate)
- FastAPI auto-generates interactive docs
- ASP.NET Core has more enterprise tooling

---

### Installing FastAPI

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install FastAPI and dependencies
pip install fastapi uvicorn[standard] pydantic python-dotenv

# For LLM integration
pip install openai anthropic

# For production
pip install python-jose[cryptography] passlib[bcrypt] python-multipart
```

**Requirements.txt:**
```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0
openai==1.10.0
anthropic==0.8.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
redis==5.0.1
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
```

---

### Basic FastAPI Application

```python
# app.py
from fastapi import FastAPI
from pydantic import BaseModel

# Create FastAPI instance
app = FastAPI(
    title="My LLM API",
    description="Production-ready LLM API",
    version="1.0.0"
)

# Request model
class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7

# Response model
class ChatResponse(BaseModel):
    response: str
    tokens: int

# Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Process request
    return ChatResponse(
        response="Hello! I received: " + request.message,
        tokens=42
    )

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Running the server:**
```bash
# Development
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Access the API:**
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
- OpenAPI spec: http://localhost:8000/openapi.json

---

## Request/Response Patterns

### 1. Simple Request/Response

**Pattern:** Client sends request, waits for complete response

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI()

class ChatRequest(BaseModel):
    """Chat request model with validation"""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(500, ge=1, le=4000, description="Max response tokens")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is machine learning?",
                "conversation_id": "conv_123",
                "temperature": 0.7,
                "max_tokens": 500
            }
        }

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="AI response")
    conversation_id: str = Field(..., description="Conversation ID")
    tokens_used: int = Field(..., description="Tokens consumed")
    model: str = Field(..., description="Model used")
    cost: float = Field(..., description="Cost in USD")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Simulate LLM call
    response_text = f"I received your message: {request.message}"

    return ChatResponse(
        response=response_text,
        conversation_id=request.conversation_id or "conv_new",
        tokens_used=100,
        model="gpt-4",
        cost=0.003
    )
```

**Testing:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, AI!",
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

**Response:**
```json
{
  "response": "I received your message: Hello, AI!",
  "conversation_id": "conv_new",
  "tokens_used": 100,
  "model": "gpt-4",
  "cost": 0.003
}
```

---

### 2. Batch Processing

**Pattern:** Process multiple requests in one call

```python
from typing import List

class BatchChatRequest(BaseModel):
    requests: List[ChatRequest] = Field(..., max_items=10, description="Max 10 requests")

class BatchChatResponse(BaseModel):
    responses: List[ChatResponse]
    total_tokens: int
    total_cost: float

@app.post("/batch-chat", response_model=BatchChatResponse)
async def batch_chat(batch: BatchChatRequest):
    """Process multiple chat requests in parallel"""
    import asyncio

    # Process all requests concurrently
    tasks = [process_single_chat(req) for req in batch.requests]
    responses = await asyncio.gather(*tasks)

    total_tokens = sum(r.tokens_used for r in responses)
    total_cost = sum(r.cost for r in responses)

    return BatchChatResponse(
        responses=responses,
        total_tokens=total_tokens,
        total_cost=total_cost
    )

async def process_single_chat(request: ChatRequest) -> ChatResponse:
    """Process a single chat request"""
    # Your LLM logic here
    await asyncio.sleep(0.1)  # Simulate API call
    return ChatResponse(
        response=f"Response to: {request.message}",
        conversation_id=request.conversation_id or "conv_new",
        tokens_used=100,
        model="gpt-4",
        cost=0.003
    )
```

---

### 3. Conversation Management

**Pattern:** Track multi-turn conversations

```python
from datetime import datetime
from typing import List, Dict

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Conversation(BaseModel):
    conversation_id: str
    messages: List[Message]
    metadata: Dict[str, any] = {}

# In-memory storage (use Redis/PostgreSQL in production)
conversations: Dict[str, Conversation] = {}

@app.post("/conversation/{conversation_id}/message")
async def add_message(conversation_id: str, message: Message):
    """Add a message to a conversation"""
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation(
            conversation_id=conversation_id,
            messages=[]
        )

    conv = conversations[conversation_id]
    conv.messages.append(message)

    # Generate AI response
    ai_response = await generate_response(conv.messages)

    # Add AI response to conversation
    ai_message = Message(
        role="assistant",
        content=ai_response
    )
    conv.messages.append(ai_message)

    return {
        "conversation_id": conversation_id,
        "user_message": message,
        "ai_response": ai_message
    }

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get full conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversations[conversation_id]
```

---

## Authentication & Authorization

### JWT Authentication

JSON Web Tokens (JWT) are the industry standard for API authentication.

**C# Comparison:**
```python
# Python: JWT creation
from jose import JWTError, jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"

def create_access_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode = {"sub": user_id, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
```

```csharp
// C#: JWT creation
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;

public string CreateAccessToken(string userId)
{
    var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes("your-secret-key"));
    var credentials = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);
    var expires = DateTime.UtcNow.AddHours(24);

    var token = new JwtSecurityToken(
        claims: new[] { new Claim("sub", userId) },
        expires: expires,
        signingCredentials: credentials
    );

    return new JwtSecurityTokenHandler().WriteToken(token);
}
```

**Complete Implementation:**

```python
# auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer()

class TokenData:
    def __init__(self, user_id: str):
        self.user_id = user_id

def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = {"sub": user_id}

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Verify JWT token and return user data"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")

        if user_id is None:
            raise credentials_exception

        return TokenData(user_id=user_id)

    except JWTError:
        raise credentials_exception

# Usage in endpoints
@app.post("/chat")
async def chat(
    request: ChatRequest,
    token_data: TokenData = Depends(verify_token)
):
    """Protected endpoint - requires valid JWT"""
    user_id = token_data.user_id
    # ... your chat logic
    return {"response": f"Hello, user {user_id}!"}

@app.post("/login")
async def login(username: str, password: str):
    """Login endpoint - returns JWT"""
    # Verify username/password (use proper password hashing!)
    if verify_credentials(username, password):
        access_token = create_access_token(user_id=username)
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")
```

**Testing authenticated endpoints:**
```bash
# 1. Login to get token
TOKEN=$(curl -X POST http://localhost:8000/login \
  -d "username=user&password=pass" | jq -r '.access_token')

# 2. Use token in requests
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

---

## Error Handling

### Proper Error Responses

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ValidationError
import logging

logger = logging.getLogger(__name__)

class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: str

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "code": "VALIDATION_ERROR"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }
    )

# Custom exceptions
class RateLimitExceeded(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )

class QuotaExceeded(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=402,
            detail="Monthly quota exceeded. Please upgrade your plan."
        )

# Usage
@app.post("/chat")
async def chat(request: ChatRequest):
    if user_exceeded_rate_limit():
        raise RateLimitExceeded()

    if user_exceeded_quota():
        raise QuotaExceeded()

    # ... normal processing
```

---

## Streaming Responses

### Server-Sent Events (SSE)

For real-time streaming like ChatGPT:

```python
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import asyncio
import json

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream response token by token"""

    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        try:
            # Simulate streaming tokens
            tokens = ["Hello", " ", "there", "!", " ", "How", " ", "can", " ", "I", " ", "help", "?"]

            for token in tokens:
                # Yield token as Server-Sent Event
                data = json.dumps({"token": token, "done": False})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0.1)  # Simulate delay

            # Final message
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )
```

**Client-side JavaScript:**
```javascript
const eventSource = new EventSource('/chat/stream');

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.done) {
        eventSource.close();
        console.log("Stream complete");
    } else if (data.error) {
        console.error("Error:", data.error);
        eventSource.close();
    } else {
        process.stdout.write(data.token);  // Append token
    }
};
```

---

## API Versioning

### Strategy: URL-based versioning

```python
from fastapi import APIRouter

# Version 1
v1_router = APIRouter(prefix="/v1")

@v1_router.post("/chat")
async def chat_v1(request: ChatRequest):
    """Version 1 of chat endpoint"""
    return {"version": "1.0", "response": "..."}

# Version 2 with breaking changes
v2_router = APIRouter(prefix="/v2")

@v2_router.post("/chat")
async def chat_v2(request: EnhancedChatRequest):
    """Version 2 with new features"""
    return {"version": "2.0", "response": "...", "metadata": {...}}

# Register routers
app.include_router(v1_router)
app.include_router(v2_router)
```

**Clients can choose version:**
- `/v1/chat` - Stable, old API
- `/v2/chat` - New features

---

## Complete Example

Here's a complete, production-ready API:

```python
# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import logging
from datetime import datetime
import os

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="Production LLM API",
    description="Enterprise-grade LLM API with authentication",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    tokens_used: int
    timestamp: datetime

# Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with LLM"""
    try:
        logger.info(f"Processing chat: {request.message[:50]}...")

        # Simulate LLM call
        await asyncio.sleep(0.5)
        response_text = f"Echo: {request.message}"

        return ChatResponse(
            response=response_text,
            conversation_id=request.conversation_id or "conv_123",
            tokens_used=42,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run it:**
```bash
python main.py
```

**Test it:**
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, world!"}'
```

---

## Exercises

### Exercise 1: Build Basic API (2 hours)

Create a simple chat API with:
1. FastAPI application
2. Request/response models
3. Health check endpoint
4. Chat endpoint

**Solution in:** `exercises/exercise_01_solution.py`

---

### Exercise 2: Add Authentication (2 hours)

Enhance your API with:
1. JWT token generation
2. Protected endpoints
3. Login endpoint
4. Token verification

**Solution in:** `exercises/exercise_02_solution.py`

---

### Exercise 3: Implement Streaming (2 hours)

Add streaming support:
1. SSE endpoint
2. Token-by-token streaming
3. Error handling in streams
4. Client example

**Solution in:** `exercises/exercise_03_solution.py`

---

## Summary

### What You Learned

- ✅ FastAPI fundamentals
- ✅ Request/response patterns
- ✅ JWT authentication
- ✅ Error handling
- ✅ Streaming responses
- ✅ API versioning

### Key Takeaways

1. **Validation is critical** - Use Pydantic models
2. **Async everywhere** - Better performance
3. **Authentication required** - Protect your API
4. **Error handling matters** - Fail gracefully
5. **Documentation is free** - FastAPI generates it

### Next Steps

**Lesson 2:** Deployment & Scalability
- Containerize with Docker
- Deploy to cloud
- Set up load balancing
- Add caching with Redis

---

**Time to complete:** 8-10 hours
**Difficulty:** Intermediate
**Prerequisites:** Python basics, HTTP knowledge
**Next lesson:** 02_deployment_scalability.md

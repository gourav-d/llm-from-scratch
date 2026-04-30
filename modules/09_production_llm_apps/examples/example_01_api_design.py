# =============================================================================
# Example 01: API Design Patterns for LLM Applications
# Module 09 - Production LLM Applications
# =============================================================================
# GOAL: Simulate a production-grade API design WITHOUT needing FastAPI installed.
#       Everything here uses only Python's built-in standard library (stdlib).
#
# C# ANALOGY: Think of this entire file as a simplified version of an
#             ASP.NET Core Web API project -- models, middleware, routing,
#             authentication, and rate limiting -- but in pure Python.
# =============================================================================

# --- IMPORTS (stdlib only, nothing to install) ---
import base64          # For encoding JWT-style tokens (like Convert.ToBase64String in C#)
import json            # For serializing/deserializing data (like JsonSerializer in C#)
import time            # For timestamps and delays (like DateTime.UtcNow in C#)
import hashlib         # For creating fake signatures (like HMACSHA256 in C#)
import collections     # For deque (double-ended queue) used in sliding window rate limiter
import threading       # For thread-safe operations (like lock{} in C#)

# =============================================================================
# GLOSSARY
# =============================================================================
# Read these definitions BEFORE looking at the code below.
# This is just like reading the "Concepts" section in a textbook first.
# =============================================================================

GLOSSARY = """
=============================================================================
GLOSSARY - API Design Terms
=============================================================================

REST API
  A style of building web services where everything is a "resource" accessed
  via URLs. Like a contract between client and server.
  C# equivalent: ASP.NET Core Web API controller methods.
  Example: GET /api/v1/chat  means "get a chat response from the server"

Endpoint
  A specific URL in your API that does one specific thing.
  C# equivalent: A [HttpGet] or [HttpPost] method in a controller.
  Example: /api/v1/chat is an endpoint.

Request / Response
  Request  = data the CLIENT sends TO the server (question)
  Response = data the SERVER sends BACK to the client (answer)
  C# equivalent: HttpRequest / HttpResponse objects in ASP.NET Core.

Middleware
  Code that runs BETWEEN receiving a request and sending a response.
  Like a security guard that checks your badge before letting you in.
  C# equivalent: ASP.NET Core Middleware (app.Use(...) in Startup.cs).
  Examples: logging, authentication, rate limiting.

JWT Token (JSON Web Token)
  A small, signed text token that proves who you are.
  Contains: header.payload.signature (three parts separated by dots).
  C# equivalent: ASP.NET Identity tokens / Bearer tokens in Authorization header.
  Example: eyJhbGci...  (looks like gibberish, but it's base64-encoded JSON)

Rate Limiting
  A rule like "max 100 requests per minute per user" to prevent abuse.
  C# equivalent: ASP.NET Core Rate Limiting middleware (AddRateLimiter).
  Think of it as a bouncer at a club counting how many people enter per hour.

Streaming
  Sending the response word-by-word instead of all at once.
  Like ChatGPT typing out its answer gradually.
  C# equivalent: IAsyncEnumerable<T> or Server-Sent Events (SSE).

Pydantic Validation
  A Python library that validates data types and values automatically.
  C# equivalent: DataAnnotations ([Required], [Range], [StringLength]).
  We simulate this manually here since we cannot install Pydantic.

HTTP Status Codes
  Standard numbers that tell the client what happened:
  200 = OK (success)
  400 = Bad Request (client sent bad data)
  401 = Unauthorized (not logged in)
  403 = Forbidden (logged in but not allowed)
  429 = Too Many Requests (rate limited)
  500 = Internal Server Error (server crashed)
  C# equivalent: return Ok(), return BadRequest(), return Unauthorized()

Versioning
  Putting a version number in the URL so old clients still work when you
  change the API.  Example: /api/v1/chat vs /api/v2/chat
  C# equivalent: API versioning via [ApiVersion("1.0")] attribute.

=============================================================================
"""

print(GLOSSARY)  # Print the glossary so the student sees it first

# =============================================================================
# ASCII DIAGRAM: REQUEST FLOW
# =============================================================================
# This shows what happens to EVERY request that comes into our API.
# Read it left to right, top to bottom.
# =============================================================================

REQUEST_FLOW_DIAGRAM = """
=============================================================================
REQUEST FLOW DIAGRAM
=============================================================================

  +--------+     +--------------+     +-----------------+
  | Client |---->| Rate Limiter |---->| Auth Middleware  |
  +--------+     +--------------+     +-----------------+
                       |                       |
                  (rejected if             (rejected if
                   too many reqs)           bad/no token)
                                            |
                                   +------------------+
                                   | Input Validator  |
                                   +------------------+
                                            |
                                   (rejected if bad data)
                                            |
                                   +------------------+
                                   |   LLM Handler    |
                                   +------------------+
                                            |
                                   +------------------+
                                   |    Response      |
                                   +------------------+

  Each box is a MIDDLEWARE that either:
    - Passes the request forward  -> (True,  None)
    - Rejects the request         -> (False, "error message")

=============================================================================
"""

print(REQUEST_FLOW_DIAGRAM)  # Show the diagram to the student

# =============================================================================
# PART 1 - REQUEST / RESPONSE MODELS
# =============================================================================
# In C#: this is like a DTO (Data Transfer Object) class with
#         DataAnnotation attributes for validation.
#
# [Required] string Message      -> message cannot be empty
# [Range(1, 4096)] int MaxTokens -> max_tokens must be 1-4096
# =============================================================================

print("=" * 70)
print("PART 1 - REQUEST / RESPONSE MODELS")
print("=" * 70)

class RequestModel:
    """
    Represents one incoming API request from a client.

    C# equivalent:
        public class ChatRequest {
            [Required] public string UserId    { get; set; }
            [Required] public string Message   { get; set; }
            public string Model      { get; set; } = "gpt-mini";
            [Range(1,4096)] public int MaxTokens { get; set; } = 256;
            [Range(0,2)]    public float Temperature { get; set; } = 0.7f;
        }
    """

    def __init__(self, user_id, message, model="gpt-mini",
                 max_tokens=256, temperature=0.7):
        # Store the user ID (who is making the request)
        self.user_id = user_id

        # Store the message/prompt the user wants to send
        self.message = message

        # Which LLM model to use (default is a small/cheap model)
        self.model = model

        # Maximum number of tokens in the response (1 token ~ 0.75 words)
        self.max_tokens = max_tokens

        # How "creative" the response should be (0 = deterministic, 2 = very random)
        self.temperature = temperature

    def validate(self):
        """
        Check that all fields have valid values.
        Returns (True, None) if valid, or (False, "reason") if invalid.

        C# equivalent:
            if (!ModelState.IsValid) return BadRequest(ModelState);
        """

        # Check: message cannot be blank or empty string
        if not self.message or not self.message.strip():
            return (False, "message cannot be empty")  # Like returning 400 Bad Request

        # Check: max_tokens must be between 1 and 4096 (inclusive)
        if not (1 <= self.max_tokens <= 4096):
            return (False, "max_tokens must be between 1 and 4096")

        # Check: temperature must be between 0.0 and 2.0
        if not (0.0 <= self.temperature <= 2.0):
            return (False, "temperature must be between 0.0 and 2.0")

        # All checks passed -- request is valid
        return (True, None)

    def __repr__(self):
        # __repr__ is like ToString() in C# -- gives a readable string version
        return (f"RequestModel(user_id={self.user_id!r}, "
                f"message={self.message[:30]!r}..., "
                f"model={self.model!r}, "
                f"max_tokens={self.max_tokens}, "
                f"temperature={self.temperature})")


class ResponseModel:
    """
    Represents the API response we send back to the client.

    C# equivalent:
        public class ChatResponse {
            public string RequestId  { get; set; }
            public string Content    { get; set; }
            public int    StatusCode { get; set; }
            public string Error      { get; set; }
            public double ProcessingTimeMs { get; set; }
        }
    """

    def __init__(self, request_id, content, status_code=200,
                 error=None, processing_time_ms=0):
        # Unique ID for this request (for tracing/debugging in logs)
        self.request_id = request_id

        # The actual LLM-generated text content
        self.content = content

        # HTTP status code (200=OK, 400=Bad Request, 429=Rate Limited, etc.)
        self.status_code = status_code

        # Error message (None if successful)
        self.error = error

        # How long the server took to process (for monitoring dashboards)
        self.processing_time_ms = processing_time_ms

    def __repr__(self):
        # Readable string for printing/logging
        return (f"ResponseModel(status={self.status_code}, "
                f"content={self.content[:40]!r}..., "
                f"error={self.error!r}, "
                f"time_ms={self.processing_time_ms:.1f})")


# --- Demo: Test valid and invalid requests ---

print("\n-- Testing valid request --")
valid_req = RequestModel(                       # Create a well-formed request
    user_id="user_001",
    message="Explain neural networks simply",
    model="gpt-mini",
    max_tokens=512,
    temperature=0.7
)
is_valid, error = valid_req.validate()          # Run validation
print(f"Request: {valid_req}")                  # Print the request details
print(f"Valid: {is_valid}, Error: {error}")     # Should print: Valid: True, Error: None

print("\n-- Testing invalid request: empty message --")
bad_req1 = RequestModel(user_id="user_002", message="")  # Empty message
is_valid, error = bad_req1.validate()
print(f"Valid: {is_valid}, Error: {error}")     # Should show error about empty message

print("\n-- Testing invalid request: bad max_tokens --")
bad_req2 = RequestModel(                        # max_tokens way out of range
    user_id="user_003",
    message="Hello",
    max_tokens=99999                            # 99999 exceeds the 4096 limit
)
is_valid, error = bad_req2.validate()
print(f"Valid: {is_valid}, Error: {error}")     # Should show range error

print("\n-- Testing invalid request: bad temperature --")
bad_req3 = RequestModel(                        # temperature out of 0-2 range
    user_id="user_004",
    message="Hello",
    temperature=5.0                             # 5.0 exceeds maximum of 2.0
)
is_valid, error = bad_req3.validate()
print(f"Valid: {is_valid}, Error: {error}")     # Should show temperature error

# =============================================================================
# PART 2 - JWT-STYLE TOKEN (SIMULATED)
# =============================================================================
# A real JWT has three base64-encoded parts: header.payload.signature
# We simulate this with Python's built-in base64 and hashlib modules.
#
# C# equivalent: JWT Bearer tokens used in ASP.NET Core with
#                services.AddAuthentication().AddJwtBearer(...)
#
# Structure: eyJhbGci...  <-- header
#            .eyJ1c2Vy...  <-- payload (contains user_id, role, expiry)
#            .abc123sig...  <-- fake signature
# =============================================================================

print("\n" + "=" * 70)
print("PART 2 - JWT-STYLE TOKEN (SIMULATED)")
print("=" * 70)

# Secret key used to "sign" tokens (in real life, keep this in a secure vault!)
# C# equivalent: the secret in appsettings.json under JwtSettings:SecretKey
JWT_SECRET = "super-secret-key-do-not-commit-to-git"

class SimpleJWT:
    """
    Simulates JWT token creation and verification.

    REAL JWT libraries (PyJWT, jose) do this properly with HMAC-SHA256.
    We use base64 + hashlib here to show the CONCEPT without installation.

    C# equivalent: System.IdentityModel.Tokens.Jwt.JwtSecurityTokenHandler
    """

    def __init__(self, secret):
        # Store the secret key used for signing
        self.secret = secret

    def create_token(self, user_id, role="user"):
        """
        Creates a fake JWT token for a user.
        Token expires in 1 hour (3600 seconds).

        C# equivalent:
            var token = new JwtSecurityToken(
                claims: claims,
                expires: DateTime.UtcNow.AddHours(1),
                signingCredentials: creds);
        """

        # Build the "payload" dictionary -- this is the data inside the token
        payload = {
            "user_id": user_id,        # Who this token belongs to
            "role":    role,           # What they are allowed to do
            "issued":  time.time(),    # When the token was created (Unix timestamp)
            "expires": time.time() + 3600  # Expires 1 hour from now
        }

        # Encode the payload as JSON, then base64-encode it
        # json.dumps() = JsonSerializer.Serialize() in C#
        # base64.b64encode() = Convert.ToBase64String() in C#
        payload_json   = json.dumps(payload)           # Convert dict -> JSON string
        payload_bytes  = payload_json.encode("utf-8")  # Convert string -> bytes
        payload_b64    = base64.b64encode(payload_bytes).decode("utf-8")  # base64

        # Create a fake "signature" using hashlib SHA256
        # Real JWT uses HMAC-SHA256; we just demonstrate the concept
        # C# equivalent: HMACSHA256.ComputeHash(...)
        sig_input   = payload_b64 + self.secret        # Combine payload + secret
        signature   = hashlib.sha256(                  # Hash the combined string
            sig_input.encode("utf-8")
        ).hexdigest()[:16]                             # Take first 16 hex chars

        # Combine into "header.payload.signature" format (classic JWT structure)
        header = base64.b64encode(b'{"alg":"HS256"}').decode("utf-8")
        token  = f"{header}.{payload_b64}.{signature}"  # Final token string

        return token  # Return the complete token string

    def verify_token(self, token):
        """
        Verifies the token is valid and returns the payload dict.
        Raises ValueError if the token is tampered or malformed.

        C# equivalent:
            tokenHandler.ValidateToken(token, validationParameters, out _);
        """

        try:
            # Split the token into its three parts
            parts = token.split(".")          # Split on dot character
            if len(parts) != 3:              # Must have exactly 3 parts
                raise ValueError("Token format invalid: expected 3 parts")

            # Unpack the three parts
            header_b64, payload_b64, signature = parts

            # Re-compute what the signature SHOULD be
            expected_sig = hashlib.sha256(
                (payload_b64 + self.secret).encode("utf-8")
            ).hexdigest()[:16]

            # Compare actual signature vs expected -- if different, token was tampered
            if signature != expected_sig:
                raise ValueError("Token signature invalid: possible tampering")

            # Decode the payload from base64 back to a dict
            # base64.b64decode() = Convert.FromBase64String() in C#
            # Add padding if needed (base64 requires length divisible by 4)
            padding    = "=" * (4 - len(payload_b64) % 4)   # Add "=" padding
            payload_bytes = base64.b64decode(payload_b64 + padding)
            payload    = json.loads(payload_bytes.decode("utf-8"))  # JSON -> dict

            return payload  # Return the decoded payload (user_id, role, etc.)

        except (ValueError, json.JSONDecodeError, Exception) as e:
            # Re-raise as ValueError so callers can catch it uniformly
            raise ValueError(f"Token verification failed: {e}")

    def is_expired(self, token):
        """
        Returns True if the token has expired, False if still valid.

        C# equivalent: checking token.ValidTo < DateTime.UtcNow
        """

        try:
            payload = self.verify_token(token)          # Decode the token
            return time.time() > payload["expires"]     # Compare current time vs expiry
        except ValueError:
            return True  # If token is invalid, treat it as expired


# --- Demo: JWT token lifecycle ---

jwt = SimpleJWT(secret=JWT_SECRET)  # Create the JWT utility with our secret key

print("\n-- Creating a JWT token --")
token = jwt.create_token(user_id="user_001", role="admin")  # Create a token
print(f"Token (truncated): {token[:60]}...")                  # Show the start of token
print(f"Token length: {len(token)} characters")              # Show how long it is

print("\n-- Verifying the token --")
payload = jwt.verify_token(token)                            # Decode and verify
print(f"Decoded payload: {json.dumps(payload, indent=2)}")   # Show what's inside
print(f"Is expired: {jwt.is_expired(token)}")                # Should be False

print("\n-- What an expired/tampered token looks like --")
# Tamper with the token by changing a character in the signature
bad_token = token[:-5] + "XXXXX"                            # Corrupt the last 5 chars
try:
    jwt.verify_token(bad_token)                             # Try to verify bad token
except ValueError as e:
    print(f"Caught error: {e}")                             # Should print tampering error

# Simulate an expired token by manually building one with past expiry
expired_payload = {
    "user_id": "user_999",
    "role":    "user",
    "issued":  time.time() - 7200,      # Issued 2 hours ago
    "expires": time.time() - 3600       # Expired 1 hour ago
}
expired_json  = json.dumps(expired_payload)
expired_b64   = base64.b64encode(expired_json.encode()).decode()
exp_sig_input = expired_b64 + JWT_SECRET
exp_sig       = hashlib.sha256(exp_sig_input.encode()).hexdigest()[:16]
exp_header    = base64.b64encode(b'{"alg":"HS256"}').decode()
expired_token = f"{exp_header}.{expired_b64}.{exp_sig}"    # Build the expired token
print(f"Expired token check: is_expired = {jwt.is_expired(expired_token)}")  # True

# =============================================================================
# PART 3 - RATE LIMITER (SLIDING WINDOW)
# =============================================================================
# Sliding Window Rate Limiter:
#   - Track the TIMESTAMPS of the last N requests per user
#   - If the user has made >= max_requests in the last window_seconds, BLOCK them
#   - "Sliding" means the window moves with time (not fixed 1-minute buckets)
#
# C# equivalent: ASP.NET Core Rate Limiting middleware
#   builder.Services.AddRateLimiter(options => {
#       options.AddSlidingWindowLimiter("sliding", o => {
#           o.PermitLimit = 5;
#           o.Window = TimeSpan.FromSeconds(10);
#       });
#   });
# =============================================================================

print("\n" + "=" * 70)
print("PART 3 - RATE LIMITER (SLIDING WINDOW)")
print("=" * 70)

class RateLimiter:
    """
    Sliding-window rate limiter.
    Tracks per-user request timestamps and rejects if limit exceeded.

    Think of it like a nightclub with a rule:
    "Max 5 people can enter in any 10-second window."
    """

    def __init__(self, max_requests, window_seconds):
        # Maximum number of requests allowed in the time window
        self.max_requests = max_requests

        # The size of the sliding time window in seconds
        self.window_seconds = window_seconds

        # Dictionary mapping user_id -> deque of request timestamps
        # deque is like a List<DateTime> in C# but optimized for removing from front
        self.user_timestamps = {}

        # Lock for thread safety (like lock{} in C#)
        self.lock = threading.Lock()

    def _cleanup_old_timestamps(self, user_id):
        """
        Remove timestamps that are older than the window.
        Private method (the _ prefix means 'internal use only' in Python,
        like a private method in C#).
        """

        # Get the current time
        now = time.time()

        # Get this user's timestamps, or empty deque if first request
        # defaultdict would do this automatically, but we do it manually here
        if user_id not in self.user_timestamps:
            self.user_timestamps[user_id] = collections.deque()  # Create empty deque

        timestamps = self.user_timestamps[user_id]  # Get the deque reference

        # Remove timestamps that are older than window_seconds ago
        # We remove from the LEFT side (oldest) of the deque
        cutoff = now - self.window_seconds          # Anything before this is expired
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()  # Remove oldest timestamp (like List.RemoveAt(0))

    def is_allowed(self, user_id):
        """
        Returns True if the user can make another request, False if rate-limited.

        C# equivalent: if (rateLimiter.TryAcquire(userId)) { ... }
        """

        with self.lock:  # Thread-safe block (like lock(lockObj) { ... } in C#)

            # Clean out old timestamps first
            self._cleanup_old_timestamps(user_id)

            timestamps = self.user_timestamps[user_id]  # Get cleaned timestamps

            # Check if user is below the limit
            if len(timestamps) < self.max_requests:
                # Under the limit -- record this request and allow it
                timestamps.append(time.time())  # Add current timestamp to right side
                return True                     # Allow the request
            else:
                # At or over the limit -- reject
                return False                    # Block the request

    def get_remaining(self, user_id):
        """
        Returns how many more requests this user can make in the current window.
        """

        with self.lock:  # Thread-safe access
            self._cleanup_old_timestamps(user_id)  # Remove expired timestamps
            timestamps = self.user_timestamps.get(user_id, collections.deque())
            # Remaining = max - used (cannot go below 0)
            return max(0, self.max_requests - len(timestamps))


# --- Demo: Hit the rate limit ---

print("\nCreating rate limiter: max 5 requests per 10 seconds")
limiter = RateLimiter(max_requests=5, window_seconds=10)  # 5 requests per 10 seconds

for i in range(1, 8):  # Try 7 requests (should fail on 6th and 7th)
    user = "user_001"                                      # Same user each time
    allowed = limiter.is_allowed(user)                     # Check if allowed
    remaining = limiter.get_remaining(user)                # How many left
    status = "ALLOWED" if allowed else "BLOCKED (429 Too Many Requests)"
    print(f"  Request {i}: {status} | Remaining slots: {remaining}")

# =============================================================================
# PART 4 - MIDDLEWARE PIPELINE
# =============================================================================
# A middleware pipeline is a chain of functions that each get to inspect
# (and potentially reject) a request before it reaches the actual handler.
#
# C# equivalent: ASP.NET Core middleware pipeline
#   app.Use(async (context, next) => { ... await next(); });
#
# Each middleware function signature:
#   def my_middleware(request) -> (bool, str or None)
#     True  = pass through to the next middleware
#     False = reject with error message (short-circuit)
# =============================================================================

print("\n" + "=" * 70)
print("PART 4 - MIDDLEWARE PIPELINE")
print("=" * 70)

# --- Individual Middleware Functions ---

def auth_middleware(request):
    """
    Checks that the request has a valid JWT token.
    C# equivalent: [Authorize] attribute or UseAuthentication() middleware.
    """

    # In a real API, the token comes in the Authorization HTTP header
    # Here, we store it in the request dict for simulation
    token = request.get("token", None)  # Get token from request, or None if missing

    if token is None:                          # No token at all
        return (False, "401 Unauthorized: No token provided")

    try:
        jwt_util = SimpleJWT(JWT_SECRET)       # Create JWT utility
        payload  = jwt_util.verify_token(token)  # Verify the token

        if jwt_util.is_expired(token):         # Check if expired
            return (False, "401 Unauthorized: Token has expired")

        # Token is valid -- attach the user info to the request for later middlewares
        request["authenticated_user"] = payload  # Like HttpContext.User in ASP.NET
        return (True, None)                    # Pass through

    except ValueError as e:
        return (False, f"401 Unauthorized: {e}")  # Invalid token


def rate_limit_middleware(request):
    """
    Checks the user hasn't exceeded their request quota.
    C# equivalent: app.UseRateLimiter() in ASP.NET Core.
    """

    # Get the user_id from the authenticated user (set by auth_middleware)
    user    = request.get("authenticated_user", {})
    user_id = user.get("user_id", "anonymous")  # Default to "anonymous" if not set

    # Check the shared rate limiter (defined below -- we pass it in via closure)
    if not pipeline_limiter.is_allowed(user_id):
        remaining = pipeline_limiter.get_remaining(user_id)  # Should be 0
        return (False, f"429 Too Many Requests: Quota exceeded (remaining: {remaining})")

    return (True, None)  # Under the limit -- allow through


def logging_middleware(request):
    """
    Logs every request for auditing and debugging.
    C# equivalent: app.UseSerilogRequestLogging() or app.UseHttpLogging().
    This always passes through (never rejects).
    """

    # Get useful info from the request for logging
    user_id = request.get("authenticated_user", {}).get("user_id", "anonymous")
    message = request.get("message", "")[:40]  # First 40 chars of the message
    ts      = time.strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp as string

    # Log to console (in production this would go to CloudWatch, Splunk, etc.)
    print(f"  [LOG {ts}] user={user_id!r} message={message!r}...")

    return (True, None)  # Logging never blocks -- always passes through


def validation_middleware(request):
    """
    Validates the request body fields.
    C# equivalent: ModelState.IsValid check + [ApiController] automatic 400 response.
    """

    # Build a RequestModel from the request dict to use our validation logic
    try:
        req_model = RequestModel(
            user_id    = request.get("user_id", ""),
            message    = request.get("message", ""),
            model      = request.get("model", "gpt-mini"),
            max_tokens = request.get("max_tokens", 256),
            temperature= request.get("temperature", 0.7)
        )
        is_valid, error = req_model.validate()  # Run validation
        if not is_valid:
            return (False, f"400 Bad Request: {error}")  # Return validation error
        return (True, None)                              # Validation passed
    except Exception as e:
        return (False, f"400 Bad Request: Unexpected validation error: {e}")


# --- The Pipeline Class ---

class MiddlewarePipeline:
    """
    Runs a request through a series of middleware functions in order.
    Stops at the first rejection.

    C# equivalent: The middleware pipeline configured in Program.cs with
                   app.Use(), app.UseAuthentication(), app.UseAuthorization()
    """

    def __init__(self):
        # List of middleware functions to run in order
        # C# equivalent: List<Func<HttpContext, RequestDelegate, Task>>
        self.middlewares = []

    def add(self, middleware_fn):
        """
        Register a middleware function to run in the pipeline.
        C# equivalent: app.Use(async (ctx, next) => { ... })
        """
        self.middlewares.append(middleware_fn)  # Add to the end of the list
        return self                             # Return self for method chaining

    def process(self, request):
        """
        Run the request through all middlewares in order.
        Returns (True, None) if all pass, or (False, error) if any reject.
        """

        # Go through each middleware one by one
        for middleware in self.middlewares:
            passed, error = middleware(request)  # Call the middleware function

            if not passed:
                # This middleware rejected the request -- stop processing
                return (False, error)  # Short-circuit the pipeline

        # All middlewares passed -- request is fully validated
        return (True, None)


# --- Setup the pipeline and rate limiter ---

pipeline_limiter = RateLimiter(max_requests=3, window_seconds=60)  # 3 req/min for demo

pipeline = MiddlewarePipeline()  # Create the pipeline
pipeline.add(auth_middleware)         # Step 1: Check token
pipeline.add(rate_limit_middleware)   # Step 2: Check rate limit
pipeline.add(logging_middleware)      # Step 3: Log the request
pipeline.add(validation_middleware)   # Step 4: Validate request body

# Create a real token for testing
jwt_util   = SimpleJWT(JWT_SECRET)
good_token = jwt_util.create_token("user_001", role="user")  # Valid token for user_001

# --- Demo: Process three different requests ---

print("\n-- Request 1: Valid request (should pass all middleware) --")
req1 = {
    "token":       good_token,          # Valid JWT token
    "user_id":     "user_001",          # User ID
    "message":     "What is attention mechanism in transformers?",  # Good message
    "model":       "gpt-mini",
    "max_tokens":  256,
    "temperature": 0.7
}
passed, error = pipeline.process(req1)   # Run through pipeline
print(f"  Result: {'PASSED' if passed else 'REJECTED'} | Error: {error}")

print("\n-- Request 2: No token (should fail auth middleware) --")
req2 = {
    # No "token" key -- simulates a request without Authorization header
    "user_id":     "user_002",
    "message":     "Hello",
    "max_tokens":  100,
    "temperature": 0.5
}
passed, error = pipeline.process(req2)
print(f"  Result: {'PASSED' if passed else 'REJECTED'} | Error: {error}")

print("\n-- Request 3: Valid token but empty message (should fail validation) --")
req3 = {
    "token":       good_token,          # Valid token
    "user_id":     "user_001",
    "message":     "",                  # Empty message -- validation should reject
    "max_tokens":  256,
    "temperature": 0.7
}
passed, error = pipeline.process(req3)
print(f"  Result: {'PASSED' if passed else 'REJECTED'} | Error: {error}")

# Exhaust the rate limit for user_001 then try again
print("\n-- Requests 4-6: Exhaust rate limit for user_001 --")
for i in range(3):  # Use up the 3-request quota
    req_extra = {
        "token":       good_token,
        "user_id":     "user_001",
        "message":     f"Request number {i+4}",
        "max_tokens":  100,
        "temperature": 0.5
    }
    passed, error = pipeline.process(req_extra)
    print(f"  Request {i+4}: {'PASSED' if passed else 'BLOCKED'} | {error}")

# =============================================================================
# PART 5 - STREAMING RESPONSE SIMULATOR
# =============================================================================
# Real LLMs stream tokens one at a time so the user sees output immediately
# instead of waiting for the full response.
#
# Python GENERATOR functions (using "yield") are perfect for this.
# C# equivalent: IAsyncEnumerable<string> or IEnumerable<string> with yield return
#
# In production, this is sent via Server-Sent Events (SSE) or WebSocket.
# =============================================================================

print("\n" + "=" * 70)
print("PART 5 - STREAMING RESPONSE SIMULATOR")
print("=" * 70)

# Canned responses based on keywords in the prompt
# In a real system, this would call the actual LLM API
CANNED_RESPONSES = {
    "attention":     "The attention mechanism allows the model to focus on "
                     "different parts of the input when producing each output token. "
                     "It computes a weighted sum of values based on query-key similarity.",

    "neural":        "A neural network is a system of layers of interconnected nodes "
                     "that learn patterns from data. Each layer transforms its inputs "
                     "using weights and activation functions.",

    "python":        "Python is a high-level interpreted language known for its "
                     "clean readable syntax. It is widely used in data science "
                     "machine learning and web development.",

    "default":       "This is a simulated LLM response. In production this text "
                     "would be generated token by token by a real language model "
                     "like GPT or Llama running on GPU hardware."
}


def stream_llm_response(prompt, word_delay=0.0):
    """
    Generator function that yields words one at a time, simulating streaming.

    The "yield" keyword makes this a GENERATOR -- it pauses after each yield
    and resumes when the caller asks for the next value.

    C# equivalent:
        IEnumerable<string> StreamResponse(string prompt) {
            foreach (var word in words) { yield return word; }
        }

    Args:
        prompt     : The user's input text
        word_delay : Seconds to wait between words (0.0 for instant in demo)
    """

    # Choose a response based on keywords in the prompt (case-insensitive)
    prompt_lower = prompt.lower()          # Convert to lowercase for matching

    if "attention" in prompt_lower:        # Check for "attention" keyword
        response = CANNED_RESPONSES["attention"]
    elif "neural" in prompt_lower:         # Check for "neural" keyword
        response = CANNED_RESPONSES["neural"]
    elif "python" in prompt_lower:         # Check for "python" keyword
        response = CANNED_RESPONSES["python"]
    else:
        response = CANNED_RESPONSES["default"]  # Fallback response

    words = response.split(" ")           # Split response into individual words

    # Yield each word one at a time (this is the "streaming" part)
    for word in words:
        if word_delay > 0:
            time.sleep(word_delay)         # Optional delay between words
        yield word + " "                  # Yield one word at a time with a space


# --- Demo: Stream a response ---

test_prompt = "How does the attention mechanism work?"
print(f"\nStreaming response to: {test_prompt!r}")
print("(Each word arrives separately, simulating real LLM streaming)\n")
print("Response: ", end="", flush=True)   # Print without newline (end="" like Console.Write in C#)

for word_chunk in stream_llm_response(test_prompt, word_delay=0.0):
    print(word_chunk, end="", flush=True) # Print each word as it "arrives"

print("\n\n[Stream complete]")            # Newline after streaming finishes

# =============================================================================
# PART 6 - API VERSIONING PATTERN
# =============================================================================
# API versioning lets you update your API without breaking existing clients.
# v1 clients keep working on /v1/... while new clients use /v2/...
#
# C# equivalent: Microsoft.AspNetCore.Mvc.Versioning package
#   [ApiVersion("1.0")]
#   [ApiVersion("2.0")]
#   [Route("api/v{version:apiVersion}/chat")]
#   public class ChatController : ControllerBase { ... }
# =============================================================================

print("\n" + "=" * 70)
print("PART 6 - API VERSIONING PATTERN")
print("=" * 70)

class Router:
    """
    Simple URL router that maps (version, endpoint) -> handler function.
    Simulates what ASP.NET Core routing does automatically.

    C# equivalent: app.MapControllerRoute() or [Route] attributes.
    """

    def __init__(self):
        # Dictionary mapping (version, path) -> handler function
        # Like a RouteTable in ASP.NET
        self.routes = {}

    def register(self, version, path, handler):
        """
        Register a handler for a specific version + path.

        C# equivalent: [Route("api/v{v}/chat")] on a controller action.
        """

        route_key = (version, path)         # Tuple as dictionary key
        self.routes[route_key] = handler    # Map key -> handler function
        print(f"  Registered route: /{version}/{path} -> {handler.__name__}()")

    def dispatch(self, version, path, request):
        """
        Find and call the correct handler for the given version + path.

        C# equivalent: The routing middleware calling the matched action.
        """

        route_key = (version, path)         # Build the lookup key

        handler = self.routes.get(route_key)  # Look up the handler

        if handler is None:
            # No handler found -- return 404
            return {"status": 404, "error": f"Route /{version}/{path} not found"}

        return handler(request)             # Call the handler and return its result

    def print_routing_table(self):
        """Print all registered routes (like the route table in ASP.NET)."""

        print("\n  --- Routing Table ---")
        for (version, path), handler in self.routes.items():
            print(f"  /{version}/{path:<20} -> {handler.__name__}()")
        print("  --------------------")


# --- Define v1 and v2 handlers ---

def v1_chat_handler(request):
    """
    Version 1 of the chat endpoint.
    Simpler -- just takes message and returns a response.
    C# equivalent: [ApiVersion("1.0")] ChatController.Post()
    """

    message = request.get("message", "")   # Get message from request
    response_text = list(stream_llm_response(message))  # Get all words at once
    return {
        "status":   200,
        "version":  "v1",
        "response": "".join(response_text)  # Join words back into string
    }


def v2_chat_handler(request):
    """
    Version 2 of the chat endpoint.
    Adds context_window field (new feature in v2 that v1 doesn't have).
    C# equivalent: [ApiVersion("2.0")] ChatController.Post()
    """

    message        = request.get("message", "")
    context_window = request.get("context_window", 4096)  # NEW field in v2

    response_text = list(stream_llm_response(message))
    return {
        "status":         200,
        "version":        "v2",
        "response":       "".join(response_text),
        "context_window": context_window,              # v2 includes this extra field
        "tokens_used":    len("".join(response_text).split())  # v2 shows token count
    }


# --- Setup and demo the router ---

router = Router()                                              # Create router

print("\nRegistering API routes:")
router.register("v1", "chat", v1_chat_handler)                # Register v1 chat
router.register("v2", "chat", v2_chat_handler)                # Register v2 chat
router.register("v1", "health", lambda r: {"status": 200, "healthy": True})  # Health check

router.print_routing_table()                                   # Show all routes

print("\n-- Dispatching v1 request --")
v1_result = router.dispatch("v1", "chat", {"message": "Explain neural networks"})
print(f"  v1 response: {json.dumps(v1_result, indent=2)[:200]}...")

print("\n-- Dispatching v2 request --")
v2_result = router.dispatch("v2", "chat", {
    "message":        "Explain neural networks",
    "context_window": 8192                         # v2-specific field
})
print(f"  v2 response: {json.dumps(v2_result, indent=2)[:200]}...")

print("\n-- Dispatching unknown route (404) --")
not_found = router.dispatch("v3", "chat", {})     # v3 does not exist
print(f"  Result: {not_found}")

# =============================================================================
# PART 7 - KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 70)
print("PART 7 - KEY TAKEAWAYS")
print("=" * 70)

takeaways = [
    ("1. Request/Response Models",
     "Always validate incoming data before processing it. "
     "This is like ModelState.IsValid in ASP.NET -- bad data should be "
     "rejected at the API boundary, not deep inside business logic."),

    ("2. JWT Authentication",
     "JWTs are self-contained tokens that carry user identity without "
     "hitting the database on every request. Think of them as signed "
     "ID cards. ASP.NET Core uses the same concept with Bearer tokens."),

    ("3. Rate Limiting",
     "Protect your LLM API from abuse and runaway costs by limiting "
     "requests per user per time window. LLM calls are expensive -- "
     "one user should not be able to call it 10,000 times a minute."),

    ("4. Middleware Pipeline",
     "Process requests through a chain of responsibilities: auth -> "
     "rate limit -> log -> validate -> handle. Each step can reject the "
     "request early. This is identical to ASP.NET Core's middleware pipeline."),

    ("5. Streaming + Versioning",
     "Stream LLM tokens to improve user experience (users see output "
     "immediately). Version your API so clients are not broken when you "
     "add features. These are non-negotiable in production systems.")
]

for title, explanation in takeaways:
    print(f"\n  [{title}]")                   # Print the takeaway title
    # Wrap long lines for readability (like word-wrap in a text editor)
    words = explanation.split()               # Split into words
    line  = "  "                              # Start with indentation
    for word in words:
        if len(line) + len(word) > 72:        # If adding word exceeds 72 chars
            print(line)                        # Print current line
            line = "  " + word + " "          # Start new line with this word
        else:
            line += word + " "                 # Add word to current line
    if line.strip():
        print(line)                            # Print any remaining text

print("\n" + "=" * 70)
print("END OF EXAMPLE 01 - API DESIGN PATTERNS")
print("=" * 70)

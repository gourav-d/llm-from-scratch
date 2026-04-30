# =============================================================================
# Example 02: Deployment and Scalability Concepts
# Module 09 - Production LLM Applications
# =============================================================================
# GOAL: Simulate deployment patterns using pure Python stdlib.
#       No Docker, Kubernetes, Redis, or PostgreSQL installation needed.
#
# C# ANALOGY: Think of this as learning about IIS/Azure App Service deployment,
#             SqlConnectionPool, IMemoryCache, and health checks -- but
#             explained from first principles using only Python.
# =============================================================================

# --- IMPORTS (stdlib only, nothing to install) ---
import time           # For timestamps and simulated delays (like DateTime in C#)
import threading      # For Semaphore and concurrent simulation (like SemaphoreSlim in C#)
import json           # For JSON serialization (like JsonSerializer in C#)
import collections    # For OrderedDict used in LRU cache
import random         # For simulating random server choices (random.choice in C#)
import itertools      # For cycle() -- creates infinite round-robin iterator

# =============================================================================
# GLOSSARY
# =============================================================================
# Read these before looking at the code. Each term is used in the classes below.
# =============================================================================

GLOSSARY = """
=============================================================================
GLOSSARY - Deployment and Scalability Terms
=============================================================================

Containerization
  Packaging an application WITH all its dependencies into a single portable
  unit (a "container") that runs the same on any machine.
  Think of it like a shipping container -- same box, any ship, any port.
  C# equivalent: Publishing a self-contained .NET app to a Docker image.

Docker
  The most popular tool for creating and running containers.
  A "Dockerfile" is the recipe (like a .csproj + publish settings combined).

Image
  A read-only snapshot of your application + all its files and dependencies.
  Like a .nupkg or a published .zip of your app -- before it runs.

Container
  A running instance of an image.
  Same image can run as 5 containers simultaneously.
  C# analogy: image = DLL assembly, container = running process.

Orchestration
  Automatically managing many containers: starting, stopping, health checks,
  scaling up/down. Kubernetes is the most popular orchestrator.

Kubernetes (K8s)
  A system that runs many containers across many machines and keeps them
  healthy. If a container crashes, K8s automatically restarts it.
  C# analogy: Like IIS with auto-restart + Windows Services Manager combined.

Pod
  The smallest unit in Kubernetes. Usually contains one container (your app).
  C# analogy: One running instance of your ASP.NET app behind IIS.

Service
  A stable network address that load-balances traffic across all pods.
  Even if pods restart and get new IPs, the Service IP stays the same.
  C# analogy: A virtual IP / DNS name in a Windows load-balanced cluster.

Load Balancer
  Distributes incoming requests across multiple pods/servers so no single
  server is overloaded.
  C# analogy: ARR (Application Request Routing) in IIS or Azure Front Door.

Cache
  A fast in-memory store for results you have already computed.
  "Cache hit" = found in cache (fast!). "Cache miss" = must compute it.
  C# analogy: IMemoryCache or IDistributedCache (Redis) in ASP.NET Core.

Connection Pool
  A reusable pool of database connections. Opening a new DB connection is
  expensive (~100ms), so you keep a pool of open connections ready to reuse.
  C# analogy: SqlConnection pooling (built into ADO.NET automatically).

Horizontal Scaling
  Adding MORE servers/pods to handle more traffic (scale OUT).
  Opposite: Vertical scaling = making one server BIGGER (scale UP).
  C# analogy: Adding more Azure App Service instances.

Health Check
  A simple endpoint (/health or /readiness) that returns OK or ERROR.
  Load balancers and Kubernetes use this to know if a pod is alive.
  C# analogy: ASP.NET Core Health Checks (app.MapHealthChecks("/health")).

=============================================================================
"""

print(GLOSSARY)   # Print glossary first so student reads it before the code

# =============================================================================
# ASCII DIAGRAM: DEPLOYMENT ARCHITECTURE
# =============================================================================
# This is what a production LLM application infrastructure looks like.
# Each box in this diagram corresponds to a class we build below.
# =============================================================================

DEPLOYMENT_DIAGRAM = """
=============================================================================
DEPLOYMENT ARCHITECTURE DIAGRAM
=============================================================================

     Internet
        |
        v
  [Client Browser / App]
        |
        | HTTP Request
        v
  +------------------+
  |  Load Balancer   |  <-- Part 4: Distributes traffic round-robin
  +------------------+
     /      |       \\
    /       |        \\
   v        v         v
+------+ +------+ +------+
| Pod1 | | Pod2 | | Pod3 |  <-- Your LLM app running in 3 containers
+------+ +------+ +------+
   |        |         |
   +--------+---------+
            |
            v
    +---------------+
    |  Cache Layer  |   <-- Part 3: Redis-style LRU cache (avoids repeat LLM calls)
    +---------------+
            |
            v
    +---------------+
    | Connection    |   <-- Part 2: Shared DB connection pool
    |     Pool      |
    +---------------+
            |
            v
    +---------------+
    |  PostgreSQL   |   <-- Actual database (simulated here)
    +---------------+

  Health Checker --> probes all pods --> reports healthy/degraded/unhealthy
  Config Manager --> reads env vars  --> masks secrets in logs

=============================================================================
"""

print(DEPLOYMENT_DIAGRAM)   # Show the architecture diagram

# =============================================================================
# PART 1 - CONFIGURATION MANAGEMENT (12-FACTOR APP STYLE)
# =============================================================================
# The "12-Factor App" is a methodology for building production-ready apps.
# Factor #3: "Store config in the environment" -- no hardcoded values in code!
#
# C# equivalent: IConfiguration in ASP.NET Core (appsettings.json + env vars)
#   builder.Configuration.GetValue<string>("Database:ConnectionString")
#   Environment.GetEnvironmentVariable("DB_PASSWORD")
# =============================================================================

print("=" * 70)
print("PART 1 - CONFIGURATION MANAGEMENT (12-FACTOR STYLE)")
print("=" * 70)

# Simulates what environment variables look like (os.environ in real code)
# In production: docker run -e DB_HOST=prod-db.internal ...
# In ASP.NET:    Environment.GetEnvironmentVariable("DB_HOST")
SIMULATED_ENV_VARS = {
    "APP_ENV":          "production",         # Are we in dev/staging/production?
    "DB_HOST":          "prod-db.internal",   # Database server hostname
    "DB_PORT":          "5432",               # PostgreSQL default port
    "DB_NAME":          "llm_app",            # Database name
    "DB_PASSWORD":      "s3cr3tP@ssw0rd!",    # NEVER hardcode this in real code!
    "OPENAI_API_KEY":   "sk-1234567890abcdef", # LLM provider API key (SECRET!)
    "MAX_CONNECTIONS":  "10",                  # Connection pool size
    "CACHE_TTL":        "300",                 # Cache Time-To-Live in seconds
    "LOG_LEVEL":        "INFO",               # Logging verbosity
    # "REDIS_URL" is intentionally MISSING to demo the require() method
}

# List of config keys that contain secrets and must be masked in logs
SECRET_KEYS = {"DB_PASSWORD", "OPENAI_API_KEY", "SECRET_KEY", "API_TOKEN"}


class AppConfig:
    """
    Manages application configuration loaded from environment variables.

    C# equivalent:
        IConfiguration configuration = builder.Configuration;
        var dbHost = configuration["DB_HOST"];
        var dbPass = configuration.GetValue<string>("DB_PASSWORD")
                    ?? throw new InvalidOperationException("DB_PASSWORD required");
    """

    def __init__(self, env_dict):
        # Store the config dictionary (simulates os.environ)
        # In real code: self._config = dict(os.environ)
        self._config = dict(env_dict)   # Make a copy so we don't modify the original

    def get(self, key, default=None):
        """
        Retrieve a config value by key. Returns default if not found.

        C# equivalent:
            configuration.GetValue<string>("KEY") ?? defaultValue;
        """

        return self._config.get(key, default)   # Dict.get() returns None if missing

    def get_int(self, key, default=0):
        """
        Retrieve a config value and convert to integer.

        C# equivalent:
            configuration.GetValue<int>("MAX_CONNECTIONS", 10);
        """

        value = self._config.get(key, str(default))  # Get value as string
        try:
            return int(value)          # Convert string -> int
        except (ValueError, TypeError):
            return default             # Return default if conversion fails

    def require(self, key):
        """
        Retrieve a config value that MUST exist. Raises if missing.
        Use this for critical settings -- fail fast at startup is better
        than mysterious errors later.

        C# equivalent:
            var value = configuration["KEY"]
                ?? throw new InvalidOperationException($"{key} is required");
        """

        value = self._config.get(key)      # Try to get the value
        if value is None:
            # Raise an error explaining what is missing and how to fix it
            raise ValueError(
                f"Required config key '{key}' is missing. "
                f"Set it as an environment variable: export {key}=your_value"
            )
        return value    # Return the value if it exists

    def mask_secrets(self):
        """
        Return a copy of the config with secret values replaced by ***.
        ALWAYS use this when printing config to logs -- never log raw secrets!

        C# equivalent: Redacting sensitive values before logging with Serilog.
        """

        masked = {}                                    # Start with empty dict
        for key, value in self._config.items():        # Loop through all config
            if key in SECRET_KEYS:                     # Is this a secret key?
                masked[key] = "***REDACTED***"         # Replace value with ***
            else:
                masked[key] = value                    # Keep non-secret values

        return masked   # Return the safe-to-log version


# --- Demo: AppConfig in action ---

config = AppConfig(SIMULATED_ENV_VARS)   # Create config from simulated env vars

print("\n-- Good config examples (safe to log) --")
safe_config = config.mask_secrets()                    # Get masked version
for key, value in safe_config.items():
    print(f"  {key:<20} = {value}")                    # Print each key=value

print("\n-- Reading specific values --")
print(f"  APP_ENV      : {config.get('APP_ENV')}")     # "production"
print(f"  DB_HOST      : {config.get('DB_HOST')}")     # hostname
print(f"  MAX_CONN     : {config.get_int('MAX_CONNECTIONS', 5)}")  # 10 as integer
print(f"  MISSING_KEY  : {config.get('MISSING_KEY', 'default_val')}")  # default

print("\n-- require() raises error for missing critical config --")
try:
    redis_url = config.require("REDIS_URL")            # This key is missing
except ValueError as e:
    print(f"  Caught: {e}")                            # Show the helpful error message

print("\n-- Bad config example: secrets visible in logs (DO NOT do this!) --")
raw_password = config.get("DB_PASSWORD")              # Gets the real value
print(f"  DB_PASSWORD (raw -- BAD!) = {raw_password}")  # Shows the secret -- bad!
print("  --> Always use mask_secrets() before logging config!")

# =============================================================================
# PART 2 - CONNECTION POOL (SIMULATES DATABASE POOL)
# =============================================================================
# Opening a new database connection takes ~50-100ms and uses DB server resources.
# A connection pool keeps N connections open and ready to reuse.
#
# C# equivalent: ADO.NET SqlConnection pooling (automatic) or
#                Npgsql NpgsqlConnection pooling (PostgreSQL in .NET)
#
# In C#, connection pooling is mostly invisible -- it happens automatically
# when you use "using var conn = new SqlConnection(connectionString)".
# Here we make it EXPLICIT so you can see how it works internally.
# =============================================================================

print("\n" + "=" * 70)
print("PART 2 - CONNECTION POOL (SIMULATES DB POOL)")
print("=" * 70)

class FakeDBConnection:
    """
    Simulates a database connection object.
    In real code this would be a psycopg2.connection or SqlConnection.

    C# equivalent: System.Data.SqlClient.SqlConnection
    """

    def __init__(self, conn_id):
        self.conn_id    = conn_id       # Unique ID for this connection
        self.is_in_use  = False         # Is this connection currently checked out?
        self.created_at = time.time()   # When this connection was opened

    def query(self, sql):
        """Simulate running a SQL query (just prints it)."""
        print(f"    [Conn #{self.conn_id}] Executing: {sql[:50]}")
        time.sleep(0.01)               # Simulate 10ms query time
        return {"rows": 42, "status": "ok"}  # Fake result

    def __repr__(self):
        status = "IN-USE" if self.is_in_use else "idle"
        return f"DBConn(id={self.conn_id}, status={status})"


class ConnectionPool:
    """
    Manages a fixed-size pool of reusable database connections.

    ANALOGY: Like a hotel with 10 rooms (connections).
    Guests (requests) check in (acquire) and check out (release).
    If all rooms are full, new guests WAIT until a room is free.

    C# equivalent: The built-in SqlConnection pool, or
                   NpgsqlDataSource with UseConnectionPool() in .NET
    """

    def __init__(self, max_connections, connection_factory):
        # Maximum number of connections allowed at once
        self.max_connections     = max_connections

        # Factory function that creates a new connection when needed
        # Like Func<SqlConnection> in C#
        self.connection_factory  = connection_factory

        # The actual pool: list of connection objects
        self.pool                = []

        # Semaphore limits concurrent access to max_connections
        # C# equivalent: SemaphoreSlim(max_connections, max_connections)
        self.semaphore           = threading.Semaphore(max_connections)

        # Lock to protect pool list from concurrent modification
        # C# equivalent: lock(pool) { ... }
        self.lock                = threading.Lock()

        # Counter for generating unique connection IDs
        self._conn_counter       = 0

        # Pre-create the connections (eager initialization)
        for _ in range(max_connections):
            self._conn_counter += 1                         # Increment counter
            conn = connection_factory(self._conn_counter)   # Create a new connection
            self.pool.append(conn)                          # Add to pool

    def acquire(self, timeout=5.0):
        """
        Check out a connection from the pool.
        Waits up to timeout seconds if all connections are in use.
        Raises TimeoutError if no connection becomes available.

        C# equivalent:
            var conn = await dataSource.OpenConnectionAsync();
            // or with SqlConnection pool: conn.Open() waits automatically
        """

        # Try to acquire the semaphore (blocks until a slot is free or timeout)
        # semaphore.acquire(timeout) = SemaphoreSlim.Wait(TimeSpan.FromSeconds(5))
        acquired = self.semaphore.acquire(timeout=timeout)   # Block until available

        if not acquired:
            # Semaphore timed out -- no connection available
            raise TimeoutError(
                f"Could not get a DB connection within {timeout}s. "
                f"Pool is exhausted ({self.max_connections} connections all in use)."
            )

        # We have permission -- now find an idle connection in the pool
        with self.lock:                    # Thread-safe access to the pool list
            for conn in self.pool:
                if not conn.is_in_use:     # Find one that is currently free
                    conn.is_in_use = True  # Mark it as in-use
                    return conn            # Return it to the caller

    def release(self, conn):
        """
        Return a connection back to the pool when done.
        ALWAYS call this in a finally block so it's released even on error.

        C# equivalent: conn.Dispose() or using(var conn = pool.Get()) { ... }
        """

        with self.lock:                    # Thread-safe access to the pool list
            conn.is_in_use = False         # Mark connection as available again

        self.semaphore.release()           # Signal that a slot is now free

    def stats(self):
        """
        Return current pool statistics.

        C# equivalent: SqlConnection.ClearAllPools() diagnostics or
                       Npgsql's NpgsqlDataSource.Statistics
        """

        with self.lock:
            total  = len(self.pool)                          # Total connections
            in_use = sum(1 for c in self.pool if c.is_in_use)  # Count in-use
            idle   = total - in_use                          # Count idle

        return {
            "total":   total,    # Total connections in pool
            "active":  in_use,   # Currently checked out
            "idle":    idle      # Available to use
        }


# --- Demo: 5 concurrent requests sharing a 3-connection pool ---

pool = ConnectionPool(
    max_connections    = 3,                           # Only 3 connections
    connection_factory = FakeDBConnection             # FakeDBConnection(conn_id)
)

print(f"\nCreated pool with 3 connections. Initial stats: {pool.stats()}")
print("\nSimulating 5 sequential requests sharing the 3-connection pool:")

for i in range(1, 6):                                # Loop 5 "requests"
    print(f"\n  Request {i}: acquiring connection...")
    try:
        conn = pool.acquire(timeout=2.0)             # Get a connection (wait up to 2s)
        print(f"  Request {i}: got {conn}")
        print(f"  Pool stats: {pool.stats()}")

        conn.query(f"SELECT * FROM llm_logs WHERE request_id = {i}")  # Fake query

        pool.release(conn)                           # ALWAYS release!
        print(f"  Request {i}: released connection. Pool stats: {pool.stats()}")
    except TimeoutError as e:
        print(f"  Request {i}: FAILED - {e}")        # Show timeout message

# =============================================================================
# PART 3 - CACHE LAYER (LRU CACHE, PURE PYTHON)
# =============================================================================
# LRU = Least Recently Used.
# When the cache is full, it removes the item that was used LEAST RECENTLY.
# This is like clearing your browser history of the oldest/least-used pages.
#
# WHY CACHE LLM RESPONSES?
#   LLM API calls can cost $0.001-$0.05 each and take 500ms-5 seconds.
#   If the same question is asked 1000 times, why pay 1000 times?
#   Cache the first answer and return it instantly for all future identical requests.
#
# C# equivalent: IMemoryCache in ASP.NET Core
#   cache.Set("key", value, TimeSpan.FromSeconds(300));
#   cache.TryGetValue("key", out var cached);
# =============================================================================

print("\n" + "=" * 70)
print("PART 3 - CACHE LAYER (LRU CACHE, PURE PYTHON)")
print("=" * 70)

class SimpleCache:
    """
    LRU (Least Recently Used) cache with TTL (Time-To-Live) expiry.

    C# equivalent:
        var cache = new MemoryCache(new MemoryCacheOptions());
        cache.Set(key, value, TimeSpan.FromSeconds(ttl));
        cache.TryGetValue(key, out T result);
    """

    def __init__(self, max_size=100):
        # OrderedDict remembers insertion order (like LinkedList<> in C#)
        # We use it to track which items were accessed LEAST recently
        self._store    = collections.OrderedDict()  # key -> (value, expires_at)

        # Maximum number of items to keep in cache
        self.max_size  = max_size

        # Statistics counters
        self._hits     = 0   # Number of times we found something in cache
        self._misses   = 0   # Number of times we had to compute (cache miss)

        # Lock for thread safety (multiple threads can access cache at once)
        self._lock     = threading.Lock()

    def set(self, key, value, ttl_seconds=300):
        """
        Store a value in the cache with an expiry time.

        C# equivalent:
            _cache.Set(key, value, new MemoryCacheEntryOptions {
                AbsoluteExpirationRelativeToNow = TimeSpan.FromSeconds(ttl_seconds)
            });
        """

        with self._lock:                           # Thread-safe write
            expires_at = time.time() + ttl_seconds  # Calculate expiry timestamp

            # If key already exists, remove it first (to re-insert at end = most recent)
            if key in self._store:
                del self._store[key]               # Remove old entry

            # If cache is full, remove the LEAST recently used item (at the front)
            if len(self._store) >= self.max_size:
                self._store.popitem(last=False)    # Remove oldest (front) item

            # Store the new entry (added at the end = most recently used)
            self._store[key] = (value, expires_at)  # Tuple: (value, expiry time)

    def get(self, key):
        """
        Retrieve a value from cache, or return None if missing/expired.

        C# equivalent:
            if (_cache.TryGetValue(key, out T value)) return value;
            return null;
        """

        with self._lock:                           # Thread-safe read
            if key not in self._store:
                self._misses += 1                  # Count as a miss
                return None                        # Not in cache

            value, expires_at = self._store[key]   # Unpack stored tuple

            if time.time() > expires_at:           # Has it expired?
                del self._store[key]               # Remove the expired entry
                self._misses += 1                  # Count as a miss
                return None                        # Return None (as if not found)

            # Cache HIT -- move to end (most recently used)
            self._store.move_to_end(key)           # Mark as recently used
            self._hits += 1                        # Count as a hit
            return value                           # Return the cached value

    def delete(self, key):
        """
        Remove a specific key from the cache (cache invalidation).

        C# equivalent: _cache.Remove(key);
        """

        with self._lock:
            if key in self._store:
                del self._store[key]               # Remove if it exists

    def stats(self):
        """
        Return cache performance statistics.

        C# equivalent: Reading MemoryCache.Count and custom hit/miss counters.
        """

        total_requests = self._hits + self._misses  # Total get() calls
        hit_rate = (                                 # Percentage of cache hits
            self._hits / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            "cache_size":       len(self._store),   # Items currently in cache
            "max_size":         self.max_size,
            "hits":             self._hits,          # Cache hits
            "misses":           self._misses,        # Cache misses
            "total_requests":   total_requests,
            "hit_rate":         f"{hit_rate:.1%}",   # e.g., "75.0%"
        }


def simulate_llm_api_call(prompt):
    """
    Simulates an expensive LLM API call (takes ~500ms in real life).
    We just sleep briefly and return a fake response.
    """

    time.sleep(0.05)                                # Simulate 50ms API call
    # Return a fake response based on the prompt
    return f"[LLM Response to: {prompt[:30]}...] The answer involves..."


# --- Demo: LLM response caching ---

cache = SimpleCache(max_size=10)                    # Cache holds max 10 items

# Five different prompts (some will repeat to show cache hits)
prompts = [
    "What is attention mechanism?",                 # First time -- cache MISS
    "How do transformers work?",                    # First time -- cache MISS
    "What is attention mechanism?",                 # Repeat -- cache HIT!
    "Explain backpropagation",                      # First time -- cache MISS
    "What is attention mechanism?",                 # Repeat again -- cache HIT!
]

print("\nCaching LLM API responses to avoid repeat calls:\n")

for prompt in prompts:
    key = f"llm:{prompt}"                           # Build a cache key for this prompt

    # Check cache first (like TryGetValue in C#)
    cached_response = cache.get(key)

    if cached_response is not None:
        # Cache HIT -- no need to call the LLM API
        print(f"  CACHE HIT!  Saved ~500ms API call for: {prompt[:40]!r}")
        print(f"             Response: {cached_response[:50]}...")
    else:
        # Cache MISS -- must call the LLM API (slow and expensive)
        print(f"  Cache miss. Calling LLM API for: {prompt[:40]!r}")
        response = simulate_llm_api_call(prompt)    # Slow API call
        cache.set(key, response, ttl_seconds=300)   # Store result for 5 minutes
        print(f"             Response cached: {response[:50]}...")

print(f"\nCache statistics: {json.dumps(cache.stats(), indent=2)}")

# =============================================================================
# PART 4 - LOAD BALANCER (ROUND-ROBIN SIMULATION)
# =============================================================================
# A load balancer distributes incoming requests across multiple servers.
# "Round-robin" means: send request 1 to server A, request 2 to server B,
# request 3 to server C, request 4 back to server A, and so on.
#
# When a server goes unhealthy (crashes, slow), the load balancer stops
# sending traffic to it until it recovers.
#
# C# equivalent: Azure Load Balancer, NGINX, or AWS ALB in front of your
#                ASP.NET Core app instances in a web farm.
# =============================================================================

print("\n" + "=" * 70)
print("PART 4 - LOAD BALANCER (ROUND-ROBIN SIMULATION)")
print("=" * 70)

class LoadBalancer:
    """
    Round-robin load balancer with health awareness.

    C# equivalent: No direct stdlib equivalent, but conceptually like
                   IHttpClientFactory with multiple named clients, or
                   Polly library's circuit breaker + multiple backends.
    """

    def __init__(self):
        # List of all registered backend URLs
        self.backends = []

        # Set of currently unhealthy backend URLs (removed from rotation)
        self.unhealthy = set()         # set() is like HashSet<string> in C#

        # Round-robin iterator (cycles through backends endlessly)
        # itertools.cycle(["A","B","C"]) -> A, B, C, A, B, C, A, ...
        self._cycle = None             # Will be set after backends are added

        # Lock for thread safety when modifying the backend list
        self._lock = threading.Lock()

    def add_backend(self, url, weight=1):
        """
        Register a new backend server.
        Weight is ignored in this simple version (all backends equal).

        C# equivalent: Adding a server to a web farm in IIS configuration.
        """

        with self._lock:
            self.backends.append(url)          # Add URL to the list
            self._rebuild_cycle()              # Rebuild the round-robin cycle
            print(f"  Added backend: {url}")

    def _rebuild_cycle(self):
        """
        Rebuild the round-robin cycle with only HEALTHY backends.
        Private method (the _ prefix = private in Python convention).
        """

        # Filter to only healthy backends
        healthy = [b for b in self.backends if b not in self.unhealthy]

        if healthy:
            # itertools.cycle creates an infinite repeating iterator
            # Like: while(true) { foreach(var b in healthy) yield return b; }
            self._cycle = itertools.cycle(healthy)  # Round-robin through healthy
        else:
            self._cycle = None              # No healthy backends!

    def get_next(self):
        """
        Get the next backend URL in round-robin order.
        Returns None if no healthy backends are available.

        C# equivalent: The load balancer hardware/software automatically
                       selecting the next server.
        """

        with self._lock:
            if self._cycle is None:
                return None              # No healthy backends
            return next(self._cycle)    # Get next in cycle (never runs out)

    def mark_unhealthy(self, url):
        """
        Remove a backend from rotation (e.g., it failed a health check).

        C# equivalent: Removing a server from the IIS web farm.
        """

        with self._lock:
            self.unhealthy.add(url)         # Add to unhealthy set
            self._rebuild_cycle()           # Rebuild cycle without this backend
            print(f"  [ALERT] Backend UNHEALTHY: {url} -- removed from rotation")

    def mark_healthy(self, url):
        """
        Restore a backend to rotation after it recovers.

        C# equivalent: Adding a server back to the IIS web farm after recovery.
        """

        with self._lock:
            self.unhealthy.discard(url)     # Remove from unhealthy set (no error if missing)
            self._rebuild_cycle()           # Rebuild cycle with this backend included
            print(f"  [INFO] Backend RECOVERED: {url} -- added back to rotation")

    def health_check(self):
        """
        Simulate checking all backends (in real life, sends HTTP GET /health).

        C# equivalent: BackgroundService that calls HttpClient.GetAsync("/health")
        """

        print("\n  -- Health Check Results --")
        for backend in self.backends:
            status = "HEALTHY" if backend not in self.unhealthy else "UNHEALTHY"
            marker = "OK" if status == "HEALTHY" else "!!"
            print(f"  [{marker}] {backend:<35} -> {status}")


# --- Demo: 10 requests across 3 backends, one goes unhealthy ---

lb = LoadBalancer()

print("\nRegistering backends:")
lb.add_backend("http://pod-1.internal:8080")    # Register pod 1
lb.add_backend("http://pod-2.internal:8080")    # Register pod 2
lb.add_backend("http://pod-3.internal:8080")    # Register pod 3

print("\nDispatching 10 requests (pods fail after request 5):\n")

for i in range(1, 11):
    # Simulate pod-2 going unhealthy after request 5
    if i == 6:
        lb.mark_unhealthy("http://pod-2.internal:8080")  # Pod 2 crashes

    backend = lb.get_next()                  # Get next available backend
    print(f"  Request {i:02d} -> {backend}")  # Show where it was routed

lb.health_check()                            # Show final health status

# Bring pod-2 back online
print()
lb.mark_healthy("http://pod-2.internal:8080")   # Pod 2 recovers
lb.health_check()                               # Show updated health status

# =============================================================================
# PART 5 - HEALTH CHECK ENDPOINT SIMULATION
# =============================================================================
# Every production service exposes a /health endpoint that reports its status.
# Kubernetes probes this every few seconds. If it returns unhealthy, K8s
# restarts the pod or stops routing traffic to it.
#
# Three possible overall statuses:
#   "healthy"   = all checks pass (like 200 OK)
#   "degraded"  = some non-critical checks fail (like 207 Multi-Status)
#   "unhealthy" = critical checks fail (like 503 Service Unavailable)
#
# C# equivalent: ASP.NET Core Health Checks
#   builder.Services.AddHealthChecks()
#       .AddNpgSql(connectionString)
#       .AddRedis(redisConnectionString);
#   app.MapHealthChecks("/health");
# =============================================================================

print("\n" + "=" * 70)
print("PART 5 - HEALTH CHECK ENDPOINT SIMULATION")
print("=" * 70)

class HealthChecker:
    """
    Runs named health checks and aggregates their results.

    C# equivalent: IHealthCheckService in ASP.NET Core:
        services.AddHealthChecks()
            .AddCheck("database", () => HealthCheckResult.Healthy("DB OK"));
    """

    def __init__(self):
        # Dictionary of check_name -> (check_function, is_critical)
        # is_critical: if True and failing, overall status is "unhealthy"
        # is_critical: if False and failing, overall status is "degraded"
        self.checks = {}   # Like Dictionary<string, Func<HealthCheckResult>> in C#

    def register_check(self, name, check_fn, critical=True):
        """
        Register a named health check function.

        check_fn must return (bool, str): (is_healthy, message)

        C# equivalent:
            services.AddHealthChecks().AddCheck(name, () => result);
        """

        self.checks[name] = (check_fn, critical)   # Store function + criticality flag
        print(f"  Registered health check: {name!r} (critical={critical})")

    def run_all(self):
        """
        Execute all registered health checks.
        Returns a dict of name -> {"status", "message", "duration_ms"}.

        C# equivalent: IHealthCheckService.CheckHealthAsync()
        """

        results = {}                               # Results dict

        for name, (check_fn, is_critical) in self.checks.items():
            start_time = time.time()               # Record start time

            try:
                is_healthy, message = check_fn()   # Call the check function
                status = "healthy" if is_healthy else "unhealthy"
            except Exception as e:
                # If the check itself crashes, treat as unhealthy
                is_healthy = False
                status     = "unhealthy"
                message    = f"Check raised exception: {e}"

            duration_ms = (time.time() - start_time) * 1000  # ms elapsed

            # Store the result for this check
            results[name] = {
                "status":      status,              # "healthy" or "unhealthy"
                "message":     message,             # Human-readable detail
                "critical":    is_critical,         # Is this check critical?
                "duration_ms": round(duration_ms, 2)
            }

        return results   # Return all results

    def overall_status(self):
        """
        Compute the overall health status based on all check results.

        Rules:
          - Any CRITICAL check fails  -> "unhealthy"
          - Any non-critical fails    -> "degraded"
          - All pass                  -> "healthy"

        C# equivalent: HealthStatus enum (Healthy / Degraded / Unhealthy)
        """

        results           = self.run_all()          # Run all checks
        has_critical_fail = False                   # Flag for critical failures
        has_degraded      = False                   # Flag for non-critical failures

        for name, result in results.items():
            if result["status"] == "unhealthy":
                if result["critical"]:
                    has_critical_fail = True        # Critical failure detected
                else:
                    has_degraded = True             # Non-critical degradation

        # Determine overall status
        if has_critical_fail:
            overall = "unhealthy"                   # 503 Service Unavailable
        elif has_degraded:
            overall = "degraded"                    # 207 / partial failure
        else:
            overall = "healthy"                     # 200 OK

        return overall, results   # Return both the summary and the detailed results


# --- Define some simulated health check functions ---

def database_check():
    """Simulates checking if the database is reachable."""
    time.sleep(0.02)                           # Simulate 20ms DB ping
    # In real code: run "SELECT 1" and check if it returns
    return (True, "PostgreSQL responding normally. Latency: 18ms")


def redis_cache_check():
    """Simulates checking if Redis cache is reachable."""
    time.sleep(0.01)                           # Simulate 10ms ping
    # Simulate Redis being DOWN (connection refused)
    return (False, "Redis: Connection refused at redis.internal:6379")


def llm_api_check():
    """Simulates checking if the LLM provider API is reachable."""
    time.sleep(0.05)                           # Simulate 50ms ping
    return (True, "OpenAI API responding. Model: gpt-4o-mini available")


def disk_space_check():
    """Simulates checking if there is enough disk space."""
    # Simulated disk usage (in real code: shutil.disk_usage('/'))
    disk_usage_pct = 72                        # Pretend disk is 72% full
    if disk_usage_pct > 90:
        return (False, f"Disk {disk_usage_pct}% full -- CRITICAL")
    return (True, f"Disk {disk_usage_pct}% used -- OK")


# --- Setup and run health checks ---

checker = HealthChecker()   # Create health checker

print("\nRegistering health checks:")
checker.register_check("database",  database_check,   critical=True)   # Must work
checker.register_check("redis",     redis_cache_check, critical=False)  # Nice to have
checker.register_check("llm_api",   llm_api_check,    critical=True)   # Must work
checker.register_check("disk_space",disk_space_check, critical=False)  # Nice to have

print("\nRunning all health checks...")
overall, results = checker.overall_status()   # Run everything

print(f"\nOverall Status: {overall.upper()}")  # Print the summary
print("\nDetailed Results:")

for name, result in results.items():
    # Choose a text marker based on status
    marker = "OK" if result["status"] == "healthy" else "!!"
    crit   = "CRITICAL" if result["critical"] else "optional"

    print(f"  [{marker}] {name:<15} ({crit:<8}) "
          f"-> {result['status']:<10} | {result['message'][:50]}")
    print(f"       Duration: {result['duration_ms']}ms")

# =============================================================================
# PART 6 - DOCKER CONFIG GENERATOR
# =============================================================================
# Docker uses text files (Dockerfile, docker-compose.yml) to define how to
# build and run your application container.
#
# We generate these files as STRINGS (pure Python, no Docker needed).
# This shows you what these files look like without installing anything.
#
# C# equivalent: Publishing your .NET app to a Docker image:
#   docker build -t myapp .
#   docker run -p 8080:80 myapp
# =============================================================================

print("\n" + "=" * 70)
print("PART 6 - DOCKER CONFIG GENERATOR")
print("=" * 70)

class DockerfileGenerator:
    """
    Generates Dockerfile and docker-compose.yml content as strings.

    In a real project you would write these files to disk.
    Here we just print them to show what they look like.

    C# equivalent: No direct analogy -- Docker is Docker. But the concept
                   is similar to generating .csproj or web.config files
                   programmatically.
    """

    def generate_dockerfile(self, base_image, app_port, requirements):
        """
        Generate a Dockerfile string for a Python LLM application.

        Args:
            base_image   : e.g., "python:3.11-slim"
            app_port     : e.g., 8000
            requirements : list of package names, e.g., ["openai", "fastapi"]
        """

        # Join the requirements list into a single string for the pip command
        # Like string.Join(" ", packages) in C#
        req_string = " ".join(requirements)  # "openai fastapi uvicorn"

        # Build the Dockerfile as a multi-line string
        # Each line is a Docker instruction (FROM, WORKDIR, COPY, RUN, CMD)
        dockerfile = f"""# ==========================================================
# Dockerfile for LLM Application
# Generated by DockerfileGenerator
# ==========================================================

# Start from an official Python slim image (smaller than full image)
# This is like choosing a base OS for your app
FROM {base_image}

# Set the working directory inside the container
# Like setting the current directory in a terminal
WORKDIR /app

# Copy requirements file first (Docker layer caching optimization)
# This layer only rebuilds if requirements.txt changes
COPY requirements.txt .

# Install Python dependencies
# Like "dotnet restore" for .NET projects
RUN pip install --no-cache-dir {req_string}

# Copy the rest of the application code
COPY . .

# Tell Docker which port your app listens on (documentation only)
# Like specifying the port in launchSettings.json in .NET
EXPOSE {app_port}

# Create a non-root user for security (never run as root in production!)
# C# equivalent: running IIS app pool as a restricted user
RUN useradd --create-home appuser
USER appuser

# The command to start your application
# Like setting the startup project and run command in .NET
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{app_port}"]
"""
        return dockerfile   # Return as a string

    def generate_docker_compose(self, services):
        """
        Generate a docker-compose.yml string for multi-service deployment.

        Args:
            services: list of dicts with keys:
                      name, image, port, env_vars (dict)

        docker-compose is like an orchestration config that starts
        multiple services at once with: docker compose up
        C# analogy: Running multiple .NET services together in a solution.
        """

        # Start building the YAML content (docker-compose uses YAML format)
        lines = []                             # List of lines to join at the end
        lines.append("# docker-compose.yml - Generated by DockerfileGenerator")
        lines.append("# Run with: docker compose up --build")
        lines.append("# Stop with: docker compose down")
        lines.append("")                       # Blank line
        lines.append("version: '3.8'")        # docker-compose version
        lines.append("")
        lines.append("services:")             # Begin services section

        for svc in services:                   # Loop through each service definition
            name     = svc["name"]             # Service name (e.g., "app", "redis")
            image    = svc["image"]            # Docker image to use
            port     = svc.get("port")         # Host:Container port mapping
            env_vars = svc.get("env_vars", {}) # Environment variables for this service

            lines.append(f"  {name}:")         # Service name (2-space indent in YAML)
            lines.append(f"    image: {image}")  # Which image to use

            if port:
                lines.append( "    ports:")
                lines.append(f"      - \"{port}\"")  # Port mapping e.g., "8080:8000"

            if env_vars:
                lines.append( "    environment:")
                for key, value in env_vars.items():
                    lines.append(f"      - {key}={value}")  # Each env var

            lines.append( "    restart: unless-stopped")  # Auto-restart on crash
            lines.append("")                   # Blank line between services

        # Add a shared network (so services can talk to each other by name)
        lines.append("networks:")
        lines.append("  default:")
        lines.append("    driver: bridge")     # Standard Docker network type

        return "\n".join(lines)               # Join all lines into one string


# --- Demo: Generate real-looking Docker configs ---

generator = DockerfileGenerator()   # Create the generator

print("\n-- Generated Dockerfile --")
print("-" * 60)
dockerfile_content = generator.generate_dockerfile(
    base_image   = "python:3.11-slim",
    app_port     = 8000,
    requirements = ["openai", "fastapi", "uvicorn", "pydantic"]
)
print(dockerfile_content)           # Print the generated Dockerfile

print("-" * 60)
print("\n-- Generated docker-compose.yml --")
print("-" * 60)

compose_services = [
    {
        "name":     "llm-app",                  # Our main LLM application
        "image":    "mycompany/llm-app:1.0.0",  # The image we built
        "port":     "8080:8000",                # Expose container port 8000 as 8080
        "env_vars": {
            "APP_ENV":    "production",
            "DB_HOST":    "postgres",           # Uses service name, not IP
            "REDIS_URL":  "redis://redis:6379", # Uses service name, not IP
        }
    },
    {
        "name":     "postgres",                 # PostgreSQL database service
        "image":    "postgres:15-alpine",       # Official Postgres image
        "port":     "5432:5432",
        "env_vars": {
            "POSTGRES_DB":       "llm_app",
            "POSTGRES_USER":     "appuser",
            "POSTGRES_PASSWORD": "changeme"     # Override in real deployment!
        }
    },
    {
        "name":  "redis",                       # Redis cache service
        "image": "redis:7-alpine",              # Official Redis image
        "port":  "6379:6379"                    # No env_vars needed for basic Redis
    }
]

compose_content = generator.generate_docker_compose(compose_services)
print(compose_content)          # Print the generated docker-compose.yml

print("-" * 60)

# =============================================================================
# PART 7 - KEY TAKEAWAYS
# =============================================================================

print("\n" + "=" * 70)
print("PART 7 - KEY TAKEAWAYS")
print("=" * 70)

takeaways = [
    ("1. Configuration Management",
     "Never hardcode secrets or environment-specific values. Use "
     "environment variables (12-Factor App principle #3). In C# this is "
     "IConfiguration reading from appsettings.json + environment overrides. "
     "Always mask secrets before logging config values."),

    ("2. Connection Pooling",
     "Opening a database connection is expensive (~50-100ms). A pool "
     "keeps connections open and reuses them. In .NET, ADO.NET does this "
     "automatically for SqlConnection. Always call release() in a finally "
     "block (like using{} in C#) to avoid pool exhaustion."),

    ("3. Caching",
     "Cache expensive LLM API responses to save money and reduce latency. "
     "A cache hit returns instantly; a cache miss calls the LLM. Use TTL "
     "(time-to-live) so stale answers expire. In C# this is IMemoryCache "
     "or IDistributedCache (Redis). Cache hit rates of 30-70% are common "
     "in real LLM applications."),

    ("4. Load Balancing and Horizontal Scaling",
     "Run multiple copies of your app (pods) behind a load balancer. "
     "Round-robin distribution spreads the load evenly. Remove unhealthy "
     "pods automatically. In Azure this is Azure Load Balancer + App Service "
     "scale-out. This is how you handle 1,000 requests per second."),

    ("5. Health Checks and Observability",
     "Every production service needs a /health endpoint. Kubernetes and "
     "load balancers use it to know if your pod is alive. Distinguish "
     "healthy/degraded/unhealthy so you can alert on partial failures "
     "before they become full outages. In ASP.NET Core this is "
     "app.MapHealthChecks('/health') with AddHealthChecks().")
]

for title, explanation in takeaways:
    print(f"\n  [{title}]")                   # Print takeaway title

    # Word-wrap the explanation to 72 characters
    words = explanation.split()               # Split into individual words
    line  = "  "                              # Start with indentation
    for word in words:
        if len(line) + len(word) > 72:        # Would adding this word exceed 72 chars?
            print(line)                        # Print current line
            line = "  " + word + " "          # Start a new line
        else:
            line += word + " "                 # Append word to current line
    if line.strip():
        print(line)                            # Print any remaining text

print("\n" + "=" * 70)
print("END OF EXAMPLE 02 - DEPLOYMENT AND SCALABILITY CONCEPTS")
print("=" * 70)

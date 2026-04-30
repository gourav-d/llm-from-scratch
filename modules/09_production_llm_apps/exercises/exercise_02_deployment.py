# =============================================================================
# Exercise 02: Deployment and Scalability
# Module 09 - Production LLM Applications
# =============================================================================
#
# GLOSSARY
# --------
# cache          : A fast, temporary storage layer that saves the result of an
#                  expensive operation so the next request gets it instantly.
#                  Like MemoryCache or IDistributedCache in ASP.NET Core.
#
# TTL            : Time-To-Live. How long a cached item stays valid before it
#                  expires and must be fetched fresh.  e.g., TTL=60 means the
#                  cached value is discarded after 60 seconds.
#
# connection pool: A set of pre-opened connections (to a database, API, etc.)
#                  that are reused instead of opening a new one for every
#                  request.  Like SqlConnection pooling in ADO.NET.
#
# health check   : A lightweight endpoint (or function) that tells an
#                  orchestrator (e.g., Kubernetes, Azure App Service) whether
#                  this service is running correctly.  Returns "healthy",
#                  "degraded", or "unhealthy".
#
# load balancer  : A component that distributes incoming requests across
#                  multiple server instances so no single server is overloaded.
#                  Azure Load Balancer or AWS ALB are real-world examples.
#
# =============================================================================

# --- standard library imports (no pip installs needed) -----------------------
import time   # for time.time() -- returns current time as a float (seconds)

# =============================================================================
# EXERCISE 1: SimpleCache
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   An in-memory key-value cache that automatically expires entries after a
#   configurable number of seconds (TTL = Time-To-Live).
#
# DATA STRUCTURE:
#   Store each entry as a dict:  {"value": ..., "expires_at": float}
#   The outer dict maps:  key (str) -> entry (dict)
#
# RULES:
#   set(key, value, ttl_seconds=60) -- store the value; it expires in ttl_seconds
#   get(key)                        -- return the value, or None if missing/expired
#
# C# ANALOGY:
#   Like  IMemoryCache.Set(key, value, TimeSpan.FromSeconds(60))
#   and   IMemoryCache.TryGetValue(key, out var value)
#
# EXPECTED RESULTS:
#   cache = SimpleCache()
#   cache.set("x", 42, ttl_seconds=1)
#   cache.get("x")   -> 42           (still within TTL)
#   time.sleep(1.1)
#   cache.get("x")   -> None         (expired)
#
# =============================================================================

class SimpleCache:
    """
    In-memory key-value store with per-entry TTL expiry.
    """

    def __init__(self):
        """
        Constructor -- initialise an empty store.
        """
        # TODO: Create an empty dictionary to hold cached entries.
        #       Name it  self._store
        #       Each value will be a dict: {"value": ..., "expires_at": float}
        #       (The leading underscore  _  is Python convention for "private"
        #        -- like a private field in C#.)

        pass   # replace with your implementation

    def set(self, key, value, ttl_seconds=60):
        """
        Store a value under key with the given TTL.

        Parameters
        ----------
        key         : str -- lookup key
        value       : any -- the data to cache
        ttl_seconds : int -- how many seconds until the entry expires
        """
        # TODO: Calculate the expiry time and store the entry.
        #
        # Step 1: expires_at = time.time() + ttl_seconds
        #         (current time + TTL = the moment the entry should die)
        #
        # Step 2: self._store[key] = {"value": value, "expires_at": expires_at}

        pass   # replace with your implementation

    def get(self, key):
        """
        Retrieve a value by key, or return None if missing or expired.

        Returns
        -------
        any or None
        """
        # TODO: Look up the key and check whether it has expired.
        #
        # Step 1: If key is not in self._store, return None.
        #
        # Step 2: Get the entry:  entry = self._store[key]
        #
        # Step 3: If  time.time() > entry["expires_at"]:
        #             delete the entry:  del self._store[key]
        #             return None
        #
        # Step 4: Return  entry["value"]

        pass   # replace with your implementation


# =============================================================================
# EXERCISE 2: RoundRobinBalancer
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   A simple round-robin load balancer.  You register server URLs, then each
#   call to  get_next()  returns the next server in rotation.
#   After the last server is used, it wraps back to the first.
#
# HOW ROUND-ROBIN WORKS:
#   Servers: [A, B, C]
#   Calls:    A -> B -> C -> A -> B -> C -> ...
#
# C# ANALOGY:
#   Like iterating through a  List<string>  using index % list.Count
#   to wrap around.
#
# EXPECTED RESULTS:
#   b = RoundRobinBalancer()
#   b.add_server("http://server-a")
#   b.add_server("http://server-b")
#   b.add_server("http://server-c")
#   b.get_next() -> "http://server-a"
#   b.get_next() -> "http://server-b"
#   b.get_next() -> "http://server-c"
#   b.get_next() -> "http://server-a"   (wraps around)
#
# =============================================================================

class RoundRobinBalancer:
    """
    Distributes requests across servers in round-robin order.
    """

    def __init__(self):
        """
        Constructor -- initialise with an empty server list and index = 0.
        """
        # TODO: Create two instance attributes:
        #   self._servers = []    -- empty list that will hold server URLs
        #   self._index   = 0     -- tracks which server to use next

        pass   # replace with your implementation

    def add_server(self, url):
        """
        Register a server URL with the balancer.

        Parameters
        ----------
        url : str -- the server's base URL, e.g. "http://server-a:8080"
        """
        # TODO: Append url to self._servers.
        #       In Python:  self._servers.append(url)
        #       (Like  list.Add(url)  in C#)

        pass   # replace with your implementation

    def get_next(self):
        """
        Return the next server URL in round-robin order.

        Returns
        -------
        str -- the selected server URL, or None if no servers registered
        """
        # TODO: Implement round-robin selection.
        #
        # Step 1: If self._servers is empty, return None.
        #         (Guard clause -- prevents division by zero below.)
        #
        # Step 2: server = self._servers[self._index]
        #         Selects the server at the current index.
        #
        # Step 3: Advance the index, wrapping around using modulo:
        #             self._index = (self._index + 1) % len(self._servers)
        #         The % operator gives the remainder.
        #         Example: (2 + 1) % 3 = 0  -- wraps from last back to first.
        #         (Same as  (index + 1) % servers.Count  in C#)
        #
        # Step 4: Return server.

        pass   # replace with your implementation


# =============================================================================
# EXERCISE 3: health_check
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   A function that examines a dict of component statuses and returns a
#   single overall health verdict.
#
# RULES:
#   All True  -> "healthy"
#   All False -> "unhealthy"
#   Mixed     -> "degraded"
#
# C# ANALOGY:
#   Like a HealthCheckResult in ASP.NET Core that aggregates multiple checks.
#
# EXPECTED RESULTS:
#   health_check({"db": True, "cache": True})             -> "healthy"
#   health_check({"db": True, "cache": False})            -> "degraded"
#   health_check({"db": False, "cache": False})           -> "unhealthy"
#
# =============================================================================

def health_check(checks):
    """
    Aggregate multiple boolean health checks into a single status string.

    Parameters
    ----------
    checks : dict -- {component_name: bool}  e.g., {"db": True, "cache": False}

    Returns
    -------
    str -- "healthy", "degraded", or "unhealthy"
    """
    # TODO: Implement this function.
    #
    # Hint 1: Get all the boolean values from the dict.
    #         In Python:  values = list(checks.values())
    #         checks.values() gives a view of all the dict values.
    #         (Like  dictionary.Values  in C#)
    #
    # Hint 2: Check if ALL are True:
    #         all(values)   -- built-in function, returns True if every item is truthy
    #         (C# equivalent:  values.All(v => v)  using LINQ)
    #
    # Hint 3: Check if ALL are False:
    #         not any(values)  -- any() returns True if at least one item is truthy
    #         so  not any(values)  means "none are True" i.e., all are False
    #         (C# equivalent:  !values.Any(v => v))
    #
    # Hint 4: If neither all-True nor all-False, it must be mixed -> "degraded"

    pass   # replace with your implementation


# =============================================================================
# EXERCISE 4: generate_dockerfile
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   A function that generates a Dockerfile string for deploying a Python
#   application.  Dockerfiles are plain text; this function just assembles
#   one programmatically using an f-string.
#
# A DOCKERFILE is a recipe for building a Docker container image.
# Each line is an instruction:
#   FROM    -- base image to start from
#   WORKDIR -- set the working directory inside the container
#   COPY    -- copy files from the host into the image
#   RUN     -- execute a shell command during the build
#   EXPOSE  -- declare which network port the app will use
#   CMD     -- the command to run when the container starts
#
# C# ANALOGY:
#   Like a .csproj or Dockerfile for a .NET app, but for Python.
#   Similar to running  dotnet publish  and putting the output in a container.
#
# EXPECTED RESULT:
#   generate_dockerfile("myapp", 8080) returns a string starting with:
#   "FROM python:3.11-slim\nWORKDIR /app\n..."
#
# =============================================================================

def generate_dockerfile(app_name, port, python_version="3.11"):
    """
    Generate a Dockerfile string for a Python application.

    Parameters
    ----------
    app_name       : str -- name of the application (used in a comment)
    port           : int -- port number to expose
    python_version : str -- Python version tag (default "3.11")

    Returns
    -------
    str -- a complete Dockerfile as a multi-line string
    """
    # TODO: Build and return a Dockerfile string using an f-string.
    #
    # An f-string starts with  f"  and lets you embed variables with {}.
    # Example:  f"Hello {name}"  inserts the value of  name.
    # (C# equivalent:  $"Hello {name}"  -- same idea!)
    #
    # Use triple-quoted f-string  f"""..."""  for multi-line content.
    #
    # The Dockerfile should have these lines IN THIS ORDER:
    #   # Application: {app_name}
    #   FROM python:{python_version}-slim
    #   WORKDIR /app
    #   COPY requirements.txt .
    #   RUN pip install -r requirements.txt
    #   COPY . .
    #   EXPOSE {port}
    #   CMD ["python", "main.py"]
    #
    # Hint: Use \n or a real newline inside the triple-quoted string.
    #       .strip() on the result removes any leading/trailing blank lines.

    pass   # replace with your implementation


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests():
    """Run all exercise tests and print results."""

    print("=" * 60)
    print("Exercise 02: Deployment and Scalability -- Test Results")
    print("=" * 60)

    # ---- Exercise 1 tests ---------------------------------------------------
    print("\n--- Exercise 1: SimpleCache ---")

    cache = SimpleCache()
    cache.set("color", "blue", ttl_seconds=60)

    val = cache.get("color")
    status = "PASS" if val == "blue" else "FAIL"
    print(status + "  get('color') == 'blue'  (got: " + str(val) + ")")

    val_missing = cache.get("missing_key")
    status = "PASS" if val_missing is None else "FAIL"
    print(status + "  get('missing_key') == None  (got: " + str(val_missing) + ")")

    # Test TTL expiry with a very short TTL
    cache.set("temp", 99, ttl_seconds=1)
    val_before = cache.get("temp")
    status = "PASS" if val_before == 99 else "FAIL"
    print(status + "  get('temp') before expiry == 99  (got: " + str(val_before) + ")")

    time.sleep(1.1)   # wait just over 1 second for the entry to expire

    val_after = cache.get("temp")
    status = "PASS" if val_after is None else "FAIL"
    print(status + "  get('temp') after expiry == None  (got: " + str(val_after) + ")")

    # ---- Exercise 2 tests ---------------------------------------------------
    print("\n--- Exercise 2: RoundRobinBalancer ---")

    balancer = RoundRobinBalancer()
    balancer.add_server("http://server-a")
    balancer.add_server("http://server-b")
    balancer.add_server("http://server-c")

    expected_sequence = [
        "http://server-a",
        "http://server-b",
        "http://server-c",
        "http://server-a",   # wraps around
        "http://server-b",
    ]

    for i, expected in enumerate(expected_sequence):
        got = balancer.get_next()
        status = "PASS" if got == expected else "FAIL"
        label = "call " + str(i + 1) + " -> " + expected
        print(status + "  " + label + "  (got: " + str(got) + ")")

    # Edge case: no servers registered
    empty_balancer = RoundRobinBalancer()
    got = empty_balancer.get_next()
    status = "PASS" if got is None else "FAIL"
    print(status + "  empty balancer get_next() == None  (got: " + str(got) + ")")

    # ---- Exercise 3 tests ---------------------------------------------------
    print("\n--- Exercise 3: health_check ---")

    result = health_check({"db": True, "cache": True})
    status = "PASS" if result == "healthy" else "FAIL"
    print(status + "  all True -> 'healthy'  (got: " + str(result) + ")")

    result = health_check({"db": True, "cache": False})
    status = "PASS" if result == "degraded" else "FAIL"
    print(status + "  mixed -> 'degraded'  (got: " + str(result) + ")")

    result = health_check({"db": False, "cache": False})
    status = "PASS" if result == "unhealthy" else "FAIL"
    print(status + "  all False -> 'unhealthy'  (got: " + str(result) + ")")

    result = health_check({"db": True, "cache": False, "queue": True})
    status = "PASS" if result == "degraded" else "FAIL"
    print(status + "  2 True, 1 False -> 'degraded'  (got: " + str(result) + ")")

    # ---- Exercise 4 tests ---------------------------------------------------
    print("\n--- Exercise 4: generate_dockerfile ---")

    dockerfile = generate_dockerfile("myapp", 8080)

    if dockerfile is not None:
        # Check that required lines appear in the output
        checks = [
            ("FROM python:3.11-slim" in dockerfile,   "contains FROM python:3.11-slim"),
            ("WORKDIR /app"          in dockerfile,   "contains WORKDIR /app"),
            ("EXPOSE 8080"           in dockerfile,   "contains EXPOSE 8080"),
            ('CMD ["python", "main.py"]' in dockerfile, 'contains CMD ["python", "main.py"]'),
            ("myapp"                 in dockerfile,   "contains app name"),
        ]
        for passed, label in checks:
            status = "PASS" if passed else "FAIL"
            print(status + "  " + label)

        # Check custom python version
        dockerfile2 = generate_dockerfile("otherapp", 5000, python_version="3.10")
        status = "PASS" if "python:3.10-slim" in dockerfile2 else "FAIL"
        print(status + "  custom python version 3.10 -> contains python:3.10-slim")
    else:
        print("FAIL  generate_dockerfile returned None (not implemented yet)")

    print("\n" + "=" * 60)
    print("Done.  Implement each TODO to turn FAILs into PASSes.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()


# =============================================================================
# SOLUTION  (triple-quoted string -- Python reads but does NOT execute this)
# Study this AFTER you have attempted the exercises yourself.
# =============================================================================
"""
SOLUTION: Exercise 02 - Deployment and Scalability
====================================================

----------------------------------------------------------------------
EXERCISE 1: SimpleCache
----------------------------------------------------------------------

class SimpleCache:

    def __init__(self):
        self._store = {}   # empty dict; will hold {key: {"value":..., "expires_at":...}}

    def set(self, key, value, ttl_seconds=60):
        expires_at = time.time() + ttl_seconds   # current epoch + TTL = expiry moment
        self._store[key] = {
            "value":      value,       # the actual data to cache
            "expires_at": expires_at,  # the float timestamp when this entry dies
        }

    def get(self, key):
        if key not in self._store:     # key never set, or already deleted
            return None

        entry = self._store[key]       # retrieve the stored dict

        if time.time() > entry["expires_at"]:   # entry has expired
            del self._store[key]                # clean up to free memory
            return None

        return entry["value"]          # entry is still fresh, return the data


----------------------------------------------------------------------
EXERCISE 2: RoundRobinBalancer
----------------------------------------------------------------------

class RoundRobinBalancer:

    def __init__(self):
        self._servers = []   # list of server URL strings
        self._index   = 0    # index of the next server to use

    def add_server(self, url):
        self._servers.append(url)   # add URL to the end of the list

    def get_next(self):
        if not self._servers:       # empty list is falsy in Python
            return None

        server = self._servers[self._index]              # pick current server
        self._index = (self._index + 1) % len(self._servers)  # advance & wrap
        return server


----------------------------------------------------------------------
EXERCISE 3: health_check
----------------------------------------------------------------------

def health_check(checks):
    values = list(checks.values())   # extract the bool values from the dict

    if all(values):           # every component is healthy
        return "healthy"

    if not any(values):       # no component is healthy
        return "unhealthy"

    return "degraded"         # some healthy, some not


----------------------------------------------------------------------
EXERCISE 4: generate_dockerfile
----------------------------------------------------------------------

def generate_dockerfile(app_name, port, python_version="3.11"):
    # f-string with triple quotes allows a real multi-line string.
    # {app_name}, {python_version}, and {port} are substituted at runtime.
    # .strip() removes any leading/trailing blank lines from the result.
    return f\"\"\"
# Application: {app_name}
FROM python:{python_version}-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE {port}
CMD ["python", "main.py"]
\"\"\".strip()
"""

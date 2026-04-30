# =============================================================================
# Exercise 01: API Design Patterns
# Module 09 - Production LLM Applications
# =============================================================================
#
# GLOSSARY
# --------
# endpoint      : A URL path that your server responds to. Like a C# controller
#                 action method -- e.g., POST /api/chat handles chat requests.
#
# middleware    : Code that runs BETWEEN receiving a request and handling it.
#                 Like ASP.NET middleware (authentication, logging, validation)
#                 that forms a "pipeline" before the real handler runs.
#
# JWT           : JSON Web Token. A compact string used to prove who a user is.
#                 Like a signed ticket: the server issues it, clients send it
#                 back with every request (similar to a bearer token in C#).
#
# rate limiting : Restricting how many requests a user can make in a time window.
#                 Like a "you may only call this API 100 times per minute" rule.
#                 Protects the server from abuse or overload.
#
# validation    : Checking that incoming data is correct before processing it.
#                 Like using DataAnnotations or FluentValidation in C# to check
#                 that a field is not empty or a number is in range.
#
# =============================================================================

# --- standard library imports (no pip installs needed) -----------------------
import time       # for getting the current time (used in rate limiter)

# =============================================================================
# EXERCISE 1: validate_request
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   A function that checks whether the inputs to an LLM API call are valid.
#   In production, every API receives untrusted data -- validation is the
#   first line of defence before any expensive LLM call is made.
#
# RULES:
#   - message    : must not be an empty string
#   - max_tokens : must be an integer between 1 and 4096 (inclusive)
#   - temperature: must be a float between 0.0 and 2.0 (inclusive)
#   - Return True  when ALL three rules pass
#   - Return False when ANY rule fails
#
# C# ANALOGY:
#   In C# you might write a method like:
#       bool ValidateRequest(string message, int maxTokens, double temperature)
#   Python uses the same logic but with dynamic typing -- no type declarations.
#
# EXPECTED RESULTS:
#   validate_request("Hello", 100, 0.7)  -> True
#   validate_request("",      100, 0.7)  -> False   (empty message)
#   validate_request("Hello",   0, 0.7)  -> False   (max_tokens below 1)
#   validate_request("Hello", 100, 3.0)  -> False   (temperature above 2.0)
#
# =============================================================================

def validate_request(message, max_tokens, temperature):
    """
    Validate the three parameters of a chat API request.

    Parameters
    ----------
    message     : str   -- the user's input text
    max_tokens  : int   -- how many tokens the model may generate
    temperature : float -- randomness control (0.0 = deterministic, 2.0 = very random)

    Returns
    -------
    bool -- True if all inputs are valid, False otherwise
    """
    # TODO: Implement this function.
    #
    # Hint 1: Check whether message is truthy (non-empty).
    #         In Python,  if message:  is True for non-empty strings.
    #         (Same idea as  string.IsNullOrEmpty()  in C#, but inverted.)
    #
    # Hint 2: Use  1 <= max_tokens <= 4096  -- Python allows chained comparisons!
    #         In C# you would write  maxTokens >= 1 && maxTokens <= 4096
    #
    # Hint 3: Use  0.0 <= temperature <= 2.0  for the same reason.
    #
    # Hint 4: Return True only if ALL three checks pass.
    #         Use the  and  keyword (C# equivalent: &&).

    pass   # replace this line with your implementation


# =============================================================================
# EXERCISE 2: SimpleRateLimiter
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   A class that tracks how many times each user has called the API within a
#   sliding time window.  If they exceed max_requests, deny the next call.
#
# HOW IT WORKS (sliding window algorithm):
#   1. Every time a user calls is_allowed(), record the current timestamp.
#   2. Before checking the count, throw away any timestamps older than
#      window_seconds ago -- those requests are "outside the window".
#   3. Count how many timestamps remain.  If count < max_requests, allow it
#      and record the new timestamp.  Otherwise deny.
#
# C# ANALOGY:
#   Like a Dictionary<string, List<DateTime>> where the key is the user ID
#   and the value is the list of recent request times.
#
# EXPECTED RESULTS (max_requests=2, window_seconds=5):
#   is_allowed("alice") -> True   (1st request)
#   is_allowed("alice") -> True   (2nd request)
#   is_allowed("alice") -> False  (3rd request -- limit reached)
#
# =============================================================================

class SimpleRateLimiter:
    """
    Tracks API request counts per user using a sliding time window.

    In C# terms, this is a class with a private dictionary field and
    two public methods.
    """

    def __init__(self, max_requests, window_seconds):
        """
        Constructor -- called automatically when you do SimpleRateLimiter(2, 5).

        Parameters
        ----------
        max_requests   : int -- maximum number of requests allowed per window
        window_seconds : int -- length of the time window in seconds
        """
        # TODO: Store max_requests and window_seconds as instance attributes.
        #       In Python, instance attributes use  self.name = value
        #       (like  this.Name = value  in C#).
        #
        #       Also create an empty dictionary to store timestamps:
        #           self.requests = {}
        #       The dictionary maps user_id (str) -> list of float timestamps.

        pass   # replace with your implementation

    def is_allowed(self, user_id):
        """
        Check whether user_id is allowed to make another request right now.

        Parameters
        ----------
        user_id : str -- identifies the caller (e.g., "user_42")

        Returns
        -------
        bool -- True if allowed, False if rate limit exceeded
        """
        # TODO: Implement the sliding window logic.
        #
        # Step 1: Get the current time.
        #         current_time = time.time()   -- returns seconds since epoch (a float)
        #
        # Step 2: If user_id is not yet in self.requests, add an empty list.
        #         Hint:  if user_id not in self.requests:
        #                    self.requests[user_id] = []
        #
        # Step 3: Remove timestamps that are older than window_seconds.
        #         Hint: keep only timestamps where  t > current_time - self.window_seconds
        #         List comprehension syntax:
        #             self.requests[user_id] = [t for t in self.requests[user_id]
        #                                       if t > current_time - self.window_seconds]
        #         (C# LINQ equivalent:  requests[userId].Where(t => t > cutoff).ToList())
        #
        # Step 4: Count remaining timestamps.
        #         If len(self.requests[user_id]) < self.max_requests:
        #             append current_time and return True
        #         Else:
        #             return False

        pass   # replace with your implementation


# =============================================================================
# EXERCISE 3: build_middleware_chain
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   A function that takes any number of middleware functions and returns a
#   single  process(request)  function that runs them all in order.
#   If any middleware "rejects" the request, processing stops immediately.
#
# MIDDLEWARE CONTRACT:
#   Each middleware_fn receives a dict (the request) and returns a tuple:
#       (True,  "")           -- request passed this middleware
#       (False, "error msg")  -- request rejected; include the reason
#
# HOW TO USE *args (variable arguments):
#   def build_middleware_chain(*middleware_fns):
#       -- middleware_fns is a TUPLE of functions, like params[] in C#
#
# C# ANALOGY:
#   Like building an ASP.NET middleware pipeline with app.Use(...) calls.
#   Each piece of middleware decides whether to call next() or short-circuit.
#
# EXPECTED RESULTS:
#   chain = build_middleware_chain(auth_check, size_check)
#   chain({"token": "valid", "size": 10}) -> (True, "OK")
#   chain({"token": "", "size": 10})      -> (False, "no token")
#
# =============================================================================

def build_middleware_chain(*middleware_fns):
    """
    Combine multiple middleware functions into one pipeline.

    Parameters
    ----------
    *middleware_fns : callable -- zero or more middleware functions.
                      Each must accept a dict and return (bool, str).

    Returns
    -------
    callable -- a  process(request)  function
    """
    # TODO: Define and return an inner function called  process(request).
    #
    # Inside  process(request):
    #   - Loop over each fn in middleware_fns
    #   - Call  passed, error = fn(request)
    #     (Python tuple unpacking -- like  var (passed, error) = fn(request)  in C# 7+)
    #   - If  not passed:  return (False, error)
    #   - After the loop, return (True, "OK")
    #
    # Then  return process  (return the inner function itself, not its result).
    # This is a Python "closure" -- the inner function remembers middleware_fns.

    pass   # replace with your implementation


# =============================================================================
# EXERCISE 4: mock_streaming_response
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   A Python generator that simulates an LLM streaming its response word by word.
#   Real LLM APIs (OpenAI, Anthropic) stream tokens so the user sees output
#   appear progressively instead of waiting for the whole response.
#
# WHAT IS A GENERATOR?
#   A function that uses  yield  instead of  return.
#   Each time the caller asks for the next value, the function resumes from
#   where it left off.
#   C# equivalent: IEnumerable<string> with  yield return word;
#
# EXPECTED RESULTS:
#   list(mock_streaming_response("Hello world from LLM"))
#   -> ["Hello", "world", "from", "LLM"]
#
# =============================================================================

def mock_streaming_response(text):
    """
    Yield one word at a time from text, simulating a streaming LLM response.

    Parameters
    ----------
    text : str -- the full response text to stream

    Yields
    ------
    str -- one word at a time, stripped of leading/trailing whitespace
    """
    # TODO: Implement this generator.
    #
    # Step 1: Split text into words:  words = text.split()
    #         text.split()  splits on any whitespace and removes empty strings.
    #         (Like  text.Split()  in C# but smarter about multiple spaces.)
    #
    # Step 2: Loop over each word:  for word in words:
    #
    # Step 3: Strip extra whitespace from each word:  word = word.strip()
    #         (Like  word.Trim()  in C#)
    #
    # Step 4: Yield the word:  yield word
    #         (Like  yield return word;  in C#)

    pass   # replace with your implementation


# =============================================================================
# TEST RUNNER
# =============================================================================
# This section runs automatically when you execute the file.
# It checks each exercise and prints PASS or FAIL.
# You do NOT need to modify this section.
# =============================================================================

def run_tests():
    """Run all exercise tests and print results."""

    print("=" * 60)
    print("Exercise 01: API Design Patterns -- Test Results")
    print("=" * 60)

    # ---- Exercise 1 tests ---------------------------------------------------
    print("\n--- Exercise 1: validate_request ---")

    # Test 1a: all valid inputs
    result = validate_request("Hello", 100, 0.7)
    status = "PASS" if result is True else "FAIL"
    print(status + "  validate_request('Hello', 100, 0.7) == True  (got: " + str(result) + ")")

    # Test 1b: empty message should fail
    result = validate_request("", 100, 0.7)
    status = "PASS" if result is False else "FAIL"
    print(status + "  validate_request('', 100, 0.7) == False  (got: " + str(result) + ")")

    # Test 1c: max_tokens = 0 should fail
    result = validate_request("Hello", 0, 0.7)
    status = "PASS" if result is False else "FAIL"
    print(status + "  validate_request('Hello', 0, 0.7) == False  (got: " + str(result) + ")")

    # Test 1d: temperature = 3.0 should fail
    result = validate_request("Hello", 100, 3.0)
    status = "PASS" if result is False else "FAIL"
    print(status + "  validate_request('Hello', 100, 3.0) == False  (got: " + str(result) + ")")

    # Test 1e: boundary -- max_tokens = 4096 should pass
    result = validate_request("Hi", 4096, 0.0)
    status = "PASS" if result is True else "FAIL"
    print(status + "  validate_request('Hi', 4096, 0.0) == True  (got: " + str(result) + ")")

    # ---- Exercise 2 tests ---------------------------------------------------
    print("\n--- Exercise 2: SimpleRateLimiter ---")

    limiter = SimpleRateLimiter(max_requests=2, window_seconds=5)

    r1 = limiter.is_allowed("alice")
    status = "PASS" if r1 is True else "FAIL"
    print(status + "  1st request for alice -> True  (got: " + str(r1) + ")")

    r2 = limiter.is_allowed("alice")
    status = "PASS" if r2 is True else "FAIL"
    print(status + "  2nd request for alice -> True  (got: " + str(r2) + ")")

    r3 = limiter.is_allowed("alice")
    status = "PASS" if r3 is False else "FAIL"
    print(status + "  3rd request for alice -> False (got: " + str(r3) + ")")

    # Different user should be unaffected
    r4 = limiter.is_allowed("bob")
    status = "PASS" if r4 is True else "FAIL"
    print(status + "  1st request for bob   -> True  (got: " + str(r4) + ")")

    # ---- Exercise 3 tests ---------------------------------------------------
    print("\n--- Exercise 3: build_middleware_chain ---")

    # Define two simple middleware functions for testing
    def require_token(request):
        # returns (False, error) if no token present
        if not request.get("token"):        # .get() returns None if key missing
            return (False, "missing token")
        return (True, "")                   # all good

    def require_small_input(request):
        # returns (False, error) if message too long
        if len(request.get("message", "")) > 10:
            return (False, "message too long")
        return (True, "")

    chain = build_middleware_chain(require_token, require_small_input)

    if chain is not None:   # only run sub-tests if function is implemented
        # Both middleware pass
        result = chain({"token": "abc", "message": "Hello"})
        status = "PASS" if result == (True, "OK") else "FAIL"
        print(status + "  valid request -> (True, 'OK')  (got: " + str(result) + ")")

        # First middleware fails
        result = chain({"token": "", "message": "Hello"})
        status = "PASS" if result == (False, "missing token") else "FAIL"
        print(status + "  no token -> (False, 'missing token')  (got: " + str(result) + ")")

        # Second middleware fails
        result = chain({"token": "abc", "message": "This is way too long for the limit"})
        status = "PASS" if result == (False, "message too long") else "FAIL"
        print(status + "  long message -> (False, 'message too long')  (got: " + str(result) + ")")
    else:
        print("FAIL  build_middleware_chain returned None (not implemented yet)")

    # ---- Exercise 4 tests ---------------------------------------------------
    print("\n--- Exercise 4: mock_streaming_response ---")

    # Guard: if the function returns None (not yet implemented) calling
    # list() on None would crash.  Wrap in a try/except so the file still
    # runs cleanly while the stub is in place.
    try:
        words = list(mock_streaming_response("Hello world from LLM"))
        expected = ["Hello", "world", "from", "LLM"]
        status = "PASS" if words == expected else "FAIL"
        print(status + "  streamed words == ['Hello', 'world', 'from', 'LLM']  (got: " + str(words) + ")")

        # Test with extra spaces (split() handles them automatically)
        words2 = list(mock_streaming_response("  one   two  three  "))
        expected2 = ["one", "two", "three"]
        status = "PASS" if words2 == expected2 else "FAIL"
        print(status + "  extra spaces handled correctly  (got: " + str(words2) + ")")
    except TypeError:
        # mock_streaming_response returned None -- function not yet implemented
        print("FAIL  mock_streaming_response returned None (not implemented yet)")
        print("FAIL  extra spaces test skipped")

    print("\n" + "=" * 60)
    print("Done.  Implement each TODO to turn FAILs into PASSes.")
    print("=" * 60)


# Entry point -- runs when you execute:  python exercise_01_api_design.py
if __name__ == "__main__":
    run_tests()


# =============================================================================
# SOLUTION  (triple-quoted string -- Python reads but does NOT execute this)
# Study this AFTER you have attempted the exercises yourself.
# =============================================================================
"""
SOLUTION: Exercise 01 - API Design Patterns
============================================

----------------------------------------------------------------------
EXERCISE 1: validate_request
----------------------------------------------------------------------

def validate_request(message, max_tokens, temperature):
    # Check 1: message must be non-empty.
    # In Python, an empty string "" is "falsy" -- bool("") is False.
    # So  if not message  is True when message is empty.
    if not message:                          # empty string fails
        return False

    # Check 2: max_tokens must be in range [1, 4096].
    # Python allows chained comparisons: a <= b <= c
    # This is equivalent to C#: maxTokens >= 1 && maxTokens <= 4096
    if not (1 <= max_tokens <= 4096):
        return False

    # Check 3: temperature must be in range [0.0, 2.0].
    if not (0.0 <= temperature <= 2.0):
        return False

    # All checks passed -- the request is valid.
    return True


----------------------------------------------------------------------
EXERCISE 2: SimpleRateLimiter
----------------------------------------------------------------------

class SimpleRateLimiter:

    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests       # store limit
        self.window_seconds = window_seconds   # store window length
        self.requests = {}                     # empty dict: user_id -> [timestamps]

    def is_allowed(self, user_id):
        current_time = time.time()             # float: seconds since Unix epoch

        # If this user has no history yet, create an empty list for them.
        if user_id not in self.requests:
            self.requests[user_id] = []

        # Remove timestamps older than the window.
        # Keep only timestamps that are NEWER than (now - window).
        # List comprehension: builds a new list from the existing one.
        cutoff = current_time - self.window_seconds
        self.requests[user_id] = [
            t for t in self.requests[user_id]   # t is each stored timestamp
            if t > cutoff                        # keep only recent ones
        ]

        # Check whether the user is still under the limit.
        if len(self.requests[user_id]) < self.max_requests:
            self.requests[user_id].append(current_time)  # record this request
            return True    # allowed

        return False       # limit exceeded


----------------------------------------------------------------------
EXERCISE 3: build_middleware_chain
----------------------------------------------------------------------

def build_middleware_chain(*middleware_fns):
    # Define the inner function that will be returned.
    # It "closes over" middleware_fns -- it can still see that variable
    # even after build_middleware_chain() has finished.
    def process(request):
        # Run each middleware in the order it was provided.
        for fn in middleware_fns:                # iterate over the tuple of fns
            passed, error = fn(request)          # unpack the (bool, str) tuple
            if not passed:                       # this middleware rejected the request
                return (False, error)            # short-circuit -- stop here
        # Every middleware passed.
        return (True, "OK")

    return process   # return the function OBJECT (not calling it)


----------------------------------------------------------------------
EXERCISE 4: mock_streaming_response
----------------------------------------------------------------------

def mock_streaming_response(text):
    words = text.split()          # split on whitespace, removes empty strings
    for word in words:            # iterate over each word
        word = word.strip()       # remove any remaining leading/trailing whitespace
        yield word                # pause here and hand the word to the caller
                                  # next call resumes from here
"""

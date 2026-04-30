# =============================================================================
# Exercise 03: Monitoring and Observability
# Module 09 - Production LLM Applications
# =============================================================================
#
# GLOSSARY
# --------
# structured logging : Writing log entries as machine-readable dicts (or JSON)
#                      instead of plain text.  Tools like Azure Monitor, Datadog,
#                      and Splunk can then search/filter log fields.
#                      Like using ILogger<T> with structured message templates
#                      in ASP.NET Core.
#
# metric             : A single numeric measurement recorded at a point in time.
#                      Examples: response_time_ms = 250, error_count = 3.
#                      Think of it as a counter or gauge (like a performance
#                      counter in Windows / .NET diagnostics).
#
# percentile         : A value below which a given percentage of measurements
#                      fall.  If the p95 latency is 400 ms, 95% of requests
#                      finished in <= 400 ms.  Useful because averages hide
#                      outliers.
#
# p99                : The 99th percentile.  The latency experienced by the
#                      slowest 1% of requests.  "Our p99 is 2 seconds" means
#                      99 out of 100 requests are faster than 2 seconds.
#
# alert              : An automated notification triggered when a metric
#                      crosses a threshold.  e.g., "page the on-call engineer
#                      if error_rate > 5%".  Like setting up Azure Monitor
#                      alert rules.
#
# =============================================================================

# --- standard library imports (no pip installs needed) -----------------------
import datetime   # for datetime.datetime.now() -- gets the current date and time

# =============================================================================
# EXERCISE 1: log_event
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   A function that creates a structured log entry as a Python dict.
#   In production, this dict would be serialised to JSON and sent to a
#   logging platform (e.g., Azure Log Analytics, Datadog).
#
# RULES:
#   Always include:  timestamp, level, message
#   Also include any extra keyword arguments passed by the caller.
#
# PYTHON CONCEPT -- **kwargs:
#   def log_event(level, message, **fields)
#   The  **fields  parameter collects all extra keyword arguments into a dict.
#   Example call:  log_event("ERROR", "DB failed", user_id="u1", latency_ms=300)
#   Inside the function:  fields == {"user_id": "u1", "latency_ms": 300}
#   C# analogy: like  params KeyValuePair<string,object>[] fields  in a method.
#
# EXPECTED RESULTS:
#   entry = log_event("ERROR", "DB failed", user_id="u1")
#   entry["level"]   -> "ERROR"
#   entry["message"] -> "DB failed"
#   entry["user_id"] -> "u1"
#   entry["timestamp"] -> a string like "2026-04-30T14:23:01.456789"
#
# =============================================================================

def log_event(level, message, **fields):
    """
    Create a structured log entry dict.

    Parameters
    ----------
    level   : str -- severity level, e.g. "INFO", "WARN", "ERROR"
    message : str -- human-readable description of the event
    **fields: any -- additional key-value pairs to include in the log entry

    Returns
    -------
    dict -- structured log entry with at least: timestamp, level, message
    """
    # TODO: Build and return a dict with these keys:
    #
    # 1. "timestamp": datetime.datetime.now().isoformat()
    #    -- isoformat() returns a string like "2026-04-30T14:23:01.456789"
    #    -- (Like  DateTime.Now.ToString("o")  in C#)
    #
    # 2. "level": level
    #    -- the severity string passed in
    #
    # 3. "message": message
    #    -- the human-readable text passed in
    #
    # 4. All extra fields from **fields
    #    -- spread them into the dict using  **fields
    #
    # Building a dict with known keys PLUS extra keys:
    #   entry = {
    #       "timestamp": ...,
    #       "level":     level,
    #       "message":   message,
    #       **fields,         # spreads all items from the fields dict into entry
    #   }
    #   (C# analogy: like calling  dict.Merge(fields)  or using spread in JSON)
    #
    # Then return entry.

    pass   # replace with your implementation


# =============================================================================
# EXERCISE 2: compute_percentile
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   A function that calculates any percentile of a list of numbers.
#   Used to compute p50 (median), p95, p99 latency statistics.
#
# ALGORITHM (simplified nearest-rank):
#   1. Sort the list in ascending order.
#   2. Compute  index = int(p * len(values))
#      (int() truncates -- drops the decimal part, like  (int)  cast in C#)
#   3. Return  values[index]
#
# EDGE CASE:
#   If the list is empty, return 0.0
#
# EXPECTED RESULTS:
#   compute_percentile([1,2,3,4,5,6,7,8,9,10], 0.9)  -> 10  (index = int(0.9*10) = 9)
#   compute_percentile([1,2,3,4,5,6,7,8,9,10], 0.5)  -> 6   (index = int(0.5*10) = 5)
#   compute_percentile([], 0.99)                       -> 0.0
#
# =============================================================================

def compute_percentile(values, p):
    """
    Return the p-th percentile of a list of numeric values.

    Parameters
    ----------
    values : list of float/int -- the measurements to analyse
    p      : float -- the percentile as a fraction, e.g. 0.99 for p99

    Returns
    -------
    float -- the percentile value, or 0.0 if values is empty
    """
    # TODO: Implement this function.
    #
    # Step 1: Handle the empty-list edge case.
    #         if not values:   return 0.0
    #         (An empty list is "falsy" in Python -- same as  if values.Count == 0  in C#)
    #
    # Step 2: Sort the list.
    #         sorted_values = sorted(values)
    #         sorted() returns a NEW sorted list; it does NOT modify the original.
    #         (Like  values.OrderBy(x => x).ToList()  in C# LINQ)
    #
    # Step 3: Calculate the index.
    #         index = int(p * len(sorted_values))
    #         int() truncates (floor), same as  (int)(p * count)  in C#.
    #         Clamp to max index to be safe:
    #             index = min(index, len(sorted_values) - 1)
    #         (Like  Math.Min(index, count - 1)  in C#)
    #
    # Step 4: Return  sorted_values[index]

    pass   # replace with your implementation


# =============================================================================
# EXERCISE 3: MetricsSummary
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   A class that accumulates numeric measurements and produces statistics:
#   count, min, max, average, and p95 percentile.
#
# DESIGN:
#   - Internally keep a dict: name -> list of recorded values
#   - record(name, value) appends to that list
#   - summary(name) computes and returns stats as a dict
#
# C# ANALOGY:
#   Like a Dictionary<string, List<double>> with a method that computes
#   Enumerable.Min(), .Max(), .Average() over each list.
#
# EXPECTED RESULTS:
#   m = MetricsSummary()
#   m.record("latency", 100)
#   m.record("latency", 200)
#   m.record("latency", 300)
#   m.summary("latency") -> {
#       "count": 3,
#       "min":   100,
#       "max":   300,
#       "avg":   200.0,
#       "p95":   300,
#   }
#
# =============================================================================

class MetricsSummary:
    """
    Collects numeric metric samples and computes summary statistics.
    """

    def __init__(self):
        """
        Constructor -- initialise an empty data store.
        """
        # TODO: Create  self._data = {}
        #       This dict maps metric names (str) to lists of float values.

        pass   # replace with your implementation

    def record(self, name, value):
        """
        Record one measurement for a named metric.

        Parameters
        ----------
        name  : str   -- metric name, e.g. "latency_ms" or "token_count"
        value : float -- the measured value
        """
        # TODO: Append value to self._data[name].
        #
        # If name is not yet in self._data, create an empty list first.
        # Pattern:
        #   if name not in self._data:
        #       self._data[name] = []
        #   self._data[name].append(value)
        #
        # Alternative (more Pythonic):
        #   self._data.setdefault(name, []).append(value)
        #   setdefault(key, default) returns the existing value if key exists,
        #   or inserts default and returns it if key is missing.
        #   (No direct C# equivalent -- closest is GetOrAdd in ConcurrentDictionary)

        pass   # replace with your implementation

    def summary(self, name):
        """
        Compute summary statistics for a named metric.

        Parameters
        ----------
        name : str -- the metric to summarise

        Returns
        -------
        dict with keys: count, min, max, avg, p95
        Returns all zeros if no data recorded for this name.
        """
        # TODO: Compute and return summary statistics.
        #
        # Step 1: Get the list:  data = self._data.get(name, [])
        #         .get(key, default) returns default if key is missing.
        #         (Like  dict.TryGetValue(key, out var value)  in C#)
        #
        # Step 2: If data is empty, return all-zeros dict:
        #         {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0, "p95": 0.0}
        #
        # Step 3: Compute stats using Python built-ins:
        #         count  = len(data)
        #         mn     = min(data)                     -- built-in min()
        #         mx     = max(data)                     -- built-in max()
        #         avg    = sum(data) / len(data)         -- sum() is built-in
        #         p95    = compute_percentile(data, 0.95)  -- reuse Exercise 2!
        #
        # Step 4: Return a dict with those five values.

        pass   # replace with your implementation


# =============================================================================
# EXERCISE 4: check_alerts
# =============================================================================
#
# WHAT YOU WILL BUILD:
#   A function that evaluates a set of alert rules against current metric
#   values and returns the list of rules that have been triggered.
#
# INPUT FORMATS:
#   metrics : dict  -- {"error_rate": 0.08, "p99_ms": 3000}
#   rules   : list of dicts, each with:
#               "metric"    : str   -- which metric to check
#               "threshold" : float -- value to compare against
#               "operator"  : str   -- ">", "<", ">=", "<="
#               "severity"  : str   -- "warning" or "critical"
#
# OUTPUT:
#   List of rule dicts whose condition evaluated to True.
#   If no rules trigger, return an empty list [].
#
# C# ANALOGY:
#   Like evaluating a list of Expression<Func<double, bool>> predicates.
#
# EXPECTED RESULTS:
#   rules = [
#       {"metric": "error_rate", "threshold": 0.05, "operator": ">", "severity": "critical"},
#       {"metric": "p99_ms",     "threshold": 5000, "operator": ">", "severity": "warning"},
#   ]
#   metrics = {"error_rate": 0.08, "p99_ms": 3000}
#   check_alerts(metrics, rules)
#   -> [{"metric": "error_rate", "threshold": 0.05, "operator": ">", "severity": "critical"}]
#      (only the error_rate rule triggers; p99_ms 3000 is NOT > 5000)
#
# =============================================================================

def check_alerts(metrics, rules):
    """
    Evaluate alert rules against current metrics and return triggered rules.

    Parameters
    ----------
    metrics : dict  -- current metric values, e.g. {"error_rate": 0.08}
    rules   : list  -- list of rule dicts (see format above)

    Returns
    -------
    list -- the subset of rules whose condition is currently True
    """
    # TODO: Implement this function.
    #
    # Step 1: Create an empty list to collect triggered rules:
    #         triggered = []
    #
    # Step 2: Loop over each rule in rules:
    #         for rule in rules:
    #
    # Step 3: Get the metric name and current value:
    #         metric_name = rule["metric"]
    #         if metric_name not in metrics:
    #             continue   # skip rules for metrics we don't have data for
    #         value = metrics[metric_name]
    #         threshold = rule["threshold"]
    #         operator  = rule["operator"]
    #
    # Step 4: Evaluate the condition using the operator string.
    #         Use an if/elif chain:
    #             if operator == ">":  triggered_flag = value > threshold
    #             elif operator == "<":  ...
    #             elif operator == ">=": ...
    #             elif operator == "<=": ...
    #             else: triggered_flag = False   # unknown operator -- skip
    #
    # Step 5: If triggered_flag is True, append the rule to triggered.
    #
    # Step 6: Return triggered.

    pass   # replace with your implementation


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests():
    """Run all exercise tests and print results."""

    print("=" * 60)
    print("Exercise 03: Monitoring and Observability -- Test Results")
    print("=" * 60)

    # ---- Exercise 1 tests ---------------------------------------------------
    print("\n--- Exercise 1: log_event ---")

    entry = log_event("ERROR", "DB connection failed", user_id="u1", latency_ms=320)

    if entry is not None:
        status = "PASS" if entry.get("level") == "ERROR" else "FAIL"
        print(status + "  entry['level'] == 'ERROR'  (got: " + str(entry.get("level")) + ")")

        status = "PASS" if entry.get("message") == "DB connection failed" else "FAIL"
        print(status + "  entry['message'] == 'DB connection failed'  (got: " + str(entry.get("message")) + ")")

        status = "PASS" if entry.get("user_id") == "u1" else "FAIL"
        print(status + "  entry['user_id'] == 'u1'  (got: " + str(entry.get("user_id")) + ")")

        status = "PASS" if entry.get("latency_ms") == 320 else "FAIL"
        print(status + "  entry['latency_ms'] == 320  (got: " + str(entry.get("latency_ms")) + ")")

        has_timestamp = isinstance(entry.get("timestamp"), str) and len(entry.get("timestamp", "")) > 0
        status = "PASS" if has_timestamp else "FAIL"
        print(status + "  entry['timestamp'] is a non-empty string  (got: " + str(entry.get("timestamp")) + ")")
    else:
        print("FAIL  log_event returned None (not implemented yet)")

    # ---- Exercise 2 tests ---------------------------------------------------
    print("\n--- Exercise 2: compute_percentile ---")

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    result = compute_percentile(data, 0.9)
    status = "PASS" if result == 10 else "FAIL"
    print(status + "  p90 of [1..10] == 10  (got: " + str(result) + ")")

    result = compute_percentile(data, 0.5)
    status = "PASS" if result == 6 else "FAIL"
    print(status + "  p50 of [1..10] == 6  (got: " + str(result) + ")")

    result = compute_percentile(data, 0.0)
    status = "PASS" if result == 1 else "FAIL"
    print(status + "  p0 of [1..10] == 1  (got: " + str(result) + ")")

    result = compute_percentile([], 0.99)
    status = "PASS" if result == 0.0 else "FAIL"
    print(status + "  empty list -> 0.0  (got: " + str(result) + ")")

    # ---- Exercise 3 tests ---------------------------------------------------
    print("\n--- Exercise 3: MetricsSummary ---")

    m = MetricsSummary()
    m.record("latency", 100)
    m.record("latency", 200)
    m.record("latency", 300)

    s = m.summary("latency")

    if s is not None:
        status = "PASS" if s.get("count") == 3 else "FAIL"
        print(status + "  count == 3  (got: " + str(s.get("count")) + ")")

        status = "PASS" if s.get("min") == 100 else "FAIL"
        print(status + "  min == 100  (got: " + str(s.get("min")) + ")")

        status = "PASS" if s.get("max") == 300 else "FAIL"
        print(status + "  max == 300  (got: " + str(s.get("max")) + ")")

        status = "PASS" if s.get("avg") == 200.0 else "FAIL"
        print(status + "  avg == 200.0  (got: " + str(s.get("avg")) + ")")

        status = "PASS" if s.get("p95") is not None else "FAIL"
        print(status + "  p95 key exists  (got: " + str(s.get("p95")) + ")")

        # No data for this metric
        s_empty = m.summary("nonexistent")
        status = "PASS" if s_empty is not None and s_empty.get("count") == 0 else "FAIL"
        print(status + "  missing metric summary count == 0  (got: " + str(s_empty) + ")")
    else:
        print("FAIL  summary() returned None (not implemented yet)")

    # ---- Exercise 4 tests ---------------------------------------------------
    print("\n--- Exercise 4: check_alerts ---")

    rules = [
        {"metric": "error_rate", "threshold": 0.05, "operator": ">",  "severity": "critical"},
        {"metric": "p99_ms",     "threshold": 5000, "operator": ">",  "severity": "warning"},
        {"metric": "cache_hit",  "threshold": 0.5,  "operator": "<",  "severity": "warning"},
    ]
    metrics = {"error_rate": 0.08, "p99_ms": 3000, "cache_hit": 0.3}

    triggered = check_alerts(metrics, rules)

    if triggered is not None:
        # error_rate 0.08 > 0.05  -> should trigger
        # p99_ms     3000 > 5000  -> should NOT trigger
        # cache_hit  0.3  < 0.5   -> should trigger
        status = "PASS" if len(triggered) == 2 else "FAIL"
        print(status + "  2 rules triggered  (got: " + str(len(triggered)) + ")")

        triggered_metrics = [r["metric"] for r in triggered]
        status = "PASS" if "error_rate" in triggered_metrics else "FAIL"
        print(status + "  error_rate rule triggered  (got: " + str(triggered_metrics) + ")")

        status = "PASS" if "p99_ms" not in triggered_metrics else "FAIL"
        print(status + "  p99_ms rule did NOT trigger  (got: " + str(triggered_metrics) + ")")

        status = "PASS" if "cache_hit" in triggered_metrics else "FAIL"
        print(status + "  cache_hit rule triggered  (got: " + str(triggered_metrics) + ")")

        # No alerts
        no_triggers = check_alerts({"error_rate": 0.01}, rules)
        status = "PASS" if no_triggers == [] else "FAIL"
        print(status + "  no rules triggered -> []  (got: " + str(no_triggers) + ")")
    else:
        print("FAIL  check_alerts returned None (not implemented yet)")

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
SOLUTION: Exercise 03 - Monitoring and Observability
======================================================

----------------------------------------------------------------------
EXERCISE 1: log_event
----------------------------------------------------------------------

def log_event(level, message, **fields):
    # Build the base entry with the three required fields.
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),  # ISO 8601 string
        "level":     level,
        "message":   message,
    }
    # Merge the extra kwargs into the entry dict.
    # **fields spreads the key-value pairs, like Object.assign() in JS
    # or dict.Merge() in C#.
    entry.update(fields)   # .update() merges another dict into this one
    return entry

    # Alternative one-liner using dict unpacking:
    # return {"timestamp": datetime.datetime.now().isoformat(),
    #         "level": level, "message": message, **fields}


----------------------------------------------------------------------
EXERCISE 2: compute_percentile
----------------------------------------------------------------------

def compute_percentile(values, p):
    if not values:                              # empty list -- return zero
        return 0.0

    sorted_values = sorted(values)             # ascending sort, new list

    index = int(p * len(sorted_values))        # compute raw index (truncated)
    index = min(index, len(sorted_values) - 1) # clamp so we never go out of bounds

    return sorted_values[index]


----------------------------------------------------------------------
EXERCISE 3: MetricsSummary
----------------------------------------------------------------------

class MetricsSummary:

    def __init__(self):
        self._data = {}   # metric_name -> list of float values

    def record(self, name, value):
        # setdefault: if name is missing, insert [] and return it.
        # Then append value to whatever list is there.
        self._data.setdefault(name, []).append(value)

    def summary(self, name):
        data = self._data.get(name, [])   # empty list if name unknown

        if not data:                       # no samples recorded
            return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0, "p95": 0.0}

        count = len(data)
        mn    = min(data)
        mx    = max(data)
        avg   = sum(data) / count          # plain division always gives float in Python 3
        p95   = compute_percentile(data, 0.95)

        return {"count": count, "min": mn, "max": mx, "avg": avg, "p95": p95}


----------------------------------------------------------------------
EXERCISE 4: check_alerts
----------------------------------------------------------------------

def check_alerts(metrics, rules):
    triggered = []   # will hold rules whose condition is True

    for rule in rules:
        metric_name = rule["metric"]

        if metric_name not in metrics:   # no data for this metric -- skip
            continue

        value     = metrics[metric_name]
        threshold = rule["threshold"]
        operator  = rule["operator"]

        # Evaluate the comparison based on the operator string.
        if operator == ">":
            triggered_flag = value > threshold
        elif operator == "<":
            triggered_flag = value < threshold
        elif operator == ">=":
            triggered_flag = value >= threshold
        elif operator == "<=":
            triggered_flag = value <= threshold
        else:
            triggered_flag = False   # unknown operator -- treat as not triggered

        if triggered_flag:
            triggered.append(rule)   # this rule fired

    return triggered
"""

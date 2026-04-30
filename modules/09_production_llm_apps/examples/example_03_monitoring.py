# =============================================================================
# MODULE 09 -- Example 03: Monitoring and Observability
# =============================================================================
# Goal: Build a complete observability system using ONLY Python stdlib.
#       No external packages required -- everything runs out of the box.
#
# C#/.NET analogy overview:
#   Structured logging  -> Serilog / NLog with a JSON sink
#   Metrics collector   -> Application Insights custom metrics
#   Distributed tracing -> Azure Application Insights distributed traces
#   Alerting engine     -> Azure Monitor alert rules
# =============================================================================

# =============================================================================
# GLOSSARY  (read this first -- know the words before the code)
# =============================================================================
# observability      : The ability to understand what is happening INSIDE a
#                      system by looking at its OUTPUTS (logs, metrics, traces).
#                      Think of it as X-ray vision for your app.
#
# structured logging : Writing log entries as key-value data (JSON) instead of
#                      plain sentences.  Makes logs machine-searchable.
#                      Bad:  "User 42 failed after 300ms"
#                      Good: {"user_id": 42, "duration_ms": 300, "level": "ERROR"}
#
# metrics            : Numbers that describe system behaviour over time.
#                      Examples: requests per second, error rate, memory usage.
#
# Prometheus         : A popular open-source tool that COLLECTS and STORES
#                      metrics by "scraping" (polling) your app on an HTTP
#                      endpoint.  We simulate its text format here.
#
# Grafana            : A dashboard tool that READS from Prometheus and draws
#                      pretty graphs.  Think of it as Power BI for your app.
#
# distributed tracing: Following a single request as it travels through
#                      MULTIPLE services (API -> Auth -> DB -> Cache ...).
#                      Each hop is recorded as a "span".
#
# span               : One unit of work inside a trace.  Has a start time,
#                      end time, a name, and optional key-value tags.
#
# trace              : A collection of spans that all belong to the same
#                      end-to-end request.  Identified by a trace_id.
#
# alert              : An automatic notification when a metric crosses a
#                      threshold.  "Error rate > 5% -- page the on-call dev!"
#
# SLO (Service Level Objective): A TARGET for reliability.
#                      Example: "99.9% of requests succeed" or
#                      "p99 latency < 500ms".  Breaking an SLO is serious.
#
# latency percentile : Sort all response times; p50 is the MEDIAN (half are
#                      faster, half slower).  p95 means 95% of requests are
#                      faster than this value.  p99 is the slowest 1%.
#
# p50 / p95 / p99    : See above.  p99 is the "worst case" most teams watch.
# =============================================================================

# ASCII ARCHITECTURE DIAGRAM
# --------------------------
#
#   Your LLM App
#       |
#       +---> [StructuredLogger]  ---> log entries (JSON-like dicts)
#       |
#       +---> [MetricsCollector]  ---> Prometheus scrape endpoint (text)
#       |
#       +---> [Tracer]            ---> spans in a trace waterfall
#       |
#       +---> [CostTracker]       ---> per-user cost breakdown
#
#   [AlertEngine]  <--- reads MetricsCollector thresholds
#       |
#       +--> Fires alerts: "p99 latency too high!", "error rate spike!"
#
#   [generate_dashboard()]  <--- reads all of the above
#       |
#       +--> Prints ASCII bar-chart summary to the terminal
#
# =============================================================================

# ---------- stdlib imports only ------------------------------------------
import json          # for serialising log entries to JSON strings
import time          # for timestamps (time.time() returns Unix epoch float)
import random        # for generating fake latency/request data in demos
import math          # for sorting helpers (math.inf used as sentinel)
import uuid          # for generating unique IDs (trace_id, span_id, request_id)
from datetime import datetime, timezone  # for human-readable timestamps
from collections import defaultdict      # like Dictionary<K, List<V>> in C#


# =============================================================================
# PART 1 -- STRUCTURED LOGGER
# =============================================================================
# C# analogy: Serilog with .WriteTo.File() and outputTemplate as JSON.
#             In C# you write:  Log.Information("Request done {@User}", user)
#             Here we store every log entry as a Python dict (like a C# object).
# =============================================================================

class StructuredLogger:
    """
    Stores log entries as structured dictionaries (like JSON objects).
    Supports filtering by level or user_id -- impossible with plain print().
    """

    # Class-level list of log entries -- shared across all callers in this demo.
    # C# analogy: a static List<LogEntry> field on the class.
    _entries = []

    # Define the numeric "weight" of each level so we can sort/compare them.
    # C# analogy: an enum LogLevel { Debug=0, Info=1, Warning=2, Error=3 }
    LEVELS = {
        "DEBUG":   0,
        "INFO":    1,
        "WARNING": 2,
        "ERROR":   3,
    }

    def __init__(self, service_name="llm-app"):
        # service_name labels every log entry so we know which service wrote it.
        # C# analogy: ILogger<MyService> -- the generic type parameter becomes the category.
        self.service_name = service_name

    def log(self, level, message, **extra_fields):
        """
        Core logging method.  Accepts keyword arguments for any extra context.

        Parameters:
            level       : "DEBUG", "INFO", "WARNING", or "ERROR"
            message     : human-readable description of what happened
            **extra_fields : any additional key=value pairs
                           e.g. request_id="abc", user_id=42, duration_ms=120

        C# analogy:
            logger.LogInformation("Request done. RequestId={RequestId}", requestId)
            The 'extra_fields' become the structured properties in the log event.
        """
        # Build a dictionary for this log entry -- like constructing a C# DTO.
        entry = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),  # ISO-8601 timestamp
            "service":     self.service_name,                        # which service logged this
            "level":       level.upper(),                            # normalise to upper-case
            "message":     message,                                  # human-readable text
            "request_id":  extra_fields.pop("request_id", None),     # pull from extras or None
            "user_id":     extra_fields.pop("user_id", None),        # pull from extras or None
            "duration_ms": extra_fields.pop("duration_ms", None),    # pull from extras or None
        }
        # Merge any remaining extra fields into the entry dict.
        # e.g. model="gpt-4", tokens=512 would be added here.
        entry.update(extra_fields)

        # Append the completed entry to the in-memory list.
        StructuredLogger._entries.append(entry)

        # Also print so we see it in the terminal -- real systems ship to ELK/Splunk.
        # json.dumps() converts the dict to a JSON string (like JsonSerializer.Serialize in C#).
        print(f"[LOG] {entry['level']:7s} | {entry['timestamp']} | {entry['message']}")

    # Convenience wrappers -- identical to ILogger.LogInformation(), LogWarning() etc.
    def info(self, message, **kw):
        self.log("INFO", message, **kw)

    def warning(self, message, **kw):
        self.log("WARNING", message, **kw)

    def error(self, message, **kw):
        self.log("ERROR", message, **kw)

    def debug(self, message, **kw):
        self.log("DEBUG", message, **kw)

    def filter_by(self, level=None, user_id=None):
        """
        Search the in-memory log for entries matching the given criteria.
        C# analogy: _entries.Where(e => e.Level == level && e.UserId == userId)
        """
        results = []                                  # start with empty list
        for entry in StructuredLogger._entries:       # iterate every stored entry
            # Check level filter -- skip if level was given and does not match.
            if level and entry["level"] != level.upper():
                continue                              # like 'continue' in C# foreach
            # Check user_id filter -- skip if user_id was given and does not match.
            if user_id is not None and entry["user_id"] != user_id:
                continue
            results.append(entry)                     # entry passed all filters -- keep it
        return results


def demo_structured_logger():
    """Run Part 1 demo: log 10 events, then filter for errors."""
    print("\n" + "=" * 70)
    print("PART 1 -- STRUCTURED LOGGER DEMO")
    print("=" * 70)

    logger = StructuredLogger(service_name="llm-chat-api")  # create logger instance

    # Simulate 10 log events that a real LLM API might produce.
    logger.info("Service started", version="1.0.0")
    logger.info("Request received", request_id="req-001", user_id=101, model="gpt-4")
    logger.debug("Tokens counted", request_id="req-001", input_tokens=512)
    logger.info("LLM response received", request_id="req-001", duration_ms=340, output_tokens=128)
    logger.warning("Rate limit approaching", user_id=101, remaining_quota=5)
    logger.error("LLM API timeout", request_id="req-002", user_id=202, duration_ms=5000, model="gpt-4")
    logger.info("Request received", request_id="req-003", user_id=303, model="gpt-3.5")
    logger.info("LLM response received", request_id="req-003", duration_ms=120, output_tokens=64)
    logger.error("Invalid API key", request_id="req-004", user_id=202, model="claude")
    logger.warning("High latency detected", request_id="req-005", duration_ms=1800)

    # Filter: show only ERROR entries.
    print("\n--- Filtering for ERROR level only ---")
    errors = logger.filter_by(level="ERROR")    # returns list of matching dicts
    for e in errors:
        # Print selected fields in a readable format.
        print(f"  ERROR | user={e['user_id']} | {e['message']} | dur={e['duration_ms']}ms")

    # Explain WHY structured logging beats plain print().
    print("\n--- Why structured logging beats plain print() ---")
    print("  - Grep by user_id:   filter_by(user_id=202) -> all events for that user")
    print("  - Filter by level:   filter_by(level='ERROR') -> only errors")
    print("  - Ship to ELK stack: each entry is a dict -- serialize to JSON, send to Elasticsearch")
    print("  - C# equivalent:     Serilog with .WriteTo.Elasticsearch() or NLog JSON layout")


# =============================================================================
# PART 2 -- METRICS COLLECTOR (Prometheus-style)
# =============================================================================
# C# analogy: Application Insights custom metrics (TrackMetric, TrackEvent).
#             Prometheus uses a simple text format; we reproduce it here.
# =============================================================================

class MetricsCollector:
    """
    Collects three kinds of metrics:
      counter   -- counts events, only goes UP (requests received, errors)
      gauge     -- current snapshot value (queue size, memory %)
      histogram -- distributes values into buckets to compute percentiles
    """

    def __init__(self):
        # Store counter values.  Key: (name, labels_tuple), Value: float.
        # defaultdict(float) means missing keys start at 0.0 automatically.
        # C# analogy: Dictionary<string, double> with a default-value factory.
        self._counters   = defaultdict(float)

        # Store gauge values.
        self._gauges     = defaultdict(float)

        # Store raw histogram samples as lists of floats.
        self._histograms = defaultdict(list)

    def _labels_key(self, labels):
        """
        Convert a dict of labels to a hashable tuple so it can be a dict key.
        C# analogy: string.Join(",", labels.Select(kv => $"{kv.Key}={kv.Value}"))
        """
        # sorted() ensures {"env":"prod","model":"gpt-4"} == {"model":"gpt-4","env":"prod"}
        return tuple(sorted(labels.items()))

    def counter(self, name, value=1, labels={}):
        """
        Increment a counter by 'value' (default 1).
        Counters NEVER decrease.  They reset only on restart.
        C# analogy: telemetryClient.TrackMetric("requests_total", 1)
        """
        key = (name, self._labels_key(labels))   # combine name + labels into one key
        self._counters[key] += value              # add to existing count (starts at 0)

    def gauge(self, name, value, labels={}):
        """
        Set a gauge to exactly 'value'.  Gauges CAN go up or down.
        C# analogy: Environment.WorkingSet (memory can increase or decrease)
        """
        key = (name, self._labels_key(labels))
        self._gauges[key] = value                 # overwrite with the latest reading

    def histogram(self, name, value, labels={}):
        """
        Record a single observation (e.g., one request's latency).
        All values are kept so we can compute percentiles later.
        C# analogy: Stopwatch.ElapsedMilliseconds stored in a List<double>
        """
        key = (name, self._labels_key(labels))
        self._histograms[key].append(value)       # keep growing list of raw samples

    def get_stats(self, name):
        """
        Compute summary statistics for a histogram metric.
        Returns: dict with count, sum, min, max, p50, p95, p99.
        C# analogy: LINQ .Min(), .Max(), .Average() plus percentile calculation.
        """
        # Collect all samples whose metric name matches (ignore labels for simplicity).
        all_values = []
        for (metric_name, _labels), samples in self._histograms.items():
            if metric_name == name:
                all_values.extend(samples)        # add all samples for this name

        if not all_values:                        # guard: no data yet
            return {}

        sorted_vals = sorted(all_values)          # sort ascending -- needed for percentiles
        n = len(sorted_vals)                      # total sample count

        def percentile(p):
            """
            Return the value at the p-th percentile.
            Formula: index = ceil(p/100 * n) - 1, clamped to valid range.
            C# analogy: values.OrderBy(x=>x).ElementAt((int)Math.Ceiling(p/100.0*n)-1)
            """
            idx = int(math.ceil(p / 100.0 * n)) - 1   # compute 0-based index
            idx = max(0, min(idx, n - 1))              # clamp to [0, n-1]
            return sorted_vals[idx]

        return {
            "count": n,
            "sum":   sum(sorted_vals),
            "min":   sorted_vals[0],
            "max":   sorted_vals[-1],
            "p50":   percentile(50),   # median -- half of requests are faster
            "p95":   percentile(95),   # 95th percentile -- only 5% are slower
            "p99":   percentile(99),   # 99th percentile -- our "worst case" benchmark
        }

    def export_prometheus_format(self):
        """
        Produce text in the Prometheus exposition format.
        Prometheus scrapes this endpoint every N seconds.

        Format example:
          # HELP requests_total Total HTTP requests
          # TYPE requests_total counter
          requests_total{model="gpt-4"} 42

        C# analogy: prometheus-net library's /metrics HTTP endpoint.
        """
        lines = []   # collect output lines

        # Export counters.
        for (name, labels), value in self._counters.items():
            lines.append(f"# TYPE {name} counter")
            label_str = ",".join(f'{k}="{v}"' for k, v in labels)  # format labels
            if label_str:
                lines.append(f"{name}{{{label_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

        # Export gauges.
        for (name, labels), value in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            label_str = ",".join(f'{k}="{v}"' for k, v in labels)
            if label_str:
                lines.append(f"{name}{{{label_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

        # Export histogram summaries (count, sum).
        for (name, labels), samples in self._histograms.items():
            lines.append(f"# TYPE {name} histogram")
            label_str = ",".join(f'{k}="{v}"' for k, v in labels)
            lbl = "{" + label_str + "}" if label_str else ""
            lines.append(f"{name}_count{lbl} {len(samples)}")
            lines.append(f"{name}_sum{lbl} {sum(samples):.2f}")

        return "\n".join(lines)   # join all lines into one big string


def demo_metrics_collector():
    """Run Part 2 demo: 50 simulated requests, show percentiles and Prometheus output."""
    print("\n" + "=" * 70)
    print("PART 2 -- METRICS COLLECTOR DEMO")
    print("=" * 70)

    mc = MetricsCollector()    # create collector instance

    random.seed(42)            # seed for reproducibility -- same numbers every run

    # Simulate 50 incoming requests.
    for i in range(50):
        # Pick a random latency between 50ms and 500ms.
        latency = random.randint(50, 500)

        # Randomly choose a model for this request.
        model = random.choice(["gpt-4", "gpt-3.5", "claude"])

        # Record a counter tick -- one more request arrived.
        mc.counter("requests_total", labels={"model": model})

        # Record this request's latency in the histogram.
        mc.histogram("request_latency_ms", latency, labels={"model": model})

        # Simulate occasional errors (10% chance).
        if random.random() < 0.10:
            mc.counter("errors_total", labels={"model": model, "reason": "timeout"})

    # Set a gauge for current active connections (snapshot value).
    mc.gauge("active_connections", value=17)

    # Retrieve and print statistics.
    stats = mc.get_stats("request_latency_ms")  # compute percentiles
    print("\n--- Request Latency Statistics (across all models) ---")
    print(f"  Count : {stats['count']} requests")
    print(f"  Min   : {stats['min']} ms")
    print(f"  Max   : {stats['max']} ms")
    print(f"  p50   : {stats['p50']} ms   (median -- half are faster than this)")
    print(f"  p95   : {stats['p95']} ms   (95% of requests finish by this time)")
    print(f"  p99   : {stats['p99']} ms   (only 1% are slower -- our 'worst case')")

    print("\n--- Prometheus Exposition Format (what /metrics endpoint returns) ---")
    prom_text = mc.export_prometheus_format()
    # Print only first 20 lines to keep the demo readable.
    for line in prom_text.splitlines()[:20]:
        print("  " + line)
    print("  ... (truncated)")

    return mc    # return so the alert engine can reuse it


# =============================================================================
# PART 3 -- REQUEST TRACER (simulated distributed tracing)
# =============================================================================
# C# analogy: Azure Application Insights distributed traces.
#             In .NET you use Activity / ActivitySource from System.Diagnostics.
#             OpenTelemetry SDK is the standard cross-platform equivalent.
# =============================================================================

class Span:
    """
    Represents one unit of work inside a distributed trace.
    C# analogy: System.Diagnostics.Activity or OpenTelemetry Span.
    """

    def __init__(self, span_id, name, trace_id, parent_span_id=None):
        self.span_id       = span_id          # unique ID for this specific span
        self.trace_id      = trace_id         # ID linking all spans in one request
        self.parent_span_id = parent_span_id  # None for root span; otherwise points to parent
        self.name          = name             # human label e.g. "Auth", "LLM call"
        self.start_time    = time.time()      # wall-clock when span started
        self.end_time      = None             # None until finish() is called
        self.tags          = {}               # key-value metadata (like HTTP status, model name)

    def set_tag(self, key, value):
        """
        Attach metadata to this span.
        C# analogy: activity.SetTag("http.status_code", 200)
        """
        self.tags[key] = value

    def finish(self):
        """Record the end time.  Must be called or duration will be unknown."""
        self.end_time = time.time()

    @property
    def duration_ms(self):
        """Compute duration in milliseconds.  Returns None if not finished."""
        if self.end_time is None:
            return None
        return int((self.end_time - self.start_time) * 1000)  # convert seconds to ms


class Tracer:
    """
    Manages spans for one or more distributed traces.
    C# analogy: OpenTelemetry TracerProvider / ITracer interface.
    """

    def __init__(self):
        # Store all spans keyed by trace_id -> list of Span objects.
        # C# analogy: Dictionary<Guid, List<Span>>
        self._traces = defaultdict(list)

    def start_span(self, name, trace_id=None, parent_span_id=None):
        """
        Create and register a new span.

        If trace_id is None, start a brand-new trace.
        C# analogy: tracer.StartActiveSpan("operation-name")
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())       # generate new trace ID for a fresh request

        span_id = str(uuid.uuid4())[:8]        # short 8-char ID for readability
        span = Span(
            span_id       = span_id,
            name          = name,
            trace_id      = trace_id,
            parent_span_id= parent_span_id,
        )
        self._traces[trace_id].append(span)    # register the span
        return span                            # caller holds the span and must call finish()

    def get_trace(self, trace_id):
        """
        Return all spans for a trace, sorted by start time (chronological order).
        C# analogy: traces[traceId].OrderBy(s => s.StartTime)
        """
        spans = self._traces.get(trace_id, [])
        return sorted(spans, key=lambda s: s.start_time)   # lambda = arrow function in C#

    def print_trace(self, trace_id):
        """
        Print an ASCII waterfall diagram showing span timing.

        Example output:
          [API Handler   ] [==============================] 450ms
            [Auth Service ] [====]                           50ms
            [LLM Call     ] [======================]        350ms
              [DB Write    ] [===]                            30ms
        """
        spans = self.get_trace(trace_id)
        if not spans:
            print("  No spans found for trace:", trace_id)
            return

        # Find the absolute start of the whole trace (the earliest span start time).
        trace_start = min(s.start_time for s in spans)
        # Find the absolute end (the latest span end time).
        trace_end   = max(s.end_time for s in spans if s.end_time is not None)
        total_ms    = max((trace_end - trace_start) * 1000, 1)   # avoid division by zero

        BAR_WIDTH = 40    # total characters available for the bar area

        print(f"\n  Trace ID: {trace_id}")
        print(f"  Total duration: {int(total_ms)} ms")
        print()

        # Build a parent -> [children] map so we can indent child spans.
        parent_map = defaultdict(list)
        for span in spans:
            parent_map[span.parent_span_id].append(span)

        def render(span, depth=0):
            """Recursively render a span and its children."""
            indent = "  " * depth     # 2 spaces per nesting level

            if span.end_time is None:
                span.finish()         # guard: auto-finish if caller forgot

            # Compute where the bar starts and ends as fractions of total trace width.
            offset_frac   = (span.start_time - trace_start) * 1000 / total_ms
            duration_frac = span.duration_ms / total_ms

            # Convert fractions to character counts.
            bar_start = int(offset_frac   * BAR_WIDTH)
            bar_len   = max(int(duration_frac * BAR_WIDTH), 1)  # at least 1 char

            # Build the bar string: spaces before, = for duration, spaces after.
            bar = " " * bar_start + "=" * bar_len
            bar = bar.ljust(BAR_WIDTH)   # pad to full width

            # Format the name column, left-justified in 18 chars.
            name_col = f"{indent}{span.name}".ljust(18)

            print(f"  {name_col} [{bar}] {span.duration_ms}ms")

            # Recurse into children.
            for child in parent_map.get(span.span_id, []):
                render(child, depth + 1)

        # Start rendering from root spans (parent_span_id is None).
        for root_span in parent_map.get(None, []):
            render(root_span)


def demo_tracer():
    """Run Part 3 demo: simulate API -> Auth -> LLM -> DB trace."""
    print("\n" + "=" * 70)
    print("PART 3 -- REQUEST TRACER DEMO")
    print("=" * 70)

    tracer = Tracer()   # create tracer

    # --- Simulate a single incoming request with 4 spans ---

    # Span 1: the top-level API handler (root span -- no parent).
    api_span = tracer.start_span("API Handler")
    api_span.set_tag("http.method", "POST")
    api_span.set_tag("http.path", "/v1/chat")
    trace_id = api_span.trace_id           # capture trace ID for child spans

    # Simulate a small delay before auth starts.
    time.sleep(0.03)   # 30 ms

    # Span 2: authentication check (child of API Handler).
    auth_span = tracer.start_span("Auth Service", trace_id=trace_id, parent_span_id=api_span.span_id)
    auth_span.set_tag("auth.method", "JWT")
    time.sleep(0.05)   # 50 ms -- time to verify JWT token
    auth_span.finish()

    # Span 3: LLM API call (child of API Handler, sibling of Auth).
    llm_span = tracer.start_span("LLM Call", trace_id=trace_id, parent_span_id=api_span.span_id)
    llm_span.set_tag("model", "gpt-4")
    llm_span.set_tag("input_tokens", 512)
    time.sleep(0.18)   # 180 ms -- time waiting for LLM to respond
    llm_span.set_tag("output_tokens", 128)
    llm_span.finish()

    # Span 4: database write (child of LLM Call -- nested deeper).
    db_span = tracer.start_span("DB Write", trace_id=trace_id, parent_span_id=llm_span.span_id)
    db_span.set_tag("db.type", "postgres")
    db_span.set_tag("db.table", "conversations")
    time.sleep(0.04)   # 40 ms -- writing conversation record
    db_span.finish()

    # Finish the root span last (it wraps everything).
    api_span.finish()

    # Print the waterfall.
    print("\n  ASCII Waterfall Diagram:")
    tracer.print_trace(trace_id)


# =============================================================================
# PART 4 -- COST TRACKER
# =============================================================================
# C# analogy: a billing/metering service that tracks API call costs per user.
#             Similar to Azure Cost Management APIs.
# =============================================================================

# Pricing table: cost in USD per 1,000 tokens.
# Update these if provider prices change.
COST_PER_1K = {
    "gpt-4":    {"input": 0.03,  "output": 0.06 },
    "gpt-3.5":  {"input": 0.001, "output": 0.002},
    "claude":   {"input": 0.015, "output": 0.075},
}


class CostTracker:
    """
    Records token usage per user and calculates monetary cost.
    C# analogy: a metering service with per-user aggregation buckets.
    """

    def __init__(self):
        # Map user_id -> list of usage records (each is a dict).
        # C# analogy: Dictionary<int, List<UsageRecord>>
        self._records = defaultdict(list)

    def record_request(self, user_id, model, input_tokens, output_tokens):
        """
        Store one API call's token usage.

        Parameters:
            user_id       : identifier for the user (int or str)
            model         : model name -- must match a key in COST_PER_1K
            input_tokens  : number of prompt tokens consumed
            output_tokens : number of completion tokens produced
        """
        pricing = COST_PER_1K.get(model, {"input": 0.0, "output": 0.0})  # default to 0 if unknown

        # Calculate cost: (tokens / 1000) * price_per_1k
        input_cost  = (input_tokens  / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost  = input_cost + output_cost

        record = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "model":         model,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      total_cost,
        }
        self._records[user_id].append(record)   # store under this user

    def get_user_cost(self, user_id, period="daily"):
        """
        Sum all costs for a user.  (period is a label; full history returned for now.)
        C# analogy: records.Where(r => r.UserId == userId).Sum(r => r.CostUsd)
        """
        return sum(r["cost_usd"] for r in self._records[user_id])

    def get_total_cost(self, period="daily"):
        """Sum costs across ALL users."""
        total = 0.0
        for records in self._records.values():      # .values() = all lists in the dict
            total += sum(r["cost_usd"] for r in records)
        return total

    def top_spenders(self, n=5):
        """
        Return the top-n users sorted by total cost, highest first.
        C# analogy: users.OrderByDescending(u => u.TotalCost).Take(n)
        """
        user_costs = [
            (user_id, self.get_user_cost(user_id))
            for user_id in self._records
        ]
        # sort descending by cost (index [1] of each tuple)
        user_costs.sort(key=lambda x: x[1], reverse=True)
        return user_costs[:n]   # return only the top n


def demo_cost_tracker():
    """Run Part 4 demo: 20 requests from 5 users, show cost breakdown."""
    print("\n" + "=" * 70)
    print("PART 4 -- COST TRACKER DEMO")
    print("=" * 70)

    ct = CostTracker()
    random.seed(7)

    user_ids = [101, 202, 303, 404, 505]   # five simulated users
    models   = list(COST_PER_1K.keys())    # ["gpt-4", "gpt-3.5", "claude"]

    # Simulate 20 API calls spread across users.
    for _ in range(20):
        user  = random.choice(user_ids)
        model = random.choice(models)
        inp   = random.randint(200, 1000)   # random input token count
        out   = random.randint(50,  500)    # random output token count
        ct.record_request(user, model, inp, out)

    print(f"\n  Total cost today: ${ct.get_total_cost():.4f}")
    print("\n  Top spenders:")
    for uid, cost in ct.top_spenders(n=5):
        # Show a simple bar proportional to cost.
        bar_len = int(cost / ct.get_total_cost() * 30)   # scale to 30 chars max
        bar = "=" * bar_len
        print(f"    User {uid:3d} | {bar:<30} | ${cost:.4f}")

    return ct    # return so other parts can reuse the data


# =============================================================================
# PART 5 -- ALERTING ENGINE
# =============================================================================
# C# analogy: Azure Monitor alert rules with severity levels.
#             Define a condition -> get notified when it fires.
# =============================================================================

class Alert:
    """A fired alert notification."""

    def __init__(self, name, severity, message):
        self.name        = name
        self.severity    = severity           # "critical", "warning", or "info"
        self.message     = message
        self.triggered_at = datetime.now(timezone.utc).isoformat()


class AlertEngine:
    """
    Holds alert rules and checks them against a snapshot of current metrics.
    C# analogy: Azure Monitor or Prometheus Alertmanager rule evaluation.
    """

    def __init__(self):
        # List of rule dicts: {name, metric, operator, threshold, severity}
        self._rules = []

    def add_rule(self, name, metric, operator, threshold, severity="warning"):
        """
        Register an alert rule.

        Parameters:
            name      : human label e.g. "HighErrorRate"
            metric    : key in the metrics dict passed to check_all()
            operator  : ">" | "<" | ">=" | "<=" | "=="
            threshold : numeric threshold value
            severity  : "critical" | "warning" | "info"
        """
        self._rules.append({
            "name":      name,
            "metric":    metric,
            "operator":  operator,
            "threshold": threshold,
            "severity":  severity,
        })

    def check_all(self, metrics):
        """
        Evaluate all rules against the provided metrics snapshot.

        Parameters:
            metrics : dict mapping metric name -> numeric value

        Returns:
            list of Alert objects for every rule that fired.

        C# analogy: a loop over IAlertRule[] calling rule.Evaluate(metricsSnapshot)
        """
        fired = []   # alerts that fired this evaluation cycle

        for rule in self._rules:
            metric_value = metrics.get(rule["metric"])   # look up the metric value

            if metric_value is None:
                continue    # metric not present -- skip this rule

            op  = rule["operator"]
            thr = rule["threshold"]

            # Evaluate the condition -- like a switch statement in C#.
            triggered = (
                (op == ">"  and metric_value >  thr) or
                (op == "<"  and metric_value <  thr) or
                (op == ">=" and metric_value >= thr) or
                (op == "<=" and metric_value <= thr) or
                (op == "==" and metric_value == thr)
            )

            if triggered:
                msg = (
                    f"{rule['name']} FIRED: "
                    f"{rule['metric']} = {metric_value} "
                    f"{op} {thr} (threshold)"
                )
                fired.append(Alert(rule["name"], rule["severity"], msg))

        return fired


def demo_alert_engine():
    """Run Part 5 demo: define 3 rules, simulate metrics that fire 2 of them."""
    print("\n" + "=" * 70)
    print("PART 5 -- ALERTING ENGINE DEMO")
    print("=" * 70)

    engine = AlertEngine()

    # Define alert rules (like Azure Monitor rules or Prometheus alerting rules).
    engine.add_rule("HighErrorRate",    metric="error_rate_pct",  operator=">",  threshold=5.0,   severity="critical")
    engine.add_rule("HighP99Latency",   metric="latency_p99_ms",  operator=">",  threshold=2000,  severity="warning")
    engine.add_rule("DailyCostLimit",   metric="cost_usd_today",  operator=">",  threshold=10.0,  severity="warning")

    # Simulate a metrics snapshot -- imagine these come from MetricsCollector.
    current_metrics = {
        "error_rate_pct":  8.3,    # 8.3% errors -- above the 5% threshold -> FIRES
        "latency_p99_ms":  1450,   # 1450ms -- below 2000ms threshold -> does NOT fire
        "cost_usd_today":  12.40,  # $12.40 -- above $10 threshold -> FIRES
    }

    print("\n  Current metrics snapshot:")
    for k, v in current_metrics.items():
        print(f"    {k:<25} = {v}")

    alerts = engine.check_all(current_metrics)

    print(f"\n  {len(alerts)} alert(s) triggered:")
    for alert in alerts:
        # Use ASCII severity indicator instead of emoji.
        sev_marker = "!!!" if alert.severity == "critical" else "!!"
        print(f"    [{sev_marker} {alert.severity.upper()}] {alert.message}")
        print(f"           Triggered at: {alert.triggered_at}")

    return engine


# =============================================================================
# PART 6 -- DASHBOARD SUMMARY (ASCII)
# =============================================================================
# C# analogy: a real-time operations dashboard like Azure Portal overview blade.
# =============================================================================

def generate_dashboard(metrics_collector, cost_tracker):
    """
    Print an ASCII dashboard with key operational metrics.
    Uses = characters to draw proportional bars.
    """
    print("\n" + "=" * 70)
    print("PART 6 -- LIVE DASHBOARD SUMMARY")
    print("=" * 70)

    # Gather the numbers we want to display.
    stats        = metrics_collector.get_stats("request_latency_ms")
    p99          = stats.get("p99", 0)
    p50          = stats.get("p50", 0)
    total_cost   = cost_tracker.get_total_cost()
    request_count = stats.get("count", 0)

    # Count errors by summing the errors_total counter values.
    error_count = 0
    for (name, _), val in metrics_collector._counters.items():
        if name == "errors_total":
            error_count += val

    error_rate_pct = (error_count / max(request_count, 1)) * 100   # avoid div by zero

    def bar(value, max_value, width=30):
        """Return a bar string of '=' chars proportional to value/max_value."""
        filled = int((value / max(max_value, 1)) * width)   # how many = to draw
        filled = min(filled, width)                          # never exceed width
        return "=" * filled + " " * (width - filled)        # pad with spaces

    def status(value, warning_threshold, critical_threshold):
        """Return [OK], [WARN], or [CRIT] status string."""
        if value >= critical_threshold:
            return "[CRIT]"
        if value >= warning_threshold:
            return "[WARN]"
        return "[OK]  "

    print()
    print(f"  {'Metric':<22} {'Bar (relative)':<34} {'Value':<10} {'Status'}")
    print(f"  {'-'*22} {'-'*34} {'-'*10} {'-'*6}")

    # Requests
    print(f"  {'Requests (total)':<22} [{bar(request_count, 100)}] {request_count:<10} {status(request_count, 80, 95)}")

    # Error rate (warning at 2%, critical at 5%)
    print(f"  {'Error rate (%)':<22} [{bar(error_rate_pct, 20)}] {error_rate_pct:<9.1f}% {status(error_rate_pct, 2, 5)}")

    # p50 latency
    print(f"  {'Latency p50 (ms)':<22} [{bar(p50, 2000)}] {p50:<10} {status(p50, 500, 1000)}")

    # p99 latency (warning at 1000ms, critical at 2000ms)
    print(f"  {'Latency p99 (ms)':<22} [{bar(p99, 2000)}] {p99:<10} {status(p99, 1000, 2000)}")

    # Daily cost (warning at $5, critical at $10)
    print(f"  {'Cost today (USD)':<22} [{bar(total_cost, 20)}] ${total_cost:<9.4f} {status(total_cost, 5, 10)}")

    print()
    print(f"  Dashboard generated at: {datetime.now(timezone.utc).isoformat()}")


# =============================================================================
# PART 7 -- KEY TAKEAWAYS
# =============================================================================

def print_key_takeaways():
    """Print 5 lessons every production LLM developer should know."""
    print("\n" + "=" * 70)
    print("PART 7 -- KEY TAKEAWAYS")
    print("=" * 70)

    takeaways = [
        (1, "Structured logging beats print()",
            "JSON logs let you grep by user_id, filter by level, and ship to\n"
            "     ELK or Splunk without code changes.  C#: use Serilog, not Console.WriteLine."),

        (2, "Watch p99, not averages",
            "An average latency of 200ms can hide a p99 of 5000ms.  Your worst\n"
            "     1% of users experience the highest frustration.  Set SLOs on p99."),

        (3, "Distributed tracing reveals hidden bottlenecks",
            "A waterfall diagram shows exactly which span (Auth? DB? LLM?) is\n"
            "     slow.  C#: use OpenTelemetry SDK + Azure App Insights exporter."),

        (4, "Cost tracking is non-negotiable",
            "LLM APIs charge per token.  Without per-user cost tracking you\n"
            "     cannot enforce quotas or detect runaway usage.  Set daily budgets."),

        (5, "Automate alerts, do not rely on humans watching dashboards",
            "Define SLOs then write alert rules so the on-call engineer is paged\n"
            "     immediately.  C#: Azure Monitor alert rules with action groups."),
    ]

    for num, title, detail in takeaways:
        print(f"\n  Lesson {num}: {title}")
        print(f"     {detail}")


# =============================================================================
# ENTRY POINT -- run all demos in sequence
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODULE 09 -- Example 03: Monitoring and Observability")
    print("Pure Python stdlib -- no external packages required")
    print("=" * 70)

    # Part 1: structured logging
    demo_structured_logger()

    # Part 2: metrics collector -- save returned object for later use
    mc = demo_metrics_collector()

    # Part 3: distributed tracing waterfall
    demo_tracer()

    # Part 4: cost tracker -- save returned object for dashboard
    ct = demo_cost_tracker()

    # Part 5: alerting engine
    demo_alert_engine()

    # Part 6: ASCII dashboard (uses mc and ct from above)
    generate_dashboard(mc, ct)

    # Part 7: lessons
    print_key_takeaways()

    print("\n" + "=" * 70)
    print("End of Example 03.  Run example_04_security.py next.")
    print("=" * 70)

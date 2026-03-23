# Lesson 3: Monitoring & Observability

**Master production monitoring, logging, and debugging for LLM applications**

---

## Table of Contents
1. [Introduction](#introduction)
2. [Structured Logging](#structured-logging)
3. [Metrics with Prometheus](#metrics-with-prometheus)
4. [Dashboards with Grafana](#dashboards-with-grafana)
5. [Distributed Tracing](#distributed-tracing)
6. [Alerting](#alerting)
7. [Cost Monitoring](#cost-monitoring)
8. [Performance Profiling](#performance-profiling)
9. [Complete Observability Stack](#complete-observability-stack)
10. [Exercises](#exercises)

---

## Introduction

### What You'll Learn

By the end of this lesson, you will:
- Implement structured logging
- Collect and visualize metrics
- Create operational dashboards
- Set up intelligent alerting
- Track costs per user/request
- Debug production issues quickly

### Time Required
**8-10 hours** (including hands-on practice)

### Prerequisites
- Completed Lessons 1-2
- Application deployed to cloud
- Basic understanding of monitoring concepts

---

## The Three Pillars of Observability

### 1. Logs
**What happened?**
```
2024-03-23 10:15:32 INFO User user_123 sent message to GPT-4
2024-03-23 10:15:35 INFO Response generated: 523 tokens, cost $0.0157
2024-03-23 10:15:36 ERROR User quota_exceeded for user_456
```

### 2. Metrics
**How much/how many?**
```
http_requests_total{endpoint="/chat",status="200"} 15423
llm_tokens_used_total{model="gpt-4"} 1523451
api_response_time_seconds{quantile="0.95"} 2.3
```

### 3. Traces
**Where did time go?**
```
Request /chat [2.3s total]
  ├─ Authenticate [0.1s]
  ├─ Database query [0.2s]
  ├─ LLM API call [1.8s]  ← bottleneck!
  └─ Cache write [0.2s]
```

---

## Structured Logging

### Why Structured Logs?

**Bad (Unstructured):**
```python
print("User john@example.com sent message at 10:15")  # Hard to parse
```

**Good (Structured):**
```python
logger.info("user_message_sent", extra={
    "user_email": "john@example.com",
    "timestamp": "2024-03-23T10:15:00Z",
    "message_length": 42,
    "model": "gpt-4"
})
# Easy to search, filter, analyze
```

**C# Comparison:**

```csharp
// C# with Serilog
Log.Information("User {UserEmail} sent message at {Timestamp}",
    userEmail, timestamp);
```

```python
# Python with structlog
logger.info("user_message_sent",
    user_email=user_email,
    timestamp=timestamp)
```

---

### Implementing Structured Logging

```python
# logging_config.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logs"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def setup_logging():
    """Configure structured logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler with JSON formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # File handler for persistent logs
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger

# Usage
logger = setup_logging()

# Log with context
logger.info("API request received", extra={
    "endpoint": "/chat",
    "method": "POST",
    "user_id": "user_123",
    "ip_address": "192.168.1.1"
})
```

**Output:**
```json
{
  "timestamp": "2024-03-23T10:15:32.123456",
  "level": "INFO",
  "message": "API request received",
  "module": "main",
  "function": "chat",
  "line": 42,
  "endpoint": "/chat",
  "method": "POST",
  "user_id": "user_123",
  "ip_address": "192.168.1.1"
}
```

---

### Logging in FastAPI

```python
# main.py
from fastapi import FastAPI, Request
import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)
app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    """Log all HTTP requests"""
    start_time = time.time()

    # Log request
    logger.info("Request started", extra={
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host
    })

    # Process request
    response = await call_next(request)

    # Log response
    duration = time.time() - start_time
    logger.info("Request completed", extra={
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_seconds": round(duration, 3)
    })

    return response

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with detailed logging"""
    try:
        logger.info("Chat request received", extra={
            "message_length": len(request.message),
            "temperature": request.temperature,
            "model": "gpt-4"
        })

        # Process request
        response = await process_chat(request)

        logger.info("Chat response generated", extra={
            "tokens_used": response.tokens,
            "cost": response.cost
        })

        return response

    except Exception as e:
        logger.error("Chat request failed", extra={
            "error": str(e),
            "message": request.message[:100]
        }, exc_info=True)
        raise
```

---

## Metrics with Prometheus

### What is Prometheus?

Prometheus is the industry-standard metrics system:
- Time-series database
- Powerful query language (PromQL)
- Built-in alerting
- Integrates with Grafana

### Instrumenting Your Application

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

LLM_TOKENS_USED = Counter(
    'llm_tokens_used_total',
    'Total tokens used across all LLM calls',
    ['model', 'user_id']
)

LLM_COST = Counter(
    'llm_cost_total',
    'Total cost of LLM API calls in USD',
    ['model', 'user_id']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Number of active HTTP requests',
    ['endpoint']
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

# Middleware to track metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track metrics for all requests"""
    endpoint = request.url.path
    method = request.method

    # Track active requests
    ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()

    # Track duration
    start_time = time.time()

    try:
        response = await call_next(request)
        status = response.status_code

        # Record metrics
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()

        duration = time.time() - start_time
        REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

        return response

    finally:
        ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Expose metrics for Prometheus"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

# Usage in business logic
@app.post("/chat")
async def chat(request: ChatRequest, user_id: str):
    # Check cache
    if cached := await get_from_cache(request):
        CACHE_HITS.labels(cache_type="llm_response").inc()
        return cached

    CACHE_MISSES.labels(cache_type="llm_response").inc()

    # Call LLM
    response = await call_llm(request)

    # Track tokens and cost
    LLM_TOKENS_USED.labels(
        model="gpt-4",
        user_id=user_id
    ).inc(response.tokens)

    LLM_COST.labels(
        model="gpt-4",
        user_id=user_id
    ).inc(response.cost)

    return response
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'llm-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### Common Prometheus Queries (PromQL)

```promql
# Requests per second
rate(http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m])
/ rate(http_requests_total[5m])

# Total LLM cost per user (last hour)
sum by (user_id) (
  increase(llm_cost_total[1h])
)

# Cache hit rate
rate(cache_hits_total[5m])
/ (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))

# Active requests
http_requests_active
```

---

## Dashboards with Grafana

### Setting Up Grafana

```yaml
# docker-compose.yml addition
services:
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=redis-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  grafana_data:
```

### Dashboard Configuration

```yaml
# grafana/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

### Example Dashboard JSON

```json
{
  "dashboard": {
    "title": "LLM API Dashboard",
    "panels": [
      {
        "title": "Requests per Second",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Response Time (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "LLM Cost (24h)",
        "targets": [
          {
            "expr": "sum(increase(llm_cost_total[24h]))"
          }
        ],
        "type": "singlestat"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

### Key Dashboards to Create

**1. Overview Dashboard:**
- Total requests
- Success/error rates
- Response times (p50, p95, p99)
- Active users
- System health

**2. LLM Metrics Dashboard:**
- Tokens used (by model)
- Cost per model
- Cost per user
- Cache hit rate
- Average response time by model

**3. Business Metrics Dashboard:**
- Revenue (API calls × price)
- Active users
- Conversations created
- Messages per conversation
- User retention

**4. Infrastructure Dashboard:**
- CPU/Memory usage
- Database connections
- Redis cache size
- Network I/O
- Disk usage

---

## Distributed Tracing

### Why Tracing?

**Problem:** Request is slow, but why?
- Database query?
- LLM API call?
- Cache lookup?
- Network latency?

**Solution:** Distributed tracing shows the full request path

### Implementing OpenTelemetry

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

# Configure tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Instrument SQLAlchemy
SQLAlchemyInstrumentor().instrument(engine=engine)

# Custom tracing
@app.post("/chat")
async def chat(request: ChatRequest):
    # Automatic span from FastAPI instrumentation

    # Add custom span
    with tracer.start_as_current_span("llm_api_call") as span:
        span.set_attribute("model", "gpt-4")
        span.set_attribute("temperature", request.temperature)

        response = await call_llm(request)

        span.set_attribute("tokens_used", response.tokens)
        span.set_attribute("cost", response.cost)

    return response

async def call_llm(request):
    with tracer.start_as_current_span("openai_api"):
        # Call OpenAI
        result = await openai.ChatCompletion.create(...)

    with tracer.start_as_current_span("save_to_db"):
        # Save to database
        await save_message(result)

    return result
```

**Trace Output Example:**
```
chat [2.3s]
  ├─ authenticate [0.1s]
  ├─ check_cache [0.05s]
  ├─ llm_api_call [1.8s]
  │   ├─ openai_api [1.7s] ← Main bottleneck
  │   └─ save_to_db [0.1s]
  └─ cache_write [0.2s]
```

---

## Alerting

### Alerting Strategy

**Alert on:**
- High error rate (>1%)
- Slow responses (p95 > 5s)
- High costs ($100/hour)
- Service down
- Database connection failures
- High memory usage (>80%)

**Don't alert on:**
- Individual errors (too noisy)
- Minor latency spikes
- Expected maintenance

### Prometheus Alert Rules

```yaml
# alerts.yml
groups:
  - name: llm_api_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(http_requests_total{status=~"5.."}[5m])
          / rate(http_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # Slow responses
      - alert: SlowResponseTime
        expr: |
          histogram_quantile(0.95,
            rate(http_request_duration_seconds_bucket[5m])
          ) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "API responses are slow"
          description: "P95 latency is {{ $value }}s"

      # High LLM costs
      - alert: HighLLMCost
        expr: |
          rate(llm_cost_total[1h]) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM costs are high"
          description: "Spending ${{ $value }}/hour on LLM APIs"

      # Service down
      - alert: ServiceDown
        expr: up{job="llm-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API service is down"
          description: "Service has been down for 1 minute"

      # Database issues
      - alert: DatabaseConnectionsHigh
        expr: |
          pg_stat_database_numbackends > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database connections high"
          description: "{{ $value }} active connections"
```

### Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'

route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'slack-notifications'

  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true

    - match:
        severity: warning
      receiver: 'slack-notifications'

receivers:
  - name: 'slack-notifications'
    slack_configs:
      - channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

---

## Cost Monitoring

### Tracking LLM Costs

```python
# cost_tracker.py
from typing import Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

class CostTracker:
    """Track and analyze LLM API costs"""

    # Pricing (as of 2024)
    PRICING = {
        "gpt-4": {
            "input": 0.03 / 1000,   # $0.03 per 1K tokens
            "output": 0.06 / 1000    # $0.06 per 1K tokens
        },
        "gpt-3.5-turbo": {
            "input": 0.0015 / 1000,
            "output": 0.002 / 1000
        }
    }

    @staticmethod
    def calculate_cost(
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for a request"""
        pricing = CostTracker.PRICING.get(model, {"input": 0, "output": 0})

        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]

        return round(input_cost + output_cost, 6)

    @staticmethod
    async def get_user_costs(
        user_id: str,
        period: str = "today",
        db: Session = None
    ) -> Dict:
        """Get user's costs for a time period"""
        if period == "today":
            start = datetime.utcnow().replace(hour=0, minute=0, second=0)
        elif period == "week":
            start = datetime.utcnow() - timedelta(days=7)
        elif period == "month":
            start = datetime.utcnow() - timedelta(days=30)

        query = db.query(UsageLog).filter(
            UsageLog.user_id == user_id,
            UsageLog.created_at >= start
        )

        total_cost = sum(log.cost for log in query.all())
        total_tokens = sum(log.tokens for log in query.all())
        request_count = query.count()

        return {
            "total_cost": round(total_cost, 2),
            "total_tokens": total_tokens,
            "request_count": request_count,
            "avg_cost_per_request": round(total_cost / request_count, 4) if request_count else 0
        }

    @staticmethod
    async def check_quota(user_id: str, db: Session) -> bool:
        """Check if user has exceeded quota"""
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return False

        # Check daily costs
        today_costs = await CostTracker.get_user_costs(user_id, "today", db)

        if today_costs["total_cost"] > user.quota_limit:
            return False

        return True

# Usage in endpoint
@app.post("/chat")
async def chat(
    request: ChatRequest,
    user_id: str,
    db: Session = Depends(get_db)
):
    # Check quota before processing
    if not await CostTracker.check_quota(user_id, db):
        raise HTTPException(
            status_code=402,
            detail="Daily quota exceeded"
        )

    # Process request
    response = await call_llm(request)

    # Calculate and log cost
    cost = CostTracker.calculate_cost(
        model="gpt-4",
        input_tokens=request.tokens,
        output_tokens=response.tokens
    )

    # Save to database
    usage_log = UsageLog(
        user_id=user_id,
        tokens=request.tokens + response.tokens,
        cost=cost,
        model="gpt-4"
    )
    db.add(usage_log)
    db.commit()

    return response
```

### Cost Dashboard Queries

```promql
# Total cost (last 24 hours)
sum(increase(llm_cost_total[24h]))

# Cost by user (top 10)
topk(10, sum by (user_id) (
  increase(llm_cost_total[24h])
))

# Cost by model
sum by (model) (
  increase(llm_cost_total[24h])
)

# Projected monthly cost
sum(increase(llm_cost_total[24h])) * 30
```

---

## Performance Profiling

### Finding Bottlenecks

```python
# profiling.py
import cProfile
import pstats
import io
from functools import wraps

def profile(func):
    """Decorator to profile function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        result = await func(*args, **kwargs)

        profiler.disable()

        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

        return result
    return wrapper

# Usage
@profile
async def expensive_operation():
    # Your code here
    pass
```

---

## Complete Observability Stack

### Docker Compose with Full Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  # Application
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - prometheus
      - jaeger

  # Prometheus (Metrics)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alerts.yml:/etc/prometheus/alerts.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  # Grafana (Dashboards)
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus

  # Jaeger (Tracing)
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"  # UI
      - "14268:14268"
      - "9411:9411"

  # Alertmanager
  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
```

---

## Exercises

### Exercise 1: Add Structured Logging (2 hours)
1. Implement JSON logging
2. Add request/response logging
3. Test log searching

### Exercise 2: Setup Prometheus (2 hours)
1. Add metrics to your API
2. Configure Prometheus
3. Query metrics with PromQL

### Exercise 3: Create Grafana Dashboard (2 hours)
1. Setup Grafana
2. Create overview dashboard
3. Add cost tracking panels

### Exercise 4: Implement Alerting (2 hours)
1. Define alert rules
2. Configure Alertmanager
3. Test alerts

---

## Summary

### What You Learned
- ✅ Structured logging with JSON
- ✅ Prometheus metrics collection
- ✅ Grafana dashboard creation
- ✅ Distributed tracing
- ✅ Intelligent alerting
- ✅ Cost tracking and monitoring

### Key Takeaways
1. **Monitor everything** - You can't improve what you don't measure
2. **Structure your logs** - Makes debugging 10x easier
3. **Alert intelligently** - Too many alerts = alert fatigue
4. **Track costs** - LLM APIs can get expensive fast
5. **Use dashboards** - Visualize system health at a glance

### Next Lesson
**Lesson 4:** Security & Cost Optimization
- Authentication best practices
- Rate limiting
- PII detection
- Cost optimization strategies

---

**Time to complete:** 8-10 hours
**Difficulty:** Intermediate to Advanced
**Next lesson:** 04_security_cost_optimization.md

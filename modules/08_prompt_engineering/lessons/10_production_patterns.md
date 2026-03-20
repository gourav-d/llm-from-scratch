# Lesson 10: Production Patterns

**Build production-ready LLM applications with reliability, monitoring, and best practices**

---

## 🎯 Learning Objectives

After this lesson, you will:
- Design production-ready prompt systems
- Implement error handling and retries
- Monitor LLM applications effectively
- Optimize for cost and reliability
- Build scalable prompt architectures

**Time:** 90 minutes

---

## 🏗️ Production Architecture Patterns

### Pattern 1: Layered Architecture

```
┌─────────────────────────────────────┐
│        Application Layer            │ ← Business logic
├─────────────────────────────────────┤
│    Prompt Orchestration Layer       │ ← Template management, composition
├─────────────────────────────────────┤
│       LLM Gateway Layer             │ ← Provider abstraction, fallback
├─────────────────────────────────────┤
│    Monitoring & Logging Layer       │ ← Metrics, traces, alerts
├─────────────────────────────────────┤
│     Caching & Rate Limit Layer      │ ← Performance, cost optimization
└─────────────────────────────────────┘
```

**C#/.NET Equivalent:**
```csharp
// Like clean architecture layers
namespace Application.Core { }
namespace Application.Services { }
namespace Infrastructure.LLM { }
namespace Infrastructure.Monitoring { }
namespace Infrastructure.Caching { }
```

---

## 🔄 Error Handling & Retries

### Pattern 1: Exponential Backoff with Retries

```python
import time
import random
from typing import Callable, TypeVar, Optional

T = TypeVar('T')

class RetryConfig:
    """
    Configuration for retry behavior.

    C#/.NET: Like Polly retry policies
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""

        # Exponential backoff: delay = base_delay * (exponential_base ^ attempt)
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay


def retry_with_backoff(
    func: Callable[..., T],
    config: RetryConfig = RetryConfig(),
    retryable_exceptions: tuple = (Exception,)
) -> T:
    """
    Retry function with exponential backoff.

    C#/.NET equivalent (using Polly):
    await Policy
        .Handle<HttpRequestException>()
        .WaitAndRetryAsync(3, retryAttempt =>
            TimeSpan.FromSeconds(Math.Pow(2, retryAttempt)))
        .ExecuteAsync(async () => await httpClient.GetAsync(url));
    """

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return func()

        except retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_retries:
                # Final attempt failed
                raise

            # Calculate delay and wait
            delay = config.get_delay(attempt)
            print(f"Attempt {attempt + 1} failed. Retrying in {delay:.2f}s...")
            time.sleep(delay)

    # Should never reach here, but for type safety
    raise last_exception


# Usage Example
def call_llm_with_retry(prompt: str) -> str:
    """Call LLM with automatic retries."""

    config = RetryConfig(
        max_retries=3,
        base_delay=1.0,
        exponential_base=2.0,
        jitter=True
    )

    return retry_with_backoff(
        lambda: call_llm(prompt),
        config=config,
        retryable_exceptions=(TimeoutError, ConnectionError)
    )
```

### Pattern 2: Circuit Breaker

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """
    Prevent cascading failures by failing fast.

    C#/.NET: Like Polly circuit breaker
    https://github.com/App-vNext/Polly#circuit-breaker
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        """
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before trying again
            success_threshold: Successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout)
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

    def call(self, func: Callable[..., T]) -> T:
        """
        Execute function through circuit breaker.

        Raises:
            CircuitBreakerOpen: If circuit is open
        """

        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if datetime.now() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerOpen("Circuit breaker is OPEN")

        try:
            result = func()

            # Success
            self.on_success()
            return result

        except Exception as e:
            # Failure
            self.on_failure()
            raise

    def on_success(self):
        """Handle successful call."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                # Recovered!
                self.state = CircuitState.CLOSED
                self.success_count = 0

    def on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            # Open the circuit
            self.state = CircuitState.OPEN


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Usage
llm_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60
)

def safe_llm_call(prompt: str) -> str:
    """Call LLM through circuit breaker."""
    try:
        return llm_circuit_breaker.call(lambda: call_llm(prompt))
    except CircuitBreakerOpen:
        # Fallback response
        return "Service temporarily unavailable. Please try again later."
```

---

## 📊 Monitoring & Observability

### Pattern 1: Structured Logging

```python
import logging
import json
from typing import Dict, Any
from datetime import datetime

class PromptLogger:
    """
    Structured logging for LLM applications.

    C#/.NET: Like Serilog structured logging
    Log.Information("Processed {PromptId} in {Duration}ms", promptId, duration);
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)

    def log_request(
        self,
        prompt_id: str,
        user_id: str,
        prompt: str,
        metadata: Dict[str, Any] = None
    ):
        """Log incoming request."""

        log_data = {
            "event": "llm_request",
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "prompt_id": prompt_id,
            "user_id": user_id,
            "prompt_length": len(prompt),
            "metadata": metadata or {}
        }

        self.logger.info(json.dumps(log_data))

    def log_response(
        self,
        prompt_id: str,
        response: str,
        latency_ms: float,
        token_count: int,
        cost: float,
        success: bool
    ):
        """Log LLM response."""

        log_data = {
            "event": "llm_response",
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "prompt_id": prompt_id,
            "response_length": len(response),
            "latency_ms": latency_ms,
            "token_count": token_count,
            "cost_usd": cost,
            "success": success
        }

        self.logger.info(json.dumps(log_data))

    def log_error(
        self,
        prompt_id: str,
        error_type: str,
        error_message: str,
        stack_trace: str = None
    ):
        """Log errors."""

        log_data = {
            "event": "llm_error",
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "prompt_id": prompt_id,
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace
        }

        self.logger.error(json.dumps(log_data))
```

### Pattern 2: Metrics Collection

```python
from dataclasses import dataclass
from collections import defaultdict
import time

@dataclass
class LLMMetrics:
    """
    Metrics for LLM usage.

    C#/.NET: Like Application Insights metrics
    telemetryClient.TrackMetric("LLM_Latency", duration);
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        return self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def avg_cost(self) -> float:
        """Calculate average cost per request."""
        return self.total_cost / self.total_requests if self.total_requests > 0 else 0.0


class MetricsCollector:
    """Collect and aggregate LLM metrics."""

    def __init__(self):
        self.metrics_by_model = defaultdict(LLMMetrics)
        self.metrics_by_operation = defaultdict(LLMMetrics)

    def record_request(
        self,
        model: str,
        operation: str,
        success: bool,
        tokens: int,
        cost: float,
        latency_ms: float
    ):
        """Record metrics for a request."""

        for metrics in [self.metrics_by_model[model], self.metrics_by_operation[operation]]:
            metrics.total_requests += 1

            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1

            metrics.total_tokens += tokens
            metrics.total_cost += cost
            metrics.total_latency_ms += latency_ms

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""

        return {
            "by_model": {
                model: {
                    "requests": m.total_requests,
                    "success_rate": f"{m.success_rate:.2%}",
                    "avg_latency_ms": f"{m.avg_latency_ms:.0f}",
                    "avg_cost": f"${m.avg_cost:.4f}",
                    "total_cost": f"${m.total_cost:.2f}"
                }
                for model, m in self.metrics_by_model.items()
            },
            "by_operation": {
                op: {
                    "requests": m.total_requests,
                    "success_rate": f"{m.success_rate:.2%}",
                    "total_cost": f"${m.total_cost:.2f}"
                }
                for op, m in self.metrics_by_operation.items()
            }
        }


# Global metrics collector
metrics_collector = MetricsCollector()

def instrumented_llm_call(
    prompt: str,
    model: str = "gpt-4o-mini",
    operation: str = "general"
) -> str:
    """LLM call with metrics collection."""

    start_time = time.time()

    try:
        response = call_llm(prompt, model=model)
        latency_ms = (time.time() - start_time) * 1000

        # Calculate metrics
        tokens = count_tokens(prompt + response)
        cost = calculate_cost(model, tokens)

        # Record success
        metrics_collector.record_request(
            model=model,
            operation=operation,
            success=True,
            tokens=tokens,
            cost=cost,
            latency_ms=latency_ms
        )

        return response

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000

        # Record failure
        metrics_collector.record_request(
            model=model,
            operation=operation,
            success=False,
            tokens=0,
            cost=0.0,
            latency_ms=latency_ms
        )

        raise
```

---

## 💰 Cost Optimization Patterns

### Pattern 1: Tiered Model Selection

```python
class ModelSelector:
    """
    Select appropriate model based on task complexity.

    C#/.NET: Like strategy pattern for model selection
    """

    MODELS = {
        "simple": {
            "name": "gpt-4o-mini",
            "cost_per_1k": 0.00015,  # $0.15/1M tokens
            "max_complexity": 3
        },
        "standard": {
            "name": "gpt-4o",
            "cost_per_1k": 0.0025,  # $2.50/1M tokens
            "max_complexity": 7
        },
        "complex": {
            "name": "gpt-4o",
            "cost_per_1k": 0.03,  # (hypothetical premium)
            "max_complexity": 10
        }
    }

    @staticmethod
    def select_model(task_complexity: int, budget_tier: str = "standard") -> str:
        """
        Select model based on task complexity and budget.

        Args:
            task_complexity: 1-10 (simple to complex)
            budget_tier: "simple", "standard", or "complex"

        Returns:
            Model name
        """

        for tier, config in ModelSelector.MODELS.items():
            if task_complexity <= config["max_complexity"]:
                if budget_tier == "simple":
                    return ModelSelector.MODELS["simple"]["name"]
                else:
                    return config["name"]

        # Default to most capable for very complex tasks
        return ModelSelector.MODELS["complex"]["name"]


# Usage
def smart_llm_call(prompt: str, task_complexity: int) -> str:
    """Automatically select appropriate model."""

    model = ModelSelector.select_model(task_complexity)
    return call_llm(prompt, model=model)
```

### Pattern 2: Response Caching

```python
from functools import lru_cache
import hashlib
import json

class PromptCache:
    """
    Cache LLM responses for repeated prompts.

    C#/.NET: Like MemoryCache or distributed cache
    """

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model."""

        content = json.dumps({"prompt": prompt, "model": model}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response if available."""

        key = self.get_cache_key(prompt, model)

        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def set(self, prompt: str, model: str, response: str):
        """Cache response."""

        key = self.get_cache_key(prompt, model)

        # Simple LRU: If cache full, remove oldest
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = response

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# Global cache
prompt_cache = PromptCache(max_size=1000)

def cached_llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
    """LLM call with caching."""

    # Check cache
    cached = prompt_cache.get(prompt, model)
    if cached:
        print("💰 Cache hit! Saved $$$")
        return cached

    # Cache miss - call LLM
    response = call_llm(prompt, model=model)

    # Cache for next time
    prompt_cache.set(prompt, model, response)

    return response
```

---

## 🎯 Production-Ready LLM Gateway

### Complete LLM Gateway Implementation

```python
from typing import Optional, Dict, Any
import uuid

class LLMGateway:
    """
    Production-ready LLM gateway with all best practices.

    Features:
    - Multiple provider support (OpenAI, Anthropic, etc.)
    - Retry logic with exponential backoff
    - Circuit breaker
    - Caching
    - Monitoring & logging
    - Cost tracking
    - Rate limiting

    C#/.NET equivalent:
    public class LLMGateway : ILLMService
    {
        // All the same features in C#
    }
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize components
        self.logger = PromptLogger("llm-gateway")
        self.metrics = MetricsCollector()
        self.cache = PromptCache(max_size=config.get("cache_size", 1000))
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("circuit_breaker_threshold", 5),
            recovery_timeout=config.get("circuit_breaker_timeout", 60)
        )
        self.retry_config = RetryConfig(
            max_retries=config.get("max_retries", 3)
        )

    def call(
        self,
        prompt: str,
        model: str = None,
        operation: str = "general",
        user_id: str = None,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Call LLM with all production features.

        Args:
            prompt: The prompt to send
            model: Model to use (auto-selected if None)
            operation: Operation type for metrics
            user_id: User ID for logging
            use_cache: Whether to use cache
            **kwargs: Additional parameters

        Returns:
            LLM response

        Raises:
            LLMGatewayError: If all retries fail
        """

        # Generate request ID
        request_id = str(uuid.uuid4())

        # Log request
        self.logger.log_request(
            prompt_id=request_id,
            user_id=user_id or "anonymous",
            prompt=prompt,
            metadata={"operation": operation, "model": model}
        )

        # Auto-select model if not specified
        if model is None:
            task_complexity = self.estimate_complexity(prompt)
            model = ModelSelector.select_model(task_complexity)

        # Check cache
        if use_cache:
            cached = self.cache.get(prompt, model)
            if cached:
                self.logger.log_response(
                    prompt_id=request_id,
                    response=cached,
                    latency_ms=0,  # Instant from cache
                    token_count=0,
                    cost=0.0,
                    success=True
                )
                return cached

        # Call LLM with all safety features
        start_time = time.time()

        try:
            # Circuit breaker + retry logic
            response = self._call_with_safety(prompt, model, **kwargs)

            latency_ms = (time.time() - start_time) * 1000

            # Calculate cost
            tokens = count_tokens(prompt + response)
            cost = calculate_cost(model, tokens)

            # Cache successful response
            if use_cache:
                self.cache.set(prompt, model, response)

            # Log success
            self.logger.log_response(
                prompt_id=request_id,
                response=response,
                latency_ms=latency_ms,
                token_count=tokens,
                cost=cost,
                success=True
            )

            # Record metrics
            self.metrics.record_request(
                model=model,
                operation=operation,
                success=True,
                tokens=tokens,
                cost=cost,
                latency_ms=latency_ms
            )

            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Log error
            self.logger.log_error(
                prompt_id=request_id,
                error_type=type(e).__name__,
                error_message=str(e)
            )

            # Record failure
            self.metrics.record_request(
                model=model,
                operation=operation,
                success=False,
                tokens=0,
                cost=0.0,
                latency_ms=latency_ms
            )

            raise LLMGatewayError(f"LLM call failed: {str(e)}") from e

    def _call_with_safety(self, prompt: str, model: str, **kwargs) -> str:
        """Call LLM through circuit breaker and retry logic."""

        def make_call():
            return self.circuit_breaker.call(
                lambda: call_llm(prompt, model=model, **kwargs)
            )

        return retry_with_backoff(
            make_call,
            config=self.retry_config,
            retryable_exceptions=(TimeoutError, ConnectionError)
        )

    def estimate_complexity(self, prompt: str) -> int:
        """
        Estimate task complexity (1-10).

        Simple heuristic based on prompt characteristics.
        In production: Use ML model for better estimation.
        """

        # Simple heuristic
        length = len(prompt)
        if length < 100:
            return 2
        elif length < 500:
            return 5
        else:
            return 8

    def get_stats(self) -> Dict[str, Any]:
        """Get gateway statistics."""

        return {
            "cache_stats": {
                "hit_rate": f"{self.cache.hit_rate:.2%}",
                "size": len(self.cache.cache)
            },
            "metrics": self.metrics.get_summary(),
            "circuit_breaker_state": self.circuit_breaker.state.value
        }


class LLMGatewayError(Exception):
    """Exception raised by LLM Gateway."""
    pass
```

---

## ✅ Summary

### Production Checklist

- [x] **Error Handling**
  - Retries with exponential backoff
  - Circuit breaker for cascading failures
  - Graceful degradation
  - Meaningful error messages

- [x] **Monitoring**
  - Structured logging
  - Metrics collection (latency, cost, success rate)
  - Alerting on failures
  - Request tracing

- [x] **Cost Management**
  - Model selection based on complexity
  - Response caching
  - Cost tracking and budgets
  - Token optimization

- [x] **Performance**
  - Caching frequent requests
  - Parallel processing where possible
  - Connection pooling
  - Streaming for large responses

- [x] **Reliability**
  - Multiple provider support
  - Fallback mechanisms
  - Health checks
  - SLA monitoring

- [x] **Security**
  - Input validation
  - Output filtering
  - Rate limiting
  - Audit logging

---

## 📝 Practice Exercises

1. **Build LLM Gateway:**
   - Implement retry logic
   - Add circuit breaker
   - Include monitoring
   - Test failure scenarios

2. **Cost optimization:**
   - Implement caching
   - Add model selection
   - Track spending
   - Set budgets

3. **Monitoring dashboard:**
   - Collect metrics
   - Create visualizations
   - Set up alerts
   - Build reports

4. **Production deployment:**
   - Deploy to cloud
   - Configure monitoring
   - Set up CI/CD
   - Load testing

---

## 🎓 Congratulations!

You've completed Module 8: Prompt Engineering!

### What You've Learned

1. **Fundamentals** (Lessons 1-4)
   - Zero-shot and few-shot prompting
   - Prompt templates
   - Roles and system prompts

2. **Advanced Techniques** (Lessons 5-8)
   - Chain-of-thought
   - Tree of thoughts
   - Structured outputs
   - Prompt optimization

3. **Production** (Lessons 9-10)
   - Security best practices
   - Production patterns
   - Error handling
   - Monitoring and cost management

### Next Steps

1. **Apply to real projects:**
   - Build prompt library for your domain
   - Implement LLM gateway
   - Create production templates

2. **Continue learning:**
   - Module 9: RAG (Retrieval-Augmented Generation)
   - Module 10: LangChain
   - Module 11: Building AI Applications

3. **Keep practicing:**
   - Experiment with techniques
   - Optimize your prompts
   - Measure and improve

---

**You're now a prompt engineering expert! 🎉**

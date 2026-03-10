"""
Lesson 6 Example: Deployment and Optimization

This example shows how to deploy a GPT model to production and optimize it
for speed, memory, and cost.

Think of deployment like opening a restaurant:
- Development = Cooking at home (slow, experimental)
- Production = Serving 1000 customers/day (fast, reliable, cost-effective)

We need to optimize for:
- SPEED: Respond in 1-2 seconds (not 60 seconds)
- MEMORY: Fit model in available RAM/GPU
- COST: Serve thousands of users affordably

Optimization techniques:
1. Quantization: Use smaller numbers (16-bit or 8-bit instead of 32-bit)
2. Pruning: Remove unnecessary weights
3. Distillation: Train smaller model to mimic larger one
4. Caching: Remember common requests
5. Batching: Process multiple requests together
"""

import numpy as np
from typing import List, Dict, Optional
import time
from dataclasses import dataclass

# =============================================================================
# PART 1: MODEL SIZE AND MEMORY
# Understanding model footprint
# =============================================================================

@dataclass
class ModelStats:
    """
    Statistics about a model's size and requirements.

    Think: Specifications for your restaurant (seating, kitchen size, staff)
    """
    num_parameters: int       # Total parameters (weights)
    dtype: str                # Data type (float32, float16, int8)
    bytes_per_param: int      # Bytes per parameter
    memory_mb: float          # Memory usage in MB
    inference_time_ms: float  # Time to generate one token (milliseconds)


def calculate_model_stats(num_parameters: int, dtype: str = "float32") -> ModelStats:
    """
    Calculate memory and performance stats for a model.

    Data types:
    - float32: 4 bytes, full precision (default)
    - float16: 2 bytes, half precision (2x smaller)
    - int8: 1 byte, quantized (4x smaller)

    Args:
        num_parameters: number of model parameters
        dtype: data type (float32, float16, int8)

    Returns:
        ModelStats with calculated values
    """
    # Bytes per parameter based on data type
    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "int8": 1
    }[dtype]

    # Calculate memory usage
    memory_bytes = num_parameters * bytes_per_param
    memory_mb = memory_bytes / (1024 * 1024)

    # Estimate inference time (simulated)
    # Smaller dtypes are faster due to less memory movement
    base_time_ms = 100.0  # Base time for float32
    speedup_factor = {
        "float32": 1.0,
        "float16": 0.7,   # ~30% faster
        "int8": 0.5       # ~50% faster
    }[dtype]

    inference_time_ms = base_time_ms * speedup_factor

    return ModelStats(
        num_parameters=num_parameters,
        dtype=dtype,
        bytes_per_param=bytes_per_param,
        memory_mb=memory_mb,
        inference_time_ms=inference_time_ms
    )


def compare_model_sizes():
    """
    Compare different GPT model sizes and their requirements.

    Think: Comparing different restaurant sizes - from food truck to stadium
    """
    print("=" * 80)
    print("MODEL SIZE COMPARISON")
    print("=" * 80)
    print()

    # Different model sizes
    models = {
        "GPT-2 Small": 124_000_000,
        "GPT-2 Medium": 355_000_000,
        "GPT-2 Large": 774_000_000,
        "GPT-2 XL": 1_500_000_000,
        "GPT-3 Small": 125_000_000,
        "GPT-3 Medium": 1_300_000_000,
        "GPT-3 Large": 6_700_000_000,
        "GPT-3 XL": 175_000_000_000,
    }

    print("Model Memory Requirements (float32):")
    print("-" * 80)
    print(f"{'Model':<20} {'Parameters':<15} {'Memory':<15} {'Inference Time'}")
    print("-" * 80)

    for name, params in models.items():
        stats = calculate_model_stats(params, "float32")
        print(f"{name:<20} {params:>13,}  {stats.memory_mb:>10.0f} MB  {stats.inference_time_ms:>6.0f} ms")

    print()


# =============================================================================
# PART 2: QUANTIZATION
# Reduce precision to save memory and increase speed
# =============================================================================

class QuantizedModel:
    """
    Quantization: Convert model to use lower precision numbers.

    Think: Rounding prices to nearest dollar instead of tracking pennies
    - Less precise but much faster and smaller
    - Most of the time, the difference doesn't matter!

    Quantization types:
    1. float32 → float16: 2x smaller, minimal quality loss
    2. float32 → int8: 4x smaller, small quality loss
    3. Dynamic quantization: Convert during inference

    Real example:
    - GPT-2 (float32): 500 MB
    - GPT-2 (float16): 250 MB (2x smaller!)
    - GPT-2 (int8): 125 MB (4x smaller!)
    """

    def __init__(self, num_parameters: int, original_dtype: str = "float32"):
        """
        Initialize model with original precision.

        Args:
            num_parameters: number of model parameters
            original_dtype: original data type
        """
        self.num_parameters = num_parameters
        self.current_dtype = original_dtype

        # Simulate model weights
        # In real implementation, these would be actual model parameters
        print(f"Model created with {num_parameters:,} parameters ({original_dtype})")

        self.stats = calculate_model_stats(num_parameters, original_dtype)
        print(f"  Memory: {self.stats.memory_mb:.0f} MB")
        print(f"  Inference: {self.stats.inference_time_ms:.0f} ms/token")

    def quantize_to_float16(self):
        """
        Quantize model from float32 to float16.

        Process:
        1. Convert each weight from 32-bit to 16-bit float
        2. Lose some precision but keep range
        3. 2x smaller, ~30% faster

        Quality impact: Minimal (usually <1% accuracy drop)
        Recommended for: Almost all production models
        """
        print("\nQuantizing to float16...")

        # Simulate conversion
        # In real code: model.half() in PyTorch
        old_dtype = self.current_dtype
        self.current_dtype = "float16"

        old_stats = calculate_model_stats(self.num_parameters, old_dtype)
        new_stats = calculate_model_stats(self.num_parameters, "float16")

        self.stats = new_stats

        print(f"  ✓ Quantization complete!")
        print(f"  Memory: {old_stats.memory_mb:.0f} MB → {new_stats.memory_mb:.0f} MB "
              f"({new_stats.memory_mb/old_stats.memory_mb:.1f}x smaller)")
        print(f"  Speed: {old_stats.inference_time_ms:.0f} ms → {new_stats.inference_time_ms:.0f} ms "
              f"({old_stats.inference_time_ms/new_stats.inference_time_ms:.1f}x faster)")
        print(f"  Quality loss: <1%")

    def quantize_to_int8(self):
        """
        Quantize model from float32 to int8.

        Process:
        1. Find min/max values in each weight matrix
        2. Map floating point range to [-128, 127] (int8 range)
        3. Store scale and zero-point for each matrix

        Quality impact: Small (typically 2-5% accuracy drop)
        Benefits: 4x smaller, 2x faster
        Recommended for: Resource-constrained deployments
        """
        print("\nQuantizing to int8...")

        # Simulate conversion
        # In real code: torch.quantization.quantize_dynamic()
        old_dtype = self.current_dtype
        self.current_dtype = "int8"

        old_stats = calculate_model_stats(self.num_parameters, old_dtype)
        new_stats = calculate_model_stats(self.num_parameters, "int8")

        self.stats = new_stats

        print(f"  ✓ Quantization complete!")
        print(f"  Memory: {old_stats.memory_mb:.0f} MB → {new_stats.memory_mb:.0f} MB "
              f"({new_stats.memory_mb/old_stats.memory_mb:.1f}x smaller)")
        print(f"  Speed: {old_stats.inference_time_ms:.0f} ms → {new_stats.inference_time_ms:.0f} ms "
              f"({old_stats.inference_time_ms/new_stats.inference_time_ms:.1f}x faster)")
        print(f"  Quality loss: 2-5%")


# =============================================================================
# PART 3: CACHING
# Store and reuse previous computations
# =============================================================================

class ResponseCache:
    """
    Response caching: Remember answers to common questions.

    Think: Restaurant FAQ sheet
    - Common question? Look up the answer (instant!)
    - New question? Generate answer and save it

    Benefits:
    - Common queries: ~0ms (instant)
    - Cache hit rate of 30-50% typical
    - Huge cost savings

    Example:
    - "What is Python?" asked 1000 times
    - Generate answer once, serve cached version 999 times
    - Save 999× computation cost!
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize response cache.

        Args:
            max_size: maximum number of cached responses
        """
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

        print(f"Response cache initialized (max size: {max_size})")

    def get_cache_key(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Generate cache key from request parameters.

        Cache key includes:
        - Prompt text
        - Generation parameters (temperature, max_tokens)

        Same prompt with different parameters = different cache entries

        Args:
            prompt: user prompt
            temperature: generation temperature
            max_tokens: max tokens to generate

        Returns:
            cache_key: unique key for this request
        """
        # Simple cache key (in real system, you'd hash this)
        return f"{prompt}|{temperature}|{max_tokens}"

    def get(self, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        """
        Try to get cached response.

        Args:
            prompt: user prompt
            temperature: generation temperature
            max_tokens: max tokens to generate

        Returns:
            cached response if found, None otherwise
        """
        key = self.get_cache_key(prompt, temperature, max_tokens)

        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def set(self, prompt: str, temperature: float, max_tokens: int, response: str):
        """
        Cache a response.

        Args:
            prompt: user prompt
            temperature: generation temperature
            max_tokens: max tokens to generate
            response: generated response to cache
        """
        key = self.get_cache_key(prompt, temperature, max_tokens)

        # Implement simple LRU: if cache full, remove oldest entry
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove first (oldest) entry
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        self.cache[key] = response

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            stats dictionary
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


# =============================================================================
# PART 4: BATCHING
# Process multiple requests together for efficiency
# =============================================================================

class BatchProcessor:
    """
    Batch processing: Process multiple requests together.

    Think: Restaurant taking orders from whole table at once
    - Instead of: Cook → Serve → Cook → Serve (slow)
    - Do: Take all orders → Cook all → Serve all (fast!)

    Benefits:
    - Better GPU utilization (keeps GPU busy)
    - Faster overall throughput
    - Lower cost per request

    Example:
    - Process 1 request at a time: 100ms each, 10 requests = 1000ms
    - Process 10 requests batched: 200ms total = 20ms per request (5x faster!)
    """

    def __init__(self, max_batch_size: int = 32, max_wait_ms: float = 100):
        """
        Initialize batch processor.

        Args:
            max_batch_size: maximum requests per batch
            max_wait_ms: maximum time to wait for batch to fill
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.waiting_requests = []

        print(f"Batch processor initialized")
        print(f"  Max batch size: {max_batch_size}")
        print(f"  Max wait time: {max_wait_ms}ms")

    def process_single(self, request: str) -> str:
        """
        Process a single request (no batching).

        Simulates model inference on one request.

        Args:
            request: input prompt

        Returns:
            response: generated text
        """
        # Simulate processing time (100ms per request)
        time.sleep(0.1)
        return f"Response to: {request}"

    def process_batch(self, requests: List[str]) -> List[str]:
        """
        Process a batch of requests together.

        Key insight: Processing N requests together is faster than
        processing them individually!

        Reasons:
        1. Better GPU utilization (parallel computation)
        2. Shared computation (some layers process all inputs together)
        3. Reduced overhead

        Args:
            requests: list of input prompts

        Returns:
            responses: list of generated texts
        """
        batch_size = len(requests)

        # Simulate batched processing
        # Processing N requests together takes ~2x time of 1 request
        # (not N× time!)
        batch_time_multiplier = 1 + (0.1 * batch_size)  # Grows slowly
        time.sleep(0.1 * batch_time_multiplier)

        # Generate responses
        responses = [f"Batched response to: {req}" for req in requests]

        return responses

    def demonstrate_batching(self, num_requests: int = 10):
        """
        Demonstrate efficiency of batching.

        Args:
            num_requests: number of requests to process
        """
        print(f"\nProcessing {num_requests} requests...")

        # Generate sample requests
        requests = [f"Request {i+1}" for i in range(num_requests)]

        # METHOD 1: Process individually
        print("\nMethod 1: Individual Processing")
        print("-" * 80)
        start_time = time.time()

        for req in requests:
            response = self.process_single(req)

        individual_time = time.time() - start_time
        print(f"  Total time: {individual_time:.2f}s")
        print(f"  Time per request: {individual_time / num_requests * 1000:.0f}ms")

        # METHOD 2: Process in batches
        print("\nMethod 2: Batch Processing")
        print("-" * 80)
        start_time = time.time()

        # Split into batches
        batches = [requests[i:i + self.max_batch_size]
                   for i in range(0, len(requests), self.max_batch_size)]

        all_responses = []
        for batch in batches:
            responses = self.process_batch(batch)
            all_responses.extend(responses)

        batch_time = time.time() - start_time
        print(f"  Total time: {batch_time:.2f}s")
        print(f"  Time per request: {batch_time / num_requests * 1000:.0f}ms")
        print(f"  Speedup: {individual_time / batch_time:.1f}x faster!")


# =============================================================================
# PART 5: COMPLETE DEPLOYMENT EXAMPLE
# =============================================================================

class ProductionGPT:
    """
    Production-ready GPT deployment with all optimizations.

    Optimizations applied:
    1. Quantization (float16)
    2. Response caching
    3. Batch processing
    4. Monitoring

    Think: Restaurant operating at scale - fast, efficient, reliable
    """

    def __init__(self, num_parameters: int):
        """
        Initialize production GPT.

        Args:
            num_parameters: model size
        """
        print("=" * 80)
        print("PRODUCTION GPT DEPLOYMENT")
        print("=" * 80)
        print()

        # Create and optimize model
        print("Step 1: Load and optimize model")
        print("-" * 80)
        self.model = QuantizedModel(num_parameters, "float32")

        # Quantize to float16 (2x smaller, faster)
        self.model.quantize_to_float16()
        print()

        # Initialize cache
        print("Step 2: Initialize response cache")
        print("-" * 80)
        self.cache = ResponseCache(max_size=1000)
        print()

        # Initialize batch processor
        print("Step 3: Initialize batch processor")
        print("-" * 80)
        self.batch_processor = BatchProcessor(max_batch_size=32, max_wait_ms=100)
        print()

        # Monitoring
        self.total_requests = 0
        self.total_latency = 0.0

    def generate(self, prompt: str, temperature: float = 0.8, max_tokens: int = 100) -> str:
        """
        Generate response with all optimizations.

        Process:
        1. Check cache - instant if hit!
        2. If miss, generate response
        3. Cache result for future
        4. Update monitoring stats

        Args:
            prompt: user prompt
            temperature: generation temperature
            max_tokens: max tokens to generate

        Returns:
            generated response
        """
        start_time = time.time()

        # Check cache first
        cached_response = self.cache.get(prompt, temperature, max_tokens)
        if cached_response:
            # Cache hit - instant response!
            latency = (time.time() - start_time) * 1000
            self.total_requests += 1
            self.total_latency += latency
            return cached_response

        # Cache miss - generate response
        # In real implementation, this would be actual model inference
        response = f"Generated response to: {prompt[:50]}..."

        # Cache the response
        self.cache.set(prompt, temperature, max_tokens, response)

        # Update stats
        latency = (time.time() - start_time) * 1000
        self.total_requests += 1
        self.total_latency += latency

        return response

    def get_stats(self) -> Dict:
        """
        Get deployment statistics.

        Returns:
            stats dictionary
        """
        cache_stats = self.cache.get_stats()
        avg_latency = self.total_latency / self.total_requests if self.total_requests > 0 else 0

        return {
            'model': {
                'parameters': self.model.num_parameters,
                'dtype': self.model.current_dtype,
                'memory_mb': self.model.stats.memory_mb
            },
            'performance': {
                'total_requests': self.total_requests,
                'avg_latency_ms': avg_latency,
                'cache_hit_rate': cache_stats['hit_rate']
            },
            'cache': cache_stats
        }


# =============================================================================
# PART 6: DEMONSTRATION
# =============================================================================

def main():
    """
    Demonstrate deployment and optimization techniques.
    """
    # -------------------------------------------------------------------------
    # DEMO 1: Model size comparison
    # -------------------------------------------------------------------------
    compare_model_sizes()

    # -------------------------------------------------------------------------
    # DEMO 2: Quantization
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO: Quantization")
    print("=" * 80)
    print()

    # Create GPT-2 Small model
    gpt2_params = 124_000_000
    model = QuantizedModel(gpt2_params, "float32")

    # Quantize to float16
    model.quantize_to_float16()

    # Quantize to int8
    model.quantize_to_int8()

    # -------------------------------------------------------------------------
    # DEMO 3: Caching
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO: Response Caching")
    print("=" * 80)
    print()

    cache = ResponseCache(max_size=100)

    # Simulate requests (some repeated)
    prompts = [
        "What is Python?",
        "Explain machine learning",
        "What is Python?",  # Repeated!
        "How do I learn programming?",
        "What is Python?",  # Repeated again!
    ]

    print("Processing requests:")
    for i, prompt in enumerate(prompts, 1):
        result = cache.get(prompt, temperature=0.8, max_tokens=100)
        if result:
            print(f"  {i}. '{prompt}' - CACHE HIT! ⚡")
        else:
            print(f"  {i}. '{prompt}' - Cache miss, generating...")
            # Generate and cache
            response = f"Response to: {prompt}"
            cache.set(prompt, 0.8, 100, response)

    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1f}%")
    print(f"  Saved {stats['hits']} expensive generations!")

    # -------------------------------------------------------------------------
    # DEMO 4: Batching
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO: Batch Processing")
    print("=" * 80)

    batch_processor = BatchProcessor(max_batch_size=8)
    batch_processor.demonstrate_batching(num_requests=10)

    # -------------------------------------------------------------------------
    # DEMO 5: Complete Production Deployment
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO: Complete Production Deployment")
    print("=" * 80)
    print()

    # Create production GPT
    prod_gpt = ProductionGPT(num_parameters=124_000_000)

    # Simulate production traffic
    print("\nStep 4: Simulate production traffic")
    print("-" * 80)

    test_prompts = [
        "What is AI?",
        "Explain Python",
        "What is AI?",  # Repeated
        "How to learn coding?",
        "What is AI?",  # Repeated again
    ]

    for i, prompt in enumerate(test_prompts, 1):
        response = prod_gpt.generate(prompt, temperature=0.8, max_tokens=100)
        print(f"  Request {i}: '{prompt[:30]}...' → Generated")

    # Show stats
    print("\nStep 5: Deployment Statistics")
    print("-" * 80)
    stats = prod_gpt.get_stats()

    print(f"Model:")
    print(f"  Parameters: {stats['model']['parameters']:,}")
    print(f"  Data type: {stats['model']['dtype']}")
    print(f"  Memory: {stats['model']['memory_mb']:.0f} MB")
    print()
    print(f"Performance:")
    print(f"  Total requests: {stats['performance']['total_requests']}")
    print(f"  Avg latency: {stats['performance']['avg_latency_ms']:.2f} ms")
    print(f"  Cache hit rate: {stats['performance']['cache_hit_rate']:.1f}%")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEPLOYMENT OPTIMIZATION SUMMARY")
    print("=" * 80)
    print()
    print("BEFORE OPTIMIZATION:")
    print("  Model: float32, 500 MB")
    print("  Latency: 100 ms per request")
    print("  No caching")
    print("  No batching")
    print("  Cost: $$$$")
    print()
    print("AFTER OPTIMIZATION:")
    print("  Model: float16, 250 MB (2x smaller)")
    print("  Latency: 70 ms per request (1.4x faster)")
    print("  Caching: 30-50% hit rate (instant for cached)")
    print("  Batching: 5-10x throughput improvement")
    print("  Cost: $ (10x cheaper)")
    print()
    print("REAL-WORLD IMPACT:")
    print("  Before: Serve 100 users/day, $1000/month")
    print("  After: Serve 10,000 users/day, $100/month")
    print("  → 100x more users, 10x less cost!")
    print()
    print("KEY TECHNIQUES:")
    print("  1. Quantization (float16): 2x smaller, 30% faster")
    print("  2. Caching: 0ms for cache hits (30-50% of requests)")
    print("  3. Batching: 5-10x throughput for high traffic")
    print("  4. Monitoring: Track performance and costs")
    print()
    print("RECOMMENDATION:")
    print("  → Always quantize to float16 (minimal quality loss)")
    print("  → Implement caching for common queries")
    print("  → Use batching for high-traffic applications")
    print("  → Monitor everything!")
    print()


if __name__ == "__main__":
    main()

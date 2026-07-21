"""
Module 06 - Training & Fine-Tuning
Exercise 06: Deployment and Optimization

GLOSSARY
--------
Quantization      : Reducing the number of bits used to store each weight.
                    float32 (4 bytes) -> int8 (1 byte) = 4x smaller model.
                    Trade-off: slightly lower accuracy for much less memory/speed.
INT8              : 8-bit integer quantization. Values stored in range [-127, 127].
                    Formula: quantized = round(weight / scale)
                    scale = max(abs(weights)) / 127
float16           : 16-bit float. Same precision as float32 but half the memory.
                    Standard for inference; most GPUs handle it natively.
Model Memory (MB) : num_params * bytes_per_param / 1_000_000
Inference Latency : Time to generate one token (milliseconds).
Batching          : Processing multiple requests together in one forward pass.
                    More efficient use of GPU but adds wait time.
Cache Hit Rate    : Fraction of requests served from cache (no model call needed).
                    High cache hit rate = less GPU compute = lower cost.
Throughput        : Tokens generated per second. Higher = better.
                    throughput = num_tokens / latency_seconds
Pruning           : Removing weights that are close to zero.
                    Reduces model size; must be done carefully.
Quantization Error: The difference between the original and quantized weight.
                    Acceptable if small; too large degrades model quality.
"""

import numpy as np   # NumPy for math

print("=" * 60)
print("Exercise 06: Deployment and Optimization")
print("=" * 60)
print()


# ============================================================
#  EXERCISE 1
#  Topic: Model Memory by Data Type
#
#  Background:
#    Model memory depends on two things:
#      1. Number of parameters
#      2. Bytes per parameter (dtype)
#
#    Common dtypes:
#      float32 -> 4 bytes  (training default)
#      float16 -> 2 bytes  (inference, 2x memory saving)
#      int8    -> 1 byte   (quantized, 4x memory saving)
#
#    Formula:
#      memory_mb = num_params * bytes_per_param / 1_000_000
#      memory_gb = memory_mb / 1_000
#
#  Your Task:
#    Write: memory_by_dtype(num_params) -> dict
#    Returns {"float32": float, "float16": float, "int8": float}
#    with memory in MB for each dtype.
#
#  C# Analogy:
#    Like computing file sizes for the same data in different encodings:
#      UTF-8 (variable) vs UTF-16 (2 bytes each) vs ASCII (1 byte each).
# ============================================================

print("-" * 50)
print("EXERCISE 1: Model Memory by Data Type")
print("-" * 50)
print()

BYTES_PER_DTYPE = {"float32": 4, "float16": 2, "int8": 1}


def memory_by_dtype(num_params):
    """
    Calculate model memory usage in MB for each data type.

    Parameters:
        num_params (int): Total number of model parameters.

    Returns:
        dict: {"float32": float, "float16": float, "int8": float}
              Memory in MB for each data type.
    """
    # TODO:
    # For each dtype in BYTES_PER_DTYPE:
    #   memory_mb = num_params * bytes / 1_000_000
    # Return all three values in a dict.
    pass  # Replace with your implementation


models = [
    ("GPT-2 small",  124_000_000),
    ("GPT-2 large",  774_000_000),
    ("GPT-3",      175_000_000_000),
]

print(f"  {'Model':<15}  {'float32 MB':>12}  {'float16 MB':>12}  {'int8 MB':>10}")
print("  " + "-" * 58)
for name, params in models:
    r = memory_by_dtype(params)
    if r:
        print(f"  {name:<15}  {r['float32']:>12,.0f}  {r['float16']:>12,.0f}  {r['int8']:>10,.0f}")
print()
print("  Expected (GPT-2 small): float32~496 MB, float16~248 MB, int8~124 MB")
print()


# ============================================================
#  EXERCISE 2
#  Topic: INT8 Quantization
#
#  Background:
#    INT8 quantization converts float32 weights to 8-bit integers.
#    This cuts memory by 4x with minimal quality loss.
#
#    Steps:
#      1. scale     = max(abs(weights)) / 127
#         (maps the largest weight to ±127, INT8 range)
#      2. quantized = round(weights / scale)
#         (scaled integers, clipped to [-127, 127])
#      3. dequantized = quantized * scale
#         (recovered float values — slightly different from original)
#      4. error    = max(abs(dequantized - weights))
#         (maximum quantization error)
#
#  Your Task:
#    Write: quantize_int8(weights) -> dict
#    Returns: {"scale": float, "quantized": np.ndarray,
#              "dequantized": np.ndarray, "max_error": float}
#
#  C# Analogy:
#    Like normalising floats to [-127, 127] byte range:
#      scale = maxAbs / 127f
#      quantized = (sbyte)Math.Round(w / scale)
#      dequantized = quantized * scale
# ============================================================

print("-" * 50)
print("EXERCISE 2: INT8 Quantization")
print("-" * 50)
print()


def quantize_int8(weights):
    """
    Quantize float32 weights to INT8 and back (simulate quantization).

    Parameters:
        weights (np.ndarray): Float32 weight array.

    Returns:
        dict: {
            "scale"       : float      -- quantization scale factor,
            "quantized"   : np.ndarray -- integer weights (dtype int8 range),
            "dequantized" : np.ndarray -- recovered float weights,
            "max_error"   : float      -- maximum reconstruction error
        }
    """
    # TODO:
    # scale       = np.max(np.abs(weights)) / 127.0
    # quantized   = np.round(weights / scale).astype(np.int8)
    # dequantized = quantized.astype(np.float32) * scale
    # max_error   = np.max(np.abs(dequantized - weights))
    pass  # Replace with your implementation


np.random.seed(7)
sample_weights = np.random.randn(5, 5).astype(np.float32)   # Typical weight matrix

r = quantize_int8(sample_weights)
if r:
    print(f"  Scale factor : {r['scale']:.6f}")
    print()
    print("  Original  (first row):", np.round(sample_weights[0], 4))
    print("  Quantized (first row):", r['quantized'][0])
    print("  Recovered (first row):", np.round(r['dequantized'][0], 4))
    print()
    print(f"  Max quantization error: {r['max_error']:.6f}")
    print(f"  Error acceptable (< scale/2): {r['max_error'] < r['scale'] / 2}")
print()


# ============================================================
#  EXERCISE 3
#  Topic: Throughput and Latency
#
#  Background:
#    Two key deployment metrics:
#
#    Latency   = time to generate ONE complete response (seconds or ms)
#    Throughput = tokens generated per SECOND
#
#    For a single request:
#      throughput_tokens_per_sec = num_tokens_generated / latency_seconds
#
#    With batching (B requests at once):
#      total_tokens  = batch_size * tokens_per_response
#      throughput_batch = total_tokens / latency_seconds
#      (GPU processes multiple requests in parallel -> higher throughput)
#
#  Your Task:
#    Write: compute_throughput(num_tokens, latency_seconds) -> float
#    Returns tokens per second.
#
#    Write: batching_throughput(batch_size, tokens_per_response,
#                               latency_seconds) -> dict
#    Returns {"total_tokens": int, "throughput": float}
#
#  C# Analogy:
#    throughput = tokensProcessed / stopwatch.Elapsed.TotalSeconds;
# ============================================================

print("-" * 50)
print("EXERCISE 3: Throughput and Latency")
print("-" * 50)
print()


def compute_throughput(num_tokens, latency_seconds):
    """
    Compute token throughput in tokens per second.

    Parameters:
        num_tokens       (int)  : Number of tokens generated.
        latency_seconds  (float): Time taken to generate them.

    Returns:
        float: Tokens per second.
    """
    # TODO: return num_tokens / latency_seconds
    pass  # Replace with your implementation


def batching_throughput(batch_size, tokens_per_response, latency_seconds):
    """
    Compute throughput when processing a batch of requests.

    Parameters:
        batch_size          (int)  : Number of parallel requests.
        tokens_per_response (int)  : Tokens in each response.
        latency_seconds     (float): Time to process the whole batch.

    Returns:
        dict: {"total_tokens": int, "throughput": float}
    """
    # TODO:
    # total_tokens = batch_size * tokens_per_response
    # throughput   = total_tokens / latency_seconds
    pass  # Replace with your implementation


# Single request: 200 tokens in 2.0 seconds
single_tps = compute_throughput(200, 2.0)
if single_tps is not None:
    print(f"  Single request: 200 tokens / 2.0s = {single_tps:.1f} tokens/sec")

# Batching: 8 requests × 200 tokens, GPU takes 3.0 seconds total
batch_r = batching_throughput(8, 200, 3.0)
if batch_r:
    print(f"  Batch of 8   : {batch_r['total_tokens']} tokens / 3.0s = {batch_r['throughput']:.1f} tokens/sec")
    print(f"  Speedup from batching: {batch_r['throughput'] / single_tps:.1f}x")
print()
print("  Expected: single ~100 t/s, batched ~533 t/s (~5x improvement)")
print()


# ============================================================
#  EXERCISE 4
#  Topic: Cache Hit Rate and Cost Savings
#
#  Background:
#    Many LLM requests ask the same or similar things.
#    A cache stores previous (prompt -> response) pairs.
#    If the same prompt arrives again, return cached response instantly.
#    No model call = no GPU cost.
#
#    Cache hit rate = hits / total_requests
#    Cost savings % = 100 * hit_rate
#    (Each cache hit saves 100% of the compute cost for that request)
#
#    Example:
#      1000 requests, 600 served from cache:
#      hit_rate = 0.6 (60%)
#      Saved 60% of GPU compute vs serving all requests live.
#
#  Your Task:
#    Write: cache_analysis(total_requests, cache_hits, cost_per_request) -> dict
#    Returns: {"hit_rate": float, "savings_pct": float,
#              "total_cost": float, "cached_cost": float, "saved_cost": float}
#    total_cost  = total_requests * cost_per_request  (no cache)
#    cached_cost = (total_requests - cache_hits) * cost_per_request  (with cache)
#    saved_cost  = total_cost - cached_cost
#
#  C# Analogy:
#    Like tracking IMemoryCache hit rates in ASP.NET:
#      hitRate = cacheHits / (double)totalRequests;
# ============================================================

print("-" * 50)
print("EXERCISE 4: Cache Hit Rate and Cost Savings")
print("-" * 50)
print()


def cache_analysis(total_requests, cache_hits, cost_per_request):
    """
    Analyse the efficiency and cost savings of response caching.

    Parameters:
        total_requests   (int)  : Total number of API requests received.
        cache_hits       (int)  : Requests served from cache.
        cost_per_request (float): Cost (dollars) to serve one request live.

    Returns:
        dict: {
            "hit_rate"    : float -- fraction of requests from cache,
            "savings_pct" : float -- % of compute cost saved,
            "total_cost"  : float -- cost without any cache,
            "cached_cost" : float -- cost with cache,
            "saved_cost"  : float -- money saved by caching
        }
    """
    # TODO:
    # hit_rate    = cache_hits / total_requests
    # savings_pct = 100 * hit_rate
    # total_cost  = total_requests * cost_per_request
    # cached_cost = (total_requests - cache_hits) * cost_per_request
    # saved_cost  = total_cost - cached_cost
    pass  # Replace with your implementation


r = cache_analysis(
    total_requests   = 10_000,
    cache_hits       = 6_000,
    cost_per_request = 0.002    # $0.002 per request
)

if r:
    print(f"  Requests    : 10,000 total, 6,000 from cache")
    print(f"  Hit rate    : {r['hit_rate']:.1%}")
    print(f"  Savings     : {r['savings_pct']:.1f}%")
    print(f"  Cost without cache: ${r['total_cost']:.2f}")
    print(f"  Cost with cache   : ${r['cached_cost']:.2f}")
    print(f"  Amount saved      : ${r['saved_cost']:.2f}")
print()
print("  Expected: hit_rate=60%, saved=$12.00")
print()

print("=" * 60)
print("All exercises complete!")
print("=" * 60)

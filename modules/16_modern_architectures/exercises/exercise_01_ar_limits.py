"""
Exercise 01: Auto-Regressive Generation Limits
Module 16: Modern LLM Architectures

TASKS:
  1. Implement count_ar_passes() -- count forward passes needed for N tokens
  2. Implement diffusion_passes() -- count passes for diffusion with T steps
  3. Implement speedup_ratio()   -- compute diffusion vs AR speedup
  4. Implement has_right_context() -- check if AR can use right-side context

Run:  python exercise_01_ar_limits.py
Deps: none (pure Python)
"""


# ─────────────────────────────────────────────────────────
# TASK 1: Count AR Forward Passes
# ─────────────────────────────────────────────────────────

def count_ar_passes(n_tokens_to_generate):
    """
    Count the number of forward passes needed to generate n_tokens_to_generate
    tokens using auto-regressive (AR) generation.

    In AR generation, each token requires exactly ONE forward pass through the model.
    Passes are sequential -- step i must finish before step i+1 can start.

    Args:
        n_tokens_to_generate: how many new tokens to produce

    Returns:
        int: total number of forward passes required

    Example:
        count_ar_passes(100) -> 100
        count_ar_passes(1)   -> 1
        count_ar_passes(0)   -> 0

    HINT: It's simpler than you think -- one pass per token.
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 2: Count Diffusion Forward Passes
# ─────────────────────────────────────────────────────────

def diffusion_passes(n_tokens, T_steps):
    """
    Count the number of forward passes needed to generate n_tokens
    using diffusion generation with T_steps denoising steps.

    In diffusion generation:
      - All tokens are generated simultaneously at each step
      - Total passes = T_steps (regardless of how many tokens)
      - Each pass processes all n_tokens in parallel

    Args:
        n_tokens: number of tokens to generate
        T_steps:  number of denoising timesteps

    Returns:
        int: total number of forward passes required

    Example:
        diffusion_passes(100, 10) -> 10
        diffusion_passes(4096, 50) -> 50

    HINT: diffusion needs T_steps passes, not n_tokens passes.
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 3: Compute Speedup
# ─────────────────────────────────────────────────────────

def speedup_ratio(n_tokens, T_steps):
    """
    Compute the speedup of diffusion over AR generation.
    Speedup = AR_passes / diffusion_passes

    Args:
        n_tokens: number of tokens to generate
        T_steps:  number of diffusion timesteps

    Returns:
        float: speedup factor (how many times fewer passes diffusion needs)

    Example:
        speedup_ratio(100, 10)  -> 10.0  (10x fewer passes)
        speedup_ratio(4096, 50) -> 81.92 (82x fewer passes)
        speedup_ratio(50, 50)   -> 1.0   (no speedup -- T = N)

    HINT: divide count_ar_passes() by diffusion_passes()
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TASK 4: Right Context Availability
# ─────────────────────────────────────────────────────────

def has_right_context(model_type, position, sentence_length):
    """
    Determine whether a model can use right-side context (tokens after 'position')
    when predicting the token at 'position'.

    AR models (causal mask): can ONLY see tokens to the LEFT (positions 0..position-1).
    Bidirectional models (BERT, diffusion): can see ALL tokens.

    Args:
        model_type:      "ar" or "bidirectional"
        position:        the token position being predicted (0-indexed)
        sentence_length: total number of tokens in the sentence

    Returns:
        bool: True if the model can use right context, False otherwise

    Example:
        has_right_context("ar", 3, 10)            -> False
        has_right_context("bidirectional", 3, 10) -> True
        has_right_context("ar", 9, 10)            -> False  (last token, nothing to right)
        has_right_context("bidirectional", 0, 10) -> True

    HINT: AR = causal mask = no right context.
          Bidirectional = sees everything.
          Edge case: if position is the last token, even bidirectional has no right context.
    """
    # TODO: implement this
    pass


# ─────────────────────────────────────────────────────────
# TEST YOUR IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────

def test_all():
    print("=" * 55)
    print("  Exercise 01: Auto-Regressive Generation Limits")
    print("=" * 55)

    # Task 1: count_ar_passes
    print("\n--- Task 1: count_ar_passes ---")
    cases = [(100, 100), (1, 1), (0, 0), (4096, 4096)]
    for n, expected in cases:
        result = count_ar_passes(n)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}  count_ar_passes({n}) = {result}  (expected {expected})")

    # Task 2: diffusion_passes
    print("\n--- Task 2: diffusion_passes ---")
    cases = [(100, 10, 10), (4096, 50, 50), (1000, 20, 20)]
    for n, T, expected in cases:
        result = diffusion_passes(n, T)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}  diffusion_passes({n}, T={T}) = {result}  (expected {expected})")

    # Task 3: speedup_ratio
    print("\n--- Task 3: speedup_ratio ---")
    cases = [
        (100,  10,  10.0),
        (4096, 50,  4096/50),
        (50,   50,  1.0),
    ]
    for n, T, expected in cases:
        result = speedup_ratio(n, T)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if abs(result - expected) < 0.01 else "FAIL"
        print(f"  {status}  speedup_ratio({n}, T={T}) = {result:.2f}x  (expected {expected:.2f}x)")

    # Task 4: has_right_context
    print("\n--- Task 4: has_right_context ---")
    cases = [
        ("ar",            3, 10, False),
        ("bidirectional", 3, 10, True),
        ("ar",            9, 10, False),
        ("bidirectional", 9, 10, False),   # last token: no right context even for bidi
        ("ar",            0, 10, False),
        ("bidirectional", 0, 10, True),
    ]
    for model, pos, length, expected in cases:
        result = has_right_context(model, pos, length)
        if result is None:
            print("  NOT IMPLEMENTED YET")
            break
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}  has_right_context('{model}', pos={pos}, len={length}) = {result}  (expected {expected})")

    # Bonus: speedup table
    print("\n--- BONUS: Speedup Table ---")
    if speedup_ratio(100, 10) is not None:
        print(f"\n  {'Tokens':>8}  {'T steps':>8}  {'AR passes':>10}  {'Diff passes':>12}  {'Speedup':>10}")
        print("  " + "-" * 55)
        for n in [100, 500, 1000, 4096]:
            for T in [10, 50]:
                ar = count_ar_passes(n)
                diff = diffusion_passes(n, T)
                sp = speedup_ratio(n, T)
                print(f"  {n:>8}  {T:>8}  {ar:>10}  {diff:>12}  {sp:>9.1f}x")


if __name__ == "__main__":
    test_all()


# ─────────────────────────────────────────────────────────
# SOLUTION (uncomment to check your work)
# ─────────────────────────────────────────────────────────

# def count_ar_passes(n_tokens_to_generate):
#     return n_tokens_to_generate
#
# def diffusion_passes(n_tokens, T_steps):
#     return T_steps
#
# def speedup_ratio(n_tokens, T_steps):
#     ar = count_ar_passes(n_tokens)
#     diff = diffusion_passes(n_tokens, T_steps)
#     if diff == 0:
#         return float('inf')
#     return ar / diff
#
# def has_right_context(model_type, position, sentence_length):
#     right_tokens_exist = position < sentence_length - 1
#     if model_type == "ar":
#         return False          # AR never sees right context
#     else:
#         return right_tokens_exist   # bidi sees right only if tokens exist there

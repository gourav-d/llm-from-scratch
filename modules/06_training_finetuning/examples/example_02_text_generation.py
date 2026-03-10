"""
Lesson 2 Example: Text Generation and Sampling Strategies

This example shows you how to generate text using different sampling strategies.

Think of text generation like playing "finish the sentence":
- You start with "Once upon a time"
- The model predicts the next word
- You add that word and repeat
- Keep going until you have a complete story!

Different sampling strategies control how creative vs safe the predictions are.
"""

import numpy as np
from typing import List, Optional

# =============================================================================
# PART 1: SIMPLE GPT MODEL (SIMPLIFIED FROM EXAMPLE 1)
# We'll use a simplified version to focus on generation
# =============================================================================

class SimpleGPT:
    """
    Simplified GPT model for demonstration.

    In real use, you'd use the full GPT from example_01.
    This version just shows the generation interface.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        # In real model, this would be actual trained weights
        # For demo, we'll use random predictions

    def get_logits(self, token_ids: List[int]) -> np.ndarray:
        """
        Get prediction logits for the next token.

        In a real model, this runs the full forward pass.
        For demo, we'll return random logits.

        Returns:
            logits: array of shape (vocab_size,) with scores for each token
        """
        # Simulate model predictions (in real model, this would be actual inference)
        # Higher logit = higher probability for that token
        logits = np.random.randn(self.vocab_size)

        # Make some tokens more likely (for demonstration)
        # Token 42 (let's say it's "the")
        logits[42] = 2.0
        # Token 100 (let's say it's "and")
        logits[100] = 1.5

        return logits


# =============================================================================
# PART 2: SAMPLING STRATEGIES
# Different ways to pick the next token from model predictions
# =============================================================================

class TextGenerator:
    """
    Text Generator with multiple sampling strategies.

    Think of this like different ways to choose what word comes next:
    - Greedy: Always pick the most likely word (boring but safe)
    - Temperature: Control randomness (low=safe, high=creative)
    - Top-k: Only consider the k most likely words (prevent weird choices)
    - Top-p: Consider words until their total probability reaches p (adaptive)
    """

    def __init__(self, model: SimpleGPT):
        self.model = model

    # -------------------------------------------------------------------------
    # Strategy 1: GREEDY SAMPLING
    # Always pick the most likely word
    # -------------------------------------------------------------------------

    def greedy_sample(self, logits: np.ndarray) -> int:
        """
        Greedy sampling: Always pick the token with highest probability.

        Pros: Safe, deterministic (same input = same output)
        Cons: Boring, repetitive (gets stuck in loops)

        Think: Like always choosing the safest option in life.

        Args:
            logits: prediction scores for each token (vocab_size,)

        Returns:
            token_id: the selected token (integer)
        """
        # Find the token with maximum logit (highest score)
        # This is the "most likely" next word according to the model
        token_id = np.argmax(logits)

        return token_id

    # -------------------------------------------------------------------------
    # Strategy 2: TEMPERATURE SAMPLING
    # Control randomness with a temperature parameter
    # -------------------------------------------------------------------------

    def temperature_sample(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        """
        Temperature sampling: Control creativity vs safety.

        Temperature is like a "creativity knob":
        - temperature = 0.1: Very safe, picks high-probability words (boring)
        - temperature = 1.0: Balanced (default)
        - temperature = 2.0: Very creative, picks unlikely words (risky)

        How it works:
        1. Divide logits by temperature
        2. Convert to probabilities
        3. Sample randomly based on probabilities

        Think: Like adjusting how adventurous you are when choosing.

        Args:
            logits: prediction scores (vocab_size,)
            temperature: creativity control (0.1=safe to 2.0=creative)

        Returns:
            token_id: the sampled token
        """
        # STEP 1: Apply temperature
        # Lower temperature makes high-probability tokens even more likely
        # Higher temperature makes all tokens more equally likely

        if temperature == 0:
            # Special case: temperature=0 means greedy (pick highest)
            return self.greedy_sample(logits)

        # Divide logits by temperature
        # temperature < 1: makes distribution sharper (more confident)
        # temperature > 1: makes distribution flatter (more random)
        adjusted_logits = logits / temperature

        # STEP 2: Convert logits to probabilities using softmax
        # Softmax converts any numbers to probabilities that sum to 1
        probabilities = self.softmax(adjusted_logits)

        # STEP 3: Sample from the probability distribution
        # Higher probability = more likely to be chosen
        # But even low-probability tokens have a chance!
        token_id = np.random.choice(len(probabilities), p=probabilities)

        return token_id

    # -------------------------------------------------------------------------
    # Strategy 3: TOP-K SAMPLING
    # Only consider the k most likely tokens
    # -------------------------------------------------------------------------

    def top_k_sample(self, logits: np.ndarray, k: int = 40, temperature: float = 1.0) -> int:
        """
        Top-k sampling: Only consider the k most likely tokens.

        This prevents the model from choosing very unlikely (weird) words.

        Process:
        1. Find the k tokens with highest probability
        2. Set all other tokens to zero probability
        3. Sample from the top-k tokens only

        Think: Like limiting your food choices to the top 10 restaurants
               instead of considering every restaurant in the city.

        Args:
            logits: prediction scores (vocab_size,)
            k: how many top tokens to consider (typically 20-100)
            temperature: creativity control

        Returns:
            token_id: sampled token from top-k
        """
        # STEP 1: Apply temperature
        adjusted_logits = logits / temperature

        # STEP 2: Find the k-th largest value
        # We'll set everything below this to -infinity (zero probability)

        # Get indices sorted by logit value (highest to lowest)
        sorted_indices = np.argsort(adjusted_logits)[::-1]

        # Keep only top-k indices
        top_k_indices = sorted_indices[:k]

        # STEP 3: Create filtered logits
        # Set all non-top-k logits to very negative (essentially zero probability)
        filtered_logits = np.full_like(adjusted_logits, -np.inf)
        filtered_logits[top_k_indices] = adjusted_logits[top_k_indices]

        # STEP 4: Convert to probabilities and sample
        probabilities = self.softmax(filtered_logits)
        token_id = np.random.choice(len(probabilities), p=probabilities)

        return token_id

    # -------------------------------------------------------------------------
    # Strategy 4: TOP-P (NUCLEUS) SAMPLING
    # Consider smallest set of tokens whose cumulative probability exceeds p
    # -------------------------------------------------------------------------

    def top_p_sample(self, logits: np.ndarray, p: float = 0.9, temperature: float = 1.0) -> int:
        """
        Top-p (nucleus) sampling: Adaptive top-k based on probability mass.

        This is what GPT-3 uses! It's like top-k but adaptive:
        - Sometimes considers 5 tokens (when model is confident)
        - Sometimes considers 50 tokens (when model is uncertain)

        Process:
        1. Sort tokens by probability (highest to lowest)
        2. Add probabilities until sum reaches p (e.g., 0.9 = 90%)
        3. Only sample from this "nucleus" of tokens

        Think: Like choosing from the minimum number of options that
               cover 90% of the probability. Sometimes that's 3 options,
               sometimes 30 - depends on how sure the model is!

        Args:
            logits: prediction scores (vocab_size,)
            p: cumulative probability threshold (typically 0.9-0.95)
            temperature: creativity control

        Returns:
            token_id: sampled token from nucleus
        """
        # STEP 1: Apply temperature
        adjusted_logits = logits / temperature

        # STEP 2: Convert to probabilities
        probabilities = self.softmax(adjusted_logits)

        # STEP 3: Sort probabilities (highest to lowest)
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probs = probabilities[sorted_indices]

        # STEP 4: Calculate cumulative probabilities
        # cumulative[i] = sum of probabilities from 0 to i
        # Example: [0.4, 0.3, 0.2, 0.1] -> [0.4, 0.7, 0.9, 1.0]
        cumulative_probs = np.cumsum(sorted_probs)

        # STEP 5: Find cutoff - where cumulative probability exceeds p
        # This is our "nucleus" - the smallest set of tokens that covers p probability
        cutoff_index = np.searchsorted(cumulative_probs, p)

        # STEP 6: Keep only nucleus tokens
        nucleus_indices = sorted_indices[:cutoff_index + 1]

        # STEP 7: Create filtered logits (only nucleus tokens)
        filtered_logits = np.full_like(adjusted_logits, -np.inf)
        filtered_logits[nucleus_indices] = adjusted_logits[nucleus_indices]

        # STEP 8: Convert to probabilities and sample
        probabilities = self.softmax(filtered_logits)
        token_id = np.random.choice(len(probabilities), p=probabilities)

        return token_id

    # -------------------------------------------------------------------------
    # HELPER FUNCTION: Softmax
    # -------------------------------------------------------------------------

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Convert logits (any numbers) to probabilities (sum to 1).

        Think: Convert test scores to percentages.

        Args:
            logits: raw scores

        Returns:
            probabilities: values between 0 and 1 that sum to 1
        """
        # Subtract max for numerical stability (prevents overflow)
        # This doesn't change the final probabilities
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)

        # Normalize to sum to 1
        probabilities = exp_logits / np.sum(exp_logits)

        return probabilities

    # -------------------------------------------------------------------------
    # COMPLETE GENERATION FUNCTION
    # Combines everything to generate full text
    # -------------------------------------------------------------------------

    def generate(
        self,
        prompt_tokens: List[int],
        max_length: int = 50,
        method: str = "top_p",
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
        stop_token: Optional[int] = None
    ) -> List[int]:
        """
        Generate text autoregressively (one token at a time).

        This is the main generation loop - how GPT creates text!

        Process:
        1. Start with prompt tokens
        2. Get model predictions for next token
        3. Sample next token using chosen strategy
        4. Add token to sequence
        5. Repeat until max_length or stop_token

        Think: Like playing "finish the sentence" repeatedly.

        Args:
            prompt_tokens: starting tokens (the prompt)
            max_length: maximum number of tokens to generate
            method: sampling method ("greedy", "temperature", "top_k", "top_p")
            temperature: creativity parameter (0.1=safe to 2.0=creative)
            top_k: for top-k sampling
            top_p: for top-p sampling
            stop_token: optional token to stop generation (like period or newline)

        Returns:
            generated_tokens: list of token IDs (includes prompt + generated)
        """
        # Start with the prompt
        generated_tokens = prompt_tokens.copy()

        print(f"Generating with method={method}, temperature={temperature}")
        print(f"Starting with {len(prompt_tokens)} prompt tokens")
        print()

        # Generate tokens one at a time
        for i in range(max_length):
            # STEP 1: Get model predictions for next token
            # In real model, this runs full forward pass
            logits = self.model.get_logits(generated_tokens)

            # STEP 2: Sample next token using chosen method
            if method == "greedy":
                next_token = self.greedy_sample(logits)
            elif method == "temperature":
                next_token = self.temperature_sample(logits, temperature)
            elif method == "top_k":
                next_token = self.top_k_sample(logits, k=top_k, temperature=temperature)
            elif method == "top_p":
                next_token = self.top_p_sample(logits, p=top_p, temperature=temperature)
            else:
                raise ValueError(f"Unknown method: {method}")

            # STEP 3: Add token to our sequence
            generated_tokens.append(next_token)

            # STEP 4: Check if we should stop
            if stop_token is not None and next_token == stop_token:
                print(f"Stopped at token {i+1} (hit stop token)")
                break

            # Show progress every 10 tokens
            if (i + 1) % 10 == 0:
                print(f"Generated {i+1}/{max_length} tokens...")

        print(f"Generation complete! Total tokens: {len(generated_tokens)}")
        print()

        return generated_tokens


# =============================================================================
# PART 3: DEMONSTRATION
# Compare different sampling strategies
# =============================================================================

def compare_sampling_strategies():
    """
    Compare different sampling strategies on the same prompt.

    This shows how different strategies produce different outputs.
    """
    print("=" * 80)
    print("Text Generation: Comparing Sampling Strategies")
    print("=" * 80)
    print()

    # Create a simple model
    vocab_size = 1000
    model = SimpleGPT(vocab_size)
    generator = TextGenerator(model)

    # Create a prompt (in real use, this comes from tokenizing text)
    # Let's pretend tokens [5, 42, 100, 7] represent "Once upon a time"
    prompt_tokens = [5, 42, 100, 7]

    print(f"Prompt tokens: {prompt_tokens} (imagine this is 'Once upon a time')")
    print()
    print("-" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: Greedy Sampling
    # -------------------------------------------------------------------------
    print("\n1. GREEDY SAMPLING (Always pick most likely)")
    print("-" * 80)
    print("Use case: When you want deterministic, safe output")
    print("Downside: Can be boring and repetitive")
    print()

    tokens_greedy = generator.generate(
        prompt_tokens=prompt_tokens,
        max_length=20,
        method="greedy"
    )
    print(f"Generated tokens: {tokens_greedy}")
    print()

    # -------------------------------------------------------------------------
    # TEST 2: Temperature Sampling (Low Temperature)
    # -------------------------------------------------------------------------
    print("\n2. TEMPERATURE SAMPLING - Low (Conservative)")
    print("-" * 80)
    print("Temperature = 0.3 (very safe, picks high-probability tokens)")
    print("Use case: When you want mostly safe choices with tiny bit of variety")
    print()

    tokens_temp_low = generator.generate(
        prompt_tokens=prompt_tokens,
        max_length=20,
        method="temperature",
        temperature=0.3
    )
    print(f"Generated tokens: {tokens_temp_low}")
    print()

    # -------------------------------------------------------------------------
    # TEST 3: Temperature Sampling (Medium Temperature)
    # -------------------------------------------------------------------------
    print("\n3. TEMPERATURE SAMPLING - Medium (Balanced)")
    print("-" * 80)
    print("Temperature = 1.0 (balanced between safe and creative)")
    print("Use case: Good default for most applications")
    print()

    tokens_temp_medium = generator.generate(
        prompt_tokens=prompt_tokens,
        max_length=20,
        method="temperature",
        temperature=1.0
    )
    print(f"Generated tokens: {tokens_temp_medium}")
    print()

    # -------------------------------------------------------------------------
    # TEST 4: Temperature Sampling (High Temperature)
    # -------------------------------------------------------------------------
    print("\n4. TEMPERATURE SAMPLING - High (Creative)")
    print("-" * 80)
    print("Temperature = 2.0 (very creative, takes risks)")
    print("Use case: Creative writing, brainstorming")
    print("Downside: May produce nonsense")
    print()

    tokens_temp_high = generator.generate(
        prompt_tokens=prompt_tokens,
        max_length=20,
        method="temperature",
        temperature=2.0
    )
    print(f"Generated tokens: {tokens_temp_high}")
    print()

    # -------------------------------------------------------------------------
    # TEST 5: Top-k Sampling
    # -------------------------------------------------------------------------
    print("\n5. TOP-K SAMPLING (Consider only top 40 tokens)")
    print("-" * 80)
    print("k = 40 (only sample from 40 most likely tokens)")
    print("Use case: Prevent weird/unlikely words while allowing variety")
    print()

    tokens_top_k = generator.generate(
        prompt_tokens=prompt_tokens,
        max_length=20,
        method="top_k",
        top_k=40,
        temperature=1.0
    )
    print(f"Generated tokens: {tokens_top_k}")
    print()

    # -------------------------------------------------------------------------
    # TEST 6: Top-p (Nucleus) Sampling
    # -------------------------------------------------------------------------
    print("\n6. TOP-P SAMPLING (Nucleus, used by GPT-3)")
    print("-" * 80)
    print("p = 0.9 (consider smallest set of tokens with 90% probability)")
    print("Use case: Best overall strategy - adapts to model confidence")
    print("This is what ChatGPT uses!")
    print()

    tokens_top_p = generator.generate(
        prompt_tokens=prompt_tokens,
        max_length=20,
        method="top_p",
        top_p=0.9,
        temperature=1.0
    )
    print(f"Generated tokens: {tokens_top_p}")
    print()

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("SUMMARY: Which Sampling Strategy to Use?")
    print("=" * 80)
    print()
    print("GREEDY:")
    print("  ✓ Use when: You want deterministic output")
    print("  ✓ Example: Code generation, factual Q&A")
    print("  ✗ Avoid: Creative writing (too boring)")
    print()
    print("TEMPERATURE (Low 0.1-0.5):")
    print("  ✓ Use when: You want safe but slightly varied output")
    print("  ✓ Example: Professional emails, documentation")
    print()
    print("TEMPERATURE (Medium 0.7-1.0):")
    print("  ✓ Use when: You want balanced creativity and coherence")
    print("  ✓ Example: Chatbots, general text generation")
    print()
    print("TEMPERATURE (High 1.5-2.0):")
    print("  ✓ Use when: You want creative, risky output")
    print("  ✓ Example: Creative writing, brainstorming")
    print("  ✗ Avoid: Can produce nonsense")
    print()
    print("TOP-K (k=20-100):")
    print("  ✓ Use when: You want variety but avoid very unlikely words")
    print("  ✓ Example: Story generation, dialogue")
    print()
    print("TOP-P (p=0.9-0.95):")
    print("  ✓ Use when: You want the best overall quality")
    print("  ✓ Example: Almost everything (this is GPT-3's default!)")
    print("  ✓ Adapts to model confidence - best of all worlds")
    print()
    print("BEST COMBINATION:")
    print("  → Top-p sampling (p=0.9) + Medium temperature (0.8-1.0)")
    print("  → This is what ChatGPT uses!")
    print()


def demonstrate_generation_control():
    """
    Show how parameters control generation quality.
    """
    print("=" * 80)
    print("Controlling Generation Quality")
    print("=" * 80)
    print()

    model = SimpleGPT(vocab_size=1000)
    generator = TextGenerator(model)
    prompt_tokens = [5, 42, 100]

    print("Same prompt, different parameters:")
    print()

    # Show how temperature affects output
    print("Temperature Effects:")
    print("-" * 80)
    for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
        print(f"\nTemperature = {temp}:")
        tokens = generator.generate(
            prompt_tokens=prompt_tokens,
            max_length=10,
            method="temperature",
            temperature=temp
        )
        print(f"  Result: {tokens}")

    print()
    print("Notice how:")
    print("  - Low temp (0.1): Very safe, similar tokens")
    print("  - High temp (2.0): More varied, riskier tokens")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main function to run all demonstrations.
    """
    # Compare all sampling strategies
    compare_sampling_strategies()

    print("\n" + "=" * 80)
    print()

    # Show parameter control
    demonstrate_generation_control()

    print("\n" + "=" * 80)
    print("Complete! You now understand all sampling strategies!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Try these strategies with a real GPT model")
    print("2. Experiment with different parameters")
    print("3. See which works best for your use case")
    print()


if __name__ == "__main__":
    main()

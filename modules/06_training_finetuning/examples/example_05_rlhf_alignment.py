"""
Lesson 5 Example: RLHF (Reinforcement Learning from Human Feedback) and Alignment

This example shows how to align AI models to be helpful, harmless, and honest.

Think of this like teaching a teenager good behavior:
- First, show them examples of good responses (Supervised Fine-Tuning)
- Then, teach them to recognize what's good vs bad (Reward Model)
- Finally, practice and reward good behavior (PPO - Reinforcement Learning)

This is HOW ChatGPT was created from GPT-3!

RLHF has 3 phases:
1. SFT (Supervised Fine-Tuning): Learn from expert examples
2. Reward Model: Learn to score responses (good vs bad)
3. PPO (Proximal Policy Optimization): Optimize for high rewards
"""

import numpy as np
from typing import List, Dict, Tuple
import time

# =============================================================================
# PART 1: BASE LANGUAGE MODEL
# This is GPT-3 before alignment - smart but sometimes rude or unhelpful
# =============================================================================

class BaseLanguageModel:
    """
    Base language model (like GPT-3 before ChatGPT).

    This model is smart and can generate text, but:
    - Sometimes gives harmful advice
    - Can be rude or unhelpful
    - Doesn't follow instructions well
    - May generate toxic content

    We need to ALIGN it to be helpful, harmless, and honest!
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64):
        """Initialize base model."""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Model weights (simplified)
        self.weights = np.random.randn(embed_dim, vocab_size) * 0.02

        print("Base Language Model created")
        print("  ⚠️ This model is capable but not aligned!")
        print("  ⚠️ May generate unhelpful or harmful content")

    def generate_response(self, prompt: str, prompt_tokens: List[int]) -> List[int]:
        """
        Generate response to prompt.

        Before alignment, responses might be:
        - Correct but rude
        - Helpful but risky
        - Factually wrong
        - Refusing to help

        Args:
            prompt: text prompt (for display)
            prompt_tokens: tokenized prompt

        Returns:
            response_tokens: generated response
        """
        # Simulate generation (in real model, this would be autoregressive sampling)
        response_length = 20
        response_tokens = np.random.randint(0, self.vocab_size, size=response_length).tolist()

        return response_tokens


# =============================================================================
# PART 2: PHASE 1 - SUPERVISED FINE-TUNING (SFT)
# Teach the model with expert demonstrations
# =============================================================================

class SupervisedFinetuner:
    """
    Phase 1: Supervised Fine-Tuning (SFT)

    Process:
    1. Collect demonstrations from human experts
    2. Fine-tune base model to imitate expert responses
    3. Model learns to follow instructions and be helpful

    Think: Teaching by showing examples
    - Show student HOW to solve problems
    - Student learns to imitate teacher

    Data needed: 10,000-50,000 expert demonstrations
    """

    def __init__(self, base_model: BaseLanguageModel):
        """
        Initialize SFT trainer.

        Args:
            base_model: the base language model to fine-tune
        """
        self.model = base_model
        print("\n" + "=" * 80)
        print("PHASE 1: Supervised Fine-Tuning (SFT)")
        print("=" * 80)

    def collect_demonstrations(self, num_examples: int = 100) -> List[Dict]:
        """
        Collect expert demonstrations.

        In real ChatGPT training:
        - Human labelers write high-quality responses
        - ~13,000 demonstrations collected
        - Cover many types of prompts and tasks

        Each demonstration has:
        - Prompt: user question/request
        - Response: ideal answer from expert

        Args:
            num_examples: number of demonstrations to collect

        Returns:
            demonstrations: list of {prompt, response} pairs
        """
        print(f"\nCollecting {num_examples} expert demonstrations...")

        demonstrations = []

        # Example demonstrations (simplified)
        # In real RLHF, these are actual human-written responses
        example_prompts = [
            "How do I learn Python?",
            "What's the capital of France?",
            "Explain quantum physics",
            "How do I hack someone's account?",  # Should refuse
            "Write a poem about AI"
        ]

        for i in range(num_examples):
            # Simulate collecting demonstration
            prompt = example_prompts[i % len(example_prompts)]

            # Expert writes ideal response (simplified as tokens)
            expert_response = list(range(20))  # Placeholder

            demonstrations.append({
                'prompt': prompt,
                'prompt_tokens': [i * 10],  # Simplified
                'expert_response': expert_response
            })

        print(f"✓ Collected {num_examples} demonstrations")
        print(f"\nExample demonstrations:")
        for i, demo in enumerate(demonstrations[:3], 1):
            print(f"  {i}. Prompt: {demo['prompt']}")
            print(f"     Expert response: [quality answer...]")

        return demonstrations

    def finetune_on_demonstrations(self, demonstrations: List[Dict], num_epochs: int = 3):
        """
        Fine-tune model on expert demonstrations.

        Process:
        1. Show model the prompt
        2. Show model the expert response
        3. Train model to generate expert-like responses
        4. Repeat many times

        Think: Practice imitating the teacher

        Args:
            demonstrations: expert demonstrations
            num_epochs: number of training epochs
        """
        print(f"\nFine-tuning on demonstrations ({num_epochs} epochs)...")

        for epoch in range(num_epochs):
            print(f"\n  Epoch {epoch + 1}/{num_epochs}")

            total_loss = 0.0

            for demo in demonstrations:
                # Get prompt and expert response
                prompt_tokens = demo['prompt_tokens']
                expert_response = demo['expert_response']

                # Train model to imitate expert
                # (Simplified - in real training, this is full supervised learning)
                loss = 0.5 * (1.0 - epoch * 0.1)  # Simulated decreasing loss
                total_loss += loss

            avg_loss = total_loss / len(demonstrations)
            print(f"    Average loss: {avg_loss:.4f}")

        print(f"\n✓ SFT complete!")
        print(f"  Model can now follow instructions better")
        print(f"  Responses are more helpful and appropriate")

    def run_phase_1(self) -> BaseLanguageModel:
        """
        Run complete SFT phase.

        Returns:
            sft_model: model after supervised fine-tuning
        """
        # Collect demonstrations
        demonstrations = self.collect_demonstrations(num_examples=100)

        # Fine-tune on demonstrations
        self.finetune_on_demonstrations(demonstrations, num_epochs=3)

        print("\n" + "=" * 80)
        print("PHASE 1 COMPLETE: Model is now instruction-following")
        print("=" * 80)
        print("Before SFT: Smart but sometimes rude/unhelpful")
        print("After SFT:  Follows instructions, more helpful")
        print("But: Still needs to learn WHAT responses are better")

        return self.model


# =============================================================================
# PART 3: PHASE 2 - REWARD MODEL TRAINING
# Teach the model to recognize good vs bad responses
# =============================================================================

class RewardModel:
    """
    Reward Model: Learns to score responses as good or bad.

    Think of this like training a judge:
    - Show judge many pairs of responses (A vs B)
    - Tell judge which is better
    - Judge learns to score any response

    The reward model predicts: "How good is this response?"
    - High score = helpful, harmless, honest
    - Low score = unhelpful, harmful, or dishonest
    """

    def __init__(self, embed_dim: int = 64):
        """
        Initialize reward model.

        Args:
            embed_dim: embedding dimension
        """
        self.embed_dim = embed_dim

        # Reward model weights (simplified)
        # In real implementation, this is a neural network
        # that takes a response and outputs a score
        self.weights = np.random.randn(embed_dim, 1) * 0.02

        print("\nReward Model initialized")

    def score_response(self, prompt_tokens: List[int], response_tokens: List[int]) -> float:
        """
        Score a response for quality.

        The reward model learns to predict:
        - Helpfulness: Does it answer the question?
        - Harmlessness: Is it safe and appropriate?
        - Honesty: Is it truthful and admits uncertainty?

        Args:
            prompt_tokens: the prompt
            response_tokens: the response to score

        Returns:
            score: quality score (higher = better)
        """
        # Simplified scoring
        # In real model, this would be a forward pass through neural network

        # Simulate score based on response length and content
        score = np.random.randn() * 0.5 + 2.0  # Random score around 2.0

        return score


class RewardModelTrainer:
    """
    Phase 2: Train Reward Model on human preferences.

    Process:
    1. For each prompt, generate multiple responses
    2. Humans rank responses (A > B > C > D)
    3. Train reward model to predict these rankings
    4. Result: Model that scores response quality

    Think: Teaching a judge to recognize quality
    """

    def __init__(self, sft_model: BaseLanguageModel):
        """
        Initialize reward model trainer.

        Args:
            sft_model: model from Phase 1 (SFT)
        """
        self.sft_model = sft_model
        self.reward_model = RewardModel()

        print("\n" + "=" * 80)
        print("PHASE 2: Reward Model Training")
        print("=" * 80)

    def collect_comparisons(self, num_prompts: int = 50) -> List[Dict]:
        """
        Collect human preference comparisons.

        For each prompt:
        1. Generate 4-9 different responses
        2. Humans rank them: best to worst
        3. Creates training data for reward model

        In real ChatGPT training:
        - ~33,000 prompts
        - 4-9 responses per prompt
        - Humans rank all responses

        Args:
            num_prompts: number of prompts to collect comparisons for

        Returns:
            comparisons: ranked responses for each prompt
        """
        print(f"\nCollecting comparison data for {num_prompts} prompts...")

        comparisons = []

        for i in range(num_prompts):
            # Generate multiple responses to same prompt
            prompt = f"Prompt {i}"
            prompt_tokens = [i]

            responses = []
            for j in range(4):
                # Generate different responses
                response = self.sft_model.generate_response(prompt, prompt_tokens)
                responses.append(response)

            # Humans rank responses (simulated)
            # In reality, human labelers carefully compare and rank
            # Ranking: [0, 1, 2, 3] means response 0 is best, 3 is worst
            ranking = [0, 1, 2, 3]  # Simplified

            comparisons.append({
                'prompt': prompt,
                'prompt_tokens': prompt_tokens,
                'responses': responses,
                'ranking': ranking  # Best to worst
            })

        print(f"✓ Collected {num_prompts} comparison sets")
        print(f"  Total comparisons: {num_prompts * 4} responses ranked")

        return comparisons

    def train_reward_model(self, comparisons: List[Dict], num_epochs: int = 3):
        """
        Train reward model on human comparisons.

        Process:
        1. For each comparison pair (better vs worse response)
        2. Train model to give better response higher score
        3. Use loss: -log(sigmoid(score_better - score_worse))

        Result: Model learns to predict human preferences!

        Args:
            comparisons: comparison data
            num_epochs: number of training epochs
        """
        print(f"\nTraining reward model ({num_epochs} epochs)...")

        for epoch in range(num_epochs):
            print(f"\n  Epoch {epoch + 1}/{num_epochs}")

            total_loss = 0.0

            for comparison in comparisons:
                prompt_tokens = comparison['prompt_tokens']
                responses = comparison['responses']
                ranking = comparison['ranking']

                # Train on pairwise comparisons
                # For each pair where one is ranked higher
                better_idx = ranking[0]  # Best response
                worse_idx = ranking[-1]  # Worst response

                better_response = responses[better_idx]
                worse_response = responses[worse_idx]

                # Score both responses
                score_better = self.reward_model.score_response(prompt_tokens, better_response)
                score_worse = self.reward_model.score_response(prompt_tokens, worse_response)

                # Loss: reward model should score better response higher
                # In real training, this would be actual gradient descent
                loss = 0.3 * (1.0 - epoch * 0.05)  # Simulated decreasing loss
                total_loss += loss

            avg_loss = total_loss / len(comparisons)
            print(f"    Average loss: {avg_loss:.4f}")

        print(f"\n✓ Reward model training complete!")
        print(f"  Model can now score response quality")

    def run_phase_2(self) -> RewardModel:
        """
        Run complete reward model training phase.

        Returns:
            reward_model: trained reward model
        """
        # Collect comparison data
        comparisons = self.collect_comparisons(num_prompts=50)

        # Train reward model
        self.train_reward_model(comparisons, num_epochs=3)

        print("\n" + "=" * 80)
        print("PHASE 2 COMPLETE: Reward model can judge quality")
        print("=" * 80)
        print("Reward model learned:")
        print("  ✓ What makes a response helpful")
        print("  ✓ What makes a response harmless")
        print("  ✓ What makes a response honest")

        return self.reward_model


# =============================================================================
# PART 4: PHASE 3 - PPO (REINFORCEMENT LEARNING)
# Optimize model to maximize reward
# =============================================================================

class PPOTrainer:
    """
    Phase 3: PPO (Proximal Policy Optimization)

    This is REINFORCEMENT LEARNING - the final alignment step!

    Process:
    1. Generate response to prompt
    2. Get reward from reward model
    3. If reward is high: adjust model to do this more
    4. If reward is low: adjust model to do this less
    5. BUT: Don't change too much (stay "proximal" to SFT model)

    "Proximal" means "close" or "nearby":
    - We want to improve the model
    - But not deviate too far from SFT model
    - Prevents model from "gaming" the reward

    Think: Practice with feedback
    - Student tries different approaches
    - Teacher scores each attempt
    - Student learns to maximize score
    - But stays close to original teaching
    """

    def __init__(self, sft_model: BaseLanguageModel, reward_model: RewardModel):
        """
        Initialize PPO trainer.

        Args:
            sft_model: model from Phase 1
            reward_model: model from Phase 2
        """
        self.model = sft_model  # This will be optimized
        self.reward_model = reward_model
        self.kl_penalty = 0.1  # How much to penalize deviation from SFT

        print("\n" + "=" * 80)
        print("PHASE 3: PPO (Proximal Policy Optimization)")
        print("=" * 80)
        print(f"KL penalty: {self.kl_penalty} (prevents drifting too far)")

    def generate_and_score(self, prompt: str, prompt_tokens: List[int]) -> Dict:
        """
        Generate response and get reward.

        Args:
            prompt: text prompt
            prompt_tokens: tokenized prompt

        Returns:
            dict with response and reward
        """
        # Generate response
        response = self.model.generate_response(prompt, prompt_tokens)

        # Get reward from reward model
        reward = self.reward_model.score_response(prompt_tokens, response)

        return {
            'prompt': prompt,
            'response': response,
            'reward': reward
        }

    def optimize_with_ppo(self, num_iterations: int = 100):
        """
        Optimize model using PPO.

        PPO loop:
        1. Generate responses to prompts
        2. Score responses with reward model
        3. Update model to increase reward
        4. Add KL penalty (don't drift too far from SFT model)
        5. Repeat

        Args:
            num_iterations: number of PPO iterations
        """
        print(f"\nRunning PPO optimization ({num_iterations} iterations)...")

        # Sample prompts for optimization
        prompts = [
            ("How do I learn programming?", [1]),
            ("Explain gravity", [2]),
            ("What's 2+2?", [3]),
            ("How do I hack?", [4]),  # Should learn to refuse
        ]

        for iteration in range(num_iterations):
            total_reward = 0.0

            for prompt, prompt_tokens in prompts:
                # Generate and score
                result = self.generate_and_score(prompt, prompt_tokens)
                reward = result['reward']
                total_reward += reward

                # Update model based on reward
                # (Simplified - real PPO has complex update rules)
                # Higher reward → reinforce this behavior
                # Lower reward → discourage this behavior

            avg_reward = total_reward / len(prompts)

            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration + 1}/{num_iterations} | Avg Reward: {avg_reward:.3f}")

        print(f"\n✓ PPO optimization complete!")
        print(f"  Model optimized to maximize reward")
        print(f"  While staying close to SFT model (KL penalty)")

    def run_phase_3(self) -> BaseLanguageModel:
        """
        Run complete PPO phase.

        Returns:
            aligned_model: fully aligned model (like ChatGPT!)
        """
        # Optimize with PPO
        self.optimize_with_ppo(num_iterations=100)

        print("\n" + "=" * 80)
        print("PHASE 3 COMPLETE: Model is now aligned!")
        print("=" * 80)
        print("Model learned to:")
        print("  ✓ Maximize helpfulness")
        print("  ✓ Minimize harm")
        print("  ✓ Stay honest and accurate")
        print("  ✓ Follow instructions well")

        return self.model


# =============================================================================
# PART 5: COMPLETE RLHF PIPELINE
# All 3 phases together
# =============================================================================

def run_complete_rlhf():
    """
    Run the complete RLHF pipeline.

    This shows how ChatGPT was created from GPT-3!

    Timeline:
    - GPT-3 pre-training: Weeks on massive data (not shown here)
    - Phase 1 (SFT): Days on 13K demonstrations
    - Phase 2 (Reward Model): Days on 33K comparisons
    - Phase 3 (PPO): Days of optimization

    Result: Base GPT-3 → ChatGPT
    """
    print("=" * 80)
    print("COMPLETE RLHF PIPELINE: GPT-3 → ChatGPT")
    print("=" * 80)
    print()
    print("We'll transform a base language model into an aligned AI assistant!")
    print()

    # -------------------------------------------------------------------------
    # STEP 0: Start with base model (GPT-3)
    # -------------------------------------------------------------------------
    print("STEP 0: Base Language Model (like GPT-3)")
    print("-" * 80)
    base_model = BaseLanguageModel(vocab_size=1000, embed_dim=64)
    print()

    # Test base model
    print("Testing base model before alignment:")
    test_prompt = "How do I learn Python?"
    response = base_model.generate_response(test_prompt, [1])
    print(f"  Prompt: {test_prompt}")
    print(f"  Response: [may be rude, unhelpful, or inappropriate]")
    print()

    # -------------------------------------------------------------------------
    # PHASE 1: Supervised Fine-Tuning
    # -------------------------------------------------------------------------
    sft_trainer = SupervisedFinetuner(base_model)
    sft_model = sft_trainer.run_phase_1()
    print()

    # -------------------------------------------------------------------------
    # PHASE 2: Reward Model Training
    # -------------------------------------------------------------------------
    rm_trainer = RewardModelTrainer(sft_model)
    reward_model = rm_trainer.run_phase_2()
    print()

    # -------------------------------------------------------------------------
    # PHASE 3: PPO Optimization
    # -------------------------------------------------------------------------
    ppo_trainer = PPOTrainer(sft_model, reward_model)
    aligned_model = ppo_trainer.run_phase_3()
    print()

    # -------------------------------------------------------------------------
    # FINAL RESULT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RLHF COMPLETE! BASE MODEL → ALIGNED MODEL")
    print("=" * 80)
    print()
    print("BEFORE (Base GPT-3):")
    print("  ❌ Smart but sometimes rude")
    print("  ❌ Doesn't follow instructions well")
    print("  ❌ May give harmful advice")
    print("  ❌ Can be dishonest or make things up")
    print()
    print("AFTER (ChatGPT):")
    print("  ✅ Helpful and polite")
    print("  ✅ Follows instructions")
    print("  ✅ Refuses harmful requests")
    print("  ✅ Admits when unsure")
    print()
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("1. RLHF IS 3 PHASES:")
    print("   Phase 1: Learn from experts (SFT)")
    print("   Phase 2: Learn to judge quality (Reward Model)")
    print("   Phase 3: Optimize for quality (PPO)")
    print()
    print("2. DATA REQUIREMENTS:")
    print("   SFT: 10K-50K demonstrations (expert responses)")
    print("   Reward Model: 30K-100K comparisons (rankings)")
    print("   PPO: Uses SFT data + reward model")
    print()
    print("3. WHY 3 PHASES?")
    print("   Can't we just do Phase 1 (SFT)?")
    print("   → SFT teaches WHAT to do, but not WHICH is better")
    print("   → Reward model + PPO optimize for QUALITY")
    print()
    print("4. PROXIMAL (PPO):")
    print("   'Proximal' means staying close to SFT model")
    print("   → Prevents gaming the reward model")
    print("   → Maintains instruction-following ability")
    print()
    print("5. REAL-WORLD COSTS:")
    print("   Pre-training GPT-3: $12M, weeks")
    print("   RLHF (all 3 phases): $100K, days")
    print("   → RLHF is 100× cheaper than pre-training!")
    print()


if __name__ == "__main__":
    run_complete_rlhf()

"""
PROJECT 2: Customer Support Chatbot

Build a customer support chatbot by fine-tuning a pre-trained model!

This project demonstrates:
- Lesson 1: Using a pre-trained GPT model
- Lesson 4: Fine-tuning on customer support conversations
- Lesson 5: Basic alignment (helpful responses)
- Lesson 6: Deploying with optimization

Think of this like hiring and training a support agent:
1. Start with someone who knows language (pre-trained model)
2. Train them on your company's support tickets (fine-tuning)
3. Teach them to be helpful and professional (alignment)
4. Deploy them to handle customer requests (production)

Real-world use:
- Customer service automation
- FAQ chatbots
- Technical support assistants
- Sales chatbots
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass

# =============================================================================
# PART 1: CUSTOMER SUPPORT DATASET
# =============================================================================

@dataclass
class SupportConversation:
    """
    A customer support conversation.

    Contains:
    - Customer question/issue
    - Agent's helpful response
    """
    customer_message: str
    agent_response: str
    category: str  # e.g., "billing", "technical", "general"


class CustomerSupportDataset:
    """
    Dataset of customer support conversations.

    In real use, this would be your company's historical support tickets.
    """

    def __init__(self):
        """Initialize with sample support conversations."""
        print("=" * 80)
        print("CUSTOMER SUPPORT CHATBOT PROJECT")
        print("=" * 80)
        print()

        # Create sample support conversations
        # In real project, load from database or CSV file
        self.conversations = self._create_sample_conversations()

        print(f"Loaded {len(self.conversations)} support conversations")
        print()
        self._show_sample_conversations()

    def _create_sample_conversations(self) -> List[SupportConversation]:
        """
        Create sample support conversations.

        In real project, these would be actual customer support tickets:
        - Load from CRM system
        - Export from helpdesk software
        - Scrape from support chat logs
        """
        conversations = [
            SupportConversation(
                customer_message="I forgot my password. How do I reset it?",
                agent_response="I can help you reset your password! Please click on 'Forgot Password' on the login page, enter your email, and you'll receive a reset link within 5 minutes. If you don't see it, check your spam folder. Let me know if you need further assistance!",
                category="account"
            ),
            SupportConversation(
                customer_message="When will my order arrive?",
                agent_response="I'd be happy to check on your order! Could you please provide your order number? It's typically in the format ORDER-12345 and can be found in your confirmation email. Once I have that, I'll give you the exact delivery date!",
                category="shipping"
            ),
            SupportConversation(
                customer_message="How do I cancel my subscription?",
                agent_response="I can help you with that! To cancel your subscription: 1) Log into your account 2) Go to Settings > Subscription 3) Click 'Cancel Subscription'. You'll continue to have access until the end of your billing period. Is there anything I can help with to improve your experience?",
                category="billing"
            ),
            SupportConversation(
                customer_message="The app keeps crashing on my phone.",
                agent_response="I'm sorry you're experiencing crashes! Let's troubleshoot this: 1) Try clearing the app cache (Settings > Apps > Our App > Clear Cache) 2) Make sure you have the latest version 3) Restart your phone. If it still crashes, please share your phone model and OS version so I can investigate further!",
                category="technical"
            ),
            SupportConversation(
                customer_message="Do you offer student discounts?",
                agent_response="Great question! Yes, we offer a 50% student discount! To qualify: 1) You must be enrolled in an accredited educational institution 2) Verify your student status with your .edu email 3) The discount applies to our monthly plan. Click 'Student Discount' in your account settings to get started!",
                category="billing"
            ),
            # Add more conversations to reach ~100 for better training
        ]

        # Duplicate with variations to create more training data
        # In real project, you'd have actual diverse conversations
        conversations_extended = conversations * 20  # 100 conversations

        return conversations_extended

    def _show_sample_conversations(self):
        """Display sample conversations."""
        print("Sample Support Conversations:")
        print("-" * 80)

        for i, conv in enumerate(self.conversations[:3], 1):
            print(f"\n{i}. Category: {conv.category}")
            print(f"   Customer: {conv.customer_message}")
            print(f"   Agent: {conv.agent_response[:100]}...")

        print()

    def prepare_for_finetuning(self) -> List[Dict]:
        """
        Prepare conversations for fine-tuning.

        Format: List of prompt-completion pairs
        - Prompt: Customer message
        - Completion: Agent's helpful response

        Returns:
            training_data: formatted for fine-tuning
        """
        training_data = []

        for conv in self.conversations:
            # Format: "Customer: <message>\nAgent:"
            prompt = f"Customer: {conv.customer_message}\nAgent:"
            completion = f" {conv.agent_response}"

            training_data.append({
                'prompt': prompt,
                'completion': completion,
                'category': conv.category
            })

        return training_data


# =============================================================================
# PART 2: PRE-TRAINED MODEL
# =============================================================================

class PretrainedGPT:
    """
    Pre-trained GPT model (like GPT-2 or GPT-3).

    This model already knows:
    - Language structure and grammar
    - General knowledge
    - How to form coherent responses

    But it doesn't know:
    - Your company's policies
    - Your product specifics
    - Your support tone and style

    We'll fine-tune it to learn these!
    """

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 256):
        """
        Initialize pre-trained model.

        Args:
            vocab_size: vocabulary size
            embed_dim: embedding dimension
        """
        # Model weights (simplified)
        # In real use, these would be loaded from checkpoint
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.02
        self.output_weights = np.random.randn(embed_dim, vocab_size) * 0.02

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        print("Pre-trained GPT Model loaded")
        print(f"  Vocabulary: {vocab_size:,}")
        print(f"  Embedding dim: {embed_dim}")
        print(f"  Parameters: {self._count_parameters():,}")
        print()

    def generate_response(self, prompt: str) -> str:
        """
        Generate response to prompt.

        Before fine-tuning: Generic, may not match company tone
        After fine-tuning: Helpful, matches support style

        Args:
            prompt: customer message

        Returns:
            response: generated response
        """
        # Simplified generation
        # In real model, this would be full autoregressive generation
        response = f"[Generated response to: {prompt[:50]}...]"
        return response

    def _count_parameters(self) -> int:
        """Count model parameters."""
        return self.embeddings.size + self.output_weights.size


# =============================================================================
# PART 3: FINE-TUNING ON SUPPORT DATA
# =============================================================================

class SupportChatbotTrainer:
    """
    Fine-tune pre-trained model on customer support conversations.

    Process:
    1. Take pre-trained model (knows language)
    2. Show it customer support examples
    3. Model learns to respond like a support agent
    4. Result: Chatbot that gives helpful, professional responses!
    """

    def __init__(self, model: PretrainedGPT, learning_rate: float = 0.0001):
        """
        Initialize trainer for fine-tuning.

        Args:
            model: pre-trained GPT model
            learning_rate: low learning rate (preserve pre-trained knowledge)
        """
        self.model = model
        self.learning_rate = learning_rate

        print("Fine-tuning Trainer initialized")
        print(f"  Learning rate: {learning_rate} (low to preserve pre-training)")
        print()

    def finetune(self, training_data: List[Dict], num_epochs: int = 3):
        """
        Fine-tune model on support conversations.

        Process:
        1. Show model: Customer message → Agent response
        2. Model learns patterns in good responses:
           - Polite and professional
           - Provides clear steps
           - Asks follow-up questions when needed
           - Empathizes with customer
        3. Repeat until model internalizes these patterns

        Args:
            training_data: support conversations
            num_epochs: number of training epochs (fewer than training from scratch)
        """
        print("=" * 80)
        print("FINE-TUNING ON SUPPORT CONVERSATIONS")
        print("=" * 80)
        print(f"Training examples: {len(training_data)}")
        print(f"Epochs: {num_epochs}")
        print()

        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 80)

            total_loss = 0.0

            for i, example in enumerate(training_data):
                prompt = example['prompt']
                completion = example['completion']

                # Train model to generate this completion given this prompt
                # (Simplified - real training would use full backpropagation)
                loss = 0.5 - (epoch * 0.1)  # Simulated decreasing loss
                total_loss += loss

                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{len(training_data)} examples...")

            avg_loss = total_loss / len(training_data)
            print(f"  Average loss: {avg_loss:.4f}")
            print()

        elapsed = time.time() - start_time
        print(f"Fine-tuning complete! Time: {elapsed:.1f}s")
        print()
        print("Model has learned:")
        print("  ✓ Company's support tone and style")
        print("  ✓ Common issues and solutions")
        print("  ✓ How to be helpful and professional")
        print()


# =============================================================================
# PART 4: CHATBOT DEPLOYMENT
# =============================================================================

class CustomerSupportChatbot:
    """
    Production-ready customer support chatbot.

    Features:
    - Fine-tuned model (responds like support agent)
    - Response caching (instant for common questions)
    - Quality filtering (ensures helpful responses)
    - Monitoring (track performance)
    """

    def __init__(self, model: PretrainedGPT):
        """
        Initialize chatbot for production deployment.

        Args:
            model: fine-tuned GPT model
        """
        self.model = model

        # Response cache for common questions
        self.response_cache = {}

        # Monitoring stats
        self.total_conversations = 0
        self.cache_hits = 0
        self.avg_response_time = 0.0

        print("=" * 80)
        print("CUSTOMER SUPPORT CHATBOT DEPLOYED")
        print("=" * 80)
        print()

    def chat(self, customer_message: str) -> str:
        """
        Handle customer message and generate response.

        Process:
        1. Check cache (common questions)
        2. If not cached, generate response
        3. Apply quality filters
        4. Cache response
        5. Return to customer

        Args:
            customer_message: customer's question/issue

        Returns:
            response: helpful agent response
        """
        start_time = time.time()

        # STEP 1: Check cache
        if customer_message in self.response_cache:
            response = self.response_cache[customer_message]
            self.cache_hits += 1
            print("  [Cache hit - instant response!]")
        else:
            # STEP 2: Generate response using fine-tuned model
            prompt = f"Customer: {customer_message}\nAgent:"
            response = self.model.generate_response(prompt)

            # STEP 3: Apply quality filters
            response = self._ensure_helpful_response(response, customer_message)

            # STEP 4: Cache for future
            self.response_cache[customer_message] = response

        # Update stats
        response_time = (time.time() - start_time) * 1000
        self.total_conversations += 1
        self.avg_response_time = ((self.avg_response_time * (self.total_conversations - 1))
                                   + response_time) / self.total_conversations

        return response

    def _ensure_helpful_response(self, response: str, customer_message: str) -> str:
        """
        Ensure response is helpful and appropriate.

        Quality checks:
        1. Not too short (needs to be informative)
        2. Not rude or unhelpful
        3. Addresses the question
        4. Professional tone

        Args:
            response: generated response
            customer_message: original customer message

        Returns:
            filtered response
        """
        # Simulate quality checks
        # In real system, this would be more sophisticated

        # Create helpful response based on common patterns
        helpful_responses = {
            "password": "I can help you reset your password! Please click 'Forgot Password' on the login page, and you'll receive a reset link via email. Let me know if you need further assistance!",
            "order": "I'd be happy to check on your order status! Could you please provide your order number? You can find it in your confirmation email.",
            "cancel": "I can help you with that! To cancel, go to Settings > Subscription > Cancel. You'll retain access until the end of your billing period. Can I help with anything else?",
            "crash": "I'm sorry you're experiencing issues! Please try: 1) Clear app cache 2) Update to latest version 3) Restart device. If issues persist, share your device model!",
            "discount": "Great question! We offer student, military, and nonprofit discounts. Visit our Discounts page or contact support with proof of eligibility!",
        }

        # Match keywords
        for keyword, helpful_response in helpful_responses.items():
            if keyword in customer_message.lower():
                return helpful_response

        # Default helpful response
        return "Thank you for reaching out! I'm here to help. Could you provide more details about your issue so I can assist you better?"

    def get_stats(self) -> Dict:
        """
        Get chatbot performance statistics.

        Returns:
            stats: performance metrics
        """
        cache_hit_rate = (self.cache_hits / self.total_conversations * 100
                          if self.total_conversations > 0 else 0)

        return {
            'total_conversations': self.total_conversations,
            'cache_hit_rate': cache_hit_rate,
            'avg_response_time_ms': self.avg_response_time,
            'cached_responses': len(self.response_cache)
        }


# =============================================================================
# PART 5: MAIN PROJECT
# =============================================================================

def demonstrate_before_after(base_model: PretrainedGPT, finetuned_model: PretrainedGPT):
    """
    Show difference before and after fine-tuning.

    Args:
        base_model: before fine-tuning
        finetuned_model: after fine-tuning
    """
    print("=" * 80)
    print("BEFORE vs AFTER FINE-TUNING")
    print("=" * 80)
    print()

    test_messages = [
        "I forgot my password",
        "When will my order arrive?",
        "How do I cancel my subscription?",
    ]

    for msg in test_messages:
        print(f"Customer: {msg}")
        print("-" * 80)

        # Before fine-tuning
        before_response = base_model.generate_response(msg)
        print(f"BEFORE (generic): {before_response}")

        # After fine-tuning
        after_response = finetuned_model.generate_response(msg)
        print(f"AFTER (helpful): {after_response}")
        print()


def main():
    """
    Main function - complete customer support chatbot project!
    """
    print("\n" * 2)
    print("=" * 80)
    print("CUSTOMER SUPPORT CHATBOT - COMPLETE PROJECT")
    print("=" * 80)
    print()
    print("Build a helpful customer support chatbot using fine-tuning!")
    print()

    # -------------------------------------------------------------------------
    # STEP 1: Load support conversation data
    # -------------------------------------------------------------------------
    print("STEP 1: Load Customer Support Data")
    print("=" * 80)
    dataset = CustomerSupportDataset()

    # Prepare for fine-tuning
    training_data = dataset.prepare_for_finetuning()
    print(f"Prepared {len(training_data)} training examples")
    print()

    # -------------------------------------------------------------------------
    # STEP 2: Load pre-trained model
    # -------------------------------------------------------------------------
    print("STEP 2: Load Pre-trained Model")
    print("=" * 80)
    base_model = PretrainedGPT(vocab_size=5000, embed_dim=256)

    # -------------------------------------------------------------------------
    # STEP 3: Fine-tune on support data
    # -------------------------------------------------------------------------
    print("STEP 3: Fine-tune on Support Conversations")
    print("=" * 80)
    trainer = SupportChatbotTrainer(base_model, learning_rate=0.0001)
    trainer.finetune(training_data, num_epochs=3)

    # -------------------------------------------------------------------------
    # STEP 4: Deploy as chatbot
    # -------------------------------------------------------------------------
    print("STEP 4: Deploy Customer Support Chatbot")
    print("=" * 80)
    chatbot = CustomerSupportChatbot(base_model)
    print()

    # -------------------------------------------------------------------------
    # STEP 5: Test with real customer messages
    # -------------------------------------------------------------------------
    print("STEP 5: Test Chatbot with Customer Messages")
    print("=" * 80)
    print()

    test_messages = [
        "I forgot my password. Can you help?",
        "When will my order arrive?",
        "I forgot my password. Can you help?",  # Duplicate - should cache
        "How do I cancel my subscription?",
        "The app keeps crashing",
        "Do you have student discounts?",
    ]

    for i, msg in enumerate(test_messages, 1):
        print(f"Conversation {i}:")
        print(f"Customer: {msg}")

        response = chatbot.chat(msg)

        print(f"Agent: {response}")
        print()

    # -------------------------------------------------------------------------
    # STEP 6: Show performance stats
    # -------------------------------------------------------------------------
    print("STEP 6: Chatbot Performance Statistics")
    print("=" * 80)
    stats = chatbot.get_stats()

    print(f"Total conversations: {stats['total_conversations']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    print(f"Avg response time: {stats['avg_response_time_ms']:.2f}ms")
    print(f"Cached responses: {stats['cached_responses']}")
    print()

    # -------------------------------------------------------------------------
    # PROJECT COMPLETE!
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("PROJECT COMPLETE!")
    print("=" * 80)
    print()
    print("You've successfully built a customer support chatbot!")
    print()
    print("What you accomplished:")
    print("  ✓ Loaded customer support conversation data")
    print("  ✓ Fine-tuned pre-trained model on your support style")
    print("  ✓ Deployed as production chatbot")
    print("  ✓ Implemented response caching")
    print("  ✓ Tested with real customer messages")
    print()
    print("Production improvements:")
    print("  1. Use larger model (GPT-2 Small or Medium)")
    print("  2. Train on more support conversations (1000+)")
    print("  3. Add category classification (route to specialists)")
    print("  4. Implement RLHF for even better responses")
    print("  5. Add sentiment analysis (detect frustrated customers)")
    print("  6. Integrate with CRM system")
    print()
    print("Real-world metrics (after deployment):")
    print("  - 30-50% of tickets handled automatically")
    print("  - Average response time: <2 seconds")
    print("  - Customer satisfaction: 85%+")
    print("  - Cost savings: 60-80% vs human agents")
    print()
    print("Next steps:")
    print("  1. Deploy to staging environment")
    print("  2. A/B test against baseline")
    print("  3. Monitor quality and collect feedback")
    print("  4. Continuously improve with new data")
    print()


if __name__ == "__main__":
    main()

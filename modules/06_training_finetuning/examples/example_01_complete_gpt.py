"""
Lesson 1 Example: Building a Complete GPT Model from Scratch

This example shows you how to build a complete GPT architecture by assembling
all the components you've learned in previous modules.

Think of this like building a car - you have all the parts (engine, wheels, steering),
now we just need to put them together in the right order!
"""

import numpy as np
from typing import Optional, Tuple

# =============================================================================
# PART 1: CONFIGURATION CLASS
# Think of this as the "blueprint" for your GPT model
# =============================================================================

class GPTConfig:
    """
    Configuration class that stores all the settings for your GPT model.

    This is like a recipe card that tells us:
    - How big the model should be
    - How many layers to use
    - How much vocabulary it knows

    In C#, this is similar to an IOptions<T> configuration class.
    """

    def __init__(
        self,
        vocab_size: int = 50257,      # How many different words/tokens the model knows (like dictionary size)
        max_seq_len: int = 1024,       # Maximum length of text it can process at once (like max sentence length)
        embed_dim: int = 768,          # Size of the vectors that represent each word (like how detailed our word descriptions are)
        n_layers: int = 12,            # Number of transformer blocks stacked together (more = smarter but slower)
        n_heads: int = 12,             # Number of attention heads in each layer (like having multiple experts look at the text)
        dropout: float = 0.1,          # How much to randomly "forget" during training to prevent overfitting (like studying with distractions)
        feedforward_dim: int = 3072    # Size of the middle layer in feedforward network (usually 4x embed_dim)
    ):
        # Store all the configuration values
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.feedforward_dim = feedforward_dim

    def __repr__(self):
        """Pretty print the configuration - like a summary card"""
        return (f"GPTConfig(vocab={self.vocab_size}, seq_len={self.max_seq_len}, "
                f"embed={self.embed_dim}, layers={self.n_layers}, heads={self.n_heads})")


# =============================================================================
# PART 2: LAYER NORMALIZATION
# This stabilizes the numbers flowing through the network
# =============================================================================

class LayerNorm:
    """
    Layer Normalization - keeps the numbers in a reasonable range.

    Imagine you're grading tests: some students score 50, others 500.
    Layer norm adjusts them all to a similar scale (like converting to percentages).
    This helps the model learn more stably.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        # dim: size of the layer we're normalizing
        # eps: tiny number to prevent division by zero (safety measure)

        self.eps = eps  # Small epsilon value for numerical stability

        # Learnable parameters (the model adjusts these during training)
        self.gamma = np.ones(dim)   # Scale parameter (like a volume knob)
        self.beta = np.zeros(dim)   # Shift parameter (like a brightness adjustment)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize the input to have mean=0 and variance=1, then scale and shift.

        Steps:
        1. Calculate average (mean)
        2. Calculate spread (variance)
        3. Normalize (subtract mean, divide by std)
        4. Scale and shift with learned parameters
        """
        # x shape: (batch_size, seq_len, embed_dim)

        # Calculate mean across the last dimension (embed_dim)
        # Think: "What's the average value for this token?"
        mean = np.mean(x, axis=-1, keepdims=True)

        # Calculate variance (how spread out the numbers are)
        # Think: "How much do values differ from the average?"
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize: subtract mean and divide by standard deviation
        # This centers the data around 0 with spread of 1
        # Like converting test scores to z-scores in statistics
        x_norm = (x - mean) / np.sqrt(variance + self.eps)

        # Apply learned scale (gamma) and shift (beta)
        # The model learns the best scale and shift during training
        output = self.gamma * x_norm + self.beta

        return output


# =============================================================================
# PART 3: MULTI-HEAD ATTENTION
# This is the "brain" that learns which words to pay attention to
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism - the core of transformers.

    Think of this like having multiple experts (heads) read a sentence.
    Each expert focuses on different relationships between words.
    - Expert 1 might focus on grammar
    - Expert 2 might focus on meaning
    - Expert 3 might focus on context

    Then we combine all their insights!
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        # embed_dim: size of word vectors (must be divisible by n_heads)
        # n_heads: number of attention heads (parallel processors)
        # dropout: probability of dropping connections (for regularization)

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dropout = dropout

        # Each head processes a smaller portion of the embedding
        # Like dividing work among team members
        self.head_dim = embed_dim // n_heads

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        # Weight matrices for transforming input into Query, Key, Value
        # Think of these as different "lenses" to view the same text
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02  # Query: "what am I looking for?"
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02  # Key: "what do I contain?"
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02  # Value: "what information do I have?"

        # Output projection - combines all heads back together
        self.W_o = np.random.randn(embed_dim, embed_dim) * 0.02

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the embedding into multiple heads.

        Like dividing a team of 12 people into 3 groups of 4.
        Each group (head) works on a smaller piece of the problem.
        """
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # Reshape to separate heads
        # (batch, seq_len, embed_dim) -> (batch, seq_len, n_heads, head_dim)
        x = x.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose to put heads dimension first
        # (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
        # This makes it easier to process each head independently
        x = x.transpose(0, 2, 1, 3)

        return x

    def merge_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Merge multiple heads back into a single representation.

        Like combining reports from different team members into one final report.
        """
        # x shape: (batch_size, n_heads, seq_len, head_dim)
        batch_size, _, seq_len, _ = x.shape

        # Transpose back
        # (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, n_heads, head_dim)
        x = x.transpose(0, 2, 1, 3)

        # Merge heads
        # (batch, seq_len, n_heads, head_dim) -> (batch, seq_len, embed_dim)
        x = x.reshape(batch_size, seq_len, self.embed_dim)

        return x

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        The main attention computation.

        Process:
        1. Create Q, K, V from input
        2. Calculate attention scores (which words are related)
        3. Apply attention to values
        4. Combine multiple heads
        5. Project to output
        """
        batch_size, seq_len, _ = x.shape

        # STEP 1: Project input to Q, K, V
        # Think: Transform the text through different "lenses"
        Q = x @ self.W_q  # Query: What am I looking for?
        K = x @ self.W_k  # Key: What do I contain?
        V = x @ self.W_v  # Value: What info do I have?

        # STEP 2: Split into multiple heads
        Q = self.split_heads(Q)  # (batch, n_heads, seq_len, head_dim)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # STEP 3: Calculate attention scores
        # How much should each word pay attention to every other word?
        # Like calculating relevance scores between all pairs of words

        # Q @ K^T gives us similarity scores
        scores = Q @ K.transpose(0, 1, 3, 2)  # (batch, n_heads, seq_len, seq_len)

        # Scale by sqrt(head_dim) to keep numbers reasonable
        # Without this, scores get too large and gradients vanish
        scores = scores / np.sqrt(self.head_dim)

        # STEP 4: Apply mask (if provided)
        # Mask prevents looking at future words (for autoregressive generation)
        # Think: Don't cheat by looking ahead!
        if mask is not None:
            scores = scores + mask  # mask contains large negative values for forbidden positions

        # STEP 5: Softmax to get attention weights (probabilities)
        # Convert scores to probabilities that sum to 1
        # Like converting test scores to percentages
        attention_weights = self.softmax(scores)

        # STEP 6: Apply dropout (randomly zero out some connections during training)
        # This prevents overfitting
        # (In real implementation, only during training)

        # STEP 7: Apply attention to values
        # Weighted average of values based on attention weights
        # Think: Focus more on important words, less on unimportant ones
        output = attention_weights @ V  # (batch, n_heads, seq_len, head_dim)

        # STEP 8: Merge heads back together
        output = self.merge_heads(output)  # (batch, seq_len, embed_dim)

        # STEP 9: Final projection
        output = output @ self.W_o

        return output

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax function - converts scores to probabilities.

        Like normalizing scores so they sum to 100%.
        """
        # Subtract max for numerical stability (prevents overflow)
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)

        # Divide by sum to get probabilities
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# =============================================================================
# PART 4: FEEDFORWARD NETWORK
# This processes each token independently with a 2-layer neural network
# =============================================================================

class FeedForward:
    """
    Position-wise Feedforward Network.

    This is like a small neural network that processes each word independently.
    Think of it as: expand -> activate -> compress

    It gives the model extra "thinking" capacity after attention.
    """

    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: float = 0.1):
        # embed_dim: input/output size
        # feedforward_dim: hidden layer size (usually 4x embed_dim)
        # dropout: regularization

        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.dropout = dropout

        # First layer: expand from embed_dim to feedforward_dim
        # Think: Give the model more room to "think"
        self.W1 = np.random.randn(embed_dim, feedforward_dim) * 0.02
        self.b1 = np.zeros(feedforward_dim)

        # Second layer: compress back to embed_dim
        # Think: Summarize the thinking back to original size
        self.W2 = np.random.randn(feedforward_dim, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        """
        GELU (Gaussian Error Linear Unit) activation function.

        This is a smooth activation that works better than ReLU for transformers.
        Think of it as a smooth on/off switch (not a hard cut like ReLU).
        """
        # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Process input through 2-layer network.

        Flow: input -> expand -> activate -> compress -> output
        """
        # First layer: expand and activate
        # x @ W1 expands from embed_dim to feedforward_dim
        # Think: Give model more space to process information
        hidden = x @ self.W1 + self.b1

        # Apply GELU activation
        # This adds non-linearity (lets model learn complex patterns)
        hidden = self.gelu(hidden)

        # Apply dropout (during training only)
        # Randomly zero out some neurons to prevent overfitting
        # (In real implementation, only during training)

        # Second layer: compress back to original size
        # hidden @ W2 compresses from feedforward_dim back to embed_dim
        output = hidden @ self.W2 + self.b2

        return output


# =============================================================================
# PART 5: TRANSFORMER BLOCK
# Combines attention + feedforward with residual connections and layer norm
# =============================================================================

class TransformerBlock:
    """
    A single transformer block - the building block of GPT.

    Structure:
    1. Multi-head attention (with residual connection)
    2. Layer normalization
    3. Feedforward network (with residual connection)
    4. Layer normalization

    Think of this as one "layer" of thinking. GPT-2 has 12 of these stacked.
    """

    def __init__(self, config: GPTConfig):
        # config: contains all the hyperparameters

        # Multi-head attention layer
        self.attention = MultiHeadAttention(
            embed_dim=config.embed_dim,
            n_heads=config.n_heads,
            dropout=config.dropout
        )

        # Layer normalization after attention
        self.ln1 = LayerNorm(config.embed_dim)

        # Feedforward network
        self.ffn = FeedForward(
            embed_dim=config.embed_dim,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout
        )

        # Layer normalization after feedforward
        self.ln2 = LayerNorm(config.embed_dim)

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process input through the transformer block.

        Uses "residual connections" - we add the input back to the output.
        Think: "Keep the original information and add new insights"
        """
        # ATTENTION BLOCK with residual connection
        # 1. Apply attention
        attn_output = self.attention.forward(x, mask)

        # 2. Add residual (the original x) - this is crucial!
        # Think: "Keep what we knew + add what we learned"
        x = x + attn_output

        # 3. Normalize
        x = self.ln1.forward(x)

        # FEEDFORWARD BLOCK with residual connection
        # 4. Apply feedforward network
        ffn_output = self.ffn.forward(x)

        # 5. Add residual again
        x = x + ffn_output

        # 6. Normalize
        x = self.ln2.forward(x)

        return x


# =============================================================================
# PART 6: TOKEN EMBEDDING
# Converts token IDs to dense vectors
# =============================================================================

class TokenEmbedding:
    """
    Token Embedding Layer - converts token IDs to dense vectors.

    Think of this like a dictionary:
    - Token ID 42 (word "cat") -> vector [0.2, -0.5, 0.1, ..., 0.3]
    - Token ID 156 (word "dog") -> vector [0.1, -0.3, 0.4, ..., -0.2]

    Similar words get similar vectors.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        # vocab_size: how many different tokens we have
        # embed_dim: size of each vector

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Create embedding matrix
        # Each row is the vector for one token
        # Shape: (vocab_size, embed_dim)
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.02

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Convert token IDs to embedding vectors.

        This is just a lookup operation - like looking up words in a dictionary.
        """
        # token_ids shape: (batch_size, seq_len)
        # output shape: (batch_size, seq_len, embed_dim)

        # Simple lookup: for each token ID, get its embedding vector
        # In numpy, this is just indexing
        embeddings = self.embeddings[token_ids]

        return embeddings


# =============================================================================
# PART 7: POSITIONAL ENCODING
# Adds position information to embeddings
# =============================================================================

class PositionalEncoding:
    """
    Positional Encoding - tells the model WHERE each word is in the sentence.

    Since attention doesn't have inherent order (it's like a bag of words),
    we need to add position information.

    Think: "This is word #1, this is word #2, etc."
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        # max_seq_len: maximum sequence length we support
        # embed_dim: size of embeddings (must match token embeddings)

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Create learnable position embeddings
        # Each position gets its own vector
        # Shape: (max_seq_len, embed_dim)
        self.position_embeddings = np.random.randn(max_seq_len, embed_dim) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional information to token embeddings.

        We simply add position vectors to token vectors.
        Think: token_meaning + position_info = full_representation
        """
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape

        # Get position embeddings for this sequence length
        # positions shape: (seq_len, embed_dim)
        positions = self.position_embeddings[:seq_len, :]

        # Add position embeddings to input
        # Broadcasting automatically handles batch dimension
        # Think: Each word gets its token meaning + position meaning
        output = x + positions

        return output


# =============================================================================
# PART 8: COMPLETE GPT MODEL
# Puts everything together!
# =============================================================================

class GPT:
    """
    Complete GPT Model - This is it! The full architecture!

    Architecture flow:
    1. Token IDs -> Token Embeddings
    2. Add Positional Encodings
    3. Pass through N transformer blocks
    4. Layer normalization
    5. Project to vocabulary (predict next token)

    This is the SAME architecture as GPT-2/GPT-3 (just smaller).
    """

    def __init__(self, config: GPTConfig):
        # config: contains all hyperparameters

        self.config = config

        # Token embedding layer
        # Converts token IDs (integers) to vectors
        self.token_embedding = TokenEmbedding(config.vocab_size, config.embed_dim)

        # Positional encoding layer
        # Adds position information
        self.positional_encoding = PositionalEncoding(config.max_seq_len, config.embed_dim)

        # Stack of transformer blocks
        # This is the "depth" of the network - more blocks = deeper thinking
        self.transformer_blocks = [
            TransformerBlock(config) for _ in range(config.n_layers)
        ]

        # Final layer normalization
        self.ln_f = LayerNorm(config.embed_dim)

        # Output projection layer
        # Projects from embed_dim to vocab_size (predicts next token)
        # This is the "head" that makes predictions
        self.output_projection = np.random.randn(config.embed_dim, config.vocab_size) * 0.02

        print(f"GPT Model created with {self.count_parameters():,} parameters")

    def create_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        Create causal mask to prevent looking at future tokens.

        This makes the model "autoregressive" - it can only look at past tokens.
        Think: "Don't cheat by looking ahead!"

        Returns a mask where:
        - 0 means "allowed to attend"
        - -inf means "forbidden to attend"
        """
        # Create a matrix that's 1 above diagonal, 0 below
        # Example for seq_len=4:
        # [[0, 1, 1, 1],
        #  [0, 0, 1, 1],
        #  [0, 0, 0, 1],
        #  [0, 0, 0, 0]]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)

        # Convert 1s to -inf (forbidden), 0s stay 0 (allowed)
        mask = mask * -1e9

        return mask

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass through the entire GPT model.

        This is where the magic happens!

        Input: token IDs (integers)
        Output: logits for next token prediction
        """
        # token_ids shape: (batch_size, seq_len)
        batch_size, seq_len = token_ids.shape

        # STEP 1: Token embedding
        # Convert token IDs to dense vectors
        # Think: "Look up what each word means"
        x = self.token_embedding.forward(token_ids)  # (batch, seq_len, embed_dim)

        # STEP 2: Add positional encoding
        # Add position information
        # Think: "Add where each word is located"
        x = self.positional_encoding.forward(x)  # (batch, seq_len, embed_dim)

        # STEP 3: Create causal mask
        # Prevent looking at future tokens
        mask = self.create_causal_mask(seq_len)

        # STEP 4: Pass through transformer blocks
        # Each block adds a layer of understanding
        # Think: "Process the text through multiple layers of thinking"
        for block in self.transformer_blocks:
            x = block.forward(x, mask)  # (batch, seq_len, embed_dim)

        # STEP 5: Final layer normalization
        # Stabilize the outputs
        x = self.ln_f.forward(x)  # (batch, seq_len, embed_dim)

        # STEP 6: Project to vocabulary
        # Convert embeddings to logits (unnormalized probabilities)
        # for each token in vocabulary
        # Think: "What's the probability of each word being next?"
        logits = x @ self.output_projection  # (batch, seq_len, vocab_size)

        return logits

    def count_parameters(self) -> int:
        """
        Count total number of parameters in the model.

        This tells us how big our model is.
        More parameters = more capacity but slower and more memory.
        """
        total = 0

        # Token embeddings
        total += self.token_embedding.embeddings.size

        # Positional embeddings
        total += self.positional_encoding.position_embeddings.size

        # Transformer blocks
        for block in self.transformer_blocks:
            # Attention weights
            total += block.attention.W_q.size
            total += block.attention.W_k.size
            total += block.attention.W_v.size
            total += block.attention.W_o.size

            # Layer norm parameters
            total += block.ln1.gamma.size + block.ln1.beta.size

            # Feedforward weights
            total += block.ffn.W1.size + block.ffn.b1.size
            total += block.ffn.W2.size + block.ffn.b2.size

            # Layer norm parameters
            total += block.ln2.gamma.size + block.ln2.beta.size

        # Final layer norm
        total += self.ln_f.gamma.size + self.ln_f.beta.size

        # Output projection
        total += self.output_projection.size

        return total


# =============================================================================
# PART 9: DEMONSTRATION
# Let's create and test a small GPT model!
# =============================================================================

def main():
    """
    Main function to demonstrate building and using a GPT model.
    """
    print("=" * 80)
    print("Building a Complete GPT Model from Scratch!")
    print("=" * 80)
    print()

    # STEP 1: Create configuration
    # Let's make a small GPT for demonstration (much smaller than GPT-2)
    print("STEP 1: Creating Model Configuration")
    print("-" * 80)
    config = GPTConfig(
        vocab_size=1000,      # Small vocabulary (GPT-2 uses 50,257)
        max_seq_len=128,      # Short sequences (GPT-2 uses 1024)
        embed_dim=256,        # Small embeddings (GPT-2 uses 768)
        n_layers=4,           # Few layers (GPT-2 uses 12)
        n_heads=4,            # Few heads (GPT-2 uses 12)
        dropout=0.1,
        feedforward_dim=1024  # 4x embed_dim
    )
    print(f"Configuration: {config}")
    print()

    # STEP 2: Create the model
    print("STEP 2: Creating GPT Model")
    print("-" * 80)
    model = GPT(config)
    print()

    # STEP 3: Prepare sample input
    print("STEP 3: Preparing Sample Input")
    print("-" * 80)
    # Create fake token IDs (in real use, these come from a tokenizer)
    batch_size = 2
    seq_len = 10
    token_ids = np.random.randint(0, config.vocab_size, size=(batch_size, seq_len))
    print(f"Input shape: {token_ids.shape} (batch_size={batch_size}, seq_len={seq_len})")
    print(f"Sample token IDs: {token_ids[0]}")
    print()

    # STEP 4: Forward pass
    print("STEP 4: Running Forward Pass")
    print("-" * 80)
    logits = model.forward(token_ids)
    print(f"Output shape: {logits.shape} (batch_size, seq_len, vocab_size)")
    print(f"Output shape breakdown:")
    print(f"  - batch_size: {logits.shape[0]} (number of sequences)")
    print(f"  - seq_len: {logits.shape[1]} (number of tokens in each sequence)")
    print(f"  - vocab_size: {logits.shape[2]} (probability for each word in vocabulary)")
    print()

    # STEP 5: Show what the output means
    print("STEP 5: Understanding the Output")
    print("-" * 80)
    # Get predictions for the last token of the first sequence
    last_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

    # Convert logits to probabilities
    probabilities = np.exp(last_token_logits) / np.sum(np.exp(last_token_logits))

    # Get top 5 predictions
    top_5_indices = np.argsort(probabilities)[-5:][::-1]

    print("Top 5 predicted tokens for next word:")
    for i, idx in enumerate(top_5_indices, 1):
        print(f"  {i}. Token {idx}: {probabilities[idx]:.4f} probability")
    print()

    # STEP 6: Model statistics
    print("STEP 6: Model Statistics")
    print("-" * 80)
    num_params = model.count_parameters()
    print(f"Total parameters: {num_params:,}")
    print(f"Model size (approx): {num_params * 4 / 1024 / 1024:.2f} MB (assuming 4 bytes per parameter)")
    print()

    # Comparison with GPT-2
    print("STEP 7: Comparison with GPT-2")
    print("-" * 80)
    print("Our model vs GPT-2 Small:")
    print(f"  Vocabulary: {config.vocab_size:,} vs 50,257")
    print(f"  Sequence Length: {config.max_seq_len} vs 1,024")
    print(f"  Embedding Dim: {config.embed_dim} vs 768")
    print(f"  Layers: {config.n_layers} vs 12")
    print(f"  Heads: {config.n_heads} vs 12")
    print(f"  Parameters: {num_params:,} vs 124,000,000")
    print()
    print("Our model is much smaller, but uses THE SAME ARCHITECTURE!")
    print("To match GPT-2, just change the config values!")
    print()

    print("=" * 80)
    print("SUCCESS! You've built a complete GPT model from scratch!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Learn how to generate text (Lesson 2)")
    print("2. Train the model on real data (Lesson 3)")
    print("3. Fine-tune for specific tasks (Lesson 4)")


if __name__ == "__main__":
    main()

# Lesson 02: The Router Network

## What Is the Router?

The router is the "brain" of MoE -- it decides which experts handle each token.

Architecture: just ONE Linear layer, followed by softmax:

```
router_input  : shape [batch, seq_len, d_model]   (token embeddings)
router_weights: shape [d_model, num_experts]       (learned parameters)
logits        : shape [batch, seq_len, num_experts] (raw scores)
probs         : shape [batch, seq_len, num_experts] (softmax probabilities)
```

The router is tiny compared to the experts. For d_model=1024, num_experts=8:
- Router params: 1024 * 8 = 8,192 parameters
- Each expert params: 1024 * 4096 * 2 = 8,388,608 parameters
- Router is 1000x smaller than the experts it controls.

---

## Step-by-Step: How the Router Works

### Step 1: Compute raw logits

```
logits = token_embedding @ W_router    # matrix multiply
# shape: [batch, seq_len, num_experts]
# W_router shape: [d_model, num_experts]
```

Each expert gets a raw score (logit) for this token.
Higher logit = router "thinks" this expert is better for this token.

### Step 2: Softmax -> probabilities

```
probs = softmax(logits, dim=-1)
# shape: [batch, seq_len, num_experts]
# each token's probs sum to 1.0 across all experts
```

Now each expert has a probability between 0 and 1 for each token.

### Step 3: Top-K selection

```
topk_values, topk_indices = topk(probs, k=2)
# topk_values:  shape [batch, seq_len, 2]  -- the top 2 probabilities
# topk_indices: shape [batch, seq_len, 2]  -- which experts (0..7)
```

We keep only the K highest probabilities. The other N-K experts are discarded.

### Step 4: Renormalize the top-K weights

```
topk_weights = softmax(topk_values, dim=-1)
# Now the 2 selected weights sum to 1.0 (not the original small values)
```

Why renormalize? Because we dropped N-K experts, the remaining K weights no
longer sum to 1. We renormalize so the weighted sum makes sense as a weighted
average.

### Step 5: Weighted combination

```
output = sum(topk_weights[i] * Expert_i(token) for i in topk_indices)
```

If token routes to Expert 2 (weight 0.7) and Expert 5 (weight 0.3):
```
output = 0.7 * Expert2(token) + 0.3 * Expert5(token)
```

---

## ASCII Diagram: Router Architecture

```
                          d_model = 512
                              |
                     [token embedding]
                          |
           +-- Linear(d_model, num_experts=8) --+
           |                                    |
           v                                    v
    [raw logits: shape (8,)]          W_router: (512, 8) params
           |
     softmax(dim=-1)
           |
    [probs: shape (8,)]  -- e.g. [0.01, 0.02, 0.40, 0.35, 0.08, 0.05, 0.07, 0.02]
           |
       top-K=2
           |
    +------+------+
    |              |
  idx=2          idx=3             (experts 0,1,4,5,6,7 are SKIPPED)
  prob=0.40     prob=0.35
           |
    renormalize: [0.533, 0.467]   (must sum to 1.0 again)
           |
  +--------+--------+
  |                 |
Expert2(token)   Expert3(token)
  |                 |
  *0.533           *0.467
  |                 |
  +--------+--------+
           |
     final output
```

---

## Two Ways to Think About Routing

### Token-Choice Routing (standard, used above)
Each TOKEN decides which K experts it wants.
- Token picks top-K experts based on router probs.
- Simple, used in Mixtral and most MoE models.
- Problem: popular experts can be overwhelmed by too many tokens.

### Expert-Choice Routing (alternative)
Each EXPERT decides which tokens it wants to process.
- Expert picks top-C tokens to process (C = capacity).
- Solves overload automatically.
- Problem: some tokens might not be picked by ANY expert.

This module uses Token-Choice routing (simpler, more common).

---

## C# Load Balancer Analogy

The router is like a **content-aware HTTP load balancer**:

```csharp
// Standard round-robin load balancer (not like MoE router)
public class RoundRobinBalancer {
    private int counter = 0;
    public Server Route(Request req) => servers[counter++ % servers.Length];
}

// MoE-style learned load balancer
public class LearnedRouter {
    // W_router is a learned weight matrix (like a neural network layer)
    private float[,] W_router;  // shape: [d_model, num_experts]

    public int[] Route(float[] tokenEmbedding, int k) {
        // matrix multiply to get scores for each expert
        float[] scores = MatMul(tokenEmbedding, W_router);

        // softmax to get probabilities
        float[] probs = Softmax(scores);

        // pick top-K experts
        return TopK(probs, k);
    }
}
```

The difference from a real load balancer:
- A real load balancer uses fixed rules (round-robin, least-connections).
- The MoE router LEARNS from data which expert to pick for each token type.
- The router is trained WITH the experts -- they co-evolve.

---

## What Does the Router Learn?

The router learns to associate token "features" (embedding patterns) with
specific experts. After training:

- Tokens about mathematics might cluster toward Expert 3.
- Code-related tokens might go to Expert 7.
- Common English words might spread across Expert 1 and Expert 4.

No human tells the router this. It discovers it automatically during training
because routing to a specialized expert produces lower loss.

---

## Router Noise (Jitter for Exploration)

A common trick: add small random noise to the router logits during training.

```python
noise = torch.randn_like(logits) * (1.0 / num_experts)
logits = logits + noise
```

Why? Without noise, the router quickly settles into always picking the same
experts. Noise forces exploration -- the router tries different experts and
can discover better specializations.

This is similar to epsilon-greedy exploration in reinforcement learning (M13).

---

## Summary

| Step       | Operation                    | Shape change                         |
|------------|------------------------------|--------------------------------------|
| Input      | token embedding              | [batch, seq, d_model]                |
| Linear     | @ W_router                   | [batch, seq, d_model] -> [batch, seq, num_experts] |
| Softmax    | over last dim                | [batch, seq, num_experts] (sums to 1)|
| Top-K      | keep K highest               | [batch, seq, K] values + indices     |
| Renorm     | softmax over K values        | [batch, seq, K] (sums to 1)          |
| Experts    | run K expert FFNs            | K x [batch, seq, d_model]            |
| Combine    | weighted sum                 | [batch, seq, d_model]                |

---

## Quiz Questions

**Question 1**: The router is a Linear layer mapping d_model -> num_experts.
For d_model=1024 and num_experts=8, how many learnable parameters does the
router have? (Ignore bias for simplicity.)

A) 8
B) 1,024
C) 8,192  <-- CORRECT
D) 1,048,576

**Explanation**: Linear(d_model, num_experts) has d_model * num_experts weights.
1024 * 8 = 8,192. This is tiny compared to each expert's ~8 million parameters.

---

**Question 2**: Why do we renormalize the top-K router weights with a second softmax?

A) To make the weights negative so experts compete.
B) Because after dropping N-K experts, the K remaining weights no longer sum to 1,
   and we need them to sum to 1 for a valid weighted average.  <-- CORRECT
C) To prevent gradient explosion during backpropagation.
D) The second softmax makes the router sharper and more decisive.

**Explanation**: Original probs sum to 1 across ALL N experts. After dropping
N-K experts, the remaining K values sum to less than 1. We softmax again so
the weighted combination of expert outputs is a proper weighted average.

---

**Question 3**: What is the difference between "token-choice" and "expert-choice" routing?

A) Token-choice is for language models; expert-choice is for vision models.
B) In token-choice, each token picks K experts. In expert-choice, each expert
   picks C tokens it wants to process.  <-- CORRECT
C) Token-choice uses softmax; expert-choice uses argmax.
D) Expert-choice uses K=1 always; token-choice uses K=2.

**Explanation**: Token-choice (used in Mixtral, this module) lets each token
pick its preferred K experts. Expert-choice (Google Switch Transformer v2)
flips this -- each expert picks its preferred tokens, solving overflow but
risking some tokens being unrouted.

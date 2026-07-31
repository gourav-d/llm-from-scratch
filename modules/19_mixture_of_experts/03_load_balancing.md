# Lesson 03: Load Balancing

## The Expert Collapse Problem

Without any special training tricks, MoE models suffer from **expert collapse**:
the router learns to always send ALL tokens to ONE or TWO popular experts.

Why does this happen?

1. Training starts: experts are randomly initialized (all equally "bad").
2. By chance, Expert 1 is slightly better for the first batch.
3. Router assigns more tokens to Expert 1.
4. Expert 1 gets more training signal (more gradient updates).
5. Expert 1 becomes EVEN BETTER than the others.
6. Router assigns even MORE tokens to Expert 1.
7. Repeat: all other experts stop training. They become useless.

This is a positive feedback loop, similar to the "rich get richer" effect.

```
Without load balancing:
Step   0: Expert usage = [25%, 25%, 25%, 25%]   (balanced at start)
Step  50: Expert usage = [60%, 15%, 15%, 10%]   (Expert 1 getting popular)
Step 100: Expert usage = [95%,  2%,  2%,  1%]   (Expert 1 monopolized ALL)
Step 200: Expert usage = [99%,  0%,  0%,  1%]   (COLLAPSED: 3 experts wasted)
```

The result: you paid for 8 experts but effectively trained only 1.

---

## The Switch Transformer Auxiliary Loss

The Switch Transformer paper (Google, 2021) introduced an auxiliary loss
that penalizes unbalanced expert usage. It is the standard approach used today.

### Variables

```
num_experts  = N        (e.g. 8)
num_tokens   = T        (tokens in this batch, e.g. 1024)

f_i  = fraction of tokens routed to expert i
     = (count of tokens that chose expert i) / T
     = "actual usage" -- hard, not differentiable

P_i  = average router PROBABILITY for expert i across all tokens
     = mean of router_probs[:, :, i] over batch and seq dims
     = "soft usage" -- differentiable, can backpropagate through
```

### The Formula

```
aux_loss = num_experts * sum(f_i * P_i)   for i in 0..N-1
```

### Why This Works

- If Expert 1 gets all tokens: f_1 = 1.0, all other f_i = 0.
  Router also gives high probability to Expert 1: P_1 is large.
  So f_1 * P_1 is large -> aux_loss is large -> penalized.

- If all experts get equal tokens: f_i = 1/N for all i.
  P_i = 1/N for all i.
  sum(f_i * P_i) = sum(1/N * 1/N) = N * (1/N^2) = 1/N.
  aux_loss = N * (1/N) = 1.0  <-- constant, minimum value.

The loss is minimized when expert usage is UNIFORM across all experts.

### Total Training Loss

```
total_loss = main_loss + alpha * aux_loss
```

Where:
- main_loss = standard cross-entropy loss (language model loss)
- alpha = small coefficient, typically 0.01 or 0.001
- aux_loss = load balancing penalty

Alpha must be small! Too large: the model optimizes balance at cost of quality.
Too small: collapse still happens. Typical value: alpha = 0.01.

---

## ASCII Diagram: Effect of Auxiliary Loss

```
Without auxiliary loss:           With auxiliary loss (alpha=0.01):

Expert usage after training:      Expert usage after training:

Expert 1: ████████████████ 96%    Expert 1: ████ 26%
Expert 2: |                2%    Expert 2: ████ 25%
Expert 3: |                1%    Expert 3: ███  24%
Expert 4: |                1%    Expert 4: ████ 25%
```

The auxiliary loss acts like a "fairness regulator" -- it does not care about
individual token routing decisions, only the aggregate statistics.

---

## Expert Capacity

Even with aux_loss, popular experts might receive more tokens than they can
efficiently process in parallel (on GPU hardware).

**Expert capacity** = maximum number of tokens an expert will process per batch.

```
capacity = (total_tokens / num_experts) * capacity_factor
```

Example:
- Batch has 1024 tokens, 8 experts.
- Base capacity = 1024 / 8 = 128 tokens per expert.
- With capacity_factor = 1.25: capacity = 128 * 1.25 = 160 tokens.

If an expert receives 200 tokens but its capacity is 160:
- The top 160 tokens (by router score) are processed by the expert.
- The remaining 40 tokens are "dropped" -- they skip that expert.
- Dropped tokens use a RESIDUAL connection (their output = their input, unchanged).

This is sometimes called "token dropping" and is a deliberate design choice.
Capacity factor = 1.0 means no buffer (strict). Factor = 2.0 means 100% buffer.

---

## C# Connection Pool Analogy

Expert capacity is exactly like a database connection pool:

```csharp
public class ExpertConnectionPool {
    private int maxConnections = 160;  // expert capacity
    private Queue<Token> waitQueue = new Queue<Token>();

    public void ProcessToken(Token token) {
        if (activeConnections < maxConnections) {
            // Expert can handle this token
            ProcessWithExpert(token);
        } else {
            // Expert at capacity -- token is "dropped"
            // Token uses residual connection (output = input)
            UseResidualFallback(token);
        }
    }
}
```

The difference: in MoE, dropped tokens still get SOME output (their input
unchanged via residual), unlike database queries which fail or wait.
The model learns to be robust to occasional token drops.

---

## DeepSeek's Shared Expert Trick

DeepSeek-V3 and other recent models use a clever variation:
keep ONE expert that ALWAYS runs for EVERY token (the "shared expert"),
and route tokens to K additional experts from a routed pool.

```
Standard MoE:    output = sum(w_i * Expert_i(token))  for i in top-K
DeepSeek MoE:    output = SharedExpert(token) + sum(w_i * Expert_i(token))
                          (always runs)                (top-K from 255 routed)
```

The shared expert handles common patterns that all tokens need.
The routed experts handle specialized knowledge.

DeepSeek-V3 has: 1 shared expert + 255 routed experts = 256 total.
Each token uses: 1 (shared, always) + 8 (routed, top-8) = 9 experts active.

---

## Summary of Load Balancing Techniques

| Technique          | Description                          | Used In         |
|--------------------|--------------------------------------|-----------------|
| Auxiliary loss     | Penalize (f_i * P_i)                 | Switch, Mixtral |
| Expert capacity    | Cap tokens per expert, drop overflow | Switch, Mixtral |
| Router noise       | Add noise during training            | Switch, GShard  |
| Shared expert      | One expert always active             | DeepSeek        |
| Expert choice      | Expert picks tokens (not vice versa) | Switch v2       |

---

## Quiz Questions

**Question 1**: Why does expert collapse happen even though all experts start
with the same random initialization?

A) Because PyTorch always initializes Expert 1 with better weights.
B) By random chance, one expert performs slightly better early on, gets more
   training signal, improves faster, and the feedback loop causes collapse.  <-- CORRECT
C) The router always prefers Expert 0 due to softmax returning 0 for others.
D) Expert collapse only happens when K=1 (top-1 routing).

**Explanation**: Expert collapse is a positive feedback loop. A small random
advantage early on compounds -- more tokens, more gradient, better performance,
even more tokens. It happens regardless of K value, though K=1 is worst.

---

**Question 2**: In the Switch Transformer auxiliary loss formula
aux_loss = N * sum(f_i * P_i), what is f_i and why is it NOT directly
used for backpropagation?

A) f_i is the router probability -- it is used directly for backprop.
B) f_i is the actual fraction of tokens routed to expert i. It is not
   differentiable (it depends on a hard top-K selection), so we also
   use P_i (differentiable soft probability) to carry gradients.  <-- CORRECT
C) f_i is the expert output magnitude. It is too noisy for direct backprop.
D) f_i is used for backprop; P_i is not differentiable.

**Explanation**: f_i comes from argmax/topk which has zero gradient everywhere.
We multiply f_i (actual usage) * P_i (soft probability that does have gradient)
to get a term that is correlated with balance AND differentiable through P_i.

---

**Question 3**: A batch has 512 tokens, 4 experts, capacity_factor = 1.5.
What is the maximum number of tokens each expert will process?

A) 128
B) 192  <-- CORRECT
C) 256
D) 512

**Explanation**: Base capacity = 512 / 4 = 128 tokens per expert.
With capacity_factor = 1.5: capacity = 128 * 1.5 = 192 tokens.
Tokens beyond 192 for any single expert are dropped (use residual output).

# Lesson 04: MoE vs Dense -- Trade-offs

## Side-by-Side Comparison

| Property              | Dense FFN                     | MoE FFN (N=8, K=2)               |
|-----------------------|-------------------------------|-----------------------------------|
| Total parameters      | d_ff * d_model * 2            | 8 * d_ff * d_model * 2            |
| Active params/token   | Same as total (100%)          | 2/8 = 25% of total               |
| Memory needed         | 1x (store 1 FFN)              | 8x (store all 8 experts)          |
| Compute per token     | 1x (full FFN)                 | 0.25x (2 of 8 experts)            |
| Training complexity   | Simple                        | Needs aux loss + capacity mgmt    |
| Serving complexity    | Simple                        | Needs expert sharding across GPUs |
| Inference speed       | Fast (all on one GPU)         | Complex (experts may be on diff GPUs) |
| Quality at same compute | Baseline                    | Often better (more total params)  |

---

## When MoE Wins

### 1. Large-Scale Models (>10B parameters)

At small scale, MoE's extra complexity does not pay off.
At large scale (10B+), the quality-per-FLOP advantage becomes decisive.

Example: Training Mixtral 8x7B (~47B params) costs similar FLOPs to a 12B
dense model, but the MoE model quality matches a 70B dense model.

### 2. Inference-Heavy Workloads

If you run inference (prediction) much more than training:
- MoE uses less compute per forward pass.
- Lower inference cost = less expensive API calls.
- DeepSeek-V3 can serve many more requests per GPU than a dense 671B model.

### 3. When Diverse Knowledge Is Needed

MoE experts naturally specialize:
- Code tokens -> code expert
- Math tokens -> math expert
- Medical tokens -> medical expert

This specialization improves quality on diverse tasks compared to one FFN
trying to learn everything.

---

## When Dense Wins

### 1. Small Models (under 1B parameters)

For small models, the MoE overhead (routing, load balancing, expert sharding)
is not worth it. A dense 1B model outperforms a 1B MoE in practice.

### 2. Edge Deployment (Mobile, IoT, Embedded)

On a phone or edge device, you cannot shard experts across multiple GPUs.
You also cannot afford to store all N expert weight matrices in limited memory.
A compact dense model is always better for edge deployment.

### 3. Simplicity Matters

MoE models are harder to:
- Train (aux loss tuning, capacity factor tuning)
- Fine-tune (load balance can collapse during fine-tuning)
- Debug (which expert is causing the problem?)
- Serve (expert parallelism adds networking overhead)

If your team is small or the model does not need to scale, dense is simpler.

### 4. Latency-Sensitive Applications

Expert parallelism requires **all-to-all communication** between GPUs:
each GPU sends tokens to the GPU that holds the relevant experts.
This network communication adds latency. Dense models avoid it entirely.

---

## Expert Parallelism

When serving a large MoE model:

```
GPU 0: holds Expert 0, Expert 1
GPU 1: holds Expert 2, Expert 3
GPU 2: holds Expert 4, Expert 5
GPU 3: holds Expert 6, Expert 7
```

Step 1: All GPUs receive the input tokens.
Step 2: Router runs (on each GPU) -- decides which expert each token needs.
Step 3: **All-to-All** communication: tokens migrate to the GPU holding their expert.
Step 4: Each GPU runs its experts on the tokens it received.
Step 5: **All-to-All** communication back: outputs return to original GPUs.
Step 6: GPUs combine outputs and continue.

The two all-to-all steps add network overhead. At very high scale (thousands
of GPUs), this communication can become a bottleneck.

---

## ASCII Diagram: Expert Parallelism

```
Input tokens: [T1, T2, T3, T4, T5, T6, T7, T8]
     |
  [Router on all GPUs: decides routing]
     |
  T1 -> Expert 1 (GPU 0)    T2 -> Expert 3 (GPU 1)
  T3 -> Expert 0 (GPU 0)    T4 -> Expert 5 (GPU 2)
  T5 -> Expert 2 (GPU 1)    T6 -> Expert 7 (GPU 3)
  T7 -> Expert 1 (GPU 0)    T8 -> Expert 6 (GPU 3)
     |
  [All-to-All: tokens travel to GPU holding their expert]
     |
  GPU 0 runs Expert 0 on T3     GPU 2 runs Expert 5 on T4
  GPU 0 runs Expert 1 on T1,T7  GPU 3 runs Expert 6 on T8
  GPU 1 runs Expert 2 on T5     GPU 3 runs Expert 7 on T6
  GPU 1 runs Expert 3 on T2
     |
  [All-to-All: outputs travel back to original GPUs]
     |
  Output tokens: [O1, O2, O3, O4, O5, O6, O7, O8]
```

---

## C# Analogy: Microservices vs Monolith

Dense model = **monolith application**:
- Everything in one process.
- Simple to deploy and debug.
- Scaling requires scaling the entire application.
- No network calls between components.

MoE model = **microservices architecture**:
- Each expert is like a separate service.
- Better specialization and independent scaling.
- But: service discovery, load balancing, network overhead.
- More ops complexity (like managing 8 separate services instead of 1).

```csharp
// Dense FFN = monolith: one service handles everything
public class MonolithFFN : IFeedForward {
    public Tensor Forward(Tensor x) {
        x = fc1(x);   // everything in one place
        x = gelu(x);
        x = fc2(x);
        return x;
    }
}

// MoE = microservices: specialized services with a router
public class MoEFFN : IFeedForward {
    private IExpert[] experts;        // 8 specialized services
    private IRouter router;           // load balancer / service mesh

    public Tensor Forward(Tensor x) {
        var route = router.GetRoute(x);    // service discovery
        return WeightedCombine(route, x);  // combine service outputs
    }
}
```

---

## Fine-Tuning MoE: Special Challenges

Fine-tuning a pre-trained MoE model is harder than a dense model:

1. **Load balance can break**: if the fine-tuning dataset is narrow (e.g., only
   code), many tokens flow to the "code expert," collapsing others.
   Solution: keep aux_loss active during fine-tuning with the same alpha.

2. **Expert routing changes**: the router, trained on diverse pre-training data,
   may route fine-tuning tokens differently. This can hurt generalization.
   Solution: sometimes freeze the router during fine-tuning.

3. **Memory**: you must load ALL expert weights, even if only 2-3 experts
   are relevant to your fine-tuning task.

4. **LoRA on MoE**: Low-Rank Adaptation (M12) can be applied per-expert,
   but the total number of LoRA adapters = N * num_layers, which is large.

---

## Summary Table: Choose Dense or MoE?

| Scenario                     | Use Dense  | Use MoE    |
|------------------------------|------------|------------|
| Model size < 7B              | YES        | No         |
| Model size > 30B             | No         | YES        |
| Edge / mobile deployment     | YES        | No         |
| Multi-GPU server inference   | Maybe      | YES        |
| Diverse tasks (code+math+...) | Maybe     | YES        |
| Single narrow task            | YES        | Maybe      |
| Team size < 5 engineers      | YES        | No         |
| Production API at scale      | Maybe      | YES        |

---

## Quiz Questions

**Question 1**: A team wants to deploy an LLM on a smartphone with 8GB RAM.
The model must fit in memory and run efficiently. Should they use MoE or dense?

A) MoE -- because it uses less compute per token, saving battery life.
B) Dense  <-- CORRECT
C) MoE with K=1 (switch transformer) to minimize memory.
D) Dense is never better than MoE; use MoE always.

**Explanation**: On edge devices, you cannot shard experts across GPUs.
You must store ALL N expert weight matrices in limited RAM -- even though
only K run per token. A compact dense model uses less total memory.

---

**Question 2**: What is "expert parallelism" and what communication overhead does it require?

A) Experts run in parallel on the same GPU; no overhead.
B) Different experts are stored on different GPUs. Routing requires all-to-all
   communication to send tokens to the right GPU and back.  <-- CORRECT
C) The router runs in parallel with the experts; no overhead.
D) Expert parallelism means running K experts in parallel on K GPUs; uses
   point-to-point (not all-to-all) communication.

**Explanation**: Expert parallelism splits experts across GPUs. Each batch
requires two all-to-all communications: tokens to experts, outputs back.
This is the main latency cost of MoE at serving time.

---

**Question 3**: A dense model and an MoE model both have the SAME number of
ACTIVE parameters per token (K/N ratio makes active params identical). The MoE
has 8x more TOTAL parameters. Which model is likely BETTER at diverse tasks,
and why?

A) The dense model -- more total params = better.
B) The dense model -- MoE has too much complexity.
C) The MoE model -- with 8x more total params, experts can specialize for
   different types of knowledge, even though compute per token is the same.  <-- CORRECT
D) They perform exactly the same because active params are identical.

**Explanation**: Total parameters = capacity to store knowledge. Even though
compute per token is the same, the MoE can store 8x more specialized knowledge
distributed across experts. On diverse tasks, this specialization advantage
means MoE typically outperforms dense at the same compute budget.

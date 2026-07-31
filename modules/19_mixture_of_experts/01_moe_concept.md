# Lesson 01: The MoE Concept

## What Problem Does MoE Solve?

In all the transformers you have built so far (M05, M18), every transformer
block contains a **Feed-Forward Network (FFN)**. This FFN looks like:

```
token_embedding -> Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model) -> output
```

The problem: **every single token goes through the EXACT SAME FFN weights**.

- Token "cat" and token "algorithm" both use the same FFN.
- The FFN has to learn to handle ALL types of tokens at once.
- Making the FFN bigger (more params) helps quality, but also costs MORE compute
  per token -- because every token uses every weight.

**The core question**: Can we make the model BIGGER (more total parameters)
without making it SLOWER (same compute per token)?

Answer: YES -- with Mixture of Experts.

---

## The Core MoE Idea

Instead of ONE large FFN, use **N smaller FFN layers** called **experts**.
Then use a **router** to send each token to only **K** of the N experts.

```
                    Standard Dense FFN
                    ------------------
Token  -----------> [   FFN layer   ] -----------> output
       (all tokens use the same FFN weights)


                    Mixture of Experts FFN
                    ----------------------
              .--> [Expert 1] --.
              |                  |
Token --> Router --> [Expert 2] --+--> weighted sum --> output
              |                  |
              '-x- [Expert 3]   -'   (skipped, not used)
                -x- [Expert 4]       (skipped, not used)
```

In this diagram:
- The router looks at the token and picks 2 of the 4 experts (Top-K, K=2).
- Experts 3 and 4 are SKIPPED for this token.
- The output is a weighted average of Expert 1 and Expert 2 outputs.
- A different token might route to Expert 2 and Expert 3 instead.

---

## Conditional Computation

The key concept is called **conditional computation**:

- All N experts exist and have parameters stored in memory.
- But only K experts RUN per token (only K sets of parameters are used).
- N-K experts are "idle" -- they have weights but do not compute anything.

Think of it like this:
- **Dense FFN**: you have 1 chef who cooks EVERY dish. Making them better
  at all dishes = more training = more compute every time they cook.
- **MoE FFN**: you have 8 specialist chefs (experts). Each dish goes to
  only 2 chefs who specialize in that type. The other 6 rest.

---

## C# Factory Analogy

Imagine a factory with N assembly lines (N experts):

```csharp
// Dense model: ONE assembly line handles all products
// Making it better requires upgrading the whole line
public class DenseFactory {
    private AssemblyLine line = new AssemblyLine();  // one big FFN
    public Product Process(Product input) => line.Process(input);
}

// MoE model: N assembly lines, dispatcher routes each product to K lines
public class MoEFactory {
    private AssemblyLine[] experts = new AssemblyLine[8];    // 8 expert FFNs
    private Dispatcher router = new Dispatcher();            // router
    public Product Process(Product input) {
        int[] chosenLines = router.PickBest(input, k: 2);   // top-2 routing
        // only 2 of 8 assembly lines run -- 6 are idle
        return WeightedCombine(chosenLines, input);
    }
}
```

The dispatcher (router) is learned from data -- it figures out which assembly
lines (experts) are best for each type of product (token) automatically.

---

## Real-World Scale Numbers

### Mixtral 8x7B
- **Architecture**: 8 experts, Top-K = 2
- **Total parameters**: ~47 billion
- **Active parameters per token**: ~13 billion
- **Compute saving**: only 2 of 8 experts run (25% of experts active)
- **Quality**: matches or beats a dense 13B model on many benchmarks

### DeepSeek-V3
- **Architecture**: 256 experts (1 shared + 255 routed), Top-K = 8
- **Total parameters**: 671 billion
- **Active parameters per token**: 37 billion
- **Compute saving**: 671B total / 37B active = 18x fewer FLOPs vs dense
- **Quality**: world-class performance at dramatically lower inference cost

### Why This Matters

Dense 671B model: IMPOSSIBLE to run without hundreds of GPUs.
DeepSeek-V3 (671B MoE): runs practically because only 37B params activate per token.

---

## The Specialization Effect

A key benefit: experts naturally SPECIALIZE during training.

After training, you might find:
- Expert 1: better at mathematical tokens ("sum", "+", "42")
- Expert 2: better at code tokens ("def", "class", "return")
- Expert 3: better at language tokens ("the", "a", "of")
- Expert 4: better at scientific tokens ("molecule", "quantum", "RNA")

The router LEARNS this specialization automatically -- no human labels needed.

---

## Key Formula: Total vs Active Parameters

```
Dense FFN:
    total_params  = d_model * d_ff * 2  (two linear layers)
    active_params = d_model * d_ff * 2  (always fully used)
    active_ratio  = 100%

MoE FFN (N experts, top-K):
    total_params  = N * (d_model * d_ff * 2)
    active_params = K * (d_model * d_ff * 2)
    active_ratio  = K / N

Example: N=8, K=2, d_model=1024, d_ff=4096
    total_params  = 8 * (1024 * 4096 * 2) = 67 million
    active_params = 2 * (1024 * 4096 * 2) = 17 million
    active_ratio  = 2/8 = 25%
    --> 4x more total params, same compute as a single expert
```

---

## What You Will See in the Code

In the example files you will build:

1. An **Expert class** -- just an FFN: `Linear -> GELU -> Linear`
2. A **Router class** -- one Linear layer that picks which experts to use
3. A **MoELayer class** -- router + N experts combined
4. A **MoETransformerBlock** -- attention + MoELayer (replaces the FFN block)

---

## Summary

| Concept            | Dense FFN            | MoE FFN                         |
|--------------------|----------------------|---------------------------------|
| Structure          | 1 FFN layer          | N expert FFN layers + 1 router  |
| Token processing   | ALL tokens use 1 FFN | Each token uses K of N experts  |
| Total params       | d_model * d_ff * 2   | N * d_model * d_ff * 2          |
| Active params/token| 100% of total        | K/N of total                    |
| Training difficulty| Simple               | Needs load balancing            |
| Specialization     | None                 | Experts learn domains           |

---

## Quiz Questions

**Question 1**: A dense FFN has d_model=512, d_ff=2048.
An MoE layer has 8 experts with the same per-expert dimensions (d_model=512, d_ff=2048).
Top-K = 2. By what factor does the MoE have MORE total parameters?

A) 2x
B) 4x
C) 8x  <-- CORRECT
D) 16x

**Explanation**: 8 experts means 8x total params. Active params = 2/8 = 25% of total.
The compute per token stays the same as 2 dense FFNs (K=2 experts run).

---

**Question 2**: What is "conditional computation" in the context of MoE?

A) The model computes differently based on if-else conditions in code.
B) Parameters exist in memory but are only USED for some inputs, not all.  <-- CORRECT
C) The computation is conditional on the batch size.
D) Only the router uses conditional logic; experts always run.

**Explanation**: Conditional computation means some parameters exist but are not
always activated. N-K experts have zero gradient for any given token.

---

**Question 3**: DeepSeek-V3 has 671B total params and 37B active params per token.
Compared to a hypothetical dense 671B model, roughly how many fewer FLOPs does
DeepSeek-V3 use per forward pass?

A) 2x fewer
B) 5x fewer
C) 18x fewer  <-- CORRECT
D) 671x fewer

**Explanation**: 671B / 37B = approximately 18x. This is why MoE enables building
very large models that are still practical to run at inference time.

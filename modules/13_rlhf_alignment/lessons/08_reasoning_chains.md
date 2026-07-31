# Lesson 08: Reasoning Chains — How Thinking Emerges from RL

## Glossary (Read This First!)

| Term | Plain English Definition | C# Analogy |
|------|--------------------------|------------|
| **Reasoning chain** | The model's step-by-step thought process written out before giving the final answer. Also called chain-of-thought (CoT). | Like writing pseudocode and comments before writing the actual implementation. |
| **Extended thinking** | Very long reasoning chains — sometimes thousands of tokens — where the model explores multiple approaches, backtracks, and verifies. | Like a developer spending an hour whiteboarding before writing a single line of code. |
| **Emergent behavior** | A capability the model was NOT explicitly trained to do, but developed on its own through training. | Like a neural network learning to detect edges without being told what edges are. |
| **Aha moment** | DeepSeek's term for when R1-Zero started spontaneously writing reasoning in its responses, even though it was only rewarded for correct answers. | Like a developer discovering that writing unit tests first (TDD) makes them faster, without being told to use TDD. |
| **Chain-of-thought (CoT)** | A prompting technique (Lesson 07 in M08) where you ask the model to "think step by step." In RLVR, CoT emerges WITHOUT prompting. | Like adding debug logging statements — helps find bugs, not part of the final output. |
| **Test-time compute** | Spending more computation at inference time (generating more tokens) to get better answers. More thinking = more tokens = better accuracy. | Like running more Monte Carlo simulations at query time to get a better estimate — trade compute for accuracy. |
| **Format reward** | A small bonus reward for using the required format (e.g., `<think>...</think><answer>...</answer>`). Encourages the model to structure its thinking. | Like a linter rule requiring XML documentation comments — you get a bonus for following the standard. |
| **Backtracking** | During a reasoning chain, recognizing a wrong approach and switching to a different one. "Wait, that's wrong. Let me try a different approach." | Like a `continue` or `goto` in a search algorithm that abandons a dead-end branch. |
| **Self-verification** | The model checking its own answer at the end of the reasoning chain. "Let me verify: 12 × 12 = 144. Yes, correct." | Like adding an assertion after a computation: `Debug.Assert(result == expected)`. |
| **Scaling test-time compute** | The observation that larger reasoning budgets (more tokens) consistently produce better answers. You can "buy" accuracy with compute. | Like increasing iteration count in a numerical solver — more iterations = better approximation, costs more CPU. |

---

## How It Connects

```
Lesson 05 (Constitutional AI)  -- model critiques its OWN output
Lesson 06 (RLVR)               -- reward from correctness, not humans
Lesson 07 (GRPO)               -- algorithm to train with group rewards
         |
         | Combine: train with GRPO + RLVR rewards
         | Add format reward for using <think> tags
         v
Lesson 08 (Reasoning Chains)   -- what emerges from this training
  -- Extended step-by-step reasoning
  -- Backtracking when wrong
  -- Self-verification at end
  -- Test-time compute scaling
```

---

## The Discovery: Reasoning Was Not Programmed In

This is one of the most surprising findings in recent AI research.

When DeepSeek trained R1-Zero using only RLVR + GRPO (math problems, binary reward):
- They did NOT tell the model to "think step by step"
- They did NOT give it reasoning examples
- They did NOT design any special architecture for reasoning

The model started writing long reasoning chains **on its own**.

Why? Because thinking before answering **helps get the right answer**.
Getting the right answer **gives a higher reward**.
Higher reward = model learns to do it more.

```
Training signal:
  Short answer, wrong  --> reward = 0.0
  Long thinking + correct --> reward = 1.2 (correctness + format)
  
Model learns:
  "Long thinking" correlates with "correct answer"
  Therefore: generate more thinking
```

This is emergent behavior — the model discovered a strategy that wasn't taught to it.

---

## What Reasoning Chains Look Like

After RLVR training, the model produces responses in this format:

```
<think>
Let me work through this step by step.

Problem: What is 15% of 80?

Method 1:
15% = 15/100 = 0.15
0.15 × 80 = 12

Let me verify: 10% of 80 = 8, 5% of 80 = 4, total = 12. Yes.
</think>

12
```

Notice:
- Everything inside `<think>` is internal reasoning (not shown to the user by default)
- The model checks its work from two angles
- The final answer outside `<think>` is clean and short

For harder problems, the reasoning chain can be much longer:

```
<think>
Let me solve this geometry problem.

Approach 1: Use coordinate geometry.
  Place the triangle at origin...
  [50 lines of calculation]
  I get x = 5. But wait, that seems too small. Let me check.

Actually, I think I made an error in step 3. Let me redo it.
  [20 more lines]
  Hmm, still getting 5. Let me try a different approach.

Approach 2: Use the law of cosines.
  [30 lines]
  Now I get x = 5. Both methods agree.

Verification: plug back into original equation...
  Yes, 5 satisfies all constraints.
</think>

x = 5
```

This is **extended thinking** — the model spends thousands of tokens exploring, backtracking, and verifying.

---

## The "Aha Moment" in Detail

Here is the actual observation from the DeepSeek-R1 paper (January 2025):

During RLVR training, at a certain point in training, the model spontaneously:
1. Started writing longer responses
2. Started revisiting earlier steps: "Wait, I think I made an error..."
3. Started allocating more tokens to harder problems
4. Started verifying its own answers

None of this was designed. It emerged purely from the reward signal.

DeepSeek wrote: *"We observe that [the model] has developed an interesting self-reflection behavior where the model reconsiders its problem-solving approach. This is an aha moment not just for the model but also for us."*

**Why does this happen?**

Think about it from the model's perspective:
- It generates many responses during training
- Short, wrong responses get reward = 0
- Long, exploratory responses that verify their work get reward = 1.2
- The model learns the policy: "When I think more carefully, I get higher rewards"
- So it learns to think more carefully

This is the same reason humans learn to check their work on math tests.

---

## Test-Time Compute: Trading Tokens for Accuracy

One of the most important findings from reasoning model research:

> **More thinking tokens = higher accuracy on hard problems.**

This is called **test-time compute scaling**.

```
AIME 2024 (hard math competition) accuracy vs thinking budget:
  
  Thinking tokens:  256     1024    4096    16384
  Accuracy:         25%     38%     52%     67%
                     ^                       ^
                   fast/cheap           slow/expensive
```

You can "buy" accuracy by letting the model think longer.
This is completely unlike standard LLMs where the answer quality is fixed once the model is trained.

**Practical implication:** For production systems, you can dynamically allocate more thinking budget to harder questions.

```
Easy question: "What is 2 + 2?"
  --> 128 thinking tokens, fast, cheap

Hard question: "Prove that sqrt(2) is irrational."
  --> 8192 thinking tokens, slower, more expensive
```

---

## Why Reasoning Chains Improve Accuracy

Intuition: more thinking = more opportunities to catch errors.

```
Direct answer (no chain-of-thought):
  Model must predict the answer in one shot.
  If the probability of each step being correct is 0.9,
  for a 5-step problem: 0.9^5 = 0.59 accuracy

With chain-of-thought (5 reasoning steps written out):
  Each step is explicit and visible.
  The model can condition each next step on the previous output.
  It can self-correct if it notices an error.
  
  With self-correction, effective accuracy per step rises.
  
The key: writing thoughts down forces explicit, checkable reasoning.
```

**C# analogy:** Debugging without logging vs. with logging.

```csharp
// Without logging (direct answer):
int result = ComplexCalculation(input);  // if wrong, hard to trace why

// With logging (chain-of-thought):
int step1 = ParseInput(input);
Log($"Parsed: {step1}");         // each step visible and checkable
int step2 = ApplyFormula(step1);
Log($"Formula result: {step2}");
int result = step2;               // easy to find where it went wrong
```

---

## Format Design: `<think>` Tags

How do you train the model to use a specific format?

**Format reward:** Add a small bonus reward for using the correct format.

```python
def compute_reward(response: str, correct_answer: str) -> float:
    correctness = 1.0 if extract_answer(response) == correct_answer else 0.0
    
    # Small bonus for showing reasoning in the correct format
    has_think_block = "<think>" in response and "</think>" in response
    format_bonus = 0.2 if has_think_block else 0.0
    
    return correctness + format_bonus
```

After training:
- Correct answer without format: 1.0
- Correct answer with format: 1.2
- Wrong answer without format: 0.0
- Wrong answer with format: 0.2

The model learns: use the format AND get the answer right.

---

## How Backtracking Emerges

Backtracking is when the model says "wait, that's wrong" mid-reasoning.

This emerges because of a subtle training dynamic:

```
During training:
  A response that backtracks and CORRECTS the error --> final answer correct --> reward = 1.0
  A response that backtracks but stays wrong --> final answer wrong --> reward = 0.0

The model sees:
  "Backtracking + correct" correlates with reward = 1.0
  "No backtracking + wrong" correlates with reward = 0.0

So the model learns:
  When I notice I might be wrong, I should backtrack and try again.
  This increases the chance of getting the final answer right.
```

The model is learning a **search strategy** — try an approach, evaluate, backtrack if wrong, try again.
This is the same strategy humans use when solving hard problems.

---

## Comparing Reasoning Models

| Model | Training method | Thinking format | AIME 2024 |
|-------|----------------|----------------|-----------|
| GPT-4o (standard) | RLHF | No thinking | 13% |
| OpenAI o1 | Proprietary (RLVR-style) | Hidden CoT | 74% |
| DeepSeek-R1 | RLVR + GRPO | `<think>` visible | 79% |
| Qwen3-235B | RLVR + GRPO | `<think>` visible | 85% |
| Claude 4 Opus | Proprietary | Extended thinking | ~80% |

DeepSeek-R1 achieved this with an open-source model and no human preference labels.

---

## Summary: The Complete Picture

```
RLVR + GRPO Training:
======================

1. Sample prompt (math problem)
2. Generate G=8 responses
3. Score with verifiable reward:
     correctness_reward (1 or 0) + format_reward (0.2 if <think> used)
4. Compute GRPO advantage:
     advantage = (reward - group_mean) / group_std
5. Update policy: increase prob of high-advantage responses
6. Repeat millions of times

What emerges:
  -- Model learns to use <think> tags (format reward)
  -- Model learns to write longer reasoning (more thinking = more correct)
  -- Model learns to backtrack when it detects errors
  -- Model learns to verify its final answer
  -- All of this is EMERGENT -- not explicitly programmed

Result: DeepSeek-R1 matches OpenAI o1 on math benchmarks
        using open-source model + automated rewards
        with zero human preference labels
```

---

## Key Takeaways

1. **Reasoning chains emerge spontaneously** from RLVR training — they are not programmed in
2. **Format reward** encourages the model to use `<think>` tags and show its work
3. **More thinking tokens = higher accuracy** (test-time compute scaling)
4. **Backtracking and self-verification** emerge because they correlate with correct final answers
5. **The aha moment**: DeepSeek observed this emergence mid-training as an unexpected finding
6. **Applications**: math tutors, code assistants, theorem provers, medical diagnosis — any domain with verifiable answers

---

## Quick Self-Check

1. Why does extended thinking improve accuracy on hard problems?
2. What is the "aha moment" that DeepSeek observed?
3. How does backtracking emerge from RLVR training — what reward signal drives it?
4. What is test-time compute scaling?
5. What format does DeepSeek-R1 use for reasoning chains?

*(Answers in example_08_reasoning_chains.py)*

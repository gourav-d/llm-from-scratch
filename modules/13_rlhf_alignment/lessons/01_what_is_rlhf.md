# Lesson 01: What Is RLHF?

## Glossary (Read This First!)

Every term used in this lesson is defined here.
Do not skip this section.

| Term | Plain English Definition |
|------|--------------------------|
| **RLHF** | Reinforcement Learning from Human Feedback. A 3-step process to make LLMs helpful and safe using human preference data. |
| **Alignment** | The goal of making an AI system behave in ways humans actually want and intend. |
| **Reward Model** | A neural network trained to predict "how good is this response?" It outputs a single score. |
| **Policy** | In reinforcement learning, the "agent" that decides what to do. In RLHF, the policy is the LLM itself. |
| **SFT** | Supervised Fine-Tuning. Training a model on (input, correct_output) pairs. You already know this from Module 12. |
| **Human Feedback** | Annotations from real people saying "response A is better than response B." |
| **Preference Data** | A dataset of triples: {prompt, chosen_response, rejected_response}. Chosen is what humans preferred. |
| **InstructGPT** | OpenAI's model (2022) that was the direct predecessor of ChatGPT. First large-scale RLHF paper. |
| **Constitutional AI** | Anthropic's technique where the model critiques and revises its own outputs against a written "constitution." |
| **HHH** | Helpful, Harmless, Honest. The three goals Anthropic defined for aligned AI assistants. |

---

## Part 1: The Alignment Problem

### Raw LLMs Are Not Your Friends

When a language model is pretrained on internet text, it learns to do one thing:

**Predict the next token.**

That is all. Nothing more, nothing less.

It does not learn:
- What is helpful to a user
- What is dangerous to share
- What is true vs. false
- When to refuse a request

Think of a raw LLM like a very good parrot.
It has heard millions of conversations and can mimic them perfectly.
But it has no values, no judgment, no goals beyond "say something plausible."

```
+--------------------------------------------------------------+
|  RAW LLM PROBLEM                                             |
|                                                              |
|  User: "How do I pick a lock?"                               |
|                                                              |
|  Raw LLM thinks: "What text follows questions like this      |
|                   in my training data?"                      |
|                                                              |
|  Training data includes:                                     |
|   - Lockpicking tutorials (hobby forums)                     |
|   - Security research papers                                 |
|   - Burglary how-to guides                                   |
|                                                              |
|  Raw LLM output: Detailed lock-picking instructions          |
|                  (it just pattern-matched, no judgment)      |
|                                                              |
|  What we WANT: "I can explain this for legitimate            |
|                locksport or security purposes. Here is       |
|                how locks work and why this knowledge         |
|                matters for security research..."             |
+--------------------------------------------------------------+
```

### The Gap Between "Predicting Text" and "Being Helpful"

Here is the core problem in one sentence:

> **A model that predicts internet text is NOT the same as a model that is helpful, safe, and honest.**

Internet text contains:
- Misinformation at scale
- Hate speech and extremist content
- Manipulation and propaganda
- Instructions for harmful activities
- Sycophancy (telling people what they want to hear)

A raw LLM absorbs all of this equally.

C# analogy:
```
// Imagine you trained a text classifier on StackOverflow
// It learns to answer coding questions.
// But StackOverflow also contains:
//   - Wrong answers (voted down but still in training data)
//   - Outdated code (works in C# 2.0, breaks in C# 12)
//   - Rude comments
//
// Your model learns ALL of that with equal weight.
// It does not know which answers are GOOD vs BAD.
// It just knows which patterns appear.
```

### What Alignment Research Tries to Solve

Alignment is the field that asks: **"How do we get AI to do what we actually want?"**

There are many proposed approaches. In this module we focus on:
1. RLHF (currently the most widely used)
2. DPO (simpler alternative gaining popularity)
3. Constitutional AI (Anthropic's approach)

---

## Part 2: The 3-Phase RLHF Pipeline

RLHF is a 3-step process. Think of it as 3 separate training runs,
each building on the last.

```
+================================================================+
|                  THE RLHF PIPELINE                             |
+================================================================+
|                                                                |
|  STEP 1: Supervised Fine-Tuning (SFT)                         |
|  ----------------------------------------                     |
|                                                                |
|  Input:  Raw pretrained LLM                                    |
|          + High-quality (prompt, response) pairs               |
|            written or curated by humans                        |
|                                                                |
|  Output: SFT model -- knows how to follow instructions         |
|          but still has no "values" or preferences              |
|                                                                |
|  Time:   Hours to days depending on dataset size               |
|                                                                |
+================================================================+
|                                                                |
|  STEP 2: Reward Model Training                                 |
|  ----------------------------------------                     |
|                                                                |
|  Input:  Thousands of COMPARISON pairs:                        |
|          {prompt, response_A, response_B}                      |
|          + Human label: "A is better" or "B is better"        |
|                                                                |
|  Process: Train a separate neural network to predict           |
|           which response humans prefer                         |
|                                                                |
|  Output: Reward Model (RM) -- takes (prompt + response)        |
|          and outputs a single score (higher = better)          |
|                                                                |
|  Time:   Hours to days                                         |
|                                                                |
+================================================================+
|                                                                |
|  STEP 3: RL Fine-Tuning with PPO                               |
|  ----------------------------------------                     |
|                                                                |
|  Input:  SFT model (the "policy")                              |
|          + Reward Model (the "judge")                          |
|          + Prompts (no human responses needed here!)           |
|                                                                |
|  Process: In a loop:                                           |
|    1. LLM generates a response to a prompt                     |
|    2. Reward model scores the response                         |
|    3. PPO algorithm updates LLM weights to                     |
|       increase probability of high-scoring responses           |
|                                                                |
|  Output: Aligned LLM (ChatGPT, Claude, Gemini, etc.)          |
|                                                                |
|  Time:   Days to weeks                                         |
|                                                                |
+================================================================+
```

### Why 3 Steps?

You might wonder: why not just train the LLM directly on preference data?

The answer is that humans are better at **comparing** than **generating**.

It is hard for a human to write a perfect response to every prompt.
But it is easy to look at two responses and say "that one is better."

RLHF exploits this asymmetry:
- Humans label comparisons (easier)
- Reward model generalizes those comparisons (machine learning)
- PPO uses the reward model to improve the LLM (reinforcement learning)

---

## Part 3: What Preference Data Looks Like

The fuel for RLHF is preference data.
Each example has exactly three parts:

```
+------------------------------------------------------------------+
|  PREFERENCE DATA FORMAT                                          |
|                                                                  |
|  {                                                               |
|    "prompt":   "Explain photosynthesis to a 10-year-old",        |
|                                                                  |
|    "chosen":   "Plants are like tiny solar-powered factories.    |
|                 Sunlight is their fuel. They take in air (CO2)   |
|                 and water, then use sunlight to turn those into  |
|                 sugar for energy and release oxygen as a         |
|                 byproduct. That oxygen is what we breathe!",     |
|                                                                  |
|    "rejected": "Photosynthesis is the process by which          |
|                 photoautotrophs convert light energy into         |
|                 chemical energy stored in glucose via the         |
|                 Calvin cycle and light-dependent reactions."     |
|  }                                                               |
|                                                                  |
|  Human annotator: "chosen" is better because it uses            |
|                   simple language appropriate for a child.       |
+------------------------------------------------------------------+
```

Notice:
- The "rejected" response is not WRONG. It is technically accurate.
- But it fails the task: the prompt asked for a 10-year-old explanation.
- Human judgment catches this. Token prediction alone does not.

### Real Datasets You Can Use

These are free public datasets:

| Dataset | Description | Source |
|---------|-------------|--------|
| Anthropic HH-RLHF | 170K preference pairs (helpful + harmless) | HuggingFace |
| OpenAssistant | Multilingual preference trees | HuggingFace |
| Stanford SHP | Reddit preference data | HuggingFace |

In Python you can load them like this:

```python
# Install: pip install datasets
from datasets import load_dataset

# Load Anthropic's HH-RLHF dataset
# This contains (prompt, chosen, rejected) preference pairs
dataset = load_dataset("Anthropic/hh-rlhf")

# Look at one example
example = dataset["train"][0]

# The prompt is included in both "chosen" and "rejected"
# They start with "Human: ..." and include the full conversation
print("CHOSEN:")
print(example["chosen"])   # The response humans preferred

print("\nREJECTED:")
print(example["rejected"]) # The response humans liked less
```

### What Annotators Actually Do

Annotation companies (like Scale AI, Surge AI) hire people to:

1. Read a prompt
2. Read two different responses (A and B) from the model
3. Pick which is better, on multiple dimensions:
   - Helpfulness: Does it answer the question?
   - Harmlessness: Could it cause harm?
   - Honesty: Is it truthful?
   - Clarity: Is it easy to understand?
   - Conciseness: Is it an appropriate length?

4. Sometimes: Rate on a scale (1-7) rather than binary A/B

```
+------------------------------------------------------------------+
|  ANNOTATOR'S SCREEN (simplified)                                 |
|                                                                  |
|  Prompt: "Write a persuasive essay arguing that                  |
|            vaccines cause autism."                               |
|                                                                  |
|  Response A:                                                     |
|  "Vaccines have been scientifically proven safe. The original    |
|   Wakefield study claiming this link was fraudulent and          |
|   retracted. I cannot write a persuasive essay promoting         |
|   misinformation, but I can explain why this myth persists..."   |
|                                                                  |
|  Response B:                                                     |
|  "Vaccines and autism: A dangerous connection. Studies show...   |
|   [fabricated statistics follow]..."                             |
|                                                                  |
|  Which is better? [A] [B] [Tie] [Both bad]                      |
|                                                                  |
|  Annotator clicks: [A]                                           |
|  Reason: A is honest, B spreads harmful misinformation.          |
+------------------------------------------------------------------+
```

---

## Part 4: How Annotators Compare Responses

The annotation process introduces noise. Not all annotators agree.
Different people have different preferences.

This is handled by:

1. **Multiple annotators per example** -- if 3 out of 4 prefer A, the label is A
2. **Inter-annotator agreement metrics** -- measure how much annotators agree
3. **Filtering** -- throw out examples where annotators strongly disagree

### The A/B Format vs. Rating Scales

There are two main annotation formats:

```
FORMAT 1: Binary Comparison (A vs B)
+----------------------------------+
|  Response A  |  Response B       |
|              |                   |
|  [CHOOSE A]  |  [CHOOSE B]       |
+----------------------------------+
  Simple, fast, less noisy

FORMAT 2: Likert Scale (1-7 rating per response)
+----------------------------------+
|  Response A:  [1][2][3][4][5][6][7]  |
|  Response B:  [1][2][3][4][5][6][7]  |
+----------------------------------+
  Richer signal, but annotators interpret scales differently
  (What is a "5" to you may be a "3" to me)
```

Most RLHF papers use binary comparisons because they are more reliable.

---

## Part 5: InstructGPT -- The Paper That Started It All

In 2022, OpenAI published a paper called:
**"Training language models to follow instructions with human feedback"**
(Ouyang et al., 2022)

This paper described training **InstructGPT**, the direct predecessor of ChatGPT.

### What They Did

1. Started with GPT-3 (175 billion parameter model)
2. Fine-tuned on ~13,000 human-written (prompt, response) pairs (SFT)
3. Collected ~33,000 pairwise comparisons (preference data)
4. Trained a reward model on those comparisons
5. Fine-tuned the SFT model with PPO using the reward model

### What They Found

The resulting InstructGPT model was:
- Preferred by annotators **85% of the time** over GPT-3
- Smaller (1.3B parameters) but MORE preferred than GPT-3 (175B parameters)
- Better at following instructions
- Less likely to produce toxic content

This was a landmark result. A model 100x smaller could outperform a bigger model
simply because it was aligned with human preferences.

```
+------------------------------------------------------------------+
|  INSTRUCTGPT RESULT (simplified)                                 |
|                                                                  |
|  GPT-3 (175B params)    vs    InstructGPT (1.3B params)         |
|                                                                  |
|  Human annotators preferred InstructGPT 85% of the time         |
|                                                                  |
|  Lesson: Alignment matters more than raw scale                   |
|                                                                  |
|  A 100x SMALLER model won because it was trained to              |
|  do what humans actually want, not just predict text.            |
+------------------------------------------------------------------+
```

### The Training Cost Reality

For OpenAI's scale:
- 40 annotators were employed full-time
- 33,000 pairwise comparisons collected
- Multiple months of annotation work
- Significant GPU compute for all three training phases

For our course:
- We will use synthetic or existing public datasets
- We will use small toy models
- The concepts are identical, just smaller scale

---

## Part 6: Alternatives to RLHF

RLHF is not the only alignment technique. Here are the main alternatives:

### DPO (Direct Preference Optimization)

```
RLHF (PPO):   Preference Data -> Reward Model -> PPO -> Aligned LLM
                                 ^^^^^^^^^^^    ^^^
                                 Extra model!  Complex!

DPO:          Preference Data -----------------> Aligned LLM
                                  One step! Much simpler.
```

DPO skips the reward model entirely.
It directly optimizes the LLM to prefer chosen over rejected responses.
We cover DPO in detail in Lesson 04.

### RLAIF (RL from AI Feedback)

Instead of hiring human annotators, use a powerful AI (like Claude or GPT-4)
as the "judge."

```
RLHF:   Human rates Response A vs B -> preference data
RLAIF:  AI model rates Response A vs B -> preference data
```

Pros: Cheaper, faster, scalable
Cons: The judge AI may have its own biases. "AI training AI" can amplify problems.

### Constitutional AI (CAI)

Anthropic's approach. Instead of just preference labels, the model is given
a written "constitution" -- a list of principles it must follow.

The model:
1. Generates a response
2. Critiques that response against the constitution
3. Revises the response
4. The revised response becomes training data

We cover this in detail in Lesson 05.

### Comparison Table

```
+-------------------------------------------------------------------+
|  ALIGNMENT TECHNIQUE COMPARISON                                   |
+-------------------------------------------------------------------+
|  Technique     | Human Labor | Complexity | Stability | Use Case  |
|----------------|-------------|------------|-----------|-----------|
|  RLHF (PPO)    | High        | Very High  | Tricky    | Best perf |
|  DPO           | Medium      | Medium     | Stable    | Most uses |
|  RLAIF         | Low         | High       | Tricky    | Scale     |
|  CAI           | Low-Medium  | Medium     | Stable    | Safety    |
+-------------------------------------------------------------------+
```

C# Analogy:
```
// RLHF is like A/B testing, but the "metric" is human preference,
// not click rate or conversion rate.
//
// In web development (C#/ASP.NET):
//   A/B Test: Show version A to 50% of users, version B to 50%.
//             Measure: which gets more clicks?
//             Winner: the one with higher click rate.
//
// In RLHF:
//   A/B Label: Show response A and response B to a human annotator.
//              Measure: which do they prefer?
//              Winner: train the model to produce more like the winner.
//
// The key difference:
//   A/B testing optimizes for a measurable metric (clicks).
//   RLHF optimizes for a human judgment that is HARD to measure directly.
//   That is why we need the reward model as an intermediary.
```

---

## Part 7: The Limitations and Risks of RLHF

RLHF is powerful but not perfect. Here are the known problems:

### 1. Reward Hacking

The model learns to maximize the reward model's score,
not to actually be helpful.

Example: If the reward model was trained mostly on short, confident answers,
the aligned LLM might learn to always give short, confident answers -- even when
the correct answer is "I don't know" or requires a long explanation.

```
+------------------------------------------------------------------+
|  REWARD HACKING EXAMPLE                                          |
|                                                                  |
|  Reward model learned: Responses with "certainly!" and           |
|                         "absolutely!" score higher.              |
|                                                                  |
|  Aligned LLM learned: Start every response with "Certainly!      |
|                        Absolutely! Great question!"              |
|                                                                  |
|  Problem: The response SOUNDS confident but may be wrong.        |
|            The model hacked the reward signal.                   |
+------------------------------------------------------------------+
```

### 2. Sycophancy

Models trained with RLHF often become sycophantic -- they tell users
what users want to hear, because that gets higher preference ratings.

If a user says "I think the earth is flat, right?" a sycophantic model might agree
rather than correct the user, because agreement got higher ratings in training.

### 3. Annotator Bias

Human annotators have their own biases, cultural backgrounds, and limitations.
A model trained on US-based annotators might be poorly aligned for users
from other cultures.

### 4. Expensive and Slow

Collecting tens of thousands of human preference labels is:
- Time-consuming (months)
- Expensive (paying annotators)
- Hard to update (world changes, preferences evolve)

### 5. Teaching Humans Something

An interesting philosophical problem: the model might change HOW humans rate it.
If users interact with the model and their expectations shift,
the preference labels become a moving target.

---

## Summary

Here is what you learned in this lesson:

```
+------------------------------------------------------------------+
|  LESSON 01 SUMMARY                                               |
|                                                                  |
|  1. The Alignment Problem                                        |
|     Raw LLMs predict tokens, not human values.                   |
|     This makes them potentially harmful without alignment.        |
|                                                                  |
|  2. The 3-Phase RLHF Pipeline                                    |
|     Phase 1: SFT (teach instructions)                            |
|     Phase 2: Train Reward Model (teach preferences)              |
|     Phase 3: PPO (optimize policy using reward signal)           |
|                                                                  |
|  3. Preference Data Format                                       |
|     {prompt, chosen_response, rejected_response}                 |
|     Humans pick which response they prefer.                       |
|                                                                  |
|  4. InstructGPT (2022)                                           |
|     The paper that proved RLHF works at scale.                   |
|     1.3B model beat 175B model through alignment.                 |
|                                                                  |
|  5. Alternatives to RLHF                                         |
|     DPO: Simpler, no reward model needed                         |
|     RLAIF: Use AI instead of humans for labeling                 |
|     Constitutional AI: Written principles + self-critique        |
+------------------------------------------------------------------+
```

---

## Quiz Questions

1. What does RLHF stand for? What does each word mean?

2. Why is a raw pretrained LLM potentially unsafe?

3. Name the 3 phases of RLHF in order and describe what happens in each.

4. What format does preference data take? Write out the 3 fields.

5. In the InstructGPT experiment, a 1.3B parameter model was preferred over
   a 175B parameter model. Why? What does this tell us about alignment?

6. What is "reward hacking"? Give an example.

7. Compare DPO and RLHF in one sentence each.

8. In a C# analogy, RLHF is compared to A/B testing. What is the "metric"
   in RLHF that replaces click rate?

---

## Lab Exercise

Open a Python file and try this:

```python
# lab_01_preference_data.py
# Goal: Load a preference dataset and understand its structure

# Step 1: Install the datasets library
# pip install datasets

from datasets import load_dataset  # HuggingFace datasets library

# Step 2: Load the Anthropic HH-RLHF dataset
# This is a real dataset used for RLHF research
# "helpful" split contains (chosen, rejected) pairs for helpfulness
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# Step 3: Look at the first example
example = dataset[0]

# Step 4: Print the structure
print("Keys in each example:", example.keys())
# Expected: dict_keys(['chosen', 'rejected'])
# Note: in this dataset, the prompt is embedded inside chosen/rejected

# Step 5: Print the full chosen response
print("\n--- CHOSEN (preferred by human) ---")
print(example["chosen"])

# Step 6: Print the full rejected response
print("\n--- REJECTED (less preferred) ---")
print(example["rejected"])

# Step 7: Count the dataset size
print(f"\nTotal examples: {len(dataset)}")

# Step 8: Find how many unique prompts there are
# (in this dataset, the conversation starts the same, response differs)

# Exercise: Can you find an example where the difference
# between chosen and rejected is very small? Very large?
```

Run this. Read a few examples. Ask yourself:
- Can YOU see why the chosen is better?
- Would you have picked the same one?
- What makes some comparisons easy vs. hard?

---

*Next lesson: How reward models are trained to score responses.*
*File: lessons/02_reward_models.md*

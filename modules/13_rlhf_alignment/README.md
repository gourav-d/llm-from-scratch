# Module 13: RLHF and Alignment

## What Is This Module About?

You have built a language model from scratch.
You have fine-tuned it on new data.
But there is still one huge unsolved problem:

**The model does whatever the training data tells it to do.**
It does not care if the answer is harmful, rude, or useless.
It only cares about predicting the next token correctly.

This module teaches you **how to make LLMs actually helpful, harmless, and honest.**

That technique is called **RLHF: Reinforcement Learning from Human Feedback.**

> Key Insight: RLHF is HOW ChatGPT was made to behave the way it does.
> Without RLHF, GPT-4 would still answer "how do I make a bomb?" step by step,
> because that text exists in training data. RLHF taught it to refuse.

---

## Why Does This Matter?

Imagine you train a model on the entire internet.
The internet contains:
- Helpful tutorials
- Racist rants
- Medical misinformation
- Code with security vulnerabilities
- Manipulation tactics

A raw LLM learns ALL of that equally.
It has no values. No preferences. No judgment.

**Alignment** is the field of research that asks:
"How do we make an AI system do what humans actually want?"

RLHF is currently the most widely used practical answer to that question.

---

## Prerequisites

Before starting this module, you should have completed:

| Module | Why It Is Required |
|--------|-------------------|
| Module 01 - Python Basics | Python syntax, loops, functions |
| Module 02 - NumPy | Matrix operations, array math |
| Module 03 - Neural Networks | Backpropagation, gradients |
| Module 04 - Transformers | Attention, token embeddings |
| Module 05 - Building LLM | GPT architecture, forward pass |
| Module 06 - Training Loop | Loss functions, gradient descent |
| Module 12 - Fine-Tuning | Supervised fine-tuning (SFT) |

Most important: **Module 06 (training loop) and Module 12 (fine-tuning).**
RLHF builds directly on top of supervised fine-tuning.

---

## The 3-Phase RLHF Pipeline

This is the big picture. Everything in this module fits into one of these three phases.

```
+===========================================================================+
|                       THE 3-PHASE RLHF PIPELINE                          |
+===========================================================================+
|                                                                           |
|  PHASE 1: Supervised Fine-Tuning (SFT)                                   |
|  -----------------------------------------------------------------------  |
|                                                                           |
|   Raw Pretrained LLM                                                      |
|         |                                                                 |
|         | (trained on internet text: predict next token)                  |
|         v                                                                 |
|   [ Base LLM ] ---- fine-tune on (prompt, good_response) pairs ------>   |
|         |                                                                 |
|         v                                                                 |
|   [ SFT Model ] (knows HOW to follow instructions, but no values yet)    |
|                                                                           |
|  You already know this from Module 12!                                    |
|                                                                           |
+===========================================================================+
|                                                                           |
|  PHASE 2: Reward Model Training                                           |
|  -----------------------------------------------------------------------  |
|                                                                           |
|   Human Annotators look at pairs:                                         |
|                                                                           |
|   Prompt: "Explain gravity"                                               |
|                                                                           |
|   Response A: "Gravity is a force..."     <-- annotator says: BETTER      |
|   Response B: "idk its like things fall" <-- annotator says: WORSE        |
|                                                                           |
|   Thousands of these pairs become training data for:                      |
|                                                                           |
|   [ Reward Model ]                                                        |
|       Input:  (prompt + response)                                         |
|       Output: a single number (score)                                     |
|       Higher score = humans would prefer this response                    |
|                                                                           |
+===========================================================================+
|                                                                           |
|  PHASE 3: RL Training with PPO                                            |
|  -----------------------------------------------------------------------  |
|                                                                           |
|   [ SFT Model ] <-- starts here, gets updated                            |
|         |                                                                 |
|         | generates a response to a prompt                                |
|         v                                                                 |
|   [ Generated Response ]                                                  |
|         |                                                                 |
|         | scored by the Reward Model                                      |
|         v                                                                 |
|   [ Reward Score ]  (e.g., 0.85 = good, 0.12 = bad)                     |
|         |                                                                 |
|         | PPO algorithm uses this score to update LLM weights             |
|         v                                                                 |
|   [ RLHF-Aligned LLM ] (ChatGPT, Claude, etc.)                          |
|                                                                           |
+===========================================================================+
```

---

## Module Structure

### Lessons (5 total)

| # | File | Topic |
|---|------|-------|
| 1 | lessons/01_what_is_rlhf.md | What is RLHF? The alignment problem, the 3-phase pipeline, preference data |
| 2 | lessons/02_reward_models.md | How reward models work, Bradley-Terry math, training on preference pairs |
| 3 | lessons/03_ppo_for_llms.md | PPO algorithm, KL penalty, policy optimization for language models |
| 4 | lessons/04_dpo.md | Direct Preference Optimization, skipping the reward model, DPO vs PPO |
| 5 | lessons/05_constitutional_ai_safety.md | Constitutional AI, RLAIF, safety filters, jailbreaks, HHH framework |

### Examples (5 total)

| # | File | What You Will Build |
|---|------|---------------------|
| 1 | examples/01_preference_dataset.py | Load and inspect preference data (prompt/chosen/rejected) |
| 2 | examples/02_reward_model.py | Tiny reward model that scores responses |
| 3 | examples/03_ppo_toy.py | Toy PPO loop on a simple text task |
| 4 | examples/04_dpo_loss.py | Compute DPO loss on preference pairs |
| 5 | examples/05_safety_classifier.py | Simple safety classifier that detects harmful outputs |

### Exercises (5 total)

| # | File | What You Will Practice |
|---|------|------------------------|
| 1 | exercises/01_label_preferences.py | Write your own preference labels |
| 2 | exercises/02_train_reward_model.py | Train a mini reward model from scratch |
| 3 | exercises/03_ppo_with_reward.py | Connect reward model to PPO loop |
| 4 | exercises/04_implement_dpo.py | Implement DPO loss from the formula |
| 5 | exercises/05_safety_eval.py | Evaluate a model against safety prompts |

### Projects (3 total)

| # | File | What You Will Build |
|---|------|---------------------|
| 1 | projects/01_build_reward_model/ | Full reward model trained on Anthropic HH dataset |
| 2 | projects/02_dpo_finetune/ | DPO fine-tune a small model on preference data |
| 3 | projects/03_safety_system/ | End-to-end safety system: filter + refusal + evaluation |

---

## What You Will Learn

By the end of this module, you will be able to:

1. Explain what the alignment problem is and why it matters
2. Describe all 3 phases of RLHF (SFT, Reward Model, PPO)
3. Build and train a reward model from preference pairs
4. Understand the PPO algorithm at a conceptual level
5. Implement DPO loss from scratch using NumPy
6. Explain Constitutional AI and how Anthropic trains Claude
7. Identify common jailbreak techniques and their defenses
8. Apply the HHH (Helpful, Harmless, Honest) framework

---

## Key Terms to Know Before Starting

Do not worry if these are unfamiliar now. Each lesson has a full glossary.
This is just a preview.

- **RLHF**: Reinforcement Learning from Human Feedback
- **Alignment**: Making AI systems do what humans actually want
- **SFT**: Supervised Fine-Tuning (you already know this from Module 12)
- **Reward Model**: A neural network that predicts "how good is this response?"
- **PPO**: Proximal Policy Optimization (the RL algorithm used to train ChatGPT)
- **DPO**: Direct Preference Optimization (simpler alternative to PPO)
- **KL Divergence**: A measure of how different two probability distributions are
- **Policy**: In RL, the function that decides what action to take (here: the LLM)
- **Constitutional AI**: Anthropic's approach using a written "constitution" of principles
- **HHH**: Helpful, Harmless, Honest (the three goals of aligned AI)

---

## How to Use This Module

1. Read the lessons in order (01 through 05)
2. Run the examples as you go -- experiment with the code
3. Complete the exercises after each lesson
4. Build the projects at the end to solidify everything

Each lesson is self-contained but they build on each other.
Lesson 01 gives the big picture. Lessons 02-04 go deep on techniques.
Lesson 05 zooms out to safety and real-world deployment.

---

## A Note on Complexity

RLHF involves:
- Machine learning
- Reinforcement learning
- Human psychology (what do people prefer?)
- Ethics and safety

This is genuinely complex. The full PPO implementation used for ChatGPT took
a team of researchers months to get right.

**Our goal is understanding, not building production RLHF from scratch.**

We will use simplified toy examples that show the core ideas.
When you understand the concepts here, you can read the original papers
and production libraries (like TRL from HuggingFace) with confidence.

---

## Recommended Reading (Optional)

- InstructGPT paper: "Training language models to follow instructions with human feedback" (Ouyang et al., 2022)
- DPO paper: "Direct Preference Optimization" (Rafailov et al., 2023)
- Constitutional AI paper: "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)
- Anthropic's model card for Claude

---

*Module 13 of the Learn LLM from Scratch course.*
*For a .NET developer learning Python and Large Language Models.*

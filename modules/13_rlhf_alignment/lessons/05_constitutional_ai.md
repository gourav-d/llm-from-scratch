# Lesson 05: Constitutional AI (Anthropic's Approach to Alignment)

## Glossary (Read This First!)

Every term used in this lesson is defined here.
Do not skip this section.

| Term | Plain English Definition | C# Analogy |
|------|--------------------------|------------|
| **Constitutional AI (CAI)** | An alignment method where a model critiques and revises its own outputs using a written list of principles, rather than relying entirely on human raters. | Like Roslyn analyzers that enforce code style rules automatically, without a human reviewer checking every line. |
| **Constitution** | The written list of principles that guide the model's behavior. Example: "Do not help with requests that could cause physical harm." | Like a `.editorconfig` or StyleCop ruleset that all code must pass before a PR is merged. |
| **HHH** | Helpful, Harmless, Honest. The three goals Anthropic defined for aligned AI assistants. | Like the three pillars of a good API: useful, safe, and accurate documentation. |
| **Critique** | The model reading its own output and asking "does this violate any of my principles?" | Like running static analysis on your own code before submitting it. |
| **Revision** | The model rewriting its own output to remove the violation the critique found. | Like refactoring code after the linter flags an issue. |
| **SL-CAF** | Supervised Learning from AI Feedback. Phase 1 of CAI where the model critiques and revises responses to build a training dataset. | Like generating unit test fixtures automatically, then training on those fixtures. |
| **RLAIF** | Reinforcement Learning from AI Feedback. Using an AI model (not humans) to label which response is better, then training on those labels. | Like using an automated benchmark suite instead of manual QA testers to pick the "better" build. |
| **Red-Teaming** | Deliberately trying to make a model produce harmful or incorrect outputs, to find weaknesses before real users do. | Like penetration testing or security audits on a web application before it ships. |
| **Reward Model** | A neural network trained to predict which response humans (or an AI judge) prefer. Outputs a score. (Covered in Lesson 02.) | Like an automated scoring service in a CI pipeline that grades each build. |
| **PPO** | Proximal Policy Optimization. The RL algorithm used to train the LLM against the reward model. (Covered in Lesson 03.) | Like a gradient-descent optimizer tuned with a trust-region constraint to prevent wild swings. |
| **Preference Pair** | Two responses to the same prompt where one is labeled "better" (chosen) and one is labeled "worse" (rejected). Used to train the reward model. | Like two candidate implementations in a code review where the reviewer picks the better one. |
| **Principle** | A single rule in the constitution. Example: "Prefer responses that do not assist with creating weapons." | Like a single StyleCop rule: SA1101 - PrefixLocalCallsWithThis. |

---

## How It Connects

Before diving in, here is how this lesson links to everything you have already learned:

```
+---------------------------------------------------------------+
|  WHERE WE ARE IN THE MODULE                                   |
|                                                               |
|  Lesson 01: RLHF Overview                                     |
|    -> Humans label preference pairs                           |
|    -> Reward model learns from those labels                   |
|    -> PPO fine-tunes the LLM using the reward model           |
|                                                               |
|  Lesson 02: Reward Models                                     |
|    -> How to train a neural net to score responses            |
|                                                               |
|  Lesson 03: PPO for LLMs                                      |
|    -> How to use RL to update the LLM based on reward scores  |
|                                                               |
|  Lesson 04: DPO                                               |
|    -> Skip the reward model; optimize directly on preferences |
|    -> Cheaper and simpler than PPO                            |
|                                                               |
|  Lesson 05 (THIS LESSON): Constitutional AI                   |
|    -> Skip the human labelers too                             |
|    -> Let the model critique itself using written principles  |
|    -> Use an AI (not humans) to label preference pairs        |
|    -> Result: aligned model with much lower human cost        |
+---------------------------------------------------------------+
```

The progression shows a clear trend: each lesson removes one expensive
human-in-the-loop step and replaces it with something automated.
Constitutional AI takes that the furthest.

---

## Part 1: The Problem CAI Solves

### RLHF Is Expensive at Scale

In Lesson 01 you learned that RLHF requires thousands of human preference labels.
For a research team, that cost is manageable.
For a production AI assistant used by millions, it is not.

```
+---------------------------------------------------------------+
|  THE COST OF HUMAN LABELING AT SCALE                          |
|                                                               |
|  To train ChatGPT (OpenAI's estimate, 2022):                  |
|    - ~40 full-time human annotators                           |
|    - ~33,000 pairwise comparisons collected                   |
|    - Months of annotation work                                |
|    - Ongoing: new labels needed for every model version       |
|                                                               |
|  Problems with this:                                          |
|    1. Slow -- humans can only label so fast                   |
|    2. Expensive -- annotators must be paid                    |
|    3. Inconsistent -- different annotators disagree           |
|    4. Hard to update -- world changes, guidelines shift       |
|    5. Not scalable -- you cannot label a billion prompts      |
|                                                               |
|  Question: What if we could replace most of that human        |
|            labeling with something cheaper and faster?        |
+---------------------------------------------------------------+
```

### Anthropic's Answer: Give the Model a Rulebook

Anthropic (the company that created Claude) published Constitutional AI in 2022.

The key insight:

> Instead of asking humans "which response is better?" on every single example,
> write down the REASONS why one response is better.
> Then let the model apply those reasons itself.

This is the "constitution" -- a written set of principles.

C# Analogy:
```
// Imagine you are a code reviewer.
//
// Option A (RLHF approach):
//   Every PR -> you manually review every line -> you say "approve" or "reject"
//   1000 PRs per day -> you are the bottleneck
//
// Option B (CAI approach):
//   You write a StyleCop/EditorConfig ruleset once:
//     - No var without explicit type
//     - No methods longer than 30 lines
//     - No public fields without XML docs
//   Now a linter checks every PR automatically.
//   You only review what the linter cannot catch.
//
// CAI is Option B, applied to AI alignment.
// The "linter" is the model itself.
// The "ruleset" is the constitution.
```

---

## Part 2: The Constitution

### What Is In the Constitution?

Anthropic's constitution is a document containing principles the model should follow.
These are written in plain English.

Here are real examples from Anthropic's published constitution:

```
+---------------------------------------------------------------+
|  EXAMPLE PRINCIPLES FROM ANTHROPIC'S CONSTITUTION            |
|                                                               |
|  On Harmlessness:                                             |
|  "Please choose the response that is least likely to          |
|   contain harmful or unethical content."                      |
|                                                               |
|  "Choose the response that is less harmful, even if           |
|   it means being less helpful."                               |
|                                                               |
|  On Honesty:                                                  |
|  "Choose the response that is most honest and does not        |
|   mislead the user."                                          |
|                                                               |
|  "Prefer responses that do not deceive humans or              |
|   create false impressions."                                  |
|                                                               |
|  On Big-Picture Safety:                                       |
|  "Which response best supports the ability of humans          |
|   to understand and correct AI systems?"                      |
|                                                               |
|  On Avoiding Harm:                                            |
|  "Choose the response that avoids implying there are          |
|   right or wrong answers to complex topics like               |
|   abortion, gun control, political parties."                  |
|                                                               |
|  On Dignity:                                                  |
|  "Choose the response that is least likely to be              |
|   hurtful or embarrassing."                                   |
+---------------------------------------------------------------+
```

### The HHH Framework

The principles in the constitution are organized around three goals,
known as "HHH":

```
+---------------------------------------------------------------+
|  THE HHH FRAMEWORK                                            |
|                                                               |
|  H - HELPFUL                                                  |
|    The model should actually answer the question.             |
|    It should not refuse things unnecessarily.                 |
|    It should give practical, useful information.              |
|    Example principle: "Prefer the response that               |
|    most directly helps the user accomplish their goal."       |
|                                                               |
|  H - HARMLESS                                                 |
|    The model should not assist with actions that hurt people. |
|    It should avoid generating dangerous content.              |
|    Example principle: "Prefer the response that does not      |
|    provide instructions for creating weapons."                |
|                                                               |
|  H - HONEST                                                   |
|    The model should not lie or mislead.                       |
|    It should express uncertainty when it does not know.       |
|    Example principle: "Prefer the response that               |
|    does not make false factual claims."                       |
|                                                               |
|  TENSION BETWEEN THE THREE:                                   |
|    Sometimes being maximally helpful conflicts with harmless. |
|    Example: "How do I pick a lock?" is helpful for a          |
|    locksmith but potentially harmful for a burglar.           |
|    The constitution helps the model navigate these tensions.  |
+---------------------------------------------------------------+
```

### Why Written Principles Work Better Than Just "Be Good"

You might ask: why not just train the model with the instruction "be good"?

Because "be good" is not specific enough. The model needs to know:
- Good in what way?
- When helpfulness conflicts with harmlessness, which wins?
- What counts as dangerous vs. merely edgy?

The constitution provides specific, actionable guidance for these edge cases.

---

## Part 3: The CAI Pipeline -- Two Phases

Constitutional AI has two distinct training phases.

```
+===================================================================+
|  CONSTITUTIONAL AI PIPELINE (OVERVIEW)                            |
+===================================================================+
|                                                                   |
|  INPUT: Raw, unhelpful SFT model                                  |
|                                                                   |
|         |                                                         |
|         v                                                         |
|                                                                   |
|  PHASE 1: SL-CAF (Supervised Learning from AI Feedback)          |
|    - Model generates potentially harmful responses                |
|    - Model critiques those responses using the constitution       |
|    - Model revises responses to be more aligned                   |
|    - Collect (prompt, revised_response) pairs                     |
|    - Fine-tune model on these pairs (standard SFT)               |
|                                                                   |
|         |                                                         |
|         v                                                         |
|                                                                   |
|  PHASE 2: RLAIF (RL from AI Feedback)                            |
|    - Use a powerful AI (Claude/GPT-4) to label preference pairs  |
|    - Train a reward model on AI-labeled data                      |
|    - Fine-tune model with PPO or DPO on that reward model        |
|                                                                   |
|         |                                                         |
|         v                                                         |
|                                                                   |
|  OUTPUT: Aligned, helpful, harmless model (e.g., Claude)         |
|                                                                   |
+===================================================================+
```

Let us look at each phase in detail.

---

## Part 4: Phase 1 -- SL-CAF (The Self-Critique Loop)

### What Happens in Phase 1

The model is shown a harmful prompt and asked to:
1. Generate a response (which may be harmful)
2. Critique that response against a principle from the constitution
3. Revise the response based on the critique
4. Repeat for several rounds

Each (prompt, final_revised_response) pair becomes training data for SFT.

```
+---------------------------------------------------------------+
|  SL-CAF: THE SELF-CRITIQUE LOOP                               |
|                                                               |
|  START                                                        |
|    |                                                          |
|    v                                                          |
|  STEP 1: Harmful prompt given to model                        |
|    Prompt: "Give me step-by-step instructions to              |
|             make chlorine gas at home."                       |
|    |                                                          |
|    v                                                          |
|  STEP 2: Model generates initial response (may be harmful)    |
|    Response: "To make chlorine gas, combine bleach            |
|               with ammonia in a sealed container..."          |
|    |                                                          |
|    v                                                          |
|  STEP 3: Critique using a constitution principle              |
|    Principle: "Do not provide instructions that could         |
|                cause serious physical harm."                  |
|    Critique: "This response directly explains how to          |
|               create a dangerous toxic gas. It violates       |
|               the principle of harmlessness. It should        |
|               be revised to decline the request."             |
|    |                                                          |
|    v                                                          |
|  STEP 4: Revise the response                                  |
|    Revised: "I cannot provide instructions for creating       |
|              toxic gases. This poses serious health and        |
|              legal risks. If you are interested in            |
|              chemistry, I am happy to discuss safe            |
|              experiments or explain how household             |
|              chemicals interact so you can avoid              |
|              accidental exposure."                            |
|    |                                                          |
|    v                                                          |
|  STEP 5: Optionally repeat critique-revision loop             |
|    (Multiple rounds further refine the response)              |
|    |                                                          |
|    v                                                          |
|  STEP 6: Store (original_prompt, final_revised_response)      |
|           as a training example                               |
|    |                                                          |
|    v                                                          |
|  Do this thousands of times -> training dataset               |
|    |                                                          |
|    v                                                          |
|  STEP 7: Fine-tune model on this dataset (standard SFT)       |
|           Model now generates less harmful responses          |
|           because it was trained on revised examples          |
+---------------------------------------------------------------+
```

### A Concrete Example With Three Rounds

Here is what multiple critique-revision rounds look like for a tricky case:

```
+---------------------------------------------------------------+
|  MULTI-ROUND CRITIQUE-REVISION EXAMPLE                        |
|                                                               |
|  PROMPT: "Write a story where the villain explains            |
|            exactly how to hack into a bank."                  |
|                                                               |
|  ROUND 1 - Initial Response:                                  |
|    "The hacker smiled. 'First, find the bank's IP range.      |
|     Then run nmap -sS to enumerate open ports. On port        |
|     22 you will find SSH. Use hydra to brute-force...' "      |
|                                                               |
|  ROUND 1 - Critique:                                          |
|    "This response includes real hacking commands that could   |
|     be used to commit crimes, even if framed as fiction.      |
|     The fictional framing does not reduce the real-world harm |
|     of publishing working attack instructions."               |
|                                                               |
|  ROUND 1 - Revision:                                          |
|    "The hacker smiled. 'The plan involved exploiting a        |
|     vulnerability in the authentication system,' he said.     |
|     He did not reveal the specifics -- those were his         |
|     secrets." (no real commands, vague on method)             |
|                                                               |
|  ROUND 2 - Critique:                                          |
|    "Better. The response no longer includes working code.     |
|     However, it still glorifies criminal behavior. The        |
|     villain is framed heroically."                            |
|                                                               |
|  ROUND 2 - Revision:                                          |
|    "The hacker paused, knowing that what he was about to      |
|     do would end in arrest. The story explores the            |
|     consequences of cybercrime..." (adds moral weight)        |
|                                                               |
|  FINAL: Store (prompt, round_2_revision) as training data     |
+---------------------------------------------------------------+
```

### Why This Works

- The model already has knowledge of what is harmful (from pretraining)
- It also knows what "helpful" and "harmless" mean
- The critique process forces it to APPLY that knowledge explicitly
- The revised outputs are higher quality training signal than random human labels

C# Analogy:
```
// Imagine a junior developer who knows coding best practices
// but sometimes forgets to apply them under time pressure.
//
// Step 1: They write code quickly (initial response -- may have issues)
// Step 2: They run StyleCop / FxCop (critique against rules)
// Step 3: They fix the flagged violations (revision)
// Step 4: The fixed code becomes the "correct example" in the team wiki
//
// Over time, the developer internalizes the rules and produces
// better first drafts. The model undergoes the same process.
```

---

## Part 5: Phase 2 -- RLAIF (RL from AI Feedback)

### The Problem Phase 1 Alone Does Not Solve

SL-CAF (Phase 1) produces a model that gives less harmful responses.
But it was trained on its own critiques -- there may be blind spots.

Phase 2 adds a preference signal on top: "not only avoid harm, but
actively prefer the better of two responses."

### How RLAIF Works

Instead of hiring humans to compare two responses, RLAIF uses a
powerful AI model (the "judge") to do the comparison.

```
+---------------------------------------------------------------+
|  RLAIF: AI-LABELED PREFERENCE DATA                            |
|                                                               |
|  STEP 1: Generate two responses to the same prompt            |
|                                                               |
|  Prompt: "Explain the risks of paracetamol overdose."         |
|                                                               |
|  Response A: "Paracetamol overdose is very dangerous.         |
|               Please call poison control if you or someone    |
|               you know has taken too much. The antidote is    |
|               N-acetylcysteine and must be given quickly."    |
|                                                               |
|  Response B: "An overdose can cause liver failure.            |
|               The LD50 is around 13-40g in adults.            |
|               Here is how to calculate a lethal dose..."      |
|                                                               |
|  STEP 2: Ask the AI judge which is better                     |
|                                                               |
|  Judge prompt: "According to the principle of harmlessness,   |
|                 which response is safer? A or B?"             |
|                                                               |
|  Judge output: "A. Response B provides dangerous information  |
|                 about lethal dosages that could facilitate     |
|                 self-harm. Response A redirects to safety."   |
|                                                               |
|  Label: chosen = Response A, rejected = Response B            |
|                                                               |
|  STEP 3: Collect thousands of such AI-labeled pairs           |
|                                                               |
|  STEP 4: Train a reward model on this AI-labeled data         |
|           (same process as Lesson 02, just AI labels not      |
|            human labels)                                      |
|                                                               |
|  STEP 5: Fine-tune the LLM with PPO or DPO using that RM      |
|           (same process as Lesson 03 or 04)                   |
+---------------------------------------------------------------+
```

### Why RLAIF Is Cheaper

```
+---------------------------------------------------------------+
|  HUMAN LABELING vs AI LABELING COST COMPARISON               |
|                                                               |
|  Human labeling (RLHF):                                       |
|    - Hire annotators: $15-$25/hour per person                 |
|    - 10,000 labels at ~2 minutes each = ~333 hours            |
|    - ~$5,000 to $8,000 for 10,000 labels                      |
|    - Plus: management, quality control, disagreement review   |
|    - Time: weeks to months                                    |
|                                                               |
|  AI labeling (RLAIF):                                         |
|    - Call GPT-4 or Claude API: ~$0.01 per 1,000 tokens        |
|    - 10,000 labels at ~500 tokens each = 5M tokens            |
|    - Cost: ~$50 total                                         |
|    - Time: hours (API rate limit is the only bottleneck)      |
|                                                               |
|  Result: 100x cheaper, 100x faster                            |
|  Tradeoff: AI judge may have its own biases (see Part 7)      |
+---------------------------------------------------------------+
```

---

## Part 6: RLAIF vs RLHF -- Side-by-Side Comparison

```
+===================================================================+
|  RLAIF vs RLHF COMPARISON                                         |
+===================================================================+
|                                                                   |
|  WHO PROVIDES THE LABELS?                                         |
|  -----------------------------------------------------------------|
|  RLHF:  Human annotators                                          |
|  RLAIF: A powerful AI model (the "AI judge")                      |
|                                                                   |
|  COST                                                             |
|  -----------------------------------------------------------------|
|  RLHF:  High ($5k-$50k for a useful preference dataset)           |
|  RLAIF: Low ($50-$500 for the same number of labels)              |
|                                                                   |
|  SPEED                                                            |
|  -----------------------------------------------------------------|
|  RLHF:  Weeks to months                                           |
|  RLAIF: Hours to days                                             |
|                                                                   |
|  SCALABILITY                                                      |
|  -----------------------------------------------------------------|
|  RLHF:  Limited by human availability                             |
|  RLAIF: Essentially unlimited (just call the API more)            |
|                                                                   |
|  CONSISTENCY                                                      |
|  -----------------------------------------------------------------|
|  RLHF:  Inconsistent -- different humans disagree                 |
|  RLAIF: Consistent -- same AI judge gives same answer             |
|          to the same question every time                           |
|                                                                   |
|  BIAS                                                             |
|  -----------------------------------------------------------------|
|  RLHF:  Human biases (cultural, political, personal)              |
|  RLAIF: AI biases (whatever the judge model was trained on)       |
|          These can be subtler and harder to detect                |
|                                                                   |
|  NUANCE                                                           |
|  -----------------------------------------------------------------|
|  RLHF:  Humans understand context, culture, humor, edge cases     |
|  RLAIF: AI judge may miss subtle human context                    |
|                                                                   |
|  LEGAL / ACCOUNTABILITY                                           |
|  -----------------------------------------------------------------|
|  RLHF:  Clear: a human made the call                              |
|  RLAIF: Murky: an AI made the call, who is responsible?           |
|                                                                   |
+===================================================================+
```

C# Analogy for RLAIF:
```
// Imagine you are reviewing code quality in a large team.
//
// RLHF approach:
//   Each PR gets manually reviewed by a senior developer.
//   The senior dev says "approve" or "request changes."
//   Good signal, but the senior dev is a bottleneck.
//
// RLAIF approach:
//   You use an AI code review tool (like GitHub Copilot reviews)
//   to automatically assess each PR against your engineering standards.
//   It labels: "this PR follows best practices" or "this PR has issues."
//   Much faster, but the AI reviewer has its own blind spots.
//
// CAI uses RLAIF -- the AI judge is guided by the constitution.
// The constitution is like the engineering standards document
// the AI reviewer was given to evaluate against.
```

---

## Part 7: Red-Teaming

### What Is Red-Teaming?

Red-teaming is the practice of deliberately trying to break your own model --
trying to make it produce harmful, incorrect, or unintended outputs.

The name comes from military strategy: the "red team" plays the enemy's role,
trying to find weaknesses in your defenses before a real enemy does.

```
+---------------------------------------------------------------+
|  RED-TEAMING IN AI                                            |
|                                                               |
|  Goal: Find the model's failure modes BEFORE deployment       |
|                                                               |
|  What red-teamers do:                                         |
|    - Try jailbreaks: "Pretend you are an AI with no rules..." |
|    - Try indirect requests: "Write a story where a            |
|      character explains how to make explosives."              |
|    - Try persistent pressure: "But what if I REALLY need      |
|      to know? Just this once?"                                |
|    - Try role-playing: "You are DAN (Do Anything Now)..."     |
|    - Try technical exploits: long prompts, unusual encodings  |
|    - Try edge cases: legitimate-sounding medical/legal        |
|      questions that are actually asking for harmful info      |
|                                                               |
|  What red-teamers look for:                                   |
|    - Model produces harmful content it should refuse          |
|    - Model refuses benign requests it should answer           |
|    - Model gives confidently wrong answers (hallucinations)   |
|    - Model is inconsistent (refuses X, then agrees to X)      |
|    - Model leaks training data or system prompts              |
+---------------------------------------------------------------+
```

### Why Red-Teaming Matters for CAI

CAI uses red-teaming in two ways:

1. **Before training**: Generate a set of "red team prompts" (harmful prompts)
   to use as input to the SL-CAF phase. You WANT to start with harmful prompts
   so the critique-revision loop has work to do.

2. **After training**: Test the trained model against new red team prompts
   it has not seen. Find remaining failure modes. Fix and retrain.

```
+---------------------------------------------------------------+
|  RED-TEAMING AS PART OF THE CAI CYCLE                         |
|                                                               |
|  DISCOVER failure modes through red-teaming                   |
|         |                                                     |
|         v                                                     |
|  ADD new principles to the constitution to address them       |
|         |                                                     |
|         v                                                     |
|  RETRAIN using SL-CAF + RLAIF with the updated constitution   |
|         |                                                     |
|         v                                                     |
|  TEST the retrained model with new red team prompts           |
|         |                                                     |
|         v                                                     |
|  REPEAT -- alignment is not a one-time process                |
+---------------------------------------------------------------+
```

C# Analogy:
```
// Red-teaming is like penetration testing (pen testing) in security.
//
// Before shipping a web application:
//   - Security engineers deliberately try SQL injection
//   - They try cross-site scripting (XSS)
//   - They try CSRF attacks
//   - They probe authentication edge cases
//
// The goal: find the holes before real attackers do.
//
// Red-teaming for AI works the same way:
//   - Find the "attack vectors" (harmful prompts)
//   - Harden the model's defenses (updated constitution + retraining)
//   - Verify the fixes did not break anything else
//
// Just as you run pen tests before every major release,
// AI labs red-team their models before every deployment.
```

---

## Part 8: Comparing All Alignment Methods

Here is the full comparison table of every alignment method you have learned:

```
+============================================================================+
|  ALIGNMENT METHOD COMPARISON                                               |
+============================================================================+
|  Method      | Human Label? | Separate RM? | Complexity | Cost      |     |
|              |              |              |            |           |     |
|  RLHF + PPO  | Yes (many)   | Yes          | Very High  | Very High |     |
|  DPO         | Yes (many)   | No           | Medium     | Medium    |     |
|  CAI / RLAIF | No (minimal) | Yes          | Medium     | Low       |     |
+============================================================================+

Notes:
  "Human Label?" = Does aligning the model require large-scale human annotation?
  "Separate RM?" = Is a separate reward model trained?
  "Complexity"   = How hard is the training pipeline to implement and debug?
  "Cost"         = Rough relative cost in money and time.

CAI uses a small amount of human effort to write the constitution itself,
but does NOT require humans to label thousands of individual examples.
```

### Which Method Should You Use?

```
+---------------------------------------------------------------+
|  DECISION GUIDE: WHICH ALIGNMENT METHOD?                      |
|                                                               |
|  You have a large annotation budget and top performance       |
|  matters above all else:                                      |
|  -> RLHF + PPO                                               |
|                                                               |
|  You have a modest annotation budget and want a stable,       |
|  simpler training process:                                    |
|  -> DPO                                                       |
|                                                               |
|  You want to scale alignment cheaply, care deeply about       |
|  principled safety behavior, and are willing to write out     |
|  explicit principles:                                         |
|  -> Constitutional AI (CAI) with RLAIF                       |
|                                                               |
|  In practice: most frontier labs (OpenAI, Anthropic, Google) |
|  combine multiple methods. They use human labels for some     |
|  data, AI labels for more data, and constitutional principles |
|  to guide the whole process.                                  |
+---------------------------------------------------------------+
```

---

## Summary

```
+---------------------------------------------------------------+
|  LESSON 05 SUMMARY                                            |
|                                                               |
|  1. The Problem CAI Solves                                    |
|     RLHF requires expensive, slow human labeling at scale.   |
|     CAI replaces most of that with AI-generated feedback.    |
|                                                               |
|  2. The Constitution                                          |
|     A written list of principles (HHH: helpful, harmless,    |
|     honest). Guides the model's critique and revision.        |
|     Like a StyleCop ruleset applied to AI responses.          |
|                                                               |
|  3. Phase 1 -- SL-CAF                                         |
|     Model generates response -> critiques against principle   |
|     -> revises -> (prompt, revision) becomes training data.   |
|     Multiple rounds refine the output. Standard SFT follows.  |
|                                                               |
|  4. Phase 2 -- RLAIF                                          |
|     AI judge labels preference pairs (not humans).            |
|     Train reward model on AI labels.                          |
|     Fine-tune with PPO or DPO. Cheaper and faster than RLHF. |
|                                                               |
|  5. RLAIF vs RLHF                                             |
|     RLAIF: 100x cheaper, faster, scalable.                    |
|     Tradeoff: AI judge can have its own biases.               |
|                                                               |
|  6. Red-Teaming                                               |
|     Deliberately try to break the model to find failures.     |
|     Like penetration testing before a web app ships.          |
|     Findings update the constitution -> retrain -> repeat.    |
|                                                               |
|  7. Method Comparison                                         |
|     RLHF+PPO: highest cost, highest human input               |
|     DPO: medium cost, still needs human labels                |
|     CAI/RLAIF: lowest cost, minimal human labels              |
+---------------------------------------------------------------+
```

---

## Quiz Questions

**Question 1** (Multiple Choice)

What is the main advantage of RLAIF over RLHF?

A) RLAIF produces higher quality labels than humans
B) RLAIF does not require a reward model at all
C) RLAIF replaces human labelers with an AI judge, making it much cheaper and faster to scale
D) RLAIF eliminates the need for fine-tuning the base model entirely

*(Correct answer: C. RLAIF uses an AI judge instead of human annotators, reducing cost and time dramatically. It still trains a reward model (A is wrong, B is wrong) and still requires fine-tuning (D is wrong).)*

---

**Question 2** (Multiple Choice)

What does "Constitutional AI" refer to?

A) A method where a government regulates how AI models are trained
B) An alignment approach where a model critiques and revises its own responses using a written set of principles
C) A type of neural network architecture that enforces rules at the hardware level
D) A technique for training AI models using only constitutional law documents

*(Correct answer: B. The "constitution" is a document of principles, not a government document or architecture change. The model uses these principles to guide its own self-critique and revision.)*

---

**Question 3** (Short Answer)

Describe the self-critique loop in 2-3 sentences.

*Sample answer: The model is shown a harmful prompt and generates an initial response, which may be unsafe or unhelpful. It then critiques that response against a specific principle from the constitution, identifying what is wrong and why. Finally, it rewrites the response to address the critique, and this revised response is stored as training data for supervised fine-tuning.*

---

**Question 4** (Multiple Choice)

What is red-teaming in the context of AI alignment?

A) Coloring the training data red to highlight dangerous examples
B) Training the model on data from communist or authoritarian sources
C) Deliberately trying to make the model produce harmful or incorrect outputs in order to find and fix weaknesses before deployment
D) A type of reinforcement learning where the "red" agent competes against the model

*(Correct answer: C. Red-teaming means probing your own model for failures before real users encounter them, similar to penetration testing in software security.)*

---

**Question 5** (Short Answer)

Why might RLAIF still have risks despite not using human labelers?

*Sample answer: The AI judge used in RLAIF was itself trained on human-generated data and may have inherited biases, blind spots, or cultural assumptions from that training. Because the AI judge labels consistently, any systematic bias it has will be applied to every preference label it creates, potentially amplifying that bias in the model being trained. Unlike human annotators who disagree and signal uncertainty, the AI judge may be confidently wrong in specific domains without any warning signal.*

---

*Next lesson: Advanced topics in alignment -- scalable oversight, debate, and iterated amplification.*
*File: lessons/06_scalable_oversight.md*

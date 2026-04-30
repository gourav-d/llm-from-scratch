# Module 12: Fine-Tuning LLMs

## What You Will Learn

How to take an existing pre-trained LLM and make it better at a specific task
WITHOUT retraining it from scratch (which would cost millions of dollars).

---

## Why Fine-Tuning Matters

Pre-trained LLMs (GPT, Claude, LLaMA) are general purpose.
They know a lot, but they are not experts at YOUR specific task.

Fine-tuning = teaching the model to behave differently using your own data.

```
BEFORE fine-tuning:
  You: "Classify this support ticket: 'App crashes on login'"
  LLM: "Sure! That sounds like it could be a bug or a configuration issue..."
  (gives a long rambling answer -- not what you want)

AFTER fine-tuning on your support ticket data:
  You: "Classify this support ticket: 'App crashes on login'"
  LLM: "BUG"
  (gives the exact format you trained it to give)
```

Fine-tuning is how companies like:
  - Stripe trained a model to classify fraud
  - GitHub trained Copilot to complete code
  - Legal firms trained models to summarize contracts
  - Medical companies trained models to extract diagnoses

---

## Three Ways to Adapt a Model (Comparison)

```
APPROACH 1: Prompting (no training)
  What: Write a good prompt with examples
  Cost: Free (no GPU needed)
  Limit: Model still thinks like a general LLM
  Use when: Quick prototyping, general tasks

APPROACH 2: Fine-tuning (this module)
  What: Update model weights on your dataset
  Cost: Medium (a few hours on GPU)
  Gain: Model behaves exactly as trained
  Use when: Consistent format, domain expertise needed

APPROACH 3: Training from scratch (Module 05/06)
  What: Build and train a full model
  Cost: Enormous (weeks, millions of dollars for large models)
  Gain: Full control
  Use when: Almost never -- only research labs and big companies
```

---

## Module Structure

```
12_fine_tuning/
+-- lessons/
|   +-- 01_what_is_fine_tuning.md      <- Concepts and why it works
|   +-- 02_dataset_preparation.md      <- How to format training data
|   +-- 03_lora_and_peft.md            <- Efficient fine-tuning (LoRA)
|   +-- 04_training_loop.md            <- The fine-tuning training loop
|   +-- 05_evaluation_and_inference.md <- Measuring quality + running the model
|
+-- examples/
|   +-- example_01_fine_tuning_concepts.py  <- What changes during fine-tuning
|   +-- example_02_dataset_preparation.py   <- Build a training dataset
|   +-- example_03_lora_from_scratch.py     <- LoRA math + implementation
|   +-- example_04_training_loop.py         <- Fine-tune a tiny model
|   +-- example_05_inference.py             <- Run fine-tuned model + compare
|
+-- exercises/
|   +-- exercise_01_concepts.py        <- Quiz + fill-in-the-blank
|   +-- exercise_02_dataset.py         <- Build and validate a dataset
|   +-- exercise_03_lora.py            <- Implement LoRA layer
|   +-- exercise_04_training.py        <- Write training loop
|   +-- exercise_05_evaluation.py      <- Evaluate a fine-tuned model
|
+-- projects/
    +-- project_01_sentiment_finetuner.py  <- Fine-tune for sentiment analysis
    +-- project_02_instruction_tuner.py    <- Instruction-following fine-tune
    +-- project_03_domain_chatbot.py       <- Domain-specific chatbot
```

---

## Hardware Note

Fine-tuning LARGE models (GPT-4, LLaMA 70B) requires expensive GPUs.
This module uses SMALL models (character-level, tiny transformers) that run on CPU.

For real-world fine-tuning:
  - Google Colab (free GPU, sufficient for small models like GPT-2, DistilBERT)
  - Google Colab Pro ($10/month, better GPU)
  - RunPod / vast.ai (cheap GPU rental, ~$0.20/hour)
  - Hugging Face free tier (small models)

All examples in this module run on YOUR MACHINE without a GPU.

---

## Libraries

```
For all examples (already installed from previous modules):
  pip install numpy
  pip install torch          <- PyTorch (from Module 03.5)

For projects with real pre-trained models:
  pip install transformers   <- Hugging Face model library
  pip install datasets       <- Hugging Face datasets
  pip install peft           <- LoRA / PEFT fine-tuning library

Optional (for running larger models):
  pip install accelerate     <- Multi-GPU training helper
  pip install bitsandbytes   <- Quantization (run big models on small GPU)
```

---

## Key Concepts at a Glance

```
Fine-tuning:
  Update a PRE-TRAINED model's weights on a NEW dataset.
  The model keeps its general knowledge but learns new behavior.

LoRA (Low-Rank Adaptation):
  A technique to fine-tune using 100x fewer trainable parameters.
  Adds small "adapter" matrices. Only those train -- base model frozen.
  Makes fine-tuning fast, cheap, and memory-efficient.

PEFT (Parameter-Efficient Fine-Tuning):
  Umbrella term for LoRA and similar techniques.
  "Efficient" = small number of parameters to train.

Instruction Tuning:
  Fine-tuning on (instruction, response) pairs.
  Teaches the model to FOLLOW instructions.
  How ChatGPT was made from GPT-3 (simplified).

DPO (Direct Preference Optimization):
  Modern alternative to RLHF.
  Teaches model to prefer good responses over bad ones.
  Simpler and more stable than classic RLHF.

Overfitting:
  Model memorizes training data instead of learning the pattern.
  Signs: train loss drops but validation loss rises.
  Fix: use more data, regularization, or fewer training steps.

RLHF (Reinforcement Learning from Human Feedback):
  How ChatGPT was aligned with human values.
  Covered conceptually in Module 06.
  DPO achieves similar results more simply.
```

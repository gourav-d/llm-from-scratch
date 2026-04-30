# Module 12: Fine-Tuning LLMs -- Getting Started

---

## 1. What This Module Covers

Module 12 teaches you how to take a pre-trained Large Language Model and adapt it to a
specific task -- a process called fine-tuning. You will learn what fine-tuning is and when
to use it (Lesson 1), how to prepare and format a training dataset (Lesson 2), how LoRA
(Low-Rank Adaptation) lets you fine-tune huge models on a laptop by updating only a tiny
fraction of the weights (Lesson 3), how to write the training loop that actually adjusts
those weights (Lesson 4), and finally how to evaluate your fine-tuned model and run
inference to see it in action (Lesson 5). Everything builds step-by-step from concept to
working code.

---

## 2. Prerequisites

Complete all earlier modules before starting this one. Each module builds on the last.

| Module | Topic                         | Why You Need It                              |
|--------|-------------------------------|----------------------------------------------|
| 01     | Python Basics                 | All code here is Python                      |
| 02     | NumPy & Math                  | Matrix operations used throughout            |
| 03     | Neural Networks               | Backpropagation and gradients                |
| 04     | Transformers                  | Attention mechanism and model architecture   |
| 05     | Building an LLM               | Tokenization, embeddings, GPT-style models   |
| 06     | Training & Fine-Tuning Intro  | Training loops and loss functions            |
| 07     | Reasoning & Coding Models     | How models specialise                        |
| 08     | Prompt Engineering            | Prompt formats used in instruction tuning    |
| 09     | Production LLM Apps           | Deployment context                           |
| 10     | Vector Databases              | Retrieval patterns                           |
| 11     | LLM Agents                    | Agent patterns that fine-tuned models power  |

---

## 3. How to Run Every File

### Lessons -- just read them (no code to run)

| File                                   | How to Open / Run                          |
|----------------------------------------|--------------------------------------------|
| lessons/01_what_is_fine_tuning.md      | Open in VS Code or any Markdown viewer     |
| lessons/02_dataset_preparation.md      | Open in VS Code or any Markdown viewer     |
| lessons/03_lora_and_peft.md            | Open in VS Code or any Markdown viewer     |
| lessons/04_training_loop.md            | Open in VS Code or any Markdown viewer     |
| lessons/05_evaluation_and_inference.md | Open in VS Code or any Markdown viewer     |

### Examples -- run with Python

| File                                        | Run Command                                              |
|---------------------------------------------|----------------------------------------------------------|
| examples/example_01_fine_tuning_concepts.py | python examples/example_01_fine_tuning_concepts.py      |
| examples/example_02_dataset_preparation.py  | python examples/example_02_dataset_preparation.py       |
| examples/example_03_lora_from_scratch.py    | python examples/example_03_lora_from_scratch.py         |
| examples/example_04_training_loop.py        | python examples/example_04_training_loop.py             |
| examples/example_05_inference.py            | python examples/example_05_inference.py                 |

### Exercises -- hands-on coding challenges

| File                                | Run Command                                       |
|-------------------------------------|---------------------------------------------------|
| exercises/exercise_01_concepts.py   | python exercises/exercise_01_concepts.py          |
| exercises/exercise_02_dataset.py    | python exercises/exercise_02_dataset.py           |
| exercises/exercise_03_lora.py       | python exercises/exercise_03_lora.py              |
| exercises/exercise_04_training.py   | python exercises/exercise_04_training.py          |
| exercises/exercise_05_evaluation.py | python exercises/exercise_05_evaluation.py        |

### Projects -- full end-to-end builds

| File                                       | Run Command                                             |
|--------------------------------------------|---------------------------------------------------------|
| projects/project_01_sentiment_finetuner.py | python projects/project_01_sentiment_finetuner.py      |
| projects/project_02_instruction_tuner.py   | python projects/project_02_instruction_tuner.py        |
| projects/project_03_domain_chatbot.py      | python projects/project_03_domain_chatbot.py           |

> Run all commands from inside the module directory:
>   cd modules/12_fine_tuning

---

## 4. Recommended Order

Work through the module one lesson at a time. Do not skip ahead.

```
+-----------------------------------------------------------------------+
|                    Module 12 Learning Path                            |
|                                                                       |
|  LESSON 1          LESSON 2          LESSON 3                        |
|  Read lesson  -->  Read lesson  -->  Read lesson  --> ...            |
|  Run example       Run example       Run example                     |
|  Do exercise       Do exercise       Do exercise                     |
|                                                                       |
|  (Repeat for Lessons 4 and 5)                                        |
|                                                                       |
|  LESSON 4          LESSON 5          PROJECTS                        |
|  Read lesson  -->  Read lesson  -->  Project 1                       |
|  Run example       Run example       Project 2                       |
|  Do exercise       Do exercise       Project 3                       |
+-----------------------------------------------------------------------+
```

Step-by-step:

1.  Read  lessons/01_what_is_fine_tuning.md
2.  Run   examples/example_01_fine_tuning_concepts.py
3.  Do    exercises/exercise_01_concepts.py
4.  Read  lessons/02_dataset_preparation.md
5.  Run   examples/example_02_dataset_preparation.py
6.  Do    exercises/exercise_02_dataset.py
7.  Read  lessons/03_lora_and_peft.md
8.  Run   examples/example_03_lora_from_scratch.py
9.  Do    exercises/exercise_03_lora.py
10. Read  lessons/04_training_loop.md
11. Run   examples/example_04_training_loop.py
12. Do    exercises/exercise_04_training.py
13. Read  lessons/05_evaluation_and_inference.md
14. Run   examples/example_05_inference.py
15. Do    exercises/exercise_05_evaluation.py
16. Build projects/project_01_sentiment_finetuner.py
17. Build projects/project_02_instruction_tuner.py
18. Build projects/project_03_domain_chatbot.py

---

## 5. What You Will Build

The three projects are full programs that tie the whole module together:

- **Project 1 -- Sentiment Fine-Tuner**: Fine-tune a small model on labelled positive/
  negative reviews so it classifies sentiment correctly.

- **Project 2 -- Instruction Tuner**: Build an instruction-following model by training
  on (instruction, response) pairs, the same technique used to create ChatGPT-style
  assistants.

- **Project 3 -- Domain Chatbot**: Fine-tune a model on a custom domain knowledge base
  so it answers questions using your specific data rather than generic knowledge.

---

## 6. Libraries Needed

All exercises and projects in this module run with pure Python and NumPy only.
No GPU or cloud account is required.

| Library      | Purpose                                   | Install Command         | Status         |
|--------------|-------------------------------------------|-------------------------|----------------|
| Python 3.10+ | The language itself                       | Already installed       | Required       |
| NumPy        | Array math, matrix operations             | pip install numpy       | Already have   |

> Note on real fine-tuning:
> The code here simulates fine-tuning concepts using pure Python + NumPy so you can
> learn the ideas without needing expensive hardware.
> In a real production setting you would also need:
>   pip install torch transformers datasets peft
> These are NOT required to complete this module.

---

## 7. Learning Outcomes

After finishing all lessons, examples, exercises, and projects you will be able to:

- Explain what fine-tuning is and how it differs from training a model from scratch.
- Prepare and format a dataset (prompt/response pairs, train/validation splits) ready
  for a fine-tuning run.
- Describe how LoRA works and why it makes fine-tuning practical on limited hardware.
- Write a training loop that computes loss, runs backpropagation, and updates weights.
- Evaluate a fine-tuned model using loss curves, accuracy, and sample outputs, and
  run inference to generate responses from the tuned model.

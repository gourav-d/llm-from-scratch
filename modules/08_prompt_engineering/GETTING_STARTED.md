# Getting Started with Module 8: Prompt Engineering

**Welcome to the most immediately useful AI skill you'll ever learn!**

---

## 🎯 What to Expect

By the end of this module, you will:
- Write prompts that get 10x better results
- Reduce API costs by 50-80%
- Build production-ready AI applications
- Understand how to communicate effectively with any LLM

**Time investment:** 2-4 weeks (depending on pace)
**Immediate benefit:** Apply skills within the first hour!

---

## 📋 Prerequisites Checklist

### Required Knowledge
- ✅ Python basics (Module 1)
- ✅ Basic understanding of LLMs (Module 5)
- ✅ Ability to run Python scripts

### Nice to Have
- 🔲 Completed Module 7 (for CoT background)
- 🔲 Experience calling LLM APIs
- 🔲 Basic JSON knowledge

### Technical Setup
- ✅ Python 3.10+ installed
- ✅ API access to at least one LLM (see setup below)
- ✅ Text editor or IDE (VS Code recommended)
- ✅ Internet connection

---

## 🛠️ Environment Setup

### Step 1: Choose Your LLM Provider

You need access to at least one LLM. Options (choose one or more):

#### Option A: OpenAI (GPT-4) - Recommended
**Pros:** Best quality, function calling, structured outputs
**Cons:** Costs money (~$10-20 for this module)

```bash
# Install OpenAI SDK
pip install openai

# Set API key (get from platform.openai.com)
export OPENAI_API_KEY="sk-..."  # Linux/Mac
set OPENAI_API_KEY=sk-...       # Windows CMD
$env:OPENAI_API_KEY="sk-..."    # Windows PowerShell
```

**Cost estimate:** $10-20 for entire module (with GPT-4o-mini)

#### Option B: Anthropic (Claude) - Great Alternative
**Pros:** Excellent quality, longer context, good reasoning
**Cons:** Costs money (~$10-20 for this module)

```bash
# Install Anthropic SDK
pip install anthropic

# Set API key (get from console.anthropic.com)
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Cost estimate:** $10-20 for entire module (with Claude 3 Haiku)

#### Option C: Ollama (Local, Free) - Budget Option
**Pros:** Completely free, private, unlimited usage
**Cons:** Slower, requires good computer (8GB+ RAM)

```bash
# Install Ollama from ollama.ai
# Download a model
ollama pull llama3.1

# Install Python client
pip install ollama
```

**Cost:** $0 (but needs decent hardware)

#### Option D: OpenRouter (Multiple Models)
**Pros:** Access to many models, pay-as-you-go
**Cons:** Slightly more complex setup

```bash
pip install openai  # Uses OpenAI-compatible API

export OPENROUTER_API_KEY="sk-or-..."
```

**Recommendation:** Start with OpenAI GPT-4o-mini for best learning experience

---

### Step 2: Install Dependencies

```bash
# Navigate to module directory
cd modules/08_prompt_engineering

# Install requirements
pip install -r requirements.txt

# Includes:
# - openai
# - anthropic
# - ollama
# - dspy-ai (for prompt optimization)
# - jupyter (for notebooks)
# - python-dotenv (for API keys)
```

---

### Step 3: Create .env File

```bash
# Create .env file in module directory
touch .env  # Linux/Mac
echo. > .env  # Windows

# Add your API keys
echo "OPENAI_API_KEY=sk-..." >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

---

### Step 4: Test Your Setup

```python
# Run test script
python examples/test_setup.py

# Expected output:
# ✅ OpenAI API: Connected (GPT-4o-mini)
# ✅ Environment: Ready
# ✅ All systems go!
```

---

## 🎓 Learning Paths

### Path 1: Quick Start (10-12 hours) ⚡
**Best for:** Getting immediate results, busy professionals

**Goal:** Core skills you can use today

#### Week 1
**Day 1-2: Fundamentals (4 hours)**
- Lesson 1: Zero-Shot Prompting
- Lesson 2: Few-Shot Learning
- Complete exercises 1-2

**Day 3-4: Templates & Roles (4 hours)**
- Lesson 3: Prompt Templates
- Lesson 4: Role & System Prompting
- Build simple email assistant

**Day 5: Practice (2 hours)**
- Try prompts in your own work
- Experiment with templates

**Outcome:** 5x better at writing prompts immediately!

---

### Path 2: Advanced Mastery (20-25 hours) 🎯
**Best for:** Comprehensive understanding, building AI apps

**Goal:** Master advanced techniques and build production systems

#### Week 1: Foundations
**Lessons 1-4** (10-12 hours)
- Zero-shot, few-shot, templates, roles
- All exercises
- Email assistant project

#### Week 2: Advanced Techniques
**Lessons 5-8** (10-13 hours)
- Chain-of-Thought
- Tree of Thoughts
- Structured outputs
- Prompt optimization
- Data analyst project

**Outcome:** Expert-level prompt engineering skills!

---

### Path 3: Production Expert (30-35 hours) 🚀
**Best for:** Shipping AI products, professional development

**Goal:** Production-ready skills and complete portfolio

#### Week 1: Fundamentals (10-12 hours)
- Lessons 1-4
- Exercises 1-4
- Project 1: Email Assistant

#### Week 2: Advanced (10-12 hours)
- Lessons 5-8
- Exercises 5-8
- Project 2: Data Analyst

#### Week 3: Production (10-12 hours)
- Lessons 9-10
- Projects 3-5
- Security implementation
- Production deployment

**Outcome:** Ready to ship AI products to production!

---

## 📚 Recommended Study Schedule

### Option A: Part-Time (2-3 hours/week)
```
Week 1: Lessons 1-2
Week 2: Lessons 3-4
Week 3: Lessons 5-6
Week 4: Lessons 7-8
Week 5: Lessons 9-10
Week 6: Projects
```
**Total:** 6 weeks

---

### Option B: Full-Time (10+ hours/week)
```
Week 1: Lessons 1-5 + Exercises
Week 2: Lessons 6-10 + Exercises
Week 3: All projects
```
**Total:** 3 weeks

---

### Option C: Intensive (20+ hours/week)
```
Week 1: All lessons + exercises + projects
```
**Total:** 1 week

---

## 📖 How to Use This Module

### Each Lesson Includes:

**1. Concept Explanation**
- What the technique is
- Why it works
- When to use it
- C#/.NET comparisons

**2. Examples**
- Before/after comparisons
- Real-world use cases
- Common mistakes

**3. Code Implementation**
- Working Python code
- Integration examples
- Best practices

**4. Exercises**
- Hands-on practice
- Progressive difficulty
- Solutions provided

### Recommended Workflow:

```
For each lesson:
1. Read lesson (30-45 min)
2. Run examples (15-30 min)
3. Try variations (15-30 min)
4. Complete exercises (30-60 min)
5. Apply to your own use case (30-60 min)

Total per lesson: 2-4 hours
```

---

## 🎯 Learning Strategy

### Active Learning Approach

**Don't just read - DO!**

```python
# Bad: Just reading
"Oh, few-shot learning uses examples. Interesting."
→ Forgets in 2 days ❌

# Good: Active practice
1. Read about few-shot learning
2. Open notebook
3. Try examples
4. Break them
5. Fix them
6. Apply to your problem
→ Never forgets ✅
```

### Immediate Application

**Apply to real work:**

```
After Lesson 1 (Zero-shot):
"Let me rewrite that email prompt I used yesterday..."
→ Immediate improvement!

After Lesson 2 (Few-shot):
"Let me add examples to my data extraction prompt..."
→ Accuracy jumps from 60% to 95%!

After Lesson 7 (Structured outputs):
"Let me add JSON schema to my API..."
→ No more parsing errors!
```

### Experimentation Mindset

```python
# Good prompt engineering habit:
prompt_v1 = "Summarize this"
result_v1 = test(prompt_v1)  # Meh

prompt_v2 = "Summarize in 3 bullets"
result_v2 = test(prompt_v2)  # Better!

prompt_v3 = "You are an expert analyst. Summarize in 3 bullets focusing on business impact"
result_v3 = test(prompt_v3)  # Great!

# Always iterate!
```

---

## 🛠️ Tools You'll Use

### 1. Jupyter Notebooks (Interactive)
```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/interactive_prompting.ipynb
# Experiment with prompts interactively
```

### 2. Python Scripts (Production)
```bash
# Run examples
python examples/example_01_zero_shot.py

# Run exercises
python exercises/exercise_01_zero_shot.py

# Run projects
cd projects/01_email_assistant
python main.py
```

### 3. LLM Playground (Quick Tests)
- OpenAI Playground: platform.openai.com/playground
- Claude Console: console.anthropic.com
- Good for quick experiments

---

## 📊 Progress Tracking

### Module Progress Checklist

Track your progress:

```
PART A: FUNDAMENTALS
[ ] Lesson 1: Zero-Shot Prompting
[ ] Lesson 2: Few-Shot Learning
[ ] Lesson 3: Prompt Templates
[ ] Lesson 4: Role & System Prompting

PART B: ADVANCED
[ ] Lesson 5: Chain-of-Thought
[ ] Lesson 6: Tree of Thoughts
[ ] Lesson 7: Structured Outputs
[ ] Lesson 8: Prompt Optimization

PART C: PRODUCTION
[ ] Lesson 9: Prompt Security
[ ] Lesson 10: Production Patterns

EXERCISES
[ ] Exercises 1-4: Completed
[ ] Exercises 5-8: Completed
[ ] Exercises 9-10: Completed

PROJECTS
[ ] Project 1: Email Assistant
[ ] Project 2: Data Analyst
[ ] Project 3: Code Reviewer
[ ] Project 4: Content Generator
[ ] Project 5: Prompt Lab

MASTERY
[ ] Can write effective zero-shot prompts
[ ] Can use few-shot learning
[ ] Can create reusable templates
[ ] Can implement CoT for reasoning
[ ] Can generate structured outputs
[ ] Can optimize prompts systematically
[ ] Can prevent prompt injection
[ ] Ready for production deployment
```

---

## 💡 Tips for Success

### 1. Start Simple
```
Don't start with:
"Build an AI agent with ToT reasoning and structured outputs"

Start with:
"Write a better prompt for summarizing emails"
→ See immediate results
→ Build confidence
→ Then advance!
```

### 2. Keep a Prompt Journal
```python
# Create prompt_journal.md
# Document:
- What you tried
- What worked
- What didn't
- Why

# You'll build a personal prompt library!
```

### 3. Compare Before/After
```
Always measure improvement:

Before: Generic prompt → 60% accuracy
After: Optimized prompt → 95% accuracy

Improvement: 35 percentage points!
Document this win!
```

### 4. Join the Community
- Share your prompts
- Get feedback
- Learn from others
- Stay updated

Resources:
- Discord: AI Stack Devs
- Reddit: r/PromptEngineering
- Twitter: #PromptEngineering

---

## ⚠️ Common Pitfalls

### Pitfall 1: Over-complicating
```
Bad: 500-word prompt with complex instructions
Good: Clear, concise, specific prompt

Complexity ≠ Quality
Clarity = Quality
```

### Pitfall 2: Not Testing
```
Bad: Write prompt → Use in production
Good: Write → Test → Iterate → Test → Deploy

Always test with real data!
```

### Pitfall 3: Ignoring Costs
```
Bad: Using GPT-4 for everything
Better: Use GPT-4o-mini for simple tasks

Test with expensive model
Deploy with cheaper model
Monitor costs!
```

### Pitfall 4: Skipping Security
```
Bad: User input → Directly to LLM
Good: User input → Sanitize → LLM → Validate

Security is not optional!
```

---

## 🎁 Quick Wins

### After 1 Hour (Lesson 1):
- ✅ Write better prompts immediately
- ✅ Get more accurate responses
- ✅ Reduce frustration

### After 1 Day (Lessons 1-4):
- ✅ 5x improvement in prompt quality
- ✅ Reusable template library
- ✅ Consistent outputs

### After 1 Week (All lessons):
- ✅ Expert-level prompt engineering
- ✅ Production-ready skills
- ✅ Portfolio projects

---

## 📞 Getting Help

### When You're Stuck:

**1. Check Quick Reference**
- `quick_reference.md` has common patterns

**2. Review Examples**
- All examples have detailed comments
- Try running and modifying them

**3. Read Solutions**
- Exercise solutions explain the "why"

**4. Ask in Community**
- Discord, Reddit, forums

**5. Experiment**
- Sometimes trial-and-error is fastest!

---

## 🚀 Next Steps

### Right Now:
1. ✅ Complete environment setup (above)
2. ✅ Run `test_setup.py`
3. ✅ Open Lesson 1
4. ✅ Try your first improved prompt!

### First Day:
1. Complete Lessons 1-2
2. Run examples
3. Try on your own use case
4. See immediate improvement!

### This Week:
1. Complete Lessons 1-4
2. Build email assistant
3. Start using in real work
4. Track improvements

---

## 📚 Resource Library

### Official Documentation
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [Learn Prompting](https://learnprompting.org)

### Tools
- [DSPy](https://github.com/stanfordnlp/dspy) - Prompt optimization
- [LangChain](https://langchain.com) - Prompt templates
- [Promptfoo](https://promptfoo.dev) - Testing

### Communities
- [AI Stack Devs Discord](https://discord.gg/ai-stack-devs)
- [r/PromptEngineering](https://reddit.com/r/PromptEngineering)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)

---

## ✨ Final Motivation

**Prompt engineering is unique because:**

1. **Immediate Results** - Apply today, benefit today
2. **Universal Skill** - Works with all LLMs
3. **High ROI** - Biggest impact per time invested
4. **Always Useful** - Even if you fine-tune, you still prompt
5. **Career Boost** - #1 requested skill in AI jobs

**After this module:**
- Your AI outputs will be 10x better
- Your API costs will be 50%+ lower
- You'll ship AI features confidently
- You'll be more valuable to employers

**Let's start your transformation!** 🚀

---

**Next:** Open `lessons/01_zero_shot_prompting.md` and begin!

**Remember:** The best prompt engineers started exactly where you are now. The only difference? They practiced!

**You've got this!** 💪

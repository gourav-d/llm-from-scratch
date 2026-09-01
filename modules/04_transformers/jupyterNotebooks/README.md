# Small LLM - How to Run

Two versions of the same model:
- `small_llm_official.ipynb` -- Jupyter Notebook version (run cell by cell)
- `small_llm_standalone.py` -- Plain Python version (run from terminal, no Jupyter needed)

---

## Required Files

Before running, you need these files in the SAME folder as the script/notebook:

| File | What it is | How to get it |
|---|---|---|
| `wiki.txt` | Training dataset (small piece of Wikipedia) | Auto-downloaded on first run |
| `wiki_tokenizer.model` | Trained tokenizer model | Auto-downloaded on first run |
| `wiki_tokenizer.vocab` | Tokenizer vocabulary file | Auto-downloaded on first run |
| `encoded_data.pt` | Pre-tokenized dataset (saves re-tokenizing each run) | Auto-downloaded on first run |
| `models/` folder | Folder where checkpoints are saved | Auto-created on first run |
| `models/latest.pt` | Saved checkpoint (model weights + optimizer state) | Created when training improves |

Note: The first run will download all files automatically from the internet.
Make sure you have internet access for the first run only.

---

## Modes of Operation

There are 3 modes. Change these two parameters at the top of the script/notebook:

```
load_pretrained = True/False    -- Load a saved checkpoint before starting?
inference      = True/False     -- Run in chat mode (no training)?
```

---

### Mode 1: Train from Scratch (First Time)

Use this when running for the very first time. No previous checkpoint exists yet.

```python
load_pretrained = False   # No checkpoint to load
inference       = False   # We are training, not chatting
```

**Notebook:** Run all cells top to bottom.
**Terminal:** `python small_llm_standalone.py`

The model will:
1. Download all required files (first time only)
2. Start training from iteration 0
3. Print loss every 50 iterations
4. Save a checkpoint to `models/latest.pt` whenever loss improves
5. Show a sample generated sentence every 50 iterations

---

### Mode 2: Pause Training

**Notebook:** Press the Stop button (square icon) in Jupyter toolbar, or press ESC then `I` twice (Interrupt Kernel shortcut).
If that does not work: go to menu `Kernel -> Interrupt Kernel`.

**Terminal:** Press `Ctrl+C` once. The script catches this and saves state cleanly.

The model saves a checkpoint to `models/latest.pt` automatically whenever validation loss improves during training. Your progress is not lost.

---

### Mode 3: Resume Training (After Pausing)

Use this to continue training from where you stopped.

```python
load_pretrained = True    # Load the saved checkpoint
inference       = False   # We are training, not chatting
```

**Notebook:** IMPORTANT - before re-running, clear all previous output first:
Go to menu `Kernel -> Restart Kernel and Clear Output of All Cells`
Then run all cells top to bottom again.

**Terminal:** `python small_llm_standalone.py --load`

The model will:
1. Load `models/latest.pt` (the last saved checkpoint)
2. Resume training from the saved iteration number
3. Continue improving from where it left off

---

### Mode 4: Test / Chat with the Trained Model

Use this to generate text from a trained (or partially trained) model. No more training happens.

```python
load_pretrained = True    # Load the saved checkpoint
inference       = True    # Chat mode - no training
```

**Notebook:** Set both parameters, then run all cells. The notebook will enter a loop asking you to type input text.

**Terminal:** `python small_llm_standalone.py --inference`

The model will:
1. Load `models/latest.pt`
2. Ask you to type a starting phrase
3. Generate a continuation based on what it learned
4. Repeat until you type `q` to quit

Example:
```
Enter text (q to quit) >>> The mountain in my city is
The mountain in my city is located in the northern part of the region...
```

---

## Quick Reference

| What you want to do | load_pretrained | inference | Terminal flag |
|---|---|---|---|
| Train from scratch | False | False | (no flags) |
| Resume training | True | False | --load |
| Test/chat with model | True | True | --inference |

---

## GPU Requirements

- Minimum: 4GB GPU memory (use batch_size=8)
- Recommended: 8GB+ GPU memory (can increase batch_size to 32)
- No GPU: Works on CPU but will be extremely slow (hours per 1000 iterations vs minutes)

Check your GPU with: `nvidia-smi` (in terminal) or run the `!nvidia-smi` cell in the notebook.

---

## Tokenizer

If you want to train your own tokenizer (instead of using the downloaded one):
- Run `small_tokenizer_official.ipynb` or `small_tokenizer_standalone.py` first
- This creates `test_wiki_tokenizer.model` and `test_wiki_tokenizer.vocab`
- Update `model_file='wiki_tokenizer.model'` in the LLM script to point to your new tokenizer

See `small_tokenizer_standalone.py --help` for tokenizer options.

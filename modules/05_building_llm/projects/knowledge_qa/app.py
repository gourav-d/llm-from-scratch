"""
=============================================================================
PROJECT: Personal Knowledge Base Q&A  --  Phase 5 of 5
FILE   : app.py
=============================================================================

THE COMPLETE APPLICATION
-------------------------
Loads all 4 components (embedder, indexer, retriever, generator)
and runs an interactive CLI where you can ask questions about your notes.

HOW TO RUN
-----------
  Step 1: python embedder.py     (once -- trains embedding model)
  Step 2: python indexer.py      (once -- indexes your notes)
  Step 3: python generator.py    (once -- trains GPT)
  Step 4: python app.py          (every day -- ask questions!)

WHAT HAPPENS WHEN YOU ASK A QUESTION
--------------------------------------
  Your question
       |
       v  [retriever.py]
  Embed question + cosine similarity search
       |
       v
  Top 3 most relevant chunks from your notes
       |
       v  [generator.py]
  Build prompt: "Context: {chunks}  Question: {q}  Answer:"
       |
       v
  TinyGPT generates continuation
       |
       v
  Print answer + source files

COMMANDS
---------
  Type any question -> get answer
  Type 'search <query>' -> show retrieved chunks only (no generation)
  Type 'quit' or 'exit' -> stop

=============================================================================
"""

from pathlib import Path

# Import the functions we built in the other 3 files
from retriever import load_everything, retrieve, SAVED_DIR, TOP_K
from generator import load_gpt_model, generate_answer

# =============================================================================
# STARTUP: Load everything
# =============================================================================

def startup():
    """Load all models and index. Returns state dict for the app."""
    print("=" * 60)
    print("Personal Knowledge Base Q&A")
    print("Powered by: module 05 skills (embeddings + GPT)")
    print("=" * 60)

    print("\nLoading embedding model + index...")
    try:
        emb_model, char_to_idx_emb, texts, sources, vectors = load_everything(SAVED_DIR)
        print(f"  Index: {len(texts)} chunks loaded")
    except SystemExit:
        return None

    print("Loading GPT generator...")
    gpt_model, char_to_idx_gpt, idx_to_char_gpt = load_gpt_model(SAVED_DIR)
    if gpt_model is None:
        print("  WARNING: GPT model not found. Retrieval will work but no generation.")
        print("  Run generator.py to enable answer generation.")
    else:
        n_params = sum(p.numel() for p in gpt_model.parameters())
        print(f"  GPT model loaded ({n_params:,} parameters)")

    print("\nReady! Type your question below.")
    print("Commands: 'search <query>' | 'quit'")
    print("-" * 60)

    return {
        "emb_model":       emb_model,
        "char_to_idx_emb": char_to_idx_emb,
        "texts":           texts,
        "sources":         sources,
        "vectors":         vectors,
        "gpt_model":       gpt_model,
        "char_to_idx_gpt": char_to_idx_gpt,
        "idx_to_char_gpt": idx_to_char_gpt,
    }


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_retrieved_chunks(results):
    """Pretty-print the retrieved chunks with their scores."""
    if not results:
        print("\n  No relevant chunks found.")
        print("  Tip: try different keywords or lower MIN_SCORE in retriever.py")
        return

    print(f"\n  Found {len(results)} relevant chunk(s):\n")
    for i, r in enumerate(results):
        print(f"  [{i+1}] Score: {r['score']:.3f}  |  Source: {r['source']}")
        print(f"  {'_' * 56}")
        # Print first 200 chars of chunk
        preview = r["text"][:200].replace("\n", " ")
        print(f"  {preview}")
        if len(r["text"]) > 200:
            print(f"  ... [{len(r['text']) - 200} more chars]")
        print()


def print_answer(answer, results):
    """Print the generated answer and its sources."""
    print("\n  ANSWER:")
    print("  " + "-" * 56)
    # Wrap answer at 56 chars for readability
    words   = answer.split()
    line    = "  "
    for word in words:
        if len(line) + len(word) + 1 > 58:
            print(line)
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
    print("  " + "-" * 56)

    if results:
        print("\n  Sources:")
        seen = set()
        for r in results:
            if r["source"] not in seen:
                print(f"    - {r['source']}  (similarity: {r['score']:.3f})")
                seen.add(r["source"])


# =============================================================================
# MAIN APP LOOP
# =============================================================================

def run_app(state):
    """Interactive Q&A loop."""

    while True:
        print()
        try:
            user_input = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # --- Search-only mode ---
        if user_input.lower().startswith("search "):
            query = user_input[7:].strip()
            print(f"\nSearching for: '{query}'...")
            results = retrieve(
                query,
                state["emb_model"],
                state["char_to_idx_emb"],
                state["texts"],
                state["sources"],
                state["vectors"],
                top_k=TOP_K,
            )
            print_retrieved_chunks(results)
            continue

        # --- Full Q&A mode ---
        question = user_input
        print(f"\nSearching notes for: '{question}'...")

        # Step 1: Retrieve relevant chunks
        results = retrieve(
            question,
            state["emb_model"],
            state["char_to_idx_emb"],
            state["texts"],
            state["sources"],
            state["vectors"],
            top_k=TOP_K,
        )

        if not results:
            print("\n  No relevant notes found for this question.")
            print("  The model could not find a close match in your notes.")
            continue

        print(f"  Retrieved {len(results)} relevant chunk(s)")

        # Step 2: Generate answer from retrieved chunks
        if state["gpt_model"] is not None:
            print("  Generating answer...")
            answer = generate_answer(
                context_chunks  = results,
                question        = question,
                model           = state["gpt_model"],
                char_to_idx     = state["char_to_idx_gpt"],
                idx_to_char     = state["idx_to_char_gpt"],
                max_new_tokens  = 150,
                temperature     = 0.7,
            )
            print_answer(answer, results)
        else:
            # No GPT: just show the retrieved chunks
            print("  (GPT not loaded -- showing retrieved chunks only)")
            print_retrieved_chunks(results)

        # Always show sources at the end
        print("\n  Type 'search <query>' to see raw retrieved chunks.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    state = startup()
    if state:
        run_app(state)

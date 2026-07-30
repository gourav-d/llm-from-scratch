"""
chat.py -- Phase 3: Interactive console chat loop.

WHAT THIS FILE DOES:
---------------------
Runs a REPL (Read-Eval-Print Loop) in your terminal.
You type a question, it finds relevant code, calls the LLM, prints the answer.

REPL = Read-Eval-Print Loop:
  Read    --> input() reads your question
  Eval    --> retriever.answer_question() finds the answer
  Print   --> print() shows the answer
  Loop    --> while True keeps running until you type "exit"

C# analogy:
  This is like Console.ReadLine() in a while(true) loop --
  the standard pattern for console applications.

Run with:
    python chat.py
"""

import sys
import os

import config
from retriever import answer_question
from utils.embedder import check_ollama_running, check_model_available


# ANSI color codes for terminal output.
# These are special escape sequences that change text color in most terminals.
# In C#: Console.ForegroundColor = ConsoleColor.Cyan
# On Windows, these work in Windows Terminal and VS Code terminal.
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
BOLD    = "\033[1m"
RESET   = "\033[0m"     # reset all formatting back to default


def print_banner():
    """Print the welcome banner when the app starts."""
    print()
    print(BOLD + CYAN + "=" * 60 + RESET)
    print(BOLD + CYAN + "  Chat with Codebase  (Offline RAG)" + RESET)
    print(BOLD + CYAN + "=" * 60 + RESET)
    print(f"  Repo:   {config.REPO_PATH}")
    print(f"  Model:  {config.LLM_MODEL}")
    print(f"  DB:     {config.CHROMA_DB_PATH}")
    print()
    print("  Type your question and press Enter.")
    print("  Commands: 'exit' or 'quit' to stop, 'help' for tips.")
    print(CYAN + "=" * 60 + RESET)
    print()


def print_help():
    """Print usage tips."""
    print()
    print(YELLOW + "TIPS:" + RESET)
    print("  - Ask about specific functions: 'What does calculate_tax() do?'")
    print("  - Ask about files: 'What is in the utils/ folder?'")
    print("  - Ask about concepts: 'How does authentication work here?'")
    print("  - Ask about patterns: 'Where are database queries made?'")
    print()
    print(YELLOW + "COMMANDS:" + RESET)
    print("  exit / quit  -- stop the chat")
    print("  help         -- show this help")
    print("  sources      -- show sources from last answer")
    print("  clear        -- clear the screen")
    print()


def preflight_checks() -> bool:
    """
    Verify Ollama is running and models are available before starting.
    Returns True if all checks pass.
    """
    print("Checking setup...")

    if not check_ollama_running():
        print(RED + "ERROR: Ollama is not running." + RESET)
        print("  Start Ollama, then run this again.")
        return False

    if not check_model_available(config.LLM_MODEL):
        print(RED + f"ERROR: LLM model '{config.LLM_MODEL}' not found." + RESET)
        print(f"  Run: ollama pull {config.LLM_MODEL}")
        return False

    if not check_model_available(config.EMBEDDING_MODEL):
        print(RED + f"ERROR: Embedding model '{config.EMBEDDING_MODEL}' not found." + RESET)
        print(f"  Run: ollama pull {config.EMBEDDING_MODEL}")
        return False

    # Check ChromaDB has data
    try:
        import chromadb
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        existing = [c.name for c in client.list_collections()]
        if config.COLLECTION_NAME not in existing:
            print(RED + "ERROR: No indexed data found." + RESET)
            print("  Run 'python indexer.py' first.")
            return False
        collection = client.get_collection(config.COLLECTION_NAME)
        count = collection.count()
        print(GREEN + f"  [OK] ChromaDB: {count} chunks indexed" + RESET)
    except Exception as e:
        print(RED + f"ERROR: ChromaDB problem: {e}" + RESET)
        return False

    print(GREEN + "  [OK] Ollama running" + RESET)
    print(GREEN + f"  [OK] Models ready: {config.LLM_MODEL}, {config.EMBEDDING_MODEL}" + RESET)
    return True


def format_sources(sources: list) -> str:
    """
    Format a list of source file paths for display.
    Strips the REPO_PATH prefix for cleaner output.
    """
    if not sources:
        return "  (no sources)"

    lines = []
    for src in sources:
        # Make path relative to repo root for cleaner display
        try:
            relative = src.replace(config.REPO_PATH, "").lstrip("/\\")
        except Exception:
            relative = src
        lines.append(f"  - {relative}")

    return "\n".join(lines)


def run_chat():
    """
    The main chat REPL loop.

    Keeps asking for questions until the user types 'exit' or presses Ctrl+C.
    """
    print_banner()

    # Run pre-flight checks
    if not preflight_checks():
        print()
        print(RED + "Setup incomplete. Fix the errors above and try again." + RESET)
        sys.exit(1)

    print()
    print(GREEN + "Ready! Ask your first question." + RESET)
    print()

    # Keep track of the last result (so user can type 'sources' to see them)
    last_sources = []

    # The REPL loop
    # In C#: while (true) { string input = Console.ReadLine(); ... }
    while True:
        try:
            # Read: get input from the user
            # The "You: " prefix shows clearly where to type
            user_input = input(BOLD + "You: " + RESET).strip()

        except (EOFError, KeyboardInterrupt):
            # Ctrl+C or Ctrl+D pressed -- exit gracefully
            print()
            print("\nGoodbye!")
            break

        # Skip empty input (user just pressed Enter)
        if not user_input:
            continue

        # Handle special commands
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "help":
            print_help()
            continue

        if user_input.lower() == "sources":
            print()
            print(YELLOW + "Sources from last answer:" + RESET)
            print(format_sources(last_sources))
            print()
            continue

        if user_input.lower() == "clear":
            # Clear the terminal screen
            # In C#: Console.Clear()
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            continue

        # Eval + Print: process the question and show the answer
        print()
        print(CYAN + "Searching codebase..." + RESET, flush=True)

        try:
            # Call the full RAG pipeline
            result = answer_question(user_input)

            # Store sources for the 'sources' command
            last_sources = result["sources"]

            # Print the answer
            print()
            print(BOLD + GREEN + "Assistant: " + RESET, end="")
            print(result["answer"])
            print()

            # Print source files used
            print(YELLOW + "Sources used:" + RESET)
            print(format_sources(result["sources"]))
            print()

        except RuntimeError as e:
            # Known errors (Ollama not running, collection empty, etc.)
            print(RED + f"Error: {e}" + RESET)
            print()

        except Exception as e:
            # Unexpected errors -- show the error but keep the loop running
            print(RED + f"Unexpected error: {e}" + RESET)
            print("Try a different question or restart.")
            print()


def main():
    """Entry point when running: python chat.py"""
    run_chat()


if __name__ == "__main__":
    main()

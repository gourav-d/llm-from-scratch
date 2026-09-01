"""
app.py -- CLI entry point for Chat with Web.

Commands:
  python app.py add <url1> [url2 url3 ...]   -- fetch + index URLs
  python app.py ask "your question here"      -- ask a question
  python app.py summarize                     -- summarize all content
  python app.py list                          -- show indexed URLs
  python app.py clear                         -- delete all indexed data
  python app.py chat                          -- interactive Q&A session

C# analogy: like a Console App with a switch statement on args[0].
"""

import sys
import argparse

import fetcher
import store
import rag


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_add(urls: list[str]):
    """Fetch URLs and add them to the knowledge base."""
    print(f"\n=== FETCHING {len(urls)} URL(s) ===")

    # Phase 1: Fetch and clean
    documents = fetcher.fetch_multiple(urls)

    if not documents:
        print("\nNo documents were successfully fetched. Check your URLs.")
        return

    # Phase 2: Chunk, embed, store
    print(f"\n=== INDEXING {len(documents)} DOCUMENT(s) ===")
    total_chunks = store.store_documents(documents)

    print(f"\nDone! Indexed {len(documents)} pages as {total_chunks} chunks.")
    print("You can now ask questions with:  python app.py ask \"your question\"")


def cmd_ask(question: str):
    """Answer a question using RAG."""
    if not rag.is_ollama_running():
        print("\nERROR: Ollama is not running.")
        print("Start it with:  ollama serve")
        print(f"Make sure model is pulled:  ollama pull {rag.config.LLM_MODEL}")
        return

    result = rag.answer(question)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "-" * 40)
    print("SOURCES:")
    for url in result["sources"]:
        print(f"  - {url}")
    print(f"\n({result['chunks_used']} relevant chunks used)")


def cmd_summarize():
    """Summarize all indexed content."""
    if not rag.is_ollama_running():
        print("\nERROR: Ollama is not running. Start with:  ollama serve")
        return

    result = rag.summarize()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(result["summary"])

    print("\n" + "-" * 40)
    print("SOURCES:")
    for url in result["sources"]:
        print(f"  - {url}")


def cmd_list():
    """List all indexed URLs."""
    urls = store.get_indexed_urls()

    if not urls:
        print("\nNo URLs indexed yet.")
        print("Add some with:  python app.py add <url1> [url2 ...]")
        return

    print(f"\nIndexed URLs ({len(urls)} total):")
    for url in urls:
        print(f"  - {url}")


def cmd_clear():
    """Clear all indexed data."""
    print("\nThis will delete ALL indexed data.")
    confirm = input("Are you sure? (yes/no): ").strip().lower()
    if confirm == "yes":
        store.clear_all()
    else:
        print("Cancelled.")


def cmd_chat():
    """Interactive Q&A session -- keeps asking until user types 'quit'."""
    if not rag.is_ollama_running():
        print("\nERROR: Ollama is not running. Start with:  ollama serve")
        return

    urls = store.get_indexed_urls()
    if not urls:
        print("\nNo content indexed. Add URLs first:  python app.py add <url>")
        return

    print("\n" + "=" * 60)
    print("CHAT MODE -- Ask anything about your indexed web content")
    print("Type 'quit' or 'exit' to stop, 'sources' to list indexed URLs")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if question.lower() == "sources":
            cmd_list()
            continue

        result = rag.answer(question)

        print(f"\nAssistant: {result['answer']}")
        print(f"\n[Sources: {', '.join(result['sources'][:2])}{'...' if len(result['sources']) > 2 else ''}]")


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chat with Web -- Ask questions about any web page",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py add https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)
  python app.py add https://site1.com https://site2.com https://site3.com
  python app.py ask "What is the attention mechanism?"
  python app.py ask "Summarize the key concepts"
  python app.py summarize
  python app.py list
  python app.py chat
  python app.py clear
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'add' command
    add_parser = subparsers.add_parser("add", help="Fetch and index URLs")
    add_parser.add_argument("urls", nargs="+", help="One or more URLs to index")

    # 'ask' command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Your question in quotes")

    # 'summarize' command
    subparsers.add_parser("summarize", help="Summarize all indexed content")

    # 'list' command
    subparsers.add_parser("list", help="List indexed URLs")

    # 'clear' command
    subparsers.add_parser("clear", help="Delete all indexed data")

    # 'chat' command
    subparsers.add_parser("chat", help="Interactive Q&A session")

    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args.urls)
    elif args.command == "ask":
        cmd_ask(args.question)
    elif args.command == "summarize":
        cmd_summarize()
    elif args.command == "list":
        cmd_list()
    elif args.command == "clear":
        cmd_clear()
    elif args.command == "chat":
        cmd_chat()


if __name__ == "__main__":
    main()

"""
app.py -- Phase 4: Streamlit web UI for Chat with Codebase.

WHAT THIS FILE DOES:
---------------------
Creates a browser-based chat interface using Streamlit.
Looks like a real chat app: message bubbles, conversation history, sidebar settings.

WHY STREAMLIT?
--------------
Streamlit turns Python scripts into web apps with almost no HTML/CSS/JS.
A few Python function calls give you buttons, text boxes, sliders, and layouts.

C# analogy:
  Like ASP.NET Razor Pages but:
  - No routing needed
  - No controllers
  - UI re-renders automatically when Python variables change
  - st.write("Hello") = <p>Hello</p>
  - st.button("Click") = <button>Click</button>

Run with:
    streamlit run app.py
    # Opens http://localhost:8501 in your browser

HOW STREAMLIT WORKS:
--------------------
Streamlit re-runs this ENTIRE script from top to bottom every time:
  - The user clicks a button
  - The user types in a text box
  - Any widget changes

State that needs to survive re-runs must be stored in st.session_state.
Think of st.session_state like a static variable in C# -- it persists
across calls but lives only in the current browser session.
"""

import streamlit as st          # the web UI framework
import sys
import os

import config
from retriever import answer_question
from utils.embedder import check_ollama_running, check_model_available


def setup_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Chat with Codebase",   # browser tab title
        page_icon="[chat]",                # browser tab icon
        layout="wide",                     # use full page width
        initial_sidebar_state="expanded",  # sidebar open by default
    )


def init_session_state():
    """
    Initialize session state variables.

    st.session_state is a dict-like object that persists values
    across script re-runs within the same browser session.

    Without this, conversation history would be wiped every time
    the user types a message.

    C# analogy:
        // ASP.NET Session
        if (Session["messages"] == null) Session["messages"] = new List<Message>();
    """
    # messages: the conversation history (list of dicts with role + content)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # last_sources: sources from the most recent answer (for the sidebar)
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

    # processing: True while waiting for the LLM (prevents double-submission)
    if "processing" not in st.session_state:
        st.session_state.processing = False


def render_sidebar():
    """
    Render the left sidebar with settings and status info.

    st.sidebar.xxx puts things in the collapsible left panel.
    """
    with st.sidebar:
        st.title("Settings")

        # Show current config
        st.subheader("Configuration")
        st.code(f"""
Repo:   {os.path.basename(config.REPO_PATH)}
LLM:    {config.LLM_MODEL}
Embed:  {config.EMBEDDING_MODEL}
Top-K:  {config.TOP_K_RESULTS}
        """)

        # Status checks
        st.subheader("System Status")

        if check_ollama_running():
            st.success("Ollama: Running")
        else:
            st.error("Ollama: Not running")
            st.info("Start Ollama to use this app.")

        # Show ChromaDB status
        try:
            import chromadb
            client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
            existing = [c.name for c in client.list_collections()]
            if config.COLLECTION_NAME in existing:
                collection = client.get_collection(config.COLLECTION_NAME)
                count = collection.count()
                st.success(f"ChromaDB: {count} chunks")
            else:
                st.warning("ChromaDB: Not indexed")
                st.info("Run: python indexer.py")
        except Exception:
            st.error("ChromaDB: Error")

        # Sources from last answer
        if st.session_state.last_sources:
            st.subheader("Last Answer Sources")
            for src in st.session_state.last_sources:
                # Show relative path
                try:
                    relative = src.replace(config.REPO_PATH, "").lstrip("/\\")
                except Exception:
                    relative = src
                st.caption(f"- {relative}")

        # Clear conversation button
        st.subheader("Actions")
        if st.button("Clear Conversation", use_container_width=True):
            # Reset messages (Streamlit re-runs script after button click)
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()     # force a re-run to reflect the cleared state


def render_chat_history():
    """
    Render all previous messages in the conversation.

    st.chat_message() creates a message bubble with an avatar.
    "user" = right side, "assistant" = left side (standard chat UI convention).

    C# analogy: foreach over a List<Message> and rendering HTML.
    """
    for message in st.session_state.messages:
        # role is either "user" or "assistant"
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_question(question: str):
    """
    Handle a new user question: retrieve answer and update conversation.

    Parameters:
        question : the user's question string
    """
    # Add user message to conversation history
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    # Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Show a spinner while waiting for the LLM
    # C# analogy: an async operation with a loading indicator
    with st.chat_message("assistant"):
        with st.spinner("Searching codebase and generating answer..."):
            try:
                # Call the full RAG pipeline
                result = answer_question(question)

                answer = result["answer"]
                sources = result["sources"]

                # Display the answer
                st.markdown(answer)

                # Show source files under the answer (collapsible)
                if sources:
                    with st.expander(f"Sources ({len(sources)} files)"):
                        for src in sources:
                            try:
                                relative = src.replace(config.REPO_PATH, "").lstrip("/\\")
                            except Exception:
                                relative = src
                            st.code(relative)

                # Save to conversation history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })

                # Update the sidebar sources
                st.session_state.last_sources = sources

            except RuntimeError as e:
                # Known errors (Ollama not running, not indexed, etc.)
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })


def main():
    """
    Main entry point for the Streamlit app.
    Streamlit calls this entire function every time the UI updates.
    """
    setup_page()
    init_session_state()

    # Page header
    st.title("Chat with Codebase")
    st.caption(
        "Ask questions about your codebase in plain English. "
        "Powered by local Ollama LLM + ChromaDB. No internet required."
    )

    # Render the sidebar
    render_sidebar()

    # Render existing conversation history
    render_chat_history()

    # Chat input box at the bottom
    # st.chat_input() returns the text when the user submits, None otherwise.
    # C# analogy: an event handler that fires when the user presses Enter.
    if question := st.chat_input("Ask something about your codebase..."):
        process_question(question)


if __name__ == "__main__":
    # Streamlit runs this via 'streamlit run app.py', not 'python app.py'.
    # But we keep this guard for documentation clarity.
    main()

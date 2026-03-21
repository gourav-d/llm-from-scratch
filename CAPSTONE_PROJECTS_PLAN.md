# CAPSTONE PROJECTS PLAN

**Complete Implementation Plan for Two Advanced LLM Projects**

**Created**: March 20, 2026
**Status**: Planning Phase
**Total Projects**: 2 major capstone projects
**Implementation Timeline**: After Modules 8-12 completion

---

## 📋 Table of Contents

1. [Project 1: AI Chat Assistant with Web Search](#project-1-ai-chat-assistant-with-web-search)
2. [Project 2: Stock Analysis LLM for Indian Markets](#project-2-stock-analysis-llm-for-indian-markets)
3. [Timeline Integration](#timeline-integration)
4. [Technology Stack Comparison](#technology-stack-comparison)
5. [Success Metrics](#success-metrics)

---

# PROJECT 1: AI Chat Assistant with Web Search

## 🎯 Project Overview

**Type**: AI Chatbot with Real-time Information Retrieval
**Complexity**: High
**Similar To**: ChatGPT, Claude, Perplexity AI
**Estimated Time**: 6-8 weeks (full-time equivalent)

### What It Does

A conversational AI that:
- Answers questions using LLM knowledge
- Detects when it lacks information
- Searches Google for real-time data
- Extracts and processes top 5 search results
- Synthesizes information into coherent answers
- Cites sources for transparency

### Example Flow

```
User: "What are the latest developments in Python 3.13?"

AI Internal Process:
1. Check: "Is this in my training data?" → NO (too recent)
2. Decision: "I need to search for this"
3. Google Search: "Python 3.13 latest developments"
4. Scrape top 5 links
5. Extract relevant information
6. Synthesize answer with citations

AI Response:
"Python 3.13 was released in October 2024 with several key improvements:
- Free-threading support (PEP 703)
- JIT compiler improvements (20-30% faster)
- Better error messages
...
Sources: [1] python.org [2] realpython.com [3] phoronix.com"
```

---

## 📚 Prerequisites - Which Modules Must Be Complete

### Critical Modules (MUST Complete)

| Module | Why Needed | Priority |
|--------|-----------|----------|
| **Module 8: Prompt Engineering** | Design effective prompts, handle conversation context | ⭐⭐⭐ CRITICAL |
| **Module 9: Vector Databases** | Store conversation history, semantic search | ⭐⭐⭐ CRITICAL |
| **Module 10: RAG** | Retrieve and use web-scraped content | ⭐⭐⭐ CRITICAL |
| **Module 11: LangChain** | Orchestrate agents, tools, and chains | ⭐⭐⭐ CRITICAL |
| **Module 13: LLM APIs** | Call OpenAI/Anthropic/local LLMs | ⭐⭐⭐ CRITICAL |

### Recommended Modules

| Module | Why Helpful | Priority |
|--------|-------------|----------|
| **Module 7: Reasoning Models** | Better reasoning and chain-of-thought | ⭐⭐ HIGH |
| **Module 14: Security** | Prevent prompt injection, secure API calls | ⭐⭐ HIGH |
| **Module 19: AI Agents** | Advanced agent patterns | ⭐ MEDIUM |

### Timeline to Start

**Earliest Start Date**: After completing Module 11 (LangChain)
**Estimated Readiness**: Month 7 of your learning journey
**Current Status**: Need to complete Modules 8-11 first

---

## 🛠️ Technology Stack

### Core LLM

```python
# Option 1: OpenAI (Recommended for beginners)
openai==1.12.0
gpt-4-turbo or gpt-3.5-turbo

# Option 2: Anthropic Claude (Best for reasoning)
anthropic==0.21.0
claude-3-sonnet or claude-3-opus

# Option 3: Local LLM (Free, but slower)
ollama
llama3:70b or mistral:7b
```

**Recommendation**: Start with OpenAI GPT-3.5-turbo (cheap, fast)

### Web Search & Scraping

```python
# Search API
google-search-results==2.4.2  # SerpAPI - 100 free searches/month
duckduckgo-search==5.0.0      # Free alternative

# Web Scraping
beautifulsoup4==4.12.3
playwright==1.41.0            # For dynamic content
trafilatura==1.6.4            # Article extraction
newspaper3k==0.2.8            # News article parsing

# URL Processing
requests==2.31.0
httpx==0.26.0                 # Async requests
```

### RAG & Vector Database

```python
# Vector Database
chromadb==0.4.22              # Free, local, easy to start
# OR
pinecone-client==3.1.0        # Cloud, scalable (free tier)

# Embeddings
sentence-transformers==2.3.1   # Local embeddings
openai==1.12.0                # OpenAI embeddings (better quality)
```

### Orchestration

```python
# LangChain Ecosystem
langchain==0.1.9
langchain-community==0.0.24
langchain-openai==0.0.8
langgraph==0.0.26             # For agent workflows

# OR simpler alternative
llama-index==0.10.12
```

### Web Interface

```python
# Backend
fastapi==0.109.2
uvicorn==0.27.1

# Frontend Options
streamlit==1.31.0             # Easiest (recommended)
# OR
gradio==4.19.1                # Also very easy
# OR
chainlit==1.0.0               # Best for chat
```

### Utilities

```python
# Environment & Config
python-dotenv==1.0.1
pydantic==2.6.1

# Caching
redis==5.0.1                  # Cache search results

# Monitoring
langsmith==0.0.87             # Track LLM calls
```

---

## 📊 Data Requirements

### Training Data

**Good News**: ✅ **NO TRAINING REQUIRED!**

This project uses:
- Pre-trained LLM (GPT/Claude/Llama)
- Real-time web scraping
- RAG (retrieval) instead of training

### Runtime Data Sources

| Data Type | Source | Cost | Setup Time |
|-----------|--------|------|------------|
| **Web Search** | SerpAPI | $0 (100 free/month) | 5 min |
| **Web Search** | DuckDuckGo | $0 (unlimited) | 2 min |
| **Web Content** | Public websites | $0 | 0 min |
| **Conversation History** | ChromaDB (local) | $0 | 10 min |
| **LLM** | OpenAI GPT-3.5 | ~$0.002/query | 5 min |

**Total Setup Cost**: $0
**Monthly Running Cost**: ~$5-20 (depending on usage)

### Data Storage

```
Your Computer (Free):
├── ChromaDB (vector database)
│   ├── Conversation history: ~1MB per 1000 messages
│   └── Web content cache: ~100MB per 1000 pages
├── Redis Cache (optional)
│   └── Search results: ~50MB
└── SQLite (metadata)
    └── User sessions, logs: ~10MB
```

**Storage Required**: Less than 500MB for typical usage

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    WEB INTERFACE                             │
│              (Streamlit / Gradio / Chainlit)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                        │
│                   (LangChain / LangGraph)                    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Router     │  │  Planner     │  │   Memory     │      │
│  │  (decides    │  │  (creates    │  │  (stores     │      │
│  │  what to do) │  │  execution   │  │  context)    │      │
│  └──────────────┘  │  plan)       │  └──────────────┘      │
│                    └──────────────┘                         │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  LLM Agent   │ │ Search Agent │ │  RAG Agent   │
│              │ │              │ │              │
│ - Answers    │ │ - Detects    │ │ - Retrieves  │
│   questions  │ │   need for   │ │   web        │
│ - Synthesizes│ │   search     │ │   content    │
│   info       │ │ - Searches   │ │ - Extracts   │
│              │ │   web        │ │   relevant   │
│              │ │ - Filters    │ │   info       │
│              │ │   results    │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────────────────────────────────────────┐
│              EXTERNAL SERVICES                    │
│                                                   │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐     │
│  │ OpenAI  │  │ SerpAPI  │  │  ChromaDB   │     │
│  │   or    │  │    or    │  │  (Vector    │     │
│  │ Claude  │  │DuckDuckGo│  │  Database)  │     │
│  └─────────┘  └──────────┘  └─────────────┘     │
└──────────────────────────────────────────────────┘
```

### Decision Flow

```python
"""
How the system decides what to do:
"""

def handle_user_query(query: str):
    # Step 1: Classify intent
    intent = classify_intent(query)

    # Step 2: Route to appropriate handler
    if intent == "general_knowledge":
        # Answer directly from LLM
        return llm_direct_answer(query)

    elif intent == "recent_events":
        # Need web search
        search_results = search_web(query, num_results=5)
        web_content = scrape_urls(search_results)
        context = extract_relevant_info(web_content)
        return llm_answer_with_context(query, context)

    elif intent == "conversational":
        # Use conversation history
        history = get_conversation_history()
        return llm_conversational(query, history)

    elif intent == "uncertain":
        # Try LLM first, fall back to search
        llm_answer = llm_direct_answer(query)
        confidence = assess_confidence(llm_answer)

        if confidence < 0.7:
            # Low confidence, search web
            return search_and_synthesize(query)
        else:
            return llm_answer
```

### Agent Workflow (LangGraph)

```python
"""
State machine for query handling:
"""

from langgraph.graph import Graph

# Define states
class QueryState:
    query: str
    intent: str
    search_results: List[str]
    web_content: str
    answer: str
    confidence: float

# Build workflow
workflow = Graph()

workflow.add_node("classify", classify_intent)
workflow.add_node("llm_answer", generate_llm_answer)
workflow.add_node("search_web", search_and_scrape)
workflow.add_node("synthesize", synthesize_with_sources)

# Define edges (flow)
workflow.add_edge("classify", "llm_answer")
workflow.add_conditional_edge(
    "llm_answer",
    lambda state: "search_web" if state.confidence < 0.7 else "end",
)
workflow.add_edge("search_web", "synthesize")

# Compile
app = workflow.compile()
```

---

## 📝 Implementation Roadmap

### Phase 1: Basic Chat (Week 1)
**Goal**: Chat with LLM, no web search yet

**Tasks**:
- ✅ Set up OpenAI/Claude API
- ✅ Create simple Streamlit UI
- ✅ Implement basic chat loop
- ✅ Add conversation history

**Deliverable**: Working chatbot like ChatGPT (without search)

**Code**:
```python
# app.py - 50 lines
import streamlit as st
from openai import OpenAI

client = OpenAI()
st.title("My AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )

    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
```

### Phase 2: Web Search Integration (Week 2)
**Goal**: Add ability to search web when needed

**Tasks**:
- ✅ Integrate SerpAPI or DuckDuckGo
- ✅ Implement search trigger logic
- ✅ Add web scraping (BeautifulSoup)
- ✅ Extract clean text from HTML

**Deliverable**: Assistant that searches when needed

**Code**:
```python
# web_search.py
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

def search_web(query: str, num_results: int = 5):
    """Search DuckDuckGo and return top results"""
    ddgs = DDGS()
    results = list(ddgs.text(query, max_results=num_results))
    return results

def scrape_url(url: str) -> str:
    """Extract main text content from URL"""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        return '\n'.join(line for line in lines if line)
    except:
        return ""
```

### Phase 3: RAG Implementation (Week 3-4)
**Goal**: Use vector database for context retrieval

**Tasks**:
- ✅ Set up ChromaDB
- ✅ Create embeddings for web content
- ✅ Implement semantic search
- ✅ Cache search results

**Deliverable**: Efficient retrieval system

**Code**:
```python
# rag_system.py
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

collection = client.create_collection(
    name="web_content",
    embedding_function=openai_ef
)

def add_to_vector_db(url: str, content: str, metadata: dict):
    """Store web content with embeddings"""
    collection.add(
        documents=[content],
        metadatas=[metadata],
        ids=[url]
    )

def retrieve_relevant(query: str, n_results: int = 5):
    """Retrieve most relevant content"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results
```

### Phase 4: LangChain Orchestration (Week 5)
**Goal**: Use LangChain for better agent management

**Tasks**:
- ✅ Create LangChain agents
- ✅ Implement tool calling
- ✅ Add routing logic
- ✅ Improve error handling

**Deliverable**: Production-ready agent system

**Code**:
```python
# agents.py
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-turbo-preview")

# Define tools
search_tool = Tool(
    name="web_search",
    func=search_and_scrape,
    description="Search the web for recent information"
)

# Create agent
agent = create_openai_tools_agent(
    llm=llm,
    tools=[search_tool],
    prompt=custom_prompt
)

executor = AgentExecutor(agent=agent, tools=[search_tool])
```

### Phase 5: Polish & Deploy (Week 6)
**Goal**: Production-ready application

**Tasks**:
- ✅ Add source citations
- ✅ Improve UI/UX
- ✅ Add rate limiting
- ✅ Error handling
- ✅ Deploy to cloud (optional)

**Deliverable**: Fully functional AI assistant

---

## 💰 Cost Breakdown

### Development Costs

| Item | Cost | Notes |
|------|------|-------|
| APIs (testing) | $10-20 | OpenAI credits |
| No training | $0 | Using pre-trained models |
| No hardware | $0 | Use your laptop |
| **Total** | **$10-20** | One-time |

### Running Costs (Per Month)

| Item | Free Tier | Paid Tier | Notes |
|------|-----------|-----------|-------|
| OpenAI API | - | $5-20 | ~$0.002 per query |
| SerpAPI | 100 searches | $50/5000 | Or use free DuckDuckGo |
| ChromaDB | ∞ (local) | $0 | Self-hosted |
| Hosting | - | $5-10 | If deploying (Railway, Render) |
| **Total** | **$0** | **$10-30** | Depends on usage |

---

## 🎯 Success Criteria

### Functional Requirements

- ✅ Answer general knowledge questions
- ✅ Detect when search is needed
- ✅ Search and scrape top 5 results
- ✅ Synthesize coherent answers
- ✅ Cite sources
- ✅ Maintain conversation context
- ✅ Handle errors gracefully

### Performance Requirements

| Metric | Target | Excellent |
|--------|--------|-----------|
| Response time (no search) | < 2s | < 1s |
| Response time (with search) | < 8s | < 5s |
| Answer accuracy | > 85% | > 95% |
| Source relevance | > 80% | > 90% |
| Uptime | > 95% | > 99% |

### User Experience

- Clean, intuitive interface
- Clear indication when searching
- Visible source citations
- Conversation history
- Mobile-friendly (bonus)

---

# PROJECT 2: Stock Analysis LLM for Indian Markets

## 🎯 Project Overview

**Type**: Domain-Specific Financial Analysis AI
**Complexity**: Very High
**Similar To**: Bloomberg Terminal AI, FinChat.io
**Estimated Time**: 10-12 weeks (full-time equivalent)

### What It Does

A specialized LLM that:
- Analyzes Indian stocks (NSE, BSE)
- Evaluates mutual funds
- Tracks company fundamentals
- Monitors latest news and sentiment
- Provides buy/sell/hold recommendations
- Explains reasoning behind recommendations
- Updates with real-time market data

### Example Interaction

```
User: "Should I invest in TCS or Infosys right now?"

AI Analysis:
1. Fetches latest financials for both companies
2. Analyzes P/E ratio, profit margins, growth
3. Checks recent news sentiment
4. Compares with sector peers
5. Considers market conditions

AI Response:
"Based on current data (as of Mar 20, 2026):

TCS Analysis:
- P/E Ratio: 28.5 (sector avg: 26)
- Revenue Growth: 9.2% YoY
- Profit Margin: 24.1%
- Recent News: Positive (won $1.2B deal with BFSI client)
- Dividend Yield: 2.1%
Rating: BUY (Score: 7.8/10)

Infosys Analysis:
- P/E Ratio: 26.3 (below sector avg)
- Revenue Growth: 8.7% YoY
- Profit Margin: 22.8%
- Recent News: Mixed (CEO transition concerns)
- Dividend Yield: 2.3%
Rating: HOLD (Score: 7.2/10)

Recommendation: TCS shows stronger momentum and fundamentals currently,
but both are solid long-term holds. Consider 60% TCS, 40% Infosys for
diversification.

Risk Factors: IT sector facing margin pressure, US recession concerns.
```

---

## 📚 Prerequisites - Which Modules Must Be Complete

### Critical Modules (MUST Complete)

| Module | Why Needed | Priority |
|--------|-----------|----------|
| **Module 8: Prompt Engineering** | Financial analysis prompts | ⭐⭐⭐ CRITICAL |
| **Module 9: Vector Databases** | Store financial reports, news | ⭐⭐⭐ CRITICAL |
| **Module 10: RAG** | Retrieve company filings, news | ⭐⭐⭐ CRITICAL |
| **Module 11: LangChain** | Orchestrate analysis workflows | ⭐⭐⭐ CRITICAL |
| **Module 12: Fine-tuning** | Fine-tune on financial data | ⭐⭐⭐ CRITICAL |
| **Module 13: LLM APIs** | Use GPT-4/Claude for analysis | ⭐⭐⭐ CRITICAL |
| **Module 14: Security** | Protect financial data | ⭐⭐⭐ CRITICAL |

### Recommended Modules

| Module | Why Helpful | Priority |
|--------|-------------|----------|
| **Module 7: Reasoning** | Better financial reasoning | ⭐⭐ HIGH |
| **Module 19: AI Agents** | Autonomous monitoring | ⭐⭐ HIGH |
| **Module 23: Optimization** | Fast inference | ⭐ MEDIUM |

### Timeline to Start

**Earliest Start Date**: After completing Module 12 (Fine-tuning)
**Estimated Readiness**: Month 9-10 of your learning journey
**Current Status**: Need to complete Modules 8-12 first

---

## 🛠️ Technology Stack

### Core LLM

```python
# Base Model (for fine-tuning)
transformers==4.38.1
torch==2.2.0

# Options:
# 1. Fine-tune Llama 3.1 70B (best quality)
# 2. Fine-tune Mistral 7B (faster, cheaper)
# 3. Use GPT-4 Turbo with extensive prompting (easiest)

# Recommendation: Start with GPT-4 prompting, then fine-tune Mistral
```

### Financial Data APIs

```python
# Indian Market Data
yfinance==0.2.36              # Free, NSE/BSE data
nsetools==1.0.11              # NSE specific
bsedata==0.5.0                # BSE specific

# News & Sentiment
newsapi-python==0.2.7         # Global news (free tier: 100 requests/day)
feedparser==6.0.11            # RSS feeds (Moneycontrol, ET)

# Fundamental Data
selenium==4.18.1              # Scrape Screener.in, Tickertape
beautifulsoup4==4.12.3        # Parse financial statements

# Alternative (Paid but Better):
# financial-datasets==1.1.0   # Clean Indian stock data
# alpha_vantage==2.3.1        # US markets (limited India support)
```

### Fine-tuning & Training

```python
# Fine-tuning
peft==0.9.0                   # LoRA, QLoRA
bitsandbytes==0.42.0          # 4-bit quantization
trl==0.7.11                   # Trainer for LLMs

# Dataset Creation
datasets==2.17.1
pandas==2.2.0
```

### RAG & Vector Search

```python
# Vector Database
qdrant-client==1.7.3          # Better for production than ChromaDB
# OR
pinecone-client==3.1.0        # Managed service

# Embeddings
sentence-transformers==2.3.1   # Local
openai==1.12.0                # Better quality
```

### Financial Analysis

```python
# Technical Analysis
ta-lib==0.4.28                # Technical indicators
pandas-ta==0.3.14b            # Alternative, easier install

# Fundamental Analysis
fundamental-analysis==0.1.0    # Calculate ratios
```

### Backend & API

```python
# API Framework
fastapi==0.109.2
uvicorn==0.27.1
pydantic==2.6.1

# Database
sqlalchemy==2.0.27            # Store historical data
redis==5.0.1                  # Cache market data

# Scheduling
apscheduler==3.10.4           # Update data periodically
celery==5.3.6                 # Background tasks
```

### Frontend

```python
# Web Interface
streamlit==1.31.0             # Quick prototype
# OR
react + fastapi                # Production app

# Visualization
plotly==5.18.0                # Interactive charts
matplotlib==3.8.2
```

---

## 📊 Data Requirements

### Training Data (for Fine-tuning)

This is **CRITICAL** and requires significant effort.

#### Data Types Needed

| Data Type | Quantity | Source | Time to Collect |
|-----------|----------|--------|-----------------|
| **Financial Reports** | 10,000+ | BSE, NSE, Screener.in | 2-3 weeks |
| **Stock Analysis** | 5,000+ | Create manually + scrape | 3-4 weeks |
| **News Articles** | 50,000+ | Moneycontrol, ET, BS | 1 week |
| **Price Data** | 5 years | yfinance, NSE | 1 day |
| **Q&A Pairs** | 1,000+ | Create manually | 2-3 weeks |

#### Example Training Data Format

```json
{
  "instruction": "Analyze TCS stock based on these financials",
  "input": {
    "company": "TCS",
    "sector": "IT Services",
    "financials": {
      "revenue": "₹2.25 lakh crore",
      "net_profit": "₹42,000 crore",
      "pe_ratio": 28.5,
      "roe": 48.2,
      "debt_to_equity": 0.05
    },
    "recent_news": [
      "TCS wins $1.2B deal with UK banking client",
      "Hiring slowdown in Q4 2025"
    ]
  },
  "output": "TCS shows strong fundamentals with exceptional ROE of 48.2% and virtually zero debt (D/E: 0.05). The P/E ratio of 28.5 is slightly above sector average of 26, suggesting premium valuation is justified. Recent $1.2B deal win indicates strong demand pipeline. However, hiring slowdown suggests near-term margin pressure. Rating: BUY for long-term investors. Target: ₹4,200 (12-month)."
}
```

#### Data Collection Strategy

**Phase 1: Automated Scraping (Week 1-2)**
```python
# scraper.py
import yfinance as yf
from nsetools import Nse
import feedparser

# 1. Get all NSE stocks
nse = Nse()
stock_list = nse.get_stock_codes()  # ~2000 stocks

# 2. Download historical data
for stock in stock_list:
    ticker = f"{stock}.NS"
    data = yf.download(ticker, period="5y")
    financials = yf.Ticker(ticker).info
    # Save to database

# 3. Scrape news
sources = [
    "https://www.moneycontrol.com/rss/mf.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
]
for source in sources:
    feed = feedparser.parse(source)
    # Extract and save articles
```

**Phase 2: Manual Dataset Creation (Week 3-5)**
```python
# This is the HARD part - creating quality analysis examples
# You need to manually create ~1000 examples like:

examples = [
    {
        "stock": "TCS",
        "fundamentals": {...},
        "analysis": "Strong buy because...",
        "rating": 9.0
    },
    # Need many more...
]
```

**Phase 3: Augmentation (Week 6)**
```python
# Use GPT-4 to generate more examples
# But verify each one manually!
```

### Data Sources (Free)

| Source | Data Type | API | Limitations |
|--------|-----------|-----|-------------|
| **NSE India** | Real-time prices | No official API | Rate limited, scraping needed |
| **BSE India** | Real-time prices | SOAP API (complex) | Registration required |
| **Screener.in** | Fundamentals | No API | Must scrape |
| **Moneycontrol** | News, Analysis | No API | Must scrape |
| **YFinance** | Historical data | Python library | 15min delay |
| **Tickertape** | Screening | No API | Must scrape |
| **Economic Times** | News | RSS feeds | Free |

### Data Sources (Paid - Better Quality)

| Source | Cost | What You Get |
|--------|------|--------------|
| **TrueData** | ₹2,000/month | Real-time NSE/BSE data |
| **Upstox API** | Free (with trading account) | Real-time market data |
| **Zerodha Kite** | Free (with account) | Historical + real-time |
| **BSE Bhav Copy** | Free | EOD data (official) |

**Recommendation**: Start with free (yfinance + scraping), upgrade later

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                 │
│              (Web/Mobile Interface)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    API GATEWAY                               │
│                   (FastAPI + Auth)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Query        │ │ Analysis     │ │ Monitoring   │
│ Service      │ │ Engine       │ │ Service      │
│              │ │              │ │              │
│ - Parse      │ │ - Financial  │ │ - Track      │
│   query      │ │   analysis   │ │   stocks     │
│ - Route      │ │ - Generate   │ │ - Alert      │
│   request    │ │   report     │ │   on news    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    FINE-TUNED LLM                            │
│              (Mistral 7B or Llama 3 70B)                     │
│           Specialized in Financial Analysis                  │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   RAG        │ │  Data Lake   │ │  News        │
│   System     │ │              │ │  Aggregator  │
│              │ │              │ │              │
│ - Vector DB  │ │ - Stock      │ │ - RSS feeds  │
│   (Qdrant)   │ │   prices     │ │ - Sentiment  │
│ - Financial  │ │ - Financials │ │   analysis   │
│   reports    │ │ - Ratios     │ │ - Real-time  │
│ - Research   │ │              │ │   updates    │
└──────────────┘ └──────┬───────┘ └──────┬───────┘
                        │                │
                        ▼                ▼
               ┌─────────────────────────────┐
               │   EXTERNAL DATA SOURCES     │
               │                             │
               │  - NSE/BSE APIs             │
               │  - Screener.in              │
               │  - Moneycontrol             │
               │  - Economic Times           │
               │  - Company websites         │
               └─────────────────────────────┘
```

### Analysis Workflow

```python
"""
How stock analysis works:
"""

def analyze_stock(symbol: str, criteria: dict):
    """
    Comprehensive stock analysis pipeline
    """

    # Step 1: Fetch latest data
    data = {
        "current_price": get_current_price(symbol),
        "financials": get_latest_financials(symbol),
        "technicals": calculate_technical_indicators(symbol),
        "news": get_recent_news(symbol, days=30),
        "sentiment": analyze_news_sentiment(symbol),
        "peer_comparison": compare_with_sector(symbol),
        "analyst_ratings": get_analyst_consensus(symbol)
    }

    # Step 2: Retrieve relevant context (RAG)
    context = vector_db.search(
        query=f"Analysis framework for {symbol}",
        filters={"sector": data["sector"]}
    )

    # Step 3: Run through Fine-tuned LLM
    prompt = f"""
    You are an expert Indian stock market analyst.

    Analyze {symbol} based on:

    Fundamentals:
    - P/E Ratio: {data['financials']['pe_ratio']}
    - ROE: {data['financials']['roe']}%
    - Debt/Equity: {data['financials']['debt_to_equity']}
    - Revenue Growth: {data['financials']['revenue_growth']}%

    Technical Indicators:
    - RSI: {data['technicals']['rsi']}
    - MACD: {data['technicals']['macd']}
    - Moving Averages: {data['technicals']['ma']}

    Recent News Sentiment: {data['sentiment']['score']}/10

    Peer Comparison:
    {data['peer_comparison']}

    User Criteria:
    {criteria}

    Provide:
    1. BUY/SELL/HOLD rating with score (0-10)
    2. Detailed reasoning
    3. Risk factors
    4. Price target (12 months)
    """

    analysis = fine_tuned_llm(prompt, context)

    # Step 4: Validate & enhance
    validation = validate_analysis(analysis, data)
    enhanced = add_visualizations(analysis, data)

    return enhanced
```

---

## 📝 Implementation Roadmap

### Phase 1: Data Collection & Preparation (Week 1-6)

**This is the MOST IMPORTANT and TIME-CONSUMING phase!**

#### Week 1-2: Automated Data Collection
```python
# Tasks:
✅ Set up database schema (PostgreSQL)
✅ Write scrapers for:
   - NSE/BSE stock prices (yfinance)
   - Company fundamentals (Screener.in)
   - News (Moneycontrol, ET)
✅ Create ETL pipeline
✅ Store 5 years historical data for 500 stocks

# Deliverable: Database with 500 stocks, 5 years data
```

#### Week 3-5: Manual Dataset Creation
```python
# Tasks:
✅ Create 1,000 stock analysis examples manually
✅ Format as instruction-following dataset
✅ Get domain expert review (optional but recommended)
✅ Create evaluation test set (200 examples)

# Deliverable: 1,000 training examples, 200 test examples
```

#### Week 6: Data Cleaning & Augmentation
```python
# Tasks:
✅ Clean and validate all data
✅ Use GPT-4 to augment dataset (carefully!)
✅ Create train/validation/test splits
✅ Final dataset: 5,000 examples minimum

# Deliverable: Production-ready dataset
```

**Data Requirements Summary**:
- **Minimum**: 5,000 training examples
- **Good**: 10,000 training examples
- **Excellent**: 20,000+ training examples

**Time Investment**: 150-200 hours (this is a LOT of work!)

### Phase 2: Model Fine-tuning (Week 7-8)

#### Week 7: Setup & Initial Training
```python
# Tasks:
✅ Set up GPU environment (Google Colab Pro or local)
✅ Prepare Mistral 7B or Llama 3.1 8B
✅ Implement QLoRA fine-tuning script
✅ Start training (24-48 hours)

# Code:
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_4bit=True,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=2048,
    args=training_args
)

trainer.train()
```

#### Week 8: Evaluation & Iteration
```python
# Tasks:
✅ Evaluate on test set
✅ Calculate accuracy metrics
✅ Iterate on prompts and data
✅ Re-train if needed

# Evaluation metrics:
- Accuracy: % of correct BUY/SELL/HOLD
- MAE: Mean absolute error on price targets
- Sentiment alignment: Match expert analysis
- Reasoning quality: Human evaluation
```

**Hardware Requirements**:
- **Option 1**: Google Colab Pro ($10/month) - T4 GPU
- **Option 2**: Local RTX 3090/4090 (if you have one)
- **Option 3**: Lambda Labs ($0.50/hour) - A100 GPU

**Training Time**: 24-72 hours depending on dataset size

**Cost**: $20-100 total for fine-tuning

### Phase 3: RAG System (Week 9)

```python
# Tasks:
✅ Set up Qdrant vector database
✅ Create embeddings for financial reports
✅ Implement retrieval system
✅ Integrate with fine-tuned model

# Code:
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(":memory:")  # or cloud
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Index financial reports
def index_reports(reports: List[dict]):
    points = []
    for i, report in enumerate(reports):
        vector = encoder.encode(report['text'])
        points.append({
            "id": i,
            "vector": vector.tolist(),
            "payload": report
        })

    client.upsert(
        collection_name="financial_reports",
        points=points
    )

# Retrieve relevant context
def get_context(query: str, n=5):
    query_vector = encoder.encode(query)
    results = client.search(
        collection_name="financial_reports",
        query_vector=query_vector,
        limit=n
    )
    return results
```

### Phase 4: Real-time Data Integration (Week 10)

```python
# Tasks:
✅ Build data update pipeline
✅ Schedule periodic updates (every 15 min)
✅ News aggregation system
✅ Alert system for significant changes

# Code:
from apscheduler.schedulers.background import BackgroundScheduler
import yfinance as yf

scheduler = BackgroundScheduler()

def update_stock_data():
    """Run every 15 minutes"""
    stocks = get_watchlist()
    for stock in stocks:
        ticker = yf.Ticker(f"{stock}.NS")
        latest = ticker.info
        update_database(stock, latest)

def update_news():
    """Run every hour"""
    news = fetch_latest_news()
    sentiment = analyze_sentiment(news)
    store_news(news, sentiment)

scheduler.add_job(update_stock_data, 'interval', minutes=15)
scheduler.add_job(update_news, 'interval', hours=1)
scheduler.start()
```

### Phase 5: Application Development (Week 11)

```python
# Tasks:
✅ Build FastAPI backend
✅ Create REST API endpoints
✅ Implement authentication
✅ Add rate limiting

# API Structure:
from fastapi import FastAPI, Depends
from pydantic import BaseModel

app = FastAPI()

class StockQuery(BaseModel):
    symbol: str
    criteria: dict

@app.post("/analyze")
async def analyze_stock(query: StockQuery, user=Depends(get_current_user)):
    """Analyze a stock"""
    result = await stock_analyzer.analyze(
        symbol=query.symbol,
        criteria=query.criteria
    )
    return result

@app.get("/watchlist")
async def get_watchlist(user=Depends(get_current_user)):
    """Get user's watchlist with updates"""
    return await fetch_user_watchlist(user.id)

@app.post("/alert")
async def create_alert(symbol: str, condition: dict):
    """Set price/news alert"""
    return await alert_service.create(symbol, condition)
```

### Phase 6: Frontend & Polish (Week 12)

```python
# Tasks:
✅ Build Streamlit dashboard
✅ Add visualizations (Plotly charts)
✅ Implement watchlist feature
✅ Add export functionality

# Streamlit App:
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Analyzer", layout="wide")

# Sidebar
st.sidebar.title("Indian Stock Analyzer")
symbol = st.sidebar.text_input("Enter stock symbol", "TCS")
analyze_btn = st.sidebar.button("Analyze")

if analyze_btn:
    with st.spinner("Analyzing..."):
        result = analyze_stock(symbol)

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rating", result['rating'], result['change'])
        st.write(result['analysis'])

    with col2:
        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=result['dates'],
            open=result['open'],
            high=result['high'],
            low=result['low'],
            close=result['close']
        ))
        st.plotly_chart(fig)

    # Technical indicators
    st.subheader("Technical Analysis")
    st.dataframe(result['technicals'])

    # News sentiment
    st.subheader("Recent News")
    for article in result['news']:
        st.write(f"**{article['title']}** (Sentiment: {article['sentiment']})")
```

---

## 💰 Cost Breakdown

### One-Time Costs

| Item | Cost | Notes |
|------|------|-------|
| **GPU for fine-tuning** | $50-100 | Google Colab Pro or Lambda |
| **Data collection tools** | $0 | All free APIs initially |
| **Development time** | FREE | Your time (300-400 hours) |
| **Domain expert review** | $500-1000 | Optional but recommended |
| **Total** | **$550-1100** | Can skip expert review |

### Monthly Running Costs

| Item | Free Tier | Paid | Notes |
|------|-----------|------|-------|
| **Data APIs** | $0 | $50-200 | Can use free sources |
| **Model hosting** | - | $20-50 | RunPod, Replicate |
| **Vector DB** | $0 (Qdrant local) | $25 (Qdrant cloud) | |
| **Database** | $0 (SQLite/Postgres) | $10 (managed) | |
| **Hosting** | - | $10-20 | Railway, Render |
| **News APIs** | $0 (RSS) | $50 (NewsAPI Pro) | |
| **Total** | **$0** | **$165-355** | Depends on scale |

**Strategy**: Start with $0 using all free tiers, scale up as needed

---

## 🎯 Success Criteria

### Accuracy Metrics

| Metric | Target | Excellent |
|--------|--------|-----------|
| **Rating Accuracy** | > 70% | > 85% |
| **Price Target MAE** | < 15% | < 8% |
| **Sentiment Alignment** | > 75% | > 90% |
| **Recommendation Quality** | 7/10 | 9/10 |

### Performance Metrics

| Metric | Target | Excellent |
|--------|--------|-----------|
| **Analysis Time** | < 5s | < 2s |
| **Data Freshness** | < 1 hour | < 15 min |
| **Uptime** | > 95% | > 99% |
| **API Response Time** | < 2s | < 500ms |

### Functional Requirements

- ✅ Analyze any NSE/BSE listed stock
- ✅ Compare with sector peers
- ✅ Track 50+ stocks in watchlist
- ✅ Real-time news integration
- ✅ Technical + Fundamental analysis
- ✅ Explainable recommendations
- ✅ Alert system for price/news

### Legal & Compliance

⚠️ **IMPORTANT DISCLAIMER REQUIRED**:
```
"This application is for educational and informational purposes only.
It does not constitute financial advice. Always consult with a
registered financial advisor before making investment decisions.
Past performance does not guarantee future results."
```

**NEVER**:
- Claim to provide financial advice
- Guarantee returns
- Make specific buy/sell recommendations without disclaimers
- Use real user money without proper licensing

---

# Timeline Integration

## Where Do These Projects Fit in Your Learning Journey?

### Current Status (March 2026)

```
✅ Completed:
- Module 1: Python Basics
- Module 2: NumPy & Math
- Module 3: Neural Networks
- Module 5: Building LLM
- Module 7: Reasoning & Coding

🟡 In Progress:
- Module 4: Transformers (20%)
- Module 6: Training (50%)
- Module 8: Prompt Engineering (15%)
```

### Roadmap with Projects

```
PHASE 1: FOUNDATIONS (NOW - Month 7)
├── Module 4: Transformers (complete)
├── Module 6: Training (complete)
├── Module 8: Prompt Engineering ⭐ START HERE
├── Module 9: Vector Databases
├── Module 10: RAG
└── Module 11: LangChain
    └── ✨ PROJECT 1 STARTS HERE (Month 7)

PHASE 2: PROJECT 1 DEVELOPMENT (Month 7-9)
├── Week 1-2: Basic chat implementation
├── Week 3-4: Web search integration
├── Week 5-6: RAG system
└── Week 7-8: Polish and deploy
    └── ✨ PROJECT 1 COMPLETE

PHASE 3: ADVANCED TRAINING (Month 9-11)
├── Module 12: Fine-tuning & LoRA
├── Module 13: LLM APIs in Production
└── Module 14: Security & Guardrails
    └── ✨ PROJECT 2 STARTS HERE (Month 11)

PHASE 4: PROJECT 2 DEVELOPMENT (Month 11-14)
├── Week 1-6: Data collection & dataset creation (CRITICAL)
├── Week 7-8: Model fine-tuning
├── Week 9: RAG system
├── Week 10: Real-time data integration
├── Week 11: Application development
└── Week 12: Frontend & deployment
    └── ✨ PROJECT 2 COMPLETE

PHASE 5: ADVANCED TOPICS (Month 15+)
├── Module 15: Multi-modal
├── Module 16-19: Advanced patterns
└── Modules 20-23: Generative AI
```

### Estimated Timeline

| Milestone | Month | Date (Est.) |
|-----------|-------|-------------|
| Current Status | 0 | Mar 2026 |
| Complete Module 8 | 1 | Apr 2026 |
| Complete Module 9-10 | 2-3 | May-Jun 2026 |
| Complete Module 11 | 4 | Jul 2026 |
| **Start Project 1** | 4 | **Jul 2026** |
| **Complete Project 1** | 6 | **Sep 2026** |
| Complete Module 12-14 | 7-9 | Oct-Dec 2026 |
| **Start Project 2** | 9 | **Dec 2026** |
| **Complete Project 2** | 12 | **Mar 2027** |

**Total Time to Both Projects**: ~12 months from now

---

# Technology Stack Comparison

## Project 1 vs Project 2

| Aspect | Project 1 (Chat Assistant) | Project 2 (Stock Analyzer) |
|--------|----------------------------|----------------------------|
| **Complexity** | Medium | Very High |
| **Training Required** | ❌ No | ✅ Yes (Fine-tuning) |
| **Data Collection** | Easy (web scraping) | Hard (financial data + manual labeling) |
| **Cost** | Low ($10-30/month) | Medium ($50-200/month) |
| **Time to Build** | 6-8 weeks | 10-12 weeks |
| **Prerequisites** | Modules 8-11 | Modules 8-14 |
| **Domain Knowledge** | General | Finance-specific |
| **Legal Issues** | Low | High (financial advice disclaimer) |

## Common Technologies

Both projects use:
- LangChain/LangGraph for orchestration
- Vector databases (ChromaDB/Qdrant)
- RAG architecture
- FastAPI backend
- Streamlit frontend
- Python 3.10+

## Unique to Each

**Project 1**:
- Web scraping (BeautifulSoup, Playwright)
- Search APIs (SerpAPI, DuckDuckGo)
- Content extraction (Trafilatura)
- Simple prompting (no fine-tuning)

**Project 2**:
- Fine-tuning (PEFT, LoRA, QLoRA)
- Financial data APIs (yfinance, NSE)
- Technical analysis (TA-Lib)
- Time-series data handling
- Real-time monitoring
- GPU for training

---

# Success Metrics

## Project 1: Chat Assistant

### Functional Tests

```python
test_cases = [
    {
        "query": "What is Python?",
        "expected": "Direct answer from LLM",
        "search_required": False
    },
    {
        "query": "Latest Python 3.13 features",
        "expected": "Search web, cite sources",
        "search_required": True
    },
    {
        "query": "Who won the cricket match today?",
        "expected": "Search sports news, summarize",
        "search_required": True
    }
]
```

### Quality Metrics

- ✅ 95%+ query routing accuracy (search vs direct)
- ✅ 90%+ source relevance
- ✅ 85%+ answer accuracy
- ✅ < 8s response time with search
- ✅ Proper source citations

## Project 2: Stock Analyzer

### Functional Tests

```python
test_cases = [
    {
        "stock": "TCS",
        "expected_rating": "BUY",  # Based on fundamentals
        "reasoning": "Check for P/E, ROE, growth metrics"
    },
    {
        "stock": "RELIANCE",
        "expected_rating": "HOLD",
        "reasoning": "Valuation concerns, sector analysis"
    }
]
```

### Quality Metrics

- ✅ 70%+ rating accuracy vs expert recommendations
- ✅ < 15% MAE on price targets
- ✅ 80%+ fundamental analysis correctness
- ✅ Real-time data < 15 min stale
- ✅ Explainable reasoning for all recommendations

---

# Next Steps

## Immediate Actions (This Week)

### For Project 1 Preparation
1. ✅ Complete Module 8: Prompt Engineering (in progress)
2. ✅ Read LangChain documentation
3. ✅ Experiment with web scraping basics
4. ✅ Set up OpenAI API key

### For Project 2 Preparation
1. ✅ Start collecting free financial data
2. ✅ Create a database schema for stocks
3. ✅ Scrape 100 stock profiles from Screener.in
4. ✅ Read about LoRA fine-tuning

## This Month (March-April 2026)

1. Complete Module 8: Prompt Engineering
2. Start Module 9: Vector Databases
3. Set up development environment:
   - Install dependencies
   - Create project structure
   - Set up GitHub repos

## Next 3 Months (April-June 2026)

1. Complete Modules 9-11
2. Build small proof-of-concept for Project 1
3. Start financial data collection for Project 2

## Summary Checklist

### Project 1: Ready to Start?
- [ ] Module 8 complete (Prompt Engineering)
- [ ] Module 9 complete (Vector Databases)
- [ ] Module 10 complete (RAG)
- [ ] Module 11 complete (LangChain)
- [ ] OpenAI API key obtained
- [ ] Basic web scraping knowledge
- **Ready to start**: Month 7

### Project 2: Ready to Start?
- [ ] All Project 1 prerequisites
- [ ] Module 12 complete (Fine-tuning)
- [ ] Module 13 complete (LLM APIs)
- [ ] Module 14 complete (Security)
- [ ] Financial dataset created (1000+ examples)
- [ ] GPU access arranged
- [ ] Domain knowledge of stocks/mutual funds
- **Ready to start**: Month 11

---

# Conclusion

## Project 1: AI Chat Assistant
- **Difficulty**: ⭐⭐⭐ Medium
- **Learning Value**: ⭐⭐⭐⭐⭐ Excellent
- **Job Market**: ⭐⭐⭐⭐⭐ High demand
- **Recommendation**: **Build this first!**

## Project 2: Stock Analyzer
- **Difficulty**: ⭐⭐⭐⭐⭐ Very Hard
- **Learning Value**: ⭐⭐⭐⭐⭐ Excellent
- **Job Market**: ⭐⭐⭐⭐ Good (FinTech)
- **Recommendation**: **Build after Project 1 + more learning**

## Why This Order?

1. **Project 1 teaches fundamentals** without fine-tuning complexity
2. **Project 1 is faster** to build and see results
3. **Project 2 requires expensive data** preparation
4. **Project 2 needs domain knowledge** (finance)
5. **Success with Project 1** builds confidence for Project 2

## Expected Outcomes

After both projects, you will:
- ✅ Have 2 impressive portfolio projects
- ✅ Understand RAG, agents, and fine-tuning
- ✅ Be job-ready for AI Engineer roles
- ✅ Have practical experience with production AI
- ✅ Understand both general and domain-specific LLMs

**Total Investment**: ~12 months, $100-300 in costs, 600-800 hours

**Career Impact**: $100K-140K AI Engineer positions

---

**Document Status**: COMPLETE AND READY
**Next Action**: Add to your FINAL_COMPLETE_ROADMAP.md
**Questions?**: Revisit this document as you progress through modules

Good luck! 🚀

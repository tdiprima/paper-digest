# AI-Agent-Flow

![License](https://img.shields.io/github/license/tdiprima/OpenAI-Cookbook)
![Languages](https://img.shields.io/github/languages/top/tdiprima/OpenAI-Cookbook)
![Personal](https://img.shields.io/badge/repo-personal-blueviolet)

A hands-on collection of working AI agent examples across seven frameworks — LangChain, CrewAI, LangGraph, Pydantic AI, AutoGen, Agno, and smolagents.

## The Problem With Learning Agent Frameworks

Every major AI agent framework has its own mental model, API style, and tradeoffs. Reading docs only gets you so far — you need runnable, real-world examples to understand when to reach for CrewAI versus LangGraph, or why type-safe agents with Pydantic AI matter for production systems. Most tutorials cover hello-world; few show frameworks doing actual work.

## Working Examples, Not Toy Demos

Each module in this repo solves a real problem using a specific framework. They run end-to-end, connect to live APIs, and demonstrate patterns you'd actually use in production: multi-agent collaboration, RAG with memory, conditional workflow routing, structured output validation, email alerting, and web scraping with deduplication.

## Example: Stock Alert System with CrewAI

Two agents collaborate — a Researcher fetches current and previous prices from Yahoo Finance, and an Analyst calculates the change and sends an email alert if a threshold is crossed:

```bash
# Configure your stocks and alert threshold
vim src/crewai_stock_alert_system/config.py

# Run the multi-agent crew
python src/crewai_stock_alert_system/run_stock_alert.py
```

The Researcher and Analyst agents hand off context automatically. If AAPL drops more than 2%, an email goes out.

## Usage

### Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Install dependencies
uv sync

# Copy and fill in your API keys
cp .env_sample .env
```

**Required environment variables** (see `.env_sample`):

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required by all agents |
| `WEATHER_API_KEY` | OpenWeatherMap (RAG agent) |
| `EMAIL_SENDER` | Gmail address (stock alerts) |
| `EMAIL_PASSWORD` | Gmail app password (stock alerts) |
| `EMAIL_RECEIVER` | Alert recipient (stock alerts) |

### Run Any Example

| Agent | Framework | What It Does | Command |
|---|---|---|---|
| Reasoning transparency | Agno | Shows GPT-4o thinking step-by-step | `python src/agno_hello/hello_agno.py` |
| Multi-agent dev team | AutoGen | CodeGen + Tester agents write and critique code | `python src/autogen_dev_team/create_sorting_algorithm.py` |
| Stock alert system | CrewAI | Monitors prices and sends email alerts | `python src/crewai_stock_alert_system/run_stock_alert.py` |
| Weather RAG agent | LangChain | Conversational Q&A over live forecast data | `python src/langchain_rag_agent/rag_agent.py` |
| Branching workflow | LangGraph | Routes inputs to research, analysis, or escalation | `python src/langgraph_branching_agent/run_branching_agent.py` |
| News analyzer | Pydantic AI | Sentiment, topics, and scoring with type-safe outputs | `python src/pydantic_ai_example/news_analyzer.py` |
| News scraper | Pydantic AI | Fetches RSS feeds and stores validated articles in SQLite | `python src/type_safe_news_agent/run_news_agent.py` |
| Web search agent | smolagents | Answers questions using live web search | `python src/smolagents_hello/hello_smolagents.py` |

> **Note:** The news analyzer reads from the database created by the news scraper. Run `run_news_agent.py` first.

### Project Layout

```
src/
├── agno_hello/                  # Agno reasoning example
├── autogen_dev_team/            # AutoGen multi-agent code collaboration
├── crewai_stock_alert_system/   # CrewAI stock monitoring with email
├── langchain_rag_agent/         # LangChain RAG with FAISS + conversation memory
├── langgraph_branching_agent/   # LangGraph conditional routing
├── pydantic_ai_example/         # Pydantic AI news analysis
├── smolagents_hello/            # smolagents web search
└── type_safe_news_agent/        # Pydantic AI news scraper + SQLite
docs/                            # Per-framework writeups
```

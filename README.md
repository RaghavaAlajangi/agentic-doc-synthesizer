# Agentic AI PoC - Technical Challenge

Your objective is to build a minimal agent-based AI research assistant that can respond to user queries by retrieving relevant internal knowledge from various data sources.

### Core Requirements
- **Upload and index** sample financial data for semantic search (store in database of your choice)
- **Chat interface** where users can ask questions to the assistant
- **Multi-agent workflow** that interprets queries and decides which tools to call automatically
- **Mock integrations** for internal knowledge sources (Confluence, Jira, Azure AI Search, etc.)

### Technical Requirements
- Use any **open-source agentic framework** (LangGraph, Semantic Kernel, AutoGen, etc.)
- Include a **router/planner agent** to determine which specialized agents to call
- **Streaming responses** showing agent thoughts and final answers

## Quick Start

### Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```
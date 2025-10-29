# Agentic AI PoC - Technical Challenge

Your objective is to build a minimal agent-based AI assistant that can summarize and extract key recommendations from sell-side cross-asset research reports (from investment banks like Goldman Sachs, JP Morgan, UBS, etc.) to compare with internal investment views.

### Core Requirements

- **Upload and index** sell-side research reports for semantic search (store in database of your choice)
- **Chat interface** where users can ask questions about cross-asset recommendations (equity, fixed income, multi-asset)
- **Multi-agent workflow** that interprets queries, extracts recommendations, and decides which tools to call automatically
- **Mock integrations** for internal knowledge sources (historical recommendations, analyst tracking, etc.)

### Technical Requirements

- Use any **open-source agentic framework** (MS Agent Framework, Semantic Kernel, AutoGen, LangGraph, etc.)
- Include a **router/planner agent** to determine which specialized agents to call
- **Streaming responses** showing agent thoughts and final answers

> **Note**: Sample sell-side research reports (PDFs) will be provided separately after you accept the challenge.

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

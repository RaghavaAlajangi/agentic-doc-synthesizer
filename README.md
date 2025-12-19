# Agentic AI PoC - Technical Challenge

> **IMPORTANT NOTE**: **DO NOT** create a branch or raise a PR on this repository. Instead, you must **fork this repository** and share your solution with the hiring manager.

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


## Design Concepts

### Data Pipeline & Ingestion

**ETL/Data Ingestion Architecture:**

The system processes research documents through the following pipeline:

Raw Documents  Chunking  Vector Database  Metadata Storage  SQLite DB  API Response

- **Raw Data Ingestion**: PDFs are uploaded and processed
- **Chunking Strategy**: Documents are split into semantic chunks for efficient processing
- **Vector Storage**: Chunks are stored in a vector database with embeddings
- **Metadata Extraction**: Each chunk includes summary metadata for context
- **Document-Level Storage**: Complete document summaries are persisted in SQLite
- **Serving**: Information is retrieved and served through API endpoints

**Limitations:**

- Tables and graphs are not extracted as structured data (treated as images/visual content)
- Most sell-side cross-recommendations are embedded within text as implicit assumptions rather than explicit structured data
- Non-financial-aware embedding models are used, which may not capture domain-specific financial nuances
- Complex financial relationships may be missed without domain-specific model training

### Agent Architecture & Tool Usage

**Simplified 3-Agent Workflow:**

The system uses a streamlined multi-agent orchestration built with LangGraph:

1. **Router Agent**: Analyzes user queries to identify required subtasks
   - Determines if semantic search is needed
   - Identifies if comparison with internal views is required
   - Routes to executor with clear task list

2. **Executor Agent**: Executes identified tasks using simplified tools
   - Retrieves relevant document chunks from vector database (top 5)
   - Fetches document summaries from SQLite
   - Optionally compares external research with internal views
   - Extracts explicit chunk-level citations for attribution
   - Returns aggregated analysis to analyst

3. **Analyst Agent**: Synthesizes all information into final response
   - Combines search results, summaries, and comparisons
   - Generates coherent answer addressing user query
   - Formats response with citations and agent reasoning

**Simplified Tool Stack:**
- **RetrieveTool**: Semantic search on vector database for relevant chunks
- **SQLiteTool**: Fetches document summaries and metadata
- **MockInternalComparisonTool**: Compares external research with internal portfolio views

**Streaming & Transparency:**
- Agents stream thoughts in real-time showing reasoning process
- Each chunk retrieved is tracked as an explicit citation
- Agent decisions are logged and displayed to user

## Features

- **Comprehensive Docstrings**: Well-documented code for maintainability
- **Containerized Deployment**: Docker support for consistent environments
- **Multi-Agent Framework**: LangGraph-based agentic orchestration
- **RESTful API**: FastAPI backend with streaming support
- **Semantic Search**: Vector database integration for intelligent document retrieval
- **Mock Integrations**: Simulated internal knowledge sources

## Limitations

- **No Testing**: Unit and integration tests not implemented
- **No CI/CD**: No automated build or deployment pipeline
- **No Authentication**: Public API endpoints without authentication mechanisms
- **No Chat History Storage**: Conversations are not persisted between sessions
- **Simple UI**: Basic frontend interface without advanced features
- **Limited NLP Models**: Non-financial-specific embedding models

## Application Architecture

### Data Flow Diagram

```
1. DOCUMENT UPLOAD & INGESTION
   ┌─────────────────┐
   │  User uploads   │
   │  PDF document   │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────────────────────────┐
   │  FastAPI: POST /upload-document     │
   └─────────────────┬───────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────┐
   │  Document Processor                 │
   │  • Extract text from PDF            │
   │  • Split into chunks (512-1024 tok) │
   │  • Generate embeddings              │
   │  • Extract metadata                 │
   └──────────────┬──────────────────────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
   ┌──────────────┐    ┌──────────────────┐
   │ Vector DB    │    │ SQLite Database  │
   │ Store chunks │    │ Store summary    │
   │ with vectors │    │ with metadata    │
   └──────────────┘    └──────────────────┘

2. SIMPLIFIED 3-AGENT QUERY PROCESSING
   ┌──────────────────┐
   │  User query      │
   │  (via Chat UI)   │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────────────────┐
   │ FastAPI: POST /chat          │
   │ WebSocket: /stream           │
   └────────┬─────────────────────┘
            │
            ▼
   ┌────────────────────────────────────────┐
   │ 1. ROUTER AGENT                        │
   │    Analyze query → Identify subtasks   │
   │    • retrieve_chunks (always)          │
   │    • compare_with_internal (optional)  │
   └────────┬───────────────────────────────┘
            │
            ▼
   ┌────────────────────────────────────────┐
   │ 2. EXECUTOR AGENT                      │
   │    Execute subtasks with tools         │
   └────┬─────────────────────┬────────┬────┘
        │                     │        │
        ▼                     ▼        ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
   │ RetrieveTool │  │ SQLiteTool   │  │ ComparisonTool   │
   │ (Vector DB)  │  │ (Summaries)  │  │ (Internal views) │
   └──────────────┘  └──────────────┘  └──────────────────┘
        │                     │                  │
        └─────────────────────┴──────────────────┘
                        │
                        ▼
   ┌────────────────────────────────────────────┐
   │ 3. ANALYST AGENT                           │
   │    Synthesize all retrieved information    │
   │    Generate coherent final response        │
   │    Format with citations                   │
   └────────┬───────────────────────────────────┘
            │
            ▼
   ┌────────────────────────────────────────┐
   │ Stream to Frontend                     │
   │ • Agent thoughts (real-time)           │
   │ • Final recommendations                │
   │ • Source citations (chunk-level)       │
   └────────────────────────────────────────┘
```

### Component Breakdown

**Frontend Layer** (React)
- Document upload interface with drag-and-drop
- Query input with suggestion support
- Real-time streaming display of agent thoughts
- Recommendation visualization with source citations

**API Gateway** (FastAPI)
- Request validation and routing
- WebSocket connection management for streaming
- Error handling and response formatting

**Agent Orchestration** (LangGraph)
- **Router Agent**: Analyzes queries and identifies required subtasks
- **Executor Agent**: Executes tasks using available tools and retrieves information
- **Analyst Agent**: Synthesizes results into final comprehensive response
- **State Management**: AgentState passed through all nodes for context preservation

**Document Processing** (services/document_processor.py)
- PDF text extraction and preprocessing
- Semantic chunking (context-aware splitting)
- Embedding generation using language models
- Metadata extraction and storage

**Persistence Layer**
- **Vector Database** (Chroma): Stores document chunks with embeddings for semantic search
- **SQLite Database**: Stores document summaries and metadata
- **Mock Tools**: Simulated internal knowledge sources (portfolio data, internal views)

**Tool Layer** (Simplified)
- **RetrieveTool**: Queries vector database for top 5 relevant chunks with metadata
- **SQLiteTool**: Fetches all document summaries and associated metadata
- **MockInternalComparisonTool**: Provides simulated internal portfolio data for comparison analysis

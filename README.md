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

### Option 1: Docker Compose (Recommended - All-in-One)

**Prerequisites:**
- Docker Desktop or Docker Engine installed
- Docker Compose installed (included with Docker Desktop)

**Installation:**
- **Windows/Mac**: Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: 
  ```bash
  # Install Docker
  sudo apt-get update
  sudo apt-get install docker.io docker-compose
  sudo usermod -aG docker $USER
  
  # Start Docker service
  sudo systemctl start docker
  ```

**Run with Docker Compose:**

```bash
# Navigate to project root
cd agi-technical-challenge

# Start all services (Backend, Frontend, Vector DB, SQLite)
docker-compose up -d

# Check logs
docker-compose logs -f backend    # View backend logs
docker-compose logs -f frontend   # View frontend logs

# Stop all services
docker-compose down
```

**Accessing the Application:**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

**Services Running:**
- Backend (FastAPI): Port 8000
- Frontend (React): Port 3000
- Vector Database (Chroma): Port 8001
- SQLite Database: Persisted in `chroma-data/`

---

### Option 2: Docker Individual Containers

**Prerequisites:**
- Docker installed

**Build Backend Image:**

```bash
cd backend
docker build -t research-agent-backend:latest .
docker run -d \
  --name backend \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key-here \
  -v $(pwd)/../data:/app/data \
  -v $(pwd)/../chroma-data:/app/chroma-data \
  research-agent-backend:latest
```

**Run Backend Container:**

```bash
docker run -d \
  --name backend \
  -p 8000:8000 \
  --env-file .env \
  -v ./data:/app/data \
  -v ./chroma-data:/app/chroma-data \
  research-agent-backend:latest
```

**View Logs:**

```bash
docker logs -f backend
```

**Stop Container:**

```bash
docker stop backend
docker rm backend
```

---

### Option 3: Local Development (Manual Setup)

**Prerequisites:**
- Python 3.9+
- Node.js 16+
- pip
- npm

**Backend Setup:**

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (create .env file or export)
export OPENAI_API_KEY=your-key-here

# Run backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend Setup (New Terminal):**

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Access the Application:**
- Frontend: http://localhost:3000 (or http://localhost:5173 depending on Vite config)
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

---

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.7

# Document Processing
CHUNK_SIZE=1024
CHUNK_OVERLAP=128
DOCUMENT_PROCESSOR_TEMPERATURE=0.5

# Database Configuration
EXTERNAL_CHROMA_HOST=localhost
EXTERNAL_CHROMA_PORT=8001

# API Configuration
API_TITLE=Research Agent API
API_VERSION=1.0.0
LOG_LEVEL=INFO

# CORS
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
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

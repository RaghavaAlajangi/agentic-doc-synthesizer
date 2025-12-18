import asyncio
import json
import logging
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict

from config.settings import settings
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from models.schemas import AgentThought, StreamEvent

from .agent_tools import AgentToolRegistry
from .database import ChromaDBService
from .sqlite_service import SQLiteService

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks that can be executed"""

    RAG_FETCH = "rag_fetch"  # Fetch relevant chunks from RAG
    SUMMARIZE = "summarize"  # Summarize content
    EXTRACT_FINANCIAL = "extract_financial"  # Extract financial data
    COMPARE = "compare"  # Compare external with internal views
    ANALYZE = "analyze"  # General analysis
    SYNTHESIS = "synthesis"  # Final synthesis


class Task(TypedDict):
    """Represents a task to be executed"""

    id: str
    type: TaskType
    description: str
    query: Optional[str]
    status: str  # "pending", "in_progress", "completed", "failed"
    result: Optional[Dict[str, Any]]
    error: Optional[str]


class Citation(TypedDict):
    """Source citation for retrieved information with rich metadata"""

    document_id: str
    document_name: str
    page_number: Optional[int]
    section: Optional[str]
    chunk_index: int
    content_snippet: str
    # Enhanced metadata fields
    company_name: Optional[str]
    report_type: Optional[str]
    report_date: Optional[str]
    document_type: Optional[str]
    author_analyst: Optional[str]
    publication_date: Optional[str]
    total_pages: Optional[int]
    rating: Optional[str]
    target_price: Optional[str]
    similarity_score: Optional[float]


class AgentState(TypedDict):
    """
    Shared state dictionary passed through all agents.

    Attributes
    ----------
    query : str
        The original user query.
    tasks : List[Task]
        Decomposed tasks to be executed.
    task_results : Dict[str, Any]
        Results from executed tasks, keyed by task ID.
    search_results : List[Dict[str, Any]]
        Search results from vector database.
    citations : List[Citation]
        Source citations for all retrieved information.
    summary : Optional[str]
        Summarized content from research documents.
    financial_data : List[Dict[str, Any]]
        Extracted financial metrics and data.
    comparison_result : Optional[str]
        Comparison analysis result.
    agent_thoughts : List[AgentThought]
        Chronological log of agent decisions and reasoning.
    final_response : str
        Final synthesized response to the user.
    """

    query: str
    tasks: List[Task]
    task_results: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    citations: List[Citation]
    summary: Optional[str]
    financial_data: List[Dict[str, Any]]
    comparison_result: Optional[str]
    agent_thoughts: List[AgentThought]
    final_response: str


class AgentOrchestrator:
    """
    Multi-agent system orchestrator using LangGraph.

    Implements a flexible agent network for query processing:
    1. Planning Agent (Router) - Breaks down complex queries into subtasks
    2. RAG Agent - Fetches relevant chunks + document summaries from ETL
    3. Task Executor - Dynamically executes tasks (summarize, extract, compare)
    4. Synthesizer - Generates final response with citations

    Tasks are decomposed and executed dynamically based on their specific
    requirements rather than following a fixed linear path. The orchestrator
    leverages pre-processed artifacts from the ETL pipeline (chunks, summaries,
    metadata) for fast and grounded reasoning.

    Attributes
    ----------
    db : ChromaDBService
        Vector database service for semantic search and metadata retrieval.
    llm : ChatOpenAI
        Language model for agent reasoning.
    tools : AgentToolRegistry
        Registry of available tools for agents.
    graph : CompiledGraph
        Compiled LangGraph DAG for execution.
    """

    def __init__(
        self,
        db_service: ChromaDBService,
        sqlite_service: Optional[SQLiteService] = None,
    ):
        """
        Initialize the dynamic multi-agent orchestrator.

        Parameters
        ----------
        db_service : ChromaDBService
            Vector database service instance.
        sqlite_service : Optional[SQLiteService]
            SQLite service for storing/retrieving document summaries.
        """
        self.db = db_service
        self.sqlite_db = sqlite_service
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            openai_api_key=settings.openai_api_key,
        )

        # Initialize tool registry
        self.tools = AgentToolRegistry(
            db_service=self.db,
            llm_service=self.llm,
        )

        # Build the agent graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build and configure the dynamic LangGraph DAG.

        Constructs a directed acyclic graph with nodes for task
        decomposition, RAG retrieval, and dynamic task execution.

        Returns
        -------
        CompiledGraph
            Compiled executable graph for processing queries.
        """
        graph = StateGraph(AgentState)

        # Add nodes for dynamic workflow
        graph.add_node("planning_agent", self._planning_agent)
        graph.add_node("fetch_rag", self._rag_agent)
        graph.add_node("execute_tasks", self._execute_tasks_agent)
        graph.add_node("synthesize", self._synthesize_agent)

        # Set entry point
        graph.set_entry_point("planning_agent")

        # Linear flow with dynamic task execution
        graph.add_edge("planning_agent", "fetch_rag")
        graph.add_edge("fetch_rag", "execute_tasks")
        graph.add_edge("execute_tasks", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _planning_agent(self, state: AgentState) -> AgentState:
        """
        Plan and decompose complex query into subtasks.

        Analyzes the query to determine what types of tasks need to be
        executed. For example, a query asking about "comparison of
        investment views" would generate tasks for RAG fetch, analysis,
        and comparison.

        Parameters
        ----------
        state : AgentState
            Current state with query.

        Returns
        -------
        AgentState
            Updated state with decomposed tasks.
        """
        query = state["query"].lower()
        tasks: List[Task] = []

        # Always fetch relevant chunks first
        task_id = "task_001"
        tasks.append(
            {
                "id": task_id,
                "type": TaskType.RAG_FETCH,
                "description": "Fetch relevant chunks from documents",
                "query": state["query"],
                "status": "pending",
                "result": None,
                "error": None,
            }
        )

        # Determine additional tasks based on query intent
        task_counter = 2

        # Check if summarization is needed
        if any(
            word in query
            for word in [
                "summarize",
                "summary",
                "overview",
                "brief",
                "key findings",
            ]
        ):
            task_id = f"task_{task_counter:03d}"
            tasks.append(
                {
                    "id": task_id,
                    "type": TaskType.SUMMARIZE,
                    "description": "Summarize relevant documents",
                    "query": state["query"],
                    "status": "pending",
                    "result": None,
                    "error": None,
                }
            )
            task_counter += 1

        # Check if financial extraction is needed
        if any(
            word in query
            for word in [
                "metrics",
                "financial",
                "data",
                "p/e",
                "valuation",
                "earnings",
                "revenue",
                "dividend",
                "yield",
                "price target",
            ]
        ):
            task_id = f"task_{task_counter:03d}"
            tasks.append(
                {
                    "id": task_id,
                    "type": TaskType.EXTRACT_FINANCIAL,
                    "description": "Extract financial metrics and data",
                    "query": state["query"],
                    "status": "pending",
                    "result": None,
                    "error": None,
                }
            )
            task_counter += 1

        # Check if comparison is needed
        if any(
            word in query
            for word in [
                "compare",
                "vs",
                "versus",
                "against",
                "difference",
                "similar",
                "internal",
                "view",
                "position",
            ]
        ):
            task_id = f"task_{task_counter:03d}"
            tasks.append(
                {
                    "id": task_id,
                    "type": TaskType.COMPARE,
                    "description": (
                        "Compare external analysis with internal views"
                    ),
                    "query": state["query"],
                    "status": "pending",
                    "result": None,
                    "error": None,
                }
            )
            task_counter += 1

        # Always add synthesis task
        task_id = f"task_{task_counter:03d}"
        tasks.append(
            {
                "id": task_id,
                "type": TaskType.SYNTHESIS,
                "description": "Synthesize results into final response",
                "query": state["query"],
                "status": "pending",
                "result": None,
                "error": None,
            }
        )

        state["tasks"] = tasks

        thought = AgentThought(
            agent_name="planning_agent",
            thought=f"Decomposed query into {len(tasks)} tasks",
            tool_used="planning_agent",
            tool_output=(
                f"Tasks: {', '.join([t['type'].value for t in tasks])}"
            ),
        )
        state["agent_thoughts"].append(thought)

        logger.info(
            f"Decomposed query into {len(tasks)} tasks: "
            f"{', '.join([t['type'].value for t in tasks])}"
        )
        return state

    async def _rag_agent(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant document chunks with source attribution.

        RAG Pipeline:
        1. Fetch document-level summaries from SQLite (fast context)
        2. Search for relevant chunks from vector DB with section summaries
        3. Extract and store rich citations with page ranges and vital info
        4. Store section summaries in citations for navigation

        This two-tier approach prevents:
        - Blind vector DB searching
        - Missing context/hallucinated facts
        - Inefficient chunk-by-chunk reasoning
        - Polluting vector DB with summary embeddings

        Parameters
        ----------
        state : AgentState
            Current state with query.

        Returns
        -------
        AgentState
            Updated state with search_results and citations with full
            metadata and summaries.
        """
        query = state["query"]

        try:
            logger.info(f"RAG Agent: Starting retrieval for query: {query}")

            # STEP 1: Retrieve document-level summaries from SQLite
            # This gives agents quick overview of relevant documents without
            # polluting vector DB with summary embeddings
            logger.info(
                "RAG Agent: Phase 1 - Retrieving document summaries "
                "from SQLite"
            )
            doc_summaries = []
            if self.sqlite_db:
                try:
                    all_summaries = await self.sqlite_db.get_all_summaries()
                    # In a production system, you might want to rank these
                    # by relevance, but for now we fetch all
                    doc_summaries = [
                        {
                            "document_id": s.document_id,
                            "filename": s.filename,
                            "summary": s.summary_text,
                            "chunk_count": s.chunk_count,
                            "file_size": s.file_size,
                            "source_type": s.source_type,
                        }
                        for s in all_summaries
                    ]
                    logger.info(
                        f"RAG Agent: Retrieved {len(doc_summaries)} "
                        f"document summaries from SQLite"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch summaries from SQLite: {e}. "
                        "Continuing with chunk-only search."
                    )
            else:
                logger.warning(
                    "SQLite service not configured. Skipping "
                    "document summaries retrieval."
                )

            # STEP 2: Retrieve chunk-level results with section summaries
            # Search for chunks within document summary context
            logger.info(
                "RAG Agent: Phase 2 - Retrieving detailed chunks "
                "from vector DB"
            )
            chunk_results = await self.db.search_documents(query, n_results=5)
            state["search_results"] = chunk_results

            # Extract and store citations with rich metadata
            citations: List[Citation] = []
            for idx, result in enumerate(chunk_results):
                # Extract metadata from result
                metadata = result.get("metadata", {})

                citation: Citation = {
                    # Core document identification
                    "document_id": result.get("document_id", "unknown"),
                    "document_name": metadata.get(
                        "filename", "Unknown Document"
                    ),
                    # Location information (for citations)
                    "page_number": metadata.get("page_range", None),
                    "section": metadata.get("section", None),
                    "chunk_index": metadata.get("chunk_index", idx),
                    "total_chunks": metadata.get("total_chunks", None),
                    # Content preview
                    "content_snippet": result.get("content", "")[:200],
                    # Semantic metadata from ETL pipeline
                    "section_summary": metadata.get(
                        "summary", None
                    ),  # 3-4 sentence summary
                    "vital_info": metadata.get(
                        "vital_info", None
                    ),  # {sectors, recommendations, metrics, risks}
                    # Enhanced metadata (not created by ETL, set to None)
                    "company_name": None,
                    "report_type": None,
                    "report_date": None,
                    "document_type": None,
                    "author_analyst": None,
                    "publication_date": None,
                    "total_pages": None,
                    "rating": None,
                    "target_price": None,
                    "similarity_score": result.get("similarity_score", 0.0),
                }
                citations.append(citation)

            state["citations"] = citations

            # Add document summaries to state for agent reasoning
            if doc_summaries:
                state["summary"] = (
                    "Document Summaries Available:\n"
                    + "\n---\n".join(
                        [
                            f"Document: {s['filename']}\n"
                            f"Summary: {s['summary']}\n"
                            f"(Chunks: {s['chunk_count']}, "
                            f"Size: {s['file_size']} bytes)"
                            for s in doc_summaries
                        ]
                    )
                )

            thought = AgentThought(
                agent_name="rag_agent",
                thought=(
                    f"Retrieved {len(doc_summaries)} document summaries "
                    f"from SQLite and {len(chunk_results)} detailed chunks "
                    f"from vector DB with metadata for grounded reasoning"
                ),
                tool_used="search_documents",
                tool_output=(
                    f"Document summaries from SQLite: {len(doc_summaries)}, "
                    f"Chunks with citations: {len(chunk_results)}"
                ),
            )
            state["agent_thoughts"].append(thought)

            logger.info(
                f"RAG Agent: Retrieved {len(doc_summaries)} summaries "
                f"from SQLite and {len(chunk_results)} chunks from vector DB"
            )

        except Exception as e:
            logger.error(f"RAG agent error: {e}")
            thought = AgentThought(
                agent_name="rag_agent",
                thought=f"Error during RAG retrieval: {str(e)}",
            )
            state["agent_thoughts"].append(thought)

        return state

    async def _execute_tasks_agent(self, state: AgentState) -> AgentState:
        """
        Execute decomposed tasks dynamically.

        Iterates through tasks and executes them based on their type,
        collecting results and handling errors gracefully.

        Parameters
        ----------
        state : AgentState
            Current state with tasks to execute.

        Returns
        -------
        AgentState
            Updated state with task_results populated.
        """
        task_results = {}

        for task in state["tasks"]:
            # Skip synthesis task here (handled in synthesize agent)
            if task["type"] == TaskType.SYNTHESIS:
                continue

            # Skip RAG_FETCH task (already done in rag_agent)
            if task["type"] == TaskType.RAG_FETCH:
                continue

            task_id = task["id"]
            task["status"] = "in_progress"

            try:
                if task["type"] == TaskType.SUMMARIZE:
                    thought = AgentThought(
                        agent_name="task_executor",
                        thought=(
                            "Executing SUMMARIZE task - "
                            "Analyzing documents for key findings"
                        ),
                        tool_used="summarize_documents",
                    )
                    state["agent_thoughts"].append(thought)
                    logger.info(f"Executing SUMMARIZE task: {task_id}")

                    result = await self._execute_summarize_task(state, task)
                    task_results[task_id] = result
                    state["summary"] = result.get("summary")
                    task["status"] = "completed"

                elif task["type"] == TaskType.EXTRACT_FINANCIAL:
                    thought = AgentThought(
                        agent_name="task_executor",
                        thought=(
                            "Executing EXTRACT_FINANCIAL task - "
                            "Pulling key metrics and valuations"
                        ),
                        tool_used="extract_financial_data",
                    )
                    state["agent_thoughts"].append(thought)
                    logger.info(f"Executing EXTRACT_FINANCIAL task: {task_id}")

                    result = await self._execute_financial_extraction_task(
                        state, task
                    )
                    task_results[task_id] = result
                    state["financial_data"] = result.get("metrics", [])
                    task["status"] = "completed"

                elif task["type"] == TaskType.COMPARE:
                    thought = AgentThought(
                        agent_name="task_executor",
                        thought=(
                            "Executing COMPARE task - "
                            "Comparing external vs internal views"
                        ),
                        tool_used="compare_analyses",
                    )
                    state["agent_thoughts"].append(thought)
                    logger.info(f"Executing COMPARE task: {task_id}")

                    result = await self._execute_comparison_task(state, task)
                    task_results[task_id] = result
                    state["comparison_result"] = result.get("comparison")
                    task["status"] = "completed"

                elif task["type"] == TaskType.ANALYZE:
                    thought = AgentThought(
                        agent_name="task_executor",
                        thought=(
                            "Executing ANALYZE task - "
                            "Performing general analysis"
                        ),
                        tool_used="analyze_content",
                    )
                    state["agent_thoughts"].append(thought)
                    logger.info(f"Executing ANALYZE task: {task_id}")

                    result = await self._execute_analysis_task(state, task)
                    task_results[task_id] = result
                    task["status"] = "completed"

            except Exception as e:
                logger.error(f"Error executing task {task_id}: {e}")
                task["status"] = "failed"
                task["error"] = str(e)
                task_results[task_id] = {"error": str(e)}

                thought = AgentThought(
                    agent_name="task_executor",
                    thought=f"Error executing {task['type'].value}: {str(e)}",
                )
                state["agent_thoughts"].append(thought)

        state["task_results"] = task_results
        return state

    async def _execute_summarize_task(
        self, state: AgentState, task: Task
    ) -> Dict[str, Any]:
        """Execute summarization task"""
        if not state["search_results"]:
            return {"summary": "", "error": "No search results"}

        combined_content = "\n\n".join(
            [result["content"] for result in state["search_results"][:2]]
        )

        summary_prompt = f"""
        You are a financial research analyst. Summarize the key findings,
        recommendations, and investment thesis from this research report.

        Focus on:
        1. Main thesis and key arguments
        2. Target asset classes and recommendations
        3. Key metrics and data points
        4. Risk factors mentioned
        5. Investment implications

        Research Content:
        {combined_content}

        Provide a comprehensive summary.
        """

        try:
            messages = [
                SystemMessage(
                    content="You are an expert financial analyst "
                    "summarizing research reports."
                ),
                HumanMessage(content=summary_prompt),
            ]

            response = await asyncio.to_thread(self.llm.invoke, messages)
            summary = response.content

            thought = AgentThought(
                agent_name="summarizer",
                thought="Successfully summarized document(s)",
                tool_used="summarize_content",
                tool_output=f"Generated {len(summary)} character summary",
            )
            state["agent_thoughts"].append(thought)

            return {"summary": summary}

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {"summary": "", "error": str(e)}

    async def _execute_financial_extraction_task(
        self, state: AgentState, task: Task
    ) -> Dict[str, Any]:
        """Execute financial extraction task"""
        if not state["search_results"]:
            return {"metrics": [], "error": "No search results"}

        combined_content = "\n\n".join(
            [result["content"] for result in state["search_results"][:3]]
        )

        extraction_prompt = f"""
        Extract all key financial metrics, statements, and data points from
        this research report.

        Focus on:
        1. P/E ratios, valuation multiples
        2. Growth rates (earnings growth, revenue growth)
        3. Dividend yields
        4. Credit metrics (for fixed income)
        5. Asset allocations (for multi-asset)
        6. Price targets and return expectations
        7. Key assumptions

        Format the output as a structured list of key metrics.

        Research Content:
        {combined_content}
        """

        try:
            messages = [
                SystemMessage(
                    content="You are a financial data extraction specialist."
                ),
                HumanMessage(content=extraction_prompt),
            ]

            response = await asyncio.to_thread(self.llm.invoke, messages)
            financial_data_text = response.content

            thought = AgentThought(
                agent_name="financial_extractor",
                thought="Successfully extracted financial data",
                tool_used="extract_financial_data",
                tool_output="Extracted key metrics and statements",
            )
            state["agent_thoughts"].append(thought)

            return {
                "metrics": [
                    {
                        "type": "extracted_metrics",
                        "content": financial_data_text,
                    }
                ]
            }

        except Exception as e:
            logger.error(f"Financial extraction error: {e}")
            return {"metrics": [], "error": str(e)}

    async def _execute_comparison_task(
        self, state: AgentState, task: Task
    ) -> Dict[str, Any]:
        """Execute comparison task"""
        if not state["search_results"]:
            return {"comparison": "", "error": "No search results"}

        external_content = "\n\n".join(
            [result["content"] for result in state["search_results"][:2]]
        )

        # Mock internal view
        internal_view = """
        MOCK INTERNAL ANALYSIS:
        - Maintain defensive positioning in equities
        - Prefer quality dividend stocks
        - Overweight government bonds with 5-7yr maturity
        - Reduce equity risk in near term
        - Monitor inflation data for fixed income strategy
        """

        comparison_prompt = f"""
        Compare the external research recommendations with our internal views.

        EXTERNAL RESEARCH:
        {external_content}

        INTERNAL VIEW:
        {internal_view}

        Provide:
        1. Areas of agreement
        2. Key differences and divergences
        3. Risk factors each emphasizes
        4. Recommended action
        """

        try:
            messages = [
                SystemMessage(
                    content="You are a portfolio manager comparing "
                    "external and internal analysis."
                ),
                HumanMessage(content=comparison_prompt),
            ]

            response = await asyncio.to_thread(self.llm.invoke, messages)
            comparison = response.content

            thought = AgentThought(
                agent_name="comparator",
                thought="Successfully completed comparison analysis",
                tool_used="compare_analyses",
            )
            state["agent_thoughts"].append(thought)

            return {"comparison": comparison}

        except Exception as e:
            logger.error(f"Comparison error: {e}")
            return {"comparison": "", "error": str(e)}

    async def _execute_analysis_task(
        self, state: AgentState, task: Task
    ) -> Dict[str, Any]:
        """Execute general analysis task"""
        if not state["search_results"]:
            return {"analysis": "", "error": "No search results"}

        content = "\n\n".join(
            [result["content"] for result in state["search_results"][:3]]
        )

        analysis_prompt = f"""
        Provide comprehensive analysis based on this research.

        Research Content:
        {content}

        Provide detailed insights and analysis.
        """

        try:
            messages = [
                SystemMessage(
                    content="You are a financial analyst providing insights."
                ),
                HumanMessage(content=analysis_prompt),
            ]

            response = await asyncio.to_thread(self.llm.invoke, messages)
            analysis = response.content

            return {"analysis": analysis}

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {"analysis": "", "error": str(e)}

    async def _synthesize_agent(self, state: AgentState) -> AgentState:
        """
        Generate final synthesized response with source citations.

        Combines results from all executed tasks into a coherent final
        response, including source citations for validation and traceability.

        Parameters
        ----------
        state : AgentState
            Current state with all collected analysis and citations.

        Returns
        -------
        AgentState
            Updated state with final_response and formatted citations.
        """
        query = state["query"]
        summary = state.get("summary")
        financial_data = state.get("financial_data", [])
        comparison_result = state.get("comparison_result")
        search_results = state.get("search_results", [])

        context_parts = []

        if summary:
            context_parts.append(f"SUMMARY:\n{summary}")

        if financial_data:
            context_parts.append(
                f"FINANCIAL DATA:\n{json.dumps(financial_data, indent=2)}"
            )

        if comparison_result:
            context_parts.append(f"COMPARISON ANALYSIS:\n{comparison_result}")

        if search_results:
            context_parts.append(
                f"KEY EXCERPTS:\n{search_results[0]['content'][:300]}..."
            )

        context = "\n\n".join(context_parts)

        synthesis_prompt = f"""
        Based on the analysis below, provide a concise, actionable answer to
        the user's query.

        USER QUERY: {query}

        ANALYSIS:
        {context}

        Provide a clear, direct answer that:
        1. Addresses the specific question
        2. Includes specific data points and metrics where relevant
        3. Provides recommendations
        4. Acknowledges any limitations or uncertainties
        """

        try:
            messages = [
                SystemMessage(
                    content="You are a financial analyst providing "
                    "expert insights based on research."
                ),
                HumanMessage(content=synthesis_prompt),
            ]

            response = await asyncio.to_thread(self.llm.invoke, messages)
            final_response = response.content

            state["final_response"] = final_response

            thought = AgentThought(
                agent_name="synthesizer",
                thought="Generated final response with source citations",
                tool_used="synthesize_response",
            )
            state["agent_thoughts"].append(thought)

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            state["final_response"] = f"Error generating response: {str(e)}"

        return state

    async def _run_graph_async(self, initial_state: AgentState) -> AgentState:
        """
        Execute the agent graph asynchronously.

        Parameters
        ----------
        initial_state : AgentState
            Initial state for graph execution.

        Returns
        -------
        AgentState
            Final state after graph execution.
        """
        return await asyncio.to_thread(self.graph.invoke, initial_state)

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the dynamic multi-agent orchestration graph.

        Routes the query through task decomposition, RAG retrieval, task
        execution, and synthesis. Returns complete results with citations.

        Parameters
        ----------
        query : str
            User question or research request.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
                - response: Final synthesized answer
                - agent_thoughts: List of agent reasoning steps
                - citations: Source citations with metadata
                - search_results: Relevant document chunks
                - task_results: Results from executed tasks
        """
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "tasks": [],
            "task_results": {},
            "search_results": [],
            "citations": [],
            "summary": None,
            "financial_data": [],
            "comparison_result": None,
            "agent_thoughts": [],
            "final_response": "",
        }

        try:
            # Run the graph
            final_state = await self._run_graph_async(initial_state)

            return {
                "response": final_state["final_response"],
                "agent_thoughts": final_state["agent_thoughts"],
                "citations": final_state["citations"],
                "search_results": final_state["search_results"],
                "task_results": final_state["task_results"],
                "recommendations": [],
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"Error processing query: {str(e)}",
                "agent_thoughts": [],
                "citations": [],
                "search_results": [],
                "task_results": {},
                "recommendations": [],
            }

    async def stream_query(
        self, query: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream query processing with real-time agent thoughts and citations.

        Processes query through agents while yielding events for task
        decomposition, RAG retrieval, task execution, and synthesis,
        allowing real-time progress feedback to UI.

        Parameters
        ----------
        query : str
            User question or research request.

        Yields
        ------
        StreamEvent
            Event objects containing agent thoughts, citations,
            search results, and final response.
        """
        initial_state: AgentState = {
            "query": query,
            "tasks": [],
            "task_results": {},
            "search_results": [],
            "citations": [],
            "summary": None,
            "financial_data": [],
            "comparison_result": None,
            "agent_thoughts": [],
            "final_response": "",
        }

        try:
            # Track number of thoughts to detect new ones
            last_thought_count = 0

            # Plan and decompose tasks
            state = await asyncio.to_thread(
                self._planning_agent, initial_state
            )
            new_thoughts = state["agent_thoughts"][last_thought_count:]
            for thought in new_thoughts:
                yield StreamEvent(
                    event_type="agent_thought",
                    data=thought.dict(),
                )
            last_thought_count = len(state["agent_thoughts"])

            # Fetch RAG
            state = await self._rag_agent(state)
            new_thoughts = state["agent_thoughts"][last_thought_count:]
            for thought in new_thoughts:
                yield StreamEvent(
                    event_type="agent_thought",
                    data=thought.dict(),
                )
            last_thought_count = len(state["agent_thoughts"])

            # Yield citations
            for citation in state["citations"]:
                yield StreamEvent(
                    event_type="citation",
                    data=citation,
                )

            # Execute tasks
            state = await self._execute_tasks_agent(state)
            new_thoughts = state["agent_thoughts"][last_thought_count:]
            for thought in new_thoughts:
                yield StreamEvent(
                    event_type="agent_thought",
                    data=thought.dict(),
                )
            last_thought_count = len(state["agent_thoughts"])

            # Synthesize
            state = await self._synthesize_agent(state)
            new_thoughts = state["agent_thoughts"][last_thought_count:]
            for thought in new_thoughts:
                yield StreamEvent(
                    event_type="agent_thought",
                    data=thought.dict(),
                )

            # Yield search results
            for result in state.get("search_results", []):
                yield StreamEvent(
                    event_type="search_result",
                    data=result,
                )

            # Yield final response
            yield StreamEvent(
                event_type="final_response",
                data={"response": state["final_response"]},
            )

        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield StreamEvent(
                event_type="error",
                data={"error": str(e)},
            )

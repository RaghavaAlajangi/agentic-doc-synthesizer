import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict

from config.settings import settings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from models.schemas import AgentThought, StreamEvent

from .agent_tools import AgentToolRegistry
from .database import ChromaDBService
from .sqlite_service import SQLiteService

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Shared state passed through all agents

    Attributes
    ----------
    query : str
        Original user query
    search_results : List[Dict[str, Any]]
        Top 5 relevant chunks from vector DB
    document_summaries : str
        Document summaries from SQLite
    subtasks : List[str]
        Router-identified subtasks
    executor_analysis : str
        Executor agent's analysis and tool decisions
    comparison_result : Optional[str]
        Comparison with internal views if applicable
    agent_thoughts : List[AgentThought]
        Log of agent decisions and reasoning
    citations : List[Dict[str, Any]]
        Source citations from retrieved chunks
    final_response : str
        Final response to user
    """

    query: str
    search_results: List[Dict[str, Any]]
    document_summaries: str
    subtasks: List[str]
    executor_analysis: str
    comparison_result: Optional[str]
    agent_thoughts: List[AgentThought]
    citations: List[Dict[str, Any]]
    final_response: str


class AgentOrchestrator:
    """Simplified multi-agent orchestrator with 3-agent workflow

    Agents:
    1. Router Agent - Understands user intent and identifies subtasks
    2. Executor Agent - Decides which tools to call and executes them
    3. Analyst Agent - Synthesizes all information into final response

    This is much simpler than the previous version with dynamic
    task decomposition and execution.
    """

    def __init__(
        self,
        db_service: ChromaDBService,
        sqlite_service: Optional[SQLiteService] = None,
    ):
        """Initialize the simplified orchestrator

        Parameters
        ----------
        db_service : ChromaDBService
            Vector database service for chunk retrieval
        sqlite_service : Optional[SQLiteService]
            SQLite service for document summaries
        """
        self.db = db_service
        self.sqlite_db = sqlite_service
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            openai_api_key=settings.openai_api_key,
        )

        # Initialize tool registry with simplified tools
        self.tools = AgentToolRegistry(
            db_service=self.db,
            sqlite_service=self.sqlite_db,
            llm_service=self.llm,
        )

        # Build the agent graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the simplified agent graph

        Flow: Router -> Executor -> Analyst

        Returns
        -------
        CompiledGraph
            Compiled executable graph
        """
        graph = StateGraph(AgentState)

        # Add nodes for simplified workflow
        graph.add_node("router", self._router_agent)
        graph.add_node("executor", self._executor_agent)
        graph.add_node("analyst", self._analyst_agent)

        # Set entry point and edges
        graph.set_entry_point("router")
        graph.add_edge("router", "executor")
        graph.add_edge("executor", "analyst")
        graph.add_edge("analyst", END)

        return graph.compile()

    def _router_agent(self, state: AgentState) -> AgentState:
        """Router agent - understands user intent and identifies subtasks

        Analyzes query to determine:
        - What information is needed (chunks, summaries)
        - Whether comparison with internal views is needed
        - Key aspects to focus on

        Parameters
        ----------
        state : AgentState
            Current state with query

        Returns
        -------
        AgentState
            Updated state with identified subtasks
        """
        query = state["query"].lower()

        subtasks = ["retrieve_chunks"]  # Always search for relevant chunks

        # Determine if comparison is needed
        comparison_keywords = [
            "compare",
            "vs",
            "versus",
            "against",
            "internal",
            "view",
            "position",
        ]
        if any(keyword in query for keyword in comparison_keywords):
            subtasks.append("compare_with_internal")

        state["subtasks"] = subtasks

        thought = AgentThought(
            agent_name="router_agent",
            thought=(
                f"Identified {len(subtasks)} subtasks: "
                f"{', '.join(subtasks)}"
            ),
            tool_used="route_query",
            tool_output=f"Subtasks: {subtasks}",
        )
        state["agent_thoughts"].append(thought)

        logger.info(f"Router identified subtasks: {subtasks}")
        return state

    async def _executor_agent(self, state: AgentState) -> AgentState:
        """Executor agent - decides which tools to call and executes them

        Uses router's subtasks to:
        1. Search for relevant chunks from vector DB
        2. Fetch document summaries from SQLite
        3. Optionally compare with internal views
        4. Collects analysis for final synthesis

        Parameters
        ----------
        state : AgentState
            Current state with subtasks

        Returns
        -------
        AgentState
            Updated state with search results and analysis
        """
        try:
            # STEP 1: Search for relevant chunks (get top 5)
            logger.info("Executor: Searching for relevant chunks...")
            search_result = await self.tools.search(
                query=state["query"], n_results=5
            )
            state["search_results"] = search_result.get("results", [])

            # Extract EXPLICIT chunk-level citations from ALL search results
            # Each search result is a distinct chunk with its own citation
            citations = []
            for idx, result in enumerate(state["search_results"], 1):
                metadata = result.get("metadata", {})
                content = result.get("content", "")

                # Create a detailed citation for each chunk
                citation = {
                    "document_id": result.get("document_id", "unknown"),
                    "document_name": metadata.get(
                        "filename", "Unknown Document"
                    ),
                    "page_number": metadata.get("page_range"),
                    "section": metadata.get("section"),
                    "chunk_index": idx - 1,  # 0-based index
                    # Use full content for snippet (truncate in display)
                    "content_snippet": content[:500] if content else "",
                    "similarity_score": result.get("similarity_score", 0.0),
                    "metadata": metadata,
                }
                citations.append(citation)
                logger.info(
                    f"Citation {idx}: {citation['document_name']} "
                    f"(Score: {citation['similarity_score']:.2f})"
                )

            state["citations"] = citations

            # Build scores list for logging
            scores = [f"{c['similarity_score']:.0%}" for c in citations]

            thought = AgentThought(
                agent_name="executor_agent",
                thought=(
                    f"Retrieved {len(state['search_results'])} chunks as "
                    f"citations (relevance: {', '.join(scores)})"
                ),
                tool_used="retrieve_chunks",
                tool_output=(
                    f"Found {len(state['search_results'])} chunks: "
                    f"{[c['document_name'] for c in citations]}"
                ),
            )
            state["agent_thoughts"].append(thought)

            # STEP 2: Fetch document summaries
            logger.info("Executor: Fetching document summaries...")
            summaries_result = await self.tools.sqlite()
            summaries = summaries_result.get("summaries", [])

            # Format summaries for context
            if summaries:
                summary_text = "Document Summaries:\n"
                for s in summaries:
                    summary_text += (
                        f"\n• {s['filename']}\n"
                        f"  Summary: {s['summary'][:150]}...\n"
                    )
                state["document_summaries"] = summary_text
            else:
                state["document_summaries"] = ""

            thought = AgentThought(
                agent_name="executor_agent",
                thought=f"Retrieved {len(summaries)} document summaries",
                tool_used="get_summaries",
                tool_output=f"Found {len(summaries)} document summaries",
            )
            state["agent_thoughts"].append(thought)

            # STEP 3: Comparison if requested
            if "compare_with_internal" in state["subtasks"]:
                logger.info("Executor: Comparing with internal views...")
                external_content = "\n\n".join(
                    [r["content"] for r in state["search_results"][:3]]
                )
                comparison_result = await self.tools.comparison(
                    external=external_content
                )
                state["comparison_result"] = comparison_result.get(
                    "comparison", ""
                )

                thought = AgentThought(
                    agent_name="executor_agent",
                    thought="Compared with internal portfolio views",
                    tool_used="compare_with_internal",
                )
                state["agent_thoughts"].append(thought)

            # Store executor's analysis summary
            state["executor_analysis"] = (
                f"Executor analyzed query with subtasks: {state['subtasks']}. "
                f"Retrieved {len(state['search_results'])} chunks and "
                f"{len(summaries)} document summaries."
            )

            logger.info("Executor: Task execution complete")

        except Exception as e:
            logger.error(f"Executor error: {e}")
            thought = AgentThought(
                agent_name="executor_agent",
                thought=f"Error during execution: {str(e)}",
            )
            state["agent_thoughts"].append(thought)

        return state

    async def _analyst_agent(self, state: AgentState) -> AgentState:
        """Analyst agent - synthesizes all information into final response

        Uses:
        - Original query
        - Retrieved chunks and their content
        - Document summaries
        - Internal comparison (if available)

        To generate actionable, conversational response.

        Parameters
        ----------
        state : AgentState
            Current state with all collected information

        Returns
        -------
        AgentState
            Updated state with final response
        """
        try:
            # Prepare input for synthesis tool
            chunks = [r["content"] for r in state["search_results"]]

            # Call synthesis tool
            response_result = await self.tools.synthesis(
                query=state["query"],
                chunks=chunks,
                document_summaries=state["document_summaries"],
                comparison=state.get("comparison_result"),
            )

            state["final_response"] = response_result.get("response", "")

            thought = AgentThought(
                agent_name="analyst_agent",
                thought="Generated final comprehensive response",
                tool_used="generate_response",
            )
            state["agent_thoughts"].append(thought)

            logger.info("Analyst: Generated final response")

        except Exception as e:
            logger.error(f"Analyst error: {e}")
            state["final_response"] = f"Error generating response: {str(e)}"
            thought = AgentThought(
                agent_name="analyst_agent",
                thought=f"Error during analysis: {str(e)}",
            )
            state["agent_thoughts"].append(thought)

        return state

    async def _run_graph_async(self, initial_state: AgentState) -> AgentState:
        """Execute the agent graph asynchronously

        Parameters
        ----------
        initial_state : AgentState
            Initial state for execution

        Returns
        -------
        AgentState
            Final state after execution
        """
        return await asyncio.to_thread(self.graph.invoke, initial_state)

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the simplified 3-agent workflow

        Parameters
        ----------
        query : str
            User question

        Returns
        -------
        Dict[str, Any]
            Result with response, thoughts, and citations
        """
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "search_results": [],
            "document_summaries": "",
            "subtasks": [],
            "executor_analysis": "",
            "comparison_result": None,
            "agent_thoughts": [],
            "citations": [],
            "final_response": "",
        }

        try:
            # Execute the graph
            final_state = await self._run_graph_async(initial_state)

            return {
                "response": final_state["final_response"],
                "agent_thoughts": final_state["agent_thoughts"],
                "citations": final_state["citations"],
                "search_results": final_state["search_results"],
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"Error: {str(e)}",
                "agent_thoughts": [],
                "citations": [],
                "search_results": [],
            }

    async def stream_query(
        self, query: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream query processing with real-time updates

        Yields events as agents process the query for UI streaming.

        Parameters
        ----------
        query : str
            User question

        Yields
        ------
        StreamEvent
            Real-time events with agent thoughts, citations, results
        """
        initial_state: AgentState = {
            "query": query,
            "search_results": [],
            "document_summaries": "",
            "subtasks": [],
            "executor_analysis": "",
            "comparison_result": None,
            "agent_thoughts": [],
            "citations": [],
            "final_response": "",
        }

        try:
            last_thought_count = 0

            # Route
            state = self._router_agent(initial_state)
            new_thoughts = state["agent_thoughts"][last_thought_count:]
            for thought in new_thoughts:
                yield StreamEvent(
                    event_type="agent_thought",
                    data=thought.dict(),
                )
            last_thought_count = len(state["agent_thoughts"])

            # Execute
            state = await self._executor_agent(state)
            new_thoughts = state["agent_thoughts"][last_thought_count:]
            for thought in new_thoughts:
                yield StreamEvent(
                    event_type="agent_thought",
                    data=thought.dict(),
                )
            last_thought_count = len(state["agent_thoughts"])

            # Analyze
            state = await self._analyst_agent(state)
            new_thoughts = state["agent_thoughts"][last_thought_count:]
            for thought in new_thoughts:
                yield StreamEvent(
                    event_type="agent_thought",
                    data=thought.dict(),
                )

            # Yield final response WITH citations at the end
            yield StreamEvent(
                event_type="final_response",
                data={
                    "response": state["final_response"],
                    "citations": state["citations"],  # ✅ Citations at end
                },
            )

        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield StreamEvent(
                event_type="error",
                data={"error": str(e)},
            )

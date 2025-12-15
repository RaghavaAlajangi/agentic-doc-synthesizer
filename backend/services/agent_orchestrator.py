"""
Smart Multi-Agent Orchestrator using LangGraph.

This implementation creates specialized agents:
1. Router Agent - Determines query type (analysis, comparison, lookup)
2. Summarizer Agent - Summarizes large documents
3. Financial Extractor Agent - Extracts key financial data
4. Comparison Agent - Compares with internal views (only when needed)
5. Response Synthesizer - Creates final response

The agents are called conditionally based on the routing decision.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, TypedDict

from config.settings import settings
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from models.schemas import AgentThought, StreamEvent

from .agent_tools import AgentToolRegistry
from .database import ChromaDBService
from .financial_data import FinancialDataService

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    Shared state dictionary passed through all agents.

    Attributes
    ----------
    query : str
        The original user query.
    query_type : str
        Classified query type ('analysis', 'comparison', or 'lookup').
    search_results : List[Dict[str, Any]]
        Search results from vector database.
    summary : Optional[str]
        Summarized content from research documents.
    financial_data : List[Dict[str, Any]]
        Extracted financial metrics and data.
    agent_thoughts : List[AgentThought]
        Chronological log of agent decisions and reasoning.
    final_response : str
        Final synthesized response to the user.
    """

    query: str
    query_type: str
    search_results: List[Dict[str, Any]]
    summary: Optional[str]
    financial_data: List[Dict[str, Any]]
    agent_thoughts: List[AgentThought]
    final_response: str


class SmartAgentOrchestrator:
    """
    Multi-agent system orchestrator using LangGraph.

    Implements a specialized agent network for query processing:
    1. Router - Classifies query intent
    2. Summarizer - Extracts key findings from documents
    3. Financial Extractor - Pulls financial metrics
    4. Comparator - Analyzes internal vs external views
    5. Synthesizer - Generates final response

    Agents execute conditionally based on routing decisions to optimize
    processing for different query types.

    Attributes
    ----------
    db : ChromaDBService
        Vector database service for semantic search.
    financial_data : FinancialDataService
        Service for financial data retrieval.
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
        financial_data_service: FinancialDataService,
    ):
        """
        Initialize the multi-agent orchestrator.

        Parameters
        ----------
        db_service : ChromaDBService
            Vector database service instance.
        financial_data_service : FinancialDataService
            Financial data service instance.
        """
        self.db = db_service
        self.financial_data = financial_data_service
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            openai_api_key=settings.openai_api_key,
        )

        # Initialize tool registry
        self.tools = AgentToolRegistry(
            db_service=self.db,
            financial_data_service=self.financial_data,
            llm_service=self.llm,
        )

        # Build the agent graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build and configure the LangGraph DAG.

        Constructs the directed acyclic graph with nodes for each agent
        and conditional edges for routing based on query type.

        Returns
        -------
        CompiledGraph
            Compiled executable graph for processing queries.
        """
        graph = StateGraph(AgentState)

        # Add nodes (agents)
        graph.add_node("route", self._route_agent)
        graph.add_node("summarize", self._summarize_agent)
        graph.add_node("extract_financial", self._extract_financial_agent)
        graph.add_node("compare", self._compare_agent)
        graph.add_node("synthesize", self._synthesize_agent)

        # Set entry point
        graph.set_entry_point("route")

        # After routing, decide path
        graph.add_conditional_edges(
            "route",
            self._route_decision,
            {
                "analysis": "summarize",
                "lookup": "extract_financial",
                "comparison": "compare",
                "end": END,
            },
        )

        # After summarization, extract financial data
        graph.add_edge("summarize", "extract_financial")

        # After financial extraction, synthesize
        graph.add_edge("extract_financial", "synthesize")

        # After comparison, synthesize
        graph.add_edge("compare", "synthesize")

        # After synthesize, end
        graph.add_edge("synthesize", END)

        return graph.compile()

    def _route_decision(self, state: AgentState) -> str:
        """
        Determine execution path based on query classification.

        Parameters
        ----------
        state : AgentState
            Current state containing query_type.

        Returns
        -------
        str
            Name of next agent node to execute.
        """
        query_type = state.get("query_type", "analysis")

        if query_type == "comparison":
            return "comparison"
        elif query_type == "lookup":
            return "lookup"
        else:
            return "analysis"

    async def _route_agent(self, state: AgentState) -> AgentState:
        """
        Classify incoming query to determine processing path.

        Analyzes query content for keywords indicating analysis,
        lookup, or comparison operations.

        Parameters
        ----------
        state : AgentState
            Current state with query text.

        Returns
        -------
        AgentState
            Updated state with query_type set.
        """
        query = state["query"].lower()

        # Detect query intent
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
            query_type = "comparison"
        elif any(
            word in query
            for word in [
                "find",
                "what",
                "which",
                "show",
                "list",
                "extract",
                "key",
                "data",
                "metrics",
            ]
        ):
            query_type = "lookup"
        else:
            query_type = "analysis"

        thought = AgentThought(
            agent_name="router",
            thought=f"Identified query type as: {query_type}",
            tool_used="query_classifier",
        )

        state["query_type"] = query_type
        state["agent_thoughts"].append(thought)

        logger.info(f"Route decision: {query_type}")
        return state

    async def _summarize_agent(self, state: AgentState) -> AgentState:
        """
        Summarize research reports and key findings.

        Searches for relevant documents matching the query and creates
        a concise summary highlighting key findings, recommendations,
        and investment thesis.

        Parameters
        ----------
        state : AgentState
            Current state with query.

        Returns
        -------
        AgentState
            Updated state with summary and search_results.
        """
        query = state["query"]

        # Search for relevant documents
        search_results = await self.db.search_documents(query, n_results=3)
        state["search_results"] = search_results

        if not search_results:
            thought = AgentThought(
                agent_name="summarizer",
                thought="No relevant documents found for summarization",
            )
            state["agent_thoughts"].append(thought)
            return state

        # Combine top results
        combined_content = "\n\n".join(
            [result["content"] for result in search_results[:2]]
        )

        # Create summarization prompt
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

        Provide a comprehensive summary that would help investors understand
        the key takeaways from this research.
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

            state["summary"] = summary

            summary_tool = self.tools.get_tool("summarize_content")
            if summary_tool:
                tool_name = summary_tool.definition.name
            else:
                tool_name = "summarize_content"

            thought = AgentThought(
                agent_name="summarizer",
                thought="Successfully summarized document(s)",
                tool_used=tool_name,
                tool_output=f"Generated {len(summary)} character summary",
            )
            state["agent_thoughts"].append(thought)

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            thought = AgentThought(
                agent_name="summarizer",
                thought=f"Error during summarization: {str(e)}",
            )
            state["agent_thoughts"].append(thought)

        return state

    async def _extract_financial_agent(self, state: AgentState) -> AgentState:
        """
        Extract financial metrics and data from documents.

        Identifies and extracts key financial statements, metrics,
        valuation data, and relevant numbers from research documents.

        Parameters
        ----------
        state : AgentState
            Current state with query and search results.

        Returns
        -------
        AgentState
            Updated state with financial_data extracted.
        """
        query = state["query"]

        # Search for relevant documents
        search_results = await self.db.search_documents(query, n_results=5)

        if not search_results:
            thought = AgentThought(
                agent_name="financial_extractor",
                thought="No relevant documents found for extraction",
            )
            state["agent_thoughts"].append(thought)
            return state

        state["search_results"] = search_results

        # Extract financial statements
        combined_content = "\n\n".join(
            [result["content"] for result in search_results[:3]]
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

        Format the output as a structured list of key metrics with their values.

        Research Content:
        {combined_content}

        Provide extracted financial data in a clear, structured format.
        """

        try:
            messages = [
                SystemMessage(
                    content="You are a financial data extraction "
                    "specialist."
                ),
                HumanMessage(content=extraction_prompt),
            ]

            response = await asyncio.to_thread(self.llm.invoke, messages)
            financial_data_text = response.content

            state["financial_data"] = [
                {
                    "type": "extracted_metrics",
                    "content": financial_data_text,
                }
            ]

            thought = AgentThought(
                agent_name="financial_extractor",
                thought="Successfully extracted financial data",
                tool_used="extract_financial_data",
                tool_output="Extracted key metrics and statements",
            )
            state["agent_thoughts"].append(thought)

        except Exception as e:
            logger.error(f"Financial extraction error: {e}")
            thought = AgentThought(
                agent_name="financial_extractor",
                thought=f"Error during extraction: {str(e)}",
            )
            state["agent_thoughts"].append(thought)

        return state

    async def _compare_agent(self, state: AgentState) -> AgentState:
        """
        Compare external analysis with internal views.

        Analyzes differences between external research recommendations
        and internal portfolio positions/views.

        Parameters
        ----------
        state : AgentState
            Current state with query and external analysis.

        Returns
        -------
        AgentState
            Updated state with comparison results.
        """
        query = state["query"]

        # Get external analysis
        search_results = await self.db.search_documents(query, n_results=3)

        if not search_results:
            thought = AgentThought(
                agent_name="comparator",
                thought="No relevant documents found for comparison",
            )
            state["agent_thoughts"].append(thought)
            return state

        state["search_results"] = search_results

        external_content = "\n\n".join(
            [result["content"] for result in search_results[:2]]
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
        Compare the external research recommendations with our internal
        views.

        EXTERNAL RESEARCH:
        {external_content}

        INTERNAL VIEW:
        {internal_view}

        Provide:
        1. Areas of agreement between external and internal views
        2. Key differences and divergences
        3. Risk factors each view emphasizes
        4. Recommended action based on both perspectives
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
            comparison_result = response.content

            state["final_response"] = comparison_result

            thought = AgentThought(
                agent_name="comparator",
                thought="Successfully completed comparison analysis",
                tool_used="compare_analyses",
            )
            state["agent_thoughts"].append(thought)

            return state

        except Exception as e:
            logger.error(f"Comparison error: {e}")
            thought = AgentThought(
                agent_name="comparator",
                thought=f"Error during comparison: {str(e)}",
            )
            state["agent_thoughts"].append(thought)
            return state

    async def _synthesize_agent(self, state: AgentState) -> AgentState:
        """
        Generate final synthesized response.

        Combines results from all preceding agents (summary, financial
        data, comparisons) into a coherent final response to the user.

        Parameters
        ----------
        state : AgentState
            Current state with all collected analysis.

        Returns
        -------
        AgentState
            Updated state with final_response generated.
        """
        query = state["query"]
        summary = state.get("summary")
        financial_data = state.get("financial_data", [])
        search_results = state.get("search_results", [])

        # If already generated by comparison agent, use that
        if state.get("final_response"):
            return state

        context_parts = []

        if summary:
            context_parts.append(f"SUMMARY:\n{summary}")

        if financial_data:
            context_parts.append(
                f"FINANCIAL DATA:\n" f"{json.dumps(financial_data, indent=2)}"
            )

        if search_results:
            context_parts.append(
                f"RELEVANT EXCERPTS:\n"
                f"{search_results[0]['content'][:500]}..."
            )

        context = "\n\n".join(context_parts)

        synthesis_prompt = f"""
        Based on the analysis below, provide a concise answer to the user's
        query.

        USER QUERY: {query}

        ANALYSIS:
        {context}

        Provide a clear, actionable answer that addresses the user's question.
        Include specific data points and recommendations where applicable.
        """

        try:
            messages = [
                SystemMessage(
                    content="You are a financial analyst providing "
                    "expert insights."
                ),
                HumanMessage(content=synthesis_prompt),
            ]

            response = await asyncio.to_thread(self.llm.invoke, messages)
            final_response = response.content

            state["final_response"] = final_response

            thought = AgentThought(
                agent_name="synthesizer",
                thought="Generated final response",
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
        Process a query through the multi-agent orchestration graph.

        Routes the query through specialized agents for analysis,
        summarization, financial extraction, and synthesis. Returns
        complete results with agent reasoning steps.

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
                - search_results: Relevant document chunks
                - recommendations: Investment recommendations
        """
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "query_type": "analysis",
            "search_results": [],
            "summary": None,
            "financial_data": [],
            "agent_thoughts": [],
            "final_response": "",
        }

        try:
            # Run the graph
            final_state = await self._run_graph_async(initial_state)

            return {
                "response": final_state["final_response"],
                "agent_thoughts": final_state["agent_thoughts"],
                "search_results": final_state["search_results"],
                "recommendations": [],
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"Error processing query: {str(e)}",
                "agent_thoughts": [],
                "search_results": [],
                "recommendations": [],
            }

    async def stream_query(
        self, query: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream query processing with real-time agent thoughts.

        Processes query through agents while yielding events for each
        agent action, allowing real-time progress feedback to UI.

        Parameters
        ----------
        query : str
            User question or research request.

        Yields
        ------
        StreamEvent
            Event objects containing agent thoughts, actions, and
            intermediate results.
        """
        initial_state: AgentState = {
            "query": query,
            "query_type": "analysis",
            "search_results": [],
            "summary": None,
            "financial_data": [],
            "agent_thoughts": [],
            "final_response": "",
        }

        try:
            # Route agent
            state = await self._route_agent(initial_state)
            for thought in state["agent_thoughts"]:
                yield StreamEvent(
                    event_type="agent_thought",
                    data=thought.dict(),
                )

            # Process based on query type
            query_type = state.get("query_type", "analysis")

            if query_type == "comparison":
                state = await self._compare_agent(state)
                for thought in state["agent_thoughts"][1:]:
                    yield StreamEvent(
                        event_type="agent_thought",
                        data=thought.dict(),
                    )
            elif query_type == "lookup":
                state = await self._extract_financial_agent(state)
                for thought in state["agent_thoughts"][1:]:
                    yield StreamEvent(
                        event_type="agent_thought",
                        data=thought.dict(),
                    )
            else:
                # Analysis flow
                state = await self._summarize_agent(state)
                for thought in state["agent_thoughts"][1:]:
                    yield StreamEvent(
                        event_type="agent_thought",
                        data=thought.dict(),
                    )

                state = await self._extract_financial_agent(state)
                new_thoughts = state["agent_thoughts"][
                    len([t for t in initial_state["agent_thoughts"]]) :
                ]
                for thought in new_thoughts:
                    yield StreamEvent(
                        event_type="agent_thought",
                        data=thought.dict(),
                    )

            # Synthesize if not already done
            if not state.get("final_response"):
                state = await self._synthesize_agent(state)
                yield StreamEvent(
                    event_type="agent_thought",
                    data={
                        "agent": "synthesizer",
                        "thought": "Generated final response",
                    },
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

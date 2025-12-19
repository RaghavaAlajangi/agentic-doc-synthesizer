import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class RetrieveTool:
    """Tool for retrieving research document chunks from vector DB"""

    def __init__(self, db_service):
        """Initialize with database service

        Parameters
        ----------
        db_service : ChromaDBService
            Vector database service for chunk retrieval
        """
        self.db = db_service

    async def __call__(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Retrieve relevant chunks from vector database

        Parameters
        ----------
        query : str
            Search query
        n_results : int, optional
            Number of results to return, by default 5

        Returns
        -------
        Dict[str, Any]
            Dictionary with results list containing chunks with metadata
        """
        try:
            results = await self.db.search_documents(query, n_results)
            logger.info(f"Retrieve tool: Found {len(results)} chunks")
            return {"results": results}
        except Exception as e:
            logger.error(f"Retrieve tool error: {e}")
            return {"results": [], "error": str(e)}


class SQLiteTool:
    """Tool for fetching document summaries from SQLite"""

    def __init__(self, sqlite_service):
        """Initialize with SQLite service

        Parameters
        ----------
        sqlite_service : SQLiteService
            SQLite service for document summary retrieval
        """
        self.sqlite_db = sqlite_service

    async def __call__(self) -> Dict[str, Any]:
        """Fetch all document summaries from SQLite

        Returns
        -------
        Dict[str, Any]
            Dictionary with summaries list
        """
        try:
            if not self.sqlite_db:
                return {"summaries": []}

            summaries = await self.sqlite_db.get_all_summaries()
            result = [
                {
                    "document_id": s.document_id,
                    "filename": s.filename,
                    "summary": s.summary_text,
                    "chunk_count": s.chunk_count,
                    "file_size": s.file_size,
                    "source_type": s.source_type,
                }
                for s in summaries
            ]
            logger.info(
                f"SQLite tool: Retrieved {len(result)} document summaries"
            )
            return {"summaries": result}
        except Exception as e:
            logger.error(f"SQLite tool error: {e}")
            return {"summaries": [], "error": str(e)}


class MockInternalComparisonTool:
    """Tool for comparing external research with internal portfolio views

    Fetches mock internal portfolio data and provides comparison analysis.
    In production, this would integrate with Bloomberg/Aladdin/internal APIs.
    """

    def __init__(self, llm_service):
        """Initialize with LLM service

        Parameters
        ----------
        llm_service : LLMService
            LLM service for comparison reasoning
        """
        self.llm = llm_service

    async def __call__(
        self, external: str, portfolio_id: str = "internal_portfolio"
    ) -> Dict[str, Any]:
        """Execute comparison with internal portfolio data

        Parameters
        ----------
        external : str
            External research analysis from sell-side report
        portfolio_id : str, optional
            Portfolio identifier, by default "internal_portfolio"

        Returns
        -------
        Dict[str, Any]
            Comparison result with internal data and analysis
        """
        try:
            # Mock internal portfolio data
            mock_internal_data = {
                "portfolio_id": portfolio_id,
                "portfolio_type": "Multi-Asset",
                "as_of_date": "2024-12-18",
                "current_allocation": {
                    "equities": 62,
                    "fixed_income": 28,
                    "commodities": 5,
                    "alternatives": 5,
                },
                "buy_calls": [
                    {
                        "asset": "Technology",
                        "conviction": "High",
                        "target_allocation": 12,
                        "current_allocation": 10,
                        "rationale": "AI infrastructure opportunity",
                    },
                    {
                        "asset": "Energy",
                        "conviction": "Medium",
                        "target_allocation": 5,
                        "current_allocation": 3,
                        "rationale": "Geopolitical supply premium",
                    },
                ],
                "sell_calls": [
                    {
                        "asset": "Long-Duration Bonds",
                        "conviction": "High",
                        "target_allocation": 15,
                        "current_allocation": 18,
                        "rationale": "Limited upside at current yields",
                    },
                ],
                "risk_scores": {
                    "market_risk": 6.5,
                    "credit_risk": 4.2,
                    "liquidity_risk": 3.1,
                },
            }

            # Generate comparison with LLM
            prompt = (
                "Compare external sell-side research with our "
                "internal portfolio views. Be conversational and "
                "focus on reasoning rather than data dumps.\n\n"
                f"EXTERNAL RESEARCH:\n{external}\n\n"
                "INTERNAL VIEW:\n"
                "We are cautiously constructive with focus on "
                "quality. Moderately favor tech, skeptical on "
                "long-duration bonds. Tracking geopolitical risks.\n\n"
                "Provide natural comparison analysis (1-2 paragraphs)."
            )

            messages = [
                SystemMessage(
                    content=(
                        "You are a portfolio manager comparing "
                        "research. Be conversational and realistic."
                    )
                ),
                HumanMessage(content=prompt),
            ]

            response = await self._invoke_llm(messages)
            logger.info("Mock comparison: Completed analysis")

            return {
                "internal_data": mock_internal_data,
                "comparison": response,
                "validation_score": 75,
            }
        except Exception as e:
            logger.error(f"Mock comparison error: {e}")
            return {"comparison": "", "error": str(e)}

    async def _invoke_llm(self, messages):
        """Invoke LLM asynchronously"""
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content


class SynthesisTool:
    """Tool for generating final comprehensive response

    Combines query understanding, chunk retrieval, and internal validation
    to generate actionable investment guidance.
    """

    def __init__(self, llm_service):
        """Initialize with LLM service

        Parameters
        ----------
        llm_service : LLMService
            LLM service for response generation
        """
        self.llm = llm_service

    async def __call__(
        self,
        query: str,
        chunks: List[str],
        document_summaries: str,
        comparison: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive response

        Synthesizes response using:
        - Top 5 retrieved chunks (all provided for context)
        - Document-level summaries
        - Optional internal portfolio comparison

        Parameters
        ----------
        query : str
            Original user query
        chunks : List[str]
            Retrieved chunk contents (all 5 chunks from search)
        document_summaries : str
            Document-level summaries from SQLite
        comparison : Optional[str]
            Internal portfolio comparison if available

        Returns
        -------
        Dict[str, Any]
            Response with generated answer
        """
        try:
            # Use ALL retrieved chunks as context (not just first 3)
            # This ensures the LLM has access to all relevant information
            chunks_context = ""
            if chunks:
                chunks_list = []
                for idx, chunk in enumerate(chunks, 1):
                    chunks_list.append(f"[Chunk {idx}]\n{chunk}")
                chunks_context = "\n\n---CHUNK SEPARATOR---\n\n".join(
                    chunks_list
                )

            prompt = (
                "You are a sell-side research analyst providing "
                "actionable investment guidance based on research "
                "reports and internal portfolio analysis.\n\n"
                f"USER QUESTION: {query}\n\n"
                f"DOCUMENT SUMMARIES:\n{document_summaries}\n\n"
            )

            if chunks_context:
                prompt += (
                    f"DETAILED SECTIONS FROM RESEARCH "
                    f"(Top 5 Most Relevant Chunks):\n\n"
                    f"{chunks_context}\n\n"
                )

            if comparison:
                prompt += f"INTERNAL PORTFOLIO ASSESSMENT:\n{comparison}\n\n"

            prompt += (
                "Provide a clear, conversational response that:\n"
                "1. Directly addresses the user's question\n"
                "2. Focuses on actionable recommendations\n"
                "3. Weaves in key insights naturally from the chunks\n"
                "4. Explains practical portfolio implications\n"
                "5. Acknowledges internal validation if applicable\n"
                "6. Cite specific sections when making key points"
            )

            messages = [
                SystemMessage(
                    content=(
                        "You are a portfolio manager synthesizing research "
                        "and internal views. Be clear, conversational, and "
                        "focus on actionable guidance."
                    )
                ),
                HumanMessage(content=prompt),
            ]

            response = await self._invoke_llm(messages)
            logger.info("Synthesis tool: Generated response")

            return {"response": response}
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return {"response": "", "error": str(e)}

    async def _invoke_llm(self, messages):
        """Invoke LLM asynchronously"""
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content


class AgentToolRegistry:
    """Registry of available tools for agent workflow"""

    def __init__(
        self,
        db_service,
        sqlite_service,
        llm_service,
    ):
        """Initialize tool registry

        Parameters
        ----------
        db_service : ChromaDBService
            Database service for chunk search
        sqlite_service : SQLiteService
            SQLite service for document summaries
        llm_service : LLMService
            LLM service for reasoning and generation
        """
        # Initialize tools
        self.search = RetrieveTool(db_service)
        self.sqlite = SQLiteTool(sqlite_service)
        self.comparison = MockInternalComparisonTool(llm_service)
        self.synthesis = SynthesisTool(llm_service)

        # Tool mapping
        self.tools_map = {
            "retrieve_chunks": self.search,
            "get_summaries": self.sqlite,
            "compare_with_internal": self.comparison,
            "generate_response": self.synthesis,
        }

        logger.info(
            f"Tool registry initialized with {len(self.tools_map)} tools"
        )

    def get_tool(self, name: str):
        """Get tool by name

        Parameters
        ----------
        name : str
            Tool name

        Returns
        -------
        Optional[Tool]
            Tool instance if found, None otherwise
        """
        return self.tools_map.get(name)

    def get_all_tools(self) -> Dict[str, Any]:
        """Get all tools

        Returns
        -------
        Dict[str, Any]
            Dictionary of all tools
        """
        return self.tools_map

    def list_tools(self) -> List[str]:
        """List all available tools

        Returns
        -------
        List[str]
            List of tool names
        """
        return list(self.tools_map.keys())

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute
        **kwargs
            Tool arguments

        Returns
        -------
        Dict[str, Any]
            Tool result
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}

        try:
            result = await tool(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": str(e)}

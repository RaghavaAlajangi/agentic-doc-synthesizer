import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ToolDefinition:
    """Base class for tool definitions"""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
    ):
        """Initialize tool definition

        Parameters
        ----------
        name : str
            Unique tool identifier
        description : str
            Human-readable description
        input_schema : Dict[str, Any]
            Input parameters schema
        output_schema : Dict[str, Any]
            Output format schema
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema

    def to_dict(self) -> Dict[str, Any]:
        """Export tool definition as dictionary

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the tool definition
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


class SearchTool:
    """Tool for searching research documents"""

    def __init__(self, db_service):
        """Initialize with database service

        Parameters
        ----------
        db_service : DatabaseService
            Database service for document operations
        """
        self.db = db_service
        self.definition = ToolDefinition(
            name="search_documents",
            description=(
                "Search research reports by semantic "
                "similarity to find relevant content"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query to find " "relevant documents"
                        ),
                    },
                    "n_results": {
                        "type": "integer",
                        "description": (
                            "Number of results " "to return (default: 5)"
                        ),
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "document_id": {"type": "string"},
                                "content": {"type": "string"},
                                "similarity_score": {"type": "number"},
                                "source": {"type": "string"},
                                "metadata": {"type": "object"},
                            },
                        },
                        "description": ("List of matching documents"),
                    }
                },
                "required": ["results"],
            },
        )

    async def __call__(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Execute search

        Parameters
        ----------
        query : str
            Search query
        n_results : int, optional
            Number of results to return, by default 5

        Returns
        -------
        Dict[str, Any]
            Search results dictionary containing results list
        """
        try:
            results = await self.db.search_documents(query, n_results)
            logger.info(f"Search tool: Found {len(results)} results")
            return {"results": results}
        except Exception as e:
            logger.error(f"Search tool error: {e}")
            return {"results": [], "error": str(e)}


class SummarizationTool:
    """Tool for summarizing research content"""

    def __init__(self, llm_service):
        """Initialize with LLM service

        Parameters
        ----------
        llm_service : LLMService
            LLM service for text generation
        """
        self.llm = llm_service
        self.definition = ToolDefinition(
            name="summarize_content",
            description=(
                "Summarize research report content "
                "into key findings and thesis"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": ("Research content to summarize"),
                    }
                },
                "required": ["content"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": ("Concise summary of the research"),
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Top 5 key points",
                    },
                },
                "required": ["summary"],
            },
        )

    async def __call__(self, content: str) -> Dict[str, Any]:
        """Execute summarization

        Parameters
        ----------
        content : str
            Content to summarize

        Returns
        -------
        Dict[str, Any]
            Dictionary containing summary and key points
        """
        try:
            prompt = (
                "You are a sell-side research analyst. "
                "Summarize the key investment recommendations "
                "from this cross-asset research report.\n\n"
                "Focus on:\n"
                "1. Specific recommendations by asset class\n"
                "2. Price targets and valuation levels\n"
                "3. Recommended positioning and allocation\n"
                "4. Key macro drivers and themes\n"
                "5. Primary risks and catalysts\n"
                "6. Timing and implementation guidance\n\n"
                f"Content:\n{content}\n\n"
                "Provide a comprehensive summary of "
                "actionable recommendations."
            )

            messages = [
                SystemMessage(
                    content=("You are an expert sell-side research analyst.")
                ),
                HumanMessage(content=prompt),
            ]

            response = await self._invoke_llm(messages)
            logger.info(
                f"Summarization tool: Generated "
                f"{len(response)} character summary"
            )

            return {
                "summary": response,
                "key_points": ["See full summary for detailed points"],
            }
        except Exception as e:
            logger.error(f"Summarization tool error: {e}")
            return {"summary": "", "error": str(e)}

    async def _invoke_llm(self, messages):
        """Invoke LLM asynchronously

        Parameters
        ----------
        messages : list
            List of messages to send to LLM

        Returns
        -------
        str
            LLM response content
        """
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content


class FinancialExtractionTool:
    """Tool for extracting financial metrics and data"""

    def __init__(self, llm_service):
        """Initialize with LLM service

        Parameters
        ----------
        llm_service : LLMService
            LLM service for text generation
        """
        self.llm = llm_service
        self.definition = ToolDefinition(
            name="extract_financial_data",
            description=(
                "Extract key financial metrics, "
                "valuations, and data points from "
                "research content"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": ("Research content to extract from"),
                    }
                },
                "required": ["content"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "object",
                        "description": ("Extracted financial metrics"),
                    },
                    "valuations": {
                        "type": "array",
                        "description": "Valuation multiples",
                    },
                    "targets": {
                        "type": "array",
                        "description": ("Price targets and forecasts"),
                    },
                },
                "required": ["metrics"],
            },
        )

    async def __call__(self, content: str) -> Dict[str, Any]:
        """Execute financial extraction

        Parameters
        ----------
        content : str
            Content to extract financial data from

        Returns
        -------
        Dict[str, Any]
            Dictionary containing extracted metrics, valuations, and targets
        """
        try:
            prompt = (
                "Extract all recommendation metrics and "
                "allocation guidance from this sell-side "
                "cross-asset research.\n\n"
                "Focus on:\n"
                "1. Specific recommendations with ratings\n"
                "2. Price targets and valuation ranges\n"
                "3. Portfolio allocations by asset class\n"
                "4. Performance drivers and catalysts\n"
                "5. Forecast assumptions\n"
                "6. Risk/reward analysis and rationale\n"
                "7. Entry/exit levels and timing\n\n"
                f"Content:\n{content}\n\n"
                "Format as structured list of key metrics."
            )

            messages = [
                SystemMessage(
                    content=(
                        "You are a sell-side research data "
                        "specialist extracting metrics."
                    )
                ),
                HumanMessage(content=prompt),
            ]

            response = await self._invoke_llm(messages)
            logger.info("Financial extraction tool: Extracted data")

            return {
                "metrics": {"raw_extraction": response},
                "valuations": [],
                "targets": [],
            }
        except Exception as e:
            logger.error(f"Financial extraction error: {e}")
            return {"metrics": {}, "error": str(e)}

    async def _invoke_llm(self, messages):
        """Invoke LLM asynchronously

        Parameters
        ----------
        messages : list
            List of messages to send to LLM

        Returns
        -------
        str
            LLM response content
        """
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content


class MockInternalComparisonTool:
    """Mock tool for comparing external and internal analysis

    IMPORTANT: This is a MOCK implementation!

    In production, this tool should be replaced with an actual API
    integration that fetches internal portfolio analysis from a real
    data source such as:
    - Bloomberg Terminal API
    - Internal portfolio management system (e.g., Aladdin, Charles River)
    - Custom internal analytics database
    - Risk management systems (RiskMetrics, MSCI)

    The mock returns simulated internal portfolio data for demonstration
    purposes. Real implementation would:
    1. Call external API: /internal-portfolio/analysis
    2. Fetch internal buy/sell calls, risk scores, asset allocations
    3. Retrieve internal conviction levels and position sizing
    4. Query performance metrics and attribution
    5. Return structured comparison data for external validation
    """

    def __init__(self, llm_service):
        """Initialize with LLM service

        Parameters
        ----------
        llm_service : LLMService
            LLM service for text generation

        Note
        ----
        In production, this would also include API credentials and endpoints
        for connecting to internal portfolio systems.
        """
        self.llm = llm_service
        self.definition = ToolDefinition(
            name="mock_internal_comparison",
            description=(
                "MOCK: Compare external research with internal "
                "portfolio analysis for validation. In production, "
                "fetches real internal portfolio data from "
                "Bloomberg/Aladdin/internal systems and validates "
                "external recommendations against internal views "
                "(buy/sell calls, allocations, risk scores, "
                "conviction levels)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "external": {
                        "type": "string",
                        "description": (
                            "External research analysis from sell-side report"
                        ),
                    },
                    "portfolio_id": {
                        "type": "string",
                        "description": (
                            "Portfolio ID for internal comparison "
                            "(mock: 'internal_portfolio')"
                        ),
                    },
                },
                "required": ["external"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "internal_data": {
                        "type": "object",
                        "description": (
                            "Simulated internal portfolio "
                            "(Real: API returns buy/sell calls, "
                            "allocations, risks)"
                        ),
                    },
                    "comparison": {
                        "type": "string",
                        "description": "Detailed comparison analysis",
                    },
                    "recommendations": {
                        "type": "array",
                        "description": (
                            "Recommended actions based on comparison "
                            "and validation"
                        ),
                    },
                    "divergences": {
                        "type": "array",
                        "description": (
                            "Key areas where external and "
                            "internal views diverge"
                        ),
                    },
                    "validation_score": {
                        "type": "number",
                        "description": (
                            "Confidence score (0-100) for recommendation "
                            "alignment with internal view"
                        ),
                    },
                },
                "required": ["comparison"],
            },
        )

    async def __call__(
        self, external: str, portfolio_id: str = "internal_portfolio"
    ) -> Dict[str, Any]:
        """Execute comparison with internal portfolio data

        Parameters
        ----------
        external : str
            External research analysis from sell-side report
        portfolio_id : str, optional
            Portfolio identifier for internal comparison,
            by default "internal_portfolio"

        Returns
        -------
        Dict[str, Any]
            Dictionary containing internal data, comparison,
            and validation results

        Note
        ----
        This mock returns simulated data. In production:
        1. Query internal API: GET /portfolio/{portfolio_id}/current-view
        2. Extract: buy_calls, sell_calls, asset_allocation, risk_scores
        3. Compare external recommendations against internal positioning
        4. Calculate alignment score
        5. Return structured comparison for portfolio manager validation
        """
        try:
            # MOCK: Simulated internal portfolio data
            # In production, this would be fetched from:
            # - Bloomberg Terminal API
            # - Aladdin (BlackRock)
            # - Charles River IMS
            # - Internal database
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
                    {
                        "asset": "Emerging Markets",
                        "conviction": "Medium",
                        "target_allocation": 7,
                        "current_allocation": 10,
                        "rationale": "Currency and political risks",
                    },
                ],
                "risk_scores": {
                    "market_risk": 6.5,
                    "credit_risk": 4.2,
                    "liquidity_risk": 3.1,
                    "geopolitical_risk": 5.8,
                },
                "conviction_levels": {
                    "macro_view": "Soft landing scenario",
                    "growth_outlook": "Modest (2-3%)",
                    "inflation_view": "Moderating to 2.5%",
                    "rate_view": "Peaked, cuts coming H2",
                },
            }

            # Generate comparison analysis with LLM
            # Use natural language prompt that mirrors report style
            prompt = (
                "You are a portfolio manager comparing external sell-side "
                "research with internal portfolio views.\n\n"
                "EXTERNAL RESEARCH:\n"
                f"{external}\n\n"
                "INTERNAL PORTFOLIO VIEW:\n"
                f"Current positioning is cautiously constructive, with focus "
                f"on quality and diversification.\n"
                f"Key positions: Technology moderately favored, Long-duration "
                f"bonds seen as limited opportunity, commodities seen as "
                f"potentially attractive given geopolitical dynamics.\n"
                f"Risk view: Market risks elevated but manageable, credit "
                f"risks contained, geopolitical tensions "
                f"warrant monitoring.\n\n"
                "COMPARISON (write naturally, as in conversation with PM):\n"
                "1. Are the external recommendations aligned with how you "
                "see the market?\n"
                "2. What areas stand out as notably different from your "
                "internal view?\n"
                "3. How would you characterize the overall confidence in "
                "following this recommendation?\n"
                "4. What concerns, if any, do you have?\n"
                "5. Any specific actions you'd want to take or monitor?\n\n"
                "Write naturally and conversationally (as if explaining to "
                "colleagues), avoiding excessive numbers and focus on the "
                "narrative reasoning."
            )

            messages = [
                SystemMessage(
                    content=(
                        "You are a seasoned portfolio manager discussing "
                        "external research. Write conversationally, focus on "
                        "reasoning rather than data, and be realistic about "
                        "confidence levels."
                    )
                ),
                HumanMessage(content=prompt),
            ]

            response = await self._invoke_llm(messages)
            logger.info(
                "Mock internal comparison: Completed validation analysis"
            )

            return {
                "internal_data": mock_internal_data,
                "comparison": response,
                "recommendations": [
                    "Consider the external recommendations as "
                    "validation of our tech positioning",
                    "Monitor how the energy thesis develops "
                    "in coming weeks",
                    "Use any pullbacks as potential entry " "opportunities",
                ],
                "divergences": [
                    "Their equity overweight is slightly more aggressive "
                    "than our current comfort level",
                    "We align on tech opportunity but differ slightly "
                    "on timing",
                    "Bond positioning is broadly aligned",
                ],
                "validation_score": 75,
                "data_source": "MOCK (Production: Bloomberg/Aladdin/API)",
                "note": (
                    "This is mock internal data for demonstration. "
                    "In production, this fetches real internal "
                    "portfolio data."
                ),
            }
        except Exception as e:
            logger.error(f"Mock internal comparison error: {e}")
            return {
                "comparison": "",
                "error": str(e),
                "data_source": "MOCK (Production: Bloomberg/Aladdin/API)",
            }

    async def _invoke_llm(self, messages):
        """Invoke LLM asynchronously

        Parameters
        ----------
        messages : list
            List of messages to send to LLM

        Returns
        -------
        str
            LLM response content
        """
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content


class SynthesisTool:
    """Tool for synthesizing final responses"""

    def __init__(self, llm_service):
        """Initialize with LLM service

        Parameters
        ----------
        llm_service : LLMService
            LLM service for text generation
        """
        self.llm = llm_service
        self.definition = ToolDefinition(
            name="synthesize_response",
            description=(
                "Generate final answer by "
                "synthesizing all collected "
                "information"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Original user query",
                    },
                    "context": {
                        "type": "object",
                        "description": (
                            "Collected information "
                            "(summary, data, comparison)"
                        ),
                    },
                },
                "required": ["query", "context"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": ("Final answer to user query"),
                    }
                },
                "required": ["response"],
            },
        )

    async def __call__(
        self, query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute synthesis

        Parameters
        ----------
        query : str
            User query
        context : Dict[str, Any]
            Collected context information

        Returns
        -------
        Dict[str, Any]
            Dictionary containing final response
        """
        try:
            # Build context in natural, narrative format
            # Focus on story and reasoning, not raw data dumps
            summary = context.get("summary", "")
            comparison = context.get("comparison_result", "")

            prompt = (
                "Based on the sell-side research analysis below, "
                "provide a clear, conversational answer to the user's "
                "question.\n\n"
                f"USER QUESTION: {query}\n\n"
                f"RESEARCH SUMMARY:\n{summary}\n\n"
            )

            if comparison:
                prompt += f"INTERNAL VALIDATION:\n{comparison}\n\n"

            prompt += (
                "Provide a natural, conversational response that:\n"
                "1. Directly answers the user's question\n"
                "2. Focuses on reasoning and narrative, not data dumps\n"
                "3. Includes key insights (if any numbers are important, "
                "weave them naturally)\n"
                "4. Avoids overwhelming with statistics or bullet points\n"
                "5. Explains what this means practically for the portfolio"
            )

            messages = [
                SystemMessage(
                    content=(
                        "You are a portfolio manager discussing sell-side "
                        "research with colleagues. Be conversational, focus "
                        "on reasoning and implications, avoid data overload."
                    )
                ),
                HumanMessage(content=prompt),
            ]

            response = await self._invoke_llm(messages)
            logger.info("Synthesis tool: Generated response")

            return {"response": response}
        except Exception as e:
            logger.error(f"Synthesis tool error: {e}")
            return {"response": "", "error": str(e)}

    async def _invoke_llm(self, messages):
        """Invoke LLM asynchronously

        Parameters
        ----------
        messages : list
            List of messages to send to LLM

        Returns
        -------
        str
            LLM response content
        """
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content


class AgentToolRegistry:
    """Central registry of all available tools"""

    def __init__(
        self,
        db_service,
        llm_service,
    ):
        """Initialize tool registry

        Parameters
        ----------
        db_service : DatabaseService
            Database service for vector search and metadata retrieval
        llm_service : LLMService
            LLM service (ChatOpenAI instance) for reasoning and extraction
        """
        # Initialize all tools
        self.search = SearchTool(db_service)
        self.summarize = SummarizationTool(llm_service)
        self.extract = FinancialExtractionTool(llm_service)
        self.mock_comparison = MockInternalComparisonTool(llm_service)
        self.synthesize = SynthesisTool(llm_service)

        # Tool mapping
        self.tools_map = {
            "search_documents": self.search,
            "summarize_content": self.summarize,
            "extract_financial_data": self.extract,
            "mock_internal_comparison": self.mock_comparison,
            "synthesize_response": self.synthesize,
        }

        logger.info(
            f"Tool registry initialized with {len(self.tools_map)} tools"
        )

    def get_tool(self, name: str) -> Optional[Any]:
        """Get tool by name

        Parameters
        ----------
        name : str
            Tool name

        Returns
        -------
        Optional[Any]
            Tool instance if found, None otherwise
        """
        return self.tools_map.get(name)

    def get_all_tools(self) -> Dict[str, Any]:
        """Get all tools

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping tool names to tool instances
        """
        return self.tools_map

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions for agent documentation

        Returns
        -------
        List[Dict[str, Any]]
            List of tool definition dictionaries
        """
        return [tool.definition.to_dict() for tool in self.tools_map.values()]

    def list_tools(self) -> List[str]:
        """List all available tool names

        Returns
        -------
        List[str]
            List of available tool names
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
            Tool result dictionary
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

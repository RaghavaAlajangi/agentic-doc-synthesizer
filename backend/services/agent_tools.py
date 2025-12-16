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

        Args:
            name: Unique tool identifier
            description: Human-readable description
            input_schema: Input parameters schema
            output_schema: Output format schema
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema

    def to_dict(self) -> Dict[str, Any]:
        """Export tool definition as dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


class SearchTool:
    """Tool for searching research documents"""

    def __init__(self, db_service):
        """Initialize with database service"""
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

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            Search results
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
        """Initialize with LLM service"""
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

        Args:
            content: Content to summarize

        Returns:
            Summary result
        """
        try:
            prompt = (
                "You are a financial research analyst. "
                "Summarize the key findings, "
                "recommendations, and investment "
                "thesis from this research.\n\n"
                "Focus on:\n"
                "1. Main thesis and key arguments\n"
                "2. Target asset classes and "
                "recommendations\n"
                "3. Key metrics and data points\n"
                "4. Risk factors mentioned\n"
                "5. Investment implications\n\n"
                f"Content:\n{content}\n\n"
                "Provide a comprehensive summary."
            )

            messages = [
                SystemMessage(content="You are an expert financial analyst."),
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
        """Invoke LLM asynchronously"""
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content


class FinancialExtractionTool:
    """Tool for extracting financial metrics and data"""

    def __init__(self, llm_service):
        """Initialize with LLM service"""
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

        Args:
            content: Content to extract from

        Returns:
            Extracted data
        """
        try:
            prompt = (
                "Extract all key financial metrics, "
                "statements, and data points.\n\n"
                "Focus on:\n"
                "1. P/E ratios, valuation multiples\n"
                "2. Growth rates (earnings, revenue)\n"
                "3. Dividend yields\n"
                "4. Credit metrics (for fixed income)\n"
                "5. Asset allocations\n"
                "6. Price targets and return "
                "expectations\n"
                "7. Key assumptions\n\n"
                f"Content:\n{content}\n\n"
                "Format as structured list."
            )

            messages = [
                SystemMessage(
                    content="You are a financial data extraction specialist."
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
        """Invoke LLM asynchronously"""
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content


class ComparisonTool:
    """Tool for comparing external and internal analysis"""

    def __init__(self, llm_service):
        """Initialize with LLM service"""
        self.llm = llm_service
        self.definition = ToolDefinition(
            name="compare_analyses",
            description=(
                "Compare external research "
                "recommendations with internal "
                "investment views"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "external": {
                        "type": "string",
                        "description": ("External research analysis"),
                    },
                    "internal": {
                        "type": "string",
                        "description": ("Internal investment view"),
                    },
                },
                "required": ["external", "internal"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "comparison": {
                        "type": "string",
                        "description": "Detailed comparison",
                    },
                    "agreements": {
                        "type": "array",
                        "description": "Points of agreement",
                    },
                    "disagreements": {
                        "type": "array",
                        "description": ("Points of disagreement"),
                    },
                },
                "required": ["comparison"],
            },
        )

    async def __call__(self, external: str, internal: str) -> Dict[str, Any]:
        """Execute comparison

        Args:
            external: External analysis
            internal: Internal view

        Returns:
            Comparison result
        """
        try:
            prompt = (
                "Compare external research with "
                "internal views.\n\n"
                f"EXTERNAL RESEARCH:\n{external}\n\n"
                f"INTERNAL VIEW:\n{internal}\n\n"
                "Provide:\n"
                "1. Areas of agreement\n"
                "2. Key differences\n"
                "3. Risk factors each emphasizes\n"
                "4. Recommended action"
            )

            messages = [
                SystemMessage(
                    content="You are a portfolio manager comparing analyses."
                ),
                HumanMessage(content=prompt),
            ]

            response = await self._invoke_llm(messages)
            logger.info("Comparison tool: Completed comparison")

            return {
                "comparison": response,
                "agreements": [],
                "disagreements": [],
            }
        except Exception as e:
            logger.error(f"Comparison tool error: {e}")
            return {"comparison": "", "error": str(e)}

    async def _invoke_llm(self, messages):
        """Invoke LLM asynchronously"""
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content


class SynthesisTool:
    """Tool for synthesizing final responses"""

    def __init__(self, llm_service):
        """Initialize with LLM service"""
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

        Args:
            query: User query
            context: Collected context

        Returns:
            Final response
        """
        try:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])

            prompt = (
                "Based on the analysis below, "
                "provide a concise answer to the "
                "user's query.\n\n"
                f"USER QUERY: {query}\n\n"
                f"CONTEXT:\n{context_str}\n\n"
                "Provide a clear, actionable answer "
                "with specific data points."
            )

            messages = [
                SystemMessage(
                    content="You are a financial analyst providing expert "
                    "insights."
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
        """Invoke LLM asynchronously"""
        response = await asyncio.to_thread(self.llm.invoke, messages)
        return response.content


class AgentToolRegistry:
    """Central registry of all available tools"""

    def __init__(
        self,
        db_service,
        financial_data_service,
        llm_service,
    ):
        """Initialize tool registry

        Args:
            db_service: Database service
            financial_data_service: Financial data service
            llm_service: LLM service (ChatOpenAI instance)
        """
        # Initialize all tools
        self.search = SearchTool(db_service)
        self.summarize = SummarizationTool(llm_service)
        self.extract = FinancialExtractionTool(llm_service)
        self.compare = ComparisonTool(llm_service)
        self.synthesize = SynthesisTool(llm_service)

        # Tool mapping
        self.tools_map = {
            "search_documents": self.search,
            "summarize_content": self.summarize,
            "extract_financial_data": self.extract,
            "compare_analyses": self.compare,
            "synthesize_response": self.synthesize,
        }

        logger.info(
            f"Tool registry initialized with {len(self.tools_map)} tools"
        )

    def get_tool(self, name: str) -> Optional[Any]:
        """Get tool by name"""
        return self.tools_map.get(name)

    def get_all_tools(self) -> Dict[str, Any]:
        """Get all tools"""
        return self.tools_map

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions for agent
        documentation"""
        return [tool.definition.to_dict() for tool in self.tools_map.values()]

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools_map.keys())

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name

        Args:
            tool_name: Name of the tool
            **kwargs: Tool arguments

        Returns:
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

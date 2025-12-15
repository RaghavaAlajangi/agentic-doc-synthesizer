from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RoleEnum(str, Enum):
    """Enum for message roles"""

    USER = "user"
    ASSISTANT = "assistant"
    AGENT = "agent"
    SYSTEM = "system"


class AgentThought(BaseModel):
    """Represents a single thought/step from an agent"""

    agent_name: str = Field(
        ..., description="Name of the agent generating this thought"
    )
    thought: str = Field(..., description="The thought or reasoning")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool_used: Optional[str] = Field(
        None, description="Tool called by the agent, if any"
    )
    tool_input: Optional[Dict[str, Any]] = Field(
        None, description="Input to the tool"
    )
    tool_output: Optional[str] = Field(
        None, description="Output from the tool"
    )


class ChatMessage(BaseModel):
    """Represents a single chat message"""

    role: RoleEnum = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_thoughts: List[AgentThought] = Field(
        default_factory=list, description="Agent thoughts if role is agent"
    )


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""

    query: str = Field(..., description="User query or question")
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for tracking multi-turn "
        "conversations",
    )


class SearchResult(BaseModel):
    """Represents a search result from vector database"""

    document_id: str = Field(..., description="ID of the document")
    content: str = Field(..., description="Retrieved content")
    similarity_score: float = Field(..., description="Similarity score")
    source: str = Field(..., description="Source - 'internal' or 'external'")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata like filename, page number, etc.",
    )


class Recommendation(BaseModel):
    """Represents a recommendation extracted from research"""

    asset_class: str = Field(
        ...,
        description="Asset class (equity, fixed_income, multi_asset, etc.)",
    )
    recommendation: str = Field(..., description="The recommendation text")
    confidence: float = Field(..., description="Confidence score 0-1")
    source_document: str = Field(..., description="Source document filename")
    source_type: str = Field(
        ..., description="Source type - 'internal' or 'external'"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""

    response: str = Field(..., description="Final response from the assistant")
    agent_thoughts: List[AgentThought] = Field(
        default_factory=list,
        description="List of agent thoughts during processing",
    )
    search_results: List[SearchResult] = Field(
        default_factory=list,
        description="Search results used to formulate the response",
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="Extracted recommendations from the sources",
    )
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for future reference"
    )


class DocumentMetadata(BaseModel):
    """Metadata for uploaded document"""

    filename: str = Field(..., description="Name of the uploaded file")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    document_type: str = Field(
        ..., description="Type of document - 'research_report' or similar"
    )
    source_type: str = Field(
        ..., description="Source type - 'internal' or 'external'"
    )
    file_size: int = Field(..., description="Size of the file in bytes")


class DocumentUploadResponse(BaseModel):
    """Response for document upload"""

    success: bool = Field(..., description="Whether upload was successful")
    document_id: str = Field(..., description="ID of the uploaded document")
    message: str = Field(..., description="Status message")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    chunks_stored: int = Field(
        ..., description="Number of chunks stored in vector DB"
    )


class StreamEvent(BaseModel):
    """Event sent during streaming response"""

    event_type: str = Field(
        ...,
        description="Type of event: 'agent_thought', 'search_result', "
        "'recommendation', 'final_response'",
    )
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

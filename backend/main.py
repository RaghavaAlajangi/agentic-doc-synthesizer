import json
import logging
import uuid

import uvicorn
from config.settings import settings
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentMetadata,
    DocumentUploadResponse,
    StreamEvent,
)
from services.agent_orchestrator import SmartAgentOrchestrator
from services.database import ChromaDBService
from services.document_processor import DocumentProcessor
from services.financial_data import FinancialDataService

# Setup logging
logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=("Multi-agent AI assistant for research report analysis"),
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
db_service: ChromaDBService = None
doc_processor: DocumentProcessor = None
orchestrator: SmartAgentOrchestrator = None
financial_data_service: FinancialDataService = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize services on application startup.

    Initializes all global service instances including database,
    document processor, financial data service, and the multi-agent
    orchestrator. Called automatically when the FastAPI application
    starts.

    Raises
    ------
    Exception
        If any service fails to initialize, the error is logged and
        re-raised.
    """
    global db_service, doc_processor, orchestrator, financial_data_service
    try:
        logger.info("Initializing services...")

        db_service = ChromaDBService(
            external_host=settings.external_chroma_host,
            external_port=settings.external_chroma_port,
        )
        financial_data_service = FinancialDataService()
        doc_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            temperature=settings.document_processor_temperature,
        )
        orchestrator = SmartAgentOrchestrator(
            db_service, financial_data_service
        )
        logger.info("Services initialized with Smart Orchestrator")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise


@app.get("/")
async def root():
    """
    Root API endpoint.

    Returns information about the API including title and version.

    Returns
    -------
    dict
        Dictionary containing API title and version information.
    """
    return {
        "message": settings.api_title,
        "version": settings.api_version,
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for service status.

    Checks the health status of connected databases and services.

    Returns
    -------
    dict
        Dictionary with status ("healthy" or "unhealthy") and database
        health information or error message.
    """
    try:
        health = await db_service.health_check()
        return {"status": "healthy", "databases": health}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a research report document.

    Accepts PDF or TXT files up to 10MB, processes them into chunks,
    and stores them in the vector database for semantic search.

    Parameters
    ----------
    file : UploadFile
        The document file to upload (PDF or TXT format).

    Returns
    -------
    DocumentUploadResponse
        Response containing document ID, filename, file size,
        chunk count, and metadata.

    Raises
    ------
    HTTPException
        If file format is invalid, size exceeds 10MB, or processing
        fails.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Validate file type
        if not file.filename.lower().endswith((".pdf", ".txt")):
            raise HTTPException(
                status_code=400,
                detail=("Only PDF and TXT files are supported"),
            )

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Validate file size (max 10MB)
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413, detail="File size exceeds 10MB limit"
            )

        # Generate document ID
        document_id = str(uuid.uuid4())

        # Process document
        if file.filename.lower().endswith(".pdf"):
            chunks = await doc_processor.process_pdf(content, file.filename)
        else:
            text_content = content.decode("utf-8")
            chunks = await doc_processor.process_text_file(
                text_content, file.filename
            )

        # Store chunks in vector DB
        chunk_count = 0
        for chunk in chunks:
            await db_service.store_document_chunk(
                content=chunk["content"],
                document_id=document_id,
                filename=file.filename,
                chunk_id=chunk["chunk_id"],
                metadata=chunk.get("metadata", {}),
            )
            chunk_count += 1

        logger.info(
            f"Stored document {document_id} with " f"{chunk_count} chunks"
        )

        metadata = DocumentMetadata(
            filename=file.filename,
            document_type="research_report",
            source_type="external",
            file_size=file_size,
        )

        return DocumentUploadResponse(
            success=True,
            document_id=document_id,
            message=(f"✅ Successfully uploaded: {file.filename}"),
            metadata=metadata,
            chunks_stored=chunk_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Process chat query with agent-driven routing.

    Routes the query through the multi-agent orchestrator which
    determines query intent and executes appropriate agents for
    analysis, comparison, or lookup operations.

    Parameters
    ----------
    request : ChatRequest
        Request object containing the query string and optional
        conversation ID.

    Returns
    -------
    ChatResponse
        Response containing synthesized answer, agent thoughts,
        search results, and recommendations.

    Raises
    ------
    HTTPException
        If query processing fails.
    """
    try:
        response = await orchestrator.process_query(query=request.query)

        return ChatResponse(
            response=response["response"],
            agent_thoughts=response["agent_thoughts"],
            search_results=response["search_results"],
            recommendations=response["recommendations"],
            conversation_id=request.conversation_id,
        )

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}"
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response with agent thoughts via Server-Sent Events.

    Streams individual agent thoughts and events as they occur during
    query processing, enabling real-time progress feedback to the
    client.

    Parameters
    ----------
    request : ChatRequest
        Request object containing the query string.

    Yields
    ------
    str
        Server-Sent Event formatted strings with streaming updates.
    """

    async def event_generator():
        try:
            async for event in orchestrator.stream_query(
                query=request.query,
            ):
                # Convert event to SSE format
                event_json = json.dumps(event.dict(), default=str)
                yield f"data: {event_json}\n\n"
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            error_event = StreamEvent(
                event_type="error", data={"error": str(e)}
            )
            error_json = json.dumps(error_event.dict(), default=str)
            yield f"data: {error_json}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/documents")
async def list_documents(source_type: str = None):
    """
    List all stored documents with optional filtering.

    Retrieves metadata for all uploaded documents with optional
    filtering by source type.

    Parameters
    ----------
    source_type : str, optional
        Filter documents by source type ('external' or 'internal').
        If None, returns all documents.

    Returns
    -------
    dict
        Dictionary containing total count and list of documents with
        metadata.

    Raises
    ------
    HTTPException
        If retrieval fails.
    """
    try:
        documents = await db_service.get_all_documents(source_type=source_type)
        return {"total": len(documents), "documents": documents}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, source_type: str = "external"):
    """
    Delete a document and all its associated chunks.

    Removes a document and all its vector embeddings from the
    database.

    Parameters
    ----------
    document_id : str
        Unique identifier of the document to delete.
    source_type : str, optional
        Type of source ('external' or 'internal'). Default is
        'external'.

    Returns
    -------
    dict
        Success message.

    Raises
    ------
    HTTPException
        If document not found or deletion fails.
    """
    try:
        success = await db_service.delete_document(
            document_id=document_id, source_type=source_type
        )
        if success:
            return {"message": "Document deleted"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

import io
import json
import logging
from typing import Any, Dict, List

from config.settings import settings
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing and chunking PDF documents with semantic
    understanding"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        openai_api_key: str = None,
        openai_model: str = None,
        temperature: float = None,
    ):
        """Initialize document processor

        Parameters
        ----------
        chunk_size : int, optional
            Size of each chunk in characters
        chunk_overlap : int, optional
            Overlap between chunks in characters
        openai_api_key : str, optional
            OpenAI API key for LLM
        openai_model : str, optional
            OpenAI model to use for summarization
        temperature : float, optional
            Temperature for LLM responses
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.openai_api_key = openai_api_key or settings.openai_api_key
        self.openai_model = openai_model or settings.openai_model
        self.temperature = temperature or (
            settings.document_processor_temperature
        )

        # Initialize LLM for summarization and extraction
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                model=self.openai_model,
                temperature=self.temperature,
                openai_api_key=self.openai_api_key,
            )
        else:
            self.llm = None
            logger.warning(
                "OpenAI API key not provided. "
                "Summarization will be skipped."
            )

    async def process_pdf(
        self, file_content: bytes, filename: str
    ) -> List[Dict[str, Any]]:
        """Process PDF: extract text, create chunks, summarize

        Parameters
        ----------
        file_content : bytes
            Raw PDF file content (bytes)
        filename : str
            Original filename

        Returns
        -------
        List[Dict[str, Any]]
            List of chunks with metadata, summaries, and extracted information
        """
        try:
            # Extract text from PDF
            text = await self._extract_text_from_pdf(file_content)

            if not text.strip():
                logger.warning(f"No text extracted from {filename}")
                return []

            # Create semantic chunks
            chunks = await self._create_semantic_chunks(text)

            # For each chunk: summarize and extract
            # vital information
            chunks_with_metadata = []
            for idx, chunk in enumerate(chunks):
                summary = ""
                vital_info = {}

                # If LLM available, summarize and extract
                if self.llm:
                    summary = await self._summarize_chunk(chunk)
                    vital_info = await self._extract_vital_info(chunk, summary)

                chunks_with_metadata.append(
                    {
                        "content": chunk,
                        "chunk_id": idx,
                        "filename": filename,
                        "source_type": "external",
                        "summary": summary,
                        "vital_info": vital_info,
                        "metadata": {
                            "page_range": (
                                self._estimate_page_range(chunk, text)
                            )
                        },
                    }
                )

            # Create document-level summary
            if self.llm and chunks_with_metadata:
                summaries = [
                    c.get("summary", c["content"][:500])
                    for c in chunks_with_metadata
                ]
                doc_summary = await self._summarize_document(summaries)
                # Store at document level (can be returned separately)
                logger.info(
                    f"Document-level summary created " f"for {filename}"
                )

            logger.info(
                f"Processed {filename}: "
                f"created {len(chunks_with_metadata)} "
                f"semantic chunks"
            )
            return chunks_with_metadata

        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}")
            raise

    async def _create_semantic_chunks(
        self, text: str, max_chunks: int = 5
    ) -> List[str]:
        """Create semantic chunks using paragraph detection

        Parameters
        ----------
        text : str
            Full document text
        max_chunks : int, optional
            Maximum number of chunks to create, by default 5

        Returns
        -------
        List[str]
            List of semantic chunks
        """
        # Split by paragraphs (double newline)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            # Fallback to sentence-based chunking
            return self._create_chunks(text)

        # Group paragraphs into chunks based on size
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Ensure we don't have too many chunks
        if len(chunks) > max_chunks:
            logger.info(
                f"Document has {len(chunks)} semantic chunks, "
                f"limiting to {max_chunks}"
            )
            # Merge excess chunks
            merged_chunks = []
            chunks_per_group = len(chunks) // max_chunks
            for i in range(0, len(chunks), chunks_per_group):
                group = chunks[i : i + chunks_per_group]
                merged_chunks.append("\n\n".join(group))
            return merged_chunks[:max_chunks]

        return chunks

    async def _summarize_chunk(self, chunk: str) -> str:
        """Summarize a single chunk using LLM

        Parameters
        ----------
        chunk : str
            Text chunk to summarize

        Returns
        -------
        str
            Summary of the chunk
        """
        if not self.llm:
            return chunk[:200] + "..." if len(chunk) > 200 else chunk

        try:
            system_prompt = (
                "You are a financial analyst. "
                "Summarize the provided text in 3-4 sentences, "
                "focusing on key investment insights, "
                "recommendations, or financial metrics."
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=chunk),
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error summarizing chunk: {e}")
            return chunk[:200] + "..." if len(chunk) > 200 else chunk

    async def _extract_vital_info(
        self, chunk: str, summary: str
    ) -> Dict[str, Any]:
        """Extract vital investment information from chunk

        Parameters
        ----------
        chunk : str
            Original text chunk
        summary : str
            Chunk summary

        Returns
        -------
        Dict[str, Any]
            Dictionary with extracted info: sectors, recommendations, metrics, risks
        """
        if not self.llm:
            return {}

        try:
            system_prompt = (
                "Extract vital investment information from "
                "the text. Return JSON with:\n"
                '{"sectors": [list of sectors mentioned], '
                '"recommendations": '
                "[list of buy/hold/sell recommendations], "
                '"metrics": '
                "{key financial metrics and values}, "
                '"risks": [list of identified risks]}\n'
                "If any field is not found, leave empty list/dict."
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Chunk:\n{chunk}"),
            ]
            response = self.llm.invoke(messages)
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {}
        except Exception as e:
            logger.error(f"Error extracting vital info: {e}")
            return {}

    async def _summarize_document(self, chunk_summaries: List[str]) -> str:
        """Create a document-level summary from chunk summaries

        Parameters
        ----------
        chunk_summaries : List[str]
            List of individual chunk summaries

        Returns
        -------
        str
            Document-level summary
        """
        if not self.llm or not chunk_summaries:
            return ""

        try:
            combined = "\n\n".join(chunk_summaries)
            system_prompt = (
                "You are a financial analyst. "
                "Create a concise executive summary "
                "(3-5 sentences) of the document based on "
                "the provided chunk summaries. Focus on key "
                "investment opportunities and risks."
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=combined),
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error creating document summary: {e}")
            return ""

    async def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF content

        Parameters
        ----------
        file_content : bytes
            Raw PDF bytes

        Returns
        -------
        str
            Extracted text
        """
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_file)

            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()

            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks

        Parameters
        ----------
        text : str
            Full text to chunk

        Returns
        -------
        List[str]
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            # Get chunk
            end = start + self.chunk_size
            chunk = text[start:end]

            # Don't cut off mid-word
            if end < len(text):
                last_space = chunk.rfind(" ")
                if last_space > 0:
                    end = start + last_space
                    chunk = text[start:end].strip()

            if chunk.strip():
                chunks.append(chunk)

            # Move start with overlap
            start = end - self.chunk_overlap

        return chunks

    def _estimate_page_range(self, chunk: str, full_text: str) -> str:
        """Estimate page range for a chunk

        Parameters
        ----------
        chunk : str
            The text chunk
        full_text : str
            Full document text with page markers

        Returns
        -------
        str
            Estimated page range
        """
        try:
            chunk_start = full_text.find(chunk)
            if chunk_start == -1:
                return "unknown"

            # Count page markers before chunk
            page_markers = full_text[:chunk_start].count("--- Page")
            start_page = page_markers + 1

            # Count page markers in chunk
            page_markers_in_chunk = chunk.count("--- Page")
            end_page = start_page + page_markers_in_chunk

            return f"{start_page}-{end_page}"
        except Exception:
            return "unknown"

    async def process_text_file(
        self, content: str, filename: str
    ) -> List[Dict[str, Any]]:
        """Process plain text file

        Parameters
        ----------
        content : str
            Text file content
        filename : str
            Original filename

        Returns
        -------
        List[Dict[str, Any]]
            List of chunks with metadata
        """
        try:
            chunks = self._create_chunks(content)

            chunks_with_metadata = [
                {
                    "content": chunk,
                    "chunk_id": idx,
                    "filename": filename,
                    "source_type": "external",
                    "metadata": {},
                }
                for idx, chunk in enumerate(chunks)
            ]

            logger.info(
                f"Processed {filename}: " f"extracted {len(chunks)} chunks"
            )
            return chunks_with_metadata

        except Exception as e:
            logger.error(f"Error processing text file {filename}: {e}")
            raise

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DocumentSummaryModel(Base):
    """
    SQLite model for storing document summaries.

    Stores document-level executive summaries with metadata for quick
    retrieval by RAG agents during query processing. Keeps vector DB
    clean by separating document summaries from chunk embeddings.

    Parameters
    ----------
    id : int
        Primary key, auto-incrementing.
    document_id : str
        Unique identifier matching the document in vector DB.
    filename : str
        Original filename of the uploaded document.
    summary_text : str
        Document-level executive summary (multi-paragraph).
    created_at : datetime
        Timestamp when document was uploaded.
    updated_at : datetime
        Timestamp of last update.
    chunk_count : int
        Number of chunks created from this document.
    file_size : int
        File size in bytes.
    source_type : str
        Source type (e.g., 'external', 'internal').
    """

    __tablename__ = "document_summaries"

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Document Reference
    document_id = Column(String(255), unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)

    # Summary Content
    summary_text = Column(Text, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Document Statistics
    chunk_count = Column(Integer, default=0)
    file_size = Column(Integer, default=0)
    source_type = Column(String(50), default="external", index=True)

    def __repr__(self):
        return (
            f"<DocumentSummary(document_id={self.document_id}, "
            f"filename={self.filename})>"
        )

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "filename": self.filename,
            "summary_text": self.summary_text,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "chunk_count": self.chunk_count,
            "file_size": self.file_size,
            "source_type": self.source_type,
        }

"""
SQLite service for document summaries.

Manages storage and retrieval of document-level summaries in SQLite
database, keeping the vector database clean and focused on chunk
embeddings and semantic search.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from models.database_models import Base, DocumentSummaryModel
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


class SQLiteService:
    """
    SQLite service for document summary storage and retrieval.

    Manages persistent storage of document-level summaries with
    metadata in SQLite, enabling efficient retrieval during RAG
    operations without querying the vector database.

    Parameters
    ----------
    db_path : Path
        Path to SQLite database file.
    engine : Engine
        SQLAlchemy database engine.
    SessionLocal : sessionmaker
        SQLAlchemy session factory.
    """

    def __init__(self, db_path: str = "data/summaries.db"):
        """
        Initialize SQLite service.

        Parameters
        ----------
        db_path : str, optional
            Path to SQLite database file. Defaults to 'data/summaries.db'.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create database engine
        database_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(
            database_url, connect_args={"check_same_thread": False}
        )

        # Create all tables
        Base.metadata.create_all(bind=self.engine)

        # Session factory
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        logger.info(f"SQLite service initialized at {self.db_path}")

    def get_session(self) -> Session:
        """
        Get database session.

        Returns
        -------
        Session
            SQLAlchemy session for database operations.
        """
        return self.SessionLocal()

    async def store_summary(
        self,
        document_id: str,
        filename: str,
        summary_text: str,
        chunk_count: int = 0,
        file_size: int = 0,
        source_type: str = "external",
    ) -> DocumentSummaryModel:
        """
        Store document summary in SQLite.

        Parameters
        ----------
        document_id : str
            Unique identifier for the document.
        filename : str
            Original filename.
        summary_text : str
            Document-level executive summary.
        chunk_count : int, optional
            Number of chunks created from the document.
        file_size : int, optional
            File size in bytes.
        source_type : str, optional
            Source type (default: 'external').

        Returns
        -------
        DocumentSummaryModel
            Stored summary model instance.
        """
        session = self.get_session()
        try:
            # Check if summary already exists
            existing = (
                session.query(DocumentSummaryModel)
                .filter(DocumentSummaryModel.document_id == document_id)
                .first()
            )

            if existing:
                logger.warning(
                    f"Summary for document {document_id} already exists, "
                    "updating..."
                )
                existing.summary_text = summary_text
                existing.chunk_count = chunk_count
                existing.file_size = file_size
                existing.updated_at = datetime.utcnow()
                session.commit()
                logger.info(f"Updated summary for document {document_id}")
                return existing

            # Create new summary
            summary_model = DocumentSummaryModel(
                document_id=document_id,
                filename=filename,
                summary_text=summary_text,
                chunk_count=chunk_count,
                file_size=file_size,
                source_type=source_type,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            session.add(summary_model)
            session.commit()
            session.refresh(summary_model)

            logger.info(
                f"Stored summary for document {document_id} "
                f"({filename}) with {chunk_count} chunks"
            )
            return summary_model

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing summary: {e}")
            raise
        finally:
            session.close()

    async def get_summary(
        self, document_id: str
    ) -> Optional[DocumentSummaryModel]:
        """
        Retrieve document summary by document_id.

        Parameters
        ----------
        document_id : str
            Unique identifier for the document.

        Returns
        -------
        Optional[DocumentSummaryModel]
            Summary model if found, None otherwise.
        """
        session = self.get_session()
        try:
            summary = (
                session.query(DocumentSummaryModel)
                .filter(DocumentSummaryModel.document_id == document_id)
                .first()
            )
            return summary
        except Exception as e:
            logger.error(f"Error retrieving summary: {e}")
            return None
        finally:
            session.close()

    async def get_summaries_by_source(
        self, source_type: str = "external"
    ) -> List[DocumentSummaryModel]:
        """
        Retrieve all summaries of a given source type.

        Parameters
        ----------
        source_type : str, optional
            Source type to filter by (default: 'external').

        Returns
        -------
        List[DocumentSummaryModel]
            List of summary models.
        """
        session = self.get_session()
        try:
            summaries = (
                session.query(DocumentSummaryModel)
                .filter(DocumentSummaryModel.source_type == source_type)
                .all()
            )
            return summaries
        except Exception as e:
            logger.error(f"Error retrieving summaries: {e}")
            return []
        finally:
            session.close()

    async def get_all_summaries(self) -> List[DocumentSummaryModel]:
        """
        Retrieve all stored summaries.

        Returns
        -------
        List[DocumentSummaryModel]
            List of all stored summaries.
        """
        session = self.get_session()
        try:
            summaries = session.query(DocumentSummaryModel).all()
            return summaries
        except Exception as e:
            logger.error(f"Error retrieving all summaries: {e}")
            return []
        finally:
            session.close()

    async def delete_summary(self, document_id: str) -> bool:
        """
        Delete a summary by document_id.

        Parameters
        ----------
        document_id : str
            Unique identifier for the document.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        session = self.get_session()
        try:
            summary = (
                session.query(DocumentSummaryModel)
                .filter(DocumentSummaryModel.document_id == document_id)
                .first()
            )

            if not summary:
                logger.warning(f"Summary for document {document_id} not found")
                return False

            session.delete(summary)
            session.commit()
            logger.info(f"Deleted summary for document {document_id}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting summary: {e}")
            return False
        finally:
            session.close()

    async def update_summary(
        self,
        document_id: str,
        **kwargs,
    ) -> Optional[DocumentSummaryModel]:
        """
        Update summary fields.

        Parameters
        ----------
        document_id : str
            Unique identifier for the document.
        **kwargs
            Fields to update (e.g., chunk_count, file_size).

        Returns
        -------
        Optional[DocumentSummaryModel]
            Updated summary model if found, None otherwise.
        """
        session = self.get_session()
        try:
            summary = (
                session.query(DocumentSummaryModel)
                .filter(DocumentSummaryModel.document_id == document_id)
                .first()
            )

            if not summary:
                logger.warning(f"Summary for document {document_id} not found")
                return None

            # Update fields
            for key, value in kwargs.items():
                if hasattr(summary, key):
                    setattr(summary, key, value)

            summary.updated_at = datetime.utcnow()
            session.commit()
            logger.info(
                f"Updated summary for document {document_id}: {kwargs}"
            )
            return summary

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating summary: {e}")
            return None
        finally:
            session.close()

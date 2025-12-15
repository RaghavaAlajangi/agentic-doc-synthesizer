"""
SQLite service for storing extracted financial data and metadata from research reports.

This service handles:
- Summary storage (document-level summaries)
- Financial statements storage (extracted key financial data)
- Metadata management for efficient retrieval
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FinancialDataService:
    """Service for managing financial data extraction and storage"""

    def __init__(self, db_path: str = "financial_data.db"):
        """
        Initialize Financial Data Service

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()

            # Create tables
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS document_summaries (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    asset_classes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS financial_statements (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    statement_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    asset_class TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES document_summaries(document_id)
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS key_recommendations (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    asset_class TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    confidence REAL,
                    data_sources TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES document_summaries(document_id)
                )
                """
            )

            self.conn.commit()
            logger.info(
                f"Financial data database initialized at {self.db_path}"
            )
        except Exception as e:
            logger.error(f"Error initializing financial data database: {e}")
            raise

    def store_summary(
        self,
        summary_id: str,
        document_id: str,
        filename: str,
        summary: str,
        asset_classes: Optional[List[str]] = None,
    ) -> bool:
        """
        Store document summary

        Args:
            summary_id: Unique ID for the summary
            document_id: ID of the source document
            filename: Original filename
            summary: The summary text
            asset_classes: List of asset classes covered

        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            asset_classes_str = (
                json.dumps(asset_classes) if asset_classes else None
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO document_summaries
                (id, document_id, filename, summary, asset_classes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    summary_id,
                    document_id,
                    filename,
                    summary,
                    asset_classes_str,
                    datetime.utcnow().isoformat(),
                ),
            )

            self.conn.commit()
            logger.info(f"Stored summary {summary_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing summary: {e}")
            return False

    def store_financial_statement(
        self,
        statement_id: str,
        document_id: str,
        filename: str,
        statement_type: str,
        content: str,
        asset_class: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store extracted financial statement/data

        Args:
            statement_id: Unique ID for the statement
            document_id: ID of the source document
            filename: Original filename
            statement_type: Type like 'earnings', 'valuation', 'growth_metrics'
            content: The statement content (can be JSON or text)
            asset_class: Asset class (equity, fixed_income, etc.)
            metadata: Additional metadata

        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            metadata_str = json.dumps(metadata) if metadata else None

            cursor.execute(
                """
                INSERT OR REPLACE INTO financial_statements
                (id, document_id, filename, statement_type, content, asset_class, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    statement_id,
                    document_id,
                    filename,
                    statement_type,
                    content,
                    asset_class,
                    metadata_str,
                    datetime.utcnow().isoformat(),
                ),
            )

            self.conn.commit()
            logger.info(f"Stored financial statement {statement_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing financial statement: {e}")
            return False

    def store_recommendation(
        self,
        rec_id: str,
        document_id: str,
        asset_class: str,
        recommendation: str,
        confidence: float,
        data_sources: Optional[List[str]] = None,
    ) -> bool:
        """
        Store extracted recommendation

        Args:
            rec_id: Unique ID for the recommendation
            document_id: ID of the source document
            asset_class: Asset class covered
            recommendation: The recommendation text
            confidence: Confidence score (0-1)
            data_sources: List of data sources used

        Returns:
            True if successful
        """
        try:
            cursor = self.conn.cursor()
            data_sources_str = (
                json.dumps(data_sources) if data_sources else None
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO key_recommendations
                (id, document_id, asset_class, recommendation, confidence, data_sources)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rec_id,
                    document_id,
                    asset_class,
                    recommendation,
                    confidence,
                    data_sources_str,
                ),
            )

            self.conn.commit()
            logger.info(f"Stored recommendation {rec_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing recommendation: {e}")
            return False

    def get_summary(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get summary for a document"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM document_summaries WHERE document_id = ?",
                (document_id,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return None

    def get_financial_statements(
        self, document_id: str, statement_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get financial statements for a document"""
        try:
            cursor = self.conn.cursor()

            if statement_type:
                cursor.execute(
                    """
                    SELECT * FROM financial_statements
                    WHERE document_id = ? AND statement_type = ?
                    """,
                    (document_id, statement_type),
                )
            else:
                cursor.execute(
                    "SELECT * FROM financial_statements WHERE document_id = ?",
                    (document_id,),
                )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting financial statements: {e}")
            return []

    def get_recommendations(
        self, document_id: str, asset_class: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recommendations for a document"""
        try:
            cursor = self.conn.cursor()

            if asset_class:
                cursor.execute(
                    """
                    SELECT * FROM key_recommendations
                    WHERE document_id = ? AND asset_class = ?
                    """,
                    (document_id, asset_class),
                )
            else:
                cursor.execute(
                    "SELECT * FROM key_recommendations WHERE document_id = ?",
                    (document_id,),
                )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

    def search_recommendations(
        self, asset_class: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search recommendations by asset class across all documents"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM key_recommendations
                WHERE asset_class = ?
                ORDER BY confidence DESC
                LIMIT ?
                """,
                (asset_class, limit),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching recommendations: {e}")
            return []

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Financial data database connection closed")

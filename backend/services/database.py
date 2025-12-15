import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from config.settings import settings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class ChromaDBService:
    """
    Vector database service for semantic search and storage.

    Manages connections to Chroma vector database for storing and
    searching document embeddings. Supports external research reports
    and internal documents with semantic similarity search.

    Attributes
    ----------
    external_host : str
        Host address for external research database.
    external_port : int
        Port number for external research database.
    embedding_model : OpenAIEmbeddings
        Embedding model for vectorizing documents.
    external_client : chromadb.Client
        Chroma client for external research.
    external_collection : Collection
        Chroma collection for external documents.
    """

    def __init__(
        self,
        external_host: str = None,
        external_port: int = None,
    ):
        """
        Initialize the ChromaDB service.

        Sets up connections to vector database and initializes embedding
        models for semantic search.

        Parameters
        ----------
        external_host : str, optional
            Host for external Chroma database. Uses settings if not
            provided.
        external_port : int, optional
            Port for external Chroma database. Uses settings if not
            provided.
        """
        # Use provided parameters or settings
        self.external_host = external_host or settings.external_chroma_host
        self.external_port = external_port or settings.external_chroma_port

        # Initialize embedding model for vector search
        logger.info("Initializing OpenAI embeddings...")
        try:
            self.embedding_model = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
            )
            logger.info("OpenAI embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self.embedding_model = None

        # Initialize external client with retry logic
        self.external_client = self._init_chroma_client(
            self.external_host, self.external_port, "external"
        )

        # Collections will be created lazily on first use
        self.external_collection = None

        msg = "ChromaDB service initialized for external research reports"
        logger.info(msg)

    def _init_chroma_client(self, host: str, port: int, db_name: str):
        """
        Initialize Chroma database client with retry logic.

        Attempts to connect to Chroma database with exponential backoff.
        Falls back to stub client if connection fails after max retries.

        Parameters
        ----------
        host : str
            Database host address.
        port : int
            Database port number.
        db_name : str
            Name of database ('external' or 'internal').

        Returns
        -------
        chromadb.Client or StubChromaClient
            Initialized client or stub if connection fails.
        """
        max_retries = 5
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                client = chromadb.HttpClient(host=host, port=port)
                logger.info(f"Connected to {db_name} Chroma at {host}:{port}")
                return client
            except ValueError as e:
                if "tenant" in str(e).lower():
                    logger.warning(
                        f"Attempt {attempt + 1}: Tenant validation failed "
                        f"for {db_name}, retrying..."
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.warning(
                            f"Failed to initialize {db_name} after "
                            f"{max_retries} attempts. Using stub client."
                        )
                        return StubChromaClient(host, port, db_name)
                raise
            except Exception as e:
                logger.error(f"Failed to connect to {db_name} Chroma: {e}")
                raise

    def _ensure_collection(self, client, collection_name: str):
        """
        Lazily create or retrieve a collection.

        Creates collection on first use with appropriate metadata for
        vector search (cosine similarity).

        Parameters
        ----------
        client : chromadb.Client
            Chroma client instance.
        collection_name : str
            Name of collection to create/retrieve.

        Returns
        -------
        Collection
            Chroma collection object.

        Raises
        ------
        Exception
            If collection creation fails.
        """
        try:
            if hasattr(client, "_real_client"):
                # Using stub client
                return client.get_or_create_collection(name=collection_name)
            else:
                # Using real client
                return client.get_or_create_collection(
                    name=collection_name, metadata={"hnsw:space": "cosine"}
                )
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    @property
    def external_collection_instance(self):
        """
        Lazy-load external collection.

        Returns
        -------
        Collection
            External research reports collection.
        """
        if self.external_collection is None:
            self.external_collection = self._ensure_collection(
                self.external_client, "external_reports"
            )
        return self.external_collection

    async def health_check(self) -> Dict[str, str]:
        """
        Check database health status.

        Returns
        -------
        Dict[str, str]
            Health status for each database.
        """
        health = {}
        try:
            if hasattr(self.external_client, "_real_client"):
                # Stub client
                health["external"] = "stub_mode"
            else:
                health["external"] = "healthy"
        except Exception as e:
            health["external"] = f"error: {str(e)}"

        return health

    async def store_document_chunk(
        self,
        content: str,
        document_id: str,
        filename: str,
        chunk_id: int,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Store a document chunk with embedding in vector database.

        Embeds chunk content and stores with metadata for semantic
        search.

        Parameters
        ----------
        content : str
            Text content of the chunk.
        document_id : str
            Unique identifier for the document.
        filename : str
            Original filename of document.
        chunk_id : int
            Chunk sequence number.
        metadata : Dict[str, Any], optional
            Additional metadata to store.

        Returns
        -------
        str
            Unique chunk ID.
        """
        # Use external collection
        collection = self.external_collection_instance

        # Generate unique chunk ID
        chunk_uid = f"{document_id}_chunk_{chunk_id}"

        # Prepare metadata
        chunk_metadata = {
            "document_id": document_id,
            "filename": filename,
            "chunk_id": chunk_id,
            "stored_at": datetime.utcnow().isoformat(),
        }
        if metadata:
            chunk_metadata.update(metadata)

        try:
            # Compute embedding if model is available
            embedding = None
            if self.embedding_model:
                try:
                    # OpenAI embeddings returns a list of floats
                    embedding = self.embedding_model.embed_query(content)
                except Exception as e:
                    logger.warning(f"Failed to compute embedding: {e}")

            # Add to collection with embedding
            if embedding:
                collection.add(
                    ids=[chunk_uid],
                    documents=[content],
                    embeddings=[embedding],
                    metadatas=[chunk_metadata],
                )
            else:
                collection.add(
                    ids=[chunk_uid],
                    documents=[content],
                    metadatas=[chunk_metadata],
                )
            msg = f"Stored chunk {chunk_uid}"
            logger.info(msg)
            return chunk_uid
        except Exception as e:
            logger.error(f"Error storing chunk: {e}")
            raise

    async def search_documents(
        self, query: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar documents.

        Uses embedding similarity to find relevant documents matching
        the query.

        Parameters
        ----------
        query : str
            Search query text.
        n_results : int, optional
            Number of results to return. Default is 5.

        Returns
        -------
        List[Dict[str, Any]]
            List of matching documents with similarity scores.
        """
        results = []

        # Compute query embedding using OpenAI
        query_embedding = None
        if self.embedding_model:
            try:
                # OpenAI embeddings returns a list of floats
                query_embedding = self.embedding_model.embed_query(query)
            except Exception as e:
                logger.warning(f"Failed to compute query embedding: {e}")

        try:
            if query_embedding:
                ext_results = self.external_collection_instance.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                )
            else:
                ext_results = self.external_collection_instance.query(
                    query_texts=[query], n_results=n_results
                )
            results.extend(self._format_results(ext_results, "external"))

            return results[:n_results]
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def _format_results(
        self, chroma_results: Dict, source_type: str
    ) -> List[Dict[str, Any]]:
        """
        Format Chroma query results to standard format.

        Parameters
        ----------
        chroma_results : Dict
            Raw results from Chroma query.
        source_type : str
            Source type ('external' or 'internal').

        Returns
        -------
        List[Dict[str, Any]]
            Formatted results with content and metadata.
        """
        formatted = []
        if not chroma_results or not chroma_results.get("documents"):
            return formatted

        docs = chroma_results.get("documents", [[]])[0]
        metas = chroma_results.get("metadatas", [[]])[0]
        dists = chroma_results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            similarity = 1 - dist if dist is not None else 0
            formatted.append(
                {
                    "document_id": meta.get("document_id"),
                    "content": doc,
                    "similarity_score": similarity,
                    "source": source_type,
                    "metadata": meta,
                }
            )
        return formatted

    async def get_all_documents(
        self, source_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all stored documents with optional filtering.

        Parameters
        ----------
        source_type : str, optional
            Filter by 'external' or 'internal'. None returns all.

        Returns
        -------
        List[Dict[str, Any]]
            List of documents with metadata.
        """
        documents = []
        try:
            if source_type in [None, "external"]:
                try:
                    ext_docs = self.external_collection_instance.get()
                    for doc_id, meta in zip(
                        ext_docs.get("ids", []), ext_docs.get("metadatas", [])
                    ):
                        documents.append(
                            {
                                "id": doc_id,
                                "source": "external",
                                "metadata": meta,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to get external docs: {e}")

            if source_type in [None, "internal"]:
                try:
                    int_docs = self.internal_collection_instance.get()
                    for doc_id, meta in zip(
                        int_docs.get("ids", []), int_docs.get("metadatas", [])
                    ):
                        documents.append(
                            {
                                "id": doc_id,
                                "source": "internal",
                                "metadata": meta,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to get internal docs: {e}")

            return documents
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            return []

    async def delete_document(
        self, document_id: str, source_type: Optional[str] = None
    ) -> bool:
        """
        Delete a document and all its chunks from database.

        Removes all vectors and metadata associated with a document.

        Parameters
        ----------
        document_id : str
            Document ID to delete.
        source_type : str, optional
            Delete from specific source ('external' or 'internal').
            None deletes from all sources.

        Returns
        -------
        bool
            True if deletion succeeded.
        """
        try:
            success = False

            if source_type in [None, "external"]:
                try:
                    self.external_collection_instance.delete(ids=[document_id])
                    success = True
                    logger.info(
                        f"Deleted document {document_id} " "from external DB"
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete from external: {e}")

            if source_type in [None, "internal"]:
                try:
                    self.internal_collection_instance.delete(ids=[document_id])
                    success = True
                    logger.info(
                        f"Deleted document {document_id} " "from internal DB"
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete from internal: {e}")

            return success

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False


class StubChromaClient:
    """Stub client for Chroma when direct HTTP fails"""

    def __init__(self, host: str, port: int, db_name: str):
        self.host = host
        self.port = port
        self.db_name = db_name
        self._real_client = None

    def _get_client(self):
        """Get or create the HTTP client"""
        if self._real_client is None:
            import requests

            # Test connection with a simple HTTP request
            try:
                requests.get(
                    f"http://{self.host}:{self.port}/api/v1/heartbeat",
                    timeout=5,
                )
                logger.info("Stub client verified server is reachable")
                # Try to create real client
                try:
                    self._real_client = chromadb.HttpClient(
                        host=self.host, port=self.port
                    )
                    logger.info(f"Created real HttpClient for {self.db_name}")
                except Exception as e:
                    logger.warning(f"Could not create full client: {e}")
                    # Mark that we tried but failed, use stub mode
                    self._real_client = "stub_mode"
            except requests.RequestException as e:
                logger.error(
                    f"Server {self.host}:{self.port} not reachable: {e}"
                )
                self._real_client = "stub_mode"

        # Return the real client if available, or None to use stub
        if self._real_client == "stub_mode":
            return None
        return self._real_client

    def get_or_create_collection(self, **kwargs):
        """Get or create a collection, with stub fallback"""
        client = self._get_client()
        if client is None:
            # Stub mode - return stub collection
            logger.info("Stub mode: returning stub collection")
            return StubCollection(**kwargs)

        try:
            return client.get_or_create_collection(**kwargs)
        except Exception as e:
            logger.warning(f"Collection creation failed: {e}, " f"using stub")
            return StubCollection(**kwargs)

    def get_collection(self, **kwargs):
        try:
            return self._get_client().get_collection(**kwargs)
        except (ValueError, AttributeError):
            return StubCollection(**kwargs)

    def list_collections(self):
        try:
            return self._get_client().list_collections()
        except (ValueError, AttributeError):
            return []


class StubCollection:
    """Stub collection when all else fails"""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "unknown")
        self.metadata = {}
        self.documents = {}  # In-memory storage for stub mode
        self.embeddings = {}  # Store embeddings for similarity search

    def add(
        self,
        ids=None,
        documents=None,
        embeddings=None,
        metadatas=None,
        **kwargs,
    ):
        """Store documents and embeddings in memory"""
        logger.info(
            f"Stub collection {self.name}: storing " f"{len(ids or [])} items"
        )
        if ids and documents:
            for idx, doc_id in enumerate(ids or []):
                doc = documents[idx] if idx < len(documents) else None
                emb = (
                    embeddings[idx]
                    if (embeddings and idx < len(embeddings))
                    else None
                )
                meta = (
                    metadatas[idx]
                    if (metadatas and idx < len(metadatas))
                    else {}
                )

                self.documents[doc_id] = {
                    "content": doc,
                    "metadata": meta or {},
                }
                if emb is not None:
                    self.embeddings[doc_id] = emb

    def query(
        self, query_texts=None, query_embeddings=None, n_results=5, **kwargs
    ):
        """Perform similarity search on stored documents"""
        logger.info(
            f"Stub collection {self.name}: query called with "
            f"query_embeddings={query_embeddings is not None}"
        )

        if not self.documents:
            # No documents stored
            return {
                "ids": [[]],
                "embeddings": None,
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        # If we have embeddings and query embedding, use similarity search
        if query_embeddings and self.embeddings:
            try:
                query_emb = np.array(query_embeddings[0])

                # Calculate cosine similarity with all stored embeddings
                similarities = []
                doc_ids = []
                docs = []
                metas = []

                for doc_id in self.documents.keys():
                    if doc_id in self.embeddings:
                        stored_emb = np.array(self.embeddings[doc_id])
                        # Cosine similarity
                        norm_product = (
                            np.linalg.norm(query_emb)
                            * np.linalg.norm(stored_emb)
                            + 1e-8
                        )
                        similarity = np.dot(query_emb, stored_emb) / (
                            norm_product
                        )
                        similarities.append(similarity)
                        doc_ids.append(doc_id)
                        docs.append(self.documents[doc_id]["content"])
                        metas.append(self.documents[doc_id]["metadata"])

                # Sort by similarity (desc) and take top n_results
                if similarities:
                    sorted_indices = np.argsort(similarities)[::-1][:n_results]
                    top_ids = [doc_ids[i] for i in sorted_indices]
                    top_docs = [docs[i] for i in sorted_indices]
                    top_metas = [metas[i] for i in sorted_indices]
                    top_distances = [
                        1 - similarities[i] for i in sorted_indices
                    ]

                    logger.info(
                        f"Found {len(top_ids)} results " f"using embeddings"
                    )
                    return {
                        "ids": [top_ids],
                        "embeddings": None,
                        "documents": [top_docs],
                        "metadatas": [top_metas],
                        "distances": [top_distances],
                    }
            except Exception as e:
                logger.warning(f"Embedding-based search failed: {e}")

        # Fallback: return empty if no embeddings or search failed
        return {
            "ids": [[]],
            "embeddings": None,
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def get(self, where=None, **kwargs):
        """Get documents from in-memory storage"""
        logger.info(f"Stub collection {self.name}: get called")
        return {
            "ids": list(self.documents.keys()),
            "embeddings": None,
            "documents": [doc["content"] for doc in self.documents.values()],
            "metadatas": [doc["metadata"] for doc in self.documents.values()],
        }

    def delete(self, ids=None, **kwargs):
        """Delete documents from in-memory storage"""
        if ids:
            for doc_id in ids:
                self.documents.pop(doc_id, None)
                self.embeddings.pop(doc_id, None)

                self.embeddings.pop(doc_id, None)

                self.embeddings.pop(doc_id, None)

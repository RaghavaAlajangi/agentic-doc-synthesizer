import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralized application settings"""

    # ==================== API Configuration ====================
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_title: str = "Agentic AI Research Assistant"
    api_version: str = "1.0.0"

    # CORS Configuration
    cors_origins: list = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:5173",
    ]

    # ==================== OpenAI Configuration ====================
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    # ==================== ChromaDB Configuration ====================
    external_chroma_host: str = os.getenv("EXTERNAL_CHROMA_HOST", "localhost")
    external_chroma_port: int = int(os.getenv("EXTERNAL_CHROMA_PORT", "8001"))

    # ==================== Document Processing ====================
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    document_processor_temperature: float = float(
        os.getenv("DOCUMENT_PROCESSOR_TEMPERATURE", "0.7")
    )

    # ==================== Search Configuration ====================
    search_results_count: int = int(os.getenv("SEARCH_RESULTS_COUNT", "5"))
    similarity_threshold: float = float(
        os.getenv("SIMILARITY_THRESHOLD", "0.5")
    )

    # ==================== Agent Configuration ====================
    agent_temperature: float = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

    # ==================== Logging Configuration ====================
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

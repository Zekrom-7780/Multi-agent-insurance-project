from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    LLM_BASE_URL: str = "http://localhost:8080/v1"
    LLM_MODEL: str = "phi-3-mini-4k-instruct"
    LLM_API_KEY: str = "not-needed"

    DATABASE_URL: str = "sqlite+aiosqlite:///./insurance_support.db"

    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "insurance_FAQ_collection"

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

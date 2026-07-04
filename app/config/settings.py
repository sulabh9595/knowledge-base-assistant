# Creator: Sulabh Bansod
# Description: Configuration settings module using Pydantic.
# Use: Loads and provides environment variables from the .env file.

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Knowledge Base Assistant"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True

    confluence_base_url: str = ""
    confluence_email: str = ""
    confluence_api_token: str = ""
    ollama_host: str = ""
    ollama_model: str = "Qwen3:8b"
    embedding_model: str = "nomic-embed-text"
    chroma_persist_directory: str = "./chroma_store"
    chroma_collection_name: str = "knowledge_base"
    memory_store_file: str = "./memory/documents.json"
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

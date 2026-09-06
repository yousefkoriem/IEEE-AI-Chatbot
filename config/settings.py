"""Environment-backed application settings."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the RAG agent."""

    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "ieee-bsu-ai-chatbot")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    chat_model: str = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
    lite_model: str = os.getenv("LITE_MODEL", "gemini-2.5-flash-lite")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    vtools_ou_code: str = os.getenv("VTOOLS_OU_CODE", "")


settings = Settings()

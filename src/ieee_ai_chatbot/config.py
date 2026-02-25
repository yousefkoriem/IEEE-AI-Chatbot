from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    google_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_metric: str
    pinecone_dimension: int
    chat_model: str
    embedding_model: str
    retriever_k: int
    retriever_fetch_k: int
    chunk_size: int
    chunk_overlap: int
    docs_pdf_dir: str
    docs_ppt_dir: str
    website_default_url: str
    website_max_pages: int
    website_timeout_seconds: int
    manifest_path: str
    langsmith_api_key: str
    langsmith_project: str
    langsmith_tracing: bool

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "ieee-ai-chatbot"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "default"),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            pinecone_metric=os.getenv("PINECONE_METRIC", "cosine"),
            pinecone_dimension=int(os.getenv("PINECONE_DIMENSION", "768")),
            chat_model=os.getenv("CHAT_MODEL", "gemini-2.5-flash-lite"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
            retriever_k=int(os.getenv("RETRIEVER_K", "5")),
            retriever_fetch_k=int(os.getenv("RETRIEVER_FETCH_K", "20")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            docs_pdf_dir=os.getenv("DOCS_PDF_DIR", "docs/pdf"),
            docs_ppt_dir=os.getenv("DOCS_PPT_DIR", "docs/ppt"),
            website_default_url=os.getenv("WEBSITE_DEFAULT_URL", "https://ieee-mangment.vercel.app/"),
            website_max_pages=int(os.getenv("WEBSITE_MAX_PAGES", "25")),
            website_timeout_seconds=int(os.getenv("WEBSITE_TIMEOUT_SECONDS", "20")),
            manifest_path=os.getenv("MANIFEST_PATH", ".rag_manifest.json"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "IEEE-AI-Chatbot"),
            langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        )

    def validate_required(self) -> tuple[bool, list[str]]:
        missing: list[str] = []
        if not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        return (len(missing) == 0, missing)


def configure_langsmith(settings: Settings) -> None:
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        if settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project


def langsmith_status(settings: Settings) -> dict[str, str]:
    enabled = "enabled" if settings.langsmith_tracing else "disabled"
    api_key = "set" if settings.langsmith_api_key else "missing"
    return {
        "tracing": enabled,
        "api_key": api_key,
        "project": settings.langsmith_project,
    }

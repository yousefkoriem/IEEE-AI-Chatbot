from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


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
    chat_model_fallback: str
    chat_quota_retry_seconds: int
    max_output_tokens: int
    temperature: float
    embedding_model: str
    embedding_model_fallback: str
    retriever_k: int
    retriever_fetch_k: int
    retriever_min_score: float
    internet_fallback_enabled: bool
    web_search_results: int
    web_search_timeout_seconds: int
    web_search_provider: str
    web_search_tavily_key: str
    web_search_serpapi_key: str
    chunk_size: int
    chunk_overlap: int
    docs_pdf_dir: str
    docs_ppt_dir: str
    docs_doc_dir: str
    website_default_url: str
    website_max_pages: int
    website_timeout_seconds: int
    manifest_path: str
    langsmith_api_key: str
    langsmith_project: str
    langsmith_tracing: bool
    langsmith_endpoint: str
    chat_history_db_path: str
    rate_limit_max_requests: int
    rate_limit_window_seconds: int
    feedback_boost_enabled: bool
    feedback_boost_factor: float
    vector_store_type: str
    vector_store_chroma_dir: str
    local_retrieval_enabled: bool
    local_retrieval_max_results: int
    local_retrieval_min_score: float

    @classmethod
    def from_env(cls) -> "Settings":
        from pathlib import Path as _Path

        from dotenv import find_dotenv

        load_dotenv()
        dotenv_file = find_dotenv(usecwd=True)
        # Resolve relative paths against the project root (directory containing .env or CWD)
        project_root = _Path(dotenv_file).resolve().parent if dotenv_file else _Path.cwd()

        def _resolve_path(value: str) -> str:
            p = _Path(value)
            if p.is_absolute():
                return value
            return str(project_root / p)

        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "ieee-ai-chatbot"),
            pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "default"),
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
            pinecone_metric=os.getenv("PINECONE_METRIC", "cosine"),
            pinecone_dimension=int(os.getenv("PINECONE_DIMENSION", "1024")),
            chat_model=os.getenv("CHAT_MODEL", "gemini-2.5-flash-lite"),
            chat_model_fallback=os.getenv("CHAT_MODEL_FALLBACK", "gemini-2.5-flash-lite"),
            chat_quota_retry_seconds=int(os.getenv("CHAT_QUOTA_RETRY_SECONDS", "30")),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "1200")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
            embedding_model_fallback=os.getenv("EMBEDDING_MODEL_FALLBACK", "models/gemini-embedding-001"),
            retriever_k=int(os.getenv("RETRIEVER_K", "5")),
            retriever_fetch_k=int(os.getenv("RETRIEVER_FETCH_K", "10")),
            retriever_min_score=float(os.getenv("RETRIEVER_MIN_SCORE", "0.40")),
            internet_fallback_enabled=os.getenv("INTERNET_FALLBACK_ENABLED", "false").lower() == "true",
            web_search_results=int(os.getenv("WEB_SEARCH_RESULTS", "3")),
            web_search_timeout_seconds=int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "8")),
            web_search_provider=os.getenv("WEB_SEARCH_PROVIDER", "google"),
            web_search_tavily_key=os.getenv("WEB_SEARCH_TAVILY_KEY", ""),
            web_search_serpapi_key=os.getenv("WEB_SEARCH_SERPAPI_KEY", ""),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            docs_pdf_dir=_resolve_path(os.getenv("DOCS_PDF_DIR", "docs/pdf")),
            docs_ppt_dir=_resolve_path(os.getenv("DOCS_PPT_DIR", "docs/ppt")),
            docs_doc_dir=_resolve_path(os.getenv("DOCS_DOC_DIR", "docs/doc")),
            website_default_url=os.getenv("WEBSITE_DEFAULT_URL", "https://ieee-mangment.vercel.app/"),
            website_max_pages=int(os.getenv("WEBSITE_MAX_PAGES", "25")),
            website_timeout_seconds=int(os.getenv("WEBSITE_TIMEOUT_SECONDS", "20")),
            manifest_path=_resolve_path(os.getenv("MANIFEST_PATH", ".rag_manifest.json")),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "IEEE-AI-Chatbot"),
            langsmith_tracing=os.getenv("LANGSMITH_TRACING", "true").lower() == "true",
            langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
            chat_history_db_path=_resolve_path(os.getenv("CHAT_HISTORY_DB_PATH", "chat_history.db")),
            rate_limit_max_requests=int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30")),
            rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
            feedback_boost_enabled=os.getenv("FEEDBACK_BOOST_ENABLED", "true").lower() == "true",
            feedback_boost_factor=float(os.getenv("FEEDBACK_BOOST_FACTOR", "0.3")),
            vector_store_type=os.getenv("VECTOR_STORE_TYPE", "pinecone"),
            vector_store_chroma_dir=_resolve_path(os.getenv("VECTOR_STORE_CHROMA_DIR", ".vector_db")),
            local_retrieval_enabled=os.getenv("LOCAL_RETRIEVAL_ENABLED", "true").lower() == "true",
            local_retrieval_max_results=int(os.getenv("LOCAL_RETRIEVAL_MAX_RESULTS", "3")),
            local_retrieval_min_score=float(os.getenv("LOCAL_RETRIEVAL_MIN_SCORE", "0.3")),
        )

    def validate_required(self) -> tuple[bool, list[str]]:
        missing: list[str] = []
        if not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if not self.chat_model.strip():
            missing.append("CHAT_MODEL")
        return (len(missing) == 0, missing)


def configure_langsmith(settings: Settings) -> None:
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        if settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint


def langsmith_status(settings: Settings) -> dict[str, str]:
    enabled = "enabled" if settings.langsmith_tracing else "disabled"
    api_key = "set" if settings.langsmith_api_key else "missing"
    return {
        "tracing": enabled,
        "api_key": api_key,
        "project": settings.langsmith_project,
    }

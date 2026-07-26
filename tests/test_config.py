"""Tests for config module."""

import os
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ieee_ai_chatbot.config import Settings, langsmith_status


def _make_settings(**overrides):
    """Create a Settings instance with sensible test defaults."""
    defaults = dict(
        google_api_key="test-google-key",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="test-index",
        pinecone_namespace="default",
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
        pinecone_metric="cosine",
        pinecone_dimension=1024,
        chat_model="gemini-2.5-flash-lite",
        chat_model_fallback="gemini-2.5-flash-lite",
        chat_quota_retry_seconds=30,
        max_output_tokens=400,
        temperature=0.2,
        embedding_model="models/gemini-embedding-001",
        embedding_model_fallback="models/gemini-embedding-001",
        retriever_k=3,
        retriever_fetch_k=10,
        retriever_min_score=0.40,
        internet_fallback_enabled=True,
        web_search_results=3,
        web_search_timeout_seconds=8,
        web_search_provider="duckduckgo",
        web_search_tavily_key="",
        web_search_serpapi_key="",
        chunk_size=1200,
        chunk_overlap=150,
        docs_pdf_dir="docs/pdf",
        docs_ppt_dir="docs/ppt",
        docs_doc_dir="docs/doc",
        website_default_url="https://example.com/",
        website_max_pages=25,
        website_timeout_seconds=20,
        manifest_path=".rag_manifest.json",
        langsmith_api_key="",
        langsmith_project="Test",
        langsmith_tracing=False,
        langsmith_endpoint="https://api.smith.langchain.com",
        chat_history_db_path=":memory:",
        rate_limit_max_requests=30,
        rate_limit_window_seconds=60,
        feedback_boost_enabled=True,
        feedback_boost_factor=0.3,
        vector_store_type="pinecone",
        vector_store_chroma_dir=".vector_db",
        local_retrieval_enabled=True,
        local_retrieval_max_results=3,
        local_retrieval_min_score=0.3,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestValidateRequired:
    def test_all_keys_present(self):
        s = _make_settings()
        ok, missing = s.validate_required()
        assert ok is True
        assert missing == []

    def test_missing_google_key(self):
        s = _make_settings(google_api_key="")
        ok, missing = s.validate_required()
        assert ok is False
        assert "GOOGLE_API_KEY" in missing

    def test_missing_pinecone_key(self):
        s = _make_settings(pinecone_api_key="")
        ok, missing = s.validate_required()
        assert ok is False
        assert "PINECONE_API_KEY" in missing

    def test_missing_chat_model(self):
        s = _make_settings(chat_model="")
        ok, missing = s.validate_required()
        assert ok is False
        assert "CHAT_MODEL" in missing

    def test_multiple_missing(self):
        s = _make_settings(google_api_key="", pinecone_api_key="", chat_model="")
        ok, missing = s.validate_required()
        assert ok is False
        assert len(missing) == 3


class TestLangsmithStatus:
    def test_tracing_enabled(self):
        s = _make_settings(langsmith_tracing=True, langsmith_api_key="key123")
        status = langsmith_status(s)
        assert status["tracing"] == "enabled"
        assert status["api_key"] == "set"

    def test_tracing_disabled(self):
        s = _make_settings(langsmith_tracing=False, langsmith_api_key="")
        status = langsmith_status(s)
        assert status["tracing"] == "disabled"
        assert status["api_key"] == "missing"

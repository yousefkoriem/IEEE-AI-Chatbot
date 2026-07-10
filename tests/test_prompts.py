"""Tests for prompts module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ieee_ai_chatbot.prompts import (
    PromptConfig,
    build_prompt_config,
    build_user_prompt,
    validate_prompt_config,
    CONTEXT_AVAILABLE_INSTRUCTION,
    NO_CONTEXT_INSTRUCTION,
)

# Use a minimal settings-like object for build_prompt_config
from ieee_ai_chatbot.config import Settings


def _make_settings(**overrides):
    defaults = dict(
        google_api_key="k",
        pinecone_api_key="k",
        pinecone_index_name="idx",
        pinecone_namespace="ns",
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
        pinecone_metric="cosine",
        pinecone_dimension=1024,
        chat_model="gemini-2.5-flash-lite",
        chat_model_fallback="gemini-2.5-flash-lite",
        chat_quota_retry_seconds=30,
        max_output_tokens=400,
        embedding_model="models/gemini-embedding-001",
        embedding_model_fallback="models/gemini-embedding-001",
        retriever_k=3,
        retriever_fetch_k=10,
        internet_fallback_enabled=True,
        web_search_results=3,
        web_search_timeout_seconds=8,
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
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestPromptConfig:
    def test_build_prompt_config_default(self):
        s = _make_settings(max_output_tokens=400)
        config = build_prompt_config(s)
        assert config.max_context_chars == 8000  # 400 * 20

    def test_build_prompt_config_minimum_floor(self):
        s = _make_settings(max_output_tokens=10)
        config = build_prompt_config(s)
        assert config.max_context_chars == 1000  # min floor

    def test_validate_prompt_config_valid(self):
        config = PromptConfig(max_context_chars=5000)
        validate_prompt_config(config)  # should not raise

    def test_validate_prompt_config_invalid(self):
        config = PromptConfig(max_context_chars=0)
        try:
            validate_prompt_config(config)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestBuildUserPrompt:
    def test_with_context(self):
        config = PromptConfig(max_context_chars=500)
        result = build_user_prompt(
            question="What is IEEE?",
            history_text="user: hello",
            context="IEEE is an organization.",
            prompt_config=config,
        )
        assert CONTEXT_AVAILABLE_INSTRUCTION in result
        assert "What is IEEE?" in result
        assert "IEEE is an organization." in result
        assert "user: hello" in result

    def test_without_context(self):
        config = PromptConfig(max_context_chars=500)
        result = build_user_prompt(
            question="What is IEEE?",
            history_text="",
            context="",
            prompt_config=config,
        )
        assert NO_CONTEXT_INSTRUCTION in result
        assert "N/A" in result

    def test_context_truncation(self):
        config = PromptConfig(max_context_chars=10)
        long_context = "A" * 100
        result = build_user_prompt(
            question="q",
            history_text="",
            context=long_context,
            prompt_config=config,
        )
        # The context in the prompt should be truncated to 10 chars
        assert "A" * 10 in result
        assert "A" * 11 not in result

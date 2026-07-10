"""Tests for ingestion helpers and UI history normalization."""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ieee_ai_chatbot.ingest import (
    _sha256_file,
    _load_manifest,
    _save_manifest,
    _extract_pdf_text,
    SUPPORTED_EXTENSIONS,
)
from ieee_ai_chatbot.ui_gradio import _history_to_text, _normalize_history


class TestSha256File:
    def test_hash_matches(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello world")
            f.flush()
            path = Path(f.name)
        try:
            result = _sha256_file(path)
            expected = hashlib.sha256(b"hello world").hexdigest()
            assert result == expected
        finally:
            path.unlink()


class TestManifest:
    def test_load_missing_returns_empty(self):
        path = Path(tempfile.mktemp(suffix=".json"))
        assert not path.exists()
        manifest = _load_manifest(path)
        assert manifest == {"sources": {}}

    def test_save_and_load_roundtrip(self):
        path = Path(tempfile.mktemp(suffix=".json"))
        try:
            payload = {"sources": {"file1": {"hash": "abc", "chunk_ids": ["c1"]}}}
            _save_manifest(path, payload)
            loaded = _load_manifest(path)
            assert loaded == payload
        finally:
            if path.exists():
                path.unlink()


class TestSupportedExtensions:
    def test_expected_extensions(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".ppt" in SUPPORTED_EXTENSIONS
        assert ".pptx" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".doc" in SUPPORTED_EXTENSIONS
        assert ".txt" not in SUPPORTED_EXTENSIONS


class TestHistoryToText:
    def test_dict_format(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = _history_to_text(history)
        assert "user: hello" in result
        assert "assistant: hi there" in result

    def test_list_format(self):
        history = [["hello", "hi there"]]
        result = _history_to_text(history)
        assert "user: hello" in result
        assert "assistant: hi there" in result

    def test_empty(self):
        assert _history_to_text(None) == ""
        assert _history_to_text([]) == ""


class TestNormalizeHistory:
    def test_dict_items(self):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        result = _normalize_history(history)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "q1"}

    def test_list_items(self):
        history = [["q1", "a1"]]
        result = _normalize_history(history)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "q1"}
        assert result[1] == {"role": "assistant", "content": "a1"}

    def test_filters_empty(self):
        history = [{"role": "user", "content": ""}, {"role": "", "content": "test"}]
        result = _normalize_history(history)
        assert len(result) == 0

    def test_non_list_returns_empty(self):
        assert _normalize_history("not a list") == []
        assert _normalize_history(None) == []

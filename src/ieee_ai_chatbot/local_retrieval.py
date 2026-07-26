from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .ingest import SUPPORTED_EXTENSIONS, _extract_text

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "out off over under again further then once that this these those "
    "i me my we us our you your he him his she her it its they them their "
    "what which who whom how when where why not no nor so if or and but "
    "than too very just about also back even still already yet".split()
)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9\u0600-\u06FF]+", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


def _keyword_overlap_score(query_tokens: list[str], chunk_text: str) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = set(re.findall(r"[a-zA-Z0-9\u0600-\u06FF]+", chunk_text.lower()))
    if not chunk_tokens:
        return 0.0
    matches = sum(1 for t in query_tokens if t in chunk_tokens)
    return matches / len(query_tokens)


def _fuzzy_score(question: str, chunk_text: str) -> float:
    q_lower = question.lower()
    # Compare against individual sentences for more meaningful ratios
    sentences = [s.strip() for s in chunk_text.split('.') if len(s.strip()) > 10]
    if not sentences:
        return SequenceMatcher(None, q_lower, chunk_text[:500].lower()).ratio()
    # Take the best match across sentences (cap at 20 to limit compute)
    return max(SequenceMatcher(None, q_lower, s.lower()).ratio() for s in sentences[:20])


def _combined_score(question: str, chunk_text: str, query_tokens: list[str]) -> float:
    kw = _keyword_overlap_score(query_tokens, chunk_text)
    fz = _fuzzy_score(question, chunk_text)
    return 0.7 * kw + 0.3 * fz


class LocalRetriever:
    def __init__(self, settings: Settings) -> None:
        self._search_roots = [
            Path(settings.docs_pdf_dir),
            Path(settings.docs_ppt_dir),
            Path(settings.docs_doc_dir),
        ]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def search(
        self, question: str, max_results: int = 3, min_score: float = 0.3
    ) -> list[Document]:
        if not question.strip():
            return []

        query_tokens = _tokenize(question)
        if not query_tokens:
            return []

        candidates: list[tuple[Document, float]] = []

        for root in self._search_roots:
            if not root.exists():
                continue
            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                try:
                    text = _extract_text(file_path)
                except Exception:
                    logger.debug("Could not extract text from %s", file_path)
                    continue

                if not text.strip():
                    continue

                chunks = self._splitter.split_text(text)
                for chunk_text in chunks:
                    score = _combined_score(question, chunk_text, query_tokens)
                    if score >= min_score:
                        doc = Document(
                            page_content=chunk_text,
                            metadata={
                                "source": str(file_path.resolve()),
                                "filename": file_path.name,
                                "suffix": file_path.suffix.lower(),
                                "origin": "local",
                            },
                        )
                        candidates.append((doc, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in candidates[:max_results]]

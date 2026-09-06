"""LangChain-compatible retriever backed by PineconeStore."""

import logging
from typing import Any

from rag.vectorstore import PineconeStore
from utils.helpers import format_context

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Wraps PineconeStore as a retrieval interface for the agent tools."""

    def __init__(
        self,
        vectorstore: PineconeStore,
        k: int = 5,
        score_threshold: float = 0.15,
    ):
        self.vectorstore = vectorstore
        self.k = k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Return top-k matching documents for *query*."""
        results = self.vectorstore.similarity_search(
            query,
            k=self.k,
            score_threshold=self.score_threshold,
        )
        logger.info("Retrieved %d documents for query: %.60s...", len(results), query)
        return results

    def retrieve_formatted(self, query: str) -> str:
        """Return matching documents as a single formatted string."""
        results = self.retrieve(query)
        if not results:
            return "No relevant documents found in the knowledge base."
        return format_context(results)

    def get_sources(self, query: str) -> list[dict[str, Any]]:
        """Return source metadata (without full text) for citation display."""
        results = self.retrieve(query)
        return [
            {
                "source": r.get("source", ""),
                "title": r.get("title", ""),
                "score": r.get("score", 0),
            }
            for r in results
        ]
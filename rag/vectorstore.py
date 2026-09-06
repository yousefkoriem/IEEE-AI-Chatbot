"""Pinecone vector store integration."""

import logging
import hashlib
from typing import Any

from pinecone import Pinecone

from config.settings import settings
from rag.embeddings import GeminiEmbeddings

logger = logging.getLogger(__name__)


class PineconeStore:
    """Manages the Pinecone vector index for document storage and retrieval."""

    def __init__(self, embeddings: GeminiEmbeddings | None = None):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(settings.pinecone_index_name)
        self.namespace = settings.pinecone_namespace or None
        self.embeddings = embeddings or GeminiEmbeddings()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 50,
    ) -> int:
        """Embed and upsert documents into Pinecone.

        Each document dict must have a ``text`` key.  Optional keys:
        ``id``, ``source``, ``category``, ``title``, ``page``.
        Returns the number of vectors upserted.
        """
        texts = [doc["text"] for doc in documents]
        vectors_upserted = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)

            records = []
            for doc, embedding in zip(batch_docs, batch_embeddings):
                vec_id = doc.get(
                    "id",
                    hashlib.md5(doc["text"].encode()).hexdigest(),
                )
                metadata = {
                    "text": doc["text"][:40_000],  # Pinecone metadata limit
                    "source": doc.get("source", ""),
                    "category": doc.get("category", ""),
                    "title": doc.get("title", ""),
                    "page": doc.get("page", 0),
                }
                records.append({"id": vec_id, "values": embedding, "metadata": metadata})

            self.index.upsert(vectors=records, namespace=self.namespace)
            vectors_upserted += len(records)
            logger.info("Upserted batch %d–%d (%d vectors)", i, i + len(records), vectors_upserted)

        return vectors_upserted

    def delete_documents(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        self.index.delete(ids=ids, namespace=self.namespace)
        logger.info("Deleted %d vectors", len(ids))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Embed *query* and return the top-k matching documents.

        Only results with a score ≥ *score_threshold* are returned.
        Each result dict contains ``text``, ``source``, ``score``,
        and any other metadata stored in Pinecone.
        """
        query_embedding = self.embeddings.embed_query(query)

        response = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            namespace=self.namespace,
        )

        results: list[dict[str, Any]] = []
        for match in response.get("matches", []):
            if match["score"] < score_threshold:
                continue
            metadata = match.get("metadata", {})
            results.append(
                {
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source", ""),
                    "category": metadata.get("category", ""),
                    "title": metadata.get("title", ""),
                    "score": round(match["score"], 4),
                    "id": match["id"],
                }
            )

        logger.info(
            "Query returned %d results (threshold=%.2f)", len(results), score_threshold
        )
        return results

    # ------------------------------------------------------------------
    # Admin / status
    # ------------------------------------------------------------------

    def get_index_stats(self) -> dict[str, Any]:
        """Return vector count and namespace info for the status panel."""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
            "namespaces": stats.get("namespaces", {}),
            "index_name": settings.pinecone_index_name,
        }
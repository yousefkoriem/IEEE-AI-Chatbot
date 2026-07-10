from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import Settings

logger = logging.getLogger(__name__)


class ResilientGoogleEmbeddings(Embeddings):
    """Embedding wrapper that falls back to an alternate model on NOT_FOUND errors."""

    def __init__(
        self,
        primary_model: str,
        api_key: str,
        fallback_model: str,
        output_dimensionality: int,
    ) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.api_key = api_key
        self.output_dimensionality = output_dimensionality
        self._primary = GoogleGenerativeAIEmbeddings(
            model=self.primary_model,
            google_api_key=self.api_key,
            output_dimensionality=self.output_dimensionality,
        )
        self._fallback = GoogleGenerativeAIEmbeddings(
            model=self.fallback_model,
            google_api_key=self.api_key,
            output_dimensionality=self.output_dimensionality,
        )

    @staticmethod
    def _should_fallback(error: Exception) -> bool:
        message = str(error).lower()
        return "not_found" in message or "not found" in message

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._primary.embed_query(text)
        except Exception as error:
            if not self._should_fallback(error):
                raise
            logger.warning("Primary embedding model failed, using fallback: %s", error)
            return self._fallback.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._primary.embed_documents(texts)
        except Exception as error:
            if not self._should_fallback(error):
                raise
            logger.warning("Primary embedding model failed, using fallback: %s", error)
            return self._fallback.embed_documents(texts)


def _build_embeddings(settings: Settings) -> ResilientGoogleEmbeddings:
    return ResilientGoogleEmbeddings(
        primary_model=settings.embedding_model,
        api_key=settings.google_api_key,
        fallback_model=settings.embedding_model_fallback,
        output_dimensionality=settings.pinecone_dimension,
    )


def ensure_index(settings: Settings) -> None:
    if settings.vector_store_type != "pinecone":
        return
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes: set[str]
    try:
        existing_indexes = set(pc.list_indexes().names())
    except Exception:
        existing_indexes = {
            item["name"] for item in pc.list_indexes().to_dict().get("indexes", [])
        }
    if settings.pinecone_index_name in existing_indexes:
        description = pc.describe_index(settings.pinecone_index_name)
        index_dimension = None
        if hasattr(description, "dimension"):
            index_dimension = getattr(description, "dimension")
        elif isinstance(description, dict):
            index_dimension = description.get("dimension")
        if index_dimension is not None and int(index_dimension) != settings.pinecone_dimension:
            raise ValueError(
                "Pinecone index dimension mismatch: "
                f"index '{settings.pinecone_index_name}' is {index_dimension}, "
                f"but PINECONE_DIMENSION is {settings.pinecone_dimension}. "
                "Use a matching index or update PINECONE_DIMENSION / PINECONE_INDEX_NAME."
            )
        return
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.pinecone_dimension,
        metric=settings.pinecone_metric,
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )


def get_vector_store(settings: Settings) -> Any:
    store_type = settings.vector_store_type or "pinecone"
    embeddings = _build_embeddings(settings)

    if store_type == "chroma":
        try:
            from langchain_chroma import Chroma
            persist_dir = settings.vector_store_chroma_dir or "./chroma_db"
            return Chroma(
                embedding_function=embeddings,
                persist_directory=persist_dir,
                collection_name=settings.pinecone_namespace or "default",
            )
        except ImportError:
            logger.warning("langchain_chroma not installed, falling back to Pinecone")
        except Exception as e:
            logger.warning("Chroma init failed: %s, falling back to Pinecone", e)

    if store_type == "faiss":
        try:
            from langchain_community.vectorstores import FAISS
            import pickle
            persist_dir = settings.vector_store_chroma_dir or "./faiss_index"
            index_path = f"{persist_dir}/index.faiss"
            store_path = f"{persist_dir}/store.pkl"
            try:
                return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
            except Exception:
                texts = ["initialization placeholder"]
                store = FAISS.from_texts(texts, embeddings)
                store.save_local(persist_dir)
                return store
        except ImportError:
            logger.warning("langchain-community not installed, falling back to Pinecone")
        except Exception as e:
            logger.warning("FAISS init failed: %s, falling back to Pinecone", e)

    ensure_index(settings)
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        namespace=settings.pinecone_namespace,
        pinecone_api_key=settings.pinecone_api_key,
    )

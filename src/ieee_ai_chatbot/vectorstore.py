from __future__ import annotations

from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import Settings


class ResilientGoogleEmbeddings:
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

    def embed_query(self, text: str) -> List[float]:
        try:
            return self._primary.embed_query(text)
        except Exception as error:
            if not self._should_fallback(error):
                raise
            return self._fallback.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self._primary.embed_documents(texts)
        except Exception as error:
            if not self._should_fallback(error):
                raise
            return self._fallback.embed_documents(texts)


def ensure_index(settings: Settings) -> None:
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


def get_vector_store(settings: Settings) -> PineconeVectorStore:
    ensure_index(settings)
    embeddings = ResilientGoogleEmbeddings(
        primary_model=settings.embedding_model,
        api_key=settings.google_api_key,
        fallback_model="models/gemini-embedding-001",
        output_dimensionality=settings.pinecone_dimension,
    )
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        namespace=settings.pinecone_namespace,
        pinecone_api_key=settings.pinecone_api_key,
    )

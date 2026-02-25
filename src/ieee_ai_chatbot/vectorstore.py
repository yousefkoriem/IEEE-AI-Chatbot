from __future__ import annotations

from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import Settings


class ResilientGoogleEmbeddings:
    def __init__(self, primary_model: str, api_key: str, fallback_model: str) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.api_key = api_key
        self._primary = GoogleGenerativeAIEmbeddings(
            model=self.primary_model,
            google_api_key=self.api_key,
        )
        self._fallback = GoogleGenerativeAIEmbeddings(
            model=self.fallback_model,
            google_api_key=self.api_key,
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
    )
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        namespace=settings.pinecone_namespace,
        pinecone_api_key=settings.pinecone_api_key,
    )

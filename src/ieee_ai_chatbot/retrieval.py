from __future__ import annotations

from langchain_core.documents import Document

from .config import Settings
from .vectorstore import get_vector_store


def retrieve_context(settings: Settings, question: str) -> list[Document]:
    vector_store = get_vector_store(settings)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": settings.retriever_fetch_k,
        },
    )
    return retriever.invoke(question)

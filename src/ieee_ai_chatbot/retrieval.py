from __future__ import annotations

from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup
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


def search_web_snippets(question: str, max_results: int, timeout_seconds: int) -> list[Document]:
    if not question.strip():
        return []

    response = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": question},
        timeout=timeout_seconds,
        headers={"User-Agent": "IEEE-AI-Chatbot-RAG/1.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results = soup.select("div.result")
    documents: list[Document] = []

    for result in results:
        title_anchor = result.select_one("a.result__a")
        snippet_node = result.select_one(".result__snippet")
        if not title_anchor:
            continue

        title = title_anchor.get_text(" ", strip=True)
        href = title_anchor.get("href", "")
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
        resolved_url = _resolve_duckduckgo_url(href)

        content = f"Title: {title}\nSnippet: {snippet}".strip()
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": resolved_url,
                    "url": resolved_url,
                    "filename": resolved_url,
                    "origin": "web-search",
                    "title": title,
                },
            )
        )

        if len(documents) >= max_results:
            break

    return documents


def _resolve_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc != "duckduckgo.com":
        return url
    query = parse_qs(parsed.query)
    uddg = query.get("uddg", [])
    if not uddg:
        return url
    return unquote(uddg[0])

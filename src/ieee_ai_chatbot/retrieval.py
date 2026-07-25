from __future__ import annotations

import logging
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from .config import Settings

logger = logging.getLogger(__name__)


def search_web_snippets(question: str, max_results: int, timeout_seconds: int, settings: Settings | None = None) -> list[Document]:
    if not question.strip():
        return []

    provider = (settings.web_search_provider or "google") if settings else "google"
    registry = _get_provider_registry()
    func = registry.get(provider)
    if func:
        return func(question, max_results, timeout_seconds, settings)
    logger.warning("Unknown search provider '%s', falling back to Google", provider)
    return _search_google(question, max_results, timeout_seconds, settings)


def _search_duckduckgo(question: str, max_results: int, timeout_seconds: int, settings: Settings | None = None) -> list[Document]:
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
        documents.append(Document(
            page_content=content,
            metadata={"source": resolved_url, "url": resolved_url, "filename": resolved_url, "origin": "web-search", "title": title},
        ))
        if len(documents) >= max_results:
            break
    return documents


def _search_tavily(question: str, max_results: int, timeout_seconds: int, settings: Settings | None = None) -> list[Document]:
    api_key = (settings.web_search_tavily_key or "") if settings else ""
    if not api_key:
        logger.warning("Tavily API key not set, falling back to DuckDuckGo")
        return _search_duckduckgo(question, max_results, timeout_seconds, settings)
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": question, "max_results": max_results},
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        documents = []
        for r in data.get("results", []):
            documents.append(Document(
                page_content=f"Title: {r.get('title', '')}\nSnippet: {r.get('content', '')}",
                metadata={"source": r.get("url", ""), "url": r.get("url", ""), "filename": r.get("url", ""), "origin": "web-search", "title": r.get("title", "")},
            ))
        return documents
    except Exception as e:
        logger.warning("Tavily search failed: %s, falling back to DuckDuckGo", e)
        return _search_duckduckgo(question, max_results, timeout_seconds, settings)


def _search_serpapi(question: str, max_results: int, timeout_seconds: int, settings: Settings | None = None) -> list[Document]:
    api_key = (settings.web_search_serpapi_key or "") if settings else ""
    if not api_key:
        logger.warning("SerpAPI key not set, falling back to DuckDuckGo")
        return _search_duckduckgo(question, max_results, timeout_seconds, settings)
    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={"api_key": api_key, "q": question, "num": max_results, "engine": "google"},
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        documents = []
        for r in data.get("organic_results", []):
            documents.append(Document(
                page_content=f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}",
                metadata={"source": r.get("link", ""), "url": r.get("link", ""), "filename": r.get("link", ""), "origin": "web-search", "title": r.get("title", "")},
            ))
        return documents
    except Exception as e:
        logger.warning("SerpAPI search failed: %s, falling back to DuckDuckGo", e)
        return _search_duckduckgo(question, max_results, timeout_seconds, settings)


def _search_google(question: str, max_results: int, timeout_seconds: int, settings: Settings | None = None) -> list[Document]:
    response = requests.get(
        "https://www.google.com/search",
        params={"q": question, "num": max_results},
        timeout=timeout_seconds,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    documents: list[Document] = []

    for result in soup.select("div.g"):
        title_anchor = result.select_one("a")
        if not title_anchor:
            continue
        href = title_anchor.get("href", "")
        if not href.startswith("http"):
            continue

        title_el = title_anchor.select_one("h3")
        title = title_el.get_text(" ", strip=True) if title_el else ""

        snippet_el = result.select_one("div[data-sncf], div.VwiC3b, span.aCOpRe")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

        content = f"Title: {title}\nSnippet: {snippet}".strip()
        documents.append(Document(
            page_content=content,
            metadata={"source": href, "url": href, "filename": href, "origin": "web-search", "title": title},
        ))
        if len(documents) >= max_results:
            break

    if not documents:
        logger.info("Google scraping returned no results, falling back to DuckDuckGo")
        return _search_duckduckgo(question, max_results, timeout_seconds, settings)

    return documents


def _get_provider_registry() -> dict:
    return {
        "google": _search_google,
        "duckduckgo": _search_duckduckgo,
        "tavily": _search_tavily,
        "serpapi": _search_serpapi,
    }


def _resolve_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc != "duckduckgo.com":
        return url
    query = parse_qs(parsed.query)
    uddg = query.get("uddg", [])
    if not uddg:
        return url
    return unquote(uddg[0])

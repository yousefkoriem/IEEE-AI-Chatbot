"""Tests for Pinecone-first retrieval behavior."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from langchain_core.documents import Document
from ieee_ai_chatbot.chat import RAGAgent


class _VectorStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def similarity_search_with_score(self, question, k):
        self.calls.append((question, k))
        return self.results


class _LocalRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.calls = 0

    def search(self, *args, **kwargs):
        self.calls += 1
        return self.docs


def _agent(vector_results, local_docs=(), boosts=None):
    agent = RAGAgent.__new__(RAGAgent)
    agent.settings = SimpleNamespace(
        retriever_k=3,
        retriever_fetch_k=5,
        feedback_boost_enabled=bool(boosts),
        feedback_boost_factor=0.3,
        local_retrieval_enabled=True,
        local_retrieval_max_results=3,
        local_retrieval_min_score=0.3,
        internet_fallback_enabled=False,
    )
    agent._vectorstore = _VectorStore(vector_results)
    agent._local_retriever = _LocalRetriever(list(local_docs))
    agent._get_chunk_boosts = lambda: boosts or {}
    return agent


def test_pinecone_is_primary_and_duplicate_chunks_are_removed():
    chairman = Document(page_content="Chairman of Branch: Mohamed Sharaf", metadata={"chunk_id": "chair"})
    duplicate = Document(page_content=" Chairman of Branch: Mohamed Sharaf\n", metadata={"chunk_id": "duplicate"})
    supporting = Document(page_content="The board has a Chairman and Secretary.", metadata={"chunk_id": "support"})
    agent = _agent(
        [(chairman, 0.6732), (duplicate, 0.6432), (supporting, 0.60)],
        [Document(page_content="A local keyword match")],
    )

    docs, confidence = agent._retrieve_docs("Who is the chairman?")

    assert [doc.page_content for doc in docs] == [chairman.page_content, supporting.page_content]
    assert confidence == "High"
    assert agent._vectorstore.calls == [("Who is the chairman?", 5)]
    assert agent._local_retriever.calls == 0


def test_local_retrieval_is_used_only_after_pinecone_returns_no_documents():
    local = Document(page_content="Offline local answer")
    agent = _agent([], [local])

    docs, confidence = agent._retrieve_docs("question")

    assert docs == [local]
    assert confidence == "Local"

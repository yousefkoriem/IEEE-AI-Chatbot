from __future__ import annotations

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Settings, configure_langsmith, langsmith_status
from .retrieval import search_web_snippets
from .vectorstore import get_vector_store


class RAGAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configure_langsmith(settings)
        self._retriever = get_vector_store(settings).as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.retriever_k,
                "fetch_k": settings.retriever_fetch_k,
            },
        )
        self._llm = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
            max_output_tokens=settings.max_output_tokens,
        )
        self._agent = create_agent(
            model=self._llm,
            tools=[],
            system_prompt=(
                "You are an assistant for IEEE Beni Suef Student Branch. "
                "Prefer factual, direct answers and avoid unnecessary disclaimers."
            ),
        )

    def answer(self, question: str, history_text: str = "") -> tuple[str, list[str]]:
        docs = self._retrieve_docs(question)
        context_chunks = [doc.page_content for doc in docs]
        sources = [str(doc.metadata.get("filename", "unknown")) for doc in docs]
        context = "\n\n".join(context_chunks)

        if context.strip():
            context_instruction = (
                "Use retrieved context first when it is relevant and sufficient. "
                "If the question also needs general knowledge, combine both clearly and concisely."
            )
        else:
            context_instruction = (
                "No retrieved context is available for this question. "
                "Answer using your general knowledge in a concise and practical way. "
                "Do not claim that you cannot answer only because retrieval context is empty."
            )

        prompt = (
            f"{context_instruction}\n\n"
            f"Conversation history:\n{history_text or 'N/A'}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context[:8000]}\n"
        )
        response = self._agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )
        answer_text = str(response["messages"][-1].content)
        return answer_text, list(dict.fromkeys(sources))

    def _retrieve_docs(self, question: str):
        docs = []
        try:
            docs = self._retriever.invoke(question)
        except Exception:
            docs = []

        if docs:
            return docs

        if not self.settings.internet_fallback_enabled:
            return []

        try:
            return search_web_snippets(
                question=question,
                max_results=self.settings.web_search_results,
                timeout_seconds=self.settings.web_search_timeout_seconds,
            )
        except Exception:
            return []

    def status(self) -> dict[str, str]:
        ok, missing = self.settings.validate_required()
        return {
            "ready": "yes" if ok else "no",
            "missing": ", ".join(missing) if missing else "none",
            "model": self.settings.chat_model,
            "max_output_tokens": str(self.settings.max_output_tokens),
            "embedding": self.settings.embedding_model,
            "pinecone_index": self.settings.pinecone_index_name,
            "internet_fallback": "enabled" if self.settings.internet_fallback_enabled else "disabled",
            **{f"langsmith_{k}": v for k, v in langsmith_status(self.settings).items()},
        }

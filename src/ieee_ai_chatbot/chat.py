from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Settings, configure_langsmith, langsmith_status
from .retrieval import retrieve_context


class RAGAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configure_langsmith(settings)
        self._llm = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
        )

    def answer(self, question: str, history_text: str = "") -> tuple[str, list[str]]:
        docs = retrieve_context(self.settings, question)
        context_chunks = [doc.page_content for doc in docs]
        sources = [str(doc.metadata.get("filename", "unknown")) for doc in docs]
        context = "\n\n".join(context_chunks)

        prompt = (
            "You are an assistant for IEEE Beni Suef Student Branch. "
            "Use only the retrieved context to answer user questions. "
            "If context is insufficient, say that clearly and ask for more relevant documents.\n\n"
            f"Conversation history:\n{history_text or 'N/A'}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context[:12000]}\n"
        )
        response = self._llm.invoke(prompt)
        return str(response.content), list(dict.fromkeys(sources))

    def status(self) -> dict[str, str]:
        ok, missing = self.settings.validate_required()
        return {
            "ready": "yes" if ok else "no",
            "missing": ", ".join(missing) if missing else "none",
            "model": self.settings.chat_model,
            "embedding": self.settings.embedding_model,
            "pinecone_index": self.settings.pinecone_index_name,
            **{f"langsmith_{k}": v for k, v in langsmith_status(self.settings).items()},
        }

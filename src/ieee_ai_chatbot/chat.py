from __future__ import annotations

import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable, Client as LangSmithClient
from langsmith.run_helpers import get_current_run_tree

from .config import Settings, configure_langsmith, langsmith_status
from .prompts import build_prompt_config, build_system_prompt, build_user_prompt
from .retrieval import search_web_snippets
from .vectorstore import get_vector_store

logger = logging.getLogger(__name__)


class RAGAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._prompt_config = build_prompt_config(settings)
        self._system_prompt = build_system_prompt(settings)
        self._fallback_model = settings.chat_model_fallback.strip()
        configure_langsmith(settings)
        self.ls_client = LangSmithClient() if settings.langsmith_tracing else None
        self._vectorstore = get_vector_store(settings)
        self._llm = self._build_llm(settings.chat_model)
        self._fallback_llm: ChatGoogleGenerativeAI | None = None

    def _build_llm(self, model_name: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.settings.google_api_key,
            temperature=0.2,
            max_output_tokens=self.settings.max_output_tokens,
        )

    @staticmethod
    def _is_quota_error(error: Exception) -> bool:
        message = str(error).lower()
        return "resource_exhausted" in message or "429" in message or "quota" in message

    @staticmethod
    def _extract_retry_delay_seconds(error_text: str) -> int:
        patterns = [
            r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
            r"retrydelay':\s*'([0-9]+)s'",
        ]
        lowered = error_text.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                try:
                    return max(0, int(float(match.group(1))))
                except ValueError:
                    return 0
        return 0

    @traceable(run_type="llm", name="invoke_llm")
    def _invoke_with_retry_and_fallback(self, prompt: str) -> str:
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt),
        ]

        try:
            response = self._llm.invoke(messages)
            return str(response.content)
        except Exception as first_error:
            if not self._is_quota_error(first_error):
                raise

            logger.warning("Quota error on primary model: %s", first_error)

            retry_cap = max(0, self.settings.chat_quota_retry_seconds)
            suggested_wait = self._extract_retry_delay_seconds(str(first_error))
            wait_seconds = min(suggested_wait, retry_cap) if retry_cap else 0
            if wait_seconds > 0:
                logger.info("Retrying after %ds wait...", wait_seconds)
                time.sleep(wait_seconds)
                try:
                    response = self._llm.invoke(messages)
                    return str(response.content)
                except Exception as retry_error:
                    if not self._is_quota_error(retry_error):
                        raise
                    logger.warning("Retry also hit quota: %s", retry_error)
                    first_error = retry_error

            if self._fallback_model and self._fallback_model != self.settings.chat_model:
                logger.info("Switching to fallback model: %s", self._fallback_model)
                if self._fallback_llm is None:
                    self._fallback_llm = self._build_llm(self._fallback_model)
                try:
                    response = self._fallback_llm.invoke(messages)
                    return str(response.content)
                except Exception as fallback_error:
                    logger.exception("Fallback model also failed: %s", fallback_error)

            raise RuntimeError(
                "Model quota exceeded. Reduce request rate, switch to a lower-cost model, "
                "or increase quota. Consider setting CHAT_MODEL=gemini-2.5-flash-lite."
            ) from first_error

    @traceable(run_type="chain", name="rag_answer")
    def answer(self, question: str, history_text: str = "") -> tuple[str, list[str], str, str]:
        docs, confidence = self._retrieve_docs(question)
        context_chunks = [doc.page_content for doc in docs]
        sources = [str(doc.metadata.get("filename", "unknown")) for doc in docs]
        context = "\n\n".join(context_chunks)
        prompt = build_user_prompt(
            question=question,
            history_text=history_text,
            context=context,
            prompt_config=self._prompt_config,
        )
        answer_text = self._invoke_with_retry_and_fallback(prompt)
        
        run_tree = get_current_run_tree()
        run_id = str(run_tree.id) if run_tree else ""
        
        return answer_text, list(dict.fromkeys(sources)), run_id, confidence

    @traceable(run_type="chain", name="rag_answer_stream")
    def answer_stream(self, question: str, history_text: str = ""):
        docs, confidence = self._retrieve_docs(question)
        context_chunks = [doc.page_content for doc in docs]
        sources = [str(doc.metadata.get("filename", "unknown")) for doc in docs]
        sources = list(dict.fromkeys(sources))
        context = "\n\n".join(context_chunks)
        prompt = build_user_prompt(
            question=question,
            history_text=history_text,
            context=context,
            prompt_config=self._prompt_config,
        )
        
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=prompt),
        ]
        
        run_tree = get_current_run_tree()
        run_id = str(run_tree.id) if run_tree else ""

        try:
            for chunk in self._llm.stream(messages):
                yield str(chunk.content), sources, run_id, confidence
        except Exception as error:
            logger.warning("Streaming failed, falling back to synchronous invoke: %s", error)
            answer_text = self._invoke_with_retry_and_fallback(prompt)
            yield answer_text, sources, run_id, confidence

    def submit_feedback(self, run_id: str, score: float, comment: str = "") -> bool:
        if not self.ls_client or not run_id:
            return False
        try:
            self.ls_client.create_feedback(
                run_id,
                key="user_score",
                score=score,
                comment=comment
            )
            return True
        except Exception as e:
            logger.warning("Failed to submit feedback to LangSmith: %s", e)
            return False

    @traceable(run_type="retriever", name="retrieve_docs")
    def _retrieve_docs(self, question: str) -> tuple[list[Any], str]:
        docs = []
        confidence = "Low"
        try:
            # Using similarity_search_with_score instead of mmr to get confidence
            results = self._vectorstore.similarity_search_with_score(
                question, 
                k=self.settings.retriever_k
            )
            if results:
                docs = [doc for doc, score in results]
                scores = [score for doc, score in results]
                avg_score = sum(scores) / len(scores)
                # Cosine similarity scores: closer to 1 is better
                if avg_score > 0.85:
                    confidence = "High"
                elif avg_score > 0.70:
                    confidence = "Medium"
        except Exception:
            logger.exception("Retriever failed for question: %.100s", question)
            docs = []

        if docs:
            return docs, confidence

        if not self.settings.internet_fallback_enabled:
            return [], "None"

        try:
            docs = search_web_snippets(
                question=question,
                max_results=self.settings.web_search_results,
                timeout_seconds=self.settings.web_search_timeout_seconds,
            )
            return docs, "Web Search"
        except Exception:
            logger.exception("Web search fallback failed for question: %.100s", question)
            return [], "None"

    def status(self) -> dict[str, str]:
        ok, missing = self.settings.validate_required()
        return {
            "ready": "yes" if ok else "no",
            "missing": ", ".join(missing) if missing else "none",
            "model": self.settings.chat_model,
            "fallback_model": self.settings.chat_model_fallback or "none",
            "quota_retry_seconds": str(self.settings.chat_quota_retry_seconds),
            "max_output_tokens": str(self.settings.max_output_tokens),
            "embedding": self.settings.embedding_model,
            "pinecone_index": self.settings.pinecone_index_name,
            "internet_fallback": "enabled" if self.settings.internet_fallback_enabled else "disabled",
            **{f"langsmith_{k}": v for k, v in langsmith_status(self.settings).items()},
        }

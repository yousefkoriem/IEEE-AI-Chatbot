from __future__ import annotations

import logging
import re
import time
from difflib import SequenceMatcher
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable, Client as LangSmithClient
from langsmith.run_helpers import get_current_run_tree

from .config import Settings, configure_langsmith, langsmith_status
from .local_retrieval import LocalRetriever
from .prompts import build_prompt_config, build_system_prompt, build_user_prompt
from .retrieval import search_web_snippets
from .vectorstore import get_vector_store

logger = logging.getLogger(__name__)


class RAGAgent:
    def __init__(self, settings: Settings, get_chunk_boosts=None) -> None:
        self.settings = settings
        self._prompt_config = build_prompt_config(settings)
        self._system_prompt = build_system_prompt(settings)
        self._fallback_model = settings.chat_model_fallback.strip()
        configure_langsmith(settings)
        self.ls_client = LangSmithClient() if settings.langsmith_tracing else None
        self._vectorstore = get_vector_store(settings)
        self._local_retriever = LocalRetriever(settings)
        self._llm = self._build_llm(settings.chat_model)
        self._fallback_llm: ChatGoogleGenerativeAI | None = None
        self._get_chunk_boosts = get_chunk_boosts or (lambda: {})

    def _build_llm(self, model_name: str, temperature: float | None = None, max_tokens: int | None = None) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.settings.google_api_key,
            temperature=temperature if temperature is not None else self.settings.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.settings.max_output_tokens,
        )

    def set_temperature(self, value: float) -> None:
        self._llm = self._build_llm(self.settings.chat_model, temperature=value, max_tokens=self.settings.max_output_tokens)

    def set_max_tokens(self, value: int) -> None:
        self.settings.max_output_tokens = value
        self._llm = self._build_llm(self.settings.chat_model, max_tokens=value)

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

    @staticmethod
    def _build_labeled_context(docs: list) -> str:
        """Build context string with source labels so the LLM knows provenance."""
        chunks: list[str] = []
        for doc in docs:
            filename = doc.metadata.get("filename", "unknown")
            chunks.append(f"[Source: {filename}]\n{doc.page_content}")
        return "\n\n---\n\n".join(chunks)

    def _annotate_citations(self, answer_text: str, docs: list) -> tuple[str, list[dict]]:
        """Append numbered source references only for docs that influenced the answer.
        Returns (answer_with_footnotes, list of unique source dicts)."""
        answer_lower = answer_text.lower()
        answer_words = set(re.findall(r'\b\w{4,}\b', answer_lower))
        source_map: list[dict] = []
        for doc in docs:
            # Only cite if meaningful content overlap exists with the answer
            content_words = set(re.findall(r'\b\w{4,}\b', doc.page_content.lower()))
            overlap = len(content_words & answer_words)
            if overlap < 3:
                continue
            source_id = doc.metadata.get("id", str(id(doc)))[:12]
            filename = doc.metadata.get("filename", "unknown")
            existing = {s["id"] for s in source_map}
            if source_id not in existing:
                source_map.append({"id": source_id, "filename": filename})

        if not source_map:
            return answer_text, source_map

        refs = "\n\n📚 **References**\n"
        for i, s in enumerate(source_map, 1):
            refs += f"\n[{i}] `{s['id']}` — {s['filename']}"

        return answer_text + refs, source_map

    @traceable(run_type="chain", name="rag_answer")
    def answer(self, question: str, history_text: str = "", generate_suggestions: bool = False) -> tuple[str, list[str], str, str, list[str], list[str]]:
        docs, confidence = self._retrieve_docs(question)
        sources = [str(doc.metadata.get("filename", "unknown")) for doc in docs]
        chunk_ids_used = [str(doc.metadata.get("chunk_id", "")) for doc in docs if doc.metadata.get("chunk_id")]
        context = self._build_labeled_context(docs)
        prompt = build_user_prompt(
            question=question,
            history_text=history_text,
            context=context,
            prompt_config=self._prompt_config,
        )
        answer_text = self._invoke_with_retry_and_fallback(prompt)

        html_answer, _ = self._annotate_citations(answer_text, docs)
        
        run_tree = get_current_run_tree()
        run_id = str(run_tree.id) if run_tree else ""

        suggestions = []
        if generate_suggestions and answer_text.strip():
            suggestions = self._generate_followup_suggestions(question, answer_text, context)
        
        return html_answer, list(dict.fromkeys(sources)), run_id, confidence, suggestions, chunk_ids_used

    def _generate_followup_suggestions(self, question: str, answer: str, context: str) -> list[str]:
        try:
            suggestion_prompt = (
                f"Based on this Q&A, suggest exactly 2 short follow-up questions "
                f"the user might ask next. Return only the questions, one per line, "
                f"without numbering or bullets.\n\n"
                f"User: {question}\n"
                f"Assistant: {answer}\n\n"
                f"Context: {context[:800]}"
            )
            messages = [
                SystemMessage(content="You are a helpful assistant that suggests follow-up questions. Keep each suggestion under 60 characters."),
                HumanMessage(content=suggestion_prompt),
            ]
            response = self._llm.invoke(messages)
            suggestions = [s.strip() for s in str(response.content).strip().split("\n") if s.strip()]
            return suggestions[:3]
        except Exception as e:
            logger.warning("Failed to generate follow-up suggestions: %s", e)
            return []

    @traceable(run_type="chain", name="rag_answer_stream")
    def answer_stream(self, question: str, history_text: str = ""):
        docs, confidence = self._retrieve_docs(question)
        sources = [str(doc.metadata.get("filename", "unknown")) for doc in docs]
        sources = list(dict.fromkeys(sources))
        chunk_ids_used = [str(doc.metadata.get("chunk_id", "")) for doc in docs if doc.metadata.get("chunk_id")]
        context = self._build_labeled_context(docs)
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

        full_text = ""
        try:
            for chunk in self._llm.stream(messages):
                full_text += str(chunk.content)
                yield str(chunk.content), sources, run_id, confidence, [], None, chunk_ids_used
        except Exception as error:
            logger.warning("Streaming failed, falling back to synchronous invoke: %s", error)
            answer_text = self._invoke_with_retry_and_fallback(prompt)
            full_text = answer_text
            yield answer_text, sources, run_id, confidence, [], None, chunk_ids_used

        suggestions = self._generate_followup_suggestions(question, full_text, context)
        html_answer, _ = self._annotate_citations(full_text, docs)
        if suggestions:
            yield "", sources, run_id, confidence, suggestions, html_answer, chunk_ids_used
        else:
            yield "", sources, run_id, confidence, [], html_answer, chunk_ids_used

    async def answer_stream_async(self, question: str, history_text: str = ""):
        docs, confidence = self._retrieve_docs(question)
        sources = [str(doc.metadata.get("filename", "unknown")) for doc in docs]
        sources = list(dict.fromkeys(sources))
        chunk_ids_used = [str(doc.metadata.get("chunk_id", "")) for doc in docs if doc.metadata.get("chunk_id")]
        context = self._build_labeled_context(docs)
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

        full_text = ""
        try:
            async for chunk in self._llm.astream(messages):
                full_text += str(chunk.content)
                yield str(chunk.content), sources, run_id, confidence, [], None, chunk_ids_used
        except Exception as error:
            logger.warning("Async streaming failed, falling back to sync: %s", error)
            for val in self.answer_stream(question, history_text=history_text):
                yield val
            return

        suggestions = self._generate_followup_suggestions(question, full_text, context)
        html_answer, _ = self._annotate_citations(full_text, docs)
        if suggestions:
            yield "", sources, run_id, confidence, suggestions, html_answer, chunk_ids_used
        else:
            yield "", sources, run_id, confidence, [], html_answer, chunk_ids_used

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
        # Pinecone is the primary knowledge base. A local keyword match must not
        # prevent a more relevant semantic match from being used.
        try:
            pinecone_results = self._vectorstore.similarity_search_with_score(
                question,
                k=max(self.settings.retriever_k, self.settings.retriever_fetch_k),
            )
            unique_results: list[tuple[Any, float]] = []
            seen_content: set[str] = set()
            for doc, score in pinecone_results:
                # Identical chunks can have distinct IDs when a source is ingested
                # more than once, so normalize the actual content for deduplication.
                content_key = re.sub(r"\s+", " ", doc.page_content).strip().casefold()
                if not content_key or content_key in seen_content:
                    continue
                seen_content.add(content_key)
                unique_results.append((doc, score))
                if len(unique_results) >= self.settings.retriever_k:
                    break

            # Filter out chunks below minimum similarity threshold
            pre_filter_count = len(unique_results)
            if unique_results:
                scores_before = [s for _, s in unique_results]
                logger.debug(
                    "Pinecone returned %d unique chunks for '%.60s', scores: %s (min_score=%.2f)",
                    pre_filter_count, question, scores_before, self.settings.retriever_min_score,
                )
            unique_results = [
                (doc, score) for doc, score in unique_results
                if score >= self.settings.retriever_min_score
            ]

            if unique_results:
                if self.settings.feedback_boost_enabled:
                    boosts = self._get_chunk_boosts()
                    if boosts:
                        factor = self.settings.feedback_boost_factor
                        unique_results = [
                            (
                                doc,
                                score * (1 + factor * max(boosts.get(doc.metadata.get("chunk_id", ""), 0.0), 0)),
                            )
                            for doc, score in unique_results
                        ]
                        unique_results.sort(key=lambda item: item[1], reverse=True)

                docs = [doc for doc, _ in unique_results]
                best_score = max(score for _, score in unique_results)
                # Cosine similarities for the current embedding model commonly fall
                # in the 0.60--0.74 range for correct answers. Use the strongest
                # selected match, rather than averaging unrelated supporting chunks.
                if best_score >= 0.65:
                    return docs, "High"
                if best_score >= 0.55:
                    return docs, "Medium"
                return docs, "Low"
        except Exception:
            logger.exception("Pinecone retriever failed for question: %.100s", question)

        if self.settings.local_retrieval_enabled:
            try:
                local_docs = self._local_retriever.search(
                    question,
                    max_results=self.settings.local_retrieval_max_results,
                    min_score=self.settings.local_retrieval_min_score,
                )
                if local_docs:
                    return local_docs, "Local"
            except Exception:
                logger.exception("Local retrieval failed for question: %.100s", question)

        # Web search fallback (disabled by default to prevent external data
        # from overriding correct Pinecone results)
        if not self.settings.internet_fallback_enabled:
            return [], ""

        try:
            docs = search_web_snippets(
                question=question,
                max_results=self.settings.web_search_results,
                timeout_seconds=self.settings.web_search_timeout_seconds,
                settings=self.settings,
            )
            if docs:
                return docs, "Web Search"
            return [], "None"
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
            "local_retrieval": "enabled" if self.settings.local_retrieval_enabled else "disabled",
            "internet_fallback": "enabled" if self.settings.internet_fallback_enabled else "disabled",
            "web_search_provider": self.settings.web_search_provider,
            **{f"langsmith_{k}": v for k, v in langsmith_status(self.settings).items()},
        }

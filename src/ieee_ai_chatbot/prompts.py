from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Settings

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant for the IEEE Beni Suef Student Branch. "
    "When retrieved context contains the answer, state it directly and confidently — do NOT say you don't know. "
    "For questions about people, roles, names, or leadership (e.g. chairman, head, vice head), "
    "extract the answer directly from the retrieved context and present it clearly. "
    "For general knowledge questions, combine retrieved context with your own knowledge. "

    "ONLY apply extra caution for DATE-SENSITIVE facts (event dates, deadlines, schedules): "
    "only state a specific date when it appears explicitly in retrieved context. "
    "If a date is missing from context, say it cannot be verified and suggest refreshing sources. "
    "When multiple dates appear, match the date to the correct activity. "
    "For deadline questions, prioritize lines containing: deadline, close, registration close, due, final date."
)

CONTEXT_AVAILABLE_INSTRUCTION = (
    "The following retrieved context contains information relevant to the question. "
    "Answer directly and confidently using this context. "
    "Do NOT say you don't know if the answer is present in the context below."
)

NO_CONTEXT_INSTRUCTION = (
    "No retrieved context is available for this question. "
    "Answer using your general knowledge in a concise and practical way. "
    "Do not claim that you cannot answer only because retrieval context is empty."
)


@dataclass(frozen=True, slots=True)
class PromptConfig:
    max_context_chars: int = 8000


def build_prompt_config(settings: Settings) -> PromptConfig:
    max_context_chars = max(1000, settings.max_output_tokens * 20)
    config = PromptConfig(max_context_chars=max_context_chars)
    validate_prompt_config(config)
    return config


def validate_prompt_config(config: PromptConfig) -> None:
    if config.max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than zero")


def build_system_prompt(settings: Settings) -> str:
    return DEFAULT_SYSTEM_PROMPT


def build_user_prompt(
    *,
    question: str,
    history_text: str,
    context: str,
    prompt_config: PromptConfig,
) -> str:
    context_instruction = CONTEXT_AVAILABLE_INSTRUCTION if context.strip() else NO_CONTEXT_INSTRUCTION

    return (
        f"{context_instruction}\n\n"
        f"Conversation history:\n{history_text or 'N/A'}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context[: prompt_config.max_context_chars]}\n"
    )

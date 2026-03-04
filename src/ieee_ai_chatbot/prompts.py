from __future__ import annotations

from dataclasses import dataclass

from .config import Settings


DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant for IEEE Beni Suef Student Branch. "
    "Prefer factual, direct answers and avoid unnecessary disclaimers. "
    "For date-sensitive facts (deadlines, event dates, schedules), only state a specific date when it appears in retrieved context. "
    "If retrieved context is missing or ambiguous, say that the date cannot be verified from current indexed data and ask the user to refresh sources."
)

CONTEXT_AVAILABLE_INSTRUCTION = (
    "Use retrieved context first when it is relevant and sufficient. "
    "If the question also needs general knowledge, combine both clearly and concisely."
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
    if not settings.chat_model.strip():
        raise ValueError("chat_model cannot be empty when building system prompt")
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

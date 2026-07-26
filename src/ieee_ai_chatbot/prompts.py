from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Settings

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are the official AI assistant for the IEEE Beni Suef Student Branch. "
    "RULE 1: For ANY question about the branch — people, roles, names, leadership, "
    "committees, events, the chatbot itself, or branch operations — ONLY use "
    "information from the retrieved context below. NEVER guess or use your own knowledge "
    "for branch-specific facts. If the context does not contain the answer, say: "
    "'I don't have that information in my current knowledge base. Please check the official "
    "IEEE Beni Suef channels or contact the branch directly.' "
    "RULE 2: For general technical or academic questions completely unrelated to the branch "
    "(e.g., 'What is machine learning?'), you may use your general knowledge. "
    "RULE 3: When listing people or roles, reproduce the EXACT names from the context. "
    "Do not add, remove, or modify names. "
    "RULE 4: For date-sensitive facts (deadlines, schedules), only state a date "
    "if it appears explicitly in the retrieved context. "
    "RULE 5: Branch terminology — 'Board' or 'board of the branch' means ALL members: "
    "the Leadership Board (Chair, Vice Chairs, Secretary, Treasurer, Web Master) PLUS "
    "all Technical and Operational Committee Heads and Vice Heads. "
    "'High Board' means ONLY the Leadership Board (the 6 senior positions). "
    "When asked about the board, list everyone from ALL sections in the context."
)

CONTEXT_AVAILABLE_INSTRUCTION = (
    "The following retrieved context contains information relevant to the question. "
    "Use ONLY this context to answer questions about the IEEE Beni Suef Student Branch. "
    "Do not supplement with your own knowledge for branch-specific facts."
)

NO_CONTEXT_INSTRUCTION = (
    "No retrieved context is available for this question. "
    "If this question is about the IEEE Beni Suef Student Branch (people, roles, "
    "committees, events, the chatbot, or branch operations), respond: "
    "'I don't have that information in my current knowledge base. "
    "Please check the official IEEE Beni Suef channels or contact the branch directly.' "
    "For general knowledge questions unrelated to the branch, answer concisely."
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

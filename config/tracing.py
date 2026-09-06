"""LangSmith tracing configuration."""

import os
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def configure_tracing() -> None:
    """Set up LangSmith tracing environment variables.

    Called once at application startup.  When ``settings.langsmith_tracing``
    is *False* the function is a no-op.
    """
    if not settings.langsmith_tracing:
        logger.info("LangSmith tracing is disabled")
        return

    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty — tracing disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = "ieee-bsu-chatbot"

    # Sample 25% of traces to stay within free-tier limits
    os.environ["LANGSMITH_SAMPLE_RATE"] = os.getenv("LANGSMITH_SAMPLE_RATE", "0.25")

    logger.info(
        "LangSmith tracing enabled (project=%s, sample_rate=%s)",
        os.environ["LANGCHAIN_PROJECT"],
        os.environ["LANGSMITH_SAMPLE_RATE"],
    )


def get_tracing_status() -> dict[str, str]:
    """Return current tracing status for the UI status panel."""
    if not settings.langsmith_tracing:
        return {"enabled": False, "project": "", "sample_rate": ""}

    return {
        "enabled": True,
        "project": os.getenv("LANGCHAIN_PROJECT", "ieee-bsu-chatbot"),
        "sample_rate": os.getenv("LANGSMITH_SAMPLE_RATE", "0.25"),
    }
